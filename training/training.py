#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the training and evaluation funtionality.

@author: Matthijs de Jong
"""

# Standard library imports
import collections
from typing import Optional

# Third-party library imports
import torch
import wandb
from torch import Tensor
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Project imports
from training.models import Model, GCN, FCNN
from training.dataloader import DataLoader
import training.evaluation as metrics
import auxiliary.util as util
from auxiliary.config import ModelType, LayerType, LabelWeightsType
import auxiliary.grid2op_util as g2o_util
from training.postprocessing import ActSpaceCache
from auxiliary.config import get_config


def BCELoss_labels_weighted(P: torch.Tensor, Y: torch.Tensor, W: torch.Tensor) \
        -> torch.Tensor:
    """
    Binary cross entropy loss which allows for different weights for different labels.

    Parameters
    ----------
    P : torch.Tensor
        The predicted labels.
    Y : torch.Tensor
        The true labels.
    W : torch.Tensor
        The weights per label.

    Returns
    -------
    loss : torch.Tensor
        Tensor object of size (1,1) containing the loss value.
    """
    P = torch.clamp(P, min=1e-7, max=1 - 1e-7)
    bce = W * (- Y * torch.log(P) - (1 - Y) * torch.log(1 - P))
    loss = torch.mean(bce)
    return loss


class Run:
    """
    Class that specifies the running of a model.
    """

    def __init__(self):
        # Save some configurations
        config = get_config()
        self.train_config = train_config = config['training']
        processed_data_path = config['paths']['data']['processed']
        matrix_cache_path = processed_data_path + 'auxiliary_data_objects/con_matrix_cache.json'
        feature_statistics_path = processed_data_path + 'auxiliary_data_objects/feature_stats.json'

        # Specify device to use
        self.device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Init model
        if train_config['hyperparams']['model_type'] == ModelType.GCN:
            self.model = GCN(train_config['hyperparams']['LReLu_neg_slope'],
                             train_config['hyperparams']['weight_init_std'],
                             train_config['GCN']['constants']['N_f_gen'],
                             train_config['GCN']['constants']['N_f_load'],
                             train_config['GCN']['constants']['N_f_endpoint'],
                             train_config['GCN']['hyperparams']['N_GCN_layers'],
                             train_config['hyperparams']['N_node_hidden'],
                             train_config['GCN']['hyperparams']['aggr'],
                             train_config['GCN']['hyperparams']['network_type'],
                             train_config['GCN']['hyperparams']['layer_type'],
                             train_config['GCN']['hyperparams']['GINConv_nn_depth'])
        elif train_config['hyperparams']['model_type'] == ModelType.FCNN:
            self.model = FCNN(train_config['hyperparams']['LReLu_neg_slope'],
                              train_config['hyperparams']['weight_init_std'],
                              train_config['FCNN']['constants']['size_in'],
                              train_config['FCNN']['constants']['size_out'],
                              train_config['FCNN']['hyperparams']['N_layers'],
                              train_config['hyperparams']['N_node_hidden'])
        else:
            raise ValueError("Invalid model_type value.")
        self.model.to(self.device)

        # Init optimizer
        w_decay = train_config['hyperparams']['weight_decay']
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=train_config['hyperparams']['lr'],
                                          weight_decay=w_decay)

        # Initialize dataloaders
        network_type = train_config['GCN']['hyperparams']['network_type']
        model_type = train_config['hyperparams']['model_type']
        self.train_dl = DataLoader(processed_data_path + '/train',
                                   matrix_cache_path,
                                   feature_statistics_path,
                                   device=self.device,
                                   model_type=model_type,
                                   network_type=network_type)
        self.val_dl = DataLoader(processed_data_path + '/val',
                                 matrix_cache_path,
                                 feature_statistics_path,
                                 device=self.device,
                                 model_type=model_type,
                                 network_type=network_type)

        # Initialize metrics objects
        IA = metrics.IncrementalAverage
        metrics_dict = {
            'macro_accuracy': (metrics.macro_accuracy, IA()),
            'micro_accuracy': (metrics.micro_accuracy, IA()),
            'n_predicted_changes': (metrics.n_predicted_changes, IA()),
            'any_predicted_changes': (metrics.any_predicted_changes, IA()),
            'macro_accuracy_one_sub': (metrics.macro_accuracy_one_sub, IA()),
            'micro_accuracy_one_sub': (metrics.micro_accuracy_one_sub, IA()),
            'train_loss': (lambda **kwargs: kwargs['l'], IA()),
            'accuracy_predicted_substation':
                (metrics.accuracy_predicted_substation, IA())
        }
        train_metrics_dict = dict([('train_' + k, v) for k, v in metrics_dict.items()])
        val_metrics_dict = dict([('val_' + k, v) for k, v in metrics_dict.items()])
        val_metrics_dict['val_macro_accuracy_valid'] = (metrics.macro_accuracy_valid, IA())
        val_metrics_dict['val_micro_accuracy_valid'] = (metrics.micro_accuracy_valid, IA())
        IAM = metrics.IncrementalAverageMetrics
        self.train_metrics = IAM(train_metrics_dict)
        self.val_metrics = IAM(val_metrics_dict)

        # Initialize action space cache used for
        self.as_cache = ActSpaceCache(line_outages_considered=train_config['settings']['line_outages_considered'])

        # Start wandb run
        self.run = wandb.init(project=train_config['wandb']["project"],
                              entity=train_config['wandb']["entity"],
                              name=train_config['wandb']['model_name'],
                              group=train_config['wandb']['group'],
                              tags=train_config['wandb']['model_tags'],
                              mode=train_config['wandb']['mode'],
                              dir=config['paths']['wandb'],
                              config=train_config)
        self.run.watch(self.model,
                       log_freq=train_config['settings']['train_log_freq'],
                       log='all',
                       log_graph=True)

    def train_on_datapoint(self, dp: dict, step: int):
        """
        Train on a datapoint datapoint. Obtains the loss, and computes the gradients. When the batch is filled,
        updates the model and resets the gradients.

        Parameters
        ----------
        dp : dict
            The datapoint.
        step : int
            The current step in the run.
        """
        lss = process_datapoint(dp, self.model, self.as_cache, self.train_metrics, self.device)[0]
        lss.backward()

        # If the batch is filled, update the model, reset gradients
        batch_size = self.train_config['hyperparams']['batch_size']
        if (not step % batch_size) and (step != 0):
            self.optimizer.step()
            self.model.zero_grad()

    def start(self):
        """
        Start the training run. Includes periodic evaluation on the validation
        set.
        """
        with self.run as run:

            # Initialize progress bar
            n_epoch = self.train_config['hyperparams']['n_epoch']
            est_tsize = self.train_config['constants']['estimated_train_size']
            pbar = tqdm(total=n_epoch * est_tsize)

            # Early stopping parameter
            stop_countdown = self.train_config['hyperparams']['early_stopping_patience']
            best_score = 0

            self.model.train()
            self.model.zero_grad()
            step = 0

            for e in range(n_epoch):
                for dp in self.train_dl:
                    # Process a single train datapoint
                    self.train_on_datapoint(dp, step)

                    # Periodically log train metrics
                    train_log_freq = self.train_config['settings']['train_log_freq']
                    if (not step % train_log_freq) and (step != 0):
                        self.train_metrics.log_to_wandb(run, step)
                        self.train_metrics.reset()

                    # Periodically evaluate the validation set
                    val_log_freq = self.train_config['settings']['val_log_freq']
                    if (not step % val_log_freq) and (step != 0):
                        self.model.eval()
                        _ = evaluate_dataset(self.model, self.val_dl, self.as_cache, self.val_metrics, self.run, step)

                        self.model.train()

                        # Check early stopping
                        val_macro_accuracy_valid = self.val_metrics.metrics_dict['val_macro_accuracy_valid'][1].get()
                        if val_macro_accuracy_valid > best_score:
                            best_score = val_macro_accuracy_valid
                            stop_countdown = self.train_config['hyperparams']['early_stopping_patience']
                            torch.save(self.model.state_dict(), "models/" + run.name)
                            run.log({'best_val_macro_accuracy_valid': best_score}, step=step)
                        else:
                            stop_countdown -= 1
                        if stop_countdown < 1:
                            quit()

                        self.val_metrics.reset()

                    step += 1
                    pbar.update(1)
            pbar.close()


def get_label_weights(Y_sub_mask: torch.Tensor, P_sub_mask: torch.Tensor) -> torch.Tensor:
    """
    Give the masked labels a specific weight value, and the other weights value 1.

    Parameters
    ----------
    Y_sub_mask : torch.Tensor
        The mask indicating the true selected substation.
    P_sub_mask : torch.Tensor
        The mask indicating the predicted selected substation.

    Returns
    -------
    weights : torch.Tensor[float]
        The resulting label weights, consisting of the values '1' and 'w_when_zero'.
    """
    train_config = get_config()['training']

    if train_config['hyperparams']['label_weights']['type'] == LabelWeightsType.ALL:
        mask = torch.ones_like(Y_sub_mask)
    elif train_config['hyperparams']['label_weights']['type'] == LabelWeightsType.Y:
        mask = Y_sub_mask
    elif train_config['hyperparams']['label_weights']['type'] == LabelWeightsType.P:
        mask = P_sub_mask
    elif train_config['hyperparams']['label_weights']['type'] == LabelWeightsType.Y_AND_P:
        mask = torch.logical_or(Y_sub_mask, P_sub_mask)
    else:
        raise ValueError('Invalid value for label weights type.')

    label_weights = torch.ones_like(mask)
    label_weights[~mask.detach().bool()] = train_config['hyperparams']['label_weights']['non_masked_weight']
    return label_weights


def predict_datapoint(dp: dict, model: Model) -> torch.Tensor:
    """
    Extract the necessary information from a datapoint, and use it to
    make a prediction from the model.

    Parameters
    ----------
    dp : dict
        The datapoint.
    model : Model
        The ML model to predict with.

    Returns
    -------
    P : torch.Tensor[float]
        The prediction of the model. Should have a length corresponding to
        the number of objects in the environment. All elements should be in
        range (0,1).
    """
    if type(model) is GCN:
        # Extract features
        X_gen = dp['gen_features']
        X_load = dp['load_features']
        X_or = dp['reduced_or_features']
        X_ex = dp['reduced_ex_features']

        # Extract the position topology vector, which relates the
        # objects ordered by type to their position in the topology vector
        pos_topo_vect = dp['reduced_pos_topo_vect']

        # Extract the edges
        E = dp['edges']

        # Pass through the model
        P = model(X_gen, X_load, X_or, X_ex, E, pos_topo_vect).reshape((-1))

        # Unreduce P to full length
        if dp['line_disabled'] != -1:
            min_disabled_endpoint_pos = min(dp['disabled_or_pos'], dp['disabled_ex_pos'])
            max_disabled_endpoint_pos = max(dp['disabled_or_pos'], dp['disabled_ex_pos'])
            P = torch.cat((P[:min_disabled_endpoint_pos], torch.tensor([0]),
                           P[min_disabled_endpoint_pos:max_disabled_endpoint_pos-1], torch.tensor([0]),
                           P[max_disabled_endpoint_pos-1:]))
            assert torch.count_nonzero(P[(min_disabled_endpoint_pos, max_disabled_endpoint_pos),]) == 0, \
                ("Disabled line endpoints must be zero in the topo_vect after unreducing.")

    elif type(model) is FCNN:
        # Pass through the model
        P = model(dp['features']).reshape((-1))
    else:
        raise ValueError('Invalid model type.')

    return P


def process_datapoint(dp: dict,
                      model: Model,
                      as_cache: ActSpaceCache,
                      running_metrics: metrics.IncrementalAverageMetrics,
                      device: torch.device = 'cpu') \
        -> tuple[Tensor, Tensor, Tensor, Tensor, int | None, Tensor, int | None, Tensor]:
    """
    Process a single validation datapoint. This involves:
        (1) Making a model prediction
        (2) Extracting the label and smoothing it
        (3) Computing the weighted loss
        (4) Updating the validation metrics
        (5) Returning statistics for further analysis

    Parameters
    ----------
    dp : dict
        The datapoint.
    model : Model
        The ML model to make the prediction with.
    as_cache : ActSpaceCache
        The action space cache used to find valid action.

    Returns
    -------
    y : torch.Tensor[int]
        The label. Should have the length equal to the number of objects in
        the network. Elements should be in {0,1}.
    P : torch.Tensor[int]
        The prediction of the model.
    nearest_valid_P : torch.Tensor[int]
        The valid action nearest to the prediction. Should have the length
        equal to the number of objects in the network. Elements should be
        in {0,1}.
    Y_sub_idx : int
        The index of substation changed in the label.
    Y_sub_mask : torch.Tensor[bool]
        The mask indicating which objects in the topology vector
        correspond to the true changed substation. Should have the length
        equal to the number of objects in the network. Elements should be
        in {0,1}.
    P_subchanged_idx : int
        The index of substation changed in the predicted action. Computed
        based on nearest_valid_P.
    nearest_valid_actions : torch.Tensor[int]
        Matrix indicating the order of the valid actions in order of
        nearness to the prediction actions. Rows indicate actions.
        Should have dimensions (n_actions, n_objects).
    """
    config = get_config()
    train_config = config['training']

    # Make model prediction
    P = predict_datapoint(dp, model)
    assert len(P) == config['rte_case14_realistic']['n_objects'], "P has the wrong length; should be full, not reduced."

    # Extract the label, apply label smoothing
    Y = dp['full_change_topo_vect']
    label_smth_alpha = train_config['hyperparams']['label_smoothing_alpha']
    Y_smth = ((1 - label_smth_alpha) * dp['full_change_topo_vect'] +
              label_smth_alpha * 0.5 * torch.ones_like(Y, device=device))

    # Compute the weights for the loss
    sub_info = config['rte_case14_realistic']['sub_info']
    Y_sub_mask, Y_sub_idx = g2o_util.select_single_substation_from_topovect(Y, sub_info)
    P_sub_mask, P_sub_idx = g2o_util.select_single_substation_from_topovect(P, sub_info,
                                                                            f=lambda x:
                                                                            torch.sum(torch.clamp(x - 0.5, min=0)))
    weights = get_label_weights(Y_sub_mask, P_sub_mask)

    # Compute the loss
    loss = BCELoss_labels_weighted(P, Y_smth, weights)

    # Calculate statistics for metrics
    one_sub_P = torch.zeros_like(P)
    one_sub_P[torch.logical_and(P_sub_mask, P > 0.5)] = 1
    get_cabnp = as_cache.get_change_actspace_by_nearness_pred
    nearest_valid_actions = get_cabnp(dp['line_disabled'], dp['full_topo_vect'], P, device)
    nearest_valid_P = nearest_valid_actions[0]

    # Update metrics, if any
    running_metrics.log(P=P, Y=Y, one_sub_P=one_sub_P, l=loss, P_subchanged_idx=P_sub_idx, Y_subchanged_idx=Y_sub_idx,
                        nearest_valid_P=nearest_valid_P)

    # Return statistics used in further analysis
    return loss, Y, P, nearest_valid_P, Y_sub_idx, Y_sub_mask, P_sub_idx, nearest_valid_actions


def evaluate_dataset(model: Model,
                     dataloader: DataLoader,
                     as_cache: ActSpaceCache,
                     running_metrics: metrics.IncrementalAverageMetrics,
                     wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
                     step: Optional[int] = None):
    """
    Evaluate the validation set. Consists of:
        (1) updating validation metrics,
        (2) creating a confusion matrix for the substations,
        (3) creating a histogram of the ranks of the true actions in the
        list of valid actions sorted by nearness to the predicted actions,
        (4) creating histograms of the difference between self weights and
        other weights,
        (5) creating a stacked histogram of the labels and the (in)correct
        classifications of those.
    All of these are logged to a wandb run.

    Parameters
    ----------
    step : int
        The current step.
    run : wandb.sdk.wandb_run.Run
        The wandb run to log the analysis results to.
    """
    config = get_config()
    train_config = config['training']

    # Initializing lists for tracking the predicted/true substations
    Y_subs = []
    P_subs = []

    # Initializing distributions for tracking the true/predicted/postprocessed predicted objects
    Y_obs = P_obs = nearest_valid_P_obs = None

    # Initializing lists for tracking the ranks of the true actions in the list of valid actions sorted by
    # nearness to the predicted actions
    Y_rank_in_nearest_v_acts = []

    # Initializing counters for counting the number of (in)correct classifications of each label
    correct_counter_label = collections.Counter()
    wrong_counter_label = collections.Counter()

    # Initializing counters for counting the number of (in)correct classifications of each topology vector
    correct_counter_topovect = collections.Counter()
    wrong_counter_topovect = collections.Counter()

    # Initializing counters for counting the number of (in)correct classifications of each topology vector
    correct_counter_lineout = collections.Counter()
    wrong_counter_lineout = collections.Counter()

    # Init object for tracking MAD scores
    MAD_scores = None

    # Init config
    config = get_config()

    with torch.no_grad():
        i = 0
        for dp in dataloader:
            l, Y, P, nearest_valid_P, Y_sub_idx, Y_sub_mask, P_subchanged_idx, \
                nearest_valid_actions = process_datapoint(dp, model, as_cache, running_metrics)
            sub_Y = Y[Y_sub_mask.bool()] if Y_sub_idx is not None else torch.tensor([])
            topo_vect = dp['full_topo_vect']

            if not train_config['settings']['advanced_val_analysis']:
                continue

            # TODO: remove?
            '''
            # Create 'expanded' version of vectors - expanded vertors have the size of the topovect without. This
            # line disabled. This is necessary for comparisons of topologies with different lines disabled.
            if dp['line_disabled'] == -1:
                expanded_Y = Y
                expanded_P = P
                expanded_valid_P = nearest_valid_P
                expanded_topo_vect = dp['topo_vect']
                expanded_sub_Y = Y[Y_sub_mask.bool()]
            else:
                or_index = config['rte_case14_realistic']['line_or_pos_topo_vect'][dp['line_disabled']]
                ex_index = config['rte_case14_realistic']['line_ex_pos_topo_vect'][dp['line_disabled']]
                smll_idx = min(or_index, ex_index)
                big_idx = max(or_index, ex_index) - 1
                ins = torch.tensor(-1).view(1)

                expanded_Y = torch.cat([Y[:smll_idx], ins, Y[smll_idx:big_idx], ins, Y[big_idx:]], 0)
                expanded_P = torch.cat([P[:smll_idx], ins, P[smll_idx:big_idx], ins, P[big_idx:]], 0)
                expanded_valid_P = torch.cat([nearest_valid_P[:smll_idx], ins, nearest_valid_P[smll_idx:big_idx],
                                              ins, nearest_valid_P[big_idx:]], 0)
                expanded_topo_vect = torch.cat([dp['topo_vect'][:smll_idx], ins, dp['topo_vect'][smll_idx:big_idx],
                                                ins, dp['topo_vect'][big_idx:]], 0)
                
                if Y_sub_idx is not None:
                    expanded_sub_Y = g2o_util.tv_groupby_subst(expanded_Y,
                                                               config['rte_case14_realistic']['sub_info'])[
                        Y_sub_idx]
                else:
                    expanded_sub_Y = torch.tensor([])
                '''

            # Increment the counters for counting the number of (in)correct classifications of each label
            # and topology vector
            if torch.equal(nearest_valid_P, torch.round(Y)):
                correct_counter_label[(Y_sub_idx, tuple(sub_Y.tolist()))] += 1
                correct_counter_topovect[(Y_sub_idx, tuple(topo_vect.tolist()))] += 1
                correct_counter_lineout[dp['line_disabled']] += 1
            else:
                wrong_counter_label[(Y_sub_idx, tuple(sub_Y.tolist()))] += 1
                wrong_counter_topovect[(Y_sub_idx, tuple(topo_vect.tolist()))] += 1
                wrong_counter_lineout[dp['line_disabled']] += 1

            # Update lists for tracking the predicted/true substations
            Y_subs.append(Y_sub_idx)
            P_subs.append(P_subchanged_idx)

            # Update distributions for the true/predicted/postprocessed-predicted objects
            if Y_obs is None:
                Y_obs = Y.clip(0)
                P_obs = 1 * (P > 0.5)
                nearest_valid_P_obs = nearest_valid_P.clip(0)
            else:
                Y_obs += Y.clip(0)
                P_obs += P > 0.5
                nearest_valid_P_obs += nearest_valid_P.clip(0)

            # Update lists for tracking the ranks of the true actions in
            # the list of valid actions sorted by nearness to the predicted
            # actions
            Y_index_in_valid = torch.where((nearest_valid_actions == Y).all(dim=1))[0].item()
            Y_rank_in_nearest_v_acts.append(Y_index_in_valid)

            # For the first 100 datapoints, compute the mads
            i += 1
            if i <= 100 and train_config['hyperparams']['model_type'] == ModelType.GCN:
                # Extract features
                X_gen = dp['gen_features']
                X_load = dp['load_features']
                X_or = dp['reduced_or_features']
                X_ex = dp['reduced_ex_features']

                # Extract the position topology vector, which relates the
                # objects ordered by type to their position in the topology vector
                pos_topo_vect = dp['reduced_pos_topo_vect']

                # Extract the edges
                E = dp['edges']

                # Pass through the model
                if MAD_scores is None:
                    MAD_scores = model.get_MADs(X_gen, X_load, X_or, X_ex, E, pos_topo_vect).reshape(1, -1)
                else:
                    MAD = model.get_MADs(X_gen, X_load, X_or, X_ex, E, pos_topo_vect).reshape(1, -1)
                    MAD_scores = np.concatenate((MAD_scores, MAD), axis=0)

            # Stop evaluating validation datapoints
            if i > config['training']['settings']['dp_per_val_log']:
                break

        # If any, log metrics to wandb
        if wandb_run is not None and step is not None:
            running_metrics.log_to_wandb(wandb_run, step)

        if not train_config['settings']['advanced_val_analysis']:
            return

        figures = {}

        # Creating substation confusion matrix
        Y_subs = [(v if v is not None else -1) for v in Y_subs]
        P_subs = [(v if v is not None else -1) for v in P_subs]
        n_subs = config['rte_case14_realistic']['n_subs']
        classes = np.arange(-1, n_subs).tolist()
        disp = ConfusionMatrixDisplay.from_predictions(Y_subs, P_subs, labels=classes)
        confusion_matrix_figure = disp.figure_
        confusion_matrix_figure.set_size_inches(12, 12)
        figures['substation_confusion_matrix'] = confusion_matrix_figure

        # Plotting distributions for the true/predicted/postprocessed-predicted objects
        affected_object_figure, ax = plt.subplots(3, 1, sharex=True)
        n_obs = len(Y_obs)
        ax[0].bar(range(n_obs), Y_obs.cpu())
        ax[0].title.set_text('True object action distribution')
        ax[1].bar(range(n_obs), nearest_valid_P_obs.cpu())
        ax[1].title.set_text('Postprocessed predicted object action distribution')
        ax[2].bar(range(n_obs), P_obs.cpu())
        ax[2].title.set_text('Predicted object action distribution')
        affected_object_figure.tight_layout()
        figures['affected_objects'] = affected_object_figure

        # Logging label counters as stacked histogram
        comb_counter_mc = (correct_counter_label + wrong_counter_label).most_common()
        correct_indices = util.flatten([correct_counter_label[a] * [i] for i, (a, _) in enumerate(comb_counter_mc)])
        wrong_indices = util.flatten([wrong_counter_label[a] * [i] for i, (a, _) in enumerate(comb_counter_mc)])
        accuracy_per_label_figure, ax = plt.subplots()
        n_bins = max(correct_indices + wrong_indices) + 1
        ax.hist([correct_indices, wrong_indices], color=['lime', 'red'], bins=n_bins, stacked=True)
        figures['accuracy_per_label'] = accuracy_per_label_figure

        # Logging topoly vector counters as stacked histogram
        comb_counter_mc = (correct_counter_topovect + wrong_counter_topovect).most_common()
        correct_indices = util.flatten([correct_counter_topovect[a] * [i] for i, (a, _)
                                        in enumerate(comb_counter_mc)])
        wrong_indices = util.flatten([wrong_counter_topovect[a] * [i] for i, (a, _)
                                      in enumerate(comb_counter_mc)])
        accuracy_per_topo_vect_figure, ax = plt.subplots()
        n_bins = max(correct_indices + wrong_indices) + 1
        ax.hist([correct_indices, wrong_indices], color=['lime', 'red'], bins=n_bins, stacked=True)
        figures['accuracy_per_topovect'] = accuracy_per_topo_vect_figure

        # Logging lineout counters as a stacker bar plot
        accuracy_per_lineout_figure, ax = plt.subplots()
        bins = config['training']['settings']['line_outages_considered']
        ax.bar(bins,
               [0 if (k not in correct_counter_lineout.keys()) else correct_counter_lineout[k] for k in bins],
               color='lime')
        ax.bar(bins,
               [0 if (k not in wrong_counter_lineout.keys()) else wrong_counter_lineout[k] for k in bins],
               bottom=[0 if (k not in correct_counter_lineout.keys()) else correct_counter_lineout[k]
                       for k in bins],
               color='red')
        figures['accuracy_per_lineout'] = accuracy_per_lineout_figure

        # Logging mean MADs as a bar plot
        if train_config['hyperparams']['model_type'] == ModelType.GCN:
            mean_MAD = MAD_scores.mean(axis=0)
            mad_figure, ax = plt.subplots()
            ax.bar([str(i) for i in range(len(mean_MAD))], mean_MAD)
            figures['MAD'] = mad_figure

        if wandb_run is not None and step is not None:
            wandb.log({key: wandb.Image(fig) for key, fig in figures.items()}, step=step)

            # Logging histogram of the ranks of the true actions in the list of
            # valid actions sorted by nearness to the predicted actions,
            # and the corresponding top-k accuracies
            wandb_run.log({"Y_rank_in_nearest_v_acts": wandb.Histogram(Y_rank_in_nearest_v_acts)}, step=step)
            wandb_run.log({
                "top-1 accuracy": np.mean([x < 1 for x in Y_rank_in_nearest_v_acts]),
                "top-2 accuracy": np.mean([x < 2 for x in Y_rank_in_nearest_v_acts]),
                "top-3 accuracy": np.mean([x < 3 for x in Y_rank_in_nearest_v_acts]),
                "top-5 accuracy": np.mean([x < 5 for x in Y_rank_in_nearest_v_acts]),
                "top-10 accuracy": np.mean([x < 10 for x in Y_rank_in_nearest_v_acts]),
                "top-20 accuracy": np.mean([x < 20 for x in Y_rank_in_nearest_v_acts])
            }, step=step)

            # Logging difference between the self weights and the other weights
            if train_config['hyperparams']['model_type'] == ModelType.GCN and \
                    train_config['GCN']['hyperparams']['layer_type'] == LayerType.SAGECONV:
                diffs = model.compute_difference_weights()
                diffs = dict([('diffs_weights_' + k, v) for k, v in diffs.items()])
                wandb_run.log(dict([(k, wandb.Histogram(v)) for k, v in diffs.items()]),
                              step)

    return figures
