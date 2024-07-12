#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class with metrics for evaluation during training.

@author: Matthijs de Jong
"""

# Standard library imports
from typing import Optional, Dict, Callable, List, Tuple, Unpack

# Third-party library imports
import torch
import wandb


class IncrementalAverage:
    """
    A class for keeping an average that can be incremented one value at the time.
    """

    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def log(self, val: Optional[float]):
        """
        Increment the average with a value.

        Parameters
        ----------
        val : Optional[float]
            The value. Can be 'None', and it will then be ignored.
        """
        if val is not None:
            self.sum += val
            self.n += 1

    def get(self) -> float:
        """
        Get the value of the average.
        """
        return self.sum / self.n

    def reset(self):
        """
        Reset the average.
        """
        self.sum = 0.0
        self.n = 0


class IncrementalAverageMetrics:
    """
    A class for storing, updating, retrieving, printing, and resetting
    multiple incremental averages.
    """

    def __init__(self, metrics_dict: Dict[str, Tuple[Callable, IncrementalAverage]]):
        """
        Parameters
        ----------
        metrics_dict : dict
            The dictionary is of the typeDict[str,tuple(Callable,IncrementalAverage)].
            The string is the metric name.
            The 'Callable' is the function that computes the metric.
            The 'IncrementalAverage' is, surprisingly, the incremental average.
        """
        self.metrics_dict = metrics_dict

    def log(self, **kwargs: Unpack):
        """
        Update the incremental averages.

        Parameters
        ----------
        **kwargs : dict
            This dictionary contains arbitrary information that the metrics
            might need to update.
        """
        [run_avr.log(f(**kwargs)) for f, run_avr in self.metrics_dict.values()]

    def get_values(self) -> List[Tuple[str, float]]:
        """
        Returns
        -------
        List[Tuple[str,float]]
            List over the different metrics consisting of tuples.
            First element of each tuple represents the name of the metric,
            the second its average value.

        """
        return [(key, val[1].get()) for key, val in self.metrics_dict.items()]

    def log_to_wandb(self, run: wandb.sdk.wandb_run.Run, step: int):
        #  run: wandb.sdk.wandb_run.Run
        """
        Log the metrics to wandb.

        Parameters
        ----------
        run : wandb.sdk.wandb_run.Run
            The run to log to.
        step : int
            The current step in the run.
        """
        run.log(dict(self.get_values()), step=step)

    def reset(self):
        """
        Reset all incremental averages.
        """
        [run_avr.reset() for _, run_avr in self.metrics_dict.values()]

    def __str__(self):
        """
        Returns human-readable string representation of the metrics.

        Returns
        -------
        str
            The string.
        """
        return '\n'.join([f'{n}: {v}' for n, v in self.get_values()])


def macro_accuracy(**kwargs: dict) -> bool:
    """
    Calculates whether the predicted output wholly matches the true output.
    Differs from micro_accuracy in that it doesn't check the element-wise
    accuracy.

    Parameters
    ----------
    **kwargs['P'] : torch.Tensor[float]
        The output of the model. Elements are floats in the range (0,1).
        A value below 0.5 represents no change; above 0.5, change.
    **kwargs['Y'] : torch.Tensor[float]
        The label of the datapoints Elements are floats in {0,1}.
        A value below 0 represents no change; of 1, change.

    Returns
    -------
    bool
        Whether the predicted output matches the true output.
    """
    P = kwargs['P']
    Y = kwargs['Y']
    return torch.equal(torch.round(P), torch.round(Y))


def micro_accuracy(**kwargs: dict) -> float:
    """
    Calculates the element-wise accuracy between the predicted
    output and the true output.
    Differs from micro_accuracy in that this function checks the
    element-wise accuracy.

    Parameters
    ----------
    **kwargs['P'] : torch.Tensor[float]
        The output of the model. Elements are floats in the range (0,1).
        A value below 0.5 represents no change; above 0.5, change.
    **kwargs['Y'] : torch.Tensor[float]
        The label of the datapoints Elements are floats in {0,1}.
        A value below 0 represents no change; of 1, change.

    Returns
    -------
    float
        The element-wise accuracy.
    """
    P = kwargs['P']
    Y = kwargs['Y']
    return torch.mean(torch.eq(torch.round(P), torch.round(Y)).float()).item()


def macro_accuracy_one_sub(**kwargs: dict) -> bool:
    """
    Calculates whether the predicted output, after the postprocessing step of
    selecting the single most 'changed' substation has been applied, wholly
    matches the true output.
    Differs from micro_accuracy_one_sub in that it doesn't check the
    element-wise accuracy.
    Differs from macro_accuracy and macro_accuracy_valid in the
    postprocessing that has been applied to the prediction.

    Parameters
    ----------
    **kwargs['one_sub_P'] : torch.Tensor[float]
        The output of the model after the postprocessing step of
        selecting the single most 'changed' substation has been applied.
        Elements are floats in the range (0,1). A value below 0.5 represents
        no change; above 0.5, change.
    **kwargs['Y'] : torch.Tensor[float]
        The label of the datapoints Elements are floats in {0,1}.
        A value below 0 represents no change; of 1, change.

    Returns
    -------
    bool
        Whether the post-processed predicted output matches the true output.
    """
    one_sub_P = kwargs['one_sub_P']
    Y = kwargs['Y']
    return torch.equal(torch.round(one_sub_P), torch.round(Y))


def micro_accuracy_one_sub(**kwargs: dict) -> float:
    """
    Calculates the element-wise accuracy between predicted output, after the
    postprocessing step of selecting the single most 'changed' substation has
    been applied, and the true output.
    Differs from macro_accuracy_one_sub in that this function checks the
    element-wise accuracy.
    Differs from micro_accuracy and micro_accuracy_valid in the
    post-processing that has been applied to the prediction.

    Parameters
    ----------
    **kwargs['one_sub_P'] : torch.Tensor[float]
        The output of the model after the postprocessing step of
        selecting the single most 'changed' substation has been applied.
        Elements are floats in the range (0,1). A value below 0.5 represents
        no change; above 0.5, change.
    **kwargs['Y'] : torch.Tensor[float]
        The label of the datapoints Elements are floats in {0,1}.
        A value below 0 represents no change; of 1, change.

    Returns
    -------
    bool
        Whether the predicted output matches the true output.
    """
    one_sub_P = kwargs['one_sub_P']
    Y = kwargs['Y']
    return torch.mean(torch.eq(torch.round(one_sub_P),
                               torch.round(Y)).float()
                      ).item()


def macro_accuracy_valid(**kwargs: dict) -> bool:
    """
    Calculates whether the predicted output, after the postprocessing step of
    selecting the closest valid action has been applied, wholly
    matches the true output.
    Differs from micro_accuracy_valid in that it doesn't check the
    element-wise accuracy.
    Differs from macro_accuracy and macro_accuracy_one_sub in the
    postprocessing that has been applied to the prediction.

    Parameters
    ----------
    **kwargs['nearest_valid_P'] : torch.Tensor[float]
        The output of the model after the postprocessing step of
        selecting the nearest valid action has been applied.
        Elements are floats in the range (0,1). A value below 0.5 represents
        no change; above 0.5, change.
    **kwargs['Y'] : torch.Tensor[float]
        The label of the datapoints Elements are floats in {0,1}.
        A value below 0 represents no change; of 1, change.

    Returns
    -------
    bool
        Whether the post-processed predicted output matches the true output.
    """
    nearest_valid_P = kwargs['nearest_valid_P']
    Y = kwargs['Y']
    return torch.equal(nearest_valid_P, torch.round(Y))


def micro_accuracy_valid(**kwargs: dict) -> float:
    """
    Calculates the element-wise accuracy between predicted output, after the
    postprocessing step of selecting the closest valid action has been applied,
    and the true output.
    Differs from macro_accuracy_valid in that this function checks the
    element-wise accuracy.
    Differs from micro_accuracy and micro_accuracy_one_sub in the
    postprocessing that has been applied to the prediction.

    Parameters
    ----------
    **kwargs['one_sub_P'] : torch.Tensor[float]
        The output of the model after the postprocessing step of selecting the
        closest valid action has been applied
        Elements are floats in the range (0,1). A value below 0.5 represents
        no change; above 0.5, change.
    **kwargs['Y'] : torch.Tensor[float]
        The label of the datapoints Elements are floats in {0,1}.
        A value below 0 represents no change; of 1, change.

    Returns
    -------
    bool
        Whether the predicted output matches the true output.
    """
    nearest_valid_P = kwargs['nearest_valid_P']
    Y = kwargs['Y']
    return torch.mean(torch.eq(nearest_valid_P, torch.round(Y)).float()).item()


def n_predicted_changes(**kwargs: dict) -> int:
    """
    Calculates the number of predicted changes.

    Parameters
    ----------
    **kwargs['P'] : torch.Tensor[float]
        The output of the model. Elements are floats in the range (0,1).
        A value below 0.5 represents no change; above 0.5, change.

    Returns
    -------
    int
        The number of predicted changes.
    """
    P = kwargs['P']
    return torch.sum(torch.round(P)).item()


def any_predicted_changes(**kwargs: dict) -> bool:
    """
    Calculates whether there were any predicted changes.

    Parameters
    ----------
    kwargs
        The dictionary containing the necessary information for computing
        the changes.

    Returns
    -------
    bool
        Whether there were any predicted changes.
    """
    return n_predicted_changes(**kwargs) > 0


def accuracy_predicted_substation(**kwargs: dict) -> bool:
    """
    Calculates whether the substation where the changes are predicted
    at corresponds to the substation with the label changes.

    Parameters
    ----------
    **kwargs['Y_subchanged_idx'] : int
        The index of the substation where the 'true' changes would be applied.
    **kwargs['P_subchanged_idx'] : int
        The index of the substation where the predicted changes would be applied.

    Returns
    -------
    bool
        Whether the two substations indices match.
    """
    Y_subchanged_idx = kwargs['Y_subchanged_idx']
    P_subchanged_idx = kwargs['P_subchanged_idx']
    return Y_subchanged_idx == P_subchanged_idx

# =============================================================================
# def correct_whether_changes(**kwargs: dict):
#     '''
#     Check whether the model correctly predicted whether there were any
#     changes in the imitation example.
#     
#     Parameters
#     ----------
#     kwargs['any_changes'] : bool
#         Whether the imitation example had any changes.
#     kwargs['out'] : torch.Tensor
#         The output of the model. Elements are floats in the range (0,1).
#         A value below 0.5 represents no change; above 0.5, change.
#             
#     Returns
#     -------
#     float
#         Metric value. For this metric, 0 or 1.
#     '''
#     any_changes = kwargs['any_changes']
#     out = kwargs['out']
#     return any_changes == any(out>=0.5)
#     
# def whether_changes(**kwargs):
#     '''
#     Check whether the model predicted any changes.
#     
#     Parameters
#     ----------
#     kwargs['out'] : torch.Tensor
#         The output of the model. Elements are floats in the range (0,1).
#         A value below 0.5 represents no change; above 0.5, change.
#         
#     Returns
#     -------
#     float
#         Metric value. For this metric, 0 (no changes) or 1 (changes).
#     '''
#     out = kwargs['out']
#     return any(out>=0.5)
# 
# def accuracy_if_changes(**kwargs):
#     '''
#     Check the accuracy of the output assuming the imitation example included
#     changes. Returns 'None' otherwise.
#     
#     Parameters
#     ----------
#     kwargs['any_changes'] : bool
#         Whether the imitation example had any changes.
#     kwargs['set_idxs'] : torch.Tensor
#         The indexes of what objects were set (NOT changed!) in the imitation
#         example.
#     kwargs['set_objects'] : np.array[int]
#         The busbar-object connections that have been set in the imitation 
#         example, and whether that 'set' action caused a change. 
#         0 represents no change; 1, change.
#         The vector only represents the objects indexed by 'set_idxs'.
#     kwargs['out'] : torch.Tensor
#         The output of the model. Elements are floats in the range (0,1).
#         A value below 0.5 represents no change; above 0.5, change.
#         
#     Returns
#     -------
#     Optional[float]
#         If float, the accuracy.
#     '''
#     if kwargs['any_changes']:
#         out = kwargs['out'][kwargs['set_idxs']].round()
#         changed_objects = kwargs['set_objects']
#         return float(sum(out==changed_objects))/len(out)
#     else:
#         return None
# =============================================================================
