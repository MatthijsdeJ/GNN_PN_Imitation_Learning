"""
Module for the config file.

The config dict is set as followed:
1) The config.yaml file is loaded in.
2) The config dict can be overwritten by overwrite_config() ONLY BEFORE IT HAS BEEN ACCESSED BY get_config().
This functionality exists to allow command-line arguments of scripts  to override the values in the config dict. To
reduces errors, this overwriting can only occur before the config dict has been accessed. Consequently, access to the
config dict from other modules should always happen through get_config().

@author: Matthijs de Jong
"""
# Standard library imports
from typing import Dict, Hashable, Sequence
from enum import Enum, unique

# Third party imports
import yaml

@unique
class NetworkType(Enum):
    HETERO = 'heterogeneous'
    HOMO = 'homogeneous'


@unique
class AggrType(Enum):
    ADD = 'add'
    MEAN = 'mean'


@unique
class ModelType(Enum):
    GCN = 'GCN'
    FCNN = 'FCNN'


@unique
class LayerType(Enum):
    SAGECONV = 'SAGEConv'
    GINCONV = 'GINConv'


@unique
class StrategyType(Enum):
    IDLE = 'idle'
    GREEDY = 'greedy'
    N_MINUS_ONE = 'nminusone'
    NAIVE_ML = 'naive_ml'
    VERIFY_ML = 'verify_ml'
    VERIFY_GREEDY_HYBRID = 'verify_greedy_hybrid'
    VERIFY_N_MINUS_ONE_HYBRID = 'verify_nminusone_hybrid'


@unique
class LabelWeightsType(Enum):
    ALL = 'ALL'
    Y = 'Y'
    P = 'P'
    Y_AND_P = 'Y_AND_P'


_config = None
_has_been_accessed = False

with open('config.yaml') as stream:
    try:
        _config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise exc


# Perform some assertions
def assert_config():
    """
    Check validity of the config values.
    """
    for prm, n in [(_config['simulation']['n_chronics'], 'n_chronics'),
                   (_config['rte_case14_realistic']['ts_in_day'], 'ts_in_day'),
                   (_config['rte_case14_realistic']['n_subs'], 'n_subs'),
                   (_config['training']['settings']['train_log_freq'], 'train_log_freq'),
                   (_config['training']['settings']['val_log_freq'], 'val_log_freq'),
                   (_config['training']['hyperparams']['n_epoch'], 'n_epoch'),
                   (_config['training']['hyperparams']['lr'], 'lr'),
                   (_config['training']['hyperparams']['N_node_hidden'], 'N_node_hidden'),
                   (_config['training']['hyperparams']['LReLu_neg_slope'], 'LReLu_neg_slope'),
                   (_config['training']['hyperparams']['batch_size'], 'batch_size'),
                   (_config['training']['hyperparams']['label_smoothing_alpha'], 'label_smoothing_alpha'),
                   (_config['training']['hyperparams']['weight_init_std'], 'weight_init_std'),
                   (_config['training']['hyperparams']['weight_decay'], 'weight_decay'),
                   (_config['training']['hyperparams']['label_weights']['non_masked_weight'],
                    'non_sub_weight'),
                   (_config['training']['hyperparams']['early_stopping_patience'], 'early_stopping_patience'),
                   (_config['training']['constants']['estimated_train_size'], 'estimated_train_size'),
                   (_config['training']['GCN']['hyperparams']['N_GCN_layers'], 'N_GCN_layers'),
                   (_config['training']['GCN']['constants']['N_f_gen'], 'N_f_gen'),
                   (_config['training']['GCN']['constants']['N_f_load'], 'N_f_load'),
                   (_config['training']['GCN']['constants']['N_f_endpoint'], 'N_f_endpoint'),
                   (_config['training']['FCNN']['hyperparams']['N_layers'], 'N_layers'),
                   (_config['training']['FCNN']['constants']['size_in'], 'size_in'),
                   (_config['training']['FCNN']['constants']['size_out'], 'size_out'),
                   (_config['training']['GCN']['hyperparams']['GINConv_nn_depth'], 'GINConv_nn_depth')]:
        assert prm >= 0, f'Parameter {n} should not be negative.'
    assert all(line >= 0 for line in _config['simulation']['NMinusOne_strategy']['line_idxs_to_consider_N-1']), \
        "Line idx cannot be negative."
    assert all(line >= 0 for line in _config['rte_case14_realistic']['thermal_limits']), \
        "Thermal limit cannot be negative."
    assert (max(_config['simulation']['NMinusOne_strategy']['line_idxs_to_consider_N-1']) + 1 <=
            len(_config['rte_case14_realistic']['thermal_limits'])), "Line idx plus one cannot be higher than" + \
                                                                     " the number of lines."
    assert _config['training']['wandb']['mode'] in ['online', 'offline', 'disabled'], \
        "WandB mode should be online, offline, or disabled."


def cast_config_to_enums():
    """
    Cast the values in the config types that have an enum representation to their enum types.
    """
    _config['training']['hyperparams']['model_type'] = ModelType(_config['training']['hyperparams']['model_type'])
    _config['training']['GCN']['hyperparams']['network_type'] = NetworkType(_config['training']
                                                                            ['GCN']['hyperparams']['network_type'])
    _config['training']['GCN']['hyperparams']['aggr'] = AggrType(_config['training']['GCN']['hyperparams']['aggr'])
    _config['training']['GCN']['hyperparams']['layer_type'] = LayerType(_config['training']['GCN']['hyperparams']
                                                                        ['layer_type'])
    _config['simulation']['strategy'] = StrategyType(_config['simulation']['strategy'])
    _config['training']['hyperparams']['label_weights']['type'] = LabelWeightsType(_config['training']['hyperparams']
                                                                                   ['label_weights']['type'])


assert_config()
cast_config_to_enums()


def nested_overwrite(dic: Dict, keys: Sequence[Hashable], value):
    """
    Overwrite a value in a nested dict based on a sequence of keys and a value. Raises IndexException if any of the keys
    are not yet in the nested structure. Casts value to the type of the to-be-overwritten value.
    """
    for key in keys[:-1]:
        dic = dic[key]

    if keys[-1] not in dic:
        raise IndexError(f"Key {keys[-1]} does not already exist.")

    if ',' in value:
        # Treat it as a list
        value = value.split(',')
        value = [type(dic[keys[-1]][0])(v) for v in value]
    else:
        value = type(dic[keys[-1]])(value)

    dic[keys[-1]] = value


def overwrite_config(keys: Sequence[Hashable], value):
    """
    Overwrite the config dict. Casts value to the type of

    Parameters
    ----------
    keys: Sequence[Hashable]
        Sequence of keys for the value to write over.
    value
        The value to write.
    """
    if _has_been_accessed:
        raise Exception("Overwriting is not allowed after the config has been accessed.")

    nested_overwrite(_config, keys, value)


def get_config() -> Dict:
    """
    Gets the config dictionary.

    Returns
    -------
    _config: Dict
        The config dictionary.
    """
    global _has_been_accessed

    # At the first access, check assertions again.
    if not _has_been_accessed:
        assert_config()

    _has_been_accessed = True
    return _config


def parse_args_overwrite_config(args):
    """
    Given a sequence of args of the form keys=value, where keys themselves consist of multiple strings seperated by '.'.
    Use the keys to overwrite value in the config.

    Parameters
    ----------
    args : Sequence[Hashable]
        The sequence of args.
    """
    for arg in args:
        if '=' in arg:
            keys, value = arg.split('=', 1)
            overwrite_config(keys.split('.'), value)
        else:
            raise ValueError(f"Argument '{arg}' does not contain '=' sign.")
