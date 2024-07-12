#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Util file for Grid2Op functionality.

@author: Matthijs de Jong
"""
# Standard library imports
from typing import Sequence, Tuple, List, Optional, Callable, Dict
import math

# Third party imports
import numpy as np
import grid2op
import torch
from grid2op.dtypes import dt_int

# Project imports
from auxiliary.config import get_config
import auxiliary.util as util


def connectivity_matrices(sub_info: Sequence[int],
                          topo_vect: Sequence[int],
                          line_or_pos_topo_vect: Sequence[int],
                          line_ex_pos_topo_vect: Sequence[int]
                          ) -> Tuple[np.array, np.array, np.array]:
    """
    Computes and return three connectivity matrices, based on three possible relations between objects. Matrices are
    returned as sparse matrices, represented by the indices of the edges. All relations are bidirectional, i.e.
    duplicated. Adapted from the Grid2Op package.

    Parameters
    ----------
    sub_info : Sequence[int]
        The number of objects per substation.
    topo_vect : Sequence[int]
        The bus to which each object is connected.
    line_or_pos_topo_vect : Sequence[int]
        The indices in the topo vector of the line origins.
    line_ex_pos_topo_vect : Sequence[int]
        The indices in the topo vector of the line extremities.


    Returns
    -------
    connectivity_matrix_samebus: np.array
        The sparse connectivity matrix between objects connected to the same bus
        of their substation.
    connectivity_matrix_otherbus = np.array
        The sparse connectivity matrix between objects connected to the other bus
        of their substation.
    connectivity_matrix_line = np.array
        The sparse connectivity matrix between objects connected by lines.
    """

    beg_ = 0
    end_ = 0
    row_ind_samebus = []
    col_ind_samebus = []
    row_ind_otherbus = []
    col_ind_otherbus = []
    row_ind_line = []
    col_ind_line = []

    for sub_id, nb_obj in enumerate(sub_info):
        # it must be a vanilla python integer, otherwise it's not handled by 
        # some backend
        # especially if written in c++
        nb_obj = int(nb_obj)
        end_ += nb_obj
        # tmp = np.zeros(shape=(nb_obj, nb_obj), dtype=dt_float)
        for obj1 in range(nb_obj):
            my_bus = topo_vect[beg_ + obj1]
            if my_bus == -1:
                # object is disconnected, nothing is done
                continue
            # connect an object to itself
            #                 row_ind.append(beg_ + obj1)
            #                 col_ind.append(beg_ + obj1)
            #                 WHY??

            # connect the other objects to it
            for obj2 in range(obj1 + 1, nb_obj):
                my_bus2 = topo_vect[beg_ + obj2]
                if my_bus2 == -1:
                    # object is disconnected, nothing is done
                    continue
                if my_bus == my_bus2:
                    # objects are on the same bus
                    # tmp[obj1, obj2] = 1
                    # tmp[obj2, obj1] = 1
                    row_ind_samebus.append(beg_ + obj2)
                    col_ind_samebus.append(beg_ + obj1)
                    row_ind_samebus.append(beg_ + obj1)
                    col_ind_samebus.append(beg_ + obj2)
                else:
                    # objects are on different bus 
                    row_ind_otherbus.append(beg_ + obj2)
                    col_ind_otherbus.append(beg_ + obj1)
                    row_ind_otherbus.append(beg_ + obj1)
                    col_ind_otherbus.append(beg_ + obj2)
        beg_ += nb_obj

    # both ends of a line are connected together (if line is connected)
    for q_id in range(len(line_or_pos_topo_vect)):
        if topo_vect[line_or_pos_topo_vect][q_id] != -1:
            # if powerline is connected, connect both its side
            row_ind_line.append(line_or_pos_topo_vect[q_id])
            col_ind_line.append(line_ex_pos_topo_vect[q_id])
            row_ind_line.append(line_ex_pos_topo_vect[q_id])
            col_ind_line.append(line_or_pos_topo_vect[q_id])

    row_ind_samebus = np.array(row_ind_samebus).astype(dt_int)
    col_ind_samebus = np.array(col_ind_samebus).astype(dt_int)
    row_ind_otherbus = np.array(row_ind_otherbus).astype(dt_int)
    col_ind_otherbus = np.array(col_ind_otherbus).astype(dt_int)
    row_ind_line = np.array(row_ind_line).astype(dt_int)
    col_ind_line = np.array(col_ind_line).astype(dt_int)

    assert all([i != j for i, j in list(zip(row_ind_samebus, col_ind_samebus))]), \
        "No object should be connected to itself."
    assert all([i != j for i, j in list(zip(row_ind_otherbus, col_ind_otherbus))]), \
        "No object should be connected to itself."
    assert all([i != j for i, j in list(zip(row_ind_line, col_ind_line))]), \
        "No object should be connected to itself."

    connectivity_matrix_samebus = np.stack((row_ind_samebus, col_ind_samebus))
    connectivity_matrix_otherbus = np.stack((row_ind_otherbus, col_ind_otherbus))
    connectivity_matrix_line = np.stack((row_ind_line, col_ind_line))

    return connectivity_matrix_samebus, \
        connectivity_matrix_otherbus, \
        connectivity_matrix_line


def connectivity_matrices_to_hetero_connectivity_matrices(gen_pos_topo_vect: np.array,
                                                          load_pos_topo_vect: np.array,
                                                          line_or_pos_topo_vect: np.array,
                                                          line_ex_pos_topo_vect: np.array,
                                                          edges_dict: Dict[str, Tuple[List, List]]) \
        -> Dict[Tuple[str, str, str], List[Tuple[int, int]]]:
    """
    Given a dictionary of edge types and their corresponding edges, split these edges into edges based on
    the edge type and the endpoint types. The point of this class is to store the data in such a way that makes
    it easy to initialize a Pytorch Geometric HeteroData class from it.

    Parameters
    ----------
    gen_pos_topo_vect : np.array
        Vector representing the indices of the generators in the topo_vect.
    load_pos_topo_vect : np.array
        Vector representing the indices of the loads in the topo_vect.
    line_or_pos_topo_vect : np.array
        Vector representing the indices of the line origins in the topo vect.
    line_ex_pos_topo_vect : np.array
        Vector representing the indices of the line extremities in the topo vect.
    edges_dict: Dict[str. Tuple(List, List)]
        The dictionary of edges types and their corresponding edges.

    Returns
    -------
    hetero_edge_dict: Dict[Tuple(str, str, str), List(Tuple(int, int)]
    """

    hetero_edges_dict = {}

    # For each edge type
    for edge_type, edges in edges_dict.items():
        # For each edge of that type
        for incident1, incident2 in list(zip(*edges)):
            incident1_type = None
            incident1_pos = None
            incident2_type = None
            incident2_pos = None

            # Find out the types and positions of the connected nodes
            for node_type, pos_topo_vect in [('gen', gen_pos_topo_vect),
                                             ('load', load_pos_topo_vect),
                                             ('or', line_or_pos_topo_vect),
                                             ('ex', line_ex_pos_topo_vect)]:
                if incident1 in pos_topo_vect:
                    incident1_type = node_type
                    incident1_pos = pos_topo_vect.index(incident1)
                if incident2 in pos_topo_vect:
                    incident2_type = node_type
                    incident2_pos = pos_topo_vect.index(incident2)

            assert incident1_pos is not None and incident2_pos is not None, \
                "Incidents positions should have been found among the pos topo vects."

            # Combine information into hetero edge type and node incidences
            hetero_edge_type = (incident1_type, edge_type, incident2_type)
            hetero_edge_pos = (incident1_pos, incident2_pos)

            # Append to the dict
            if hetero_edge_type in hetero_edges_dict:
                hetero_edges_dict[hetero_edge_type].append(hetero_edge_pos)
            else:
                hetero_edges_dict[hetero_edge_type] = [hetero_edge_pos]

    return hetero_edges_dict


def tv_groupby_subst(tv: Sequence, sub_info: Sequence[int]) -> \
        List[Sequence]:
    """
    Group a sequence the shape of the topology vector by the substations.

    Parameters
    ----------
    tv : Sequence
        Sequence the shape of the topology vector.
    sub_info : Sequence[int]
        Sequence with elements containing the number of object connected to each substation.

    Returns
    -------
    List[Sequence]
        List, each element corresponding to a Sequence of objects in tv that belong to a particular substation.
    """
    i = 0
    gs = []
    for ss in sub_info:
        gs.append(tv[i:i + ss])
        i += ss
    return gs


def select_single_substation_from_topovect(topo_vect: torch.Tensor,
                                           sub_info: torch.Tensor,
                                           f: Callable = torch.sum,
                                           select_nothing_condition: Callable = lambda tv: all(tv < 0.5)) \
        -> Tuple[torch.Tensor, Optional[int]]:
    """
    Given a topology vector, select the substation whose objects maximize some function. From this substation, the mask
    in the topology vector and the index are returned. If a certain condition is met, the function can also select
    no substation, returning a zeroed array and an index of None.

    Parameters
    ----------
    topo_vect : torch.Tensor
        The vector based on which select a substation.
    sub_info : torch.Tensor
        Vector describing the number of objects per substation. Used to group topo_vect into objects of separate
        substations.
    f : Callable
        Function, based on the argmax of which the substation is selected.
    select_nothing_condition : Callable
        A condition on the topo_vect, which, if it holds true, will select no substation.

    Returns
    -------
    torch.Tensor
        The mask of the selected substation (one at the substation, zero everywhere else). Fully zeroed if 
        select_nothing_condition evaluates to true.
    Optional[int]
        Index of the substation. None if select_nothing_condition evaluates to true.
    """
    assert len(topo_vect) == sum(sub_info), "Length of topo vect should correspond to the sum of the " \
                                            "substation objects."

    if select_nothing_condition(topo_vect):
        return torch.zeros_like(topo_vect), None

    topo_vect_grouped = tv_groupby_subst(topo_vect, sub_info)
    selected_substation_idx = util.argmax_f(topo_vect_grouped, f)
    selected_substation_mask = torch.cat([(torch.ones_like(sub)
                                           if i == selected_substation_idx
                                           else torch.zeros_like(sub))
                                          for i, sub in enumerate(topo_vect_grouped)]).bool()

    return selected_substation_mask, selected_substation_idx


def init_env() -> grid2op.Environment.Environment:
    """
    Prepares the Grid2Op environment from a dictionary containing configuration setting.

    Returns
    -------
    env : TYPE
        The Grid2Op environment.
    """
    config = get_config()
    data_path = config['paths']['rte_case14_realistic']
    scenario_path = data_path + 'chronics/'

    # Required for topology reversal
    param = grid2op.Parameters.Parameters()
    param.MAX_SUB_CHANGED = 14

    env = grid2op.make(dataset=data_path,
                       chronics_path=scenario_path,
                       gamerules_class=grid2op.Rules.DefaultRules,
                       param=param,
                       test=True)

    # for reproducible experiments
    env.seed(config['simulation']['seed'])

    # Set custom thermal limits
    thermal_limits = config['rte_case14_realistic']['thermal_limits']
    env.set_thermal_limit(thermal_limits)

    return env


def ts_to_day(ts: int, ts_in_day: int) -> int:
    """
    Calculate what day (as a number) a timestep is in.

    Parameters
    ----------
    ts : int
        The timestep.
    ts_in_day : int
        The number of timesteps in a day.

    Returns
    -------
    int
        The day.
    """
    return math.floor(ts / ts_in_day)


def skip_to_next_day(env: grid2op.Environment.Environment, ts_in_day: int, chronic_id: int, disable_line: int):
    """
    Skip the environment to the next day.

    Parameters
    ----------
    env : grid2op.Environment.Environment
        The environment to fast-forward to the next day in.
    ts_in_day : int
        The number of timesteps in a day.
    chronic_id : int
        The current chronic id.
    disable_line : int
        The index of the line to be disabled.

    Raises
    -------
    grid2op.Exception.DivergingPowerFlowException
    """
    # Reset environment
    ts_next_day = ts_in_day * (1 + ts_to_day(env.nb_time_step, ts_in_day))
    env.set_id(chronic_id)
    env.reset()

    # Fast-forward to the day, disable lines if necessary
    env.fast_forward_chronics(ts_next_day - 1)
    if disable_line != -1:
        env_step_raise_exception(env, env.action_space({"set_line_status": (disable_line, -1)}))
    else:
        env_step_raise_exception(env, env.action_space())


def env_step_raise_exception(env: grid2op.Environment.Environment, action: grid2op.Action.BaseAction) \
        -> grid2op.Observation.BaseObservation:
    """
    Performs a step in the grid2op environment. Raises exceptions if they occur in the grid.

    Parameters
    ----------
    env : grid2op.Environment.Environment
        The grid2op environment.
    action : grid2op.Action.BaseAction
        The action to take in the former environment.

    Raises
    -------
    ExceptionGroup

    Returns
    -------
    obs : grid2op.Observation.CompleteObservation
        The observation resulting from the action.
    """
    obs, _, _, info = env.step(action)

    if len(info['exception']) > 1:
        raise ExceptionGroup('Exceptions', info['exception'])
    elif len(info['exception']) == 1:
        raise info['exception'][0]

    return obs
