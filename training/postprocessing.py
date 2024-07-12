#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the postprocessing step that matches model outputs to valid actions.

@author: Matthijs de Jong
"""

# Standard library imports
from typing import Sequence, Optional, Union

# Third-party library imports
import torch
import grid2op

# Project imports
from auxiliary.generate_action_space import get_env_actions
import auxiliary.grid2op_util as g2o_util


class ActSpaceCache:
    """
    Class for storing the action spaces per line removed, so to make
    retrieving the action spaces more efficient.

    Supports functionality for finding the valid actions nearest to a
    predicted action.
    """

    def __init__(self,
                 env: Optional[grid2op.Environment.Environment] = None,
                 line_outages_considered: Sequence[int] = [-1],
                 reduce: bool = False):
        """
        Parameters
        ----------
        env : Optional[grid2op.Environment.Environment]
            The environment to make the actions for. If None, a new environment will be initialized.
        line_outages_considered : Sequence[int], optional
            For which lines removed to store the action space.
            -1 in this list represent no line removed. The default is [-1].
        """
        if env is None:
            env = g2o_util.init_env(grid2op.Rules.AlwaysLegal)

        self.set_act_space_per_lo = {}
        for lo in line_outages_considered:
            self.set_act_space_per_lo[lo] = torch.tensor([a._set_topo_vect for a in get_env_actions(env, lo)])

            if lo != -1:
                if reduce:
                    indices = [i for i in range(self.set_act_space_per_lo[lo].shape[1])
                               if i not in [env.line_or_pos_topo_vect[lo], env.line_ex_pos_topo_vect[lo]]]
                    self.set_act_space_per_lo[lo] = self.set_act_space_per_lo[lo][:, indices]
                else:
                    self.set_act_space_per_lo[lo][:, (env.line_or_pos_topo_vect[lo], env.line_ex_pos_topo_vect[lo])] = 0

    def get_change_actspace_by_nearness_pred(self,
                                             line_disabled: int,
                                             topo_vect: torch.Tensor,
                                             P: torch.Tensor,
                                             device: Union[torch.device, str]) -> torch.Tensor:
        """
        Given a prediction from the model:
            (1) compute the corresponding 'change' action space
            (2) order the corresponding 'change' action space based on the
            nearness to the prediction.

        Parameters
        ----------
        line_disabled : int
            The line disabled, used to index the correct action space.
            -1 represent the action space with no line disabled.
        topo_vect : torch.Tensor[int]
            The current topoly vector. Should have elements in {1,2}.
            Used to compute the 'change' action space. Should have a length
            corresponding to the number of objects in the network.
        P : torch.Tensor[float]
            The predictions from the model. Predictions are in range (0,1).
            Should have a length corresponding to the number of objects
            in the network.
        device : torch.device
            What device to load the data structures on.

        Returns
        -------
        change_act_space : torch.Tensor
            Sorted 'change' action space. Should have the shape
            (N_ACTIONS, N_OBJECTS). Sorted by increasing nearness to the
            predicted action space.
        """
        # Index the action space for 'set' actions
        set_act_space = self.set_act_space_per_lo[line_disabled].to(device)

        # Compute the 'change' action space
        topo_vect = topo_vect.reshape((1, -1))
        topo_vect_rpt = topo_vect.repeat(set_act_space.shape[0], 1)
        change_act_space = (set_act_space != 0) * (set_act_space != topo_vect_rpt)

        # Remove all do-nothing action
        change_act_space = change_act_space[change_act_space.sum(dim=1) != 0]

        # Add back a single do-nothing action
        change_act_space = torch.cat([torch.zeros((1, change_act_space.shape[1]), device=device), change_act_space])

        # Calculate the row-wise abs. difference between the predicted action
        # and the 'change' action space
        P_rpt = P.repeat(change_act_space.shape[0], 1)
        P_diff = abs(P_rpt - change_act_space)

        # Sort the 'change' action space by the difference with the
        # predicted action
        P_diff_ind_sorted = torch.argsort(P_diff.sum(axis=1))
        change_act_space = change_act_space[P_diff_ind_sorted]

        return change_act_space
