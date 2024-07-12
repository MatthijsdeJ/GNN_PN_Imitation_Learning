#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class of the ML models.

@author: Matthijs de Jong
"""
# Standard library imports
from typing import Sequence
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

# Third-party library imports
import numpy as np
import torch
import torch_geometric.nn
from torch_geometric.nn import SAGEConv, Linear, HeteroConv, GINConv

# Project imports
from auxiliary.config import NetworkType, AggrType, LayerType


class Model(ABC, torch.nn.Module):
    """
    Base class for the models.
    """

    @abstractmethod
    def init_weights_normalized_normal(self, weight_init_std: float):
        """
        Initialize the weights of the network according to the normal
        distribution, but with the std divided by the number of in channels.
        The biases are initialized to zero.

        Parameters
        ----------
        weight_init_std : float
            The standard deviation of the normal distribution.
        """
        pass


class GCN(Model):
    """
    Graph convolutional network model.
    Consists of: two embedding layers that should embed the different object
    features into a common embedding; and a number of GCN layers.
    """

    def __init__(self,
                 LReLu_neg_slope: float,
                 weight_init_std: float,
                 N_f_gen: int,
                 N_f_load: int,
                 N_f_endpoint: int,
                 N_GCN_layers: int,
                 N_node_hidden: int,
                 aggr: AggrType,
                 network_type: NetworkType,
                 layer_type: LayerType,
                 GINConv_nn_depth: Optional[int]):
        """
        Parameters
        ----------
        LReLu_neg_slope : float
            The negative slope of the LReLu activation function.
        weight_init_std: float,
            The standard deviation of the normal distribution according to
            which the weights are initialized.
        N_f_gen : int
            The number of features per generator object.
        N_f_load : int
            The number of features per load object.
        N_f_endpoint : int
            The number of features per endpoint object.
        N_GCN_layers : int
            The number of GCN layers.
        N_node_hidden : int
            The number of hidden nodes in the hidden layers.
        aggr : AggrType
            The aggregation function for GCN layers. Should be 'add' or 'mean'.
        network_type : NetworkType
            The type of network.
        layer_type : LayerType
            The type of layer.
        GINConv_nn_depth : Optional[int]
            The depth of the nn used for GINConv layers.
        """
        super().__init__()
        self.network_type = network_type
        self.aggr_str = aggr.value
        self.layer_type = layer_type
        self.GINConv_nn_depth = GINConv_nn_depth
        self.N_node_hidden = N_node_hidden
        self.LReLu_neg_slope = LReLu_neg_slope

        # The activation function. Inplace helps make it work for both
        # network types.
        self.LReLu_neg_slope = LReLu_neg_slope
        self.activation_f = torch.nn.LeakyReLU(LReLu_neg_slope, inplace=True)

        # The embedding layers
        self.lin_gen_1 = Linear(N_f_gen, N_node_hidden)
        self.lin_gen_2 = Linear(N_node_hidden, N_node_hidden)

        self.lin_load_1 = Linear(N_f_load, N_node_hidden)
        self.lin_load_2 = Linear(N_node_hidden, N_node_hidden)

        self.lin_line_1 = Linear(N_f_endpoint, N_node_hidden)
        self.lin_line_2 = Linear(N_node_hidden, N_node_hidden)

        # Create the GCN layers
        self.GCN_layers = torch.nn.ModuleList([self.create_GCN_layer(N_node_hidden, N_node_hidden)
                                               for _ in range(N_GCN_layers - 1)])
        # Create the final layer
        self.GCN_layers.append(self.create_GCN_layer(N_node_hidden, 1))

        # Initialize weights according to normal distribution
        self.init_weights_normalized_normal(weight_init_std)

    def forward(self,
                x_gen: torch.Tensor,
                x_load: torch.Tensor,
                x_or: torch.Tensor,
                x_ex: torch.Tensor,
                edge_index: torch.Tensor,
                object_ptv: torch.Tensor) -> torch.Tensor:
        """
        Passes the datapoint through the network.

        Parameters
        ----------
        x_gen : torch.Tensor
            The generator features. Columns represent features, rows different
            generator objects.
        x_load : torch.Tensor
            The load features. Columns represent features, rows different
            load objects.
        x_or : torch.Tensor
            The line origin features. Columns represent features, rows different
            line origin objects.
        x_ex : torch.Tensor
            The line extremity features. Columns represent features, rows different
            line extremity objects.
        edge_index : torch.Tensor
            The edge indices of the adjacency matrix. Edge indices indicate
            the indices of objects in the topoly vector.
            Should have shape (2,num_edges). Should be symmetric.
        object_ptv : torch.Tensor
            Vector describing the permutation required to set the different object
            at their position in the topology vector.

        Returns
        -------
        x : torch.Tensor
            The output vector. Values should be in range (0,1).
        """
        HOMO = NetworkType.HOMO
        HETERO = NetworkType.HETERO

        # Passing the object features through their respective
        # embedding layers
        x_gen = self.lin_gen_1(x_gen)
        self.activation_f(x_gen)
        x_gen = self.lin_gen_2(x_gen)
        self.activation_f(x_gen)
        x_load = self.lin_load_1(x_load)
        self.activation_f(x_load)
        x_load = self.lin_load_2(x_load)
        self.activation_f(x_load)
        x_or = self.lin_line_1(x_or)
        self.activation_f(x_or)
        x_or = self.lin_line_2(x_or)
        self.activation_f(x_or)
        x_ex = self.lin_line_1(x_ex)
        self.activation_f(x_ex)
        x_ex = self.lin_line_2(x_ex)
        self.activation_f(x_ex)

        # Combining different object states into the order of the topology vector
        x = torch.cat([x_gen, x_load, x_or, x_ex], axis=0)
        x = x[object_ptv]

        if self.network_type == HETERO:
            x = {'object': x}

        # Pass through the GCN layers
        for layer in self.GCN_layers[:-1]:
            x = layer(x, edge_index)
            self.activation_f(x if self.network_type == HOMO else x['object'])

        # Pass through the final layer and the sigmoid activation function
        x = self.GCN_layers[-1](x, edge_index)
        x = torch.sigmoid(x if self.network_type == HOMO else x['object'])

        return x

    def create_GCN_layer(self, n_in: int, n_out: int) -> torch_geometric.nn.MessagePassing:
        """
        Creates a GCN layer.

        Parameters
        ----------
        n_in : int
            The dimensionality of the layer output.
        n_out : int
            The dimensionality of the layer input.

        Returns
        -------
        torch_geometric.nn.MessagePassing
            The output layer.
        """
        if self.layer_type == LayerType.SAGECONV:
            if self.network_type == NetworkType.HOMO:
                return SAGEConv(n_in, n_out, root_weight=True, aggr=self.aggr_str)
            elif self.network_type == NetworkType.HETERO:
                return HeteroConv({
                    ('object', 'line', 'object'):
                        SAGEConv(n_in, n_out, root_weight=False, aggr=self.aggr_str, bias=False),
                    ('object', 'same_busbar', 'object'):
                        SAGEConv(n_in, n_out, root_weight=True, aggr=self.aggr_str, bias=True),
                    ('object', 'other_busbar', 'object'):
                        SAGEConv(n_in, n_out, root_weight=False, aggr=self.aggr_str, bias=False),
                }, aggr='sum' if self.aggr_str == 'add' else self.aggr_str)
            else:
                raise ValueError('Invalid value for network_type.')
        elif self.layer_type == LayerType.GINCONV:
            if self.network_type == NetworkType.HOMO:
                return GINConv(self._create_GINCov_NN(n_in, n_out))
            elif self.network_type == NetworkType.HETERO:
                return HeteroConv({
                    ('object', 'line', 'object'): GINConv(self._create_GINCov_NN(n_in, n_out)),
                    ('object', 'same_busbar', 'object'): GINConv(self._create_GINCov_NN(n_in, n_out)),
                    ('object', 'other_busbar', 'object'): GINConv(self._create_GINCov_NN(n_in, n_out)),
                }, aggr='sum' if self.aggr_str == 'add' else self.aggr_str)
            else:
                raise ValueError('Invalid value for network_type.')
        else:
            raise ValueError('Invalid value for layer_type.')

    def _create_GINCov_NN(self, n_in, n_out) -> torch.nn.Module:
        """
        Creates a neural network that is used by the GINConv layer.

        Parameters
        ----------
        n_in : int
            The dimensionality of the layer output.
        n_out : int
            The dimensionality of the layer input.

        Returns
        -------
        torch.nn.Module
            The NN to be used by the GINConv Layer.
        """
        if self.GINConv_nn_depth == 1:
            return Linear(n_in, n_out, bias=True)

        modules = []
        modules.append(Linear(n_in, self.N_node_hidden, bias=True))
        modules.append(torch.nn.LeakyReLU(self.LReLu_neg_slope))
        for _ in range(self.GINConv_nn_depth - 2):
            modules.append(Linear(self.N_node_hidden, self.N_node_hidden))
            modules.append(torch.nn.LeakyReLU(self.LReLu_neg_slope))
        modules.append(Linear(self.N_node_hidden, n_out, bias=True))
        return torch.nn.Sequential(*modules)

    def init_weights_kaiming(self):
        """
        Initialize the weights of this network according to the kaiming uniform
        distribution. The biases are initialized to zero.
        """

        def layer_weights_kaiming(m):
            """
            Apply kaiming initialization to a single layer.
            """
            classname = m.__class__.__name__
            # for every Linear layer in a model.
            if classname.find('Linear') != -1:
                torch.nn.init.kaiming_normal_(m.weight, a=self.LReLu_neg_slope)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(layer_weights_kaiming)

    def init_weights_normalized_normal(self, weight_init_std: float):
        """
        Initialize the weights of this network according to the normal
        distribution, but with the std divided by the number of in channels.
        The biases are initialized to zero.

        Parameters
        ----------
        weight_init_std : float
            The standard deviation of the normal distribution.
        """

        def layer_weights_normal(m):
            """
            Apply initialization to a single layer.
            """
            classname = m.__class__.__name__
            # for every Linear layer in a model.
            if classname.find('Linear') != -1:
                std = weight_init_std / m.in_channels
                torch.nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(layer_weights_normal)

    def compute_difference_weights(self) -> Dict[str, List[float]]:
        """
        Compute the difference between the self weights and the neighbour
        weights.

        Returns
        -------
        diffs : Dict[str,List[float]]
            The dictionary  of lists (each lists corresponding to one neighbour
            weight type) with differences (each difference corresponding to a
            one layer). The number of lists depends on the network type.
        """

        def l_w_norm(layer):
            """
            Calculate the norm of the weights of a layer.
            """
            return abs(layer.weight).sum().item()

        diffs = {}
        if self.network_type == NetworkType.HETERO:

            diffs['self_line_neigh'] = []
            diffs['self_sb_neigh'] = []
            diffs['self_ob_neigh'] = []

            for layer in self.GCN_layers:
                l_key = '<object___line___object>'
                sb_key = '<object___same_busbar___object>'
                ob_key = '<object___other_busbar___object>'

                norm_w_self = l_w_norm(layer.convs[sb_key].lin_r)
                norm_w_line = l_w_norm(layer.convs[l_key].lin_l)
                norm_w_sb = l_w_norm(layer.convs[sb_key].lin_l)
                norm_w_ob = l_w_norm(layer.convs[ob_key].lin_l)

                diffs['self_line_neigh'].append(norm_w_self - norm_w_line)
                diffs['self_sb_neigh'].append(norm_w_self - norm_w_sb)
                diffs['self_ob_neigh'].append(norm_w_self - norm_w_ob)

        elif self.network_type == NetworkType.HOMO:
            diffs['self_neigh'] = []

            for layer in self.GCN_layers:
                norm_w_self = l_w_norm(layer.lin_r)
                norm_w_neigh = l_w_norm(layer.lin_l)
                diffs['self_neigh'].append(norm_w_self - norm_w_neigh)

        return diffs

    def get_MADs(self,
                 x_gen: torch.Tensor,
                 x_load: torch.Tensor,
                 x_or: torch.Tensor,
                 x_ex: torch.Tensor,
                 edge_index: torch.Tensor,
                 object_ptv: torch.Tensor) -> Sequence[float]:
        """
        Passes the datapoint through the network, and return the mean average distance (MAD) for the different
        GCN layers. Used to inspect the degree of oversmoothing.

        Parameters
        ----------
        x_gen : torch.Tensor
            The generator features. Columns represent features, rows different
            generator objects.
        x_load : torch.Tensor
            The load features. Columns represent features, rows different
            load objects.
        x_or : torch.Tensor
            The line origin features. Columns represent features, rows different
            line origin objects.
        x_ex : torch.Tensor
            The line extremity features. Columns represent features, rows different
            line extremity objects.
        edge_index : torch.Tensor
            The edge indices of the adjacency matrix. Edge indices indicate
            the indices of objects in the topoly vector.
            Should have shape (2,num_edges). Should be symmetric.
        object_ptv : torch.Tensor
            Vector describing the permutation required to set the different object
            at their position in the topology vector.

        Returns
        -------
        MADs : np.array[float]
            Array of MADs metrics before each layer.
        """
        HOMO = NetworkType.HOMO
        HETERO = NetworkType.HETERO

        # Passing the object features through their respective
        # embedding layers
        x_gen = self.lin_gen_1(x_gen)
        self.activation_f(x_gen)
        x_gen = self.lin_gen_2(x_gen)
        self.activation_f(x_gen)
        x_load = self.lin_load_1(x_load)
        self.activation_f(x_load)
        x_load = self.lin_load_2(x_load)
        self.activation_f(x_load)
        x_or = self.lin_line_1(x_or)
        self.activation_f(x_or)
        x_or = self.lin_line_2(x_or)
        self.activation_f(x_or)
        x_ex = self.lin_line_1(x_ex)
        self.activation_f(x_ex)
        x_ex = self.lin_line_2(x_ex)
        self.activation_f(x_ex)

        # Combining different object states into the order of the
        # topology vector
        MADs = np.array([])
        x = torch.cat([x_gen, x_load, x_or, x_ex], axis=0)
        x = x[object_ptv]
        combined_edges = edge_index if self.network_type == HOMO else torch.cat(tuple(edge_index.values()), axis=1)

        if self.network_type == HETERO:
            x = {'object': x}
        MADs = np.append(MADs, self.get_MAD_from_hidden_state(x if self.network_type == HOMO else x['object'],
                                                              combined_edges))

        # Pass through the GCN layers
        for l in self.GCN_layers[:-1]:
            x = l(x, edge_index)
            self.activation_f(x if self.network_type == HOMO else x['object'])
            MADs = np.append(MADs, self.get_MAD_from_hidden_state(x if self.network_type == HOMO else x['object'],
                                                                  combined_edges))

        # Pass through the final layer and the sigmoid activation function
        x = self.GCN_layers[-1](x, edge_index)
        x = torch.sigmoid(x if self.network_type == HOMO else x['object'])

        return MADs

    def get_MAD_from_hidden_state(self,
                                  H: torch.Tensor,
                                  edge_index: torch.Tensor) -> float:
        """
        For a given hidden state, compute the MAD (mean average distance) between neighbouring nodes.

        Parameters
        ----------
        H : torch.Tensor
            The hidden state.
        edge_index : torch.Tensor
            The tensor of edge indices, indicating which nodes are connected.

        Returns
        -------
        MAD : float
            The mean average distance
        """
        # Calculate the cosine distance between nodes
        similarity_between_nodes = H @ H.t()
        H_norm_per_node = torch.linalg.vector_norm(H, dim=1).reshape(-1, 1)
        cosine_dist_between_nodes = 1 - (similarity_between_nodes / (H_norm_per_node @ H_norm_per_node.t()))

        # Calculate the MAD
        adjacency_matrix = torch.squeeze(torch_geometric.utils.to_dense_adj(edge_index))
        cosine_dist_between_connected_nodes = cosine_dist_between_nodes * adjacency_matrix
        # TODO: this function only works for networks where each node has at least one connection.
        # Otherwise, a division by zero occurs.
        mean_cos_dist_node_to_neighbours = cosine_dist_between_connected_nodes.sum(axis=0) \
                                           / adjacency_matrix.sum(axis=0)
        MAD = mean_cos_dist_node_to_neighbours.mean().item()

        return MAD


class FCNN(Model):
    """
    Fully connected neural network. Consists of multiple feedforward layers.
    """

    def __init__(self,
                 LReLu_neg_slope: float,
                 weight_init_std: float,
                 size_in: int,
                 size_out: int,
                 N_layers: int,
                 N_node_hidden: int):
        """
        Parameters
        ----------
        LReLu_neg_slope : float
            The negative slope of the LReLu activation function.
        weight_init_std: float,
            The standard deviation of the normal distribution according to which the weights are initialized.
        N_layers : int
            The number of layers.
        N_node_hidden : int
            The number of nodes in the hidden layers.
        """
        super().__init__()

        # The activation function.
        self.LReLu_neg_slope = LReLu_neg_slope
        self.activation_f = torch.nn.LeakyReLU(LReLu_neg_slope)

        # The first layer
        self.lin_first = Linear(size_in, N_node_hidden)

        # Create the middle layers
        self.lin_middle_layers = torch.nn.ModuleList([Linear(N_node_hidden, N_node_hidden)
                                                      for _ in range(N_layers - 2)])

        # The last layer
        self.lin_last = Linear(N_node_hidden, size_out)

        # Initialize weights according to normal distribution
        self.init_weights_normalized_normal(weight_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the datapoint through the network.

        Parameters
        ----------
        x : torch.Tensor
            The input of the model (i.e. the features).
        Returns
        -------
        x : torch.Tensor
            The output vector. Values should be in range (0,1).
        """
        # Passing the states through the consecutive linear (i.e. fully connected) layers
        x = self.lin_first(x)
        x = self.activation_f(x)

        for l in self.lin_middle_layers:
            x = l(x)
            x = self.activation_f(x)

        x = self.lin_last(x)
        x = torch.sigmoid(x)

        return x

    def init_weights_normalized_normal(self, weight_init_std: float):
        """
        Initialize the weights of this network according to the normal
        distribution, but with the std divided by the number of in channels.
        The biases are initialized to zero.

        Parameters
        ----------
        weight_init_std : float
            The standard deviation of the normal distribution.
        """

        def layer_weights_normal(m):
            """
            Apply initialization to a single layer.
            """
            classname = m.__class__.__name__
            # for every Linear layer in a model.
            if classname.find('Linear') != -1:
                std = weight_init_std / m.in_channels
                torch.nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(layer_weights_normal)
