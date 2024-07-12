#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The dataloader for imitation learning.

@author: Matthijs de Jong
"""

# Standard library imports
import os
import json
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Any

# Third-party library imports
import numpy as np
import torch

# Project imports
import data_preprocessing_analysis.data_preprocessing as idp
from auxiliary.config import NetworkType, ModelType


class DataLoader:
    """
    Class for loading the dataset.
    """

    def __init__(self,
                 root: str,
                 matrix_cache_path: str,
                 feature_statistics_path: str,
                 device: torch.device,
                 model_type: ModelType,
                 network_type: Optional[NetworkType]):
        """
        Parameters
        ----------
        root : str
            The directory where the data files are located.
        matrix_cache_path : str
            The path of the matrix cache file.
        feature_statistics_path : str
            The path of the feature statistics file.
        device : torch.device
            What device to load the data on.
        network_type : NetworkType
            The type of network. Based on this, data needs to be delivered
            in a different format.
        """
        self._file_names = os.listdir(root)
        self._file_paths = [os.path.join(root, fn) for fn in self._file_names]
        with open(feature_statistics_path, 'r') as file:
            feature_statistics = json.loads(file.read())
        if model_type == ModelType.GCN:
            assert isinstance(network_type, NetworkType), 'Invalid network type'
            matrix_cache = idp.ConMatrixCache.load(matrix_cache_path)
            self.process_dp_strategy = ProcessDataPointGCN(device, feature_statistics, network_type, matrix_cache)
        elif model_type == ModelType.FCNN:
            self.process_dp_strategy = ProcessDataPointFCNN(device, feature_statistics)
        else:
            raise ValueError("Invalid model_type value.")

    def get_file_datapoints(self, idx: int) -> List[dict]:
        """
        Load the datapoints in a particular file. The file is indexed by an int, representing the index of the file in
        the list of file paths.

        Parameters
        ----------
        idx : int
            The index of the file in the list of file paths.

        Returns
        -------
        processed_datapoints : List[dict]
            The list of datapoints. Each datapoint is a dictionary.
        """
        # 'raw' is not fully true, as these datapoints should already have been preprocessed
        with open(self._file_paths[idx], 'r') as file:
            raw_datapoints = json.loads(file.read())

        processed_datapoints = []
        for raw_dp in raw_datapoints:
            # process datapoint and add it to the file list
            dp = self.process_dp_strategy.process_datapoint(raw_dp)
            processed_datapoints.append(dp)

        return processed_datapoints

    def __iter__(self, shuffle: bool = True) -> dict:
        """
        Iterate over the datapoints.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle the data files. Does NOT mean that the dps
            in the files are also shuffled. The default is True.

        Yields
        ------
        dp : dict
            The datapoint.

        """
        file_idxs = list(range(len(self._file_paths)))
        if shuffle:
            random.shuffle(file_idxs)

        for idx in file_idxs:
            datapoints = self.get_file_datapoints(idx)
            for dp in datapoints:
                yield dp


class ProcessDataPointStrategy(ABC):
    """
    Abstract base class for strategies of processing a single datapoint.
    Exists because different models require different information from a datapoint,
    and hence require different processing.
    """

    def __init__(self,
                 device: torch.device,
                 feature_statistics: dict):
        """
        Parameters
        ----------
        device : torch.device
            The device to set torch tensors to.
        feature_statistics : dict
            Dictionary with information (mean, std) used to normalize features.
        """
        self.device = device
        self.feature_statistics = feature_statistics

    @abstractmethod
    def process_datapoint(self, raw_dp: int):
        """
        Process a single datapoint, from raw_dp to dp.

        Parameters
        ----------
        raw_dp : dict
            The 'raw' datapoint (not fully true; these raw datapoints should be preprocessed already) from
            which information is extracted.

        Returns
        -------
        dp : dict
            The resulting datapoint.
        """
        pass


class ProcessDataPointGCN(ProcessDataPointStrategy):
    """
    Process a datapoint to obtain the information used by, and in the format used by, a GCN.
    """

    def __init__(self,
                 device: torch.device,
                 feature_statistics: dict,
                 network_type: NetworkType,
                 matrix_cache: idp.ConMatrixCache):
        """
        Parameters
        ----------
        device : torch.device
            The device to set torch tensors to.
        feature_statistics : dict
            Dictionary with information (mean, std) used to normalize features.
        network_type : NetworkType
            The type of the GCN network.
        """
        super().__init__(device, feature_statistics)
        self.network_type = network_type
        self.matrix_cache = matrix_cache

    def process_datapoint(self, raw_dp: dict):
        """
        Process a single datapoint, from raw_dp to dp, with the information and formatting for a GCN model.

        Parameters
        ----------
        raw_dp : dict
            The 'raw' datapoint (not fully true; these raw datapoints should be preprocessed already) from
            which information is extracted.

        Returns
        -------
        dp : dict
            The resulting datapoint.
        """
        dp_reduced = raw_dp['reduced']
        dp: dict[str, Any] = {}

        # Add the label
        dp['full_change_topo_vect'] = torch.tensor(raw_dp['full']['change_topo_vect'],
                                                   device=self.device, dtype=torch.float)

        # Create the position topology vector, which relates the objects ordered by type to their position in
        # the topology vector
        dp['reduced_pos_topo_vect'] = dp_reduced['pos_topo_vect']

        # Load the sub info array, which contains info about to which substation each object belongs
        dp['reduced_sub_info'] = dp_reduced['sub_info']

        # Load, normalize features, turn them into tensors
        fstats = self.feature_statistics
        norm_gen_features = (np.array(raw_dp['gen_features']) - fstats['gen']['mean']) / fstats['gen']['std']
        dp['gen_features'] = torch.tensor(norm_gen_features, device=self.device, dtype=torch.float)
        norm_load_features = (np.array(raw_dp['load_features']) - fstats['load']['mean']) / fstats['load']['std']
        dp['load_features'] = torch.tensor(norm_load_features, device=self.device, dtype=torch.float)
        norm_or_features = (np.array(dp_reduced['or_features']) - fstats['line']['mean']) / fstats['line']['std']
        dp['reduced_or_features'] = torch.tensor(norm_or_features, device=self.device, dtype=torch.float)
        norm_ex_features = (np.array(dp_reduced['ex_features']) - fstats['line']['mean']) / fstats['line']['std']
        dp['reduced_ex_features'] = torch.tensor(norm_ex_features, device=self.device, dtype=torch.float)

        # Load the connectivity matrix, combine the edges for the specified network type
        con_matrix_hash = str(dp_reduced['cm_index'])
        same_busbar_e, other_busbar_e, line_e = self.matrix_cache.con_matrices[con_matrix_hash][1]
        if self.network_type == NetworkType.HOMO:
            # noinspection PyTypeChecker
            dp['edges'] = torch.tensor(np.append(same_busbar_e, line_e, axis=1), device=self.device, dtype=torch.long)
        elif self.network_type == NetworkType.HETERO:
            dp['edges'] = {('object', 'line', 'object'):
                           torch.tensor(line_e, device=self.device, dtype=torch.long),
                           ('object', 'same_busbar', 'object'):
                           torch.tensor(same_busbar_e, device=self.device, dtype=torch.long),
                           ('object', 'other_busbar', 'object'):
                           torch.tensor(other_busbar_e, device=self.device, dtype=torch.long)}

        # Load line_disabled, line_or_pos, line_ex_pos, and topo_vect
        dp['line_disabled'] = raw_dp['line_disabled']
        if dp['line_disabled'] != -1:
            dp['disabled_or_pos'] = raw_dp['full']['disabled_or_pos']
            dp['disabled_ex_pos'] = raw_dp['full']['disabled_ex_pos']
        dp['reduced_topo_vect'] = torch.tensor(dp_reduced['topo_vect'], device=self.device, dtype=torch.long)
        dp['full_topo_vect'] = torch.tensor(raw_dp['full']['topo_vect'], device=self.device, dtype=torch.long)

        return dp


class ProcessDataPointFCNN(ProcessDataPointStrategy):
    """
    Process a datapoint to obtain the information used by, and in the format used by, a FCNN.
    """

    def __init__(self,
                 device: torch.device,
                 feature_statistics: dict):
        """
        Parameters
        ----------
        device : torch.device
            The device to set torch tensors to.
        feature_statistics : dict
            Dictionary with information (mean, std) used to normalize features.
        """
        super().__init__(device, feature_statistics)

    def process_datapoint(self, raw_dp: dict):
        """
        Process a single datapoint, from raw_dp to dp, with the information and formatting for a FCNN model.

        Parameters
        ----------
        raw_dp : dict
            The 'raw' datapoint (not fully true; these raw datapoints should be preprocessed already) from
            which information is extracted.

        Returns
        -------
        dp : dict
            The resulting datapoint.
        """
        dp_full = raw_dp['full']
        dp: dict[str, Any] = {}

        # Add the label
        dp['full_change_topo_vect'] = torch.tensor(dp_full['change_topo_vect'], device=self.device, dtype=torch.float)

        # Load, normalize features (including the topology vector), turn them into a single tensor
        fstats = self.feature_statistics
        norm_gen_features = (np.array(raw_dp['gen_features']) - fstats['gen']['mean']) / fstats['gen']['std']
        norm_load_features = (np.array(raw_dp['load_features']) - fstats['load']['mean']) / fstats['load']['std']
        norm_or_features = (np.array(dp_full['or_features']) - fstats['line']['mean']) / fstats['line']['std']
        norm_ex_features = (np.array(dp_full['ex_features']) - fstats['line']['mean']) / fstats['line']['std']
        full_topo_vect = dp_full['topo_vect']
        dp['features'] = torch.tensor(np.concatenate((norm_gen_features.flatten(),
                                                      norm_load_features.flatten(),
                                                      norm_or_features.flatten(),
                                                      norm_ex_features.flatten(),
                                                      full_topo_vect)),
                                      device=self.device, dtype=torch.float)

        dp['line_disabled'] = raw_dp['line_disabled']
        if dp['line_disabled'] != -1:
            dp['disabled_or_pos'] = dp_full['disabled_or_pos']
            dp['disabled_ex_pos'] = dp_full['disabled_ex_pos']
        dp['full_topo_vect'] = torch.tensor(full_topo_vect, device=self.device, dtype=torch.long)

        return dp
