#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The util file for this project.

@author: Matthijs de Jong
"""
# Standard library imports
import numpy as np
import os
from typing import List, Sequence, Callable
import json


def set_wd_to_package_root():
    """
    Set the working directory to the root of the package.
    """
    os.chdir(os.path.dirname(__file__) + '/..')


def flatten(t: Sequence[Sequence]) -> List:
    """
    Flatten a sequence of sequences.

    Parameters
    ----------
    t : Sequence[Sequence]
        The sequence of sequences to flatten.

    Returns
    -------
    List
        Flattened sequence.
    """
    return [item for sublist in t for item in sublist]


def hash_nparray(arr: np.array) -> int:
    """
    Hashes a numpy array.

    Parameters
    ----------
    arr : np.array
        The array.

    Returns
    -------
    int
        The hash value.
    """
    return hash(arr.data.tobytes())


class NumpyEncoder(json.JSONEncoder):
    """
    Class that can be used in json.dump() to encode np.array objects.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def argmax_f(s: Sequence, f: Callable) -> int:
    """
    Take the argmax (i.e. the index) based on the maximum of a particular
    function.

    Parameters
    ----------
    s : Sequence
        The sequence to find the argmax of.
    f : Callable
        The function to apply to the elements

    Returns
    -------
    int
        The index produced by the argmax.

    """
    return max(enumerate([f(d) for d in s]),
               key=lambda x: x[1])[0]
