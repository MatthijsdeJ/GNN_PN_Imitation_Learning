#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a NN.

@author: Matthijs de Jong
"""

# Standard library imports
import sys

# Project imports
from training.training import Run
import auxiliary.config


def main():
    # Overwrite config with command line arguments
    auxiliary.config.parse_args_overwrite_config(sys.argv[1:])

    # Start the run
    r = Run()
    r.start()


if __name__ == "__main__":
    main()
