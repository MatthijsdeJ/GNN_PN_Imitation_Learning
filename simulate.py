"""
Script to simulate an agent on the power grid.

@author: Matthijs de Jong
"""
# Standard library imports
import sys

# Project imports
import auxiliary.config
import simulation.simulation as simulation


def main():
    # Overwrite config with command line arguments
    auxiliary.config.parse_args_overwrite_config(sys.argv[1:])

    # Start simulation
    simulation.simulate()


if __name__ == "__main__":
    main()
