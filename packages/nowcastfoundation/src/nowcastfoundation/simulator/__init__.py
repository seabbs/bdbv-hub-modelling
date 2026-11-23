# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""The grid simulator of the fine-tuning simulations.

`simulate_batch` (`batch`) is its one entry point: the design and value
draws (`draws`), then each simulation as the joint PyRenew model
(`nowcastrenewal.simulation.outbreak`), writing the batch's files.
Run one batch from the command line with
`python -m nowcastfoundation.simulator GRID N SEED OUT`.
"""

from .batch import simulate_batch

__all__ = ["simulate_batch"]
