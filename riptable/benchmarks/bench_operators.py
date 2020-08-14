"""
Benchmarks for 'FastArray' operators (e.g. ``+``, ``*``, ``%``).
"""
__all__ = [
    'bench_op_add',
]

import itertools
import logging
import numpy as np

from numpy.random import default_rng
from itertools import product
from typing import List, Tuple

from .benchmark import timestamper
from .rand_data import rand_dataset
from .runner import create_comparison_dataset, create_trial_dataset, benchmark
from ..rt_categorical import Categorical
from ..rt_dataset import Dataset
from ..rt_numpy import empty


logger = logging.getLogger(__name__)
"""The logger for this module."""


def bench_op_add(**kwargs) -> Dataset:
    # Implement a benchmark that uses the __add__ operator on two FastArrays
    raise NotImplementedError()

# TODO: Implement benchmarks for other operators; for each one, need to try all applicable cases:
#   * ndarray vs. FastArray
#   * scalar OP array
#   * array OP scalar
#   * array OP array
#   * for each of the above, test various combinations of dtypes as well e.g. add an int16 array to an int64 array
