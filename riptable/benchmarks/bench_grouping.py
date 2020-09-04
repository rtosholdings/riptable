"""
Benchmarks for 'grouping' functions.
"""
__all__ = ["bench_groupbyhash", "bench_groupbylex", "bench_groupbypack"]

import itertools
import logging
from typing import List

import numpy as np
from numpy.random import default_rng

from .benchmark import _timestamp_funcs
from .rand_data import rand_array, rand_fancyindex, rand_keyarray
from .runner import create_comparison_dataset, create_trial_dataset

from ..rt_dataset import Dataset
from ..rt_grouping import Grouping
from ..rt_numpy import empty, groupbyhash, groupbylex, groupbypack


logger = logging.getLogger(__name__)
"""The logger for this module."""

timestamper = _timestamp_funcs["get_nano_time"]
"""The timestamping function to use in benchmarks."""


def bench_groupbyhash(**kwargs) -> Dataset:
    # TODO: Add additional dimensions:
    #  * key dtype(s) -- use this to also bench with multiple keys
    #  * provide pcutoffs array?
    #  * number of threads
    #  * recycler on/off

    # Fixed parameters which apply to all of the trials in this benchmark.
    warmup_iters = 1
    iters = 5

    # Parameters we'll sweep over for the benchmark.
    rng_seeds = [12345]
    dtypes = [np.int16, np.int32, np.dtype("S4"), np.dtype("S10"), np.dtype("<U8")]
    key_unique_counts = [
        100,
        10_000,
    ]  # TODO: Maybe also sweep over constant (across rowcounts) key frequency, e.g. 0.1%, 1%, 10%, 20% unique keys -- this would work better for larger rowcounts
    rowcounts = [
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
    ]  # TODO: Add 1G and 2G -- these need to be optional since smaller machines will run out of memory
    use_hints = [False, True]

    setup_params = itertools.product(rng_seeds, dtypes, key_unique_counts, rowcounts)

    # Datasets containing timing data and parameters from the trials in this benchmark.
    benchmark_data: List[Dataset] = []

    for rng_seed, dtype, key_unique_count, rowcount in setup_params:
        #
        # Setup phase. The data here is used for both the warmup and the real, timed function invocations.
        #
        # Make sure to re-initialize the RNG each time so we get a repeatable result.
        rng = default_rng(rng_seed)

        key_data = rand_keyarray(
            rng, rowcount, dtype=np.dtype(dtype), unique_count=key_unique_count
        )

        # Sweep over other parameters that aren't required by the setup phase.
        other_params = use_hints  # itertools.product(use_hints, ...)
        for use_hint in other_params:
            # Allocate an array to hold the raw timing data.
            # TODO: Change to use TimeSpan?
            timing_data = empty(iters, dtype=np.int64)

            for is_warmup in (True, False):
                loop_count = warmup_iters if is_warmup else iters

                for i in range(loop_count):
                    start_time_ns = timestamper()

                    ### The actual function invocation ###
                    hint_size = key_unique_count if use_hint else 0
                    groupbyhash(key_data, hint_size=hint_size)

                    ### Store the timing results (if this was a real invocation).
                    call_nanos = timestamper() - start_time_ns
                    if not is_warmup:
                        timing_data[i] = call_nanos

            # Create a mini Dataset with the timing results for this run.
            # Capture the timing results along with the other options used for the function invocations.
            trial_data = create_trial_dataset(
                timing_data,
                {
                    # Setup parameters
                    "rng_seed": rng_seed,
                    "dtype": np.dtype(dtype).name,
                    "key_unique_count": key_unique_count,
                    "rowcount": rowcount,
                    # Other parameters
                    "use_hint": use_hint,
                },
            )
            benchmark_data.append(trial_data)

    # hstack all of the individual Datasets together into one large Dataset and return it.
    return Dataset.hstack(benchmark_data, destroy=True)


def bench_groupbylex(**kwargs) -> Dataset:
    # TODO: Add additional dimensions:
    #  * key dtype(s) -- use this to also bench with multiple keys
    #  * provide pcutoffs array?
    #  * number of threads
    #  * recycler on/off

    # Fixed parameters which apply to all of the trials in this benchmark.
    warmup_iters = 1
    iters = 5

    # Parameters we'll sweep over for the benchmark and are required for the setup phase.
    rng_seeds = [12345]
    dtypes = [np.int16, np.int32, np.dtype("S4"), np.dtype("S10"), np.dtype("<U8")]
    key_unique_counts = [
        100,
        10_000,
    ]  # TODO: Maybe also sweep over constant (across rowcounts) key frequency, e.g. 0.1%, 1%, 10%, 20% unique keys -- this would work better for larger rowcounts
    rowcounts = [
        10_000,
        100_000,
        1_000_000,
        10_000_000,
    ]  # , 100_000_000]   # TODO: Add 1G and 2G -- these need to be optional since smaller machines will run out of memory

    setup_params = itertools.product(rng_seeds, dtypes, key_unique_counts, rowcounts)

    # Parameters to sweep over for the benchmark which are *not* required for the setup phase.
    # TODO: Enable this parameter once we support multi-keys in this benchmark.
    as_recarrays = [
        False
    ]  # , True]     # Use to set the 'rec' parameter for groupbylex; only meaningful for multi-key

    # Datasets containing timing data and parameters from the trials in this benchmark.
    benchmark_data: List[Dataset] = []

    for rng_seed, dtype, key_unique_count, rowcount in setup_params:
        #
        # Setup phase. The data here is used for both the warmup and the real, timed function invocations.
        #
        # Make sure to re-initialize the RNG each time so we get a repeatable result.
        rng = default_rng(rng_seed)

        key_data = rand_keyarray(
            rng, rowcount, dtype=np.dtype(dtype), unique_count=key_unique_count
        )

        # Sweep over other parameters that aren't required by the setup phase.
        for as_recarray in as_recarrays:
            # Allocate an array to hold the raw timing data.
            # TODO: Change to use TimeSpan?
            timing_data = empty(iters, dtype=np.int64)

            for is_warmup in (True, False):
                loop_count = warmup_iters if is_warmup else iters

                for i in range(loop_count):
                    start_time_ns = timestamper()

                    ### The actual function invocation ###
                    groupbylex(key_data, rec=as_recarray)

                    ### Store the timing results (if this was a real invocation).
                    call_nanos = timestamper() - start_time_ns
                    if not is_warmup:
                        timing_data[i] = call_nanos

            # Create a mini Dataset with the timing results for this run.
            # Capture the timing results along with the other options used for the function invocations.
            trial_data = create_trial_dataset(
                timing_data,
                {
                    # Setup parameters
                    "rng_seed": rng_seed,
                    "dtype": np.dtype(dtype).name,
                    "key_unique_count": key_unique_count,
                    "rowcount": rowcount,
                    # Other parameters
                    "as_recarray": as_recarray,
                },
            )
            benchmark_data.append(trial_data)

    # hstack all of the individual Datasets together into one large Dataset and return it.
    return Dataset.hstack(benchmark_data, destroy=True)


def bench_groupbypack(**kwargs) -> Dataset:
    import riptide_cpp as rc

    # TODO: Add additional dimensions:
    #  * provide pcutoffs array?
    #  * number of threads
    #  * recycler on/off

    # Fixed parameters which apply to all of the trials in this benchmark.
    warmup_iters = 1
    iters = 5

    # Parameters we'll sweep over for the benchmark.
    rng_seeds = [12345]
    dtypes = [np.int16, np.int32, np.dtype("S4"), np.dtype("S10"), np.dtype("<U8")]
    key_unique_counts = [
        100,
        10_000,
    ]  # TODO: Maybe also sweep over constant (across rowcounts) key frequency, e.g. 0.1%, 1%, 10%, 20% unique keys -- this would work better for larger rowcounts
    rowcounts = [
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
    ]  # TODO: Add 1G and 2G -- these need to be optional since smaller machines will run out of memory

    setup_params = itertools.product(rng_seeds, dtypes, key_unique_counts, rowcounts)

    # Datasets containing timing data and parameters from the trials in this benchmark.
    benchmark_data: List[Dataset] = []

    for rng_seed, dtype, key_unique_count, rowcount in setup_params:
        #
        # Setup phase. The data here is used for both the warmup and the real, timed function invocations.
        #
        # Make sure to re-initialize the RNG each time so we get a repeatable result.
        rng = default_rng(rng_seed)

        key_data = rand_keyarray(
            rng, rowcount, dtype=np.dtype(dtype), unique_count=key_unique_count
        )
        gb_data = groupbyhash(key_data, hint_size=key_unique_count)

        # Sweep over other parameters that aren't required by the setup phase.
        # (There are no other parameters to sweep over for this benchmark.)
        for _ in [None]:  # Empty dimension for "other" parameters
            # Allocate an array to hold the raw timing data.
            # TODO: Change to use TimeSpan?
            timing_data = empty(iters, dtype=np.int64)

            for is_warmup in (True, False):
                loop_count = warmup_iters if is_warmup else iters

                for i in range(loop_count):
                    start_time_ns = timestamper()

                    ### The actual function invocation ###
                    ncountgroup = rc.BinCount(
                        gb_data["iKey"], gb_data["unique_count"] + 1
                    )
                    groupbypack(gb_data["iKey"], ncountgroup)

                    ### Store the timing results (if this was a real invocation).
                    call_nanos = timestamper() - start_time_ns
                    if not is_warmup:
                        timing_data[i] = call_nanos

            # Create a mini Dataset with the timing results for this run.
            # Capture the timing results along with the other options used for the function invocations.
            trial_data = create_trial_dataset(
                timing_data,
                {
                    # Setup parameters
                    "rng_seed": rng_seed,
                    "dtype": np.dtype(dtype).name,
                    "key_unique_count": key_unique_count,
                    "rowcount": rowcount,
                    # Other parameters
                    # (None)
                },
            )
            benchmark_data.append(trial_data)

    # hstack all of the individual Datasets together into one large Dataset and return it.
    return Dataset.hstack(benchmark_data, destroy=True)
