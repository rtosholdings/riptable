"""
Benchmarks for 'merge' functions.
"""
__all__ = [
    "bench_merge",
    "bench_merge2",
    "bench_merge2_with_settings",
    "bench_merge_pandas",
    "compare_merge",
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
from ..rt_merge import merge, merge2
from ..rt_numpy import empty


logger = logging.getLogger(__name__)
"""The logger for this module."""


def _range_slice(s: slice, endpoint: bool = True) -> List[int]:
    """
    Create a list of integers as specified by the slice.

    Parameters
    ----------
    s : slice
    endpoint : bool

    Returns
    -------
    slice_indices : list of int

    Examples
    --------
    >>> _range_slice(slice(200, 500, 50))
    [200,
     250,
     300,
     350,
     400,
     450,
     500]
    """
    # If the caller wants the right endpoint to be included, adjust
    # the 'stop' value by 1 (or -1) so the .indices() method includes it.
    if endpoint:
        adjust = 1 if s.step >= 0 else -1
        s = slice(s.start, s.stop + adjust, s.step)
    return list(range(*(s.indices(s.stop))))


def generate_merge_datasets(
    rng_seed=12345,
    left_key_unique_count=25,
    right_key_unique_count=2500,
    left_dataset_max_rowcount=250_000,
    right_dataset_max_rowcount=62_500,
) -> Tuple[Dataset, Dataset]:
    """Generates the left and right Datasets for merge benchmarking."""
    left_step = int(left_dataset_max_rowcount / 4)
    left_rowcounts = list(range(left_step, left_dataset_max_rowcount + 1, left_step))
    right_step = int(right_dataset_max_rowcount / 4)
    right_rowcounts = list(
        range(right_step, right_dataset_max_rowcount + 1, right_step)
    )

    left_datasets, right_datasets = list(), list()
    for left_rowcount in left_rowcounts:
        for right_rowcount in right_rowcounts:
            rng = default_rng(rng_seed)
            left_datasets.append(
                rand_dataset(left_rowcount, rng, left_key_unique_count)
            )
            right_datasets.append(
                rand_dataset(right_rowcount, rng, right_key_unique_count)
            )
    return left_datasets, right_datasets


def bench_merge(**kwargs) -> Dataset:
    # TODO: Add additional dimensions:
    #  * key is Cat [True, False]
    #  * key dtype(s) -- use this to also bench with multiple keys
    #  * number of threads
    #  * recycler on/off
    #  * number of additional (non-key) columns in the left and/or right (just make them all the same dtype, it doesn't matter much for this benchmark)
    #  * (advanced) different key distributions
    #  * (advanced) key clustering (i.e. are keys already more or less occurring near each other, or are they all spread out?)
    #  * (advanced) key dispersion -- if the unique keys are created like arange(1, 100), does grouping/merge go any
    #       faster than if the keys are created like arange(1, 1000, 10) or arange(1, 1_000_000_000, 10_000_000_000)?
    #       Make sure to control for dtype to ensure that's the same for all cases tested here.
    #       This could be used to diagnose issues with hashing/grouping implementations.

    # Fixed parameters which apply to all of the trials in this benchmark.
    warmup_iters = 1
    iters = 5

    # Setup parameters.
    rng_seeds = [12345]
    left_key_unique_counts = [100]
    right_key_unique_counts = [10000]
    left_rowcounts = _range_slice(slice(500_000, 2_000_000, 500_000))
    right_rowcounts = _range_slice(slice(250_000, 500_000, 250_000))

    setup_params = itertools.product(
        rng_seeds,
        left_key_unique_counts,
        right_key_unique_counts,
        left_rowcounts,
        right_rowcounts,
    )

    # Trial parameters.
    hows = ["left", "right", "inner"]  # TODO: Add 'outer'; seems to be broken though?
    trial_params = hows

    # Datasets containing timing data and parameters from the trials in this benchmark.
    benchmark_data: List[Dataset] = []

    for (
        rng_seed,
        left_key_unique_count,
        right_key_unique_count,
        left_rowcount,
        right_rowcount,
    ) in setup_params:
        #
        # Setup phase. The data here is used for both the warmup and the real, timed function invocations.
        #
        # Make sure to re-initialize the RNG each time so we get a repeatable result.
        rng = default_rng(rng_seed)

        left_ds = rand_dataset(left_rowcount, rng, left_key_unique_count)
        right_ds = rand_dataset(right_rowcount, rng, right_key_unique_count)

        # Sweep over trial parameters; all trials sharing the same setup parameters
        # share the same data created from those setup parameters.
        for how in trial_params:
            # Allocate an array to hold the raw timing data.
            # TODO: Change to use TimeSpan?
            timing_data = empty(iters, dtype=np.int64)

            for is_warmup in (True, False):
                loop_count = warmup_iters if is_warmup else iters
                logger.info(
                    f"bench_merge:\tmode={'warmup' if is_warmup else 'bench'}\tloops={loop_count}\tleft_rowcount={left_rowcount}\tright_rowcount={right_rowcount}\thow={how}"
                )

                for i in range(loop_count):
                    start_time_ns = timestamper()

                    ### The actual function invocation ###
                    # TODO: This could be passed in as a lambda, and we call it with the data constructed
                    #  in the setup phase + any "parameters" that weren't used in the setup phase; or for
                    #  simplicity just pass all the parameters + the data constructed in the setup phase,
                    #  the lambda doesn't need to use all of it.

                    merge(left_ds, right_ds, on="key", how=how)

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
                    "left_key_unique_count": left_key_unique_count,
                    "right_key_unique_count": right_key_unique_count,
                    "left_rowcount": left_rowcount,
                    "right_rowcount": right_rowcount,
                    # Trial parameters
                    "how": how,
                },
            )
            benchmark_data.append(trial_data)

    # hstack all of the individual Datasets together into one large Dataset and return it.
    return Dataset.hstack(benchmark_data, destroy=True)


def bench_merge2(**kwargs) -> Dataset:
    # TODO: Add additional dimensions:
    #  * key is Cat [True, False]
    #  * key dtype(s) -- use this to also bench with multiple keys
    #  * number of threads
    #  * recycler on/off
    #  * number of additional (non-key) columns in the left and/or right (just make them all the same dtype, it doesn't matter much for this benchmark)
    #  * (advanced) different key distributions
    #  * (advanced) key clustering (i.e. are keys already more or less occurring near each other, or are they all spread out?)
    #  * (advanced) key dispersion -- if the unique keys are created like arange(1, 100), does grouping/merge go any
    #       faster than if the keys are created like arange(1, 1000, 10) or arange(1, 1_000_000_000, 10_000_000_000)?
    #       Make sure to control for dtype to ensure that's the same for all cases tested here.
    #       This could be used to diagnose issues with hashing/grouping implementations.

    # Fixed parameters which apply to all of the trials in this benchmark.
    warmup_iters = 1
    iters = 5

    # Setup parameters.
    rng_seeds = [12345]
    left_key_unique_counts = [100]
    right_key_unique_counts = [10000]
    left_rowcounts = _range_slice(
        slice(500_000, 2_000_000, 500_000)
    )  # TODO: For larger-memory machines, we could use a larger max here, e.g. 7M
    right_rowcounts = _range_slice(
        slice(250_000, 500_000, 250_000)
    )  # TODO: For larger-memory machines, we could use a larger max here, e.g. 5M

    setup_params = itertools.product(
        rng_seeds,
        left_key_unique_counts,
        right_key_unique_counts,
        left_rowcounts,
        right_rowcounts,
    )

    # Trial parameters.
    hows = [
        "left",
        "right",
        "inner",
    ]  # TODO: Add 'outer'; skip 'right', that's the same logic as left
    left_keeps = [None, "first", "last"]
    right_keeps = [None, "first", "last"]

    trial_params = lambda: itertools.product(hows, left_keeps, right_keeps)

    # Datasets containing timing data and parameters from the trials in this benchmark.
    benchmark_data: List[Dataset] = []

    for (
        rng_seed,
        left_key_unique_count,
        right_key_unique_count,
        left_rowcount,
        right_rowcount,
    ) in setup_params:
        #
        # Setup phase. The data here is used for both the warmup and the real, timed function invocations.
        #
        # Make sure to re-initialize the RNG each time so we get a repeatable result.
        rng = default_rng(rng_seed)

        left_ds = rand_dataset(left_rowcount, rng, left_key_unique_count)
        right_ds = rand_dataset(right_rowcount, rng, right_key_unique_count)

        # Sweep over trial parameters; all trials sharing the same setup parameters
        # share the same data created from those setup parameters.
        for how, keep_left, keep_right in trial_params():
            # Allocate an array to hold the raw timing data.
            # TODO: Change to use TimeSpan?
            timing_data = empty(iters, dtype=np.int64)

            for is_warmup in (True, False):
                loop_count = warmup_iters if is_warmup else iters
                logger.info(
                    f"bench_merge2:\tmode={'warmup' if is_warmup else 'bench'}\tloops={loop_count}\tleft_rowcount={left_rowcount}\tright_rowcount={right_rowcount}\thow={how}\tkeep_left={keep_left}\tkeep_right={keep_right}"
                )

                for i in range(loop_count):
                    start_time_ns = timestamper()

                    ### The actual function invocation ###
                    # TODO: This could be passed in as a lambda, and we call it with the data constructed
                    #  in the setup phase + any "parameters" that weren't used in the setup phase; or for
                    #  simplicity just pass all the parameters + the data constructed in the setup phase,
                    #  the lambda doesn't need to use all of it.
                    keep = keep_left, keep_right
                    merge2(
                        left_ds,
                        right_ds,
                        on="key",
                        how=how,
                        keep=keep,
                        suffixes=("_x", "_y"),
                    )

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
                    "left_key_unique_count": left_key_unique_count,
                    "right_key_unique_count": right_key_unique_count,
                    "left_rowcount": left_rowcount,
                    "right_rowcount": right_rowcount,
                    # Trial parameters
                    "how": how,
                    "keep_left": "" if keep_left is None else keep_left,
                    "keep_right": "" if keep_right is None else keep_right,
                },
            )
            benchmark_data.append(trial_data)

    # hstack all of the individual Datasets together into one large Dataset and return it.
    return Dataset.hstack(benchmark_data, destroy=True)


def bench_merge2_with_settings():
    # Define the dimensions (and their values) we want to sweep over for the benchmarking
    how_options = [
        "left",
        "right",
        "inner",
    ]  # TODO: Add 'outer'; skip 'right', that's the same logic as left
    keeps = [None, "first", "last"]
    keep_options = list(product(keeps, keeps))
    left_datasets, right_datasets = generate_merge_datasets()
    warmup_iters, benchmark_iters = 1, 5

    @benchmark(
        benchmark_params={
            # _merge2 parameters
            "left_ds": left_datasets,
            "right_ds": right_datasets,
            "on": ["key"],
            "how": how_options,
            "keeps": keep_options,
            # riptable settings
            "thread_count": [1, 2, 4, 8],
            "recycler": [True, False],
        },
        # benchmark options
        warmup_iterations=warmup_iters,
        benchmark_iterations=benchmark_iters,
        enable_bench_estimators=True,  # TODO
    )
    def bench_merge2(left_ds, right_ds, on, how, keeps):
        merge2(left_ds, right_ds, on=on, how=how, keep=keeps, suffixes=("_x", "_y"))

    return bench_merge2()


def bench_merge_pandas(**kwargs) -> Dataset:
    """Merge benchmark for pandas."""
    import pandas as pd

    # Fixed parameters which apply to all of the trials in this benchmark.
    warmup_iters = 1
    iters = 5

    # Setup parameters.
    rng_seeds = [12345]
    left_key_unique_counts = [100]
    right_key_unique_counts = [10000]
    left_rowcounts = _range_slice(
        slice(500_000, 2_000_000, 500_000)
    )  # TODO: For larger-memory machines, we could use a larger max here, e.g. 7M
    right_rowcounts = _range_slice(
        slice(250_000, 500_000, 250_000)
    )  # TODO: For larger-memory machines, we could use a larger max here, e.g. 5M

    setup_params = itertools.product(
        rng_seeds,
        left_key_unique_counts,
        right_key_unique_counts,
        left_rowcounts,
        right_rowcounts,
    )

    # Trial parameters.
    hows = [
        "left",
        "right",
        "inner",
    ]  # TODO: Add 'outer' once we add it for merge and merge2.
    left_keeps = [None, "first", "last"]
    right_keeps = [None, "first", "last"]

    trial_params = lambda: itertools.product(hows, left_keeps, right_keeps)

    # Datasets containing timing data and parameters from the trials in this benchmark.
    benchmark_data: List[Dataset] = []

    for (
        rng_seed,
        left_key_unique_count,
        right_key_unique_count,
        left_rowcount,
        right_rowcount,
    ) in setup_params:
        #
        # Setup phase. The data here is used for both the warmup and the real, timed function invocations.
        #
        # Make sure to re-initialize the RNG each time so we get a repeatable result.
        rng = default_rng(rng_seed)

        left_df = rand_dataset(left_rowcount, rng, left_key_unique_count).to_pandas(
            use_nullable=True
        )
        right_df = rand_dataset(right_rowcount, rng, right_key_unique_count).to_pandas(
            use_nullable=True
        )

        # Sweep over trial parameters; all trials sharing the same setup parameters
        # share the same data created from those setup parameters.
        for how, keep_left, keep_right in trial_params():
            # Allocate an array to hold the raw timing data.
            # TODO: Change to use TimeSpan?
            timing_data = empty(iters, dtype=np.int64)

            for is_warmup in (True, False):
                loop_count = warmup_iters if is_warmup else iters

                for i in range(loop_count):
                    start_time_ns = timestamper()

                    ### The actual function invocation ###
                    # Pandas merge doesn't support 'keep' (yet: see https://github.com/pandas-dev/pandas/issues/31332 ),
                    # so we need to do it manually with drop_duplicates first.
                    if keep_left is not None:
                        left_df = left_df.drop_duplicates(subset="key", keep=keep_left)
                    if keep_right is not None:
                        right_df = right_df.drop_duplicates(
                            subset="key", keep=keep_right
                        )

                    left_df.merge(right_df, on="key", how=how)
                    ### Actual function invocation ends here ###

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
                    "left_key_unique_count": left_key_unique_count,
                    "right_key_unique_count": right_key_unique_count,
                    "left_rowcount": left_rowcount,
                    "right_rowcount": right_rowcount,
                    # Trial parameters
                    "how": how,
                    "keep_left": "" if keep_left is None else keep_left,
                    "keep_right": "" if keep_right is None else keep_right,
                },
            )
            benchmark_data.append(trial_data)

    # hstack all of the individual Datasets together into one large Dataset and return it.
    return Dataset.hstack(benchmark_data, destroy=True)


def compare_merge(**kwargs) -> Dataset:
    """Run all 'merge' benchmarks and return the combined results."""

    # Run the benchmark for rt.merge.
    rtmerge_bench_data = bench_merge(**kwargs)

    # Using the 'how' column and our knowledge of how rt.merge operates,
    # synthesize 'keep_left' and 'keep_right' Categoricals for the rt.merge
    # results so they can be better compared to the other implementations.
    # TODO: This could be removed in the future once we support jagged/non-rectangular
    #       benchmark parameters, since we can pass these in to the benchmark
    #       and have them recorded in the usual way.
    rtmerge_bench_data["keep_left"] = Categorical(
        rtmerge_bench_data["how"].map(
            {"right": "last", "inner": "last", "outer": "last"}, invalid=""
        )
    )
    rtmerge_bench_data["keep_right"] = Categorical(
        rtmerge_bench_data["how"].map(
            {"left": "last", "inner": "last", "outer": "last"}, invalid=""
        )
    )

    # Run the benchmark for rt.merge2.
    rtmerge2_bench_data = bench_merge2(**kwargs)

    # Run the benchmark for pandas.
    pdmerge_bench_data = bench_merge_pandas(**kwargs)

    # Combine the results for the different implementations into
    # a single Dataset and return it.
    return create_comparison_dataset(
        {
            "rt_merge": rtmerge_bench_data,
            "rt_merge2": rtmerge2_bench_data,
            "pandas": pdmerge_bench_data,
        },
        destroy=True,
    )
