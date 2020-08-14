# -*- coding: utf-8 -*-
"""
Benchmarks for primitive/low-level array operations.
"""
__all__ = [
    "bench_astype",
    "bench_astype_numba",
    "bench_astype_numpy",
    "bench_bool_index",
    "bench_bool_index_numpy",
    "bench_compare_ops",
    "bench_compare_ops_numpy",
    "bench_mbget",
    "bench_mbget_numba",
    # comparisons
    "compare_astype",
    "compare_bool_index",
    "compare_compare_ops",
    "compare_mbget",
]

import itertools
import logging
import operator
from typing import List

import numpy as np
from numpy.random import default_rng
import numba as nb
from .benchmark import _timestamp_funcs
from .rand_data import rand_array, rand_fancyindex
from .runner import create_comparison_dataset, create_trial_dataset
from ..rt_enum import TypeRegister, NumpyCharTypes
from ..rt_dataset import Dataset
from ..rt_numpy import empty
from ..rt_utils import mbget, _mbget_2dims #, mbget_numba


logger = logging.getLogger(__name__)
"""The logger for this module."""

timestamper = _timestamp_funcs["get_nano_time"]
"""The timestamping function to use in benchmarks."""

# TODO: Additional benchmarks which would be useful for riptable development and comparison to other frameworks:
#   * mbget vs. numpy fancy indexing (only on valid array indices -- -len(arr) <= x < len(arr))
#   * mbget vs. numba-based equivalent to look for compiled code optimizations + thread scaling
#   * indexing with a boolean mask (riptable vs. numba)
#   * array conversion (i.e. elementwise type conversion) (arr1.astype(np.float32))
#       * make sure to include the self-conversion case so that we look for optimizations there (like just calling memcpy)
#   * equality and comparisons
#       * elementwise array equality (arr1 == arr2)
#       * array vs. scalar equality (arr == 123, arr == "foo", arr != '', etc.)
#       * elementwise array comparison (arr1 < arr2)
#       * array vs. scalar comparison (arr1 < 1.23, arr > 123, etc.)
#       * it would also be useful (for demonstration purposes) to demo here how much faster these operations
#         are on a string categorical compared to a normal array of strings (like the Categorical's .expand_array).
#   * conversion-assignment, e.g. result[:] = arr[:]


def mbget_numba(aValues, aIndex) -> np.ndarray:
    """
    Re-implementation of the 'mbget' fancy-indexing function with numba, for comparison with the riptide_cpp implementation.

    Parameters
    ----------
    aValues
    aIndex

    Returns
    -------

    """
    # make sure a aValues and aIndex are both numpy arrays
    if isinstance(aValues, (list, tuple)):
        aValues = TypeRegister.FastArray(aValues)

    if isinstance(aIndex, (list, tuple)):
        aIndex = TypeRegister.FastArray(aIndex)

    if not isinstance(aValues, np.ndarray) or not isinstance(aIndex, np.ndarray):
        raise TypeError(f"Values and index must be numpy arrays. Got {type(aValues)} {type(aIndex)}")

    elif aValues.dtype.char == 'O':
        raise TypeError(f"mbget does not support object types")

    elif aIndex.dtype.char not in NumpyCharTypes.AllInteger:
        raise TypeError(f"indices provided to mbget must be an integer type not {aIndex.dtype}")

    if aValues.ndim == 2:
        return _mbget_2dims(aValues, aIndex)

    # TODO: probably need special code or parameter to set custom default value for NAN_TIME
    if aValues.dtype.char in NumpyCharTypes.AllInteger + NumpyCharTypes.AllFloat:
        result = _mbget_numeric(aValues, aIndex)
    elif aValues.dtype.char in "SU":
        result = _mbget_string(aValues, aIndex)
    else:
        raise Exception(f"mbget can't operate on an array of this type: {aValues.dtype}")

    result = TypeRegister.newclassfrominstance(result, aValues)

    return result


def _mbget_numeric(aValues, aIndex) -> np.ndarray:
    result = empty(len(aIndex), dtype=aValues.dtype)

    # Choose different implementation for signed vs. unsigned dtype.
    # See comment in mbget_string for details.
    _mbget_numeric_impl = _mbget_numeric_unsigned_impl if aIndex.dtype.kind == 'u' else _mbget_numeric_signed_impl
    _mbget_numeric_impl(aValues, aIndex, result, result.inv)
    return result


@nb.njit(cache=True, parallel=True, nogil=True)
def _mbget_numeric_signed_impl(aValues, aIndex, result, default_val):
    num_elmnts = len(aValues)
    for i in nb.prange(aIndex.shape[0]):
        # This has one less branch (in the code) than the riptide_cpp implementation of mbget,
        # because numba handles the negative/wraparound indexing for us. So the conditional logic
        # to handle the negative indexing is still there; it may or may not be in the generated
        # machine code depending on how numba chooses to generate it.
        index = aIndex[i]
        result[i] = aValues[index] if -num_elmnts <= index < num_elmnts else default_val


@nb.njit(cache=True, parallel=True, nogil=True)
def _mbget_numeric_unsigned_impl(aValues, aIndex, result, default_val):
    num_elmnts = len(aValues)
    for i in nb.prange(aIndex.shape[0]):
        # This has one less branch (in the code) than the riptide_cpp implementation of mbget,
        # because numba handles the negative/wraparound indexing for us. So the conditional logic
        # to handle the negative indexing is still there; it may or may not be in the generated
        # machine code depending on how numba chooses to generate it.
        index = aIndex[i]
        result[i] = aValues[index] if index < num_elmnts else default_val


#not using a default value here since we're handling byte strings only (default val. is 0)
def _mbget_string(aValues, aIndex) -> np.ndarray:
    result = empty(len(aIndex), dtype=aValues.dtype)
    itemsize = aValues.dtype.itemsize // 1  # ASCII

    # Choose different implementation for signed vs. unsigned dtype.
    # This is both for performance reasons and also because if we try to use the signed implementation
    # with unsigned integer types, numba ends up doing extra/unnecessary conversions so the performance
    # is poor; for that same reason, numba fails with an error on the uint64 type since it tries to cast
    # the index value to a float before we use it as an array index (which isn't allowed).
    # TODO: This decision could probably be pushed into the numba JIT-specialized generic so we don't need to choose here?
    _mbget_string_impl = _mbget_string_unsigned_impl if aIndex.dtype.kind == 'u' else _mbget_string_signed_impl
    _mbget_string_impl(aValues.view(np.uint8), aIndex, result.view(np.uint8), itemsize)
    return result


@nb.njit(cache=True, parallel=True, nogil=True)
def _mbget_string_signed_impl(aValues, aIndex, result, itemsize):  # byte array
    numstrings = aValues.shape[0] // itemsize
    for i in nb.prange(aIndex.shape[0]):
        index = aIndex[i]
        if -numstrings <= index < numstrings:
            str_idx = index if index >= 0 else numstrings + aIndex[i]
            for j in range(itemsize):
                result[itemsize * i + j] = aValues[itemsize * str_idx + j]

        else:
            for j in range(itemsize):
                result[itemsize * i + j] = 0


@nb.njit(cache=True, parallel=True, nogil=True)
def _mbget_string_unsigned_impl(aValues, aIndex, result, itemsize):  # byte array
    numstrings = aValues.shape[0] // itemsize
    for i in nb.prange(aIndex.shape[0]):
        index = aIndex[i]
        if index < numstrings:
            for j in range(itemsize):
                result[itemsize * i + j] = aValues[itemsize * index + j]

        else:
            for j in range(itemsize):
                result[itemsize * i + j] = 0


def astype_numba(arr, dst_dtype):
    #only supports numeric-to-numeric type conversions
    if arr.dtype.char in "SU" or dst_dtype.char in "SU":
        raise Exception (f"Only numeric-to-numeric type conversions are supported.")
    result = empty(arr.shape[0], dtype=dst_dtype)
    _astype_numba(arr, result)
    return result


# numba seems to emit poor quality code for this simple loop, and the performance is
# massively worse when parallel=True is specified. (Tested with numba 0.48, 0.50.1)
# Manually splitting the loop so the input data is chunked does not improve the performance either.
@nb.njit(cache=True, parallel=False, nogil=True)
def _astype_numba(arr, result):
    for i in nb.prange(len(arr)):
        # conversion occurs implicitly, and numba only supports conversion
        # between arrays of numeric types.
        result[i] = arr[i]


def bench_bool_index(**kwargs) -> Dataset:
    warmup_iters = 0
    iters = 21
    rng_seeds = [12345]
    data_dtypes = [
        np.int16,
        np.int32,
        np.float64,
        # TODO: Enable these additional data types; they're somewhat slow though, so we'd only want to
        #       run them under a 'detailed' / 'long-running' scenario
        # np.dtype('S4'),
        # np.dtype('S10'),
        # np.dtype('<U8')
    ]
    data_lengths = [
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        # TODO: Add 100M, 1G and 2G -- these need to be optional since smaller machines will run out of memory
        #       and also take longer than typical trials
    ]
    true_ratios = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0
    ]

    setup_params = itertools.product(
        rng_seeds,
        data_dtypes,
        data_lengths,
        true_ratios
    )

    # Datasets containing timing data and parameters from the trials in this benchmark.
    benchmark_data: List[Dataset] = []

    for (
            rng_seed,
            data_dtype,
            data_length,
            true_ratio,
    ) in setup_params:
        # HACK: Until we have a better approach for supporting non-rectangular parameter spaces,
        #       or otherwise being able to skip certain combinations of parameters (e.g. because
        #       they're invalid, non-realistic, or otherwise don't make sense).
        # if np.iinfo(index_dtype).max < data_length:
        #     continue

        #
        # Setup phase. The data here is used for both the warmup and the real, timed function invocations.
        #
        # Make sure to re-initialize the RNG each time so we get a repeatable result.
        rng = default_rng(rng_seed)

        data_array = rand_array(rng, data_length, dtype=np.dtype(data_dtype))
        mask = rng.random(data_length) < true_ratio

        # Sweep over other parameters that aren't required by the setup phase.
        other_params = [None]
        for _ in other_params:
            # Allocate an array to hold the raw timing data.
            # TODO: Change to use TimeSpan?
            timing_data = empty(iters, dtype=np.int64)

            for is_warmup in (True, False):
                loop_count = warmup_iters if is_warmup else iters

                for i in range(loop_count):
                    start_time_ns = timestamper()

                    ### The actual function invocation ###
                    _ = data_array[mask]

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
                    "data_dtype": np.dtype(data_dtype),
                    "data_length": data_length,
                    "true_ratio": true_ratio,
                    # Other parameters
                    # (None)
                },
            )
            benchmark_data.append(trial_data)

        # hstack all of the individual Datasets together into one large Dataset and return it.
    return Dataset.hstack(benchmark_data, destroy=True)


def bench_bool_index_numpy(**kwargs) -> Dataset:
    warmup_iters = 0
    iters = 21
    rng_seeds = [12345]
    data_dtypes = [
        np.int16,
        np.int32,
        np.float64,
        # TODO: Enable these additional data types; they're somewhat slow though, so we'd only want to
        #       run them under a 'detailed' / 'long-running' scenario
        # np.dtype('S4'),
        # np.dtype('S10'),
        # np.dtype('<U8')
    ]
    data_lengths = [
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        # TODO: Add 100M, 1G and 2G -- these need to be optional since smaller machines will run out of memory
        #       and also take longer than typical trials
    ]
    true_ratios = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0
    ]

    setup_params = itertools.product(
        rng_seeds,
        data_dtypes,
        data_lengths,
        true_ratios
    )

    # Datasets containing timing data and parameters from the trials in this benchmark.
    benchmark_data: List[Dataset] = []

    for (
            rng_seed,
            data_dtype,
            data_length,
            true_ratio,
    ) in setup_params:
        # HACK: Until we have a better approach for supporting non-rectangular parameter spaces,
        #       or otherwise being able to skip certain combinations of parameters (e.g. because
        #       they're invalid, non-realistic, or otherwise don't make sense).
        # if np.iinfo(index_dtype).max < data_length:
        #     continue

        #
        # Setup phase. The data here is used for both the warmup and the real, timed function invocations.
        #
        # Make sure to re-initialize the RNG each time so we get a repeatable result.
        rng = default_rng(rng_seed)

        data_array = rand_array(rng, data_length, dtype=np.dtype(data_dtype))
        if hasattr(data_array, "_np"):
            data_array = data_array._np
        mask = rng.random(data_length) < true_ratio

        # Sweep over other parameters that aren't required by the setup phase.
        other_params = [None]
        for _ in other_params:
            # Allocate an array to hold the raw timing data.
            # TODO: Change to use TimeSpan?
            timing_data = empty(iters, dtype=np.int64)

            for is_warmup in (True, False):
                loop_count = warmup_iters if is_warmup else iters

                for i in range(loop_count):
                    start_time_ns = timestamper()

                    ### The actual function invocation ###
                    _ = data_array[mask]

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
                    "data_dtype": np.dtype(data_dtype),
                    "data_length": data_length,
                    "true_ratio": true_ratio,
                    # Other parameters
                    # (None)
                },
            )
            benchmark_data.append(trial_data)

        # hstack all of the individual Datasets together into one large Dataset and return it.
    return Dataset.hstack(benchmark_data, destroy=True)


def bench_mbget(**kwargs) -> Dataset:
    # TODO: Add additional dimensions:
    #  * number of threads
    #  * recycler on/off
    #  * different key multiplicity distributions (in the rand_fancyindex function)
    #  * different amounts of 'sortedness' of the fancy index (from the rand_fancyindex function)

    # Fixed parameters which apply to all of the trials in this benchmark.
    warmup_iters = 0
    iters = 21  # This duration of this function is (usually) fairly short, so the performance is prone to random noise -- using more iterations helps

    # Parameters we'll sweep over for the benchmark.
    rng_seeds = [12345]
    data_dtypes = [
        np.int16,
        np.int32,
        np.float64,
        # TODO: Enable these additional data types; they're somewhat slow though, so we'd only want to
        #       run them under a 'detailed' / 'long-running' scenario
        np.dtype('S11'),
    ]
    index_dtypes = [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        # TODO: Add float32 / float64 once rand_fancyindex() supports them
    ]
    data_lengths = [
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        # TODO: Add 100M, 1G and 2G -- these need to be optional since smaller machines will run out of memory
        #       and also take longer than typical trials
    ]
    index_lengths = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    invalid_ratios = [
        0.0,
        0.01,
        0.1,
        # TODO: Enable these additional values for the 'detailed' scenario
        # 0.5,
        # 0.9,
    ]

    setup_params = itertools.product(
        rng_seeds,
        data_dtypes,
        index_dtypes,
        data_lengths,
        index_lengths,
        invalid_ratios,
    )

    # Datasets containing timing data and parameters from the trials in this benchmark.
    benchmark_data: List[Dataset] = []

    for (
        rng_seed,
        data_dtype,
        index_dtype,
        data_length,
        index_length,
        invalid_ratio,
    ) in setup_params:
        # HACK: Until we have a better approach for supporting non-rectangular parameter spaces,
        #       or otherwise being able to skip certain combinations of parameters (e.g. because
        #       they're invalid, non-realistic, or otherwise don't make sense).
        if np.iinfo(index_dtype).max < data_length:
            continue

        #
        # Setup phase. The data here is used for both the warmup and the real, timed function invocations.
        #
        # Make sure to re-initialize the RNG each time so we get a repeatable result.
        rng = default_rng(rng_seed)

        data_array = rand_array(rng, data_length, dtype=np.dtype(data_dtype), invalid_ratio=invalid_ratio)
        fancyindex = rand_fancyindex(
            rng,
            index_length,
            dtype=np.dtype(index_dtype),
            source_arr_len=data_length,
            invalid_ratio=invalid_ratio,
        )

        # Sweep over other parameters that aren't required by the setup phase.
        other_params = [None]
        for _ in other_params:
            # Allocate an array to hold the raw timing data.
            # TODO: Change to use TimeSpan?
            timing_data = empty(iters, dtype=np.int64)

            for is_warmup in (True, False):
                loop_count = warmup_iters if is_warmup else iters

                for i in range(loop_count):
                    start_time_ns = timestamper()

                    ### The actual function invocation ###
                    mbget(data_array, fancyindex)

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
                    "data_dtype": np.dtype(data_dtype),
                    "index_dtype": np.dtype(index_dtype),
                    "data_length": data_length,
                    "index_length": index_length,
                    "invalid_ratio": invalid_ratio,
                    # Other parameters
                    # (None)
                },
            )
            benchmark_data.append(trial_data)

    # hstack all of the individual Datasets together into one large Dataset and return it.
    return Dataset.hstack(benchmark_data, destroy=True)


def bench_mbget_numba(**kwargs) -> Dataset:
    # TODO: Add additional dimensions:
    #  * number of threads
    #  * recycler on/off
    #  * different key multiplicity distributions (in the rand_fancyindex function)
    #  * different amounts of 'sortedness' of the fancy index (from the rand_fancyindex function)

    # Fixed parameters which apply to all of the trials in this benchmark.
    warmup_iters = 1
    iters = 21  # This duration of this function is (usually) fairly short, so the performance is prone to random noise -- using more iterations helps

    # Parameters we'll sweep over for the benchmark.
    rng_seeds = [12345]
    data_dtypes = [
        np.int16,
        np.int32,
        np.float64,
        # TODO: Enable these additional data types; they're somewhat slow though, so we'd only want to
        #       run them under a 'detailed' / 'long-running' scenario
        np.dtype('S11'),
    ]
    index_dtypes = [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        # TODO: Add float32 / float64 once rand_fancyindex() supports them
    ]
    data_lengths = [
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        # TODO: Add 100M, 1G and 2G -- these need to be optional since smaller machines will run out of memory
        #       and also take longer than typical trials
    ]
    index_lengths = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    invalid_ratios = [
        0.0,
        0.01,
        0.1,
        # TODO: Enable these additional values for the 'detailed' scenario
        # 0.5,
        # 0.9,
    ]

    setup_params = itertools.product(
        rng_seeds,
        data_dtypes,
        index_dtypes,
        data_lengths,
        index_lengths,
        invalid_ratios,
    )

    # Datasets containing timing data and parameters from the trials in this benchmark.
    benchmark_data: List[Dataset] = []
    for (
            rng_seed,
            data_dtype,
            index_dtype,
            data_length,
            index_length,
            invalid_ratio,
    ) in setup_params:
        # HACK: Until we have a better approach for supporting non-rectangular parameter spaces,
        #       or otherwise being able to skip certain combinations of parameters (e.g. because
        #       they're invalid, non-realistic, or otherwise don't make sense).
        if np.iinfo(index_dtype).max < data_length:
            continue

        #
        # Setup phase. The data here is used for both the warmup and the real, timed function invocations.
        #
        # Make sure to re-initialize the RNG each time so we get a repeatable result.
        rng = default_rng(rng_seed)

        data_array = rand_array(rng, data_length, dtype=np.dtype(data_dtype), invalid_ratio=invalid_ratio)
        fancyindex = rand_fancyindex(
            rng,
            index_length,
            dtype=np.dtype(index_dtype),
            source_arr_len=data_length,
            invalid_ratio=invalid_ratio,
        )

        # Sweep over other parameters that aren't required by the setup phase.
        other_params = [None]
        for _ in other_params:
            # Allocate an array to hold the raw timing data.
            # TODO: Change to use TimeSpan?
            timing_data = empty(iters, dtype=np.int64)

            for is_warmup in (True, False):
                loop_count = warmup_iters if is_warmup else iters

                for i in range(loop_count):
                    start_time_ns = timestamper()

                    ### The actual function invocation ###
                    mbget_numba(data_array, fancyindex)

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
                    "data_dtype": np.dtype(data_dtype),
                    "index_dtype": np.dtype(index_dtype),
                    "data_length": data_length,
                    "index_length": index_length,
                    "invalid_ratio": invalid_ratio,
                    # Other parameters
                    # (None)
                },
            )
            benchmark_data.append(trial_data)

    # hstack all of the individual Datasets together into one large Dataset and return it.
    return Dataset.hstack(benchmark_data, destroy=True)


def bench_astype(**kwargs):
    # TODO: Add additional dimensions:
    #  * number of threads
    #  * recycler on/off
    #  * different key multiplicity distributions (in the rand_fancyindex function)
    #  * different amounts of 'sortedness' of the fancy index (from the rand_fancyindex function)

    # Fixed parameters which apply to all of the trials in this benchmark.
    warmup_iters = 0
    iters = 21  # This duration of this function is (usually) fairly short, so the performance is prone to random noise -- using more iterations helps

    # Parameters we'll sweep over for the benchmark.
    rng_seeds = [12345]
    src_dtypes = [
        np.int16,
        np.int32,
        np.float64,
        # np.dtype('S11'),
    ]
    dst_dtypes = [
        np.int16,
        np.int32,
        np.float64,
        # np.dtype('S11'),
    ]
    data_lengths = [
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000
        # TODO: Add 100M, 1G and 2G -- these need to be optional since smaller machines will run out of memory
        #       and also take longer than typical trials
    ]

    invalid_ratios = [
        0.0,
        0.01,
        0.1,
        # TODO: Enable these additional values for the 'detailed' scenario
        # 0.5,
        # 0.9,
    ]
    setup_params = itertools.product(
        rng_seeds,
        src_dtypes,
        dst_dtypes,
        data_lengths,
        invalid_ratios,
    )

    # Datasets containing timing data and parameters from the trials in this benchmark.
    benchmark_data: List[Dataset] = []

    for (
            rng_seed,
            src_dtype,
            dst_dtype,
            data_length,
            invalid_ratio,
    ) in setup_params:
        # Setup phase. The data here is used for both the warmup and the real, timed function invocations.
        #
        # Make sure to re-initialize the RNG each time so we get a repeatable result.
        rng = default_rng(rng_seed)

        data_array = rand_array(rng, data_length, dtype=np.dtype(src_dtype), invalid_ratio=invalid_ratio)

        # Sweep over other parameters that aren't required by the setup phase.
        other_params = [None]
        for _ in other_params:
            # Allocate an array to hold the raw timing data.
            # TODO: Change to use TimeSpan?
            timing_data = empty(iters, dtype=np.int64)

            for is_warmup in (True, False):
                loop_count = warmup_iters if is_warmup else iters

                for i in range(loop_count):
                    start_time_ns = timestamper()

                    ### The actual function invocation ###
                    data_array.astype(dtype=dst_dtype)

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
                    "src_dtype": np.dtype(src_dtype),
                    "dst_dtype": np.dtype(dst_dtype),
                    "data_length": data_length,
                    "invalid_ratio": invalid_ratio,
                    # Other parameters
                    # (None)
                },
            )
            benchmark_data.append(trial_data)

    # hstack all of the individual Datasets together into one large Dataset and return it.
    return Dataset.hstack(benchmark_data, destroy=True)


def bench_astype_numpy(**kwargs):
    # TODO: Add additional dimensions:
    #  * number of threads
    #  * recycler on/off
    #  * different key multiplicity distributions (in the rand_fancyindex function)
    #  * different amounts of 'sortedness' of the fancy index (from the rand_fancyindex function)

    # Fixed parameters which apply to all of the trials in this benchmark.
    warmup_iters = 0
    iters = 21  # This duration of this function is (usually) fairly short, so the performance is prone to random noise -- using more iterations helps

    # Parameters we'll sweep over for the benchmark.
    rng_seeds = [12345]
    src_dtypes = [
        np.int16,
        np.int32,
        np.float64,
        # np.dtype('S11'),
    ]
    dst_dtypes = [
        np.int16,
        np.int32,
        np.float64,
        # np.dtype('S11'),
    ]
    data_lengths = [
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000
        # TODO: Add 100M, 1G and 2G -- these need to be optional since smaller machines will run out of memory
        #       and also take longer than typical trials
    ]

    invalid_ratios = [
        0.0,
        0.01,
        0.1,
        # TODO: Enable these additional values for the 'detailed' scenario
        # 0.5,
        # 0.9,
    ]
    setup_params = itertools.product(
        rng_seeds,
        src_dtypes,
        dst_dtypes,
        data_lengths,
        invalid_ratios,
    )

    # Datasets containing timing data and parameters from the trials in this benchmark.
    benchmark_data: List[Dataset] = []

    for (
            rng_seed,
            src_dtype,
            dst_dtype,
            data_length,
            invalid_ratio,
    ) in setup_params:
        # Setup phase. The data here is used for both the warmup and the real, timed function invocations.
        #
        # Make sure to re-initialize the RNG each time so we get a repeatable result.
        rng = default_rng(rng_seed)

        data_array = rand_array(rng, data_length, dtype=np.dtype(src_dtype), invalid_ratio=invalid_ratio)
        if hasattr(data_array, '_np'):
            data_array = data_array._np

        # Sweep over other parameters that aren't required by the setup phase.
        other_params = [None]
        for _ in other_params:
            # Allocate an array to hold the raw timing data.
            # TODO: Change to use TimeSpan?
            timing_data = empty(iters, dtype=np.int64)

            for is_warmup in (True, False):
                loop_count = warmup_iters if is_warmup else iters

                for i in range(loop_count):
                    start_time_ns = timestamper()

                    ### The actual function invocation ###
                    data_array.astype(dtype=dst_dtype)

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
                    "src_dtype": np.dtype(src_dtype),
                    "dst_dtype": np.dtype(dst_dtype),
                    "data_length": data_length,
                    "invalid_ratio": invalid_ratio,
                    # Other parameters
                    # (None)
                },
            )
            benchmark_data.append(trial_data)

    # hstack all of the individual Datasets together into one large Dataset and return it.
    return Dataset.hstack(benchmark_data, destroy=True)


def bench_astype_numba(**kwargs):
    # TODO: Add additional dimensions:
    #  * number of threads
    #  * recycler on/off
    #  * different key multiplicity distributions (in the rand_fancyindex function)
    #  * different amounts of 'sortedness' of the fancy index (from the rand_fancyindex function)

    # Fixed parameters which apply to all of the trials in this benchmark.
    warmup_iters = 1
    iters = 21  # This duration of this function is (usually) fairly short, so the performance is prone to random noise -- using more iterations helps

    # Parameters we'll sweep over for the benchmark.
    rng_seeds = [12345]
    src_dtypes = [
        np.int16,
        np.int32,
        np.float64,
    ]
    dst_dtypes = [
        np.int16,
        np.int32,
        np.float64,
    ]
    data_lengths = [
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000
        # TODO: Add 100M, 1G and 2G -- these need to be optional since smaller machines will run out of memory
        #       and also take longer than typical trials
    ]
    invalid_ratios = [
        0.0,
        0.01,
        0.1,
        # TODO: Enable these additional values for the 'detailed' scenario
        # 0.5,
        # 0.9,
    ]
    setup_params = itertools.product(
        rng_seeds,
        src_dtypes,
        dst_dtypes,
        data_lengths,
        invalid_ratios,
    )

    # Datasets containing timing data and parameters from the trials in this benchmark.
    benchmark_data: List[Dataset] = []

    for (
            rng_seed,
            src_dtype,
            dst_dtype,
            data_length,
            invalid_ratio,
    ) in setup_params:
        # Setup phase. The data here is used for both the warmup and the real, timed function invocations.
        #
        # Make sure to re-initialize the RNG each time so we get a repeatable result.
        rng = default_rng(rng_seed)

        data_array = rand_array(rng, data_length, dtype=np.dtype(src_dtype), invalid_ratio=invalid_ratio)

        # Sweep over other parameters that aren't required by the setup phase.
        other_params = [None]
        for _ in other_params:
            # Allocate an array to hold the raw timing data.
            # TODO: Change to use TimeSpan?
            timing_data = empty(iters, dtype=np.int64)

            for is_warmup in (True, False):
                loop_count = warmup_iters if is_warmup else iters

                for i in range(loop_count):
                    start_time_ns = timestamper()

                    ### The actual function invocation ###
                    astype_numba(data_array, np.dtype(dst_dtype))

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
                    "src_dtype": np.dtype(src_dtype),
                    "dst_dtype": np.dtype(dst_dtype),
                    "data_length": data_length,
                    "invalid_ratio": invalid_ratio
                    # Other parameters
                    # (None)
                },
            )
            benchmark_data.append(trial_data)

    # hstack all of the individual Datasets together into one large Dataset and return it.
    return Dataset.hstack(benchmark_data, destroy=True)


def bench_compare_ops(**kwargs):
    # TODO: Add additional dimensions:
    #  * number of threads
    #  * recycler on/off
    #  * different key multiplicity distributions (in the rand_fancyindex function)
    #  * different amounts of 'sortedness' of the fancy index (from the rand_fancyindex function)

    # Fixed parameters which apply to all of the trials in this benchmark.
    warmup_iters = 0
    iters = 21  # This duration of this function is (usually) fairly short, so the performance is prone to random noise -- using more iterations helps

    # Parameters we'll sweep over for the benchmark.
    rng_seeds = [12345]
    arr1_dtypes = [
        np.int16,
        np.int32,
        np.float64,
    ]
    arr2_dtypes = [
        np.int16,
        np.int32,
        np.float64,
    ]
    data_lengths = [
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        # TODO: Add 100M, 1G and 2G -- these need to be optional since smaller machines will run out of memory
        #       and also take longer than typical trials
    ]
    invalid_ratios = [
        0.0,
        0.01,
        0.1,
        # TODO: Enable these additional values for the 'detailed' scenario
        # 0.5,
        # 0.9,
    ]
    ops = [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.ge,
        operator.gt
    ]
    setup_params = itertools.product(
        rng_seeds,
        arr1_dtypes,
        arr2_dtypes,
        data_lengths,
        invalid_ratios,
        ops,
    )

    # Datasets containing timing data and parameters from the trials in this benchmark.
    benchmark_data: List[Dataset] = []

    for (
            rng_seed,
            arr1_dtype,
            arr2_dtype,
            data_length,
            invalid_ratio,
            op,
    ) in setup_params:
        # Setup phase. The data here is used for both the warmup and the real, timed function invocations.
        #
        # Make sure to re-initialize the RNG each time so we get a repeatable result.
        rng = default_rng(rng_seed)

        arr1 = rand_array(rng, data_length, dtype=np.dtype(arr1_dtype), invalid_ratio=invalid_ratio)
        arr2 = rand_array(rng, data_length, dtype=np.dtype(arr2_dtype), invalid_ratio=invalid_ratio)

        # Sweep over other parameters that aren't required by the setup phase.
        other_params = [None]
        for _ in other_params:
            # Allocate an array to hold the raw timing data.
            # TODO: Change to use TimeSpan?
            timing_data = empty(iters, dtype=np.int64)
            for is_warmup in (True, False):
                loop_count = warmup_iters if is_warmup else iters

                for i in range(loop_count):

                    start_time_ns = timestamper()

                    #invocation of actual actual function
                    op(arr1, arr2)

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
                    "arr1_dtype": np.dtype(arr1_dtype),
                    "arr2_dtype": np.dtype(arr2_dtype),
                    "operation": op.__name__,
                    "data_length": data_length,
                    "invalid_ratio": invalid_ratio,
                    # Other parameters
                    # (None)
                },
            )
            benchmark_data.append(trial_data)

    # hstack all of the individual Datasets together into one large Dataset and return it.
    return Dataset.hstack(benchmark_data, destroy=True)


def bench_compare_ops_numpy(**kwargs):
    # TODO: Add additional dimensions:
    #  * number of threads
    #  * recycler on/off
    #  * different key multiplicity distributions (in the rand_fancyindex function)
    #  * different amounts of 'sortedness' of the fancy index (from the rand_fancyindex function)

    # Fixed parameters which apply to all of the trials in this benchmark.
    warmup_iters = 0
    iters = 21  # This duration of this function is (usually) fairly short, so the performance is prone to random noise -- using more iterations helps

    # Parameters we'll sweep over for the benchmark.
    rng_seeds = [12345]
    arr1_dtypes = [
        np.int16,
        np.int32,
        np.float64,
    ]
    arr2_dtypes = [
        np.int16,
        np.int32,
        np.float64,
    ]
    data_lengths = [
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        # TODO: Add 100M, 1G and 2G -- these need to be optional since smaller machines will run out of memory
        #       and also take longer than typical trials
    ]
    ops = [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.ge,
        operator.gt
    ]
    setup_params = itertools.product(
        rng_seeds,
        arr1_dtypes,
        arr2_dtypes,
        data_lengths,
        ops,
    )

    # Datasets containing timing data and parameters from the trials in this benchmark.
    benchmark_data: List[Dataset] = []

    for (
            rng_seed,
            arr1_dtype,
            arr2_dtype,
            data_length,
            op,
    ) in setup_params:
        # Setup phase. The data here is used for both the warmup and the real, timed function invocations.
        #
        # Make sure to re-initialize the RNG each time so we get a repeatable result.
        rng = default_rng(rng_seed)

        arr1 = rand_array(rng, data_length, dtype=np.dtype(arr1_dtype))._np
        arr2 = rand_array(rng, data_length, dtype=np.dtype(arr2_dtype))._np
        # Sweep over other parameters that aren't required by the setup phase.
        other_params = [None]
        for _ in other_params:
            # Allocate an array to hold the raw timing data.
            # TODO: Change to use TimeSpan?
            timing_data = empty(iters, dtype=np.int64)

            for is_warmup in (True, False):
                loop_count = warmup_iters if is_warmup else iters

                for i in range(loop_count):

                    start_time_ns = timestamper()

                    #invocation of actual actual function
                    op(arr1, arr2)

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
                    "arr1_dtype": np.dtype(arr1_dtype),
                    "arr2_dtype": np.dtype(arr2_dtype),
                    "operation": op.__name__,
                    "data_length": data_length,
                    # Other parameters
                    # (None)
                },
            )
            benchmark_data.append(trial_data)

    # hstack all of the individual Datasets together into one large Dataset and return it.
    return Dataset.hstack(benchmark_data, destroy=True)


def compare_mbget():
    return create_comparison_dataset(
        {
            "mbget": bench_mbget(),
            "mbget_numba": bench_mbget_numba(),
        }
    )


def compare_astype():
    return create_comparison_dataset(
        {
            "astype": bench_astype(),
            "astype_numpy": bench_astype_numpy(),
            "astype_numba": bench_astype_numba(),
        }
    )


def compare_bool_index():
    return create_comparison_dataset(
        {
            "bool_index": bench_bool_index(),
            "bool_index_numpy": bench_bool_index_numpy()
        }
    )


def compare_compare_ops():
    return create_comparison_dataset(
        {
            "compare_ops": bench_compare_ops(),
            "compare_ops_numpy": bench_compare_ops_numpy(),
        }
    )