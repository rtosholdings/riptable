# -*- coding: utf-8 -*-
"""
Benchmarks for 'merge' functions.
"""
__all__ = [
    "create_trial_dataset",
    "elapsed_ns_colname",
    "implementation_colname",
    "loop_iter_colname",
    "quick_analysis",
]


import riptide_cpp as rc
import numpy as np
import inspect

from collections import namedtuple
from contextlib import contextmanager
from functools import wraps
from itertools import product
from typing import List, Mapping, Optional, Union, Callable
from .benchmark import timestamper
from ..rt_enum import INVALID_SHORT_NAME
from ..rt_categorical import Categorical
from ..rt_dataset import Dataset
from ..rt_fastarray import FastArray
from ..rt_multiset import Multiset
from ..rt_numpy import arange, full


elapsed_ns_colname: str = "elapsed_ns"
"""
Name to use in benchmark Datasets for the column holding the elapsed time (in ns)
for the function invocations. 
"""

loop_iter_colname: str = "loop_idx"
"""
Name to use in benchmark Datasets for the column holding the loop index
for each function invocation. 
"""

implementation_colname: str = "impl"
"""
Name to use in benchmark _comparison_ Datasets for the column holding the
implementation name.
"""

# TODO move this into a settings library file.
def set_thread(thread_count):
    # Consdier setting wake up threads to the number of worker threads, i.e.
    # thread_count minus the main thread.
    rc.SetThreadWakeUp(thread_count)


def set_recycler(recycle):
    if recycle:
        FastArray._RON(quiet=True)
    else:
        FastArray._ROFF(quiet=True)


def get_thread():
    thread_count = rc.GetThreadWakeUp()
    return thread_count


def get_recycler():
    return FastArray.Recycle


SettingEntry = namedtuple("SettingEntry", ("getter_func", "setter_func"))
"""SettingEntry consists of a pointer to a setting getter and setter function.
The call to `setter_func(getter_func())` should set the original state.
"""

BENCHMARK_SETTINGS_TABLE = {
    # TODO - FasterUfunc, Recycler timeouts, and MathLedger settings
    "thread_count": SettingEntry(get_thread, set_thread),
    "recycler": SettingEntry(get_recycler, set_recycler),
}
"""BENCHMARK_SETTINGS_TABLE is the global map of setting names to SettingEntries
that allow benchmarks to apply stateful settings. E.g., setting riptable thread
counts or recycler modes that vary across benchmark runs.
"""


def get_settings_getter_func(key: str) -> Optional[Callable]:
    if key in BENCHMARK_SETTINGS_TABLE:
        return BENCHMARK_SETTINGS_TABLE[key].getter_func
    return None


def get_settings_setter_func(key: str) -> Optional[Callable]:
    if key in BENCHMARK_SETTINGS_TABLE:
        return BENCHMARK_SETTINGS_TABLE[key].setter_func
    return None


def create_comparison_dataset(
    bench_datasets: Mapping[str, Dataset], destroy: bool = False
) -> Dataset:
    """
    TODO

    Parameters
    ----------
    bench_datasets
    destroy : bool, defaults to False

    Returns
    -------
    compare_dataset : Dataset
    """
    # For each dataset, create a shallow copy (to avoid modifying the original),
    # then add a Categorical column with the implementation name to it.
    impl_datasets: List[Dataset] = []
    for impl_name, impl_bench_data in bench_datasets.items():
        data_copy = impl_bench_data if destroy else impl_bench_data.copy(deep=False)
        data_copy[implementation_colname] = Categorical([impl_name]).tile(
            len(data_copy)
        )
        data_copy.col_move_to_front([implementation_colname])
        impl_datasets.append(data_copy)

    # hstack the Datasets together and return.
    return Dataset.hstack(impl_datasets, destroy=destroy)


def create_trial_dataset(
    timing_data: np.ndarray,
    trial_params: Optional[Mapping[str, Union[int, str, bytes, float, np.dtype]]],
) -> Dataset:
    """
    Create the Dataset instance for a trial (repeated invocations of a function to be benchmarked).

    Parameters
    ----------
    timing_data : FastArray
        Array containing the elapsed time, in nanoseconds, for each invocation
        of the function being benchmarked.
    trial_params : dict
        A dictionary containing the trial parameters by name.
        Parameter names should all be valid for use as column names in a Dataset.

    Returns
    -------
    trial_dataset : Dataset
    """
    trial_data = Dataset()

    # Add the timing data and loop index columns.
    trial_length = len(timing_data)
    trial_data[loop_iter_colname] = arange(trial_length)
    trial_data[elapsed_ns_colname] = timing_data

    # If trial parameters were specified, add them to the dataset.
    if trial_params is not None:
        for param_name, param_value in trial_params.items():
            # Create an array for this parameter the same length as the number of function invocations,
            # and fill it with the parameter value. For string-valued parameters, create a Categorical
            # so the array is more compact in memory.
            if isinstance(param_value, int):
                dtype = np.min_scalar_type(param_value)
                param_val_array = full(trial_length, param_value, dtype)
            elif isinstance(param_value, (bytes, str)):
                dtype = np.min_scalar_type(param_value)
                param_val_array = Categorical(full(trial_length, param_value, dtype))
            elif isinstance(param_value, float):
                param_val_array = full(trial_length, param_value, dtype=np.float64)
            elif isinstance(param_value, np.dtype):
                # Same as for the bytes/str case; we just extract the dtype name and use that.
                param_value = param_value.name
                dtype = np.min_scalar_type(param_value)
                param_val_array = Categorical(full(trial_length, param_value, dtype))
            else:
                raise ValueError(
                    f"Unsupported type '{type(param_value)}' specified for trial parameter '{param_name}'."
                )

            # Add the trial parameter array to the Dataset.
            trial_data[param_name] = param_val_array

    # Return the constructed Dataset.
    return trial_data


def quick_analysis(bench_data: Dataset) -> Multiset:
    """
    Performs a simple, quick analysis on raw benchmark data

    Parameters
    ----------
    bench_data : Dataset
        A Dataset created with `create_trial_dataset`, or a Dataset
        created by `hstack`-ing such Datasets (from multiple trials) together.

    Returns
    -------
    quick_bench_analysis : Multiset
    """
    # Group the results by all columns _except_ the per-invocation columns (the loop index and elapsed time);
    # this effectively gets us a group per trial.
    gb_keys = list(bench_data.keys())
    gb_keys.remove(loop_iter_colname)
    gb_keys.remove(elapsed_ns_colname)
    gb_trial = bench_data.groupby(gb_keys)

    # Get the min/median/max elapsed time per trial and return it as a Multiset for display.
    return gb_trial[elapsed_ns_colname].agg(["min", "median", "max"])


@contextmanager
def _benchmark_setting_context(setting_to_values):
    # getter / setter for benchmark setting context
    # check all settings have defined getters and setters and they're callable

    # save original values
    setting_to_originals = dict.fromkeys(setting_to_values)
    for k, v in setting_to_values.items():
        setting_to_originals[k] = get_settings_getter_func(k)()
    try:
        # set new values
        for k, v in setting_to_values.items():
            get_settings_setter_func(k)(v)
        yield
    finally:
        # set original values
        for k, v in setting_to_originals.items():
            get_settings_setter_func(k)(v)


def benchmark(*_benchmark_arguments, **_benchmark_kwargs):
    """main entry point for riptable benchmarking library"""

    def decorate(func):
        # validate function arguments
        if inspect.isclass(func):
            # @benchmark assumes it's decorating a function
            raise ValueError(
                f"'@benchmark' cannot be applied to a class {func.__name__}"
            )

        # TODO handle benchmark functions that are instance or static methods
        # 'cls' and 'self' will appear in the argspec

        argspec_args = inspect.getfullargspec(func)[0]
        # TODO handle positional arguments from _benchmark_arguments
        benchmark_kwargs = dict(_benchmark_kwargs)

        # Handle required keyword arguments that are core to the benchmark framework.
        if "benchmark_params" not in benchmark_kwargs:
            raise ValueError(
                f"'benchmark_params' is a required keyword argument, got keyword arguments {benchmark_kwargs}"
            )
        benchmark_params = benchmark_kwargs["benchmark_params"]
        bench_iters = benchmark_kwargs.get("benchmark_iterations", 5)
        warmup_iters = _benchmark_kwargs.get("warmup_iterations", 1)

        # Benchmark parameter and setting checks.
        func_params = set(argspec_args)
        setting_params = benchmark_params.keys() - func_params
        param_symmetric_diff = func_params.union(setting_params).symmetric_difference(
            benchmark_params.keys()
        )
        if param_symmetric_diff:
            raise ValueError(
                f"'@benchmark', when benchmarking function {func.__name__}, got param and / or settings that are not in benchmark_params.\nBenchmark parameters {benchmark_params}.\nBenchmark function parameters {func_params}\nBenchmark settings {setting_params}\nSymmetric difference between {param_symmetric_diff}"
            )

        @wraps(func)
        def wrapper(*args, **kwargs):
            benchmark_results: List[Dataset] = []
            # param_product contains a product sweep of all the parameters parameter values.
            param_product = product(
                *[
                    benchmark_params[param_name]
                    for param_name in benchmark_params.keys()
                ]
            )
            for i, param_values in enumerate(param_product):
                # Maintain two dictionaries for this trials parameters and settings values.
                setting_param_to_value = {}
                trial_param_to_value = dict.fromkeys(
                    benchmark_params, INVALID_SHORT_NAME
                )
                for k, v in zip(benchmark_params.keys(), param_values):
                    if k in setting_params:
                        setting_param_to_value[k] = v
                    trial_param_to_value[k] = v
                with _benchmark_setting_context(setting_param_to_value):
                    timing_data = np.empty(bench_iters, dtype=np.int64)

                    # Copy trial parameters as arguments to func.
                    for arg_spec in argspec_args:
                        if arg_spec in trial_param_to_value:
                            kwargs[arg_spec] = trial_param_to_value[arg_spec]

                    # warm up
                    for i in range(warmup_iters):
                        func(*args, **kwargs)

                    # trial run
                    for i in range(bench_iters):
                        start_time_ns = timestamper()
                        # Note, there is overhead when unpacking positional and keyword arguments within timestamper.
                        # Running cProfile with Python 3.7.1 on a dummy function that takes four positional
                        # and four keyword arguments showed the splat operator has some overhead
                        # - 100,000 iterations has a difference of 0.007 seconds
                        # - 1 million iterations has a difference of 0.071 seconds
                        # - 10 million iterations has a difference of 0.801 seconds
                        # more compared to normal parameter passing.
                        func(*args, **kwargs)
                        timing_data[i] = timestamper() - start_time_ns

                    # TODO - move this to its own function process_trial_params
                    # Should handle print friendly representations
                    for k, v in trial_param_to_value.items():
                        if isinstance(v, tuple):
                            trial_param_to_value[k] = str(v)
                        if isinstance(v, (np.ndarray, Dataset)):
                            trial_param_to_value[k] = len(v)

                    trial_data = create_trial_dataset(timing_data, trial_param_to_value)
                    benchmark_results.append(trial_data)

            result = Dataset.hstack(benchmark_results, destroy=True)

            return result

        return wrapper

    return decorate
