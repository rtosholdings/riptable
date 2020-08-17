# -*- coding: utf-8 -*-
import numbers
import numpy as np

from typing import Any, Callable, Mapping, Tuple, Union, Generator
from .results import RunResult
from ..rt_fastarray import FastArray


enable_bench_estimators: bool = False
"""
Controls whether our new, criterion-like benchmark estimator code
is enabled for benchmarking runs. For now, the estimated results
are not included in benchmark datasets, but are just printed to stdout.
"""


def jackknife_mean_and_var(samples: np.ndarray) -> Tuple[float, float]:
    """
    Estimate the population mean and variance from an array of samples using jackknife resampling.

    Parameters
    ----------
    samples : np.ndarray
        An array of sample datapoints to estimate the mean and variance of.

    Returns
    -------
    estimated_mean : float
        The estimated population mean.
    estimated_var : float
        The estimated population variance.

    See Also
    --------
    https://en.wikipedia.org/wiki/Jackknife_resampling

    Notes
    -----
    TODO: Make this function like `_bootstrap_resample` where it takes a dictionary/Mapping
          of aggregation functions and applies them to the leave-one-out arrays to produce
          the jackknife estimates of those aggregate functions.
          We can (relatively) efficiently perform the leave-one-out operations while also supporting
          arbitrary reductions if we create a boolean mask the same length as `samples`; then, iterate
          from 0 to len(samples) and set the previous element in the mask to True while setting the
          current (i-th) element to False. Call np.ma.array(samples, mask=(the mask array))
    """
    num_samples = len(samples)

    # Calculate the total sum and force to float64 so later steps
    # don't hit integer division rounding issues.
    total_sum = np.sum(samples).astype(np.float64)

    # Use broadcasting to get an array which contains for each
    # element the sum of the array except that element.
    subsample_sums = total_sum - samples
    subsample_means = subsample_sums / (num_samples - 1)

    # Compute the estimate of the population mean.
    estimated_mean = np.mean(subsample_means)

    # Compute the estimate of the population variance.
    sample_errors = subsample_sums - estimated_mean
    squared_errors = sample_errors * sample_errors
    estimated_var = np.mean(squared_errors) / (num_samples - 1)

    return estimated_mean, estimated_var


def median_absolute_deviation(samples: np.ndarray) -> Any:
    """
    Calculate the median absolute deviation (MAD) of an array of samples.
    """
    # Determine the median of the samples.
    median = np.median(samples)

    # Calculate the residuals (deviations) from the median of the samples.
    # We want the absolute value of the residuals since we're looking to calculate
    # the median ABSOLUTE deviation.
    abs_residuals: np.ndarray = np.abs(samples - median)

    # Sort the absolute residuals in place to make finding the median faster.
    # (So we hopefully don't allocate an array internally.)
    abs_residuals.sort()

    # Return the median of the absolute residuals.
    return np.median(abs_residuals)


def _bootstrap_resample(
    rng: Generator,
    samples: np.ndarray,
    resample_count: int,
    agg_funcs: Mapping[
        str, Union[Callable[[np.ndarray], int], Callable[[np.ndarray], float]]
    ],
) -> Mapping[str, np.ndarray]:
    """
    Bootstrap resampling of timing samples.

    This function performs bootstrap resampling, applying specified aggregation functions to
    each of the resampled arrays to produce new arrays containing the aggregate values.
    These resulting arrays can be used to produce aggregates at a specified confidence level
    for the original data.

    Parameters
    ----------
    rng : np.random.Generator
        A `Generator` used to randomly choose samples from `samples`.
    samples : np.ndarray
        Raw, nanosecond-level timing sample data.
    resample_count : int
        The total number of times to resample from `samples`.
    agg_funcs
        A dictionary holding aggregation/reduction functions; i.e. when applied to an array they produce a
        scalar result. These functions are applied to each resampled array to produce a resampled aggregate.

    Returns
    -------
    resampled_aggs
        A dictionary containing aggregates calculated from the resampled arrays.

    See Also
    --------
    https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
    https://github.com/bos/criterion
    https://github.com/bheisler/criterion.rs
    https://blogs.sas.com/content/iml/2017/07/12/bootstrap-bca-interval.html

    Notes
    -----
    TODO: Need to properly implement the BCa technique (bias-corrected and accelerated);
          Use the jackknife implementation above to estimate the acceleration parameter.
    """
    # Determine how many samples to take each time we resample.
    resample_len: int = resample_count // len(samples)
    resample_iters: int = resample_count // resample_len

    resampled_aggs = {}
    for i in range(resample_iters):
        # Create an array of integers uniformly distributed over [0, len(samples)).
        # Use it as a fancy index to resample / sample-with-replacement from the timing samples.
        resample_indices = rng.integers(low=0, high=len(samples), size=resample_len)
        resamples = samples[resample_indices]

        # Invoke each of the aggregation/reduction functions with the resampled data.
        for key, agg_func in agg_funcs.items():
            # Calculate the aggregate value.
            agg_val = agg_func(resamples)

            # If the array for this aggregate hasn't been created yet, create it.
            # Otherwise, fetch the existing array.
            if i == 0:
                dtype = (
                    np.int64 if isinstance(agg_val, numbers.Integral) else np.float64
                )
                agg_arr = np.empty(resample_iters, dtype=dtype)
                resampled_aggs[key] = agg_arr
            else:
                agg_arr = resampled_aggs[key]

            # Save the aggregated value into the array.
            agg_arr[i] = agg_val

    # Return the resampled aggregates.
    return resampled_aggs


def _analyze_results(run_nano_times: np.ndarray) -> RunResult:
    """
    Analyze the raw, nanosecond-granularity timing data from a benchmark run to produce aggregated results.

    Parameters
    ----------
    run_nano_times : np.ndarray
        Raw, nanosecond-granularity timing data from individual invocations of
        a function being benchmarked.

    Returns
    -------
    RunResult
        A `RunResult` holding aggregated data about this benchmark run.

    Notes
    -----
    TODO: Implement outlier analysis (that looks for how much / to what degree outliers are responsible for variance in the results).
    """
    # Calculate some statistics on the benchmark time samples.
    min_time = np.min(run_nano_times)
    med_time = np.median(run_nano_times)
    max_time = np.max(run_nano_times)
    est_mean, est_variance = jackknife_mean_and_var(run_nano_times)

    # TEMP: While developing the estimator code for benchmarking, make it easy to enable/disable

    #if enable_bench_estimators:
    #    # Calculate the cumulative sums of the timing samples, then perform
    #    # a linear regression to fit a line to the cumsums. The more stable
    #    # the timing results, the higher (closer to 1) the R^2 value will be.
    #    timing_cumsums = run_nano_times.view(FastArray).cumsum()
    #    iteration_numbers = np.arange(len(run_nano_times))
    #    slope, intercept, r_value, p_value, std_err = st.linregress(
    #        iteration_numbers, timing_cumsums
    #    )
    #
    #    # TEMP: Print out regression results for debugging.
    #    print(
    #        f"slope: {slope}\tintercept: {intercept}\tr_value: {r_value}\tp_value: {p_value}\tstd_err: {std_err}"
    #    )
    #
    #    # TEMP: Use bootstrap to produce 95% confidence interval (alpha=0.05) estimates of min/median/max.
    #    agg_funcs = {"min": np.min, "median": np.median, "max": np.max, "stdev": np.std}
    #    rng = np.random.default_rng()
    #    bootstrap_aggs = _bootstrap_resample(
    #        rng, run_nano_times, 10 * len(run_nano_times), agg_funcs
    #    )
    #
    #    # Print the median and confidence interval endpoints for each of the aggregates.
    #    for key, agg_arr in bootstrap_aggs.items():
    #        pcts = np.nanpercentile(agg_arr, [2.5, 50.0, 97.5])
    #        print(f"Bootstrap (95% CI)[{key}]: {pcts[1]} ({pcts[0]} ... {pcts[2]})")

    return RunResult(min_time, med_time, max_time, int(est_mean), est_variance)
