# -*- coding: utf-8 -*-
from typing import NamedTuple


class RunResult(NamedTuple):
    """
    Timing results for a benchmark run (statistics calculated from multiple
    invocations of a function to be benchmarked).
    """

    min_ns: int
    """
    The minimum duration, in nanoseconds, of any single invocation of the function.
    """

    median_ns: int
    """
    The median duration, in nanoseconds, of any single invocation of the function.
    """

    max_ns: int
    """
    The maximum duration, in nanoseconds, of any single invocation of the function.
    """

    est_mean_ns: int
    """
    The estimated population mean, in nanoseconds, of any single invocation of the function.
    """

    est_variance: float
    """
    The estimated population variance, in nanoseconds^2, from the `est_mean_duration`
    for each invocation of the function.
    """
