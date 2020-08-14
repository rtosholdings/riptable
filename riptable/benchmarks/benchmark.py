# -*- coding: utf-8 -*-
import numpy as np
import riptide_cpp as rc

from typing import Callable
from .analysis import enable_bench_estimators, _analyze_results
from ..rt_dataset import Dataset
from ..rt_fastarray import FastArray
from ..rt_fastarray import FastArray as FA
from ..rt_multiset import Multiset
from ..rt_numpy import empty


# TODO:  When running in Py37+, add time.perf_counter_ns() (and maybe time.time_ns()).
_timestamp_funcs = {"get_nano_time": rc.GetNanoTime, "tsc": rc.GetTSC}
"""
Table of functions for getting a nanosecond-precision timestamp.
"""


timestamper = _timestamp_funcs["get_nano_time"]
"""The timestamping function to use in benchmarks."""


def _makename(dtype, asize):
    nm = dtype.__name__
    last = len(nm)
    while last >= 0:
        last = last - 1
        chr = nm[last]
        if chr < "0" or chr > "9":
            break
    if last > 0 and last < len(nm) - 1:
        nm = nm[0] + nm[last + 1 :]

    number = str(asize)
    if asize >= 1000:
        number = str(asize // 1000) + "K"
        if asize >= 1_000_000:
            number = str(asize // 1_000_000) + "M"
    return nm + "_" + number


def _time_run(
    timerfunc: Callable[[], int], func, *args, loops: int, **kwargs
) -> np.ndarray:
    """
    Invoke a function to be benchmarked repeatedly with the specified arguments to capture timing information.

    Parameters
    ----------
    timerfunc : callable
        Function returning a timestamp (in nanoseconds) when invoked.
    func : callable
        The function to be benchmarked.
    args
        Arguments to be passed to `func`.
    loops : int
        The number of times to invoke `func`.
    kwargs
        Keyword arguments to be passed to `func`.

    Returns
    -------
    RunResult
        A `RunResult` holding aggregated data about this benchmark run.
    """
    # preallocate array of int64 for nanoseconds
    atimers = np.zeros(loops, dtype=np.int64)

    for i in range(loops):
        start_time = timerfunc()
        func(*args, **kwargs)
        end_time = timerfunc()
        atimers[i] = end_time - start_time

    return atimers


def benchmark(
    funcs,
    dtypes,
    asizes,
    threads,
    loops=10,
    ratio=False,
    recycle=True,
    funcargs=1,
    scalar=False,
):
    """
    riptable benchmarking comparison performance FA vs numpy

    Will return a Dataset or Multiset depending on ratio=True/False.
    Columns will be arraysizes or dtypes across top
    Rows will be func or threads

    Parameters
    ----------
    funcs:
    dtypes:
    asizes:
    threads: list of threads to use
    loops: int, default 10
    ratio: bool, default True
        True to return a Dataset with the ratio of FastArray/numpy (higher is better)
        False to return a MultiSet with the times in seconds.
    recycle:
    funcargs:
    scalar:

    Examples
    --------
    benchmark(dtypes=np.int32, ratio=True)
    benchmark(funcs=np.add, dtypes=np.float32, funcargs=2, ratio=True)

    Notes
    -----
    Better way to test add on small array
    a=np.arange(1000, dtype=np.int32)
    %timeit np.add(a,a)
    %timeit rc.BasicMathTwoInputs((a, a), 1, 7)
    """

    def fix_scalar(s):
        # make sure always a list
        if not isinstance(s, list):
            return [s]
        return s

    funcs = fix_scalar(funcs)
    threads = fix_scalar(threads)
    dtypes = fix_scalar(dtypes)
    asizes = fix_scalar(asizes)

    func_len = len(funcs)
    thread_len = len(threads)
    dtype_len = len(dtypes)
    asize_len = len(asizes)

    rowlen = func_len * thread_len
    ds = {}

    # now make rownames
    threadnames = []
    funcnames = []
    for func in funcs:
        for thread in threads:
            threadnames.append(thread)
            funcnames.append(func.__name__)

    ds["Thread"] = threadnames
    ds["Func"] = funcnames

    # now make the column names
    colnames = []
    for dtype in dtypes:
        for asize in asizes:
            name = _makename(dtype, asize)
            ds[name] = empty(rowlen, dtype=np.float64)

    ds_fa = Dataset(ds)
    ds_fa.label_set_names(["Thread", "Func"])
    ds_np = ds_fa.copy()

    if not recycle:
        FA._ROFF()
        FA._GCNOW()

    # preserve threading modes
    savemode = rc.ThreadingMode(1)
    savethreads = rc.SetThreadWakeUp(8)

    # Get the timestamping function to use.
    # TODO: Make this user-configurable.
    timestamp_func = _timestamp_funcs["get_nano_time"]

    # make two passes
    dscount = 0
    for ds in [ds_fa, ds_np]:
        col = 2
        for dtype in dtypes:
            for asize in asizes:
                row = 0
                # can make random arrays also
                array = np.arange(asize, dtype=dtype)
                if dscount == 0:
                    # convert to FastArray for first run
                    array = array.view(FastArray)

                for func in funcs:
                    for thread in threads:
                        if thread == 1:
                            rc.ThreadingMode(1)
                        else:
                            rc.ThreadingMode(0)
                            # subtract since this is worker threads and we are counting main thread
                            rc.SetThreadWakeUp(thread - 1)
                        if funcargs == 1:
                            raw_timings = _time_run(
                                timestamp_func, func, array, loops=loops
                            )
                        else:
                            if scalar:
                                # use a scalar as second input
                                raw_timings = _time_run(
                                    timestamp_func, func, array, array[0], loops=loops
                                )
                            else:
                                raw_timings = _time_run(
                                    timestamp_func, func, array, array, loops=loops
                                )

                        # Analyze the raw result data.
                        analysis = _analyze_results(raw_timings)

                        # If robust stats are enabled, print the whole result so we can see what it looks like.
                        if enable_bench_estimators:
                            print(repr(analysis))

                        ds[col][row] = analysis.median_ns
                        row = row + 1
                        # end of thread loop
                    # end of func loop
                col += 1
                # end of asize loop
            # end of dtype loop
        dscount += 1
        # end of ds loop

    rc.ThreadingMode(savemode)
    rc.SetThreadWakeUp(savethreads)

    if not recycle:
        FA._RON()

    if ratio:
        # return a ratio
        return ds_np / ds_fa

    ms = Multiset({"FastA": ds_fa, "Numpy": ds_np})
    ms.label_set_names(["Thread", "Func"])
    return ms
