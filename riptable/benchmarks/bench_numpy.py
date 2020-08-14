# -*- coding: utf-8 -*-
import numpy as np
from itertools import product
from typing import Generator, List, Optional
from .benchmark import benchmark
from .runner import benchmark as bench, create_comparison_dataset
from ..rt_multiset import Multiset
from ..rt_fastarray import FastArray
from ..rt_numpy import (
    all,
    any,
    arange,
    argmax,
    argmin,
    argsort,
    isnan,
    lexsort,
    max,
    mean,
    median,
    min,
    minimum,
    nanargmax,
    nanargmin,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanpercentile,
    nansum,
    nanstd,
    nanvar,
    percentile,
    std,
    sum,
    trunc,
    var,
)


DEFAULT_ARRAY_SIZES: List[int] = [100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]
DEFAULT_DTYPES: List[type] = [np.int32, np.float32]
DEFAULT_THREADS: List[int] = [1, 2, 4, 8]
DEFAULT_RECYCLER: List[bool] = [True, False]
DEFAULT_REDUCE_FUNCS: List[callable] = [np.sum, np.min, np.std]
DEFAULT_BINARY_FUNCS: List[callable] = [np.add, np.minimum, np.true_divide]
DEFAULT_UNARY_FUNCS: List[callable] = [np.isnan, np.trunc, np.sqrt]


def _gen_arange(
    asizes: Optional[List[int]] = None,
    dtypes: Optional[List[type]] = None,
    nan_indexes: Optional[List[int]] = None,
) -> Generator[np.ndarray, None, None]:
    """_gen_arange is a generator that yields an arrays of cross product of
    array sizes, array data types, and possibly nan-specified indexes.
    The array data is monotonically increasing."""
    if asizes is None:
        asizes = DEFAULT_ARRAY_SIZES
    if dtypes is None:
        dtypes = DEFAULT_DTYPES
    if nan_indexes is None:
        nan_indexes = [None]
    for asize, dtype, nan_index in product(asizes, dtypes, nan_indexes):
        arr = np.arange(asize, dtype=dtype)
        if nan_index is not None and np.dtype(dtype).type in [np.float32, np.float64]:
            arr[nan_index] = np.nan
        yield arr


def gen_arange(
    asizes: Optional[List[int]] = None,
    dtypes: Optional[List[type]] = None,
    nan_indexes: Optional[List[int]] = None,
) -> Generator[np.ndarray, None, None]:
    """gen_arange is a generator of numpy arrays composed of the cross product of
    array sizes, array data types, and possibly nan-specified indexes.
    The array data is monotonically increasing."""
    for arr in _gen_arange(asizes, dtypes, nan_indexes):
        yield arr


def gen_farange(
    asizes: Optional[List[int]] = None,
    dtypes: Optional[List[type]] = None,
    nan_indexes: Optional[List[int]] = None,
) -> Generator[np.ndarray, None, None]:
    """gen_farange is a generator of riptable FastArray composed of the cross product of
    array sizes, array data types, and possibly nan-specified indexes.
    The array data is monotonically increasing."""
    for arr in _gen_arange(asizes, dtypes, nan_indexes):
        yield arr.view(FastArray)


def gen_arange_reverse(
    asizes: Optional[List[int]] = None,
    dtypes: Optional[List[type]] = None,
    nan_indexes: Optional[List[int]] = None,
) -> Generator[np.ndarray, None, None]:
    """A generator of numpy.flip of ``gen_arange``."""
    for arr in _gen_arange(asizes, dtypes, nan_indexes):
        yield np.flip(arr)


def gen_farange_reverse(
    asizes: Optional[List[int]] = None,
    dtypes: Optional[List[type]] = None,
    nan_indexes: Optional[List[int]] = None,
) -> Generator[np.ndarray, None, None]:
    """A generator of numpy.flip of ``gen_farange``."""
    for arr in gen_arange(asizes, dtypes, nan_indexes):
        yield np.flip(arr).view(FastArray)


# ----------------------------------------------------
def benchmark_reduce(
    funcs=DEFAULT_REDUCE_FUNCS,
    dtypes=DEFAULT_DTYPES,
    asizes=DEFAULT_ARRAY_SIZES,
    threads=DEFAULT_THREADS,
    loops=10,
    ratio=True,
    recycle=True,
    scalar=False,
):
    return benchmark(
        funcs=funcs,
        dtypes=dtypes,
        asizes=asizes,
        threads=threads,
        loops=loops,
        ratio=ratio,
        recycle=recycle,
        funcargs=1,
        scalar=scalar,
    )


# ----------------------------------------------------
def benchmark_binary(
    funcs=DEFAULT_BINARY_FUNCS,
    dtypes=DEFAULT_DTYPES,
    asizes=DEFAULT_ARRAY_SIZES,
    threads=DEFAULT_THREADS,
    loops=10,
    ratio=True,
    recycle=True,
    scalar=False,
):
    return benchmark(
        funcs=funcs,
        dtypes=dtypes,
        asizes=asizes,
        threads=threads,
        loops=loops,
        ratio=ratio,
        recycle=recycle,
        funcargs=2,
        scalar=scalar,
    )


# ----------------------------------------------------
def benchmark_unary(
    funcs=DEFAULT_UNARY_FUNCS,
    dtypes=DEFAULT_DTYPES,
    asizes=DEFAULT_ARRAY_SIZES,
    threads=DEFAULT_THREADS,
    loops=10,
    ratio=True,
    recycle=True,
    scalar=False,
):

    return benchmark(
        funcs=funcs,
        dtypes=dtypes,
        asizes=asizes,
        threads=threads,
        loops=loops,
        ratio=ratio,
        recycle=recycle,
        funcargs=1,
        scalar=scalar,
    )


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS})
def bench_rt_all(farr):
    all(farr)


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange()})
def bench_np_all(arr):
    np.all(arr)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "farr": gen_farange(dtypes=[np.float32, np.float64], nan_indexes=[0, -1]),
        "thread_count": DEFAULT_THREADS,
    }
)
def bench_rt_all_with_nan(farr):
    all(farr)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "arr": gen_arange(dtypes=[np.float32, np.float64], nan_indexes=[0, -1]),
    }
)
def bench_np_all_with_nan(arr):
    np.all(arr)


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS})
def bench_rt_any(farr):
    any(farr)


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange()})
def bench_np_any(arr):
    np.any(arr)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "farr": gen_farange(dtypes=[np.float32, np.float64], nan_indexes=[0, -1]),
        "thread_count": DEFAULT_THREADS,
    }
)
def bench_rt_any_with_nan(farr):
    any(farr)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "arr": gen_arange(dtypes=[np.float32, np.float64], nan_indexes=[0, -1]),
    }
)
def bench_np_any_with_nan(arr):
    np.any(arr)


# ----------------------------------------------------
# 20200406 rt.arange punts to np.arange - no need to compare benchmarks
@bench(benchmark_params={"sz": DEFAULT_ARRAY_SIZES, "thread_count": DEFAULT_THREADS})
def bench_rt_arange(sz):
    arange(sz)


# ----------------------------------------------------
@bench(benchmark_params={"sz": DEFAULT_ARRAY_SIZES})
def bench_np_arange(sz):
    np.arange(sz)


# ----------------------------------------------------
# 20200406 rt.argsort punts to np.argsort - no need to compare benchmarks
@bench(
    benchmark_params={
        "farr": gen_farange(),
        "kind": ["quicksort", "mergesort", "heapsort", "stable"],
        "thread_count": DEFAULT_THREADS,
    }
)
def bench_rt_argsort(farr, kind):
    argsort(farr, kind=kind)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "arr": gen_arange(),
        "kind": ["quicksort", "mergesort", "heapsort", "stable"],
    }
)
def bench_np_argsort(arr, kind):
    np.argsort(arr, kind)


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS})
def bench_rt_argmax(farr):
    argmax(farr)


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS})
def bench_rt_nanargmax(farr):
    nanargmax(farr)


# ----------------------------------------------------
@bench(
    benchmark_params={"arr": gen_arange(),}
)
def bench_np_argmax(arr):
    np.argmax(arr)


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS})
def bench_rt_argmin(farr):
    argmin(farr)


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS})
def bench_rt_nanargmin(farr):
    nanargmin(farr)


# ----------------------------------------------------
@bench(
    benchmark_params={"arr": gen_arange(),}
)
def bench_np_argmin(arr):
    np.argmin(arr)


# ----------------------------------------------------
# TODO different data types and different ordering (e.g., random, reversed, ordered)
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS})
def bench_rt_lexsort_ordered(farr):
    lexsort((farr, farr))


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange_reverse()})
def bench_np_lexsort_ordered(arr):
    np.lexsort((arr, arr))


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS})
def bench_rt_lexsort_reversed(farr):
    lexsort((farr, farr))


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange_reverse()})
def bench_np_lexsort_reversed(arr):
    np.lexsort((arr, arr))


# ----------------------------------------------------
# TODO vary percentile and interpolation
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS,},)
def bench_rt_percentile(farr):
    percentile(farr, 50)


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS,},)
def bench_rt_nanpercentile(farr):
    nanpercentile(farr, 50)


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange()},)
def bench_np_percentile(arr):
    np.percentile(arr, 50)


# ----------------------------------------------------
@bench(
    benchmark_params={
        # benchmark function parameters
        "farr": gen_farange(),
        # riptable settings
        "thread_count": DEFAULT_THREADS,
        "recycler": DEFAULT_RECYCLER,
    },
    # other benchmark paramters
    benchmark_iterations=5,
)
def bench_rt_sum(farr):
    sum(farr)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "farr": gen_farange(),
        "thread_count": DEFAULT_THREADS,
        "recycler": DEFAULT_RECYCLER,
    },
)
def bench_rt_nansum(farr):
    nansum(farr)


# ----------------------------------------------------
@bench(
    benchmark_params={"arr": gen_arange()}, benchmark_iterations=5,
)
def bench_np_sum(arr):
    np.sum(arr)


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS,},)
def bench_rt_max(farr):
    max(farr)


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS,},)
def bench_rt_nanmax(farr):
    nanmax(farr)


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange()},)
def bench_np_max(arr):
    np.max(arr)


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS,},)
def bench_rt_mean(farr):
    mean(farr)


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS,},)
def bench_rt_nanmean(farr):
    nanmean(farr)


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange()},)
def bench_np_mean(arr):
    np.mean(arr)


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS,},)
def bench_rt_median(farr):
    median(farr)


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS,},)
def bench_rt_nanmedian(farr):
    nanmedian(farr)


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange()},)
def bench_np_median(arr):
    np.median(arr)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "farr": gen_farange(),
        "thread_count": DEFAULT_THREADS,
        "recycler": DEFAULT_RECYCLER,
    },
)
def bench_rt_min(farr):
    min(farr)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "farr": gen_farange(),
        "thread_count": DEFAULT_THREADS,
        "recycler": DEFAULT_RECYCLER,
    },
)
def bench_rt_nanmin(farr):
    nanmin(farr)


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange()},)
def bench_np_min(arr):
    np.min(arr)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "farr": gen_farange(),
        "thread_count": DEFAULT_THREADS,
        "recycler": DEFAULT_RECYCLER,
    },
)
def bench_rt_std(farr):
    std(farr)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "farr": gen_farange(),
        "thread_count": DEFAULT_THREADS,
        "recycler": DEFAULT_RECYCLER,
    },
)
def bench_rt_nanstd(farr):
    nanstd(farr)


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange()},)
def bench_np_std(arr):
    np.std(arr)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "farr": gen_farange(),
        "thread_count": DEFAULT_THREADS,
        "recycler": DEFAULT_RECYCLER,
    },
)
def bench_rt_add(farr):
    # N.B, 20200401 - riptable does not have an implementation for `add` so use the Numpy
    # implementation with FastArray arguments.
    np.add(farr, farr)


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange()},)
def bench_np_add(arr):
    np.add(arr, arr)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "farr": gen_farange(),
        "thread_count": DEFAULT_THREADS,
        "recycler": DEFAULT_RECYCLER,
    },
)
def bench_rt_minimum(farr):
    minimum(farr, farr)


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange()},)
def bench_np_minimum(arr):
    np.minimum(arr, arr)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "farr": gen_farange(),
        "thread_count": DEFAULT_THREADS,
        "recycler": DEFAULT_RECYCLER,
    },
)
def bench_rt_true_divide(farr):
    # N.B, 20200401 - riptable does not have an implementation for `true_divide` so use the Numpy
    # implementation with FastArray arguments.
    np.true_divide(farr, farr)


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange()},)
def bench_np_true_divide(arr):
    np.true_divide(arr, arr)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "farr": gen_farange(),
        "thread_count": DEFAULT_THREADS,
        "recycler": DEFAULT_RECYCLER,
    },
)
def bench_rt_isnan(farr):
    isnan(farr)


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange()},)
def bench_np_isnan(arr):
    np.isnan(arr)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "farr": gen_farange(),
        "thread_count": DEFAULT_THREADS,
        "recycler": DEFAULT_RECYCLER,
    },
)
def bench_rt_trunc(farr):
    trunc(farr)


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange()},)
def bench_np_trunc(arr):
    np.trunc(arr)


# ----------------------------------------------------
@bench(
    benchmark_params={
        "farr": gen_farange(),
        "thread_count": DEFAULT_THREADS,
        "recycler": DEFAULT_RECYCLER,
    },
)
def bench_rt_sqrt(farr):
    # N.B, 20200401 - riptable does not have an implementation for `sqrt` so use the Numpy
    # implementation with FastArray arguments.
    np.sqrt(farr)


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange()},)
def bench_np_sqrt(arr):
    np.sqrt(arr)


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS,},)
def bench_rt_var(farr):
    var(farr)


# ----------------------------------------------------
@bench(benchmark_params={"farr": gen_farange(), "thread_count": DEFAULT_THREADS,},)
def bench_rt_nanvar(farr):
    nanvar(farr)


# ----------------------------------------------------
@bench(benchmark_params={"arr": gen_arange()},)
def bench_np_var(arr):
    np.var(arr)


# ----------------------------------------------------
def compare_argmin():
    return create_comparison_dataset(
        {
            "rt_argmin": bench_rt_argmin(),
            "rt_nanargmin": bench_rt_nanargmin(),
            "np_argmin": bench_np_argmin(),
        }
    )


# ----------------------------------------------------
def compare_argmax():
    return create_comparison_dataset(
        {
            "rt_argmax": bench_rt_argmax(),
            "rt_nanargmax": bench_rt_nanargmax(),
            "np_argmax": bench_np_argmax(),
        }
    )


# ----------------------------------------------------
def compare_all():
    return create_comparison_dataset(
        {"rt_all": bench_rt_all(), "np_all": bench_np_all(),}
    )


# ----------------------------------------------------
def compare_any():
    return create_comparison_dataset(
        {"rt_any": bench_rt_any(), "np_any": bench_np_any(),}
    )


# ----------------------------------------------------
def compare_all_with_nan():
    return create_comparison_dataset(
        {"rt_all": bench_rt_all_with_nan(), "np_all": bench_np_all_with_nan(),}
    )


# ----------------------------------------------------
def compare_any_with_nan():
    return create_comparison_dataset(
        {"rt_any": bench_rt_any_with_nan(), "np_any": bench_np_any_with_nan(),}
    )


# ----------------------------------------------------
def compare_lexsort():
    return create_comparison_dataset(
        {
            "rt_lexsort_ordered": bench_rt_lexsort_ordered(),
            "np_lexsort_ordered": bench_np_lexsort_ordered(),
            "rt_lexsort_reversed": bench_rt_lexsort_reversed(),
            "np_lexsort_reversed": bench_np_lexsort_reversed(),
        }
    )


# ----------------------------------------------------
def compare_max():
    return create_comparison_dataset(
        {
            "rt_max": bench_rt_max(),
            "rt_nanmax": bench_rt_nanmax(),
            "np_max": bench_np_max(),
        }
    )


# ----------------------------------------------------
def compare_mean():
    return create_comparison_dataset(
        {
            "rt_min": bench_rt_mean(),
            "rt_nanmin": bench_rt_nanmean(),
            "np_min": bench_np_mean(),
        }
    )


# ----------------------------------------------------
def compare_median():
    return create_comparison_dataset(
        {
            "rt_median": bench_rt_median(),
            "rt_nanmedian": bench_rt_nanmedian(),
            "np_median": bench_np_median(),
        }
    )


# ----------------------------------------------------
def compare_min():
    return create_comparison_dataset(
        {
            "rt_min": bench_rt_min(),
            "rt_nanmin": bench_rt_nanmin(),
            "np_min": bench_np_min(),
        }
    )


# ----------------------------------------------------
def compare_percentile():
    return create_comparison_dataset(
        {
            "rt_percentile": bench_rt_percentile(),
            "rt_nanpercentile": bench_rt_nanpercentile(),
            "np_percentile": bench_np_percentile(),
        }
    )


# ----------------------------------------------------
def compare_std():
    return create_comparison_dataset(
        {
            "rt_std": bench_rt_std(),
            "rt_nanstd": bench_rt_nanstd(),
            "np_std": bench_np_std(),
        }
    )


# ----------------------------------------------------
def compare_sum():
    return create_comparison_dataset(
        {
            "rt_sum": bench_rt_sum(),
            "rt_nansum": bench_rt_nansum(),
            "np_sum": bench_np_sum(),
        }
    )


# ----------------------------------------------------
def compare_add():
    return create_comparison_dataset(
        {"rt_add": bench_rt_add(), "np_add": bench_np_add(),}
    )


# ----------------------------------------------------
def compare_minimum():
    return create_comparison_dataset(
        {"rt_minimum": bench_rt_minimum(), "np_minimum": bench_np_minimum(),}
    )


# ----------------------------------------------------
def compare_true_divide():
    return create_comparison_dataset(
        {
            "rt_true_divide": bench_rt_true_divide(),
            "np_true_divide": bench_np_true_divide(),
        }
    )


# ----------------------------------------------------
def compare_isnan():
    return create_comparison_dataset(
        {"rt_isnan": bench_rt_isnan(), "np_isnan": bench_np_isnan(),}
    )


# ----------------------------------------------------
def compare_trunc():
    return create_comparison_dataset(
        {"rt_trunc": bench_rt_trunc(), "np_trunc": bench_np_trunc(),}
    )


# ----------------------------------------------------
def compare_sqrt():
    return create_comparison_dataset(
        {"rt_sqrt": bench_rt_sqrt(), "np_sqrt": bench_np_sqrt(),}
    )


# ----------------------------------------------------
def compare_var():
    return create_comparison_dataset(
        {
            "rt_var": bench_rt_var(),
            "rt_nanvar": bench_rt_nanvar(),
            "np_var": bench_np_var(),
        }
    )


# ----------------------------------------------------
def compare_numpy_reduce():
    return Multiset({"min": compare_min(), "std": compare_std(), "sum": compare_sum(),})


# ----------------------------------------------------
def compare_numpy_binary():
    return Multiset(
        {
            "add": compare_add(),
            "minimum": compare_minimum(),
            "true_divide": compare_true_divide(),
        }
    )


# ----------------------------------------------------
def compare_numpy_unary():
    return Multiset(
        {"isnan": compare_isnan(), "trunc": compare_trunc(), "sqrt": compare_sqrt(),}
    )
