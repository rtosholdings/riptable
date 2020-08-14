"""
Functions for generating random data for use in benchmarking.
"""
__all__ = ["rand_array", "rand_dataset", "rand_fancyindex", "rand_keyarray"]

from typing import List, Optional, Union
import numpy as np

from ..rt_categorical import Categorical
from ..rt_dataset import Dataset
from ..rt_fastarray import FastArray
from ..rt_numpy import arange, putmask

def check_params(dtype, invalid_ratio):
    if not isinstance(dtype, np.dtype):
        raise TypeError(
            f"The argument provided for the `dtype` parameter has the type '{type(dtype)}' but only instances of numpy.dtype are allowed."
        )
    elif invalid_ratio is not None:
        if 0.0 <= invalid_ratio <= 1.0:
            pass
        else:
            raise ValueError(
                f"Invalid value specified for `invalid_ratio`: {invalid_ratio}"
            )
def rand_fancyindex(
    rng: np.random.Generator,
    index_length: int,
    dtype: np.dtype,
    source_arr_len: int,
    invalid_ratio: Optional[float] = None,
) -> np.ndarray:
    """Create a random fancy index with the specified length and dtype."""
    check_params(dtype, invalid_ratio)
    if dtype.kind not in "iu":  # TODO: Also support floats, since mbget allows that
        raise ValueError(
            f"Only integer dtypes are currently supported by this method. dtype={dtype.name}"
        )

    # Generate the fancy index from the uniform integer distribution.
    fancyindex = FastArray(
        rng.integers(0, source_arr_len, size=index_length, dtype=dtype)
    )

    # If the fancy index should have some invalids/NA values, add those in now.
    if invalid_ratio is not None and invalid_ratio > 0.0:
        # TODO: Also add in some out-of-bounds accesses (and not just invalid/NA values) here?
        invalid_outcomes = FastArray(rng.random(size=index_length))
        putmask(fancyindex, invalid_outcomes < invalid_ratio, fancyindex.inv)

    return fancyindex


def rand_array(rng: np.random.Generator, length: int, dtype: np.dtype, invalid_ratio: Optional[float] = None) -> np.ndarray:
    # TODO: Implement a flag that controls whether invalid values are included in the array? Or (instead) an invalid_ratio parameter like our other functions?
    check_params(dtype, invalid_ratio)

    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        arr = FastArray(rng.integers(info.min, info.max, size=length, dtype=dtype))

    elif dtype.kind == "f":
        # PERF: Use an FMA function here if we ever implement one
        arr = (FastArray(rng.random(size=length, dtype=dtype)) * 1e10) - 0.5e10

    elif dtype.kind == "S":
        # Generate integers in the upper ASCII range, then use a view to expose those
        # values as fixed-length ASCII strings.
        # TODO: Support other character ranges (lower-range ASCII 0-127, full ASCII 0-255, lowercase+uppercase+digits).
        arr = FastArray(rng.integers(
            65, 90, size=length * dtype.itemsize, dtype=np.int8, endpoint=True
        ).view(dtype))

    elif dtype.kind == "U":
        # Generate integers in the upper ASCII range.
        # TODO: Support other character ranges (lower-range ASCII 0-127, full ASCII 0-255, lowercase+uppercase+digits, Unicode chars >255).
        arr = FastArray(rng.integers(
            65, 90, size=length * (dtype.itemsize // 4), dtype=np.int32, endpoint=True
        ).view(dtype))

    else:
        # TODO: Handle other dtypes
        raise NotImplementedError(
            f"The dtype {dtype} is not yet supported by this function."
        )

     # If the fancy index should have some invalids/NA values, add those in now.
    if invalid_ratio is not None and invalid_ratio > 0.0:
        # TODO: Also add in some out-of-bounds accesses (and not just invalid/NA values) here?
        invalid_outcomes = FastArray(rng.random(size=length))
        putmask(arr, invalid_outcomes < invalid_ratio, arr.inv)

    return arr

def rand_keyarray(
    rng: np.random.Generator,
    length: int,
    dtype: np.dtype,
    unique_count: int,
    invalid_ratio: Optional[float] = None,
) -> np.ndarray:
    """
    Generate a random array representing an array to be used as an input to a set-like
    operation (ismember, groupbyhash, etc.)

    Parameters
    ----------
    rng : np.random.Generator
    length : int
    dtype : np.dtype
    unique_count : int
    invalid_ratio : float, optional

    Returns
    -------
    np.ndarray

    Notes
    -----
    TODO: additional, advanced options:
      * different distributions (e.g. rng.zipf()) for key multiplicity (i.e. when creating the fancy index we use to
        fetch from the unique values array).
      * key clustering (i.e. are keys already more or less occurring near each other, or are they all spread out?)
      * key dispersion -- if the unique keys are created like arange(1, 100), does grouping/merge go any
        faster than if the keys are created like arange(1, 1000, 10) or arange(1, 1_000_000_000, 10_000_000_000)?
        Make sure to control for dtype to ensure that's the same for all cases tested here.
        This could be used to diagnose issues with hashing/grouping implementations.
      * iKey sortedness
    """
    if not isinstance(dtype, np.dtype):
        raise TypeError(
            f"The argument provided for the `dtype` parameter has the type '{type(dtype)}' but only instances of numpy.dtype are allowed."
        )

    # Generate array of unique values.
    if dtype.kind in "iu":
        # TODO: Support non-consecutive values here
        unique_values = arange(0, unique_count, dtype=dtype)

    elif dtype.kind == "S":
        # Generate integers in the upper ASCII range, then use a view to expose those
        # values as fixed-length ASCII strings.
        # TODO: Support other character ranges (lower-range ASCII 0-127, full ASCII 0-255, lowercase+uppercase+digits).
        # TODO: Use uniques() or similar to make sure all of the values generated here are actually unique.
        unique_values = rng.integers(
            65, 90, size=unique_count * dtype.itemsize, dtype=np.int8, endpoint=True
        ).view(dtype)

    elif dtype.kind == "U":
        # Generate integers in the upper ASCII range.
        # TODO: Support other character ranges (lower-range ASCII 0-127, full ASCII 0-255, lowercase+uppercase+digits, Unicode chars >255).
        # TODO: Use uniques() or similar to make sure all of the values generated here are actually unique.
        unique_values = rng.integers(
            65,
            90,
            size=unique_count * (dtype.itemsize // 4),
            dtype=np.int32,
            endpoint=True,
        ).view(dtype)

    else:
        # TODO: Support other dtypes, e.g. floats
        raise NotImplementedError(f"dtype {dtype} not yet supported by this function.")

    # Generate a fancy index into the array of unique values.
    fancyindex = rand_fancyindex(
        rng,
        length,
        np.min_scalar_type(length),
        unique_count,
        invalid_ratio=invalid_ratio,
    )

    # Use the fancy index to select elements from the array of unique values, then return the result.
    return unique_values[fancyindex]


def rand_multikeyarray(
    rng: np.random.Generator, length: int, dtypes: List[np.dtype]
) -> List[np.ndarray]:
    # TODO: Implement a function similar to rand_keyarray but which produces a multikey (a tuple or list of arrays)
    raise NotImplementedError()


def rand_dataset(
    rowcount: int,
    rng: np.random.Generator,
    # Can the next few params be combined into some "KeySpec" NamedTuple to help simplify passing them?
    # TODO: Allow the key name(s) to be specified?
    unique_key_count: Union[int, List[int]],
    key_dtype: Optional[Union[np.dtype, List[np.dtype]]] = None,
    make_cat: Union[bool, List[bool]] = True,
    xtra_col_dtypes: Optional[List[np.dtype]] = None,
) -> Dataset:
    """
    Create a random Dataset with the specified shape and key(s).

    Parameters
    ----------
    rowcount
    rng
    unique_key_count
    key_dtype
        When None, defaults to np.int64.
    make_cat
    xtra_col_dtypes

    Returns
    -------
    Dataset
    """
    if key_dtype is None:
        key_dtype = np.dtype(np.int64)
    elif isinstance(key_dtype, list):
        for ty in key_dtype:
            if not isinstance(ty, np.dtype):
                raise TypeError(
                    f"The list provided for the `key_dtype` parameter contains an instance of type '{type(ty)}' but only instances of numpy.dtype are allowed."
                )

    elif not isinstance(key_dtype, np.dtype):
        raise TypeError(
            f"The argument provided for the `key_dtype` parameter has the type '{type(key_dtype)}' but only instances of numpy.dtype are allowed."
        )

    ds = Dataset()

    # Create the key column.
    # TODO: Offer an option to use other key distributions like rng.zipf(). Or allow the distribution to be specified per column so we can e.g. have one uniform column and one zipf column, then combine them into a multikey.
    keydata = (
        rand_keyarray(rng, rowcount, key_dtype, unique_key_count)
        if isinstance(key_dtype, np.dtype)
        else rand_multikeyarray(rng, rowcount, key_dtype)
    )

    # Create a Categorical from this data if specified.
    if make_cat:
        keydata = Categorical(keydata)

    # TODO: If keydata is a list here, we need to assign the columns into the Dataset individually.
    ds["key"] = keydata

    # Add a column with the row index (just so there's at least one additional column besides the key).
    ds["rowidx"] = arange(rowcount)

    # Add additional columns with random data if requested.
    if xtra_col_dtypes:
        raise NotImplementedError()

    return ds
