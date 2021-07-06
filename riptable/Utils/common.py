"""
Logic / helper functions used throughout the riptable benchmark suite.
"""
import numbers
from typing import Mapping, List, Tuple, Optional
import numpy as np
import riptable as rt

try:
    from functools import cached_property
except ImportError:
    cached_property = property


_SEED = 1234
"""
Value used to seed random number generators so we get "random" data for
the benchmarks while also allowing the benchmarks to be repeatable.
"""

_INT_16_MAX = np.iinfo(np.int16).max
_INT_32_MAX = np.iinfo(np.int32).max
_INT_MAX = np.int64(1 << 33)  # 8_589_934_592


def trial_size(low=250, high=_INT_16_MAX, scale_factor=2):
    """Generates a list of input sizes to benchmark against."""
    lst, i = [low], 0
    while lst[i] <= high:
        i += 1
        lst.append(lst[i - 1] * scale_factor)
    return lst


_DOUBLING_TRIAL_16 = trial_size(high=_INT_16_MAX, scale_factor=2)
_DOUBLING_TRIAL_32 = trial_size(high=_INT_32_MAX, scale_factor=2)
_DOUBLING_TRIAL_MAX = trial_size(high=_INT_MAX, scale_factor=2)


_dtypes_by_group = {k: [np.dtype(x) for x in v] for (k, v) in np.typecodes.items()}

# Add a few additional categories to our dtypes dictionary for convenience.
_dtypes_by_group["Boolean"] = [np.dtype("?")]
_dtypes_by_group["StandardFloat"] = [
    np.dtype(x) for x in np.typecodes["Float"] if np.dtype(x).itemsize > 2
]
_dtypes_by_group["RiptableNumeric"] = (
    _dtypes_by_group["AllInteger"] + _dtypes_by_group["StandardFloat"]
)

dtypes_by_group: Mapping[str, List[np.dtype]] = _dtypes_by_group
"""
``np.typecodes`` but the value for each entry is a list containing the
dtypes corresponding to the typecode(s) for the original entry.

See Also
--------
riptable.rt_enum.NumpyCharTypes
"""


def zeros_eager(shape, dtype: np.dtype) -> np.ndarray:
    # Allocate a zeroed-out array but use np.full rather than np.zeros
    # as the latter doesn't actually touch any of the data pages during
    # allocation; that leads to page faults occurring during the benchmark.
    return np.full(shape, 0, dtype=dtype)


def integer_valid_range(dtype: np.dtype) -> Tuple[int, int]:
    """
    Given an integer dtype, return the range [lo, hi] of valid values
    representable by that dtype.

    Parameters
    ----------
    dtype : data-type
        An integer dtype.
    """
    if not issubclass(dtype.type, numbers.Integral):
        raise ValueError(f"'{dtype}' is not an integral/integer dtype.")

    # Handle bool specially
    if np.issubdtype(dtype, bool):
        return (0, 1)

    # Get the info for this integer dtype
    dtype_info = np.iinfo(dtype)

    # Get the invalid value for this integer dtype
    try:
        invalid_value = rt.INVALID_DICT[dtype.num]
    except KeyError:
        # This type doesn't have an invalid, so return it's full range.
        return (dtype_info.min, dtype_info.max)

    # The invalid value should be either the min or max value of the integer.
    # Determine which one it is, increment/decrement the integer range and return that.
    if invalid_value == dtype_info.min:
        return (dtype_info.min + 1, dtype_info.max)
    elif invalid_value == dtype_info.max:
        return (dtype_info.min, dtype_info.max - 1)
    else:
        raise ValueError(
            f"Unable to determine the valid range for dtype '{dtype}'. invalid_value={invalid_value}\tiinfo={dtype_info}"
        )


def integer_range(dtype: np.dtype, include_invalid: bool = False) -> Tuple[int, int]:
    """
    Given an integer dtype, return the range [lo, hi] of values representable
    by the type, optionally excluding the invalid value (if any) for the type.
    """
    if not issubclass(dtype.type, numbers.Integral):
        raise ValueError(f"'{dtype}' is not an integral/integer dtype.")

    # Handle bool specially
    if np.issubdtype(dtype, bool):
        return (0, 1)

    # Determine the range based on whether the caller wants to include
    # or exclude the invalid value for this dtype.
    if include_invalid:
        dtype_info = np.iinfo(dtype)
        return (dtype_info.min, dtype_info.max)
    else:
        return integer_valid_range(dtype)


def rand_integers(
    gen: np.random.Generator,
    size: Optional[int] = None,
    dtype: np.dtype = np.int64,
    include_invalid: bool = False,
) -> np.ndarray:
    """
    Generate a random array of integers with the specified length and dtype.

    The elements of the array will span the representable range of the dtype,
    optionally including the 'invalid' value for the type. The elements of the
    array are drawn from the 'discrete uniform' distribution.
    """
    # Determine the range for the dtype.
    lo, hi = integer_range(dtype, include_invalid)
    return gen.integers(lo, hi, size, dtype=dtype, endpoint=True)


def rand_floats(
    gen: np.random.Generator,
    low: float = 0.0,
    high: float = 1.0,
    size: Optional[int] = None,
    dtype: np.dtype = np.float64,
    include_invalid: bool = False,
) -> np.ndarray:
    """
    Generate a random array of floating-point values with the specified length and dtype.

    The elements of the array are drawn from the uniform distribution over the
    range ``[low, high)``.
    """
    # Generate a random array for the given floating-point type.
    arr = gen.uniform(low=low, high=high, size=size).astype(dtype)

    # If we're including invalid values (NaN for floating-point types), draw a random integer
    # indicating how many elements we'll set to NaN; then generate a random integer array of
    # that length whose elements will be a fancy index we'll use to assign NaNs into the generated
    # floating-point array.
    # NOTE: The nancount we generate is approximate because we'll don't enforce that all the
    #       elements of the fancy index are unique.
    if include_invalid:
        nancount = gen.integers(0, size, endpoint=True)
        nan_indices = gen.integers(0, size, size=nancount)
        arr[nan_indices] = np.nan

    return arr


# TODO: Define functions for determining things like array sizes to parameterize benchmarks with.
#       If possible, utilize either the asv API or the general Python API to determine how much memory
#       the current machine has installed and use that to determine the upper bound for these sizes;
#       the exact value will depend on things like the dtype, how many arrays are needed for that particular
#       benchmark, etc.


# TODO: Define functions for generating random inputs (using the numpy.random.Generator API)
#       For testing any set-based functionality (including ismember, Grouping, merge, key-alignment)
#       we'll want a way to generate arrays where the elements' multiplicities are drawn from
#       different distributions (e.g. uniform, zipf, geometric) to more-closely model real-world datasets.
