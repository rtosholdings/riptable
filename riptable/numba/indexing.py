"""
Indexing-related helper functions for use when implementing other numba-based functions.
"""
__all__ = ["deref_idx", "scalar_or_lookup"]

from typing import Optional

import numpy as np
import numpy.typing as npt
import numba as nb


@nb.generated_jit(nopython=True)
def deref_idx(deref: Optional[np.ndarray], idx: np.integer):
    """
    A numba function for turning an 'indirect' index into a true index if a deference idx exists.

    This main use of this function is to seamlessly deal with the existence/non-existence
    of ``iGroup`` (from a `Grouping` object).

    - When writing a function to operate on a single array, there are no groups. The data is contiguous,
      so there is no ``iGroup`` and an index doesn't need to be 'dereferenced' to get a 'real' index.
    - When writing a function to operate over `Grouping` data, we need to use `Grouping.igroup` to turn
      a 0-based index (scalar) within a particular group's data into an 0-based index within the larger
      array whose shape matches the `Grouping` object itself.

    Parameters
    ----------
    deref: None or Array
        an array to use for dereferencing or None.
    idx: Integer
        the index to grab (or to be returned if deref is None).

    Returns
    -------
    int
        `idx` if `deref` is ``None``; otherwise ``deref[idx]``.
    """
    if isinstance(deref, nb.types.NoneType):
        return lambda deref, idx: idx
    else:
        return lambda deref, idx: deref[idx]


@nb.generated_jit(nopython=True)
def scalar_or_lookup(val: np.ndarray, idx: np.integer):
    """
    Allows a numba-based function to accept either a scalar value or an array for some parameter.
    Scalars are passed through, but if an array is provided, the array element at the specified index is returned.

    For example, in a numba-based function operating over grouped data (with a `Grouping`),
    the function could accept a scalar parameter; by using `scalar_or_lookup`, the code can
    easily accept either a scalar (to be applied to all groups) or an array containing a specific
    value for each group.

    Parameters
    ----------
    val
    idx

    Returns
    -------
    retval
    """
    if isinstance(val, nb.types.Array):
        return lambda val, idx: val[idx]
    else:
        return lambda val, idx: val
