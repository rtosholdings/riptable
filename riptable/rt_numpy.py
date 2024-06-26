from __future__ import annotations
import numbers

__all__ = [
    # types listed first
    "int16",
    "int32",
    "int64",
    "int8",
    "int0",
    "uint0",
    "bool_",
    "bytes_",
    "str_",
    "float32",
    "float64",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    # functions
    "absolute",
    "abs",
    "all",
    "any",
    "arange",
    "argsort",
    "asanyarray",
    "asarray",
    "assoc_copy",
    "assoc_index",
    "bincount",
    "bitcount",
    "bool_to_fancy",
    "cat2keys",
    "ceil",
    "combine2keys",
    "concatenate",
    "crc32c",
    "crc64",
    "cumsum",
    "combine_filter",
    "combine_accum1_filter",
    "combine_accum2_filter",
    "diff",
    "double",
    "empty",
    "empty_like",
    "floor",
    "full",
    "full_like",
    "get_dtype",
    "get_common_dtype",
    "groupby",
    "groupbyhash",
    "groupbylex",
    "groupbypack",
    "hstack",
    "isfinite",
    "isnotfinite",
    "isinf",
    "isnotinf",
    "ismember",
    "isnan",
    "isnanorzero",
    "isnotnan",
    "issorted",
    "interp",
    "interp_extrap",
    "lexsort",
    "logical",
    "log",
    "log10",
    "makeifirst",
    "makeilast",
    "makeinext",
    "makeiprev",
    "max",
    "mean",
    "median",
    "min",
    "min_scalar_type",
    "multikeyhash",
    "minimum",
    "maximum",
    "mask_or",
    "mask_and",
    "mask_xor",
    "mask_andnot",
    "mask_ori",
    "mask_andi",
    "mask_xori",
    "mask_andnoti",
    "nan_to_num",
    "nan_to_zero",
    "nanargmin",
    "nanargmax",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanpercentile",
    "nanstd",
    "nansum",
    "nanvar",
    "ones",
    "ones_like",
    "percentile",
    "putmask",
    "reindex_fast",
    "repeat",
    "reshape",
    "round",
    "searchsorted",
    "_searchsorted",
    "single",
    "sort",
    "sortinplaceindirect",
    "std",
    "sum",
    "tile",
    "transpose",
    "trunc",
    "unique",
    "unique32",
    "var",
    "vstack",
    "where",
    "zeros",
    "zeros_like",
]

import builtins
import inspect
import sys
import warnings
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
import riptide_cpp as rc
from riptide_cpp import LedgerFunction

from .rt_enum import (
    INVALID_DICT,
    MATH_OPERATION,
    REDUCE_FUNCTIONS,
    CategoryMode,
    NumpyCharTypes,
    TypeRegister,
)

if TYPE_CHECKING:
    from .rt_categorical import Categorical
    from .rt_dataset import Dataset
    from .rt_fastarray import FastArray
    from .rt_struct import Struct


ArraysOrDataset = Union[np.ndarray, List[np.ndarray], TypeRegister.Dataset]


def _is_array_container(arg):
    return isinstance(arg, (tuple, list)) and len(arg) and not np.isscalar(arg[0])


def _cast_to_fa(
    arr: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray], Mapping[str, np.ndarray], "Dataset"]
) -> Union["FastArray", List["FastArray"]]:
    """
    Helper for casting array inputs either as single arrays or collections
     of arrays such as list, tuples or even Datasets.
     2-D arrays are treated as lists of 'columns'.
    """
    if _is_array_container(arr):
        arr = type(arr)(map(TypeRegister.FastArray, arr))
    elif hasattr(arr, "items"):
        arr = [TypeRegister.FastArray(v) for k, v in arr.items()]
    elif not np.isscalar(arr):
        arr = TypeRegister.FastArray(arr)
        if arr.ndim == 2:
            arr = [arr[:, i] for i in range(arr.shape[1])]
    return arr


def _args_to_fast_arrays(*arg_names) -> Callable:
    """
    A decorator factory which allows use to cast specified inputs to FastArrays or collections
    thereof.

    Parameters
    ----------
    arg_names: tuple[str]
        Specifies the arguments to which we apply _cast_to_fa
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            signature = inspect.signature(func)
            bound_args = signature.bind(*args, **kwargs)
            for arg in arg_names:
                bound_args.arguments[arg] = _cast_to_fa(bound_args.arguments[arg])
            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator


def min_scalar_type(val, promote_invalid=False, prefer_signed=False):
    """
    For scalar `val`, returns the data type of smallest size and smallest kind
    that can hold its value. If passed a non-scalar ndarray/FastArray,
    returns the `val.dtype` unmodified.

    Parameters
    ----------
    val: scalar or array_like
        The value to get the minimal dtype of.

    promote_invalid: bool
        Whether to promote this value as a valid value of the next larger dtype if it's an riptable invalid sentinel.
        Defaults to False.

    prefer_signed: bool
        Whether to prefer signed type for positive values.
        Defaults to False.

    Returns
    -------
    out: dtype
        The minimal data type.


    Examples
    --------
    >>> rt.min_scalar_type(255)
    dtype('uint8')

    >>> rt.min_scalar_type(255, promote_invalid=True)
    dtype('uint16')

    >>> rt.min_scalar_type(255, promote_invalid=True, prefer_signed=True)
    dtype('int16')

    >>> rt.min_scalar_type(rt.uint64(255))
    dtype('uint8')

    >>> rt.min_scalar_type(3.13)
    dtype('float16')

    >>> rt.min_scalar_type("foo")
    dtype('<U3')
    """
    if not np.isscalar(val) or not isinstance(val, numbers.Number) or isinstance(val, bool):
        return np.min_scalar_type(val)

    if promote_invalid:
        val += 1 if val >= 0 else -1

    if prefer_signed and val >= 0:
        val = -val if val != 0 else -1

    return np.min_scalar_type(val)


# --------------------------------------------------------------
def get_dtype(val) -> np.dtype:
    """
    Return the dtype of an array, list, or builtin int, float, bool, str, bytes.

    Parameters
    ----------
    val
        An object to get the dtype of.

    Returns
    -------
    data-type
        The data-type (dtype) for `val` (if it has a dtype), or a dtype
        compatible with `val`.

    Notes
    -----
    if a python integer, return smallest integer type that can hold this value (always prefers unsigned)
    for a python float, always returns float64
    for a string, will return U or S with size

    TODO: consider pushing down into C++

    Examples
    --------
    >>> get_dtype(10)
    dtype('int32')

    >>> get_dtype(123.45)
    dtype('float64')

    >>> get_dtype('hello')
    dtype('<U5')

    >>> get_dtype(b'hello')
    dtype('S5')
    """
    return val.dtype if hasattr(val, "dtype") else min_scalar_type(val)


# --------------------------------------------------------------
def get_common_dtype(x, y) -> np.dtype:
    """
    Return the dtype of two arrays, or two scalars, or a scalar and an array.

    Will dtype normal python ints to int32 or int64 (not int8 or int16).
    Used in where, put, take, putmask.

    Parameters
    ----------
    x, y : scalar or array_like
        A scalar and/or array to find the common dtype of.

    Returns
    -------
    data-type
        The data type (dtype) common to both `x` and `y`. If the objects don't
        have exactly the same dtype, returns the dtype which both types could
        be implicitly coerced to.

    Examples
    --------
    >>> get_common_type('test','hello')
    dtype('<U5')

    >>> get_common_type(14,'hello')
    dtype('<U16')

    >>> get_common_type(14,b'hello')
    dtype('<S16')

    >>> get_common_type(14, 17)
    dtype('int32')

    >>> get_common_type(arange(10), arange(10.0))
    dtype('float64')

    >>> get_common_type(arange(10).astype(bool), True)
    dtype('bool')
    """
    type1 = get_dtype(x)
    type2 = get_dtype(y)

    # NOTE: find_common_type has a bug where int32 num 7 gets flipped to int32 num 5.
    if type1.num != type2.num:
        common = np.result_type(type1, type2)
    else:
        # for strings and unicode, pick the larger itemsize
        if type1.itemsize >= type2.itemsize:
            common = type1
        else:
            common = type2

    if common.char != "O":
        return common

    # situation where we have a string but find_common_type flips to object too easily
    if type1.char in "SU":
        # get itemsize
        itemsize1 = type1.itemsize
        if type1.char == "U":
            itemsize1 = itemsize1 // 4
        if type2.char in "SU":
            itemsize2 = type2.itemsize
            if type2.char == "U":
                itemsize2 = itemsize2 // 4

            # get the max of the two strings
            maxsize = str(max(itemsize1, itemsize2))
            if type1.char == "U" or type2.char == "U":
                common = np.dtype("U" + maxsize)
            else:
                common = np.dtype("S" + maxsize)

        # 13 is long double
        # 14,15,16 CFLOAT, CDOUBLE, CLONGDOUBLE
        # 17 is object
        # 18 = ASCII string
        # 19 = UNICODE string
        elif type2.num <= 13:
            # handle case where we have an int/float/bool and a string
            if type2.num <= 10:
                maxsize = str(max(itemsize1, 16))
            else:
                maxsize = str(max(itemsize1, 32))

            if type1.char == "U":
                common = np.dtype("U" + maxsize)
            else:
                common = np.dtype("S" + maxsize)

    elif type2.char in "SU":
        if type1.num <= 13:
            # handle case where we have an int/float/bool and a string
            # get itemsize
            itemsize2 = type2.itemsize
            if type2.char == "U":
                itemsize2 = itemsize2 // 4

            if type1.num <= 10:
                maxsize = str(max(itemsize2, 16))
            else:
                maxsize = str(max(itemsize2, 32))

            if type2.char == "U":
                common = np.dtype("U" + maxsize)
            else:
                common = np.dtype("S" + maxsize)

    return common


def _find_lossless_common_type(dt1: np.dtype, dt2: np.dtype) -> Union[np.dtype, None]:
    """
    Finds the lossless common type, or None if not found.
    """
    dtc = np.result_type(dt1, dt2)

    if not np.issubdtype(dt1, np.number) or not np.issubdtype(dt2, np.number):
        return dtc

    def _get_info(dt: np.dtype) -> dict:
        if np.issubdtype(dt, np.integer):
            info = np.iinfo(dt)
            return {"min": info.min, "max": info.max, "prec": info.bits - (1 if info.min < 0 else 0)}
        info = np.finfo(dt)
        return {"min": info.min, "max": info.max, "prec": info.nmant}

    infoc = _get_info(dtc)
    info1 = _get_info(dt1)
    info2 = _get_info(dt2)

    def can_represent(itest, itarget):
        return itest["min"] >= itarget["min"] and itest["max"] <= itarget["max"] and itest["prec"] <= itarget["prec"]

    return dtc if can_represent(info1, infoc) and can_represent(info2, infoc) else None


def _find_lossless_common_array_type(arr1: np.array, arr2: np.array) -> Union[np.dtype, None]:
    """
    Finds the lossless common type for the array values, or None if not found.
    """
    dt1 = arr1.dtype
    dt2 = arr2.dtype

    dtc = _find_lossless_common_type(dt1, dt2)
    if dtc is not None:
        return dtc

    # Check values to see if all of them can fit in the type of the other.
    # Only consider coercion from inexact to exact types (simplifies tests).

    def _get_info(dt: np.dtype) -> dict:
        if np.issubdtype(dt, np.integer):
            info = np.iinfo(dt)
            return {"min": info.min, "max": info.max, "exact": True}
        info = np.finfo(dt)
        return {"min": info.min, "max": info.max, "exact": False}

    def _can_coerce(arr: np.array, array_info: dict, target_info: dict) -> bool:
        if not target_info["exact"]:
            return False
        np_arr = arr.view(np.ndarray)  # use np.ndarray since rt.FA is unreliable at these limits
        if array_info["min"] < 0 and (np_arr < target_info["min"]).all():
            return False
        if (np_arr > target_info["max"]).all():
            return False
        return True

    info1 = _get_info(dt1)
    info2 = _get_info(dt2)

    if _can_coerce(arr2, info2, info1):
        return dt1

    if _can_coerce(arr1, info1, info2):
        return dt2

    return None


def _get_lossless_common_array_type(arr1: np.array, arr2: np.array) -> Union[np.dtype, None]:
    common_type = _find_lossless_common_array_type(arr1, arr2)
    if not common_type:
        raise TypeError(f"Cannot find lossless common type of {arr1.dtype} and {arr2.dtype}")
    return common_type


def empty(shape, dtype: Union[str, np.dtype, type] = float, order: str = "C") -> "FastArray":
    """
    Return a new array of specified shape and type, without initializing entries.

    Unlike :py:func:`~.rt_numpy.zeros`, :py:func:`~.rt_numpy.empty` doesn't set the
    array values to zero, so it may be marginally faster. On the other hand, it requires
    the user to manually set all the values in the array, so it should be used with
    caution.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array, e.g., ``(2, 3)`` or ``2``. Note that although
        multi-dimensional arrays are technically supported by Riptable,
        you may get unexpected results when working with them.
    dtype : str or :py:class:`numpy.dtype` or Riptable dtype, default :py:obj:`numpy.float64`
        The desired data type for the array.
    order : {'C', 'F'}, default 'C'
        Whether to store multi-dimensional data in row-major (C-style) or
        column-major (Fortran-style) order in memory.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray`
        A new :py:class:`~.rt_fastarray.FastArray` of uninitialized (arbitrary) data of
        the specified shape and type.

    See Also
    --------
    :py:func:`.rt_numpy.empty_like`
    :py:func:`.rt_numpy.ones`
    :py:func:`.rt_numpy.ones_like`
    :py:func:`.rt_numpy.zeros`
    :py:func:`.rt_numpy.zeros_like`
    :py:func:`.rt_numpy.empty`
    :py:func:`.rt_numpy.full`
    :py:meth:`.rt_categorical.Categorical.full`

    Examples
    --------
    >>> rt.empty(5)  # doctest: +SKIP
    FastArray([0.  , 0.25, 0.5 , 0.75, 1.  ])  # uninitialized

    Note that the results from :py:func:`~.rt_numpy.empty` vary, given that the
    entries in the resulting :py:class:`~.rt_fastarray.FastArray` objects are uninitialized.
    For example:

    >>> rt.empty(5) # doctest: +SKIP
    FastArray([3.21142670e-322, 0.00000000e+000, 1.42173718e-312,
           2.48273508e-312, 2.46151512e-312])  # uninitialized

    >>> rt.empty(5, dtype=int)  # doctest: +SKIP
    FastArray([80288976,        0,        0,        0,        1])  # uninitialized
    """
    # return LedgerFunction(np.empty, shape, dtype=dtype, order=order)

    # make into list of ints
    try:
        shape = [int(k) for k in shape]
    except:
        shape = [int(shape)]

    dtype = np.dtype(dtype)

    # try to use recycler
    result = rc.Empty(shape, dtype.num, dtype.itemsize, order == "F")
    if result is None:
        return LedgerFunction(np.empty, shape, dtype=dtype, order=order)
    else:
        return result


def empty_like(
    array: np.ndarray,
    dtype: Optional[Union[str, np.dtype, type]] = None,
    order: str = "K",
    subok: bool = True,
    shape: Optional[Union[int, Sequence[int]]] = None,
) -> "FastArray":
    """
    Return a new array with the same shape and type as the specified array,
    without initializing entries.

    Parameters
    ----------
    array : array
        The shape and data type of ``array`` define the same attributes of the
        returned array. Note that although multi-dimensional arrays are
        technically supported by Riptable, you may get unexpected results when
        working with them.
    dtype : str or :py:class:`numpy.dtype` or Riptable dtype, optional
        Overrides the data type of the result.
    order : {'K', C', 'F', or 'A'}, default 'K'
        Overrides the memory layout of the result. 'K' (the default) means
        match the layout of ``array`` as closely as possible. 'C' means
        row-major (C-style); 'F' means column-major (Fortran-style); 'A'
        means 'F' if ``array`` is Fortran-contiguous, 'C' otherwise.
    subok : bool, default `True`
        If `True` (the default), then the newly created array uses the
        sub-class type of ``array``, otherwise it is a base-class array.
    shape : int or sequence of ints, optional
        Overrides the shape of the result. If ``order='K'`` and the number of
        dimensions is unchanged, it tries to keep the same order; otherwise,
        ``order='C'`` is implied. Note that although multi-dimensional arrays are
        technically supported by Riptable, you may get unexpected results when
        working with them.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray`
        A new :py:class:`~.rt_fastarray.FastArray` of uninitialized (arbitrary) data
        with the same shape and type as ``array``.

    See Also
    --------
    :py:func:`.rt_numpy.empty`
    :py:func:`.rt_numpy.ones`
    :py:func:`.rt_numpy.ones_like`
    :py:func:`.rt_numpy.zeros`
    :py:func:`.rt_numpy.zeros_like`
    :py:func:`.rt_numpy.full`
    :py:meth:`.rt_categorical.Categorical.full`

    Examples
    --------
    >>> a = rt.FastArray([1, 2, 3, 4])
    >>> rt.empty_like(a) # doctest: +SKIP
    FastArray([1, 2, 4, 7])  # uninitialized

    Note that the results from :py:func:`~.rt_numpy.empty_like` vary, given that the
    entries from the resulting :py:class:`~.rt_fastarray.FastArray` objects are uninitialized.

    >>> rt.empty_like(a, dtype=float) # doctest: +SKIP
    FastArray([0. , 0. , 6.4, 4.8])  # uninitialized
    """
    # TODO: call recycler

    # NOTE: np.empty_like preserves the subclass
    if isinstance(array, TypeRegister.FastArray):
        array = array._np
    result = LedgerFunction(np.empty_like, array, dtype=dtype, order=order, subok=subok, shape=shape)
    return result


# -------------------------------------------------------
def _searchsorted(array, v, side="left", sorter=None):
    from .rt_utils import possibly_convert

    def _punt_to_numpy(array, v, side, sorter):
        # numpy does not like fastarrays for this routine
        if isinstance(array, TypeRegister.FastArray):
            array = array._np
        if isinstance(v, TypeRegister.FastArray):
            v = v._np
        return LedgerFunction(np.searchsorted, array, v, side=side, sorter=sorter)

    is_scalar = np.isscalar(v)
    dtype = get_common_dtype(array, v)

    if not dtype.kind in 'biuf':
        return _punt_to_numpy(array, v, side, sorter)

    if not isinstance(array, np.ndarray):
        array = np.array(array, dtype=dtype)

    if not isinstance(v, np.ndarray):
        v = np.array(v, dtype=dtype)

    array = possibly_convert(array, dtype)
    v = possibly_convert(v, dtype)
    # we cannot handle a sorter
    if sorter is None:
        try:
            res = None
            if side == "leftplus":
                res = rc.BinsToCutsBSearch(v, array, 0)
            elif side == "left":
                res = rc.BinsToCutsBSearch(v, array, 1)
            else:
                res = rc.BinsToCutsBSearch(v, array, 2)

            if is_scalar and not np.isscalar(res):
                res = res[0]

            return res
        except:
            # fall into numpy
            pass

    return _punt_to_numpy(array, v, side, sorter)


# -------------------------------------------------------
def searchsorted(a, v, side="left", sorter=None) -> int:
    """see np.searchsorted
    side ='leftplus' is a new option in riptable where values > get a 0
    """
    return _searchsorted(a, v, side=side, sorter=sorter)


# -------------------------------------------------------
def issorted(*args) -> bool:
    """
    Return `True` if the array is sorted, `False` otherwise.

    ``NaN`` values at the end of an array are considered sorted.

    Parameters
    ----------
    *args : ndarray
        The array to check. It must be one-dimensional and contiguous.

    Returns
    -------
    bool
        `True` if the array is sorted, `False` otherwise.

    See Also
    --------
    :py:meth:`.rt_fastarray.FastArray.issorted`

    Examples
    --------
    >>> a = rt.FastArray(['a', 'c', 'b'])
    >>> rt.issorted(a)
    False

    >>> a = rt.FastArray([1.0, 2.0, 3.0, rt.nan])
    >>> rt.issorted(a)
    True

    >>> cat = rt.Categorical(['a', 'a', 'a', 'b', 'b'])
    >>> rt.issorted(cat)
    True

    >>> dt = rt.Date.range('20190201', '20190208')
    >>> rt.issorted(dt)
    True

    >>> dtn = rt.DateTimeNano(['6/30/19', '1/30/19'], format='%m/%d/%y', from_tz='NYC')
    >>> rt.issorted(dtn)
    False
    """
    return LedgerFunction(rc.IsSorted, *args)


# -------------------------------------------------------
def unique(
    arr: Union[np.ndarray, List[np.ndarray]],
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    sorted: bool = True,
    lex: bool = False,
    dtype: Optional[Union[str, np.dtype]] = None,
    filter: Optional[np.ndarray] = None,
) -> Union["FastArray", Tuple["FastArray", ...], List["FastArray"], tuple]:
    """
    Find the unique elements of an array or the unique combinations of elements with
    corresponding indices in multiple arrays.

    Parameters
    ----------
    arr : array_like or list of array_like
        Input array, or a list of arrays that are the same shape. If a list of arrays is
        provided, it's treated as a multikey in which the arrays' values at
        corresponding indices are associated.
    return_index : bool, default `False`
        If `True`, also return the indices of the first occurrences of the unique values
        (for one input array) or unique combinations (for multiple input arrays) in
        ``arr``.
    return_inverse : bool, default `False`
        If `True`, also return the indices of the unique array (for one input array) or
        combinations (for multiple input arrays) that can be used to reconstruct ``arr``.
    return_counts : bool, default `False`
        If `True`, also return the number of times each unique item (for one input array)
        or combination (for multiple input arrays) appears in ``arr``.
    sorted : bool, default `True`
        Indicates whether the results are returned in sorted order. Defaults to `True`,
        which replicates the behavior of the NumPy version of this function. When `False`
        (which is often faster), the display order is first appearance. If ``lex`` is set
        to `True`, the value of this parameter is ignored and the results are always
        returned in sorted order.
    lex : bool, default `False`
        Controls whether the function uses hashing- or sorting-based logic to find the
        unique values in ``arr``. Defaults to `False` (hashing). Set to `True` to use a
        lexicographical sort instead; this can be faster when ``arr`` is a large array
        with a relatively high proportion of unique values.
    dtype : {None, 'b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q', 'p', 'P'}, default `None`
        If an index is returned via ``return_index`` or ``return_inverse``, you can use
        a NumPy data type character code to specify the data type of the returned index.
        For definitions of the character codes for integer types, see
        :ref:`arrays.scalars.character-codes`.
    filter : ndarray of bool, default `None`
        If provided, any `False` values are ignored in the calculation. If provided
        and ``return_inverse`` is `True`, a filtered-out location is -1.

    Returns
    -------
    unique : ndarray or list of ndarrays
        For one input array, one array is returned that contains the unique values. For
        multiple input arrays, a list of arrays is returned that collectively contains
        every unique combination of values found in the arrays' corresponding indices.
    unique_indices : :py:class:`~.rt_fastarray.FastArray`, optional
        The indices of the first occurrences of the unique values in the original array.
        Only provided if ``return_index`` is `True`.
    unique_inverse : :py:class:`~.rt_fastarray.FastArray`, optional
        The indices of the unique array (for one input array) or unique combinations
        (for multiple input arrays) that can be used to reconstruct ``arr``. Only
        provided if ``return_inverse`` is `True`.
    unique_counts : :py:class:`~.rt_fastarray.FastArray`, optional
        The number of times each of the unique values comes up in the original array.
        Only provided if ``return_counts`` is `True`.

    Notes
    -----
    :py:func:`~.rt_numpy.unique` often performs faster than :py:func:`numpy.unique` for
    strings and numeric types.

    :py:class:`~.rt_categorical.Categorical` objects passed in as ``arr`` ignore the ``sorted``
    flag and return their current order.

    Examples
    --------
    >>> rt.unique(['b','b','a','d','d'])
    FastArray(['a', 'b', 'd'], dtype='<U1')

    With ``sorted=False``, the returned array displays the unique values in the order
    of their first appearance in the original array.

    >>> rt.unique(['b','b','a','d','d'], sorted = False)
    FastArray(['b', 'a', 'd'], dtype='<U1')

    When multiple arrays are passed, they're treated as a multikey. The result is a list
    of arrays that collectively contains every unique combination of values found in the
    arrays' corresponding indices.

    >>> rt.unique([['b','b','a','d','d'],
    ...            ['b','b','c','d','d']])
    [FastArray(['a', 'b', 'd'], dtype='<U1'),
     FastArray(['c', 'b', 'd'], dtype='<U1')]

    Return the indices of the first occurrences of the unique values in the original
    array:

    >>> a = rt.FastArray(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = rt.unique(a, return_index = True)
    >>> u
    FastArray([b'a', b'b', b'c'], dtype='|S1')
    >>> indices
    FastArray([0, 1, 3])
    >>> a[indices]
    FastArray([b'a', b'b', b'c'], dtype='|S1')

    Reconstruct the input array from the unique values and inverse. Note that this
    method of reconstruction doesn't work for multiple input arrays or if the original
    array is filtered.

    >>> a = rt.FastArray([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = rt.unique(a, return_inverse = True)
    >>> u
    FastArray([1, 2, 3, 4, 6])
    >>> indices
    FastArray([0, 1, 4, 3, 1, 2, 1], dtype=int8)
    >>> u[indices]
    FastArray([1, 2, 6, 4, 2, 3, 2])

    Reconstruct the input values from the unique values and counts. Note that this
    doesn't reconstruct the array in order; it just reconstructs the same number of each
    element. This method of reconstruction doesn't work for multiple input arrays or if
    the original array is filtered.

    >>> a = rt.FastArray([1, 2, 6, 4, 2, 3, 2])
    >>> values, counts = rt.unique(a, return_counts = True)
    >>> values
    FastArray([1, 2, 3, 4, 6])
    >>> counts
    FastArray([1, 3, 1, 1, 1], dtype=int32)
    >>> rt.repeat(values, counts)
    FastArray([1, 2, 2, 2, 3, 4, 6])
    """
    if dtype is not None:
        if dtype not in NumpyCharTypes.AllInteger:
            dtype = None

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts
    mark_readonly = False

    if isinstance(arr, TypeRegister.Categorical):
        # NOTE if the categorical is not dirty, filter should do nothing
        # TODO: need to set dirty flag g=arr.set_valid().grouping
        g = arr.grouping

        # check for Dictionary mode
        if arr.category_mode == CategoryMode.Dictionary:
            if filter is not None:
                arr = arr.set_valid(filter)
            else:
                if g.isdirty:
                    arr = arr.set_valid(filter)
                else:
                    mark_readonly = True

            # get back grouping in case it changed
            g = arr.grouping

        else:
            if filter is not None:
                g = g.regroup(filter)
            else:
                if g.isdirty:
                    # dirty flag means a bool or fancy index mask was applied
                    g = g.regroup()
                else:
                    mark_readonly = True

        # NOTE the existing categorical is already ordered/unordered and thus will disobey the sorted flag
    else:
        if not isinstance(arr, np.ndarray):
            if isinstance(arr, list) and isinstance(arr[0], (np.ndarray, list)):
                # user passing a list of arrays, or list of lists, assume multikey unique
                if isinstance(arr[0], list):
                    arr = [TypeRegister.FastArray(v, unicode=True) for v in arr]
            else:
                arr = TypeRegister.FastArray(arr, unicode=True)

        # Grouping is faster than Categorical and preserves ifirstkey
        if lex is True or sorted is False:
            g = TypeRegister.Grouping(arr, lex=lex, filter=filter)
        else:
            # TODO: need flag to preserve ifirstkey when making a Categorical
            # TODO: or grouping needs to obey ordered flag (then don't need to make Categorical)
            g = TypeRegister.Categorical(arr, ordered=sorted, lex=lex, filter=filter).grouping

    un = g.uniquelist

    # check for multikey
    if len(un) == 1:
        un = un[0]

    if mark_readonly:
        # make an object copy to mark it readonly
        un = un.view(TypeRegister.FastArray)
        un.flags.writeable = False

    if return_counts:
        # handles both base0 and base1
        counts = g.ncountgroup[1:]

    if not optional_returns:
        return un

    else:
        ret = tuple()

        # index of first appearance in original array
        if return_index:
            idx = g.ifirstkey
            if dtype is not None:
                idx = idx.astype(dtype, copy=False)
            ret += (idx,)

        # an array mapping original values to their indices in the unique array
        if return_inverse:
            inv = g.ikey
            if g.base_index == 1:
                inv = inv - 1

            if dtype is not None:
                inv = inv.astype(dtype, copy=False)
            ret += (inv,)

        # counts of each unique item in the original array
        if return_counts:
            ret += (counts,)

        ret = (un,) + ret
        return ret


def _possibly_match_categoricals(a: "Categorical", b):
    """
    Parameters
    ----------
    a : Categorical
    b
        To be matched to a Categorical

    Notes
    -----
    For correct evaluation in ismember, categoricals either need to be re-expanded or have matching modes.
    This routine will make sure the re-expansion is valid.
    """

    if isinstance(b, TypeRegister.Categorical):
        if a.category_mode != b.category_mode:
            raise TypeError(
                f"ismember on categorical must be with a categorical in the same mode. Got {a.category_mode} {b.category_mode}"
            )
        else:
            mode = a.category_mode
            # multikey categoricals need to have the same number of columns
            # regular multikey ismember will check for matching dtypes within the dictionaries
            if mode == CategoryMode.MultiKey:
                adict = a.category_dict
                bdict = b.category_dict

                if len(adict) != len(bdict):
                    raise ValueError(
                        f"Multikey dictionaries in ismember categorical did not have the same number of keys. {len(adict)} vs. {len(bdict)}"
                    )

            # if codes exist in both mappings, their values must be consistent
            elif mode in [CategoryMode.Dictionary, CategoryMode.IntEnum]:
                adict = a.category_mapping
                bdict = b.category_mapping

                match = True
                for code, aval in adict.items():
                    bval = bdict.get(code, None)
                    if bval is not None:
                        if bval != aval:
                            match = False
                            break
                # use arrays of integer codes
                if match:
                    a = a._fa
                    b = b._fa

                else:
                    raise ValueError(f"Mapped categoricals had non-matching mappings. Could not perform ismember.")

            # ismember code will perform an ismember on these, take a different final path
            elif mode in [CategoryMode.StringArray, CategoryMode.NumericArray]:
                pass

            else:
                raise NotImplementedError

    # it's faster to make a categorical than to reexpand
    # turn other array argument into categorical before performing final ismember
    elif a.category_mode == CategoryMode.StringArray:
        if b.dtype.char in ("U", "S"):
            # future optimization: don't sort the bins when making the throwaway categorical
            # if a.unique_count < 30_000_000 and TypeRegister.Categorical.TestIsMemberVerbose == True:
            #    _, idx = ismember(b, a.category_array)
            #    if a.base_index == 1:
            #        idx += 1
            #    return a._fa, idx
            unicode = False
            if b.dtype.char == "U":
                unicode = True
            b = TypeRegister.Categorical(b, unicode=unicode, ordered=False)
        else:
            raise TypeError(f"Cannot perform ismember on categorical in string array mode and array of dtype {b.dtype}")

    elif a.category_mode == CategoryMode.NumericArray:
        if b.dtype.char in NumpyCharTypes.AllInteger:
            b = TypeRegister.Categorical(b)

        elif b.dtype.char in NumpyCharTypes.SupportedFloat:
            b = TypeRegister.Categorical(b)
        else:
            raise TypeError(f"Could not perform ismember on numeric categorical and array with dtype {b.dtype}")

    else:
        raise TypeError(
            f"Could not perform ismember on categorical in {a.category_mode.name} and array with dtype {b.dtype}"
        )

    return a, b


def _ismember_align_multikey(a, b):
    """
    Checks that types in each column match, string widths are the same for columns in the corresponding lists.

    Unless the columns have matching types and itemsizes, the CPP multikey ismember call will not work, or
    produce incorrect results.

    Parameters
    ----------
    a : list of arrays
    b : list of arrays

    Notes
    -----
    TODO: push these into methods that can be used to normalize single-key ``ismember()``.
    """

    def _as_fastarrays(col):
        # flip all input arrays to FastArray
        # re-expand single key or enum categoricals for IsMemberMultikey
        if not isinstance(col, np.ndarray):
            col = TypeRegister.FastArray(col)
        elif isinstance(col, TypeRegister.Categorical):
            if col.ismultikey:
                raise TypeError(
                    f"Multikey ismember could not re-expand array for categorical in mode {col.category_mode.name}."
                )
            col = col.expand_array
        return col

    allowed_int = "bhilqpBHILQP"
    allowed_float = "fdg"
    allowed_types = allowed_int + allowed_float

    # make sure original container items don't get blown away during fixup
    if isinstance(a, tuple):
        a = list(a)
    if isinstance(b, tuple):
        b = list(b)
    if isinstance(a, list):
        a = a.copy()
    if isinstance(b, list):
        b = b.copy()

    for idx, a_col in enumerate(a):
        b_col = b[idx]
        a_col = _as_fastarrays(a_col)
        b_col = _as_fastarrays(b_col)

        a_char = a_col.dtype.char
        b_char = b_col.dtype.char
        # if a column was string, need to match string width in b
        if a_char in "US":
            if b_char in "US":
                # TODO: find a prettier way of doing this...
                if a_char != b_char:
                    # if unicode is present (probably rare), need to upcast both
                    if a_char == "U":
                        a_width = a_col.itemsize // 4
                        b_width = b_col.itemsize
                    else:
                        a_width = a_col.itemsize
                        b_width = b_col.itemsize // 4
                    dtype_letter = "U"

                # both unicode or both bytes, just match width
                else:
                    dtype_letter = a_char
                    if dtype_letter == "U":
                        a_width = a_col.itemsize // 4
                        b_width = b_col.itemsize // 4
                    else:
                        a_width = a_col.itemsize
                        b_width = b_col.itemsize

                # prepare string for final dtype e.g. 'S5', 'U12', etc.
                final_width = max(a_width, b_width)
                dt_char = dtype_letter + str(final_width)
                a_col = a_col.astype(dt_char, copy=False)
                b_col = b_col.astype(dt_char, copy=False)
            else:
                raise TypeError(f"Could not perform multikey ismember on types {a_col.dtype} and {b_col.dtype}")

        else:
            # make sure both are supported numeric types
            if a_char not in allowed_types:
                raise TypeError(f"{a_col.dtype} not in allowed types for ismember with {b_col.dtype}")
            if b_char not in allowed_types:
                raise TypeError(f"{b_col.dtype} not in allowed types for ismember with {a_col.dtype}")

            # cast if necessary
            if a_char != b_char:
                # warnings.warn(f"Performance warning: numeric arrays in ismember had different dtypes {a.dtype} {b.dtype}")
                common_type = _get_lossless_common_array_type(a_col, b_col)
                a_col = a_col.astype(common_type, copy=False)
                b_col = b_col.astype(common_type, copy=False)

        a[idx] = a_col
        b[idx] = b_col

    return a, b


@_args_to_fast_arrays("a", "b")
def ismember(
    a: ArraysOrDataset, b: ArraysOrDataset, h=2, hint_size: int = 0, base_index: int = 0
) -> Tuple[Union[int, "FastArray"], "FastArray"]:
    """
    The ismember function is meant to mimic the ismember function in MATLab. It takes two sets of data
    and returns two - a boolean array and array of indices of the first occurrence of an element in `a` in
    `b` - otherwise NaN.

    Parameters
    ----------
    a : A python list (strings), python tuple (strings), chararray, ndarray of unicode strings,
        ndarray of int32, int64, float32, or float64.
    b : A list with the same constraints as `a`. Note: if a contains string data, b must also contain
        string data. If it contains different numerical data, casting will occur in either `a` or `b`.
    h : There are currently two different hashing functions that can be used to execute ismember.
        Depending on the size, type, and number of matches in the data, the hashes perform differently.
        Currently accepts 1 or 2.
        1=PRIME number (might be faster for floats - uses less memory)
        2=MASK using power of 2 (usually faster but uses more memory)
    hint_size : int, default 0
        For large arrays with a low unique count, setting this value to 4*expected unique
        count may speed up hashing.
    base_index : int, default 0
        When set to 1 the first return argument is no longer a boolean array but an integer that is 1 or 0.
        A return value of 1 indicates there exists values in `b` that do not exist in `a`.

    Returns
    -------
    c : int or np.ndarray of bool
        A boolean array the same size as a indicating whether or not the element at the corresponding
        index in `a` was found in `b`.
    d : np.ndarray of int
        An array of indices the same size as `a` which each indicate where an element in a first occured
        in `b` or NaN otherwise.

    Raises
    ------
    TypeError
        input must be ndarray, python list, or python tuple
    ValueError
        data must be int32, int64, float32, float64, chararray, or unicode strings.
        If a contains string data, b must also contain string data and vice versa.

    Examples
    --------
    >>> a = [1.0, 2.0, 3.0, 4.0]
    >>> b = [1.0, 3.0, 4.0, 4.0]
    >>> c,d = ismember(a,b)
    >>> c
    FastArray([ True, False,  True,  True])
    >>> d
    FastArray([   0, -128,    1,    2], dtype=int8)


    NaN values do not behave the same way as other elements. A NaN in the first will not register as existing in the second array.
    This is the expected behavior (to match MatLab nan MATLab nan handling):

    >>> a = FastArray([1.,2.,3.,np.nan])
    >>> b = FastArray([2.,3.,np.nan])
    >>> c,d = ismember(a,b)
    >>> c
    FastArray([False,  True,  True, False])
    >>> d
    FastArray([-128,    0,    1, -128], dtype=int8)
    """

    # make sure a and b are listlike, and not empty
    if not (isinstance(a, (list, tuple, np.ndarray)) and isinstance(b, (list, tuple, np.ndarray))):
        raise TypeError("Input must be python list, tuple or np.ndarray.")
    allowed_int = "bhilqpBHILQP"
    allowed_float = "fdg"
    allowed_types = allowed_int + allowed_float

    len_a = len(a)
    len_b = len(b)
    is_multikey = False

    if len_a == 0 or len_b == 0:
        indexer_type = np.dtype(np.int32)
        indexer = full(len_a, INVALID_DICT[indexer_type.num], dtype=indexer_type)
        return zeros(len(a), dtype=bool), indexer

    if isinstance(a, TypeRegister.Categorical) or isinstance(b, TypeRegister.Categorical):
        if isinstance(a, TypeRegister.Categorical):
            if not isinstance(b, np.ndarray):
                b = TypeRegister.FastArray(b)
            a, b = _possibly_match_categoricals(a, b)

        if isinstance(b, TypeRegister.Categorical):
            if not isinstance(a, np.ndarray):
                a = TypeRegister.FastArray(a)
            if not isinstance(a, TypeRegister.Categorical):
                b, a = _possibly_match_categoricals(b, a)

        # re-expansion has happened, use regular ismember
        # enum/mapped categoricals with consistent mappings (but not necessarily the same ones) will take this path
        if not isinstance(a, TypeRegister.Categorical):
            return ismember(a, b)

        # special categorical ismember needs to be called
        if a.issinglekey or a.ismultikey:
            acats, bcats = list(a.category_dict.values()), list(b.category_dict.values())
            num_unique_b = len(bcats[0])
        else:
            raise NotImplementedError(
                f"Have not yet found a solution for ismember on categoricals in {a.category_mode.name} mode"
            )

        _, on_unique = ismember(acats, bcats)
        # rc.IsMemberCategorical:
        # arg1 - underlying FastArray of a Categorical
        # arg2 - underlying FastArray of b Categorical
        # arg3 - first occurrence of a's uniques into b's uniques
        # arg4 - number of unique in b
        # arg5 - a base index
        # arg6 - b base index
        b, f = rc.IsMemberCategoricalFixup(
            a._fa, b._fa, on_unique.astype(np.int32), int(num_unique_b), a.base_index, b.base_index
        )
        return b, f

    # a and b contain list like, probably a multikey
    a_is_multi, b_is_multi = _is_array_container(a), _is_array_container(b)
    if a_is_multi or b_is_multi:
        if not (a_is_multi and b_is_multi):
            raise ValueError("ismember found a multi-key in exactly one argument, must be both or neither")
        is_multikey = True
        if all(len(x) == 0 for x in a):
            return ismember([], [])

    # different number of key columns
    if is_multikey:
        if len_a == len_b:
            # single key "multikey", send through regular ismember
            if len_a == 1 and len_b == 1:
                return ismember(a[0], b[0], h)

            a, b = _ismember_align_multikey(a, b)

            return rc.MultiKeyIsMember32((a,), (b,), hint_size)
        else:
            raise ValueError(
                f"Multikey ismember must have the same number of keys in each item. a had {len_a}, b had {len_b}"
            )

    # convert both to FastArray
    if isinstance(a, (list, tuple)):
        a = TypeRegister.FastArray(a)
    if isinstance(b, (list, tuple)):
        b = TypeRegister.FastArray(b)

    a_char = a.dtype.char
    b_char = b.dtype.char

    # handle strings
    if a_char in ("U", "S"):
        if b_char in ("U", "S"):
            # if the string types do not match, always try to use byte strings for the final operation
            if a_char != b_char:
                if a_char == "U":
                    try:
                        a = a.astype("S")
                    except:
                        b = b.astype("U")
                else:
                    try:
                        b = b.astype("S")
                    except:
                        a = a.astype("U")
        else:
            raise TypeError(
                f"The first parameter is a string but the second parameter is not and cannot be compared. {a.dtype} vs. {b.dtype}"
            )

    # will only be hit if a is not strings
    elif b_char in ("U", "S"):
        raise TypeError(
            f"The second parameter is a string but the first parameter is not and cannot be compared. {a.dtype} vs. {b.dtype}"
        )

    else:
        # make sure both are supported numeric types
        if a_char not in allowed_types:
            raise TypeError(f"{a.dtype} not in allowed types for ismember")
        if b_char not in allowed_types:
            raise TypeError(f"{b.dtype} not in allowed types for ismember")

        # cast if necessary
        if a_char != b_char:
            # import traceback
            # for line in traceback.format_stack():
            #    print(line.strip())

            # warnings.warn(f"Performance warning: numeric arrays in ismember had different dtypes {a.dtype} {b.dtype}")
            # raise TypeError('numeric arrays in ismember need to be the same dtype')
            common_type = _get_lossless_common_array_type(a, b)
            if a.dtype != common_type:
                a = a.astype(common_type)
            if b.dtype != common_type:
                b = b.astype(common_type)

    # send to fastmath
    if base_index == 1:
        return rc.IsMemberCategorical(a, b, h, hint_size)
    elif base_index is None or base_index == 0:
        return rc.IsMember32(a, b, h, hint_size)
    else:
        raise ValueError(f"base_index must be 0, 1, or None not {base_index!r}")


def assoc_index(key1: ArraysOrDataset, key2: ArraysOrDataset) -> "FastArray":
    """
    Parameters
    ----------
    key1 : ndarray / list thereof or a Dataset
        Numpy arrays to match against; all arrays must be same length.
    key2 : ndarray / list thereof or a Dataset
        Numpy arrays that will be matched with `key1`; all arrays must be same length.

    Returns
    -------
    fancy_index : ndarray of ints
        Fancy index where the index of `key2` is matched against `key1`;
        if there was no match, the minimum integer (aka sentinel) is the index value.

    Examples
    --------
    >>> np.random.seed(12345)
    >>> ds = rt.Dataset({'time': rt.arange(200_000_000.0)})
    >>> ds.data = np.random.randint(7, size=200_000_000)
    >>> ds.symbol = rt.Cat(1 + rt.arange(200_000_000) % 7, ['AAPL','AMZN', 'FB', 'GOOG', 'IBM','MSFT','UBER'])
    >>> dsa = rt.Dataset({'data': rt.repeat(rt.arange(7), 7), 'symbol': rt.tile(rt.FastArray(['AAPL','AMZN', 'FB', 'GOOG', 'IBM','MSFT','UBER']), 7)})
    >>> rt.assoc_index([ds.symbol, ds.data], [dsa.symbol, dsa.data])
    FastArray([35, 43,  2, ..., 43, 37, 24])
    """
    return ismember(key1, key2)[1]


def assoc_copy(
    key1: ArraysOrDataset, key2: ArraysOrDataset, arr: Union[np.ndarray, TypeRegister.Dataset]
) -> Union[FastArray, Dataset]:
    """
    Parameters
    ----------
    key1 : ndarray / list thereof or a Dataset
        Numpy arrays to match against; all arrays must be same length.
    key2 : ndarray / list thereof or a Dataset
        Numpy arrays that will be matched with `key1`; all arrays must be same length.
    arr : ndarray / Dataset
        An array or Dataset the same length as key2 arrays which will be mapped to the size of `key1`
        In the case of an array, the output will be cast to FastArray to accomodate support of fancy-indexing with sentinel values

    Returns
    -------
    array_like
        A new array the same length as `key1` arrays which has mapped the input `arr` from `key2` to `key1`
        the array's dtype will match the dtype of the input array (3rd parameter).
        However, outputs will be FastArrays when the input array is a numpy arrays such that
        fancy indexing with sentinels works correctly.

    Examples
    --------
    >>> np.random.seed(12345)
    >>> ds=Dataset({'time': rt.arange(200_000_000.0)})
    >>> ds.data = np.random.randint(7, size=200_000_000)
    >>> ds.symbol = rt.Cat(1 + rt.arange(200_000_000) % 7, ['AAPL','AMZN', 'FB', 'GOOG', 'IBM','MSFT','UBER'])
    >>> dsa = rt.Dataset({'data': rt.repeat(rt.arange(7), 7), 'symbol': rt.tile(rt.FastArray(['AAPL','AMZN', 'FB', 'GOOG', 'IBM','MSFT','UBER']), 7), 'time': 48 - rt.arange(49.0)})
    >>> rt.assoc_copy([ds.symbol, ds.data], [dsa.symbol, dsa.data], dsa.time)
    FastArray([13.,  5., 46., ...,  5., 11., 24.])
    """
    fancyindex = assoc_index(key1, key2)
    if isinstance(arr, TypeRegister.Dataset):
        return arr[fancyindex, :]
    else:
        return TypeRegister.FastArray(arr)[fancyindex]


def unique32(list_keys: List[np.ndarray], hintSize: int = 0, filter: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return the index location of the first occurence of each key.

    Parameters
    ----------
    list_keys : list of ndarray
        A list of numpy arrays to hash on (multikey);
        if there is just one item it still needs to be in a list such as ``[array1]``.
    hintSize : int
        Integer hint if the number of unique keys (in `list_keys`) is known in advance, defaults to 0.
    filter : ndarray of bools, optional
        Boolean array used to pre-filter the array(s) in `list_keys` prior to
        processing them, defaults to None.

    Returns
    -------
    ndarray of ints
        An array the size of the total unique values;
        the array contains the INDEX to the first occurence of the unique value.
        the second array contains the INDEX to the last occurence of the unique value.
    """
    return rc.MultiKeyUnique32(list_keys, hintSize, filter)


# -------------------------------------------------------
def combine_filter(key, filter) -> FastArray:
    """
    Parameters
    ----------
    key : ndarray of ints
        index array (int8, int16, int32 or int64)
    filter : ndarray of bools
        Boolean array same length as `key`.

    Returns
    -------
    ndarray of ints
        1 based index array with each False value setting the index to 0.
        The equivalent function is ``return index*filter`` or ``np.where(filter, index, 0)``.

    Notes
    -----
    This routine can run in parallel.
    """
    return rc.CombineFilter(key, filter)


# -------------------------------------------------------
def combine_accum1_filter(key1, unique_count1: int, filter=None):
    """
    Parameters
    ----------
    key1 : ndarray of ints
        index array (int8, int16, int32 or int64) [must be base 1 -- if base 0, increment by 1]
        often referred to as iKey or the bin array for categoricals
    unique_count1 : int
        Maximum number of uniques in `key1` array.
    filter : ndarray of bool, optional
        Boolean array same length as `key1` array, defaults to None.

    Returns
    -------
    iKey:  a new 1 based index array with each False value setting the index to 0
           iKey dtype will match the dtype in Arg1
    iFirstKey: an INT32 array, the fixup for first since some bins may have been removed
    unique_count: INT32 and is the new unique_count1. It is the length of `iFirstKey`

    Example
    -------
    >>> a = rt.arange(20) % 10
    >>> b = a.astype('S')
    >>> c = rt.Cat(b)
    >>> rt.combine_accum1_filter(c, c.unique_count, rt.logical(rt.arange(20) % 2))
    {'iKey': FastArray([0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5],
               dtype=int8),
     'iFirstKey': FastArray([1, 3, 5, 7, 9]),
     'unique_count': 5}

    """
    iKey, iFirstKey, unique_count = rc.CombineAccum1Filter(key1, unique_count1, filter)
    mkdict = {"iKey": iKey, "iFirstKey": iFirstKey, "unique_count": unique_count}
    return mkdict


# -------------------------------------------------------
def combine_accum2_filter(key1, key2, unique_count1: int, unique_count2: int, filter=None):
    """
    Parameters
    ----------
    key1 : ndarray of ints
        First index array (int8, int16, int32 or int64).
    key2 : ndarray of ints
        Second index array (int8, int16, int32 or int64).
    unique_count1 : int
        Maximum number of unique values in `key1`.
    unique_count2 : int
        Maximum number of unique values in `key2`.
    filter : ndarray of bools, optional
        Boolean array with same length as `key1` array, defaults to None.

    Returns
    -------
    TWO ARRAYs (iKey (for 2 dims), nCountGroup)
    bin is a 1 based index array with each False value setting the index to 0
    nCountGroup is INT32 array with size = to (unique_count1 + 1)*(unique_count2 + 1)
    """
    return rc.CombineAccum2Filter(key1, key2, unique_count1, unique_count2, filter)


# -------------------------------------------------------
def combine2keys(key1, key2, unique_count1: int, unique_count2: int, filter=None):
    """
    Parameters
    ----------
    key1 : ndarray of ints
        First index array (int8, int16, int32 or int64).
    key2 : ndarray of ints
        Second index array (int8, int16, int32 or int64).
    unique_count1 : int
        Number of unique values in `key1` (often returned by ``groupbyhash``/``groupbylex``).
    unique_count2 : int
        Number of unique values in `key2`.
    filter : ndarray of bools, optional
        Boolean array with same length as `key1` array, defaults to None.

    Returns
    -------
    TWO ARRAYs (iKey (for 2 dims), nCountGroup)
    bin is a 1 based index array with each False value setting the index to 0
    nCountGroup is INT32 array with size = to (unique_count1 + 1)*(unique_count2 + 1)
    """
    iKey, nCountGroup = rc.CombineAccum2Filter(key1, key2, unique_count1, unique_count2, filter)
    mkdict = {"iKey": iKey, "nCountGroup": nCountGroup}
    return mkdict


# -------------------------------------------------------
def cat2keys(
    key1: Union["Categorical", np.ndarray, List[np.ndarray]],
    key2: Union["Categorical", np.ndarray, List[np.ndarray]],
    filter: Optional[np.ndarray] = None,
    ordered: bool = True,
    sort_gb: bool = False,
    invalid: bool = False,
    fuse: bool = False,
) -> "Categorical":
    """
    Create a `Categorical` from two keys or two `Categorical` objects with all possible unique combinations.

    Notes
    -----
    Code assumes Categoricals are base 1.

    Parameters
    ----------
    key1 : Categorical, ndarray, or list of ndarray
        If a list of arrays is passed for this parameter, all arrays in the list
        must have the same length.
    key2 : Categorical, ndarray, or list of ndarray
        If a list of arrays is passed for this parameter, all arrays in the list
        must have the same length.
    filter : ndarray of bool, optional
        only valid when invalid is set to True
    ordered : bool, default True
        only applies when key1 or key2 is not a categorical
    sort_gb : bool, default False
        only applies when key1 or key2 is not a categorical
    invalid : bool, default False
        Specifies whether or not to insert the invalid when creating the n x m unique matrix.
    fuse : bool, default False
        When True, forces the resulting categorical to have 2 keys, one for rows, and one for columns.

    Returns
    -------
    Categorical
        A multikey categorical that has at least 2 keys.

    Examples
    --------
    The following examples demonstrate using cat2keys on keys as lists and arrays, lists of arrays, and Categoricals.
    In each of the examples, you can determine the unique combinations by zipping the same position of each
    of the values of the category dictionary.

    Creating a MultiKey Categorical from two lists of equal length.

    >>> rt.cat2keys(list('abc'), list('xyz'))
    Categorical([(a, x), (b, y), (c, z)]) Length: 3
      FastArray([1, 5, 9], dtype=int64) Base Index: 1
      {'key_0': FastArray([b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b', b'c'], dtype='|S1'), 'key_01': FastArray([b'x', b'x', b'x', b'y', b'y', b'y', b'z', b'z', b'z'], dtype='|S1')} Unique count: 9

    >>> rt.cat2keys(np.array(list('abc')), np.array(list('xyz')))
    Categorical([(a, x), (b, y), (c, z)]) Length: 3
      FastArray([1, 5, 9], dtype=int64) Base Index: 1
      {'key_0': FastArray([b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b', b'c'], dtype='|S1'), 'key_01': FastArray([b'x', b'x', b'x', b'y', b'y', b'y', b'z', b'z', b'z'], dtype='|S1')} Unique count: 9

    >>> key1, key2 = [rt.FA(list('abc')), rt.FA(list('def'))], [rt.FA(list('uvw')), rt.FA(list('xyz'))]
    >>> rt.cat2keys(key1, key2)
    Categorical([(a, d, u, x), (b, e, v, y), (c, f, w, z)]) Length: 3
      FastArray([1, 5, 9], dtype=int64) Base Index: 1
      {'key_0': FastArray([b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b', b'c'], dtype='|S1'), 'key_1': FastArray([b'd', b'e', b'f', b'd', b'e', b'f', b'd', b'e', b'f'], dtype='|S1'), 'key_01': FastArray([b'u', b'u', b'u', b'v', b'v', b'v', b'w', b'w', b'w'], dtype='|S1'), 'key_11': FastArray([b'x', b'x', b'x', b'y', b'y', b'y', b'z', b'z', b'z'], dtype='|S1')} Unique count: 9

    >>> cat.category_dict
    {'key_0': FastArray([b'a', b'b', b'c', b'a', b'b', b'c', b'a', b'b', b'c'],
               dtype='|S1'),
     'key_1': FastArray([b'd', b'e', b'f', b'd', b'e', b'f', b'd', b'e', b'f'],
               dtype='|S1'),
     'key_01': FastArray([b'u', b'u', b'u', b'v', b'v', b'v', b'w', b'w', b'w'],
               dtype='|S1'),
     'key_11': FastArray([b'x', b'x', b'x', b'y', b'y', b'y', b'z', b'z', b'z'],
               dtype='|S1')}
    """
    Cat = TypeRegister.Categorical

    try:
        if not isinstance(key1, Cat):
            key1 = Cat(key1, ordered=ordered, sort_gb=sort_gb)
    except Exception as e:
        warnings.warn(f"cat2keys: Got exception {e}", RuntimeWarning, stacklevel=2)

    if not isinstance(key1, Cat):
        raise TypeError(
            f"cat2keys: Argument 1 must be a categorical or an array that can be made into a categorical not type {type(key1)!r}"
        )

    try:
        if not isinstance(key2, Cat):
            key2 = Cat(key2, ordered=ordered, sort_gb=sort_gb)
    except Exception as e:
        warnings.warn(f"cat2keys: Got exception {e}", RuntimeWarning, stacklevel=2)

    if not isinstance(key2, Cat):
        raise TypeError(
            f"cat2keys: Argument 2 must be a categorical or an array that can be made into a categorical not type {type(key2)!r}"
        )

    group_row = key1.grouping
    group_col = key2.grouping

    numrows = group_row.unique_count
    numcols = group_col.unique_count

    # have to check for ==0 first
    if not invalid:
        if np.sum(group_row.ikey == 0) > 0 or np.sum(group_col.ikey == 0) > 0:
            warnings.warn(
                "catmatrix: Invalid found in key array, please use invalid=True to avoid this warning.", stacklevel=2
            )
            invalid = True
        else:
            # now we can remove the invalid and reassign
            ikey = group_col.ikey.astype(np.int64) - 1
            # inplace operations for speed
            ikey *= numrows
            ikey += group_row.ikey

    if invalid:
        ikey = combine2keys(group_row.ikey, group_col.ikey, numrows, numcols, filter=filter)["iKey"]

    # also check if the only want 2 keys with fuse
    if invalid or fuse:
        row_name, row_arr = group_row.onedict(invalid=invalid)
        col_name, col_arr = group_col.onedict(invalid=invalid)

        # handle case when same name
        if row_name == col_name:
            col_name = col_name + "1"
        if invalid:
            # invalid was inserted as first unique, so need to make room
            numrows += 1
            numcols += 1
            ikey += 1

        newgroup = TypeRegister.Grouping(ikey, {row_name: row_arr.tile(numcols), col_name: col_arr.repeat(numrows)})

    else:
        # construct grouping object with a multikey
        gdict = dict()
        for k, v in group_row._grouping_unique_dict.items():
            gdict[k] = v.tile(numcols)
        for k, v in group_col._grouping_unique_dict.items():
            # Handle column name conflicts (if present).
            if k in gdict:
                counter = 1
                origk = k
                # Suffix an integer to the original column name,
                # iterating until we find a column name that hasn't
                # been used yet.
                while k in gdict:
                    k = origk + str(counter)
                    counter += 1
            gdict[k] = v.repeat(numrows)
        newgroup = TypeRegister.Grouping(ikey, gdict)

    # create the categorical from the grouping object
    result = Cat(newgroup)

    # save for later in case the categorical needs to be rectangularized like Accum2
    result._numrows = numrows
    result._numcols = numcols
    return result


# -------------------------------------------------------
def makeifirst(key, unique_count: int, filter=None) -> np.ndarray:
    """
    Parameters
    ----------
    key : ndarray of ints
        Index array (int8, int16, int32 or int64).
    unique_count : int
        Maximum number of unique values in `key` array.
    filter : ndarray of bools, optional
        Boolean array same length as `key` array, defaults to None.

    Returns
    -------
    index : ndarray of ints
        An index array of the same dtype and length of the `key` passed in.
        The index array will have the invalid value for the array's dtype set at any locations it could not find a first occurrence.

    Notes
    -----
    makeifirst will NOT reduce the index/ikey unique size even when a filter is passed.
    Based on the integer dtype int8/16/32/64, all locations that have no first will be set to invalid.
    If an invalid is used as a riptable fancy index, it will pull in another invalid, for example '' empty string
    """

    return rc.MakeiFirst(key, unique_count, filter, 0)


# -------------------------------------------------------
def makeilast(key, unique_count: int, filter=None) -> np.ndarray:
    """
    Parameters
    ----------
    key : ndarray of ints
        Index array (int8, int16, int32 or int64).
    unique_count : int
        Maximum number of unique values in `key` array.
    filter : ndarray of bools, optional
        Boolean array same length as `key` array, defaults to None.

    Returns
    -------
    index : ndarray of ints
        An index array of the same dtype and length of the `key` passed in.
        The index array will have the invalid value for the array's dtype set at any locations it could not find a last occurrence.

    Notes
    -----
    makeilast will NOT reduce the index/ikey unique size even when a filter is passed.
    Based on the integer dtype int8/16/32/64, all locations that have no last will be set to invalid.
    If an invalid is used as a riptable fancy index, it will pull in another invalid, for example '' empty string
    """

    return rc.MakeiFirst(key, unique_count, filter, 1)


# -------------------------------------------------------
def makeinext(key, unique_count: int) -> np.ndarray:
    """
    Parameters
    ----------
    key : ndarray of integers
        index array (int8, int16, int32 or int64)
    unique_count : int
        max uniques in 'key' array

    Returns
    -------
    An index array of the same dtype and length of the next row
    The index array will have -MAX_INT set to any locations it could not find a next
    """
    return rc.MakeiNext(key, unique_count, 0)


# -------------------------------------------------------
def makeiprev(key, unique_count: int) -> np.ndarray:
    """
    Parameters
    ----------
    key : ndarray of integers
        index array (int8, int16, int32 or int64)
    unique_count : int
        max uniques in 'key' array

    Returns
    -------
    The index array will have -MAX_INT set to any locations it could not find a previous
    """
    return rc.MakeiNext(key, unique_count, 1)


# -------------------------------------------------------
def _groupbycalculateall(*args):
    """'
    Arg1 = list or tuple of numpy arrays which has the values to accumulate (often all the columns in a dataset)
    Arg2 = numpy array (int8/16/32/64) which has the index to the unique keys (ikey from MultiKeyGroupBy32)
    Arg3 = integer unique rows
    Arg4 = integer (function number to execute for sum,mean,min, max)

    Example: GroupByOp2(array, self.grouping.ikey, unique_rows, 3)

    Returns a tuple of arrays that match the order of Arg1
    Each array has the length of unique_rows (Arg3)
    The returned arrays have the cells(accum operation) filled in
    """
    return LedgerFunction(rc.GroupByAll32, *args)


# -------------------------------------------------------
def _groupbycalculateallpack(*args):
    """'
    Arg1 = list or tuple of numpy arrays which has the values to accumulate (often all the columns in a dataset)
    Arg2 = iKey = numpy array (int8/16/32/64) which has the index to the unique keys (ikey from MultiKeyGroupBy32)
    Arg3 = iGroup: array size is same as multikey, unique keys are grouped together
    Arg4 = iFirst: index into iGroup
    Arg5 = Count: array size is number of unique keys for the group, is how many member of the group (paired with iFirst)
    Arg6 = integer unique rows
    Arg7 = integer (function number to execute for sum,mean,min, max)
    Arg8 = integer param

    Returns a tuple of arrays that match the order of Arg1
    Each array has the length of unique_rows (Arg6)
    The returned arrays have the cells(accum operation) filled in
    """

    return LedgerFunction(rc.GroupByAllPack32, *args)


# -------------------------------------------------------
# def _groupbycrunch(*args):
#    #return rc.GroupByOp32(*args)
#    return LedgerFunction(rc.GroupByOp32,*args)


# -------------------------------------------------------
def groupbypack(ikey, ncountgroup, unique_count=None, cutoffs=None) -> dict:
    """
    A routine often called after groupbyhash or groupbylex.
    Operates on binned integer arrays only (int8, int16, int32, or int64).

    Parameters
    ----------
    ikey : ndarray of ints
        iKey from groupbyhash or groupbylex
    ncountgroup : ndarray of ints, optional
        From rc.BinCount or hash, if passed in it will be returned unchanged as part of this function's output.
    unique_count : int, optional
        required if `ncountgroup` is None, otherwise not unique_count (scalar int) (must include the 0 bin so +1 often added)
    cutoffs : array_like, optional
        cutoff array for parallel processing

    Returns
    -------
    3 arrays in a dict
    ['iGroup']: array size is same as ikey, unique keys are grouped together
    ['iFirstGroup']: array size is number of unique keys, indexes into iGroup
    ['nCountGroup']: array size is number of unique keys, how many in each group

    Examples
    --------
    >>> np.random.seed(12345)
    >>> c = np.random.randint(0, 8, 10_000)
    >>> x = rt.groupbyhash(c)
    >>> ncountgroup = rc.BinCount(x['iKey'], x['unique_count'] + 1)
    >>> rt.groupbypack(x['iKey'], ncountgroup)
    {'iGroup': FastArray([   0,    9,   21, ..., 9988, 9991, 9992]),
     'iFirstGroup': FastArray([   0,    0, 1213, 2465, 3761, 4987, 6239, 7522, 8797]),
     'nCountGroup': FastArray([   0, 1213, 1252, 1296, 1226, 1252, 1283, 1275, 1203])}

    The sum of the entries in the ``nCountGroup`` array returned by ``groupbypack``
    matches the length of the original array.

    >>> rt.groupbypack(x['iKey'], ncountgroup)['nCountGroup'].sum()
    10000
    """
    dnum = ikey.dtype.num
    if dnum not in [1, 3, 5, 7, 9]:
        raise ValueError("ikey must be int8, int16, int32, or int64")

    if cutoffs is None:
        #
        # new routine written Oct, 2019
        #
        if ncountgroup is None:
            if unique_count is None or not np.isscalar(unique_count):
                raise ValueError("groupbypack: unique_count must be a scalar value if ncountgroup is None")

            # get the unique_count ratio
            ratio = len(ikey) / unique_count
            if len(ikey) > 1_000_000 and ratio < 40:
                nCountGroup = rc.BinCount(ikey, unique_count)
                iGroup, iFirstGroup = rc.GroupFromBinCount(ikey, nCountGroup)
            else:
                # normal path (speed path from Ryan)
                nCountGroup, iGroup, iFirstGroup = rc.BinCount(ikey, unique_count, pack=True)

        else:
            # TJD Oct 2019, this routine is probably slower than BinCount with pack=True
            # high unique routine...
            iGroup, iFirstGroup = rc.GroupFromBinCount(ikey, ncountgroup)
            nCountGroup = ncountgroup

    else:
        #
        # old routine which can take cutoffs
        # TODO: Delete this routine
        iGroup, iFirstGroup, nCountGroup = rc.GroupByPack32(ikey, None, unique_count, cutoffs=cutoffs)

    mkdict = {"iGroup": iGroup, "iFirstGroup": iFirstGroup, "nCountGroup": nCountGroup}
    return mkdict


# -------------------------------------------------------
def groupbyhash(
    list_arrays, hint_size: int = 0, filter=None, hash_mode: int = 2, cutoffs=None, pack: bool = False
) -> dict:
    """
    Find unique values in an array using a linear hashing algorithm.

    Find unique values in an array using a linear hashing algorithm; it will then bin each group
    according to first appearance. The zero bin is reserved for anything filtered out.

    Parameters
    ----------
    list_arrays : ndarray or list of ndarray
        a single numpy array or
        a list of numpy arrays to hash on (multikey) - all arrays must be the same size
    hint_size : int, optional
        An integer hint if the number of unique keys is known in advance, defaults to zero.
    filter : ndarray of bool, optional
        A boolean filter to pre-filter the values on, defaults to None.
    hash_mode : int
        Setting for controlling the hashing mode; defaults to 2. Users generally should not override the default value of this parameter.
    cutoffs : ndarray, optional
        An int64 array of cutoffs, defaults to None.
    pack : bool
        Set to True to return iGroup, iFirstGroup, nCountGroup also; defaults to False.

    Returns
    -------
    A dictionary of 3 arrays
    'iKey' : array size is same as multikey, the unique key for which this row in multikey belongs
    'iFirstKey' : array size is same as unique keys, index into the first row for that unique key
    'unique_count' : number of uniques (not including the zero bin)

    Examples
    --------
    >>> np.random.seed(12345)
    >>> c = np.random.randint(0, 8000, 2_000_000)
    >>> rt.groupbyhash(c)
    {'iKey': FastArray([   1,    2,    3, ..., 6061, 7889, 3002]),
     'iFirstKey': FastArray([    0,     1,     2, ..., 67072, 67697, 68250]),
     'unique_count': 8000,
     'iGroup': None,
     'iFirstGroup': None,
     'nCountGroup': None}

    The 'pack' parameter can be overridden to True to calculate additional information
    about the relationship between elements in the input array and their group. Note this
    information is the same type of information ``groupbylex`` returns by default.

    >>> rt.groupbyhash(c, pack=True)
    {'iKey': FastArray([1, 2, 2, ..., 4, 6, 1]),
     'iFirstKey': FastArray([ 0,  1,  3,  4,  6, 14, 18, 20]),
     'unique_count': 8,
     'iGroup': FastArray([   0,    9,   21, ..., 9988, 9991, 9992]),
     'iFirstGroup': FastArray([   0,    0, 1213, 2465, 3761, 4987, 6239, 7522, 8797]),
     'nCountGroup': FastArray([   0, 1213, 1252, 1296, 1226, 1252, 1283, 1275, 1203])}

    The output from ``groupbyhash`` is useful as an input to ``rc.BinCount``:

    >>> x = rt.groupbyhash(c)
    >>> rc.BinCount(x['iKey'], x['unique_count'] + 1)
    FastArray([  0, 251, 262, ..., 239, 217, 246])

    A filter (boolean array) can be passed to ``groupbyhash``; this causes ``groupbyhash`` to only operate
    on the elements of the input array where the filter has a corresponding True value.

    >>> f = (c % 3).astype(bool)
    >>> rt.groupbyhash(c, filter=f)
    {'iKey': FastArray([   0,    1,    2, ...,    0, 5250, 1973]),
     'iFirstKey': FastArray([    1,     2,     3, ..., 54422, 58655, 68250]),
     'unique_count': 5333,
     'iGroup': None,
     'iFirstGroup': None,
     'nCountGroup': None}

    The ``groupbyhash`` function can also operate on multikeys (tuple keys).

    >>> d = np.random.randint(0, 8000, 2_000_000)
    >>> rt.groupbyhash([c, d])
    {'iKey': FastArray([      1,       2,       3, ..., 1968854, 1968855, 1968856]),
     'iFirstKey': FastArray([      0,       1,       2, ..., 1999997, 1999998, 1999999]),
     'unique_count': 1968856,
     'iGroup': None,
     'iFirstGroup': None,
     'nCountGroup': None}
    """
    if isinstance(list_arrays, np.ndarray):
        list_arrays = [list_arrays]

    if isinstance(list_arrays, list) and len(list_arrays) > 0:
        common_len = {len(arr) for arr in list_arrays}

        if len(common_len) == 1:
            common_len = common_len.pop()
            if common_len != 0:
                iKey, iFirstKey, unique_count = rc.MultiKeyGroupBy32(
                    list_arrays, hint_size, filter, hash_mode, cutoffs=cutoffs
                )
            else:
                iKey = TypeRegister.FastArray([], dtype=np.int32)
                iFirstKey = iKey
                unique_count = 0

            mkdict = {"iKey": iKey, "iFirstKey": iFirstKey, "unique_count": unique_count}

            if pack:
                packdict = groupbypack(iKey, None, unique_count + 1)
                for k, v in packdict.items():
                    mkdict[k] = v
            else:
                # leave empty
                for k in ["iGroup", "iFirstGroup", "nCountGroup"]:
                    mkdict[k] = None

            return mkdict
        raise ValueError(f"groupbyhash all arrays must have same length not {common_len}")
    raise ValueError("groupbyhash first argument is not a list of numpy arrays")


# -------------------------------------------------------
def groupbylex(list_arrays, filter=None, cutoffs=None, base_index: int = 1, rec: bool = False) -> dict:
    """
    Parameters
    ----------
    list_arrays : ndarray or list of ndarray
        A list of numpy arrays to hash on (multikey). All arrays must be the same size.
    filter : ndarray of bool, optional
        A boolean array of true/false filters, defaults to None.
    cutoffs : ndarray, optional
        INT64 array of cutoffs
    base_index : int
    rec : bool
        When set to true, a record array is created, and then the data is sorted.
        A record array is faster, but may not produce a true lexicographical sort.
        Defaults to False.

    Returns
    -------
    A dict of 6 numpy arrays
    iKey: array size is same as multikey, the unique key for which this row in multikey belongs
    iFirstKey: array size is same as unique keys, index into the first row for that unique key
    unique_count: number of uniques
    iGroup: result from lexsort (fancy index sort of list_arrays)
    iFirstGroup: array size is same as unique keys + 1: offset into iGroup
    nCountGroup: array size is same as unique keys + 1: length of slice in iGroup

    Examples
    --------
    >>> a = rt.arange(100).astype('S')
    >>> f = rt.logical(rt.arange(100) % 3)
    >>> rt.groupbylex([a], filter=f)
    {'iKey': FastArray([ 0,  1,  9,  0, 23, 31,  0, 45, 53,  0,  2,  3,  0,  4,  5,  0,
            6,  7,  0,  8, 10,  0, 11, 12,  0, 13, 14,  0, 15, 16,  0, 17,
           18,  0, 19, 20,  0, 21, 22,  0, 24, 25,  0, 26, 27,  0, 28, 29,
            0, 30, 32,  0, 33, 34,  0, 35, 36,  0, 37, 38,  0, 39, 40,  0,
           41, 42,  0, 43, 44,  0, 46, 47,  0, 48, 49,  0, 50, 51,  0, 52,
           54,  0, 55, 56,  0, 57, 58,  0, 59, 60,  0, 61, 62,  0, 63, 64,
            0, 65, 66,  0]),
     'iFirstKey': FastArray([ 1, 10, 11, 13, 14, 16, 17, 19,  2, 20, 22, 23, 25, 26, 28, 29,
           31, 32, 34, 35, 37, 38,  4, 40, 41, 43, 44, 46, 47, 49,  5, 50,
           52, 53, 55, 56, 58, 59, 61, 62, 64, 65, 67, 68,  7, 70, 71, 73,
           74, 76, 77, 79,  8, 80, 82, 83, 85, 86, 88, 89, 91, 92, 94, 95,
           97, 98]),
     'unique_count': 66,
     'iGroup': FastArray([ 1, 10, 11, 13, 14, 16, 17, 19,  2, 20, 22, 23, 25, 26, 28, 29,
           31, 32, 34, 35, 37, 38,  4, 40, 41, 43, 44, 46, 47, 49,  5, 50,
           52, 53, 55, 56, 58, 59, 61, 62, 64, 65, 67, 68,  7, 70, 71, 73,
           74, 76, 77, 79,  8, 80, 82, 83, 85, 86, 88, 89, 91, 92, 94, 95,
           97, 98]),
     'iFirstGroup': FastArray([66,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
           15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
           47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
           63, 64, 65]),
     'nCountGroup': FastArray([34,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1])}
    """
    if base_index != 1 and base_index != 0:
        raise ValueError(f"Invalid base_index {base_index!r}")

    if base_index == 0 and filter is not None:
        raise ValueError("Filter and base_index of 0 cannot be combined")

    if isinstance(list_arrays, np.ndarray):
        list_arrays = [list_arrays]

    if isinstance(list_arrays, list) and len(list_arrays) > 0:
        if not isinstance(list_arrays[0], np.ndarray):
            raise ValueError("groupbylex first argument is not a list of numpy arrays")

        if len(list_arrays) > 1:
            # make a record array for greater comparison speed
            # TODO: future optimization - rotate in parallel
            value_array = np.core.records.fromarrays(list_arrays)

            if rec:
                # user can also specify to sort with all arrays lumped together
                list_arrays = value_array
            else:
                # reverse the order (for lexsort to sort properly)
                list_arrays = list_arrays[::-1]
        else:
            # just one array (no need to make a record array)
            value_array = list_arrays[0]

        index = None

        # check for filtering
        if filter is not None:
            filtertrue, truecount = bool_to_fancy(filter, both=True)
            totalcount = len(filter)

            # build a fancy index to pass to lexsort
            filterfalse = filtertrue[truecount:totalcount]
            filtertrue = filtertrue[0:truecount]
            index = filtertrue

        # lexsort
        if len(value_array) > 2e9:
            iGroup = rc.LexSort64(list_arrays, cutoffs=cutoffs, index=index)
        else:
            iGroup = rc.LexSort32(list_arrays, cutoffs=cutoffs, index=index)

        # make a record array if we did not already because GroupFromLexSort can only handle that
        if isinstance(value_array, list):
            value_array = np.core.records.fromarrays(list_arrays)
        retval = rc.GroupFromLexSort(iGroup, value_array, cutoffs=cutoffs, base_index=base_index)

        if len(retval) == 3:
            iKey, iFirstKey, nCountGroup = retval
        else:
            iKey, iFirstKey, nCountGroup, nUniqueCutoffs = retval

        # print('igroup', len(iGroup), iGroup)
        # print("ikey", len(iKey), iKey)
        # print("ifirstkey", iFirstKey)
        # print("nCountGroup", len(nCountGroup), nCountGroup)

        if base_index == 0:
            iKey += 1
        else:
            # invalid bin count is 0 but we will fix up later if we have a filter
            nCountGroup[0] = 0

        # derive iFirstGroup from nCountGroup
        iFirstGroup = nCountGroup.copy()
        iFirstGroup[1:] = nCountGroup.cumsum()[:-1]

        if filter is not None:
            # the number in the invalid bin is equal to the false filter
            nCountGroup[0] = len(filterfalse)

            # ikey has to get 0s where the filter is false
            iKey[filterfalse] = 0

            # the invalids are after all the valids in the iGroup
            iFirstGroup[0] = len(filtertrue)

        mkdict = {
            "iKey": iKey,
            "iFirstKey": iFirstKey,
            "unique_count": len(iFirstKey),
            "iGroup": iGroup,
            "iFirstGroup": iFirstGroup,
            "nCountGroup": nCountGroup,
        }
        if len(retval) == 4:
            mkdict["nUniqueCutoffs"] = nUniqueCutoffs
        return mkdict

    raise ValueError("groupbylex first argument is not a list of numpy arrays")


# -------------------------------------------------------
def groupby(
    list_arrays,
    filter=None,
    cutoffs=None,
    base_index: int = 1,
    lex: bool = False,
    rec: bool = False,
    pack: bool = False,
    hint_size: int = 0,
):
    """
    Main routine used to groupby one or more keys.

    Parameters
    ----------
    list_arrays : list of ndarray
        A list of numpy arrays to hash on (multikey). All arrays must be the same size.
    filter : ndarray of bool, optional
        A boolean array the same length as the arrays in `list_arrays` used to pre-filter
        the input data before passing it to the grouping algorithm, defaults to None.
    cutoffs : ndarray, optional
        INT64 array of cutoffs
    base_index : int
    lex: defaults to False. if False will call groupbyhash
        If set to true will call groupbylex
    rec : bool
        When set to true, a record array is created, and then the data is sorted.
        A record array is faster, but may not produce a true lexicographical sort.
        Defaults to False. Only applicable when `lex` is True.
    pack : bool
        Set to True to return iGroup, iFirstGroup, nCountGroup also; defaults to False.
        This is only meaningful when using hash-based grouping -- when `lex` is True,
        the sorting-based grouping always computes and returns this information.
    hint_size : int
        An integer hint if the number of unique keys is known in advance, defaults to zero.
        Only applicable when using hash-based grouping (i.e. `lex` is False).

    Notes
    -----
    Ends up calling groupbyhash or groupbylex.

    See Also
    --------
    groupbyhash
    groupbylex
    """
    if lex:
        return groupbylex(list_arrays, filter=filter, cutoffs=cutoffs, base_index=base_index, rec=rec)
    return groupbyhash(list_arrays, filter=filter, cutoffs=cutoffs, pack=pack, hint_size=hint_size)


# -------------------------------------------------------
def multikeyhash(*args):
    """
    Returns 7 arrays to help navigate data.

    Parameters
    ----------
    key
        the unique occurence
    nth
        the nth unique occurence
    bktsize
        how many unique occurences occur
    next, prev
        index to the next unique occurence and previous
    first, last
        index to the first unique occurence and last

    Examples
    --------
    >>> myarr = rt.arange(10) % 3
    >>> myarr
    FastArray([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    >>> mkgrp = rt.Dataset(rt.multikeyhash([myarr]).asdict())
    >>> mkgrp.a = myarr
    >>> mkgrp
    #   key   nth   bktsize   next   prev   first   last   a
    -   ---   ---   -------   ----   ----   -----   ----   -
    0     1     1         4      3     -1       0      9   0
    1     2     1         3      4     -1       1      7   1
    2     3     1         3      5     -1       2      8   2
    3     1     2         4      6      0       0      9   0
    4     2     2         3      7      1       1      7   1
    5     3     2         3      8      2       2      8   2
    6     1     3         4      9      3       0      9   0
    7     2     3         3     -1      4       1      7   1
    8     3     3         3     -1      5       2      8   2
    9     1     4         4     -1      6       0      9   0
    """
    key, nth, bktsize, iprev, inext, ifirst, ilast = rc.MultiKeyHash(*args)
    mkdict = {"key": key, "nth": nth, "bktsize": bktsize, "next": inext, "prev": iprev, "first": ifirst, "last": ilast}
    return TypeRegister.Struct(mkdict)


# -----------------------------------------------------------------------------
# START OF NUMPY OVERLOADS ---------------------------------------------------
# -----------------------------------------------------------------------------
def all(*args, **kwargs) -> bool:
    if isinstance(args[0], np.ndarray):
        return LedgerFunction(np.all, *args, **kwargs)
    # has python built-in
    return builtins.all(*args, **kwargs)


# -------------------------------------------------------
def any(*args, **kwargs) -> bool:
    if isinstance(args[0], np.ndarray):
        return LedgerFunction(np.any, *args, **kwargs)
    # has python built-in
    return builtins.any(*args, **kwargs)


# -------------------------------------------------------
def arange(
    start: Union[int, float] = None,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[npt.Dtype] = None,
    like: npt.ArrayLike = None,
) -> "FastArray":
    """
    Return an array of evenly spaced values within a specified interval.

    The half-open interval includes ``start`` but excludes ``stop``: ``[start, stop)``.

    For integer arguments the function is roughly equivalent to the Python
    built-in :py:obj:`range`, but returns a :py:class:`~.rt_fastarray.FastArray` rather
    than a :py:obj:`range` instance.

    When using a non-integer ``step``, such as 0.1, it's often better to use
    :py:func:`numpy.linspace`.

    For additional warnings, see :py:func:`numpy.arange`.

    Parameters
    ----------
    start : int or float, default 0
        Start of interval. The interval includes this value.
    stop : int or float
        End of interval. The interval does not include this value, except in
        some cases where ``step`` is not an integer and floating point round-off
        affects the length of the output.
    step : int or float, default 1
        Spacing between values. For any output ``out``, this is the distance
        between two adjacent values: ``out[i+1] - out[i]``. If ``step``
        is specified as a positional argument, ``start`` must also be given.
    dtype : str or :py:class:`numpy.dtype` or Riptable dtype, optional
        The type of the output array. If ``dtype`` is not given, the data type
        is inferred from the other input arguments.
    like : array_like, optional
        Reference object to allow the creation of arrays that are not NumPy
        arrays. If an array-like passed in as ``like`` supports the
        ``__array_function__`` protocol, the result is defined by it.
        In this case, it ensures the creation of an array object compatible
        with that passed in via this argument.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray`
        A :py:class:`~.rt_fastarray.FastArray` of evenly spaced numbers within the
        specified interval. For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``. Because of floating point overflow, this rule may
        result in the last element of the output being greater than ``stop``.

    See Also
    --------
    :py:func:`numpy.arange`
    :py:func:`.rt_numpy.ones`
    :py:func:`.rt_numpy.ones_like`
    :py:func:`.rt_numpy.zeros`
    :py:func:`.rt_numpy.zeros_like`
    :py:func:`.rt_numpy.empty`
    :py:func:`.rt_numpy.empty_like`
    :py:func:`.rt_numpy.full`
    :py:func:`.rt_numpy.arange`
    :py:meth:`.rt_categorical.Categorical.full`

    Examples
    --------
    >>> rt.arange(3)
    FastArray([0, 1, 2])

    >>> rt.arange(3.0)
    FastArray([0., 1., 2.])

    >>> rt.arange(3, 7)
    FastArray([3, 4, 5, 6])

    >>> rt.arange(3, 7, 2)
    FastArray([3, 5])
    """
    kwargs = {}
    # Avoid passing thru default 'like' (https://github.com/numpy/numpy/issues/22069, fixed in NumPy-1.24)
    if like is not None:
        kwargs["like"] = like

    if start is None:
        if stop is None:
            return np.arange(step=step, dtype=dtype, **kwargs)  # always an error
        return LedgerFunction(np.arange, stop, step=step, dtype=dtype, **kwargs)

    return LedgerFunction(np.arange, start, stop=stop, step=step, dtype=dtype, **kwargs)


# -------------------------------------------------------
# If argsort implementation changes then add test cases to Python/core/riptable/tests/test_riptable_numpy_equivalency.py.
def argsort(*args, **kwargs) -> FastArray:
    return LedgerFunction(np.argsort, *args, **kwargs)


# -------------------------------------------------------
# This is redefined down below...
# def ceil(*args,**kwargs): return LedgerFunction(np.ceil,*args,**kwargs)


# -------------------------------------------------------
def concatenate(*args, **kwargs):
    firstarg, *_ = args[0]
    if type(firstarg) not in (np.ndarray, TypeRegister.FastArray):
        if kwargs:
            raise ValueError(
                f"concatenate: keyword arguments not supported for arrays of type {type(firstarg)}\n\tGot keyword arguments {kwargs}"
            )
        from .rt_hstack import stack_rows

        return stack_rows(args[0])
    result = LedgerFunction(np.concatenate, *args, **kwargs)
    return result


# -------------------------------------------------------
def crc32c(arr: np.ndarray) -> int:
    """
    Calculate the 32-bit CRC of the data in an array using the Castagnoli polynomial (CRC32C).

    This function does not consider the array's shape or strides when calculating the CRC,
    it simply calculates the CRC value over the entire buffer described by the array.

    Parameters
    ----------
    arr

    Returns
    -------
    int
        The 32-bit CRC value calculated from the array data.

    Notes
    -----
    TODO: Warn when the array has non-default striding, as that is not currently respected by
        the implementation of this function.
    """
    return rc.CalculateCRC(arr)


# -------------------------------------------------------
def crc64(arr: np.ndarray) -> int:
    # TODO: Enable this warning once the remaining code within riptable itself has been migrated to crc32c.
    # warnings.warn("This function is deprecated in favor of the crc32c function and will be removed in the next major version of riptable.", FutureWarning, stacklevel=2)
    return crc32c(arr)


# -------------------------------------------------------
def cumsum(*args, **kwargs) -> FastArray:
    return LedgerFunction(np.cumsum, *args, **kwargs)


# -------------------------------------------------------
def cumprod(*args, **kwargs) -> FastArray:
    return LedgerFunction(np.cumprod, *args, **kwargs)


# -------------------------------------------------------
def diff(*args, **kwargs) -> FastArray:
    return LedgerFunction(np.diff, *args, **kwargs)


# -------------------------------------------------------
# this is a ufunc no need to take over def floor(*args,**kwargs): return LedgerFunction(np.floor,*args,**kwargs)


# -------------------------------------------------------
def full(shape, fill_value, dtype=None, order="C") -> FastArray:
    """
    Return a new array of a specified shape and type, filled with a specified value.

    Parameters
    ----------
    shape : int or sequence of int
        Shape of the new array, e.g., ``(2, 3)`` or ``2``. Note that although
        multi-dimensional arrays are technically supported by Riptable,
        you may get unexpected results when working with them.
    fill_value : scalar or array
        Fill value. For 1-dimensional arrays, only scalar values are accepted.
    dtype : str or :py:class:`numpy.dtype` or Riptable dtype, optional
        The desired data type for the array. The default is the data type that
        would result from creating a :py:class:`~.rt_fastarray.FastArray` with the
        specified ``fill_value``: ``rt.FastArray(fill_value).dtype``.
    order : {'C', 'F'}, default 'C'
        Whether to store multi-dimensional data in row-major (C-style) or
        column-major (Fortran-style) order in memory.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray`
       A new :py:class:`~.rt_fastarray.FastArray` of the specified shape and type,
       filled with the specified value.

    See Also
    --------
    :py:meth:`.rt_categorical.Categorical.full`
    :py:func:`.rt_numpy.ones`
    :py:func:`.rt_numpy.ones_like`
    :py:func:`.rt_numpy.zeros`
    :py:func:`.rt_numpy.zeros_like`
    :py:func:`.rt_numpy.empty`
    :py:func:`.rt_numpy.empty_like`

    Examples
    --------
    >>> rt.full(5, 2)
    FastArray([2, 2, 2, 2, 2])

    >>> rt.full(5, 2.0)
    FastArray([2., 2., 2., 2., 2.])

    Specify a data type:

    >>> rt.full(5, 2, dtype = float)
    FastArray([2., 2., 2., 2., 2.])
    """
    result = LedgerFunction(np.full, shape, fill_value, dtype=dtype, order=order)
    if hasattr(fill_value, "newclassfrominstance"):
        result = fill_value.newclassfrominstance(result, fill_value)
    return result


def full_like(
    a, fill_value, dtype: Optional[npt.DTypeLike] = None, order="K", subok: bool = True, shape=None
) -> "FastArray":
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : array
        The shape and data type of `a` define the same attributes of the
        returned array. Note that although multi-dimensional arrays are
        technically supported by Riptable, you may get unexpected results when
        working with them.
    fill_value : scalar or array_like
        Fill value.
    dtype : str or :py:class:`numpy.dtype` or Riptable dtype, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, default 'K'
        Overrides the memory layout of the result. 'C' means row-major (C-style),
        'F' means column-major (Fortran-style), 'A' means 'F' if `a` is
        Fortran-contiguous, 'C' otherwise. 'K' means match the layout of `a` as
        closely as possible.
    subok : bool, default True
        If True (the default), then the newly created array will use the sub-class
        type of `a`, otherwise it will be a base-class array.
    shape : int or sequence of int, optional
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, it will try to keep the same order; otherwise,
        order='C' is implied. Note that although multi-dimensional arrays are
        technically supported by Riptable, you may get unexpected results when
        working with them.

    Returns
    -------
    `FastArray`
        A `FastArray` with the same shape and data type as the specified array,
        filled with `fill_value`.

    See Also
    --------
    riptable.ones
    riptable.zeros
    riptable.zeros_like
    riptable.empty,
    riptable.empty_like
    riptable.full

    Examples
    --------
    >>> a = rt.FastArray([1, 2, 3, 4])
    >>> rt.full_like(a, 9)
    FastArray([9, 9, 9, 9])

    >>> rt.ones_like(a, dtype = float)
    FastArray([1., 1., 1., 1.])
    """
    result = LedgerFunction(np.full_like, a, fill_value, dtype=dtype, order=order, subok=subok, shape=shape)
    if hasattr(fill_value, "newclassfrominstance"):
        result = fill_value.newclassfrominstance(result, fill_value)
    return result


# -------------------------------------------------------
def lexsort(*args, **kwargs) -> FastArray:
    firstarg = args[0]
    if isinstance(firstarg, tuple):
        firstarg = list(firstarg)
        args = tuple(firstarg) + args[1:]
    if isinstance(firstarg, list):
        firstarg = firstarg[0]

    if isinstance(firstarg, np.ndarray):
        # make sure fastarray
        # also if arraysize > 2billiom call LexSort64 instead
        if firstarg.size > 2e9:
            return rc.LexSort64(*args, **kwargs)
        return rc.LexSort32(*args, **kwargs)
    else:
        return LedgerFunction(np.lexsort, *args, **kwargs)


# -------------------------------------------------------
def ones(shape, dtype=None, order="C", *, like=None) -> "FastArray":
    """
    Return a new array of the specified shape and data type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of int
        Shape of the new array, e.g., ``(2, 3)`` or ``2``. Note that although
        multi-dimensional arrays are technically supported by Riptable,
        you may get unexpected results when working with them.
    dtype : str or :py:class:`numpy.dtype` or Riptable dytpe, default :py:obj:`numpy.float64`
        The desired data type for the array.
    order : {'C', 'F'}, default 'C'
        Whether to store multi-dimensional data in row-major (C-style) or
        column-major (Fortran-style) order in memory.
    like : array_like, default `None`
        Reference object to allow the creation of arrays that are not NumPy
        arrays. If an array-like passed in as ``like`` supports the
        ``__array_function__`` protocol, the result is defined by it.
        In this case, it ensures the creation of an array object compatible
        with that passed in via this argument.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray`
        A new :py:class:`~.rt_fastarray.FastArray` of the specified shape and type,
        filled with ones.

    See Also
    --------
    :py:func:`.rt_numpy.ones_like`
    :py:func:`.rt_numpy.zeros`
    :py:func:`.rt_numpy.zeros_like`
    :py:func:`.rt_numpy.empty`
    :py:func:`.rt_numpy.empty_like`
    :py:func:`.rt_numpy.full`

    Examples
    --------
    >>> rt.ones(5)
    FastArray([1., 1., 1., 1., 1.])

    >>> rt.ones(5, dtype='int8')
    FastArray([1, 1, 1, 1, 1], dtype=int8)
    """
    return LedgerFunction(np.ones, shape, dtype=dtype, order=order, like=like)


def ones_like(a, dtype=None, order="K", subok=True, shape=None) -> "FastArray":
    """
    Return an array of ones with the same shape and data type as the specified array.

    Parameters
    ----------
    a : array
        The shape and data type of ``a`` define the same attributes of the
        returned array. Note that although multi-dimensional arrays are
        technically supported by Riptable, you may get unexpected results when
        working with them.
    dtype : str or :py:class:`numpy.dtype` or Riptable dtype, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, default 'K'
        Overrides the memory layout of the result. 'C' means row-major (C-style),
        'F' means column-major (Fortran-style), 'A' means 'F' if ``a`` is
        Fortran-contiguous, 'C' otherwise. 'K' means match the layout of ``a`` as
        closely as possible.
    subok : bool, default `True`
        If `True` (the default), then the newly created array uses the sub-class
        type of ``a``, otherwise it is a base-class array.
    shape : int or sequence of int, optional
        Overrides the shape of the result. If ``order='K'`` and the number of
        dimensions is unchanged, it tries to keep the same order; otherwise,
        ``order='C'`` is implied. Note that although multi-dimensional arrays are
        technically supported by Riptable, you may get unexpected results when
        working with them.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray`
        A :py:class:`~.rt_fastarray.FastArray` with the same shape and data type as the
        specified array, filled with ones.

    See Also
    --------
    :py:func:`.rt_numpy.ones`
    :py:func:`.rt_numpy.zeros`
    :py:func:`.rt_numpy.zeros_like`
    :py:func:`.rt_numpy.empty`
    :py:func:`.rt_numpy.empty_like`
    :py:func:`.rt_numpy.full`

    Examples
    --------
    >>> a = rt.FastArray([1, 2, 3, 4])
    >>> rt.ones_like(a)
    FastArray([1, 1, 1, 1])

    >>> rt.ones_like(a, dtype = float)
    FastArray([1., 1., 1., 1.])
    """
    return LedgerFunction(np.ones_like, a, dtype=dtype, order=order, subok=subok, shape=shape)


# -------------------------------------------------------
def zeros(shape, dtype=None, order="C", *, like=None) -> "FastArray":
    """
    Return a new array of the specified shape and data type, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of int
        Shape of the new array, e.g., ``(2, 3)`` or ``2``. Note that although
        multi-dimensional arrays are technically supported by Riptable,
        you may get unexpected results when working with them.
    dtype : str or :py:class:`numpy.dtype` or Riptable dtype, default :py:obj:`numpy.float64`
        The desired data type for the array.
    order : {'C', 'F'}, default 'C'
        Whether to store multi-dimensional data in row-major (C-style) or
        column-major (Fortran-style) order in memory.
    like : array_like, default `None`
        Reference object to allow the creation of arrays that are not NumPy
        arrays. If an array-like passed in as ``like`` supports the
        ``__array_function__`` protocol, the result is defined by it.
        In this case, it ensures the creation of an array object compatible
        with that passed in via this argument.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray`
        A new :py:class:`~.rt_fastarray.FastArray` of the specified shape and type,
        filled with zeros.

    See Also
    --------
    :py:func:`.rt_numpy.zeros_like`
    :py:func:`.rt_numpy.ones`
    :py:func:`.rt_numpy.ones_like`
    :py:func:`.rt_numpy.empty`
    :py:func:`.rt_numpy.empty_like`
    :py:func:`.rt_numpy.full`

    Examples
    --------
    >>> rt.zeros(5)
    FastArray([0., 0., 0., 0., 0.])

    >>> rt.zeros(5, dtype = 'int8')
    FastArray([0, 0, 0, 0, 0], dtype=int8)
    """
    kwargs = {}
    # Avoid passing thru default 'like' (https://github.com/numpy/numpy/issues/22069, fixed in NumPy-1.24)
    if like is not None:
        kwargs["like"] = like

    return LedgerFunction(np.zeros, shape, dtype=dtype, order=order, **kwargs)


def zeros_like(a, dtype=None, order="k", subok=True, shape=None) -> "FastArray":
    """
    Return an array of zeros with the same shape and data type as the specified array.

    Parameters
    ----------
    a : array
        The shape and data type of ``a`` define the same attributes of the
        returned array. Note that although multi-dimensional arrays are
        technically supported by Riptable, you may get unexpected results when
        working with them.
    dtype : str or :py:class:`numpy.dtype` or Riptable dtype, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, default 'K'
        Overrides the memory layout of the result. 'C' means row-major (C-style),
        'F' means column-major (Fortran-style), 'A' means 'F' if ``a`` is
        Fortran-contiguous, 'C' otherwise. 'K' means match the layout of ``a`` as
        closely as possible.
    subok : bool, default `True`
        If `True` (the default), then the newly created array uses the sub-class
        type of ``a``, otherwise it is a base-class array.
    shape : int or sequence of int, optional
        Overrides the shape of the result. If ``order='K'`` and the number of
        dimensions is unchanged, it tries to keep the same order; otherwise,
        ``order='C'`` is implied. Note that although multi-dimensional arrays are
        technically supported by Riptable, you may get unexpected results when
        working with them.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray`
        A :py:class:`~.rt_fastarray.FastArray` with the same shape and data type as the
        specified array, filled with zeros.

    See Also
    --------
    :py:func:`.rt_numpy.zeros`
    :py:func:`.rt_numpy.ones`
    :py:func:`.rt_numpy.ones_like`
    :py:func:`.rt_numpy.empty`
    :py:func:`.rt_numpy.empty_like`
    :py:func:`.rt_numpy.full`

    Examples
    --------
    >>> a = rt.FastArray([1, 2, 3, 4])
    >>> rt.zeros_like(a)
    FastArray([0, 0, 0, 0])

    >>> rt.zeros_like(a, dtype=float)
    FastArray([0., 0., 0., 0.])
    """
    return LedgerFunction(np.zeros_like, a, dtype=dtype, order=order, subok=subok, shape=shape)


# -------------------------------------------------------
def reshape(*args, **kwargs):
    return LedgerFunction(np.reshape, *args, **kwargs)


# -------------------------------------------------------
# a faster way to do array index masks
def reindex_fast(index, array):
    if isinstance(index, np.ndarray) and isinstance(array, np.ndarray):
        return rc.ReIndex(index, array)
    raise TypeError(
        "two arguments, both args must be numpy arrays.  the first argument indexes into the second argument."
    )


# -------------------------------------------------------
# If sort implementation changes then add test cases to Python/core/riptable/tests/test_riptable_numpy_equivalency.py.
def sort(*args, **kwargs) -> FastArray:
    return LedgerFunction(np.sort, *args, **kwargs)


# -------------------------------------------------------
# If transpose implementation changes then add test cases to Python/core/riptable/tests/test_riptable_numpy_equivalency.py.
def transpose(*args, **kwargs):
    return LedgerFunction(np.transpose, *args, **kwargs)


# -------------------------------------------------------
def where(condition, x=None, y=None) -> FastArray | tuple[FastArray, ...]:
    """
    Return a new :py:class:`~.rt_fastarray.FastArray` or
    :py:class:`~.rt_categorical.Categorical` with elements from ``x`` or ``y``
    depending on whether ``condition`` is `True`.

    For 1-dimensional arrays, this function is equivalent to::

        [xv if c else yv
         for c, xv, yv in zip(condition, x, y)]

    If only ``condition`` is provided, this function returns a tuple containing
    an integer :py:class:`~.rt_fastarray.FastArray` with the indices where ``condition``
    is `True`. Note that this usage of :py:func:`~.rt_numpy.where` is not supported for
    :py:class:`~.rt_fastarray.FastArray` objects of more than one dimension.

    Note also that this case of :py:func:`~.rt_numpy.where` uses
    :py:func:`~.rt_numpy.bool_to_fancy`. Using :py:func:`~.rt_numpy.bool_to_fancy`
    directly is preferred, as it behaves correctly for subclasses.

    Parameters
    ----------
    condition : bool or array of bool
        Where `True`, yield ``x``, otherwise yield ``y``.
    x : scalar, array, or callable, optional
        The value to use where ``condition`` is `True`. If ``x`` is provided, ``y``
        must also be provided, and ``x`` and ``y`` should be the same type. If ``x``
        is an array, a callable that returns an array, or a
        :py:class:`~.rt_categorical.Categorical`, it must be the same length as
        ``condition``. The value of ``x`` that corresponds to the `True` value is used.
    y : scalar, array, or callable, optional
        The value to use where ``condition`` is `False`. If ``y`` is provided, ``x``
        must also be provided, and ``x`` and ``y`` should be the same type. If ``y``
        is an array, a callable that returns an array, or a
        :py:class:`~.rt_categorical.Categorical`, it must be the same length as
        ``condition``. The value of ``y`` that corresponds to the `False` value is used.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray` or :py:class:`~.rt_categorical.Categorical` or tuple
        If ``x`` and ``y`` are :py:class:`~.rt_categorical.Categorical` objects, a
        :py:class:`~.rt_categorical.Categorical` is returned. Otherwise, if ``x`` and
        ``y`` are provided a :py:class:`~.rt_fastarray.FastArray` is returned. When
        only ``condition`` is provided, a tuple is returned containing an integer
        :py:class:`~.rt_fastarray.FastArray` with the indices where the condition is
        `True`.

    See Also
    --------
    :py:meth:`.rt_fastarray.FastArray.where` :
        Replace values where a given condition is `False`.
    :py:func:`.rt_numpy.bool_to_fancy` :
        The function called when ``x`` and ``y`` are omitted.

    Examples
    --------
    ``condition`` is a comparison that creates an array of booleans, and ``x``
    and ``y`` are scalars:

    >>> a = rt.FastArray(rt.arange(5))
    >>> a
    FastArray([0, 1, 2, 3, 4])
    >>> rt.where(a < 2, 100, 200)
    FastArray([100, 100, 200, 200, 200], dtype=uint8)

    ``condition`` and ``x`` are same-length arrays, and ``y`` is a
    scalar:

    >>> condition = rt.FastArray([False, False, True, True, True])
    >>> x = rt.FastArray([100, 101, 102, 103, 104])
    >>> y = 200
    >>> rt.where(condition, x, y)
    FastArray([200, 200, 102, 103, 104])

    When ``x`` and ``y`` are :py:class:`~.rt_categorical.Categorical` objects, a
    :py:class:`~.rt_categorical.Categorical` is returned:

    >>> primary_traders = rt.Cat(['John', 'Mary', 'John', 'Mary', 'John', 'Mary'])
    >>> secondary_traders = rt.Cat(['Chris', 'Duncan', 'Chris', 'Duncan', 'Duncan', 'Chris'])
    >>> is_primary = rt.FA([True, True, False, True, False, True])
    >>> rt.where(is_primary, primary_traders, secondary_traders)
    Categorical([John, Mary, Chris, Mary, Duncan, Mary]) Length: 6
      FastArray([3, 4, 1, 4, 2, 4], dtype=int8) Base Index: 1
      FastArray([b'Chris', b'Duncan', b'John', b'Mary'], dtype='|S6') Unique count: 4

    When ``x`` and ``y`` are :py:class:`~.rt_datetime.Date` objects, a
    :py:class:`~.rt_fastarray.FastArray` of integers is returned that can be converted
    to a :py:class:`~.rt_datetime.Date` (other :py:obj:`.rt_datetime` objects are similar):

    >>> x = rt.Date(['20230101', '20220101', '20210101'])
    >>> y = rt.Date(['20190101', '20180101', '20170101'])
    >>> condition = x > rt.Date(['20211231'])
    >>> rt.where(condition, x, y)
    FastArray([19358, 18993, 17167], dtype=int32)
    >>> rt.FastArray([19358, 18993, 17167])
    FastArray([19358, 18993, 17167])
    >>> rt.Date(_)
    Date(['2023-01-01', '2022-01-01', '2017-01-01'])

    When only a ``condition`` is provided, a tuple is returned containing a
    :py:class:`~.rt_fastarray.FastArray` with the indices where the ``condition`` is
    `True`:

    >>> a = rt.FastArray([10, 20, 30, 40, 50])
    >>> rt.where(a < 40)
    (FastArray([0, 1, 2], dtype=int32),)
    """
    if isinstance(x, TypeRegister.Categorical) and isinstance(y, TypeRegister.Categorical):
        z = TypeRegister.Categorical.hstack([x, y])
        my_fa = where(condition, x=z._fa[: len(x)], y=z._fa[len(x) :])
        return TypeRegister.Categorical(my_fa, z._categories)

    if isinstance(x, TypeRegister.Categorical):
        x = x.expand_array
    if isinstance(y, TypeRegister.Categorical):
        y = y.expand_array

    missing = (x is None, y is None).count(True)
    if missing == 2:
        return (bool_to_fancy(condition),)

    # handle the single-argument case
    if missing == 1:
        raise ValueError(f"where: must provide both 'x' and 'y' or neither. x={x}  y={y}")

    # Invoke np.where, requiring that both alternates must be non-empty arrays.
    def delegate_to_numpy(condition, x, y):
        if not np.isscalar(x) and len(x) == 0:
            raise ValueError(f"where: x must not be empty")
        if not np.isscalar(y) and len(y) == 0:
            raise ValueError(f"where: y must not be empty")

        return LedgerFunction(np.where, condition, x, y)

    common_dtype = get_common_dtype(x, y)

    if not isinstance(condition, np.ndarray):
        if condition is False or condition is True:
            # punt to normal numpy instead of error which may process None differently
            return delegate_to_numpy(condition, x, y)

        condition = TypeRegister.FastArray(condition)

    if len(condition) == 1:
        # punt to normal numpy since an array of 1
        return delegate_to_numpy(condition, x, y)

    if len(condition) == 0:
        return TypeRegister.FastArray([], dtype=common_dtype)

    if condition.ndim > 1:
        # punt to normal numpy since more than one dimension
        return delegate_to_numpy(condition, x, y)

    if condition.dtype != bool:
        # NOTE: believe numpy just flips it to boolean using astype, where object arrays handled differently with None and 0
        condition = condition != 0

    # this is the normal 3 argument where

    # see if we can accelerate where
    if common_dtype.char in NumpyCharTypes.SupportedAlternate:

        def _possibly_convert(arr):
            # NOTE detect scalars first?
            try:
                if arr.dtype.num != common_dtype.num:
                    # perform a safe conversion understanding sentinels
                    # print("Converting1 to", common_dtype, arr.dtype.num, common_dtype.num)
                    arr = TypeRegister.MathLedger._AS_FA_TYPE(arr, common_dtype.num)
                elif arr.itemsize != common_dtype.itemsize:
                    # make strings sizes the same
                    arr = arr.astype(common_dtype)

            except:
                arr = TypeRegister.FastArray(arr)
                if arr.dtype.num != common_dtype.num:
                    # print("Converting2 to", common_dtype, arr.dtype.num, common_dtype.num)
                    arr = arr.astype(common_dtype)

            # strided check
            # if arr.ndim ==1 and arr.itemsize != arr.strides[0]:
            #    arr = arr.copy()

            # check if can make like a scalar
            try:
                if len(arr) == 1:
                    # print("array len1 detected")
                    if common_dtype.char in NumpyCharTypes.AllFloat:
                        arr = float(arr[0])
                    elif common_dtype.char in NumpyCharTypes.AllInteger:
                        arr = int(arr[0])
                    else:
                        arr = arr[0]
            except:
                # probably cannot take len, might be numpy scalar
                num = arr.dtype.num
                if num == 0:
                    arr = bool(arr)
                elif num <= 10:
                    arr = int(arr)
                elif num <= 13:
                    arr = float(arr)
                elif num == 18:
                    arr = str(arr)
                elif num == 19:
                    arr = bytes(arr)
                else:
                    raise TypeError(f"Do not understand type {arr.dtype!r}")

            return arr

        # check if we need to upcast
        x = _possibly_convert(x)
        y = _possibly_convert(y)

        # call down into C++ version of where.
        # result will be None if operation is not supported.
        result = rc.Where(condition, (x, y), common_dtype.num, common_dtype.itemsize)
        if result is not None:
            return result

    # punt to normal numpy
    return delegate_to_numpy(condition, x, y)


# -------------------------------------------------------
def sortinplaceindirect(*args, **kwargs):
    return LedgerFunction(rc.SortInPlaceIndirect, *args, **kwargs)


# -------------------------------------------------------
# is a ufunc def trunc(*args,**kwargs): return LedgerFunction(rc.IsSorted,*args,**kwargs)


# -------------------------------------------------------
def _unary_func(func, *args, **kwargs):
    """
    pooling of unary functions
    if not a fastarray, it will try to convert it
    then it will call the normal numpy routine, which will call FastArray unary func (which is parallelized)
    if a user calls rt.log(pandasarray) with a pandas array, it will get parallelized now
    """
    if len(args) == 1:
        a = args[0]
        # if they pass a list, we do not bother to convert it (possible future improvement)
        if isinstance(a, np.ndarray):
            try:
                # try to convert to FastArray so that it will route to TypeRegister.FastArray's array_ufunc
                if not isinstance(a, TypeRegister.FastArray):
                    a = a.view(TypeRegister.FastArray)
                return func(a, **kwargs)
            except:
                # fall through and call normal numpy
                pass
    return func(*args, **kwargs)


# -------------------------------------------------------
def _convert_cat_args(args):
    if len(args) == 1:
        if isinstance(args[0], TypeRegister.Categorical):
            args = (args[0]._fa,)
        return args
    return args


# -------------------------------------------------------
def nan_to_num(*args, **kwargs):
    """
    arg1: ndarray
    returns: ndarray with nan_to_num
    notes: if you want to do this inplace contact TJD
    """
    return np.nan_to_num(*args, **kwargs)


# -------------------------------------------------------
def nan_to_zero(a: np.ndarray) -> np.ndarray:
    """
    Replace the NaN or invalid values in an array with zeroes.

    This is an in-place operation -- the input array is returned after being modified.

    Parameters
    ----------
    a : ndarray
        The input array.

    Returns
    -------
    ndarray
        The input array `a` (after it's been modified).
    """
    where_are_NaNs = isnan(a)
    a[where_are_NaNs] = 0
    return a


# -------------------------------------------------------
def ceil(*args, **kwargs) -> FastArray | np.number:
    return _unary_func(np.ceil, *args, **kwargs)


# -------------------------------------------------------
def floor(*args, **kwargs) -> FastArray | np.number:
    return _unary_func(np.floor, *args, **kwargs)


# -------------------------------------------------------
def trunc(*args, **kwargs) -> FastArray | np.number:
    return _unary_func(np.trunc, *args, **kwargs)


# -------------------------------------------------------
def log(*args, **kwargs) -> FastArray | np.number:
    return _unary_func(np.log, *args, **kwargs)


# -------------------------------------------------------
def log10(*args, **kwargs) -> FastArray | np.number:
    return _unary_func(np.log10, *args, **kwargs)


# -------------------------------------------------------
def absolute(*args, **kwargs) -> FastArray | np.number:
    return _unary_func(np.absolute, *args, **kwargs)


# -------------------------------------------------------
def power(*args, **kwargs) -> FastArray | np.number:
    return _unary_func(np.power, *args, **kwargs)


# -------------------------------------------------------
def abs(*args, **kwargs) -> FastArray | np.number | Dataset:
    """
    This will check for numpy array first and call np.abs
    """
    a = args[0]
    if isinstance(a, np.ndarray):
        if not isinstance(a, TypeRegister.FastArray):
            a = TypeRegister.FastArray(a)
        return np.abs(*args, **kwargs)
    return builtins.abs(a)


# -------------------------------------------------------
def round(*args, **kwargs) -> FastArray | np.number:
    """
    This will check for numpy array first and call np.round
    """
    a = args[0]
    if isinstance(a, np.ndarray):
        return np.round(*args, **kwargs)
    return builtins.round(a)


def _np_keyword_wrapper(filter=None, dtype=None, **kwargs):
    if dtype is not None:
        kwargs["dtype"] = dtype
    if filter is not None:
        kwargs["filter"] = filter

    return kwargs


# -------------------------------------------------------
def sum(*args, filter=None, dtype=None, **kwargs) -> np.number | Dataset:
    """
    Compute the sum of the values in the first argument.

    When possible, ``rt.sum(x, *args)`` calls ``x.sum(*args)``; look there for
    documentation. In particular, note whether the called function accepts the
    keyword arguments listed below. For example, :py:meth:`.rt_dataset.Dataset.sum`
    does not accept the ``filter`` or ``dtype`` keyword arguments.

    When a :py:class:`~.rt_fastarray.FastArray` is passed to :py:func:`.rt_numpy.sum`,
    :py:func:`numpy.sum` is called. See the documentation for :py:func:`numpy.sum`, but
    note the following:

    * Until a reported bug is fixed, the ``dtype`` keyword argument may not work
      as expected:

        * Riptable data types (for example, :py:obj:`.rt_numpy.float64`) are ignored.
        * NumPy integer data types (for example, :py:obj:`numpy.int32`) are also ignored.
        * NumPy floating point data types are applied as :py:obj:`numpy.float64`.

    * If you include another NumPy parameter (for example, ``axis=0``), :py:func:`numpy.sum`
      is used and the ``dtype`` is used to compute the sum.

    Parameters
    ----------
    *args : array or iterable or scalar value
        Contains the values that are used to calculate the sum.
    filter : array of bool, default `None`
        Specifies which elements to include in the sum calculation.
    dtype : :py:class:`numpy.dtype` or Riptable dtype, optional
        The data type of the result. If not specified, the default ``dtype`` depends on
        the input values. For example:

            - For a :py:class:`~.rt_fastarray.FastArray` with `int` values, the resulting
              ``dtype`` is :py:obj:`numpy.int64`.
            - For a :py:class:`~.rt_fastarray.FastArray` with `float` values, the
              resulting ``dtype`` is :py:obj:`numpy.float64`.
            - For a list with `int` values, the resulting ``dtype`` is `int`.
            - For a list with `float` values, the resulting ``dtype`` is `float`.

        See the notes above about using this keyword argument with
        :py:class:`~.rt_fastarray.FastArray` objects as input.
    **kwargs :
        Additional keyword arguments to be passed to the function. See
        :py:func:`numpy.sum` for additional keyword arguments.

    Returns
    -------
    scalar or :py:class:`~.rt_dataset.Dataset`
        Scalar for :py:class:`~.rt_fastarray.FastArray` input. For
        :py:class:`~.rt_dataset.Dataset` input, returns a :py:class:`~.rt_dataset.Dataset`
        consisting of a row with each numerical column's sum.

    See Also
    --------
    :py:func:`numpy.sum` :
        Sum of array elements over a given axis.
    :py:func:`.rt_numpy.nansum` :
        Sums the values, ignoring ``NaN`` values.
    :py:meth:`.rt_dataset.Dataset.sum` :
        Sums the values of numerical :py:class:`~.rt_dataset.Dataset` columns.
    :py:meth:`.rt_groupbyops.GroupByOps.sum` :
        Sums the values of each group. Used by :py:class:`~.rt_categorical.Categorical`
        objects.

    Examples
    --------
    >>> a = rt.FastArray([1, 3, 5, 7])
    >>> rt.sum(a)
    16

    >>> a = rt.FastArray([1.0, 3.0, 5.0, 7.0])
    >>> rt.sum(a)
    16.0
    """
    kwargs = _np_keyword_wrapper(filter=filter, dtype=dtype, **kwargs)
    args = _convert_cat_args(args)
    if hasattr(args[0], "sum"):
        return args[0].sum(*args[1:], **kwargs)
    return builtins.sum(*args, **kwargs)


# -------------------------------------------------------
def nansum(*args, filter=None, dtype=None, **kwargs) -> np.number | Dataset:
    """
    Compute the sum of the values in the first argument, ignoring ``NaN`` values.

    If all values in the first argument are ``NaN`` values, ``0.0`` is returned.

    When possible, ``rt.nansum(x, *args)`` calls ``x.nansum(*args)``; look there for
    documentation. In particular, note whether the called function accepts the keyword
    arguments listed below.

    For example, :py:meth:`.rt_fastarray.FastArray.nansum` accepts the ``filter`` and
    ``dtype`` keyword arguments, but :py:meth:`.rt_dataset.Dataset.nansum` does not.

    Parameters
    ----------
    *args : array or iterable or scalar value
        Contains the values that are used to calculate the sum.
    filter : array of bool, default `None`
        Specifies which elements to include in the sum calculation. If the filter is
        uniformly `False`, the method returns ``0.0``.
    dtype : :py:class:`numpy.dtype` or Riptable dtype, default :py:obj:`numpy.float64`
        The data type of the result. For a :py:class:`~.rt_fastarray.FastArray` ``x``,
        ``x.nansum(dtype = my_type)`` is equivalent to ``my_type(x.nansum())``.
    **kwargs :
        Additional keyword arguments to be passed to the function. See
        :py:func:`numpy.nansum` for additional keyword arguments.

    Returns
    -------
    scalar or :py:class:`~.rt_dataset.Dataset`
        Scalar for :py:class:`~.rt_fastarray.FastArray` input. For
        :py:class:`~.rt_dataset.Dataset` input, returns a :py:class:`~.rt_dataset.Dataset`
        consisting of a row with each numerical column's sum.

    See Also
    --------
    :py:func:`numpy.nansum` :
        Return the sum of array elements over a given axis treating Not a Numbers
        (``NaN``) as zero.
    :py:func:`.rt_numpy.sum` :
        Sums the values of the input.
    :py:meth:`.rt_fastarray.FastArray.nansum` :
        Sums the values of a :py:class:`~.rt_fastarray.FastArray`, ignoring ``NaN`` values.
    :py:meth:`.rt_dataset.Dataset.nansum` :
        Sums the values of numerical :py:class:`~.rt_dataset.Dataset` columns, ignoring
        ``NaN`` values.
    :py:meth:`.rt_groupbyops.GroupByOps.nansum` :
        Sums the values of each group, ignoring ``NaN`` values. Used by
        :py:class:`~.rt_categorical.Categorical` objects.

    Notes
    -----
    The ``dtype`` parameter specifies the data type of the result. This
    differs from :py:func:`numpy.nansum`, where it specifies the data type used to
    compute the sum.

    Examples
    --------
    >>> a = rt.FastArray( [1, 3, 5, 7, rt.nan])
    >>> rt.nansum(a)
    16.0

    With a ``dtype`` specified:

    >>> a = rt.FastArray([1.0, 3.0, 5.0, 7.0, rt.nan])
    >>> rt.nansum(a, dtype = rt.int32)
    16

    With a ``filter``:

    >>> a = rt.FastArray([1, 3, 5, 7, rt.nan])
    >>> b = rt.FastArray([False, True, False, True, True])
    >>> rt.nansum(a, filter = b)
    10.0
    """
    kwargs = _np_keyword_wrapper(filter=filter, dtype=dtype, **kwargs)
    args = _convert_cat_args(args)
    if hasattr(args[0], "nansum"):
        return args[0].nansum(*args[1:], **kwargs)
    return np.nansum(*args, **kwargs)


# -------------------------------------------------------
def argmax(*args, **kwargs) -> int:
    args = _convert_cat_args(args)
    if isinstance(args[0], np.ndarray):
        return args[0].argmax(**kwargs)
    return np.argmax(*args, **kwargs)


# -------------------------------------------------------
def argmin(*args, **kwargs) -> int:
    args = _convert_cat_args(args)
    if isinstance(args[0], np.ndarray):
        return args[0].argmin(**kwargs)
    return np.argmin(*args, **kwargs)


# -------------------------------------------------------
def nanargmax(*args, **kwargs) -> int:
    args = _convert_cat_args(args)
    if isinstance(args[0], np.ndarray):
        return args[0].nanargmax(**kwargs)
    return np.nanargmax(*args, **kwargs)


# -------------------------------------------------------
def nanargmin(*args, **kwargs) -> int:
    args = _convert_cat_args(args)
    if isinstance(args[0], np.ndarray):
        return args[0].nanargmin(**kwargs)
    return np.nanargmin(*args, **kwargs)


# -------------------------------------------------------
def _reclaim_type(arr, x1, x2):
    if isinstance(arr, np.ndarray) and isinstance(x1, (TypeRegister.DateTimeBase, TypeRegister.DateBase)):
        # handle case when DateTime used (only checks first array not second)
        arrtype = type(x1)
        if not isinstance(arr, arrtype):
            arr = arrtype(arr)
    return arr


# -------------------------------------------------------
def maximum(x1, x2, *args, **kwargs):
    # two arrays are passed to maximum, minimum
    return _reclaim_type(np.maximum(x1, x2, *args, **kwargs), x1, x2)


# -------------------------------------------------------
def minimum(x1, x2, *args, **kwargs):
    return _reclaim_type(np.minimum(x1, x2, *args, **kwargs), x1, x2)


# -------------------------------------------------------
def max(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0], np.ndarray):
        badlist = ["S", "O", "U"]
        if not args[0].dtype.char in badlist:
            if len(args) > 1:
                return maximum(*args, **kwargs)
            return args[0].max(**kwargs)
        else:
            # Object, String, Unicode
            if len(args) == 1:
                # assuming they want length of string
                return builtins.max([item for item in args[0]], **kwargs)
            else:
                warnings.warn("Getting the max of two objects or string arrays is not currently allowed", stacklevel=2)
                return None
    return builtins.max(*args, **kwargs)


# -------------------------------------------------------
def min(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0], np.ndarray):
        badlist = ["S", "O", "U"]
        if not args[0].dtype.char in badlist:
            if len(args) > 1:
                return minimum(*args, **kwargs)
            return args[0].min(**kwargs)
        else:
            # Object, String, Unicode
            if len(args) == 1:
                # assuming they want length of string
                return builtins.min([item for item in args[0]], **kwargs)
            else:
                warnings.warn("Getting the max of two objects or string arrays is not currently allowed", stacklevel=2)
                return None
    return builtins.min(*args, **kwargs)


def nanmin(*args, **kwargs):
    if not args:
        raise ValueError("No arguments provided.")

    # If needed, convert the first argument to an array;
    # use np.asanyarray so if the first argument is already an array,
    # it's subclass (if applicable) will be preserved.
    firstarg = np.asanyarray(args[0])

    # If the first argument is a Categorical, it must be ordered. Comparison operations
    # are undefined (semantically speaking) for an unordered Categorical; and if elements
    # can't be compared, min/max operations are meaningless.
    # If this is an ordered Categorical, extract the underlying array -- we'll operate on that.
    is_cat = isinstance(firstarg, TypeRegister.Categorical)
    if is_cat:
        if firstarg.ordered:
            firstarg = firstarg._fa
        else:
            raise ValueError("Cannot calculate a comparison-based reduction like 'nanmin' on an unordered Categorical.")

    if len(args) > 1:
        return np.fmin(*args, **kwargs)

    else:
        # Call the implementation of this function in riptable_cpp via the Ledger class/dispatcher.
        if isinstance(firstarg, TypeRegister.FastArray):
            result = TypeRegister.MathLedger._REDUCE(firstarg, REDUCE_FUNCTIONS.REDUCE_NANMIN)
        else:
            result = LedgerFunction(np.nanmin, *args, **kwargs)

        # If the result is a NaN or an array containing a NaN, the input data contained
        # all NaNs (or at least a slice of the specified axis did).
        # Issue a warning here before returning to match the behavior of numpy.
        if np.isnan(result).any():
            warnings.warn("All-NaN axis encountered", RuntimeWarning, stacklevel=2)

        # TODO: Based on the original input type, need to convert the raw result
        #       back to an array or scalar of the correct type. E.g. if the input
        #       is a Date instance, we need to return a DateScalar.
        return result


def nanmax(*args, **kwargs):
    if not args:
        raise ValueError("No arguments provided.")

    # If needed, convert the first argument to an array;
    # use np.asanyarray so if the first argument is already an array,
    # it's subclass (if applicable) will be preserved.
    firstarg = np.asanyarray(args[0])

    # If the first argument is a Categorical, it must be ordered. Comparison operations
    # are undefined (semantically speaking) for an unordered Categorical; and if elements
    # can't be compared, min/max operations are meaningless.
    # If this is an ordered Categorical, extract the underlying array -- we'll operate on that.
    is_cat = isinstance(firstarg, TypeRegister.Categorical)
    if is_cat:
        if firstarg.ordered:
            firstarg = firstarg._fa
        else:
            raise ValueError("Cannot calculate a comparison-based reduction like 'nanmax' on an unordered Categorical.")

    if len(args) > 1:
        return np.fmax(*args, **kwargs)

    else:
        # Call the implementation of this function in riptable_cpp via the Ledger class/dispatcher.
        if isinstance(firstarg, TypeRegister.FastArray):
            result = TypeRegister.MathLedger._REDUCE(firstarg, REDUCE_FUNCTIONS.REDUCE_NANMAX)
        else:
            result = LedgerFunction(np.nanmax, *args, **kwargs)

        # If the result is a NaN or an array containing a NaN, the input data contained
        # all NaNs (or at least a slice of the specified axis did).
        # Issue a warning here before returning to match the behavior of numpy.
        if np.isnan(result).any():
            warnings.warn("All-NaN axis encountered", RuntimeWarning, stacklevel=2)

        # TODO: Based on the original input type, need to convert the raw result
        #       back to an array or scalar of the correct type. E.g. if the input
        #       is a Date instance, we need to return a DateScalar.
        return result


# -------------------------------------------------------
def mean(*args, filter=None, dtype=None, **kwargs) -> np.number | Dataset:
    """
    Compute the arithmetic mean of the values in the first argument.

    When possible, ``rt.mean(x, *args)`` calls ``x.mean(*args)``; look there for
    documentation. In particular, note whether the called function accepts the keyword
    arguments listed below.

    For example, :py:meth:`.rt_fastarray.FastArray.mean` accepts the ``filter`` and
    ``dtype`` keyword arguments, but :py:meth:`.rt_dataset.Dataset.mean` does not.

    Parameters
    ----------
    *args : array or iterable or scalar value
        Contains the values that are used to calculate the mean.
    filter : array of bool, default `None`
        Specifies which elements to include in the mean calculation. If the filter is
        uniformly `False`, :py:func:`~.rt_numpy.mean` returns a :py:class:`ZeroDivisionError`.
    dtype : :py:class:`numpy.dtype` or Riptable dtype, default :py:obj:`numpy.float64`
        The data type of the result. For a :py:class:`~.rt_fastarray.FastArray` ``x``,
        ``x.mean(dtype = my_type)`` is equivalent to ``my_type(x.mean())``.
    **kwargs :
        Additional keyword arguments to be passed to the function. See
        :py:func:`numpy.mean` for additional keyword arguments.

    Returns
    -------
    scalar or :py:class:`~.rt_dataset.Dataset`
        Scalar for :py:class:`~.rt_fastarray.FastArray` input. For
        :py:class:`~.rt_dataset.Dataset` input, returns a :py:class:`~.rt_dataset.Dataset`
        consisting of a row with each numerical column's mean.

    See Also
    --------
    :py:func:`numpy.mean` :
        Computes the arithmetic mean along the specified axis.
    :py:func:`.rt_numpy.nanmean` :
        Computes the mean, ignoring ``NaN`` values.
    :py:meth:`.rt_dataset.Dataset.mean` :
        Computes the mean of numerical :py:class:`~.rt_dataset.Dataset` columns.
    :py:meth:`.rt_fastarray.FastArray.mean` :
        Computes the mean of :py:class:`~.rt_fastarray.FastArray` values.
    :py:meth:`.rt_groupbyops.GroupByOps.mean` :
        Computes the mean of each group. Used by :py:class:`~.rt_categorical.Categorical`
        objects.

    Notes
    -----
    The ``dtype`` parameter specifies the data type of the result. This differs from
    :py:func:`numpy.mean`, where it specifies the data type used to compute the mean.

    Examples
    --------
    >>> a = rt.FastArray([1, 3, 5, 7])
    >>> rt.mean(a)
    4.0

    With a ``dtype`` specified:

    >>> a = rt.FastArray([1, 3, 5, 7])
    >>> rt.mean(a, dtype = rt.int32)
    4

    With a ``filter``:

    >>> a = rt.FastArray([1, 3, 5, 7])
    >>> b = rt.FastArray([False, True, False, True])
    >>> rt.mean(a, filter = b)
    5.0
    """
    args = _convert_cat_args(args)
    kwargs = _np_keyword_wrapper(filter=filter, dtype=dtype, **kwargs)
    if hasattr(args[0], "mean"):
        return args[0].mean(**kwargs)
    return np.mean(*args, **kwargs)


# -------------------------------------------------------
def nanmean(*args, filter=None, dtype=None, **kwargs) -> np.number | Dataset:
    """
    Compute the arithmetic mean of the values in the first argument, ignoring ``NaN``
    values.

    If all values in the first argument are ``NaN`` values, ``0.0`` is returned.

    When possible, ``rt.nanmean(x, *args)`` calls ``x.nanmean(*args)``; look there for
    documentation. In particular, note whether the called function accepts the keyword
    arguments listed below.

    For example, :py:meth:`.rt_fastarray.FastArray.nanmean` accepts the ``filter`` and
    ``dtype`` keyword arguments, but :py:meth:`.rt_dataset.Dataset.nanmean` does not.

    Parameters
    ----------
    *args : array or iterable or scalar value
        Contains the values that are used to calculate the mean.
    filter : array of bool, default `None`
        Specifies which elements to include in the mean calculation. If the filter is
        uniformly `False`, the method returns a :py:class:`ZeroDivisionError`.
    dtype : :py:class:`numpy.dtype` or Riptable dtype, default :py:obj:`numpy.float64`
        The data type of the result. For a :py:class:`~.rt_fastarray.FastArray` ``x``,
        ``x.nanmean(dtype = my_type)`` is equivalent to ``my_type(x.nanmean())``.
    **kwargs :
        Additional keyword arguments to be passed to the function. See
        :py:func:`numpy.nanmean` for additional keyword arguments.

    Returns
    -------
    scalar or :py:class:`~.rt_dataset.Dataset`
        Scalar for :py:class:`~.rt_fastarray.FastArray` input. For
        :py:class:`~.rt_dataset.Dataset` input, returns a :py:class:`~.rt_dataset.Dataset`
        consisting of a row with each numerical column's mean.

    See Also
    --------
    :py:func:`numpy.nanmean` :
        Compute the arithmetic mean along the specified axis, ignoring ``NaN`` values.
    :py:func:`.rt_numpy.mean` :
        Computes the mean.
    :py:meth:`.rt_dataset.Dataset.nanmean` :
        Computes the mean of numerical :py:class:`~.rt_dataset.Dataset` columns,
        ignoring ``NaN`` values.
    :py:meth:`.rt_fastarray.FastArray.nanmean` :
        Computes the mean of :py:class:`~.rt_fastarray.FastArray` values, ignoring ``NaN`` values.
    :py:meth:`.rt_groupbyops.GroupByOps.nanmean` :
        Computes the mean of each group, ignoring ``NaN`` values. Used by
        :py:class:`~.rt_categorical.Categorical` objects.

    Notes
    -----
    The ``dtype`` parameter specifies the data type of the result. This differs from
    :py:func:`numpy.nanmean`, where it specifies the data type used to compute the mean.

    Examples
    --------
    >>> a = rt.FastArray([1, 3, 5, rt.nan])
    >>> rt.nanmean(a)
    3.0

    With a ``dtype`` specified:

    >>> a = rt.FastArray([1, 3, 5, rt.nan])
    >>> rt.nanmean(a, dtype = rt.int32)
    3

    With a ``filter``:

    >>> a = rt.FastArray([1, 3, 5, rt.nan])
    >>> b = rt.FastArray([False, True, True, True])
    >>> rt.nanmean(a, filter = b)
    4.0
    """
    kwargs = _np_keyword_wrapper(filter=filter, dtype=dtype, **kwargs)
    args = _convert_cat_args(args)
    if hasattr(args[0], "nanmean"):
        return args[0].nanmean(**kwargs)
    return np.nanmean(*args, **kwargs)


# -------------------------------------------------------
def median(*args, **kwargs) -> np.number:
    args = _convert_cat_args(args)
    if isinstance(args[0], np.ndarray):
        return np.median(*args, **kwargs)
    return builtins.median(*args, **kwargs)


# -------------------------------------------------------
def nanmedian(*args, **kwargs) -> np.number:
    args = _convert_cat_args(args)
    if isinstance(args[0], np.ndarray):
        return np.nanmedian(*args, **kwargs)
    return np.nanmedian(*args, **kwargs)


# -------------------------------------------------------
def var(*args, filter=None, dtype=None, **kwargs) -> np.number | Dataset:
    """
    Compute the variance of the values in the first argument.

    Riptable uses the convention that ``ddof = 1``, meaning the variance of
    ``[x_1, ..., x_n]`` is defined by ``var = 1/(n - 1) * sum(x_i - mean )**2`` (note
    the ``n - 1`` instead of ``n``). This differs from NumPy, which uses ``ddof = 0`` by
    default.

    When possible, ``rt.var(x, *args)`` calls ``x.var(*args)``; look there for
    documentation. In particular, note whether the called function accepts the keyword
    arguments listed below.

    For example, :py:meth:`.rt_fastarray.FastArray.var` accepts the ``filter`` and
    ``dtype`` keyword arguments, but :py:meth:`.rt_dataset.Dataset.var` does not.

    Parameters
    ----------
    *args : array or iterable or scalar value
        Contains the values that are used to calculate the variance.
    filter : array of bool, default `None`
        Specifies which elements to include in the variance calculation. If the ``filter``
        is uniformly `False`, the method returns a :py:class:`ZeroDivisionError`.
    dtype : :py:class:`numpy.dtype` or Riptable dtype, default :py:obj:`numpy.float64`
        The data type of the result. For a :py:class:`~.rt_fastarray.FastArray` ``x``,
        ``x.var(dtype = my_type)`` is equivalent to ``my_type(x.var())``.
    **kwargs :
        Additional keyword arguments to be passed to the function. See
        :py:func:`numpy.var` for additional keyword arguments.

    Returns
    -------
    scalar or :py:class:`~.rt_dataset.Dataset`
        Scalar for :py:class:`~.rt_fastarray.FastArray` input. For
        :py:class:`~.rt_dataset.Dataset` input, returns a :py:class:`~.rt_dataset.Dataset`
        consisting of a row with each numerical column's variance.

    See Also
    --------
    :py:func:`numpy.var` :
        Compute the variance along the specified axis.
    :py:func:`.rt_numpy.nanvar` :
        Computes the variance, ignoring ``NaN`` values.
    :py:meth:`.rt_fastarray.FastArray.var` :
        Computes the variance of :py:class:`~.rt_fastarray.FastArray` values.
    :py:meth:`.rt_dataset.Dataset.var` :
        Computes the variance of numerical :py:class:`~.rt_dataset.Dataset` columns.
    :py:meth:`.rt_groupbyops.GroupByOps.var` :
        Computes the variance of each group. Used by
        :py:class:`~.rt_categorical.Categorical` objects.

    Notes
    -----
    The ``dtype`` parameter specifies the data type of the result. This differs
    from :py:func:`numpy.var`, where it specifies the data type used to compute the
    variance.

    Examples
    --------
    >>> a = rt.FastArray([1, 2, 3])
    >>> rt.var(a)
    1.0

    With a ``dtype`` specified:

    >>> a = rt.FastArray([1, 2, 3])
    >>> rt.var(a, dtype = rt.int32)
    1

    With a ``filter``:

    >>> a = rt.FastArray([1, 2, 3])
    >>> b = rt.FastArray([False, True, True])
    >>> rt.var(a, filter = b)
    0.5
    """
    kwargs = _np_keyword_wrapper(filter=filter, dtype=dtype, **kwargs)
    args = _convert_cat_args(args)
    if hasattr(args[0], "var"):
        return args[0].var(**kwargs)
    return builtins.var(*args, **kwargs)


# -------------------------------------------------------
def nanvar(*args, filter=None, dtype=None, **kwargs) -> np.number | Dataset:
    """
    Compute the variance of the values in the first argument, ignoring ``NaN`` values.

    If all values in the first argument are ``NaN`` values, ``NaN`` is returned.

    Riptable uses the convention that ``ddof = 1``, meaning the variance of
    ``[x_1, ..., x_n]`` is defined by ``var = 1/(n - 1) * sum(x_i - mean )**2`` (note
    the ``n - 1`` instead of ``n``). This differs from NumPy, which uses ``ddof = 0`` by
    default.

    When possible, ``rt.nanvar(x, *args)`` calls ``x.nanvar(*args)``; look there for
    documentation. In particular, note whether the called function accepts the keyword
    arguments listed below.

    For example, :py:meth:`.rt_fastarray.FastArray.nanvar` accepts the ``filter`` and
    ``dtype`` keyword arguments, but :py:meth:`.rt_dataset.Dataset.nanvar` does not.

    Parameters
    ----------
    *args : array or iterable or scalar value
        Contains the values that are used to calculate the variance.
    filter : array of bool, default `None`
        Specifies which elements to include in the variance calculation. If the filter
        is uniformly `False`, the method returns a :py:class:`ZeroDivisionError`.
    dtype : :py:class:`numpy.dtype` or Riptable dtype, default :py:obj:`numpy.float64`
        The data type of the result. For a :py:class:`~.rt_fastarray.FastArray` ``x``,
        ``x.nanvar(dtype = my_type)`` is equivalent to ``my_type(x.nanvar())``.
    **kwargs :
        Additional keyword arguments to be passed to the function. See
        :py:func:`numpy.nanvar` for additional keyword arguments.

    Returns
    -------
    scalar or :py:class:`~.rt_dataset.Dataset`
        Scalar for :py:class:`~.rt_fastarray.FastArray` input. For
        :py:class:`~.rt_dataset.Dataset` input, returns a :py:class:`~.rt_dataset.Dataset`
        consisting of a row with each numerical column's variance.

    See Also
    --------
    :py:func:`numpy.nanvar` :
        Compute the variance along the specified axis, while ignoring ``NaN`` values.
    :py:func:`.rt_numpy.var` :
        Computes the variance.
    :py:meth:`.rt_fastarray.FastArray.nanvar` :
        Computes the variance of :py:class:`~.rt_fastarray.FastArray` values, ignoring
        ``NaN`` values.
    :py:meth:`.rt_dataset.Dataset.nanvar` :
        Computes the variance of numerical :py:class:`~.rt_dataset.Dataset` columns,
        ignoring ``NaN`` values.
    :py:meth:`.rt_groupbyops.GroupByOps.nanvar` :
        Computes the variance of each group, ignoring ``NaN`` values. Used by
        :py:class:`~.rt_categorical.Categorical` objects.

    Notes
    -----
    The ``dtype`` parameter specifies the data type of the result. This differs from
    :py:func:`numpy.nanvar`, where it specifies the data type used to compute the
    variance.

    Examples
    --------
    >>> a = rt.FastArray([1, 2, 3, rt.nan])
    >>> rt.nanvar(a)
    1.0

    With a ``dtype`` specified:

    >>> a = rt.FastArray([1, 2, 3, rt.nan])
    >>> rt.nanvar(a, dtype = rt.int32)
    1

    With a ``filter``:

    >>> a = rt.FastArray([1, 2, 3, rt.nan])
    >>> b = rt.FastArray([False, True, True, True])
    >>> rt.nanvar(a, filter = b)
    0.5
    """
    kwargs = _np_keyword_wrapper(filter=filter, dtype=dtype, **kwargs)
    args = _convert_cat_args(args)
    if hasattr(args[0], "nanvar"):
        return args[0].nanvar(**kwargs)
    return np.nanvar(*args, **kwargs)


# -------------------------------------------------------
def std(*args, filter=None, dtype=None, **kwargs):
    """
    Compute the standard deviation of the values in the first argument.

    Riptable uses the convention that ``ddof = 1``, meaning the standard deviation of
    ``[x_1, ..., x_n]`` is defined by ``std = 1/(n - 1) * sum(x_i - mean )**2`` (note
    the ``n - 1`` instead of ``n``). This differs from NumPy, which uses ``ddof = 0`` by
    default.

    When possible, ``rt.std(x, *args)`` calls ``x.std(*args)``; look there for
    documentation. In particular, note whether the called function accepts the keyword
    arguments listed below.

    For example, :py:meth:`.rt_fastarray.FastArray.std` accepts the ``filter`` and
    ``dtype`` keyword arguments, but :py:meth:`.rt_dataset.Dataset.std` does not.

    Parameters
    ----------
    *args : array or iterable or scalar value
        Contains the values that are used to calculate the standard deviation.
    filter : array of bool, default `None`
        Specifies which elements to include in the standard deviation calculation. If
        the filter is uniformly `False`, the method returns a :py:class:`ZeroDivisionError`.
    dtype : :py:class:`numpy.dtype` or Riptable dtype, default :py:obj:`numpy.float64`
        The data type of the result. For a :py:class:`~.rt_fastarray.FastArray` ``x``,
        ``x.std(dtype = my_type)`` is equivalent to ``my_type(x.std())``.
    **kwargs :
        Additional keyword arguments to be passed to the function. See
        :py:func:`numpy.std` for additional keyword arguments.

    Returns
    -------
    scalar or :py:class:`~.rt_dataset.Dataset`
        Scalar for :py:class:`~.rt_fastarray.FastArray` input. For
        :py:class:`~.rt_dataset.Dataset` input, returns a :py:class:`~.rt_dataset.Dataset`
        consisting of a row with each numerical column's standard deviation.

    See Also
    --------
    :py:func:`numpy.std` :
        Compute the standard deviation along the specified axis.
    :py:func:`.rt_numpy.nanstd` :
        Computes the standard deviation, ignoring ``NaN`` values.
    :py:meth:`.rt_fastarray.FastArray.std` :
        Computes the standard deviation of :py:class:`~.rt_fastarray.FastArray` values.
    :py:meth:`.rt_dataset.Dataset.std` :
        Computes the standard deviation of numerical :py:class:`~.rt_dataset.Dataset`
        columns.
    :py:meth:`.rt_groupbyops.GroupByOps.std` :
        Computes the standard deviation of each group. Used by
        :py:class:`~.rt_categorical.Categorical` objects.

    Notes
    -----
    The ``dtype`` parameter specifies the data type of the result. This differs
    from :py:func:`numpy.std`, where it specifies the data type used to compute the
    standard deviation.

    Examples
    --------
    >>> a = rt.FastArray([1, 2, 3])
    >>> rt.std(a)
    1.0

    With a ``dtype`` specified:

    >>> a = rt.FastArray([1, 2, 3])
    >>> rt.std(a, dtype = rt.int32)
    1

    With a ``filter``:

    >>> a = rt.FastArray([1, 2, 3])
    >>> b = rt.FA([False, True, True])
    >>> rt.std(a, filter = b)
    0.7071067811865476
    """
    kwargs = _np_keyword_wrapper(filter=filter, dtype=dtype, **kwargs)
    args = _convert_cat_args(args)
    if hasattr(args[0], "std"):
        return args[0].std(**kwargs)
    return builtins.var(*args, **kwargs)


# -------------------------------------------------------
def nanstd(*args, filter=None, dtype=None, **kwargs) -> np.number | Dataset:
    """
    Compute the standard deviation of the values in the first argument, ignoring ``NaN``
    values.

    If all values in the first argument are ``NaN`` values, ``NaN`` is returned.

    Riptable uses the convention that ``ddof = 1``, meaning the standard deviation of
    ``[x_1, ..., x_n]`` is defined by ``std = 1/(n - 1) * sum(x_i - mean )**2`` (note
    the ``n - 1`` instead of ``n``). This differs from NumPy, which uses ``ddof = 0`` by
    default.

    When possible, ``rt.nanstd(x, *args)`` calls ``x.nanstd(*args)``; look there for
    documentation. In particular, note whether the called function accepts the keyword
    arguments listed below.

    For example, :py:meth:`.rt_fastarray.FastArray.nanstd` accepts the ``filter`` and
    ``dtype`` keyword arguments, but :py:meth:`.rt_dataset.Dataset.nanstd` does not.

    Parameters
    ----------
    *args : array or iterable or scalar value
        Contains the values that are used to calculate the standard deviation.
    filter : array of bool, default `None`
        Specifies which elements to include in the standard deviation calculation. If
        the filter is uniformly `False`, the method returns a :py:class:`ZeroDivisionError`.
    dtype : :py:class:`numpy.dtype` or Riptable dtype, default :py:obj:`numpy.float64`
        The data type of the result. For a :py:class:`~.rt_fastarray.FastArray` ``x``,
        ``x.nanstd(dtype = my_type)`` is equivalent to ``my_type(x.nanstd())``.
    **kwargs :
        Additional keyword arguments to be passed to the function. See
        :py:func:`numpy.nanstd` for additional keyword arguments.

    Returns
    -------
    scalar or :py:class:`~.rt_dataset.Dataset`
        Scalar for :py:class:`~.rt_fastarray.FastArray` input. For
        :py:class:`~.rt_dataset.Dataset` input, returns a :py:class:`~.rt_dataset.Dataset`
        consisting of a row with each numerical column's standard deviation.

    See Also
    --------
    :py:func:`numpy.nanstd` :
        Compute the standard deviation along the specified axis, while ignoring ``NaN``
        values.
    :py:func:`.rt_numpy.std` :
        Computes the standard deviation.
    :py:meth:`.rt_fastarray.FastArray.nanstd` :
        Computes the standard deviation of :py:class:`~.rt_fastarray.FastArray` values,
        ignoring ``NaN`` values.
    :py:meth:`.rt_dataset.Dataset.nanstd` :
        Computes the standard deviation of numerical :py:class:`~.rt_dataset.Dataset`
        columns, ignoring ``NaN`` values.
    :py:meth:`.rt_groupbyops.GroupByOps.nanstd` :
        Computes the standard deviation of each group, ignoring ``NaN`` values. Used by
        :py:class:`~.rt_categorical.Categorical` objects.

    Notes
    -----
    The ``dtype`` parameter specifies the data type of the result. This differs from
    :py:func:`numpy.nanstd`, where it specifies the data type used to compute
    the standard deviation.

    Examples
    --------
    >>> a = rt.FastArray([1, 2, 3, rt.nan])
    >>> rt.nanstd(a)
    1.0

    With a ``dtype`` specified:

    >>> a = rt.FastArray([1, 2, 3, rt.nan])
    >>> rt.nanstd(a, dtype = rt.int32)
    1

    With ``filter``:

    >>> a = rt.FastArray([1, 2, 3, rt.nan])
    >>> b = rt.FastArray([False, True, True, True])
    >>> rt.nanstd(a, filter = b)
    0.7071067811865476
    """
    kwargs = _np_keyword_wrapper(filter=filter, dtype=dtype, **kwargs)
    args = _convert_cat_args(args)
    if hasattr(args[0], "nanstd"):
        return args[0].nanstd(**kwargs)
    return np.nanstd(*args, **kwargs)


QUANTILE_METHOD_NP_KW = "method"


def gb_np_quantile(a, q, is_nan_function):
    """
    Applies a correct numpy function for aggregation, used in accum2.
    Only uses "midpoint" interpolation method, because this is the one
    used in the quantile function on cpp level.
    Handles undesired behaviour of np.quantile when infs are present in a

    Parameters
    ----------
    a : rt.FastArray
        Data to compute quantile for
    q : float, must be between 0. and 1.
        Quantile to compute
    is_nan_function : bool
        flag indicating if apply nan- or non-nan- verison of a function
    Returns
    -------
    Statistic (quantile) computed with a corresponding numpy function
    """

    # np.min/max are overwritten for FA, so convert to np.arrays

    if q == 0.5:
        if is_nan_function:
            return np.nanmedian(a._np)
        else:
            return np.median(a._np)
    elif q == 0.0:
        if is_nan_function:
            return np.nanmin(a._np)
        else:
            return np.min(a._np)
    elif q == 1.0:
        if is_nan_function:
            return np.nanmax(a._np)
        else:
            return np.max(a._np)
    else:
        with_infs = np.isinf(a).any()
        if not with_infs:
            kwargs = {QUANTILE_METHOD_NP_KW: "midpoint"}
            if is_nan_function:
                return np.nanquantile(a._np, q, **kwargs)
            else:
                return np.quantile(a._np, q, **kwargs)
        else:
            return gb_np_quantile_infs(a._np, q, is_nan_function)


def gb_np_quantile_infs(a, q, is_nan_function, **kwargs):
    """
    Function for handling +/-infs in np.(nan)quantile

    np.quantile doesn't give desired results when infs are present,
    due to abmiguities with arithmetic operations with infs. See for instance:
    https://github.com/numpy/numpy/issues/21932
    https://github.com/numpy/numpy/issues/21091
    Example:
    np.quantile([rt.inf, rt.inf], q=0.5, method="midpoint") returns np.nan,
    while
    np.median([rt.inf, rt.inf]) returns rt.inf,
    although arguably these should give the same result. The behaviour of
    np.median is also more expected.
    the following will always give the same result as np.median(a):
    (np.quantile(a, q=0.5, method="lower") + np.quantile(a, q=0.5, method="higher")) / 2
    It is also clear that this essentially is the same as method="midpoint".

    Parameters
    ----------
    a : array-like
        Data to compute quantile for, which might contains +/-inf values
    q : float, must be between 0. and 1.
        Quantile to compute
    is_nan_function : bool
        flag indicating if apply nan- or non-nan- verison of a function
    Returns
    -------
    Statistic (quantile) computed with a corresponding numpy function while
    handling +/-infs in a mroe expected way.

    """
    if is_nan_function:
        np_function = np.nanquantile
    else:
        np_function = np.quantile

    kwargs[QUANTILE_METHOD_NP_KW] = "lower"
    lower_quantile = np_function(a, q, **kwargs)
    kwargs[QUANTILE_METHOD_NP_KW] = "higher"
    higher_quantile = np_function(a, q, **kwargs)

    midpoint_quantile = (lower_quantile + higher_quantile) / 2
    return midpoint_quantile


def np_rolling_nanquantile(a, q, window):
    def strided_array(a, window):
        nrows = a.size - window + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows, window), strides=(n, n))

    return gb_np_quantile_infs(a=strided_array(a, window), q=q, is_nan_function=True, axis=-1)


# -------------------------------------------------------
def percentile(*args, **kwargs) -> np.number:
    args = _convert_cat_args(args)
    if isinstance(args[0], np.ndarray):
        return np.percentile(*args, **kwargs)
    return np.percentile(*args, **kwargs)


# -------------------------------------------------------
def nanpercentile(*args, **kwargs) -> np.number:
    args = _convert_cat_args(args)
    if isinstance(args[0], np.ndarray):
        return np.nanpercentile(*args, **kwargs)
    return np.nanpercentile(*args, **kwargs)


# -------------------------------------------------------
def bincount(*args, **kwargs) -> int:
    args = _convert_cat_args(args)
    if isinstance(args[0], np.ndarray):
        return np.bincount(*args, **kwargs)
    return np.bincount(*args, **kwargs)


# -------------------------------------------------------
def isnan(*args, **kwargs) -> FastArray | bool:
    """
    Return `True` for each element that's a ``NaN`` (Not a Number), `False` otherwise.

    Parameters
    ----------
    *args :
        See :py:obj:`numpy.isnan`.
    **kwargs :
        See :py:obj:`numpy.isnan`.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray` or bool
        For array input, a :py:class:`~.rt_fastarray.FastArray` of booleans is returned
        that's `True` for each element that's a ``NaN``, `False` otherwise. For scalar
        input, a boolean is returned.

    See Also
    --------
    :py:func:`.rt_numpy.isnotnan`
    :py:func:`.rt_numpy.isnanorzero`
    :py:meth:`.rt_fastarray.FastArray.isnan`
    :py:meth:`.rt_fastarray.FastArray.isnotnan`
    :py:meth:`.rt_fastarray.FastArray.notna`
    :py:meth:`.rt_fastarray.FastArray.isnanorzero`
    :py:meth:`.rt_categorical.Categorical.isnan`
    :py:meth:`.rt_categorical.Categorical.isnotnan`
    :py:meth:`.rt_categorical.Categorical.notna`
    :py:meth:`.rt_datetime.Date.isnan`
    :py:meth:`.rt_datetime.Date.isnotnan`
    :py:meth:`.rt_datetime.DateTimeNano.isnan`
    :py:meth:`.rt_datetime.DateTimeNano.isnotnan`
    :py:meth:`.rt_dataset.Dataset.mask_or_isnan` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that contains at least one ``NaN``.
    :py:meth:`.rt_dataset.Dataset.mask_and_isnan` :
        Return a boolean array that's `True` for each row that contains only ``NaN``
        values.

    Examples
    --------
    >>> a = rt.FastArray([rt.nan, rt.inf, 2])
    >>> rt.isnan(a)
    FastArray([ True, False, False])

    >>> rt.isnan(0)
    False
    """
    try:
        return args[0].isnan(**kwargs)
    except:
        return _unary_func(np.isnan, *args, **kwargs)


# -------------------------------------------------------
def isnotnan(*args, **kwargs) -> FastArray | bool:
    """
    Return `True` for each element that's not a ``NaN`` (Not a Number), `False` otherwise.

    Parameters
    ----------
    *args :
        See :py:obj:`numpy.isnan`.
    **kwargs :
        See :py:obj:`numpy.isnan`.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray` or bool
        For array input, a :py:class:`~.rt_fastarray.FastArray` of booleans is returned
        that's `True` for each element that's not a ``NaN``, `False` otherwise. For scalar
        input, a boolean is returned.

    See Also
    --------
    :py:func:`.rt_numpy.isnan`
    :py:func:`.rt_numpy.isnanorzero`
    :py:meth:`.rt_fastarray.FastArray.isnan`
    :py:meth:`.rt_fastarray.FastArray.isnotnan`
    :py:meth:`.rt_fastarray.FastArray.notna`
    :py:meth:`.rt_fastarray.FastArray.isnanorzero`
    :py:meth:`.rt_categorical.Categorical.isnan`
    :py:meth:`.rt_categorical.Categorical.isnotnan`
    :py:meth:`.rt_categorical.Categorical.notna`
    :py:meth:`.rt_datetime.Date.isnan`
    :py:meth:`.rt_datetime.Date.isnotnan`
    :py:meth:`.rt_datetime.DateTimeNano.isnan`
    :py:meth:`.rt_datetime.DateTimeNano.isnotnan`
    :py:meth:`.rt_dataset.Dataset.mask_or_isnan` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that contains at least one ``NaN``.
    :py:meth:`.rt_dataset.Dataset.mask_and_isnan` :
        Return a boolean array that's `True` for each row that contains only ``NaN``
        values.

    Examples
    --------
    >>> a = rt.FastArray([rt.nan, rt.inf, 2])
    >>> rt.isnotnan(a)
    FastArray([False,  True,  True])

    >>> rt.isnotnan(0)
    True
    """
    try:
        return args[0].isnotnan(**kwargs)
    except:
        return ~np.isnan(*args, **kwargs)


# -------------------------------------------------------
def isnanorzero(*args, **kwargs) -> FastArray | bool:
    """
    Return `True` for each element that's a ``NaN`` (Not a Number) or zero, `False`
    otherwise.

    Parameters
    ----------
    *args :
        See :py:obj:`numpy.isnan`.
    **kwargs :
        See :py:obj:`numpy.isnan`.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray` or bool
        For array input, a :py:class:`~.rt_fastarray.FastArray` of booleans is returned
        that's `True` for each element that's a ``NaN`` or zero, `False` otherwise. For
        scalar input, a boolean is returned.

    See Also
    --------
    :py:func:`.rt_numpy.isnan`
    :py:func:`.rt_numpy.isnotnan`
    :py:meth:`.rt_fastarray.FastArray.isnan`
    :py:meth:`.rt_fastarray.FastArray.isnotnan`
    :py:meth:`.rt_fastarray.FastArray.isnanorzero`
    :py:meth:`.rt_categorical.Categorical.isnan`
    :py:meth:`.rt_categorical.Categorical.isnotnan`
    :py:meth:`.rt_datetime.Date.isnan`
    :py:meth:`.rt_datetime.Date.isnotnan`
    :py:meth:`.rt_datetime.DateTimeNano.isnan`
    :py:meth:`.rt_datetime.DateTimeNano.isnotnan`
    :py:meth:`.rt_dataset.Dataset.mask_or_isnan` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that contains at least one ``NaN``.
    :py:meth:`.rt_dataset.Dataset.mask_and_isnan` :
        Return a boolean array that's `True` for each row that contains only ``NaN``
        values.

    Examples
    --------
    >>> a = rt.FastArray([0, rt.nan, rt.inf, 3])
    >>> rt.isnanorzero(a)
    FastArray([ True,  True, False, False])

    >>> rt.isnanorzero(0)
    True
    """
    try:
        return args[0].isnanorzero(**kwargs)
    except:
        # slow way
        result = np.isnan(*args, **kwargs)
        result += args[0] == 0
        return result


# -------------------------------------------------------
def isfinite(*args, **kwargs) -> FastArray | bool:
    """
    Return `True` for each finite element, `False` otherwise.

    A value is considered to be finite if it's not positive or negative infinity
    or a ``NaN`` (Not a Number).

    Parameters
    ----------
    *args :
        See :py:obj:`numpy.isfinite`.
    **kwargs :
        See :py:obj:`numpy.isfinite`.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray` or bool
        For array input, a :py:class:`~.rt_fastarray.FastArray` of booleans is returned
        that's `True` for each element that's finite, `False` otherwise. For scalar
        input, a boolean is returned.

    See Also
    --------
    :py:func:`.rt_numpy.isnotfinite`
    :py:func:`.rt_numpy.isinf`
    :py:func:`.rt_numpy.isnotinf`
    :py:meth:`.rt_fastarray.FastArray.isfinite`
    :py:meth:`.rt_fastarray.FastArray.isnotfinite`
    :py:meth:`.rt_fastarray.FastArray.isinf`
    :py:meth:`.rt_fastarray.FastArray.isnotinf`
    :py:meth:`.rt_dataset.Dataset.mask_or_isfinite` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that has at least one finite value.
    :py:meth:`.rt_dataset.Dataset.mask_and_isfinite` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that contains all finite values.
    :py:meth:`.rt_dataset.Dataset.mask_or_isinf` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that has at least one value that's positive or negative infinity.
    :py:meth:`.rt_dataset.Dataset.mask_and_isinf` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that contains all infinite values.

    Examples
    --------
    >>> a = rt.FastArray([rt.inf, -rt.inf, rt.nan, 0])
    >>> rt.isfinite(a)
    FastArray([False, False, False, True])

    >>> rt.isfinite(1)
    True
    """
    try:
        return args[0].isfinite(**kwargs)
    except:
        return _unary_func(np.isfinite, *args, **kwargs)


# -------------------------------------------------------
def isnotfinite(*args, **kwargs) -> FastArray | bool:
    """
    Return `True` for each non-finite element, `False` otherwise.

    A value is considered to be finite if it's not positive or negative infinity
    or a ``NaN`` (Not a Number).

    Parameters
    ----------
    *args :
        See :py:obj:`numpy.isfinite`.
    **kwargs :
        See :py:obj:`numpy.isfinite`.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray` or bool
        For array input, a :py:class:`~.rt_fastarray.FastArray` of booleans is returned
        that's `True` for each non-finite element, `False` otherwise. For scalar input,
        a boolean is returned.

    See Also
    --------
    :py:func:`.rt_numpy.isfinite`
    :py:func:`.rt_numpy.isinf`
    :py:func:`.rt_numpy.isnotinf`
    :py:meth:`.rt_fastarray.FastArray.isfinite`
    :py:meth:`.rt_fastarray.FastArray.isnotfinite`
    :py:meth:`.rt_fastarray.FastArray.isinf`
    :py:meth:`.rt_fastarray.FastArray.isnotinf`
    :py:meth:`.rt_dataset.Dataset.mask_or_isfinite` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that has at least one finite value.
    :py:meth:`.rt_dataset.Dataset.mask_and_isfinite` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that contains all finite values.
    :py:meth:`.rt_dataset.Dataset.mask_or_isinf` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that has at least one value that's positive or negative infinity.
    :py:meth:`.rt_dataset.Dataset.mask_and_isinf` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that contains all infinite values.

    Examples
    --------
    >>> a = rt.FastArray([rt.inf, -rt.inf, rt.nan, 0])
    >>> rt.isnotfinite(a)
    FastArray([ True,  True,  True, False])

    >>> rt.isnotfinite(1)
    False
    """
    try:
        return args[0].isnotfinite(**kwargs)
    except:
        return ~np.isfinite(*args, **kwargs)


# -------------------------------------------------------
def isinf(*args, **kwargs) -> FastArray | bool:
    """
    Return `True` for each element that's positive or negative infinity, `False`
    otherwise.

    Parameters
    ----------
    *args :
        See :py:obj:`numpy.isinf`.
    **kwargs :
        See :py:obj:`numpy.isinf`.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray` or bool
        For array input, a :py:class:`~.rt_fastarray.FastArray` of booleans is returned
        that's `True` for each element that's positive or negative infinity, `False`
        otherwise. For scalar input, a boolean is returned.

    See Also
    --------
    :py:func:`.rt_numpy.isnotinf`
    :py:func:`.rt_numpy.isfinite`
    :py:func:`.rt_numpy.isnotfinite`
    :py:meth:`.rt_fastarray.FastArray.isinf`
    :py:meth:`.rt_fastarray.FastArray.isnotinf`
    :py:meth:`.rt_fastarray.FastArray.isfinite`
    :py:meth:`.rt_fastarray.FastArray.isnotfinite`
    :py:meth:`.rt_dataset.Dataset.mask_or_isfinite` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that has at least one finite value.
    :py:meth:`.rt_dataset.Dataset.mask_and_isfinite` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that contains all finite values.
    :py:meth:`.rt_dataset.Dataset.mask_or_isinf` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that has at least one value that's positive or negative infinity.
    :py:meth:`.rt_dataset.Dataset.mask_and_isinf` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that contains all infinite values.

    Examples
    --------
    >>> a = rt.FastArray([rt.inf, -rt.inf, rt.nan, 0])
    >>> rt.isinf(a)
    FastArray([ True,  True, False, False])

    >>> rt.isinf(1)
    False
    """
    try:
        return args[0].isinf(**kwargs)
    except:
        return _unary_func(np.isinf, *args, **kwargs)


# -------------------------------------------------------
def isnotinf(*args, **kwargs) -> FastArray | bool:
    """
    Return `True` for each element that's not positive or negative infinity,
    `False` otherwise.

    Parameters
    ----------
    *args :
        See :py:obj:`numpy.isinf`.
    **kwargs :
        See :py:obj:`numpy.isinf`.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray` or bool
        For array input, a :py:class:`~.rt_fastarray.FastArray` of booleans is returned
        that's `True` for each element that's not positive or negative infinity, `False`
        otherwise. For scalar input, a boolean is returned.

    See Also
    --------
    :py:func:`.rt_numpy.isinf`
    :py:func:`.rt_numpy.isfinite`
    :py:func:`.rt_numpy.isnotfinite`
    :py:meth:`.rt_fastarray.FastArray.isnotinf`
    :py:meth:`.rt_fastarray.FastArray.isinf`
    :py:meth:`.rt_fastarray.FastArray.isfinite`
    :py:meth:`.rt_fastarray.FastArray.isnotfinite`
    :py:meth:`.rt_dataset.Dataset.mask_or_isfinite` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that has at least one finite value.
    :py:meth:`.rt_dataset.Dataset.mask_and_isfinite` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that contains all finite values.
    :py:meth:`.rt_dataset.Dataset.mask_or_isinf` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that has at least one value that's positive or negative infinity.
    :py:meth:`.rt_dataset.Dataset.mask_and_isinf` :
        Return a boolean array that's `True` for each :py:class:`~.rt_dataset.Dataset`
        row that contains all infinite values.

    Examples
    --------
    >>> a = rt.FastArray([rt.inf, -rt.inf, rt.nan, 0])
    >>> rt.isnotinf(a)
    FastArray([False, False,  True,  True])

    >>> rt.isnotinf(1)
    True
    """
    try:
        return args[0].isnotinf(**kwargs)
    except:
        return ~np.isinf(*args, **kwargs)


# ------------------------------------------------------------
def putmask(a, mask, values) -> FastArray:
    """
    This is roughly the equivalent of arr[mask] = arr2[mask].

    Examples
    --------
    >>> arr = rt.FastArray([10, 10, 10, 10])
    >>> arr2 = rt.FastArray([1, 2, 3, 4])
    >>> mask = rt.FastArray([False, True, True, False])
    >>> rt.putmask(arr, mask, arr2)
    >>> arr
    FastArray([10,  2,  3, 10])

    It's important to note that the length of `arr` and `arr2` are presumed to be the same, otherwise
    the values in `arr2` are repeated until they have the same dimension.

    It should NOT be used to replace this operation:

    >>> arr = rt.FastArray([10, 10, 10, 10])
    >>> arr2 = rt.FastArray([1, 2])
    >>> mask = rt.FastArray([False, True, True, False])
    >>> arr[mask] = arr2
    >>> arr
    FastArray([10,  1,  2, 10])

    `arr2` is repeated to create ``rt.FastArray([1, 2, 1, 2])`` before performing the operation, hence the different result.

    >>> arr = rt.FastArray([10, 10, 10, 10])
    >>> arr2 = rt.FastArray([1, 2])
    >>> mask = rt.FastArray([False, True, True, False])
    >>> rt.putmask(arr, mask, arr2)
    >>> arr
    FastArray([10,  2,  1, 10])
    """
    try:
        # BUG what about categoricals, etc
        return a.putmask(mask, values)
    except:
        # false is returned on failure
        if not rc.PutMask(a, mask, values):
            # final attempt use numpy
            return np.putmask(a, mask, values)


# ------------------------------------------------------------
def vstack(arrlist, dtype=None, order="C"):
    """
    Parameters
    ----------
    arrlist: list of 1d numpy arrays of the same length
             these arrays are considered the columns
    order: defaults to 'C' for row major. 'F' will be column major.
    dtype: defaults to None.  Can specifiy the final dtype for order='F' only.

    WARNING: when order='F' riptable vstack will return a diffrent shape
    from np.vstack since it will try to keep the first dim the same length
    while keeping data contiguous.

    If order='F' is not passed, order='C' is assumed.
    If riptable fails, then normal np.vstack will be called.
    For large arrays, riptable can run in parallel while converting to the dtype on the fly.

    Returns
    -------
    a 2d array that is column major and can be insert into a dataset
    Use v[:,0]  then v[:,1] to access the columns instead of
    v[0] and v[1] which would be the method with np.vstack

    See Also
    --------
    np.vstack
    np.column_stack

    Examples
    --------
    >>> a = rt.arange(100)
    >>> b = rt.arange(100.0)
    >>> v = rt.vstack([a,b], order='F')
    >>> v.strides
    (8, 800)

    >>> v.flags
    C_CONTIGUOUS : False
    F_CONTIGUOUS : True

    >>> v.shape
    (100,2)
    """
    # check to make sure all numpy arrays
    # TODO: consider handling a dataset
    try:
        # make sure the columns are all the same length
        rowlength = {col.shape[0] for col in arrlist}
        if len(rowlength) == 1:
            numrows = rowlength.pop()
            numcols = len(arrlist)
            h = hstack(arrlist, dtype=dtype)
            if order == "F" or order == "f":
                return h.reshape((numrows, numcols), order="F")
            else:
                return h.reshape((numcols, numrows), order="C")
    except Exception:
        warnings.warn(f"vstack with order={order!r} failed, calling np.vstack", stacklevel=2)

    return np.vstack(arrlist)


# ------------------------------------------------------------
def repeat(a, repeats, axis=None):
    """
    Construct an array in which each element of a specified array is repeated
    consecutively a specified number of times.

    Parameters
    ----------
    a : array or scalar
        The input array or scalar. Each element is repeated consecutively
        ``repeats`` times. If no ``axis`` is specified, multi-dimensional arrays are
        flattened and a flattened array is returned.
    repeats : int or array of int
        The number of consecutive repetitions for each element of ``a``. If an
        ``axis`` is specified, the elements are repeated along that axis.
    axis : int, optional
        The axis along which to repeat the values. If no axis is specified, the
        input array is flattened and a flattened array is returned. For examples
        of repeats of multi-dimensional arrays, see :py:func:`numpy.repeat`. Note that
        although multi-dimensional arrays are technically supported by Riptable,
        you may get unexpected results when working with them.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray`
        A new :py:class:`~.rt_fastarray.FastArray` that has the same shape as ``a``,
        except along the given axis.

    See Also
    --------
    :py:func:`.rt_numpy.tile` : Construct an array by repeating a specified array.

    Examples
    --------
    Repeat a scalar:

    >>> rt.repeat(2, 5)
    FastArray([2, 2, 2, 2, 2])

    Repeat each element of an array:

    >>> x = rt.FastArray([1, 2, 3, 4])
    >>> rt.repeat(x, 2)
    FastArray([1, 1, 2, 2, 3, 3, 4, 4])

    Use an array for ``repeats``:

    >>> rt.repeat(x, [1, 2, 3, 4])
    FastArray([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    """
    # similar bug as tile, calls reshape which maintains class, but kills attributes
    if isinstance(a, TypeRegister.FastArray):
        result = np.repeat(a._np, repeats, axis=axis).view(TypeRegister.FastArray)
        return TypeRegister.newclassfrominstance(result, a)
    return np.repeat(a, repeats, axis=axis).view(TypeRegister.FastArray)


# ------------------------------------------------------------
def tile(arr, reps):
    """
    Construct an array by repeating an input array a specified number of times.

    Parameters
    ----------
    arr : array or scalar
        The input array or scalar.
    reps : int or array of int
        The number of repetitions of ``arr`` along each axis. For examples of
        :py:func:`~.rt_numpy.tile` used with multi-dimensional arrays, see
        :py:func:`numpy.tile`. Note that although multi-dimensional arrays are
        technically supported by Riptable, you may get unexpected results when working
        with them.

    Returns
    -------
    :py:class:`~.rt_fastarray.FastArray`
        A new :py:class:`~.rt_fastarray.FastArray` of the repeated input arrays.

    See Also
    --------
    :py:func:`.rt_numpy.repeat` :
        Construct an array by repeating each element of a specified array.

    Examples
    --------
    Tile a scalar:

    >>> rt.tile(2, 5)
    FastArray([2, 2, 2, 2, 2])

    Tile an array:

    >>> x = rt.FA([1, 2, 3, 4])
    >>> rt.tile(x, 2)
    FastArray([1, 2, 3, 4, 1, 2, 3, 4])
    """
    if isinstance(arr, TypeRegister.FastArray):
        # bug in tile, have to flip to normal numpy array first)
        result = np.tile(arr._np, reps).view(TypeRegister.FastArray)
        return TypeRegister.newclassfrominstance(result, arr)

    return np.tile(arr, reps).view(TypeRegister.FastArray)


# -------------------------------------------------------
# like in matlab, convert to int8
def logical(a):
    if isinstance(a, np.ndarray):
        if a.dtype == np.bool_:
            return a
        return a.astype(np.bool_)
    # TODO: Check for scalar here? Then we can be maybe use np.asanyarray(..., dtype=bool).view(TypeRegister.FastArray)
    #       to replace the use of the deprecated `np.bool`.
    return np.bool_(a).view(TypeRegister.FastArray)


##-------------------------------------------------------
# not allowed
# class bool(np.bool):
#    pass


# -------------------------------------------------------
class bool_(np.bool_):
    """
    The Riptable equivalent of `numpy.bool_`, with the concept of an invalid added.

    See Also
    --------
    numpy.bool_
    float32, float64, int8, uint8, int16, uint16, int32, uint32, int64, uint64, bytes_, str_

    Examples
    --------
    >>> rt.bool_.inv
    False
    """

    # Allow np.bool.inv  to work
    inv = INVALID_DICT[0]

    def __new__(cls, *args, **kwargs):
        if len(args) == 1:
            # check if converting an existing array
            if isinstance(args[0], np.ndarray):
                return TypeRegister.FastArray.astype(args[0], np.bool_, **kwargs)
        instance = np.bool_(*args, **kwargs)
        if np.isscalar(instance):
            return instance
        return instance.view(TypeRegister.FastArray)


# -------------------------------------------------------
class int8(np.int8):
    """
    The Riptable equivalent of `numpy.int8`, with the concept of an invalid added.

    See Also
    --------
    numpy.int8
    float32, float64, uint8, int16, uint16, int32, uint32, int64, uint64, bytes_, str_, bool_

    Examples
    --------
    >>> rt.int8.inv
    -128
    """

    # Allow np.int8.inv  to work
    inv = INVALID_DICT[1]

    def __new__(cls, *args, **kwargs):
        if len(args) == 1:
            # check if converting an existing array
            if isinstance(args[0], np.ndarray):
                return TypeRegister.FastArray.astype(args[0], np.int8, **kwargs)
        instance = np.int8(*args, **kwargs)
        if np.isscalar(instance):
            return instance
        return instance.view(TypeRegister.FastArray)


# -------------------------------------------------------
class uint8(np.uint8):
    """
    The Riptable equivalent of `numpy.uint8`, with the concept of an invalid added.

    See Also
    --------
    numpy.uint8
    float32, float64, int8, int16, uint16, int32, uint32, int64, uint64, bytes_, str_, bool_

    Examples
    --------
    >>> rt.uint8.inv
    255
    """

    # Allow np.uint8.inv  to work
    inv = INVALID_DICT[2]

    def __new__(cls, *args, **kwargs):
        if len(args) == 1:
            # check if converting an existing array
            if isinstance(args[0], np.ndarray):
                return TypeRegister.FastArray.astype(args[0], np.uint8, **kwargs)
        instance = np.uint8(*args, **kwargs)
        if np.isscalar(instance):
            return instance
        return instance.view(TypeRegister.FastArray)


# -------------------------------------------------------
class int16(np.int16):
    """
    The Riptable equivalent of `numpy.int16`, with the concept of an invalid added.

    See Also
    --------
    numpy.int16
    float32, float64, int8, uint8, uint16, int32, uint32, int64, uint64, bytes_, str_, bool_

    Examples
    --------
    >>> rt.int16.inv
    -32768
    """

    # Allow np.int16.inv  to work
    inv = INVALID_DICT[3]

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.int16, **kwargs)
        instance = np.int16(*args, **kwargs)
        if np.isscalar(instance):
            return instance
        return instance.view(TypeRegister.FastArray)


# -------------------------------------------------------
class uint16(np.uint16):
    """
    The Riptable equivalent of `numpy.uint16`, with the concept of an invalid added.

    See Also
    --------
    numpy.uint16
    float32, float64, int8, uint8, int16, int32, uint32, int64, uint64, bytes_, str_, bool_

    Examples
    --------
    >>> rt.uint16.inv
    65535
    """

    # Allow np.uint16.inv  to work
    inv = INVALID_DICT[4]

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.uint16, **kwargs)
        instance = np.uint16(*args, **kwargs)
        if np.isscalar(instance):
            return instance
        return instance.view(TypeRegister.FastArray)


# -------------------------------------------------------
class int32(np.int32):
    """
    The Riptable equivalent of `numpy.int32`, with the concept of an invalid added.

    See Also
    --------
    numpy.int32
    float32, float64, int8, uint8, int16, uint16, uint32, int64, uint64, bytes_, str_, bool_

    Examples
    --------
    >>> rt.int32.inv
    -2147483648
    """

    # Allow np.int32.inv  to work
    inv = INVALID_DICT[5]

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.int32, **kwargs)
        instance = np.int32(*args, **kwargs)
        if np.isscalar(instance):
            return instance
        return instance.view(TypeRegister.FastArray)


# -------------------------------------------------------
class uint32(np.uint32):
    """
    The Riptable equivalent of `numpy.uint32`, with the concept of an invalid added.

    See Also
    --------
    numpy.uint32
    float32, float64, int8, uint8, int16, uint16, int32, int64, uint64, bytes_, str_, bool_

    Examples
    --------
    >>> rt.uint32.inv
    4294967295
    """

    # Allow np.uint32.inv  to work
    inv = INVALID_DICT[6]

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.uint32, **kwargs)
        instance = np.uint32(*args, **kwargs)
        if np.isscalar(instance):
            return instance
        return instance.view(TypeRegister.FastArray)


# -------------------------------------------------------
class int64(np.int64):
    """
    The Riptable equivalent of `numpy.int64`, with the concept of an invalid added.

    See Also
    --------
    numpy.int64
    float32, float64, int8, uint8, int16, uint16, int32, uint32, uint64, bytes_, str_, bool_

    Examples
    --------
    >>> rt.int64.inv
    -9223372036854775808
    """

    # Allow np.int64.inv  to work
    inv = INVALID_DICT[9]

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.int64, **kwargs)
        instance = np.int64(*args, **kwargs)
        if np.isscalar(instance):
            return instance
        return instance.view(TypeRegister.FastArray)


# -------------------------------------------------------
class uint64(np.uint64):
    """
    The Riptable equivalent of `numpy.uint64`, with the concept of an invalid added.

    See Also
    --------
    numpy.uint64
    float32, float64, int8, uint8, int16, uint16, int32, uint32, int64, bytes_, str_, bool_

    Examples
    --------
    >>> rt.uint64.inv
    18446744073709551615
    """

    # Allow np.uint64.inv  to work
    inv = INVALID_DICT[10]

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.uint64, **kwargs)
        instance = np.uint64(*args, **kwargs)
        if np.isscalar(instance):
            return instance
        return instance.view(TypeRegister.FastArray)


# -------------------------------------------------------
class int0(int64):
    pass


# -------------------------------------------------------
class uint0(uint64):
    pass


# -------------------------------------------------------
class bytes_(np.bytes_):
    """
    The Riptable equivalent of `numpy.bytes_`, with the concept of an invalid added.

    See Also
    --------
    np.bytes_
    float32, float64, int8, uint8, int16, uint16, int32, uint32, int64, uint64, str_, bool_

    Examples
    --------
    >>> rt.bytes_.inv
    b''
    """

    # Allow np.bytes_.inv  to work
    inv = INVALID_DICT[18]

    def __new__(cls, arg0, *args, **kwargs):
        if np.isscalar(arg0):
            return np.bytes_(arg0, *args, **kwargs)
        return TypeRegister.FastArray(arg0, *args, dtype="S", **kwargs)


# -------------------------------------------------------
class str_(np.str_):
    """
    The Riptable equivalent of `numpy.str_`, with the concept of an invalid added.

    See Also
    --------
    numpy.str_
    float32, float64, int8, uint8, int16, uint16, int32, uint32, int64, uint64, bytes_, bool_

    Examples
    --------
    >>> rt.str_.inv
    ''
    """

    # Allow np.str_.inv  to work
    inv = INVALID_DICT[19]

    def __new__(cls, arg0, *args, **kwargs):
        if np.isscalar(arg0):
            return np.str_(arg0, *args, **kwargs)
        return TypeRegister.FastArray(arg0, *args, unicode=True, dtype="U", **kwargs)


# -------------------------------------------------------
# like in numpy, convert to a half
def half(a):
    if isinstance(a, np.ndarray):
        if a.dtype == np.float16:
            return a
        return a.astype(np.float16)
    return np.float16(a).view(TypeRegister.FastArray)


# -------------------------------------------------------
# like in matlab, convert to a single
def single(a):
    if isinstance(a, np.ndarray):
        if a.dtype == np.float32:
            return a
        return a.astype(np.float32)
    return np.float32(a).view(TypeRegister.FastArray)


# -------------------------------------------------------
# like in matlab, convert to a double
def double(a):
    if isinstance(a, np.ndarray):
        if a.dtype == np.float64:
            return a
        return a.astype(np.float64)
    return np.float64(a).view(TypeRegister.FastArray)


# -------------------------------------------------------
# like in numpy, convert to a longdouble
def longdouble(a):
    if isinstance(a, np.ndarray):
        if a.dtype == np.longdouble:
            return a
        return a.astype(np.longdouble)
    return np.longdouble(a).view(TypeRegister.FastArray)


# -------------------------------------------------------
class float32(np.float32):
    """
    The Riptable equivalent of `numpy.float32`, with the concept of an invalid added.

    See Also
    --------
    numpy.float32
    float64, int8, uint8, int16, uint16, int32, uint32, int64, uint64, bytes_, str_, bool_

    Examples
    --------
    >>> rt.float32.inv
    nan
    """

    # Allow np.float32.inv  to work
    inv = INVALID_DICT[11]

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.float32, **kwargs)
        instance = np.float32(*args, **kwargs)
        if np.isscalar(instance):
            return instance
        return instance.view(TypeRegister.FastArray)


# -------------------------------------------------------
class float64(np.float64):
    """
    The Riptable equivalent of `numpy.float64`, with the concept of an invalid added.

    See Also
    --------
    numpy.float64
    float32, int8, uint8, int16, uint16, int32, uint32, int64, uint64, bytes_, str_, bool_

    Examples
    --------
    >>> rt.float64.inv
    nan
    """

    # Allow np.float64.inv  to work
    inv = INVALID_DICT[12]

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.float64, **kwargs)
        instance = np.float64(*args, **kwargs)
        if np.isscalar(instance):
            return instance
        return instance.view(TypeRegister.FastArray)


# -------------------------------------------------------
# linux only
# class float128(np.float128):


# -------------------------------------------------------
def interp(x, xp, fp):
    """
    One-dimensional or two-dimensional linear interpolation with clipping.

    Returns the one-dimensional piecewise linear interpolant to a function
    with given discrete data points (`xp`, `fp`), evaluated at `x`.

    Parameters
    ----------
    x : array of float32 or float64
        The x-coordinates at which to evaluate the interpolated values.

    xp : 1-D or 2-D sequence of float32 or float64
        The x-coordinates of the data points, must be increasing if argument
        `period` is not specified. Otherwise, `xp` is internally sorted after
        normalizing the periodic boundaries with ``xp = xp % period``.

    fp : 1-D or 2-D sequence of float32 or float64
        The y-coordinates of the data points, same length as `xp`.

    Returns
    -------
    y : float32 or float64 (corresponding to fp) or ndarray
        The interpolated values, same shape as `x`.

    See Also
    --------
    np.interp
    rt.interp_extrap

    Notes
    -----
    riptable version does not handle kwargs left/right whereas np does
    riptable version handles floats or doubles, whereas np is always a double
    riptable will warn if first parameter is a float32, but xp or yp is a double
    """
    if not isinstance(x, np.ndarray):
        x = TypeRegister.FastArray(x)

    if not isinstance(xp, np.ndarray):
        xp = TypeRegister.FastArray(xp)

    if not isinstance(fp, np.ndarray):
        fp = TypeRegister.FastArray(fp)

    # check for float32
    if x.dtype.num == 11 and (xp.dtype.num != 11 or fp.dtype.num != 11):
        warnings.warn("rt.interp is downcasting to a float32 to match first array", stacklevel=2)
        xp = xp.astype(np.float32)
        fp = fp.astype(np.float32)
    return rc.InterpExtrap2d(x, xp, fp, 1)


# -------------------------------------------------------
@_args_to_fast_arrays("x", "xp", "fp")
def interp_extrap(x, xp, fp):
    """
    One-dimensional or two-dimensional linear interpolation without clipping.

    Returns the one-dimensional piecewise linear interpolant to a function
    with given discrete data points (`xp`, `fp`), evaluated at `x`.

    See Also
    --------
    np.interp
    rt.interp

    Notes
    -----
    * riptable version handles floats or doubles, wheras np is always a double
    * 2d mode is auto-detected based on `xp`/`fp`
    """
    return rc.InterpExtrap2d(x, xp, fp, 0)


# -------------------------------------------------------
def bitcount(a: Union[int, Sequence, np.array]) -> Union[int, np.array]:
    """
    Count the number of set (True) bits in an integer or in each integer within an array of
    integers. This operation is also known as population count or Hamming weight.

    Parameters
    ----------
    a : int or sequence or numpy.array
        A Python integer or a sequence of integers or a numpy integer array.

    Returns
    -------
    int or numpy.array
        If the input is Python int the return is int. If the input is sequence or numpy array the
        return is a numpy array with dtype int8.

    Examples
    --------
    >>> arr = rt.FastArray([741858, 77285, 916765, 395393, 347556, 896425, 921598, 86398])
    >>> rt.bitcount(arr)
    FastArray([10, 10, 14,  5,  9, 12, 14, 10], dtype=int8)
    """
    if not np.isscalar(a):
        # check if we can use the fast routine
        if not isinstance(a, np.ndarray):
            a = np.asanyarray(a)
        if np.issubdtype(a.dtype, np.integer) or a.dtype == bool:
            if not a.flags.c_contiguous:
                a = a.copy()
            return rc.BitCount(a)
        else:
            raise ValueError(f"Unsupported array dtype {a.dtype}")
    else:
        if isinstance(a, (int, np.integer)):
            return bin(a).count("1")
        else:
            raise ValueError(f"Unsupported input type {type(a)}")


# -------------------------------------------------------
def bool_to_fancy(arr: np.ndarray, both: bool = False) -> FastArray:
    """
    Parameters
    ----------
    arr : ndarray of bools
        A boolean array of True/False values
    both : bool
        Controls whether to return a
        the True and False elements in `arr`. Defaults to False.

    Returns
    -------
    fancy_index : ndarray of bools
        Fancy index array of where the True values are.
        If `both` is True, there are two fancy index array sections:
        The first array slice is where the True values are;
        The second array slice is where the False values are.
        The True count is returned.
    true_count : int, optional
        When `both` is True, this value is returned to indicate how many True
        values were in `arr`; this is then used to slice `fancy_index` into two
        slices indicating where the True and False values are, respectively, within `arr`.

    Notes
    -----
    runs in parallel

    Examples
    --------
    >>> np.random.seed(12345)
    >>> bools = np.random.randint(2, size=20, dtype=np.int8).astype(bool)
    >>> rt.bool_to_fancy(bools)
    FastArray([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 12, 15, 17, 18, 19])

    Setting the `both` parameter to True causes the function to return an array containing
    the indices of the True values in `arr` followed by the indices of the False values,
    along with the number (count) of True values. This count can be used to slice the returned
    array if you want just the True indices and False indices.

    >>> fancy_index, true_count = rt.bool_to_fancy(bools, both=True)
    >>> fancy_index[:true_count], fancy_index[true_count:]
    (FastArray([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 12, 15, 17, 18, 19]), FastArray([ 0, 11, 13, 14, 16]))
    """
    if isinstance(arr, np.ndarray):
        if arr.dtype.char == "?":
            return rc.BooleanToFancy(arr, both=both)
        else:
            raise TypeError(f"Input array must be boolean. Got {arr.dtype}")
    else:
        raise TypeError(f"Input must be ndarray. Got {type(arr)}")


# -------------------------------------------------------
def mask_or(*args, **kwargs):
    """pass in a tuple or list of boolean arrays to OR together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_OR, False)


# -------------------------------------------------------
def mask_and(*args, **kwargs):
    """pass in a tuple or list of boolean arrays to AND together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_AND, False)


# -------------------------------------------------------
def mask_xor(*args, **kwargs):
    """pass in a tuple or list of boolean arrays to XOR together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_XOR, False)


# -------------------------------------------------------
def mask_andnot(*args, **kwargs):
    """pass in a tuple or list of boolean arrays to ANDNOT together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_ANDNOT, False)


# -------------------------------------------------------
def mask_ori(*args, **kwargs):
    """inplace version: pass in a tuple or list of boolean arrays to OR together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_OR, True)


# -------------------------------------------------------
def mask_andi(*args, **kwargs):
    """inplace version: pass in a tuple or list of boolean arrays to AND together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_AND, True)


# -------------------------------------------------------
def mask_xori(*args, **kwargs):
    """inplace version: pass in a tuple or list of boolean arrays to XOR together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_XOR, True)


# -------------------------------------------------------
def mask_andnoti(*args, **kwargs):
    """inplace version: pass in a tuple or list of boolean arrays to ANDNOT together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_ANDNOT, True)


# -------------------------------------------------------
def _mask_op(bool_list, funcNum, inplace=False):
    # size check done by TypeRegister.FastArray cpp code
    # we do an all boolen check here for CPP code
    lenbool = len(bool_list)
    if lenbool == 1 and isinstance(bool_list[0], (list, tuple)):
        bool_list = bool_list[0]
        lenbool = len(bool_list)

    if lenbool == 0:
        raise ValueError(f"Nothing passed")

    # check if nothing to do because just one boolean array in list
    if lenbool == 1:
        return bool_list[0]

    # we could support all int types here as well
    dtype = 0
    for v in bool_list:
        # allow for scalar bool, as well as bool arrays.
        dtype += 0 if isinstance(v, bool) else v.dtype.num
    if dtype != 0:
        raise TypeError(f"Must all be boolean types")

    # we have at least two items
    # grabbing the func pointer speeds things up in testing
    ledgerFunc = TypeRegister.MathLedger._BASICMATH_TWO_INPUTS

    # Wrapper to detect not-supported result from rc and raise an error.
    def func(*args):
        result = ledgerFunc(*args)
        if result is None:
            # Operation not supported, so raise appropriate error.
            # Check all boolean arrays are same size
            size = len(bool_list[0])
            for i in range(lenbool):
                if len(bool_list[i]) != size:
                    raise ValueError("Boolean arrays must be the same length")
            raise ValueError(f"Cannot perform mask_op {funcNum}")
        return result

    if inplace:
        # assume first value can be reused
        result = bool_list[0]
        if not isinstance(result, np.ndarray):
            raise TypeError("First argument must be an array")
        func((result, bool_list[1], result), funcNum, 0)
        i = 2

        while i < lenbool:
            # this will do inplace
            func((result, bool_list[i], result), funcNum, 0)
            i += 1
    else:
        result = func((bool_list[0], bool_list[1]), funcNum, 0)
        i = 2

        while i < lenbool:
            func((result, bool_list[i], result), funcNum, 0)
            i += 1

    return result


# ------------------------------------------------------------
def hstack(tup: Sequence[np.ndarray], dtype: Optional[Union[str, type, np.dtype]] = None, **kwargs) -> np.ndarray:
    """
    see numpy hstack
    riptable can also take a dtype (it will convert all arrays to that dtype while stacking)
    riptable version will preserve sentinels
    riptable version is multithreaded
    for special classes like categorical and dataset, it will check to see if the
    class has it's own hstack and it will call that
    """
    # the riptable test suite causes this to segfault (in riptide_cpp 1.6.28), so it's commented out
    # for now. When that's fixed, uncomment this to allow riptable to handle more hstacking cases
    # instead of punting to numpy.
    # tup = tuple(map(np.asanyarray, tup))

    # Check to see if we have one homogenized type
    set_of_types = {type(i) for i in tup}
    single_arr_type = set_of_types.pop() if len(set_of_types) == 1 else None

    # Does this type have an 'hstack' method defined?
    # If so, we *must* use it since the existence of the method indicates
    # the array needs special treatment during hstacking; it is not safe/correct
    # to fall back to the standard rc.HStack() / np.hstack() for such arrays.
    if single_arr_type is not None and hasattr(single_arr_type, "hstack"):
        arr_type_hstack_func = getattr(single_arr_type, "hstack")
        # pass kwargs in case special type has unique keywords
        return arr_type_hstack_func(tup, **kwargs)
    elif single_arr_type is None:
        single_arr_type = TypeRegister.FastArray

    dtypenum = -1

    if dtype is not None:
        try:
            dtypenum = dtype.num
        except:
            dtypenum = np.dtype(dtype).num

    try:
        hstack_result = rc.HStack(tup, dtypenum)
    except:
        hstack_result = np.hstack(tup)

    return hstack_result.view(single_arr_type)


# ------------------------------------------------------------
def asanyarray(a, dtype=None, order=None):
    # note will get hooked directly
    return rc.AsAnyArray(a, dtype=dtype, order=order)


# ------------------------------------------------------------
def asarray(a, dtype=None, order=None):
    # note will get hooked directly
    return rc.AsFastArray(a, dtype=dtype, order=order)


# ------------------------------------------------------------
def _FixupDocStrings():
    """
    Load all the member function of this module
    Load all the member functions of the np module
    If we find a match, copy over the doc strings
    """
    mymodule = sys.modules[__name__]
    all_myfunctions = inspect.getmembers(mymodule, inspect.isfunction)
    all_npfunctions = inspect.getmembers(np)
    # all_npfunctions += inspect.getmembers(np, inspect.isfunction)
    # all_npfunctions += inspect.getmembers(np, inspect.isbuiltin)

    # build dictionary
    npdict = {}
    for funcs in all_npfunctions:
        npdict[funcs[0]] = funcs[1]

    # now for each function that has an np flavor, copy over the doc strings
    for funcs in all_myfunctions:
        if funcs[0] in npdict:
            # print("doc fix", funcs[0])

            # combine riptable docstring with numpy docstring
            npdoc = npdict[funcs[0]].__doc__
            if npdoc is not None:
                if funcs[1].__doc__ is None:
                    funcs[1].__doc__ = npdoc

            # old, only uses numpy docstring
            # funcs[1].__doc__ = npdict[funcs[0]].__doc__
        else:
            pass
            # print("reject", funcs[0])


# keep this function last
# -- fixup the doc strings for numpy functions we take over
_FixupDocStrings()

# wire asarray to the C function directly
asanyarray = rc.AsAnyArray
asarray = rc.AsFastArray
