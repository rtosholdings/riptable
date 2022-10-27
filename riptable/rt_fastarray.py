from __future__ import annotations

__all__ = ["FastArray", "Threading", "Recycle", "Ledger"]

import logging
import os
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import riptide_cpp as rc
from numpy.core.numeric import ScalarType

from .rt_enum import (
    INVALID_DICT,
    MATH_OPERATION,
    REDUCE_FUNCTIONS,
    ROLLING_FUNCTIONS,
    TIMEWINDOW_FUNCTIONS,
    NumpyCharTypes,
    TypeRegister,
    gBinaryBitwiseUFuncs,
    gBinaryLogicalUFuncs,
    gBinaryUFuncs,
    gNumpyScalarType,
    gReduceUFuncs,
    gUnaryUFuncs,
)
from .rt_grouping import Grouping
from .rt_mlutils import normalize_minmax, normalize_zscore
from .rt_numpy import (
    _searchsorted,
    asanyarray,
    bool_to_fancy,
    crc32c,
    empty,
    empty_like,
    full,
    groupbyhash,
    hstack,
    ismember,
    issorted,
    ones,
    repeat,
    searchsorted,
    sort,
    tile,
    unique,
    where,
    zeros,
)
from .rt_sds import save_sds
from .rt_stats import statx
from .rt_utils import describe, sample
from .Utils.common import cached_weakref_property
from .Utils.rt_display_properties import (
    DisplayConvert,
    ItemFormat,
    default_item_formats,
)

try:
    # optional extra routines if bottleneck installed
    import bottleneck as bn
except Exception:
    pass

if TYPE_CHECKING:
    from .rt_dataset import Dataset
    from .rt_str import FAString

    # pyarrow is an optional dependency.
    try:
        import pyarrow as pa
    except ImportError:
        pass


# Create a logger for this module.
logger = logging.getLogger(__name__)


NUMPY_CONVERSION_TABLE: Mapping[Callable, REDUCE_FUNCTIONS] = {
    np.sum: REDUCE_FUNCTIONS.REDUCE_SUM,
    np.nansum: REDUCE_FUNCTIONS.REDUCE_NANSUM,
    np.amin: REDUCE_FUNCTIONS.REDUCE_MIN,
    np.nanmin: REDUCE_FUNCTIONS.REDUCE_NANMIN,
    np.amax: REDUCE_FUNCTIONS.REDUCE_MAX,
    np.nanmax: REDUCE_FUNCTIONS.REDUCE_NANMAX,
    np.var: REDUCE_FUNCTIONS.REDUCE_VAR,
    np.nanvar: REDUCE_FUNCTIONS.REDUCE_NANVAR,
    np.mean: REDUCE_FUNCTIONS.REDUCE_MEAN,
    np.nanmean: REDUCE_FUNCTIONS.REDUCE_NANMEAN,
    np.std: REDUCE_FUNCTIONS.REDUCE_STD,
    np.nanstd: REDUCE_FUNCTIONS.REDUCE_NANSTD,
    np.argmin: REDUCE_FUNCTIONS.REDUCE_ARGMIN,
    np.nanargmin: REDUCE_FUNCTIONS.REDUCE_NANARGMIN,
    np.argmax: REDUCE_FUNCTIONS.REDUCE_ARGMAX,
    np.nanargmax: REDUCE_FUNCTIONS.REDUCE_NANARGMAX,
    # np.any: REDUCE_FUNCTIONS.REDUCE_ANY,
    # np.all: REDUCE_FUNCTIONS.REDUCE_ALL,
}

import math

import numba as nb


@nb.generated_jit()
def _isnan(x):
    if x == nb.int8:
        return lambda x: x == nb.int8(-128)
    elif x == nb.int16:
        return lambda x: x == nb.int16(-32768)
    elif x == nb.int32:
        return lambda x: x == nb.int32(0x80000000)
    elif x == nb.int64:
        return lambda x: x == nb.int64(0x8000000000000000)
    elif x == nb.uint8:
        return lambda x: x == nb.uint8(0xFF)
    elif x == nb.uint16:
        return lambda x: x == nb.uint16(0xFFFF)
    elif x == nb.uint32:
        return lambda x: x == nb.uint32(0xFFFFFFFF)
    elif x == nb.uint64:
        return lambda x: x == nb.uint64(0xFFFFFFFFFFFFFFFF)
    else:
        return lambda x: math.isnan(x)


@nb.njit(parallel=True)
def _fnansumhelper(x, filter):
    ret = 0
    length = 0
    for i in nb.prange(len(x)):
        if filter[i] and not _isnan(x[i]):
            ret += x[i]
            length += 1
    return (ret, length)


def _fnansum(x, filter):
    return _fnansumhelper(x, filter)[0]


def _fnanmean(x, filter):
    (tot, n) = _fnansumhelper(x, filter)
    return tot / n


@nb.njit(parallel=True)
def _fnanvar(x, filter):

    abc = 0.0
    length = 0

    for i in nb.prange(len(x)):
        if filter[i] and not _isnan(x[i]):
            abc += x[i]
            length += 1

    mean = abc / length

    ret = 0.0

    for i in nb.prange(len(x)):
        if filter[i] and not _isnan(x[i]):
            ret += (x[i] - mean) ** 2
    if length > 1:
        return ret / (length - 1)
    if length == 1:
        return np.NaN
    if length == 0:
        raise ValueError("Tried to take the variance of an empty array.")


def _fnanstd(x, filter):
    return math.sqrt(_fnanvar(x, filter))


@nb.njit(parallel=True)
def _fsumhelper(x, filter):
    ret = 0
    length = 0
    for i in nb.prange(len(x)):
        if filter[i]:
            ret += x[i]
            length += 1
    return (ret, length)


def _fsum(x, filter):
    return _fsumhelper(x, filter)[0]


def _fmean(x, filter):
    (tot, n) = _fsumhelper(x, filter)
    return tot / n


@nb.njit(parallel=True)
def _fvar(x, filter):

    abc = 0.0
    length = 0

    for i in nb.prange(len(x)):
        if filter[i]:
            abc += x[i]
            length += 1

    mean = abc / length

    ret = 0.0

    for i in nb.prange(len(x)):
        if filter[i]:
            ret += (x[i] - mean) ** 2
    if length > 1:
        return ret / (length - 1)
    if length == 1:
        return np.NaN
    if length == 0:
        raise ValueError("Tried to take the variance of an empty array.")


def _fstd(x, filter):
    return math.sqrt(_fvar(x, filter))


# --------------------------------------------------------------
def FA_FROM_UINT8(uint8arr):
    """
    Used in de-pickling
    """
    return rc.CompressDecompressArrays([uint8arr], 1)[0]


# --------------------------------------------------------------
def FA_FROM_BYTESTRING(bytestring):
    """
    Used in de-pickling when tostring() used (currently disabled)
    """
    return FA_FROM_UINT8(np.frombuffer(bytestring, dtype=np.uint8))


# --------------------------------------------------------------
def logical_find_common_type(arraytypes, scalartypes, scalarval):
    """
    assumes one scalar and one array

    """
    scalar = scalartypes[0]
    array = arraytypes[0]

    unsigned = False
    isinteger = False

    # TJD this routine needs to be rewritten
    # can check isinstance(scalar,(np.integer, int))

    # if this comes in as np.int64 and not a dtype, we convert to a dtype
    if not hasattr(scalar, "char"):
        scalar = np.dtype(scalar)

    if scalar.char in NumpyCharTypes.UnsignedInteger:
        unsigned = True
        isinteger = True
    if scalar.char in NumpyCharTypes.Integer:
        isinteger = True

    if not isinteger:
        # go by numpy upscale rules
        # NOTE: should consider allowing integer ^ True -- or changing a bool scalar to an int
        # print("punting not integer scalar", scalar)
        return np.find_common_type(arraytypes, scalartypes)

    unsigned = False
    isinteger = False

    try:
        if array.char in NumpyCharTypes.UnsignedInteger:
            unsigned = True
            isinteger = True
        if array.char in NumpyCharTypes.Integer:
            isinteger = True
    except:
        pass

    # if isinstance(array, int):
    #    isinteger = True

    # IF ARRAY IS UNSIGNED BY SCALAR IS SIGNED upcast

    if not isinteger:
        # go by numpy upscale rules
        # NOTE: should consider allowing integer ^ True -- or changing a bool scalar to an int
        # print("punting not integer array", array)
        return np.find_common_type(arraytypes, scalartypes)

    final = None

    scalarval = int(scalarval)

    # Determine the possible integer upscaling based on the scalar value
    if unsigned:
        if scalarval <= 255:
            final = np.uint8
        elif scalarval <= 65535:
            final = np.uint16
        elif scalarval <= (2**32 - 1):
            final = np.uint32
        elif scalarval <= (2**64 - 1):
            final = np.uint64
        else:
            final = np.float64
    else:
        if scalarval >= -128 and scalarval <= 127:
            final = np.int8
        elif scalarval >= -32768 and scalarval <= 32767:
            final = np.int16
        elif scalarval >= -(2**31) and scalarval <= (2**31 - 1):
            final = np.int32
        elif scalarval >= -(2**63) and scalarval <= (2**63 - 1):
            final = np.int64
        else:
            final = np.float64

    final = np.dtype(final)

    # do not allow downcasting
    if array.num < final.num:
        # print("returning final", final)
        return final

    return array

    # if type(args[0]) in ScalarType:
    #    print("converting arg2 to ", final_dtype)
    #    args[1] = args[1].astype(final_dtype);
    # else:
    #    print("converting arg1 to ", final_dtype)
    #    args[0] = args[0].astype(final_dtype);


# --------------------------------------------------------------
def _ASTYPE(self, dtype):
    """internal call from array_ufunc to convert arrays.  returns numpy arrays"""
    # return self.astype(dtype)
    to_num = dtype.num
    if self.dtype.num <= 13 and to_num <= 13:
        if FastArray.SafeConversions:
            # perform a safe conversion understanding sentinels
            return TypeRegister.MathLedger._AS_FA_TYPE(self, to_num)._np
        else:
            # perform unsafe conversion NOT understanding sentinels
            return TypeRegister.MathLedger._AS_FA_TYPE_UNSAFE(self, to_num)._np

    return self.astype(dtype)


# --------------------------------------------------------------
# --------------------------------------------------------------
class FastArray(np.ndarray):
    """

    A `FastArray` is a 1-dimensional array of items that are the same data type.

    Because it's a subclass of NumPy's `numpy.ndarray`, all ``ndarray`` functions and attributes
    can be used with `FastArray` objects. However, Riptable optimizes many of NumPy's
    functions to make them faster and more memory-efficient. Riptable has also added
    some methods.

    `FastArray` objects with more than 1 dimension are not supported.

    See `NumPy's
    docs <https://numpy.org/devdocs/reference/generated/numpy.ndarray.html>`_ for
    details on all ``ndarray`` methods and attributes.

    Parameters
    ----------
    arr : array, iterable, or scalar value
        Contains data to be stored in the `FastArray`.

    **kwargs
        Additional keyword arguments to be passed to the function.

    Notes
    -----
    To improve performance, `FastArray` objects take over some of NumPy's universal functions
    (ufuncs), use array recycling and multiple threads, and pass certain method calls to
    `Bottleneck <https://kwgoodman.github.io/bottleneck-doc/index.html>`_.

    Note that whenever Riptable has implemented its own version of
    an existing NumPy method, a call to the NumPy method results in a call to the
    optimized Riptable version instead. We encourage users to directly call the Riptable
    method in order to avoid any confusion as to what method is actually being called.

    See the list of `NumPy Methods Optimized by Riptable for FastArrays
    <https://eot.gitlab.ds.susq.com/sigpydata/riptable/riptable/tutorial/tutorial_numpy_rt.html>`_.


    Examples
    --------
    **Construct a FastArray**

    Pass a list to the constructor:

    >>> rt.FastArray([1, 2, 3, 4, 5])
    FastArray([1, 2, 3, 4, 5])

    >>> #NOTE: rt.FA also works.
    >>> rt.FA([1.0, 2.0, 3.0, 4.0, 5.0])
    FastArray([1., 2., 3., 4., 5.])

    Or use a utility function:

    >>> rt.full(10, 0.7)
    FastArray([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])

    >>> rt.arange(10)
    FastArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    You can optionally specify a data type:

    >>> x = rt.FastArray([3, 6, 10],  dtype = rt.float64)
    >>> x, x.dtype
    (FastArray([ 3.,  6., 10.]), dtype('float64'))

    >>> # Using a string shortcut:
    >>> x = rt.FastArray([3,6,10],  dtype = 'float64')
    >>> x, x.dtype
    (FastArray([ 3.,  6., 10.]), dtype('float64'))

    By default, characters are stored as byte strings. When ``unicode=True``,
    the `FastArray` allows Unicode characters.

    >>> rt.FA(list('abc'), unicode=True)
    FastArray(['a', 'b', 'c'], dtype='<U1')

    To convert an existing NumPy array, use the `FastArray` constructor.

    >>> np_arr = np.array([1, 2, 3])
    >>> rt.FA(np_arr)
    FastArray([1, 2, 3])

    To view the NumPy array as a `FastArray` (which is slightly less expensive than
    using the constructor), use the `view` method.

    >>> fa = np_arr.view(FA)
    >>> fa
    FastArray([1, 2, 3])

    To view it as a NumPy array again:

    >>> fa.view(np.ndarray)
    array([1, 2, 3])

    >>> # Alternatively:
    >>> fa._np
    array([1, 2, 3])

    **Get a Subset of a FastArray**

    You can use standard Python slicing notation or fancy indexing to access a
    subset of a `FastArray`.

    >>> # Create a FastArray:
    >>> array = rt.arange(8)**2
    >>> array
    FastArray([0, 1, 4, 9, 16, 25, 36, 49])
    >>> # Use Python slicing to get elements 2, 3, and 4:
    >>> array[2:5]
    FastArray([4, 9, 16])

    >>> # Use fancy indexing to get elements 2, 4, and 1 (in that order):
    >>> array[[2, 4, 1]]
    FastArray([4, 16, 1])

    For more details, see the examples for 1-dimensional arrays in NumPy's docs:
    `Indexing on ndarrays <https://numpy.org/doc/stable/user/basics.indexing.html>`_.

    Note that slicing creates a view of the array and does not copy the underlying data;
    modifying the slice modifies the original array. Fancy indexing creates a copy of
    the extracted data; modifying this array does not modify the original array.

    You can also pass a Boolean mask array.

    >>> # Create a Boolean mask:
    >>> evenMask = (array % 2 == 0)
    >>> evenMask
    FastArray([True, False, True, False, True, False, True, False])
    >>> # Index using the Boolean mask:
    >>> array[evenMask]
    FastArray([0, 4, 16, 36])

    **How to Subclass FastArray**

    Include the required class definition:

    >>> class TestSubclass(FastArray):
    ...     def __new__(cls, arr, **args):
    ...         # Before this call, arr needs to be a np.ndarray instance.
    ...         return arr.view(cls)
    ...     def __init__(self, arr, **args):
    ...         pass

    If the subclass is computable, you might define your own math operations. In these
    operations, you might define what the subclass can be computed with. For examples of
    new definitions, see the `DateTimeNano` class.

    Common operations to hook are comparisons (``__eq__()``, ``__ne__()``, ``__gt__()``,
    ``__lt__()``, ``__le__()``, ``__ge__()``) and basic math functions (``__add__()``,
    ``__sub__()``, ``__mul__()``, etc.).

    Bracket indexing operations are very common. If the subclass needs to set or return
    a value other than that in the underlying array, you need to take over
    `__getitem__()` or `__setitem__()`.

    Indexing is also used in display. For regular console/notebook display, you need to
    take over:

    * `__repr__()`
    * `__str__()`
    * `_repr_html_()` (for JupyterLab and Jupyter notebooks)

    If the array is being displayed in a `Dataset` and you require certain formatting, you
    need to define two more methods:

    ``display_query_properties()``
        Returns an `ItemFormat` object (see `rt.Utils.rt_display_properties`)

    ``display_convert_func()``
        The conversion function returned by ``display_query_properties()``
        must return a string. Each item being displayed, the result of ``__getitem__()``
        at a single index, will go through this function individually, accompanied by
        an `ItemFormat` object.

    Many Riptable operations need to return arrays of the same class they received. To
    ensure that your subclass will retain its special properties, you need to take over
    `newclassfrominstance()`. Failure to take this over will often result in an object
    with uninitialized variables.

    `copy()` is another method that is called generically in Riptable routines, and
    needs to be taken over to retain subclass properties.

    For a view of the underlying `FastArray`, you can use the `_fa` property.
    """

    # Defines a generic np.ndarray subclass, that can cache numpy arrays
    # Static Class VARIABLES

    # change this to show or less values on __repr__
    MAX_DISPLAY_LEN = 10

    # set to 2 or 3 for extra debug information
    Verbose = 1

    # set to true for reusing numpy arrays instead of deleting them completely
    Recycle = True

    # set to true to preserve sentinels during internal array_ufunc calculations
    SafeConversions = True

    # set to false to be just normal numpy
    FasterUFunc = True

    NEW_ARRAY_FUNCTION_ENABLED = False
    """Enable implementation of array function protocol (default False)."""

    # 0=Quiet, 1=Warn, 2=Exception
    WarningLevel = 1

    # set to true to not allow ararys we do not support
    NoTolerance = False

    # set to false to not compress when pickling
    CompressPickle = True

    # a dictionary to avoid repeating warnings in multiple places
    # TODO: wrap this in a class so that warnings can be turned on/off
    WarningDict = {
        "multiple_dimensions": "FastArray contains two or more dimensions greater than one - shape:{}.  Problems may occur."
    }

    # For reduction operations, the identity element of the operation (for operations
    # where such an element is defined).
    # N.B. As of numpy 1.19 it does not appear there's a straightforward way of getting from
    #   something like ``np.sum`` back to ``np.add``, from which we could get the .identity property.
    #   If that ever changes, this dictionary would no longer be necessary so it can+should be removed.
    _reduce_op_identity_value: Mapping[REDUCE_FUNCTIONS, Any] = {
        REDUCE_FUNCTIONS.REDUCE_ALL: True,  # np.all(np.array([]))
        REDUCE_FUNCTIONS.REDUCE_ANY: False,  # np.any(np.array([]))
        REDUCE_FUNCTIONS.REDUCE_NANSUM: np.add.identity,
        REDUCE_FUNCTIONS.REDUCE_SUM: np.add.identity,
    }

    # --------------------------------------------------------------------------
    class _ArrayFunctionHelper:
        # TODO add usage examples
        """
        Array function helper is responsible maintaining the array function protocol array implementations in the
        form of the following API:

        - get_array_function: given the Numpy function, returns overridden array function
        - get_array_function_type_compatibility_check: given the Numpy function, returns overridden array function type compatibility check
        - register_array_function: a function decorator whose argument is the Numpy function to override and the function that will override it
        - register_array_function_type_compatibility: similar to register_array_function, but guards against incompatible array function protocol type arguments for the given Numpy function
        - deregister: deregistration of the Numpy function and type compatibility override
        - deregister_array_function_type_compatibility: deregistration of Numpy function type compatibility override

        """
        # TODO design consideration - using a single dict with tuple type compatibility and redirected callables
        # where a default type compatibility check can be the default value
        # a dictionary that maps numpy functions to our custom variants
        HANDLED_FUNCTIONS: Dict[callable, callable] = {}
        """Dictionary of Numpy API function with overridden functions."""
        HANDLED_TYPE_COMPATIBILITY_CHECK: Dict[callable, callable] = {}
        """Dictionary of type compatibility functions per each Numpy API overridden function."""

        @classmethod
        def get_array_function(cls, np_function: Callable) -> Optional[Callable]:
            """
            Given the Numpy function, returns overridden array function if implemented, otherwise None.

            Parameters
            ----------
            np_function: callable
                The overridden Numpy array function.

            Returns
            -------
            callable, optional
                The overridden function as a callable or None if it's not implemented.
            """
            return cls.HANDLED_FUNCTIONS.get(np_function, None)

        @classmethod
        def get_array_function_type_compatibility_check(cls, np_function: Callable) -> Optional[Callable]:
            """
            Given the Numpy function, returns the corresponding array function type compatibility callable, otherwise None.

            Parameters
            ----------
            np_function: callable
                The overridden Numpy array function.

            Returns
            -------
            callable, optional
                The overridden type compatibility function as a callable or None if it's not implemented.
            """
            return cls.HANDLED_TYPE_COMPATIBILITY_CHECK.get(np_function, None)

        @classmethod
        def register_array_function(cls, np_function: Callable) -> Callable:
            """
             A function decorator whose argument is the Numpy function to override and the function that will override it.
             This registers the `np_function` with the function that it decorates.

            Parameters
            ----------
            np_function: callable
                The overridden Numpy array function.

            Returns
            -------
            callable
                The decorator that registers `np_function` with the decorated function.
            """
            # @wraps(np_function)
            def decorator(func):
                cls.HANDLED_FUNCTIONS[np_function] = func
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"{cls.__name__}.register_array_function: registered {repr(func.__name__)} in place of {np_function.__name__}"
                    )
                return func

            return decorator

        @classmethod
        def register_array_function_type_compatibility(cls, np_function: Callable) -> Callable:
            """
            This registers the type compatibility check for the `np_function` with the function that it decorates.

            Parameters
            ----------
            np_function: callable
                The overridden Numpy array function.

            Returns
            -------
            callable
                The decorator that registers the type compatibility check for the `np_function` with the decorated function.
            """
            # @wraps(np_function)
            def decorator(check_type_compatibility):
                cls.HANDLED_TYPE_COMPATIBILITY_CHECK[np_function] = check_type_compatibility
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"{cls.__name__}.register_array_function_type_compatibility: registered type compatibility check {repr(check_type_compatibility)} for array function {np_function.__name__}"
                    )
                return check_type_compatibility

            return decorator

        @classmethod
        def deregister_array_function(cls, np_function: Callable) -> None:
            """
            Deregistration of the Numpy function and type compatibility override.

            Parameters
            ----------
            np_function: callable
                The overridden Numpy array function.
            """
            if cls.get_array_function(np_function) is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"{cls.__name__}.deregister_array_function: deregistered {repr(np_function.__name__)}")
                del cls.HANDLED_FUNCTIONS[np_function]

        @classmethod
        def deregister_array_function_type_compatibility(cls, np_function: Callable) -> None:
            """
            Deregistration of the Numpy function and type compatibility override.

            Parameters
            ----------
            np_function: callable
                The overridden Numpy array function.
            """
            if cls.HANDLED_TYPE_COMPATIBILITY_CHECK.get(np_function, None) is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"{cls.__name__}.deregister_array_function_type_compatibility: deregistered {repr(np_function.__name__)}"
                    )
                del cls.HANDLED_TYPE_COMPATIBILITY_CHECK[np_function]

        @classmethod
        def deregister(cls, np_function: Callable) -> None:
            cls.deregister_array_function(np_function)
            cls.deregister_array_function_type_compatibility(np_function)

    # --------------------------------------------------------------------------
    @classmethod
    def _possibly_warn(cls, warning_string: str) -> Optional[bool]:
        if cls.WarningLevel == 0:
            return False
        if cls.WarningLevel == 1:
            warnings.warn(warning_string)
            return True
        raise TypeError(warning_string)

    # --------------------------------------------------------------------------
    def __new__(cls, arr, **kwargs) -> FastArray:

        allow_unicode = kwargs.get("unicode", False)
        try:
            del kwargs["unicode"]
        except:
            pass
        # If already a numpy array no need to call asany
        if isinstance(arr, np.ndarray) and len(kwargs) == 0:
            instance = arr
            if isinstance(instance, cls) and instance.dtype.char != "U":
                if instance.dtype.char not in NumpyCharTypes.Supported:
                    cls._possibly_warn(
                        f"FastArray contains an unsupported type '{instance.dtype}'.  Problems may occur.  Consider categoricals."
                    )
                # if already a FastArray, do not rewrap this
                return instance
        else:
            # flip the list or other object to a numpy array
            instance = np.asanyarray(arr, **kwargs)

        if not allow_unicode and instance.dtype.char == "U":
            try:
                instance = np.asarray(instance, dtype="S")
            except:
                pass

        if len(instance.shape) == 0:
            if instance.dtype.char in NumpyCharTypes.Supported:
                instance = np.asanyarray([instance], **kwargs)
            else:
                # np.asarray on a set will return an object of 1
                if isinstance(arr, set):
                    instance = np.asarray(list(arr), **kwargs)
                else:
                    raise TypeError(f"FastArray cannot initialize {arr}")

        if instance.ndim > 1:
            # only one dimension can be greater than one
            if cls._check_ndim(instance) > 1:
                cls._possibly_warn(FastArray.WarningDict["multiple_dimensions"].format(instance.shape))
                # warnings.warn(f"FastArray contains two or more dimensions greater than one - shape:{instance.shape}.  Problems may occur.")
        elif not (instance.flags.f_contiguous or instance.flags.c_contiguous):
            # copy should eliminate strides problem
            instance = instance.copy()
            cls._possibly_warn(f"FastArray initialized with strides.")

        # for arrays that can cause problems but we allow now
        if cls.NoTolerance:
            if not (instance.flags.f_contiguous or instance.flags.c_contiguous):
                # copy should eliminate strides problem
                instance = instance.copy()
                cls._possibly_warn(f"FastArray initialized with strides.")

        if instance.dtype.char not in NumpyCharTypes.Supported:
            cls._possibly_warn(
                f"FastArray contains an unsupported type '{instance.dtype}'.  Problems may occur.  Consider categoricals."
            )

        return instance.view(cls)

    def __array_finalize__(self, obj):
        """Finalizes self from other, called as part of ndarray.__new__()"""
        if obj is None:
            return
        from_peer = isinstance(obj, FastArray)
        if from_peer and hasattr(obj, "_name"):
            self._name = obj._name

    # --------------------------------------------------------------------------
    def __reduce__(self):
        """
        Used for pickling.
        For just a FastArray we pass back the view of the np.ndarray, which then knows how to pickle itself.
        NOTE: I think there is a faster way.. possible returning a byte string.
        """
        cls = type(self)

        # check if subclassed routine knows how to serialize itself
        if hasattr(self, "_build_sds_meta_data"):
            try:
                name = self._name
            except:
                name = "unknown"

            tups = self._build_sds_meta_data(name)
            return (cls._load_from_sds_meta_data, (name, self.view(FastArray), tups[1], tups[0]))

        # set to true to turn compression on
        if cls.CompressPickle and len(self) > 0:
            # create a single compressed array of uint8
            carr = rc.CompressDecompressArrays([self], 0)[0]
            return (FA_FROM_UINT8, (carr.view(np.ndarray),))
        else:
            return (
                cls.__new__,
                (
                    cls,
                    self.view(np.ndarray),
                ),
            )

    # --------------------------------------------------------------------------
    @classmethod
    def _check_ndim(cls, instance):
        """
        Iterates through dimensions of an array, counting how many dimensions have values greater than 1.
        Problems may occure with multidimensional FastArrays, and the user will be warned.
        """
        index = 0
        aboveone = 0
        while index < instance.ndim:
            if instance.shape[index] > 1:
                aboveone += 1
            index += 1
        return aboveone

    # --------------------------------------------------------------------------
    def get_name(self) -> str:
        """
        Get the name that's assigned to a `FastArray`.

        When a `FastArray` object is created, it has no name. It can be assigned a name
        via `set_name`. For details, see :meth:`FastArray.set_name`.

        Returns
        -------
        str or None
            The assigned name, or None if the array has not been named.

        See Also
        --------
        FastArray.set_name

        Examples
        --------
        Assign the `FastArray` a name using :meth:`FastArray.set_name`:

        >>> a = rt.arange(5)
        >>> a.set_name('FA Name')
        FastArray([0, 1, 2, 3, 4])

        Get the name:

        >>> a.get_name()
        'FA Name'
        """
        name = None
        try:
            name = self._name
        except:
            pass
        return name

    # --------------------------------------------------------------------------
    def set_name(self, name) -> FastArray:
        """
        Assign a name to a `FastArray`.

        A `FastArray` is a wrapper around a NumPy `ndarray`. When a `FastArray` is
        created, it has no name. You can assign it a name using `set_name`.

        **Interactions with Dataset Objects**

        When an unnamed `FastArray` is added to a `Dataset`:

        - The `FastArray` inherits the name of the `Dataset` column.
        - Calling ``fa.set_name`` or ``ds.col.set_name``, or changing the displayed
          column name via ``ds.col_rename``, changes the name assigned to the
          `FastArray`.
            - Note that calling ``fa.set_name`` or ``ds.col.set_name`` doesn't change the
              displayed column name.

        When a named `FastArray` is added to a `Dataset`:

        - A new `FastArray` instance is created that inherits the `Dataset` column name.
        - Calling ``ds.col.set_name`` or changing the displayed column name via
          ``ds.col_rename`` changes the new instance's name.
        - Calling `set_name` on the original `FastArray` instance changes only that instance's
          name.

        In both cases, the NumPy array underlying the `FastArray` is shared -- changes
        to its values appear in the `Dataset` column, and vice-versa.

        **Interactions with FastArray Objects**

        - When a `FastArray` is created as a view of another, named `FastArray`, the new
          `FastArray` instance inherits the name from the original `FastArray`.
        - Whether the original `FastArray` is named or unnamed, calling `set_name` on
          either `FastArray` does not change the name of the other `FastArray`.

        Parameters
        ----------
        name : str
            The name to assign to the `FastArray`.

        Returns
        -------
        `FastArray`
            The `FastArray` is returned. The name can be accessed using
            :meth:`FastArray.get_name`.

        See Also
        --------
        FastArray.get_name

        Examples
        --------
        >>> a = rt.arange(5)
        >>> a.set_name('FA Name')
        FastArray([0, 1, 2, 3, 4])

        You can get the name using :meth:`FastArray.get_name`:

        >>> a.get_name()
        'FA Name'

        When an unnamed `FastArray` is added to a `Dataset` column, the `FastArray`
        inherits the name of the column.

        >>> a = rt.FastArray([1, 2, 3])
        >>> ds = rt.Dataset()
        >>> ds.Column_Name = a
        >>> a.get_name()
        'Column_Name'

        Calling ``ds.col.set_name`` changes the name assigned to the `FastArray`
        (but not the displayed column name).

        >>> ds.Column_Name.set_name('New Name')
        >>> a.get_name()
        'New Name'
        >>> ds
        #   Column_Name
        -   -----------
        0             1
        1             2
        2             3

        When a named `FastArray` is added to a `Dataset` column, a new `FastArray`
        instance is created that inherits the column name. The original instance is
        not renamed.

        >>> a = rt.FastArray([1, 2, 3])
        >>> a.set_name('FA Name')
        >>> ds = rt.Dataset()
        >>> ds.Column_Name = a
        >>> ds.Column_Name.get_name()
        'Column_Name'
        >>> a.get_name()
        'FA Name'

        Changing the displayed column name affects the name of the new instance,
        but not the name of the original `FastArray`.

        >>> ds.col_rename('Column_Name', 'New_Column')
        >>> ds.New_Column.get_name()
        'New_Column'
        >>> a.get_name()
        'FA Name'
        """
        self._name = name
        return self

    # --------------------------------------------------------------------------
    @staticmethod
    def _FastFunctionsOn():
        if FastArray.Verbose > 0:
            print(f"FASTFUNC ON: fastfunc was {FastArray.FasterUFunc}")
        FastArray.FasterUFunc = True

    @staticmethod
    def _FastFunctionsOff():
        if FastArray.Verbose > 0:
            print(f"FASTFUNC OFF: fastfunc was {FastArray.FasterUFunc}")
        FastArray.FasterUFunc = False

    @property
    def _np(self) -> np.ndarray:
        """
        quick way to return a numpy array instead of fast array
        """
        return self.view(np.ndarray)

    @staticmethod
    def _V0():
        print("setting verbose level to 0")
        FastArray.Verbose = 0
        return FastArray.Verbose

    @staticmethod
    def _V1():
        print("setting verbose level to 1")
        FastArray.Verbose = 1
        return FastArray.Verbose

    @staticmethod
    def _V2():
        print("setting verbose level to 2")
        FastArray.Verbose = 2
        return FastArray.Verbose

    @staticmethod
    def _ON():
        """
        enable intercepting array ufunc
        """
        return FastArray._FastFunctionsOn()

    @staticmethod
    def _OFF():
        """
        disable intercepting of array ufunc
        """
        return FastArray._FastFunctionsOff()

    @staticmethod
    def _TON():
        print("Threading on")
        return rc.ThreadingMode(0)

    @staticmethod
    def _TOFF():
        print("Threading off")
        return rc.ThreadingMode(1)

    @staticmethod
    def _RON(quiet=False):
        """
        Turn on recycling.

        Parameters
        ----------
        quiet: bool, optional

        Returns
        -------
        True if recycling was previously on, else False
        """
        if not quiet:
            print("Recycling numpy arrays on")
        result = rc.SetRecycleMode(0)
        FastArray.Recycle = True
        return result

    @staticmethod
    def _ROFF(quiet=False):
        """
        Turn off recycling.

        Parameters
        ----------
        quiet: bool, optional

        Returns
        -------
        True if recycling was previously on, else False
        """
        if not quiet:
            print("Recycling numpy arrays off")
        result = rc.SetRecycleMode(1)
        FastArray.Recycle = False
        return result

    @staticmethod
    def _RDUMP():
        """
        Displays to server's stdout

        Returns
        -------
        Total size of items not in use
        """
        return rc.RecycleDump()

    @staticmethod
    def _GCNOW(timeout: int = 0):
        """
        Pass the garbage collector timeout value to cleanup.
        Passing 0 will force an immediate garbage collection.

        Returns
        -------
        Dictionary of memory heuristics including 'TotalDeleted'
        """
        import gc

        gc.collect()
        result = rc.RecycleGarbageCollectNow(timeout)
        totalDeleted = result["TotalDeleted"]
        if totalDeleted > 0:
            FastArray._GCNOW(timeout)
        return result

    @staticmethod
    def _GCSET(timeout: int = 100):
        """
        Pass the garbage collector timeout value to expire
        The timeout value is roughly in 2/5 secs
        A value of 100 is usually about 40 seconds

        Returns
        -------
        Previous timespan
        """
        return rc.RecycleSetGarbageCollectTimeout(timeout)

    @staticmethod
    def _LON():
        """Turn the math ledger on to record all array math routines"""
        return TypeRegister.MathLedger._LedgerOn()

    @staticmethod
    def _LOFF():
        """Turn the math ledger off"""
        return TypeRegister.MathLedger._LedgerOff()

    @staticmethod
    def _LDUMP(dataset=True):
        """Print out the math ledger"""
        return TypeRegister.MathLedger._LedgerDump(dataset=dataset)

    @staticmethod
    def _LDUMPF(filename):
        """Save the math ledger to a file"""
        return TypeRegister.MathLedger._LedgerDumpFile(filename)

    @staticmethod
    def _LCLEAR():
        """Clear all the entries in the math ledger"""
        return TypeRegister.MathLedger._LedgerClear()

    # --------------------------------------------------------------------------
    def __setitem__(self, fld, value):
        """
        Used on the left hand side of
        arr[fld] = value

        This routine tries to convert invalid dtypes to that invalids are preserved when setting
        The mbset portion of this is no written (which will not raise an indexerror on out of bounds)

        Parameters
        ----------
        fld: scalar, boolean, fancy index mask, slice, sequence, or list
        value: scalar, sequence or dataset value as follows
                sequence can be list, tuple, np.ndarray, FastArray

        Raises
        -------
        IndexError

        """
        newvalue = None

        # try to make an array, even if array of 1
        if np.isscalar(value):
            if not isinstance(value, (str, bytes, np.bytes_, np.str_)):
                # convert to array of 1 item
                newvalue = FastArray([value])
        elif isinstance(value, (list, tuple)):
            # convert to numpy array
            newvalue = FastArray(value, unicode=True)
        elif isinstance(value, np.ndarray):
            # just reference it
            newvalue = value

        if newvalue is not None:

            # now we have a numpy array.. convert the dtype to match us
            # this should take care of invalids
            # convert first 14 common types (bool, ints, floats)
            if newvalue.dtype != self.dtype and newvalue.dtype.num <= 13:
                newvalue = newvalue.astype(self.dtype)

            # check for boolean array since we do not handle fancy index yet
            if isinstance(fld, np.ndarray) and fld.dtype.num == 0:
                is_unsupported = self._is_not_supported(newvalue)
                if is_unsupported:
                    # make it contiguous, in case that's the problem?
                    newvalue = newvalue.copy()
                    # re-test support, just to be sure.
                    is_unsupported = self._is_not_supported(newvalue)

                # if supported, call our setitem, it will return False if it fails
                if not is_unsupported:
                    if rc.SetItem(self, fld, newvalue):
                        return
            try:
                np.ndarray.__setitem__(self, fld, newvalue)
            except Exception:
                # odd ball cases handled here like ufunc tests
                np.ndarray.__setitem__(self, fld, value)
            return

        # punt to normal numpy
        np.ndarray.__setitem__(self, fld, value)

    # --------------------------------------------------------------------------
    def __getitem__(self, fld) -> FastArray:
        """
        riptable has special routines to handle array input in the indexer.
        Everything else will go to numpy getitem.
        """
        if isinstance(fld, np.ndarray):
            # result= super(FastArray, self).__getitem__(fld).view(FastArray)
            if fld.dtype == np.bool_:
                # make sure no striding
                # NOTE: will fail on self.dtype.byteorder as little endian
                if self.flags.f_contiguous:
                    # dimensions must match
                    if self.ndim == fld.ndim and self.ndim == 1:
                        return TypeRegister.MathLedger._INDEX_BOOL(self, fld)

            # if we have fancy indexing and we support the array type, make sure we do not have stride problem
            if fld.dtype.char in NumpyCharTypes.AllInteger and self.dtype.char in NumpyCharTypes.SupportedAlternate:
                if self.flags.f_contiguous and fld.flags.f_contiguous:
                    if len(self.shape) == 1:
                        return TypeRegister.MathLedger._MBGET(self, fld)

            result = TypeRegister.MathLedger._GETITEM(super(FastArray, self), fld)
            return result.view(FastArray)
        else:
            # could be a list which is often converted to an array

            # This assumes that FastArray has a sole parent, np.ndarray
            # If this changes, the super() call needs to be used
            return np.ndarray.__getitem__(self, fld)
            # return super(FastArray, self).__getitem__(fld)

    # --------------------------------------------------------------------------
    def display_query_properties(self):
        """
        Returns an ItemFormat object and a function for converting the FastArrays items to strings.
        Basic types: Bool, Int, Float, Bytes, String all have default formats / conversion functions.
        (see Utils.rt_display_properties)

        If a new type is a subclass of FastArray and needs to be displayed in format
        different from its underlying type, it will need to take over this routine.
        """
        arr_type, convert_func = DisplayConvert.get_display_convert(self)
        display_format = default_item_formats.get(arr_type, ItemFormat())
        if len(self.shape) > 1:
            display_format.convert = convert_func
            convert_func = DisplayConvert.convertMultiDims

        # add sentinel value for integer
        if display_format.invalid is None:
            display_format = display_format.copy()
            if self.dtype.char in NumpyCharTypes.AllInteger:
                display_format.invalid = INVALID_DICT[self.dtype.num]
        return display_format, convert_func

    # --------------------------------------------------------------------------
    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True) -> FastArray:
        # result= super(FastArray, self).astype(dtype, order,casting,subok,copy)
        # 17 is object
        # 18 = ASCII string
        # 19 = UNICODE string
        to_num = np.dtype(dtype).num

        # check for contiguous in one or two dimensions
        if self.flags.f_contiguous or self.flags.c_contiguous:
            if order == "K" and subok and copy and self.dtype.num <= 13 and to_num <= 13:
                # perform a safe conversion understanding sentinels
                return TypeRegister.MathLedger._AS_FA_TYPE(self, to_num)

        # punt to numpy
        result = TypeRegister.MathLedger._ASTYPE(super(FastArray, self), dtype, order, casting, subok, copy)
        return result.view(FastArray)

    # --------------------------------------------------------------------------
    def _view_internal(self, type=None):
        """
        FastArray subclasses need to take this over if they want to make a shallow copy of
        a fastarray instead of viewing themselves as a fastarray (which drops their other properties).
        Taking over view directly may have a lot of unintended consequences.
        """
        if type is not FastArray or type is not None:
            newarr = self.view(type)
            # copy all the properties
            newarr.__dict__ = self.__dict__.copy()
            return newarr
        return self.view(FastArray)

    # --------------------------------------------------------------------------
    def copy(self, order="K") -> FastArray:
        # result= super(FastArray, self).copy(order)
        if self.flags.f_contiguous or self.flags.c_contiguous:
            if order == "K" and self.dtype.num <= 13:
                # perform a faster multithreaded copy
                return TypeRegister.MathLedger._AS_FA_TYPE(self, self.dtype.num)

        result = TypeRegister.MathLedger._COPY(super(FastArray, self), order)
        return result.view(FastArray)

    # --------------------------------------------------------------------------
    def copy_invalid(self) -> FastArray:
        """
        Makes a copy of the array filled with invalids.

        Examples
        --------
        >>> rt.arange(5).copy_invalid()
        FastArray([-2147483648, -2147483648, -2147483648, -2147483648, -2147483648])

        >>> rt.arange(5).copy_invalid().astype(np.float32)
        FastArray([nan, nan, nan, nan, nan], dtype=float32)

        See Also
        --------
        FastArray.inv
        FastArray.fill_invalid
        """
        return self.fill_invalid(inplace=False)

    # --------------------------------------------------------------------------
    @property
    def inv(self) -> np.number:
        """
        Returns the invalid value for the array.
        np.int8: -128
        np.uint8: 255
        np.int16: -32768
        ...and so on..

        Examples
        --------
        >>> rt.arange(5).inv
        -2147483648

        See Also
        --------
        FastArray.copy_invalid
        FastArray.fill_invalid
        INVALID_DICT
        """
        return INVALID_DICT[self.dtype.num]

    # --------------------------------------------------------------------------
    def fill_invalid(self, shape=None, dtype=None, inplace=True) -> FastArray:
        """
        Fills array or returns copy of array with invalid value of array's dtype or a specified one.
        Warning: by default this operation is inplace.

        Examples
        --------
        >>> a=rt.arange(5).fill_invalid()
        >>> a
        FastArray([-2147483648, -2147483648, -2147483648, -2147483648, -2147483648])

        See Also
        --------
        FastArray.inv
        FastArray.fill_invalid
        """
        return self._fill_invalid_internal(shape=shape, dtype=dtype, inplace=inplace)

    def _fill_invalid_internal(self, shape=None, dtype=None, inplace=True, fill_val=None):
        if dtype is None:
            dtype = self.dtype
        if shape is None:
            shape = self.shape
        elif not isinstance(shape, tuple):
            shape = (shape,)

        if fill_val is None:
            inv = INVALID_DICT[dtype.num]
        else:
            inv = fill_val

        if inplace is True:
            if shape != self.shape:
                raise ValueError(
                    f"Inplace fill invalid cannot be different number of rows than existing array. Got {shape} vs. length {len(self)}"
                )
            if dtype != self.dtype:
                raise ValueError(
                    f"Inplace fill invalid cannot be different dtype than existing categorical. Got {dtype} vs. {len(self.dtype)}"
                )

            self.fill(inv)
        else:
            arr = full(shape, inv, dtype=dtype)
            return arr

    # -------------------------------------------------------------------------
    def isin(self, test_elements, assume_unique=False, invert=False) -> FastArray:
        """
        Calculates `self in test_elements`, broadcasting over `self` only.
        Returns a boolean array of the same shape as `self` that is True
        where an element of `self` is in `test_elements` and False otherwise.

        Parameters
        ----------
        test_elements : array_like
            The values against which to test each value of `element`.
            This argument is flattened if it is an array or array_like.
            See notes for behavior with non-array-like parameters.
        assume_unique : bool, optional
            If True, the input arrays are both assumed to be unique, which
            can speed up the calculation.  Default is False.
        invert : bool, optional
            If True, the values in the returned array are inverted, as if
            calculating `element not in test_elements`. Default is False.
            ``np.isin(a, b, invert=True)`` is equivalent to (but faster
            than) ``np.invert(np.isin(a, b))``.

        Returns
        -------
        isin : ndarray, bool
            Has the same shape as `element`. The values `element[isin]`
            are in `test_elements`.

        Note: behavior differs from pandas
        - Riptable favors bytestrings, and will make conversions from unicode/bytes to match for operations as necessary.
        - We will also accept single scalars for values.
        - Pandas series will return another series - we have no series, and will return a FastArray

        Examples
        --------
        >>> from riptable import *
        >>> a = FA(['a','b','c','d','e'], unicode=False)
        >>> a.isin(['a','b'])
        FastArray([ True,  True, False, False, False])
        >>> a.isin('a')
        FastArray([ True,  False, False, False, False])
        >>> a.isin({'b'})
        FastArray([ False, True, False, False, False])
        """
        if isinstance(test_elements, set):
            test_elements = list(test_elements)

        if not isinstance(test_elements, np.ndarray):
            # align byte string vs unicode
            if self.dtype.char in "SU":
                if np.isscalar(test_elements):
                    test_elements = np.asarray([test_elements], dtype=self.dtype.char)
                else:
                    test_elements = np.asarray(test_elements, dtype=self.dtype.char)
            else:
                if isinstance(test_elements, tuple):
                    raise ValueError(
                        "isin does not currently support tuples.  In the future a tuple will be used to represent a multi-key."
                    )
                test_elements = rc.AsFastArray(test_elements)

        try:
            # optimization: if we have just one element, we can just parallel compare that one element
            if len(test_elements) == 1:
                # string comparison to int will fail
                result = self == test_elements[0]
                # check for failed result
                if np.isscalar(result):
                    result = ismember(self, test_elements)[0]
            else:
                result = ismember(self, test_elements)[0]
            if invert:
                np.logical_not(result, out=result)
            return result
        except Exception:
            # punt non-supported types to numpy
            return np.isin(self._np, test_elements, assume_unique=assume_unique, invert=invert)

    # -------------------------------------------------------------------------
    def between(self, low, high, include_low: bool = True, include_high: bool = False) -> FastArray:
        """
        Determine which elements of the array are in a a given interval.

        Return a boolean mask indicating which elements are between `low` and `high` (including/excluding endpoints
        can be controlled by the `include_low` and `include_high` arguments).

        Default behaviour is equivalent to (self >= low) & (self < high).

        Parameters
        ----------
        low: scalar, array_like
            Lower bound for test interval.  If array, should have the same size as `self` and comparisons are done elementwise.
        high: scalar, array_like
            Upper bound for test interval.  If array, should have the same size as `self` and comparisons are done elementwise.
        include_low: bool
            Should the left endpoint included in the test interval
        include_high: bool
            Should the right endpoint included in the test interval

        Returns
        -------
        array_like[bool]
            An boolean mask indicating if the associated elements are in the test interval
        """
        low = asanyarray(low)
        high = asanyarray(high)

        if include_low:
            ret = self >= low
        else:
            ret = self > low
        if include_high:
            ret &= self <= high
        else:
            ret &= self < high
        return ret

    # --------------------------------------------------------------------------
    def sample(
        self,
        N: int = 10,
        filter: Optional[np.ndarray] = None,
        seed: Optional[Union[int, Sequence[int], np.random.SeedSequence, np.random.Generator]] = None,
    ) -> FastArray:
        """
        Examples
        --------
        >>> a=rt.arange(10)
        >>> a.sample(3)
        FastArray([0, 4, 9])
        """
        return sample(self, N=N, filter=filter, seed=seed)

    # --------------------------------------------------------------------------
    def duplicated(self, keep="first", high_unique=False) -> FastArray:
        """
        See pandas.Series.duplicated

        Duplicated values are indicated as True values in the resulting
        FastArray. Either all duplicates, all except the first or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            - 'first' : Mark duplicates as True except for the first occurrence.
            - 'last' : Mark duplicates as True except for the last occurrence.
            - False : Mark values with just one occurrence as False.

        """
        arr = self

        if keep == "last":
            arr = arr[::-1].copy()

        elif keep is not False and keep != "first":
            raise ValueError(f'keep must be either "first", "last" or False')

        # create an return array all set to True
        result = ones(len(arr), dtype=np.bool_)

        g = Grouping(arr._fa if hasattr(arr, "_fa") else arr, lex=high_unique)

        if keep is False:
            # search for groups with a count of 1
            result[g.ifirstkey[g.ncountgroup[1:] == 1]] = False
        else:
            result[g.ifirstkey] = False

            if keep == "last":
                result = result[::-1].copy()
        return result

    # --------------------------------------------------------------------------
    def save(
        self,
        filepath: Union[str, os.PathLike],
        share: Optional[str] = None,
        compress: bool = True,
        overwrite: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """
        Save a single array in an .sds file.

        Parameters
        ----------
        filepath: str or os.PathLike
        share : str, optional, default None
        compress : bool, default True
        overwrite : bool, default True
        name : str, optional, default None

        See Also
        --------
        rt_sds.save_sds
        """
        save_sds(filepath, self, share=share, compress=compress, overwrite=overwrite, name=name)

    # --------------------------------------------------------------------------
    def reshape(self, *args, **kwargs) -> FastArray:
        result = super(FastArray, self).reshape(*args, **kwargs)
        # this warning happens too much now
        # if FastArray._check_ndim(result) != 1:
        #    warnings.warn(FastArray.WarningDict["multiple_dimensions"].format(result.shape))

        if not (result.flags.c_contiguous or result.flags.f_contiguous):
            # fix strides problem
            return result.copy()
        return result

    # --------------------------------------------------------------------------
    def repeat(self, repeats, axis=None) -> FastArray:
        """see rt.repeat"""
        return repeat(self, repeats, axis=axis)

    # --------------------------------------------------------------------------
    def tile(self, reps) -> FastArray:
        """see rt.tile"""
        return tile(self, reps)

    # --------------------------------------------------------------------------
    def _kwarg_check(self, *args, **kwargs):
        # we handle dtype
        if ("ddof" in kwargs and kwargs["ddof"] != 1) or "axis" in kwargs or "keepdims" in kwargs:
            return True

    # --------------------------------------------------------------------------
    def _reduce_check(self, reduceFunc: REDUCE_FUNCTIONS, npFunc, *args, **kwargs):
        """
        Arg2: npFunc pass in None if no numpy equivalent function
        """
        if npFunc is not None and self._kwarg_check(*args, **kwargs):
            # TODO: add to math ledger
            # set ddof=1 if NOT set which is FastArray default to match matlab/pandas
            if "ddof" not in kwargs and reduceFunc in [
                REDUCE_FUNCTIONS.REDUCE_VAR,
                REDUCE_FUNCTIONS.REDUCE_NANVAR,
                REDUCE_FUNCTIONS.REDUCE_STD,
                REDUCE_FUNCTIONS.REDUCE_NANSTD,
            ]:
                kwargs["ddof"] = 1

            result = npFunc(self._np, *args, **kwargs)
            return result

        result = TypeRegister.MathLedger._REDUCE(self, reduceFunc)

        # It's possible there was no result returned from the reduction function;
        # e.g. if the input was empty. If the function being called is well-defined
        # for empty lists -- i.e. it is a reduction operation with a defined
        # identity element -- set the result to the identity element so the rest of
        # the logic below will work correctly.
        # If there is no identity element for this operation, raise an exception to
        # let the user know; we'd raise an exception below *anyway*, and this allows
        # us to provide the user with a more-descriptive/actionable error message.
        if result is None:
            op_identity_val = type(self)._reduce_op_identity_value.get(reduceFunc, None)
            if op_identity_val is not None:
                result = op_identity_val
            else:
                raise ValueError(
                    f"Reduction '{str(reduceFunc)}' does not have an identity element so cannot be computed over an empty array."
                )

        # Was an output dtype was explicitly specified?
        dtype = kwargs.get("dtype", None)
        if dtype is not None:
            # user forced dtype return value
            return dtype(result)

        # preserve type for min/max/nanmin/nanmax
        if reduceFunc in [
            REDUCE_FUNCTIONS.REDUCE_MIN,
            REDUCE_FUNCTIONS.REDUCE_NANMIN,
            REDUCE_FUNCTIONS.REDUCE_MAX,
            REDUCE_FUNCTIONS.REDUCE_NANMAX,
        ]:
            return self.dtype.type(result)

        # internally numpy expects a dtype returned for nanstd and other calculations
        if isinstance(result, (int, np.integer)):
            # for uint64, the high bit must be preserved
            if self.dtype.char in NumpyCharTypes.UnsignedInteger64:
                return np.uint64(result)
            return np.int64(result)

        return np.float64(result)

    # ---------------------------------------------------------------------------
    def _compare_check(self, func, other) -> FastArray:
        # a user might type in a string and we want a bytes string
        if self.dtype.char in "SU":
            if isinstance(other, str):
                if self.dtype.char == "S":
                    # we are byte strings but scalar unicode passed in
                    other = str.encode(other)

            if isinstance(other, list):
                # convert the list so a comparison can be made to the byte string array
                other = FastArray(other)

            result = func(other)

            # NOTE: numpy does call FA ufunc for strings
            if not isinstance(result, FastArray) and isinstance(result, np.ndarray):
                result = result.view(FastArray)
            return result

        result = func(other)
        return result

    def __ne__(self, other):
        return self._compare_check(super().__ne__, other)

    def __eq__(self, other):
        return self._compare_check(super().__eq__, other)

    def __ge__(self, other):
        return self._compare_check(super().__ge__, other)

    def __gt__(self, other):
        return self._compare_check(super().__gt__, other)

    def __le__(self, other):
        return self._compare_check(super().__le__, other)

    def __lt__(self, other):
        return self._compare_check(super().__lt__, other)

    def eq(self, other):
        return self.__eq__(other)

    def ne(self, other):
        return self.__ne__(other)

    def ge(self, other):
        return self.__ge__(other)

    def le(self, other):
        return self.__le__(other)

    def gt(self, other):
        return self.__gt__(other)

    def lt(self, other):
        return self.__lt__(other)

    add = np.ndarray.__add__
    sub = np.ndarray.__sub__
    mul = np.ndarray.__mul__
    div = np.ndarray.__truediv__
    floordiv = np.ndarray.__floordiv__
    pow = np.ndarray.__pow__
    mod = np.ndarray.__mod__

    # ---------------------------------------------------------------------------
    def str_append(self, other):
        if self.dtype.num == other.dtype.num:
            func = TypeRegister.MathLedger._BASICMATH_TWO_INPUTS
            return func((self, other), MATH_OPERATION.ADD, self.dtype.num)
        raise TypeError("cannot concat")

    # ---------------------------------------------------------------------------
    def squeeze(self, *args, **kwargs):
        return self._np.squeeze(*args, **kwargs)

    # ---------------------------------------------------------------------------
    def iscomputable(self) -> bool:
        return TypeRegister.is_computable(self)

    #############################################
    # nep-18 array function protocol implementation
    #############################################
    @classmethod
    def _py_number_to_np_dtype(
        cls, val: Union[int, np.integer, None], dtype: np.dtype
    ) -> Union[np.uint, np.int64, np.float64, None]:
        """Convert a python type to numpy dtype.
        Only handles integers."""
        if val is not None:
            # internally numpy expects a dtype returned for nanstd and other calculations
            if isinstance(val, (int, np.integer)):
                # for uint64, the high bit must be preserved
                if dtype.char in NumpyCharTypes.UnsignedInteger64:
                    return np.uint64(val)
                return np.int64(val)
            return np.float64(val)
        return val

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.argmax)
    def _argmax(a, axis=None, out=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_ARGMAX, 0)
        return FastArray._py_number_to_np_dtype(result, a.dtype)

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.nanargmax)
    def _nanargmax(a, axis=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_NANARGMAX, 0)
        return FastArray._py_number_to_np_dtype(result, a.dtype)

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.argmin)
    def _argmin(a, axis=None, out=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_ARGMIN, 0)
        return FastArray._py_number_to_np_dtype(result, a.dtype)

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.nanargmin)
    def _nanargmin(a, axis=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_NANARGMIN, 0)
        return FastArray._py_number_to_np_dtype(result, a.dtype)

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.empty_like)
    def _empty_like(
        array: "FastArray",
        dtype: Optional[Union[str, np.dtype]] = None,
        order: str = "K",
        subok: bool = True,
        shape: Optional[Union[int, Sequence[int]]] = None,
    ) -> "FastArray":
        array = array._np
        result = rc.LedgerFunction(np.empty_like, array, dtype=dtype, order=order, subok=subok, shape=shape)
        return result

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.max)
    def _max(a, axis=None, out=None, keepdims=None, initial=None, where=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_MAX, 0)
        if result is not None:
            return a.dtype.type(result)
        return result

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.nanmax)
    def _nanmax(a, axis=None, out=None, keepdims=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_NANMAX, 0)
        if result is not None:
            return a.dtype.type(result)
        return result

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.mean)
    def _mean(a, axis=None, dtype=None, out=None, keepdims=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_MEAN, 0)
        return FastArray._py_number_to_np_dtype(result, a.dtype)

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.nanmean)
    def _nanmean(a, axis=None, dtype=None, out=None, keepdims=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_NANMEAN, 0)
        return FastArray._py_number_to_np_dtype(result, a.dtype)

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.min)
    def _min(a, axis=None, out=None, keepdims=None, initial=None, where=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_MIN, 0)
        if result is not None:
            return a.dtype.type(result)
        return result

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.nanmin)
    def _nanmin(a, axis=None, out=None, keepdims=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_NANMIN, 0)
        if result is not None:
            return a.dtype.type(result)
        return result

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.std)
    def _std(a, axis=None, dtype=None, out=None, ddof=None, keepdims=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_STD, 0)
        return FastArray._py_number_to_np_dtype(result, a.dtype)

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.nanstd)
    def _nanstd(a, axis=None, dtype=None, out=None, ddof=None, keepdims=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_NANSTD, 0)
        return FastArray._py_number_to_np_dtype(result, a.dtype)

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.round)
    @_ArrayFunctionHelper.register_array_function(np.around)
    @_ArrayFunctionHelper.register_array_function(np.round_)  # N.B, round_ is an alias for around
    def _round_(a, decimals=None, out=None):
        # TODO handle `decimal` and `out` arguments
        # If callers decide to use this FastArray staticmethod outside the scope of array function protocol
        # provide argument checks since it may become unclear when things fail at the C extension layer.
        if not isinstance(a, FastArray):
            raise ValueError(f"{FastArray.__name__}._round_ expected FastArray subtype, got {type(a)}")

        original_dtype = a.dtype
        a = a.astype(np.float64)
        fast_function = gUnaryUFuncs.get(np.round, None)
        if fast_function is None:
            raise ValueError(
                f"{FastArray.__name__}._round_ unhandled array function {np.round}\nKnown numpy array function to riptable functions: {repr(gUnaryUFuncs)}"
            )

        # For MATH_OPERATION.ROUND, _BASICMATH_ONE_INPUT returns an array `array(None, dtype=object)`
        # if the input dtype is not a float64. As a workaround cast to float64 dtype, perform the operation,
        # then cast back to the original dtype.
        result = TypeRegister.MathLedger._BASICMATH_ONE_INPUT(a, fast_function, 0)

        if not isinstance(result, FastArray) and isinstance(result, np.ndarray):
            result = result.view(FastArray)

        if result.dtype != original_dtype:
            result = result.astype(original_dtype)

        return result

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.sum)
    def _sum(a, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_SUM, 0)
        return FastArray._py_number_to_np_dtype(result, a.dtype)

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.nansum)
    def _nansum(a, axis=None, dtype=None, out=None, keepdims=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_NANSUM, 0)
        return FastArray._py_number_to_np_dtype(result, a.dtype)

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.var)
    def _var(a, axis=None, dtype=None, out=None, ddof=None, keepdims=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_VAR, 0)
        return FastArray._py_number_to_np_dtype(result, a.dtype)

    @staticmethod
    @_ArrayFunctionHelper.register_array_function(np.nanvar)
    def _nanvar(a, axis=None, dtype=None, out=None, ddof=None, keepdims=None):
        result = rc.Reduce(a, REDUCE_FUNCTIONS.REDUCE_NANVAR, 0)
        return FastArray._py_number_to_np_dtype(result, a.dtype)

    #############################################
    # Helper section
    #############################################
    def abs(self, **kwargs) -> FastArray:
        return np.abs(self, **kwargs)

    def median(self, **kwargs) -> np.number:
        return np.median(self, **kwargs)

    def nanmedian(self, **kwargs) -> np.number:
        return np.nanmedian(self, **kwargs)

    def clip_lower(self, a_min, **kwargs) -> FastArray:
        return self.clip(a_min, None, **kwargs)

    def clip_upper(self, a_max, **kwargs) -> FastArray:
        return self.clip(None, a_max, **kwargs)

    def sign(self, **kwargs) -> FastArray:
        return np.sign(self, **kwargs)

    def trunc(self, **kwargs) -> FastArray:
        return np.trunc(self, **kwargs)

    def where(self, condition, y=np.nan, **kwargs) -> FastArray:
        return where(condition, self, y, **kwargs)

    def count(self, sorted=True) -> Dataset:
        """
        The count of each unique value.

        This returns the same information that ``.unique(return_counts = True)``
        does, except in a `Dataset` instead of a tuple.

        Parameters
        ----------
        sorted : bool, default True
            When True (the default), unique values are returned in sorted order. Set to
            False to return them in order of first appearance.

        Returns
        -------
        Dataset
            A `Dataset` containing the unique values and their counts.

        See Also
        --------
        FastArray.unique

        Examples
        --------
        >>> a = rt.FastArray([0, 2, 1, 3, 3, 2, 2])
        >>> a.count()
        *Unique   Count
        -------   -----
              0       1
              1       1
              2       3
              3       2

        With ``sorted = False``:

        >>> a.count(sorted = False)
        *Unique   Count
        -------   -----
              0       1
              2       3
              1       1
              3       2
        """
        unique_counts = unique(self, sorted=sorted, return_counts=True)
        name = self.get_name()
        if name is None:
            name = "Unique"
        ds = TypeRegister.Dataset({name: unique_counts[0], "Count": unique_counts[1]})
        ds.label_set_names([name])
        return ds

    #############################################
    # Rolling section (cannot handle strides)
    #############################################
    def rolling_sum(self, window: int = 3) -> FastArray:
        return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_SUM, window)

    def rolling_nansum(self, window: int = 3) -> FastArray:
        return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_NANSUM, window)

    def rolling_mean(self, window: int = 3) -> FastArray:
        return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_MEAN, window)

    def rolling_nanmean(self, window: int = 3) -> FastArray:
        return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_NANMEAN, window)

    def rolling_var(self, window: int = 3) -> FastArray:
        return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_VAR, window)

    def rolling_nanvar(self, window: int = 3) -> FastArray:
        return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_NANVAR, window)

    def rolling_std(self, window: int = 3) -> FastArray:
        return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_STD, window)

    def rolling_nanstd(self, window: int = 3) -> FastArray:
        return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_NANSTD, window)

    #############################################
    # TimeWindow section (cannot handle strides), time_array must be INT64
    #############################################
    def timewindow_sum(self, time_array, time_dist):
        """
        The input array must be int64 and sorted with ever increasing values.
        Sums up the values for a given time window.

        Parameters
        ----------
        time_array: sorted integer array of timestamps
        time_dist: integer value of the time window size

        Examples
        --------
        >>> a=rt.arange(10, dtype=rt.int64)
        >>> a.timewindow_sum(a,5)
        FastArray([ 0,  1,  3,  6, 10, 15, 21, 27, 33, 39], dtype=int64)

        """
        return rc.TimeWindow(self, time_array, TIMEWINDOW_FUNCTIONS.TIMEWINDOW_SUM, time_dist)

    def timewindow_prod(self, time_array, time_dist):
        """
        The input array must be int64 and sorted with ever increasing values.
        Multiplies up the values for a given time window.

        Parameters
        ----------
        time_array: sorted integer array of timestamps
        time_dist: integer value of the time window size

        Examples
        --------
        >>> a=rt.arange(10, dtype=rt.int64)
        >>> a.timewindow_prod(a,5)
        FastArray([    0,     0,     0,     0,     0,     0,   720,  5040, 20160, 60480], dtype=int64)
        """
        return rc.TimeWindow(self, time_array, TIMEWINDOW_FUNCTIONS.TIMEWINDOW_PROD, time_dist)

    #############################################
    # Bottleneck section (only handles int32/int64/float32/float64)
    # bottleneck is optional
    #############################################
    def move_sum(self, *args, **kwargs):
        return bn.move_sum(self, *args, **kwargs)

    def move_mean(self, *args, **kwargs):
        return bn.move_mean(self, *args, **kwargs)

    def move_std(self, *args, **kwargs):
        return bn.move_std(self, *args, **kwargs)

    def move_var(self, *args, **kwargs):
        return bn.move_var(self, *args, **kwargs)

    def move_min(self, *args, **kwargs):
        return bn.move_min(self, *args, **kwargs)

    def move_max(self, *args, **kwargs):
        return bn.move_max(self, *args, **kwargs)

    def move_argmin(self, *args, **kwargs):
        return bn.move_argmin(self, *args, **kwargs)

    def move_argmax(self, *args, **kwargs):
        return bn.move_argmax(self, *args, **kwargs)

    def move_median(self, *args, **kwargs):
        return bn.move_median(self, *args, **kwargs)

    def move_rank(self, *args, **kwargs):
        return bn.move_rank(self, *args, **kwargs)

    # ---------------------------------------------------------------------------
    def replace(self, old, new):
        return bn.replace(self, old, new)

    def partition2(self, *args, **kwargs):
        return bn.partition(self, *args, **kwargs)

    def argpartition2(self, *args, **kwargs):
        return bn.argpartition(self, *args, **kwargs)

    def rankdata(self, *args, **kwargs):
        return bn.rankdata(self, *args, **kwargs)

    def nanrankdata(self, *args, **kwargs):
        return bn.nanrankdata(self, *args, **kwargs)

    def push(self, *args, **kwargs):
        return bn.push(self, *args, **kwargs)

    # ---------------------------------------------------------------------------
    def issorted(self) -> bool:
        """returns True if the array is sorted otherwise False
        If the data is likely to be sorted, call the issorted property to check.
        """
        return issorted(self)

    # ---------------------------------------------------------------------------
    def _unary_op(self, funcnum, fancy=False) -> FastArray:
        if self._is_not_supported(self):
            # make it contiguous
            arr = self.copy()
        else:
            arr = self

        func = TypeRegister.MathLedger._BASICMATH_ONE_INPUT
        result = func(arr, funcnum, 0)

        if result is None:
            raise TypeError(f"Could not perform operation {funcnum} on FastArray of dtype {arr.dtype}")
        if fancy:
            result = bool_to_fancy(result)
        return result

    #############################################
    # Boolean section
    #############################################
    def isnotfinite(self, fancy=False):
        return self._unary_op(MATH_OPERATION.ISNOTFINITE, fancy=fancy)

    def isinf(self, fancy=False):
        return self._unary_op(MATH_OPERATION.ISINF, fancy=fancy)

    def isnotinf(self, fancy=False):
        return self._unary_op(MATH_OPERATION.ISNOTINF, fancy=fancy)

    def isnormal(self, fancy=False):
        return self._unary_op(MATH_OPERATION.ISNORMAL, fancy=fancy)

    def isnotnormal(self, fancy=False):
        return self._unary_op(MATH_OPERATION.ISNOTNORMAL, fancy=fancy)

    def isnan(self, fancy=False):
        """
        Return a boolean array that's True for each element that's a NaN (Not a Number),
        False otherwise.

        Parameters
        ----------
        fancy : bool, default False
            Set to True to instead return the indices of the True (NaN) values.

        Returns
        -------
        `FastArray`
            A `FastArray` of booleans or indices.

        See Also
        --------
        FastArray.isnotnan, FastArray.notna, FastArray.isnanorzero, riptable.isnan,
        riptable.isnotnan, riptable.isnanorzero, Categorical.isnan,
        Categorical.isnotnan, Categorical.notna, Date.isnan, Date.isnotnan,
        DateTimeNano.isnan, DateTimeNano.isnotnan
        Dataset.mask_or_isnan :
            Return a boolean array that's True for each `Dataset` row that contains
            at least one NaN.
        Dataset.mask_and_isnan :
            Return a boolean array that's True for each all-NaN `Dataset` row.

        Examples
        --------
        >>> a = rt.FastArray([rt.nan, rt.nan, np.inf, 3])
        >>> a.isnan()
        FastArray([ True,  True, False, False])

        With ``fancy = True``:

        >>> a.isnan(fancy = True)
        FastArray([0, 1])
        """
        return self._unary_op(MATH_OPERATION.ISNAN, fancy=fancy)

    def isnotnan(self, fancy=False):
        """
        Return a boolean array that's True for each element that's not a NaN (Not a
        Number), False otherwise.

        Parameters
        ----------
        fancy : bool, default False
            Set to True to instead return the indices of the True (non-NaN) values.

        Returns
        -------
        `FastArray`
            A `FastArray` of booleans or indices.

        See Also
        --------
        FastArray.isnan, FastArray.notna, FastArray.isnanorzero, riptable.isnan,
        riptable.isnotnan, riptable.isnanorzero, Categorical.isnan,
        Categorical.isnotnan, Categorical.notna, Date.isnan, Date.isnotnan,
        DateTimeNano.isnan, DateTimeNano.isnotnan
        Dataset.mask_or_isnan :
            Return a boolean array that's True for each `Dataset` row that contains
            at least one NaN.
        Dataset.mask_and_isnan :
            Return a boolean array that's True for each all-NaN `Dataset` row.

        Examples
        --------
        >>> a = rt.FastArray([rt.nan, np.inf, 2])
        >>> a.isnotnan()
        FastArray([False,  True,  True])

        With ``fancy = True``:

        >>> a.isnotnan(fancy = True)
        FastArray([1, 2])
        """
        return self._unary_op(MATH_OPERATION.ISNOTNAN, fancy=fancy)

    def isnanorzero(self, fancy=False):
        """
        Return a boolean array that's True for each element that's a NaN (Not a Number)
        or zero, False otherwise.

        Parameters
        ----------
        fancy : bool, default False
            Set to True to instead return the indices of the True (NaN or zero) values.

        Returns
        -------
        `FastArray`
            A `FastArray` of booleans or indices.

        See Also
        --------
        riptable.isnanorzero, riptable.isnan, riptable.isnotnan, FastArray.isnan,
        FastArray.isnotnan, Categorical.isnan, Categorical.isnotnan, Date.isnan,
        Date.isnotnan, DateTimeNano.isnan, DateTimeNano.isnotnan
        Dataset.mask_or_isnan :
            Return a boolean array that's True for each `Dataset` row that contains
            at least one NaN.
        Dataset.mask_and_isnan :
            Return a boolean array that's True for each all-NaN `Dataset` row.

        Examples
        --------
        >>> a = rt.FastArray([0, rt.nan, np.inf, 3])
        >>> a.isnanorzero()
        FastArray([ True,  True, False, False])

        With ``fancy = True``:

        >>> a.isnanorzero(fancy = True)
        FastArray([0, 1])
        """
        return self._unary_op(MATH_OPERATION.ISNANORZERO, fancy=fancy)

    def isfinite(self, fancy=False):
        """
        Return a boolean array that's True for each finite `FastArray` element, False
        otherwise.

        A value is considered to be finite if it's not positive or negative infinity
        or a NaN (Not a Number).

        Parameters
        ----------
        fancy : bool, default False
            Set to True to instead return the indices of the True (finite) values.

        Returns
        -------
        `FastArray`
            An array or booleans or indices.

        See Also
        --------
        FastArray.isnotfinite, riptable.isfinite, riptable.isnotfinite, riptable.isinf,
        riptable.isnotinf, FastArray.isinf, FastArray.isnotinf
        Dataset.mask_or_isfinite :
            Return a boolean array that's True for each `Dataset` row that has at least
            one finite value.
        Dataset.mask_and_isfinite :
            Return a boolean array that's True for each `Dataset` row that contains all
            finite values.
        Dataset.mask_or_isinf :
            Return a boolean array that's True for each `Dataset` row that has at least
            one value that's positive or negative infinity.
        Dataset.mask_and_isinf :
            Return a boolean array that's True for each `Dataset` row that contains all
            infinite values.

        Examples
        --------
        >>> a = rt.FastArray([np.inf, np.NINF, rt.nan, 0])
        >>> a.isfinite()
        FastArray([False, False, False,  True])

        With ``fancy = True``:

        >>> a.isfinite(fancy = True)
        FastArray([3])
        """
        return self._unary_op(MATH_OPERATION.ISFINITE, fancy=fancy)

    #############################################
    # Reduce section
    #############################################

    def _fa_filter_wrapper(self, myFunc, filter=None, dtype=None):

        if filter is True:
            filter = ones(len(self), dtype=bool)
        if filter is False:
            filter = zeros(len(self), dtype=bool)
        if len(filter) != len(self):
            raise ValueError("Filter and input not the same length.")

        if not self.iscomputable():
            return np.NaN

        if dtype is not None:
            return dtype(myFunc(self, filter))

        return myFunc(self, filter)

    def _fa_keyword_wrapper(self, filter=None, dtype=None, axis=None, keepdims=None, ddof=None, **kwargs):

        if self.dtype.char in "OSU":
            raise TypeError("FastArray operation applied to string or object array.")

        if "out" in kwargs:
            if kwargs["out"] is None:
                kwargs.pop("out")

        if any(kwargs):
            logging.warning(
                "Unexpected FastArray operation keyword(s): " + ", ".join([key for key, value in kwargs.items()])
            )

        if dtype is not None:
            kwargs["dtype"] = dtype
        if axis:
            kwargs["axis"] = axis
        if keepdims:
            kwargs["keepdims"] = keepdims
        if ddof is not None:
            kwargs["ddof"] = ddof
        if filter is not None:
            kwargs["filter"] = filter

        if (filter is not None) and ((axis is not None) or (keepdims is not None) or (ddof is not None)):
            logging.warning("Since Filter keyword is present, FastArray operations ignore axis, keepdims and ddof")

        return kwargs

    def nansum(self, filter=None, dtype=None, axis=None, keepdims=None, **kwargs) -> np.number:
        """
        Compute the sum of the values in the first argument, ignoring NaNs.

        If all values in the first argument are NaNs, ``0.0`` is returned.

        Parameters
        ----------
        filter : array of bool, default None
            Specifies which elements to include in the sum calculation. If the filter is
            uniformly ``False``, `nansum` returns ``0.0``.
        dtype : rt.dtype or numpy.dtype, default float64
            The data type of the result. For a `FastArray` ``x``,
            ``x.nansum(dtype = my_type)`` is equivalent to ``my_type(x.nansum())``.

        Returns
        -------
        scalar
            The sum of the values.

        See Also
        --------
        numpy.nansum
        Dataset.nansum : Sums the values of numerical `Dataset` columns, ignoring NaNs.
        GroupByOps.nansum : Sums the values of each group, ignoring NaNs. Used by
                            `Categorical` objects.

        Notes
        -----
        The `dtype` keyword for `FastArray.nansum` specifies the data type of the
        result. This differs from `numpy.nansum`, where it specifies the data type used
        to compute the sum.

        **Notes on Using NumPy Parameters**

        Using either of the following NumPy parameters will cause Riptable to switch to
        the NumPy implementation of this method (`numpy.nansum`). However, until a
        reported bug is fixed, if you also include the `dtype` parameter it will be
        applied to the result, not used to compute the sum as it is in `numpy.nansum`.

        Also note that if you use either of the following NumPy parameters and also
        include a `filter` keyword argument (which `numpy.nansum` does not accept),
        Riptable's implementation of `nansum` will be used with the filter argument and
        the NumPy parameters will be ignored.

        axis : {int, tuple of int, None}, optional
            Axis or axes along which the sum is computed. The default is to compute the
            sum of the flattened array.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the original input array.

            If the value is anything but the default, then `keepdims` will be passed
            through to the `mean` or `sum` methods of sub-classes of `ndarray`. If the
            sub-classes' methods do not implement `keepdims`, any exceptions will be
            raised.

        Examples
        --------
        >>> a = rt.FastArray([1, 3, 5, 7, rt.nan])
        >>> a.nansum()
        16.0

        With a `dtype` specified:

        >>> a = rt.FastArray([1.0, 3.0, 5.0, 7.0, rt.nan])
        >>> a.nansum(dtype = rt.int32)
        16

        With a filter:

        >>> a = rt.FastArray([1, 3, 5, 7, rt.nan])
        >>> b = rt.FastArray([False, True, False, True, True])
        >>> a.nansum(filter = b)
        10.0
        """

        kwargs = self._fa_keyword_wrapper(filter=filter, dtype=dtype, axis=axis, keepdims=keepdims, ddof=None, **kwargs)

        if filter is not None:
            return self._fa_filter_wrapper(_fnansum, filter=filter, dtype=dtype)

        return self._reduce_check(REDUCE_FUNCTIONS.REDUCE_NANSUM, np.nansum, **kwargs)

    def mean(self, filter=None, dtype=None, axis=None, keepdims=None, **kwargs) -> np.number:
        """
        Compute the arithmetic mean of the values in the first argument.

        Parameters
        ----------
        filter : array of bool, default None
            Specifies which elements to include in the mean calculation. If the filter
            is uniformly ``False``, `mean` returns a `ZeroDivisionError`.
        dtype : rt.dtype or numpy.dtype, default float64
            The data type of the result. For a `FastArray` ``x``,
            ``x.mean(dtype = my_type)`` is equivalent to ``my_type(x.mean())``.

        Returns
        -------
        scalar
            The mean of the values.

        See Also
        --------
        numpy.mean
        FastArray.nanmean : Computes the mean of `FastArray` values, ignoring NaNs.
        Dataset.mean : Computes the mean of numerical `Dataset` columns.
        GroupByOps.mean : Computes the mean of each group. Used by `Categorical` objects.

        Notes
        -----
        The `dtype` keyword for `FastArray.mean` specifies the data type of the result.
        This differs from `numpy.mean`, where it specifies the data type used to compute
        the mean.

        **Notes on Using NumPy Parameters**

        Using either of the following NumPy parameters will cause Riptable to switch to
        the NumPy implementation of this method (`numpy.mean`). However, until a
        reported bug is fixed, if you also include the `dtype` parameter it will be
        applied to the result, not used to compute the mean as it is in `numpy.mean`.

        Also note that if you use either of the following NumPy parameters and also
        include a `filter` keyword argument (which `numpy.mean` does not accept),
        Riptable's implementation of `mean` will be used with the filter argument and
        the NumPy parameters will be ignored.

        axis : None or int or tuple of ints, optional
            Axis or axes along which the means are computed. The default is to compute
            the mean of the flattened array.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the original input array.

            If the default value is passed, then `keepdims` will not be passed through
            to the `mean` method of sub-classes of `ndarray`, however any non-default
            value will be. If the sub-class's method does not implement `keepdims`, any
            exceptions will be raised.

        Examples
        --------
        >>> a = rt.FastArray([1, 3, 5, 7])
        >>> a.mean()
        4.0

        With a `dtype` specified:

        >>> a = rt.FastArray([1, 3, 5, 7])
        >>> a.mean(dtype = rt.int32)
        4

        With a filter:

        >>> a = rt.FastArray([1, 3, 5, 7])
        >>> b = rt.FastArray([False, True, False, True])
        >>> a.mean(filter = b)
        5.0
        """

        kwargs = self._fa_keyword_wrapper(filter=filter, dtype=dtype, axis=axis, keepdims=keepdims, ddof=None, **kwargs)

        if filter is not None:
            return self._fa_filter_wrapper(_fmean, filter=filter, dtype=dtype)

        return self._reduce_check(REDUCE_FUNCTIONS.REDUCE_MEAN, np.mean, **kwargs)

    def nanmean(self, filter=None, dtype=None, axis=None, keepdims=None, **kwargs) -> np.number:
        """
        Compute the arithmetic mean of the values in the first argument, ignoring NaNs.

        If all values in the first argument are NaNs, ``0.0`` is returned.

        Parameters
        ----------
        filter : array of bool, default None
            Specifies which elements to include in the mean calculation. If the filter
            is uniformly ``False``, `nanmean` returns a `ZeroDivisionError`.
        dtype : rt.dtype or numpy.dtype, default float64
            The data type of the result. For a `FastArray` ``x``,
            ``x.nanmean(dtype = my_type)`` is equivalent to ``my_type(x.nanmean())``.

        Returns
        -------
        scalar
            The mean of the values.

        See Also
        --------
        numpy.nanmean
        FastArray.mean : Computes the mean of `FastArray` values.
        Dataset.nanmean : Computes the mean of numerical `Dataset` columns, ignoring
                          NaNs.
        GroupByOps.nanmean : Computes the mean of each group, ignoring NaNs. Used by
                             `Categorical` objects.

        Notes
        -----
        The `dtype` keyword for `FastArray.nanmean` specifies the data type of the
        result. This differs from `numpy.nanmean`, where it specifies the data type used
        to compute the mean.

        **Notes on Using NumPy Parameters**

        Using either of the following NumPy parameters will cause Riptable to switch to
        the NumPy implementation of this method (`numpy.nanmean`). However, until a
        reported bug is fixed, if you also include the `dtype` parameter it will be
        applied to the result, not used to compute the mean as it is in `numpy.nanmean`.

        Also note that if you use either of the following NumPy parameters and also
        include a `filter` keyword argument (which `numpy.nanmean` does not accept),
        Riptable's implementation of `nanmean` will be used with the filter argument
        and the NumPy parameters will be ignored.

        axis : {int, tuple of int, None}, optional
            Axis or axes along which the means are computed. The default is to compute
            the mean of the flattened array.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the original input array.

            If the value is anything but the default, then `keepdims` will be passed
            through to the `mean` or `sum` methods of sub-classes of `ndarray`. If the
            sub-classes' methods do not implement `keepdims`, any exceptions will be
            raised.

        Examples
        --------
        >>> a = rt.FastArray([1, 3, 5, rt.nan])
        >>> a.nanmean()
        3.0

        With a `dtype` specified:

        >>> a = rt.FastArray([1, 3, 5, rt.nan])
        >>> a.nanmean(dtype = rt.int32)
        3

        With a filter:

        >>> a = rt.FastArray([1, 3, 5, rt.nan])
        >>> b = rt.FastArray([False, True, True, True])
        >>> a.nanmean(filter = b)
        4.0
        """

        kwargs = self._fa_keyword_wrapper(filter=filter, dtype=dtype, axis=axis, keepdims=keepdims, ddof=None, **kwargs)

        if filter is not None:
            return self._fa_filter_wrapper(_fnanmean, filter=filter, dtype=dtype)

        return self._reduce_check(REDUCE_FUNCTIONS.REDUCE_NANMEAN, np.nanmean, **kwargs)

    # ---------------------------------------------------------------------------
    # these function take a ddof kwarg
    def var(self, filter=None, dtype=None, axis=None, keepdims=None, ddof=None, **kwargs):
        """
        Compute the variance of the values in the first argument.

        Riptable uses the convention that ``ddof = 1``, meaning the variance of
        ``[x_1, ..., x_n]`` is defined by ``var = 1/(n - 1) * sum(x_i - mean )**2``
        (note the ``n - 1`` instead of ``n``). This differs from NumPy, which uses
        ``ddof = 0`` by default.

        Parameters
        ----------
        filter : array of bool, default None
            Specifies which elements to include in the variance calculation. If the
            filter is uniformly ``False``, `var` returns a `ZeroDivisionError`.

        dtype : rt.dtype or numpy.dtype, default float64
            The data type of the result. For a `FastArray` ``x``,
            ``x.var(dtype = my_type)`` is equivalent to ``my_type(x.var())``.

        Returns
        -------
        scalar
            The variance of the values.

        See Also
        --------
        numpy.var
        FastArray.nanvar : Computes the variance of `FastArray` values, ignoring NaNs.
        Dataset.var : Computes the variance of numerical `Dataset` columns.
        GroupByOps.var : Computes the variance of each group. Used by `Categorical`
                         objects.

        Notes
        -----
        The `dtype` keyword for `FastArray.var` specifies the data type of the result.
        This differs from `numpy.var`, where it specifies the data type used to compute
        the variance.

        **Notes on Using NumPy Parameters**

        Using any of the following NumPy parameters will cause Riptable to switch to
        the NumPy implementation of this method (`numpy.var`). However, until a
        reported bug is fixed, if you also include the `dtype` parameter it will be
        applied to the result, not used to compute the variance as it is in `numpy.var`.

        Also note that if you use any of the following NumPy parameters and also
        include a `filter` keyword argument (which `numpy.var` does not accept),
        Riptable's implementation of `var` will be used with the filter argument
        and the NumPy parameters will be ignored.

        axis : None or int or tuple of ints, optional
            Axis or axes along which the variance is computed. The default is to
            compute the variance of the flattened array.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the input array.

            If the default value is passed, then `keepdims` will not be passed through
            to the `var` method of sub-classes of `ndarray`, however any non-default
            value will be. If the sub-classes' method does not implement `keepdims`, any
            exceptions will be raised.

        ddof : int, optional
            "Delta Degrees of Freedom": the divisor used in the calculation is
            ``N - ddof``, where ``N`` represents the number of elements. By default
            `ddof` is zero for the NumPy implementation, versus one for the Riptable
            implementation.

        Examples
        --------
        >>> a = rt.FastArray([1, 2, 3])
        >>> a.var()
        1.0

        With a `dtype` specified:

        >>> a = rt.FastArray([1, 2, 3])
        >>> a.var(dtype = rt.int32)
        1

        With a filter:

        >>> a = rt.FastArray([1, 2, 3])
        >>> b = rt.FastArray([False, True, True])
        >>> a.var(filter = b)
        0.5
        """

        kwargs = self._fa_keyword_wrapper(filter=filter, dtype=dtype, axis=axis, keepdims=keepdims, ddof=ddof, **kwargs)

        if filter is not None:
            return self._fa_filter_wrapper(_fvar, filter=filter, dtype=dtype)

        return self._reduce_check(REDUCE_FUNCTIONS.REDUCE_VAR, np.var, **kwargs)

    def nanvar(self, filter=None, dtype=None, axis=None, keepdims=None, ddof=None, **kwargs) -> np.number:
        """
        Compute the variance of the values in the first argument, ignoring NaNs.

        If all values in the first argument are NaNs, ``NaN`` is returned.

        Riptable uses the convention that ``ddof = 1``, meaning the variance of
        ``[x_1, ..., x_n]`` is defined by ``var = 1/(n - 1) * sum(x_i - mean )**2`` (note
        the ``n - 1`` instead of ``n``). This differs from NumPy, which uses ``ddof = 0`` by
        default.

        Parameters
        ----------
        filter : array of bool, default None
            Specifies which elements to include in the variance calculation. If the filter
            is uniformly ``False``, `nanvar` returns a `ZeroDivisionError`.

        dtype : rt.dtype or numpy.dtype, default float64
            The data type of the result. For a `FastArray` ``x``,
            ``x.nanvar(dtype = my_type)`` is equivalent to ``my_type(x.nanvar())``.

        Returns
        -------
        scalar
            The variance of the values.

        See Also
        --------
        numpy.nanvar
        FastArray.var : Computes the variance of `FastArray` values.
        Dataset.nanvar : Computes the variance of numerical `Dataset` columns,
                         ignoring NaNs.
        GroupByOps.nanvar : Computes the variance of each group, ignoring NaNs. Used by
                            `Categorical` objects.

        Notes
        -----
        The `dtype` keyword for `FastArray.nanvar` specifies the data type of the
        result. This differs from `numpy.nanvar`, where it specifies the data type used
        to compute the variance.

        **Notes on Using NumPy Parameters**

        Using any of the following NumPy parameters will cause Riptable to switch to
        the NumPy implementation of this method (`numpy.nanvar`). However, until a
        reported bug is fixed, if you also include the `dtype` parameter it will be
        applied to the result, not used to compute the variance as it is in
        `numpy.nanvar`.

        Also note that if you use any of the following NumPy parameters and also
        include a `filter` keyword argument (which `numpy.nanvar` does not accept),
        Riptable's implementation of `nanvar` will be used with the filter argument
        and the NumPy parameters will be ignored.

        axis : {int, tuple of int, None}, optional
            Axis or axes along which the variance is computed. The default is to
            compute the variance of the flattened array.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the original input array.

        ddof : int, optional
            "Delta Degrees of Freedom": the divisor used in the calculation is
            ``N - ddof``, where ``N`` represents the number of non-NaN elements. By
            default `ddof` is zero for the NumPy implementation, versus one for the
            Riptable implementation.

        Examples
        --------
        >>> a = rt.FastArray([1, 2, 3, rt.nan])
        >>> a.nanvar()
        1.0

        With a `dtype` specified:

        >>> a = rt.FastArray([1, 2, 3, rt.nan])
        >>> a.nanvar(dtype = rt.int32)
        1

        With a filter:

        >>> a = rt.FastArray([1, 2, 3, rt.nan])
        >>> b = rt.FastArray([False, True, True, True])
        >>> a.nanvar(filter = b)
        0.5
        """

        kwargs = self._fa_keyword_wrapper(filter=filter, dtype=dtype, axis=axis, keepdims=keepdims, ddof=ddof, **kwargs)

        if filter is not None:
            return self._fa_filter_wrapper(_fnanvar, filter=filter, dtype=dtype)

        return self._reduce_check(REDUCE_FUNCTIONS.REDUCE_NANVAR, np.nanvar, **kwargs)

    def std(self, filter=None, dtype=None, axis=None, keepdims=None, ddof=None, **kwargs) -> np.number:
        """
        Compute the standard deviation of the values in the first argument.

        Riptable uses the convention that ``ddof = 1``, meaning the standard deviation of
        ``[x_1, ..., x_n]`` is defined by ``std = 1/(n - 1) * sum(x_i - mean )**2`` (note
        the ``n - 1`` instead of ``n``). This differs from NumPy, which uses ``ddof = 0`` by
        default.

        Parameters
        ----------
        filter : array of bool, default None
            Specifies which elements to include in the standard deviation calculation. If
            the filter is uniformly ``False``, `std` returns a `ZeroDivisionError`.

        dtype : rt.dtype or numpy.dtype, default float64
            The data type of the result. For a `FastArray` ``x``,
            ``x.std(dtype = my_type)`` is equivalent to ``my_type(x.std())``.

        Returns
        -------
        scalar
            The standard deviation of the values.

        See Also
        --------
        numpy.std
        FastArray.nanstd : Computes the standard deviation of `FastArray` values, ignoring
                           NaNs.
        Dataset.std : Computes the standard deviation of numerical `Dataset` columns.
        GroupByOps.std : Computes the standard deviation of each group. Used by
                         `Categorical` objects.

        Notes
        -----
        The `dtype` keyword for `FastArray.std` specifies the data type of the result.
        This differs from `numpy.std`, where it specifies the data type used to compute
        the standard deviation.

        **Notes on Using NumPy Parameters**

        Using any of the following NumPy parameters will cause Riptable to switch to
        the NumPy implementation of this method (`numpy.std`). However, until a
        reported bug is fixed, if you also include the `dtype` parameter it will be
        applied to the result, not used to compute the variance as it is in
        `numpy.std`.

        Also note that if you use any of the following NumPy parameters and also
        include a `filter` keyword argument (which `numpy.std` does not accept),
        Riptable's implementation of `std` will be used with the filter argument
        and the NumPy parameters will be ignored.

        axis : None or int or tuple of ints, optional
            Axis or axes along which the standard deviation is computed. The
            default is to compute the standard deviation of the flattened array.

            .. versionadded:: 1.7.0

            If this is a tuple of ints, a standard deviation is performed over multiple
            axes, instead of a single axis or all the axes as before.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the input array.

            If the default value is passed, then `keepdims` will not be passed through
            to the `std` method of sub-classes of `ndarray`, however any non-default
            value will be. If the sub-class' method does not implement `keepdims`,
            any exceptions will be raised.

        ddof : int, optional
            "Delta Degrees of Freedom": the divisor used in the calculation is
            ``N - ddof``, where ``N`` represents the number of elements. By default
            `ddof` is zero for the NumPy implementation, versus one for the
            Riptable implementation.

        Examples
        --------
        >>> a = rt.FastArray([1, 2, 3])
        >>> a.std()
        1.0

        With a `dtype` specified:

        >>> a = rt.FastArray([1, 2, 3])
        >>> a.std(dtype = rt.int32)
        1

        With a filter:

        >>> a = rt.FastArray([1, 2, 3])
        >>> b = rt.FA([False, True, True])
        >>> a.std(filter = b)
        0.7071067811865476
        """

        kwargs = self._fa_keyword_wrapper(filter=filter, dtype=dtype, axis=axis, keepdims=keepdims, ddof=ddof, **kwargs)

        if filter is not None:
            return self._fa_filter_wrapper(_fstd, filter=filter, dtype=dtype)

        return self._reduce_check(REDUCE_FUNCTIONS.REDUCE_STD, np.std, **kwargs)

    def nanstd(self, filter=None, dtype=None, axis=None, keepdims=None, ddof=None, **kwargs) -> np.number:
        """
        Compute the standard deviation of the values in the first argument, ignoring NaNs.

        If all values in the first argument are NaNs, ``NaN`` is returned.

        Riptable uses the convention that ``ddof = 1``, meaning the standard deviation of
        ``[x_1, ..., x_n]`` is defined by ``std = 1/(n - 1) * sum(x_i - mean )**2`` (note
        the ``n - 1`` instead of ``n``). This differs from NumPy, which uses ``ddof = 0`` by
        default.

        Parameters
        ----------
        filter : array of bool, default None
            Specifies which elements to include in the standard deviation calculation. If
            the filter is uniformly ``False``, `nanstd` returns a `ZeroDivisionError`.

        dtype : rt.dtype or numpy.dtype, default float64
            The data type of the result. For a `FastArray` ``x``,
            ``x.nanstd(dtype = my_type)`` is equivalent to ``my_type(x.nanstd())``.

        Returns
        -------
        scalar
            The standard deviation of the values.

        See Also
        --------
        numpy.nanstd
        FastArray.std : Computes the standard deviation of `FastArray` values.
        Dataset.nanstd : Computes the standard deviation of numerical `Dataset` columns,
                         ignoring NaNs.
        GroupByOps.nanstd : Computes the standard deviation of each group, ignoring NaNs.
                            Used by `Categorical` objects.

        Notes
        -----
        The `dtype` keyword for `FastArray.nanstd` specifies the data type of the
        result. This differs from `numpy.nanstd`, where it specifies the data type used
        to compute the standard deviation.

        **Notes on Using NumPy Parameters**

        Using any of the following NumPy parameters will cause Riptable to switch to
        the NumPy implementation of this method (`numpy.nanstd`). However, until a
        reported bug is fixed, if you also include the `dtype` parameter it will be
        applied to the result, not used to compute the variance as it is in
        `numpy.nanstd`.

        Also note that if you use any of the following NumPy parameters and also
        include a `filter` keyword argument (which `numpy.nanstd` does not accept),
        Riptable's implementation of `nanstd` will be used with the filter argument
        and the NumPy parameters will be ignored.

        axis : {int, tuple of int, None}, optional
            Axis or axes along which the standard deviation is computed. The default is
            to compute the standard deviation of the flattened array.

        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the original input array.

            If this value is anything but the default it is passed through as-is to the
            relevant functions of the sub-classes. If these functions do not have a
            `keepdims` kwarg, a RuntimeError will be raised.

        ddof : int, optional
            "Delta Degrees of Freedom": the divisor used in the calculation is
            ``N - ddof``, where ``N`` represents the number of elements. By default
            `ddof` is zero for the NumPy implementation, versus one for the
            Riptable implementation.

        Examples
        --------
        >>> a = rt.FastArray([1, 2, 3, rt.nan])
        >>> a.nanstd()
        1.0

        With a `dtype` specified:

        >>> a = rt.FastArray([1, 2, 3, rt.nan])
        >>> a.nanstd(dtype = rt.int32)
        1

        With filter:

        >>> a = rt.FastArray([1, 2, 3, rt.nan])
        >>> b = rt.FastArray([False, True, True, True])
        >>> a.nanstd(filter = b)
        0.7071067811865476
        """
        kwargs = self._fa_keyword_wrapper(filter=filter, dtype=dtype, axis=axis, keepdims=keepdims, ddof=ddof, **kwargs)

        if filter is not None:
            return self._fa_filter_wrapper(_fnanstd, filter=filter, dtype=dtype)

        return self._reduce_check(REDUCE_FUNCTIONS.REDUCE_NANSTD, np.nanstd, **kwargs)

    # ---------------------------------------------------------------------------
    def nanmin(self, **kwargs) -> np.number:
        return self._reduce_check(REDUCE_FUNCTIONS.REDUCE_NANMIN, np.nanmin, **kwargs)

    def nanmax(self, **kwargs) -> np.number:
        return self._reduce_check(REDUCE_FUNCTIONS.REDUCE_NANMAX, np.nanmax, **kwargs)

    # ---------------------------------------------------------------------------
    def argmin(self, **kwargs) -> int:
        return self._reduce_check(REDUCE_FUNCTIONS.REDUCE_ARGMIN, np.argmin, **kwargs)

    def argmax(self, **kwargs) -> int:
        return self._reduce_check(REDUCE_FUNCTIONS.REDUCE_ARGMAX, np.argmax, **kwargs)

    def nanargmin(self, **kwargs) -> int:
        return self._reduce_check(REDUCE_FUNCTIONS.REDUCE_NANARGMIN, np.nanargmin, **kwargs)

    def nanargmax(self, **kwargs) -> int:
        return self._reduce_check(REDUCE_FUNCTIONS.REDUCE_NANARGMAX, np.nanargmax, **kwargs)

    #############################################
    # Stats/ML section
    #############################################
    def normalize_zscore(self) -> FastArray:
        return normalize_zscore(self)

    def normalize_minmax(self) -> FastArray:
        return normalize_minmax(self)

    #############################################
    # BasicMath section (to be hooked at C level now)
    #############################################
    # def __add__(self, value):   result=rc.BasicMathTwoInputs((self, value), 1, 0); result= result if result is not None else np.add(self,value); return result
    # def __add__(self, value):   return rc.BasicMathTwoInputs((self, value), 1, 0)

    @property
    def crc(self) -> int:
        """
        Calculate the 32-bit CRC of the data in this array using the Castagnoli polynomial (CRC32C).

        This function does not consider the array's shape or strides when calculating the CRC,
        it simply calculates the CRC value over the entire buffer described by the array.

        Examples
        --------

        can be used to compare two arrays for structural equality
        >>> a = arange(100)
        >>> b = arange(100.0)
        >>> a.crc == b.crc
        False
        """
        return crc32c(self)

    # todo: range/nanrange
    # todo: stats/nanstats

    # -------------------------------------------------------
    def unique(
        self,
        return_index: bool = False,
        return_inverse: bool = False,
        return_counts: bool = False,
        sorted: bool = True,
        lex: bool = False,
        dtype: Optional[Union[str, np.dtype]] = None,
        filter: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Union["FastArray", Tuple["FastArray", ...], List["FastArray"], tuple]:
        """
        Find the unique elements of an array or the unique combinations of elements with
        corresponding indices in multiple arrays.

        See :meth:`riptable.unique` for full documentation.
        """
        return unique(
            self,
            return_index=return_index,
            return_inverse=return_inverse,
            return_counts=return_counts,
            sorted=sorted,
            lex=lex,
            dtype=dtype,
            filter=filter,
            **kwargs,
        )

    # -------------------------------------------------------
    def nunique(self) -> int:
        """
        Returns number of unique values in array. Does not include nan or sentinel values in the count.

        Examples
        --------

        Float with nan:

        >>> a = FastArray([1.,2.,3.,np.nan])
        >>> a.nunique()
        3

        Signed integer with sentinel:

        >>> a = FastArray([-128, 2, 3], dtype=np.int8)
        >>> a.nunique()
        2

        Unsigned integer with sentinel:

        >>> a = FastArray([255, 2, 3], dtype=np.uint8)
        >>> a.nunique()
        2

        """
        un = unique(self)
        count = len(un)
        if count > 0:
            # unique is sorted, so check for sentinel based on dtype
            inv = INVALID_DICT[self.dtype.num]
            if self.dtype.char in NumpyCharTypes.AllFloat:
                # check if last item is nan
                if un[count - 1] != un[count - 1]:
                    count -= 1
            # unsigned int uses high number as sentinel
            elif self.dtype.char in NumpyCharTypes.UnsignedInteger:
                if un[count - 1] == inv:
                    count -= 1
            # all other sentinels are lowest number
            else:
                if un[0] == inv:
                    count -= 1
        return count

    # -------------------------------------------------------
    def searchsorted(self, v, side="left", sorter=None) -> np.number:
        return _searchsorted(self, v, side=side, sorter=sorter)

    # ---------------------------------------------------------------------------
    def map_old(self, npdict: dict):
        """
        Example
        -------
        >>> d = {1:10, 2:20}
        >>> dat['c'] = dat.a.map(d)
        >>> print(dat)
           a  b   cb   c
        0  1  0  0.0  10
        1  1  1  1.0  10
        2  1  2  3.0  10
        3  2  3  5.0  20
        4  2  4  7.0  20
        5  2  5  9.0  20
        """
        outArray = self.copy()
        for k, v in npdict.items():
            outArray[self == k] = v
        return outArray

    def map(self, npdict: dict) -> FastArray:
        """
        Notes
        -----
        Uses ismember and can handle large dictionaries

        Examples
        --------
        >>> a=arange(3)
        >>> a.map({1: 'a', 2:'b', 3:'c'})
        FastArray(['', 'a', 'b'], dtype='<U1')
        >>> a=arange(3)+1
        >>> a.map({1: 'a', 2:'b', 3:'c'})
        FastArray(['a', 'b', 'c'], dtype='<U1')
        """
        orig = FastArray([*npdict], unicode=True)
        replace = FastArray([*npdict.values()], unicode=True)
        outArray = self.fill_invalid(self.shape, dtype=replace.dtype, inplace=False)
        found, idx = ismember(self, orig)
        outArray[found] = replace[idx[found]]
        return outArray

    # ---------------------------------------------------------------------------
    def shift(self, periods=1, invalid=None) -> FastArray:
        """
        Modeled on pandas.shift.
        Values in the array will be shifted to the right for positive, to the left for negative.
        Spaces at either end will be filled with an invalid based on the datatype.
        If abs(periods) >= the length of the FastArray, it will return a FastArray full of invalid
        will be returned.

        Parameters
        ----------
        periods: int, 1
             number of elements to shift right (if positive) or left (if negative), defaults to 1
        invalid: None, default
             optional invalid value to fill

        Returns
        -------
        FastArray shifted right or left by number of periods

        Examples
        --------
        >>> arange(5).shift(2)
        FastArray([-2147483648, -2147483648,           0,           1,            2])
        """

        if periods == 0:
            return self

        if invalid is None:
            if isinstance(self, TypeRegister.Categorical):
                invalid = 0
            else:
                try:
                    invalid = INVALID_DICT[self.dtype.num]
                except Exception:
                    raise TypeError(f"shift does not support the dtype {self.dtype.name!r}")

        # we know that this is a simple vector: shape == (len, )
        # TODO: get recycled
        temp = empty_like(self)
        if abs(periods) >= len(self):
            temp.fill(invalid)
        elif periods > 0:
            temp[:periods] = invalid
            temp[periods:] = self[:-periods]
        else:
            temp[:periods] = self[-periods:]
            temp[periods:] = invalid

        # to rewrap categoricals or datelike
        if hasattr(self, "newclassfrominstance"):
            temp = self.newclassfrominstance(temp, self)

        return temp

    # -------------------------------------------------------
    def _internal_self_compare(self, math_op, periods=1, fancy=False):
        """internal routine used for differs and transitions"""
        result = empty_like(self, dtype=np.bool_)

        if periods == 0:
            raise ValueError("periods of 0 is invalid for transitions")

        if periods > 0:
            TypeRegister.MathLedger._BASICMATH_TWO_INPUTS(
                (self[periods:], self[:-periods], result[periods:]), math_op, 0
            )
            # fill upfront with invalids
            result[:periods] = False
        else:
            TypeRegister.MathLedger._BASICMATH_TWO_INPUTS(
                (self[:periods], self[-periods:], result[:periods]), math_op, 0
            )
            # fill back with invalids (periods is negative)
            result[periods:] = False

        if fancy:
            return bool_to_fancy(result)
        return result

    # -------------------------------------------------------
    def differs(self, periods=1, fancy=False) -> FastArray:
        """
        Returns a boolean array.
        The boolean array is set to True when the previous item in the array equals the current.
        Use -1 instead of 1 if you want True set when the next item in the array equals the previous.
        See also: ``transitions``

        ::param periods: The number of elements to look ahead (or behind), defaults to 1
        :type periods: int
        :param fancy: Indicates whether to return a fancy_index instead of a boolean array, defaults to False.
        :type fancy: bool
        :return: boolean ``FastArray``, or fancyIndex (see: `fancy` kwarg)
        """
        if self.dtype.num > 13:
            result = self != self.shift(periods)
            if fancy:
                return bool_to_fancy(result)
            return result
        return self._internal_self_compare(MATH_OPERATION.CMP_EQ, periods=periods, fancy=fancy)

    # ---------------------------------------------------------------------------
    def transitions(self, periods=1, fancy=False) -> FastArray:
        """
        Returns a boolean array.
        The boolean array is set to True when the previous item in the array does not equal the current.
        Use -1 instead of 1 if you want True set when the next item in the array does not equal the previous.
        See also: ``differs``

        :param periods: The number of elements to look ahead (or behind), defaults to 1
        :type periods: int
        :param fancy: Indicates whether to return a fancy_index instead of a boolean array, defaults to False.
        :type fancy: bool
        :return: boolean ``FastArray``, or fancyIndex (see: `fancy` kwarg)

        >>> a = FastArray([0,1,2,3,3,3,4])
        >>> a.transitions(periods=1)
        FastArray([False, True, True, True, False, False, True])

        >>> a.transitions(periods=2)
        FastArray([False, False, True, True, True, False, True])

        >>> a.transitions(periods=-1)
        FastArray([ True, True, True, False, False, True, False])
        """
        if self.dtype.num > 13:
            result = self != self.shift(periods)
            if fancy:
                return bool_to_fancy(result)
            return result
        return self._internal_self_compare(MATH_OPERATION.CMP_NE, periods=periods, fancy=fancy)

    # -------------------------------------------------------
    def diff(self, periods=1) -> FastArray:
        """
        Only works for integers and floats.

        Parameters
        ----------
        periods: int, defaults to 1.  How many elements to shift the data before subtracting.

        Returns
        -------
        FastArray same length as current array.  Invalids will fill the beginning based on the periods.

        Examples
        --------
        >>> a=rt.arange(3, dtype=rt.int32); a.diff()
        FastArray([-2147483648,           1,           1])

        """
        try:
            invalid = INVALID_DICT[self.dtype.num]
        except:
            raise TypeError(f"shift does not support the dtype {self.dtype.name!r}")

        temp = empty(self.shape, dtype=self.dtype)
        if abs(periods) >= len(self):
            temp.fill(invalid)
        elif periods > 0:
            temp[:periods] = invalid

            # output into the empty array we created, np.subtract will call FastArray's subtract
            np.subtract(self[periods:], self[:-periods], out=temp[periods:])
        else:
            temp[periods:] = invalid
            np.subtract(self[:periods], self[-periods:], out=temp[:periods])
        return temp

    # -------------------------------------------------------
    def isna(self) -> FastArray:
        """
        isnan is mapped directly to isnan()
        Categoricals and DateTime take over isnan.
        FastArray handles sentinels.

        >>> a=arange(100.0)
        >>> a[5]=np.nan
        >>> a[87]=np.nan
        >>> sum(a.isna())
        2
        >>> sum(a.astype(np.int32).isna())
        2
        """
        return self.isnan()

    def notna(self) -> FastArray:
        """
        notna is mapped directly to isnotnan()
        Categoricals and DateTime take over isnotnan.
        FastArray handles sentinels.

        >>> a=arange(100.0)
        >>> a[5]=np.nan
        >>> a[87]=np.nan
        >>> sum(a.notna())
        98
        >>> sum(a.astype(np.int32).notna())
        98
        """
        return self.isnotnan()

    def replacena(self, value, inplace=False) -> FastArray:
        """
        Return a `FastArray` with all NaN and invalid values set to the specified value.

        Optionally, you can modify the original `FastArray` if it's not locked.

        Parameters
        ----------
        value : scalar or array
            A value or an array of values to replace all NaN and invalid values. If an
            array, the number of values must equal the number of NaN and invalid values.
        inplace : bool, default False
            If False, return a copy of the `FastArray`. If True, modify the original.
            This will modify any other views on this object. This fails if the
            `FastArray` is locked.

        Returns
        -------
        `FastArray` or None
            The `FastArray` will be the same size and dtype as the original array.
            Returns None if ``inplace = True``.

        See Also
        --------
        FastArray.fillna : Replace NaN and invalid values with a specified value or nearby data.
        Dataset.fillna : Replace NaN and invalid values with a specified value or nearby data.
        Categorical.fill_forward : Replace NaN and invalid values with the last valid group value.
        Categorical.fill_backward : Replace NaN and invalid values with the next valid group value.
        GroupBy.fill_forward : Replace NaN and invalid values with the last valid group value.
        GroupBy.fill_backward : Replace NaN and invalid values with the next valid group value.

        Examples
        --------
        Replace all instances of NaN with a single value:

        >>> a = rt.FastArray([rt.nan, 1.0, rt.nan, 3.0])
        >>> a.replacena(0)
        FastArray([0., 1., 0., 3.])

        Replace all invalid values with 0s:

        >>> b = rt.FastArray([0, 1, 2, 3, 4, 5])
        >>> b[0:3] = b.inv
        >>> b.replacena(0)
        FastArray([0, 0, 0, 3, 4, 5])

        Replace each instance of NaN with a different value:

        >>> a.replacena([0, 2])
        FastArray([0., 1., 2., 3.])
        """
        inst = self if inplace else self.copy()
        isna = inst.isna()
        if isna.any():
            inst[isna] = value
        if inplace:
            return None
        return inst

    def fillna(self, value=None, method=None, inplace=False, limit=None) -> FastArray:
        """
        Replace NaN and invalid values with a specified value or nearby data.

        Optionally, you can modify the original `FastArray` if it's not locked.

        Parameters
        ----------
        value : scalar or array, default None
            A value or an array of values to replace all NaN and invalid values.
            A `value` is required if ``method = None``. An array can be used only when
            ``method = None``. If an array is used, the number of values in the array
            must equal the number of NaN and invalid values.
        method : {None, 'backfill', 'bfill', 'pad', 'ffill'}, default None
            Method to use to propagate valid values.

            * backfill/bfill: Propagates the next encountered valid value backward.
              Calls `FastArray.fill_backward <https://eot.gitlab.ds.susq.com/sigpydata/riptable/riptable/autoapi/riptable/rt_fastarraynumba/index.html#riptable.rt_fastarraynumba.fill_backward>`_.
            * pad/ffill: Propagates the last encountered valid value forward. Calls
              `FastArray.fill_forward <https://eot.gitlab.ds.susq.com/sigpydata/riptable/riptable/autoapi/riptable/rt_fastarraynumba/index.html#riptable.rt_fastarraynumba.fill_forward>`_.
            * None: A replacement value is required if ``method = None``. Calls
              :meth:`FastArray.replacena`.
            If there's not a valid value to propagate forward or backward, the NaN or
            invalid value is not replaced unless you also specify a `value`.
        inplace : bool, default False
            If False, return a copy of the `FastArray`. If True, modify original data.
            This will modify any other views on this object. This fails if the
            `FastArray` is locked.
        limit : int, default None
            If `method` is specified, this is the maximium number of consecutive NaN or
            invalid values to fill. If there is a gap with more than this number of
            consecutive NaN or invalid values, the gap will be only partially filled.

        Returns
        -------
        `FastArray`
            The `FastArray` will be the same size and dtype as the original array.

        See Also
        --------
        riptable.rt_fastarraynumba.fill_forward : Replace NaN and invalid values with
            the last valid value.
        riptable.rt_fastarraynumba.fill_backward : Replace NaN and invalid values with
            the next valid value.
        riptable.fill_forward : Replace NaN and invalid values with the last valid
            value.
        riptable.fill_backward : Replace NaN and invalid values with the next valid
            value.
        Dataset.fillna : Replace NaN and invalid values with a specified value or
            nearby data.
        FastArray.replacena : Replace NaN and invalid values with a specified value.
        Categorical.fill_forward : Replace NaN and invalid values with the last valid
            group value.
        Categorical.fill_backward : Replace NaN and invalid values with the next valid
            group value.
        GroupBy.fill_forward : Replace NaN and invalid values with the last valid
            group value.
        GroupBy.fill_backward : Replace NaN and invalid values with the next valid
            group value.

        Examples
        --------
        Replace all NaN values with 0s:

        >>> a = rt.FastArray([rt.nan, 1.0, rt.nan, rt.nan, rt.nan, 5.0])
        >>> a.fillna(0)
        FastArray([0., 1., 0., 0., 0., 5.])

        Replace all invalid values with 0s:

        >>> b = rt.FastArray([0, 1, 2, 3, 4, 5])
        >>> b[0:3] = b.inv
        >>> b.fillna(0)
        FastArray([0, 0, 0, 3, 4, 5])

        Replace each instance of NaN with a different value:

        >>> a.fillna([0, 2, 3, 4])
        FastArray([0., 1., 2., 3., 4., 5.])

        Propagate the last encountered valid value forward. Note that where there's no
        valid value to propagate, the NaN or invalid value isn't replaced.

        >>> a.fillna(method = 'ffill')
        FastArray([nan,  1.,  1.,  1.,  1.,  5.])

        You can use the `value` parameter to specify a value to use where there's no
        valid value to propagate.

        >>> a.fillna(value = 0, method = 'ffill')
        FastArray([0., 1., 1., 1., 1., 5.])

        Replace only the first NaN or invalid value in any consecutive series of NaN or
        invalid values.

        >>> a.fillna(method = 'bfill', limit = 1)
        FastArray([ 1.,  1., nan, nan,  5.,  5.])
        """
        if method is not None:
            if method in ["backfill", "bfill"]:
                return self.fill_backward(value, inplace=inplace, limit=limit)
            if method in ["pad", "ffill"]:
                return self.fill_forward(value, inplace=inplace, limit=limit)
            raise KeyError(f"fillna: The method {method!r} must be 'backfill', 'bfill', 'pad', 'ffill'")

        if value is None:
            raise ValueError(f"fillna: Must specify either a 'value' that is not None or a 'method' that is not None.")

        if limit is not None:
            raise KeyError(f"fillna: There is no limit when method is None")

        return self.replacena(value, inplace=inplace)

    def statx(self) -> Dataset:
        return statx(self)

    # ---------------------------------------------------------------------------
    def _is_not_supported(self, arr):
        """returns True if a numpy array is not FastArray internally supported"""
        if not (arr.flags.c_contiguous or arr.flags.f_contiguous):
            # TODO enable this warning in a future minor release
            # FastArray._possibly_warn(f'_is_not_supported: unsupported array flags {arr.flags}')
            return True
        if arr.dtype.char not in NumpyCharTypes.Supported:
            # TODO enable this warning in a future minor release
            # FastArray._possibly_warn(f'_is_not_supported: unsupported array dtype {arr.dtype}\nSupported dtypes {NumpyCharTypes.Supported}')
            return True
        if len(arr.strides) == 0:
            # TODO enable this warning in a future minor release
            # FastArray._possibly_warn(f'_is_not_supported: unsupported array strides {arr.strides}')
            return True
        return False

    # ---------------------------------------------------------------------------
    def __array_function__(self, func, types, args, kwargs):
        if self.NEW_ARRAY_FUNCTION_ENABLED:
            return self._new_array_function(func, types, args, kwargs)
        else:
            return self._legacy_array_function(func, types, args, kwargs)

    # ---------------------------------------------------------------------------
    def _legacy_array_function(self, func, types, args, kwargs):
        """
        Called before array_ufunc.
        Does not get called for every function np.isnan/trunc/true_divide for instance.
        """
        reduceFunc = NUMPY_CONVERSION_TABLE.get(func, None)

        # TODO:
        # kwargs of 'axis': None  'out': None should be accepted

        if reduceFunc is not None and len(kwargs) == 0:
            # speed path (todo add call to ledger)
            # default to ddof=0 when no kwargs passed
            result = rc.Reduce(args[0], reduceFunc, 0)

            if result is not None:
                # TypeRegister.MathLedger._REDUCE(args[0], newfunc)

                dtype = kwargs.get("dtype", None)
                if dtype is not None:
                    # user forced dtype return value
                    return dtype(result)

                # preserve type for min/max/nanmin/nanmax
                if reduceFunc in [
                    REDUCE_FUNCTIONS.REDUCE_MIN,
                    REDUCE_FUNCTIONS.REDUCE_NANMIN,
                    REDUCE_FUNCTIONS.REDUCE_MAX,
                    REDUCE_FUNCTIONS.REDUCE_NANMAX,
                ]:
                    return self.dtype.type(result)

                # internally numpy expects a dtype returned for nanstd and other calculations
                if isinstance(result, (int, np.integer)):
                    # for uint64, the high bit must be preserved
                    if self.dtype.char in NumpyCharTypes.UnsignedInteger64:
                        return np.uint64(result)
                    return np.int64(result)

                return np.float64(result)
        # call the version numpy wanted use to
        return super(FastArray, self).__array_function__(func, types, args, kwargs)

    # ---------------------------------------------------------------------------
    def _new_array_function(self, func: Callable, types: tuple, args: tuple, kwargs: dict):
        """
        FastArray implementation of the array function protocol.

        Parameters
        ----------
        func: callable
            An callable exposed by NumPy’s public API, which was called in the form ``func(*args, **kwargs)``.
        types: tuple
            A tuple of unique argument types from the original NumPy function call that implement ``__array_function__``.
        args: tuple
            The tuple of arguments that will be passed to `func`.
        kwargs: dict
            The dictionary of keyword arguments that will be passed to `func`.

        Raises
        ------
        TypeError
            If `func` is not overridden by a corresponding riptable array function then a TypeError is raised.

        Notes
        -----
        This array function implementation requires each class, such as FastArray and any other derived class,
        to implement their own version of the Numpy array function API. In the event these array functions defer to the
        inheriting class they will need to either re-wrap the results in the correct type or raise exception if a
        particular operation is not well-defined nor meaningful for the derived class.
        If an array function, which is also a universal function, is not overridden as an array function, but defined
        as a ufunc then it will not be called unless it is registered with the array function helper since array function
        protocol takes priority over the universal function protocol.

        See Also
        --------
        For information around the Numpy array function protocol see NEP 18:
        https://numpy.org/neps/nep-0018-array-function-protocol.html
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{FastArray.__name__}._new_array_function(fun={func}, types={types}, args={args}, kwargs={kwargs})"
            )
        # handle `func` argument
        array_func: Callable = FastArray._ArrayFunctionHelper.get_array_function(func)
        if array_func is None:
            # fallback to numpy for unhandled array functions and attempt to cast back to FastArray
            result = super().__array_function__(func, types, args, kwargs)
            if result is NotImplemented:
                return NotImplemented
            elif isinstance(result, np.ndarray):
                return result.view(FastArray)
            elif isinstance(result, list):
                return [(x.view(FastArray) if isinstance(x, np.ndarray) else x) for x in result]
            elif isinstance(result, tuple):
                return tuple([(x.view(FastArray) if isinstance(x, np.ndarray) else x) for x in result])
            else:
                # Unknown result type.
                raise TypeError(f"Unknown result type '{type(result)}' returned by ndarray.{func}.")

        # handle `types` argument
        array_func_type_check: Callable = FastArray._ArrayFunctionHelper.get_array_function_type_compatibility_check(
            func
        )
        if array_func_type_check is None:
            # no custom type compatibility check; default type compatibility check
            # this allows subclasses that don't override __array_function__ to handle FastArray objects
            for typ in types:
                if not issubclass(typ, FastArray):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"{FastArray.__name__}.__array_function__: unsupported type {repr(typ)}")
                    return NotImplemented
        else:  # custom type compatibility check
            valid: bool = array_func_type_check(types)
            if not valid:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"{FastArray.__name__}.__array_function__: unsupported type in {repr(types)}")
                return NotImplemented

        return array_func(*args, **kwargs)

    # ---------------------------------------------------------------------------
    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs: Any, **kwargs: Any):
        """
        The FastArray universal function (or ufunc) override offers multithreaded C/C++ implementation at the RiptideCPP layer.

        When FastArray receives a `ufunc` callable it will attempt to handle it in priority order:
            1. considering ``FastArray`` ``FastFunction`` is enabled, ufunc is handled by an explicit ufunc override, otherwise
            2. ufunc is handled at the Riptable / Numpy API overrides level, otherwise
            3. ufunc is handled at the Numpy API level.

        Given a combination of `ufunc`, `inputs`, and `kwargs`, if neither of the aforementioned cases support this
        then a warning is emitted.

        The following references to supported ufuncs are grouped by method type.
            - For `method` type ``reduce``, see ``gReduceUFuncs``.
            - For `method` type ``__call__``, see ``gBinaryUFuncs``, ``gBinaryLogicalUFuncs``, ``gBinaryBitwiseUFuncs``, and ``gUnaryUFuncs``.
            - For `method` type ``at`` return ``None``.

        If `out` argument is specified, then an extra array copy is performed on the result of the ufunc computation.

        If a `dtype` keyword is specified, all efforts are made to respect the `dtype` on the result of the computation.

        Parameters
        ----------
        ufunc : callable
            The ufunc object that was called.
        method : str
            A string indicating which Ufunc method was called (one of "__call__", "reduce", "reduceat", "accumulate", "outer", "inner").
        inputs
            A tuple of the input arguments to the ufunc.
        kwargs
            A dictionary containing the optional input arguments of the ufunc. If given, any out arguments, both positional and keyword, are passed as a tuple in kwargs.

        Returns
        -------
            The method should return either the result of the operation, or NotImplemented if the operation requested is not implemented.

        Notes
        -----
        The current implementation does not support the following keyword arguments: `casting`, `sig`, `signature`, and
        `core_signature`.

        It has partial support for keyword arguments: `where`, `axis`, and `axes`, if they match
        the default values.

        If FastArray's ``WarningLevel`` is enabled, then warnings will be emitted if any of unsupported or partially
        supported keyword arguments are passed.

        TODO document custom up casting rules.

        See Also
        --------
        For more information on ufunc see the following numpy documents:
            - https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__
            - https://numpy.org/doc/stable/reference/ufuncs.html

        Note, the docstring Parameters and Return section is repeated from the numpy
        `__array_ufunc__` docstring since this is overriding that method.
        """
        # TODO consider using type annotation typing.Final for these read-only variables when moving to Python 3.8
        # Python 3.8 added support for typing.Final. Final will catch unintended assignments for constants when running
        # static type checkers such as mypy.
        _UNSUPPORTED_KEYWORDS: Tuple[str, str, str, str] = ("casting", "sig", "signature", "core_signature")
        _PARTIALLY_SUPPORTED_KEYWORDS_TO_DEFAULTS: Mapping[str, Union[None, bool]] = {
            "where": True,
            "axis": None,
            "axes": None,
        }
        toplevel_abort: bool = False

        if FastArray.Verbose > 2:
            print("*** top level array_ufunc", ufunc, method, *inputs, kwargs)

        # flip any inputs that are fastarrays back to an ndarray...
        args: List[Any] = []
        for input in inputs:
            if isinstance(input, np.ndarray):
                is_not_supported = self._is_not_supported(input)
                if is_not_supported:
                    # TODO enable this warning in a future minor release
                    # FastArray._possibly_warn(f'__array_ufunc__: unsupported input "{input}"')
                    toplevel_abort |= is_not_supported
            args.append(input)

        # Check for numpy rules that we cannot handle.
        for kw in _UNSUPPORTED_KEYWORDS:
            if kw in kwargs:
                # TODO enable this warning in a future minor release
                # FastArray._possibly_warn(f'__array_ufunc__: unsupported keyword argument "{kw}"')
                toplevel_abort |= True

        # Check for numpy rules that we partially support; that is, where we only support
        # the keyword if the value is some default value and otherwise punt to numpy.
        # The value associated with each keyword in the dictionary is the only value we'll
        # support for that keyword.
        # For example, in numpy 1.17 the sum() function passes where=True by default.
        for kw, default_val in _PARTIALLY_SUPPORTED_KEYWORDS_TO_DEFAULTS.items():
            if kw in kwargs:
                # Use a type check before equality here to avoid errors caused
                # by checking equality between bools and arrays.
                kwarg_val = kwargs[kw]
                if type(default_val) != type(kwarg_val) or kwarg_val != default_val:
                    toplevel_abort |= True

        dtype: Optional[np.dtype] = kwargs.get("dtype", None)

        has_outputs: bool = False
        out_args: List[Any] = []

        # flip any outputs to ndarray...
        outputs = kwargs.pop("out", None)
        if outputs:
            has_outputs = True
            for output in outputs:
                if isinstance(output, np.ndarray):
                    is_not_supported = self._is_not_supported(output)
                    if is_not_supported:
                        # TODO enable this warning in a future minor release
                        # FastArray._possibly_warn(f'__array_ufunc__: unsupported output "{output}"')
                        toplevel_abort |= is_not_supported
                out_args.append(output)
            # replace out
            kwargs["out"] = tuple(out_args)
        else:
            # TJD - here outputs was not specified
            # now if UFunc.nout ==1, this function requires an output
            outputs = (None,) * ufunc.nout

        # See https://docs.python.org/3/c-api/typeobj.html
        # See Number Object Structures and Mapping Object Structure for indexing

        # ufunc.nin	    The number of inputs.
        # ufunc.nout	    The number of outputs.
        # ufunc.nargs	The number of arguments.
        # ufunc.ntypes	The number of types.
        # ufunc.types	Returns a list with types grouped input->output.
        # ufunc.identity	The identity value.

        final_dtype: Optional[np.dtype] = None
        fast_function: Optional[MATH_OPERATION] = None
        reduce_func: Optional[REDUCE_FUNCTIONS] = None

        # Handle reduce ufunc methods.
        # note: when method is 'at' this is an inplace unbuffered operation
        # this can speed up routines that use heavy masked operations
        if method == "reduce" and FastArray.FasterUFunc and not toplevel_abort:

            # a.any() and a.all() are logical reduce operations
            # Examples
            # Look for axis:None -- otherwise ABORT
            # Then look for Keepdims wihch means to wrap result in list/array?
            # Then check dtype also
            #
            # In [22]: t=FA([[3,4,5],[6,7,8]])
            # In [23]: np.add.reduce(t)
            #!!reduce  reduce nin: 2 1 <ufunc 'add'> [array([[3, 4, 5],
            #       [6, 7, 8]])] out: (None,) kwargs: {}
            # results [ 9 11 13]
            # Out[23]: array([ 9, 11, 13])
            # In [24]: np.add.reduce(t, axis=None)
            #!!reduce  reduce nin: 2 1 <ufunc 'add'> [array([[3, 4, 5],
            #       [6, 7, 8]])] out: (None,) kwargs: {'axis': None}
            # results 33
            # Out[24]: 33
            # In [25]: np.add.reduce(t, axis=None, keepdims=True)
            #!!reduce  reduce nin: 2 1 <ufunc 'add'> [array([[3, 4, 5],
            #       [6, 7, 8]])] out: (None,) kwargs: {'axis': None, 'keepdims': True}
            # results [[33]]
            # Out[25]: array([[33]])
            # In [26]: np.add.reduce(t, axis=None, keepdims=True, dtype=np.float32)
            #!!reduce  reduce nin: 2 1 <ufunc 'add'> [array([[3, 4, 5],
            #       [6, 7, 8]])] out: (None,) kwargs: {'axis': None, 'keepdims': True, 'dtype': <class 'numpy.float32'>}
            # results [[33.]]
            # Out[26]: array([[33.]], dtype=float32)
            # print("!!reduce ", method, 'nin:', ufunc.nin, ufunc.nout, ufunc,  args, 'out:', outputs,  'kwargs:', kwargs,'ndim', args[0].ndim)
            # resultN = super(FastArray, self).__array_ufunc__(ufunc, method,**kwargs)
            # print("!!result numpy", resultN, type(resultN))
            # NOTE:
            # look for reduce logical_or
            # look for reduce_logical_and   (used with np.fmin for instance)

            reduce_func = gReduceUFuncs.get(ufunc, None)

        # check if we can proceed to calculate a faster way
        if method == "__call__" and FastArray.FasterUFunc and not toplevel_abort:

            # check for binary ufunc
            if len(args) == 2 and ufunc.nout == 1:

                ###########################################################################
                ## BINARY
                ###########################################################################
                array_types: List[np.dtype] = []
                scalar_types: List[ScalarType] = []

                scalars: int = 0
                abort: int = 0
                for arr in args:
                    arrType = type(arr)
                    if arrType in ScalarType:
                        scalars += 1
                        scalar_types.append(arrType)
                    else:
                        try:
                            array_types.append(arr.dtype)
                            # check for non contingous arrays
                            if arr.itemsize != arr.strides[0]:
                                abort = 1
                        except:
                            abort = 1
                            # can happen when None or a python list is passed
                            if FastArray.Verbose > 1:
                                print(f"**dont know how to handle array {arr} args: {args}")

                if abort == 0:
                    if scalars < 2:
                        is_logical = 0
                        # check for add, sub, mul, divide, power
                        fast_function = gBinaryUFuncs.get(ufunc, None)
                        if fast_function is None:
                            # check for comparison and logical or/and functions
                            fast_function = gBinaryLogicalUFuncs.get(ufunc, None)
                            if fast_function is not None:
                                if FastArray.Verbose > 2:
                                    print(f"**logical function called {ufunc} args: {args}")
                                is_logical = 1
                                final_dtype = np.bool_

                        if fast_function is None:
                            # check for bitwise functions? (test this)
                            fast_function = gBinaryBitwiseUFuncs.get(ufunc, None)

                        if fast_function is not None:
                            if has_outputs and is_logical == 0:
                                # have to conform to output
                                final_dtype = out_args[0].dtype
                            else:
                                if is_logical == 1 and scalars == 1:
                                    # NOTE: scalar upcast rules -- just apply to logicals so that arr < 5 does not upcast?
                                    #       or globally apply this rule so that arr = arr + 5
                                    # if scalars == 1:
                                    # special case have to see if scalar is in range
                                    if type(args[0]) in ScalarType:
                                        scalar_val = args[0]
                                    else:
                                        scalar_val = args[1]

                                    final_dtype = logical_find_common_type(array_types, scalar_types, scalar_val)

                                else:
                                    print
                                    # TODO: check for bug where np.int32 type 7 gets flipped to np.int32 type 5
                                    if scalars == 0 and len(array_types) == 2 and (array_types[0] == array_types[1]):
                                        final_dtype = array_types[0]
                                    else:
                                        # check for int scalar against int
                                        # bug where np.int8 and then add +1999 or larger number.  need to upcast
                                        if scalars == 1 and array_types[0].num <= 10:
                                            if type(args[0]) in ScalarType:
                                                scalar_val = args[0]
                                            else:
                                                scalar_val = args[1]

                                            final_dtype = logical_find_common_type(
                                                array_types, scalar_types, scalar_val
                                            )
                                        else:
                                            final_dtype = np.find_common_type(array_types, scalar_types)

                            # if we are adding two strings or unicode, special case
                            # if we think the final dtype is an object, check if this is really two strings
                            if fast_function == MATH_OPERATION.ADD and (
                                array_types[0].num == 18 or array_types[0].num == 19
                            ):
                                # assume addition of two strings
                                final_dtype = array_types[0]
                                if scalars != 0:
                                    # we have a scalar... make sure we convert it
                                    if type(args[0]) in ScalarType:
                                        # fix scalar type make sure string or unicode
                                        if array_types[0].num == 18:
                                            args[0] = str.encode(str(args[0]))
                                        if array_types[0].num == 19:
                                            args[0] = str(args[0])
                                    else:
                                        if array_types[0].num == 18:
                                            args[1] = str.encode(str(args[1]))
                                        if array_types[0].num == 19:
                                            args[1] = str(args[1])
                                else:
                                    # we have two arrays, if one array is not proper string type, convert it
                                    if array_types[1] != final_dtype:
                                        if array_types[0].num == 18:
                                            args[1] = args[1].astype("S")
                                        if array_types[0].num == 19:
                                            args[1] = args[1].astype("U")

                                if FastArray.Verbose > 2:
                                    print("ADD string operation", array_types, scalar_types)

                            elif scalars == 0:
                                if array_types[0] != array_types[1]:
                                    # UPCAST RULES
                                    if array_types[0] == final_dtype and array_types[1] != final_dtype:
                                        # print("!!!upcast rules second", array_types[0], array_types[1], final_dtype)
                                        # convert to the proper type befor calculation
                                        args[1] = _ASTYPE(args[1], final_dtype)

                                    elif array_types[0] != final_dtype and array_types[1] == final_dtype:
                                        # print("!!!upcast rules first", array_types[0], array_types[1], final_dtype)
                                        # convert to the proper type befor calculation
                                        args[0] = _ASTYPE(args[0], final_dtype)
                                    else:
                                        # sometimes both of them must be upcast...
                                        # consider  int8 * uint8 ==> will upcast to int16
                                        # print("!!!cannot understand upcast rules", arraytypes[0], arraytypes[1], final_dtype)
                                        args[0] = _ASTYPE(args[0], final_dtype)
                                        args[1] = _ASTYPE(args[1], final_dtype)
                                        # TJD check logic here... what does numpy when int* * uint8 ? speed test
                                        ##UseNumpy = True
                            else:
                                # UPCAST RULES when one is a scalar
                                if array_types[0] != final_dtype:
                                    # which argument is the scalar?  convert the other one
                                    if type(args[0]) in ScalarType:
                                        # print("converting arg2 from", args[1], final_dtype)
                                        args[1] = _ASTYPE(args[1], final_dtype)
                                    else:
                                        # print("converting arg1 from ", args[0], final_dtype)
                                        args[0] = _ASTYPE(args[0], final_dtype)

            # not a binary ufunc, check for unary ufunc
            # check for just 1 input (unary)

            elif (ufunc.nin == 1) and (ufunc.nout == 1):
                ###########################################################################
                ## UNARY
                ###########################################################################
                fast_function = gUnaryUFuncs.get(ufunc, None)
            else:
                if FastArray.Verbose > 1:
                    print("***unknown ufunc arg style: ", ufunc.nin, ufunc.nout, ufunc, args, kwargs)

        # -------------------------------------------------------------------------------------------------------------
        if not FastArray.FasterUFunc:
            fast_function = None
            reduce_func = None

        # check for a reduce func like sum or min
        if reduce_func is not None:
            keepdims: bool = kwargs.get("keepdims", False)
            if dtype is None:
                dtype = args[0].dtype

            # MathLedger
            result = TypeRegister.MathLedger._REDUCE(args[0], reduce_func)
            char = np.dtype(dtype).char
            if FastArray.Verbose > 1:
                print("***result from reduce", result, type(result), dtype, char)

            if result is not None:
                # print("reduce called", ufunc, keepdims, dtype)
                if reduce_func in [REDUCE_FUNCTIONS.REDUCE_SUM, REDUCE_FUNCTIONS.REDUCE_NANSUM] and isinstance(
                    result, float
                ):
                    result = np.float64(result)
                elif dtype != np.float32 and dtype != np.float64:
                    # preserve integers
                    if char in NumpyCharTypes.UnsignedInteger64:
                        # preserve high bit
                        result = np.uint64(result)
                    else:
                        result = np.int64(result)
                else:
                    result = np.float64(result)

                # MIN/MAX need to return same type
                if reduce_func >= REDUCE_FUNCTIONS.REDUCE_MIN:
                    # min max not allowed on empty array per unit test
                    if len(args[0]) == 0:
                        raise ValueError("min/max arg is an empty sequence.")

                    # min/max/nanmin/nanmax -- same result
                    if dtype == np.bool_:
                        result = np.bool(result)
                    else:
                        result = dtype.type(result)

                    if keepdims:
                        result = FastArray([result]).astype(dtype)
                elif keepdims:
                    # force back into an array from scalar
                    result = FastArray([result])

                # we did the reduce, now return the result
                return result

        # check for normal call function
        elif fast_function is not None:
            # Call the FastArray APIs instead of numpy
            # callmode = 'f'
            results = None
            if ufunc.nin == 2:
                final_num = -1
                if final_dtype is not None:
                    if final_dtype == np.bool_:
                        final_num = 0
                    else:
                        final_num = final_dtype.num

                # because scalars can be passed as np.int64(864000)
                if type(args[0]) in gNumpyScalarType:
                    # print('converting arg1', args[0])
                    args[0] = np.asarray(args[0])

                if type(args[1]) in gNumpyScalarType:
                    # print('converting arg2', args[1])
                    args[1] = np.asarray(args[1])

                if FastArray.Verbose > 2:
                    print(
                        "*** binary think we can call",
                        fast_function,
                        ufunc.nin,
                        ufunc.nout,
                        "arg1",
                        args[0],
                        "arg2",
                        args[1],
                        "out",
                        out_args,
                        "final",
                        final_num,
                    )
                if len(out_args) == 1:
                    results = TypeRegister.MathLedger._BASICMATH_TWO_INPUTS(
                        (args[0], args[1], out_args[0]), fast_function, final_num
                    )
                else:
                    results = TypeRegister.MathLedger._BASICMATH_TWO_INPUTS(
                        (args[0], args[1]), fast_function, final_num
                    )
            else:
                # for conversion functions
                # dtype=kwargs.get('dtype',None)
                if FastArray.Verbose > 2:
                    print(
                        "*** unary think we can call",
                        fast_function,
                        ufunc.nin,
                        ufunc.nout,
                        "arg1",
                        args[0],
                        "out",
                        out_args,
                    )

                if len(out_args) == 1:
                    results = TypeRegister.MathLedger._BASICMATH_ONE_INPUT((args[0], out_args[0]), fast_function, 0)
                else:
                    results = TypeRegister.MathLedger._BASICMATH_ONE_INPUT((args[0]), fast_function, 0)

            if results is not None and len(out_args) == 1:
                # when the output argument is forced but we calculate it into another array we need to copy the result into the output
                if not rc.CompareNumpyMemAddress(out_args[0], results):
                    if FastArray.Verbose > 2:
                        print(
                            "*** performing an extra copy to match output request",
                            id(out_args[0]),
                            id(results),
                            out_args[0],
                            results,
                        )
                    out_args[0][...] = results
                    results = out_args[0]

            if results is None:
                # punted
                # callmode='p'
                if FastArray.Verbose > 1:
                    print("***punted ufunc: ", ufunc.nin, ufunc.nout, ufunc, args, kwargs)
                fast_function = None
                # fall to "if fast_function is None" and run through numpy...

            # respect dtype
            elif dtype is not None and isinstance(results, np.ndarray):
                if dtype is not results.dtype:
                    if FastArray.Verbose > 1:
                        print("***result from reduce", results, results.dtype, dtype)
                    # convert
                    results = results.astype(dtype)

        if fast_function is None:
            # Call the numpy APIs
            # Check if we can use the recycled arrays to avoid an allocation for the output array

            if FastArray.Verbose > 1:
                print("**punted on numpy!", ufunc)

            # NOTE: We are going to let numpy process it
            # We must change all FastArrays to normal numpy arrays
            args = []
            for input in inputs:
                # flip back to numpy to avoid errors when numpy calculates
                if isinstance(input, FastArray):
                    args.append(input.view(np.ndarray))
                else:
                    args.append(input)

            if has_outputs:
                outputs = kwargs.pop("out", None)
                if outputs:
                    out_args = []
                    for output in outputs:
                        if isinstance(output, FastArray):
                            out_args.append(output.view(np.ndarray))
                        else:
                            out_args.append(output)
                    # replace out
                    kwargs["out"] = tuple(out_args)

            # NOTE: If the specified ufunc + inputs combination isn't supported by numpy either,
            #       as of numpy 1.17.x this call will end up raising a UFuncTypeError so the rest
            #       of the FastArray.__array_ufunc__ body (below) won't end up executing.
            results = TypeRegister.MathLedger._ARRAY_UFUNC(super(FastArray, self), ufunc, method, *args, **kwargs)

        # If riptable has not implemented a certain ufunc (or doesn't support it for the given arguments),
        # emit a warning about it to let the user know.
        # When numpy does not support the ufunc+inputs either, we won't reach this point (as of numpy 1.17.x),
        # since numpy will raise a UFuncTypeError earlier (before this point) rather than after we return NotImplemented.
        if results is NotImplemented:
            warnings.warn(f"***ufunc {ufunc} {args} {kwargs} is not implemented")
            return NotImplemented

        # Ufuncs also have a fifth method that allows in place operations to be performed using fancy indexing.
        # No buffering is used on the dimensions where fancy indexing is used, so the fancy index can list an item more than once
        #     and the operation will be performed on the result of the previous operation for that item.
        # ufunc.reduce(a[, axis, dtype, out, keepdims])	Reduces a's dimension by one, by applying ufunc along one axis.
        # ufunc.accumulate(array[, axis, dtype, out])	Accumulate the result of applying the operator to all elements.
        # ufunc.reduceat(a, indices[, axis, dtype, out])	Performs a (local) reduce with specified slices over a single axis.
        # ufunc.outer(A, B)	Apply the ufunc op to all pairs (a, b) with a in A and b in B.
        # ufunc.at(a, indices[, b])	Performs unbuffered in place operation on operand 'a' for elements specified by 'indices'.

        if method == "at":
            return

        if ufunc.nout == 1:
            # check if we used our own output

            # if isinstance(outArray, np.ndarray):
            #    return outArray.view(FastArray)

            # if (final_dtype != None and final_dtype != results.dtype):
            #    print("****** mispredicted final", final_dtype, results.dtype, ufunc, scalartypes, args, outputs, kwargs);
            # results = (results,)

            if not isinstance(results, FastArray) and isinstance(results, np.ndarray):
                return results.view(FastArray)

            # think hit here for sum wihch does not return an array, just a number
            return results

        # more than one item, so we are making a tuple
        # can result in __array_finalize__ being called
        results = tuple(
            (np.asarray(result).view(FastArray) if output is None else output)
            for result, output in zip(results, outputs)
        )

        # check if we have a tuple of one item, if so just return the one item
        if len(results) == 1:
            results = results[0]
        return results

    @property
    def numbastring(self):
        """
        converts byte string and unicode strings to a 2dimensional array
        so that numba can process it correctly

        Examples
        --------
        >>> @numba.jit(nopython=True)
        ... def numba_str(txt):
        ...     x=0
        ...     for i in range(txt.shape[0]):
        ...         if (txt[i,0]==116 and  # 't'
        ...             txt[i,1]==101 and  # 'e'
        ...             txt[i,2]==120 and  # 'x'
        ...             txt[i,3]==116):    # 't'
        ...             x += 1
        ...     return x
        >>>
        >>> x=FastArray(['some','text','this','is'])
        >>> numba_str(x.view(np.uint8).reshape((len(x), x.itemsize)))
        >>> numba_str(x.numbastring)
        """

        intype = self.dtype.__str__()
        if intype[0] == "|" or intype[0] == "<":
            if intype[1] == "S":
                return self.view(np.uint8).reshape((len(self), self.itemsize))
            if intype[1] == "U":
                return self.view(np.uint32).reshape((len(self), self.itemsize // 4))
        return self

    # -----------------------------------------------------------
    def apply_numba(self, *args, otype=None, myfunc="myfunc", name=None):
        """
        Print to screen an example numba signature for the array.

        You can then copy this example to build your own numba function.

        Parameters
        ----------
        *args:
            Test arguments

        otype: str, default None
            A different output data type

        myfunc: str, default 'myfunc'
            A string to call the function

        name: str, default None
            A string to name the array

        Examples
        --------
        >>> import numba
        >>> @numba.guvectorize(['void(int64[:], int64[:])'], '(n)->(n)')
        ... def squarev(x,out):
        ...     for i in range(len(x)):
        ...         out[i]=x[i]**2
        ...
        >>> a=arange(1_000_000).astype(np.int64)
        >>> squarev(a)
        FastArray([           0,            1,            4, ..., 999994000009,
                   999996000004, 999998000001], dtype=int64)
        """
        if name is None:
            # try first to get the name
            name = self.get_name()

            if name is None:
                name = "a"

        intype = self.dtype.__str__()

        if otype is None:
            outtype = self.dtype.__str__()
        else:
            outtype = np.dtype(otype).__str__()

        # TODO: what if unicode or string?  .frombuffer/.view(np.uint8)

        preamble = "import numba\n@numba.guvectorize([\n"

        middle = f"'void({intype}[:], {outtype}[:])',       # <-- can stack multiple different dtypes  x.view(np.uint8).reshape(-1, x.itemsize)\n"

        postamble = "    ], '(n)->(n)', target='cpu')\n"
        code = f"def {myfunc}(data_in, data_out):\n    for i in range(len(data_in)):\n        data_out[i]=data_in[i]   #<-- put your code here\n"
        exec = preamble + middle + postamble + code

        print("Copy the code snippet below and rename myfunc")
        print("---------------------------------------------")
        print(exec)
        print("---------------------------------------------")
        if intype[0] == "|" or intype[0] == "<":
            if intype[1] == "S":
                print(
                    f"Then call {myfunc}({name}.numbastring,empty_like({name}).numbastring) where {name} is the input array"
                )
            elif intype[1] == "U":
                print(
                    f"Then call {myfunc}({name}.numbastring,empty_like({name}).numbastring) where {name} is the input array"
                )
        else:
            print(f"Then call {myfunc}({name},empty_like({name})) where {name} is the input array")
        # return exec

    def apply(self, pyfunc, *args, otypes=None, doc=None, excluded=None, cache=False, signature=None):
        """
        Generalized function class.  see: np.vectorize

        Creates and then applies a vectorized function which takes a nested sequence of objects or
        numpy arrays as inputs and returns an single or tuple of numpy array as
        output. The vectorized function evaluates `pyfunc` over successive tuples
        of the input arrays like the python map function, except it uses the
        broadcasting rules of numpy.

        The data type of the output of `vectorized` is determined by calling
        the function with the first element of the input.  This can be avoided
        by specifying the `otypes` argument.

        Parameters
        ----------
        pyfunc : callable
            A python function or method.
        otypes : str or list of dtypes, optional
            The output data type. It must be specified as either a string of
            typecode characters or a list of data type specifiers. There should
            be one data type specifier for each output.
        doc : str, optional
            The docstring for the function. If `None`, the docstring will be the
            ``pyfunc.__doc__``.
        excluded : set, optional
            Set of strings or integers representing the positional or keyword
            arguments for which the function will not be vectorized.  These will be
            passed directly to `pyfunc` unmodified.

            .. versionadded:: 1.7.0

        cache : bool, optional
           If `True`, then cache the first function call that determines the number
           of outputs if `otypes` is not provided.

            .. versionadded:: 1.7.0

        signature : string, optional
            Generalized universal function signature, e.g., ``(m,n),(n)->(m)`` for
            vectorized matrix-vector multiplication. If provided, ``pyfunc`` will
            be called with (and expected to return) arrays with shapes given by the
            size of corresponding core dimensions. By default, ``pyfunc`` is
            assumed to take scalars as input and output.

            .. versionadded:: 1.12.0

        Returns
        -------
        vectorized : callable
            Vectorized function.

        See Also
        --------
        FastArray.apply_numba
        FastArray.apply_pandas

        Examples
        --------
        >>> def myfunc(a, b):
        ...     "Return a-b if a>b, otherwise return a+b"
        ...     if a > b:
        ...         return a - b
        ...     else:
        ...         return a + b
        >>>
        >>> a=arange(10)
        >>> b=arange(10)+1
        >>> a.apply(myfunc,b)
        FastArray([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19])

        Example with one input array

        >>> def square(x):
        ...     return x**2
        >>>
        >>> a=arange(10)
        >>> a.apply(square)
        FastArray([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])

        Example with lambda

        >>> a=arange(10)
        >>> a.apply(lambda x: x**2)
        FastArray([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])

        Example with numba

        >>> from numba import jit
        >>> @jit
        ... def squareit(x):
        ...     return x**2
        >>> a.apply(squareit)
        FastArray([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])

        Examples to use existing builtin oct function but change the output from string, to unicode, to object

        >>> a=arange(10)
        >>> a.apply(oct, otypes=['S'])
        FastArray([b'0o0', b'0o1', b'0o2', b'0o3', b'0o4', b'0o5', b'0o6', b'0o7', b'0o10', b'0o11'], dtype='|S4')

        >>> a=arange(10)
        >>> a.apply(oct, otypes=['U'])
        FastArray(['0o0', '0o1', '0o2', '0o3', '0o4', '0o5', '0o6', '0o7', '0o10', '0o11'], dtype='<U4')

        >>> a=arange(10)
        >>> a.apply(oct, otypes=['O'])
        FastArray(['0o0', '0o1', '0o2', '0o3', '0o4', '0o5', '0o6', '0o7', '0o10', '0o11'], dtype=object)

        """

        vfunc = np.vectorize(pyfunc, otypes=otypes, doc=doc, excluded=excluded, cache=cache, signature=signature)
        result = vfunc(self, *args)
        return result

    # -----------------------------------------------------------
    def apply_pandas(self, func, convert_dtype=True, args=(), **kwds):
        """
        Invoke function on values of FastArray. Can be ufunc (a NumPy function
        that applies to the entire FastArray) or a Python function that only works
        on single values

        Parameters
        ----------
        func : function
        convert_dtype : boolean, default True
            Try to find better dtype for elementwise function results. If
            False, leave as dtype=object
        args : tuple
            Positional arguments to pass to function in addition to the value
        Additional keyword arguments will be passed as keywords to the function

        Returns
        -------
        y : FastArray or Dataset if func returns a FastArray

        See Also
        --------
        FastArray.map: For element-wise operations
        FastArray.agg: only perform aggregating type operations
        FastArray.transform: only perform transforming type operations

        Examples
        --------
        Create a FastArray with typical summer temperatures for each city.

        >>> fa = rt.FastArray([20, 21, 12], index=['London', 'New York','Helsinki'])
        >>> fa
        London      20
        New York    21
        Helsinki    12
        dtype: int64

        Square the values by defining a function and passing it as an
        argument to ``apply()``.

        >>> def square(x):
        ...     return x**2
        >>> fa.apply(square)
        London      400
        New York    441
        Helsinki    144
        dtype: int64

        Square the values by passing an anonymous function as an
        argument to ``apply()``.

        >>> fa.apply(lambda x: x**2)
        London      400
        New York    441
        Helsinki    144
        dtype: int64

        Define a custom function that needs additional positional
        arguments and pass these additional arguments using the
        ``args`` keyword.

        >>> def subtract_custom_value(x, custom_value):
        ...     return x-custom_value
        >>> fa.apply(subtract_custom_value, args=(5,))
        London      15
        New York    16
        Helsinki     7
        dtype: int64

        Define a custom function that takes keyword arguments
        and pass these arguments to ``apply``.

        >>> def add_custom_values(x, **kwargs):
        ...     for month in kwargs:
        ...         x+=kwargs[month]
        ...     return x
        >>> fa.apply(add_custom_values, june=30, july=20, august=25)
        London      95
        New York    96
        Helsinki    87
        dtype: int64

        Use a function from the Numpy library.

        >>> fa.apply(np.log)
        London      2.995732
        New York    3.044522
        Helsinki    2.484907
        dtype: float64
        """
        import pandas as pd

        series = pd.Series(self)
        result = series.apply(func, convert_dtype=convert_dtype, args=args, **kwds)
        return result.values

    # -----------------------------------------------------------
    @cached_weakref_property
    def str(self) -> "FAString":
        r"""Casts an array of byte strings or unicode as ``FAString``.

        Enables a variety of useful string manipulation methods.

        Returns
        -------
        FAString

        Raises
        ------
        TypeError
            If the FastArray is of dtype other than byte string or unicode

        See Also
        --------
        np.chararray
        np.char
        rt.FAString.apply

        Examples
        --------
        >>> s=FA(['this','that','test ']*100_000)
        >>> s.str.upper
        FastArray([b'THIS', b'THAT', b'TEST ', ..., b'THIS', b'THAT', b'TEST '],
                  dtype='|S5')

        >>> s.str.lower
        FastArray([b'this', b'that', b'test ', ..., b'this', b'that', b'test '],
                  dtype='|S5')

        >>> s.str.removetrailing()
        FastArray([b'this', b'that', b'test', ..., b'this', b'that', b'test'],
                  dtype='|S5')
        """
        if self.dtype.char in "US":
            return TypeRegister.FAString(self)
        if self.dtype.char == "O":
            # try to convert to string (might have come from pandas)
            try:
                conv = self.astype("S")
            except:
                conv = self.astype("U")
            return TypeRegister.FAString(conv)

        raise TypeError(f"The .str function can only be used on byte string and unicode not {self.dtype!r}")

    @staticmethod
    def from_arrow(
        arr: Union["pa.Array", "pa.ChunkedArray"],
        zero_copy_only: bool = True,
        writable: bool = False,
        auto_widen: bool = False,
    ) -> "FastArray":
        """
        Convert a pyarrow `Array` to a riptable `FastArray`.

        Parameters
        ----------
        arr : pyarrow.Array or pyarrow.ChunkedArray
        zero_copy_only : bool, default True
            If True, an exception will be raised if the conversion to a `FastArray` would require copying the
            underlying data (e.g. in presence of nulls, or for non-primitive types).
        writable : bool, default False
            For a `FastArray` created with zero copy (view on the Arrow data), the resulting array is not writable (Arrow data is immutable).
            By setting this to True, a copy of the array is made to ensure it is writable.
        auto_widen : bool, optional, default to False
            When False (the default), if an arrow array contains a value which would be considered
            the 'invalid'/NA value for the equivalent dtype in a `FastArray`, raise an exception
            because direct conversion would be lossy / change the semantic meaning of the data.
            When True, the converted array will be widened (if possible) to the next-largest dtype
            to ensure the data will be interpreted in the same way.

        Returns
        -------
        FastArray
        """
        import pyarrow.types as pat

        # Based on the type of the array, dispatch to type-specific implementations of .from_arrow().
        pa_arr_type = arr.type
        if (
            pat.is_boolean(pa_arr_type)
            or pat.is_integer(pa_arr_type)
            or pat.is_floating(pa_arr_type)
            or pat.is_string(pa_arr_type)
            or pat.is_binary(pa_arr_type)
            or pat.is_fixed_size_binary(pa_arr_type)
        ):
            # TODO: Check whether this column has a user-specified fill value provided; if so, pass it along to
            #       the FastArray.from_arrow() method call below.
            return FastArray._from_arrow(arr, zero_copy_only=zero_copy_only, writable=writable, auto_widen=auto_widen)

        elif pat.is_dictionary(pa_arr_type) or pat.is_struct(pa_arr_type):
            return TypeRegister.Categorical._from_arrow(arr, zero_copy_only=zero_copy_only, writable=writable)

        elif pat.is_timestamp(pa_arr_type):
            return TypeRegister.DateTimeNano._from_arrow(arr, zero_copy_only=zero_copy_only, writable=writable)

        elif pat.is_date(pa_arr_type):
            return TypeRegister.Date._from_arrow(arr, zero_copy_only=zero_copy_only, writable=writable)

        elif pat.is_duration(pa_arr_type):
            return TypeRegister.TimeSpan._from_arrow(arr, zero_copy_only=zero_copy_only, writable=writable)

        else:
            # Unknown/unsupported array type -- can't convert.
            raise NotImplementedError(f"pyarrow arrays of type '{pa_arr_type}' can't be converted to riptable arrays.")

    @staticmethod
    def _from_arrow(
        arr: Union["pa.Array", "pa.ChunkedArray"],
        zero_copy_only: bool = True,
        writable: bool = False,
        auto_widen: bool = False,
    ) -> "FastArray":
        """
        Convert a pyarrow `Array` to a riptable `FastArray`.

        Parameters
        ----------
        arr : pyarrow.Array or pyarrow.ChunkedArray
        zero_copy_only : bool, default True
            If True, an exception will be raised if the conversion to a `FastArray` would require copying the
            underlying data (e.g. in presence of nulls, or for non-primitive types).
        writable : bool, default False
            For a `FastArray` created with zero copy (view on the Arrow data), the resulting array is not writable (Arrow data is immutable).
            By setting this to True, a copy of the array is made to ensure it is writable.
        auto_widen : bool, optional, default to False
            When False (the default), if an arrow array contains a value which would be considered
            the 'invalid'/NA value for the equivalent dtype in a `FastArray`, raise an exception
            because direct conversion would be lossy / change the semantic meaning of the data.
            When True, the converted array will be widened (if possible) to the next-largest dtype
            to ensure the data will be interpreted in the same way.

        Returns
        -------
        FastArray
        """
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.types as pat

        # Make sure the input array is one of the pyarrow array types.
        if not isinstance(arr, (pa.Array, pa.ChunkedArray)):
            raise TypeError("The array is not an instance of `pyarrow.Array` or `pyarrow.ChunkedArray`.")

        # ChunkedArrays need special handling.
        if isinstance(arr, pa.ChunkedArray):
            # A single-chunk ChunkedArray can be handled by just extracting that chunk
            # and recursively processing it.
            if arr.num_chunks == 1:
                return FastArray._from_arrow(
                    arr.chunk(0), zero_copy_only=zero_copy_only, writable=writable, auto_widen=auto_widen
                )
            else:
                # TODO: Benchmark this vs. using ChunkedArray.combine_chunks() then converting.
                # TODO: Look at `zero_copy_only` and `writable` -- the converted arrays could be destroyed while hstacking
                #       since we know they'll have just been created; this could reduce peak memory utilization.
                return hstack(
                    [
                        FastArray._from_arrow(
                            arr_chunk, zero_copy_only=zero_copy_only, writable=writable, auto_widen=auto_widen
                        )
                        for arr_chunk in arr.iterchunks()
                    ]
                )

        # Handle based on the type of the input array.
        if pat.is_integer(arr.type):
            # For arrays of primitive types, pa.DataType.to_pandas_dtype() actually returns the equivalent numpy dtype.
            arr_dtype = arr.type.to_pandas_dtype()

            # Get the riptable invalid value for this array type.
            arr_rt_inv = INVALID_DICT[np.dtype(arr_dtype).num]

            # Get min and max value of the input array, so we know if we need to promote
            # to the next-largest dtype to be able to correctly represent nulls.
            # This must be done even if the input array has no nulls, because otherwise the non-null
            # values in the input corresponding to riptable integer invalids would then be recognized
            # as such after conversion.
            min_max_result: pa.StructScalar = pc.min_max(arr)
            min_value = min_max_result["min"]
            max_value = min_max_result["max"]

            arr_pa_dtype_widened: Optional[pa.DataType] = None
            if min_value == arr_rt_inv or max_value == arr_rt_inv:
                # If the input array holds 64-bit integers (signed or unsigned), we can't do a lossless conversion,
                # since there is no wider integer available.
                if zero_copy_only:
                    raise ValueError(
                        "Cannot perform a zero-copy conversion of an arrow array containing the riptable invalid value for the array dtype."
                    )
                elif arr_dtype.itemsize == 8:
                    raise ValueError(
                        "Cannot losslessly convert an arrow array of (u)int64 containing the riptable invalid value to a riptable array."
                    )
                elif not auto_widen:
                    raise ValueError(
                        "Input array requires widening for lossless conversion. Specify auto_widen=True if you want to allow the widening conversion (which requires an array copy)."
                    )
                else:
                    # Widen the dtype of the output array.
                    output_dtype = np.min_scalar_type(2 * arr_rt_inv)
                    arr_pa_dtype_widened = pa.from_numpy_dtype(output_dtype)

            # Create the output array, performing a widening conversion + filling in nulls with the riptable invalid if necessary.
            # TODO: This could be faster -- if there's a way to get a numpy boolean array from a pyarrow array's null-mask,
            #       we can convert directly to numpy/riptable; then, widen the FastArray (which'll be parallelized);
            #       then use rt.copy_to() / rt.putmask() to overwrite the elements of the widened FastArray
            #       corresponding to the nulls from the mask with the riptable invalid value for the output array type.
            if arr_pa_dtype_widened is not None:
                arr: pa.Array = arr.cast(arr_pa_dtype_widened)

            return arr.fill_null(arr_rt_inv).to_numpy(zero_copy_only=False, writable=writable).view(FastArray)

        elif pat.is_floating(arr.type):
            # Floating-point arrays can be converted directly to numpy, since pyarrow will automatically
            # fill null values with NaN.
            return arr.to_numpy(zero_copy_only=zero_copy_only, writable=writable).view(FastArray)

        elif pat.is_boolean(arr.type):
            # Boolean arrays can only be converted when they do not contain nulls.
            # riptable does not support an 'invalid'/NA value for boolean, so pyarrow arrays
            # with nulls can't be represented in riptable.
            if arr.null_count == 0:
                return arr.to_numpy(zero_copy_only=zero_copy_only, writable=writable).view(FastArray)
            else:
                raise ValueError(
                    "riptable boolean arrays do not support an invalid value, so they cannot be created from pyarrow arrays containing nulls."
                )

        elif pat.is_string(arr.type) or pat.is_large_string(arr.type):
            # pyarrow variable-length string arrays can _never_ be zero-copy converted to fixed-length numpy/riptable arrays
            # because of differences in the memory layout.
            if zero_copy_only:
                raise ValueError(
                    "pyarrow variable-length string arrays cannot be zero-copy converted to riptable arrays."
                )

            # Check for whether the array contains only ASCII strings.
            # This is used to guide how the FastArray is created.
            has_unicode = not pc.all(pc.string_is_ascii(arr))

            # Convert the array to a numpy array.
            # Unfortunately, as of pyarrow 4.0, this conversion always produces a numpy object array containing the
            # strings (as Python strings) rather than a numpy string array.
            # We're able to handle this to return the sensible thing for riptable users, but it does mean this conversion
            # is slower than necessary right now.
            # TODO: Ask pyarrow-dev about implementing an option to return a numpy 'S' or 'U' array instead, it'll be
            #       much more efficient, even though some space will be wasted due to numpy not supporting variable-length strings.
            # TODO: Consider converting the pyarrow array to a dictionary-encoded array -- if there are only a few uniques,
            #       it'll be more efficient (even though doing more work) by avoiding repetitive creation of the Python string objects.
            if arr.null_count == 0:
                tmp = arr.to_numpy(zero_copy_only=False, writable=writable)

            else:
                # Need to fill nulls with an empty string before converting to numpy.
                # (INVALID_DICT[np.dtype('U').num] == '').
                tmp = arr.fill_null("").to_numpy(zero_copy_only=False, writable=writable)

            result = FastArray(tmp, dtype=str, unicode=has_unicode)
            if not writable:
                result.flags.writeable = False
            return result

        elif pat.is_fixed_size_binary(arr.type):
            null_count = arr.null_count
            if null_count != 0:
                if zero_copy_only:
                    raise ValueError(
                        "Can't perform a zero-copy conversion of a fixed-size binary array to riptable when the input array contains nulls."
                    )

                arr = arr.fill_null(
                    b"\x00" * arr.type.byte_width
                )  # can't fill with b"", since b"" is not valid for fixed width type

            # Calling pa.Array.to_numpy with zero_copy=True raises an error with fixed sized binary type.
            # Calling pa.Array.to_numpy with zero_copy=False returns a numpy array where types are python bytes objects.
            # Workaround below creates the numpy buffer of type "S" manually.
            buf = np.frombuffer(
                arr.buffers()[1],
                dtype="S" + str(arr.type.byte_width),
            )

            if writable and null_count == 0:  # already made a copy if null_count != 0
                result = FastArray(np.copy(buf))
                result.flags.writeable = writable
                return result

            result = FastArray(buf)
            result.flags.writeable = writable
            return result

        else:
            raise ValueError(
                f"FastArray cannot be created from a pyarrow array of type '{arr.type}'. You may need to call the `from_arrow` method on one of the derived subclasses instead."
            )

    def to_arrow(
        self,
        type: Optional["pa.DataType"] = None,
        *,
        preserve_fixed_bytes: bool = False,
        empty_strings_to_null: bool = True,
    ) -> Union["pa.Array", "pa.ChunkedArray"]:
        """
        Convert this `FastArray` to a `pyarrow.Array`.

        Parameters
        ----------
        type : pyarrow.DataType, optional, defaults to None
        preserve_fixed_bytes : bool, optional, defaults to False
            If this `FastArray` is an ASCII string array (dtype.kind == 'S'),
            set this parameter to True to produce a fixed-length binary array
            instead of a variable-length string array.
        empty_strings_to_null : bool, optional, defaults To True
            If this `FastArray` is an ASCII or Unicode string array,
            specify True for this parameter to convert empty strings to nulls in the output.
            riptable inconsistently recognizes the empty string as an 'invalid',
            so this parameter allows the caller to specify which interpretation
            they want.

        Returns
        -------
        pyarrow.Array or pyarrow.ChunkedArray

        Notes
        -----
        TODO: Add bool parameter which directs the conversion to choose the most-compact output type possible?
              This would be relevant to indices of categorical/dictionary-encoded arrays, but could also make sense
              for regular FastArray types (e.g. to use an int8 instead of an int32 when it'd be a lossless conversion).
        """
        import builtins

        import pyarrow as pa

        # Derived array types MUST implement their own overload of this function
        # for correctness; for that reason, raise an error if someone attempts to
        # call *this* implementation of the method for a derived array type.
        if builtins.type(self) != FastArray:
            raise NotImplementedError(
                f"The `{builtins.type(self).__qualname__}` type must implement it's own override of the `to_arrow()` method."
            )

        # riptable (at least as of 1.0.56) does not *truly* support invalid/NA values
        # in bool or ascii/unicode string-typed arrays. Handle those dtypes specially.
        if np.issubdtype(self.dtype, np.integer):
            # TODO: If this array has .ndims >= 2, need to convert to pyarrow using pa.Tensor.from_numpy(...). That doesn't handle masks as of pyarrow 4.0.

            # Get a mask of invalids.
            invalids_mask = self.isnan()

            # If all values are valid, don't bother creating an all-False mask, it's just wasting memory.
            if not invalids_mask.any():
                invalids_mask = None

            # Create the pyarrow array from it + this array.
            return pa.array(self._np, mask=invalids_mask, type=type)

        elif np.issubdtype(self.dtype, np.floating):
            # Using floating-point NaN to signal both NaN and NA/null in riptable means we
            # need to make a decision here on whether to mark those values as NA/null values
            # in the returned pyarrow array.
            # For now, we don't -- we just pass the data along directly, so the caller can decide
            # on whether they want to handle that. (Ideally, we'd have a way to parameterize this
            # but the protocol doesn't support it as of pyarrow 4.0. In any case, only the bitmask
            # needs to be re-created later if the user wants to consider the NaNs as NA/null values.)
            if self.ndim >= 2:
                # NOTE: As of pyarrow 4.0, this method doesn't support a `type` argument.
                return pa.Tensor.from_numpy(self._np)
            else:
                return pa.array(self._np, type=type)

        elif np.issubdtype(self.dtype, bool):
            if self.ndim >= 2:
                # NOTE: As of pyarrow 4.0, this method doesn't support a `type` argument.
                return pa.Tensor.from_numpy(self._np)
            else:
                return pa.array(self._np, type=type)

        elif np.issubdtype(self.dtype, bytes):
            # If the caller wants to convert empty strings to nulls, get a mask of invalids.
            if empty_strings_to_null:
                invalids_mask = self == self.inv

                # If all values are valid, don't bother creating an all-False mask, it's just wasting memory.
                if not invalids_mask.any():
                    invalids_mask = None
            else:
                invalids_mask = None

            # Does the caller want to preserve the fixed-length binary data?
            if preserve_fixed_bytes:
                # Convert to a fixed-length binary ('bytes') type.
                element_str_length = np.dtype(self.dtype).itemsize
                arr_type = pa.binary(element_str_length) if type is None else type
                return pa.array(self._np, mask=invalids_mask, type=arr_type)
            else:
                # Convert this array to a pyarrow variable-length string array.
                if type is None:
                    type = pa.string()
                # Convert as Unicode ndarray to stringarray, as bytestring does not preserve element length (riptable#249)
                return pa.array(np.array(self._np, dtype="U"), mask=invalids_mask, type=type)

        elif np.issubdtype(self.dtype, str):
            # If the caller wants to convert empty strings to nulls, get a mask of invalids.
            if empty_strings_to_null:
                invalids_mask = self == self.inv

                # If all values are valid, don't bother creating an all-False mask, it's just wasting memory.
                if not invalids_mask.any():
                    invalids_mask = None
            else:
                invalids_mask = None

            # pyarrow (as of v4.0) does not have a fixed-size Unicode string data type, so unlike the 'bytes'
            # handling above for ASCII strings, we have to use the variable-length string array type.
            if type is None:
                type = pa.string()
            return pa.array(self._np, mask=invalids_mask, type=type)

        else:
            raise NotImplementedError(f"FastArray with dtype '{np.dtype(self.dtype)}' is not supported.")

    def __arrow_array__(self, type: Optional["pa.DataType"] = None) -> Union["pa.Array", "pa.ChunkedArray"]:
        """
        Implementation of the ``__arrow_array__`` protocol for conversion to a pyarrow array.

        Parameters
        ----------
        type : pyarrow.DataType, optional, defaults to None

        Returns
        -------
        pyarrow.Array or pyarrow.ChunkedArray

        Notes
        -----
        https://arrow.apache.org/docs/python/extending_types.html#controlling-conversion-to-pyarrow-array-with-the-arrow-array-protocol
        """
        return self.to_arrow(type=type, preserve_fixed_bytes=False, empty_strings_to_null=True)

    # -----------------------------------------------------------
    @classmethod
    def register_function(cls, name, func):
        """
        Used to register functions to FastArray.
        Used by rt_fastarraynumba
        """
        setattr(cls, name, func)

    def apply_schema(self, schema):
        """
        Apply a schema containing descriptive information to the FastArray

        :param schema: dict
        :return: dictionary of deviations from the schema
        """
        from .rt_meta import apply_schema as _apply_schema

        return _apply_schema(self, schema)

    def info(self, **kwargs):
        """
        Print a description of the FastArray's contents
        """
        from .rt_meta import info as _info

        return _info(self, **kwargs)

    @property
    def doc(self):
        """
        The Doc object for the structure
        """
        from .rt_meta import doc as _doc

        return _doc(self)

    # ====================== END OF CLASS DEFINITION ===============================


# -----------------------------------------------------------
def _setfastarrayview(arr):
    """
    Call from CPP into python to flip array view
    """
    if isinstance(arr, FastArray):
        if FastArray.Verbose > 2:
            print("no need to setfastarrayview", arr.dtype, len(arr))
        return arr

    if FastArray.Verbose > 2:
        print("setfastarrayview", arr.dtype, len(arr))

    return arr.view(FastArray)


# -----------------------------------------------------------
def _setfastarraytype():
    # -----------------------------------------------------------
    # calling this function will force fm to return FastArray subclass
    # rc.BasicMathHook(FastArray, np.ndarray)
    # Coming next build
    fa = np.arange(1).view(FastArray)
    rc.SetFastArrayType(fa, _setfastarrayview)
    rc.BasicMathHook(fa, fa._np)


# -----------------------------------------------------------
def _FixupDocStrings():
    """
    Load all the member function of this module
    Load all the member functions of the np module
    If we find a match, copy over the doc strings
    """
    import inspect
    import sys

    mymodule = sys.modules[__name__]
    all_myfunctions = inspect.getmembers(FastArray, inspect.isfunction)

    try:
        # bottleneck is optional
        all_bnfunctions = inspect.getmembers(bn, inspect.isfunction)
        all_bnfunctions += inspect.getmembers(bn, inspect.isbuiltin)

        # build dictionary of bottleneck docs
        bndict = {}
        for funcs in all_bnfunctions:
            bndict[funcs[0]] = funcs[1]

        # now for each function that has an bn flavor, copy over the doc strings
        for funcs in all_myfunctions:
            if (funcs[0] in bndict) and (funcs[1].__doc__ is None):
                funcs[1].__doc__ = bndict[funcs[0]].__doc__
    except Exception:
        pass

    all_npfunctions = [func for func in inspect.getmembers(np.ndarray) if not func[0].startswith("_")]

    # build dictionary of np.ndarray docs
    npdict = {}
    for funcs in all_npfunctions:
        npdict[funcs[0]] = funcs[1]

    # now for each function that has an np flavor, copy over the doc strings
    for funcs in all_myfunctions:
        if (funcs[0] in npdict) and (funcs[1].__doc__ is None):
            funcs[1].__doc__ = npdict[funcs[0]].__doc__

    # now do just plain np
    all_npfunctions = [func for func in inspect.getmembers(np) if "__" not in funcs[0]]

    # build dictionary of np docs
    npdict = {}
    for funcs in all_npfunctions:
        # print("getting doc string for ", funcs[0])
        npdict[funcs[0]] = funcs[1]

    # now for each function that has an np flavor, copy over the doc strings
    for funcs in all_myfunctions:
        if (funcs[0] in npdict) and (funcs[1].__doc__ is None):
            funcs[1].__doc__ = npdict[funcs[0]].__doc__


# ----------------------------------------------------------
class Threading:
    @staticmethod
    def on():
        """
        Turn riptable threading on.
        Used only when riptable threading was turned off.

        Example
        -------
        a=rt.arange(1_000_00)
        Threading.off()
        %time a+=1
        Threading.on()
        %time a+=1

        Returns
        -------
        Previously whether threading was on or not. 0 or 1. 0=threading was off before.

        """
        return FastArray._TON()

    @staticmethod
    def off():
        """
        Turn riptable threading off.
        Useful for when the system has other processes using other threads
        or to limit threading resources.

        Example
        -------
        a=rt.arange(1_000_00)
        Threading.off()
        %time a+=1
        Threading.on()
        %time a+=1

        Returns
        -------
        Previously whether threading was on or not. 0 or 1. 0=threading was off before.
        """
        return FastArray._TOFF()

    @staticmethod
    def threads(threadcount):
        """
        Set how many worker threads riptable can use.
        Often defaults to 12 and cannot be set below 1 or > 31.

        To turn riptable threading off completely use Threading.off()
        Useful for when the system has other processes using other threads
        or to limit threading resources.

        Example
        -------
        Threading.threads(8)

        Returns
        -------
        number of threads previously used
        """
        return rc.SetThreadWakeUp(threadcount)


# ----------------------------------------------------------
class Recycle:
    @staticmethod
    def on():
        """
        Turn riptable recycling on.
        Used only when riptable recycling was turned off.

        Example
        -------
        a=arange(1_000_00)
        Recycle.off()
        %timeit a=a + 1
        Recycle.on()
        %timeit a=a + 1

        """
        return FastArray._RON()

    @staticmethod
    def off():
        return FastArray._ROFF()

    @staticmethod
    def now(timeout: int = 0):
        """
        Pass the garbage collector timeout value to cleanup.
        Also calls the python garbage collector.

        Parameters
        ----------
        timeout: default to 0.  0 will not set a timeout

        Returns
        -------
        total arrays deleted
        """
        import gc

        gc.collect()
        result = rc.RecycleGarbageCollectNow(timeout)["TotalDeleted"]
        if result > 0:
            rc.RecycleGarbageCollectNow(timeout)
        return result

    @staticmethod
    def timeout(timeout: int = 100):
        """
        Pass the garbage collector timeout value to expire.
        The timeout value is roughly in 2/5 secs.
        A value of 100 is usually about 40 seconds.
        If an array has not been reused by the timeout, it is permanently deleted.

        Returns
        -------
        previous timespan
        """
        return rc.RecycleSetGarbageCollectTimeout(timeout)


# ----------------------------------------------------------
class Ledger:
    @staticmethod
    def on():
        """Turn the math ledger on to record all array math routines"""
        return TypeRegister.MathLedger._LedgerOn()

    @staticmethod
    def off():
        """Turn the math ledger off"""
        return TypeRegister.MathLedger._LedgerOff()

    @staticmethod
    def dump(dataset=True):
        """Print out the math ledger"""
        return TypeRegister.MathLedger._LedgerDump(dataset=dataset)

    @staticmethod
    def to_file(filename):
        """Save the math ledger to a file"""
        return TypeRegister.MathLedger._LedgerDumpFile(filename)

    @staticmethod
    def clear():
        """Clear all the entries in the math ledger"""
        return TypeRegister.MathLedger._LedgerClear()


# ----------------------------------------------------------
# this is called when the module is loaded
_FixupDocStrings()

# NOTE: Keep this at the end of the file
# -----------------------------------------------------------
# calling this function will force fm to return FastArray subclass
_setfastarraytype()

TypeRegister.FastArray = FastArray

FastArray.register_function("describe", describe)
