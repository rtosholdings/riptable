"""
Functions for working with numba.

Functionality includes:

- Functions for numba <-> riptable interop
- Convenience functions and decorators used when implementing numba-accelerated functions.

Notes
-----
Many functions in this module implement both a standard Python version and
an overload using `numba.extending.overload`. This is done to allow the code
to work when ``NUMBA_DISABLE_JIT=1`` is specified on the command line, e.g.
to allow functions to be debugged.
"""
__all__ = ["get_invalid", "get_max_valid", "get_min_valid", "is_valid"]

import numpy as np
import numpy.typing as npt
import numba as nb
from numba.extending import overload


def is_valid(x) -> bool:
    """
    A function for checking if data is valid.  This works for both floats and integers.

    - For floats, the invalid is NaN.
    - For signed integers, the invalid is the most NEGATIVE value of the type.
    - For unsigned integers, the invalid is the most POSITIVE value of the type.

    Parameters
    ----------
    x
        The value to check

    Returns
    -------
    A bool for whether the data is valid.
    """
    if x is None:
        return False

    x_dtype = x.dtype
    if np.issubdtype(x_dtype, np.floating):
        return not np.isnan(x)

    elif np.issubdtype(x_dtype, np.integer):
        dtype_iinfo = np.iinfo(x_dtype)
        inv_val = dtype_iinfo.min if np.issubdtype(x_dtype, np.signedinteger) else dtype_iinfo.max
        return x != inv_val

    elif np.issubdtype(x_dtype, bool) or np.issubdtype(x_dtype, np.bytes_) or np.issubdtype(x_dtype, str):
        return True

    else:
        raise TypeError(f"Valid detection has not been implemented for the type '{type(x)}'.")


@overload(is_valid)
def _is_valid_nb(x):
    """
    A function for checking if data is valid.  This works for both floats and integers.

    - For floats, the invalid is NaN.
    - For signed integers, the invalid is the most NEGATIVE value of the type.
    - For unsigned integers, the invalid is the most POSITIVE value of the type.

    Parameters
    ----------
    x
        The value to check

    Returns
    -------
    A bool for whether the data is valid.
    """
    if isinstance(x, nb.types.Float):
        return lambda x: not np.isnan(x)

    elif isinstance(x, nb.types.Integer):
        if x.signed:
            inv_val = x.minval
        else:
            inv_val = x.maxval
        return lambda x: x != inv_val

    elif isinstance(x, nb.types.NoneType):
        return lambda x: False

    else:
        raise TypeError("Valid detection has not been implemented for this type.")


def get_invalid(x):
    """
    A function for getting the invalid for a type of element.

    This works for both floats and integers.

    - For floats, the invalid is NaN.
    - For signed integers, the invalid is the most NEGATIVE value of the type.
    - For unsigned integers, the invalid is the most POSITIVE value of the type.

    For arrays, the invalid of the dtype of the array is returned.

    Parameters
    ----------
    x
        An element of the type you want the invalid for

    Returns
    -------
    The invalid value for `x`'s type/dtype.
    """
    x_dtype = x.dtype

    if np.issubdtype(x_dtype, np.floating):
        return np.nan

    elif np.issubdtype(x_dtype, np.integer):
        dtype_iinfo = np.iinfo(x_dtype)
        return dtype_iinfo.min if np.issubdtype(x_dtype, np.signedinteger) else dtype_iinfo.max

    else:
        raise TypeError(f'No invalid has not been implemented for type "{x}".')


@overload(get_invalid)
def _get_invalid_nb(x):
    """
    A numba function for getting the invalid for a type of element.

    This works for both floats and integers.

    - For floats, the invalid is NaN.
    - For signed integers, the invalid is the most NEGATIVE value of the type.
    - For unsigned integers, the invalid is the most POSITIVE value of the type.

    For arrays, the invalid of the dtype of the array is returned.

    Parameters
    ----------
    x
        An element of the type you want the invalid for

    Returns
    -------
    The invalid value
    """
    # If an array, get it's dtype
    if isinstance(x, nb.types.Array):
        x = x.dtype

    if isinstance(x, nb.types.Float):
        return lambda x: np.nan

    elif isinstance(x, nb.types.Integer):
        if x.signed:
            ret_val = x.minval
        else:
            ret_val = x.maxval
        return lambda x: ret_val

    elif isinstance(x, nb.types.NoneType):
        ret_val = nb.int64.minval
        return lambda x: ret_val

    else:
        raise TypeError(f'No invalid has not been implemented for type "{x}".')


def get_max_valid(x):
    """
    Get the maximum valid value for the dtype of an array or scalar.

    This function supports integer and floating-point values.

    Parameters
    ----------
    x
        An array or scalar value of the type you want the maximum valid value for.

    Returns
    -------
    max_valid
        The maximum valid value for `x`'s dtype.
    """
    x_dtype = x.dtype

    if np.issubdtype(x_dtype, np.floating):
        fi = np.finfo(x_dtype)
        if fi.bits == 64:
            return np.finfo(np.float64).max
        elif fi.bits == 32:
            return np.finfo(np.float32).max
        elif fi.bits == 16:
            return np.finfo(np.float16).max

    elif np.issubdtype(x_dtype, np.integer):
        dtype_iinfo = np.iinfo(x_dtype)
        return dtype_iinfo.max if np.issubdtype(x_dtype, np.signedinteger) else dtype_iinfo.max - 1

    elif np.issubdtype(x_dtype, bool):
        return True

    else:
        raise TypeError(f'get_max_valid() has not been implemented for type "{x}".')


@overload(get_max_valid)
def _get_max_valid_nb(x):
    """
    Get the maximum valid value for the dtype of an array or scalar.

    This function supports integer and floating-point values.

    Parameters
    ----------
    x
        An array or scalar value of the type you want the maximum valid value for.

    Returns
    -------
    max_valid
        The maximum valid value for `x`'s dtype.
    """
    # If an array, get it's dtype
    if isinstance(x, nb.types.Array):
        x = x.dtype

    if isinstance(x, nb.types.Float):
        if x.bitwidth == 64:
            ret_val = np.finfo(np.float64).max
        elif x.bitwidth == 32:
            ret_val = np.finfo(np.float32).max
        elif x.bitwidth == 16:
            ret_val = np.finfo(np.float16).max
        return lambda x: ret_val

    elif isinstance(x, nb.types.Integer):
        ret_val = x.maxval if x.signed else x.maxval - 1
        return lambda x: ret_val

    elif isinstance(x, nb.types.Boolean):
        return lambda x: True

    else:
        raise TypeError(f'_get_max_valid_nb() has not been implemented for type "{x}".')


def get_min_valid(x):
    """
    Get the minimum valid value for the dtype of an array or scalar.

    Parameters
    ----------
    x
        An array or scalar value of the type you want the minimum valid value for.

    Returns
    -------
    min_valid
        The minimum valid value for `x`'s dtype.
    """
    x_dtype = x.dtype

    if np.issubdtype(x_dtype, np.floating):
        fi = np.finfo(x_dtype)
        if fi.bits == 64:
            return np.finfo(np.float64).min
        elif fi.bits == 32:
            return np.finfo(np.float32).min
        elif fi.bits == 16:
            return np.finfo(np.float16).min

    elif np.issubdtype(x_dtype, np.integer):
        dtype_iinfo = np.iinfo(x_dtype)
        return dtype_iinfo.min + 1 if np.issubdtype(x_dtype, np.signedinteger) else dtype_iinfo.max

    elif np.issubdtype(x_dtype, bool):
        return False

    elif np.issubdtype(x_dtype, np.bytes_):
        return b""

    elif np.issubdtype(x_dtype, str):
        return ""

    else:
        raise TypeError(f'get_min_valid() has not been implemented for type "{x}".')


@overload(get_min_valid)
def _get_min_valid_nb(x):
    """
    Get the minimum valid value for the dtype of an array or scalar.

    Parameters
    ----------
    x
        An array or scalar value of the type you want the minimum valid value for.

    Returns
    -------
    min_valid
        The minimum valid value for `x`'s dtype.
    """
    # If an array, get it's dtype
    if isinstance(x, nb.types.Array):
        x = x.dtype

    if isinstance(x, nb.types.Float):
        if x.bitwidth == 64:
            ret_val = np.finfo(np.float64).min
        elif x.bitwidth == 32:
            ret_val = np.finfo(np.float32).min
        elif x.bitwidth == 16:
            ret_val = np.finfo(np.float16).min
        return lambda x: ret_val

    elif isinstance(x, nb.types.Integer):
        ret_val = x.minval + 1 if x.signed else x.minval
        return lambda x: ret_val

    elif isinstance(x, nb.types.Boolean):
        return lambda x: False

    elif isinstance(x, nb.types.Bytes):
        return lambda x: b""

    elif isinstance(x, nb.types.UnicodeType):
        return lambda x: ""

    else:
        raise TypeError(f'_get_min_valid_nb() has not been implemented for type "{x}".')
