__all__ = [
    # types listed first
    'int16', 'int32', 'int64', 'int8', 'int0', 'uint0', 'bool_', 'bytes_', 'str_',
    'float32', 'float64', 'uint16', 'uint32', 'uint64', 'uint8',
    # functions
    'absolute', 'abs', 'all', 'any', 'arange', 'argsort', 'asanyarray', 'asarray', 'assoc_copy', 'assoc_index',
    'bincount', 'bitcount', 'bool_to_fancy',
    'cat2keys', 'ceil', 'combine2keys', 'concatenate', 'crc32c', 'crc64', 'cumsum',
    'combine_filter', 'combine_accum1_filter', 'combine_accum2_filter',
    'diff', 'double', 'empty', 'empty_like',
    'floor', 'full',
    'get_dtype', 'get_common_dtype', 'groupby', 'groupbyhash', 'groupbylex', 'groupbypack',
    'hstack',
    'isfinite', 'isnotfinite', 'isinf', 'isnotinf', 'ismember', 'isnan', 'isnanorzero', 'isnotnan', 'issorted', 'interp', 'interp_extrap',
    'lexsort', 'logical', 'log', 'log10',
    'makeifirst', 'makeilast', 'makeinext', 'makeiprev', 'max', 'mean', 'median', 'min', 'multikeyhash', 'minimum', 'maximum',
    'mask_or', 'mask_and', 'mask_xor', 'mask_andnot',
    'mask_ori', 'mask_andi', 'mask_xori', 'mask_andnoti',
    'nan_to_num', 'nan_to_zero', 'nanargmin', 'nanargmax',
    'nanmax', 'nanmean', 'nanmedian', 'nanmin', 'nanpercentile', 'nanstd', 'nansum', 'nanvar',
    'ones', 'ones_like',
    'percentile', 'putmask',
    'reindex_fast', 'reshape', 'round',
    'searchsorted', '_searchsorted', 'single', 'sort', 'sortinplaceindirect', 'std', 'sum',
    'tile', 'transpose', 'trunc',
    'unique32',
    'var', 'vstack',
    'where',
    'zeros', 'zeros_like',
]

import sys
import builtins
from typing import Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np
import inspect
import warnings

import riptide_cpp as rc
from riptide_cpp import LedgerFunction
from .rt_enum import INVALID_DICT, TypeRegister, REDUCE_FUNCTIONS, MATH_OPERATION, NumpyCharTypes, CategoryMode


if TYPE_CHECKING:
    from .rt_categorical import Categorical
    from .rt_fastarray import FastArray
    from .rt_struct import Struct

#--------------------------------------------------------------
def get_dtype(val):
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
    if a python integer, will use int32 or int64 (never uint)
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
    try:
        # check first if it has a dtype
        final = val.dtype
    except:
        if isinstance(val,bool):
            final = np.dtype(np.bool)

        elif isinstance(val, int):
            val = abs(val)
            if val <= (2**32-1): final=np.dtype(np.int32)
            elif val <= (2**64-1): final=np.dtype(np.int64)
            else: final=np.dtype(np.float64)

        elif isinstance(val,float):
            final = np.dtype(np.float64)

        elif isinstance(val, str):
            final = np.dtype('U'+str(len(val)))

        elif isinstance(val, bytes):
            final = np.dtype('S'+str(len(val)))

        else:
            temp = np.asanyarray(val)
            final = temp.dtype

    return final

#--------------------------------------------------------------
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

    >>> get_common_type(arange(10).astype(np.bool), True)
    dtype('bool')
    """
    type1 = get_dtype(x)
    type2 = get_dtype(y)

    # NOTE: find_common_type has a bug where int32 num 7 gets flipped to int32 num 5.
    if type1.num != type2.num:
        common = np.find_common_type([type1,type2],[])
    else:
        # for strings and unicode, pick the larger itemsize
        if type1.itemsize >= type2.itemsize:
            common = type1
        else:
            common = type2

    if common.char != 'O':
        return common

    #situation where we have a string but find_common_type flips to object too easily
    if type1.char in 'SU':
        # get itemsize
        itemsize1=type1.itemsize
        if type1.char == 'U':
            itemsize1 = itemsize1 //4
        if type2.char in 'SU':
            itemsize2=type2.itemsize
            if type2.char == 'U':
                itemsize2 = itemsize2 //4

            # get the max of the two strings
            maxsize = str(max(itemsize1,itemsize2))
            if type1.char == 'U' or type2.char == 'U':
                common = np.dtype('U'+maxsize)
            else:
                common = np.dtype('S'+maxsize)

        # 13 is long double
        # 14,15,16 CFLOAT, CDOUBLE, CLONGDOUBLE
        # 17 is object
        # 18 = ASCII string
        # 19 = UNICODE string
        elif type2.num <= 13:
            # handle case where we have an int/float/bool and a string
            if type2.num <= 10:
                maxsize = str(max(itemsize1,16))
            else:
                maxsize = str(max(itemsize1,32))

            if type1.char == 'U':
                common = np.dtype('U'+maxsize)
            else:
                common = np.dtype('S'+maxsize)

    elif type2.char in 'SU':
        if type1.num <= 13:
            # handle case where we have an int/float/bool and a string
            # get itemsize
            itemsize2=type2.itemsize
            if type2.char == 'U':
                itemsize2 = itemsize2 //4

            if type1.num <= 10:
                maxsize = str(max(itemsize2,16))
            else:
                maxsize = str(max(itemsize2,32))

            if type2.char == 'U':
                common = np.dtype('U'+maxsize)
            else:
                common = np.dtype('S'+maxsize)

    return common


def empty(shape, dtype: Union[str, np.dtype, type] = np.float, order: str = 'C') -> 'FastArray':
    #return LedgerFunction(np.empty, shape, dtype=dtype, order=order)

    # make into list of ints
    try:
        shape = [int(k) for k in shape]
    except:
        shape = [int(shape)]

    dtype= np.dtype(dtype)

    # try to use recycler
    result= rc.Empty(shape, dtype.num, dtype.itemsize, order=='F')
    if result is None:
        return LedgerFunction(np.empty, shape, dtype=dtype, order=order)
    else:
        return result


def empty_like(
    array: np.ndarray,
    dtype: Optional[Union[str, np.dtype, type]] = None,
    order: str = 'K',
    subok: bool = True,
    shape: Optional[Union[int, Sequence[int]]] = None
) -> 'FastArray':
    # TODO: call recycler

    # NOTE: np.empty_like preserves the subclass
    if isinstance(array, TypeRegister.FastArray):
        array=array._np
    result = LedgerFunction(np.empty_like, array, dtype=dtype, order=order, subok=subok, shape=shape)
    return result

#-------------------------------------------------------
def _searchsorted(array, v, side='left', sorter=None):

    # we cannot handle a sorter
    if sorter is None:
        try:
            if side == 'leftplus':
                return rc.BinsToCutsBSearch(v, array, 0)
            elif side == 'left':
                return rc.BinsToCutsBSearch(v, array, 1)
            else:
                return rc.BinsToCutsBSearch(v, array, 2)
        except:
            # fall into numpy
            pass

    # numpy does not like fastarrays for this routine
    if isinstance(array, TypeRegister.FastArray):
        array=array._np
    return LedgerFunction(np.searchsorted, array, v, side=side, sorter=sorter)

#-------------------------------------------------------
def searchsorted(a, v, side='left', sorter=None):
    """ see np.searchsorted
        side ='leftplus' is a new option in riptable where values > get a 0
    """
    return _searchsorted(a, v, side=side, sorter=sorter)

#-------------------------------------------------------
def issorted(*args,**kwargs):
    """
    Examples
    --------
    rt.arange(10).issorted()
    """
    return LedgerFunction(rc.IsSorted,*args,**kwargs)

#-------------------------------------------------------
def unique(
    arr: Union[np.ndarray, List[np.ndarray]],
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    sorted: bool = True,
    lex: bool = False,
    dtype: Optional[Union[str, np.dtype]] = None,
    filter: Optional[np.ndarray] = None
) -> Union['FastArray', Tuple['FastArray', ...], List['FastArray'], tuple]:
    """
    Find the unique elements of an array.

    Parameters
    ----------
    arr : array_like or list of array_like
        Input array, or list of arrays (a multikey). If a list is provided, all arrays
        in the list must have the same shape.
    return_index : bool, optional
        If True, also return the indices of `arr` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.
    return_inverse : bool, optional
        If True, also return the indices of the unique array (for the specified
        axis, if provided) that can be used to reconstruct `arr`.
    return_counts : bool, optional
        If True, also return the number of times each unique item appears in `arr`.
    sorted : bool
        Indicates whether the results are returned in sorted order.
        Defaults to True, which replicates the behavior of the numpy version of this function.
        When `lex` is set to True, the value of this parameter is ignored and the
        results are always returned in sorted order.
        Same as ordered in Categorical sorted=False is often faster.
    lex : bool
        Controls whether the function uses hashing- or sorting-based logic to find
        the unique values in `arr`. Defaults to False (hashing), set to True to
        use a lexicographical sort instead; this can be faster when `arr` is a large
        array with a relatively high proportion of unique values.
    dtype : numpy dtype, optional
        If provided the index will be returned in the dtype.
    filter: ndarray of bool, optional
        If provided, any False values will be ignored in the calculation.
        If provided and return_index is True, a filtered out location will be -1.

    Returns
    -------
    the unique values or a list of unique values (if multiple arrays passed)
    (optionally the index location)
    (optionally how often each value occurs)

    Notes
    -----
    riptable unique often performs faster than ``np.unique`` for strings and numeric types.
    Categoricals passed in as `arr` will ignore the `sorted` flag and return their current order.

    Examples
    --------
    >>> rt.unique(['b','b','a','d','d'])
    FastArray(['a', 'b', 'd'], dtype='<U1')

    >>> rt.unique(['b','b','a','d','d'], sorted=False)
    FastArray(['b', 'a', 'd'], dtype='<U1')

    >>> rt.unique([['b','b','a','d','d'],['b','b','c','d','d']])
    [FastArray(['a', 'b', 'd'], dtype='<U1'),
     FastArray(['c', 'b', 'd'], dtype='<U1')]

    >>> rt.unique([['b','b','a','d','d'],['b','b','c','d','d']], sorted=False)
    [FastArray(['b', 'a', 'd'], dtype='<U1'),
     FastArray(['b', 'c', 'd'], dtype='<U1')]
    """
    if dtype is not None:
        if dtype not in NumpyCharTypes.AllInteger:
            dtype = None

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts
    mark_readonly = False

    if isinstance(arr, TypeRegister.Categorical):
        # NOTE if the categorical is not dirty, filter should do nothing
        # TODO: need to set dirty flag g=arr.filter().grouping
        g=arr.grouping

        # check for Dictionary mode
        if (arr.category_mode == CategoryMode.Dictionary):
            if filter is not None:
                arr=arr.filter(filter)
            else:
                if g.isdirty:
                    arr=arr.filter(filter)
                else:
                    mark_readonly = True

            # get back grouping in case it changed
            g=arr.grouping

        else:
            if filter is not None:
                g=g.regroup(filter)
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
                    arr=[TypeRegister.FastArray(v, unicode=True) for v in arr]
            else:
                arr = TypeRegister.FastArray(arr, unicode=True)

        # Grouping is faster than Categorical and preserves ifirstkey
        if lex is True or sorted is False:
            g=TypeRegister.Grouping(arr, lex=lex, filter=filter)
        else:
            # TODO: need flag to preserve ifirstkey when making a Categorical
            # TODO: or grouping needs to obey ordered flag (then don't need to make Categorical)
            g=TypeRegister.Categorical(arr, ordered=sorted, lex=lex, filter=filter).grouping

    un = g.uniquelist

    # check for multikey
    if len(un)==1:
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
            if g.base_index ==1:
                inv = inv -1

            if dtype is not None:
                inv = inv.astype(dtype, copy=False)
            ret += (inv,)

        # counts of each unique item in the original array
        if return_counts:
            ret += (counts,)

        ret = (un,) + ret
        return ret


def _possibly_match_categoricals(a: 'Categorical', b):
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
            raise TypeError(f"ismember on categorical must be with a categorical in the same mode. Got {a.category_mode} {b.category_mode}")
        else:
            mode = a.category_mode
            # multikey categoricals need to have the same number of columns
            # regular multikey ismember will check for matching dtypes within the dictionaries
            if mode == CategoryMode.MultiKey:
                adict = a.category_dict
                bdict = b.category_dict

                if len(adict) != len(bdict):
                    raise ValueError(f"Multikey dictionaries in ismember categorical did not have the same number of keys. {len(adict)} vs. {len(bdict)}")

            # if codes exist in both mappings, their values must be consistent
            elif mode in [ CategoryMode.Dictionary, CategoryMode.IntEnum ]:
                adict = a.category_mapping
                bdict = b.category_mapping

                match = True
                for code, aval in adict.items():
                    bval = bdict.get(code,None)
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
            elif mode in [ CategoryMode.StringArray, CategoryMode.NumericArray ]:
                pass

            else:
                raise NotImplementedError

    # it's faster to make a categorical than to reexpand
    # turn other array argument into categorical before performing final ismember
    elif a.category_mode == CategoryMode.StringArray:
        if b.dtype.char in ('U','S'):
            # future optimization: don't sort the bins when making the throwaway categorical
            #if a.unique_count < 30_000_000 and TypeRegister.Categorical.TestIsMemberVerbose == True:
            #    _, idx = ismember(b, a.category_array)
            #    if a.base_index == 1:
            #        idx += 1
            #    return a._fa, idx
            unicode=False
            if b.dtype.char == 'U':
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
        raise TypeError(f"Could not perform ismember on categorical in {a.category_mode.name} and array with dtype {b.dtype}")

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
                raise TypeError(f'Multikey ismember could not re-expand array for categorical in mode {col.category_mode.name}.')
            col = col.expand_array
        return col

    allowed_int = 'bhilqpBHILQP'
    allowed_float = 'fdg'
    allowed_types = allowed_int + allowed_float

    # make sure original container items don't get blown away during fixup
    if isinstance(a, tuple): a = list(a)
    if isinstance(b, tuple): b = list(b)
    if isinstance(a, list): a = a.copy()
    if isinstance(b, list): b = b.copy()

    for idx, a_col in enumerate(a):
        b_col = b[idx]
        a_col = _as_fastarrays(a_col)
        b_col = _as_fastarrays(b_col)

        a_char = a_col.dtype.char
        b_char = b_col.dtype.char
        # if a column was string, need to match string width in b
        if a_char in 'US':
            if b_char in 'US':
                # TODO: find a prettier way of doing this...
                if a_char != b_char:
                    # if unicode is present (probably rare), need to upcast both
                    if a_char == 'U':
                        a_width = a_col.itemsize // 4
                        b_width = b_col.itemsize
                    else:
                        a_width = a_col.itemsize
                        b_width = b_col.itemsize // 4
                    dtype_letter = 'U'

                # both unicode or both bytes, just match width
                else:
                    dtype_letter = a_char
                    if dtype_letter == 'U':
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
                raise TypeError(f'Could not perform multikey ismember on types {a_col.dtype} and {b_col.dtype}')

        else:
            # make sure both are supported numeric types
            if a_char not in allowed_types:
                raise TypeError(f"{a_col.dtype} not in allowed types for ismember with {b_col.dtype}")
            if b_char not in allowed_types:
                raise TypeError(f"{b_col.dtype} not in allowed types for ismember with {a_col.dtype}")

            # cast if necessary
            if a_char != b_char:
                #warnings.warn(f"Performance warning: numeric arrays in ismember had different dtypes {a.dtype} {b.dtype}")
                #raise TypeError('numeric arrays in ismember need to be the same dtype')
                common_type = np.find_common_type([a_col.dtype,b_col.dtype],[])
                a_col = a_col.astype(common_type, copy=False)
                b_col = b_col.astype(common_type, copy=False)

        a[idx] = a_col
        b[idx] = b_col

    return a, b


def ismember(a, b, h=2, hint_size: int = 0, base_index: int = 0) -> Tuple[Union[int, 'FastArray'], 'FastArray']:
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
    allowed_int = 'bhilqpBHILQP'
    allowed_float = 'fdg'
    allowed_types = allowed_int + allowed_float

    len_a = len(a)
    len_b = len(b)
    is_multikey = False

    if len_a == 0 or len_b == 0:
        return zeros(len(a), dtype=np.bool), full(len_a, INVALID_DICT[np.dtype(np.int32).num])

    if isinstance(a, TypeRegister.Categorical) or isinstance(b, TypeRegister.Categorical):
        if isinstance(a, TypeRegister.Categorical):
            if not isinstance(b, np.ndarray):
                b = TypeRegister.FastArray(b)
            a, b = _possibly_match_categoricals(a,b)

        if isinstance(b, TypeRegister.Categorical):
            if not isinstance(a, np.ndarray):
                a = TypeRegister.FastArray(a)
            if not isinstance(a, TypeRegister.Categorical):
                b, a = _possibly_match_categoricals(b,a)

        # re-expansion has happened, use regular ismember
        # enum/mapped categoricals with consistent mappings (but not necessarily the same ones) will take this path
        if not isinstance(a, TypeRegister.Categorical):
            return ismember(a,b)

        # special categorical ismember needs to be called
        if a.issinglekey or a.ismultikey:
            acats, bcats = list(a.category_dict.values()), list(b.category_dict.values())
            num_unique_b = len(bcats[0])
        else:
            raise NotImplementedError(f"Have not yet found a solution for ismember on categoricals in {a.category_mode.name} mode")

        _, on_unique = ismember(acats, bcats)
        #rc.IsMemberCategorical:
        #arg1 - underlying FastArray of a Categorical
        #arg2 - underlying FastArray of b Categorical
        #arg3 - first occurrence of a's uniques into b's uniques
        #arg4 - number of unique in b
        #arg5 - a base index
        #arg6 - b base index
        b, f = rc.IsMemberCategoricalFixup( a._fa, b._fa, on_unique.astype(np.int32), int(num_unique_b), a.base_index, b.base_index )
        return b,f

    # a and b contain list like, probably a multikey
    if (isinstance(a[0], (np.ndarray, list, tuple)) and
        isinstance(b[0], (np.ndarray, list, tuple))):
        is_multikey = True

    # different number of key columns
    if is_multikey:
        if len_a == len_b:
            # single key "multikey", send through regular ismember
            if len_a == 1 and len_b == 1:
                return ismember(a[0], b[0], h)

            a, b = _ismember_align_multikey(a, b)

            return rc.MultiKeyIsMember32((a,), (b,), hint_size)
        else:
            raise ValueError(f"Multikey ismember must have the same number of keys in each item. a had {len_a}, b had {len_b}")

    # convert both to FastArray
    if isinstance(a, (list, tuple)):
        a = TypeRegister.FastArray(a)
    if isinstance(b, (list, tuple)):
        b = TypeRegister.FastArray(b)

    a_char = a.dtype.char
    b_char = b.dtype.char

    # handle strings
    if a_char in ('U','S'):
        if b_char in ('U', 'S'):
            # if the string types do not match, always try to use byte strings for the final operation
            if a_char != b_char:
                if a_char == 'U':
                    try:
                        a = a.astype('S')
                    except:
                        b = b.astype('U')
                else:
                    try:
                        b = b.astype('S')
                    except:
                        a = a.astype('U')
        else:
            raise TypeError(f"The first parameter is a string but the second parameter is not and cannot be compared. {a.dtype} vs. {b.dtype}")

    # will only be hit if a is not strings
    elif b_char in ('U', 'S'):
        raise TypeError(f"The second parameter is a string but the first parameter is not and cannot be compared. {a.dtype} vs. {b.dtype}")

    else:
        # make sure both are supported numeric types
        if a_char not in allowed_types:
            raise TypeError(f"{b.dtype} not in allowed types for ismember")
        if b_char not in allowed_types:
            raise TypeError(f"{b.dtype} not in allowed types for ismember")

        # cast if necessary
        if a_char != b_char:

            #import traceback
            #for line in traceback.format_stack():
            #    print(line.strip())

            #warnings.warn(f"Performance warning: numeric arrays in ismember had different dtypes {a.dtype} {b.dtype}")
            #raise TypeError('numeric arrays in ismember need to be the same dtype')
            common_type = np.find_common_type([a.dtype,b.dtype],[])
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


def assoc_index(key1: List[np.ndarray], key2: List[np.ndarray]) -> 'FastArray':
    """
    Parameters
    ----------
    key1 : list of ndarray
        List of numpy arrays to match against; all arrays must be same length.
    key2 : list of ndarray
        List of numpy arrays that will be matched with `key1`; all arrays must be same length.

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


def assoc_copy(key1: List[np.ndarray], key2: List[np.ndarray], arr: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    key1 : list of ndarray
        List of numpy arrays to match against; all arrays must be same length.
    key2 : list of ndarray
        List of numpy arrays that will be matched with `key1`; all arrays must be same length.
    arr : ndarray
        An array the same length as key2 arrays which will be mapped to the size of `key1`

    Returns
    -------
    array_like
        A new array the same length as `key1` arrays which has mapped the input `arr` from `key2` to `key1`
        the array's dtype will match the dtype of the input array (3rd parameter).

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
    fancyindex = assoc_index(key1,key2)
    return arr[fancyindex]


def unique32(list_keys: List[np.ndarray], hintSize: int = 0, filter: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Returns the index location of the first occurence of each key.

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

#-------------------------------------------------------
def combine_filter(key, filter):
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

#-------------------------------------------------------
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
    mkdict ={'iKey': iKey, 'iFirstKey':iFirstKey, 'unique_count': unique_count}
    return mkdict

#-------------------------------------------------------
def combine_accum2_filter(key1, key2, unique_count1:int, unique_count2:int, filter=None):
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

#-------------------------------------------------------
def combine2keys(key1, key2, unique_count1:int, unique_count2:int, filter=None):
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
    iKey, nCountGroup= rc.CombineAccum2Filter(key1, key2, unique_count1, unique_count2, filter)
    mkdict ={'iKey': iKey, 'nCountGroup': nCountGroup}
    return mkdict

#-------------------------------------------------------
def cat2keys(
    key1: Union['Categorical', np.ndarray, List[np.ndarray]],
    key2: Union['Categorical', np.ndarray, List[np.ndarray]],
    filter: Optional[np.ndarray] = None,
    ordered: bool = True,
    sort_gb: bool = False,
    invalid: bool = False,
    fuse: bool = False
) -> 'Categorical':
    """
    Create a `Categorical` from two keys or two `Categorical`s with all possible unique combinations.

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
            key1=Cat(key1, ordered=ordered, sort_gb=sort_gb)
    except Exception as e:
        warnings.warn(f"cat2keys: Got exception {e}", RuntimeWarning)

    if not isinstance(key1, Cat):
        raise TypeError(f"cat2keys: Argument 1 must be a categorical or an array that can be made into a categorical not type {type(key1)!r}")

    try:
        if not isinstance(key2, Cat):
            key2=Cat(key2, ordered=ordered, sort_gb=sort_gb)
    except Exception as e:
        warnings.warn(f"cat2keys: Got exception {e}", RuntimeWarning)

    if not isinstance(key2, Cat):
        raise TypeError(f"cat2keys: Argument 2 must be a categorical or an array that can be made into a categorical not type {type(key2)!r}")

    group_row = key1.grouping
    group_col = key2.grouping

    numrows = group_row.unique_count
    numcols = group_col.unique_count

    # have to check for ==0 first
    if not invalid:
        if np.sum(group_row.ikey==0) > 0 or np.sum(group_col.ikey==0) > 0:
            warnings.warn(f"catmatrix: Invalid found in key array, please use invalid=True to avoid this warning.")
            invalid = True
        else:
            # now we can remove the invalid and reassign
            ikey = group_col.ikey.astype(np.int64)-1
            # inplace operations for speed
            ikey *= numrows
            ikey += group_row.ikey

    if invalid:
        ikey=combine2keys(group_row.ikey, group_col.ikey, numrows, numcols, filter=filter)['iKey']

    # also check if the only want 2 keys with fuse
    if invalid or fuse:
        row_name, row_arr = group_row.onedict(invalid=invalid)
        col_name, col_arr = group_col.onedict(invalid=invalid)

        # handle case when same name
        if row_name == col_name: col_name = col_name+'1'
        if invalid:
            # invalid was inserted as first unique, so need to make room
            numrows +=1
            numcols +=1
            ikey +=1

        newgroup = TypeRegister.Grouping( ikey,  {row_name: row_arr.tile(numcols), col_name: col_arr.repeat(numrows)})

    else:
        # construct grouping object with a multikey
        gdict = dict()
        for k,v in group_row._grouping_unique_dict.items(): gdict[k]=v.tile(numcols)
        for k,v in group_col._grouping_unique_dict.items():
            # Handle column name conflicts (if present).
            if k in gdict:
                counter = 1
                origk = k
                # Suffix an integer to the original column name,
                # iterating until we find a column name that hasn't
                # been used yet.
                while k in gdict:
                    k=origk+str(counter)
                    counter += 1
            gdict[k]=v.repeat(numrows)
        newgroup = TypeRegister.Grouping( ikey,  gdict)

    # create the categorical from the grouping object
    result= Cat(newgroup)

    # save for later in case the categorical needs to be rectangularized like Accum2
    result._numrows = numrows
    result._numcols = numcols
    return result

#-------------------------------------------------------
def makeifirst(key, unique_count : int, filter=None) -> np.ndarray:
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

#-------------------------------------------------------
def makeilast(key, unique_count : int, filter=None) -> np.ndarray:
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


#-------------------------------------------------------
def makeinext(key, unique_count:int) -> np.ndarray:
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

#-------------------------------------------------------
def makeiprev(key, unique_count:int) -> np.ndarray:
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

#-------------------------------------------------------
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
    return LedgerFunction(rc.GroupByAll32,*args)

#-------------------------------------------------------
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

    return LedgerFunction(rc.GroupByAllPack32,*args)
#-------------------------------------------------------
#def _groupbycrunch(*args):
#    #return rc.GroupByOp32(*args)
#    return LedgerFunction(rc.GroupByOp32,*args)

#-------------------------------------------------------
def groupbypack(ikey, ncountgroup, unique_count=None, cutoffs=None):
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
    dnum =ikey.dtype.num
    if dnum not in [1,3,5,7,9]:
        raise ValueError("ikey must be int8, int16, int32, or int64")

    if cutoffs is None:
        #
        # new routine written Oct, 2019
        #
        if ncountgroup is None:
            if unique_count is None or not np.isscalar(unique_count):
                raise ValueError("groupbypack: unique_count must be a scalar value if ncountgroup is None")

            # get the unique_count ratio
            ratio =  len(ikey) / unique_count
            if len(ikey) > 1_000_000 and ratio < 40:
                nCountGroup = rc.BinCount(ikey, unique_count)
                iGroup, iFirstGroup  = rc.GroupFromBinCount(ikey, nCountGroup)
            else:
                # normal path (speed path from Ryan)
                nCountGroup, iGroup, iFirstGroup = rc.BinCount(ikey, unique_count, pack=True)

        else:
            # TJD Oct 2019, this routine is probably slower than BinCount with pack=True
            # high unique routine...
            iGroup, iFirstGroup  = rc.GroupFromBinCount(ikey, ncountgroup)
            nCountGroup = ncountgroup

    else:
        #
        # old routine which can take cutoffs
        # TODO: Delete this routine
        iGroup, iFirstGroup, nCountGroup = rc.GroupByPack32(ikey, None, unique_count, cutoffs=cutoffs)

    mkdict= {'iGroup':iGroup, 'iFirstGroup':iFirstGroup, 'nCountGroup':nCountGroup}
    return mkdict

#-------------------------------------------------------
def groupbyhash(list_arrays, hint_size:int=0, filter=None, hash_mode:int=2, cutoffs=None, pack:bool=False):
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

    >>> f = (c % 3).astype(np.bool)
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
        list_arrays=[list_arrays]

    if isinstance(list_arrays, list) and len(list_arrays)> 0:
        common_len = {len(arr) for arr in list_arrays}

        if len(common_len) == 1:
            common_len = common_len.pop()
            if common_len !=0:
                iKey, iFirstKey, unique_count = rc.MultiKeyGroupBy32(list_arrays, hint_size, filter, hash_mode, cutoffs=cutoffs)
            else:
                iKey = TypeRegister.FastArray([], dtype=np.int32)
                iFirstKey = iKey
                unique_count =0

            mkdict= {'iKey':iKey, 'iFirstKey':iFirstKey, 'unique_count': unique_count}

            if pack:
                packdict= groupbypack(iKey, None, unique_count + 1)
                for k,v in packdict.items():
                    mkdict[k]=v
            else:
                # leave empty
                for k in ['iGroup', 'iFirstGroup', 'nCountGroup']:
                    mkdict[k]=None

            return mkdict
        raise ValueError(f'groupbyhash all arrays must have same length not {common_len}')
    raise ValueError('groupbyhash first argument is not a list of numpy arrays')


#-------------------------------------------------------
def groupbylex(list_arrays, filter=None, cutoffs=None, base_index:int=1, rec:bool=False):
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
    if base_index != 1 and base_index !=0:
        raise ValueError(f"Invalid base_index {base_index!r}")

    if base_index == 0 and filter is not None:
        raise ValueError("Filter and base_index of 0 cannot be combined")

    if isinstance(list_arrays, np.ndarray):
        list_arrays=[list_arrays]

    if isinstance(list_arrays, list) and len(list_arrays)> 0:
        if not isinstance(list_arrays[0], np.ndarray):
            raise ValueError('groupbylex first argument is not a list of numpy arrays')

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

        index=None

        # check for filtering
        if filter is not None:
            filtertrue, truecount = bool_to_fancy(filter, both=True)
            totalcount = len(filter)

            # build a fancy index to pass to lexsort
            filterfalse = filtertrue[truecount:totalcount]
            filtertrue = filtertrue[0:truecount]
            index=filtertrue

        # lexsort
        if len(value_array) > 2e9:
            iGroup = rc.LexSort64(list_arrays, cutoffs=cutoffs, index=index)
        else:
            iGroup = rc.LexSort32(list_arrays, cutoffs=cutoffs, index=index)


        # make a record array if we did not already because GroupFromLexSort can only handle that
        if isinstance(value_array, list):
            value_array = np.core.records.fromarrays(list_arrays)
        retval = rc.GroupFromLexSort(iGroup, value_array, cutoffs=cutoffs, base_index=base_index)

        if len(retval)==3:
            iKey, iFirstKey, nCountGroup = retval
        else:
            iKey, iFirstKey, nCountGroup, nUniqueCutoffs = retval

        #print('igroup', len(iGroup), iGroup)
        #print("ikey", len(iKey), iKey)
        #print("ifirstkey", iFirstKey)
        #print("nCountGroup", len(nCountGroup), nCountGroup)

        if base_index == 0:
            iKey += 1
        else:
            # invalid bin count is 0 but we will fix up later if we have a filter
            nCountGroup[0]=0

        #derive iFirstGroup from nCountGroup
        iFirstGroup = nCountGroup.copy()
        iFirstGroup[1:] = nCountGroup.cumsum()[:-1]

        if filter is not None:
            # the number in the invalid bin is equal to the false filter
            nCountGroup[0] = len(filterfalse)

            # ikey has to get 0s where the filter is false
            iKey[filterfalse] = 0

            # the invalids are after all the valids in the iGroup
            iFirstGroup[0]=len(filtertrue)

        mkdict= {'iKey':iKey, 'iFirstKey':iFirstKey, 'unique_count': len(iFirstKey), 'iGroup':iGroup,  'iFirstGroup': iFirstGroup,  'nCountGroup': nCountGroup }
        if len(retval)==4:
            mkdict['nUniqueCutoffs']=nUniqueCutoffs
        return mkdict

    raise ValueError('groupbylex first argument is not a list of numpy arrays')

#-------------------------------------------------------
def groupby(list_arrays, filter=None, cutoffs=None, base_index:int=1, lex:bool=False, rec:bool=False, pack:bool=False, hint_size:int=0):
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

#-------------------------------------------------------
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
    mkdict= {'key':key, 'nth':nth, 'bktsize': bktsize, 'next':inext, 'prev':iprev, 'first':ifirst, 'last':ilast}
    return TypeRegister.Struct(mkdict)

#-----------------------------------------------------------------------------
# START OF NUMPY OVERLOADS ---------------------------------------------------
#-----------------------------------------------------------------------------
def all(*args,**kwargs):
    if isinstance(args[0], np.ndarray):
        return LedgerFunction(np.all,*args,**kwargs)
    # has python built-in
    return builtins.all(*args,**kwargs)

#-------------------------------------------------------
def any(*args,**kwargs):
    if isinstance(args[0], np.ndarray):
        return LedgerFunction(np.any,*args,**kwargs)
    # has python built-in
    return builtins.any(*args,**kwargs)

#-------------------------------------------------------
def arange(*args,**kwargs) -> 'FastArray':
    return LedgerFunction(np.arange,*args,**kwargs)

#-------------------------------------------------------
# If argsort implementation changes then add test cases to Python/core/riptable/tests/test_riptable_numpy_equivalency.py.
def argsort(*args,**kwargs): return LedgerFunction(np.argsort,*args,**kwargs)

#-------------------------------------------------------
# This is redefined down below...
# def ceil(*args,**kwargs): return LedgerFunction(np.ceil,*args,**kwargs)

#-------------------------------------------------------
def concatenate(*args,**kwargs):
    firstarg, *_ = args[0]
    if type(firstarg) not in (np.ndarray, TypeRegister.FastArray):
        if kwargs:
            raise ValueError(f'concatenate: keyword arguments not supported for arrays of type {type(firstarg)}\n\tGot keyword arguments {kwargs}')
        from .rt_hstack import stack_rows
        return stack_rows(args[0])
    result = LedgerFunction(np.concatenate,*args,**kwargs)
    return result

#-------------------------------------------------------
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

#-------------------------------------------------------
def crc64(arr: np.ndarray) -> int:
    # TODO: Enable this warning once the remaining code within riptable itself has been migrated to crc32c.
    #warnings.warn("This function is deprecated in favor of the crc32c function and will be removed in the next major version of riptable.", FutureWarning, stacklevel=2)
    return crc32c(arr)

#-------------------------------------------------------
def cumsum(*args,**kwargs): return LedgerFunction(np.cumsum,*args,**kwargs)

#-------------------------------------------------------
def cumprod(*args,**kwargs): return LedgerFunction(np.cumprod,*args,**kwargs)

#-------------------------------------------------------
def diff(*args,**kwargs): return LedgerFunction(np.diff,*args,**kwargs)

#-------------------------------------------------------
# this is a ufunc no need to take over def floor(*args,**kwargs): return LedgerFunction(np.floor,*args,**kwargs)

#-------------------------------------------------------
def full(shape, fill_value, dtype=None, order='C'):
    result= LedgerFunction(np.full, shape, fill_value, dtype=dtype, order=order)
    if hasattr(fill_value, 'newclassfrominstance'):
        result = fill_value.newclassfrominstance(result, fill_value)
    return result

#-------------------------------------------------------
def lexsort(*args,**kwargs):
    firstarg=args[0]
    if isinstance(firstarg,tuple):
        firstarg=list(firstarg)
        args=tuple(firstarg) + args[1:]
    if isinstance(firstarg,list):
        firstarg=firstarg[0]

    if isinstance(firstarg, np.ndarray):
        # make sure fastarray
        # also if arraysize > 2billiom call LexSort64 instead
        if firstarg.size > 2e9:
            return rc.LexSort64(*args,**kwargs)
        return rc.LexSort32(*args,**kwargs)
    else:
        return LedgerFunction(np.lexsort,*args,**kwargs)


#-------------------------------------------------------
def ones(*args,**kwargs) -> 'FastArray': return LedgerFunction(np.ones,*args,**kwargs)
def ones_like(*args,**kwargs) -> 'FastArray': return LedgerFunction(np.ones_like,*args,**kwargs)

#-------------------------------------------------------
def zeros(*args,**kwargs) -> 'FastArray': return LedgerFunction(np.zeros,*args,**kwargs)
def zeros_like(*args,**kwargs) -> 'FastArray': return LedgerFunction(np.zeros_like,*args,**kwargs)

#-------------------------------------------------------
def reshape(*args,**kwargs): return LedgerFunction(np.reshape,*args,**kwargs)

#-------------------------------------------------------
# a faster way to do array index masks
def reindex_fast(index, array):
    if isinstance(index, np.ndarray) and isinstance(array, np.ndarray):
        return rc.ReIndex(index, array)
    raise TypeError("two arguments, both args must be numpy arrays.  the first argument indexes into the second argument.")

#-------------------------------------------------------
# If sort implementation changes then add test cases to Python/core/riptable/tests/test_riptable_numpy_equivalency.py.
def sort(*args,**kwargs): return LedgerFunction(np.sort,*args,**kwargs)

#-------------------------------------------------------
# If transpose implementation changes then add test cases to Python/core/riptable/tests/test_riptable_numpy_equivalency.py.
def transpose(*args,**kwargs): return LedgerFunction(np.transpose,*args,**kwargs)

#-------------------------------------------------------
def where(condition, x=None, y=None):
    # handle the single-argument case
    missing = (x is None, y is None).count(True)
    if missing == 1:
        raise ValueError(f"where: must provide both 'x' and 'y' or neither. x={x}  y={y}")

    if not isinstance(condition, np.ndarray):
        if condition is False or condition is True:
            # punt to normal numpy instead of error which may process None differently
            return LedgerFunction(np.where,condition,x,y)

        condition = TypeRegister.FastArray(condition)
    elif len(condition) == 1:
        # punt to normal numpy since an array of 1
        return LedgerFunction(np.where,condition,x,y)

    if condition.ndim > 1:
        # punt to normal numpy since more than one dimension
        return LedgerFunction(np.where,condition,x,y)

    if condition.dtype != np.bool:
        #NOTE: believe numpy just flips it to boolean using astype, where object arrays handled differently with None and 0
        condition = condition != 0

    if missing == 2:
        return (bool_to_fancy(condition),)

    # this is the normal 3 argument where
    common_dtype = get_common_dtype(x,y)

    # see if we can accelerate where
    if common_dtype.char in NumpyCharTypes.SupportedAlternate:

        def _possibly_convert(arr):
            # NOTE detect scalars first?
            try:
                if arr.dtype.num != common_dtype.num:
                    # perform a safe conversion understanding sentinels
                    #print("Converting1 to", common_dtype, arr.dtype.num, common_dtype.num)
                    arr = TypeRegister.MathLedger._AS_FA_TYPE(arr, common_dtype.num)
                elif arr.itemsize != common_dtype.itemsize:
                    # make strings sizes the same
                    arr = arr.astype(common_dtype)

            except:
                arr = TypeRegister.FastArray(arr)
                if arr.dtype.num != common_dtype.num:
                    #print("Converting2 to", common_dtype, arr.dtype.num, common_dtype.num)
                    arr = arr.astype(common_dtype)

            # strided check
            #if arr.ndim ==1 and arr.itemsize != arr.strides[0]:
            #    arr = arr.copy()

            #check if can make like a scalar
            try:
                if len(arr) == 1:
                    #print("array len1 detected")
                    if common_dtype.char in NumpyCharTypes.AllFloat:
                        arr=float(arr[0])
                    elif common_dtype.char in NumpyCharTypes.AllInteger:
                        arr=int(arr[0])
                    else:
                        arr=arr[0]
            except:
                # probably cannot take len, might be numpy scalar
                num = arr.dtype.num
                if num ==0:
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

        # call down into C++ version of where
        return rc.Where(condition, (x, y), common_dtype.num, common_dtype.itemsize)

    # punt to normal numpy
    return LedgerFunction(np.where,condition,x,y)

#-------------------------------------------------------
def sortinplaceindirect(*args,**kwargs): return LedgerFunction(rc.SortInPlaceIndirect,*args,**kwargs)

#-------------------------------------------------------
# is a ufunc def trunc(*args,**kwargs): return LedgerFunction(rc.IsSorted,*args,**kwargs)

#-------------------------------------------------------
def _unary_func(func, *args, **kwargs):
    """
    pooling of unary functions
    if not a fastarray, it will try to convert it
    then it will call the normal numpy routine, which will call FastArray unary func (which is parallelized)
    if a user calls rt.log(pandasarray) with a pandas array, it will get parallelized now
    """
    if len(args) ==1:
        a = args[0]
        # if they pass a list, we do not bother to convert it (possible future improvement)
        if isinstance(a, np.ndarray):
            try:
                # try to convert to FastArray so that it will route to TypeRegister.FastArray's array_ufunc
                if not isinstance(a, TypeRegister.FastArray):
                    a=a.view(TypeRegister.FastArray)
                return func(a, **kwargs)
            except:
                # fall through and call normal numpy
                pass
    return func(*args, **kwargs)

#-------------------------------------------------------
def _convert_cat_args(args):
    if len(args) == 1:
        if isinstance(args[0], TypeRegister.Categorical):
            args = (args[0]._fa,)
        return args
    return args

#-------------------------------------------------------
def nan_to_num(*args,**kwargs):
    """
    arg1: ndarray
    returns: ndarray with nan_to_num
    notes: if you want to do this inplace contact TJD
    """
    return np.nan_to_num(*args,**kwargs)

#-------------------------------------------------------
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

#-------------------------------------------------------
def ceil(*args,**kwargs):
    return _unary_func(np.ceil,*args,**kwargs)

#-------------------------------------------------------
def floor(*args,**kwargs):
    return _unary_func(np.floor,*args,**kwargs)

#-------------------------------------------------------
def trunc(*args,**kwargs):
    return _unary_func(np.trunc,*args,**kwargs)

#-------------------------------------------------------
def log(*args,**kwargs):
    return _unary_func(np.log,*args,**kwargs)

#-------------------------------------------------------
def log10(*args,**kwargs):
    return _unary_func(np.log10,*args,**kwargs)

#-------------------------------------------------------
def absolute(*args,**kwargs):
    return _unary_func(np.absolute,*args,**kwargs)

#-------------------------------------------------------
def abs(*args,**kwargs):
    """
    This will check for numpy array first and call np.abs
    """
    a = args[0]
    if isinstance(a, np.ndarray):
        if not isinstance(a, TypeRegister.FastArray):
            a=TypeRegister.FastArray(a)
        return np.abs(*args, **kwargs)
    return builtins.abs(a)

#-------------------------------------------------------
def round(*args,**kwargs):
    """
    This will check for numpy array first and call np.round
    """
    a = args[0]
    if isinstance(a, np.ndarray):
        return np.round(*args, **kwargs)
    return builtins.round(a)

#-------------------------------------------------------
def sum(*args,**kwargs):
    """
    This will check for numpy array first and call np.sum
    otherwise use builtin
    for instance c={1,2,3} and sum(c) should work also
    """
    args = _convert_cat_args(args)
    if isinstance(args[0], np.ndarray):
        return args[0].sum(*args[1:], **kwargs)
    return builtins.sum(*args,**kwargs)

#-------------------------------------------------------
def nansum(*args, **kwargs):
    """
    This will check for numpy array first and call np.sum
    otherwise use builtin
    for isntance c={1,2,3} and sum(c) should work also
    """
    args = _convert_cat_args(args)
    if isinstance(args[0], np.ndarray):
        return args[0].nansum()
    return np.nansum(*args, **kwargs)

#-------------------------------------------------------
def argmax(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray): return args[0].argmax(**kwargs)
    return np.argmax(*args, **kwargs)

#-------------------------------------------------------
def argmin(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray): return args[0].argmin(**kwargs)
    return np.argmin(*args, **kwargs)

#-------------------------------------------------------
def nanargmax(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray): return args[0].nanargmax(**kwargs)
    return np.nanargmax(*args, **kwargs)

#-------------------------------------------------------
def nanargmin(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray): return args[0].nanargmin(**kwargs)
    return np.nanargmin(*args, **kwargs)

#-------------------------------------------------------
def _reclaim_type(arr, x1, x2):
    if isinstance(arr, np.ndarray) and isinstance(x1, (TypeRegister.DateTimeBase, TypeRegister.DateBase)):
        # handle case when DateTime used (only checks first array not second)
        arrtype = type(x1)
        if not isinstance(arr, arrtype):
            arr= arrtype(arr)
    return arr

#-------------------------------------------------------
def maximum(x1, x2, *args,**kwargs):
    # two arrays are passed to maximum, minimum
    return _reclaim_type(np.maximum(x1, x2, *args, **kwargs), x1, x2)

#-------------------------------------------------------
def minimum(x1, x2, *args, **kwargs):
    return _reclaim_type(np.minimum(x1, x2, *args, **kwargs), x1, x2)

#-------------------------------------------------------
def max(*args,**kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray):
        badlist=['S','O','U']
        if not args[0].dtype.char in badlist:
            if len(args) > 1:
                return maximum(*args,**kwargs)
            return args[0].max(**kwargs)
        else:
            # Object, String, Unicode
            if len(args) == 1:
                # assuming they want length of string
                return builtins.max([item for item in args[0]], **kwargs)
            else:
                warnings.warn("Getting the max of two objects or string arrays is not currently allowed")
                return None
    return builtins.max(*args, **kwargs)

#-------------------------------------------------------
def min(*args,**kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray):
        badlist=['S','O','U']
        if not args[0].dtype.char in badlist:
            if len(args) > 1:
                return minimum(*args,**kwargs)
            return args[0].min(**kwargs)
        else:
            # Object, String, Unicode
            if len(args) == 1:
                # assuming they want length of string
                return builtins.min([item for item in args[0]], **kwargs)
            else:
                warnings.warn("Getting the max of two objects or string arrays is not currently allowed")
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


#-------------------------------------------------------
def mean(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray): return args[0].mean(**kwargs)
    return np.mean(*args, **kwargs)

#-------------------------------------------------------
def nanmean(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray): return args[0].nanmean(**kwargs)
    return np.nanmean(*args, **kwargs)

#-------------------------------------------------------
def median(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray): return np.median(*args,**kwargs)
    return builtins.median(*args, **kwargs)

#-------------------------------------------------------
def nanmedian(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray): return np.nanmedian(*args, **kwargs)
    return np.nanmedian(*args, **kwargs)

#-------------------------------------------------------
def var(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray): return args[0].var(**kwargs)
    return builtins.var(*args, **kwargs)

#-------------------------------------------------------
def nanvar(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray): return args[0].nanvar(**kwargs)
    return np.nanvar(*args, **kwargs)

#-------------------------------------------------------
def std(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray): return args[0].std(**kwargs)
    return builtins.var(*args, **kwargs)

#-------------------------------------------------------
def nanstd(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray): return args[0].nanstd(**kwargs)
    return np.nanstd(*args, **kwargs)

#-------------------------------------------------------
def percentile(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray): return np.percentile(*args,**kwargs)
    return np.percentile(*args, **kwargs)

#-------------------------------------------------------
def nanpercentile(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray): return np.nanpercentile(*args,**kwargs)
    return np.nanpercentile(*args, **kwargs)

#-------------------------------------------------------
def bincount(*args, **kwargs):
    args = _convert_cat_args(args)
    if isinstance(args[0],np.ndarray): return np.bincount(*args,**kwargs)
    return np.bincount(*args, **kwargs)

#-------------------------------------------------------
def isnan(*args, **kwargs):
    try:
        return args[0].isnan(**kwargs)
    except:
        return _unary_func(np.isnan,*args,**kwargs)

#-------------------------------------------------------
def isnotnan(*args, **kwargs):
    """ opposite of isnan """
    try:
        return args[0].isnotnan(**kwargs)
    except:
        return ~np.isnan(*args, **kwargs)

#-------------------------------------------------------
def isnanorzero(*args, **kwargs):
    try:
        return args[0].isnanorzero(**kwargs)
    except:
        # slow way
        result= np.isnan(*args, **kwargs)
        result += (args[0] ==0)
        return result

#-------------------------------------------------------
def isfinite(*args, **kwargs):
    try:
        return args[0].isfinite(**kwargs)
    except:
        return _unary_func(np.isfinite,*args,**kwargs)

#-------------------------------------------------------
def isnotfinite(*args, **kwargs):
    try:
        return args[0].isnotfinite(**kwargs)
    except:
        return ~np.isfinite(*args, **kwargs)

#-------------------------------------------------------
def isinf(*args, **kwargs):
    try:
        return args[0].isinf(**kwargs)
    except:
        return _unary_func(np.isinf,*args,**kwargs)

#-------------------------------------------------------
def isnotinf(*args, **kwargs):
    try:
        return args[0].isnotinf(**kwargs)
    except:
        return ~np.isinf(*args, **kwargs)

# ------------------------------------------------------------
def putmask(a, mask, values):
    """
    This is roughly the equivalent of arr[mask] = arr2[mask].

    Examples:
    ---------
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
            #final attempt use numpy
            return np.putmask(a, mask, values)

# ------------------------------------------------------------
def vstack(arrlist, dtype=None, order='C'):
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
            h=hstack(arrlist, dtype=dtype)
            if (order == 'F' or order =='f'):
                return h.reshape((numrows, numcols), order='F')
            else:
                return h.reshape((numcols, numrows), order='C')
    except Exception:
        warnings.warn(f"vstack with order={order!r} failed, calling np.vstack")

    return np.vstack(arrlist)

# ------------------------------------------------------------
def repeat(a, repeats, axis=None):
    """ see np.repeat """
    # similar bug as tile, calls reshape which maintains class, but kills attributes
    if isinstance(a, TypeRegister.FastArray):
        result = np.repeat(a._np, repeats, axis=axis).view(TypeRegister.FastArray)
        return TypeRegister.newclassfrominstance(result, a)
    return np.repeat(a, repeats, axis=axis).view(TypeRegister.FastArray)

# ------------------------------------------------------------
def tile(arr, reps):
    """ see np.tile """
    if isinstance(arr,TypeRegister.FastArray):
        # bug in tile, have to flip to normal numpy array first)
        result = np.tile(arr._np, reps).view(TypeRegister.FastArray)
        return TypeRegister.newclassfrominstance(result, arr)

    return np.tile(arr, reps).view(TypeRegister.FastArray)

#-------------------------------------------------------
# like in matlab, convert to int8
def logical(a):
    if isinstance(a, np.ndarray):
        if a.dtype==np.bool: return a
        return a.astype(np.bool)
    return np.bool(a).view(TypeRegister.FastArray)

##-------------------------------------------------------
# not allowed
#class bool(np.bool):
#    pass

#-------------------------------------------------------
class bool_(np.bool_):
    """
    riptable equivalent of np.bool_
    has invalid concept
    rt.interp?
    See Also
    --------
    np.bool_
    np.bool
    """

    # Allow np.bool.inv  to work
    inv = INVALID_DICT[0]

    def __new__(cls, *args, **kwargs):
        if len(args) ==1:
            # check if converting an existing array
            if isinstance(args[0], np.ndarray):
                return TypeRegister.FastArray.astype(args[0], np.bool_, **kwargs)
        instance = np.bool_(*args, **kwargs)
        if np.isscalar(instance): return instance
        return instance.view(TypeRegister.FastArray)

#-------------------------------------------------------
class int8(np.int8):
    """
    riptable equivalent of np.int8
    has invalid concept

    See Also
    --------
    np.int8
    """

    # Allow np.int8.inv  to work
    inv = INVALID_DICT[1]

    def __new__(cls, *args, **kwargs):
        if len(args) ==1:
            # check if converting an existing array
            if isinstance(args[0], np.ndarray):
                return TypeRegister.FastArray.astype(args[0], np.int8, **kwargs)
        instance = np.int8(*args, **kwargs)
        if np.isscalar(instance): return instance
        return instance.view(TypeRegister.FastArray)

#-------------------------------------------------------
class uint8(np.uint8):
    """
    riptable equivalent of np.uint8
    has invalid concept

    See Also
    --------
    np.uint8
    """

    # Allow np.uint8.inv  to work
    inv = INVALID_DICT[2]

    def __new__(cls, *args, **kwargs):
        if len(args) ==1:
            # check if converting an existing array
            if isinstance(args[0], np.ndarray):
                return TypeRegister.FastArray.astype(args[0], np.uint8, **kwargs)
        instance = np.uint8(*args, **kwargs)
        if np.isscalar(instance): return instance
        return instance.view(TypeRegister.FastArray)

#-------------------------------------------------------
class int16(np.int16):
    """
    riptable equivalent of np.int16
    has invalid concept

    See Also
    --------
    np.int16
    """

    # Allow np.int16.inv  to work
    inv = INVALID_DICT[3]

    def __new__(cls, *args, **kwargs):
        if len(args) ==1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.int16, **kwargs)
        instance = np.int16(*args, **kwargs)
        if np.isscalar(instance): return instance
        return instance.view(TypeRegister.FastArray)

#-------------------------------------------------------
class uint16(np.uint16):
    """
    riptable equivalent of np.uint16
    has invalid concept

    See Also
    --------
    np.uint16
    """

    # Allow np.uint16.inv  to work
    inv = INVALID_DICT[4]

    def __new__(cls, *args, **kwargs):
        if len(args) ==1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.uint16, **kwargs)
        instance = np.uint16(*args, **kwargs)
        if np.isscalar(instance): return instance
        return instance.view(TypeRegister.FastArray)

#-------------------------------------------------------
class int32(np.int32):
    """
    riptable equivalent of np.int32
    has invalid concept

    See Also
    --------
    np.int32
    """

    # Allow np.int32.inv  to work
    inv = INVALID_DICT[5]

    def __new__(cls, *args, **kwargs):
        if len(args) ==1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.int32, **kwargs)
        instance = np.int32(*args, **kwargs)
        if np.isscalar(instance): return instance
        return instance.view(TypeRegister.FastArray)

#-------------------------------------------------------
class uint32(np.uint32):
    """
    riptable equivalent of np.uint32
    has invalid concept

    See Also
    --------
    np.uint32
    """

    # Allow np.uint32.inv  to work
    inv = INVALID_DICT[6]

    def __new__(cls, *args, **kwargs):
        if len(args) ==1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.uint32, **kwargs)
        instance = np.uint32(*args, **kwargs)
        if np.isscalar(instance): return instance
        return instance.view(TypeRegister.FastArray)

#-------------------------------------------------------
class int64(np.int64):
    """
    riptable equivalent of np.int64
    has invalid concept

    See Also
    --------
    np.int64
    """

    # Allow np.int64.inv  to work
    inv = INVALID_DICT[9]

    def __new__(cls, *args, **kwargs):
        if len(args) ==1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.int64, **kwargs)
        instance = np.int64(*args, **kwargs)
        if np.isscalar(instance): return instance
        return instance.view(TypeRegister.FastArray)

#-------------------------------------------------------
class uint64(np.uint64):
    """
    riptable equivalent of np.uint64
    has invalid concept

    See Also
    --------
    np.uint64
    """

    # Allow np.uint64.inv  to work
    inv = INVALID_DICT[10]

    def __new__(cls, *args, **kwargs):
        if len(args) ==1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.uint64, **kwargs)
        instance = np.uint64(*args, **kwargs)
        if np.isscalar(instance): return instance
        return instance.view(TypeRegister.FastArray)

#-------------------------------------------------------
class int0(int64):
    pass

#-------------------------------------------------------
class uint0(uint64):
    pass

#-------------------------------------------------------
class bytes_(np.bytes_):
    """
    riptable equivalent of np.bytes_
    has invalid concept

    See Also
    --------
    np.bytes_
    """

    # Allow np.bytes_.inv  to work
    inv = INVALID_DICT[18]

    def __new__(cls, arg0, *args, **kwargs):
        if np.isscalar(arg0):
            return np.bytes_(arg0, *args, **kwargs)
        return TypeRegister.FastArray(arg0, *args, dtype = 'S', **kwargs)

#-------------------------------------------------------
class str_(np.str_):
    """
    riptable equivalent of np.str_
    has invalid concept

    See Also
    --------
    np.str_
    """

    # Allow np.str_.inv  to work
    inv = INVALID_DICT[19]

    def __new__(cls, arg0, *args, **kwargs):
        if np.isscalar(arg0):
            return np.str_(arg0, *args, **kwargs)
        return TypeRegister.FastArray(arg0, *args, unicode=True, dtype = 'U', **kwargs)

#-------------------------------------------------------
# like in numpy, convert to a half
def half(a):
    if isinstance(a, np.ndarray):
        if a.dtype==np.float16: return a
        return a.astype(np.float16)
    return np.float16(a).view(TypeRegister.FastArray)

#-------------------------------------------------------
# like in matlab, convert to a single
def single(a):
    if isinstance(a, np.ndarray):
        if a.dtype==np.float32: return a
        return a.astype(np.float32)
    return np.float32(a).view(TypeRegister.FastArray)

#-------------------------------------------------------
# like in matlab, convert to a double
def double(a):
    if isinstance(a, np.ndarray):
        if a.dtype==np.float64: return a
        return a.astype(np.float64)
    return np.float64(a).view(TypeRegister.FastArray)

#-------------------------------------------------------
# like in numpy, convert to a longdouble
def longdouble(a):
    if isinstance(a, np.ndarray):
        if a.dtype==np.longdouble: return a
        return a.astype(np.longdouble)
    return np.longdouble(a).view(TypeRegister.FastArray)

#-------------------------------------------------------
class float32(np.float32):
    """
    riptable equivalent of np.float32
    has invalid concept

    See Also
    --------
    np.float32
    """

    # Allow np.float32.inv  to work
    inv = INVALID_DICT[11]

    def __new__(cls, *args, **kwargs):
        if len(args) ==1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.float32, **kwargs)
        instance = np.float32(*args, **kwargs)
        if np.isscalar(instance): return instance
        return instance.view(TypeRegister.FastArray)

#-------------------------------------------------------
class float64(np.float64):
    """
    riptable equivalent of np.float64
    has invalid concept

    See Also
    --------
    np.float64
    """

    # Allow np.float64.inv  to work
    inv = INVALID_DICT[12]

    def __new__(cls, *args, **kwargs):
        if len(args) ==1 and isinstance(args[0], np.ndarray):
            return TypeRegister.FastArray.astype(args[0], np.float64, **kwargs)
        instance = np.float64(*args, **kwargs)
        if np.isscalar(instance): return instance
        return instance.view(TypeRegister.FastArray)

#-------------------------------------------------------
# linux only
#class float128(np.float128):

#-------------------------------------------------------
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
        x=TypeRegister.FastArray(x)

    if not isinstance(xp, np.ndarray):
        xp=TypeRegister.FastArray(xp)

    if not isinstance(fp, np.ndarray):
        fp=TypeRegister.FastArray(fp)

    # check for float32
    if x.dtype.num == 11 and (xp.dtype.num != 11 or fp.dtype.num != 11):
        warnings.warn('rt.interp is downcasting to a float32 to match first array')
        xp = xp.astype(np.float32)
        fp = fp.astype(np.float32)
    return rc.InterpExtrap2d(x, xp, fp, 1)

#-------------------------------------------------------
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
    if not isinstance(xp, np.ndarray):
        xp=TypeRegister.FastArray(xp)

    if not isinstance(fp, np.ndarray):
        fp=TypeRegister.FastArray(fp)

    return rc.InterpExtrap2d(x, xp, fp, 0)

#-------------------------------------------------------
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
        if np.issubdtype(a.dtype, np.integer) or a.dtype == np.bool:
            if not a.flags.c_contiguous:
                a = a.copy()
            return rc.BitCount(a)
        else:
            raise ValueError(f'Unsupported array dtype {a.dtype}')
    else:
        if isinstance(a, (int, np.integer)):
            return bin(a).count('1')
        else:
            raise ValueError(f'Unsupported input type {type(a)}')

#-------------------------------------------------------
def bool_to_fancy(arr: np.ndarray, both:bool=False):
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
    >>> bools = np.random.randint(2, size=20, dtype=np.int8).astype(np.bool_)
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
        if arr.dtype.char == '?':
            return rc.BooleanToFancy(arr, both=both)
        else:
            raise TypeError(f"Input array must be boolean. Got {arr.dtype}")
    else:
        raise TypeError(f"Input must be ndarray. Got {type(arr)}")

#-------------------------------------------------------
def mask_or(*args, **kwargs):
    """pass in a tuple or list of boolean arrays to OR together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_OR, False)

#-------------------------------------------------------
def mask_and(*args, **kwargs):
    """pass in a tuple or list of boolean arrays to AND together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_AND, False)

#-------------------------------------------------------
def mask_xor(*args, **kwargs):
    """pass in a tuple or list of boolean arrays to XOR together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_XOR, False)

#-------------------------------------------------------
def mask_andnot(*args, **kwargs):
    """pass in a tuple or list of boolean arrays to ANDNOT together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_ANDNOT, False)


#-------------------------------------------------------
def mask_ori(*args, **kwargs):
    """inplace version: pass in a tuple or list of boolean arrays to OR together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_OR, True)

#-------------------------------------------------------
def mask_andi(*args, **kwargs):
    """inplace version: pass in a tuple or list of boolean arrays to AND together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_AND, True)

#-------------------------------------------------------
def mask_xori(*args, **kwargs):
    """inplace version: pass in a tuple or list of boolean arrays to XOR together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_XOR, True)

#-------------------------------------------------------
def mask_andnoti(*args, **kwargs):
    """inplace version: pass in a tuple or list of boolean arrays to ANDNOT together"""
    return _mask_op((args), MATH_OPERATION.BITWISE_ANDNOT, True)

#-------------------------------------------------------
def _mask_op( bool_list, funcNum, inplace = False):
    # size check done by TypeRegister.FastArray cpp code
    # we do an all boolen check here for CPP code
    lenbool = len(bool_list)
    if lenbool == 1 and isinstance(bool_list[0], (list,tuple)):
        bool_list = bool_list[0]
        lenbool = len(bool_list)

    if lenbool == 0:
        raise ValueError(f"Nothing passed")

    # check if nothing to do because just one boolean array in list
    if lenbool == 1:
        return bool_list[0]

    # we could support all int types here as well
    dtype = 0
    for v in bool_list: dtype += v.dtype.num
    if dtype != 0:
        raise TypeError(f"Must all be boolean types")

    # we have at least two items
    # grabbing the func pointer speeds things up in testing
    func=TypeRegister.MathLedger._BASICMATH_TWO_INPUTS
    if inplace:
        #assume first value can be reused
        result = bool_list[0]
        func((result, bool_list[1], result), funcNum, 0)
        i=2

        while i < lenbool:
            # this will do inplace
            func((result, bool_list[i], result), funcNum, 0)
            i+=1
    else:
        result= func((bool_list[0], bool_list[1]), funcNum, 0)
        i=2

        while i < lenbool:
            func((result, bool_list[i], result), funcNum, 0)
            i+=1

    return result

# ------------------------------------------------------------
def hstack(tup, dtype=None, **kwargs):
    """
    see numpy hstack
    riptable can also take a dtype (it will convert all arrays to that dtype while stacking)
    riptable version will preserve sentinels
    riptable version is multithreaded
    for special classes like categorical and dataset, it will check to see if the
    class has it's own hstack and it will call that
    """

    # Check to see if we have one homogenized type
    set_of_types = {type(i) for i in tup}

    if len(set_of_types)==1:
        # we know the data is all the same type
        # check if this is a special type that we know how to hstack
        try:
            # pass kwargs in case special type has unique keywords
            return [*set_of_types][0].hstack(tup, **kwargs)
        except:
            pass

    dtypenum=-1

    if dtype is not None:
        try:
            dtypenum = dtype.num
        except:
            dtypenum = np.dtype(dtype).num

    try:
        return rc.HStack(tup, dtypenum)
    except:
        return np.hstack(tup).view(TypeRegister.FastArray)

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
    mymodule=sys.modules[__name__]
    all_myfunctions = inspect.getmembers(mymodule, inspect.isfunction)
    all_npfunctions = inspect.getmembers(np)
    #all_npfunctions += inspect.getmembers(np, inspect.isfunction)
    #all_npfunctions += inspect.getmembers(np, inspect.isbuiltin)

    # build dictionary
    npdict={}
    for funcs in all_npfunctions:
        npdict[funcs[0]]=funcs[1]

    # now for each function that has an np flavor, copy over the doc strings
    for funcs in all_myfunctions:
        if funcs[0] in npdict:
            #print("doc fix", funcs[0])

            # combine riptable docstring with numpy docstring
            npdoc = npdict[funcs[0]].__doc__
            if npdoc is not None:
                if funcs[1].__doc__ is None:
                    funcs[1].__doc__ = ''
                funcs[1].__doc__ += npdoc

            # old, only uses numpy docstring
            #funcs[1].__doc__ = npdict[funcs[0]].__doc__
        else:
            pass
            #print("reject", funcs[0])


# keep this function last
# -- fixup the doc strings for numpy functions we take over
_FixupDocStrings()

# wire asarray to the C function directly
asanyarray = rc.AsAnyArray
asarray = rc.AsFastArray
