# -*- coding: utf-8 -*-
__all__ = ['Dataset', ]

from collections import abc, Counter, namedtuple
import operator
import os
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import warnings

import numpy as np

from .rt_struct import Struct
from .rt_fastarray import FastArray
from .rt_enum import TypeId, DS_DISPLAY_TYPES, NumpyCharTypes, DisplayDetectModes, INVALID_DICT, MATH_OPERATION, \
    ApplyType, SDSFileType, ColHeader, TOTAL_LONG_NAME, CategoryMode
from .Utils.rt_display_properties import format_scalar
from .rt_hstack import hstack_any
from .rt_groupby import GroupBy
from .rt_timers import GetTSC
from .rt_numpy import full, lexsort, reindex_fast, unique, ismember, empty, arange, tile, ones, zeros
from .rt_numpy import sum, mean, var, std, argmax, argmin, min, max, median, cumsum, putmask, mask_ori, mask_andi
from .rt_numpy import nanargmin, nanargmax, nansum, nanmean, nanvar, nanstd, nanmedian, nanmin, nanmax
from .rt_numpy import isnan, isnanorzero, isnotfinite, bool_to_fancy, hstack, vstack, combine2keys, cat2keys
from .rt_grouping import combine2groups
from .rt_display import DisplayTable, DisplayDetect, DisplayString
from .rt_misc import build_header_tuples
from . import rt_merge
from .rt_sort_cache import SortCache
from .rt_utils import describe, quantile, is_list_like, get_default_value, _possibly_convert_rec_array, sample
from .rt_sds import save_sds, compress_dataset_internal, COMPRESSION_TYPE_NONE, COMPRESSION_TYPE_ZSTD,  _sds_path_single, _write_to_sds, load_sds
from .Utils.rt_metadata import MetaData
from .rt_mlutils import normalize_minmax, normalize_zscore
from .rt_itemcontainer import ItemContainer
from .rt_imatrix import IMatrix

if TYPE_CHECKING:
    from datetime import timedelta
    from .rt_accum2 import Accum2
    from .rt_categorical import Categorical
    from .rt_multiset import Multiset

    # pandas is an optional dependency.
    try:
        import pandas as pd
    except ImportError:
        pass

    # pyarrow is an optional dependency.
    try:
        import pyarrow as pa
    except ImportError:
        pass


ArrayCompatible = Union[list, abc.Iterable, np.ndarray]

class Dataset(Struct):
    """
    The Dataset class is the workhorse of riptable; it may be considered as an NxK array of values (of mixed type,
    constant by column) where the rows are integer indexed and the columns are indexed by name (as well as
    integer index).  Alternatively it may be regarded as a dictionary of arrays, all of the same length.

    The Dataset constructor takes dictionaries (dict, OrderedDict, etc...), as well as single instances of
    Dataset or Struct (if all entries are of the same length).
    Dataset() := Dataset({}).

    The constructor dictionary keys (or element/column names added later) must be legal Python
    variable names, not starting with '_' and not conflicting with any Dataset member names.

    **Column indexing behavior**::

        >>> st['b'] # get a column (equiv. st.b)
        >>> st[['a', 'e']] # get some columns
        >>> st[[0, 4]] # get some columns (order is that of iterating st (== list(st))
        >>> st[1:5:2] # standard slice notation, indexing corresponding to previous
        >>> st[bool_vector_len5] # get 'True' columns

    In all of the above: ``st[col_spec] := st[:, colspec]``

    **Row indexing behavior**::

        >>> st[2, :] # get a row (all columns)
        >>> st[[3, 7], :] # get some rows (all columns)
        >>> st[1:5:2, :] # standard slice notation (all columns)
        >>> st[bool_vector_len5, :] # get 'True' rows (all columns)
        >>> st[row_spec, col_spec] # get specified rows for specified columns

    Note that because ``st[spec] := st[:, spec]``, to specify rows one *must* specify columns
    as well, at least as 'the all-slice': e.g., ``st[row_spec, :]``.

    Wherever possible, views into the original data are returned.  Use
    :meth:`~rt.rt_dataset.Dataset.copy` where necessary.

    Examples
    --------
    A Dataset with six integral columns of length 10::

    >>> import string
    >>> ds = rt.Dataset({_k: list(range(_i * 10, (_i + 1) * 10)) for _i, _k in enumerate(string.ascii_lowercase[:6])})

    Add a column of strings (stored internally as ascii bytes)::

    >>> ds.S = list('ABCDEFGHIJ')

    Add a column of non-ascii strings (stored internally as a Categorical column)::

    >>> ds.U = list('ℙƴ☂ℌøἤ-613')
    >>> print(ds)
    #   a    b    c    d    e    f   S   U
    -   -   --   --   --   --   --   -   -
    0   0   10   20   30   40   50   A   ℙ
    1   1   11   21   31   41   51   B   ƴ
    2   2   12   22   32   42   52   C   ☂
    3   3   13   23   33   43   53   D   ℌ
    4   4   14   24   34   44   54   E   ø
    5   5   15   25   35   45   55   F   ἤ
    6   6   16   26   36   46   56   G   -
    7   7   17   27   37   47   57   H   6
    8   8   18   28   38   48   58   I   1
    9   9   19   29   39   49   59   J   3

    >>> ds.get_ncols()
    8
    >>> ds.get_nrows()
    10

    ``len`` applied to a Dataset returns the number of rows in the Dataset.

    >>> len(ds)
    10
    >>> # Not too dissimilar from numpy/pandas in many ways.
    >>> ds.shape
    (10, 8)
    >>> ds.size
    80
    >>> ds.head()
    >>> ds.tail(n=3)

    >>> assert (ds.c == ds['c']).all() and (ds.c == ds[2]).all()

    >>> print(ds[1:8:3, :3])
    #   a    b    c
    -   -   --   --
    0   1   11   21
    1   4   14   24
    2   7   17   27

    >>> ds.newcol = np.arange(100, 110) # okay, a new entry
    >>> ds.newcol = np.arange(200, 210) # okay, replace the entry
    >>> ds['another'] = 6 # okay (scalar is promoted to correct length vector)
    >>> ds['another'] = ds.another.astype(np.float32) # redefines type of column

    >>> ds.col_remove(['newcol', 'another'])

    Fancy indexing for get/set::

    >>> ds[1:8:3, :3] = ds[2:9:3, ['d', 'e', 'f']]

    Equivalents::

    >>> for colname in ds: print(colname, ds[colname])
    >>> for colname, array in ds.items(): print(colname, array)
    >>> for colname, array in zip(ds.keys(), ds.values()): print(colname, array)
    >>> for colname, array in zip(ds, ds.values()): print(colname, array)

    >>> if key in ds:
    ...    assert getattr(ds, key) is ds[key]

    Context manager::

    >>> with Dataset({'a': 1, 'b': 'fish'}) as ds0:
    ...     print(ds0.a)
    [1]

    >>> assert not hasattr(ds0, 'a')

    Dataset cannot be used in a boolean context ``(if ds: ...)``,
    use ``ds.any(axis='all')`` or ``ds.all(axis='all')`` instead::

    >>> ds1 = ds[:-2] # Drop the string columns, Categoricals are 'funny' here.
    >>> ds1.any(axis='all')
    True

    >>> ds1.all(axis='all')
    False

    >>> ds1.a[0] = -99
    >>> ds1.all(axis='all')
    True

    >>> if (ds2 <= ds3).all(axis='all'): ...

    Do math::

    >>> ds1 += 5
    >>> ds1 + 3 * ds2 - np.ones(10)
    >>> ds1 ** 5
    >>> ds.abs()

    >>> ds.sum(axis=0, as_dataset=True)
        #    a     b     c     d     e     f
        -   --   ---   ---   ---   ---   ---
        0   39   238   338   345   445   545

    >>> ds.sum(axis=1)
    array([ 51, 249, 162, 168, 267, 180, 186, 285, 198, 204])

    >>> ds.sum(axis=None)
    1950
    """
    def __init__(
        self,
        inputval: Optional[Union[ArrayCompatible, dict, Iterable[ArrayCompatible], Iterable[Tuple[str, ArrayCompatible]], 'ItemContainer']] = None,
        base_index: int = 0,
        sort: bool = False,
        unicode: bool = False):
        if inputval is None:
            inputval = dict()

        self._pre_init(sort=sort)

        # fast track for itemcontainer from dataset/subclass
        if isinstance(inputval, ItemContainer):
            self._init_from_itemcontainer(inputval)

        elif isinstance(inputval, list):
            # dataset raises an error, pdataset does not
            raise TypeError(
                'Dataset can be created from list or iterable of values with Dataset.concat_rows(), Dataset.concat_columns, Dataset.from_rows() or Dataset.from_tagged_rows().')

        # all other initializers will be flipped to a dictionary, or raise an error
        else:
            inputval = self._init_columns_as_dict(inputval, base_index=base_index, sort=sort, unicode=unicode)
            self._init_from_dict(inputval, unicode=unicode)

        self._post_init()

    # ------------------------------------------------------------
    def _init_columns_as_dict(self, columns, base_index=0, sort=True, unicode=False):
        """
        Most methods of dataset construction will be turned into a dictionary before
        setting dataset columns. This will return the resulting dictionary for each type
        or raise an error.
        """

        if isinstance(columns, dict):
            pass

        # TODO: pull out itemcontainer
        elif isinstance(columns, Struct):
            columns = columns._as_dictionary()

        # check for pandas without importing
        elif columns.__class__.__name__ == 'DataFrame':
            columns = self._init_from_pandas_df(columns, unicode=unicode)

        # record arrays have a void dtype
        elif isinstance(columns, np.ndarray):
            if columns.dtype.char == 'V':
                columns = _possibly_convert_rec_array(columns)
            else:
                raise TypeError(f"Can only initialize datasets from arrays that are numpy record arrays.")

        # If we get an Iterable of 2-tuples (a string key and a list/iterable/array)
        # or an iterable of arrays (where we'll generate names like 'col_0', 'col_1', etc.).
        # NOTE: The latter one shouldn't go here; it should go in Dataset.from_rows() or similar instead.
        elif isinstance(columns, abc.Iterable) and not isinstance(columns, (str, bytes)):
            raise NotImplementedError("Need to implement support for creating a Dataset from an iterable.")

        else:
            raise TypeError('Unexpected argument in Dataset.__init__', type(columns))

        return columns

    # ------------------------------------------------------------
    def _init_from_itemcontainer(self, columns):
        """
        Store the itemcontainer and set _nrows.
        """
        self._all_items = columns
        self._nrows = len(list(self._all_items.values())[0][0])

    # ------------------------------------------------------------
    def _pre_init(self, sort=False):
        """
        Leave this here to chain init that only Dataset has.
        """
        super()._pre_init()
        self._sort_display = sort

    # ------------------------------------------------------------
    def _post_init(self):
        """
        Leave this here to chain init that only Dataset has.
        """
        super()._post_init()

    # ------------------------------------------------------------
    def _possibly_convert_array(self, v, name, unicode=False):
        """
        If an array contains objects, it will attempt to flip based on the type of the first item.

        By default, flip any numpy arrays to FastArray. (See UseFastArray flag)
        The constructor will warn the user whenever object arrays appear, and raise an error if conversion
        was unsuccessful.

        Examples
        --------
        String objects:

        >>> ds = rt.Dataset({'col1': np.array(['a','b','c'], dtype=object)})
        >>> ds.col1
        FastArray([b'a', b'b', b'c'], dtype='|S1')

        Numeric objects:

        >>> ds = rt.Dataset({'col1': np.array([1.,2.,3.], dtype=object)})
        >>> ds.col1
        FastArray([1., 2., 3.])

        Mixed type objects:

        >>> ds = rt.Dataset({'col1': np.array([np.nan, 'str', 1], dtype=object)})
        ValueError: could not convert string to float: 'str'
        TypeError: Cannot handle a numpy object array of type <class 'float'>

        Note: depending on the order of mixed types in an object array, they may be converted to strings.
              for performance, only the type of the first item is examined

        Mixed type objects starting with string:

        >>> ds = rt.Dataset({'col1': np.array(['str', np.nan, 1], dtype=object)})
        >>> ds.col1
        FastArray([b'str', b'nan', b'1'], dtype='|S3')
        """
        if self.UseFastArray:
            # flip value to FastArray
            if not isinstance(v, TypeRegister.Categorical):
                if isinstance(v, np.ndarray):
                    c = v.dtype.char
                    if c == 'O':
                        # make sure, scalar type so no python objects like dicts come through
                        # try float, but most objects will flip to bytes or unicode
                        # TODO: Simplify to use np.isscalar() here?
                        if isinstance(v[0], (str, np.str_, bytes, np.bytes_, int, float, bool, np.integer, np.floating, np.bool_)):
                            try:
                                # attempt to autodetect based on first element
                                # NOTE: if the first element is a float and Nan.. does that mean keep looking?
                                if isinstance(v[0], (str, np.str_)):
                                    # NOTE this might get converted to 'S' if unicode is False for FastArrays
                                    v=v.astype('U')
                                elif isinstance(v[0], (bytes, np.bytes_)):
                                    v=v.astype('S')
                                elif isinstance(v[0], (int, np.integer)):
                                    v=v.astype(np.int64)
                                elif isinstance(v[0], (bool, np.bool_)):
                                    v=v.astype(np.bool_)
                                else:
                                    v = v.astype(np.float64)
                            except:
                                v = self._object_as_string(name, v)
                        else:
                            raise TypeError(f'Cannot convert object array {v} containing {type(v[0])}')

                    elif c == 'M':
                    # handle numpy datetime, will be in UTC
                        v = TypeRegister.DateTimeNano(v, from_tz='GMT', to_tz='GMT')

                    # numpy arrays with bytes will be converted here unless unicode was requested
                    # fast arrays will not be flipped, even if unicode
                    if not isinstance(v, FastArray):
                        v = FastArray(v, unicode=unicode)
        else:
            if isinstance(v, FastArray):
                v = v._np
        # possible expanson of scalars or arrays of 1
        if v.shape[0]== 1 and self._nrows is not None and self._nrows > 1:
            # try to use repeat to solve mismatch problem
            v = v.repeat(self._nrows)
        return v

    # ------------------------------------------------------------
    def _object_as_string(self, name, v):
        """
        After failing to convert objects to a numeric type, or when the first item is
        a string or bytes, try to flip the array to a bytes array, then unicode array.
        """
        try:
            v = v.astype('S')
        except (UnicodeEncodeError, SystemError):
            try:
                v = v.astype('U')
            except:
                raise ValueError(f"Object strings could not be converted to bytestrings or unicode for {name!r}. First item was {type(v[0])}")
        return v

    # ------------------------------------------------------------
    def _possibly_convert(self, name, v, unicode=False):
        """
        Input: any data type that can be added to a dataset
        Returns: a numpy based array
        """
        if not isinstance(v, np.ndarray):
            # pandas Series containing Categorical
            if hasattr(v, 'cat'):
                v = TypeRegister.Categorical(v.values)
            # pandas Categorical
            elif hasattr(v, '_codes'):
                v = TypeRegister.Categorical(v)
            elif isinstance(v, (tuple, Struct)):
                raise TypeError(f'Cannot create a Dataset column out of a {type(v).__name__}.')
            elif not isinstance(v, list):
                # convert scalar to list then to array
                v = np.asanyarray([v])
            else:
                # convert list to an array
                v = np.asanyarray(v)
            v = self._ensure_vector(v)
        v = self._possibly_convert_array(v, name, unicode=unicode)
        return v

    # ------------------------------------------------------------
    def _ensure_vector(self, vec):
        if len(vec.shape) != 1:
            vec = vec.squeeze()
            if len(vec.shape) == 0:
                vec = vec.reshape((1,))
        return vec

    # ------------------------------------------------------------
    def _check_addtype(self, name, value):
        # TODO use _possibly_convert -- why are these two routines different?
        if not isinstance(value, np.ndarray):
            if isinstance(value,set):
                raise TypeError(f'Cannot create Dataset column {name!r} out of tuples or sets {value!r}.')
            # following pandas
            if self._nrows is None:
                if isinstance(value, (list, tuple)):
                    self._nrows = len(value)
                else:
                    # how to get here:
                    # ds=Dataset()
                    # ds[['g','c']]=3
                    self._nrows = 1

            if isinstance(value, (list, tuple)):
                rowlen = len(value)
                if self._nrows != rowlen and rowlen !=1:
                    raise TypeError("Row mismatch in Dataset._check_addtype", self._nrows, len(value), value)
                value = np.asanyarray(value)
                if value.shape[0] ==1 and self._nrows != 1:
                    # for when user types in a list of 1 item and wants it to repeat
                    value = value.repeat(self._nrows)
            else:
                # if they try to add a dataset to a single column
                # then if the dataset has one column, use that
                if isinstance(value, Dataset):
                    if self._nrows != value._nrows:
                        raise TypeError("Row mismatch in Dataset._check_addtype.  Tried to add Dataset of different lengths", self._nrows, value._nrows)

                    if value._ncols==1:
                        return value[0]
                    else:
                        # skip over groupbykeys
                        labels = value.label_get_names()
                        count =0
                        first = None
                        # loop over all columns, not including labels
                        for c in value.keys():
                            if c not in labels:
                                first = c
                                count += 1
                        if count == 1:
                            return value[first]
                        else:
                            # perhaps see if we can find the same name?
                            raise TypeError(f"Cannot determine which column of Dataset to add to the Dataset column {name!r}.")

                if callable(getattr(value, 'repeat', None)):
                    # for when user types in a list of 1 item and wants it to repeat to match dataset row length
                    value = value.repeat(self._nrows)
                else:
                    try:
                        # NOT an array, or a list, tuple, or Dataset at this point
                        value = full(self._nrows, value)
                    except Exception as ex:
                        raise TypeError(f'Cannot create a single Dataset column {name!r} out of type {type(value)!r}.  Error {ex}')

            value = self._ensure_vector(value)

        # this code will add the name
        value = self._possibly_convert_array(value, name)
        self._check_add_dimensions(value)

        return value

    # ------------------------------------------------------------
    def _init_from_pandas_df(self, df, unicode=False):
        """
        Pulls data from pandas dataframes. Uses get attribute, so does not need to import pandas.
        """
        df_dict = {}
        for k in df.columns:
            col = df[k]
            # categoricals will be preserved in _possibly_convert
            if hasattr(col, 'cat'):
                pass
            # series column (added with underlying array)
            elif hasattr(col, 'values'):
                col = col.values
            else:
                raise TypeError(f"Cannot initialize column of type {type(col)}")
            #col = self._possibly_convert(k, col, unicode=unicode)
            df_dict[k] = col
        return df_dict

    # ------------------------------------------------------------
    def _init_from_dict(self, dictionary, unicode=False):
        # all __init__ paths funnel into this
        allnames = Struct.AllNames
        self._validate_names(dictionary)
        self._nrows = None
        self._ncols = 0

        if allnames:
            for colname, arr in dictionary.items():
                arr = self._possibly_convert(colname, arr, unicode=unicode)
                self._add_allnames(colname, arr, 0)
        else:
            for colname, arr in dictionary.items():
                if colname[0] != '_':
                    # many different types of data can be passed in here
                    arr = self._possibly_convert(colname, arr, unicode=unicode)
                    # add the array to this class
                    self._superadditem(colname, arr)

        # pull the items so getattr doesn't need to be called
        items = self._all_items.get_dict_values()
        for i in items:
            # dict values are in a list
            col = i[0]
            self._check_add_dimensions(col)

        # as in pandas DataFrame, these are attributes that must be updated when modifying columns/rows
        # self._superadditem('columns', list(self.keys()))

    # ------------------------------------------------------------
    def _check_add_dimensions(self, col):
        """
        Used in _init_from_dict and _replaceitem.
        If _nrows has not been set, it will be here.
        """
        if col.ndim > 0:
            if self._nrows is None:
                self._nrows = col.shape[0]
            else:
                if self._nrows != col.shape[0]:
                    raise ValueError(f"Column length mismatch in Dataset constructor: Dataset had {self._nrows}, cannot add column with length {col.shape[0]} and ndims {col.ndim} col : {col}")
        else:
            raise ValueError(f"Datasets only support columns of 1 or more dimensions. Got {col.ndim} dimensions.")

    # ------------------------------------------------------------
    def __del__(self):
        # print("**Tell the sort cache we are gone")
        # print(f"dataset size deleted")
        # import traceback
        # traceback.print_stack()
        try:
            SortCache.invalidate(self._uniqueid)
        except AttributeError:
            pass

    # --------------------------------------------------------
    def _copy_attributes(self, ds, deep=False):
        """
        After constructing a new dataset or pdataset, copy over attributes for sort, labels, footers, etc.
        Called by Dataset._copy(), PDataset._copy()
        """
        # copy over the sort list
        if self._col_sortlist is not None:
            new_sortlist = [_k for _k in self._col_sortlist if _k in ds]
            if len(new_sortlist) > 0:
                ds._col_sortlist = new_sortlist

        # reassign labels
        ds.label_set_names(self.label_get_names())

        # copy footers
        # TODO NW The _footers is now deprecated, I think, and should be removed throughout
        if hasattr( self, '_footers' ):
            footers = {}
            for f, item in self._footers.items():
                footers[f] = item.copy() if (deep and item) else item
            ds._footers = footers

        return ds

    # --------------------------------------------------------
    def _copy(self, deep=False, rows=None, cols=None, base_index=0, cls=None):
        """
        Bracket indexing that returns a dataset will funnel into this routine.

        deep : if True, perform a deep copy on column array
        rows : row mask
        cols : column mask
        base_index : used for head/tail slicing
        cls : class of return type, for subclass super() calls
        First argument must be deep.  Deep cannnot be set to None.  It must be True or False.
        """
        if cls is None:
            cls = type(self)

        newcols = self._as_itemcontainer(deep=deep, rows=rows, cols=cols, base_index=base_index)
        # newcols is either an ItemContainer or a dictionary
        ds = cls(newcols, base_index=base_index)
        ds = self._copy_attributes(ds, deep=deep)

        ## # ! TO DO fixup sortkeys, this block would change type of self._col_sortlist from [] to {}.
        ## if self._col_sortlist is not None:
        ##     # copy the dictionary
        ##     # TODO: turn these keys into new_sort or active sort if there wasn't one
        ##     keylist =  {_k:  _v for _k, _v in self._col_sortlist.items()}
        ##     # also copy keylist here
        ##     keylist = self._copy_from_dict(keylist, copy=deep, rows=rows, cols=cols)
        ##     ds._col_sortlist = keylist
        return ds

    # --------------------------------------------------------
    def _as_itemcontainer(self, deep=False, rows=None, cols=None, base_index=0):
        """
        Returns an ItemContainer object for quick reconstruction or slicing/indexing of a dataset.
        Will perform a deep copy if requested and necessary.
        """
        def apply_rowmask(arr, mask):
            # callback for applying mask/slice to columns
            name = arr.get_name()
            arr = arr[mask]
            arr.set_name(name)
            return arr

        if rows is None:
            # item container copy, with or without a column selection
            newcols = self._all_items.copy(cols=cols)

        else:
            # get array data, slice, send back to item container for copy
            # slice will take a view of array (same memory)
            # boolean/fancy index will always make copy
            # will also slice/restore FastArray subclasses
            newcols = self._all_items.copy_apply(apply_rowmask, rows, cols=cols)

        # only slices, full arrays need a deep copy
        if deep and (isinstance(rows, slice) or rows is None):
            for v in newcols.iter_values():
                name = v[0].get_name()
                v[0] = v[0].copy()
                v[0].set_name(name)
                # deep copy item_attributes
                for i, vn in enumerate(v[1:]):
                    v[i+1] = vn.copy() if hasattr(vn, 'copy') else vn

        return newcols

    # --------------------------------------------------------
    def _autocomplete(self) -> str:
        return f'Dataset{self.shape}'

    # --------------------------------------------------------
    def copy(self, deep=True):
        """
        Make a copy of the Dataset.

        Parameters
        ----------
        deep : bool
            Indicates whether the underlying data should be copied too. Defaults to True.

        Returns
        -------
        Dataset

        Examples
        --------
        >>> ds = rt.Dataset({'a': np.arange(-3,3), 'b':3*['A', 'B'], 'c':3*[True, False]})
        >>> ds
        #    a   b       c
        -   --   -   -----
        0   -3   A    True
        1   -2   B   False
        2   -1   A    True
        3    0   B   False
        4    1   A    True
        5    2   B   False

        >>> ds1 = ds.copy()
        >>> ds.a = ds.a + 1
        >>> ds1
        #    a   b       c
        -   --   -   -----
        0   -3   A    True
        1   -2   B   False
        2   -1   A    True
        3    0   B   False
        4    1   A    True
        5    2   B   False

        Even though we have changed ds, ds1 remains unchanged.

        """
        return self._copy(deep)

    # --------------------------------------------------------
    def filter(self, rowfilter: np.ndarray, inplace:bool=False) -> 'Dataset':
        """
        Use a row filter to make a copy of the Dataset.

        Parameters
        ----------
        rowfilter: array, fancy index or boolean mask
        inplace : bool
            When set to True will reduce memory overhead. Defaults to False.

        Examples
        --------
        Filter a Dataset using the least memory possible

        >>> ds = rt.Dataset({'a': rt.arange(10_000_000), 'b': rt.arange(10_000_000.0)})
        >>> f = rt.logical(rt.arange(10_000_000) % 2)
        >>> ds.filter(f, inplace=True)
            #         a           b
        -------   -------   ---------
            0         1        1.00
            1         3        3.00
            2         5        5.00
            ...       ...         ...
        4999997   9999995   1.000e+07
        4999998   9999997   1.000e+07
        4999999   9999999   1.000e+07
        <BLANKLINE>
        [5000000 rows x 2 columns] total bytes: 57.2 MB
        """
        if inplace:
            # normalize rowfilter
            if np.isscalar(rowfilter):
                rowfilter=np.asanyarray([rowfilter])
            elif not isinstance(rowfilter, np.ndarray):
                rowfilter=np.asanyarray(rowfilter)

            self._all_items.copy_inplace(rowfilter)

            # check for boolean array
            if rowfilter.dtype.char == '?':
                newlen = np.sum(rowfilter)
            else:
                newlen = len(rowfilter)
            self._nrows = newlen

            return self
        else:
            return self._copy(False, rowfilter)

    def get_nrows(self):
        """
        Get the number of elements in each column of the Dataset.

        Returns
        -------
        int
            The number of elements in each column of the Dataset.
        """
        return self._nrows

    ## -------------------------------------------------------
    #def save_uncompressed(self, path, name):
    #    """
    #    *not implemented*
    #    """
    #    self.save(self, path, name, compress=False)

    # -------------------------------------------------------
    def save(self, path: Union[str, os.PathLike] = '', share: Optional[str] = None, compress:bool=True, overwrite:bool=True, name: Optional[str] = None, onefile:bool=False,
            bandsize: Optional[int] = None, append: Optional[str] = None, complevel: Optional[int] = None):
        """
        Save a dataset to a single .sds file or shared memory.

        Parameters
        ----------
        path : str or os.PathLike
            full path to save location + file name (if no .sds extension is included, it will be added)
        share : str, optional
            Shared memory name. If set, dataset will be saved to shared memory and NOT to disk
            when shared memory is specified, a filename must be included in path. only this will be used,
            the rest of the path will be discarded.
        compress : bool
            Use compression when saving the file. Shared memory is always saved uncompressed.
        overwrite : bool
            Defaults to True. If False, prompt the user when overwriting an existing .sds file;
            mainly useful for Struct.save(), which may call Dataset.save() multiple times.
        name : str, optional
        bandsize : int, optional
            If set to an integer > 10000 it will compress column data every bandsize rows
        append : str, optional
            If set to a string it will append to the file with the section name.
        complevel : int, optional
            Compression level from 0 to 9. 2 (default) is average. 1 is faster, less compressed, 3 is slower, more compressed.

        Examples
        --------
        >>> ds = rt.Dataset({'col_'+str(i):a rt.range(5) for i in range(3)})
        >>> ds.save('my_data')
        >>> os.path.exists('my_data.sds')
        True

        >>> ds.save('my_data', overwrite=False)
        my_data.sds already exists and is a file. Overwrite? (y/n) n
        No file was saved.

        >>> ds.save('my_data', overwrite=True)
        Overwriting file with my_data.sds

        >>> ds.save('shareds1', share='sharename')
        >>> os.path.exists('shareds1.sds')
        False

        See Also
        --------
        Dataset.load(), Struct.save(), Struct.load(), load_sds(), load_h5()
        """
        if share is not None:
            if path=='':
                raise ValueError(f'Must provide single .sds file name for item with share name {share}. e.g. my_ds.save("dataset1.sds", share="{share}")')

        save_sds(path, self, share=share, compress=compress, overwrite=overwrite, name=name, onefile=onefile, bandsize=bandsize, append=append, complevel=complevel)

    # -------------------------------------------------------
    @classmethod
    def load(cls, path: Union[str, os.PathLike] = '', share=None, decompress:bool=True, info:bool=False, include: Optional[Sequence[str]] = None,
             filter: Optional[np.ndarray] = None, sections: Optional[Sequence[str]] = None, threads: Optional[int] = None):
        """
        Load dataset from .sds file or shared memory.

        Parameters
        ----------
        path : str
            full path to load location + file name (if no .sds extension is included, it will be added)
        share : str, optional
            shared memory name. loader will check for dataset in shared memory first. if it's not there, the
            data (if file found on disk) will be loaded into the user's workspace AND shared memory. a sharename
            must be accompanied by a file name. (the rest of a full path will be trimmed off internally)
        decompress : bool
            **not implemented. the internal .sds loader will detect if the file is compressed
        info : bool
            Defaults to False. If True, load information about the contained arrays instead of loading them from file.
        include : sequence of str, optional
            Defaults to None. If provided, only load certain columns from the dataset.
        filter : np.ndarray of int or np.ndarray of bool, optional
        sections : sequence of str, optional
        threads : int, optional
            Defaults to None. Request certain number of threads during load.

        Examples
        --------
        >>> ds = rt.Dataset({'col_'+str(i):np.random.rand(5) for i in range(3)})
        >>> ds.save('my_data')
        >>> rt.Dataset.load('my_data')
        #   col_0   col_1   col_2
        -   -----   -----   -----
        0    0.94    0.88    0.87
        1    0.95    0.93    0.16
        2    0.18    0.94    0.95
        3    0.41    0.60    0.05
        4    0.53    0.23    0.71

        >>> ds = rt.Dataset.load('my_data', share='sharename')
        >>> os.remove('my_data.sds')
        >>> os.path.exists('my_data.sds')
        False

        >>> rt.Dataset.load('my_data', share='sharename')
        #   col_0   col_1   col_2
        -   -----   -----   -----
        0    0.94    0.88    0.87
        1    0.95    0.93    0.16
        2    0.18    0.94    0.95
        3    0.41    0.60    0.05
        4    0.53    0.23    0.71
        """
        return load_sds(path, share=share, info=info, include=include, filter=filter, sections=sections, threads=threads)

    # -------------------------------------------------------
    @property
    def size(self) -> int:
        """
        Number of elements in the Dataset (nrows x ncols).

        Returns
        -------
        int
            The number of elements in the Dataset (nrows x ncols).
        """
        return self._ncols * self._nrows

    ### We can recreate this once we have a non-display transpose() method.
    ## @property
    ## def T(self):
    ##     return self.transpose()

    # -------------------------------------------------------
    def _add_allnames(self, colname, arr, nrows) -> None:
        '''
        Internal routine used to add columns only when AllNames is True.
        '''
        if nrows == 0 or nrows == self.get_nrows():
            if self._all_items.item_exists(colname):
                self._replaceitem_allnames(colname, arr)
            else:
                self._addnewitem_allnames(colname, arr)
        else:
            raise NotImplementedError(f'Cannot set {colname!r} because rows are different lengths.')

    # -------------------------------------------------------
    def __setitem__(self, fld, value):
        """
        Parameters
        ----------
        fld : (rowspec, colspec) or colspec (=> rowspec of :)
        value : scalar, sequence or dataset value

            * scalar is always valid
            * if (rowspec, colspec) is an NxK selection:
                (1xK), K>1: allow |sequence| == K
                (Nx1), N>1: allow |sequence| == N
                (NxK), N, K>1: allow only w/ |dataset| = NxK
            * sequence can be list, tuple, np.ndarray, FastArray

        Raises
        ------
        IndexError
        """
        def setitem_mask(arr, mask, value):
            arr[mask] = value
        def setitem_fill(value, nrows):
            return full

        col_idx, row_idx, ncols, nrows, row_arg = self._extract_indexing(fld)
        if col_idx is None:
            col_idx = list(self.keys())

        # BUG: set item with dataset for only one column
        #print('col_idx',col_idx)
        #print('row_idx',row_idx)
        #print('ncols',ncols)
        #print('row_arg',row_arg)

        if ncols <= 1:
            # this path is also for when the dataset is empty
            if not isinstance(col_idx, str): col_idx = col_idx[0]
            if col_idx in self:
                if row_idx is None:
                    self.__setattr__(col_idx, value)
                    #self._superadditem(col_idx, value)
                    #setattr(self, col_idx, value)
                else:
                    # apply row mask
                    arr=getattr(self, col_idx)

                    # setting a single col dataset from a dataset
                    if isinstance(value, Dataset):
                        arr[row_idx] = value[0]
                    else:
                        arr[row_idx] = value
            elif Struct.AllNames:
                self._add_allnames(col_idx, value, nrows)

            elif self.is_valid_colname(col_idx):
                if nrows == self.get_nrows() or nrows ==0:
                    self.__setattr__(col_idx, value)
                else:
                    raise NotImplementedError(f'Cannot set {col_idx!r} because rows are different lengths.')
            elif col_idx in ['True','False','None']:
                col_idx = col_idx.lower()
                if nrows == self.get_nrows() or nrows ==0:
                    self.__setattr__(col_idx, value)
                else:
                    raise NotImplementedError(f'Cannot set {col_idx!r} because rows are different lengths.')
            else:
                raise IndexError(f'Invalid column name: {col_idx!r}')
        elif nrows == 1:
            if not all(self.col_exists(colname) for colname in col_idx):
                raise IndexError('If creating a new column can only do one at a time.')
            if np.isscalar(value):
                self._all_items.apply(setitem_mask, row_idx, value, cols=col_idx)

            elif isinstance(value, Dataset) and value.shape == (1, len(col_idx)):
                # this case comes up crucially in ds[3, :] /= 2, for example
                for colname, _cn in zip(col_idx, value):
                    getattr(self, colname)[row_idx] = value[_cn][0]
            elif len(value) == len(col_idx):
                for colname, array in zip(col_idx, value):
                    getattr(self, colname)[row_idx] =array
            else:
                raise ValueError('Must have equal len keys and value when setting with a sequence.')
        else:
            if np.isscalar(value):
                #if not all(self.col_exists(_k) for _k in col_idx):
                #    raise IndexError('If creating a new column can only do one at a time.')
                if row_idx is not None:
                    self._all_items.apply(setitem_mask, row_idx, value, cols=col_idx)

                else:
                    # fill column with scalar
                    for colname in col_idx:
                        setattr(self, colname, value)

            elif isinstance(value, Dataset):
                # TJD 10.2018 - the row mask appears to have already been applied to value
                # NOTE: if the row mask is a boolean, we could sum it to get the count
                # NOTE: if the row mask is fancy indexing, we could get length
                if row_idx is not None and col_idx is not None:
                    # both row and col mask
                    for i,c in enumerate(col_idx):
                        # inplace operation
                        #self[i][row_idx] = value[i]
                        getattr(self, c)[row_idx]=value[i]
                elif row_idx is not None:
                    #no col mask
                    for i in range(ncols):
                        # inplace operation
                        self[i][row_idx] = value[i]
                elif col_idx is not None:
                    #no row mask
                    # example:  ds[['g','c']]=Dataset({'a':arange(10),'b':arange(10.0)}):
                    for i,c in enumerate(col_idx):
                        setattr(self, c, value[i])
                else:
                    #no row and no col mask
                    for i in range(ncols):
                        self[i] = value[i]
            else:
                raise ValueError(f'Must have same-shape Dataset when setting {nrows}x{ncols} sub-Dataset. Type: {type(value)}')
        return

    # -------------------------------------------------------
    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : (rowspec, colspec) or colspec

        Returns
        -------
        the indexed row(s), cols(s), sub-dataset or single value

        Raises
        ------
        IndexError
            When an invalid column name is supplied.
        TypeError
        """
        def single_array(col_idx, row_idx):
            # will either return or return an error
            try:
                np_arr = self.col_get_value(col_idx)
            except:
                raise IndexError(f"Could not find column named: {col_idx}")

            if row_idx is not None:
                # array indexing takes place early here
                return np_arr[row_idx]
            else:
                return np_arr

        # optimization for default case
        if isinstance(index, str):
            return self.col_get_value(index)

        col_idx, row_idx, ncols, nrows, row_arg = self._extract_indexing(index)

        # check for a single string which selects a single column
        if isinstance(col_idx, str):
            return single_array(col_idx, row_idx)

        # if a single integer specified, make a list of one number for fancy column indexing
        if isinstance(row_arg, (int, np.integer)):
            row_idx = [row_arg]

        return self._copy(deep=False, rows=row_idx, cols=col_idx)

    # ------------------------------------------------------------
    def _dataset_compare_check(self, func_name, lhs):
        # comparison function will be called by an array the size of the indexes, either
        # interperetted as integers, or as categorical strings
        # if compared to string, make sure the string matches the string type in categories
        if isinstance(lhs, Dataset):
            nrows = self.get_nrows()
            if lhs.get_nrows() != nrows:
                # Allow is length is 1 so that broadcasting applies?
                # N.B. Right now this causes a DeprecationWarning in numpy, not sure what type it will be.
                raise ValueError("The two Datasets have different lengths and cannot be compared")
            else:
                # returns a new dataset
                newds = {}
                # for all columns that match
                for colname in self.keys():
                    # if the lhs dataset has the same column name, compare
                    if hasattr(lhs, colname):
                        # get the function reference for the comparison operator
                        func = getattr(self[colname], func_name)
                        # add the boolean array to the new dataset
                        newds[colname] = func(lhs[colname])
                    else:
                        newds[colname] = np.array([False] * nrows)
                for colname in lhs:
                    if colname not in newds:
                        newds[colname] = np.array([False] * nrows)
                return type(self)(newds)
        else:
            raise TypeError(f'Cannot compare a Dataset to type {type(lhs).__name__}.')

    # ------------------------------------------------------------
    def __ne__(self, lhs):
        return self._dataset_compare_check('__ne__', lhs)

    def __eq__(self, lhs):
        return self._dataset_compare_check('__eq__', lhs)

    def __ge__(self, lhs):
        return self._dataset_compare_check('__ge__', lhs)

    def __gt__(self, lhs):
        return self._dataset_compare_check('__gt__', lhs)

    def __le__(self, lhs):
        return self._dataset_compare_check('__le__', lhs)

    def __lt__(self, lhs):
        return self._dataset_compare_check('__lt__', lhs)

    # ------------------------------------------------------------
    def __len__(self):
        # Debated October 2019
        # For Dataset we will return the number of rows for length
        rows= self._nrows
        if rows is None:
            rows = 0
        return rows

    # ------------------------------------------------------------
    def putmask(self, mask, values):
        """
        Call riptable ``putmask`` routine which is faster than ``__setitem__`` with bracket indexing.

        Parameters
        ----------
        mask : ndarray of bools
            boolean numpy array with a length equal to the number of rows in the dataset.
        values : rt.Dataset or ndarray
            * Dataset: Corresponding column values will be copied, must have same shape as calling dataset.
            * ndarray: Values will be copied to each column, must have length equal to calling dataset's nrows.

        Returns
        -------
        None

        Examples
        --------
        >>> ds = rt.Dataset({'a': np.arange(-3,3), 'b':np.arange(6), 'c':np.arange(10,70,10)})
        >>> ds
        #    a   b    c
        -   --   -   --
        0   -3   0   10
        1   -2   1   20
        2   -1   2   30
        3    0   3   40
        4    1   4   50
        5    2   5   60

        >>> ds1 = ds.copy()
        >>> ds.putmask(ds.a < 0, np.arange(100,106))
        >>> ds
        #     a     b     c
        -   ---   ---   ---
        0   100   100   100
        1   101   101   101
        2   102   102   102
        3     0     3    40
        4     1     4    50
        5     2     5    60

        >>> ds.putmask(np.array([True, True, False, False, False, False]), ds1)
        >>> ds
        #     a     b     c
        -   ---   ---   ---
        0    -3     0    10
        1    -2     1    20
        2   102   102   102
        3     0     3    40
        4     1     4    50
        5     2     5    60


        """

        if not(isinstance(mask, np.ndarray) and mask.dtype.char == '?' and len(mask) == self._nrows):
            raise ValueError(f"Mask must be a boolean numpy array of the same length as the number of rows in the dataset.")

        if isinstance(values, Dataset):
            if self.shape == values.shape:
                col_src = list(values.values())
                col_dst = list(self.values())

                for i in range(self._ncols):
                    putmask( col_dst[i], mask, col_src[i] )

            else:
                raise ValueError(f"Dataset put values must have same shape as other dataset. Got {self.shape} vs. {values.shape}")

        elif isinstance(values, np.ndarray):
            if len(values) == self._nrows:
                col_dst = list(self.values())
                for i in range(self._ncols):
                    putmask( col_dst[i], mask, values )

            else:
                raise ValueError(f"Array put values must have a length equal to dataset's rows. Got {len(values)} vs. {self._nrows}")

        else:
            raise TypeError(f"Cannot call dataset putmask with type {type(values)}.")


    ## ------------------------------------------------------------
    #def iterrows(self):
    #    """
    #    NOTE: This routine is slow

    #    It returns a struct with scalar values for each row.
    #    It does not preserve dtypes.

    #    Do not modify anything you are iterating over.

    #    Example:
    #    --------
    #    >>> ds=Dataset({'test':arange(10)*3, 'test2':arange(10.0)/2})
    #    >>> temp=[*ds.iterrows()]
    #    >>> temp[2]
    #    (2,
    #     #   Name    Type      Size   0     1   2
    #     -   -----   -------   ----   ---   -   -
    #     0   test    int32     0      27
    #     1   test2   float64   0      4.5

    #     [2 columns])

    #    """
    #    mykeys = self.keys()
    #    temp_struct = TypeRegister.Struct({colname:0 for colname in mykeys})

    #    # for all the rows in the dataset
    #    for rownum in range(self._nrows):
    #        # for all the columns
    #        for colname in mykeys:
    #            temp_struct[colname]=self[colname][rownum]
    #        yield rownum, temp_struct

    # ------------------------------------------------------------
    def iterrows(self):
        """
        NOTE: This routine is slow

        It returns a struct with scalar values for each row.
        It does not preserve dtypes.

        Do not modify anything you are iterating over.

        Examples
        --------
        >>> ds = rt.Dataset({'test': rt.arange(10)*3, 'test2': rt.arange(10.0)/2})
        >>> temp=[*ds.iterrows()]
        >>> temp[2]
        (2,
         #   Name    Type      Size   0     1   2
         -   -----   -------   ----   ---   -   -
         0   test    int32     0      27
         1   test2   float64   0      4.5
        <BLANKLINE>
         [2 columns])
        """
        full_columns = tuple(self.values())
        temp_struct = TypeRegister.Struct({})

        # make shallow copies of all lists containing column data, so original columns don't swapped out
        temp_items = self._all_items._items.copy()
        temp_struct._all_items._items = temp_items
        for k, v in temp_items.items():
            temp_items[k] = v.copy()

        # manually set item dict, number of columns
        temp_struct._all_items._items = temp_items
        temp_struct._ncols = self._ncols

        # these values will be swapped internally
        temp_vals = temp_struct._all_items.get_dict_values()

        # check if any there are any array/fastarray subclasses in the columns
        np_safe = True
        for v in full_columns:
            if TypeRegister.is_array_subclass(v):
                np_safe = False
                break

        # if there are no subclasses in the dataset, we take the fast path and call np getitem directly
        if np_safe:
            # faster to store function pointer
            npget = np.ndarray.__getitem__

            # for each row, swap out the item values in the temporary struct's item container
            for rownum in range(self._nrows):
                for ci in range(self._ncols):
                    temp_vals[ci][0] = npget(full_columns[ci],rownum)
                yield rownum, temp_struct

        else:
            # for each row, swap out the item values in the temporary struct's item container
            for rownum in range(self._nrows):
                for ci in range(self._ncols):
                    temp_vals[ci][0] = full_columns[ci][rownum]
                yield rownum, temp_struct

    # ------------------------------------------------------------
    def isin(self, values):
        """
        Call :meth:`~rt.rt_fastarray.FastArray.isin` for each column in the `Dataset`.

        Parameters
        ----------
        values : scalar or list or array_like
            A list or single value to be searched for.

        Returns
        -------
        Dataset
            Dataset of boolean arrays with the same column headers as the original dataset.
            True indicates that the column element occurred in the provided values.

        Notes
        -----
        Note: different behavior than pandas DataFrames:

        * Pandas handles object arrays, and will make the comparison for each element type in the provided list.
        * Riptable favors bytestrings, and will make conversions from unicode/bytes to match for operations as necessary.
        * We will also accept single scalars for values.

        Examples
        --------
        >>> data = {'nums': rt.arange(5), 'strs': rt.FA(['a','b','c','d','e'], unicode=True)}
        >>> ds = rt.Dataset(data)
        >>> ds.isin([2, 'b'])
        #    nums    strs
        -   -----   -----
        0   False   False
        1   False    True
        2   False   False
        3   False   False
        4   False   False

        >>> df = pd.DataFrame(data)
        >>> df.isin([2, 'b'])
            nums   strs
        0  False  False
        1  False   True
        2   True  False
        3  False  False
        4  False  False

        See Also
        --------
        pandas.DataFrame.isin()
        """
        # this is repeat code from FastArray isin, but this way, the values only need to be converted once for each column
        #x = values
        #if isinstance(values, (bool, np.bool_, bytes, str, int, np.integer, float, np.floating)):
        #    x = np.array([x])

        ## numpy will find the common dtype (strings will always win)
        #elif isinstance(x, list):
        #    x = np.array(x)

        data = {}
        for name, col in self.items():
            data[name] = col.isin(values)
        return type(self)(data)

    # -------------------------------------------------------
    @property
    def imatrix(self) -> Optional[np.ndarray]:
        """
        Returns the 2d array created from `imatrix_make`.

        Returns
        -------
        imatrix : np.ndarray, optional
            If `imatrix_make` was previously called, returns the 2D array created and cached internally
            by that method. Otherwise, returns ``None``.

        Examples
        --------
        >>> ds = rt.Dataset({'a': np.arange(-3,3), 'b':np.arange(6), 'c':np.arange(10,70,10)})
        >>> ds
        #    a   b    c
        -   --   -   --
        0   -3   0   10
        1   -2   1   20
        2   -1   2   30
        3    0   3   40
        4    1   4   50
        5    2   5   60

        >>> ds.imatrix  # returns nothing since we have not called imatrix_make
        >>> ds.imatrix_make()
        FastArray([[-3,  0, 10],
                   [-2,  1, 20],
                   [-1,  2, 30],
                   [ 0,  3, 40],
                   [ 1,  4, 50],
                   [ 2,  5, 60]])
        >>> ds.imatrix
        FastArray([[-3,  0, 10],
                   [-2,  1, 20],
                   [-1,  2, 30],
                   [ 0,  3, 40],
                   [ 1,  4, 50],
                   [ 2,  5, 60]])

        >>> ds.a = np.arange(6)
        >>> ds
        #   a   b    c
        -   -   -   --
        0   0   0   10
        1   1   1   20
        2   2   2   30
        3   3   3   40
        4   4   4   50
        5   5   5   60

        >>> ds.imatrix    # even after changing the dataset, the matrix remains the same.
        FastArray([[-3,  0, 10],
                   [-2,  1, 20],
                   [-1,  2, 30],
                   [ 0,  3, 40],
                   [ 1,  4, 50],
                   [ 2,  5, 60]])

        """
        try:
            return self._imatrix.imatrix
        except:
            return None

    @property
    def imatrix_ds(self):
        """
        Returns the dataset of the 2d array created from `imatrix_make`.

        Examples
        --------
        >>> ds = rt.Dataset({'a': np.arange(-3,3), 'b':np.arange(6), 'c':np.arange(10,70,10)})
        >>> ds
        #    a   b    c
        -   --   -   --
        0   -3   0   10
        1   -2   1   20
        2   -1   2   30
        3    0   3   40
        4    1   4   50
        5    2   5   60
        <BLANKLINE>
        [6 rows x 3 columns] total bytes: 144.0 B

        >>> ds.imatrix_make(colnames = ['a', 'c'])
        FastArray([[-3, 10],
                   [-2, 20],
                   [-1, 30],
                   [ 0, 40],
                   [ 1, 50],
                   [ 2, 60]])

        >>> ds.imatrix_ds
        #    a    c
        -   --   --
        0   -3   10
        1   -2   20
        2   -1   30
        3    0   40
        4    1   50
        5    2   60

        """
        try:
            return self._imatrix.dataset
        except:
            return None

    @property
    def imatrix_cls(self):
        """
        Returns the `IMatrix` class created by `imatrix_make`.

        """
        try:
            return self._imatrix
        except:
            return None

    # -------------------------------------------------------
    def imatrix_make(
        self,
        dtype: Optional[Union[str, np.dtype]] = None,
        order: str = 'F',
        colnames: Optional[List[str]] = None,
        cats: bool = False,
        gb: bool = False,
        inplace: bool = True,
        retnames: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """
        Parameters
        ----------
        dtype : str or np.dtype, optional, default None
            Defaults to None, can force a final dtype such as ``np.float32``.
        order : {'F', 'C'}
            Defaults to 'F', can be 'C' also;
            when 'C' is used, `inplace` cannot be True since the shape will not match.
        colnames : list of str, optional
            Column names to turn into a 2d matrix.
            If None is passed, it will use all computable columns in the Dataset.
        cats : bool, default False
            If set to True will include categoricals.
        gb : bool, default False
            If set to True will include the groupby keys.
        inplace : bool, default True
            If set to True (default) will rearrange and stack the columns in the dataset to be part of the matrix.
            If set to False, the columns in the existing dataset will not be affected.
        retnames : bool, default False
            Defaults to False. If set to True will return the column names it used.

        Returns
        -------
        imatrix : np.ndarray
            A 2D array (matrix) containing the data from this `Dataset` with the specified `order`.
        colnames : list of str, optional
            If `retnames` is True, a list of the column names included in the returned matrix;
            otherwise, this list is not returned.

        Examples
        --------
        >>> arrsize=3
        >>> ds=rt.Dataset({'time': rt.arange(arrsize * 1.0), 'data': rt.arange(arrsize)})
        >>> ds.imatrix_make(dtype=rt.int32)
        FastArray([[0, 0],
                   [1, 1],
                   [2, 2]])
        """
        if order != 'F' and order != 'C':
            raise ValueError(f"Invalid order '{order}' specified. The order must be either 'F' or 'C'.")
        if order != 'F' and inplace:
            raise ValueError("Only the 'F' order may be specified when `inplace` is True.")

        if inplace:
            ds=self
        else:
            ds=self.copy(deep=False)

        if colnames is None:
            #just use the computables?
            colnames=[]
            labels = self.label_get_names()

            for colname, array in ds.items():
                append = False
                if array.iscomputable():
                    append=True
                else:
                    # todo specific check for date/datetime also
                    if isinstance(array, TypeRegister.Categorical):
                        if cats is True:
                            append=True
                    else:
                        # possibly handle
                        pass

                if append:
                    if gb is True or colname not in labels:
                        colnames.append(colname)

        if not isinstance(colnames,list):
            raise TypeError(f"Pass in a list of column names such as imatrix_make(['Exch1','Exch2', 'Exch3'])")

        if len(colnames) < 1:
            raise ValueError(f"The colnames list must contain at least one item")

        ds._imatrix= IMatrix(ds, dtype=dtype, order=order, colnames=colnames)

        #reassign the columns
        ids = ds.imatrix_ds
        for c in colnames:
            ds[c]=ids[c]

        if retnames:
            return ds._imatrix.imatrix, colnames
        else:
            return ds._imatrix.imatrix

    # -------------------------------------------------------
    # 2d arithmetic functions.
    def imatrix_y(self, func: Union[callable, str, List[Union[callable, str]]], name: Optional[Union[str, List[str]]] = None) -> 'Dataset':
        """
        Parameters
        ----------
        func : callable or str or list of callable
            Function or method name of function.
        name : str or list of str, optional

        Returns
        -------
        Dataset
            Y axis calculations for the functions

        Example
        -------
        >>> ds = rt.Dataset({'a1': rt.arange(3)%2, 'b1': rt.arange(3)})
        >>> ds.imatrix_y([np.sum, np.mean])
        #   a1   b1   Sum   Mean
        -   --   --   ---   ----
        0    0    0     0   0.00
        1    1    1     2   1.00
        2    0    2     2   1.00
        """
        try:
            if self.imatrix is None:
                self.imatrix_make()
        except:
            raise ValueError(f'No imatrix or failed to create one.  Use imatrix_make to create one.')

        if not isinstance(func, list):
            func = [func]

        if name is not None:
            if not isinstance(name, list):
                name = [name]
            for f, n in zip(func, name):
                self._imatrix_y_internal(f, name=n)

        else:
            for f in func:
                self._imatrix_y_internal(f)
        return self

    # -------------------------------------------------------
    # 2d arithmetic functions.
    def _imatrix_y_internal(self, func, name: Optional[str] = None, showfilter: bool = True) -> Optional[Tuple[Any, str, callable]]:
        """
        Parameters
        ----------
        func: function or method name of function

        Returns
        -------
        Y axis calculations
        name of the column used
        func used
        """
        imatrix = self.imatrix

        if not callable(func):
            func = getattr(imatrix, func)

        if callable(func):
            if name is None:
                name = func.__name__
                name = str.capitalize(name)

            row_count, col_count = imatrix.shape

            # horizontal func
            #print("im0", imatrix.nansum())
            resultY = func(imatrix, axis=1)

            # possibly remove filtered top row
            if not showfilter:
                resultY = resultY[1:]

            # add the Total column to the dataset
            # BUG? check for existing colname?
            self[name]=resultY

            oldsummary = self.summary_get_names()
            if name not in oldsummary:
                oldsummary.append(name)
                self.summary_set_names(oldsummary)

            return resultY, name, func
        return None

    # -------------------------------------------------------
    # 2d arithmetic functions.
    def imatrix_xy(self, func: Union[callable, str], name: Optional[str] = None, showfilter: bool = True) -> Tuple[Optional['Dataset'], Optional['Dataset'], Optional[str]]:
        """
        Parameters
        ----------
        func : str or callable
            function or method name of function
        name
        showfilter : bool

        Returns
        -------
        X and Y axis calculations
        """
        resultY, name, func = self._imatrix_y_internal(func, name=name, showfilter=showfilter)

        if resultY is not None:
            imatrix = self.imatrix
            row_count, col_count = imatrix.shape

            # reserve an extra for the total of result
            resultX = empty(col_count+1, dtype=resultY.dtype)

            # based on the size...consider #imatrix.nansum(axis=0, out=resultX)
            for i in range(col_count):
                arrslice = imatrix[:,i]

                # possibly skip over first value
                if not showfilter:
                    arrslice =arrslice[1:]

                resultX[i] = func(arrslice)

            # calc total of result - cell on far right and bottom
            resultX[-1] = func(resultY)

            return resultX, resultY, name

        return None, None, None

    # -------------------------------------------------------
    def imatrix_totals(self, colnames=None, name=None):
        if self.imatrix is None:
            self.imatrix_make(colnames=colnames)

        totalsX, totalsY, name = self.imatrix_xy(np.sum, name=name)

        if totalsY is not None:

            # tell display that this dataset has a footer
            footerdict = dict(zip(self.imatrix_ds,totalsX))
            footerdict[name] = totalsX[-1]
            self.footer_set_values(name, footerdict)
            return self

    # -------------------------------------------------------
    def fillna(self, value=None, method: Optional[str] = None, inplace: bool = False, limit: Optional[int] = None) -> Optional['Dataset']:
        """
        Returns a copy with all invalid values set to the given value.
        Optionally modify the original, this might fail if locked.

        Parameters
        ----------
        value
            A replacement value (CANNOT be a dict yet)
        method : {'backfill', 'bfill', 'pad', 'ffill', None}
            * backfill/bfill: calls :meth:`~rt.rt_fastarray.FastArray.fill_backward`
            * pad/ffill: calls :meth:`~rt.rt_fastarray.FastArray.fill_forward`
            * None: calls :meth:`~rt.rt_fastarray.FastArray.replacena`
        inplace : bool, default False
            If True, modify original column arrays.
        limit : int, optional, default None
            Only valid when `method` is not None.
            The maximium number of consecutive invalid values.
            A gap with more than this will be partially filled.

        Returns
        -------
        Dataset, optional

        Examples
        --------
        >>> ds = rt.Dataset({'A': rt.arange(3), 'B': rt.arange(3.0)})
        >>> ds.A[2]=ds.A.inv
        >>> ds.B[1]=np.nan
        >>> ds.fillna(rt.FastArray.fillna, 0)
        #   A      B
        -   -   ----
        0   0   0.00
        1   1   0.00
        2   0   2.00

        >>> ds = rt.Dataset({'A':[np.nan, 2, np.nan, 0], 'B': [3, 4, np.nan, 1],
        ...   'C':[np.nan, np.nan, np.nan, 5], 'D':[np.nan, 3, np.nan, 4]})
        >>> ds.fillna(method='ffill')
        #      A      B      C      D
        -   ----   ----   ----   ----
        0    nan   3.00    nan    nan
        1   2.00   4.00    nan   3.00
        2   2.00   4.00    nan   3.00
        3   0.00   1.00   5.00   4.00
        """

        if method is not None:
            if method in ['backfill','bfill']:
                return self.apply_cols(FastArray.fill_backward, value, inplace=inplace, limit=limit)
            if method in ['pad','ffill']:
                return self.apply_cols(FastArray.fill_forward, value, inplace=inplace, limit=limit)
            raise KeyError(f"fillna: The method {method!r} must be 'backfill', 'bfill', 'pad', 'ffill'")

        if value is None:
            raise ValueError(f"fillna: Must specify either a 'value' that is not None or a 'method' that is not None.")

        if limit is not None:
            raise KeyError(f"fillna: There is no limit when method is None")

        return self.apply_cols(FastArray.replacena, value, inplace=inplace)

    # -------------------------------------------------------
    # Arithmetic functions.
    def apply_cols(
        self, func_or_method_name, *args, fill_value=None, unary: bool = False,
        labels: bool = False, **kwargs
    ) -> Optional['Dataset']:
        """
        Apply function (or named method) on each column.
        If results are all None (*=, +=, for example), None is returned;
        otherwise a Dataset of the return values will be returned (+, *, abs);
        in this case they are expected to be scalars or vectors of same length.

        Constraints on first elem. of args (if unary is False, as for func being an arith op.).
        lhs can be::

        1. a numeric scalar
        2. a list of numeric scalars, length nrows (operating on each column)
        3. an array of numeric scalars, length nrows (operating on each column)
        4. a column vector of numeric scalars, shape (nrows, 1) (reshaped and operating on each column)
        5. a Dataset of numeric scalars, shape (nrows, k) (operating on each matching column by name)
        6. a Struct of (possibly mixed) (1), (2), (3), (4) (operating on each matching column by name)

        Parameters
        ----------
        func_or_method_name: callable or name of method to be called on each column
        args: arguments passed to the func call.
        fill_value
            The fill value to use for columns with non-computable types.

            * None: return original column in result
            * alt_func (callable): force computation with alt_func
            * scalar: apply as uniform fill value
            * dict / defaultdict: Mapping of colname->fill_value.
                Specify per-column `fill_value` behavior.
                Column names can be mapped to one of the other value
                Columns whose names are missing from the mapping (or are mapped to ``None``)
                will be dropped.
                Key-value pairs where the value is ``None``, or an absent column name
                None, or an absent column name if not a ``defaultdict`` still means
              None (or absent if not a defaultdict) still means drop column
              and an alt_func still means force compute via alt_func.
        unary: If False (default) then enforce shape constraints on first positional arg.
        labels: If False (default) then do not apply the function to any label columns.
        kwargs: all other kwargs are passed to func.

        Returns
        -------
        Dataset, optional

        Examples
        --------
        >>> ds = rt.Dataset({'A': rt.arange(3), 'B': rt.arange(3.0)})
        >>> ds.A[2]=ds.A.inv
        >>> ds.B[1]=np.nan
        >>> ds
        #     A      B
        -   ---   ----
        0     0   0.00
        1     1    nan
        2   Inv   2.00

        >>> ds.apply_cols(rt.FastArray.fillna, 0)
        >>> ds
        #   A      B
        -   -   ----
        0   0   0.00
        1   1   0.00
        2   0   2.00

        """
        _is_numeric = lambda _x: isinstance(_x, (int, float, np.integer, np.floating))
        _is_ok_list = lambda _x: isinstance(_x, list) and len(_x) == nrows and all(_is_numeric(_e) for _e in _x)
        _is_ok_array = lambda _x: isinstance(_x, np.ndarray) and _x.shape == (nrows,)
        _is_ok_col_vector = lambda _x: isinstance(_x, np.ndarray) and _x.shape == (nrows, 1)
        _is_for_column = lambda _x: _is_numeric(_x) or _is_ok_list(_x) or _is_ok_array(_x) or _is_ok_col_vector(_x)

        if len(args) == 0 and not unary:
            unary = True

        if not unary:
            lhs = args[0]
            nrows = self.get_nrows()
            if _is_numeric(lhs):
                pass
            elif lhs is None:
                pass
            elif _is_ok_list(lhs):
                pass
            elif _is_ok_array(lhs):
                pass
            elif _is_ok_col_vector(lhs):
                args = (lhs.ravel(),) + args[1:] if len(args) > 1 else (lhs.ravel(),)
            elif isinstance(lhs, Dataset) and all(_is_ok_col_vector(_v) for _k, _v in lhs.items() if _k in self):
                return self._operate_iter_input_cols(args, fill_value, func_or_method_name, kwargs, lhs)
            elif isinstance(lhs, Struct) and all(_is_for_column(_v) for _k, _v in lhs.items() if _k in self):
                return self._operate_iter_input_cols(args, fill_value, func_or_method_name, kwargs, lhs)
            else:
                raise ValueError(
                    f'{self.__class__.__name__}.apply_cols(): lhs must be scalar or flat list/array or column vector of length nrows (for column-wise); a Struct/Dataset of same for (row/element-wise).')

        # Otherwise unary, so just an operation on one array
        def _operate_on_array(array, func_or_method_name, *args, **kwargs):
            if array.iscomputable():
                if callable(func_or_method_name):
                    ret_array = func_or_method_name(array, *args, **kwargs)
                else:
                    # print('v',type(array))
                    # print('func',func_or_method_name)
                    # print('kwargs',kwargs)
                    func = getattr(array, func_or_method_name)
                    ret_array = func(*args, **kwargs)
            elif callable(fval):
                ret_array = fval(array, *args, **kwargs)
            elif fval is not None:
                ret_array = fval
            else:
                ret_array = array
            return ret_array

        od = {}
        for colname, array in self.items():
            # not all arrays are computable, such as *= for a string array

            if colname in self.label_get_names() and not labels:
                od[colname] = array
            else:
                if isinstance(fill_value, dict):
                    # try/catch instead of get() to support defaultdict usage
                    try:
                        fval = fill_value[colname]
                    except KeyError:
                        fval = None
                else:
                    fval = fill_value
                od[colname] = _operate_on_array(array, func_or_method_name, *args, **kwargs)
        if all(_x is None for _x in od.values()):
            return None

        try:
            ret_obj = type(self)(od)
        except Exception:
            raise ValueError(f"the return {od} could not be made into a dataset.")

        # Handle summary columns
        summary_colnames = []
        if self.summary_get_names():
            for i, name in enumerate(self.summary_get_names()):
                summary_colnames += ['Summary' + str(i)]
                ret_obj.col_rename(name, summary_colnames[i])
        # Handle footers
        footers = {}
        if self.footer_get_values():
            try:
                num_labels = len(self.label_get_names()) if self.label_get_names() else 0
                arrays = []
                for self_footervals in self.footer_get_values().values():
                    array = FastArray(self_footervals[num_labels:])
                    arrays += [_operate_on_array(array, func_or_method_name, *args, **kwargs)]
                footers = self._construct_new_footers(arrays, num_labels, summary_colnames)
            except:
                footers = None
        ret_obj = self._add_labels_footers_summaries(ret_obj, summary_colnames, footers)
        return ret_obj

    def _construct_new_footers(self, arrays, num_labels, summary_colnames):
        footers = {}
        try:
            for arr in arrays:
                col_vals = {}
                summary_colnum = 0
                for i_raw, col_name in enumerate(list(self.keys())):
                    i = i_raw - num_labels
                    if i < 0:
                        continue
                    if col_name in self.summary_get_names():
                        col_vals[summary_colnames[summary_colnum]] = arr[i]
                        summary_colnum += 1
                    else:
                        col_vals[col_name] = arr[i]
                footers['Footer' + str(len(footers))] = col_vals
            return footers
        except:
            return None

    def _add_labels_footers_summaries(self, ret_obj, summary_colnames, footers):
        if self.label_get_names():
            ret_obj.label_set_names(self.label_get_names())
        if summary_colnames:
            ret_obj.summary_set_names(summary_colnames)
        if footers:
            for label, footerdict in footers.items():
                ret_obj.footer_set_values(label, footerdict)
        return ret_obj

    def _operate_iter_input_cols(self, args, fill_value, func_or_method_name, kwargs, lhs):
        """
        Operate iteratively across all columns in the dataset and matching ones
        in lhs.

        In order to operate on summary columns and footer rows, such as those
        generated by accum2, require that self and lhs conform in the sense
        of having the same number of labels, footers, and summary columns,
        with all label columns to the left and all summary columns to the
        right. The operation is then performed on positionally corresponding
        elements in the summary columns and footer rows, skipping the label column(s).
        """
        od = {}
        conform = self._labels_footers_summaries_conform(lhs)
        summary_colnames = []
        for colname in self.keys():
            lhs_colname = colname
            od_colname = colname
            if conform and self.summary_get_names() and colname in self.summary_get_names():
                od_colname = 'Summary' + str(len(summary_colnames))
                lhs_colname = lhs.summary_get_names()[len(summary_colnames)]
                summary_colnames += [od_colname]
            if lhs_colname in lhs and colname not in self.label_get_names():
                self1 = Dataset({'a': self[colname]})
                _v = getattr(lhs, lhs_colname)
                args1 = (_v,) + args[1:] if len(args) > 1 else (_v,)
                self1 = self1.apply_cols(func_or_method_name, *args1, fill_value=fill_value, **kwargs)
                od[od_colname] = getattr(self1, 'a')
            else:
                od[od_colname] = getattr(self, colname)
        if all(_x is None for _x in od.values()):
            return None
        # Handle footers
        footers = {}
        if conform and self.footer_get_values():
            num_labels = len(self.label_get_names()) if self.label_get_names() else 0
            arrays = []
            for self_footervals, lhs_footervals in zip(
                self.footer_get_values(fill_value=np.nan).values(),
                    lhs.footer_get_values(fill_value=np.nan).values()):
                self1 = Dataset({'v1': self_footervals[num_labels:]})
                _v = FastArray(lhs_footervals[num_labels:])
                args1 = (_v,) + args[1:] if len(args) > 1 else (_v,)
                self1 = self1.apply_cols(func_or_method_name, *args1, fill_value=fill_value, **kwargs)
                arrays += [self1['v1']]
            footers = self._construct_new_footers(arrays, num_labels, summary_colnames)
        ret_obj = self._add_labels_footers_summaries(type(self)(od), summary_colnames, footers)
        return ret_obj

    def _labels_footers_summaries_conform(self, other):
        def _footers_conform():
            self_footers = self.footer_get_values()
            other_footers = other.footer_get_values()
            if bool(self_footers) != bool(other_footers):
                return False
            if self_footers:
                if len(self_footers) != len(other_footers):
                    return False
                for v1, v2 in zip(self_footers.values(), other_footers.values()):
                    if len(v1) != len(v2):
                        return False
            return True

        def _columns_conform(func, left_or_right='left'):
            def _get_indexes(ds, names):
                return [ds.keys().index(names[i]) for i in range(len(names))]
            self_names = func(self)
            other_names = func(other)
            if bool(self_names) != bool(other_names):
                return False
            if self_names:
                self_indexes = _get_indexes(self, self_names)
                other_indexes = _get_indexes(other, other_names)
                if self_indexes != other_indexes:
                    return False
                if left_or_right == 'left':
                    if self_indexes != list(range(len(self_names))):
                        return False
                if left_or_right == 'right':
                    if self_indexes != list(range(len(self.keys())))[-len(self_names):]:
                        return False
            return True

        if isinstance(other, Dataset) and _footers_conform() and\
            _columns_conform(Dataset.label_get_names, 'left') and\
            _columns_conform(Dataset.summary_get_names, 'right'):
            return True
        else:
            return False

    def __iadd__(self, lhs):
        return self.apply_cols('__iadd__', lhs)

    def __isub__(self, lhs):
        return self.apply_cols('__isub__', lhs)

    def __imul__(self, lhs):
        return self.apply_cols('__imul__', lhs)

    # def __imatmul__(self, lhs): return self.apply_cols('__imatmul__', lhs)
    def __itruediv__(self, lhs):
        return self.apply_cols('__itruediv__', lhs)

    def __ifloordiv__(self, lhs):
        return self.apply_cols('__ifloordiv__', lhs)

    def __imod__(self, lhs):
        return self.apply_cols('__imod__', lhs)

    def __ipow__(self, lhs, modulo=None):
        if modulo is not None:
            return self.apply_cols('__ipow__', lhs, modulo)
        else:
            return self.apply_cols('__ipow__', lhs)

    def __ilshift__(self, lhs):
        return self.apply_cols('__ilshift__', lhs)

    def __irshift__(self, lhs):
        return self.apply_cols('__irshift__', lhs)

    def __iand__(self, lhs):
        return self.apply_cols('__iand__', lhs)

    def __ixor__(self, lhs):
        return self.apply_cols('__ixor__', lhs)

    def __ior__(self, lhs):
        return self.apply_cols('__ior__', lhs)

    # Not all 'reflected' ops are defined (for example 5<<ds), are not reasonable to support;
    # divmod(a, b) returns two values, maybe support one day returning pair of datasets?
    def __radd__(self, lhs):
        return self.apply_cols('__radd__', lhs)

    def __rsub__(self, lhs):
        return self.apply_cols('__rsub__', lhs)

    def __rmul__(self, lhs):
        return self.apply_cols('__rmul__', lhs)

    def __rtruediv__(self, lhs):
        return self.apply_cols('__rtruediv__', lhs)

    def __rfloordiv__(self, lhs):
        return self.apply_cols('__rfloordiv__', lhs)

    def __rmod__(self, lhs):
        return self.apply_cols('__rmod__', lhs)

    def __rpow__(self, lhs):
        return self.apply_cols('__rpow__', lhs)

    def __rand__(self, lhs):
        return self.apply_cols('__rand__', lhs)

    def __rxor__(self, lhs):
        return self.apply_cols('__rxor__', lhs)

    def __ror__(self, lhs):
        return self.apply_cols('__ror__', lhs)

    def __add__(self, lhs):
        return self.apply_cols('__add__', lhs)

    def __sub__(self, lhs):
        return self.apply_cols('__sub__', lhs)

    def __mul__(self, lhs):
        return self.apply_cols('__mul__', lhs)

    # def __matmul__(self, lhs): return self.apply_cols('__matmul__', lhs)
    def __truediv__(self, lhs):
        return self.apply_cols('__truediv__', lhs)

    def __floordiv__(self, lhs):
        return self.apply_cols('__floordiv__', lhs)

    def __mod__(self, lhs):
        return self.apply_cols('__mod__', lhs)

    def __pow__(self, lhs, modulo=None):
        if modulo is not None:
            return self.apply_cols('__pow__', lhs, modulo)
        else:
            return self.apply_cols('__pow__', lhs)

    def __lshift__(self, lhs):
        return self.apply_cols('__lshift__', lhs)

    def __rshift__(self, lhs):
        return self.apply_cols('__rshift__', lhs)

    def __and__(self, lhs):
        return self.apply_cols('__and__', lhs)

    def __xor__(self, lhs):
        return self.apply_cols('__xor__', lhs)

    def __or__(self, lhs):
        return self.apply_cols('__or__', lhs)

    def __neg__(self):
        return self.apply_cols('__neg__', unary=True)

    def __pos__(self):
        return self.apply_cols('__pos__', unary=True)

    def __abs__(self):
        return self.apply_cols('__abs__', unary=True)

    def __invert__(self):
        return self.apply_cols('__invert__', unary=True)

    def abs(self) -> 'Dataset':
        """
        Return a dataset where all elements are replaced, as appropriate, by their absolute value.

        Returns
        -------
        Dataset

        Examples
        --------
        >>> ds = rt.Dataset({'a': np.arange(-3,3), 'b':3*['A', 'B'], 'c':3*[True, False]})
        >>> ds
        #    a   b       c
        -   --   -   -----
        0   -3   A    True
        1   -2   B   False
        2   -1   A    True
        3    0   B   False
        4    1   A    True
        5    2   B   False

        >>> ds.abs()
        #   a   b       c
        -   -   -   -----
        0   3   A    True
        1   2   B   False
        2   1   A    True
        3   0   B   False
        4   1   A    True
        5   2   B   False


        """
        return abs(self)

    @property
    def dtypes(self) -> Mapping[str, np.dtype]:
        """
        Returns dictionary of dtype for each column.

        Returns
        -------
        dict
            Dictionary containing the dtype for each column in the Dataset.
        """
        return {colname: getattr(self, colname).dtype for colname in self.keys()}

    def astype(self, new_type, ignore_non_computable: bool = True):
        """
        Return a new Dataset w/ changed types.

        Will ignore string and categorical columns unless forced.
        Do not do this unless you know they will convert nicely.

        Parameters
        ----------
        new_type : a suitable type object for each row
        ignore_non_computable : bool
            If True then try to convert string and categoricals. Defaults to False.

        Returns
        -------
        Dataset
            A new Dataset w/ changed types.

        Examples
        --------
        >>> ds = rt.Dataset({'a': np.arange(-3,3), 'b':3*['A', 'B'], 'c':3*[True, False]})
        >>> ds
        #    a   b       c
        -   --   -   -----
        0   -3   A    True
        1   -2   B   False
        2   -1   A    True
        3    0   B   False
        4    1   A    True
        5    2   B   False
        <BLANKLINE>
        [6 rows x 3 columns] total bytes: 36.0 B

        >>> ds.astype(int)
        #    a   b   c
        -   --   -   -
        0   -3   A   1
        1   -2   B   0
        2   -1   A   1
        3    0   B   0
        4    1   A   1
        5    2   B   0
        <BLANKLINE>
        [6 rows x 3 columns] total bytes: 54.0 B

        >>> ds.astype(bool)
        #       a   b       c
        -   -----   -   -----
        0    True   A    True
        1    True   B   False
        2    True   A    True
        3   False   B   False
        4    True   A    True
        5    True   B   False
        <BLANKLINE>
        [6 rows x 3 columns] total bytes: 18.0 B
        """
        fval = None if ignore_non_computable else (lambda _v, _t: _v.astype(_t))
        return self.apply_cols('astype', new_type, unary=True, fill_value=fval)

    # -------------------------------------------------------------
    def one_hot_encode(self, columns: Optional[List[str]] = None, exclude: Optional[Union[str, List[str]]] = None) -> None:
        """
        Replaces categorical columns with one-hot-encoded columns for their categories.
        Original columns will be removed from the dataset.

        Default is to encode all categorical columns. Otherwise, certain columns can be specified.
        Also an optional exclude list for convenience.

        Parameters
        ----------
        columns : list of str, optional
            specify columns to encode (if set, exclude param will be ignored)
        exclude : str or list of str, optional
            exclude certain columns from being encoded
        """
        # build column name list
        if columns is None:
            columns = self.keys()
            if exclude is not None:
                if not isinstance(exclude, list):
                    exclude = [exclude]
                columns = [c for c in columns if c not in exclude]

        cat_cols = []
        for c in columns:
            col = getattr(self, c)
            if isinstance(col, TypeRegister.Categorical):
                cat_cols.append(c)
                cat_list, one_hot_cols = col.one_hot_encode()

                for name, one_hot in zip(cat_list, one_hot_cols):
                    setattr(self, c+'__'+name, one_hot)

        self.col_remove(cat_cols)

    def head(self, n: int = 20) -> 'Dataset':
        """
        Return view into beginning of Dataset.

        Parameters
        ----------
        n : int
            Number of rows at the head to return.

        Returns
        -------
        Dataset
            A new dataset which is a view into the original.
        """
        if self._nrows is None: self._nrows = 0
        rows = min(self._nrows, n)
        return self[:rows, :]

    def tail(self, n: int = 20) -> 'Dataset':
        """
        Return view into end of Dataset.

        Parameters
        ----------
        n : int
            Number of rows at the tail to return.

        Returns
        -------
        Dataset
            A new dataset which is a view into the original.
        """
        if self._nrows is None:
            self._nrows = 0
            return self[:0, :]
        rows = min(self._nrows, n)
        return self[-rows:, :]

    def dhead(self, n: int = 0) -> None:
        """
        Displays the head of the Dataset. Compare with :meth:`~rt.rt_dataset.Dataset.head` which returns a new Dataset.
        """
        table = DisplayTable()
        if n == 0:
            # use default if empty
            n = table.options.HEAD_ROWS
        print(self.head(n=n)._V)

    def dtail(self, n: int = 0) -> None:
        """
        Displays the tail of the Dataset. Compare with :meth:`~rt.rt_dataset.Dataset.tail` which returns a new Dataset.
        """
        table = DisplayTable()
        if n == 0:
            # use default if empty
            n = table.options.TAIL_ROWS
        temp = self.tail(n=n)
        print(temp)

    def asrows(self, as_type: Union[str, type] = 'Dataset', dtype: Optional[Union[str, np.dtype]] = None):
        """
        Iterate over rows in any number of of ways, set as_type as appropriate.

        When some columns are strings (unicode or byte) and as_type is 'array',
        best to set dtype=object.

        Parameters
        ----------

        as_type : {'Dataset', 'Struct', 'dict', 'OrderedDict', 'namedtuple', 'tuple', 'list', 'array', 'iter'}
            A string selector which determines return type of iteration, defaults to 'Dataset'.
        dtype : str or np.dtype, optional
            For ``as_type='array'``; if set, force the numpy type of the returned array. Defaults to None.

        Returns
        -------
        iterator over selected type.
        """
        if type(as_type) is type:
            as_type = as_type.__name__

        if as_type == 'Dataset':
            # special case treatment results in large speedup
            for _i in range(self.get_nrows()):
                yield self._copy(rows=[_i])
            return
        elif as_type == 'Struct':
            func = lambda _v, _c=list(self): Struct(dict(zip(_c, _v)))
        elif as_type == 'dict':
            func = lambda _v, _c=list(self): dict(zip(_c, _v))
        elif as_type == 'OrderedDict':
            from collections import OrderedDict
            func = lambda _v, _c=list(self): OrderedDict(zip(_c, _v))
        elif as_type == 'namedtuple':
            DatasetRow = namedtuple('DatasetRow', list(self))
            func = lambda _v, _dr=DatasetRow: _dr(*_v)
        elif as_type == 'tuple':
            func = tuple
        elif as_type == 'list':
            func = list
        elif as_type == 'array':
            func = lambda _v, _dt=dtype: np.array(list(_v), dtype=_dt)
        elif as_type in {'iter', 'iterator'}:
            cols = list(self.values())
            for _i in range(self.get_nrows()):
                yield (_c[_i] for _c in cols)
            return
        else:
            raise ValueError(f'Dataset.asrows(as_type={as_type!r}) not valid.')

        cols = list(self.values())
        for _i in range(self.get_nrows()):
            yield func(_c[_i] for _c in cols)

    def tolist(self):
        """
        Return list of lists of values, by rows.

        Returns
        -------
        list of lists.
        """
        if self.size > 10_000:
            warnings.warn(f"Dataset has {self.size} elements. Performance will suffer when converting values to python lists.")

        # TJD this code is slow and needs review
        return [[self[_i, _c] for _c in self.keys()] for _i in range(self.get_nrows())]

    def to_pandas(self, unicode: bool = True, use_nullable: bool = True) -> 'pd.DataFrame':
        """
        Create a pandas DataFrame from this riptable.Dataset.
        Will attempt to preserve single-key categoricals, otherwise will appear as
        an index array. Any byte strings will be converted to unicode unless unicode=False.

        Parameters
        ----------
        unicode : bool
            Set to False to keep byte strings as byte strings. Defaults to True.
        use_nullable : bool
            Whether to use pandas nullable integer dtype for integer columns (default: True).

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        NotImplementedError
            If a ``CategoryMode`` is not handled for a given column.

        Notes
        -----
        As of Pandas v1.1.0 ``pandas.Categorical`` does not handle riptable ``CategoryMode``s for ``Dictionary``,
        ``MultiKey``, nor ``IntEnum``. Converting a Categorical of these category modes will result in loss of information
        and emit a warning. Although the column values will be respected, the underlying category codes will be remapped
        as a single key categorical.

        See Also
        --------
        riptable.Dataset.from_pandas
        """
        import pandas as pd

        def _to_unicode_if_string(arr):
            if arr.dtype.char == 'S':
                arr = arr.astype('U')
            return arr

        data = self.asdict()
        for key, col in self.items():
            dtype = col.dtype
            if isinstance(col, TypeRegister.Categorical):
                if col.category_mode in (CategoryMode.Default, CategoryMode.StringArray, CategoryMode.NumericArray):
                    pass  # already compatible with pandas; no special handling needed
                elif col.category_mode in (CategoryMode.Dictionary, CategoryMode.MultiKey, CategoryMode.IntEnum):
                    # Pandas does not have a notion of a IntEnum, Dictionary, and Multikey category mode.
                    # Encode dictionary codes to a monotonically increasing sequence and construct
                    # pandas Categorical as if it was a string or numeric array category mode.
                    old_category_mode = col.category_mode
                    col = col.as_singlekey()
                    warnings.warn(f"Dataset.to_pandas: column '{key}' converted from {repr(CategoryMode(old_category_mode))} to {repr(CategoryMode(col.category_mode))}.")
                else:
                    raise NotImplementedError(f'Dataset.to_pandas: Unhandled category mode {repr(CategoryMode(col.category_mode))}')

                base_index = 0 if col.base_index is None else col.base_index
                codes = np.asarray(col) - base_index
                categories = _to_unicode_if_string(col.category_array) if unicode else col.category_array
                data[key]: pd.Categorical = pd.Categorical.from_codes(codes, categories=categories)
            elif isinstance(col, TypeRegister.DateTimeNano):
                ccol = col.copy()
                arr = ccol._timezone.fix_dst(ccol._fa)
                arr = arr.astype('datetime64[ns]')
                tz = _RIPTABLE_TO_PANDAS_TZ[col._timezone._to_tz]
                data[key] = pd.to_datetime(arr).tz_localize(tz)
            elif isinstance(col, TypeRegister.TimeSpan):
                data[key] = pd.to_timedelta(col)
            # TODO: riptable.DateSpan doesn't have a counterpart in pandas, what do we want to do?
            elif use_nullable and np.issubdtype(dtype, np.integer):
                # N.B. Has to use the same dtype for `isin` otherwise riptable will convert the dtype
                #      and the invalid value.
                is_invalid = col.isin(FastArray([INVALID_DICT[dtype.num]], dtype=dtype))
                # N.B. Have to make a copy of the array to numpy array otherwise pandas seg
                #      fault in DataFrame.
                # NOTE: not all versions of pandas have pd.arrays
                if hasattr(pd, 'arrays'):
                    data[key] = pd.arrays.IntegerArray(np.array(col), mask=is_invalid)
                else:
                    data[key] = np.array(col)

            else:
                data[key] = _to_unicode_if_string(col) if unicode else col
        return pd.DataFrame(data)

    def as_pandas_df(self):
        """
        This method is deprecated, please use riptable.Dataset.to_pandas.

        Create a pandas DataFrame from this riptable.Dataset.
        Will attempt to preserve single-key categoricals, otherwise will appear as
        an index array. Any bytestrings will be converted to unicode.

        Returns
        -------
        pandas.DataFrame

        See Also
        --------
        riptable.Dataset.to_pandas
        riptable.Dataset.from_pandas
        """
        warnings.warn('as_pandas_df is deprecated and will be removed in future release, '
                      'please use "to_pandas" method',
                      FutureWarning, stacklevel=2)
        return self.to_pandas()

    @classmethod
    def from_pandas(cls, df: 'pd.DataFrame', tz: str = 'UTC') -> 'Dataset':
        """
        Creates a riptable Dataset from a pandas DataFrame. Pandas categoricals
        and datetime arrays are converted to their riptable counterparts.
        Any timezone-unaware datetime arrays (or those using a timezone not
        recognized by riptable) are localized to the timezone specified by the
        tz parameter.

        Recognized pandas timezones:
            UTC, GMT, US/Eastern, and Europe/Dublin

        Parameters
        ----------
        df: pandas.DataFrame
            The pandas DataFrame to be converted
        tz: string
            A riptable-supported timezone ('UTC', 'NYC', 'DUBLIN', 'GMT') as fallback timezone.

        Returns
        -------
        riptable.Dataset

        See Also
        --------
        riptable.Dataset.to_pandas
        """
        import pandas as pd
        data = {}
        for key in df.columns:
            col = df[key]
            dtype = col.dtype
            dtype_kind = dtype.kind
            iscat = False
            if hasattr(pd, 'CategoricalDtype'):
                iscat = isinstance(dtype, pd.CategoricalDtype)
            else:
                iscat = dtype.num == 100

            if iscat or isinstance(col, pd.Categorical):
                codes = col.cat.codes
                categories = col.cat.categories

                # check for newer version of pandas
                if hasattr(codes, 'to_numpy'):
                    codes = codes.to_numpy()
                    categories = categories.to_numpy()
                else:
                    codes = np.asarray(codes)
                    categories = np.asarray(categories)

                data[key] = TypeRegister.Categorical(codes + 1,  categories=categories)
            elif hasattr(pd, 'Int8Dtype') and \
                    isinstance(dtype, (pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype,
                                       pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype,
                                       pd.UInt64Dtype)):
                data[key] = np.asarray(col.fillna(INVALID_DICT[dtype.numpy_dtype.num]),
                                       dtype=dtype.numpy_dtype)
            elif dtype_kind == 'M':
                try:
                    ptz = str(dtype.tz)
                    try:
                        _tz = _PANDAS_TO_RIPTABLE_TZ[ptz]
                    except KeyError:
                        raise ValueError(
                            "Unable to convert a datetime array with timezone={}".format(ptz))
                except AttributeError:
                    _tz = tz
                data[key] = TypeRegister.DateTimeNano(np.asarray(col, dtype='i8'),
                                                      from_tz='UTC', to_tz=_tz)
            elif dtype_kind == 'm':
                data[key] = TypeRegister.TimeSpan(np.asarray(col, dtype='i8'))
            elif dtype_kind == 'O':
                if len(col) > 0:
                    first_element = col.iloc[0]
                    if isinstance(first_element, (int, float, np.number)):
                        # An object array with number (int or float) in it probably means there is
                        # NaN in it so convert to float64.
                        new_col = np.asarray(col, dtype='f8')
                    else:
                        new_col = np.asarray(col, dtype='S')
                else:
                    new_col = np.asarray(col, dtype='S')
                data[key] = new_col
            else:
                data[key] = df[key]
        return cls(data)

    @staticmethod
    def from_arrow(
        tbl: 'pa.Table', zero_copy_only: bool = True, writable: bool = False, auto_widen: bool = False,
        fill_value: Optional[Mapping[str, Any]] = None
    ) -> 'Dataset':
        """
        Convert a pyarrow `Table` to a riptable `Dataset`.

        Parameters
        ----------
        tbl : pyarrow.Table
        zero_copy_only : bool, default True
            If True, an exception will be raised if the conversion to a `FastArray` would require copying the
            underlying data (e.g. in presence of nulls, or for non-primitive types).
        writable : bool, default False
            For `FastArray`s created with zero copy (view on the Arrow data), the resulting array is not writable (Arrow data is immutable).
            By setting this to True, a copy of the array is made to ensure it is writable.
        auto_widen : bool, optional, default to False
            When False (the default), if an arrow array contains a value which would be considered
            the 'invalid'/NA value for the equivalent dtype in a `FastArray`, raise an exception.
            When True, the converted array
        fill_value : Mapping[str, int or float or str or bytes or bool], optional, defaults to None
            Optional mapping providing non-default fill values to be used. May specify as many or as few columns
            as the caller likes. When None (or for any columns which don't have a fill value specified in the mapping)
            the riptable invalid value for the column (given it's dtype) will be used.

        Returns
        -------
        Dataset

        Notes
        -----
        This function does not currently support pyarrow's nested Tables. A future version of riptable may support
        nested Datasets in the same way (where a Dataset contains a mixture of arrays/columns or nested Datasets having
        the same number of rows), which would make it trivial to support that conversion.
        """
        import pyarrow as pa

        ds_cols = {}
        for col_name, col in zip(tbl.column_names, tbl.columns):
            if isinstance(col, (pa.Array, pa.ChunkedArray)):
                rt_arr = FastArray.from_arrow(col, zero_copy_only=zero_copy_only, writable=writable, auto_widen=auto_widen)

            else:
                # Unknown/unsupported type being used as a column -- can't convert.
                raise RuntimeError(f"Unable to convert column '{col_name}' from object of type '{type(col)}'.")

            ds_cols[col_name] = rt_arr

        return Dataset(ds_cols)

    def to_arrow(self, *, preserve_fixed_bytes: bool = False, empty_strings_to_null: bool = True) -> 'pa.Table':
        """
        Convert a riptable `Dataset` to a pyarrow `Table`.

        Parameters
        ----------
        preserve_fixed_bytes : bool, optional, defaults to False
            For `FastArray` columns which are ASCII string arrays (dtype.kind == 'S'),
            set this parameter to True to produce a fixed-length binary array
            instead of a variable-length string array.
        empty_strings_to_null : bool, optional, defaults To True
            For `FastArray` columns which are  ASCII or Unicode string arrays,
            specify True for this parameter to convert empty strings to nulls in the output.
            riptable inconsistently recognizes the empty string as an 'invalid',
            so this parameter allows the caller to specify which interpretation
            they want.

        Returns
        -------
        pyarrow.Table

        Notes
        -----
        TODO: Maybe add a ``destroy`` bool parameter here to indicate the original arrays should be deleted
              immediately after being converted to a pyarrow array? We'd need to handle the case where the
              pyarrow array object was created in "zero-copy" style and wraps our original array (vs. a new
              array having been allocated via pyarrow); in that case, it won't be safe to delete the original
              array. Or, maybe we just call 'del' anyway to decrement the object's refcount so it can be
              cleaned up sooner (if possible) vs. waiting for this whole method to complete and the GC and
              riptable "Recycler" to run?
        """
        import pyarrow as pa

        # Convert each of the columns to a pyarrow array.
        arrow_col_dict = {}
        for col_name in self.keys():
            orig_col = self[col_name]

            try:
                # Convert the column/array using the FastArray.to_arrow() method (or the inherited overload
                # for derived classes). This allows additional options to be passed when converting, to give
                # callers more flexibility.
                arrow_col = orig_col.to_arrow(
                    preserve_fixed_bytes=preserve_fixed_bytes,
                    empty_strings_to_null=empty_strings_to_null
                )
            except BaseException as exc:
                # Create another exception which wraps the given exception and provides
                # the column name in the error message to make it easier to diagnose issues.
                raise RuntimeError(f"Unable to convert column '{col_name}' to a pyarrow array.") from exc

            arrow_col_dict[col_name] = arrow_col

        # Create the pyarrow.Table from the dictionary of pyarrow arrays.
        return pa.table(arrow_col_dict)

    @staticmethod
    def _axis_key(axis):
        try:
            return {0: 0, 'c': 0, 'C': 0, 'col': 0, 'COL': 0, 'column': 0, 'COLUMN': 0,
                    1: 1, 'r': 1, 'R': 1, 'row': 1, 'ROW': 1,
                    None: None, 'all': None, 'ALL': None}[axis]
        except KeyError:
            raise NotImplementedError(f'Not a valid value for axis: {axis!r}.')

    # -------------------------------------------------------------
    def any(self, axis: Optional[int] = 0, as_dataset: bool = True):
        """
        Returns truth 'any' value along `axis`. Behavior for ``axis=None`` differs from pandas!

        Parameters
        ----------
        axis : int, optional, default axis=0
            * axis=0 (dflt.) -> over columns          (returns Struct (or Dataset) of bools)
                                string synonyms: c, C, col, COL, column, COLUMN
            * axis=1         -> over rows             (returns array of bools)
                                string synonyms: r, R, row, ROW
            * axis=None      -> over rows and columns (returns bool)
                                string synonyms: all, ALL
        as_dataset : bool
            When ``axis=0``, return Dataset instead of Struct. Defaults to False.

        Returns
        -------
        Struct (or Dataset) or list or bool
        """

        def _col_any(_col):
            try:
                return bool(_col.any())
            except TypeError:
                return any(_col)

        axis = self._axis_key(axis)
        cond_rtn_type = type(self) if as_dataset else Struct
        if axis == 0:
            return cond_rtn_type({_cn: _col_any(_val) for _cn, _val in self.items()})
        if axis is None:
            return any(_col_any(_val) for _cn, _val in self.items())
        if axis == 1:
            # for each col,  !=0 to get back bool array.  then inplace OR all those results, careful with string arrays
            temparray=zeros(len(self), dtype=bool)
            for arr in self.values():
                if arr.dtype.num <= 13:
                    # inplace OR for numerical data
                    # for cats we will assume 0 is the invalid and !=0 check works
                    # not sure about nan handling
                    temparray += arr != 0
                else:
                    # care about string array?
                    if arr.dtype.char in 'US':
                        temparray += arr != ''
                    else:
                        # skip this datatype
                        pass
            return temparray
        raise NotImplementedError('Dataset.any(axis=<0, 1, None>)')

    # -------------------------------------------------------------
    def duplicated(self, subset: Optional[Union[str, List[str]]] = None, keep: Union[bool, str] = 'first'):
        """
        Return a boolean FastArray set to True where duplicate rows exist,
        optionally only considering certain columns

        Parameters
        ----------
        subset : str or list of str, optional
            A column label or list of column labels to inspect for duplicate values.
            When ``None``, all columns will be examined.
        keep : {'first', 'last', False}, default 'first'
            * ``first`` : keep duplicates except for the first occurrence.
            * ``last`` : keep duplicates except for the last occurrence.
            * False : set to True for all duplicates.

        Examples
        --------
        >>> ds=rt.Dataset({'somenans': [0., 1., 2., rt.nan, 0., 5.], 's2': [0., 1., rt.nan, rt.nan, 0., 5.]})
        >>> ds
        #   somenans     s2
        -   --------   ----
        0       0.00   0.00
        1       1.00   1.00
        2       2.00    nan
        3        nan    nan
        4       0.00   0.00
        5       5.00   5.00

        >>> ds.duplicated()
        FastArray([False, False, False, False,  True, False])

        Notes
        -----
        Consider using ``rt.Grouping(subset).ifirstkey`` as a fancy index to pull in unique rows.
        """
        if subset is None:
            subset = list(self.keys())
        else:
            if not isinstance(subset, list):
                subset = [subset]

        g = self.gbu(subset).get_groupings()
        igroup = g['iGroup']
        ifirstgroup= g['iFirstGroup']
        ncountgroup = g['nCountGroup']

        result = ones(igroup.shape, dtype=bool)

        # return row of first occurrence
        if keep == 'first':
            # remove invalid bin
            ifirstgroup = ifirstgroup[1:]
            result[igroup[ifirstgroup]]=False

        # return row of last occurrence (however, keys will be in order of their first occurrence)
        elif keep == 'last':
            lastindex = ifirstgroup[-1] + ncountgroup[-1] -1

            # skip invalid and shift everything
            ilast = ifirstgroup[2:]
            ilast -=1
            result[igroup[ilast]]=False

            # set the last one
            result[lastindex]=False

        # only return rows that occur once
        elif keep is False:
            ifirstgroup = ifirstgroup[ncountgroup==1]
            result[igroup[ifirstgroup]]=False

        return result

    # -------------------------------------------------------------
    def drop_duplicates(self, subset=None, keep: Union[bool, str] = 'first', inplace: bool = False) -> 'Dataset':
        """
        Return Dataset with duplicate rows removed, optionally only
        considering certain columns

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns

        keep : {'first', 'last', False}, default 'first'
            - ``first`` : Drop duplicates except for the first occurrence.
            - ``last`` : Drop duplicates except for the last occurrence.
            - False : Drop all duplicates.

        inplace : boolean, default False
            Whether to drop duplicates in place or to return a copy

        Returns
        -------
        deduplicated : Dataset

        Notes
        -----
        If `keep` is 'last', the rows in the result will match pandas, but the order will be based
        on first occurrence of the unique key.

        Examples
        --------
        >>> np.random.seed(12345)
        >>> ds = rt.Dataset({
        ...     'strcol' : np.random.choice(['a','b','c','d'], 15),
        ...     'intcol' : np.random.randint(0, 3, 15),
        ...     'rand' : np.random.rand(15)
        ... })
        >>> ds
         #   strcol   intcol   rand
        --   ------   ------   ----
        0   c             2   0.05
        1   b             1   0.81
        2   b             2   0.93
        3   b             0   0.36
        4   a             2   0.69
        5   b             1   0.13
        6   c             1   0.83
        7   c             2   0.32
        8   b             1   0.74
        9   c             2   0.60
        10   b             2   0.36
        11   b             1   0.79
        12   c             0   0.70
        13   b             1   0.82
        14   d             1   0.90
        <BLANKLINE>
        [15 rows x 3 columns] total bytes: 195.0 B

        Keep only the row of the first occurrence:

        >>> ds.drop_duplicates(['strcol','intcol'])
        #   strcol   intcol   rand
        -   ------   ------   ----
        0   c             2   0.05
        1   b             1   0.81
        2   b             2   0.93
        3   b             0   0.36
        4   a             2   0.69
        5   c             1   0.83
        6   c             0   0.70
        7   d             1   0.90
        <BLANKLINE>
        [8 rows x 3 columns] total bytes: 104.0 B

        Keep only the row of the last occurrence:

        >>> ds.drop_duplicates(['strcol','intcol'], keep='last')
        #   strcol   intcol   rand
        -   ------   ------   ----
        0   c             2   0.60
        1   b             1   0.82
        2   b             2   0.36
        3   b             0   0.36
        4   a             2   0.69
        5   c             1   0.83
        6   c             0   0.70
        7   d             1   0.90
        <BLANKLINE>
        [8 rows x 3 columns] total bytes: 104.0 B

        Keep only the rows which only occur once:

        >>> ds.drop_duplicates(['strcol','intcol'], keep=False)
        #   strcol   intcol   rand
        -   ------   ------   ----
        0   b             0   0.36
        1   a             2   0.69
        2   c             1   0.83
        3   c             0   0.70
        4   d             1   0.90
        <BLANKLINE>
        [5 rows x 3 columns] total bytes: 65.0 B
        """
        if self.shape[0] == 0:
            if inplace:
                return self
            else:
                return TypeRegister.Dataset(self)

        if subset is None:
            subset = list(self.keys())
        else:
            if not isinstance(subset, list):
                subset = [subset]

        gb = self.gbu(subset)

        # return row of first occurrence
        if keep == 'first':
            deduplicated = gb.first()
            deduplicated.label_remove()

        # return row of last occurrence (however, keys will be in order of their first occurrence)
        elif keep == 'last':
            deduplicated = gb.last()
            deduplicated.label_remove()

        # only return rows that occur once
        elif keep is False:
            non_duplicated = gb.count().Count == 1
            deduplicated = gb.first()
            deduplicated.label_remove()
            deduplicated = deduplicated[non_duplicated,:]

        else:
            raise ValueError(f"Got unexpected value for keep {keep}.")

        # replace all columns in dictionary
        if inplace is True:
            if deduplicated._nrows != self._nrows:
                # swap out all column data
                self._nrows = deduplicated._nrows
                self._col_sortlist = None
                self.col_replace_all(deduplicated, check_exists=False)
                return self

        return deduplicated

    # -------------------------------------------------------------
    def col_replace_all(self, newdict, check_exists: bool = True) -> None:
        """
        Replace the data for each item in the item dict. Original attributes
        will be retained. Useful for internal routines that need to swap out all columns quickly.

        Parameters
        ----------
        newdict : dictionary of item names -> new item data (can also be a Dataset)

        check_exists : bool
            if True, all newdict keys and old item keys will be compared to ensure a match
        """
        self._all_items.item_replace_all(newdict, check_exists=check_exists)

    # -------------------------------------------------------------
    def all(self, axis=0, as_dataset: bool = True):
        """
        Returns truth value 'all' along axis.  Behavior for ``axis=None`` differs from pandas!

        Parameters
        ----------

        axis : int, optional
            * axis=0 (dflt.) -> over columns          (returns Struct (or Dataset) of bools)
                                string synonyms: c, C, col, COL, column, COLUMN
            * axis=1         -> over rows             (returns array of bools)
                                string synonyms: r, R, row, ROW
            * axis=None      -> over rows and columns (returns bool)
                                string synonyms: all, ALL
        as_dataset : bool
            When ``axis=0``, return Dataset instead of Struct. Defaults to False.

        Returns
        -------
        Struct (or Dataset) or list or bool
        """

        def _col_all(_col):
            try:
                return bool(_col.all())
            except TypeError:
                return all(_col)

        axis = self._axis_key(axis)
        cond_rtn_type = type(self) if as_dataset else Struct
        if axis == 0:
            return cond_rtn_type({_cn: _col_all(_val) for _cn, _val in self.items()})
        if axis is None:
            return all(_col_all(_val) for _cn, _val in self.items())
        if axis == 1:
            # for each col,  !=0 to get back bool array.  then inplace AND all those results, careful with string arrays
            temparray=ones(len(self), dtype=bool)
            for arr in self.values():
                if arr.dtype.num <= 13:
                    # inplace AND for numerical data
                    # for cats we will assume 0 is the invalid and !=0 check works
                    temparray *= arr != 0
                else:
                    # care about string array?
                    if arr.dtype.char in 'US':
                        temparray *= arr != ''
                    else:
                        # skip this datatype
                        pass

            return temparray
        raise NotImplementedError('Dataset.all(axis=<0, 1, None>)')

    def sorts_on(self) -> None:
        """
        Turns on all row/column sorts for display.  False by default.
        sorts_view must have been called before

        :return: None
        """
        if self._col_sortlist is None:
            warnings.warn(f"sort_view was not called first.  Display sorting will remain off.")
            return

        self._sort_display = True

    def sorts_off(self) -> None:
        """
        Turns off all row/column sorts for display (happens when sort_view is called)
        If sort is cached, it will remain in cache in case sorts are toggled back on.

        :return: None
        """
        self._col_sortlist = None
        self._sort_display = False

    # -------------------------------------------------------
    def get_row_sort_info(self):
        sortdict = None
        # general row sort will take precedence
        if self._col_sortlist is not None:
            for col in self._col_sortlist:
                if col not in self:
                    print(str(col), "is not a valid key to sort by.")
                    # clear invalid sort from dataset
                    self._col_sortlist = None
                    break
            else:
                #sortdict = {col: self.__getattribute__(col) for col in self._col_sortlist}
                sortdict = {col: self.col_get_value(col) for col in self._col_sortlist}

        return self._uniqueid, self._nrows, sortdict

    # -------------------------------------------------------
    def _sort_lexsort(self, by, ascending=True):
        bylist = by

        if not isinstance(by, list):
            bylist = [bylist]

        sortkeys = []
        for col in bylist:
            sortkeys.append(self.col_get_value(col))

        # larger sort
        sort_rows = lexsort([sortkeys[i] for i in range(len(sortkeys) - 1, -1, -1)])

        # need to truly reverse it inplace
        if ascending is False:
            sort_rows = sort_rows[::-1].copy()

        #print("**lexsort", sort_rows)
        return sort_rows

    # -------------------------------------------------------
    def _sort_values(self, by, axis=0, ascending=True, inplace=False, kind='mergesort',
                    na_position='last', copy=False, sort_rows=None):
        """
        Accepts a single column name or list of column names and adds them to the dataset's column sort list.

        The actual sort is performed during display; the dataset itself is not affected
        unless ``inplace=True``.
        When the dataset is being fed into display, the sort cache gets checked to see if a sorted
        index index is being held for the keys with the dataset's matching unique ID. If a sorted
        index is found, it gets passed to display. If no index is found, a lexsort is performed,
        and the sort is stored in the cache.

        Parameters
        ----------
        by : string or list of strings
            The column name or list of column names by which to sort
        axis : int
            not used
        ascending : bool
            not used
        inplace : bool
            Sort the dataset itself.
        kind : str
            not used
        na_position : str
            not used
        sortrows : fancy index array
            used to pass in your own sort

        Returns
        -------
        Dataset
        """
        # TODO: build a better routine to check both regular columns and groupby keys for requested sort
        # this has too many repeat conditionals
        # test sort keys
        bylist = by

        if not isinstance(by, list):
            bylist = [bylist]

        for col in bylist:
            if col not in self:
                raise ValueError(f'{col} is not a valid key to sort by.')

        if inplace or copy:
            if self._sort_display is True and copy is False:
                # turn it off because user just specified a new sort
                self.sorts_off()
                #raise ValueError("sorts are turned off for display. Use ds.sort_display() to reactivate.")

            # larger sort
            self._natural_sort = tuple(bylist)
            if sort_rows is None:
                sort_rows = self._sort_lexsort(bylist, ascending)

            if inplace:
                #for k, v in npdict.items():
                #    #self.__setattr__(k, reindex_fast(sort_rows, v))
                #    self._superadditem(k, reindex_fast(sort_rows, v))
                values = list(self.values())
                keys = list(self.keys())

                # TJD optimization
                # Get all the same dtypes so that we can use on column as a temporary and write it into
                for i,k in enumerate(keys):
                    self[k] = values[i][sort_rows]
                    # allow recycler to kick in
                    values[i]=None
                return self

            elif copy:
                npdict = self._as_dictionary()
                newdict = {}
                for k, v in npdict.items():
                    newdict[k] = v[sort_rows]
                # TODO: add routine to copy other ds properties/attributes (regular copy only does the dict and sortlist)
                # making a copy of the dataset first and then doing a sort is twice as expensive

                newds = type(self)(newdict)
                newds.label_set_names(self.label_get_names())

                if hasattr( self, '_footers' ):
                    footers = {}
                    for f, item in self._footers.items():
                        footers[f] = item.copy()
                    newds._footers = footers

                return newds

        # if drops into here, sort_view was called
        if ascending is False:
            self._sort_ascending = False
        self._col_sortlist = bylist
        self.sorts_on()

        # TJD New code.. once display, turn sorts_off
        return self

    # -------------------------------------------------------
    def sort_view(self, by, ascending=True, kind='mergesort', na_position='last'):
        """
        Sorts all columns by the labels only when displayed. This routine is fast and does not change data underneath.

        Parameters
        ----------
        by : string or list of strings
            The column name or list of column names by which to sort
        ascending : bool
            Determines if the order of sorting is ascending or not.

        Examples
        ----------
        >>> ds = rt.Dataset({'a': np.arange(10), 'b':5*['A', 'B'], 'c':3*[10,20,30]+[10]})
        >>> ds
        #   a   b    c
        -   -   -   --
        0   0   A   10
        1   1   B   20
        2   2   A   30
        3   3   B   10
        4   4   A   20
        5   5   B   30
        6   6   A   10
        7   7   B   20
        8   8   A   30
        9   9   B   10


        >>> ds.sort_view(['b','c'])
        #   a   b    c
        -   -   -   --
        0   0   A   10
        1   6   A   10
        2   4   A   20
        3   2   A   30
        4   8   A   30
        5   3   B   10
        6   9   B   10
        7   1   B   20
        8   7   B   20
        9   5   B   30


        >>> ds.sort_view('a', ascending = False)
        #   a   b    c
        -   -   -   --
        0   9   B   10
        1   8   A   30
        2   7   B   20
        3   6   A   10
        4   5   B   30
        5   4   A   20
        6   3   B   10
        7   2   A   30
        8   1   B   20
        9   0   A   10


        """
        self._sort_values(by, ascending=ascending, inplace=False, kind=kind, na_position=na_position, copy=False)
        return self

    # -------------------------------------------------------
    def sort_inplace(self, by: Union[str, List[str]], ascending: bool = True, kind: str = 'mergesort', na_position: str = 'last') -> 'Dataset':
        """
        Sorts all columns by the labels inplace. This routine will modify the order of all columns.

        Parameters
        ----------
        by : str or list of str
            The column name or list of column names by which to sort
        ascending : bool
            Determines if the order of sorting is ascending or not.

        Returns
        -------
        Dataset
            The reference to the input Dataset is returned to allow for method chaining.

        Examples
        ----------
        >>> ds = rt.Dataset({'a': np.arange(10), 'b':5*['A', 'B'], 'c':3*[10,20,30]+[10]})
        >>> ds
        #   a   b    c
        -   -   -   --
        0   0   A   10
        1   1   B   20
        2   2   A   30
        3   3   B   10
        4   4   A   20
        5   5   B   30
        6   6   A   10
        7   7   B   20
        8   8   A   30
        9   9   B   10


        >>> ds.sort_inplace(['b','c'])
        #   a   b    c
        -   -   -   --
        0   0   A   10
        1   6   A   10
        2   4   A   20
        3   2   A   30
        4   8   A   30
        5   3   B   10
        6   9   B   10
        7   1   B   20
        8   7   B   20
        9   5   B   30


        >>> ds.sort_inplace('a', ascending = False)
        #   a   b    c
        -   -   -   --
        0   9   B   10
        1   8   A   30
        2   7   B   20
        3   6   A   10
        4   5   B   30
        5   4   A   20
        6   3   B   10
        7   2   A   30
        8   1   B   20
        9   0   A   10

        """
        return self._sort_values(by, ascending=ascending, inplace=True, kind=kind, na_position=na_position, copy=False)


    def sort_copy(self, by: Union[str, List[str]], ascending: bool = True, kind: str = 'mergesort', na_position: str ='last') -> 'Dataset':
        """
        Sorts all columns by the labels and returns a copy.  The original dataset is not modified.

        Parameters
        ----------
        by : str or list of str
            The column name or list of column names by which to sort
        ascending : bool
            Determines if the order of sorting is ascending or not.

        Returns
        -------
        Dataset

        Examples
        ----------
        >>> ds = rt.Dataset({'a': np.arange(10), 'b':5*['A', 'B'], 'c':3*[10,20,30]+[10]})
        >>> ds
        #   a   b    c
        -   -   -   --
        0   0   A   10
        1   1   B   20
        2   2   A   30
        3   3   B   10
        4   4   A   20
        5   5   B   30
        6   6   A   10
        7   7   B   20
        8   8   A   30
        9   9   B   10


        >>> ds.sort_copy(['b','c'])
        #   a   b    c
        -   -   -   --
        0   0   A   10
        1   6   A   10
        2   4   A   20
        3   2   A   30
        4   8   A   30
        5   3   B   10
        6   9   B   10
        7   1   B   20
        8   7   B   20
        9   5   B   30


        >>> ds.sort_copy('a', ascending = False)
        #   a   b    c
        -   -   -   --
        0   9   B   10
        1   8   A   30
        2   7   B   20
        3   6   A   10
        4   5   B   30
        5   4   A   20
        6   3   B   10
        7   2   A   30
        8   1   B   20
        9   0   A   10


        """
        return self._sort_values(by, ascending=ascending, inplace=False, kind=kind, na_position=na_position, copy=True)

    # -------------------------------------------------------
    def _apply_outlier(self, func, name, col_keep):
        pos=func()
        row_func=[]
        row_namefunc=[]
        row_pos=[]
        colnames =self.keys()

        # for all the columns
        for c in colnames:
            # categoricals and strings might be eliminated
            if c != col_keep:
                try:
                    #get first value
                    val=pos[c][0]
                    row_pos.append(val)

                    row_func.append(self[c][val])
                    row_namefunc.append(self[col_keep][val])
                except:
                    invalid=INVALID_DICT[self[c].dtype.num]
                    #print("**invalid", invalid)
                    row_func.append(np.nan)
                    row_namefunc.append(get_default_value(self[col_keep]))
                    row_pos.append(-1)

        ds=type(self)({})
        ds[name] = FastArray(row_func)
        ds[col_keep] = FastArray(row_namefunc)
        ds['Pos'] = FastArray(row_pos)

        return ds

    def outliers(self, col_keep) -> 'Multiset':
        """return a dataset with the min/max outliers for each column"""

        maxds=self._apply_outlier(self.nanargmax, 'Values', col_keep)
        minds=self._apply_outlier(self.nanargmin, 'Values', col_keep)

        rownames=[]
        colnames =self.keys()

        # for all the columns
        for c in colnames:
            # categoricals and strings might be eliminated
            if c != col_keep:
                rownames.append(c)

        maxds['Names'] = FastArray(rownames)  # needs auto_rewrap
        maxds.label_set_names(['Names'])

        minds['Names'] = FastArray(rownames)  # needs auto_rewrap
        minds.label_set_names(['Names'])

        ms=TypeRegister.Multiset({})
        ms['Min']=minds
        ms['Max']=maxds
        ms._gbkeys = {'Names' :FastArray(rownames)}

        return ms

    # -------------------------------------------------------
    def computable(self) -> Mapping[str, FastArray]:
        """returns a dict of computable columns.  does not include groupby keys"""
        return_dict = {}
        labels = self.label_get_names()
        for name, arr in self.items():
            # any current groupby keys we will not count either
            if arr.iscomputable() and name not in labels:
                return_dict[name]=arr
        return return_dict

    # -------------------------------------------------------
    def noncomputable(self) -> Mapping[str, FastArray]:
        """returns a dict of noncomputable columns.  includes groupby keys"""
        return_dict = {}
        labels = self.label_get_names()
        for name, arr in self.items():
            if not arr.iscomputable() or name in labels:
                return_dict[name]=arr
        return return_dict

    # -------------------------------------------------------
    @property
    def crc(self) -> 'Dataset':
        """
        Returns a new dataset with the 64 bit CRC value of every column.

        Useful for comparing the binary equality of columns in two datasets

        Examples
        --------
        >>> ds1 = rt.Dataset({'test': rt.arange(100), 'test2': rt.arange(100.0)})
        >>> ds2 = rt.Dataset({'test': rt.arange(100), 'test2': rt.arange(100)})
        >>> ds1.crc == ds2.crc
        #   test   test2
        -   ----   -----
        0   True   False
        """
        newds={}
        for colname,arr in self.items():
            newds[colname]=arr.crc
        return type(self)(newds)

    # -------------------------------------------------------
    def _mask_reduce(self, func, is_ormask: bool):
        """helper function for boolean masks: see mask_or_isnan, et al"""
        mask = None
        funcmask=TypeRegister.MathLedger._BASICMATH_TWO_INPUTS

        if is_ormask:
            funcNum=MATH_OPERATION.BITWISE_OR
        else:
            funcNum=MATH_OPERATION.BITWISE_AND

        # loop through all computable columns
        cols = self.computable()

        for col in cols.values():
            bool_mask = func(col)
            if mask is None:
                mask=bool_mask
            else:
                #inplace is faster
                funcmask((mask, bool_mask, mask), funcNum, 0)
        return mask

    def mask_or_isnan(self) -> FastArray:
        """
        Returns a boolean mask of all columns ORed with :meth:`~rt.rt_numpy.isnan`.

        Useful to see if any elements in the dataset contain a NaN.

        Returns
        -------
        FastArray

        Examples
        --------
        >>> ds = rt.Dataset({'a' : [1,2,np.nan], 'b':[0, np.nan, np.nan]})
        >>> ds
        #      a     b
        -   ----   ---
        0   1.00   0.00
        1   2.00    inf
        2    inf    inf

        [3 rows x 2 columns] total bytes: 48.0 B

        >>> ds.mask_or_isnan()
        FastArray([False, True,  True])
        """
        return self._mask_reduce(np.isnan, True)

    def mask_and_isnan(self) -> FastArray:
        """
        Returns a boolean mask of all columns ANDed with :meth:`~rt.rt_numpy.isnan`.

        Returns
        -------
        FastArray

        Examples
        --------
        >>> ds = rt.Dataset({'a' : [1,2,np.nan], 'b':[0, np.nan, np.nan]})
        >>> ds
        #      a     b
        -   ----   ---
        0   1.00   0.00
        1   2.00    inf
        2    inf    inf

        [3 rows x 2 columns] total bytes: 48.0 B

        >>> ds.mask_and_isnan()
        FastArray([False, False,  True])
        """
        return self._mask_reduce(np.isnan, False)

    def mask_or_isfinite(self) -> FastArray:
        """
        Returns a boolean mask of all columns ORed with :meth:`~rt.rt_numpy.isfinite`.

        Returns
        -------
        FastArray

        Examples
        --------
        >>> ds = rt.Dataset({'a' : [1,2,np.inf], 'b':[0, np.inf, np.inf]})
        >>> ds
        #      a     b
        -   ----   ---
        0   1.00   0.00
        1   2.00    inf
        2    inf    inf

        [3 rows x 2 columns] total bytes: 48.0 B

        >>> ds.mask_or_isfinite()
        FastArray([True, True,  False])
        """
        return self._mask_reduce(np.isfinite, True)

    def mask_and_isfinite(self) -> FastArray:
        """
        Returns a boolean mask of all columns ANDed with :meth:`~rt.rt_numpy.isfinite`.

        Returns
        -------
        FastArray

        Examples
        --------
        >>> ds = rt.Dataset({'a' : [1,2,np.inf], 'b':[0, np.inf, np.inf]})
        >>> ds
        #      a     b
        -   ----   ---
        0   1.00   0.00
        1   2.00    inf
        2    inf    inf

        [3 rows x 2 columns] total bytes: 48.0 B

        >>> ds.mask_and_isfinite()
        FastArray([True, False,  False])
        """
        return self._mask_reduce(np.isfinite, False)

    def mask_or_isinf(self) -> FastArray:
        """
        returns a boolean mask of all columns ORed with isinf

        Returns
        -------
        FastArray

        Examples
        --------
        >>> ds = rt.Dataset({'a' : [1,2,np.inf], 'b':[0, np.inf, np.inf]})
        >>> ds
        #      a     b
        -   ----   ---
        0   1.00   0.00
        1   2.00    inf
        2    inf    inf

        [3 rows x 2 columns] total bytes: 48.0 B

        >>> ds.mask_or_isinf()
        FastArray([False, True,  True])
        """
        return self._mask_reduce(np.isinf, True)

    def mask_and_isinf(self) -> FastArray:
        """
        returns a boolean mask of all columns ANDed with isinf

        Returns
        -------
        FastArray

        Examples
        --------
        >>> ds = rt.Dataset({'a' : [1,2,np.inf], 'b':[0, np.inf, np.inf]})
        >>> ds
        #      a     b
        -   ----   ---
        0   1.00   0.00
        1   2.00    inf
        2    inf    inf

        [3 rows x 2 columns] total bytes: 48.0 B

        >>> ds.mask_and_isinf()
        FastArray([False, False,  True])
        """
        return self._mask_reduce(np.isinf, False)

    def merge(
        self,
        right: 'Dataset',
        on: Optional[Union[str, List[str]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        how: str = 'left',
        suffixes: Tuple[str, str] = ('_x', '_y'),
        indicator: Union[bool, str] = False,
        columns_left: Optional[Union[str, List[str]]] = None,
        columns_right: Optional[Union[str, List[str]]] = None,
        verbose: bool = False,
        hint_size: int = 0
    ) -> 'Dataset':
        return rt_merge.merge(self, right, on=on, left_on=left_on, right_on=right_on, how=how,
                              suffixes=suffixes, indicator=indicator, columns_left=columns_left,
                              columns_right=columns_right, verbose=verbose, hint_size=hint_size)
    merge.__doc__ = rt_merge.merge.__doc__

    def merge2(
        self,
        right: 'Dataset',
        on: Optional[Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        how: str = 'left',
        suffixes: Optional[Tuple[str, str]] = None,
        copy: bool = True,
        indicator: Union[bool, str] = False,
        columns_left: Optional[Union[str, List[str]]] = None,
        columns_right: Optional[Union[str, List[str]]] = None,
        validate: Optional[str] = None,
        keep: Optional[Union[str, Tuple[Optional[str], Optional[str]]]] = None,
        high_card: Optional[Union[bool, Tuple[Optional[bool], Optional[bool]]]] = None,
        hint_size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None
    ) -> 'Dataset':
        return rt_merge.merge2(
            self, right, on=on, left_on=left_on, right_on=right_on, how=how,
            suffixes=suffixes, copy=copy, indicator=indicator, columns_left=columns_left, columns_right=columns_right,
            validate=validate, keep=keep, high_card=high_card, hint_size=hint_size)
    merge2.__doc__ = rt_merge.merge2.__doc__

    def merge_asof(
        self,
        right: 'Dataset',
        on: Optional[Union[str, Tuple[str, str]]] = None,
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        by: Optional[Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]]] = None,
        left_by: Optional[Union[str, List[str]]] = None,
        right_by: Optional[Union[str, List[str]]] = None,
        suffixes: Optional[Tuple[str, str]] = None,
        copy: bool = True,
        columns_left: Optional[Union[str, List[str]]] = None,
        columns_right: Optional[Union[str, List[str]]] = None,
        tolerance: Optional[Union[int, 'timedelta']] = None,
        allow_exact_matches: bool = True,
        direction: str = "backward",
        check_sorted: bool = True,
        matched_on: Union[bool, str] = False,
        **kwargs
    ) -> 'Dataset':
        # TODO: Adapt the logic from merge_lookup() to allow this method to support an in-place merge mode.
        return rt_merge.merge_asof(
            self, right,
            on=on, left_on=left_on, right_on=right_on,
            by=by, left_by=left_by, right_by=right_by,
            suffixes=suffixes, copy=copy, columns_left=columns_left, columns_right=columns_right,
            tolerance=tolerance,
            allow_exact_matches=allow_exact_matches,
            direction=direction, check_sorted=check_sorted,
            matched_on=matched_on,
            **kwargs
        )

    merge_asof.__doc__ = rt_merge.merge_asof.__doc__

    def merge_lookup(
        self,
        right: 'Dataset',
        on: Optional[Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        require_match: bool = False,
        suffix: Optional[str] = None,
        copy: bool = True,
        columns_left: Optional[Union[str, List[str]]] = None,
        columns_right: Optional[Union[str, List[str]]] = None,
        keep: Optional[str] = None,
        inplace: bool = False,
        high_card: Optional[Union[bool, Tuple[Optional[bool], Optional[bool]]]] = None,
        hint_size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None
    ) -> 'Dataset':
        # This method supports an in-place mode; unless the user specifies that one,
        # call the normal module-based implementation.
        suffixes = ('', suffix)
        if not inplace:
            return rt_merge.merge_lookup(
                self, right, on=on, left_on=left_on, right_on=right_on, require_match=require_match, suffixes=suffixes,
                copy=copy, columns_left=columns_left, columns_right=columns_right, keep=keep,
                high_card=high_card, hint_size=hint_size)

        # Specifying 'columns_left' is meaningless for an in-place merge, so don't allow it.
        # If the caller wants to also drop columns from this Dataset, they should do that separately.
        if columns_left:
            raise ValueError("'columns_left' cannot be specified when performing an in-place merge_lookup.")

        # The caller selected the in-place merge; columns from the other Dataset are merged and added into this Dataset.
        # Do this by calling the module version of merge_lookup but don't select any columns from the
        # left Dataset (this instance). Add the resulting columns -- all taken from the right side --
        # to this instance.
        lookup_result = rt_merge.merge_lookup(
            self, right, on=on, left_on=left_on, right_on=right_on, require_match=require_match, suffixes=suffixes,
            copy=copy, columns_left=[], columns_right=columns_right, keep=keep,
            high_card=high_card, hint_size=hint_size)

        # Before adding the lookup result columns to this Dataset,
        # we need to perform the column name conflict resolution step that's
        # normally done while performing the merge. That won't have happened in
        # in our call above since we only selected columns from the 'right' Dataset.
        # NOTE: This must be done prior to adding the resulting columns to this Dataset,
        # so that if there are any unresolvable naming conflicts (in which case we raise
        # an exception), this Dataset won't have been changed at all.
        left_on = rt_merge._extract_on_columns(on, left_on, True, 'on', is_optional=False)
        right_on = rt_merge._extract_on_columns(on, right_on, False, 'on', is_optional=False)
        columns_left = rt_merge._normalize_selected_columns(self, None)
        columns_right = rt_merge._normalize_selected_columns(right, columns_right)
        _, right_colname_mapping, _ = rt_merge._construct_colname_mapping(
            left_on, right_on, suffixes=suffixes, columns_left=columns_left, columns_right=columns_right)
        right_colname_map = dict(zip(*right_colname_mapping))

        # Add the resulting columns to this Dataset.
        for right_col_name in lookup_result.keys():
            # The columns in the merge result won't have gone through the name-conflict resolution
            # process during the merge (since we passed an empty list for the left columns), so we
            # need to apply any name-mappings here when adding the result columns to this instance.
            new_col_name = right_colname_map.get(right_col_name, right_col_name)
            self[new_col_name] = lookup_result[right_col_name]

        return self
    merge_lookup.__doc__ = rt_merge.merge_lookup.__doc__

    @property
    def total_size(self) -> int:
        """
        Returns total size of all (columnar) data in bytes.

        Returns
        -------
        int
            The total size, in bytes, of all columnar data in this instance.
        """
        npdict = self._as_dictionary()
        totalSize = 0
        for k, v in npdict.items():
            try:
                totalSize += v._total_size
            except:
                totalSize += v.size * v.itemsize
        return totalSize

    def _last_row_stats(self):
        return f"[{self._nrows} rows x {self._ncols} columns] total bytes: {self._sizeof_fmt(self.total_size)}"

    @property
    def memory_stats(self) -> None:
        print(self._last_row_stats())

    # ------------------------------------------------------
    def get_sorted_col_data(self, col_name):
        """
        Private method.
        :param col_name:
        :return: numpy array
        """
        if col_name in self:
            #col = self.__getattribute__(col_name)
            col = self.col_get_value(col_name)
            sort_id = self.get_row_sort_info()
            sorted_row_idx = SortCache.get_sorted_row_index(*sort_id)
            if sorted_row_idx is not None:
                return col[sorted_row_idx]
            else:
                return np.copy(col)
        else:
            print(str(col_name), "not found in dataset.")

    # -------------------------------------------------------
    @property
    def _sort_columns(self):
        if self._col_sortlist is not None:
            return self._sort_column_styles

    # -------------------------------------------------------
    def _footers_exist(self, labels):
        """Return a list of occurring footers from user-specified labels.
        If labels is None, return list of all footer labels.
        If none occur, returns None.

        See Also
        --------
        footer_remove(), footer_get_values()
        """
        if labels is None:
            # remove all labels
            final_labels = list(self.footers)
        else:
            # remove specific labels
            if not isinstance(labels, list):
                labels = [labels]
            final_labels = [fname  for fname in labels if fname in self.footers]
            if len(final_labels)==0:
                warnings.warn(f"No footers found for names {labels}.")
                return
        return final_labels

    # -------------------------------------------------------
    def footer_remove(self, labels=None, columns=None):
        """Remove all or specific footers from all or specific columns.

        Parameters
        ----------
        labels : string or list of strings, default None
            If provided, remove only footers under these names.
        columns : string or list of strings, default None
            If provided, only remove (possibly specified) footers from these columns.

        Examples
        --------
        >>> ds = rt.Dataset({'colA': rt.arange(3),'colB': rt.arange(3)*2})
        >>> ds.footer_set_values('sum', {'colA':3, 'colB':6}
        >>> ds.footer_set_values('mean', {'colA':1.0, 'colB':2.0})
        >>> ds
           #   colA   colB
        ----   ----   ----
           0      0      0
           1      1      2
           2      2      4
        ----   ----   ----
         sum      3      6
        mean   1.00   2.00

        Remove single footer from single column

        >>> ds.footer_remove('sum','colA')
        >>> ds
           #   colA   colB
        ----   ----   ----
           0      0      0
           1      1      2
           2      2      4
        ----   ----   ----
         sum             6
        mean   1.00   2.00

        Remove single footer from all columns

        >>> ds.footer_remove('mean')
        >>> ds
          #   colA   colB
        ---   ----   ----
          0      0      0
          1      1      2
          2      2      4
        ---   ----   ----
        sum             6

        Remove all footers from all columns

        >>> ds.footer_remove()
        >>> ds
        #   colA   colB
        -   ----   ----
        0      0      0
        1      1      2
        2      2      4

        Notes
        -----
        Calling this method with no keywords will clear all footers from all columns.

        See Also
        --------
        Dataset.footer_set_values()
        """
        if self.footers is None:
            return
        # get list of existing, or use all footer labels if not specified
        labels = self._footers_exist(labels)
        if labels is None:
            return

        remove_all = False

        # remove from all columns
        if columns is None:
            remove_all = True
            columns = self.keys()
        else:
            # remove from specific columns
            if not isinstance(columns, list):
                columns = [columns]
            # prevent partial footers from being removed
            self._ensure_atomic(columns, self.footer_remove)

        # pop value from each column's footer dict
        for colname in columns:
            coldict = self.col_get_attribute(colname, 'Footer')
            if coldict is None:
                continue
            for label in labels:
                coldict.pop(label,None)

        # if removed from all columns, remove name from master footer row
        if remove_all:
            for label in labels:
                del self.footers[label]

            # None left, remove for future display
            if len(self.footers)==0:
                del self.__dict__['_footers']

    # -------------------------------------------------------
    def footer_get_values(self, labels=None, columns=None, fill_value=None):
        """
        Dictionary of footer rows. Missing footer values will be returned as None.

        Parameters
        ----------
        labels : list, optional
            Footer rows to return values for. If not provided, all footer rows will be returned.
        columns : list, optional
            Columns to return footer values for. If not provided, all column footers will be returned.
        fill_value : optional, default None
            Value to use when no footer is found.

        Examples
        --------
        >>> ds = rt.Dataset({'colA': rt.arange(5), 'colB': rt.arange(5), 'colC': rt.arange(5)})
        >>> ds.footer_set_values('row1', {'colA':1, 'colC':2})
        >>> ds.footer_get_values()
        {'row1': [1, None, 2]}

        >>> ds.footer_get_values(columns=['colC','colA'])
        {'row1': [2, 1]}

        >>> ds.footer_remove()
        >>> ds.footer_get_values()
        {}

        Returns
        -------
        footers : dictionary
            Keys are footer row names.
            Values are lists of footer values or None, if missing.
        """
        if self.footers is None:
            return {}
        labels = self._footers_exist(labels)
        if labels is None:
            return {}

        if columns is None:
            columns = self.keys()
        if not isinstance(columns, list):
            columns = [columns]

        footerdict = { fname:[] for fname in labels }
        for colname in columns:
            coldict = self.col_get_attribute(colname, 'Footer')
            # column had no footers, fill with None
            if coldict is None:
                for v in footerdict.values():
                    v.append(fill_value)
            else:
                for k, v in footerdict.items():
                    v.append(coldict.get(k, fill_value))
        return footerdict

    # -------------------------------------------------------
    def footer_get_dict(self, labels=None, columns=None):
        """
        Dictionary of footer rows, the latter in dictionary form.

        Parameters
        ----------
        labels : list, optional
            Footer rows to return values for. If not provided, all footer rows will be returned.
        columns : list of str, optional
            Columns to return footer values for. If not provided, all column footers will be returned.

        Examples
        --------
        >>> ds = rt.Dataset({'colA': rt.arange(5), 'colB': rt.arange(5), 'colC': rt.arange(5)})
        >>> ds.footer_set_values('row1', {'colA':1, 'colC':2})
        >>> ds.footer_get_dict()
        {'row1': {'colA': 1, 'colC': 2}}

        >>> ds.footer_get_dict(columns=['colC','colA'])
        {'row1': [2, 1]}

        >>> ds.footer_remove()
        >>> ds.footer_get_dict()
        {}

        Returns
        -------
        footers : dictionary
            Keys are footer row names.
            Values are dictionaries of column name and value pairs.
        """
        if self.footers is None:
            return {}
        labels = self._footers_exist(labels)
        if labels is None:
            return {}

        if columns is None:
            columns = self.keys()
        if not isinstance(columns, list):
            columns = [columns]

        footerdict = { fname:{} for fname in labels }
        for colname in columns:
            coldict = self.col_get_attribute(colname, 'Footer')
            # column had no footers, fill with None
            if coldict is not None:
                for k, d in footerdict.items():
                    v = coldict.get(k, None)
                    if v:
                        d[colname] = v
        return footerdict

    # -------------------------------------------------------
    def footer_set_values(self, label:str, footerdict) -> None:
        """Assign footer values to specific columns.

        Parameters
        ----------
        label : string
            Name of existing or new footer row.
            This string will appear as a label on the left, below the right-most label key or row numbers.
        footerdict : dictionary
            Keys are valid column names (otherwise raises ValueError).
            Values are scalars. They will appear as a string with their default type formatting.

        Returns
        -------
        None

        Examples
        --------
        >>> ds = rt.Dataset({'colA': rt.arange(3), 'colB': rt.arange(3)*2})
        >>> ds.footer_set_values('sum', {'colA':3, 'colB':6})
        >>> ds
          #   colA   colB
        ---   ----   ----
          0      0      0
          1      1      2
          2      2      4
        ---   ----   ----
        sum      3      6

        >>> ds.colC = rt.ones(3)
        >>> ds.footer_set_values('mean', {'colC': 1.0})
        >>> ds
           #   colA   colB   colC
        ----   ----   ----   ----
           0      0      0   1.00
           1      1      2   1.00
           2      2      4   1.00
        ----   ----   ----   ----
         sum      3      6
        mean                 1.00

        Notes
        -----
        - Not all footers need to be set. Missing footers will appear as blank in final display.
        - Footers will appear in dataset slices as they do in the original dataset.
        - If the footer is a column total, it may need to be recalculated.
        - This routine can also be used to replace existing footers.

        See Also
        --------
        Dataset.footer_remove()
        """
        if not isinstance(label, str):
            raise TypeError(f"Footer labels must be string values, got {type(label)}")
        if not isinstance(footerdict, dict):
            raise TypeError(f"Footer mapping must be a dictionary of column names -> footer values for specified label {label}. Got {type(footerdict)}.")
        # prevent partial footers from being set
        self._ensure_atomic(footerdict,self.footer_set_values)

        if self.footers is None:
            # use a dict so footer row order is preserved
            self._footers = dict()

        self._footers[label]=None

        for colname, value in footerdict.items():
            coldict = self.col_get_attribute(colname, 'Footer')
            # create a new footer dict
            if coldict is None:
                coldict = {label:value}
                self.col_set_attribute(colname, 'Footer', coldict)

            # modify existing footer dict
            else:
                coldict[label]=value

    # -------------------------------------------------------
    def _prepare_display_data(self):
        """Prepare column headers, arrays, and column footers for display.
        Arrays will be aranged in order: Labels, sort columns, regular columns, right columns.
        """
        header_tups = None
        footer_tups = None
        array_data = None

        leftkeys = self.label_get_names()
        # no labels
        if len(leftkeys) == 0:
            leftcols = []
            # no row numbers callback
            if self._row_numbers is None:
                # use default row number header
                leftkeys = ['#']
        else:
            leftcols = [self[k] for k in leftkeys]

        sortkeys = []
        # col_sortlist might still be set even though sorts are off
        # only pull it if sorts are on
        if self._sort_display:
            if self._col_sortlist is not None:
                sortkeys = self._col_sortlist
        sortcols = [self[k] for k in sortkeys]

        rightkeys = self.summary_get_names()
        rightcols = [self[k] for k in rightkeys]

        mainkeys = [c for c in self if c not in leftkeys and c not in rightkeys and c not in sortkeys]
        maincols = [self[k] for k in mainkeys]

        footers = self.footers
        cols_with_footer = sortkeys + mainkeys + rightkeys
        if footers is not None:
            # create row for each footer label
            footerkeys = [*footers]
            # align footer label with right-most label column or row number column
            # assume not displaying label footers for now
            numleft = len(leftcols)
            if numleft < 2:
                padding = []
            else:
                # pad each row
                padding = [''] * (numleft-1)

            cols_with_footer = sortkeys + mainkeys + rightkeys
            footerdict = self.footer_get_values(columns=cols_with_footer, fill_value='')
            # lists for each footer row, empty string for blanks
            footerrows = [padding + [rowname] + footervals for rowname, footervals in footerdict.items()]
            # column footer tuples with string repr of each value
            footer_tups = [[ ColHeader(format_scalar(fval),1,0) for fval in frow] for frow in footerrows]

        # build all column header tuples
        allkeys = leftkeys + cols_with_footer
        header_tups = [[ ColHeader(k,1,0) for k in allkeys ]]

        # all arrays in one list
        array_data = leftcols + sortcols + maincols + rightcols

        return header_tups, array_data, footer_tups

    # -------------------------------------------------------
    def __str__(self):
        return self.make_table(DS_DISPLAY_TYPES.STR)

    # -------------------------------------------------------
    def __repr__(self):
        #if Struct._lastreprhtml != 0 and Struct._lastrepr > Struct._lastreprhtml and TypeRegister.DisplayOptions.HTML_DISPLAY:
        #    # this is an ODD condition
        #    print("HMTL is on, but repr called back to back.  consider rt.Display.display_html(False)")

        Struct._lastrepr =GetTSC()
        # this will be called before _repr_html_ in jupyter
        if TypeRegister.DisplayOptions.HTML_DISPLAY is False:
            result= self.make_table(DS_DISPLAY_TYPES.STR)
            # always turn off sorting once displayed
            self.sorts_off()
        else:
            result =self.make_table(DS_DISPLAY_TYPES.REPR)

        return result

    # -------------------------------------------------------
    def _repr_html_(self):
        Struct._lastreprhtml =GetTSC()
        if TypeRegister.DisplayOptions.HTML_DISPLAY is False:
            plainstring = self.make_table(DS_DISPLAY_TYPES.STR)
            # TJD this is a hack that needs to be reviewed
            # Believe it exists to display ds in a list
            print(DisplayString(plainstring))
            # jupyter lab will turn plain string into non-monospace font
            result = ""
        else:
            result =self.make_table(DS_DISPLAY_TYPES.HTML)

        # always turn off sorting once displayed
        self.sorts_off()
        return result

    # -------------------------------------------------------
    def add_matrix(self, arr, names: Optional[List[str]] = None) -> None:
        """
        Add a 2-dimensional matrix as columns in a dataset.

        Parameters
        ----------
        arr : 2-d ndarray
        names : list of str, optional
            optionally provide column names
        """

        if names is not None:
            if arr.shape[1] != len(names):
                raise ValueError(f'Provided names must match number of columns.')
        else:
            names = ['col_'+str(i) for i in range(arr.shape[1])]
        arr = arr.T
        for idx, name in enumerate(names):
            if name in self:
                warnings.warn(f"Overwriting column named {name}.")
            setattr(self, name, arr[idx])

    # -------------------------------------------------------
    def transpose(self, colnames: Optional[List[str]] = None, cats: bool = False, gb: bool = False, headername: str = 'Col') -> 'Dataset':
        """
        Return a transposed version of the Dataset.

        Parameters
        ----------
        colnames : list of str, optional
            Set to list of colnames you want transposed; defaults to None, which means all columns are included.
        cats : bool
             Set to True to include Categoricals in transposition. Defaults to False.
        gb : bool
            Set to True to include groupby keys (labels) in transposition. Defaults to False.
        headername : str
            The name of the column which was once all the column names. Defaults to 'Col'.

        Returns
        -------
        Dataset
            A transposed version of this Dataset instance.
        """

        def col_as_string(colname):
            c = self[colname]
            if isinstance(c, TypeRegister.Categorical):
                # todo should use expand_dict or categoricals should have a new routine
                return c.expand_array
            else:
                return c.astype('U')

        oldlabels = self.label_get_names()

        # first homogenize all the data to same dtype, and make 2d matrix
        t_array, colnames = self.imatrix_make(colnames =colnames, cats=cats, gb=gb, inplace=False, retnames=True)

        # rotate the matrix 90
        t_array = t_array.transpose()

        # the column names are now the rownames
        tds = Dataset({headername:colnames})
        numcols = t_array.shape[1]

        if len(oldlabels) == 0:
            # Just label all the column C0, C1, C2, etc.
            colnames = 'C' + arange(numcols).astype('U')
        else:
            # handle multikey with _ separator
            colnames = col_as_string(oldlabels[0])
            for i in range(1,len(oldlabels)):
                colnames = colnames + '_' + col_as_string(oldlabels[i])

        # extract each column in the 2d matrix
        for i in range(numcols):
            tds[colnames[i]] = t_array[:,i]

        # takes the column names running horiz, and makes them vertical
        tds.label_set_names([headername])
        return tds

    # -------------------------------------------------------
    def show_all(self, max_cols: int = 8) -> None:
        """
        Display all rows and up to the specified number of columns.

        Parameters
        ----------
        max_cols : int
            The maximum number of columns to display.

        Notes
        -----
        TODO: This method currently displays the data using 'print'; it should be deprecated or adapted
            to use our normal display code so it works e.g. in a Jupyter notebook.
        """
        i = 0
        num_cols = self.get_ncols()
        while i < num_cols:
            print(self[:, i:i + max_cols])
            i += max_cols

    # -------------------------------------------------------
    def sample(
        self, N: int = 10, filter: Optional[np.ndarray] = None,
        seed: Optional[Union[int, Sequence[int], np.random.SeedSequence, np.random.Generator]] = None
    ) -> 'Dataset':
        """
        Select N random samples from `Dataset` or `FastArray`.

        Parameters
        ----------
        N : int, optional, defaults to 10
            Number of rows to sample.
        filter : array-like (bool or rownums), optional, defaults to None
            Filter for rows to sample.
        seed : {None, int, array_like[ints], SeedSequence, Generator}, optional, defaults to None
            A seed to initialize the `Generator`. If None, the generator is initialized using
            fresh, random entropy data gathered from the OS.
            See the docstring for `np.random.default_rng` for additional details.

        Returns
        -------
        Dataset
        """

        return sample(self, N=N, filter=filter, seed=seed)

    # -------------------------------------------------------
    def _get_columns(self, cols: Union[str, Iterable[str]]) -> List[FastArray]:
        """internal routine used to create a list of one or more columns"""
        if not isinstance(cols, list):
            if isinstance(cols, str):
                cols=[cols]
            else:
                raise TypeError(f'The argument for accum2 or cat must be a list of column name(s) or a single column name.')

        cols = [self[colname] for colname in cols]
        return cols

    # -------------------------------------------------------
    def _makecat(self, cols):
        if not isinstance(cols, np.ndarray):
            cols = self._get_columns(cols)
            # if just one item in the list, extract it
            if len(cols)==1:
                cols = cols[0]

        return cols

    # -------------------------------------------------------
    def cat(self, cols: Union[str, Iterable[str]], **kwargs) -> 'Categorical':
        """
        Parameters
        ----------
        cols   : str or list of str
            A single column name or list of names to indicate which columns to build the categorical from
            or a numpy array to build the categoricals from
        kwargs : any valid keywords in the categorical constructor

        Returns
        -------
        Categorical
            A categorical with dataset set to self for groupby operations.

        Examples
        --------
        >>> np.random.seed(12345)
        >>> ds = rt.Dataset({'strcol': np.random.choice(['a','b','c'],4), 'numcol': rt.arange(4)})
        >>> ds
        #   strcol   numcol
        -   ------   ------
        0   c             0
        1   b             1
        2   b             2
        3   a             3

        >>> ds.cat('strcol').sum()
        *strcol   numcol
        -------   ------
        a              3
        b              3
        c              0
        """
        cols = self._makecat(cols)
        if not isinstance(cols, TypeRegister.Categorical):
            cols = TypeRegister.Categorical(cols, **kwargs)

        cols._dataset = self
        return cols

    # -------------------------------------------------------
    def cat2keys(
        self,
        cat_rows: Union[str, List[str]],
        cat_cols: Union[str, List[str]],
        filter: Optional[np.ndarray] = None,
        ordered: bool = True,
        sort_gb: bool = False,
        invalid: bool = False,
        fuse: bool = False
    ) -> 'Categorical':
        """
        Creates a :class:`~rt.rt_categorical.Categorical` with two sets of keys which have all possible unique combinations.

        Parameters
        ----------
        cat_rows : str or list of str
            A single column name or list of names to indicate which columns to build the categorical from
            or a numpy array to build the categoricals from.
        cat_cols : str or list of str
            A single column name or list of names to indicate which columns to build the categorical from
            or a numpy array to build the categoricals from.
        filter : ndarray of bools, optional
            only valid when invalid is set to True
        ordered : bool, default True
            only applies when `key1` or `key2` is not a categorical
        sort_gb : bool, default False
            only applies when `key1` or `key2` is not a categorical
        invalid : bool, default False
            Specifies whether or not to insert the invalid when creating the n x m unique matrix.
        fuse : bool, default False
            When True, forces the resulting categorical to have 2 keys, one for rows, and one for columns.

        Returns
        -------
        Categorical
            A categorical with at least 2 keys dataset set to self for groupby operations.

        Examples
        --------
        >>> ds = rt.Dataset({_k: list(range(_i * 2, (_i + 1) * 2)) for _i, _k in enumerate(["alpha", "beta", "gamma"])}); ds
        #   alpha   beta   gamma
        -   -----   ----   -----
        0       0      2       4
        1       1      3       5
        [2 rows x 3 columns] total bytes: 24.0 B
        >>> ds.cat2keys(['alpha', 'beta'], 'gamma').sum(rt.arange(len(ds)))
        *alpha   *beta   *gamma   col_0
        ------   -----   ------   -----
             0       2        4       0
             1       3        4       0
             0       2        5       0
             1       3        5       1

        [4 rows x 4 columns] total bytes: 80.0 B

        See Also
        --------
        rt_numpy.cat2keys
        rt_dataset.accum2
        """
        cat_rows = self._makecat(cat_rows)
        cat_cols = self._makecat(cat_cols)
        result = cat2keys(cat_rows, cat_cols, filter = filter, ordered=ordered, sort_gb=sort_gb, invalid=invalid, fuse=fuse)
        result._dataset = self
        return result

   # -------------------------------------------------------
    def accum1(self, cat_rows: List[str], filter=None, showfilter:bool=False, ordered:bool=True, **kwargs) -> GroupBy:
        """
        Returns the :class:`~rt.rt_groupby.GroupBy` object constructed from the Dataset
        with a 'Totals' column and footer.

        Parameters
        ----------
        cat_rows : list of str
            The list of column names to group by on the row axis. These columns will be
            made into a :class:`~rt.rt_categorical.Categorical`.
        filter : ndarray of bools, optional
            This parameter is unused.
        showfilter : bool, default False
            This parameter is unused.
        ordered : bool, default True
            This parameter is unused.
        sort_gb : bool, default True
            Set to False to change the display order.
        kwargs
            May be any of the arguments allowed by the Categorical constructor

        Returns
        -------
        GroupBy

        Examples
        --------
        >>> ds.accum1('symbol').sum(ds.TradeSize)
        """

        cat_rows = self.cat(cat_rows)
        return GroupBy(self, cat_rows, totals=True, **kwargs)

    # -------------------------------------------------------
    def accum2(
        self, cat_rows, cat_cols, filter=None, showfilter: bool = False,
        ordered: Optional[bool] = None, lex: Optional[bool] = None, totals: bool = True
    ) -> 'Accum2':
        """
        Returns the Accum2 object constructed from the dataset.

        Parameters
        ----------
        cat_rows : list
            The list of column names to group by on the row axis.  This will be made into a categorical.
        cat_cols : list
            The list of column names to group by on the column axis.  This will be made into a categorical.
        filter
            TODO
        showfilter : bool
            Used in Accum2 to show filtered out data.
        ordered : bool, optional
            Defaults to None.  Set to True or False to change the display order.
        lex : bool
            Defaults to None.  Set to True for high unique counts.  It will override `ordered` when set to True.
        totals : bool, default True
            Set to False to not show Total column.

        Returns
        -------
        Accum2

        Examples
        --------
        >>> ds.accum2('symbol', 'exchange').sum(ds.TradeSize)
        >>> ds.accum2(['symbol','exchange'], 'date', ordered=True).sum(ds.TradeSize)
        """

        cat_rows = self.cat(cat_rows, ordered=ordered, lex=lex)
        cat_cols = self.cat(cat_cols, ordered=ordered, lex=lex)

        # calling with rows, cols to match unstack() more closely
        result = TypeRegister.Accum2(cat_rows, cat_cols, filter= filter, showfilter = showfilter, ordered=ordered, totals=totals)
        # attach dataset to accum2 object so argument can be ommitted during calculation
        result._dataset = self
        return result

    # -------------------------------------------------------
    def groupby(self, by: Union[str, List[str]], **kwargs) -> GroupBy:
        """
        Returns an :class:`~rt.rt_groupby.GroupBy` object constructed from the dataset.

        This function can accept any keyword arguments (in `kwargs`) allowed by the :class:`~rt.rt_groupby.GroupBy` constructor.

        Parameters
        ----------
        by: str or list of str
            The list of column names to group by

        Other Parameters
        ----------------
        filter: ndarray of bool
            Pass in a boolean array to filter data.  If a key no longer exists after filtering
            it will not be displayed.
        sort_display : bool
            Defaults to True. set to False if you want to display data in the order of appearance.
        lex : bool
            When True, use a lexsort to the data.

        Returns
        -------
        GroupBy

        Examples
        --------
        All calculations from GroupBy objects will return a Dataset. Operations can be called in the following ways:

        Initialize dataset and groupby a single key:

        >>> #TODO: Need to call np.random.seed(12345) here to deterministically init the RNG used below
        >>> d = {'strings':np.random.choice(['a','b','c','d','e'], 30)}
        >>> for i in range(5): d['col'+str(i)] = np.random.rand(30)
        >>> ds = rt.Dataset(d)
        >>> gb = ds.groupby('strings')

        Perform operation on all columns:

        >>> gb.sum()
        *strings   col0   col1   col2   col3   col4
        --------   ----   ----   ----   ----   ----
        a          2.67   3.35   3.74   3.46   4.20
        b          1.36   1.53   2.59   1.24   0.73
        c          3.91   2.00   2.76   2.62   2.10
        d          4.76   5.13   4.30   3.46   2.21
        e          4.18   2.86   2.95   3.22   3.14

        Perform operation on a single column:

        >>> gb['col1'].mean()
        *strings   col1
        --------   ----
        a          0.48
        e          0.38
        d          0.40
        d          0.64
        c          0.48

        Perform operation on multiple columns:

        >>> gb[['col1','col2','col4']].min()
        *strings   col1   col2   col4
        --------   ----   ----   ----
        a          0.05   0.03   0.02
        e          0.02   0.24   0.02
        d          0.03   0.15   0.16
        d          0.17   0.19   0.05
        c          0.00   0.03   0.28

        Perform specific operations on specific columns:

        >>> gb.agg({'col1':['min','max'], 'col2':['sum','mean']})
                      col1          col2
        *strings    Min    Max    Sum   Mean
        --------   ----   ----   ----   ----
        a          0.05   0.92   3.74   0.53
        b          0.02   0.72   2.59   0.65
        c          0.03   0.73   2.76   0.55
        d          0.17   0.96   4.30   0.54
        e          0.00   0.82   2.95   0.49

        GroupBy objects can also be grouped by multiple keys:

        >>> gbmk = ds.groupby(['strings', 'col1'])
        >>> gbmk
        *strings   *col1   Count
        --------   -----   -----
        a           0.05       1
        .           0.11       1
        .           0.16       1
        .           0.55       1
        .           0.69       1
                 ...     ...
        e           0.33       1
        .           0.36       1
        .           0.68       1
        .           0.68       1
        .           0.82       1
        """
        return GroupBy(self, by, **kwargs)

    # -------------------------------------------------------
    def gb(self, by, **kwargs):
        """Equivalent to :meth:`~rt.rt_dataset.Dataset.groupby`"""
        return self.groupby(by, **kwargs)

    # -------------------------------------------------------
    def gbu(self, by, **kwargs):
        """Equivalent to :meth:`~rt.rt_dataset.Dataset.groupby` with sort=False"""
        kwargs['sort_display'] = False
        return self.groupby(by, **kwargs)

    #--------------------------------------------------------------------------
    def gbrows(self, strings:bool=False, dtype=None, **kwargs) -> GroupBy:
        """
        Create a GroupBy object based on "computable" rows or string rows.

        Parameters
        ----------
        strings : bool
            Defaults to False. Set to True to process strings.
        dtype : str or numpy.dtype, optional
            Defaults to None.  When set, all columns will be cast to this dtype.
        kwargs
            Any other kwargs will be passed to ``groupby()``.

        Returns
        -------
        GroupBy

        Examples
        --------
        >>> ds = rt.Dataset({'a': rt.arange(3), 'b': rt.arange(3.0), 'c':['Jim','Jason','John']})
        >>> ds.gbrows()
        GroupBy Keys ['RowNum'] @ [2 x 3]
        ikey:True  iFirstKey:False  iNextKey:False  nCountGroup:False _filter:False  _return_all:False
        <BLANKLINE>
        *RowNum   Count
        -------   -----
            0       2
            1       2
            2       2

        >>> ds.gbrows().sum()
        *RowNum    Row
        -------   ----
            0   0.00
            1   2.00
            2   4.00
        <BLANKLINE>
        [3 rows x 2 columns] total bytes: 36.0 B

        Example usage of the string-processing mode of ``gbrows()``:

        >>> ds.gbrows(strings=True)
        GroupBy Keys ['RowNum'] @ [2 x 3]
        ikey:True  iFirstKey:False  iNextKey:False  nCountGroup:False _filter:False  _return_all:False
        <BLANKLINE>
        *RowNum   Count
        -------   -----
            0       1
            1       1
            2       1
        """
        if strings:
            rowlist = list(self.noncomputable().values())
        else:
            rowlist = list(self.computable().values())

        # use our hstack
        hs = hstack(rowlist, dtype=dtype)

        #create a categorical of integers so we can group by
        arng = arange(self._nrows)
        cat = TypeRegister.Categorical(tile(arng, len(rowlist)), arng, base_index=0)

        #create a dataset with two columns
        ds=Dataset({'Row':hs,'RowNum':cat})
        return ds.groupby('RowNum', **kwargs)

    # -------------------------------------------------------
    # Reduction functions.
    def reduce(self, func, axis: Optional[int] = 0, as_dataset: bool = True, fill_value=None, **kwargs) -> Union['Dataset', Struct, FastArray, np.generic]:
        """
        Returns calculated reduction along axis.

        .. note::

            Behavior for ``axis=None`` differs from pandas!

            The default `fill_value` is ``None`` (drop) to ensure the most sensible default
            behavior for ``axis=None`` and ``axis=1``. As a thought problem, consider all
            three axis behaviors for func=sum or product.

        Parameters
        ----------
        func : reduction function (e.g. numpy.sum, numpy.std, ...)
        axis : int, optional
            * 0: reduce over columns, returning a Struct (or Dataset) of scalars.
              Reasonably cheap. String synonyms: ``c``, ``C``, ``col``, ``COL``, ``column``, ``COLUMN``.
            * 1: reduce over rows, returning an array of scalars.
              Could well be expensive/slow. String synonyms: ``r``, ``R``, ``row``, ``ROW``.
            * ``None``: reduce over rows and columns, returning a scalar.
              Could well be very expensive/slow. String synonyms: ``all``, ``ALL``.
        as_dataset : bool
            When `axis` is 0, this flag specifies a Dataset should be returned instead of a Struct. Defaults to False.
        fill_value
            * fill_value=None (default) -> drop all non-computable type columns from result

            * fill_value=alt_func -> force computation with alt_func
                                      (for axis=1 must work on indiv. elements)
            * fill_value=scalar   -> apply as uniform fill value

            * fill_value=dict (defaultdict) of colname->fill_value, where
                   None (or absent if not a defaultdict) still means drop column
                   and an alt_func still means force compute via alt_func.
        kwargs
            all other kwargs are passed to `func`

        Returns
        -------
        Struct or Dataset or array or scalar
        """

        def _reduce_fill_values( fill_value):
            """
            return two lists:
                fvals: set to None if computable, set to fill value if noncomputable
                noncomp: set to True if not computable, otherwise False
            """

            noncomp = [False] * self.get_ncols()
            fvals = [None] * self.get_ncols()
            for colnum, colname in enumerate(self.keys()):
                _v = self.col_get_value(colname)
                if not _v.iscomputable():
                    noncomp[colnum] = True
                    if isinstance(fill_value, dict):
                        # try/catch instead of get() to support defaultdict usage
                        try:
                            fvals[colnum] = fill_value[colname]
                        except KeyError:
                            pass
                    else:
                        fvals[colnum] = fill_value
            return fvals, noncomp


        axis = self._axis_key(axis)
        cond_rtn_type = type(self) if as_dataset else Struct
        fvals, noncomp = _reduce_fill_values(fill_value)

        if axis == 0:
            od = {}

            # remove axis from kwargs
            kwargs.pop('axis', None)

            for _i, _k in enumerate(self.keys()):
                _v = self.col_get_value(_k)
                #print("func", func,  'colname', _k, 'dtype', _v.dtype, "v", _v, "kwargs:", kwargs)
                # not all arrays are computable, such as the std of a string array
                fval = fvals[_i]
                if not noncomp[_i]:
                    od[_k] = func(_v, **kwargs)
                elif callable(fval):
                    od[_k] = fval(_v, **kwargs)
                elif fval is not None:
                    od[_k] = fval
            return cond_rtn_type(od)

        if axis == 1:
            if fill_value is None:
                # new fast path
                return func(self.imatrix_make(), axis=1, **kwargs)

            if not any(noncomp):
                # does not respect noncomputable cols.
                # 2.74 ms ± 6.18 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
                # return np.array([func(np.array(self[_r, :].tolist()), **kwargs) for _r in range(self.get_nrows())])
                # 267 µs ± 2 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
                return FastArray([func(_r, **kwargs) for _r in self.asrows(as_type='array')])
            # respects noncomputable cols.
            # 448 µs ± 1.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
            def _row(_i):
                _r = [arr[_i] for arr in self.values()]
                _keep = np.ones(len(_r), dtype=bool)
                for _i, _nc in enumerate(noncomp):
                    if _nc:
                        fval = fvals[_i]
                        if callable(fval):
                            _r[_i] = fval(_r[_i], **kwargs)
                        elif fval is not None:
                            _r[_i] = fval
                        else:
                            _keep[_i] = False
                if _keep.all():
                    return _r
                return [_x for _i, _x in enumerate(_r) if _keep[_i]]  # cannot use np.take!!!

            # TJD this code is slow and needs review
            return np.array([func(_row(_i), **kwargs) for _i in range(self.get_nrows())])

        if axis is None:
            if not any(noncomp):
                # does not respect noncomputable cols.
                # np.ravel doc suggests this to be the most likely to be efficient
                # 34.9 µs ± 57.9 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
                return func(np.reshape([self.col_get_value(_k) for _k in self.keys()], -1), **kwargs)
            # respects noncomputable cols.
            # 290 µs ± 1.86 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
            bycols = self.reduce(func, axis=0, as_dataset=True, fill_value=fill_value, **kwargs)
            return func(np.array(list(bycols.values())))
        raise NotImplementedError('Dataset.reduce(axis=<0, 1, None>)')

    def argmax(self, axis=0, as_dataset=True, fill_value=None):
        return self.reduce(argmax, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    def argmin(self, axis=0, as_dataset=True, fill_value=None):
        return self.reduce(argmin, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    def normalize_zscore(self, axis=0, as_dataset=True, fill_value=None):
        return self.reduce(normalize_zscore, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    def normalize_minmax(self,  axis=0, as_dataset=True, fill_value=None):
        return self.reduce(normalize_minmax, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    def sum(self, axis=0, as_dataset=True, fill_value=None):
        """See documentation of :meth:`~rt.rt_dataset.Dataset.reduce`"""
        return self.reduce(sum, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    def mean(self, axis=0, as_dataset=True, fill_value=None):
        """See documentation of :meth:`~rt.rt_dataset.Dataset.reduce`"""
        return self.reduce(mean, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    def var(self, axis=0, ddof=1, as_dataset=True, fill_value=None):
        """See documentation of :meth:`~rt.rt_dataset.Dataset.reduce`"""
        return self.reduce(var, axis=axis, as_dataset=as_dataset, fill_value=fill_value, ddof=ddof)

    def std(self, axis=0, ddof=1, as_dataset=True, fill_value=None):
        """See documentation of :meth:`~rt.rt_dataset.Dataset.reduce`"""
        return self.reduce(std, axis=axis, as_dataset=as_dataset, fill_value=fill_value, ddof=ddof)

    def median(self, axis=0, as_dataset=True, fill_value=None):
        """See documentation of :meth:`~rt.rt_dataset.Dataset.reduce`"""
        return self.reduce(median, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    def min(self, axis=0, as_dataset=True, fill_value=min):
        """See documentation of :meth:`~rt.rt_dataset.Dataset.reduce`"""
        return self.reduce(min, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    def max(self, axis=0, as_dataset=True, fill_value=max):
        """See documentation of :meth:`~rt.rt_dataset.Dataset.reduce`"""
        return self.reduce(max, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    def count(self, axis=0, as_dataset=True, fill_value=len):
        """See documentation of :meth:`~rt.rt_dataset.Dataset.reduce`"""
        # We should have another counting the non-no-data elements, but need to wait on safe-arrays.
        return self.reduce(len, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    #---NAN FUNCS--------------------------------------------------------------
    def nanargmax(self, axis=0, as_dataset=True, fill_value=None):
        return self.reduce(nanargmax, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    def nanargmin(self, axis=0, as_dataset=True, fill_value=None):
        return self.reduce(nanargmin, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    def nansum(self, axis=0, as_dataset=True, fill_value=None):
        """See documentation of :meth:`~rt.rt_dataset.Dataset.reduce`"""
        return self.reduce(nansum, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    def nanmean(self, axis=0, as_dataset=True, fill_value=None):
        """See documentation of :meth:`~rt.rt_dataset.Dataset.reduce`"""
        return self.reduce(nanmean, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    def nanvar(self, axis=0, ddof=1, as_dataset=True, fill_value=None):
        """See documentation of :meth:`~rt.rt_dataset.Dataset.reduce`"""
        return self.reduce(nanvar, axis=axis, as_dataset=as_dataset, fill_value=fill_value, ddof=ddof)

    def nanstd(self, axis=0, ddof=1, as_dataset=True, fill_value=None):
        """See documentation of :meth:`~rt.rt_dataset.Dataset.reduce`"""
        return self.reduce(nanstd, axis=axis, as_dataset=as_dataset, fill_value=fill_value, ddof=ddof)

    def nanmedian(self, axis=0, as_dataset=True, fill_value=None):
        """See documentation of :meth:`~rt.rt_dataset.Dataset.reduce`"""
        return self.reduce(nanmedian, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    def nanmin(self, axis=0, as_dataset=True, fill_value=min):
        """See documentation of :meth:`~rt.rt_dataset.Dataset.reduce`"""
        return self.reduce(nanmin, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    def nanmax(self, axis=0, as_dataset=True, fill_value=max):
        """See documentation of :meth:`~rt.rt_dataset.Dataset.reduce`"""
        return self.reduce(nanmax, axis=axis, as_dataset=as_dataset, fill_value=fill_value)

    #--------------------------------------------------------------------------
    def quantile(self, q: Optional[List[float]] = None, fill_value=None):
        """

        Parameters
        ----------
        q: defaults to [0.50], list of quantiles
        fill_value: optional place-holder value for non-computable columns

        Returns
        -------
        Dataset.
        """
        if q is None:
            q = [0.50]

        # TODO NW Should be a String
        labels = np.asanyarray(q)
        if not isinstance(fill_value, (list, np.ndarray, dict, type(None))):
            fill_value = [fill_value] * len(labels)
        retval = self.reduce(quantile, q=q, as_dataset=True, fill_value=fill_value)
        retval.Stats = labels
        retval.col_move_to_front(['Stats'])
        retval.label_set_names(['Stats'])
        return retval

    #--------------------------------------------------------------------------
    def describe(self, q: Optional[List[float]] = None, fill_value = None) -> 'Dataset':
        """
        Similar to pandas describe; columns remain stable, with extra column (Stats) added for names.

        .. Caution:: This routine can be expensive if the dataset is large.

        Parameters
        ----------
        q : list of float, optional
            List of quantiles to calculate.
            If not specified, defaults to ``[0.10, 0.25, 0.50, 0.75, 0.90]``.
        fill_value: optional
            Optional place-holder value for non-computable columns.

        Returns
        -------
        Dataset
            A Dataset containing the calculated, per-column quantile values.

        See Also
        --------
        FastArray.describe()
        """
        return describe(self, q=q, fill_value=fill_value)

    #--------------------------------------------------------------------------
    def melt(self, id_vars=None, value_vars=None, var_name:Optional[str]=None, value_name:str='value', trim:bool=False) -> 'Dataset':
        """
        "Unpivots" a Dataset from wide format to long format, optionally leaving identifier
        variables set.

        This function is useful to massage a Dataset into a format where one or more columns
        are identifier variables (id_vars), while all other columns, considered measured variables
        (value_vars), are "unpivoted" to the row axis, leaving just two non-identifier columns,
        'variable' and 'value'.

        Parameters
        ----------
        id_vars : tuple, list, or ndarray, optional
            Column(s) to use as identifier variables.
        value_vars : tuple, list, or ndarray, optional
            Column(s) to unpivot. If not specified, uses all columns that are not set as id_vars.
        var_name : str, optional
            Name to use for the 'variable' column. If None it uses 'variable'.
        value_name : str
            Name to use for the 'value' column. Defaults to 'value'.
        trim : bool
            defaults to False.  Set to True to drop zeros or nan (trims a dataset)

        Notes
        -----
        BUG: the current version does not handle categoricals correctly.
        """
        if id_vars is not None:
            if not is_list_like(id_vars):
                id_vars = [id_vars]
            else:
                id_vars = list(id_vars)
        else:
            id_vars = []

        if value_vars is not None:
            if not is_list_like(value_vars):
                value_vars = [value_vars]
            else:
                value_vars = list(value_vars)
            tempdict = self[id_vars + value_vars].asdict()
        else:
            tempdict = self.asdict()

        if var_name is None:
            var_name = 'variable'

        N = self._nrows
        K = len(tempdict) - len(id_vars)

        #create an empty dataset
        mdata = type(self)({})

        # reexpand any categoricals
        for col in id_vars:
            id_data = tempdict.pop(col)
            if TypeRegister.is_binned_array(id_data):
                # note: multikey categorical expands to a tuple of arrays
                # previously raised an error on expand array
                id_data = id_data.expand_array
            mdata[col] = np.tile(id_data._np,K)

        mdata[var_name] = FastArray(list(tempdict.keys())).repeat(N)
        mdata[value_name] = hstack(tempdict.values())
        if trim:
            goodmask = ~mdata[value_name].isnanorzero()
            mdata=mdata[goodmask,:]
        return mdata

    #--------------------------------------------------------------------------
    @classmethod
    def hstack(cls, ds_list, destroy: bool = False) -> 'Dataset':
        """
        Stacks columns from multiple datasets.

        See Also
        --------
        Dataset.concat_rows
        """
        return cls.concat_rows(ds_list, destroy=destroy)

    #--------------------------------------------------------------------------
    @classmethod
    def concat_rows(cls, ds_list: Iterable['Dataset'], destroy: bool = False) -> 'Dataset':
        """
        Stacks columns from multiple datasets.

        If a dataset is missing a column that appears in others, it will fill the gap with the default invalid for that column's type.
        Categoricals will be merged and stacked.
        Column types will be checked to make sure they can be safely stacked - no general type mismatch allowed.
        Columns of the same name must have the same number of dimension in each dataset (1 or 2 dimensions allowed)

        Parameters
        ----------
        ds_list : iterable of Dataset
            The Datasets to be concatenated
        destroy : bool
            Set to True to destroy any dataset in the list to save memory. Defaults to False.

        Returns
        -------
        Dataset
            A new Dataset created from the concatenated rows of the input Datasets.

        Examples
        --------
        Basic:

        >>> ds1 = rt.Dataset({'col_'+str(i):np.random.rand(5) for i in range(3)})
        >>> ds2 = rt.Dataset({'col_'+str(i):np.random.rand(5) for i in range(3)})
        >>> ds1
        #   col_0   col_1   col_2
        -   -----   -----   -----
        0    0.39    0.80    0.64
        1    0.54    0.80    0.36
        2    0.14    0.75    0.86
        3    0.05    0.61    0.95
        4    0.37    0.39    0.03

        >>> ds2
        #   col_0   col_1   col_2
        -   -----   -----   -----
        0    0.09    0.75    0.37
        1    0.90    0.34    0.17
        2    0.52    0.32    0.78
        3    0.37    0.20    0.34
        4    0.73    0.69    0.41

        >>> rt.Dataset.concat_rows([ds1, ds2])
        #   col_0   col_1   col_2
        -   -----   -----   -----
        0    0.39    0.80    0.64
        1    0.54    0.80    0.36
        2    0.14    0.75    0.86
        3    0.05    0.61    0.95
        4    0.37    0.39    0.03
        5    0.09    0.75    0.37
        6    0.90    0.34    0.17
        7    0.52    0.32    0.78
        8    0.37    0.20    0.34
        9    0.73    0.69    0.41

        With columns missing in one from some datasets:

        >>> ds1 = rt.Dataset({'col_'+str(i):np.random.rand(5) for i in range(3)})
        >>> ds2 = rt.Dataset({'col_'+str(i):np.random.rand(5) for i in range(2)})
        >>> rt.Dataset.concat_rows([ds1, ds2])
        #   col_0   col_1   col_2
        -   -----   -----   -----
        0    0.78    0.64    0.98
        1    0.61    0.87    0.85
        2    0.57    0.42    0.90
        3    0.82    0.50    0.60
        4    0.19    0.16    0.23
        5    0.69    0.83     nan
        6    0.07    0.82     nan
        7    0.58    0.34     nan
        8    0.69    0.38     nan
        9    0.89    0.07     nan

        With categorical column:

        >>> ds1 = rt.Dataset({'cat_col': rt.Categorical(['a','a','b','c','a']),
        ...                   'num_col': np.random.rand(5)})
        >>> ds2 = rt.Dataset({'cat_col': rt.Categorical(['b','b','a','c','d']),
        ...                   'num_col': np.random.rand(5)})
        >>> rt.Dataset.concat_rows([ds1, ds2])
        #   cat_col   num_col
        -   -------   -------
        0   a            0.38
        1   a            0.71
        2   b            0.84
        3   c            0.47
        4   a            0.18
        5   b            0.18
        6   b            0.47
        7   a            0.16
        8   c            0.96
        9   d            0.88

        Multiple dimensions (note: numpy v-stack will be used to concatenate 2-dimensional columns):

        >>> ds1 = rt.Dataset({'nums': rt.ones((4,4))})
        >>> ds1
        #                       nums
        -   ------------------------
        0   [1.00, 1.00, 1.00, 1.00]
        1   [1.00, 1.00, 1.00, 1.00]
        2   [1.00, 1.00, 1.00, 1.00]
        3   [1.00, 1.00, 1.00, 1.00]

        >>> ds2 = rt.Dataset({'nums': rt.zeros((4,4))})
        >>> ds2
        #                       nums
        -   ------------------------
        0   [0.00, 0.00, 0.00, 0.00]
        1   [0.00, 0.00, 0.00, 0.00]
        2   [0.00, 0.00, 0.00, 0.00]
        3   [0.00, 0.00, 0.00, 0.00]

        >>> rt.Dataset.concat_rows([ds1, ds2])
        #                       nums
        -   ------------------------
        0   [1.00, 1.00, 1.00, 1.00]
        1   [1.00, 1.00, 1.00, 1.00]
        2   [1.00, 1.00, 1.00, 1.00]
        3   [1.00, 1.00, 1.00, 1.00]
        4   [0.00, 0.00, 0.00, 0.00]
        5   [0.00, 0.00, 0.00, 0.00]
        6   [0.00, 0.00, 0.00, 0.00]
        7   [0.00, 0.00, 0.00, 0.00]

        Multiple dimensions with missing columns (sentinels/invalids will be flipped to final vstack dtype)

        >>> ds1 = rt.Dataset({'nums': rt.ones((5,5)), 'nums2': rt.zeros((5,5), dtype=np.float64)})
        >>> ds2 = rt.Dataset({'nums': rt.ones((5,5))})
        >>> ds3 = rt.Dataset({'nums': rt.ones((5,5)), 'nums2': rt.zeros((5,5), dtype=np.int8)})
        >>> rt.Dataset.concat_rows([ds1, ds2, ds3])
         #                             nums                            nums2
        --   ------------------------------   ------------------------------
         0   [1.00, 1.00, 1.00, 1.00, 1.00]   [0.00, 0.00, 0.00, 0.00, 0.00]
         1   [1.00, 1.00, 1.00, 1.00, 1.00]   [0.00, 0.00, 0.00, 0.00, 0.00]
         2   [1.00, 1.00, 1.00, 1.00, 1.00]   [0.00, 0.00, 0.00, 0.00, 0.00]
         3   [1.00, 1.00, 1.00, 1.00, 1.00]   [0.00, 0.00, 0.00, 0.00, 0.00]
         4   [1.00, 1.00, 1.00, 1.00, 1.00]   [0.00, 0.00, 0.00, 0.00, 0.00]
         5   [1.00, 1.00, 1.00, 1.00, 1.00]        [nan, nan, nan, nan, nan]
         6   [1.00, 1.00, 1.00, 1.00, 1.00]        [nan, nan, nan, nan, nan]
         7   [1.00, 1.00, 1.00, 1.00, 1.00]        [nan, nan, nan, nan, nan]
         8   [1.00, 1.00, 1.00, 1.00, 1.00]        [nan, nan, nan, nan, nan]
         9   [1.00, 1.00, 1.00, 1.00, 1.00]        [nan, nan, nan, nan, nan]
        10   [1.00, 1.00, 1.00, 1.00, 1.00]   [0.00, 0.00, 0.00, 0.00, 0.00]
        11   [1.00, 1.00, 1.00, 1.00, 1.00]   [0.00, 0.00, 0.00, 0.00, 0.00]
        12   [1.00, 1.00, 1.00, 1.00, 1.00]   [0.00, 0.00, 0.00, 0.00, 0.00]
        13   [1.00, 1.00, 1.00, 1.00, 1.00]   [0.00, 0.00, 0.00, 0.00, 0.00]
        14   [1.00, 1.00, 1.00, 1.00, 1.00]   [0.00, 0.00, 0.00, 0.00, 0.00]
        """
        return hstack_any(ds_list, cls, Dataset, destroy=destroy)

    #--------------------------------------------------------------------------
    @classmethod
    def concat_columns(cls, dsets, do_copy:bool, on_duplicate:str='raise', on_mismatch:str='warn'):
        r"""
        Concatenates a list of Datasets or Structs horizontally.

        Parameters
        ----------
        cls : class
            The class (Dataset)
        dsets : iterable
            An iterable of Datasets
        do_copy : bool
            Makes deep copies of arrays if set to True
        on_duplicate : {'raise', 'first', 'last'}
            Governs behavior in case of duplicate columns.
        on_mismatch : {'warn', 'raise', 'ignore'}
            Optional, governs behavior for allowed duplicate column names, how to
            address mismatched column values; can be 'warn' (default), 'raise' or 'ignore'.

        Returns
        -------
        Dataset
            The resulting dataset after concatenation.

        Examples
        --------
        With the ``'last'`` `on_duplicate` option:

        >>> N = 5
        >>> dset1 = rt.Dataset(dict(A=rt.arange(N), B=rt.ones(N), C=N*['c']))
        >>> dset2 = rt.Dataset(dict(A=rt.arange(N, 2*N, 1), B=rt.zeros(N), D=N*['d']))
        >>> dsets = [dset1, dset2]
        >>> rt.Dataset.concat_columns(dsets, do_copy=True, on_duplicate='last')
        #   A      B   C   D
        -   -   ----   -   -
        0   5   0.00   c   d
        1   6   0.00   c   d
        2   7   0.00   c   d
        3   8   0.00   c   d
        4   9   0.00   c   d
        <BLANKLINE>
        [5 rows x 4 columns] total bytes: 70.0 B

        With the default (``'raise'``) for the `on_duplicate` option:

        >>> rt.Dataset.concat_columns(dsets, do_copy=True)
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        File "C:\ProgramData\Anaconda3\envs\riptable-dev37\lib\site-packages\riptable-0.0.0-py3.7-win-amd64.egg\riptable\rt_dataset.py", line 4308, in concat_columns
            raise KeyError(f"Duplicate column '{column}'")
        KeyError: "Duplicate column 'A'"
        """
        # check that all Datasets have the same number of rows
        if on_duplicate not in ('raise', 'first', 'last'):
            raise ValueError(f"Invalid on_duplicate '{on_duplicate}'")
        if on_mismatch not in ('raise', 'warn', 'ignore'):
            raise ValueError(f"Invalid on_mismatch '{on_mismatch}'")

        # if there are no Datasets ...
        if len(dsets) == 0:
            raise ValueError("No Datasets to concatenate")
        if len(dsets) == 1 and not do_copy:
            return dsets[0]

        #try to convert any structs to dsets
        newdset=[]
        for d in dsets:
            # check if even a dataset, if not try to convert it
            try:
                # test to see if a dataset
                rownum = d._nrows
            except:
                #try to convert to a dataset (probably from struct)
                try:
                    d = Dataset(d)
                except:
                    #for c in d:
                    #    print("col", c, type(d[c]), len(d[c]), d[c])
                    raise ValueError(f"Unable to convert {d!r} to a Dataset")
            newdset.append(d)

        dsets = newdset
        # check for same length
        rownum_set = set([d.shape[0] for d in dsets])
        if len(rownum_set) != 1:
            raise ValueError(f'Inconsistent Dataset lengths {rownum_set}')

        # create dictionary
        dict_retval = {}
        columns = set()
        dups = set()
        for column, a in [(c, v) for d in dsets for c, v in d.items()]:
            if column in columns:
                if on_mismatch != 'ignore':
                    # print(f'on_mismatch={on_mismatch} column={column}')
                    dups.add(column)
                if on_duplicate == 'raise':
                    raise KeyError(f"Duplicate column '{column}'")
                elif on_duplicate == 'first':
                    pass
                else:
                    dict_retval[column] = a.copy() if do_copy else a
            else:
                columns.add(column)
                dict_retval[column] = a.copy() if do_copy else a

        if on_mismatch != 'ignore':
            if len(dups) > 0:
                if on_mismatch == 'warn':
                    warnings.warn(f'concat_columns() duplicate column mismatch: {dups!r}')
                if on_mismatch == 'raise':
                    raise RuntimeError(f'concat_columns() duplicate column mismatch: {dups!r}')

        return cls(dict_retval)

    # TODO: get .char and check list
    #--------------------------------------------------------------------------
    def _is_float_encodable(self, xtype):
        return xtype in (int, float, np.integer, np.floating,
                         np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64,
                         np.float16, np.float32, np.float64)

    #--------------------------------------------------------------------------
    def _ipython_key_completions_(self):
        return self.keys()

    #--------------------------------------------------------------------------
    def _normalize_column(self, x, field_key):
        original_type = x.dtype
        category_values = None
        is_categorical = False
        if self._is_float_encodable(original_type):
            if isinstance(x, TypeRegister.Categorical):
                category_values = x._categories
                is_categorical = True
            vals = x.astype(np.float64)
        else:
            if field_key is None:
                category_values, vals = unique(x, return_inverse=True)
                vals = vals.astype(np.float64)
            else:
                category_values = field_key
                isValid, vals = ismember(x, category_values, 1)
                vals = vals.astype(np.float64)
                vals[~isValid] = np.nan
        return vals, original_type, is_categorical, category_values

    #--------------------------------------------------------------------------
    def as_matrix(self, save_metadata=True, column_data={}):
        columns = list(self.keys())
        nrows = self.shape[0]
        ncols = self.shape[1]  # TODO: may expand this for 64-bit columns
        out_array = empty((nrows, ncols), dtype=np.float64)
        column_info = {}
        for col in range(ncols):
            field_key = column_data.get(columns[col])
            out_array[:, col], original_type, is_categorical, category_values = self._normalize_column(
                self[columns[col]], field_key)
            column_info[columns[col]] = {'dtype': original_type, 'category_values': category_values,
                                         'is_categorical': is_categorical}

        if save_metadata:
            return out_array, column_info
        else:
            return out_array

    # -------------------------------------------------------------------
    def as_recordarray(self):
        """
        Convert Dataset to one array (record array).

        Wrapped class arrays such as Categorical and DateTime will lose their type
        TODO: Expand categoricals

        Examples
        --------
        >>> ds = rt.Dataset({'a': rt.arange(3), 'b': rt.arange(3.0), 'c':['Jim','Jason','John']})
        >>> ds.as_recordarray()
        rec.array([(0, 0., b'Jim'), (1, 1., b'Jason'), (2, 2., b'John')],
                  dtype=[('a', '<i4'), ('b', '<f8'), ('c', 'S5')])

        >>> ds.as_recordarray().c
        array([b'Jim', b'Jason', b'John'], dtype='|S5')

        See Also
        --------
        numpy.core.records.array
        """
        # TODO: optionally? expand categoricals
        vals = self.values()
        names = self.keys()
        ra=np.core.records.fromarrays(list(vals), names=names)
        return ra

    # -------------------------------------------------------------------
    def as_struct(self):
        # TJD: NOTE need test for this
        """
        Convert a dataset to a struct.

        If the dataset is only one row, the struct will be of scalars.

        Returns
        -------
        Struct
        """

        mydict = self.asdict()
        if self._nrows == 1:
            olddict=mydict
            mydict={}
            # copy over just first and only element
            for colname, array in olddict.items():
                mydict[colname]=array[0]
        return TypeRegister.Struct(mydict)

    # -------------------------------------------------------------------
    def apply_rows(self, pyfunc, *args, otypes=None, doc=None, excluded =None, cache=False, signature=None):
        """
        Will convert the dataset to a recordarray and then call np.vectorize

        Applies a vectorized function which takes a nested sequence of objects or
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

        Example
        -------
        >>> ds = rt.Dataset({'a':arange(3), 'b':arange(3.0), 'c':['Jim','Jason','John']}, unicode=True)
        >>> ds.apply_rows(lambda x: x[2] + str(x[1]))
        rec.array(['Jim0.0', 'Jason1.0', 'John2.0'], dtype=<U8)
        """
        vfunc = np.vectorize(pyfunc, otypes=otypes, doc=doc, excluded=excluded, cache=cache, signature=signature)
        ra = self.as_recordarray()
        result=vfunc(ra, *args)
        return result

    # -------------------------------------------------------------------
    def apply_rows_numba(self, *args, otype=None, myfunc="myfunc"):
        """
        Usage:
        -----
        Prints to screen an example numba signature for the apply function.
        You can then copy this example to build your own numba function.

        Inputs:
        ------
        Can pass in multiple test arguments.

        Examples
        --------
        >>> ds = rt.Dataset({'a':rt.arange(10), 'b': rt.arange(10)*2, 'c': rt.arange(10)*3})
        >>> ds.apply_rows_numba()
        Copy the code snippet below and rename myfunc
        ---------------------------------------------
        import numba
        @numba.jit
        def myfunc(data_out, a, b, c):
            for i in range(len(a)):
                data_out[i]=a[i]   #<-- put your code here
        <BLANKLINE>
        ---------------------------------------------
        Then call
        data_out = rt.empty_like(ds.a)
        myfunc(data_out, ds.a, ds.b, ds.c)

        >>> import numba
        >>> @numba.jit
        ... def myfunc(data_out, a, b, c):
        ...     for i in range(len(a)):
        ...         data_out[i]=a[i]+b[i]+c[i]
        >>> data_out = rt.empty_like(ds.a)
        >>> myfunc(data_out, ds.a, ds.b, ds.c)
        >>> ds.data_out=data_out
        >>> ds
        #   a    b    c   data_out
        -   -   --   --   --------
        0   0    0    0          0
        1   1    2    3          6
        2   2    4    6         12
        """

        preamble = "import numba\n@numba.jit\n"

        list_inputs = ""
        list_inputs_tostring = ""
        firstinput = None
        for c in self.keys():
            if len(list_inputs) > 0:
                list_inputs = list_inputs +  ', '
                list_inputs_tostring = list_inputs_tostring +  ', '
            else:
                firstinput = c
            list_inputs=list_inputs + c

            if self[c].dtype.char in ['U','S']:
                list_inputs_tostring=list_inputs_tostring + "ds." + c + ".numbastring"
            else:
                list_inputs_tostring=list_inputs_tostring + "ds." + c

        code=f"def {myfunc}(data_out, {list_inputs}):\n    for i in range(len({firstinput})):\n        data_out[i]={firstinput}[i]   #<-- put your code here\n"
        exec = preamble+code

        print("Copy the code snippet below and rename myfunc")
        print("---------------------------------------------")
        print(exec)
        print("---------------------------------------------")
        print(f"Then call ")
        print(f"data_out = rt.empty_like(ds.{firstinput})")
        print(f"{myfunc}(data_out, {list_inputs_tostring})")
        #return exec

    # -------------------------------------------------------------------
    def apply(self, funcs, *args, check_op: bool = True, **kwargs):
        """
        The apply method returns a Dataset the same size
        as the current dataset. The transform function is applied
        column-by-column. The transform function must:

        * Return an array that is the same size as the input array.
        * Not perform in-place operations on the input array. Arrays
          should be treated as immutable, and changes to an array may
          produce unexpected results.

        Parameters
        ----------
        funcs : callable or list of callable
            the function or list of functions applied to each column.
        check_op : bool
            Defaults to True.  Whether or not to check if dataset has its own version, like ``sum``.

        Returns
        -------
        Dataset or Multiset

        Examples
        --------
        >>> ds = rt.Dataset({'a': rt.arange(3), 'b': rt.arange(3.0).tile(7), 'c':['Jim','Jason','John']})
        >>> ds.apply(lambda x: x+1)
        #   a       b   c
        -   -   -----   ------
        0   1    1.00   Jim1
        1   2    8.00   Jason1
        2   3   15.00   John1

        In the example below sum is not possible for a string so it is removed.

        >>> ds.apply([rt.sum, rt.min, rt.max])
                   a                   b                  c
        #   Sum   Min   Max    Sum    Min     Max     Min    Max
        -   ---   ---   ---   -----   ----   -----   -----   ----
        0     3     0     2   21.00   0.00   14.00   Jason   John
        """

        if not isinstance(funcs, list):
            funcs = [funcs]

        if len(funcs)==0:
            raise ValueError("The second argument funcs must not be empty")

        for f in funcs:
            if not callable(f):
                raise TypeError(f"{f} is not callable. Could not be applied to dataset.")

        results = {}

        # loop over all the functions supplied
        # if more than one function supplied, we will return a multiset
        for f in funcs:
            ds = type(self)()
            dsname =f.__name__.capitalize()

            call_user_func = True

            if check_op:
                # check to see if dataset has its own version of the operation)
                try:
                    ds= getattr(self, f.__name__)()
                    call_user_func = False
                except:
                    pass

            if call_user_func:
                # the dataset does not have its own version
                # call the user supplied function
                for colname, array in self.items():
                    ds[colname] = f(array, *args, **kwargs)

            results[dsname]=ds

        if len(funcs)==1:
            return ds
        else:
            return TypeRegister.Multiset(results)

    # -------------------------------------------------------------------
    @classmethod
    def from_tagged_rows(cls, rows_iter):
        """
        Create a Dataset from an iterable of 'rows', each to be a dict, Struct, or named_tuple of
        scalar values.

        Parameters
        ----------
        rows_iter : iterable of dict, Struct or named_tuple of scalars

        Returns
        -------
        Dataset
            A new Dataset.

        Notes
        -----
        Still TODO: Handle case w/ not all rows having same keys. This is waiting on SafeArray
        and there are stop-gaps to use until that point.

        Examples
        --------
        >>> ds1 = rt.Dataset.from_tagged_rows([{'a': 1, 'b': 11}, {'a': 2, 'b': 12}])
        >>> ds2 = rt.Dataset({'a': [1, 2], 'b': [11, 12]})
        >>> (ds1 == ds2).all(axis=None)
        True
        """
        keys = Counter()
        rows = []
        n_have_getitem = 0
        for row in rows_iter:
            if isinstance(row, tuple) and hasattr(row, '_fields'):  # proxy for a namedtuple
                keys.update(row._fields)
                row = row._asdict()
            elif isinstance(row, (Struct, dict)):
                keys.update(row.keys())
            else:
                raise TypeError(f'{cls.__name__}.from_tagged_rows: input must be iterable of dict or Struct.')
            n_have_getitem += hasattr(row, '__getitem__')
            rows.append(row)
        if len(rows) == 0 or len(keys) == 0:
            return cls({})
        if len(set(keys.values())) != 1:
            raise NotImplementedError(f'{cls.__name__}.from_tagged_rows(): All rows must have same keys.')
        retval = {_k: [] for _k in sorted(keys)}  # no reason to priv. the key order of any one row
        if n_have_getitem == 0:
            for row in rows:
                for _k in row:
                    retval[_k].append(getattr(row, _k))
        elif n_have_getitem == len(rows):
            for row in rows:
                for _k in row:
                    retval[_k].append(row[_k])
        else:
            for row in rows:
                for _k in row:
                    retval[_k].append(row[_k] if hasattr(row, '__getitem__') else getattr(row, _k))
        return cls(retval)

    @classmethod
    def from_rows(cls, rows_iter, column_names):
        """
        Create a Dataset from an iterable of 'rows', each to be an iterable of scalar values,
        all having the same length, that being the length of column_names.

        Parameters
        ----------
        rows_iter : iterable of iterable of scalars
        column_names : list of str
            list of column names matching length of each row

        Returns
        -------
        Dataset
            A new Dataset

        Examples
        --------
        >>> ds1 = rt.Dataset.from_rows([[1, 11], [2, 12]], ['a', 'b'])
        >>> ds2 = rt.Dataset({'a': [1, 2], 'b': [11, 12]})
        >>> (ds1 == ds2).all(axis=None)
        True
        """
        ncols = len(column_names)
        if ncols == 0:
            return cls({})
        cols = [[] for _k in column_names]
        for row in rows_iter:
            if isinstance(row, (dict, Struct, Dataset)):  # other dict types?
                raise TypeError(f'{cls.__name__}.from_rows: rows can not be "dictionaries".')
            if len(row) != ncols:
                raise ValueError(f'{cls.__name__}.from_rows: all rows must have same length as column_names.')
            for _i, _e in enumerate(row):
                cols[_i].append(_e)
        return cls(dict(zip(column_names, cols)))

    @classmethod
    def from_jagged_rows(cls, rows, column_name_base='C', fill_value=None):
        """
        Returns a Dataset from rows of different lengths. All columns in Dataset will be bytes or unicode. Bytes will be used if possible.

        Parameters
        ----------
        rows
            list of numpy arrays, lists, scalars, or anything that can be turned into a numpy array.
        column_name_base : str
            columns will by default be numbered. this is an optional prefix which defaults to 'C'.
        fill_value : str, optional
            custom fill value for missing cells. will default to the invalid string

        Notes
        -----
        *performance warning*: this routine iterates over rows in non-contiguous memory to fill in final column values.
        TODO: maybe build all final columns in the same array and fill in a snake-like manner like Accum2.
        """

        # get final dataset dims, flip all input to array
        nrows = len(rows)

        # always favor bytestrings
        dt = 'S'
        for i, r in enumerate(rows):
            # re-expand categoricals
            # note: multikey categorical expands to a tuple of arrays
            # previously raised an error on expand array
            if TypeRegister.is_binned_array(r):
                r = r.expand_array

            # possibly flip all arrays/lists/scalars to string arrays
            flip_to_fa = False
            if not isinstance(r, np.ndarray):
                flip_to_fa = True
            elif r.dtype.char not in 'US':
                flip_to_fa = True
            if flip_to_fa:
                r = TypeRegister.FastArray(r, dtype='S')
            rows[i] = r

            # final dtype will be unicode
            if rows[i].dtype.char == 'U':
                dt = 'U'

        ncols = len(max(rows, key=len))
        # get the string itemsize so the max string fits
        width = max(rows, key= lambda x: x.itemsize).itemsize

        # set fill value
        if fill_value is not None:
            # match to dtype
            if isinstance(fill_value, str):
                if dt == 'S':
                    inv = fill_value.encode()
            elif isinstance(fill_value, bytes):
                if dt == 'U':
                    inv = fill_value.decode()
            else:
                inv = str(fill_value)
        else:
            # use default
            inv = INVALID_DICT[np.dtype(dt).num]

        # make sure final array itemsize can fit all strings
        if dt == 'U':
            width /= 4
        final_dt = dt+str(width)

        # build final dict, column by column
        # this is slow for larger data because it has to loop over rows
        final = {}
        for i in range(ncols):
            col = empty(nrows, dtype=final_dt)
            for j, r in enumerate(rows):
                # if there are no more items in the column, fill with invalid
                if i >= len(r):
                    fill = inv
                else:
                    fill = rows[j][i]
                col[j] = fill
            # column name will be a number
            final[column_name_base+str(i)]=col

        return cls(final)

    @classmethod
    def from_jagged_dict(cls, dct, fill_value=None, stacked=False):
        """
        Creates a Dataset from a dict where each key represents a column name base and each value
        an iterable of 'rows'. Each row in the values iterable is, in turn, a scalar or an
        iterable of scalar values having variable length.

        Parameters
        ----------
        dct
            a dictionary of columns that are to be formed into rows
        fill_value
            value to fill missing values with, or if None, with the NODATA value
            of the type of the first value from the first row with values for the given key
        stacked : bool
            Whether to create stacked rows in the output when an input row
            in one of the input values objects contains an iterable.

        Returns
        -------
        Dataset
            A new Dataset.

        Notes
        -----
        For a given key, if each row in the corresponding values iterable is a scalar, a
        single column will be created with a column name equal to the key name.

        If for a given key, a row in the corresponding values iterable is an iterable, the
        behavior is determined by the stacked parameter.

        If stacked is False (the default), as many columns will be created as necessary to
        contain the maximum number of scalar values in the value rows. The column names will
        be the key name plus a zero based index. Any empty elements in a row will be filled with
        the specified fill_value, or if None, with a NODATA value of the type corresponding to the
        first value from the first row with values for the given key.

        If stacked is True, one column will be created for each input key, and for each row
        of input values, a row will be created in the output for every combination of
        value elements from each column in the input row.

        Examples
        --------
        >>> d = {'name': ['bob', 'mary', 'sue', 'john'],
        ...     'letters': [['A', 'B', 'C'], ['D'], ['E', 'F', 'G'], 'H']}
        >>> ds1 = rt.Dataset.from_jagged_dict(d)
        >>> nd = rt.INVALID_DICT[np.dtype(str).num]
        >>> ds2 = rt.Dataset({'name': ['bob', 'mary', 'sue', 'john'],
        ...     'letters0': ['A','D','E','H'], 'letters1': ['B',nd,'F',nd],
        ...     'letters2': ['C',nd,'G',nd]})
        >>> (ds1 == ds2).all(axis=None)
        True

        >>> ds3 = rt.Dataset.from_jagged_dict(d, stacked=True)
        >>> ds4 = rt.Dataset({'name': ['bob', 'bob', 'bob', 'mary', 'sue', 'sue', 'sue', 'john'],
        ...     'letters': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']})
        >>> (ds3 == ds4).all(axis=None)
        True
        """
        # Determine how many input rows and assure all columns conform
        num_in_rows = 0
        for k, v in dct.items():
            if num_in_rows == 0:
                num_in_rows = len(v)
            else:
                if len(v) != num_in_rows:
                    raise ValueError(f'{cls.__name__}.from_jagged_ rows: all values must ' +
                                     'have same length.')

        # If not stacked, concatenate columns constructed from each key/value
        if not stacked:
            ds = cls()
            for k, v in dct.items():
                ids = Dataset.from_jagged_rows(v, column_name_base=k, fill_value=fill_value)
                for ik in ids.keys():
                    ds[ik] = ids[ik]
            return ds

        # If stacked
        else:
            # Determine total number of output rows
            num_rows_ar = np.ones(num_in_rows, dtype=np.int64)
            for vals in dct.values():
                for i, r in enumerate(vals):
                    num_rows_ar[i] *= len(r) if is_list_like(r) else 1
            num_rows = num_rows_ar.sum()

            # Determine the type of each output column by creating arrays
            # (necessary to run through full, flattened list to get max string size)
            type_cols = []
            for vals in dct.values():
                type_cols.append(np.array([item for sublist in vals for item in
                              (sublist if is_list_like(sublist) else [sublist])]))

            # Allocate the output columns, as necessary
            cols = [0] * len(type_cols)
            col_done = [0] * len(type_cols)
            for j, type_col in enumerate(type_cols):
                (cols[j], col_done[j]) = (type_col, True) if len(type_col) == num_rows\
                    else (np.zeros(num_rows, type_col.dtype), False)

            # Fill the output columns, as necessary
            column_names = list(dct.keys())
            out_row_num = 0
            for in_row_num in range(num_in_rows):
                num_repeats = 1
                num_out_rows = num_rows_ar[in_row_num]
                for j, vals in enumerate(dct.values()):
                    if col_done[j]:
                        continue
                    val = vals[in_row_num]
                    if not is_list_like(val):
                        val = [val]
                    num_tiles = int(num_out_rows/(num_repeats*len(val)))
                    col_row_num = out_row_num
                    for tile_num in range(num_tiles):
                        for v in val:
                            for repeat_num in range(num_repeats):
                                cols[j][col_row_num] = v
                                col_row_num += 1
                    num_repeats *= len(val)
                out_row_num += num_out_rows

            return cls(dict(zip(column_names, cols)))

    # -------------------------------------------------------
    def trim(
        self,
        func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        zeros: bool = True,
        nans: bool = True,
        rows: bool = True,
        keep: bool = False,
        ret_filters: bool = False
    ) -> Union['Dataset', Tuple['Dataset', np.ndarray, np.ndarray]]:
        """
        Returns a Dataset with columns removed that contain all zeros or all nans (or either).

        If `rows` is True (the default), any rows which are all zeros or all nans will also be removed.
        If `func` is set, it will bypass the zeros and nan check and instead call `func`.
          - any column that contains all True after calling `func` will be removed.
          - any row that contains all True after calling `func` will be removed if `rows` is True.

        Parameters
        ----------
        func
            A function which inputs an array and returns a boolean mask.
        zeros : bool
            Defaults to True. Values must be non-zero.
        nans : bool
            Defaults to True. Values cannot be nan.
        rows : bool
            Defaults to True. Reduce rows also if entire row filtered.
        keep : bool
            Defaults to False.  When set to True, does the opposite.
        ret_filters : bool
            If True, return row and column filters based on the comparisons

        Returns
        -------
        Dataset or (Dataset, row_filter, col_filter)

        Example
        -------
        >>> ds = rt.Dataset({'a': rt.arange(3), 'b': rt.arange(3.0)})
        >>> ds.trim()
        #   a      b
        -   -   ----
        0   1   1.00
        1   2   2.00

        >>> ds.trim(lambda x: x > 1)
        #   a      b
        -   -   ----
        0   0   0.00
        1   1   1.00

        >>> ds.trim(isfinite)
        Dataset is empty (has no rows).
        """
        def iszero(arr):
            return arr == 0


        # Remove columns that don't pass
        col_filter = []
        col_filter_mask = []

        if func is None:
            if zeros and nans:
                func = isnanorzero
            elif zeros:
                func = iszero
            elif nans:
                func = isnan
            else:
                raise ValueError("func must be set, or zeros or nans must be true")

        labels = self.label_get_names()

        colboolmask  = np.zeros(self._ncols, dtype='?')

        # loop through all computable columns
        for i, (col, arr) in enumerate(self.items()):
            if col not in labels and arr.iscomputable():
                result=func(arr)
                if result.dtype.num ==0:
                    if keep:
                        # check if all FALSE
                        addcol = sum(result) != 0
                    else:
                        # check if all TRUE
                        #print('**col ', col, sum(result), len(arr))
                        addcol = sum(result) != len(arr)

                    if addcol:
                        col_filter_mask.append(result)
                        col_filter.append(col)
                        colboolmask[i]=True
                else:
                    #add because did not return bool
                    col_filter.append(col)
                    colboolmask[i]=True

            else:
                #add non-computable
                col_filter.append(col)
                colboolmask[i]=True

        # check for empty dataset?
        rowmask = None
        if rows:
            for arr in col_filter_mask:
                if rowmask is None:
                    # first one, just set the value
                    rowmask = arr
                else:
                    # timed, didn't seem to make much difference
                    #if keep:  rowmask = mask_ori(col_filter_mask)
                    #else:  rowmask = mask_andi(col_filter_mask)

                    # inplace OR on boolean mask
                    if keep:
                        rowmask += arr
                    else:
                        # inplace AND on boolean mask
                        # print('**and', col, sum(arr), sum(rowmask))
                        rowmask *= arr


        # remove rows that are all true
        applyrowmask = None
        if rowmask is not None:
            if keep:
                # check if anything to filter on
                if sum(rowmask) != len(rowmask):
                    #reduce all the rows
                    applyrowmask = rowmask
            else:
                # check if anything to negatively filter on
                #print('**col', col, sum(rowmask))
                if sum(rowmask) != 0:
                    #reduce all the rows
                    applyrowmask = ~rowmask

        # remove cols that are not in list
        # remove rows that are all False
        if applyrowmask is not None:
            newds=self[applyrowmask, col_filter]
        else:
            newds = self[col_filter]

        # If we had summary, we need to apply the col_filter
        # and recalculate the totals

        if ret_filters:
            return (newds, applyrowmask, col_filter)
        else:
            return newds

    # -------------------------------------------------------
    def keep(self, func, rows:bool= True):
        """
        `func` must be set.  Examples of `func` include ``isfinite``, ``isnan``, ``lambda x: x==0``
          - any column that contains all False after calling `func` will be removed.
          - any row that contains all False after calling `func` will be removed if `rows` is True.

        Parameters
        ----------
        func : callable
            A function which accepts an array and returns a boolean mask of the same shape as the input.
        rows : bool
            If `rows` is True (the default), any rows which are all zeros or all nans will also be removed.

        Returns
        -------
        Dataset

        Example
        -------
        >>> ds = rt.Dataset({'a': rt.arange(3), 'b': rt.arange(3.0)})
        >>> ds.keep(lambda x: x > 1)
        #   a      b
        -   -   ----
        2   2   2.00

        >>> ds.keep(rt.isfinite)
        #   a      b
        -   -   ----
        0   0   0.00
        1   1   1.00
        2   2   2.00
        """
        return self.trim(func=func, rows=rows, keep=True)

    # -------------------------------------------------------
    def pivot(
        self, labels=None, columns=None, values=None, ordered: bool = True, lex: Optional[bool] = None, filter=None
    ) -> Union['Dataset', 'Multiset']:
        """
        Return reshaped Dataset or Multiset organized by labels / column values.

        Uses unique values from specified `labels` / `columns` to form axes of the
        resulting Dataset. This function does not support data aggregation,
        multiple values will result in a Multiset in the columns.

        Parameters
        ----------
        labels : str or list of str, optional
            Column to use to make new labels. If None, uses existing labels.
        columns : str
            Column to use to make new columns.
        values : str or list of str, optional
            Column(s) to use for populating new values. If not
            specified, all remaining columns will be used and the result will
            have a Multiset.
        ordered: bool, defaults to True
        lex: bool, defaults to None
        filter: ndarray of bool, optional

        Returns
        -------
        Dataset or Multiset

        Raises
        ------
        ValueError:
            When there are any `labels`, `columns` combinations with multiple values.

        Examples
        --------
        >>> ds = rt.Dataset({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
        ...                  'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
        ...                  'baz': [1, 2, 3, 4, 5, 6],
        ...                  'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
        >>> ds
        #   foo   bar   baz   zoo
        -   ---   ---   ---   ---
        0   one   A       1   x
        1   one   B       2   y
        2   one   C       3   z
        3   two   A       4   q
        4   two   B       5   w
        5   two   C       6   t

        >>> ds.pivot(labels='foo', columns='bar', values='baz')
        foo   A   B   C
        ---  --  --  --
        one   1   2   3
        two   4   5   6
        """
        if labels is None:
            # see if existing labels exist
            labels = self.labels_get_names()
        elif np.isscalar(labels):
            labels=[labels]

        if not isinstance(labels, list) or len(labels) ==0:
            raise ValueError('The parameter "labels" must exist and be passed as a string or list of strings.')

        if columns is None or not isinstance(columns, (str, list)):
            raise ValueError('The parameter "columns" must exist and be passed as a string or list of strings.')
        if np.isscalar(columns):
            columns = [columns]
        if not isinstance(columns, list) or len(columns) ==0:
            raise ValueError('The parameter "columns" must exist and be passed as a list of one or more strings.')

        if values is None:
            values = []
            allkeys=labels+columns
            for colname in self.keys():
                if colname not in allkeys:
                    values.append(colname)

        elif np.isscalar(values):
            values=[values]

        if not isinstance(values, list) or len(values) ==0:
            raise ValueError(f'The parameter "values" could not be used {values!r}.')

        # build similar to Accum2
        grows = self.cat(labels, ordered=ordered, lex=lex).grouping
        gcols = self.cat(columns, ordered=ordered, lex=lex).grouping
        g = combine2groups(grows, gcols, filter=filter)

        # need ifirstkey to pull from original into matrix
        ifirstkey = g.ifirstkey

        # make labels
        crd=grows.uniquedict
        ccd=gcols.uniquedict

        # make a dataset with the cat_rows as labels
        ds_crd = Dataset(crd)
        ds_crd.label_set_names(labels)

        # +1 to include the filter (0 bin) since used combine2groups
        row_len=len(ds_crd)+1

        # check for duplicates
        ncountgroup = g.ncountgroup
        pos = ncountgroup.argmax()
        if ncountgroup[pos] > 1:
            # find out where a duplicate is
            raise ValueError(f'Duplicates exist, cannot reshape. Duplicate count is {ncountgroup[pos]}.  Pos is {pos!r}.')

        #=========================================
        # sub function to slice up original arrays
        def make_dataset(coldict, val, newds):
            # colnames must be unicode
            colnames = [colstr.astype('U') for colstr in coldict.values()]
            innerloop = len(colnames)
            outerloop= len(colnames[0])

            # if this is multikey columns (if len(coldict) > 1) we may need to create a tuple of value pairings

            # pull into one long array
            arr_long = val[ifirstkey]
            start=row_len

            # this loops adds the colname + the value
            for i in range(0, outerloop):
                for j in range(0, innerloop):
                    if j==0:
                        c=colnames[j][i]
                    else:
                        # multikey name, insert underscore
                        c=c+'_'+colnames[j][i]

                # slice up the one long array
                newds[c] = arr_long[start:start + row_len -1]
                start = start + row_len
            return newds

        # if just 1, make a dataset, otherwise multiset
        ms= {}
        for colname in values:
            ds_ms=ds_crd.copy(False)
            val = self[colname]

            # make a dataset per values key passed in
            ms[colname] = make_dataset(ccd, val, ds_ms)

        if len(ms) == 1:
            # return the one dataset
            return ms.popitem()[1]

        ms = TypeRegister.Multiset(ms)

        # make sure labels on left are lifted up for multiset
        ms.label_set_names(labels)
        return ms

    # -------------------------------------------------------
    def equals(self, other, axis: Optional[int] = None, labels: bool = False, exact: bool = False):
        """
        Test whether two Datasets contain the same elements in each column.
        NaNs in the same location are considered equal.

        Parameters
        ----------
        other : Dataset or dict
            another dataset or dict to compare to
        axis : int, optional
            * None: returns a True or False for all columns
            * 0 : to return a boolean result per column
            * 1 : to return an array of booleans per column
        labels : bool
            Indicates whether or not to include column labels in the comparison.
        exact : bool
            When True, the exact order of all columns (including labels) must match

        Returns
        -------
        bool or Dataset
            Based on the value of `axis`, a boolean or Dataset containing the equality comparison results.

        See Also
        --------
        Dataset.crc, ==, >=, <=, >,  <

        Examples
        --------
        >>> ds = rt.Dataset({'somenans': [0., 1., 2., nan, 4., 5.]})
        >>> ds2 = rt.Dataset({'somenans': [0., 1., nan, 3., 4., 5.]})
        >>> ds.equals(ds)
        True

        >>> ds.equals(ds2, axis=0)
        #   somenans
        -   --------
        0      False

        >>> ds.equals(ds, axis=0)
        #   somenans
        -   --------
        0       True

        >>> ds.equals(ds2, axis=1)
        #   somenans
        -   --------
        0       True
        1       True
        2      False
        3      False
        4       True
        5       True

        >>> ds.equals(ds2, axis=0, exact=True)
        FastArray([False])

        >>> ds.equals(ds, axis=0, exact=True)
        FastArray([True])

        >>> ds.equals(ds2, axis=1, exact=True)
        FastArray([[ True],
                   [ True],
                   [False],
                   [False],
                   [ True],
                   [ True]])
        """
        if not isinstance(other, Dataset):
            try:
                # try to make it a dataset
                other = Dataset(other)
            except:
                other = False

        # check if all the nans are in the same place
        def ds_isnan(ds):
            # call isnan in the order
            result = []
            for v in ds.values():
                try:
                    if v.dtype.char not in 'SU':
                        result.append(v.isnan())
                    else:
                        # if it has no nan, then no nans
                        result.append(np.zeros(v.shape, '?'))
                except Exception:
                    # if it has no nan, then no nans
                    result.append(np.zeros(v.shape, '?'))
            return vstack(result, order='F')

        if exact:
            try:
                # create a nan mask -- where both are nans
                # this does an inplace and
                result = ds_isnan(self)
                result *= ds_isnan(other)

                # now make the comparions, the column order must be the same (names are ignored)
                result2=[v1 == v2 for v1,v2 in zip(self.values(), other.values())]
                result |= vstack(result2, order='F')

            except Exception:
                # anything went wrong, assume nothing matches
                result = False
                if axis != 1:
                    result=np.zeros(1, dtype='?')
            if axis != 1:
                result = np.all(result, axis=axis)

        else:
            try:
                result = self.apply_cols(isnan, labels=labels) & other.apply_cols(isnan, labels=labels)
                result |= (self == other)
            except:
                result = False
                if axis != 1:
                    result=np.zeros(1, dtype='?')

            if axis != 1:
                result = result.all(axis=axis)

        return result

_RIPTABLE_TO_PANDAS_TZ = {
    'UTC': 'UTC',
    'NYC': 'US/Eastern',
    'DUBLIN': 'Europe/Dublin',
    'GMT': 'GMT'
}
_PANDAS_TO_RIPTABLE_TZ = dict([(v, k) for (k, v) in _RIPTABLE_TO_PANDAS_TZ.items()])

# keep this as the last line
from .rt_enum import TypeRegister

TypeRegister.Dataset = Dataset
