__all__ = ['Struct', ]

import keyword
import warnings
import os
import sys
import logging
import json
import itertools
from typing import TYPE_CHECKING, List, Mapping, Optional, Sequence, Set, Tuple, Union
from collections import OrderedDict
from re import IGNORECASE, compile

import numpy as np
import riptide_cpp as rc

#from IPython import get_ipython
#from IPython.display import display, HTML

from .rt_utils import bytes_to_str, is_list_like, h5io_to_struct
from .rt_sds  import save_sds, _load_sds, _multistack_categoricals, _rebuild_rootfile
from .rt_timers import GetTSC
from .rt_enum import DS_DISPLAY_TYPES, DisplayDetectModes, TypeRegister, NumpyCharTypes, TypeId, CategoryMode, INVALID_FILE_CHARS, SDSFlag, DisplayColumnColors, ColHeader, ColumnStyle
from .rt_hstack import hstack_any
from .Utils.rt_display_nested import DisplayNested
from .Utils.rt_metadata import MetaData
from .rt_misc import build_header_tuples
from .rt_itemcontainer import ItemContainer, ATTRIBUTE_MARGIN_COLUMN, ATTRIBUTE_NUMBER_OF_FOOTER_ROWS
from .rt_numpy import mask_ori, mask_andi, arange, hstack
from .Utils.rt_display_properties import get_array_formatter
from .rt_display import DisplayTable, DisplayDetect, DisplayString

# Type-checking-only imports.
if TYPE_CHECKING:
    # py36 doesn't have re.Pattern so we can't do a normal import of it above.
    # Work around it by using the 'forward reference' style annotation below.
    import re


# Create a logger for this module.
# TODO: Maybe just put this inside Struct as e.g. cls._logger?
logger = logging.getLogger(__name__)


class _DefaultGetAttrPlaceholder:
    pass


_default_getattr_placeholder = _DefaultGetAttrPlaceholder()


class Struct:
    """
    The Struct class is at the root of much of the riptable class design; both Dataset and Multiset
    inherit from Struct.

    Struct represents a collection of (mixed-type) data members, with standard attribute get/set
    behavior, as well as dictionary-style retrieval.

    The Struct constructor takes a dictionary (dict, OrderedDict, etc...) as its required argument.
    When :attr:`Struct.UseFastArray` is True (the default), any numpy arrays among the dictionary values
    will be cast into FastArray.  Struct() := Struct({}).

    The constructor dictionary keys (or element/column names added later) must not conflict with any
    Struct member names. Additionally, if :attr:`Struct.AllowAnyName` is False (it is True by default), a
    column name must be a legal Python variable name, not starting with '_'.

    Parameters
    ----------
    dictionary : dict
        A dictionary of named objects.

    Examples
    --------
    >>> st = rt.Struct({'a': 1, 'b': 'fish', 'c': [5.6, 7.8], 'd': {'A': 'david', 'B': 'matthew'},
    ... 'e': np.ones(7, dtype=np.int64)})
    >>> print(st)
    #   Name   Type    Size   0      1     2
    -   ----   -----   ----   ----   ---   -
    0   a      int     0      1
    1   b      str     0      fish
    2   c      list    2      5.6    7.8
    3   d      dict    2      A      B
    4   e      int64   7      1      1     1
    >>> st.a
    1
    >>> st['a']
    1
    >>> print(st[3:])
    #   Name   Type    Rows   0   1   2
    -   ----   -----   ----   -   -   -
    0   d      dict    2      A   B
    1   e      int64   7      1   1   1
    >>> st.newcol = 5  # okay, a new entry
    >>> st.newcol = [5, 7]  # okay, replace the entry
    >>> st['another'] = 6  # also works
    >>> st['newcol'] = 6  # and this works as well

    **Indexing behavior**

    >>> st['b'] # get a 'column' (equiv. st.b)
    >>> st[['a', 'e']] # get some columns
    >>> st[[0, 4]] # get some columns (order is that of iterating st (== list(st))
    >>> st[1:5:2] # standard slice notation, indexing corresponding to previous
    >>> st[bool_vector] # get 'True' columns

    **Equivalents**

    >>> assert len(st) == st.get_ncols()
    >>> for _k in st: print(_k, st[_k])
    >>> for _k, _v in st.items(): print(_k, _v)
    >>> for _k, _v in zip(st.keys(), st.values()): print(_k, _v)
    >>> for _k, _v in zip(st, st.values()): print(_k, _v)
    >>> if key in st:
    ...     assert getattr(st, key) is st[key]

    **Context manager**

    >>> with Struct({'a': 1, 'b': 'fish'}) as st:
    ...     st.a)
    >>> assert not hasattr(st, 'a')
    """

    UseFastArray = True
    """bool: True if np.ndarray is flipped to FastArray on init."""

    AllowAnyName = True
    """bool: True if any name for a column name is permitted, but will be renamed."""

    AllNames = False
    """bool: True if any name for a column name is permitted without renaming."""

    WarnOnInvalidNames = False
    """bool: True if a warning is generated on invalid names."""

    _summary_len = 3
    _restricted_names = {}

    # track repr calls now also
    _lastrepr = 0
    _lastreprhtml = 0

    # global Struct dict will allow us to make certain variables immutable if we want

    # All subclassing inherits these members
    # ------------------------------------------------------------
    def _pre_init(self):
        self._all_items = ItemContainer()
        self._col_sortlist = None
        self._sort_display = False
        self._sort_ascending = True
        self._transpose_on = False
        self._natural_sort = None
        self._badcols = None
        self._badrows = None

        # upon creation create a unique id to track sorting
        self._uniqueid = GetTSC()

    # ------------------------------------------------------------
    def _copy_base(self, from_Struct):
        """
        This copies the underlying special variables but
        does not copy _all_items or _uniqueid or any of the 'columns'.

        :param from_Struct: the Struct being copied
        :return: is_locked() (must unlock/relock around rest of copy)
        """
        self._col_sortlist = from_Struct._col_sortlist
        self._sort_display = from_Struct._sort_display
        self._transpose_on = from_Struct._transpose_on
        self._natural_sort = from_Struct._natural_sort

        if from_Struct.is_locked():
            self._locked = True
            return True
        return False

    # ------------------------------------------------------------
    def _post_init(self):
        """
        Call self._run_once() to cleanup or init anything else, override _run_once()
        in subclasses if needed.
        :return: None
        """
        self._run_once()
        self._ncols = self.col_get_len()

    # ------------------------------------------------------------
    def is_valid_colname(self, name):
        """
        Checks if item's name is a valid column name.
        Restricted names include python keywords and class method names.

        Parameters
        ----------
        name: String
            to be checked for validity as a Struct 'column' name.

        Returns
        -------
        bool
            True of `name` is valid, otherwise False.

        See Also
        --------
        Struct.get_restricted_names
        """
        if len(name) < 1: return False

        # check for str or bytes and not leading underscore
        if isinstance(name, str):
            if name[0]=='_': return False
        elif isinstance(name, bytes):
            if name[0]==b'_': return False
        else:
            return False

        if Struct.AllNames:
            return True
        elif Struct.AllowAnyName:
            return name not in self.get_restricted_names()
        else:
            return name.isidentifier() and name not in self.get_restricted_names()

    # ------------------------------------------------------------
    def get_restricted_names(self):
        """
        Returns list of tokens which are ineligible for use as 'column' names;
        for example, python keywords and class method names.
        This method only generates the result once. Afterwards, it is stored as a class variable.

        Returns
        -------
        obj:`set`
            All invalid name tokens.

        See Also
        --------
        Struct.is_valid_colname

        """
        try:
            rnames = self.__class__._restricted_names[self.__class__.__name__]
        except KeyError:
            rnames = self.__class__._restricted_names[self.__class__.__name__] = set(keyword.kwlist)
            rnames.update(dir(self.__class__))
        return rnames

    # ------------------------------------------------------------
    def _validate_names(self, names):
        invalid = []
        seen = set()
        # note: this code must remain high performance since called when
        # initializing datasets and structs
        for _nm in names:
            seen.add(_nm)
            if not Struct.AllNames:
                try:
                    if _nm.startswith('_'):
                        invalid.append(_nm)
                except:
                    if _nm.startswith(b'_'):
                        invalid.append(_nm)

        if len(invalid) > 0:
            if self.__class__.WarnOnInvalidNames:
                warnings.warn(
                    f'Invalid column name(s) {invalid} --- please rename the column(s) or the class will lose this method.')
            else:
                raise ValueError('Invalid column names passed: {}'.format(', '.join(invalid)))
        if len(names) > len(seen):
            raise ValueError('Duplicate column names passed.')

    # ------------------------------------------------------------
    def _escape_invalid_file_chars(self, name):
        '''
        Certain characters will cause problems in item names if a Struct needs to name an
        SDS file.
        ('\\', ':', '<', '>', '!', '|', '*', '?')

        '''
        if not isinstance(name,str):
            name = name.decode('utf-8')
        replace_list = []
        for char in INVALID_FILE_CHARS:
            if char in name:
                replace_list.append(char)

        for c in replace_list:
            name = name.replace(c,'_')

        return name

    # ------------------------------------------------------------
    def _init_from_dict(self, dictionary):
        if isinstance(dictionary, Struct):
            # no need to validate columns if initializing from struct or dataset
            # dataset has a different init_from_dict because it needs to check column length
            self._all_items = dictionary._all_items.copy()
            return
        if not isinstance(dictionary, dict):
            raise TypeError(f'Unexpected type passed to Struct ctor: {type(dictionary).__name__}')
        self._validate_names(dictionary)

        allnames = Struct.AllNames
        for k, v in dictionary.items():
            if allnames or k[0] != '_':
                # remember the original attributes in case we want to freeze later
                if self.UseFastArray and isinstance(v, np.ndarray):
                    # flip value to FastArray
                    if not isinstance(v, TypeRegister.FastArray):
                        v = v.view(TypeRegister.FastArray)
                # add the item to this class
                self.col_set_value(k, v)

    # ------------------------------------------------------------
    def __init__(self, dictionary={}):
        # keep track of all original names
        self._pre_init()
        self._init_from_dict(dictionary)
        self._post_init()

    # prepare for with statement ---------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # This is a form of destructor, even though the object remains in the containing
        # namespace (we sigh for C++-style scoping), so it is perfectly kosher to force
        # unlocking of the object (if nec.) before we delete all of its contents.
        self._unlock()
        self.col_remove(list(self.keys()))

    # routines to override in a subclass ----------------------------
    def _superadditem(self, name, value):
        # the item will be added to the class dict
        try:
            hasunderscore= name.startswith('_')
        except:
            # assume it failed because bytes passed
            name=name.decode('utf-8')
            hasunderscore= name.startswith('_')

        if hasunderscore:
            # an underscore means it cannot be part of struct/dataset
            super().__setattr__(name, value)
        else:
            self.col_set_value(name, value)

    # ------------------------------------------------------------
    def _check_addtype(self, name, value):
        '''
        override to check types
        '''
        return value

    # ------------------------------------------------------------
    def _replaceitem_allnames(self, name, value):
        # dataset specific
        # override to check replacing
        if name in self:
            value = self._check_addtype(name, value)
        if value is not None:
            # if column is part of active sort, sort is no longer valid
            self._update_sort(name)
            self.col_set_value(name, value)

    # ------------------------------------------------------------
    def _addnewitem_allnames(self, name, value):
        # check if in existing dictionary
        value = self._check_addtype(name, value)
        if value is not None:
            self.col_set_value(name, value)
            ncols = self._ncols + 1
            self._superadditem('_ncols', ncols)

    # ------------------------------------------------------------
    def _replaceitem(self, name, value):
        # dataset specific
        # override to check replacing
        if name in self:
            value = self._check_addtype(name, value)
        if value is not None:
            # if column is part of active sort, sort is no longer valid
            self._update_sort(name)

            self._superadditem(name, value)

    # ------------------------------------------------------------
    def _addnewitem(self, name, value):
        # check if in existing dictionary
        value = self._check_addtype(name, value)
        self._superadditem(name, value)
        ncols = self._ncols + 1
        self._superadditem('_ncols', ncols)

    # ------------------------------------------------------------
    def _deleteitem(self, name):
        if self.col_exists(name):
            self._all_items.item_delete(name)
            self._ncols -= 1

            # if column is part of active sort, sort is no longer valid
            self._update_sort(name)

            if self._ncols <= 0:
                if hasattr(self, '_nrows'):
                    # print("Dataset is empty (has no rows).")
                    self.__setattr__('_nrows', None)
        else:
            raise IndexError(f'Unknown column {name!r} does not exist and cannot be deleted.')

    # ------------------------------------------------------------
    def _update_sort(self, name):
        '''
        Discard sort index if sortby item was removed or replaced.
        '''
        if self._col_sortlist is not None:
            if name in self._col_sortlist:
                self._col_sortlist = None

    # ------------------------------------------------------------
    def is_locked(self):
        """Returns True if object is locked (unable to add/remove/rename elements).

        NB: Currently behaves as does tuple: the contained data will still be
        mutable when possible.

        Returns
        -------
        bool
            True if object is locked
        """
        return hasattr(self, "_locked") and self._locked

    # ------------------------------------------------------------
    def _lock(self):
        ### do we want to set self.vector.flags.writable = False or self.dataset._lock()
        ### etc where these make sense for all of the contents?  it will not help for contained
        ### dictionaries, lists, etc...
        self._locked = True

    # ------------------------------------------------------------
    def _unlock(self):
        self._locked = False

    # ------------------------------------------------------------
    def __delattr__(self, name):
        # when an item is removed
        if self.is_locked():
            # locked and item is new
            raise AttributeError(f'Not allowed to delete item {name} in locked object')
        else:
            self._deleteitem(name)

    # ------------------------------------------------------------
    def __delitem__(self, name):
        self.__delattr__(name)

    # ------------------------------------------------------------
    def __setattr__(self, name, value):
        # when an item is replaced or added
        if name.startswith('_'):
            # internal or special item that we do not track
            # to add items with '_' set AllNames to True and user ['_someitem'] to add it
            # self._setattr(name,value)
            super().__setattr__(name, value)
            return

        elif self.is_locked():
            # another keyword/reserved word block needs to go here
            if hasattr(self, name):
                # locked but not a new item
                # NOTE: maybe we want to allow them to change?  Then uncomment and remove raise statement.
                # self._replaceitem(name, value)
                raise AttributeError(f'Not allowed to replace item {name} in locked object.')
            else:
                # locked and item is new
                raise AttributeError(f'Not allowed to create new item {name} in locked object.')

        elif self.__contains__(name):
            # Name already exists as an item in the Struct, so we know it's not a reserved name.
            # We can just overwrite it without having to do additional checks to determine if
            # we're going to overwrite a method, etc.
            self._replaceitem(name, value)

        else:  # never called _lock(), so go on
            # If this object has an attribute with the given name but it's _not_ in the ItemContainer,
            # it's likely we're going to overwrite a method and we don't allow that.
            obj = getattr(self, name, _default_getattr_placeholder)
            if callable(obj):
                # special protect certain names which would ruin class
                warnings.warn(f'The method {name} is readonly and cannot be assigned.')
                return AttributeError(f'The method {name} is readonly and cannot be assigned.')
            elif obj is _default_getattr_placeholder:
                self._addnewitem(name, value)
            else:
                # Log the name and type(obj) here -- there shouldn't really be anything hitting
                # this branch anymore, so if there is somehow it'd be good to have some data on
                # how we're getting here.
                logger.debug(
                    "Replacing non-item attribute '%s' in `__setattr__`.", name,
                    extra={'name': name, 'attr': obj}
                )
                self._replaceitem(name, value)

    # ------------------------------------------------------------
    def __getattr__(self, name):
        # check first if the name is in one of the dicts
        impossible = '____'
        inlocal = self.__dict__.get(name, impossible)
        if inlocal is impossible:
            inlocal = self.__class__.__dict__.get(name, impossible)
        if inlocal is impossible:
            try:
                # assume they are typing in a column name
                return self.col_get_value(name)
            except:
                pass
            return object.__getattribute__(self, name)

        return inlocal

        #if Struct.AllNames:
        #    # code optimization, try is faster
        #    try:
        #        return self.col_get_value(name)
        #    except:
        #        pass
        #    return object.__getattribute__(self, name)

        #else:
        #    # code optimization, try is faster
        #    try:
        #        return self.col_get_value(name)
        #    except:
        #        pass


    # ------------------------------------------------------------
    def _get_count_for_slice(self, idx, for_rows):
        count = self.get_nrows() if for_rows else self.get_ncols()
        if idx is None:
            nrows = count
        elif isinstance(idx, slice):
            nrows = len(range(count)[idx])
        else:
            nrows = len(idx)
        return nrows

    # ------------------------------------------------------------
    def _mask_get_item(self, idx, by_col_arg=True):
        """
        _mask_get_item applies a mask to a row or a column

        :param idx: the argument from the get/set-item [] brackets
        :param by_col_arg:  is this a column mask (instead of row mask)
        :return: list of actual indexes or None
        """
        mask = None
        # for error messages
        if by_col_arg:
            by = "column"
        else:
            if hasattr(self, '_nrows') and self._nrows is not None:
                by = "row"
            else:
                raise IndexError(f'{self.__class__.__name__} row index out of range (empty)')
        if isinstance(idx, slice):
            # slice other than [:]
            if not (idx.start is None and idx.stop is None and idx.step is None):
                # mask is now a slice
                mask = idx
            else:
                idx = None
        elif isinstance(idx, list):
            if len(idx) > 0:
                first_item = idx[0]
                if isinstance(first_item, bool): # a bool is an int so must test first
                    if by_col_arg:
                        mask = idx
                elif isinstance(first_item, (int, np.integer)):
                    mask = idx
                elif isinstance(first_item, (str, bytes)): # string-ish
                    if by_col_arg:
                        idx = [bytes_to_str(_i) for _i in idx]
                    else:
                        mask = self._index_from_row_labels(idx)
                        #raise TypeError(f'Error in {by} slice; rows cannot be string indexed.')
                elif isinstance(first_item, tuple):
                    if not by_col_arg:
                        mask = self._index_from_row_labels(idx)
                    else:
                        raise TypeError(f"Cannot index {by} with tuple.")
                else:
                    raise TypeError(
                        f'Error in {by} slice; lists cannot be {type(first_item).__name__} indexed.')
            else:
                raise IndexError(f'Error in {by} slice; empty list.')
        elif isinstance(idx, np.ndarray):
            if len(idx) > 0:
                dtype_char = idx.dtype.char
                # int based ----------------------
                if dtype_char in NumpyCharTypes.AllInteger:
                    mask = idx
                # bool based ---------------------
                elif dtype_char == '?':
                    if by_col_arg:
                        mask = idx
                # string based -------------------
                elif dtype_char in ('S', 'U'):
                    if by_col_arg:
                        if dtype_char == 'S':
                            idx = idx.astype('U')
                    else:
                        mask = self._index_from_row_labels(idx)
                # other ----------------------------
                else:
                    # let numpy report an error
                    mask = idx
                    raise TypeError(f"Error in {by} slice; numpy arrays cannot be {idx.dtype!r} indexed.")
        elif isinstance(idx, (str, bytes)):
            if not by_col_arg:
                mask = self._index_from_row_labels(idx)
        elif idx is None:
            raise TypeError('Cannot index by None.')
        elif isinstance(idx, (int, np.integer)): # single item, never true in current usage
            mask = slice(idx, idx + 1)
        else: # some other type?
            # row indexing allows tuples
            if by_col_arg:
                raise TypeError(
                    f'Error in {by} slice; {self.__class__.__name__} cannot be {type(idx).__name__} indexed.')
            else:
                mask = self._index_from_row_labels(idx)
        if mask is not None:
            # is the mask for col or for the row?
            if by_col_arg:
                # mask is for the column
                # make a numpy string array of all the column names, and then apply the mask to the numpy array
                data = np.array(list(self.keys()))
                return data[mask]
            else:
                # return the mask for the row
                return mask
        return idx
    # ------------------------------------------------------------
    def _index_from_row_labels(self, fld):
        '''
        Use this if row index was a string or tuple. Will only be applied to the Dataset's label columns (if it has any).
        '''

        if isinstance(fld, (str, bytes, tuple)):
            fld = [fld]
        labels = self.label_get_names()

        first_item = fld[0]
        if isinstance(first_item, tuple):
            num_cols = len(first_item)
        else:
            num_cols = 1

        if len(labels) != num_cols:
            raise IndexError(f"This structure has {len(labels)} label columns. Cannot use string row indexing with {len(fld)} {type(fld[0])}.")

        # maybe TODO: this multi-column matching loop happens in a couple other spots, is there a way to genericize it?
        idx_mask = []
        # for each key to match on
        for key in fld:
            # multikey
            if isinstance(key, tuple):
                tup_mask = []
                for i, k in enumerate(key):
                    col = getattr(self, labels[i])
                    mask = col == k
                    tup_mask.append(mask)
                # match needs to be on all columns
                tup_mask = mask_andi(tup_mask)
                idx_mask.append(tup_mask)
            # single column, match all to the first column
            else:
                col = getattr(self, labels[0])
                mask = col == key
                idx_mask.append(mask)

        # or all of the matching indices together
        idx_mask = mask_ori(idx_mask)

        return idx_mask

    # ------------------------------------------------------------
    def _extract_indexing(self, index):
        """
        Internal method common to get/set item.

        Parameters
        ----------
        index
            (rowspec, colspec) or colspec (=> rowspec of :)

        Returns
        -------
        col_idx
        row_idx
        ncols
        nrows
        row_arg
            NB Any column names will be converted to str (from bytes or np.str_).
        """
        if isinstance(index, tuple):  # general case: ds[r, c]
            if len(index) == 2:
                row_arg, col_arg = index
            else:
                raise IndexError(
                    f'Can only index {self.__class__.__name__} as ds[r, c], ds[r, :], ds[:, c] or ds[c].')

            if isinstance(row_arg, (int, np.integer)):
                row_idx = row_arg
                nrows = 1
            elif row_arg is None:
                raise TypeError('Cannot index rows with None.')
            else:
                row_idx = self._mask_get_item(row_arg, by_col_arg=False)
                nrows = self._get_count_for_slice(row_idx, True)
        else:  # provide syntactic sugar: ds[c] := ds[:, c]
            col_arg = index
            row_idx = row_arg = None
            nrows, ncols = self.shape

        if isinstance(col_arg, (str, bytes)):
            col_idx = bytes_to_str(col_arg)
            ncols = 1
        elif isinstance(col_arg, (int, np.integer)):
            col_idx = list(self.keys())[int(col_arg)]
            ncols = 1
        elif col_arg is None:
            raise TypeError('Cannot index cols with None.')
        else:
            col_idx = self._mask_get_item(col_arg, by_col_arg=True)
            ncols = self._get_count_for_slice(col_idx, False)

        return col_idx, row_idx, ncols, nrows, row_arg

    # ------------------------------------------------------------
    def __getitem__(self, index):
        """
        Parameters
        ----------
        index: colspec

        Returns
        -------
        result:
            The indexed item(s), that is, 'column(s)'.
            If index resolves to multiple 'cols' then another 'Struct' will be returned
            with those items as a shallow copy.

        Raises
        ------
        IndexError
        TypeError
        """
        if isinstance(index, (tuple, type(None))):
            raise IndexError(
                f'Can only index {self.__class__.__name__} as st[c], where c is colname, list of colnames or boolean mask.')
        col_idx, _, _, _, _ = self._extract_indexing(index)
        if isinstance(col_idx, str):
            if self.col_exists(col_idx):
                return getattr(self, col_idx)
            else:
                raise IndexError(f'Could not find column named: {col_idx}')
        elif isinstance(col_idx, (list, np.ndarray)) and len(set(col_idx)) < len(col_idx):
            raise IndexError('Cannot index cols with duplicates.')
        return self.__class__({_k: getattr(self, _k) for _k in col_idx})

    # ------------------------------------------------------------
    def __setitem__(self, index, value):
        """
        :param index: colspec
        :param value: May be any type
        :return: None
        :raise IndexError:
        :raise TypeError:
        """
        if isinstance(index, (tuple, type(None))):
            raise IndexError(
                f'Can only index {self.__class__.__name__} as st[c], where c is colname, list of colnames or boolean mask.')
        col_idx, _, ncols, _, _ = self._extract_indexing(index)
        if col_idx is None:
            col_idx = list(self.keys())
        if ncols == 0:
            return
        if ncols == 1:
            if not isinstance(col_idx, str): col_idx = col_idx[0]
            #if self.col_exists(col_idx) or self.is_valid_colname(col_idx):
            if self.is_valid_colname(col_idx):
                setattr(self, col_idx, value)
            else:
                raise IndexError(f'Invalid column name: {col_idx!r}')
        else:
            if not all([self.col_exists(_k) for _k in col_idx]):
                raise IndexError('If creating a new column can only do one at a time.')
            if isinstance(value, (tuple, list)) and len(value) == ncols:
                for _k1, _v2 in zip(col_idx, value):
                    setattr(self, _k1, _v2)
            else:
                raise IndexError('Can only set multiple columns from a list/tuple of matching size.')
        return


    # ------------------------------------------------------------
    def _struct_compare_check(self, func_name, lhs):
        """
        Returns a Struct consisting of union of key names with value self.X == self.Y.
        If a key is missing from one or the other it will have value False.
        If any comparison fails (exception) the value will be False.
        If any comparisons value Y cannot be cast to bool, Y.all() and all(Y) will be attempted.

        :param func_name: comparison function name (e.g., '__eq__')
        :param lhs:

        :return: Struct of bools
        """
        # NB Don't want comparable or return types to vary on inheritance.
        if isinstance(lhs, Struct):
            newds = {}
            for colname in self:
                if hasattr(lhs, colname):
                    self_val = getattr(self, colname)
                    lhs_val = getattr(lhs, colname)
                    func = getattr(self_val, func_name)
                    res = func(lhs_val)
                    if isinstance(res, type(NotImplemented)):
                        raise NotImplementedError(
                            f'Cannot compare types for key {colname!r}; {type(self_val).__name__} and {type(lhs_val).__name__}')
                    try:
                        # print(self_val, lhs_val, res, hasattr(res, 'all'))
                        # import pdb; pdb.set_trace()
                        res = bool(res)
                    except (TypeError, ValueError, NotImplementedError):  # cannot cast to bool
                        if hasattr(res, 'all'):
                            res = res.all()
                        else:
                            res = all(res)
                    newds[colname] = res
                else:
                    newds[colname] = False
            for colname in lhs:
                if colname not in newds:
                    newds[colname] = False
            return Struct(newds)
        else:
            raise TypeError(f'Cannot compare a Struct to type {type(lhs).__name__}')

    # ------------------------------------------------------------
    def __ne__(self, lhs):
        return self._struct_compare_check('__ne__', lhs)

    def __eq__(self, lhs):
        return self._struct_compare_check('__eq__', lhs)

    def __ge__(self, lhs):
        return self._struct_compare_check('__ge__', lhs)

    def __gt__(self, lhs):
        return self._struct_compare_check('__gt__', lhs)

    def __le__(self, lhs):
        return self._struct_compare_check('__le__', lhs)

    def __lt__(self, lhs):
        return self._struct_compare_check('__lt__', lhs)

    # -------------------------------------------------------
    def _run_once(self):
        """
        Other classes may override _run_once to initialize data, see _post_init()
        :return: None
        """
        pass

    # ------------------------------------------------------------
    def _ipython_key_completions_(self):
        # For tab autocomplete with __getitem__
        # NOTE: %config IPCompleter.greedy=True   might have to be set
        # autocompleter will sort the keys
        return self.keys()

    # ------------------------------------------------------------
    def __dir__(self):
        # this will display the columns sorted + the instance variables + the class dict
        # NOTE: as single underscore is missing such as _nrows -- we could add the from __dict__
        # NOTE: %config IPCompleter.use_jedi=True  seems to help reduce the auto complete for .
        return list(itertools.chain(sorted(self.keys()), super().__dir__()))

    # ------------------------------------------------------------
    def __iter__(self):
        return self._all_items.__iter__()

    # ------------------------------------------------------------
    def __reversed__(self):
        return reversed(list(self.keys()))

    # ------------------------------------------------------------
    def __len__(self):
        # Debated October 2019
        # For Struct we will return the number of columns for length
        # since the row length is variable.  Another option would be to return 0.
        return self._ncols

    # ------------------------------------------------------------
    def __contains__(self, item):
        return item in self._all_items

    def _meta_dict(self, name=None):
        classname = self.__class__.__name__
        if name is None:
            name = classname
        metadict = {
            'name' : name,
            'classname' : classname,
            'author' : 'python',

            #'item_names' : [], # ***remove list of all item names
            'item_meta' : [],  # list of special fastarray metadata strings

            'labels' : self.label_get_names(),
            '_col_sortlist' : self._col_sortlist,
            'footers' : []
        }
        return metadict

    @classmethod
    def _from_meta_data(cls, itemdict, itemflags, meta):
        meta = MetaData(meta)
        allitems = {}

        # flip item meta to a dictionary lookup of item name -> meta
        # names are only held here
        item_meta = meta.get('item_meta', [])
        item_meta = [ MetaData(imeta) for imeta in item_meta ]
        item_meta = { imeta['name']:imeta for imeta in item_meta }

        try:
            from_matlab = meta['author']=='matlab'
        except:
            from_matlab = False

        for itemidx, (itemname, item) in enumerate(itemdict.items()):
            itemflag = itemflags[itemidx]

            # top-level item
            if itemflag & SDSFlag.OriginalContainer:

                # this items arrays and other info are in a dictionary
                if itemflag & SDSFlag.Nested:
                    idict = item
                    iflags = item.pop('_flags')
                    imeta = item.pop('_meta')
                    # also use the generic routine for all classes here
                    item = TypeRegister.from_meta_data( idict, iflags, imeta )
                    allitems[itemname] = item

                else:
                    # if None is in arrays, it means the item was excluded
                    if item is None:
                        continue

                    # pop scalar value unless it's from matlab (matlab has no scalars)
                    if (itemflag & SDSFlag.Scalar) and not from_matlab:
                        item = item[0]

                    # regular array or underlying array of FastArray subclass
                    else:
                        imeta = item_meta.get(itemname, None)

                        # special rebuild
                        if imeta is not None:
                            # this needs to do a pass over the whole dict for each special item...
                            # can't pop the item because dict is being iterated over
                            # maybe use another dict?
                            idict = { k:v for k,v in itemdict.items() if k.startswith(itemname+'!') }
                            idict[itemname] = item
                            # flags only seem important for container loading
                            iflags = []

                            item = TypeRegister.from_meta_data( itemdict=idict, meta=imeta )

                    if not from_matlab or item.strides[0] != 0:
                        # matlab can save 0 length arrays
                        allitems[itemname] = item

        result = cls(allitems)

        # TODO: add code to restore footers
        try:
            result.label_set_names(meta['labels'])
        except:
            pass

        return result


    # ------------------------------------------------------------
    def _as_meta_data(self, name=None, nested=True):

        itemdict = {}
        itemflags = []
        meta = MetaData( self._meta_dict(name=name) )

        # need to be able to add '_meta' as an item in Struct
        warnstate = Struct.WarnOnInvalidNames
        Struct.WarnOnInvalidNames = True

        for itemname, item in self.items():
            try:
                if item.__module__ == 'hdf5.io':
                    item = h5io_to_struct(item)
            except:
                pass

            # arrays / array sublclasses
            if isinstance(item, np.ndarray):
                itemflag = SDSFlag.OriginalContainer

                # get arrays, flags, meta from array subclass
                if hasattr(item, '_as_meta_data'):
                    idict, iflags, imeta = item._as_meta_data(name=itemname)
                    for k,v in idict.items():
                        itemdict[k] = v
                    itemflags = itemflags + iflags
                    # like this or store as separate item?
                    meta['item_meta'].append(imeta)

                # regular array
                else:
                    # might need to copy strided data
                    if item.ndim == 1:
                        if item.strides[0] != item.itemsize:
                            warnings.warn(f'array named {itemname} had bad 1d strides')
                            item = item.copy()

                    itemflag += SDSFlag.Stackable
                    itemflags.append(itemflag)
                    itemdict[itemname] = item

            # nested containers
            elif isinstance(item, Struct):
                # add nested struct to itemdict?
                # how does its metadata get added?
                # how do flags get added?
                idict, iflags, imeta = item._as_meta_data(name=itemname,nested=nested)
                idict['_meta'] = imeta
                idict['_flags'] = iflags
                itemdict[itemname] = idict

                itemflag = SDSFlag.OriginalContainer + SDSFlag.Nested
                itemflags.append(itemflag)

            # scalars
            else:
                item = np.asarray([item])
                # if scalar created an object array, flip to string and warn
                if item.dtype.char == 'O':
                    warnings.warn(f'Item {item[0]} was not a supported scalar type. Saving as bytestring.')
                    item = item.astype('S')
                itemdict[itemname] = item
                itemflag = SDSFlag.Scalar + SDSFlag.OriginalContainer
                itemflags.append(itemflag)

        #FOOTERS
        # just re-sort using the column sort list from other meta data
        ## get sorted row index (Dataset only)
        #if hasattr( self, 'get_row_sort_info' ):
        #    sort_id = self.get_row_sort_info()
        #    sort_idx = TypeRegister.SortCache.get_sorted_row_index(*sort_id)
        #    if sort_idx is not None:
        #        items.append(sort_idx)

        #SORT
        # new footers - cannot be saved this way
        ## get footers (currently only implemented for Accum2 result)
        #if hasattr( self, '_footers'):
        #    for k, v in self._footers.items():
        #        meta['footers'].append(k)
        #        items.append(v)

        # flip warning mode back for column names
        Struct.WarnOnInvalidNames = warnstate

        return itemdict, itemflags, meta.string

    # --------------------------------------------------------------------------------------------------
    def _autocomplete(self) -> str:
        return f'Struct{self.shape}'

    # --------------------------------------------------------------------------------------------------
    def _build_sds_meta_data(self, name=None, nesting=True, **kwargs):
        '''
        Final SDS file will be laid out as follows:
        --------------
        header
        -------------
        meta data string (json, includes scalars)
        -------------
        arrays
        --------------
        special arrays
        --------------
        meta tuples [tuple(item name, SDSFlags) for all items]

        Nested data structures will generate their own SDS files.
        '''

        if name is None:
            name = kwargs.get('name', 'anon_struct0')

        # structs don't have nrows
        meta = MetaData({
            'name' : name,
            'classname' : self.__class__.__name__,
            'author' : 'python',
            'version' : 1,

            'item_meta' : [],  # list of special fastarray metadata strings

            'labels' : self.label_get_names(),
            '_col_sortlist' : self._col_sortlist,
            'footers' : [],
        })

        items = []
        spec_items = []

        # for matlab meta / console summary
        meta_tups = []
        spec_tups = []

        # for testing all items with meta tuple
        all_tups = []

        if nesting:
            for k, item in self.items():
                try:
                    if item.__module__ == 'hdf5.io':
                        item = h5io_to_struct(item)
                except:
                    pass

                # Some code below needs 'k' as a str; some needs it as 'bytes'.
                # It is possible to get 'k' as either type, so normalize here to
                # make it easier on the code below.
                k_bytes = k.encode() if isinstance(k, str) else k
                k = k.decode() if isinstance(k, bytes) else k

                # create a master list of item names in order
                #meta['item_names'].append(k)

                # add item to master list of item names
                if isinstance(item, np.ndarray):
                    array_flags = SDSFlag.OriginalContainer

                    # might need to copy strided data
                    additem = item
                    if item.ndim == 1:
                        if item.strides[0] != item.itemsize:
                            warnings.warn(f'array named {k} had bad 1d strides')
                            additem = item.copy()

                    items.append(additem)

                    # other arrays may need to be stored
                    if hasattr(item, '_build_sds_meta_data'):
                        i_meta, i_items, i_tups = item._build_sds_meta_data(k, **kwargs)

                        meta['item_meta'].append(i_meta.string)
                        for spec_idx, i_item in enumerate(i_items):
                            spec_items.append(i_item)

                        # check if underlying array is stackable
                        # no struct arrays are stackable
                        if type(self) != TypeRegister.Struct:
                            array_flags += i_meta.get('_base_is_stackable', 0)

                        # if in struct, extra array tuples need to be rebuilt
                        else:
                            for i, tup in enumerate(i_tups):
                                # 0 means not stackable/not in dataset
                                i_tups[i] = tuple(( tup[0], 0 ))

                        for tup in i_tups:
                            spec_tups.append(tup)

                    else:
                        if type(self) != TypeRegister.Struct:
                            array_flags += SDSFlag.Stackable

                    # Matlab, C# et. al. will not read meta data, must rely on column names + enum
                    mtup = tuple(( k_bytes, array_flags ))
                    meta_tups.append(mtup)

                    all_tups.append(mtup)

                # do we include info about contained structures in the SDS file?
                # will the store class take care of making directories, querying nested structures for meta data?
                elif isinstance(item, Struct):
                    #containers not used
                    #meta['containers'].append(k)
                    # put None as a placeholder
                    items.append(None)

                    itemnum = SDSFlag.Nested | SDSFlag.OriginalContainer
                    t = tuple(( k_bytes, itemnum ))
                    all_tups.append(t)

                # misc items, scalars, etc. get added to config dict
                else:
                    # for now keep a separate reference to their names

                    item = np.asarray([item])
                    # if scalar created an object array, flip to string and warn
                    if item.dtype.char == 'O':
                        warnings.warn(f'Item {item[0]} was not a supported scalar type. Saving as bytestring.')
                        item = item.astype('S')
                    items.append(item)

                    itemnum = SDSFlag.Scalar | SDSFlag.OriginalContainer
                    t = tuple(( k_bytes, itemnum ))
                    all_tups.append(t)

            # add special columns like categorical categories to array list
            # add tuples for special items, will be same length as spec_items
            for spec_idx, item in enumerate(spec_items):
                items.append(item)
                t = spec_tups[spec_idx]
                meta_tups.append(t)
                all_tups.append(t)

        # get sorted row index (Dataset only)
        if hasattr( self, 'get_row_sort_info' ):
            sort_id = self.get_row_sort_info()
            sort_idx = TypeRegister.SortCache.get_sorted_row_index(*sort_id)
            if sort_idx is not None:
                items.append(sort_idx)

        # get footers (currently only implemented for Accum2 result)
        if hasattr( self, '_footers'):
            for k, v in self._footers.items():
                meta['footers'].append(k)
                items.append(v)

        # test for loading items in order with tuples only
        # TJD remove this by July 2019 since it is no longer checked
        meta['load_from_tuples']=1
        meta_tups = all_tups

        return meta, items, meta_tups

    # -------------------------------------------------------
    @classmethod
    def _tree_from_sds_meta_data(cls, meta, arrays, meta_tups, file_header):
        '''
        SDS loads in info mode (no data loaded, just metadata + file header information)

        Returns
        -------
        str
            Tree display of nested structures in SDS directory.
        '''
        if not isinstance(meta, MetaData):
            meta = MetaData(meta)

        data = {}
        spec_items = {}
        scalars = {}
        containers = {}
        arr_idx = 0

        # no longer check this
        # from_tups = meta.get('load_from_tuples',1)
        for idx, tup in enumerate(meta_tups):
            name = tup[0].decode()
            itemenum = tup[1]

            if itemenum & SDSFlag.OriginalContainer:

                # TODO: write a routine to display scalars
                if itemenum & SDSFlag.Scalar:
                    scalars[name] = cls._scalar_summary(arrays[idx])

                # these items will be summarized from another file's info
                elif itemenum & SDSFlag.Nested:
                    pass

                # add array to dictionary for final summary
                else:
                    data[name] = arrays[idx]

            # mark the base item as special
            else:
                spec_items[name] = True

        result = cls._array_summary(data)

        for sc, sc_info in scalars.items():
            result[sc] = sc_info

        # hard coded for categorical right now, maybe add more info about SDSFlags
        for spec in spec_items:
            result[spec] = 'CAT'

        return result

    # -------------------------------------------------------
    @classmethod
    def _scalar_summary(cls, scalar_tup):
        '''
        Scalars are stored as arrays in SDS, but a flag is set in the meta tuple.
        They will be labeled as scalar and their dtype will be displayed.
        '''
        info_str = ['scalar']
        typename = scalar_tup[1]
        typename = str(np.sctypeDict[typename].__name__)
        itemsize = str(scalar_tup[3])

        info_str.append(typename)
        info_str.append(itemsize+' bytes')

        return " ".join(info_str)

    # -------------------------------------------------------
    @classmethod
    def _array_info_list(cls, arrinfo):
        '''
        Build list of info for single array.
        Used for all arrays in a container or a single array stored in single SDS file.

        returns ['FA', 'shape', 'dtype name', 'i+itemsize']
        '''
        shape, typenum, flagnum, itemsize = arrinfo
        info_str = []
        info_str.append("FA")
        info_str.append(str(shape))
        info_str.append(str(np.sctypeDict[typenum].__name__))
        info_str.append("i"+str(itemsize))
        return info_str

    @classmethod
    def _array_summary_single(cls, arrinfo):
        pass

    # -------------------------------------------------------
    @classmethod
    def _array_summary(cls, data, name=None):
        '''
        :param data: Tuple of array info from CompressionType.Info
            tup1: (tuple) shape
            tup2: (int) dtype.num
            tup3: (int) bitmask for numpy flags
            tup4: (int) itemsize
        :param name: Optional name for top-level Struct.

        Intenal routine for tree from meta summary (info only, no arrays)

        :return: String of array info for a single struct.
        '''

        # in case a struct has no arrays
        if len(data) == 0:
            return Struct({})

        all_strs = []
        for n, arrinfo in data.items():
            info_str = cls._array_info_list(arrinfo)
            all_strs.append(info_str)
            # add flag info

        # calc max widths
        num_fields = len(all_strs[0])
        max_widths = []
        for i in range(num_fields):
            max_len = len(max([ info[i] for info in all_strs], key=len))
            max_widths.append(max_len)

        # fix alignment for console display
        final_summaries = cls._align_array_info(all_strs, max_widths)

        # build temp struct
        result = Struct({ n : final_summaries[idx] for idx, n in enumerate(data.keys()) })
        return result

        ##  return struct of summary strings
        #return result.tree(name=name)

    # -------------------------------------------------------
    @classmethod
    def _align_array_info(cls, allinfo, maxwidths):

        finalinfo = []
        for arrinfo in allinfo:
            newinfo = []
            for idx, item in enumerate(arrinfo):
                padding = " "
                padding *= maxwidths[idx]- len(item) + 1
                newinfo.append(item+padding)
            finalinfo.append(" ".join(newinfo))

        return finalinfo

    # --------------------------------------------------------
    def _copy(self, deep: bool = False, cls: Optional[type] = None):
        """
        Parameters
        ----------
        deep : bool, default True
           if True, perform a deep copy calling each object depth first with ``.copy(True)``
           if False, a shallow ``.copy(False)`` is called, often just copying the containers dict.

        cls : type, optional
            Class of return type, for subclass super() calls

        First argument must be deep.  Deep cannnot be set to None.  It must be True or False.
        """
        if cls is None:
            cls = type(self)

        st = Struct({})
        st_locked = self.is_locked()
        if st_locked:
            st._unlock()

        for name, obj in self.items():
            if deep:
                # try varying order of copy
                try:
                    st[name]= obj.copy(deep=deep)
                except Exception:
                    try:
                        st[name]= obj.copy()
                    except Exception:
                        st[name]= obj
            else:
                st[name]= obj

        if st_locked:
            st._lock()
        return st

    # --------------------------------------------------------
    def copy(self, deep: bool = True):
        '''
        Returns a shallow or deep copy of the `Struct`.
        Defaults to a deep copy.

        Parameters
        ----------
        deep : bool, default True
           if True, perform a deep copy calling each object depth first with ``.copy(True)``
           if False, a shallow ``.copy(False)`` is called, often just copying the containers dict.

        Examples
        --------
        >>> ds=rt.Dataset({'somenans': [0., 1., 2., nan, 4., 5.], 'morestuff': ['A','B','C','D','E','F']})
        >>> ds2=rt.Dataset({'somenans': [0., 1., nan, 3., 4., 5.], 'morestuff':['H','I','J','K','L','M']})
        >>> st=Struct({'test':ds, 'test2': Struct({'ds2':ds2}), 'arr': arange(10)})
        >>> st.copy()
        #   Name    Type      Size              0     1   2
        -   -----   -------   ---------------   ---   -   -
        0   test    Dataset   6 rows x 2 cols
        1   test2   Struct    1                 ds2
        2   arr     int32     10                0     1   2
        '''
        return self._copy(deep)

    # -------------------------------------------------------
    @classmethod
    def concat_structs(cls, struct_list):
        '''Merges data from multiple structs.

        Structs must have the same keys, and contain only Structs, Datasets, arrays, and riptable arrays.

        A struct utility for merging data from multiple structs (useful for multiday loading).
        Structs must have the same keys, and contain only Structs, Datasets, Categoricals, and Numpy Arrays.

        Parameters
        ----------
        struct_list : list of `Struct`

        Returns
        -------
        obj:`Struct`

        See Also
        --------
        :func:`hstack`
        '''
        return cls.hstack(struct_list)

    # -------------------------------------------------------
    @classmethod
    def hstack(cls, struct_list):
        '''
        Merges data from multiple structs.
        Structs must have the same keys, and contain only Structs, Datasets, arrays, and riptable arrays.

        Parameters
        ----------
        struct_list : list of `Struct`

        A struct utility for merging data from multiple structs (useful for multiday loading).
        Structs must have the same keys, and contain only Structs, Datasets, Categoricals, and Numpy Arrays.

        Returns
        -------
        obj:`Struct`

        See Also
        --------
        riptable.hstack
        '''
        return hstack_any(struct_list, cls, Struct)

    # -------------------------------------------------------
    @classmethod
    def _load_from_sds_meta_data_nested(cls, name, meta, arrdict):
        ds_struct = cls(arrdict)
        ds_struct.label_set_names(meta['labels'])
        return ds_struct

    # -------------------------------------------------------
    @classmethod
    def _load_from_sds_meta_data(cls, meta, arrays, meta_tups=[], file_header={}, include=None):
        '''
        Iterates over sections of the meta data object to rebuild a data structure.

        Arrays will be in the following order:
        - Main arrays (or underlying FastArrays for subclasses)
        - Secondary arrays for FastArray subclasses that require additional contiguous data (e.g. Categorical)
        - Array of fancy indices to sort by

        A dictionary will be constructed.
        All arrays will be inserted by name from 'item_names' in meta object.
        All (if any) meta data will be read from 'item_meta' in meta object, and FastArray subclasses will be constructed.
        The container object will be constructed from the dictionary.
        Any labels (gbkeys) will be set.
        If sorted column names exist, they will be set, and the sorted index will be added to the SortCache.

        Parameters
        ----------
        meta : riptable MetaData object (see Utils/rt_metadata.py)
        arrays : list of numpy arrays from an expanded SDS file

        Returns
        -------
        Struct, Dataset, or Multiset
            For now, Struct, Dataset, and Multiset all use this parent method.
        '''
        # TODO: check possible include list - will tuples still be returned for excluded items?
        # currently getting removed from root struct only

        def load_class_meta(data, arrays, meta):
            arr_idx = 0
            data.label_set_names(meta['labels'])

            # send the sort index to the SortCache
            data._col_sortlist = meta.get('_col_sortlist', None)
            if data._col_sortlist is not None:
                # if a sorted index exists, it's the last array in the list
                uid = data._uniqueid
                sortlist = [ data[col] for col in data._col_sortlist ]
                sortidx = arrays[arr_idx]
                TypeRegister.SortCache.store_sort(uid, sortlist, sortidx)
                arr_idx += 1

            # attach footers (accum2 operation results)
            if len(meta['footers']) > 0:
                footers = {}
                for f in meta['footers']:
                    footers[f] = arrays[arr_idx]
                    arr_idx+=1
                data._footers = footers

        #---- start of method below ------------------------
        if not isinstance(meta, MetaData):
            try:
                meta = MetaData(meta)
            except:
                warnings.warn(f'meta data did not contain a valid json string.')

        data = {}

        if isinstance(meta, MetaData):
            author = meta.get('author', 'unknown')
            try:
                from_matlab = author =='matlab'
            except:
                from_matlab = False
            spec_items = {}
            spec_items2 = {}

            for item_idx, tup in enumerate(meta_tups):
                itemname = tup[0].decode()
                itemenum = tup[1]

                # regular item (array, scalar, container)
                if itemenum & SDSFlag.OriginalContainer:
                    # initialize the container name to preserve order
                    if itemenum & SDSFlag.Nested:
                        data[itemname] = None

                    else:
                        # if None is in arrays, it means the item was excluded
                        try:
                            if arrays[item_idx] is None:
                                pass
                            else:
                                # pop scalar value unless it's from matlab (matlab has no scalars)
                                if (itemenum & SDSFlag.Scalar) and not from_matlab:
                                    item = arrays[item_idx][0]

                                # regular array or underlying array of FastArray subclass
                                else:
                                    item = arrays[item_idx]

                                if not from_matlab or item.strides[0] != 0:
                                    # matlab can save 0 length arrays
                                    data[itemname] = item
                        except Exception:
                            # TODO: change this when matlab stores metadata
                            if not from_matlab:
                                origerror = sys.exc_info()[1]
                                warnings.warn(f'Error occured when processing meta data on index {item_idx}.  Author: {author}. Error: {origerror!r}')


                # auxilery item (categorical uniques, etc.)
                # python only
                else:
                    sep_idx = itemname.find('!')
                    spec_name = itemname[:sep_idx]

                    # each dictionary key in spec_arrays corresponds to an item in the original container
                    # spec_items = {itemname: [arr1, arr2, arr3...]}
                    spec_list = spec_items.setdefault(spec_name,[])
                    spec_list.append(arrays[item_idx])

            # only use the meta data for rebuilding special subclasses
            # python only
            item_meta = meta.get('item_meta', [])
            for i_meta in item_meta:
                i_meta = MetaData(i_meta)

                i_name = i_meta['name']
                # item may have been excluded, but will still appear in item_meta list
                if i_name in data:
                    underlying_arr = data[i_name]

                    # assign the array a name early so that categorical can use it
                    underlying_arr.set_name(i_name)
                    arrlist = spec_items.get(i_name, [])

                    i_class = i_meta.itemclass

                    # TEST
                    data[i_name] = i_class._load_from_sds_meta_data(i_name, underlying_arr, arrlist, i_meta)

            # build data structure from dictionary
            # TEST
            data = cls(data)

            # only python has labels, sort index, footers
            try:
                load_class_meta(data, arrays, meta)
            except:
                pass
        else:
            # need an example of this, non-python/matlab sds file?
            data = cls._load_without_meta_data(meta, arrays, meta_tups, file_header)

        return data

    #--------------------------------------------------------------------------
    @classmethod
    def _load_without_meta_data(cls, meta, arrays, meta_tups, file_header: Optional[dict] = None):
        '''
        Loads from meta tuples only (e.g. when no metadata is generated by Matlab)
        '''
        if file_header is None:
            file_header = dict()
        data = {}
        extra_cols = [] # not used yet

        for idx, meta_tup in enumerate(meta_tups):
            name = meta_tup[0].decode()
            meta_enum = meta_tup[1]
            if meta_enum & SDSFlag.OriginalContainer:
                # extract scalars
                if meta_enum & SDSFlag.Scalar:
                    item = arrays[idx][0]

                # single array or container
                else:
                    # TODO: add path if item was nested container - has the container already been built?
                    item = arrays[idx]
                # TODO: eventually check stackable flag (will be applied with multiday load)

                data[name] = item
            else:
                #extra_cols.append(meta_tup)
                print(f'column {name} was not from original dataset')

        try:
            # TODO: get dataset/struct/table info from SDS load call
            data = cls(data)
        except:
            # always fall back on struct
            data = Struct(data)

        return data

    #--------------------------------------------------------------------------
    def make_categoricals(self, columnlist=None, dtype=None) -> None:
        """
        Converts specified string/bytes columns or all string/bytes columns to Categorical.
        Will also crawl through nested structs/datasets and convert their strings to categoricals.

        Parameters
        ----------
        columnlist : `str` or `list`, optional
            Single name, or list of names of items to convert to categoricals.
        dtype : `numpy.dtype`, optional
            Integer dtype for the categoricals' underlying arrays.

        Raises
        ------
        TypeError
            If the dtype was set to a non-dtype object.
        ValueError
            If a requested item could not be found in the container.

        Notes
        -----
        Error checking will complete in the root structure before any conversion begins.
        """

        if dtype is not None and isinstance(dtype, np.dtype) is False:
            raise TypeError(f"dtype keyword was not numpy dtype. got {type(dtype)} instead.")

        if columnlist is None:
            columnlist = self.keys()

        # make sure all columns exist before starting conversion
        else:
            if not isinstance(columnlist, list):
                columnlist = [columnlist]

            for c in columnlist:
                try:
                    getattr(self, c)
                except:
                    raise ValueError(f"Could not find column {c} in Dataset. Could not flip to categorical.")

        for colname in columnlist:
            col = self[colname]
            # check to see if we need to make a nested call
            if hasattr(col, 'make_categoricals'):
                col.make_categoricals(dtype=dtype)
            else:
                if isinstance(col, np.ndarray) and col.dtype.char in 'US':
                    self[colname] = TypeRegister.Categorical(col, dtype=dtype)


    #--------------------------------------------------------------------------
    def make_struct_from_categories(self, prefix=None, keep_prefix=False):
        '''
        Build a struct of unique arrays from all categoricals in the container, or those with a specified prefix.

        Parameters
        ----------
        prefix : `str`, optional
            Only include columns with names that begin with this string.
        keep_prefix : bool, default False
            If True, keep the prefix when naming the item in the new structure.

        Examples
        --------
        TODO - sanitize - add example that makes a struct from categoricals and prints its representation
        See the version history for structure of older examples.

        Returns
        -------
        cats : Struct

        Notes
        -----
        This is a partial inverse operation of Struct.make_matlab_categoricals
        '''
        if prefix is None:
            names = self.keys()
            plen = 0
        else:
            names = [ n for n in self.keys() if n.startswith(prefix) ]
            if keep_prefix:
                plen = 0
            else:
                plen = len(prefix)

        cats = {}
        for n in names:
            col = self[n]
            if isinstance(col, TypeRegister.Categorical):
                try:
                    cats[n[plen:]]=col.category_array
                except:
                    NotImplementedError(f"Categorical struct return only supports categoricals from single arrays.")

        cats = TypeRegister.Struct(cats)
        return cats

    # -------------------------------------------------------
    def make_matlab_datetimes(self, dtcols: Optional[Union[str, list]] = None, gmt: bool = False, auto: bool = True) -> None:
        """
        Convert datetime columns from Matlab to DateTimeNano and TimeSpan arrays.

        Parameters
        ----------
        dtcols : `str` or `list`
            Name or list of names of columns to convert to DateTimeNano arrays.
        gmt : bool, optional, default False
            Not implemented.
        auto : bool, optional, default True
            If True, look for 'MS' in the names of all columns, and flip them to TimeSpan objects.

        """
        if dtcols is None:
            dtcols = list()
        if isinstance(dtcols, str):
            dtcols = [dtcols]

        if auto:
            for name, col in self.items():
                if name.endswith('MS') and col.dtype.char in NumpyCharTypes.AllFloat:
                    self[name] = TypeRegister.TimeSpan(col*1_000_000)

        for name in dtcols:
            self[name] = TypeRegister.DateTimeNano(self[name], from_tz='GMT', to_tz='NYC')

    # -------------------------------------------------------
    def make_matlab_categoricals(self, xtra, remove_trailing=True, dtype=None, prefix='p', keep_prefix=True) -> None:
        """
        Turn matlab categorical indices and corresponding unique arrays into riptable categoricals.

        Parameters
        ----------
        xtra : `Struct`
            Container holding unique arrays.
        remove_trailing : bool, optional, default True
            If True, remove trailing spaces from Matlab strings.
        dtype : `numpy.dtype`, optional, default None
            Integer dtype for underlying array of constructed categoricals.
        prefix : `str`, optional, default 'p'
            Prefix for integer arrays in calling dataset - columns that will be looked for in the struct.
        keep_prefix : bool, default True
            If True, Drop the prefix after flipping the column to categorical in the dataset. If the a column exists with that name, the user will be warned.
        """
        if not isinstance(xtra, dict):
            xtra = xtra.asdict()
        # place a p in front of all names
        pNames = [prefix + k for k in xtra]

        strConvert = []
        strCount = 0
        for pname in pNames:
            if hasattr(self, pname):
                # move past the letter 'p' in the name to get to the array the index is referencing
                strNames = xtra[pname[len(prefix):]]

                # check to see if we have the pVersion of the name
                pArray = getattr(self, pname)
                if not isinstance(pArray, TypeRegister.Categorical):
                    strConvert.append(pname)
                    strCount += 1
                    try:
                        # convert matlab scalars to array of one item
                        if not isinstance(strNames, np.ndarray):
                            strNames = TypeRegister.FastArray(strNames)

                        # matlab has padded strings, unpad for correct comparisons
                        if remove_trailing and strNames.dtype.char in ('U','S'):
                            strNames = rc.RemoveTrailingSpaces(strNames)
                        newcategory = TypeRegister.Categorical(pArray, strNames, base_index=1, from_matlab=True)

                        # drop the leading prefix from the index column, remove index column
                        if not keep_prefix:
                            trimmed_name = pname[len(prefix):]
                            try:
                                self.col_remove(trimmed_name)
                                warnings.warn(f"Replaced column {trimmed_name} with categorical from matlab.")
                            except:
                                pass

                            # if renaming fails, try capitalized
                            try:
                                self.col_rename(pname, trimmed_name)
                            except Exception:
                                trimmed_name = trimmed_name.capitalize()
                                self.col_rename(pname, trimmed_name)

                            self.col_set_value(trimmed_name, newcategory)
                        # keep the leading prefix
                        else:
                            self.col_set_value(pname, newcategory)

                        #help recycler
                        del newcategory
                    except:
                        raise TypeError(f'Cannot convert {pname} as a Categorical')

                # help recycler
                del pArray

        print(f'Converted {strCount} arrays: {strConvert}')

    # -------------------------------------------------------
    @classmethod
    def load(cls, path: Union[str, os.PathLike] = '', name: Optional[str] = None, share: Optional[str] = None, info: bool = False,
             columns=None, include_all_sds=False, include: Optional[Sequence[str]] = None, threads: Optional[int] = None, folders: Optional[Sequence[str]] = None):
        """
        Load a Struct from a directory or single SDS file.

        Parameters
        ----------
        path : str or os.PathLike
            Full path to directory or single SDS file with Struct data.
        name : `str`, optional, default None
            Name of a nested container to search for in the root directory. Multiple tiers can be separated by '//'
        info : bool, optional, default False
            If True, no array data will be loaded, instead a display tree of information about nested structures
            and their contents will be returned.
        columns : `list`, optional, default None
            Not implemented
        include_all_sds : bool, optional, default False
            If False, when additional files were found in a directory, and they were not in the root structs meta data, the user
            will be prompted to load them. If True, all files will be automatically loaded.
        include : list of str, optional, default None
            A list of specific items to load. This list will only be applied to the root Struct - not to nested containers.
        threads : int, optional, default None
            Number of threads to use during the SDS load. Number of threads before the load will be restored after the load
            or if the load fails. See also `riptide_cpp.SetThreadWakeUp`.

        Returns
        -------
        `Struct`
            Loaded data with possibly nested containers and riptable classes restored.

        See Also
        --------
        riptable.load_sds

        """
        result = _load_sds(path, name=name, sharename=share, info=info, include_all_sds=include_all_sds, include=include, threads=threads, folders=folders)
        if info:
            result = cls._info_tree(path, result)
        return result

    # -------------------------------------------------------
    @classmethod
    def _info_tree(cls, path: Union[str, os.PathLike], data):
        """
        Converts nested structure to tree view of file info for Struct and Dataset.
        Top level will be named based on single file or directory.
        """
        path = os.path.basename(path)
        name = os.path.splitext(path)[0]
        return data.tree(name=name, info=True)

    # -------------------------------------------------------
    def save(self, path: Union[str, os.PathLike] = '', name: Optional[str] = None, share: Optional[str] = None,
             overwrite: bool = True, compress: bool = True, onefile: bool = False, bandsize: Optional[int] = None):
        """
        Save a struct to a directory. If the struct contains only arrays, will be saved as a single .SDS file.

        Parameters
        ----------
        path : str or os.PathLike
            Full path to save. Directory will be created automatically if it doesn't exist.
            .SDS extension will be appended if a single file is being saved and is necessary.
        name : str, optional
            Name for the root structure if it's being appended to an existing struct's directory.
            The existing _root.sds does not get overwritten, and structs can be combined without a full load.
        share : str, optional
        overwrite : bool, optional, default True
            If True, user will not be prompted on whether or not to overwrite existing .SDS files. Otherwise,
            prompt will appear if directory exists.
        compress : bool, optional, default True
            If True, ZStandard compression will be used when writing to SDS, otherwise, no compression
            will be used.
        onefile : bool, optional, default False
            If True will collapse all nesting Structs
        bandsize : int, optional, default None
        """
        save_sds(path, self, share=share, compress=compress, overwrite=overwrite, name=name, onefile=onefile, bandsize=bandsize)

    # -------------------------------------------------------
    @classmethod
    def _serialize_item(cls, item, itemname):
        '''
        return a dict of {name: array}
        a matching list of ints which are the arrayflags
        a metastring if it exists
        '''
        itemdict = {}
        itemflags=[]
        metastring = None
        # arrays / array sublclasses
        if isinstance(item, np.ndarray):

            # get arrays, flags, meta from array subclass
            if hasattr(item, '_build_sds_meta_data'):
                # store the instance array
                itemdict[itemname] = item._fa
                itemflags.append(SDSFlag.OriginalContainer + SDSFlag.Stackable + SDSFlag.Meta)

                imeta, arrlist, nameflagtup = item._build_sds_meta_data(name=itemname)

                # return this
                metastring = imeta.string

                # now pick up the rest
                for arr, tup in zip(arrlist, nameflagtup):
                    itemdict[tup[0].decode()] = arr
                    itemflags.append(tup[1])

            # regular array
            else:
                # might need to copy strided data
                if item.ndim == 1:
                    if item.strides[0] != item.itemsize:
                        warnings.warn(f'array named {itemname} had bad 1d strides')
                        item = item.copy()

                itemdict[itemname] = item
                itemflags.append(SDSFlag.OriginalContainer + SDSFlag.Stackable)

        # scalars
        else:
            item = np.asarray([item])
            # if scalar created an object array, flip to string and warn
            if item.dtype.char == 'O':
                warnings.warn(f'Item {item[0]} was not a supported scalar type. Saving as bytestring.')
                item = item.astype('S')
            itemdict[itemname] = item
            itemflags.append(SDSFlag.Scalar + SDSFlag.OriginalContainer)

        return itemdict, itemflags, metastring

    # -------------------------------------------------------
    def flatten(self, sep='/', level=0):
        '''
        Flattens or collapses a Struct, recursively called

        Parameters
        ----------
        sep='/'   the separating string to use
            Please note that some chars are not allowed and will be replaced with _.

        Returns
        -------
        New Struct with collapsed names (separated by specified char) which can then be saved

        Note
        ----
        _sep is stored in the __dict__ to help with undo or saving to file
        arrayflags, metastring are now exposed

        See Also
        --------
        flatten_undo
        '''

        def make_nested_meta(v, name, metastringdict_other):
            metajson = {}
            # nested items have their own metastring
            metastring = v._build_sds_meta_data(name=name, nesting=False)[0].string
            metajson['_root'] = metastring
            for metaname, metavalue in metastringdict_other.items():
                metajson[metaname] = metavalue
            return np.asanyarray([json.dumps(metajson)], dtype='S')

        flattened = {}
        metastringdict = {}
        arrayflags = []

        for name,v in self.items():
            if isinstance(v, Struct):
                # first add the nested container (dataset or struct)
                # nested item ends in slash (sep)
                # always create two in a row, second one for meta
                prefix = name + sep

                #place holder for ordered dict
                flattened[prefix]=None
                arrayflags.append(0 + SDSFlag.Nested)

                flattened_other, arrayflags_other, metastringdict_other = v.flatten(sep, level + 1)

                # finally put in the json
                flattened[prefix]=make_nested_meta(v, name, metastringdict_other)

            else:
                prefix=''
                flattened_other, arrayflags_other, metastring = Struct._serialize_item(v, name)
                if metastring:
                    metastringdict[name]=metastring

            for k,v in flattened_other.items():
                flattened[prefix + k]=v

            arrayflags += arrayflags_other

        if level == 0:
            st=Struct(flattened)

            # remember info to undo the flattening
            st._sep = sep
            st.arrayflags = TypeRegister.FastArray(arrayflags)

            # use json.loads(bytes(st._metastring))
            st.metastring = bytes(make_nested_meta(self, '_root', metastringdict))
            return st

        return flattened, arrayflags, metastringdict


    # -------------------------------------------------------
    def flatten_undo(self, sep=None, startname='', obj_array=None):
        '''
        Restores a Struct to original form before Struct.flatten()

        Parameters
        ----------
        sep=None, user may pass in the separating string to use such as '/'

        Returns
        -------
        New Struct that is back to original form before Struct.flatten()

        See Also
        --------
        flatten
        '''

        if hasattr(self, 'metastring'):
            metastring = self.metastring
        else:
            raise TypeError("This structure has not been flattened, no metastring")

        if hasattr(self, 'arrayflags'):
            arrayflags = self.arrayflags
        else:
            raise TypeError("This structure has not been flattened, no arrayflags")

        # put back later
        del self.metastring
        del self.arrayflags

        if obj_array is None:
            # first pass, build an array of (colname, value, arrayflag)
            obj_array = np.empty(self._ncols, dtype='O')
            for i, ((colname, arr), af) in enumerate(zip(self.items(),arrayflags)):
                obj_array[i]=(colname, arr, af)

        if sep is None:
            if hasattr(self, '_sep'):
                sep=self._sep
            else:
                sep = '/'

        result = Struct._flatten_undo(sep, 0, startname, obj_array, meta=metastring )

        # put it back
        self.arrayflags = arrayflags
        self.metastring = metastring

        return result

    # -------------------------------------------------------
    @classmethod
    def _flatten_undo(cls, sep, startpos, startname, obj_array, meta=None, cutoffs=None ):
        '''
        internal routine
        '''
        def build_class_nested(metastring, npdict):
            if metastring is not None:
                meta_struct = json.loads(metastring)
                classname = meta_struct['classname']
                name = meta_struct['name']
                # done with this subsection, return our last location
                newclass = getattr(TypeRegister, classname)
                #print("**newclass", name, meta_struct)
                return newclass._load_from_sds_meta_data_nested(name, meta_struct, npdict)
            return None


        #----------------------------------------------
        if meta is not None:
            metastringdict = json.loads(bytes(meta))
        else:
            metastringdict = {}

        seplen=len(sep)
        newstruct={}
        i=startpos
        stoppos= len(obj_array)

        while i < stoppos:
            itemname = obj_array[i][0]
            # check if we are at the end of a substruct
            if startpos != 0 and not itemname.startswith(startname):

                # build the nested class we collected earlier
                return build_class_nested(metastringdict.get('_root',None), newstruct), i

            # get the value for the itemname and its arrayflag
            value= obj_array[i][1]
            if value is not None:
                arrayflags = obj_array[i][2]

                # remove nested portion of name
                purename = itemname[len(startname):]

                # check for nesting
                if arrayflags & SDSFlag.Nested:
                    newmeta = value

                    # remove trailing sep
                    newname = itemname[:-seplen]
                    purename = purename[:-seplen]

                    # will return a new i
                    newstruct[purename], i=Struct._flatten_undo(sep, i+1, itemname, obj_array, meta=newmeta, cutoffs=cutoffs)
                else:
                    #no nesting
                    if arrayflags & SDSFlag.Meta:
                        # eat up all the arrays belonging to this advanced class
                        metabytes = metastringdict.get(purename,None)
                        #print("**meta for", purename, metabytes)
                        if metabytes is not None:
                            meta_struct = json.loads(metabytes)
                            classname = meta_struct['classname']
                            newclass = getattr(TypeRegister, classname)
                            if cutoffs is not None and TypeRegister.is_binned_type(newclass):
                                idx_cutoffs = cutoffs[i]
                                # blindly assume next value is the categorical unique
                                unique_cutoffs = []
                                cols =[]
                                while (i + 1) < stoppos:
                                    colname = obj_array[i+1][0]
                                    # note this code assumes ! is the delimiter
                                    if not colname.startswith(itemname + '!'):
                                        break
                                    i = i +1
                                    unique_cutoffs.append(cutoffs[i])
                                    cols.append(obj_array[i][1])

                                value = _multistack_categoricals(purename, [meta_struct], value, cols, idx_cutoffs, unique_cutoffs)
                                #value = newclass._load_from_sds_meta_data(purename, value, cols, meta_struct)
                            else:
                                cols =[]
                                while (i + 1) < stoppos:
                                    colname = obj_array[i+1][0]
                                    # note this code assumes ! is the delimiter
                                    if not colname.startswith(itemname + '!'):
                                        break
                                    i = i +1
                                    cols.append(obj_array[i][1])
                                value = newclass._load_from_sds_meta_data(purename, value, cols, meta_struct)

                    if arrayflags & SDSFlag.Scalar:
                        value = value[0]
                    newstruct[purename] = value
                    i=i+1
            else:
                i=i+1

        # check if in middle of processing
        newclass = build_class_nested(metastringdict.get('_root',None), newstruct)

        if startpos != 0:
            return newclass, i
            #return Struct(newstruct), i
        return newclass

    # -------------------------------------------------------
    @classmethod
    def _from_sds_onefile(cls, arrs, meta_tups, meta=None, folders=None):
        '''
        Special routine called after loading an SDS onefile to re-expand
        '''
        sep = '/'
        startname =''

        # TODO: move this routine to Struct
        # build an array of objects (colname, value, arrayflag) - required to reverse flatten
        obj_array = np.empty(len(arrs), dtype='O')
        for i, (arr, (colname, af)) in enumerate(zip(arrs, meta_tups)):
            obj_array[i]=(colname.decode(), arr, af)

        newstruct = TypeRegister.Struct._flatten_undo(sep, 0, startname, obj_array, meta=meta)

        # for just one folder, return the actual Dataset or Struct object
        if folders is not None and len(folders) == 1:
            # remove the last /
            newstruct = newstruct[folders[0][:-1]]
        return newstruct


    # -------------------------------------------------------
    @property
    def has_nested_containers(self) -> bool:
        """bool: True if the Struct contains other Struct-like objects."""
        has_nested = False
        for v in self.values():
            if isinstance(v, Struct):
                has_nested = True
                break
        return has_nested

    # -------------------------------------------------------
    @property
    def shape(self):
        """tuple of int: Number of rows and columns."""
        if hasattr(self, '_nrows') and self._nrows is not None:
            return self._nrows, self._ncols
        return 0, self._ncols

    # -------------------------------------------------------
    def get_nrows(self):
        """
        Retunrs 0, as a Struct has no rows.

        Returns
        -------
        int
            0

        Note
        ----
        Subclasses need to define this explicitly.
        """
        return 0

    # -------------------------------------------------------
    def get_ncols(self):
        """ Return the number of items in the Struct.

        Returns
        -------
        ncols : int
            The number of items in the Struct
        """
        return self._ncols

    # ------------------------------------------------------------
    def keys(self):
        """
        Returns
        -------
        list
            Item names.
        """
        return self._all_items.keys()

    # ------------------------------------------------------------
    def values(self):
        """
        Values are individual items from the struct (no attribute from item container).

        Returns
        -------
        dict_values
            Items.
        """
        return self._as_dictionary().values()

    # ------------------------------------------------------------
    def items(self):
        """
        Dictionary-iterator access to Struct items.

        Returns
        -------
        dict_items
            Name, Item pairs.

        :return: iterator to column keys and values
        """
        return self._as_dictionary().items()

    # ------------------------------------------------------------
    def get_attribute(self, attrib_name, default=None):
        '''
        Get an attribute that applies to all items/columns.

        Parameters
        ----------
        attrib_name
            name of the attribute
        default
            return value if attrib_name is not a valid attribute

        Returns
        -------
        val : attribute value or None

        See Also
        --------
        col_get_attribute, set_attribute
        '''
        return getattr(self._all_items, attrib_name, default)

    # -------------------------------------------------------
    def set_attribute(self, attrib_name, attrib_value):
        '''
        Set an attribute that applies to all items/columns.

        Parameters
        ----------
        attrib_name
            name of the attribute
        attrib_value
            value of the attribute

        See Also
        --------
        col_set_attribute, get_attribute
        '''
        setattr(self._all_items, attrib_name, attrib_value)

    # -------------------------------------------------------
    def col_delete(self, name: Union[str, List[str]]) -> None:
        """
        Remove and item from the struct.

        Parameters
        ----------
        name : str or list of str
            Name or list of item names to be removed.

        Raises
        ------
        IndexError
            Item not found with given name.
        """
        self.col_remove(name)

    # -------------------------------------------------------
    def col_get_value(self, name:str):
        """
        Return a single item.

        Parameters
        ----------
        name : string
            Item name.

        Returns
        -------
        obj
            Item from item container (no attribute).

        Raises
        ------
        KeyError
            Item not found with given `name`.

        """
        return self._all_items.item_get_value(name)

    # -------------------------------------------------------
    def col_set_value(self, name:str, value):
        """
        Check if item name is allowed, possibly escape. Set the value portion of the item to value.

        Parameters
        ----------
        name : str
            Item name.
        value : object
            For structs, nearly anything. For datasets, array.

        """
        if not Struct.AllNames and not self.is_valid_colname(name):
            if self.__class__.WarnOnInvalidNames:
                warnings.warn(
                    f'Invalid column name {name!r} --- please rename the column or the class may lose this method.')
            else:
                if str.islower(name[0]):
                    warnings.warn(
                        f'Invalid column name {name!r} --- column auto capitalized to {name.capitalize()!r}.')
                    name=name.capitalize()
                    self.col_set_value(name, value)

                else:
                    warnings.warn(
                        f'Adding column with invalid name {name!r}.')
                    #raise ValueError(f'Invalid column name {name!r}')

        # all columns are set here
        # for FastArrays we automatically attach the column name to the instance
        if isinstance(value, TypeRegister.FastArray):
            curname = value.get_name()
            if curname is not None and curname != name:
                # make a new view so we do not clobber old name
                # TJD this make a shallow copy of the object's dict
                # This code may need review
                value = value._view_internal(type(value))
            value.set_name(name)

        self._all_items.item_set_value(name, value)
        self._update_sort(name)

    # -------------------------------------------------------
    def col_get_attribute(self, name:str, attrib_name:str, default=None):
        '''
        Gets the attribute of the specified column, the `attrib_name` must be used to indicate which attribute.

        Parameters
        ----------
        name : str
            The name of the column
        attrib_name : str
            The name of the attribute
        default
            Default value returned when attribute not found.

        Examples
        --------
        >>> ds.col_set_attribute('col1', 'TEST', 417)
        >>> ds.col_get_attribute('col1', 'TEST')
        417
        >>> ds.col_get_attribute('col1', 'TEST', nan)
        417
        >>> ds.col_get_attribute('col1', 'DOESNOTEXIST', nan)
        nan
        '''
        return self._all_items.item_get_attribute(name, attrib_name, default)

    # -------------------------------------------------------
    def col_set_attribute(self, name:str, attrib_name:str, attrib_value) -> None:
        '''
        Sets the attribute of the specified column, the attrib_name must be used to indicate which attribute.

        Parameters
        ----------
        name : str
            The name of the column
        attrib_name : str
            The name of the attribute
        attrib_value
            The value of the attribute

        Examples
        --------
        >>> ds.col_set_attribute('col1', 'TEST', 417)
        >>> ds.col_get_attribute('col1', 'TEST')
        417
        '''
        self._all_items.item_set_attribute(name, attrib_name, attrib_value)

    # -------------------------------------------------------
    def col_get_len(self):
        """ Gets the number of columns (or items) in the Struct"""
        return self._all_items.item_get_len()

    # -------------------------------------------------------
    def col_exists(self, name):
        """ Return True if the column name already exists"""
        return self._all_items.item_exists(name)

    # -------------------------------------------------------
    def col_filter(self, items=None, like=None, regex: Optional['re.Pattern'] = None, axis=None):
        """
        Subset rows or columns of dataset according to labels in the specified index.

        Note that this routine does not filter a dataset on its
        contents. The filter is applied to the column names.

        Parameters
        ----------
        items : list-like
            List of axis to restrict to (must not all be present).
        like : string, optional
            Keep axis where "arg in col == True".
        regex : str, optional
            Regular expression string. Keep axis with re.search(regex, col) == True.
        axis : int, optional

        Returns
        -------
        same type as input object

        See Also
        --------
        Dataset.__get_item__

        Notes
        -----
        The `items`, like`, and regex` parameters are enforced to be mutually exclusive.

        Examples
        --------
        Select columns by name

        >>> ds = rt.Dataset({'one': rt.arange(3), 'two': rt.arange(3) % 2, 'three': rt.arange(3) % 3})
        >>> ds.col_filter(items=['one', 'three'])
        #   one   three
        -   ---   -----
        0     0       0
        1     1       1
        2     2       2

        Select columns by regular expression

        >>> ds.col_filter(regex='e$')
        #   one   three
        -   ---   -----
        0     0       0
        1     1       1
        2     2       2
        """
        import re

        tempsum = sum([items is None, like is None, regex is None])

        if tempsum != 2:
            raise TypeError('Keyword arguments `items`, `like`, or `regex` are mutually exclusive.  one must be supplied')

        if items is not None:
            newlist = items
            if not isinstance(items, list):
                newlist=[items]
        elif like:
            newlist=[]
            for k in self.keys():
                if like in k:
                    newlist.append(k)
        elif regex:
            matcher = re.compile(regex)
            newlist=[]
            for k in self.keys():
                if matcher.search(k) is not None:
                    newlist.append(k)
        else:
            raise TypeError('Must pass either `items`, `like`, or `regex`')

        if len(newlist) == 0:
            raise ValueError('All columns were removed')

        return self[newlist]

    # -------------------------------------------------------
    def label_set_names(self, listnames):
        """ Set which column names can be used as labels in display"""
        self._all_items.label_set_names(listnames)

    # -------------------------------------------------------
    def label_get_names(self):
        """ Gets the column names used as labels in display"""
        return self._all_items.label_get_names()

    # -------------------------------------------------------
    def label_as_dict(self):
        """ Gets the column names used as labels in display"""
        return self._all_items.label_as_dict()

    # -------------------------------------------------------
    def label_remove(self):
        """ Reomves any labels used in display"""
        return self._all_items.label_remove()

    # -------------------------------------------------------
    def label_filter(self, items=None, like=None, regex=None, axis=None):
        '''
        Subset rows of dataset according to value in its label column.

        TODO: how should multikey be handled?

        Parameters
        ----------
        items : list-like
            List of specific values to match in label column.
        like : string
            Keep items where 'like' occurs in label column.
        regex : string (regular expression)
            Keep axis with re.search(regex, col) == True.

        Examples
        --------

        >>> ds
         #   col_7   col_8   col_9   keycol
        --   -----   -----   -----   --------------
         0    0.53    0.52    0.47   paul
         1    0.10    0.78    0.09   ray
         2    0.50    0.79    0.50   paul
         3    0.81    0.68    0.72   ray
         4    0.08    0.71    0.02   john
         5    0.38    0.19    0.90   ray
         6    0.53    0.33    0.46   mary katherine
         7    0.75    0.48    0.94   john
         8    1.00    0.70    0.79   mary ann
         9    0.47    0.64    0.16   ray
        10    0.80    0.43    0.08   mary ann
        11    0.54    0.19    0.43   joe
        12    0.89    0.08    0.81   mary katherine
        13    0.96    0.91    0.33   paul
        14    0.18    0.55    0.44   ray
        15    0.42    0.49    0.66   mary ann
        16    0.05    0.53    0.66   paul
        17    0.60    0.56    0.03   joe
        18    0.62    0.42    0.56   mary ann
        19    0.63    0.33    0.95   paul

        >>> gb = ds.gb('keycol').sum()
        >>> gb.label_filter(items='john')
        *keycol   col_7   col_8   col_9
        -------   -----   -----   -----
        john       0.82    1.19    0.96

        >>> gb.label_filter(like=['ar', 'p'])
        *keycol          col_7   col_8   col_9
        --------------   -----   -----   -----
        mary ann          2.85    2.05    2.08
        mary katherine    1.43    0.41    1.27
        paul              2.66    3.08    2.92

        >>> gb.label_filter(regex='n$')
        *keycol    col_7   col_8   col_9
        --------   -----   -----   -----
        john        0.82    1.19    0.96
        mary ann    2.85    2.05    2.08
        '''
        import re

        labels = self.label_get_names()
        if labels is None:
            raise TypeError(f'Dataset has no label columns. Cannot filter rows from label column.')

        tempsum = sum([items is None, like is None, regex is None])

        if tempsum != 2:
            raise TypeError('Keyword arguments `items`, `like`, or `regex` are mutually exclusive.  one must be supplied')

        labels = {k: self[k] for k in labels}

        if items is not None:
            newlist = items
            if not isinstance(items, list):
                newlist=[items]
            filter = [ item == col for col in labels.values() for item in newlist ]
            newlist = mask_ori(filter)

        elif like:
            if not isinstance(like, list):
                like = [like]
            newlist = like
            for idx, item in enumerate(newlist):
                if isinstance(item, str):
                    item = item.encode()
            # slow, need faster string contains method
            newlist = [ col.apply(lambda x: item in x) for col in labels.values() for item in newlist ]
            newlist = mask_ori(newlist)

        # regex also slow because it has to be applied to each row item
        elif regex:
            matcher = re.compile(regex)
            newlist = [ col.apply(lambda x: matcher.search(x) is not None) for col in labels.values()]
            newlist = mask_ori(newlist)
        else:
            raise TypeError('Must pass either `items`, `like`, or `regex`')

        #if sum(newlist) == 0:
        #    raise ValueError('All rows were removed')

        return self[newlist,:]

    # -------------------------------------------------------
    def summary_set_names(self, listnames):
        """ Set which column names can be used as rights in display"""
        self._all_items.summary_set_names(listnames)

    # -------------------------------------------------------
    def summary_get_names(self):
        """ Gets the column names used as rights in display"""
        return self._all_items.summary_get_names()

    # -------------------------------------------------------
    def summary_as_dict(self):
        """ Gets the column names used as rights in display"""
        return self._all_items.summary_as_dict()

    # -------------------------------------------------------
    def summary_remove(self):
        """ Reomves any rights used in display"""
        return self._all_items.summary_remove()

    # -------------------------------------------------------
    def col_swap(self, from_cols, to_cols):
        """
        Swaps column values, names retain current order.

        Parameters
        ----------
        from_cols: list
            a list of unique extant column names
        to_cols:  list
            a list of unique extant column names

        Examples
        --------
        >>> st = Struct({'a': 1, 'b': 'fish', 'c': [5.6, 7.8], 'd': {'A': 'david', 'B': 'matthew'},
        ... 'e': np.ones(7, dtype=np.int64)})
        >>> st
        #   Name   Type    Rows   0      1     2
        -   ----   -----   ----   ----   ---   -
        0   a      int     0      1
        1   b      str     0      fish
        2   c      list    2      5.6    7.8
        3   d      dict    2      A      B
        4   e      int64   7      1      1     1

        >>> st.col_swap(list('abc'), list('cba'))
        >>> st
        #   Name   Type    Rows   0      1     2
        -   ----   -----   ----   ----   ---   -
        0   a      list    2      5.6    7.8
        1   b      str     0      fish
        2   c      int     0      1
        3   d      dict    2      A      B
        4   e      int64   7      1      1     1
        """
        if self.is_locked():
            raise AttributeError('Not allowed to call col_swap() on locked object.')
        if not isinstance(from_cols, list) and set(from_cols).issubset(self):
            raise ValueError(f'{self.__class__.__name__}.col_swap(): Invalid from_cols.')
        if not isinstance(to_cols, list) and set(to_cols).issubset(self):
            raise ValueError(f'{self.__class__.__name__}.col_swap(): Invalid to_cols.')
        if from_cols == to_cols:
            return
        if not (len(from_cols) == len(to_cols) == len(set(from_cols)) and set(from_cols) == set(to_cols)):
            raise ValueError(f'{self.__class__.__name__}.col_swap(): To list must be a permutation of from list.')
        temp = {_k: getattr(self, _k) for _k in from_cols}
        for _k1, _k2 in zip(to_cols, from_cols):
            setattr(self, _k1, temp[_k2])

    # -------------------------------------------------------
    def col_move(self, flist, blist):
        """
        Move single column or group of columns to back of list for iteration/indexing/display.
        Values of columns will remain unchanged.

        Parameters
        ----------
        flist : `list` of `str`
            Item names to move to front.
        blist : `list` of `str`
            Item names to move to back.

        See Also
        --------
        Struct.col_move_to_front, Struct.col_move_to_back
        """
        self.col_move_to_front(flist)
        self.col_move_to_back(blist)

    # -------------------------------------------------------
    def col_move_to_front(self, cols):
        """
        Move single column or group of columns to front of list for iteration/indexing/display.
        Values of columns will remain unchanged.

        Parameters
        ----------
        flist : `list` of `str`
            Item names to move to front.

        Examples
        --------
        >>> #TODO Call np.random.seed(12345) here to make the example output deterministic
        >>> ds = rt.Dataset({'col_'+str(i): np.random.rand(5) for i in range(5)})
        >>> ds
        #   col_0   col_1   col_2   col_3   col_4
        -   -----   -----   -----   -----   -----
        0    0.60    0.50    0.77    0.72    0.73
        1    0.48    0.65    0.96    0.17    0.99
        2    0.06    0.54    0.81    0.20    0.30
        3    0.18    0.85    0.24    0.44    0.38
        4    0.04    0.84    0.64    0.66    0.97

        >>> ds.col_move_to_front(['col_4', 'col_2'])
        >>> ds
        #   col_4   col_2   col_0   col_1   col_3
        -   -----   -----   -----   -----   -----
        0    0.73    0.77    0.60    0.50    0.72
        1    0.99    0.96    0.48    0.65    0.17
        2    0.30    0.81    0.06    0.54    0.20
        3    0.38    0.24    0.18    0.85    0.44
        4    0.97    0.64    0.04    0.84    0.66

        See Also
        --------
        Struct.col_move_to_front, Struct.col_move
        """
        if self.is_locked():
            raise AttributeError('Not allowed to call col_move_to_front() on locked object.')

        self._all_items.item_move_to_front(cols)

    # -------------------------------------------------------
    def col_move_to_back(self, cols):
        """
        Move single column or group of columns to front of list for iteration/indexing/display.

        Values of columns will remain unchanged.

        Parameters
        ----------
        flist : `list` of `str`
            Item names to move to back.

        Examples
        --------
        >>> #TODO Call np.random.seed(12345) here to make the example output deterministic
        >>> ds = rt.Dataset({'col_'+str(i): np.random.rand(5) for i in range(5)})
        >>> ds
        #   col_0   col_1   col_2   col_3   col_4
        -   -----   -----   -----   -----   -----
        0    0.28    0.84    0.24    0.72    0.81
        1    0.72    0.44    0.41    0.53    0.17
        2    0.37    0.66    0.61    0.52    0.50
        3    0.08    0.31    0.15    0.65    0.98
        4    0.63    0.89    0.25    0.13    0.16

        >>> ds.col_move_to_back(['col_2','col_0'])
        #   col_1   col_3   col_4   col_2   col_0
        -   -----   -----   -----   -----   -----
        0    0.84    0.72    0.81    0.24    0.28
        1    0.44    0.53    0.17    0.41    0.72
        2    0.66    0.52    0.50    0.61    0.37
        3    0.31    0.65    0.98    0.15    0.08
        4    0.89    0.13    0.16    0.25    0.63

        See Also
        --------
        Struct.col_move_to_back
        Struct.col_move
        """
        if self.is_locked():
            raise AttributeError('Not allowed to call col_move_to_back() on locked object.')

        self._all_items.item_move_to_back(cols)

    # -------------------------------------------------------
    def col_map(self, rename_dict:Mapping[str, str]) -> None:
        """
        Rename columns and re-arrange names of columns based on the rules set forth in the supplied dictionary.

        Parameters
        ----------
        rename_dict : dict
            Dictionary defining a remapping of (some/all) column names.

        Returns
        -------
        None

        Examples
        --------
        >>> #TODO Call np.random.seed(12345) here to make the example output deterministic
        >>> ds = rt.Dataset({'col_'+str(i): np.random.rand(5) for i in range(5)})
        >>> ds.col_map({'col_1':'AAA', 'col_2':'BBB'})
        >>> ds
        #   col_0    AAA    BBB   col_3   col_4
        -   -----   ----   ----   -----   -----
        0    0.55   0.21   0.27    0.85    0.03
        1    0.77   0.75   0.65    0.97    0.24
        2    0.09   0.07   0.40    0.81    0.62
        3    0.50   0.93   0.98    0.99    0.99
        4    0.40   0.45   0.53    0.49    0.76
        """
        if self.is_locked():
            raise AttributeError('Not allowed to call col_map() on locked object.')
        # the keys of rename_dict are the old column names
        # the values of rename_dict are the new column names
        if isinstance(rename_dict, dict):
            if len(set(rename_dict.values())) < len(set(rename_dict)):
                raise ValueError('Cannot rename multiple columns to same column name.')
            if not (all(isinstance(_k, (str, bytes)) for _k in rename_dict) and
                    all(isinstance(_k, (str, bytes)) for _k in rename_dict.values())):
                raise TypeError("Name map must be a dictionary of string ids to string ids.")
            if len(set(rename_dict.values()) & set(self)):
                rename_dict, swaps = self._safe_reordering_of_renames(rename_dict)
            else:
                swaps = []
            for old, new in rename_dict.items():
                self.col_rename(old, new)
            for old, new in swaps:
                self.col_swap(old, new)
        else:
            raise TypeError("Name map must be a dictionary. Use rename_col for single-column renaming.")

    # -------------------------------------------------------
    def _safe_reordering_of_renames(self, orig_dict):
        # fromcols = set(orig_dict)
        # tocols = set(orig_dict.values())
        protected = set(self)
        rename_dict = {}
        swaps = []
        transitions = []
        seen = set()
        for _o, _n in orig_dict.items():
            if _o in seen: continue
            if _o not in protected: # no-ops
                seen.add(_o)
            else:
                seqd, is_cycle = self._get_seq(orig_dict, protected, _o)
                if is_cycle:
                    swaps.append((list(seqd), list(seqd.values())))
                    seen.update(seqd)
                else:
                    transitions.append(seqd) # not yet to be marked as seen
        transitions.sort(key=len, reverse=True)
        while transitions:
            trans = transitions.pop(0)
            if len(seen.intersection(trans)) == 0:
                # must insert in reversed order but cannot reverse dict iterators
                for _o, _n in reversed(list(trans.items())):
                    rename_dict[_o] = _n
                    seen.add(_o)
        if len(seen) != len(orig_dict):
            raise ValueError('Cannot rename columns to extant column names.')
        return rename_dict, swaps

    @staticmethod
    def _get_seq(map, protected, start):
        seqd = {}
        _from = start
        while _from not in seqd:
            _to = map[_from]
            seqd[_from] = _to
            if _to not in protected:
                return seqd, False
            _from = _to
        return seqd, True

    # -------------------------------------------------------
    def col_add_prefix(self, prefix:str) -> None:
        '''
        Add the same prefix to all items in the Struct/Dataset.

        Rather than renaming the columns in a col_rename loop - which would have to rebuild the underlying dictionary N times,
        this clears the original dictionary, and rebuilds a new one once.
        Label columns and sortby columns will also be fixed to match the new names.

        Parameters
        ----------
        prefix : str
            String to add before every each item name

        Returns
        -------
        None

        Examples
        --------
        >>> #TODO Need to call np.random.seed(12345) first to ensure example runs deterministically
        >>> ds = rt.Dataset({'col_'+str(i):np.random.rand(5) for i in range(5)})
        >>> ds.col_add_prefix('NEW_')
        #   NEW_col_0   NEW_col_1   NEW_col_2   NEW_col_3   NEW_col_4
        -   ---------   ---------   ---------   ---------   ---------
        0        0.70        0.52        0.07        0.81        0.26
        1        0.13        0.43        0.01        0.46        0.45
        2        0.34        0.24        0.87        0.81        0.80
        3        0.63        0.22        0.85        0.60        0.91
        4        0.46        0.70        0.02        0.49        0.34
        '''
        sorts = self._col_sortlist

        # clear item dict first so if prefix + oldname exists, data won't be overwritten
        self._all_items.item_add_prefix(prefix)

        # fix sorts
        if sorts is not None:
            self._col_sortlist = [prefix+s for s in sorts]

    # -------------------------------------------------------
    def col_rename(self, old:str, new:str) -> None:
        """
        Rename a single column.

        Parameters
        ----------
        old : str
            Current column name.
        new : str
            New column name.

        Returns
        -------
        None

        Examples
        --------
        >>> ds = rt.Dataset({'a': np.random.rand(5), 'b': rt.arange(5)})
        >>> ds.sort_view('a')
        #      a   b
        -   ----   -
        0   0.20   2
        1   0.48   3
        2   0.53   4
        3   0.66   1
        4   0.83   0

        >>> ds.col_rename('a', 'new_a')
        >>> ds
        #   new_a   b
        -   -----   -
        0    0.20   2
        1    0.48   3
        2    0.53   4
        3    0.66   1
        4    0.83   0

        """
        if old == new:
            return
        if self.is_locked():
            raise AttributeError('Not allowed to call col_rename() on locked object.')
        if not self.is_valid_colname(new):
            raise ValueError('Invalid column name: {}'.format(new))

        value = self._all_items.item_rename(old, new)

        # for FastArrays we automatically attach the column name to the instance
        if isinstance(value, TypeRegister.FastArray):
            # ref count check?
            value.set_name(new)

        # going to go away when we do attributes
        # update column sort list if nec.
        if self._col_sortlist is not None:
            for i, col in enumerate(self._col_sortlist):
                if col == old:
                    self._col_sortlist[i] = new

    # --------------------------------------------------------
    def _ensure_atomic(self, colnames, func):
        """Only proceed with certain operations if all columns exist in table.
        Pass in the function for a more informative error.
        """
        missing = set(colnames) - set(self)
        if len(missing) != 0:
            raise IndexError(
                f'Unknown column(s) {missing} do(es) not exist. Cannot proceed with function {func.__name__}')

    # --------------------------------------------------------
    def col_remove(self, flist):
        """
        Remove single column or list of columns.

        Parameters
        ----------
        flist : string, list, or dict

        Returns
        -------
        None

        Examples
        --------
        >>> ds = Dataset({'col_'+str(i): rt.arange(5) for i in range(5)})
        >>> ds.col_remove(['col_2', 'col_0'])
        >>> ds
        #   col_1   col_3   col_4
        -   -----   -----   -----
        0       0       0       0
        1       1       1       1
        2       2       2       2
        3       3       3       3
        4       4       4       4
        """
        if self.is_locked():
            raise AttributeError('Not allowed to call col_remove() on locked object.')
        # flip single strings to lists to use the same routine
        if isinstance(flist, (str, bytes)):
            flist = [flist]
        if isinstance(flist, (np.ndarray, list, tuple, dict)):
            # want to ensure atomic behavior:
            self._ensure_atomic(flist, self.col_remove)

            for i in flist:
                self._deleteitem(i)
        else:
            raise TypeError(
                "Fields must be list, tuple, ndarray, dictionary (keys) or a single unicode or byte string.")

    # --------------------------------------------------------
    def col_pop(self, colspec):
        """
        colspec is as for [] (getitem).
        List input will return a sub-Struct, removing it from current object.
        Single-column ("string", single integer) input will return a single "column".

        Parameters
        ----------
        colspec : list, string, or integer

        Returns
        -------
        obj
            Single value or new (same-type) object containing the removed data.

        Examples
        --------
        >>> ds = rt.Dataset({'col_'+str(i): rt.arange(5) for i in range(3)})
        >>> ds
        #   col_0   col_1   col_2
        -   -----   -----   -----
        0       0       0       0
        1       1       1       1
        2       2       2       2
        3       3       3       3
        4       4       4       4
        >>> col = ds.col_pop('col_1')
        >>> ds
        #   col_0   col_2
        -   -----   -----
        0       0       0
        1       1       1
        2       2       2
        3       3       3
        4       4       4
        >>> col
        FastArray([0, 1, 2, 3, 4])
        """
        col_idx, _, _, _, _ = self._extract_indexing(colspec)
        if col_idx is None:
            col_idx = list(self.keys())
        # TODO: use _ensure_atomic here
        col_set = set(col_idx if is_list_like(col_idx) else [col_idx])
        missing = col_set - set(self.keys())
        if missing:
            raise IndexError(
                'Unknown column(s) {} do(es) not exist and cannot be popped.'.format(sorted(missing)))

        if isinstance(col_idx, list):
            newds = self[col_idx]
        else:
            newds = getattr(self, col_idx)
        self.col_remove(col_idx)
        return newds

    # --------------------------------------------------------
    def col_str_match(self, expression, flags=0):
        """
        Create a boolean mask vector for columns whose names match the regex.

        Uses ``re.match()``, not ``re.search()``.

        Parameters
        ----------
        expression : str
            regular expression
        flags
            regex flags (from ``re`` module).

        Returns
        -------
        FastArray
            Array of bools (len ncols) which is true for columns which match the regex.

        Examples
        --------
        >>> st = rt.Struct({
        ... 'price' : arange(5),
        ... 'trade_time' : rt.arange(5) * 1000,        # expected to regex match `.*time.*`
        ... 'name' : rt.FA(['a','b','c','d','e']),
        ... 'other_trade_time' : rt.arange(5) * 1000,  # expected to regex match `.*time.*`
        ... })
        >>> st.col_str_match(r'.*time.*')
        FastArray([False,  True, False, True])
        """
        return TypeRegister.FastArray(self._all_items.item_str_match(expression, flags=flags))

    # --------------------------------------------------------
    def col_str_replace(self, old:str, new:str, max:int=-1):
        '''
        If a column name contains the old string, the old string will be replaced with the new one.
        If replacing the string will conflict with an existing column name, an error will be raised.
        Labels / sortby columns will be fixed if their names are modified.

        Parameters
        ----------
        old : str
            String to look for within individual names of columns.
        new : str
            String to replace old string in column names.
        max : int
            Optionally limit the number of occurrences per column name to replace; defaults to -1 which will replace all.

        Examples
        --------
        Replace all occurrences in each names:

        >>> ds = rt.Dataset({
        ... 'aaa': rt.arange(5),
        ... 'a' : rt.arange(5),
        ... 'aab': rt.arange(5)
        ... })
        >>> ds.col_str_replace('a', 'A')
        >>> ds
        #   AAA   A   AAb
        -   ---   -   ---
        0     0   0     0
        1     1   1     1
        2     2   2     2
        3     3   3     3
        4     4   4     4

        Limit number of replacements per name:

        >>> ds = rt.Dataset({
        ... 'aaa': rt.arange(5),
        ... 'a' : rt.arange(5),
        ... 'aab': rt.arange(5)
        ... })
        >>> ds.col_str_replace('a','A',max=1)
        >>> ds
        #   Aaa   A   Aab
        -   ---   -   ---
        0     0   0     0
        1     1   1     1
        2     2   2     2
        3     3   3     3
        4     4   4     4

        Replacing will create a conflict:

        >>> ds = rt.Dataset({'a': rt.arange(5), 'A': rt.arange(5)})
        ValueError: Item A already existed, cannot make replacement in item.
        '''
        # will return true if any column names changed
        replaced = self._all_items.item_str_replace(old, new, maxr=max)
        if replaced and self._col_sortlist is not None:
            # fix sorts
            for i, name in self._col_sortlist:
                r = name.replace(old, new, max)
                if r != name:
                    self._col_sortlist[i] = r

    # ------------------------------------------------------------
    def _as_dictionary(self, copy=False, rows=None, cols=None):
        '''
        Return a dictionary of numpy arrays.
        '''
        # if there is a groupby with key columns
        # TJD fast way to return dict
        if (not copy) and rows is None and cols is None:
            return self._all_items.items_as_dict()


        if cols is not None:
            col_selection = cols
        else:
            col_selection = self.keys()
        source_dict = {}
        for k in col_selection:
            source_dict[k] = getattr(self, k)

        return self._copy_from_dict(source_dict, copy, rows, cols)

    # ------------------------------------------------------------
    def as_ordered_dictionary(self, sublist:List[str] = None):
        """
        Returns contents of Struct as a collections.OrderedDict instance.

        Parameters
        ----------
        sublist : list of str
            Optional list restricting columns to return.

        Returns
        -------
        OrderedDict
        """
        from collections import OrderedDict
        odict = OrderedDict()
        if sublist is None:
            sublist = self._all_items
        for k in sublist:
            #if k in self.__dict__:
            #    odict[k] = self.__dict__[k]
            #else:
            #    raise ValueError(k, "is not a valid key in __dict__.")
            if self.col_exists(k):
                odict[k] = self.col_get_value(k)
            else:
                raise ValueError(k, "is not a valid key in _all_items.")
        return odict


    # -------------------------------------------------------
    def equals(self, other):
        '''
        Test whether two Structs contain the same elements in each column.
        NaNs in the same location are considered equal.

        Parameters
        ----------
        other: another Struct or dict to compare to

        Returns
        -------
        bool

        See also
        --------
        Dataset.crc, ==, >=, <=, >,  <

        Examples
        --------
        >>> s1 = rt.Struct({'t': 54, 'test': np.int64(34), 'test2': rt.arange(200)})
        >>> s2 = rt.Struct({'t': 54, 'test': np.int64(34), 'test2': rt.arange(200)})
        >>> s1.equals(s2)
        True
        '''
        if not isinstance(other, Struct):
            # try to make it a dataset
            other = Struct(other)

        if self._ncols != other._ncols:
            print("The structs are not the same size")
            return False

        for c1, c2 in zip(self, other):
            if c1 == c2:
                v1=self[c1]
                v2=other[c2]
                if isinstance(v1, np.ndarray):
                    if isinstance(v2, np.ndarray):
                        if rc.CalculateCRC(v2) != rc.CalculateCRC(v1):
                            print(f"The columns {c1!r} and {c2!r} do not crc check")
                            return False
                    else:
                        print(f"The columns {c1!r} and {c2!r} are not the same type")
                        return False
                else:
                    if not isinstance(v2, np.ndarray):
                        if isinstance(v2, Struct):
                            comptest = v1.equals(v2)
                        else:
                            comptest = (v1== v2)
                        if not comptest:
                            print(f"The columns {c1!r} and {c2!r} do not compare the same")
                            return False
                    else:
                        print(f"The columns {c1!r} and {c2!r} are not the same type")
                        return False
            else:
                print(f"The column names {c1!r} and {c2!r} do not match in the same order")
                return False
        return True


    # ------------------------------------------------------------
    # can be used to copy keylists as well
    # generic routine to copy from an ordered dict to an ordered dict
    def _copy_from_dict(self, source_dict, copy=False, rows=None, cols=None):
        # TODO: Add check to rows, e.g., check array shapes--(nrows,1), (1,nrows), (nrows,)
        # TODO: Clean this up after writing more general indexer
        npdict = {}

        rowmask = None
        if rows is not None:
            if isinstance(rows, (int, np.integer)):
                if rows > 0:
                    # take a head slice
                    rowmask=slice(0,rows)
                else:
                    # take a tail slice
                    rowmask=slice(rows,None)
            else:
                # could be a boolean mask
                rowmask =rows

        for k, arr in source_dict.items():
            if rowmask is not None:
                # could be a slice, bool mask, or fancy index
                arr = arr[rowmask]
            if copy:
                # BUG HERE, cannot just copy an array as the array might be a Categorical or some other class
                # ADDITIONAL BUG HERE MUST ABIDE BY SORT
                # arr.copy()    arr[SORTINDEX]
                npdict[k] = arr.copy()
            else:
                npdict[k] = arr
        return npdict

    # -------------------------------------------------------
    def asdict(self, sublist : List[str] = None, copy:bool=False):
        """
        Return contents of Struct as a dictionary.

        Parameters
        ----------
        sublist : list of str
            Optional list restricting columns to return.
        copy : bool
            If set to True then copy() will be called on columns where appropriate.

        Returns
        -------
        dict

        Examples
        --------
        This is useful if, for whatever reason, a riptable Dataset needs to go into a pandas DataFrame:

        >>> import pandas as pd
        >>> ds = rt.Dataset({'col_'+str(i): rt.arange(5) for i in range(5)})
        >>> df = pd.DataFrame(ds.asdict())
        >>> df
           col_0  col_1  col_2  col_3  col_4
        0      0      0      0      0      0
        1      1      1      1      1      1
        2      2      2      2      2      2
        3      3      3      3      3      3
        4      4      4      4      4      4

        Certain items can be requested with the `sublist` keyword:

        >>> ds.asdict(sublist=['col_1','col_3'])
        {'col_1': FastArray([0, 1, 2, 3, 4]), 'col_3': FastArray([0, 1, 2, 3, 4])}

        """
        return self._as_dictionary(copy=copy, cols=sublist)

    def __bool__(self):
        raise ValueError(
            f'The truth value of a {self.__class__.__name__} with more than one element is ambiguous. Use a.any() or a.all()')

    def tolist(self) -> list:
        """
        Returns data values in a list.  Output equivalent to list(st.values()).

        :return: list
        """
        return self._all_items.items_tolist()

    def any(self):
        """
        For use in boolean contexts: Does there exist an element (val) which either::

        1. val casts to True, or
        2. returns True for val.any() or any(val)

        Returns
        -------
        bool

        Examples
        --------
        >>> s=rt.Struct()
        >>> s.a=rt.Dataset()
        >>> s.any()
        False

        """
        # Want the following, but need to handle special cases (strings in arrays).
        for _v in self.values():
            if isinstance(_v, Struct):
                # avoid calling Dataset.any()
                if Struct.any(_v): return True
            elif hasattr(_v, 'any'):
                try:
                    if _v.any(): return True
                except TypeError:
                    if any(_v): return True
            elif hasattr(_v, '__iter__'):
                if any(_v): return True
            elif bool(_v): return True
        return False

    def all(self):
        """
        For use in boolean contexts: Is it true that for all elements (val) either::

        1. val casts to True, or
        2. returns True for val.all() or all(val)

        Returns
        -------
        bool
        """
        # Want the following, but need to handle special cases (strings in arrays).
        # return all(getattr(self, _cn) for _cn in self._all_items)
        for _v in self.values():
            if isinstance(_v, Struct):
                if not Struct.all(_v): return False
            elif hasattr(_v, 'all'):
                try:
                    if not _v.all(): return False
                except TypeError:
                    if not all(_v): return False
            elif hasattr(_v, '__iter__'):
                if not all(_v): return False
            elif not bool(_v): return False
        return True

    @staticmethod
    def _sizeof_fmt(num, suffix='B'):
        for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Y', suffix)

    # ------------------------------------------------------------
    def _last_row_stats(self):
        return f"[{self._ncols} columns]"

    # ------------------------------------------------------------
    def tree(self, name=None, showpaths=False, info=False):
        '''
        Returns a hierarchical view of the Struct.

        Parameters
        ----------
        name : str
            Optional name for the top of the tree
        showpaths
            TODO purpose unknown, may raise error if true
        info
            TODO purpose unknown

        Returns
        -------
        tree : str
            A hierarchical view of the Struct

        Examples
        --------
        >>> st1 = rt.Struct({'A': rt.FA([1, 2, 3]), 'B': rt.FA([4, 5])})
        >>> st2 = rt.Struct({'C': st1, 'D': rt.FA([6, 7, 8])})
        >>> st2.tree()
        Struct
         ├──── C (Struct)
         │     ├──── A int32 (3,) 4
         │     └──── B int32 (2,) 4
         └──── D int32 (3,) 4
        >>> st2.tree(name='foo')
        foo
         ├──── C (Struct)
         │     ├──── A int32 (3,) 4
         │     └──── B int32 (2,) 4
         └──── D int32 (3,) 4
        '''
        if name is None:
            name = type(self).__name__

        nested = DisplayNested()
        if TypeRegister.DisplayDetect.Mode == DisplayDetectModes.HTML:
            return nested.build_nested_html(self, name=name)
        else:
            return nested.build_nested_string(self, name=name, showpaths=showpaths, info=info)

    # -------------------------------------------------------
    def dtranspose(self, plain=False):
        """
        For display only.
        Return a transposed version of the container's string representation.

        Parameters
        ----------
        plain: bool, False
            If true then should not be colored.

        Returns
        -------
        string
            Formatted, transposed version of this instance; intended for display.

        Examples
        --------
        >>> st = rt.Struct({'a': 1, 'b': 'fish', 'c': [5.6, 7.8], 'd': {'A': 'david', 'B': 'matthew'},
        ... 'e': np.ones(7, dtype=np.int64)})
        >>> st
        #   Name    Type   Size      0     1   2
        -   ----   -----   ----   ----   ---   -
        0      a     int      0      1
        1      b     str      0   fish
        2      c    list      2    5.6   7.8
        3      d    dict      2      A     B
        4      e   int64      7      1     1   1
        [5 columns]
        >>> st.dtranspose()
        Fields:     0      1      2      3       4
        -------   ---   ----   ----   ----   -----
           Name     a      b      c      d       e
           Type   int    str   list   dict   int64
           Size     0      0      2      2       7
              0     1   fish    5.6      A       1
              1                 7.8      B       1
              2                                  1
        [5 columns]
        """
        prev_t_state = self._transpose_on

        # force transpose
        self._transpose_on = True

        display_type = self._get_final_display_mode(plain=plain)

        result_table = self.make_table(display_type)
        table_string = DisplayString(result_table)

        self._transpose_on = prev_t_state

        return table_string

    # -------------------------------------------------------
    @property
    def _T(self):
        '''
        Display transposed view of table; all columns will be shown as rows, will not be abbreviated.
        '''
        oldmaxwidth=TypeRegister.DisplayOptions.MAX_STRING_WIDTH
        if oldmaxwidth < 32:
            TypeRegister.DisplayOptions.MAX_STRING_WIDTH=32
        result = self.dtranspose()
        TypeRegister.DisplayOptions.MAX_STRING_WIDTH=oldmaxwidth
        return result

    # -------------------------------------------------------
    @property
    def _V(self):
        '''
        Display all rows (up to 10,000), instead of using ellipses.
        '''
        maxrows = 10_000
        numrows = self._nrows if hasattr(self, '_nrows') else len(self)
        if numrows < maxrows:
            return self._temp_display('ROW_ALL', True)
        else:
            raise ValueError(f"Dataset has more than 10,000 rows. Cannot display all rows. (to override, set Display.options.ROW_ALL = True and print with repr)")

    # -------------------------------------------------------
    @property
    def _H(self):
        '''
        Display all columns, allow long strings to be displayed.
        '''
        return self._temp_display(['COL_ALL', 'MAX_STRING_WIDTH'], [True, 1000])

    # -------------------------------------------------------
    @property
    def _G(self):
        '''
        Display all columns, allow long strings to be displayed.
        '''
        savemode = (DisplayTable.options.COL_ALL, DisplayTable.options.CONSOLE_X_HTML)
        DisplayTable.options.COL_ALL=True
        DisplayTable.options.CONSOLE_X_HTML=150
        print(self)
        DisplayTable.options.COL_ALL=savemode[0]
        DisplayTable.options.CONSOLE_X_HTML=savemode[1]
        return None

    # -------------------------------------------------------
    @property
    def _A(self):
        '''
        Display all columns and all rows, allow long strings to be displayed.
        '''
        maxrows = 10_000
        numrows = self._nrows
        if numrows > maxrows:
            raise ValueError(f"Dataset has more than 10,000 rows. Cannot display all rows. To override, set Display.options.ROW_ALL = True and print(ds)")
        return self._temp_display(['COL_ALL', 'ROW_ALL', 'MAX_STRING_WIDTH'], [True, True, 1000])

    # -------------------------------------------------------
    def _temp_display(self, option, value):
        '''
        Temporarily modify a display option when generating dataset display.
        User configured option (or default) will be restored after display string is generated.
        '''
        if not isinstance(option, list):
            option = [option]
        if not isinstance(value, list):
            value = [value]

        # store setting
        revert = [getattr(TypeRegister.DisplayOptions, opt) for opt in option ]
        for i, v in enumerate(value):
            setattr(TypeRegister.DisplayOptions, option[i], v)

        # build string (repr or html)
        display_type = self._get_final_display_mode()
        output = self.make_table(display_type)

        #restore setting
        for i, rev in enumerate(revert):
            setattr(TypeRegister.DisplayOptions, option[i], rev)
        return TypeRegister.DisplayString(output)


    # -------------------------------------------------------
    def _get_final_display_mode(self, plain=False):
        if TypeRegister.DisplayOptions.HTML_DISPLAY is False:
            display_type = DS_DISPLAY_TYPES.STR
        elif plain or TypeRegister.DisplayDetect.Mode == DisplayDetectModes.Console:
            display_type = DS_DISPLAY_TYPES.STR
        elif TypeRegister.DisplayDetect.Mode == DisplayDetectModes.HTML:
            display_type = DS_DISPLAY_TYPES.HTML
        elif TypeRegister.DisplayDetect.Mode == DisplayDetectModes.Jupyter:
            display_type = DS_DISPLAY_TYPES.HTML
        else:
            display_type = DS_DISPLAY_TYPES.REPR

        return display_type

    # -------------------------------------------------------
    def display_attributes(self):
        '''
        Returns a dict of display attributes, currently consisting of
        NumberOfFooterRows and a list of MarginColumns.

        Returns
        -------
        d : dict
            A dictionary of display attributes
        '''
        attribs = dict()
        # Footer rows
        attribs[TypeRegister.DisplayAttributes.NUMBER_OF_FOOTER_ROWS] =\
            getattr(self._all_items, ATTRIBUTE_NUMBER_OF_FOOTER_ROWS, 0)
        # Margin columns
        cols = []
        for name in self._all_items.keys():
            if self._all_items.item_get_attribute(name, ATTRIBUTE_MARGIN_COLUMN):
                cols.append(name)
        attribs[TypeRegister.DisplayAttributes.MARGIN_COLUMNS] = cols

        return attribs

    # -------------------------------------------------------
    def make_table(self, display_type):
        """
        Pretty-print code used by infrastructure.

        Parameters
        ----------
        display_type : rt.rt_enum.DS_DISPLAY_TYPES

        Returns
        -------
        obj or str
            Display object or string.
        """
        # __str__, __repr__, _repr_html_, T will all funnel into this
        # prepares row sorts and header tuples for rt_display
        table = TypeRegister.DisplayTable(attribs=self.display_attributes())
        from_str = False
        row_stats = ""

        # _repr_html_
        if display_type == DS_DISPLAY_TYPES.HTML:
            if TypeRegister.DisplayTable.FORCE_REPR is False:
                TypeRegister.DisplayDetect.Mode = DisplayDetectModes.HTML
            table._display_mode = DisplayDetectModes.HTML
            from_str = False
            row_stats = "\n\n" + self._last_row_stats()

        # __repr__
        elif display_type == DS_DISPLAY_TYPES.REPR:
            # this repr will be called before _repr_html_ in jupyter
            if TypeRegister.DisplayDetect.Mode == DisplayDetectModes.HTML:
                if TypeRegister.DisplayTable.FORCE_REPR is False:
                    return ""

            elif TypeRegister.DisplayDetect.Mode == DisplayDetectModes.Console:
                from_str = True

            row_stats = "\n\n" + self._last_row_stats()

        # __str__
        else:
            # no row/column/memory stats will be displayed
            from_str = True

        # only dataset has sort_view, it is off by default
        # it can be turned on with a call to sort_view
        if self._sort_display is True:
            # check to see if already in the sort cache
            sort_id = self.get_row_sort_info()
            sorted_row_idx = TypeRegister.SortCache.get_sorted_row_index(*sort_id)
            if self._sort_ascending is False:
                sorted_row_idx = sorted_row_idx[::-1]
        else:
            # remove all row sorts
            sorted_row_idx = None

        label_keys = self.label_as_dict()
        summary_cols = self.summary_as_dict()
        header_tups, main_data, footer_tups = self._prepare_display_data()

        restore_str_width = TypeRegister.DisplayOptions.MAX_STRING_WIDTH
        if hasattr(self, '_nrows'):
            nrows = self._nrows
        # number of rows is set to number of columns for structs
        else:
            nrows = self._ncols
            # increase max string width for structs
            TypeRegister.DisplayOptions.MAX_STRING_WIDTH = 32

        # check if accum2 had any badcols
        badcols = None
        if self._badcols is not None:
            bad_color = table.get_bad_color()
            badcols = { k: bad_color for k in self._badcols}

        badrows = None
        if self._badrows is not None:
            bad_color = table.get_bad_color()
            badrows = { k: bad_color for k in self._badrows}

        # send to rt_display to build a result string for console or html
        result = table.build_result_table(
            header_tups, # list of lists of ColHeader tuples
            main_data, # list of column arrays
            nrows, # number of rows ( dataset only )

            keys=label_keys,               # dictionary of groupby keys taht will appear on the far left
            row_numbers=self._row_numbers, # callback function to return custom left-hand table
            right_cols=summary_cols,       # dictionary of keys->arrays that will appear on the far right
            footer_tups=footer_tups,       # list of lists of ColHeader tuples

            sortkeys=self._col_sortlist,    # list of names of columns the table was sorted by
            sorted_row_idx=sorted_row_idx,  # fancy index for sort when table has a sorted view

            from_str=from_str,              # display called from str() or plain console
            transpose_on=self._transpose_on,# display called from ._T
            badrows=badrows,
            badcols=badcols,
            styles=self._styles,            # dictionary of column name -> ColumnStyle object **not implemented
            callback=getattr(self,'_display_callback',None)    )  # user can specify their own callback
                                            # callback signature must be callback(cols, stylefunc, rows=True)
                                            # from riptable.rt_display import DisplayColumnColors
                                            # callback(list of DisplayColumns, stylefunc, rows=True)
                                            # Example:
                                            #def makered(cols, stylefunc, rows=True):
                                            #    for col in cols:
                                            #        for cell in col:
                                            #            if cell.string.startswith('-'): cell.color=DisplayColumnColors.Red
                                            #ds._display_callback = makered

        TypeRegister.DisplayOptions.MAX_STRING_WIDTH = restore_str_width

        return result + row_stats

    # -------------------------------------------------------
    def set_display_callback(self, userfunc, scope=None):
        '''
        Set the user display callback for styling text.

        Parameters
        ----------
        userfunc: a callable function with the signature
                    def userfunc(cols, **kwargs):

        scope: default, None.  The callback for just this dataset, or all datasets.
                    can be None, 'Dataset', or 'Struct'
        Examples
        --------
        >>> from riptable.rt_display import DisplayColumnColors
        >>> def make_red(cols, **kwargs):
                location = kwargs['location']  # could left, right, or main
                if location == 'main':
                    for col in cols:
                        for cell in col:
                            if cell.string.startswith('-'): cell.string = '(' + cell.string[1:] + ')'; cell.color=DisplayColumnColors.Red
        >>> ds=rt.Dataset({'test':rt.arange(5)-3, 'another':rt.arange(5.0)-2})
        >>> ds.set_display_callback(make_red)
        >>> ds

        '''
        if not callable(userfunc):
            raise TypeError("The userfunc passed must be a callable function.")
        if scope is None:
            self._display_callback = userfunc
        elif scope == 'Dataset':
            TypeRegister.Dataset._display_callback = userfunc
        elif scope == 'Struct':
            TypeRegister.Struct._display_callback = userfunc
        else:
            raise ValueError("The 'scope' must be None, 'Dataset', or 'Struct'")

    # -------------------------------------------------------
    def _sort_column_styles(self, style):
        '''
        Callback to return sort-by columns.

        style : default sort style from display

        Returns dictionary of column name -> tuple( array, ColumnStyle )
        These columns will be moved to the left of the table.
        '''
        cols = {}
        sort_style = ColumnStyle(color=DisplayColumnColors.Sort)
        for name in self._col_sortlist:
            cols[name] = ( self[name], sort_style )
        return cols

    # -------------------------------------------------------
    @property
    def _styles(self):
        '''
        Subclasses can return a callback function which takes no arguments
        Returns dictionary of column names -> ColumnStyle objects
        '''
        if hasattr(self, '_column_styles'):
            return self._column_styles


    # -------------------------------------------------------
    @property
    def _row_numbers(self):
        '''
        Subclasses can define their own callback function to customize the left side of the table.
        If not defined, normal row numbers will be displayed

        Parameters
        ----------
        arr : array
            Fancy index array of row numbers
        style : `ColumnStyle`
            Default style object for final row numbers column.

        Returns
        -------
        header : string
        label_array : ndarray
        style : `ColumnStyle`
        '''
        return None

    # -------------------------------------------------------
    @property
    def _sort_columns(self):
        '''
        Subclasses can define their own callback function to return columns they were sorted by, and styles.
        Callback function will receive trimmed fancy index (based on sort index) and return a dictionary of column headers -> (masked_array, ColumnStyle objects)
        These columns will be moved to the left side of the table (but to the right of row labels, groupbykeys, row numbers, etc.)
        '''
        return None

    # -------------------------------------------------------
    def __str__(self):
        return self.make_table(DS_DISPLAY_TYPES.STR)

    # -------------------------------------------------------
    def __repr__(self):
        Struct._lastrepr =GetTSC()
        if hasattr(self, '_repr_override'):
            return self._repr_override(self)
        if TypeRegister.DisplayOptions.HTML_DISPLAY is False:
            return self.make_table(DS_DISPLAY_TYPES.STR)
        # this will be called before _repr_html_ in jupyter
        return self.make_table(DS_DISPLAY_TYPES.REPR)

    # -------------------------------------------------------
    def _repr_html_(self):
        Struct._lastreprhtml =GetTSC()
        if TypeRegister.DisplayOptions.HTML_DISPLAY is False:
            plainstring = self.make_table(DS_DISPLAY_TYPES.STR)
            print(DisplayString(plainstring))
            # jupyter lab will turn plain string into non-monospace font
            return ""
        return self.make_table(DS_DISPLAY_TYPES.HTML)

    # -------------------------------------------------------
    @property
    def footers(self):
        '''
        Returns the footer attributes.

        For example, Accum2 and AccumTable objects can have footers.
        '''
        try:
            footers = getattr(self, '_footers')
        except:
            footers = None
        return footers

    # -------------------------------------------------------
    def _prepare_display_data(self):
        """
        Returns a list of lists (all column data) and a list of header tuples for display.

        :return: list(list), list(tuple)
        """
        # move this to a class default variable or display option
        item_numbers = [str(i) for i in range(self._summary_len)]
        summary_labels = ['Name', 'Type', 'Size']
        summary_headers = summary_labels + item_numbers
        header_tups = [build_header_tuples(summary_headers, 1, 0)]
        # structs will always show row numbers
        header_tups[-1].insert( 0, ColHeader("#",1,0) )

        footer_tups = self.footers

        names = np.array(list(self.keys()))
        types = np.array([type(a).__name__ for a in names])

        types = []
        sizes = []
        summary_items = []
        for a in names:
            item_type = ""
            size = 0
            items = ["", "", ""]
            data = self.__getattr__(a)

            if isinstance(data, np.ndarray):
                #if data.ndim!=1:print(f'{a} did not have 2 dims got {data.ndim}')
                item_type = str(data.dtype)
                try:
                    size = len(data)
                except:
                    # handle unsized objects happens with x=np.asanyarray('test')
                    data=np.asanyarray([data])
                    size = len(data)

                if isinstance(data, TypeRegister.Categorical):
                    items = np.array([str(i) for i in data[:self._summary_len]])
                else:
                    # bug when sending 2dim columns to struct display
                    # cannot hstack string repr with column name
                    try:
                        if data.ndim == 1:
                            items = data[:self._summary_len].astype(str)
                        else:
                            items = np.array([str(i) for i in data[:self._summary_len]])
                    except:
                        # fix bug in record arrays
                        items = np.array([str(i) for i in data[:self._summary_len]])


                if size < self._summary_len:
                    # fill the rest with empty strings
                    right = np.full(self._summary_len - size, "")
                    items = np.hstack((items, right))

            elif isinstance(data, TypeRegister.Dataset):
                item_type = type(data).__name__
                num_rows = str(data._nrows)
                num_cols = str(data._ncols)
                size = num_rows+" rows x "+num_cols+" cols"
                items = np.full(self._summary_len, "")

            elif isinstance(data, Struct):
                item_type = type(data).__name__
                size = data._ncols
                data = list(data.keys())
                if size >= self._summary_len:
                    data = data[:self._summary_len]
                else:
                    left = np.array(data, dtype=str)
                    right = np.full(self._summary_len - size, "")
                    items = np.hstack((left, right))

            else:
                item_type = type(data).__name__
                # convert dict keys and sets to list for slicing
                if isinstance(data, dict):
                    data = list(data.keys())
                elif isinstance(data, (set, tuple)):
                    data = list(data)


                if isinstance(data, list):
                    size = len(data)
                    items = data[:self._summary_len]
                    if size < self._summary_len:
                        diff = self._summary_len - size
                        for i in range(diff):
                            items.append("")

                elif np.isscalar(data):
                    left = [str(data)]
                    right = [""] * (self._summary_len - 1)
                    items = left + right

            types.append(item_type)
            sizes.append(size)
            summary_items.append(items)

        main_data = [np.array(names, dtype=str), np.array(types, dtype=str),
                     np.array(sizes, dtype=str)]

        summary_items = np.array(summary_items)

        summary_items = summary_items.transpose()
        for s in summary_items:
            main_data.append(s)

        return header_tups, main_data, footer_tups

    # ------------------------------------------------------------
    def apply_schema(self, schema):
        """Apply a schema containing descriptive information recursively to
        the Struct.

        Parameters
        ----------
        schema : dict
            A dictionary of schema information.  See
            :func:`.rt_meta.apply_schema` for more
            information on the format of the dictionary.

        Returns
        -------
        res : dict
            Dictionary of deviations from the schema

        See Also
        --------
        info, doc, :func:`.rt_meta.apply_schema`
        """
        from .rt_meta import apply_schema as _apply_schema
        return _apply_schema(self, schema)

    def info(self, **kwargs):
        """Return an object containing a description of the structure's contents.

        Parameters
        ----------
        kwargs : dict
            Optional keyword arguments passed to :func:`.rt_meta.info`

        Returns
        -------
        info : :class:`.rt_meta.Info`
            A description of the structure's contents.

        """
        from .rt_meta import info as _info
        return _info(self, **kwargs)

    @property
    def doc(self):
        """:class:`.rt_meta.Doc` The descriptive documentation object for the structure."""
        from .rt_meta import doc as _doc
        return _doc(self)

    # ------------------------------------------------------------
    # cannot be class method here
    @staticmethod
    def set_fast_array(val: bool):
        """Set to true to force the casting of numpy arrays to FastArray when
        constructing a Struct or adding a new column.

        Parameters
        ----------
        val : bool
            True or False
        """
        Struct.UseFastArray = val

    @property
    def total_sizes(self) -> Tuple[int, int]:
        """
        The total physical and logical size of all (columnar) data in bytes within this Struct.

        Returns
        -------
        total_physical_size : int
            The total size, in bytes, of all columnar data in this instance, not counting any duplicate/alias object instances.
        total_logical_size : int
            The total size, in bytes, of all columnar data in this instance, including duplicate/alias object instances.
            This value is always at least as large as `total_physical_size`.
        """

        import numbers
        from collections.abc import Hashable

        class IdWrapper(Hashable):
            """
            Wrapper type around an object instance which uses physical equality to check whether
            an object is the _same_ (not equivalent, but the same) instance as another.
            Used for detecting duplicate/aliased objects.
            """

            __slots__ = ['obj']
            def __init__(self, obj):
                self.obj = obj

            def __hash__(self):
                return id(self.obj)

            def __eq__(self, other):
                return isinstance(other, IdWrapper) and id(self.obj) == id(other.obj)

        # A list of objects which still need to be processed.
        # The algorithm below doesn't care much about traversal order, and using this list to
        # handle nested objects (e.g. Struct, list, dict) avoids the need for a recursive solution that'd enforce a strict DFS traversal order.
        pending_objects = [self]

        # A set of objects we've seen while traversing the Struct's contents.
        # Only "complex" objects are included here -- objects of scalar types like integer and float values won't be included
        # because there's nothing meaningful we could do to de-duplicate them.
        seen_objects: Set[IdWrapper] = set()

        # Physical size: the actual size, in bytes, the data is taking up in memory.
        # i.e., duplicate/aliased objects are not included in this tally.
        total_physical_size = 0

        # Logical size: the size, in bytes, the data is "logically" taking up in memory;
        # i.e. the size that would be occupied by the Struct in memory if any duplicated/aliased
        # objects were to be replaced by a deep-copy (at which point all objects within the struct
        # would be unique instances).
        total_logical_size = 0

        # Iterate until we've processed all the Struct instances (which includes any instances of derived classes)
        while len(pending_objects) > 0:
            current_obj = pending_objects.pop()
            for name, obj in current_obj.items():
                if obj is None: pass
                elif isinstance(obj, TypeRegister.Categorical):
                    # Categorical needs to include the storage for categories.
                    size = obj._total_size
                    total_logical_size += size
                    while obj.base is not None:
                        obj = obj.base
                    col_idwrapper = IdWrapper(obj)
                    if col_idwrapper not in seen_objects:
                        seen_objects.add(col_idwrapper)
                        total_physical_size += size
                elif isinstance(obj, TypeRegister.FastArray):
                    # For FastArray, we can have different FastArray instances which wrap the same underlying memory;
                    # since that's really what we're after here -- to see where we have aliased arrays -- we check
                    # that underlying array object for uniqueness rather than the FastArray itself.
                    array_data = obj._np

                    # Handle array views by searching until we find the "root" base array.
                    # An ndarray is a view if it returns a non-None instance from it's .base property.
                    while array_data.base is not None:
                        array_data = array_data.base

                    # Create an IdWrapper around the column data.
                    col_idwrapper = IdWrapper(array_data)

                    # Always add this object's size to the total logical size of the Struct data
                    # (without regard to whether this is a duplicate/aliased object).
                    total_logical_size += array_data.nbytes       # assumes data is an np.ndarray or a subclass of it

                    # If the data hasn't been seen yet (i.e. it's not an alias of some other data we've seen before)
                    # add it to the "seen objects" set and add it's size to the total physical size.
                    if col_idwrapper not in seen_objects:
                        seen_objects.add(col_idwrapper)
                        total_physical_size += array_data.nbytes

                elif isinstance(obj, np.ndarray):
                    # Handle array views by searching until we find the "root" base array.
                    # An ndarray is a view if it returns a non-None instance from it's .base property.
                    while obj.base is not None:
                        obj = obj.base

                    # Create an IdWrapper around the column data.
                    col_idwrapper = IdWrapper(obj)

                    # Always add this object's size to the total logical size of the Struct data
                    # (without regard to whether this is a duplicate/aliased object).
                    total_logical_size += obj.nbytes       # assumes data is an np.ndarray or a subclass of it

                    # If the data hasn't been seen yet (i.e. it's not an alias of some other data we've seen before)
                    # add it to the "seen objects" set and add it's size to the total physical size.
                    if col_idwrapper not in seen_objects:
                        seen_objects.add(col_idwrapper)
                        total_physical_size += obj.nbytes

                # If the current item is a Struct, recurse downwards.
                # This _must_ be the last case before the fallthrough so that any classes
                # derived from Struct are handled in some more-specific way before taking
                # the general case for Struct.
                elif isinstance(obj, Struct):
                    # Create an IdWrapper around the column data.
                    col_idwrapper = IdWrapper(obj)

                    # If the data hasn't been seen yet (i.e. it's not an alias of some other data we've seen before)
                    # add it to the "seen objects" set and add to the list of pending Struct instances to process.
                    if col_idwrapper not in seen_objects:
                        seen_objects.add(col_idwrapper)
                        pending_objects.append(obj)

                # Handle some scalar types
                elif isinstance(obj, numbers.Integral):
                    # TODO: Do we need to handle integer sizes other than 32-bit here?
                    #       We may need to if `obj` is a numpy array scalar.
                    total_physical_size += 4
                    total_logical_size += 4

                # Process list and sets by just concatenating their items to the list of pending items.
                # TODO: Need to implement this -- we'll need to restructure our work loop a bit to accommodate them.
                # elif isinstance(obj, (list, set)):
                #     pending_objects.extend(obj)

                else:
                    # Log the object type so we know what's not being handled.
                    logger.debug(f'set_memory_stats: Unhandled object \'{name}\' of type \'{type(obj)}\'.')

        return (total_physical_size, total_logical_size)


# keep this as the last line
TypeRegister.Struct = Struct
