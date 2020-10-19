__all__ = [
    # Classes/types
    'Categorical',
    'Categories',
    # functions
    'CatZero',
    'categorical_convert',
    'categorical_merge_dict',
]

from enum import IntEnum, EnumMeta
from typing import Any, Collection, Dict, List, Mapping, Optional, Tuple, Union, TYPE_CHECKING
import warnings

import numpy as np

from .rt_fastarray import FastArray
from .rt_grouping import Grouping, GroupingEnum, merge_cats
from .rt_utils import bytes_to_str, crc_match
from .rt_enum import (
    NumpyCharTypes, TypeRegister, TypeId, DisplayLength,
    INVALID_SHORT_NAME, DisplayJustification, INVALID_DICT, FILTERED_LONG_NAME,
    int_dtype_from_len,
    CategoryMode, CategoryStringMode, CategoricalOrigin, CategoricalConstructor, SDSFlag, GB_FUNCTIONS)
from .rt_numpy import (
    mask_or, mask_and, mask_ori, mask_andi, sort, unique32, ismember, unique, argsort, zeros,
    bool_to_fancy, full, empty, sum, putmask, nan_to_zero, issorted, crc64, hstack, isnan, ones)
from .rt_hstack import hstack_any
from .Utils.rt_display_properties import ItemFormat, DisplayConvert, default_item_formats
from .Utils.rt_metadata import MetaData

# groupby imports
from .rt_groupbyops import GroupByOps
from .rt_groupbykeys import GroupByKeys
from .rt_str import FAString
from .rt_fastarraynumba import fill_forward, fill_backward

if TYPE_CHECKING:
    import pandas as pd
    from .rt_dataset import Dataset


# ------------------------------------------------------------
def _copy_name(src: FastArray, dst: FastArray) -> None:
    """Copy a name from FastArray if it has been set.
    If not a FastArray, no name will be set.
    """
    try:
        name = src.get_name()
    except:
        name = None

    if name is not None:
        try:
            dst._name = name
        except:
            pass


# ------------------------------------------------------------
def categorical_convert(v: 'pd.Categorical', base_index: int = 0) -> Tuple[FastArray, np.ndarray]:
    '''
    Parameters
    ----------
    v: a pandas categorical

    Returns
    -------
    Returns the two building blocks to make an rt categorical: integer array, and what that indexes into
    whatever the pandas categorical underlying object is we try to convert it to a string to
    detach from object references and free of pandas references

    pandas also uses -1 to indicate an out of bounds value, when we detect this, we insert an item in the beginning

    Examples
    --------
    >>> p=pd.Categorical(['a','b','b','a','a','c','b','c','a','a'], categories=['a','b'])
    >>> test=Categorical(p)

    from a cut

    >>> a=rt.FA(rt.arange(10.0)+.1)
    >>> p=pd.cut(a,[0,3,6,7])
    (0, 3], (0, 3], (3, 6], (3, 6], (3, 6], (6, 7], NaN, NaN, NaN]
    >>> test=Categorical(p)
    Categorical([(0, 3], (0, 3], (0, 3], (3, 6], (3, 6], (3, 6], (6, 7], nan, nan, nan])
    '''
    int_array = FastArray(v._codes)
    string_array = make_string_array(v.categories._values)

    # invalid indices will be converted to zero invalid
    # this will always get hit
    if base_index != 0:
        return int_array + base_index, string_array

    minval = np.min(int_array)

    if minval < -1:
        raise TypeError("pandas categorical has an index below -1")

    if minval == -1:
        if string_array.dtype.char in ['V']:
            print("!!!Warning, -1 index exists but do not know how to add an invalid object")

        # shift all indexes by 1 as we will insert a new index
        int_array = int_array + 1
        sa_len = string_array.shape[0]
        string_array = np.resize(string_array, (sa_len + 1))
        string_array[1:(sa_len + 1)] = string_array[0:sa_len]

        # need to make decisions on what to insert here
        if (string_array.dtype.char == 'S'):
            string_array[0] = INVALID_SHORT_NAME
            if (string_array.itemsize == 1):
                string_array[0] = b'\0'

            # string_array[0]=v[-1]

        elif (string_array.dtype.char == 'U'):
            string_array[0] = INVALID_SHORT_NAME
            if (string_array.itemsize < 12):
                string_array[0] = '\0'

            # string_array[0]=v[-1]
        else:
            print("!!!Warning, cannot create invalid entry")
            # try anyway
            string_array[0] = v[-1]

    return int_array, string_array


# ------------------------------------------------------------
def make_string_array(categories):
    '''
    *** TODO: remove after testing new Categories class
    systematically try to convert whatever is in the list to bytes, then unicode, then object
    '''

    string_array = np.asanyarray(categories)
    # try to convert to bytes first
    if string_array.dtype.char in ['O', 'U']:
        try:
            string_array = string_array.astype('S')
        except:
            pass

    # then try to convert to unicode if that failed
    if string_array.dtype.char in ['O']:
        try:
            string_array = string_array.astype('U')
        except:
            print("!!!Warning: unable to convert object type to string.")

    return string_array


class Categories:
    '''
    Holds categories for each Categorical instance. This adds a layer of abstraction to Categorical.

    Categories objects are constructed in Categorical's constructor and other internal routines such as merging operations.
    The Categories object is responsible for translating the values in the Categorical's underlying fast array
    into the correct bin in the categories. It performs different operations to retrieve the correct bins based on it's mode.

    Parameters
    ----------
    categories
        main categories data - can also be empty list
    invalid_category : str
        string that will be displayed for an invalid index
    invalid_index
        sentinel value for a particular index; this invalid will be displayed differntly in IntEnum/Dictionary modes
    ordered : bool
        flag for list list modes, ordered categories can use a binary search for finding bins
    auto_add_categories
        if a setitem (bracket-indexing with a value) is called, and the value is not in the categories, this flag allows it to be added automatically.
    na_added
        for some constructors, the calling Categorical has already added the invalid category
    base_index
        the calling Categorical passes in the index offset for list and grouping modes
    multikey
        the categories information is stored in a multikey dictionary *up for deletion*
    groupby
        *possibly merge with the multikey flag*

    Notes
    -----
    There are multiple modes in which a Categories object can operate.

    **StringArray**: *(list_modes)*
    Two paths for initializations use the categories routines: TB Filled in LATER
    array and list of unique categories.
    String mode will be set to unicode or bytes so the correct encoding/decoding can be performed before comparison/searching operations.
    - from list of strings (unique/ismember)
    - from list of strings paired with unique string categories (unique/ismember)
    - from codes paired with unique string categories (assignment will happen without unique/ismember)
    - from pandas categoricals (with string categories) (assignment will happen without unique/ismember)
    - from matlab categoricals (with string categories) (assignment will happen without unique/ismember)

    **NumericArray:** *(list_modes)* this is not currently implemented as default behavior, but if enabled it will handle these constructors
    - from list of integers
    - from list of floats
    - from codes paired with unique integer categories
    - from codes paired with unique float categories
    - from list of floats paired with unique float categories
    - from pandas categoricals with numeric categories

    **IntEnum / Dictionary:** *(dict_modes)*
    Two dictionaries will be held: one mapping strings to integers, another mapping integers to strings.
    This mode requires that all strings and their corresponding codes are one-to-one.
    - from codes paired with IntEnum object
    - from codes paired with Integer -> String dictionary
    - from codes paired wtih String -> Integer dictionary *not implemented*

    **Grouping**
    All categories objects in Grouping mode hold categories in a dictionary, even if the dictionary only contains one item.
    Information for indexed items will appear in a tuple if multiple columns are being held.
    - from list of key columns
    - from dictionary of key columns
    - from single list of numeric type
    - from dataset *not implemented*
    '''
    default_colname = "key_0"
    multikey_spacer = " "

    list_modes    = [ CategoryMode.StringArray, CategoryMode.NumericArray ]
    dict_modes    = [ CategoryMode.IntEnum, CategoryMode.Dictionary ]
    string_modes  = [ CategoryMode.StringArray, CategoryMode.IntEnum, CategoryMode.Dictionary ]
    numeric_modes = [ CategoryMode.NumericArray ]

    # ------------------------------------------------------------------------------
    def __init__(self, *args, base_index=1, invalid_category=None, ordered=False, unicode=False, _from_categorical=False, **kwargs):

        self._list = []
        self._column_dict = {}

        self._ordered = ordered
        self._sorted = ordered
        self._auto_add_categories = False
        self._name = None

        # any values that were filtered will STILL appear in the unique categories
        # any item with a 0 index will be shown as 'Filtered' (also add a method to change this string)
        # if holding an invalid category, operations like isnan() will look up its index, return bool, else all False
        self._invalid_category = invalid_category
        self._filtered_name = FILTERED_LONG_NAME

        if _from_categorical:
            return

        if len(args) == 1:
            # list modes
            if isinstance(args[0], np.ndarray):
                self._list = args[0]
                # preserve the name of the input array
                try:
                    self._name = self._list._name
                except:
                    pass
                # check if list is string / numeric for correct indexing
                typechar = self._list.dtype.char
                if typechar in NumpyCharTypes.AllInteger+NumpyCharTypes.AllFloat+'?':
                    self._mode = CategoryMode.NumericArray

                    # will always use nan/sentinel for numeric categoricals
                    # ***changed behavior of invalid category
                    if invalid_category is not None:
                        if not np.isreal(invalid_category):
                            self._invalid_category = INVALID_DICT[self._list.dtype.num]
                            warnings.warn(f"invalid_category was set to {invalid_category} - non-numeric/real value. Using sentinel {self._invalid_category} instead.")
                        else:
                            self._invalid_category = invalid_category

                elif typechar in 'US':
                    self._mode = CategoryMode.StringArray

                else:
                    raise ValueError(f"Can't construct categories array with dtype {self._list.dtype}")

                # last spot to flip ALL category arrays to FastArray
                if not isinstance(self._list, FastArray):
                    self._list = FastArray(self._list, unicode=unicode)

            # multikey
            elif isinstance(args[0], dict):
                self._column_dict = args[0]
                self._mode = CategoryMode.MultiKey

                for name, col in self._column_dict.items():
                    if not isinstance(col, FastArray):
                        self._column_dict[name] = FastArray(col, unicode=unicode)

            # probably won't be hit - Categories constructor should only be called internally
            else:
                raise TypeError(f"Don't know how to construct categories from single argument of {type(args[0])}")

        elif len(args) == 2:
            # two mapped dictionaries
            if isinstance(args[0], dict):
                self._int_to_str_dict = args[0]
                self._str_to_int_dict = args[1]
                self._max_int = max(self._int_to_str_dict.keys())

            # two numpy arrays for dictionaries (restoring from load)
            elif isinstance(args[0], np.ndarray):
                self._int_to_str_dict = dict(zip(args[0], args[1]))
                self._str_to_int_dict = dict(zip(args[1], args[0]))
                self._max_int = max(args[0])

            else:
                raise TypeError(f"Two arguments were not dictionaries or arrays.")
            self._mode = CategoryMode.Dictionary

        else:
            raise ValueError(f"Received {len(args)} inplace arguments in Categories constructor.")

    # ------------------------------------------------------------------------------
    @classmethod
    def from_grouping(cls, grouping, invalid_category=None):
        ordered = grouping.isordered
        # grouping has already flipped bytes to unicode, or kept unicode
        unicode = True

        if grouping.isenum:
            # add a public method to get to the GroupingEnum object
            cats = Categories( grouping._enum._int_to_str_dict, grouping._enum._str_to_int_dict, invalid_category=invalid_category)
        else:
            if len(grouping.uniquedict)>1:
                cats = Categories(grouping.uniquedict, unicode=unicode, ordered=ordered, invalid_category=invalid_category)
            else:
                cats = Categories([*grouping.uniquedict.values()][0], unicode=unicode, ordered=ordered, invalid_category=invalid_category)

        # copying attributes now - after the move these will only need to get checked in grouping
        cats._grouping = grouping.copy(deep=False)

        return cats

    # ------------------------------------------------------------------------------
    @property
    def name(self):
        return self._name

    # ------------------------------------------------------------------------------
    @property
    def ncols(self):
        """
        Returns the number of key columns in a multikey categorical or 1 if a single key's categories
        are being held in a dictionary.
        """
        return len(self.uniquedict)

    # ------------------------------------------------------------------------------
    @property
    def nrows(self):
        """
        Returns the number of unique categories in a multikey categorical.
        """
        return len(self.uniquelist[0])

    # ------------------------------------------------------------------------------
    def __len__(self):
        """
        TODO: consider changing length of enum/dict mode categories to be the length of the dictionary.
        using max int so the calling Categorical can properly recast the integer array.
        """
        if self.isenum:
            return self._max_int
        elif self.issinglekey or self.ismultikey:
            return self.nrows
        else:
            raise TypeError(f'Critical error in Categories length. Mode was {CategoryMode(self.mode).name}')

    # ------------------------------------------------------------------------------
    def _copy(self, deep=True):
        """
        Creates a new categories object and possibly performs a deep copy of category list.
        Currently only supports Categories in list modes.
        """
        c = self.__class__([], _from_categorical=True)
        #TJD c._grouping = self.grouping.copy(deep=deep)
        c._mode = self._mode
        c._ordered = self._ordered
        c._sorted = self._sorted
        c._auto_add_categories = self._auto_add_categories
        c._name = self._name
        c._filtered_name = self._filtered_name

        if self.isenum:
            if deep:
                c._str_to_int_dict = self.str2intdict.copy()
                c._int_to_str_dict = self.int2strdict.copy()
            else:
                c._str_to_int_dict = self.str2intdict
                c._int_to_str_dict = self.int2strdict
        elif self.issinglekey or self.ismultikey:
            c._column_dict = self._column_dict.copy()
            if deep:
                c._list = self._list.copy()
                for k, v in c._column_dict.items():
                    c._column_dict[k]=v.copy()
            else:
                c._list = self._list

        return c

    # ------------------------------------------------------------------------------
    def copy(self, deep=True):
        """
        Wrapper for internal _copy.
        """
        return self._copy(deep=deep)

    # ------------------------------------------------------------------------------
    def get_categories(self):
        """
        TODO: decide what to return for int enum categories. for now returning list of category strings
        """
        if self.issinglekey:
            return self.uniquelist[0]
        elif self.ismultikey:
            return self.uniquedict
        elif self.isenum:
            return list(self.str2intdict.keys())
        else:
            raise ValueError(f'Critical error in get_categories. Mode was {CategoryMode(self.mode).name}.')

    # THESE PROPERTIES WILL BE RETRIEVED FROM GROUPING -----------------------------
    # ------------------------------------------------------------------------------
    @property
    def grouping(self):
        return self._grouping

    @property
    def str2intdict(self):
        return self.grouping._enum._str_to_int_dict

    @property
    def int2strdict(self):
        return self.grouping._enum._int_to_str_dict

    @property
    def uniquedict(self):
        return self.grouping.uniquedict

    @property
    def uniquelist(self):
        return self.grouping.uniquelist

    @property
    def issinglekey(self):
        """True if unique dict holds single array.
        False if unique dict hodls multiple arrays or in enum mode.
        """
        if self.isenum:
            return False
        return len(self.grouping.uniquedict) == 1

    @property
    def ismultikey(self):
        """True if unique dict holds multiple arrays.
        False if unique dict holds single array or in enum mode.
        """
        if self.isenum:
            return False
        return len(self.grouping.uniquedict) > 1

    @property
    def isenum(self):
        """True if uniques have an enum / dictionary mapping for uniques.
        Otherwise False.

        See also: GroupingEnum
        """
        return self.grouping.isenum

    @property
    def isunicode(self):
        """True if uniques are held in single array of unicode.
        Otherwise False.
        """
        if self.issinglekey:
            return self.uniquelist[0].dtype.char == 'U'
        return False

    @property
    def isbytes(self):
        """True if uniques are held in single array of bytes.
        Otherwise False.
        """
        if self.issinglekey:
            return self.uniquelist[0].dtype.char == 'S'
        return False

    @property
    def base_index(self):
        return self.grouping.base_index

    # ------------------------------------------------------------------------------
    @property
    def mode(self):
        return self._mode

    # ------------------------------------------------------------------------------
    def _possibly_add_categories(self, new_categories):
        """
        Add non-existing categories to categories. If categories were added, an array is returned to fix the old indexes.
        If no categories were added, returns None.
        """

        fix_index = None

        if self.issinglekey:
            # force list like
            if self._mode == CategoryMode.StringArray:
                new_categories = self.match_str_to_category(new_categories)
            if not isinstance(new_categories, (np.ndarray, list)):
                new_categories = [new_categories]

            # collect non-existing categories
            cats_to_add = []
            for c in new_categories:
                if c not in self._list:
                    cats_to_add.append(c)

            # uniquify and sort
            if len(cats_to_add) > 0:
                if self._auto_add_categories:
                    all_together = hstack([self._list, cats_to_add])
                    self._list, fix_index = unique(all_together, return_inverse=True)
                    if self.isunicode:
                        self._list = self._list.astype('U', copy=False)
                    self._ordered = True
                    self._sorted = True
                else:
                    raise ValueError(f"Cannot automatically add categories {cats_to_add} while auto_add_categories is set to False. Set flag to True in Categorical init.")

            return fix_index

        elif self.isenum:
            raise NotImplementedError(f"Add categories not supported for {self._mode}.")

        else:
            raise NotImplementedError(f"Add categories not supported for {self._mode}.")

    # -----------------------------------------------------------
    def match_str_to_category(self, fld):
        """
        If necessary, convert the string or list of strings to the same type as the categories so
        that correct comparisons can be made.
        """
        # single item
        if isinstance(fld, (bytes, str)):
            if self.isbytes:
                if not isinstance(fld, bytes):
                    try:
                        fld = fld.encode('ascii')
                    except UnicodeEncodeError:
                        raise TypeError(f"Unable to convert unicode string to bytes.")
            elif self.isunicode:
                if not isinstance(fld, str):
                    fld = fld.decode()

        # list/array NOTE: this isn't very fast as it allocates a new numpy array if necessary
        elif isinstance(fld, (list, np.ndarray)):
            if isinstance(fld, list):
                fld = np.array(fld)

            if fld.dtype.char in ('U','S'):
                if self.isbytes:
                    if fld.dtype.char != 'S':
                        try:
                            fld = fld.astype('S')
                        except UnicodeEncodeError:
                            raise TypeError(f"Unable to convert unicode string to bytes.")
                elif self.isunicode:
                    if fld.dtype.char != 'U':
                        fld = fld.astype('U')
            else:
                raise TypeError(f"Categories cannot be selected with array of unknown type {fld.dtype}")

        else:
            raise TypeError(f"{fld} was not a valid string or list of strings to match to categories")

        return fld

    # -----------------------------------------------------------
    def get_multikey_index(self, multikey):
        """
        Multikey categoricals can be indexed by tuple.
        This is an internal routine for getitem, setitem, and logical comparisons.
        Valid return will be adjusted for the base index of the categorical (currently always 1 for multikey)

        Parameters
        ---------
        multikey: tuple of items to search for in multiple columns

        Returns
        -------
        int
            location of multikey + base index, or -1 if not found

        Examples
        --------
        >>> c = rt.Categorical([rt.arange(5), rt.arange(5)])
        >>> c
        Categorical([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]) Length: 5
          FastArray([1, 2, 3, 4, 5], dtype=int8) Base Index: 1
          {'key_0': FastArray([0, 1, 2, 3, 4]), 'key_1': FastArray([0, 1, 2, 3, 4])} Unique count: 5

        >>> c._categories_wrap.get_multikey_index((0,0))
        1
        """
        if len(multikey) != self.ncols:
            raise ValueError(f"This categorical has {self.ncols} key columns. Cannot be compared to tuple of {len(multikey)} values.")

        masks = []
        index = -2
        mk_cols = list(self.uniquedict.values())
        for col_idx, key_item in enumerate(multikey):
            current_col = mk_cols[col_idx]
            col_type = current_col.dtype.char
            search_item = key_item

            # match string to column's data if necessary
            if col_type in ('U','S'):
                if isinstance(key_item, str):
                    if col_type == 'S':
                        search_item = key_item.encode('ascii')
                elif isinstance(key_item, bytes):
                    if col_type == 'U':
                        search_item = key_item.decode()
                else:
                    # exit early, non string being compared to string column
                    return index+self.base_index

            masks.append(mk_cols[col_idx] == search_item)

        masks = mask_and(masks)
        found = bool_to_fancy(masks) #np.where(masks == True)[0]
        if len(found)>0:
            # safe because multikeys will always be unique
            index = found[0]

        return index+self.base_index

    # -----------------------------------------------------------
    def get_category_index(self, s):
        """
        Returns an integer or float for logical comparisons with the Categorical's index array.
        Floating point return ensures that LTE/GTE functions work properly
        """
        if isinstance(s, tuple):
            return self.get_multikey_index(s)

        str_idx = None
        if self.issinglekey:
            if self._mode == CategoryMode.StringArray:
                s = self.match_str_to_category(s)
            # sorted categories
            if self._sorted:
                # if larger than all strings, str_idx will be len(self._categories)
                str_idx = np.searchsorted(self._first_list, s)
                if str_idx < len(self._first_list):
                    # insertion point, not exact match
                    if s != self._first_list[str_idx]:
                        if Categorical.DebugMode: print("***no match")
                        # adjust for le, ge comparisons
                        # str_idx -= 0.5
                        str_idx -= 0.5
                str_idx += self.base_index

            # unsorted categories
            else:
                str_idx = bool_to_fancy(self._first_list == s)
                if len(str_idx) != 0:
                    str_idx = str_idx[0]+self.base_index  # get value from array
                else:
                    str_idx = len(self._first_list)+self.base_index

        elif self.isenum:
            s = self.match_str_to_category(s)
            str_idx = self.str2intdict.get(s, None)
            if str_idx is None:
                raise ValueError(f"{s} was not a valid category in categorical from mapping.")
        else:
            raise ValueError(f"{s} was not a valid category in categorical from mapping.")

        return str_idx

    # ------------------------------------------------------------
    def get_category_match_index(self, fld):
        """
        Returns the indices of matching strings in the unique list.
        The Categorical instance will compare these integers to those in its underlying array to generate a boolean mask.
        """
        if self.issinglekey:
            if isinstance(fld, (str, bytes, int, np.integer, float, np.float)):
                fld = [fld]
            string_matches = []
            for s in fld:
                str_idx = self.get_category_index(s)
                if isinstance(str_idx, (int, np.integer)) and str_idx < len(self._first_list)+self.base_index:
                    string_matches.append(str_idx)

            return string_matches
        else:
            raise NotImplementedError(f"Categories can only return boolean mask from string list, current mode is {self._mode}")

    # ------------------------------------------------------------
    @property
    def _first_list(self):
        """
        Returns the first column when categories are in a dictionary, or the list if the categories are in a list mode.
        """
        if self.mode in Categories.list_modes:
            return self._list

        return self._column_dict[list(self._column_dict.keys())[0]]

    # ------------------------------------------------------------
    def possibly_invalid(self, value):
        """
        If the calling categorical's values are set to a bad index, the !<badindex> will be returned.
        If the bad index is the sentinel value for that integer type, !<inv> will be returned
        """
        # TODO: reduce this routine... ran into trouble with multikey __repr__
        invalid_str = None

        if self.isenum:
            if value not in self.int2strdict:
                # bad code
                # use custom invalid string or generate one
                if self._invalid_category is not None:
                    invalid_str = self._invalid_category
                else:
                    invalid_str = f"!<{str(value)}>"

        else:
            if self.base_index == 1:
                if value == 0:
                    # filtered value
                    invalid_str = self._filtered_name
                elif value < 0 or value > len(self):
                    # bad index
                    invalid_str = f"!<{str(value)}>"
            else:
                if value < 0 or value >= len(self):
                    # bad index
                    invalid_str = f"!<{str(value)}>"

        # if invalid_str is still None, index/code will be looked up in uniques
        return invalid_str

    # ------------------------------------------------------------
    def _getitem_multikey(self, value):
        # ------------------------------------------------------------
        def _tuple_format(uniquedict, value):
            result = []
            for col in uniquedict.values():
                # ask each column how it would like to be displayed
                if hasattr(col, 'display_query_properties'):
                    display_format, func = col.display_query_properties()
                else:
                    arr_type, func = DisplayConvert.get_display_convert(col)
                    display_format = default_item_formats.get(arr_type, ItemFormat())
                s = col[value]
                result.append(func(s, display_format))

            return tuple(result)
        # ------------------------------------------------------------
        result = None

        # [['string','string','string']]
        # [[b'string',b'string',b'string']]
        # returns index of matching strings in stringarray
        if isinstance(value, list):
            value = FastArray(value)

        # [np.array(['string','string','string'])]
        # [np.array([b'string',b'string',b'string'])]
        # returns index of matching strings in stringarray
        if isinstance(value, np.ndarray):
            if value.dtype.char in NumpyCharTypes.AllInteger:
                result = [ _tuple_format(self.uniquedict, i) for i in value ]
                if len(result) == 1:
                    result = result[0]
            else:
                raise TypeError(f"Categorical cannot be index by numpy array of dtype {value.dtype}")

        # cat[int]
        # returns single string
        elif isinstance(value, (int, np.integer)):
            invalid_str = self.possibly_invalid(value)
            if invalid_str is None:
                if self.base_index!=0:
                    value-=self.base_index
                result = _tuple_format(self.uniquedict, value)
            else:
                result = invalid_str

        # ['string']
        # [b'string']
        elif isinstance(value, (str, bytes)):
            raise NotImplementedError(f"Cannot perform getitem with strings for multikey categoricals.")

        return result

    # ------------------------------------------------------------
    # SINGLE KEY
    def _getitem_singlekey(self, value):
        result = None

        if isinstance(value, list):
            value = FastArray(value)
        # [np.array(['string','string','string'])]
        # [np.array([b'string',b'string',b'string'])]
        # returns index of matching strings in stringarray
        if isinstance(value, np.ndarray):
            if value.dtype.char in NumpyCharTypes.AllInteger:
                result = self._first_list[value]
            elif value.dtype.char in ('U', 'S'):
                result = self.get_category_match_index(value)
            else:
                raise TypeError(f"Categorical cannot be index by numpy array of dtype {value.dtype}")

        # cat[int]
        # returns single string
        elif isinstance(value, (int, np.integer)):
            # check for invalid index
            invalid_str = self.possibly_invalid(value)
            if invalid_str is None:
                if self.base_index!=0:
                    value-=self.base_index
                result = self._first_list[value]
                if isinstance(result, bytes):
                    try:
                        result = bytes.decode(result)
                    except UnicodeDecodeError:
                        result = str(result)
            else:
                result = invalid_str

        # ['string']
        # [b'string']
        elif isinstance(value, (str, bytes)):
            result = [self.get_category_index(value)]

        return result

    # ------------------------------------------------------------------------------
    def _getitem_enum(self, value):
        """
        At this point, the categorical's underlying fast array's __getitem__ has already been hit. It will only
        execute if the return value was scalar. No need to handle lists/arrays/etc. - which take a different path
        in Categorical.__getitem__

        The value should always be a single integer.

        this will return a single item or list of items from int/string index
        Enums will always return an array of values, even if there is only one entry.
        Enums dictionaries can only be looked up with unicode strings, so bytes will be converted.
        """
        # single int
        if isinstance(value, (int, np.integer)):
            # *** replace this with a direct call to grouping._enum.from_code()
            # need a property to get to GroupingEnum object
            return self.int2strdict.get(value, self.possibly_invalid(value))

        else:
            raise TypeError(f"Indexing by type {type(value)} not supported for categoricals with enum categories.")

    # ------------------------------------------------------------
    def __getitem__(self, value):
        if self.isenum:
            newcat=self._getitem_enum(value)

        elif self.issinglekey:
            newcat=self._getitem_singlekey(value)

        elif self.ismultikey:
            newcat=self._getitem_multikey(value)

        else:
            raise TypeError(f"Critical error in categories")

        return newcat

    # ------------------------------------------------------------
    def categories_as_dict(self):
        """
        Groupby keys can be prepared for the calling Categorical.
        """
        as_dict = {}
        if self.mode == CategoryMode.MultiKey:
            as_dict = self._column_dict

        elif self.mode in Categories.dict_modes:
            as_dict = { self.default_colname : FastArray(list(self.str2intdict.keys())) }

        elif self.mode in Categories.list_modes:
            name = self._list.get_name()
            if name is None:
                name = self.default_colname
            as_dict = { name : self._list }

        else:
            raise TypeError(f"Don't know how to return category dictionary for categories in mode: {self.mode}")

        return as_dict

    # -------------- GET ALL CATEGORIES---------------------------
    # ------------------------------------------------------------
    def _get_array(self):
        if self.issinglekey:
            # make this switch after the modify / setitem functions are sent to a Grouping API
            #return self.uniquelist[0]
            return self._first_list

        elif self.ismultikey:
            raise TypeError(f"Cannot return single array for multikey categoricals.")

        elif self.isenum:
            return FastArray(list(self.str2intdict.keys()))
        else:
            raise TypeError(f"Don't know how to return category array for categories in mode: {self.mode}")

    # ------------------------------------------------------------
    def _get_codes(self):
        if self.isenum:
            return FastArray(list(self.int2strdict.keys()))
        else:
            raise TypeError(f"Can't return codes for categories in mode: {self.mode} Use Categorical._fa instead.")

    # ------------------------------------------------------------
    def _get_mapping(self):
        if self.isenum:
            return self.int2strdict
        else:
            raise TypeError(f"Dictionary mapping can only be returned from Categories in dictionary mode, not {self.mode}")

    # ------------------------------------------------------------
    def _get_dict(self):
        return self.categories_as_dict()

    # -------------- MODIFY CATEGORY FUNCTIONS -------------------
    # ------------------------------------------------------------
    def _mapping_edit(self, code, value=None, how='add'):
        #Grouping object needs methods for:
        #- replace enum with new mapping
        #- add new mapping int->str, str->int
        #- remove mapping
        #- replace mapping
        #- should categorical still check the dictionary?
        if self.isenum:
            if isinstance(code, (int, np.integer)):
                exists = self.int2strdict.get(code,False)

                # -ADD------------------------------
                if how == 'add':
                    if value is not None:
                        if exists is False:
                            self.int2strdict[code] = value
                            self.str2intdict[value] = code
                        else:
                            raise ValueError(f"Mapping already exists for {code} -> {exists}. Use mapping_replace() instead.")
                    else:
                        raise ValueError("code and value must be passed to mapping_add")

                # -REMOVE------------------------------
                elif how == 'remove':
                    if exists is not False:
                        del self.int2strdict[code]
                        del self.str2intdict[exists]
                    else:
                        raise ValueError(f"Mapping doesn't exist for {code}. Nothing was removed.")

                # -REPLACE-----------------------------
                elif how == 'replace':
                    if value is not None:
                        if exists is not False:
                            self.int2strdict[code] = value
                            self.str2intdict[value] = code
                        else:
                            raise ValueError(f"Mapping doesn't exists for {code}. Nothing was replaced.")
                    else:
                        raise ValueError("code and value must be passed to mapping_replace")
                else:
                    raise ValueError(f"Invalid value {how} for how keyword. Must be 'add', 'remove', or 'replace'.")

            else:
                raise TypeError(f"Code must be integer.")
        else:
            raise TypeError(f"Cannot add mapping unless category mode is in dictionary or enum mode.")

    # ------------------------------------------------------------
    def _mapping_new(self, mapping):
        if self.isenum:
            if isinstance(mapping, (dict, EnumMeta)):
                self.grouping._enum = GroupingEnum(mapping)
            else:
                raise TypeError(f"New mapping must be a dictionary or IntEnum, not {type(mapping)}")
            self._max_int = max(self.int2strdict.keys())
        else:
            raise ValueError(f"Categories cannot be replaced with new category mappings unless they are in Dictionary/Enum mode, not {self.mode}")

    # ------------------------------------------------------------
    def _is_valid_mapping_code(self, value):
        return value in self.int2strdict

    # ------------------------------------------------------------
    def _array_edit(self, value, new_value=None, how='add'):
        #Grouping object needs methods for:
        #- add to uniquedict (array length will change)
        #- remove from uniquedict (array length will change)
        #- replace item in uniquedict (array length same size)
        #- ordered/sorted flags may be invalidated
        #- dirty flag will be set
        #- drop any lazy generated data based on previous uniquedict
        #- should categorical still check the array, match the values?

        if self._mode == CategoryMode.StringArray:
            value = self.match_str_to_category(value)
            if how == 'add':
                # only add if doesn't exist
                if len(self.get_category_match_index(value)) == 0:
                    # always add to the end (no index fixing)
                    self._list = hstack([self._list, value])
                    self._ordered = False
                    self._sorted = False
                else:
                    raise ValueError(f"Category {value} already found in categories array.")

            elif how == 'remove':
                remove_idx = self.get_category_match_index(value)
                # only remove if exists
                if len(remove_idx) == 1:
                    remove_idx = remove_idx[0]-self.base_index
                    # slice around single item
                    self._list = hstack([self._list[:remove_idx], self._list[remove_idx+1:]])
                    # return to categorical - indices >= add need to be fixed
                    return remove_idx+self.base_index
                else:
                    raise ValueError(f"Category {value} not found")

            elif how == 'replace':
                if new_value is not None:
                    replace_idx = self.get_category_match_index(value)
                    if len(replace_idx) == 1:
                        replace_idx = replace_idx[0]-self.base_index
                    else:
                        raise ValueError(f"Category {value} not found")
                    # also check if replacement exists
                    new_value = self.match_str_to_category(new_value)
                    new_exists = self.get_category_match_index(new_value)
                    # if replacement category exists, old will not be changed, but the indices will
                    if len(new_exists) == 1:
                        return replace_idx+self.base_index, new_exists[0]
                    else:
                        self._list[replace_idx] = new_value
                        self._ordered = False
                        self._sorted = False
                else:
                    raise ValueError(f"New value must be provided for category replacement.")
            else:
                raise ValueError(f"Invalid value {how} for how keyword. Must be 'add', 'remove', or 'replace'.")


        elif self._mode == CategoryMode.MultiKey:
            raise NotImplementedError

        else:
            raise ValueError(f"Category arrays can only be modified for categoricals based on a string array or single key dictionary.")

    # ------------------------------------------------------------
    def __repr__(self):
        if self.isenum:
            return self.grouping._enum.__repr__()
        return self.get_categories().__repr__()

    def __str__(self):
        return str(self.get_categories())

    def _build_string(self):
        pass

    # ------------------------------------------------------------
    @classmethod
    def build_dicts_python(cls, python_dict):
        """
        Categoricals can be initialized with a dictionary of string to integer or integer to string.
        Python dictionaries accept multiple types for their keys, so the dictionaries need to check types as they're being constructed.
        """
        invalid = []
        str_to_int_dict = {}
        int_to_str_dict = {}

        key_list = list(python_dict.keys())
        value_list = list(python_dict.values())

        # determine which way the keys and values of the dictionary are pointing
        if isinstance(key_list[0], (str, bytes)):
            string_list = key_list
            if isinstance(value_list[0], (int, np.integer)):
                int_list = value_list
            else:
                raise TypeError(f"Invalid type {type(value_list[0])} encountered in dictionary values. Dictionaries must be string -> integer or integer -> string")
        elif isinstance(key_list[0], (int, np.integer)):
            int_list = key_list
            if isinstance(value_list[0], (str, bytes)):
                string_list = value_list
            else:
                raise TypeError(f"Invalid type {type(value_list[0])} encountered in dictionary values. Dictionaries must be string -> integer or integer -> string")
        else:
            raise TypeError(f"Invalid type {type(key_list[0])} encountered in dictionary values. Dictionaries must be string -> integer or integer -> string")

        for k, v in zip(string_list, int_list):
            # make sure types remain consistent
            if not isinstance(v, (int, np.integer)):
                raise TypeError(f"Invalid type {type(v)} in dictionary integer values. All values must be integer.")
            if not isinstance(k, (str, bytes)):
                raise TypeError(f"Invalid type {type(k)} in dictionary string values. All values must be string.")

            # allowing support for negative integer values in dictionary
            # if v >= 0:
            if True:
                # make sure entire dictionary has same string type
                if isinstance(k, bytes):
                    k = k.decode()
                if k in str_to_int_dict:
                    warnings.warn(f"{k} already found in dict. problems may occur.")
                str_to_int_dict[k] = v
                if v in int_to_str_dict:
                    warnings.warn(f"{k} already found in dict. problems may occur.")
                int_to_str_dict[v] = k
            else:
                invalid.append(k)

        # warn with list of entries that weren't added
        if len(invalid) > 0:
            warnings.warn(f"The following items had a code < 0 and were not added: {invalid}")

        return str_to_int_dict, int_to_str_dict

    # ------------------------------------------------------------
    @classmethod
    def build_dicts_enum(cls, enum):
        """
        Builds forward/backward dictionaries from IntEnums. If there are multiple identifiers with the same, WARN!
        """
        invalid = []
        str_to_int_dict = {}
        int_to_str_dict = {}
        for k, v in enum.__members__.items():
            int_v = v.value
            if True:
                if k in str_to_int_dict:
                    warnings.warn(f"{k} already found in dict. problems may occur.")
                str_to_int_dict[k] = int_v
                if int_v in int_to_str_dict:
                    warnings.warn(f"{int_v} already found in dict. problems may occur.")
                int_to_str_dict[int_v] = k
            else:
                invalid.append(k)
        if len(invalid) > 0:
            warnings.warn(f"The following items had a code < 0 and were not added: {invalid}")
        return str_to_int_dict, int_to_str_dict


# ------------------------------------------------------------
class Categorical(GroupByOps, FastArray):
    '''
    An riptable Categorical maps integer values to unique categories, which are held in an array, IntEnum/IntString Dictionary, or Multikey Dictionary.

    Certain methods of construction provide enough information for groupby operations to use bin information that the Categorical is already storing.
    The underlying array is always integer based, in a type based on the number of unique categories or maximum value in a code mapping.
    It will be an ``int8``, ``int16``, ``int32``, or ``int64``.

    Unless otherwise specified via ordered, categories are maintained in an unsorted order.
    Constructing a categorical from an unsorted list of strings and forcing an ordered assumption will lead to unexpected
    results in comparison operations.

    Parameters
    ----------
    values
        * list or array of string values. the list will be made unique, and an integer array will be constructed to index into it
        * list or array of numeric values. the list will be made unique with a groupby operation and an integer array will be constructed to index into it
        * list or array of float values. (matlab indexing) categories must be set to a unique array, and from_matlab must be set to True.
        * list or array of integer values. (user specified indexing) categories must be set to a unique array or IntEnum/dictionary of integer->string
        * list or array of integer values. (internally specified indexing) categories must come from another rt categorical. the _from_categorical flag must be set. a shallow copy is made.
        * rt Categorical object - a deep copy of categories is performed
        * pandas Categorical object - a deep copy is performed. indices will be incremented by 1 to translate pandas invalids
        * list of numpy arrays. the keys will be made unique with a groupby operation and an integer array will be constructed to index into it
        * dictionary of numpy arrays. the keys will be made unique with a groupby operation and an integer array will be constructed to index into it
    categories
        * list or array of unique categories
        * intenum or dictionary of integer code mapping to string values. must be paired with integer array of codes
    ordered : bool, optional, default None
        If a categorical's ordered flag is set to True, a sort will be performed when the categorical is made,
        otherwise the categorical's display order is dependent on `lex`==True/False. If `lex`==False, the display order is first appearance.
        Sorted groupby operations can be requested by setting the `sort_gb` keyword to True (see below).
    sort_display : bool, optional, default None
        See `sort_gb`.
    sort_gb : bool, optional, default None
        By default, groupby operations will be unsorted. If `sort_gb` is set to True in the constructor, a sort will be performed (lazily) and
        applied. If the categorical is naturally sorted (see above), no sort will be performed or applied.
    lex : bool, optional, default None
        By default hashing will be used to discover unique elements in the arrays.  If lex is set to True, a lexsort is used instead.
        For high unique counts (more than 50% of the elements are unique), lex=True maybe faster.
    locked : bool, default False
        When set to True, prevents categories from being added. The locked flag is automatically set to True after a groupby operation
        to prevent unexpected data corruption in operations to follow. If categories are added after a groupby operation, the groupby indexing
        may no longer be accurate.
    from_matlab : bool, default False
        When set to True, allows floating point indexing into unique categories. The indices will be flipped to an integer type and the base
        index will always be set to 1. 0 is always an invalid index in matlab, so the indices are already compatible with the rt base-1 default.
    _from_categorical : (None)
        flag for internal routines to skip checking/sorting/uniquifying values and categories. Categories will be passed through this internal keyword.
        NOTE: this does not perform a deep copy of the categorical's categories or underlying array
    dtype : np.dtype or str, optional
        force the dtype of the underlying integer array. by default, the constructor will opt for the smallest type based on
        the size of the unique categories
    invalid : str, optional
        specify a string to use when an invalid category is returned or displayed
    auto_add : bool, default False
        when set to True, categories that do not exist in the unique categories can be added with the setitem method. by default an error is raised.

    Examples
    --------
    A single list of bytes or unicode

    >>> c = rt.Categorical(['a','a','b','a','c','c','b'])
    >>> print(c)
    a, a, b, a, c, c, b

    A view of the underlying integer array (will default to base-1 indexing)

    >>> print(c._fa)
    [1 1 2 1 3 3 2]

    The constructor attempts to convert unicode strings to bytestrings

    >>> c.categories()
    FastArray([b'a', b'b', b'c'], dtype='|S1')

    A list of integer indices into a list of unique strings

    >>> c = rt.Categorical([0,1,1,0,2,1,1,1,2,0], categories=['a','b','c'])
    >>> print(c)
    a, b, b, a, c, b, b, b, c, a

    A list of non-unique string values and a unique list of category strings

    >>> c = rt.Categorical(['a','a','b','c','a','c','c','c'], categories=['a','b','c'])
    >>> print(c)
    a, a, b, c, a, c, c, c

    Setting an option to add an invalid category when missing values are encountered (otherwise an error is raised).

    >>> c = rt.Categorical(['a','z','b','c','a','y','c','c'], categories=['a','b','c'], invalid='Inv')
    >>> print(c)
    a, Inv, b, c, a, Inv, c, c

    The invalid category will not be added to the categories. instead the categorical will use base-1 indexing and
    display the invalid indices as the specified invalid category string.

    >>> c.categories()
    FastArray([b'a', b'b', b'c'], dtype='|S1')

    A list of floating point indexes into a list of unique strings (Matlab data)

    >>> c = rt.Categorical([1.0, 1.0, 2.0, 3.0, 1.0, 1.0], categories=['a','b','c'], from_matlab=True)
    >>> print(c)
    a, a, b, c, a, a

    Matlab uses 1-based indexing, reserving the 0 bin for invalids - just like the default _riptable_ behavior.
    All initialization from Matlab indices + matlab categories will use 1-based indexing.

    >>> print(c.view(FastArray))
    [1 1 2 3 1 1]

    A pandas ``Categorical`` with invalid

    >>> pdc = pd.Categorical(['a','a','z','b','c'],['a','b','c'])
    >>> print(pdc)
    [a, a, NaN, b, c]
    Categories (3, object): [a, b, c]

    >>> c = rt.Categorical(pdc)
    >>> print(c)
    a, a, Inv, b, c

    An IntEnum paired with integer values

    >>> from enum import IntEnum
    >>> class LikertDecision(IntEnum):
    ...     """A Likert scale with the typical five-level Likert item format."""
    ...     StronglyAgree = 44
    ...     Agree = 133
    ...     Disagree = 75
    ...     StronglyDisagree = 1
    ...     NeitherAgreeNorDisagree = 144
    >>> codes = [1, 44, 44, 133, 75]
    >>> c = rt.Categorical(codes, LikertDecision)
    >>> print(c)
    StronglyDisagree, StronglyAgree, StronglyAgree, Agree, Disagree

    A python dictionary of integers to strings

    >>> d = { 'StronglyAgree': 44, 'Agree': 133, 'Disagree': 75, 'StronglyDisagree': 1, 'NeitherAgreeNorDisagree': 144 }
    >>> codes = [1, 44, 44, 133, 75]
    >>> c = rt.Categorical(codes, d)
    >>> print(c)
    StronglyDisagree, StronglyAgree, StronglyAgree, Agree, Disagree

    A python dictionary of strings to integers

    >>> d = { 'StronglyAgree': 44, 'Agree': 133, 'Disagree': 75, 'StronglyDisagree': 1, 'NeitherAgreeNorDisagree': 144 }
    >>> codes = [1, 44, 44, 133, 75]
    >>> c = rt.Categorical(codes, d)
    >>> print(c)
    StronglyDisagree, StronglyAgree, StronglyAgree, Agree, Disagree

    Indexing behavior

    >>> c = rt.Categorical(['a','a','b','a','c','c','b'])
    >>> print(c)
    a, a, b, a, c, c, b
    >>> c['a']
    FastArray([ True,  True, False,  True, False, False, False])
    >>> c[['a','b']]
    FastArray([ True,  True,  True,  True, False, False,  True])
    >>> c[3]
    'a'
    >>> c[[3, 4]]
    Categorical([a, c])
    >>> c[:2]
    Categorical([a, a])

    Comparison behavior

    >>> c = rt.Categorical(['a','a','b','a','c','c','b'])
    >>> print(c)
    a, a, b, a, c, c, b
    >>> c == 'a'
    FastArray([ True,  True, False,  True, False, False, False])
    >>> c == b'a'
    FastArray([ True,  True, False,  True, False, False, False])
    >>> c == 2
    FastArray([False, False,  True, False, False, False,  True])

    Use `Categorical` to perform aggregations over arbitrary arrays (of the same dimension as the `Categorical`),
    just like a `GroupBy` object.

    >>> c = rt.Categorical(['a','a','b','a','c','c','b'])
    >>> int_arr = np.array([3, 10, 2, 5, 4, 1, 1])
    >>> flt_arr = np.array([1.2, 3.4, 5.6, 4.0, 2.1, 0.6, 11.3])
    >>> c.sum(int_arr, flt_arr)
    *gb_key   col_0   col_1
    -------   -----   -----
    a            18    8.60
    b             3   16.90
    c             5    2.70
    '''
    # current metadata version and default values necessary for final reconstruction
    MetaVersion = 1
    MetaDefault = {
        # vars for container loader
        'name': 'Categorical',
        'typeid': TypeId.Categorical,
        'version': 0,       # if no version, assume before versions implemented

        # vars for additional arrays
        'colnames' : [],
        'ncols' : 0,

        # vars to rebuild the same categorical
        'instance_vars' : {
            'mode'       : None,
            'base_index' : 1,
            'ordered'    : False,
            'sort_gb'    : False
        },

        # vars to rebuild categories object
        'cat_vars' : {
            '_invalid_category' : None,
            '_filtered_name' : FILTERED_LONG_NAME
        }
    }
    # flag for printouts, assertions
    DebugMode = False
    GroupingDebugMode = False

    # flags for ismember testing
    TestIsMemberVerbose = False
    _test_cat_ismember = ""

    def __new__(
        cls,
        # main data
        values,
        categories=None,
        # sorting/hashing
        ordered: Optional[bool] = None,
        sort_gb: Optional[bool] = None,
        sort_display: Optional[bool] = None,
        lex: Optional[bool] = None,
        # priority options
        base_index: Optional[int] = None,
        filter: Optional[np.ndarray] = None,
        # misc options
        dtype: Optional[Union[np.dtype, str]] = None,
        unicode: Optional[bool] = None,
        invalid: Optional[str] = None,
        auto_add: bool = False,
        # origin, possible fast track
        from_matlab: bool = False,
        _from_categorical = None
    ):

        invalid_category=invalid
        # possibly set categories with defaults
        # raise certain impossible combination errors immediately
        # not allowed:
        if base_index == 0:
            if filter is not None:
                raise ValueError(f"Filtering is not allowed for base index 0. Use base-1 indexing instead.")

        index = values
        instance = None
        grouping = None

        # prepare to eliminate sort_gb
        if sort_display is not None:
            sort_gb = sort_display

        # for how final display is sorted
        if sort_gb is None:
            _sort_gb = False
        else:
            _sort_gb = sort_gb

        # default to hash for uniques
        if lex is None:
            _lex = False
        else:
            _lex = lex

        # default to bytestrings - more performant, less memory
        if unicode is None:
            unicode = False

        # default to 1-based indexing (filtering, etc. fully supported in this mode)
        if base_index is None:
            base_index = 1

        # pop all single items from lists, or wrap in array
        if isinstance(values, list):
            if len(values) == 0:
                raise ValueError("Categorical: values was an empty list and is not allowed.")

            elif len(values) == 1:
                if isinstance(values[0], np.ndarray):
                    values = values[0]
                else:
                    values = FastArray(values, unicode=unicode)
            else:
                # multikey always ordered now by default
                # TJD Oct 2019 -- if nothing else set, default to ordered =True
                # note this differs from groupby's default mode
                if ordered is None and lex is None and sort_gb is None and from_matlab is False:
                    pass
                    # TJD  want to force default sort in future
                    #ordered = True

        # from categorical, deep copy - send to regular categorical.copy() to correctly preserve attributes
        if isinstance(values, Categorical):
            return values.copy(categories=categories,                    # main data
                ordered=ordered, sort_gb=sort_gb,  lex=lex,       # sorting/hashing
                base_index=base_index, filter=filter,                    # priority options
                dtype=dtype, unicode=unicode, invalid=invalid, auto_add=auto_add, # misc options
                from_matlab=from_matlab, _from_categorical=_from_categorical)

        # all constructors will funnel to this branch
        elif isinstance(values, Grouping):
            grouping = values
            categories = Categories.from_grouping(grouping, invalid_category=invalid_category)
            base_index = grouping.base_index
            _sort_gb = grouping.isdisplaysorted
            # will be different for base index 1, 0, enum
            index = grouping.catinstance

        # from internal routine, fast track
        elif _from_categorical is not None:
            # use defaults for all keywords in fast track
            # **** flip all internal construction to grouping object here
            if not isinstance(_from_categorical, Grouping):
                # categories object
                if isinstance(_from_categorical, Categories):
                    if hasattr(_from_categorical, '_grouping'):
                        grouping = _from_categorical.grouping.copy(deep=False)
                    else:
                        if cls.DebugMode: warnings.warn(f'This Categories object did not a have a a grouping object.')
                        # this path will be removed if grouping is always attached
                        # will raise an error instead
                        if _from_categorical.mode in Categories.dict_modes:
                            grouping = Grouping(index, categories=_from_categorical._str_to_int_dict, filter=filter, sort_display=_sort_gb, base_index=base_index, dtype=dtype, _trusted=True)
                        else:
                            grouping = Grouping(index, categories=_from_categorical.categories_as_dict(), sort_display=_sort_gb, categorical=True, dtype=dtype, _trusted=True)

                    # don't need to reconstruct from grouping
                    categories = _from_categorical

                # build a grouping object from SDS load
                # categories holds the unique array(s)
                elif isinstance(_from_categorical, MetaData):
                    meta = _from_categorical
                    vars = meta['instance_vars']
                    mode = vars['mode']
                    # enum
                    if mode in Categories.dict_modes:
                        catmode = False
                        ints = categories[0]
                        strs = categories[1].astype('U', copy=False)
                        cats = dict(zip(strs,ints))
                    else:
                        # pull column name from meta tuples to send single/multikey down same path
                        catmode = True
                        # single key or multikey
                        cats = {}
                        for colname, arr in zip(meta['colnames'], categories):
                            # check for an array that also has nested meta data
                            newmeta = meta.get(colname + '_meta', None)
                            if newmeta is not None:
                                # load that class special
                                newclass = getattr(TypeRegister, newmeta['classname'])
                                arr = newclass._from_meta_data({colname: arr}, None, newmeta)
                            cats[colname] = arr

                    grp = Grouping(index, cats, base_index=vars['base_index'], ordered=vars['ordered'], sort_display=vars['sort_gb'], categorical=catmode, unicode=True, _trusted=True)

                    # build the categorical from grouping
                    result = cls(grp)

                    # restore extra categories vars
                    # these include invalid category, string to display for filtered items
                    cats = result._categories_wrap
                    for k,v in meta['cat_vars'].items():
                        setattr(cats, k, v)
                    return result

                # unique array list or dict
                else:
                    grp = Grouping(index, _from_categorical, base_index=base_index, ordered=ordered, unicode=True, _trusted=True)
                    return cls(grp)

            # grouping object, shallow copy with new ikey
            if isinstance(_from_categorical, Grouping):
                grouping = Grouping.newclassfrominstance(index, _from_categorical)
                categories = Categories.from_grouping(grouping, invalid_category=invalid_category)

        # from pandas categorical, faster track
        elif hasattr(values, '_codes'):
            if base_index != 1:
                raise ValueError(f"To preserve invalids, pandas categoricals must be 1-based.")

            # pandas invalid -1 turns into riptable invalid 0
            # just like regular int + categories, never change the order of pandas categories
            categories = values.categories.values
            if dtype is None:
                newdt = int_dtype_from_len(len(categories))
                index = np.add(values._codes, 1, dtype=newdt)
            else:
                index = values._codes + 1

            ordered = values.ordered

            grouping = Grouping(index, categories=categories, filter=filter, sort_display=_sort_gb, ordered=ordered, categorical=True, dtype=dtype, unicode=unicode)
            return cls(grouping, invalid=invalid_category)


        # all branches above will be ready for final construction
        else:
            ismultikey = False
            if isinstance(values, list):
                # single item has already been popped
                if isinstance(values[0], np.ndarray):
                    ismultikey = True
                else:
                    values = FastArray(values, unicode=unicode)

            if isinstance(values, dict) or ismultikey:
                if len(values) == 1 and isinstance(values, dict):
                    # pop single item
                    single_val = [*values.values()][0]
                    if isinstance(single_val, np.ndarray):
                        values = single_val
                else:
                    if categories is not None:
                        raise NotImplementedError(f"Multikey categoricals do not currently support user-defined categories.")

                    # different than the default for single key
                    if ordered is None:
                        if lex is True:
                            ordered = True
                        else:
                            ordered = False

                    # multikey will also store a grouping object in its constructor
                    # TODO: add one routine so multikey uniques can be stored sorted
                    grouping = Grouping(values, base_index=base_index, filter=filter, ordered=ordered, sort_display=_sort_gb, lex=_lex, categorical=True, dtype=dtype, unicode=unicode)
                    return cls(grouping)

            # most common path --- values as array
            if isinstance(values, np.ndarray):
                if cls.DebugMode: print('values was ndarray')
                # only values were provided, need to generate uniques
                if categories is None:
                    if cls.DebugMode: print('categories was none, calling unique')

                    # default to sort when generating our own uniques
                    if ordered is None:
                        ordered = True
                    if ordered:
                        #if sort_gb is False:
                        #    warnings.warn(f"sort_gb was set to False, but groupby results will appear in order by default. Set keyword ordered=False for first-occurrence, unsorted results.")
                        if cls.DebugMode: print('will perform ordered after unique')

                    # single array of non-unique values
                    grouping = Grouping(values, sort_display=_sort_gb, ordered=ordered, base_index=base_index, filter=filter, lex=lex, categorical=True, dtype=dtype, unicode=unicode)
                    result = cls(grouping, invalid=invalid_category)
                    _copy_name(values, result)
                    return result

                # uniques, others provided
                else:
                    # lexsort can only be used if non-uniques are provided alone (single or multikey)
                    if _lex:
                        raise TypeError(f'Cannot bin using lexsort and user-suplied categories.')
                    # init from mapping
                    if isinstance(categories, (EnumMeta, dict)):
                        index = values
                        grouping = Grouping(values, categories=categories, filter=filter, sort_display=_sort_gb, base_index=base_index, dtype=dtype, unicode=unicode)

                        # this can replace the rest of this block when setitem / modify category methods have been implemented with grouping
                        #return cls(grouping, invalid=invalid_category)

                        int2str = grouping._enum._int_to_str_dict
                        str2int = grouping._enum._str_to_int_dict
                        categories = Categories(int2str, str2int, invalid_category=invalid_category)
                        _copy_name(index, categories)

                        # code mappings will display invalid string on sentinel
                        ordered = None
                        base_index = None

                    # flip list to numpy array, check for supported types
                    elif isinstance(categories, list):
                        if cls.DebugMode: print('categories was list')
                        catlen = len(categories)
                        if catlen == 0:
                            raise ValueError(f"Provided categories were empty.")

                        # possibly extract single array
                        elif catlen == 1:
                            if isinstance(categories[0], np.ndarray):
                                categories = categories[0]

                        # multidimensional array of uniques, or more than one in uniques
                        else:
                            if isinstance(categories[0], np.ndarray):
                                raise TypeError(f"Cannot construct categorical from categories that was a list of numpy arrays.")

                    # flip lists, wrap scalars, catch everything else here
                    if not isinstance(categories, (np.ndarray, Categories)):
                        categories = FastArray(categories, unicode=unicode)

                    # handle array of provided categories
                    if isinstance(categories, np.ndarray):
                        # catch float indices first in case of matlab
                        if values.dtype.char in NumpyCharTypes.AllFloat:
                            if cls.DebugMode: print('values was float array')
                            # matlab
                            if from_matlab:
                                if base_index != 1:
                                    raise ValueError(f"Categoricals from matlab must have a base index of 1, got {base_index}.")
                                newdt = int_dtype_from_len(len(categories))
                                # flip to int, flip all sentinel nan to 0s
                                values = values.astype(newdt)
                                nan_to_zero(values)

                        # indices -> unique categories
                        if values.dtype.char in NumpyCharTypes.AllInteger:
                            grouping = Grouping(values, categories=categories, filter=filter, sort_display=_sort_gb, base_index=base_index, categorical=True, dtype=dtype)
                            return cls(grouping, invalid=invalid_category)

                        # non-unique values -> unique cateogires
                        # grouping will use ismember
                        else:
                            grouping = Grouping(values, categories=categories, sort_display=_sort_gb, base_index=base_index, filter=filter, categorical=True, dtype=dtype, unicode=unicode)

                            # these errors will replace code below, need to fix map() first
                            ikey = grouping.catinstance
                            # check for values that were not found
                            # only allowed if a filter was provided
                            if base_index == 0:
                                if (min(ikey) < 0) or (max(ikey) > grouping.unique_count-1):
                                    raise ValueError(f"Cannot initialize base index 0 categorical with invalid values.")
                            else:
                                if filter is None:
                                    inv_fancy = ikey == 0
                                    hasinv = sum(inv_fancy) > 0
                                    if hasinv:
                                        if invalid_category is None:
                                            raise ValueError(f"Found values that were not in provided categories: {values[inv_fancy]}")
                                        else:
                                            raise ValueError(f"Found values that were not in provided categories: {values[inv_fancy]}. The user-supplied categories (second argument) must also contain the invalid item {invalid_category}. For example: Categorical(['b','a','Inv','a'], ['a','b','Inv'], invalid='Inv')")
                                else:
                                    if invalid_category is not None:
                                        warnings.warn(f"Invalid category was set to {invalid_category}. If not in provided categories, will also appear as filtered. For example: print(Categorical(['a','a','b'], ['b'], filter=FA([True, True, False]), invalid='a')) -> Filtered, Filtered, Filtered")
                            return cls(grouping, invalid=invalid_category)

            else:
                if grouping is None:
                    raise TypeError(f"Don't know how to construct categorical from values input of type {type(values)}.")

        if cls.DebugMode: print('initializing final instance variables...')
        instance = index.view(cls)
        instance._ordered = ordered
        instance._sort_gb = _sort_gb

        # ***attach grouping object to categories for accessing uniques
        if not hasattr(categories, '_grouping'):
            categories._grouping = grouping
        instance._categories_wrap = categories

        instance._grouping = grouping
        instance._unicode = unicode
        instance._gb_keychain = None

        instance._sorted = ordered
        instance._locked = False
        instance._dtype  = dtype
        instance._auto_add_categories=auto_add
        instance._categories_wrap._auto_add_categories=auto_add
        instance._dataset = None
        instance._filter = None

        if _from_categorical is not None and isinstance(_from_categorical, Categories):
            # this should really be from a copy
            categories._filtered_name = _from_categorical._filtered_name

        # maybe change name to a categorical property
        if instance._categories_wrap.name is not None:
            instance.set_name(instance._categories_wrap.name)

        #print(f'_ordered {instance._ordered}')
        #print(f'_sorted {instance._sorted}')
        #print(f'_locked {instance._locked}')
        #print(f'_dtype {instance._dtype}')
        #print(f'_auto_add_categories {instance._auto_add_categories}')
        #print(f'_categories_wrap {instance._categories_wrap}')
        #print(f'_unicode {instance._unicode}')
        #print(f'_grouping {instance._grouping}')
        #print(f'_sort_gb {instance._sort_gb}')
        #print(f'_gb_keychain {instance._gb_keychain}')

        return instance

    # Ensure API signature matches Categorical new
    def __init__(self, values, categories=None,                   # main data
                 ordered=None, sort_gb=None, sort_display=None, lex=None,       # sorting/hashing
                 base_index=None, filter=None,                  # priority options
                 dtype=None, unicode=None, invalid=None, auto_add=False, # misc options
                 from_matlab=False, _from_categorical=None):       # origin, possible fast track
        pass

    # ------------------------------------------------------------
    def argsort(self):
        return argsort(self._fa)

    # ------------------------------------------------------------
    def _nan_idx(self):
        """
        Internal - for isnan, isnotnan
        """
        # maybe expose this in a different API (has nan?)
        idx = None
        if self.invalid_category is None:
            pass
        else:
            try:
                idx = self.from_category(self.invalid_category)
            except:
                pass
        return idx

    # ------------------------------------------------------------
    def _nanfunc(self, func, fillval):
        idx = self._nan_idx()

        if idx is None:
            return full(len(self), fillval)
        return func(idx)

    # ------------------------------------------------------------
    def isnan(self, *args, **kwargs):
        return self._nanfunc(self._fa.__eq__, False)

    # ------------------------------------------------------------
    def isnotnan(self, *args, **kwargs):
        return self._nanfunc(self._fa.__ne__, True)

    # ------------------------------------------------------------
    def isna(self, *args, **kwargs):
        return self.isnan()

    # ------------------------------------------------------------
    def notna(self, *args, **kwargs):
        return self.isnotnan()

    # ------------------------------------------------------------
    def fill_forward(self, *args, limit:int=0, fill_val=None,inplace:bool=False):
        """
        Forward fill the values of the categorical, by group.
        By default this is done inplace.

        Parameters
        ----------
        list of one or more arrays the same len as the categorical to fill

        Other Parameters
        ----------------
        limit : integer, optional
            limit of how many values to fill
        inplace: defaults to False

        Examples
        --------
        >>> x = rt.FA([1, 4, 9, 16, np.nan, np.nan])
        >>> y = rt.Categorical(['A', 'B', 'A', 'B', 'A', 'B'])
        >>> y.fill_forward(x)[0]
        FastArray([ 1.,  4.,  9., 16.,  9., 16.])

        See Also
        --------
        rt.fill_forward
        rt.Cat.fill_backward
        rt.GroupBy.fill_forward
        """
        result = self.apply_nonreduce(fill_forward, *args, fill_val=fill_val, limit=limit, inplace=True)
        if inplace is True:
            for i in range(len(args)):
                x=args[i]
                # copy inplace
                x[...] = result[i]
        return result


    # ------------------------------------------------------------
    def fill_backward(self, *args, limit:int=0, fill_val=None, inplace:bool=False):
        """
        Backward fill the values of the categorical, by group.
        By default this is done inplace.

        Parameters
        ----------
        list of one or more arrays the same len as the categorical to fill

        Other Parameters
        ----------------
        limit : integer, optional
            limit of how many values to fill
        inplace: defaults to False

        Examples
        --------
        >>> x = rt.FA([1, 4, np.nan, np.nan, 9, 16])
        >>> y = rt.Categorical(['A', 'B', 'A', 'B', 'A', 'B'])
        >>> y.fill_backward(x)[0]
        FastArray([ 1.,  4.,  9., 16.,  9., 16.])

        See Also
        --------
        rt.fill_forward
        rt.Cat.fill_forward
        rt.GroupBy.fill_backward
        """
        result = self.apply_nonreduce(fill_backward, *args, fill_val=fill_val, limit=limit, inplace=True)
        if inplace is True:
            for i in range(len(args)):
                x=args[i]
                # copy inplace
                x[...] = result[i]
        return result

    # ------------------------------------------------------------
    def isfiltered(self):
        """
        True where bin == 0.
        Only applies to categoricals with base index 1, otherwise returns all False.
        Different than invalid category.

        See Also
        --------
        Categorical.isnan
        Categorical.isnotnan
        """
        if self.base_index == 1:
            return self._fa == 0
        else:
            return zeros(len(self),dtype=np.bool)

    # ------------------------------------------------------------
    def set_name(self, name):
        """
        If the grouping dict contains a single item, rename it.

        See Also
        --------
        Grouping.set_name()
        FastArray.set_name()
        """

        self.grouping.set_name(name)
        # key chain also has the name
        self._gb_keychain = None
        return super().set_name(name)

    # ------------------------------------------------------------
    @property
    def _fa(self):
        result = self.view(FastArray)
        _copy_name(self, result)
        return result

    # ------------------------------------------------------------
    @property
    def base_index(self):
        return self.grouping.base_index

    # -----------------------------------------------------------------------------------
    @property
    def _total_size(self) -> int:
        """
        Returns total size in bytes of Categorical's Index FastArray and category array(s).
        """
        total_size = self._fa.itemsize * len(self._fa)
        if not self.isenum:
            for arr in self._categories_wrap.uniquedict.values():
                total_size += arr.itemsize * len(arr)
        return total_size

    # ------------------------------------------------------------
    def _ipython_key_completions_(self):
        """
        For tab completions with bracket indexing (__getitem__)
        The IPython completer needs a python list or dict keys/values.
        If no return (e.g. multikey categorical), return an empty list.
        Also returns empty if categorical has > 10_000 unique values.
        If an IPython environment is detected, the 'greedy' property is set to True in riptable's __init__
        """
        if self.unique_count < 10_000:
            if self.category_mode in {CategoryMode.StringArray, CategoryMode.NumericArray}:
                return list(self.category_array.astype('U',copy=False))
            elif self.isenum:
                return self.category_mapping.values()
        else:
            return ['!!!too large for autocomplete']
        return []

    # -----------------------------------------------------------------------------------
    def categories(self, showfilter:bool=True):
        """
        If the categories are stored in a single array or single-key dictionary, an array will be returned.
        If the categories are stored in a multikey dictionary, a dictionary will be returned.
        If the categories are a mapping, a dictionary of the mapping will be returned (int -> string)

        Note: you can also request categories in a certain format when possible using properties:
        `category_array`, `category_dict`, `category_mapping`.

        Parameters
        ----------
        showfilter : bool, defaults to True
            If True (default), the invalid category will be prepended to the returned array or multikey columns.
            Does not apply when mapping is returned.

        Returns
        -------
        np.ndarray or dict

        Examples
        --------
        >>> c = rt.Categorical(['a','a','b','c','d'])
        >>> c.categories()
        FastArray([b'Inv', b'a', b'b', b'c', b'd'], dtype='|S1')

        >>> c = rt.Categorical([rt.arange(3), rt.FA(['a','b','c'])])
        >>> c.categories()
        {'key_0': FastArray([-2147483648,           0,           1,           2]),
         'key_1': FastArray([b'Inv', b'a', b'b', b'c'], dtype='|S3')}

        >>> c = rt.Categorical(rt.arange(3), {'a':0, 'b':1, 'c':2})
        >>> c.categories()
        {0: 'a', 1: 'b', 2: 'c'}
        """
        # mapping
        if self.isenum:
            return self.category_mapping

        if self.ismultikey:
            cdict = self.category_dict
            # note: multikey categoricals don't support custom invalid,
            # will use default for array dtype
            if showfilter and self.base_index == 1:
                stacked = {}
                for k,v in cdict.items():
                    stacked[k] = self._prepend_invalid(v)
                cdict = stacked
            return cdict

        # single key
        else:
            arr = self.category_array
            if showfilter and self.base_index == 1:
                arr = self._prepend_invalid(arr)
            return arr

    @property
    def _categories(self):
        return self._categories_wrap.get_categories()

    # -----------------------------------------------------------------------------------
    @property
    def category_array(self):
        """
        When possible, returns the array of stored unique categories, otherwise raises an error.

        Unlike the default for categories(), this will not prepend the invalid category.
        """
        return self._categories_wrap._get_array()

    @property
    def category_codes(self):
        return self._categories_wrap._get_codes()

    @property
    def category_mapping(self):
        return self._categories_wrap._get_mapping()

    @property
    def category_dict(self) -> Mapping[str, FastArray]:
        """
        When possible, returns the dictionary of stored unique categories, otherwise raises an error.

        Unlike the default for categories(), this will not prepend the invalid category to each array.
        """
        return self._categories_wrap._get_dict()

    # -----------------------------------------------------------------------------------
    @property
    def ordered(self) -> bool:
        """
        If the categorical is tagged as ordered, the unique categories will remain in the order they were provided in.

        `ordered` is also true if a sort was performed when generating the unique categories.
        """
        return self.grouping.isordered

    # -----------------------------------------------------------------------------------
    @property
    def sorted(self) -> bool:
        """
        If the categorical is tagged as sorted, it can use a binary search when performing a lookup in the unique categories.

        If a sorted groupby operation is performed, no sort will need to be applied.
        """
        return self._sorted
        #return self.grouping.isordered

    # -----------------------------------------------------------------------------------
    @property
    def invalid_category(self):
        """The value considered invalid. Not the same as filtered (which are marked with bin 0).

        Invalid category may still be part of the unique values.

        See Also
        --------
        Categorical.filtered_name
        """
        return self._categories_wrap._invalid_category

    def invalid_set(self, inv: Union[bytes, str]) -> None:
        """
        Set a new string to be displayed for invalid items.
        """
        if isinstance(inv, bytes):
            inv.decode()
        if not isinstance(inv, str):
            raise TypeError(f"Invalid category must be a string, not {type(inv)}")
        self._categories_wrap._invalid_category = inv

    @property
    def filtered_name(self) -> str:
        """Item displayed when a 0 bin is encountered.
        Will be omitted from groupby results by default.
        """
        return self._categories_wrap._filtered_name

    def filtered_set_name(self, name:str):
        """
        Set the name or value that will be displayed for filtered categories.
        Default is FILTERED_LONG_NAME
        """
        #**changed invalid behavior, imitates what invalid category used to do
        self._categories_wrap._filtered_name = name

    # -----------------------------------------------------------------------------------
    def copy_invalid(self):
        return self.fill_invalid(inplace=False)

    # -----------------------------------------------------------------------------------
    def fill_invalid(self, shape=None, dtype=None, order=None, inplace=True):
        """
        Returns a Categorical full of invalids, with reference to same categories.
        Must be base index 1.
        """
        if self.base_index == 1:
            if shape is None:
                shape = self.shape

            elif not isinstance(shape, tuple):
                shape = (shape,)

            if dtype is None:
                dtype = self.dtype

            if inplace is True:
                # inplace must have same length and dtype
                if shape != self.shape:
                   raise ValueError(f"Inplace fill invalid cannot be different number of rows than existing categorical. Got {shape} vs. length {len(self)}")
                if dtype != self.dtype:
                    raise ValueError(f"Inplace fill invalid cannot be different dtype than existing categorical. Got {dtype} vs. {len(self.dtype)}")
                self._fa.fill(0)
            else:
                arr = full(shape, 0, dtype=dtype)
                return type(self)(arr, _from_categorical=self.grouping)
        else:
            raise TypeError(f"Cannot return invalid copy when base index is not 1.")

    # -----------------------------------------------------------------------------------
    @property
    def nan_index(self) -> int:
        if self.base_index == 1:
            return 0
        else:
            raise TypeError(f"Categorical of base index {self.base_index} has no explicit invalid index.")

    # -----------------------------------------------------------------------------------
    @property
    def sort_gb(self) -> bool:
        return self._sort_gb

    # -----------------------------------------------------------------------------------
    def one_hot_encode(self, dtype:Optional[np.dtype]=None, categories=None, return_labels:bool=True) -> Tuple[FastArray, List[FastArray]]:
        """
        Generate one hot encoded arrays from each unique category.

        Parameters
        ----------
        dtype : data-type, optional
            The numpy data type to use for the one-hot encoded arrays. If `dtype` is not specified (i.e. is ``None``),
            the encoded arrays will default to using a ``np.float32`` representation.
        categories : list or array-like, optional
            List or array containing unique category values to one-hot encode.
            Specify this when you only want to encode a subset of the unique category values.
            Defaults to None, in which case all categories are encoded.
        return_labels : bool
            Not implemented.

        Returns
        -------
        col_names : FastArray
            FastArray of column names (unique categories as unicode strings)
        encoded_arrays : list of FastArray
            list of one-hot encoded arrays for each category

        Notes
        -----
        Unicode is used because the column names are often going to a dataset.

        *performance warning for large amount of uniques - an array will be generated for ALL of them

        Examples
        --------
        Default:

        >>> c = rt.Categorical(FA(['a','a','b','c','a']))
        >>> c.one_hot_encode()
        (FastArray(['a', 'b', 'c'], dtype='<U1'),
         [FastArray([1., 1., 0., 0., 1.], dtype=float32),
          FastArray([0., 0., 1., 0., 0.], dtype=float32),
          FastArray([0., 0., 0., 1., 0.], dtype=float32)])

        Custom dtype:

        >>> c.one_hot_encode(dtype=np.int8)
        c.one_hot_encode(dtype=np.int8)
        (FastArray(['a', 'b', 'c'], dtype='<U1'),
         [FastArray([1, 1, 0, 0, 1], dtype=int8),
          FastArray([0, 0, 1, 0, 0], dtype=int8),
          FastArray([0, 0, 0, 1, 0], dtype=int8)])

        Specific categories:

        >>> c.one_hot_encode(categories=['a','b'])
        (FastArray(['a', 'b'], dtype='<U1'),
         [FastArray([ True,  True, False, False,  True]),
          FastArray([False, False,  True, False, False])])

        Multikey:
        _Note: the double-quotes in the category names are not part of the actual string_.

        >>> c = rt.Categorical([rt.FA(['a','a','b','c','a']), rt.FA([1, 1, 2, 3, 1]) ] )
        >>> c.one_hot_encode()
        (FastArray(["('a', '1')", "('b', '2')", "('c', '3')"], dtype='<U10'),
         [FastArray([1., 1., 0., 0., 1.], dtype=float32),
          FastArray([0., 0., 1., 0., 0.], dtype=float32),
          FastArray([0., 0., 0., 1., 0.], dtype=float32)])

        Mapping:

        >>> c = rt.Categorical(rt.arange(3), {'a':0, 'b':1, 'c':2})
        >>> c.one_hot_encode()
        (FastArray(['a', 'b', 'c'], dtype='<U1'),
         [FastArray([1., 0., 0.], dtype=float32),
          FastArray([0., 1., 0.], dtype=float32),
          FastArray([0., 0., 1.], dtype=float32)])
        """
        # default to float 32
        if dtype is None:
            dtype = np.dtype(np.float32)
        else:
            dtype = np.dtype(dtype)

        # don't need to make a copy if same itemsize as boolean
        use_view = False
        if dtype.itemsize == 1:
            use_view = True

        one_hot_list = []

        # generate a column for all categories
        if categories is None:
            # array or single key
            if self.issinglekey:
                cat_list = self.category_array.astype('U')
                idx_list = range(self.base_index, len(cat_list)+self.base_index)

            # multikey
            elif self.ismultikey:
                cat_list = FastArray([str(label) for label in self.ismultikey_labels], dtype='U', unicode=True)
                idx_list = range(self.base_index, len(cat_list)+self.base_index)

            # mapping
            elif self.isenum:
                cdict = self.category_mapping
                cat_list = FastArray(list(cdict.values()), dtype='U', unicode=True)
                # use codes instead of range
                idx_list = list(cdict.keys())
            else:
                raise NotImplementedError

            # create one hot encoded arrays
            for idx in idx_list:
                # itemsize was the same e.g. bool -> int8
                if use_view:
                    one_hot_list.append( (self._fa == idx).view(dtype) )
                # itemsize was different e.g. bool -> float32
                else:
                    one_hot_list.append( (self._fa == idx).astype(dtype) )

        # only generate columns for specific categories
        else:
            if not isinstance(categories, list):
                categories = []
            for c in categories:
                one_hot_list.append(self == c)
            cat_list = FastArray(categories, dtype='U', unicode=True)

        return cat_list, one_hot_list

    # -------------------------------------------------------------------------
    def _copy_extra(self, cat_copy):
        """
        Internal routine to move over some extra data from self
        """
        _copy_name(self, cat_copy)

    # -------------------------------------------------------------------------
    def copy(self, categories=None,                    # main data
                ordered=None, sort_gb=None,  lex=None,       # sorting/hashing
                base_index=None, filter=None,          # priority options
                dtype=None, unicode=None, invalid=None, auto_add=False, # misc options
                from_matlab=False, _from_categorical=None,
                deep=True, order='K'):      # origin, possible fast track

        # raise error on keywords supplied that don't make sense
        error_kwargs = {'categories':categories, '_from_categorical':_from_categorical}
        for k, v in error_kwargs.items():
            if v is not None:
                raise ValueError(f'Cannot set keyword {k} if copy or construction from categorical.')

        # warn on soft keywords that won't be transfered
        # TODO: see if we can change any of these
        warn_kwargs = {'ordered':ordered, 'sort_gb':sort_gb, 'lex':lex,
                  'base_index':base_index, 'dtype':dtype, 'unicode':unicode,
                  'invalid': invalid}
        for k, v in warn_kwargs.items():
            if v is not None:
                warnings.warn(f'Setting keyword {k} not supported. Using original instead.')

        # categories object will be copied within filtered routine
        if filter is not None:
            return self.filter(filter=filter)

        # TODO: copy grouping object and pass to new categorical
        # unless filter is provided, don't trim unused categories

        # NOTE: there was a deep grouping copy here (removed)
        # and another deep copy when the class is made
        cat_copy = self._categories_wrap.copy(deep=False)

        if deep:
            # TJD something off about this since grouping will copy the ikey and thus ignore this copy
            idx_copy = self._fa.copy()
        else:
            idx_copy = self._fa

        self._copy_extra(cat_copy)

        # most attributes are sent to Categories object
        cat_copy= __class__(idx_copy, _from_categorical=cat_copy,
                              base_index=self.base_index, sort_gb=self._sort_gb,
                              ordered=self._ordered, invalid=self.invalid_category)

        return cat_copy

    # ------------------------------------------------------------------------------
    def filter(self, filter:Optional[np.ndarray]=None) -> 'Categorical':
        """
        Apply a filter to the categorical's values. If values no longer occur in the uniques,
        the uniques will be reduced, and the index will be recalculated.

        Parameters
        ----------
        filter : boolean array, optional
            If provided, must be the same size as the categorical's underlying array. Will be used
            to mask non-unique values.
            If not provided, categorical may still reduce its unique values to the unique occuring values.

        Returns
        -------
        c : Categorical
            New categorical with possibly reduced uniques.
        """
        # mapped categoricals will be flipped to array
        if self.isenum:
            ikey=self._fa
            if filter is not None:
                if filter.dtype.char == '?':
                    # set the invalids (technically filtering not allowed on an enum)
                    ikey[~filter] = ikey.inv
                else:
                    mask = ones(len(ikey), dtype='?')
                    mask[filter] = False
                    ikey[mask] = ikey.inv

            # get the uniques
            uniques = unique(ikey, sorted=False)

            # now get expected uniques
            unumbers=FastArray(list(self.categories().keys()))
            ustrings=FastArray(list(self.categories().values()))

            # find out which values still remain
            mask, index = ismember(unumbers, uniques)
            newdict = {k:v for k,v in zip(unumbers[mask], ustrings[mask])}

            if filter is not None:
                # add filtered into the dict
                newdict[ikey.inv] = 'Filtered'

            result = Categorical( ikey, newdict, ordered=False, sort_gb=self._sort_gb)
            # need to unset new grouping's dirty flag

        # all others will be flipped to base index 1
        else:
            newgroup = self.grouping.regroup(filter=filter, ikey=self._fa)
            if self.base_index == 0:
                warnings.warn(f'Base index was 0, returned categorical will use 1-based indexing.')
            result = Categorical(newgroup)

        self._copy_extra(result)
        return result

    # ------------------------------------------------------------------------------
    @classmethod
    def newclassfrominstance(cls, instance, origin):
        """
        Used when the FastArray portion of the Categorical is updated, but not the reset of the class attributes.

        Examples
        --------
        >>> c=rt.Cat(['a','b','c'])
        >>> rt.Cat.newclassfrominstance(c._fa[1:2],c)
        Categorical([b]) Length: 1
          FastArray([2], dtype=int8) Base Index: 1
          FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3
        """
        if isinstance(instance, cls):
            instance = instance._fa
        return cls(instance, _from_categorical=origin.grouping, base_index=origin.base_index, ordered=origin._ordered, sort_gb=origin._sort_gb)

    # ------------------------------------------------------------
    def shift_cat(self, periods:int=1) -> 'Categorical':
        """
        See FastArray.shift()
        Instead of nan or sentinel values, like shift on a FastArray, the invalid category will appear.
        Returns a new categorical.

        Examples
        --------
        >>> rt.Cat(['a','b','c']).shift(1)
        Categorical([Filtered, a, b]) Length: 3
        FastArray([0, 1, 2], dtype=int8) Base Index: 1
        FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3
        """
        temp = FastArray.shift(self, periods=periods, invalid=0)
        return self.newclassfrominstance(temp, self)

    #-------------------------------------------------------
    def shift(self, *args, window:int=1, **kwargs):
        """
        Shift each group by periods observations
        Parameters
        ----------
        window : integer, default 1 number of periods to shift
        periods: optional support, same as window
        """
        # support for pandas periods keyword
        window = kwargs.get('periods',window)
        return self._calculate_all(GB_FUNCTIONS.GB_ROLLING_SHIFT, *args, func_param=(window), **kwargs)

    @classmethod
    def _from_meta_data(cls, arrdict, arrflags, meta):
        meta = MetaData(meta)
        name = meta['name']

        # load defaults for the current version
        vars = meta['instance_vars']
        for k,v in cls.MetaDefault['instance_vars'].items():
            vars.setdefault(k,v)
        for k,v in cls.MetaDefault.items():
            meta.setdefault(k,v)
        mode = vars['mode']

        instance = arrdict.pop(name)
        prefix_len = len(name)
        arrdict = { k[prefix_len:]:v for k,v in arrdict.items() }

        # enum
        if mode in Categories.dict_modes:
            catmode = False
            cats = dict(zip(arrdict['codes'], arrdict['values']))
        else:
            catmode = True
            cats = arrdict

        grp = Grouping(instance, cats, base_index=vars['base_index'], ordered=vars['ordered'], sort_display=vars['sort_gb'], categorical=catmode, unicode=True, _trusted=True)
        result = cls(grp)

        # build the categorical from grouping
        cats = result._categories_wrap
        for k,v in meta['cat_vars'].items():
            setattr(cats, k, v)

        return result

    def _meta_dict(self, name=None):
        classname= self.__class__.__name__
        if name is None:
            name = classname

        metadict = {
            # vars for container loader
            'name': name,
            'typeid': getattr( TypeId, classname),
            'classname' : classname,
            'version': self.MetaVersion,
            'author' : 'python',

            # vars for additional arrays
            'colnames' : [],
            'ncols' : 0,

            # vars to rebuild the same categorical
            'instance_vars' : {
                'mode' : self.category_mode,
                'base_index' : self.base_index,
                'ordered' : self.ordered,
                'sorted' : self._sorted,
                'sort_gb' : self._sort_gb
            },

            'cat_vars' : {
                '_invalid_category' : self.invalid_category,
                '_filtered_name' : self.filtered_name
            }
        }
        return metadict

    def _as_meta_data(self, name=None):
        """
        Parameters
        ----------
        name : string, optional
            If not specified, will attempt to get name with get_name(), otherwise use class name.

        Returns
        -------
        arrdict : dictionary
            Dictionary of column names -> arrays.
            Extra columns (for unique categories) will have the name+'!' before their keys.
        arrtypes : list
            List of SDSFlags, same length as arrdict.
        meta : json-encoded string
            Meta data for the categorical.

        See Also
        --------
        _from_meta_data
        """
        # default to assigned name here
        # if still None, _meta_dict will use class name
        if name is None:
            name = self.get_name()

        meta = MetaData(self._meta_dict(name=name))
        name = meta['name']
        arrprefix = name+'!'

        if self.isenum:
            # still no API to access grouping enum object
            arrdict = {}
            # what are these arrays called?
            arrdict[arrprefix+'codes']=self.grouping._enum.code_array
            arrdict[arrprefix+'values']=self.grouping._enum.category_array
        else:
            arrdict = { arrprefix+k:v for k,v in self.grouping.uniquedict.items() }

        # copied from _build_sds_meta_data()
        meta['ncols']=len(arrdict)
        # use name without ! prefix here
        meta['colnames'] = [ colname[len(arrprefix): ] for colname in arrdict ]
        arrtypes = [SDSFlag.Stackable]*meta['ncols'] + [SDSFlag.OriginalContainer + SDSFlag.Stackable]

        # add the instance array
        arrdict[name]=self._fa

        return arrdict, arrtypes, meta.string

    # --------------------------------------------------------------------------------------------------
    def _autocomplete(self) -> str:
        return f'Cat u:{self.unique_count}'

    # --------------------------------------------------------------------------------------------------
    def _build_sds_meta_data(self, name, **kwargs) -> Tuple[MetaData, List[FastArray], List[Tuple[str, SDSFlag]]]:
        """
        Generates meta data from calling categorical, assembles arrays to represent its unique categories.

        Parameters
        ----------
        name : name of the categorical in the calling structure, or Categorical by default

        Returns
        -------
        meta : MetaData
            Metadata object for final save
        cols : list of FastArray
            arrays to represent unique categories - regardless of CategoryMode
        tups : tuples with names of addtl. cols - still determining enum for second item in tuple (will relate to multiday load/concatenation)
               names will be in the format 'name!col_' followed by column number
        """
        def addmeta(arr, name):
            cols.append(arr)
            meta['colnames'].append(name)
            # check if the unique array is special (example DateTimeNano class)
            if hasattr(arr, '_build_sds_meta_data'):
                newmeta, _, _ = arr._build_sds_meta_data(name)
                # add the meta data for the special class
                meta[name + '_meta'] = newmeta.dict

        meta = MetaData(self._meta_dict(name=name))

        cols : List[FastArray] = []
        # flags for meta tuples in SDS file format (see SDSFlag in rt_enum)
        array_flags : SDSFlag = 0

        # stringarray
        if self.issinglekey:
            addmeta(self.category_array, Categories.default_colname)
            array_flags += SDSFlag.Stackable

        # multikey
        elif self.ismultikey:
            # values pulled into list of arrays, custom names stored in metadata
            for colname, arr in self.category_dict.items():
                addmeta(arr, colname)
            array_flags += SDSFlag.Stackable

        # mapping
        elif self.isenum:
            # mapping will split its dictionary into an array of keys, and array of values
            # will be re-zipped during load
            mapping = self.category_mapping
            cols.append(FastArray(list(mapping.keys())))
            cols.append(FastArray(list(mapping.values())))
            meta['colnames'].append('codes')
            meta['colnames'].append('values')

        else:
            raise NotImplementedError(f"Don't know how to save Categorical in type {self.category_mode.name}")

        meta['ncols'] = len(cols)
        # generate tuples for extra columns
        # TODO: change the 6 to something indicative of hstack
        # will categorical uniques always get hstacked?
        # TODO: Create column name with f-string here instead.
        tups = [((name+'!col_'+str(i)).encode(), array_flags) for i in range(len(cols))]
        return meta, cols, tups


    # --------------------------------------------------------------------------------------------------
    @classmethod
    def _load_from_sds_meta_data(cls, name, arr, cols, meta):
        """
        Builds a categorical object from metadata and arrays.

        Will translate metadata, array/column layout from older versions to be compatible with current loader.
        Raises an error if the metadata version is higher than the class's meta version (user will need to update riptable)

        Parameters
        ----------
        name : item's name in the calling container, or the classname Categorical by default
        arr  : the underlying index array for the categorical
        cols : additional arrays to rebuild unique categories
        meta : meta data generated by build_sds_meta_data() routine

        Returns
        -------
        Categorical
            Reconstructed categorical object.

        Examples
        --------
        >>> m = y._build_sds_meta_data('y')
        >>> rt.Categorical._load_from_sds_meta_data('y', y._fa, m[1], m[0])
        """
        # build meta data from json string
        if not isinstance(meta, MetaData):
            meta = MetaData(meta)

        # load defaults for the current version
        vars = meta['instance_vars']
        for k,v in cls.MetaDefault['instance_vars'].items():
            vars.setdefault(k,v)
        for k,v in cls.MetaDefault.items():
            meta.setdefault(k,v)

        version = meta['version']
        # conversion code for each previous version. data may be stored differently, need to extract in the correct
        # way for the current version's loader
        if version != cls.MetaVersion:
            if version == 0:
                # Changes from version 0:
                # single numeric arrays are now held as lists after the constructor
                # perviously, they were held in single-key dictionaries. they will continue to be loaded as single-key dictionaries
                pass
            elif version == 1:
                pass
            else:
                raise ValueError(f"Categorical cannot load.  Version {version!r} not supported. Update riptable.")
        # catch reconstruction without extra columns (will be passed in as list of None for each extra column)
        for c in cols:
            if c is None:
                raise ValueError(f"Could not reconstruct Categorical in {CategoryMode(vars['mode']).name} mode without extra data for unique values.")

        return cls(arr, cols, _from_categorical=meta)

    # ------------------------------------------------------------
    def lock(self):
        """
        Locks the categories to none can be added, removed, or change.
        """
        self._locked = True

    # ------------------------------------------------------------
    def unlock(self):
        """
        Unlocks the categories so new categories can be added, or existing categories can be removed or changed.
        """
        self._locked = False

    # -------------------------------------------------------
    def auto_add_on(self):
        """
        If the categorical is unlocked, this sets the _auto_add_categories flag to be True.
        If _auto_add_categories is set to False, the following assignment will raise an error.
        If the categorical is locked, auto_add_on() will warn the user and the flag will not change.

        Examples
        --------
        >>> c = rt.Categorical(['a','a','b','c','a'])
        >>> c._categories
        FastArray([b'a', b'b', b'c'], dtype='|S1')
        >>> c.auto_add_on()
        >>> c[0] = 'z'
        >>> print(c)
        z, a, b, c, a
        >>> c._categories
        FastArray([b'a', b'b', b'c', b'z'], dtype='|S1')
        """
        if self._locked is False:
            self._auto_add_categories=True
            self._categories_wrap._auto_add_categories=True
        else:
            warnings.warn(f"Categorical is locked and cannot automatically add categories.")

    # -------------------------------------------------------
    def auto_add_off(self):
        """
        Sets the _auto_add_categories flag to False. Category assignment with a non-existing categorical
        will raise an error.

        Examples
        --------
        >>> c = rt.Categorical(['a','a','b','c','a'], auto_add_categories=True)
        >>> c._categories
        FastArray([b'a', b'b', b'c'], dtype='|S1')
        >>> c.auto_add_off()
        >>> c[0] = 'z'
        ValueError: Cannot automatically add categories [b'z'] while auto_add_categories is set to False.
        """
        self._auto_add_categories=False
        self._categories_wrap._auto_add_categories=False

    # -------------------------------------------------------
    def mapping_add(self, code, value):
        """
        Add a new code -> value mapping to categories.
        """
        if self._locked is False:
            self._categories_wrap._mapping_edit(code, value=value, how='add')
        else:
            raise ValueError(f"Cannot add mapping to a locked Categorical. Call unlock() first.")
        self.groupby_reset()

    # -------------------------------------------------------
    def mapping_remove(self, code):
        """
        Remove the category associated with an integer code.
        """
        if self._locked is False:
            self._categories_wrap._mapping_edit(code, how='remove')
        else:
            raise ValueError(f"Cannot remove mapping a locked Categorical. Call unlock() first.")
        self.groupby_reset()

    # -------------------------------------------------------
    def mapping_replace(self, code, value):
        """
        Replace a single integer code with a single value.
        """
        if self._locked is False:
            self._categories_wrap._mapping_edit(code, value=value, how='replace')
        else:
            raise ValueError(f"Cannot replace mapping in a locked Categorical. Call unlock() first.")
        self.groupby_reset()

    # -------------------------------------------------------
    def mapping_new(self, mapping):
        """
        Replace entire mapping dictionary. No codes in the Categorical's integer FastArray will be changed. If they are not in the
        new mapping, they will appear as Invalid.
        """
        if self._locked is False:
            self._categories_wrap._mapping_new(mapping)
        else:
            raise ValueError(f"Cannot replace mapping dictionary in a locked Categorical. Call unlock() first.")
        self.groupby_reset()

    # -------------------------------------------------------
    def category_add(self, value):
        """
        New category will always be added to the end of the category array.
        """
        if self._locked is False:
            self._categories_wrap._array_edit(value, how='add')
            self._ordered = False
            self._sorted = False
        else:
            raise ValueError(f"Cannot add category to locked Categorical. Call unlock() first.")
        self.groupby_reset()

    # -------------------------------------------------------
    def category_remove(self, value):
        """
        Performance may suffer as indices need to be fixed up. All previous matches to the removed
        category will be flipped to invalid.
        """
        if self._locked is False:
            remove_code = self._categories_wrap._array_edit(value, how='remove')
            if remove_code is not None:
                prev_match = self._fa == remove_code
                if self.base_index >= 1:
                    inv = 0
                else:
                    inv = -1
                self._fa[prev_match] = inv
                gt_match = self._fa > remove_code
                self._fa[gt_match] -= 1
        else:
            raise ValueError(f"Cannot remove category from locked Categorical. Call unlock() first.")
        self.groupby_reset()

    # -------------------------------------------------------
    def category_replace(self, value, new_value):
        if self._locked is False:
            fix_index_tup = self._categories_wrap._array_edit(value, new_value=new_value, how='replace')
            if fix_index_tup is not None:
                replace_mask = self._fa == fix_index_tup[0]
                self._fa[replace_mask] = fix_index_tup[1]
            else:
                self._ordered = False
                self._sorted = False
        else:
            raise ValueError(f"Cannot remove category from locked Categorical. Call unlock() first.")
        self.groupby_reset()

    # -------------------------------------------------------
    def map(self, mapper: Union[dict, np.array], invalid=None) -> FastArray:
        """
        Maps existing categories to new categories and returns a re-expanded array.

        Parameters
        ----------
        mapper : dictionary or numpy.array or FastArray
            - dictionary maps existing categories -> new categories
            - array must be the same size as the existing category array
        invalid
            Optionally specify an invalid value to insert for existing categories that were not found in the new mapping.
            If no invalid is set, the default invalid for the result's dtype will be used.

        Returns
        -------
        FastArray
            Re-expanded array.

        Notes
        -----
        Maybe to add:
        - option to return categorical instead of re-expanding
        - dtype for return array

        Examples
        --------
        New strings (all exist, no invalids in original):

        >>> c = rt.Categorical(['b','b','c','a','d'], ordered=False)
        >>> mapping = {'a': 'AA', 'b': 'BB', 'c': 'CC', 'd': 'DD'}
        >>> c.map(mapping)
        FastArray([b'BB', b'BB', b'CC', b'AA', b'DD'], dtype='|S3')

        New strings (not all exist, no invalids in original):

        >>> mapping = {'a': 'AA', 'b': 'BB', 'c': 'CC'}
        >>> c.map(mapping, invalid='INVALID')
        FastArray([b'BB', b'BB', b'CC', b'AA', b'INVALID'], dtype='|S7')

        String to float:

        >>> mapping = {'a': 1., 'b': 2., 'c': 3.}
        >>> c.map(mapping, invalid=666)
        FastArray([  2.,   2.,   3.,   1., 666.])

        If no invalid is specified, the default invalid will be used:

        >>> c.map(mapping)
        FastArray([ 2.,  2.,  3.,  1., nan])

        Mapping as array (must be the same size):

        >>> mapping = rt.FastArray(['w','x','y','z'])
        >>> c.map(mapping)
        FastArray([b'w', b'w', b'x', b'y', b'z'], dtype='|S3')
        """
        # --------------------
        def invalid_value(invalid, newcats):
            # return an invalid string or sentinel value
            # string values display as Inv - not empty string
            if invalid is None:
                if newcats.dtype.char in NumpyCharTypes.AllInteger+NumpyCharTypes.AllFloat:
                    invalid = INVALID_DICT[newcats.dtype.num]
                elif newcats.dtype.char in 'US':
                    invalid = 'Inv'
                else:
                    raise TypeError(f"No invalid map fill for array of type {newcats.dtype}")
            return invalid
        # --------------------
        def set_invalid(c, invalid):
            inv_mask = None
            inv_fill = invalid_value(invalid, c.category_array)
            # item will be inserted automatically if it's the filtered string
            if isinstance(inv_fill, (str, bytes)):
                c.filtered_set_name(inv_fill)

            # otherwise build a mask, insert later
            # maybe support numeric in filtered name?
            else:
                inv_mask = c.isfiltered()

            return c, inv_mask, inv_fill
        # --------------------

        inv_mask = None
        if self.issinglekey:
            if isinstance(mapper, dict):
                oldcats = FastArray([*mapper])
                newcats = FastArray([*mapper.values()])

                has_inv, catidx = ismember(self.category_array, oldcats, base_index=1)

                # base 1 only
                if has_inv:
                    expanded = self.expand_array
                    # 0 bin for values not found in the mapping
                    _, instance = ismember(expanded, oldcats, base_index=1)
                    grp = Grouping(instance, newcats, base_index=1, categorical=True, _trusted=True)
                    c = Categorical(grp)
                    c, inv_mask, inv_fill = set_invalid(c, invalid)

                # all categories were found, quick swap
                else:
                    grp = Grouping(self._fa, newcats[catidx-1], base_index=self.base_index, categorical=True, _trusted=True)
                    c = Categorical(grp)

            # assumes that array input corresponds to unique category array
            elif isinstance(mapper, np.ndarray):
                if len(mapper) == len(self.category_array):
                    grp = Grouping(self._fa, mapper, categorical=True, _trusted=True, base_index=self.base_index)
                    c = Categorical(grp)
                    if self.base_index == 1:
                        c, inv_mask, inv_fill = set_invalid(c, invalid)
                else:
                    raise ValueError(f"Length of replacement values {len(mapper)} did not match length of existing uniques {len(self.category_array)}")

            else:
                raise TypeError(f"mapping must be a dictionary or array. Got {type(mapper)}")

        else:
            raise TypeError(f"Could not perform map on categorical in mode {CategoryMode(self.category_mode)}.")

        result = c.expand_array
        if inv_mask is not None:
            putmask(result, inv_mask, inv_fill)

        return result


    # -------------------------------------------------------
    def shrink(self, newcats, misc=None, inplace:bool=False) -> 'Categorical':
        """
        Parameters
        ----------
        newcats : array-like
            New categories to replace the old - typically a reduced set.
        misc : scalar, optional (often a string)
            Value to use as category for items not found in new categories. This will be added to the new categories.
            If not provided, all items not found will be set to a filtered bin.
        inplace : bool
            If True, re-index the categorical's underlying FastArray.
            Otherwise, return a new categorical with a new index and grouping object.

        Returns
        -------
        Categorical
            A new Categorical with the new index.

        Examples
        --------
        Base index 1, no misc

        >>> c = rt.Categorical([1,2,3,1,2,3,0], ['a','b','c'])
        >>> c.shrink(['b','c'])
        Categorical([Filtered, b, c, Filtered, b, c, Filtered]) Length: 7
          FastArray([0, 1, 2, 0, 1, 2, 0]) Base Index: 1
          FastArray([b'b', b'c'], dtype='|S1') Unique count: 2

        Base index 1, filtered bins and misc

        >>> c.shrink(['b','c'], 'AAA').sum(rt.arange(7), showfilter=True)
        *key_0     col_0
        --------   -----
        Filtered       6
        AAA            3
        b              5
        c              7

        Base index 0, with misc

        >>> c = rt.Categorical([0,1,2,0,1,2], ['a','b','c'], base_index=0)
        >>> c.shrink(['b','c'], 'AAA')
        Categorical([AAA, b, c, AAA, b, c]) Length: 6
          FastArray([0, 1, 2, 0, 1, 2], dtype=int8) Base Index: 0
          FastArray(['AAA', 'b', 'c'], dtype='<U3') Unique count: 3

        See also
        --------
        Categorical.map()
        """

        # generate integer array for new categorical
        grp = self.grouping.shrink(newcats, misc=misc, inplace=inplace, name=self.get_name())

        # write over own index array
        if inplace:
            self[:] = grp.catinstance
            self._grouping = grp

            # because not going through __init__, need to sync up new grouping uniques with categories wrap
            self._categories_wrap = Categories.from_grouping(grp, invalid_category=self.invalid_category)
            return self

        # return a new categorical
        else:
            result = Categorical(grp)
            result.filtered_set_name(self.filtered_name)
            return result


    # -------------------------------------------------------
    def isin(self, values) -> FastArray:
        """
        Parameters
        ----------
        values: a list-like or single value to be searched for

        Returns
        -------
        FastArray
            Boolean array with the same size as `self`. True indicates that the array element
            occured in the provided `values`.

        Notes
        -----
        Behavior differs from pandas in the following ways:
        * Riptable favors bytestrings, and will make conversions from unicode/bytes to match for operations as necessary.
        * We also accept single scalars for `values`.
        * Pandas series will return another series - we have no series, and will return a FastArray.

        Examples
        --------
        >>> c = rt.Categorical(['a','b','c','d','e'], unicode=False)
        >>> c.isin(['a','b'])
        FastArray([ True,  True, False, False, False])

        See Also
        --------
        pandas.Categorical.isin()
        """
        x = values

        if isinstance(x, Categorical):
            if x.ismultikey:
                return ismember(self, x)[0]
        # handle enum + non-categorical with grouping
        elif self.isenum:
            return self.grouping.isin(x)

        elif isinstance(values, (bool, np.bool, bytes, str, int, np.integer, float, np.floating)):
            x = np.array([x])
        # numpy will find the common dtype (strings will always win)
        elif isinstance(x, (list, tuple)):
            if self.category_mode == CategoryMode.NumericArray and isinstance(x,list):
                # user allowed to pass in floats as strings
                x=np.asarray(x,dtype=self.category_array.dtype)
            x = np.array(x)

        # both ismember and == handle categorical specially
        if isinstance(x, (list, np.ndarray)):
            if len(x) > 1:
                return ismember(self, x)[0]
            elif np.isscalar(x[0]):
                return self == x[0]
        return self == x

    # -------------------------------------------------------
    def __setitem2__(self, key, value):
        """
        Use grouping object isin, single item accessor instead of Categories object.
        """

        if self._locked:
            raise IndexError(f"Cannot set item because Categorical is locked.")

        # LEFT SIDE
        if isinstance(key, list):
            key = FastArray(key)

        # let boolean, fancy, or single index pass through
        if not (isinstance(key, np.ndarray) and key.dtype.char in NumpyCharTypes.AllInteger+'?') and \
            not isinstance(key, (int, np.integer)):
            # single item, arrays of items in unique
            # possibly convert to boolean array
            key = self.grouping.isin(key)

        # let single int, fancy index pass through
        if not (isinstance(value, np.ndarray) and value.dtype.char in NumpyCharTypes.AllInteger) and \
            not isinstance(value, (int, np.integer)):
            # need a method in grouping to get index for single item or tuple
            # possibly add category with set item, or keep the same?
            str_idx = None
            if self.issinglekey:
                uniquelist = self.grouping.uniquelist[0]

                if self.category_mode == CategoryMode.StringArray:

                    # TODO: push the string matching up to categorical
                    value = self._categories_wrap.match_str_to_category(value)

                # sorted categories
                if self.sorted:
                    # if larger than all strings, str_idx will be len(self._categories)
                    str_idx = np.searchsorted(uniquelist, value)
                    if str_idx < self.unique_count:
                        # insertion point, not exact match
                        if value != uniquelist[str_idx]:

                            # adjust for le, ge comparisons
                            # str_idx -= 0.5
                            str_idx -= 0.5
                    str_idx += self.base_index

                # unsorted categories
                else:
                    str_idx = bool_to_fancy(uniquelist == value)
                    if len(str_idx) != 0:
                        str_idx = str_idx[0]+self.base_index  # get value from array
                    else:
                        str_idx = self.unique_count+self.base_index

            #elif self.isenum:
            #    s = self.match_str_to_category(s)
            #    str_idx = self.str2intdict.get(s, None)
            #    if str_idx is None:
            #        raise ValueError(f"{s} was not a valid category in categorical from mapping.")

            else:
                raise NotImplementedError

            if isinstance(str_idx, list):
                str_idx = str_idx[0]
            elif isinstance(str_idx, (float, np.floating)):
                raise ValueError(f"{value} was not a valid category in categorical.")
            value = str_idx

        self._fa[key] = value
        self.grouping.set_dirty()


    # -------------------------------------------------------
    def __setitem__(self, index, value):
        """
        Parameters
        ----------
        index: int or string (depends on category mode)
        value: sequence or scalar value
            The value may represent a category or category index.

        Raises
        ------
        IndexError
        """

        if isinstance(value, Categorical):
            # have to align the categoricals
            # check if already aligned, if so can use same integers
            is_same, catlist = Categorical.categories_equal([self, value])
            value = catlist[1]
            if not is_same:
                # convert back to strings (slow)
                # TODO: multikey will fail here
                value = self.expand_array

        # first check if the value is string like
        # if it is, we have to convert it to an index first
        if isinstance(value, (str, bytes, float, np.floating)):
            if self._locked:
                raise IndexError(f"Cannot add a new category {value} because index is locked.")

            if self.isenum:
                # flip string to index
                # TODO: add check for existence, possibly add if flag isn't set
                value = self.from_category(value)

            else:
                # add the category, clean up the index array afterwards
                fix_index = self._categories_wrap._possibly_add_categories(value)
                if fix_index is not None:
                    # must be inplace to change self
                    self._fa[:] = fix_index[self._fa - self.base_index] + self.base_index
                # convert string to int index
                # we know value will have an exact match
                value = self._categories_wrap.get_category_match_index(value)
                value = value[0]

        elif isinstance(value, (int, np.integer)):
            # path to replace one mapping with another
            if self.isenum:
                if self._categories_wrap._is_valid_mapping_code(value) is False:
                    raise ValueError(f"{value} was not a valid mapping code. Use mapping_add() or mapping_replace() first.")

            # check bin index for string-based categoricals
            elif self.category_mode == CategoryMode.StringArray:
                if value < 0 or value > len(self._categories_wrap)-1+self.base_index:
                    raise IndexError(f"Invalid index in category dictionary.")

            else:
                if value < 0 or value > len(self._categories_wrap)-1+self.base_index:
                    raise IndexError(f"Invalid index in category dictionary.")

        elif isinstance(value, tuple):
            if self.ismultikey:
                cat_idx = self._categories_wrap.get_multikey_index(value)
                if cat_idx > -1:
                    value = cat_idx
                else:
                    raise ValueError(f"Provided value {value} was not a valid multikey.")

        # stringlike
        if isinstance(index, (str, bytes, float, np.floating)):
            # convert string to index, let from_category raise error if not found
            index = self.from_category(index)
            index = self._fa == index

        # multikey
        elif isinstance(index, tuple):
            if self.ismultikey:
                cat_idx = self._categories_wrap.get_multikey_index(index)
                if cat_idx > -1:
                    index = self._fa == cat_idx
                else:
                    raise ValueError(f"Provided index {index} was not a valid multikey.")

        # pass final index, value to underlying fast array
        super().__setitem__(index, value)
        self.groupby_reset()

    # ------------------------------------------------------------
    @property
    def category_mode(self) -> CategoryMode:
        """
        Returns the category mode of the Categorical's Categories object.
        List modes are when the categorical has gone through the unique/mbget process of binning.
        Dict modes are when the categorical was constructed with a dictionary mapping or IntEnum.
        Grouping mode is when the categorical was binned with the groupby hash (numeric list, multikey, etc.)

        Returns
        -------
        IntEnum
            see CategoryMode in rt_enum.py
        """
        return self._categories_wrap.mode

    # ------------------------------------------------------------
    def from_bin(self, bin):
        """
        Returns the category corresponding to a single integer.
        Raises error if index is out of range (accounts for base index) - or does not exist in mapping.

        Notes
        -----
        String values will appear as the scalar type they are stored in, however FastArray,
        Categorical, and other riptable routines will convert/compensate for unicode/bytestring mismatches.

        Examples
        --------
        Base-1 Indexing:

        >>> c = rt.Categorical(['a','a','b','c','a'])
        >>> c.category_array
        FastArray([b'a', b'b', b'c'], dtype='|S1')
        >>> c.category_from_bin(2)
        b'b'

        >>> c.category_from_bin(4)
        IndexError

        Base-0 Indexing:

        >>> c = rt.Categorical(['a','a','b','c','a'], base_index=0)
        >>> c.category_from_bin(2)
        b'c'
        """

        if self.base_index is not None:
            if bin < self.base_index:
                raise ValueError(f"Bin {bin} is out of range for categorical with base index {self.base_index}")

        if not isinstance(bin, (int, np.integer)):
            raise TypeError(f"Bin must be a single integer.")
        if self.issinglekey:
            # will raise if invalid
            return self.category_array[bin-self.base_index]
        elif self.isenum:
            # will raise if mapping doesn't exist
            return self.category_mapping[bin]
        else:
            # possibly single key
            try:
                return self.category_array[bin-self.base_index]
            except:
                cdict = self.category_dict
                result = []
                for c in cdict.values():
                    result.append(c[bin-self.base_index])
                return tuple(result)

    # ------------------------------------------------------------
    def from_category(self, category):
        """
        Returns the bin associated with a category.
        If the category doesn't exist, an error will be raised.

        Note: the bin returned is the value as it appears in the underlying integer FastArray.
        It may not be a direct index into the stored unique categories.

        Unicode/bytes conversion will be handled internally.

        Examples
        --------
        Single Key (base-1):

        >>> c = rt.Categorical(['a','a','b','c','a'])
        >>> c.bin_from_category('a')
        1
        >>> c = rt.Categorical(['a','a','b','c','a'])
        >>> c.bin_from_category(b'c')
        3

        Single Key (base-0):

        >>> c = rt.Categorical(['a','a','b','c','a'], base_index=0)
        >>> c.bin_from_category('a')
        0

        Multikey:

        >>> c = rt.Categorical([rt.FA(['a','b','c']), rt.arange(3)])
        >>> c.bin_from_category(('a', 0))
        1

        Mapping:

        >>> c = rt.Categorical([1,2,3], {'a':1, 'b':2, 'c':3})
        >>> c.bin_from_category('c')
        >>> 3

        Numeric:

        >>> c = rt.Categorical(rt.FA([3.33, 5.55, 6.66]))
        >>> c.bin_from_category(3.33)
        1

        """
        bin = self._categories_wrap.get_category_index(category)
        # mapping error will be handled by Categories object
        if not self.isenum:
            if bin == len(self._categories_wrap)+self.base_index or isinstance(bin, float):
                raise ValueError(f'{category} not found in uniques.')
        return bin

    # ------------------------------------------------------------
    def __getitem__(self, fld):
        """
        Indexing
        --------
        Bracket indexing for Categoricals will *always* hit the FastArray of indices/codes first.
        If indexed by integer, the retrieved index or code will be passed to the Categories object so the
        corresponding Category can be returned. Otherwise, a new Categorical will be returned, using the
        same Categories as the original Categorical with a different index/code array.

        The following examples will use this Categorical:

        >>> c = rt.Categorical(['a','a','a','b','c','a','b'])
        >>> c
        Categorical([a, a, a, b, c, a, b]) Length: 7
            FastArray([1, 1, 1, 2, 3, 1, 2], dtype=int8) Base Index: 1
            FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

        Single Integer
        ~~~~~~~~~~~~~~
        For convenience, any bytestrings will be returned/displayed as unicode strings.

        >>> c[3]
        'b'

        Multiple Integers
        ~~~~~~~~~~~~~~~~~
        >>> c[[1,2,3,4]]
        Categorical([a, a, b, c]) Length: 4
            FastArray([1, 1, 2, 3], dtype=int8) Base Index: 1
            FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

        >>> c[np.arange(4,6)]
        Categorical([c, a]) Length: 2
            FastArray([3, 1], dtype=int8) Base Index: 1
            FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

        Boolean Array
        ~~~~~~~~~~~~~
        >>> mask = FastArray([False,  True,  True,  True,  True,  True, False])
        >>> c[mask]
        Categorical([a, a, b, c, a]) Length: 5
            FastArray([1, 1, 2, 3, 1], dtype=int8) Base Index: 1
            FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

        Slice
        ~~~~~
        >>> c[2:5]
        Categorical([a, b, c]) Length: 3
            FastArray([1, 2, 3], dtype=int8) Base Index: 1
            FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

        """
        if np.isscalar(fld):
            # for convenience:
            # c = Categorical(['a','a','b'])
            # c['a']
            # [True, True, False]
            if isinstance(fld, (str, bytes)):
                newcat = self == fld
            # pull single values from uniques
            else:
                newcat = self._getsingleitem(fld)
                # just a single value
                return newcat
        else:
            # NEW PATH
            # slice the grouping object, rebuild from grouping
            try:
                # check for list of lists and route to isin if found
                if isinstance(fld, list) and len(fld) > 0 and isinstance(fld[0], (str, bytes, tuple)):
                    return self.isin(fld)
                result = self.grouping[fld]
                newcat = self.__class__(result, _from_categorical=self._categories_wrap, base_index=self.base_index)

            # OLD PATH
            # rewriting indexing for c[['string1', 'string2']], etc.
            except (TypeError, NotImplementedError):
                fld = self._fa[fld]
                if isinstance(fld, np.ndarray):
                    newcat = self.__class__(fld, _from_categorical=self._categories_wrap, base_index=self.base_index)

                # get the uniques, base index, etc. from grouping object
                # send to categories object to translate
                newcat = self._categories_wrap[fld]
        oldname =self.get_name()
        if oldname is not None:
            newcat.set_name(oldname)
        return newcat

    # ----GETITEM OPS FROM CATEGORIES CLASS-----------------------
    # ------------------------------------------------------------
    def _getsingleitem(self, fld):
        """If the getitem indexing operation returned a scalar, translate
        it according to how the uniques are being held.

        Returns
        -------
        Scalar or tuple based on unique type.
        """
        # pull value from array (must be integer)
        fld = self._fa[fld]

        if self.isenum:
            # also integers only
            # return string <!badint> if not found
            # TODO: need a method to interface with the enum dict in grouping
            return self.grouping._enum.from_code(fld)

            # from_enum_code() ?

        # pass the single or multikey itemfunc for after validation
        # filtered and bad integer flds handled the same way for both
        elif self.issinglekey or self.ismultikey:
            if isinstance(fld, (int, np.integer)):
                if self.base_index != 0:
                    idx = fld - self.base_index
                else:
                    idx = fld
                # may need to check the dirty flag here
                if idx < 0 or idx >= self.unique_count:
                    # special display for filtered item
                    # need _filtered_string from Categories
                    # filtered and bad integer flds handled the same way for single and multikey
                    if self.base_index == 1 and fld == 0:
                        return self.filtered_name
                    return "!<"+str(fld)+">"

                # return the corresponding fld
                # adjust fld, use as index into unique array(s)
                result = [ c[idx] for c in self.grouping.uniquelist ]
                # return bytes like item(s) as strings
                for i, item in enumerate(result):
                    if isinstance(item, bytes):
                        result[i] = item.decode()
                if len(result)==1:
                    return result[0]
                # format the multikey tuple as string here, or return flds as-is?
                # (based on display_query_properties from arrays)
                # dataset display appears to handle as-is version
                return tuple(result)
            else:
                raise TypeError(f"Get single item not implemented for type {type(fld)}")
        else:
            raise TypeError(f"Critical error in Categorical getitem. Mode was {self.category_mode}")

    # ------------------------------------------------------------
    def display_query_properties(self):
        """
        Takes over display query properties for fastarray. By default, all categoricals will use left alignment.
        """
        item_format = ItemFormat(
            length          = DisplayLength.Long,
            justification   = DisplayJustification.Left,
            can_have_spaces = True,
            decoration      = None
        )
        convert_func = self.display_convert_func
        return item_format, convert_func

    # ------------------------------------------------------------
    @staticmethod
    def display_convert_func(item, itemformat:ItemFormat):
        """
        Used in conjunction with display_query_properties for final display of a categorical in a dataset.
        Removes quotation marks from multikey categorical tuples so display is easier to read.
        """
        # TODO: apply ItemFormat options that were passed in
        # strip quotation marks to avoid confusion with tuple displayed
        return str(item).replace("'","")

    # ------------------------------------------------------------
    @property
    def issinglekey(self) -> bool:
        """See Categories.singlekey
        """
        return self._categories_wrap.issinglekey

    # ------------------------------------------------------------
    @property
    def ismultikey(self) -> bool:
        """See Categories.multikey
        """
        return self._categories_wrap.ismultikey

    # ------------------------------------------------------------
    @property
    def isenum(self) -> bool:
        """See Categories.enum
        """
        return self._categories_wrap.isenum

    # --------------------------------------------------------
    def _categorical_compare_check(self, func_name, other):
        """
        Converts a category to a valid index for faster logical comparison operations on the underlying
        index fastarray.
        """

        caller = self._fa
        func = None

        # COMPARE TO INTEGER (numeric array categoricals will get handled differently)
        if isinstance(other, (int, np.integer, float, np.float)):
            # error will be raised if doesn't match categories
            if not self.isenum:
                other = self._categories_wrap.get_category_index(other)

        # COMPARE TO STRING----------------------------------
        elif isinstance(other, (bytes, str)):
            if self.category_mode != CategoryMode.StringArray and not self.isenum:
                # try to convert to int
                # this happens when c=Cat([1,2,3]); c['2']
                try:
                    # extract float or integer
                    fnum=float(other)
                    if round(fnum) == fnum:
                        other=int(other)
                    else:
                        other=fnum
                except Exception as ex:
                    raise TypeError(f"Comparisons to single strings can only be made to categoricals in StringArray mode - not {self.category_mode.name} mode.  Error {ex}")
            if func_name not in ['__eq__', '__ne__'] and not self.isenum:
                if self._ordered is False:
                    raise ValueError(f"Cannot make accurate comparison with {func_name} on unordered Categorical.")
            other = self._categories_wrap.get_category_index(other)

        # COMPARE TO ANOTHER CATEGORICAL------------------------
        elif isinstance(other, Categorical):
            if self.ismultikey:
                if other.ismultikey:
                    raise NotImplementedError(f"Comparing multikey categoricals is not currently implemented.")
                    # test if same number of columns
                    # test if same number of rows
                    # test if same type in each column
                else:
                    raise ValueError(f"Cannot compare multikey categorical to single key categorical.")

            # TODO: send this to the general hstack code
            # need a way to do this without actually stacking them
            if self.category_mode != other.category_mode:
                raise TypeError(f"Cannot compare categoricals with different modes {self.category_mode} and {other._categories_wrap.mode}")
            if self.isenum:
                if categorical_merge_dict([self, other], return_is_safe=True):
                    func = getattr(caller, func_name)
                    return func(other._np)
                else:
                    raise ValueError(f"Could not compare categoricals because of conflicting items in dictionaries.")

            else:
                oldidx = [ self._fa, other._fa ]
                oldcats = [[ self.category_array, other.category_array ]]
                newidx, _ = merge_cats(oldidx, oldcats)
                #print('***newidx', newidx)
                # merge index returns stacked
                newidx_self = newidx[:len(self)]
                newidx_other = newidx[len(self):]
                func = getattr(newidx_self, func_name)
                return func(newidx_other)

        # COMPARE TO LIST ------------------------------------------
        elif isinstance(other, (list, np.ndarray)):
            if len(other) == 0:
                raise ValueError("List was empty.")

            first_item = other[0]
            if isinstance(first_item, (str, bytes)):
                if len(other) == len(self):
                    warnings.warn(f"Comparing categorical to string array of the same array differs from regular numpy string array comparisons. Compare two categoricals to match behavior.")

                # TODO: merge this with something similar to .isin()
                other = [ self._categories_wrap.get_category_index(item) for item in other ]
                func = getattr(caller, func_name)
                return mask_ori([func(item) for item in other])

            elif isinstance(first_item, tuple):
                if self.ismultikey:
                    other = [ self._categories_wrap.get_multikey_index(item) for item in other ]
                    func = getattr(caller, func_name)
                    return mask_ori([func(item) for item in other])


        # COMPARE TO TUPLE--------------------------------------------
        elif isinstance(other, tuple):
            if self.ismultikey:
                if len(other) == self._categories_wrap.ncols:
                    other = self._categories_wrap.get_multikey_index(other)
                else:
                    raise ValueError("Number of items in tuple must match number of keys in multikey. input had {len(other)} items, this categorical has {self._categories_wrap.ncols}")
            else:
                raise TypeError("Only multikey categoricals can be accessed with compared to tuples.")
        func = getattr(caller, func_name)
        return func(other)

    # -------------------COMPARISONS------------------------------
    # ------------------------------------------------------------
    def __ne__(self, other):
        return self._categorical_compare_check('__ne__', other)

    def __eq__(self, other):
        return self._categorical_compare_check('__eq__', other)

    def __ge__(self, other):
        return self._categorical_compare_check('__ge__', other)

    def __gt__(self, other):
        return self._categorical_compare_check('__gt__', other)

    def __le__(self, other):
        return self._categorical_compare_check('__le__', other)

    def __lt__(self, other):
        return self._categorical_compare_check('__lt__', other)

    # ------POSSIBLY LAZY EVALUATIONS FOR GROUPBY-----------------
    # ------------------------------------------------------------
    def groupby_reset(self):
        """
        Resets all lazily evaluated groupby information. The categorical will go back to the state it was in
        just after construction. This is called any time the categories are modified.
        """
        # gb_keychain to be replaced by label generating methods
        self._gb_keychain = None
        # will be marked dirty / repaired by
        # internal set / modify methods in Grouping
        self.grouping.set_dirty()

    # -------------------------------------------------------
    @property
    def ikey(self):
        """
        Returns the grouping object's iKey. This will always be a 1-base index, and is often the same array as the Categorical.
        See also: grouping.ikey (may return base 0 index)
        """
        return self.grouping.ikey

    # ------------------------------------------------------------
    @property
    def ifirstkey(self):
        """
        Index of first occurrence of each unique key.
        May also trigger lazy evaluation of grouping object.
        If grouping object used the Groupby hash, it will have an iFirstKey array, otherwise returns None.
        """
        return self.grouping.ifirstkey

    # ------------------------------------------------------------
    @property
    def ilastkey(self):
        """
        Index of last occurrence of each unique key.
        May also trigger lazy evaluation of grouping object.
        If grouping object used the Groupby hash, it will have an iLastKey array, otherwise returns None.
        """
        return self.grouping.ilastkey

    # ------------------------------------------------------------
    @property
    def unique_count(self):
        """
        Number of unique values in the categorical.
        It is necessary for every groupby operation.

        Notes
        -----
        For categoricals in dict / enum mode that have generated their grouping object, this
        will reflect the number of unique values that `occur` in the non-unique values. Empty
        bins will not be included in the count.
        """
        return self.grouping.unique_count

    # ------------------------------------------------------------
    def nunique(self):
        """
        Number of unique values that occur in the Categorical.
        Does not include invalids. Not the same as the length of possible uniques.

        Categoricals based on dictionary mapping / enum will return unique count including all possibly
        invalid values from underlying array.

        See Also
        --------
        Categorical.unique_count
        """
        un = unique(self._fa, sorted=False)
        count = len(un)
        # all will be counted
        if self.isenum or self.base_index == 0:
            pass
        # array / multikey categoricals (base index 1) have invalids at 0 bin
        else:
            haszero = un == 0
            if haszero.sum():
                count -= 1
        return count

    # ------------------------------------------------------------
    @property
    def grouping_dict(self):
        """
        Grouping dict held by Grouping object.
        May trigger lazy build of Grouping object.
        """
        return self.grouping.uniquedict

    # ---------------GROUPBY OPERATIONS---------------------------
    @property
    def grouping(self):
        """
        Grouping object that is called to perform calculations on grouped data.
        In the constructor, a grouping object provides a categorical with its instance array.
        The grouping object stores and generates other groupby information, like grouping indices, first occurrence, count, etc.
        The grouping object should be queried for all grouping-related properties.
        This is also a property in GroupBy, and is called by many routines in the GroupByOps parent class.

        See Also: Grouping
        """
        return self._grouping

    #-------------------------------------------------------
    @property
    def transform(self):
        """
        TO BE DEPRECATED

        Examples
        --------
        >>> c = rt.Categorical(ds.symbol)
        >>> c.transform.sum(ds.TradeSize)
        """
        warnings.warn("Deprecation warning: Use kwarg transform=True instead of transform.")
        self._transform=True
        return self

    # ------------------------------------------------------------
    def _calculate_all(self, funcNum, *args, func_param=0, **kwargs):
        origdict, user_args, tups = self._prepare_gb_data('Categorical', funcNum, *args, **kwargs)

        # lock after groupby operation
        self._locked = True
        keychain = self.gb_keychain

        result_ds = self.grouping._calculate_all(origdict, funcNum, func_param=func_param, keychain=keychain, user_args=user_args, tups=tups, **kwargs)
        return self._possibly_transform(result_ds, label_keys=keychain.keys(), **kwargs)

    # ------------------------------------------------------------
    def apply(self, userfunc=None, *args, dataset=None, **kwargs):
        """
        See Grouping.apply for examples.
        Categorical needs remove unused bins from its uniques before an apply.
        """
        clean_c = self.filter(None)
        result = super(Categorical, clean_c).apply(userfunc, *args, dataset=dataset, label_keys=clean_c.gb_keychain, **kwargs)
        # result is the same size as original, attach categorical (the key column) to result
        if result.shape[0] == len(clean_c):
            name = self.get_header_names([self], default='gb_key_')[0]
            result[name] = self
            result.label_set_names(name)
        return result

    # ------------------------------------------------------------
    def apply_nonreduce(self, userfunc=None, *args, dataset=None, **kwargs):
        """
        See GroupByOps.apply_nonreduce for examples.
        Categorical needs remove unused bins from its uniques before an apply.
        """
        clean_c = self.filter(None)
        result = super(Categorical, clean_c).apply_nonreduce(userfunc, *args, dataset=dataset, label_keys=clean_c.gb_keychain, **kwargs)
        # result is the same size as original, attach categorical (the key column) to result
        if result.shape[0] == len(clean_c):
            name = self.get_header_names([self], default='gb_key_')[0]
            result[name] = self
            result.label_set_names(name)
        return result

    # ------------------------------------------------------------
    @property
    def gb_keychain(self):
        if self._gb_keychain is None:
            # categorical grouping dict might not contain unique values
            # see if the grouping object has an ifirstkey, otherwise None
            # TODO: move the gb_keychain to the grouping object since it all of the properties are
            # coming from there
            prebinned = True
            gbkeys = self.grouping.uniquedict
            ifirstkey = self.grouping.iFirstKey

            # keychain will perform the sort if necessary
            if self._sorted is None:
                # mapped categoricals have no natural order, so will always be unsorted going into gbkeys
                # groupby results can be sorted or unsorted based on the sort_gb keyword
                sorted = False
            else:
                sorted = self._sorted
            self._gb_keychain = GroupByKeys(gbkeys, ifirstkey=ifirstkey, sort_display=self._sort_gb, pre_sorted=sorted, prebinned=prebinned)
        return self._gb_keychain

    # ------------------------------------------------------------
    def count(self, filter: Optional[np.ndarray] = None, transform: bool = False) -> 'Dataset':
        """
        Returns the counts for each unique category. Unlike other groupby operations, does not take a parameter for data.

        By default, invalid categories will be hidden from the result. The showfilter keyword can be set to display them.

        Parameters
        ----------
        filter : np.ndarray of bool, optional
        transform : bool, defaults to False

        Returns
        -------
        Dataset

        Examples
        --------
        >>> c = rt.Categorical(['a','a','b','c','a','c'])
        >>> c.count()
        *gb_key_0   Count
        ---------   -----
        a               3
        b               1
        c               2

        >>> c = rt.Categorical(['a','a','b','c','d','d'], invalid='d')
        >>> c
        Categorical([a, a, b, c, d, d]) Length: 6
          FastArray([1, 1, 2, 3, 0, 0], dtype=int8) Base Index: 1
          FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

        >>> c.count()
        *gb_key_0   Count
        ---------   -----
        a               2
        b               1
        c               1

        >>> c.count(showfilter=True)
        *gb_key_0   Count
        ---------   -----
        Filtered        2
        a               2
        b               1
        c               1
        """
        # grouping and groupbykeys objects will always be built for count
        # TJD bug here
        # if th gb keys are multikey, and sort_gb is true then not sure keychain.isortrows is correct
        return self.grouping.count(keychain=self.gb_keychain, filter=filter, transform=transform)

    # ------------------------------------------------------------
    @property
    def groupby_data(self):
        """
        All GroupByOps objects can hold a default dataset to perform operations on.
        GroupBy always holds a dataset. Categorical and Accum2 do not.

        Examples
        --------
        By default, requires data to be passed:

        >>> c = rt.Categorical(['a','b','c'])
        >>> c.sum()
        ValueError: Useable data has not been specified in (). Pass in array data to operate on.

        After the result of a Dataset.cat() operation, groupby data is set.

        >>> ds = rt.Dataset({'groups':np.random.choice(['a','b','c'],10), 'data': rt.arange(10), 'data2': rt.arange(10)})
        >>> ds
        #   groups   data   data2
        -   ------   ----   -----
        0   a           0       0
        1   a           1       1
        2   c           2       2
        3   c           3       3
        4   a           4       4
        5   a           5       5
        6   c           6       6
        7   b           7       7
        8   c           8       8
        9   a           9       9
        >>> c = ds.cat('groups')
        >>> c.sum()
        *groups   data   data2
        -------   ----   -----
        a           19      19
        b            7       7
        c           19      19

        """
        return self._dataset

    # ------------------------------------------------------------
    def groupby_data_set(self, ds):
        """
        Store data to apply future groupby operations to. This will make the categorical behave like a groupby object
        that was created from a dataset. If data is specified during an operation, it will be used instead of the stored
        dataset.

        Parameters
        ----------
        ds : Dataset

        Examples
        --------
        >>> c = rt.Categorical(['a','b','c','c','a','a'])
        >>> a = np.arange(6)
        >>> ds = rt.Dataset({'col':a})
        >>> c.groupby_data_set(ds)
        >>> c.sum()
        *gb_key   col
        -------   ---
        a           9
        b           1
        c           5
        """
        self._dataset = ds

    # ------------------------------------------------------------
    def groupby_data_clear(self):
        """
        Remove any stored dataset for future groupby operations.
        """
        self._dataset = None

    # ------------------------------------------------------------
    @property
    def as_string_array(self):
        """
        Returns
        -------
        Array of string value of each index (applies index mask to categories)
        NOTE: this routine is costly as it re-expands the strings
        """
        if self.isenum:
            return self.as_singlekey().expand_array

        elif self.issinglekey:
            string_list = self.category_array
            return self._expand_array(string_list)

        elif self.ismultikey:
            return self.as_singlekey().expand_array
        else:
            raise ValueError(f"Could not re-expand string array with Categorical in {CategoryMode(self.category_mode).name}.")

    # ------------------------------------------------------------
    def as_singlekey(self, ordered=False, sep='_'):
        '''
        Normalizes categoricals by returning a base 1 single key categorical.

        Enum or dict based categoricals will be converted to single key categoricals.
        Multikey categoricals will be converted to single key categoricals.
        If the categorical is already single key, base 0 it will be returned as base 1.
        If the categorical is already single key, base 1 it will be returned as is.

        Parameters
        ----------
        ordered: bool, defaults False
                 whether or not to sort the result
        sep: char, defaults ='_'
                 only valid for multikey since this is the multikey separator

        Examples
        --------
        >>> c=rt.Cat([5, -3, 7], {-3:'one', 2:'two', 5: 'three', 7:'four'})
        >>> d=c.as_singlekey()
        >>> c._fa
        FastArray([ 5, -3,  7])

        >>> d._fa
        FastArray([3, 2, 1], dtype=int8)

        Returns
        -------
        A single key base 1 categorical.
        '''
        if self.isenum:
            c = self.categories()
            # assume and int:str based dictionary
            strings = FastArray(list(c.values()))
            numbers = FastArray(list(c.keys()))
            mask, ikey = ismember(self._fa, numbers)
            # flip mask using inplace
            np.logical_not(mask, out=mask)
            # if the strings are sorted, they may still not be in dictionary order
            if ordered is True:
                c= Categorical(strings[ikey], ordered=ordered)
                # mark invalids
                c._fa[mask]=0
                return c
            else:
                ikey+=1
                # mark all invalids as 0
                ikey[mask]= 0
                return Categorical(ikey, strings, ordered=ordered)

        elif self.ismultikey:
            # use onedict
            name, arr = self.grouping.onedict(invalid=False, sep=sep)
            return Categorical(self._fa, arr, ordered=ordered)
        else:
            if self.base_index == 0:
                return Categorical(self._fa +1, self.categories())
            return self

    # -----------------------------------------------------
    @property
    def str(self):
        # we already are a str
        if self.isenum:
            raise ValueError(f"Could not use str in enum mode.  Email for help")
        elif self.issinglekey:
            string_list = self.category_array

            # indicate we are a categorical that requires expansion
            ikey = self.ikey
            if self.base_index > 0:
                ikey = ikey - self.base_index
            return FAString(string_list, ikey=ikey)

        elif self.ismultikey:
            raise ValueError(f"Could not use .str in multikey mode.")
        else:
            raise ValueError(f"Could not use .str in {CategoryMode(self.category_mode).name}.")

    # ------------------------------------------------------------
    def expand_any(self, categories):
        '''
        Parameters
        ----------
        categories: list or np.ndarray same size as categories array

        Returns
        -------
        A re-expanded array of mapping categories passed in.

        Examples
        --------
        >>> c = rt.Categorical(['a','a','b','c','a'])
        >>> c.expand_any(['d','e','f'])
        FastArray(['d', 'd', 'e', 'f', 'd'], dtype='<U8')
        '''
        categories=np.asanyarray(categories)

        # only adjust index once - not for each column
        if self.base_index == 0:
            index_arr = self._fa + 1
        else:
            index_arr = self._fa
        return self._expand_array( categories, index_arr )

    # ------------------------------------------------------------
    @property
    def expand_array(self) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Returns
        -------
        FastArray or tuple of FastArray
            A re-expanded array or return instance array of mapping codes.
            Filtered items will use the filtered string for stringlike columns, or numeric sentinel value for numeric columns.

        Notes
        -----
        Will warn the user if a large categorical ( > 100,000 items ) is being re-expanded.

        If strings are held, the result column's itemsize will be large of the categories or the invalid category.

        Examples
        --------
        Singlekey categorical:

        >>> c = rt.Categorical(['a','a','b','c','a'])
        >>> c.expand_array
        FastArray([b'a', b'a', b'b', b'c', b'a'], dtype='|S3')

        Multikey:

        >>> c = rt.Categorical([rt.FastArray(['a','b','c','a']), rt.FastArray([1,2,3,1])])
        >>> c.expand_array
        (FastArray([b'a', b'b', b'c', b'a'], dtype='|S8'), FastArray([1, 2, 3, 1]))

        Enum:

        >>> c = rt.Categorical([2, 2, 2, 1, 3], {'a':1,'b':2,'c':3})
        >>> c.expand_array
        FastArray([2, 2, 2, 1, 3])
        """
        if len(self) > 100_000:
            warnings.warn(f"Performance warning: re-expanding categorical of {len(self)} items.")

        # enums return integer instance array
        if self.isenum:
            # consider doing what as_string_array does here
            return self._fa

        else:
            # only adjust index once - not for each column
            if self.base_index == 0:
                index_arr = self._fa + 1
            else:
                index_arr = self._fa
            expanded = [ self._expand_array( unique_arr, index_arr ) for unique_arr in self.grouping.uniquelist ]

            # TODO: Should we iterate through the list of expanded arrays and for each one that's a
            #       FastArray (likely all of them) set the array name to the name of the corresponding key?

            if len(expanded) == 1:
                return expanded[0]
            return tuple(expanded)

    # ------------------------------------------------------------
    @property
    def expand_dict(self) -> Dict['str', FastArray]:
        """
        Returns
        -------
        dict
            A dictionary of expanded single or multikey columns.

        Notes
        -----
        Will warn the user if a large categorical ( > 100,000 items ) is being re-expanded.

        Examples
        --------
        >>> c = rt.Categorical([FA(['a','a','b','c','a']), rt.arange(5)])
        >>> c.expand_dict
        {'key_0': FastArray([b'a', b'a', b'b', b'c', b'a'], dtype='|S3'),
         'key_1': FastArray([0, 1, 2, 3, 4])}
        """
        if len(self) > 100_000:
            warnings.warn(f"Performance warning: re-expanding categorical of {len(self)} items.")

        if self.isenum:
            xdict = { 'codes' : self._fa }
        else:
            xdict = {}
            for i, col in self.category_dict.items():
                xdict[i] = self._expand_array(col, index=self._fa)
        return xdict

    @property
    def filtered_string(self):
        return self._categories_wrap._filtered_name

    # ------------------------------------------------------------
    def _prepend_invalid(self, arr):
        """
        For base index 1 categoricals, add the invalid category to the beginning of the array of unique categories.

        Parameters
        ----------
        arr : FastArray
            The array holding the unique category values for this Categorical.
            This array may be a `FastArray` or a subclass of `FastArray`.

        Returns
        -------
        FastArray
            An array of the same type as `arr` whose length is ``len(arr) + 1``,
            where the first (0th) element of the array is the invalid value for
            that array type.
        """
        # ------------------------------------------------------------
        def _match_invalid(arr):
            """
            Select the appropriate invalid category for the array.
            If possible, use the string for the invalid category - otherwise use the default for the array dtype.
            """
            # ***changed behavior of invalid category
            # will always appear in non-uniques, no need to prepend it

            if arr.dtype.char in NumpyCharTypes.AllFloat+NumpyCharTypes.AllInteger:
                # for numeric types, can't append filtered string, otherwise the whole array will flip!
                inv = INVALID_DICT[arr.dtype.num]
            elif arr.dtype.char == 'U':
                inv = self.filtered_string
            elif arr.dtype.char == 'S':
                inv = self.filtered_string.encode()
            else:
                raise TypeError(f"Don't know how to write invalid category for {arr.dtype}")
            return inv

        inv = _match_invalid(arr)

        # It's important we create the array holding the single invalid value
        # we're going to prepend to 'arr' so it's the same type as 'arr'.
        # This is because the 'hstack' function will return an array of the same type as
        # the first/leftmost argument it's given, and we want this function to return
        # an array of the same type as 'arr' so the type information is preserved.
        invarr = np.array([inv])
        invarr = TypeRegister.newclassfrominstance(invarr, arr)

        # TODO: Revisit after upgrading to numpy 1.17+ -- this might be a better approach compared to
        #       going through TypeRegister.
        #invarr = np.empty_like(arr, shape=1)
        #invarr[0] = inv

        arr = hstack((invarr, arr))
        return arr

    # ------------------------------------------------------------
    def _expand_array(self, arr, index: Optional[np.ndarray] = None):
        """
        Internal routine to h-stack an invalid with an array for re-expanding single or multikey categoricals.
        This allows invalids to be retained in the re-expanded array(s)
        """

        basearray = self._prepend_invalid(arr)

        if index is None:
            index = self._fa
            if self.base_index == 0:
                index = self._fa + 1

        result = basearray[index]
        if Categorical.DebugMode:
            if not isinstance(result, FastArray):
                raise ValueError("Something wrong with expand array", type(result))
        return TypeRegister.newclassfrominstance(result, arr)

    # ------------------------------------------------------------
    def _build_string(self):
        _maxlen = 10
        _slicesize = int(np.floor(_maxlen / 2))
        index_array = self._fa
        _asize = len(index_array)

        cat_wrap = self._categories_wrap

        # print with break
        if _asize > _maxlen:
            left_idx = index_array[:_slicesize]
            right_idx = index_array[-_slicesize:]

            left_strings = [bytes_to_str(cat_wrap[i]).replace("'","") for i in left_idx]
            break_string = ["..."]
            right_strings = [bytes_to_str(cat_wrap[i]).replace("'","") for i in right_idx]
            all_strings = left_strings + break_string + right_strings

        # print full
        else:
            all_strings = [bytes_to_str(cat_wrap[i]).replace("'","") for i in index_array]

        result = ", ".join(all_strings)
        return result

    # ------------------------------------------------------------
    def __str__(self):
        return self._build_string()

    # ------------------------------------------------------------
    def _tf_spacer(self, tf_string):
        for idx, item in enumerate(tf_string):
            if item is True:
                tf_string[idx] = "True "
            elif item is False:
                tf_string[idx] = "False"
        return "".join(tf_string)

    @property
    def unique_repr(self):
        # get the string only for the Categories' uniques
        return self._categories_wrap.__repr__()

    # ------------------------------------------------------------
    def __repr__(self, verbose=False):
        repr_strings = []

        printopts = np.get_printoptions()
        thresh = printopts['threshold']
        edge = printopts['edgeitems']
        line = printopts['linewidth']

        np.set_printoptions(threshold=10)
        np.set_printoptions(edgeitems=5)
        np.set_printoptions(linewidth=1000)
        repr_strings.append(f"{self.__class__.__name__}([{self._build_string()}]) Length: {len(self)}")
        repr_strings.append(f"  {self.view(FastArray).__repr__()} Base Index: {self.base_index}")
        repr_strings.append(f"  {self.unique_repr} Unique count: {self.unique_count}")

        if verbose:
            repr_strings.append(f"  Mode: {CategoryMode(self.category_mode).name}\tLocked: {self._locked}")

        # restore options after building categorical's array display
        np.set_printoptions(threshold=thresh)
        np.set_printoptions(edgeitems=edge)
        np.set_printoptions(linewidth=line)


        return "\n".join(repr_strings)

    # ------------------------------------------------------------
    def info(self) -> None:
        """
        The three arrays in info:
        Categories mapped to their indices, often making the categorical appear to be a string array. Length of array.
        Underlying array of integer indices, dtype. Base index (normally 1 to reserve 0 as an invalid bin for groupby - much better for performance)
        Categories - list or dictionary

        The CategoryMode is also displayed:

        Mode:

        Default - no example
        StringArray - categories are held in a single string array
        IntEnum - categories are held in a dictionary generated from an IntEnum
        Dictionary - categories are held in a dictionary generated from a code-mapping dictionary
        NumericArray - categories are held in a single numeric array
        MultiKey - categories are held in a dictionary (when constructed with multikey, or numeric categories the groupby hash does the binning)

        Locked:

        If True, categories may be changed.
        """

        print(self.__repr__(verbose=True))

    # ------------------------------------------------------------
    def __del__(self):
        """
        Called when a Categorical is deleted.
        """
        # python has trouble deleting objects with circular references
        del self._categories_wrap
        self._grouping = None

    # ------------------------------------------------------------
    @classmethod
    def hstack(cls, cats: Collection['Categorical']) -> 'Categorical':
        """
        Cats must be a list of categoricals.
        The unique categories will be merged into a new unique list.
        The indices will be fixed to point to the new category array.
        The indices are hstacks and a new categorical is returned.

        Examples
        --------
        >>> c1 = rt.Categorical(['a','b','c'])
        >>> c2 = rt.Categorical(['d','e','f'])
        >>> combined = rt.Categorical.hstack([c1,c2])
        >>> combined
        Categorical([a, b, c, d, e, f]) Length: 6
          FastArray([1, 2, 3, 4, 5, 6]) Base Index: 1
          FastArray([b'a', b'b', b'c', b'd', b'e', b'f'], dtype='|S1') Unique count: 6
        """
        return hstack_any(cats, cls, Categorical)

    # ------------------------------------------------------------
    @classmethod
    def categories_equal(
        cls,
        cats: List[Union['Categorical', np.ndarray, Tuple[np.ndarray, ...]]]
     ) -> Tuple[bool, List['Categorical']]:
        """
        Check if all `Categorical`s or arrays have the same categories (same unique values in the same order).

        Parameters
        ----------
        cats : list of Categorical or np.ndarray or tuple of np.ndarray
            `cats` must be a list of `Categorical`s or arrays that can be converted to a `Categorical`.

        Returns
        -------
        match : bool
            True if all the `Categorical`s have the same categories (same unique values in same order),
            otherwise False.
        fixed_cats : list of Categorical
            List of `Categorical`s which may have been fixed up.

        Notes
        -----
        TODO: Can the type annotation for `cats` be relaxed to Collection instead of List?
        """

        crc_list= []
        newcats = []
        mkcheck=set()
        lencheck=set()
        for c in cats:
            # try to make into a categorical if not already
            if not isinstance(c, cls):
                c= cls(c)
            d=c.category_dict

            # check dict len for multikey
            mkcheck.add(len(d))

            # see if unique counts are the same
            lencheck.add(c.unique_count)

            crc_list.append([*d.values()])
            newcats.append(c)

        cats = newcats
        if len(mkcheck)==1 and len(lencheck)==1:
            # TODO: The CRC-based check we're doing here won't consider two arrays which otherwise have the same
            #       categories in the same order but different dtypes (e.g. int8 vs. int16) to be the same.
            #       Do we want to consider those arrays/Categoricals to be equal?
            crc_check=set()
            # might not need to hstack anything
            for arr_list in crc_list:
                # Could be multikey, so compute the CRC/hash per key array.
                # The logic here is similar to that of the crc_match() function in rt_utils.py;
                # that handles some edge cases this does not, while this implementation handles lists/tuples.
                crc_check.add(tuple([crc64(arr) for arr in arr_list]))

            if len(crc_check) == 1:
                return True, cats

        return False, cats

    # ------------------------------------------------------------
    @classmethod
    def align(cls, cats:List['Categorical']) -> List['Categorical']:
        """
        Cats must be a list of categoricals.
        The unique categories will be merged into a new unique list.
        The indices will be fixed to point to the new category array.

        Returns
        -------
        A list of (possibly) new categoricals which share the same categories (and thus bin numbering).

        Examples
        --------
        >>> c1 = rt.Categorical(['a','b','c'])
        >>> c2 = rt.Categorical(['d','e','f'])
        >>> c3 = rt.Categorical(['c','f','z'])
        >>> rt.Categorical.align([c1,c2,c3])
        [Categorical([a, b, c]) Length: 3
          FastArray([1, 2, 3], dtype=int8) Base Index: 1
          FastArray([b'a', b'b', b'c', b'd', b'e', b'f', b'z'], dtype='|S1') Unique count: 7
        Categorical([d, e, f]) Length: 3
          FastArray([4, 5, 6], dtype=int8) Base Index: 1
          FastArray([b'a', b'b', b'c', b'd', b'e', b'f', b'z'], dtype='|S1') Unique count: 7
        Categorical([c, f, z]) Length: 3
          FastArray([3, 6, 7], dtype=int8) Base Index: 1
          FastArray([b'a', b'b', b'c', b'd', b'e', b'f', b'z'], dtype='|S1') Unique count: 7]
        """
        is_same, cats = cls.categories_equal(cats)
        if is_same:
            # fasttrack
            return cats

        combined = cls.hstack(cats)
        res : List['Categorical'] = []
        start_idx = 0
        for cat in cats:
            end_idx = start_idx + len(cat)
            res += [combined[start_idx:end_idx]]
            start_idx = end_idx
        return res

# ------------------------------------------------------------
def categorical_merge_dict(list_categories, return_is_safe:bool=False, return_type:type=Categorical):
    '''
    Checks to make sure all unique string values in all dictionaries have the same corresponding integer in every categorical they appear in.
    Checks to make sure all unique integer values in all dictionaries have the same corresponding string in every categorical they appear in.
    '''
    # ensure all items are categorical in dict mode
    for c in list_categories:
        if not isinstance(c, Categorical):
            raise TypeError(f"Categorical merge dict is for categoricals, not {type(c)}")
        else:
            if not c.isenum:
                raise TypeError(f"Categorical merge dict is for categoricals in dict mode, not {c.category_mode.name}. Try categorical_merge instead.")

    # TODO: speed this up: python is making set objects, iterating over items one-by-one
    # one way: do a multikey unique on the keys+values of the dicts (do we have this, or use the groupby hash?)
    # if the length of unique of the result columns is the same length as the result columns, there is a 1-to-1 key -> value relationship across all dicts
    # zip the result columns to make the final dict
    all_strings = {s for category in list_categories for s in category._categories_wrap.str2intdict}
    all_ints = {i for category in list_categories for i in category._categories_wrap.int2strdict}

    for s in all_strings:
        int_codes = {category._categories_wrap.str2intdict[s] for category in list_categories if s in category._categories_wrap.str2intdict }
        if len(int_codes) > 1:
            raise ValueError(f"Couldn't merge dictionaries because of conflicting codes for {s}: {int_codes}")

    for i in all_ints:
        str_values = {category._categories_wrap.int2strdict[i] for category in list_categories if i in category._categories_wrap.int2strdict }
        if len(str_values) > 1:
            raise ValueError(f"Couldn't merge dictionaries because of conflicting values for {i}: {str_values}")

    # early return for if all we need is to validate the dictionaries
    if return_is_safe:
        return True

    else:
        combined_dict = {}
        for s in all_strings:
            for category in list_categories:
                i = category._categories_wrap.str2intdict.get(s,None)
                if i is not None:
                    combined_dict[s] = i
                    break
        if return_type == dict:
            return combined_dict

        # pass in final combined str -> int mapping dictionary to a new grouping object for each categorical
        groupings = [ Grouping(c._fa, combined_dict, _trusted=True) for c in list_categories ]
        return [ Categorical(grp) for grp in groupings ]

# ------------------------------------------------------------
# Ensure API signature matches Categorical
def CatZero(values, categories=None,                             # main data
                ordered=None, sort_gb=None,  lex=None,       # sorting/hashing
                base_index=0, **kwargs):
    '''
    Calls Categorical() with base_index keyword set to 0.
    '''

    if base_index != 0:
        raise ValueError(f"CatZero base index must be 0! Use Categorical() instead.")

    return Categorical(values, categories=categories, ordered=ordered, sort_gb=sort_gb, lex=lex, base_index=base_index, **kwargs)



# keep this as the last line
TypeRegister.Categorical = Categorical
TypeRegister.Categories = Categories
