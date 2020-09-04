__all__ = [
    # classes/types
    'Grouping',
    # functions
    'combine2groups',
    'hstack_groupings',
    'hstack_test',
    'merge_cats'
]

# TODO: Enable this and use it in code below to replace print() calls;
#       can remove 'verbose' parameters too since logging can be turned on externally.
#import logging

import warnings
from enum import IntEnum, EnumMeta
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import numba as nb
import riptide_cpp as rc

from .rt_numpy import (
    arange,
    combine_accum1_filter, combine_accum2_filter, combine_filter, combine2keys,
    empty, empty_like,
    full,
    _groupbycalculateall, _groupbycalculateallpack, groupby, groupbyhash, groupbylex, groupbypack,
    hstack,
    int8, ismember, issorted,
    lexsort,
    makeifirst, makeilast, makeinext, makeiprev, max, min,
    sort, sortinplaceindirect,
    unique, where, zeros
)
from .rt_enum import (
    ApplyType,
    INVALID_DICT,
    FILTERED_LONG_NAME,
    GB_FUNCTIONS, GB_STRING_ALLOWED, GB_DATE_ALLOWED, GB_FUNC_COUNT, GB_FUNC_USER, GROUPBY_KEY_PREFIX, GB_PACKUNPACK, GB_FUNC_NUMBA,
    int_dtype_from_len,
    NumpyCharTypes
)
#from .rt_groupbykeys import GroupByKeys
from .rt_timers import tic, toc
from .rt_utils import crc_match

if TYPE_CHECKING:
    from .rt_dataset import Dataset
    from .rt_fastarray import FastArray


def combine2groups(
    group_row: 'Grouping',
    group_col: 'Grouping',
    filter: Optional[np.ndarray] = None,
    showfilter: bool = False
) -> 'Grouping':
    '''
    The group_row unique keys are used in the grouping_dict returned.
    The group_cols unique keys are expected to become columns.

    Parameters
    ----------
    group_row : Grouping
        Grouping object for the rows
    group_col : Grouping
        Grouping object for the cols
    filter : np.ndarray of bool, optional
        A boolean filter of values to remove on the rows.
        Should be same length as group_row.ikey array (can pass in ``None``).
    showfilter : bool

    Returns
    -------
    Grouping
        A new Grouping object
        The new ikey will always the number of ``(group_row.unique_count+1)*(group_col.unique_count+1)``.
        The grouping_dict in the Grouping object will be for the rows only.
    '''
    # call CPP algo to merge two bins into one
    result = combine2keys(
        group_row.ikey, group_col.ikey, group_row.unique_count, group_col.unique_count, filter=filter)

    grouping = Grouping(None)

    # copy both dicts
    grouping._grouping_dict = group_row._grouping_dict
    grouping._grouping_unique_dict = group_row._grouping_unique_dict

    grouping._group_row = group_row
    grouping._group_col = group_col
    grouping._showfilter = showfilter

    grouping._iKey = result['iKey']
    grouping.nCountGroup = result['nCountGroup']

    # came from group_row ??? Thomas look at this
    grouping._categorical = group_row._categorical

    # include the filtered bins
    # ??
    unique_count = (group_row.unique_count+1)*(group_col.unique_count+1)
    grouping._unique_count = unique_count

    return grouping

def merge_cats(
    indices,
    listcats,
    idx_cutoffs=None,
    unique_cutoffs=None,
    from_mapping=False,
    stack=True,
    base_index =1,
    ordered=False,
    verbose=False
):
    '''
    For hstacking Categoricals possibly from a stacked .sds load.

    Supports Categoricals from single array or dictionary mapping.

    Parameters
    ----------
    indices :  single stacked array or list of indices
                if single array, needs idx_cutoffs for slicing
    listcats : list of stacked unique category arrays (needs unique_cutoffs)
                or list of lists of uniques
                if the uniques in file1 are 'A,'C'  and the uniques in file2 are 'B','C,'D'
                then listcats is [FastArray('A','C','B','C','D')]
    idx_cutoffs : ndarray of int64, optional
        int64 array of the cutoffs to the `indices`.
        if the index length is 30 and 20 the idx_cutoffs is [30,50]
    unique_cutoffs: list of one int64 array of the cutoffs to the listcats
               if the index length is 2 and 3 the idx_cutoffs is [2,5]
    from_mapping : bool
    stack : bool
    base_index : int
    ordered : bool
    verbose : bool

    Returns
    -------
    Returns two items:
    - list of fixed indices, or array of fixed contiguous indices.
    - stacked unique values

    Notes
    -----
    TODO: Needs to support multikey cats.
    '''
    # ------------------------------------------------------------
    if verbose:
        print("**indices", indices)
        print("**listcats", listcats)
        print("**idx_cutoffs", idx_cutoffs)
        print("**ucutoffs", unique_cutoffs)
        print("**from_mapping", from_mapping)

    if unique_cutoffs is not None:
        if verbose:
            print('unique cutoffs was not none')

        oldcats = []
        # all combined uniques will have the same cutoff points
        unique_cutoffs = unique_cutoffs[0]
        match = True
        for cat_array in listcats:
            oc = []
            start = 0
            for end in unique_cutoffs:
                oc.append(cat_array[start:end])
                start = end
            # build list of slices
            oldcats.append(oc)

    # single list of unique categories (not from slices)
    else:
        oldcats = listcats

    if verbose:
        print('**oldcats',oldcats)
        tic()

    # check to see if all categories were the same
    match = True
    catlen = len(oldcats[0])
    if catlen > 1:
        for oc in oldcats:
            # check the length first
            if catlen == len(oc):
                if not crc_match(oc):
                    match = False
                    break
            else:
                match = False
                break

    if match:
        # first uniques are the same for all
        newcats = [oc[0] for oc in oldcats]

        if from_mapping:
            newcats[1] = newcats[1].astype('U', copy=False)
        else:
            if len(newcats) == 0:
                newcats = newcats[0]

        # now indices will always be stacked
        # maybe set a flag for comparisons - the stack isn't necessary if the categories match
        if unique_cutoffs is None:
            indices = hstack(indices)
        if verbose:
            print("**from_mapping match exactly", indices, newcats[0])

    elif from_mapping:
        # listcats is two arrays:
        # the first is the combined uniques of codes
        # the second is the combined uniques of names
        codes, uidx = unique(listcats[0], return_index=True, sorted=False)
        names = listcats[1][uidx].astype('U', copy=False)
        newcats = [codes, names]
        if verbose:
            print("**from_mapping does NOT match exactly", names)

        # use first occurance of codes to get uniques for both codes and names
        #return indices, newcats

    # need to perform own hstack
    # this will get hit by Categorical.hstack() for single/multikey
    # nothing has been stacked
    else:
        # unique_cutoffs can be None
        indices, newcats = hstack_groupings(indices, listcats, i_cutoffs=idx_cutoffs, u_cutoffs=unique_cutoffs, base_index=base_index, ordered=ordered, verbose=verbose)
    if verbose:
        toc()
    return indices, newcats

def hstack_groupings(
    ikey,
    uniques,
    i_cutoffs=None,
    u_cutoffs=None,
    from_mapping: bool = False,
    base_index: int = 1,
    ordered: bool = False,
    verbose: bool = False
) -> Tuple[Union[list, np.ndarray], List[np.ndarray]]:
    '''
    For hstacking Categoricals or fixing indices in a categorical from a stacked .sds load
    Supports Categoricals from single array or dictionary mapping

    Parameters
    ----------
    indices : single stacked array or list of indices
        if single array, needs idx_cutoffs for slicing
    uniques : list of stacked unique category arrays (needs ``unique_cutoffs``)
        or list of lists of uniques
    i_cutoffs
    u_cutoffs
    from_mapping : bool
    base_index : int
    ordered : bool
    verbose : bool

    Returns
    -------
    list or array_like
        list of fixed indices, or array of fixed contiguous indices.
    list of ndarray
        stacked unique values
    '''
    def lengths_from_cutoffs(cutoffs):
        lengths = cutoffs.copy()
        lengths[1:] -=  cutoffs[:-1]
        return lengths

    if i_cutoffs is None:
        # stack as many as we need to
        if len(ikey) ==1:
            # nothing to do
            return ikey[0], uniques[0]
        else:
            # Turn separate arrays into an array with cutoffs
            i_lengths = TypeRegister.FastArray([len(i) for i in ikey], dtype=np.int64)
            i_cutoffs = i_lengths.cumsum()
            ikey = hstack(ikey)

            u_lengths = TypeRegister.FastArray([len(u) for u in uniques[0]], dtype=np.int64)
            u_cutoffs = u_lengths.cumsum()
            uniques = [hstack(u) for u in uniques]

    else:
        if len(i_cutoffs) == 1:
            # nothing to do
            return ikey[0], uniques[0]

        i_lengths = lengths_from_cutoffs(i_cutoffs)
        u_lengths = lengths_from_cutoffs(u_cutoffs)

    if verbose:
        print("**ikey",ikey, ikey.dtype)
        print("**i_lengths",i_lengths)
        print("**i_cutoffs",i_cutoffs)
        print("**uniques", uniques)
        print("**u_lengths",u_lengths)
        print("**u_cutoffs",u_cutoffs)

    if ordered:
        # TODO: a grouping object can keep the igroup
        g= groupbylex(uniques)
    else:
        # For columns containing higher numbers of unique values, this is where a majority
        # of time will be spent.
        # One example of this would be a Dataset containing all trades reported to OPRA on a given date;
        # if grouping by a string-based column containing the OCC Option Symbology Initiative (OSI)
        # symbol for the option represented by a trade record, there will be on the order of 10^6
        # unique option symbols for a typical day.
        # https://en.wikipedia.org/wiki/Option_symbol
        g= groupbyhash(uniques)

    if base_index == 0:
        uikey=g['iKey'] - 1
    else:
        uikey=g['iKey']

    #--------- START OF C++ ROUTINE -------------
    #based on how many uniques we have, allocate the new ikey
    # do we have a routine for this?
    uikey_length = max(uikey)
    dtype = int_dtype_from_len(uikey_length)
    dtypei = dtype.itemsize

    if base_index==1 and uikey.itemsize == 4 and dtypei <= ikey.itemsize:
        # TODO: handle base_index ==0
        # TODO: handle new array (currently rewrites ikey)
        u_cutoffs=u_cutoffs.astype(np.int64)
        i_cutoffs=i_cutoffs.astype(np.int64)
        newikey = rc.ReIndexGroups(ikey, uikey, u_cutoffs, i_cutoffs)
    else:
        #print(f"bad match {uikey.itemsize} {ikey.itemsize} {uikey_length}")
        newikey = empty((len(ikey),), dtype=dtype)

        start =0
        starti = 0
        for i in range(len(u_cutoffs)):
            stop = u_cutoffs[i]
            stopi = i_cutoffs[i]
            uikey_slice = uikey[start:stop]
            oldikey_slice = ikey[starti:stopi]

            if verbose:
                print("fixing ",starti, stopi)
                print("newikey ",newikey)
                print("oldikey_slice ",oldikey_slice)

            if base_index==1:
                # write a routine for this in C++
                # if 0 and base_index=1, then keep the 0
                filtermask = oldikey_slice == 0
                newikey[starti:stopi] = uikey_slice[oldikey_slice-1]
                if filtermask.sum() > 0:
                    newikey[starti:stopi][filtermask] = 0
            else:
                newikey[starti:stopi] = uikey_slice[oldikey_slice]

            start = stop
            starti = stopi
        #END C++ ROUTINE ---------------------------------

    newuniques = []
    for u in uniques:
        newuniques.append(u[g['iFirstKey']])

    return newikey, newuniques


class Grouping:
    '''
    Every GroupBy and Categorical object holds a grouping in self.grouping;
    this class informs the groupby algorithms how to group the data.

    Stage 1
    -------
    **Initializing from a GroupBy object or unbinned Categorical object:**
        grouping_dict: dictionary of non-unique key columns (hash will be performed)
        iKey: array size is same as multikey, the unique key for which this row in multikey belongs
        iFirstKey: array size is same as unique keys, index into the first row for that unique key
        iNextKey:array  size is same as multikey, index to the next row that hashed to same value
        nCountGroup: array size is same as unique keys, for each unique item, how many values

    **Initializing form a pre-binned Categorical object:**
        grouping_dict: dictionary of pre-binned columns (no hash performed)
        iKey: array size is same as Categorical's underlying index array - often uses the same array.
        unique_count : unique number of items in the categorical.

    Stage 2
    -------
    iGroup :      unique keys are grouped together
    iFirstGroup:  index into first row for the group
    nCountGroup:  number of items in the group

    Properties:
    -----------
    categorical:
    ikey:
    firstkey:
    unique_count:
    isortrows:
    gbkeys:
    iprevkey:
    inextkey:
    igroup:
    ifirstgroup:
    ncountgroup:
    packed : boolean, whether or not packed (Stage 2)
    uniquedict:  lazily evaluated
        if uniquedict is None: (all the time unless returning from regroup)
        cannot use ifirstkey to get uniques

    Performing calculations:
    ------------------------
    (See Grouping._calculate_all)

    :param origdict: a dictionary of all data to perform the operation on
    :param funcNum: a unique code for each math operation
    :func_parm: 0 extra parameter for operations that take more than one argument - will be a tuple if used

    1. Check the keywords for "invalid" (wether or not an invalid bin will be included in the result table)
    2. Check the keywords for a filter and store it.

    3. If the function requires packing, call pack_by_group.
    pack_by_group(filter=None, mustrepack=False)
        1. If the grouping object has already been packed and no filter is present, return.
        2. If a filter is present, discard any existing iNextKey and combine the filter with the iKey.
        3. Call the groupbypack routine -> sends info to CPP.
        4. iGroup, iFirstGroup, nCountGroup are returned and stored.

    4. Prepare the origdict for calculation.
    _get_calculate_dict(origdict, funcNum, func=None, func_param=0)
        1. Check for "col_idx" in keywords (used for agg function mapping certain operations to certain columns)
        2. The grouping object has a _grouping_dict (keys). If these columns are in origdict, they are removed.
        3. Most operations cannot be performed on strings or string-based categoricals. Remove columns of those types.
        4. Return the cleaned up dictionary, and a list of its columns. (npdict, values)

    5. Perform the operation.
        * rc.EmaAll32 - for cumsum, cumprod, ema_decay, etc.
        * _groupbycalculateall - for basic functions that don't require packing (combine filter if exists)
        * _groupbycalculateallpack - for level 2 functions that require packing

    accum_tuple is a series of columns after the operation. The data has not been sorted.
    accum_tuple has an invalid item at [0] for each column. If no invalid was requested, trim it off.
    Store the columns in a list.
    If the function was called from accum2, return here.

    6. Make a dictionary from the list of calculated columns.
    _make_accum_dataset
        1. Make a dictionary from the list of calculated columns. Use the names from npdict (see step 4)
        2. If nothing was calculated for the column, the value will be None. Remove it.
        3. If the column was a categorical, the calculate dict only has its indices. Pull the categories from the original dictionary and build a new categorical (shallow copy)

    7. Make a dataset from the dictionary of calculated columns.
    _return_dataset
        1. If the function is in cumsum, cumprod, ema_decay, etc. no groupby keys will appear (set to None)
        2. If the function is count, it will have a single column (Count) - build a dataset from this.
        3. Initialize an empty diciontary (newdict).
        4. Iterate over the column names in the *original* dictionary and copy them to the newdict. accumdict only contains
            columns that were operated on. If the return_all flag was set to True, these columns still need to be included.
        5. If the function is in cumsum, cumprod, ema_decay, etc. no sort will be applied, no labels (gbkeys) will be tagged
        6. Otherwise, apply a sort (default for GroupBy) to each column with isortrows (from the GroupByKeys object). Tag all label columns in final dataset.

    8. Return the dataset

    '''

    # test/debug flags
    DebugMode = False
    GroupingInit = {
        # from initial hash
       '_iKey':          None,
       'iFirstKey':     None,
       'iLastKey':      None,
       'iNextKey':      None,
       'iPrevKey':      None,
       'Ordered':       None,
       '_unique_count': None,
       '_grouping_dict':None,
       '_grouping_unique_dict':None,
       '_enum'         :None,
       '_categorical':  False,
       '_isdirty':      False,
       '_catinstance':  None,

       #packed info
       '_packed':       False,
       '_packedwithfilter' : False,
       'iGroup':        None,
       'iFirstGroup':   None,
       'nCountGroup':   None,
       'iGroupReverse': None,

        # misc
       '_gbkeys':       None,
       '_sort_display': False,      # Note consider making Nones
       '_base_index':   1,
       '_pcutoffs':     None,
       }

    # Used to register tables from other tooling
    # key : (funcname, requirespacking, frontend, backend, grouper)
    REGISTERED_REVERSE_TABLES=[]

    #---------------------------------------------------------------
    def copy_from(self, other: Optional['Grouping'] = None) -> None:
        '''
        Initializes a new Grouping object if other is None.
        Otherwise shallow copy all necessary attributes from another grouping object to self.

        Parameters
        ----------
        other : `Grouping`

        '''
        for name, defaultvalue in Grouping.GroupingInit.items():
            if other is None:
                self.__setattr__(name, defaultvalue)
            else:
                attr = other.__getattribute__(name)
                if isinstance(attr, dict):
                    # make a new dict - still a shallow copy of arrays
                    attr = attr.copy()
                self.__setattr__(name, attr)

    #---------------------------------------------------------------
    def _from_categories(self, grouping, categories, arr_len:int, base_index: int, filter, dtype, ordered:bool, _trusted:bool):
        """
        Initialize a Grouping object from pre-defined uniques.

        Parameters
        ----------
        grouping : dict of single array
            Pre-defined iKey or non-unique values.
        categories : dict of arrays
            Pre-defined dictionary of unique categories or enum mapping (not implemented)
        arr_len : int
            Length of arrays in ``categories`` dict.
        filter : boolean array
            Pre-filter the same length as the non-unique values.
        _trusted : bool
            If True, data will not be validated with min / max check.

        Returns
        -------
        ikey : ndarray of ints
            Base 0 or Base 1 ikey
        ordered_flag : bool
            Flag indicating whether the categories were/are ordered. This is the `ordered` flag just being passed through.
        """
        if len(grouping) != 1:
            raise ValueError(f'Grouping only supports construction from categories for single-key.')
        grouping = [*grouping.values()][0]
        cats = [*categories.values()]

        # flip unsigned integers
        if grouping.dtype.char in NumpyCharTypes.UnsignedInteger:
            grouping = self.possibly_recast(grouping, arr_len, dtype=dtype)

        # turn non-unique values into an ikey
        if grouping.dtype.char not in NumpyCharTypes.Integer:
            if len(categories) != 1:
                raise ValueError(f'Grouping only supports non-unique values -> pre-defined categories for single key.')
            _, ikey = ismember(grouping, cats[0], base_index=base_index)
            ordered = issorted(cats[0])
            _trusted = True

        else:
            ikey = grouping

        # NOTE: assumes that all provided uniques are unique, otherwise duplicate keys when grouped
        if not _trusted and len(ikey) > 0:
            # min / max check for external bins, out of range will break operations
            minval = ikey.min()
            maxval = ikey.max()
            maxindex = arr_len + base_index
            if minval < 0:
                raise ValueError(f"Invalid index {minval} found. Indices must be between invalid bin 0 and {maxindex}.")
            if maxval >= maxindex:
                raise ValueError(f"Invalid index {maxval} found. Indices must be between invalid bin 0 and {maxindex}.")
            if len(cats) == 1:
                ordered = issorted(cats[0])
        # base-0 indexing not allowed
        if filter is not None:
            ikey = combine_filter(ikey, filter)

        # always return in the base_index requested
        return ikey, ordered

    #---------------------------------------------------------------
    def __init__(
        self,
        grouping,
        categories = None,
        ordered = None,
        dtype = None,
        base_index:int =1,
        sort_display:bool=False,
        filter = None,

        # advanced flags----
        lex:bool = False,
        rec:bool = False,
        categorical:bool = False,
        cutoffs = None,
        next:bool = False,
        unicode:bool = False,
        name:Optional[str] = None,
        hint_size:int=0,
        hash_mode:int=2,
        _trusted:bool=False,
        verbose:bool=False):
        """
        Create a new Grouping object.

        Parameters
        ----------
        grouping : `dict` or `list` of `np.ndarray`
            or a single np.ndarray
            Data to bin (store if already unique).
        categories : array, optional
            The unique categories for the grouping.
            It may also be a python enum.
            If ``_trusted`` is True, will not be valided. Otherwise, min / max check will occur.
            Defaults to None.
        ordered
            When set to True, the uniques will be sorted (tbd - when gb sets this)
        dtype : str or np.dtype
            The numpy dtype to hold `ikey` in.
        filter : ndarry of bools, optional
            Boolean array which can be used to pre-filter values out of the input data before applying the grouping logic (e.g. hash/sort).
        base_index : int, default 1
            Base index of the ``iKey`` (default 1, to reserve 0 bin for filtered/invalid values).
            ``MultiKeyGroupBy32`` will always produce base-1 ``iKey``.
        sort_display : bool
            Sort dataset by unique key values in final display after operation. Defaults to False.

        Other Parameters
        ----------------
        lex : bool
            When set to False, the binning will be first appearance.
            When set to True, uses lexsort to find groups (otherwise hashes);
            the binning will be in lexigraphical order and ``iGroup``, ``iFirstGroup``, ``nCountGroup``
            will also be generated during construction.
            Defaults to False.
        categorical : bool
            When set to True, the categories will be generated and held.
            If grouping is an enum, categories will be a dictionary pair;
            otherwise, ``iFirstKey`` is used to pull from the self._grouping_dict
        cutoffs : array, optional
            Array of cutoffs for natural groups of a PDataset.
        next : bool
            If True, calculate ``iNextKey``. Defaults to False.
        unicode : bool
            If True, keep strings as unicode if converting data to FastArrays;
            otherwise, possibly flip unicode to bytestring during FastArray construction.
            Defaults to False.
        name : str, optional
            Set the name of the grouping dict.
        hash_mode : int
            Setting for hash technique. Defaults to 2, users should not override this.
        hint_size : int
            If number of uniques is known, provide it as a hint to the hashing logic. Defaults to zero.
        _trusted : bool
            If True, data used to initialize will not be validated. e.g. generation from a slice. Defaults to False.
        """
        #-------------------------------------------------
        def data_as_dict(grouping, name_def:Optional[str]=None) -> dict:
            # flip all input to dictionary, preserve names
            grouping_dict = {}
            if not isinstance(grouping, list):
                grouping = [grouping]
            if len(grouping) != 1:
                # cannot set name for multiple keys
                name_def = None
            for idx, arr in enumerate(grouping):
                try:
                    name = arr.get_name()
                except:
                    name = None

                if name is None:
                    name = name_def
                if name is None:
                    name = GROUPBY_KEY_PREFIX+'_'+str(idx)
                if name in grouping_dict:
                    # must use a unique name
                    name = GROUPBY_KEY_PREFIX+'_c'+str(idx)
                grouping_dict[name] = arr
            return grouping_dict
        #-------------------------------------------------
        def data_as_fastarray(grouping, unicode:bool) -> Tuple[dict, int]:
            # return dict and length of the array(s)
            #--------
            def object_to_string(arr, unicode:bool):
                if isinstance(arr, np.ndarray) and arr.dtype.char == 'O':
                    if unicode:
                        try:
                            arr = arr.astype('U')
                        except:
                            raise TypeError(f"Failed to convert object array to unicode.")
                    else:
                        try:
                            arr = arr.astype('S')
                        except:
                            try:
                                arr = arr.astype('U')
                            except:
                                raise TypeError(f"Failed to convert object array bytestrings and unicode.")
                return arr
            #--------
            newdict={}
            len_set=set()
            for k,v in grouping.items():
                if not isinstance(v, TypeRegister.FastArray):
                    v = object_to_string(v, unicode)
                    v = TypeRegister.FastArray(v, unicode=unicode)
                len_set.add(v.shape[0])
                newdict[k] = v

            if len(len_set) != 1:
                raise ValueError(f"length of all grouping keys must be the same, not {len_set}.")
            return newdict, len_set.pop()
        #-------------------------------------------------
        def is_enumlike(categories) -> bool:
            # detect IntEnum or dictionary mapping
            isenum = False
            if isinstance(categories, EnumMeta):
                isenum = True
            elif isinstance(categories, dict):
                if len(categories) == 0:
                    raise ValueError(f'Categories dict was empty.')

                if not isinstance([*categories.values()][0], np.ndarray):
                    if filter is not None:
                        raise TypeError(f"Grouping from enum does not support pre-filtering.")
                    isenum = True
            return isenum
        #-------------------------------------------------


        # init the Grouping object to defaults
        self.copy_from(None)

        # if the grouping dict is None (short circuit)
        if grouping is None:
            return

        if verbose:
            if filter is not None: print("gb with filter:", filter)

        self._pcutoffs = cutoffs
        self._base_index = base_index
        self._categorical = categorical

        # ---- take care of the ordered kwarg -----
        if ordered is False and lex is True:
            # force ordered to True since the data will be ordered
            ordered=True
        if ordered is None:
            self.Ordered = False
        else:
            if ordered is not False and ordered is not True:
                raise TypeError(" The kwarg 'ordered' must be True, False, or None")
            self.Ordered = ordered

        # self.Ordered is set -- and three possible paths
        # 1) if False, use hashing
        # 2) if True and lex is False: use hashing +sort
        # 3) if True and lex if True: straight lexsort

        # _sort_display ends up being a directive
        if sort_display and not self.Ordered:
            self._sort_display=True
        else:
            self._sort_display=False

        # ---- requested dtype ----
        possibly_recast = False

        # turn grouping into a grouping dict, pull names if possible
        if not isinstance(grouping, dict):
            grouping = data_as_dict(grouping, name_def=name)

        # flip all to fastarray, length check
        grouping, arr_len = data_as_fastarray(grouping, unicode)

        # check for user defined bins
        if categories is not None:
            # all user defined bins take this path
            if is_enumlike(categories):
                #print('categories was enumlike')
                # grouping dict is non-unique integer codes
                list_values = [*grouping.values()]
                dtchar = list_values[0].dtype.char
                if not ((len(grouping)==1) and (dtchar in NumpyCharTypes.AllInteger)):
                    raise TypeError(f"Codes paired with enum must be integers. Got {dtchar} instead.")

                self._enum = GroupingEnum(categories, _trusted=_trusted)
                # this is repeated below in normal groupby path
                # switch to lazy-evaluate, leave here for test
                if filter is not None:
                    self._make_enumikey(list_values, filter=filter)

                self._grouping_dict = grouping
                # muting this warning - maybe set grouping base_index kwarg to default None?
                #if base_index is not None:
                #    warnings.warn(f"Groupings from enum do not have a base index. Cannot set to {base_index}. Will be set to None.")
                self._base_index = None
                return

            # normalize user defined bins
            if not isinstance(categories, dict):
                categories = data_as_dict(categories, name_def=name)
            categories, category_arr_len = data_as_fastarray(categories, unicode)

            tempikey, self.Ordered = self._from_categories(grouping, categories, category_arr_len, base_index, filter, dtype, self.Ordered, _trusted)
            #print('temp ikey was',tempikey)
            self._unique_count = category_arr_len
            self._grouping_unique_dict = categories

            if dtype is not None:
                possibly_recast = True

            if base_index==0:
                # possibly recast the tempikey
                if possibly_recast:
                    tempikey = self.possibly_recast(tempikey, self._unique_count, dtype=dtype)
                self._catinstance = tempikey
                # return here to force lazy evaluation of ikey
                return
            else:
                self._iKey = tempikey
                # fall through since iKey is safe because base_index ==1

        # for regular groupby hash off dataset or multikey categorical
        else:
            list_values=[*grouping.values()]

            # try to 'borrow' bins from pre-binned object
            if (len(list_values) == 1 and TypeRegister.is_binned_array(list_values[0])):
                if verbose:
                    print("categorical supplied", list_values)
                cat = list_values[0]
                if filter is None:
                    grp = cat.grouping

                else:
                    # TJD July 2019, this routine does not work properly
                    # it has been disabled since filter is None on this branch
                    if verbose:
                        print("refiltering categorical", list_values)
                    cat = cat.filter(filter)
                    grp = cat.grouping
                self.copy_from(grp)
                return

            else:
                possibly_recast = True

                if next:
                    # older routine left for testing
                    raise ValueError("Next not working")
                    #self._iKey, self.iNextKey, self.iFirstKey, self.nCountGroup = rc.MultiKeyGroupBy32Super(list_values, hint_size, filter, hash_mode)
                    #self._unique_count = len(self.nCountGroup)
                else:
                    # faster routine but calculates less
                    # NOTE: rec is defaulting to False
                    hashdict = groupby(list_values, filter=filter, cutoffs=cutoffs, base_index=base_index, lex=lex, rec=rec, hint_size=hint_size)

                    tempikey, self.iFirstKey, self._unique_count, self.iGroup, self.iFirstGroup, self.nCountGroup = \
                        hashdict['iKey'], hashdict['iFirstKey'], hashdict['unique_count'], hashdict['iGroup'], hashdict['iFirstGroup'], hashdict['nCountGroup']

                    # if we come from a lexsort, we are already packed
                    if lex:
                        self._packed = True

                # pull and store only the uniques
                if categorical:
                    self._grouping_unique_dict = self._build_unique_dict(grouping)

                    # skip this if after lex
                    if self.Ordered and lex is not True:

                        # since the original lengths are being dropped, iFirstKey is no longer valid
                        self.iFirstKey = None

                        # this came from a hash (first occurrence)
                        # so now we sort the uniques and get back sortidx
                        sortidx = self._make_isortrows(self._grouping_unique_dict)
                        for k,v in self._grouping_unique_dict.items():
                            self._grouping_unique_dict[k]=v[sortidx]

                        # need to fix the temp ikey to match the sort
                        if base_index == 0:
                            _, tempikey = ismember(tempikey-1, sortidx, base_index=1)
                        else:
                            _, tempikey = ismember(tempikey, sortidx+1, base_index=base_index)
                    elif lex is True:
                        # Believe there is nothing to do
                        pass

                    if verbose:
                        print("tempikey from cat", tempikey)

                    # set either catinstance or iKey
                    if base_index == 0:
                        self._catinstance = tempikey-1
                    else:
                        self._iKey = tempikey

                else:
                    # remember the grouping dictionary
                    self._grouping_dict = grouping

                    # what if groupby comes in as base-zero index??
                    self._iKey = tempikey

        # if iKey might be lazy evaluated, how do we possibly recast? - recast the temp ikey?
        # possibly recast the iKey
        if possibly_recast and self._iKey is not None:
            self._iKey = self.possibly_recast(self._iKey, self._unique_count, dtype=dtype)

    # ------------------------------------------------------------
    def _make_enumikey(self, list_values, filter=None):
        '''
        internal routine to lazy generate ikey for enum
        if a filter is passed on init, have to generate upfront

        will generate ikey, ifirstkey, unique_count also
        '''
        hashdict = groupby(list_values, filter=filter)
        self._iKey, self.iFirstKey, self._unique_count, self.iGroup, self.iFirstGroup, self.nCountGroup = \
            hashdict['iKey'], hashdict['iFirstKey'], hashdict['unique_count'], hashdict['iGroup'], hashdict['iFirstGroup'], hashdict['nCountGroup']

    # ------------------------------------------------------------
    @classmethod
    def possibly_recast(cls, arr, unique_count: int, dtype=None):
        """
        `unique_count` is checked and compared against preferred (minimal) dtype size is calculated.

        If a dtype has been provided, it will be used (only if it is large enough to fit the maximum value for the calculated dtype).

        Parameters
        ----------
        arr : ndarray of ints
        unique_count : int
            The number of unique bins corresponding to `arr`.
        dtype : str or np.dtype, optional
            Optionally force a dtype for the returned integer array (see dtype keyword in the Categorical constructor), defaults to None.

        Returns
        -------
        new_arr : ndarray of ints
            A recasted array with a smaller dtype, requested dtype, or possibly the same array as `arr` if no changes were needed.
        """
        newdtype = int_dtype_from_len(unique_count)

        # dtype requested, check if safe to override
        if dtype is not None:
            req_dtype = np.dtype(dtype)
            if not req_dtype.char in NumpyCharTypes.Integer:
                raise TypeError(f"A type of {dtype} is not a signed integer type")

            # check if ok to keep forced dtype
            if req_dtype.itemsize >= newdtype.itemsize:
                newdtype = req_dtype
            else:
                # need warning here
                warnings.warn(f"A type of {dtype} was too small, upcasting.")

        return arr.astype(newdtype, copy=False)

    #---------------------------------------------------------------
    def __repr__(self):
        repr_str = []

        repr_str.append(f'')
        # from initial hash
        repr_str.append(f'_iKey: {self._iKey}')
        repr_str.append(f'iFirstKey: {self.iFirstKey}')
        repr_str.append(f'iLastKey: {self.iLastKey}')
        repr_str.append(f'iNextKey: {self.iNextKey}')
        repr_str.append(f'_unique_count: {self._unique_count}')
        repr_str.append(f'_grouping_dict: {self._grouping_dict}')
        repr_str.append(f'_grouping_unique_dict: {self._grouping_unique_dict}')
        repr_str.append(f'_enum: {self._enum}')
        repr_str.append(f'_categorical: {self._categorical}')
        repr_str.append(f'isenum: {self.isenum}')
        repr_str.append(f'isdirty: {self.isdirty}')
        repr_str.append(f'_catinstance: {self._catinstance}')

        repr_str.append(f'')
        # packed info
        repr_str.append(f'_packed: {self._packed}')
        repr_str.append(f'iGroup: {self.iGroup}')
        repr_str.append(f'iFirstGroup: {self.iFirstGroup}')
        repr_str.append(f'nCountGroup: {self.nCountGroup}')

        repr_str.append(f'')
        # misc
        repr_str.append(f'_gbkeys: {self._gbkeys}')
        repr_str.append(f'_sort_display: {self._sort_display}')
        repr_str.append(f'Ordered: {self.Ordered}')
        repr_str.append(f'_base_index: {self._base_index}')

        return '\n'.join(repr_str)

    ##---------------------------------------------------------------
    def copy(self, deep: bool=True) -> 'Grouping':
        """
        Create a shallow or deep copy of the grouping object.

        Parameters
        ----------
        deep : bool, default True
            If True, makes a deep copy of all array data.

        Returns
        -------
        newgrouping : `Grouping`

        Note: a shallow copy will always make new dictionaries, but does not copy array data.
        """

        newgrouping = Grouping(None)
        newgrouping.copy_from(self)
        # GroupingEnum has its own copy method, makes dict copies
        if self.isenum:
            newgrouping._enum = newgrouping._enum.copy()

        if deep:
            # NOTE: deep has not been tested much
            newgrouping._iKey = newgrouping.ikey.copy()

        return newgrouping

    #---------------------------------------------------------------
    def _set_newinstance(self, newinstance):
        if self.isenum:
            # set dirty will blow away _iKey
            if self._grouping_dict is None:
                raise ValueError("Internal error enum must have a grouping_dict")
            self._grouping_dict = { gname: newinstance for gname, gcol in self._grouping_dict.items() }

            self._catinstance = newinstance

        # if set, these three will be / hold arrays of the same length
        elif self._catinstance is not None:
            # must be base 0 cat
            self._catinstance = newinstance
        else:
            # base 1 cat (ALLOWED)
            # base 1 groupby set newinstance (NOT ALLOWED)
            if self._grouping_dict is not None:
                raise ValueError("Internal error dictionary exists when set_newinstance -- trying to slice or change groupby?")

            if self._iKey is None:
                raise ValueError("Internal error in grouping _set_newinstance")

            self._iKey = newinstance

        # after index / slice, packed information is no longer valid
        # set dirty flag to reset these
        self.set_dirty()

    #---------------------------------------------------------------
    @classmethod
    def newclassfrominstance(cls, instance, origin):
        if isinstance(origin, cls):
            return origin.newgroupfrominstance(instance)
        raise TypeError(f"Origin must be type {cls}. Got {type(origin)}")

    #---------------------------------------------------------------
    def newgroupfrominstance(self, newinstance):
        '''
        calculate_all may change the instance

        Parameters
        ----------
        newinstance: integer based array (codes or bins)

        Returns
        -------
        a new grouping object
        '''
        newgroup = self.copy(deep=False)
        newgroup._set_newinstance(newinstance)
        return newgroup

    #---------------------------------------------------------------
    def __getitem__(self, fld):
        """Perform an indexing / slice operation on iKey, _catinstance, _grouping_dict
        if they have been set.

        Parameters
        ----------
        fld : integer (single item) raise error
              slice
              integer array (fancy index)
              boolean array (true/false mask)
              string, list of strings: raise error

        Returns
        -------
        newgroup : `Grouping` or scalar or tuple
            A copy of the grouping object with a reindexed iKey. The dirty flag in the result
            will be set to True.
            A single scalar value (for enum/singlekey grouping)
            A tuple of scalar values (for multikey grouping)

        """
        ## can also take out these errors - let them fail on the array indexing below
        #if not isinstance(fld, (np.array, list, slice)):
        #    raise TypeError(f'Cannot index grouping with type {(type(fld))}')

        has_slice = False

        if isinstance(fld, np.ndarray):
            # fancy / boolean index
            if fld.dtype.char in NumpyCharTypes.AllInteger+'?':
                pass
            # string (maybe support this in the future)
            elif fld.dtype.char in 'US':
                raise NotImplementedError(f'Grouping does not currently support indexing by strings or arrays of string values.')
            else:
                raise TypeError(f'Unsupported array index. dtype was {fld.dtype}')
        elif isinstance(fld, (slice, list)):
            # string lists will raise error too

            # convert lists to array
            if isinstance(fld, list):
                fld = np.asarray(fld)
            else:
                has_slice = True
        else:
            raise TypeError(f'Cannot index grouping with type {(type(fld))}')

        # enum
        # base 0 cat
        # base 1 cat
        # base 1 groupby (NOT ALLOWED will raise error)

        if self.isenum or self._catinstance is not None:
            # must be base 0 cat or enum
            newinstance = self.catinstance[fld]
        else:
            # base 1 cat
            # base 1 groupby
            if self._iKey is None:
                raise ValueError("Internal error in grouping __getitem__")

            # TJD new code to handle invalids for categoricals
            # check for any integer or float array which we route to MBGet
            if not has_slice and fld.dtype.num >= 1 and fld.dtype.num <=13 and self._base_index > 0:
                # pass in the invalid as 0
                # TODO: Modify to call mbget through the mbget function in rt_utils.py (which wraps the ledger call).
                if fld.ndim == 1:
                    if fld.strides[0] != fld.itemsize:
                        fld = fld.copy()
                newinstance = TypeRegister.MathLedger._MBGET(self._iKey,fld, 0)
            else:
                newinstance= self._iKey[fld]

            # NOTE: if we hit here, the categorical/grouping is possibly dirty because it has been sliced or reduced

        oldname= self.get_name()
        if oldname is not None:
            newinstance.set_name(oldname)

        return self.newgroupfrominstance(newinstance)

    #---------------------------------------------------------------
    def ismember(self, values, reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Used to match against the unique categories
        NOTE: This does not match against the entire array, just the uniques

        Parameters
        ----------
        reverse : bool, defaults to False.
             Set to True to reverse the ``ismember(A, B)`` to ``ismember(B,A)``.

        Returns
        -------
        member_mask : np.ndarray of bool
            boolean array of matches to unique categories
        member_indices : np.ndarray of int
            fancy index array of location in unique categories

        Examples
        --------
        >>> a = rt.Cat(['b','c','d']).tile(5)
        >>> b = rt.Cat(['a','b','d','e','f']).tile(5)
        >>> tf1 = rt.ismember(a, b)[0]
        >>> tf2 = b.grouping.ismember(a.categories())[1][b-1] != -128
        >>> np.all(tf1 == tf2)
        True

        >>> a = rt.Cat(['BABL','COKE','DELT']).tile(50000)
        >>> b = rt.Cat(['AAPL','BABL','DELT','ECHO','FB']).tile(33333333)
        >>> %time tf1 = rt.ismember(a,b)[0]
         197 ms
        >>> %time tf3 = rt.ismember(a.category_array, b.category_array)[1][a-1] != -128
         1 ms
        >>> np.all(tf1 == tf3)
        True

        See Also
        --------
        rt.Grouping.isin
        rt.ismember
        """
        if self.isenum:
            if not isinstance(values, np.ndarray):
                values = TypeRegister.FastArray(values, unicode=True)

            if values.dtype.char in NumpyCharTypes.AllInteger:
                # pull first occurrence of codes
                uniquelist = self.catinstance[self.ifirstkey]

            elif values.dtype.char in 'US':
                # translates codes -> strings
                uniquelist = self.uniquelist[0]
            else:
                raise TypeError(f"Grouping from enum isin() can only take integers and strings. Got {values.dtype}")

            if reverse:
                return ismember( values, uniquelist )
            else:
                return ismember( uniquelist, values )

        else:

            result_len = len(self.catinstance)

            if isinstance(values, tuple) or np.isscalar(values):
                # convert to list
                values = [ values ]

            if isinstance(values, list):
                # nothing passed in list?
                if len(values) == 0:
                    return zeros(result_len, dtype=np.bool), None

                # make separate arrays from each tuple index (rotate)
                # how much error checking should happen here?
                if isinstance(values[0], tuple):
                    # split tuple into list for item at each position
                    # base on first tuple length
                    tup_len = len(values[0])
                    values = [ TypeRegister.FastArray([t[i] for t in values]) for i in range(tup_len) ]
                else:
                    values = [ TypeRegister.FastArray(values) ]
            elif isinstance(values, np.ndarray):
                values = [ values ]

            else:
                # not found
                mask = zeros(result_len, dtype=np.bool)
                et = empty(len(values), dtype=np.int32)
                et.fill_invalid()
                return mask, et

        # values is a list of arrays (single and multikey hit same path)
        if reverse:
            return ismember( values, self.uniquelist )
        else:
            return ismember( self.uniquelist, values )


        ## expand with a zero-base index
        #if self.base_index == 0:
        #    return found[self.catinstance]
        #else:
        #    return found[self.catinstance-1]


    #---------------------------------------------------------------
    def isin(self, values):
        """
        Used to match values

        Returns
        -------
        numpy array of bools where the values are found

        See Also
        --------
        rt.Grouping.isin
        """
        if self.isenum:
            found, _ = self.ismember(values)
            #expand boolean found
            return found[self.ikey-1]
        else:
            found, _ = self.ismember(values)
            # expand with a zero-base index
            if self.base_index == 0:
                return found[self.catinstance]
            else:
                return found[self.catinstance-1]

        #mask = None
        #result_len = len(self.catinstance)

        #if isinstance(values, tuple) or np.isscalar(values):
        #    values = [ values ]

        #if isinstance(values, list):
        #    if len(values) == 0:
        #        return zeros(result_len, dtype=np.bool)
        #    # make separate arrays from each tuple index (rotate)
        #    # how much error checking should happen here?
        #    if isinstance(values[0], tuple):
        #        # split tuple into list for item at each position
        #        # base on first tuple length
        #        tup_len = len(values[0])
        #        values = [ TypeRegister.FastArray([t[i] for t in values]) for i in range(tup_len) ]
        #    else:
        #        values = [ TypeRegister.FastArray(values) ]
        #elif isinstance(values, np.ndarray):
        #    values = [ values ]

        #else:
        #    mask = zeros(result_len, dtype=np.bool)

        #if mask is None:
        #    # values is a list of arrays (single and multikey hit same path)
        #    found, _ = ismember( self.uniquelist, values )
        #    # expand with a zero-base index
        #    if self.base_index == 0:
        #        return found[self.catinstance]
        #    else:
        #        return found[self.catinstance-1]

        #return mask

    #---------------------------------------------------------------
    @property
    def catinstance(self):
        """Integer array for constructing Categorical or Categorical-like array.

        Returns
        -------
        instance_array : `FastArray`
            If base index is 1, returns the ikey.
            If base index is 0, stores and returns ikey - 1.
            If in enum mode, returns integers from _grouping_dict.
        """
        if self._catinstance is None:
            if self.isenum:
                if self._grouping_dict is None:
                    raise ValueError("Internal error with enum and grouping_dict")
                return [*self._grouping_dict.values()][0]

            elif self.base_index == 0:
                self._catinstance = self._iKey - 1

            elif self.base_index == 1:
                return self._iKey

            else:
                raise ValueError(f'Critical error: could not determine cat instance.')
        return self._catinstance

    #--MODE PROPERTIES----------------------------------------------
    #---------------------------------------------------------------
    @property
    def isenum(self) -> bool:
        return self._enum is not None

    @property
    def isdisplaysorted(self) -> bool:
        return self._sort_display

    @property
    def isordered(self) -> bool:
        return self.Ordered

    @property
    def issinglekey(self) -> bool:
        """True if unique dict holds single array.
        False if unique dict hodls multiple arrays or in enum mode.
        """
        if self.isenum:
            return False
        return len(self.uniquedict) == 1

    @property
    def ismultikey(self) -> bool:
        """True if unique dict holds multiple arrays.
        False if unique dict holds single array or in enum mode.
        """
        if self.isenum:
            return False
        return len(self.uniquedict) > 1

    @property
    def iscategorical(self) -> bool:
        """True if only uniques are being held - no reference to original data.
        """
        return self._categorical

    @property
    def isdirty(self) -> bool:
        '''
        isdirty : bool, default False
            If True, it's possible that not all of the values in between 0 and the unique count appear in the iKey.
            Number of unique occurring values may be different than number of possible unique values.
            e.g. after slicing a Categorical.
        '''
        return self._isdirty

    def set_dirty(self):
        """If the shared information (like a Categorical's instance array) has been changed outside
        of the grouping object, the changing routine can call this on the grouping object.
        """

        # related to the old ikey / cat instance
        self.iFirstKey = None
        self.iLastKey = None
        self.iPrevKey = None
        self.iNextKey = None

        # packed info
        self._packed = False
        self.iGroup = None
        self.iGroupReverse = None
        self.iFirstGroup = None
        self.nCountGroup = None

        # misc
        self._isdirty = True
        self._gbkeys = None
        self._pcutoffs = None
        # TODO: Also clear _isortrows since _gbkeys (which it's derived from) was cleared?
        # self._isortrows = None

        # force enum to regen also
        if self.isenum or self._base_index == 0:
            self._iKey = None

    #---------------------------------------------------------------
    @property
    def ikey(self):
        '''
        Returns a 1-based integer array with the bin number for each row.

        Bin 0 is reserved for filtered out rows.
        This property will return +1 for base-0 grouping.

        Returns
        -------
        ikey : np.ndarray of int
        '''
        if self._iKey is None:
            if self.isenum:
                # generate ikey
                list_values = [*self._grouping_dict.values()]
                self._make_enumikey(list_values)
            elif self._base_index == 0:
                # once generated remember (use set_dirty() to clear)
                self._iKey = self._catinstance + 1
            else:
                raise ValueError("Internal error generating ikey")

            # enum path needs to go through hash, lazy-eval here
        return self._iKey

    #---------------------------------------------------------------
    @property
    def base_index(self) -> int:
        """The starting index from which keys (valid groups) are numbered. Always equal to 0 or 1."""
        return self._base_index

    @property
    def all_unique(self) -> bool:
        """Indicates whether all keys/groups occur exactly once."""
        return self.unique_count == len(self.ikey)

    #---------------------------------------------------------------
    @property
    def ifirstkey(self):
        '''
        returns the row locations of the first member of the group

        Returns
        -------
        ifirstkey : np.ndarray of int
        '''
        # only categoricals do not have an iFirstKey
        if self.iFirstKey is None:

            # TJD TODO: if iFirstGroup exists, we can derive ifirstkey
            # skip over 0 bin
            if self.base_index ==1:
                self.iFirstKey = makeifirst(self.ikey, self.unique_count + 1)[1:-1]
            else:
                self.iFirstKey = makeifirst(self.ikey, self.unique_count)[1:]

        return self.iFirstKey

    #---------------------------------------------------------------
    @property
    def ilastkey(self):
        '''
        returns the row locations of the last member of the group

        Returns
        -------
        ilastkey : np.ndarray of int
        '''
        if self.iLastKey is None:

            # skip over 0 bin
            if self.base_index ==1:
                self.iLastKey = makeilast(self.ikey, self.unique_count + 1)[1:-1]
            else:
                self.iLastKey = makeilast(self.ikey, self.unique_count)[1:]

        return self.iLastKey

    #---------------------------------------------------------------
    @property
    def inextkey(self):
        '''
        returns the row locations of the next member of the group (or invalid int).

        Returns
        -------
        inextkey : np.ndarray of int
        '''
        if self.iNextKey is None:
            self.iNextKey = makeinext(self.ikey, self.unique_count)
        return self.iNextKey

    #---------------------------------------------------------------
    @property
    def iprevkey(self):
        '''
        returns the row locations of the previous member of the group (or invalid int)

        Returns
        -------
        iprevkey : np.ndarray of int
        '''
        if self.iPrevKey is None:
            self.iPrevKey = makeiprev(self.ikey, self.unique_count)
        return self.iPrevKey

    #---------------------------------------------------------------
    @property
    def igroup(self):
        '''
        returns a fancy index that when applied will make all the groups contiguous (packed together)

        Returns
        -------
        igroup : np.ndarray of int

        See Also
        --------
        ifirstgroup
        ncountgroup
        igroupreverse
        '''
        self.pack_by_group()
        return self.iGroup

    #---------------------------------------------------------------
    @property
    def igroupreverse(self):
        '''
        Returns the fancy index to reverse the shuffle from `igroup`.

        Returns
        -------
        igroupreverse : np.ndarray of int

        See Also
        --------
        igroup
        '''
        # only categoricals do not have an iFirstKey
        if self.iGroupReverse is None:
            self.iGroupReverse = rc.ReverseShuffle(self.igroup)
        return self.iGroupReverse

    #---------------------------------------------------------------
    @property
    def ncountgroup(self):
        '''
        returns a sister array used with `ifirstgroup` and `igroup`.

        Returns
        -------
        ncountgroup : np.ndarray of int

        See Also
        --------
        igroup
        ifirstgroup
        igroupreverse
        '''
        if self.nCountGroup is None:
            self.nCountGroup = rc.BinCount(self.ikey, self.unique_count +1)
        return self.nCountGroup

    #---------------------------------------------------------------
    @property
    def ifirstgroup(self):
        '''
        Returns a sister array used with `ncountgroup` and `igroup`.

        Returns
        -------
        ifirstgroup : np.ndarray of int

        See Also
        --------
        igroup
        ncountgroup
        igroupreverse
        '''
        self.pack_by_group()
        return self.iFirstGroup

    #---------------------------------------------------------------
    def _build_unique_dict(self, grouping:dict) -> dict:
        """Pull values from the non-unique grouping dict using the `iFirstKey` index.
        If enumstring is True, translate enum codes to their strings.
        """
        # regenerate
        # check isdirty flag
        # possibly apply filter, regroup
        uniquedict={}

        if self.iFirstKey is None and self.isenum:
            # force enum to generate ikey, ifirstkey, uniquecount
            _ = self.ikey

        for k, v in grouping.items():
            arr = v[self.iFirstKey]
            #if self.iFirstKey is None:
            #    arr = v
            #else:
            #    arr = v[self.iFirstKey]

            # flip enum codes to string
            if self.isenum:
                arr = self._enum.unique_categories(arr)

            # prepend an invalid bin here?
            uniquedict[k] = arr

        return uniquedict

    @property
    def _anydict(self):
        """Either the _grouping_dict or _grouping_unique_dict.
        Only be used for names, array datatypes.
        Will check for and return _grouping_dict first.
        """
        if self._grouping_dict is not None:
            return self._grouping_dict
        elif self._grouping_unique_dict is not None:
            return self._grouping_unique_dict
        else:
            raise ValueError(f'Error in anydict - neither grouping nor grouping unique dicts were set.')
    def _set_anydict(self, d):
        """Replace the dict returned by _anydict
        Will check for and set _grouping_dict first.
        """
        if self._grouping_dict is not None:
            self._grouping_dict = d
        elif self._grouping_unique_dict is not None:
            self._grouping_unique_dict = d
        else:
            raise ValueError(f'Error in anydict - neither grouping nor grouping unique dicts were set.')

    #---------------------------------------------------------------
    @property
    def uniquedict(self) -> Mapping[str, np.ndarray]:
        """
        Dictionary of key names -> array(s) of unique categories.

        `GroupBy` will pull values from non-unique dictionary using `iFirstKey`.
        `Categorical` already holds a unique dictionary.
        Enums will pull with `iFirstKey`, and return unique strings after translating integer codes.

        Returns
        -------
        dict
            Dictionary of key names -> array(s) of unique categories.

        Notes
        -----
        No sort is applied here.
        """
        if self._grouping_unique_dict is None:
            return self._build_unique_dict(self._grouping_dict)
        return self._grouping_unique_dict

    @property
    def uniquelist(self):
        """See Grouping.uniquedict
        Sets FastArray names as key names.
        """
        ulist = [*self.uniquedict.values()]
        for i, k in enumerate(self.uniquedict):
            ulist[i].set_name(k)
        return ulist

    #---------------------------------------------------------------
    @property
    def unique_count(self) -> int:
        '''
        Number of unique groups.
        '''
        if self.isenum and self._unique_count is None:
            #force regen of unique_count
            _ = self.ikey
        return self._unique_count

    #---------------------------------------------------------------
    @property
    def packed(self) -> bool:
        '''
        The grouping operation has performed an operation that requires packing e.g. ``median()``
        If `packed`, `iGroup`, `iFirstGroup`, and `nCountGroup` have been generated.
        '''
        return self._packed

    #---------------------------------------------------------------
    @property
    def gbkeys(self) -> Mapping[str, np.ndarray]:
        if self._gbkeys is None:
            self._gbkeys = self.uniquedict
        return self._gbkeys

    #---------------------------------------------------------------
    @property
    def isortrows(self):
        if self._isortrows is None:
            self._isortrows = self._make_isortrows(self.gbkeys)
        return self._isortrows

    #---------------------------------------------------------------
    @property
    def ncountkey(self):
        '''
        Returns
        -------
        ncountkey : np.ndarray of int
            An array with the number of unique counts per key
            Does include the zero bin
        '''
        return self.ncountgroup[1:]

    #---------------------------------------------------------------
    def set_name(self, name: str) -> None:
        """
        If the grouping dict contains a single item, rename it.

        This will make categorical results consistent with groupby results if they've been constructed
        before being added to a dataset.
        Ensures that label names are consistent with categorical names.

        Parameters
        ----------
        name : str
            The new name to use for the single column in the internal grouping dictionary.

        Examples
        --------
        Single key Categorical added to a Dataset, grouping picks up name:

        >>> c = rt.Categorical(['a','a','b','c','a'])
        >>> print(c.get_name())
        None

        >>> ds = rt.Dataset({'catcol':c})
        >>> ds.catcol.sum(rt.arange(5))
        *catcol   col_0
        -------   -----
        a             5
        b             2
        c             3

        Multikey Categorical, no names:

        >>> c = rt.Categorical([rt.FA(['a','a','b','c','a']), rt.FA([1,1,2,3,1])])
        >>> print(c.get_name())
        None

        >>> ds = rt.Dataset({'mkcol': c})
        >>> ds.mkcol.sum(rt.arange(5))
        *mkcol_0   *mkcol_1   col_0
        --------   --------   -----
        a                 1       5
        b                 2       2
        c                 3       3

        Multikey Categorical, already has names for its columns (names are preserved):

        >>> arr1 = rt.FA(['a','a','b','c','a'])
        >>> arr1.set_name('mystrings')
        >>> arr2 = rt.FA([1,1,2,3,1])
        >>> arr2.set_name('myints')
        >>> c = rt.Categorical([arr1, arr2])
        >>> ds = rt.Dataset({'mkcol': c})
        >>> ds.mkcol.sum(rt.arange(5))
        *mystrings   *myints   col_0
        ----------   -------   -----
        a                  1       5
        b                  2       2
        c                  3       3
        """
        gdict = self._anydict

        # always replace name for single key
        if len(gdict) == 1:
            newdict = {name : [*gdict.values()][0]}

        # multikey
        else:
            newdict = {}
            for i, (k,v) in enumerate(gdict.items()):
                # check if multikey is set to default keyname
                # keep name if different than default
                if k == GROUPBY_KEY_PREFIX+'_'+str(i):
                    k = name+'_'+str(i)
                newdict[k] = v

        self._set_anydict(newdict)

    #---------------------------------------------------------------
    def get_name(self):
        """List of grouping or grouping unique dict keys.
        """
        return [*self._anydict]

    #---------------------------------------------------------------
    def shrink(self, newcats, misc=None, inplace=False, name=None) -> 'Grouping':
        '''
        Parameters:
        ----------
        newcats : array_like
            New categories to replace the old - typically a reduced set of strings
        misc : scalar, optional
            Value to use as category for items not found in new categories. This will be added to the new categories.
            If not provided, all items not found will be set to a filtered bin.
        inplace : bool, not implemented
            If True, re-index the categorical's underlying FastArray.
            Otherwise, return a new categorical with a new index and grouping object.
        name

        Returns
        -------
        Grouping
            A new Grouping object based on this instance's data and the new set of labels provided in `newcats`.
        '''
        def ensure_scalar(misc):
            # if exists ensure is scalar, wrap in tuple so all modes hit same path
            if misc is not None:
                if np.isscalar(misc):
                    # wrap to hit same path as multikey
                    misc = tuple((misc,))
                else:
                    raise TypeError(f"If provided, misc must be a scalar value to match array of type {newcats.dtype}. Got {type(misc)}")
            return misc

        if self.isenum:
            # enum will flip to array-based grouping
            # need a 1-to-1 relationship between codes and categories
            if not isinstance(newcats, np.ndarray):
                newcats = TypeRegister.FastArray(newcats)
            misc = ensure_scalar(misc)

            if newcats.dtype.char in 'US':
                mask, _ = ismember(self._enum.category_array, newcats, base_index=0)
                idx = self._enum.code_array[mask]
            # possible TODO: add support for newcats in other formats?
            # integer array, new mapping, etc. need to decide on behavior
            else:
                raise TypeError(f"Shrinking enum/mapped grouping can only be done with a string array. Got type {newcats.dtype}")
            # will always return base-1 single key
            base_index = 1
            newcats = [newcats]

        else:
            base_index = self.base_index
            if base_index != 1:
                raise TypeError('Shrink can only be done if base_index =1')

            # ----singlekey
            if self.issinglekey:
                if not isinstance(newcats, np.ndarray):
                    newcats = TypeRegister.FastArray(newcats)

                # SupportedAlternate does not include Object or Void type
                if newcats.dtype.char not in NumpyCharTypes.SupportedAlternate:
                    raise TypeError(f"New uniques cannot be type {type(newcats)}.")

                misc = ensure_scalar(misc)

            # ----multikey
            else:
                if misc is not None:
                    # require scalar for each array
                    if not isinstance(misc, tuple):
                        arrtypes = tuple([a.dtype for a in self.uniquelist])
                        raise TypeError(f"If provided, misc must be a tuple of values corresponding to types for each array held {arrtypes}")

            if not isinstance(newcats, list):
                newcats = [newcats]

            oldcats = self.uniquelist

            # translate old index to new index position
            mask, idx = ismember(oldcats, newcats, base_index=base_index)

            # base 1 indexing
            idx +=1

            if misc is not None:
                # everything already 0 stays 0
                zeromask = self.catinstance == 0
                maxvalue = max(idx)+1
                # everything filtered out gets MISC
                idx[~mask] = maxvalue
                idx = hstack([0,idx,maxvalue])
                oldcats = newcats
                newcats = []

                # add the misc to end of the categories (handles multikey)
                for catkey, misckey in zip(oldcats, misc):
                    newcats.append(hstack([catkey, misckey]))
                newinstance = idx[self.catinstance]

                #put back the zeros (what was invalid before)
                newinstance[zeromask] =0

            else:
                # everything filtered out gets 0
                idx[~mask] = 0

                # TJD optimization here
                idx = hstack([0,idx])
                newinstance = idx[self.catinstance]

        # build new Grouping object based on new indexing and new categories
        result= Grouping(newinstance, newcats, categorical=True, base_index=base_index, sort_display=self.isdisplaysorted, ordered=False, _trusted=True, name=name)
        return result

    #---------------------------------------------------------------
    def regroup(self, filter=None, ikey=None) -> 'Grouping':
        """Regenerate the groupings iKey, possibly with a filter and/or eliminating unique values.

        Parameters
        ----------
        filter : np.ndarray of bool, optional
            Filtered bins will be marked as zero in the resulting iKey.
            If not provided, uniques will be reduced to the ones that occur in the iKey.
        ikey : np.ndarray of int, optional
            Only used when the grouping is in enum mode.

        Returns
        -------
        Grouping
            New Grouping object created by regenerating the `ikey`, `ifirstkey`, and `unique_count`
            using data from this instance.
        """
        def ifirstkey_regroup(ikey, ifirstkey, lex=False):

            '''
            Returns
            -------
            ifkremap is a fancy index mapping from the new ikey -> final ikey
            sortkey only if lex is True  (has unique old bin values)
            '''
            ifkremap = empty(len(ifirstkey)+1, dtype=self.ikey.dtype)
            idx = None
            ifkremap[0]=0
            sortkey= ikey[ifirstkey]
            if lex:
                # sort by the keys
                # assume base 1 which is why subtracts by 1

                # TJD to be looked at
                # Consider C++ routine which takes ikey_new, sortkey and returns the final ikey

                idx = lexsort(sortkey)
                idx = lexsort(idx)
                #skip over zero bin
                ifkremap[1:]=idx
                ifkremap[1:]+=1
                # ifkremap is NOT ifirstkey here
                # it is a fancy index used to remap ikey
            else:
                ifkremap[1:]=sortkey
                sortkey=None

            return ifkremap, sortkey
        #------------------------------
        def copy_gdict(src, dst, sortidx=None):
            # maybe put this in a more general copy routine?
            # need an example that isn't from categorical
            if src._grouping_dict is not None:
                if sortidx is None:
                    dst._grouping_dict = src._grouping_dict.copy()
                else:
                    dst._grouping_dict = {k:v[sortidx] for k,v in src._grouping_dict.items()}
            elif src._grouping_unique_dict is not None:
                if sortidx is None:
                    dst._grouping_unique_dict = src._grouping_unique_dict.copy()
                else:
                    dst._grouping_unique_dict = {k:v[sortidx] for k,v in src._grouping_unique_dict.items()}
            else:
                raise ValueError(f'Critical error in regroup - neither grouping nor grouping unique dicts were set.')
        #------------------------------

        if self.isenum:
            # apply filter if we have it
            if filter is not None:
                ikey=ikey[filter]

            # get the uniques
            uniques = unique(ikey, sorted=False)

            # now get expected uniques
            unumbers=self._enum.code_array
            ustrings=self._enum.category_array

            # find out which values still remain
            mask, index = ismember(unumbers, uniques)
            newdict = {k:v for k,v in zip(unumbers[mask], ustrings[mask])}

            # create a blank grouping object
            newgrouping = Grouping(None)
            newgrouping._enum = GroupingEnum(newdict, _trusted=True)
            copy_gdict(self, newgrouping)
            newgrouping._sort_display = self._sort_display
            newgrouping._categorical = self._categorical

        else:
            fdict = combine_accum1_filter(self.ikey, self.unique_count, filter=filter)
            ifirstkey = fdict['iFirstKey']
            ikey_new = fdict['iKey']
            unique_count_new = fdict['unique_count']

            # create a blank grouping object
            newgrouping = Grouping(None)

            # uniques didn't change, fix the new ikey to match old ordering
            if self._unique_count == unique_count_new:
                if filter is None:
                    ikey_fix = self.ikey[ifirstkey][ikey_new-1]
                else:
                    # TJD this path needs to be tested
                    ifkremap, sortkey = ifirstkey_regroup(self.ikey, ifirstkey)
                    ikey_fix = ifkremap[ikey_new]

                copy_gdict(self, newgrouping)

            # different number of uniques, need to fix up old index and new index
            else:
                ifkremap, sortkey = ifirstkey_regroup(self.ikey, ifirstkey, lex=True)
                # TJD ideally want an inplace remapping for speeed
                ikey_fix = ifkremap[ikey_new]

                # remove base 1 indexing
                sortkey-=1

                # a sort inplace is fine since we know they are all unique
                # TJD consider radix sort since this is just a reduction
                idx = sort(sortkey)
                #print('sortidx', idx)
                copy_gdict(self, newgrouping, sortidx=idx)

            # init defaults most attributes to none, set the bare minimum
            newgrouping._iKey = ikey_fix
            newgrouping.iFirstKey = ifirstkey
            newgrouping._unique_count = unique_count_new
            newgrouping._sort_display = self._sort_display
            newgrouping._categorical = self._categorical
            # how does enum get trimmed down?
            newgrouping._enum = self._enum

        return newgrouping

    #---------------------------------------------------------------
    def _make_isortrows(self, gbkeys):
        """Sort a single or multikey dictionary of unique values.
        Return the sorted index.
        """
        sortlist=list(gbkeys.values())
        sortlist.reverse()
        return lexsort(sortlist)

    #---------------------------------------------------------------
    def _finalize_dataset(self, accumdict, keychain, gbkeys, transform=False, showfilter=False, addkeys=False, **kwargs) -> 'Dataset':
        '''
        possibly transform?  TODO: move to here
        possibly reattach keys
        possibly sort

        Parameters
        ----------
        accumdict: dict or Dataset
        keychain:
        gbkeys may be passed as None
        '''
        # check if we have to sort to match gbkeys
        # only groupby does this
        # if we are transforming, do not do the sorting at end

        if gbkeys is None:
            if showfilter:
                gbkeys = keychain.gbkeys_filtered
            else:
                gbkeys = keychain.gbkeys

        isortrows = None
        if keychain.sort_gb_data and not transform:
            isortrows = keychain.isortrows
            if showfilter:
                isortrows = hstack((0,isortrows+1))

            # apply the sort
            for key, value in accumdict.items():
                # gbkeys are already sorted
                if key not in gbkeys:
                    accumdict[key]=accumdict[key][isortrows]

        accumDS = TypeRegister.Dataset(accumdict)

        if addkeys and not transform:
            # add back in groupbykeys last
            for key, value in gbkeys.items():
                if isortrows is None:
                    accumDS[key]=value
                else:
                    # TJD bug note here
                    # accumDS[key]=value[isortrows]
                    accumDS[key]=value

        # tag groupby keys in the dataset so they display
        accumDS.label_set_names(list(gbkeys))
        return accumDS

    #---------------------------------------------------------------
    def _return_dataset(self, origdict, accumdict:dict, func_num, return_all=False, col_idx=None, keychain=None, **kwargs) -> 'Dataset':
        '''

        '''
        showfilter = kwargs.get("showfilter",False)
        invalid = kwargs.get("invalid", False)
        showkeys = kwargs.get("showkeys",True)

        if showfilter:
            gbkeys = keychain.gbkeys_filtered
        else:
            gbkeys = keychain.gbkeys
            #gbkeys = self.uniquedict

        #print('gbkeys in return dataset',gbkeys)

        return_full = False

        # check for custom function and what they want returned
        if func_num >= GB_FUNC_NUMBA:
            for tbl in Grouping.REGISTERED_REVERSE_TABLES:
                lookup= tbl.get(func_num, None)
                if lookup is not None:
                    return_full =  lookup['return_full']

        if ((func_num >= GB_FUNCTIONS.GB_ROLLING_SUM and func_num < GB_FUNC_USER) or return_full or
            showkeys is False):
            # don't include gbkeys for these functions
            gbkeys = None

        if func_num == GB_FUNC_COUNT:
            # Count special column
            newdict = accumdict

        else:
            newdict = {}
            unique_rows = self.unique_count

            # go through origdict, because no-operation columns might be returned
            for key, value in origdict.items():
                if key in accumdict:
                    #print('key',key,'in accumdict')
                    #print('accumdict[key]',accumdict[key])
                    newdict[key] = accumdict[key]

                else:
                    # check if large dataset returned
                    if gbkeys is not None:
                        if key in gbkeys:
                            newdict[key]=gbkeys[key]

                        elif return_all:
                            # check if they want all columns returned
                            # TODO: do we really need a copy?
                            newdict[key] = value.fill_invalid(inplace=False, shape=unique_rows)

        #cumsum cumprod, rolling, are not keyed because long length
        if ((func_num >=GB_FUNCTIONS.GB_ROLLING_SUM and func_num < GB_FUNC_USER) or return_full or showkeys is False):
            if return_all:
                # they want the original gbkeys columns back
                for colname in keychain.gbkeys.keys():
                    newdict[colname] = origdict[colname]
            return TypeRegister.Dataset(newdict)
        else:
            # check if we have to sort to match gbkeys, only groupby does this
            return self._finalize_dataset(newdict, keychain, gbkeys, addkeys = (gbkeys is not None), **kwargs)

    #---------------------------------------------------------------
    def _make_accum_dataset(self, origdict, npdict, accum, funcNum, return_all=False, keychain=None, **kwargs):
        '''
        Returns a Dataset
        '''
        #fix up keys, for all values of None in accum, pull in the original
        accumdict = dict(zip(npdict,accum))
        removelist=[]
        for key, value in accumdict.items():
            #if we could not calculate this value but is a key, we have to pull it in
            if value is None:
                # remove from dictionary because nothing was calculated
                if Grouping.DebugMode: print("removing because value is none", key)
                removelist.append(key)

            #properly restore fastarray subclasses
            elif key in origdict:
                accumdict[key] = TypeRegister.newclassfrominstance(value, origdict[key])

        for key in removelist:
            del accumdict[key]

        return self._return_dataset(origdict, accumdict, funcNum, return_all=return_all, keychain=keychain, **kwargs)

    #---------------------------------------------------------------
    def pack_by_group(self, filter=None, mustrepack=False):
        '''
        Used to prepare data for custom functions

        Preapres 3 arrays
        -----------------
        iGroup: array size is same as multikey, unique keys are grouped together
        iFirstGroup: array size is number of unique keys for that group, indexes into isort
        nCountGroup: array size is number of unique keys for the group

        the user should use...
        igroup, ifirstgroup, ncountgroup

        If a filter is passed, it is remembered
        '''
        #self.unsort()
        if Grouping.DebugMode: print("packbygroup filter", filter)

        # if we did not pack by filter, and we are alerady packed, and there is not a filter...
        # do nothing
        if self._packed and filter is None:
           if not self._packedwithfilter:
               #keep lack of filter
               return

           if not mustrepack:
               #keep existing filter
               return

        self._packed=True

        iKey = self.ikey
        iNextKey = self.iNextKey

        if filter is not None:
            # force rebuild of None
            self._packedwithfilter=True
            iNextKey = None
            iKey = combine_filter(iKey, filter)
        else:
            self._packedwithfilter=False

        if Grouping.DebugMode: print("repacking", iKey, iNextKey)
        if Grouping.DebugMode: print("first", self.iFirstKey)

        # HACK: always pass  None for iNextKey until can normalize to int32
        # TODO: `groupbypack` says it takes nCountGroup (not iNextKey) -- can we pass that here to improve performance?
        packing= groupbypack(iKey, None, self.unique_count + 1, cutoffs = self._pcutoffs)
        self.iGroup = packing['iGroup']
        self.iFirstGroup = packing['iFirstGroup']
        self.nCountGroup = packing['nCountGroup']

        # reset since we do not know what past filter was
        self.iGroupReverse = None

        if Grouping.DebugMode: print("done repacking -- group", self.iGroup, self.iGroup.dtype,"\nFirst and Count", self.iFirstGroup, self.iFirstGroup.dtype, self.nCountGroup, self.nCountGroup.dtype)


    #---------------------------------------------------------------
    def _get_calculate_dict(self, origdict, funcNum, func=None, return_all=False, computable=True, func_param=0, **kwargs):
        '''
        Builds a dictionary to perform the groupby calculation on.

        If string/string-like columns cannot be computed, they will not be included.
        If specific columns have been specified (in col_idx, see GroupBy.agg), only they will be included.

        Returns
        -------
        npdict : dict
            Final dictionary for calculation.
        values : list
            List of columns in `npdict`. (NOTE: this is repetitive as npdict has these values also.)
        '''
        # incase of references, make a copy since we will delete certain keys
        npdict = origdict.copy()

        if 'col_idx' in kwargs:
            col_names = kwargs['col_idx']
            omit_list = [ key for key in npdict if (key not in col_names) and (key not in self._anydict) ]
            for item in omit_list:
                del npdict[item]

        # remove certain arrays from the calculation
        #except for count, we never calculate the groupby keys
        if return_all is False:
            for k in self._anydict:
                if k in npdict:
                    del npdict[k]

        # remove string like and categoricals from most calculations
        if computable:
            removelist=[]
            for k in npdict.keys():
                if not TypeRegister.is_spanlike(npdict[k]):
                    if TypeRegister.is_datelike(npdict[k]):
                        if funcNum == GB_FUNC_USER or funcNum not in GB_DATE_ALLOWED:
                            if Grouping.DebugMode: print("calc removing",k,funcNum)
                            removelist.append(k)

                    elif TypeRegister.is_string_or_object(npdict[k]):
                        if funcNum == GB_FUNC_USER or funcNum not in GB_STRING_ALLOWED:
                            if Grouping.DebugMode: print("calc removing",k,funcNum)
                            removelist.append(k)

            for key in removelist:
                if key in npdict:
                    del npdict[key]

        # check for just an array returned
        if isinstance(npdict, np.ndarray):
            try:
                name = npdict.get_name()
                if name is None:
                    name= 'col_0'
            except:
                name= 'col_0'

            # use the array name if it has one
            npdict = {name: npdict}


        values=list(npdict.values())

        #print('returning from get calculate dict')
        #print('igroup',self.iGroup, self.iGroup.dtype)
        #print('ifirstgroup',self.iFirstGroup, self.iFirstGroup.dtype)
        #print('ncountgroup',self.nCountGroup, self.nCountGroup.dtype)

        return npdict,values

    #---------------------------------------------------------------
    # not have to swallow all our special kwargs here so that we do not pass it on to userfunc
    # which might not be expecting it
    def apply_helper(self, isreduce:bool, origdict, userfunc: Callable, *args, tups=0, filter=None, showfilter:bool=False,
            label_keys=None, func_param=None, dtype=None, badrows=None, badcols=None, computable=True, **kwargs):
        '''
        Grouping apply_reduce/apply_nonreduce (for Categorical, groupby, accum2)

        For every column of data to be computed:
            The userfunc will be called back per group as a single array.  The order of the groups is either:
                1) Order of first apperance (when coming from a hash)
                2) Lexigraphical order (when lex=True or a Categorical with ordered=True)

        A reduce function must take an array as its first argument and return back a single scalar value.
        A non-reduce function must take an array as its first argument and return back another array.
        The first argument to apply MUST be the callable user function.
        The second argument to apply contains one or more arrays to operate one
             If passed as a list, the userfunc is called for each array in the list
             If passed as a tuple, the userfunc is called once with all the arrays as parameters

        Parameters
        ----------
        isreduce : bool
            Must be set.  True for reduce, False for non-reduce.
        origdict : dict of name:arrays
            The column names and arrays to apply the functio on.
        userfunc : callable
            A callable that takes one or more arrays as its first argument, and returns an array or scalar.
            If `isreduce` is True, `userfunc` is a reduction and should return a scalar;
            when `isreduce` is False, `userfunc` is a nonreduce/scan/prefix-sum and should return an array.
            In addition the callable may take positional arguments and keyword arguments.
        *args
            Any additional user arguments to pass to `userfunc`.

        Other Parameters
        ----------------
        tups : 0
            Set to 1 if `userfunc` wants multiple arrays passed fixed up by iGroup. Defaults to False.
            Set to 2 for passing in constants
        showfilter : bool
            Set to True to calculate filter. Defaults to False.
        filter : ndarray of bools
            optional boolean filter to apply
        label_keys: rt.GroupByKeys, the labels on the left
        func_param : tuple, optional
            Caller may pass ``func_param=(arg1, arg2)`` to pass arguments to `userfunc`.
        dtype : str or np.dtype, or dict of np.dtypes, optional
            Explicitly specify the dtype for the output array. Defaults to None, which means the function chooses
            a compatible dtype for the output.
            If a dict of np.dtypes is passed, multiple output arrays are allocated based on the specified dtypes.
        badrows
            not used, may be passed from Acccum2
        badcols
            not used

        Notes
        -----
        All other arguments passed to this function (if any remaining) will be passed through to `userfunc`.

        See Also
        --------
        GroupByOps.apply_reduce
        GroupByOps.apply_nonreduce
        '''
        # TODO: apply and apply_reduce share similar code
        # TODO: write a common routine for both

        if Grouping.DebugMode: print("origdict:", origdict, "\nuserfunc:", userfunc, "\nargs:", args, "\nkwargs:", kwargs, "\nlabels:", label_keys, "\ntups:", tups)
        if not callable(userfunc):
            raise TypeError(f"userfunc {userfunc!r} is not callable and cannot be applied to the dataset.")

        if tups != 1:
            # pass in all columns including strings, datetime, etc
            npdict, values = self._get_calculate_dict(origdict, GB_FUNC_USER, userfunc, return_all=False, computable=computable, **kwargs)
        else:
            # blindly pass in what user wants
            npdict = origdict

        #print("npdict", npdict, "values", values)
        # happens when apply_user and the array is removed in get_calculate_dict
        if len(npdict)==0:
            return None

        # ALWAYS force this to clear out any previous filter
        self._packed = False
        self.pack_by_group(filter)

        base_bin=1
        if showfilter:
            base_bin=0

        if func_param is None:
            func_param = ()

        #print("basebin", base_bin, "sf", showfilter)

        # +1 for invalid bin
        return_rows = self.unique_count +1
        #bins = range(1, unique_rows+1)

        return_arr = None
        return_ds = TypeRegister.Dataset({})

        # check if they want full array returned (not a reduction)
        if not isreduce:
            reverse_back = self.igroupreverse

        # check if passing multiple arrays at same time
        if tups == 1:
            # change npdict to list of arrays
            arr_list = []
            firstname = None
            for colname, arr in npdict.items():
                if firstname is None:
                    firstname = colname
                arr_list.append(arr)

            npdict={firstname: arr_list}

            # use a set to make sure only one length
            arr_len = {len(v) for v in arr_list}
            if len(arr_len) != 1:
                raise ValueError(f"More than one array was passed to apply_*, but the arrays are different lengths: {arr_len!r}.  All arrays must be of length:{len(self.iGroup)}")
        elif tups==2:
            arr_const = tuple(v[self.iGroup] for v in args[0])
            # move real arguments over since the first argument are the array constants
            args=args[1:]

        if isinstance(dtype, dict):
            # a dict specified in dtype indicates one or more named return arrays that will be passed in
            # this mode indicates the possibility of future parallelization
            # return_arr is one or more named arays
            if isreduce:
                return_arr = { colname:empty((return_rows,), dtype=dtypewanted) for colname, dtypewanted in dtype.items() }
            else:
                return_arr = { colname:empty(self.iGroup.shape, dtype=dtypewanted) for colname, dtypewanted in dtype.items() }

            outputs = tuple(return_arr.values())
        else:
            outputs = None

        ifirst = self.iFirstGroup
        ncount = self.nCountGroup

        # for all the columns we have to process
        for colname, arr in npdict.items():

            # reorder the data so groups are contiguous
            # one big fancy index pull
            if tups > 0:
                # TJD check for mode where tups are passed, but so is a dataset
                # reorder multiple arrays to make groups contiguous
                if tups == 1:
                    arr=tuple(v[self.iGroup] for v in arr)
                else:
                    # additional args passed not in tuple form
                    # TODO?? do we convert these?
                    #arr=tuple(v[self.iGroup] for v in args)
                    # add in constant args
                    arr=(arr[self.iGroup],)+ arr_const + args

            else:
                # reorder the array to make groups contiguous
                arr=arr[self.iGroup]
                if outputs is not None:
                    arr=tuple(arr)

            if isreduce:
                #-------------------------------------------
                # Reduce path -- expecting a scalar returned
                #-------------------------------------------
                needinvalidfill=False
                invalid = 0

                # check for multiple output array path (specified by a dict dtype)
                if outputs is not None:
                    # think about showfilter where base is 0
                    # loop over all the groups
                    for i in range(1, ifirst.shape[0]):
                        first=ifirst[i]
                        last=first + ncount[i]
                        # call the user func with one or more input arrays, one or more output arrays, any additional args, kwargs
                        userfunc(*tuple(v[first:last] for v in arr), *tuple(v[i:i+1] for v in outputs), *args, **kwargs)

                    # build dataset column by column
                    for colname, arr in return_arr.items():
                        return_ds[colname]=arr[base_bin:]

                else:
                    if dtype is not None:
                        # if they specified a single dtype, allocate the array
                        return_arr = empty((return_rows,), dtype=dtype)
                        invalid = INVALID_DICT[dtype.num]

                    # loop over all groups
                    # TODO: consider c++ loop? (i.e. vectorize this)
                    for i, (first, count) in enumerate(zip(ifirst, ncount)):
                        if count > 0:
                            last=first + count

                            if tups > 0:
                                # call user function with multiple array inputs (with slice for the group)
                                result = userfunc(*tuple(v[first:last] for v in arr), *args, *func_param, **kwargs)
                            else:
                                # call user function with input (with slice for the group)
                                result = userfunc(arr[first:last], *args, *func_param, **kwargs)

                            if return_arr is None:
                                # first time check
                                if not np.isscalar(result) and len(result) != 1:
                                    if not isinstance(result, (tuple, np.ndarray, list, dict)):
                                        raise TypeError(f'apply_reduce user function must return a scalar, tuple, or dict not {result!r} with type:{type(result)!r}')
                                    # create an object array and vstack it later
                                    return_arr = empty((return_rows,), dtype='O')
                                else:
                                    # first time we got back data -- we can determine the dtype now
                                    if hasattr(result,'dtype'):
                                        dtype = result.dtype
                                    else:
                                        dtype = np.dtype(type(result))
                                    return_arr = empty((return_rows,), dtype=dtype)
                                    invalid = INVALID_DICT[dtype.num]

                            return_arr[i] = result
                        else:
                            needinvalidfill=True

                    if return_arr.dtype.char != 'O':
                        if needinvalidfill:
                            # what if everything is filtered out?
                            return_arr[self.nCountGroup==0]=invalid
                        # build dataset column by column
                        return_ds[colname]=return_arr[base_bin:]
                    else:
                        # see what user returned
                        return_arr = return_arr[base_bin:]
                        firstelement=return_arr[0]
                        if isinstance(firstelement, (list,tuple,np.ndarray)):
                            tlen = len(firstelement)
                            multiarray=np.vstack(return_arr)
                            if multiarray.dtype.char == 'O':
                                multiarray = multiarray.astype(np.asarray(firstelement[0]).dtype)
                            return_ds[colname]=multiarray
                        else:
                            raise TypeError(f'firstelement user function must return a scalar, tuple, or dict not {firstelement!r} with type: {type(firstelement)!r}')

            else:
                #----------------------------------------------
                # NonReduce path -- expecting an array returned
                #----------------------------------------------
                # check for multiple output array path
                if outputs is not None:

                    if False:
                        pass
                        # TJD future speed ups will use a numba loop such as below
                        #import numba as nb
                        #@nb.jit(parallel=True, nopython=True)
                        #def _numba_nonreduce2_1(userfunc, ifirst, ncount, in1, in2, kwarg1):
                        #    for i in nb.prange(1, ifirst.shape[0]):
                        #        first=ifirst[i]
                        #        last=first + ncount[i]
                        #        # call the user func with one or more input arrays, one or more output arrays, any additional args, kwargs
                        #        userfunc(in1[first:last], in2[first:last], kwarg1)

                        #@nb.jit(parallel=True, nopython=True)
                        #def _numba_nonreduce2_2(userfunc, ifirst, ncount, in1, in2, kwarg1, kwarg2):
                        #    for i in nb.prange(1, ifirst.shape[0]):
                        #        first=ifirst[i]
                        #        last=first + ncount[i]
                        #        # call the user func with one or more input arrays, one or more output arrays, any additional args, kwargs
                        #        userfunc(in1[first:last], in2[first:last], kwarg1, kwarg2)

                        #@nb.jit(parallel=True, nopython=True)
                        #def _numba_nonreduce3_1(userfunc, ifirst, ncount, in1, in2, in3, kwarg1):
                        #    for i in nb.prange(1, ifirst.shape[0]):
                        #        first=ifirst[i]
                        #        last=first + ncount[i]
                        #        # call the user func with one or more input arrays, one or more output arrays, any additional args, kwargs
                        #        userfunc(in1[first:last], in2[first:last], in3[first:last], kwarg1)

                        #@nb.jit(parallel=True, nopython=True)
                        #def _numba_nonreduce3_2(userfunc, ifirst, ncount, in1, in2, in3, kwarg1, kwarg2):
                        #    for i in nb.prange(1, ifirst.shape[0]):
                        #        first=ifirst[i]
                        #        last=first + ncount[i]
                        #        # call the user func with one or more input arrays, one or more output arrays, any additional args, kwargs
                        #        userfunc(in1[first:last], in2[first:last], in3[first:last], kwarg1, kwarg2)

                        ## combine args and kwargs into a list
                        ## pass all arg first, then all kwargs
                        #inlist = [*arr] + [*outputs]
                        #arglist=[*args]
                        #klist = arglist + [*kwargs.values()]
                        #if len(inlist) == 2:
                        #    if len(klist) ==1:
                        #        _numba_nonreduce2_1(userfunc, ifirst, ncount, inlist[0], inlist[1], inlist[2], klist[0])
                        #    elif len(klist) ==2:
                        #        _numba_nonreduce2_2(userfunc, ifirst, ncount, inlist[0], inlist[1], inlist[2], klist[0], klist[1])
                        #    else:
                        #        raise ValueError("experimental numba not ready", len(inlist))
                        #elif len(inlist) == 3:
                        #    if len(klist) ==1:
                        #        _numba_nonreduce3_1(userfunc, ifirst, ncount, inlist[0], inlist[1], inlist[2], klist[0])
                        #    elif len(klist) ==2:
                        #        _numba_nonreduce3_2(userfunc, ifirst, ncount, inlist[0], inlist[1], inlist[2], klist[0], klist[1])
                        #    else:
                        #        raise ValueError("experimental numba not ready", len(inlist))
                        #else:
                        #    raise ValueError("experimental numba not ready", len(inlist))

                    else:
                        # loop over all the groups
                        for i in range(1, ifirst.shape[0]):
                            first=ifirst[i]
                            last=first + ncount[i]
                            # call the user func with one or more input arrays, one or more output arrays, any additional args, kwargs
                            userfunc(*tuple(v[first:last] for v in arr), *tuple(v[first:last] for v in outputs), *args, **kwargs)

                    # build dataset column by column
                    for colname, ret_arr in return_arr.items():
                        return_ds[colname]=ret_arr[reverse_back]

                else:
                    if dtype is not None:
                        return_arr = empty(self.iGroup.shape, dtype=dtype)

                    # loop over all groups
                    # TODO: consider c++ loop? (i.e. vectorize this)
                    for first, count in zip(ifirst, ncount):

                        # currently we do not pass in empty arrays... should we?i put in t
                        if count > 0:
                            last=first + count

                            if tups > 0:
                                # call user function with multiple array inputs (with slice for the group)
                                #inputarr = (v[first:last] for v in arr)
                                #print("**", arr, type(arr))
                                result = userfunc(*tuple(v[first:last] for v in arr), *args, *func_param, **kwargs)
                            else:
                                # call user function with input (with slice for the group)
                                result = userfunc(arr[first:last], *args, *func_param, **kwargs)

                            if return_arr is None:
                                if not isinstance(result, np.ndarray):
                                    if not hasattr(result, 'dtype'):
                                        raise TypeError(f'The apply_nonreduce user function must return an object with a  dtype: {result!r}.  The apply_reduce function handles a scalar.')
                                    #raise TypeError(f'The apply_nonreduce user function must return an array not {result!r}')

                                # first time we got back data -- we can determine the dtype now
                                return_arr = empty(self.iGroup.shape, dtype=result.dtype)

                                # to rewrap categoricals or datelike
                                if hasattr(result, 'newclassfrominstance'):
                                    return_arr = result.newclassfrominstance(return_arr, result)

                            try:
                                return_arr[first:last] = result
                            except Exception:
                                raise ValueError(f"The user function called from apply_nonreduce, did not return an array of the proper length.  Expecting {last-first} with dtype {return_arr.dtype!r} but got {result!r}")

                    return_ds[colname]=return_arr[reverse_back]
                    #help recycling
                    return_arr = None

        # reattach keys
        if label_keys is not None and isreduce:
            return_ds = self._finalize_dataset(return_ds, label_keys, label_keys.gbkeys, showfilter=showfilter, addkeys=True, **kwargs)

        return return_ds

    #---------------------------------------------------------------
    def apply(self, origdict, userfunc, *args, tups=0, filter=None, label_keys=None, return_all=False, **kwargs):
        """
        Grouping apply (for Categorical, groupby, accum2)
        Apply function userfunc group-wise and combine the results together.
        The userfunc will be called back per group.  The order of the groups is either:
            1) Order of first apperance (when coming from a hash)
            2) Lexigraphical order (when lex=True or a Categorical with ordered=True)

        If a group from a categorical has no rows (an empty group), then a dataset
        with one row of invalids (as a place holder) will be used and the userfunc will be called.

        The function passed to apply must take a Dataset as its first argument and return a
            1) Dataset (with one or more rows returned)
            2) dictionary of name:array pairs
            3) single array

        The set of returned columns must be consistent for each input (group) dataset.
        ``apply`` will then take care of combining the results back
        together into a Dataset with the groupby key(s) in the initial column(s).
        ``apply`` is therefore a highly flexible grouping method.

        While ``apply`` is a very flexible method, its downside is that using it can be quite a bit slower
        than using more specific methods. riptable offers a wide range of methods that will be much faster
        than using ``apply`` for their specific purposes, so try to use them before reaching for ``apply``.

        Parameters
        ----------
        userfunc: callable
            A callable that takes a Dataset as its first argument, and returns a Dataset, dict, or single array.
            In addition the callable may take positional and keyword arguments.
        args, kwargs: tuple and dict
            Optional positional and keyword arguments to pass to `userfunc`

        Returns (2 possible)
        -------
        Dataset that is grouped by (reduced from original dataset)
        Dataset of original length (not grouped by)

        Examples
        --------
        >>> ds = rt.Dataset({'A': 'a a b'.split(), 'B': [1,2,3], 'C': [4,6, 5]})
        >>> g = rt.GroupBy(ds, 'A')

        From ``ds`` above we can see that ``g`` has two groups, ``a`` and ``b``. Calling ``apply`` in various ways,
        we can get different grouping results.

        Example 1: below the function passed to apply takes a Dataset as its argument and returns a
        Dataset or dictionary with one row for each row in each group.
        ``apply`` combines the result for each group together into a new Dataset:

        >>> g.apply(lambda x: x.sum())
        *A   B    C
        --   -   --
        a    3   10
        b    3    5

        >>> g.apply(lambda x: {'B':x.B.sum()})
        *A   B
        --   -
        a    3
        b    3

        Example 2: The function passed to ``apply`` takes a Dataset as its argument and returns a
        Dataset with one row per group. ``apply`` combines the result for each group together into a new Dataset:

        >>> g.apply(lambda x: x.max() - x.min())
        *A   B   C
        --   -   -
        a    1   2
        b    0   0

        Example 3: The function passed to ``apply`` takes a Dataset as its argument and returns a
        Dataset with one row and one column per group (i.e., a scalar). ``apply`` combines the
        result for each group together into a Dataset:

        >>> g.apply(lambda x: rt.Dataset({'val': [x.C.max() - x.B.min()]}))
        *A   val
        --   ---
        a      5
        b      2

        Example 4: The function returns a Dataset with more than one row.

        >>> g.apply(lambda x: x.cumsum())
        *A   B    C
        --   -   --
        a    1    4
        a    3   10
        b    3    5

        Example 5: A non-lambda, user-supplied function which creates a new column in the existing Dataset.

        >>> def userfunc(x):
                x.Sub = x.C - x.B
                return x
        >>> g.apply(userfunc)
        *A   B   C   Sub
        --   -   -   ---
        a    1   4     3
        a    2   6     4
        b    3   5     2
        """
        def make_invalid_dict(inv_count, result):
            # make an invalid dict from user returned dataset
            inv_dict={}
            for k,v in result.items():
                # NOTE: what about 2 dimensional array?
                inv_dict[k]=v.fill_invalid(shape=(invcount,), inplace=False)
            return inv_dict

        if Grouping.DebugMode: print("origdict", origdict, "userfunc", userfunc, "args", args)
        if not callable(userfunc):
            raise TypeError(f"userfunc {userfunc!r} is not callable and cannot be applied to the dataset.")

        self.pack_by_group(filter, mustrepack=True)

        # pass in all columns including strings, datetime, etc
        npdict, values = self._get_calculate_dict(origdict, GB_FUNC_USER, userfunc, return_all=return_all, computable=False, **kwargs)

        unique_rows = self.unique_count

        # list of datasets for each operation result
        accum=[]

        # skip over zero bin?  what if user wants to see it
        bins = range(1, unique_rows+1)
        #bins = self.isortrows

        if Grouping.DebugMode: print(f"user wants to call function with name {userfunc.__name__!r}")

        call_user_func = True
        # check to see if dataset has its own version of the operation)
        try:
            dsfunc= getattr(TypeRegister.Dataset, userfunc.__name__)
            if callable(dsfunc):
                call_user_func = False
                if Grouping.DebugMode: print(f"dataset has function for {userfunc.__name__!r}")
        except:
            pass

        emptygroup=None
        lastexception = None
        userfuncsuccess = False
        totalcallbacks =0

        for i in bins:
            # groupby ikey uses base-1 indexing
            filter_group = self.as_filter(i)
            if Grouping.DebugMode: print("index", i, "filter is", filter_group)

            # check for an empty group
            if len(filter_group) > 0:
                # make a new dictionary using the groupby filter
                newdict = {k:v[filter_group] for k,v in npdict.items()}

                #else:
                #    # this group was probably filtered out, pass the user a row full of invalids
                #    # only build the emptyGroup once
                #    newdict={}
                #    for k,v in npdict.items():
                #        # NOTE: what about 2 dimensional array?
                #        newdict[k]=v.fill_invalid(shape=(1,), inplace=False)

                # NOTE: to optimize, reuse same dataset over and over
                newds = TypeRegister.Dataset(newdict)

                success = False
                if call_user_func:
                    # testing: need to check what the error is
                    #accumGbDset = userfunc(newds, *args, **kwargs)
                    try:
                        # call the user function
                        accumGbDset = userfunc(newds, *args, **kwargs)
                        success = True
                        # indicate we succeeded at least once
                        userfuncsuccess =True
                    except Exception as e:
                        # if we were successful before but not now
                        if userfuncsuccess:
                            raise e
                        # stop going down this path
                        lastexception = e
                        call_user_func = False

                if not success:
                    try:
                        # check to see if the dataset has a way to perform the operation - e.g. Dataset.sum()
                        accumGbDset = newds.apply(userfunc, *args, **kwargs)
                    except Exception as e:
                        if lastexception is not None:
                            raise lastexception
                        raise e

                if Grouping.DebugMode: print(f"accumbin for {i} is {accumGbDset}")

                # if they just return an array, make it a dictionary
                if isinstance(accumGbDset, np.ndarray):
                    accumGbDset={'col1': accumGbDset}

                if not isinstance(accumGbDset, TypeRegister.Dataset):
                    # if nothing returned, assume they modified the ds they were passed in
                    if accumGbDset is None:
                        accumGbDset = newds
                    else:
                        accumGbDset=TypeRegister.Dataset(accumGbDset)
                    #raise TypeError(f"return from groupby apply is not a Dataset {type(accumGbDset)}")

                accum.append(accumGbDset)
                totalcallbacks += 1

        if len(accum) < 1:
            raise ValueError("no datasets were returned, groupby may be empty")

        if Grouping.DebugMode: print("apply completed!", accum)

        # concatenate all the arrays together for each column
        # return a dataset
        result = hstack(accum)

        result_len = result.shape[0]

        if Grouping.DebugMode: print("result_len, unique_rows, bins, totalcallbacks", result_len, unique_rows, len(bins), totalcallbacks)

        # check if the user wanted a reduce
        if result_len == totalcallbacks:

            inv_delta = unique_rows - totalcallbacks
            if inv_delta > 0:
                f = self.nCountGroup[1:] > 0
                if Grouping.DebugMode: print("reduce with a filter", f)
                # if a reduce function was used, and some rows were filtered out
                # filter out the gbkeys the same way
                for k, v in label_keys.gbkeys.items():
                    if Grouping.DebugMode: print("adding column", k, "value", v, "filter", f)
                    result[k] = v[f]

            result = self._finalize_dataset(result, label_keys, label_keys.gbkeys, addkeys=(inv_delta <= 0), **kwargs)
        else:

            if self.nCountGroup[0] > 0:
                # this group was probably filtered out, pass the user a row full of invalids
                # only build the emptyGroup once
                invcount = self.nCountGroup[0]
                newdict=make_invalid_dict(invcount, result)

                if Grouping.DebugMode: print(f"**should insert {invcount} at..{self.iFirstGroup[0]}")
                inv_ds = TypeRegister.Dataset(newdict)
                result = hstack([inv_ds, result])

            # if function was not a reduce function, possibly attach the keys
            # should this be the return_all keyword?

            # use will return mini dataset per group
            # the datasets were hstacked and are contiguous per group
            # we need to ungroup them back into the original order

            # NOTE: what happens with filtered out groups (they are in the zero bin)
            result[self.iGroup,:] = result.copy()

            # should this be an option
            try:
                for k,v in self._grouping_dict.items():
                    result[k]=v
                # set as label column if not unique?
                result.label_set_names(list(self._grouping_dict))

            # grouping dict is not length of ikey
            except:
                pass

        return result

    ##---------------------------------------------------------------
    def onedict(self, unicode=False, invalid=True, sep='_'):
        """
        Concatenates multikey groupings with underscore to make a single key.
        Adds 'Inv' to first element if kwarg Invalid=True.

        Parameters
        ----------
        unicode: boolean, default False
            whether to create a string or unicode based array
        invalid: boolean, default True
            whether or not to add 'Inv' as the first unique

        Returns
        -------
        a string of the new key name
        a new single array of the uniques concatenated
        """
        stringtype='S'
        if unicode:
            stringtype='U'

        name= ''
        arr=None
        if invalid:
            invalidarr = TypeRegister.FastArray(['Inv'], unicode=unicode)

        for k,v in self._grouping_unique_dict.items():
            if len(name) > 0: name += sep
            name += k
            if invalid:
                v=hstack([invalidarr, v])
            if arr is None: arr=v.astype(stringtype)
            else: arr = arr + sep + v.astype(stringtype)

        # name the array
        arr.set_name(name)
        return name,  arr


    ##---------------------------------------------------------------
    @classmethod
    def register_functions(cls, functable):
        cls.REGISTERED_REVERSE_TABLES.append(functable)

    #---------------------------------------------------------------
    def _calculate_all(self, origdict, funcNum, func_param=0,
                       keychain=None, user_args=(), tups=0, accum2=False, return_all=False, **kwargs):
        '''
        All groupby calculations from GroupBy, Categorical, Accum2, and some groupbyops will be enter through this method.

        Parameters
        ----------
        orgidict
        funcNum
        func_param : int
            parameters from GroupByOps (often simple scalars)
        keychain
            option groupby keys to apply to the final dataset at end
        user_args
            A tuple of None or more arguments to pass to user_function.
            `user_args` only exists for apply* related function calls
        tups : int, 0
            Defaults to 0. 1 if user functions had tuples () indicating to pass in all arrays.
            `tups` is only > 0 for apply* related function calls where the first parameter was (arr1, ..)
        accum2 : bool
        return_all : bool

        Other Parameters
        ----------------
        showfilter : bool, optional
            If set will calculate contents in the 0 bin.

        See Also
        --------
        Grouping
        '''

        invalid = kwargs.get("showfilter", False)
        if invalid:
            base_bin=0
        else:
            base_bin=1

        # always show all row and calculate the filtered row even if there is no filter
        if funcNum >= GB_FUNCTIONS.GB_ROLLING_SUM and funcNum < GB_FUNCTIONS.GB_CUMSUM:
            base_bin = 0

        filter=kwargs.get('filter',None)

        mustpack = False
        lookup = None
        # pack for certain operations if not already packed
        # TODO: store funcNum/gb level information in a friendly variable up front
        if funcNum >= GB_FUNCTIONS.GB_FIRST and funcNum < GB_FUNCTIONS.GB_CUMSUM:
            mustpack=True

        # check for custom function
        # check for packing
        if funcNum >= GB_FUNC_NUMBA:
            for tbl in Grouping.REGISTERED_REVERSE_TABLES:
                lookup= tbl.get(funcNum, None)
                if lookup is not None:
                    break
            if lookup is not None and lookup['packing']  == GB_PACKUNPACK.PACK:
                mustpack = True

        # fetch ikey once
        ikey = self.ikey

        # check if we have to pack the data
        if mustpack:
            self.pack_by_group(filter, mustrepack=True)
        else:
            if filter is not None:
                ikey = combine_filter(ikey, filter)

        npdict, values = self._get_calculate_dict(origdict, funcNum, return_all=return_all, **kwargs)
        unique_rows = self.unique_count

        accum=[]
        accum_tuple = None
        fullrows = False
        empty_allowed = self._empty_allowed(funcNum)

        if len(values) != 0 or empty_allowed:

            if Grouping.DebugMode:
                print('values',values, values[0].dtype)
                print('ikey',self.ikey, self.ikey.dtype)
                if mustpack:
                    print('igroup',self.iGroup, self.iGroup.dtype)
                    print('ifirstgroup',self.iFirstGroup, self.iFirstGroup.dtype)
                    print('ncountgroup',self.nCountGroup, self.nCountGroup.dtype)
                print('unique_rows',unique_rows)
                print('funcNum',funcNum)
                print('func_param',func_param)
                print('base_bin',base_bin)
                min = ikey.min()
                max = ikey.max()
                print("***minmax", min, max, funcNum, func_param)
                if (min < 0): print("min out of range", min)
                if (max > unique_rows): print("max out of range", max, unique_rows)

            # special check
            if funcNum == GB_FUNCTIONS.GB_ROLLING_COUNT:
                # TJD note cumcount is similar to count() -- it is special in that it returns one array always
                values = [ikey]

            funcList = [funcNum]*len(values)
            binLowList = [base_bin]*len(values)
            binHighList = [unique_rows+1]*len(values)

            # check for custom function
            if funcNum >= GB_FUNC_NUMBA:
                if lookup is not None:
                    func_gb = lookup['func_gb']
                    if func_param ==0:
                        func_param = []

                    if lookup['packing']  == GB_PACKUNPACK.PACK:
                        accum = func_gb(values, ikey, self.iGroup, self.iFirstGroup, self.nCountGroup, unique_rows, funcList, binLowList, binHighList, func_param)
                    else:
                        accum_tuple = func_gb(values, ikey, unique_rows, funcList, binLowList, binHighList, func_param)

            # accum not packed
            elif funcNum >= GB_FUNCTIONS.GB_CUMSUM:
                # NOTE: filter is the third argument of the func_param tuple
                # TODO: do you want to merge with iKey?
                accum = rc.EmaAll32(values, ikey, unique_rows, funcNum, func_param)

            # basic not packed
            elif funcNum >= GB_FUNCTIONS.GB_SUM and funcNum < GB_FUNCTIONS.GB_FIRST:

                accum_tuple = _groupbycalculateall(values, ikey, unique_rows, funcList, binLowList, binHighList, func_param)

                if Grouping.DebugMode: print("accum tuple", accum_tuple)

            # packed
            elif funcNum >= GB_FUNCTIONS.GB_FIRST and funcNum < GB_FUNCTIONS.GB_CUMSUM:

                accum_tuple = _groupbycalculateallpack(values, ikey, self.iGroup, self.iFirstGroup, self.nCountGroup, unique_rows, funcList, binLowList, binHighList, func_param)

                # early exit
                if funcNum == GB_FUNCTIONS.GB_ROLLING_COUNT:
                    return accum_tuple[0]

            # non-accum
            if accum_tuple is not None:
                for v in accum_tuple:
                    if accum2 or base_bin == 0:
                        #keep all rows for accum
                        accum.append(v)
                    else:
                        #remove first row (invalid)
                        accum.append(v[1:])

            # accum2 will take over from here
            if accum2:
                return dict(zip(npdict,accum))

        else:
            if accum2:
                raise TypeError(f"Nothing was calculated for Accum2 operation.")
            print("Warning: nothing calculated.")

        #create a new dataset from the groupby results
        if Grouping.DebugMode: print("calculateallpacked!", len(accum[0]), accum)
        dset = self._make_accum_dataset(origdict, npdict, accum, funcNum, return_all=return_all, keychain=keychain, **kwargs)

        return dset


    #---------------------------------------------------------------
    def sort(self, keylist):
        raise TypeError(f"Sorting of groupby keys and groupby data is now handled by the GroupByKeys class.")

    #---------------------------------------------------------------
    def as_filter(self,index):
        '''
        Returns an index filter for a given unique key

        Examples
        -------

        '''
        first=self.ifirstgroup[index]
        last=first + self.ncountgroup[index]
        return self.igroup[first:last]

    #---------------------------------------------------------------
    def _empty_allowed(self, funcNum):
        '''
        Operations like cumcount do not need an origdict to calculate. Calculations are made only
        on binned columns. Might be more later, so keep here.
        '''
        return funcNum == GB_FUNCTIONS.GB_ROLLING_COUNT

    #---------------------------------------------------------------
    def count(self, gbkeys=None, isortrows=None, keychain=None, filter=None, transform=False, **kwargs) -> 'Dataset':
        '''
        Compute count of each unique key
        Returns a dataset containing a single column. The Grouping object has the ability to generate
        this column on its own, and therefore skips straight to _return_dataset versus other
        groupby calculations (which pass through _calculate_all) first.
        '''
        # make a new dataset with the same number of rows

        showfilter=kwargs.get('showfilter', False)
        if showfilter is True:
            base=0
        else:
            base=1

        if filter is not None:
            # create a temp filter where nonfilter is 0
            ikey = where(filter, self.ikey, 0)
            ncountkey = rc.BinCount(ikey, self.unique_count+1)
        else:
            # ncountgroup will call BinCount if it needs to
            ncountkey = self.ncountgroup
            ikey=self.ikey

        # skip if base 1
        countcol = ncountkey[base:]
        if transform:
            if not showfilter and self.base_index == 1:
                ikey = ikey - 1
            countcol = countcol[ikey]

        accumdict = {'Count':countcol}
        #return self._make_accum_dataset(origdict, accumdict, accumdict['Count'], GB_FUNC_COUNT)
        return self._return_dataset(None, accumdict, GB_FUNC_COUNT, keychain=keychain, transform=transform, **kwargs)

    @staticmethod
    def _hstack(
        glist: List['Grouping'],
        _trusted: bool = False,
        base_index: int = 1,
        ordered: bool = False,
        destroy: bool = False
    ) -> 'Grouping':
        """
        'hstack' operation for Grouping instances.

        Parameters
        ----------
        glist : list of Grouping
            A list of Grouping objects.
        _trusted : bool
            Indicates whether we need to validate the data in the supplied Grouping
            instances for consistency / correctness before using it. In certain cases,
            the caller knows the data is safe to use directly (e.g. because they've just
            created it), so the validation can be skipped.
        base_index : int
            The base index to use for the resulting Categorical.
        ordered : bool
            Indicates whether the resulting Categorical will be an 'ordered' Categorical
            (sometimes called an 'Ordinal').
        destroy : bool
            This parameter is unused.

        Returns
        -------
        Grouping
        """
        # need to vet base index, enum mode, number of columns, etc.
        if not _trusted:
            warnings.warn(f'still implementing grouping hstack validation')

            # TODO: add more tests for single vs. multikey (without unnecessary calculation of uniquedict)
            for grp in glist:
                same_mode = set()
                if not isinstance(grp, Grouping):
                    raise TypeError(f"Grouping hstack is for categoricals, not {type(grp)}")
                same_mode.add(grp.isenum)

            if len(same_mode) != 1:
                raise TypeError(f"Grouping hstack received a mix of different modes.")

        firstgroup = glist[0]
        sort_display = firstgroup.isdisplaysorted

        # mapping
        if firstgroup.isenum:
            # stack underlying arrays from all categoricals (held in grouping's grouping_dict)
            underlying = hstack([[*g._grouping_dict.values()][0] for g in glist])
            # stack all unique string arrays
            listnames = hstack([g._enum.category_array for g in glist])

            # collect, measure, stack integer arrays
            listcodes = [g._enum.code_array for g in glist]
            cutoffs = [TypeRegister.FastArray([len(c) for c in listcodes], dtype=np.int64).cumsum()]
            listcodes = hstack(listcodes)

            # send in as two arrays
            listcats = [listcodes, listnames]

            # will return new unique arrays for codes, names
            underlying, newcats = merge_cats(underlying, listcats, unique_cutoffs=cutoffs, from_mapping=True,
                                             ordered=ordered)
            newgroup = Grouping(underlying, categories=dict(zip(newcats[1], newcats[0])), _trusted=True)
        else:
            # use catinstance for base 0 or 1
            listidx = [g.catinstance for g in glist]
            cat_tuples = [tuple(g.uniquedict.values()) for g in glist]
            listcats = [[v[i] for v in cat_tuples] for i in range(len(cat_tuples[0]))]
            underlying, newcats = merge_cats(listidx, listcats, base_index=base_index, ordered=ordered)
            newgroup = Grouping(underlying, categories=newcats, categorical=True, _trusted=True,
                                             base_index=base_index, ordered=ordered, sort_display=sort_display)

        return newgroup

    @staticmethod
    def take_groups(
        grouped_data: np.ndarray,
        indices: np.ndarray,
        ncountgroup: np.ndarray,
        ifirstgroup: np.ndarray
    ) -> np.ndarray:
        """
        Take groups of elements from an array.

        This function provides fancy-indexing over groups of data -- so a fancy index can be used to
        specify _groups_ of data, rather than just individual elements, and the grouped elements will
        be copied to the output.

        Parameters
        ----------
        grouped_data : np.ndarray
        indices : np.ndarray of int
        ncountgroup : np.ndarray of int
        ifirstgroup : np.ndarray of int

        Returns
        -------
        np.ndarray

        Raises
        ------
        ValueError
            When `ncountgroup` and `ifirstgroup` have different shapes.

        See Also
        --------
        numpy.take

        Examples
        --------
        Select data from an array, where the elements belong to the 2nd, 4th, and 6th groups within the Grouping object.

        >>> key_data = rt.FA([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6])
        >>> data = rt.arange(len(key_data))
        >>> g = rt.Grouping(key_data)
        >>> group_indices = rt.FA([2, 4, 6])
        >>> Grouping.take_groups(data, group_indices, g.ncountgroup, g.ifirstgroup)
        FastArray([1, 2, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20])
        """
        # TODO: Input validation
        if ifirstgroup.shape != ncountgroup.shape:
            raise ValueError("The shape of 'ifirstgroup' must match the shape of 'ncountgroup'.")

        # Determine the size of the output array
        output_length = ncountgroup[indices].nansum(dtype=np.int64)

        # Create the output array.
        result = empty(output_length, dtype=grouped_data.dtype)

        @nb.njit(cache=True)
        def impl(grouped_data, indices, ncountgroup, ifirstgroup, out):
            # TODO: If we ever want to use nb.prange() in the loop below, we'll need to
            #       do something like a partial cumsum() on ncountgroup[indices] to determine
            #       how to segment the output space so it can be written to in parallel.
            #       The way 'out_idx' is incremented below won't work in a parallelized loop.
            #       One idea is to have a boolean parameter to indicate that ncountgroup is already
            #       in cumulative form; we need to do a little extra work to deal with that above,
            #       but then we'd be able to more easily parallelize here.
            out_idx = 0
            for i in range(len(indices)):
                curr_group = indices[i]
                group_length = ncountgroup[curr_group]

                # Copy the elements of the current group to the output.
                group_start_idx = ifirstgroup[curr_group]
                out[out_idx:out_idx + group_length] = grouped_data[group_start_idx:group_start_idx+group_length]

                # Advance the current index within the output array.
                out_idx += group_length

        # Call the numba implementation of the function to build the result.
        impl(grouped_data, indices, ncountgroup, ifirstgroup, result)
        return result

    @staticmethod
    def extract_groups(
        condition: np.ndarray,
        grouped_data: np.ndarray,
        ncountgroup: np.ndarray,
        ifirstgroup: np.ndarray
    ) -> np.ndarray:
        """
        Take groups of elements from an array, where the groups are selected by a boolean mask.

        This function provides boolean-indexing over groups of data -- so a boolean mask can be used to
        select _groups_ of data, rather than just individual elements, and the grouped elements will
        be copied to the output.

        Parameters
        ----------
        condition : np.ndarray of bool
            An array whose nonzero or True entries indicate the groups in `ncountgroup` whose
            elements will be extracted from `grouped_data`.
        grouped_data : np.ndarray
        ncountgroup : np.ndarray of int
        ifirstgroup : np.ndarray of int

        Returns
        -------
        np.ndarray

        Raises
        ------
        ValueError
            When `condition` is not a boolean/logical array.
            When `condition` and `ncountgroup` have different shapes.
            When `ncountgroup` and `ifirstgroup` have different shapes.

        See Also
        --------
        numpy.extract

        Examples
        --------
        Select data from an array, where the elements belong to even-numbered groups within the Grouping object.

        >>> key_data = rt.FA([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6])
        >>> data = rt.arange(len(key_data))
        >>> g = rt.Grouping(key_data)
        >>> group_mask = rt.arange(len(g.ncountgroup)) % 2 == 0
        >>> Grouping.extract_groups(group_mask, data, g.ncountgroup, g.ifirstgroup)
        FastArray([1, 2, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20])
        """
        if condition.dtype.char != '?':
            # TODO: This condition can be relaxed for compatibility with np.extract -- we just need to make a few
            #       compatibility modifications below (e.g. to use np.extract or rt.extract to get the elements to
            #       pass to the nansum() below).
            raise ValueError("'condition' must be a boolean/logical array.")
        elif condition.shape != ncountgroup.shape:
            raise ValueError("The shapes of the 'condition' and 'ncountgroup' arrays must be the same.")
        elif ifirstgroup.shape != ncountgroup.shape:
            raise ValueError("The shape of 'ifirstgroup' must match the shape of 'ncountgroup'.")

        # Determine the size of the output array
        output_length = ncountgroup[condition].nansum(dtype=np.int64)

        # Create the output array.
        result = empty(output_length, dtype=grouped_data.dtype)

        @nb.njit(cache=True)
        def impl(grouped_data, condition, ncountgroup, ifirstgroup, out):
            # TODO: If we ever want to use nb.prange() in the loop below, we'll need to
            #       do something like a partial cumsum() on ncountgroup[condition] to determine
            #       how to segment the output space so it can be written to in parallel.
            #       The way 'out_idx' is incremented below won't work in a parallelized loop.
            #       One idea is to have a boolean parameter to indicate that ncountgroup is already
            #       in cumulative form; we need to do a little extra work to deal with that above,
            #       but then we'd be able to more easily parallelize here.
            out_idx = 0
            for curr_group in range(len(condition)):
                if condition[curr_group]:
                    group_length = ncountgroup[curr_group]

                    # Copy the elements of the current group to the output.
                    group_start_idx = ifirstgroup[curr_group]
                    out[out_idx:out_idx + group_length] = grouped_data[group_start_idx:group_start_idx + group_length]

                    # Advance the current index within the output array.
                    out_idx += group_length

        # Call the numba implementation of the function to build the result.
        impl(grouped_data, condition, ncountgroup, ifirstgroup, result)
        return result


class GroupingEnum:
    """Holds enum mapping for grouping object's integer codes.
    Used to translate unique codes into strings for groupby keys.

    Attributes
    ----------
    code_array : `FastArray`
        Array of unique integer codes.
    category_array : `FastArray`
        Array of unique category strings (unicode).
    unique_count : int
        Number of mappings in enum.

    Methods
    -------
    from_category : int
        Integer code from category string.
    from_code : string
        Category string from integer code, or `!<code>` if no mapping exists.

    """
    def __init__(self, mapping=None, _trusted=False):
        if mapping is None:
            self._str_to_int_dict = {}
            self._int_to_str_dict = {}
            return

        if _trusted:
            mapping_reverse = {v:k for k,v in mapping.items()}
            if len(mapping)!=0:
                # mapping is trusted, one-to-one, but guarantee dicts set correctly
                if isinstance([*mapping][0], str):
                    str2int = mapping
                    int2str = mapping_reverse
                else:
                    str2int = mapping_reverse
                    int2str = mapping
                self._int_to_str_dict = int2str
                self._str_to_int_dict = str2int
            else:
                self._str_to_int_dict, self._int_to_str_dict = {}, {}
        else:
            # ensure correct types for mappings (int -> str or str -> int)
            # warn if mappings are not 1-to-1
            if isinstance(mapping, dict):
                dict_builder = self._build_dicts_python
            # enums are more restrictive, less validation neeeded
            elif isinstance(mapping, EnumMeta):
                dict_builder = self._build_dicts_enum
            else:
                raise TypeError(f'Cannot initialize GroupingEnum with item of type {type(mapping)}.')

            self._str_to_int_dict, self._int_to_str_dict = dict_builder(mapping)

    # ------------------------------------------------------------
    def copy(self):
        new_enum = GroupingEnum()
        new_enum._str_to_int_dict = self._str_to_int_dict.copy()
        new_enum._int_to_str_dict = self._int_to_str_dict.copy()
        return new_enum

    # ------------------------------------------------------------
    def _build_dicts_python(self, python_dict):
        '''
        Categoricals can be initialized with a dictionary of string to integer or integer to string.
        Python dictionaries accept multiple types for their keys, so the dictionaries need to check types as they're being constructed.
        '''
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
    def _build_dicts_enum(self, enum):
        '''
        Builds forward/backward dictionaries from IntEnums. If there are multiple identifiers with the same, WARN!
        '''
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
    def from_category(self, category):
        """Translate a single string or bytestring to corresponding integer code.
        """
        if isinstance(category, bytes):
            category = category.decode()
        if isinstance(category, str):
            return self._str_to_int_dict[category]
        raise TypeError(f'category must be string or bytes, got {type(category)}')

    # ------------------------------------------------------------
    def from_code(self, code):
        """Translate a single integer code to corresponding category string.
        If code doesn't exist, return `!<code>`
        """
        if isinstance(code, (int, np.integer)):
            return self._from_code(code)
        raise TypeError(f'code must be integer, got {type(code)}')
    # ------------------------------------------------------------
    def _from_code(self, code):
        """Internal from_code without type check.
        """
        return self._int_to_str_dict.get(code,'!<'+str(code)+'>')
    # ------------------------------------------------------------
    def unique_categories(self, codes=None):
        """Category strings from codesm, may generate invalid strings for invalid codes.
        """
        if codes is None:
            return self.category_array
        # NOTE: this is a slow python loop
        return TypeRegister.FastArray([self.from_code(c) for c in codes], unicode=True)

    # ------------------------------------------------------------
    @property
    def code_array(self):
        return TypeRegister.FastArray(list(self._int_to_str_dict))
    @property
    def category_array(self):
        return TypeRegister.FastArray(list(self._str_to_int_dict), unicode=True)
    @property
    def unique_count(self):
        return len(self._int_to_str_dict)

    # ------------------------------------------------------------
    def _build_string(self):
        def _pairstring(k,v):
            return str(k)+':'+v.__repr__()

        # build string for mapping dictionary (abbreviate, and stop newlines)
        _maxlen = 6
        numitems = len(self._int_to_str_dict)
        if numitems <= _maxlen:
            reprdict = self._int_to_str_dict
            items = [_pairstring(k,v) for k,v in zip(self._int_to_str_dict, self._str_to_int_dict)]

        else:
            _slicesize = int(np.floor(_maxlen/2))
            codes = list(self._int_to_str_dict)
            strs = list(self._str_to_int_dict)
            keys = codes[:_slicesize] + ['...'] + codes[-_slicesize:]
            vals = strs[:_slicesize] + ['...'] + strs[-_slicesize:]

            itemsleft = [_pairstring(k,v) for k,v in zip(codes[:_slicesize], vals[:_slicesize])]
            breakstr = ['...']
            itemsright = [_pairstring(k,v) for k,v in zip(codes[-_slicesize:], vals[-_slicesize:])]
            items = itemsleft+breakstr+itemsright

        return "".join(['{', ", ".join(items), '}'])
    def __str__(self):
        return self._build_string()
    def __repr__(self):
        return self._build_string()
    # ------------------------------------------------------------


def hstack_test(arr_list):
    hashes =[groupbyhash(a) for a in arr_list]
    indices = [h['iKey'] for h in hashes]
    uq = [a[h['iFirstKey']] for h,a in zip(hashes, arr_list)]
    return hstack_groupings(indices, uq)

# keep this as the last line
from .rt_enum import TypeRegister
TypeRegister.Grouping = Grouping
