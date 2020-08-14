import numpy as np
import copy
import warnings

from .rt_numpy import mask_andi, lexsort, hstack
from .rt_enum import NumpyCharTypes, TypeRegister, INVALID_DICT, FILTERED_LONG_NAME


class GroupByKeys():
    """Handles masking, appending invalid, and sorting of key columns for a groupby operation.

    Parameters
    ----------
    grouping_dict : dict
        Non-unique or unique key columns.
    ifirstkey : array, optional
        If set, first occurence to generate unique values from non-unique keys.
        Sometimes these are lazily evaluated and do not correspond to the grouping dict held.
        See the ``prebinned`` keyword.
    isortrows : array, optional
        A sorted index for the unique array.
        May be calculated later on, after ``grouping_dict`` is reduced to unique values.
    sort_display : bool, default False
        If True, unique keys in result of operation will be sorted. Otherwise, will appear unsorted.
    pre_sorted : bool, default False
        Unique ``grouping_dict`` is already in a sorted order, do not apply / calculate ``isortrows``,
        even if sort on is True.
    prebinned : bool, default False
        If True, ``grouping_dict`` contains unique values. ``ifirstkey`` will not be stored.
        If False, ``grouping_dict contains non-unique values, the default if constructed by a GroupBy object.
    
    Constructor
    -----------
    GroupByKeys has two main ways of initialization:

    1. From a non-unique grouping_dict and ifirstkey (fancy index to unique values). The object 
        will hold on to both, and lazily generate the groupby keys as necessary.
    2. From already binned gbkeys (unique values). Most categoricals will initialize GroupByKeys this way.
       Because Categoricals are sometimes naturally sorted, they may set the pre_sorted keyword to True.

    If sort_display is True and the keys are not already sorted, the gbkeys will be sorted in-place the first time 
    a groupby calculation is made. After being sorted, the internal _sort_applied flag will be set. Despite the keys 
    being sorted, the sort might still need to be applied to the data columns of the groupby calculation's result.

    Lazy Evaluation
    ---------------
    - If isortrows is not provided and the gbkeys are not pre-sorted, a lexsort will be performed, and the keys will be sorted inplace.
    - If gbkeys are requested with a filter bin, a new bin will be permanently prepended to each key array. After the filtered bin is added,
    the gbkeys will still default to return arrays without the filter (a view of the held arrays offset by 1).
    - Multikey labels are a list of strings of the tuples that will appear when a multikey is displayed. They will also add a filtered tuple 
    as necessary, and default to a reduced view after the addition - just like the gbkeys. Multikey labels will not be generated until 
    they are requested for display (because they are constructed in a python loop, generating these is expensive).
    """

    def __init__(self, grouping_dict, ifirstkey=None, isortrows=None, sort_display=False, pre_sorted=False, prebinned=False):
        self._grouping_dict = grouping_dict
        self._gbkeys = None

        # arrays in the dict are already unique
        if prebinned:
            # iFirstKey is only relevant to non-unique arrays, discard it
            # will still be held in the Grouping object
            self._ifirstkey = None
            self._gbkeys = grouping_dict.copy()
            self._unique_count = len(list(grouping_dict.values())[0])

        else:
            self._ifirstkey = ifirstkey
            self._unique_count = len(ifirstkey)

        self._isortrows = isortrows
        self._sort_display = sort_display
        self._pre_sorted = pre_sorted

        self._sort_applied = False          # sort has been applied to the key columns
        self._filter_applied = False        # filter bin has been added to the key columns
        self._filter_applied_labels = False # filter has been added to the multikey labels
        self._multikey_labels = None        # (lazily generated)

    @property
    # ------------------------------------------------------------
    def sort_gb_data(self):
        '''
        If a sort has been applied to the gbkeys, they do not need to be sorted, however 
        the data resulting from a groupby calculation is naturally unsorted and will still need 
        a sort applied.
        '''
        return self._sort_display and self._pre_sorted is False

    # ------------------------------------------------------------
    @property
    def isortrows(self):
        '''
        Generates isortrows (index to sort groupby keys). Possibly performs a lexsort.
        '''
        if self._isortrows is None:
            self._isortrows = self._make_isortrows()
        return self._isortrows

    # ------------------------------------------------------------
    @property
    def gbkeys(self):
        '''
        Generates groupby keys if necessary. Returns groupby keys.
        '''
        return self.keys()

    # ------------------------------------------------------------
    @property
    def gbkeys_filtered(self):
        '''
        Adds a filter to the gbkeys, or returns the already filtered gbkeys.
        '''
        self._insert_filter_bin()
        return self.keys(showfilter=True)

    # ------------------------------------------------------------
    def _insert_filter_bin(self):
        if self._filter_applied is False:
            # use the property for the iterator, so they can be generated if necessary
            for k,v in self.gbkeys.items():
                filtername = self._get_filter_bin_name(v)
                self._gbkeys[k] = hstack((filtername, v))
            self._filter_applied = True

    # ------------------------------------------------------------
    def unique_unsorted(self):
        """Pull the unique keys unsorted, using iFirstKey or the prebinned uniques.
        """
        if self._gbkeys is not None:
            return self._gbkeys
        else:
            return self._pull_from_ifirstkey()

    # ------------------------------------------------------------
    def _pull_from_ifirstkey(self):
        if self._gbkeys is None:
            if self._ifirstkey is not None:
                unique_keys = {}
                for k, v in self._grouping_dict.items():
                    unique_keys[k] = v[self._ifirstkey]
            else:
                raise ValueError(f"Groupby keys need to be generated with an iFirstKey array, which the groupby key object does not have.")
        else:
            unique_keys = self._gbkeys
        return unique_keys

    # ------------------------------------------------------------
    def keys(self, sort=None, showfilter=False):
        """Return unique keys, possibly apply a sort and add a filter bin.
        """
        # groupby keys need to be generated from the non-unique grouping dict
        self._gbkeys = self._pull_from_ifirstkey()

        if sort is not None:
            self._sort_display = sort

        keys = self._gbkeys
        # only apply sort if requested and not already sorted
        if self._sort_display and self._pre_sorted is False:
            if self._sort_applied is False:
                if self._isortrows is None:
                    self._isortrows = self._make_isortrows()
                keys = {}
                for k,v in self._gbkeys.items():
                    keys[k] = v[self._isortrows]
                self._gbkeys = keys
                self._sort_applied = True

        # hide the filter bin by slicing the array if filter was turned off after adding
        if showfilter is False and self._filter_applied is True:
            keys = self._trim_keys(keys)

        return keys

    # ------------------------------------------------------------
    @property
    def multikey_labels(self):
        return self.labels()

    # ------------------------------------------------------------
    @property
    def multikey_labels_filtered(self):
        return self.labels(showfilter=True)

    # ------------------------------------------------------------
    def _insert_filter_label(self):
        labels = self.multikey_labels
        if self._filter_applied_labels is False:
            label = (FILTERED_LONG_NAME,) * len(self.gbkeys)
            self._multikey_labels.insert(0,label)
            self._filter_applied_labels = True

    # ------------------------------------------------------------
    def labels(self, showfilter=False):
        '''
        Generates list of tuples from multikey columns.
        '''
        if self._multikey_labels is None:
            self._multikey_labels = []
            for i in range(self.unique_count):
                key = self.get_bin_from_index(i)
                key = str(key)
                key.replace("'","")
                self._multikey_labels.append(key)

        if self._filter_applied_labels and showfilter is False:
            return self._trim_keys(self._multikey_labels)
        return self._multikey_labels

    # ------------------------------------------------------------
    def _trim_keys(self, keys):
        '''
        Return a trimmed view of the keys so the filtered bin is not included.
        Also trims list of multikey labels
        '''

        if isinstance(keys, dict):
            trimmed = {}
            for k,v in keys.items():
                trimmed[k]=v[1:]
        else:
            trimmed = keys[1:]
        return trimmed

    # ------------------------------------------------------------
    def _get_filter_bin_name(self, arr):
        if arr.dtype.char == 'U':
            filtername = FILTERED_LONG_NAME
        elif arr.dtype.char == 'S':
            filtername = FILTERED_LONG_NAME.encode()
        else:
            # subclasses will use correct invalid and class will be preserved in hstack
            filtername = arr.fill_invalid(shape=1, inplace=False)
        return filtername
    
    # ------------------------------------------------------------
    @property
    def unique_count(self):
        '''
        Returns number of unique groupby keys - lazily evaluated and stored.
        '''
        #print('unique count hit!')
        if self._unique_count is None:
            first_key = list(self.gbkeys.values())[0]
            self._unique_count = len(first_key)
        return self._unique_count

    # ------------------------------------------------------------
    def _get_index_from_tuple(self, tup):
        '''
        If the GroupByKeys object is holding a multikey dictionary, it can be indexed by 
        a tuple. This internal routine (called by get_index_from_bin/__getitem__) will return
        the bin index of matching multikey entries or -1 if not found. Any string/bytes values 
        will be fixed to match the string/bytes column.
        '''
        if len(tup) == len(self.gbkeys):
            # build a boolean mask for each item in tuple
            match = []
            dictlist = list(self.gbkeys.values())
            for colnum, item in enumerate(tup):
                # match string types if necessary
                if isinstance(item, bytes):
                    if dictlist[colnum].dtype.char == 'U':
                        item = bytes.decode(item)
                elif isinstance(item, str):
                    if dictlist[colnum].dtype.char == 'S':
                        item = item.encode()
                match.append(dictlist[colnum] == item)
            match = mask_andi(match)
            match = np.where(match)[0]
            if len(match) == 1:
                return match[0]
            else:
                return -1
        else:
            raise ValueError(f"Tuple must contain an item for each groupby key column.")

    # ------------------------------------------------------------
    def get_index_from_bin(self, bin):
        '''
        :param bin: a tuple of multiple keys or a single key (will be converted to tuple)

        :return index: the bin index, or -1 if not found.
        '''
        if not isinstance(bin, tuple):
            bin = (bin,)
        return self._get_index_from_tuple(bin)

    # ------------------------------------------------------------
    def get_bin_from_index(self, index):
        '''
        :param index: int or list of integers
        :return result_bins: matching bins for provided indices or an empty list
        '''
        if index >= 0 and index < self.unique_count:
            result_bins = []
            for column in self.gbkeys.values():
                bin = column[index]
                if isinstance(bin, bytes):
                    bin = bytes.decode(bin)
                result_bins.append(bin)
            if len(result_bins) == 1:
                return result_bins[0]
            else:
                return tuple(result_bins)
        else:
            raise ValueError(f"Bin index {index} was out of range for gbkeys of length {self.unique_count}")

    # ------------------------------------------------------------
    def get_bin(self, index):
        '''
        :param index: int or list of integers
        :return result_bins: matching bins for provided indices or an empty list
        '''
        if isinstance(index, (int, np.integer)):
            result = self.get_bin_from_index(index)
            
        elif isinstance(index, list):
            if isinstance(index[0], (int, np.integer)):
                result = []
                for i in index:
                    result.append(self.get_bin_from_index(i))
                if len(result) == 1:
                    result = result[0]
            else:
                raise TypeError(f"Bins can only be retrieved by lists of integers, not {type(index[0])}")

        elif isinstance(index, np.ndarray):
            if index.dtype.char in NumpyCharTypes.AllInteger:
                result = []
                for i in index:
                    result.append(self.get_bin_from_index(i))
                if len(result) == 1:
                    result = result[0]
            else:
                raise TypeError(f"Bins can only be retrieved by numpy arrays of integer type, not {index.dtype}")

        else:
            raise TypeError(f"Bins must be retrieved by integer, not {type(index)}")

        return result

    # ------------------------------------------------------------
    def __getitem__(self, index):
        return self.get_bin(index)

    # ------------------------------------------------------------
    def _make_isortrows(self):
        sortlist = list(self._gbkeys.values())
        sortlist.reverse()
        isortrows = lexsort(sortlist)
        return isortrows

    # ------------------------------------------------------------
    def unsort(self):
        '''
        Sets the internal _sort_display flag to False. Will warn the user if the groupby keys are already sorted or were pre-sorted 
        when GroupByKeys were constructed.
        '''
        if self._pre_sorted:
            warnings.warn("Groupby keys were naturally sorted.")
        elif self._sort_applied:
            raise ValueError("Groupby keys were already sorted. Data mis-match will occur.")
        self._sort_display = False

    # ------------------------------------------------------------
    def copy(self, deep=False):
        '''
        Creates a deep or shallow copy of the grouping 
        '''
        if deep:
            new_keychain = GroupByKeys(copy.deepcopy(self._grouping_dict), ifirstkey=self._ifirstkey)
            if self._isortrows is not None:
                new_keychain._isortrows = self._isortrows.copy()

        else:
            new_keychain = GroupByKeys(self._grouping_dict.copy(), ifirstkey=self._ifirstkey)
            if self._isortrows is not None:
                new_keychain._isortrows = self._isortrows

        new_keychain._sort_display = self._sort_display
        new_keychain._pre_sorted = self._pre_sorted
        new_keychain._sort_applied = self._sort_applied
        new_keychain._filter_applied = self._filter_applied
        new_keychain._filter_applied_labels = self._filter_applied_labels
        new_keychain._unique_count = self._unique_count
        new_keychain._multikey_labels = self._multikey_labels

        return new_keychain
    # ------------------------------------------------------------
    def __repr__(self):
        return self._build_string()
    # ------------------------------------------------------------
    def __str__(self):
        return self._build_string()
    # ------------------------------------------------------------
    def _build_string(self):
        summary_str = []
        summary_str.append("gbkeys: "+str(self._gbkeys))
        summary_str.append("isortrows: "+str(self._isortrows))
        summary_str.append("sort on: "+str(self._sort_display))
        summary_str.append("naturally sorted: "+str(self._pre_sorted))
        summary_str.append("sort applied: "+str(self._sort_applied))
        summary_str.append("filter applied: "+str(self._filter_applied))
        return "\n".join(summary_str)
    
    # ------------------------------------------------------------
    @property
    def multikey(self):
        '''
        Returns True if GroupByKeys object is holding multiple columns in _gbkeys
        '''
        if self._gbkeys is not None:
            return len(self._gbkeys) > 1
        return False

    # ------------------------------------------------------------
    @property
    def singlekey(self):
        '''
        Returns True if GroupByKeys object is holding a single column in _gbkeys
        '''
        if self._gbkeys is not None:
            return len(self._gbkeys) == 1
        return False
