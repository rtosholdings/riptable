__all__ = ['GroupBy']

import numpy as np
import warnings

from .rt_numpy import  _groupbycalculateall, groupbypack, sortinplaceindirect,_groupbycalculateallpack,empty_like
from .rt_fastarraynumba import fill_forward, fill_backward
from .rt_utils import mbget, bool_to_fancy
from .rt_timers import GetTSC
from .rt_enum import GB_FUNCTIONS, GB_STRING_ALLOWED, GB_FUNC_COUNT, CategoryMode
from .rt_groupbykeys import GroupByKeys
import riptide_cpp as rc


#'''
#Series.abs()	Return an object with absolute value taken�only applicable to objects that are all numeric.
#Series.all([axis, bool_only, skipna, level])	Return whether all elements are True over requested axis
#Series.any([axis, bool_only, skipna, level])	Return whether any element is True over requested axis
#Series.autocorr([lag])	Lag-N autocorrelation
#Series.between(left, right[, inclusive])	Return boolean Series equivalent to left <= series <= right.
#Series.clip([lower, upper, axis, inplace])	Trim values at input threshold(s).
#Series.clip_lower(threshold[, axis, inplace])	Return copy of the input with values below given value(s) truncated.
#Series.clip_upper(threshold[, axis, inplace])	Return copy of input with values above given value(s) truncated.
#Series.corr(other[, method, min_periods])	Compute correlation with other Series, excluding missing values
#Series.count([level])	Return number of non-NA/null observations in the Series
#Series.cov(other[, min_periods])	Compute covariance with Series, excluding missing values
#Series.cummax([axis, skipna])	Return cumulative max over requested axis.
#Series.cummin([axis, skipna])	Return cumulative minimum over requested axis.
#Series.cumprod([axis, skipna])	Return cumulative product over requested axis.
#Series.cumsum([axis, skipna])	Return cumulative sum over requested axis.
#Series.describe([percentiles, include, exclude])	Generates descriptive statistics that summarize the central tendency, dispersion and shape of a datasets distribution, excluding NaN values.
#Series.diff([periods])	1st discrete difference of object
#Series.factorize([sort, na_sentinel])	Encode the object as an enumerated type or categorical variable
#Series.kurt([axis, skipna, level, numeric_only])	Return unbiased kurtosis over requested axis using Fisher�s definition of kurtosis (kurtosis of normal == 0.0).
#Series.mad([axis, skipna, level])	Return the mean absolute deviation of the values for the requested axis
#Series.max([axis, skipna, level, numeric_only])	This method returns the maximum of the values in the object.
#Series.mean([axis, skipna, level, numeric_only])	Return the mean of the values for the requested axis
#Series.median([axis, skipna, level, ...])	Return the median of the values for the requested axis
#Series.min([axis, skipna, level, numeric_only])	This method returns the minimum of the values in the object.
#Series.mode()	Return the mode(s) of the dataset.
#Series.nlargest([n, keep])	Return the largest n elements.
#Series.nsmallest([n, keep])	Return the smallest n elements.
#Series.pct_change([periods, fill_method, ...])	Percent change over given number of periods.
#Series.prod([axis, skipna, level, ...])	Return the product of the values for the requested axis
#Series.quantile([q, interpolation])	Return value at the given quantile, a la numpy.percentile.
#Series.rank([axis, method, numeric_only, ...])	Compute numerical data ranks (1 through n) along axis.
#Series.sem([axis, skipna, level, ddof, ...])	Return unbiased standard error of the mean over requested axis.
#Series.skew([axis, skipna, level, numeric_only])	Return unbiased skew over requested axis
#Series.std([axis, skipna, level, ddof, ...])	Return sample standard deviation over requested axis.
#Series.sum([axis, skipna, level, ...])	Return the sum of the values for the requested axis
#Series.var([axis, skipna, level, ddof, ...])	Return unbiased variance over requested axis.
#Series.unique()	Return unique values in the object.
#Series.nunique([dropna])	Return number of unique elements in the object.
#Series.is_unique	Return boolean if values in the object are unique
#Series.is_monotonic	Return boolean if values in the object are
#Series.is_monotonic_increasing	Return boolean if values in the object are
#Series.is_monotonic_decreasing	Return boolean if values in the object are
#Series.value_counts([normalize, sort, ...])	Returns object containing counts of unique values.

from enum import IntEnum

from .rt_groupbyops import GroupByOps


#=====================================================================================================
#=====================================================================================================
class GroupBy(GroupByOps):
    """
    Parameters
    ----------
    dataset: Dataset
        The dataset object

    keys: list. List of column names to groupby

    filter: None. Boolean mask array applied as filter before grouping

    return_all: bool.  Default to False. When set to True will return all 
          the dataset columns for every operation.

    hint_size: int.  Hint size for the hash (optional)

    sort_display: bool. Default to True.  Indicates 

    lex: bool
        Defaults to False. When True uses a lexsort to find the groups (otherwise uses a hash).
    
    totals: bool

    Notes
    -----
    None at this time.

    Properties
    ----------
    gbkeys:  dictionary of numpy arrays binned from
    isortrows: sorted index or None

    """
    DebugMode=False
    TestCatGb = True

    def __init__(self, dataset, 
                 keys:list=None, 
                 filter=None, 
                 ordered=None,
                 sort_display=None, 
                 return_all = False, 
                 hint_size=0, 
                 lex:bool=None,
                 rec:bool=False,
                 totals= False,
                 copy=False, 
                 cutoffs=None, 
                 verbose=False,
                 **kwargs):

        # upon creation create a unique id to track sorting
        self._uniqueid = GetTSC()

        # short circuit in copy mode
        if copy:
            return

        # copy over sort list
        self._col_sortlist = dataset._col_sortlist

        grouping_dict = None

        if not isinstance(keys, list):
            # if just one str passed in, wrap it in a list
            if isinstance(keys, str):
                keys=[keys]
            else:
                if isinstance(keys, np.ndarray):
                    if len(keys) != dataset.shape[0]:
                        raise TypeError(f'The argument "keys" passed in GroupBy must be the same length as the dataset {len(keys)} vs {dataset.shape[0]}.')
                    else:
                        name='col_0'
                        try:
                            name = keys.get_name()
                        except:
                            pass
                        # make into a dict
                        keys={name: keys}
                        grouping_dict = keys
                else:
                    raise TypeError(f'The argument "keys" passed in GroupBy init must be a list or str.  It must contain column names.')

        self._sortedbykey=False
        self._isortrows = None
        #self._gbkeys = None
        self._gb_keychain = None

        self._return_all = return_all
        self._filter = filter
        self._pcutoffs = cutoffs
        self._totals = totals

        if grouping_dict is None:
            grouping_dict = {k:dataset[k] for k in keys}

        if ordered is not None and sort_display is not None:
            raise TypeError(f'The kwarg ordered and sort_display cannot be used in combination')

        if ordered is not None:
            sort_display = ordered

        if sort_display is None:
            # automatically sort display if nothing passed
            if ordered is None and lex is None:
                sort_display = True
                try:
                    # if the first key has a grouping object and it does not want the display sorted and is itself not sorted,
                    # then we do not sort the display
                    # this happens after a 'cut' where we want to preserve label order
                    for v in grouping_dict.values():
                        if not v.grouping.isdisplaysorted and not v.grouping.isordered:
                            sort_display = False;
                        break;
                except:
                    pass               

        self.grouping = TypeRegister.Grouping(
                            grouping_dict, hint_size=hint_size, filter=filter, 
                            sort_display=sort_display, lex=lex, rec=rec, 
                            cutoffs=cutoffs, verbose=verbose)

        # always use the names of the keys in the grouping dict
        # grouping dict should only live in grouping
        gvals = list(self.grouping._anydict.values())
        if len(keys) == len(gvals):
            self._grouping_dict = {k: gvals[i] for i, k in enumerate(keys)}
        # grouping dict may be larger than keys (multikey categorical splits into multiple arrays)
        else:
            self._grouping_dict = self.grouping._anydict

        # todo: remember change count on dataset
        self._dataset = dataset
        
        self._sort_display=sort_display

    #---------------------------------------------------------------
    def copy(self, deep = True):
        '''
        Called from getitem when user follows gb with []
        '''
        newgb = type(self)(None, copy=True)
        newgb._col_sortlist = self._col_sortlist 
        newgb._sortedbykey = self._sortedbykey
        try:
            newgb._gb_keychain = self._gb_keychain.copy(deep=deep)
        except:
            newgb._gb_keychain = None

        newgb._grouping_dict = self._grouping_dict.copy()
        newgb.grouping = self.grouping.copy(deep=deep)
        if deep:
            newgb._dataset = self._dataset.copy() 
            if self._filter is None:
                newgb._filter = self._filter
            else:
                newgb._filter = self._filter.copy()

            #newgb.grouping = self.grouping.copy(grouping_dict=newgb._grouping_dict)

        else:
            newgb._dataset = self._dataset
            newgb._filter = self._filter

            # reference vs copy
            #newgb.grouping = self.grouping

        newgb._return_all = self._return_all
        newgb._pcutoffs = self._pcutoffs

        newgb._sort_display=  self._sort_display
        #newgb._gbkeys = self._gbkeys
        newgb._isortrows = self._isortrows
        newgb._totals = self._totals

        return newgb

    #---------------------------------------------------------------
    @property
    def gbkeys(self):
        return self.gb_keychain.gbkeys

    #---------------------------------------------------------------
    @property
    def isortrows(self):
        return self.gb_keychain.isortrows

    #---------------------------------------------------------------
    @property
    def ifirstkey(self):
        return self.grouping.ifirstkey

    #---------------------------------------------------------------
    @property
    def ilastkey(self):
        return self.grouping.ilastkey

    #---------------------------------------------------------------
    @property
    def gb_keychain(self):
        if self._gb_keychain is None:
            # this will change when groupby keys are handled only by the Grouping object
            #self._gb_keychain = GroupByKeys(self.grouping._anydict, self.ifirstkey, sort_display=self._sort_display, prebinned=self.grouping.iscategorical)
            # always send in the uniques
            self._gb_keychain = GroupByKeys(self.grouping.uniquedict, None, sort_display=self._sort_display, prebinned=True)
        return self._gb_keychain

    #---------------------------------------------------------------
    def __iter__(self):
        '''
        Generates tuples of key, value pairs.
        Keys are key values for single key, or tuples of key values for multikey.
        Values are datasets containing all rows from data in group for that key.
        '''
        return self._iter_internal_contiguous()

    #---------------------------------------------------------------
    def _pop_gb_data(self, calledfrom, userfunc, *args, **kwargs):
        """
        GroupBy holds on to its dataset. There may be no additional data provided.
        """
        kwargs.setdefault('dataset', self._dataset)
        return super()._pop_gb_data(calledfrom, userfunc, *args, **kwargs)

    #---------------------------------------------------------------
    def add_totals(self, gb_ds):
        total = {}
        labels = gb_ds.label_get_names()
        for colname, arr in gb_ds.items():
            if colname not in labels:
                if arr.iscomputable():
                    total[colname]=arr.sum()
                else:
                    # leave blank
                    total[colname]=''
                
        gb_ds.footer_set_values('Total', total )
        return gb_ds

    #---------------------------------------------------------------
    # OVERRIDEN from groupbyops
    def _calculate_all(self, funcNum, *args, func_param=0, **kwargs):
        '''
        Generate a GroupByKeys object if necessary and ask for the result of a calculation from the grouping object.
        Returns: a grouped by dataset with the result from the calculation
        '''
        keychain = self.gb_keychain
        kwargs.setdefault('dataset', self._dataset)
        return_all = kwargs.pop('return_all', self._return_all)

        origdict, user_args, tups = self._prepare_gb_data('GroupBy', funcNum, *args, **kwargs)
        gb_ds = self.grouping._calculate_all(
                    origdict, funcNum, func_param=func_param,
                    return_all=return_all, keychain=keychain, user_args=user_args, tups=tups, **kwargs)

        # check for accum1 like functionality
        if self._totals:
            return self.add_totals(gb_ds)
        else:
            return self._possibly_transform(gb_ds, label_keys=keychain.keys(), **kwargs)

    #---------------------------------------------------------------
    def count(self, **kwargs):
        """Compute count of group"""
        return self.grouping.count(keychain=self.gb_keychain, **kwargs)

    ## ------------------------------------------------------------
    def __getattr__(self, name):
        '''
        __getattr__ is hit when '.' is used to trim a single column.
        
        ds = Dataset({'col_'+str(i): np.random.rand(5) for i in range(5)})
        >>> ds.keycol = FA(['a','a','b','c','a'])
        >>> ds.gb('keycol').col_4.mean()
        *keycol   col_4
        -------   -----
        a          0.73
        b          0.03
        c          0.76

        '''
        # check if the name exists in the dataset
        if name[0]=='_' or name not in self._dataset:
            return super().__getattribute__(name)

        return self._getitem(name)

    #---------------------------------------------------------------
    def as_categorical(self):
        '''
        Returns a categorical using the same binning information as the GroupBy object (no addtl. hash required).
        New categorical will not share a grouping object with this groupby object, but will share a reference to the iKey.
        Categorical operation results will be sorted or unsorted depending on if 'gb' or 'gbu' called this.
        '''

        idx = self.grouping.ikey
        gbkeys = self.gbkeys

        return TypeRegister.Categorical(idx, _from_categorical=gbkeys, sort_gb=self._sort_display)

    #---------------------------------------------------------------
    def _grouping_data_as_dict(self, ds):
        return self._dataset.as_ordered_dictionary()

    #-------------------------------------------------------
    def pad(self, limit=0, fill_val=None, **kwargs):
        """
        Forward fill the values

        Parameters
        ----------
        limit : integer, optional
            limit of how many values to fill

        See Also
        --------
        fill_forward
        fill_backward
        fill_invalid
        """
        return self.fill_forward(fill_val=fill_val, limit=limit, **kwargs)

    #-------------------------------------------------------
    def fill_forward(self, limit=0, fill_val=None, **kwargs):
        """
        Forward fill the values

        Parameters
        ----------
        limit : integer, optional
            limit of how many values to fill

        See Also
        --------
        fill_forward
        fill_backward
        fill_invalid
        """
        return self.apply_nonreduce(fill_forward, fill_val=fill_val, limit=limit, inplace=True)

    #-------------------------------------------------------
    def backfill(self, limit=0, fill_val=None, **kwargs):
        """
        Backward fill the values
        
        Parameters
        ----------
        limit : integer, optional
            limit of how many values to fill
        
        See Also
        --------
        fill_forward
        fill_backward
        fill_invalid
        """
        # stub for pandas
        return self.fill_backward(fill_val=fill_val, limit=limit, **kwargs)

    #-------------------------------------------------------
    def fill_backward(self, limit=0, fill_val=None, **kwargs):
        """
        Backward fill the values
        
        Parameters
        ----------
        limit : integer, optional
            limit of how many values to fill
        
        See Also
        --------
        fill_forward
        fill_backward
        fill_invalid
        """
        return self.apply_nonreduce(fill_backward, fill_val=fill_val, limit=limit, inplace=True)

    #-------------------------------------------------------
    def expanding(self, **kwargs):
        raise NotImplementedError("riptable uses the Grouping object (see .group)ck")

    #-------------------------------------------------------
    def stack(self, **kwargs):
        raise NotImplementedError("riptable uses the Accum2 class for stack/unstack")

    #-------------------------------------------------------
    def unstack(self, **kwargs):
        raise NotImplementedError("riptable uses the Accum2 class for stack/unstack")

    #-------------------------------------------------------
    @property
    def transform(self):
        '''
        The property transform sets a flag so that the next reduce function called after transform,
        will repopulate the original array with the reduced value.

        Example:
        -------
        ds.groupby(['side', 'venue']).transform.sum()
        '''
        warnings.warn("Deprecation warning: Use kwarg transform=True instead of transform.")
        self._transform=True
        return self

    #-------------------------------------------------------
    def get_group(self, category, **kwargs):
        '''
        The name of the group to get as a Dataset.

        Parameters
        ----------
        category: string or tuple
            A value from the column used to construct the GroupBy, or if
            multiple columns were used, a tuple of the multiple columns.

        Returns
        -------
        Dataset

        Examples
        --------
        ds.groupby('symbol').get_group('AAPL')
        '''
        # categorical has method to get the bin
        cat = self.as_categorical()

        bin = cat.from_category(category)
        return self._dataset[self.grouping.ikey==bin,:]


    #-------------------------------------------------------
    def _build_string(self):
        newString = f"GroupBy Keys {list(self._grouping_dict.keys())} @ [{self._dataset._ncols} x {self.grouping.unique_count}]\n"
        newString += f"ikey:{not isinstance(self.grouping.ikey, type(None))}  iFirstKey:{not isinstance(self.grouping.iFirstKey, type(None))}  iNextKey:{not isinstance(self.grouping.iNextKey, type(None))}  nCountGroup:{not isinstance(self.grouping.nCountGroup, type(None))} "
        #newString += f" _isortrows:{not isinstance(self._isortrows, type(None))}  _gbkeys:{not isinstance(self._gbkeys, type(None))}  _filter:{not isinstance(self._filter, type(None))}  _return_all:{self._return_all}\n\n"

        newString += f"_filter:{not isinstance(self._filter, type(None))}  _return_all:{self._return_all}\n\n"

        # temporarily build a dataset
        dset=self.count()

        # dump of count
        resultString= dset.__str__()

        return newString+resultString

    #-------------------------------------------------------
    def __repr__(self):
        return self._build_string()

    #-------------------------------------------------------
    def __str__(self):
        return self._build_string()

    #-------------------------------------------------------
    def _getitem(self, fld):
        '''
        Called by __getitem__ and __getattr__.
        Uses the field to index into the stored dataset.
        Often used to limit the data the groupby operation is being performed on.
        Returns a shallow copy of the groupby object.

        This routine gets hit during the following common code pattern:

        >>> ds = Dataset({'col_'+str(i): np.random.rand(5) for i in range(5)})
        >>> ds.keycol = FA(['a','a','b','c','a'])
        >>> ds.gb('keycol')[['col_1', 'col_2']].sum()
        *keycol   col_1   col_2
        -------   -----   -----
        a          1.92    0.89
        b          0.70    0.46
        c          0.07    0.42

        >>> ds.gb('keycol').col_4.mean()
        *keycol   col_4
        -------   -----
        a          0.73
        b          0.03
        c          0.76

        '''
        # return new groupby
        newgb = self.copy(deep=False)

        # reduce the datasets
        if isinstance(fld,(str, bytes)):
            # need a dataset back, not a fastarray
            index=[fld]
        elif isinstance(fld,list):
            index=fld.copy()
        # turn to list of multiple strings
        elif isinstance(fld, tuple):
            index=list(fld)
        else:
            raise TypeError("Dont know how to make groupby from",fld)

        # we must add back in the grouping keys or we cannot group
        for item in self._grouping_dict.keys():
            if item not in index:
                index.append(item)

        newgb._dataset = newgb._dataset[index]
        return newgb

    #-------------------------------------------------------
    def __getitem__(self, fld):
        return self._getitem(fld)

# keep this as the last line
from .rt_enum import TypeRegister
TypeRegister.GroupBy = GroupBy

