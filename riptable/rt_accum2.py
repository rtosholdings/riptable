__all__ = ['Accum2']

import numpy as np
import warnings

from .rt_fastarray import FastArray
from .rt_categorical import Categorical
from .rt_groupbyops import GroupByOps
from .rt_grouping import combine2groups
from .rt_numpy import empty
from .rt_enum import DisplayLength, DisplayJustification, TypeId
from .rt_enum import GB_FUNCTIONS, GB_FUNC_COUNT, INVALID_SHORT_NAME, FILTERED_LONG_NAME, INVALID_DICT
from .rt_enum import TypeRegister
from .rt_numpy import _groupbycalculateall, groupbypack, _groupbycalculateallpack, full, where, ones, zeros, bool_to_fancy, ismember
from .Utils.rt_display_properties import ItemFormat
from .Utils.rt_metadata import MetaData


class Accum2(GroupByOps, FastArray):
    """
    The Accum2 object is very similar to a GroupBy object that has been initialized with a multikey Categorical.

    The Accum2 object is very similar to a GroupBy object that has been initialized with a multikey Categorical.
    Because it also inherits from GroupByOps, all calculations will be sent to _calculate_all in a Grouping object.
    Accum2 generates a single array of data, and splits it into multiple columns - one for each x-axis bin.
    There is always an invalid bin, but it is omitted by default when the single array is split into columns.
    Datasets resulting from an Accum2 groupby calculation will be displayed with a footer row of column totals,
    and an additional vertical column of row totals.

    In addition to inheriting from GroupByOps, Accum2 also inherits from FastArray. This way, it can exist as a
    column in a Dataset. Its cell data will appear as a tuple of values from its X and Y axis.

    Parameters
    ----------
    cat_rows: Categorical
        Categorical for the rows axis, or an array which will be converted to a Categorical.

    cat_cols: Categorical
        Categorical for the column axis, or an array which will be converted to a Categorical.

    Keywords
    --------
    invalid: defaults to False. Set to True to show filtered columns
    ordered: defaults to None. See Categorical
    sort_gb: defaults to False. See Categorical
    ylabel: defaults to None. Set to a string to override the name of the left column
    totals: defaults to True.
    There is no sort_display option

    Returns
    -------
    Accum2 object which can be used to perform calculations
    Accum2 subclasses from FastArray and can be added to a dataset

    Accum2.operation is then supported.  Accum2(catx, caty).min(array1)
    See: groupbyops

    Examples
    --------
    >>> int_fa = FastArray([1,2,3,4]*4)
    >>> str_fa = FastArray(['a','b','c','d','b','c','d','a','c','d','b','a','d','a','b','c'])
    >>> data_col = np.random.rand(16)*10
    >>> data_col
    array([6.7337479 , 1.69561884, 8.20657899, 6.12821287, 3.95380641,
            1.06706672, 9.51679965, 3.57184704, 7.86268264, 9.0136061 ,
            2.12355667, 3.64954958, 8.40952542, 0.06431684, 9.52872172,
            3.94938333])   #random

    >>> c_x = Categorical(str_fa)
    >>> c_y = Categorical(int_fa)
    >>> ac = Accum2(c_x, c_y)
    >>> ac
    Accum2 Keys
     X:[b'a' b'b' b'c' b'd']
     Y:{'key_0': FastArray([1, 2, 3, 4])}
     Bins:25   Rows:16
    <BLANKLINE>
    *YLabel   a   b   c   d   Total
    -------   -   -   -   -   -----
          1   1   1   1   1       4
          2   1   1   1   1       4
          3   0   2   1   1       4
          4   2   0   1   1       4
    -------   -   -   -   -   -----
      Total   4   4   4   4      16

    >>> ac.sum(data_col)
    *YLabel       a       b       c       d   Total
    -------   -----   -----   -----   -----   -----
          1    6.73    3.95    7.86    8.41   26.96
          2    0.06    1.70    1.07    9.01   11.84
          3    0.00   11.65    8.21    9.52   29.38
          4    7.22    0.00    3.95    6.13   17.30
    -------   -----   -----   -----   -----   -----
      Total   14.02   17.30   21.09   33.07   85.48
    """

    DebugMode: bool = False
    # max value set for x-axis labels (multikey only). performance will suffer if large array of tuple strings is generated.
    ACCUM_X_MAX: int = 10_000

    def __new__(cls, cat_rows, cat_cols, filter = None, showfilter=False, ordered=None, sort_gb=False, totals=True, ylabel=None):
        # sort_display/sort_gb is not allowed so that the imatrix is always correct
        try:
            if not isinstance(cat_rows,Categorical):
                cat_rows=Categorical(cat_rows, ordered=ordered, sort_gb=sort_gb)
        except:
            pass

        if not isinstance(cat_rows,Categorical):
            raise TypeError(f"accum2: Argument 1 must be a categorical or an array that can be made into a categorical not type {type(cat_rows)!r}")

        try:
            if not isinstance(cat_cols,Categorical):
                cat_cols=Categorical(cat_cols, ordered=ordered, sort_gb=sort_gb)
        except:
            pass

        if not isinstance(cat_cols,Categorical):
            raise TypeError(f"accum2: Argument 2 must be a categorical or an array that can be made into a categorical not type {type(cat_cols)!r}")

        # enum or dict type categoricals are not 0 or 1 based, so we convert them
        if cat_rows.isenum:
            cat_rows = cat_rows.as_singlekey(ordered=ordered)

        if cat_cols.isenum:
            cat_cols = cat_cols.as_singlekey(ordered=ordered)

        # test uniqueness of cols categorical - large multikey will cause performance errors
        col_keys = cat_cols.gb_keychain
        if col_keys.multikey:
            if col_keys.unique_count > cls.ACCUM_X_MAX:
                raise ValueError(f"Multikey categorical's groupby keys are too large for column-axis. Use a unique amount smaller than {cls.ACCUM_X_MAX} or use in the row-axis")

        # this will group the two categoricals
        # generate iKey,unique_count and grouping object

        # call CPP algo to merge two bins into one
        grouping = combine2groups(cat_rows.grouping, cat_cols.grouping, filter=filter)
        instance = grouping.ikey.view(cls)
        instance.grouping = grouping
        instance._cat_cols = cat_cols
        instance._cat_rows = cat_rows
        instance._return_all = False
        instance._showfilter = showfilter
        instance._totals = totals
        instance._dataset = None
        instance._gb_keychain = None
        instance._ylabel = ylabel
        # filter already applied but groupbyops looks for it by default
        instance._filter =None

        return instance

    #---------------------------------------------------------------
    def __init__(cls, cat_rows, cat_cols, filter=None, showfilter=False, ordered=None, sort_gb=False, totals=True,
                ylabel=None):
        pass

    # ------------------------------------------------------------
    def __len__(self):
        return super().__len__()

    # ------------------------------------------------------------
    def __del__(self):
        """
        Called when a Categorical is deleted.
        """
        # python has trouble deleting objects with circular references
        del self.grouping
        self.grouping = None
        del self._cat_cols
        del self._cat_rows
        if self._dataset: del self._dataset
        if self._gb_keychain: del self._gb_keychain
        if self._ylabel: del self._ylabel

    # ------------------------------------------------------------
    @property
    def size(self):
        return self.__len__()

    #---------------------------------------------------------------
    @property
    def gbkeys(self):
        return self._cat_rows.grouping_dict

    #---------------------------------------------------------------
    @property
    def isortrows(self):
        return self._cat_rows.isortrows

    #---------------------------------------------------------------
    @property
    def gb_keychain(self):
        """
        Request a GroupByKeys from the y-axis categorical.

        This provides unique keys, a possible sorted index, and the ability to add a filtered bin to the final table from groupby calculations.
        """
        if self._gb_keychain is None:
            self._gb_keychain = self._cat_rows.gb_keychain
        return self._gb_keychain

    #------------------------------------------------------------
    def display_query_properties(self):
        """
        Take over display query properties from parent class FastArray.

        When displayed in a Dataset, Accum2 data will be displayed as a tuple composite of its categorical (x,y) bin values.
        """
        item_format = ItemFormat(
            length          = DisplayLength.Long,
            justification   = DisplayJustification.Left,
            can_have_spaces = True,
            decoration      = None
        )
        convert_func = self.display_convert_func
        return item_format, convert_func

    #------------------------------------------------------------
    def display_convert_func(self, index, itemformat:ItemFormat):
        # TODO: apply ItemFormat options that were passed in
        item= str(self._internal_getitem(index))
        return str(item).replace("'","")

    # ------------------------------------------------------------
    def _internal_getitem(self, matrix_index):
        xidx = matrix_index // (self._cat_rows.unique_count+1)
        yidx = matrix_index % (self._cat_rows.unique_count+1)

        if xidx ==0:
            xcat=INVALID_SHORT_NAME
        else:
            xidx -=1
            xcat = self._cat_cols.gb_keychain.get_bin(xidx)

        if yidx ==0:
            ycat=INVALID_SHORT_NAME
        else:
            yidx -=1
            ycat = self._cat_rows.gb_keychain.get_bin(yidx)

        if isinstance(xcat, bytes): xcat=bytes.decode(xcat)
        if isinstance(ycat, bytes): ycat=bytes.decode(ycat)
        result = (xcat,ycat)
        return result

    # ------------------------------------------------------------
    def __getitem__(self, fld):
        """
        Bracket indexing for Accum2.
        """
        if Accum2.DebugMode: print("***get item from:", fld)

        result = None
        # cat[int]
        # returns single category
        if isinstance(fld, (int, np.integer)):
            if Accum2.DebugMode: print("***self.view(FastArray)\n", self.view(FastArray))
            matrix_index = self._np[fld]
            return self._internal_getitem(matrix_index)

        else:
            return super(Accum2, self).__getitem__(fld)

    #---------------------------------------------------------------
    def _get_gbkeyname(self):
        if self._ylabel is not None:
            return self._ylabel

        # for single key
        gbkeyname = self._cat_rows.get_name()
        if gbkeyname is None:
            gbkeyname = "YLabel"
        return gbkeyname

    #---------------------------------------------------------------
    def _get_gbkeys(self, showfilter=False):
        if self.gb_keychain.singlekey:
            gbkeyname = self._get_gbkeyname()
            if showfilter:
                keys = self.gb_keychain.gbkeys_filtered
            else:
                keys = self.gb_keychain.gbkeys

            keycol = list(keys.values())[0]
            if self._cat_rows.category_mode in TypeRegister.Categories.dict_modes:
                keycol = Categorical(keycol, _from_categorical=self._cat_rows._categories_wrap)

            return {gbkeyname: keycol}
           # return {gbkeyname:self._cat_rows._categories}
        else:
            if showfilter:
                return self.gb_keychain.gbkeys_filtered
            else:
                return self.gb_keychain.gbkeys

    #---------------------------------------------------------------
    def _make_imatrix(self, input_arr, col_keys, row_keys, showfilter=False):
        """
        return a fortan ordered 2d matrix
        if showfilter is False, the first column is removed
            shape is (row_keys.unique_count+1, col_keys.unique_count)
        else if showfilter is True
            shape is (row_keys.unique_count +1, col_keys.unique_count+1)
        """
        if showfilter:
            tempi = input_arr[0:-1]
            imatrix = tempi.reshape((row_keys.unique_count +1, col_keys.unique_count+1), order='F')
        else:
            # skip over first column (the filtered column)
            tempi = input_arr[ row_keys.unique_count+1:-1]
            imatrix = tempi.reshape((row_keys.unique_count+1, col_keys.unique_count  ), order='F')
        return imatrix

    #---------------------------------------------------------------
    def make_dataset(self, arr, showfilter=False):
        """

        Parameters
        ----------
        arr: input array of data

        Returns
        -------
        ds
        col_keys
        row_keys
        """
        # default to no showfilter
        showfilter_base = 1

        # check if we need to show showfilter columns
        if showfilter:
            showfilter_base = 0

        # possibly attach filter bin to gbkey columns
        gbkeys= self._get_gbkeys(showfilter=showfilter)

        # put the grouping keys as the first columns
        newds=TypeRegister.Dataset(gbkeys)

        col_keys = self._cat_cols.gb_keychain
        row_keys = self._cat_rows.gb_keychain

        # x-axis headers need to be a single list
        if col_keys.singlekey:
            #xcategories = self._cat_cols._categories
            xcategories = list(col_keys.gbkeys.values())[0]
        else:
            # generate tuple strings for multikey
            if showfilter:
                xcategories = col_keys.multikey_labels_filtered
            else:
                xcategories = col_keys.multikey_labels

        offsety = row_keys.unique_count+1

        if Accum2.DebugMode: print("**stack len", len(arr), offsety, "  showfilterbase:", showfilter_base)

        # add showfilter column first if we have to
        if showfilter:
            newds[FILTERED_LONG_NAME] = arr[0: offsety]

        # skip showfilter row for loop below (already added or not added above)
        offset = offsety

        # fix bug for enums, need to reattach code mapping for correct string
        xmode = self._cat_cols.category_mode
        if xmode in TypeRegister.Categories.dict_modes:
            xcategories = TypeRegister.Categorical(xcategories, _from_categorical=self._cat_cols._categories_wrap)

        # cut main array into multiple columns
        for i in range(col_keys.unique_count):
            new_colname = xcategories[i]

            if isinstance(new_colname, bytes):
                new_colname = new_colname.decode()

            if isinstance(new_colname, str):
                if len(new_colname) == 0:
                    #make up a column name
                    new_colname=INVALID_SHORT_NAME+str(i)
            else:
                new_colname = str(new_colname)

            start = showfilter_base + offset
            stop = offset + offsety
            offset += offsety

            # possibly skip over filter
            arridx = slice(start,stop)
            newds[new_colname] = arr[arridx]


        return {'ds':newds, 'col_keys':col_keys, 'row_keys':row_keys, 'gbkeys' : gbkeys}


    #---------------------------------------------------------------
    @classmethod
    def _apply_2d_operation(self, func, imatrix, showfilter=True,
                           filter_rows = None, filter_cols=None):
        """
        Called from routines like sum or min where we can make one pass

        If there are badrows, then filter_rows is set to the row indexes that are bad
        If there are badcols, then filter_cols is set to the col indexes that are bad
        filter_rows is a fancy index or none
        """

        if callable(func):

            row_count, col_count = imatrix.shape

            # horizontal add
            #print("im0", imatrix.nansum())
            totalsY = func(imatrix, axis=1)  #[showfilter_base:]

            # vertical func operation
            totalsX = empty(col_count, dtype=totalsY.dtype)

            # possibly remove filtered top row
            if not showfilter:
                totalsY = totalsY[1:]

            # consider #imatrix.nansum(axis=0, out=totalsX)
            for i in range(col_count):
                arrslice = imatrix[:,i]

                # possibly skip over first value
                if not showfilter:
                    arrslice =arrslice[1:]

                totalsX[i] = func(arrslice)

            return totalsX, totalsY

        # function was not callable
        return None, None

    #---------------------------------------------------------------
    @classmethod
    def _accum1_pass(cls, cat, origarr, funcNum, showfilter=False, filter=None, func_param=0, **kwargs):
        """
        internal call to calculate the Y or X summary axis
        the filter muse be passed correctly
        returns array with result of operation, size of array is number of uniques
        """

        basebin =1
        if showfilter:
            basebin =0

        if callable(funcNum):
            # from apply_reduce
            #funcList = [GB_FUNCTIONS.GB_SUM]
            #accum_tuple = _groupbycalculateall([origarr], ikey, numkeys, funcList, binLowList, binHighList, func_param)

            # need a new option here, which is that we want to allocate for a filter
            # but we might not use it
            # ALSO dont want back a dataaset
            accum_tuple = cat.apply_reduce(funcNum, origarr, showfilter=showfilter, filter=filter, nokeys=True, **kwargs)

            # the showfilter is handled automatically
            return accum_tuple[0]

        else:
            ikey = cat.grouping.ikey

            # if zero base, we need 1 base for these calculations
            if cat.grouping.base_index == 0:
                ikey = ikey + 1

            # Optimization: combine_filter was previously called
            if filter is not None:
                # N.B. We are going to change ikey, make a copy instead of changing the input. The input
                #      data will be used again when user calls method on the Accum2 object again.
                # zero out anything not in the filter
                ikey = where(filter, ikey, 0)

            numkeys = cat.unique_count

            funcList = [funcNum]

            binLowList = [basebin]
            binHighList = [numkeys + 1]

            if funcNum >= GB_FUNCTIONS.GB_SUM and funcNum < GB_FUNCTIONS.GB_FIRST:
                accum_tuple = _groupbycalculateall([origarr], ikey, numkeys, funcList, binLowList, binHighList, func_param)

            elif funcNum >= GB_FUNCTIONS.GB_FIRST and funcNum < GB_FUNCTIONS.GB_CUMSUM:
                # TODO break out as function
                packing= groupbypack(ikey, None, numkeys + 1)
                iGroup = packing['iGroup']
                iFirstGroup = packing['iFirstGroup']
                nCountGroup = packing['nCountGroup']
                accum_tuple = _groupbycalculateallpack([origarr], ikey, iGroup, iFirstGroup, nCountGroup, numkeys, funcList, binLowList, binHighList, func_param)

        # whether or not they want to see the filter
        if basebin != 0:
            return accum_tuple[0][basebin:]
        else:
            return accum_tuple[0]


    #---------------------------------------------------------------
    @classmethod
    def _add_totals(cls, cat_rows, newds, name, totalsX, totalsY, totalOfTotals):
        """
        Adds a summary column on the right (totalsY)
        Adds a footer on the bottom (totalsX)
        """
        if totalsY is not None:
            if newds.shape[0] != len(totalsY):
                # this path is from custom apply_reduce
                emptyarr = empty((newds.shape[0],), dtype=totalsY.dtype)
                emptyarr[0:len(totalsY)] = totalsY
                emptyarr[-1] = totalOfTotals
                newds[name]=emptyarr

            else:
                # add the Total column to the dataset
                newds[name]=totalsY

            # add to the right summary
            newds.summary_set_names([name])

            # tell display that this dataset has a footer
            # have to skip over the colkeys
            keycount=len(cat_rows.gb_keychain.gbkeys)

            # totalsX runs in the horizontal direction on the bottom
            # for each column name in the dictionary, give a value
            footerdict= dict(zip( [*newds][keycount:], totalsX))

            # lower right corner sometimes passed alone
            if totalOfTotals is not None:
                footerdict[name]=totalOfTotals

            newds.footer_set_values( name, footerdict)

    #---------------------------------------------------------------
    @classmethod
    def _calc_badslots(cls, cat, badslots, filter, wantfancy):
        """
        internal routine
        will combine (row or col filter) badslots with common filter

        if there are not badslots, the common filter is returned
        otherwise a new filter is returned
        the filter is negative (badslots locations are false)

        if wantfancy is true, returns fancy index to cols or rows
        otherwise full boolean mask combined with existing filter (if exists)
        """
        if badslots is None:
            if wantfancy:
                return None
            # user did not pass in any, stick with current filter
            return filter

        badslots = np.atleast_1d(badslots)
        dtypenum = badslots.dtype.num

        if wantfancy:
            # find out which columns are to be filtered out
            # are cols an integer or a string?
            if dtypenum > 10:
                _, newfilter = cat.grouping.ismember(badslots, reverse=True)
            else:
                # assume user passed in row or col numbers that are bad
                # such as badrows=[3,4]
                newfilter = badslots
            return newfilter

        # are they passing in a boolean filter?
        if dtypenum ==0:
            # convert bool mask to row numbers and use that mask
            badslots = bool_to_fancy(badslots)

        if dtypenum <=10:
            #assumes there is not Cat of integers..otherwise ambiguous
            # add 1 because of base_index
            # should we check showfilter?
            badslots = badslots + 1
            if len(badslots) ==1:
                newfilter = cat._fa != badslots[0]
            else:
                newfilter, _ = ismember(cat._fa, badslots)
                # inplace logical not (this is a negative filter)
                np.logical_not(newfilter, out=newfilter)
        else:
            # create filter
            newfilter = cat.isin(badslots)
            # inplace logical not (this is a negative filter)
            np.logical_not(newfilter, out=newfilter)


        if filter is not None:
            # combine both filters using inplace and of filter
            np.logical_and(newfilter, filter, out= newfilter)

        #print('newfilter', len(newfilter), newfilter.sum(), newfilter)
        # return a new filter
        return newfilter


    #---------------------------------------------------------------
    @classmethod
    def _calc_multipass(cls, cat_cols, cat_rows, newds, origarr, funcNum, func, imatrix,
                       name=None, showfilter=False, filter=None, badrows=None, badcols=None, badcalc=True, **kwargs):
        """
        For functions that require multiple passes to get the proper result.
        such as mean or median.

        If the grid is 7 x 11: there will be 77 + 11 + 7 + 1 => 96 passes

        Other Parameters
        ----------------
        func: userfunction to call calculate
        name: optional column name (otherwise function name used)
        badrows: optional list of bad row keys, will be combined with filter
        badcols: optional list of bad col keys, will be combined with filter

        badrows/cols is just the keys that are bad (not a boolean filter)
        for example badrows=['AAPL','GOOG']

        Need new algo to take:
            bad bins + ikey + existing boolean filter ==> create a new boolean filter
            walk ikey, see if bin is bad in lookup table, if so set filter to False
            else copy from existing filter value
        """
        if name is None:
            name = str.capitalize(func.__name__)

        # get a negative boolean filter
        newfilterX = cls._calc_badslots(cat_cols, badcols, filter, False)
        newfilterY = cls._calc_badslots(cat_rows, badrows, filter, False)

        newfilter = None
        # first check for any row and col filters
        if badrows is not None or badcols is not None:

            # the common filter is already merged into the row or col filter
            if badrows is not None and badcols is not None:

                # both col and row filter are in use so combine the filters
                newfilter = newfilterX & newfilterY
            else:
                if badrows is not None:
                    newfilter = newfilterY
                else:
                    newfilter = newfilterX
        else:
            newfilter = filter

        # if there is not filter, the value will be None
        if Accum2.DebugMode:
            print("filterrows", newfilterY)
            print("filtercols", newfilterX)
            print("filter    ", newfilter)

        # set to False so that totalsX has invalid where the badcols are
        # set to False so that totalsY has invalid where the badrows are
        #badcalc =True

        if badcalc:
            #  pass in original filter
            totalsX = cls._accum1_pass(cat_cols, origarr, funcNum, showfilter=showfilter, filter=newfilterY, **kwargs)
            totalsY = cls._accum1_pass(cat_rows, origarr, funcNum, showfilter=showfilter, filter=newfilterX, **kwargs)

        else:
            #  pass in combined filter since the filter was handled on class
            totalsX = cls._accum1_pass(cat_cols, origarr, funcNum, showfilter=showfilter, filter=newfilter, **kwargs)
            totalsY = cls._accum1_pass(cat_rows, origarr, funcNum, showfilter=showfilter, filter=newfilter, **kwargs)

        # calculate total of totals
        if func is not None:
            # we can have common filters, row filters, and col filters

            if newfilter is not None:
                totalOfTotals = func(origarr[newfilter])
            else:
                totalOfTotals = func(origarr)
        else:
            # todo, get invalid
            totalOfTotals = 0

        cls._add_totals(cat_rows, newds, name, totalsX, totalsY, totalOfTotals)

    #---------------------------------------------------------------
    @classmethod
    def _calc_onepass(cls, cat_cols, cat_rows, newds, origarr, funcNum, func, imatrix, name=None,
                      showfilter=False, filter=None, badrows=None, badcols=None, badcalc=True, **kwargs):
        """
        For functions such as sum or min that require one pass to get the proper result.

        The first pass calculates all the cells.  Once the cells are calculated,
        an imatrix is made.  Since functions like sum or min can calculate proper values
        for horizontal or vertical operations without making another pass, we use
        the imatrix to calculate the rest.

        The user may also pass in badrows or badcols, or both.
        When badrows is passed, the CELLS for that row are still calculated normally.
        However, the totalOfTotals will not include the badrows or cols.
        """

        # to make sure column names do not conflict with methods we capitalize first letter
        if name is None:
            name = str.capitalize(func.__name__)

        newfilter_cols = cls._calc_badslots(cat_cols, badcols, filter, True)
        newfilter_rows = cls._calc_badslots(cat_rows, badrows, filter, True)

        #newfilter_cols/rows is a fancy index
        if Accum2.DebugMode:
            print("newfilter_cols", newfilter_cols)
            print("newfilter_rows", newfilter_rows)

        # do both horizontal and vertical calculations which are clean
        # TODO optimization -- can just use empty for totalsY and not calculate it if
        #                       newfilter_rows is set  (same for totalsX)
        totalsX, totalsY =  cls._apply_2d_operation(func, imatrix, showfilter,
                                                    filter_rows = newfilter_rows, filter_cols=newfilter_cols)


        # set to False so that totalsX has invalid where the badcols are
        # set to False so that totalsY has invalid where the badrows are
        #badcalc =True

        invalid = INVALID_DICT[imatrix.dtype.num]

        # if nothing is filtered, the calculation is simple
        if newfilter_rows is None and newfilter_cols is None:
            # calc total of totals - cell on far right and bottom
            totalOfTotals = func(totalsY)

        else:
            im = imatrix
            if not showfilter:
                # remove invalid row (columns in imatrix already removed)
                im = im[1:,:]

            # create row and col mask, init to all True
            boolmaskY = ones(len(totalsY), dtype=bool)
            boolmaskX = ones(len(totalsX), dtype=bool)

            # do both horizontal and vertical calculations which are dirty
            newTotalsX, newTotalsY =  cls._apply_2d_operation(func, im, True)

            if newfilter_rows is not None:
                if showfilter:
                    # all the rows are shifted one over because Filtered row comes first
                    newfilter_rows += 1
                boolmaskY[newfilter_rows] = False

                # now set the invalids (hide the good value)
                #if not badcalc:
                #    totalsY[newfilter_rows] = invalid

                # shrink rows by removing bad values
                im = im[boolmaskY, :]

            if newfilter_cols is not None:
                if showfilter:
                    # all the cols are shifted one over because Filtered row comes first
                    newfilter_cols += 1
                boolmaskX[newfilter_cols] = False

                # now set the invalids (hide the good value)
                #if not badcalc:
                #    totalsX[newfilter_cols] = invalid

                # shrink cols
                im = im[:,boolmaskX]

            # do both horizontal and vertical calculations which are dirty
            newTotalsX, newTotalsY =  cls._apply_2d_operation(func, im, True)

            if Accum2.DebugMode:
                numrows, numcols=imatrix.shape
                print("orig imatrix rows:", numrows, " cols:", numcols)
                numrows, numcols=im.shape
                print("new imatrix rows:", numrows, " cols:", numcols)

                print("oldtotalsX:", totalsX, " oldtotalsY:", totalsY)
                print("newtotalsX:", newTotalsX, " newtotalsY:", newTotalsY)

            # now repopulate with new calculation (with bad rows or cols removed)
            totalsY[boolmaskY] = newTotalsY
            totalsX[boolmaskX] = newTotalsX

            #if badcalc:
            #    if newfilter_rows is not None:
            #        if showfilter:
            #            # all the rows are shifted one over because Filtered row comes first
            #            newfilter_rows += 1
            #        boolmaskY[newfilter_rows] = False

            #        # shrink rows by removing bad values
            #        im = im[boolmaskY, :]

            #    if newfilter_cols is not None:
            #        if showfilter:
            #            # all the cols are shifted one over because Filtered row comes first
            #            newfilter_cols += 1
            #        boolmaskX[newfilter_cols] = False

            #        # shrink cols
            #        im = im[:,boolmaskX]

            # calc totals with both rows and cols removed
            totalOfTotals = func(im)

        # push calculations to dataset (newds)
        cls._add_totals(cat_rows, newds, name, totalsX, totalsY, totalOfTotals)

    #---------------------------------------------------------------
    def _stack_dataset(self, arr, origarr, funcNum, showfilter: bool=False, tups=0, **kwargs):
        """
        Accum2 uses a single array but returns a dataset that is stacked.
        The long column is unrolled into columns.

        Parameters
        ----------
        arr :
        origarr :
        funcNum
        showfilter : bool
        kwargs : dict-like
            Keyword args to pass to the function specified by `funcNum`.

        """
        result = self.make_dataset(arr, showfilter=showfilter)

        newds = result['ds']
        col_keys=result['col_keys']
        row_keys=result['row_keys']
        gbkeys = result['gbkeys']

        # when user types in something like
        # ac.apply_reduce(lambda x,y:np.sum(np.maximum(x,y)), (newds.data, newds.data2))
        # we do not have the second parameter to pass in when we get to totals
        if self._totals and tups==0:
            imatrix = self._make_imatrix(arr, col_keys, row_keys, showfilter)
            if not callable(funcNum):
                # get the name, func, routine to call
                func = apply_dict_total.get(funcNum)
            else:
                # assume _reduce
                # funcNum is really a callable function, we can get the name
                name = funcNum.__name__.capitalize()
                if name.startswith('<'):
                    name = 'Lambda'

                func = (name, funcNum, Accum2._calc_multipass)

            if func is not None:
                # Calling single or multipass
                # may come from apply_reduce
                func[2](self._cat_cols, self._cat_rows, newds, origarr, funcNum, func[1], imatrix,
                        name=func[0], showfilter=showfilter, **kwargs)

        # tell display which columns are grouped by
        newds.label_set_names([k for k in gbkeys])

        # set any badrows/cols if we have them
        badrows = kwargs.get("badrows", None)
        if badrows is not None:
            # badrows wants fancy index
            badrows = Accum2._calc_badslots(self._cat_rows, badrows, None, True)
            if showfilter:
                badrows += 1
            newds._badrows = badrows

        badcols = kwargs.get("badcols", None)
        if badcols is not None:
            badcols = Accum2._calc_badslots(self._cat_cols, badcols, None, True)
            # badcols wants strings
            newds._badcols = self._cat_cols.category_array[badcols].astype('U')
            #print("set badcols to", newds._badcols, badcols)

        return newds

    #---------------------------------------------------------------
    # OVERRIDEN from groupbyops
    def _calculate_all(self, funcNum, *args, func_param=0, **kwargs):
        """
        Can be called from apply_reduce

        """

        keychain = self.gb_keychain
        origdict, user_args, tups = self._prepare_gb_data('Accum2', funcNum, *args, **kwargs)

        #insert showfilter from Accum2 init if not overridden
        kwargs['showfilter'] = kwargs.get('showfilter', self._showfilter)

        if len(origdict) > 0:
            test_col = list(origdict.values())[0]
            if not isinstance(test_col, np.ndarray):
                # to get here pass in something like a lambda function as input
                raise ValueError(f"Data passed in to Accum2 must be numpy arrays and not type {type(test_col)!r}")

            if len(test_col.view(FastArray)) != len(self._cat_rows._fa):
                raise ValueError(f"Data did not have the same length has categoricals in Accum2 object, {len(test_col.view(FastArray))} vs. {len(self._cat_rows._fa)}")
        else:
            warnings.warn(f"Accum2: No data was calculated")
            return

        accum_dict = self.grouping._calculate_all(origdict, funcNum, func_param=func_param,
                                            keychain=keychain, user_args = user_args, tups=tups, return_all=self._return_all,
                                           accum2=True, **kwargs)

        return self._finish_calculate_all(origdict, accum_dict, funcNum, func_param=func_param, tups=tups, **kwargs)

    #---------------------------------------------------------------
    def _finish_calculate_all(self, origdict, accum_dict, funcNum, func_param=0, tups=0, transform=False, **kwargs):
        """

        Parameters
        ----------
        origdict: original dataset input
        accum_list: input data we can calculate on
        funcNum: internal riptable groupby function number  OR
                 a callable reduce function
        func_param: optional, parameters for the function
        """
        # check if transform was called earlier
        if transform:
            ikey=self.ikey
            # use fancy indexing to pull the values from the cells, back to original array
            newds= {colname:arr[ikey] for colname,arr in accum_dict.items()}

            return TypeRegister.Dataset(newds)

        accum_list = []
        for k, v in accum_dict.items():
            if isinstance(origdict[k], TypeRegister.Categorical):
                v = TypeRegister.Categorical(v, _from_categorical=origdict[k]._categories_wrap)
            accum_list.append(v)

        if Accum2.DebugMode:
            print('**accumlist was',accum_list)
            print('**origdict was',origdict)

        # if long mode
        if not callable(funcNum) and funcNum >= GB_FUNCTIONS.GB_CUMSUM:
            return accum_list

        # todo -- make ds columns?
        if len(accum_list)==1:
            # just one input so normal dataset returned
            return self._stack_dataset( accum_list[0], origdict.popitem()[1], funcNum, func_param=func_param, tups=tups, **kwargs)

        elif len(accum_list) > 1:
            # multiple inputs, so use multiset
            # get the row and colkeys
            gbkeys = self._get_gbkeys()
            colkeys = self._cat_cols.grouping_dict

            ms=TypeRegister.Multiset({})
            for i,v in enumerate(accum_dict.keys()):

                # Accum2 uses two categoricals for rows and cols
                # dont bother to calculate the row or col keys
                # the row keys are the gbkeys
                if v not in gbkeys and v not in colkeys:
                    result = self._stack_dataset( accum_list[i], origdict[v], funcNum, func_param=func_param, tups=tups, **kwargs)
                    ms[v] = result

            gbkeyname = self._get_gbkeyname()
            ms._gbkeys = gbkeys
            #ms.label_set_names([gbkeyname])
            return ms
        else:
            print("No data was calculated")
            return None

    #---------------------------------------------------------------
    def apply_reduce(self, userfunc, *args, dataset=None, label_keys=None, func_param=None, dtype=None, transform=False,
                     **kwargs):

        '''
        Accum2:apply_reduce calls Grouping:apply_helper

        Parameters
        ----------
        userfunc : callable
            A callable that takes a contiguous array as its first argument, and returns a scalar
            In addition the callable may take positional and keyword arguments.
        args
            Used to pass in columnar data from other datasets

        Other Parameters
        ----------------
        dataset: None
            User may pass in an entire dataset to compute.
        label_keys: None
            Not supported, will use the existing groupby keys as labels.
        func_param : tuple, optional
            Set to a tuple to pass as arguments to the routine.
        dtype : str or np.dtype, optional
            Change to a numpy dtype to return an array with that dtype. Defaults to None.
        transform : bool
            Set to True to re-expand the results of the calculation. Defaults to False.
        filter:
        kwargs
            Optional positional and keyword arguments to pass to ``userfunc``

        Notes
        -----
        See Grouping.apply_reduce
        '''

        args, kwargs, origdict, tups = self._pop_gb_data('apply_reduce', userfunc, *args, **kwargs, dataset=dataset)
        # temp pop showfilter
        realshowfilter=kwargs.get('showfilter', False)
        kwargs['showfilter']=True

        accum_dict= self.grouping.apply_helper(True, origdict, userfunc, *args, tups=tups, label_keys=None,
                                              func_param=func_param, dtype=dtype, **kwargs)
        kwargs['showfilter']=realshowfilter

        return self._finish_calculate_all(origdict, accum_dict, userfunc, func_param=func_param, tups=tups, transform=transform, **kwargs)

    #---------------------------------------------------------------
    @property
    def ncountkey(self):
        """See: Grouping.ncountkey"""
        return self.grouping.ncountkey

    #---------------------------------------------------------------
    @property
    def ncountgroup(self):
        """See: Grouping.ncountgroup"""
        return self.grouping.ncountgroup

    #---------------------------------------------------------------
    @property
    def ikey(self):
        return self.grouping.ikey

    #---------------------------------------------------------------
    def count(self, **kwargs):
        """Compute count of group"""
        kwargs['showfilter'] = kwargs.get('showfilter', self._showfilter)
        #print("count array", self.ncountgroup)
        result = self._stack_dataset(self.ncountgroup, self.ncountkey, GB_FUNC_COUNT, **kwargs)
        if self._totals:
            result = result.imatrix_totals()
        return result

    #-------------------------------------------------------
    def _build_string(self):
        # build a count dataset by default
        dset=self.count()
        resultString= dset.__str__()
        return "Accum2 Keys\n X:" + self._cat_cols.unique_repr + "\n Y:" + self._cat_rows.unique_repr +"\n Bins:" + str(self.grouping.unique_count) + "   Rows:" + str(len(self._cat_rows._np))+ "\n\n" +resultString


    #-------------------------------------------------------
    def _build_sds_meta_data(self, name, **kwargs):
        meta = MetaData({
            # vars for container loader
            'name': name,
            'typeid': TypeId.Accum2,
            'classname' : self.__class__.__name__,
            '_base_is_stackable' : 0,
            'author' : 'python',

            # accum2 will always have 2 categoricals for x and y axis
            'cat_meta' : [],
            'ncols' : 0, # one for each cat underlying array

            # what are these
            'instance_vars' : {
                '_showfilter' : self._showfilter,
            }
        })

        cols = []

        for cat in [ self._cat_cols, self._cat_rows]:
            cols.append( cat._fa )
            cat_meta, cat_cols = cat._build_sds_meta_data( TypeId.Categorical.name)
            for c in cat_cols:
                cols.append( c )
            meta['cat_meta'].append( cat_meta.string )

        meta['ncols'] = len(cols)

        return meta, cols


    #-------------------------------------------------------
    @classmethod
    def _load_from_sds_meta_data(self, name, arr, cols, meta):
        if not isinstance(meta, MetaData):
            meta = MetaData(meta)

        # what instance vars need to be stored
        vars = meta['instance_vars']

        col_idx = 0
        cats = []
        # build categoricals for x and y
        for cat_meta in meta['cat_meta']:
            cat_meta = MetaData(cat_meta)
            # first array in block is categoricals underlying
            cat_arr = cols[col_idx]
            col_idx+=1

            # next columns are for categories
            cat_cols = cols[col_idx : col_idx + cat_meta['ncols']]
            col_idx += cat_meta['ncols']

            # ask categorical to reconstruct
            cat = TypeRegister.Categorical._load_from_sds_meta_data(cat_meta['name'], cat_arr, cat_cols, cat_meta)
            cats.append(cat)

        # TODO: add a fast-track routine to reconstruct accum2 object with cats, underlying array
        return Accum2(cats[0], cats[1], showfilter=vars['_showfilter'])


    #-------------------------------------------------------
    def __repr__(self):
        return self._build_string()

    #-------------------------------------------------------
    def __str__(self):
        return self._build_string()


# mapping of internal functions to the proper np routine
# for sum we only need to make one pass (calc_onepass)
# for mean we need to make multiple passes
apply_dict_total = {
    GB_FUNCTIONS.GB_SUM :    ('Total', np.sum, Accum2._calc_onepass),
    GB_FUNCTIONS.GB_NANSUM : ('Nansum', np.nansum,  Accum2._calc_onepass),
    GB_FUNCTIONS.GB_MIN :    ('Min', np.min, Accum2._calc_onepass),
    GB_FUNCTIONS.GB_NANMIN : ('Nanmin', np.nanmin, Accum2._calc_onepass),
    GB_FUNCTIONS.GB_MAX :    ('Max', np.max, Accum2._calc_onepass),
    GB_FUNCTIONS.GB_NANMAX : ('Nanmax', np.nanmax, Accum2._calc_onepass),

    GB_FUNCTIONS.GB_MEAN :   ('Mean', np.mean, Accum2._calc_multipass),
    GB_FUNCTIONS.GB_NANMEAN: ('Nanmean', np.nanmean, Accum2._calc_multipass),
    GB_FUNCTIONS.GB_VAR :    ('Var', np.var, Accum2._calc_multipass),
    GB_FUNCTIONS.GB_NANVAR : ('Nanvar', np.nanvar, Accum2._calc_multipass),
    GB_FUNCTIONS.GB_STD :    ('Std', np.std, Accum2._calc_multipass),
    GB_FUNCTIONS.GB_NANSTD : ('Nanstd', np.nanstd, Accum2._calc_multipass),

    GB_FUNCTIONS.GB_MEDIAN : ('Median', np.nanmedian, Accum2._calc_multipass),
    GB_FUNCTIONS.GB_MODE :   ('Mode', None, Accum2._calc_multipass),
    GB_FUNCTIONS.GB_TRIMBR : ('Trimbr', None, Accum2._calc_multipass),

    # -- transform (or same size) functions
    #GB_FUNCTIONS.GB_CUMSUM : ('Cumsum', np.cumsum, Accum2._calc_multipass),
    }


# keep this as the last line
TypeRegister.Accum2 = Accum2
