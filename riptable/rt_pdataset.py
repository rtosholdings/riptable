from typing import Union, List

__all__ = ['PDataset']
import warnings
import numpy as np

from .rt_fastarray import FastArray
from .rt_enum import (
    TypeRegister,
    DisplayJustification,
)
from .rt_numpy import (
    unique,
    empty,
    cumsum,
    searchsorted,
    max,
)
from .rt_dataset import Dataset
from .rt_sds import load_sds
from .rt_itemcontainer import ItemContainer
from .rt_groupby import GroupBy


class PDataset(Dataset):
    '''
    The PDataset class inherits from Dataset.  It holds multiple datasets (preivously stacked together) in contiguous slices.
    Each partition has a name and a contiguous slice that can be used to extract it from the larger Dataset.
    Extracting a partition is zero-copy.  Partitions can be extracted using partition(), or bracket [] indexing.

    A PDataset is often returned when:
        Multiple Datasets are hstacked, i.e. hstack([ds1, ds2, ds3])
        Calling load_sds with stack=True, i.e. load_sds([file1, file2, file3], stack=True)

    Properties: prows, pdict, pnames, pcount, pgb, pgbu, pgroupby, pslices, piter, pcutoffs
    Methods: partition(), pslice(), showpartitions()

    pds['20190204']  or  pds[20190204]  will return a dataset for the given partition name

    Construction:
    -------------
    inputval : -list of files to load and stack
               -list of datasets to stack
               -regular dataset inputval (will only have one partition)

    PDataset([path1, path2, path3], (pnames))
    -call load_sds(stack=True)
    -paths become filenames
    -if pnames specified, use those, otherwise look for dates
    -if no dates, auto generate pnames

    PDataset([ds1, ds2, ds3], (filenames, pnames))
    PDataset(ds, (filenames, pnames))
    -call Dataset.hstack()
    -if pnames specified, use those
    -if filenames, look for dates
    -if no dates, auto generate pnames

    PDataset(arraydict, cutoffs, (filenames, pnames))
    -constructor from load_sds()
    -if pnames specified, use those
    -if filenames, look for dates
    -if no dates, auto generate pnames

    '''

    # ------------------------------------------------------------
    def __init__(
        self,
        inputval: Union[list, dict, 'Dataset', 'ItemContainer'] = None,
        cutoffs=None,
        filenames: List[str] = None,
        pnames=None,
        showpartitions=True,
        **kwargs,
    ):
        if inputval is None:
            inputval = dict()
        if filenames is None:
            filenames = list()
        if type(inputval) == TypeRegister.Dataset:
            inputval = [inputval]

        # stack datasets or load from list of files
        if isinstance(inputval, list):
            inputval, cutoffs, filenames, pnames = self._init_from_list(
                inputval, filenames, pnames
            )

        self._pre_init()

        # fast track for itemcontainer
        if isinstance(inputval, ItemContainer):
            self._init_from_itemcontainer(inputval)
        # load items from object that can be turned into dictionary
        else:
            inputval = self._init_columns_as_dict(inputval)
            self._init_from_dict(inputval)

        self._post_init(
            cutoffs=cutoffs,
            filenames=filenames,
            pnames=pnames,
            showpartitions=showpartitions,
        )

    # ------------------------------------------------------------
    def _pre_init(self):
        '''
        Keep this in for chaining pre-inits in parent classes.
        '''
        super()._pre_init()

    # ------------------------------------------------------------
    def _post_init(self, cutoffs, filenames, pnames, showpartitions):
        '''
        Final initializer for variables specific to PDataset.
        Also initializes variables from parent class.
        '''
        super()._post_init()

        self._showpartitions = showpartitions

        # cutoffs will be the same for dataset columns
        if cutoffs is not None:
            self._pcutoffs = list(cutoffs.values())[0]
        else:
            # assume one row, init from dataset
            self._pcutoffs = FastArray([self._nrows])

        # number of rows in each partition
        self._prows = self._pcutoffs.copy()

        if len(self._prows) > 1:
            # calculate row length
            self._prows[1:] -= self._pcutoffs[:-1]

        # look for dates in filenames or autogenerate names
        if pnames is None:
            pnames, filenames = self._init_pnames_filenames(
                len(self._prows), pnames, filenames
            )
            self._pfilenames = filenames
            self._pnames = {p: i for i, p in enumerate(pnames)}
        # use provided pnames
        else:
            self._pfilenames = filenames
            if isinstance(pnames, list):
                pnames = {p: i for i, p in enumerate(pnames)}
            self._pnames = pnames

        self._pcat = None

    # ------------------------------------------------------------
    @classmethod
    def _filenames_to_pnames(cls, filenames):
        '''
        At least two filenames must be present to compare
        Algo will reverse the string on the assumption that pathnames can vary in the front of the string
        It also assumes that the filenames end similarly, such as ".SDS"
        It will search for the difference and look for digits, then try to extract the digits
        '''
        # reverse all the filenames
        if len(filenames) > 0:
            rfilenames = [f[::-1] for f in filenames]

            str_arr = TypeRegister.FastArray(rfilenames)
            str_numba = str_arr.numbastring

            if len(filenames) > 1:
                match_mask = str_numba[0] != str_numba[1]

                str_len = len(match_mask)

                for i in range(len(filenames) - 2):
                    # inplace OR loop so that the TRUE propagates
                    match_mask += str_numba[0] != str_numba[i + 2]

                for i in range(str_len):
                    if match_mask[i]:
                        break

                start = i

                for i in range(start + 1, str_len):
                    if not match_mask[i]:
                        break
                end = i

                # expand start if possible
                while start > 0:
                    char = str_numba[0][start - 1]
                    # as long as a numeric digit, keep expanding
                    if char >= 48 and char <= 58:
                        start = start - 1
                    else:
                        break

                # expand end if possible
                while end < str_len:
                    char = str_numba[0][end]
                    if char >= 48 and char <= 58:
                        end = end + 1
                    else:
                        break

                # check to see if we captured a number
                firstchar = str_numba[0][start]
                lastchar = str_numba[0][end - 1]
                if (
                    start < end
                    and firstchar >= 48
                    and firstchar <= 58
                    and lastchar >= 48
                    and lastchar <= 58
                ):
                    pnames = []
                    viewtype = 'S' + str(end - start)
                    for i in range(len(filenames)):
                        newstring = str_numba[i][start:end].view(viewtype)
                        newstring = newstring[0].astype('U')
                        # append the reverse
                        pnames.append(newstring[::-1])

                    u = unique(pnames)
                    if len(u) == len(filenames):
                        return pnames
                    # removed, prints during every column index/copy
                    # print(f"Failed to find unique numbers in filenames {pnames}")
            else:
                # only one file
                filename = str(rfilenames[0])
                start = -1
                stop = -1
                # search for first number
                for i in range(len(filename)):
                    if filename[i].isdigit():
                        if start == -1:
                            start = i
                    elif start != -1:
                        stop = i
                        break

                if start != -1:
                    if stop == -1:
                        stop = start + 1

                    # extract just the number
                    filename = filename[start:stop]
                    return [filename[::-1]]

        # failed to find unique strings in filenames
        # default to  p0, p1, p2
        pnames = cls._auto_pnames(len(filenames))

        return pnames

    # ------------------------------------------------------------
    @classmethod
    def _init_from_list(cls, dlist, filenames, pnames):
        '''
        Construct a PDataset from multiple datasets, or by loading multiple files.
        '''
        # make sure only one type
        listtype = {type(i) for i in dlist}
        if len(listtype) == 1:
            listtype = list(listtype)[0]
        else:
            raise TypeError(f'Found multiple types in constructor list {listtype}')

        # hstack datasets
        if listtype == Dataset:
            start = 0
            cutoffs = cumsum([ds.shape[0] for ds in dlist])
            cutoffs = {'cutoffs': cutoffs}
            ds = TypeRegister.Dataset.concat_rows(dlist)
            # extract itemcontainer
            ds = ds._all_items
            pnames, filenames = cls._init_pnames_filenames(
                len(dlist), pnames, filenames
            )

        # perform a .sds load from multiple files
        elif listtype == str or listtype == bytes:
            ds = load_sds(dlist, stack=True)
            cutoffs = {'cutoffs': ds._pcutoffs}
            filenames = ds._pfilenames
            if pnames is None:
                pnames = ds._pnames  # dict
            # extract itemcontainer
            ds = ds._all_items

        else:
            raise TypeError(f'Cannot construct from list of type {listtype}')

        return ds, cutoffs, filenames, pnames

    # ------------------------------------------------------------
    @classmethod
    def _auto_pnames(cls, pcount):
        '''
        Auto generate partition names if none provided and no date found in filenames.
        '''
        return ['p' + str(i) for i in range(pcount)]

    # ------------------------------------------------------------
    def _autocomplete(self) -> str:
        return f'PDataset{self.shape}'

    # ------------------------------------------------------------
    @classmethod
    def _init_pnames_filenames(cls, pcount, pnames, filenames):
        '''
        Initialize filenames, pnames based on what was provided to the constructor.
        If no pnames provided, try to derive a date from filenames
        If no date found, or no filenames provided, use default names [p0, p1, p2 ...]
        
        pcount    : number of partitions, in case names need to be auto generated
        pnames    : list of partion names or None
        filenames : list of file paths (possibly empty)

        '''
        if pnames is None:
            if filenames is None or len(filenames) == 0:
                filenames = []
                pnames = cls._auto_pnames(pcount)
            else:
                pnames = cls._filenames_to_pnames(filenames)

        return pnames, filenames

    # ------------------------------------------------------------
    def _copy(self, deep=False, rows=None, cols=None, base_index=0, cls=None):
        ''' returns a PDataset if no row selection, otherwise Dataset'''

        if rows is None:
            newcols = self._as_itemcontainer(
                deep=deep, rows=rows, cols=cols, base_index=base_index
            )
            # create a new PDataset
            pds = type(self)(
                newcols,
                cutoffs={'cutoffs': self.pcutoffs},
                filenames=self._pfilenames,
                pnames=self._pnames,
                base_index=base_index,
            )
            pds = self._copy_attributes(pds, deep=deep)
        else:
            # row slicing will break partitions, return a regular Dataset
            cls = TypeRegister.Dataset
            pds = super()._copy(
                deep=deep, rows=rows, cols=cols, base_index=base_index, cls=cls
            )

        return pds

    # ------------------------------------------------------------
    def _ipython_key_completions_(self):
        # For tab autocomplete with __getitem__
        # NOTE: %config IPCompleter.greedy=True   might have to be set
        # autocompleter will sort the keys
        return self.keys() + self.pnames

    # ------------------------------------------------------------
    @property
    def pcutoffs(self):
        '''
        Returns
        -------
        Cutoffs for partition. For slicing, maintain contiguous arrays.

        Examples
        --------
        >>> pds.pcutoffs
        FastArray([1447138, 3046565, 5344567], dtype=int64)
        '''
        return self._pcutoffs

    # ------------------------------------------------------------
    @property
    def prows(self):
        '''
        Returns
        -------
        An array with the number of rows in each partition.

        Examples
        --------
        Example below assumes 3 filenames date encoded with datasets

        >>> pds = load_sds([file1, file2, file3], stack=True)
        >>> pds.prows
        FastArray([1447138, 2599427, 1909895], dtype=int64)
        '''
        return self._prows

    # ------------------------------------------------------------
    @property
    def pcount(self):
        '''
        Returns
        -------
        Number of partitions

        Examples
        --------
        Example below assumes 3 filenames date encoded with datasets

        >>> pds = load_sds([file1, file2, file3], stack=True)
        >>> pds.pcount
        3
        '''
        return len(self._prows)

    # ------------------------------------------------------------
    @property
    def pnames(self):
        '''
        Returns
        -------
        A list with the names of the partitions

        Example
        --------
        Example below assumes 3 filenames date encoded with datasets

        >>> pds = load_sds([file1, file2, file3], stack=True)
        >>> pds.pnames
        ['20190205', '20190206', '20190207']
        '''
        return [*self._pnames.keys()]

    def set_pnames(self, pnames):
        '''
        Input
        -----
        A list of strings

        Examples
        --------
        Example below assumes 3 filenames date encoded with datasets

        >>> pds = load_sds([file1, file2, file3], stack=True)
        >>> pds.pnames
        ['20190205', '20190206', '20190207']
        >>> pds.set_pnames(['Jane', 'John', 'Jill'])
        ['Jane', 'John', 'Jill']
        '''
        if isinstance(pnames, list):
            if len(pnames) == len(self._pnames):
                newpnames = {}
                for i in range(len(pnames)):
                    newpnames[pnames[i]] = i

                if len(newpnames) == len(self._pnames):
                    self._pnames = newpnames
                else:
                    raise ValueError(f'The new pnames must be unique names: {pnames}')

            else:
                raise ValueError(
                    f'The length of the new pnames must match the length of the old pnames: {len(self._pnames)}'
                )
        else:
            raise ValueError(f'A list of string must be passed in')

        return [*self._pnames.keys()]

    # ------------------------------------------------------------
    @property
    def pdict(self):
        '''
        Returns
        --------
        A dictionary with the partition names and the partition slices.

        Examples
        --------
        Example below assumes 3 filenames date encoded with datasets

        >>> pds = load_sds([file1, file2, file3], stack=True)
        >>> pds.pdict
        {'20190204': slice(0, 1447138, None),
         '20190205': slice(1447138, 3046565, None),
         '20190206': slice(3046565, 4509322, None)}
        '''
        pdict = {name: self.pslice(i) for i, name in enumerate(self.pnames)}
        return pdict

    # ------------------------------------------------------------
    # -------------------------------------------------------
    def pgb(self, by, **kwargs):
        """Equivalent to :meth:`~rt.rt_dataset.Dataset.pgroupby`"""
        kwargs['sort'] = True
        return self.pgroupby(by, **kwargs)

    # -------------------------------------------------------
    def pgroupby(self, by, **kwargs):
        return GroupBy(self, by, cutoffs=self._pcutoffs, **kwargs)

    def igroupby(self):
        '''
        Lazily generate a categorical binned by each partition.
        Data will be attached to categorical, so operations can be called without specifying data.
        This allows reduce functions to be applied per partion.

        Examples
        --------
        Example below assumes 3 filenames date encoded with datasets

        >>> pds = load_sds([file1,file2, file2], stack=True)
        >>> pds.pgroupby['AskSize'].sum()
        *Partition   TradeSize
        ----------   ---------
        20190204     1.561e+07
        20190205     1.950e+07
        20190206     1.532e+07

        See Also: Dataset.groupby, Dataset.gb, Dataset.gbu
        '''
        reserved_name = 'Partition'

        if reserved_name not in self:
            self[reserved_name] = self.pcat
            self.col_move_to_front(reserved_name)

        return self.gb(reserved_name)

    @property
    def pcat(self):
        '''
        Lazy generates a categorical for row labels callback or pgroupby
        '''
        if self._pcat is None:
            idx = empty((self.shape[0],), dtype=np.int32)
            for i in range(self.pcount):
                idx[self.pslice(i)] = i + 1

            label = self.pnames
            self._pcat = TypeRegister.Categorical(idx, label)
        return self._pcat

    # ------------------------------------------------------------
    def prow_labeler(self, rownumbers, style):
        '''
        Display calls this routine back to replace row numbers.
        rownumbers   : fancy index of row numbers being displayed
        style : ColumnStyle object - default from DisplayTable, can be changed

        Returns: label header, label array, style
        '''
        if self._showpartitions:
            style.align = DisplayJustification.Right

            # use the cutoffs to generate which partition index
            pindex = searchsorted(self._pcutoffs, rownumbers, side='right')
            plabels = TypeRegister.FastArray(self.pnames)[pindex]

            # find the maximum string width for the rownumber
            if len(rownumbers) > 0: maxnum = max(rownumbers)
            else: maxnum = 0
            width = len(str(maxnum))

            # right justify numbers
            rownumbers = rownumbers.astype('S')
            rownumbers = np.chararray.rjust(rownumbers, width)

            # column header
            header = 'partition + #'
            rownumbers = plabels + ' ' + rownumbers

            # set the style width to override the string trim
            style.width = rownumbers.itemsize

            return header, rownumbers, style
        else:
            return '#', rownumbers, style

    # ------------------------------------------------------------
    @property
    def _row_numbers(self):
        # display will check for the existence of this method
        # return a callback to change the row numbers
        return self.prow_labeler

    # ------------------------------------------------------------
    def showpartitions(self, show=True):
        ''' toggle whether partitions are shown on the left '''
        if show:
            self._showpartitions = True
        else:
            self._showpartitions = False

    # ------------------------------------------------------------
    @property
    def piter(self):
        '''
        Iterate over dictionary of arrays for each partition.
        Yields key (load source) -> value (dataset as dictionary)

        Examples
        --------
        Example below assumes 3 filenames date encoded with datasets

        >>> pds = load_sds([file1,file2, file2], stack=True)
        >>> for name, ds in pds.iter: print(name)
        20190204
        20190205
        20190206
        '''

        label = self.pnames
        start = 0
        for i in range(self.pcount):
            yield label[i], self.partition(i)

    # -------------------------------------------------------
    @property
    def pslices(self):
        ''' 
        Return the slice (start,end) associated with the partition number  

        See Also
        --------
        pslices, pdict

        Examples
        --------
        Example below assumes 3 filenames date encoded with datasets

        >>> pds = load_sds([file1,file2, file2], stack=True)
        >>> pds.pslices
        [slice(0, 1447138, None),
         slice(1447138, 3046565, None),
         slice(3046565, 4509322, None)]
        '''

        pslices = [self.pslice(i) for i in range(self.pcount)]
        return pslices

    # -------------------------------------------------------
    def pslice(self, index):
        ''' 
        Return the slice (start,end) associated with the partition number  

        See Also
        --------
        pslices, pdict

        Examples
        --------
        >>> pds.pslice(0)
        slice(0, 1447138, None)

        '''
        if isinstance(index, (int, np.int)):
            if index == 0:
                return slice(0, self.pcutoffs[index])
            else:
                return slice(self.pcutoffs[index - 1], self.pcutoffs[index])

        raise IndexError(
            f'Cannot slice a parition with type {type(index)!r}.  Use an integer instead.'
        )

    # -------------------------------------------------------
    def partition(self, index):
        '''
        Return the Dataset associated with the partition number  

        Examples
        --------
        Example below assumes 3 filenames with datasets
        
        >>> pds = load_sds([file1, file2, file2], stack=True)
        >>> pds.partition(0)
        '''

        if isinstance(index, (int, np.int)):
            # this will route to the dataset
            return self._copy(rows=self.pslice(index))

        if isinstance(index, str):
            # this will loop back if the string is a partition name
            return self[index]

        raise IndexError(
            f'Cannot index a parition with type {type(index)!r}.  Use an integer instead.'
        )

    # -------------------------------------------------------
    def __getitem__(self, index):
        """
        :param index: (rowspec, colspec) or colspec
        :return: the indexed row(s), cols(s), sub-dataset or single value
        :raise IndexError:
        :raise TypeError:
        :raise KeyError:
        """
        try:
            return super().__getitem__(index)
        except:
            # if it fails, maybe it was a partition selection
            if isinstance(index, (int, np.int)):
                # convert int to string to lookup
                index = str(index)

            # the string was not a column name, now check for partition name
            if isinstance(index, str):
                if index in self._pnames:
                    # return the dataset for that partition
                    return self.partition(self._pnames[index])
                else:
                    raise KeyError(
                        f'the key {index!r} was not found as column name or parition name'
                    )
            else:
                raise KeyError(f'could not index PDataset with {type(index)}')

    # --------------------------------------------------------------------------
    def save(
        self,
        path='',
        share=None,
        compress=True,
        overwrite=True,
        name=None,
        onefile: bool = False,
        bandsize=None,
        append=None,
        complevel=None,
    ):
        warnings.warn(
            f"To be implemented. PDataset will currently be saved / loaded as a Dataset."
        )
        super().save(
            path=path,
            share=share,
            compress=compress,
            overwrite=overwrite,
            name=name,
            onefile=onefile,
            bandsize=bandsize,
            append=append,
            complevel=complevel,
        )

    # --------------------------------------------------------------------------
    @classmethod
    def hstack(cls, pds_list):
        '''
        Stacks columns from multiple datasets.
        see: Dataset.concat_rows
        '''
        raise NotImplementedError("PDataset does not stack yet")

    # ------------------------------------------------------------
    @classmethod
    def pload(cls, path, start, end, include=None, threads=None, folders=None):
        '''
        Returns a PDataset of stacked files from multiple days.
        Will load all files found within the date range provided.

        Parameters:
        -----------
        path  : format string for filepath, {} in place of YYYYMMDD. {} may appear multiple times.
        start : integer or string start date in format YYYYMMDD
        end   : integer or string end date in format YYYYMMDD
        '''
        # insert date string at each of these
        fmtcount = path.count('{}')

        # final loader will check if dates exist, kill warnings?
        pnames = TypeRegister.Date.range(str(start), str(end)).yyyymmdd.astype('U')

        try:
            import sotpath

            files = [sotpath.path2platform(path.format(*[d] * fmtcount)) for d in pnames]
        except:
            files = [path.format(*[d] * fmtcount) for d in pnames]

        pds = load_sds(
            files, include=include, stack=True, threads=threads, folders=folders
        )

        return pds

    # ------------------------------------------------------------
    def psave(self):
        '''
        Does not work yet.  Would save backout all the partitions.
        '''
        raise NotImplementedError(f'not implemented yet')


TypeRegister.PDataset = PDataset
