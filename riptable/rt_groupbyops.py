__all__ = ['GroupByOps']
import warnings
import numpy as np
import riptide_cpp as rc
from .rt_enum import GB_FUNCTIONS, GB_STRING_ALLOWED, GB_FUNC_COUNT, GB_PACKUNPACK, TypeRegister, ApplyType
from .rt_numpy import zeros_like, bool_to_fancy, empty_like, groupbyhash
from .rt_timers import tt

#=====================================================================================================
#=====================================================================================================
class GroupByOps(object):
    """
    Holds all the functions for groupby

    Only used when inherited
    Child class must have self.grouping

    Child class must override apply() func --> internal_apply
    Child class must override count()
    Child class must have its own _calculate_all
    """

    DebugMode=False
    AggNames = {'count',
                'cumsum',
                'first',
                'last',
                'max',
                'mean',
                'median',
                'min',
                'nanmax',
                'nanmean',
                'nanmedian',
                'nanmin',
                'nanstd',
                'nansum',
                'nanvar',
                'nth',
                'std',
                'sum',
                'var'}

    # after pulling name from numpy method, route to different groupbyops method
    NumpyAggNames = {
        'amin' : 'min',
        'amax' : 'max',
    }


    def __init__(self):
        self._gbkeys = None
        self._groups = None

    #---------------------------------------------------------------
    @classmethod
    def register_functions(cls, functable):
        '''
        Registration should follow the NUMBA_REVERSE_TABLE layout at the bottom of rt_groupbynumba.py
        If we register again, the last to register will be executed.
        NUMBA_REVERSE_TABLE[i + GB_FUNC_NUMBA]={'name': k,  'packing': v[0], 'func_front': v[1], 'func_back': v[2],  'func_gb':v[3],  'func_dtype': v[4], 'return_full': v[5]}
        '''
        for v in functable.values():
            # dict looks like --> name : {packing, func_front, func_back, ...}
            # use the func_frontend
            setattr(cls, v['name'], v['func_front'])


    #---------------------------------------------------------------
    def as_filter(self,index):
        ''' return an index filter for a given unique key'''
        return self.grouping.as_filter(index)

    #---------------------------------------------------------------
    @property
    def groups(self):
        '''
        Returns a dictionary of unique key values -> their fancy indices of occurrence in the original data.
        '''
        # make sure we get the unsorted list
        #gbkeys = self.gb_keychain.gbkeys
        gbkeys = self.grouping.gbkeys
        col = list(gbkeys.values())
        unique_count = self.grouping.unique_count

        # make tuples from multikey values
        if len(gbkeys) > 1:
            col = [ tuple( c[i] for c in col ) for i in range(unique_count) ]
        # use single values from array
        else:
            col = col[0]
        fancy_idx = [self.as_filter(i+1) for i in range(unique_count)]
        return dict(zip(col, fancy_idx))

    #---------------------------------------------------------------
    def _dict_val_at_index(self, index):
        '''
        Returns the value of the group label for a given index.
        A single-key grouping will return a single value.
        A multi-key grouping will return a tuple of values.
        '''
        keycols = list(self.grouping.gbkeys.values())
        labels = []
        for c in keycols:
            labels.append(c[index])
        if len(labels) == 1:
            return labels[0]
        else:
            return tuple(labels)

    #---------------------------------------------------------------
    def key_from_bin(self, bin):
        '''
        Returns the value of the group label for a given index. (uses zero-based indexing)
        A single-key grouping will return a single value.
        A multi-key grouping will return a tuple of values.
        '''
        return self._dict_val_at_index(bin)

    #---------------------------------------------------------------
    def iter_groups(self):
        '''
        Very similar to the 'groups' property, but uses a generator instead of building the entire dictionary.
        Returned pairs will be group label value (or tuple of multikey group label values) --> fancy index for that group (base-0).
        '''
        return self._iter_internal()

    #---------------------------------------------------------------
    def _iter_internal(self, dataset=None):
        '''
        Generates pairs of labels and the stored dataset sliced by their fancy indices.
        Right now, this is only called by categorical. Groupby has a faster way of return dataset slices.
        '''
        self.grouping.pack_by_group()
        igroup = self.grouping.iGroup
        ifirstgroup = self.grouping.iFirstGroup
        ncountgroup = self.grouping.nCountGroup
        for i in range(self.grouping.unique_count):
            key = self.key_from_bin(i)
            first=ifirstgroup[i+1]
            last=first + ncountgroup[i+1]
            fancy_idx = igroup[first:last]
            if dataset is None:
                yield key, fancy_idx
            else:
                yield key, dataset[fancy_idx,:]

    #---------------------------------------------------------------
    def _iter_internal_contiguous(self):
        '''
        Sorts the data by group to create contiguous memory.
        Returns key + dataset view of key's rows for each group.
        '''

        self.grouping.pack_by_group()
        sortidx = self.grouping.iGroup
        ifirst = self.grouping.iFirstGroup[1:]
        ncountgroup = self.grouping.nCountGroup[1:]

        # perform a sort up front so the dataset can be sliced contiguously
        cds = self._dataset[sortidx,:]

        # the original columns, to make the views (no copies will be made)
        full_columns = list(cds.values())

        # the lists in this tuple will change
        item_tup = cds._all_items.get_dict_values()
        ncols = len(item_tup)

        # iterate over every group
        for i, glen in enumerate(ncountgroup):
            start = ifirst[i]
            end = start + glen

            # change the array slice
            for ci in range(ncols):
                item_tup[ci][0] = full_columns[ci][start:end]
            cds._nrows = glen
            yield self.key_from_bin(i), cds

    #---------------------------------------------------------------
    def get_groupings(self, filter=None):
        '''
        Parameters
        ----------
        filter : ndarray of bools, optional
            pass in a boolean filter

        Returns
        -------
        dict containing ndarrays calculated in ``pack_by_group()``.
            iGroup - the fancy indices for all groups, sorted by group. see iFirstGroup and nCountGroup for how to walk this.
            iFirstGroup - first index for each group in the igroup array. the first index is invalid
            nCountGroup - count for each unique group. the first count in this array is the invalid count.
        '''
        self.grouping.pack_by_group(filter=filter, mustrepack=True)
        return_dict = {'iGroup' : self.grouping.iGroup, 'iFirstGroup': self.grouping.iFirstGroup, 'nCountGroup':self.grouping.nCountGroup}
        return return_dict

    #---------------------------------------------------------------
    @property
    def first_fancy(self):
        '''
        Return a fancy index mask of the first occurrence

        Notes
        -----
        NOTE: not optimized for groupby which has grouping.ikey always set
        NOTE: categorical needs to lazy evaluate ikey

        Examples
        --------
        >>> c = rt.Cat(['this','this','that','that','this'])
        >>> c.first
        FastArray([0, 2])

        >>> c=Cat(['this','this','that','that','this'], ordered=False)
        >>> c.first
        FastArray([2, 0])
        '''
        # note, cache this value?
        # fancy index
        self.grouping.pack_by_group()
        return self.grouping.iGroup[self.grouping.iFirstGroup[1:]]

    #---------------------------------------------------------------
    @property
    def first_bool(self):
        '''
        Return a boolean mask of the first occurrence.

        Examples
        --------
        >>> c = rt.Cat(['this','this','that','that','this'])
        >>> c.first_bool
        FastArray([ True, False,  True, False, False])
        '''
        # boolean mask set to False
        fancy = self.first_fancy
        result=zeros_like(self.grouping.iGroup,dtype='?')

        # set boolean mask to True for only the first occurrence
        result[fancy]=True
        return result

    #---------------------------------------------------------------
    def _possibly_transform(self, gb_ds, label_keys=None, **kwargs):
        '''
        Called after a reduce operation to possibly re-expand back.
        Check transform flag.
        '''
        transform = kwargs.get("transform", False)

        # check if transform was called earlier
        if getattr(self, '_transform', False) or transform:
            ikey=self.grouping.ikey
            showfilter = kwargs.get("showfilter", False)
            if not showfilter and self.grouping.base_index == 1:
                ikey = ikey - 1
            # use fancy indexing to pull the values from the cells, back to original array
            newds= { }
            isort = None

            # a two key groupby (not gbu) often has display sort turned on
            if hasattr(self,'_sort_display'):
                if self._sort_display and self.grouping.Ordered is True:
                    # transform will put numbers back in original order
                    isort = rc.ReverseShuffle(self.isortrows)
                    ikey = isort[ikey]

            # no need to re-expand the labels keys or return them
            for colname,arr in gb_ds.items():
                if colname not in label_keys:
                    newds[colname] = arr[ikey]

            # turn transform back off in case used again
            self._transform = False
            return TypeRegister.Dataset(newds)
        return gb_ds

    #---------------------------------------------------------------
    def apply_reduce(self, userfunc, *args, dataset=None, label_keys=None, nokeys=False, func_param=None, dtype=None, transform=False, **kwargs):
        '''
        GroupByOps:apply_reduce calls Grouping:apply_reduce

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
        Grouping apply_reduce (for Categorical, groupby, accum2)
        For every column of data to be computed:
            The userfunc will be called back per group as a single array.  The order of the groups is either:
                1) Order of first apperance (when coming from a hash)
                2) Lexigraphical order (when ``lex=True`` or a Categorical with ordered=True)

        The function passed to apply must take an array as its first argument and return back a single scalar value.

        Examples
        --------
        From a Dataset groupby:
        >>> ds.gb(['Symbol'])['TradeSize'].apply_reduce(np.sum)

        From an existing categorical:

        >>> ds.Symbol.apply_reduce(np.sum, ds.TradeSize)

        Create your own with forced dtype:

        >>> def mycumprodsum(arr):
        >>>     return arr.cumprod().sum()
        >>> ds.Symbol.apply_reduce(mycumprodsum, ds.TradeSize, dtype=np.float32)
        '''
        if not callable(userfunc):
            raise TypeError(f'the first argument to apply_reduce must be callable not type {type(userfunc)!r}')
        args, kwargs, origdict, tups = self._pop_gb_data('apply_reduce'+type(self).__name__, userfunc, *args, **kwargs, dataset=dataset)

        # accum2 does not want any keys, it will set nokeys to True
        if label_keys is None and not nokeys:
            label_keys = self.gb_keychain

        #NOTE: apply_helper will take a filter= and use it
        result= self.grouping.apply_helper(True, origdict, userfunc, *args, tups=tups, label_keys=label_keys, func_param=func_param, dtype=dtype, **kwargs)
        if transform:
            kwargs['transform']=True
            return self._possibly_transform(result, label_keys=label_keys.keys(), **kwargs)
        else:
            return result

    #---------------------------------------------------------------
    def apply_nonreduce(self, userfunc, *args, dataset=None, label_keys=None, func_param=None, dtype=None, **kwargs):
        '''
        GroupByOps:apply_nonreduce calls Grouping:apply_reduce

        Parameters
        ----------
        userfunc : callable
            A callable that takes a contiguous array as its first argument, and returns a scalar.
            In addition the callable may take positional and keyword arguments.
        args
            used to pass in columnar data from other datasets
        dataset : None
            User may pass in an entire dataset to compute.
        label_keys : None.
            Not supported, will use the existing groupby keys as labels.
        dtype: str or np.dtype, optional
            Change to a numpy dtype to return an array with that dtype. Defaults to None.
        kwargs
            Optional positional and keyword arguments to pass to `userfunc`

        Notes
        -----
        Grouping apply_reduce (for Categorical, groupby, accum2)
        For every column of data to be computed:
            The userfunc will be called back per group as a single array.  The order of the groups is either:
                1) Order of first apperance (when coming from a hash)
                2) Lexigraphical order (when lex=True or a Categorical with ordered=True)

        The function passed to apply must take an array as its first argument and return back a single scalar value.

        Examples
        --------
        From a Dataset groupby:

        >>> ds.gb(['Symbol'])['TradeSize'].apply_reduce(np.sum)

        From an existing categorical:

        >>> ds.Symbol.apply_reduce(np.sum, ds.TradeSize)

        Create your own with forced dtype:

        >>> def mycumprodsum(arr):
        >>>     return arr.cumprod().sum()
        >>> ds.Symbol.apply_reduce(mycumprodsum, ds.TradeSize, dtype=np.float32)
        '''
        if not callable(userfunc):
            raise TypeError(f'the first argument to apply_nonreduce must be callable not type {type(userfunc)!r}')

        args, kwargs, origdict, tups = self._pop_gb_data('apply_nonreduce'+type(self).__name__, userfunc, *args, **kwargs, dataset=dataset)
        return self.grouping.apply_helper(False, origdict, userfunc, *args, tups=tups, label_keys=self.gb_keychain, func_param=func_param, dtype=dtype, **kwargs)

    #---------------------------------------------------------------
    def apply(self, userfunc, *args, dataset=None, label_keys=None, **kwargs):
        '''
        GroupByOps:apply calls Grouping:apply

        Parameters
        ----------
        userfunc : callable
            userfunction to call
        dataset: None
        label_keys: None
        '''
        # pop inplace data args first, put in dataset kwarg (might use groupby's stored dataset)
        if not callable(userfunc):
            raise TypeError(f'the first argument to apply must be callable not type {type(userfunc)!r}')

        args, kwargs, origdict,tups = self._pop_gb_data('apply'+type(self).__name__, userfunc, *args, **kwargs, dataset=dataset)
        result= self.grouping.apply(origdict, userfunc, *args, tups=tups, label_keys=self.gb_keychain, **kwargs)
        return self._possibly_transform(result, label_keys=self.gb_keychain.keys(), **kwargs)

    #---------------------------------------------------------------
    def _keys_as_list(self):
        gbkeys = self.grouping.uniquedict
        # return tuple of column values for multikey
        if len(gbkeys) > 1:
            return list(zip(*gbkeys.values()))
        return list(gbkeys.values())[0]

    #---------------------------------------------------------------
    def _calculate_all(self, funcNum, *args, func_param=0, gbkeys=None, isortrows=None, **kwargs):
        raise TypeError("_calculate_all should have been overriden!")


    #---------------------------------------------------------------
    @staticmethod
    def contains_np_arrays(container):
        '''
        Check to see if all items in a list-like container are numpy arrays.
        '''
        has_np = False
        if len(container) > 0:
            container_instance = [isinstance(item, np.ndarray) for item in container]
            if all(container_instance):
                has_np = True
        return has_np


    # ------------------------------------------------------------
    @classmethod
    def get_header_names(cls, columns, default='col_'):

        #---------------------------------------------------------------
        def get_array_name(arr, default, i):
            name = None
            try:
                name = arr.get_name()
            except:
                pass

            if name is None:
                return default+str(i)
            return name

        if isinstance(columns, dict):
            final_headers = list(columns.keys())
        else:
            # user friendly names for fast arrays if present
            headers = [ get_array_name(c, default, i) for i, c in enumerate(columns) ]

            # make sure there are no conflicts in friendly names, fix them up (columns only, shouldn't be too slow)
            # TODO: find a faster way of doing this
            unique_dict = {}
            final_headers = []
            for name in headers:
                new_name = name
                if name in unique_dict:
                    counter = unique_dict[name]
                    new_name = name+str(counter)

                    # make sure name+number is not also in the dict
                    while(new_name in headers):
                        counter+=1
                        new_name = name+str(counter)

                    # adjust the counter for that name
                    unique_dict[name]=counter+1
                else:
                    unique_dict[name]=1
                final_headers.append(new_name)

        return final_headers

    # ------------------------------------------------------------
    def _pop_gb_data(self, calledfrom, userfunc, *args, **kwargs):
        """
        Pop the groupby data from the args and keyword args, possibly combining.
        Avoid repeating this step when the data doesn't change.

        Parameters
        ----------
        calledfrom : {'apply_reduce', 'apply_nonreduce', 'apply', 'agg'}
        userfunc : callable or int (function number)

        Returns
        -------
        4 return values:
        any user arguments
        the kwargs (with 'dataset' removed)
        the dictionary of numpy arrays to operarte on
        tups: 0 or 1 or 2 depending on whether the first argument was a tuple of arrays

        See Also
        --------
        GroupByOps.agg()
        """
        kwargs['dataset'], user_args, tups = self._prepare_gb_data(calledfrom, userfunc, *args, **kwargs)

        origdict = kwargs.pop('dataset')
        return user_args, kwargs, origdict, tups

    #---------------------------------------------------------------
    def _prepare_gb_data(self, calledfrom, userfunc, *args, dataset=None, **kwargs):
        '''
        Parameters
        ----------
        calledfrom: 'Accum2', 'Categorical','GroupBy','apply_reduce','apply_nonreduce','apply','agg'
        userfunc: a callable function or a function number
        args or dataset must be present (both also allowed)
            if just args: make a dictionary from that
            if just dataset: make dictionary
            if both: make a new dataset, then make a dictionary from that
            if neither: error

            from Grouping, normally just a dataset
            from Categorical, normally just args (but user can use kwarg 'dataset' to supply one)

        This routine normalizes input from Grouping, Accum2, Categorical

        GroupBy defaults to use the _dataset variable that it sets after being constructed from a dataset.
        Accum2 and Categorical default to using input data for the calculation methods.
        Accum2 and Categorical can also set _dataset just like Groupby. See Dataset.accum2 and Dataset.cat for
        examples.
        If a _dataset has been set, no input data is required for the calculation methods.

        internal function to parse argument and search for numpy arrays

        Returns
        -------
        a dictionary of arrays to be used as input to many groupby algorithms
        user_args if any (the first argument might be removed)
        tups: 0 or or 2.  Will be set to T> 0 if the first argument is a tuple

        Raises
        ------
        ValueError

        '''

        # Autodetect what is in args
        # if dataset exists (from a groupby) and the first param in args
        #                   is a LIST of arrays, then those are additional arrays to add
        #                 **a list of scalars, then that is an argument to the function
        #                 **an array: argument to function if 'apply*'
        #                   a tuple: additional array constants to add together
        #                   not a list or tuple, then those are arguments to the function
        #
        # if dataset does not exist, the first param
        #                   a list of arrays: additional arrays to add
        #                   an array: a single array to add
        #                   a list of scalars: a single array to add
        #           possibly second parameter:
        #                   a tuple: additional array constants to add together
        # in the new code, any args after the first argument MUST be user_args
        # to pass in multiple arrays to operate a function on, pass them in as a list [arr1, arr2, arr3]

        ds = None
        zip_dict = False
        tups = 0
        user_args = args
        first_arg = None

        if len(args) >= 1:
            user_args = args[1:]
            # pop off first argument
            first_arg=args[0]
            if isinstance(first_arg, (list, tuple)):
                if len(first_arg) ==0:
                    # backout
                    first_arg = None
                    user_args = args
                if isinstance(first_arg, tuple):
                    # user_args has moved over
                    tups=1

        if first_arg is not None:
            if isinstance(first_arg, (list, tuple)):
                first_element = first_arg[0]
                if isinstance(first_element, np.ndarray):
                    #print('first_arg was list/tuple of ndarrays')
                    if len(first_arg)>=1 and tups > 0 and (self._dataset is not None) and calledfrom.endswith('GroupBy'):
                        #print("special mode", dataset, ds, self._dataset)
                        #first_arg = None
                        #user_args = args
                        tups=2
                        #zip_dict = True
                    else:
                        zip_dict = True
                elif isinstance(first_element, list):
                    # list of lists?  convert the lists to arrays
                    first_arg = [np.asarray(v) for v in first_arg]
                    zip_dict = True
                elif isinstance(first_element, TypeRegister.Dataset):
                    #print('first_element was single dataset')
                    ds = {name:arr for name,arr in first_element.items()}

                elif isinstance(first_element, dict):
                    # shallow copy, might be modified by key later
                    ds = first_element.copy()
                elif np.isscalar(first_element):
                    if dataset is None or not calledfrom.startswith('apply'):
                        zip_dict = True
                        # assume a list or tuple of scalars
                        first_arg = [np.asarray(first_arg)]

            elif isinstance(first_arg, np.ndarray):
                #print('first_arg was single ndarray')
                if dataset is None or not calledfrom.startswith('apply'):
                    zip_dict = True
                    first_arg = [first_arg]
                else:
                    # assume userfunc argument
                    pass

            elif isinstance(first_arg, TypeRegister.Dataset):
                #print('first_arg was single dataset')
                ds = {name:arr for name,arr in first_arg.items()}

            elif isinstance(first_arg, dict):
                # shallow copy, might be modified by key later
                ds = first_arg.copy()

            # check for a tuple passed as second argument
            # check if we ate the first argument
            if (ds is not None or zip_dict) and tups ==0:
                # move over one argument, we ate it
                args= args[1:]
                user_args = args
                if len(user_args) > 0:
                    # check for tuples
                    # pop off first argument
                    addl_arg=user_args[0]
                    if isinstance(addl_arg, tuple) and len(addl_arg) > 0:
                        tups=1
                        first_element = addl_arg[0]
                        if isinstance(first_element, np.ndarray):
                            # passing in array constants after a list or dict or dataset
                            tups=2

        if ds is None and not zip_dict and first_arg is not None:
            # recombine
            user_args = args

        if zip_dict:
            headers = self.get_header_names(first_arg)
            ds = dict(zip(headers, first_arg))

        if dataset is not None:
            final_dict = {name:col for name,col in dataset.items()}
            # in special mode, remove the arrays in the user arguments names
            if tups ==2:
                # remove extra names
                for ua in user_args[0]:
                    try:
                        name = ua.get_name()
                        if name in final_dict:
                            # is this the same array?
                            if id(final_dict[name]) == id(ua):
                                del final_dict[name]
                    except Exception:
                        pass

            if ds is not None:
                # combine dataset with extra data
                for name, col in ds.items():
                    # if calling from a groupby, and the user created a tuple then we are in tups==2 mode
                    # in this mode, the arguments are constants that are passed in for each column
                    # so we want to remove the constant arrays 
                    alreadyexists = name in final_dict
                    if alreadyexists:
                        warnings.warn(f'Found conflicting items for name {name}. Using item from arguments.')
                    final_dict[name] = col
        else:
            # extra data only, already a dict
            if ds is None:
                final_dict = self._dataset
            else:
                final_dict = ds

        # no data found
        if final_dict is None:
            funcname = CPP_REVERSE_TABLE.get(userfunc, None)
            if funcname is None:
                try:
                    funcname = userfunc.__name__
                except Exception:
                    pass

            if funcname is None:
                if np.isscalar(funcname):
                    funcname = str(userfunc)
                else:
                    funcname = 'somefunc'

            errorstring = f"Useable data for the function {calledfrom!r} has not been specified in {args!r}. Pass in array data to operate on.\n"
            if calledfrom.startswith('apply'):
                errorstring+=f"For example: call .{calledfrom}({funcname}, array_data)"
            else:
                errorstring+=f"For example: call {calledfrom}.{funcname}(array_data)"

            raise ValueError(errorstring)

        return final_dict, user_args, tups

    #---------------------------------------------------------------
    def aggregate(self, func):
        return self.agg(func)

    #---------------------------------------------------------------
    def _get_agg_func(self, item):
        """
        Translates user input into name and method for groupby aggregations.

        Parameters
        ----------
        item : str or function
            String or supported numpy math function. See GroupByOps.AggNames.

        Returns
        -------
        name : str
            Lowercase name for aggregation function.
        func : function
            GroupByOps method.

        """
        if callable(item):
            # pull function name
            item = item.__name__
            # in case we need to route a numpy func to a different name
            item = self.NumpyAggNames.get(item, item)
        if item in self.AggNames:
            return item, getattr(self.__class__, item)
        raise ValueError(f"{item} is not a valid function to aggregate.")

    #---------------------------------------------------------------
    def agg(self, func=None, *args, dataset=None, **kwargs):
        """
        Parameters
        ----------
        func : callable, string, dictionary, or list of string/callables
            Function to use for aggregating the data. If a function, must either
            work when passed a DataFrame or when passed to DataFrame.apply. For
            a DataFrame, can pass a dict, if the keys are DataFrame column names.

            Accepted Combinations are:

            - string function name
            - function
            - list of functions
            - dict of column names -> functions (or list of functions)

        Returns
        -------
        aggregated : Multiset

        Notes
        -----
        Numpy functions mean/median/prod/sum/std/var are special cased so the
        default behavior is applying the function along axis=0

        Examples
        --------
        Aggregate these functions across all columns

        >>> gb.agg(['sum', 'min'])
                    A         B         C
        sum -0.182253 -0.614014 -2.909534
        min -1.916563 -1.460076 -1.568297

        Different aggregations per column

        >>> gb.agg({'A' : ['sum', 'min'], 'B' : ['min', 'max']})
                    A         B
        max       NaN  1.514318
        min -1.916563 -1.460076
        sum -0.182253       NaN

        >>> gb.agg({'C': np.sum, 'D': lambda x: np.std(x,ddof=1)})

        """
        if dataset is None:
            try:
                dataset=self._dataset
            except Exception:
                pass

        # tups will be false since we pass in a list as first argument
        args, kwargs, data, tups = self._pop_gb_data('agg', func, [*args], **kwargs, dataset=dataset)

        # put back in dataset that got popped because kwargs is passed to aggfunc
        kwargs['dataset'] = data

        if func is None or len(func) == 0:
            raise ValueError("The first argument to the agg function is a dictionary or list, such as gb.agg({'data':np.sum})")

        # create blank multiset class
        multiset = TypeRegister.Multiset({})

        if isinstance(func,str):
            func = [func]

        if isinstance(func,list):
            # run through list -- we do not check for duplicates
            for item in func:
                name, aggfunc = self._get_agg_func(item)
                caps=name.capitalize()
                multiset[caps] = aggfunc(self, *args, **kwargs)

        elif isinstance(func, dict):
            # two passes, build dictionary first
            func_dict = {}
            for col_name, operations in func.items():
                if col_name in data.keys():
                    if not isinstance(operations, (list, tuple)):
                        operations = [operations]

                    if isinstance(operations,(list,tuple)):
                        for op in operations:
                            name, aggfunc = self._get_agg_func(op)
                            f_list = func_dict.setdefault(aggfunc, [])
                            f_list.append(col_name)

                else:
                    raise ValueError(f"{col_name} is not a valid column name")

            # second pass, loop through dictionary
            for aggfunc, col_list in func_dict.items():
                name = aggfunc.__name__.capitalize()
                multiset[name] = aggfunc(self, *args, col_idx=col_list, **kwargs)

        multiset._gbkeys = self.gb_keychain.gbkeys
        return multiset

    #---------------------------------------------------------------
    def null(self, showfilter=False):
        '''
        Performs a reduced no-op.  No operation is performed.

        Parameters
        ----------
        showfilter: bool, False

        Returns
        -------
        Dataset with grouping keys.  No operation is performed.

        Examples
        --------
        >>> rt.Cat(np.random.choice(['SPY','IBM'], 100)).null(showfilter=True)
        '''
        return self.grouping._finalize_dataset(TypeRegister.Dataset({}),self.gb_keychain, None, addkeys=True, showfilter=showfilter)

    #---------------------------------------------------------------
    def count_uniques(self, *args, **kwargs):
        """
        Compute unique count of group
        
        Returns
        -------
        Dataset with grouped key plus the unique count for each column by group.

        Examples
        --------
        >>> N = 17; np.random.seed(1)
        >>> ds =Dataset( 
                dict(
                    Symbol = Cat(np.random.choice(['SPY','IBM'], N)),
                    Exchange = Cat(np.random.choice(['AMEX','NYSE'], N)),
                    TradeSize = np.random.choice([1,5,10], N),
                    TradePrice = np.random.choice([1.1,2.2,3.3], N),
                    ))
        >>> ds.cat(['Symbol','Exchange']).count_uniques()
        *Symbol   *Exchange   TradeSize   TradePrice
        -------   ---------   ---------   ----------
        IBM       NYSE                2            2
        .         AMEX                2            3
        SPY       AMEX                3            2
        .         NYSE                1            2
        """
        origdict, user_args, tups = self._prepare_gb_data('count_uniques', None, *args, **kwargs)

        label_keys = self.gb_keychain
        g=self.grouping

        # get way to make groups contiguous
        igroup=g.igroup
        cutoffs =g.ncountgroup.cumsum(dtype=np.int64)[1:]
        newdict={}
        for colname, arr in origdict.items():
            gbk= label_keys.gbkeys
            if colname not in gbk:
                ifirstkey = groupbyhash(arr[igroup], cutoffs=cutoffs)['iFirstKey'][1]
                # the cutoffs will generate iFirstKey cutoffs that help us determine the unique counts
                result=ifirstkey.diff()
                result[0]=ifirstkey[0]
                newdict[colname]= result
        return g._finalize_dataset(newdict,label_keys, None, addkeys=True, **kwargs)

    #---------------------------------------------------------------
    def count(self):
        """Compute count of group"""
        raise ValueError("subclass must take over count")
        # make a new dataset with the same number of rows
        #origdict = self._dataset.as_ordered_dictionary()
        #return self.grouping.count(gbkeys, isortrows)

    #---------------------------------------------------------------
    def sum(self, *args, **kwargs):
        """Compute sum of group"""
        return self._calculate_all(GB_FUNCTIONS.GB_SUM, *args, **kwargs)

    #---------------------------------------------------------------
    def mean(self, *args, **kwargs):
        """
        Compute mean of groups
        """
        return self._calculate_all(GB_FUNCTIONS.GB_MEAN, *args, **kwargs)

    #---------------------------------------------------------------
    def mode(self, *args, **kwargs):
        """
        Compute mode of groups (auto handles nan)
        """
        return self._calculate_all(GB_FUNCTIONS.GB_MODE, *args, **kwargs)

    #---------------------------------------------------------------
    def trimbr(self, *args, **kwargs):
        """
        Compute trimmed mean br of groups (auto handles nan)
        """
        return self._calculate_all(GB_FUNCTIONS.GB_TRIMBR, *args, **kwargs)

    #---------------------------------------------------------------
    def nanmean(self, *args, **kwargs):
        """Compute mean of group, excluding missing values"""
        return self._calculate_all(GB_FUNCTIONS.GB_NANMEAN, *args, **kwargs)

    #---------------------------------------------------------------
    def nanmedian(self, *args, **kwargs):
        """Compute median of group, excluding missing values"""
        return self._calculate_all(GB_FUNCTIONS.GB_MEDIAN, *args, **kwargs)

    #---------------------------------------------------------------
    def nanmin(self, *args, **kwargs):
        """Compute min of group, excluding missing values"""
        return self._calculate_all(GB_FUNCTIONS.GB_NANMIN, *args, **kwargs)

    #---------------------------------------------------------------
    def nanmax(self, *args, **kwargs):
        """Compute max of group, excluding missing values"""
        return self._calculate_all(GB_FUNCTIONS.GB_NANMAX, *args, **kwargs)

    #---------------------------------------------------------------
    def nansum(self, *args, **kwargs):
        """Compute sum of group, excluding missing values"""
        return self._calculate_all(GB_FUNCTIONS.GB_NANSUM, *args, **kwargs)

    #---------------------------------------------------------------
    def min(self, *args, **kwargs):
        """Compute min of group"""
        return self._calculate_all(GB_FUNCTIONS.GB_MIN, *args, **kwargs)

    #---------------------------------------------------------------
    def max(self, *args, **kwargs):
        """Compute max of group"""
        return self._calculate_all(GB_FUNCTIONS.GB_MAX, *args, **kwargs)

    #---------------------------------------------------------------
    def first(self, *args, **kwargs):
        """First value in the group"""
        return self._calculate_all(GB_FUNCTIONS.GB_FIRST, *args, **kwargs)

    #---------------------------------------------------------------
    def last(self, *args, **kwargs):
        """Last value in the group"""
        return self._calculate_all(GB_FUNCTIONS.GB_LAST, *args, **kwargs)

    #---------------------------------------------------------------
    def median(self, *args, **kwargs):
        """
        Compute median of groups
        For multiple groupings, the result will be a MultiSet
        """
        return self._calculate_all(GB_FUNCTIONS.GB_MEDIAN, *args, **kwargs)

    #---------------------------------------------------------------
    def std(self, *args, **kwargs):
        """
        Compute standard deviation of groups
        For multiple groupings, the result will be a MultiSet

        Parameters
        ----------
        ddof : integer, default 1
            degrees of freedom
        """
        return self._calculate_all(GB_FUNCTIONS.GB_STD, *args, **kwargs)

    #---------------------------------------------------------------
    def nanstd(self, *args, **kwargs):
        """
        Compute standard deviation of groups, excluding missing values
        """

        return self._calculate_all(GB_FUNCTIONS.GB_NANSTD, *args, **kwargs)

    #---------------------------------------------------------------
    def var(self, *args, **kwargs):
        """
        Compute variance of groups
        For multiple groupings, the result will be a MultiSet

        Parameters
        ----------
        ddof : integer, default 1
            degrees of freedom
        """
        return self._calculate_all(GB_FUNCTIONS.GB_VAR, *args, **kwargs)

    #---------------------------------------------------------------
    def nanvar(self, *args, **kwargs):
        """
        Compute variance of groups, excluding missing values
        For multiple groupings, the result will be a MultiSet
        """
        return self._calculate_all(GB_FUNCTIONS.GB_NANVAR, *args, **kwargs)

    #---------------------------------------------------------------
    def rolling_sum(self, *args, window=3, **kwargs):
        """rolling sum for each group

        Parameters
        ----------
        window: optional, window size, defaults to 3

        Returns
        -------
        Dataset same rows as original dataset
        """
        return self._calculate_all(GB_FUNCTIONS.GB_ROLLING_SUM, *args, func_param=(window), **kwargs)

    #---------------------------------------------------------------
    def rolling_nansum(self, *args, window=3, **kwargs):
        """rolling nan sum for each group

        Parameters
        ----------
        window: optional, window size, defaults to 3

        Returns
        -------
        Dataset same rows as original dataset
        """
        return self._calculate_all(GB_FUNCTIONS.GB_ROLLING_NANSUM, *args, func_param=(window), **kwargs)

    #---------------------------------------------------------------
    def rolling_mean(self, *args, window=3, **kwargs):
        """rolling mean for each group

        Parameters
        ----------
        window: optional, window size, defaults to 3

        Returns
        -------
        Dataset same rows as original dataset
        """
        return self._calculate_all(GB_FUNCTIONS.GB_ROLLING_MEAN, *args, func_param=(window), **kwargs)

    #---------------------------------------------------------------
    def rolling_nanmean(self, *args, window=3, **kwargs):
        """rolling nan mean for each group

        Parameters
        ----------
        window: optional, window size, defaults to 3

        Returns
        -------
        Dataset same rows as original dataset
        """
        return self._calculate_all(GB_FUNCTIONS.GB_ROLLING_NANMEAN, *args, func_param=(window), **kwargs)


    #---------------------------------------------------------------
    def rolling_count(self, *args, window=3, **kwargs):
        """rolling count for each group

        Parameters
        ----------
        window: optional, window size, defaults to 3

        Returns
        -------
        Dataset same rows as original dataset
        """
        return self._calculate_all(GB_FUNCTIONS.GB_ROLLING_COUNT, *args, func_param=(window), **kwargs)

    #---------------------------------------------------------------
    def rolling_shift(self, *args, window=1, **kwargs):
        """rolling shift for each group

        Parameters
        ----------
        window: optional, window size, defaults to 1
        windows can be negative

        Returns
        -------
        Dataset same rows as original dataset
        """
        return self._calculate_all(GB_FUNCTIONS.GB_ROLLING_SHIFT, *args, func_param=(window), **kwargs)


    #---------------------------------------------------------------
    def rolling_diff(self, *args, window=1, **kwargs):
        """rolling diff for each group

        Parameters
        ----------
        window: optional, window size, defaults to 1

        Returns
        -------
        Dataset same rows as original dataset
        """
        return self._calculate_all(GB_FUNCTIONS.GB_ROLLING_DIFF, *args, func_param=(window), **kwargs)

    #---------------------------------------------------------------
    def cumcount(self, *args, ascending=True, **kwargs):
        """rolling count for each group
        Number each item in each group from 0 to the length of that group - 1.

        Parameters
        ----------
        ascending : bool, default True

        Returns
        -------
        A single array, same size as the original grouping dict/categorical.
        If a filter was applied, integer sentinels will appear in those slots.
        """
        param=1

        if not ascending:
            param=-1

        # cumcount doesn't need an origdict, pass it in empty
        result= self.grouping._calculate_all({}, GB_FUNCTIONS.GB_ROLLING_COUNT, func_param=(param), keychain=self.gb_keychain, **kwargs)
        return result

    #---------------------------------------------------------------
    def cumsum(self, *args, filter = None, reset_filter=None, **kwargs):
        """Cumulative sum for each group

        Parameters
        ----------
        filter: optional, boolean mask array of included
        reset_filter: optional, boolean mask array

        Returns
        -------
        Dataset same rows as original dataset
        """
        if filter is None:
            filter = self._filter

        return self._calculate_all(GB_FUNCTIONS.GB_CUMSUM, *args, func_param=(0.0, None, filter, reset_filter),**kwargs)

    #---------------------------------------------------------------
    def cumprod(self, *args, filter = None, reset_filter=None, **kwargs):
        """Cumulative product for each group

        Parameters
        ----------
        filter: optional, boolean mask array of included
        reset_filter: optional, boolean mask array

        Returns
        -------
        Dataset same rows as original dataset
        """
        if filter is None:
            filter = self._filter

        return self._calculate_all(GB_FUNCTIONS.GB_CUMPROD, *args, func_param=(0.0, None, filter, reset_filter),**kwargs)


    #---------------------------------------------------------------
    def findnth(self, *args, filter = None, **kwargs):
        """FindNth

        Parameters
        ----------
        filter: optional, boolean mask array of included
        TAKES NO ARGUMENTS -- operates on bin

        Returns
        -------
        Dataset same rows as original dataset
        """
        if filter is None:
            filter = self._filter

        return self._calculate_all(GB_FUNCTIONS.GB_FINDNTH, *args, func_param=(0.0, None, filter, None),**kwargs)

    #---------------------------------------------------------------
    def _ema_op(self, function, *args, time=None, decay_rate = 1.0, filter = None, reset_filter=None, **kwargs):
        """
        Ema base function for time based ema functions

        Formula
        -------
        grp loops over each item in a groupby group
            i loops over eachitem in the original dataset
                Output[i] = <some formula>

        Parameters
        ----------
        time: float or int array used to calculate time difference
        decay_rate: see formula, used a half life
        filter: optional, boolean mask array of included
        reset_filter: optional, boolean mask array

        Returns
        -------
        Dataset same rows as original dataset
        """
        if time is None:
            raise ValueError("The 'time' kwarg is required when calling ema functions")

        if filter is None:
            filter = self._filter

        if filter is not None:
            if len(time) != len(filter):
                raise ValueError(f"The 'time' array length {len(time)} must match the length of the filter")

        return self._calculate_all(function, *args, func_param=(decay_rate, time, filter, reset_filter), **kwargs)

    #---------------------------------------------------------------
    def ema_decay(self, *args, time=None, decay_rate = None, filter = None, reset_filter=None, **kwargs):
        """
        Ema decay for each group

        Formula
        -------
        grp loops over each item in a groupby group
            i loops over eachitem in the original dataset
                Output[i] = Column[i] + LastEma[grp] * exp(-decay_rate * (Time[i] - LastTime[grp]));
                LastEma[grp] = Output[i]
                LastTime[grp] = Time[i]

        Parameters
        ----------
        time: float or int array used to calculate time difference
        decay_rate: see formula, used a half life
        filter: optional, boolean mask array of included
        reset_filter: optional, boolean mask array

        Returns
        -------
        Dataset same rows as original dataset

        Example
        -------
        >>> aapl
        #    delta     sym       org    time
        -   ------     ----   ------   -----
        0    -3.11     AAPL    -3.11   25.65
        1   210.54     AAPL   210.54   38.37
        2    49.97     AAPL    42.11   41.66

        >>> np.log(2)/(1e3*100)
        6.9314718055994526e-06

        >>> aapl.groupby('sym')['delta'].ema_decay(time=aapl.time, decay_rate=np.log(2)/(1e3*100))[0]
        FastArray([ -3.11271882, 207.42784495, 257.39155897])
        """
        if decay_rate is None:
            raise ValueError("ema_decay function requires a kwarg 'decay_rate' floating point value as input")

        return self._ema_op(GB_FUNCTIONS.GB_EMADECAY, *args, time=time,  decay_rate=decay_rate, filter=filter, reset_filter=reset_filter, **kwargs)

    #---------------------------------------------------------------
    def ema_normal(self, *args, time=None, decay_rate = None, filter = None, reset_filter=None, **kwargs):
        """
        Ema decay for each group

        Formula
        -------
        grp loops over each item in a groupby group
           i loops over eachitem in the original dataset
               decayedWeight = exp(-decayRate * (Time[i] - LastTime[grp]));
               LastEma[grp] = Column[i] * (1 - decayedWeight) + LastEma[grp] * decayedWeight
               Output[i] = LastEma[grp]
               LastTime[grp] = Time[i]

        Parameters
        ----------
        time: float or int array used to calculate time difference
        decay_rate: see formula, used a half life (defaults to 1.0)
        filter: optional, boolean mask array of included
        reset_filter: optional, boolean mask array

        Returns
        -------
        Dataset same rows as original dataset

        Example
        -------
        >>> ds = rt.Dataset({'test': rt.arange(10), 'group2': rt.arange(10) % 3})
        >>> ds.normal = ds.gb('group2')['test'].ema_normal(decay_rate=1.0, time = rt.arange(10))['test']
        >>> ds.weighted = ds.gb('group2')['test'].ema_weighted(decay_rate=0.5)['test']
        >>> ds
        #   test   group2   normal   weighted
        -   ----   ------   ------   --------
        0      0        0     0.00       0.00
        1      1        1     1.00       1.00
        2      2        2     2.00       2.00
        3      3        0     2.85       1.50
        4      4        1     3.85       2.50
        5      5        2     4.85       3.50
        6      6        0     5.84       3.75
        7      7        1     6.84       4.75
        8      8        2     7.84       5.75
        9      9        0     8.84       6.38

        See Also
        --------
        ema_weighted
        ema_decay
        """
        if decay_rate is None:
            raise ValueError('ema_normal function requires a decay_rate floating point value')

        if time is None:
            raise ValueError('ema_normal function requires a time array.  Use the "time" kwarg')

        if not isinstance(time, np.ndarray):
            raise ValueError('ema_normal function requires a time numpy array.')

        # cannot support int16/uint16
        if time.dtype.num < 5:
            time = time.astype(np.int32)

        return self._ema_op(GB_FUNCTIONS.GB_EMANORMAL, *args, time=time,  decay_rate=decay_rate, filter=filter, reset_filter=reset_filter, **kwargs)

    #---------------------------------------------------------------
    def ema_weighted(self, *args, decay_rate = None, filter = None, reset_filter=None, **kwargs):
        """
        Ema decay for each group with constant decay value (no time parameter)

        Formula
        -------
        grp loops over each item in a groupby group
           i loops over eachitem in the original dataset
               LastEma[grp] = Column[i] * (1 - decay_rate) + LastEma[grp] * decay_rate
               Output[i] = LastEma[grp]

        Parameters
        ----------
        time: <not used>
        decay_rate: see formula, used a half life
        filter: optional, boolean mask array of included
        reset_filter: optional, boolean mask array

        Returns
        -------
        Dataset same rows as original dataset

        Example
        -------
        >>> ds = rt.Dataset({'test': rt.arange(10), 'group2': rt.arange(10) % 3})
        >>> ds.normal = ds.gb('group2')['test'].ema_normal(decay_rate=1.0, time=rt.arange(10))['test']
        >>> ds.weighted = ds.gb('group2')['test'].ema_weighted(decay_rate=0.5)['test']
        >>> ds
        #   test   group2   normal   weighted
        -   ----   ------   ------   --------
        0      0        0     0.00       0.00
        1      1        1     1.00       1.00
        2      2        2     2.00       2.00
        3      3        0     2.85       1.50
        4      4        1     3.85       2.50
        5      5        2     4.85       3.50
        6      6        0     5.84       3.75
        7      7        1     6.84       4.75
        8      8        2     7.84       5.75
        9      9        0     8.84       6.38

        See Also
        --------
        ema_normal
        ema_decay
        """
        if decay_rate is None:
            raise ValueError('ema_weighted function requires a decay_rate floating point value')

        # put in fake time array
        time_array = np.arange(self._dataset.shape[0])
        return self._ema_op(GB_FUNCTIONS.GB_EMAWEIGHTED, *args, time=time_array,  decay_rate=decay_rate, filter=filter, reset_filter=reset_filter, **kwargs)

    #-------------------------------------------------------
    def sem(self, **kwargs):
        """
        Compute standard error of the mean of groups
        For multiple groupings, the result index will be a MultiIndex

        Parameters
        ----------
        ddof : integer, default 1
            degrees of freedom
        """
        raise NotImplementedError
        #return self.std(ddof=ddof) / np.sqrt(self.count())

    #-------------------------------------------------------
    def ohlc(self, **kwargs):
        """
        Compute sum of values, excluding missing values
        For multiple groupings, the result index will be a MultiIndex
        """
        raise NotImplementedError
        #return self._apply_to_column_groupbys(
        #    lambda x: x._cython_agg_general('ohlc'))

    #-------------------------------------------------------
    def describe(self, **kwargs):
        raise NotImplementedError
        #self._set_group_selection()
        #result = self.apply(lambda x: x.describe(**kwargs))
        #if self.axis == 1:
        #    return result.T
        #return result.unstack()

    #-------------------------------------------------------
    def resample(self, rule, *args, **kwargs):
        """
        Provide resampling when using a TimeGrouper
        Return a new grouper with our resampler appended
        """
        raise NotImplementedError
        #from pandas.core.resample import get_resampler_for_grouping
        #return get_resampler_for_grouping(self, rule, *args, **kwargs)

    #-------------------------------------------------------
    def nth(self, *args, n=1, **kwargs):
        """
        Take the nth row from each group if `n` is an int, or a subset of rows if `n` is a list of ints.

        If dropna, will take the nth non-null row, dropna is either
        Truthy (if a Series) or 'all', 'any' (if a DataFrame);
        this is equivalent to calling dropna(how=dropna) before the
        groupby.

        Parameters
        ----------
        n : int or list of ints
            a single nth value for the row or a list of nth values
        dropna : None or str, optional
            apply the specified dropna operation before counting which row is
            the nth row. Needs to be None, 'any' or 'all'

        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5]}, columns=['A', 'B'])
        >>> g = df.groupby('A')
        >>> g.nth(0)
             B
        A
        1  NaN
        2  3.0

        >>> g.nth(1)
             B
        A
        1  2.0
        2  5.0

        >>> g.nth(-1)
             B
        A
        1  4.0
        2  5.0

        >>> g.nth([0, 1])
             B
        A
        1  NaN
        1  2.0
        2  3.0
        2  5.0

        Specifying ``dropna`` allows count ignoring NaN

        >>> g.nth(0, dropna='any')
             B
        A
        1  2.0
        2  3.0

        NaNs denote group exhausted when using dropna

        >>> g.nth(3, dropna='any')
            B
        A
        1 NaN
        2 NaN

        Specifying ``as_index=False`` in ``groupby`` keeps the original index.

        >>> df.groupby('A', as_index=False).nth(1)
           A    B
        1  1  2.0
        4  2  5.0
        """
        return self._calculate_all(GB_FUNCTIONS.GB_NTH, *args, func_param=(n), **kwargs)
        #raise NotImplementedError

    ##-------------------------------------------------------
    def diff(self, period=1, **kwargs):
        """rolling diff for each group

        Parameters
        ----------
        period: optional, period size, defaults to 1

        Returns
        -------
        Dataset same rows as original dataset
        """
        return self._calculate_all(GB_FUNCTIONS.GB_ROLLING_DIFF, tuple(), func_param=(period), **kwargs)

    #-------------------------------------------------------
    def ngroup(self, ascending=True, **kwargs):
        """
        Number each group from 0 to the number of groups - 1.
        This is the enumerative complement of cumcount.  Note that the
        numbers given to the groups match the order in which the groups
        would be seen when iterating over the groupby object, not the
        order they are first observed.

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from number of group - 1 to 0.

        Examples
        --------
        >>> df = pd.DataFrame({"A": list("aaabba")})
        >>> df
           A
        0  a
        1  a
        2  a
        3  b
        4  b
        5  a

        >>> df.groupby('A').ngroup()
        0    0
        1    0
        2    0
        3    1
        4    1
        5    0
        dtype: int64

        >>> df.groupby('A').ngroup(ascending=False)
        0    1
        1    1
        2    1
        3    0
        4    0
        5    1
        dtype: int64

        >>> df.groupby(["A", [1,1,2,3,2,1]]).ngroup()
        0    0
        1    0
        2    1
        3    3
        4    2
        5    0
        dtype: int64

        See also
        --------
        cumcount : Number the rows in each group.
        """
        raise NotImplementedError

        #self._set_group_selection()

        #index = self._selected_obj.index
        #result = Series(self.grouper.group_info[0], index)
        #if not ascending:
        #    result = self.ngroups - 1 - result
        #return result

    #-------------------------------------------------------
    def rank(self, method='average', ascending=True, na_option='keep',
             pct=False, axis=0, **kwargs):
        """
        Provides the rank of values within each group

        Parameters
        ----------
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
            * average: average rank of group
            * min: lowest rank in group
            * max: highest rank in group
            * first: ranks assigned in order they appear in the array
            * dense: like 'min', but rank always increases by 1 between groups
        method :  {'keep', 'top', 'bottom'}, default 'keep'
            * keep: leave NA values where they are
            * top: smallest rank if ascending
            * bottom: smallest rank if descending
        ascending : boolean, default True
            False for ranks by high (1) to low (N)
        pct : boolean, default False
            Compute percentage rank of data within each group

        Returns
        -----
        DataFrame with ranking of values within each group
        """
        raise NotImplementedError


    #-------------------------------------------------------
    def cummin(self, axis=0, **kwargs):
        """Cumulative min for each group"""
        raise NotImplementedError

    #-------------------------------------------------------
    def cummax(self, axis=0, **kwargs):
        """Cumulative max for each group"""
        raise NotImplementedError

    #-------------------------------------------------------
    def shift(self, window=1, **kwargs):
        """
        Shift each group by periods observations
        Parameters
        ----------
        window : integer, default 1 number of periods to shift
        periods: optional support, same as window
        """
        # support for pandas periods keyword
        window = kwargs.get('periods',window)
        return self._calculate_all(GB_FUNCTIONS.GB_ROLLING_SHIFT, tuple(), func_param=(window), **kwargs)

    #-------------------------------------------------------
    def head(self, n=5, **kwargs):
        """
        Returns first n rows of each group.

        Essentially equivalent to ``.apply(lambda x: x.head(n))``,
        except ignores `as_index` flag.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [1, 4], [5, 6]], columns=['A', 'B'])
        >>> df.groupby('A', as_index=False).head(1)
           A  B
        0  1  2
        2  5  6

        >>> df.groupby('A').head(1)
           A  B
        0  1  2
        2  5  6
        """
        raise NotImplementedError
        #self._reset_group_selection()
        #mask = self._cumcount_array() < n
        #return self._selected_obj[mask]

    #-------------------------------------------------------
    def tail(self, n=5, **kwargs):
        """
        Returns last n rows of each group
        Essentially equivalent to ``.apply(lambda x: x.tail(n))``,
        except ignores `as_index` flag.

        Examples
        --------
        >>> df = pd.DataFrame([['a', 1], ['a', 2], ['b', 1], ['b', 2]], columns=['A', 'B'])
        >>> df.groupby('A').tail(1)
           A  B
        1  a  2
        3  b  2

        >>> df.groupby('A').head(1)
           A  B
        0  a  1
        2  b  1
        """
        raise NotImplementedError
        #self._reset_group_selection()
        #mask = self._cumcount_array(ascending=False) < n
        #return self._selected_obj[mask]


#------------------------------------------------------------
#     cppnum     name:      (basic/packing,       func_frontend,             func_backend,          gb_function,  dtype,              return_full True/False)
#     -----     ------      -------------------   ------------------         ------------------     -----------   ----------------    ------------------
GBF = GB_FUNCTIONS

# GB_FUNC_COUNT is special right now
CPP_GB_TABLE= [
   (GBF.GB_SUM,     'sum',    GB_PACKUNPACK.UNPACK, GroupByOps.sum,      None,  None, None, False),
   (GBF.GB_MEAN,    'mean',   GB_PACKUNPACK.UNPACK, GroupByOps.mean,     None,  None, None, False),
   (GBF.GB_MIN,     'min',    GB_PACKUNPACK.UNPACK, GroupByOps.min,      None,  None, None, False),
   (GBF.GB_MAX,     'max',    GB_PACKUNPACK.UNPACK, GroupByOps.max,      None,  None, None, False),

   # STD uses VAR with the param set to 1
   (GBF.GB_VAR,     'var',    GB_PACKUNPACK.UNPACK, GroupByOps.var,      None,  None, None, False),
   (GBF.GB_STD,     'std',    GB_PACKUNPACK.UNPACK, GroupByOps.std,     None,  None, None, False),

   (GBF.GB_NANSUM,  'nansum', GB_PACKUNPACK.UNPACK,  GroupByOps.nansum,   None,  None, None, False),
   (GBF.GB_NANMEAN, 'nanmean', GB_PACKUNPACK.UNPACK, GroupByOps.nanmean,  None,  None, None, False),
   (GBF.GB_NANMIN,  'nanmin', GB_PACKUNPACK.UNPACK,  GroupByOps.nanmin,   None,  None, None, False),
   (GBF.GB_NANMAX,  'nanmax', GB_PACKUNPACK.UNPACK,  GroupByOps.nanmax,   None,  None, None, False),
   (GBF.GB_NANVAR,  'nanvar', GB_PACKUNPACK.UNPACK,  GroupByOps.nanvar,   None,  None, None, False),
   (GBF.GB_NANSTD,  'nanstd', GB_PACKUNPACK.UNPACK,  GroupByOps.nanstd,   None,  None, None, False),

   (GBF.GB_FIRST,   'first',  GB_PACKUNPACK.PACK,   GroupByOps.first,   None,  None, None, False),
   (GBF.GB_NTH,     'nth',    GB_PACKUNPACK.PACK,   GroupByOps.nth,     None,  None, None, False),
   (GBF.GB_LAST,    'last',   GB_PACKUNPACK.PACK,   GroupByOps.last,    None,  None, None, False),

   # requires parallel qsort
   (GBF.GB_MEDIAN,  'median', GB_PACKUNPACK.PACK,   GroupByOps.median,  None,  None, None, False), # auto handles nan
   (GBF.GB_MODE,    'mode',   GB_PACKUNPACK.PACK,   GroupByOps.mode,    None,  None, None, False), # auto handles nan
   (GBF.GB_TRIMBR,  'trimbr', GB_PACKUNPACK.PACK,   GroupByOps.trimbr,  None,  None, None, False), # auto handles nan


   # All int/uints output upgraded to INT64
   # Output is all elements (not just grouped)
   # takes window= as parameter
   (GBF.GB_ROLLING_SUM,    'rolling_sum',       GB_PACKUNPACK.PACK,   GroupByOps.rolling_sum,      None,  None, None, True),
   (GBF.GB_ROLLING_NANSUM, 'rolling_nansum',    GB_PACKUNPACK.PACK,   GroupByOps.rolling_nansum,   None,  None, None, True),
   (GBF.GB_ROLLING_DIFF,   'rolling_diff',      GB_PACKUNPACK.PACK,   GroupByOps.rolling_diff,     None,  None, None, True),
   (GBF.GB_ROLLING_SHIFT,  'rolling_shift',     GB_PACKUNPACK.PACK,   GroupByOps.rolling_shift,    None,  None, None, True),
   (GBF.GB_ROLLING_COUNT,  'rolling_count',     GB_PACKUNPACK.PACK,   GroupByOps.rolling_count,    None,  None, None, True),
   (GBF.GB_ROLLING_MEAN,   'rolling_mean',      GB_PACKUNPACK.PACK,   GroupByOps.rolling_mean,     None,  None, None, True),
   (GBF.GB_ROLLING_NANMEAN,'rolling_nanmean',   GB_PACKUNPACK.PACK,   GroupByOps.rolling_nanmean,  None,  None, None, True),

   # In ema.cpp
   (GBF.GB_CUMSUM,         'cumsum',            GB_PACKUNPACK.PACK,   GroupByOps.cumsum,      None,  None, None, True),
   (GBF.GB_CUMPROD,        'cumprod',           GB_PACKUNPACK.PACK,   GroupByOps.cumprod,     None,  None, None, True),

   # returns x elements ahead
   (GBF.GB_FINDNTH,        'findnth',           GB_PACKUNPACK.PACK,   GroupByOps.findnth,     None,  None, None, True),

   # takes
   (GBF.GB_EMADECAY,       'ema_decay',          GB_PACKUNPACK.PACK,   GroupByOps.ema_decay,      None,  None, None, True),
   (GBF.GB_EMANORMAL,      'ema_normal',         GB_PACKUNPACK.PACK,   GroupByOps.ema_normal,     None,  None, None, True),
   (GBF.GB_EMAWEIGHTED,    'ema_weighted',       GB_PACKUNPACK.PACK,   GroupByOps.ema_weighted,   None,  None, None, True),
   ]

# NOT DONE YET
# cummin
# cummax
# sem
# ohlc
# resample
# describe
# head
# tail
# rank
# ngroup

CPP_REVERSE_TABLE={}

# Build CPP funcnum table
for v in CPP_GB_TABLE:
    funcnum = int(v[0])
    CPP_REVERSE_TABLE[funcnum]={
        'name': v[1],
        'packing': v[2],
        'func_front': v[3],
        'func_back': v[4],
        'func_gb':v[5],
        'func_dtype': v[6],
        'return_full': v[7]
        }

