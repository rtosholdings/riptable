__all__ = ['hstack_any', 'merge_cats', 'stack_rows']

import warnings
from collections import defaultdict
from typing import List, Union, Optional, Mapping

import numpy as np
import riptide_cpp as rc
from .rt_enum import TypeRegister, NumpyCharTypes, CategoryMode, int_dtype_from_len
from .rt_numpy import arange, unique, ismember, crc64, hstack, empty
from .rt_timers import tic, toc
from .rt_grouping import hstack_groupings


def hstack_any(itemlist:Union[list, Mapping[str, np.ndarray]], cls:Optional[type]=None, baseclass:Optional[type]=None, destroy:bool=False, **kwargs):
    '''
    stack_rows or hstack_any is the main routine to row stack riptable classes.
    It stacks categoricals, datasets, time objects, structs.
    It can now stack a dictionary of numpy arrays to return a single array and a categorical.

    Parameters
    ----------
    itemlist:
        a list of objects to stack for arrays, datasets, categoricals
        a dictionary of numpy arrays

    Other Parameters
    ----------------
    cls: None.  the type of class we are stacking
    baseclass: the baseclass we are stacking
    destroy: bool, False.  Only valid for Datasets
        !! This is dangerous so make sure you do not want the data anymore in the original datasets.

    Returns
    -------
    In the case of a list: returns a single new array, dataset, categorical or specified object.
    In the case of a dict: returns a single new array and a new categorical (two objects returned).

    Examples
    --------
    >>> stack_rows([arange(3), arange(2)])
    FastArray([0, 1, 2, 0, 1])

    >>> d={'test1':arange(3), 'test2':arange(1), 'test3':arange(2)}
    >>> arr, cat = stack_rows(d)
    >>> Dataset({'Data':arr, 'Cat': cat})
    #   Data   Cat
    -   ----   -----
    0      0   test1
    1      1   test1
    2      2   test1
    3      0   test2
    4      0   test3
    5      1   test3

    See Also
    --------
    np.hstack
    rt.Categorical.align
    rt.Dataset.concat_rows
    '''
    #------------------------------
    def hstack_containers(itemlist, fill_inv=True, destroy=False):
        # assembles all columns into lists for final hstack call
        # keeps running count for gap, creates invalid fill array
        # the invalid fill should get pushed down, provide cutoffs instead
        # should there be a flag to fill with invalid? can't get row length from structs

        # final call to hstack or vstack
        # hstack will call subclass hstack

        # keep track of item order as more are added
        # use the first container to initialize order
        id = len(itemlist[0])
        allkeys = dict(zip(itemlist[0], arange(id)))
        alldicts = {
            'Array':defaultdict(list),
            'Struct':defaultdict(list),
            'Dataset':defaultdict(list),
            'PDataset':defaultdict(list)
        }

        newitems = []
        for container in itemlist:
            for key, item in container.items():
                exists = allkeys.setdefault(key, id)
                if exists == id:
                    newitems.append(key)
                id += 1

                # wrap scalars
                if np.isscalar(item):
                    item = TypeRegister.FastArray(item)

                if isinstance(item, np.ndarray):
                    dname = 'Array'

                elif issubclass(type(item), TypeRegister.Struct):
                    dname = type(item).__name__

                else:
                    raise TypeError(f'Unsupported type {type(item)} in container hstack.')

                # check for different types in different containers
                for dkey, d in alldicts.items():
                    if dkey == dname:
                        continue
                    if key in d:
                        # maybe raise a value error?
                        warnings.warn(f'Found conflicting types for item {key} {dkey} vs. {dname}. Data will be lost.')

                # add to list of items for that type
                alldicts[dname][key].append(item)

        # warn on missing items
        if len(newitems) > 0:
            warnings.warn(f"These items were not found in every container: {newitems}")

        # array will fix subclass
        for k,v in alldicts['Array'].items():
            alldicts['Array'][k] = hstack_arrays(v)

        # for all container types (possibly allow these to get stacked together?)
        for k,v in alldicts['Struct'].items():
            alldicts['Struct'][k] = hstack_containers(v)

        for k,v in alldicts['Dataset'].items():
            alldicts['Dataset'][k] = hstack_containers(v, destroy=destroy)

        for k,v in alldicts['PDataset'].items():
            alldicts['PDataset'][k] = hstack_containers(v)

        # combine dicts, allkeys is final sort order
        result = {}
        for d in alldicts.values():
            result.update(d)
        return result, list(allkeys)

    #------------------------------
    def hstack_arrays(itemlist):
        # maybe pass cutoffs to hstack arrays for invalid fill?

        # for all array-like items:
        # check type
        # check number of dimensions
        # check shape

        # for dtypes that don't match, check in python or send down?
        # right now:
        # checks that all are a numeric type or all are a string type, warns
        # cannot stack FA subclass with regular FA
        return hstack(itemlist)


    #------------------------------
    def hstack_dict(itemdict, **kwargs):
        '''
        Will create a Categorical from the keys in the dictionary.

        Examples
        --------
        >>> mydict={}
        >>> basepath = r'C:\path\to\data'
        >>> for j in range(11):
        ...    filename = f'{basepath}{j}.sds'
        ...    ds = rt.load_sds(filename)
        ...    mydict[f'cat{j}']=ds
        >>> z, mycat = rt.stack_rows(mydict)
        >>> z.mycat
        '''
        dictlen = len(itemdict)
        if dictlen ==0:
            raise ValueError(f'List of dictionary items to stack was empty.')

        # create an array of cutoffs for slicing later
        lengths = TypeRegister.FastArray([z.shape[0] for z in itemdict.values()], dtype=np.int64).cumsum()
        keynames = [*itemdict.keys()]

        return_arr= hstack([*itemdict.values()])

        # choose proper ikey size
        total_len = return_arr.shape[0]

        if total_len != lengths[-1]:
            raise ValueError(f'Final lengths do not match.  The hstack resulted in length {total_len} but counted a total length of {lengths[-1]}.')

        ikey = empty((total_len,), dtype=int_dtype_from_len(total_len))

        startpos =0
        for i in range(len(lengths)):
            endpos = lengths[i]
            ikey[startpos:endpos] = i + 1
            startpos = endpos

        return_cat = TypeRegister.Categorical(ikey, keynames, ordered=False)
        return return_arr, return_cat

    # start of hstack_any ---------
    #------------------------------
    # map each riptable type to a custom hstack function
    hstack_funcs = {
        TypeRegister.Date : _hstack_date,
        TypeRegister.DateSpan : _hstack_datespan,
        TypeRegister.DateTimeNano : _hstack_datetimenano,
        TypeRegister.TimeSpan : _hstack_timespan,
        TypeRegister.Categorical : _hstack_categorical,
        TypeRegister.Dataset : _hstack_dataset,
        TypeRegister.Struct : _hstack_struct
    }

    # items need to come in a tuple or list or dict
    if isinstance(itemlist, dict):
        return hstack_dict(itemlist, **kwargs)

    if not isinstance(itemlist, (list, tuple)):
        raise TypeError(f"Input must be a list or tuple of datasets. Got {type(itemlist)}")

    if len(itemlist) == 1:
        return itemlist[0]
    elif len(itemlist) == 0:
        raise ValueError(f'List of items to stack was empty.')

    sameclass = {type(o) for o in itemlist}
    if len(sameclass) != 1:
        raise TypeError(f"itemlist must contain objects with the same class: {sameclass!r}")

    if cls is None:
        cls = sameclass.pop()

    # hit the old routines
    if cls is not None:
        func = hstack_funcs.get(cls)
        if func is not None:
            return func(itemlist, destroy=destroy)

    # fallback to baseclass
    if baseclass is not None:
        func = hstack_funcs.get(baseclass)
        if func is not None:
            return func(itemlist, destroy=destroy)

    # default to normal hstack
    return hstack_arrays(itemlist)


#--------------------------------------------------------------------------
def _hstack_struct(struct_list, destroy:bool=False):
    '''
    Merges data from multiple structs.

    A struct utility for merging data from multiple structs (useful for multiday loading).
    Structs must have the same keys, and contain only Structs, Datasets, arrays, and riptable arrays.

    Parameters
    ----------
    struct_list : obj:`list` of obj:`Struct`
        A list of Struct instances to hstack.
    destroy : bool
        This parameter is currently ignored.

    Returns
    -------
    obj:`Struct`

    See Also
    --------
    riptable.hstack
    '''
    # items will be sorted into lists by type for final hstack call
    alldicts = defaultdict(dict)

    if len(struct_list) == 1 and isinstance(struct_list[0], TypeRegister.Struct):
        return struct_list[0]
    elif len(struct_list) > 0:
        # assumes that all keys are in the first struct
        #model_keys = struct_list[0].keys()
        # initialize keys with those from the first struct
        id = len(struct_list[0])
        all_keys = dict(zip(struct_list[0].keys(), arange(id)))
    else:
        raise ValueError(f"List of structs was empty.")

    new_items = []
    for st in struct_list:
        for key, item in st.items():
            # check if item has already been marked for concatenation
            exists = all_keys.setdefault(key, id)
            if exists == id:
                # this is a new item that did not appear in the structs before it
                # warn later
                new_items.append(key)
            id += 1

            # might change this in the future. matlab scalars now always load as arrays.
            if np.isscalar(item):
                item = TypeRegister.FastArray(item)

            # group together riptable types
            if isinstance(item, TypeRegister.Struct) or TypeRegister.is_array_subclass(item):
                dname = type(item).__name__

            # for all regular numpy / fastarray
            elif isinstance(item, np.ndarray):
                dname = 'Array'

            else:
                raise TypeError(f"Unsupported type {type(item)} in Struct append.")

            # check for key in other type dictionaries
            for dkey, d in alldicts.items():
                if dkey == dname:
                    continue
                if key in d:
                    warnings.warn(f'Found conflicting types for item {key} {dkey} vs. {dname}. Data will be lost.')

            data_dict = alldicts[dname]

            # use a get() to account for first struct
            data_list = data_dict.get(key,[])
            data_list.append(item)
            data_dict[key] = data_list

    if len(new_items) > 0:
        warnings.warn(f"These items were not found in every struct: {new_items}")

    final_dict = {}
    for name, fa_list in alldicts['Array'].items():
        # make sure all dimensions are the same, possibly perform an hstack
        dims = list({f.ndim for f in fa_list})
        if len(dims) == 1:
            dims = dims[0]
            if dims == 1:
                stack_func = hstack
            elif dims == 2:
                stack_func = np.vstack
            else:
                raise ValueError(f"Can only stack arrays of one or two dimensions. Got {dims} dimensions.")
        else:
            raise ValueError(f"Found conflicting dimensions {dims} for item {name} in multiple structs.")
        try:
            final_dict[name] = stack_func(fa_list)
        except Exception:
            raise TypeError(f"Could not perform {stack_func} on arrays for column {name}. Arrays had shapes {[f.shape for f in fa_list]}")

    # hstack lists of the same type - rt.hstack will route these to the correct routine
    for typename, typedict in alldicts.items():
        for name, itemlist in typedict.items():
            final_dict[name] = hstack(itemlist)

    result = TypeRegister.Struct(final_dict)
    # columns in final struct will appear in order of first occurrence in struct list
    order = list(all_keys.keys())
    result.col_move_to_front(order)
    return result


#--------------------------------------------------------------------------
def _hstack_dataset(ds_list: Union[list, tuple], destroy:bool=False):
    '''
    Stacks columns from multiple datasets.

    If a dataset is missing a column that appears in others, it will fill the gap with the invalid for that column's dtype.
    Categoricals will be merged and stacked.
    Column types will be checked to make sure they can be safely stacked - no general type mismatch allowed.
    Columns of the same name must have the same number of dimension in each dataset (1 or 2 dimensions allowed)

    Parameters
    ----------
    ds_list : list of Dataset or tuple of Dataset

    Other Parameters
    ----------------
    destroy: bool, False
        Set to True to destroy the contents of the dataset to save on memory.
        !! This is dangerous so make sure you do not want the data anymore in the original datasets.
    '''

    if not isinstance(ds_list, (list, tuple)):
        raise TypeError(f"Input must be a list or tuple of datasets. Got {type(ds_list)}")

    num_datasets = len(ds_list)
    if num_datasets == 0:
        # empty dict
        return TypeRegister.Dataset({})
    if num_datasets == 1:
        return ds_list[0]

    # collect dtypes, dimensions, shapes from all columns with given name
    types = {}
    ndims = {}
    shapes = {}
    for ds in ds_list:
        for name, col in ds.items():
            typelist = types.setdefault(name, [])
            # array subclasses should use their type, not their dtype
            if TypeRegister.is_array_subclass(col):
                typelist.append(type(col))
            else:
                typelist.append(col.dtype)

            # collecting number of dimensions for now
            # certain concatenations of different dimensions are allowed in numpy
            # we will raise an error
            dimlist = ndims.setdefault(name, [])
            dimlist.append(col.ndim)

            # for generating correct invalid before stack (only necessary for 2-dims)
            if col.ndim == 2:
                shapelist = shapes.setdefault(name, [])
                shapelist.append(col.shape[1])

    # check to make sure every column can be safely h-stacked
    for name, dts in types.items():
        types[name] = set(dts)
        types[name] = [dt for dt in types[name]]

        # matching types or compatible dtypes
        if len(types[name]) > 1:
            try:
                # possibly all numeric
                safe = all( [ t.char in NumpyCharTypes.AllInteger+NumpyCharTypes.AllFloat for t in types[name] ] )
                if not safe:
                    # possibly all string
                    safe = all( [t.char in 'US' for t in types[name] ] )
                # if conflicting warn the user
                if not safe:
                    warnings.warn(f"Mixing numeric with string types in column {name}. Unexpected results may occur.")
            # possibly a binned class
            except Exception:
                raise TypeError(f"Custom class type was mixed with regular dtype in {name} column.  Types are {types[name]}.")

        # matching number of dimensions
        ndims[name] = list(set(ndims[name]))
        if len(ndims[name]) > 1:
            raise ValueError(f"Found multiple dimensions for column {name}: {ndims[name]}")
        else:
            dim = ndims[name][0]
            if dim != 1 and dim != 2:
                raise NotImplementedError(f"concat rows only supports columns of 1 or 2 dimensions, got {ndims[name]}")
            else:
                # two dimensional shape[1] matches (catch before numpy fails/invalids are generated later)
                if dim == 2:
                    width = list(set(shapes[name]))
                    if len(width) > 1:
                        raise ValueError(f"Found mismatch in axis-1 length for {name}, got {width}")
                    shapes[name] = width[0]
                ndims[name] = dim

    # we have our own hstack to avoid multiple copies when using safe types
    newcols = {}

    if destroy:
        # temporarily stop recycling (1=off)
        recycle_mode = TypeRegister.FastArray._ROFF(quiet=True)
        # force memory cleanup now
        rc.RecycleGarbageCollectNow(0)

    for name, t in types.items():
        # for each column, attempt to get it from each dataset, or create an invalid one
        column_list = []
        coldim = ndims[name]
        if coldim == 1:
            inv_count = 0
        else:
            inv_count = (0, shapes[name])
        #is_binned = t[0] in binned_types
        for ds in ds_list:
            nrows = ds._nrows
            # column was found in current dataset
            if name in ds:
                col = ds[name]

                # might add gap column, reset the running invalid column
                inv_count, column_list = _possibly_add_concat_gap(coldim, inv_count, col, column_list, t)

                # add the existing column
                column_list.append(col)

            # reserve room for the gap column, just add to gap length to minimize invalid array generation
            else:
                if coldim == 1:
                    inv_count+=nrows
                else:
                    inv_count = (inv_count[0]+nrows, inv_count[1])

        # check to make sure there's not a leftover invalid column to be added
        # note: col is the last col value from the loop above
        inv_count, column_list = _possibly_add_concat_gap(coldim, inv_count, col, column_list, t)
        del col

        # 2-dimensions need to be handled differently
        if coldim == 1:
            stackfunc = hstack
        else:
            # NOTE: if riptable takes over vstack, remove the dtype fixup that happens in _possibly_add_concat_gap
            stackfunc = np.vstack

        # add to final dataset dict
        newcols[name] = stackfunc(column_list)
        del column_list

        # check if destroy the data inside the datasets
        if destroy:
            for ds in ds_list:
                # column was found in current dataset
                if name in ds:
                    del ds[name]

    returnds = TypeRegister.Dataset(newcols)
    del newcols

    if destroy:
        # turn recycling mode back to what it was
        if recycle_mode: TypeRegister.FastArray._RON(quiet=True)

    return returnds

#--------------------------------------------------------------------------
def _possibly_add_concat_gap(coldim:int, inv_count, col, column_list, types):
    '''
    Called by _hstack_dataset
    Column existed in calling loop's current dataset, might need to create/add an invalid column
    before adding an existing column. If invalid was added, reset the invalid count.

    Parameters:
    -----------
    coldim      : dimensions of column (1 or 2)
    inv_count   : running count of invalid gap. will be single int value for 1dim length or tuple of 2dim shape
    col         : existing column, used to generate the correct invalid value for the gap
    column_list : running list of all columns existing/invalid for final stack
    types       : dtypes of arrays - 2dim may need to generate invalid differently, 1dim will be fixed in hstack

    '''
    if coldim == 1:
        if inv_count > 0:
            column_list.append( col.fill_invalid(shape=inv_count, inplace=False) )
            inv_count = 0
    else:
        if inv_count[0] > 0:
            dtype=None
            if len(types) > 1:
                common = np.find_common_type(types,[])
                # numeric, supported, ensure final vstack result dtype - strings will be automatic
                if common.num <= 13:
                    # future optimization: store this for performance, reduce types list to final type
                    dtype = common
            column_list.append( col.fill_invalid(shape=inv_count, inplace=False, dtype=dtype) )
            inv_count = (0, inv_count[1])

    return inv_count, column_list

# ------------------------------------------------------------
def _hstack_categorical(cats:list, verbose:bool=False, destroy:bool=False):
    '''
    HStack Categoricals.

    The unique categories will be merged into a new unique list.
    The indices will be fixed to point to the new category array.
    The indices are hstacked and a new categorical is returned.

    Parameters
    ----------
    cats : list of Categorical
        Cats must be a list of categoricals.
    verbose : bool
        Enable verbose output. Defaults to False.
    destroy : bool
        This parameter is ignored by this function.

    Returns
    -------
    Categorical

    Examples
    --------
    >>> c1 = Categorical(['a','b','c'])
    >>> c2 = Categorical(['d','e','f'])
    >>> combined = Categorical.hstack([c1,c2])
    >>> combined
    Categorical([a, b, c, d, e, f]) Length: 6
        FastArray([1, 2, 3, 4, 5, 6]) Base Index: 1
        FastArray([b'a', b'b', b'c', b'd', b'e', b'f'], dtype='|S1') Unique count: 6
    '''
    def attrs_match(attrlist, name):
        # ensure certain attributes are the same for all categoricals being stacked
        attrs = set(attrlist)
        if len(attrs) != 1:
            raise TypeError(f"hstack found {len(attrlist)} different values of the '{name}' attribute in provided Categoricals. Must all be the same.")
        return list(attrs)[0]

    # collect all the categorical modes and all the base indexes
    modes = []
    bases = []
    for cat in cats:
        if not isinstance(cat, TypeRegister.Categorical):
            raise TypeError(f"Categorical hstack is for categoricals, not {type(cat)}")
        #if cat.base_index not in (1, None):
        #    raise TypeError(f"only categoricals with base index 1 can be merged (to preserve invalid values).")
        modes.append(cat.category_mode)
        bases.append(cat.base_index)

    # all categoricals must be in same mode and have same base index
    mode = attrs_match(modes, 'mode')
    base_index = attrs_match(bases, 'base index')

    # the first categorical determines the ordered kwarg
    ordered = cats[0].ordered
    sort_display = cats[0].sort_gb


    #==========================
    # todo: see _multistack_categoricals int rt_sds.py
    # stack indices
    # this will stack the fastarrays
    indices = rc.HStack(cats)
    idx_cutoffs = TypeRegister.FastArray([len(c._fa) for c in cats], dtype=np.int64).cumsum()

    #------------------------- start rebuild here
    if mode in (CategoryMode.Dictionary, CategoryMode.IntEnum):

        # -----------------------
        # use info from grouping objects to stack
        glist = [c.grouping for c in cats]

        underlying = hstack([[*g._grouping_dict.values()][0] for g in glist])
        # stack all unique string arrays
        listnames = hstack([g._enum.category_array for g in glist])

        # collect, measure, stack integer arrays
        listcodes = [g._enum.code_array for g in glist]
        unique_cutoffs = [ TypeRegister.FastArray([len(c) for c in listcodes], dtype=np.int64).cumsum() ]
        listcodes = hstack(listcodes)

        # send in as two arrays
        listcats = [ listcodes, listnames ]

        # -----------------------
        base_index = None
        indices, listcats = merge_cats(indices, listcats, unique_cutoffs=unique_cutoffs, from_mapping=True, ordered=ordered, verbose=verbose)
        # TJD added check
        code=listcats[0][0]
        if isinstance(code, (int, np.integer)):
            # EXCEPT first value is string, and second is int
            newcats = dict(zip(listcats[1], listcats[0]))
        else:
            newcats = dict(zip(listcats[0], listcats[1]))

    else:
        category_dict = {}
        # now we need stack the unique cats
        for c in cats:

            # it might be multikey
            for i, v in enumerate(c.category_dict.values()):
                cv = category_dict.get(i, None)
                if cv is None:
                    category_dict[i] = [v]
                else:
                    cv.append(v)
                    category_dict[i] = cv

        listcats = []
        lastv = []
        for v in category_dict.values():
            listcats.append(hstack(v))
            lastv = v

        unique_cutoffs = [TypeRegister.FastArray([len(v) for v in lastv], dtype=np.int64).cumsum()]

        indices, newcats = merge_cats(indices, listcats, idx_cutoffs=idx_cutoffs, unique_cutoffs=unique_cutoffs, verbose=verbose, base_index=base_index, ordered=ordered)

    newcats = TypeRegister.Grouping(indices, categories=newcats, _trusted=True, base_index=base_index, ordered=ordered, sort_display=sort_display)
    result = TypeRegister.Categorical(newcats)
    return result


# ------------------------------------------------------------
def _hstack_grouping(glist, _trusted:bool=False, base_index:int=1, ordered:bool=False, destroy:bool=False):
    '''
    HStack for Grouping instances.

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
    '''
    # need to vet base index, enum mode, number of columns, etc.
    if not _trusted:
        warnings.warn(f'still implementing grouping hstack validation')

        # TODO: add more tests for single vs. multikey (without unnecessary calculation of uniquedict)
        for grp in glist:
            same_mode = set()
            if not isinstance(grp, TypeRegister.Grouping):
                raise TypeError(f"Grouping hstack is for categoricals, not {type(grp)}")
            same_mode.add(grp.isenum)

        if len(same_mode)!=1:
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
        cutoffs = [ TypeRegister.FastArray([len(c) for c in listcodes], dtype=np.int64).cumsum() ]
        listcodes = hstack(listcodes)

        # send in as two arrays
        listcats = [ listcodes, listnames ]

        # will return new unique arrays for codes, names
        underlying, newcats = merge_cats(underlying, listcats, unique_cutoffs=cutoffs, from_mapping=True, ordered=ordered)
        newgroup = TypeRegister.Grouping(underlying, categories=dict(zip(newcats[1], newcats[0])), _trusted=True)
    else:
        # use catinstance for base 0 or 1
        listidx = [g.catinstance for g in glist]
        cat_tuples = [tuple(g.uniquedict.values()) for g in glist]
        listcats = [ [v[i] for v in cat_tuples ] for i in range(len(cat_tuples[0])) ]
        underlying, newcats = merge_cats(listidx, listcats, base_index=base_index, ordered=ordered)
        newgroup = TypeRegister.Grouping(underlying, categories=newcats, categorical=True, _trusted=True, base_index=base_index, ordered=ordered, sort_display=sort_display)

    return newgroup

# ------------------------------------------------------------
def _hstack_datetimenano(dtlist, destroy=False):
    '''
    Performs an hstack on a list of DateTimeNano objects.
    All items in list must have their display set to the same timezone.
    NOTE: destroy ignored
    '''
    # make sure all of the date time nano objects are set to be displayed relative to the same timezone
    timezone = dtlist[0]._timezone._timezone_str
    for dt in dtlist:
        if not isinstance(dt, TypeRegister.DateTimeNano):
            raise TypeError(f"Items to be hstacked must be DateTimeNano objects.")
        if dt._timezone._timezone_str != timezone:
            raise NotImplementedError(f"Can only hstack DateTimeNano objects in the same timezone.")
    if len(dtlist) == 1:
        return dtlist

    # hstack int64 utc nano arrays
    arr = rc.HStack(dtlist)

    # reconstruct with first item
    return TypeRegister.DateTimeNano.newclassfrominstance(arr, dtlist[0])

#------------------------------------------------------------
def _hstack_timespan(tspans, destroy=False):
    '''
    TODO: maybe add type checking?
    This is a very simple class, rewrap the hstack result in class.
    NOTE: destroy ignored
    '''
    ts = rc.HStack(tspans)
    return TypeRegister.TimeSpan(ts)

#------------------------------------------------------------
def _hstack_date(dates, destroy=False):
    '''
    NOTE: destroy ignored
    '''
    return _hstack_date_internal(dates, subclass=TypeRegister.Date)

#------------------------------------------------------------
def _hstack_datespan(dates, destroy=False):
    '''
    NOTE: destroy ignored
    '''
    return _hstack_date_internal(dates, subclass=TypeRegister.DateSpan)

#------------------------------------------------------------
def _hstack_date_internal(dates, subclass=TypeRegister.Date):
    '''
    hstacks Date / DateSpan objects and returns a new Date / DateSpan object.

    Will be called by riptable.hstack() if the first item in the sequence is a Date object.

    Parameters
    ----------
    dates : list or tuple of Date / DateSpan objects

    Examples
    --------
    >>> d1 = Date('2015-02-01')
    >>> d2 = Date(['2016-02-01', '2017-02-01', '2018-02-01'])
    >>> hstack([d1, d2])
    Date([2015-02-01, 2016-02-01, 2017-02-01, 2018-02-01])
    '''
    if len(dates) == 0:
        return subclass([])
    for d in dates:
        if not isinstance(d, subclass):
            # maybe extend this to support stacking with regular DateTimeNano objects?
            raise TypeError(f'Could not perform Date.hstack() on item of type {type(d)}')
    if len(dates) == 1:
        return dates

    stacked = rc.HStack(dates)
    return subclass.newclassfrominstance(stacked, dates[0])

# ------------------------------------------------------------
def crc_match(arrlist: List[np.ndarray]) -> bool:
    """
    Perform a CRC check on every array in list, returns True if they were all a match.

    Parameters
    ----------
    arrlist : list of numpy arrays

    Returns
    -------
    bool
        True if all arrays in `arrlist` are structurally equal; otherwise, False.

    See Also
    --------
    numpy.array_equal
    """
    # This function also compares the shapes of the arrays in addition to the CRC value.
    # This is necessary for correctness because this function is (essentially) implementing a structural
    # equality comparison for arrays; a CRC value may not be impacted by zeros in some cases, e.g.
    #   crc64(FA([b'', b'abcdef'])) == crc64(FA([b'abcdef']))
    # which will give an incorrect result (since the arrays actually aren't structurally equal).

    crcs = {(arr.shape, crc64(arr)) for arr in arrlist}
    return len(crcs) == 1

# ------------------------------------------------------------
def merge_cats(indices, listcats, idx_cutoffs=None, unique_cutoffs=None, from_mapping=False, stack=True, base_index =1, ordered=False, verbose=False):
    '''
    For hstacking Categoricals possibly from a stacked .sds load.

    Supports categoricals from single array or dictionary mapping.

    Parameters
    ----------
    indices :  single stacked array or list of indices
                if single array, needs idx_cutoffs for slicing
    listcats : list of stacked unique category arrays (needs unique_cutoffs)
                or list of lists of uniques
                if the uniques in file1 are 'A,'C'  and the uniques in file2 are 'B','C,'D'
                then listcats is [FastArray('A','C','B','C','D')]
    idx_cutoffs: int64 array of the cutoffs to the  indices
               if the index length is 30 and 20 the idx_cutoffs is [30,50]
    unique_cutoffs: list of one int64 array of the cutoffs to the listcats
               if the index length is 2 and 3 the idx_cutoffs is [2,5]

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
        print("**indicees", indices)
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

# create a stack_rows alias for now
stack_rows = hstack_any