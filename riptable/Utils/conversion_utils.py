__all__ = [
    'dataset_as_matrix',
    'numpy2d_to_dict',
    'dset_dict_to_list',
    'append_dataset_dict',
    'numpy_array_to_dict',
    'numpy_array_to_dataset',
]

import numpy
from riptable.rt_datetime import UTC_1970_SPLITS, DateTimeNano
import riptable as rt

# TODO: Add more thorough type checking support, e.g, what to do with int64, uint64, datetime
# TODO: Should we convert to multiple columns for 64 bits, e.g, date, time-of-day for datetime?


def _is_float_encodable(xtype):
    return xtype in (
        int,
        float,
        numpy.integer,
        numpy.floating,
        numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,
        numpy.uint8,
        numpy.uint16,
        numpy.uint32,
        numpy.uint64,
        numpy.float16,
        numpy.float32,
        numpy.float64,
    )


def _normalize_column(x, field_key):
    original_type = x.dtype
    category_values = None
    is_categorical = False
    if _is_float_encodable(original_type):
        if isinstance(x, rt.rt_categorical.Categorical):
            category_values = x._categories
            is_categorical = True
        vals = x.astype(numpy.float64)
    else:
        if field_key is None:
            category_values, vals = numpy.unique(x, return_inverse=True)
            vals = vals.astype(numpy.float64)
        else:
            category_values = field_key
            isValid, vals = rt.ismember(x, category_values, 1)
            vals = vals.astype(numpy.float64)
            vals[~isValid] = numpy.nan
    return vals, original_type, is_categorical, category_values


def dataset_as_matrix(ds, save_metadata=True, column_data={}):
    columns = list(ds.keys())
    nrows = ds.shape[0]
    ncols = ds.shape[1]  # TODO: may expand this for 64-bit columns
    out_array = numpy.empty((nrows, ncols), dtype=numpy.float64)
    column_info = dict()
    for col in range(ncols):
        field_key = column_data.get(columns[col])
        (
            out_array[:, col],
            original_type,
            is_categorical,
            category_values,
        ) = _normalize_column(ds[columns[col]], field_key)
        column_info[columns[col]] = {
            'dtype': original_type,
            'category_values': category_values,
            'is_categorical': is_categorical,
        }

    if save_metadata:
        return out_array, column_info
    else:
        return out_array


def numpy_array_to_dict(inarray, columns=None):
    if len(inarray.shape) > 2:
        raise TypeError('Only 1 and 2-dimensional arrays are supported')
    out = dict()
    if inarray.dtype.fields is None:
        if len(inarray.shape) == 1:
            if columns is not None:
                if isinstance(columns, list):
                    if len(columns) > 1:
                        raise ValueError("Incompatible arrayShape and columns")
                    out[columns[0]] = inarray
                else:
                    if not isinstance(columns, str):
                        raise TypeError("Unexpected columns type")
                    out[columns] = inarray
            else:
                out['Var'] = inarray
            return out
        else:
            if columns is not None:
                if len(columns) != inarray.shape[1]:
                    raise ValueError("Incompatible arrayShape")
            else:
                columns = [
                    'Var' + '_' + str(col_num) for col_num in range(inarray.shape[1])
                ]
            for col_num in range(inarray.shape[1]):
                out[columns[col_num]] = inarray[:, col_num].copy()
            return out
    else:
        if len(inarray.shape) > 1:
            raise TypeError('Only 1-dimensional structured arrays are supported')
        if columns is not None:
            if len(inarray.dtype.fields) != len(columns):
                raise ValueError("Incompatible arrayShape")
        else:
            columns = [str(fld) for fld in inarray.dtype.fields]
        for col_num, fld in enumerate(inarray.dtype.fields):
            out[columns[col_num]] = inarray[fld].copy()
        return out


def numpy_array_to_dataset(inarray, columns=None):
    out = rt.Dataset(numpy_array_to_dict(inarray, columns=columns))
    return out


# TODO: Replace all calls with numpy_array_to_dic? -CLH
def numpy2d_to_dict(arr, columns):
    """
    Converts arr 2D ndarray and column names to arr dict (is ordered) of ndarray's
    suitable for the Dataset constructor:

    :param arr: numpy NxM ndarray
    :param columns: list of M column names
    :return: dictionary suitable for rt.Dataset constructor

    Example:
    import numpy as np
    import riptable as rt
    from rt.Utils.conversion_utils import numpy2d_to_dict

    arr = np.array([[1, 2, 3], [4, 5, 6]])
    columns = ['c1', 'c2', 'c3']
    dset = rt.Dataset(numpy2d_to_dict(arr, columns))
    print(dset)
    """
    if len(columns) != arr.shape[1]:
        raise ValueError(
            "Incompatible arrayShape={} and len(columns)={:d}".format(
                arr.shape, len(columns)
            )
        )
    return dict(zip(columns, arr.T))


def dset_dict_to_list(ds_dict, key_field_name, allow_overwrite=False):
    """
    Converts a dict of Datasets to a list, appending the keyname as a new field key_field_name.
    NB: This modifies the Datasets!

    TODO: allow option of inplace or copy

    :param ds_dict: dictionary of Datasets.  Keys MUST be ascii strings (or bytes)!
    :param key_field_name: New column to add to each to which Dataset will be assigned
                           the constant value of the key.
    :param allow_overwrite: Unless set to True the key_field_name may not exist in any of the
                            input Datasets.
    :return: list of original Datasets _modified_.
    """
    if not allow_overwrite:
        for _d in ds_dict.values():
            if key_field_name in _d:
                raise ValueError(
                    'dset_dict_to_list(): key_field_name cannot be column name in any Dataset unless allow_overwrite=True.'
                )
    max_length = 1
    for key in ds_dict:
        if type(key) is not bytes:
            try:
                _ = key.encode('ascii')
            except (AttributeError, UnicodeEncodeError):
                raise ValueError(
                    'dset_dict_to_list(): all keys must be ascii strings or bytes.'
                )
        max_length = max(max_length, len(key))
    out = []
    typestr = f'|S{max_length}'
    for key, val in ds_dict.items():
        col_val = numpy.empty(shape=val.shape[0], dtype=typestr)
        col_val[:] = key
        # TODO: Replace with appropriate accessor
        val.__setattr__(key_field_name, col_val)
        out.append(val)
    return out


# -------------------------------------------------------
def append_dataset_dict(ds_dict, key_field_name):
    """
    Converts a dictionary of Datasets to a single Dataset appending the dictionary
    keys as key_field_name to distinguish them.
    NB: This modifies the original Datasets!

    TODO: add support for harmonizing fields, e.g.,
          fill missing values and deal with differing types

    :param ds_dict: dictionary of Datasets.  Keys MUST be ascii strings (or bytes)!
    :param key_field_name: New column to add to each to which Dataset will be assigned
                           the constant value of the key.
    :return: New dataset.
    """
    # check sort cache
    return rt.Dataset.concat_rows(dset_dict_to_list(ds_dict, key_field_name))


def possibly_convert_to_nanotime(vec):
    """
    Try to auto-detect nanotime and return DateTimeNano vector.

    :param vec: A numpy.ndarray, dtype numpy.int64.
    :return: DateTimeNano, True if successful, OrigVect, False if not.
    """
    # TODO: investigate reported problems with load
    if vec.dtype != numpy.int64:
        return vec, False
    # update this time window as necessary
    # print("**in possible nanotime")
    mintime = UTC_1970_SPLITS[44]
    maxtime = UTC_1970_SPLITS[49]
    # sample every 10
    arrlen = vec.size
    stepsize = arrlen // 10
    if arrlen < 10:
        sample = vec
    else:
        sample = vec[0:arrlen:stepsize]
    vmin = numpy.min(sample)
    vmax = numpy.max(sample)
    # if sampled data falls within 2015 - 2019 time window
    if vmin > mintime and vmax < maxtime:
        # print("**detected nano")
        return DateTimeNano(vec, from_tz='GMT', to_tz='NYC'), True
    # print("**failed nano")
    return vec, False
