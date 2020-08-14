__all__ = ['fill_forward', 'fill_backward']

#---------------------------------------------------------------------
# This file is for numba routines that work on numpy arrays
# It is different from CPP routines in the riptide_cpp module
# Others are encouraged to add
#
import numpy as np
import numba as nb
import warnings
from typing import Optional, Union

from .rt_enum import INVALID_DICT
from .rt_fastarray import FastArray
from .rt_numpy import empty_like

#-----------------------------------------------------
@nb.jit(cache=True)
def fill_forward_float(arr, fill_val, limit):
    lastgood = fill_val
    if limit <= 0:
        for idx in range(arr.shape[0]):
            if np.isnan(arr[idx]):
                arr[idx] = lastgood
            else:
                lastgood = arr[idx]
    else:
        counter = limit
        for idx in range(arr.shape[0]):
            if np.isnan(arr[idx]):
                # leave the value if counter <= 0
                if counter > 0:
                    arr[idx] = lastgood
                    counter -=1
            else:
                # reset counter
                counter = limit
                lastgood = arr[idx]

#-----------------------------------------------------
@nb.jit(nopython=True, cache=True)
def fill_forward_int(arr, inv, fill_val, limit):
    lastgood = fill_val
    if limit <= 0:
        for idx in range(arr.shape[0]):
            if arr[idx] == inv:
                arr[idx] = lastgood
            else:
                lastgood = arr[idx]
    else:
        counter = limit
        for idx in range(arr.shape[0]):
            if arr[idx] == inv:
                # leave the value if counter <= 0
                if counter > 0:
                    arr[idx] = lastgood
                    counter -=1
            else:
                # reset counter
                counter = limit
                lastgood = arr[idx]


#-----------------------------------------------------
@nb.jit(nopython=True, cache=True)
def fill_backward_float(arr, fill_val, limit):
    lastgood = fill_val
    if limit <= 0:
        for idx in range(arr.shape[0]-1, -1, -1):
            if np.isnan(arr[idx]):
                arr[idx] = lastgood
            else:
                lastgood = arr[idx]
    else:
        counter = limit
        for idx in range(arr.shape[0]-1, -1, -1):
            if np.isnan(arr[idx]):
                # leave the value if counter <= 0
                if counter > 0:
                    arr[idx] = lastgood
                    counter -=1
            else:
                # reset counter
                counter = limit
                lastgood = arr[idx]

#-----------------------------------------------------
@nb.jit(nopython=True, cache=True)
def fill_backward_int(arr, inv, fill_val, limit):
    lastgood = fill_val
    if limit <= 0:
        for idx in range(arr.shape[0]-1, -1, -1):
            if arr[idx] == inv:
                arr[idx] = lastgood
            else:
                lastgood = arr[idx]
    else:
        counter = limit
        for idx in range(arr.shape[0]-1, -1, -1):
            if arr[idx] == inv:
                # leave the value if counter <= 0
                if counter > 0:
                    arr[idx] = lastgood
                    counter -=1
            else:
                # reset counter
                counter = limit
                lastgood = arr[idx]

#-----------------------------------------------------
def _check_fill_values(arr, fill_val, inplace:bool, limit:int):
    if arr.dtype.num > 13 or arr.dtype.num == 0:
        # fill string, boolean, other?
        raise TypeError(f"Filling for type {type(arr)} is currently not supported.")

    if limit is None: limit =0

    limit = np.int64(limit)

    if limit < 0:
        raise TypeError(f"The limit kwarg cannot be less than 0.")

    if not inplace:
        arr= arr.copy()

    dtype = arr.dtype
    inv = INVALID_DICT[dtype.num]
    inv = np.array([inv], dtype=dtype)[0]

    if fill_val is None:
        # optionally could raise error
        fill_val =inv

    # force into np scalar with dtype
    # TODO there is a better way to do this
    fill_val = np.array([fill_val], dtype=dtype)[0]

    return arr, fill_val, inv, dtype, limit

#-----------------------------------------------------
def fill_forward(arr: np.ndarray, fill_val=None, inplace:bool=False, limit:int=0):
    """
    Fills array forward replacing invalids using last good value.

    Parameters
    ----------
    fill_val : scalar, optional
        Defaults to invalid. The fill value before a first good value is found.
        Set to single scalar value.
    inplace : bool
        Default False. Set to True to fill in place.
    limit : int
        Default 0 (disabled). The maximium number of consecutive invalid values.
        A gap with more than this will be partially filled.

    Returns
    -------
    FastArray

    Examples
    --------
    >>> vals = rt.arange(10).astype(np.float64)
    >>> vals[[1, 2, 5]] = rt.float64.inv
    >>> vals
    FastArray([ 0., nan, nan,  3.,  4., nan,  6.,  7.,  8.,  9.])

    >>> vals.fill_forward()
    FastArray([0., 0., 0., 3., 4., 4., 6., 7., 8., 9.])

    >>> vals.fill_forward(fill_val=-1.0)
    FastArray([0., 0., 0., 3., 4., 4., 6., 7., 8., 9.])

    >>> vals.fill_forward(fill_val=-1.0, limit=1)
    FastArray([ 0.,  0., nan,  3.,  4.,  4.,  6.,  7.,  8.,  9.])

    See Also
    --------
    fill_invalid
    fill_backward

    Notes
    -----
    TODO: handle axis
    """
    arr, fill_val, inv, dtype, limit = _check_fill_values(arr, fill_val, inplace, limit)

    if dtype.num <= 10:
       # fill integers or boolean
       fill_forward_int(arr, inv, fill_val, limit)
    else:
       # fill float
       fill_forward_float(arr, fill_val, limit)

    return arr

#-----------------------------------------------------
def fill_backward(arr: np.ndarray, fill_val=None, inplace:bool=False, limit:int=0):
    """
    Fills array forward replacing invalids using previous good value.

    Parameters
    ----------
    fill_val : scalar, optional
        Defaults to invalid. The fill value before a first previous value is found.
        Set to single scalar value.
    inplace : bool
        Default False. Set to True to fill in place.
    limit : int
        Default 0 (disabled). The maximium number of consecutive invalid values.
        A gap with more than this will be partially filled.

    Returns
    -------
    FastArray

    Examples
    --------
    >>> vals = rt.arange(10).astype(np.float64)
    >>> vals[[1, 2, 5]] = rt.float64.inv
    >>> vals
    FastArray([ 0., nan, nan,  3.,  4., nan,  6.,  7.,  8.,  9.])

    >>> vals.fill_backward()
    FastArray([0., 3., 3., 3., 4., 6., 6., 7., 8., 9.])

    >>> vals.fill_backward(fill_val=-1.0)
    FastArray([0., 3., 3., 3., 4., 6., 6., 7., 8., 9.])

    >>> vals.fill_backward(fill_val=-1.0, limit=1)
    FastArray([ 0., nan,  3.,  3.,  4.,  6.,  6.,  7.,  8.,  9.])

    See Also
    --------
    fill_invalid
    fill_forward

    Notes
    -----
    TODO: handle axis
    """
    arr, fill_val, inv, dtype, limit = _check_fill_values(arr, fill_val, inplace, limit)

    if dtype.num <= 10:
       # fill integers or boolean
       fill_backward_int(arr, inv, fill_val, limit)
    else:
       # fill float
       fill_backward_float(arr, fill_val, limit)

    return arr

#-----------------------------------------------------
@nb.jit(nopython=True, cache=True)
def nb_cummin_int(arr: np.ndarray, ret: np.ndarray, inv, skipna):
    if skipna:
        for j in range(len(arr)):
            running_min = arr[j]
            ret[j]=running_min
            if running_min != inv:
                break

        for i in range(j, len(arr)):
            val = arr[i]
            if val != inv and val < running_min:
                running_min=val
            ret[i]=running_min
    else:
        running_min = arr[0]
        for i in range(len(arr)):
            val = arr[i]
            if val == inv or val < running_min:
                running_min=val
            ret[i]=running_min

#-----------------------------------------------------
@nb.jit(nopython=True, cache=True)
def nb_cummin_float(arr: np.ndarray, ret: np.ndarray, skipna):
    if skipna:
        for j in range(len(arr)):
            running_min = arr[j]
            ret[j]=running_min
            if running_min == running_min:
                break

        for i in range(j, len(arr)):
            val = arr[i]
            if val == val and val < running_min:
                running_min=val
            ret[i]=running_min
    else:
        running_min = arr[0]
        for i in range(len(arr)):
            val = arr[i]
            if val != val or val < running_min:
                running_min=val
            ret[i]=running_min


#-----------------------------------------------------
@nb.jit(nopython=True, cache=True)
def nb_cummax_int(arr: np.ndarray, ret: np.ndarray, inv, skipna):
    if skipna:
        for j in range(len(arr)):
            running_max = arr[j]
            ret[j]=running_max
            if running_max != inv:
                break

        for i in range(j, len(arr)):
            val = arr[i]
            if val != inv and val > running_max:
                running_max=val
            ret[i]=running_max
    else:
        running_max = arr[0]
        for i in range(len(arr)):
            val = arr[i]
            if val == inv or val > running_max:
                running_max=val
            ret[i]=running_max

#-----------------------------------------------------
@nb.jit(nopython=True, cache=True)
def nb_cummax_float(arr: np.ndarray, ret: np.ndarray, skipna):
    if skipna:
        for j in range(len(arr)):
            running_max = arr[j]
            ret[j]=running_max
            if running_max == running_max:
                break

        for i in range(j, len(arr)):
            val = arr[i]
            if val == val and val > running_max:
                running_max=val
            ret[i]=running_max
    else:
        running_max = arr[0]
        for i in range(len(arr)):
            val = arr[i]
            if val != val or val > running_max:
                running_max=val
            ret[i]=running_max

def cummax(arr: np.ndarray, skipna=True):
    '''
    Return the running maximum over an array.

    Parameters
    ----------
    skipna : boolean, default True
        Exclude nan/invalid values.

    By default, nan values are ignored.

    Examples
    --------
    a=rt.FA([1,2,3,4,2,3,5,5,6,2,7])
    >>> a.cummax()
    FastArray([1, 2, 3, 4, 4, 4, 5, 5, 6, 6, 7])

    >>> a[a >= a.cummax()]
    FastArray([1, 2, 3, 4, 5, 5, 6, 7])

    >>> a[1]=rt.nan
    >>> a.cummax()
    FastArray([1, 1, 3, 4, 4, 4, 5, 5, 6, 6, 7])

    See Also
    --------
    cummin, cumprod, cumsum
    '''
    ret = empty_like(arr)
    if len(arr) > 0:
        dtype = arr.dtype
        inv = INVALID_DICT[dtype.num]

        if dtype.num <= 10:
            nb_cummax_int(arr, ret, inv, skipna)
        elif dtype.num <= 13:
            nb_cummax_float(arr, ret, skipna)
        else:
            raise TypeError("cummax only handles integers or floats.")

    return ret

def cummin(arr: np.ndarray, skipna=True):
    '''
    Return the running minimum over an array.

    Parameters
    ----------
    skipna : boolean, default True
        Exclude nan/invalid values.

    By default, nan values are ignored.

    Examples
    --------
    >>> a=FA([1,2,3,-4,2,3,5,5,-6,2,7])
    >>> a.cummin()
    FastArray([ 1,  1,  1, -4, -4, -4, -4, -4, -6, -6, -6])

    See Also
    --------
    cummax, cumprod, cumsum
    '''
    ret = empty_like(arr)
    if len(arr) > 0:
        dtype = arr.dtype
        inv = INVALID_DICT[dtype.num]

        if dtype.num <= 10:
            nb_cummin_int(arr, ret, inv, skipna)
        elif dtype.num <= 13:
            nb_cummin_float(arr, ret, skipna)
        else:
            raise TypeError("cummin only handles integers or floats.")

    return ret


#-----------------------------------------------------
@nb.jit(nopython=True, cache=True)
def nb_ema_decay_with_filter_and_reset(arr, dest, time, decayRate,  filter, resetmask):
    lastEma = 0
    lastTime = 0
    for i in range(len(arr)):
        value = 0

        # NOTE: fill in last value
        if filter[i] != 0:
            value = arr[i]

            if resetmask[i]:
                lastEma = 0
                lastTime = 0

        lastEma = value + lastEma * np.exp(-decayRate * (time[i] - lastTime))
        lastTime = time[i]
        dest[i] = lastEma

#-----------------------------------------------------
@nb.jit(nopython=True, cache=True)
def nb_ema_decay_with_filter(arr, dest, time, decayRate, filter):
    lastEma = 0
    lastTime = 0
    for i in range(len(arr)):
        value = 0

        # NOTE: fill in last value
        if filter[i] != 0:
            value = arr[i]

        lastEma = value + lastEma * np.exp(-decayRate * (time[i] - lastTime))
        lastTime = time[i]
        dest[i] = lastEma

#-----------------------------------------------------
@nb.jit(nopython=True, cache=True)
def nb_ema_decay(arr, dest, time, decayRate):
    lastEma = 0.0
    lastTime = 0
    for i in range(len(arr)):
        value = arr[i]

        # nan check (assumes float nans and not invalids)
        if value != value:
            value = 0

        lastEma = value + lastEma * np.exp(-decayRate * (time[i] - lastTime))
        lastTime = time[i]
        dest[i] = lastEma

#-----------------------------------------------------
def ema_decay(arr: np.ndarray, time:np.ndarray, decay_rate:float, filter:Optional[np.ndarray]=None,
        reset:Optional[np.ndarray]=None, dtype=np.float32):
    """
    Calculate the EMA using a fixed decay rate.

    Parameters
    ----------
    arr : array-like
        An array whose value is decayed over time.
    time : array-like
        A time array of equal length to `arr`. Often an int64 array.
    decay_rate : float
        A scalar value float such as 2.27.
    dtype : string or data-type
        The dtype to force for the output array. Defaults to ``np.float32``, can change to ``np.float64`` for more accuracy.

    Other Parameters
    ----------------
    filter : array-like, optional
        A boolean mask. If supplied, must be the same length as `arr`.
    reset : array-like, optional
        A boolean mask. Only valid if `filter` is set.

    Returns
    -------
    FastArray
        A float array of EMA values. The dtype of this array is specified by the `dtype` parameter.

    Examples
    --------
    `ema_decay` with a decay factor of 0 means "no decay", which means the time component
    is effectively ignored; in this case, `ema_decay` gives the same result as a cumulative-sum.

    >>> data = rt.ones(10)
    >>> times = rt.FastArray([0, 1, 1, 3, 4, 5, 5.5, 10.5, 10.55, 11])
    >>> rt.FastArray.ema_decay(data, times, 0)
    FastArray([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.],
            dtype=float32)

    Simple use case with a 50% decay rate per time-unit.

    >>> rt.FastArray.ema_decay(data, times, 0.5)
    FastArray([1.       , 1.6065307, 2.6065307, 1.958889 , 2.1881263,
            2.3271656, 2.8123984, 1.2308557, 2.2004657, 2.7571077],
            dtype=float32)

    Specify the `dtype` argument explicitly to force the type of the output array
    for additional precision when needed.

    >>> rt.FastArray.ema_decay(data, times, 0.5, dtype=np.float64)
    FastArray([1.        , 1.60653066, 2.60653066, 1.95888904, 2.18812626,
            2.32716567, 2.81239844, 1.23085572, 2.20046579, 2.75710762])

    Provide a filter (boolean mask) to mask out elements which should be skipped over
    during the EMA calculation. Whenever an element has been masked out / skipped,
    the corresponding element of the output array will be the same as the previous
    value.

    >>> filt = rt.FA([True, True, False, True, True, False, False, True, True, True])
    >>> rt.FastArray.ema_decay(data, times, 0.5, filter=filt)
    FastArray([1.        , 1.6065307 , 1.6065307 , 1.5910096 , 1.9649961 ,
            1.1918304 , 0.92819846, 1.0761912 , 2.04962   , 2.6366549 ],
            dtype=float32)

    A reset mask (boolean array) can be also be provided when using a filter.
    Each each position where the reset mask is True, the EMA value is reset to the
    corresponding element of `arr`.

    >>> reset_mask = rt.FA([False, True, False, False, False, False, False, False, False, True])
    >>> rt.FastArray.ema_decay(data, times, 0.5, filter=filt, reset=reset_mask)
    FastArray([1.        , 1.        , 1.        , 1.3678794 , 1.8296608 ,
            1.1097454 , 0.86427057, 1.0709436 , 2.044502  , 1.        ],
            dtype=float32)
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asanyarray(arr)

    if decay_rate is None:
        raise ValueError("ema_decay function requires a kwarg 'decay_rate' floating point value as input")

    if time is None:
        raise ValueError('ema_decay function requires a time array.  Use the "time" kwarg')

    if not isinstance(time, np.ndarray):
        raise ValueError('ema_decay function requires a time numpy array.')

    # require: len(arr) == len(time)
    if arr.shape != time.shape:
        raise ValueError('ema_decay requires the `time` array to be the same shape as the `arr` array.')

    # Allocate the output array
    output = empty_like(arr, dtype=dtype)

    if filter is not None:
        if not isinstance(filter, np.ndarray):
            raise ValueError('ema_decay function requires a filter numpy array.')

        # require: len(arr) == len(filter)
        if arr.shape != filter.shape:
            raise ValueError('ema_decay requires the `filter` array, when supplied, to be the same shape as the `arr` array.')

        if reset is not None:
            if not isinstance(reset, np.ndarray):
                raise ValueError('ema_decay function requires a reset numpy array.')

            # require: len(arr) == len(reset)
            if arr.shape != reset.shape:
                raise ValueError('ema_decay requires the `reset` array, when supplied, to be the same shape as the `arr` array.')

            nb_ema_decay_with_filter_and_reset(arr, output, time, decay_rate, filter, reset)
        else:
            nb_ema_decay_with_filter(arr, output, time, decay_rate, filter)
    else:
        # If a 'reset' was supplied by the user, raise a warning to notify the user the reset won't be applied since they didn't provide a filter.
        if reset is not None:
            raise UserWarning('ema_decay will not apply the `reset` array to the calculation because a filter was not specified.')

        nb_ema_decay(arr, output, time, decay_rate)
    return output

#-------------------------------------------------------
# Keep at bottom of this file
FastArray.register_function('fill_forward', fill_forward)
FastArray.register_function('fill_backward', fill_backward)
FastArray.register_function('ema_decay', ema_decay)
FastArray.register_function('cummax', cummax)
FastArray.register_function('cummin', cummin)