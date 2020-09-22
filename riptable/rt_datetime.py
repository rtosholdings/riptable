__all__ = ['DateTimeBase', 'DateTimeNano', 'TimeSpan', 'Date', 'DateSpan', 'DateTimeUTC',
           'DateTimeNanoScalar', 'TimeSpanScalar', 'DateScalar', 'DateSpanScalar',
           'parse_epoch', 'timestring_to_nano', 'datestring_to_nano', 'datetimestring_to_nano',
           'strptime_to_nano']

import numpy as np
from datetime import datetime as dt
from datetime import date, timezone
from dateutil import tz
import time
import warnings
# import starfish as sf
import riptide_cpp as rc
from typing import Union, Tuple, List, Optional

from .rt_fastarray import FastArray
from .rt_enum import TypeRegister, DisplayArrayTypes, DisplayLength, DisplayJustification, DisplayTextDecoration, \
    TypeId, TimeFormat, NumpyCharTypes, INVALID_DICT, DayOfWeek, SDSFlag, MATH_OPERATION
from .rt_numpy import mask_ori, mask_andi, mask_xori, searchsorted, arange, putmask, isnan, empty, sum, zeros, full, \
    hstack
from .rt_hstack import hstack_any
from .rt_timers import *
from .Utils.rt_display_properties import ItemFormat
from .Utils.rt_metadata import MetaData, meta_from_version, META_VERSION
from .rt_categorical import Categorical

NANOS_PER_MICROSECOND = 1_000
NANOS_PER_MILLISECOND = 1_000_000
NANOS_PER_SECOND = 1_000_000_000
NANOS_PER_MINUTE = NANOS_PER_SECOND * 60
NANOS_PER_HOUR = NANOS_PER_MINUTE * 60
NANOS_PER_DAY = NANOS_PER_HOUR * 24
NANOS_PER_YEAR = NANOS_PER_DAY * 365
NANOS_PER_LEAPYEAR = NANOS_PER_DAY * 366
NANOS_AT_2000 = (NANOS_PER_YEAR * 30) + (NANOS_PER_DAY * 7)
SECONDS_PER_DAY = 60 * 60 * 24
DAYS_PER_YEAR = 365
DAYS_PER_LEAPYEAR = 366
DAYS_AT_2000 = (DAYS_PER_YEAR * 30) + 7
UTC_1970_DAY_SPLITS = FastArray([
    0,  # 1970
    DAYS_PER_YEAR,
    2 * DAYS_PER_YEAR,
    (3 * DAYS_PER_YEAR) + (1),
    (4 * DAYS_PER_YEAR) + (1),
    (5 * DAYS_PER_YEAR) + (1),
    (6 * DAYS_PER_YEAR) + (1),
    (7 * DAYS_PER_YEAR) + (2),
    (8 * DAYS_PER_YEAR) + (2),
    (9 * DAYS_PER_YEAR) + (2),
    (10 * DAYS_PER_YEAR) + (2),  # 1980
    (11 * DAYS_PER_YEAR) + (3),
    (12 * DAYS_PER_YEAR) + (3),
    (13 * DAYS_PER_YEAR) + (3),
    (14 * DAYS_PER_YEAR) + (3),
    (15 * DAYS_PER_YEAR) + (4),
    (16 * DAYS_PER_YEAR) + (4),
    (17 * DAYS_PER_YEAR) + (4),
    (18 * DAYS_PER_YEAR) + (4),
    (19 * DAYS_PER_YEAR) + (5),
    (20 * DAYS_PER_YEAR) + (5),  # 1990
    (21 * DAYS_PER_YEAR) + (5),
    (22 * DAYS_PER_YEAR) + (5),
    (23 * DAYS_PER_YEAR) + (6),
    (24 * DAYS_PER_YEAR) + (6),
    (25 * DAYS_PER_YEAR) + (6),
    (26 * DAYS_PER_YEAR) + (6),
    (27 * DAYS_PER_YEAR) + (7),
    (28 * DAYS_PER_YEAR) + (7),
    (29 * DAYS_PER_YEAR) + (7),
    DAYS_AT_2000,  # 2000
    DAYS_AT_2000 + DAYS_PER_LEAPYEAR,
    DAYS_AT_2000 + (2 * DAYS_PER_YEAR) + (1),
    DAYS_AT_2000 + (3 * DAYS_PER_YEAR) + (1),
    DAYS_AT_2000 + (4 * DAYS_PER_YEAR) + (1),
    DAYS_AT_2000 + (5 * DAYS_PER_YEAR) + (2),
    DAYS_AT_2000 + (6 * DAYS_PER_YEAR) + (2),
    DAYS_AT_2000 + (7 * DAYS_PER_YEAR) + (2),
    DAYS_AT_2000 + (8 * DAYS_PER_YEAR) + (2),
    DAYS_AT_2000 + (9 * DAYS_PER_YEAR) + (3),
    DAYS_AT_2000 + (10 * DAYS_PER_YEAR) + (3),  # 2010
    DAYS_AT_2000 + (11 * DAYS_PER_YEAR) + (3),
    DAYS_AT_2000 + (12 * DAYS_PER_YEAR) + (3),
    DAYS_AT_2000 + (13 * DAYS_PER_YEAR) + (4),
    DAYS_AT_2000 + (14 * DAYS_PER_YEAR) + (4),
    DAYS_AT_2000 + (15 * DAYS_PER_YEAR) + (4),
    DAYS_AT_2000 + (16 * DAYS_PER_YEAR) + (4),
    DAYS_AT_2000 + (17 * DAYS_PER_YEAR) + (5),
    DAYS_AT_2000 + (18 * DAYS_PER_YEAR) + (5),
    DAYS_AT_2000 + (19 * DAYS_PER_YEAR) + (5),
    DAYS_AT_2000 + (20 * DAYS_PER_YEAR) + (5),  # 2020
    DAYS_AT_2000 + (21 * DAYS_PER_YEAR) + (6),
    DAYS_AT_2000 + (22 * DAYS_PER_YEAR) + (6),
    DAYS_AT_2000 + (23 * DAYS_PER_YEAR) + (6),
    DAYS_AT_2000 + (24 * DAYS_PER_YEAR) + (6),
    DAYS_AT_2000 + (25 * DAYS_PER_YEAR) + (7),
    DAYS_AT_2000 + (26 * DAYS_PER_YEAR) + (7),
    DAYS_AT_2000 + (27 * DAYS_PER_YEAR) + (7),
    DAYS_AT_2000 + (28 * DAYS_PER_YEAR) + (7),
    DAYS_AT_2000 + (29 * DAYS_PER_YEAR) + (8),
    DAYS_AT_2000 + (30 * DAYS_PER_YEAR) + (8),  # 2030
    DAYS_AT_2000 + (31 * DAYS_PER_YEAR) + (8),
    DAYS_AT_2000 + (32 * DAYS_PER_YEAR) + (8),
    DAYS_AT_2000 + (33 * DAYS_PER_YEAR) + (9),
    DAYS_AT_2000 + (34 * DAYS_PER_YEAR) + (9),
    DAYS_AT_2000 + (35 * DAYS_PER_YEAR) + (9),
    DAYS_AT_2000 + (36 * DAYS_PER_YEAR) + (9),
    DAYS_AT_2000 + (37 * DAYS_PER_YEAR) + (10),
    DAYS_AT_2000 + (38 * DAYS_PER_YEAR) + (10),
    DAYS_AT_2000 + (39 * DAYS_PER_YEAR) + (10),
    DAYS_AT_2000 + (40 * DAYS_PER_YEAR) + (10),  # 2040
])

# UTC @ midnight, years 1970 - 2040
UTC_1970_SPLITS = FastArray([
    0,  # 1970
    NANOS_PER_YEAR,
    2 * NANOS_PER_YEAR,
    (3 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 1),
    (4 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 1),
    (5 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 1),
    (6 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 1),
    (7 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 2),
    (8 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 2),
    (9 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 2),
    (10 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 2),  # 1980
    (11 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 3),
    (12 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 3),
    (13 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 3),
    (14 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 3),
    (15 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 4),
    (16 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 4),
    (17 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 4),
    (18 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 4),
    (19 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 5),
    (20 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 5),  # 1990
    (21 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 5),
    (22 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 5),
    (23 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 6),
    (24 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 6),
    (25 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 6),
    (26 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 6),
    (27 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 7),
    (28 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 7),
    (29 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 7),
    NANOS_AT_2000,  # 2000
    NANOS_AT_2000 + NANOS_PER_LEAPYEAR,
    NANOS_AT_2000 + (2 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 1),
    NANOS_AT_2000 + (3 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 1),
    NANOS_AT_2000 + (4 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 1),
    NANOS_AT_2000 + (5 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 2),
    NANOS_AT_2000 + (6 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 2),
    NANOS_AT_2000 + (7 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 2),
    NANOS_AT_2000 + (8 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 2),
    NANOS_AT_2000 + (9 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 3),
    NANOS_AT_2000 + (10 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 3),  # 2010
    NANOS_AT_2000 + (11 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 3),
    NANOS_AT_2000 + (12 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 3),
    NANOS_AT_2000 + (13 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 4),
    NANOS_AT_2000 + (14 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 4),
    NANOS_AT_2000 + (15 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 4),
    NANOS_AT_2000 + (16 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 4),
    NANOS_AT_2000 + (17 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 5),
    NANOS_AT_2000 + (18 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 5),
    NANOS_AT_2000 + (19 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 5),
    NANOS_AT_2000 + (20 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 5),  # 2020
    NANOS_AT_2000 + (21 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 6),
    NANOS_AT_2000 + (22 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 6),
    NANOS_AT_2000 + (23 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 6),
    NANOS_AT_2000 + (24 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 6),
    NANOS_AT_2000 + (25 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 7),
    NANOS_AT_2000 + (26 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 7),
    NANOS_AT_2000 + (27 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 7),
    NANOS_AT_2000 + (28 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 7),
    NANOS_AT_2000 + (29 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 8),
    NANOS_AT_2000 + (30 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 8),  # 2030
    NANOS_AT_2000 + (31 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 8),
    NANOS_AT_2000 + (32 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 8),
    NANOS_AT_2000 + (33 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 9),
    NANOS_AT_2000 + (34 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 9),
    NANOS_AT_2000 + (35 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 9),
    NANOS_AT_2000 + (36 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 9),
    NANOS_AT_2000 + (37 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 10),
    NANOS_AT_2000 + (38 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 10),
    NANOS_AT_2000 + (39 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 10),
    NANOS_AT_2000 + (40 * NANOS_PER_YEAR) + (NANOS_PER_DAY * 10),  # 2040
])

MATLAB_EPOCH_DATENUM = 719529
EPOCH_DAY_OF_WEEK = DayOfWeek.Thursday
YDAY_SPLITS = FastArray([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
YDAY_SPLITS_LEAP = FastArray([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
MONTH_STR_ARRAY = FastArray(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# need to hard code the nano cutoffs because FastArray can't do math yet
UTC_YDAY_SPLITS = FastArray([
    (NANOS_PER_DAY * 0),
    (NANOS_PER_DAY * 31),
    (NANOS_PER_DAY * 59),
    (NANOS_PER_DAY * 90),
    (NANOS_PER_DAY * 120),
    (NANOS_PER_DAY * 151),
    (NANOS_PER_DAY * 181),
    (NANOS_PER_DAY * 212),
    (NANOS_PER_DAY * 243),
    (NANOS_PER_DAY * 273),
    (NANOS_PER_DAY * 304),
    (NANOS_PER_DAY * 334)
])

UTC_YDAY_SPLITS_LEAP = FastArray([
    (NANOS_PER_DAY * 0),
    (NANOS_PER_DAY * 31),
    (NANOS_PER_DAY * 60),
    (NANOS_PER_DAY * 91),
    (NANOS_PER_DAY * 121),
    (NANOS_PER_DAY * 152),
    (NANOS_PER_DAY * 182),
    (NANOS_PER_DAY * 213),
    (NANOS_PER_DAY * 244),
    (NANOS_PER_DAY * 274),
    (NANOS_PER_DAY * 305),
    (NANOS_PER_DAY * 335)
])

TIME_FORMATS = {
    1: "%Y%m%d",  # ordinal date
    2: "%#H:%M %p",  # ms from midnight
    3: "%Y%m%d %H:%M:%S",
    4: "%H:%M:%S",
    5: "%H:%M"
}


# ------------------------------------------------------------------------------------
def strptime_to_nano(dtstrings, format, from_tz=None, to_tz='NYC'):
    '''
    Converts datetime string to DateTimeNano object with user-specified format.

    Parameters
    ----------
    dtstrings : array of timestrings
    format    : timestring format

                Currently supports the following escape codes:

                Date:
                -----
                %y      Year without century as zero-padded decimal number.
                %Y      Year with century as decimal number.
                %m      Month as a decimal number (with or without zero-padding).
                %B      Full month name: ['January', 'February', 'March', 'April', 'May', 'June',
                                          'July', 'August', 'September', 'October', 'November', 'December']
                %b      Abbreviated month name: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                %d      Day of the month as a decimal number (with or without zero-padding).

                Time:
                -----
                %H      Hour (24-hour clock) as a decimal number (with or without zero-padding).
                        (Note: if a %p formatter is present, this will be interpretted as a 12-hour clock hour)
                %I      Hour (12-hour clock) as a decimal number (with or without zero-padding).
                        (Note: unlike %H, must be 1-12)
                %p      Locale’s equivalent of either AM or PM.
                %M	    Minute as a decimal number (with or without zero-padding).
                %S	    Second as a decimal number (with or without zero-padding).

    from_tz   : a string for the timezone of origin: 'NYC', 'GMT', 'DUBLIN', etc.
    to_tz     : a string for the timezone that the time will be displayed in


    Other Notes:
    ------------
    Works best with timestrings that include a date:
    - If no year is present in the string, an invalid time will be returned for all values.
    - If no form of year/month/day is present, values will yield a time in 1970.
      Consider using timestring_to_nano(), which also will accept one datestring for all times.

    If the timestring ends in a '.', the following numbers will be parsed as a second fraction. This happens
    automatically, no escape character is required in the format string.

    If no time escape characters are present, will return midnight at all date values.
    If formatted correctly, consider using datestring_to_nano()

    Examples
    --------
    Date, with/without padding:
    >>> dt = FastArray(['02/01/1992', '2/1/1992'])
    >>> fmt = '%m/%d/%Y'
    >>> strptime_to_nano(dt, fmt, from_tz='NYC')
    DateTimeNano([19920201 00:00:00.000000000, 19920201 00:00:00.000000000])

    Date + 24-hour clock:
    >>> dt = FastArray(['02/01/1992 7:48:30', '2/1/1992 19:48:30'])
    >>> fmt = '%m/%d/%Y %H:%M:%S'
    >>> strptime_to_nano(dt, fmt, from_tz='NYC')
    DateTimeNano([19920201 07:48:30.000000000, 19920201 19:48:30.000000000])

    Date + 12-hour clock + am/pm:
    >>> dt = FastArray(['02/01/1992 7:48:30 AM', '2/1/1992 7:48:30 PM'])
    >>> fmt = '%m/%d/%Y %I:%M:%S %p'
    >>> strptime_to_nano(dt, fmt, from_tz='NYC')
    DateTimeNano([19920201 07:48:30.000000000, 19920201 19:48:30.000000000])

    Date + time + second fraction:
    >>> dt = FastArray(['02/01/1992 7:48:30.123456789', '2/1/1992 15:48:30.000000006'])
    >>> fmt = '%m/%d/%Y %H:%M:%S'
    >>> strptime_to_nano(dt, fmt, from_tz='NYC')
    DateTimeNano([19920201 07:48:30.123456789, 19920201 15:48:30.000000006])

    '''
    if isinstance(format, str):
        format = format.encode()

    nano_times = rc.StrptimeToNanos(dtstrings, format)
    return DateTimeNano(nano_times, from_tz=from_tz, to_tz=to_tz)


# ------------------------------------------------------------------------------------
def _possibly_convert_cat(arr):
    '''
    When a cateorical is passed into DateTime functions, we extract the unique categories
    and then re-expand at the end

    Returns
    -------
    samearry, None: if not a categorical
    uniques, cat: if a categorical
    '''
    if isinstance(arr, TypeRegister.Categorical):
        return arr.category_array, arr
    return arr, None


# ------------------------------------------------------------------------------------
def datetimestring_to_nano(dtstring, from_tz=None, to_tz='NYC'):
    '''
    Converts datetime string to DateTimeNano object.
    By default, the timestrings are assumed to be in Eastern Time. If they are already in UTC time, set gmt=True.

    Parameters
    ----------
    dtstring : array of timestrings in format YYYY-MM-DD HH:MM:SS, YYYYMMDD HH:MM:SS.ffffff, etc. (bytestrings/unicode supported)
    from_tz    : a string for the timezone of origin: 'NYC', 'GMT', 'DUBLIN', etc.
    to_tz      : a string for the timezone that the time will be displayed in

    returns DateTimeNano

    See Also: timestring_to_nano(), datestring_to_nano()

    Examples
    --------
    >>> dts = FA(['2012-12-12 12:34:56.001002', '20130303 1:14:15', '2008-07-06 15:14:13'])
    >>> datetimestring_to_nano(dts, from_tz='NYC')
    DateTimeNano([20121212 12:34:56.001002000, 20130303 01:14:15.000000000, 20080706 15:14:13.000000000])

    '''
    nano_times = rc.DateTimeStringToNanos(dtstring)
    return DateTimeNano(nano_times, from_tz=from_tz, to_tz=to_tz)


# ------------------------------------------------------------------------------------
def datestring_to_nano(datestring, time=None, from_tz=None, to_tz='NYC'):
    '''
    Converts date string to DateTimeNano object (default midnight).
    By default, the timestrings are assumed to be in Eastern Time. If they are already in UTC time, set gmt=True.

    Parameters
    ----------
    datestring : array of datestrings in format YYYY-MM-DD or YYYYMMDD (bytestrings/unicode supported)
    time       : a single string or array of strings in the format HH:MM:SS.ffffff (bytestrings/unicode supported)
    from_tz    : a string for the timezone of origin: 'NYC', 'GMT', 'DUBLIN', etc.
    to_tz      : a string for the timezone that the time will be displayed in

    returns DateTimenano

    See Also: timestring_to_nano(), datetimestring_to_nano()

    Examples
    --------
    Date only:

    >>> dates = FA(['2018-01-01', '2018-01-02', '2018-01-03'])
    >>> datestring_to_nano(dates, from_tz='NYC')
    DateTimeNano([20180101 00:00:00.000000000, 20180102 00:00:00.000000000, 20180103 00:00:00.000000000])

    With time:

    >>> dates = FA(['2018-01-01', '2018-01-02', '2018-01-03'])
    >>> datestring_to_nano(dates, time='9:30:00', from_tz='NYC')
    DateTimeNano([20180101 09:30:00.000000000, 20180102 09:30:00.000000000, 20180103 09:30:00.000000000])
    '''

    nano_dates = rc.DateStringToNanos(datestring)
    if time is None:
        result = nano_dates
    else:
        if isinstance(time, (str, bytes)):
            time = TypeRegister.FastArray([time])
        time = rc.TimeStringToNanos(time)

        result = nano_dates + time

    result = DateTimeNano(result, from_tz=from_tz, to_tz=to_tz)

    return result


# ------------------------------------------------------------------------------------
def timestring_to_nano(timestring, date=None, from_tz=None, to_tz='NYC'):
    '''
    Converts timestring to TimeSpan or DateTimeNano object.
    By default, the timestrings are assumed to be in Eastern Time. If they are already in UTC time, set gmt=True.
    If a date is specified, a DateTimeNano object will be returned.
    If a date is not specified, a TimeSpan will be returned.

    Parameters:
    -----------
    timestring : array of timestrings in format HH:MM:SS, H:MM:SS, HH:MM:SS.ffffff (bytestrings/unicode supported)
    date       : a single string or array of date strings in format YYYY-MM-DD (bytestrings/unicode supported)
    from_tz    : a string for the timezone of origin: 'NYC', 'GMT', 'DUBLIN', etc.
    to_tz      : a string for the timezone that the time will be displayed in

    returns TimeSpan or DateTimeNano

    See Also: datestring_to_nano(), datetimestring_to_nano()

    Examples
    ---------

    Return TimeSpan:

    >>> ts = FA(['1:23:45', '12:34:56.000100', '       14:00:00'])
    >>> timestring_to_nano(ts, from_tz='NYC')
    TimeSpan([01:23:45.000000000, 12:34:56.000100000, 14:00:00.000000000])

    With single date string:

    >>> ts = FA(['1:23:45', '12:34:56', '23:22:21'])
    >>> timestring_to_nano(ts, date='2018-02-01', from_tz='NYC')
    DateTimeNano([20180201 01:23:45.000000000, 20180201 12:34:56.000000000, 20180201 23:22:21.000000000])

    Multiple date strings:

    >>> ts = FA(['1:23:45', '12:34:56', '23:22:21'])
    >>> dts = FA(['2018-02-01', '2018-02-07', '2018-05-12'])
    >>> timestring_to_nano(ts, date=dts, from_tz='NYC')
    DateTimeNano([20180201 01:23:45.000000000, 20180207 12:34:56.000000000, 20180512 23:22:21.000000000])

    '''
    nano_times = rc.TimeStringToNanos(timestring)
    if date is None:
        result = TimeSpan(nano_times)
    else:

        if isinstance(date, (str, bytes)):
            date = TypeRegister.FastArray([date])

        date = rc.DateStringToNanos(date)
        result = date + nano_times
        result = DateTimeNano(result, from_tz=from_tz, to_tz=to_tz)

    return result


# ===========================================================================================
def parse_epoch(etime, to_tz='NYC'):
    """Days since epoch and milliseconds since midnight from nanosecond timestamps.

    Parameters
    ----------
    etime : array-like
        UTC nanoseconds.
    to_tz : str, default 'NYC'
        TimeZone short string - see riptable.rt_timezone.
        This routine didn't used to take a timezone, so it defaults to the previous setting.

    Used in the phonyx data loader.
    Returns
    -------
    days : array (int32)
        Days since epoch.
    millis : array (float64)
        Milliseconds since midnight.
    """
    dtn = DateTimeNano(etime, from_tz='UTC', to_tz=to_tz)
    return dtn.days_since_epoch, dtn.millis_since_midnight()


# ------------------------------------------------------------
def _apply_inv_mask(arr1, arr2, fillval=None, arr1_inv_mask=None, arr2_inv_mask=None):
    """Preserve NaN date and time values in the final result of date/time class operations.
    Called by time fraction properties and math operations.
    """
    if isinstance(arr1, np.ndarray):
        if len(arr1) == 1:
            # broadcast array of 1 path
            if arr1[0] <= 0:
                return TypeRegister.FastArray([INVALID_DICT[arr2.dtype.num]])
            return arr2
        else:
            if arr1_inv_mask is None:
                arr1_inv_mask = arr1.isnan()

            if fillval is None:
                # use the sentinel or nan for the return array type, e.g. year() returns int32
                fillval = INVALID_DICT[arr2.dtype.num]
            putmask(arr2, arr1_inv_mask, fillval)

            # apply the invalid mask from an operation with another array
            if arr2_inv_mask is not None:
                # return invalid fill, fixes broadcasting if math operations
                # was with a scalar or single item array
                if np.isscalar(arr2_inv_mask):
                    if arr2_inv_mask:
                        arr2[:] = fillval
                elif len(arr2_inv_mask) == 1:
                    if arr2_inv_mask[0]:
                        arr2[:] = fillval
                else:
                    putmask(arr2, arr2_inv_mask, fillval)
            return arr2
    else:
        # scalar path
        if arr1 <= 0:
            return INVALID_DICT[arr2.dtype.num]
        return arr2


# ========================================================
class DateTimeBase(FastArray):
    """Base class for DateTimeNano and TimeSpan.
    Both of these subclasses have times with nanosecond precision.
    """
    DEFAULT_FORMATTER = time.strftime
    PRECISION = 9
    NAN_TIME = 0

    # ------------------------------------------------------------
    def __new__(cls, values):
        instance = np.asarray(values).view(cls)
        instance._display_length = DisplayLength.Long
        return instance

    # ------------------------------------------------------------
    @property
    def _fa(self):
        return self.view(FastArray)

    # ------------------------------------------------------------
    @property
    def display_length(self):
        if not hasattr(self, '_display_length'):
            self._display_length = DisplayLength.Long
        return self._display_length

    # ------------------------------------------------------------
    def get_classname(self):
        return __class__.__name__

    # ------------------------------------------------------------
    def display_item(self, utcnano):
        raise NotImplementedError(f"DateTimeBase subclasses need to override this method.")

    # ------------------------------------------------------------
    def _meta_dict(self, name=None):
        raise NotImplementedError(f"DateTimeBase subclasses need to override this method.")

    # ------------------------------------------------------------
    def _as_meta_data(self, name=None):
        # ** TODO: Date, DateSpan, DateTimeNano, TimeSpan all have very similar
        # versions of this routine - collapse into one
        if name is None:
            name = self.get_name()
        meta = MetaData(self._meta_dict(name))
        return {meta['name']: self._fa}, [SDSFlag.OriginalContainer + SDSFlag.Stackable], meta.string

    # ------------------------------------------------------------
    def _build_sds_meta_data(self, name, **kwargs):
        """Build meta data for DateTimeNano
        """
        meta = MetaData(self._meta_dict(name=name))
        # for now there's only one array in this FastArray subclass
        cols = []
        tups = []
        return meta, cols, tups

    # ------------------------------------------------------------
    def _build_string(self):
        def qwrap(timestring):
            return "".join(["'", timestring, "'"])

        _slicesize = int(np.floor(DateTimeBase.MAX_DISPLAY_LEN / 2))
        _asize = len(self)

        # DFUNC = self.display_item
        fmt, DFUNC = self.display_query_properties()

        # print with break
        if _asize > DateTimeBase.MAX_DISPLAY_LEN:
            left_idx = self.view(FastArray)[:_slicesize]
            right_idx = self.view(FastArray)[-_slicesize:]

            left_strings = [qwrap(DFUNC(i, fmt)) for i in left_idx]
            break_string = ["..."]
            right_strings = [qwrap(DFUNC(i, fmt)) for i in right_idx]
            all_strings = left_strings + break_string + right_strings


        # print full
        else:
            all_strings = [qwrap(DFUNC(i, fmt)) for i in self]

        result = ", ".join(all_strings)
        return result

    # ------------------------------------------------------------
    @staticmethod
    def _add_nano_ext(utcnano, timestr):
        precision = DateTimeBase.PRECISION
        if precision > 0:
            if precision > 9:
                precision = 9

            power = 10 ** precision
            nanos = int(utcnano % power)
            nanostr = str(nanos).zfill(precision)
            timestr = timestr + "." + nanostr
        return timestr

    # ------------------------------------------------------------
    def __str__(self):
        return self._build_string()

    # ------------------------------------------------------------
    def __repr__(self):
        return self.get_classname() + "([" + self._build_string() + "])"

    # ------------------------------------------------------------
    def __getitem__(self, fld):
        result = self._fa.__getitem__(fld)
        if isinstance(result, FastArray):
            # possible fix for strides bug
            # if result.strides[0] != result.itemsize:
            #    result = result.copy()
            result = self.newclassfrominstance(result, self)
        if np.isscalar(result):
            return self.get_scalar(result)
        return result

    # -------------------------------------------------------------
    def _math_error_string(self, value, operator, reverse=False):
        if reverse:
            a = value
            b = self
        else:
            a = self
            b = value
        return f"unsupported operand type(s) for {operator}: {type(a).__name__} {type(b).__name__}"

    # ------------------------------------------------------------
    def _funnel_mathops(self, funcname, value):
        '''
        Wrapper for all math operations on Date and DateSpan.

        Both subclasses need to take over:
        _check_mathops_nano()
        _check_mathops()

        maybe... still testing
        _build_mathops_result()

        Easier to catch forbidden operations here.
        '''

        # if funcname in self.forbidden_mathops:
        #    raise TypeError(f'Cannot perform {funcname} on {self.__class__.__name__} object.')

        inv_mask = self.isnan()
        other_inv_mask = None
        return_type = None
        caller = self._fa

        # check if operand has nano precision, set invalid, return type accordingly
        value, other_inv_mask, return_type, caller = self._check_mathops_base(funcname, value, other_inv_mask, return_type, caller)

        # perform main math operation on fast array
        func = getattr(caller, funcname)
        result = func(value)

        # set return type, preserve invalids for non-nano operands
        if return_type is None:
            return_type, other_inv_mask = self._check_mathops(funcname, value)

        # if return type is still None, returning invalid fill
        if return_type is None:
            return other_inv_mask

        # apply invalid mask(s) and wrap result in final return type
        result = self._build_mathops_result(value, result, inv_mask, other_inv_mask, return_type)
        return result

    # ------------------------------------------------------------
    def copy(self, order='K'):
        instance = self._fa.copy(order=order)
        return self.newclassfrominstance(instance, self)


# ========================================================
class TimeStampBase():
    """Parent class for DateTimeNano and Date.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------
    def _year(self, arr, fix_dst=False):
        """
        Parameters
        ----------
        arr : array
            Underlying FastArray or result of previous timezone fixup.
        fix_dst : bool, default False
            If True, adjust array's stored times to match display. (DateTimeNano only)

        Returns
        -------
        int32 FastArray of the year.  For example 2003 is the integer 2003.
        """
        if fix_dst:
            arr = self._timezone.fix_dst(arr)
        result = self._year_splits.searchsorted(arr, side='right').astype(np.int32, copy=False) + 1969
        return result

    # ------------------------------------------------------------
    def _month(self, arr=None, fix_dst=False):
        '''
        Internal year to avoid performing the daylight savings fixup multiple times.
        '''
        if arr is None:
            if fix_dst:
                arr = self._timezone.fix_dst(self._fa)
                fix_dst = False
            else:
                arr = self._fa
        year = self._year(arr, fix_dst=fix_dst)
        startyear = arr - self._year_splits[year - 1970]

        maskleap = (year % 4) == 0

        # get the months for non-leaps
        smonth = self._yearday_splits.searchsorted(startyear, side='right')

        # get the months for leap and fix any leapyears with maskleap
        putmask(smonth, maskleap, self._yearday_splits_leap.searchsorted(startyear, side='right'))
        return smonth.astype(np.int32, copy=False).view(FastArray)

    # ------------------------------------------------------------
    def _preserve_invalid_comparison(self, caller, other, funcname):
        """Date and DateTimeNano have multiple values for nan (0 and integer sentinel).
        Both of their compare checks need to preserve nans in the result the same way.
        """
        func = getattr(caller, funcname)
        result = func(other)
        if funcname == '__ne__':
            result += self.isnan()
        else:
            result *= self.isnotnan()
        return result


# ========================================================
class DateBase(FastArray):
    """Parent class for Date and Datespan.
    Both of these subclasses have times with day precision.
    """
    # NAN_DATE = INVALID_DICT[np.dtype(np.int32).num]
    NAN_DATE = 0

    # ------------------------------------------------------------
    def __new__(cls, arr, **kwargs):
        return arr.view(cls)

    # ------------------------------------------------------------
    def __init__(cls, arr, **kwargs):
        pass

    # ------------------------------------------------------------
    @property
    def _fa(self):
        return self.view(FastArray)

    # ------------------------------------------------------------
    def __str__(self):
        return self._build_string()

    # ------------------------------------------------------------
    def __repr__(self):
        return self.get_classname() + "([" + self._build_string() + "])"

    # ------------------------------------------------------------
    def strftime(self, format, dtype='O'):
        '''
        Converts DateTimeNano to an array of object strings or a scalar string.
        This routine has not been sped up yet.

        Other Parameters
        ----------------
        dtype: defaults to 'O', can change to 'S' or 'U'

        Examples
        --------
        >>> rt.Date(rt.utcnow(4)).strftime('%D')
        array(['11/04/19', '11/04/19', '11/04/19', '11/04/19'], dtype=object)

        See Also
        ---------
        http://strftime.org  for format strings
        datetime.datetime.strftime

        '''
        if isinstance(self, np.ndarray):
            return np.asarray([dt.utcfromtimestamp(timestamp).strftime(format) for timestamp in self._fa* SECONDS_PER_DAY], dtype=dtype)
        else:
            return dt.strftime(dt.utcfromtimestamp(self * SECONDS_PER_DAY), format)

    # ------------------------------------------------------------
    @property
    def display_length(self):
        if not hasattr(self, '_display_length'):
            self._display_length = DisplayLength.Long
        return self._display_length

    # # TODO uncomment when starfish is implemented and imported
    # def _sf_display_query_properties(self):
    #     itemformat = sf.ItemFormat({'length':self.display_length,
    #                                 'align':sf.DisplayAlign.Right})
    #     return itemformat, self.display_convert_func

    # ------------------------------------------------------------
    def display_query_properties(self):
        # if TypeRegister.DisplayOptions.STARFISH:
        #    return self._sf_display_query_properties()
        '''
        Each instance knows how to format its time strings. The formatter is specified in TIME_FORMATS
        The length property of item_format stores the index into TIME_FORMATS for the display_convert_func
        '''
        item_format = ItemFormat(
            length=self.display_length,
            justification=DisplayJustification.Right,
            can_have_spaces=True,
            decoration=None,
        )
        convert_func = self.display_convert_func
        return item_format, convert_func

    # ------------------------------------------------------------
    def _build_string(self):
        _slicesize = int(np.floor(DateTimeBase.MAX_DISPLAY_LEN / 2))
        _asize = len(self)

        fmt, DFUNC = self.display_query_properties()

        # print with break
        if _asize > DateTimeBase.MAX_DISPLAY_LEN:
            left_idx = self.view(FastArray)[:_slicesize]
            right_idx = self.view(FastArray)[-_slicesize:]

            left_strings = [f"'{DFUNC(i, fmt)}'" for i in left_idx]
            break_string = ["..."]
            right_strings = [f"'{DFUNC(i, fmt)}'" for i in right_idx]
            all_strings = left_strings + break_string + right_strings

        # print full
        else:
            all_strings = [f"'{DFUNC(i, fmt)}'" for i in self]

        result = ", ".join(all_strings)
        return result

    def __getitem__(self, fld):
        '''
        Restore the Date/DateSpan class after the indexing operation.
        '''
        result = self._fa[fld]
        if isinstance(result, np.ndarray):
            # possible fix for strides bug
            # if result.strides[0] != result.itemsize:
            #    result = result.copy()
            return self.newclassfrominstance(result, self)
        if np.isscalar(result):
            return self.get_scalar(result)

        return result

    # ------------------------------------------------------------
    def _funnel_mathops(self, funcname, value):
        '''
        Wrapper for all math operations on Date and DateSpan.

        Both subclasses need to take over:
        _check_mathops_nano()
        _check_mathops()

        maybe... still testing
        _build_mathops_result()

        Easier to catch forbidden operations here.
        '''
        if funcname in self.forbidden_mathops:
            raise TypeError(f'Cannot perform {funcname} on {self.__class__.__name__} object.')

        inv_mask = self.isnan()
        other_inv_mask = None
        return_type = None
        caller = self._fa

        # check if operand has nano precision, set invalid, return type accordingly
        value, other_inv_mask, return_type, caller = self._check_mathops_nano(funcname, value, other_inv_mask, return_type, caller)

        # perform main math operation on fast array
        func = getattr(caller, funcname)
        result = func(value)
        # set return type, preserve invalids for non-nano operands
        if return_type is None:
            return_type, other_inv_mask = self._check_mathops(funcname, value)

        # if return type is still None, returning invalid fill
        if return_type is None:
            return other_inv_mask

        # apply invalid mask(s) and wrap result in final return type
        result = self._build_mathops_result(value, result, inv_mask, other_inv_mask, return_type)
        return result

    # ------------------------------------------------------------
    def _build_mathops_result(self, value, result, inv_mask, other_inv_mask, return_type):
        # restore invalid for Date and other operand if necessary
        # print('**DateBase._build_mathops_result')
        # print('value',value)
        # print('result',result)
        # print('inv_mask',inv_mask)
        # print('other_inv_mask',other_inv_mask)
        # print('return type',return_type)

        result = _apply_inv_mask(self, result, fillval=self.NAN_DATE, arr1_inv_mask=inv_mask, arr2_inv_mask=other_inv_mask)

        if not isinstance(result, return_type):
            if return_type == DateTimeNano:
                try:
                    # base on original to_tz
                    # use a try, because this may be hit by TimeSpan operand (no timezone)
                    to_tz = value._timezone._to_tz
                except:
                    to_tz = 'GMT'
                result = DateTimeNano(result, from_tz='GMT', to_tz=to_tz)

            else:
                result = return_type(result)

        return result

    # -------------------------------------------------------------
    def min(self, **kwargs):
        '''
        Earliest date / datespan in array.

        Returns
        -------
        obj:`Date`
            An array with length 1.

        Note
        ----
        This returns an array, not a scalar. However, broadcasting rules will apply to operations with it.
        '''
        return self.__class__([self._fa.min()])

    # -------------------------------------------------------------
    def max(self, **kwargs):
        '''
        Latest date / datespan in array.

        Returns
        -------
        obj:`Date`
            An array with length 1.

        Note
        ----
        This returns an array, not a scalar. However, broadcasting rules will apply to operations with it.
        '''
        return self.__class__([self._fa.max()])

    def _meta_dict(self, name=None):
        classname = self.__class__.__name__
        if name is None:
            name = classname
        metadict = {
            'name': name,
            'typeid': getattr(TypeId, classname),
            'classname': classname,
            'ncols': 0,
            'version': self.MetaVersion,
            'author': 'python',

            'instance_vars': {
                '_display_length': self.display_length,
            },

            '_base_is_stackable': SDSFlag.Stackable
        }
        return metadict

    # ------------------------------------------------------------
    def _as_meta_data(self, name=None):
        if name is None:
            name = self.get_name()
        meta = MetaData(self._meta_dict(name=name))
        return {meta['name']: self._fa}, [SDSFlag.OriginalContainer + SDSFlag.Stackable], meta.string

    # ------------------------------------------------------------
    def _build_sds_meta_data(self, name):
        meta = MetaData(self._meta_dict(name=name))
        cols = []
        tups = []
        return meta, cols, tups

    # ------------------------------------------------------------
    @classmethod
    def _from_meta_data(cls, arrdict, arrflags, meta):
        meta = MetaData(meta)
        instance = cls([*arrdict.values()][0])
        # combine loaded meta variables with class defaults
        vars = meta['instance_vars']
        for k, v in cls.MetaDefault.items():
            vars.setdefault(k, v)
        for k, v in vars.items():
            setattr(instance, k, v)
        return instance

    # ------------------------------------------------------------
    def copy(self, order='K'):
        instance = self._fa.copy(order=order)
        return self.newclassfrominstance(instance, self)

    # ------------------------------------------------------------
    @classmethod
    def newclassfrominstance(cls, instance, origin):
        result = instance.view(cls)
        result._display_length = origin.display_length
        return result


# ========================================================
class Date(DateBase, TimeStampBase):
    '''
    Date arrays have an underlying int32 array. The array values are number of days since January 1st. 1970.
    Can be initialized from integer date values, strings, or matlab ordinal dates.

    Parameters:
    -----------
    arr         : array, categorical, list, or scalar
    from_matlab : indicates that values are from matlab datenum
    format      : if initialized with string, specify a format string for strptime to parse date information
                  otherwise, will assume format is YYYYMMDD

    Examples
    ---------
    From strings:

    >>> datestrings = tile(np.array(['2018-02-01', '2018-03-01', '2018-04-01']), 3)
    >>> Date(datestrings)
    Date([2018-02-01, 2018-03-01, 2018-04-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-02-01, 2018-03-01, 2018-04-01])

    From riptable.Categorical (sometimes Matlab data comes in this way):

    >>> c = Categorical(datestrings)
    >>> c
    Categorical([2018-02-01, 2018-03-01, 2018-04-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-02-01, 2018-03-01, 2018-04-01]) Length: 9
      FastArray([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=int8) Base Index: 1
      FastArray(['2018-02-01', '2018-03-01', '2018-04-01'], dtype='<U10') Unique count: 3
    >>> d = Date(c)
    >>> d
    Date([2018-02-01, 2018-03-01, 2018-04-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-02-01, 2018-03-01, 2018-04-01])

    From Matlab datenum:

    >>> d = FA([737061.0, 737062.0, 737063.0, 737064.0, 737065.0])
    >>> Date(dates, from_matlab=True)
    Date([2018-01-01, 2018-01-02, 2018-01-03, 2018-01-04, 2018-01-05])

    From riptable DateTimeNano:

    >>> dtn = DateTimeNano.random(5)
    >>> dtn
    DateTimeNano([20150318 13:28:01.853344227, 20150814 17:34:43.991344669, 19761204 04:30:52.680683459, 20120524 06:44:13.482424912, 19830803 17:12:54.771824294])
    >>> Date(dtn)
    Date([2015-03-18, 2015-08-14, 1976-12-04, 2012-05-24, 1983-08-03])
    '''
    # for .SDS file format
    MetaVersion = 1
    MetaDefault = {
        # vars for container loader
        'name': 'Date',
        'typeid': TypeId.Date,
        'version': 0,  # if no version, assume before versions implemented

        'instance_vars': {
            '_display_length': DisplayLength.Long
        }
    }
    forbidden_mathops = ('__mul__', '__imul__')

    def __new__(cls, arr, from_matlab=False, format=None):
        instance = None
        if isinstance(arr, list) or np.isscalar(arr):
            arr = FastArray(arr)

        if isinstance(arr, np.ndarray):

            # if this the same class, do nothing
            if not isinstance(arr, Date):

                # sometimes matlab dates are categoricals
                if isinstance(arr, TypeRegister.Categorical):
                    try:
                        cats = arr.category_array
                        # flip to correct integer before re-expanding
                        if cats.dtype.char in ('U', 'S'):
                            cats = cls._convert_datestring(cats).astype(np.int32, copy=False)
                            arr = TypeRegister.Categorical(arr._fa, cats)
                        arr = arr.expand_array
                    except:
                        raise TypeError(f'Could not re-expand categorical to array in mode {arr.category_mode.name}')

                # fix datetimenano so the days match display (account for previous daylight savings fixup)
                elif isinstance(arr, TypeRegister.DateTimeNano):
                    # there is a bug here -- do not think a timezone fixup is nec
                    # arr = arr._timezone.fix_dst(arr._fa, arr._timezone._dst_cutoffs)
                    arr = arr._fa // NANOS_PER_DAY

                # flip strings to days from 1970
                if arr.dtype.char in ('U', 'S'):
                    arr = cls._convert_datestring(arr, format=format)

                # flip matlab ordinal dates to days from 1970
                if from_matlab:
                    arr = cls._convert_matlab_days(arr)

                elif arr.dtype.char in NumpyCharTypes.AllInteger + NumpyCharTypes.AllFloat:
                    arr = arr.astype(np.int32, copy=False)

                else:
                    raise TypeError(f'Could not initialize Date object with array of type {arr.dtype}.')

        else:
            raise TypeError(f'Date objects must be initialized with numeric or string arrays, lists or scalars. Got {type(arr)}')


        instance = arr.view(cls)
        instance._display_length = DisplayLength.Long
        return instance

    # ------------------------------------------------------------
    def __init__(self, arr, from_matlab=False, format=None):
        pass

    # ------------------------------------------------------------
    def get_scalar(self, scalarval):
        return DateScalar(scalarval, _from=self)

    # -------------------------------------------------------
    def diff(self, periods=1):
        '''
        Returns
        -------
        DateSpan
        '''
        result = self._fa.diff(periods=periods)
        return DateSpan(result)

    # ------------------------------------------------------------
    @classmethod
    def _convert_datestring(cls, arr, format=None):
        '''
        For construction from array of strings or categorical.
        '''

        if format is None:
            arr = datestring_to_nano(arr, from_tz='UTC')._fa // NANOS_PER_DAY
        # default assumes YYYYMMDD
        else:
            arr = strptime_to_nano(arr, format, from_tz='UTC')._fa // NANOS_PER_DAY

        return arr

    # ------------------------------------------------------------
    @classmethod
    def _convert_matlab_days(cls, arr):
        '''
        TODO: move this to a more generic superclass - almost exactly the same as DateTimeNano._convert_matlab_days

        Parameters:
        -----------

        arr      : array of matlab datenums (1 is 1-Jan-0000)
        timezone : TimeZone object from DateTimeNano constructor


        Converts matlab datenums to an array of int64 containing utc nanoseconds.
        '''
        inv_mask = isnan(arr)

        # matlab dates come in as float
        arr = FastArray(arr, dtype=np.int32)
        arr = arr - MATLAB_EPOCH_DATENUM

        putmask(arr, inv_mask, cls.NAN_DATE)
        return arr

    # ------------------------------------------------------------
    def get_classname(self):
        return __class__.__name__

    # ------------------------------------------------------------
    @staticmethod
    def display_convert_func(date_num, itemformat: ItemFormat):
        # TODO: apply ItemFormat options that were passed in
        return Date.format_date_num(date_num, itemformat)

    # ------------------------------------------------------------
    @staticmethod
    def format_date_num(date_num, itemformat):
        if date_num == DateBase.NAN_DATE or date_num == INVALID_DICT[np.dtype(np.int32).num]:
            return 'Inv'
        format_str = Date._parse_item_format(itemformat)
        localzone = tz.gettz('GMT')
        try:
            timestr = dt.fromtimestamp((date_num * SECONDS_PER_DAY), timezone.utc)
            timestr = timestr.astimezone(localzone)
            timestr = timestr.strftime(format_str)
        except:
            raise ValueError(f'Date number {date_num} is not a valid value for Date() object.')
        return timestr

    # ------------------------------------------------------------
    @staticmethod
    def _parse_item_format(itemformat):
        return '%Y-%m-%d'

    # ------------------------------------------------------------
    def fill_invalid(self, shape=None, dtype=None, inplace=True):
        arr = self._fill_invalid_internal(shape=shape, dtype=self.dtype, fill_val=self.NAN_DATE, inplace=inplace)
        if arr is None:
            return
        return Date(arr)

    # ------------------------------------------------------------
    def isnan(self):
        """Both NaN date (0) and integer sentinel value are considered NaN.
        """
        return self._fa.isnanorzero()

    # ------------------------------------------------------------
    def isnotnan(self):
        """Both NaN date (0) and integer sentinel value are considered NaN.
        """
        return ~self.isnan()

    # ------------------------------------------------------------
    @property
    def yyyymmdd(self):
        return DateTimeNano(self._fa * NANOS_PER_DAY, from_tz='GMT', to_tz='GMT').yyyymmdd

    # ------------------------------------------------------------
    @property
    def _year_splits(self):
        """Midnght on Jan. 1st from 1970 - 1940 in utc nanoseconds."""
        return UTC_1970_DAY_SPLITS

    # ------------------------------------------------------------
    @property
    def _yearday_splits(self):
        """Midnight on the 1st of the month in dayssince the beginning of the year."""
        return YDAY_SPLITS

    # ------------------------------------------------------------
    @property
    def _yearday_splits_leap(self):
        """Midnight on the 1st of the month in days since the beginning of the year during a leap year."""
        return YDAY_SPLITS_LEAP

    # ------------------------------------------------------------
    @property
    def year(self):
        '''
        Returns integer array of year value
        Currently limited to 1970 - 2040, add to UTC_1970_DAY_SPLITS table to expand range.

        >>> d = Date(['2016-02-01', '2017-02-01', '2018-02-01'])
        >>> d.year
        FastArray([2016, 2017, 2018])
        '''
        year = self._year(self._fa, fix_dst=False)
        return _apply_inv_mask(self, year)

    # ------------------------------------------------------------
    @property
    def month(self, arr=None):
        '''
        Returns zero-based integer array of month value 1=Jan, 2=Feb, etc.

        >>> d = Date(['2000-02-29', '2018-12-25'])
        >>> d.month
        FastArray([ 2, 12])
        '''
        return _apply_inv_mask(self, self._month())

    # ------------------------------------------------------------
    @property
    def monthyear(self, arr=None):
        '''
        Returns a string with three letter month + 4 digit year

        Examples
        --------
        >>> d = Date(['2000-02-29', '2018-12-25'])
        >>> d.monthyear
        FastArray([ 'Feb2000','Dec2018'])
        '''
        month = self.month
        yearstr = self.year.astype('S')
        return MONTH_STR_ARRAY[month - 1] + yearstr

    # ------------------------------------------------------------
    @property
    def is_leapyear(self):
        '''
        Returns boolean array, True when date was during a leap year.

        >>> d = Date(['2000-01-01', '2001-01-01'])
        >>> d.is_leapyear
        FastArray([ True, False])
        '''
        year = self._year(self._fa, fix_dst=False)
        arr = self._fa - self._year_splits[year - 1970]
        maskleap = year % 4 == 0
        return maskleap

    # ------------------------------------------------------------
    @property
    def day_of_year(self):
        '''
        Returns one-based integer array where January 1st = 1

        >>> d = Date(['2019-01-01', '2019-02-14'])
        >>> d.day_of_year

        FastArray([ 1, 45])
        '''
        year = self._year(self._fa, fix_dst=False)
        arr = self._fa - self._year_splits[year - 1970]
        arr += 1
        return _apply_inv_mask(self, arr)

    # ------------------------------------------------------------
    @property
    def day_of_month(self):
        year = self._year(self._fa, fix_dst=False)

        # subtract the days from start of year so all times are in MM-DD, etc.
        startyear = self._fa - self._year_splits[year - 1970]

        # treat the whole array like a non-leapyear
        startmonth_idx = self._yearday_splits.searchsorted(startyear, side='right') - 1
        monthtime = startyear - self._yearday_splits[startmonth_idx]

        # fix up the leapyears with a different yearday split table
        leapmask = (year % 4) == 0
        startmonth_idx_leap = self._yearday_splits_leap.searchsorted(startyear[leapmask], side='right') - 1
        monthtime[leapmask] = startyear[leapmask] - self._yearday_splits_leap[startmonth_idx_leap]

        # unlike month, weekday, hour, etc. day of month starts at 1
        monthday = monthtime + 1

        return _apply_inv_mask(self, monthday)

    # ------------------------------------------------------------
    @property
    def day_of_week(self):
        '''
        Returns array of integers from Monday (0) -> Sunday (6)

        >>> d = Date(['2019-02-11', '2019-02-12', '2019-02-13', '2019-02-14', '2019-02-15', '2019-02-16', '2019-02-17'])
        >>> d.day_of_week
        FastArray([0, 1, 2, 3, 4, 5, 6])

        '''
        arr = (self._fa + EPOCH_DAY_OF_WEEK) % 7
        return _apply_inv_mask(self, arr)

    # ------------------------------------------------------------
    @property
    def is_weekend(self):
        '''
        Returns boolean array, True when day of week is Saturday or Sunday

        >>> d = Date(['2019-02-11', '2019-02-12', '2019-02-13', '2019-02-14', '2019-02-15', '2019-02-16', '2019-02-17'])
        >>> d.is_weekend
        FastArray([False, False, False, False, False,  True,  True])

        '''
        return _apply_inv_mask(self, self.day_of_week > 4)

    # ------------------------------------------------------------
    @property
    def is_weekday(self):
        '''
        Returns boolean array, True when day of week is Monday-Friday

        >>> d = Date(['2019-02-11', '2019-02-12', '2019-02-13', '2019-02-14', '2019-02-15', '2019-02-16', '2019-02-17'])
        >>> d.is_weekday
        FastArray([ True,  True,  True,  True,  True, False, False])

        '''
        return _apply_inv_mask(self, self.day_of_week < 5)

    # ------------------------------------------------------------
    @property
    def seconds_since_epoch(self):
        '''
        Many existing python datetime routines expect seconds since epoch.
        This call is to eliminate "magic numbers" like 3600 from code.
        '''
        return _apply_inv_mask(self, self._fa * SECONDS_PER_DAY)

    # ------------------------------------------------------------
    @classmethod
    def hstack(cls, dates):
        '''
        hstacks Date objects and returns a new Date object.
        Will be called by riptable.hstack() if the first item in the sequence is a Date object.

        Parameters:
        -----------
        dates : list or tuple of Date objects


        >>> d1 = Date('2015-02-01')
        >>> d2 = Date(['2016-02-01', '2017-02-01', '2018-02-01'])
        >>> hstack([d1, d2])
        Date([2015-02-01, 2016-02-01, 2017-02-01, 2018-02-01])

        '''
        # pass the subclass to the parent class routine
        return hstack_any(dates, cls, Date)

    # ------------------------------------------------------------
    @classmethod
    def range(cls, start, end=None, days=None, step=1, format=None, closed=None):
        """
        Returns a Date object of dates from start date to end date.

        Parameters
        ----------
        start : str or int
            Start date in int format YYYYMMDD, or string in ``format``.
        end : str or int, optional
            Start date in int format YYYYMMDD, or string in ``format``.
            If not specified, days is required.
        days : int, optional (required if ``end`` is None)
            Number of days to generate.
        step : int, optional, default 1
            Spacing between date values.
        format : str, optional
            Format to convert start/end values if they are string
        closed : `left`, `right`, or None (default)
            If `left`, omit the end date.
            If `right`, omit the start date.
            If None, include both.
            Only applies when constructing from start, end date with step of 1.

        Examples
        --------
        >>> Date.range('2019-02-01', '2019-02-07')
        Date([2019-02-01, 2019-02-02, 2019-02-03, 2019-02-04, 2019-02-05, 2019-02-06, 2019-02-07])

        >>> Date.range('2019-02-01', '2019-02-07', step=2)
        Date([2019-02-01, 2019-02-03, 2019-02-05])

        >>> Date.range('2019-02-01', '2019-02-07', closed='right')
        Date([2019-02-02, 2019-02-03, 2019-02-04, 2019-02-05, 2019-02-06, 2019-02-07])

        Returns
        -------
        `Date`
            Range of dates in given interval spaced by `step`.

        """
        if isinstance(start, (int, np.integer)):
            start = str(start)

        # convert separately for more accurate error
        if isinstance(start, (str, bytes)):
            start = cls(start, format=format)._fa[0]
        else:
            raise TypeError(f'Start date must be string or integer. Got {type(start)}')

        if end is None:
            if days is None:
                raise ValueError(f'Must set either ``end`` or ``days`` keyword.')
            # compensate for step
            end = start + (days * step)
            end = cls(end)._fa[0]
        else:
            if isinstance(end, (int, np.integer)):
                end = str(end)
            if not isinstance(end, (str, bytes)):
                raise TypeError(f'End date must be string or integer. Got {type(start)}')
        end = cls(end, format=format)._fa[0]

        if days is None and step == 1:
            # include one or both ends
            if closed is None:
                end += 1
            elif closed == 'right':
                end += 1
                start += 1
            elif closed == 'left':
                pass
            else:
                raise ValueError(f'Closed has to be either "left", "right" or None. Got {closed}')

        arr = arange(start, end, step, dtype=np.int32)
        return cls(arr)

    # ------------------------------------------------------------
    def _date_compare_check(self, funcname, other):
        '''
        Funnel for all comparison operations.
        Helps Date interact with DateTimeNano, TimeSpan.
        '''

        caller = self._fa

        if isinstance(other, (DateSpan, TimeSpan, DateSpanScalar, TimeSpanScalar)):
            raise TypeError(f'Cannot perform {funcname} comparison operation between {type(self)} and {type(other)}.')

        elif isinstance(other, DateTimeNano):
            caller = self._fa * NANOS_PER_DAY
            to_tz = other._timezone._to_tz
            # fix the timezone to match the display of the DateTimeNano
            caller = DateTimeNano(self._fa * NANOS_PER_DAY, from_tz=to_tz, to_tz=to_tz)

        # looks weird now, saving explicit branchesfor if any forbidden types appear
        elif isinstance(other, Date):
            other = other._fa

        elif isinstance(other, (str, bytes)):
            other = Date(other)

        # Categorical will fall through to constructor too
        elif isinstance(other, np.ndarray):
            other = Date(other)
        # let everything else fall through for FastArray to catch

        # restore invalids
        return self._preserve_invalid_comparison(caller, other, funcname)

    # -------------------COMPARISONS------------------------------
    # ------------------------------------------------------------
    def __ne__(self, other):
        return self._date_compare_check('__ne__', other)

    def __eq__(self, other):
        return self._date_compare_check('__eq__', other)

    def __ge__(self, other):
        return self._date_compare_check('__ge__', other)

    def __gt__(self, other):
        return self._date_compare_check('__gt__', other)

    def __le__(self, other):
        return self._date_compare_check('__le__', other)

    def __lt__(self, other):
        return self._date_compare_check('__lt__', other)

    # ------------------------------------------------------------
    def __add__(self, value):
        '''
        Addition rules:
        ------------------

        Date + Date = TypeError
        Date + DateTimeNano = TypeError
        Date + DateSpan = Date
        Date + TimeSpan = DateTimeNano

        All other operands will be treated as DateSpan and return Date.
        '''
        return self._funnel_mathops('__add__', value)

    def __iadd__(self, value):
        return self._funnel_mathops('__iadd__', value)

    def __radd__(self, value):
        return self._funnel_mathops('__add__', value)

    # ------------------------------------------------------------
    def __sub__(self, value):
        '''
        Subtraction rules:
        ------------------

        Date - Date = DateSpan
        Date - DateSpan = Date
        Date - DateTimeNano = TimeSpan
        Date - TimeSpan = DateTimeNano

        All other operands will be treated as DateSpan and return Date.
        '''
        if isinstance(value, Date):
            func = TypeRegister.MathLedger._BASICMATH_TWO_INPUTS
            # need routine for int32 - int32 => int32 (operands have 0 as invalid, result has sentinel as invalid)
            # right now, using the double return, gets recasted in the constructor
            op = MATH_OPERATION.SUBDATETIMES

            functup = (self, value)
            result = func(functup, op, 0)
            return DateSpan(result)

        elif isinstance(value, DateTimeNano):
            caller = DateTimeNano(self._fa * NANOS_PER_DAY, from_tz=value._timezone._from_tz)
            return caller - value

        else:
            return self._funnel_mathops('__sub__', value)

    def __isub__(self, value):
        return self._funnel_mathops('__isub__', value)

    def __rsub__(self, value):
        if isinstance(value, (Date, DateTimeNano)):
            return value.__sub__(self)
        else:
            raise NotImplementedError

    def __mul__(self, other): raise NotImplementedError

    def __matmul__(self, other): raise NotImplementedError

    # need to check properties to see if division is happening
    #def __truediv__(self, other): raise NotImplementedError
    #def __floordiv__(self, other): raise NotImplementedError
    #def __mod__(self, other): raise NotImplementedError
    #def __divmod__(self, other): raise NotImplementedError
    
    def __pow__(self, other, modulo=None): raise NotImplementedError

    def __lshift__(self, other): raise NotImplementedError

    def __rshift__(self, other): raise NotImplementedError

    def __and__(self, other): raise NotImplementedError

    def __xor__(self, other): raise NotImplementedError

    def __or__(self, other): raise NotImplementedError

    def __rmul__(self, other): raise NotImplementedError

    def __rmatmul__(self, other): raise NotImplementedError

    def __rtruediv__(self, other): raise NotImplementedError

    def __rfloordiv__(self, other): raise NotImplementedError

    def __rmod__(self, other): raise NotImplementedError

    def __rdivmod__(self, other): raise NotImplementedError

    def __rpow__(self, other): raise NotImplementedError

    def __rlshift__(self, other): raise NotImplementedError

    def __rrshift__(self, other): raise NotImplementedError

    def __rand__(self, other): raise NotImplementedError

    def __rxor__(self, other): raise NotImplementedError

    def __ror__(self, other): raise NotImplementedError

    def __imul__(self, other): raise NotImplementedError

    def __imatmul__(self, other): raise NotImplementedError

    def __itruediv__(self, other): raise NotImplementedError

    def __ifloordiv__(self, other): raise NotImplementedError

    def __imod__(self, other): raise NotImplementedError

    def __ipow__(self, other, modulo=None): raise NotImplementedError

    def __ilshift__(self, other): raise NotImplementedError

    def __irshift__(self, other): raise NotImplementedError

    def __iand__(self, other): raise NotImplementedError

    def __ixor__(self, other): raise NotImplementedError

    def __ior__(self, other): raise NotImplementedError

    def __neg__(self): raise NotImplementedError

    def __pos__(self): raise NotImplementedError

    def __abs__(self): raise NotImplementedError

    def __invert__(self): raise NotImplementedError

    def __complex__(self): raise NotImplementedError

    def __int__(self): raise NotImplementedError

    def __float__(self): raise NotImplementedError

    def __round__(self, ndigits=0): raise NotImplementedError

    def __trunc__(self): raise NotImplementedError

    def __floor__(self): raise NotImplementedError

    def __ceil__(self): raise NotImplementedError

    # ------------------------------------------------------------
    def _check_mathops(self, funcname, value):
        '''
        This gets called after a math operation has been performed on the Date's FastArray.
        Return type may differ based on operation. Preserves invalids from original input.

        Parameters:
        -----------
        funcname       : name of ufunc
        value          : original operand in math operation

        returns return_type, other_inv_mask
        '''

        # for now, make Date the default return type
        return_type = Date
        other_inv_mask = None

        if isinstance(value, Date):
            if funcname in ('__add__', '__iadd__', '__isub__'):
                raise TypeError(f'Cannot {funcname} operation between Date and Date')

            return_type = DateSpan
            other_inv_mask = value.isnan()

        # invalid gets early exit
        elif isinstance(value, (int, float, np.number)):
            # return same length Date full of NAN_DATE
            if isnan(value):
                # other_inv_mask will hold the final return
                return_type = None
                other_inv_mask = Date(self.copy_invalid())

        elif isinstance(value, np.ndarray):
            other_inv_mask = isnan(value)

        return return_type, other_inv_mask

    # ------------------------------------------------------------
    def _check_mathops_nano(self, funcname, value, other_inv_mask, return_type, caller):
        '''
        Operations with TimeSpan and DateTimeNano will flip to nano precision, or raise an error.

        Parameters:
        -----------
        funcname       : name of ufunc
        value          : original operand in math operation
        other_inv_mask : None, might be set in this routine
        return_type    : None, might be set to TimeSpan or DateTimeNano
        caller         : FastArray view of Date object.

        '''
        if isinstance(value, TimeSpan):
            return_type = DateTimeNano
            other_inv_mask = value.isnan()
            caller = self._fa * NANOS_PER_DAY
            value = value._fa.astype(np.int64)

        elif isinstance(value, DateTimeNano):
            if funcname in ('__add__', '__iadd__', '__isub__'):
                raise TypeError(f'Cannot perform addition between Date and DateTimeNano')

            return_type = TimeSpan
            other_inv_mask = value.isnan()
            caller = self._fa * NANOS_PER_DAY
            value = value._fa

        return value, other_inv_mask, return_type, caller

    # ------------------------------------------------------------
    @classmethod
    def _load_from_sds_meta_data(cls, name, arr, cols, meta):
        '''
        Restore Date class after loading from .sds file.
        '''

        # **** remove after implementing new metadata routine

        if not isinstance(meta, MetaData):
            meta = MetaData(meta)
        arr = cls(arr)

        # combine loaded meta variables with class defaults
        vars = meta['instance_vars']
        for k, v in cls.MetaDefault.items():
            vars.setdefault(k, v)
        for k, v in vars.items():
            setattr(arr, k, v)

        return arr

    ## ------------------------------------------------------------
    @property
    def start_of_month(self):
        """

        Returns
        -------
        rt.Date array of first of self's month
        """
        return self - self.day_of_month + 1

    @property
    def start_of_week(self):
        """

        Returns
        -------
        rt.Date array of previous Monday
        """
        return self - self.day_of_week


# ========================================================
class DateSpan(DateBase):
    '''
    DateSpan arrays have an underlying int32 array. The array values are in number of days.
    These are created as the result of certain math operations on Date objects.

    Parameters:
    -----------
    arr  : numeric array, list, or scalar
    unit : can set units to 'd' (day) or 'w' (week)

    '''
    # for .SDS file format
    MetaVersion = 1
    MetaDefault = {
        # vars for container loader
        'name': 'Date',
        'typeid': TypeId.DateSpan,
        'version': 0,  # if no version, assume before versions implemented

        'instance_vars': {
            '_display_length': DisplayLength.Long
        }
    }
    NAN_DATE = INVALID_DICT[7]  # int32 sentinel
    forbidden_mathops = ()

    def __new__(cls, arr, unit=None):
        instance = None
        if isinstance(arr, list) or np.isscalar(arr):
            arr = FastArray(arr, dtype=np.int32)

        if isinstance(arr, np.ndarray):
            if arr.dtype.char in NumpyCharTypes.AllInteger + NumpyCharTypes.AllFloat:
                # is this unit really necessary?
                if unit in ('W', 'w'):
                    arr = arr * 7
                arr = arr.astype(np.int32, copy=False)
            else:
                raise TypeError(f'Could not initialize Date object with array of type {arr.dtype}.')

        else:
            raise TypeError(f'DateSpan objects must be initialized with numeric arrays, lists or scalars. Got {type(arr)}')

        instance = arr.view(cls)
        instance._display_length = DisplayLength.Long
        return instance

    # ------------------------------------------------------------
    def __init__(self, arr, unit=None):
        pass

    # ------------------------------------------------------------
    def get_classname(self):
        return __class__.__name__

    # ------------------------------------------------------------
    def get_scalar(self, scalarval):
        return DateSpanScalar(scalarval, _from=self)

    # ------------------------------------------------------------
    @staticmethod
    def display_convert_func(date_num, itemformat: ItemFormat):
        '''
        Called by main rt_display() routine to format items in array correctly in Dataset display.
        Also called by DateSpan's __str__() and __repr__().
        '''
        return DateSpan.format_date_span(date_num, itemformat)

    # ------------------------------------------------------------
    @staticmethod
    def format_date_span(date_span, itemformat):
        '''
        Turn a single value in the DateSpan array into a string for display.
        '''
        if date_span == DateSpan.NAN_DATE:
            return 'Inv'
        if itemformat.length == DisplayLength.Short:
            unit_str = 'd'
        else:
            if date_span == 1:
                unit_str = ' day'
            else:
                unit_str = ' days'

        # remove extra scalar wrapper
        if isinstance(date_span, np.int32):
            date_span = np.int32(date_span)

        return str(date_span) + unit_str

    # ------------------------------------------------------------
    @property
    def format_short(self): self._display_length = DisplayLength.Short

    @property
    def format_long(self): self._display_length = DisplayLength.Long

    # ------------------------------------------------------------
    @classmethod
    def _load_from_sds_meta_data(cls, name, arr, cols, meta):
        '''
        Restore Date class after loading from .sds file.
        '''

        if not isinstance(meta, MetaData):
            meta = MetaData(meta)
        arr = cls(arr)

        # combine loaded meta variables with class defaults
        vars = meta['instance_vars']
        for k, v in cls.MetaDefault.items():
            vars.setdefault(k, v)
        for k, v in vars.items():
            setattr(arr, k, v)

        return arr

    # ------------------------------------------------------------
    def fill_invalid(self, shape=None, dtype=None, inplace=True):
        arr = self._fill_invalid_internal(shape=shape, dtype=self.dtype, inplace=inplace)
        if arr is None:
            return
        return DateSpan(arr)

    # ------------------------------------------------------------
    @classmethod
    def hstack(cls, dates):
        '''
        hstacks DateSpan objects and returns a new DateSpan object.
        Will be called by riptable.hstack() if the first item in the sequence is a DateSpan object.

        Parameters:
        -----------
        dates : list or tuple of DateSpan objects


        >>> d1 = Date('2015-02-01')
        >>> d2 = Date(['2016-02-01', '2017-02-01', '2018-02-01'])
        >>> hstack([d1, d2])
        Date([2015-02-01, 2016-02-01, 2017-02-01, 2018-02-01])

        '''
        # pass the subclass to the parent class routine
        return hstack_any(dates, cls, DateSpan)

    # ------------------------------------------------------------
    def _check_mathops_nano(self, funcname, value, other_inv_mask, return_type, caller):
        '''
        Operations with TimeSpan and DateTimeNano will flip to nano precision, or raise an error.

        Parameters:
        -----------
        funcname       : name of ufunc
        value          : original operand in math operation
        other_inv_mask : None, might be set in this routine
        return_type    : None, might be set to TimeSpan or DateTimeNano
        caller         : FastArray view of Date object.

        '''
        if isinstance(value, TimeSpan):
            return_type = TimeSpan
            other_inv_mask = value.isnan()
            caller = self._fa * NANOS_PER_DAY

        elif isinstance(value, DateTimeNano):
            if funcname in ('__sub__', '__isub__'):
                raise TypeError(f'Cannot perform {funcname} operation between DateSpan and DateTimeNano')
            return_type = DateTimeNano
            other_inv_mask = value.isnan()
            caller = self._fa * NANOS_PER_DAY
            value = value._fa

        return value, other_inv_mask, return_type, caller

    # ------------------------------------------------------------
    def _check_mathops(self, funcname, value):
        '''
        This gets called after a math operation has been performed on the Date's FastArray.
        Return type may differ based on operation. Preserves invalids from original input.

        Parameters:
        -----------
        funcname       : name of ufunc
        value          : original operand in math operation

        returns return_type, other_inv_mask
        '''

        # for now, make Date the default return type
        return_type = DateSpan
        other_inv_mask = None

        if isinstance(value, Date):
            if funcname in ('__sub__', '__isub__'):
                raise TypeError(f'Cannot perform {funcname} operation between DateSpan and Date')
            return_type = Date
            other_inv_mask = value.isnan()

        # invalid gets early exit
        elif isinstance(value, (int, float, np.number)):
            # return same length Date full of NAN_DATE
            if isnan(value) or value == self.NAN_DATE:
                # other_inv_mask will hold the final return
                return_type = None
                other_inv_mask = DateSpan(self.copy_invalid())

        elif isinstance(value, np.ndarray):
            other_inv_mask = isnan(value)

        return return_type, other_inv_mask

    # ------------------------------------------------------------
    def __add__(self, value):
        return self._funnel_mathops('__add__', value)

    def __iadd__(self, value):
        return self._funnel_mathops('__iadd__', value)

    def __sub__(self, value):
        return self._funnel_mathops('__sub__', value)

    def __isub__(self, value):
        return self._funnel_mathops('__isub__', value)

    # ------------------------------------------------------------
    def _datespan_compare_check(self, funcname, other):
        '''
        Funnel for all comparison operations.
        Helps Date interact with DateTimeNano, TimeSpan.
        '''

        caller = self._fa

        if isinstance(other, (Date, DateTimeNano, TypeRegister.Categorical)):
            # Date allows categorical comparisons, DateSpan does not
            raise TypeError(f'Cannot perform {funcname} comparison operation between {type(self)} and {type(other)}.')

        elif isinstance(other, TimeSpan):
            caller = self._fa * NANOS_PER_DAY

        # looks weird now, saving explicit branchesfor if any forbidden types appear
        elif isinstance(other, DateSpan):
            other = other._fa

        # Categorical will fall through to constructor too
        elif isinstance(other, np.ndarray):
            other = Date(other)

        # let everything else fall through for FastArray to catch
        func = getattr(caller, funcname)
        return func(other)

    # -------------------COMPARISONS------------------------------
    # ------------------------------------------------------------
    def __ne__(self, other):
        return self._datespan_compare_check('__ne__', other)

    def __eq__(self, other):
        return self._datespan_compare_check('__eq__', other)

    def __ge__(self, other):
        return self._datespan_compare_check('__ge__', other)

    def __gt__(self, other):
        return self._datespan_compare_check('__gt__', other)

    def __le__(self, other):
        return self._datespan_compare_check('__le__', other)

    def __lt__(self, other):
        return self._datespan_compare_check('__lt__', other)


# ------------------------------------------------------------
def DateTimeUTC(arr, to_tz='NYC', from_matlab=False, format=None, start_date=None, gmt=None):
    """Forces DateTimeNano ``from_tz`` keyword to 'UTC'.
    For more see DateTimeNano.
    """
    return DateTimeNano(arr, from_tz='UTC', to_tz=to_tz, from_matlab=from_matlab, format=format, start_date=start_date, gmt=gmt)


# ========================================================
class DateTimeCommon:
    '''
    Common functions shared between the array based class and the scalar
    This class must be combine with another class because of dependency on _timezone
    '''

    # -CLOCK HH:MM------------------------------------------------
    @property
    def format_clock(self):
        '''Set time to be displayed as HH:MM:SS'''
        self._display_length = DisplayLength.Short

    @property
    def format_short(self):
        '''Set time to be displayed as HH:MM:SS'''
        self._display_length = DisplayLength.Short

    # -YYYYMMDD----------------------------------------------------
    @property
    def format_medium(self):
        '''Set time to be displayed as YYYYMMDD'''
        self._display_length = DisplayLength.Medium

    @property
    def format_ymd(self):
        '''Set time to be displayed as YYYYMMDD'''
        self._display_length = DisplayLength.Medium

    @property
    def format_day(self):
        '''Set time to be displayed as YYYYMMDD'''
        self._display_length = DisplayLength.Medium

    # -YYYYMMDD HH:MM:SS.nanosecond ---------------------------------
    @property
    def format_long(self):
        '''Set time to be displayed as YYYYMMDD HH:MM:SS.fffffffff'''
        self._display_length = DisplayLength.Long

    @property
    def format_full(self):
        '''Set time to be displayed as YYYYMMDD HH:MM:SS.fffffffff'''
        self._display_length = DisplayLength.Long

    @property
    def format_sig(self):
        '''Set time to be displayed as YYYYMMDD HH:MM:SS.fffffffff'''
        self._display_length = DisplayLength.Long

    # ------------------------------------------------------------
    @property
    def days_since_epoch(self):
        '''
        Number of days since epoch.

        Examples
        --------
        >>> dtn = DateTimeNano(['1970-01-11'], from_tz='NYC')
        >>> dtn.days_since_epoch
        FastArray([10], dtype=int64)

        Returns
        -------
        int64 array

        '''
        arr = self._timezone.fix_dst(self)
        return arr // NANOS_PER_DAY

    # ------------------------------------------------------------
    @property
    def seconds_since_epoch(self):
        '''
        Number of seconds since epoch.

        Examples
        --------
        >>> dtn = DateTimeNano(['1970-01-02'], from_tz='NYC')
        >>> dtn.seconds_since_epoch
        FastArray([86400], dtype=int64)

        Returns
        -------
        int64 array
        '''
        arr = self._timezone.fix_dst(self)
        return arr // NANOS_PER_SECOND

    # ------------------------------------------------------------
    def nanos_since_midnight(self):
        '''
        Nanosecond since midnight of the current day.

        Examples
        --------
        >>> dtn = DateTimeNano(['2018-01-01 00:00:00.000123456'],from_tz='NYC')
        >>> dtn.nanos_since_midnight()
        FastArray([123456], dtype=int64)

        Returns
        -------
        int64 array

        See Also
        --------
        DateTimeNano.time_since_midnight

        '''
        arr = self._timezone.fix_dst(self)
        arr = arr % NANOS_PER_DAY
        return _apply_inv_mask(self, arr)

    # ------------------------------------------------------------
    def millis_since_midnight(self):
        '''
        Milliseconds since midnight of the current day.

        Examples
        --------
        >>> dtn = DateTimeNano(['1992-02-01 00:00:01.002003004'], from_tz='NYC')
        >>> dtn.millis_since_midnight()
        FastArray([1002.003004])

        Returns
        -------
        float64 array

        Note
        ----
        Unlike similar methods, this returns floating point, similar to common columns
        in Matlab datasets.

        '''
        arr = self._timezone.fix_dst(self)
        arr = arr % NANOS_PER_DAY
        arr = arr / NANOS_PER_MILLISECOND
        return _apply_inv_mask(self, arr)

    # ------------------------------------------------------------
    def date(self):
        '''
        Copies the object and removes hours, minutes, seconds, and second fractions.
        All resulting times will be at midnight.

        Examples
        --------
        >>> dtn = DateTimeNano(['2019-01-04 12:34', '2019-06-06 14:00'], from_tz='NYC')
        >>> dtn.date()
        DateTimeNano([20190104 00:00:00.000000000, 20190606 00:00:00.000000000])

        Returns
        -------
        obj:`DateTimeNano`


        '''
        if self._timezone._dst_reverse is not None:
            arr = self._timezone.fix_dst(self._fa)
            arr = arr - (arr % NANOS_PER_DAY)
        else:
            arr = self._fa
            arr = arr - (arr % NANOS_PER_DAY)
        # from_tz needs to match to_tz (similar to from_matlab_days, except can't force 'GMT' because of DST fixup)
        # return DateTimeNano(arr, from_tz=self._timezone._to_tz, to_tz='UTC')
        result = DateTimeNano(arr, from_tz=self._timezone._to_tz, to_tz=self._timezone._to_tz)
        if isinstance(self, DateTimeNanoScalar):
            return result[0]
        return result

    # ------------------------------------------------------------
    @property
    def yyyymmdd(self):
        '''
        Returns integers in the format YYYYMMDD.
        Accounts for daylight savings time, leap years.

        Examples
        --------
        >>> dtn = DateTimeNano(['2018-01-09', '2000-02-29', '2000-03-01', '2019-12-31'], from_tz='NYC')
        >>> dtn.yyyymmdd
        FastArray([20180109, 20000229, 20000301, 20191231])

        Returns
        -------
        int32 array

        Note
        ----
        this routine is very similar to day_of_month - can probably internal routines to avoid repeating code

        '''
        arr = self._fa
        arr = self._timezone.fix_dst(arr)
        year = self._year(arr, fix_dst=False)
        # initialize result
        final = year * 10_000

        # subtract the nanos from start of year so all times are in MM-DD HH:MM:SS, etc.
        startyear = arr - self._year_splits[year - 1970]

        # treat the whole array like a non-leapyear
        monthnum = self._yearday_splits.searchsorted(startyear, side='right')
        startmonth_idx = monthnum - 1
        monthtime = startyear - self._yearday_splits[startmonth_idx]
        # fix up the leapyears with a different yearday split table
        leapmask = (year % 4) == 0
        monthnum_leap = self._yearday_splits_leap.searchsorted(startyear[leapmask], side='right')
        startmonth_idx_leap = monthnum_leap - 1
        monthnum[leapmask] = monthnum_leap
        monthtime[leapmask] = startyear[leapmask] - self._yearday_splits_leap[startmonth_idx_leap]

        # future optimization, takeover place, or send __setitem__ indexer to our version of it
        # np.place(monthnum, leapmask, monthnum_leap)
        # np.place(monthtime, leapmask, startyear[leapmask] - UTC_YDAY_SPLITS_LEAP[startmonth_idx_leap])

        # add month and day values to final
        final += monthnum.astype(np.int32) * 100
        final += (monthtime // NANOS_PER_DAY) + 1

        return final

    # ------------------------------------------------------------
    @property
    def _year_splits(self):
        """Midnght on Jan. 1st from 1970 - 1940 in utc nanoseconds."""
        return UTC_1970_SPLITS

    # ------------------------------------------------------------
    @property
    def _yearday_splits(self):
        """Midnight on the 1st of the month in nanoseconds since the beginning of the year."""
        return UTC_YDAY_SPLITS

    # ------------------------------------------------------------
    @property
    def _yearday_splits_leap(self):
        """Midnight on the 1st of the month in nanoseconds since the beginning of the year during a leap year."""
        return UTC_YDAY_SPLITS_LEAP

    # ------------------------------------------------------------
    def year(self):
        '''
        The year value for each entry in the array

        Examples
        ---------
        >>> dtn = DateTimeNano(['1984-02-01', '1992-02-01', '2018-02-01'], from_tz='NYC')
        >>> dtn.year()
        FastArray([1984, 1992, 2018])

        Returns
        -------
        int32 array

        '''
        year = self._year(self._fa, fix_dst=True)
        return _apply_inv_mask(self, year)

    # ------------------------------------------------------------
    def month(self):
        '''
        The month value for each entry in the array.
        1=Jan, 2=Feb, etc. ( is leap-year aware )

        Examples
        --------
        >>> dtn = DateTimeNano(['2000-02-29', '2018-12-25'], from_tz='NYC')
        >>> dtn.month()
        FastArray([ 2, 12])

        Returns
        -------
        int32 array

        '''
        return _apply_inv_mask(self, self._month(fix_dst=True))

    # ------------------------------------------------------------
    def monthyear(self, arr=None):
        '''
        Returns a string with 3 letter month + 4 digit year

        Examples
        --------
        >>> d = DateTimeNano(['2000-02-29', '2018-12-25'], from_tz='NYC')
        >>> d.monthyear()
        FastArray([ 'Feb2000','Dec2018'])
        '''
        month = self.month()
        yearstr = self.year().astype('S')
        return MONTH_STR_ARRAY[month - 1] + yearstr

    # ------------------------------------------------------------
    @property
    def day_of_year(self):
        '''
        The day of year value for each entry in the array.
        Day values are from 1 to 365 (or 366 if leap year)

        Examples
        --------
        >>> dtn = DateTimeNano(['2019-01-01', '2019-02-01', '2019-12-31 23:59', '2000-12-31 23:59'], from_tz='NYC')
        FastArray([  1,  32, 365, 366], dtype=int64)

        Returns
        -------
        int32 array

        '''
        result = self.nanos_since_start_of_year()
        if isinstance(result, np.ndarray):
            np.floor_divide(result, NANOS_PER_DAY, out=result)
        else:
            result = result // NANOS_PER_DAY
        result += 1

        return result

    # ------------------------------------------------------------
    @property
    def day_of_month(self):
        '''
        The day of month value for each entry in the array.
        Day values are from 1 to 31
        Adjusts for daylight savings time, leap year

        Examples
        --------
        >>> dtn = DateTimeNano(['2018-01-09', '2000-02-29', '2000-03-01', '2019-12-31'], from_tz='NYC')
        >>> dtn.day_of_month
        FastArray([ 9, 29,  1, 31], dtype=int64)

        Returns
        -------
        int32 array

        '''
        arr = self._fa
        year = self._year(arr, fix_dst=True)

        # subtract the nanos from start of year so all times are in MM-DD HH:MM:SS, etc.
        startyear = arr - self._year_splits[year - 1970]

        # treat the whole array like a non-leapyear
        startmonth_idx = self._yearday_splits.searchsorted(startyear, side='right') - 1
        monthtime = startyear - self._yearday_splits[startmonth_idx]

        # fix up the leapyears with a different yearday split table
        leapmask = (year % 4) == 0
        startmonth_idx_leap = self._yearday_splits_leap.searchsorted(startyear[leapmask], side='right') - 1
        monthtime[leapmask] = startyear[leapmask] - self._yearday_splits_leap[startmonth_idx_leap]

        # unlike month, weekday, hour, etc. day of month starts at 1
        if isinstance(monthtime, np.ndarray):
            np.floor_divide(monthtime, NANOS_PER_DAY, out=monthtime)
        else:
            monthtime = monthtime // NANOS_PER_DAY
        monthtime += 1

        return monthtime

    # ------------------------------------------------------------
    @property
    def day_of_week(self):
        '''
        Day of week value for each entry in the array.
        Monday (0) -> Sunday (6)

        January 1st 1970 was a Thursday! (3)

        Examples
        --------
        >>> dtn = DateTimeNano(['1992-02-01 19:48:00', '1995-05-12 05:12:00'], from_tz='NYC')
        >>> dtn.day_of_week
        FastArray([5, 4])

        Returns
        -------
        int32 array

        '''
        arr = self.days_since_epoch
        arr += EPOCH_DAY_OF_WEEK

        if isinstance(arr, np.ndarray):
            # inplace operation
            np.mod(arr, 7, out=arr)
        else:
            arr = arr % 7
        return arr

    # ------------------------------------------------------------
    @property
    def start_of_week(self):
        '''
        Return the Monday for the week the TimeStamp is in
        Returns a Date or DateScalar
        '''
        arr = self.days_since_epoch
        arr += EPOCH_DAY_OF_WEEK
        adjust = arr % 7
        arr -= adjust
        arr -= EPOCH_DAY_OF_WEEK

        result = Date(arr)
        if not isinstance(arr, np.ndarray):
            return result[0]
        return result

    # ------------------------------------------------------------
    @property
    def is_dst(self):
        '''
        Boolean array, True if a time value was in daylight savings time for the displayed timezone.
        If the timezone is GMT, returns False for all items, including invalid times.

        Examples
        --------
        >>> dtn = DateTimeNano(['2018-11-03 12:34', '2018-11-04 12:34'], from_tz='NYC')
        >>> dtn.is_dst
        FastArray([ True, False])

        >>> dtn = DateTimeNano(['2019-03-30 12:34', '2019-03-31 12:34'], from_tz='DUBLIN')
        >>> dtn.is_dst
        FastArray([False,  True])

        >>> dtn = DateTimeNano(['2019-03-30 12:34', '2019-03-31 12:34'], from_tz='GMT', to_tz='GMT')
        >>> dtn.is_dst
        FastArray([False, False])

        Returns
        -------
        bool array

        '''
        return self._timezone._is_dst(self._fa)

    # ------------------------------------------------------------
    @property
    def tz_offset(self):
        '''
        Array of hour offset from GMT. Accounts for daylight savings time in timezone set by to_tz.
        If the timezone is GMT, returns all 0.

        Examples
        --------
        dtn = DateTimeNano(['2018-11-03 12:34', '2018-11-04 12:34'], from_tz='NYC')
        >>> dtn.tz_offset
        FastArray([-4, -5])

        >>> dtn = DateTimeNano(['2019-03-30 12:34', '2019-03-31 12:34'], from_tz='DUBLIN', from_tz='DUBLIN')
        >>> dtn.tz_offset
        FastArray([0, 1])

        >>> dtn = DateTimeNano(['2019-03-30 12:34', '2019-03-31 12:34'], from_tz='GMT', to_tz='GMT')
        >>> dtn.tz_offset
        FastArray([0, 0])

        Returns
        -------
        int32 array
        '''
        return self._timezone._tz_offset(self._fa)

    # -----------------------------------------------------
    def putmask(self, arr1, filter, arr2):
        '''
        scalar or array putmask
        '''
        if isinstance(arr1, np.ndarray):
            return putmask(arr1, filter, arr2)
        else:
            if filter:
                return arr2
            else:
                return arr1

    # ------------------------------------------------------------
    @property
    def is_weekday(self):
        '''
        Returns boolean array of wether or not each time occured on a weekday.

        Examples
        --------
        (Monday, Thursday, Saturday)
        >>> dtn = DateTimeNano(['2019-01-07', '2019-01-10', '2019-01-12'],from_tz='NYC')
        >>> dtn.is_weekday
        FastArray([ True,  True, False])

        Returns
        -------
        bool array

        '''
        inv_mask = self.isnan()
        isweekday = self.day_of_week < 5
        self.putmask(isweekday, inv_mask, False)
        return isweekday

    # ------------------------------------------------------------
    @property
    def is_weekend(self):
        '''
        Returns boolean array of wether or not each time occured on a weekend.

        Examples
        --------
        (Monday, Thursday, Saturday)
        >>> dtn = DateTimeNano(['2019-01-07', '2019-01-10', '2019-01-12'],from_tz='NYC')
        >>> dtn.is_weekend
        FastArray([False, False,  True])

        Returns
        -------
        bool array

        '''
        inv_mask = self.isnan()
        isweekend = self.day_of_week > 4
        self.putmask(isweekend, inv_mask, False)
        return isweekend

    # ------------------------------------------------------------
    @property
    def day(self):
        '''
        Fractional day time relative to 24 hours.

        Examples
        --------
        >>> dtn = DateTimeNano(['2000-02-01 19:48:00.000000'], from_tz='NYC')
        >>> dtn.day
        FastArray([0.825])

        Returns
        -------
        float64 array

        Notes
        -----
        this is different than properties for hour, minute, and second as the
        relative unit is its own unit.
        '''
        inv_mask = self.isnan()
        arr = self._timezone.fix_dst(self._fa)
        arr = arr % NANOS_PER_DAY
        arr = arr / NANOS_PER_DAY
        self.putmask(arr, inv_mask, np.nan)
        return arr

    # ------------------------------------------------------------
    @property
    def hour(self):
        '''
        Hours relative to the current day (with partial hour decimal).

        Examples
        --------
        >>> dtn = DateTimeNano(['2000-02-01 19:48:00.000000'], from_tz='NYC')
        >>> dtn.hour
        >>> FastArray([19.8])

        Returns
        -------
        float64 array

        See Also
        --------
        DateTimeNano.hour_span
        '''
        return self._hour()

    # -----------------------------------------------------
    @property
    def hour_span(self):
        '''
         Hours relative to the current day (with partial hour decimal) as a TimeSpan object.

        Examples
        --------
        >>> dtn = DateTimeNano(['2000-02-01 19:48:00.000000'], from_tz='NYC')
        >>> dtn.hour_span
        TimeSpan([19:48:00.000000000])

        Returns
        -------
        obj:`TimeSpan`

        See Also
        --------
        DateTimeNano.hour
        '''
        return self._hour(span=True)

    def _hour(self, span=False):
        inv_mask = self.isnan()
        arr = self._timezone.fix_dst(self._fa)
        arr = arr % NANOS_PER_DAY
        if span:
            result = TypeRegister.TimeSpan(arr)
        else:
            result = arr / NANOS_PER_HOUR
        self.putmask(result, inv_mask, np.nan)
        return result

    # ------------------------------------------------------------
    def _time_fraction(self, modulo, divisor, span=False):
        '''
        Internal routine for minute, second, millisecond, microsecond, nanosecond (+span) properties.
        None of these need to account for timezone.
        '''
        inv_mask = self.isnan()
        arr = self._fa % modulo
        if span:
            if isinstance(self, DateTimeNano):
                result = TypeRegister.TimeSpan(arr)
            else:
                result = TypeRegister.TimeSpanScalar(arr)
        else:
            result = arr / divisor
        self.putmask(result, inv_mask, np.nan)
        return result

    # ------------------------------------------------------------
    @property
    def minute(self):
        '''
        Minutes relative to the current hour (with partial minute decimal).

        Examples
        --------
        >>> dtn = DateTimeNano(['2000-02-01 19:48:30.000000'], from_tz='NYC')
        >>> dtn.minute
        >>> FastArray([48.5])

        Returns
        -------
        float64 array

        See Also
        --------
        DateTimeNano.minute_span
        '''
        return self._time_fraction(NANOS_PER_HOUR, NANOS_PER_MINUTE)

    @property
    def minute_span(self):
        '''
        Minutes relative to the current hour (with partial minute decimal) as a TimeSpan object

        Examples
        --------
        >>> dtn = DateTimeNano(['2000-02-01 19:48:30.000000'], from_tz='NYC')
        >>> dtn.minute_span
        >>> TimeSpan([00:48:30.000000000])

        Returns
        -------
        obj:`TimeSpan`

        See Also
        --------
        DateTimeNano.minute
        '''
        return self._time_fraction(NANOS_PER_HOUR, NANOS_PER_MINUTE, span=True)

    # ------------------------------------------------------------
    @property
    def second(self):
        '''
        Seconds relative to the current minute (with partial second decimal).

        Examples
        --------
        >>> dtn = DateTimeNano(['2000-02-01 19:48:30.100000'], from_tz='NYC')
        >>> dtn.seconds
        >>> FastArray([30.1])

        Returns
        -------
        float64 array

        See Also
        --------
        DateTimeNano.second_span
        '''
        return self._time_fraction(NANOS_PER_MINUTE, NANOS_PER_SECOND)

    @property
    def second_span(self):
        '''
        Seconds relative to the current minute (with partial second decimal) as a TimeSpan object.

        Examples
        --------
        >>> dtn = DateTimeNano(['2000-02-01 19:48:30.100000'], from_tz='NYC')
        >>> dtn.second_span
        TimeSpan([00:00:30.100000000])
        '''
        return self._time_fraction(NANOS_PER_MINUTE, NANOS_PER_SECOND, span=True)

    # ------------------------------------------------------------
    @property
    def millisecond(self):
        '''
        Milliseconds relative to the current second (with partial millisecond decimal).

        Examples
        --------
        >>> dtn = DateTimeNano(['1992-02-01 12:00:01.123000000'], from_tz='NYC')
        >>> dtn.millisecond
        FastArray([123.])

        Returns
        -------
        float64 array

        See Also
        --------
        DateTimeNano.millisecond_span
        '''
        return self._time_fraction(NANOS_PER_SECOND, NANOS_PER_MILLISECOND)

    @property
    def millisecond_span(self):
        '''
        Milliseconds relative to the current second (with partial millisecond decimal) as a TimeSpan object.

        Examples
        --------
        >>> dtn = DateTimeNano(['1992-02-01 12:00:01.123000000'], from_tz='NYC')
        >>> dtn.millisecond_span
        TimeSpan([00:00:00.123000000])

        Returns
        -------
        obj:`TimeSpan`

        See Also
        --------
        DateTimeNano.millisecond

        '''
        return self._time_fraction(NANOS_PER_SECOND, NANOS_PER_MILLISECOND, span=True)

    # ------------------------------------------------------------
    @property
    def microsecond(self):
        '''
        Microseconds relative to the current millisecond (with partial microsecond decimal)

        Examples
        --------
        >>> dtn = DateTimeNano(['1992-02-01 12:00:01.000123000'], from_tz='NYC')
        >>> dtn.microsecond
        FastArray([123.])

        Returns
        -------
        float64 array

        See Also
        --------
        DateTimeNano.microsecond_span
        '''
        return self._time_fraction(NANOS_PER_MILLISECOND, NANOS_PER_MICROSECOND)

    @property
    def microsecond_span(self):
        '''
        Microseconds relative to the current millisecond (with partial microsecond decimal) as a TimeSpan object.

        Examples
        --------
        >>> dtn = DateTimeNano(['1992-02-01 12:00:01.000123000'], from_tz='NYC')
        >>> dtn.microsecond_span
        TimeSpan([00:00:00.000123000])

        Returns
        -------
        obj:`TimeSpan`

        See Also
        --------
        DateTimeNano.microsecond
        '''
        return self._time_fraction(NANOS_PER_MILLISECOND, NANOS_PER_MICROSECOND, span=True)

    # ------------------------------------------------------------
    @property
    def nanosecond(self):
        '''
        Nanoseconds relative to the current microsecond.

        Examples
        --------
        >>> dtn = DateTimeNano(['1992-02-01 12:00:01.000000123'], from_tz='NYC')
        >>> dtn.nanosecond
        FastArray([123.])

        Returns
        -------
        float64 array

        See Also
        --------
        DateTimeNano.nanosecond_span

        '''
        return self._time_fraction(NANOS_PER_MICROSECOND, 1)

    @property
    def nanosecond_span(self):
        '''
        Nanoseconds relative to the current microsecond as a TimeSpan object.

        Examples
        --------
        >>> dtn = DateTimeNano(['1992-02-01 12:00:01.000000123'], from_tz='NYC')
        >>> dtn.nanosecond_span
        TimeSpan([00:00:00.000000123])

        Returns
        -------
        obj:`TimeSpan`

        See Also
        --------
        DateTimeNano.nanosecond

        '''
        return self._time_fraction(NANOS_PER_MICROSECOND, 1, span=True)

    # ------------------------------------------------------------
    def nanos_since_start_of_year(self):
        '''
        Nanoseconds since Jan. 1st at midnight of the current year.

        Examples
        --------
        >>> dtn = DateTimeNano(['2018-01-01 00:00:00.000123456'],from_tz='NYC')
        >>> dtn.nanos_since_start_of_year()
        FastArray([123456], dtype=int64)

        Returns
        -------
        int64 array

        See Also
        --------
        DateTimeNano.time_since_start_of_year

        '''
        arr = self._timezone.fix_dst(self._fa)
        year = self._year(arr, fix_dst=False)
        arr = arr - self._year_splits[year - 1970]
        return arr

    # ------------------------------------------------------------
    def time_since_start_of_year(self):
        '''
        Nanoseconds since Jan. 1st at midnight of the current year as a TimeSpan object.

        Examples
        --------
        >>> dtn = DateTimeNano(['2018-01-01 16:00:00.000123456'],from_tz='NYC')
        >>> dtn.time_since_start_of_year()
        TimeSpan([16:00:00.000123456])

        Returns
        -------
        obj:`TimeSpan`

        See Also
        --------
        DateTimeNano.nanos_since_start_of_year

        Note
        ----
        Nanosecond precision will be lost after ~52 days
        '''

        result = TimeSpan(self.nanos_since_start_of_year())

        if isinstance(self, DateTimeNano):
            return result
        return result[0]

    # ------------------------------------------------------------
    def time_since_midnight(self):
        '''
        Elapsed time since midnight as a TimeSpan object.

        Examples
        --------
        >>> dtn = DateTimeNano(['2000-02-29 00:00:00.000000100','2000-02-29 00:00:00.000123456'], from_tz='NYC')
        >>> dtn.nanos_since_midnight()
        TimeSpan([00:00:00.000000100, 00:00:00.000123456])

        Returns
        -------
        obj:`TimeSpan`

        See Also
        --------
        DateTimeNano.nanos_since_midnight, DateTimeNano.millis_since_midnight

        '''
        return self.hour_span

    # ------------------------------------------------------------
    # for DateTimeNano and DateTimeNanoScalar
    def _build_mathops_result(self, other, funcname, call_super, other_inv_mask, inplace, op, return_type):
        '''
        Operates on fastarray or takes invalid fast track for DateTimeNano math operations like add/sub
        '''
        input1 = self
        if not isinstance(self, np.ndarray):
            input1 = DateTimeNano(self)

        func = TypeRegister.MathLedger._BASICMATH_TWO_INPUTS
        if call_super:
            if inplace:
                # inplace operations need to save invalids beforehand
                input1_mask = input1.isnan()
            else:
                input1_mask = None
            # also need to apply invalid from operand
            if other_inv_mask is None:
                other_inv_mask = isnan(other)
            func = getattr(input1._fa, funcname)
            result = func(other)
            result = _apply_inv_mask(input1, result, fillval=DateTimeBase.NAN_TIME, arr1_inv_mask=input1_mask, arr2_inv_mask=other_inv_mask)
        else:
            if inplace:
                functup = (input1, other, input1)
            else:
                functup = (input1, other)

            result = func(functup, op, 0)
            if result is None:
                raise RuntimeError(f'Could not perform {funcname} operation with DateTimeNano and {type(other)} {other}')

        if return_type == DateTimeNano:
            result = DateTimeNano(result, from_tz='GMT', to_tz=input1._timezone._to_tz)
        else:
            result = return_type(result)

        # check if both were scalars, then return a scalar
        if not isinstance(self, np.ndarray) and not isinstance(other, np.ndarray):
            return result[0]
        return result

    def strftime(self, format, dtype='O'):
        '''
        Converts DateTimeNano to an array of object strings or a scalar string.
        This routine has not been sped up yet.

        Other Parameters
        ----------------
        dtype: defaults to 'O', can change to 'S' or 'U'

        Examples
        --------
        >>> rt.utcnow(4).strftime('%c')
        array(['Thu Oct 31 14:55:14 2019', 'Thu Oct 31 14:55:14 2019',
               'Thu Oct 31 14:55:14 2019', 'Thu Oct 31 14:55:14 2019'], dtype='<U24')
        >>> rt.utcnow(4).strftime('%X.%f')
        array(['15:03:04.697686', '15:03:04.697687', '15:03:04.697687',
               '15:03:04.697687'], dtype='<U15')

        See Also
        ---------
        http://strftime.org  for format strings
        datetime.datetime.strftime

        '''
        in_seconds = self / NANOS_PER_SECOND
        to_tz = self._timezone._to_tz
        if to_tz in ['GMT', 'UTC']:
            if isinstance(in_seconds, np.ndarray):
                return np.asarray([dt.utcfromtimestamp(timestamp).strftime(format) for timestamp in in_seconds], dtype=dtype)
            else:
                return dt.strftime(dt.utcfromtimestamp(in_seconds), format)

        else:
            # Choose timezone from to_tz
            localzone = tz.gettz(self._timezone._timezone_str)

            if isinstance(in_seconds, np.ndarray):
                return np.asarray([dt.fromtimestamp(timestamp, localzone).strftime(format) for timestamp in in_seconds], dtype=dtype)
            else:
                return dt.strftime(dt.fromtimestamp(in_seconds, localzone), format)


# ========================================================
class DateTimeNano(DateTimeBase, TimeStampBase, DateTimeCommon):
    '''

    DateTimeNano arrays have an underlying int64 array. The array is in UTC nanosecond time that defaults to display
    correctly in eastern/NYC time, accounting for daylight savings time.

    Parameters
    ----------
    arr         : nanoseconds in integer, timestrings, numpy datetime64 array
    from_tz     : if initialized from strings, user is required to specify the timezone of origin (otherwise default is UTC)
                    Currently supported:
                    'NYC'    : US/Eastern (accounts for daylight savings)
                    'DUBLIN' : Dublin (accounts for daylight savings)
                    'GMT'    : Greenwich Mean Time
                    'UTC'    : (not a timezone, but accepted as an alias for GMT)
    to_tz       : controls how the data will be displayed. if not set, will match the from_tz keyword
    from_matlab : indicates that the input was matlab ordinal days (may also have fraction of day, so specify from_tz)
    format      : specify a format for timestrings - will be ignored unless strings are in arr
                    See strptime_to_nano() docstring for format info
    start_date  : specify a string start date for times in format YYYYMMDD. all values in the provided array
                  will be interpretted as nanoseconds, timespan, or clock strings in HH:MM
    (gmt **deprecated)


    Examples
    --------
    From DateTimeNano timestamps already in GMT:

    >>> dtn = DateTimeNano([1514828730123456000], from_tz='GMT')
    >>> dtn
    DateTimeNano([20180101 12:45:30.123456000])

    From DateTimeStrings in NYC time:

    >>> dtn = DateTimeNano(['2018-01-01 12:45:30.123456000'], from_tz='NYC')
    >>> dtn
    DateTimeNano([20180101 12:45:30.123456000])

    From numpy datetime64 array (different resolution, note numpy drops precision):

    >>> dtn = np.array(['2018-11-02 09:30:00.002201', '2018-11-02 09:30:00.004212', '2018-11-02 09:30:00.005351'], dtype='datetime64[ms]')
    >>> dtn = DateTimeNano(dtn, from_tz='NYC')
    >>> dtn
    DateTimeNano([20181102 09:30:00.002000000, 20181102 09:30:00.004000000, 20181102 09:30:00.005000000])

    String parsing differences:
    ---------------------------
    - riptable DateTimeNano string parsing is more forgiving than Numpy datetime64 arrays
    - In some cases an object is returned when numpy returns an error.
    - In other cases, a different result is returned.
    - The lower limit for DateTimeNano string parsing is unix epoch time.
    - You can always guarantee the same results by using the full ISO-8601 datetime format (YYYY-MM-DDTHH:mm:ss.fffffffff)

    Without zero padding:
    >>> DateTimeNano(['2018-1-1 1'], from_tz='NYC')
    DateTimeNano([20180101 01:00:00.000000000])
    >>> np.array(['2018-1-1 1'], dtype='datetime64[ns]')
    ValueError: Error parsing datetime string "2018-1-1 1" at position 5

    Extra characters:
    >>> DateTimeNano(['2018-10-11 10:11:00.123           '], from_tz='NYC')
    DateTimeNano([20181011 10:11:00.123000000])
    >>> np.array(['2018-10-11 10:11:00.123           '], dtype='datetime64[ns]')
    DeprecationWarning: parsing timezone aware datetimes is deprecated; this will raise an error in the future
    array(['2018-10-11T10:11:00.123000000'], dtype='datetime64[ns]')

    Without separators:
    >>> DateTimeNano(['20181231'], from_tz='NYC')
    DateTimeNano([20181231 00:00:00.000000000])
    >>> np.array(['20181231'], dtype='datetime64[ns]')
    array(['1840-08-31T19:51:12.568664064'], dtype='datetime64[ns]')

    ISO-8601 format:
    >>> DateTimeNano(['2018-12-31T12:34:56.789123456'],from_tz='NYC')
    DateTimeNano([20181231 12:34:56.789123456])
    >>> np.array(['2018-12-31T12:34:56.789123456'], dtype='datetime64[ns]')
    array(['2018-12-31T12:34:56.789123456'], dtype='datetime64[ns]')

    strptime like formatting
    >>> a=DateTimeNano(['12/31/19', '6/30/19'], format='%m/%d/%y', from_tz='NYC')
    >>> a=DateTimeNano(['12/31/2019', '6/30/2019'], format='%m/%d/%Y', from_tz='NYC')
    DateTimeNano([20191231 00:00:00.000000000, 20190630 00:00:00.000000000])

    From Matlab:
    >>> a=DateTimeNano([737426], from_matlab=True, from_tz='NYC')
    >>> a
    DateTimeNano([20190101 00:00:00.000000000])
    >>> a.format_day; a
    DateTimeNano([20190101])

    From utcnow:
    >> from datetime import datetime as dt
    >> dt.utcnow()
    datetime.datetime(2019, 2, 7, 20, 12, 44, 116810)
    >> DateTimeNano([GetNanoTime()], from_tz="GMT")
    DateTimeNano([20190207 15:12:44.116810400])

    Math operations:
    ----------------
    The following math operations can be performed and will yeild the following object types

    DateTimeNano - DateTimeNano = TimeSpan
    DateTimeNano - TimeSpan = DateTimeNano
    DateTimeNano + TimeSpan = DateTimeNano
    TimeSpan - TimeSpan = TimeSpan
    TimeSpan + TimeSpan = TimeSpan

    Other notes:
    ------------
    - The constructor does not attempt to preserve NaN times from python datetime objects.


    '''
    MetaVersion = 0
    MetaDefault = {
        'name': 'DateTimeNano',
        'typeid': TypeId.DateTimeNano,
        'ncols': 0,
        'version': 0,

        'instance_vars': {
            '_display_length': DisplayLength.Long,
            '_to_tz': 'NYC'
        }
    }
    # TODO: add more intervals here and to DateTimeNano quarters
    # need to interact with the business calendar class
    # maybe merge these with TimeSpan unit conversion dict?
    FrequencyStrings = {
        'H': 'h',
        'T': 'm',
        'MIN': 's',
        'S': 's',
        'L': 'ms',
        'MS': 'ms',
        'U': 'us',
        'US': 'us',
        'N': 'ns',
        'NS': 'ns',
    }
    _INVALID_FREQ_ERROR = "Invalid frequency: {}"

    # ------------------------------------------------------------
    def __new__(cls, arr, from_tz=None, to_tz=None, from_matlab=False, format=None, start_date=None, gmt=None):
        '''

        Array of nanoseconds since Unix Epoch (held in int64)
        All DateTimeNano objects hold nanoseconds in GMT time.

        Parameters:
        -----------
        arr         : nanoseconds in integer, timestrings, numpy datetime64 array
        from_tz     : if initialized from strings, user is required to specify the timezone of origin (otherwise default is UTC)
                      Currently supported:
                      'NYC'    : US/Eastern (accounts for daylight savings)
                      'DUBLIN' : Dublin (accounts for daylight savings)
                      'GMT'    : Greenwich Mean Time
        to_tz       : controls how the data will be displayed. if not set, will match the from_tz keyword
        from_matlab : indicates that the input was matlab ordinal days (may also have fraction of day, so specify from_tz)
        format      : specify a format for timestrings - will be ignored unless strings are in arr
                      See strptime_to_nano() docstring for format info
        start_date   : specify a single date for datetime nano. all times in provided array will be interpretted
                      as nanoseconds.
        (gmt **deprecated)

        Notes:
        ------
        - If the integer data in a DateTimeNano object is extracted, it is in GMT time. To initialize another
          DateTimeNano with the same underlying array, need to set from_tz='GMT' or 'UTC'
        - the gmt keyword is no longer used, need to add a deprication warning at some point
        - DateTimeNano has no knowledge of timezones. All timezone operations are handled by the TimeZone class

        Examples
        --------

        >>> dtn = DateTimeNano(['20180201 12:34'], from_tz='NYC')
        >>> dtn
        DateTimeNano([20180201 12:34:00.000000000])


        >>> dtn = DateTimeNano(['2/1/1992', '5/12/1995'], from_tz='NYC', format='%m/%d/%Y')
        >>> dtn
        DateTimeNano([19920201 00:00:00.000000000, 19950512 00:00:00.000000000])

        '''
        # changing defaults / requirments based on constructor
        # non-string constructors don't require from_tz keyword to be set
        # need to store original keyword values to check in the funnel (saving all in case we add more)
        _orig_from_tz = from_tz
        if from_tz is None:
            from_tz = 'UTC'

        _from_matlab = from_matlab
        _format = format
        _start_date = start_date

        # check for categorical of string or dates
        arr, cat = _possibly_convert_cat(arr)

        if isinstance(arr, TypeRegister.Date):
            if to_tz is None:
                to_tz = 'UTC'
            # will automatically flip to int64, send through as nanosecond integer array
            arr = arr._fa * NANOS_PER_DAY
        else:
            if to_tz is None:
                to_tz = 'NYC'

        # create a timezone object to handle daylight savings, any necessary conversion, etc.
        _timezone = TypeRegister.TimeZone(from_tz=from_tz, to_tz=to_tz)

        if from_matlab:
            instance = cls._convert_matlab_days(arr, _timezone)
        else:
            if start_date is not None:

                if not isinstance(arr, np.ndarray):
                    arr = FastArray(arr)
                # if array was strings, interpret as timespan
                # numeric arrays will also be interpretted as timespan
                if arr.dtype.char in 'US':
                    arr = TimeSpan(arr)

                # interpret as start date in nanoseconds
                if isinstance(start_date, (str, bytes)):
                    start_date = FastArray(start_date)
                    start_date = rc.DateStringToNanos(start_date)[0]
                elif isinstance(start_date, Date):
                    if len(start_date) == len(arr):
                        # user has passed in multiple start dates
                        start_date = start_date._fa * NANOS_PER_DAY
                    else:
                        start_date = start_date[0] * NANOS_PER_DAY
                else:
                    raise TypeError(f'Start date must be string in format YYYYMMDD or Date object. Got type {type(start_date)}')


            instance = None
            if isinstance(arr, list) or np.isscalar(arr):
                arr = FastArray(arr)
            if isinstance(arr, np.ndarray):

                if arr.dtype.char == 'O':
                    # possibly python datetime object
                    if isinstance(arr[0], dt):
                        # warn if it will take more than 1 second
                        if len(arr) > 750_000:
                            warnings.warn(f"Python is converting {len(arr)} datetime objects. Performance may suffer.")
                        arr = np.array([t.isoformat() for t in arr], dtype='datetime64[ns]')

                # string
                if arr.dtype.char in 'US':
                    if _orig_from_tz is None:
                        raise ValueError(TypeRegister.TimeZone.tz_error_msg)

                    # if format specified, use our strptime
                    if format is not None:
                        instance = strptime_to_nano(arr, format, from_tz=from_tz, to_tz=to_tz)
                    else:
                        # otherwise assume ISO-8601 format
                        instance = datetimestring_to_nano(arr, from_tz=from_tz, to_tz=to_tz)

                    # check for categorical of string
                    if cat is not None:
                        # re-expand since it came in as a categorical
                        instance = cat.expand_any(instance)
                    return instance

                # flip numpy datetime64 array
                elif arr.dtype.char == 'M':
                    instance = arr.astype('datetime64[ns]', copy=False).view(np.int64)

                # don't allow timespan arrays without start date
                elif isinstance(arr, TimeSpan) and start_date is None:
                    raise TypeError(f'Cannot create DateTimeNano from TimeSpan array unless start_date is provided.')

                elif arr.dtype.char in NumpyCharTypes.AllInteger + NumpyCharTypes.AllFloat:
                    pass

                else:
                    raise TypeError(f"Cannot create DateTimeNano object from {arr.dtype}")

                # only flip to int64 if necessary
                # TODO: for uint64 do we want a .view() so we dont have to convert?
                instance = arr.astype(np.int64, copy=False)

                if start_date is not None:
                    instance = instance + start_date

                # match stored utc nano to desired display
                instance = _timezone.to_utc(instance)
            else:
                raise TypeError(f"Cannot initialize DateTimeNano with type {type(arr)}, must be list or array.")

        # check for categorical of string
        if cat is not None:
            # re-expand since it came in as a categorical
            instance = cat.expand_any(instance)

        instance = instance.view(cls)
        instance._display_length = DisplayLength.Long
        instance._timezone = _timezone

        return instance

    # ------------------------------------------------------------
    def __init__(self, arr, from_matlab=False, from_tz=None, to_tz=None, format=None, start_date=None, gmt=None):
        pass

    # ------------------------------------------------------------
    def get_classname(self):
        '''
        Return object's class name for array repr.

        Returns
        -------
        obj:`str`
            Object's class name.
        '''
        return __class__.__name__

    # ------------------------------------------------------------
    def get_scalar(self, scalarval):
        return DateTimeNanoScalar(scalarval, _from=self)

    # ------------------------------------------------------------
    @classmethod
    def _convert_matlab_days(cls, arr, timezone):
        '''
        Parameters:
        -----------

        arr      : array of matlab datenums (1 is 1-Jan-0000)
        timezone : TimeZone object from DateTimeNano constructor


        Converts matlab datenums to an array of int64 containing utc nanoseconds.
        '''
        inv_mask = isnan(arr)

        # matlab dates come in as float
        # first, flip to float64 so no precision is lost
        arr = arr.astype(np.float64, copy=False)
        arr = arr - MATLAB_EPOCH_DATENUM
        # might be a better way to do this with fewer array copies
        arr *= NANOS_PER_DAY
        arr = arr.astype(np.int64)
        arr = timezone.to_utc(arr, inv_mask=inv_mask)

        putmask(arr, inv_mask, cls.NAN_TIME)

        return arr

    # ------------------------------------------------------------
    def set_timezone(self, tz):
        '''
        Changes the timezone that the times are displayed in.
        Different lookup array will be used for daylight savings fixups.
        Does not modify the underlying array.

        Parameters
        ----------
        tz : str
            Abbreviated name of desired timezone. See rt.TimeZone.valid_timezones

        Examples
        --------
        Normal:
        >>> dtn = DateTimeNano(['2019-01-07 10:36'], from_tz='NYC', to_tz='NYC')
        >>> dtn
        DateTimeNano([20190107 10:36:00.000000000])
        >>> dtn.set_timezone('DUBLIN')
        >>> dtn
        DateTimeNano([20190107 15:36:00.000000000])

        NYC is in daylight savings time, Dublin is not:
        >>> dtn = DateTimeNano(['2019-03-15 10:36'], from_tz='NYC', to_tz='NYC')
        >>> dtn
        DateTimeNano([20190315 10:36:00.000000000])
        >>> dtn.set_timezone('DUBLIN')
        >>> dtn
        DateTimeNano([20190315 14:36:00.000000000])
        '''
        self._timezone._set_timezone(tz)

    # ------------------------------------------------------------
    def astimezone(self, tz):
        '''
        Returns a new DateTimeNano object in a different displayed timezone.
        The new object holds a reference to the same underlying array.

        Parameters
        ----------
        tz : str
            Abbreviated name of desired timezone. See rt.TimeZone.valid_timezones

        Returns
        -------
        obj:`DateTimeNano`

        Notes
        -----
        Unlike Python's datetime.datetime.astimezone(), accepts strings, not timezone objects.
        '''
        return DateTimeNano(self._fa, from_tz='GMT', to_tz=tz)

    # ------------------------------------------------------------
    def to_iso(self):
        '''
        Generates a FastArray of ISO-8601 timestamp bytestrings.
        The string will match the time +/- timezone offset displayed in the output of the DateTimeNano object.

        Examples
        --------
        >>> dtn = DateTimeNano(['2019-01-22 12:34'],from_tz='NYC')
        >>> dtn
        DateTimeNano([20190122 12:34:00.000000000])
        >>> dtn.to_iso()
        FastArray([b'2019-01-22T12:34:00.000000000'], dtype='|S48')

        >>> dtn = DateTimeNano(['2019-01-22'],from_tz='GMT',to_tz='NYC')
        >>> dtn
        DateTimeNano([20190121 19:00:00.000000000])
        >>> dtn.to_iso()
        FastArray([b'2019-01-21T19:00:00.000000000'], dtype='|S48')

        Returns
        -------
        obj:`FastArray`

        '''
        inv_mask = self.isnan()
        arr = self._timezone.fix_dst(self._fa)
        arr = arr.astype('datetime64[ns]')
        putmask(arr, inv_mask, np.datetime64('nat'))
        return arr.astype('S')

    @property
    def display_length(self):
        if not hasattr(self, '_display_length'):
            self._display_length = DisplayLength.Long
        return self._display_length

    # TODO uncomment when starfish is implemented and imported
    # def _sf_display_query_properties(self):
    #     itemformat = sf.ItemFormat({'length':self.display_length,
    #                                 'align':sf.DisplayAlign.Right,
    #                                 'timezone_str':self._timezone._timezone_str})
    #     return itemformat, self.display_convert_func

    # ------------------------------------------------------------
    def display_query_properties(self):
        # if TypeRegister.DisplayOptions.STARFISH:
        #    return self._sf_display_query_properties()
        '''
        Call back for display functions to get the formatting function and style for timestrings.
        Each instance knows how to format its time strings. The formatter is specified in TIME_FORMATS
        The length property of item_format stores the index into TIME_FORMATS for the display_convert_func

        Returns
        -------
        obj:`ItemFormat`
            See riptable.Utils.rt_display_properties
        function
            Callback function for formatting the timestring
        '''
        item_format = ItemFormat(
            length=self.display_length,
            justification=DisplayJustification.Right,
            can_have_spaces=True,
            decoration=None,
            timezone_str=self._timezone._timezone_str
        )
        convert_func = self.display_convert_func
        return item_format, convert_func

    # ------------------------------------------------------------
    @staticmethod
    def display_convert_func(utcnano, itemformat: ItemFormat):
        '''
        Convert a utc nanosecond timestamp to a string for display.

        Parameters
        ----------
        utcnano : int
            Timestamp in nanoseconds, a single value from a DateTimeNano array
        itemformat : obj:`ItemFormat`
            Style object retrieved from display callback.

        Returns
        -------
        str
            Timestamp as string.

        See Also
        --------
        DateTimeNano.display_query_properties
        riptable.Utils.rt_display_properties

        '''
        # TODO: apply ItemFormat options that were passed in
        return DateTimeNano.format_nano_time(utcnano, itemformat)

    # ------------------------------------------------------------
    def display_item(self, utcnano):
        '''
        Convert a utc nanosecond timestamp to a string for array repr.

        Parameters
        ----------
        utcnano : int
            Timestamp in nanoseconds, a single value from a DateTimeNano array

        Returns
        -------
        str
            Timestamp as string.
        '''
        itemformat, _ = self.display_query_properties()
        return self.format_nano_time(utcnano, itemformat)

    # -----------------------------------------------------------
    @classmethod
    def _parse_item_format(cls, itemformat):
        '''
        Translate a value in the DisplayLength enum into a time format string
        '''
        if itemformat.length == DisplayLength.Short:
            format_str = TIME_FORMATS[4]
            display_nano = False

        elif itemformat.length == DisplayLength.Medium:
            format_str = TIME_FORMATS[1]
            display_nano = False

        elif itemformat.length == DisplayLength.Long:
            format_str = TIME_FORMATS[3]
            display_nano = True
        else:
            raise ValueError(f"Don't know how to interpret display length: {itemformat.length.name}")

        return format_str, display_nano

    # -----------------------------------------------------------
    @staticmethod
    def format_nano_time(utcnano, itemformat):
        '''
        Convert a utc nanosecond timestamp to a string for display.

        Parameters
        ----------
        utcnano : int
            Timestamp in nanoseconds, a single value from a DateTimeNano array
        itemformat : obj:`ItemFormat`
            Style object retrieved from display callback.

        Returns
        -------
        str
            Timestamp as string.

        Notes
        -----
        Uses Python's datetime module for final string conversion.
        '''
        # TODO: cache the time format string returned
        format_str, display_nano = DateTimeNano._parse_item_format(itemformat)

        if utcnano == INVALID_DICT[np.dtype(np.int64).num] or utcnano == 0:
            return 'Inv'

        # view UTC time as local time
        # tz is dateutil.tz
        # dt is datetime.datetime
        # timezone is datetime.timezone
        localzone = tz.gettz(itemformat.timezone_str)
        try:
            timestr = dt.fromtimestamp((utcnano // NANOS_PER_SECOND), timezone.utc)
            timestr = timestr.astimezone(localzone)
            timestr = timestr.strftime(format_str)
        except:
            raise ValueError(f"DateTime: the utc nano value {utcnano!r} for {timezone.utc!r} is not valid.")

        # possible add ms,us,ns precision to seconds
        # each instance should attempt to set its own precision based on how it was constructed
        if display_nano:
            timestr = DateTimeBase._add_nano_ext(utcnano, timestr)

        return timestr

    # ------------------------------------------------------------
    @classmethod
    def _from_meta_data(cls, arrdict, arrflags, meta):
        meta = MetaData(meta)

        # combine saved attributes with defaults based on version number
        vars = meta['instance_vars']
        for k, v in cls.MetaDefault['instance_vars'].items():
            vars.setdefault(k, v)
        for k, v in cls.MetaDefault.items():
            meta.setdefault(k, v)

        # preparing for future versions in case reconstruction changes
        version = meta['version']
        if version != cls.MetaVersion:
            # current version is 0, will not get hit
            if version == 0:
                pass
            else:
                raise ValueError(
                    f"DateTimeNano cannot load. Version {version!r} not supported. Current version installed is {cls.MetaVersion!r}. Update riptable.")

        # datetime nano integers are always in GMT
        instance = [*arrdict.values()][0]
        instance = cls(instance, from_tz='GMT', to_tz=vars['_to_tz'])

        # after constructor is called, restore all instance variables
        # only need to set this one, to_tz, timezone_str are handled by TimeZone class
        instance._display_length = vars['_display_length']

        return instance

    def _meta_dict(self, name=None):
        """Meta dictionary for _build_sds_meta_data, _as_meta_data
        """
        classname = self.__class__.__name__
        if name is None:
            name = classname
        metadict = {
            'name': name,
            'typeid': getattr(TypeId, classname),
            'classname': classname,
            'ncols': 0,
            'version': self.MetaVersion,
            'author': 'python',

            'instance_vars': {
                '_display_length': self.display_length,
                '_to_tz': self._timezone._to_tz
            },

            '_base_is_stackable': SDSFlag.Stackable
        }
        return metadict

    # ------------------------------------------------------------
    @classmethod
    def _load_from_sds_meta_data(cls, name, arr, cols, meta, tups: Optional[list] = None):
        '''
        Note
        ----
        This will be changed to a private method with a different name as it only pertains
        to the SDS file format.

        Load DateTimeNano from an SDS file as the correct class.
        Restore formatting if different than default.

        Parameters
        ----------

        name : item's name in the calling container, or the classname 'DateTimeNano' by default
        arr  : underlying integer FastArray in UTC nanoseconds
        cols : empty list (not used for this class)
        meta : meta data generated by build_meeta_data() routine
        tups : empty list (not used for this class)

        returns reconstructed DateTimeNano object.

        '''
        if tups is None:
            tups = list()
        if not isinstance(meta, MetaData):
            meta = MetaData(meta)

        # combine saved attributes with defaults based on version number
        vars = meta['instance_vars']
        for k, v in cls.MetaDefault['instance_vars'].items():
            vars.setdefault(k, v)
        for k, v in cls.MetaDefault.items():
            meta.setdefault(k, v)

        # preparing for future versions in case reconstruction changes
        version = meta['version']
        if version != cls.MetaVersion:
            # current version is 0, will not get hit
            if version == 0:
                pass
            else:
                raise ValueError(
                    f"DateTimeNano cannot load. Version {version!r} not supported. Current version installed is {cls.MetaVersion!r}. Update riptable.")

        # datetime nano integers are always in GMT
        instance = DateTimeNano(arr, from_tz='GMT', to_tz=vars['_to_tz'])

        # after constructor is called, restore all instance variables
        # only need to set this one, to_tz, timezone_str are handled by TimeZone class
        instance._display_length = vars['_display_length']

        return instance

    # ------------------------------------------------------------
    @classmethod
    def newclassfrominstance(cls, instance, origin):
        '''
        Restore timezone/length info.
        '''
        result = instance.view(cls)
        result._timezone = origin._timezone.copy()
        result._display_length = origin._display_length

        return result

    # ------------------------------------------------------------
    def info(self):
        '''
        Returns
        -------
        str
            Verbose array repr with timezone information.
        '''
        print(self.__repr__(verbose=True))

    # -------------------------------------------------------
    def diff(self, periods=1):
        '''
        Returns
        -------
        TimeSpan
        '''
        result = self._fa.diff(periods=periods)
        return TimeSpan(result)

    # ------------------------------------------------------------
    def __repr__(self, verbose=False):
        repr_strings = []
        tz_string = f", to_tz='{self._timezone._to_tz}'"
        repr_strings.append(self.get_classname() + "([" + self._build_string() + "]" + tz_string + ")")

        if verbose is False:
            return "\n".join(repr_strings)

        repr_strings.append(f"Displaying in timezone: {self._timezone._timezone_str}")
        repr_strings.append(f"Origin: {self._timezone._from_tz}")
        repr_strings.append(f"Offset: {self._timezone._offset} hours")
        return "\n".join(repr_strings)

    # ------------------------------------------------------------
    @classmethod
    def hstack(cls, dtlist):
        '''
        Performs an hstack on a list of DateTimeNano objects.
        All items in list must have their display set to the same timezone.

        Parameters
        ----------
        dtlist : obj:`list` of obj:`DateTimeNano`
            DateTimeNano objects to be stacked.

        Examples
        --------

        >>> dtn1 = DateTimeNano(['2019-01-01', '2019-01-02'], from_tz='NYC')
        >>> dtn2 = DateTimeNano(['2019-01-03', '2019-01-04'], from_tz='NYC')
        >>> DateTimeNano.hstack([dtn1, dtn2])
        DateTimeNano([20190101 00:00:00.000000000, 20190102 00:00:00.000000000, 20190103 00:00:00.000000000, 20190104 00:00:00.000000000])

        Returns
        -------
        obj:`DateTimeNano`

        '''
        return hstack_any(dtlist, cls, DateTimeNano)

    # ------------------------------------------------------------
    def shift(self, periods=1):
        '''
        Modeled on pandas.shift.
        Values in the array will be shifted to the right if periods is positive, to the left if negative.
        Spaces at either end will be filled with invalid.
        If abs(periods) >= the length of the array, the result will be full of invalid.

        Parameters
        ----------
        periods : int
            Number of periods to move, can be positive or negative

        Returns
        -------
        obj:`DateTimeNano`

        '''
        temp = FastArray.shift(self, periods=periods)
        return self.newclassfrominstance(temp, self)

    # -------------------------------------------------------------
    def cut_time(self, buckets: Union[int, 'TimeSpan', List], start_time: Tuple = None,
                 end_time: Tuple = None, add_pre_bucket: bool = False, add_post_bucket: bool = False,
                 label: str = "left", label_fmt: str = "%H:%M", nyc: bool = False) -> TypeRegister.Categorical:
        """
        Analogous to rt.cut() but for times. We ignore the date part and cut based on time of day component only.

        Parameters
        ----------
        buckets: int or rt.TimeSpan or a list of for custom buckets
            Specify your bucket size or buckets. Supply either an int for the common use case of equally sized minute buckets or a custom list

            Acceptable lists formats:
                [(h, m, s, ms)] - it'll assume fields are 0 if length is less than 4
        start_time: optional if buckets is explicitly supplied, (h, m) or (h, m, s) or (h, m , s, ms) tuple
            left end point of first bucket, this type may change in future
        end_time:
            see start_time, right end point of last bucket
        add_pre_bucket: bool
            add a pre-open bucket or treat as invalid ?
        add_post_bucket: bool
            add a after close bucket or treat as invalid ?
        label: optional str
            "left": for left end points
            "right": for right end points
        label_fmt: optional str
            strftime format for label
        nyc: bool, default is False
            convenience shortcut to default to NYC start and end time, ignored if buckets explicitly supplied

        Returns
        -------
        rt.Categorical

        See Also
        --------
        inspired from pandas TimeGrouper

        Examples
        --------
        TODO - sanitize - add cut_time examples
        See the version history for structure of older examples.
        """

        # first define some helper functions
        def timetuple_to_nsm(tup) -> int:
            if not 2 <= len(tup) <= 4:
                raise ValueError("Expected (h,m), (h,m,s) or (h,m,s,ms)")

            zeros = (0,) * (4 - len(tup))
            h, m, s, ms = tup + zeros

            return 1_000_000 * (ms + 1000 * (s + 60 * (m + 60 * h)))

        def nano_to_label(nanos: int) -> str:
            # Just care about minutes, hours
            return DateTimeNanoScalar(nanos, to_tz='UTC').strftime(label_fmt)

        def scalar(arr_or_scalar):
            try:
                len(arr_or_scalar)
            except Exception:
                return arr_or_scalar

            if len(arr_or_scalar) == 1:
                return arr_or_scalar[0]

            raise ValueError("not a length 1 array")

        # end helper functions

        is_already_list = False
        if isinstance(buckets, int):
            buckets = TimeSpan(buckets, "m")
        elif isinstance(buckets, TimeSpan):
            pass
        elif isinstance(buckets, type([])):
            is_already_list = True
        else:
            raise ValueError(f"Unknown bucket_size type, got: {type(buckets)}")

        # two cases bucket_size is already a list or it's a TimeSpan
        if is_already_list:
            bucket_cut_points = [timetuple_to_nsm(xx) for xx in sorted(buckets)]
        else:
            if nyc and (start_time is not None and end_time is not None):
                raise ValueError("If nyc is True then you can't set both start and end time bounds")

            if nyc:
                if start_time is None: start_time = (9, 30)
                if end_time is None: end_time = (16, 15)

            if start_time is None or end_time is None:
                raise ValueError("Need to specify start and end times")

            bucket_cut_points = []

            now_nsm = timetuple_to_nsm(start_time)
            end_time_nsm = timetuple_to_nsm(end_time)
            bucket_size_nsm = buckets.nanoseconds

            while now_nsm < end_time_nsm:
                bucket_cut_points.append(scalar(now_nsm))
                now_nsm += bucket_size_nsm

            bucket_cut_points.append(end_time_nsm)

        if add_pre_bucket:
            bucket_cut_points.insert(timetuple_to_nsm((0, 0)), 0)

        if add_post_bucket:
            bucket_cut_points.append(timetuple_to_nsm((24, 0)))

        if label == "right":
            bucket_cut_labels = [nano_to_label(xx) for xx in bucket_cut_points[1:]]
        elif label == "left":
            bucket_cut_labels = [nano_to_label(xx) for xx in bucket_cut_points[:-1]]
        else:
            raise ValueError(f"Unknown label, got {label}")

        if add_pre_bucket:
            bucket_cut_labels[0] = "pre"
        if add_post_bucket:
            bucket_cut_labels[-1] = "post"

        cat = searchsorted(bucket_cut_points, self.nanos_since_midnight())

        # map right side invalid to 0
        cat[cat >= len(bucket_cut_points)] = 0

        return TypeRegister.Categorical(cat, bucket_cut_labels, base_index=1, ordered=False)

    # -------------------------------------------------------------
    def fill_invalid(self, shape=None, dtype=None, inplace=True):
        arr = self._fill_invalid_internal(shape=shape, dtype=self.dtype, fill_val=self.NAN_TIME, inplace=inplace)
        if arr is None:
            return
        return DateTimeNano(arr, from_tz='GMT', to_tz=self._timezone._to_tz)

    # -------------------------------------------------------------
    def isnan(self):
        '''
        Boolean array, True where DateTimeNano == NaN time or int64 sentinel (min int)

        Returns
        -------
        bool array
            True where object == NaN time or int64 sentinel (min int), otherwise False.

        Notes
        -----
        Currently using 0 for NaN time. This constant is held in the DateTimeBase class.
        '''
        return self._fa.isnanorzero()

    # -------------------------------------------------------------
    def isnotnan(self):
        '''
        Boolean array, True where DateTimeNano != NaN time or int64 sentinel (min int).

        Returns
        -------
        bool array
            True where object != NaN time or int64 sentinel (min int), otherwise False.

        Notes
        -----
        Currently using 0 for NaN time. This constant is held in the DateTimeBase class.
        '''
        return ~self.isnan()

    # -------------------------------------------------------------
    def _datetimenano_compare_check(self, funcname, other):
        caller = self._fa

        if isinstance(other, (DateTimeNano, DateTimeNanoScalar)):
            if other._timezone._to_tz != self._timezone._to_tz:
                warnings.warn(
                    f'DateTimeNano objects are being displayed in different timezones. Results may not appear to be correct for {funcname}')

        elif isinstance(other, (Date, DateScalar)):
            other = DateTimeNano(other._fa * NANOS_PER_DAY, from_tz=self._timezone._to_tz, to_tz=self._timezone._to_tz)

        elif isinstance(other, (TimeSpan, DateSpan, TimeSpanScalar, DateSpanScalar)):
            raise TypeError(f'Cannot compare DateTimeNano with {type(other)}')
        # let everything else fall through to fast array

        # restore invalids
        return self._preserve_invalid_comparison(caller, other, funcname)

    # -------------------COMPARISONS------------------------------
    # ------------------------------------------------------------
    def __ne__(self, other):
        return self._datetimenano_compare_check('__ne__', other)

    def __eq__(self, other):
        return self._datetimenano_compare_check('__eq__', other)

    def __ge__(self, other):
        return self._datetimenano_compare_check('__ge__', other)

    def __gt__(self, other):
        return self._datetimenano_compare_check('__gt__', other)

    def __le__(self, other):
        return self._datetimenano_compare_check('__le__', other)

    def __lt__(self, other):
        return self._datetimenano_compare_check('__lt__', other)

    # -------------------------------------------------------------
    def min(self, **kwargs):
        '''
        Earliest timestamp in array.

        Returns
        -------
        obj:`DateTimeNano`
            An array with length 1.

        Note
        ----
        This returns an array, not a scalar. However, broadcasting rules will apply to operations with it.
        '''
        return DateTimeNano([self._fa.min()], from_tz='GMT', to_tz=self._timezone._to_tz)
        # return DateTimeNanoScalar(self._fa.min(), timezone=self._timezone)

    # -------------------------------------------------------------
    def max(self, **kwargs):
        '''
        Latest timestamp in array.

        Returns
        -------
        obj:`DateTimeNano`
            An array with length 1.

        Note
        ----
        This returns an array, not a scalar. However, broadcasting rules will apply to operations with it.
        '''
        return DateTimeNano([self._fa.max()], from_tz='GMT', to_tz=self._timezone._to_tz)
        # return DateTimeNanoScalar(self._fa.max(), timezone=self._timezone)

    # -------------------------------------------------------------
    def diff(self, periods=1):
        '''
        Calculate the n-th discrete difference.

        Parameters
        ----------
        periods : int, optional
            The number of times values are differenced. If zero, the input
            is returned as-is.

        Returns
        -------
        obj:`TimeSpan`
        '''
        return TimeSpan(self._fa.diff(periods=periods).astype(np.float64))

    # -------------------------------------------------------------
    def __radd__(self, other):
        return self.__add__(other)

    # -------------------------------------------------------------
    def __iadd__(self, other):
        # warnings.warn(f'Currently allowing inplace operation __iadd__ on DateTimeNano. May change in the future.')
        return self.__add__(other, inplace=True)

    # -------------------------------------------------------------
    def __add__(self, other, inplace=False):
        call_super = False
        other_inv_mask = None
        func = TypeRegister.MathLedger._BASICMATH_TWO_INPUTS
        op = None

        return_type = DateTimeNano

        if not isinstance(other, np.ndarray) and not isinstance(other, (DateTimeNanoScalar, DateScalar, TimeSpanScalar, DateSpanScalar)):
            # TJD change
            if np.isscalar(other):
                other = np.int64(other)
            else:
                other = FastArray(other, dtype=np.int64)
            # op = MATH_OPERATION.ADDDATES
            call_super = True

        else:
            if isinstance(other, (DateTimeNano, DateTimeNanoScalar)):
                raise TypeError(f'Cannot add two objects {type(self)} and {type(other)}')
            elif isinstance(other, (Date, DateScalar)):
                raise TypeError(f'Cannot add two objects {type(self)} and {type(other)}')
            elif isinstance(other, (TimeSpan, TimeSpanScalar)):
                other_inv_mask = isnan(other)
                other = other.astype(np.int64)
                call_super = True
                # op = MATH_OPERATION.ADDDATES
            elif isinstance(other, (DateSpan, DateSpanScalar)):
                other_inv_mask = isnan(other)
                other = other.astype(np.int64) * NANOS_PER_DAY
                call_super = True
                # op = MATH_OPERATION.ADDDATES
            else:
                other = other.view(FastArray)
                other = other.astype(np.int64, copy=False)
                call_super = True
                # op = MATH_OPERATION.ADDDATES

        if inplace:
            funcname = '__iadd__'
        else:
            funcname = '__add__'

        return self._build_mathops_result(other, funcname, call_super, other_inv_mask, inplace, op, return_type)

    # -------------------------------------------------------------
    def __rsub__(self, other):
        if isinstance(other, (Date, DateScalar)):
            return other.__sub__(self)
        else:
            raise TypeError(f'DateTimeNano can only be subtracted from DateTimeNano or Date.')

    # -------------------------------------------------------------
    def __isub__(self, other):
        warnings.warn(f'Currently allowing inplace operation __isub__ on DateTimeNano. May change in the future.')
        return self.__sub__(other, inplace=True)

    # -------------------------------------------------------------
    def __sub__(self, other, inplace=False):
        call_super = False
        other_inv_mask = None
        func = TypeRegister.MathLedger._BASICMATH_TWO_INPUTS
        op = None

        if not isinstance(other, np.ndarray) and not isinstance(other, (DateTimeNanoScalar, DateScalar, TimeSpanScalar, DateSpanScalar)):
            return_type = DateTimeNano
            # TJD change
            if np.isscalar(other):
                other = np.int64(other)
            else:
                other = FastArray(other, dtype=np.int64)
            call_super = True

        else:
            if isinstance(other, (DateTimeNano, DateTimeNanoScalar)):
                # ready to go
                return_type = TimeSpan
                if inplace:
                    raise TypeError(f'__sub__ returns TimeSpan. Cannot perform inplace.')
                op = MATH_OPERATION.SUBDATETIMES

            elif isinstance(other, (Date, DateScalar)):
                return_type = TimeSpan
                op = MATH_OPERATION.SUBDATETIMES
                if inplace:
                    raise TypeError(f'__sub__ returns TimeSpan. Cannot perform inplace.')
                # upcast Date
                other = other.astype(np.int64) * NANOS_PER_DAY

            elif isinstance(other, (TimeSpan, TimeSpanScalar)):
                # apply our own mask during this track
                return_type = DateTimeNano
                # upcast TimeSpan to preserve precision
                other = other.astype(np.int64)
                call_super = True

            elif isinstance(other, (DateSpan, DateSpanScalar)):
                return_type = DateTimeNano
                # need to get mask before upcasting
                other_inv_mask = isnan(other)
                other = other.astype(np.int64) * NANOS_PER_DAY
                call_super = True

            else:
                # user fastarray operation
                return_type = DateTimeNano
                other = other.view(FastArray)
                other = other.astype(np.int64, copy=False)
                # op = MATH_OPERATION.SUBDATETIMESLEFT
                call_super = True

        if inplace:
            funcname = '__isub__'
        else:
            funcname = '__sub__'

        return self._build_mathops_result(other, funcname, call_super, other_inv_mask, inplace, op, return_type)

    def __matmul__(self, other): raise NotImplementedError

    # need to check properties to see if division is happening
    #def __truediv__(self, other): raise NotImplementedError
    #def __floordiv__(self, other): raise NotImplementedError
    #def __mod__(self, other): raise NotImplementedError
    #def __divmod__(self, other): raise NotImplementedError
    
    def __pow__(self, other, modulo=None): raise NotImplementedError

    def __lshift__(self, other): raise NotImplementedError

    def __rshift__(self, other): raise NotImplementedError

    def __and__(self, other): raise NotImplementedError

    def __xor__(self, other): raise NotImplementedError

    def __or__(self, other): raise NotImplementedError

    def __rmul__(self, other): raise NotImplementedError

    def __rmatmul__(self, other): raise NotImplementedError

    def __rtruediv__(self, other): raise NotImplementedError

    def __rfloordiv__(self, other): raise NotImplementedError

    def __rmod__(self, other): raise NotImplementedError

    def __rdivmod__(self, other): raise NotImplementedError

    def __rpow__(self, other): raise NotImplementedError

    def __rlshift__(self, other): raise NotImplementedError

    def __rrshift__(self, other): raise NotImplementedError

    def __rand__(self, other): raise NotImplementedError

    def __rxor__(self, other): raise NotImplementedError

    def __ror__(self, other): raise NotImplementedError

    def __imul__(self, other): raise NotImplementedError

    def __imatmul__(self, other): raise NotImplementedError

    def __itruediv__(self, other): raise NotImplementedError

    def __ifloordiv__(self, other): raise NotImplementedError

    def __imod__(self, other): raise NotImplementedError

    def __ipow__(self, other, modulo=None): raise NotImplementedError

    def __ilshift__(self, other): raise NotImplementedError

    def __irshift__(self, other): raise NotImplementedError

    def __iand__(self, other): raise NotImplementedError

    def __ixor__(self, other): raise NotImplementedError

    def __ior__(self, other): raise NotImplementedError

    def __neg__(self): raise NotImplementedError

    def __pos__(self): raise NotImplementedError

    def __invert__(self): raise NotImplementedError

    def __complex__(self): raise NotImplementedError

    def __int__(self): raise NotImplementedError

    def __float__(self): raise NotImplementedError

    def __round__(self, ndigits=0): raise NotImplementedError

    def __trunc__(self): raise NotImplementedError

    def __floor__(self): raise NotImplementedError

    def __ceil__(self): raise NotImplementedError

    # -------------------------------------------------------------
    # ----raise error on certain math operations-------------------
    # def __radd__(self, value):
    #    return self.__add__(value)

    def __mul__(self, value):
        return self._guard_math_op(value, '__mul__')

    def __floordiv__(self, value):
        return self._guard_math_op(value, '__floordiv__')

    def __truediv__(self, value):
        return self._guard_math_op(value, '__truediv__')

    def __abs__(self):
        raise TypeError(f"Cannot perform absolute value on DateTimeNano object.")

    def _guard_math_op(self, value, op_name):
        if isinstance(value, DateTimeBase):
            raise TypeError(f"Cannot perform operation {op_name} between DateTimeNano and {type(value)}")
        op = getattr(self._fa, op_name)
        return op(value)

    # -------------------------------------------------------------
    @classmethod
    def _random(cls, sz, to_tz='NYC', from_tz='NYC', inv=None, start=None, end=None):
        '''
        Internal routine for random(), random_invalid()
        '''
        if start is None:
            start = NANOS_PER_YEAR
            end = NANOS_PER_YEAR * 50
        else:
            start = (start - 1970) * NANOS_PER_YEAR
            if end is None:
                # maybe test if leap year?
                end = start + NANOS_PER_YEAR

        arr = np.random.randint(start, end, sz, dtype=np.int64)
        if inv is not None:
            putmask(arr, inv, 0)
        return DateTimeNano(arr, to_tz=to_tz, from_tz=from_tz)

    @classmethod
    def random(cls, sz, to_tz='NYC', from_tz='NYC', inv=None, start=None, end=None):
        '''
        Returns a random DateTimeNano object.
        Times will range from from 1971 -> 2021 unless `start` and `end` are specified.

        Parameters
        ----------
        sz : int
            Length of generated array
        to_tz : str, optional, default 'NYC'
            Timezone string for display
        from_tz : str, optional, default 'NYC'
            Timezone string for timezone of origin
        inv : bool array, optional, default None
            An invalid mask True where invalid times should be inserted
        start : int, optional, default None
            Start year for range of random times. If no end year provided, all times will be within start year
        end : int, optional, default None
            End year for range of random times. Only used if `start` provided.

        Examples
        --------
        >>> DateTimeNano.random(3)
        DateTimeNano([19980912 15:31:08.025189457, 19931121 15:48:32.855425859, 19930915 14:58:31.376750294])

        Returns
        -------
        obj:`DateTimeNano`

        See Also
        --------
        DateTimeNano.random_invalid

        '''
        return cls._random(sz, to_tz=to_tz, from_tz=from_tz, inv=inv, start=start, end=end)

    @classmethod
    def random_invalid(cls, sz, to_tz='NYC', from_tz='NYC', start=None, end=None):
        '''
        Returns a random DateTimeNano object. Inserts invalids using a random boolean mask.
        Times will range from from 1971 -> 2021 unless `start` and `end` are specified.

        Parameters
        ----------
        sz : int
            Length of generated array
        to_tz : str, optional, default 'NYC'
            Timezone string for display
        from_tz : str, optional, default 'NYC'
            Timezone string for timezone of origin
        start : int, optional, default None
            Start year for range of random times. If no end year provided, all times will be within start year
        end : int, optional, default None
            End year for range of random times. Only used if `start` provided.
        Same as DateTimeNano.random(), but random invalid mask is also generated.

        Examples
        --------
        >>> DateTimeNano.random_invalid(3)
        DateTimeNano([19920830 16:17:24.935335183, Inv, Inv])

        Returns
        -------
        obj:`DateTimeNano`

        See Also
        --------
        DateTimeNano.random
        '''
        inv = np.random.randint(0, 2, sz, dtype=np.bool)
        return cls._random(sz, to_tz=to_tz, from_tz=from_tz, inv=inv, start=start, end=end)

    # -------------------------------------------------------------
    def resample(self, rule, dropna=False):
        """Convenience method for frequency conversion and resampling of
        DateTimeNano arrays.

        Parameters
        ----------
        rule : string
            The offset string or object representing target conversion.
            Can also begin the string with a number e.g. '3H'
            Currently supported:
            H       hour
            T, min  minute
            S       second
            L, ms   millisecond
            U, us   microsecond
            N, ns   nanosecond

        dropna : bool, default False
            If True, returns a DateTimeNano the same length as caller, with all values rounded to specified frequency.
            If False, returns a DateTimeNano range from caller's min to max with values at every specified frequency.

        Examples
        --------
        >>> dtn = DateTimeNano(['2015-04-15 14:26:54.735321368',
                                '2015-04-20 07:30:00.858219615',
                                '2015-04-23 13:15:24.526871083',
                                '2015-04-21 02:25:11.768548100',
                                '2015-04-24 07:47:54.737776979',
                                '2015-04-10 23:59:59.376589955'],
                                     from_tz='UTC', to_tz='UTC')
        >>> dtn.resample('L', dropna=True)
        DateTimeNano(['20150415 14:26:54.735000000', '20150420 07:30:00.858000000', '20150423 13:15:24.526000000', '20150421 02:25:11.768000000', '20150424 07:47:54.737000000', '20150410 23:59:59.376000000'], to_tz='UTC')

        >>> dtn = DateTimeNano(['20190417 17:47:00.000001',
                                '20190417 17:47:00.000003',
                                '20190417 17:47:00.000005'],
                                              from_tz='NYC')
        >>> dtn.resample('1us')
        DateTimeNano(['20190417 17:47:00.000001000', '20190417 17:47:00.000002000', '20190417 17:47:00.000003000', '20190417 17:47:00.000004000', '20190417 17:47:00.000005000'], to_tz='NYC')

        Returns
        -------
        dtn : `DateTimeNano`
        """

        # -------------------------------------------------------
        def parse_rule(rule):
            # returns an integer or float amount and unit string
            amount = None
            for i, c in enumerate(rule):
                if not c.isnumeric() and c != '.':
                    if i == 0:
                        amount = 1
                    else:
                        amount = rule[:i]
                        try:
                            amount = int(amount)
                        except:
                            amount = float(amount)
                    break

            # never hit a string interval code
            if amount is None:
                raise ValueError(self._INVALID_FREQ_ERROR.format(rule))

            unit = rule[i:].upper()
            unit = self.FrequencyStrings.get(unit, None)
            if unit is None:
                raise ValueError(self._INVALID_FREQ_ERROR.format(rule))

            return amount, unit

        # -------------------------------------------------------
        def get_time_unit(unit):
            if unit in TimeSpan.unit_convert_factors:
                unit = TimeSpan.unit_convert_factors[unit]
            else:
                raise NotImplementedError(f'Currently supports frequency strings {[*self.FrequencyStrings]}')
            return unit

        # -------------------------------------------------------
        def time_interval(amount, unit):
            # amount is a multiplier for the unit
            # unit is a TimeSpan unit or for larger units, will be assigned separately to maintain precision

            # TODO: check for nan times
            # should these be included in any min/max calculation?
            # TJD note this needs to be reviewed... min max should return a scalar not an array of 1
            start = np.int64(self.min()[0])
            end = np.int64(self.max()[0])
            unit = get_time_unit(unit)

            step = np.int64(amount * unit)
            start = start - (start % step)
            # should this include both ends?
            end = (end - (end % step) + step)

            stamps = arange(start, end, step=step)
            interval = DateTimeNano(stamps, to_tz=self._timezone._to_tz)

            return interval

        # -------------------------------------------------------
        def as_time_interval(amount, unit):
            # returns a date time nano the same length as the original
            # may have repeats, empty will not appear
            unit = get_time_unit(unit)
            step = np.int64(amount * unit)
            timediff = self._fa % step
            return self - timediff

        # -------------------------------------------------------

        if not isinstance(rule, str):
            raise TypeError(f'Rule must be a string. Got {type(rule)}.')

        amount, unit = parse_rule(rule)

        if dropna:
            resampled = as_time_interval(amount, unit)
        else:
            resampled = time_interval(amount, unit)

        return resampled


# ========================================================
class TimeSpanBase:
    """
    """
    ReduceFuncs = False
    unit_convert_factors = {
        'Y': NANOS_PER_YEAR,
        'W': NANOS_PER_DAY * 7,
        'D': NANOS_PER_DAY,
        'h': NANOS_PER_HOUR,
        'm': NANOS_PER_MINUTE,
        's': NANOS_PER_SECOND,
        'ms': NANOS_PER_MILLISECOND,
        'us': NANOS_PER_MICROSECOND,
        'ns': 1
    }

    # ------------------------------------------------------------
    def get_classname(self):
        return __class__.__name__

    # ------------------------------------------------------------
    def strftime(self, format, dtype='U'):
        '''
        Converts DateTimeNano to an array of object strings or a scalar string.
        This routine has not been sped up yet.

        Other Parameters
        ----------------
        dtype: defaults to 'U', can change to 'S' or 'U'

        Examples
        --------
        >>> rt.Date(rt.utcnow(4)).strftime('%D')
        array(['11/04/19', '11/04/19', '11/04/19', '11/04/19'], dtype=object)

        See Also
        ---------
        http://strftime.org  for format strings
        datetime.datetime.strftime

        '''
        # get negative mask since strftime does not handle negative
        isnegative = self._fa < 0

        if isinstance(self, np.ndarray):
            result= np.asarray([dt.utcfromtimestamp(timestamp).strftime(format) for timestamp in self._fa.abs() / 1_000_000_000.0], dtype=dtype)
            if isnegative.sum() > 0:
                if dtype == 'S':
                    negcol = zeros(result.shape, dtype='S1')
                    negcol[isnegative] = b'-'
                else:
                    negcol = zeros(result.shape, dtype='U1')
                    negcol[isnegative] = '-'
                result = negcol + result
        else:
            result = dt.strftime(dt.utcfromtimestamp(abs(self) / 1_000_000_000.0), format)
            if isnegative:
                # check dtype 'S'
                if dtype == 'S':
                    result = b'-' + result
                else:
                    result = '-' + result
        return result

    # ------------------------------------------------------------
    # --------RETURN FLOAT ARRAY AT DIFFERENT RESOLUTIONS---------
    @property
    def days(self):
        """Timespan as float64 array of days.
        Note
        ----
        Loss of nanosecond precision at ~52 days.
        """
        return self._fa / NANOS_PER_DAY

    @property
    def hours(self):
        """Timespan as float64 array of hours."""
        return self._fa / NANOS_PER_HOUR

    @property
    def minutes(self):
        """Timespan as float64 array of minutes."""
        return self._fa / NANOS_PER_MINUTE

    @property
    def seconds(self):
        """Timespan as float64 array of seconds."""
        return self._fa / NANOS_PER_SECOND

    @property
    def milliseconds(self):
        """Timespan as float64 array of milliseconds."""
        return self._fa / NANOS_PER_MILLISECOND

    @property
    def microseconds(self):
        """Timespan as float64 array of microseconds."""
        return self._fa / NANOS_PER_MICROSECOND

    @property
    def nanoseconds(self):
        """Timespan as float64 array of nanoseconds (same as underlying array)."""
        return self._fa

    @property
    def hhmmss(self):
        """Timespan as int64 array in format HHMMSS."""
        SEC_PER_MIN = 60
        hour, remainder = divmod(self.astype(np.int64) // NANOS_PER_SECOND, 3600)
        minutes, seconds = divmod(remainder, SEC_PER_MIN)
        return (10_000 * hour + 100 * minutes + seconds).astype(np.int64)

    # ------------------------------------------------------------
    @classmethod
    def _unit_to_nano_span(cls, values, unit):
        '''
        :param values: FastArray from calling constructor
        :param unit: unit string (see numpy's timedelta64 dtype)
        '''
        if isinstance(unit, bytes):
            unit = unit.decode()

        try:
            mult = cls.unit_convert_factors[unit]
        except:
            raise ValueError(f"Cannot initialize span with {unit} units.")

        if mult != 1:
            values = values * mult

        return values

    # ------------------------------------------------------------
    @staticmethod
    def display_item(nanosecs, itemformat=None):

        if itemformat is not None:
            length = itemformat.length
        else:
            length = DisplayLength.Short

        if length == DisplayLength.Medium:
            return TimeSpan.display_item_unit(nanosecs)

        else:
            return TimeSpan.display_item_clock(nanosecs)

    # ------------------------------------------------------------
    @staticmethod
    def display_item_unit(nanosecs):
        '''
        For each item, finds the highest unit to express it in amounts between 1 and 1000 or standard time measure.
        e.g. 59.123m, 678.823ms, 30ns
        '''
        if np.isnan(nanosecs):
            return "Inv"

        # TODO add different formatting for large time spans (> 1 day)
        divisor, unit_str = TimeSpan._display_resolution(nanosecs)
        if divisor == 1:
            delta = str(nanosecs)
        else:
            delta = nanosecs / divisor
            delta = "{0:.3f}".format(delta)

        return delta + unit_str

    @staticmethod
    def _display_resolution(nanosecs):
        '''
        Get extension and divisor for display_item_unit() (see above)
        '''
        nanosecs = abs(nanosecs)
        divisor = NANOS_PER_HOUR
        unit_str = 'h'

        if nanosecs < 1_000:
            divisor = 1
            unit_str = 'ns'

        elif nanosecs < 1_000_000:
            divisor = 1_000
            unit_str = 'us'

        elif nanosecs < NANOS_PER_SECOND:
            divisor = 1_000_000
            unit_str = 'ms'

        elif nanosecs < NANOS_PER_MINUTE:
            divisor = NANOS_PER_SECOND
            unit_str = 's'

        elif nanosecs < NANOS_PER_HOUR:
            divisor = NANOS_PER_MINUTE
            unit_str = 'm'

        # we should probably use a different format past this point
        # maybe a formatting string with more info
        # elif max_time < NANOS_PER_DAY:
        #    divisor = NANOS_PER_HOUR
        #    unit_str = 'h'

        return divisor, unit_str

    # ------------------------------------------------------------
    @staticmethod
    def display_item_clock(nanosecs):
        '''
        Long clock format (default) HH:MM:SS.<nano-decimal>
        '''
        format_str = "%H:%M:%S"
        item = abs(nanosecs)

        if isnan(item):
            timestr = 'Inv'

        else:
            gmt_time = time.gmtime(item / NANOS_PER_SECOND)
            timestr = DateTimeBase.DEFAULT_FORMATTER(format_str, gmt_time)

            days = np.int64(item) // NANOS_PER_DAY
            
            if days > 0:
                timestr = str(days) + 'd ' + timestr
            if nanosecs < 0:
                timestr = "-" + timestr

            timestr = DateTimeBase._add_nano_ext(item, timestr)

        return timestr

    # ------------------------------------------------------------
    @staticmethod
    def display_convert_func(nanosecs, itemformat: ItemFormat):
        return TimeSpan.display_item(nanosecs, itemformat=itemformat)

    # TODO uncomment when starfish is implemented and imported
    # def _sf_display_query_properties(self):
    #     itemformat = sf.ItemFormat({'length':self.display_length,
    #                                 'align':sf.DisplayAlign.Right})
    #     return itemformat, self.display_convert_func

    # ------------------------------------------------------------
    def display_query_properties(self):
        # if TypeRegister.DisplayOptions.STARFISH:
        #    return self._sf_display_query_properties()
        item_format = ItemFormat(
            length=self.display_length,
            justification=DisplayJustification.Right,
            can_have_spaces=True,
            decoration=None
        )
        convert_func = self.display_convert_func
        return item_format, convert_func

    # --BINARY OPERATIONS------------------------------------------
    # -------------------------------------------------------------
    def __add__(self, value):
        other_inv_mask = None

        # TimeSpan add
        if not isinstance(value, np.ndarray):
            value = FastArray(value).astype(np.float64)
        else:
            # DateTimeNano / Date will fix up this operation
            if isinstance(value, (DateTimeNano, DateTimeNanoScalar, Date, DateScalar)):
                return value.__add__(self)

            elif isinstance(value, (DateSpan, DateSpanScalar)):
                other_inv_mask = isnan(value)
                value = value._fa * NANOS_PER_DAY

            else:
                other_inv_mask = isnan(value)
                value = value.view(FastArray)
                value = value.astype(np.float64, copy=False)

        return self._fix_binary_ops(value, '__add__', other_inv_mask=other_inv_mask)

    # -------------------------------------------------------------
    def __radd__(self, value):
        return self.__add__(value)

    # -------------------------------------------------------------
    def __sub__(self, value):
        if isinstance(value, (DateTimeNano, DateTimeNanoScalar, Date, DateScalar)):
            return value.__rsub__(self)
        return self._fix_binary_ops(value, '__sub__')

    # -------------------------------------------------------------
    def __rsub__(self, value):
        if not isinstance(value, np.ndarray):
            value = FastArray(value).astype(np.float64)

        else:
            if isinstance(value, (DateTimeNano, DateTimeNanoScalar, Date, DateScalar)):
                return value.__sub__(self)

            elif isinstance(value, (DateSpan, DateSpanScalar)):
                other_inv_mask = isnan(value)
                value = value._fa * NANOS_PER_DAY

            # interpret everything else as nanosecond timespan values
            else:
                other_inv_mask = isnan(value)
                value = value.view(FastArray)
                value = value.astype(np.float64, copy=False)

        return self._fix_binary_ops(value, '__rsub__')

    # -------------------------------------------------------------
    def __mul__(self, value):
        if isinstance(value, (TimeSpan, DateSpan, Date, DateTimeNano, TimeSpanScalar, DateSpanScalar, DateScalar, DateTimeNanoScalar)):
            raise TypeError(f"Cannot multiply TimeSpan by {type(value)} object.")
        if not isinstance(value, np.ndarray):
            value = FastArray(value).astype(np.float64)
        return self._fix_binary_ops(value, '__mul__')

    # -------------------------------------------------------------
    def __rmul__(self, other):
        return self.__mul__(other)

    # -------------------------------------------------------------
    def __floordiv__(self, value):
        if isinstance(value, (TimeSpan, TimeSpanScalar)):
            result = self._fa.__floordiv__(value)
            return result
        else:
            raise TypeError(f"Can only floor divide TimeSpan objects with other timespan objects not type {type(value)}.")

    # -------------------------------------------------------------
    def __truediv__(self, value):
        # handle TimeSpan('00:30:00') / TimeSpan('01:00:00') with truediv
        if isinstance(value, (TimeSpan, TimeSpanScalar)):
            return self._fa.__truediv__(value)
        return self._fix_binary_ops(value, '__truediv__')

    # -------------------------------------------------------------
    def _fix_binary_ops(self, value, op_name, other_inv_mask=None):
        '''
        Preserves invalids from integer arrays. If valid, wraps result fastarray in TimeSpan object.
        '''
        # print("binary", type(self), type(value), op_name)
        if np.isscalar(self):
            op = getattr(np.float64, op_name)
            result = op(self, value)
        else:
            # get the array version
            op = getattr(FastArray, op_name)
            result = op(self, value)

        if np.isscalar(result):
            result = TimeSpanScalar(result)

        elif isinstance(result, np.ndarray):
            if other_inv_mask is None:
                # this shouldn't get hit, test
                if result.dtype.char in NumpyCharTypes.AllInteger:
                    inv_mask = value == INVALID_DICT[result.dtype.num]
                    result[inv_mask] = np.nan
            else:
                # possible nan fill
                if len(other_inv_mask) == 1:
                    if isnan(other_inv_mask)[0]:
                        result = TimeSpan(full(len(self), np.nan, dtype=np.float64))
                else:
                    result[other_inv_mask] = np.nan
            result = TimeSpan(result)

        return result

    def __pow__(self, other, modulo=None): raise NotImplementedError

    def __lshift__(self, other): raise NotImplementedError

    def __rshift__(self, other): raise NotImplementedError

    def __and__(self, other): raise NotImplementedError

    def __xor__(self, other): raise NotImplementedError

    def __or__(self, other): raise NotImplementedError

    # def __rmul__(self, other): raise NotImplementedError

    def __rmatmul__(self, other): raise NotImplementedError

    def __rtruediv__(self, other): raise NotImplementedError

    def __rfloordiv__(self, other): raise NotImplementedError

    def __rmod__(self, other): raise NotImplementedError

    def __rdivmod__(self, other): raise NotImplementedError

    def __rpow__(self, other): raise NotImplementedError

    def __rlshift__(self, other): raise NotImplementedError

    def __rrshift__(self, other): raise NotImplementedError

    def __rand__(self, other): raise NotImplementedError

    def __rxor__(self, other): raise NotImplementedError

    def __ror__(self, other): raise NotImplementedError

    def __imul__(self, other): raise NotImplementedError

    def __imatmul__(self, other): raise NotImplementedError

    def __itruediv__(self, other): raise NotImplementedError

    def __ifloordiv__(self, other): raise NotImplementedError

    def __imod__(self, other): raise NotImplementedError

    def __ipow__(self, other, modulo=None): raise NotImplementedError

    def __ilshift__(self, other): raise NotImplementedError

    def __irshift__(self, other): raise NotImplementedError

    def __iand__(self, other): raise NotImplementedError

    def __ixor__(self, other): raise NotImplementedError

    def __ior__(self, other): raise NotImplementedError

    # def __neg__(self): raise NotImplementedError

    # def __pos__(self): raise NotImplementedError

    # def __abs__(self): raise NotImplementedError

    def __invert__(self): raise NotImplementedError

    def __complex__(self): raise NotImplementedError

    def __int__(self): raise NotImplementedError

    # def __float__(self): raise NotImplementedError

    def __round__(self, ndigits=0): raise NotImplementedError

    def __trunc__(self): raise NotImplementedError

    def __floor__(self): raise NotImplementedError

    def __ceil__(self): raise NotImplementedError

    #--UNARY OPERATIONS-------------------------------------------
    #-------------------------------------------------------------
    def __abs__(self): return self._unary_ufunc_builder('__abs__')

    def __neg__(self): return self._unary_ufunc_builder('__neg__')

    def __pos__(self): return self._unary_ufunc_builder('__pos__')

    def abs(self): return self.__abs__()

    def _unary_ufunc_builder(self, op_name):
        if np.isscalar(self):
            func = getattr(np.float64, op_name)
            return TimeSpanScalar(func(self))
        else:
            # call the fastarray version of the function
            return TimeSpan(getattr(self._fa, op_name)())

    # ------------------------------------------------------------
    @classmethod
    def _reduce_func_builder(cls):
        '''
        Generates all reduce functions - which return a single value (in nanoseconds).
        The value will be flipped to float64 (we don't need higher precision than nanoseconds), and put in a
        new TimeSpan.
        '''
        for name in ['sum', 'mean', 'std', 'var', 'min', 'max', 'median',
                     'nansum', 'nanmean', 'nanstd', 'nanvar', 'nanmin', 'nanmax', 'nanmedian']:
            func_string = []
            func_string.append("def " + name + "(self, **kwargs):")
            func_string.append("    r = self._fa." + name + "()")
            func_string.append("    r = FastArray(r, dtype=np.float64)")
            func_string.append("    return TimeSpan(r)")
            func_string.append("setattr(cls, '" + name + "', " + name + ")")
            exec("\n".join(func_string))

    # ------------------------------------------------------------
    # -------------------------------------------------------------
    def _timespan_compare_check(self, funcname, other):
        func = getattr(self._fa, funcname)

        if isinstance(other, (str, bytes)):
            other = TimeSpan(other)[0]

        if isinstance(other, (DateTimeNano, Date)):
            raise TypeError(f'Cannot compare TimeSpan with {type(other)}')
        # upcast DateSpan to nanoseconds
        elif isinstance(other, DateSpan):
            other = (other._fa * NANOS_PER_DAY).astype(np.float64)

        # let everything else fall through to fast array
        result = func(other)

        # invalid will automatically be handled because TimeSpan is float
        return result

    # -------------------COMPARISONS------------------------------
    # ------------------------------------------------------------
    def __ne__(self, other):
        return self._timespan_compare_check('__ne__', other)

    def __eq__(self, other):
        return self._timespan_compare_check('__eq__', other)

    def __ge__(self, other):
        return self._timespan_compare_check('__ge__', other)

    def __gt__(self, other):
        return self._timespan_compare_check('__gt__', other)

    def __le__(self, other):
        return self._timespan_compare_check('__le__', other)

    def __lt__(self, other):
        return self._timespan_compare_check('__lt__', other)


# ========================================================
class TimeSpan(TimeSpanBase, DateTimeBase):
    """Array of time delta in nanoseconds, held in float64.

    Parameters:
    -----------

    values : numeric or string array or scalar
        If string, interpreted as HH:MM:SS.ffffff ( seconds/second fractions optional )
        If numeric, interpreted as nanoseconds, unless `unit` provided.
             single number or array / list of numbers (unless unit is specified, will assume nanoseconds)
    unit : str, optional, default 'ns'
        Precision of data in the constructor. All will be converted to nanoseconds.
        Valid units: 'Y', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'

    Examples
    ---------
    From single string:
    >>> dts = TimeSpan('12:34')
    >>> dts
    TimeSpan([12:34:00.000000000])

    From milliseconds since midnight:
    >>> dts = TimeSpan(FA([34500000., 36500000., 38500000.,]), unit='ms')
    >>> dts
    TimeSpan([09:35:00.000000000, 10:08:20.000000000, 10:41:40.000000000])

    From the result of DateTimeNano subtraction:
    >>> dtn1 = DateTimeNano(['2018-01-01 09:35:00'], from_tz='NYC')
    >>> dtn2 = DateTimeNano(['2018-01-01 07:15:00'], from_tz='NYC')
    >>> dtn1 - dtn2
    TimeSpan([02:20:00.000000000])

    Certain DateTimeNano properties can return a TimeSpan:
    >>> dtn = DateTimeNano(['2018-01-01 09:35:00'], from_tz='NYC')
    >>> dtn.hour_span
    TimeSpan([09:35:00.000000000])

    Can be added to DateTimeNano objects:
    >>> dtn = DateTimeNano(['2018-01-01 09:35:00'], from_tz='NYC')
    >>> ts = TimeSpan(FA([8400000000000.0]))
    >>> dtn + ts
    DateTimeNano([20180101 11:55:00.000000000])

    Can be multiplied / divided by scalars:
    >>> ts = TimeSpan(FA([8400000000000.0]))
    >>> ts
    TimeSpan([02:20:00.000000000])
    >>> ts / 2
    TimeSpan([01:10:00.000000000])
    >>> ts * 5.6
    TimeSpan([13:04:00.000000000])
    """

    # ------------------------------------------------------------
    def __new__(cls, values, unit=None):
        # handle all input as array, scalars -> array of one item
        if not isinstance(values, np.ndarray):
            values = FastArray(values)

        # strings must be in format HH:MM / HH:MM:SS / HH:MM:SS.ffffff
        if values.dtype.char in 'US':
            # send to wrapper for strptime
            return timestring_to_nano(values)

        # init class math funcs
        if cls.ReduceFuncs is False:
            cls._reduce_func_builder()
            cls.ReduceFuncs = True

        # handle all others as numeric
        instance = values.astype(np.float64, copy=False)
        if unit is not None:
            instance = cls._unit_to_nano_span(instance, unit)

        # wrap in class
        instance = instance.view(cls)
        instance._display_length = DisplayLength.Short
        return instance

    # ------------------------------------------------------------
    def get_classname(self):
        return __class__.__name__

    # ------------------------------------------------------------
    def get_scalar(self, scalarval):
        return TimeSpanScalar(scalarval, _from=self)

    # ------------------------------------------------------------
    @classmethod
    def newclassfrominstance(cls, instance, origin):
        result = instance.view(cls)
        result._display_length = origin.display_length
        return result

    # ------------------------------------------------------------
    @classmethod
    def hstack(cls, tspans):
        '''
        TODO: maybe add type checking?
        This is a very simple class, rewrap the hstack result in class.
        '''
        return hstack_any(tspans, cls, TimeSpan)

    # ------------------------------------------------------------
    def fill_invalid(self, shape=None, dtype=None, inplace=True):
        arr = self._fill_invalid_internal(shape=shape, dtype=self.dtype, inplace=inplace)
        if arr is None:
            return
        return TimeSpan(arr)

    @classmethod
    def _from_meta_data(cls, arrdict, arrflags, meta):
        if not isinstance(meta, MetaData):
            meta = MetaData(meta)

        version = meta.get('version', 0)
        default_meta = meta_from_version(cls, version)
        # combine saved attributes with defaults based on version number
        vars = meta['instance_vars']
        for k, v in default_meta.items():
            meta.setdefault(k, v)
        for k, v in default_meta['instance_vars'].items():
            vars.setdefault(k, v)

        instance = [*arrdict.values()][0]
        instance = TimeSpan(instance)

        # restore all instance variables
        vars = meta['instance_vars']
        for name, value in vars.items():
            setattr(instance, name, value)

        return instance

    def _meta_dict(self, name=None):
        classname = self.__class__.__name__
        if name is None:
            name = classname
        metadict = {
            'name': name,
            'typeid': getattr(TypeId, classname),
            'classname': classname,
            'ncols': 0,
            'version': META_VERSION,
            'author': 'python',

            'instance_vars': {
                '_display_length': self.display_length
            },

            '_base_is_stackable': SDSFlag.Stackable
        }
        return metadict

    # ------------------------------------------------------------
    @classmethod
    def _load_from_sds_meta_data(cls, name, arr, cols, meta, tups: Optional[list] = None):
        '''
        Load DateTimeNano from an SDS file as the correct class.
        Restore formatting if different than default.
        '''
        if tups is None:
            tups = list()

        if not isinstance(meta, MetaData):
            meta = MetaData(meta)

        version = meta.get('version', 0)
        default_meta = meta_from_version(cls, version)
        # combine saved attributes with defaults based on version number
        vars = meta['instance_vars']
        for k, v in default_meta.items():
            meta.setdefault(k, v)
        for k, v in default_meta['instance_vars'].items():
            vars.setdefault(k, v)

        instance = TimeSpan(arr)

        # restore all instance variables
        vars = meta['instance_vars']

        for name, value in vars.items():
            setattr(instance, name, value)

        return instance


# ==========================================================
# Scalars
# ==========================================================
class DateScalar(np.int32):
    '''
    Derived from np.int32
    days since unix epoch in 1970
    TODO: need to inherit math functions
    '''

    __slots__ = '_display_length'
    # ------------------------------------------------------------
    def __new__(cls, arr, **kwargs):
        return super().__new__(cls, arr)

    # ------------------------------------------------------------
    def __init__(*args, **kwargs):
        self = args[0]
        _from = kwargs.get('_from', None)
        if _from is not None and hasattr(_from, '_display_length'):
            self._display_length = _from._display_length
        else:
            self._display_length = DisplayLength.Long

    def get_item_format(self):
        item_format = ItemFormat(
            length=self._display_length,
            justification=DisplayJustification.Right,
            can_have_spaces=True,
            decoration=None,
        )
        return item_format

    # ------------------------------------------------------------
    @property
    def _fa(self):
        return self

    # ------------------------------------------------------------
    def get_classname(self):
        return __class__.__name__

    def __repr__(self):
        itemformat = self.get_item_format()
        return Date.format_date_num(self._np, itemformat)

    def __str__(self):
        itemformat = self.get_item_format()
        return Date.format_date_num(self._np, itemformat)

    # ------------------------------------------------------------
    def strftime(self, format, dtype='O'):
        '''
        Converts Date to an array of object strings or a scalar string.
        This routine has not been sped up yet.

        Other Parameters
        ----------------
        dtype: defaults to 'O', can change to 'S' or 'U'

        Examples
        --------
        >>> rt.Date(rt.utcnow(4))[0].strftime('%D')
        '11/04/19'

        See Also
        ---------
        http://strftime.org  for format strings
        datetime.datetime.strftime
        '''
        return dt.strftime(dt.utcfromtimestamp(self * SECONDS_PER_DAY), format)

    # ------------------------------------------------------------
    @property
    def _np(self):
        return self.view(np.int32)

    # used in adding a scalar to a Dataset
    def repeat(self, repeats, axis=None):
        return Date(self._np.repeat(repeats, axis=axis))

    def tile(self, repeats):
        return Date(self._np.tile(repeats))


# ==========================================================
class DateSpanScalar(np.int32):
    '''
    Derived from np.int32
    Number of days between two dates
    '''

    __slots__ = '_display_length'

    # ------------------------------------------------------------
    def __new__(cls, arr, **kwargs):
        return super().__new__(cls, arr)

    # ------------------------------------------------------------
    def __init__(*args, **kwargs):
        self = args[0]
        _from = kwargs.get('_from', None)

        if _from is not None:
            self._display_length = _from._display_length
        else:
            self._display_length = DisplayLength.Long

    def get_item_format(self):
        item_format = ItemFormat(
            length=self._display_length,
            justification=DisplayJustification.Right,
            can_have_spaces=True,
            decoration=None
        )
        return item_format

    # ------------------------------------------------------------
    def get_classname(self):
        return __class__.__name__

    def __repr__(self):
        itemformat = self.get_item_format()
        return DateSpan.format_date_span(self._np, itemformat)

    def __str__(self):
        itemformat = self.get_item_format()
        return DateSpan.format_date_span(self._np, itemformat)

    # ------------------------------------------------------------
    @property
    def _np(self):
        return self.view(np.int32)

    # ------------------------------------------------------------
    @property
    def _fa(self):
        return self.view(np.int32)

    # used in adding a scalar to a Dataset
    def repeat(self, repeats, axis=None):
        return DateSpan(self._np.repeat(repeats, axis=axis))

    def tile(self, repeats):
        return DateSpan(self._np.tile(repeats))


# ==========================================================
class DateTimeNanoScalar(np.int64, DateTimeCommon, TimeStampBase):
    '''
    Derived from np.int64
    NOTE: np.int64 is a SLOT wrapper and does not have a __dict__
    Number of nanoseconds since unix epoch 1970 in UTC
    '''

    __slots__ = '_display_length', '_timezone'

    # ------------------------------------------------------------
    def __new__(cls, arr, **kwargs):
        return super().__new__(cls, arr)

    # ------------------------------------------------------------
    def __init__(*args, **kwargs):
        # This needs more work, especially when init with a string
        self = args[0]
        _from = kwargs.get('_from', None)

        if _from is not None and hasattr(_from, '_timezone'):
            self._timezone = _from._timezone
        else:
            to_tz = kwargs.get('to_tz', None)
            from_tz = kwargs.get('from_tz', None)
            if from_tz is None:
                from_tz = 'UTC'

            if isinstance(self, TypeRegister.Date):
                if to_tz is None:
                    to_tz = 'UTC'
                # will automatically flip to int64, send through as nanosecond integer array
                self = np.int64(self) * NANOS_PER_DAY
            else:
                if to_tz is None:
                    to_tz = 'NYC'

            # create a timezone object to handle daylight savings, any necessary conversion, etc.
            _timezone = TypeRegister.TimeZone(from_tz=from_tz, to_tz=to_tz)
            self._timezone = _timezone

        self._display_length = DisplayLength.Long
        if _from is not None and hasattr(_from, '_display_length'):
            self._display_length = _from._display_length

    def get_item_format(self):
        item_format = ItemFormat(
            length=self._display_length,
            justification=DisplayJustification.Right,
            can_have_spaces=True,
            decoration=None,
            timezone_str=self._timezone._timezone_str
        )
        return item_format

    # ------------------------------------------------------------
    def isnan(self):
        return self <= 0

    # ------------------------------------------------------------
    @property
    def _np(self):
        return self.view(np.int64)

    # ------------------------------------------------------------
    @property
    def _fa(self):
        return self.view(np.int64)

    # ------------------------------------------------------------
    def get_classname(self):
        return __class__.__name__

    # ------------------------------------------------------------
    def __repr__(self):
        itemformat = self.get_item_format()
        # return DateTimeNano.format_nano_time(self._np, itemformat)
        return f"{self.get_classname()}('{DateTimeNano.format_nano_time(self._np, itemformat)}')"

    def __str__(self):
        itemformat = self.get_item_format()
        return DateTimeNano.format_nano_time(self._np, itemformat)

    # --BINARY OPERATIONS------------------------------------------
    # -------------------------------------------------------------
    def __add__(self, value):
        # reroute this back to the nonscalar
        return DateTimeNano.__add__(self, value)

    def __sub__(self, value):
        # reroute this back to the nonscalar
        return DateTimeNano.__sub__(self, value)

    # used in adding a scalar to a Dataset
    def repeat(self, repeats, axis=None):
        return DateTimeNano(self._np.repeat(repeats, axis=axis), to_tz=self._timezone._to_tz, from_tz=self._timezone._from_tz)

    def tile(self, repeats):
        return DateTimeNano(self._np.tile(repeats), to_tz=self._timezone._to_tz, from_tz=self._timezone._from_tz)


# ==========================================================
class TimeSpanScalar(np.float64, TimeSpanBase):
    '''
    Derived from np.float64
    ************ not implemented
    Holds single float values for TimeSpan arrays.
    These will be returned from operations that currently return a TimeSpan of a single item.
    '''

    __slots__ = '_display_length'

    # ------------------------------------------------------------
    def __new__(cls, arr, **kwargs):
        return super().__new__(cls, arr)

    def __new__(cls, scalar, **kwargs):
        # strings must be in format HH:MM / HH:MM:SS / HH:MM:SS.ffffff
        if isinstance(scalar, (str, bytes, np.bytes_, np.str_)):
            # send to wrapper for strptime
            scalar = timestring_to_nano(np.asarray([scalar]))[0]

        return super(TimeSpanScalar, cls).__new__(cls, scalar, **kwargs)

    def __init__(*args, **kwargs):
        self = args[0]
        _from = kwargs.get('_from', None)

        # TimeSpan has no timezone
        if _from is not None:
            self._display_length = _from._display_length
        else:
            self._display_length = DisplayLength.Long

    def get_item_format(self):
        item_format = ItemFormat(
            length=self._display_length,
            justification=DisplayJustification.Right,
            can_have_spaces=True,
            decoration=None
        )
        return item_format

    # ------------------------------------------------------------
    @property
    def _fa(self):
        # must go to numpy or it will flip back to an array
        return self.view(np.float64)

    # ------------------------------------------------------------
    @property
    def _np(self):
        return self.view(np.float64)

    # ------------------------------------------------------------
    def get_classname(self):
        return __class__.__name__

    # ------------------------------------------------------------
    def __repr__(self):
        itemformat = self.get_item_format()
        return f"{self.get_classname()}('{TimeSpan.display_item_clock(self._np)}')"

    def __str__(self):
        itemformat = self.get_item_format()
        return TimeSpan.display_item_clock(self._np)

    # because np.float64 is first, it hooks these before TimeSpanBase

    def __abs__(self): return self._unary_ufunc_builder('__abs__')

    def __neg__(self): return self._unary_ufunc_builder('__neg__')

    def __pos__(self): return self._unary_ufunc_builder('__pos__')

    def abs(self):   return self.__abs__()

    #--BINARY OPERATIONS------------------------------------------
    #-------------------------------------------------------------
    def __add__(self, value):      return TimeSpanBase.__add__(self, value)

    def __radd__(self, value):     return TimeSpanBase.__radd__(self, value)

    def __sub__(self, value):      return TimeSpanBase.__sub__(self, value)

    def __rsub__(self, value):     return TimeSpanBase.__rsub__(self, value)

    def __mul__(self, value):      return TimeSpanBase.__mul__(self, value)

    def __rmul__(self, other):     return TimeSpanBase.__rmul__(self, value)

    def __floordiv__(self, value): return TimeSpanBase.__floordiv__(self, value)

    def __truediv__(self, value):  return TimeSpanBase.__truediv__(self, value)

    def __eq__(self, other):
        return self._timespan_compare_check('__eq__', other)

    # used in adding a scalar to a Dataset
    def repeat(self, repeats, axis=None):
        return TimeSpan(self._np.repeat(repeats, axis=axis))

    def tile(self, repeats):
        return TimeSpan(self._np.tile(repeats))


# -----------------------------------------------------
# keep this at end of file
TypeRegister.DateTimeBase = DateTimeBase
TypeRegister.DateTimeNano = DateTimeNano
TypeRegister.TimeSpan = TimeSpan

TypeRegister.DateBase = DateBase
TypeRegister.Date = Date
TypeRegister.DateSpan = DateSpan

TypeRegister.DateTimeNanoScalar = DateTimeNanoScalar
TypeRegister.TimeSpanScalar = TimeSpanScalar

TypeRegister.DateScalar = DateScalar
TypeRegister.DateSpanScalar = DateSpanScalar
