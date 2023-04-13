Work with Dates and Times
=========================

In Riptable, there are three fundamental date and time classes:

-  ``rt.Date``, used for date information with no time attached to it.
-  ``rt.DateTimeNano``, used for data with both date and time
   information (including time zone), to nanosecond precision.
-  ``rt.TimeSpan``, used for “time since midnight data,” with no date
   information attached.

Here, we’ll cover how to create date and time objects, how to extract
data from these objects, how to use date and time arithmetic to build
useful date and time representations, and how to reformat date and time
information for display.

``Date`` Objects
----------------

A Date object stores an array of dates with no time data attached. You
can create Date arrays from strings, integer date values, or Matlab
ordinal dates. For Matlab details, see `Matlab Dates and Times <https://www.mathworks.com/help/matlab/date-and-time-operations.html>`__. 

Creating Date arrays from strings is fairly common. If your string dates are in YYYYMMDD format, you can simply pass the
list of strings to ``rt.Date()``::

    >>> rt.Date(['20210101', '20210519', '20220308'])
    Date(['2021-01-01', '2021-05-19', '2022-03-08'])

If your string dates are in another format, you can tell ``rt.Date()``
what to expect using Python ``strptime`` format code::

    >>> rt.Date(['12/31/19', '6/30/19', '02/21/19'], format='%m/%d/%y')
    Date(['2019-12-31', '2019-06-30', '2019-02-21'])

For a list of format codes and ``strptime`` implementation details, see `Python’s
'strftime' cheatsheet <https://strftime.org/>`__. The formatting codes are the same for ``strftime`` and ``strptime``. 

Note: Under the hood, dates are stored as integers – specifically, as
the number of days since the Unix epoch, 01-01-1970::

    >>> date_arr = rt.Date(['19700102', '19700103', '19700212'])
    >>> date_arr._fa
    FastArray([ 1,  2, 42])

Dates have various properties (a.k.a. attributes) that give you
information about a Date.

Let’s create a Dataset with a column of Dates, then use Date properties
to extract information into new columns::

    >>> ds = rt.Dataset()
    >>> # Generate a range of dates, spaced 15 days apart
    >>> ds.Dates = rt.Date.range('2019-01-01', '2019-02-30', step=15) 
    >>> # Some useful Date properties
    >>> ds.Year = ds.Dates.year
    >>> ds.Month = ds.Dates.month  # 1=Jan, 12=Dec
    >>> ds.Day_of_Month = ds.Dates.day_of_month
    >>> ds.Day_of_Week = ds.Dates.day_of_week  # 0=Mon, 6=Sun
    >>> ds.Day_of_Year = ds.Dates.day_of_year
    >>> ds
    #        Dates    Year   Month   Day_of_Month   Day_of_Week   Day_of_Year
    -   ----------   -----   -----   ------------   -----------   -----------
    0   2019-01-01   2,019       1              1             1             1
    1   2019-01-16   2,019       1             16             2            16
    2   2019-01-31   2,019       1             31             3            31
    3   2019-02-15   2,019       2             15             4            46

The following two properties are particularly useful when you want to
group data by month or week. We’ll see some examples when we talk about
Categoricals and Accums::

    >>> ds.Start_of_Month = ds.Dates.start_of_month
    >>> ds.Start_of_Week = ds.Dates.start_of_week  # Returns the date of the previous Monday
    >>> ds
    #        Dates    Year   Month   Day_of_Month   Day_of_Week   Day_of_Year   Start_of_Month   Start_of_Week
    -   ----------   -----   -----   ------------   -----------   -----------   --------------   -------------
    0   2019-01-01   2,019       1              1             1             1       2019-01-01      2018-12-31
    1   2019-01-16   2,019       1             16             2            16       2019-01-01      2019-01-14
    2   2019-01-31   2,019       1             31             3            31       2019-01-01      2019-01-28
    3   2019-02-15   2,019       2             15             4            46       2019-02-01      2019-02-11


We used Python’s ``strptime`` format code above to tell ``rt.Date()``
how to parse our data. Riptable date and time objects can also use the
``strftime()`` method to format data for display::

    >>> ds.MonthYear = ds.Dates.strftime('%b%y')
    >>> ds.col_filter(['Dates', 'MonthYear'])
    #        Dates   MonthYear
    -   ----------   ---------
    0   2019-01-01   Jan19    
    1   2019-01-16   Jan19    
    2   2019-01-31   Jan19    
    3   2019-02-15   Feb19  

You can do some arithmetic with date and time objects. For example, we
can get the number of days between two dates by subtracting one date
from another::

    >>> date_span = ds.Dates.max() - ds.Dates.min()
    >>> date_span
    DateSpan(['45 days'])

This returns a DateSpan object, which is a way to represent the delta,
or duration, between two dates. You can convert it to an integer if you
prefer::

    >>> date_span.astype(int)
    FastArray([45])

If you add a DateSpan to a Date, you get a Date::

    >>> ds.Dates.min() + date_span
    Date(['2019-02-15'])

Subtracting an array of dates from an array of dates gives you an array
of DateSpans. The two Date arrays must be the same length::

    >>> ds.DateDiff = ds.Dates - ds.Start_of_Month
    >>> ds.col_filter(['Dates', 'Start_of_Month', 'DateDiff'])
    #        Dates   Start_of_Month   DateDiff
    -   ----------   --------------   --------
    0   2019-01-01       2019-01-01     0 days
    1   2019-01-16       2019-01-01    15 days
    2   2019-01-31       2019-01-01    30 days
    3   2019-02-15       2019-02-01    14 days


Or you can subtract one Date from every record in a Date array::

    >>> ds.Dates2 = ds.Dates - rt.Date('20190102')
    >>> ds.col_filter(['Dates', 'Dates2'])
    #        Dates    Dates2
    -   ----------   -------
    0   2019-01-01   -1 days
    1   2019-01-16   14 days
    2   2019-01-31   29 days
    3   2019-02-15   44 days

``DateTimeNano`` Objects
------------------------

A ``DateTimeNano`` object stores data that has both date and time
information, with the time specified to nanosecond precision.

Like ``Date`` objects, ``DateTimeNano`` objects can be created from
strings. Strings are common when the data is from, say, a CSV file.

Unlike ``Date`` objects, ``DateTimeNano``\ s are time-zone-aware. When
you create a ``DateTimeNano``, you need to specify the time zone of
origin with the ``from_tz`` argument. Since Riptable is mainly used for
financial market data, its time zone options are limited to NYC, DUBLIN,
and (as of Riptable 1.3.6) Australia/Sydney, plus GMT and UTC (which is
an alias for GMT).

(If you’re wondering why ‘Australia/Sydney’ isn’t abbreviated, it’s
because Riptable uses the standard time zone name from the `tz
database <https://en.wikipedia.org/wiki/Tz_database>`__. In the future,
Riptable will support only the `standard
names <https://en.wikipedia.org/wiki/List_of_tz_database_time_zones>`__
in the tz database.)

::

    >>> rt.DateTimeNano(['20210101 09:31:15', '20210519 05:21:17'], from_tz='GMT')
    DateTimeNano(['20210101 04:31:15.000000000', '20210519 01:21:17.000000000'], to_tz='NYC')

Notice that the ``DateTimeNano`` is returned with ``to_tz='NYC'``. This
is the time zone the data is displayed in; NYC is the default. You can
change the display time zone when you create the ``DateTimeNano`` by
using ``to_tz``::

    >>> time_arr = rt.DateTimeNano(['20210101 09:31:15', '20210519 05:21:17'], 
    ...                            from_tz='GMT', to_tz='GMT')
    >>> time_arr
    DateTimeNano(['20210101 09:31:15.000000000', '20210519 05:21:17.000000000'], to_tz='GMT')

And as with Dates, you can specify the format of your string data::

    >>> rt.DateTimeNano(['12/31/19', '6/30/19'], format='%m/%d/%y', from_tz='NYC')
    DateTimeNano(['20191231 00:00:00.000000000', '20190630 00:00:00.000000000'], to_tz='NYC')

When you’re dealing with large amounts of data, it’s more typical to get
dates and times that are represented as nanoseconds since the Unix epoch
(01-01-1970). In fact, that is how ``DateTimeNano`` objects are stored
(it’s much more efficient to store numbers than strings)::

    >>> time_arr._fa
    FastArray([1609493475000000000, 1621401677000000000], dtype=int64)

If your data comes in this way, ``rt.DateTimeNano()`` can convert it
easily. Just supply the time zone::

    >>> rt.DateTimeNano([1609511475000000000, 1621416077000000000], from_tz='NYC')
    DateTimeNano(['20210101 14:31:15.000000000', '20210519 09:21:17.000000000'], to_tz='NYC')

To split the date off a DateTimeNano, use ``rt.Date()``::

    >>> rt.Date(time_arr)
    Date(['2021-01-01', '2021-05-19'])

To get the time, use ``time_since_midnight()``::

    >>> time_arr.time_since_midnight()
    TimeSpan(['09:31:15.000000000', '05:21:17.000000000'])

Note that the result is a TimeSpan. We’ll look at these more in the next
section.

You can also get the time in nanoseconds since midnight::

    >>> time_arr.nanos_since_midnight()
    FastArray([34275000000000, 19277000000000], dtype=int64)

``DateTimeNano``\ s can be reformatted for display using ``strftime()``::

    >>> time_arr.strftime('%m/%d/%y %H:%M:%S')  # Date and time
    array(['01/01/21 09:31:15', '05/19/21 05:21:17'], dtype=object)

Just the time::

    >>> time_arr.strftime('%H:%M:%S')
    array(['09:31:15', '05:21:17'], dtype=object)

Some arithmetic::

    >>> # Create two DateTimeNano arrays
    >>> time_arr1 = rt.DateTimeNano(['20220101 12:00:00', '20220301 13:00:00'], from_tz='NYC', to_tz='NYC')
    >>> time_arr2 = rt.DateTimeNano(['20190101 11:00:00', '20190301 11:30:00'], from_tz='NYC', to_tz='NYC')

``DateTimeNano`` - ``DateTimeNano`` = ``TimeSpan``

:: 

    >>> timespan1 = time_arr1 - time_arr2
    >>> timespan1
    TimeSpan(['1096d 01:00:00.000000000', '1096d 01:30:00.000000000'])

``DateTimeNano`` + ``TimeSpan`` = ``DateTimeNano``

::

    >>> dtn1 = time_arr1 + timespan1
    >>> dtn1
    DateTimeNano(['20250101 13:00:00.000000000', '20250301 14:30:00.000000000'], to_tz='NYC')

``DateTimeNano`` - ``TimeSpan`` = ``DateTimeNano``

::

    >>> dtn2 = dtn1 - timespan1   
    >>> dtn2
    DateTimeNano(['20220101 12:00:00.000000000', '20220301 13:00:00.000000000'], to_tz='NYC')

``TimeSpan`` Objects
--------------------

You saw above how a ``TimeSpan`` represents a duration of time between
two ``DateTimeNano``\ s. You can also think of it as a representation of
a time of day.

Recall that you can split a ``TimeSpan`` off a ``DateTimeNano`` using
``time_since_midnight()``. Just keep in mind that a ``TimeSpan`` by
itself has no absolute reference to Midnight of any day in particular.

As an example, let’s say you want to find out which trades were made
before a certain time of day (on any day). If your data has
``DateTimeNano``\ s, you can split off the ``TimeSpan``, then filter for
the times you’re interested in::

    >>> rng = np.random.default_rng(seed=42)
    >>> ds = rt.Dataset()
    >>> N = 100  # Length of the Dataset
    >>> ds.Symbol = rt.FA(rng.choice(['AAPL', 'AMZN', 'TSLA', 'SPY', 'GME'], N))
    >>> ds.Size = rng.random(N) * 100
    >>> # Create a column of randomly generated DateTimeNanos
    >>> ds.TradeDateTime = rt.DateTimeNano.random(N)
    >>> ds.TradeTime = ds.TradeDateTime.time_since_midnight()
    >>> ds
      #   Symbol    Size                 TradeDateTime            TradeTime
    ---   ------   -----   ---------------------------   ------------------
      0   AAPL     19.99   20190614 13:07:21.352420597   13:07:21.352420597
      1   SPY       0.74   19970809 19:34:40.178693393   19:34:40.178693393
      2   SPY      78.69   19861130 20:06:31.775222495   20:06:31.775222495
      3   TSLA     66.49   20081111 04:15:24.079385833   04:15:24.079385833
      4   TSLA     70.52   20190419 06:21:31.197889103   06:21:31.197889103
      5   GME      78.07   19861112 05:20:14.239289462   05:20:14.239289462
      6   AAPL     45.89   20110329 20:55:07.198530171   20:55:07.198530171
      7   SPY      56.87   19780303 03:19:32.676920289   03:19:32.676920289
      8   AMZN     13.98   19930305 22:34:02.767331408   22:34:02.767331408
      9   AAPL     11.45   19840723 04:08:10.118105881   04:08:10.118105881
     10   TSLA     66.84   19940814 03:08:03.730164619   03:08:03.730164619
     11   GME      47.11   19730612 22:33:46.871406555   22:33:46.871406555
     12   SPY      56.52   19840118 14:01:10.111423986   14:01:10.111423986
     13   SPY      76.50   19740813 15:26:44.457459450   15:26:44.457459450
     14   SPY      63.47   20050106 18:13:57.982489010   18:13:57.982489010
    ...   ...        ...                           ...                  ...
     85   SPY       2.28   19930706 00:24:05.337093375   00:24:05.337093375
     86   AAPL     95.86   20140823 11:35:14.816318096   11:35:14.816318096
     87   AMZN     48.23   20070929 22:49:10.456157805   22:49:10.456157805
     88   SPY      78.27   19930616 20:30:27.490477141   20:30:27.490477141
     89   GME       8.27   19860626 07:48:16.756213658   07:48:16.756213658
     90   TSLA     48.67   20060824 19:29:19.583638324   19:29:19.583638324
     91   GME      49.07   19751026 20:29:32.616225869   20:29:32.616225869
     92   GME      93.78   19911222 14:53:30.879285646   14:53:30.879285646
     93   AMZN     57.17   19970715 20:26:36.179803660   20:26:36.179803660
     94   GME      47.35   19961214 10:26:16.609357094   10:26:16.609357094
     95   AMZN     26.70   19830606 14:02:30.699183111   14:02:30.699183111
     96   AMZN     33.16   19821114 05:56:13.504071773   05:56:13.504071773
     97   SPY      52.07   19740606 03:47:03.798827481   03:47:03.798827481
     98   SPY      43.89   19881226 22:19:55.209671459   22:19:55.209671459
     99   AAPL      2.16   19840720 11:51:26.734190049   11:51:26.734190049
    
If we want to find the trades that happened before 10:00 a.m., we need a
TimeSpan that represents 10:00 a.m. Then we can can compare our
TradeTimes against it.

To construct a TimeSpan from scratch, you can pass time strings in
``%H:%M:%S`` format::

    >>> rt.TimeSpan(['09:00', '10:45', '02:30', '15:00', '23:10'])
    TimeSpan(['09:00:00.000000000', '10:45:00.000000000', '02:30:00.000000000', '15:00:00.000000000', '23:10:00.000000000'])

Or from an array of numerics, along with a unit, like hours::

    >>> rt.TimeSpan([9, 10, 12, 14, 18], unit='h')
    TimeSpan(['09:00:00.000000000', '10:00:00.000000000', '12:00:00.000000000', '14:00:00.000000000', '18:00:00.000000000'])

For our purposes, this will do::

    >>> tenAM = rt.TimeSpan(10, unit='h')
    >>> tenAM
    TimeSpan(['10:00:00.000000000'])

Now we can compare the TradeTime values against it. We’ll put the
results of the comparison into a column so we can spot check them::

    >>> ds.TradesBefore10am = (ds.TradeTime < tenAM)
    >>> ds
      #   Symbol    Size                 TradeDateTime            TradeTime   TradesBefore10am
    ---   ------   -----   ---------------------------   ------------------   ----------------
      0   AAPL     19.99   20190614 13:07:21.352420597   13:07:21.352420597              False
      1   SPY       0.74   19970809 19:34:40.178693393   19:34:40.178693393              False
      2   SPY      78.69   19861130 20:06:31.775222495   20:06:31.775222495              False
      3   TSLA     66.49   20081111 04:15:24.079385833   04:15:24.079385833               True
      4   TSLA     70.52   20190419 06:21:31.197889103   06:21:31.197889103               True
      5   GME      78.07   19861112 05:20:14.239289462   05:20:14.239289462               True
      6   AAPL     45.89   20110329 20:55:07.198530171   20:55:07.198530171              False
      7   SPY      56.87   19780303 03:19:32.676920289   03:19:32.676920289               True
      8   AMZN     13.98   19930305 22:34:02.767331408   22:34:02.767331408              False
      9   AAPL     11.45   19840723 04:08:10.118105881   04:08:10.118105881               True
     10   TSLA     66.84   19940814 03:08:03.730164619   03:08:03.730164619               True
     11   GME      47.11   19730612 22:33:46.871406555   22:33:46.871406555              False
     12   SPY      56.52   19840118 14:01:10.111423986   14:01:10.111423986              False
     13   SPY      76.50   19740813 15:26:44.457459450   15:26:44.457459450              False
     14   SPY      63.47   20050106 18:13:57.982489010   18:13:57.982489010              False
    ...   ...        ...                           ...                  ...                ...
     85   SPY       2.28   19930706 00:24:05.337093375   00:24:05.337093375               True
     86   AAPL     95.86   20140823 11:35:14.816318096   11:35:14.816318096              False
     87   AMZN     48.23   20070929 22:49:10.456157805   22:49:10.456157805              False
     88   SPY      78.27   19930616 20:30:27.490477141   20:30:27.490477141              False
     89   GME       8.27   19860626 07:48:16.756213658   07:48:16.756213658               True
     90   TSLA     48.67   20060824 19:29:19.583638324   19:29:19.583638324              False
     91   GME      49.07   19751026 20:29:32.616225869   20:29:32.616225869              False
     92   GME      93.78   19911222 14:53:30.879285646   14:53:30.879285646              False
     93   AMZN     57.17   19970715 20:26:36.179803660   20:26:36.179803660              False
     94   GME      47.35   19961214 10:26:16.609357094   10:26:16.609357094              False
     95   AMZN     26.70   19830606 14:02:30.699183111   14:02:30.699183111              False
     96   AMZN     33.16   19821114 05:56:13.504071773   05:56:13.504071773               True
     97   SPY      52.07   19740606 03:47:03.798827481   03:47:03.798827481               True
     98   SPY      43.89   19881226 22:19:55.209671459   22:19:55.209671459              False
     99   AAPL      2.16   19840720 11:51:26.734190049   11:51:26.734190049              False

And of course, we can use the Boolean array to filter the Dataset::

    >>> ds.filter(ds.TradesBefore10am)
      #   Symbol    Size                 TradeDateTime            TradeTime   TradesBefore10am
    ---   ------   -----   ---------------------------   ------------------   ----------------
      0   TSLA     66.49   20081111 04:15:24.079385833   04:15:24.079385833               True
      1   TSLA     70.52   20190419 06:21:31.197889103   06:21:31.197889103               True
      2   GME      78.07   19861112 05:20:14.239289462   05:20:14.239289462               True
      3   SPY      56.87   19780303 03:19:32.676920289   03:19:32.676920289               True
      4   AAPL     11.45   19840723 04:08:10.118105881   04:08:10.118105881               True
      5   TSLA     66.84   19940814 03:08:03.730164619   03:08:03.730164619               True
      6   SPY      55.36   20010615 00:14:45.718385740   00:14:45.718385740               True
      7   GME      23.39   19751116 06:06:50.777397710   06:06:50.777397710               True
      8   TSLA     29.36   19920606 01:44:12.762930709   01:44:12.762930709               True
      9   GME      66.19   20150907 07:56:58.291001076   07:56:58.291001076               True
     10   GME      46.19   19771105 07:18:54.592658284   07:18:54.592658284               True
     11   SPY      50.10   19980211 08:39:58.366644251   08:39:58.366644251               True
     12   AAPL     15.23   19840811 03:03:32.341618015   03:03:32.341618015               True
     13   AMZN     38.10   19730321 08:49:53.629495873   08:49:53.629495873               True
     14   AAPL     30.15   20091103 04:56:46.941815206   04:56:46.941815206               True
    ...   ...        ...                           ...                  ...                ...
     19   GME      75.85   19870605 00:16:50.617990376   00:16:50.617990376               True
     20   AAPL     43.21   19880730 01:20:25.325405869   01:20:25.325405869               True
     21   AAPL     64.98   19750705 03:28:57.626851689   03:28:57.626851689               True
     22   AAPL     41.58   19900712 07:39:20.866244793   07:39:20.866244793               True
     23   SPY       4.16   20090512 03:17:20.112309966   03:17:20.112309966               True
     24   AMZN     32.99   20010910 02:18:44.384567415   02:18:44.384567415               True
     25   AMZN     14.45   19901004 00:53:54.407173923   00:53:54.407173923               True
     26   TSLA     10.34   19961220 04:54:14.777983172   04:54:14.777983172               True
     27   SPY      58.76   20070922 04:55:14.156355503   04:55:14.156355503               True
     28   TSLA     92.51   19851209 01:52:03.199471749   01:52:03.199471749               True
     29   GME      34.69   20160202 09:57:41.083925341   09:57:41.083925341               True
     30   SPY       2.28   19930706 00:24:05.337093375   00:24:05.337093375               True
     31   GME       8.27   19860626 07:48:16.756213658   07:48:16.756213658               True
     32   AMZN     33.16   19821114 05:56:13.504071773   05:56:13.504071773               True
     33   SPY      52.07   19740606 03:47:03.798827481   03:47:03.798827481               True

If we only want to see certain columns of the Dataset, we can combine
the filter with slicing::

    >>> ds[ds.TradesBefore10am, ['Symbol', 'Size']]
      #   Symbol    Size
    ---   ------   -----
      0   TSLA     66.49
      1   TSLA     70.52
      2   GME      78.07
      3   SPY      56.87
      4   AAPL     11.45
      5   TSLA     66.84
      6   SPY      55.36
      7   GME      23.39
      8   TSLA     29.36
      9   GME      66.19
     10   GME      46.19
     11   SPY      50.10
     12   AAPL     15.23
     13   AMZN     38.10
     14   AAPL     30.15
    ...   ...        ...
     19   GME      75.85
     20   AAPL     43.21
     21   AAPL     64.98
     22   AAPL     41.58
     23   SPY       4.16
     24   AMZN     32.99
     25   AMZN     14.45
     26   TSLA     10.34
     27   SPY      58.76
     28   TSLA     92.51
     29   GME      34.69
     30   SPY       2.28
     31   GME       8.27
     32   AMZN     33.16
     33   SPY      52.07

Or if we just want the total size of AAPL trades before 10am::

    >>> aapl10 = (ds.Symbol == 'AAPL') & (ds.TradesBefore10am)
    >>> ds.Size.nansum(filter = aapl10)
    274.92741837733035

Other Useful things to Do with TimeSpans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can compare two ``DateTimeNano`` columns to find times that are close
together – for example, those less than 10ms apart.

To illustrate this, we’ll create some randomly generated small
``TimeSpan``\ s to add to our column of ``DateTimeNano``\ s::

    >>> # Create TimeSpans from 1 millisecond to 19 milliseconds
    >>> some_ms = rt.TimeSpan(rng.integers(low=1, high=20, size=N), 'ms') 
    >>> # Offset the TimeSpans in our original DateTimeNano 
    >>> ds.TradeDateTime2 = ds.TradeDateTime + some_ms
    >>> ds.col_filter(['Symbol', 'TradeDateTime', 'TradeDateTime2']).head()
     #   Symbol                 TradeDateTime                TradeDateTime2
    --   ------   ---------------------------   ---------------------------
     0   AAPL     20100614 01:47:46.306210225   20100614 01:47:46.313210225
     1   SPY      20131004 12:02:28.251037257   20131004 12:02:28.267037257
     2   SPY      19721212 00:54:12.641763127   19721212 00:54:12.642763127
     3   TSLA     19720118 19:33:36.911790260   19720118 19:33:36.929790260
     4   TSLA     19750331 15:04:15.847968984   19750331 15:04:15.858968984
     5   GME      19740912 18:18:46.660464416   19740912 18:18:46.663464416
     6   AAPL     19820906 09:31:02.911852383   19820906 09:31:02.917852383
     7   SPY      19900810 10:42:02.603793160   19900810 10:42:02.614793160
     8   AMZN     19870318 06:54:30.389382275   19870318 06:54:30.395382275
     9   AAPL     20031029 09:53:06.898676308   20031029 09:53:06.901676308
    10   TSLA     20160319 00:33:40.035581577   20160319 00:33:40.048581577
    11   GME      19801024 01:38:46.310440408   19801024 01:38:46.323440408
    12   SPY      19791105 17:08:46.460502123   19791105 17:08:46.463502123
    13   SPY      20110304 07:11:03.437823831   20110304 07:11:03.443823831
    14   SPY      20140303 01:58:10.917868743   20140303 01:58:10.922868743
    15   SPY      19990514 19:33:06.261903491   19990514 19:33:06.274903491
    16   TSLA     19840808 16:34:56.776803922   19840808 16:34:56.790803922
    17   AAPL     19711222 11:39:46.898769893   19711222 11:39:46.912769893
    18   GME      20090605 13:23:02.120390523   20090605 13:23:02.138390523
    19   TSLA     19900227 19:36:40.067192555   19900227 19:36:40.082192555

Now we can find the trades that occurred within 10ms of each other, and
again put the results into a new column for a spot check.

    >>> ds.Within10ms = (abs(ds.TradeDateTime.time_since_midnight() 
    ...                  - ds.TradeDateTime2.time_since_midnight())) < rt.TimeSpan(10, 'ms')
    >>> ds.col_filter(['Symbol', 'TradeDateTime', 'TradeDateTime2', 'Within10ms']).head()
     #   Symbol                 TradeDateTime                TradeDateTime2   Within10ms
    --   ------   ---------------------------   ---------------------------   ----------
     0   AAPL     19771006 11:46:39.512132962   19771006 11:46:39.519132962         True
     1   SPY      20000321 15:00:25.630646023   20000321 15:00:25.646646023        False
     2   SPY      19720130 05:36:37.195744004   19720130 05:36:37.196744004         True
     3   TSLA     19960902 00:45:11.619930786   19960902 00:45:11.637930786        False
     4   TSLA     19901216 15:52:53.935112408   19901216 15:52:53.946112408        False
     5   GME      19900910 22:20:09.846455444   19900910 22:20:09.849455444         True
     6   AAPL     20000825 20:59:19.248822244   20000825 20:59:19.254822244         True
     7   SPY      19740216 18:32:16.051989951   19740216 18:32:16.062989951        False
     8   AMZN     19951222 07:27:43.668483372   19951222 07:27:43.674483372         True
     9   AAPL     20180708 11:19:48.016609690   20180708 11:19:48.019609690         True
    10   TSLA     20110429 21:11:34.789939106   20110429 21:11:34.802939106        False
    11   GME      19921202 20:27:45.957970537   19921202 20:27:45.970970537        False
    12   SPY      19980801 10:04:29.793513895   19980801 10:04:29.796513895         True
    13   SPY      19970217 08:00:06.615346852   19970217 08:00:06.621346852         True
    14   SPY      20060915 20:18:28.369763536   20060915 20:18:28.374763536         True
    15   SPY      19991220 16:10:56.841720714   19991220 16:10:56.854720714        False
    16   TSLA     19730131 01:08:43.413049524   19730131 01:08:43.427049524        False
    17   AAPL     20040518 15:53:50.561136824   20040518 15:53:50.575136824        False
    18   GME      19710809 14:51:55.347200052   19710809 14:51:55.365200052        False
    19   TSLA     19980613 01:40:56.278221632   19980613 01:40:56.293221632        False

And again we can use the result as a mask array::

    >>> ds[ds.Within10ms, ['Symbol', 'Size']]
      #   Symbol    Size
    ---   ------   -----
      0   AAPL     19.99
      1   SPY      78.69
      2   GME      78.07
      3   AAPL     45.89
      4   AMZN     13.98
      5   AAPL     11.45
      6   SPY      56.52
      7   SPY      76.50
      8   SPY      63.47
      9   TSLA     21.46
     10   AMZN     40.85
     11   SPY      28.14
     12   TSLA     29.36
     13   GME      66.19
     14   TSLA     55.70
    ...   ...        ...
     37   TSLA     49.40
     38   TSLA     10.34
     39   SPY      58.76
     40   GME      17.06
     41   GME      34.69
     42   SPY      59.09
     43   SPY       2.28
     44   AAPL     95.86
     45   GME       8.27
     46   GME      49.07
     47   GME      93.78
     48   AMZN     33.16
     49   SPY      52.07
     50   SPY      43.89
     51   AAPL      2.16

A common situation is having dates as date strings and times in nanos
since midnight. You can use some arithmetic to build a DateTimeNano:
``Date`` + ``TimeSpan`` = ``DateTimeNano``::

    >>> ds = rt.Dataset({
    ...     'Date': ['20111111', '20200202', '20220222'],
    ...     'Time': [44_275_000_000_000, 39_287_000_000_000, 55_705_000_000_000]
    ...     })
    >>> # Convert the date strings to rt.Date objects
    >>> ds.Date = rt.Date(ds.Date)
    >>> # Convert the times to rt.TimeSpan objects
    >>> ds.Time = rt.TimeSpan(ds.Time)
    >>> ds
    #         Date                 Time
    -   ----------   ------------------
    0   2011-11-11   12:17:55.000000000
    1   2020-02-02   10:54:47.000000000
    2   2022-02-22   15:28:25.000000000

At this point, you might want to simply add ``ds.Date`` and ``ds.Time``
to get a ``DateTimeNano``::

    >>> ds.DateTime = ds.Date + ds.Time
    >>> ds
    #         Date                 Time                      DateTime
    -   ----------   ------------------   ---------------------------
    0   2011-11-11   12:17:55.000000000   20111111 12:17:55.000000000
    1   2020-02-02   10:54:47.000000000   20200202 10:54:47.000000000
    2   2022-02-22   15:28:25.000000000   20220222 15:28:25.000000000

And that seems to work. However, remember that ``DateTimeNano``\ s need
to have a time zone. Here, GMT was assumed::

    >>> ds.DateTime
    DateTimeNano(['20111111 12:17:55.000000000', '20200202 10:54:47.000000000', '20220222 15:28:25.000000000'], to_tz='GMT')

Specify your desired time zone so you don’t end up with unexpected
results down the line::

    >>> ds.DateTime2 = rt.DateTimeNano((ds.Date + ds.Time), from_tz='NYC')
    >>> ds.DateTime2
    DateTimeNano(['20111111 12:17:55.000000000', '20200202 10:54:47.000000000', '20220222 15:28:25.000000000'], to_tz='NYC')

Warning: Given that ``TimeSpan + Date = DateTimeNano``, and also that
you can use ``rt.Date(my_dtn)`` to get a ``Date`` from a
``DateTimeNano``, you might reasonably think you can get the
``TimeSpan`` from a ``DateTimeNano`` using ``rt.TimeSpan(my_dtn)``.

However, that result includes the number of days since January 1, 1970.
To get the ``TimeSpan`` from a ``DateTimeNano``, use
``time_since_midnight()`` instead.

+----------------------------------------+
| **Datetime Arithmetic**                |
+========================================+
| Date + Date = TypeError                |
+----------------------------------------+
| Date + DateTimeNano = TypeError        |
+----------------------------------------+
| Date + DateSpan = Date                 |
+----------------------------------------+
| Date + TimeSpan = DateTimeNano         |
+----------------------------------------+
|                                        |
+----------------------------------------+
| Date - Date = DateSpan                 |
+----------------------------------------+
| Date - DateSpan = Date                 |
+----------------------------------------+
| Date - DateTimeNano = TimeSpan         |
+----------------------------------------+
| Date - TimeSpan = DateTimeNano         |
+----------------------------------------+
|                                        |
+----------------------------------------+
| DateTimeNano - DateTimeNano = TimeSpan |
+----------------------------------------+
| DateTimeNano - TimeSpan = DateTimeNano |
+----------------------------------------+
| DateTimeNano + TimeSpan = DateTimeNano |
+----------------------------------------+
|                                        |
+----------------------------------------+
| TimeSpan - TimeSpan = TimeSpan         |
+----------------------------------------+
| TimeSpan + TimeSpan = TimeSpan         |
+----------------------------------------+

Next, we’ll look at Riptable’s vehicle for group operations: `Perform
Group Operations with Categoricals <tutorial_categoricals.rst>`__.

--------------

Questions or comments about this guide? Email
RiptableDocumentation@sig.com.
