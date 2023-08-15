Merge Datasets
==============

Merging gives you more flexibility to bring data from different Datasets
together.

A merge operation connects rows in Datasets using a “key” column that
the Datasets have in common.

Riptable’s two main Dataset merge functions are ``merge_lookup()`` and
``merge_asof()``. Generally speaking, ``merge_lookup()`` aligns data
based on identical keys, while ``merge_asof()`` aligns data based on the
nearest key.

For more general merges, ``merge2()`` does database-style left, right,
inner, and outer joins.

``merge_lookup()``
------------------

Let’s start with ``merge_lookup()``. It’s common to have one Dataset
that has most of the information you need, and another, usually smaller
Dataset that has information you want to add to the first Dataset to
enrich it.

Here we’ll create a larger Dataset with symbols and size values, and a
smaller Dataset that has symbols associated with trader names. We’ll use
the shared Symbol column as the key to add the trader info to the larger
Dataset::

    >>> rng = np.random.default_rng(seed=42)
    >>> N = 25
    >>> # Larger Dataset
    >>> ds = rt.Dataset({'Symbol': rng.choice(['GME', 'AMZN', 'TSLA', 'SPY'], N),
    ...                  'Size': rng.integers(1, 1000, N),})
    >>> # Smaller Dataset, with data used to enrich the larger Dataset
    >>> ds_symbol_trader = rt.Dataset({'Symbol': ['GME', 'TSLA', 'SPY', 'AMZN'],
    ...                                'Trader': ['Nate', 'Elon', 'Josh', 'Dan']})
    >>> ds.head()
     #   Symbol   Size
    --   ------   ----
     0   GME       644
     1   SPY       403
     2   TSLA      822
     3   AMZN      545
     4   AMZN      443
     5   SPY       451
     6   GME       228
     7   TSLA       93
     8   GME       555
     9   GME       888
    10   TSLA       64
    11   SPY       858
    12   TSLA      827
    13   SPY       277
    14   TSLA      632
    15   SPY       166
    16   TSLA      758
    17   GME       700
    18   SPY       355
    19   AMZN       68

    >>> ds_symbol_trader
    #   Symbol   Trader
    -   ------   ------
    0   GME      Nate  
    1   TSLA     Elon  
    2   SPY      Josh  
    3   AMZN     Dan   

``merge_lookup()`` with Key Columns That Have the Same Name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we’ll use ``merge_lookup()`` to add the trader information to the
larger Dataset. ``merge_lookup()`` will align the data based on exact
matches in the shared Symbol column.

A note about terms: When you merge two Datasets, the Dataset you’re
merging data into is the *left Dataset*; the one you’re getting data
from is the *right Dataset*.

Here, we call ``merge_lookup()`` on our left Dataset, ``ds``. We pass it
the name of the right Dataset, and tell it what column to use as the
key::

    >>> ds.merge_lookup(ds_symbol_trader, on='Symbol')
     #   Symbol   Size   Trader
    --   ------   ----   ------
     0   GME       644   Nate  
     1   SPY       403   Josh  
     2   TSLA      822   Elon  
     3   AMZN      545   Dan   
     4   AMZN      443   Dan   
     5   SPY       451   Josh  
     6   GME       228   Nate  
     7   TSLA       93   Elon  
     8   GME       555   Nate  
     9   GME       888   Nate  
    10   TSLA       64   Elon  
    11   SPY       858   Josh  
    12   TSLA      827   Elon  
    13   SPY       277   Josh  
    14   TSLA      632   Elon  
    15   SPY       166   Josh  
    16   TSLA      758   Elon  
    17   GME       700   Nate  
    18   SPY       355   Josh  
    19   AMZN       68   Dan   
    20   TSLA      970   Elon  
    21   AMZN      446   Dan   
    22   GME       893   Nate  
    23   SPY       678   Josh  
    24   SPY       778   Josh  

The left Dataset now has the trader information, correctly aligned.

You can also use the following syntax, passing ``merge_lookup()`` the
names of the left and right Datasets, along with the key::

    >>> rt.merge_lookup(ds, ds_symbol_trader, on='Symbol')
     #   Symbol   Size   Trader
    --   ------   ----   ------
     0   GME       644   Nate  
     1   SPY       403   Josh  
     2   TSLA      822   Elon  
     3   AMZN      545   Dan   
     4   AMZN      443   Dan   
     5   SPY       451   Josh  
     6   GME       228   Nate  
     7   TSLA       93   Elon  
     8   GME       555   Nate  
     9   GME       888   Nate  
    10   TSLA       64   Elon  
    11   SPY       858   Josh  
    12   TSLA      827   Elon  
    13   SPY       277   Josh  
    14   TSLA      632   Elon  
    15   SPY       166   Josh  
    16   TSLA      758   Elon  
    17   GME       700   Nate  
    18   SPY       355   Josh  
    19   AMZN       68   Dan   
    20   TSLA      970   Elon  
    21   AMZN      446   Dan   
    22   GME       893   Nate  
    23   SPY       678   Josh  
    24   SPY       778   Josh  

``merge_lookup`` with Key Columns That Have Different Names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the key column has a different name in each Dataset, just specify
each column name with ``left_on`` and ``right_on``::

    >>> # For illustrative purposes, rename the key column in the right Dataset.
    >>> ds_symbol_trader.col_rename('Symbol', 'UnderlyingSymbol')
    >>> ds.merge_lookup(ds_symbol_trader, left_on='Symbol', right_on='UnderlyingSymbol')
     #   Symbol   Size   UnderlyingSymbol   Trader
    --   ------   ----   ----------------   ------
     0   GME       644   GME                Nate  
     1   SPY       403   SPY                Josh  
     2   TSLA      822   TSLA               Elon  
     3   AMZN      545   AMZN               Dan   
     4   AMZN      443   AMZN               Dan   
     5   SPY       451   SPY                Josh  
     6   GME       228   GME                Nate  
     7   TSLA       93   TSLA               Elon  
     8   GME       555   GME                Nate  
     9   GME       888   GME                Nate  
    10   TSLA       64   TSLA               Elon  
    11   SPY       858   SPY                Josh  
    12   TSLA      827   TSLA               Elon  
    13   SPY       277   SPY                Josh  
    14   TSLA      632   TSLA               Elon  
    15   SPY       166   SPY                Josh  
    16   TSLA      758   TSLA               Elon  
    17   GME       700   GME                Nate  
    18   SPY       355   SPY                Josh  
    19   AMZN       68   AMZN               Dan   
    20   TSLA      970   TSLA               Elon  
    21   AMZN      446   AMZN               Dan   
    22   GME       893   GME                Nate  
    23   SPY       678   SPY                Josh  
    24   SPY       778   SPY                Josh  

Notice that when the key columns have different names, both are kept. If
you want keep only certain columns from the left or right Dataset, you
can specify them with ``columns_left`` or ``columns_right``::

    >>> ds.merge_lookup(ds_symbol_trader, left_on='Symbol', right_on='UnderlyingSymbol', 
    ...                 columns_right='Trader')
     #   Symbol   Size   Trader
    --   ------   ----   ------
     0   GME       644   Nate  
     1   SPY       403   Josh  
     2   TSLA      822   Elon  
     3   AMZN      545   Dan   
     4   AMZN      443   Dan   
     5   SPY       451   Josh  
     6   GME       228   Nate  
     7   TSLA       93   Elon  
     8   GME       555   Nate  
     9   GME       888   Nate  
    10   TSLA       64   Elon  
    11   SPY       858   Josh  
    12   TSLA      827   Elon  
    13   SPY       277   Josh  
    14   TSLA      632   Elon  
    15   SPY       166   Josh  
    16   TSLA      758   Elon  
    17   GME       700   Nate  
    18   SPY       355   Josh  
    19   AMZN       68   Dan   
    20   TSLA      970   Elon  
    21   AMZN      446   Dan   
    22   GME       893   Nate  
    23   SPY       678   Josh  
    24   SPY       778   Josh

Note: ``merge_lookup()`` Keeps Only the Keys in the Left Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One thing to note about ``merge_lookup()`` is that it keeps only the
keys are that are in the left Dataset (it’s equivalent to a SQL left
join). If there are keys in the right Dataset that aren’t in the left
Dataset, they’re discarded in the merged data::

    >>> # Create a right Dataset with an extra symbol key ('MSFT').
    >>> ds_symbol_trader2 = rt.Dataset({'Symbol': ['GME', 'TSLA', 'SPY', 'AMZN', 'MSFT'], 
    ...                                 'Trader': ['Nate', 'Elon', 'Josh', 'Dan', 'Lauren']})
    >>> # Change 'UnderlyingSymbol' back to 'Symbol' for simplicity.
    >>> ds_symbol_trader.col_rename('UnderlyingSymbol', 'Symbol')
    >>> ds.merge_lookup(ds_symbol_trader2, on='Symbol', columns_right='Trader')
     #   Symbol   Size   Trader
    --   ------   ----   ------
     0   GME       644   Nate  
     1   SPY       403   Josh  
     2   TSLA      822   Elon  
     3   AMZN      545   Dan   
     4   AMZN      443   Dan   
     5   SPY       451   Josh  
     6   GME       228   Nate  
     7   TSLA       93   Elon  
     8   GME       555   Nate  
     9   GME       888   Nate  
    10   TSLA       64   Elon  
    11   SPY       858   Josh  
    12   TSLA      827   Elon  
    13   SPY       277   Josh  
    14   TSLA      632   Elon  
    15   SPY       166   Josh  
    16   TSLA      758   Elon  
    17   GME       700   Nate  
    18   SPY       355   Josh  
    19   AMZN       68   Dan   
    20   TSLA      970   Elon  
    21   AMZN      446   Dan   
    22   GME       893   Nate  
    23   SPY       678   Josh  
    24   SPY       778   Josh  

``merge_lookup()`` with Overlapping Columns That Aren’t Keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As we saw above, if the two key columns have the same name in both
Datasets, only one is kept. For columns that aren’t used as keys, you’ll
get a name collision error when you try to merge::

    >>> # Add a Size column to the right Dataset
    >>> ds_symbol_trader.Size = rng.integers(1, 1000, 4)

    >>> try:
    ...     rt.merge_lookup(ds, ds_symbol_trader, on='Symbol')
    ... except ValueError as e:
    ...     print("ValueError:", e)
    ValueError: columns overlap but no suffix specified: {'Size'}

If you want to keep both columns, add a suffix to each column name to
disambiguate them::

    >>> rt.merge_lookup(ds, ds_symbol_trader, on='Symbol', suffixes=('_1', '_2'))
     #   Symbol   Size_1   Trader   Size_2
    --   ------   ------   ------   ------
     0   GME         644   Nate        760
     1   SPY         403   Josh        364
     2   TSLA        822   Elon        195
     3   AMZN        545   Dan         467
     4   AMZN        443   Dan         467
     5   SPY         451   Josh        364
     6   GME         228   Nate        760
     7   TSLA         93   Elon        195
     8   GME         555   Nate        760
     9   GME         888   Nate        760
    10   TSLA         64   Elon        195
    11   SPY         858   Josh        364
    12   TSLA        827   Elon        195
    13   SPY         277   Josh        364
    14   TSLA        632   Elon        195
    15   SPY         166   Josh        364
    16   TSLA        758   Elon        195
    17   GME         700   Nate        760
    18   SPY         355   Josh        364
    19   AMZN         68   Dan         467
    20   TSLA        970   Elon        195
    21   AMZN        446   Dan         467
    22   GME         893   Nate        760
    23   SPY         678   Josh        364
    24   SPY         778   Josh        364

``merge_lookup()`` with a Right Dataset That Has Duplicate Keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the right Dataset has more than one match for a unique key in the
left Dataset, you can specify whether to use the first or the last match
encountered in the right Dataset::

    >>> # Create a right Dataset with a second GME key, associated to Lauren
    >>> ds_symbol_trader3 = rt.Dataset({'Symbol': ['GME', 'TSLA', 'SPY', 'AMZN', 'GME'], 
    ...                                 'Trader': ['Nate', 'Elon', 'Josh', 'Dan', 'Lauren']})
    >>> ds_symbol_trader3
    #   Symbol   Trader
    -   ------   ------
    0   GME      Nate  
    1   TSLA     Elon  
    2   SPY      Josh  
    3   AMZN     Dan   
    4   GME      Lauren

We’ll keep the last match::

    >>> ds.merge_lookup(ds_symbol_trader3, on='Symbol', columns_right='Trader', keep='last')
     #   Symbol   Size   Trader
    --   ------   ----   ------
     0   GME       644   Lauren
     1   SPY       403   Josh  
     2   TSLA      822   Elon  
     3   AMZN      545   Dan   
     4   AMZN      443   Dan   
     5   SPY       451   Josh  
     6   GME       228   Lauren
     7   TSLA       93   Elon  
     8   GME       555   Lauren
     9   GME       888   Lauren
    10   TSLA       64   Elon  
    11   SPY       858   Josh  
    12   TSLA      827   Elon  
    13   SPY       277   Josh  
    14   TSLA      632   Elon  
    15   SPY       166   Josh  
    16   TSLA      758   Elon  
    17   GME       700   Lauren
    18   SPY       355   Josh  
    19   AMZN       68   Dan   
    20   TSLA      970   Elon  
    21   AMZN      446   Dan   
    22   GME       893   Lauren
    23   SPY       678   Josh  
    24   SPY       778   Josh  

``merge_asof()``
----------------

In a ``merge_asof()``, Riptable matches on the nearest key rather than
an equal key.

This is useful for merges based on keys that are times, where the times
in one Dataset are not an exact match for the times in another Dataset,
but they’re close enough to be used to merge the data.

Note: To most efficiently find the nearest match, ``merge_asof()``
requires both key columns to be sorted. The key columns must also be
numeric, such as a datetime, integer, or float. You can check whether a
column is sorted with ``issorted()``, or just sort it using
``sort_inplace()``. (If the key columns aren’t sorted, Riptable will
give you an error when you try to merge.)

With ``merge_asof()``, you need to specify how you want to find the
closest match: 

- ``direction='forward'`` matches based on the closest key in the right Dataset 
  that’s greater than the key in the left Dataset. 
- ``direction='backward'`` matches based on the closest key in the right Dataset 
  that’s less than the key in the left Dataset. 
- ``direction='nearest'`` matches based on the closest key in the right Dataset, 
  regardless of whether it’s greater than or less than the key in the left Dataset.

Let’s see an example based on closest times. The left Dataset has three
trades and their times. The right Dataset has spot prices and times that
are not all exact matches. We’ll merge the spot prices from the right
Dataset by getting the values associated with the nearest earlier times.

    >>> # Left Dataset with trades and times
    >>> ds = rt.Dataset({'Symbol': ['AAPL', 'AMZN', 'AAPL'], 
    ...                  'Venue': ['A', 'I', 'A'],
    ...                  'Time': rt.TimeSpan(['09:30', '10:00', '10:20'])})
    >>> # Right Dataset with spot prices and nearby times
    >>> spot_ds = rt.Dataset({'Symbol': ['AMZN', 'AMZN', 'AMZN', 'AAPL', 'AAPL', 'AAPL'],
    ...                       'Spot Price': [2000.0, 2025.0, 2030.0, 500.0, 510.0, 520.0],
    ...                       'Time': rt.TimeSpan(['09:30', '10:00', '10:25', '09:25', '10:00', '10:25'])})
    >>> ds
    #   Symbol   Venue                 Time
    -   ------   -----   ------------------
    0   AAPL     A       09:30:00.000000000
    1   AMZN     I       10:00:00.000000000
    2   AAPL     A       10:20:00.000000000

    >>> spot_ds
    #   Symbol   Spot Price                 Time
    -   ------   ----------   ------------------
    0   AMZN       2,000.00   09:30:00.000000000
    1   AMZN       2,025.00   10:00:00.000000000
    2   AMZN       2,030.00   10:25:00.000000000
    3   AAPL         500.00   09:25:00.000000000
    4   AAPL         510.00   10:00:00.000000000
    5   AAPL         520.00   10:25:00.000000000

Note that an as-of merge requires the ``on`` columns to be sorted. Before the merge,
the ``on`` columns are always checked. If they're not sorted, by default they are
sorted before the merge; the original order is then restored in the returned merged
Dataset.

If you don't need to preserve the existing ordering, it's faster to sort the
``on`` columns in place first::

    >>> spot_ds.sort_inplace('Time')
    #   Symbol   Spot Price                 Time
    -   ------   ----------   ------------------
    0   AAPL         500.00   09:25:00.000000000
    1   AMZN       2,000.00   09:30:00.000000000
    2   AMZN       2,025.00   10:00:00.000000000
    3   AAPL         510.00   10:00:00.000000000
    4   AAPL         520.00   10:25:00.000000000
    5   AMZN       2,030.00   10:25:00.000000000

Now we can merge based on the nearest earlier time. But not just any
nearest earlier time – we want to make sure it’s the nearest earlier
time associated with the same symbol. We use the optional ``by``
parameter to make sure we match on the symbol before getting the nearest
earlier time. We'll also use the ``matched_on`` argument to show which
key in ``spot_ds`` was matched on::

    >>> ds.merge_asof(spot_ds, on='Time', by='Symbol', direction='backward', matched_on=True)
    #   Symbol                 Time   Venue   Spot Price           matched_on
    -   ------   ------------------   -----   ----------   ------------------
    0   AAPL     09:30:00.000000000   A           500.00   09:25:00.000000000
    1   AMZN     10:00:00.000000000   I         2,025.00   10:00:00.000000000
    2   AAPL     10:20:00.000000000   A           510.00   10:00:00.000000000

We can see that both AAPL trades were matched based on the nearest
earlier time.

Merge based on the nearest later time::

    >>> ds.merge_asof(spot_ds, on='Time', by='Symbol', direction='forward', matched_on=True)
    #   Symbol                 Time   Venue   Spot Price           matched_on
    -   ------   ------------------   -----   ----------   ------------------
    0   AAPL     09:30:00.000000000   A           510.00   10:00:00.000000000
    1   AMZN     10:00:00.000000000   I         2,025.00   10:00:00.000000000
    2   AAPL     10:20:00.000000000   A           520.00   10:25:00.000000000

Both AAPL trades were matched based on the nearest later time.

Here, we get the spot price associated with whatever time is nearest,
whether it’s earlier or later::

    >>> ds.merge_asof(spot_ds, on='Time', by='Symbol', direction='nearest', matched_on=True)
    #   Symbol                 Time   Venue   Spot Price           matched_on
    -   ------   ------------------   -----   ----------   ------------------
    0   AAPL     09:30:00.000000000   A           500.00   09:25:00.000000000
    1   AMZN     10:00:00.000000000   I         2,025.00   10:00:00.000000000
    2   AAPL     10:20:00.000000000   A           520.00   10:25:00.000000000

For the first AAPL trade, the nearest time is earlier. For the second
AAPL trade, the nearest time is later.

We won’t spend time on examples of ``merge2()``, which is Riptable’s
more general merge function that does database-style joins (left, right,
inner, outer). Check out the API Reference for details.

Next, we’ll briefly cover a couple of ways to change the shape of your
Dataset: `Reshape Data with Pivot and
Transpose <tutorial_reshape.rst>`__.

--------------

Questions or comments about this guide? Email
RiptableDocumentation@sig.com.
