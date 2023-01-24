Perform Group Operations with Categoricals
==========================================

Riptable Categoricals have two related uses:

-  They efficiently store string (or other large dtype) arrays that have
   repeated values. The repeated values are partitioned into groups. For
   example, in a Categorical that contains three ‘AAPL’ symbols and four
   ‘MSFT’ symbols, the data is partitioned into an ‘AAPL’ group and a
   ‘MSFT’ group. Each group is mapped to an integer (‘AAPL’ = 1, ‘MSFT’
   = 2). This integer mapping allows the data to be stored and operated
   on more efficiently.
-  They’re Riptable’s class for doing group operations. A method applied
   to a Categorical is applied to each group separately.

We’ll talk about group operations first, then look at how Categoricals
store data under the hood.

Let’s create a Dataset with repeated stock symbols and some random
values::

    >>> rng = np.random.default_rng(42)
    >>> N = 50
    >>> ds = rt.Dataset()
    >>> ds.Symbol = rt.FA(rng.choice(['AAPL', 'AMZN', 'TSLA', 'SPY', 'GME'], N))
    >>> ds.Value = rng.random(N) * 100
    >>> ds
      #   Symbol   Value
    ---   ------   -----
      0   AAPL     19.46
      1   SPY      46.67
      2   SPY       4.38
      3   TSLA     15.43
      4   TSLA     68.30
      5   GME      74.48
      6   AAPL     96.75
      7   SPY      32.58
      8   AMZN     37.05
      9   AAPL     46.96
     10   TSLA     18.95
     11   GME      12.99
     12   SPY      47.57
     13   SPY      22.69
     14   SPY      66.98
    ...   ...        ...
     35   AAPL     66.84
     36   GME      47.11
     37   GME      56.52
     38   AMZN     76.50
     39   SPY      63.47
     40   AAPL     55.36
     41   SPY      55.92
     42   SPY      30.40
     43   AMZN      3.08
     44   AAPL     43.67
     45   GME      21.46
     46   TSLA     40.85
     47   GME      85.34
     48   SPY      23.39
     49   SPY       5.83

Categoricals for Group Operations
---------------------------------

We know how to get the sum of the Value column::

    >>> ds.Value.sum()
    2271.438342018526

But what if we want the sum for each symbol?

We could filter for each symbol and perform the operation on each
subset. Something like::

    >>> aapl = (ds.Symbol == 'AAPL')
    >>> ds.Value[aapl].nansum()
    484.77384051282024

… and so on.

Or we can turn the Symbol column into a Categorical. The Categorical
groups the repeated symbols, and allows us to compute the sum for each
symbol group all at once::

    >>> ds.Symbol = rt.Categorical(ds.Symbol)  # Note: rt.Cat() also works
    >>> ds
      #   Symbol   Value
    ---   ------   -----
      0   AAPL     19.46
      1   SPY      46.67
      2   SPY       4.38
      3   TSLA     15.43
      4   TSLA     68.30
      5   GME      74.48
      6   AAPL     96.75
      7   SPY      32.58
      8   AMZN     37.05
      9   AAPL     46.96
     10   TSLA     18.95
     11   GME      12.99
     12   SPY      47.57
     13   SPY      22.69
     14   SPY      66.98
    ...   ...        ...
     35   AAPL     66.84
     36   GME      47.11
     37   GME      56.52
     38   AMZN     76.50
     39   SPY      63.47
     40   AAPL     55.36
     41   SPY      55.92
     42   SPY      30.40
     43   AMZN      3.08
     44   AAPL     43.67
     45   GME      21.46
     46   TSLA     40.85
     47   GME      85.34
     48   SPY      23.39
     49   SPY       5.83

It doesn’t look any different (yet), but getting the sum per symbol
becomes a one-and-done operation.

    >>> ds.Symbol.sum(ds.Value)
    *Symbol    Value
    -------   ------
    AAPL      484.77
    AMZN      201.27
    GME       487.53
    SPY       477.57
    TSLA      620.29

When you call a method (usually a reducing operation) on a Categorical,
the method is applied separately to each group.

Hadley Wickham, known for his work on Rstats, described the operation
(also known as a “group by” operation) as *split, apply, combine*.

The illustration below shows how the groups are split based on the “key”
(or, in Riptable’s case, the Categorical group). The sum method is then
applied to each group separately, and the results are combined into an
output array.

.. figure:: split-apply-combine-gray.svg
   :alt: The split-apply-combine operation

For Riptable Categoricals, each group’s result is displayed aligned to
the group label.

Categoricals support most common reducing functions, including the
following.

======================== ============================
**Reducing Function**    **Description**
======================== ============================
``count()``              Total number of items
``first()``, ``last()``  First item, last item
``mean()``, ``median()`` Mean, median
``min()``, ``max()``     Minimum, maximum
``std()``, ``var()``     Standard deviation, variance
``prod()``               Product of all items
``sum()``                Sum of all items
======================== ============================

Here’s the `complete list of Categorical reducing
functions <tutorial_cat_reduce.rst>`__.

You can apply a function to multiple columns by passing a list of column
names::

    >>> ds.Value2 = ds.Value * 2
    >>> ds.Symbol.max([ds.Value, ds.Value2])
    *Symbol   Value   Value2
    -------   -----   ------
    AAPL      96.75   193.50
    AMZN      76.50   153.00
    GME       85.34   170.68
    SPY       66.98   133.96
    TSLA      83.27   166.54

Or to a whole Dataset. Any column for which the function fails – for
example, a numerical function on a string column – is not returned::

    >>> # Add a string column
    >>> ds.OptionType = rng.choice(['P', 'C'], N)
    >>> ds.Symbol.max(ds)
    *Symbol   Value   Value2
    -------   -----   ------
    AAPL      96.75   193.50
    AMZN      76.50   153.00
    GME       85.34   170.68
    SPY       66.98   133.96
    TSLA      83.27   166.54

What about non-reducing operations? Categoricals support them, but the
results are a little different.

For example, take ``cumsum()``, which is a running total.

When it’s applied to a Categorical, the function does get applied to
each group separately. But because it’s a non-reducing function, it
returns one value per row of the original data::

    >>> ds.Symbol.cumsum(ds.Value)
      #    Value
    ---   ------
      0    19.46
      1    46.67
      2    51.05
      3    15.43
      4    83.73
      5    74.48
      6   116.21
      7    83.64
      8    37.05
      9   163.17
     10   102.68
     11    87.47
     12   131.21
     13   153.90
     14   220.88
    ...      ...
     35   385.74
     36   324.21
     37   380.73
     38   198.19
     39   362.03
     40   441.10
     41   417.95
     42   448.35
     43   201.27
     44   484.77
     45   402.19
     46   620.29
     47   487.53
     48   471.74
     49   477.57

The result is a Dataset with a column of results that’s aligned to the
original data. If you like, you can add the results to the Dataset::

    >>> ds.CumValue = ds.Symbol.cumsum(ds.Value)
    >>> # Sort to make the cumulative sum more clear, then display only the relevant columns.
    >>> ds.sort_copy(['Symbol', 'CumValue']).col_filter(['Symbol', 'Value', 'CumValue'])
      #   Symbol   Value   CumValue
    ---   ------   -----   --------
      0   AAPL     19.46      19.46
      1   AAPL     96.75     116.21
      2   AAPL     46.96     163.17
      3   AAPL     70.03     233.20
      4   AAPL     28.83     262.03
      5   AAPL     56.87     318.90
      6   AAPL     66.84     385.74
      7   AAPL     55.36     441.10
      8   AAPL     43.67     484.77
      9   AMZN     37.05      37.05
     10   AMZN     38.75      75.79
     11   AMZN     45.89     121.69
     12   AMZN     76.50     198.19
     13   AMZN      3.08     201.27
     14   GME      74.48      74.48
    ...   ...        ...        ...
     35   SPY      30.40     448.35
     36   SPY      23.39     471.74
     37   SPY       5.83     477.57
     38   TSLA     15.43      15.43
     39   TSLA     68.30      83.73
     40   TSLA     18.95     102.68
     41   TSLA     83.27     185.95
     42   TSLA     83.23     269.17
     43   TSLA     80.48     349.65
     44   TSLA      0.74     350.39
     45   TSLA     66.49     416.87
     46   TSLA     70.52     487.39
     47   TSLA     78.07     565.46
     48   TSLA     13.98     579.44
     49   TSLA     40.85     620.29

A commonly used non-reducing function is ``shift()``. You can use it to
compare values with shifted versions of themselves – for example,
today’s price compared to yesterday’s price, the volume compared to the
volume an hour ago, etc.

Where a category has no previous value to shift forward, the missing
value is filled with ``nan``::

    >>> ds.PrevValue = ds.Symbol.shift(ds.Value)
    >>> ds.col_filter(['Symbol', 'Value', 'PrevValue'])
      #   Symbol   Value   PrevValue
    ---   ------   -----   ---------
      0   AAPL     19.46         nan
      1   SPY      46.67         nan
      2   SPY       4.38       46.67
      3   TSLA     15.43         nan
      4   TSLA     68.30       15.43
      5   GME      74.48         nan
      6   AAPL     96.75       19.46
      7   SPY      32.58        4.38
      8   AMZN     37.05         nan
      9   AAPL     46.96       96.75
     10   TSLA     18.95       68.30
     11   GME      12.99       74.48
     12   SPY      47.57       32.58
     13   SPY      22.69       47.57
     14   SPY      66.98       22.69
    ...   ...        ...         ...
     35   AAPL     66.84       56.87
     36   GME      47.11       11.45
     37   GME      56.52       47.11
     38   AMZN     76.50       45.89
     39   SPY      63.47       19.99
     40   AAPL     55.36       66.84
     41   SPY      55.92       63.47
     42   SPY      30.40       55.92
     43   AMZN      3.08       76.50
     44   AAPL     43.67       55.36
     45   GME      21.46       56.52
     46   TSLA     40.85       13.98
     47   GME      85.34       21.46
     48   SPY      23.39       30.40
     49   SPY       5.83       23.39

Other non-reducing fuctions include ``rolling_sum()``,
``rolling_mean()`` and their nan-versions ``rolling_nansum()`` and
``rolling_nanmean()``, and ``cumsum()`` and ``cumprod()``.

Other functions not listed here can also be applied to Categoricals,
including lambda functions and other user-defined functions, with the
help of ``apply()`` functions. We’ll see how those work below.

Notice that if we try to add the result of a *reducing* operation to a
Dataset, Riptable complains that the result isn’t the right length::

    >>> try:
    ...     ds.Mean = ds.Symbol.mean(ds.Value)
    ... except TypeError as e:
    ...     print("TypeError:", e)
    TypeError: ('Row mismatch in Dataset._check_addtype.  Tried to add Dataset of different lengths', 50, 5)

You can expand the result of a reducing function so that it’s aligned
with the original data by passing ``transform=True`` to the function::

    >>> ds.MaxValue = ds.Symbol.max(ds.Value, transform=True)
    >>> ds.sort_copy(['Symbol', 'Value']).col_filter(['Symbol', 'Value', 'MaxValue'])
      #   Symbol   Value   MaxValue
    ---   ------   -----   --------
      0   AAPL     19.46      96.75
      1   AAPL     28.83      96.75
      2   AAPL     43.67      96.75
      3   AAPL     46.96      96.75
      4   AAPL     55.36      96.75
      5   AAPL     56.87      96.75
      6   AAPL     66.84      96.75
      7   AAPL     70.03      96.75
      8   AAPL     96.75      96.75
      9   AMZN      3.08      76.50
     10   AMZN     37.05      76.50
     11   AMZN     38.75      76.50
     12   AMZN     45.89      76.50
     13   AMZN     76.50      76.50
     14   GME      11.45      85.34
    ...   ...        ...        ...
     35   SPY      55.92      66.98
     36   SPY      63.47      66.98
     37   SPY      66.98      66.98
     38   TSLA      0.74      83.27
     39   TSLA     13.98      83.27
     40   TSLA     15.43      83.27
     41   TSLA     18.95      83.27
     42   TSLA     40.85      83.27
     43   TSLA     66.49      83.27
     44   TSLA     68.30      83.27
     45   TSLA     70.52      83.27
     46   TSLA     78.07      83.27
     47   TSLA     80.48      83.27
     48   TSLA     83.23      83.27
     49   TSLA     83.27      83.27

The max value per symbol is repeated for every instance of the symbol.

Note the syntax for adding two columns of results to the Dataset::

    >>> ds[['MeanValue', 'MeanMaxValue']] = ds.Symbol.mean([ds.Value, ds.MaxValue],
    ...                                                    transform=True)[['Value', 'MaxValue']]
    >>> ds.col_filter(['Symbol', 'Value', 'MeanValue', 'MaxValue', 
    ...                'MeanMaxValue']).sort_copy('Symbol').head(25)
     #   Symbol   Value   MeanValue   MaxValue   MeanMaxValue
    --   ------   -----   ---------   --------   ------------
     0   AAPL     19.46       53.86      96.75          96.75
     1   AAPL     96.75       53.86      96.75          96.75
     2   AAPL     46.96       53.86      96.75          96.75
     3   AAPL     70.03       53.86      96.75          96.75
     4   AAPL     28.83       53.86      96.75          96.75
     5   AAPL     56.87       53.86      96.75          96.75
     6   AAPL     66.84       53.86      96.75          96.75
     7   AAPL     55.36       53.86      96.75          96.75
     8   AAPL     43.67       53.86      96.75          96.75
     9   AMZN     37.05       40.25      76.50          76.50
    10   AMZN     38.75       40.25      76.50          76.50
    11   AMZN     45.89       40.25      76.50          76.50
    12   AMZN     76.50       40.25      76.50          76.50
    13   AMZN      3.08       40.25      76.50          76.50
    14   GME      74.48       48.75      85.34          85.34
    15   GME      12.99       48.75      85.34          85.34
    16   GME      31.24       48.75      85.34          85.34
    17   GME      68.25       48.75      85.34          85.34
    18   GME      78.69       48.75      85.34          85.34
    19   GME      11.45       48.75      85.34          85.34
    20   GME      47.11       48.75      85.34          85.34
    21   GME      56.52       48.75      85.34          85.34
    22   GME      21.46       48.75      85.34          85.34
    23   GME      85.34       48.75      85.34          85.34
    24   SPY      46.67       34.11      66.98          66.98

Categoricals for Storing Strings
--------------------------------

To get a better sense of how Categoricals store data, let’s look at one
under the hood::

    >>> ds.Symbol
    Categorical([AAPL, SPY, SPY, TSLA, TSLA, ..., GME, TSLA, GME, SPY, SPY]) Length: 50
      FastArray([1, 4, 4, 5, 5, ..., 3, 5, 3, 4, 4], dtype=int8) Base Index: 1
      FastArray([b'AAPL', b'AMZN', b'GME', b'SPY', b'TSLA'], dtype='|S4') Unique count: 5

The first line shows the 50 symbols (elided with ‘…’). We can access the
entire array with ``as_string_array``::

    >>> ds.Symbol.as_string_array
    FastArray([b'AAPL', b'SPY', b'SPY', b'TSLA', b'TSLA', b'GME', b'AAPL',
               b'SPY', b'AMZN', b'AAPL', b'TSLA', b'GME', b'SPY', b'SPY',
               b'SPY', b'SPY', b'TSLA', b'AAPL', b'GME', b'TSLA', b'TSLA',
               b'AMZN', b'AAPL', b'GME', b'SPY', b'SPY', b'TSLA', b'GME',
               b'TSLA', b'TSLA', b'TSLA', b'AMZN', b'AAPL', b'TSLA', b'GME',
               b'AAPL', b'GME', b'GME', b'AMZN', b'SPY', b'AAPL', b'SPY',
               b'SPY', b'AMZN', b'AAPL', b'GME', b'TSLA', b'GME', b'SPY',
               b'SPY'], dtype='|S8')

The second line is a FastArray of integers. We can access the full
FastArray with ``_fa``::

    >>> ds.Symbol._fa
    FastArray([1, 4, 4, 5, 5, 3, 1, 4, 2, 1, 5, 3, 4, 4, 4, 4, 5, 1, 3, 5, 5,
               2, 1, 3, 4, 4, 5, 3, 5, 5, 5, 2, 1, 5, 3, 1, 3, 3, 2, 4, 1, 4,
               4, 2, 1, 3, 5, 3, 4, 4], dtype=int8)

In a Categorical, each unique category is mapped to an integer.

The list of unique categories is shown in the third line. It’s the same
thing we get if we do::

    >>> ds.Symbol.unique()
    FastArray([b'AAPL', b'AMZN', b'GME', b'SPY', b'TSLA'], dtype='|S4')

We can also access the list of unique categories with
``category_array``::

    >>> ds.Symbol.category_array
    FastArray([b'AAPL', b'AMZN', b'GME', b'SPY', b'TSLA'], dtype='|S4')

We can get a better picture of the mapping by putting the integer
FastArray into the Dataset::

    >>> ds.Mapping = ds.Symbol._fa
    >>> ds.col_filter(['Symbol', 'Mapping'])
      #   Symbol   Mapping
    ---   ------   -------
      0   AAPL           1
      1   SPY            4
      2   SPY            4
      3   TSLA           5
      4   TSLA           5
      5   GME            3
      6   AAPL           1
      7   SPY            4
      8   AMZN           2
      9   AAPL           1
     10   TSLA           5
     11   GME            3
     12   SPY            4
     13   SPY            4
     14   SPY            4
    ...   ...          ...
     35   AAPL           1
     36   GME            3
     37   GME            3
     38   AMZN           2
     39   SPY            4
     40   AAPL           1
     41   SPY            4
     42   SPY            4
     43   AMZN           2
     44   AAPL           1
     45   GME            3
     46   TSLA           5
     47   GME            3
     48   SPY            4
     49   SPY            4

It’s easier to see now that ‘AAPL’ is mapped to 1, ‘AMZN’ is mapped to
2, ‘GME’ is mapped to 3, etc.

Because it’s much more efficient to pass around integers than it is to
pass around strings, it’s common for string data with repeated values to
be stored using integer mapping.

If you have data stored this way, you can create a Categorical using the
integer array and the array of unique categories::

    >>> c = rt.Categorical([1, 3, 2, 2, 1, 3, 3, 1], categories=['a','b','c'])
    >>> c
    Categorical([a, c, b, b, a, c, c, a]) Length: 8
      FastArray([1, 3, 2, 2, 1, 3, 3, 1]) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

Notice that in this Categorical and the one we created above, the base
index is 1, not 0. This brings us to an important note about
Categoricals: By default, the base index is 1; 0 is reserved for the
‘Filtered’ category::

    >>> c1 = rt.Categorical([0, 2, 1, 1, 0, 2, 2, 0], categories=['a','b','c'])
    >>> c1
    Categorical([Filtered, b, a, a, Filtered, b, b, Filtered]) Length: 8
      FastArray([0, 2, 1, 1, 0, 2, 2, 0]) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

The Filtered category is the result of filtering at the time of
Categorical creation. One way to create a Categorical with a Filtered
category is to simply map a category to a 0. In this case, the ‘c’
category is mapped to 0.

When a category is Filtered, its rows are omitted from *all*
calculations on the Categorical::

    >>> c1.count()
    *key_0   Count
    ------   -----
    a            2
    b            3
    c            0

Another way to create a Filtered category at the time of Categorical
creation is to explicitly use a filter.

To show this, we’ll create a Dataset with symbols that are associated
with various exchanges::

    >>> N = 50
    >>> symbol_exchange = rt.Dataset()
    >>> symbol_exchange.Symbol = rt.FA(rng.choice(['A', 'B'], N))
    >>> symbol_exchange.Exchange = rt.FA(rng.choice(['X', 'Y', 'Z'], N))
    >>> symbol_exchange
      #   Symbol   Exchange
    ---   ------   --------
      0   A        X
      1   B        Y
      2   A        X
      3   B        X
      4   A        Y
      5   A        X
      6   B        Z
      7   B        X
      8   A        X
      9   B        Y
     10   A        Z
     11   B        X
     12   B        X
     13   A        Z
     14   B        Z
    ...   ...      ...
     35   B        Z
     36   B        X
     37   A        Y
     38   A        Y
     39   B        Y
     40   A        Z
     41   B        X
     42   A        Z
     43   B        X
     44   B        Y
     45   A        Y
     46   B        Z
     47   A        Y
     48   B        X
     49   A        X

Now we’ll create a Categorical that keeps only the symbols associated to
the ‘X’ exchange::

    >>> exchangeXFilter = symbol_exchange.Exchange == 'X'
    >>> # Group by symbol using only trades on exchange X
    >>> exchangeFilteredSymbolCat = rt.Cat(symbol_exchange.Symbol, filter=exchangeXFilter)
    >>> exchangeFilteredSymbolCat
    Categorical([A, Filtered, A, B, Filtered, ..., Filtered, Filtered, Filtered, B, A]) Length: 50
      FastArray([1, 0, 1, 2, 0, ..., 0, 0, 0, 2, 1], dtype=int8) Base Index: 1
      FastArray([b'A', b'B'], dtype='|S1') Unique count: 2

Now, a group operation applied to the Categorical omits the filtered
categories::

    >>> exchangeFilteredSymbolCat.count()
    *Symbol   Count
    -------   -----
    A             9
    B            10

To check::

    >>> aX = (symbol_exchange.Symbol == 'A') & (symbol_exchange.Exchange == 'X')
    >>> bX = (symbol_exchange.Symbol == 'B') & (symbol_exchange.Exchange == 'X')
    >>> print('aX sum: %i, bX sum: %i' %(aX.sum(), bX.sum()))
    aX sum: 9, bX sum: 10

It’s also possible to filter a category for only a certain operation –
we cover that below in `Perform Per-Group Operations on a Subset of
Data <#cat-subset>`__.

A Useful Way to Instantiate a Categorical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It can sometimes be useful to instantiate a Categorical with only one
category, then fill it in as needed.

For example, let’s say we have a Dataset with a column that has a lot of
categories, and we want to create a new Categorical column that keeps
two of those categories, properly aligned with the rest of the data in
the Dataset, and lumps the other categories into a category called
‘Other.’

Our Dataset, with a column of many categories::

    >>> ds_buildcat = rt.Dataset({'big_cat': rng.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], N)})
    >>> ds_buildcat
      #   big_cat
    ---   -------
      0   D      
      1   I      
      2   A      
      3   I      
      4   F      
      5   B      
      6   D      
      7   F      
      8   D      
      9   B      
     10   G      
     11   G      
     12   B      
     13   C      
     14   C      
    ...   ...    
     35   I      
     36   J      
     37   D      
     38   C      
     39   J      
     40   G      
     41   C      
     42   G      
     43   F      
     44   J      
     45   C      
     46   J      
     47   J      
     48   B      
     49   B    

We create our ‘small’ Categorical instantiated with 3s, which fills the
column with the ‘Other’ category::

    >>> ds_buildcat.small_cat = rt.Cat(rt.full(ds_buildcat.shape[0], 3), categories=['B', 'D', 'Other']) 
    >>> ds_buildcat.small_cat
    >>> ds_buildcat
      #   big_cat   small_cat
    ---   -------   ---------
      0   D         Other    
      1   I         Other    
      2   A         Other    
      3   I         Other    
      4   F         Other    
      5   B         Other    
      6   D         Other    
      7   F         Other    
      8   D         Other    
      9   B         Other    
     10   G         Other    
     11   G         Other    
     12   B         Other    
     13   C         Other    
     14   C         Other    
    ...   ...       ...      
     35   I         Other    
     36   J         Other    
     37   D         Other    
     38   C         Other    
     39   J         Other    
     40   G         Other    
     41   C         Other    
     42   G         Other    
     43   F         Other    
     44   J         Other    
     45   C         Other    
     46   J         Other    
     47   J         Other    
     48   B         Other    
     49   B         Other  

Now we can fill in the aligned ‘B’ and ‘D’ categories::

    >>> ds_buildcat.small_cat[ds_buildcat.big_cat == 'B'] = 'B'
    >>> ds_buildcat.small_cat[ds_buildcat.big_cat == 'D'] = 'D'
    >>> ds_buildcat
      #   big_cat   small_cat
    ---   -------   ---------
      0   D         D        
      1   I         Other    
      2   A         Other    
      3   I         Other    
      4   F         Other    
      5   B         B        
      6   D         D        
      7   F         Other    
      8   D         D        
      9   B         B        
      10  G         Other    
      11  G         Other    
      12  B         B        
      13  C         Other    
      14  C         Other    
     ...  ...       ...      
      35  I         Other    
      36  J         Other    
      37  D         D        
      38  C         Other    
      39  J         Other    
      40  G         Other    
      41  C         Other    
      42  G         Other    
      43  F         Other    
      44  J         Other    
      45  C         Other    
      46  J         Other    
      47  J         Other    
      48  B         B        
      49  B         B  

Perform Per-Group Operations on a Subset of Data
------------------------------------------------

As with columns/FastArrays, you can limit operations on Categoricals to
only the records that satisfy a given condition.

With Categoricals, you do this by passing a filter keyword argument to
the function called on the Cateogical.

For example, compute the average value per symbol for values greater
than 30.0::

    >>> ds.Symbol.mean(ds.Value, filter=ds.Value > 30.0)
    *Symbol   Value
    -------   -----
    AAPL      62.35
    AMZN      49.55
    GME       63.09
    SPY       48.41
    TSLA      71.40

The data that doesn’t meet the condition is omitted from the computation
for only that operation.

If you want to check your filter before applying a function to the
filtered data, you can call ``set_valid())`` on the Categorical, which works
similarly to how the ``filter`` method works on a Dataset::

    >>> ds.Symbol.set_valid(ds.Value > 30.0)
    Categorical([Filtered, SPY, Filtered, Filtered, TSLA, ..., Filtered, TSLA, GME, Filtered, Filtered]) Length: 50
      FastArray([0, 4, 0, 0, 5, ..., 0, 5, 3, 0, 0], dtype=int8) Base Index: 1
      FastArray([b'AAPL', b'AMZN', b'GME', b'SPY', b'TSLA'], dtype='|S4') Unique count: 5

To more closely spot-check, put the filtered categories in a Dataset::

    >>> ds_test = rt.Dataset()
    >>> ds_test.SymbolTest = ds.Symbol.set_valid(ds.Value > 30.0)
    >>> ds_test.ValueTest = ds.Value
    >>> ds_test.sample()
    #   SymbolTest   ValueTest
    -   ----------   ---------
    0   TSLA             62.15
    1   Filtered         11.85
    2   Filtered         27.46
    3   GME              40.13
    4   AMZN             52.90
    5   Filtered         16.19
    6   AMZN             61.77
    7   SPY              62.31
    8   AMZN             70.44
    9   AAPL             53.34

Now we can get the minimum value of the filtered data::

    >>> ds.Symbol.set_valid(ds.Value > 30.0).min(ds.Value)
    *SymbolTest   Value
    -----------   -----
    AAPL          53.34
    AMZN          52.90
    GME           40.13
    SPY           32.91
    TSLA          37.91

Note that ``set_valid()`` on a Categorical returns a Categorical of the
same length in which everywhere the filter result is False, the category
gets set to ‘Filtered’ and the associated index value is 0. This in
contrast to filtered Datasets, where ``filter()`` returns a smaller
Dataset, reduced to only the rows where the filter result is True (where
the filter condition is met).

The advice to avoid making unnecessary copies of large amounts of data
using ``set_valid()`` also applies to Categoricals.

The advice to name your mask filters for future reuse also applies::

    >>> my_filt = (ds.Value > 30.0)
    >>> ds.Symbol.sum(ds.Value, filter=my_filt)
    *Symbol    Value
    -------   ------
    AAPL      436.48
    AMZN      198.19
    GME       441.63
    SPY       387.31
    TSLA      571.20

Multi-Key Categoricals
----------------------

Multi-key Categoricals let you create and operate on groupings based on
two categories.

An example is a symbol-month pair, which you could use to get the
average value of a stock for each month in your data::

    >>> ds_mk = rt.Dataset()
    >>> N = 100
    >>> ds_mk.Symbol = rt.FA(rng.choice(['AAPL', 'AMZN', 'MSFT'], N))
    >>> ds_mk.Value = rt.FA(rng.random(N))
    >>> ds_mk.Date = rt.Date.range('20210101', days = 100)  # Dates from January to mid-April
    >>> ds_mk.Month = ds_mk.Date.start_of_month
    >>> ds_mk

We want to group the dates by month. An easy way to do this is by using
``start_of_month``::

    >>> ds_mk.Month = ds_mk.Date.start_of_month
    >>> ds_mk
      #   Symbol   Value         Date        Month
    ---   ------   -----   ----------   ----------
      0   AMZN      0.92   2021-01-01   2021-01-01
      1   AAPL      0.02   2021-01-02   2021-01-01
      2   AAPL      0.56   2021-01-03   2021-01-01
      3   AMZN      0.63   2021-01-04   2021-01-01
      4   MSFT      0.11   2021-01-05   2021-01-01
      5   MSFT      0.14   2021-01-06   2021-01-01
      6   AAPL      0.42   2021-01-07   2021-01-01
      7   MSFT      0.97   2021-01-08   2021-01-01
      8   AAPL      0.60   2021-01-09   2021-01-01
      9   MSFT      0.93   2021-01-10   2021-01-01
     10   MSFT      0.80   2021-01-11   2021-01-01
     11   MSFT      0.47   2021-01-12   2021-01-01
     12   AAPL      0.78   2021-01-13   2021-01-01
     13   MSFT      0.02   2021-01-14   2021-01-01
     14   AAPL      0.11   2021-01-15   2021-01-01
    ...   ...        ...          ...          ...
     85   AAPL      0.48   2021-03-27   2021-03-01
     86   AMZN      0.42   2021-03-28   2021-03-01
     87   MSFT      0.23   2021-03-29   2021-03-01
     88   AAPL      0.37   2021-03-30   2021-03-01
     89   AMZN      0.37   2021-03-31   2021-03-01
     90   MSFT      0.33   2021-04-01   2021-04-01
     91   AAPL      0.38   2021-04-02   2021-04-01
     92   AAPL      0.69   2021-04-03   2021-04-01
     93   AMZN      0.30   2021-04-04   2021-04-01
     94   AMZN      0.95   2021-04-05   2021-04-01
     95   AMZN      0.92   2021-04-06   2021-04-01
     96   AAPL      0.48   2021-04-07   2021-04-01
     97   MSFT      0.33   2021-04-08   2021-04-01
     98   AMZN      0.54   2021-04-09   2021-04-01
     99   AAPL      0.85   2021-04-10   2021-04-01

Now all Dates in January are associated to 2021-01-01, all Dates in
February are associated to 2021-02-01, etc. These firsts of the month
are our month groups.

Now we create a multi-key Categorical by passing ``rt.Cat()`` the Symbol
and Month columns::

    >>> ds_mk.Symbol_Month = rt.Cat([ds_mk.Symbol, ds_mk.Month])
    >>> ds_mk.Symbol_Month
    Categorical([(AMZN, 2021-01-01), (AAPL, 2021-01-01), (AAPL, 2021-01-01), (AMZN, 2021-01-01), (MSFT, 2021-01-01), ..., (AMZN, 2021-04-01), (AAPL, 2021-04-01), (MSFT, 2021-04-01), (AMZN, 2021-04-01), (AAPL, 2021-04-01)]) Length: 100
      FastArray([ 1,  2,  2,  1,  3, ..., 12, 11, 10, 12, 11], dtype=int8) Base Index: 1
      {'Symbol': FastArray([b'AMZN', b'AAPL', b'MSFT', b'MSFT', b'AAPL', ..., b'MSFT', b'AMZN', b'MSFT', b'AAPL', b'AMZN'], dtype='|S4'), 'Month': Date(['2021-01-01', '2021-01-01', '2021-01-01', '2021-02-01', '2021-02-01', ..., '2021-03-01', '2021-03-01', '2021-04-01', '2021-04-01', '2021-04-01'])} Unique count: 12

Note: We could have skipped creating a column for the firsts of the
month by using method chaining::

    >>> ds_mk.Symbol_Month = rt.Cat([ds_mk.Symbol, 
    ...                              ds_mk.Date.start_of_month.set_name('Month')])

Applying ``set_name()`` here gives a name to the FastArray holding the
start-of-month groups.

Now we can get the average value for each symbol-month pair::

    >>> ds_mk.Symbol_Month.mean(ds_mk.Value)
    *Symbol       *Month   Value
    -------   ----------   -----
    AMZN      2021-01-01    0.63
    AAPL      2021-01-01    0.44
    MSFT      2021-01-01    0.53
    .         2021-02-01    0.31
    AAPL      2021-02-01    0.42
    AMZN      2021-02-01    0.27
    AAPL      2021-03-01    0.42
    MSFT      2021-03-01    0.58
    AMZN      2021-03-01    0.53
    MSFT      2021-04-01    0.33
    AAPL      2021-04-01    0.60
    AMZN      2021-04-01    0.67

The aggregated results are presented with the two group keys arranged
hierarchically.

All the aggregation functions supported by Categoricals can also be used
for multi-key Categoricals.

You can also filter multi-key Categoricals by calling ``filter()`` on
the Categorical, and operate on filterd data by passing the filter
keyword argument to the function you use.

Get the symbol-month pairs for values over 0.4::

    >>> ds_mk.Symbol_Month.filter(ds_mk.Value > 0.4)::
    Categorical([(AMZN, 2021-01-01), Filtered, (AAPL, 2021-01-01), (AMZN, 2021-01-01), Filtered, ..., (AMZN, 2021-04-01), (AAPL, 2021-04-01), Filtered, (AMZN, 2021-04-01), (AAPL, 2021-04-01)]) Length: 100
      FastArray([ 1,  0,  2,  1,  0, ..., 11, 10,  0, 11, 10], dtype=int8) Base Index: 1
      {'Symbol': FastArray([b'AMZN', b'AAPL', b'MSFT', b'MSFT', b'AAPL', ..., b'AAPL', b'MSFT', b'AMZN', b'AAPL', b'AMZN'], dtype='|S4'), 'Month': Date(['2021-01-01', '2021-01-01', '2021-01-01', '2021-02-01', '2021-02-01', ..., '2021-03-01', '2021-03-01', '2021-03-01', '2021-04-01', '2021-04-01'])} Unique count: 11

Sum the Values that are greater than 0.4::

    >>> ds_mk.Symbol_Month.nansum(ds_mk.Value, filter=ds_mk.Value > 0.4)
    *Symbol       *Month   Value
    -------   ----------   -----
    AMZN      2021-01-01    5.07
    AAPL      2021-01-01    4.28
    MSFT      2021-01-01    5.52
    .         2021-02-01    1.69
    AAPL      2021-02-01    4.12
    AMZN      2021-02-01    0.85
    AAPL      2021-03-01    4.67
    MSFT      2021-03-01    5.55
    AMZN      2021-03-01    2.68
    MSFT      2021-04-01    0.00
    AAPL      2021-04-01    2.02
    AMZN      2021-04-01    2.40

Later on we’ll cover another Riptable function, ``Accum2()``, that
aggregates two groups similarly but provides a more styled output.

Bucket Numeric Data for Analysis
--------------------------------

When you have a large amount of numeric data, ``cut()`` and ``qcut()``
can help you split the values into Categorical bins (a.k.a. “buckets”)
for analysis.

Use ``cut()`` for buckets based on values of your choosing. Use
``qcut()`` for buckets based on sample quantiles.

Let’s create a moderately large Dataset::

    >>> N = 1_000
    >>> ds2 = rt.Dataset()
    >>> ds2.Symbol = rt.FA(rng.choice(['AAPL', 'AMZN', 'MSFT'], N))
    >>> base_price = 100 + rt.FA(np.linspace(0, 900, N))
    >>> noise = rt.FA(rng.normal(0, 50, N))
    >>> ds2.Price = base_price + noise
    >>> ds2
      #   Symbol      Price
    ---   ------   --------
      0   AMZN        93.87
      1   AMZN       150.69
      2   AAPL       154.76
      3   MSFT       153.99
      4   AMZN       105.55
      5   AMZN        62.25
      6   MSFT        51.22
      7   AMZN       123.54
      8   AAPL       126.17
      9   AAPL       172.47
     10   AAPL       164.01
     11   MSFT       103.30
     12   AAPL        48.60
     13   AAPL        95.76
     14   AMZN       123.47
    ...   ...           ...
    985   AMZN     1,027.85
    986   AAPL       993.06
    987   AMZN       867.37
    988   AAPL       940.92
    989   AAPL     1,025.38
    990   MSFT     1,052.54
    991   AAPL     1,048.25
    992   AMZN       914.09
    993   AMZN     1,009.67
    994   AAPL     1,046.27
    995   AAPL       913.48
    996   AMZN       996.90
    997   AMZN     1,011.89
    998   MSFT       984.06
    999   MSFT       907.39

With ``cut()``, you can create equal-width buckets or choose your own
intervals.

To split values into equal-width buckets, just specify an integer number
of buckets (in this case 5)::

    >>> ds2.PriceBucket = rt.cut(ds2.Price, 5)
    >>> ds2
      #   Symbol      Price   PriceBucket      
    ---   ------   --------   -----------------
      0   AMZN        93.87   -3.011->221.182  
      1   AMZN       150.69   -3.011->221.182  
      2   AAPL       154.76   -3.011->221.182  
      3   MSFT       153.99   -3.011->221.182  
      4   AMZN       105.55   -3.011->221.182  
      5   AMZN        62.25   -3.011->221.182  
      6   MSFT        51.22   -3.011->221.182  
      7   AMZN       123.54   -3.011->221.182  
      8   AAPL       126.17   -3.011->221.182  
      9   AAPL       172.47   -3.011->221.182  
      10  AAPL       164.01   -3.011->221.182  
      11  MSFT       103.30   -3.011->221.182  
      12  AAPL        48.60   -3.011->221.182  
      13  AAPL        95.76   -3.011->221.182  
      14  AMZN       123.47   -3.011->221.182  
     ...   ...          ...   ...              
     985  AMZN     1,027.85   893.763->1117.956
     986  AAPL       993.06   893.763->1117.956
     987  AMZN       867.37   669.569->893.763 
     988  AAPL       940.92   893.763->1117.956
     989  AAPL     1,025.38   893.763->1117.956
     990  MSFT     1,052.54   893.763->1117.956
     991  AAPL     1,048.25   893.763->1117.956
     992  AMZN       914.09   893.763->1117.956
     993  AMZN     1,009.67   893.763->1117.956
     994  AAPL     1,046.27   893.763->1117.956
     995  AAPL       913.48   893.763->1117.956
     996  AMZN       996.90   893.763->1117.956
     997  AMZN     1,011.89   893.763->1117.956
     998  MSFT       984.06   893.763->1117.956
     999  MSFT       907.39   893.763->1117.956

Notice that the buckets form the groups of a Categorical::

    >>> ds2.PriceBucket
    Categorical([-3.011->221.182, -3.011->221.182, -3.011->221.182, -3.011->221.182, -3.011->221.182, ..., 893.763->1117.956, 893.763->1117.956, 893.763->1117.956, 893.763->1117.956, 893.763->1117.956]) Length: 1000
      FastArray([1, 1, 1, 1, 1, ..., 5, 5, 5, 5, 5], dtype=int8) Base Index: 1
      FastArray([b'-3.011->221.182', b'221.182->445.376', b'445.376->669.569', b'669.569->893.763', b'893.763->1117.956'], dtype='|S17') Unique count: 5

To choose your own intervals, provide the endpoints. Here, we define
bins that cover two intervals: one bin for prices from 0 to 500 (0
excluded), and one for prices from 500 to 1,000 (500 excluded)::

    >>> buckets = [0, 600, 1200]
    >>> ds2.PriceBucket2 = rt.cut(ds2.Price, buckets)
    >>> ds2
      #   Symbol      Price   PriceBucket         PriceBucket2 
    ---   ------   --------   -----------------   -------------
      0   AMZN        93.87   -3.011->221.182     0.0->600.0   
      1   AMZN       150.69   -3.011->221.182     0.0->600.0   
      2   AAPL       154.76   -3.011->221.182     0.0->600.0   
      3   MSFT       153.99   -3.011->221.182     0.0->600.0   
      4   AMZN       105.55   -3.011->221.182     0.0->600.0   
      5   AMZN        62.25   -3.011->221.182     0.0->600.0   
      6   MSFT        51.22   -3.011->221.182     0.0->600.0   
      7   AMZN       123.54   -3.011->221.182     0.0->600.0   
      8   AAPL       126.17   -3.011->221.182     0.0->600.0   
      9   AAPL       172.47   -3.011->221.182     0.0->600.0   
     10   AAPL       164.01   -3.011->221.182     0.0->600.0   
     11   MSFT       103.30   -3.011->221.182     0.0->600.0   
     12   AAPL        48.60   -3.011->221.182     0.0->600.0   
     13   AAPL        95.76   -3.011->221.182     0.0->600.0   
     14   AMZN       123.47   -3.011->221.182     0.0->600.0   
    ...   ...           ...   ...                 ...          
    985   AMZN     1,027.85   893.763->1117.956   600.0->1200.0
    986   AAPL       993.06   893.763->1117.956   600.0->1200.0
    987   AMZN       867.37   669.569->893.763    600.0->1200.0
    988   AAPL       940.92   893.763->1117.956   600.0->1200.0
    989   AAPL     1,025.38   893.763->1117.956   600.0->1200.0
    990   MSFT     1,052.54   893.763->1117.956   600.0->1200.0
    991   AAPL     1,048.25   893.763->1117.956   600.0->1200.0
    992   AMZN       914.09   893.763->1117.956   600.0->1200.0
    993   AMZN     1,009.67   893.763->1117.956   600.0->1200.0
    994   AAPL     1,046.27   893.763->1117.956   600.0->1200.0
    995   AAPL       913.48   893.763->1117.956   600.0->1200.0
    996   AMZN       996.90   893.763->1117.956   600.0->1200.0
    997   AMZN     1,011.89   893.763->1117.956   600.0->1200.0
    998   MSFT       984.06   893.763->1117.956   600.0->1200.0
    999   MSFT       907.39   893.763->1117.956   600.0->1200.0

In interval notation, the intervals look like this: (0, 600] (600, 1200]

The left side of each interval is open (meaning the left value is
excluded), and the right side is closed. To switch which side is closed,
pass ``right=False``.

Use ``qcut()`` to get buckets based on sample quantiles. Unlike
``cut()``, ``qcut()`` will usually result in buckets that are of roughly
equal size – that is, each bucket will contain around the same number of
data points.

We’ll create a Dataset with symbol groups and contracts per day::

    >>> N = 1_000
    >>> ds3 = rt.Dataset()
    >>> ds3.SymbolGroup = rt.FA(rng.choice(['spx', 'eqt_comp', 'eqt300', 'eqtrest'], N))
    >>> ds3.ContractsPerDay = rng.integers(low=0, high=5_000, size=N)
    >>> ds3.head()
     #   SymbolGroup   ContractsPerDay
    --   -----------   ---------------
     0   eqt300                  1,624
     1   spx                       851
     2   spx                     3,487
     3   eqt300                    345
     4   eqtrest                 2,584
     5   spx                     3,639
     6   spx                     4,741
     7   eqtrest                 1,440
     8   eqtrest                    39
     9   spx                     3,618
    10   eqt_comp                    7
    11   eqt300                    331
    12   spx                     4,952
    13   eqt_comp                4,312
    14   eqt_comp                3,537
    15   eqt300                  4,177
    16   eqt_comp                  376
    17   eqt_comp                  444
    18   eqt_comp                1,504
    19   eqtrest                   118

Create three labeled buckets for the volume::

    >>> label_names = ['Low', 'Medium', 'High']
    >>> ds3.Volume = rt.qcut(ds3.ContractsPerDay, 3, labels=label_names)
    >>> ds3.head()
     #   SymbolGroup   ContractsPerDay   Volume
    --   -----------   ---------------   ------
     0   eqt300                  1,624   Low   
     1   spx                       851   Low   
     2   spx                     3,487   High  
     3   eqt300                    345   Low   
     4   eqtrest                 2,584   Medium
     5   spx                     3,639   High  
     6   spx                     4,741   High  
     7   eqtrest                 1,440   Low   
     8   eqtrest                    39   Low   
     9   spx                     3,618   High  
    10   eqt_comp                    7   Low   
    11   eqt300                    331   Low   
    12   spx                     4,952   High  
    13   eqt_comp                4,312   High  
    14   eqt_comp                3,537   High  
    15   eqt300                  4,177   High  
    16   eqt_comp                  376   Low   
    17   eqt_comp                  444   Low   
    18   eqt_comp                1,504   Low   
    19   eqtrest                   118   Low  

See the total number of contracts per day for each bucket::

    >>> ds3.Volume.nansum(ds3.ContractsPerDay)
    *Volume   ContractsPerDay
    -------   ---------------
    Clipped                 0
    Low               287,043
    Medium            850,758
    High            1,392,177

Similarly to ``cut()``, ``qcut()`` can take a list of quantiles (numbers
between 0 and 1, inclusive). Here, we create quartiles::

    >>> quartiles = [0, .25, .5, .75, 1.]
    >>> ds3.VolQuartiles = rt.qcut(ds3.ContractsPerDay, quartiles)
    >>> ds3.head()
     #   SymbolGroup   ContractsPerDay   Volume   VolQuartiles   
    --   -----------   ---------------   ------   ---------------
     0   eqt300                  1,624   Low      1273.75->2601.0
     1   spx                       851   Low      0.0->1273.75   
     2   spx                     3,487   High     2601.0->3793.0 
     3   eqt300                    345   Low      0.0->1273.75   
     4   eqtrest                 2,584   Medium   1273.75->2601.0
     5   spx                     3,639   High     2601.0->3793.0 
     6   spx                     4,741   High     3793.0->4991.0 
     7   eqtrest                 1,440   Low      1273.75->2601.0
     8   eqtrest                    39   Low      0.0->1273.75   
     9   spx                     3,618   High     2601.0->3793.0 
    10   eqt_comp                    7   Low      0.0->1273.75   
    11   eqt300                    331   Low      0.0->1273.75   
    12   spx                     4,952   High     3793.0->4991.0 
    13   eqt_comp                4,312   High     3793.0->4991.0 
    14   eqt_comp                3,537   High     2601.0->3793.0 
    15   eqt300                  4,177   High     3793.0->4991.0 
    16   eqt_comp                  376   Low      0.0->1273.75   
    17   eqt_comp                  444   Low      0.0->1273.75   
    18   eqt_comp                1,504   Low      1273.75->2601.0
    19   eqtrest                   118   Low      0.0->1273.75  

Per-Group Calculations with Other Functions
-------------------------------------------

Categoricals support most common functions. For functions that aren’t
supported (for example, a function you’ve written), you can use
``apply_reduce()`` to apply a reducing function and
``apply_nonreduce()`` to apply a non-reducing function.

``apply_reduce()``
~~~~~~~~~~~~~~~~~~

The function you use with ``apply_reduce()`` can take in one or multiple
columns/FastArrays as input (as long as the function you want to use can
take multiple columns as arguments), but it must return a single value.

To illustrate, we’ll use ``apply_reduce()`` with two simple lambda
functions that each return one value. (A lambda function is an anonymous
function that consists of a single statement and gives back a return
value. When you have a function that takes a function as an argument,
using a lambda function as the argument can be simpler and clearer than
defining a function separately.)

First, we’ll create a new Dataset::

    >>> N = 50
    >>> ds = rt.Dataset()
    >>> ds.Symbol = rt.Cat(rng.choice(['AAPL', 'AMZN', 'TSLA', 'SPY', 'GME'], N))
    >>> ds.Value = rng.random(N) * 100
    >>> ds.Value2 = ds.Value * 2
    >>> ds.sample()
    #   Symbol   Value   Value2
    -   ------   -----   ------
    0   SPY      41.04    82.09
    1   TSLA     93.07   186.14
    2   AMZN      2.03     4.05
    3   AAPL     16.19    32.37
    4   AMZN      2.42     4.85
    5   TSLA     98.13   196.26
    6   SPY      98.67   197.34
    7   SPY      62.31   124.61
    8   TSLA     96.79   193.58
    9   TSLA     67.35   134.70

The first lambda function takes one column as input::

    >>> # ds.Value becomes the 'x' in our lambda function
    >>> ds.Symbol.apply_reduce(lambda x: x.min() + 2, ds.Value)
    *Symbol   Value
    -------   -----
    AAPL      11.36
    AMZN       4.03
    GME       16.65
    SPY        7.76
    TSLA       2.10

Note that because we’re operating on a Categorical, the functions
actually return one value *for each group*.

Our second lambda function takes two columns as input::

    >>> ds.Symbol.apply_reduce(lambda x, y: x.sum() * y.mean(), (ds.Value, ds.Value2))
    *Symbol       Value
    -------   ---------
    AAPL      26,904.13
    AMZN      39,400.64
    GME       26,857.53
    SPY       32,560.75
    TSLA      74,124.69

Also note that in this example, the first column listed in the tuple is
the column name shown in the output.

If you like, you can use ``transform=True`` to expand the results and
assign them to a column::

    >>> ds.MyCalc1 = ds.Symbol.apply_reduce(lambda x: x.min() + 2, ds.Value, transform=True)
    >>> ds.MyCalc2 = ds.Symbol.apply_reduce(lambda x, y: x.sum() * y.mean(), (ds.Value, ds.Value2), transform=True)
    >>> ds
      #   Symbol   Value   Value2   MyCalc1     MyCalc2
    ---   ------   -----   ------   -------   ---------
      0   AAPL     12.39    24.77     11.36   26,904.13
      1   SPY      41.04    82.09      7.76   32,560.75
      2   AMZN     55.69   111.39      4.03   39,400.64
      3   TSLA     93.07   186.14      2.10   74,124.69
      4   TSLA      3.62     7.24      2.10   74,124.69
      5   TSLA     62.15   124.29      2.10   74,124.69
      6   SPY      45.77    91.55      7.76   32,560.75
      7   AMZN      2.03     4.05      4.03   39,400.64
      8   SPY      24.95    49.91      7.76   32,560.75
      9   AMZN     11.85    23.70      4.03   39,400.64
     10   AMZN     21.68    43.36      4.03   39,400.64
     11   TSLA     27.46    54.91      2.10   74,124.69
     12   GME      40.13    80.26     16.65   26,857.53
     13   AMZN     52.90   105.81      4.03   39,400.64
     14   TSLA      0.10     0.20      2.10   74,124.69
    ...   ...        ...      ...       ...         ...
     35   TSLA     38.40    76.79      2.10   74,124.69
     36   AAPL     93.12   186.25     11.36   26,904.13
     37   SPY      14.92    29.85      7.76   32,560.75
     38   AAPL     99.71   199.41     11.36   26,904.13
     39   TSLA     37.91    75.83      2.10   74,124.69
     40   GME      64.88   129.75     16.65   26,857.53
     41   TSLA     96.79   193.58      2.10   74,124.69
     42   SPY       5.76    11.52      7.76   32,560.75
     43   TSLA     92.29   184.57      2.10   74,124.69
     44   AMZN     56.78   113.56      4.03   39,400.64
     45   AMZN     70.44   140.88      4.03   39,400.64
     46   TSLA     14.92    29.84      2.10   74,124.69
     47   AAPL     53.34   106.68     11.36   26,904.13
     48   TSLA     67.35   134.70      2.10   74,124.69
     49   TSLA     45.62    91.25      2.10   74,124.69

As expected, every instance of a category gets the same value.

``apply_nonreduce()``
~~~~~~~~~~~~~~~~~~~~~

For ``apply_nonreduce()``, our lambda function computes a new value for
every element of the original input::

    >>> ds.MyCalc3 = ds.Symbol.apply_nonreduce(lambda x: x.cumsum() + 2, ds.Value)
    >>> ds
      #   Symbol   Value   Value2   MyCalc1     MyCalc2   MyCalc3
    ---   ------   -----   ------   -------   ---------   -------
      0   AAPL     12.39    24.77     11.36   26,904.13     14.39
      1   SPY      41.04    82.09      7.76   32,560.75     43.04
      2   AMZN     55.69   111.39      4.03   39,400.64     57.69
      3   TSLA     93.07   186.14      2.10   74,124.69     95.07
      4   TSLA      3.62     7.24      2.10   74,124.69     98.69
      5   TSLA     62.15   124.29      2.10   74,124.69    160.84
      6   SPY      45.77    91.55      7.76   32,560.75     88.82
      7   AMZN      2.03     4.05      4.03   39,400.64     59.72
      8   SPY      24.95    49.91      7.76   32,560.75    113.77
      9   AMZN     11.85    23.70      4.03   39,400.64     71.57
     10   AMZN     21.68    43.36      4.03   39,400.64     93.25
     11   TSLA     27.46    54.91      2.10   74,124.69    188.30
     12   GME      40.13    80.26     16.65   26,857.53     42.13
     13   AMZN     52.90   105.81      4.03   39,400.64    146.15
     14   TSLA      0.10     0.20      2.10   74,124.69    188.40
    ...   ...        ...      ...       ...         ...       ...
     35   TSLA     38.40    76.79      2.10   74,124.69    417.18
     36   AAPL     93.12   186.25     11.36   26,904.13    133.05
     37   SPY      14.92    29.85      7.76   32,560.75    399.73
     38   AAPL     99.71   199.41     11.36   26,904.13    232.76
     39   TSLA     37.91    75.83      2.10   74,124.69    455.09
     40   GME      64.88   129.75     16.65   26,857.53    261.12
     41   TSLA     96.79   193.58      2.10   74,124.69    551.88
     42   SPY       5.76    11.52      7.76   32,560.75    405.49
     43   TSLA     92.29   184.57      2.10   74,124.69    644.17
     44   AMZN     56.78   113.56      4.03   39,400.64    437.63
     45   AMZN     70.44   140.88      4.03   39,400.64    508.07
     46   TSLA     14.92    29.84      2.10   74,124.69    659.09
     47   AAPL     53.34   106.68     11.36   26,904.13    286.10
     48   TSLA     67.35   134.70      2.10   74,124.69    726.44
     49   TSLA     45.62    91.25      2.10   74,124.69    772.06

Like ``apply_reduce()``, ``apply_nonreduce()`` can take one or multiple
columns as input::

    >>> ds.MyCalc4 = ds.Symbol.apply_nonreduce(lambda x, y: x.cumsum() + y, (ds.Value, ds.Value2))
    >>> ds
      #   Symbol   Value   Value2   MyCalc1     MyCalc2   MyCalc3   MyCalc4
    ---   ------   -----   ------   -------   ---------   -------   -------
      0   AAPL     12.39    24.77     11.36   26,904.13     14.39     37.16
      1   SPY      41.04    82.09      7.76   32,560.75     43.04    123.13
      2   AMZN     55.69   111.39      4.03   39,400.64     57.69    167.08
      3   TSLA     93.07   186.14      2.10   74,124.69     95.07    279.21
      4   TSLA      3.62     7.24      2.10   74,124.69     98.69    103.94
      5   TSLA     62.15   124.29      2.10   74,124.69    160.84    283.13
      6   SPY      45.77    91.55      7.76   32,560.75     88.82    178.36
      7   AMZN      2.03     4.05      4.03   39,400.64     59.72     61.77
      8   SPY      24.95    49.91      7.76   32,560.75    113.77    161.68
      9   AMZN     11.85    23.70      4.03   39,400.64     71.57     93.27
     10   AMZN     21.68    43.36      4.03   39,400.64     93.25    134.61
     11   TSLA     27.46    54.91      2.10   74,124.69    188.30    241.21
     12   GME      40.13    80.26     16.65   26,857.53     42.13    120.39
     13   AMZN     52.90   105.81      4.03   39,400.64    146.15    249.96
     14   TSLA      0.10     0.20      2.10   74,124.69    188.40    186.59
    ...   ...        ...      ...       ...         ...       ...       ...
     35   TSLA     38.40    76.79      2.10   74,124.69    417.18    491.97
     36   AAPL     93.12   186.25     11.36   26,904.13    133.05    317.30
     37   SPY      14.92    29.85      7.76   32,560.75    399.73    427.57
     38   AAPL     99.71   199.41     11.36   26,904.13    232.76    430.17
     39   TSLA     37.91    75.83      2.10   74,124.69    455.09    528.92
     40   GME      64.88   129.75     16.65   26,857.53    261.12    388.88
     41   TSLA     96.79   193.58      2.10   74,124.69    551.88    743.46
     42   SPY       5.76    11.52      7.76   32,560.75    405.49    415.01
     43   TSLA     92.29   184.57      2.10   74,124.69    644.17    826.74
     44   AMZN     56.78   113.56      4.03   39,400.64    437.63    549.19
     45   AMZN     70.44   140.88      4.03   39,400.64    508.07    646.95
     46   TSLA     14.92    29.84      2.10   74,124.69    659.09    686.92
     47   AAPL     53.34   106.68     11.36   26,904.13    286.10    390.78
     48   TSLA     67.35   134.70      2.10   74,124.69    726.44    859.14
     49   TSLA     45.62    91.25      2.10   74,124.69    772.06    861.31

``apply()``
~~~~~~~~~~~

If you want your custom function to return multiple aggregations – for
example, you want to return both the mean value of a column and the
minimum value of a column – use ``apply()``.

Warning: Because ``apply()`` isn’t a vectorized operation, it can be
slow and use a lot of memory if you’re using it on large amounts of
data. Try to avoid it if you can.

To be used with ``apply()``, your function must be able to take in a
Dataset. It can return a Dataset, a single array, or a dictionary of
column names and values.

Here’s a function that performs two reducing operations and returns a
Dataset::

    >>> def my_apply_func(ds):
    ...     new_ds = rt.Dataset({
    ...     'Mean_Value': ds.Value.mean(),
    ...     'Min_Value': ds.Value.min()
    ... })
    ...     return new_ds 

Again, because we’re calling ``apply()`` on a Categorical, the function
is applied separately to each group::

    >>> ds.Symbol.apply(my_apply_func, ds)
    *Symbol   Mean_Value   Min_Value
    -------   ----------   ---------
    AAPL           47.35        9.36
    AMZN           38.93        2.03
    GME            51.82       14.65
    SPY            40.35        5.76
    TSLA           48.13        0.10

Our second function performs two non-reducing operations::

    >>> def my_apply_func2(ds):
    ...     new_ds = rt.Dataset({
    ...         'Val1': ds.Value * 3,
    ...         'Val2': ds.Value * 4
    ...     })
    ...     return new_ds
    >>> ds.Symbol.apply(my_apply_func2, ds)
    *gb_key_0     Val1     Val2
    ---------   ------   ------
    AAPL         37.16    49.54
    SPY         123.13   164.18
    AMZN        167.08   222.77
    TSLA        279.21   372.28
    TSLA         10.87    14.49
    TSLA        186.44   248.58
    SPY         137.32   183.09
    AMZN          6.08     8.10
    SPY          74.86    99.82
    AMZN         35.55    47.39
    AMZN         65.04    86.72
    TSLA         82.37   109.83
    GME         120.39   160.52
    AMZN        158.71   211.62
    TSLA          0.30     0.40
    ...            ...      ...
    TSLA        115.19   153.58
    AAPL        279.37   372.50
    SPY          44.77    59.69
    AAPL        299.12   398.83
    TSLA        113.74   151.65
    GME         194.63   259.51
    TSLA        290.37   387.16
    SPY          17.28    23.04
    TSLA        276.86   369.15
    AMZN        170.34   227.12
    AMZN        211.32   281.76
    TSLA         44.76    59.67
    AAPL        160.02   213.35
    TSLA        202.05   269.41
    TSLA        136.87   182.50

Because the operations in this function are non-reducing operations, the
resulting Dataset is expanded.

In the next section, `Accums <tutorial_accums.rst>`__, we look at
another way to do multi-key groupings with fancier output.

--------------

Questions or comments about this guide? Email
RiptableDocumentation@sig.com.
