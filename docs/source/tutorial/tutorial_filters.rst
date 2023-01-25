Get and Operate on Subsets of Data Using Filters
================================================

Earlier, we briefly explored operations for selecting data using
indexing and slicing.

It’s more typical to use filters to get data that meets a certain
condition or set of conditions.

We’ll work with this Dataset::

    >>> ds = rt.Dataset({
    ...     'OSISymbol': ['VIX:200520:35:0:P', 'AAPL:200417:255:0:P', 'LITE:200619:82:5:P', 
    ...                   'SPY:200406:265:0:C', 'MFA:200515:2:0:C', 'XOM:220121:60:0:C', 
    ...                   'CCL:200717:12:5:C', 'AXSM:200515:85:0:C', 'UBER:200515:33:0:C', 
    ...                   'TLT:200529:165:0:P'], 
    ...     'UnderlyingSymbol': ['VIX', 'AAPL', 'LITE', 'SPY', 'MFA', 'XOM', 'CCL', 'AXSM', 
    ...                          'UBER', 'TLT'],
    ...     'TradeDate': rt.Date(['2020-03-03', '2020-03-19', '2020-03-24', '2020-04-06', 
    ...                           '2020-04-20', '2020-04-23', '2020-04-27', '2020-05-01', 
    ...                           '2020-05-13', '2020-05-26']),
    ...     'TradeSize': [3., 1., 5., 50., 10., 5., 1., 6., 3., 1.],
    ...     'TradePrice': [13.4, 27.5, 14.8,  0.14, 0.29,  3.75,  2.55,  7.79,  0.77,  1.78 ],
    ...     'OptionType': ['P', 'P', 'P', 'C', 'C', 'C', 'C', 'C', 'C', 'P'],
    ...     'Traded': [False, False, True, False, True, True, False, True, True, False]
    ...   })
    >>> ds
    #   OSISymbol         UnderlyingSymbol    TradeDate   TradeSize   TradePrice   OptionType   Traded
    -   ---------------   ----------------   ----------   ---------   ----------   ----------   ------
    0   VIX:200520:35:0   VIX                2020-03-03        3.00        13.40   P             False
    1   AAPL:200417:255   AAPL               2020-03-19        1.00        27.50   P             False
    2   LITE:200619:82:   LITE               2020-03-24        5.00        14.80   P              True
    3   SPY:200406:265:   SPY                2020-04-06       50.00         0.14   C             False
    4   MFA:200515:2:0:   MFA                2020-04-20       10.00         0.29   C              True
    5   XOM:220121:60:0   XOM                2020-04-23        5.00         3.75   C              True
    6   CCL:200717:12:5   CCL                2020-04-27        1.00         2.55   C             False
    7   AXSM:200515:85:   AXSM               2020-05-01        6.00         7.79   C              True
    8   UBER:200515:33:   UBER               2020-05-13        3.00         0.77   C              True
    9   TLT:200529:165:   TLT                2020-05-26        1.00         1.78   P             False

Filter a Dataset 
-----------------

Suppose you want to see only the rows with options that are puts. You
can filter the Dataset by passing a predicate (a condition you want to
be met) to ``ds.filter()``::

    >>> ds.filter(ds.OptionType == 'P')
    #   OSISymbol         UnderlyingSymbol    TradeDate   TradeSize   TradePrice   OptionType   Traded
    -   ---------------   ----------------   ----------   ---------   ----------   ----------   ------
    0   VIX:200520:35:0   VIX                2020-03-03        3.00        13.40   P             False
    1   AAPL:200417:255   AAPL               2020-03-19        1.00        27.50   P             False
    2   LITE:200619:82:   LITE               2020-03-24        5.00        14.80   P              True
    3   TLT:200529:165:   TLT                2020-05-26        1.00         1.78   P             False

``ds.filter()`` returns a copy of the Dataset with the desired rows (and
all columns).

Note: Keep in mind that every time you use ``ds.filter()``, it makes a
copy of the Dataset that takes up memory. We cover a couple of
strategies for minimizing memory use below, when we talk about
operations on filtered data.

Filtering with Mask Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s take a closer look at the predicate we used::

    >>> ds.OptionType == 'P'
    FastArray([ True,  True,  True, False, False, False, False, False, False, True])

Notice that it returns a FastArray of Boolean values. In our
``ds.filter()`` example above, we were passing the filter function that
array. An array of Booleans used to filter another array (or in this
case, a Dataset) is often called a Boolean mask array. (Getting subsets
of data using Boolean mask arrays is also sometimes called Boolean
indexing.)

Where the mask array value is False, the corresponding row is omitted
from the returned Dataset. Note that the mask array you pass to
``ds.filter()`` needs to be the same length as the Dataset.

The ‘Traded’ Column in our Dataset is also an array of Booleans (and
it’s clearly the same length as the Dataset), so we can pass it to
``ds.filter()`` to see only the options that were traded::

    >>> ds.filter(ds.Traded)
    #   OSISymbol         UnderlyingSymbol    TradeDate   TradeSize   TradePrice   OptionType   Traded
    -   ---------------   ----------------   ----------   ---------   ----------   ----------   ------
    0   LITE:200619:82:   LITE               2020-03-24        5.00        14.80   P              True
    1   MFA:200515:2:0:   MFA                2020-04-20       10.00         0.29   C              True
    2   XOM:220121:60:0   XOM                2020-04-23        5.00         3.75   C              True
    3   AXSM:200515:85:   AXSM               2020-05-01        6.00         7.79   C              True
    4   UBER:200515:33:   UBER               2020-05-13        3.00         0.77   C              True

Assigning variable names to your filters makes them easier to use and
reuse, especially as your filter criteria get more complex.

If you want to return only certain columns, you can combine the saved
mask array with slicing::

    >>> f = ds.OptionType == 'P'
    >>> ds[f, ['OSISymbol', 'TradeSize']]
    #   OSISymbol         TradeSize
    -   ---------------   ---------
    0   VIX:200520:35:0        3.00
    1   AAPL:200417:255        1.00
    2   LITE:200619:82:        5.00
    3   TLT:200529:165:        1.00

You can also use the filter on one column/FastArray::

    >>> ds.OSISymbol[f]
    FastArray([b'VIX:200520:35:0:P', b'AAPL:200417:255:0:P', b'LITE:200619:82:5:P', b'TLT:200529:165:0:P'], dtype='|S19')

The ``==`` is a comparison operator, one of several you can use on
column data to create mask arrays that are aligned with the Dataset::

    >>> filt1 = (ds.TradeSize >= 5.00)
    >>> filt1
    FastArray([False, False,  True,  True,  True,  True, False,  True, False, False])

Riptable also has binary comparison methods that are analogous to the
symbol versions::

    >>> filt2 = ds.TradePrice.__lt__(1.00)
    >>> filt2
    FastArray([False, False, False,  True,  True, False, False, False,  True, False])

======================== ========== ==========
**Comparison**           **Symbol** **Method**
======================== ========== ==========
Equals                   =          \__eq_\_
Does not equal           !=         \__ne_\_
Greater than or equal to >=         \__ge_\_
Less than or equal to    <=         \__le_\_
Greater than             >          \__gt_\_
Less than                <          \__lt_\_
======================== ========== ==========

FastArray string methods are useful here, too.

OSISymbol strings that start with ‘A’::

    >>> ds.OSISymbol.str.startswith('A')
    FastArray([False,  True, False, False, False, False, False,  True, False, False])

That contain the substring ‘2005’::

    >>> ds.OSISymbol.str.contains('2005')
    FastArray([ True, False, False, False,  True, False, False,  True,  True, True])

Strings in the UnderlyingSymbol column that end with ‘L’::

    >>> ds.UnderlyingSymbol.str.regex_match('L$')
    FastArray([False,  True, False, False, False, False,  True, False, False, False])

Set Values in Columns with Filters and ``where()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use mask arrays to update values that meet the filter condition
(that is, where the mask array is ``True``)::

    >>> ds.TradeSize[filt1] = 75.0
    >>> ds
    #   OSISymbol         UnderlyingSymbol    TradeDate   TradeSize   TradePrice   OptionType   Traded
    -   ---------------   ----------------   ----------   ---------   ----------   ----------   ------
    0   VIX:200520:35:0   VIX                2020-03-03        3.00        13.40   P             False
    1   AAPL:200417:255   AAPL               2020-03-19        1.00        27.50   P             False
    2   LITE:200619:82:   LITE               2020-03-24       75.00        14.80   P              True
    3   SPY:200406:265:   SPY                2020-04-06       75.00         0.14   C             False
    4   MFA:200515:2:0:   MFA                2020-04-20       75.00         0.29   C              True
    5   XOM:220121:60:0   XOM                2020-04-23       75.00         3.75   C              True
    6   CCL:200717:12:5   CCL                2020-04-27        1.00         2.55   C             False
    7   AXSM:200515:85:   AXSM               2020-05-01       75.00         7.79   C              True
    8   UBER:200515:33:   UBER               2020-05-13        3.00         0.77   C              True
    9   TLT:200529:165:   TLT                2020-05-26        1.00         1.78   P             False

With ``where()``, you can set values in a FastArray based on whether or
not they meet a certain condition. It takes three arguments:
``condition``, ``x``, and ``y``. Where the condition is met, it returns
``x``; otherwise, it returns ``y``.

Here, for instance, ``where()`` returns ``a`` where ``a < 5``; otherwise
it returns ``10 * a``::

    >>> a = rt.FA([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> rt.where(a < 5, a, 10 * a)
    FastArray([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

Create More Complex Boolean Mask Filters with Bitwise Logic Operators (``&``, ``|``, ``~``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can build more complex filters using Python’s bitwise logic
operators, ``&`` (bitwise and), ``|`` (bitwise or), and ``~`` (bitwise
not).

Let’s say you want to construct a filter that returns True for calls
over $2.00. You can use ``&`` to ensure that both of those conditions
are met::

    >>> callsover2 = (ds.OptionType == 'C') & (ds.TradePrice > 2.00)
    >>> callsover2
    FastArray([False, False, False, False, False,  True,  True,  True, False, False])

Warning: When you use bitwise logic operators, always wrap the
expressions on either side in parentheses (as above) to make sure
they’re evaluated in the right order. Without the parentheses, operator
precedence rules would cause the expression above to be evaluated as
``ds.OptionType == ('C' & ds.TradePrice) > 2.00``, which would result in
an extremely slow call into native Python, followed by a crash. Also
note that the Python keywords ``AND``, ``OR``, and ``NOT`` do not work
with Boolean arrays. Use ``&``, ``|``, or ``~`` instead.

More examples of filter combinations::

    >>> # Define two filters
    >>> f1 = (ds.TradeSize <= 3.00)
    >>> f2 = (ds.TradePrice > 3.00)

True if both are True::

    >>> f1 & f2
    FastArray([ True,  True, False, False, False, False, False, False, False, False])

True if either one is True::

    >>> f1 | f2
    FastArray([ True,  True,  True, False, False,  True,  True,  True,  True, True])

The negation of the ``f1`` filter::

    >>> ~f1
    FastArray([False, False,  True,  True,  True,  True, False,  True, False, False])

As you create more complex filters, keep in mind another good use of a
Riptable Struct: storing your filters to save and reload them later::

    >>> s = rt.Struct()
    >>> s.ds = ds
    >>> s.callsover2 = callsover2
    >>> s
    #   Name         Type      Size               0       1       2    
    -   ----------   -------   ----------------   -----   -----   -----
    0   ds           Dataset   10 rows x 7 cols                        
    1   callsover2   bool      10                 False   False   False

Operate on Filtered Data 
-------------------------

Looking at filtered data can provide some useful insights. But often,
filtering data is just a prelude to operating on it.

Say you want to compute the total size of options that were traded.
Given that we just covered ``ds.filter()``, you might be tempted to do
this::

    >>> ds.filter(ds.Traded).TradeSize.nansum()
    303.0

However, remember that ``ds.filter()`` returns a copy of the Dataset,
filtered by the mask array. This is unnecessary here – we’re only
interested in the subset of one column of data. Fortunately, there are a
couple of ways to work only on the data we need.

We can pass a filter argument to ``nansum()`` with the Boolean array
contained in ``ds.Traded``::

    >>> ds.TradeSize.nansum(filter=ds.Traded)
    303.0

This gets the sum of only the values in the TradeSize column that meet
the filter criteria.

Note that the ``filter=`` is needed here – if you just pass the Boolean
array by itself, the array will be silently ignored::

    >>> ds.TradeSize.nansum(ds.Traded)
    384.0

Alternatively, we can pass our Boolean filter to the TradeSize column to
get only the sizes for the options that were traded. Then we get the
sum::

    >>> ds.TradeSize[ds.Traded].nansum()
    303.0

This filters the TradeSize column, then gets the sum.

Both of these methods are much more memory-friendly and computationally
efficient than filtering (and making a copy of) the entire Dataset.

Getting familiar with method chaining in Python can help you understand
the order in which chained operations are applied.

Next, we’ll check out Riptable’s datetime objects: `Work with Dates and
Times <tutorial_datetimes.rst>`__.

--------------

Questions or comments about this guide? Email
RiptableDocumentation@sig.com.
