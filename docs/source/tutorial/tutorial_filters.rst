Get and Operate on Subsets of Data Using Filters
================================================

Earlier, we used indexing and slicing to select data. You can also use
filters to get data that meets a certain condition.

Datasets and FastArrays have a ``filter()`` method that returns the subset
of data that meets a given condition. But to operate on that data, it's often
better to pass a ``filter`` keyword argument to the method you're using.

This section covers:

- How to create conditions for filtering -- specifically, how comparison 
  operators create mask arrays that can be used to filter
- What to expect when you filter a FastArray or a Dataset
- How string operations can be used to create filters
- How to create more complex filters using logic operators
- How to replace values using filters and ``rt.where``
- How to operate on filtered Datasets in a memory-efficient way


Comparison Operators and Mask Arrays
------------------------------------

When used to compare scalar values, comparison operators 
(``>``, ``<``, ``>=``, ``<=``, ``==``, ``!=``) return True or False.

    >>> x = 10
    >>> x == 10  # equal
    True
    >>> x != 12  # not equal
    True
    >>> x > 12  # greater than
    False

In NumPy and Riptable, comparison operators are ufuncs, which means they can be
used to compare arrays.

When an array is compared element-wise with a scalar value or a same-length array,
the result is an array of Booleans.

    >>> a = rt.FastArray([1, 2, 3, 4, 5])
    >>> b = rt.FastArray([0, 5, 2, 4, 8])
    >>> a > 3 
    FastArray([False, False, False,  True,  True])
    >>> a <= b
    FastArray([False,  True, False,  True,  True])

These Boolean arrays can be used to filter data. In this 
context, they're often called Boolean mask arrays.

For the FastArray and Dataset ``filter()`` methods, you can pass a Boolean mask 
array directly or pass a comparison (or other operation) that results in a mask 
array. Here, we'll focus on the various ways to generate mask arrays based on 
comparisons and other conditions.


Filter a FastArray with a Comparison
------------------------------------

Above, we compared two FastArrays, ``a`` and ``b``, using the condition 
``a <= b`` to create a Boolean mask array. 

To filter ``a`` based on that 
condition (that is, to show only the values of ``a`` for which ``a <= b`` is 
True), use the FastArray ``filter()`` method with the condition.

    >>> f = a <= b
    >>> a.filter(f)
    FastArray([2, 4, 5])

Note that the returned FastArray is a copy; the original is unchanged::

    >>> a
    FastArray([1, 2, 3, 4, 5])


Filter a Dataset with a Comparison
----------------------------------

Datasets also have a ``filter()`` method. It returns a copy of the Dataset with
only the rows that meet the desired condition.

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

Say we want to see only the rows with options that are puts. 

The syntax is the same as for FastArrays::

    >>> f = ds.OptionType == 'P'
    >>> ds.filter(f)
    #   OSISymbol         UnderlyingSymbol    TradeDate   TradeSize   TradePrice   OptionType   Traded
    -   ---------------   ----------------   ----------   ---------   ----------   ----------   ------
    0   VIX:200520:35:0   VIX                2020-03-03        3.00        13.40   P             False
    1   AAPL:200417:255   AAPL               2020-03-19        1.00        27.50   P             False
    2   LITE:200619:82:   LITE               2020-03-24        5.00        14.80   P              True
    3   TLT:200529:165:   TLT                2020-05-26        1.00         1.78   P             False

By default all columns are returned. If you want to return only certain columns, 
you can combine the mask array with column selection::

   >>> ds.filter(f).col_filter(['OSISymbol', 'TradeSize'])

Alternatively, you can use the syntax we used to select Dataset rows to select 
rows based on the filter, along with the columns you want::

    >>> ds[f, [0, 3]]
    #   OSISymbol         TradeSize
    -   ---------------   ---------
    0   VIX:200520:35:0        3.00
    1   AAPL:200417:255        1.00
    2   LITE:200619:82:        5.00
    3   TLT:200529:165:        1.00

Here it could also make sense to pass the Traded column directly as a mask array::

    >>> ds.filter(ds.Traded)
    #   OSISymbol         UnderlyingSymbol    TradeDate   TradeSize   TradePrice   OptionType   Traded
    -   ---------------   ----------------   ----------   ---------   ----------   ----------   ------
    0   LITE:200619:82:   LITE               2020-03-24        5.00        14.80   P              True
    1   MFA:200515:2:0:   MFA                2020-04-20       10.00         0.29   C              True
    2   XOM:220121:60:0   XOM                2020-04-23        5.00         3.75   C              True
    3   AXSM:200515:85:   AXSM               2020-05-01        6.00         7.79   C              True
    4   UBER:200515:33:   UBER               2020-05-13        3.00         0.77   C              True

Note: Keep in mind that every time you use ``filter()``, it makes a copy of 
the Dataset that takes up memory. We cover a couple of strategies for minimizing 
memory use below, when we talk about operations on filtered data.


Use FastArray String Methods to Create Filters
----------------------------------------------

FastArray string methods are useful for creating conditions you can use to
filter.

Create a filter for OSISymbol strings that start with ‘A’::

    >>> f = ds.OSISymbol.str.startswith('A')
    >>> f
    FastArray([False,  True, False, False, False, False, False,  True, False, False])

For OSISymbol strings that contain the substring ‘2005’::

    >>> f = ds.OSISymbol.str.contains('2005')
    >>> f
    FastArray([ True, False, False, False,  True, False, False,  True,  True, True])

For UnderlyingSymbol strings that end with ‘L’::

    >>> f = ds.UnderlyingSymbol.str.regex_match('L$')
    >>> f
    FastArray([False,  True, False, False, False, False,  True, False, False, False])


Create More Complex Boolean Mask Filters with Bitwise Logic Operators (``&``, ``|``, ``~``)
-------------------------------------------------------------------------------------------

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

If you have complex filter criteria you want to reuse, assigning variable names 
to your filters can make things easier. You can also store your filters in a 
Riptable Struct::

    >>> s = rt.Struct()
    >>> s.ds = ds
    >>> s.callsover2 = callsover2
    >>> s
    #   Name         Type      Size               0       1       2    
    -   ----------   -------   ----------------   -----   -----   -----
    0   ds           Dataset   10 rows x 7 cols                        
    1   callsover2   bool      10                 False   False   False


Set Values in Columns with Masks and ``rt.where()``
---------------------------------------------------

You can also use mask arrays to update values that meet the filter condition.

Note, though, that the values are updated in place, not copied!

Suppose you want to update all the puts to be marked as traded. The FastArray
``filter()`` method doesn't let you set new values, but you can use the following
syntax::

    >>> f = ds.OptionType == 'P'
    >>> ds.Traded[f] = True
    >>> ds
    #   OSISymbol         UnderlyingSymbol    TradeDate   TradeSize   TradePrice   OptionType   Traded
    -   ---------------   ----------------   ----------   ---------   ----------   ----------   ------
    0   VIX:200520:35:0   VIX                2020-03-03        3.00        13.40   P              True
    1   AAPL:200417:255   AAPL               2020-03-19        1.00        27.50   P              True
    2   LITE:200619:82:   LITE               2020-03-24        5.00        14.80   P              True
    3   SPY:200406:265:   SPY                2020-04-06       50.00         0.14   C             False
    4   MFA:200515:2:0:   MFA                2020-04-20       10.00         0.29   C              True
    5   XOM:220121:60:0   XOM                2020-04-23        5.00         3.75   C              True
    6   CCL:200717:12:5   CCL                2020-04-27        1.00         2.55   C             False
    7   AXSM:200515:85:   AXSM               2020-05-01        6.00         7.79   C              True
    8   UBER:200515:33:   UBER               2020-05-13        3.00         0.77   C              True
    9   TLT:200529:165:   TLT                2020-05-26        1.00         1.78   P              True


What if you want to provide one value where the mask is True and a different value
where the mask is False?

``rt.where()`` is a function that works as an if-then-else procedure. 

It takes three arguments:

- ``condition``
- ``x``
- ``y``

Where the condition is met, it returns ``x``; otherwise, it returns ``y``. (If 
``x`` or ``y`` is an array, the value that corresponds to the True or False is
used.)

Here, for instance, ``rt.where`` returns ``a`` where ``a < 5``; otherwise
it returns ``10 * a``::

    >>> a = rt.FA([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> rt.where(a < 5, a, 10 * a)
    FastArray([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

In the Dataset above, we can have ``rt.where()`` mark puts as traded and calls as
not traded. Note that ``rt.where()`` returns a FastArray, so the result needs to 
be assigned as a Dataset column.

    >>> ds.Traded = rt.where(ds.OptionType == 'P', True, False)
    >>> ds[['OptionType', 'Traded']]
    #   OptionType   Traded
    -   ----------   ------
    0   P              True
    1   P              True
    2   P              True
    3   C             False
    4   C             False
    5   C             False
    6   C             False
    7   C             False
    8   C             False
    9   P              True


Operate on Filtered Data in a Dataset
-------------------------------------

Looking at filtered data can provide some useful insights. But often,
you want to operate on it.

Say you want to compute the total size of options that were traded.
Given that we just covered ``filter()``, you might be tempted to do
this::

    >>> ds.filter(ds.Traded).TradeSize.nansum()
    303.0

However, remember that ``filter()`` returns a copy of the Dataset,
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

Alternatively, we can use the FastArray `filter()` method to get only the 
sizes for the options that were traded. Then we get the sum::

    >>> ds.TradeSize.filter(ds.Traded).nansum()
    303.0

Both of these methods are much more memory-friendly and computationally
efficient than filtering (and making a copy of) the entire Dataset.

Next, we’ll check out Riptable’s datetime objects: `Work with Dates and
Times <tutorial_datetimes.rst>`__.

--------------

Questions or comments about this guide? Email
RiptableDocumentation@sig.com.
