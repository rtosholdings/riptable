Working with Missing Data
=========================

When you work with real-world data, you often have to deal with missing
values. It’s useful to know how Riptable stores and represents missing
values, how to detect missing values in your data, and how you can use a
few strategies to fill in missing values so you can continue to work
with the data effectively.

If you convert data between Riptable and other libraries, it’s also
important to know how conversions of missing values are handled. In this
section, we show how missing values are converted between Riptable and
Pandas.

Riptable Sentinel Values
------------------------

Riptable uses sentinel values for missing data. Missing floating-point
numbers are ``NaN``\ s (Not a Number), per the IEEE Standard for
Floating-Point Arithmetic. In Riptable, missing floating-point numbers
are indicated by ``nan``. Missing integers are indicated by ``Inv``::

    >>> ds = rt.Dataset({'Ints': [1, 2, 3], 'Floats': [0.1, 1.5, 2.7]})
    >>> ds.Ints[0] = ds.Ints.inv
    >>> ds.Floats[0] = ds.Floats.inv
    >>> ds
    #   Ints   Floats
    -   ----   ------
    0    Inv      nan
    1      2     1.50
    2      3     2.70

Note the difference in how they’re stored. The floating-point ``NaN`` is
stored as ``nan``::

    >>> ds.Floats
    FastArray([nan, 1.5, 2.7])

The missing integer is a large negative number::

    >>> ds.Ints
    FastArray([-2147483648,           2,           3])

In Riptable, missing interger values are stored as ``-MAXINT`` for ints
and ``MAXINT`` for unsigned ints. This has the potential to cause
problems, which we’ll look at below.

Tip: To find out what the missing/invalid value is for an array, use
``inv`` property. The array doesn’t necessarily contain the invalid
value; what’s returned is the invalid value for the array’s dtype::

    >>> ds.Ints.inv
    -2147483648

Arithmetic with floating-point ``NaN`` values is well-established: any
operation involving a ``NaN`` is another ``NaN``::

    >>> ds.Floats.sum()
    >>> ds.FloatsPlus = ds.Floats * 2
    >>> ds
    #   Ints   Floats   FloatsPlus
    -   ----   ------   ----------
    0    Inv      nan          nan
    1      2     1.50         3.00
    2      3     2.70         5.40

To help, many arithmetic functions have NaN versions that ignore ``NaN``
values::

    >>> ds.Ints.nansum()
    5

Be careful with missing integers, however! As of this writing, missing
integer values are treated at face value in arithmetic operations::

    >>> ds.Ints.sum()
    -2147483643

Fortunately, the ``NaN`` versions ignore the missing values::

    >>> ds.Ints.nansum()
    5

There are a few methods for detecting missing values in Riptable
structures.

For FastArrays, ``isnan()`` and ``notna()`` both return Boolean mask
arrays. As you might expect, ``isnan()`` returns True where it finds a
``NaN`` value::

    >>> ds.Ints.isnan()
    FastArray([ True, False, False])

And ``notna()`` returns True where it finds a non-``NaN`` value::

    >>> ds.Floats.notna()
    FastArray([False,  True,  True])

A more general approach is to use ``isfinite()``. It returns a Boolean
array where False indicates either a ``NaN`` or a value of positive or
negative infinity::

    >>> ds.Floats[1] = np.inf
    >>> ds.Floats.isfinite()
    FastArray([False, False,  True])

And as you might imagine, ``isnotfinite()`` does the opposite::

    >>> ds.Floats.isnotfinite()
    FastArray([ True,  True, False])

Note that ``inf`` is not considered a ``NaN``. The ``NaN`` versions of
functions don’t ignore infinite values (the result is positive or
negative ``inf``), so it can be good to check for them::

    >>> ds.Floats.nansum()
    inf

For Datasets, ``mask_and_isnan()`` and ``mask_or_isnan()`` each return a
FastArray of Booleans with a value for each row.

``mask_and_isnan()`` returns True for each row in which every value is ``NaN``::

    >>> ds.mask_and_isnan()
    FastArray([ True, False, False])

``mask_or_isnan()`` returns True for each row in which at least one value is ``NaN``::

    ds.mask_or_isnan()
    FastArray([ True, False, False])

Merging with Missing Values
---------------------------

Missing values are not equivalent::

    >>> rt.nan == rt.nan
    False

This is true for integer invalid values, string invalid values, filtered values of a
Categorical, etc. That means that merge functions do not treat invalid keys as equal 
values.

For example, these two Datasets each have an invalid floating-point value in the Key 
column::

    >>> ds1 = rt.Dataset({'Key': [1.0, rt.nan, 2.0],
    ...                   'Value1': ['a', 'b', 'c']})
    >>> ds2 = rt.Dataset({'Key': [1.0, 2.0, rt.nan],
    ...                   'Value2': [1, 2, 3]})

Now we do a ``merge_lookup()`` on the Key columns::

    >>> ds1.merge_lookup(ds2, on='Key')
    #    Key   Value1   Value2
    -   ----   ------   ------
    0   1.00   a             1
    1    nan   b           Inv
    2   2.00   c             2

The ``NaN`` key and its associated value in ``ds2`` were ignored, and the invalid 
integer value was filled in.

Replacing Missing Values
------------------------

For both FastArrays and Datasets, calling ``fillna()`` with a constant
is a quick way to replace missing values::

    >>> ds.fillna(123)
    #   Ints   Floats   FloatsPlus
    -   ----   ------   ----------
    0    123   123.00       123.00
    1      2      inf         3.00
    2      3     2.70         5.40

Note that by default ``fillna()`` returns a copy; to modify the original
data, use ``inplace=True``.

For a little more nuance in how the gaps are filled, use ``fillna()``
with ``method='ffill'`` or ``method='bfill'``.

``fillna(method='ffill')`` propagates non-``NaN`` values forward::

    >>> rt.FA([1.0, 2.0, np.nan, 4.0, 5.0]).fillna(method='ffill')
    FastArray([1., 2., 2., 4., 5.])

``fillna(method='bfill')`` propagates non-NaN values backward::

    >>> rt.FA([1.0, 2.0, np.nan, 4.0, 5.0]).fillna(method='bfill')
    FastArray([1., 2., 4., 4., 5.])

For Categoricals, ``fill_forward()`` and ``fill_backward()`` propagate
values within categories::

    >>> # Create a Categorical with a NaN in each category
    >>> ds = rt.Dataset()
    >>> ds.Cat = rt.Cat(['A', 'B', 'A', 'B', 'A', 'B'])
    >>> ds.x = rt.FA([1, 4, rt.nan, rt.nan, 9, 16])
    >>> ds
    #   Cat       x
    -   ---   -----
    0   A      1.00
    1   B      4.00
    2   A       nan
    3   B       nan
    4   A      9.00
    5   B     16.00

Propagate forward the last encountered non-``NaN`` value for the
category::

    >>> ds.Cat.fill_forward(ds.x)
    *gb_key_0       x
    ---------   -----
    A            1.00
    B            4.00
    A            1.00
    B            4.00
    A            9.00
    B           16.00


Note that until a reported bug is fixed, explicit column name declarations might not be 
displayed for grouping operations.

Propagate backward the next encountered non-NaN value for the category::

    >>> ds.Cat.fill_backward(ds.x)
    *gb_key_0       x
    ---------   -----
    A            1.00
    B            4.00
    A            9.00
    B           16.00
    A            9.00
    B           16.00

Both ``fill_forward()`` and ``fill_backward()`` can take a list of
arrays to fill, and both can modify data in place with ``inplace=True``.

Note that if there is no value available to propagate forward or
backward, the ``NaN`` value isn’t changed::

    >>> ds.x[1] = rt.nan
    >>> ds.Cat.fill_forward(ds.x)
    *gb_key_0       x
    ---------   -----
    A            1.00
    B             nan
    A            1.00
    B             nan
    A            9.00
    B           16.00

Convert Missing Values to/from Pandas
-------------------------------------

This section covers some things to be aware of when you convert data
with missing values between Pandas and Riptable.

Note that while you can convert Pandas DataFrames to Riptable Datasets
using Riptable’s Dataset constructor, you should use the Dataset methods
``to_pandas`` and ``from_pandas`` to convert data with missing values.

Converting Floats
~~~~~~~~~~~~~~~~~

To represent missing floating-point values, both Pandas and Riptable use
the special floating-point ``NaN`` value that’s part of the IEEE
standard (though in Riptable, it’s displayed as ``nan``). Converting
floating-point ``NaN`` values between Pandas and Riptable poses no
issues::

    >>> df = pd.DataFrame({'A': [0.0, np.nan, 1.0]})
    >>> ds = rt.Dataset.from_pandas(df)
    >>> ds
    #      A
    -   ----
    0   0.00
    1    nan
    2   1.00

    >>> df_again = ds.to_pandas()
    >>> df_again
         A
    0  0.0
    1  NaN
    2  1.0

Converting Integers
~~~~~~~~~~~~~~~~~~~

Converting integers gets more interesting. Pandas has a new nullable
integer data type (Int64, not to be confused with NumPy’s int64 dtype).
A missing value in an Int64 column is represented by the native
``pd.NA`` value and displayed as ``<NA>``.

Before this new dtype was created, the only numeric ``NaN`` used by
Pandas was a floating-point ``NaN``, so any ``NaN`` value added to an
integer array in Pandas would cause the array to become an array of
floating-point numbers::

    >>> s1 = pd.Series([1, 2, 3, 4, 5])
    >>> s1[1] = np.nan
    >>> s1
    0    1.0
    1    NaN
    2    3.0
    3    4.0
    4    5.0
    dtype: float64

Since this is now just a column of floats, converting it to Riptable is
just as shown above.

Now, in Pandas, you can specify the new Int64 dtype (it’s not yet used
by default). Missing values are represented by ``pd.NA``, displayed as
``<NA>``::

    >>> s2 = pd.Series([1, 2, 3, 4, 5], dtype='Int64')
    >>> s2[1] = np.nan
    >>> s2
    0       1
    1    <NA>
    2       3
    3       4
    4       5
    dtype: Int64

When we convert these to Riptable, the Int64 ``<NA>`` remains an integer
(but now the int64 dtype)::

    >>> # Create a DataFrame with the series from above.
    >>> df = pd.DataFrame({'Float': s1, 'Int64': s2})
    >>> # Convert the DataFrame to a Riptable Dataset and display its dtypes.
    >>> ds2 = rt.Dataset.from_pandas(df)
    >>> ds2.dtypes
    {'Float': dtype('float64'), 'Int64': dtype('int64')}

When you convert data with missing integer values from Riptable to
Pandas, by default ``to_pandas()`` converts to the new Int64 dtype::

    >>> df_again2 = ds2.to_pandas()
    >>> df_again2.dtypes
    Float    float64
    Int64      Int64
    dtype: object

You can choose to not convert to the new nullable dtype, but your
integers might not be very useful::

    >>> df_again3 = ds2.to_pandas(use_nullable=False)
    >>> df_again3
       Float                Int64
    0    1.0                    1
    1    NaN -9223372036854775808
    2    3.0                    3
    3    4.0                    4
    4    5.0                    5

Converting Datetimes
~~~~~~~~~~~~~~~~~~~~

In Pandas, missing datetime values are represented as ``NaT``. When
those are converted to Riptable, they become an ``Inv``::

    >>> date_arr = pd.Series(pd.to_datetime(['01/01/2022', '02/01/2022', np.nan]))
    >>> df2 = pd.DataFrame({'Timestamp': date_arr})
    >>> ds3 = rt.Dataset.from_pandas(df2)
    >>> ds3
    #                     Timestamp
    -   ---------------------------
    0   20220101 00:00:00.000000000
    1   20220201 00:00:00.000000000
    2                           Inv

The missing value becomes ``NaT`` again when converted back to Pandas::

    >>> df_again3 = ds3.to_pandas()
    >>> df_again3
                      Timestamp
    0 2022-01-01 00:00:00+00:00
    1 2022-02-01 00:00:00+00:00
    2                       NaT

Converting Missing Booleans and Strings from Pandas to Riptable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    >>> str_arr = pd.Series(["aaa", "bbb"])
    >>> bool_arr = pd.Series([True, False])
    >>> df = pd.DataFrame({"Strings": str_arr, "Bools": bool_arr})
    >>> df2 = df.reindex({0, 1, 2})  # Add a row of missing values
    >>> df2
      Strings  Bools
    0     aaa   True
    1     bbb  False
    2     NaN    NaN

When we convert Pandas ``NaN`` strings and Booleans to Riptable, the
results are perhaps not quite what we expect::

    >>> ds = rt.Dataset.from_pandas(df2)
    >>> ds
    #   Strings   Bools
    -   -------   -----
    0   aaa        1.00
    1   bbb        0.00
    2   nan         nan

As you can see, the Boolean column became a column of floating-point
values with an ``rt.nan``. If we try to recast the values, we get an
unexpected result::

    >>> ds.Bools = ds.Bools.astype(bool)
    >>> ds
    #   Strings   Bools
    -   -------   -----
    0   aaa        True
    1   bbb       False
    2   nan        True

As for the “nan” in the Strings column, it is a string literal::

    >>> ds.Strings
    FastArray([b'aaa', b'bbb', b'nan'], dtype='|S3')

One way to avoid getting the string literal is to replace the missing
value in Pandas (with a space, for example). Another way to deal with
these values is to create a Boolean column that’s True if the Pandas
object is a ``NaN``, then use that column as a mask array.

**Riptable NaN values**

-  Int: -MAXINT (signed), MAXINT (unsigned)
-  Float: nan
-  String: b’’
-  Bool: False
-  Date (stored as int): -MAXINT
-  DTN (stored as int): -MAXINT
-  TS (stored as float): nan

Next we cover a few ways to `Instantiate with Placeholder Values and
Generate Sample Data <tutorial_sample_data.rst>`__.

--------------

Questions or comments about this guide? Email
RiptableDocumentation@sig.com.
