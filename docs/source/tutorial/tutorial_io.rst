Work with Riptable Files and Other File Formats
===============================================

SDS is Riptable’s native file format, and it’s the only data format
fully supported directly within Riptable. That said, there are ways to
get data that’s in other formats in and out of Riptable.

SDS
---

We’ll start with the most straightforward case – saving and loading SDS
files. You can save Datasets, FastArrays, or Structs.

Create a Dataset::

    >>> ds = rt.Dataset({'Ints': rt.arange(10, dtype=int), 'Floats': rt.arange(1, step=0.1), 
    ...                  'Categoricals': rt.Categorical(['a','a','b','a','c','c','b','a','a','b'])})
    >>> ds
    #   Ints   Floats   Categoricals
    -   ----   ------   ------------
    0      0     0.00   a           
    1      1     0.10   a           
    2      2     0.20   b           
    3      3     0.30   a           
    4      4     0.40   c           
    5      5     0.50   c           
    6      6     0.60   b           
    7      7     0.70   a           
    8      8     0.80   a           
    9      9     0.90   b   

Save the Dataset::

    >>> ds.save('ds.sds')

Load the Dataset::

    >>> ds_load_ds = rt.load_sds('ds.sds')
    >>> ds_load_ds
    #   Ints   Floats   Categoricals
    -   ----   ------   ------------
    0      0     0.00   a           
    1      1     0.10   a           
    2      2     0.20   b           
    3      3     0.30   a           
    4      4     0.40   c           
    5      5     0.50   c           
    6      6     0.60   b           
    7      7     0.70   a           
    8      8     0.80   a           
    9      9     0.90   b   

Load a subset of columns::

    >>> rt.load_sds('ds.sds', include=['Ints', 'Categoricals'])
    #   Ints   Categoricals
    -   ----   ------------
    0      0   a           
    1      1   a           
    2      2   b           
    3      3   a           
    4      4   c           
    5      5   c           
    6      6   b           
    7      7   a           
    8      8   a           
    9      9   b 

Create a FastArray::

    >>> fa = rt.FastArray(np.arange(10))
    >>> fa
    FastArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

Save the FastArray::

    >>> fa.save('fa.sds')

Load the FastArray::

    >>> fa_load_sds = rt.load_sds('fa.sds')
    >>> fa_load_sds
    FastArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)

Warning: Multi-key Categoricals can’t be saved in SDS files. When you
try to load the SDS file, it fails with an error: “Categories dict was
empty.”

CSV Files
---------

Saving to CSV isn’t supported by Riptable, but you can do it by first
converting your Riptable Dataset to a Pandas DataFrame, then calling the
Pandas ``to_csv()`` method. Later, you can load your CSV file into
Riptable as a Dataset.

Note that Categorical information will be lost in the ``to_csv()``
process. When you load the CSV file into Riptable as a Dataset, any
Categorical column will be a FastArray. You can always change the
FastArray back into a Categorical in Riptable.

The ``index`` parameter for the ``to_csv()`` method indicates whether
you want to write row (index) names. Because Riptable doesn’t use
explicit row indexing, set ``index=False``.

Convert the Dataset to a Pandas DataFrame, then save the DataFrame as a
CSV::

    >>> ds.to_pandas().to_csv('ds.csv', index=False)

Read the CSV a into Pandas DataFrame, then convert the DataFrame to a
Riptable Dataset using the Dataset constructor::

    >>> ds_from_csv = rt.Dataset(pd.read_csv('ds.csv'))
    >>> ds_from_csv
    #   Ints   Floats   Categoricals
    -   ----   ------   ------------
    0      0     0.00   a           
    1      1     0.10   a           
    2      2     0.20   b           
    3      3     0.30   a           
    4      4     0.40   c           
    5      5     0.50   c           
    6      6     0.60   b           
    7      7     0.70   a           
    8      8     0.80   a           
    9      9     0.90   b   

As you can see, the Categorical is now a FastArray::

    >>> ds_from_csv.Categoricals
    FastArray([b'a', b'a', b'b', b'a', b'c', b'c', b'b', b'a', b'a', b'b'], dtype='|S1')

But we can change it back::

    >>> ds_from_csv.Categoricals = rt.Cat(ds_from_csv.Categoricals)
    >>> ds_from_csv.Categoricals
    Categorical([a, a, b, a, c, c, b, a, a, b]) Length: 10
      FastArray([1, 1, 2, 1, 3, 3, 2, 1, 1, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

SQL Files
---------

Working with SQL files and Riptable is much like working with CSV files
and Riptable. To save a Riptable Dataset to SQL format, first convert
the Dataset to a Pandas DataFrame, then use the Pandas ``to_SQL()``
method to save it.

To get the file back into Riptable, first load it in Pandas as a
DataFrame using ``read_csv()``, then convert it to a Riptable Dataset.

H5 Files
--------

H5 files can be loaded in Riptable using ``rt.load_h5()``. To save your
data as an H5 file, convert to Pandas and use the Pandas ``to_h5()``
method.

NPY Files
---------

Like Pandas, NumPy has various IO tools for saving and loading data. See
the `NumPy
docs <https://numpy.org/doc/stable/user/basics.io.html?highlight=import>`__
for details. Note that Riptable can initialize Datasets only from NumPy
arrays that are record arrays.

Convert data for Use in Other Libraries
---------------------------------------

Sometimes, you need to access a function available only in NumPy or
Pandas. Here’s how to convert a Riptable data structure to its
equivalent in NumPy or Pandas, and then back to Riptable.

Riptable FastArray to/from NumPy Array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When we first introduced FastArrays, we created one from a NumPy array::

    >>> my_fa = rt.FA(np.array([0.1, 0.2, 0.3]))

To access a FastArray’s underlying NumPy array, use ``_np``::

    >>> np_arr = my_fa._np
    >>> np_arr
    array([0.1, 0.2, 0.3])

This is the same result you’d get in Pandas by calling
``Series.values``.

Riptable Dataset to/from NumPy Array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converting a Dataset to a 2-dimensional NumPy array is a two-step
process. First, use ``imatrix_make()`` to convert the Dataset to a
2-dimensional FastArray (``imatrix_make()`` saves only the values – your
column names will be lost). FastArrays above 1-d are not technically
supported by Riptable, so don’t stop here! Convert the FastArray to a
NumPy array with ``._np``::

    >>> ds1 = rt.Dataset({'A':[0,6,9], 'B': [1.2,3.1,9.6], 'C':[-1.6,2.7,4.6], 'D': [2.4,6.2,19.2]})
    >>> np_2d_arr = ds1.imatrix_make()._np
    >>> np_2d_arr
    array([[ 0. ,  1.2, -1.6,  2.4],
           [ 6. ,  3.1,  2.7,  6.2],
           [ 9. ,  9.6,  4.6, 19.2]])

A few things to note about ``imatrix_make()``:

-  As noted above, imatrix_make saves only column values, not column
   names.
-  Non-numerical columns are ignored.
-  You can specify which columns to convert:
   ``ds1[['A', 'B']].imatrix_make()._np``
-  Watch out for integer columns! Since NumPy arrays can’t have mixed
   types, if your ``imatrix_make`` input contains any float columns, the
   entire array will be converted to floats. It’s also possible that the
   integers in your original Dataset will be converted.
-  Also watch out for NaNs in integer columns (“Inv”). “Inv” is stored
   internally by Riptable as an out-of-bounds number, and it will be
   sent to NumPy as that number. See `Working with Missing
   Data <tutorial_missing_data.rst>`__ for more on dealing with NaNs.
-  If there are Categoricals in the Dataset, you can preserve the integer
   mapping codes by passing ``cats=True``.

To convert a 2-dimensional NumPy array back to Riptable, add it to a
Dataset using ``add_matrix()``::

    >>> ds2 = rt.Dataset()
    >>> ds2.add_matrix(np_2d_arr)
    >>> ds2
    #   col_0   col_1   col_2   col_3
    -   -----   -----   -----   -----
    0    0.00    1.20   -1.60    2.40
    1    6.00    3.10    2.70    6.20
    2    9.00    9.60    4.60   19.20

To add it with rows and columns transposed::

    >>> ds3 = rt.Dataset()
    >>> ds3.add_matrix(np_2d_arr.T)
    >>> ds3
    C:\\riptable\\rt_fastarray.py:561: UserWarning: FastArray initialized with strides.
      warnings.warn(warning_string)
    #   col_0   col_1   col_2
    -   -----   -----   -----
    0    0.00    6.00    9.00
    1    1.20    3.10    9.60
    2   -1.60    2.70    4.60
    3    2.40    6.20   19.20

Riptable Dataset to/from Pandas DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generally, you can use ``from_pandas()`` and ``to_pandas()`` to convert
a Pandas DataFrame to a Riptable Dataset and vice-versa.

We’ll create a Pandas DataFrame with categorical, timestamp, float and
integer columns. We won’t deal with NaN values here – see `Working with
Missing Data <tutorial_missing_data.rst>`__ for guidance::

    >>> rng = np.random.default_rng(seed=42)
    >>> N = 10
    >>> dates = pd.date_range('20191111','20191119')
    >>> df = pd.DataFrame( 
    ...     dict(Time = rng.choice(dates, N),
    ...          Symbol = pd.Categorical(rng.choice(['SPY','IBM'], N)),
    ...          Exchange = pd.Categorical(rng.choice(['AMEX','NYSE'], N)),
    ...          TradeSize = rng.choice([1,5,10], N),
    ...          TradePrice = rng.choice([1.1,2.2,3.3], N),
    ...         )
    ... )
    >>> df
            Time Symbol Exchange  TradeSize  TradePrice
    0 2019-11-11    IBM     NYSE          5         1.1
    1 2019-11-17    IBM     AMEX          1         3.3
    2 2019-11-16    IBM     AMEX          1         3.3
    3 2019-11-14    IBM     NYSE          5         2.2
    4 2019-11-14    IBM     NYSE         10         1.1
    5 2019-11-18    IBM     NYSE          1         3.3
    6 2019-11-11    IBM     AMEX         10         2.2
    7 2019-11-17    SPY     NYSE         10         3.3
    8 2019-11-12    IBM     NYSE          1         3.3
    9 2019-11-11    SPY     AMEX          5         3.3

The DataFrame dtypes before conversion::

    >>> df.dtypes
    Time          datetime64[ns]
    Symbol              category
    Exchange            category
    TradeSize              int32
    TradePrice           float64
    dtype: object

Use ``from_pandas()`` to convert to a Dataset::

    >>> ds = rt.Dataset.from_pandas(df)
    >>> ds.head(5)
    #                          Time   Symbol   Exchange   TradeSize   TradePrice
    -   ---------------------------   ------   --------   ---------   ----------
    0   20191111 00:00:00.000000000   IBM      NYSE               5         1.10
    1   20191117 00:00:00.000000000   IBM      AMEX               1         3.30
    2   20191116 00:00:00.000000000   IBM      AMEX               1         3.30
    3   20191114 00:00:00.000000000   IBM      NYSE               5         2.20
    4   20191114 00:00:00.000000000   IBM      NYSE              10         1.10

Note: You can also convert a Pandas DataFrame in the Dataset
constructor, but only if the DataFrame has no null values::

    >>> ds = rt.Dataset(df)

If we check the Dataset dtypes after conversion, we see only the
underlying NumPy data type::

    >>> ds.dtypes
    {'Time': dtype('int64'),
     'Symbol': dtype('int8'),
     'Exchange': dtype('int8'),
     'TradeSize': dtype('int32'),
     'TradePrice': dtype('float64')}

To see the Riptable column types, we’ll use a Python list comprehension::

    >>> {(c,ds[c].dtype ,type(ds[c])) for c in ds.keys()}
    {('Exchange', dtype('int8'), riptable.rt_categorical.Categorical),
     ('Symbol', dtype('int8'), riptable.rt_categorical.Categorical),
     ('Time', dtype('int64'), riptable.rt_datetime.DateTimeNano),
     ('TradePrice', dtype('float64'), riptable.rt_fastarray.FastArray),
     ('TradeSize', dtype('int32'), riptable.rt_fastarray.FastArray)}

Use ``to_pandas()`` to convert the Dataset back to a Pandas DataFrame::

    >>> df1 = ds.to_pandas()
    >>> df1.dtypes
    Time          datetime64[ns, GMT]
    Symbol                   category
    Exchange                 category
    TradeSize                   Int32
    TradePrice                float64
    dtype: object

Convert Dates to/from Matlab (and Other Libraries)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use Matlab (or another library) to visualize data by date, convert
the Riptable Date objects to an array of integers::

    >>> dates = rt.Date(ds.Time) 
    >>> int_dates = dates.yyyymmdd
    >>> int_dates.dtype
    dtype('int32')

MATLAB stores dates as days since 0000-01-01. To convert an array of
Matlab datenums to a Riptable ``Date`` object, first convert the
datenums to a FastArray, then to a Date object using the ``from_matlab``
keyword argument::

    >>> dates = rt.FA([737061.0, 737062.0, 737063.0, 737064.0, 737065.0])
    >>> rt_dates = rt.Date(dates, from_matlab=True)
    >>> rt_dates
    Date(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05'])

Next, we review some things to keep in mind to get the best performance
out of Riptable: `Performance
Considerations <tutorial_performance.rst>`__.

--------------

Questions or comments about this guide? Email
RiptableDocumentation@sig.com.
