Concatenate Datasets
====================

Concatentating Datasets is straightforward, with two Dataset methods:
You can concatenate rows (vertically) with ``concat_rows()`` or columns
(horizontally) with ``concat_columns()``.

However, it’s good to be aware of what happens when you concatenate two
Datasets that have different shapes or column names, so we’ll look at a
few examples.

Concatenate Rows
----------------

Concatenating data row-wise is sometimes called vertical stacking. When
two Datasets have identical column names, ``concat_rows()`` simply
stacks the data::

    >>> ds1 = rt.Dataset({'A': ['A0', 'A1', 'A2'], 'B': ['B0', 'B1', 'B2']})
    >>> ds2 = rt.Dataset({'A': ['A3', 'A4', 'A5'], 'B': ['B3', 'B4', 'B5']})
    >>> ds1
    #   A    B 
    -   --   --
    0   A0   B0
    1   A1   B1
    2   A2   B2

    >>> ds2
    #   A    B 
    -   --   --
    0   A3   B3
    1   A4   B4
    2   A5   B5

    >>> rt.Dataset.concat_rows([ds1, ds2])
    #   A    B 
    -   --   --
    0   A0   B0
    1   A1   B1
    2   A2   B2
    3   A3   B3
    4   A4   B4
    5   A5   B5

When the two Datasets have only some column names in common, the result
has a gap in the data::

    >>> # Create two Datasets with two out of three columns in common.
    >>> ds3 = rt.Dataset({'A': ['A0', 'A1', 'A2'], 'B': ['B0', 'B1', 'B2']})
    >>> ds4 = rt.Dataset({'A': ['A3', 'A4', 'A5'], 'B': ['B3', 'B4', 'B5'], 'C': ['C3', 'C4', 'C5'] })
    >>> ds3
    #   A    B 
    -   --   --
    0   A0   B0
    1   A1   B1
    2   A2   B2

    >>> ds4
    #   A    B    C 
    -   --   --   --
    0   A3   B3   C3
    1   A4   B4   C4
    2   A5   B5   C5

    >>> rt.Dataset.concat_rows([ds3, ds4])
    #   A    B    C 
    -   --   --   --
    0   A0   B0     
    1   A1   B1     
    2   A2   B2     
    3   A3   B3   C3
    4   A4   B4   C4
    5   A5   B5   C5

As you can see, Riptable’s missing string value is ’’. If the values
were floats, the empty spots would be filled with ``nan``\ s::

    >>> rng = np.random.default_rng(seed=42)
    >>> ds5 = rt.Dataset({'col_'+str(i):rng.random(2) for i in range(2)}) 
    >>> ds6 = rt.Dataset({'col_'+str(i):rng.random(2) for i in range(3)})
    >>> ds5
    #   col_0   col_1
    -   -----   -----
    0    0.77    0.86
    1    0.44    0.70

    >>> ds6
    #   col_0   col_1   col_2
    -   -----   -----   -----
    0    0.09    0.76    0.13
    1    0.98    0.79    0.45

    >>> rt.Dataset.concat_rows([ds5, ds6])
    #   col_0   col_1   col_2
    -   -----   -----   -----
    0    0.77    0.86     nan
    1    0.44    0.70     nan
    2    0.09    0.76    0.13
    3    0.98    0.79    0.45

See `Working with Missing Data <tutorial_missing_data.rst>`__ for more
about what to expect when you have missing values in Riptable.

You can also concatenate datasets row-wise with Categoricals if the
Datasets have identical column names::


    >>> a = rt.Cat(['a', 'a', 'a', 'b', 'b'])
    >>> b = rt.FA([0, 1, 2, 3, 4])
    >>> ds10 = rt.Dataset({'Cat': a, 'Val': b})
    >>> c = rt.Cat(['c', 'c', 'c', 'd', 'd'])
    >>> d = rt.FA([5, 6, 7, 8, 9])
    >>> ds11 = rt.Dataset({'Cat': c, 'Val': d})
    >>> ds10
    #   Cat   Val
    -   ---   ---
    0   a       0
    1   a       1
    2   a       2
    3   b       3
    4   b       4

    >>> ds11
    #   Cat   Val
    -   ---   ---
    0   c       5
    1   c       6
    2   c       7
    3   d       8
    4   d       9

    >>> rt.Dataset.concat_rows([ds10, ds11])
    #   Cat   Val
    -   ---   ---
    0   a       0
    1   a       1
    2   a       2
    3   b       3
    4   b       4
    5   c       5
    6   c       6
    7   c       7
    8   d       8
    9   d       9

Concatenate Columns
-------------------

Concatenating data column-wise is also called horizontal stacking. It’s
most straightforward when you’re concatenating two Datasets that have no
column names in common::

    >>> ds7 = rt.Dataset({'A': ['A0', 'A1', 'A2'], 'B': ['B0', 'B1', 'B2']})
    >>> ds8 = rt.Dataset({'C': ['C0', 'C1', 'C2'], 'D': ['D0', 'D1', 'D2']})
    >>> ds7
    #   A    B 
    -   --   --
    0   A0   B0
    1   A1   B1
    2   A2   B2

    >>> ds8
    #   C    D 
    -   --   --
    0   C0   D0
    1   C1   D1
    2   C2   D2

    >>> ds9 = rt.Dataset.concat_columns([ds7, ds8], do_copy=True)
    >>> ds9
    #   A    B    C    D 
    -   --   --   --   --
    0   A0   B0   C0   D0
    1   A1   B1   C1   D1
    2   A2   B2   C2   D2

Note that ``do_copy`` is a required argument for ``concat_columns()``.
When ``do_copy=True``, changes you make to values in the original
Datasets do not change the values in your new, concatenated Dataset, and
vice-versa.

When your two Datasets have a column name (or names) in common, you need
to specify which data you want to keep – the data from the shared
column(s) in first Dataset or the data from the shared column(s) in the
second Dataset.

We’ll give our second Dataset an ‘A’ column::

    >>> ds8.A = rt.FA(['A3', 'A4', 'A5'])

If you try to concatenate the two Datasets, you get an error::

    >>> try:
    ...     rt.Dataset.concat_columns([ds7, ds8], do_copy=True)
    ... except KeyError as e:
    ...     print("KeyError:", e)
    KeyError: "Duplicate column 'A'"

To keep the column data from the first Dataset, use
``on_duplicate='first'``. You’ll get a warning about mismatched column
names, but the concatenation is performed::

    >>> rt.Dataset.concat_columns([ds7, ds8], do_copy=True, on_duplicate='first')
    C:\\riptable\\rt_dataset.py:5628: UserWarning: concat_columns() duplicate column mismatch: {'A'}
    warnings.warn(f'concat_columns() duplicate column mismatch: {dups!r}')
    #   A    B    C    D 
    -   --   --   --   --
    0   A0   B0   C0   D0
    1   A1   B1   C1   D1
    2   A2   B2   C2   D2

You can turn off this warning by adding ``on_mismatch='ignore'``.

To keep the column data from the second dataset, use
``on_duplicate='last'``::

    >>> rt.Dataset.concat_columns([ds7, ds8], on_duplicate='last', do_copy=True)
    #   A    B    C    D 
    -   --   --   --   --
    0   A3   B0   C0   D0
    1   A4   B1   C1   D1
    2   A5   B2   C2   D2

Note: To concatenate Datasets column-wise, the columns must all be the
same length – Riptable does not fill in missing column values the way it
does missing row values::

    >>> ds9 = rt.Dataset({'E': ['E0', 'E1']})
    >>> try:
    ...     rt.Dataset.concat_columns([ds8, ds9], do_copy=True)
    ... except ValueError as e:
    ...     print("ValueError:", e)
    ValueError: Inconsistent Dataset lengths {2, 3}

Concatenation is sufficient in certain situations, but it helps to have
more flexibility to bring data from two Datasets together. Next, we’ll
cover how to `Merge Datasets <tutorial_merge.rst>`__.

--------------

Questions or comments about this guide? Email
RiptableDocumentation@sig.com.
