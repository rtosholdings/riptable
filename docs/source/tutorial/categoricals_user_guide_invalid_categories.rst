Riptable Categoricals -- Invalid Categories
*******************************************

.. currentmodule:: riptable

A category set to be invalid at Categorical creation is considered to be NaN in the sense
that `~riptable.rt_categorical.Categorical.isnan` returns `True` for the category, but 
it's mapped to a valid index and not excluded from any operations on the Categorical. 
To exclude values or categories from operations, use the ``filter`` argument.

Note that this behavior differs from `Previous invalid behavior`_.

Warning: If the invalid category isn't in the provided list of unique categories and a 
filter is also provided at Categorical creation, the invalid category also becomes Filtered.

Categorical created from values (no user-provided categories)
-------------------------------------------------------------

Because it's assigned to a regular bin, an invalid category is allowed for base-0 and 
base-1 indexing::

    >>> c = rt.Categorical(["b", "a", "a", "Inv", "c", "a", "b"], invalid="Inv", base_index=0)
    >>> c
    Categorical([b, a, a, Inv, c, a, b]) Length: 7
    FastArray([2, 1, 1, 0, 3, 1, 2]) Base Index: 0
    FastArray([b'Inv', b'a', b'b', b'c'], dtype='|S3') Unique count: 4

    >>> c.isnan()
    FastArray([False, False, False,  True, False, False, False])

    >>> c = rt.Categorical(['b', 'a', 'Inv', 'a'], invalid='Inv')
    >>> c
    Categorical([b, a, Inv, a]) Length: 4
      FastArray([3, 2, 1, 2], dtype=int8) Base Index: 1
      FastArray([b'Inv', b'a', b'b'], dtype='|S3') Unique count: 3

    >>> c.isnan()
    FastArray([False, False,  True, False])


Categorical created from values and user-provided categories
------------------------------------------------------------

If an invalid category is specified, it must also be in the list of unique categories,
otherwise an error is raised::

    >>> # Included.
    >>> c = rt.Categorical(["b", "a", "Inv", "a"], categories=["a", "b", "Inv"], invalid="Inv")
    >>> c
    Categorical([b, a, Inv, a]) Length: 4
      FastArray([2, 1, 3, 1], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'Inv'], dtype='|S3') Unique count: 3

    >>> # Not included.
    >>> try:
    ...     rt.Categorical(["b", "a", "Inv", "a"], categories=["a", "b"], invalid="Inv")
    ... except ValueError as e:
    ...     print("ValueError:", e)
    ValueError: Found values that were not in provided categories: [b'Inv']. The 
    user-supplied categories (second argument) must also contain the invalid item Inv. 
    For example: Categorical(['b','a','Inv','a'], ['a','b','Inv'], invalid='Inv')

Categorical created with a filter
---------------------------------

Be careful when mixing invalid categories and filters. 

If you filter an invalid category, it becomes Filtered and no longer invalid::

    >>> c = rt.Categorical(["Inv", "a", "b", "a"], categories=["Inv", "a", "b"], 
    ...                    filter=rt.FA([False, True, True, True]), invalid="Inv")

    >>> c
    Categorical([Filtered, a, b, a]) Length: 4
      FastArray([0, 2, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'Inv', b'a', b'b'], dtype='|S3') Unique count: 3

    >>> c.isnan()
    FastArray([False, False, False, False])


Warning: If the invalid category *isn't* included in the array of unique cagtegories 
and you *also* provide a filter, the invalid category *also becomes Filtered* even 
if it isn't filtered out directly.

For comparison, here's an example where the invalid category *is* included in the list 
of unique categories and a filter is provided. You get a warning that doesn't apply in 
this case, and the filter is applied::

    >>> c = rt.Categorical(["Inv", "a", "b", "a"], categories=["Inv", "a", "b"], 
    ...                    filter=rt.FA([True, True, False, False]), invalid="Inv")
    UserWarning: Invalid category was set to Inv. If not in provided categories, will 
    also appear as filtered. For example: print(Categorical(['a','a','b'], ['b'], 
    filter=FA([True, True, False]), invalid='a')) -> Filtered, Filtered, Filtered

The second two values are filtered::
    
    >>> c
    Categorical([Inv, a, Filtered, Filtered]) Length: 4
      FastArray([1, 2, 0, 0], dtype=int8) Base Index: 1
      FastArray([b'Inv', b'a', b'b'], dtype='|S3') Unique count: 3

And the invalid category is still invalid::

    >>> c.isnan()
    FastArray([ True, False, False, False])

However, when the invalid category *is not* included in the list of unique categories,
the warning does apply, and the invalid category also becomes Filtered::

    >>> c = rt.Categorical(["Inv", "a", "b", "a"], categories=["a", "b"], 
    ...                    filter=rt.FA([True, True, False, False]), invalid="Inv")
    UserWarning: Invalid category was set to Inv. If not in provided categories, will 
    also appear as filtered. For example: print(Categorical(['a','a','b'], ['b'], 
    filter=FA([True, True, False]), invalid='a')) -> Filtered, Filtered, Filtered

    >>> c
    Categorical([Filtered, a, Filtered, Filtered]) Length: 4
      FastArray([0, 1, 0, 0], dtype=int8) Base Index: 1
      FastArray([b'a', b'b'], dtype='|S1') Unique count: 2

And "Inv" is no longer considered an invalid category::

    >>> c.isnan()
    FastArray([False, False, False, False])

  
Invalid categories are not excluded from operations
---------------------------------------------------

Although invalid categories are recognized by the Categorical 
`~riptable.rt_categorical.Categorical.isnan` method, they are not 
excluded from operations as filtered values and categories are.

Here, "Inv" is invalid and the "b" category is filtered::

    >>> vals = rt.FA(["Inv", "b", "a", "b", "c", "c", "Inv"])
    >>> f = vals != "b"
    >>> c = rt.Categorical(vals, invalid="Inv", filter=f)
    >>> c
    Categorical([Inv, Filtered, a, Filtered, c, c, Inv]) Length: 7
      FastArray([1, 0, 2, 0, 3, 3, 1], dtype=int8) Base Index: 1
      FastArray([b'Inv', b'a', b'c'], dtype='|S3') Unique count: 3

    >>> c.isnan()
    FastArray([ True, False, False, False, False, False,  True])

Create some values to sum and put in a Dataset to see their relationsips to the
catgegories::

    >>> vals = rt.FA([1, 2, 3, 4, 5, 6, 7])
    >>> ds = rt.Dataset({"c": c, "vals": vals})
    >>> ds
    #   c          vals
    -   --------   ----
    0   Inv           1
    1   Filtered      2
    2   a             3
    3   Filtered      4
    4   c             5
    5   c             6
    6   Inv           7

Get the ``nansum``::

    >>> c.nansum(vals)
    *c    vals
    ---   ----
    Inv      8
    a        3
    c       11

The ``showfilter`` argument confirms that only the "b" values were excluded::

    >>> c.nansum(vals, showfilter=True)
    *c         vals
    --------   ----
    Filtered      6
    Inv           8
    a             3
    c            11

If you use the ``filter`` argument with ``nansum`` and filter out an invalid, the
filtered invalid value is excluded from the operation::

    >>> # Filter the first Inv, one of the already-filtered "b"s, and the first "c".
    >>> f2 = rt.FA([False, False, True, True, True, False, True])
    >>> c.nansum(vals, filter=f2, showfilter=True)
    *key_0     col_0
    --------   -----
    Filtered      13
    Inv            7
    a              3
    c              5

If both invalid values are filtered by the ``nansum`` operation, the category still 
appears in the result::

    >>> f3 = rt.FA([False, False, True, True, True, False, False])
    >>> c.nansum(vals, filter=f3)
    *c    vals
    ---   ----
    Inv      0
    a        3
    c        5

And both invalid values are still invalid::

    >>> c.isnan()
    FastArray([ True, False, False, False, False, False,  True])


Previous invalid behavior
-------------------------

Previously, the specified string was used to represent an invalid catgegory when values
missing in the categories list were encountered. The invalid category was mapped to 0 
in the index/codes array.

This is similar to how Pandas works, except that Pandas uses -1 for its NaN index::

    >>> import pandas as pd
    >>> pdc = pd.Categorical(["a", "a", "z", "b", "c"], ["a", "b", "c"])
    >>> pdc
    ['a', 'a', NaN, 'b', 'c']
    Categories (3, object): ['a', 'b', 'c']
    >>> pdc._codes
    array([ 0,  0, -1,  1,  2], dtype=int8)
    >>> pd.Series([1, 1, 1, 1, 1]).groupby(pdc).count()
    a    2
    b    1
    c    1
    dtype: int64


