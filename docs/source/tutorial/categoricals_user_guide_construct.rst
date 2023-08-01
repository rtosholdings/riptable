Riptable Categoricals -- Constructing 
*************************************

.. currentmodule:: riptable

There are many ways to construct a Categorical -- here are some of the more
common ones.

On this page:

- `From a list of strings`_
- `From a list of non-unique strings and a list of unique categories`_
- `From a list of numeric values that index into a list of string categories`_
- `From a list of numeric values with no categories provided`_
- `From an integer-string dictionary and an array of integer mapping codes`_
- `From an IntEnum and an array of integer mapping codes`_
- `From a list of arrays or a dictionary: a multi-key Categorical`_
- `From a list of float values (Matlab indexing)`_
- `From a Pandas Categorical`_
- `Using the categories of another Categorical`_
- `From an array of values using rt.cut or rt.qcut`_


From a list of strings
----------------------

A Categorical is typically created from a list of strings (unicode or byte strings). 
An array of integer mapping codes is created, along with an array of the unique 
categories::

    >>> c = rt.Categorical(["b", "a", "b", "a", "c", "c", "b"])
    >>> c
    Categorical([b, a, b, a, c, c, b]) Length: 7
      FastArray([2, 1, 2, 1, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

By default, the integer mapping arrary uses base-1 indexing, with 0 reserved for
Filtered values.


From a list of non-unique strings and a list of unique categories
-----------------------------------------------------------------

When categories are provided, they're always held in the same order, so you can
preserve a non-lexicographical ordering. 

The provided categories don't need to represent all of the provided values -- note that 
the "xsmall" category has no index in the mapping array::

    >>> rt.Categorical(["small", "small", "medium", "small", "large", "large", "medium"], 
    ...                categories=["xsmall", "small", "medium", "large"])
    Categorical([small, small, medium, small, large, large, medium]) Length: 7
      FastArray([2, 2, 3, 2, 4, 4, 3], dtype=int8) Base Index: 1
      FastArray([b'xsmall', b'small', b'medium', b'large'], dtype='|S6') Unique count: 4

However, all of the values must appear in the provided categories, otherwise an error 
is raised::

    >>> try:
    ...     rt.Categorical(["small", "small", "medium", "small", "large", "large", "medium", "xlarge"], 
    ...                    categories=["xsmall", "small", "medium", "large"])
    ... except ValueError as e:
    ...     print("ValueError:", e)
    ValueError: Found values that were not in provided categories: [b'xlarge']

From a list of numeric values that index into a list of string categories
-------------------------------------------------------------------------

If you have an array of integers that indexes into an array of provided unique 
categories, the integers are used for the integer mapping array and the categories
are held in the order provided. 

Because this is a base-1 Categorical, 0 is reserved for the Filtered category, and 1 
and 2 are mapped to "small" and "medium", respectively:: 
    
    >>> rt.Categorical([0, 1, 1, 2, 2, 0, 1, 1, 2, 1], categories=["small", "medium", "large"])
    Categorical([Filtered, small, small, medium, medium, Filtered, small, small, medium, small]) Length: 10
      FastArray([0, 1, 1, 2, 2, 0, 1, 1, 2, 1]) Base Index: 1
      FastArray([b'small', b'medium', b'large'], dtype='|S6') Unique count: 3

You can set ``base_index=0`` to make the 0 not Filtered::

    >>> rt.Categorical([0, 1, 1, 2, 2, 0, 1, 1, 2, 1], categories=["small", "medium", "large"], base_index=0)
    Categorical([small, medium, medium, large, large, small, medium, medium, large, medium]) Length: 10
      FastArray([0, 1, 1, 2, 2, 0, 1, 1, 2, 1]) Base Index: 0
      FastArray([b'small', b'medium', b'large'], dtype='|S6') Unique count: 3

From a list of numeric values with no categories provided
---------------------------------------------------------

Note that when no categories are provided, the integer mapping codes start at 1 
so that 0 values are not Filtered::

    >>> rt.Categorical([10, 0, 0, 5, 5, 10, 0, 0, 5, 0])
    Categorical([10, 0, 0, 5, 5, 10, 0, 0, 5, 0]) Length: 10
      FastArray([3, 1, 1, 2, 2, 3, 1, 1, 2, 1], dtype=int8) Base Index: 1
      FastArray([ 0,  5, 10]) Unique count: 3

From an integer-string dictionary and an array of integer mapping codes
-----------------------------------------------------------------------

A dictionary can be used for the ``categories`` argument to provide a mapping 
between possibly non-consecutive or non-sequential mapping codes and strings. The 
dictionary can map integers to strings or string to integers. 

Provide a list of integer mapping codes as the first argument to the constructor (notice
here that the provided codes have duplication and a missing entry):: 

    >>> # Integer to string mapping.
    >>> d = {44: "StronglyAgree", 133: "Agree", 75: "Disagree", 1: "StronglyDisagree", 144: "NeitherAgreeNorDisagree" }
    >>> codes = [1, 44, 44, 133, 75]
    >>> rt.Categorical(codes, categories=d)
    Categorical([StronglyDisagree, StronglyAgree, StronglyAgree, Agree, Disagree]) Length: 5
      FastArray([  1,  44,  44, 133,  75]) Base Index: None
      {44:'StronglyAgree', 133:'Agree', 75:'Disagree', 1:'StronglyDisagree', 144:'NeitherAgreeNorDisagree'} Unique count: 4

    >>> # String to integer mapping.
    >>> d = {"StronglyAgree": 44, "Agree": 133, "Disagree": 75, "StronglyDisagree": 1, "NeitherAgreeNorDisagree": 144 }
    >>> codes = [1, 44, 44, 133, 75]
    >>> c = rt.Categorical(codes, categories=d)
    >>> c
    Categorical([StronglyDisagree, StronglyAgree, StronglyAgree, Agree, Disagree]) Length: 5
      FastArray([  1,  44,  44, 133,  75]) Base Index: None
      {44:'StronglyAgree', 133:'Agree', 75:'Disagree', 1:'StronglyDisagree', 144:'NeitherAgreeNorDisagree'} Unique count: 4

Note that Categoricals created from a mapping dictionary have no base index. To see
how this affects filtering, see the page on :doc:`Filters <categoricals_user_guide_filters>`.

Also note that groupby results are displayed *not* in the order of the provided mapping 
dictionary, but the order of the underlying mapping codes array, unless you set 
``sort_gb=True`` at Categorical creation::

    >>> vals = rt.arange(5)
    >>> ds = rt.Dataset({"c": c, "vals": vals})
    >>> ds
    #   c                  vals
    -   ----------------   ----
    0   StronglyDisagree      0
    1   StronglyAgree         1
    2   StronglyAgree         2
    3   Agree                 3
    4   Disagree              5

    >>> c.sum(vals)
    *c                vals
    ---------------   ----
    StronglyDisagre      0
    StronglyAgree        3
    Agree                3
    Disagree             4


See :doc:`Sorting and Display Order <categoricals_user_guide_order>` for examples.


From an IntEnum and an array of integer mapping codes
-----------------------------------------------------

Similar to a dictionary, a Python :py:class:`~enum.IntEnum` class defines a mapping 
between strings and possibly non-consecutive, non-sequential integer mapping codes. 
Similarly, the list of the integer codes is supplied as the first argument to the 
constructor, and the :py:class:`~enum.IntEnum` is provided as the ``categories`` 
argument:: 

    >>> from enum import IntEnum
    >>> class LikertDecision(IntEnum):
    ...     # A Likert scale with the typical five-level Likert item format.
    ...     StronglyAgree = 44
    ...     Agree = 133
    ...     Disagree = 75
    ...     StronglyDisagree = 1
    ...     NeitherAgreeNorDisagree = 144

    >>> codes = [1, 44, 44, 133, 75]
    >>> c = rt.Categorical(codes, categories=LikertDecision)
    >>> c
    Categorical([StronglyDisagree, StronglyAgree, StronglyAgree, Agree, Disagree]) Length: 5
      FastArray([  1,  44,  44, 133,  75]) Base Index: None
      {44:'StronglyAgree', 133:'Agree', 75:'Disagree', 1:'StronglyDisagree', 144:'NeitherAgreeNorDisagree'} Unique count: 4

As with Categoricals created from dictionaries, a Categorical created from an 
:py:class:`~enum.IntEnum` has no base index. To see how this affects filtering, see the 
page on :doc:`Filters <categoricals_user_guide_filters>`.

Also similarly, aggregation results are displayed in the order of the mapping codes 
unless you set ``sort_gb=True`` at Categorical creation::

    >>> c.sum(vals)
    *key_0            vals
    ---------------   ----
    StronglyDisagre      1
    StronglyAgree        5
    Agree                4
    Disagree             5

See :doc:`Sorting and Display Order <categoricals_user_guide_order>` for examples.


From a list of arrays or a dictionary: a multi-key Categorical
--------------------------------------------------------------

Multi-key Categoricals let you create and operate on groupings based on multiple 
associated categories. The associated keys form a group::

    >>> strs = rt.FastArray(["a", "b", "b", "a", "b", "a"])
    >>> ints = rt.FastArray([2, 1, 1, 2, 1, 3])
    >>> c = rt.Categorical([strs, ints])  # Create with a list of arrays.
    >>> c
    Categorical([(a, 2), (b, 1), (b, 1), (a, 2), (b, 1), (a, 3)]) Length: 6
      FastArray([1, 2, 2, 1, 2, 3], dtype=int8) Base Index: 1
      {'key_0': FastArray([b'a', b'b', b'a'], dtype='|S1'), 'key_1': FastArray([2, 1, 3])} Unique count: 3

    >>> c.count()
    *key_0   *key_1   Count
    ------   ------   -----
    a             2       2
    b             1       3
    a             3       1

    >>> c2 = rt.Categorical({"Key1": strs, "Key2": ints})  # Create with a dict of key-value pairs.
    >>> c2
    Categorical([(a, 2), (b, 1), (b, 1), (a, 2), (b, 1), (a, 3)]) Length: 6
      FastArray([1, 2, 2, 1, 2, 3], dtype=int8) Base Index: 1
      {'Key1': FastArray([b'a', b'b', b'a'], dtype='|S1'), 'Key2': FastArray([2, 1, 3])} Unique count: 3

    >>> c2.count()
    *Key1   *Key2   Count
    -----   -----   -----
    a           2       2
    b           1       3
    a           3       1

From a list of float values (Matlab indexing)
---------------------------------------------

To convert a Matlab Categorical that uses float indices, set ``from_matlab=True``. The 
indices are converted to an integer type, and any 0.0 values are Filtered::

    >>> rt.Categorical([0.0, 1.0, 2.0, 3.0, 1.0, 1.0], categories=["b", "c", "a"], from_matlab=True)
    Categorical([Filtered, b, c, a, b, b]) Length: 6
      FastArray([0, 1, 2, 3, 1, 1], dtype=int8) Base Index: 1
      FastArray([b'b', b'c', b'a'], dtype='|S1') Unique count: 3

From a Pandas Categorical
-------------------------

Categoricals created from Pandas Categoricals must have a base-1 index to preserve
invalid values. The invalid values become Filtered::

    >>> import pandas as pd
    >>> pdc = pd.Categorical(["a", "a", "z", "b", "c"], ["c", "b", "a"])
    >>> pdc
    ['a', 'a', NaN, 'b', 'c']
    Categories (3, object): ['c', 'b', 'a']

    >>> rt.Categorical(pdc)
    Categorical([a, a, Filtered, b, c]) Length: 5
      FastArray([3, 3, 0, 2, 1], dtype=int8) Base Index: 1
      FastArray([b'c', b'b', b'a'], dtype='|S1') Unique count: 3

Using the categories of another Categorical
-------------------------------------------

    >>> c = rt.Categorical(["a", "a", "b", "a", "c", "c", "b"], categories=["c", "b", "a"])
    >>> c.category_array
    FastArray([b'c', b'b', b'a'], dtype='|S1')

    >>> c2 = rt.Categorical(["b", "c", "c", "b"], categories=c.category_array)
    >>> c2
    Categorical([b, c, c, b]) Length: 4
      FastArray([2, 1, 1, 2], dtype=int8) Base Index: 1
      FastArray([b'c', b'b', b'a'], dtype='|S1') Unique count: 3

Note that the ``c2.category_array`` has the same values as ``c.category_array``, but
it is a copy of and not a reference to the latter::

    >>> c.category_array is c2.category_array
    False

To create a `Categorical` that references the same categorical array, it must be
constructed with indices and categories::

    >>> c2 = rt.Categorical([1, 2, 1, 2], categories=c.category_array)
    >>> c.category_array is c2.category_array
    True

From an array of values using ``rt.cut`` or ``rt.qcut``
-------------------------------------------------------

Both `cut` and `qcut` partition values into discrete bins that form the categories
of a Categorical. 

With `cut`, values can be parititioned into a specified number of equal-width bins
or bins bounded by specified endpoints. Here, they're parititioned into 3 equal-width
bins::

    >>> rt.cut(x=rt.FA([1, 7, 5, 4, 6, 3]), bins=3)
    Categorical([1.0->3.0, 5.0->7.0, 3.0->5.0, 3.0->5.0, 5.0->7.0, 1.0->3.0]) Length: 6
      FastArray([1, 3, 2, 2, 3, 1], dtype=int8) Base Index: 1
      FastArray([b'1.0->3.0', b'3.0->5.0', b'5.0->7.0'], dtype='|S8') Unique count: 3

Here the bins are bounded by specified endpoints. Values that fall outside of the bins 
are put in the Filtered category::

    rt.cut(x=rt.FA([1, 7, 5, 4, 6, 3]), bins=[1, 3, 6])
    Categorical([1.0->3.0, Filtered, 3.0->6.0, 3.0->6.0, 3.0->6.0, 1.0->3.0]) Length: 6
      FastArray([1, 0, 2, 2, 2, 1], dtype=int8) Base Index: 1
      FastArray([b'1.0->3.0', b'3.0->6.0'], dtype='|S8') Unique count: 2


The `qcut` function lets you partition values into bins based on sample quantiles::

    >>> rt.qcut(rt.arange(5), q=4)
    Categorical([0.0->1.0, 0.0->1.0, 1.0->2.0, 2.0->3.0, 3.0->4.0]) Length: 5
      FastArray([2, 2, 3, 4, 5], dtype=int8) Base Index: 1
      FastArray([b'Clipped', b'0.0->1.0', b'1.0->2.0', b'2.0->3.0', b'3.0->4.0'], dtype='|S8') Unique count: 5

The 'Clipped' bin is created to hold any out-of-bounds values, such as when a value falls
outside of a specified range. A 'Clipped' bin is different from a 'Filtered' bin::

    >>> rt.qcut(rt.arange(5), q=[.1, .25, .5, .75, 1.], filter=[True, False, True, True, True])
    Categorical([Clipped, Filtered, 1.5->2.5, 2.5->3.25, 3.25->4.0]) Length: 5
      FastArray([1, 0, 3, 4, 5], dtype=int8) Base Index: 1
      FastArray([b'Clipped', b'0.6->1.5', b'1.5->2.5', b'2.5->3.25', b'3.25->4.0'], dtype='|S9') Unique count: 5