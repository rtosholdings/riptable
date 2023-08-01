
Riptable Categoricals -- Filtering
**********************************

.. currentmodule:: riptable

Categoricals that use base-1 indexing can be filtered when they're created or anytime 
afterwards. Filters can also be applied on a one-off basis at the time of an operation. 

Values or entire categories can be filtered. Filtered items are mapped to 0 in the 
integer mapping array and omitted from operations.

On this page:

- `Filtering at Categorical creation`_
- `Filtering after Categorical creation`_
- `Filter an operation on a Categorical`_
- `Set a name for filtered values`_
- `See the name set for filtered values`_


Filtering at Categorical creation
---------------------------------

Provide a ``filter`` argument to filter values at Categorical creation. Filtered values
are omitted from all operations on the Categorical.

Notes:

- Only base-1 indexing is supported -- the 0 is reserved for Filtered values. 
- You can't use a dictionary or :py:class:`~enum.IntEnum` to create a Categorical with a filter.

You can filter out certain values or an entire category::

    >>> f = rt.FA([True, True, False, True, True, True, True])  # The mask must be an array, not a list.
    >>> c = rt.Categorical(["a", "a", "b", "a", "c", "c", "b"], filter=f)  # One "b" value is filtered.
    >>> c
    Categorical([a, a, Filtered, a, c, c, b]) Length: 7
      FastArray([1, 1, 0, 1, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

    >>> c.count()
    *key_0   Count
    ------   -----
    a            3
    b            1
    c            2

In the example below, an entire category is filtered. If the Categorical is constructed from values 
without provided categories, categories that are entirely filtered out do not appear 
in the array of unique categories or in the results of operations::

    >>> vals = rt.FA(["a", "a", "b", "a", "c", "c", "b"])
    >>> f = (vals != "b")  # Filter out all "b" values.
    >>> c = rt.Categorical(vals, filter=f)
    >>> c
    Categorical([a, a, Filtered, a, c, c, Filtered]) Length: 7
      FastArray([1, 1, 0, 1, 2, 2, 0], dtype=int8) Base Index: 1
      FastArray([b'a', b'c'], dtype='|S1') Unique count: 2

    >>> c.count()
    *key_0   Count
    ------   -----
    a            3
    c            2

If categories are provided, entirely filtered-out categories do appear in the 
array of unique categories and the results of operations::

    >>> c = rt.Categorical(vals, categories=["a", "b", "c"], filter=f)
    >>> c
    Categorical([a, a, Filtered, a, c, c, Filtered]) Length: 7
      FastArray([1, 1, 0, 1, 3, 3, 0], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

    >>> c.count()
    *key_0   Count
    ------   -----
    a            3
    b            0
    c            2

Multi-key Categoricals can also be filtered at creation.

    >>> f = rt.FA([False, False, True, False, True, True])
    >>> vals1 = rt.FastArray(["a", "b", "b", "a", "b", "a"])
    >>> vals2 = rt.FastArray([2, 1, 1, 3, 2, 1])
    >>> rt.Categorical([vals1, vals2], filter=f)
    Categorical([Filtered, Filtered, (b, 1), Filtered, (b, 2), (a, 1)]) Length: 6
      FastArray([0, 0, 1, 0, 2, 3], dtype=int8) Base Index: 1
      {'key_0': FastArray([b'b', b'b', b'a'], dtype='|S1'), 'key_1': FastArray([1, 2, 1])} Unique count: 3

Categoricals using base-0 indexing can't be filtered at creation::

    >>> f = rt.FA([False, False, True, False, True, True, False])
    >>> try:
    ...    rt.Categorical([0, 1, 1, 2, 2, 0, 1], base_index=0, filter=f)
    ... except ValueError as e:
    ...    print("ValueError:", e)
    ValueError: Filtering is not allowed for base index 0. Use base-1 indexing instead.

Categoricals created using a dictionary or :py:class:`~enum.IntEnum` can't be filtered 
by passing a `filter` argument at creation, but a Filtered category can be included by
by using the integer sentinel value as the Filtered mapping code. They can also be 
filtered after creation using `set_valid()`. 

Using the `filter` argument gets an error::

    >>> f = rt.FA([True, False, False, False, False])
    >>> d = {44: "StronglyAgree", 133: "Agree", 75: "Disagree", 1: "StronglyDisagree", 144: "NeitherAgreeNorDisagree" }
    >>> codes = [1, 44, 144, 133, 75]
    >>> try:
    ...    rt.Categorical(codes, categories=d, filter=f)
    ... except TypeError as e:
    ...    print("TypeError:", e)
    TypeError: Grouping from enum does not support pre-filtering.

However, you can include a Filtered category by using the integer sentinel value in your 
mapping::

    >>> d = {-2147483648: "Filtered", 44: "StronglyAgree", 133: "Agree", 75: "Disagree", 1: "StronglyDisagree", 144: "NeitherAgreeNorDisagree" }
    >>> codes = [-2147483648, 44, 144, 133, 75]
    >>> c = rt.Categorical(codes, categories=d)
    >>> c
    Categorical([Filtered, StronglyAgree, NeitherAgreeNorDisagree, Agree, Disagree]) Length: 5
      FastArray([-2147483648,          44,         144,         133,          75]) Base Index: None
      {-2147483648:'Filtered', 44:'StronglyAgree', 133:'Agree', 75:'Disagree', 1:'StronglyDisagree', 144:'NeitherAgreeNorDisagree'} Unique count: 5

    >>> from enum import IntEnum
    >>> class LikertDecision(IntEnum):
    ...     # A Likert scale with the typical five-level Likert item format.
    ...     Filtered = -2147483648
    ...     StronglyAgree = 44
    ...     Agree = 133
    ...     Disagree = 75
    ...     StronglyDisagree = 1
    ...     NeitherAgreeNorDisagree = 144
    >>> codes = [-2147483648, 1, 44, 144, 133, 75]
    >>> rt.Categorical(codes, categories=LikertDecision)
    Categorical([Filtered, StronglyDisagree, StronglyAgree, NeitherAgreeNorDisagree, Agree, Disagree]) Length: 6
      FastArray([-2147483648,           1,          44,         144,         133,          75]) Base Index: None
      {-2147483648:'Filtered', 44:'StronglyAgree', 133:'Agree', 75:'Disagree', 1:'StronglyDisagree', 144:'NeitherAgreeNorDisagree'} Unique count: 6

You can also filter an existing category after creation using 
`~riptable.rt_categorical.Categorical.set_valid` (see below).


Filtering after Categorical creation
------------------------------------

Calling `~riptable.rt_categorical.Categorical.set_valid` on a Categorical returns a 
filtered copy of the Categorical.

    >>> c = rt.Categorical(["a", "a", "b", "a", "c", "c", "b"])
    >>> c
    Categorical([a, a, b, a, c, c, b]) Length: 7
      FastArray([1, 1, 2, 1, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

    >>> f = rt.FA([True, True, False, True, True, True, True])  # Filter out 1 "b" value.
    >>> c.set_valid(f)
    Categorical([a, a, Filtered, a, c, c, b]) Length: 7
      FastArray([1, 1, 0, 1, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

The original Categorical isn't modified::

    >>> c
    Categorical([a, a, b, a, c, c, b]) Length: 7
      FastArray([1, 1, 2, 1, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

Entirely filtered-out bins are removed from the array of unique categories::

    >>> vals = rt.FA(["a", "a", "b", "a", "c", "c", "b"])
    >>> f = (vals != "b")  # Filter out all "b" values.
    >>> c.set_valid(f)
    Categorical([a, a, Filtered, a, c, c, Filtered]) Length: 7
      FastArray([1, 1, 0, 1, 2, 2, 0], dtype=int8) Base Index: 1
      FastArray([b'a', b'c'], dtype='|S1') Unique count: 2

A Categorical created with a mapping dictionary or :py:class:`~enum.IntEnum` can be
filtered after creation. Filtered values are mapped to the integer sentinel value::

    >>> d = {44: "StronglyAgree", 133: "Agree", 75: "Disagree", 1: "StronglyDisagree", 144: "NeitherAgreeNorDisagree" }
    >>> codes = [1, 44, 144, 133, 75]
    >>> c = rt.Categorical(codes, categories=d)
    >>> c
    Categorical([StronglyDisagree, StronglyAgree, NeitherAgreeNorDisagree, Agree, Disagree]) Length: 5
      FastArray([  1,  44, 144, 133,  75]) Base Index: None
      {44:'StronglyAgree', 133:'Agree', 75:'Disagree', 1:'StronglyDisagree', 144:'NeitherAgreeNorDisagree'} Unique count: 5
    >>> f = rt.FA([False, True, True, True, True])  # Filter out 1: "StronglyDisagree".
    >>> c.set_valid(f)
    Categorical([Filtered, StronglyAgree, NeitherAgreeNorDisagree, Agree, Disagree]) Length: 5
      FastArray([-2147483648,          44,         144,         133,          75]) Base Index: None
      {44:'StronglyAgree', 133:'Agree', 75:'Disagree', 144:'NeitherAgreeNorDisagree', -2147483648:'Filtered'} Unique count: 5

    >>> class LikertDecision(IntEnum):
    ...     # A Likert scale with the typical five-level Likert item format.
    ...     StronglyAgree = 44
    ...     Agree = 133
    ...     Disagree = 75
    ...     StronglyDisagree = 1
    ...     NeitherAgreeNorDisagree = 144
    >>> codes = [1, 44, 144, 133, 75]
    >>> c = rt.Categorical(codes, categories=LikertDecision)
    >>> c
    Categorical([StronglyDisagree, StronglyAgree, NeitherAgreeNorDisagree, Agree, Disagree]) Length: 5
      FastArray([  1,  44, 144, 133,  75]) Base Index: None
      {44:'StronglyAgree', 133:'Agree', 75:'Disagree', 1:'StronglyDisagree', 144:'NeitherAgreeNorDisagree'} Unique count: 5
    >>> f = rt.FA([False, True, True, True, True])  # Filter out 1: "StronglyDisagree".
    >>> c.set_valid(f)
    Categorical([Filtered, StronglyAgree, NeitherAgreeNorDisagree, Agree, Disagree]) Length: 5
      FastArray([-2147483648,          44,         144,         133,          75]) Base Index: None
      {44:'StronglyAgree', 133:'Agree', 75:'Disagree', 144:'NeitherAgreeNorDisagree', -2147483648:'Filtered'} Unique count: 5

Filtering can be useful to re-index a Categorical so only its occurring uniques 
are shown::

    >>> f = (vals != "b")
    >>> c2 = c[f]
    >>> c2
    Categorical([a, a, a, c, c]) Length: 5
      FastArray([1, 1, 1, 3, 3], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

    >>> c2.sum(rt.arange(5))
    *key_0   col_0
    ------   -----
    a            3
    b            0
    c            7

    >>> # Use set_valid to create a re-indexed Categorical:.
    >>> c3 = c2.set_valid()
    >>> c3
    Categorical([a, a, a, c, c]) Length: 5
      FastArray([1, 1, 1, 2, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'c'], dtype='|S1') Unique count: 2

    >>> c3.count()
    *key_0   Count
    ------   -----
    a            3
    c            2

    >>> c3.sum(rt.arange(5))
    *key_0   col_0
    ------   -----
    a            3
    c            7

Filter an operation on a Categorical
------------------------------------

To filter one operation (such as a sum), use the ``filter`` argument for the 
operation. Filtered results are omitted, but any entirely filtered categories still 
appear in the results::

    >>> # Put the Categorical in a Dataset to better see
    >>> # the associated values used in the operation.
    >>> ds = rt.Dataset()
    >>> vals = rt.FA(["a", "a", "b", "a", "c", "c", "b"])
    >>> c = rt.Categorical(vals)
    >>> ds.cats = c
    >>> ds.ints = rt.arange(7)
    >>> ds
    #   cats   ints
    -   ----   ----
    0   a         0
    1   a         1
    2   b         2
    3   a         3
    4   c         4
    5   c         5
    6   b         6

    >>> f = rt.FA([True, True, False, True, True, True, True])  # One "b" value is filtered.
    >>> c.sum(ints, filter=f)
    *key_0   ints
    ------   ----
    a           4
    b           6
    c           9

    >>> f = (cats != "b")  # Filter out all "b" values.
    >>> c.sum(ints, filter=f)
    *key_0   ints
    ------   ----
    a           4
    b           0
    c           9

The Categorical doesn't retain the filter::

    >>> c
    Categorical([a, a, b, a, c, c, b]) Length: 7
      FastArray([1, 1, 2, 1, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

To see the results of the operation applied to all Filtered values (irrespective of
their group), use the ``showfilter`` argument::

    >>> # A "b" value (2) and a "c" value (5) are filtered.
    >>> f = rt.FA([True, True, False, True, True, False, True])
    >>> c.sum(ints, filter=f, showfilter=True)
    *key_0     ints
    --------   ----
    Filtered      7
    a             4
    b             6
    c             4

    >>> f = (cats != "a")  # Filter out all "a" values.
    >>> c.sum(ints, filter=f, showfilter=True)
    *key_0     ints
    --------   ----
    Filtered      4
    a             0
    b             8
    c             9

Set a name for filtered values
------------------------------

You can set a string for displaying filtered values using 
`~riptable.rt_categorical.Categorical.filtered_set_name`::

    >>> vals = rt.FA(["a", "a", "b", "a", "c", "c", "b"])
    >>> f = (vals != "b")
    >>> c = rt.Categorical(vals, filter=f)
    >>> c.filtered_set_name("FNAME")
    >>> c
    Categorical([a, a, FNAME, a, c, c, FNAME]) Length: 7
      FastArray([1, 1, 0, 1, 2, 2, 0], dtype=int8) Base Index: 1
      FastArray([b'a', b'c'], dtype='|S1') Unique count: 2


See the name set for filtered values
------------------------------------

To see the string used when filtered values are displayed, use the 
`~riptable.rt_categorical.Categorical.filtered_name` property::

    >>> c.filtered_name
    'FNAME'
