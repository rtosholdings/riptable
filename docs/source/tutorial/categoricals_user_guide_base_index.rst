Riptable Categoricals -- Base Index
***********************************

Categoricals default to base-1 indexing
---------------------------------------

The 0 index is reserved for Filtered values and categories::

    >>> vals = rt.FA(["b", "a", "a", "c", "a", "b"])
    >>> f = rt.FA([False, True, True, True, True, True])
    >>> rt.Categorical(vals, filter=f)
    Categorical([Filtered, a, a, c, a, b]) Length: 6
      FastArray([0, 1, 1, 3, 1, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

Note that "b" doesn't appear in the array of unique categories because it's entirely
filtered out::

    >>> f = (vals != "b")
    >>> rt.Categorical(vals, filter=f)
    Categorical([Filtered, a, a, c, a, Filtered]) Length: 6
      FastArray([0, 1, 1, 2, 1, 0], dtype=int8) Base Index: 1
      FastArray([b'a', b'c'], dtype='|S1') Unique count: 2

Provided indices are assumed to be base-1, with the 0 index indicating invalid 
values::

    >>> cats = rt.FA(["a", "b", "c"])
    >>> rt.Categorical([1, 0, 0, 2, 0, 1], categories=cats])
    Categorical([a, Filtered, Filtered, b, Filtered, a]) Length: 6
      FastArray([1, 0, 0, 2, 0, 1]) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3
    
Matlab also reserves 0 for invalid values, so a Categorical created with
``from_matlab=True`` must have a base-1 index::

    >>> rt.Categorical([0.0, 1.0, 1.0, 3.0, 1.0, 2.0], categories=cats, from_matlab=True)
    Categorical([Filtered, a, a, c, a, b]) Length: 6
      FastArray([0, 1, 1, 3, 1, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

Same with a Categorical converted from Pandas::

    >>> import pandas as pd
    >>> pdc = pd.Categorical(["a", "a", "z", "b", "c"], categories=cats)
    >>> pdc
    ['a', 'a', NaN, 'b', 'c']
    Categories (3, object): ['a', 'b', 'c']
    >>> rt.Categorical(pdc)
    Categorical([a, a, Filtered, b, c]) Length: 5
      FastArray([1, 1, 0, 2, 3], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

Multi-key Categorical::

    >>> f = rt.FA([False, False, True, False, True, True])
    >>> rt.Categorical([rt.FA(["b", "a", "a", "c", "a", "b"]), rt.arange(6)], filter=f)
    Categorical([Filtered, Filtered, (a, 2), Filtered, (a, 4), (b, 5)]) Length: 6
      FastArray([0, 0, 1, 0, 2, 3], dtype=int8) Base Index: 1
      {'key_0': FastArray([b'a', b'a', b'b'], dtype='|S1'), 'key_1': FastArray([2, 4, 5])} Unique count: 3


Categoricals with no base index
-------------------------------

Categoricals created from a mapping dictionary or :py:class:`~enum.IntEnum` have no base index::

    >>> # Integer to string mapping.
    >>> d = {44: "StronglyAgree", 133: "Agree", 75: "Disagree", 1: "StronglyDisagree", 144: "NeitherAgreeNorDisagree" }
    >>> codes = [1, 44, 144, 133, 75]
    >>> rt.Categorical(codes, categories=d)
    Categorical([StronglyDisagree, StronglyAgree, NeitherAgreeNorDisagree, Agree, Disagree]) Length: 5
      FastArray([  1,  44, 144, 133,  75]) Base Index: None
      {44:'StronglyAgree', 133:'Agree', 75:'Disagree', 1:'StronglyDisagree', 144:'NeitherAgreeNorDisagree'} Unique count: 5

    >>> # String to integer mapping.
    >>> d = {"StronglyAgree": 44, "Agree": 133, "Disagree": 75, "StronglyDisagree": 1, "NeitherAgreeNorDisagree": 144 }
    >>> codes = [1, 44, 144, 133, 75]
    >>> rt.Categorical(codes, categories=d)
    Categorical([StronglyDisagree, StronglyAgree, NeitherAgreeNorDisagree, Agree, Disagree]) Length: 5
      FastArray([  1,  44, 144, 133,  75]) Base Index: None
      {44:'StronglyAgree', 133:'Agree', 75:'Disagree', 1:'StronglyDisagree', 144:'NeitherAgreeNorDisagree'} Unique count: 5

    >>> from enum import IntEnum
    >>> class LikertDecision(IntEnum):
    ...     # A Likert scale with the typical five-level Likert item format.
    ...     StronglyAgree = 44
    ...     Agree = 133
    ...     Disagree = 75
    ...     StronglyDisagree = 1
    ...     NeitherAgreeNorDisagree = 144
    >>> codes = [1, 44, 144, 133, 75]
    >>> rt.Categorical(codes, categories=LikertDecision)
    Categorical([StronglyDisagree, StronglyAgree, NeitherAgreeNorDisagree, Agree, Disagree]) Length: 5
      FastArray([  1,  44, 144, 133,  75]) Base Index: None
      {44:'StronglyAgree', 133:'Agree', 75:'Disagree', 1:'StronglyDisagree', 144:'NeitherAgreeNorDisagree'} Unique count: 5

Note: Categoricals that have no base index can't be filtered by passing a `filter`
argument at creation, but they can be filtered by using the integer sentinel value as
the Filtered mapping code. They can also be filtered after creation using 
`~riptable.rt_categorical.Categorical.set_valid`. For examples, see 
:doc:`Filters <categoricals_user_guide_filters>`.

Some Categoricals can opt for base-0 indexing
---------------------------------------------

Base-0 can be used if:

  - A mapping dictionary isn't used. A `Categorical` created from a mapping
    dictionary does not have a base index.
  - A filter isn't used at creation.
  - A Matlab or Pandas Categorical isn't being converted. These both reserve 0
    for invalid values.

::

  >>> rt.Categorical(["b", "a", "a", "c", "a", "b"], base_index=0)
  Categorical([b, a, a, c, a, b]) Length: 6
    FastArray([1, 0, 0, 2, 0, 1]) Base Index: 0
    FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

  >>> rt.Categorical(["b", "a", "a", "c", "a", "b"], categories=cats, base_index=0)
  Categorical([b, a, a, c, a, b]) Length: 6
    FastArray([1, 0, 0, 2, 0, 1], dtype=int8) Base Index: 0
    FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

  >>> rt.Categorical([1, 0, 0, 2, 0, 1], categories=cats, base_index=0)
  Categorical([b, a, a, c, a, b]) Length: 6
    FastArray([1, 0, 0, 2, 0, 1]) Base Index: 0
    FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3


Filtering at Categorical creation prevents base-0 indexing
----------------------------------------------------------

    >>> f = rt.FA([True, True, False, True, True, True])

    >>> try:
    ...     rt.Categorical(["b", "a", "a", "c", "a", "b"], filter=f, base_index=0)
    ... except ValueError as e:
    ...    print("ValueError:", e)
    ValueError: Filtering is not allowed for base index 0. Use base-1 indexing instead.

    >>> try:
    ...     rt.Categorical(["b", "a", "a", "c", "a", "b"], categories=cats, filter=f, base_index=0)
    ... except ValueError as e:
    ...     print("ValueError:", e)
    ValueError: Filtering is not allowed for base index 0. Use base-1 indexing instead.


    >>> try:
    ...     rt.Categorical([1, 0, 0, 2, 0, 1], categories=cats, filter=f, base_index=0) 
    ... except ValueError as e:
    ...     print("ValueError:", e)
    ValueError: Filtering is not allowed for base index 0. Use base-1 indexing instead.


Categoricals created from Matlab or Pandas Categoricals can't use base-0 indexing
---------------------------------------------------------------------------------

Categoricals created from Matlab Categoricals must use a base-1 index in order to 
preserve invalid values (which are also indexed as 0 in Matlab)::

    >>> import pandas as pd
    >>> pdc = pd.Categorical(["b", "a", "a", "c", "a", "b"])
    >>> try:
    ...     rt.Categorical(pdc, base_index=0) 
    ... except ValueError as e:
    ...     print("ValueError:", e)
    ValueError: To preserve invalids, pandas categoricals must be 1-based.

    >>> try:
    ...     rt.Categorical([2.0, 1.0, 1.0, 3.0, 1.0, 2.0], categories=cats, from_matlab=True, base_index=0)
    ... except ValueError as e:
    ...     print("ValueError:", e)
    ValueError: Categoricals from matlab must have a base index of 1, got 0.