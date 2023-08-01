Riptable Categoricals -- Accessing Parts of the Categorical 
***********************************************************

.. currentmodule:: riptable

Use Categorical methods and properties to access the stored data.

Get the array of Categorical values with `~riptable.rt_categorical.Categorical.expand_array`.
Note that because the expansion constructs the complete list of values from the list of 
unique categories, it is an expensive operation::

    >>> c = rt.Categorical(["b", "a", "b", "c", "a", "c", "b"])
    >>> c
    Categorical([b, a, b, c, a, c, b]) Length: 7
      FastArray([2, 1, 2, 3, 1, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

    >>> c.expand_array
    FastArray([b'b', b'a', b'b', b'c', b'a', b'c', b'b'], dtype='|S8')


    >>> c2 = rt.Categorical([10, 0, 0, 5, 5, 10, 0, 0, 5, 0])
    >>> c2
    Categorical([10, 0, 0, 5, 5, 10, 0, 0, 5, 0]) Length: 10
      FastArray([3, 1, 1, 2, 2, 3, 1, 1, 2, 1], dtype=int8) Base Index: 1
      FastArray([ 0,  5, 10]) Unique count: 3

    >>> c2.expand_array
    FastArray([10,  0,  0,  5,  5, 10,  0,  0,  5,  0])

Note that in this base-1 Categorical with an integer mapping array and unique categories 
provided, 0 is mapped to Filtered, 1 is mapped to "b", and 2 is mapped to "a"; there is 
no 3 to be mapped to "c", so it doesn't appear in the expanded array.

    >>> c3 = rt.Categorical([0, 1, 1, 2, 2, 0, 1, 1, 2, 1], categories=["b", "a", "c"])
    >>> c3
    Categorical([Filtered, b, b, a, a, Filtered, b, b, a, b]) Length: 10
      FastArray([0, 1, 1, 2, 2, 0, 1, 1, 2, 1]) Base Index: 1
      FastArray([b'b', b'a', b'c'], dtype='|S1') Unique count: 3

    >>> c3.expand_array
    FastArray([b'Filtered', b'b', b'b', b'a', b'a', b'Filtered', b'b', b'b', b'a', b'b'], dtype='|S8')


Get the integer mapping array with `~riptable.rt_categorical.Categorical._fa`::

    >>> c._fa
    FastArray([2, 1, 2, 3, 1, 3, 2], dtype=int8)

    >>> c2._fa
    FastArray([3, 1, 1, 2, 2, 3, 1, 1, 2, 1], dtype=int8)

    >>> c3._fa
    FastArray([0, 1, 1, 2, 2, 0, 1, 1, 2, 1])


Get the array of unique categories with `~riptable.rt_categorical.Categorical.category_array`::

    >>> c.category_array
    FastArray([b'a', b'b', b'c'], dtype='|S1')

    >>> c2.category_array
    FastArray([ 0,  5, 10])

    >>> c3.category_array
    FastArray([b'b', b'a', b'c'], dtype='|S1')


Note that if you want to use `~riptable.rt_categorical.Categorical._fa` to index into `~riptable.rt_categorical.Categorical.category_array`, you'll need
to subtract 1:

    >>> c.category_array[c._fa[0]-1]
    b'b'

For multi-key Categoricals, use `~riptable.rt_categorical.Categorical.category_dict` to get a dictionary of the two category 
arrays::

    >>> strs = rt.FastArray(["a", "b", "b", "a", "b", "a"])
    >>> ints = rt.FastArray([2, 1, 1, 2, 1, 3])
    >>> c = rt.Categorical([strs, ints]) 
    >>> c
    Categorical([(a, 2), (b, 1), (b, 1), (a, 2), (b, 1), (a, 3)]) Length: 6
      FastArray([1, 2, 2, 1, 2, 3], dtype=int8) Base Index: 1
      {'key_0': FastArray([b'a', b'b', b'a'], dtype='|S1'), 'key_1': FastArray([2, 1, 3])} Unique count: 3

    >>> c.category_dict
    {'key_0': FastArray([b'a', b'b', b'a'], dtype='|S1'),
    'key_1': FastArray([2, 1, 3])}

Use `~riptable.rt_categorical.Categorical.category_mapping` to get the mapping dictionary from a Categorical created with
an :py:class:`~enum.IntEnum` or mapping dictionary::

    >>> d = {"StronglyAgree": 44, "Agree": 133, "Disagree": 75, "StronglyDisagree": 1, "NeitherAgreeNorDisagree": 144 }
    >>> codes = [1, 44, 44, 133, 75]
    >>> c = rt.Categorical(codes, categories=d)
    Categorical([StronglyDisagree, StronglyAgree, StronglyAgree, Agree, Disagree]) Length: 5
      FastArray([  1,  44,  44, 133,  75]) Base Index: None
      {44:'StronglyAgree', 133:'Agree', 75:'Disagree', 1:'StronglyDisagree', 144:'NeitherAgreeNorDisagree'} Unique count: 4
    >>> c.category_mapping
    {44: 'StronglyAgree',
     133: 'Agree',
     75: 'Disagree',
     1: 'StronglyDisagree',
     144: 'NeitherAgreeNorDisagree'}