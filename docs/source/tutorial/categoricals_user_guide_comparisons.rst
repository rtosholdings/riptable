
Riptable Categoricals -- Comparisons
************************************

.. currentmodule:: riptable

When you provide a value to compare against a Categorical, it's important to remember
that the Categorical may be ordered in a non-lexicographic way. (See 
:doc:`Sorting and Display Order <categoricals_user_guide_order>` for more details.)

For example, this Categorical's categories are held in the order in which they appear
in the provided values (when ``ordered=True``, they are lexicographically sorted instead)::

    >>> c = rt.Categorical([4, 1, 2, 2, 3, 4, 4], ordered=False)
    >>> c
    Categorical([4, 1, 2, 2, 3, 4, 4]) Length: 7
      FastArray([1, 2, 3, 3, 4, 1, 1], dtype=int8) Base Index: 1
      FastArray([4, 1, 2, 3]) Unique count: 4

    >>> c.category_array
    FastArray([4, 1, 2, 3])

The ordering of the category array is the ordering used for comparisons: 4 < 1 < 2 < 3.

So when we check whether each Categorical value is greater than 2::

    >>> c > 2
    FastArray([False, False, False, False,  True, False, False])

... the result reflects that in this Categorical, the only value greater than 2 is 3.

In practice, the comparison is performed on the array of indices. Notice that the 
indices reflect the order of the category array: The 4 category is mapped to 1 in the 
index array, the 1 category is mapped to 2, the 2 category is mapped to 3, and the 3 
category is mapped to 4. 

The value you provide for comparison is converted to its index and compared against 
the index array.

So 2's index, 3, is compared to the other indices::

    >>> 3 < c._fa
    FastArray([False, False, False, False,  True, False, False])

(If the value you provide doesn't have an index because it doesn't exist in the Categorical,
the comparison is adjusted so that it still works out.)

When you provide categories, they are always held in the order they're provided::

    >>> c = rt.Categorical([1, 1, 2, 2, 3, 4, 4], categories=[5, 7, 3, 6])
    >>> c
    Categorical([5, 5, 7, 7, 3, 6, 6]) Length: 7
      FastArray([1, 1, 2, 2, 3, 4, 4]) Base Index: 1
      FastArray([5, 7, 3, 6]) Unique count: 4

    >>> c > 3
    FastArray([False, False, False, False, False,  True,  True])

When you do comparisons with strings, unicode strings and byte strings are properly 
translated internally.

Ordering is that of the provided categories::

    >>> c = rt.Categorical(["b", "a", "b", "a", "c", "c", "b"], categories=["b", "a", "c"])
    >>> c
    Categorical([b, a, b, a, c, c, b]) Length: 7
      FastArray([1, 2, 1, 2, 3, 3, 1], dtype=int8) Base Index: 1
      FastArray([b'b', b'a', b'c'], dtype='|S1') Unique count: 3

    >>> c > "b"
    FastArray([False,  True, False,  True,  True,  True, False])

When categories aren't provided, by default they are sorted lexicographically::

    >>> c = rt.Categorical(["b", "a", "b", "a", "c", "c", "b"])
    >>> c
    Categorical([b, a, b, a, c, c, b]) Length: 7
      FastArray([2, 1, 2, 1, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

    >>> c > "b"
    FastArray([False, False, False, False,  True,  True, False])

The equality operator and `~riptable.rt_categorical.Categorical.isin` are more 
straightforward, and can be used to construct boolean filters based on categories.

  >>> c = rt.Categorical(["b", "a", "b", "a", "c", "c", "b"])
  >>> c
  Categorical([a, a, b, a, c, c, b]) Length: 7
    FastArray([1, 1, 2, 1, 3, 3, 2], dtype=int8) Base Index: 1
    FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

  >>> c == "a"
  FastArray([ True,  True, False,  True, False, False, False])

  >>> c == b"a"
  FastArray([ True,  True, False,  True, False, False, False])

  >>> c.isin("a")
  FastArray([ True,  True, False,  True, False, False, False])

  >>> c.isin(["a", "b"])
  FastArray([ True,  True,  True,  True, False, False,  True])

  >>> c = rt.Categorical([5, 6, 6, 7, 7, 6, 6, 6, 7, 5])
  >>> c
  Categorical([5, 6, 6, 7, 7, 6, 6, 6, 7, 5]) Length: 10
    FastArray([1, 2, 2, 3, 3, 2, 2, 2, 3, 1], dtype=int8) Base Index: 1
    FastArray([5, 6, 7]) Unique count: 3

  >>> c == 1
  FastArray([False, False, False, False, False, False, False, False, False, False])

  >>> c == 6
  FastArray([False,  True,  True, False, False,  True,  True,  True, False, False])

  >>> c.isin(5)
  FastArray([ True, False, False, False, False, False, False, False, False, True])
           
  >>> c.isin([5, 6])
  FastArray([ True,  True,  True, False, False,  True,  True,  True, False, True])

The underlying integer mapping array can also be used for both string and integer
Categoricals::

  >>> c._fa == 2
  FastArray([False,  True,  True, False, False,  True,  True,  True, False, False])