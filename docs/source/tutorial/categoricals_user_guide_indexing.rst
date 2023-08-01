
Riptable Categoricals -- Indexing
*********************************

Bracket indexing traverses the FastArray of indices/codes and returns the
corresponding category. 

When a Categorical is indexed with a single integer, the corresponding category is 
returned as a unicode string.

When multiple integers or a boolean array are used, a copy of the Categorical is
returned that has the same categories as the original Categorical but with an
index/code array limited to the selected elements. If you modify the returned subset,
it won't affect the original Categorical.

When a slice is used, the returned Categorical is a view, not a copy. If you modify 
the view, the original Categorical is also modified.

To set a value to a new value, the new value must be already represented in the 
existing categories array.

The following examples use this Categorical::

    >>> c = rt.Categorical(["a", "a", "b", "a", "c", "c", "b"])
    >>> c
    Categorical([a, a, b, a, c, c, b]) Length: 7
      FastArray([1, 1, 2, 1, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

Single integer
--------------
Use bracket indexing to get a single value::

    >>> c[0]
    'a'

    >>> c[1]
    'a'

    >>> c[2]
    'b'

You can also index from the end of the array with negative indices::

    >>> c[-1]
    'b'

    >>> c[-2]
    'c'

Set a value::

    >>> c[0] = "c"
    >>> c
    Categorical([c, a, b, a, c, c, b]) Length: 7
      FastArray([3, 1, 2, 1, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3
      
The value must be already represented in the existing categories array (adding
categories using ``auto_add_categories`` isn't working correctly at the time of this
writing)::

    >>> try:
    ...     c[0] = "d"
    ... except ValueError as e:
    ...     print("ValueError:", e)
    ValueError: Cannot automatically add categories [b'd'] while auto_add_categories is 
    set to False. Set flag to True in Categorical init.

Multiple integers
-----------------
    >>> c
    Categorical([c, a, b, a, c, c, b]) Length: 7
      FastArray([3, 1, 2, 1, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

Pass a list of indices (a fancy index, which also specifies ordering). The returned
Categorical is a copy of the original Categorical::

    >>> c[[0, 2]]
    Categorical([c, b]) Length: 2
      FastArray([3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3
    
    >>> c[[2, 0]]
    Categorical([b, c]) Length: 2
      FastArray([2, 3], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

    >>> c[[-1, 1]]
    Categorical([b, a]) Length: 2
      FastArray([2, 1], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

Or pass an array::

    >>> c[rt.arange(1, 3)]  # Indices 1 and 2.
    Categorical([a, b]) Length: 2
      FastArray([1, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

Set values::

    >>> c[[0, 2]] = "a"
    >>> c
    Categorical([a, a, a, a, c, c, b]) Length: 7
      FastArray([1, 1, 1, 1, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

    >>> c[rt.arange(1, 3)] = "b"
    >>> c
    Categorical([a, b, b, a, c, c, b]) Length: 7
      FastArray([1, 2, 2, 1, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3


Boolean mask array
------------------
    >>> c
    Categorical([a, b, b, a, c, c, b]) Length: 7
      FastArray([1, 2, 2, 1, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

The returned Categorical is a copy of the original Categorical::

    >>> mask = rt.FA([False, True, True, True, True, True, False])
    >>> c[mask]
    Categorical([a, b, a, c, c]) Length: 5
      FastArray([1, 2, 1, 3, 3], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

Set values:: 

    >>> c[mask] = "c"
    >>> c
    Categorical([a, c, c, c, c, c, b]) Length: 7
      FastArray([1, 3, 3, 3, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3


Slice
-----
    >>> c
    Categorical([a, c, c, c, c, c, b]) Length: 7
      FastArray([1, 3, 3, 3, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

The returned Categorical is a view of the original Categorical. Any changes to
the view also modify the original (see below)::

    >>> c[:3]  # Indices 0-2.
    Categorical([a, c, c]) Length: 3
      FastArray([1, 3, 3], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

    >>> c[1:6]  # Indices 1-5.
    Categorical([c, c, c, c, c]) Length: 5
      FastArray([3, 3, 3, 3, 3], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

Set values::

    >>> c[1:6] = "a"
    Categorical([a, a, a, a, a, a, b]) Length: 7
      FastArray([1, 1, 1, 1, 1, 1, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

Slicing returns a view, not a copy. So if you set values in the returned
subset, the original Categorical is modified::

    >>> c2 = c[1:6]
    >>> c2
    Categorical([a, a, a, a, a]) Length: 5
      FastArray([1, 1, 1, 1, 1], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

    >>> c2[1:5] = "c"  # Modify the returned view.
    >>> c2
    Categorical([a, c, c, c, c]) Length: 5
      FastArray([1, 3, 3, 3, 3], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

    >>> c  # The original is also modified.
    Categorical([a, a, c, c, c, c, b]) Length: 7
      FastArray([1, 1, 3, 3, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3