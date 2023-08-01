Riptable Categoricals -- Final dtype of Integer Mapping Array
*************************************************************

.. currentmodule:: riptable

Final dtype from provided mapping code/index array
--------------------------------------------------

If the user provides the integer array of mapping codes, the array will not be 
recast unless:
  - The integer type is unsigned. If a large enough dtype is specified with the ``dtype``
    argument, it will be used; otherwise the smallest possible dtype will be used 
    based on the number of unique categories or the maximum value provided in a mapping.
  - A dtype is specified that's large enough to accommodate the provided codes. If the
    dtype isn't large enough, the array is upcast to the smallest possible dtype that 
    can be used.

Codes with a signed integer dtype::

    >>> codes = rt.FastArray([2, 4, 4, 3, 2, 1, 3, 2, 0, 1, 3, 4, 2, 0, 4, 
    ...                       3, 1, 0, 1, 2, 3, 1, 4, 2, 2, 3, 4, 2, 0, 2], dtype=rt.int64)
    >>> cats = rt.FastArray(["a", "b", "c", "d", "e"])
    
It is unchanged::

    >>> rt.Categorical(codes, categories=cats)
    Categorical([b, d, d, c, b, ..., c, d, b, Filtered, b]) Length: 30
      FastArray([2, 4, 4, 3, 2, ..., 3, 4, 2, 0, 2], dtype=int64) Base Index: 1
      FastArray([b'a', b'b', b'c', b'd', b'e'], dtype='|S1') Unique count: 5

The codes have an unsigned dtype. No ``dtype`` argument is provided, so the smallest 
dtype is found::

    >>> c = rt.Categorical(codes.astype(rt.uint64), categories=cats)
    Categorical([b, d, d, c, b, ..., c, d, b, Filtered, b]) Length: 30
      FastArray([2, 4, 4, 3, 2, ..., 3, 4, 2, 0, 2]) Base Index: 1
      FastArray([b'a', b'b', b'c', b'd', b'e'], dtype='|S1') Unique count: 5

    >>> c._fa.dtype
    dtype('int8')

The codes have an unsigned dtype, and the specified dtype is large enough to be used::

    >>> rt.Categorical(codes.astype(rt.uint8), categories=cats, dtype=rt.int64)
    Categorical([b, d, d, c, b, ..., c, d, b, Filtered, b]) Length: 30
      FastArray([2, 4, 4, 3, 2, ..., 3, 4, 2, 0, 2], dtype=int64) Base Index: 1
      FastArray([b'a', b'b', b'c', b'd', b'e'], dtype='|S1') Unique count: 5

The codes have a signed dtype, and a different dtype is specified that's large enough
to accommodate the provided codes::

    >>> rt.Categorical(codes.astype(rt.int16), categories=cats, dtype=rt.int64)
    Categorical([b, d, d, c, b, ..., c, d, b, Filtered, b]) Length: 30
      FastArray([2, 4, 4, 3, 2, ..., 3, 4, 2, 0, 2], dtype=int64) Base Index: 1
      FastArray([b'a', b'b', b'c', b'd', b'e'], dtype='|S1') Unique count: 5

The codes have a signed dtype, but the specified dtype is too small::

    >>> big_cats = rt.FastArray(['string'+str(i) for i in range(2000)])
    >>> rt.Categorical(codes, big_cats, dtype=rt.int8)
    UserWarning: A type of <class 'riptable.rt_numpy.int8'> was too small, upcasting.
    Categorical([string1, string3, string3, string2, string1, ..., string2, string3, string1, Filtered, string1]) Length: 30
      FastArray([2, 4, 4, 3, 2, ..., 3, 4, 2, 0, 2], dtype=int16) Base Index: 1
      FastArray([b'string0', b'string1', b'string2', b'string3', b'string4', ..., b'string1995', b'string1996', b'string1997', b'string1998', b'string1999'], dtype='|S10') Unique count: 2000


Final dtype from Matlab index array
-----------------------------------

If the index array is from Matlab, it is often floating-point. Unless a dtype is 
specified, the smallest dtype will be found::

No dtype is specified; the smallest usable dtype is found::

    >>> matlab_codes = (codes + 1).astype(rt.float32)
    >>> rt.Categorical(matlab_codes, categories=cats, from_matlab=True)
    Categorical([c, e, e, d, c, ..., d, e, c, a, c]) Length: 30
      FastArray([3, 5, 5, 4, 3, ..., 4, 5, 3, 1, 3], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c', b'd', b'e'], dtype='|S1') Unique count: 5

A dtype is specified that's large enough to be used::

    >>> rt.Categorical(matlab_codes, categories=cats, from_matlab=True, dtype=rt.int64)
    Categorical([c, e, e, d, c, ..., d, e, c, a, c]) Length: 30
      FastArray([3, 5, 5, 4, 3, ..., 4, 5, 3, 1, 3], dtype=int64) Base Index: 1
      FastArray([b'a', b'b', b'c', b'd', b'e'], dtype='|S1') Unique count: 5

Final dtype from strings or strings + categories
------------------------------------------------

A new index array is generated. Unless a dtype is specified, the smallest usable dtype 
will be found.

No dtype specified; the smallest usable dtype is found::

    >>> str_fa = rt.FastArray(["c", "e", "e", "d", "c", "b", "d", "c", "a", "b", 
    ...                        "d", "e", "c", "a", "e", "d", "b", "a", "b", "c", 
    ...                        "d", "b", "e", "c", "c", "d", "e", "c", "a", "c"])
    >>> rt.Categorical(str_fa)
    Categorical([c, e, e, d, c, ..., d, e, c, a, c]) Length: 30
      FastArray([3, 5, 5, 4, 3, ..., 4, 5, 3, 1, 3], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c', b'd', b'e'], dtype='|S1') Unique count: 5

A large enough dtype is specified::

    >>> rt.Categorical(str_fa, dtype=rt.int64)
    Categorical([c, e, e, d, c, ..., d, e, c, a, c]) Length: 30
      FastArray([3, 5, 5, 4, 3, ..., 4, 5, 3, 1, 3], dtype=int64) Base Index: 1
      FastArray([b'a', b'b', b'c', b'd', b'e'], dtype='|S1') Unique count: 5

Final dtype from a multi-key Categorical 
----------------------------------------

This follows the same rules as string construction. Unless a dtype is specified, the 
smallest usable dtype is found.

No dtype specified; the smallest usable dtype is found::

    >>> rt.Categorical([str_fa, codes])
    Categorical([(c, 2), (e, 4), (e, 4), (d, 3), (c, 2), ..., (d, 3), (e, 4), (c, 2), (a, 0), (c, 2)]) Length: 30
      FastArray([1, 2, 2, 3, 1, ..., 3, 2, 1, 5, 1], dtype=int8) Base Index: 1
      {'key_0': FastArray([b'c', b'e', b'd', b'b', b'a'], dtype='|S1'), 'key_1': FastArray([2, 4, 3, 1, 0])} Unique count: 5

A large enough dtype is specified::

    >>> rt.Categorical([str_fa, codes], dtype=rt.int64)
    Categorical([(c, 2), (e, 4), (e, 4), (d, 3), (c, 2), ..., (d, 3), (e, 4), (c, 2), (a, 0), (c, 2)]) Length: 30
      FastArray([1, 2, 2, 3, 1, ..., 3, 2, 1, 5, 1], dtype=int64) Base Index: 1
      {'key_0': FastArray([b'c', b'e', b'd', b'b', b'a'], dtype='|S1'), 'key_1': FastArray([2, 4, 3, 1, 0])} Unique count: 5

Final dtype from a Pandas Categorical 
-------------------------------------

Pandas already attempts to find the smallest dtype during Categorical construction. If
a Riptable Categorical is created from a Pandas Categorical and a dtype is specified, 
Riptable uses the specified dtype.

Construction from Pandas always generates a new array because Riptable adds 1 to the 
indices::

    >>> import pandas as pd
    >>> pdc = pd.Categorical(str_fa)
    >>> pdc._codes
    array([2, 4, 4, 3, 2, 1, 3, 2, 0, 1, 3, 4, 2, 0, 4, 3, 1, 0, 1, 2, 3, 1,
           4, 2, 2, 3, 4, 2, 0, 2], dtype=int8)
    >>> c = rt.Categorical(pdc)
    >>> c
    Categorical([c, e, e, d, c, ..., d, e, c, a, c]) Length: 30
      FastArray([3, 5, 5, 4, 3, ..., 4, 5, 3, 1, 3], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c', b'd', b'e'], dtype='|S1') Unique count: 5

    >>> c = rt.Categorical(pdc, dtype=rt.int32)
    Categorical([c, e, e, d, c, ..., d, e, c, a, c]) Length: 30
      FastArray([3, 5, 5, 4, 3, ..., 4, 5, 3, 1, 3]) Base Index: 1
      FastArray([b'a', b'b', b'c', b'd', b'e'], dtype='|S1') Unique count: 5

