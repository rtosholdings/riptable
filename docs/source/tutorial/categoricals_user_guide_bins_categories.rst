Riptable Categoricals -- Get Bins from Categories and Vice-Versa
****************************************************************

.. currentmodule:: riptable

You can use the `~riptable.rt_categorical.Categorical.from_bin` method and 
`~riptable.rt_categorical.Categorical.from_category` methods 
to retrieve corresponding index / mapping codes and categories.

Category from bin
-----------------

Provide the bin to get its associated category.

With base index 1::

    >>> c = rt.Categorical(["a", "a", "b", "c", "d"])
    >>> c
    Categorical([a, a, b, c, d]) Length: 5
      FastArray([1, 1, 2, 3, 4], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c', b'd'], dtype='|S1') Unique count: 4

    >>> c.from_bin(1)
    b'a'

With base index 0::

    >>> c = rt.Categorical(["a", "a", "b", "c", "d"], base_index=0)
    >>> c
    Categorical([a, a, b, c, d]) Length: 5
      FastArray([0, 0, 1, 2, 3]) Base Index: 0
      FastArray([b'a', b'b', b'c', b'd'], dtype='|S1') Unique count: 4

    >>> c.from_bin(1)
    b'b'

An out-of-range bin returns an error::

    >>> try:
    ...     c.from_bin(10)
    ... except IndexError as e:
    ...     print("IndexError:", e)
    IndexError: index 9 is out of bounds for axis 0 with size 4

Note that with a base-1 index, ``from_bin(0)`` is also considered out of range::

    >>> try:
    ...     c.from_bin(0)
    ... except ValueError as e:
    ...     print("ValueError:", e)
    ValueError: Bin 0 is out of range for categorical with base index 1

You can also provide the code from a mapping dictionary::

    >>> c = rt.Categorical([10, 20, 20, 30, 40], categories={"a": 10, "b": 20, "c": 30, "d": 40})
    >>> c
    Categorical([a, b, b, c, d]) Length: 5
      FastArray([10, 20, 20, 30, 40]) Base Index: None
      {10:'a', 20:'b', 30:'c', 40:'d'} Unique count: 4
    >>> c.from_bin(40)
    'd'

A multi-key Categorical returns a tuple of unique values::

    >>> c = rt.Categorical([rt.FastArray(["a", "b", "c", "d", "e"]), rt.arange(5)])
    >>> c
    Categorical([(a, 0), (b, 1), (c, 2), (d, 3), (e, 4)]) Length: 5
      FastArray([1, 2, 3, 4, 5], dtype=int8) Base Index: 1
      {'key_0': FastArray([b'a', b'b', b'c', b'd', b'e'], dtype='|S1'), 'key_1': FastArray([0, 1, 2, 3, 4])} Unique count: 5
    >>> c.from_bin(1)
    (b'a', 0)

Bin from category
-----------------

Provide the unique category to get its associated bin. Note that the bin isn't always
an index into the stored category array.

Unicode strings and byte strings are properly translated internally.

With base index 1::

    >>> c = rt.Categorical(["a", "a", "b", "c", "d"])
    >>> c
    Categorical([a, a, b, c, d]) Length: 5
      FastArray([1, 1, 2, 3, 4], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c', b'd'], dtype='|S1') Unique count: 4

    >>> c.from_category("a")
    1

With base index 0::

    >>> c = rt.Categorical(["a", "a", "b", "c", "d"], base_index=0)
    >>> c
    Categorical([a, a, b, c, d]) Length: 5
      FastArray([0, 0, 1, 2, 3]) Base Index: 0
      FastArray([b'a', b'b', b'c', b'd'], dtype='|S1') Unique count: 4

    >>> c.from_category("a")
    0

A non-existent category returns an error::

    >>> try:
    ...    c.from_category("z")
    ... except ValueError as e:
    ...    print("ValueError:", e)
    ValueError: z not found in uniques.

Provide a tuple for a multi-key Categorical::

    >>> c = rt.Categorical([rt.FastArray(["a", "b", "c", "d", "e"]), rt.arange(5)])
    >>> c
    Categorical([(a, 0), (b, 1), (c, 2), (d, 3), (e, 4)]) Length: 5
      FastArray([1, 2, 3, 4, 5], dtype=int8) Base Index: 1
      {'key_0': FastArray([b'a', b'b', b'c', b'd', b'e'], dtype='|S1'), 'key_1': FastArray([0, 1, 2, 3, 4])} Unique count: 5
      
    >>> c.from_category(("d", 3))
    4