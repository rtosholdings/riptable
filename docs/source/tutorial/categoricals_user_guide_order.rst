Riptable Categoricals -- Sorting and Display Order
**************************************************

.. currentmodule:: riptable

Whether a Categorical's categories are lexicographically sorted or considered to be 
"ordered" as specified at creation depends on two parameters: ``ordered`` and ``lex``. 

- ``ordered`` controls whether categories are sorted lexicographically before they are 
  mapped to integers.
- ``lex`` controls whether hashing- or sorting-based logic is used to find unique 
  values in the input array. Note that if ``lex=True``, the categories become sorted 
  even if ``ordered=False``.

Additionally, the results of groupby operations can be displayed in sorted order with
the ``sort_gb`` parameter.

The way these parameters interact depends on whether categories are provided when the
Categorical is created.

**If categories are not provided,** then if ``ordered=True`` (the default) or ``lex=True``
they are sorted in the Categorical and in groupby results, even if ``sort_gb=False``.
If ``ordered=False`` and ``lex=False``, the categories are held in the order of first
appearance, and groupby results are sorted only if ``sort_gb=True``.

+---------+-----+--------------------+-------------------------+
| ordered | lex | categories sorted? | groupby results sorted? |
+=========+=====+====================+=========================+
|    T    |  T  |         Y          |            Y            |
+---------+-----+--------------------+-------------------------+
|    T    |  F  |         Y          |            Y            |
+---------+-----+--------------------+-------------------------+
|    F    |  T  |         Y          |            Y            |
+---------+-----+--------------------+-------------------------+
|    F    |  F  |         N          |  only if sort_gb=True   |
+---------+-----+--------------------+-------------------------+

**If categories are provided,** they are always held in the same order. The ``ordered``
argument is ignored, and ``lex`` can't be specified. Groupby results can be displayed 
in sorted order with ``sort_gb=True``.


Categorical created from values (no user-provided categories)
-------------------------------------------------------------

With the default ``ordered=True``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Categories are sorted.
- Groupby results are sorted regardless of whether ``sort_gb=True``.

::

    >>> vals = rt.arange(6)
    >>> c = rt.Categorical(["b", "a", "a", "c", "a", "b"])
    Categorical([b, a, a, c, a, b]) Length: 6
      FastArray([2, 1, 1, 3, 1, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

    >>> c.sum(vals)
    *key_0   col_0
    ------   -----
    a            7
    b            5
    c            3


With ``ordered=False``
^^^^^^^^^^^^^^^^^^^^^^

- Categories are not sorted unless ``lex=True``. Setting ``lex=True`` causes a
  lexicographical sort to be performed to find the uniques, and the Categorical's
  categories become sorted even if ``ordered=False``.
- Groupby results are not displayed sorted unless ``lex=True`` or ``sort_gb=True``.

::

    >>> vals = rt.arange(6)
    >>> c = rt.Categorical(["b", "a", "a", "c", "a", "b"], ordered=False)
    >>> c
    Categorical([b, a, a, c, a, b]) Length: 6
      FastArray([1, 2, 2, 3, 2, 1], dtype=int8) Base Index: 1
      FastArray([b'b', b'a', b'c'], dtype='|S1') Unique count: 3

    >>> c.sum(vals)
    *key_0   col_0
    ------   -----
    b            5
    a            7
    c            3

Here, ``lex=True`` causes the categories to become sorted even though ``ordered=False``.

    >>> c = rt.Categorical(["b", "a", "a", "c", "a", "b"], ordered=False, lex=True)
    >>> c
    Categorical([b, a, a, c, a, b]) Length: 6
      FastArray([2, 1, 1, 3, 1, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

    >>> c.sum(vals)
    *key_0   col_0
    ------   -----
    a            7
    b            5
    c            3

    >>> c = rt.Categorical(["b", "a", "a", "c", "a", "b"], ordered=False, sort_gb=True)
    >>> c
    Categorical([b, a, a, c, a, b]) Length: 6
      FastArray([1, 2, 2, 3, 2, 1], dtype=int8) Base Index: 1
      FastArray([b'b', b'a', b'c'], dtype='|S1') Unique count: 3

    >>> c.sum(vals)
    *key_0   col_0
    ------   -----
    a            7
    b            5
    c            3


Categorical created from values and user-provided categories (unsorted)
-----------------------------------------------------------------------

- If categories are provided, they are always held in the same order. The ``ordered``
  argument is ignored, and you can't set ``lex=True``.
- Groupby results can be displayed in sorted order with ``sort_gb=True``.

Categorical with unsorted categories::

    >>> c = rt.Categorical(["b", "a", "a", "c", "a", "b"], categories=["b", "a", "c"])
    >>> c
    Categorical([b, a, a, c, a, b]) Length: 6
      FastArray([1, 2, 2, 3, 2, 1], dtype=int8) Base Index: 1
      FastArray([b'b', b'a', b'c'], dtype='|S1') Unique count: 3

Groupby results are in the order of provided categories::

    >>> vals = rt.arange(6)
    >>> c.sum(vals)
    *key_0   col_0
    ------   -----
    b            5
    a            7
    c            3

With provided categories, ``lex`` can't be set to `True`::

    >>> try:
    ...  rt.Categorical(["b", "a", "a", "c", "a", "b"], categories=["b", "a", "c"], lex=True)
    ... except TypeError as e:
    ...    print("TypeError:", e)
    TypeError: Cannot bin using lexsort and user-suplied categories.


With ``sort_gb=True``, categories are held in the order provided but displayed 
lexicographically sorted in groupby results::

    >>> c = rt.Categorical(["b", "a", "a", "c", "a", "b"], categories=["b", "a", "c"], sort_gb=True)
    >>> c
    Categorical([b, a, a, c, a, b]) Length: 6
      FastArray([1, 2, 2, 3, 2, 1], dtype=int8) Base Index: 1
      FastArray([b'b', b'a', b'c'], dtype='|S1') Unique count: 3

    >>> c.sum(vals)
    *key_0   col_0
    ------   -----
    a            7
    b            5
    c            3

If the categories are provided in a mapping dictionary or :py:class:`~enum.IntEnum`, the
groupby results are in the order of the underlying mapping codes array unless 
``sort_gb=True``:: 

    >>> d = {"StronglyAgree": 44, "Agree": 133, "Disagree": 75, "StronglyDisagree": 1, "NeitherAgreeNorDisagree": 144 }
    >>> codes = [1, 44, 44, 133, 75]  # Note duplication and missing entry.
    >>> c = rt.Categorical(codes, categories=d)
    >>> c
    Categorical([StronglyDisagree, StronglyAgree, StronglyAgree, Agree, Disagree]) Length: 5
      FastArray([  1,  44,  44, 133,  75]) Base Index: None
      {44:'StronglyAgree', 133:'Agree', 75:'Disagree', 1:'StronglyDisagree', 144:'NeitherAgreeNorDisagree'} Unique count: 4
    >>> vals = rt.arange(5)
    >>> ds = rt.Dataset({"c": c, "vals": vals})
    >>> ds
    #   c                  vals
    -   ----------------   ----
    0   StronglyDisagree      0
    1   StronglyAgree         1
    2   StronglyAgree         2
    3   Agree                 3
    4   Disagree              4

    >>> c.sum(vals)
    *c                vals
    ---------------   ----
    StronglyDisagre      0
    StronglyAgree        3
    Agree                3
    Disagree             4


With ``sort_gb=True``, categories are displayed lexicographically sorted in groupby 
results::

    >>> c = rt.Categorical(codes, categories=d, sort_gb=True)
    >>> c
    Categorical([StronglyDisagree, StronglyAgree, StronglyAgree, Agree, Disagree]) Length: 5
      FastArray([  1,  44,  44, 133,  75]) Base Index: None
      {44:'StronglyAgree', 133:'Agree', 75:'Disagree', 1:'StronglyDisagree', 144:'NeitherAgreeNorDisagree'} Unique count: 4

    >>> c.sum(vals)
    *key_0            vals
    ---------------   ----
    Agree                3
    Disagree             4
    StronglyAgree        3
    StronglyDisagre      0


Ordering of results from rt.cut and rt.qcut operations
------------------------------------------------------

With `cut` and `qcut`, when labels are provided they are held and displayed 
in the order of first appearance and are considered ordered in the context of 
logical comparisons:: 

    >>> c = rt.cut(rt.arange(10), bins=3, labels=["z-label1", "y-label2", "x-label3"])
    >>> c
    Categorical([z-label1, z-label1, z-label1, z-label1, y-label2, y-label2, y-label2, x-label3, x-label3, x-label3]) Length: 10
      FastArray([1, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=int8) Base Index: 1
      FastArray([b'z-label1', b'y-label2', b'x-label3'], dtype='|S8') Unique count: 3

    >>> c.sum(rt.arange(10))
    *key_0     col_0
    --------   -----
    z-label1       6
    y-label2      15
    x-label3      24

    >>> c > "z-label1"
    FastArray([False, False, False, False,  True,  True,  True,  True,  True, True])

See :doc:`Comparisons <categoricals_user_guide_comparisons>` for more examples.