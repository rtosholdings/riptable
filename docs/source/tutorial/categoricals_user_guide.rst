
Riptable Categoricals User Guide
********************************

.. currentmodule:: riptable

This guide covers a few topics in more depth than the
:doc:`Categoricals </tutorial/tutorial_categoricals>` section of the :doc:`/tutorial/tutorial`
and the API reference docs for the `~rt_categorical.Categorical` class.


.. toctree::
   :maxdepth: 1

   Constructing <categoricals_user_guide_construct>
   Accessing Data <categoricals_user_guide_access_data>
   Indexing <categoricals_user_guide_indexing>
   Filters <categoricals_user_guide_filters>
   Base Index <categoricals_user_guide_base_index>
   Sorting and Display Order <categoricals_user_guide_order>
   Comparisons <categoricals_user_guide_comparisons>
   Final dtype of Integer Code Array <categoricals_user_guide_dtype>
   Invalid Categories <categoricals_user_guide_invalid_categories>
   Bins and Categories <categoricals_user_guide_bins_categories>


Riptable Categoricals have two related uses:

-  They efficiently store string (or other large dtype) arrays that have
   repeated values. The repeated values are partitioned into groups (a.k.a.
   categories), and each group is mapped to an integer. For example, in a 
   Categorical that contains three "AAPL" symbols and four "MSFT" symbols, 
   the data is partitioned into an "AAPL" group that's mapped to 1 and a 
   "MSFT" group that's mapped to 2. This integer mapping allows the data to 
   be stored and operated on more efficiently.
-  They're Riptable's class for doing group operations. A method applied
   to a Categorical is applied to each group separately.

A Categorical is typically created from a list of strings::

    >>> c = rt.Categorical(["a", "a", "b", "a", "c", "c", "b"])
    >>> c
    Categorical([a, a, b, a, c, c, b]) Length: 7
      FastArray([1, 1, 2, 1, 3, 3, 2], dtype=int8) Base Index: 1
      FastArray([b'a', b'b', b'c'], dtype='|S1') Unique count: 3

The output shows:

- The Categorical values. These are made unique to form the categories.
- The integer mapping codes that correspond to the unique categories. Because the integers can be used to index into the Categorical, they're
  also referred to as the indices. Notice that the base index of the array is also shown. 
  By default, the integer index is 1-based; 0 is reserved for Filtered categories. The 
  integer array is an `int8`, `int16`, `int32`, or `int64` array, depending on the number 
  of unique categories or the maximum value provided in a mapping. 
- The unique categories. (Both the categories and their associated integer codes are 
  sometimes called bins.) Each represents a group for groupby operations. The
  categories are held in an array (sorted by default), a dictionary that maps
  integers to strings or strings to integers, or a multi-key dictionary.

Use Categorical objects to perform aggregations over arbitrary arrays of the same
dimension as the Categorical::

    >>> c = rt.Categorical(["a", "a", "b", "a", "c", "c", "b"])
    >>> ints = rt.FA([3, 10, 2, 5, 4, 1, 1])
    >>> flts = rt.FA([1.2, 3.4, 5.6, 4.0, 2.1, 0.6, 11.3])
    >>> c.sum([ints, flts])
    *key_0   col_0   col_1
    ------   -----   -----
    a           18    8.60
    b            3   16.90
    c            5    2.70

Multi-key Categoricals let you create and operate on groupings based on multiple related 
categories::
    
    >>> strs1 = rt.FastArray(["a", "b", "b", "a", "b", "a"])
    >>> strs2 = rt.FastArray(["x", "y", "y", "z", "x", "x"])
    >>> c = rt.Categorical([strs1, strs2])
    >>> c
    Categorical([(a, x), (b, y), (b, y), (a, z), (b, x), (a, x)]) Length: 6
      FastArray([1, 2, 2, 3, 4, 1], dtype=int8) Base Index: 1
      {'key_0': FastArray([b'a', b'b', b'a', b'b'], dtype='|S1'), 'key_1': FastArray([b'x', b'y', b'z', b'x'], dtype='|S1')} Unique count: 4

    >>> c.count()
    *key_0   *key_1   Count
    ------   ------   -----
    a        x            2
    b        y            2
    a        z            1
    b        x            1


To see more ways to create a Categorical, go to 
:doc:`Constructing Categoricals <categoricals_user_guide_construct>`. To see more 
operations on Categoricals, see the
:doc:`Categoricals </tutorial/tutorial_categoricals>` section of the 
:doc:`/tutorial/tutorial`.