Performance Considerations
==========================

Riptable uses multi-threaded and vectorized operations to work quickly
and efficiently with large amounts of data. However, because memory is a
finite resource, it’s good to keep some things in mind.

Whenever possible:

-  Use universal functions (ufuncs) and ufunc methods. (Ufuncs take
   array inputs and produce array outputs.)
-  To work with a subset of an array, use slicing instead of fancy
   indexing. Fancy indexing creates a copy of the array. Slicing instead
   gives you a “view” of the array, though this means that changes to
   the slice change the original data as well. Another thing to be aware
   of is that a slice creates a reference to the original data. This
   means the original data won’t be cleared from memory until the
   reference is also deleted.
-  In general, be aware of which operations make copies of data. Use
   flags to do operations in place when you can.
-  Avoid filtering entire Datasets using ``ds.filter()``. Use Boolean
   mask arrays, or use filter keyword arguments in operations on
   columns.

   -  When it makes sense, you can use ``ds.filter(inplace=True)`` to
      modify the original Dataset.

-  Avoid string operations (creating strings, parsing strings, etc.).
   When you need to parse a string, use the FastArray string methods and
   try to do it in as few operations as possible.
-  Use Categoricals for string arrays, especially for repeated strings
   or if you’re converting data between Pandas and Riptable.
-  Delete datasets you’re not using. Though be aware that if you have
   any references to the Dataset in other objects (for example, a slice
   or any operation that gives you a “view” of the data), you might not
   actually free up memory.
-  Avoid using ``apply()`` – it’s not a vectorized operation.

--------------

Questions or comments about this guide? Email
RiptableDocumentation@sig.com.
