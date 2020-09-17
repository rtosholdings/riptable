**Guide to multithreading numpy routines**

This document describes techniques used to speed up the processing of large data arrays in python on Intel x64 based CPUs.

Problems with processing large arrays fast on python.
   1) numpy was not originally written to be multithreaded or SIMD fully aware
   2) pandas was originally written for smaller data
   3) most python data array algorithms expect numpy arrays

To solve the above issues:
1) Write SIMD aware multithreaded algorthms
2) Rethink the pandas dataset class for large data, and follow numpy array rules
3) Subclass from numpy ndarray to still be a numpy array

To multithread algos we categorize operations:
unary: abs, trunc, sqrt, etc.
1) binary: add, sub, divide, etc.
2) logical: &, |, ~, etc.
3) reduce: sum, cumsum, std, etc.
4) comparisons: ==, !=, <, <=, >, >=
5) copy/convert: .copy(), .astype(), upcasting
6) hstack/vstack
7) where, putmask
8) boolean and fancy index getitem/setitem
8) sorting/uniquesness: lexsort, searchsorted, unique (soon sort)
9) datetime routines: strftime, finding the daylight savings
10) linear_interp
11) special grouping routines which dont exist in numpy (like ismember, transform=True, ifirstkey)
12) special hashing routines
13) ema related (moving averages)
14) window related: rolling functions in pandas
15) apply related and numba.prange loops (only way out to JIT multithreaded routines)
16) row major to col major conversions and back (1d to 2d, record array)
17) concept of mask, nan, invalid, or filter (and how to loop over data)

Further numpy has not yet grown into
1) Table/DataFrame/Dataset class
2) Categorical (multikey) class
3) Save and compress large data tables and read back stacked

The groupby.apply or groupby.somefunction (such as cumsum for example) requires that the cumsum operation have at minimum TWO arrays passed in: the array to operate on, and the fancy index for the group.  This is **very important**: loop using a fancy index (not a boolean mask).

