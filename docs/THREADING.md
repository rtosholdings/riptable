**multithreading numpy routines**

This document describes techniques used to speed up the processing of large data arrays in python on Intel x64 based CPUs.

Problems we encountered when processing large arrays fast on python.
   1) numpy was not originally written to be multithreaded or SIMD fully aware
   2) pandas was not originally written for large data (e.g. the block manager)
   3) most python data array algorithms expect numpy arrays

To solve the above issues:
1) Write low level SIMD aware multithreaded algorthms
2) Rethink the pandas dataset class for large data, and follow numpy array rules
3) Subclass from numpy ndarray to still be a numpy array

To multithread algos we categorized the operations:
1) unary: abs, trunc, sqrt, etc.
2) binary: add, sub, divide, etc.
3) logical: &, |, ~, etc.
4) reduce: sum, cumsum, std, etc.
5) comparisons: ==, !=, <, <=, >, >=
6) copy/convert: .copy(), .astype(), upcasting
7) hstack/vstack
8) where, putmask
9) boolean and fancy index getitem/setitem
10) sorting/uniqueness: lexsort, searchsorted, unique (soon sort)
11) datetime routines: strftime, finding the daylight savings
12) interpolations, weighted averages: linear_interp
13) special grouping routines which dont exist in numpy (like ismember, transform=True, ifirstkey)
14) special hashing routines, binning, cuts
15) ema related (moving averages)
16) window related: rolling functions in pandas
17) apply related and numba.prange loops (only way out to JIT multithreaded routines)
18) row major to col major conversions and back (1d to 2d, record array)
19) concept of mask, nan, invalid, or filter (and how to loop over data)
20) fills - forward fills, backward fills, fill_na, group fills
21) precanned arrays: arange, zeros, ones, random, gaussian

numpy will hopefully grow into
1) Table/DataFrame/Dataset class
2) Categorical (multikey) class
3) Save and compress large data tables and read back stacked

The groupby.apply or groupby.somefunction (such as cumsum for example) requires that the cumsum operation have at minimum TWO arrays passed in: the array to operate on, and the fancy index for the group.  This is **important**: loop using a fancy index (not a boolean mask).

There is also array recycling.  Because Tables often contain columns of similar row length, the intermediate arrays in operations are often the same dtype and length.  Therefore instead of returning the memory for the array back to the memory manager, which may zero it out or page it out, it is sometimes better to hold on to the array.  This is called array recycling.

When dealing with uniqueness: the mathematical property of high/low cardinality is important (what percent of the data is unique).  A high unique count often pairs well with classical sorting algorithms, or dividing group operations into bin ranges.  A low unique count often pairs well with hashing or segmenting group operations into segmented arrays.
