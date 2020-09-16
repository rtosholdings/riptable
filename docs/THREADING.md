**Guide to multithreading numpy routines**

This document describes techniques used to speed up the processing of large data arrays in python on Intel x64 based CPUs.

Problems with processing large arrays fast on python.
   1) Numpy is not multithreaded or SIMD fully aware
   2) Pandas was originally written for smaller data
   3) Almost all third party python data array algorithms expect numpy arrays

In order to solve the above issues:
1) Write multithreaded algorthms
2) Rewrite the pandas dataset class
3) Subclass from numpy since that is the only way to speed it up and still be a numpy array

To multithread numpy we first break down numpy operations into categories:
unary: abs, trunc, sqrt, etc.
1) binary: add, sub, divide, etc.
2) logical: &, |, ~, etc.
3) reduce: sum, cumsum, std, etc.
4) comparisons: ==, !=, <, <=, >, >=
5) copy/convert: .copy(), .astype(), upcasting
6) hstack/vstack
7) where, putmask
8) binary and fancy index getitem/setitem
8) sorting/uniquesness: lexsort, searchsorted, unique (soon sort)
9) datetime routines: strftime, finding the daylight savings
10) linear_interp
11) special grouping routines which dont exist in numpy (like ismember, transform=True, ifirstkey)
12) special hashing routines
13) ema related
14) window related: rolling functions in pandas
15) apply related and numba.prange loops (only way out to JIT multithreaded routines)


Further numpy has no
1) Table/DataFrame/Dataset class
2) Categorical (multikey) class
3) Way to save and compress large data tables and read back stacked

This prevented numpy from vertical software integration (like Apple does on the iphone).  If numpy did, it would have developed those concepts more, which would have led to more group, compression related functions in numpy.  Maturity in those areas would be beneficial.


