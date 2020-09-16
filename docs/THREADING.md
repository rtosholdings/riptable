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
binary: add, sub, divide, etc.
logical: &, |, ~, etc.
reduce: sum, cumsum, std, etc.
comparisons: ==, !=, <, <=, >, >=
copy/convert: .copy(), .astype(), upcasting
hstack/vstack
where
sorting/uniquesness: lexsort, searchsorted, unique (soon sort)
datetime routines: strftime, finding the daylight savings
linear_interp
special grouping routines which dont exist in numpy
special hashing routines
ema related
window related
apply related and numba.prange loops (only way out to JIT multithreaded routines)


