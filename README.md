# Riptable

![](https://riptable.readthedocs.io/en/stable/_static/riptable_logo.PNG)

An open-source, 64-bit Python analytics engine for high-performance data analysis with
multithreading support. Riptable supports Python 3.10 through 3.11 on 64-bit Linux and
Windows.

Similar to Pandas and based on NumPy, Riptable optimizes analyzing large volumes of data
interactively, in real time. Riptable can crunch numbers often at 1.5x to 10x the speed
of NumPy or Pandas.

Riptable achieves maximum speed through the use of:

* **[Vector instrinsics](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)**
with hand-rolled loops using [AVX-256](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX2)
and with [AVX-512](https://en.wikipedia.org/wiki/AVX-512) support coming.
* **[Parallel computing](https://www.drdobbs.com/go-parallel/article/print?articleId=212903586)**
with multiple-thread deployment for large arrays.
* **[Recycling](https://en.wikipedia.org/wiki/Garbage_collection_(computer_science))**
with built-in array garbage collection.
* **[Hashing](https://en.wikipedia.org/wiki/Hash_function)** and **parallel sorts** for
core algorithms.

Intro to Riptable and reference documentation is available at:
[riptable.readthedocs.io](https://riptable.readthedocs.io/en/stable/index.html)

Basic concepts and classes
--------------------------

**[FastArray](https://riptable.readthedocs.io/en/stable/autoapi/riptable/rt_fastarray/index.html)**
is a subclass of NumPy's `ndarray` that enables built-in multithreaded number crunching.
All Scikit routines that expect a NumPy array also accept a `FastArray`.

**[Dataset](https://riptable.readthedocs.io/en/stable/autoapi/riptable/rt_dataset/index.html)**
replaces the Pandas `DataFrame` class and holds NumPy arrays of equal length.

**[Struct](https://riptable.readthedocs.io/en/stable/autoapi/riptable/rt_struct/index.html)**
holds a collection of mixed-type data members, with `Dataset` as a subclass.

**[Categorical](https://riptable.readthedocs.io/en/stable/autoapi/riptable/rt_categorical/index.html)**
replaces both the Pandas `DataFrame.groupby()` method and the Pandas `Categorical`
class. A Riptable `Categorical` supports multi-key, filterable groupings with the same
functionality of Pandas `groupby` and more.

**[Datetime](https://riptable.readthedocs.io/en/stable/autoapi/riptable/rt_datetime/index.html)**
classes replace most NumPy and Pandas date/time classes. Riptable's `DateTimeNano`,
`Date`, `TimeSpan`, and `DateSpan` classes have a design that's closer to Java, C++,
or C# date/time classes.

**[Accum2](https://riptable.readthedocs.io/en/stable/autoapi/riptable/rt_accum2/index.html)**
and **[AccumTable](https://riptable.readthedocs.io/en/stable/autoapi/riptable/rt_accumtable/index.html)**
enable cross-tabulation functionality.

**[SDS](https://riptable.readthedocs.io/en/stable/autoapi/riptable/rt_sds/index.html)**
provides a new file format which can stack multiple datasets in multiple files with
[zstd](https://github.com/facebook/zstd) compression, threads, and no extra memory
copies.

Small, medium, and large array performance
------------------------------------------

Riptable is designed for arrays of *all* sizes. For small arrays (< 100 length), low
processing overhead is important. Riptable's `FastArray` is written in hand-coded C and
processes simple arithmetic functions faster than NumPy arrays. For medium arrays
(< 100,000 length), Riptable has vector-instrinic loops. For large arrays (>= 100,000)
Riptable knows how to dynamically scale out threading, waking up threads efficiently
using a [futex](https://man7.org/linux/man-pages/man7/futex.7.html).

Install and import Riptable
---------------------------

Create a Conda environment and run this command to install Riptable on Windows or Linux:

```
conda install riptable
```

Import Riptable in your Python code to access its functions, methods, and classes:

```
import riptable as rt
```

>**Note**: We shorten the name of the Riptable module to `rt` to improve the readability
of code.

Use NumPy arrays with Riptable
------------------------------

Easily change between NumPy's `ndarray` and Riptable's `FastArray` without producing a
copy of the array.

```
import riptable as rt
import numpy as np
rtarray = rt.arange(100)
numpyarray = rtarray._np
fastarray = rt.FastArray(numpyarray)
```

Change the view of the two instances to confirm that `FastArray` is a subclass of
`ndarray`.

```
numpyarray.view(rt.FastArray)
fastarray.view(np.ndarray)
isinstance(fastarray, np.ndarray)
```

Use Pandas DataFrames with Riptable
-----------------------------------

Construct a Riptable `Dataset` directly from a Pandas `DataFrame`.

```
import riptable as rt
import numpy as np
import pandas as pd
df = pd.DataFrame({"intarray": np.arange(1_000_000), "floatarray": np.arange(1_000_000.0)})
ds = rt.Dataset(df)
```

How can I trust Riptable calculations?
--------------------------------------

Riptable has undergone years of development, and dozens of quants at a large financial
firm have tested its capabilities. We also provide a full suite of
[tests](https://github.com/rtosholdings/riptable/tree/master/riptable/tests) to ensure
that the modules are functioning as expected. But as with any project, there are still
bugs and opportunities for improvement, which can be reported using GitHub issues.

How can Riptable perform calculations faster?
---------------------------------------------

Riptable was written from day one to handle large data and multithreading using the
riptide_cpp layer for basic arithmetic functions and algorithms. Many core algorithms
have been painstakingly rewritten for multithreading.

How can I contribute?
---------------------

The Riptable engine is another building block for Python data analytics computing, and
we welcome help from users and contributors to take it to the next level. As you
encounter bugs, issues with the documentation, and opportunities for new or improved
functionality, please consider reaching out to the team.

See the [contributing guide](https://github.com/rtosholdings/riptable/blob/master/docs/CONTRIBUTING.md)
for more information.
