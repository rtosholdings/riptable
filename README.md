# RipTable
High performance, low memory, multithreaded 64-bit python data analytics engine for big data based on numpy arrays.

Enhances or replaces numpy, pandas, and includes high speed cross platform SDS file format.
RipTable can often crunch numbers at 10x the speed of numpy of pandas.  It uses less memory than pandas.

Support for Python 3.6, 3.7, 3.8 on 64 bit Windows, Linux, and Mac OS.

To install "pip install riptable".

Basic Concepts and Classes
--------------------------
**FastArray**: subclasses from a numpy array with builtin multithreaded number crunching.  All scikit routines that expect a numpy array will also accept a FastArray since it is subclassed.  isinstance(fastarray, nd.array) will return True.

**Dataset**: replaces the pandas DataFrame class

**Struct**: replaces the pandas Series class

**Categorical**: replaces both pandas groupby and Categorical class

**SDS**: a new file format which can stack multiple datasets in multiple files with zstd compression, multiple threads and no extra memory copies.  SDS also supports loading and writing datasets to shared memory (with no extra packages).

Getting Started
----------------
```
import riptable as rt
ds = rt.Dataset({'intarray': rt.arange(1_000_000), 'floatarray': rt.arange(1_000_000.0)})
```
