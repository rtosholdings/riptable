# RipTable
All in one, high performance 64 bit python analytics engine for numpy arrays with multithreaded support.

Support for Python 3.6, 3.7, 3.8 on 64 bit Windows, Linux, and Mac OS.

Enhances or replaces numpy, pandas, and includes high speed cross platform SDS file format.
RipTable can often crunch numbers at 10x the speed of numpy of pandas while using less memory than pandas.  

Maximum Speed is achieved through the use of: **vector instrinsics**: hand rolled loops, using AVX-256 with AVX-512 support coming. **smart threading**: for large arrays, multiple threads are deployed. **recycling**: built in array garbage collection.

To install "pip install riptable".

Basic Concepts and Classes
--------------------------
**FastArray**: subclasses from a numpy array with builtin multithreaded number crunching.  All scikit routines that expect a numpy array will also accept a FastArray since it is subclassed.  isinstance(fastarray, nd.array) will return True.

**Dataset**: replaces the pandas DataFrame class

**Struct**: replaces the pandas Series class

**Categorical**: replaces both pandas groupby and Categorical class

**SDS**: a new file format which can stack multiple datasets in multiple files with compression, threads, and no extra memory copies.  SDS also supports loading and writing datasets to shared memory.

Getting Started
----------------
```
import riptable as rt
ds = rt.Dataset({'intarray': rt.arange(1_000_000), 'floatarray': rt.arange(1_000_000.0)})
```
