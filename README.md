# RipTable
All in one, high performance 64 bit python analytics engine for numpy arrays with multithreaded support.

Support for Python 3.6, 3.7, 3.8 on 64 bit Linux, Windows, and Mac OS.

Enhances or replaces numpy, pandas, and includes high speed cross platform SDS file format.
RipTable can often crunch numbers at 10x the speed of numpy or pandas.  

Maximum speed is achieved through the use of: **vector instrinsics**: hand rolled loops, using AVX-256 with AVX-512 support coming. **smart threading**: for large arrays, multiple threads are deployed. **recycling**: built in array garbage collection.  **hashing** and **parallel sorts**: for core algorithms.

To install 
```
pip install riptable
```

Basic Concepts and Classes
--------------------------
**FastArray**: subclasses from a numpy array with builtin multithreaded number crunching.  All scikit routines that expect a numpy array will also accept a FastArray since it is subclassed.  isinstance(fastarray, np.ndarray) will return True.

**Dataset**: replaces the pandas DataFrame class

**Struct**: replaces the pandas Series class

**Categorical**: replaces both pandas groupby and Categorical class.  RipTable has powerful more Categorical classes.

**Date/Time Classes**: DateTimeNano, Date, TimeSpan, and DateSpan are designed more like Java, C++, or C# classes.  Replaces most numpy and pandas date time classes.

**Accum2/AccumTable**: For cross tabulation.

**SDS**: a new file format which can stack multiple datasets in multiple files with compression, threads, and no extra memory copies.  SDS also supports loading and writing datasets to shared memory.

Getting Started
----------------
```
import riptable as rt
ds = rt.Dataset({'intarray': rt.arange(1_000_000), 'floatarray': rt.arange(1_000_000.0)})
```
How can I trust RipTable calculations?
--------------------------------------
RipTable has been in development for 3 years and tested by dozens of quants at a large financial firm.  It has a full suite of testing (see: riptable/tests).  However just like any project, we still disover bugs and improvements.  Please report them using github issues.

Numpy Users
------------
FastArray is a numpy array, however they can be flipped back and forth with no array copies taking place (it just changes the view).
```
import riptable as rt
import numpy as np
a = rt.arange(100)
numpyarray = a._np
fastarray = rt.FA(numpyarray)
```
or directly by changing the view, not how a FastArray is a numpy array
```
numpyarray.view(rt.FastArray)
fastarry.view(np.ndarray)
ininstance(fastarray, np.ndarray)
```

Pandas Users
------------
Simply drop a pandas DataFrame class into a riptable Dataset and it will be auto converted.
```
import riptable as rt
import numpy as np
import pandas as pd
df = pd.DataFrame({'intarray': np.arange(1_000_000), 'floatarray': np.arange(1_000_000.0)})
ds = rt.Dataset(df)
```

How can RipTable perform the same calculations faster?
------------------------------------------------------
RipTable was written from day one to handle large data and mulithreading using the riptide_cpp layer for basic arithmetic functions and algorithms.  Many core algorithms have been painstakingly rewritten for multithreading.

Why doesn't numpy or pandas just pick up the same code?
-------------------------------------------------------
numpy does not have a multithreaded layer (we are in discussions with the numpy team to add such a layer), nor is it designed to use C++ templates or hashing algorithms.  pandas does not have a C++ layer (it uses cython instead) and is a victim of its own success making early design mistakes difficult to change (such as the block manager and lack of powerful Categoricals).
