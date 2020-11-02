Performance Benchmarks
======================

riptable Benchmarking
--------------------
riptable offers a benchmark module. From the riptable workspace change to the ``Python\core`` directory and run the
``riptable.benchmark`` module with the ``help`` flag as follows:

  .. code-block:: bash

    riptable\Python\core>python -m riptable.benchmarks --help
    usage: __main__.py [-h] [--comparison] [-c] [--debug] [--d]
                       [benchmark [benchmark ...]]

    riptable benchmark runner.

    positional arguments:
      benchmark     space delimited benchmark names to run

    optional arguments:
      -h, --help    show this help message and exit
      --comparison  run comparison benchmarks
      -c            run comparison benchmarks
      --out-file    The filename where benchmark results will be saved in SDS format.
      --debug       debug level logging
      --d           debug level logging

The benchmarks framework uses a ``@benchmark`` decorator which takes a dictionary of ``benchmark_params`` to
parameterize the type of data arguments and settings for the benchmark runs. The benchmark settings are registered in
a settings table where the settings are set via Python context managers. The resulting benchmark data takes the form of
an hstacked Dataset which can be saved off and analyzed within riptable. Comparison of benchmarked functions is
supported, e.g. see the exposed ``compare_`` functions, and returns a mapping of benchmark names to their corresponding
Dataset results.

The following is an excerpt of an example run of ``compare_all`` that compares the riptable ``all`` implementation
against the Numpy implementation. When column information doesn't apply to the benchmark run the value is filled with
``Inv``. This allows benchmarking results across dimensions, such as thread count, applicable to riptable, but not Numpy.

  .. code-block:: bash

    riptable\Python\core>python -m riptable.benchmarks -c compare_lexsort
    2020-05-13 16:57:17,430 - INFO - main: parsed args Namespace(benchmarks=['compare_lexsort'], comparison=True, debug=False)
    2020-05-13 16:57:17,430 - INFO - Running compare_lexsort
    2020-05-13 16:57:25,843 - INFO -
                                                                           elapsed_ns
    *impl                    *farr   *thread_count       *arr      Min       Median        Max
    -------------------   --------   -------------   --------   ---------   ---------   ---------
    np_lexsort_reversed        Inv             Inv        100        2200        2400        4500
    .                          Inv             Inv       1000        5000        5150        5500
    .                          Inv             Inv      10000       28600       29150       29600
    .                          Inv             Inv     100000      756900      793050      842100
    .                          Inv             Inv    1000000    10561900    11095050    13220700
    .                          Inv             Inv   10000000   105755700   107151800   112242800
    rt_lexsort_reversed        100               1        Inv        2000        2500        5700
    .                          100               2        Inv        1900        2400        2800
    .                          100               4        Inv        1800        2350        2800
    .                          100               8        Inv        1900        2350        2800
    .                         1000               1        Inv        3900        9600       15100
    .                         1000               2        Inv        3900        9700       14900
    .                         1000               4        Inv        3900        9700       15100
    .                         1000               8        Inv        4000        9600       14900
    .                        10000               1        Inv       27200      110300      193800
    .                        10000               2        Inv       27200      110150      193700
    .                        10000               4        Inv       27200      110000      193700
    .                        10000               8        Inv       27200      110300      230800
    .                       100000               1        Inv      540900     1506250     2503600
    .                       100000               2        Inv      394300      970750     1700800
    .                       100000               4        Inv      390400      709550     1203000
    .                       100000               8        Inv      364800      676450     1129100
    .                      1000000               1        Inv     5439400    17212450    28986000
    .                      1000000               2        Inv     3788000    10343050    17321600
    .                      1000000               4        Inv     2809900     6686050    14196900
    .                      1000000               8        Inv     2748800     6915700    22913100
    .                     10000000               1        Inv    65367600   212712800   364192900
    .                     10000000               2        Inv    46265400   127870400   214217900
    .                     10000000               4        Inv    39374400    88413300   151709500
    .                     10000000               8        Inv    39833700    92766950   140222000

    [30 rows x 3 columns]
    2020-05-13 16:57:25,845 - INFO - main: finished with exit code 0

The ``out-file`` instructs the benchmark runner to save the raw (non-summarized) benchmark results to the specified
path in SDS format. The raw results data can then be more easily compared across machines or riptable versions, and can
also have more-advanced analyses applied to it (that we may not want to embed in riptable itself).

Here is one example of how to run with the ``out-file`` flag and load the data back into riptable. We see a ``Dataset``
for each of the benchmarks results along with a nested ``meta`` struct that has some information
about the benchmark run and the machine it was run on.

 .. code-block:: bash

    riptable\Python\core>python -m riptable.benchmarks --out-file D:\Temp\20200717-riptable-bench-results.sds bench_merge bench_merge2 bench_merge_pandas
    riptable\Python\core>ipython
    In [1]: import numpy as np

    In [2]: import riptable as rt

    In [3]: bench_results = rt.load_sds(r'D:\Temp\20200717-riptable-bench-results.sds')

    In [4]: bench_results.keys()

    ['meta', 'bench_merge']

    In [5]: bench_results.meta

     #   Name                    Type     Size   0                                  1   2

    --   ---------------------   ------   ----   --------------------------------   -   -
     0   timestamp               str_     0      2020-07-17T15:16:03.522940

     1   python_version          <U1      3      3                                  7   4

     2   python_implementation   str_     0      CPython

     3   riptable_version        str_     0      20200715.093410

     4   sysname                 str_     0      Windows

     5   nodename                str_     0      ci-test-win-022

     6   platform                str_     0      Windows-10-10.0.17763-SP0

     7   platform_release        str_     0      10

     8   platform_version        str_     0      10.0.17763

     9   platform_machine        str_     0      AMD64

    10   platform_processor      str_     0      Intel64 Family 6 Model 85 Steppi

    11   cpuinfo                 Struct   21


    [12 columns]

    In [6]: bench_results.meta.platform_processor

    'Intel64 Family 6 Model 85 Stepping 4, GenuineIntel'

    In [7]: bench_results.meta.cpuinfo

     #   Name                     Type    Size   0                                  1               2

    --   ----------------------   -----   ----   --------------------------------   -------------   ---
     0   python_version           str_    0      3.7.4.final.0 (64 bit)

     1   cpuinfo_version          int32   3      5                                  0               0

     2   arch                     str_    0      X86_64

     3   bits                     intc    0      64

     4   count                    intc    0      20

     5   raw_arch_string          str_    0      AMD64

     6   vendor_id                str_    0      GenuineIntel

     7   brand                    str_    0      Intel(R) Xeon(R) W-2155 CPU @ 3.

     8   hz_advertised            str_    0      3.3000 GHz

     9   hz_actual                str_    0      3.3120 GHz

    10   hz_advertised_raw        int64   2      3300000000                         0

    11   hz_actual_raw            int64   2      3312000000                         0

    12   l2_cache_size            str_    0      10240 KB

    13   stepping                 intc    0      4

    14   model                    intc    0      85

    15   family                   intc    0      6

    16   l3_cache_size            str_    0      14080 KB

    17   flags                    <U13    77     3dnow                              3dnowprefetch   abm

    18   l2_cache_line_size       intc    0      6

    19   l2_cache_associativity   str_    0      0x100

    20   extended_model           intc    0      5


    [21 columns]

    In [8]: bench_results.meta.cpuinfo.flags

    FastArray(['3dnow', '3dnowprefetch', 'abm', 'acpi', 'adx', 'aes', 'apic',

               'avx', 'avx2', 'avx512bw', 'avx512cd', 'avx512dq', 'avx512f',

               'avx512vl', 'bmi1', 'bmi2', 'clflush', 'clflushopt', 'clwb',

               'cmov', 'cx16', 'cx8', 'de', 'dtes64', 'dts', 'erms', 'est',

               'f16c', 'fma', 'fpu', 'fxsr', 'hle', 'ht', 'hypervisor',

               'ia64', 'invpcid', 'lahf_lm', 'mca', 'mce', 'mmx', 'movbe',

               'mpx', 'msr', 'mtrr', 'osxsave', 'pae', 'pat', 'pbe', 'pcid',

               'pclmulqdq', 'pdcm', 'pge', 'pni', 'popcnt', 'pqe', 'pqm',

               'pse', 'pse36', 'rdrnd', 'rdseed', 'rtm', 'sep', 'serial',

               'smap', 'smep', 'ss', 'sse', 'sse2', 'sse4_1', 'sse4_2',

               'ssse3', 'tm', 'tm2', 'tsc', 'vme', 'xsave', 'xtpr'],

              dtype='<U13')

    In [10]: from riptable.benchmarks.runner import quick_analysis

    In [11]: quick_analysis(bench_results.bench_merge)
                                                                                                                        elapsed_ns
    *rng_seed   *left_key_unique_count   *right_key_unique_count   *left_rowcount   *right_rowcount   *how      Min       Median      Max

    ---------   ----------------------   -----------------------   --------------   ---------------   -----   --------   --------   --------
        12345                      100                     10000           500000            250000   inner   10038000   10196800   10485400

            .                      100                     10000           500000            250000   left     4819900    6115100    6671100

            .                      100                     10000           500000            250000   right    3554400    3580700    3639300

            .                      100                     10000           500000            500000   inner    3900800    5756100    7213600

            .                      100                     10000           500000            500000   left     3037200    6404500    6563400

            .                      100                     10000           500000            500000   right    2530400    2670600    3055800

            .                      100                     10000          1000000            250000   inner   14893000   16254900   16766700

            .                      100                     10000          1000000            250000   left    10455400   10538100   10948700

            .                      100                     10000          1000000            250000   right    2666700    3352500    3624300

            .                      100                     10000          1000000            500000   inner    8249300    9395700    9827700

            .                      100                     10000          1000000            500000   left     9665800   11309700   11546400

            .                      100                     10000          1000000            500000   right    4127800    5076400    6008800

            .                      100                     10000          1500000            250000   inner   20830200   22349300   23487700

            .                      100                     10000          1500000            250000   left     5776800   12242200   16493300

            .                      100                     10000          1500000            250000   right    1284200    1464000    1750100

            .                      100                     10000          1500000            500000   inner   21101800   23693000   24084600

            .                      100                     10000          1500000            500000   left     8886800   15342700   16484200

            .                      100                     10000          1500000            500000   right    4850500    6059600    6261000

            .                      100                     10000          2000000            250000   inner   18289200   26688400   27989800

            .                      100                     10000          2000000            250000   left    11963700   16430100   19164800

            .                      100                     10000          2000000            250000   right    1276100    1299800    1698400

            .                      100                     10000          2000000            500000   inner   15788500   18976500   22218000

            .                      100                     10000          2000000            500000   left    14078700   16785800   20837400

            .                      100                     10000          2000000            500000   right    2307400    2350400    3070700


    [24 rows x 3 columns]

