The riptide repository and source distribution utilize several libraries and
tools that are compatibly licensed. This document is meant to encompass all areas
of the riptide codebase, including things like tests, documentation, and benchmarks.

Each dependency should include an SPDX identifier for it's license. Identifiers can be found here:
https://spdx.org/licenses/

The dependencies are as follows:

## Source Code

These are included, statically-linked, or otherwise compiled directly into
the riptide source code.

Name: crc32c
SPDX-Identifier: BSD-3-Clause
Url: https://github.com/google/crc32c
Files: CPP\riptide_cpp\CRC32.cpp

Name: fast-hash
SPDX-Identifier: MIT
Url: https://code.google.com/archive/p/fast-hash/
Files: CPP\riptide_cpp\HashLinear.cpp

Name: ipython
SPDX-Identifier: BSD-3-Clause
Url: https://github.com/ipython/ipython
Files: Python\core\riptide\test_tooling_integration\ipython_compatibility_test.py

Name: NetBSD strptime
SPDX-Identifier: BSD-2-Clause-NetBSD
Files: CPP\riptide_cpp\strptime5.cpp

Name: python-3.3
SPDX-Identifier: PSF-2.0
Url: https://github.com/certik/python-3.3
Files: CPP\riptide_cpp\Sort.cpp

Name: zstandard
SPDX-Identifier: BSD-3-Clause
Url: https://github.com/facebook/zstd
Files: CPP\riptide_cpp\zstd\*


## Python Libraries

These are libraries that riptide depends on at runtime, for testing, or for benchmarking.

Name: ansi2html
SPDX-Identifier: LGPL-3.0-or-later
Url: https://github.com/ralphbean/ansi2html/
Comment: Source files still have GPLv3 text, but package was changed to LGPLv3+ on 2017-11-29;
  see the following issue and related commit for details:
  https://github.com/ralphbean/ansi2html/issues/72
  https://github.com/ralphbean/ansi2html/commit/f74316174e902e0d3f1a1aef22173ffb84cdcbbf

Name: bottleneck
SPDX-Identifier: BSD-2-Clause
Url: https://github.com/pydata/bottleneck

Name: hypothesis
SPDX-Identifier: MPL-2.0
Url: https://github.com/HypothesisWorks/hypothesis

Name: ipython
SPDX-Identifier: BSD-3-Clause
Url: https://github.com/ipython/ipython

Name: ipykernel
SPDX-Identifier: BSD-3-Clause
Url: https://github.com/ipython/ipykernel

Name: matplotlib
SPDX-Identifier: PSF-based
Url: https://github.com/matplotlib/matplotlib/
License-Url: https://github.com/matplotlib/matplotlib/blob/master/LICENSE/LICENSE

Name: numba
SPDX-Identifier: BSD-2-Clause
Url: https://github.com/numba/numba

Name: numpy
SPDX-Identifier: BSD-3-Clause
Url: https://github.com/numpy/numpy

Name: pandas
SPDX-Identifier: BSD-3-Clause
Url: https://github.com/pandas-dev/pandas

Name: py-cpuinfo
SPDX-Identifier: MIT
Url: https://github.com/workhorsy/py-cpuinfo

Name: pytest
SPDX-Identifier: MIT
Url: https://github.com/pytest-dev/pytest

Name: pytest-cov
SPDX-Identifier: MIT
Url: https://github.com/pytest-dev/pytest-cov

Name: python
SPDX-Identifier: PSF-2.0
Url: https://github.com/python/cpython

Name: scipy
SPDX-Identifier: BSD-3-Clause
Url: https://github.com/scipy/scipy

Name: tbb
SPDX-Identifier: Apache-2.0
Url: https://github.com/01org/tbb

Name: teamcity-messages
SPDX-Identifier: Apache-2.0
Url: https://github.com/JetBrains/teamcity-messages

Name: typing_extensions
SPDX-Identifier: PSF-2.0
Url: https://github.com/python/typing/tree/master/typing_extensions


## Datasets

TODO: Include licenses for any datasets used in documentation, tests, or benchmarks.

