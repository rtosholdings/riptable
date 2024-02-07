# Generates requirements for riptable.

import argparse
import platform
import sys


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_python(major: int, minor: int) -> bool:
    ver = sys.version_info
    return ver.major == major and ver.minor == minor


_BENCHMARK_REQ = "benchmark>=1.7,<1.8"
_NUMPY_REQ = "numpy>=1.23"
_TBB_VER = "==2021.6.*"
_TBB_REQ = "tbb" + _TBB_VER
_TBB_DEVEL_REQ = "tbb-devel" + _TBB_VER
_ZSTD_REQ = "zstd>=1.5.2,<1.6"

# Host toolchain requirements to build riptable.
# This also includes requirements to build riptide_cpp from source, if necessary.
toolchain_reqs = []
if is_linux():
    toolchain_reqs += [
        "binutils",
        "binutils_linux-64",
        "gcc==13.*",
        "gxx==13.*",
        "libstdcxx-ng=13.*",
        "ninja",
    ]

setuptools_reqs = [
    "setuptools>=65",
    "setuptools_scm[toml]>=7.1",
]

# Conda-build requirements.
# Most everything else will be specified in meta.yaml.
conda_reqs = [
    "boa",
    "conda-build",
    "setuptools_scm",  # Needed to construct BUILD_VERSION for meta.yaml
] + toolchain_reqs

# PyPI setup build requirements.
# Most everything *should* be in pyproject.toml, but some packages
# need to be set up manually here.
pypi_reqs = [
    "build",  # PEP-517 py build frontend
    _BENCHMARK_REQ,  # PyPI package doesn't exist
    _TBB_DEVEL_REQ,  # needed because PyPI tbb-devel pkg doesn't contain CMake files yet
    _ZSTD_REQ,  # PyPI package doesn't exist
] + toolchain_reqs

# Core runtime requirements for riptable and riptide_cpp.
# Must be available as Conda and PyPI packages
runtime_reqs = [
    # No riptide_cpp as that must be handled separately
    "ansi2html>=1.5.2",
    "numba>=0.56.2",
    _NUMPY_REQ,
    "pandas>=1.0,<3.0",
    "python-dateutil",
    _TBB_REQ,
    "typing-extensions>=4.9.0",
]

# Optional runtime requirements for riptable and riptide_cpp.
# Must be available as Conda and PyPI packages
runtime_opt_reqs = [
    "arrow",
    "ipykernel",
    "ipython<8.13" if is_python(3, 8) else "ipython",
    "matplotlib",
    "pillow==10.1.0",  # latest conda-forge version that works with matplotlib
    "pyarrow",
]

# Black formatting requirements.
black_reqs = [
    "black==23.*",
]

# Flake8 style guide requirements.
flake8_reqs = [
    "flake8==6.*",
]

# Complete test requirements for riptable tests.
tests_reqs = [
    "bokeh",
    "bottleneck",
    "hypothesis",
    "nose",
    "pymoo",
    "pytest",
    "pytest-cov",
] + runtime_opt_reqs

# Sphinx requirements for docs generation.
#
# *** IMPORTANT ***
# Any changes made here must be reflected in docs/environment.yml as well for ReadTheDocs use!
#
sphinx_reqs = (
    [
        "nbsphinx",
        "pytest",
        "sphinx_rtd_theme>=0.5.1",
        "sphinx-autoapi>=3,<4",
        "sphinx-design",
        "sphinx>=7,<8",
        "tomli",
    ]
    + runtime_reqs
    + runtime_opt_reqs
)

# Docstrings validation requirements.
# Validation requires complete riptable for iteration and evaluating examples.
docstrings_reqs = (
    [
        "numpydoc",
        "tomli",
    ]
    + flake8_reqs
    + black_reqs
    + runtime_reqs
    + tests_reqs
)

# Pydocstyle doc style requirements.
pydocstyle_reqs = [
    "pydocstyle==6.*",
    "toml",
]

# Complete developer requirements.
developer_reqs = (
    setuptools_reqs
    + black_reqs
    + conda_reqs
    + flake8_reqs
    + pydocstyle_reqs
    + pypi_reqs
    + runtime_reqs
    + tests_reqs
    + toolchain_reqs
)

target_reqs = {
    "black": black_reqs,
    "conda": conda_reqs,
    "developer": developer_reqs,
    "flake8": flake8_reqs,
    "pydocstyle": pydocstyle_reqs,
    "pypi": pypi_reqs,
    "runtime": runtime_reqs,
    "runtime_opt": runtime_opt_reqs,
    "sphinx": sphinx_reqs,
    "tests": tests_reqs,
    "toolchain": toolchain_reqs,
    "docstrings": docstrings_reqs,
}

parser = argparse.ArgumentParser()
parser.add_argument("targets", help="requirement targets", choices=target_reqs.keys(), nargs="+")
parser.add_argument("--out", help="output file", type=str)
parser.add_argument("--quote", "-q", help="quote entries", action="store_true")
args = parser.parse_args()

reqs = list({r for t in args.targets for r in target_reqs[t]})
reqs.sort()

# Emit plain list to enable usage like: conda install $(gen_requirements.py developer)
out = open(args.out, "w") if args.out else sys.stdout
try:
    quot = '"' if args.quote else ""
    for req in reqs:
        print(quot + req + quot, file=out)
finally:
    if args.out:
        out.close()
