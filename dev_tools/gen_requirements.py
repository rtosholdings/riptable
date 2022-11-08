# Generates requirements for riptable.

import argparse
import platform
import sys


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_windows() -> bool:
    return platform.system() == "Windows"


_NUMPY_REQ = "numpy>=1.21"
_TBB_DEVEL_REQ = "tbb-devel==2021.6.*"

# Host toolchain requirements to build riptable.
# This also includes requirements to build riptide_cpp from source, if necessary.
toolchain_reqs = []
if is_linux():
    toolchain_reqs += [
        "binutils",
        "binutils_linux-64",
        "gcc==8.*",
        "gxx==8.*",
        "ninja",
    ]

# Conda-build requirements.
# Most everything else will be specified in meta.yaml.
conda_reqs = [
    "conda-build",
    "setuptools_scm",  # Needed to construct BUILD_VERSION for meta.yaml
] + toolchain_reqs

# PyPI setup build requirements.
# Most everything else will be specified in setup.py.
pypi_reqs = [
    _TBB_DEVEL_REQ,  # needed because PyPI tbb-devel pkg doesn't contain CMake files yet
] + toolchain_reqs

# Core runtime requirements for riptable and riptide_cpp.
runtime_reqs = [
    # No riptide_cpp as that must be handled separately
    "ansi2html>=1.5.2",
    "numba>=0.55.2",
    _NUMPY_REQ,
    "pandas>=0.24,<2.0",
    "python-dateutil",
    "tbb==2021.6.*",
]

# Complete test requirements for riptable tests.
tests_reqs = [
    "arrow",
    "bottleneck",
    "flake8",
    "hypothesis",
    "ipykernel",
    "ipython",
    "nose",
    "pyarrow",
    "pytest",
]

# Sphinx requirements for docs generation.
sphinx_reqs = ["sphinx_rtd_theme>=0.5.1", "sphinx-autoapi", "nbsphinx"] + runtime_reqs + tests_reqs

# Complete developer requirements.
developer_reqs = ["black", "setuptools_scm"] + conda_reqs + pypi_reqs + runtime_reqs + tests_reqs + toolchain_reqs

target_reqs = {
    "conda": conda_reqs,
    "developer": developer_reqs,
    "pypi": pypi_reqs,
    "runtime": runtime_reqs,
    "sphinx": sphinx_reqs,
    "tests": tests_reqs,
    "toolchain": toolchain_reqs,
}

parser = argparse.ArgumentParser()
parser.add_argument("targets", help="requirement targets", choices=target_reqs.keys(), nargs="+")
parser.add_argument("--out", help="output file", type=str)
args = parser.parse_args()

reqs = list({r for t in args.targets for r in target_reqs[t]})
reqs.sort()

with open(args.out, "w") if args.out else open(sys.stdout.fileno(), closefd=False) as out:
    print(f"# Requirements for targets: {args.targets}", file=out)
    for req in reqs:
        print(req, file=out)
