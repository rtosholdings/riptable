# $Id$

import os
import sys

import pytest


def run_all(extra_args=None):
    """
    Run all the tooling integration tests of riptable.

    Parameters
    ----------
    extra_args : list
        List of extra arguments (e.g. ['--verbosity=3'])
    """
    if extra_args is None:
        extra_args = []
    return pytest.main(extra_args + ["-k", "test_", os.path.dirname(__file__)])


# Usage: "ipython -m riptable.test_tooling_integration.run"
# You can add more arguments to the pytest, like "ipython -m riptable.test_tooling_integration.run --verbosity=2"
if __name__ == "__main__":
    # Force ipython to exit with the exit code, as sys.exit() is caught and ignored :-/
    os._exit(run_all(sys.argv[1:]))
