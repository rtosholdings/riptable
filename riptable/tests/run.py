# $Id$

import pytest
import sys
import os


def run_all(extra_args=None):
    """
    Run all the tests of riptable.

    Parameters
    ----------
    extra_args : list
        List of extra arguments (e.g. ['--verbosity=3'])
    """
    if extra_args is None:
        extra_args = []
    pytest.main(extra_args + ['-k', 'test_', os.path.dirname(__file__)])


# Usage: "python -m riptable.tests.run"
# You can add more arguments to the pytest, like "python -m riptable.tests.run --verbosity=2"
if __name__ == "__main__":
    run_all(sys.argv[1:])
