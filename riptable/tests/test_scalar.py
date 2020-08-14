"""Test around scalar constructors and scalar methods."""
import riptable as rt
import numpy as np
import pytest

from numpy.testing import assert_almost_equal, assert_warns


class TestScalarConstructor(object):
    # Type-coercion from strings test cases adapted from numpy/core/tests/test_scalar_ctors.py.
    # https://github.com/numpy/numpy/blob/c31cc36a8a814ed4844a2a553454185601914a5a/numpy/core/tests/test_scalar_ctors.py
    @pytest.mark.parametrize(
        "scalar_ctor, numeric_string",
        [
            # simple numeric string
            ("single", "1.234"),
            ("double", "1.234"),
            ("longdouble", "1.234"),
            # numeric string with overflow overflow; expect inf value
            ("half", "1e10000"),
            ("single", "1e10000"),
            ("double", "1e10000"),
            ("longdouble", "1e10000"),
            ("longdouble", "-1e10000"),
        ],
    )
    def test_floating(self, scalar_ctor, numeric_string):
        rt_value = getattr(rt, scalar_ctor)(numeric_string)
        np_value = getattr(np, scalar_ctor)(numeric_string)
        assert_almost_equal(rt_value, np_value)

    @pytest.mark.parametrize(
        "scalar_ctor, numeric_string",
        [("longdouble", "1e10000"), ("longdouble", "-1e10000"),],
    )
    def test_overflow_warning(self, scalar_ctor, numeric_string):
        assert_warns(RuntimeWarning, getattr(np, scalar_ctor), numeric_string)
