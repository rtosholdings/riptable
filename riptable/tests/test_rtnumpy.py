import math
import unittest
import pytest
from numpy.testing import (
    assert_array_equal
)

import riptable as rt

from riptable import *
from riptable.rt_utils import alignmk


class TestRTNumpy:
    """Tests for various functions in the rt_numpy module."""

    def test_where(self):
        arr = FA([False, True, False])
        result = where(arr, True, False)
        assert_array_equal(
            result, arr,
            err_msg=f"Results did not match for where. {arr} vs. {result}"
        )

    def test_interp(self):
        x = interp(arange(3.0).astype(float32), [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        y = interp(arange(3.0), [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        z = np.interp(arange(3.0), [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert_array_equal(
            z, y, err_msg=f"Results did not match for where. {z} vs. {y}"
        )

    def test_unique(self):
        symb = FastArray(
            # Top SPX constituents by weight as of 20200414
            [
                b'MSFT',
                b'AAPL',
                b'AMZN',
                b'FB',
                b'JNJ',
                b'GOOG',
                b'GOOGL',
                b'BRK.B',
                b'PG',
                b'JPM',
                b'V',
                b'INTC',
                b'UNH',
                b'VZ',
                b'MA',
                b'T',
                b'HD',
                b'MRK',
                b'PFE',
                b'PEP',
                b'BAC',
                b'DIS',
                b'KO',
                b'WMT',
                b'CSCO',
            ],
        )

        x1, y1, z1 = np.unique(symb, return_index=True, return_inverse=True)
        x2, y2, z2 = unique(symb, return_index=True, return_inverse=True)
        assert_array_equal(
            x1, x2, err_msg=f"Results did not match for unique. {x1} vs. {x2}"
        )
        assert_array_equal(
            y1, y2, err_msg=f"Results did not match for unique index. {y1} vs. {y2}"
        )
        assert_array_equal(
            z1, z2,
            err_msg=f"Results did not match for unique inverse. {z1} vs. {z2}",
        )
        x2, y2, z2 = unique(Cat(symb), return_index=True, return_inverse=True)
        assert_array_equal(
            x1, x2, err_msg=f"Results did not match for cat unique. {x1} vs. {x2}"
        )
        assert_array_equal(
            y1, y2,
            err_msg=f"Results did not match for cat unique index. {y1} vs. {y2}",
        )
        assert_array_equal(
            z1, z2,
            err_msg=f"Results did not match for cat unique inverse. {z1} vs. {z2}",
        )

        # round 2
        symb = FastArray(
            [
                'NIHD',
                'SPY',
                'AAPL',
                'LOCO',
                'XLE',
                'MRNA',
                'JD',
                'JD',
                'QNST',
                'RUSL',
                'USO',
                'TSLA',
                'NVDA',
                'GLD',
                'GLD',
                'ZS',
                'WDC',
                'AGEN',
                'AMRS',
                'AAPL',
                'SMH',
                'PYPL',
                'AAPL',
                'SQQQ',
                'GLD',
            ],
            unicode=True,
        )

        x1, y1, z1 = np.unique(symb, return_index=True, return_inverse=True)

        x2, y2, z2 = unique(symb, return_index=True, return_inverse=True)
        assert_array_equal(
            x1, x2, err_msg=f"Results did not match for unique. {x1} vs. {x2}"
        )
        assert_array_equal(
            y1, y2, err_msg=f"Results did not match for unique index. {y1} vs. {y2}"
        )
        assert_array_equal(
            z1, z2,
            err_msg=f"Results did not match for unique inverse. {z1} vs. {z2}",
        )

        x2, y2, z2 = unique(Cat(symb), return_index=True, return_inverse=True)
        assert_array_equal(
            x1, x2, err_msg=f"Results did not match for cat unique. {x1} vs. {x2}"
        )
        assert_array_equal(
            y1, y2,
            err_msg=f"Results did not match for cat unique index. {y1} vs. {y2}",
        )
        assert_array_equal(
            z1, z2,
            err_msg=f"Results did not match for cat unique inverse. {z1} vs. {z2}",
        )

        # round 3
        x2, y2, z2 = unique(symb, return_index=True, return_inverse=True, lex=True)
        assert_array_equal(
            x1, x2, err_msg=f"Results did not match for cat unique. {x1} vs. {x2}"
        )
        assert_array_equal(
            y1, y2,
            err_msg=f"Results did not match for cat unique index. {y1} vs. {y2}",
        )
        assert_array_equal(
            z1, z2,
            err_msg=f"Results did not match for cat unique inverse. {z1} vs. {z2}",
        )

    def test_alignmk(self):
        ds1 = rt.Dataset()
        ds1['Time'] = [0, 1, 4, 6, 8, 9, 11, 16, 19, 30]
        ds1['Px'] = [10, 12, 15, 11, 10, 9, 13, 7, 9, 10]

        ds2 = rt.Dataset()
        ds2['Time'] = [0, 0, 5, 7, 8, 10, 12, 15, 17, 20]
        ds2['Vols'] = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

        # Categorical keys
        ds1['Ticker'] = rt.Categorical(['Test'] * 10)
        ds2['Ticker'] = rt.Categorical(['Test', 'Blah'] * 5)
        res = alignmk(ds1.Ticker, ds2.Ticker, ds1.Time, ds2.Time)
        target = rt.FastArray([0, 0, 0, 2, 4, 4, 4, 6, 8, 8])
        assert_array_equal(res, target)

        # char array keys
        ds1['Ticker'] = rt.FastArray(['Test'] * 10)
        ds2['Ticker'] = rt.FastArray(['Test', 'Blah'] * 5)
        res = alignmk(ds1.Ticker, ds2.Ticker, ds1.Time, ds2.Time)
        target = rt.FastArray([0, 0, 0, 2, 4, 4, 4, 6, 8, 8])
        assert_array_equal(res, target)

    def test_sample(self):
        # Test Dataset.sample
        ds = rt.Dataset({'num': [1, 2, 3, 4, 5], 'str': ['ab', 'bc', 'cd', 'de', 'ef']})
        np.random.seed(1)
        ds_sample = ds.sample(3, rt.FA([True, True, True, False, True]))
        ds_sample_expected = rt.Dataset({'num': [1, 3, 5], 'str': ['ab', 'cd', 'ef']})
        assert (ds_sample_expected == ds_sample).all(axis=None)

        # Test FastArray.sample
        fa = rt.FA([1, 2, 3, 4, 5])
        np.random.seed(1)
        fa_sample = fa.sample(2, rt.FA([False, True, True, False, True]))
        fa_sample_expected = rt.FA([2, 5])
        assert (fa_sample_expected == fa_sample).all(axis=None)

        # Test overflow
        fa_sample = fa.sample(10, rt.FA([False, True, False, False, True]))
        fa_sample_expected = rt.FA([2, 5])
        assert (fa_sample_expected == fa_sample).all(axis=None)

        # Test no filter
        np.random.seed(1)
        fa_sample = fa.sample(2)
        fa_sample_expected = rt.FA([2, 3])
        assert (fa_sample_expected == fa_sample).all(axis=None)

        # Test fancy index
        np.random.seed(1)
        fa_sample = fa.sample(2, rt.FA([1, 3, 4]))
        fa_sample_expected = rt.FA([2, 5])
        assert (fa_sample_expected == fa_sample).all(axis=None)


class TestMaximum:
    """Tests for the rt.maximum function."""

    def test_maximumscalar(self):
        # scalar vs. scalar
        x = maximum(3, 4.0, dtype=np.float32)
        assert isinstance(x, np.float32)
        assert x == 4.0

        # scalar vs. array
        scalar_value = 3
        x = maximum(scalar_value, arange(100), dtype=np.float32)
        assert x.dtype == np.float32
        expected = np.arange(100, dtype=np.float32)
        expected[:scalar_value] = scalar_value

        # array vs. scalar
        scalar_value = 3
        x = maximum(arange(100), scalar_value, dtype=np.float32)
        assert x.dtype == np.float32
        expected = np.arange(100, dtype=np.float32)
        expected[:scalar_value] = scalar_value


class TestMinimum:
    """Tests for the rt.minimum function."""

    def test_minimumscalar(self):
        # scalar vs. scalar
        x = minimum(3, 4.0, dtype=np.float32)
        assert isinstance(x, np.float32)
        assert x == 3.0

        # scalar vs. array
        scalar_value = 3
        x = minimum(scalar_value, arange(100), dtype=np.float32)
        assert x.dtype == np.float32
        expected = np.arange(100, dtype=np.float32)
        expected[scalar_value:] = scalar_value

        # array vs. scalar
        scalar_value = 3
        x = minimum(arange(100), scalar_value, dtype=np.float32)
        assert x.dtype == np.float32
        expected = np.arange(100, dtype=np.float32)
        expected[scalar_value:] = scalar_value


# TODO: Extend the tests in the TestNanmax / TestNanmin classes below to cover the following cases:
#   * non-array inputs (e.g. a list or set or scalar)
#   * other FastArray subclass, e.g. Date
#   * other unordered FastArray subclass, e.g. DateSpan (or at least, a subclass which doesn't
#       have a normal 3-way comparison because it's not a poset).
#       * For example, rt.DateSpan should probably use Allen's interval algebra which has 13 possible outcomes.
#       * Seems like we should disallow such classes; how should we check though?
#   * 2d and 3d arrays (these will probably punt to numpy, but verify)
#       * this won't work as expected on an integer dtype array; need to detect this case and just
#         disallow it rather than returning a bad result.
#   * strided/sliced arrays
#       * for sliced arrays, we're looking for issues related to data/pointer alignment in the riptide_cpp code;
#         use a reasonable-sized array (say, 1000 elements), and check we can call the function after slicing off
#         an element from the front of the array.
#   * test the above with float, integer, and string dtypes; make sure riptable invalid values (for integers; strings too?) are respected.
#   * check that any kwargs are validated (e.g. calling with a 1D array and axis=1 is not allowed).

class TestNanmax:
    """Tests for the rt.nanmax function."""

    @pytest.mark.parametrize(
        "arg",
        [
            pytest.param(math.nan, id='scalar'),
            pytest.param([math.nan, math.nan, math.nan, math.nan, math.nan], id='list-float'),
            pytest.param({np.nan}, id='set-float', marks=pytest.mark.skip("Broken in both numpy and riptable.")),
            pytest.param(np.full(100, np.nan, dtype=np.float32), id='ndarray-float'),
            pytest.param(rt.full(100, np.nan, dtype=np.float32), id='FastArray-float')
            # TODO: Ordered Categorical with all invalids
        ]
    )
    def test_allnans(self, arg):
        # Call rt.nanmax with the test input.
        # It should raise a RuntimeWarning when given an input which
        # has all NaNs **on the specified axis**.
        with pytest.warns(RuntimeWarning):
            result = rt.nanmax(arg)

        # If given a scalar or 1D array (or some collection converted to such)
        # the result should be a NaN; for higher-rank arrays, the result should
        # be an array where one of the dimensions was collapsed and if there were
        # all NaNs along the selected axis there'll be a NaN there in the result.
        # TODO: Need to fix this to assert correctly for when rt.nanmax called with a higher-rank array.
        assert rt.isnan(result)

    @pytest.mark.parametrize(
        "arg",
        [
            pytest.param([], id='list-float'),
            pytest.param({}, id='set-float', marks=pytest.mark.skip("Broken in both numpy and riptable.")),
            pytest.param(np.array([], dtype=np.float32), id='ndarray-float'),
            pytest.param(
                rt.FastArray([], dtype=np.float32), id='FastArray-float',
                marks=pytest.mark.xfail(
                    reason="RIP-417: The call to riptide_cpp via the ledger returns None, which then causes the isnan() to raise a TypeError. This needs to be fixed so we raise an error like numpy (either by checking for this and raising the exception, or fixing the way the function punts to numpy."))
            # TODO: Empty ordered Categorical (create with some categories but an empty backing array).
            # TODO: Empty Date array (representing a FastArray subclass)
        ]
    )
    def test_empty(self, arg):
        # Call rt.nanmax with an empty input -- it should raise a ValueError.
        with pytest.raises(ValueError):
            rt.nanmax(arg)

    def test_unordered_categorical_disallowed(self):
        """Test which verifies rt.nanmax raises an exception if called with an unordered Categorical."""
        # Create an unordered Categorical.
        cat = rt.Categorical(["PA", "NY", "NY", "AL", "LA", "PA", "CA", "IL", "IL", "FL", "FL", "LA"], ordered=False)
        assert not cat.ordered

        with pytest.raises(ValueError):
            rt.nanmax(cat)

    @pytest.mark.xfail(reason="RIP-417: nanmax does not (yet) support returning a category scalar.")
    def test_ordered_categorical_returns_scalar(self):
        """
        Test which verifies rt.nanmax returns a scalar (Python object or numpy scalar) representing the max Category given an ordered Categorical.
        """
        # Create an ordered Categorical (aka 'ordinal').
        cat = rt.Categorical(
            ["PA", "NY", "", "NY", "AL", "LA", "PA", "", "CA", "IL", "IL", "FL", "FL", "LA"], ordered=True)
        assert cat.ordered

        result = rt.nanmax(cat)

        # The result should either be a Python string, a numpy string scalar, or a Categorical scalar (if we implement one).
        is_py_str = isinstance(result, (bytes, str))
        is_np_scalar = isinstance(result, np.str)
        is_rt_cat = isinstance(result, rt.Categorical)
        assert is_py_str or is_np_scalar or is_rt_cat

        # Check the result is correct.
        assert result == "FL"


class TestNanmin:
    """Tests for the rt.nanmin function."""

    @pytest.mark.parametrize(
        "arg",
        [
            pytest.param(math.nan, id='scalar'),
            pytest.param([math.nan, math.nan, math.nan, math.nan, math.nan], id='list-float'),
            pytest.param({np.nan}, id='set-float', marks=pytest.mark.skip("Broken in both numpy and riptable.")),
            pytest.param(np.full(100, np.nan, dtype=np.float32), id='ndarray-float'),
            pytest.param(rt.full(100, np.nan, dtype=np.float32), id='FastArray-float')
            # TODO: Ordered Categorical with all invalids
        ]
    )
    def test_allnans(self, arg):
        # Call rt.nanmin with the test input.
        # It should raise a RuntimeWarning when given an input which
        # has all NaNs **on the specified axis**.
        with pytest.warns(RuntimeWarning):
            result = rt.nanmin(arg)

        # If given a scalar or 1D array (or some collection converted to such)
        # the result should be a NaN; for higher-rank arrays, the result should
        # be an array where one of the dimensions was collapsed and if there were
        # all NaNs along the selected axis there'll be a NaN there in the result.
        # TODO: Need to fix this to assert correctly for when rt.nanmin called with a higher-rank array.
        assert rt.isnan(result)

    @pytest.mark.parametrize(
        "arg",
        [
            pytest.param([], id='list-float'),
            pytest.param({}, id='set-float', marks=pytest.mark.skip("Broken in both numpy and riptable.")),
            pytest.param(np.array([], dtype=np.float32), id='ndarray-float'),
            pytest.param(
                rt.FastArray([], dtype=np.float32), id='FastArray-float',
                marks=pytest.mark.xfail(
                    reason="RIP-417: The call to riptide_cpp via the ledger returns None, which then causes the isnan() to raise a TypeError. This needs to be fixed so we raise an error like numpy (either by checking for this and raising the exception, or fixing the way the function punts to numpy."))
            # TODO: Empty ordered Categorical (create with some categories but an empty backing array).
            # TODO: Empty Date array (representing a FastArray subclass)
        ]
    )
    def test_empty(self, arg):
        # Call rt.nanmin with an empty input -- it should raise a ValueError.
        with pytest.raises(ValueError):
            rt.nanmin(arg)

    def test_unordered_categorical_disallowed(self):
        """Test which verifies rt.nanmin raises an exception if called with an unordered Categorical."""
        # Create an unordered Categorical.
        cat = rt.Categorical(["PA", "NY", "NY", "AL", "LA", "PA", "CA", "IL", "IL", "FL", "FL", "LA"], ordered=False)
        assert not cat.ordered

        with pytest.raises(ValueError):
            rt.nanmin(cat)

    @pytest.mark.xfail(reason="RIP-417: nanmin does not (yet) support returning a category scalar.")
    def test_ordered_categorical_returns_scalar(self):
        """
        Test which verifies rt.nanmin returns a scalar (Python object or numpy scalar) representing the min Category given an ordered Categorical.
        """
        # Create an ordered Categorical (aka 'ordinal').
        cat = rt.Categorical(
            ["PA", "NY", "", "NY", "AL", "LA", "PA", "", "CA", "IL", "IL", "FL", "FL", "LA"], ordered=True)
        assert cat.ordered

        result = rt.nanmin(cat)

        # The result should either be a Python string, a numpy string scalar, or a Categorical scalar (if we implement one).
        is_py_str = isinstance(result, (bytes, str))
        is_np_scalar = isinstance(result, np.str)
        is_rt_cat = isinstance(result, rt.Categorical)
        assert is_py_str or is_np_scalar or is_rt_cat

        # Check the result is correct.
        assert result == "PA"


if __name__ == "__main__":
    tester = unittest.main()
