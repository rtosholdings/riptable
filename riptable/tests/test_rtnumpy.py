import math
import unittest

import numpy as np

import pytest
from numpy.testing import assert_array_equal

import riptable as rt
from riptable import *
from riptable.rt_utils import alignmk
from riptable.tests.utils import get_rc_version, parse_version


class TestRTNumpy:
    """Tests for various functions in the rt_numpy module."""

    def test_where(self):
        arr = FA([False, True, False])
        result = where(arr, True, False)
        assert_array_equal(result, arr, err_msg=f"Results did not match for where. {arr} vs. {result}")

        x_raw = ["a", "b", "a", "c", "b"]
        y_raw = ["b", "a", "b", "b", "e"]
        z_raw = ["a", "c", "f", "g", "a"]
        f_raw = [True, False, True, False, False]

        x = Categorical(x_raw)
        y = Categorical(y_raw)
        z = FastArray(z_raw)
        f = FastArray(f_raw)

        out1 = where(f, x, y)
        out2 = where(f, x, z)
        out3 = where(f, z, y)
        out4 = where(f_raw, x, y)
        out5 = where(f_raw, x, z)
        assert_array_equal(out1, FastArray(["a", "a", "a", "b", "e"]))
        assert isinstance(out1, Categorical)
        assert_array_equal(out2, FastArray(["a", "c", "a", "g", "a"]))
        assert not isinstance(out2, Categorical)
        assert_array_equal(out3, FastArray(["a", "a", "f", "b", "e"]))
        assert not isinstance(out3, Categorical)
        assert_array_equal(out4, FastArray(["a", "a", "a", "b", "e"]))
        assert isinstance(out4, Categorical)
        assert_array_equal(out5, FastArray(["a", "c", "a", "g", "a"]))
        assert not isinstance(out5, Categorical)

        # make sure python list results match with np.where
        assert_array_equal(where(f_raw, x_raw, y_raw), np.where(f_raw, x_raw, y_raw))
        assert_array_equal(where(f_raw, x_raw, z_raw), np.where(f_raw, x_raw, z_raw))
        assert_array_equal(where(f_raw, z_raw, y_raw), np.where(f_raw, z_raw, y_raw))

        # make sure numeric args are resolved to the correct type
        f = [False, True, True, False]
        y = [1, 2, 3, 4]
        assert_array_equal(rt.where(f, 0, y), np.where(f, 0, y))
        assert_array_equal(rt.where(f, -5, y), np.where(f, -5, y))
        assert_array_equal(rt.where(f, 0.0, y), np.where(f, 0.0, y))
        assert_array_equal(rt.where(f, 10**50, y), np.where(f, 10**50, y))

        # test invalid
        assert_array_equal(rt.where(f, 255, 0), rt.FA([0, 255, 255, 0], dtype=rt.uint8))

    def test_where_length_one(self):
        assert_array_equal(rt.where(rt.FA([False]))[0], np.array([]))

    def test_where_empty_condition(self):
        f = FastArray([])
        x = FastArray(["a", "b", "c"])
        y = FastArray(["b", "c"])
        empty = FastArray([], dtype=get_common_dtype(x, y))

        res1 = rt.where(f, x, y)
        res2 = rt.where(f, y, x)
        assert_array_equal(res1, empty)
        assert_array_equal(res2, empty)

    def test_where_array_edge_cases(self):
        empty = FastArray([])
        c1 = FastArray([False])
        c2 = FastArray([False, False])
        c3 = FastArray([False, False, False])
        l1 = FastArray([1])
        l2 = FastArray([1, 2])
        l3 = FastArray([1, 2, 3])
        # all empty
        assert_array_equal(rt.where(empty, empty, empty), FastArray([]))

        def ensure_error(f, *args):
            with pytest.raises((ValueError, SystemError)):
                f(*args)

        # if either x or y is empty error
        ensure_error(rt.where, False, l1, empty)
        ensure_error(rt.where, c1, empty, l1)
        ensure_error(rt.where, c3, empty, empty)

        # if length is mismatched
        ensure_error(rt.where, c1, l2, l3)
        ensure_error(rt.where, c2, l2, l3)
        ensure_error(rt.where, c2, l3, l1)
        ensure_error(rt.where, c3, l2, l3)

        # valid args
        assert_array_equal(rt.where(c1, l2, FastArray([4, 5])), FastArray([4, 5]))
        assert_array_equal(rt.where(c3, FastArray([4, 5, 6]), l1), FastArray([1, 1, 1]))
        assert_array_equal(rt.where(c3, 1, 2), FastArray([2, 2, 2]))
        assert_array_equal(rt.where(c3, l1, l1), FastArray([1, 1, 1]))

    def test_where_strided(self):
        dtype = np.dtype(float32)
        arr = np.array([1, 2, 3, 4, 5, 6], dtype=dtype)
        a = np.ndarray(buffer=arr, shape=(3,), strides=(8,), dtype=dtype)
        w = rt.where(a == 0, 0, a)
        assert_array_equal(w, FastArray([1, 3, 5]))

    def test_min_scalar_type(self):
        i8 = np.iinfo(np.int8)
        u8 = np.iinfo(np.uint8)
        i16 = np.iinfo(np.int16)
        u16 = np.iinfo(np.uint16)
        i32 = np.iinfo(np.int32)
        u32 = np.iinfo(np.uint32)
        i64 = np.iinfo(np.int64)
        u64 = np.iinfo(np.uint64)

        # 1st: (False, False) Don't promote invalids and don't prefer signed
        # 2nd: (True, False) Promote invalids and don't prefer signed
        # 3rd (False, True) Don't promote invalids but prefer signed
        # 4th (True, True) Promote invalids and prefer signed

        test_cases = [
            (0, np.dtype("uint8"), np.dtype("uint8"), np.dtype("int8"), np.dtype("int8")),
            (1, np.dtype("uint8"), np.dtype("uint8"), np.dtype("int8"), np.dtype("int8")),
            (-1, np.dtype("int8"), np.dtype("int8"), np.dtype("int8"), np.dtype("int8")),
            (i8.min, np.dtype("int8"), np.dtype("int16"), np.dtype("int8"), np.dtype("int16")),
            (i8.max, np.dtype("uint8"), np.dtype("uint8"), np.dtype("int8"), np.dtype("int8")),
            (u8.max, np.dtype("uint8"), np.dtype("uint16"), np.dtype("int16"), np.dtype("int16")),
            (i16.min, np.dtype("int16"), np.dtype("int32"), np.dtype("int16"), np.dtype("int32")),
            (i16.max, np.dtype("uint16"), np.dtype("uint16"), np.dtype("int16"), np.dtype("int16")),
            (u16.max, np.dtype("uint16"), np.dtype("uint32"), np.dtype("int32"), np.dtype("int32")),
            (i32.min, np.dtype("int32"), np.dtype("int64"), np.dtype("int32"), np.dtype("int64")),
            (i32.max, np.dtype("uint32"), np.dtype("uint32"), np.dtype("int32"), np.dtype("int32")),
            (u32.max, np.dtype("uint32"), np.dtype("uint64"), np.dtype("int64"), np.dtype("int64")),
            (i64.min, np.dtype("int64"), np.dtype("O"), np.dtype("int64"), np.dtype("O")),
            (i64.max, np.dtype("uint64"), np.dtype("uint64"), np.dtype("int64"), np.dtype("int64")),
            (u64.max, np.dtype("uint64"), np.dtype("O"), np.dtype("O"), np.dtype("O")),
        ]

        for inp, ex1, ex2, ex3, ex4 in test_cases:
            assert rt.min_scalar_type(inp, False, False) == ex1
            assert rt.min_scalar_type(inp, True, False) == ex2
            assert rt.min_scalar_type(inp, False, True) == ex3
            assert rt.min_scalar_type(inp, True, True) == ex4

        assert rt.min_scalar_type("string_value") == np.min_scalar_type("string_value")
        assert rt.min_scalar_type(b"string_value") == np.min_scalar_type(b"string_value")
        assert rt.min_scalar_type([1, 2, 3, 4]) == np.min_scalar_type([1, 2, 3, 4])
        assert rt.min_scalar_type(rt.FA([1, 2, 3, 4], dtype="uint64")) == np.dtype("uint64")
        assert rt.min_scalar_type(9.93620693e37) == np.min_scalar_type(9.93620693e37)
        assert rt.min_scalar_type(False) == np.min_scalar_type(False)
        assert rt.min_scalar_type(rt.Date("2020-08-04")) == np.dtype("int32")
        assert rt.min_scalar_type(rt.int64(1234)) == np.min_scalar_type(rt.int64(1234))

    def test_get_dtype(self):
        assert rt.get_dtype(rt.int64(-2)) == np.dtype("int64")
        assert rt.get_dtype(np.int64(0)) == np.dtype("int64")

        assert rt.get_dtype(10) == np.dtype("uint8")
        assert rt.get_dtype(255) == np.dtype("uint8")
        assert rt.get_dtype(-10) == np.dtype("int8")
        assert rt.get_dtype(123.45) == np.dtype("float16")
        assert rt.get_dtype("hello") == np.dtype("<U5")
        assert rt.get_dtype(b"hello") == np.dtype("S5")

        assert rt.get_dtype(rt.FA(10, dtype="int64")) == np.dtype("int64")

    def test_interp(self):
        x = interp(arange(3.0).astype(float32), [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        y = interp(arange(3.0), [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        z = np.interp(arange(3.0), [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert_array_equal(z, y, err_msg=f"Results did not match for where. {z} vs. {y}")

    def test_unique(self):
        symb = FastArray(
            # Top SPX constituents by weight as of 20200414
            [
                b"MSFT",
                b"AAPL",
                b"AMZN",
                b"FB",
                b"JNJ",
                b"GOOG",
                b"GOOGL",
                b"BRK.B",
                b"PG",
                b"JPM",
                b"V",
                b"INTC",
                b"UNH",
                b"VZ",
                b"MA",
                b"T",
                b"HD",
                b"MRK",
                b"PFE",
                b"PEP",
                b"BAC",
                b"DIS",
                b"KO",
                b"WMT",
                b"CSCO",
            ],
        )

        x1, y1, z1 = np.unique(symb, return_index=True, return_inverse=True)
        x2, y2, z2 = unique(symb, return_index=True, return_inverse=True)
        assert_array_equal(x1, x2, err_msg=f"Results did not match for unique. {x1} vs. {x2}")
        assert_array_equal(y1, y2, err_msg=f"Results did not match for unique index. {y1} vs. {y2}")
        assert_array_equal(
            z1,
            z2,
            err_msg=f"Results did not match for unique inverse. {z1} vs. {z2}",
        )
        x2, y2, z2 = unique(Cat(symb), return_index=True, return_inverse=True)
        assert_array_equal(x1, x2, err_msg=f"Results did not match for cat unique. {x1} vs. {x2}")
        assert_array_equal(
            y1,
            y2,
            err_msg=f"Results did not match for cat unique index. {y1} vs. {y2}",
        )
        assert_array_equal(
            z1,
            z2,
            err_msg=f"Results did not match for cat unique inverse. {z1} vs. {z2}",
        )

        # round 2
        symb = FastArray(
            [
                "NIHD",
                "SPY",
                "AAPL",
                "LOCO",
                "XLE",
                "MRNA",
                "JD",
                "JD",
                "QNST",
                "RUSL",
                "USO",
                "TSLA",
                "NVDA",
                "GLD",
                "GLD",
                "ZS",
                "WDC",
                "AGEN",
                "AMRS",
                "AAPL",
                "SMH",
                "PYPL",
                "AAPL",
                "SQQQ",
                "GLD",
            ],
            unicode=True,
        )

        x1, y1, z1 = np.unique(symb, return_index=True, return_inverse=True)

        x2, y2, z2 = unique(symb, return_index=True, return_inverse=True)
        assert_array_equal(x1, x2, err_msg=f"Results did not match for unique. {x1} vs. {x2}")
        assert_array_equal(y1, y2, err_msg=f"Results did not match for unique index. {y1} vs. {y2}")
        assert_array_equal(
            z1,
            z2,
            err_msg=f"Results did not match for unique inverse. {z1} vs. {z2}",
        )

        x2, y2, z2 = unique(Cat(symb), return_index=True, return_inverse=True)
        assert_array_equal(x1, x2, err_msg=f"Results did not match for cat unique. {x1} vs. {x2}")
        assert_array_equal(
            y1,
            y2,
            err_msg=f"Results did not match for cat unique index. {y1} vs. {y2}",
        )
        assert_array_equal(
            z1,
            z2,
            err_msg=f"Results did not match for cat unique inverse. {z1} vs. {z2}",
        )

        # round 3
        x2, y2, z2 = unique(symb, return_index=True, return_inverse=True, lex=True)
        assert_array_equal(x1, x2, err_msg=f"Results did not match for cat unique. {x1} vs. {x2}")
        assert_array_equal(
            y1,
            y2,
            err_msg=f"Results did not match for cat unique index. {y1} vs. {y2}",
        )
        assert_array_equal(
            z1,
            z2,
            err_msg=f"Results did not match for cat unique inverse. {z1} vs. {z2}",
        )

    def test_alignmk(self):
        ds1 = rt.Dataset()
        ds1["Time"] = [0, 1, 4, 6, 8, 9, 11, 16, 19, 30]
        ds1["Px"] = [10, 12, 15, 11, 10, 9, 13, 7, 9, 10]

        ds2 = rt.Dataset()
        ds2["Time"] = [0, 0, 5, 7, 8, 10, 12, 15, 17, 20]
        ds2["Vols"] = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

        # Categorical keys
        ds1["Ticker"] = rt.Categorical(["Test"] * 10)
        ds2["Ticker"] = rt.Categorical(["Test", "Blah"] * 5)
        res = alignmk(ds1.Ticker, ds2.Ticker, ds1.Time, ds2.Time)
        target = rt.FastArray([0, 0, 0, 2, 4, 4, 4, 6, 8, 8])
        assert_array_equal(res, target)

        # char array keys
        ds1["Ticker"] = rt.FastArray(["Test"] * 10)
        ds2["Ticker"] = rt.FastArray(["Test", "Blah"] * 5)
        res = alignmk(ds1.Ticker, ds2.Ticker, ds1.Time, ds2.Time)
        target = rt.FastArray([0, 0, 0, 2, 4, 4, 4, 6, 8, 8])
        assert_array_equal(res, target)

    @pytest.mark.xfail(
        get_rc_version() < parse_version("1.14.2a"), strict=False, reason="Broken nan handling and unsupported uints"
    )
    @pytest.mark.parametrize(
        "nans",
        [
            pytest.param(-1, id="left"),
            pytest.param(0, id="both"),
            pytest.param(1, id="right"),
        ],
    )
    @pytest.mark.parametrize(
        "tdt",
        [
            pytest.param(np.dtype(int32), id="int32"),
            pytest.param(np.dtype(int64), id="int64"),
            pytest.param(np.dtype(float32), id="float32"),
            pytest.param(np.dtype(float64), id="float64"),
            pytest.param(np.dtype(uint32), id="uint32"),
            pytest.param(np.dtype(uint64), id="uint64"),
        ],
    )
    def test_alignmk_invalids(self, tdt, nans):
        ds1 = rt.Dataset(dict(time=rt.FA([1, 2, np.nan if nans <= 0 else 3, 4]).astype(tdt), data1=[1, 2, 3, 4]))
        ds2 = rt.Dataset(dict(time=rt.FA([1, 2, np.nan if nans >= 0 else 3, 4]).astype(tdt), data2=[1, 2, 3, 4]))
        res = alignmk(ds1.data1, ds2.data2, ds1.time, ds2.time)
        target = rt.FA([0, 1, rt.nan, 3])
        assert_array_equal(res, target)

    def test_sample(self):
        # Test Dataset.sample
        ds = rt.Dataset({"num": [1, 2, 3, 4, 5], "str": ["ab", "bc", "cd", "de", "ef"]})
        ds_sample = ds.sample(3, rt.FA([True, True, True, False, True]), seed=1)
        ds_sample_expected = rt.Dataset({"num": [1, 2, 5], "str": ["ab", "bc", "ef"]})
        assert ds_sample.keys() == ds_sample_expected.keys()
        for col_name in ds_sample_expected.keys():
            assert_array_equal(
                ds_sample_expected[col_name], ds_sample[col_name], err_msg=f"Column '{col_name}' differs."
            )

        # Test FastArray.sample
        fa = rt.FA([1, 2, 3, 4, 5])
        fa_sample = fa.sample(2, rt.FA([False, True, True, False, True]), seed=1)
        fa_sample_expected = rt.FA([2, 3])
        assert_array_equal(fa_sample_expected, fa_sample)

        # Test overflow
        fa_sample = fa.sample(10, rt.FA([False, True, False, False, True]), seed=1)
        fa_sample_expected = rt.FA([2, 5])
        assert_array_equal(fa_sample_expected, fa_sample)

        # Test no filter
        fa_sample = fa.sample(2, seed=1)
        fa_sample_expected = rt.FA([2, 3])
        assert_array_equal(fa_sample_expected, fa_sample)

        # Test fancy index
        fa_sample = fa.sample(2, rt.FA([1, 3, 4]), seed=1)
        fa_sample_expected = rt.FA([2, 4])
        assert_array_equal(fa_sample_expected, fa_sample)

    def test_hstack_returns_same_type(self):
        # Create some instances of the same derived array type.
        # TODO: Implement another version of this test using a derived array type (maybe define one just for this test)
        #       that does not define it's own 'hstack' method. Or, test what happens if we e.g. hstack something like
        #       rt.Date() and a list (that could be used to create an array of that type).
        array_type = rt.Date

        arrs = [
            array_type(
                ["20210108", "20210108", "20210115", "20210115", "20210115", "20210122", "20210129", "20210129"]
            ),
            array_type(["20200102"]),
            array_type(["20190103", "20190204", "20190305"]),
        ]
        total_len = sum(map(lambda x: len(x), arrs))

        # rt.hstack() should return an instance of the same type.
        result = rt.hstack(arrs)
        assert type(result) == array_type
        assert len(result) == total_len


class TestHStackAny:
    """Tests for the rt.hstack_any (a.k.a. rt.stack_rows) function."""

    _fa1 = rt.FastArray([100, 200])
    _fa2 = rt.FastArray([111, 222])
    _dtn1 = rt.DateTimeNano("2021-10-12 01:02:03", from_tz="UTC")
    _dtn2 = rt.DateTimeNano("1980-03-04 13:14:15", from_tz="UTC")
    _ts1 = _dtn1 - _dtn2
    _ts2 = _dtn2 - _dtn1
    _ds1 = rt.Dataset({"a": 11})
    _ds2 = rt.Dataset({"b": 22})
    _ds3 = rt.Dataset({})
    _ds4 = rt.Dataset({"a": [1, 2], "b": [3, 4]})
    _ds5 = rt.Dataset({"a": [5, 6]})
    _pds1 = rt.PDataset(_ds1)
    _pds2 = rt.PDataset(_ds2)

    @pytest.mark.parametrize(
        "inputs,expected",
        [
            pytest.param([_fa1, _fa2], rt.FastArray, id="FastArray,FastArray"),
            pytest.param([_dtn1, _dtn2], rt.DateTimeNano, id="DateTimeNano,DateTimeNano"),
            pytest.param([_dtn1, _dtn2], rt.DateTimeNano, id="DateTimeNano,DateTimeNano"),
            pytest.param([_ts1, _ts2], rt.TimeSpan, id="TimeSpan,TimeSpan"),
            pytest.param([_ds1, _ds2], rt.Dataset, id="Dataset,Dataset"),
            pytest.param([_pds1, _pds2], None, id="PDataset,PDataset"),  # notyet
            pytest.param([_dtn1, _ts2], None, id="DateTimeNano,TimeSpan"),  # neither is base
            pytest.param([_fa1, _dtn2], rt.FastArray, id="FastArray,DateTimeNano"),
            pytest.param([_ts1, _fa2], rt.FastArray, id="TimeSpan,FastArray"),
            pytest.param([_ds1, _pds2], rt.Dataset, id="Dataset,PDataset"),
            pytest.param([_pds1, _ds2], rt.Dataset, id="PDataset,Dataset"),
            pytest.param([_fa1, _ds2], None, id="FastArray,Dataset"),
            pytest.param([_ds1, _ds3], rt.Dataset, id="Dataset,Dataset"),
            pytest.param([_ds4, _ds5], rt.Dataset, id="Dataset,Dataset invalid"),
            pytest.param([_ds4, _ds3], rt.Dataset, id="Dataset,Dataset invalid_empty"),
        ],
    )
    @pytest.mark.parametrize("destroy", [pytest.param(True, id="destroy"), pytest.param(False, id="destroy_false")])
    def test_hstack_any(self, inputs, expected, destroy):
        if expected is None:
            with pytest.raises(Exception):
                rt.hstack_any(inputs, destroy=destroy)
        else:
            result = rt.hstack_any(inputs, destroy=destroy)
            assert type(result) == expected


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
            pytest.param(math.nan, id="scalar"),
            pytest.param([math.nan, math.nan, math.nan, math.nan, math.nan], id="list-float"),
            pytest.param({np.nan}, id="set-float", marks=pytest.mark.skip("Broken in both numpy and riptable.")),
            pytest.param(np.full(100, np.nan, dtype=np.float32), id="ndarray-float"),
            pytest.param(rt.full(100, np.nan, dtype=np.float32), id="FastArray-float")
            # TODO: Ordered Categorical with all invalids
        ],
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
            pytest.param([], id="list-float"),
            pytest.param({}, id="set-float", marks=pytest.mark.skip("Broken in both numpy and riptable.")),
            pytest.param(np.array([], dtype=np.float32), id="ndarray-float"),
            pytest.param(
                rt.FastArray([], dtype=np.float32),
                id="FastArray-float",
                marks=pytest.mark.xfail(
                    reason="RIP-417: The call to riptide_cpp via the ledger returns None, which then causes the isnan() to raise a TypeError. This needs to be fixed so we raise an error like numpy (either by checking for this and raising the exception, or fixing the way the function punts to numpy."
                ),
            )
            # TODO: Empty ordered Categorical (create with some categories but an empty backing array).
            # TODO: Empty Date array (representing a FastArray subclass)
        ],
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
            ["PA", "NY", "", "NY", "AL", "LA", "PA", "", "CA", "IL", "IL", "FL", "FL", "LA"], ordered=True
        )
        assert cat.ordered

        result = rt.nanmax(cat)

        # The result should either be a Python string, a numpy string scalar, or a Categorical scalar (if we implement one).
        is_py_str = isinstance(result, (bytes, str))
        is_np_scalar = isinstance(result, (np.bytes_, np.str_))
        is_rt_cat = isinstance(result, rt.Categorical)
        assert is_py_str or is_np_scalar or is_rt_cat

        # Check the result is correct.
        assert result == "FL"


class TestNanmin:
    """Tests for the rt.nanmin function."""

    @pytest.mark.parametrize(
        "arg",
        [
            pytest.param(math.nan, id="scalar"),
            pytest.param([math.nan, math.nan, math.nan, math.nan, math.nan], id="list-float"),
            pytest.param({np.nan}, id="set-float", marks=pytest.mark.skip("Broken in both numpy and riptable.")),
            pytest.param(np.full(100, np.nan, dtype=np.float32), id="ndarray-float"),
            pytest.param(rt.full(100, np.nan, dtype=np.float32), id="FastArray-float")
            # TODO: Ordered Categorical with all invalids
        ],
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
            pytest.param([], id="list-float"),
            pytest.param({}, id="set-float", marks=pytest.mark.skip("Broken in both numpy and riptable.")),
            pytest.param(np.array([], dtype=np.float32), id="ndarray-float"),
            pytest.param(
                rt.FastArray([], dtype=np.float32),
                id="FastArray-float",
                marks=pytest.mark.xfail(
                    reason="RIP-417: The call to riptide_cpp via the ledger returns None, which then causes the isnan() to raise a TypeError. This needs to be fixed so we raise an error like numpy (either by checking for this and raising the exception, or fixing the way the function punts to numpy."
                ),
            )
            # TODO: Empty ordered Categorical (create with some categories but an empty backing array).
            # TODO: Empty Date array (representing a FastArray subclass)
        ],
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
            ["PA", "NY", "", "NY", "AL", "LA", "PA", "", "CA", "IL", "IL", "FL", "FL", "LA"], ordered=True
        )
        assert cat.ordered

        result = rt.nanmin(cat)

        # The result should either be a Python string, a numpy string scalar, or a Categorical scalar (if we implement one).
        is_py_str = isinstance(result, (bytes, str))
        is_np_scalar = isinstance(result, (np.bytes_, np.str_))
        is_rt_cat = isinstance(result, rt.Categorical)
        assert is_py_str or is_np_scalar or is_rt_cat

        # Check the result is correct.
        assert result == "PA"


if __name__ == "__main__":
    tester = unittest.main()
