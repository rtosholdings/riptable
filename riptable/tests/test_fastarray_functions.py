import numpy as np
import riptable as rt
import pandas as pd
import itertools
from numpy.random import default_rng

try:
    import bottleneck as bn
except ModuleNotFoundError:
    bn = None


arithmetic_functions = [
    "__add__",
    "__iadd__",
    "__sub__",
    "__isub__",
    "__mul__",
    "__imul__",
    "__floordiv__",
    "__ifloordiv__",
    "__truediv__",
    "__itruediv__",
    "__mod__",
    "__imod__",
    "__pow__",
    "__ipow__",
]

comparison_functions = ["__lt__", "__gt__", "__ge__", "__le__", "__eq__", "__ne__"]

##accepts array as param
trig_functions = [
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "hypot",
    "arctan2",
    "degrees",
    "radians",
    "unwrap" "sinh",
    "deg2rad",
    "deg2deg",
]

##accepts array as param
# hyp_functions = [
#
#     'sinh',
#     'cosh',
#     'tanh',
#     'arccosh',
#     'arctanh'
# ]
#

reduction_functions = [
    "prod",
    "sum",
    "nanprod",
    "nansum",
    "cumprod",
    "cumsum",
    "nancumprod",
    "diff",
    "ediff1d",
    "gradient",
    # 'cross',
    # 'trapz'
]

exponent_log_functions = [
    "exp",
    "expm1",
    "exp2",
    "log",
    "log10",
    "log2",
    "log1p",
    # 'logaddexp',
    # 'logaddexp2'
]

type_list = [
    # np.bool,          ## not a numeric type
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    # np.float16,       ## not supported
    np.float32,
    np.float64,
    # np.complex64,     ## not supported
    # np.complex128     ## not supported
]

TEST_SIZE = 100
binary_functions = [arithmetic_functions, comparison_functions]
unary_functions = [exponent_log_functions, reduction_functions]

import copy
import unittest
from decimal import Decimal

import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

from ..rt_numpy import np_rolling_nanquantile


@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", rt.FastArray([False, False, True, False, True])),
        ("last", rt.FastArray([True, True, False, False, False])),
        (False, rt.FastArray([True, True, True, False, True])),
    ],
)
def test_duplicated_keep(keep, expected):
    fa = rt.FastArray([0, 1, 1, 2, 0])

    result = fa.duplicated(keep=keep)
    for r, e in zip(result, expected):
        assert r == e


@pytest.mark.parametrize(
    "decay_rate, filter, reset, dtype_override, expected",
    [
        # Decay rate == 0 means there is no decay; since there's no decay,
        # there's effectively no time component to the EMA so we just have a cumsum.
        (
            0,
            None,
            None,
            None,
            rt.FA([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        ),
        # Simple use case with a 50% decay rate per time-unit (the actual time unit used doesn't matter)
        (
            0.5,
            None,
            None,
            None,
            rt.FA(
                [
                    1.0,
                    1.606531,
                    2.606531,
                    1.958889,
                    2.188126,
                    2.327166,
                    2.812398,
                    1.230856,
                    2.200466,
                    2.757108,
                ]
            ),
        ),
        # Simple use case with a 50% decay rate per time-unit; for this case,
        # we force the dtype of the output array using an np.dtype instance.
        (
            0.5,
            None,
            None,
            np.float64,
            rt.FA(
                [
                    1.0,
                    1.606531,
                    2.606531,
                    1.958889,
                    2.188126,
                    2.327166,
                    2.812398,
                    1.230856,
                    2.200466,
                    2.757108,
                ]
            ),
        ),
        # Simple use case with a 50% decay rate per time-unit; for this case,
        # we force the dtype of the output array using the string name of a dtype.
        (
            0.5,
            None,
            None,
            "float64",
            rt.FA(
                [
                    1.0,
                    1.606531,
                    2.606531,
                    1.958889,
                    2.188126,
                    2.327166,
                    2.812398,
                    1.230856,
                    2.200466,
                    2.757108,
                ]
            ),
        ),
        # Test case for providing a filter (mask) to select only certain elements.
        (
            0.5,
            rt.FA([True, True, False, True, True, False, False, True, True, True]),
            None,
            None,
            rt.FA(
                [
                    1.0,
                    1.606531,
                    1.606531,
                    1.59101,
                    1.964996,
                    1.19183,
                    0.928198,
                    1.076191,
                    2.04962,
                    2.636655,
                ]
            ),
        ),
        # Test case for providing a filter (mask) and a reset mask.
        (
            0.9,
            rt.FA([True, True, False, True, True, False, False, True, True, True]),
            rt.FA([False, True, False, False, False, False, False, False, False, True]),
            None,
            rt.FA(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.165299,
                    1.473775,
                    0.599192,
                    0.382062,
                    1.004244,
                    1.960055,
                    1.0,
                ]
            ),
        ),
    ],
)
def test_ema_decay(decay_rate, filter, reset, dtype_override, expected):
    data = rt.ones(10)
    times = rt.FastArray([0, 1, 1, 3, 4, 5, 5.5, 10.5, 10.55, 11])

    # Call ema_decay.
    # Don't override the default dtype unless we actually have an override.
    # We don't bother doing this for the other arguments because they're either
    # non-optional or already default to None.
    if dtype_override is None:
        result = data.ema_decay(times, decay_rate, filter=filter, reset=reset)
    else:
        result = data.ema_decay(times, decay_rate, filter=filter, reset=reset, dtype=dtype_override)

    # Check the result against the expected values.
    assert_array_almost_equal(result, expected)


def test_ema_decay_reset_nofilter():
    # Create a FastArray instance to call ema_decay() with;
    # the data itself doesn't matter, since we're just looking to
    # check how the function validates preconditions.
    count = 10
    data = rt.ones(count, dtype=np.float32)
    times = rt.ones(count, dtype=np.float32)
    reset = np.full(count, True, dtype=np.bool_)

    # Check that calling ema_decay with a reset mask but no filter raises a warning
    # to notify the user that the reset mask will be ignored.
    with pytest.raises(UserWarning):
        data.ema_decay(times, 1.0, reset=reset)


def test_ema_decay_requires_matching_time_len():
    count = 10
    data = rt.ones(count, dtype=np.float32)

    # Check that calling ema_decay with a time array whose length is smaller than the input array length
    # fails with an error.
    smaller_time = np.arange(count - 1).astype(np.float32)
    with pytest.raises(ValueError):
        data.ema_decay(smaller_time, 1.0)

    # Check that calling ema_decay with a time array whose length is larger than the input array length
    # fails with an error.
    larger_time = np.arange(count * 2).astype(np.float32)
    with pytest.raises(ValueError):
        data.ema_decay(larger_time, 1.0)


def test_ema_decay_requires_matching_filter_len():
    count = 10
    data = rt.ones(count, dtype=np.float32)
    times = rt.ones(count, dtype=np.float32)

    # Check that calling ema_decay with a filter whose length is smaller than the input array length
    # fails with an error.
    smaller_filter = np.full(count - 1, True, dtype=np.bool_)
    with pytest.raises(ValueError):
        data.ema_decay(times, 1.0, filter=smaller_filter)

    # Check that calling ema_decay with a filter whose length is larger than the input array length
    # fails with an error.
    larger_filter = np.full(count * 2, False, dtype=np.bool_)
    with pytest.raises(ValueError):
        data.ema_decay(times, 1.0, filter=larger_filter)


def test_ema_decay_requires_matching_reset_len():
    count = 10
    data = rt.ones(count, dtype=np.float32)
    times = rt.ones(count, dtype=np.float32)
    filt = np.full(count, True, dtype=np.bool_)

    # Check that calling ema_decay with a reset-mask whose length is smaller than the input array length
    # fails with an error.
    smaller_reset = np.full(count - 1, True, dtype=np.bool_)
    with pytest.raises(ValueError):
        data.ema_decay(times, 1.0, filter=filt, reset=smaller_reset)

    # Check that calling ema_decay with a reset-mask whose length is larger than the input array length
    # fails with an error.
    larger_reset = np.full(count * 2, False, dtype=np.bool_)
    with pytest.raises(ValueError):
        data.ema_decay(times, 1.0, filter=filt, reset=larger_reset)


@pytest.mark.parametrize(
    "input,bools,expected",
    [
        (
            [[10, 1], [11, 2], [13, 3]],
            True,
            [[[10, 1], [11, 2], [13, 3]]],
        ),
        (
            [[10, 1], [11, 2], [13, 3]],
            [True, False, True],
            [[10, 1], [13, 3]],
        ),
        (
            [[10, 1], [11, 2], [13, 3]],
            [[True, False], [False, True], [True, False]],
            [10, 2, 13],
        ),
    ],
)
def test_boolean_indexing(input, bools, expected):
    # we want contig bool array
    f = np.array(bools)
    if not f.flags.f_contiguous:
        f = np.asfortranarray(bools)
    assert f.flags.f_contiguous

    na = np.asfortranarray(input)
    assert na.flags.f_contiguous
    assert_array_equal(na[f], expected)

    fa = rt.FA(na)
    assert fa.flags.f_contiguous
    fr = fa[f]
    assert_array_equal(super(rt.FA, fr), expected)  # use ndarray element indexing for assertion


class test_numpy_functions(unittest.TestCase):
    def assert_equal(self, lv, rv):
        def equal(l, r):
            def isNaN(value):
                return value != value

            def not_bool(value):
                return not isinstance(value, bool) and not isinstance(value, np.bool_)

            if not_bool(l) and not_bool(r):
                epsilon = 0.003  ## coosing a large value due to float32/64 conversions

                assert abs(l - r) < epsilon or (isNaN(l) and isNaN(r))
            else:
                assert l == r

        if hasattr(lv, "__len__"):
            assert len(lv) == len(rv)
            length = len(lv)
            for i in range(0, length):
                equal(lv[i], rv[i])
        else:
            equal(lv, rv)

    # TODO pytest parameterize across unary_functions, binary_functions, and function_set
    def test_operations(self):
        A_array = np.random.rand(TEST_SIZE) + 10
        B_array = np.random.rand(TEST_SIZE) + 10
        for type_ in type_list:
            a = np.array(A_array, dtype=type_)
            b = np.array(B_array, dtype=type_)

            x = rt.FastArray(np.array(A_array, dtype=type_))
            y = rt.FastArray(np.array(B_array, dtype=type_))

            for function_set in binary_functions:
                for function in function_set:
                    # print('function - ', function)
                    if not np.issubdtype(a.dtype, np.integer) and function == "__truediv__":
                        np_out = getattr(a, function)(b)
                        sf_out = getattr(x, function)(y)

                        self.assert_equal(np_out, sf_out)

            for function_set in unary_functions:
                for function in function_set:
                    ##these require 'resets' due to division
                    a = copy.copy(A_array)
                    b = copy.copy(B_array)

                    x = rt.FastArray(copy.copy(A_array))
                    y = rt.FastArray(copy.copy(B_array))

                    # print('function - ', function)
                    np_out = getattr(np, function)(a)
                    sf_out = getattr(np, function)(x)

                    self.assert_equal(np_out, sf_out)

    # def test_rolling_ops(self):
    #     ###need to add tests for rolling_nan versions
    #
    #     import pandas as pd
    #
    #     A_array = abs(np.random.rand(10))
    #     B_array = abs(np.random.rand(10))
    #
    #     a = pd.DataFrame(copy.copy(A_array))
    #     b = pd.DataFrame(copy.copy(B_array))
    #
    #     x = rt.FastArray(copy.copy(A_array))
    #     y = rt.FastArray(copy.copy(B_array))
    #
    #     window = 3
    #
    #     roll_a = a.rolling(window).sum()
    #     roll_x = x.rolling_sum(window)
    #
    #     print('output type is ', print(type(roll_x)))
    #
    #     self.assert_equal(roll_a[window:], roll_x[window:])
    #
    #     roll_a = a.rolling(window).mean()
    #     roll_x = x.rolling_mean(window)
    #     self.assert_equal(roll_a[window:], roll_x[window:])
    #
    #     roll_a = a.rolling(window).var()
    #     roll_x = x.rolling_std(window)
    #     self.assert_equal(roll_a[window:], roll_x[window:])
    #
    #     roll_a = a.rolling(window).std()
    #     roll_x = x.rolling_std(window)
    #     self.assert_equal(roll_a[window:], roll_x[window:])

    # TODO pytest parameterize type_list
    def test_reductions(self):
        for type_ in type_list:
            A_array = np.array(
                np.random.randint(100, size=TEST_SIZE) + np.array(np.random.rand(TEST_SIZE)),
                dtype=type_,
            )
            B_array = np.copy(A_array)

            a = A_array  ##copy.copy(A_array)
            x = rt.FastArray(B_array)  ##rt.FastArray(copy.copy(A_array))

            # print(a)
            # print(x)

            # print(type(a.sum()))
            # print(type(x.sum()))

            # print(type(a))
            # print(type(x))

            # print('current data type ==', type_)

            self.assert_equal(a.sum(), x.sum())
            self.assert_equal(a.min(), x.min())
            self.assert_equal(a.max(), x.max())

            ddofs = 3

            for i in range(0, ddofs):
                self.assert_equal(a.var(ddof=i), x.var(ddof=i))
                self.assert_equal(a.std(ddof=i), x.std(ddof=i))

            self.assert_equal(a.mean(), x.mean())

    def test_rounding_ops(self):
        for type_ in type_list:

            def rand_ary():
                return np.array(
                    np.random.randint(0, 100, size=TEST_SIZE) + np.random.rand(TEST_SIZE),
                    dtype=type_,
                )

            rounding_fuctions = [
                "__abs__",
                "around",
                "round_",
                "rint",
                "fix",
                "floor",
                "ceil",
                "trunc",
            ]

            a = rand_ary()
            self.assert_equal(abs(a), abs(rt.FastArray(a)))
            a = rand_ary()
            self.assert_equal(np.floor(a), rt.floor(rt.FastArray(a)))
            a = rand_ary()
            self.assert_equal(np.floor(a), rt.floor(rt.FastArray(a)))
            # a = rand_ary()
            # self.assert_equal(a.round_(), rt.FastArray(a).round_())
            # a = rand_ary()
            # self.assert_equal(a.rint(), rt.FastArray(a).rint())
            # # a = rand_ary()
            # # self.assert_equal(a.fix(), rt.FastArray(a).fix())
            # a = rand_ary()
            # self.assert_equal(np.floor(a), rt.floor(rt.FastArray(a)))

    def test_operations_recycle_off(self):
        rt.FastArray._ROFF()
        self.test_operations()
        rt.FastArray._RON()

    # def test_rolling_recycle_off(self):
    #     rt.FastArray._ROFF()
    #     self.test_rolling_ops()
    #     rt.FastArray._RON()

    def test_reductions_recycle_off(self):
        rt.FastArray._ROFF()
        self.test_reductions()
        rt.FastArray._RON()

    def test_operations_threading_off(self):
        rt.FastArray._TOFF()
        self.test_operations()
        rt.FastArray._TON()

    # def test_rolling_threading_off(self):
    #     rt.FastArray._TOFF()
    #     self.test_rolling_ops()
    #     rt.FastArray._RON()

    def test_reductions_threading_off(self):
        rt.FastArray._TOFF()
        self.test_reductions()
        rt.FastArray._TON()

    def test_operations_all_off(self):
        rt.FastArray._TOFF()
        rt.FastArray._ROFF()

        self.test_operations()
        rt.FastArray._TON()
        rt.FastArray._RON()

    #
    # def test_rolling_all_off(self):
    #     rt.FastArray._TOFF()
    #     rt.FastArray._ROFF()
    #     self.test_rolling_ops()
    #     rt.FastArray._TON()
    #     rt.FastArray._RON()

    def test_reductions_all_off(self):
        rt.FastArray._TOFF()
        rt.FastArray._ROFF()
        self.test_reductions()
        rt.FastArray._TON()
        rt.FastArray._RON()


def rolling_quantile_params_generator(dtypes, max_N, with_nans, with_infs):
    for dtype in dtypes:
        # N = 1 case
        N = 1
        window = 1
        for quantile in [0.81, 0.44, 0.321]:
            curr_frac_options = [(0, 0, 0)]
            if with_nans:
                curr_frac_options += [(1, 0, 0)]
            if with_infs:
                curr_frac_options += [(0, 1, 0), (0, 0, 1)]
            for nan_fraction, inf_fraction, minus_inf_fraction in curr_frac_options:
                yield N, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype

        frac_lists = [
            [0.0, 0.24, 1],
            [0.0, 0.14, 0.9, 1.0],
            [0.0, 0.2, 0.87, 1.0],
        ]

        frac_options = [element for element in itertools.product(*frac_lists) if sum(element) <= 1]
        frac_options_only_nans = [(x, 0, 0) for x in frac_lists[0]]

        curr_frac_options = [(0, 0, 0)]
        if with_nans:
            curr_frac_options = frac_options_only_nans
        if with_infs:
            curr_frac_options = frac_options

        # window = 1 case
        N = 500
        window = 1
        for quantile in [0.13, 0.814]:
            for nan_fraction, inf_fraction, minus_inf_fraction in curr_frac_options:
                yield N, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype

        # window > N case
        N = 500
        window = 700
        for quantile in [0.512, 0.90]:
            for nan_fraction, inf_fraction, minus_inf_fraction in curr_frac_options:
                yield N, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype

        # quantile = 0., 1. cases
        N = 666
        window = 111
        for quantile in [0.0, 1.0]:
            for nan_fraction, inf_fraction, minus_inf_fraction in curr_frac_options:
                yield N, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype

        for N, window in [(20, 2), (50, 5), (154, 51), (812, 761), (1_001, 43), (10_000, 6_000), (10_000, 431)]:
            if N > max_N:
                continue
            for quantile in [0.17, 0.435, 0.82]:
                for nan_fraction, inf_fraction, minus_inf_fraction in curr_frac_options:
                    yield N, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype


from .test_groupbyops import quantile_params_generator


@pytest.mark.parametrize(
    "N, dummy, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype",
    quantile_params_generator(
        [np.int64, np.float64], max_N=11_000, windowed=True, with_cats=False, with_nans=True, with_infs=False, seed=150
    ),
)
def test_rolling_quantile(N, dummy, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype):
    this_seed = (N * window * int(quantile * 1000) * int(nan_fraction * 1000)) % 10_000
    rng = default_rng(this_seed)

    def err_msg():
        return f"""
quantile = {quantile}
N = {N}
window = {window}
nan_fraction = {nan_fraction}
dtype = {dtype}"""

    window = min(window, N)

    # tests with pandas rolling series (without NaNs)
    A_array = rng.random(N)

    a = pd.DataFrame(copy.copy(A_array))

    x = rt.FastArray(copy.copy(A_array))
    z = copy.copy(A_array)

    roll_a = a.rolling(window).quantile(quantile=quantile, interpolation="midpoint")
    roll_x = x.rolling_quantile(q=quantile, window=window)

    assert_array_almost_equal(roll_a[0].values[window:], roll_x[window:], err_msg=err_msg())

    # cases when there are NaNs are harder to test
    # Here, test median only (using bottleneck.move_median, which handles NaNs)
    # more full tests in test_rolling_quantile_infs
    if bn is not None:
        B_array = rng.random(N)
        B_array[rng.random(N) < nan_fraction] = np.nan

        y = rt.FastArray(copy.copy(B_array))

        roll_b = bn.move_median(copy.copy(B_array), window=window, min_count=1)
        roll_x = y.rolling_quantile(q=0.5, window=window)

        assert_array_almost_equal(roll_b, roll_x, err_msg=err_msg())


@pytest.mark.parametrize(
    "N, dummy, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype",
    quantile_params_generator(
        [np.float64], max_N=1_001, windowed=True, with_cats=False, with_nans=True, with_infs=True, seed=250
    ),
)
def test_rolling_quantile_infs(N, dummy, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype):
    this_seed = (
        (N * window * int(quantile * 1000) * int(nan_fraction * 1000))
        * int(inf_fraction * 1000)
        * int(minus_inf_fraction * 1000)
        % 10_000
    )
    rng = default_rng(this_seed)

    def err_msg():
        return f"""
N = {N}
window = {window}
quantile = {quantile}
nan_fraction = {nan_fraction}
inf_fraction = {inf_fraction}
minus_inf_fraction = {minus_inf_fraction}
dtype = {dtype}"""

    window = min(window, N)

    # use very slow strided np function for full testing

    C_array = rng.uniform(-10_000, 10_000, size=N).astype(dtype)
    rand_idxs = rng.random(size=N)
    C_array[rand_idxs < (inf_fraction + minus_inf_fraction + nan_fraction)] = np.nan
    if dtype not in [np.int32, np.int64]:
        # infs only work for non-intergal
        C_array[rand_idxs < (inf_fraction + minus_inf_fraction)] = -rt.inf
        C_array[rand_idxs < (inf_fraction)] = rt.inf
    np_data = copy.copy(C_array)
    rt_data = rt.FA(copy.copy(C_array)).astype(dtype)

    roll_np = np_rolling_nanquantile(a=np_data, q=quantile, window=window)
    roll_rt = rt_data.rolling_quantile(q=quantile, window=window)

    assert_array_almost_equal(roll_rt[window - 1 :], roll_np, err_msg=err_msg())


@pytest.mark.parametrize(
    "N, dummy, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype",
    quantile_params_generator(
        [np.int32, np.int64], max_N=1_001, windowed=True, with_cats=False, with_nans=True, with_infs=False, seed=350
    ),
)
def test_rolling_quantile_integral(N, dummy, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype):
    this_seed = (N * window * int(quantile * 1000) * int(nan_fraction * 1000)) % 10_000
    rng = default_rng(this_seed)

    def err_msg():
        return f"""
N = {N}
window = {window}
quantile = {quantile}
nan_fraction = {nan_fraction}
dtype = {dtype}"""

    window = min(window, N)

    # use very slow strided np function for full testing

    C_array = rng.uniform(-10_000, 10_000, size=N).astype(dtype).astype(np.float64)
    rand_idxs = rng.random(size=N)
    C_array[rand_idxs < nan_fraction] = np.nan

    np_data = copy.copy(C_array)
    rt_data = rt.FA(copy.copy(C_array)).astype(dtype)

    roll_np = np_rolling_nanquantile(a=np_data, q=quantile, window=window)
    roll_rt = rt_data.rolling_quantile(q=quantile, window=window)

    assert_array_almost_equal(roll_rt[window - 1 :], roll_np, err_msg=err_msg())


if __name__ == "__main__":
    tester = unittest.main()
