"""Unittests for riptable module level functions such as any, all, average, percentile, quantile, etc."""
import pytest
import numpy as np
import riptable as rt
import itertools
from numpy.testing import assert_array_equal, assert_almost_equal
from riptable.Utils.teamcity_helper import is_running_in_teamcity


class TestCat2Keys:
    @pytest.mark.parametrize(
        "key1,key2",
        [
            (list('abc'), list('xyz')),
            (np.array(list('abc')), np.array(list('xyz'))),
            (np.array(list('abc')), list('xyz')),
            # add the following test cases for nested list of arrays and categoricals
            # ([FA(list('abc')), FA(list('def'))], [FA(list('uvw')), FA(list('xyz'))])
            # ([FA('a'), FA('b'), FA('c')], [FA('x'), FA('y'), FA('z')]),
            # (rt.Cat(list('abc'), ordered=True), rt.Cat(list('xyz'), ordered=True)),
        ],
    )
    def test_cat2keys(self, key1, key2):
        multi_cat = rt.cat2keys(key1, key2)
        assert len(key1) == len(key2)  # add test to check different length lists

        # these are the expected entries in the multi key categorical dictionary
        n = len(key1)
        expected_key1 = rt.FastArray([k for _ in range(n) for k in key1])
        expected_key2 = rt.FastArray([k for k in key2 for _ in range(n)])

        key_itr = iter(multi_cat.category_dict)
        actual_key1 = multi_cat.category_dict[next(key_itr)]
        actual_key2 = multi_cat.category_dict[next(key_itr)]

        assert_array_equal(expected_key1, actual_key1)
        assert_array_equal(expected_key2, actual_key2)

        # Taking the entries one by one of expected_key1 and expected_key2 should produce the
        # cartesian product of key1 and key2.
        expected_product = [k1 + k2 for k1, k2 in itertools.product(key1, key2)]
        actual_product = np.array(
            sorted([k1 + k2 for k1, k2 in zip(actual_key1, actual_key2)]), dtype="U2"
        )
        assert_array_equal(sorted(expected_product), actual_product)


class TestQuantile:
    def test_unsupported_half_float_dtype(self):
        # Related Jira issue: https://jira/browse/RIP-341
        arr = np.zeros(shape=(2,), dtype=np.float16)
        with pytest.raises(SystemError):
            rt.quantile(arr, 0, 'lower')

    def test_one_darray(self):
        arr = np.zeros(shape=(2,), dtype=np.float32)
        result, _, _ = rt.quantile(arr, 0, 'lower')
        assert result == 0.0

    @pytest.mark.xfail(reason="RIP-341: supports shapes (x,), but not (x,0)")
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    def test_one_darray_xfail(self):
        arr = np.zeros(shape=(2, 0), dtype=np.float32)
        result, _, _ = rt.quantile(arr, 0, 'lower')
        assert result == 0.0


class TestSearchSorted:
    def _mask(self, arr):
        return arr % 2 == 0

    def test_simple(self):
        sides = {'left', 'right'}

        arr = np.arange(10)
        v = arr[self._mask(arr)]

        for side in sides:
            rt_indicies = rt.searchsorted(arr, v, side)
            np_indicies = np.searchsorted(arr, v, side)
            assert_array_equal(
                rt_indicies,
                np_indicies,
                err_msg=f"array {arr} values to insert {v} side {side}",
            )

    def test_simple_left(self):
        side = 'left'

        arr = np.array([0, 0, 0, 1])
        v = arr[self._mask(arr)]

        rt_indicies = rt.searchsorted(arr, v, side)
        np_indicies = np.searchsorted(arr, v, side)
        assert_array_equal(
            rt_indicies,
            np_indicies,
            err_msg=f"using array {arr}\nvalues to insert {v}\nside {side}",
        )

    @pytest.mark.xfail(
        reason="RIP-345 - expected insertion points at position 3, got 2"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    def test_simple_right_xfail(self):
        side = 'right'

        arr = np.array([0, 0, 0, 1])
        v = arr[self._mask(arr)]

        rt_indicies = rt.searchsorted(arr, v, side)
        np_indicies = np.searchsorted(arr, v, side)
        assert_array_equal(
            rt_indicies,
            np_indicies,
            err_msg=f"using array {arr}\nvalues to insert {v}\nside {side}",
        )

    def test_simple_right2(self):
        side = 'right'

        arr = np.array([0, 2, 2, 2, 2])
        v = arr[self._mask(arr)]

        rt_indicies = rt.searchsorted(arr, v, side)
        np_indicies = np.searchsorted(arr, v, side)
        assert_array_equal(
            rt_indicies,
            np_indicies,
            err_msg=f"using array {arr}\nvalues to insert {v}\nside {side}",
        )

    @pytest.mark.xfail(reason="RIP-345: expected insertion points at position 1, got 2")
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    def test_simple_left2_xfail(self):
        side = 'left'

        arr = np.array([0, 2, 2, 2, 2])
        v = arr[self._mask(arr)]

        rt_indicies = rt.searchsorted(arr, v, side)
        np_indicies = np.searchsorted(arr, v, side)
        assert_array_equal(
            rt_indicies,
            np_indicies,
            err_msg=f"using array {arr}\nvalues to insert {v}\nside {side}",
        )

    def test_searchsorted(self):
        a = rt.arange(10.0)
        b = rt.arange(20.0) / 2
        b[3] = -np.inf
        b[7] = np.inf
        b[5] = np.nan
        b[2] = 100
        b[1] = -100
        x1 = np.searchsorted(a, b, side='left')
        x2 = rt.searchsorted(a, b, side='left')
        assert sum(x1 - x2) == 0

        x1 = rt.searchsorted(a, b, side='right')
        x2 = rt.searchsorted(a, b, side='right')
        assert sum(x1 - x2) == 0

        b = b.astype(np.int32)


class TestStd:
    @pytest.mark.xfail(
        reason="RIP-350 - std not returning the correct result for some inputs"
    )
    @pytest.mark.parametrize(
        "a",
        # ACTUAL:  0.5345224838248488
        # DESIRED: 0.0
        [
            [
                1801439850948199,
                1801439850948199,
                1801439850948199,
                1801439850948199,
                1801439850948199,
                1801439850948199,
                1801439850948199,
                1801439850948199,
            ]
        ],
    )
    def test_std(self, a):
        arr = np.array(a, dtype=np.int64)
        np_std = np.std(arr, ddof=1)
        rt_std = rt.std(rt.FastArray(arr))
        assert_almost_equal(rt_std, np_std, decimal=6)


class TestSum:
    # add tests around promotion to larger type
    # e.g. sum of int32 type that overflows will be a result of int64 type
    # this behavior differs from numpy.sum
    @pytest.mark.xfail(reason="RIP-351 - sum not working as expected for some inputs")
    @pytest.mark.parametrize(
        "a",
        # ACTUAL:  3.193344496000001e+292
        # DESIRED: 3.193344496e+292
        [
            [
                3.99168062e291,
                3.99168062e291,
                3.99168062e291,
                3.99168062e291,
                3.99168062e291,
                3.99168062e291,
                3.99168062e291,
                3.99168062e291,
            ]
        ],
    )
    def test_sum_float(self, a):
        arr = np.array(a, dtype=np.float64)
        np_sum = np.sum(arr)
        rt_sum = rt.sum(rt.FastArray(arr))
        assert_almost_equal(rt_sum, np_sum, decimal=3)


class TestWhere:
    @pytest.mark.xfail(
        reason="rt.where returns FastArray([None]), should be FastArray([])"
    )
    def test_where(self):
        arr = np.array([0], dtype=np.int64)
        min = arr.min()
        np_gt_min = np.where(arr > min)
        rt_gt_min = rt.where(arr > min)
        assert_array_equal(
            np_gt_min,
            rt_gt_min,
            err_msg=f"array elements greater than the minimum {min}",
        )
        # E       AssertionError:
        # E       Arrays are not equal
        # E       array elements greater than the minimum 0
        # E       (shapes (1, 0), (1,) mismatch)
        # E        x: array([], shape=(1, 0), dtype=int64)
        # E        y: FastArray([None], dtype=object)

    @pytest.mark.xfail(reason="expected dtype of the original type")
    def test_where_dtype_demoting(self):
        dtype = np.int64
        arr = np.array([0, 1], dtype=dtype)
        min, mean, max = arr.min(), arr.mean(), arr.max()
        np_gt_min = np.where(arr > min)
        rt_gt_min = rt.where(arr > min)
        assert_array_equal(
            np_gt_min,
            rt_gt_min,
            err_msg=f"array elements greater than the minimum {min}",
        )
        assert dtype == rt_gt_min[0].dtype, "expected dtype of the original type"
