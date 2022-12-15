import itertools
import operator
import os
from typing import Tuple, Union

import numpy
import numpy as np
import numba

import pytest
from numpy.testing import assert_array_compare, assert_array_equal

# Utilize pandas' test data
# from pandas.tests.reshape.merge.test_merge_asof import TestAsOfMerge as TestPandasAsOfMerge

import riptable as rt
from riptable.rt_merge_asof import merge_asof2
from riptable.testing.array_assert import assert_array_or_cat_equal

# TODO: Implement tests for merge_asof2 that check it's behavior when the "on" and/or "by" keycols from 'left' and 'right' have different dtypes.
#         * What's the type of the keycol(s) in the merged Dataset?

# TODO: Implement tests for merge_asof2 to check how it behaves when the "by" keycols from 'left' and/or 'right' have a Grouping
#       containing groups with 0 elements, such as with a filtered Categorical.


def check_merge_asof(ds, ds1, ds2):
    # print(ds)
    assert isinstance(ds, rt.Dataset)
    assert ds.shape[0] == ds1.shape[0]
    assert ds.shape[1] == ds1.shape[1] + ds2.shape[1]
    assert_array_equal(ds.A._np, ds1.A._np)
    assert_array_equal(ds.left_val._np, ds1.left_val._np)
    assert_array_equal(ds.right_val._np, ds.X._np)


class TestMergeAsof2:
    #    def __init__(self):
    #        # Import some pandas pytest fixtures to utilize their test data.
    #        self.trades = TestPandasAsOfMerge.trades
    #        self.quotes = TestPandasAsOfMerge.quotes
    #        self.asof = TestPandasAsOfMerge.asof
    #        self.tolerance = TestPandasAsOfMerge.tolerance
    #        self.allow_exact_matches_data = TestPandasAsOfMerge.allow_exact_matches
    #        self.allow_exact_matches_and_tolerance_data = TestPandasAsOfMerge.allow_exact_matches_and_tolerance

    @pytest.mark.parametrize(
        ("time_dtypes", "tolerance"),
        [
            pytest.param((np.int8, np.int16), None),
            pytest.param((np.int8, np.int16), 10),
            pytest.param((np.float64, np.float32), None),
            pytest.param((np.float64, np.float32), 10.0),
        ],
    )
    def test_merge_asof_backward_exact(self, time_dtypes, tolerance) -> None:
        # Unpack dtypes for the 'on' columns.
        ds1_time_dtype, ds2_time_dtype = time_dtypes

        ds1 = rt.Dataset()
        ds1["Time"] = rt.FA([-3, 0, 1, 4, 6, 8, 9, 11, 16, 19, 30], dtype=ds1_time_dtype)
        ds1["Px"] = rt.FA([8, 10, 12, 15, 11, 10, 9, 13, 7, 9, 10], dtype=np.int16)

        ds2 = rt.Dataset()
        ds2["Time"] = rt.FA([0, 0, 5, 7, 8, 8, 10, 10, 12, 15, 17, 20], dtype=ds2_time_dtype)
        ds2["Vols"] = rt.FA([0, 20, 22, 19, 3, 24, 5, 25, 26, 27, 28, 29])

        target = rt.Dataset()
        target.Time = ds1.Time
        target.Px = ds1.Px
        target.Vols = rt.FA([ds2["Vols"].inv, 20, 20, 20, 22, 24, 24, 25, 27, 28, 29], dtype=ds2["Vols"].dtype)
        # For seeing which time was matched against (from the right side) for each row
        target.Time_y = rt.FA([ds2["Time"].inv, 0, 0, 0, 5, 8, 8, 10, 15, 17, 20], dtype=ds2["Time"].dtype)

        merged = merge_asof2(ds1, ds2, on="Time", direction="backward", matched_on="Time_y", tolerance=tolerance)

        for key in merged.keys():
            assert_array_or_cat_equal(
                merged[key], target[key], err_msg=f"Column '{key}' differs between the actual and expected."
            )

    def test_merge_asof_backward_noexact(self) -> None:
        ds1 = rt.Dataset()
        ds1["Time"] = rt.FA([0, 1, 4, 6, 8, 9, 11, 16, 19, 30], dtype=np.int8)
        ds1["Px"] = rt.FA([10, 12, 15, 11, 10, 9, 13, 7, 9, 10], dtype=np.int16)

        ds2 = rt.Dataset()
        ds2["Time"] = rt.FA([0, 0, 5, 7, 8, 8, 10, 10, 12, 15, 17, 20], dtype=np.int16)
        ds2["Vols"] = rt.FA([0, 20, 22, 19, 3, 24, 5, 25, 26, 27, 28, 29])

        target = rt.Dataset()
        target.Time = ds1.Time
        target.Px = ds1.Px
        target.Vols = rt.FA([ds2["Vols"].inv, 20, 20, 22, 19, 24, 25, 27, 28, 29])
        # For seeing which time was matched against (from the right side) for each row
        target.Time_y = rt.FA([ds2["Time"].inv, 0, 0, 5, 7, 8, 10, 15, 17, 20], dtype=ds2["Time"].dtype)

        merged = merge_asof2(ds1, ds2, on="Time", direction="backward", allow_exact_matches=False, matched_on="Time_y")

        for key in merged.keys():
            assert_array_or_cat_equal(
                merged[key], target[key], err_msg=f"Column '{key}' differs between the actual and expected."
            )

    def test_merge_asof_backward_grouped_exact(self):
        ds1 = rt.Dataset({"A": [1, 5, 10], "left_val": ["a", "b", "c"], "left_grp": [1, 1, 1]})
        ds2 = rt.Dataset(
            {
                "X": [1, 2, 3, 6, 7],
                "right_val": [1, 2, 3, 6, 7],
                "right_grp": [1, 1, 1, 1, 1],
            }
        )

        ds = merge_asof2(
            ds1,
            ds2,
            left_on="A",
            right_on="X",
            left_by="left_grp",
            right_by="right_grp",
            direction="backward",
        )
        check_merge_asof(ds, ds1, ds2)
        assert_array_equal(ds.left_grp._np, ds.right_grp._np)
        assert_array_compare(operator.__ge__, ds.A, ds.X)

    def test_merge_asof_backward_grouped_noexact(self) -> None:
        ds1 = rt.Dataset({"A": [1, 5, 10], "left_val": ["a", "b", "c"], "left_grp": [1, 1, 1]})
        ds2 = rt.Dataset(
            {
                "X": [1, 2, 3, 6, 7],
                "right_val": [1, 2, 3, 6, 7],
                "right_grp": [1, 1, 1, 1, 1],
            }
        )

        ds = merge_asof2(
            ds1,
            ds2,
            left_on="A",
            right_on="X",
            left_by="left_grp",
            right_by="right_grp",
            allow_exact_matches=False,
        )
        check_merge_asof(ds, ds1, ds2)
        assert_array_equal(ds.left_grp._np[1:], ds.right_grp._np[1:])
        assert_array_compare(operator.__gt__, ds.A[1:], ds.X[1:])

    @pytest.mark.parametrize("time_multiplier", [1, 2])
    def test_merge_asof_forward_exact(self, time_multiplier: int) -> None:
        ds1 = rt.Dataset()
        ds1["Time"] = rt.FA([-3, 0, 1, 4, 6, 8, 9, 11, 16, 19, 30], dtype=np.int8)
        ds1["Px"] = rt.FA([8, 10, 12, 15, 11, 10, 9, 13, 7, 9, 10], dtype=np.int16)

        ds2 = rt.Dataset()
        end_time = 20 * time_multiplier
        ds2["Time"] = rt.FA([0, 0, 5, 7, 8, 8, 10, 10, 12, 15, 17, end_time, end_time, end_time], dtype=np.int16)
        ds2["Vols"] = rt.FA([0, 20, 22, 19, 3, 24, 5, 25, 26, 27, 28, 29, 21, 33])

        target = rt.Dataset()
        target.Time = ds1.Time
        target.Px = ds1.Px
        target.Vols = rt.FA(
            [0, 0, 22, 22, 19, 3, 5, 26, 28, 29, (ds2["Vols"].inv if time_multiplier == 1 else 29)],
            dtype=ds2["Vols"].dtype,
        )
        # For seeing which time was matched against (from the right side) for each row
        target.Time_y = rt.FA(
            [0, 0, 5, 5, 7, 8, 10, 12, 17, end_time, (ds2["Time"].inv if time_multiplier == 1 else end_time)],
            dtype=ds2["Time"].dtype,
        )

        merged = merge_asof2(ds1, ds2, on="Time", direction="forward", matched_on="Time_y")

        for key in merged.keys():
            assert_array_or_cat_equal(
                merged[key], target[key], err_msg=f"Column '{key}' differs between the actual and expected."
            )

    def test_merge_asof_forward_noexact(self) -> None:
        ds1 = rt.Dataset()
        ds1["Time"] = rt.FA([0, 1, 4, 6, 8, 9, 11, 16, 19, 30], dtype=np.int8)
        ds1["Px"] = rt.FA([10, 12, 15, 11, 10, 9, 13, 7, 9, 10], dtype=np.int16)

        ds2 = rt.Dataset()
        ds2["Time"] = rt.FA([0, 0, 5, 7, 8, 8, 10, 10, 12, 15, 17, 20], dtype=np.int16)
        ds2["Vols"] = rt.FA([0, 20, 22, 19, 3, 24, 5, 25, 26, 27, 28, 29])

        target = rt.Dataset()
        target.Time = ds1.Time
        target.Px = ds1.Px
        target.Vols = rt.FA([22, 22, 22, 19, 5, 5, 26, 28, 29, ds2["Vols"].inv])
        # For seeing which time was matched against (from the right side) for each row
        target.Time_y = rt.FA([5, 5, 5, 7, 10, 10, 12, 17, 20, ds2["Time"].inv], dtype=ds2["Time"].dtype)

        merged = merge_asof2(ds1, ds2, on="Time", direction="forward", allow_exact_matches=False, matched_on="Time_y")

        for key in merged.keys():
            assert_array_or_cat_equal(
                merged[key], target[key], err_msg=f"Column '{key}' differs between the actual and expected."
            )

    def test_merge_asof_forward_grouped_exact(self) -> None:
        ds1 = rt.Dataset({"A": [1, 5, 10], "left_val": ["a", "b", "c"], "left_grp": [1, 1, 1]})
        ds2 = rt.Dataset(
            {
                "X": [1, 2, 3, 6, 7],
                "right_val": [1, 2, 3, 6, 7],
                "right_grp": [1, 1, 1, 1, 1],
            }
        )

        ds = merge_asof2(
            ds1,
            ds2,
            left_on="A",
            right_on="X",
            left_by="left_grp",
            right_by="right_grp",
            direction="forward",
        )
        check_merge_asof(ds, ds1, ds2)
        assert_array_equal(ds.left_grp._np[:-1], ds.right_grp._np[:-1])
        assert_array_compare(operator.__le__, ds.A[:-1], ds.X[:-1])

    def test_merge_asof_forward_grouped_noexact(self) -> None:
        ds1 = rt.Dataset({"A": [1, 5, 10], "left_val": ["a", "b", "c"], "left_grp": [1, 1, 1]})
        ds2 = rt.Dataset(
            {
                "X": [1, 1, 2, 3, 6, 7],
                "right_val": [1, 0, 2, 3, 6, 7],
                "right_grp": [1, 1, 1, 1, 1, 1],
            }
        )

        ds = merge_asof2(
            ds1,
            ds2,
            left_on="A",
            right_on="X",
            left_by="left_grp",
            right_by="right_grp",
            allow_exact_matches=False,
            direction="forward",
        )
        check_merge_asof(ds, ds1, ds2)
        assert_array_equal(ds.left_grp._np[:-1], ds.right_grp._np[:-1])
        assert_array_compare(operator.__lt__, ds.A[:-1], ds.X[:-1])

    def test_merge_asof_nearest_exact(self):
        ds1 = rt.Dataset()
        ds1["Time"] = rt.FA([0, 1, 3, 7, 8, 9, 11, 16, 19, 30])
        ds1["Px"] = rt.FA([10, 12, 15, 11, 10, 9, 13, 7, 9, 10])

        ds2 = rt.Dataset()
        ds2["Time"] = rt.FA([0, 0, 4, 5, 8, 8, 10, 10, 12, 15, 17, 20])
        ds2["Vols"] = rt.FA([0, 20, 22, 19, 23, 18, 24, 25, 26, 27, 28, 29])
        ds2["idx_right"] = rt.arange(ds2.get_nrows())

        target = rt.Dataset()
        target.Time = ds1.Time
        target.Px = ds1.Px
        # For seeing which time was matched against (from the right side) for each row
        target.idx_right = rt.FA([1, 1, 2, 4, 5, 5, 7, 9, 11, 11])
        target.Time_y = ds2["Time"][target.idx_right]
        target.Vols = ds2["Vols"][target.idx_right]

        merged = merge_asof2(ds1, ds2, on="Time", direction="nearest", matched_on="Time_y")

        for key in merged.keys():
            assert_array_equal(
                merged[key], target[key], err_msg=f"Column '{key}' differs between the actual and expected."
            )

        # ds = merge_asof2(ds1, ds2, left_on='A', right_on='X', left_by='left_grp',
        #                            right_by='right_grp', direction='nearest')
        # check_merge_asof(ds, ds1, ds2)
        # for i in range(0, ds.shape[0]):
        #    diff = abs(ds.A[i] - ds.X[i])
        #    mindiff = min(ds.A - ds.X)
        #    self.assertEqual(diff, mindiff, diff + ", " + mindiff);
        # self.assertTrue((ds.left_grp._np == ds.right_grp._np).all())

    def test_merge_asof_nearest_noexact(self):
        ds1 = rt.Dataset()
        ds1["Time"] = rt.FA([0, 1, 3, 7, 8, 9, 11, 16, 19, 30])
        ds1["Px"] = rt.FA([10, 12, 15, 11, 10, 9, 13, 7, 9, 10])

        ds2 = rt.Dataset()
        ds2["Time"] = rt.FA([0, 0, 4, 5, 8, 8, 10, 10, 12, 15, 17, 20])
        ds2["Vols"] = rt.FA([0, 20, 22, 19, 23, 18, 24, 25, 26, 27, 28, 29])
        ds2["idx_right"] = rt.arange(ds2.get_nrows())

        target = rt.Dataset()
        target.Time = ds1.Time
        target.Px = ds1.Px
        # For seeing which time was matched against (from the right side) for each row
        target.idx_right = rt.FA([2, 1, 2, 4, 6, 5, 7, 9, 11, 11])
        target.Time_y = ds2["Time"][target.idx_right]
        target.Vols = ds2["Vols"][target.idx_right]

        merged = merge_asof2(ds1, ds2, on="Time", allow_exact_matches=False, direction="nearest", matched_on="Time_y")

        for key in merged.keys():
            assert_array_equal(
                merged[key], target[key], err_msg=f"Column '{key}' differs between the actual and expected."
            )

    def test_merge_asof_nearest_grouped_exact(self) -> None:
        ds1 = rt.Dataset({"A": [1, 5, 10], "left_val": ["a", "b", "c"], "left_grp": [1, 1, 1]})
        ds2 = rt.Dataset(
            {
                "X": [1, 2, 3, 6, 7],
                "right_val": [1, 2, 3, 6, 7],
                "right_grp": [1, 1, 1, 1, 1],
            }
        )

        merged = merge_asof2(
            ds1,
            ds2,
            on=("A", "X"),
            by=("left_grp", "right_grp"),
            direction="nearest",
        )
        check_merge_asof(merged, ds1, ds2)

        target = ds1.copy(deep=True)
        target.X = rt.FA([1, 6, 7])
        target.right_val = rt.FA([1, 6, 7])
        target.right_grp = rt.FA([1, 1, 1])

        for key in merged.keys():
            assert_array_equal(
                merged[key], target[key], err_msg=f"Column '{key}' differs between the actual and expected."
            )

    def test_merge_asof_nearest_grouped_noexact(self) -> None:
        ds1 = rt.Dataset({"A": [1, 5, 10], "left_val": ["a", "b", "c"], "left_grp": [1, 1, 1]})
        ds2 = rt.Dataset(
            {
                "X": [1, 1, 2, 3, 6, 7],
                "right_val": [1, 0, 2, 3, 6, 7],
                "right_grp": [1, 1, 1, 1, 1, 1],
            }
        )

        merged = merge_asof2(
            ds1,
            ds2,
            on=("A", "X"),
            by=("left_grp", "right_grp"),
            allow_exact_matches=False,
            direction="nearest",
        )
        check_merge_asof(merged, ds1, ds2)

        target = ds1.copy(deep=True)
        target.X = rt.FA([2, 6, 7])
        target.right_val = rt.FA([2, 6, 7])
        target.right_grp = rt.FA([1, 1, 1])

        for key in merged.keys():
            assert_array_equal(
                merged[key], target[key], err_msg=f"Column '{key}' differs between the actual and expected."
            )

    def test_merge_asof_categorical_and_string_keys(self):
        ds1 = rt.Dataset()
        ds1["Time"] = [0, 1, 4, 6, 8, 9, 11, 16, 19, 30]
        ds1["Px"] = [10, 12, 15, 11, 10, 9, 13, 7, 9, 10]

        ds2 = rt.Dataset()
        ds2["Time"] = [0, 0, 5, 7, 8, 10, 12, 15, 17, 20]
        ds2["Vols"] = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

        target = rt.Dataset()
        target.Time = ds1.Time
        target.Px = ds1.Px
        target.Vols = [20, 20, 20, 22, 24, 24, 24, 26, 28, 28]
        target.matched_on = [
            0,
            0,
            0,
            5,
            8,
            8,
            8,
            12,
            17,
            17,
        ]  # For seeing which time was matched against (from the right side) for each row

        # Categorical keys
        ds1["Ticker"] = rt.Categorical(["Test"] * 10)
        ds2["Ticker"] = rt.Categorical(["Test", "Blah"] * 5)
        target.Ticker = ds1.Ticker

        ds = merge_asof2(ds1, ds2, on="Time", by="Ticker", matched_on=True)

        for key in ds.keys():
            # TODO: Switch to assert_array_equal here once a string-based Categorical allows np.inf to be used in an equality check
            # assert_array_or_cat_equal(ds[key], target[key], err_msg=f"Column '{key}' differs between the actual and expected.")
            assert (ds[key] == target[key]).all()

        # String keys
        ds1["Ticker"] = rt.FastArray(["Test"] * 10)
        ds2["Ticker"] = rt.FastArray(["Test", "Blah"] * 5)
        target.Ticker = ds1.Ticker

        ds = merge_asof2(ds1, ds2, on="Time", by="Ticker")

        for key in ds.keys():
            assert_array_equal(ds[key], target[key], err_msg=f"Column '{key}' differs between the actual and expected.")

    @pytest.mark.parametrize(
        "left,right,on,should_fail",
        [
            pytest.param(
                rt.Dataset({"A": [1, 5, 10], "left_val": ["a", "b", "c"], "left_grp": [1, 1, 1]}),
                rt.Dataset(
                    {
                        "X": [1, 2, 3, 6, 7],
                        "right_val": [1, 2, 3, 6, 7],
                        "right_grp": [1, 1, 1, 1, 1],
                    }
                ),
                ("A", "X"),
                False,
                id="sorted,sorted",
            ),
            pytest.param(
                rt.Dataset({"A": [1, 10, 5], "left_val": ["a", "b", "c"], "left_grp": [1, 1, 1]}),
                rt.Dataset(
                    {
                        "X": [1, 2, 3, 6, 7],
                        "right_val": [1, 2, 3, 6, 7],
                        "right_grp": [1, 1, 1, 1, 1],
                    }
                ),
                ("A", "X"),
                True,
                id="unsorted,sorted",
            ),
            pytest.param(
                rt.Dataset({"A": [1, 5, 10], "left_val": ["a", "b", "c"], "left_grp": [1, 1, 1]}),
                rt.Dataset(
                    {
                        "X": [1, 2, 3, 6, 4],
                        "right_val": [1, 2, 3, 6, 7],
                        "right_grp": [1, 1, 1, 1, 1],
                    }
                ),
                ("A", "X"),
                True,
                id="sorted,unsorted",
            ),
        ],
    )
    def test_merge_asof_unsorted_fails(
        self,
        left: rt.Dataset,
        right: rt.Dataset,
        on: Union[str, Tuple[str, str]],
        should_fail: bool,
    ):
        """
        Test for verifying 'merge_asof' checks the sortedness of the 'on' columns before merging.
        """
        # Unpack the 'on' parameter (if needed) until 'merge_asof' has been
        # modified to allow a tuple to be passed for the 'on' parameter (instead of left_on+right_on).
        left_on, right_on = on if isinstance(on, tuple) else (on, on)

        if should_fail:
            with pytest.raises(ValueError):
                merge_asof2(
                    left,
                    right,
                    left_on=left_on,
                    right_on=right_on,
                )
        else:
            # Just check that this succeeds without raising any errors;
            # we otherwise don't care about the result of the operation.
            merge_asof2(
                left,
                right,
                left_on=left_on,
                right_on=right_on,
            )

    def test_merge_asof_backward_multiple_by(self) -> None:
        # Adapted from https://github.com/pandas-dev/pandas/blob/32b4222050c7d5eedfa8fa47d5a45fb1d6c966a5/pandas/tests/reshape/merge/test_merge_asof.py#L286
        trades = rt.Dataset(
            {
                "time": rt.DateTimeNano(
                    [
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.046",
                        "20160525 13:30:00.048",
                        "20160525 13:30:00.050",
                    ],
                    from_tz="America/New_York",
                ),
                "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
                "exch": ["ARCA", "NSDQ", "NSDQ", "BATS", "NSDQ"],
                "price": [51.95, 51.95, 720.77, 720.92, 98.00],
                "quantity": [75, 155, 100, 100, 100],
            }
        )

        quotes = rt.Dataset(
            {
                "time": rt.DateTimeNano(
                    [
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.030",
                        "20160525 13:30:00.041",
                        "20160525 13:30:00.045",
                        "20160525 13:30:00.049",
                    ],
                    from_tz="America/New_York",
                ),
                "ticker": ["GOOG", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL"],
                "exch": ["BATS", "NSDQ", "ARCA", "ARCA", "NSDQ", "ARCA"],
                "bid": [720.51, 51.95, 51.97, 51.99, 720.50, 97.99],
                "ask": [720.92, 51.96, 51.98, 52.00, 720.93, 98.01],
            }
        )

        expected = rt.Dataset(
            {
                "time": rt.DateTimeNano(
                    [
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.046",
                        "20160525 13:30:00.048",
                        "20160525 13:30:00.050",
                    ],
                    from_tz="America/New_York",
                ),
                "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
                "exch": ["ARCA", "NSDQ", "NSDQ", "BATS", "NSDQ"],
                "price": [51.95, 51.95, 720.77, 720.92, 98.00],
                "quantity": [75, 155, 100, 100, 100],
                "bid": [np.nan, 51.95, 720.50, 720.51, np.nan],
                "ask": [np.nan, 51.96, 720.93, 720.92, np.nan],
            }
        )

        merged = merge_asof2(trades, quotes, on="time", by=["ticker", "exch"])

        for key in merged.keys():
            assert_array_or_cat_equal(
                merged[key], expected[key], err_msg=f"Column '{key}' differs between the actual and expected."
            )

    @pytest.mark.parametrize(
        (
            "time_dtypes",
            "tolerance",
        ),
        [
            pytest.param(
                (np.uint8, np.uint16),
                rt.uint16.inv,
                id="Unsigned integer invalid",
                marks=[pytest.mark.skip("Unclear what the expected behavior should be in this case.")],
            ),
            pytest.param((np.uint8, np.int16), -10, id="Negative integer"),
            pytest.param((np.float64, np.float32), np.nan, id="NaN floating point"),
            pytest.param((np.float64, np.float32), -10.0, id="Negative floating point"),
        ],
    )
    def test_merge_asof_backward_bad_tolerance(self, time_dtypes, tolerance) -> None:
        """Test that merge_asof2 rejects invalid/unusable 'tolerance' values."""
        # Unpack dtypes for the 'on' columns.
        ds1_time_dtype, ds2_time_dtype = time_dtypes

        ds1 = rt.Dataset()
        ds1["Time"] = rt.FA([0, 1, 4, 6, 8, 9, 11, 16, 19, 30], dtype=ds1_time_dtype)
        ds1["Px"] = rt.FA([10, 12, 15, 11, 10, 9, 13, 7, 9, 10], dtype=np.int16)

        ds2 = rt.Dataset()
        ds2["Time"] = rt.FA([0, 0, 5, 7, 8, 8, 10, 10, 12, 15, 17, 20], dtype=ds2_time_dtype)
        ds2["Vols"] = rt.FA([0, 20, 22, 19, 3, 24, 5, 25, 26, 27, 28, 29])

        target = rt.Dataset()
        target.Time = ds1.Time
        target.Px = ds1.Px
        target.Vols = rt.FA([20, 20, 20, 22, 24, 24, 25, 27, 28, 29])
        # For seeing which time was matched against (from the right side) for each row
        target.Time_y = rt.FA([0, 0, 0, 5, 8, 8, 10, 15, 17, 20], dtype=ds2["Time"].dtype)

        with pytest.raises(ValueError):
            _ = merge_asof2(ds1, ds2, on="Time", direction="backward", matched_on="Time_y", tolerance=tolerance)
