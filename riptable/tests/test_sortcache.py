import unittest
import riptable as rt
from riptable.Utils.rt_testing import assert_array_equal
from riptable.rt_sort_cache import SortCache


class TestSortCache(unittest.TestCase):
    def test_get_sorted_row_index(self):
        ds1 = rt.Dataset({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        des = list(range(len(ds1) - 1, -1, -1))
        asc = list(range(len(ds1)))

        ds1.sort_view(["a", "b"], [False, True])
        sort_idx = SortCache.get_sorted_row_index(*ds1.get_row_sort_info())
        assert_array_equal(sort_idx, des)
        ds1.sort_view(["a", "b"], [True, True])
        sort_idx = SortCache.get_sorted_row_index(*ds1.get_row_sort_info())
        assert_array_equal(sort_idx, asc)

        # in cache path
        sort_idx = SortCache.get_sorted_row_index(*ds1.get_row_sort_info())
        assert_array_equal(sort_idx, asc)

        ds1.sort_view(["a", "b"], False)
        sort_idx = SortCache.get_sorted_row_index(*ds1.get_row_sort_info())
        assert_array_equal(sort_idx, des)

        ds1.sort_view(["a", "b"], False)
        sort_idx = SortCache.get_sorted_row_index(*ds1.get_row_sort_info())
        assert_array_equal(sort_idx, des)
