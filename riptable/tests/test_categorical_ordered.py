from riptable import *
from riptable.rt_enum import GROUPBY_KEY_PREFIX

str_list = ['b', 'b', 'a', 'c', 'b']
str_sorted = ['a', 'b', 'c']
str_unsorted = ['b', 'a', 'c']

int_list = [20, 20, 10, 30, 20]
int_sorted = [10, 20, 30]
int_unsorted = [20, 10, 30]

flt_list = [20.0, 20.0, 10.0, 30.0, 20.0]
flt_sorted = [10.0, 20.0, 30.0]
flt_unsorted = [20.0, 10.0, 30.0]

data = arange(5)
datasum_sorted = [2, 5, 3]
datasum_unsorted = [5, 2, 3]


def arr_equal(a, b):
    return bool(np.all(a == b))


class TestCategoricalOrdered:
    def test_single_values(self):

        # -------------SINGLE STRINGS----------------------------
        c = Categorical(str_list)
        ds = c.sum(data)
        assert arr_equal(c.category_array, str_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(str_list, ordered=True)
        ds = c.sum(data)
        assert arr_equal(c.category_array, str_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(str_list, ordered=False)
        ds = c.sum(data)
        assert arr_equal(c.category_array, str_unsorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_unsorted)
        assert arr_equal(ds.col_0, datasum_unsorted)

        c = Categorical(str_list, sort_gb=True)
        ds = c.sum(data)
        assert arr_equal(c.category_array, str_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(str_list, ordered=True, sort_gb=True)
        ds = c.sum(data)
        assert arr_equal(c.category_array, str_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(str_list, ordered=False, sort_gb=True)
        ds = c.sum(data)
        assert arr_equal(c.category_array, str_unsorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(str_list, sort_gb=False)
        ds = c.sum(data)
        assert arr_equal(c.category_array, str_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(str_list, ordered=True, sort_gb=False)
        ds = c.sum(data)
        assert arr_equal(c.category_array, str_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(str_list, ordered=False, sort_gb=False)
        ds = c.sum(data)
        assert arr_equal(c.category_array, str_unsorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_unsorted)
        assert arr_equal(ds.col_0, datasum_unsorted)

        # -------------SINGLE INTEGERS----------------------------
        c = Categorical(int_list)
        ds = c.sum(data)
        assert arr_equal(c.category_array, int_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], int_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(int_list, ordered=True)
        ds = c.sum(data)
        assert arr_equal(c.category_array, int_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], int_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(int_list, ordered=False)
        ds = c.sum(data)
        assert arr_equal(c.category_array, int_unsorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], int_unsorted)
        assert arr_equal(ds.col_0, datasum_unsorted)

        c = Categorical(int_list, sort_gb=True)
        ds = c.sum(data)
        assert arr_equal(c.category_array, int_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], int_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(int_list, ordered=True, sort_gb=True)
        ds = c.sum(data)
        assert arr_equal(c.category_array, int_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], int_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(int_list, ordered=False, sort_gb=True)
        ds = c.sum(data)
        assert arr_equal(c.category_array, int_unsorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], int_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(int_list, sort_gb=False)
        ds = c.sum(data)
        assert arr_equal(c.category_array, int_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], int_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(int_list, ordered=True, sort_gb=False)
        ds = c.sum(data)
        assert arr_equal(c.category_array, int_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], int_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(int_list, ordered=False, sort_gb=False)
        ds = c.sum(data)
        assert arr_equal(c.category_array, int_unsorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], int_unsorted)
        assert arr_equal(ds.col_0, datasum_unsorted)

        # -------------SINGLE FLOATS----------------------------
        c = Categorical(flt_list)
        ds = c.sum(data)
        assert arr_equal(c.category_array, flt_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], flt_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(flt_list, ordered=True)
        ds = c.sum(data)
        assert arr_equal(c.category_array, flt_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], flt_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(flt_list, ordered=False)
        ds = c.sum(data)
        assert arr_equal(c.category_array, flt_unsorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], flt_unsorted)
        assert arr_equal(ds.col_0, datasum_unsorted)

        c = Categorical(flt_list, sort_gb=True)
        ds = c.sum(data)
        assert arr_equal(c.category_array, flt_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], flt_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(flt_list, ordered=True, sort_gb=True)
        ds = c.sum(data)
        assert arr_equal(c.category_array, flt_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], flt_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(flt_list, ordered=False, sort_gb=True)
        ds = c.sum(data)
        assert arr_equal(c.category_array, flt_unsorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], flt_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(flt_list, sort_gb=False)
        ds = c.sum(data)
        assert arr_equal(c.category_array, flt_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], flt_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(flt_list, ordered=True, sort_gb=False)
        ds = c.sum(data)
        assert arr_equal(c.category_array, flt_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], flt_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical(flt_list, ordered=False, sort_gb=False)
        ds = c.sum(data)
        assert arr_equal(c.category_array, flt_unsorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], flt_unsorted)
        assert arr_equal(ds.col_0, datasum_unsorted)

    def test_multikey(self):

        c = Categorical([FA(str_list), FA(int_list)])
        ds = c.sum(data)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_unsorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_1'], int_unsorted)
        assert arr_equal(ds.col_0, datasum_unsorted)

        # 5/9/2019 - multikey will now hold uniques in sorted order if requested, behaves like single key
        # unlike single key, still defaults to holding unsorted (searchsorted doesn't apply to keys after the first one)
        c = Categorical([FA(str_list), FA(int_list)], ordered=True)
        ds = c.sum(data)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_1'], int_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical([FA(str_list), FA(int_list)], ordered=False)
        ds = c.sum(data)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_unsorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_1'], int_unsorted)
        assert arr_equal(ds.col_0, datasum_unsorted)

        c = Categorical([FA(str_list), FA(int_list)], sort_gb=True)
        ds = c.sum(data)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_1'], int_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical([FA(str_list), FA(int_list)], ordered=True, sort_gb=True)
        ds = c.sum(data)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_1'], int_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical([FA(str_list), FA(int_list)], ordered=False, sort_gb=True)
        ds = c.sum(data)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_1'], int_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical([FA(str_list), FA(int_list)], sort_gb=False)
        ds = c.sum(data)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_unsorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_1'], int_unsorted)
        assert arr_equal(ds.col_0, datasum_unsorted)

        c = Categorical([FA(str_list), FA(int_list)], ordered=True, sort_gb=False)
        ds = c.sum(data)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_sorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_1'], int_sorted)
        assert arr_equal(ds.col_0, datasum_sorted)

        c = Categorical([FA(str_list), FA(int_list)], ordered=False, sort_gb=False)
        ds = c.sum(data)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_0'], str_unsorted)
        assert arr_equal(ds[GROUPBY_KEY_PREFIX + '_1'], int_unsorted)
        assert arr_equal(ds.col_0, datasum_unsorted)

    def test_values_cats(self):

        c = Categorical(str_list, str_unsorted)
        assert arr_equal(c.category_array, str_unsorted)
        c = Categorical(str_list, str_unsorted, ordered=True)
        assert arr_equal(c.category_array, str_unsorted)
        c = Categorical(str_list, str_unsorted, ordered=False)
        assert arr_equal(c.category_array, str_unsorted)

        c = Categorical(flt_list, flt_unsorted)
        assert arr_equal(c.category_array, flt_unsorted)
        c = Categorical(flt_list, flt_unsorted, ordered=True)
        assert arr_equal(c.category_array, flt_unsorted)
        c = Categorical(flt_list, flt_unsorted, ordered=False)
        assert arr_equal(c.category_array, flt_unsorted)
