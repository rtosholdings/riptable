import pytest
import numpy as np
from riptable import *
from riptable.rt_enum import GROUPBY_KEY_PREFIX

all_int_dtypes = [np.int8, np.int16, np.int32, np.int64]


def arr_equal(a, b):
    return bool(np.all(a == b))


class TestCategoricalValues:
    '''
    Test Categorical constructor will list / array of values only.

    Use every combination of flags:
        -first_order
        -sort
        -filter
        -dtype
        -base_index

    unicode, invalid_category will also be tested, but only applicable to certain categoricals

    Test _name.
    Test all properties.
    Test all indexing.
    Test all comparisons
    Test all other funcs -- hstack, merge, etc.
    Test all groupby operations.
    Test save / load.



    ADD WARNINGS:
    - setting a dtype if the categorical is initialized from values (should this be ignored?)
    - setting keyword that doesn't apply to constructor
    
    BUGS:
    - single array in a list should hit the same path as single array


    KEYWORD CHANGES:
    - remove 'first_order', replace with 'ordered'. 'ordered' means keep the order of occurrence OR do not sort the uniques provided
    - presort=True means large amount of uniques, sort and walk
    - presort=True + ordered = False is a conflict --- raise an error
    - sort_gb applies later, stays the same
    


    '''

    def test_strings(self):
        bytes_arr = FA(['d', 'b', 'c', 'a', 'a', 'b', 'd'])

        cats_first_order = FA(['d', 'b', 'c', 'a'])
        cats_first_order_filtered = FA(['d', 'c', 'a'])
        cats_sorted = FA(['a', 'b', 'c', 'd'])
        cats_sorted_filtered = FA(['a', 'c', 'd'])

        base_1_underlying_sorted = FA([4, 2, 3, 1, 1, 2, 4])
        base_0_underlying_sorted = base_1_underlying_sorted - 1
        base_1_underlying_first_order = FA([1, 2, 3, 4, 4, 2, 1])
        base_0_underlying_first_order = base_1_underlying_first_order - 1

        underlying_sorted_filtered = FA([3, 0, 2, 1, 1, 0, 3])
        underlying_first_order_filtered = [1, 0, 2, 3, 3, 0, 1]

        b_filter = FA([True, False, True, True, True, False, True])

        gb_data = arange(7)
        gb_nums_sorted = FA([7, 6, 2, 6])
        gb_nums_sorted_filtered = FA([7, 2, 6])
        gb_nums_first_order = FA([6, 6, 2, 7])
        gb_nums_first_order_filtered = FA([6, 2, 7])

        # 0, 0, None, None, 0
        c = Categorical(
            bytes_arr,
            ordered=True,
            sort_gb=False,
            dtype=None,
            filter=None,
            base_index=0,
        )
        assert arr_equal(base_0_underlying_sorted, c._fa)
        assert arr_equal(cats_sorted, c.category_array)
        gb_result = c.sum(gb_data)
        assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted)
        assert arr_equal(gb_result.col_0, gb_nums_sorted)

        # 0, 0, None, None, 1
        c = Categorical(
            bytes_arr,
            ordered=True,
            sort_gb=False,
            dtype=None,
            filter=None,
            base_index=1,
        )
        assert arr_equal(base_1_underlying_sorted, c._fa)
        assert arr_equal(cats_sorted, c.category_array)
        gb_result = c.sum(gb_data)
        assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted)
        assert arr_equal(gb_result.col_0, gb_nums_sorted)

        # 0, 0, None, FILTER, 0
        with pytest.raises(ValueError):
            c = Categorical(
                bytes_arr,
                ordered=True,
                sort_gb=False,
                dtype=None,
                filter=b_filter,
                base_index=0,
            )

        # 0, 0, None, FILTER, 1
        c = Categorical(
            bytes_arr,
            ordered=True,
            sort_gb=False,
            dtype=None,
            filter=b_filter,
            base_index=1,
        )
        assert arr_equal(underlying_sorted_filtered, c._fa)
        assert arr_equal(cats_sorted_filtered, c.category_array)
        gb_result = c.sum(gb_data)
        assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted_filtered)
        assert arr_equal(gb_result.col_0, gb_nums_sorted_filtered)

        # 0, 0, ALL_DTYPES, None, 0
        for dt in all_int_dtypes:
            # add assert warns
            c = Categorical(
                bytes_arr,
                ordered=True,
                sort_gb=False,
                dtype=dt,
                filter=None,
                base_index=0,
            )
            assert arr_equal(base_0_underlying_sorted, c._fa)
            assert arr_equal(cats_sorted, c.category_array)
            gb_result = c.sum(gb_data)
            assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted)
            assert arr_equal(gb_result.col_0, gb_nums_sorted)

        # 0, 0, ALL_DTYPES, None, 1
        for dt in all_int_dtypes:
            # add assert warns
            c = Categorical(
                bytes_arr,
                ordered=True,
                sort_gb=False,
                dtype=dt,
                filter=None,
                base_index=1,
            )
            assert arr_equal(base_1_underlying_sorted, c._fa)
            assert arr_equal(cats_sorted, c.category_array)
            gb_result = c.sum(gb_data)
            assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted)
            assert arr_equal(gb_result.col_0, gb_nums_sorted)

        # 0, 0, ALL_DTYPES, FILTER, 0
        for dt in all_int_dtypes:
            with pytest.raises(ValueError):
                c = Categorical(
                    bytes_arr,
                    ordered=True,
                    sort_gb=False,
                    dtype=dt,
                    filter=b_filter,
                    base_index=0,
                )

        # 0, 0, ALL_DTYPES, FILTER, 1
        for dt in all_int_dtypes:
            # add assert warns
            c = Categorical(
                bytes_arr,
                ordered=True,
                sort_gb=False,
                dtype=dt,
                filter=b_filter,
                base_index=1,
            )
            assert arr_equal(underlying_sorted_filtered, c._fa)
            assert arr_equal(cats_sorted_filtered, c.category_array)
            gb_result = c.sum(gb_data)
            assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted_filtered)
            assert arr_equal(gb_result.col_0, gb_nums_sorted_filtered)

        # 0, 1, None, None, 0
        c = Categorical(
            bytes_arr, ordered=True, sort_gb=True, dtype=None, filter=None, base_index=0
        )
        assert arr_equal(base_0_underlying_sorted, c._fa)
        assert arr_equal(cats_sorted, c.category_array)
        gb_result = c.sum(gb_data)
        assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted)
        assert arr_equal(gb_result.col_0, gb_nums_sorted)

        # 0, 1, None, None, 1
        c = Categorical(
            bytes_arr, ordered=True, sort_gb=True, dtype=None, filter=None, base_index=1
        )
        assert arr_equal(base_1_underlying_sorted, c._fa)
        assert arr_equal(cats_sorted, c.category_array)
        gb_result = c.sum(gb_data)
        assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted)
        assert arr_equal(gb_result.col_0, gb_nums_sorted)

        # 0, 1, None, FILTER, 0
        with pytest.raises(ValueError):
            c = Categorical(
                bytes_arr,
                ordered=True,
                sort_gb=True,
                dtype=None,
                filter=b_filter,
                base_index=0,
            )

        # 0, 1, None, FILTER, 1
        c = Categorical(
            bytes_arr,
            ordered=True,
            sort_gb=True,
            dtype=None,
            filter=b_filter,
            base_index=1,
        )
        assert arr_equal(underlying_sorted_filtered, c._fa)
        assert arr_equal(cats_sorted_filtered, c.category_array)
        gb_result = c.sum(gb_data)
        assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted_filtered)
        assert arr_equal(gb_result.col_0, gb_nums_sorted_filtered)

        # 0, 1, ALL_DTYPES, None, 0
        for dt in all_int_dtypes:
            # add assert warns
            c = Categorical(
                bytes_arr,
                ordered=True,
                sort_gb=True,
                dtype=dt,
                filter=None,
                base_index=0,
            )
            assert arr_equal(base_0_underlying_sorted, c._fa)
            assert arr_equal(cats_sorted, c.category_array)
            gb_result = c.sum(gb_data)
            assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted)
            assert arr_equal(gb_result.col_0, gb_nums_sorted)

        # 0, 1, ALL_DTYPES, None, 1
        for dt in all_int_dtypes:
            # add assert warns
            c = Categorical(
                bytes_arr,
                ordered=True,
                sort_gb=True,
                dtype=dt,
                filter=None,
                base_index=1,
            )
            assert arr_equal(base_1_underlying_sorted, c._fa)
            assert arr_equal(cats_sorted, c.category_array)
            gb_result = c.sum(gb_data)
            assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted)
            assert arr_equal(gb_result.col_0, gb_nums_sorted)

        # 0, 1, ALL_DTYPES, FILTER, 0
        for dt in all_int_dtypes:
            with pytest.raises(ValueError):
                c = Categorical(
                    bytes_arr,
                    ordered=True,
                    sort_gb=True,
                    dtype=dt,
                    filter=b_filter,
                    base_index=0,
                )

        # 0, 1, ALL_DTYPES, FILTER, 1
        for dt in all_int_dtypes:
            # add assert warns
            c = Categorical(
                bytes_arr,
                ordered=True,
                sort_gb=False,
                dtype=dt,
                filter=b_filter,
                base_index=1,
            )
            assert arr_equal(underlying_sorted_filtered, c._fa)
            assert arr_equal(cats_sorted_filtered, c.category_array)
            gb_result = c.sum(gb_data)
            assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted_filtered)
            assert arr_equal(gb_result.col_0, gb_nums_sorted_filtered)

        # 1, 0, None, None, 0
        c = Categorical(
            bytes_arr,
            ordered=False,
            sort_gb=False,
            dtype=None,
            filter=None,
            base_index=0,
        )
        assert arr_equal(base_0_underlying_first_order, c._fa)
        assert arr_equal(cats_first_order, c.category_array)
        gb_result = c.sum(gb_data)
        assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_first_order)
        assert arr_equal(gb_result.col_0, gb_nums_first_order)

        # 1, 0, None, None, 1
        c = Categorical(
            bytes_arr,
            ordered=False,
            sort_gb=False,
            dtype=None,
            filter=None,
            base_index=1,
        )
        assert arr_equal(base_1_underlying_first_order, c._fa)
        assert arr_equal(cats_first_order, c.category_array)
        gb_result = c.sum(gb_data)
        assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_first_order)
        assert arr_equal(gb_result.col_0, gb_nums_first_order)

        # 1, 0, None, FILTER, 0
        with pytest.raises(ValueError):
            c = Categorical(
                bytes_arr,
                ordered=False,
                sort_gb=False,
                dtype=None,
                filter=b_filter,
                base_index=0,
            )

        # 1, 0, None, FILTER, 1
        c = Categorical(
            bytes_arr,
            ordered=False,
            sort_gb=False,
            dtype=None,
            filter=b_filter,
            base_index=1,
        )
        assert arr_equal(underlying_first_order_filtered, c._fa)
        assert arr_equal(cats_first_order_filtered, c.category_array)
        gb_result = c.sum(gb_data)
        assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_first_order_filtered)
        assert arr_equal(gb_result.col_0, gb_nums_first_order_filtered)

        # 1, 0, ALL_DTYPES, None, 0
        for dt in all_int_dtypes:
            # add assert warns
            c = Categorical(
                bytes_arr,
                ordered=False,
                sort_gb=False,
                dtype=dt,
                filter=None,
                base_index=0,
            )
            assert arr_equal(base_0_underlying_first_order, c._fa)
            assert arr_equal(cats_first_order, c.category_array)
            gb_result = c.sum(gb_data)
            assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_first_order)
            assert arr_equal(gb_result.col_0, gb_nums_first_order)

        # 1, 0, ALL_DTYPES, None, 1
        for dt in all_int_dtypes:
            # add assert warns
            c = Categorical(
                bytes_arr,
                ordered=False,
                sort_gb=False,
                dtype=dt,
                filter=None,
                base_index=1,
            )
            assert arr_equal(base_1_underlying_first_order, c._fa)
            assert arr_equal(cats_first_order, c.category_array)
            gb_result = c.sum(gb_data)
            assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_first_order)
            assert arr_equal(gb_result.col_0, gb_nums_first_order)

        # 1, 0, ALL_DTYPES, FILTER, 0
        for dt in all_int_dtypes:
            with pytest.raises(ValueError):
                c = Categorical(
                    bytes_arr,
                    ordered=False,
                    sort_gb=False,
                    dtype=dt,
                    filter=b_filter,
                    base_index=0,
                )

        # 1, 0, ALL_DTYPES, FILTER, 1
        for dt in all_int_dtypes:
            c = Categorical(
                bytes_arr,
                ordered=False,
                sort_gb=False,
                dtype=dt,
                filter=b_filter,
                base_index=1,
            )
            assert arr_equal(underlying_first_order_filtered, c._fa)
            assert arr_equal(cats_first_order_filtered, c.category_array)
            gb_result = c.sum(gb_data)
            assert arr_equal(
                    gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_first_order_filtered
                )
            assert arr_equal(gb_result.col_0, gb_nums_first_order_filtered)

        # 1, 1, None, None, 0
        c = Categorical(
            bytes_arr,
            ordered=False,
            sort_gb=True,
            dtype=None,
            filter=None,
            base_index=0,
        )
        assert arr_equal(base_0_underlying_first_order, c._fa)
        assert arr_equal(cats_first_order, c.category_array)
        gb_result = c.sum(gb_data)
        assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted)
        assert arr_equal(gb_result.col_0, gb_nums_sorted)

        # 1, 1, None, None, 1
        c = Categorical(
            bytes_arr,
            ordered=False,
            sort_gb=True,
            dtype=None,
            filter=None,
            base_index=1,
        )
        assert arr_equal(base_1_underlying_first_order, c._fa)
        assert arr_equal(cats_first_order, c.category_array)
        gb_result = c.sum(gb_data)
        assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted)
        assert arr_equal(gb_result.col_0, gb_nums_sorted)

        # 1, 1, None, FILTER, 0
        with pytest.raises(ValueError):
            c = Categorical(
                bytes_arr,
                ordered=False,
                sort_gb=True,
                dtype=None,
                filter=b_filter,
                base_index=0,
            )

        # 1, 1, None, FILTER, 1
        c = Categorical(
            bytes_arr,
            ordered=False,
            sort_gb=True,
            dtype=None,
            filter=b_filter,
            base_index=1,
        )
        assert arr_equal(underlying_first_order_filtered, c._fa)
        assert arr_equal(cats_first_order_filtered, c.category_array)
        gb_result = c.sum(gb_data)
        assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted_filtered)
        assert arr_equal(gb_result.col_0, gb_nums_sorted_filtered)

        # 1, 1, ALL_DTYPES, None, 0
        for dt in all_int_dtypes:
            # add assert warns
            c = Categorical(
                bytes_arr,
                ordered=False,
                sort_gb=True,
                dtype=dt,
                filter=None,
                base_index=0,
            )
            assert arr_equal(base_0_underlying_first_order, c._fa)
            assert arr_equal(cats_first_order, c.category_array)
            gb_result = c.sum(gb_data)
            assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted)
            assert arr_equal(gb_result.col_0, gb_nums_sorted)

        # 1, 1, ALL_DTYPES, None, 1
        for dt in all_int_dtypes:
            # add assert warns
            c = Categorical(
                bytes_arr,
                ordered=False,
                sort_gb=True,
                dtype=dt,
                filter=None,
                base_index=1,
            )
            assert arr_equal(base_1_underlying_first_order, c._fa)
            assert arr_equal(cats_first_order, c.category_array)
            gb_result = c.sum(gb_data)
            assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted)
            assert arr_equal(gb_result.col_0, gb_nums_sorted)

        # 1, 1, ALL_DTYPES, FILTER, 0
        with pytest.raises(ValueError):
            for dt in all_int_dtypes:
                c = Categorical(
                    bytes_arr,
                    ordered=False,
                    sort_gb=True,
                    dtype=dt,
                    filter=b_filter,
                    base_index=0,
                )

        # 1, 1, ALL_DTYPES, FILTER, 1
        for dt in all_int_dtypes:
            # add assert warns
            c = Categorical(
                bytes_arr,
                ordered=False,
                sort_gb=True,
                dtype=dt,
                filter=b_filter,
                base_index=1,
            )
            assert arr_equal(underlying_first_order_filtered, c._fa)
            assert arr_equal(cats_first_order_filtered, c.category_array)
            gb_result = c.sum(gb_data)
            assert arr_equal(gb_result[GROUPBY_KEY_PREFIX + '_0'], cats_sorted_filtered)
            assert arr_equal(gb_result.col_0, gb_nums_sorted_filtered)
