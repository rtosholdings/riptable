import numpy as np
import riptable as rt
from numpy.testing import assert_array_equal
from riptable import (
    FastArray,
    FA,
    Categorical,
    Dataset,
    logical,
    arange,
    mask_or,
    CatZero,
    Cat,
    isnan,
    isnotnan,
)
from riptable.rt_enum import GROUPBY_KEY_PREFIX
from enum import IntEnum

#%load_ext autoreload
#%autoreload 2
# from riptable import *
# from enum import IntEnum
str_fa = FastArray(
    [
        'c',
        'e',
        'e',
        'd',
        'c',
        'b',
        'd',
        'c',
        'a',
        'b',
        'd',
        'e',
        'c',
        'a',
        'e',
        'd',
        'b',
        'a',
        'b',
        'c',
        'd',
        'b',
        'e',
        'c',
        'c',
        'd',
        'e',
        'c',
        'a',
        'c',
    ]
)
int_fa = FastArray(
    [
        10,
        20,
        10,
        20,
        10,
        10,
        20,
        10,
        20,
        20,
        30,
        20,
        20,
        20,
        20,
        30,
        10,
        10,
        10,
        20,
        30,
        20,
        20,
        10,
        30,
        30,
        20,
        10,
        10,
        10,
    ]
)
flt_fa = FastArray(
    [
        88.97,
        50.05,
        36.16,
        63.22,
        27.33,
        17.98,
        23.35,
        52.60,
        26.82,
        78.75,
        34.20,
        49.35,
        19.50,
        29.91,
        74.96,
        94.41,
        27.95,
        56.65,
        38.08,
        16.51,
        36.81,
        80.14,
        53.64,
        49.55,
        65.82,
        85.28,
        61.68,
        72.85,
        91.71,
        61.12,
    ]
)
tens = FastArray([10] * 30)
ds = Dataset({'strings': str_fa.copy(), 'ints': int_fa, 'floats': flt_fa, 'tens': tens})
gb = ds.gb('strings')
ds_nums = Dataset({'ints': int_fa, 'floats': flt_fa, 'tens': tens})
data_to_compare = ['ints', 'floats', 'tens']
gbu = ds.gbu('strings')

gb_funcs_L1 = [
    'sum',
    'mean',
    'min',
    'max',
    'var',
    'std',
    'nansum',
    'nanmean',
    'nanmin',
    'nanmax',
    'nanvar',
    'nanstd',
]
gb_funcs_L2 = ['first', 'last', 'median', 'mode', 'nanmedian']
gb_funcs_L3 = ['cumsum', 'cumprod']
all_gb_ops = gb_funcs_L1 + gb_funcs_L2 + gb_funcs_L3

even_filter = logical(arange(30) % 2)
d_filter = str_fa != b'd'

d_filter_results = [
    0,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    0,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
]


complete_unique_cats = FastArray(['a', 'b', 'c', 'd', 'e'])
unsorted_unique_cats = FastArray(['b', 'c', 'e', 'a', 'd'])
complete_unique_ints = FastArray([10, 20, 30, 40, 50])
unsorted_unique_ints = FastArray([20, 30, 50, 10, 40])
complete_unique_flts = FastArray([10.0, 20.0, 30.0, 40.0, 50.0])

sorted_codes = FastArray(
    [
        2,
        4,
        4,
        3,
        2,
        1,
        3,
        2,
        0,
        1,
        3,
        4,
        2,
        0,
        4,
        3,
        1,
        0,
        1,
        2,
        3,
        1,
        4,
        2,
        2,
        3,
        4,
        2,
        0,
        2,
    ]
)
sorted_codes_matlab = FastArray(
    [
        3.0,
        5.0,
        5.0,
        4.0,
        3.0,
        2.0,
        4.0,
        3.0,
        1.0,
        2.0,
        4.0,
        5.0,
        3.0,
        1.0,
        5.0,
        4.0,
        2.0,
        1.0,
        2.0,
        3.0,
        4.0,
        2.0,
        5.0,
        3.0,
        3.0,
        4.0,
        5.0,
        3.0,
        1.0,
        3.0,
    ]
)
str_fa_with_invalid = FastArray(
    [
        'g',
        'e',
        'e',
        'd',
        'g',
        'b',
        'd',
        'g',
        'a',
        'b',
        'd',
        'e',
        'g',
        'a',
        'e',
        'd',
        'b',
        'a',
        'b',
        'g',
        'd',
        'b',
        'e',
        'g',
        'g',
        'd',
        'e',
        'g',
        'a',
        'g',
    ]
)

last_array = FastArray(
    [
        b'Bryant',
        b'Copeland',
        b'Copeland',
        b'Hill',
        b'Bryant',
        b'Smith',
        b'Hill',
        b'Bryant',
        b'Jones',
        b'Smith',
        b'Hill',
        b'Copeland',
        b'Bryant',
        b'Jones',
        b'Copeland',
        b'Hill',
        b'Smith',
        b'Jones',
        b'Smith',
        b'Bryant',
        b'Hill',
        b'Smith',
        b'Copeland',
        b'Bryant',
        b'Bryant',
        b'Hill',
        b'Copeland',
        b'Bryant',
        b'Jones',
        b'Bryant',
    ]
)

first_array = FastArray(
    [
        b'David',
        b'Susan',
        b'Susan',
        b'Paul',
        b'David',
        b'Charlie',
        b'Paul',
        b'David',
        b'Sam',
        b'Charlie',
        b'Paul',
        b'Susan',
        b'David',
        b'Sam',
        b'Susan',
        b'Paul',
        b'Charlie',
        b'Sam',
        b'Charlie',
        b'David',
        b'Paul',
        b'Charlie',
        b'Susan',
        b'David',
        b'David',
        b'Paul',
        b'Susan',
        b'David',
        b'Sam',
        b'David',
    ]
)


class str_enum(IntEnum):
    a = 0
    b = 1
    c = 2
    d = 3
    e = 4


class TestCategoricalGroupby:
    def test_friendly_name(self):
        pass

    def get_gb_results(self, op_name, gb, c, c_data, cols_to_match):
        gb_call = getattr(gb, op_name)
        gb_result = gb_call()
        gb_cat_call = getattr(c, op_name)
        gb_cat_result = gb_cat_call(c_data)
        # print('gb\n',gb_result)
        # print('cat\n',gb_cat_result)

        for col in cols_to_match:
            match = bool(np.all(gb_result[col] == gb_cat_result[col]))
        return match

    def get_filtered_bin_results(
        self, op_name, c, c_data, col_name, filter, bin_idx, correct
    ):
        match = False
        func = getattr(c, op_name)
        # print('flt_fa',flt_fa.get_name())
        empty_bin_result = func(flt_fa, filter=filter)[col_name][bin_idx]

        # if not nan
        if empty_bin_result == empty_bin_result:
            if empty_bin_result == correct:
                match = True
        else:
            if np.isnan(correct):
                match = True
        return match

    def funnel_all_tests(self, c, ds_gb, constructor_name, sorted=True):
        # no filter
        for op_name in all_gb_ops:
            # print('OP NAME WAS',op_name)
            match = self.get_gb_results(op_name, ds_gb, c, ds_nums, data_to_compare)
            assert match,\
                f"Categorical {c} constructed from: {constructor_name} Results did not match for {op_name} operation."

        # post filter
        if sorted is True:
            bin_idx = 3
        else:
            bin_idx = 2

        for op_name, correct in zip(gb_funcs_L1, d_filter_results):
            match = self.get_filtered_bin_results(
                op_name, c, flt_fa, 'floats', d_filter, bin_idx, correct
            )
            assert match,\
                f"Categorical constructed from: {constructor_name} Incorrect result for filtered bin during {op_name} operation. Expected {correct}"\

    def construction_with_invalid(self, c, constructor_name):
        for op_name, correct in zip(gb_funcs_L1, d_filter_results):
            match = self.get_filtered_bin_results(
                op_name, c, flt_fa, 'floats', None, 2, correct
            )
            assert match,\
                f"Categorical constructed from: {constructor_name} Incorrect result for filtered bin during {op_name} operation. Expected {correct}"

    def simple_string_set_item(self, *args, **kwargs):
        '''
        This test needs to be updated with different data that reflects the new comparison behavior.
        SJK: 9/24/2018

        '''

        source = kwargs['constructor_name']
        del kwargs['constructor_name']

        if 'categories' in kwargs:
            kwargs['categories'] = kwargs['categories'].copy()

        c = Categorical(*args, **kwargs)
        set_items = [
            # index by string
            (b'b', b'a'),
            (b'b', 'a'),
            # (b'b', 1),
            ('b', b'a'),
            ('b', 'a'),
            # ('b', 1),
            # index by bool array
            # boolean arrays can no longer be generated with these comparisons SJK 9/24/2018
            (c == b'b', b'a'),
            (c == b'b', 'a'),
            # (c == b'b', 1),
            (c == 'b', b'a'),
            (c == 'b', 'a'),
            # (c == 'b', 1),
            # (c == 2, b'a'),
            # (c == 2, 'a'),
            # (c == 2, 1),
            # integer index
            ([5, 9, 16, 18, 21], b'a'),
            ([5, 9, 16, 18, 21], 'a'),
            # ([ 5,  9, 16, 18, 21], 1),
            ([5, 9, 16, 18, 21], b'a'),
            ([5, 9, 16, 18, 21], 'a'),
            # ([ 5,  9, 16, 18, 21], 1),
            ([5, 9, 16, 18, 21], b'a'),
            ([5, 9, 16, 18, 21], 'a'),
            # ([ 5,  9, 16, 18, 21], 1),
        ]
        # this test needs to get reworked
        # no longer produces the correct result for all types of categoricals because of == comparison behavior
        for items in set_items:
            c = Categorical(*args, **kwargs)
            goal = c == ['a', 'b']
            c[items[0]] = items[1]
            result = c == items[1]
            all_set = np.sum(goal == result)
            assert all_set ==\
                30,\
                f"did not set c[{items[0]}] to {items[1]} for categorical from {source}"

            none_left = np.sum(c == 'b')
            assert none_left ==\
                0,\
                f"did not set c[{items[0]}] to {items[1]} for categorical from {source}"

    def mk_set_item(self, *args, **kwargs):
        source = kwargs['constructor_name']
        del kwargs['constructor_name']

        if 'categories' in kwargs:
            print('copying categories')
            kwargs['categories'] = kwargs['categories'].copy()

        c = Categorical(*args, **kwargs)
        set_items = [
            # index by string
            ((b'b', b'b'), (b'a', b'a')),
            ((b'b', b'b'), ('a', 'a')),
            ((b'b', b'b'), 5),
            (('b', 'b'), (b'a', b'a')),
            (('b', 'b'), ('a', 'a')),
            (('b', 'b'), 5),
            # index by bool array
            # (c == (b'b', b'b'), (b'a', b'a')),
            # (c == (b'b', b'b'), ('a', 'a')),
            # (c == (b'b', b'b'), 5),
            # (c == ('b', 'b'), (b'a', b'a')),
            # (c == ('b', 'b'), ('a', 'a')),
            # (c == ('b', 'b'), 5),
            # (c == 4, (b'a', b'a')),
            # (c == 4, ('a', 'a')),
            # (c == 4, 5),
            # integer index
            ([5, 9, 16, 18, 21], (b'a', b'a')),
            ([5, 9, 16, 18, 21], ('a', 'a')),
            ([5, 9, 16, 18, 21], 5),
            ([5, 9, 16, 18, 21], (b'a', b'a')),
            ([5, 9, 16, 18, 21], ('a', 'a')),
            ([5, 9, 16, 18, 21], 5),
            ([5, 9, 16, 18, 21], (b'a', b'a')),
            ([5, 9, 16, 18, 21], ('a', 'a')),
            ([5, 9, 16, 18, 21], 5),
        ]

        for items in set_items:
            c = Categorical(*args, **kwargs)
            goal = mask_or([c == ('a', 'a'), c == ('b', 'b')])
            c[items[0]] = items[1]
            result = c == items[1]
            all_set = np.sum(goal == result)
            assert all_set ==\
                30,\
                f"did not set c[{items[0]}] to {items[1]} for categorical from {source}"

            none_left = np.sum(c == ('b', 'b'))
            assert none_left ==\
                0,\
                f"did not set c[{items[0]}] to {items[1]} for categorical from {source}"

    # TODO pytest parameterize funnel_all_tests
    # --STRINGS-------------------------------------------------------------------------------
    def test_groupby_ops_string_list(self):
        c = Categorical(str_fa)

        self.funnel_all_tests(c, gb, "string list")
        self.simple_string_set_item(str_fa, constructor_name="string list")

    # --STRINGS + CATEGORIES------------------------------------------------------------------
    def test_groupby_ops_string_list_cats(self):
        c = Categorical(str_fa, complete_unique_cats)
        self.funnel_all_tests(c, gb, "string list + categories")

        self.simple_string_set_item(
            str_fa,
            categories=complete_unique_cats,
            constructor_name="string list + categories",
        )

        # 5/16/2019 invalid category must be in uniques
        # c_invalid = Categorical(str_fa_with_invalid, categories=complete_unique_cats, invalid='invalid')
        # self.construction_with_invalid(c_invalid, 'string list + categories')

    # --INDEX + CATEGORIES + BASE 0-----------------------------------------------------------
    def test_groupby_ops_user_codes_base_0(self):
        c = Categorical(
            sorted_codes.copy(), categories=complete_unique_cats, base_index=0
        )
        self.funnel_all_tests(c, gb, "index + categories + base_index 0")

        c = CatZero(sorted_codes.copy(), categories=complete_unique_cats)
        self.funnel_all_tests(c, gb, "index + categories + base_index 0")

        # self.simple_string_set_item(sorted_codes, categories=complete_unique_cats, base_index=0, constructor_name="string list")

    # --INDEX + CATEGORIES + BASE 1-----------------------------------------------------------
    def test_groupby_ops_user_codes_base_1(self):
        c = Categorical(sorted_codes + 1, complete_unique_cats, base_index=1)
        self.funnel_all_tests(c, gb, "index + categories + base_index 1")

        self.simple_string_set_item(
            sorted_codes.copy(),
            categories=complete_unique_cats,
            base_index=1,
            constructor_name="index + categories + base_index 1",
        )

    # --INDEX + INTEGER CATEGORIES + BASE 0-----------------------------------------------------------
    def test_groupby_ops_user_codes_int_base_0(self):
        c = Categorical(sorted_codes.copy(), complete_unique_ints, base_index=0)
        self.funnel_all_tests(c, gb, "index + integer categories + base_index 0")

        c = CatZero(sorted_codes.copy(), complete_unique_ints)
        self.funnel_all_tests(c, gb, "index + integer categories + base_index 0")

    # --INDEX + INTEGER CATEGORIES + BASE 1-----------------------------------------------------------
    def test_groupby_ops_user_codes_int_base_1(self):
        c = Categorical(sorted_codes + 1, complete_unique_ints, base_index=1)
        self.funnel_all_tests(c, gb, "index + integer categories + base_index 1")

    # --INDEX + FLOAT CATEGORIES + BASE 0-----------------------------------------------------------
    def test_groupby_ops_user_codes_float_base_0(self):
        c = Categorical(sorted_codes.copy(), complete_unique_flts, base_index=0)
        self.funnel_all_tests(c, gb, "index + integer categories + base_index 0")

        c = CatZero(sorted_codes.copy(), complete_unique_flts)
        self.funnel_all_tests(c, gb, "index + integer categories + base_index 0")

    # --INDEX + FLOAT CATEGORIES + BASE 1-----------------------------------------------------------
    def test_groupby_ops_user_codes_float_base_1(self):
        c = Categorical(sorted_codes + 1, complete_unique_flts, base_index=1)

    # --INDEX + INTENUM------------------------------------------------------------------------
    def test_groupby_ops_int_enum(self):
        c = Categorical(sorted_codes, str_enum)
        self.funnel_all_tests(c, gbu, "index + int enum", sorted=False)

        # self.simple_string_set_item(sorted_codes, str_enum, constructor_name="string list")

    # --INDEX + MAPPING------------------------------------------------------------------------
    def test_groupby_ops_mapping(self):
        d = dict(str_enum.__members__)
        d = {k: int(v) for k, v in d.items()}
        c = Categorical(sorted_codes, d)
        self.funnel_all_tests(c, gbu, "index + mapping dictionary", sorted=False)

    # --PANDAS CATEGORIAL----------------------------------------------------------------------
    def test_groupby_ops_pd_cat(self):
        import pandas as pd

        pdc = pd.Categorical(str_fa)
        c = Categorical(pdc)
        self.funnel_all_tests(c, gb, "pandas categorical")

        # need to copy provided categories for an accurate test
        # self.simple_string_set_item(pdc, constructor_name="pandas categorical")

    # --MATLAB CATEGORIAL----------------------------------------------------------------------
    def test_groupby_ops_matlab(self):
        c = Categorical(sorted_codes_matlab, complete_unique_cats, from_matlab=True)
        self.funnel_all_tests(c, gb, "matlab categorical")

    # --STRINGS + UNSORTED CATEGORIES-----------------------------------------------------------
    def test_groupby_ops_string_list_cats_unsorted(self):
        c = Categorical(str_fa, unsorted_unique_cats, ordered=False, sort_gb=True)
        self.funnel_all_tests(c, gb, "string list + unsorted categories")

    # --MULTIKEY DICTIONARY---------------------------------------------------------------------
    def test_groupby_ops_multikey_dict(self):
        mk_dict = {'string1': str_fa, 'string2': str_fa}
        mk_gb = Dataset(
            {
                'string1': str_fa.copy(),
                'string2': str_fa.copy(),
                'ints': int_fa,
                'floats': flt_fa,
                'tens': tens,
            }
        ).gbu(['string1', 'string2'])
        c = Categorical(mk_dict)
        self.funnel_all_tests(c, mk_gb, "multikey dictionary", sorted=False)

        # setitem hits comparison functions - need to rewrite these tests after comparison behavior change
        # self.mk_set_item(mk_dict, constructor_name="multikey dictionary")

        # conflicting names
        x = str_fa.copy()
        y = str_fa.copy()
        z = str_fa.copy()
        x.set_name('strings')
        y.set_name('strings')
        z.set_name('strings1')
        c = Categorical([x, y, z])
        assert c._categories_wrap.ncols ==\
            3,\
            f"incorrect number of columns for multikey from list. {c._categories_wrap.ncols} vs. 3"
        # 04/25/2019 all default column names now happen in grouping object
        assert list(c.categories().keys())\
            == ['strings', GROUPBY_KEY_PREFIX + '_c1', 'strings1'],\
            f"column names did not match for multikey from list. {list(c.categories().keys())} vs. ['strings','strings2','strings1']"

    # --MULTIKEY LIST---------------------------------------------------------------------------
    def test_groupby_ops_multikey_list(self):
        mk_list = [str_fa.copy(), str_fa.copy()]
        mk_gb = Dataset(
            {
                'string1': str_fa.copy(),
                'string2': str_fa.copy(),
                'ints': int_fa,
                'floats': flt_fa,
                'tens': tens,
            }
        ).gbu(['string1', 'string2'])
        c = Categorical(mk_list)
        self.funnel_all_tests(c, mk_gb, "multikey list", sorted=False)

        # setitem hits comparison functions - need to rewrite these tests after comparison behavior change
        # self.mk_set_item(mk_list, constructor_name="multikey list")

    # --MULTIKEY COUNT--------------------------------------------------------------------------
    def test_multikey_count(self):
        mk_list = [str_fa.copy(), int_fa.copy(), str_fa.copy(), int_fa.copy()]
        c_multi = Categorical(mk_list)
        result_counts = c_multi.count().Count
        correct_counts = FastArray([6, 5, 1, 2, 3, 2, 2, 4, 2, 2, 1])
        all_correct = bool(np.all(result_counts == correct_counts))
        assert all_correct,\
            f"Incorrect result for multikey count for 4 keys. {result_counts} vs. {correct_counts}"

    # --STRING-LIKE SINGLE KEY COUNT------------------------------------------------------------
    def test_single_key_string_count(self):
        correct_counts = FastArray([4, 5, 9, 6, 6])

        # for sorting/count bug fix 8/21/2018
        c_make_unique = Categorical(str_fa)
        result_counts = c_make_unique.count().Count
        match = bool(np.all(result_counts == correct_counts))
        assert match

        c_from_codes = Categorical(sorted_codes, complete_unique_cats, base_index=0)
        result_counts = c_from_codes.count().Count
        match = bool(np.all(result_counts == correct_counts))
        assert match

        c_from_codes_unsorted = Categorical(
            sorted_codes, unsorted_unique_cats, base_index=0
        )
        result_counts = c_from_codes_unsorted.count().Count
        match = bool(np.all(result_counts == correct_counts))
        assert match
        # 8/24/2018 SJK - default name for groupby key columns might change, so selected this by index
        # also, in most cases (save intenum/dict) categorical groupby no longer returns a categorical
        result_keys = c_from_codes_unsorted.count()[1]
        match = bool(np.all(result_keys == unsorted_unique_cats))
        assert match, f"Result: {result_keys} Expected: {unsorted_unique_cats}"

    def test_cumcount_vs_gb(self):
        arr = np.random.choice(['a', 'b', 'c', 'd', 'e'], 50)
        ds = Dataset({'keycol': arr, 'col1': arange(50), 'col2': arange(50)})
        gb_result = ds.gb('keycol').cumcount()

        c = Categorical(ds.keycol)
        c_result = c.cumcount()

        rdiff = gb_result - c_result
        assert sum(rdiff) == 0

        f = logical(arange(50) % 2)
        c_result = c.cumcount(filter=f)
        assert bool(np.all(isnotnan(c_result[f])))
        assert bool(np.all(isnan(c_result[~f])))

    # --MULTIKEY SINGLE STRING KEY--------------------------------------------------------------
    def test_groupby_ops_multikey_single_string(self):
        c = Categorical({'string_col': str_fa.copy()})
        self.funnel_all_tests(c, gb, "multikey single string key")

    # --MULTIKEY SINGLE NUMERIC KEY-------------------------------------------------------------
    def test_groupby_ops_multikey_single_numeric(self):
        c = Categorical({'codes': sorted_codes}, ordered=False, sort_gb=False)
        self.funnel_all_tests(c, gbu, "multikey single numeric key", sorted=False)

    # TODO pytest parameterize empty_results table
    def test_empty_category(self):
        # 5/16/2019 invalid category must be in uniques
        # c = Categorical(str_fa_with_invalid, complete_unique_cats, invalid='invalid')
        # can test empty bin like this, the third result will be empty
        c = Categorical(
            np.random.choice(['a', 'b', 'd', 'e'], 30), ['a', 'b', 'c', 'd', 'e']
        )
        empty_result = [
            ('sum', 0.0),
            ('mean', np.nan),
            ('min', np.nan),
            ('max', np.nan),
            ('var', np.nan),
            ('std', np.nan),
            ('nansum', 0.0),
            ('nanmean', np.nan),
            ('nanmin', np.nan),
            ('nanmax', np.nan),
            ('nanvar', np.nan),
            ('nanstd', np.nan),
        ]

        for correct_tup in empty_result:
            func = getattr(c, correct_tup[0])
            result = func(ds_nums).floats[2]
            a = np.isnan(correct_tup[1])

            if np.isnan(correct_tup[1]):
                assert result !=\
                    result,\
                    f"Did not product correct result for empty category after {correct_tup[0]} operation."
            else:
                assert result ==\
                    correct_tup[1],\
                    f"Did not product correct result for empty category after {correct_tup[0]} operation."

    def test_igroup_dtype(self):
        '''
        The categorical only has 3 categories, so its index array will be int8
        after pack_by_group() is called in a grouping operation, it generates iGroup - which
        needs to be large enough to hold an integer equal to the length of the categorical
        '''
        c = Categorical(np.random.choice([b'a', b'b', b'c'], 1_000_000))
        c_was_int8 = np.int8 == c.dtype
        assert c_was_int8

        _ = c.groups
        igroup_was_int32 = c.grouping.iGroup.dtype == np.int32
        assert igroup_was_int32

    # --TEST ALL CONSTRUCTOR FLAGS-----------------------------------------------------------
    def test_pre_filter(self):
        c = Categorical(str_fa, filter=even_filter)
        assert c._filter == None

        result = c.sum(ds_nums)
        one_fifty = sum(result.tens)
        assert one_fifty == 150

    def test_specify_gb_data(self):
        str_col = ['a', 'a', 'b', 'c', 'a']
        num_col = [10, 10, 20, 30, 10]
        col1 = np.arange(5)
        col2 = np.arange(5)
        small_ds = Dataset(
            {'str_col': str_col, 'num_col': num_col, 'col1': col1, 'col2': col2}
        )
        ds_to_operate_on = small_ds[['col1', 'col2']]

        c = Categorical(str_col)

        # dataset
        d = c.sum(ds_to_operate_on)

        # single
        # list
        d = c.sum([col1, col2])

        # tuple
        d = c.sum((col1, col2))

        # dict
        d = c.sum({'a': col1, 'b': col2})

        # multiple
        d = c.sum(col1, col2)

    def test_as_categorical(self):
        ds = Dataset(
            {
                'keycol1': np.random.choice(['a', 'b', 'c'], 30),
                'keycol2': np.random.choice(['a', 'b', 'c'], 30),
                'data': np.random.rand(30),
            }
        )

        gbu = ds.gbu('keycol1')
        c = Categorical(ds.keycol1, ordered=False, sort_gb=False)
        cgbu = gbu.as_categorical()

        gbu_result = gbu.sum()
        c_result = c.sum(ds.data)
        cgbu_result = cgbu.sum(ds.data)

        for name, col in gbu_result.items():
            assert bool(np.all(c_result[name] == col))
            assert bool(np.all(cgbu_result[name] == col))

    def test_gb_labels_enum(self):
        # make sure enum groupby keys are displayed as string,  not integer code
        c = Categorical(
            [10, 10, 10, 20, 30, 20, 10, 20, 20], {'a': 30, 'b': 20, 'c': 10}
        )
        c_result = c.count()
        c_labels = c_result[c_result.label_get_names()][0]

        ds = Dataset({'catcol': c, 'data': arange(9)})
        ds_result = ds.gbu('catcol').count()
        ds_labels = ds_result[ds_result.label_get_names()][0]

        assert c_labels.dtype.char == ds_labels.dtype.char
        assert bool(np.all(c_labels == ds_labels))

    def test_groupby_categorical_sort(self):
        """
        Test that groupby on a categorical sorts the dataset correctly
        """
        ds = rt.Dataset()
        cats = ['z', 'y', 'x', 'w', 'a', 'b', 'c', 'd']
        vals = [0, 1, 2, 3, 4, 5, 6, 7]
        expected = dict(zip(cats, vals))

        ds["Cat"] = rt.Categorical([cats[xx % len(cats)] for xx in range(100)])

        # two identical columns
        ds["Value1"] = [vals[xx % len(cats)] for xx in range(100)]
        ds["Value2"] = [vals[xx % len(cats)] for xx in range(100)]

        grp = ds.groupby("Cat").mean()
        grp["Expected"] = [expected[xx] for xx in grp.Cat.astype('U')]

        diff = rt.sum(rt.abs(grp.Expected - grp.Value1))
        diff += rt.sum(rt.abs(grp.Expected - grp.Value2))

        assert diff <= 1e-9

    def test_shift(self):
        """
        Test that Categorical.shift shifts the values in an array or Dataset *per group*.
        """
        result = rt.Cat([1, 1, 1, 2]).shift(arange(4), window=1)[0]
        assert result[1] == 0
        assert result[2] == 1

        result = rt.Cat([1, 1, 1, 2]).shift([5, 6, 7, 8], window=1)[0]
        assert result[1] == 5
        assert result[2] == 6

    def test_fill_backward(self):
        """
        Test that Categorical.fill_backward fills values backward *per group*.
        """
        data = rt.FA([1.0, 4.0, np.nan, np.nan, 9.0, 16.0])
        cat = rt.Categorical(['A', 'B', 'A', 'B', 'A', 'B'])

        result = cat.fill_backward(data)

        # The result of this function should be a Dataset.
        assert isinstance(result, rt.Dataset)

        # The dataset should have the same number of rows as the data arrays
        # we operated on (an invariant of apply_nonreduce/"scan"/"prefix sum").
        assert result.shape[0] == len(data)

        # The dataset should have (N+M) columns, where N is the number
        # of keys within the Categorical and M is the number of columns
        # we performed the operation on.
        expected_col_count = len(cat.category_dict) + 1
        assert result.shape[1] == expected_col_count

        # Check the resulting data; the dtype of the data should be the
        # same as the original column.
        assert_array_equal(result[0], rt.FA([1.0, 4.0, 9.0, 16.0, 9.0, 16.0]))

    def test_fill_forward(self):
        """
        Test that Categorical.fill_forward fills values forward *per group*.
        """
        data = rt.FA([1.0, 4.0, 9.0, 16.0, np.nan, np.nan])
        cat = rt.Categorical(['A', 'B', 'A', 'B', 'A', 'B'])

        result = cat.fill_forward(data)

        # The result of this function should be a Dataset.
        assert isinstance(result, rt.Dataset)

        # The dataset should have the same number of rows as the data arrays
        # we operated on (an invariant of apply_nonreduce/"scan"/"prefix sum").
        assert result.shape[0] == len(data)

        # The dataset should have (N+M) columns, where N is the number
        # of keys within the Categorical and M is the number of columns
        # we performed the operation on.
        expected_col_count = len(cat.category_dict) + 1
        assert result.shape[1] == expected_col_count

        # Check the resulting data; the dtype of the data should be the
        # same as the original column.
        assert_array_equal(result[0], rt.FA([1.0, 4.0, 9.0, 16.0, 9.0, 16.0]))

    def test_nobins(self):
        """
        Tests that Categorical.median() works correctly when there
        are no bins with data because the Categorical has been created
        with a pre-filter which has filtered out all data in the Dataset.
        """

        data = Dataset()
        data.Group = np.random.randint(0, 10, 100_000)
        data.Values = np.random.randint(0, 10, 100_000)
        x = data.cat('Group', filter=data.Group < 0)
        x.median(data.Values)

    def test_count(self):
        """
        Tests that Categorical.count(transform=True) works correctly.
        """

        x = Cat(np.arange(10) % 3)
        y = x.count(transform=True)[0]
        assert_array_equal(y, FA([4, 3, 3, 4, 3, 3, 4, 3, 3, 4]))
