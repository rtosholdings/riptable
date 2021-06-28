import pytest
import os
import pandas as pd
import riptable as rt

from enum import IntEnum
from numpy.testing import assert_array_equal
from riptable import *
from riptable import save_sds, load_sds
from riptable import FastArray, Categorical, CatZero
from riptable.rt_categorical import Categories
from riptable.rt_enum import (
    INVALID_DICT,
)
from riptable.rt_enum import (
    DisplayLength,
    DisplayJustification,
    DisplayColumnColors,
)
from riptable.rt_enum import CategoryMode, TypeRegister
from riptable.rt_numpy import isnan, isnotnan, arange, ones
from riptable.tests.test_utils import (
    get_categorical_data_factory_method,
    get_all_categorical_data,
)
from riptable.rt_sds import SDSMakeDirsOn
from riptable.tests.utils import LikertDecision



# change to true since we write into /tests directory
SDSMakeDirsOn()

three_unicode = np.array(["AAPL\u2080", "AMZN\u2082", "IBM\u2081"])
three_bytes = FastArray([b'a', b'b', b'c'])
three_ints = FastArray([1, 2, 3])

compare_func_names = ['__ne__', '__eq__', '__ge__', '__gt__', '__le__', '__lt__']
int_success = [
    np.array([True, False, True]),  # ne
    np.array([False, True, False]),  # eq
    np.array([False, True, True]),  # ge
    np.array([False, False, True]),  # gt
    np.array([True, True, False]),  # le
    np.array([True, False, False]),  # lt
]
same_success = [
    np.array([False, False, False]),  # ne
    np.array([True, True, True]),  # eq
    np.array([True, True, True]),  # ge
    np.array([False, False, False]),  # gt
    np.array([True, True, True]),  # le
    np.array([False, False, False]),  # lt
]
diff_success = [
    np.array([True, False, True]),  # ne
    np.array([False, True, False]),  # eq
    np.array([False, True, False]),  # ge
    np.array([False, False, False]),  # gt
    np.array([True, True, True]),  # le
    np.array([True, False, True]),  # lt
]
ShowCompareInfo = False

list_bytes = [b'b', b'b', b'a', b'd', b'c']
list_unicode = ['b', 'b', 'a', 'd', 'c']
list_true_unicode = [u'b\u2082', u'b\u2082', u'a\u2082', u'd\u2082', u'c\u2082']


decision_dict = dict(zip(LikertDecision.__members__.keys(), [int(v) for v in LikertDecision.__members__.values()],))


def array_equal(arr1, arr2):
    subr = arr1 - arr2
    sumr = sum(subr == 0)
    result = sumr == len(arr1)
    if not result:
        print("array comparison failed", arr1, arr2)
    return result


class TestCategorical:
    def _notimpl(self):
        pytest.skip("This test needs to be implemented.")

    def test_constructor(self):
        # from pandas categorical

        # from single parameter

        # from two parameters
        # ndarray
        # python list

        self._notimpl()

    def test_ctor_list(self):
        c_bytes = Categorical(list_bytes)
        assert c_bytes.dtype == np.int8, f"Dtype {c_bytes.dtype} was not correct for construction from small list."
        assert len(c_bytes) == 5, f"Length of underlying index array was incorrect for construction from bytes."
        unique_bytes = np.unique(list_bytes)
        assert np.all(
            c_bytes._categories_wrap._list == unique_bytes
        ), f"Categories did not generate a unique list of categories from input bytes list."

        c_unicode = Categorical(list_unicode)
        assert c_unicode.dtype == np.int8, f"Dtype {c_unicode.dtype} was not correct for construction from small list."
        assert len(c_unicode) == 5, f"Length of underlying index array was incorrect for construction from unicode."
        assert (
            len(c_unicode._categories_wrap) == 4
        ), f"Length of unique categories was incorrect for construction from unicode."
        assert (
            c_unicode._categories_wrap._list[0] == b'a'
        ), f"Unique categories were not sorted for construction from unicode."
        assert c_unicode._categories_wrap._list.dtype.char == 'S', f"Unicode strings were not flipped to byte strings."

        c_true_unicode = Categorical(list_true_unicode)
        assert (
            c_true_unicode.dtype == np.int8
        ), f"Dtype {c_true_unicode.dtype} was not correct for construction from small list."
        assert (
            len(c_true_unicode) == 5
        ), f"Length of underlying index array was incorrect for construction from true unicode."
        assert (
            len(c_true_unicode._categories_wrap) == 4
        ), f"Length of unique categories was incorrect for construction from true unicode."
        assert (
            c_true_unicode._categories_wrap._list[0] == u'a\u2082'
        ), f"Unique categories were not sorted for construction from true unicode."
        assert (
            c_true_unicode._categories_wrap._list.dtype.char == 'U'
        ), f"Unicode strings were not flipped to byte strings."

    def test_ctor_nparray(self):
        c_bytes = Categorical(np.array(list_bytes))
        assert c_bytes.dtype == np.int8, f"Dtype {c_bytes.dtype} was not correct for construction from small list."
        assert len(c_bytes) == 5, f"Length of underlying index array was incorrect for construction from bytes."
        unique_bytes = np.unique(list_bytes)
        assert np.all(
            c_bytes._categories_wrap._list == unique_bytes
        ), f"Categories did not generate a unique list of categories from input bytes list."

        c_unicode = Categorical(np.array(list_unicode))
        assert c_unicode.dtype == np.int8, f"Dtype {c_unicode.dtype} was not correct for construction from small list."
        assert len(c_unicode) == 5, f"Length of underlying index array was incorrect for construction from unicode."
        assert (
            len(c_unicode._categories_wrap._list) == 4
        ), f"Length of unique categories was incorrect for construction from unicode."
        assert (
            c_unicode._categories_wrap._list[0] == b'a'
        ), f"Unique categories were not sorted for construction from unicode."
        assert c_unicode._categories_wrap._list.dtype.char == 'S', f"Unicode strings were not flipped to byte strings."

        c_true_unicode = Categorical(np.array(list_true_unicode))
        assert (
            c_true_unicode.dtype == np.int8
        ), f"Dtype {c_true_unicode.dtype} was not correct for construction from small list."
        assert (
            len(c_true_unicode) == 5
        ), f"Length of underlying index array was incorrect for construction from true unicode."
        assert (
            len(c_true_unicode._categories_wrap._list) == 4
        ), f"Length of unique categories was incorrect for construction from true unicode."
        assert (
            c_true_unicode._categories_wrap._list[0] == u'a\u2082'
        ), f"Unique categories were not sorted for construction from true unicode."
        assert (
            c_true_unicode._categories_wrap._list.dtype.char == 'U'
        ), f"Unicode strings were not flipped to byte strings."

    def test_ctor_values_and_cats(self):
        v_bytes = [b'IBM', b'AAPL', b'AMZN', b'IBM', b'hello']
        v_str = ['IBM', 'AAPL', 'AMZN', 'IBM', 'hello']
        v_true = [
            u'IBM\u2082',
            u'AAPL\u2082',
            u'AMZN\u2082',
            u'IBM\u2082',
            u'hello\u2082',
        ]

        c_bytes = [b'AAPL', b'AMZN', b'IBM']
        c_str = ['AAPL', 'AMZN', 'IBM']
        c_true = [u'AAPL\u2082', u'AMZN\u2082', u'IBM\u2082']

        v_correct = [2, 0, 1, 2, 3]
        c_correct = [b'AAPL', b'AMZN', b'IBM', b'inv']

        valid_v = [
            v_bytes,
            v_str,
            np.array(v_bytes),
            np.array(v_str),
            FastArray(v_bytes),
            FastArray(v_str),
        ]
        valid_c = [
            c_bytes,
            c_str,
            np.array(c_bytes),
            np.array(c_str),
            FastArray(c_bytes),
            FastArray(c_str),
        ]

        for v in valid_v:
            vdt = None
            if hasattr(v, 'dtype'):
                vdt = v.dtype
            else:
                vdt = type(v)
            for c in valid_c:
                cdt = None
                if hasattr(c, 'dtype'):
                    cdt = c.dtype
                else:
                    cdt = type(c)
                # error if no invalid provided
                with pytest.raises(ValueError):
                    cat = Categorical(v, c)

                # accept invalid and correctly assign
                # cat = Categorical(v, c, invalid_category=b'inv')
                # self.assertEqual(cat._categories.dtype.char, 'S', msg=f"Categorical from v: {vdt} and c: {cdt} did not flip categories to bytestring")
                # v_is_correct = bool(np.all(v_correct == cat.view(FastArray)))
                # self.assertTrue(v_is_correct, msg=f"Did not create the correct underlying index array from v: {vdt} and c: {cdt}")
                # c_is_correct = bool(np.all(c_correct == cat._categories))
                # self.assertTrue(c_is_correct, msg=f"Did not create the correct categories from v: {vdt} and c: {cdt}")

        # v = v_true
        # vdt = "TRUE unicode"
        # for c in valid_c:
        #    if hasattr(c,'dtype'):
        #        cdt = c.dtype
        #    else:
        #        cdt = type(c)
        #        cat = Categorical(v,c)

    # ---------------------------------------------------------------------------
    def test_ctor_bad_index(self):
        idx_list = [1, 2, 3, 4, 5]
        str_list = ['a', 'b']
        with pytest.raises(ValueError):
            c = Categorical(idx_list, str_list)

    # ---------------------------------------------------------------------------
    def test_ctor_non_unique(self):
        '''
        riptable categoricals, like pandas categoricals, do not allow a non-unique list of categories when an index array is provided.
        '''
        idx_list = [0, 1]
        str_list = ['b', 'b', 'a']
        c = Categorical(idx_list, str_list)

    # ---------------------------------------------------------------------------
    def test_ctor_enum(self):
        codes = [1, 44, 44, 133, 75]
        c = Categorical(codes, LikertDecision)

    # ---------------------------------------------------------------------------
    def test_compare_enum_int(self):
        compare_func_names = [
            '__ne__',
            '__eq__',
            '__ge__',
            '__gt__',
            '__le__',
            '__lt__',
        ]
        codes = [1, 44, 44, 133, 75]
        valid_idx = 44
        bad_idx = 43
        valid_idx_correct = [
            FastArray([True, False, False, True, True]),
            FastArray([False, True, True, False, False]),
            FastArray([False, True, True, True, True]),
            FastArray([False, False, False, True, True]),
            FastArray([True, True, True, False, False]),
            FastArray([True, False, False, False, False]),
        ]
        bad_idx_correct = [
            FastArray([True, True, True, True, True]),
            FastArray([False, False, False, False, False]),
            FastArray([False, True, True, True, True]),
            FastArray([False, True, True, True, True]),
            FastArray([True, False, False, False, False]),
            FastArray([True, False, False, False, False]),
        ]
        for d in (LikertDecision, decision_dict):
            c = Categorical(codes, d)

            # test valid integer code
            for name, correct in zip(compare_func_names, valid_idx_correct):
                func = c.__getattribute__(name)
                result = func(valid_idx)
                was_correct = bool(np.all(correct == result))
                assert (
                    was_correct
                ), f"Categorical enum comparison failed with good integer index on {name} operation. {c.view(FastArray)} code: {valid_idx}"

            # test invalid integer code
            for name, correct in zip(compare_func_names, bad_idx_correct):
                func = c.__getattribute__(name)
                result = func(bad_idx)
                was_correct = bool(np.all(correct == result))
                assert was_correct, f"Categorical enum comparison failed with good integer index on {name} operation"

    # ---------------------------------------------------------------------------
    def test_compare_enum_str(self):
        compare_func_names = [
            '__ne__',
            '__eq__',
            '__ge__',
            '__gt__',
            '__le__',
            '__lt__',
        ]
        codes = [1, 44, 44, 133, 75]
        valid_idx = 'StronglyAgree'
        bad_idx = 'x'
        valid_idx_correct = [
            FastArray([True, False, False, True, True]),
            FastArray([False, True, True, False, False]),
            FastArray([False, True, True, True, True]),
            FastArray([False, False, False, True, True]),
            FastArray([True, True, True, False, False]),
            FastArray([True, False, False, False, False]),
        ]
        for d in (LikertDecision, decision_dict):
            c = Categorical(codes, d)

            # test valid category string
            for name, correct in zip(compare_func_names, valid_idx_correct):
                func = c.__getattribute__(name)
                result = func(valid_idx)
                was_correct = bool(np.all(correct == result))
                assert was_correct, f"Categorical enum comparison failed with good category string on {name} operation"

            # test invalid category string
            for name in compare_func_names:
                func = c.__getattribute__(name)
                with pytest.raises(ValueError):
                    result = func(bad_idx)

    def test_map(self):

        c = Categorical(['b', 'b', 'c', 'a', 'd'], ordered=False)
        mapping = {'a': 'AA', 'b': 'BB', 'c': 'CC', 'd': 'DD'}
        result = c.map(mapping)
        correct = FastArray([b'BB', b'BB', b'CC', b'AA', b'DD'])
        assert bool(np.all(result == correct))
        c = Categorical(['b', 'b', 'c', 'a', 'd'], ordered=False, base_index=0)
        result = c.map(mapping)
        assert bool(np.all(result == correct))

        c = Categorical(['b', 'b', 'c', 'a', 'd'], ordered=False)
        mapping = {'a': 'AA', 'b': 'BB', 'c': 'CC'}
        result = c.map(mapping, invalid='INVALID')
        correct = FastArray([b'BB', b'BB', b'CC', b'AA', b'INVALID'])
        assert bool(np.all(result == correct))
        c = Categorical(['b', 'b', 'c', 'a', 'd'], ordered=False, base_index=0)
        result = c.map(mapping, invalid='INVALID')
        assert bool(np.all(result == correct))

        c = Categorical(['b', 'b', 'c', 'a', 'd'], ordered=False)
        mapping = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        result = c.map(mapping, invalid=666)
        correct = FastArray([2.0, 2.0, 3.0, 1.0, 666.0])
        assert bool(np.all(result == correct))
        c = Categorical(['b', 'b', 'c', 'a', 'd'], ordered=False, base_index=0)
        result = c.map(mapping, invalid=666)
        assert bool(np.all(result == correct))

        c = Categorical(['b', 'b', 'c', 'a', 'd'], ordered=False)
        result = c.map(mapping)
        assert np.isnan(result[4])
        c = Categorical(['b', 'b', 'c', 'a', 'd'], ordered=False, base_index=0)
        result = c.map(mapping)
        assert np.isnan(result[4])

        c = Categorical(['b', 'b', 'c', 'a', 'd'], ordered=False)
        mapping = FastArray(['w', 'x', 'y', 'z'])
        result = c.map(mapping)
        correct = FastArray([b'w', b'w', b'x', b'y', b'z'])
        assert bool(np.all(result == correct))
        c = Categorical(['b', 'b', 'c', 'a', 'd'], ordered=False, base_index=0)
        result = c.map(mapping)
        assert bool(np.all(result == correct))

        c = Categorical([2, 2, 3, 1, 4, 0], ['a', 'b', 'c', 'd'])
        mapping = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        result = c.map(mapping, invalid=666)
        correct = FastArray([2.0, 2.0, 3.0, 1.0, 666.0, 666.0])
        assert bool(np.all(result == correct))

    # ---------------------------------------------------------------------------
    def test_from_category(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        bin = c.from_category('a')
        assert bin == 1

        c = Categorical(['a', 'a', 'b', 'c', 'a'], base_index=0)
        bin = c.from_category(b'a')
        assert bin == 0

        with pytest.raises(ValueError):
            bin = c.from_category('z')

        c = Categorical(np.arange(5, 10))
        bin = c.from_category(5)
        assert bin == 1

        with pytest.raises(ValueError):
            bin = c.from_category(100)

        c = Categorical([FastArray(['a', 'b', 'c']), np.arange(3)])
        bin = c.from_category(('c', 2))
        assert bin == 3

    # ---------------------------------------------------------------------------
    def test_getitem_enum_int(self):
        codes = [1, 44, 44, 133, 75]
        correct_strings = [
            'StronglyDisagree',
            'StronglyAgree',
            'StronglyAgree',
            'Agree',
            'Disagree',
        ]
        c = Categorical(codes, LikertDecision)

        # getitem good init
        for idx in range(5):
            assert correct_strings[idx] == c[idx], f"Failed to return correct string for valid index in categorical."

        # getitem bad init
        with pytest.raises(IndexError):
            result = c[5]

    # ---------------------------------------------------------------------------
    def test_getitem_enum_int_list(self):
        codes = [1, 44, 44, 133, 75]
        correct_strings = [
            'StronglyDisagree',
            'StronglyAgree',
            'StronglyAgree',
            'Agree',
            'Disagree',
        ]
        c = Categorical(codes, LikertDecision)

        result = c[[1, 4]]
        assert isinstance(
            result, Categorical
        ), f"Failed to return Categorical when indexing by integer list. Returned {type(result)} instead."
        assert result[0] == 'StronglyAgree'
        assert result[1] == 'Disagree'

        result = c[np.array([1, 4])]
        assert isinstance(
            result, Categorical
        ), f"Failed to return Categorical when indexing by integer list. Returned {type(result)} instead."
        assert result[0] == 'StronglyAgree'
        assert result[1] == 'Disagree'

        result = c[FastArray([1, 4])]
        assert isinstance(
            result, Categorical
        ), f"Failed to return Categorical when indexing by integer list. Returned {type(result)} instead."
        assert result[0] == 'StronglyAgree'
        assert result[1] == 'Disagree'

    def test_getitem_enum(self):
        self._notimpl()

    def test_setitem_enum(self):
        self._notimpl()

    # -------------------------------------------- MATLAB ----------------------------------
    def test_ctor_matlab(self):
        idx_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        str_list = ['a', 'b', 'c', 'd', 'e']
        with pytest.raises(TypeError):
            c = Categorical(idx_list, str_list)

        c = Categorical(idx_list, str_list, from_matlab=True)
        assert c[0] == 'a'
        assert c.dtype == np.dtype(np.int8)

    # def test_ctor_matlab_non_unique(self):
    #    idx_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    #    str_list = ['a','b','c','d','d']
    #    with self.assertRaises(ValueError, msg=f"Failed to raise error when MATLab categories were not unique."):
    #        c = Categorical(idx_list, str_list, from_matlab=True)

    # ------------------------------- PANDAS CATEGORICAL ----------------------------------
    def test_ctor_pandas_cat(self):
        idx_list = [0, 1, 2, 3, 4]
        str_list = ['a', 'b', 'c', 'd', 'e']
        pd_c = pd.Categorical.from_codes(idx_list, str_list)
        pd_c = Categorical(pd_c)
        rt_c = Categorical(idx_list, str_list)

        cats_match = bool(np.all(pd_c.category_array == rt_c.category_array))
        assert cats_match, f"Failed to create matching categories from pandas categorical"

        # idx_match = bool(np.all(pd_c.view(np.ndarray)+1 == rt_c.view(np.ndarray)))
        # self.assertTrue(idx_match, msg=f"Failed to create matching unerlying array from pandas categorical")

        # convert pandas invalid bytes
        pd_c = pd.Categorical.from_codes([-1, 0, 1, 2], ['a', 'b', 'c'])
        pd_c = Categorical(pd_c)
        cat_list = pd_c.category_array
        assert len(cat_list) == 3
        no_negative = bool(np.all(pd_c.view(FastArray) >= 0))
        assert no_negative

        # convert pandas invalid unicode
        pd_c = pd.Categorical.from_codes([-1, 0, 1, 2], [u'\u2082', u'\u2083', u'\u2084'])
        pd_c = Categorical(pd_c)
        cat_list = pd_c.category_array
        assert len(cat_list) == 3
        no_negative = bool(np.all(pd_c.view(FastArray) >= 0))
        assert no_negative

    # --------------------------------RIPTABLE CATEGORICAL ----------------------------------------
    # def test_ctor_rt_cat(self):
    #    c_unicode = Categorical(list_unicode)
    #    c = c_unicode.copy(forceunicode=True)
    #    self.assertEqual(c._categories_wrap._list.dtype.char, 'U', msg=f"Failed to force unicode on categorical copy.")

    # ------------------------------------CUSTOM CATEGORIES ----------------------------------
    def test_ctor_list_unique(self):
        unique_str = ['a', 'b', 'c', 'd', 'e', 'f']
        str_list = ['a', 'b', 'c', 'd', 'e']
        c = Categorical(str_list, unique_str)
        cats_match = bool(np.all(c._categories_wrap._list == unique_str))
        assert cats_match, f"Failed to create matching categories from unique category input."

    # ------------------------------------INTEGER ARRAY ----------------------------------
    def test_ctor_integer_array(self):
        lis = [1, 4, 9, 16, 25]
        c = Categorical(lis)
        for v1, v2 in zip(c, lis):
            assert v1 == v2

    # ------------------------------------GARBAGE ----------------------------------
    def test_ctor_garbage(self):
        with pytest.raises(TypeError):
            c = Categorical(1, 2)

    # ------------------------------------TEST FORCE DTYPE ----------------------------------
    def test_init_with_dtype(self):
        int_types = [np.int8, np.int16, np.int32, np.int64]
        float_types = [np.float32, np.float64]
        uint_types = [np.uint8, np.uint16, np.uint32, np.uint64]
        arr = ['a', 'b', 'c', 'd', 'e']
        for dt in int_types:
            c = Categorical(arr, dtype=dt)
            assert c.dtype == dt, f"Failed to force the correct dtype {dt} for categorical."

        for dt in float_types + uint_types:
            with pytest.raises(TypeError):
                c = Categorical(arr, dtype=dt)

    # ------------------------------------TEST CONVERT VALUE-------------------------------------
    def test_possibly_convert_value(self):
        '''
        TODO: fix for new Categories class
        '''
        self._notimpl()

    def test_categories_bad_init(self):
        tup = ('a', 'b', 'c')
        with pytest.raises(TypeError):
            cat = Categories(tup)

    def test_categories_len(self):
        cats_from_list = Categorical(['a', 'b', 'c'], ordered=True, base_index=1, filter=None)._categories_wrap
        assert len(cats_from_list) == 3

        cats_from_enum = Categorical(FastArray([144]), LikertDecision)._categories_wrap
        assert len(cats_from_enum) == 144

    def test_get_categories(self):
        c_list = [
            'StronglyAgree',
            'Agree',
            'Disagree',
            'StronglyDisagree',
            'NeitherAgreeNorDisagree',
        ]
        cats_from_list = Categories(c_list, unicode=True)
        cats_from_enum = Categories(LikertDecision)

        get_cats_match = bool(np.all(cats_from_list.get_categories() == cats_from_enum.get_categories()))
        assert get_cats_match

    def test_possibly_add_categories(self):
        self._notimpl()
        # uniquify and sort
        # raise exception for adding cats to intenum, etc.

    def test_categories_preserves_subtype(self):
        # Test the Categorical.categories() method preserves the array type for the category data.
        # This is important because we want the array(s) returned by this method to have the same type
        # as the internal data (i.e. what's returned by Categorical.category_array or Categorical.category_dict).

        # Single-key Categorical
        dates = rt.Date(
            [
                '2019-03-15',
                '2019-04-18',
                '2019-05-17',
                '2019-06-21',
                '2019-07-19',
                '2019-08-16',
                '2019-09-20',
                '2019-10-18',
                '2019-11-15',
                '2019-12-20',
            ]
        )
        dates.name = 'dates'
        dates_cat = rt.Cat(dates)
        cats = dates_cat.categories()
        assert type(dates) == type(cats)

        # Multi-key Categorical
        datestrs = rt.FA(
            [
                '2019-03-15',
                '2019-04-18',
                '2019-05-17',
                '2019-06-21',
                '2019-07-19',
                '2019-08-16',
                '2019-09-20',
                '2019-10-18',
                '2019-11-15',
                '2019-12-20',
            ]
        )
        datestrs.name = 'datestrs'
        mcat = rt.Cat([dates, datestrs])
        mcats = mcat.categories()
        assert type(mcats['key_0']) == type(dates)
        assert type(mcats['key_1']) == type(datestrs)

        # Empty single-key Categorical
        dates = rt.Date([])
        dates_cat = rt.Cat(dates)
        cats = dates_cat.categories()
        assert type(dates) == type(cats)

    def test_make_unique(self):
        # SJK: changed this test on 8/21/2018 - count now comes from the grouping object, not Categories.make unique
        values = FastArray(['a', 'b', 'c', 'c', 'd', 'a', 'b'])
        # c = Categories([],base_index=1)
        # index, cat_len, filter = c.make_unique(values)

        cat = Categorical(values, ordered=True, base_index=1, filter=None)
        index = cat._fa
        c = cat._categories_wrap

        assert len(index) == 7
        assert max(index) == 4
        assert c._mode == CategoryMode.StringArray
        assert c._list.dtype.char == 'S'
        assert c.isbytes

        univals = values.astype('U')
        cat = Categorical(univals, ordered=True, base_index=1, filter=None, unicode=True)
        index = cat._fa
        c = cat._categories_wrap

        assert len(index) == 7
        assert max(index) == 4
        assert c._mode == CategoryMode.StringArray
        assert c._list.dtype.char == 'U'
        assert c.isunicode

    @pytest.mark.xfail(
        reason='20200416 This test was previously overridden by a later test in the file with the same name. Need to revisit and get back in a working state.'
    )
    def test_force_base_index(self):
        filter = FastArray([True, True, False, False, True])

        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        assert c.base_index == 1, 'Did not default base index to 1'
        assert c._fa[0] == 1, 'Did not default base index to 1'

        c = Categorical(['a', 'a', 'b', 'c', 'a'], base_index=0)
        assert c.base_index == 0, 'Did not force base index to 0'
        assert c._fa[0] == 0, 'Did not force base index to 0'

        c = Categorical(['a', 'a', 'b', 'c', 'a'], filter=filter)
        assert len(c.category_array) == 1
        assert c._fa[2] == 0, 'Did not default base index to 1'

        c = Categorical(['a', 'a', 'b', 'c', 'a'], base_index=0, filter=filter)
        assert len(c.category_array) == 1
        assert c._fa[2] == INVALID_DICT[c.dtype.num], 'Did not force base index to 0'

        with pytest.raises(ValueError):
            c = Categorical(['a', 'a', 'b', 'c', 'a'], base_index=99, filter=filter)

        c = Categorical(['a', 'a', 'b', 'c', 'a'], ['a', 'b', 'c'])
        assert c.base_index == 1, 'Did not default base index to 1'
        assert c._fa[0] == 1, 'Did not default base index to 1'

        c = Categorical(['a', 'a', 'b', 'c', 'a'], ['a', 'b', 'c'], base_index=0)
        assert c.base_index == 0, 'Did not force base index to 0'
        assert c._fa[0] == 0, 'Did not force base index to 0'

        with pytest.raises(NotImplementedError):
            c = Categorical(['a', 'a', 'b', 'c', 'a'], ['a', 'b', 'c'], base_index=0, filter=filter)

        with pytest.raises(ValueError):
            c = Categorical([1.0, 2.0, 3.0], ['a', 'b', 'c'], from_matlab=True, base_index=0)

        pdc = pd.Categorical(['a', 'a', 'b', 'c', 'a'])
        with pytest.raises(ValueError):
            c = Categorical(pdc, base_index=0)

    def test_is_in_unique_strings(self):
        values = ['a', 'b', 'c', 'c', 'd', 'a', 'b']
        good_cats = ['a', 'b', 'c', 'd']
        incomplete_cats = ['a', 'b', 'c']
        bad_cats = ['a', 'a', 'b']
        invalid = 'invalid'

        ###--------REMOVED from_provided_categories, rewrite these tests to go through main constructor

        # valid bytes
        c = Categorical(values, good_cats, ordered=True, base_index=1, unicode=False, filter=None)
        cats = c._categories_wrap
        assert len(c) == 7
        assert max(c._fa) == 4
        assert cats._mode == CategoryMode.StringArray
        assert cats._list.dtype.char == 'S'
        assert cats.isbytes

        # valid unicode
        c = Categorical(values, good_cats, ordered=True, base_index=1, unicode=True, filter=None)
        cats = c._categories_wrap
        assert len(c) == 7
        assert max(c._fa) == 4
        assert cats._mode == CategoryMode.StringArray
        assert cats._list.dtype.char == 'U'
        assert cats.isunicode

        # non-unique categories
        # 4/12/2019 - no longer checks for uniqueness
        # with self.assertRaises(ValueError):
        #    c = Categories.from_provided_categories(values, bad_cats, ordered=True, base_index=1, unicode=False, filter=None)

        # not all values found in categories
        with pytest.raises(ValueError):
            c = Categorical(values, incomplete_cats, ordered=True, base_index=1, unicode=False, filter=None,)

        # insert invalid True
        # 5/16/2019 invalid must appear in provided uniques
        with pytest.raises(ValueError):
            c = Categorical(
                values, incomplete_cats, ordered=True, base_index=1, unicode=True, filter=None, invalid=invalid,
            )
            cats = c._categories_wrap
            assert len(c) == 7
            assert max(c._fa) == 3
            assert cats._mode == CategoryMode.StringArray
            assert cats._list.dtype.char == 'U'
            assert cats.isunicode

    def test_getitem_enum_str(self):
        codes = [1, 44, 44, 133, 75]
        correct = [True, False, False, False, False]
        valid_str = 'StronglyDisagree'
        invalid_str = 'q'
        c = Categorical(codes, LikertDecision)

        # with self.assertRaises(IndexError):
        mask = c[valid_str]
        is_correct = bool(np.all(mask == correct))
        assert is_correct

        with pytest.raises(ValueError):
            mask = c[invalid_str]
            assert sum(mask) == 0

    def test_match_str_to_category(self):
        single_byte = b'a'
        single_unicode = 'a'
        single_true_unicode = u'\u2082'

        byte_values = [b'a', b'b', b'c', b'c', b'd', b'a', b'b']
        values = FastArray(['a', 'b', 'c', 'c', 'd', 'a', 'b'])
        true_unicode = [u'\u2082', u'\u2083', u'\u2082']

        # 4/25/2019 - changed these tests to construct a Categorical, rather than
        # a Categories object directly. Categorical will always make a Categories object.
        # (held in _categories_wrap)
        c = Categorical(values, ordered=True, base_index=1, filter=None)
        matching_char = c._categories_wrap.match_str_to_category(single_unicode)
        assert isinstance(matching_char, bytes)
        with pytest.raises(TypeError):
            matching = c._categories_wrap.match_str_to_category(single_true_unicode)

        univals = np.array(['a', 'b', 'c', 'c', 'd', 'a', 'b'])
        c = Categorical(univals, ordered=True, base_index=1, filter=None, unicode=True)
        matching_char = c._categories_wrap.match_str_to_category(single_byte)
        assert isinstance(matching_char, str)

        c = Categorical(values, ordered=True, base_index=1, filter=None)
        matching = c._categories_wrap.match_str_to_category(values)
        assert matching.dtype.char == 'S'
        with pytest.raises(TypeError):
            matching = c._categories_wrap.match_str_to_category(true_unicode)

        c = Categorical(univals, ordered=True, base_index=1, filter=None, unicode=True)
        matching = c._categories_wrap.match_str_to_category(values)
        assert matching.dtype.char == 'U'

    # Categories object being removed
    # Disabling these tests - methods will move into Categorical
    # 4/24/2019
    # def test_get_category_index(self):
    #    values = FastArray(['a', 'b', 'c', 'c', 'd', 'a', 'b', 'g'])
    #    _, c, _, _ = Categories.from_array(values, ordered=True, base_index=1, filter=None)

    #    # when found, will return exact index
    #    str_idx = c.get_category_index('b')
    #    self.assertEqual(str_idx, 2)

    #    # when ordered, will return floating point for LTE GTE
    #    str_idx = c.get_category_index('e')
    #    self.assertEqual(str_idx, 4.5)

    #    # when unordered, will return invalid index (length of string array)
    #    c._sorted = False
    #    str_idx = c.get_category_index('e')
    #    self.assertEqual(str_idx, 6)

    # def test_get_category_match_index(self):
    #    values = FastArray(['a', 'b', 'c', 'c', 'd', 'a', 'b', 'g'])
    #    _, c, _, _ = Categories.from_array(values, ordered=False, base_index=1, filter=None)

    #    string_matches = c.get_category_match_index(['a','b'])
    #    self.assertEqual(string_matches, [1,2])

    #    c._mode = CategoryMode.IntEnum
    #    with self.assertRaises(NotImplementedError):
    #        string_matches = c.get_category_match_index(['a','b'])

    def test_possibly_invalid(self):
        values = ['a', 'b', 'c', 'c', 'd', 'a', 'b', 'g']
        c = Categorical(values, base_index=1)
        out_of_range = -50
        sentinel = INVALID_DICT[c.dtype.num]

        c.view(FastArray)[0] = out_of_range
        # c.view(FastArray)[1] = sentinel
        # **changed invalid, all will display as bad code if changed underneath and not in range
        assert c[0] == "!<-50>"
        # self.assertEqual(c[1], "!<inv>")

    def test_categories_getitem_str_list(self):
        codes = [1, 44, 44, 133, 75]
        correct = FastArray([False, True, True, False, True])
        c = Categorical(codes, LikertDecision)

        mask = c[['StronglyAgree', 'Disagree']]
        is_correct = bool(np.all(mask == correct))
        assert is_correct

        mask = c[[b'StronglyAgree', b'Disagree']]
        is_correct = bool(np.all(mask == correct))
        assert is_correct

    def test_categories_print_repr(self):
        self._notimpl()

    def test_enum_dict_warning(self):
        class DupeEnum(IntEnum):
            code_a = 1
            code_b = 1
            code_c = 1
            code_d = 2

        with pytest.warns(UserWarning):
            c = Categorical([1, 2], DupeEnum)

    # ------------------------- TEST MERGE -------------------------------------------
    # def test_merge(self):
    #    from riptable.rt_categorical import categorical_merge
    #    c_bytes = Categorical(['b','b','b','a','b','b'], ['a','b'])
    #    c_unicode = Categorical(["AAPL\u2080","AMZN\u2082"])
    #    result = categorical_merge([c_bytes, c_unicode])
    #    # self.assertTrue(result[0]._categories_wrap._list is result[1]._categories_wrap._list, msg=f"Categorical merge did not assign the same dictionary to both arrays.")
    #    self.assertEqual(result[0]._categories_wrap._list.dtype.char, 'U', msg=f"{result[0]._categories_wrap._list.dtype.char} was not 'U'. dictionary was not flipped to unicode.")
    #    for item in c_bytes._categories_wrap._list:
    #        self.assertTrue(item.decode() in result[0]._categories_wrap._list, msg=f"{item} did not appear in final categories")
    #    for item in c_unicode._categories_wrap._list:
    #        self.assertTrue(item in result[0]._categories_wrap._list, msg=f"{item} did not appear in final categories")
    #    c1 = Categorical([1, 1, 3, 2, 2], [1, 2, 3, 4, 5], from_matlab=True)
    #    c2 = Categorical([2, 2, 4, 4, 3], [1, 2, 3, 4, 5], from_matlab=True)
    #    [cm1, cm2] = categorical_merge([c1, c2])
    #    self.assertTrue((cm1 == [1, 1, 3, 2, 2]).all())
    #    self.assertTrue((cm2 == [2, 2, 4, 4, 3]).all())

    # ------------------------- TEST HSTACK -------------------------------------------
    def test_hstack(self):
        c1 = Categorical(['a', 'a', 'c', 'b', 'b'])
        c2 = Categorical(['b', 'b', 'd', 'd', 'c'])
        cm = Categorical.hstack([c1, c2])
        assert (cm.as_string_array == ['a', 'a', 'c', 'b', 'b', 'b', 'b', 'd', 'd', 'c']).all()
        c1 = Categorical([1, 1, 3, 2, 2], [1, 2, 3, 4, 5], from_matlab=True)
        c2 = Categorical([2, 2, 4, 4, 3], [1, 2, 3, 4, 5], from_matlab=True)
        cm = Categorical.hstack([c1, c2])
        assert (cm == [1, 1, 3, 2, 2, 2, 2, 4, 4, 3]).all()

    def test_hstack_fails_for_different_mode_cats(self):
        # Create a dictionary-mode Categorical (from ISO3166 data).
        # The dictionary is created manually below instead of using e.g.
        #   {k: int(v) for (k, v) in ISOCountryCode.__members__.items()}
        # so the dictionary we give to Categorical does not have the insert ordering
        # imply an ordering of the keys/values.
        country_code_dict = {
            'IRL': 372, 'USA': 840, 'AUS': 36, 'HKG': 344, 'JPN': 392,
            'MEX': 484, 'KHM': 116, 'THA': 764, 'JAM': 388, 'ARM': 51
        }

        # The values for the Categorical's backing array.
        # This includes some value(s) not in the dictionary and not all values in the dictionary are used here.
        country_num_codes = [36, 36, 344, 840, 840, 372, 840, 372, 840, 124, 840, 124, 36, 484]

        cat1 = rt.Categorical(country_num_codes, country_code_dict)
        assert cat1.category_mode == CategoryMode.Dictionary

        # Create a single-key, string-mode Categorical.
        cat2 = rt.Categorical(['AUS', 'AUS', 'HKG', 'USA', 'USA', 'IRL', 'USA', 'IRL', 'USA', 'KHM', 'IRL', 'AUS', 'MEX'])
        assert cat2.category_mode != CategoryMode.Dictionary

        # Try to hstack the two Categoricals. This should fail due to the CategoryMode values being different.
        with pytest.raises((ValueError, TypeError)):
            rt.hstack([cat1, cat2])

    def test_align(self):
        c1 = Categorical(['a', 'b', 'c'])
        c2 = Categorical(['d', 'e', 'f'])
        c3 = Categorical(['c', 'f', 'z'])
        cm = Categorical.align([c1, c2, c3])
        assert (cm[0].as_string_array == ['a', 'b', 'c']).all()
        assert (cm[1].as_string_array == ['d', 'e', 'f']).all()
        assert (cm[2].as_string_array == ['c', 'f', 'z']).all()
        assert (cm[0].categories() == FastArray([b'Filtered', b'a', b'b', b'c', b'd', b'e', b'f', b'z'])).all()
        assert (cm[0].categories() == cm[1].categories()).all()
        assert (cm[0].categories() == cm[2].categories()).all()
        c1 = Categorical([1, 1, 3, 2, 2], [1, 2, 3, 4, 5], from_matlab=True)
        c2 = Categorical([2, 2, 4, 4, 3], [1, 2, 3, 4, 5], from_matlab=True)
        cm = Categorical.align([c1, c2])
        assert (cm[0] == [1, 1, 3, 2, 2]).all()
        assert (cm[1] == [2, 2, 4, 4, 3]).all()

    def test_categorical_merge_dict(self):
        from riptable.rt_categorical import categorical_merge_dict

        d1 = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
        d2 = {'a': 1, 'e': 5, 'b': 2, 'f': 6}
        c1 = Categorical([3, 3, 4, 3, 1, 2, 5], d1)
        c2 = Categorical([1, 1, 5, 2, 2, 1, 5], d2)

        combined = categorical_merge_dict([c1, c2], return_type=dict)
        for i in range(1, 6):
            assert i in combined.values()

    def test_getitem_empty(self):
        c = Categorical([0, 1, 2], ['a', 'b', 'c'])

        empty_list = c[[]]
        assert isinstance(empty_list, Categorical)
        dict_matches = bool(np.all(empty_list.categories() == c.categories()))
        assert dict_matches

        with pytest.raises(IndexError):
            empty_np = c[np.array([])]
            assert isinstance(empty_np, Categorical)
            dict_matches = bool(np.all(empty_np.categories() == c.categories()))
            assert dict_matches

    def test_iter_groups(self):
        correct_keys = FastArray(['a', 'b', 'c', 'd', 'e'])
        correct_idx = [[8], [3], [5, 6], [2, 9], [0, 1, 4, 7]]
        str_arr = FastArray(['e', 'e', 'd', 'b', 'e', 'c', 'c', 'e', 'a', 'd'])
        c = Categorical(str_arr)
        for i, tup in enumerate(c.iter_groups()):
            assert tup[0] == correct_keys[i]
            assert bool(np.all(tup[1] == correct_idx[i]))

    def test_enum_dict_multi(self):
        self._notimpl()
        # not implemented

    def test_enum_init_errors(self):
        with pytest.raises(TypeError):
            c = Categorical(['a', 'b', 'c'], LikertDecision)

    def test_custom_invalid_category(self):
        # 5/16/2019 invalid must appear in provided uniques
        c = Categorical(
            ['a', 'b', 'c', 'my_invalid'], ['a', 'b', 'c', 'my_invalid'], invalid='my_invalid', base_index=1,
        )
        assert c[3] == 'my_invalid'
        assert c.isnan()[3]
        assert len(c.category_array) == 4

    @pytest.mark.xfail(reason="After invalid_set, the custom invalid value is not displayed.")
    def test_invalid_set(self):
        c = Categorical(
            ['a', 'b', 'c', 'my_invalid'], ['a', 'b', 'c', 'my_invalid'], invalid='my_invalid', base_index=1,
        )
        # set a new string to be displayed for invalid items and validate
        custom_invalid = "custom_invalid"
        c.invalid_set(custom_invalid)

        assert c[3] == custom_invalid
        assert c.isnan()[3]
        assert len(c.category_array) == 4

    def test_lock_unlock(self):
        self._notimpl()
        # halfway implemented

    def test_set_item(self):
        self._notimpl()
        # when index needs to be fixed after categories are added
        # setitem with integer / invalid integer
        # setitem with string / invalid category

    def test_return_empty_cat(self):
        self._notimpl()
        # this code still needs to get written

    def test_getitem_np_str(self):
        c = Categorical(['a', 'a', 'b', 'a', 'c', 'c', 'b'])
        correct = FastArray([True, True, True, True, False, False, True])
        with pytest.raises(IndexError):
            result = c[np.array(['a', 'b'])]
        # self.assertTrue(array_equal(result, correct), msg=f"incorrect getitem result when indexing by numpy array of strings")
        with pytest.raises(IndexError):
            result = c[np.array(['a', 'b']).astype('S')]
        # self.assertTrue(array_equal(result, correct), msg=f"incorrect getitem result when indexing by numpy array of strings")

    def test_getitem_slice(self):
        c = Categorical(['a', 'a', 'b', 'a', 'c', 'c', 'b'])
        result = c[:3]
        assert isinstance(result, Categorical)
        match_fa = bool(np.all(result.view(FastArray) == [1, 1, 2]))
        assert match_fa
        assert len(result) == 3
        assert len(result._categories_wrap) == 3

    def test_categorical_compare_check(self):
        self._notimpl()
        # Categories have different modes
        # categories are both enum
        # compare cat to empty list

        # non-categorical input
        # convert all to unicode if one is unicode

    # this keyword wasn't used anywhere, removed from copy()
    # def test_copy_invalid(self):
    #    c = Categorical(['a','a','b','a','c','c','b'])
    #    invalid_copy = c.copy(fill_invalid=True)
    #    all_invalid = bool(np.all(invalid_copy.view(FastArray)==-128))
    #    self.assertTrue(all_invalid)

    #    for idx, item in enumerate(c.categories()):
    #        self.assertEqual(item, invalid_copy.categories()[idx])

    #    self.assertFalse(c.categories() is invalid_copy.categories())

    def test_fill_invalid(self):
        values = list('aabaccb')
        c = Categorical(values, base_index=1)
        c.fill_invalid(inplace=True)

        assert_array_equal(FastArray([c.filtered_name] * len(values)), c.expand_array)
        assert_array_equal(FastArray([0] * len(values)), c._fa)
        expected = FastArray(sorted(set(values))).astype('|S1')

        assert_array_equal(expected, c.category_array)
        assert_array_equal(expected, c.category_dict[next(iter(c.category_dict))])  # values of first key

    def test_force_unicode(self):
        c = Categorical(['a', 'a', 'b', 'a', 'c', 'c', 'b'], unicode=True)
        result_dtype = c.categories().dtype.char
        assert result_dtype == 'U', f"Failed to force unicode when constructing categorical from list of string values"

    def test_categories_shallow_copy(self):
        codes = [10, 10, 20, 10, 30, 20, 10]
        d = {10: 'a', 20: 'b', 30: 'c'}
        c = Categorical(codes, d)
        original_cats = c._categories_wrap
        new_cats = original_cats.copy(deep=False)
        assert (
            original_cats._str_to_int_dict is new_cats._str_to_int_dict
        ), f"Categories did not use same str_to_int dictionary after shallow copy."
        assert (
            original_cats._int_to_str_dict is new_cats._int_to_str_dict
        ), f"Categories did not use same int_to_str dictionary after shallow copy."

    # 5/16/2019 invalid category must be in user provided
    # def test_two_lists_invalid(self):
    #    c = Categorical(['a','a','b','a','c','c','b'],np.array(['a','b']), invalid='inv', base_index=1)
    #    self.assertEqual(c[4],FILTERED_LONG_NAME)

    @pytest.mark.xfail(
        reason='20200416 This test was previously overridden by a later test in the file with the same name. Need to revisit and get back in a working state.'
    )
    def test_getitem_enum_list(self):
        c = Categorical([44, 133, 133, 75, 144, 1], LikertDecision)
        with pytest.raises(IndexError):
            result = c[[b'NeitherAgreeNorDisagree']]
        correct = FastArray([False, False, False, False, True, False])
        # self.assertTrue(array_equal(result, correct))

        result = c[[4]]
        assert result[0] == 'NeitherAgreeNorDisagree'

    def test_non_unique(self):
        with pytest.raises(ValueError):
            c = Categorical(['a', 'a', 'b', 'a', 'c', 'c', 'b'], ['a', 'a', 'b'])

    def test_match_to_category(self):
        c = Categorical(['a', 'a', 'b', 'a', 'c', 'c', 'b'])
        result = c._categories_wrap.match_str_to_category('a')
        assert b'a' == result

        with pytest.raises(TypeError):
            result = c._categories_wrap.match_str_to_category([1, 2, 3])

        with pytest.raises(TypeError):
            result = c._categories_wrap.match_str_to_category({1, 2, 3})

        c1 = Categorical(['abc', 'def', 'abc', 'abc'], np.array(['abc', 'def']), unicode=True)
        result = c1._categories_wrap.match_str_to_category([b'a'])
        assert result.dtype.char == 'U'

    # ------------------------------------TEST SET ITEM------------------------------------------
    def test_set_item_str_index(self):
        c = Categorical(['b', 'b', 'b', 'a', 'b', 'b'], ['a', 'b'])
        correct = [2, 2, 2, 2, 2, 2]
        c['a'] = 'b'
        is_correct = bool(np.all(c.view(FastArray) == correct))
        assert is_correct, f"Category was not correctly changed with set item on a string."
        with pytest.raises(ValueError):
            c['b'] = 'c'

    def test_set_item_int_index(self):
        c = Categorical(['b', 'b', 'b', 'a', 'b', 'b'], ['a', 'b'])
        correct = [1, 2, 2, 1, 2, 2]
        c[0] = 'a'
        is_correct = bool(np.all(c.view(FastArray) == correct))
        assert is_correct, f"Category was not correctly changed with set item on an int."
        with pytest.raises(ValueError):
            c[0] = 'c'

    # ------------------------------------TEST CALCULATE DTYPE ----------------------------------
    def test_get_dtype_from_len(self):
        '''
        Categorical will select different types
        '''

        dtype_sizes = {
            np.int8: 1,
            np.int16: 101,
            np.int32: 50001,
        }  # , np.int64:2000000001 }
        for dt, sz in dtype_sizes.items():
            LENGTH = 6
            NO_CODES = sz
            alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            np_alphabet = np.array(alphabet, dtype="|U1")
            np_codes = np.random.choice(np_alphabet, [NO_CODES, LENGTH])
            codes = ["".join(np_codes[i]) for i in range(len(np_codes))]
            c = Categorical(["".join(np_codes[i]) for i in range(len(np_codes))])
            # only perform the test if there are enough uniques
            if len(c._categories_wrap._list) >= sz:
                assert c.dtype == dt, f"Categorical did not set dtype to {dt} for array of size {sz}."

    # -------SINGLE INTEGER
    def test_getitem_int(self):
        '''
        Single integer index should return the corresponding category in unicode format.
        '''
        c = Categorical(['b', 'b', 'b', 'a', 'b', 'b'], ['a', 'b'])
        assert c[0] == 'b', f"Get item with integer did not return the correct category."
        assert isinstance(c[0], str), f"Get item with integer did not return as unicode."
        assert c[3] == 'a', f"Get item with integer did not return the correct category."
        assert isinstance(c[3], str), f"Get item with integer did not return as unicode."
        with pytest.raises(IndexError):
            d = c[10]

    # --------INTEGER MASK
    def test_getitem_int_mask(self):
        py_mask = [0, 3]
        c = Categorical(['b', 'b', 'b', 'a', 'b', 'b'], ['a', 'b'])

        for mask in [py_mask, np.array(py_mask)]:
            d = c[mask]
            assert isinstance(
                d, Categorical
            ), f"Get item with integer mask did not return a categorical. Returned {type(d).__name__} instead."
            assert len(d) == len(
                mask
            ), f"Get item with integer mask did not return categorical of {len(mask)}. returned {len(d)} instead."
            has_same_cats = bool(np.all(d._categories_wrap._list == c._categories_wrap._list))
            assert (
                has_same_cats
            ), f"Failed to copy the same categories to new categorical after getitem with integer mask."
            d = c[[0, 10]]
            assert d._fa[1] == 0, f"Failed to put invalid for out of range index."

    # -------BOOLEAN MASK
    def test_getitem_bool_mask(self):
        py_mask = [True, True, True, False, True, True]
        c = Categorical(['b', 'b', 'b', 'a', 'b', 'b'], ['a', 'b'])

        for mask in [py_mask, np.array(py_mask)]:
            d = c[mask]
            assert not (
                b'a' in d.as_string_array
            ), f"b'a' does not get trimmed out of categorical with getitem from boolean array."
            assert 5 == len(
                d
            ), f"Length {len(d)} did not match 5 in categorical getitem with a boolean array of the same size."
            has_same_cats = bool(np.all(d._categories_wrap._list == c._categories_wrap._list))
            assert (
                has_same_cats
            ), f"Failed to copy the same categories to new categorical after getitem with integer mask."

    # -------SINGLE STRING
    def test_getitem_single_string(self):
        b_result = [True, True, True, False, True, True]
        c = Categorical(['b', 'b', 'b', 'a', 'b', 'b'], ['a', 'b'])
        idx = b'c'
        # with self.assertRaises(IndexError):
        d = c[idx]
        has_true = bool(np.any(d))
        assert not has_true, f"Failed to return an array of all false for getitem with {idx}"
        assert isinstance(d, FastArray), f"Get item input {idx} did not return FastArray"
        assert d.dtype.char == '?', f"Get item input {idx} did not return FastArray"
        idx = idx.decode()
        # with self.assertRaises(IndexError):
        d = c[idx]
        has_true = bool(np.any(d))
        assert not has_true, f"Failed to return an array of all false for getitem with {idx}"
        assert isinstance(d, FastArray), f"Get item input {idx} did not return FastArray"
        assert d.dtype.char == '?', f"Get item input {idx} did not return FastArray"

        idx = b'b'
        # with self.assertRaises(IndexError):
        d = c[idx]
        is_correct = bool(np.all(d == b_result))
        assert is_correct, f"Did not return the correct array for getitem with {idx}"
        assert isinstance(d, FastArray), f"Get item input {idx} did not return FastArray"
        assert d.dtype.char == '?', f"Get item input {idx} did not return FastArray"
        idx = idx.decode()
        # with self.assertRaises(IndexError):
        d = c[idx]
        is_correct = bool(np.all(d == b_result))
        assert is_correct, f"Did not return the correct array for getitem with {idx}"
        assert isinstance(d, FastArray), f"Get item input {idx} did not return FastArray"
        assert d.dtype.char == '?', f"Get item input {idx} did not return FastArray"

    # ------MULTIPLE STRINGS
    def test_getitem_multiple_strings(self):
        c = Categorical(['b', 'b', 'b', 'a', 'b', 'b'])
        inputs = {
            (b'b',): [True, True, True, False, True, True],  # single in (list)
            (b'c',): [False, False, False, False, False, False],  # single not in (list)
            (b'a', b'b'): [True, True, True, True, True, True],  # both in (list)
            (b'c', b'd'): [False, False, False, False, False, False,],  # both not in (list)
            (b'b', b'c'): [True, True, True, False, True, True],  # mixed (list)
        }
        for idx, correct in inputs.items():
            idx = list(idx)

            d = c[idx]
            is_correct = bool(np.all(d == correct))
            assert is_correct, f"Indexing categorical {c} by {idx} did not return the correct result."
            assert d.dtype.char == '?', f"Get item input {idx} did not return FastArray"

            idx = [b.decode() for b in idx]
            d = c[idx]
            is_correct = bool(np.all(d == correct))
            assert is_correct, f"Indexing categorical {c} by {idx} did not return the correct result."
            assert d.dtype.char == '?', f"Get item input {idx} did not return FastArray"

    # ------NUMERIC GETITEM
    def test_getitem_numeric_categories(self):
        # before it was fixed, a bug was returning a string of the numeric category
        nums = np.array([1, 1, 2, 3, 4, 5, 1, 1, 1])
        c = Categorical(nums)
        assert c[0] == 1
        assert isinstance(c[0], (int, np.integer))
        nums = nums.astype(np.float32)
        c = Categorical(nums)
        assert c[0] == 1.0
        assert isinstance(c[0], (float, np.floating)), f"Expected float, got {type(c[0])}"

    # ------------------------- TEST COMPARE CHECK -------------------------------------------
    def test_compare_check(self):
        '''
        Test comparison between two 'equal' categoricals with different underlying arrays.
        '''
        compare_ops = {
            '__ne__': [False, False, False, False, False, False],
            '__eq__': [True, True, True, True, True, True],
            '__ge__': [True, True, True, True, True, True],
            '__gt__': [False, False, False, False, False, False],
            '__le__': [True, True, True, True, True, True],
            '__lt__': [False, False, False, False, False, False],
        }
        c = Categorical(['b', 'b', 'b', 'a', 'b', 'b'], ['a', 'b', 'c'])
        d = Categorical(['b', 'b', 'b', 'a', 'b', 'b'], ['a', 'b'])
        for name, correct in compare_ops.items():
            func = c.__getattribute__(name)
            result = func(d)
            is_correct = bool(np.all(result == correct))
            assert is_correct, f"Compare operation betweeen two equal categoricals did not return the correct result."

    def test_compare_return_type(self):
        '''
        Test comparison operations with single strings to make sure FastArray of boolean is returned.
        '''
        c = Categorical(['b', 'b', 'b', 'a', 'b', 'b'], ['a', 'b'])
        scalars = ['a', 'c']
        compare_ops = ['__ne__', '__eq__', '__ge__', '__gt__', '__le__', '__lt__']
        for s in scalars:
            for op in compare_ops:
                func = c.__getattribute__(op)
                result = func(s)
                assert isinstance(result, FastArray), f"comparison {op} with input {s} did not return FastArray"
                assert result.dtype.char == '?', f"comparison {op} with input {s} did not return boolean"

    def test_compare_different_modes(self):
        c1 = Categorical(['b', 'b', 'b', 'a', 'b', 'b'], ['a', 'b'])
        c2 = Categorical([0, 1], {0: 'a', 1: 'b'})
        with pytest.raises(TypeError):
            c1 == c2

    def test_compare_conflicting_dicts(self):
        c1 = Categorical([0, 1], {0: 'a', 1: 'b'})
        c2 = Categorical([0, 1], {1: 'a', 0: 'b'})
        with pytest.raises(ValueError):
            c1 == c2

    def test_compare_safe_dicts(self):
        c1 = Categorical([0, 1], {0: 'a', 1: 'b'})
        c2 = Categorical([2, 1], {2: 'c', 1: 'b'})
        correct = FastArray([False, True])
        result = c1 == c2
        match = bool(np.all(correct == result))
        assert match

    def test_isnan(self):
        c = Categorical([1, 1, 3, 2, 2], ['a', 'b', 'c'], base_index=1, invalid='a')
        is_correct = [True, True, False, False, False]
        is_not_correct = [False, False, True, True, True]
        assert bool(np.all(is_correct == isnan(c)))
        assert bool(np.all(is_correct == c.isnan()))
        assert bool(np.all(is_not_correct == isnotnan(c)))
        assert bool(np.all(is_not_correct == c.isnotnan()))

    # ------------------------------------------------------
    def test_get_categories(self):
        # string list
        c = Categorical(['a', 'b', 'c', 'd', 'e'])
        catsarray = c.category_array
        assert isinstance(catsarray, np.ndarray)
        catsdict = c.category_dict
        assert isinstance(catsdict, dict)
        assert len(catsdict) == 1
        with pytest.raises(TypeError):
            catscodes = c.category_codes
        with pytest.raises(TypeError):
            catsmapping = c.category_mapping

        # numeric list
        c = Categorical(np.array([1, 2, 3, 4, 5]))
        catsarray = c.category_array
        assert isinstance(catsarray, np.ndarray)
        catsdict = c.category_dict
        assert isinstance(catsdict, dict)
        assert len(catsdict) == 1
        with pytest.raises(TypeError):
            catscodes = c.category_codes
        with pytest.raises(TypeError):
            catsmapping = c.category_mapping

        # dict/enum
        c = Categorical([1, 2, 3, 4], {1: 'a', 2: 'b', 3: 'c', 4: 'd'})
        catsarray = c.category_array
        assert isinstance(catsarray, np.ndarray)
        catsdict = c.category_dict
        assert isinstance(catsdict, dict)
        assert len(catsdict) == 1
        catscodes = c.category_codes
        assert isinstance(catscodes, np.ndarray)
        catsmapping = c.category_mapping
        assert isinstance(catsmapping, dict)

        # multikey
        c = Categorical([np.arange(5), np.random.rand(5)])
        with pytest.raises(TypeError):
            catsarray = c.category_array
        catsdict = c.category_dict
        assert isinstance(catsdict, dict)
        assert len(catsdict), 2
        with pytest.raises(TypeError):
            catscodes = c.category_codes
        with pytest.raises(TypeError):
            catsmapping = c.category_mapping

    # ------------------------------------------------------
    def test_force_base_index2(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        assert c.base_index == 1
        assert c._fa[0] == 1

        c = Categorical(['a', 'a', 'b', 'c', 'a'], base_index=0)
        assert c.base_index == 0
        assert c._fa[0] == 0

        codes = np.array([0, 0, 1, 2, 0])
        cats = np.array(['a', 'b', 'c'])
        # c = Categorical(codes, cats)
        # self.assertEqual(c.base_index, 0)
        # self.assertEqual(c._fa[0], 0)

        codes += 1
        c = Categorical(codes, cats, base_index=1)
        assert c.base_index == 1
        assert c._fa[0] == 1

        codes = codes.astype(np.float32)
        c = Categorical(codes, cats, from_matlab=True)
        assert c.base_index == 1
        assert c._fa[0] == 1

        with pytest.raises(ValueError):
            c = Categorical(codes, cats, from_matlab=True, base_index=0)

        c = Categorical(np.array(['a', 'a', 'b', 'c', 'a']), np.array(['a', 'b', 'c']))
        assert c.base_index == 1
        assert c._fa[0] == 1

        c = Categorical(np.array(['a', 'a', 'b', 'c', 'a']), np.array(['a', 'b', 'c']), base_index=0)
        assert c.base_index == 0
        assert c._fa[0] == 0

    # ------------------------------------------------------
    def test_ordered(self):
        c = Categorical(['c', 'c', 'a', 'b', 'c'])
        cats = c.category_array
        assert cats[0] == b'a'

        c = Categorical(['c', 'c', 'a', 'b', 'c'], ordered=False)
        cats = c.category_array
        assert cats[0] == b'c'

        c = Categorical(['c', 'c', 'a', 'b', 'c'], ['c', 'a', 'b'])
        cats = c.category_array
        assert cats[0] == b'c'
        assert not c.ordered

        c = Categorical(['c', 'c', 'a', 'b', 'c'], ['a', 'b', 'c'])
        assert c.ordered

        ## removed this test - side-effect of search sorted with unsorted array (not categorical related)
        ## false claim that categories are ordered in keyword
        # c = Categorical(['c','c','a','c','c'], ['c','a','b'], ordered=True)
        # self.assertTrue(bool(np.all(c!='c')))
        # self.assertTrue(bool(np.all(c!=b'c')))

        c = Categorical(['c', 'c', 'a', 'b', 'c'], ['c', 'a', 'b'], ordered=False)
        cats = c.category_array
        assert cats[0] == b'c'
        assert not c.ordered

        codes = FastArray([0, 0, 1, 2, 0])
        cats = FastArray(['c', 'b', 'a'], unicode=True)
        c = Categorical(codes, cats)
        assert c.category_array[0] == 'c'
        assert not c.ordered

        # with self.assertWarns(UserWarning):
        #    c = Categorical(codes, cats, ordered=True)
        #    self.assertEqual(c.category_array[0], b'c')
        #    self.assertFalse(c.ordered)

    # ------------------------------------------------------
    def test_keywords_not_allowed(self):

        # filter + base index 0
        f = np.array([True, False, True])
        with pytest.raises(ValueError):
            c = Categorical(['a', 'b', 'c'], filter=f, base_index=0)

    # ------------------------------------------------------
    def test_display_properties(self):
        '''
        Categoricals take over their display properties to appear like strings (not the underlying integer array)
        (see Utils.rt_display_properties)
        '''
        c = Categorical(['b', 'b', 'b', 'a', 'b', 'b'], ['a', 'b'])
        item_format, convert_func = c.display_query_properties()
        assert item_format.length == DisplayLength.Long, f"Incorrect length for item format."
        assert item_format.justification == DisplayJustification.Left
        assert item_format.invalid == None
        assert item_format.can_have_spaces == True
        assert item_format.decoration == None
        assert item_format.color == DisplayColumnColors.Default
        assert convert_func.__name__ == 'display_convert_func'

        # this could change, right now the convert function just does a str over the item
        assert convert_func(1, item_format) == '1', f"Incorrect convert function was returned."

    # ------------------------------------------------------
    # -----MISC. COVER TESTS--------------------------------
    def test_non_array_dict_categories_ctor(self):

        with pytest.raises(TypeError):
            c = Categories(['garbage', 'list'])

    def test_too_many_args_categories_ctor(self):
        with pytest.raises(ValueError):
            c = Categories(FastArray([1]), FastArray([2]), FastArray([3]))

    def test_filter_and_invalid(self):
        c = Categorical(
            ['a', 'a', 'b', 'c', 'c'], ['c'], invalid='a', filter=FastArray([True, True, False, True, True]),
        )
        c.filtered_set_name('a')
        assert bool(np.all(c._fa == [0, 0, 0, 1, 1]))
        for i in range(3):
            assert c[i] == 'a'
        for i in range(3, 5):
            assert c[i] == 'c'

    def test_zero_base_with_invalid(self):
        with pytest.raises(ValueError):
            c = Categorical(['a', 'b', 'c'], ['b', 'c'], base_index=0)

    # removed this property from Categories 04/24/2019
    # def test_multikey_labels(self):
    #    c = Categorical([FastArray(['a','b','c']), FastArray([1,2,3])])
    #    labels = c._categories_wrap.multikey_labels
    #    self.assertTrue(isinstance(labels[0], tuple))
    #    self.assertEqual(labels[0][0],'a')

    def test_ncols_non_multikey(self):
        c = Categorical(['a', 'b', 'c'])
        assert c._categories_wrap.ncols == 1

    # now checks for single / multikey / enum, not CategoryMode
    # def test_len_undefined_mode(self):
    #    c = Categorical(['a','b','c'])
    #    c._categories_wrap._mode = CategoryMode.Default
    #    self.assertEqual(len(c._categories_wrap),0)

    def test_categories_copy_shallow(self):
        c = Categorical(['a', 'b', 'c'])
        copycat = c._categories_wrap.copy(deep=False)
        assert isinstance(copycat, Categories)

    def test_categories_copy_deep(self):
        c = Categorical([1, 2, 3], {1: 'a', 2: 'b', 3: 'c'})
        copycat = c._categories_wrap.copy(deep=False)
        assert isinstance(copycat, Categories)

        # impossible path, unless mode is forced like below. disabling 4/24/2019
        # c._categories_wrap._mode = CategoryMode.Default
        # with self.assertRaises(NotImplementedError):
        #    c = c._categories_wrap.copy()

    def test_wrap_get_categories(self):
        c = Categorical(['a', 'b', 'c'])
        arr = c._categories_wrap.get_categories()
        assert isinstance(arr, FastArray)

        c = Categorical([FastArray(['a', 'b', 'c']), FastArray([1, 2, 3])])
        d = c._categories_wrap.get_categories()
        assert isinstance(d, dict)

    def test_get_string_mode_nums(self):
        c = Categorical(np.arange(5))
        assert not c._categories_wrap.isbytes
        assert not c._categories_wrap.isunicode

    def test_pop_single_arr(self):
        c = Categorical([np.array(['a', 'b', 'c'])])
        d = Categorical(np.array(['a', 'b', 'c']))
        assert bool(np.all(c == d))

        c = Categorical({'test': np.array(['a', 'b', 'c'])})
        d = Categorical(np.array(['a', 'b', 'c']))
        assert bool(np.all(c == d))

    def test_from_cat_as_array(self):
        c = Categorical(FastArray([1, 2, 3]), _from_categorical=np.array(['a', 'b', 'c']))
        assert isinstance(c.category_array, FastArray)
        assert c.base_index == 1

    def test_from_pandas_object(self):
        pdc = pd.Categorical(['a', 'b', 'c'])
        c = Categorical(pdc, unicode=True)
        assert c.category_array.dtype.char == 'U'

        c = Categorical(pdc, unicode=False)
        assert c.category_array.dtype.char == 'S'

        pdc = pd.Categorical(three_unicode)
        c = Categorical(pdc)
        assert c.category_array.dtype.char == 'U'

    def test_empty_init(self):
        with pytest.raises(ValueError):
            c = Categorical({})

        with pytest.raises(ValueError):
            c = Categorical([])

    def test_multi_with_cats(self):
        with pytest.raises(NotImplementedError):
            c = Categorical(
                [FastArray(['a', 'b', 'c', 'a']), FastArray([1, 2, 3, 1])],
                [FastArray(['a', 'b', 'c']), FastArray([1, 2, 3])],
            )

    # 5/9/2019 removed this warning to reduce constructor paths
    # def test_unicode_warn(self):
    #    with self.assertWarns(UserWarning):
    #        c = Categorical([1,2,3],{1:'a',2:'b',3:'c'}, unicode=False)

    def test_map_non_integer(self):
        with pytest.raises(TypeError):
            c = Categorical([1.0, 2.0, 3.0], {1: 'a', 2: 'b', 3: 'c'})

    def test_category_multi_arrays(self):
        with pytest.raises(TypeError):
            c = Categorical([1, 2, 3], [np.arange(5), np.arange(5)])

    def test_getitem_enum_list2(self):
        c = Categorical([1, 1, 2, 3, 1], {'a': 1, 'b': 2, 'c': 3})
        d = c[[1, 2, 3]]
        assert d[0] == 'a'

    def test_tuple_compare_error(self):
        c = Categorical([FastArray(['a', 'b', 'c', 'a']), FastArray([1, 2, 3, 1])])
        with pytest.raises(ValueError):
            _ = c == ('a', 'b', 'c')

    def test_filter_out_bytes_from_unicode(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'], unicode=True, invalid=b'a')
        assert bool(np.all(c._fa == [1, 1, 2, 3, 1]))
        assert c.category_array.dtype.char == 'U'
        assert 'a' in c.category_array

    def test_bytes_compare_multikey(self):
        c = Categorical([np.array(['a', 'b', 'c', 'a']), FastArray([1, 2, 3, 1])], unicode=True)
        cols = c.category_dict
        bytescol = list(cols.values())[0]
        assert bytescol.dtype.char == 'U'
        result = c == (b'a', 1)
        assert bool(np.all(FastArray([True, False, False, True]) == result))

    def test_cat_zero_wronge_base(self):
        with pytest.raises(ValueError):
            c = CatZero(['a', 'a', 'b', 'c', 'a'], base_index=1)

    def test_preserve_name(self):
        ds = TypeRegister.Dataset({'strcol': np.random.choice(['a', 'b', 'c'], 10), 'numcol': arange(10)})
        c = Categorical(ds.strcol)
        assert c.get_name() == 'strcol'

        c = Categorical([ds.strcol, ds.numcol])
        ds2 = c.sum(arange(10))
        labels = ds2.label_get_names()
        assert labels[0] == 'strcol'
        assert labels[1] == 'numcol'

        ds = TypeRegister.Dataset({'mycodes': np.random.randint(1, 4, 10)})
        c = Categorical(ds.mycodes, {'a': 1, 'b': 2, 'c': 3})
        assert c.get_name() == 'mycodes'

        codes = np.random.randint(1, 4, 10)
        cats = FastArray(['a', 'b', 'c'])
        cats.set_name('test')
        c = Categorical(codes, cats)
        assert c.get_name(), 'test'

    def test_construct_from_categorical(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        d = Categorical(c)
        assert isinstance(d.category_array, np.ndarray)
        assert isinstance(d.expand_array, np.ndarray)

        d2 = Categorical([c])
        assert isinstance(d2.category_array, np.ndarray)
        assert isinstance(d2.expand_array, np.ndarray)

    def test_total_size(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        assert c._total_size == 8

        c = Categorical([arange(5, dtype=np.int32), arange(5, dtype=np.int32)])
        assert c._total_size == 45

        c = Categorical([arange(5, dtype=np.int64), arange(5, dtype=np.int64)])
        assert c._total_size == 85

    # removed while modifying groupby calculation behavior
    # def test_hold_dataset(self):
    #    ds = TypeRegister.Dataset({'strcol':np.random.choice(['a','b','c'],30), 'numcol':arange(30)})
    #    c = ds.cat('strcol')
    #    self.assertTrue(isinstance(c._dataset, TypeRegister.Dataset))
    #    result = c.sum()
    #    self.assertTrue(isinstance(result, TypeRegister.Dataset))
    #    self.assertEqual(result._nrows, 3)

    def test_expand_dict(self):
        og_strings = FastArray(['a', 'a', 'b', 'c', 'a'])
        og_nums = arange(5)

        c = Categorical([og_strings, og_nums])
        d = c.expand_dict
        assert isinstance(d, dict)
        assert len(d) == 2
        dictlist = list(d.values())
        assert bool(np.all(dictlist[0] == og_strings))
        assert bool(np.all(dictlist[1] == og_nums))

        c = Categorical([1, 2, 3], {'a': 1, 'b': 2, 'c': 3})
        d = c.expand_dict
        assert isinstance(d, dict)
        assert len(d) == 1
        dictlist = list(d.values())
        assert bool(np.all(dictlist[0] == arange(1, 4)))

        c = Categorical(np.random.randint(0, 10, 100_100))
        with pytest.warns(UserWarning):
            d = c.expand_dict

    def test_expand_array(self):
        c = Categorical([1, 2, 3], {'a': 1, 'b': 2, 'c': 3})
        arr = c.expand_array
        assert bool(np.all(arr == arange(1, 4)))

        c = Categorical([FastArray(['a', 'b', 'c', 'a']), FastArray([1, 2, 3, 1])])

        # expand array now works on multikey categoricals, returns a tuple of expanded arrays SJK: 4/29/2019
        multi_expand = c.expand_array
        assert isinstance(multi_expand, tuple)
        assert len(multi_expand) == 2
        assert bool(np.all(FastArray(['a', 'b', 'c', 'a']) == multi_expand[0]))
        assert bool(np.all(FastArray([1, 2, 3, 1]) == multi_expand[1]))

        c._fa[:] = 0
        multi_expand = c.expand_array
        assert bool(np.all(isnan(multi_expand[1])))
        assert bool(np.all(multi_expand[0] == b'Filtered'))

    def test_true_false_spacer(self):
        c = Categorical(['a', 'b', 'c'])
        t_true = c._tf_spacer(['test', True])
        assert t_true == 'testTrue '

        t_false = c._tf_spacer(['test', False])
        assert t_false == 'testFalse'

    def test_mapping_hstack(self):
        c1 = Categorical([1, 1, 1, 1, 2, 3], {'a': 1, 'b': 2, 'c': 3})
        c2 = Categorical([1, 1, 1, 1, 3, 4], {'a': 1, 'c': 3, 'd': 4})
        stacked = Categorical.hstack([c1, c2])
        assert stacked.unique_count == 4
        assert stacked.from_category('b') == 2
        assert stacked.from_category('d') == 4
        assert len(stacked) == 12

        c1 = Categorical([1, 1, 1, 1, 2, 3], {'a': 1, 'b': 2, 'd': 3})
        c2 = Categorical([1, 1, 1, 1, 3, 4], {'a': 1, 'c': 3, 'd': 4})

        # removed, hstack now relies on unique codes only SJK: 3/5/2019
        # with self.assertRaises(TypeError):
        #    c3 = Categorical.hstack([c1, c2])

    def test_matlab_nan(self):
        dts = [np.int8, np.int16, np.int32, np.int64]
        matlab_float_idx = FastArray([1.0, 0.0, np.nan])
        matlab_cats = ['a', 'b']

        for dt in dts:
            c = Categorical(matlab_float_idx, matlab_cats, dtype=dt, from_matlab=True)
            assert bool(np.all(c._fa == [1, 0, 0])), f'failed to flip nan to zero for dtype {dt}'
            assert np.dtype(dt) == c.dtype

    def test_from_provided_with_filter(self):
        # not found and filter
        c = Categorical(
            ['a', 'a', 'b', 'c', 'd'],
            ['a', 'b', 'c'],
            filter=FastArray([False, False, True, True, False]),
            invalid='INVALID',
        )
        c.filtered_set_name('INVALID')
        correct = FastArray([b'INVALID', b'INVALID', b'b', b'c', b'INVALID'])
        assert bool(np.all(c.expand_array == correct))

        # filter only (uses default invalid)
        c = Categorical(['a', 'a', 'b', 'c'], ['a', 'b', 'c'], filter=FastArray([False, False, True, True]),)
        f = c.filtered_name
        correct = FastArray([f, f, b'b', b'c'])
        assert bool(np.all(c.expand_array == correct))
        # even though filtered out, categories still untouched
        correct = FastArray([b'a', b'b', b'c'])
        assert bool(np.all(c.category_array == correct))

        # filtering not allowed for base index 0
        with pytest.raises(ValueError):
            c = Categorical(
                ['a', 'a', 'b', 'c'], ['a', 'b', 'c'], filter=FastArray([False, False, True, True]), base_index=0,
            )

    def test_numeric_invalid(self):
        # 5/16/2019 invalid category must be in provided uniques
        c = Categorical([1.0, 1.0, 2.0], [1.0, 2.0], invalid=2.0)
        assert c._fa[2] == 2
        num = c.sum(arange(1, 4)).col_0[0]
        assert num == 3

    def test_get_groupings(self):
        g, f, n = (
            FastArray([2, 3, 0, 4, 1]),
            FastArray([0, 0, 2, 4]),
            FastArray([0, 2, 2, 1]),
        )

        c = Categorical(['b', 'c', 'a', 'a', 'b'], base_index=0)
        gg = c.get_groupings()
        group = gg['iGroup']
        first = gg['iFirstGroup']
        ncount = gg['nCountGroup']

        assert bool(np.all(g == group))
        assert bool(np.all(f == first))
        assert bool(np.all(n == ncount))

        c = Categorical(['b', 'c', 'a', 'a', 'b'], base_index=1)
        gg = c.get_groupings()
        group = gg['iGroup']
        first = gg['iFirstGroup']
        ncount = gg['nCountGroup']
        assert bool(np.all(g == group))
        assert bool(np.all(f == first))
        assert bool(np.all(n == ncount))

    def test_repr(self):
        # just make sure no error for coverage
        c = Categorical(['a', 'b', 'c'])
        r = c.__repr__()
        assert r, f"Representation should not be empty for Categorical '{c}'."
        assert isinstance(r, str)

    def test_copy_deep(self):
        c = Categorical(['a', 'b', 'c'])
        d = c.copy(deep=True)

        d[0] = 'b'
        assert c[0] == 'a'
        assert c._fa[0] == 1
        assert d[0] == 'b'
        assert d._fa[0] == 2

    def test_copy_new_filter(self):
        a = Categorical('A B A B A B'.split())
        b = Categorical('B A B A B A'.split())
        c = a.copy()
        f = c == 'A'
        c[f] = b[f]
        assert c[0] == 'B'
        assert c[1] == 'B'
        assert a[0] == 'A'
        assert a[1] == 'B'
        assert b[0] == 'B'
        assert b[1] == 'A'

    def test_setitem_tuple(self):
        c = Categorical([arange(5), arange(5)])
        c[0] = (1, 1)
        assert c._fa[0] == 2

    def test_nunique(self):
        codes = np.random.randint(0, 3, 1000)
        d = {0: 'All', 1: 'ManualAndQuasi', 2: 'Manual'}
        c = Categorical(codes, d)
        n = c.nunique()
        assert n == 3
        assert len(c.unique()) == 3

        codes = np.ones(1000, dtype=np.int32)
        c = Categorical(codes, d)
        n = c.nunique()
        assert n == 1
        assert len(c.unique()) == 1

        codes = arange(5)
        c = Categorical(codes, d)
        n = c.nunique()
        assert n == 5
        assert len(c.unique()) == 5

        c = Categorical(['a', 'a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'])
        n = c.nunique()
        assert n == 4
        assert len(c.unique()) == 4

        c = Categorical(['a', 'a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'], base_index=0)
        n = c.nunique()
        assert n == 4
        assert len(c.unique()) == 4

        c = Categorical(['a', 'a', 'b', 'c', 'd'])
        c[2] = 0
        n = c.nunique()
        assert n == 3
        assert len(c.unique()) == 3
        assert c.unique_count == 4

        c = Categorical([arange(3), np.array(['a', 'b', 'c'])])
        c[0] = 0
        n = c.nunique()
        assert n == 2
        assert c.unique_count == 3

        # The following assertion is moved to it's own unit pytest along with an xfail.
        # found below and named test_multikey_categorical_unique.
        # assert len(c.unique()) == 2


    def test_unique(self):
        l = list('xyyz')
        c, c_sub = rt.Cat(l), rt.Cat(l[:3])
        assert_array_equal(c.unique(), c.category_array, 'mismatch between unique categories and category array')
        assert_array_equal(c.unique(), c.category_array.unique(), 'mismatch between unique categories and expanded category array')
        assert c.nunique() == 3, 'mismatch in number of unique categories'

        assert_array_equal(c[:3].unique(), c_sub.category_array, 'mismatch between unique categories and category array with sliced categorical')
        assert_array_equal(c[:3].unique(), c_sub.category_array.unique(), 'mismatch between unique categories and expanded category array with sliced categorical')
        assert c[:3].nunique() == 2, 'mismatch in number of unique categories with sliced categorical'


    def test_scalar_unique(self):
        idx = ones(100)
        cats = 700_000.0
        c = Categorical(idx, cats, from_matlab=True)
        assert isinstance(c, Categorical)
        assert c.unique_count == 1

    def test_stack_multikey(self):
        # TODO pytest parameterize the strings
        strs = FA(np.random.choice(['aaaaa', 'b', 'ccc'], 23))
        flts = np.random.choice([7.14, 6.66, 5.03], 23)
        c1 = Categorical([strs, flts])
        c1_str = Categorical(strs)
        c1_flt = Categorical(flts)

        strs2 = FA(np.random.choice(['b', 'aaaaa'], 17))
        flts2 = np.random.choice([5.03, 7.14], 17)
        c2 = Categorical([strs2, flts2])
        c2_str = Categorical(strs2)
        c2_flt = Categorical(flts2)

        fa_str = hstack([strs, strs2])
        fa_flt = hstack([flts, flts2])

        # TODO add assertions for multikey Categoricals
        c_str = Categorical(fa_str)
        c_flt = Categorical(fa_flt)

        # TODO move these into SDS save / load tests
        paths = [r'riptable/tests/temp/ds1.sds', r'riptable/tests/temp/ds2.sds']
        ds1 = Dataset(
            {
                'mkcat': c1,
                'strcat': c1_str,
                'fltcat': c1_flt,
                'strfa': strs,
                'fltfa': flts,
            }
        )
        ds2 = Dataset(
            {
                'mkcat': c2,
                'strcat': c2_str,
                'fltcat': c2_flt,
                'strfa': strs2,
                'fltfa': flts2,
            }
        )
        ds1.save(paths[0])
        ds2.save(paths[1])

        # normal dataset hstack
        hstack_ds = hstack([ds1, ds2])
        assert isinstance(hstack_ds, Dataset)

        # dataset hstack from load
        stack_load_ds = load_sds(paths, stack=True)
        assert isinstance(stack_load_ds, PDataset)

        # multikey cat hstack
        hstack_mkcats = hstack([c1, c2])
        assert isinstance(hstack_mkcats, Categorical)

        # normal array hstack
        hstack_strs = hstack([strs, strs2])
        hstack_flts = hstack([flts, flts2])

        # single cat hstack
        hstack_cstrs = hstack([c1_str, c2_str])
        assert isinstance(hstack_cstrs, Categorical)
        hstack_cflts = hstack([c1_flt, c2_flt])
        assert isinstance(hstack_cflts, Categorical)

        assert bool(np.all(hstack_strs == hstack_cstrs.expand_array))
        assert bool(np.all(hstack_flts == hstack_cflts.expand_array))

        mktup = [*hstack_mkcats.category_dict.values()]
        assert bool(np.all(hstack_mkcats._expand_array(mktup[0]) == fa_str))
        assert bool(np.all(hstack_mkcats._expand_array(mktup[1]) == fa_flt))

        mktup2 = [*stack_load_ds.mkcat.category_dict.values()]
        assert bool(np.all(stack_load_ds.mkcat._expand_array(mktup2[0]) == fa_str))
        assert bool(np.all(stack_load_ds.mkcat._expand_array(mktup2[1]) == fa_flt))

        mktup3 = [*hstack_ds.mkcat.category_dict.values()]
        assert bool(np.all(hstack_ds.mkcat._expand_array(mktup3[0]) == fa_str))
        assert bool(np.all(hstack_ds.mkcat._expand_array(mktup3[1]) == fa_flt))

        for p in paths:
            os.remove(p)

    # TO TEST:
    # regular python Enum
    # apply / apply_dataset, etc.

    # def test_sort_copy(self):
    #    c = Categorical(np.random.choice(['a','b','c'], 15))
    #    d = c.sort_copy()

    #    c = Categorical([np.random.choice(['a','b','c'], 15), np.random.randint(0,3,15)])
    #    d = c.sort_copy()

    # ----------------------------------------------------------
    # def test_str_repr(self):
    #    '''
    #    SJK: We're still in the early stages of deciding how to print out or summarize a categorical in the workspace.
    #    Comment it out if repr or str changes, and I will fix up.
    #    '''
    #    # no break
    #    input = ['b', 'b', 'b', 'a', 'b', 'b']
    #    str_string = ', '.join(input)
    #    repr_string = "Categorical(["+str_string+"])"
    #    c = Categorical(input)
    #    self.assertEqual(str(c),str_string, msg=f"__str__ did not produce the correct string {str_string} for categorical. got {str_string} instead")
    #    self.assertEqual(c.__repr__(),repr_string, msg=f"__repr__ did not produce the correct string {str_string} for categorical. got {str_string} instead")

    #    # add break
    #    slice_size = 5
    #    input = ['b', 'b', 'b', 'a', 'b', 'b', 'b', 'b', 'b', 'a', 'b', 'b', 'c', 'c']
    #    str_string = ', '.join(input[:slice_size]+['...']+input[-slice_size:])
    #    repr_string = "Categorical(["+str_string+"])"
    #    c = Categorical(input)
    #    self.assertEqual(str(c),str_string, msg=f"__str__ did not produce the correct string {str_string} for categorical. got {str_string} instead")
    #    self.assertEqual(c.__repr__(),repr_string, msg=f"__repr__ did not produce the correct string {str_string} for categorical. got {str_string} instead")

    def test_as_string_array(self):
        # SJK 10/4/2018 - as string array now returns bytes OR unicode (whatever type the string based categorical is holding)
        f = np.array([b'b', b'b', b'b', b'a', b'b', b'b'])
        c = Categorical(f)
        is_equal = bool(np.all(c.as_string_array == f))
        assert isinstance(c.as_string_array, FastArray), f"Categorical did not return a fastarray in as_string_array"
        assert (
            is_equal
        ), f"Categorical returned an incorrect string array {c.as_string_array} view of itself. Expected {f}"

    def test_indexing_numeric(self):
        c = Cat([1.1, 2.2, 3.3])
        result = c['2.2']
        assert np.all(result == [False, True, False])

    def test_fill_forward(self):
        fa = FA([1., np.nan, 1.])
        c = Cat([1,1,1])
        c.fill_forward(fa, inplace=True)
        assert np.all(fa == [1., 1., 1.])

    # TODO pytest parameterize `compare_func_names`
    def test_all_compare_tests(self):
        # with scalar
        # cat(unicode)
        i = 2
        c1 = Categorical(three_ints)
        if ShowCompareInfo:
            print("Categorical:", c1)
        if ShowCompareInfo:
            print("Compare unicode to int scalar: 2")
        self.compare_cat_test(c1, compare_func_names, int_success, i)

        # cat(unicode) / unicode, unicode list
        i = "AMZN\u2082"
        c3 = Categorical(three_unicode)
        if ShowCompareInfo:
            print("Categorical:", c3)
        if ShowCompareInfo:
            print("Compare unicode cat to unicode string")
        self.compare_cat_test(c3, compare_func_names, int_success, i)
        if ShowCompareInfo:
            print("Compare to list of unicode string")
        self.compare_cat_test(c3, compare_func_names, int_success, [i])
        if ShowCompareInfo:
            print("Compare to a numpy array of unicode string")
        self.compare_cat_test(c3, compare_func_names, int_success, np.array([i]))

        # cat(bytes) / bytes, bytes list
        i = b'b'
        c4 = Categorical(three_bytes)
        if ShowCompareInfo:
            print("Categorical:", c4)
        if ShowCompareInfo:
            print("Compare bytes cat to bytestring")
        self.compare_cat_test(c4, compare_func_names, int_success, i)
        if ShowCompareInfo:
            print("Compare to bytestring in list")
        self.compare_cat_test(c4, compare_func_names, int_success, [i])
        if ShowCompareInfo:
            print("Compare to bytestring in numpy array")
        self.compare_cat_test(c4, compare_func_names, int_success, np.array([i]))

        # cat(bytes) / unicode, unicode list
        i = "b"
        c5 = Categorical(three_bytes)
        if ShowCompareInfo:
            print("Categorical:", c5)
        if ShowCompareInfo:
            print("Compare bytes cat to unicode string")
        self.compare_cat_test(c5, compare_func_names, int_success, i)
        if ShowCompareInfo:
            print("Compare to unicode string in list")
        self.compare_cat_test(c5, compare_func_names, int_success, [i])
        if ShowCompareInfo:
            print("Compare to unicode string in numpy array")
        self.compare_cat_test(c5, compare_func_names, int_success, np.array([i]))

        # equal categoricals (same dictionary)
        # cat(bytes) / cat(bytes)
        if ShowCompareInfo:
            print("Compare two equal categoricals:")
        if ShowCompareInfo:
            print("Both from byte lists:")
        c1 = Categorical(three_bytes)
        c2 = Categorical(three_bytes)
        if ShowCompareInfo:
            print("cat1:", c1)
        if ShowCompareInfo:
            print("cat2:", c2)
        self.compare_cat_test(c1, compare_func_names, same_success, c2)

        # cat(unicode) / cat(unicode)
        if ShowCompareInfo:
            print("Both from unicode lists:")
        c1 = Categorical(three_unicode)
        c2 = Categorical(three_unicode)
        if ShowCompareInfo:
            print("cat1:", c1)
        if ShowCompareInfo:
            print("cat2:", c2)
        self.compare_cat_test(c1, compare_func_names, same_success, c2)

        # cat(unicode) / cat(bytes)
        if ShowCompareInfo:
            print("unicode/bytes list")
        c1 = Categorical(["a", "b", "c"])
        c2 = Categorical(three_bytes)
        if ShowCompareInfo:
            print("cat1:", c1)
        if ShowCompareInfo:
            print("cat2:", c2)
        self.compare_cat_test(c1, compare_func_names, same_success, c2)

        # unequal categoricals (same dictionary)
        # cat(bytes) / cat(bytes)
        if ShowCompareInfo:
            print("Compare two unequal categoricals (same dict):")
        if ShowCompareInfo:
            print("both bytes")
        c1 = Categorical([0, 1, 0], three_bytes)
        c2 = Categorical([2, 1, 2], three_bytes)
        if ShowCompareInfo:
            print("cat1:", c1)
        if ShowCompareInfo:
            print("cat2:", c2)
        self.compare_cat_test(c1, compare_func_names, diff_success, c2)

        # cat(unicode) / cat(unicode)
        if ShowCompareInfo:
            print("both unicode")
        c1 = Categorical([0, 1, 0], three_unicode)
        c2 = Categorical([2, 1, 2], three_unicode)
        if ShowCompareInfo:
            print("cat1:", c1)
        if ShowCompareInfo:
            print("cat2:", c2)
        self.compare_cat_test(c1, compare_func_names, diff_success, c2)

        ## cat(bytes) / int list (matching)
        # if ShowCompareInfo: print("Compare categorical to matching int list")
        # if ShowCompareInfo: print("bytes")
        # i = [1,2,3]
        # c1 = Categorical(three_bytes)
        # self.compare_cat_test(c1,compare_func_names,same_success,i)
        ## cat(unicode) / int list (matching)
        # if ShowCompareInfo: print("unicode")
        # c1 = Categorical(three_unicode)
        # self.compare_cat_test(c1,compare_func_names,same_success,i)

        ## cat(bytes) / int list (non-matching)
        # if ShowCompareInfo: print("Compare categorical to non-matching int list")
        # if ShowCompareInfo: print("bytes")
        # i = [3,2,1]
        # c1 = Categorical(three_bytes)
        # self.compare_cat_test(c1,compare_func_names,int_success,i)
        ## cat(unicode) / int list(non-matching)
        # if ShowCompareInfo: print("unicode")
        # c1 = Categorical(three_unicode)
        # self.compare_cat_test(c1,compare_func_names,int_success,i)

    # def cat_slicing(self):
    #    three_unicode =FA(["AAPL\u2080","AMZN\u2082","IBM\u2081"])
    #    three_bytes = FA([b'a',b'b',b'c'])

    #    num_rows=8
    #    idx_size=15

    #    get_item_dicts = {
    #        "single_slices" : {
    #            ":2" : slice(None,2,None),
    #            "-2:": slice(-2,None,None),
    #            "2:5": slice(2,5,None),
    #            "5:" : slice(5,None,None),
    #            ":"  : slice(None,None,None)
    #        },

    #        "bool_arrays" : {
    #            "python_bool" : [True, False, True, False, False, True, True, True, False, True, False, False, True, False, True],
    #            "numpy_bool"  : np.array([True, False, True, False, False, True, True, True, False, True, False, False, True, False, True])
    #        },

    #        "int_indices" : { "int_idx_size"+str(idx_size) : np.random.randint(low=0,high=num_rows,size=idx_size) for idx_size in range(1,num_rows) }
    #    }

    #    failures = 0

    #    idx_list = np.random.randint(low=0,high=8,size=15)
    #    s_list = np.array([b'adam',b'bob',b'charlie',b'david',b'edward',b'frank',b'greg',b'harold'])
    #    c = Categorical(idx_list, s_list)

    #    for key, test_dict in get_item_dicts.items():
    #        print("\n\n"+key)
    #        for call_str, val in test_dict.items():
    #            success = s_list[idx_list[val]]
    #            if np.all(c[val].as_string_array == success):
    #                message = "success"
    #            else:
    #                message = "failure"
    #                failures += 1
    #            print(call_str, message)

    #    print("Tests complete with",failures,"errors")

    #    return c

    @pytest.mark.xfail(
        reason="RIP-215 - lead to inconsistent Categorical state; please add hypothesis tests when resolved."
    )
    def test_category_add(self):
        cat = Categorical(list("bbcdebc"))
        e = "a"
        cat.category_add(e)
        assert e in cat, "expect the added category to be added to the Categorical"
        assert e in cat._categories, "expect the added category to be added to the Categorical._categories"
        assert e in cat.category_array, "expect the added category to be added to the Categorical.category_array"
        assert e in cat.category_dict, "expect the added category to be added to the Categorical.category_dict"

    @pytest.mark.xfail(
        reason="RIP-215 - lead to inconsistent Categorical state; please add hypothesis tests when resolved."
    )
    def test_category_remove(self):
        cat = Categorical(list("bbcdebc"))
        e = cat[0]
        cat.category_remove(e)
        assert e not in cat, "expect the removed category to be removed from the Categorical"
        assert e not in cat._categories, "expect the removed category to be removed from the Categorical._categories"
        assert (
            e not in cat.category_array
        ), "expect the removed category to be removed from the Categorical.category_array"
        assert (
            e not in cat.category_dict
        ), "expect the removed category to be removed from the Categorical.category_dict"

    # TODO move this to testing utils
    def compare_cat_test(self, cat, compare_func_names, success_bools, i):
        for fname, success in zip(compare_func_names, success_bools):
            func = getattr(cat, fname)
            result = func(i)
            assert np.all(result == success), f'fail on {fname} {cat} {i}'
            if ShowCompareInfo:
                if np.all(result == success):
                    message = "succeeded"
                else:
                    message = "failed"
                print(fname, message)

    def test_duplicated(self):
        result = Cat([2, 3, 2], list('qwery')).duplicated()
        assert np.all(result == FA([False, False, True]))

    def test_cat_copy(self):
        # add deep copy for enum, single, multi
        x = arange(6, dtype=uint16) // 2
        c = Cat(x, {0: 'Run', 1: 'Stop', 2: 'Start'}, dtype=uint16)
        c[1] = 'Start'
        a = c.copy()
        d = a[:5]
        a[1] = 'Run'
        b = a[:5]
        assert a._fa[1] == 0
        assert b._fa[1] == 0
        assert c._fa[1] == 2
        assert d._fa[1] == 0

    def test_assinglekey(self):
        c = Cat([1, 2, 1, 2, 1, 2], {'Sunny': 1, 'Thunderstorms': 2})
        # insert bad value
        c._fa[3] = 17
        c1 = c.as_singlekey(ordered=False)
        c2 = c.as_singlekey(ordered=True)
        assert np.all(c1.expand_array == c2.expand_array)
        c = Cat([-1, -2, -1, -2, -1, -2], {'Sunny': -1, 'Thunderstorms': -2})
        c._fa[3] = 17
        c3 = c.as_singlekey(ordered=False)
        c2 = c.as_singlekey(ordered=True)
        assert np.all(c1.expand_array == c2.expand_array)
        assert np.all(c3.expand_array == c2.expand_array)


# Cannot use the pytest.mark.parameterize decorator within classes that inherit from unittest.TestCase.
# Will need to migrate for unittest to pytest and fold the following categorical tests into Categorical_Test.


@pytest.mark.parametrize(
    "categoricals",
    [
        # Categorical constructed from python list data
        pytest.param(
            [
                Categorical(data)
                for data in get_categorical_data_factory_method([CategoryMode.StringArray, CategoryMode.NumericArray])
            ],
            id="cat_with_list_values",
        ),
        # Categorical constructed from numpy array
        pytest.param(
            [
                Categorical(np.array(data))
                for data in get_categorical_data_factory_method([CategoryMode.StringArray, CategoryMode.NumericArray])
            ],
            id="cat_with_np_array_values",
        ),
        # Categorical constructred from riptable fast array
        pytest.param(
            [
                Categorical(rt.FastArray(data))
                for data in get_categorical_data_factory_method([CategoryMode.StringArray, CategoryMode.NumericArray])
            ],
            id="cat_with_rt_fastarray_values",
        ),
        # failed test cases
        pytest.param(
            [Categorical(data) for data in get_categorical_data_factory_method(CategoryMode.MultiKey)],
            marks=[
                pytest.mark.xfail(
                    reason="RIP-410 - Bug for MultiKey Categoricals: AttributeError: 'Categorical' object has no attribute 'ismultikey_labels'"
                )
            ],
            id="cat_with_tuple_values",
        ),
    ],
)
def test_one_hot_encode(categoricals):
    for categorical in categoricals:
        col_names, encoded_arrays = categorical.one_hot_encode()
        category_array = categorical.category_array.astype('U')

        # Test 1.1 The col_names are the same as the category array.
        assert not set(category_array).symmetric_difference(set(col_names)), (
            f"The column names should be the same as the names in the category array",
            f"category array {category_array}\ncolumn names {col_names}",
        )

        # Test 1.2 The encoded_arrays dtypes are consistent with one another.
        encoded_arrays_dtypes = set([fa.dtype for fa in encoded_arrays])
        assert (
            len(encoded_arrays_dtypes) == 1
        ), f"Encoded array dtypes should be consistent, got {encoded_arrays_dtypes}"

        # todo for each category, assert the mask of the categorical is in the encoded_arrays


@pytest.mark.parametrize(
    "categoricals",
    [
        # Categorical constructed from python list data
        pytest.param(
            [Categorical(data) for data in get_categorical_data_factory_method([CategoryMode.StringArray])],
            id="cat_with_list_values",
        ),
        # Categorical constructed from numpy array
        pytest.param(
            [Categorical(np.array(data)) for data in get_categorical_data_factory_method([CategoryMode.StringArray])],
            id="cat_with_np_array_values",
        ),
        # Categorical constructred from riptable fast array
        pytest.param(
            [
                Categorical(rt.FastArray(data))
                for data in get_categorical_data_factory_method([CategoryMode.StringArray])
            ],
            id="cat_with_rt_fastarray_values",
        ),
    ],
)
def test_shift_cat(categoricals):
    # todo Handle numeric invalid types for categoricals with values other than strings.
    filtered_name = rt.rt_enum.FILTERED_LONG_NAME.encode("utf-8")
    for categorical in categoricals:
        cat_len = len(categorical)
        for i in range(-cat_len + 1, cat_len):  # exhaustive shift of all Categorical values.
            # shift the categorical i-places
            shift_cat = categorical.shift_cat(i)

            # The category array should remain unchanged.
            assert_array_equal(shift_cat.category_array, categorical.category_array)

            # The underlying FastArray should have the items shifted to the i-th position.
            if i > 0:  # shift forwards case
                assert_array_equal(
                    shift_cat._fa[i:], categorical._fa[:-i], f"FastArray items should be shifted by {i} postions.",
                )

                # The Categorical should have the values shifted to the i-th position.
                cat_values, shift_cat_values = (
                    categorical.expand_array,
                    shift_cat.expand_array,
                )
                assert_array_equal(
                    shift_cat_values[i:], cat_values[:-i], f"Categorical values should be shifted by {i} positions.",
                )

                # The underlying FastArray should have the first i-items to be the invalid value.
                # The Categorical values should have the first i-items be the filtered or invalid name.
                # Need to handle other invalid values and other Categorical base indexing.
                assert_array_equal(
                    shift_cat_values[:i],
                    np.full(i, filtered_name),
                    f"Shifted Categorical values up to {i}-th position should be '{filtered_name}'.",
                )
                assert_array_equal(
                    shift_cat._fa[:i],
                    np.zeros(i),
                    f"Shifted Categorical underlying FastArray items up to {i}-th position should be the invalid value 0.",
                )
            elif i < 0:  # shifted backwards case
                i = abs(i)  # slicing arithmetic based on positional value of i
                assert_array_equal(
                    shift_cat._fa[: cat_len - i],
                    categorical._fa[i:],
                    f"FastArray items should be shifted by -{i} postions.",
                )
                cat_values, shift_cat_values = (
                    categorical.expand_array,
                    shift_cat.expand_array,
                )
                assert_array_equal(
                    shift_cat_values[: cat_len - i],
                    cat_values[i:],
                    f"Categorical values should be shifted by -{i} positions.",
                )
                assert_array_equal(
                    shift_cat_values[-i:],
                    np.full(i, filtered_name),
                    f"Shifted Categorical values up to -{i}-th position should be '{filtered_name}'.",
                )
                assert_array_equal(
                    shift_cat._fa[-i:],
                    np.zeros(i),
                    f"Shifted Categorical underlying FastArray items up to -{i}-th position should be the invalid value 0.",
                )
            elif i == 0:  # zero-th shift case
                # test for equality
                assert_array_equal(shift_cat.category_array, categorical.category_array)
                assert_array_equal(shift_cat._fa, categorical._fa)
                cat_values, shift_cat_values = (
                    categorical.expand_array,
                    shift_cat.expand_array,
                )
                assert_array_equal(shift_cat_values, cat_values)

        # shift overflow for backward and forward case up to two values
        for i in list(range(-cat_len - 2, -cat_len)) + list(range(cat_len, cat_len + 2)):
            shift_cat = categorical.shift_cat(i)
            assert_array_equal(shift_cat.category_array, categorical.category_array)
            # Investigate possible bug with expanding Categorical values. E.g.:
            # given:
            # Categorical([a, a, a, a, a, a, a, a, a, a]) Length: 10
            #   FastArray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int8) Base Index: 1
            #   FastArray([b'a'], dtype='|S1') Unique count: 1
            # shifted categorical
            # Categorical([Filtered, Filtered, Filtered, Filtered, Filtered, Filtered, Filtered, Filtered, Filtered, Filtered]) Length: 10
            #   FastArray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int8) Base Index: 1
            #   FastArray([b'a'], dtype='|S1') Unique count: 1
            # got
            # E            x: FastArray([b'Filtered', b'Filtered', b'Filtered', b'Filtered',
            # E                      b'Filtered', b'Filtered', b'Filtered', b'Filtered',
            # E                      b'Filtered', b'a'], dtype='|S8')
            # E            y: array([b'Filtered', b'Filtered', b'Filtered', b'Filtered', b'Filtered',
            # E                  b'Filtered', b'Filtered', b'Filtered', b'Filtered', b'Filtered'],
            # E                 dtype='|S8')
            # Expected all values to be b'Filtered', but saw b'a'.
            # todo assert_array_equal(shift_cat_values, np.full(cat_len, filtered_name), f"Overflow shifted Categorical values. All values are expected to be invalid '{filtered_name}'.")
            assert_array_equal(
                shift_cat._fa,
                np.zeros(cat_len),
                f"Overflow shifted Categorical underlying FastArray items. All values are expected to be invalid value 0.",
            )


@pytest.mark.parametrize(
    # TODO - add base 0 and base 1 indexing w/ expectations
    "categoricals",
    [
        # Categorical constructed from python list data
        pytest.param(
            [Categorical(data) for data in get_categorical_data_factory_method([CategoryMode.StringArray])],
            id="cat_with_list_values",
        ),
        # Categorical constructed from numpy array
        pytest.param(
            [Categorical(np.array(data)) for data in get_categorical_data_factory_method([CategoryMode.StringArray])],
            id="cat_with_np_array_values",
        ),
        # Categorical constructred from riptable fast array
        pytest.param(
            [
                Categorical(rt.FastArray(data))
                for data in get_categorical_data_factory_method([CategoryMode.StringArray])
            ],
            id="cat_with_rt_fastarray_values",
        ),
    ],
)
@pytest.mark.parametrize("misc", [None, "INVALID"])  # TODO - add numeric values
@pytest.mark.parametrize("inplace", [False, True])
def test_shrink(categoricals, misc, inplace):
    for categorical in categoricals:
        cat = categorical.copy(deep=True)  # deep copy so test data remains unchanged with inplace shrinks

        # Test 1 Shrink with empty values.
        # Shrink to empty categories.
        shrink_cat = cat.shrink([], misc=misc, inplace=inplace)

        # Type is preserved after shrinking.
        assert isinstance(shrink_cat, Categorical), "shrink_cat should be a Categorical."

        if misc is None:
            # For base index 1 Categorical, the underlying FastArray should be all zeros.
            assert_array_equal(shrink_cat._fa, np.zeros(len(cat)))

            # The Categorical categories should be empty.
            expected_category_array = np.empty(0)
            assert_array_equal(
                shrink_cat.category_array, expected_category_array, f"Category dictionary values should be empty.",
            )
            for arr in shrink_cat.category_dict.values():
                assert_array_equal(
                    arr, expected_category_array, f"Category dictionary values should be empty.",
                )

            # TODO expanding shrink categorical does not return original types invalid value; instead it returns nans
            # N.B, when shrinking, the category array type changes to float64
            # E                x: FastArray([nan])
            # E                y: array([b'Filtered'], dtype='|S8')
            # assert_array_equal(shrink_cat.expand_array, np.full(len(cat), filtered_name), f"Given empty values, shrink categorical values should all be invalid '{filtered_name}'.")
        else:  # single categories being the specified misc
            # TODO - consider any constraints to assert on for the dtype?
            # The invalid value based on the dtype: e.g., for U32 its -2147483646
            # assert_array_equal(shrink_cat._fa, InvalidValuesForDtype)
            # assert_array_equal(shrink_cat.expand_array, InvalidValuesForDtypeExpanded)

            # The categories should only contain the misc value.
            expected_category_array = np.array(misc)
            assert_array_equal(
                shrink_cat.category_array,
                expected_category_array,
                f"Category array should only contain the '{misc}' category.",
            )
            for arr in shrink_cat.category_dict.values():
                assert_array_equal(
                    arr,
                    expected_category_array,
                    f"Category dictionary values should only contain the '{misc}' category.",
                )

        # Test 2 Shrink with same categories
        cat = categorical.copy(deep=True)

        # Shrink to all the same categories.
        shrink_cat = cat.shrink(cat.category_array, misc=misc, inplace=inplace)

        # Type is preserved after shrinking.
        assert isinstance(shrink_cat, Categorical), "shrink_cat should be a Categorical."

        if misc is None:  # TODO handle the misc not None case
            shrink_cat_values, cat_values = shrink_cat.expand_array, cat.expand_array
            assert_array_equal(shrink_cat_values, cat_values)
            assert_array_equal(shrink_cat._fa, cat._fa)
            assert_array_equal(shrink_cat.category_array, cat.category_array)
            for arr, expected_arr in zip(shrink_cat.category_dict.values(), cat.category_dict.values()):
                assert_array_equal(arr, expected_arr)

        # TODO Test 3 Shrink with subset of categories
        cat = categorical.copy(deep=True)

        # Shrink to all the same categories.
        n = int(len(cat) / 2)
        shrink_cat = cat.shrink(cat.category_array[:n], misc=misc, inplace=inplace)

        # Type is preserved after shrinking.
        assert isinstance(shrink_cat, Categorical), "shrink_cat should be a Categorical."


@pytest.mark.parametrize(
    "categoricals",
    [
        # TODO - test categorical construction using numpy and riptable arrays as a separate test
        # Categorical constructed from python list data
        pytest.param([Categorical(data) for data in get_categorical_data_factory_method()], id="cat_with_list_values",),
        # Categorical constructed from numpy array
        pytest.param(
            [
                Categorical(np.array(data))
                for data in get_categorical_data_factory_method([CategoryMode.StringArray, CategoryMode.NumericArray])
            ],
            id="cat_with_np_array_values",
        ),
        # Categorical constructred from riptable fast array
        pytest.param(
            [
                Categorical(rt.FastArray(data))
                for data in get_categorical_data_factory_method([CategoryMode.StringArray, CategoryMode.NumericArray])
            ],
            id="cat_with_rt_fastarray_values",
        ),
    ],
)
def test_sds(categoricals, tmpdir):
    dir = tmpdir.mkdir("test_categorical_sds")
    for i, cat in enumerate(categoricals):
        name = "categorical_" + str(i)
        p = str(dir.join(name))
        save_sds(p, cat)
        cat2 = load_sds(p)

        # Test 1 Saved and loaded categoricals should be the same.
        # TODO vary the meta version optional parameter when calling Categorical._load_from_sds_meta_data
        assert isinstance(cat2, Categorical)
        assert_array_equal(cat2._fa, cat._fa)
        if not cat.ismultikey:  # MultiKey Categorical's do not support category_array operation
            assert_array_equal(cat2.category_array, cat.category_array)
        for actual, expected in zip(cat2.category_dict.values(), cat.category_dict.values()):
            assert_array_equal(actual, expected)
        cat2_values, cat_values = cat2.expand_array, cat.expand_array
        assert_array_equal(cat2_values, cat_values)

        # Test 2 As and from meta data Categoricals should be the same.
        cat3 = Categorical._from_meta_data(*cat._as_meta_data(name=name))
        # Saved and loaded categoricals should be the same.
        assert isinstance(cat3, Categorical)
        assert_array_equal(cat3._fa, cat._fa)
        if not cat.ismultikey:  # MultiKey Categorical's do not support category_array operation
            assert_array_equal(cat3.category_array, cat.category_array)
        for actual, expected in zip(cat3.category_dict.values(), cat.category_dict.values()):
            assert_array_equal(actual, expected)
        cat3_values, cat_values = cat3.expand_array, cat.expand_array
        assert_array_equal(cat3_values, cat_values)


@pytest.mark.parametrize(
    "categoricals",
    [
        # TODO handle CategoryMode IntEnum and Default
        [
            Categorical(data)
            for data in get_categorical_data_factory_method([CategoryMode.StringArray, CategoryMode.NumericArray])
        ]
        + [
            Categorical(data, base_index=0)
            for data in get_categorical_data_factory_method([CategoryMode.StringArray, CategoryMode.NumericArray])
        ]
    ],
)
def test_from_bin(categoricals):
    for cat in categoricals:
        cat_arr_len = len(cat.category_array)
        # Test 1 All bin values are in the category array.
        if cat.base_index == 0:
            for i in range(cat_arr_len):
                assert cat.from_bin(i) in cat.category_array
        elif cat.base_index == 1:
            for i in range(1, cat_arr_len + 1):
                assert cat.from_bin(i) in cat.category_array
        else:
            raise ValueError(f"Unhandled Categorical base index {cat.base_index}")

        # Test 2 Handling of invalid input types: base_index and bin.
        # The bin is not an integer.
        with pytest.raises(TypeError):
            cat.from_bin(str(i))
            cat.from_bin(float(i))

        # Bin value out of range.
        with pytest.raises(ValueError):
            cat.from_bin(-1)
            if cat.base_index == 0:
                cat.from_bin(cat_arr_len)
            elif cat.base_index == 1:
                cat.from_bin(0)
                cat.from_bin(cat_arr_len + 1)
            else:
                raise ValueError(f"Unhandled Categorical base index {cat.base_index}")

        # The base index is None.
        cat.grouping._base_index = None
        with pytest.raises(TypeError):
            cat.from_bin(1)


@pytest.mark.parametrize("cat", get_all_categorical_data())
def test_argsort(cat):
    assert_array_equal(
        cat.argsort(),
        np.argsort(cat._fa),
        "Categorical argsort should be equivalent to the argsort of the underlying FastArray",
    )


@pytest.mark.parametrize(
    "cats",
    [
        pytest.param(
            [
                Categorical(data)
                for data in get_categorical_data_factory_method([CategoryMode.StringArray, CategoryMode.NumericArray])
            ]
        ),
        pytest.param(
            [Categorical(data) for data in get_categorical_data_factory_method(CategoryMode.MultiKey)],
            marks=[
                pytest.mark.xfail(reason="NotImplementedError: Add categories not supported for MultiKey Categoricals")
            ],
        ),
    ],
)  # TODO parameterize across base index 0 and 1
def test_auto_add(cats):
    for cat in cats:
        alpha, beta = "alpha", "beta"
        first_index, last_index = 0, len(cat) - 1

        # Test 1 auto_add_on will allow addition of a category if the Categorical is unlocked,
        # otherwise an error is raised.
        cat.auto_add_on()
        cat.unlock()  # Categorical is unlocked

        # Test 1.1 When unlocked and attempting to add a category, the categories should be added.
        # set the first and last categories
        cat[first_index] = cat[last_index] = alpha
        # auto_add_on and unlock should not allow setting beyond the first and last index of categories
        with pytest.raises(IndexError):  # index out of bounds
            cat[first_index - 1] = alpha
            cat[last_index + 1] = alpha

        # category is added at specified index
        first_category = cat.category_array[cat._fa[first_index] - 1]
        # TODO normalize the category_array value, which is sometimes a numpy str_ or bytes_ to an ascii and compare
        # assert cat.category_array[cat._fa[first_index]-1] == alpha
        # assert at.category_array[cat._fa[last_index]-1] == alpha

        # added category is in category array and dictionary
        assert alpha in cat.category_array
        for categories in cat.category_dict.values():
            assert alpha in categories

        # Test 1.2 When locked and attempting to add a category, an error is raised and the categories should not be added.
        cat.lock()  # Categorical is locked
        with pytest.raises(IndexError):  # cannot add a category since index is locked
            cat[first_index] = beta
        assert beta not in cat.category_array
        for categories in cat.category_dict.values():
            assert beta not in categories

        # Test 2 auto_add_off will prevent category assignment of non-existing categories and raise an error
        cat.auto_add_off()

        # Test 2.1 Unlocked case
        cat.unlock()  # Categorical is unlocked
        with pytest.raises(ValueError):  # cannot automatically add categories while auto_add_categories is False
            cat[first_index] = beta

        # Test 2.2 Locked case
        cat.lock()
        with pytest.raises(IndexError):  # cannot add a category since index is locked
            cat[first_index] = beta


@pytest.mark.xfail(reason="rt_numpy.unique() needs to handles multikey categoricals")
def test_multikey_categorical_unique():
    c = Categorical([arange(3), FA(list('abc'))])
    assert len(c.unique()) == c.nunique()


@pytest.mark.parametrize("values", [list_bytes, list_unicode, list_true_unicode])
def test_categorical_convert(values):
    categories = list(set(values))
    # pd_c is a pandas Categorical with a missing category.
    # pandas Categorical will designate the values with a missing category by -1.
    pd_c = pd.Categorical(values, categories=categories[:-1])

    # The output of categorical_convert, when applied to a pandas Categorical, can be used to
    # construct a riptable Categorical. We test that this handles missing categories correctly.
    rt_values, rt_categories = rt.categorical_convert(pd_c)
    cat = rt.Categorical(rt_values, categories=rt_categories)

    # The invalid category should not be in the Categorical.
    missing_category = categories[-1]
    assert missing_category not in cat
    assert missing_category not in cat._categories
    assert missing_category not in cat.category_array
    assert missing_category not in cat.category_dict[next(iter(cat.category_dict))]  # values of first key

    # All other category values should be in the Categorical.
    for e in categories[:-1]:
        # assert e in cat  # uncomment when test_categorical_convert_xfail is fixed
        assert e in cat._categories
        assert e in cat.category_array
        assert e in cat.category_dict[next(iter(cat.category_dict))]  # values of first key


@pytest.mark.xfail(reason="RIP-396 - category not in Categorical, but is in Categorical.category_array")
@pytest.mark.parametrize("values", [list_bytes,])
def test_categorical_convert_xfail(values):
    categories = list(set(values))
    # pd_c is a pandas Categorical with a missing category.
    # pandas Categorical will designate the values with a missing category by -1.
    pd_c = pd.Categorical(values, categories=categories[:-1])

    rt_values, rt_categories = rt.categorical_convert(pd_c)
    cat = rt.Categorical(rt_values, categories=rt_categories)

    # All other category values should be in the Categorical.
    for e in categories[:-1]:
        assert e in cat


def test_build_dicts_enum():
    str_to_int, int_to_str = Categories.build_dicts_enum(LikertDecision)
    codes = list(str_to_int.values()) * 2

    c = Categorical(codes, categories=LikertDecision)
    c2 = Categorical(codes, categories=str_to_int)
    c3 = Categorical(codes, categories=int_to_str)

    # c is the our oracle Categorical.
    # Categoricals constructed from any of the dictionaries built by build_dicts_enum
    # should construct the same Categorical as c.
    assert_array_equal(c, c2)
    assert_array_equal(c, c3)


@pytest.mark.parametrize("values", [list("abcdef"), [b"a", b"b", b"c", b"d", b"e", b"f"]])
def test_build_dicts_python(values):
    # int
    d = {k: v for k, v in enumerate(values)}
    str_to_int, int_to_str = Categories.build_dicts_python(d)
    codes = list(d.keys()) * 2
    c = Categorical(codes, categories=d)
    c2 = Categorical(codes, categories=str_to_int)
    c3 = Categorical(codes, categories=int_to_str)

    # c is the our oracle Categorical.
    # Categoricals constructed from any of the dictionaries built by build_dicts_python
    # should construct the same Categorical as c.
    assert_array_equal(c, c2)
    assert_array_equal(c, c3)


@pytest.mark.parametrize(
    "a,b,a_in_b,b_in_a",
    [
        pytest.param(Cat(list('abc')), Cat(list('a')), FA([True, False, False]), FA([True]), id='single_key_overlap'),
        pytest.param(
            Cat([FA(list('abc')), FA([1,2,3])]),
            Cat([FA(list('a')), FA([1])]),
            FA([True, False, False]),
            FA([True]),
            id='single_multikey_overlap'
        ),
        pytest.param(
            Cat([FA(list('abc')), FA([1,2,3])]),
            Cat([FA(list('ab')), FA([1,2])]),
            FA([True, True, False]),
            FA([True, True]),
            id='two_multikey_overlap'
        ),
        pytest.param(
            Cat([FA(list('abcde')), FA([1,2,3,4,5])]),
            Cat([FA(list('dc')), FA([4,5])]),
            FA([False, False, False, True, False]),
            FA([True, False]),
            id='single_multikey_overlap2'
        ),
        pytest.param(
            Cat([FA(list('abcde')), FA([1,2,3,4,5])]),
            Cat([FA(list('aba')), FA([1,2,1])]),
            FA([True, True, False, False, False]),
            FA([True, True, True]),
            id='repeated_key_multikey_overlap'
        ),
        pytest.param(
            Cat([FA(list('abcdeab')), FA([1,2,3,4,5,1,6])]),
            Cat([FA(list('aba')), FA([1, 2, 1])]),
            FA([True, True, False, False, False, True, False]),
            FA([True, True, True]),
            id='repeated_key_multikey_overlap2'
        ),
    ]
)
def test_multikey_categorical_isin(a, b, a_in_b, b_in_a):
    assert_array_equal(a_in_b, a.isin(b))
    assert_array_equal(b_in_a, b.isin(a))

    # TODO this is a good candidate for a hypothesis test once the CategoricalStrategy is able to generate MultiKey Categoricals
    f_msg = 'expected to be consistent with cat1.as_singlekey().isin(cat2.as_singlekey()) operation.'
    assert_array_equal(a.as_singlekey().isin(b.as_singlekey()), a.isin(b), f_msg)
    assert_array_equal(b.as_singlekey().isin(a.as_singlekey()), b.isin(a), f_msg)


_make_unique_test_cases = pytest.mark.parametrize('cat, expected', [
    (rt.Cat([1, 1, 2, 2], ['a', 'a']), rt.Cat([1, 1, 1, 1], ['a'])),
    (rt.Cat([2, 2, 2, 2], ['a', 'a']), rt.Cat([1, 1, 1, 1], ['a'])),
    (rt.Cat([1, 2, 3, 3], ['a', 'a', 'b']), rt.Cat([1, 1, 2, 2], ['a', 'b'])),
    (rt.Cat([0, 0, 1, 1], ['a', 'a'], base_index=0), rt.Cat([0, 0, 0, 0], ['a'], base_index=0)),
    (rt.Cat([1, 1, 1, 1], ['a', 'a'], base_index=0), rt.Cat([0, 0, 0, 0], ['a'], base_index=0)),
    (rt.Cat([0, 0, 1, 1], ['a', 'b'], base_index=0), rt.Cat([0, 0, 1, 1], ['a', 'b'], base_index=0)),

    (rt.Cat([1, 1, 2, 2, 3], [99, 99, 101], ), rt.Cat([1, 1, 1, 1, 2], [99, 101])),
    (rt.Cat([0, 0, 1, 1], [99, 99], base_index=0), rt.Cat([0, 0, 0, 0], [99], base_index=0)),
    (rt.Cat([0, 0, 1, 1], [99, 101], base_index=0), rt.Cat([0, 0, 1, 1], [99, 101], base_index=0)),

    (rt.Cat([0, 0, 1, 1, 2, 2], ['a', 'a'], ), rt.Cat([0, 0, 1, 1, 1, 1], ['a'], )),
    (rt.Cat([0, 0, 1, 1, 2, 2, 3, 3], ['a', 'a', 'b'], ), rt.Cat([0, 0, 1, 1, 1, 1, 2, 2], ['a', 'b'], )),
])


@_make_unique_test_cases
def test_category_make_unique_not_inplace(cat, expected):
    res = cat.category_make_unique()
    assert (res == expected).all()


@pytest.mark.parametrize('base_index', [0, 1])
def test_category_make_unique_multikey(base_index):
    c1 = Categorical(np.arange(10) % 2, ['a', 'a'], base_index=base_index)
    c2 = Categorical(np.arange(10) % 3, ['a', 'b', 'c'], base_index=base_index)
    cat = Categorical([c1, c2], base_index=base_index)

    res = cat.category_make_unique()
    assert list(cat) == list(res)
