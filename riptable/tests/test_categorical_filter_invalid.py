import pytest
import os

from riptable import *
from riptable.rt_sds import SDSMakeDirsOn

# change to true since we write into /tests directory
SDSMakeDirsOn()


def arr_eq(a, b):
    return bool(np.all(a == b))


def arr_all(a):
    return bool(np.all(a))


class TestCategoricalFilterInvalid:
    def test_copy_new_filter_single(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        orig = FastArray([1, 1, 2, 3, 1])
        assert arr_eq(c._fa, orig)

        f = FastArray([False, True, True, True, True])
        d = Categorical(c, filter=f)
        new = FastArray([0, 1, 2, 3, 1])
        assert arr_eq(d._fa, new)

    def test_copy_new_filter_single2(self):
        c = Categorical([FA(['a', 'a', 'b', 'c', 'a']), arange(5)])
        orig = FastArray([1, 2, 3, 4, 5])
        assert arr_eq(c._fa, orig)

        f = FastArray([False, True, True, True, True])
        d = Categorical(c, filter=f)
        new = FastArray([0, 1, 2, 3, 4])
        assert arr_eq(d._fa, new)

    def test_copy_filter_errors(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'], base_index=0)
        f = FastArray([False, True, True, True, True])
        with pytest.warns(UserWarning):
            d = Categorical(c, filter=f)

    def test_copy_warnings(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        with pytest.warns(UserWarning):
            d = Categorical(c, ordered=False)

        with pytest.warns(UserWarning):
            d = Categorical(c, sort_gb=False)

        with pytest.warns(UserWarning):
            d = Categorical(c, lex=False)

        with pytest.warns(UserWarning):
            d = Categorical(c, base_index=0)

        with pytest.warns(UserWarning):
            d = Categorical(c, dtype=np.int64)

        with pytest.warns(UserWarning):
            d = Categorical(c, unicode=True)

        with pytest.warns(UserWarning):
            d = Categorical(c, invalid='inv')

    def test_filtered_set_name(self):
        c = Categorical([0, 1, 1, 2, 3], ['a', 'b', 'c'])
        assert c[0] == 'Filtered'
        assert c.filtered_name == 'Filtered'
        c.filtered_set_name('FILT')
        assert c[0] == 'FILT'
        assert c.filtered_name == 'FILT'

    # TODO move this into SDS save / load test module and use pytest fixtures
    def test_saveload_invalid(self):
        c = Categorical([0, 1, 1, 2, 3], ['a', 'b', 'c'], invalid='a')
        c.filtered_set_name('FILT')
        ds = Dataset({'catcol': c})
        ds.save(r'riptable/tests/temp/ds')
        ds2 = load_sds(r'riptable/tests/temp/ds')
        c2 = ds2.catcol
        assert c.invalid_category == c2.invalid_category
        assert c.filtered_name == c2.filtered_name
        os.remove(r'riptable/tests/temp/ds.sds')

    def test_isfiltered(self):
        c = Categorical(np.random.choice(4, 100), ['a', 'b', 'c'])
        flt = c.isfiltered()
        eq_z = c._fa == 0
        assert arr_eq(flt, eq_z)

    def test_isnan(self):
        c = Categorical(np.random.choice(4, 100), ['a', 'b', 'c'], invalid='a')
        inv = c.isnan()
        eq_bin = c._fa == 1
        assert arr_eq(inv, eq_bin)

    def test_combine_filter(self):
        c = Categorical(FA([2, 2, 1, 0, 2], dtype=np.int8), ['a', 'b'])
        f = FA([True, True, False, True, False])
        combine_accum1_filter(c._fa, 2, filter=f)

    def test_single_key_filter(self):
        f = FA([True, True, False, True, False])
        pre_c = Categorical(['a', 'a', 'b', 'c', 'a'], filter=f)
        assert pre_c.unique_count == 2

        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        assert c.unique_count == 3
        d = c.filter(filter=f)
        assert pre_c.unique_count == d.unique_count
        assert arr_eq(pre_c._fa, d._fa)

    def test_multikey_filter(self):
        f = FA([True, True, False, True, False])
        pre_c = Categorical([FA(['a', 'a', 'b', 'c', 'a']), arange(5)], filter=f)
        assert pre_c.unique_count == 3

        c = Categorical([FA(['a', 'a', 'b', 'c', 'a']), arange(5)])
        assert c.unique_count == 5
        d = c.filter(filter=f)
        assert pre_c.unique_count == d.unique_count
        assert arr_eq(pre_c._fa, d._fa)

    def test_categorical_empty(self):
        c = Categorical(zeros(10, dtype=np.int8), ['a', 'b', 'c'])
        assert c.unique_count == 3

        d = c.filter()
        assert d.unique_count == 0
        assert len(d.category_array) == 0

    def test_categorical_full(self):
        f = full(5, True)
        c = Categorical([FA(['a', 'a', 'b', 'c', 'a']), arange(5)])
        d = c.filter(filter=f)
        assert arr_eq(c._fa, d._fa)

    def test_filter_deep_copy(self):
        f = FA([True, True, False, True, False])
        c = Categorical([FA(['a', 'a', 'b', 'c', 'a']), arange(5)])
        assert c.unique_count == 5
        d = c.filter(filter=f)
        assert d.unique_count == 3
        assert c.unique_count == 5

    def test_filter_base_zero(self):
        f = FA([True, True, False, True, False])
        c = Categorical(['a', 'a', 'b', 'c', 'a'], filter=f)
        c_zero = Categorical(['a', 'a', 'b', 'c', 'a'], base_index=0)
        with pytest.warns(UserWarning):
            d = c_zero.filter(filter=f)
        assert arr_eq(c._fa, d._fa)
        assert c.unique_count == d.unique_count

    def test_pre_vs_post(self):
        # unique item removed
        arr = np.random.choice(['a', 'b', 'c'], 50)
        filter = arr != 'b'
        c = Categorical(arr)
        c_pre = Categorical(arr, filter=filter)
        c_post = c.filter(filter=filter)
        c_copy = Categorical(c, filter=filter)

        assert arr_eq(c_pre.category_array, c_post.category_array)
        assert arr_eq(c_pre._fa, c_post._fa)

        assert arr_eq(c_pre.category_array, c_copy.category_array)
        assert arr_eq(c_pre._fa, c_copy._fa)

        # same uniques
        arr = np.random.choice(['a', 'b', 'c'], 50)
        filter = ones(50, dtype=np.bool)
        filter[:5] = False
        c = Categorical(arr)
        c_pre = Categorical(arr, filter=filter)
        c_post = c.filter(filter=filter)
        c_copy = Categorical(c, filter=filter)

        assert arr_eq(c_pre.category_array, c_post.category_array)
        assert arr_eq(c_pre._fa, c_post._fa)

        assert arr_eq(c_pre.category_array, c_copy.category_array)
        assert arr_eq(c_pre._fa, c_copy._fa)

    def test_expand_array_empty(self):
        arr = np.random.choice(['a', 'b', 'c'], 50)
        c = Categorical(arr)
        c2 = c.filter(filter=full(50, False))
        assert c2.unique_count == 0
        assert len(c2.category_array) == 0
        expanded = c2.expand_array
        assert arr_eq(expanded, c2.filtered_name)

    def test_expand_dict_empty(self):
        c = Categorical([arange(5), np.array(['a', 'b', 'c', 'd', 'e'])])
        c2 = c.filter(filter=full(5, False))
        assert c2.unique_count == 0

        d = list(c2.expand_dict.values())
        assert arr_all(d[0].isnan())
        assert arr_eq(d[1], c2.filtered_name)

    def test_gbc_multikey(self):
        names = FA(['brian', 'brian', 'adam', 'charlie', 'edward', 'brian'])
        names.set_name('names')
        nums = FA([1, 1, 2, 3, 4, 1])
        nums.set_name('nums')

        f = FA([False, False, True, True, True, False])

        data = arange(6)

        c_multi = Categorical([names, nums])
        c_result = c_multi.sum(data)

        ds = Dataset({'names': names, 'nums': nums})

        ds_result = ds.gbu(['names', 'nums']).sum(data)

        dsc = Dataset({'catcol': c_multi})
        dsc_result = dsc.gbu('catcol').sum(data)

        assert c_result.equals(ds_result), f'did not match for \n{c_result}\n{ds_result}'
        assert ds_result.equals(dsc_result), f'did not match for \n{ds_result}\n{dsc_result}'

    def test_enum_filter(self):
        codes = np.random.choice([10, 20, 30, 40], 50)
        d = {10: 'aaa', 20: 'bbb', 30: 'ccc'}
        c = Categorical(codes, d)
        data = arange(50)

        # filter to go to groupby
        reg = c.sum(data)
        app = c.apply(sum, data)
        assert reg.equals(app)

        # keep enum as enum
        as_arr = c.filter(None)

        for i in range(len(as_arr)):
            assert c[i] == as_arr[i]

        # post filter enum
        codes = FA([10, 10, 20, 30, 10])
        d = {10: 'aaa', 20: 'bbb', 30: 'ccc'}
        f = FA([True, True, False, True, True])
        c = Categorical(codes, d)

        c2 = c.as_singlekey().filter(f)
        assert c2.unique_count == 2
        c2.filtered_set_name('FLT')
        assert c2.unique_count == 2
        assert c2[2] == 'FLT'

        c = Cat([10, 20, 30] * 3, {10: 'A', 20: 'B', 30: 'C'})
        count = c.filter(c == 'A').count()['Count']
        assert np.all(count == [3, 6])

    def test_gbc_filter(self):
        pass

    def test_slice_empty_gb(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        c = c[:0]
        with pytest.raises(ValueError):
            _ = c.sum(arange(5))

        codes = FA([10, 10, 20, 30, 10])
        d = {10: 'aaa', 20: 'bbb', 30: 'ccc'}
        c = Categorical(codes, d)
        c = c[:0]
        with pytest.raises(ValueError):
            _ = c.sum(arange(5))


# INVAID vs. FILTERED checklist:
# c = Cat(values)
# c = Cat(num values)
# c = Cat(values, cats)
# c = Cat(num values, num cats)
# ^^^^^^^
# AND all + filter


# test:
# isnan()
# isnotnan()
# isfiltered()
# isnotfiltered()

# For methods that fill with nan, fill with filtered bin 0, or fill with inv category bin?

# What if no invalid category specified?
# Is there a default for invalid category, or does it get set to None?
# Does numeric fall back on invalid for that dtype? Does string fall back on ''?
# How are filtered bins represented as string?

# Is the inv category being preserved during:
# -copy?
# -slice?
# -index?
# -ismember/mbget/hstack, etc.?
# -sds load (in meta data?)
