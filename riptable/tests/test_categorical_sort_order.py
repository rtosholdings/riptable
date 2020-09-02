from enum import IntEnum

from riptable import *


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
enum_codes = (sorted_codes + 1) * 10
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
flt_fa.set_name('floats')
complete_unique_cats = FastArray(['a', 'b', 'c', 'd', 'e'])
unsorted_unique_cats = FastArray(['c', 'e', 'd', 'b', 'a'])
big_cats = FastArray(['string' + str(i) for i in range(2000)])
matlab_codes = (sorted_codes + 1).astype(np.float32)

sorted_float_sum = FastArray([205.09, 242.9, 454.25, 337.27, 325.84])
unsorted_float_sum = FastArray([454.25, 325.84, 337.27, 242.9, 205.09])


class str_enum(IntEnum):
    a = 10
    b = 20
    c = 30
    d = 40
    e = 50


class TestCategoricalFirstOrder:
    def test_strings(self):
        c = Categorical(str_fa)
        result = bool(np.all(c.category_array == complete_unique_cats))
        assert result
        float_sum = c.sum(flt_fa).floats
        result = bool(np.allclose(float_sum, sorted_float_sum))
        assert result

        c = Categorical(str_fa, ordered=False)
        result = bool(np.all(c.category_array == unsorted_unique_cats))
        assert result
        float_sum = c.sum(flt_fa).floats
        result = bool(np.allclose(float_sum, unsorted_float_sum))
        assert result

        c = Categorical(str_fa, complete_unique_cats)
        result = bool(np.all(c.category_array == complete_unique_cats))
        assert result
        float_sum = c.sum(flt_fa).floats
        result = bool(np.allclose(float_sum, sorted_float_sum))
        assert result

        c = Categorical(str_fa, complete_unique_cats, ordered=False)
        result = bool(np.all(c.category_array == complete_unique_cats))
        assert result
        float_sum = c.sum(flt_fa).floats
        result = bool(np.allclose(float_sum, sorted_float_sum))
        assert result

        c = Categorical(str_fa, unsorted_unique_cats, ordered=False, sort_gb=True)
        result = bool(np.all(c.category_array == unsorted_unique_cats))
        assert result
        float_sum = c.sum(flt_fa).floats
        result = bool(np.allclose(float_sum, sorted_float_sum))
        assert result

        c = Categorical(str_fa, unsorted_unique_cats, ordered=False)
        result = bool(np.all(c.category_array == unsorted_unique_cats))
        assert result
        float_sum = c.sum(flt_fa).floats
        result = bool(np.allclose(float_sum, unsorted_float_sum))
        assert result

    def test_multikey(self):
        c = Categorical([str_fa, sorted_codes])
        float_sum = c.sum(flt_fa).floats
        result = bool(np.allclose(float_sum, unsorted_float_sum))
        assert result

        c = Categorical([str_fa, sorted_codes], ordered=False, sort_gb=True)
        float_sum = c.sum(flt_fa).floats
        result = bool(np.allclose(float_sum, sorted_float_sum))
        assert result
        first_arr = c.category_dict['key_0']
        result = bool(np.all(first_arr == unsorted_unique_cats))
        assert result

        c = Categorical([str_fa, sorted_codes])
        float_sum = c.sum(flt_fa).floats
        result = bool(np.allclose(float_sum, unsorted_float_sum))
        assert result

        c = Categorical([str_fa, sorted_codes], ordered=False, sort_gb=True)
        float_sum = c.sum(flt_fa).floats
        result = bool(np.allclose(float_sum, sorted_float_sum))
        assert result
        first_arr = c.category_dict['key_0']
        result = bool(np.all(first_arr == unsorted_unique_cats))
        assert result

    def test_enum(self):
        c = Categorical(enum_codes, str_enum)
        float_sum = c.sum(flt_fa).floats
        result = bool(np.allclose(float_sum, unsorted_float_sum))
        assert result

        c = Categorical(enum_codes, str_enum, sort_gb=True)
        float_sum = c.sum(flt_fa).floats
        result = bool(np.allclose(float_sum, sorted_float_sum))
        assert result

    def test_lex(self):
        # string sort test
        a = arange(100).astype('S')
        c1 = Cat(a, lex=True)
        c2 = Cat(a, ordered=True)
        assert np.all(c1._fa == c2._fa)
        assert np.all(c1.categories() == c2.categories())

        # make sure sorted
        srt = np.sort(c1._categories)
        assert np.all(c1._categories == srt)

        # with filter
        f = logical(arange(100) % 3)
        c1 = Cat(a, lex=True, filter=f)
        c2 = Cat(a, ordered=True, filter=f)
        assert np.all(c1._fa == c2._fa)

        # larger data
        r = np.random.randint(0, 800_000, 2_000_000)
        r1 = Cat(r, lex=True)
        r2 = Cat(r, ordered=True)
        assert np.all(r1._fa == r2._fa)

        # make sure sorted
        srt = np.sort(r1._categories)
        assert np.all(r1._categories == srt)

        f = logical(arange(len(r)) % 3)
        r1 = Cat(r, lex=True, filter=f)
        r2 = Cat(r, ordered=True, filter=f)
        assert np.all(r1._fa == r2._fa)

        # larger data multikey
        q = np.random.randint(0, 800_000, 2_000_000)
        r1 = Cat([r, q], lex=True, filter=f)
        r2 = Cat([r, q], ordered=True, filter=f)
        assert np.all(r1._fa == r2._fa)

        # multikey strings
        r = np.random.randint(0, 80, 200).astype('U')
        q = np.random.randint(0, 80, 200)
        q = q + 999_000
        q = q.astype('U')

        r1 = Cat([q, r], lex=True)
        r2 = Cat([q, r], ordered=True)
        assert np.all(r1._fa == r2._fa)

        # make sure multikey is sorted
        x = r1._categories.copy()
        p1 = x.popitem()[1]
        p2 = x.popitem()[1]

        # the dict pops in reverse order of the sort
        y = r2._categories.copy()
        y1 = y.popitem()[1]
        y2 = y.popitem()[1]
        assert np.all(p1 == y1)
        assert np.all(p2 == y2)

        p = p2 + p1
        srt = np.sort(p)
        assert np.all(p == srt)
