import pytest
from riptable import *


def arr_eq(a, b):
    return bool(np.all(a == b))


def arr_all(a):
    return bool(np.all(a))


class TestCategoricalSetItem:
    def test_setitem_array_bool(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        b = logical(arange(5) % 2)
        c[b] = 'b'
        assert arr_eq(c._fa, FastArray([1, 2, 2, 2, 1]))
        with pytest.raises(ValueError):
            c[b] = 'f'
        c.auto_add_on()
        c[b] = 'f'
        assert arr_eq(c._fa, FastArray([1, 4, 2, 4, 1]))

    def test_setitem_array_singleint(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        c[1] = 'b'
        c[3] = 'b'
        assert arr_eq(c._fa, FastArray([1, 2, 2, 2, 1]))
        with pytest.raises(ValueError):
            c[1] = 'f'
        c.auto_add_on()
        c[1] = 'f'
        c[3] = 'f'
        assert arr_eq(c._fa, FastArray([1, 4, 2, 4, 1]))

    def test_setitem_array_index(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        b = bool_to_fancy(logical(arange(5) % 2))
        c[b] = 'b'
        assert arr_eq(c._fa, FastArray([1, 2, 2, 2, 1]))
        with pytest.raises(ValueError):
            c[b] = 'f'
        c.auto_add_on()
        c[b] = 'f'
        assert arr_eq(c._fa, FastArray([1, 4, 2, 4, 1]))

    def test_setitem_array_string(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        c['a'] = 'b'
        assert arr_eq(c._fa, FastArray([2, 2, 2, 3, 2]))
        with pytest.raises(ValueError):
            c['b'] = 'f'
        c.auto_add_on()
        c['b'] = 'f'
        assert arr_eq(c._fa, FastArray([4, 4, 4, 3, 4]))

    def test_setitem_array_slice(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        c[:2] = 'b'
        assert arr_eq(c._fa, FastArray([2, 2, 2, 3, 1]))
        with pytest.raises(ValueError):
            c[:2] = 'f'
        c.auto_add_on()
        c[:2] = 'f'
        assert arr_eq(c._fa, FastArray([4, 4, 2, 3, 1]))

    def test_setitem_mapping_bool(self):
        idx = FastArray([1, 1, 2, 3, 1])
        codes = {'a': 1, 'b': 2, 'c': 3}
        c = Categorical(idx, codes)
        b = logical(arange(5) % 2)
        c[b] = 'b'
        assert arr_eq(c._fa, FastArray([1, 2, 2, 2, 1]))
        with pytest.raises(ValueError):
            c[b] = 'f'
        c.mapping_add(4, 'f')
        c[b] = 'f'
        assert arr_eq(c._fa, FastArray([1, 4, 2, 4, 1]))

    def test_setitem_mapping_singleint(self):
        idx = FastArray([1, 1, 2, 3, 1])
        codes = {'a': 1, 'b': 2, 'c': 3}
        c = Categorical(idx, codes)
        c[1] = 'b'
        c[3] = 'b'
        assert arr_eq(c._fa, FastArray([1, 2, 2, 2, 1]))
        with pytest.raises(ValueError):
            c[1] = 'f'
        c.mapping_add(4, 'f')
        c[1] = 'f'
        c[3] = 'f'
        assert arr_eq(c._fa, FastArray([1, 4, 2, 4, 1]))

    def test_setitem_mapping_index(self):
        idx = FastArray([1, 1, 2, 3, 1])
        codes = {'a': 1, 'b': 2, 'c': 3}
        c = Categorical(idx, codes)
        b = bool_to_fancy(logical(arange(5) % 2))
        c[b] = 'b'
        assert arr_eq(c._fa, FastArray([1, 2, 2, 2, 1]))
        with pytest.raises(ValueError):
            c[b] = 'f'
        c.mapping_add(4, 'f')
        c[b] = 'f'
        assert arr_eq(c._fa, FastArray([1, 4, 2, 4, 1]))

    def test_setitem_mapping_string(self):
        idx = FastArray([1, 1, 2, 3, 1])
        codes = {'a': 1, 'b': 2, 'c': 3}
        c = Categorical(idx, codes)
        c['a'] = 'b'
        assert arr_eq(c._fa, FastArray([2, 2, 2, 3, 2]))
        with pytest.raises(ValueError):
            c['b'] = 'f'
        c.mapping_add(4, 'f')
        c['b'] = 'f'
        assert arr_eq(c._fa, FastArray([4, 4, 4, 3, 4]))

    def test_setitem_numeric_bool(self):
        c = Categorical([1.23, 1.23, 4.56, 7.89, 1.23])
        b = logical(arange(5) % 2)
        c[b] = 4.56
        assert arr_eq(c._fa, FastArray([1, 2, 2, 2, 1]))
        with pytest.raises(ValueError):
            c[b] = 10.1112
        c.auto_add_on()
        c[b] = 10.1112
        assert arr_eq(c._fa, FastArray([1, 4, 2, 4, 1]))

    def test_setitem_numeric_singleint(self):
        c = Categorical([1.23, 1.23, 4.56, 7.89, 1.23])
        c[1] = 4.56
        c[3] = 4.56
        assert arr_eq(c._fa, FastArray([1, 2, 2, 2, 1]))
        with pytest.raises(ValueError):
            c[1] = 10.1112
        c.auto_add_on()
        c[1] = 10.1112
        c[3] = 10.1112
        assert arr_eq(c._fa, FastArray([1, 4, 2, 4, 1]))

    def test_setitem_numeric_index(self):
        c = Categorical([1.23, 1.23, 4.56, 7.89, 1.23])
        b = bool_to_fancy(logical(arange(5) % 2))
        c[b] = 4.56
        assert arr_eq(c._fa, FastArray([1, 2, 2, 2, 1]))
        with pytest.raises(ValueError):
            c[b] = 10.1112
        c.auto_add_on()
        c[b] = 10.1112
        assert arr_eq(c._fa, FastArray([1, 4, 2, 4, 1]))

    def test_setitem_numeric_string(self):
        c = Categorical([1.23, 1.23, 4.56, 7.89, 1.23])
        c[1.23] = 4.56
        assert arr_eq(c._fa, FastArray([2, 2, 2, 3, 2]))
        with pytest.raises(ValueError):
            c[4.56] = 10.1112
        c.auto_add_on()
        c[4.56] = 10.1112
        assert arr_eq(c._fa, FastArray([4, 4, 4, 3, 4]))

    def test_setitem_filter(self):
        data = Dataset({'First': Cat('A B A B A B'.split())})
        data.Second = Cat('B A B A B A'.split())
        data.Third = data.First.copy()
        f = data.Third == 'A'
        data.First[0] = 2
        assert not arr_eq(data.First._fa, data.Third._fa)
        data.First[f] = data.Second[f]
        assert not arr_eq(data.First._fa, data.Third._fa)
        assert arr_eq(data.First._fa, FastArray([2, 2, 2, 2, 2, 2]))
