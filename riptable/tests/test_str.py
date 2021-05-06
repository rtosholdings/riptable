import pytest
parametrize = pytest.mark.parametrize

import riptable as rt
from riptable import *


from numpy.testing import assert_array_equal
from ..testing.array_assert import assert_array_or_cat_equal



SYMBOLS = ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM']
PARALLEL_MULTIPLIER = 2000
NB_PARALLEL_SYMBOLS = SYMBOLS * PARALLEL_MULTIPLIER
assert len(NB_PARALLEL_SYMBOLS) >= FAString._APPLY_PARALLEL_THRESHOLD


class TestStr:

    cat_symbol = Cat(np.tile(np.arange(len(SYMBOLS) + 1), 3), SYMBOLS)

    def test_cat(self):
        arrsize = 200
        symbol = Cat(1 + arange(arrsize) % len(SYMBOLS), SYMBOLS)
        assert_array_equal(symbol.expand_array.str.startswith('AAPL'), symbol.str.startswith(
            'AAPL'
        ))

    def test_cat_filtered(self):
        assert_array_equal(self.cat_symbol.expand_array.str.startswith('IBM'), self.cat_symbol.str.startswith(
            'IBM'
        ))

    def test_lower(self):
        result = FAString(SYMBOLS).lower
        assert (result.tolist() == [s.lower() for s in SYMBOLS])

    def test_lower_cat(self):
        result = self.cat_symbol.str.lower
        expected = Cat(self.cat_symbol.ikey, [s.lower() for s in SYMBOLS])
        assert_array_or_cat_equal(result, expected, relaxed_cat_check=True)

    def test_upper(self):
        result = FAString(SYMBOLS).upper
        assert (result.tolist() == [s.upper() for s in SYMBOLS])

    def test_upper_cat(self):
        result = self.cat_symbol.str.upper
        expected = Cat(self.cat_symbol.ikey, [s.upper() for s in SYMBOLS])
        assert_array_or_cat_equal(result, expected, relaxed_cat_check=True)

    @parametrize("str2, expected", [
        ('bb', [False, False, True]),
        ('ba', [False, True, False]),
    ])
    def test_endswith(self, str2, expected):
        result = FAString(['abab', 'ababa', 'abababb']).endswith(str2)
        assert_array_equal(result, expected)

    @parametrize("str2, expected", [
        ('A', [True, True, False, False, False]),
        ('AA', [True, False, False, False, False]),
        ('', [True] * 5),
        ('AAA', [False] * 5),
        ('AAPL', [True] + [False] * 4)
    ])
    def test_contains(self, str2, expected):
        result = FAString(SYMBOLS).contains(str2)
        assert_array_equal(result, expected)

        result = FAString(NB_PARALLEL_SYMBOLS).contains(str2)
        assert_array_equal(result, expected * PARALLEL_MULTIPLIER)

    @parametrize("str2, expected", [
        ('A', [0, 0, -1, -1, -1]),
        ('AA', [0, -1, -1, -1, -1]),
        ('AAPL', [0, -1, -1, -1, -1]),
        ('', [0] * 5),
        ('AAA', [-1] * 5),
        ('B', [-1, -1, 1, -1, 1])
    ])
    def test_strstr(self, str2, expected):
        result = FAString(SYMBOLS).strstr(str2)
        assert_array_equal(result, expected)

        result = FAString(NB_PARALLEL_SYMBOLS).strstr(str2)
        assert_array_equal(result, expected * PARALLEL_MULTIPLIER)

    def test_strstr_cat(self):
        result = self.cat_symbol.str.strstr('A')
        inv = rt.INVALID_DICT[np.dtype(result.dtype).num]
        expected = rt.FA([inv, 0, 0, -1, -1, -1], dtype=result.dtype).tile(3)
        assert_array_equal(result, expected)

    def test_strlen_cat(self):
        result = self.cat_symbol.str.strlen
        inv = rt.INVALID_DICT[np.dtype(result.dtype).num]
        expected = rt.FA([inv, 4, 4, 2, 4, 3], dtype=result.dtype).tile(3)
        assert_array_equal(result, expected)

    def test_strlen_parallel(self):
        result = rt.FAString(NB_PARALLEL_SYMBOLS).strlen
        expected = rt.FA([4, 4, 2, 4, 3], dtype=result.dtype).tile(PARALLEL_MULTIPLIER)
        assert_array_equal(result, expected)

    def test_strpbrk_cat(self):
        result = self.cat_symbol.str.strpbrk('PZG')
        inv = rt.INVALID_DICT[np.dtype(result.dtype).num]
        expected = rt.FA([inv, 2, 2, -1, 0, -1] * 3)
        assert_array_equal(result, expected)

    regex_match_test_cases = parametrize('regex, expected', [
        ('.', [True] * 5),
        ('\.', [False] * 5),
        ('A', [True, True, False, False, False]),
        ('[A|B]', [True, True, True, False, True]),
        ('B$', [False, False, True, False, False]),
    ])

    @regex_match_test_cases
    def test_regex_match(self, regex, expected):
        fa = FA(SYMBOLS)
        assert_array_equal(fa.str.regex_match(regex), expected)

    @regex_match_test_cases
    def test_regex_match_cat(self, regex, expected):
        cat = Cat(SYMBOLS * 2)   # introduce duplicity to test ikey properly
        assert_array_equal(cat.str.regex_match(regex), expected * 2)

    substr_test_cases = parametrize("start_stop", [
        (0, 1),
        (0, 2),
        (1, 3),
        (1, -1),
        (-2, 3),
        (2,),
        (-1,),
        (-3, -1),
        (-1, 1),
    ])

    @substr_test_cases
    def test_substr(self, start_stop):
        expected = [s[slice(*start_stop)] for s in SYMBOLS]
        result = FAString(SYMBOLS).substr(*start_stop)
        assert (expected == result.tolist())

    @substr_test_cases
    def test_substr_bytes(self, start_stop):
        expected = [s[slice(*start_stop)] for s in SYMBOLS]
        result = FAString(FastArray(SYMBOLS)).substr(*start_stop)
        assert_array_equal(FastArray(expected), result)

    @substr_test_cases
    def test_substr_cat(self, start_stop):
        result = self.cat_symbol.str.substr(*start_stop)
        expected = [s[slice(*start_stop)] for s in SYMBOLS]
        expected = Categorical(self.cat_symbol.ikey, expected, base_index=1)
        assert_array_or_cat_equal(expected, result, relaxed_cat_check=True)
        # check categories are unique
        assert len(set(result.category_array)) == len(result.category_array)

    @parametrize('position', [0, 1, -1, -2, 3])
    def test_char(self, position):
        result = FAString(SYMBOLS).char(position)
        expected = [s[position] if position < len(s) else '' for s in SYMBOLS]
        assert result.tolist() == expected

    @parametrize('position', [0, 1, -1, -2])
    def test_char_cat(self, position):
        result = self.cat_symbol.str.char(position)
        expected = Categorical(self.cat_symbol.ikey, [s[position] for s in SYMBOLS],
                               base_index=1)
        assert_array_or_cat_equal(expected, result, relaxed_cat_check=True)
        # check categories are unique
        assert len(set(result.category_array)) == len(result.category_array)

    def test_char_array_position(self):
        position = [-1, 2, 0, 1, 2]
        result = FAString(SYMBOLS).char(position)
        expected = [s[pos] for s, pos in zip(SYMBOLS, position)]
        assert result.tolist() == expected

    @parametrize('position', [
        -3, 6, [0, 0, 0, 0, -5]
    ])
    def test_char_failure(self, position):
        with pytest.raises(ValueError, match='Position -?\d out of bounds'):
            FAString(SYMBOLS).char(position)
