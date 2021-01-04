
import pytest
parametrize = pytest.mark.parametrize

from riptable import *


SYMBOLS = ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM']


class TestStr:

    cat_symbol = Cat(np.tile(np.arange(len(SYMBOLS) + 1), 3), SYMBOLS)

    def test_cat(self):
        arrsize = 200
        symbol = Cat(1 + arange(arrsize) % len(SYMBOLS), SYMBOLS)
        result = symbol.expand_array.str.startswith('AAPL') == symbol.str.startswith(
            'AAPL'
        )
        assert np.all(result)

    def test_cat_filtered(self):
        result = self.cat_symbol.expand_array.str.startswith('IBM') == self.cat_symbol.str.startswith(
            'IBM'
        )
        assert np.all(result)

    @parametrize("str2, expected", [
        ('bb', [False, False, True]),
        ('ba', [False, True, False]),
    ])
    def test_endswith(self, str2, expected):
        result = FAString(['abab', 'ababa', 'abababb']).endswith(str2)
        assert np.array_equal(result, expected)


regexpb_test_cases = parametrize('str2, expected', [
    ('.', [True] * 5),
    ('\.', [False] * 5),
    ('A', [True, True, False, False, False]),
    ('[A|B]', [True, True, True, False, True]),
    ('B$', [False, False, True, False, False]),
])


@regexpb_test_cases
def test_regexpb(str2, expected):
    fa = FA(SYMBOLS)
    assert np.array_equal(fa.str.regexpb(str2), expected)


@regexpb_test_cases
def test_regexpb_cat(str2, expected):
    cat = Cat(SYMBOLS * 2)   # introduce duplicity to test ikey properly
    assert np.array_equal(cat.str.regexpb(str2), expected * 2)
