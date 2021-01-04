import unittest
import pytest

from riptable import *


def arr_eq(a, b):
    return bool(np.all(a == b))


def arr_all(a):
    return bool(np.all(a))


SYMBOLS = ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM']


class StrTest(unittest.TestCase):
    def test_cat(self):
        arrsize = 200
        symbol = Cat(1 + arange(arrsize) % len(SYMBOLS), SYMBOLS)
        result = symbol.expand_array.str.startswith('AAPL') == symbol.str.startswith(
            'AAPL'
        )
        self.assertTrue(np.all(result))

    def test_basic(self):
        result = FAString(['abab', 'ababa', 'abababb']).endswith('bb')
        self.assertTrue(np.all(result == [False, False, True]))
        result = FAString(['abab', 'ababa', 'abababb']).endswith('ba')
        self.assertTrue(np.all(result == [False, True, False]))


regexpb_test_cases = pytest.mark.parametrize('str2, expected', [
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


if __name__ == '__main__':
    tester = unittest.main()
