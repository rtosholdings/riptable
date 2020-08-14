import unittest

from riptable import *


def arr_eq(a, b):
    return bool(np.all(a == b))


def arr_all(a):
    return bool(np.all(a))


class StrTest(unittest.TestCase):
    def test_cat(self):
        arrsize = 200
        symbols = ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM']
        symbol = Cat(1 + arange(arrsize) % len(symbols), symbols)
        result = symbol.expand_array.str.startswith('AAPL') == symbol.str.startswith(
            'AAPL'
        )
        self.assertTrue(np.all(result))

    def test_basic(self):
        result = FAString(['abab', 'ababa', 'abababb']).endswith('bb')
        self.assertTrue(np.all(result == [False, False, True]))
        result = FAString(['abab', 'ababa', 'abababb']).endswith('ba')
        self.assertTrue(np.all(result == [False, True, False]))


if __name__ == '__main__':
    tester = unittest.main()
