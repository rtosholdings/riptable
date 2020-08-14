import unittest
import riptable as rt
from numpy.testing import assert_array_equal


class TestBitCount(unittest.TestCase):
    def test_array(self):
        l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 123]
        expected = rt.FastArray([0, 1, 1, 2, 1, 2, 2, 3, 1, 6], dtype='i1')
        for dtype in ['i8', 'u8', 'i4', 'u4', 'i2', 'u2', 'i1', 'u1']:
            data = rt.FastArray(l, dtype=dtype)
            counts = rt.bitcount(data)
            assert_array_equal(counts, expected)

    def test_16bit_array(self):
        l = [0xFD2, 0xFD27]
        expected = rt.FastArray([8, 11], dtype='i1')
        for dtype in ['i2', 'u2']:
            data = rt.FastArray(l, dtype=dtype)
            counts = rt.bitcount(data)
            assert_array_equal(counts, expected)

    def test_bool_array(self):
        arr = rt.FastArray([0, -10, 17], dtype='i1')
        arr_bool_view = arr.view('?')
        counts = rt.bitcount(arr_bool_view)
        expected = rt.FastArray([0, 1, 1], dtype='i1')
        assert_array_equal(counts, expected)

    def test_float_array(self):
        l = [0.1, 0.2]
        for dtype in ['f4', 'f8']:
            data = rt.FastArray(l, dtype=dtype)
            with self.assertRaises(ValueError):
                rt.bitcount(data)

    def test_sliced_array(self):
        data = rt.FastArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 123], dtype='i8')[1::2]
        expected = rt.FastArray([1, 2, 2, 3, 6], dtype='i1')
        counts = rt.bitcount(data)
        assert_array_equal(counts, expected)

    def test_list(self):
        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 123]
        expected = rt.FastArray([0, 1, 1, 2, 1, 2, 2, 3, 1, 6], dtype='i1')
        counts = rt.bitcount(data)
        assert_array_equal(counts, expected)
        bad_data = [0, 1, 1.1]
        with self.assertRaises(ValueError):
            rt.bitcount(bad_data)
        bad_data = [0, 1, 'a']
        with self.assertRaises(ValueError):
            rt.bitcount(bad_data)

    def test_scalar(self):
        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 123]
        expected = [0, 1, 1, 2, 1, 2, 2, 3, 1, 6]
        for n, e in zip(data, expected):
            self.assertEqual(rt.bitcount(n), e)
        for n, e in zip(rt.FastArray(data, dtype='i8'), expected):
            self.assertEqual(rt.bitcount(n), e)
        with self.assertRaises(ValueError):
            rt.bitcount(3.14)
        with self.assertRaises(ValueError):
            rt.bitcount('a')


if __name__ == '__main__':
    unittest.main()
