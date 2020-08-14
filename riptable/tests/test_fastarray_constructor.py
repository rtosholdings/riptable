import numpy as np
import riptable as rt
import unittest


class FastArrayConstructionTest(unittest.TestCase):
    def assert_equal(self, lv, rv):
        def equal(l, r):
            def isNaN(value):
                return value != value

            def not_bool(value):
                return not isinstance(value, bool) and not isinstance(value, np.bool_)

            if not_bool(l) and not_bool(r):
                l = round(l, 3)
                r = round(r, 3)

            assert (l == r) or (isNaN(l) and isNaN(r))

        if hasattr(lv, '__len__'):
            assert len(lv) == len(rv)
            length = len(lv)
            for i in range(0, length):
                equal(lv[i], rv[i])
        else:
            equal(lv, rv)

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    dataf = [0.11, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def test_nparray(self):
        self.assert_equal(rt.FastArray(np.array(self.data)), np.array(self.data))

    def test_list(self):
        self.assert_equal(rt.FastArray(self.data), np.array(self.data))

    def test_float_list(self):
        arr_fa = rt.FastArray(self.dataf)
        arr_np = np.array(self.dataf)
        self.assert_equal(arr_fa, arr_np)

    def test_nparray_no_recycling(self):
        rt.FastArray._ROFF()
        self.assert_equal(rt.FastArray(np.array(self.data)), np.array(self.data))
        rt.FastArray._RON()

    def test_list_no_recycling(self):
        rt.FastArray._ROFF()
        self.assert_equal(rt.FastArray(self.data), np.array(self.data))
        rt.FastArray._RON()

    def test_float_list_no_recycling(self):
        rt.FastArray._ROFF()
        self.assert_equal(rt.FastArray(np.array(self.dataf)), np.array(self.dataf))
        rt.FastArray._RON()

    def test_nparray_no_threads(self):
        rt.FastArray._TOFF()
        self.assert_equal(rt.FastArray(np.array(self.data)), np.array(self.data))
        rt.FastArray._TON()

    def test_list__no_threads(self):
        rt.FastArray._TOFF()
        self.assert_equal(rt.FastArray(self.data), np.array(self.data))
        rt.FastArray._TON()

    def test_float_list_no_threads(self):
        rt.FastArray._TOFF()
        self.assert_equal(rt.FastArray(np.array(self.dataf)), np.array(self.dataf))
        rt.FastArray._TON()

    def test_nparray_all_off(self):
        rt.FastArray._TOFF()
        rt.FastArray._ROFF()
        self.assert_equal(rt.FastArray(np.array(self.data)), np.array(self.data))
        rt.FastArray._TON()
        rt.FastArray._RON()

    def test_list_all_off(self):
        rt.FastArray._TOFF()
        rt.FastArray._ROFF()
        self.assert_equal(rt.FastArray(self.data), np.array(self.data))
        rt.FastArray._TON()
        rt.FastArray._RON()

    def test_float_list_all_off(self):
        rt.FastArray._TOFF()
        rt.FastArray._ROFF()
        self.assert_equal(rt.FastArray(np.array(self.dataf)), np.array(self.dataf))
        rt.FastArray._TON()
        rt.FastArray._RON()


if __name__ == "__main__":
    tester = unittest.main()
