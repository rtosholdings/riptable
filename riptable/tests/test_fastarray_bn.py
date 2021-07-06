import unittest
import numpy as np

from io import StringIO
from numpy import nan
from numpy.testing import (
    assert_equal,
    assert_array_equal,
)
from riptable import FastArray as FA
from riptable import mask_or, mask_and, mask_andnot, mask_xor
from .utils import redirectStdoutCtx


try:
    # bottleneck optional
    import bottleneck as bn

    class FastArray_BN_Test(unittest.TestCase):
        def test_masking(self):
            np_bools = [
                np.random.randint(low=0, high=2, size=10_000_000).astype(bool)
                for i in range(6)
            ]

            bool_length = len(np_bools)
            result = np_bools[0] + np_bools[1]
            for i in range(2, bool_length):
                result += np_bools[i]

            result2 = mask_or(np_bools)
            assert_array_equal(result, result2)

            result = np_bools[0] * np_bools[1]
            for i in range(2, bool_length):
                result *= np_bools[i]

            result2 = mask_and(np_bools)
            assert_array_equal(result, result2)

            result = np_bools[0] ^ np_bools[1]
            for i in range(2, bool_length):
                result ^= np_bools[i]

            result2 = mask_xor(np_bools)
            assert_array_equal(result, result2)

            result3 = mask_xor(result2, result, result2, result)
            assert_equal(result3.sum(), 0)

            result3 = mask_xor(result2, result)
            assert_equal(result3.sum(), 0)

            result3 = mask_xor(result2, result, result2)
            assert_array_equal(result, result3)

            result = np_bools[0] * ~np_bools[1]
            for i in range(2, bool_length):
                result = result * ~np_bools[i]
            result2 = mask_andnot(np_bools)
            assert_array_equal(result, result2)

        def test_noncontig(self):
            # no longer a warning
            a = FA([1, 2, 3, 4, 5, 6])
            b = FA(a[::2])
            assert_array_equal(b, [1, 3, 5])

        def test_map(self):
            a = FA([1, 1, 1, 2, 2, 2])
            d = {1: 10, 2: 20}
            c = a.map(d)
            assert_array_equal(c, [10, 10, 10, 20, 20, 20])

        def test_push(self):
            a = FA([5, nan, nan, 6, nan])
            b = a.push()
            assert_array_equal(b, [5, 5, 5, 6, 6])

        def test_rank(self):
            a = FA([nan, 2, 2, 3])
            b = a.nanrankdata()
            assert_array_equal(b, [nan, 1.5, 1.5, 3.0])
            a = FA([0, 2, 2, 3])
            b = a.rankdata()
            assert_array_equal(b, [1, 2.5, 2.5, 4])

        def test_replace(self):
            a = FA([0, 2, 2, 3])
            b = a.replace(2, 1)
            assert_array_equal(b, [0, 1, 1, 3])

        def test_partition(self):
            a = FA([1, 0, 3, 4, 2])
            b = a.partition2(kth=2)
            assert_array_equal(b, [1, 0, 2, 4, 3])

            a = FA([10, 0, 30, 40, 20])
            b = a.argpartition2(kth=2)
            assert_array_equal(b, [0, 1, 4, 3, 2])

        def test_out(self):
            a = FA(np.arange(100))
            c = a
            b = FA(np.arange(100.0))
            np.add(b, b, out=a)
            assert_array_equal(a, c)
            # this is not supported in numpy and also not in riptable anymore
            # np.floor(b,out=a)
            # assert_array_equal(a.astype(b.dtype),b)

        def test_ledger(self):
            self.maxDiff = None
            sio = StringIO()
            with redirectStdoutCtx(sio):
                FA._V1()
                FA._V2()
                FA._OFF()
                FA._ON()
                FA._TOFF()
                FA._TON()
                FA._LON()
                FA._ROFF()
                FA._RON()
                FA._RDUMP()
                FA._LOFF()
                FA._LDUMP()
                FA._LCLEAR()
                FA._GCNOW()
                FA._GCSET(100)
                FA.Verbose = 3
                FA._V1()
                a = FA([1, 2, 3])
                b = a.copy()
                a += a
                del a
                a = np.arange(10000)
                a = FA(a)
                a = a + a
                a = a + a
                b = a._np
                del b
                del a
                FA._V0()
except:
    pass


if __name__ == "__main__":
    tester = unittest.main()
