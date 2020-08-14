# $Id: //Depot/Source/SFW/riptable/Python/core/riptable/tests/test_multiset.py#4 $
import unittest
import numpy as np

# from io import StringIO
from riptable import Struct
from riptable import Dataset
from riptable import Multiset


class Multiset_Test(unittest.TestCase):
    @staticmethod
    def get_basic_dataset(keyfield_value=None, nr=10):
        ds = Dataset(
            {
                _k: list(range(_i * nr, (_i + 1) * nr))
                for _i, _k in enumerate('abcdefghijklmnop')
            }
        )
        if keyfield_value is not None:
            ds.keyfield = keyfield_value
        return ds

    def test_ctor_01(self):
        dd = {_k: self.get_basic_dataset(_i) for _i, _k in enumerate('bac')}
        ms = Multiset(dd)
        self.assertEqual(list(ms.keys()), ['b', 'a', 'c'])
        for _i, _k in enumerate('bac'):
            self.assertEqual(ms[_k].keyfield[0], _i)
            self.assertEqual(getattr(ms, _k).keyfield[0], _i)
        with self.assertRaises(TypeError):
            _ = Multiset(ms)
        with self.assertRaises(TypeError):
            _ = Multiset(list(dd.items()))
        assert Multiset(None).shape == (
            0,
            0,
        ), f"Multiset default argument for 'None' input value should create an empty Multiset."
        ms = Multiset()
        self.assertEqual(ms.shape, (0, 0))
        ms = Multiset({})
        self.assertEqual(ms.shape, (0, 0))

    def test_ctor_02(self):
        dd = {_k: self.get_basic_dataset(_i) for _i, _k in enumerate('bac')}
        dd['d'] = self.get_basic_dataset(3)
        ms = Multiset(dd)
        self.assertEqual(list(ms.keys()), ['b', 'a', 'c', 'd'])
        dd['d'] = Multiset(
            {_k: self.get_basic_dataset(_i) for _i, _k in enumerate('bac')}
        )
        ms = Multiset(dd)
        self.assertEqual(list(ms.keys()), ['b', 'a', 'c', 'd'])
        # Number of rows in contained datasets must match
        dd['d'] = self.get_basic_dataset(3, nr=9)
        with self.assertRaises(ValueError):
            _ = Multiset(dd)
        # Number of rows in contained datasets and multisets must match
        dd['d'] = Multiset({})
        with self.assertRaises(ValueError):
            _ = Multiset(dd)
        dd['d'] = Multiset(
            {_k: self.get_basic_dataset(_i, nr=9) for _i, _k in enumerate('bac')}
        )
        with self.assertRaises(ValueError):
            _ = Multiset(dd)
        # Type of input must be valid
        dd['d'] = Struct({})
        with self.assertRaises(TypeError):
            _ = Multiset(dd)
        dd['d'] = None
        with self.assertRaises(TypeError):
            _ = Multiset(dd)
        dd['d'] = 1
        with self.assertRaises(TypeError):
            _ = Multiset(dd)

    def test_modify(self):
        dd = {_k: self.get_basic_dataset(_i) for _i, _k in enumerate('bac')}
        ms = Multiset(dd)
        self.assertEqual(list(ms.keys()), list('bac'))
        with self.assertRaises(ValueError):
            ms.d = Dataset({})
        ms.d = Dataset({'a': range(10)})
        self.assertEqual(list(ms.keys()), list('bacd'))
        with self.assertRaises(TypeError):
            ms.e = Struct({})
        with self.assertRaises(TypeError):
            ms.d = Struct({})
        self.assertEqual(list(ms.keys()), list('bacd'))
        self.assertIsInstance(ms.d, Dataset)
        ms._lock()
        with self.assertRaises(AttributeError):
            ms.e = Dataset({})
        ms._unlock()
        with self.assertRaises(ValueError):
            ms.e = Dataset({})
        ms.e = Dataset({'a': range(10)})
        self.assertEqual(list(ms.keys()), list('bacde'))
        self.assertEqual(ms.e.shape, (10, 1))
        self.assertIs(ms.e, ms[4])
        self.assertEqual(ms[4].shape, (10, 1))
        # Added Dataset must have same number of rows
        with self.assertRaises(ValueError):
            ms['f'] = Dataset({'a': 1, 'b': 2})
        ms['f'] = Dataset({'a': range(10), 'b': range(10)})
        ms[5] = Dataset({'a': [3] * 10, 'b': [4] * 10})
        with self.assertRaises(IndexError):
            ms[6] = Dataset({'a': [3] * 10, 'b': [4] * 10})
        self.assertEqual(list(ms.keys()), list('bacdef'))
        with self.assertRaises(IndexError):
            ms[['e', 'g']] = [Dataset({'a': 5, 'b': 6}), Dataset({'a': 7, 'b': 8})]
        ms[['e', 'f']] = [
            Dataset({'a': [5] * 10, 'b': [6] * 10}),
            Dataset({'a': [7] * 10, 'b': [8] * 10}),
        ]

    def test_multiget(self):
        # First testing case where all the contained Datasets are different:
        dd1 = {_k: self.get_basic_dataset(_i) for _i, _k in enumerate('ABC')}
        dd1['A'].col_remove(list('abc'))
        dd1['B'].col_remove(list('def'))
        dd1['D'] = Multiset({'SUB': dd1['C']})
        ms1 = Multiset(dd1)
        ms2 = ms1.multiget(list('aeg'))
        self.assertEqual(list(ms1.keys()), list('ABCD'))
        self.assertEqual(list(ms2.keys()), list('ABC'))
        self.assertEqual(list(ms2.A.keys()), list('eg'))
        self.assertEqual(list(ms2.B.keys()), list('ag'))
        self.assertEqual(list(ms2.C.keys()), list('aeg'))
        ms3 = ms1.multiget(slice(2, 15, 4))
        self.assertEqual(list(ms3.keys()), list('ABC'))
        self.assertEqual(list(ms3.A.keys()), list('fjn'))
        self.assertEqual(list(ms3.B.keys()), list('cjn'))
        self.assertEqual(list(ms3.C.keys()), list('cgko'))
        # Now, when they all match:
        dd2 = {_k: self.get_basic_dataset(_i) for _i, _k in enumerate('ABCD')}
        ms4 = Multiset(dd2)
        self.assertEqual(list(ms4.keys()), list('ABCD'))
        ms5 = ms4.multiget(list('cgko'))
        ms6 = ms4.multiget(slice(2, 15, 4))
        filt = np.zeros((ms4.A.get_ncols(),), dtype=bool)
        filt[2:15:4] = True
        ms7 = ms4.multiget(filt)
        for ms in (ms5, ms6, ms7):
            self.assertEqual(list(ms.keys()), list('ABCD'))
            for dskey in 'ABCD':
                self.assertEqual(list(getattr(ms, dskey).keys()), list('cgko'))
        ms8 = ms4.multiget('c')
        ms9 = ms4.multiget(['c'])
        for ms in (ms8, ms9):
            self.assertEqual(list(ms.keys()), list('ABCD'))
            for dskey in 'ABCD':
                self.assertEqual(list(getattr(ms, dskey).keys()), list('c'))

    def test_copy(self):
        dd = {_k: self.get_basic_dataset(_i) for _i, _k in enumerate('bac')}
        ms = Multiset(dd)
        ms1 = ms.copy()
        self.assertEqual(list(ms1.keys()), ['b', 'a', 'c'])
        self.assertTrue((ms1.a == ms.a).all(axis=None))
        self.assertTrue((ms1.b == ms.b).all(axis=None))
        self.assertTrue((ms1.c == ms.c).all(axis=None))
        self.assertFalse(ms1.is_locked())
        ms._lock()
        ms2 = ms.copy()
        self.assertTrue(ms2.is_locked())

    def almost_eq(self, ar1, ar2, places=5):
        for v1, v2 in zip(ar1, ar2):
            self.assertAlmostEqual(v1, v2, places)

    def test_flatten(self):
        accuracy = 7
        message_types = [
            'NEW',
            'EXECUTE',
            'NEW',
            'EXECUTE',
            'EXECUTE',
            'EXECUTE',
            'EXECUTE',
            'CANCEL',
            'EXECUTE',
            'EXECUTE',
            'EXECUTE',
            'CANCEL',
        ]
        order_ids = [1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1]
        milliseconds = [50, 70, 72, 75, 90, 88, 95, 97, 98, 115, 116, 120]
        shares = [0, 200, 0, 500, 100, 400, 100, 0, 300, 150, 150, 0]
        dat = Dataset(
            dict(
                message_type=message_types,
                order_id=order_ids,
                millisecond=milliseconds,
                shares=shares,
            )
        )
        dat = dat[['order_id', 'message_type', 'millisecond', 'shares']]
        gb = dat.groupby('order_id')
        ms1 = gb[['millisecond', 'shares']].aggregate(['sum', 'mean'])
        # Flatten horizontally
        f1 = ms1.flatten()
        self.assertEqual(
            f1.keys(),
            [
                'order_id',
                'Sum_millisecond',
                'Sum_shares',
                'Mean_millisecond',
                'Mean_shares',
            ],
        )
        self.assertTrue((f1.Sum_millisecond == [410, 676]).all(axis=None))
        # Flatten vertically
        f2 = ms1.flatten(horizontal=False)
        self.assertEqual(f2.keys(), ['order_id', 'Column', 'millisecond', 'shares'])
        self.almost_eq(f2.millisecond, [410.0, 676.0, 82.0, 96.57142857], accuracy)
        ms2 = gb[['millisecond', 'shares']].aggregate(['min', 'max'])
        # Flatten multiset containing multisets
        ms3 = Multiset({'ms1': ms1, 'ms2': ms2})
        f3 = ms3.flatten()
        self.assertEqual(
            f3.keys(),
            [
                'order_id',
                'ms1_Sum_millisecond',
                'ms1_Sum_shares',
                'ms1_Mean_millisecond',
                'ms1_Mean_shares',
                'ms2_Min_millisecond',
                'ms2_Min_shares',
                'ms2_Max_millisecond',
                'ms2_Max_shares',
            ],
        )
        self.assertTrue((f3.ms1_Sum_millisecond == [410, 676]).all(axis=None))
        f3v = ms3.flatten(horizontal=False)
        self.assertTrue(
            (
                f3v.Column
                == [
                    b'ms1_Sum',
                    b'ms1_Sum',
                    b'ms1_Mean',
                    b'ms1_Mean',
                    b'ms2_Min',
                    b'ms2_Min',
                    b'ms2_Max',
                    b'ms2_Max',
                ]
            ).all(axis=None)
        )
        # Flatten multiset containing multisets and a dataset
        ds = gb[['millisecond', 'shares']].std()
        ms4 = Multiset({'ms1': ms1, 'ms2': ms2, 'Std': ds})
        f4 = ms4.flatten()
        f4.label_remove()
        self.assertEqual(
            f4.keys(),
            [
                'order_id',
                'ms1_Sum_millisecond',
                'ms1_Sum_shares',
                'ms1_Mean_millisecond',
                'ms1_Mean_shares',
                'ms2_Min_millisecond',
                'ms2_Min_shares',
                'ms2_Max_millisecond',
                'ms2_Max_shares',
                'Std_millisecond',
                'Std_shares',
            ],
        )
        self.assertTrue((f4.ms1_Sum_millisecond == [410, 676]).all(axis=None))
        f4 = ms4.flatten(horizontal=False)
        self.assertAlmostEqual(f4[9, 'millisecond'], 15.4903964, places=5)
        self.assertAlmostEqual(f4[9, 'shares'], 148.4042099, places=5)
        # Flatten multiset containing multisets and a dataset with non-matching column
        ds1 = gb[['shares']].std()
        ms5 = Multiset({'ms1': ms1, 'ms2': ms2, 'Std': ds1})
        f5 = ms5.flatten()
        f5.label_remove()
        self.assertEqual(
            f5.keys(),
            [
                'order_id',
                'ms1_Sum_millisecond',
                'ms1_Sum_shares',
                'ms1_Mean_millisecond',
                'ms1_Mean_shares',
                'ms2_Min_millisecond',
                'ms2_Min_shares',
                'ms2_Max_millisecond',
                'ms2_Max_shares',
                'Std_shares',
            ],
        )
        self.assertTrue((f5.ms1_Sum_millisecond == [410, 676]).all(axis=None))
        with self.assertRaises(ValueError):
            ms5.flatten(horizontal=False)


if __name__ == "__main__":
    tester = unittest.main()
