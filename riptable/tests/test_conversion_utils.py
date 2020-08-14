import unittest
import numpy
from riptable import Dataset, Categorical
from riptable.rt_datetime import DateTimeNano
from riptable.Utils.conversion_utils import (
    dataset_as_matrix,
    numpy2d_to_dict,
    dset_dict_to_list,
    append_dataset_dict,
    possibly_convert_to_nanotime,
    numpy_array_to_dict,
)


class Conversion_Utility_Test(unittest.TestCase):
    def test_as_matrix(self):
        error_tol = 0.00001
        ds = Dataset({'A': [1.2, 3.1, 9.6], 'B': [-1.6, 2.7, 4.6]})
        X, _ = dataset_as_matrix(ds)
        self.assertIsInstance(X, numpy.ndarray)
        self.assertEqual(X.shape[0], ds.shape[0])
        self.assertEqual(X.shape[1], ds.shape[1])  # we may break this later
        self.assertTrue((numpy.abs(ds.A._np - X[:, 0]) < error_tol).all())
        self.assertTrue((numpy.abs(ds.B._np - X[:, 1]) < error_tol).all())

    def test_as_matrix_metadata(self):
        error_tol = 0.00001
        ds = Dataset(
            {
                'A': ['EXCH1', 'EXCH2', 'EXCH1', 'EXCH3', 'EXCH3'],
                'B': [-1.6, 2.7, 4.6, 5.7, 8.9],
                'C': Categorical([0, 0, 1, 0, 2], ['CPTYA', 'CPTYB', 'CPTYC']),
            }
        )
        X, X_data = dataset_as_matrix(ds)
        self.assertIsInstance(X, numpy.ndarray)
        self.assertEqual(X.shape[0], ds.shape[0])
        self.assertEqual(X.shape[1], ds.shape[1])  # we may break this later
        self.assertEqual(X_data['A']['dtype'], ds.A.dtype)
        self.assertEqual(X_data['B']['dtype'], ds.B.dtype)
        self.assertEqual(X_data['C']['dtype'], ds.C.dtype)
        self.assertEqual(X_data['A']['is_categorical'], False)
        self.assertEqual(X_data['B']['is_categorical'], False)
        self.assertEqual(X_data['C']['is_categorical'], True)
        self.assertTrue(
            (numpy.abs(X[:, 0] - numpy.array([0., 1., 0., 2., 2.])) < error_tol).all(),
            msg=f"got {X[:, 0]}"
        )
        self.assertTrue(
            (numpy.abs(X[:, 2] - numpy.array([0, 0, 1, 0, 2])) < error_tol).all(),
            msg=f"got {X[:, 2]}"
        )
        self.assertTrue(
            (X_data['A']['category_values'][numpy.array([0, 1, 0, 2, 2])] == ds.A).all(),
            msg=f"X_data {X_data['A']['category_values'][numpy.array([0, 1, 0, 2, 2])]}\nds.A {ds.A}"
        )

    def test_as_matrix_int(self):
        error_tol = 0.00001
        ds = Dataset(
            {
                _k: list(range(_i * 10, (_i + 1) * 10))
                for _i, _k in enumerate('ABCDEFGHIJKLMNOP')
            }
        )
        X, _ = dataset_as_matrix(ds)
        self.assertIsInstance(X, numpy.ndarray)
        self.assertEqual(X.shape[0], ds.shape[0])
        self.assertEqual(X.shape[1], ds.shape[1])  # we may break this later
        self.assertTrue((numpy.abs(ds.A._np - X[:, 0]) < error_tol).all())
        self.assertTrue((numpy.abs(ds.B._np - X[:, 1]) < error_tol).all())

    def test_numpy_array_to_dict(self):
        arr = numpy.arange(12).reshape((3, 4)).transpose()
        cols = ['A', 'C', 'B']
        dd = numpy_array_to_dict(arr, cols)
        self.assertEqual(list(dd), cols)
        self.assertTrue((dd['A'] == numpy.arange(0, 4)).all())
        self.assertTrue((dd['C'] == numpy.arange(4, 8)).all())
        self.assertTrue((dd['B'] == numpy.arange(8, 12)).all())
        arr = numpy.array(
            [(1.0, 'Q'), (-3.0, 'Z')], dtype=[('x', numpy.float64), ('y', 'S1')]
        )
        dd = numpy_array_to_dict(arr)
        self.assertEqual(list(dd), ['x', 'y'])
        self.assertTrue(
            (dd['x'] == numpy.array([1.0, -3.0], dtype=numpy.float64)).all()
        )
        self.assertTrue((dd['y'] == numpy.array(['Q', 'Z'], dtype='S1')).all())

    # TODO: Remove this? -CLH
    def test_numpy2d_to_dict(self):
        arr = numpy.arange(12).reshape((3, 4)).transpose()
        cols = ['A', 'C', 'B']
        dd = numpy2d_to_dict(arr, cols)
        self.assertEqual(list(dd), cols)
        self.assertTrue((dd['A'] == numpy.arange(0, 4)).all())
        self.assertTrue((dd['C'] == numpy.arange(4, 8)).all())
        self.assertTrue((dd['B'] == numpy.arange(8, 12)).all())

    def test_dset_dict_to_list(self):
        ds = Dataset(
            {
                _k: list(range(_i * 10, (_i + 1) * 10))
                for _i, _k in enumerate('abcdefghijklmnop')
            }
        )
        ds0 = ds[:3].copy()
        ds1 = ds[6:9].copy()
        ds2 = ds[11:15].copy()
        dd = {'one': ds0, 'two': ds1, 'μεαν': ds2}
        with self.assertRaises(ValueError):
            _ = dset_dict_to_list(dd, 'keyfield')
        dd = {'one': ds0, 'two': ds1, 3: ds2}
        with self.assertRaises(ValueError):
            _ = dset_dict_to_list(dd, 'keyfield')
        dd = {'one': ds0, 'two': ds1, 'three': ds2}
        with self.assertRaises(ValueError):
            _ = dset_dict_to_list(dd, 'a')
        lst1 = dset_dict_to_list(dd, 'keyfield')
        self.assertEqual(id(ds0), id(lst1[0]))
        self.assertEqual(id(ds1), id(lst1[1]))
        self.assertEqual(id(ds2), id(lst1[2]))
        self.assertEqual(list(ds0.keys()), ['a', 'b', 'c', 'keyfield'])
        self.assertTrue((ds0.a == list(range(10))).all())
        self.assertTrue((ds0.keyfield == 'one').all())
        lst2 = dset_dict_to_list(dd, 'a', allow_overwrite=True)
        self.assertEqual(id(ds0), id(lst1[0]))
        self.assertEqual(list(ds0.keys()), ['a', 'b', 'c', 'keyfield'])
        self.assertTrue((ds0.a == 'one').all())
        self.assertTrue((ds0.b == list(range(10, 20))).all())
        self.assertTrue((ds0.keyfield == 'one').all())

    def test_append_dataset_dict(self):
        ds = Dataset(
            {
                _k: list(range(_i * 10, (_i + 1) * 10))
                for _i, _k in enumerate('abcdefghijklmnop')
            }
        )
        ds0 = ds[:3].copy()
        ds1 = ds[6:9].copy()
        ds2 = ds[11:15].copy()
        dd = {'one': ds0, 'two': ds1, 'three': ds2}
        ds = append_dataset_dict(dd, 'keyfield')
        ucols = set()
        for _d in dd.values():
            ucols.update(_d)
        self.assertEqual(set(ds.keys()), ucols)
        self.assertEqual(ds.get_nrows(), sum(_d.get_nrows() for _d in dd.values()))
        keyfield = []
        for _d in dd.values():
            keyfield.extend(_d.keyfield)
        self.assertTrue((ds.keyfield == keyfield).all())
        self.assertTrue((ds.a[:10] == range(10)).all())
        self.assertTrue((ds.g[10:20] == range(60, 70)).all())
        self.assertTrue((ds.l[20:30] == range(110, 120)).all())

    def test_possibly_convert_to_nanotime(self):
        ns1 = numpy.int64(1528295695919153408)
        arr1 = numpy.array([ns1 + 12 * _i for _i in range(25)])
        nt1, okay1 = possibly_convert_to_nanotime(arr1)
        self.assertTrue(okay1)
        self.assertIsInstance(nt1, DateTimeNano)
        arr2 = arr1.astype(numpy.uint64)
        nt2, okay2 = possibly_convert_to_nanotime(arr2)
        self.assertFalse(okay2)
        self.assertNotIsInstance(nt2, DateTimeNano)
        self.assertTrue((nt2 == arr2).all())
        ns3 = numpy.int64(1070376029353467904)
        arr3 = numpy.array([ns3 + 12 * _i for _i in range(25)])
        nt3, okay3 = possibly_convert_to_nanotime(arr3)
        self.assertFalse(okay3)
        self.assertNotIsInstance(nt3, DateTimeNano)
        self.assertTrue((nt3 == arr3).all())


if __name__ == "__main__":
    tester = unittest.main()
