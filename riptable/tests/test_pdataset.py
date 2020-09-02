import unittest
import os

import pytest

from riptable import *
from riptable.rt_sds import SDSMakeDirsOn

# change to true since we write into /tests directory
SDSMakeDirsOn()

class PDataset_Test(unittest.TestCase):
    # TODO use pytest file fixtures
    def test_constructor_ds_list(self):
        correct_cutoffs = [5, 10, 15]
        correct_auto = ['p0', 'p1', 'p2']
        correct_custom = ['test1', 'test2', 'test3']
        correct_files = ['f20180201.sds', 'f20190201.sds', 'f20200201.sds']
        correct_dates = ['20180201', '20190201', '20200201']

        ds = Dataset({'col_' + str(i): arange(5) for i in range(5)})

        # no info
        pds = PDataset([ds, ds, ds])
        self.assertTrue(pds._pfilenames == [])
        self.assertTrue(bool(np.all(pds.pnames == correct_auto)))
        self.assertTrue(bool(np.all(pds.pcutoffs == correct_cutoffs)))
        self.assertEqual(pds.pcount, 3)
        self.assertEqual(pds._nrows, 15)
        self.assertTrue(isinstance(pds._pnames, dict))

        # with names
        pds = PDataset([ds, ds, ds], pnames=['test1', 'test2', 'test3'])
        self.assertTrue(pds._pfilenames == [])
        self.assertTrue(bool(np.all(pds.pnames == correct_custom)))
        self.assertTrue(bool(np.all(pds.pcutoffs == correct_cutoffs)))
        self.assertEqual(pds.pcount, 3)
        self.assertEqual(pds._nrows, 15)
        self.assertTrue(isinstance(pds._pnames, dict))

        # with filenames that have dates
        pds = PDataset([ds, ds, ds], filenames=correct_files)
        self.assertTrue(pds._pfilenames == correct_files)
        self.assertTrue(bool(np.all(pds.pnames == correct_dates)))
        self.assertTrue(bool(np.all(pds.pcutoffs == correct_cutoffs)))
        self.assertEqual(pds.pcount, 3)
        self.assertEqual(pds._nrows, 15)
        self.assertTrue(isinstance(pds._pnames, dict))

        # with filenames AND pnames
        pds = PDataset([ds, ds, ds], pnames=correct_custom, filenames=correct_files)
        self.assertTrue(pds._pfilenames == correct_files)
        self.assertTrue(bool(np.all(pds.pnames == correct_custom)))
        self.assertTrue(bool(np.all(pds.pcutoffs == correct_cutoffs)))
        self.assertEqual(pds.pcount, 3)
        self.assertEqual(pds._nrows, 15)
        self.assertTrue(isinstance(pds._pnames, dict))

    def test_constructor_single(self):
        ds = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        pds = PDataset(ds)
        self.assertTrue(pds._pfilenames == [])
        self.assertTrue(bool(np.all(pds.pnames == ['p0'])))
        self.assertEqual(pds.pcutoffs[0], 5)
        self.assertEqual(pds.pcount, 1)
        self.assertTrue(isinstance(pds._pnames, dict))

    def test_constructor_file_list(self):
        correct_cutoffs = [5, 10, 15]
        correct_auto = ['p0', 'p1', 'p2']
        correct_custom = ['test1', 'test2', 'test3']
        ds = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        paths = [
            rb'riptable/tests/temp/ds' + str(i).encode() + b'.sds' for i in range(3)
        ]
        for i in range(3):
            ds.save(paths[i])

        # filepath only (found changing number)
        pds = PDataset(paths)
        self.assertTrue(pds._pfilenames == paths)
        self.assertTrue(
            bool(np.all(pds.pnames == ['0', '1', '2'])),
            msg=f'{pds.pnames} vs. {correct_auto}',
        )
        self.assertTrue(bool(np.all(pds.pcutoffs == correct_cutoffs)))
        self.assertEqual(pds.pcount, 3)
        self.assertEqual(pds._nrows, 15)
        self.assertTrue(isinstance(pds._pnames, dict))

        # filepath with custom pnames
        pds = PDataset(paths, pnames=correct_custom)
        self.assertTrue(pds._pfilenames == paths)
        self.assertTrue(
            bool(np.all(pds.pnames == correct_custom)),
            msg=f'{pds.pnames} vs. {correct_auto}',
        )
        self.assertTrue(bool(np.all(pds.pcutoffs == correct_cutoffs)))
        self.assertEqual(pds.pcount, 3)
        self.assertEqual(pds._nrows, 15)
        self.assertTrue(isinstance(pds._pnames, dict))

        for p in paths:
            os.remove(p)

    def test_files_no_dates(self):
        ds = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        correct_auto = ['p0', 'p1', 'p2']
        paths = [
            rb'riptable/tests/temp/ds' + str(i).encode() + b'.sds'
            for i in ['a', 'b', 'c']
        ]
        for i in range(3):
            ds.save(paths[i])

        pds = PDataset(paths)
        self.assertTrue(bool(np.all(pds.pnames == correct_auto)))

        for p in paths:
            os.remove(p)

    @pytest.mark.skip(reason="this test no longer works and is disabled")
    def test_constructor_errors(self):
        garbage = [1, 2, 'test.sds']
        with self.assertRaises(TypeError):
            pds = PDataset(garbage)

        garbage = [1, 2, 3]
        with self.assertRaises(TypeError):
            pds = PDataset(garbage)

    # will add more extensive indexing tests later
    def test_copy_internal(self):
        ds = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        pds = PDataset([ds, ds])

        # if row index, return dataset
        pds2 = pds[:2, :]
        self.assertTrue(type(pds2) == Dataset)

        # if col index, return pdataset
        pds2 = pds[:2]
        self.assertTrue(isinstance(pds2, PDataset))

    def test_prows(self):
        ds1 = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        ds2 = Dataset({'col_' + str(i): arange(7) for i in range(5)})
        ds3 = Dataset({'col_' + str(i): arange(9) for i in range(5)})

        pds = PDataset([ds1, ds2, ds3])
        self.assertTrue(bool(np.all(pds.prows == [5, 7, 9])))

    def test_pdict(self):
        names = ['test1', 'test2', 'test3']
        slices = [slice(0, 5, None), slice(5, 12, None), slice(12, 21, None)]
        ds1 = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        ds2 = Dataset({'col_' + str(i): arange(7) for i in range(5)})
        ds3 = Dataset({'col_' + str(i): arange(9) for i in range(5)})

        pds = PDataset([ds1, ds2, ds3], pnames=names)
        for i, (k, v) in enumerate(pds.pdict.items()):
            self.assertEqual(k, names[i])
            self.assertTrue(v == slices[i])

    # disabled while building partitioned groupby
    # def rtest_pgb(self):
    #    ds1 = Dataset({'col_'+str(i):arange(5) for i in range(5)})
    #    ds2 = Dataset({'col_'+str(i):arange(7) for i in range(5)})
    #    ds3 = Dataset({'col_'+str(i):arange(9) for i in range(5)})

    #    pds = PDataset([ds1,ds2,ds3])
    #    g = pds.pgb
    #    self.assertTrue(isinstance(g,GroupBy))
    #    result = g.sum()
    #    self.assertTrue('Partition' in pds)
    #    self.assertTrue(isinstance(pds.Partition,Categorical))
    #    self.assertEqual(result.label_get_names()[0], 'Partition')

    def test_piter(self):
        ds1 = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        ds2 = Dataset({'col_' + str(i): arange(7) for i in range(5)})
        ds3 = Dataset({'col_' + str(i): arange(9) for i in range(5)})

        dlist = [ds1, ds2, ds3]
        pds = PDataset(dlist)

        for i, (name, ds) in enumerate(pds.piter):
            self.assertEqual(name, pds.pnames[i])
            self.assertTrue(type(ds) == Dataset)
            eq = ds == dlist[i]
            for col in eq.values():
                self.assertTrue(bool(np.all(col)))

    def test_pslices(self):
        ds1 = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        ds2 = Dataset({'col_' + str(i): arange(7) for i in range(5)})
        ds3 = Dataset({'col_' + str(i): arange(9) for i in range(5)})
        slices = [slice(0, 5, None), slice(5, 12, None), slice(12, 21, None)]

        dlist = [ds1, ds2, ds3]
        pds = PDataset(dlist)

        for i, s in enumerate(pds.pslices):
            self.assertEqual(s, slices[i])

    def test_pslice(self):
        ds1 = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        ds2 = Dataset({'col_' + str(i): arange(7) for i in range(5)})
        ds3 = Dataset({'col_' + str(i): arange(9) for i in range(5)})
        slices = [slice(0, 5, None), slice(5, 12, None), slice(12, 21, None)]

        dlist = [ds1, ds2, ds3]
        pds = PDataset(dlist)

        for i in range(3):
            s = pds.pslice(i)
            self.assertEqual(slices[i], s)

        with self.assertRaises(IndexError):
            s = pds.pslice('a')

    def test_partition(self):
        ds1 = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        ds2 = Dataset({'col_' + str(i): arange(7) for i in range(5)})
        ds3 = Dataset({'col_' + str(i): arange(9) for i in range(5)})

        dlist = [ds1, ds2, ds3]
        pds = PDataset(dlist)

        for i, name in enumerate(pds.pnames):
            ds = pds.partition(name)
            self.assertTrue(type(ds) == Dataset)
            eq = ds == dlist[i]
            for col in eq.values():
                self.assertTrue(bool(np.all(col)))

        for i in range(3):
            ds = pds.partition(i)
            self.assertTrue(type(ds) == Dataset)
            eq = ds == dlist[i]
            for col in eq.values():
                self.assertTrue(bool(np.all(col)))

        with self.assertRaises(IndexError):
            _ = pds.partition([])

    def test_except_index(self):
        ds1 = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        pds = PDataset(ds1, pnames=['123'])

        ds2 = pds[123]
        eq = ds1 == ds2
        for col in eq.values():
            self.assertTrue(bool(np.all(col)))

        with self.assertRaises(KeyError):
            p = pds[1234]

        with self.assertRaises(KeyError):
            p = pds[[True, False]]

    def test_pload(self):
        ds1 = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        ds2 = Dataset({'col_' + str(i): arange(7) for i in range(5)})
        ds3 = Dataset({'col_' + str(i): arange(9) for i in range(5)})
        dlist = [ds1, ds2, ds3]

        correct_files = ['f20180201.sds', 'f20180202.sds', 'f20180203.sds']
        correct_dates = ['20180201', '20180202', '20180203']
        path = 'riptable/tests/temp/f{}.sds'

        for i, fname in enumerate(correct_files):
            dlist[i].save(r'riptable/tests/temp/' + fname)

        pds = PDataset.pload(path, 20180201, 20180203)
        self.assertEqual(pds.pcount, 3)
        self.assertTrue(bool(np.all(pds.pnames == correct_dates)))

        for i, fname in enumerate(correct_files):
            os.remove(r'riptable/tests/temp/' + fname)

    def test_psave(self):
        ds1 = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        ds2 = Dataset({'col_' + str(i): arange(7) for i in range(5)})
        ds3 = Dataset({'col_' + str(i): arange(9) for i in range(5)})
        dlist = [ds1, ds2, ds3]
        pds = PDataset(dlist)
        with self.assertRaises(NotImplementedError):
            pds.psave()

    def test_partition_rename(self):

        ds = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        pds = PDataset([ds, ds, ds])
        newnames = ['part1', 'part2', 'part3']
        pds.set_pnames(newnames)

        self.assertTrue(bool(np.all(pds.pnames == newnames)))

        with self.assertRaises(ValueError):
            pds.set_pnames(newnames[:2])

        with self.assertRaises(ValueError):
            pds.set_pnames(['p0', 'p0', 'p0'])

        with self.assertRaises(ValueError):
            pds.set_pnames(dict(zip(newnames, [0, 0, 0])))


if __name__ == '__main__':
    tester = unittest.main()
