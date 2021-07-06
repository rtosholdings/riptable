import pytest
import unittest
import re
import keyword
import pandas as pd

from collections import namedtuple
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import riptable as rt
from riptable import FastArray
from riptable import Struct
from riptable import Dataset
from riptable import Categorical
from riptable import NumpyCharTypes
from riptable import TimeSpan, utcnow, Date
from riptable.rt_numpy import arange, isnan, tile, logical
from riptable.rt_utils import describe
from riptable.rt_enum import (
    INVALID_DICT,
    TypeRegister,
    FILTERED_LONG_NAME,
    CategoryMode,
)
from .test_utils import get_all_categorical_data


def almost_eq(ds, arr, places=5, rtol=1e-5, atol=1e-5, equal_nan=True):

    epsilon = 10 ** (-places)
    rtol = epsilon

    if ds.shape != arr.shape:
        return False
    for _j, _col in enumerate(ds):
        _dsc = ds[_col]
        _arrc = arr[:, _j]
        if np.issubdtype(_dsc.dtype, np.floating):
            # N.B.: Do not compare the elements that both numpy and riptable give nan or inf.
            #       In the case of modulo by zero, numpy gives nan but riptable gives inf.
            finite_mask = ~((~np.isfinite(_dsc)) & (~np.isfinite(_arrc)))
            if not np.allclose(
                _dsc[finite_mask], _arrc[finite_mask], rtol, atol, equal_nan
            ):
                return False
        else:
            try:
                if not (_dsc == _arrc).all():
                    return False
            except TypeError:
                return False
    return True


# def almost_eq(ds, arr, places=5):
#    epsilon = 10 ** (-places)
#    rval = True
#    if ds.shape != arr.shape:
#        return False
#    for _j, _col in enumerate(ds):
#        _dsc = ds[_col]
#        _arrc = arr[:, _j]
#        for v1, v2 in zip(_dsc, _arrc.astype(_dsc.dtype.type)):
#            if isinstance(v1, str):
#                rval &= (v1 == v2)
#            else:
#                if not (np.isfinite(v1) and np.isfinite(v2)):
#                    continue
#                rval &= (abs(v1 - v2) < epsilon)
#    return rval


class TestDataset(unittest.TestCase):
    @staticmethod
    def get_basic_dataset(nrows=10):
        return Dataset(
            {
                _k: list(range(_i * nrows, (_i + 1) * nrows))
                for _i, _k in enumerate('abcdefghijklmnop')
            }
        )

    @staticmethod
    def get_arith_dataset(include_strings=False, dtype=None):
        ds = Dataset(
            {
                'A': [0, 6, 9],
                'B': [1.2, 3.1, 9.6],
                'G': [-1.6, 2.7, 4.6],
                'C': [2.4, 6.2, 19.2],
            }
        )
        if dtype:
            for _k, _v in ds.items():
                ds[_k] = _v.astype(dtype)
        if include_strings:
            ds.S = ['string_00', 'string_01', 'string_02']
            ds.U = ['ℙƴ☂ℌøἤ_00', 'ℙƴ☂ℌøἤ_01', 'ℙƴ☂ℌøἤ_02']
        return ds

    def test_col_ctor_01(self):
        list1 = [(_k, [_i]) for _i, _k in enumerate('bac')]
        with self.assertRaises(TypeError):
            _ = Dataset({'a': (1,)})
        with self.assertRaises(TypeError):
            _ = Dataset(list1)
        with self.assertRaises(ValueError):
            _ = Dataset({'a': [1, 2], 'b': 3})
        list2 = [Dataset({_k: [_i]}) for _i, _k in enumerate('bac')]
        with self.assertRaises(
            TypeError
        ):  # for this use concat_rows() or concat_columns()
            _ = Dataset(list2)
        with self.assertRaises(
            TypeError
        ):  # for this use concat_rows() or concat_columns()
            _ = Dataset([Struct(_d.asdict()) for _d in list2])
        with self.assertRaises(
            TypeError
        ):  # this one will be supported later w/ a sep. method
            _ = Dataset([_d.asdict() for _d in list2])
        assert (Dataset(None) == Dataset({})).all(
            axis=None
        ), f"Dataset default argument for 'None' input value should create an empty Dataset."
        list3 = [{_k: [_i]} for _i, _k in enumerate('bac')]
        with self.assertRaises(TypeError):
            _ = Dataset(list3)
        dict1 = {_k: [_i] for _i, _k in enumerate('bac')}
        st = Struct(dict1)
        ds = Dataset(st)
        self.assertEqual(ds.shape, (1, 3))
        for _i, _k in enumerate('bac'):
            self.assertEqual(ds[0, _k], st[_k][0])
        ds1 = Dataset({'a': [1], 'b': 2})
        ds2 = Dataset({'a': [[1]], 'b': 2})
        self.assertTrue((ds1 == ds2).all(axis=None))
        self.assertEqual(ds2.shape, (1, 2))
        self.assertEqual(ds2.a.shape, (1,))
        self.assertEqual(ds2.b.shape, (1,))
        ds = Dataset()
        self.assertEqual(ds.shape, (0, 0))

    # def test_col_ctor_02(self):
    #    hold_WarnOnInvalidNames = Struct.WarnOnInvalidNames
    #    Struct.WarnOnInvalidNames = False
    #    tempsave = Struct.AllowAnyName
    #    Struct.AllowAnyName = False
    #    with self.assertRaises(ValueError):
    #        _ = Dataset({'a': [0], '0': [1]})
    #    with self.assertRaises(ValueError):
    #        _ = Dataset({'a': [0], 'a-b-c': [1]})
    #    # we auto capitalize now
    #    #for kwd in keyword.kwlist:
    #    #    _ = Dataset({'a': [0], kwd: [1]})
    #    #for memb in dir(Dataset):
    #    #    with self.assertRaises(ValueError):
    #    #        _ = Dataset({'a': [0], memb: [1]})
    #    Struct.WarnOnInvalidNames = hold_WarnOnInvalidNames
    #    Struct.AllowAnyName = tempsave

    def test_col_ctor_02(self):
        arr = arange(5)
        inv_keys = ['True', 'False', 'None', 'size']
        inv_dict = {k: arr for k in inv_keys}
        with self.assertWarns(UserWarning):
            ds = Dataset(inv_dict)
        inv_keys[-1] = 'Size'

        self.assertTrue(bool(np.all(inv_keys == list(ds))))
        for k in inv_keys:
            self.assertTrue(bool(np.all(ds[k] == arr)))

    def test_col_ctor_03(self):
        tempsave = Struct.AllowAnyName
        Struct.AllowAnyName = False
        hold_WarnOnInvalidNames = Struct.WarnOnInvalidNames
        Struct.WarnOnInvalidNames = True
        with self.assertWarns(UserWarning):
            _ = Dataset({'a': [0], '0': [1]})
        with self.assertWarns(UserWarning):
            _ = Dataset({'a': [0], 'a-b-c': [1]})
        for kwd in keyword.kwlist:
            with self.assertWarns(UserWarning):
                _ = Dataset({'a': [0], kwd: [1]})
        # note: dir has changed
        # for memb in dir(Dataset):
        #    with self.assertWarns(UserWarning, msg=memb):
        #        # We expect really stupid errors resulting from not avoiding invalid names.
        #        try:
        #            _ = Dataset({'a': [0], memb: [1]})
        #        except (TypeError, AttributeError):
        #            pass
        Struct.WarnOnInvalidNames = hold_WarnOnInvalidNames
        Struct.AllowAnyName = tempsave

    def test_col_ctor_04(self):
        hold_UseFastArray = Struct.UseFastArray
        dd = {'a': FastArray([0, 1]), 'b': np.array([10, 11])}
        Struct.set_fast_array(True)
        ds1 = Dataset(dd)
        self.assertIsInstance(ds1.a, FastArray)
        self.assertIsInstance(ds1.b, FastArray)
        Struct.set_fast_array(False)
        ds2 = Dataset(dd)
        self.assertIsInstance(ds2.a, np.ndarray)
        self.assertIsInstance(ds2.b, np.ndarray)
        self.assertNotIsInstance(ds2.a, FastArray)
        self.assertNotIsInstance(ds2.b, FastArray)
        Struct.set_fast_array(hold_UseFastArray)

    def test_col_ctor_05(self):
        dd = {'a': list(range(4)), 'b': list('abcd'), 'c': list('μεαν')}
        ds = Dataset(dd)
        self.assertEqual(ds.shape, (4, 3))
        dd = {
            'a': np.array(list(range(4))),
            'b': np.array(list('abcd'), dtype=object),
            'c': np.array(list('μεαν'), dtype=object),
        }
        ds = Dataset(dd)
        self.assertEqual(ds.shape, (4, 3))

    def test_astype(self):
        ds = self.get_arith_dataset(include_strings=True)
        dtypes0 = ds.dtypes
        ds1 = ds.astype(np.int16)
        for _k in list('ABCG'):
            self.assertEqual(ds1[_k].dtype, np.dtype(np.int16))
        for _k in list('US'):
            self.assertEqual(ds1[_k].dtype, dtypes0[_k])
        ds.S = ds.A.astype(str)
        ds.S = ds.B.astype(str)
        # 9/28/2018 SJK: Dataset no longer flips Unicode arrays to Categorical, removed unicode column
        ds1 = ds.astype(np.float16, ignore_non_computable=True)
        for _k in list('ABCG'):
            self.assertEqual(ds1[_k].dtype, np.dtype(np.float16))

    def test_view_vs_copy_01(self):
        # Here, ensure that all standard indexing produces a view, not a copy;
        # the chief exception being "fancy row-indexing".
        # Also ensure that Dataset.copy(deep=True) does the correct thing!
        # First, numeric only.  Create two datasets, identical, but from totally different memory.
        ds1 = self.get_basic_dataset(nrows=1000)
        ds1[10:] = ds1[10:].astype(np.float32)
        ds2 = self.get_basic_dataset(nrows=1000)
        ds2[10:] = ds2[10:].astype(np.float32)
        self.assertTrue((ds1 == ds2).all(axis=None))
        ds1.a[5:100:5] *= -1
        self.assertTrue((ds1.a[5:100:5] == ds2.a[5:100:5] * -1).all())
        self.assertTrue((ds1[1:] == ds2[1:]).all(axis=None))
        ds1[5:100:5, ['b', 'd', 'e']] += -2
        self.assertTrue(
            (ds1[5:100:5, ['b', 'd', 'e']] == ds2[5:100:5, ['b', 'd', 'e']] + -2).all(
                axis=None
            )
        )
        self.assertTrue((ds1.c == ds2.c).all())
        self.assertTrue((ds1[5:] == ds2[5:]).all(axis=None))
        ds1[5:100:5, 5:7] = 84
        self.assertTrue((ds1.f[5:100:5] == 84).all())
        self.assertTrue((ds1.g[5:100:5] == 84).all())
        self.assertTrue((ds1[7:] == ds2[7:]).all(axis=None))
        ds1[[2, 4, 79], ['h', 'i']] = 99
        self.assertTrue((ds1.h[[2, 4, 79]] == 99).any())
        self.assertTrue((ds1.i[[2, 4, 79]] == 99).any())
        self.assertTrue((ds1[9:] == ds2[9:]).all(axis=None))
        # Now, a copy:
        ds3 = ds1[5:100:5, ['j', 'k']].copy()
        ds3[:, :] = -2000
        # New is changed.
        self.assertTrue((ds3.j == -2000).all())
        self.assertTrue((ds3.k == -2000).all())
        # Orig. is unchanged.
        self.assertTrue((ds1.j[5:100:5] != -2000).all())
        self.assertTrue((ds1.k[5:100:5] != -2000).all())
        self.assertTrue((ds1[9:] == ds2[9:]).all(axis=None))

    def test_basic_interface(self):
        cols = ['a', 'b', 'c', 'μεαν']
        dict1 = {_k: [_i] for _i, _k in enumerate(cols)}
        ds1 = Dataset(dict1)
        self.assertEqual(list(ds1.keys()), cols)
        self.assertEqual(list(ds1.keys()), cols)
        self.assertEqual(list(ds1), cols)
        for _idx, (_k, _v) in enumerate(ds1.items()):
            self.assertEqual(ds1[_k][0], dict1[_k][0])
            self.assertEqual(cols[_idx][0], _k[0])
            self.assertEqual(ds1[_idx][0], dict1[_k][0])
            self.assertEqual(getattr(ds1, _k)[0], dict1[_k][0])
        self.assertEqual(list(ds1.keys()), cols)
        self.assertEqual(list(ds1), cols)
        self.assertEqual(list(reversed(ds1)), list(reversed(cols)))
        self.assertEqual(ds1.size, ds1.get_ncols() * ds1.get_nrows())
        self.assertEqual(ds1.shape[1], ds1.get_ncols())
        self.assertEqual(len(ds1), ds1.get_nrows())
        self.assertEqual(ds1.shape, (ds1.get_nrows(), ds1.get_ncols()))

    def test_col_map_02(self):
        st = Dataset(
            {
                _k: list(range(_i * 10, (_i + 1) * 10))
                for _i, _k in enumerate('abcdefghijklmnopqrst')
            }
        )
        cmap = {
            'a': 'x',  # 1-transition
            'b': 'c',
            'c': 'b',  # 2-cycle
            'f': 'f',  # 1-cycle
            'g': 'h',
            'h': 'j',
            'j': 'k',
            'k': 'g',  # 4-cycle
            'm': 'n',
            'n': 'o',
            'o': 'y',  # 3-transition
            'p': 'q',
            'q': 'z',
            'z': 'r',
        }  # false 3-cycle, really a 2-transition
        st.col_map(cmap)
        for _k in 'abcdefghijklmnopqrst':
            self.assertEqual(st[cmap.get(_k, _k)][1], (ord(_k) - ord('a')) * 10 + 1)

    def test_col_add_prefix(self):
        ds = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        correct = ['new_' + col for col in list(ds.keys())]
        ds.col_add_prefix('new_')
        self.assertTrue(bool(np.all(ds.keys() == correct)))
        ds['new_new_col_1'] = arange(5)

        # test no overwrite
        correct = ['new_' + col for col in list(ds.keys())]
        ds.col_add_prefix('new_')
        self.assertTrue(bool(np.all(ds.keys() == correct)))

        # test restore labels
        ds = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        ds = ds.gb(['col_1', 'col_2']).sum()
        correct = ['new_' + label for label in ds.label_get_names()]
        ds.col_add_prefix('new_')
        self.assertTrue(bool(np.all(ds.label_get_names() == correct)))

        # test restore sort
        ds = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        ds.sort_view(['col_4', 'col_3'])
        correct = ['new_' + label for label in ds._col_sortlist]
        ds.col_add_prefix('new_')
        self.assertTrue(bool(np.all(ds._col_sortlist == correct)))

    def test_np_keep_unicode(self):
        arr = np.random.choice(['aaaaa', 'bbb', 'ccc', 'dd'], 30)
        self.assertTrue(arr.dtype.char == 'U')
        ds = Dataset({'uni': arr}, unicode=True)
        self.assertTrue(ds.uni.dtype.char == 'U')

    def test_swap_01(self):
        orig_cols = ['a', 'b', 'c', 'd']
        dict1 = {_k: [_i] for _i, _k in enumerate(orig_cols)}
        st = Dataset(dict1)
        for _i, _k in enumerate(orig_cols):
            self.assertEqual(st[_k][0], _i)
        st.a, st.b = st.b, st.a
        self.assertEqual(st['a'][0], 1)
        self.assertEqual(st['b'][0], 0)
        st.col_swap(['a', 'b'], ['b', 'a'])
        for _i, _k in enumerate(orig_cols):
            self.assertEqual(st[_k][0], _i)
        st.col_swap(['a', 'b', 'c'], ['c', 'b', 'a'])
        self.assertEqual(st['a'][0], 2)
        self.assertEqual(st['b'][0], 1)
        self.assertEqual(st['c'][0], 0)
        self.assertEqual(list(st.keys()), orig_cols)
        st = Dataset(dict1)
        st.col_swap(['a', 'b', 'c', 'd'], ['c', 'b', 'a', 'd'])
        self.assertEqual(st['a'][0], 2)
        self.assertEqual(st['b'][0], 1)
        self.assertEqual(st['c'][0], 0)
        self.assertEqual(st['d'][0], 3)
        self.assertEqual(list(st.keys()), orig_cols)

    def test_swap_02(self):
        ds = self.get_basic_dataset(nrows=4)
        for _i, _k in enumerate(ds):
            self.assertEqual(ds[_k][0], _i * 4)
        ds.a, ds.b = ds.b, ds.a
        self.assertEqual(ds['a'][0], 4)
        self.assertEqual(ds['b'][0], 0)
        ds = self.get_arith_dataset()
        # this type of set item does not work because a string mask returns a dataset
        # ds[['A', 'B']] = ds[['B', 'A']]
        a = ds.A
        ds.A = ds.B
        ds.B = a
        self.assertEqual(ds.A.tolist(), [1.2, 3.1, 9.6])
        self.assertEqual(ds.B.tolist(), [0, 6, 9])
        # this is also broken because getitem happens before setitem
        # ds[1:3, ['G', 'C']] = ds[1:3, ['C', 'G']]
        g = ds.G[1:3].copy()
        c = ds.C[1:3].copy()
        ds.G[1:3] = c
        ds.C[1:3] = g
        self.assertEqual(ds.G.tolist(), [-1.6, 6.2, 19.2])
        self.assertEqual(ds.C.tolist(), [2.4, 2.7, 4.6])
        # self.col_swap

    def test_add_remove_column(self):

        tempsave = Struct.AllowAnyName
        Struct.AllowAnyName = False

        ds = Dataset({'a': [0], 'b': [1], 'c': [2], 'μεαν': [3]})
        ds.d = [4]
        self.assertEqual(list(ds.keys()), ['a', 'b', 'c', 'μεαν', 'd'])
        self.assertEqual(ds.d, [4])
        self.assertEqual(ds['d'], [4])

        with self.assertRaises(IndexError):
            ds['1'] = [5]
        with self.assertRaises(IndexError):
            ds['a b'] = [5]

        for kwd in keyword.kwlist:
            if kwd not in ['True', 'False', 'None']:
                with self.assertRaises(IndexError):
                    ds[kwd] = [5]
        for memb in dir(Dataset):
            with self.assertRaises(IndexError):
                ds[memb] = [5]
        self.assertEqual(list(ds.keys()), ['a', 'b', 'c', 'μεαν', 'd'])
        del ds.b
        # SJK 11/27/2018 - we support this method of column removal now. hits the same path as del ds.c
        # with self.assertRaises(AttributeError):
        #    del ds['c']
        ds.col_remove('c')
        del ds.μεαν
        self.assertEqual(list(ds.keys()), ['a', 'd'])
        ds.e = 6
        ds['f'] = 7
        ds.g = [8]
        ds.h = [[9]]
        self.assertEqual(ds.h.shape, (1,))
        # this is allowed now since a tuple is listlike
        # with self.assertRaises(TypeError):
        #    ds.i = (10, )
        with self.assertRaises(TypeError):
            ds.i = {10}
        with self.assertRaises(TypeError):
            ds.i = {'a': 1, 'b': 2}
        ds.j = 'a'
        ds.k = 'μεαν'
        self.assertEqual(list(ds.keys()), list('adefghjk'))
        ds = Dataset({})
        ds.A = 1
        self.assertEqual(ds.shape, (1, 1))
        ds = Dataset({})
        ds.A = [1, 2, 3]
        self.assertEqual(ds.shape, (3, 1))
        ds = Dataset({})
        ds.A = np.array([1, 2, 3])
        self.assertEqual(ds.shape, (3, 1))
        with self.assertRaises(ValueError):
            ds.B = np.array([1, 2, 3, 4])

        Struct.AllowAnyName = tempsave

    def test_broadcast(self):
        ds = Dataset({'test': arange(10)})
        ds.x = [1]
        ds.y = (1,)
        self.assertTrue(ds.x[9] == 1)
        self.assertTrue(ds.y[9] == 1)

    def test_indexing_01(self):
        ds = Dataset({'a': [0], 'b': [1], 'c': [2], 'μεαν': [3]})
        self.assertTrue((ds.a == [0]).all())
        self.assertTrue((ds.b == [1]).all())
        self.assertTrue((ds.c == [2]).all())
        self.assertTrue((ds['a'] == [0]).all())
        self.assertTrue((ds['b'] == [1]).all())
        self.assertTrue((ds['c'] == [2]).all())
        self.assertTrue((ds[np.str_('a')] == [0]).all())
        self.assertEqual(ds.μεαν, [3])
        self.assertEqual(ds['μεαν'], [3])
        ds['μεαν'] = [3]
        self.assertTrue((ds[b'c'] == [2]).all())
        self.assertTrue((ds[np.str_('c')] == [2]).all())
        with self.assertRaises(KeyError):
            _ = ds['q']
        with self.assertRaises(KeyError):
            _ = ds['1']
        with self.assertRaises(KeyError):
            _ = ds['a b']
        with self.assertRaises(TypeError):
            _ = ds[0.0]
        with self.assertRaises(TypeError):
            _ = ds[:, 0.0]
        with self.assertRaises(AttributeError):
            _ = ds.q
        for kwd in keyword.kwlist:
            with self.assertRaises(KeyError):
                _ = ds[kwd]
        for memb in dir(Dataset):
            with self.assertRaises(KeyError):
                _ = ds[memb]

    def test_indexing_02(self):
        # rows
        ds = self.get_basic_dataset()
        self.assertEqual(ds[5, :].a, 5)
        rows = [6, 6, 1, 5, 8]
        self.assertTrue((ds[rows, :].a == rows).all())
        sel = np.zeros(ds.shape[0], dtype=bool)
        sel[rows] = True
        srows = [1, 5, 6, 8]
        with self.assertRaises(IndexError):
            _ = ds[[]]
        with self.assertRaises(IndexError):
            _ = ds[
                5,
            ]
        with self.assertRaises(TypeError):
            _ = ds[0.0, :]
        with self.assertRaises(IndexError):
            _ = ds[['1'], :]
        with self.assertRaises(IndexError):
            _ = ds['1', :]
        with self.assertRaises(TypeError):
            _ = ds[[complex(1)], :]
        with self.assertRaises(TypeError):
            _ = ds[complex(1), :]
        with self.assertRaises(IndexError):
            _ = ds[
                rows,
            ]
        with self.assertRaises(IndexError):
            _ = ds[
                sel,
            ]
        with self.assertRaises(IndexError):
            _ = ds[ds.get_nrows(), :]
        with self.assertRaises(TypeError):
            _ = ds[None, :]
        self.assertTrue((ds[sel, :].a == srows).all())
        self.assertTrue((ds[sel.tolist(), :].a == srows).all())

    def test_indexing_03(self):
        # cols
        ds = self.get_basic_dataset()
        colb = list(range(10, 20))
        self.assertTrue((ds['b'] == colb).all())
        self.assertTrue((ds[:, 'b'] == colb).all())
        cols = ['b', 'c', 'd', 'g']
        row0d = {'b': 10, 'c': 20, 'd': 30, 'g': 60}
        self.assertEqual(dict(ds[:, cols][0, :].asdict()), row0d)  # temp hack
        ncols = np.array(cols)
        self.assertEqual(dict(ds[:, ncols][0, :].asdict()), row0d)  # temp hack
        self.assertIsInstance(ds.a, FastArray)
        self.assertIsInstance(ds[0], FastArray)
        self.assertIsInstance(ds[[0]], Dataset)
        self.assertEqual(list(ds[['a', 'b']].keys()), ['a', 'b'])
        self.assertEqual(list(ds[[b'a', b'b']].keys()), ['a', 'b'])
        self.assertEqual(list(ds[[b'a', 'b']].keys()), ['a', 'b'])
        self.assertEqual(list(ds[['a', b'b']].keys()), ['a', 'b'])
        self.assertEqual(list(ds[['a', np.str_('b')]].keys()), ['a', 'b'])
        dupcols = ds[:, ['b', 'c', 'd', 'c', 'g']]
        # no longer raise IndexError, check that column removed
        self.assertEqual(dupcols.shape[1], 4)
        with self.assertRaises(KeyError):
            _ = ds['q']
        with self.assertRaises(AttributeError):
            _ = ds.q

    def test_indexing_04(self):
        # cols
        # There is a behavioral change: don't let magic occur.  Index on single 'value' -> columns.
        # Since integer, list(int), logical and slice indexing are allowed for columns as well,
        # the following are all actually allowed COLUMN operations, but they do not give what is
        # perhaps expected!
        ds = self.get_basic_dataset()
        cols = [6, 6, 1, 5, 8]
        dupcols = ds[cols]  # there are duplicates in rows
        self.assertEqual(dupcols.shape[1], 4)
        sel = np.zeros(ds.shape[1], dtype=bool)
        sel[cols] = True
        sel_cnames = [_k for _i, _k in enumerate(list(ds.keys())) if sel[_i]]
        scols = [6, 1, 5, 8]
        scols_cnames = list(map(list(ds.keys()).__getitem__, scols))
        self.assertEqual(list(ds[scols].keys()), scols_cnames)
        self.assertEqual(list(ds[[4, 5]].keys()), ['e', 'f'])
        with self.assertRaises(TypeError):
            _ = ds[:, None]
        with self.assertRaises(TypeError):
            _ = ds[None]
        self.assertEqual(list(ds[sel].keys()), sel_cnames)
        self.assertEqual(list(ds[sel.tolist()].keys()), sel_cnames)

    def test_indexing_05(self):
        # rows & cols
        ds = self.get_basic_dataset()
        self.assertEqual(ds[5, 'a'], 5)
        rows = [6, 6, 1, 5, 8]
        self.assertTrue((ds[rows, 'a'] == rows).all())
        sel = np.zeros(ds.shape[0], dtype=bool)
        sel[rows] = True
        srows = [1, 5, 6, 8]
        self.assertTrue((ds[sel, 'a'] == srows).all())
        self.assertTrue((ds[sel, ['a', 'c']].a == srows).all())
        self.assertTrue((ds[sel, ['a', 'c']].c == [_i + 20 for _i in srows]).all())
        self.assertIsInstance(ds[5, 'a'], np.integer)
        with self.assertRaises(TypeError):
            _ = ds[0.0, 0.0]
        with self.assertRaises(TypeError):
            _ = ds[None, None]

    def test_indexing_06(self):
        # slicing
        ds = self.get_basic_dataset()
        nrows, ncols = ds.shape
        self.assertEqual(ds[1:-1, :].shape, (nrows - 2, ncols))
        self.assertEqual(ds[:, ::2].shape, (nrows, ncols // 2))
        self.assertEqual(ds[::2].shape, (nrows, ncols // 2))
        self.assertEqual(ds[4:-3, 1::3].shape, (nrows - 7, ncols // 3))
        self.assertEqual(sum(ds[4:-3, 1::3].sum().tolist()[0]), 1125)
        self.assertEqual(sum(ds[1:-3:2, 1:8:3].sum().tolist()[0]), 387)

    def test_indexing_07(self):
        # get/set do matching and sane things, singles
        ds = self.get_basic_dataset()
        ds['newcol'] = -1
        self.assertTrue((ds['newcol'] == FastArray([-1] * ds.get_nrows())).all())
        ds['another'] = -2
        self.assertTrue((ds['another'] == FastArray([-2] * ds.get_nrows())).all())
        with self.assertRaises(NotImplementedError):
            ds[3, 'good_and_bad'] = -3
        with self.assertRaises(IndexError):
            ds[ds.get_nrows(), :] = -999
        # now accepted
        ds[['another', 'yet_another']] = -3
        ds = self.get_basic_dataset()
        o2c = ds[2, 'c']
        ds.a = -1
        self.assertTrue((ds['a'] == FastArray([-1] * ds.get_nrows())).all())
        ds.a = range(-100, -100 + ds.shape[0])
        self.assertTrue((ds['a'] == FastArray(range(-100, -100 + ds.shape[0]))).all())
        ds.b[1] = -2
        self.assertEqual(ds[1, 'b'], -2)
        o2c = ds[2, 'c']
        ds[2, :].c = -3
        self.assertNotEqual(-3, o2c)
        self.assertEqual(ds[2, 'c'], o2c)
        o3r = ds[3, :]
        ds[3, :] = -4
        self.assertTrue(
            (ds[3, :] != o3r).all(axis=None),
            msg='Setting a row should modify in place!',
        )
        ds[4, 'e'] = -5
        self.assertEqual(ds[4, 'e'], -5)
        self.assertEqual(ds.e[4], -5)
        self.assertEqual(ds[4, :].e, -5)
        ds = self.get_basic_dataset()
        ds['a'] = list(range(-100, -100 + ds.shape[0]))
        self.assertTrue((ds.a == FastArray(range(-100, -100 + ds.shape[0]))).all())
        ds[1, :] = range(-100, -100 + ds.shape[1])
        self.assertEqual(ds[1, :].tolist(), [list(range(-100, -100 + ds.shape[1]))])

    def test_indexing_08(self):
        # get/set do matching and sane things, multiples
        ds = self.get_basic_dataset()
        ds[['a', 'b']] = -1
        self.assertTrue((ds['a'] == FastArray([-1] * ds.get_nrows())).all())
        self.assertTrue((ds['b'] == FastArray([-1] * ds.get_nrows())).all())
        ds = self.get_basic_dataset()
        with self.assertRaises(ValueError):
            ds[['a', 'b']] = list(range(-100, -90))
        with self.assertRaises(ValueError):
            ds[['a', 'b']] = [list(range(-100, -90))] * 2
        other1 = Dataset(
            {
                f'i{_i:02d}': list(
                    range(_i * ds.get_nrows(), _i * ds.get_nrows() + ds.get_nrows())
                )
                for _i in range(2)
            }
        )
        # ds[['g', 'k']] = other1
        ds['g'] = other1[0]
        ds['k'] = other1[1]
        for _i, _k in enumerate('gk'):
            self.assertTrue(
                (
                    ds[_k]
                    == FastArray(range(ds.get_nrows() * _i, ds.get_nrows() * (_i + 1)))
                ).all()
            )
        ds = self.get_basic_dataset()
        other2 = Dataset(
            {f'i{_i:02d}': list(range(_i * 3, _i * 3 + 3)) for _i in range(ds.shape[1])}
        )
        for _i, _k in enumerate('ab'):
            self.assertTrue(
                (
                    ds[_k]
                    == FastArray(range(ds.get_nrows() * _i, ds.get_nrows() * (_i + 1)))
                ).all()
            )

        # ds[[5, 7, 8], :] = other2
        for i in range(ds.shape[1]):
            ds[i][[5, 7, 8]] = other2[i]
        ds = self.get_basic_dataset()
        other3 = Dataset({'A': [-1, -2, -3], 'B': [-4, -5, -6]})
        with self.assertRaises(ValueError):
            ds[:, ['g', 'k']] = other3
        with self.assertRaises(IndexError):
            ds[[5, 7, 8], :] = other3
        for i, k in enumerate("gk"):
            ds[k][[5, 7, 8]] = other3[i]
        # ds[[5, 7, 8], ['g', 'k']] = other3
        self.assertTrue((ds.g == [60, 61, 62, 63, 64, -1, 66, -2, -3, 69]).all())
        self.assertTrue((ds.h == [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]).all())
        self.assertTrue((ds.k == [100, 101, 102, 103, 104, -4, 106, -5, -6, 109]).all())

    def test_indexing_08_previous(self):
        # get/set do matching and sane things, multiples
        ds = self.get_basic_dataset()
        ds[['a', 'b']] = -1
        self.assertTrue((ds['a'] == FastArray([-1] * ds.get_nrows())).all())
        self.assertTrue((ds['b'] == FastArray([-1] * ds.get_nrows())).all())
        ds = self.get_basic_dataset()
        with self.assertRaises(ValueError):
            ds[['a', 'b']] = list(range(-100, -90))
        with self.assertRaises(ValueError):
            ds[['a', 'b']] = [list(range(-100, -90))] * 2
        other1 = Dataset(
            {
                f'i{_i:02d}': list(
                    range(_i * ds.get_nrows(), _i * ds.get_nrows() + ds.get_nrows())
                )
                for _i in range(2)
            }
        )
        ds[['g', 'k']] = other1
        for _i, _k in enumerate('gk'):
            self.assertTrue(
                (
                    ds[_k]
                    == FastArray(range(ds.get_nrows() * _i, ds.get_nrows() * (_i + 1)))
                ).all()
            )
        ds = self.get_basic_dataset()
        other2 = Dataset(
            {f'i{_i:02d}': list(range(_i * 3, _i * 3 + 3)) for _i in range(ds.shape[1])}
        )
        for _i, _k in enumerate('ab'):
            self.assertTrue(
                (
                    ds[_k]
                    == FastArray(range(ds.get_nrows() * _i, ds.get_nrows() * (_i + 1)))
                ).all()
            )
        ds[[5, 7, 8], :] = other2
        ds = self.get_basic_dataset()
        other3 = Dataset({'A': [-1, -2, -3], 'B': [-4, -5, -6]})
        with self.assertRaises(ValueError):
            ds[:, ['g', 'k']] = other3
        with self.assertRaises(IndexError):
            ds[[5, 7, 8], :] = other3
        ds[[5, 7, 8], ['g', 'k']] = other3
        self.assertTrue((ds.g == [60, 61, 62, 63, 64, -1, 66, -2, -3, 69]).all())
        self.assertTrue((ds.h == [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]).all())
        self.assertTrue((ds.k == [100, 101, 102, 103, 104, -4, 106, -5, -6, 109]).all())

    def test_indexing_09(self):
        # get/set with row and column filters
        ds1 = Dataset({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds2 = Dataset({'a': [7, 8, 9], 'b': [10, 11, 12]})
        filt = [False, True, False]
        ds1[filt, :] = ds2[filt, :]
        self.assertEqual(ds1.a[1], 8)
        self.assertEqual(ds1.b[1], 11)
        ds3 = Dataset({'a': [13], 'b': [14]})
        ds1[filt, :] = ds3
        self.assertEqual(ds1.a[1], 13)
        self.assertEqual(ds1.b[1], 14)
        filt = [True, False, False]
        ds3 = ds1[filt, :]
        self.assertEqual(ds3.a[0], 1)
        self.assertEqual(ds3.b[0], 4)
        ds1 = ds2
        self.assertTrue((ds1.a == [7, 8, 9]).all())
        self.assertTrue((ds1.b == [10, 11, 12]).all())
        with self.assertRaises(ValueError):
            ds1[:, ['a', 'b']] = FastArray([1, 2, 3])

    # def test_row_indexing(self):
    #    strings = FastArray(['c', 'a', 'b', 'b', 'a', 'b', 'c', 'b', 'a', 'a', 'c', 'c', 'b', 'c', 'a'])
    #    int1    = FastArray([2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1])
    #    int2    = None

    def test_concat_rows(self):
        N = 5
        dset1, dset2, _ = self._concat_test_data(N)
        # print('dset1\n',dset1)
        # print(dset2)
        dset = Dataset.concat_rows([dset1, dset2])
        nr1 = dset1.get_nrows()
        nr2 = dset2.get_nrows()
        ncols = len(set(dset1) | set(dset2))
        self.assertEqual(dset.shape, (nr1 + nr2, ncols))
        for _k, _v in dset.items():
            if _k in dset1 and _k in dset2:
                self.assertTrue((_v == np.hstack((dset1[_k], dset2[_k]))).all())
            elif _k in dset1:
                self.assertTrue((_v[:nr1] == dset1[_k]).all())
            else:
                self.assertTrue((_v[nr1:] == dset2[_k]).all())

    def test_copy_names(self):

        ds = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        ds2 = ds[:2, :]
        for k, v in ds.items():
            cp = ds2[k]
            self.assertEqual(v.get_name(), cp.get_name())

    def test_concat_rows_with_categoricals(self):
        # String labels
        ds1 = Dataset(
            {'Ticker': ['a', 'a', 'c', 'b', 'b'], 'Price': 100 + np.arange(5)}
        )
        ds2 = Dataset(
            {'Ticker': ['b', 'b', 'd', 'd', 'c'], 'Price': 105 + np.arange(5)}
        )
        ds1.Ticker = Categorical(ds1.Ticker)
        ds2.Ticker = Categorical(ds2.Ticker)
        ds = Dataset.concat_rows([ds1, ds2])
        self.assertTrue(
            (
                ds.Ticker.as_string_array
                == ['a', 'a', 'c', 'b', 'b', 'b', 'b', 'd', 'd', 'c']
            ).all()
        )
        self.assertTrue((ds.Price == 100 + np.arange(10)).all())
        # Numeric labels
        ds1 = Dataset({'Ticker': [1, 1, 3, 2, 2], 'Price': 100 + np.arange(5)})
        ds2 = Dataset({'Ticker': [2, 2, 4, 4, 3], 'Price': 105 + np.arange(5)})
        ds1.Ticker = Categorical(ds1.Ticker, [1, 2, 3, 4, 5], from_matlab=True)
        ds2.Ticker = Categorical(ds2.Ticker, [1, 2, 3, 4, 5], from_matlab=True)
        ds = Dataset.concat_rows([ds1, ds2])
        self.assertTrue((ds.Ticker == [1, 1, 3, 2, 2, 2, 2, 4, 4, 3]).all())
        self.assertTrue((ds.Price == 100 + np.arange(10)).all())
        N = 16
        dset1 = Dataset(dict(A=np.arange(N), C=np.ones(N), B=N * ['c']))
        dset2 = Dataset(dict(A=np.arange(N, 2 * N, 1), D=np.zeros(N), B=N * ['d']))
        test = ['test1', 'test2']
        dset2['D'] = Categorical(dset2['D'], categories=test, from_matlab=True)
        ds3 = Dataset.concat_rows([dset1, dset2])
        self.assertTrue(all(ds3.D.view(FastArray) == np.zeros(N * 2)))
        self.assertTrue(all(ds3.D.expand_array == [FILTERED_LONG_NAME] * N * 2))

    def test_concat_datetime(self):
        dtn = TypeRegister.DateTimeNano(
            [1541239200000000000, 1541325600000000000], from_tz='NYC', to_tz='NYC'
        )
        dts = dtn.hour_span
        ds1 = Dataset({'dtn': dtn, 'dts': dts})
        ds2 = Dataset({'dtn': dtn, 'dts': dts})
        ds3 = Dataset.concat_rows([ds1, ds2])
        self.assertTrue(isinstance(ds3.dtn, TypeRegister.DateTimeNano))
        self.assertTrue(isinstance(ds3.dts, TypeRegister.TimeSpan))

    def test_concat_columns(self):
        N = 5
        dset1, dset2, dsets = self._concat_test_data(N)
        for do_copy in (True, False):
            # unequal lengths
            with self.assertRaises(ValueError):
                _ = Dataset.concat_columns([dset1, dset2.head(2)], do_copy)
            with self.assertRaises(ValueError):
                _ = Dataset.concat_columns(dsets, do_copy, on_duplicate='invalid')
                # duplicate columns
            with self.assertRaises(KeyError):
                _ = Dataset.concat_columns(dsets, do_copy)
            # first and last
            with self.assertWarns(UserWarning):
                dset = Dataset.concat_columns(dsets, do_copy, on_duplicate='first')
            self.assertTrue((dset.A == dset1.A).all())
            dset = Dataset.concat_columns(
                dsets, do_copy, on_duplicate='last', on_mismatch='ignore'
            )
            self.assertTrue((dset.A == dset2.A).all())
            # shape
            with self.assertRaises(RuntimeError):
                dset = Dataset.concat_columns(
                    dsets, do_copy, on_duplicate='first', on_mismatch='raise'
                )
            dset = Dataset.concat_columns(
                dsets, do_copy, on_duplicate='first', on_mismatch='ignore'
            )
            self.assertEqual(dset.shape, (N, 4))
            self.assertEqual(list(dset.keys()), ['A', 'B', 'C', 'D'])
        for do_copy, test in ((True, self.assertNotEqual), (False, self.assertEqual)):
            # copy vs. view
            dset1, dset2, dsets = self._concat_test_data(N)
            with self.assertWarns(UserWarning):
                dset = Dataset.concat_columns(dsets, do_copy, on_duplicate='first')
            dset.A[0] = -1
            test(dset1.A[0], dset.A[0])
            dset1, dset2, dsets = self._concat_test_data(N)
            dset = Dataset.concat_columns([dset1], do_copy)
            dset.A[0] = -1
            test(dset1.A[0], dset.A[0])
            dset = Dataset.concat_columns([dset1, dset1], do_copy, on_duplicate='first')
            self.assertTrue((dset == dset1).all(axis=None))

    def _concat_test_data(self, N):
        dset1 = Dataset(dict(A=np.arange(N), B=np.ones(N), C=N * ['c']))
        dset2 = Dataset(dict(A=np.arange(N, 2 * N, 1), B=np.zeros(N), D=N * ['d']))
        dsets = [dset1, dset2]
        return dset1, dset2, dsets

    def test_from_tagged_rows(self):
        ds0 = Dataset({'a': [1, 2, 3], 'b': [11, 12, 13]})
        ds1 = Dataset.from_tagged_rows(
            [{'a': 1, 'b': 11}, {'a': 2, 'b': 12}, {'a': 3, 'b': 13}]
        )
        ds2 = Dataset.from_tagged_rows(
            Struct(_r)
            for _r in [{'a': 1, 'b': 11}, {'a': 2, 'b': 12}, {'a': 3, 'b': 13}]
        )
        ds3 = Dataset.from_tagged_rows(
            Dataset(_r)
            for _r in [{'a': 1, 'b': 11}, {'a': 2, 'b': 12}, {'a': 3, 'b': 13}]
        )
        Row = namedtuple('Row', ['a', 'b'])
        ds4 = Dataset.from_tagged_rows(
            Row(**_r)
            for _r in [{'a': 1, 'b': 11}, {'a': 2, 'b': 12}, {'a': 3, 'b': 13}]
        )
        self.assertIsInstance(ds1, Dataset)
        self.assertTrue((ds1 == ds0).all(axis=None))
        self.assertTrue((ds2 == ds0).all(axis=None))
        self.assertTrue((ds3 == ds0).all(axis=None))
        self.assertTrue((ds4 == ds0).all(axis=None))
        with self.assertRaises(NotImplementedError):
            _ = Dataset.from_tagged_rows(
                [
                    {'a': 1, 'b': 11},
                    {'a': 2, 'b': 12},
                    {'a': 3},
                    {'b': 14},
                    {'c': 25},
                    {'a': 6, 'b': 16, 'c': 26},
                ]
            )

    def test_from_rows(self):
        ds0 = Dataset({'a': [1, 2, 3], 'b': [11, 12, 13]})
        ds1 = Dataset.from_rows([[1, 11], [2, 12], [3, 13]], ['a', 'b'])
        ds2 = Dataset.from_rows(((1, 11), (2, 12), (3, 13)), ['a', 'b'])
        ds3 = Dataset.from_rows(np.array([[1, 11], [2, 12], [3, 13]]), ['a', 'b'])
        self.assertIsInstance(ds1, Dataset)
        self.assertTrue((ds1 == ds0).all(axis=None))
        self.assertTrue((ds2 == ds0).all(axis=None))
        self.assertTrue((ds3 == ds0).all(axis=None))
        with self.assertRaises(ValueError):
            _ = Dataset.from_rows([[1, 11], [2, 12, 0], [3, 13]], ['a', 'b'])
        with self.assertRaises(ValueError):
            _ = Dataset.from_rows([[1, 11], [2,], [3, 13]], ['a', 'b'])
        with self.assertRaises(TypeError):
            _ = Dataset.from_rows([{'a': 1, 'b': 11}, {'a': 2, 'b': 12}], ['a', 'b'])

    def test_from_jagged_rows(self):
        ds1 = Dataset.from_jagged_rows([['A', 'B', 'C'], ['D'], ['E', 'F', 'GGG'], 'H'])
        # from_jagged will add the letter 'C' in front now by default
        nd = INVALID_DICT[ds1['C1'].dtype.num]
        ds0 = Dataset(
            {
                'C0': ['A', 'D', 'E', 'H'],
                'C1': ['B', nd, 'F', nd],
                'C2': ['C', nd, 'GGG', nd],
            }
        )
        self.assertIsInstance(ds1, Dataset)
        self.assertTrue((ds1 == ds0).all(axis=None))

    # 10/16/2018 - SJK removed these tests because they use NodataType, which has been removed
    # def test_from_jagged_dict(self):
    #    d = {'name': ['bob', 'mary', 'sue', 'john'],
    #        'letters': [['A', 'B', 'C'], ['D'], ['E', 'F', 'G'], 'H']}
    #    nd = Categorical.NODATA.cast(str)
    #    ds0 = Dataset({'name': ['bob', 'mary', 'sue', 'john'],
    #                    'letters0': ['A', 'D', 'E', 'H'], 'letters1': ['B', nd, 'F', nd],
    #                    'letters2': ['C', nd, 'G', nd]})
    #    ds1 = Dataset.from_jagged_dict(d)
    #    self.assertIsInstance(ds1, Dataset)
    #    self.assertTrue((ds1 == ds0).all(axis=None))
    #    with self.assertRaises(ValueError):
    #        _ = Dataset.from_jagged_dict({'name': ['bob', 'mary', 'sue'],
    #            'letters': [['A', 'B', 'C'], ['D'], ['E', 'F', 'G'], 'H']})
    #    ds2 = Dataset({'name': ['bob', 'bob', 'bob', 'mary', 'sue', 'sue', 'sue', 'john'],
    #                    'letters': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']})
    #    ds3 = Dataset.from_jagged_dict(d, stacked=True)
    #    self.assertTrue((ds2 == ds3).all(axis=None))

    def test_melt(self):
        ds = Dataset({'A': ['a', 'b', 'c'], 'B': [1, 3, 5], 'C': [2, 4, 6]})
        tm0 = Dataset(
            {'A': ['a', 'b', 'c'], 'variable': ['B', 'B', 'B'], 'value': [1, 3, 5]}
        )
        tm0t = ds.melt(id_vars=['A'], value_vars=['B'])
        self.assertTrue((tm0 == tm0t).all(axis=None))
        tm1 = Dataset(
            {
                'A': ['a', 'b', 'c', 'a', 'b', 'c'],
                'variable': ['B', 'B', 'B', 'C', 'C', 'C'],
                'value': [1, 3, 5, 2, 4, 6],
            }
        )
        tm1t = ds.melt(id_vars=['A'], value_vars=['B', 'C'])
        self.assertTrue((tm1 == tm1t).all(axis=None))
        tm2 = Dataset(
            {'A': ['a', 'b', 'c'], 'myVarname': ['B', 'B', 'B'], 'myValname': [1, 3, 5]}
        )
        tm2t = ds.melt(
            id_vars=['A'],
            value_vars=['B'],
            var_name='myVarname',
            value_name='myValname',
        )
        self.assertTrue((tm2 == tm2t).all(axis=None))

    def test_pivot(self):
        pass

    def test_tolist(self):
        ds = self.get_basic_dataset()
        self.assertEqual(
            ds[4, :].tolist(),
            [[4, 14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114, 124, 134, 144, 154]],
        )
        self.assertEqual(
            ds[[4, 5, 8], :].tolist(),
            [
                [4, 14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114, 124, 134, 144, 154],
                [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155],
                [8, 18, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118, 128, 138, 148, 158],
            ],
        )

    def test_asrows(self):
        ds = self.get_basic_dataset()
        self.assertIsInstance(next(ds.asrows()), Dataset)
        self.assertIsInstance(next(ds.asrows(as_type='array')), np.ndarray)
        _it = ds.asrows()
        for _i in range(ds.get_nrows()):
            self.assertEqual(next(_it).tolist(), ds[_i, :].tolist())
        _it = ds.asrows()
        for _i in range(ds.get_nrows()):
            self.assertEqual(next(_it).tolist(), ds[_i, :].tolist())

    def test_aggregators_01(self):
        ds = self.get_basic_dataset()
        self.assertEqual(
            ds.sum().tolist()[0],
            [
                45,
                145,
                245,
                345,
                445,
                545,
                645,
                745,
                845,
                945,
                1045,
                1145,
                1245,
                1345,
                1445,
                1545,
            ],
        )
        self.assertTrue(all((ds.sum(axis=0) == ds.sum()).all().tolist()[0]))
        self.assertEqual(
            ds.sum(axis=1).tolist(),
            [1200, 1216, 1232, 1248, 1264, 1280, 1296, 1312, 1328, 1344],
        )
        self.assertEqual(ds.sum(axis=None), 12720)
        self.assertEqual(
            ds.mean().tolist()[0],
            [
                4.5,
                14.5,
                24.5,
                34.5,
                44.5,
                54.5,
                64.5,
                74.5,
                84.5,
                94.5,
                104.5,
                114.5,
                124.5,
                134.5,
                144.5,
                154.5,
            ],
        )
        for method in 'min max sum mean std var median count'.split():
            func = getattr(ds, method)
            self.assertIsInstance(func(), Struct)
            self.assertNotIsInstance(func(as_dataset=False), Dataset)
            self.assertIsInstance(func(as_dataset=True), Dataset)
            for axislist, atype in (
                ([0, 'c', 'C', 'col', 'COL', 'column', 'COLUMN'], Struct),
                ([1, 'r', 'R', 'row', 'ROW'], np.ndarray),
                ([None, 'all', 'ALL'], (int, float, np.integer, np.floating)),
            ):
                for axis in axislist:
                    self.assertIsInstance(func(axis=axis), atype)
            with self.assertRaises(NotImplementedError):
                _ = func(axis=3)
            with self.assertRaises(NotImplementedError):
                _ = func(axis='EVERYTHING')

    def test_aggregators_02(self):
        ds = self.get_basic_dataset()[:, :3]
        dvals = b'abcdefghij'
        evals = b'ABCDEFGHIJ'
        ds.d = list(dvals.decode())
        ds.e = list(evals.decode())
        self.assertEqual(ds.count().tolist()[0], [ds.get_nrows()] * ds.get_ncols())
        self.assertEqual(ds.count(fill_value=None).tolist()[0], [ds.get_nrows()] * 3)
        self.assertEqual(ds.max().tolist()[0], [9, 19, 29, b'j', b'J'])
        self.assertEqual(ds.max(fill_value={}).tolist()[0], [9, 19, 29])
        self.assertEqual(
            ds.max(fill_value=b''.join).tolist()[0], [9, 19, 29, dvals, evals]
        )
        self.assertEqual(
            ds.max(fill_value='banana').tolist()[0], [9, 19, 29, b'banana', b'banana']
        )
        self.assertEqual(ds.max(fill_value=0).tolist()[0], [9, 19, 29, 0, 0])
        self.assertEqual(
            ds.max(fill_value={'d': None, 'e': b''.join}).tolist()[0],
            [9, 19, 29, evals],
        )
        self.assertEqual(
            ds.max(fill_value={'d': 'apple', 'e': 'orange'}).tolist()[0],
            [9, 19, 29, b'apple', b'orange'],
        )

    def test_aggregators_03(self):
        ds1 = self.get_basic_dataset()
        ds1.S = list('ABCDEFGHIJ')
        ds2 = self.get_basic_dataset()
        for axis in (0, 1, None):
            self.assertEqual(ds1.sum(axis=axis).tolist(), ds2.sum(axis=axis).tolist())
        self.assertEqual(
            ds1.sum(axis=0, fill_value=0)[list(ds2)].tolist(),
            ds2.sum(axis=0, fill_value=0).tolist(),
        )
        self.assertEqual(ds1.sum(axis=0, fill_value=0).S, 0)
        self.assertEqual(
            ds1.sum(axis=1, fill_value=0).tolist(),
            ds2.sum(axis=1, fill_value=0).tolist(),
        )
        self.assertEqual(
            [_x - 1 for _x in ds1.sum(axis=1, fill_value=1).tolist()],
            ds2.sum(axis=1, fill_value=1).tolist(),
        )
        self.assertEqual(
            ds1.sum(axis=None, fill_value=0), ds2.sum(axis=None, fill_value=0)
        )
        self.assertEqual(
            ds1.sum(axis=None, fill_value=1) - 1, ds2.sum(axis=None, fill_value=1)
        )
        self.assertEqual(
            ds1.sum(axis=None, fill_value=len) - 10, ds2.sum(axis=None, fill_value=1)
        )

    def test_any(self):
        ds1 = Dataset(
            {
                'A': [True, False, False],
                'B': [True, True, False],
                'C': [False, False, False],
            }
        )
        self.assertTrue(
            (
                ds1.any(as_dataset=False) == Struct({'A': True, 'B': True, 'C': False})
            ).all()
        )
        self.assertTrue(
            (
                ds1.any(axis=0, as_dataset=False)
                == Struct({'A': True, 'B': True, 'C': False})
            ).all()
        )
        self.assertTrue(
            (
                ds1.any(as_dataset=True) == Dataset({'A': True, 'B': True, 'C': False})
            ).all(axis=None)
        )
        self.assertTrue(
            (
                ds1.any(axis=0, as_dataset=True)
                == Dataset({'A': True, 'B': True, 'C': False})
            ).all(axis=None)
        )
        self.assertTrue(np.all(ds1.any(axis=1) == [True, True, False]))
        self.assertTrue(ds1.any(axis=None))
        self.assertFalse(
            Dataset({}).any(axis=None)
        )  # there does not exist one which is true
        ds1.S = ['X', 'Y', '']
        self.assertTrue(
            (
                ds1.any(axis=0, as_dataset=False)
                == Struct({'A': True, 'B': True, 'C': False, 'S': True})
            ).all()
        )

    def test_all(self):
        ds1 = Dataset(
            {
                'A': [True, False, False],
                'B': [True, True, True],
                'C': [False, False, False],
            }
        )
        self.assertTrue(
            (
                ds1.all(as_dataset=False) == Struct({'A': False, 'B': True, 'C': False})
            ).all()
        )
        self.assertTrue(
            (
                ds1.all(axis=0, as_dataset=False)
                == Struct({'A': False, 'B': True, 'C': False})
            ).all()
        )
        self.assertTrue(
            (
                ds1.all(as_dataset=True) == Dataset({'A': False, 'B': True, 'C': False})
            ).all(axis=None)
        )
        self.assertTrue(
            (
                ds1.all(axis=0, as_dataset=True)
                == Dataset({'A': False, 'B': True, 'C': False})
            ).all(axis=None)
        )
        self.assertTrue(np.all(ds1.all(axis=1) == [False, False, False]))
        self.assertTrue(np.all(ds1[[0, 1], ['A', 'B']].all(axis=1) == [True, False]))
        self.assertFalse(ds1.all(axis=None))
        self.assertTrue(ds1[0, ['A', 'B']].all(axis=None))
        self.assertTrue(Dataset({}).all(axis=None))  # all that exist are true
        ds1.S = ['X', 'Y', '']
        self.assertTrue(
            (
                ds1.all(axis=0, as_dataset=False)
                == Struct({'A': False, 'B': True, 'C': False, 'S': False})
            ).all()
        )

    def test_comparison_01(self):
        ds1 = Dataset(
            {_k: list(range(_i * 3, (_i + 1) * 3)) for _i, _k in enumerate('AB')}
        )
        ds2 = Dataset(
            {_k: list(range(_i * 3, (_i + 1) * 3)) for _i, _k in enumerate('AB')}
        )
        ds2[0, 'A'] += 1
        ds2[1, 'B'] += 1
        ds3 = Dataset(
            {_k: list(range(_i * 3, (_i + 1) * 3)) for _i, _k in enumerate('AC')}
        )
        self.assertTrue((ds1 == ds1).all(axis=None))
        self.assertTrue((ds1 == ds1).any(axis=None))
        self.assertFalse((ds1 == ds2).all(axis=None))
        self.assertTrue(
            (
                (ds1 == ds2)
                == Dataset({'A': [False, True, True], 'B': [True, False, True]})
            ).all(axis=None)
        )
        self.assertTrue((ds1 <= ds2).all(axis=None))
        self.assertTrue(
            (
                (ds1 < ds2)
                == Dataset({'A': [True, False, False], 'B': [False, True, False]})
            ).all(axis=None)
        )
        self.assertEqual(list((ds2 == ds3).keys()), ['A', 'B', 'C'])
        self.assertEqual(list((ds3 == ds2).keys()), ['A', 'C', 'B'])
        for comp in 'eq ne lt le gt ge'.split():
            self.assertIsInstance(
                (getattr(ds1, f'__{comp}__')(ds2)).all(axis=None), bool
            )

    def test_comparison_02(self):
        ds1 = Dataset({'A': [5.5], 'B': [20]})
        ds2 = Dataset({'A': [5.5], 'B': ['20']})
        ds3 = Dataset({'A': [6.5], 'B': [30]})
        self.assertTrue(
            ((ds1 < ds3) == Dataset({'A': [True], 'B': [True]})).all(axis=None)
        )
        with self.assertRaises(TypeError):
            _ = ds2 < ds3
        for comp in 'eq ne lt le gt ge'.split():
            self.assertIsInstance(
                (getattr(ds1, f'__{comp}__')(ds3)).all(axis=None), bool
            )

    def test_arith_ops_01(self):
        with self.assertWarns(RuntimeWarning):
            for op in '+= -= *= /= //= %= **='.split():  # @= **=
                for val in (
                    10,
                    5.5,
                    1,
                    0.5,
                    0,
                    -0.5,
                    -1,
                    -5.5 - 10,
                    [1, 2.0, 3],
                    np.array([1, 2.0, 3]),
                    FastArray([1, 2.0, 3]),
                ):
                    ds = self.get_arith_dataset(dtype=np.float32)
                    origds = ds.copy()
                    arr = np.array(ds.tolist())
                    origarr = arr.copy()
                    val2 = (
                        np.reshape(val, (3, -1))
                        if isinstance(val, (list, np.ndarray))
                        else val
                    )
                    self.assertTrue(
                        almost_eq(ds, arr, places=5), msg=f'Failure for base case.'
                    )
                    exec(f'ds {op} val; arr {op} val2')
                    self.assertTrue(
                        almost_eq(ds, arr, places=5),
                        msg=f'Failure for ds {op} {val}\n origds:{origds}\n {origarr}\nds:{ds}\narr:{arr}',
                    )
                    exec(f'ds {op} val2; arr {op} val2')
                    self.assertTrue(
                        almost_eq(ds, arr, places=5),
                        msg=f'Failure for ds {op} {val2}\n origds:{origds}\n {origarr}\nds:{ds}\narr:{arr}',
                    )
            for op in '<<= >>= &= ^= |='.split():
                for val in (
                    10,
                    5,
                    1,
                    0,
                    [1, 2, 3],
                    np.array([1, 2, 3]),
                    FastArray([1, 2, 3]),
                ):
                    ds = self.get_arith_dataset(dtype=np.int32)
                    arr = np.array(ds.tolist())
                    val2 = (
                        np.reshape(val, (3, -1))
                        if isinstance(val, (list, np.ndarray))
                        else val
                    )
                    self.assertTrue(
                        almost_eq(ds, arr, places=5), msg=f'Failure for base case.'
                    )
                    exec(f'ds {op} val; arr {op} val2')
                    self.assertTrue(
                        almost_eq(ds, arr, places=5), msg=f'Failure for ds {op} {val}'
                    )
                    exec(f'ds {op} val2; arr {op} val2')
                    self.assertTrue(
                        almost_eq(ds, arr, places=5), msg=f'Failure for ds {op} {val2}'
                    )

    def test_arith_ops_02(self):
        with self.assertWarns(RuntimeWarning):
            ds = self.get_arith_dataset(dtype=np.float32)
            arr = np.array(ds.tolist())
            self.assertTrue(almost_eq(ds, arr, places=5), msg=f'Failure for base case.')
            for op in '+ - * / // % **'.split():  # @
                for val in (
                    10,
                    5.5,
                    1,
                    0.5,
                    0,
                    -0.5,
                    -1,
                    -5.5 - 10,
                    [1, 2.0, 3],
                    np.array([1, 2.0, 3]),
                    FastArray([1, 2.0, 3]),
                ):
                    val2 = (
                        np.reshape(val, (3, -1))
                        if isinstance(val, (list, np.ndarray))
                        else val
                    )
                    ds0 = eval(
                        f'ds {op} val'
                    )  # ds {op} val2 fails for op in {'%', '**'}
                    arr0 = eval(f'arr {op} val2')
                    self.assertTrue(
                        almost_eq(ds0, arr0, places=5),
                        msg=f'Failure for ds {op} {val}\n',
                    )
                    ds1 = eval(
                        f'ds {op} val2'
                    )  # ds {op} val2 fails for op in {'%', '**'}
                    self.assertTrue(
                        almost_eq(ds1, arr0, places=5),
                        msg=f'Failure for ds {op} {val2}',
                    )
            ds = self.get_arith_dataset(dtype=np.int32)
            arr = np.array(ds.tolist())
            self.assertTrue(almost_eq(ds, arr, places=5), msg=f'Failure for base case.')
            for op in '<< >> & ^ |'.split():  # @
                for val in (
                    10,
                    5,
                    1,
                    0,
                    [1, 2, 3],
                    np.array([1, 2, 3]),
                    FastArray([1, 2, 3]),
                ):
                    val2 = (
                        np.reshape(val, (3, -1))
                        if isinstance(val, (list, np.ndarray))
                        else val
                    )
                    ds0 = eval(
                        f'ds {op} val'
                    )  # ds {op} val2 fails for op in {'<<', '>>'}
                    arr0 = eval(f'arr {op} val2')
                    self.assertTrue(
                        almost_eq(ds0, arr0, places=5), msg=f'Failure for ds {op} {val}'
                    )
                    ds1 = eval(
                        f'ds {op} val2'
                    )  # ds {op} val2 fails for op in {'<<', '>>'}
                    self.assertTrue(
                        almost_eq(ds1, arr0, places=5),
                        msg=f'Failure for ds {op} {val2}',
                    )

    def test_arith_ops_03(self):
        ds = self.get_arith_dataset()
        arr = np.array(ds.tolist())
        self.assertTrue((-ds).tolist() == (-arr).tolist(), msg=f'Failure for -ds')
        self.assertTrue((+ds).tolist() == (+arr).tolist(), msg=f'Failure for +ds')
        self.assertTrue(
            abs(ds).tolist() == abs(arr).tolist(), msg=f'Failure for abs(ds)'
        )
        self.assertTrue(
            ds.abs().tolist() == abs(ds).tolist(), msg=f'Failure for ds.abs()'
        )
        ds = self.get_arith_dataset(dtype=np.int32)
        arr = np.array(ds.tolist())
        self.assertTrue((~ds).tolist() == (~arr).tolist(), msg=f'Failure for ~ds')

    def test_arith_ops_04(self):
        ds0 = self.get_arith_dataset(include_strings=False)
        ds1 = self.get_arith_dataset(include_strings=True)
        arith_cnames = list(ds0.keys())
        str_cnames = [_c for _c in ds1 if _c not in ds0]
        orig_strs_cols = ds1[str_cnames].copy(deep=True)
        self.assertTrue((ds0.abs() == ds1.abs()[arith_cnames]).all(axis=None))
        self.assertTrue((ds1.abs()[str_cnames] == orig_strs_cols).all(axis=None))
        self.assertTrue(((ds0 + 10) == (ds1 + 10)[arith_cnames]).all(axis=None))
        self.assertTrue(((ds1 + 10)[str_cnames] == orig_strs_cols).all(axis=None))
        ds0 += 5
        ds1 += 5
        self.assertTrue((ds0 == ds1[arith_cnames]).all(axis=None))
        self.assertTrue((ds1[str_cnames] == orig_strs_cols).all(axis=None))

    def test_arith_ops_05(self):
        ds1 = self.get_arith_dataset(include_strings=True)
        for wrong in (
            None,
            [0, [1, 2], 3],
            np.ones(ds1.shape),
            np.ones((1, ds1.get_ncols())),
        ):
            with self.assertRaises((ValueError, KeyError, TypeError)):
                _ = ds1 + wrong
        st1 = Struct({'A': 1, 'B': 2, 'C1': 3, 'G': np.arange(1, 4)})
        ds2 = ds1 + st1
        self.assertEqual((ds1.A + 1).tolist(), ds2.A.tolist())
        self.assertEqual((ds1.B + 2).tolist(), ds2.B.tolist())
        self.assertEqual(ds1.C.tolist(), ds2.C.tolist())
        self.assertEqual((ds1.G + [1, 2, 3]).tolist(), ds2.G.tolist())
        self.assertEqual((ds1.S).tolist(), ds2.S.tolist())
        self.assertEqual((ds1.U).tolist(), ds2.U.tolist())
        ds3 = 2 * ds1
        ds4 = ds1 + ds3
        self.assertEqual((ds1.A * 3).tolist(), ds4.A.tolist())
        self.assertEqual((ds1.B * 3).tolist(), ds4.B.tolist())
        self.assertEqual((ds1.C * 3).tolist(), ds4.C.tolist())
        self.assertEqual((ds1.G * 3).tolist(), ds4.G.tolist())
        self.assertEqual((ds1.S).tolist(), ds4.S.tolist())
        self.assertEqual((ds1.U).tolist(), ds4.U.tolist())
        ds1 += st1
        self.assertTrue((ds1 == ds2).all(axis=None))

    def test_arith_ops_06(self):
        ds1 = self.get_arith_dataset(dtype=np.int32)
        self.assertTrue(((ds1 * 3) == (3 * ds1)).all(axis=None))
        self.assertTrue(((ds1 + 3) == (3 + ds1)).all(axis=None))
        self.assertTrue(((-ds1 + 3) == (3 - ds1)).all(axis=None))
        self.assertTrue(((ds1 | 3) == (3 | ds1)).all(axis=None))
        self.assertTrue(((ds1 & 3) == (3 & ds1)).all(axis=None))
        self.assertTrue(((ds1 ^ 3) == (3 ^ ds1)).all(axis=None))

    def test_arith_ops_07(self):
        for dtype in (np.int32, np.float32):
            # import pdb; pdb.set_trace()
            rhs = self.get_arith_dataset(dtype=dtype, include_strings=True)
            rhs[0, 'A'] = 1  # no need for div-by-zero warnings in this test
            prec = 5
            for lhs in (3, 5.5):  # used in eval
                for (
                    op
                ) in (
                    '/ // % **'.split()
                ):  # ** should be last as it modifies rhs and prec.
                    if op == '**':
                        rhs = rhs.abs()
                        rhs[2, :] /= 2  # don't get too big!
                        prec = 1
                    res = eval(f'lhs {op} rhs')
                    for _cn, _col in res.items():
                        if (
                            _col.dtype.char in NumpyCharTypes.Noncomputable
                            or isinstance(_col, Categorical)
                        ):
                            self.assertTrue((_col == rhs[_cn]).all())
                            continue
                        for _r in range(len(_col)):
                            elem = rhs[_r, _cn]  # used in eval
                            self.assertAlmostEqual(
                                _col[_r],
                                eval(f'lhs {op} elem'),
                                places=prec,
                                msg=f'operation {op} elem {elem}',
                            )

    def test_arith_ops_08(self):
        ds1 = self.get_basic_dataset(nrows=100).astype(np.float32)
        ds1.S = ds1.a.astype(str)
        ds2 = ds1.copy()
        ds1[2, :] /= 2
        self.assertTrue((ds1[2, :] == ds2[2, :] / 2).all(axis=None))
        ds1[3, ['e', 'f']] /= 3
        self.assertTrue((ds1[3, ['e', 'f']] == ds2[3, ['e', 'f']] / 3).all(axis=None))
        ds1[4:10:2, :] /= 4
        self.assertTrue((ds1[4:10:2, :] == ds2[4:10:2, :] / 4).all(axis=None))
        ds1[12:16:2, 10:12] /= 5
        self.assertTrue((ds1[12:16:2, 10:12] == ds2[12:16:2, 10:12] / 5).all(axis=None))
        self.assertTrue((ds1 == ds2)[1:9:2, :4].all(axis=None))
        self.assertTrue((ds1 == ds2)[1:9:2, 6:].all(axis=None))

    def test_dataset_objects(self):
        ds = Dataset({'float_obj': np.array([1.0, 2.0, 3.0], dtype=object)})
        self.assertTrue(ds.float_obj.dtype.char in NumpyCharTypes.AllFloat)

        # mixed object will always default to flip to string now SJK 3/7/2019
        ds = Dataset({'mixed_object': np.array([np.nan, 'str', 1], dtype=object)})
        self.assertTrue(ds.mixed_object.dtype.char == 'S')

        ds = Dataset(
            {'mixed_string_start': np.array(['str', np.nan, 1], dtype=object)}
        )

    def test_sample(self):
        arr = tile(FastArray([10, 20]), 100)
        ds = Dataset({'arr': arr})
        f = logical(arange(200) % 2)
        s = ds.sample(N=50, filter=f)
        self.assertEqual(s._nrows, 50)
        self.assertTrue(bool(np.all(s.arr == 20)))

    def test_dataset_pandas(self):
        import pandas as pd

        df = pd.DataFrame({"A": ["a", "b", "c", "a"]})
        df["B"] = df["A"].astype('category')
        ds = Dataset({'A': df.A, 'B': df.B})
        self.assertIsInstance(ds.B, TypeRegister.Categorical)
        self.assertTrue((df.A == ds.A.astype(str)).all())
        self.assertTrue((df.B == ds.B.as_string_array.astype(str)).all())

    def _test_output(self):
        ds = self.get_basic_dataset()[:4, :3]
        self.assertEqual(
            str(ds),
            '''#   a    b    c
-   -   --   --
0   0   10   20
1   1   11   21
2   2   12   22
3   3   13   23''',
        )
        self.assertTrue(
            re.match(
                r'\[4 rows x 3 columns\]\s+total\s+bytes:\s+[\d.]+\s+[KMGTPEZY]?B',
                ds._last_row_stats(),
            )
        )

    def test_head_tail(self):
        ds = self.get_basic_dataset(30)
        self.assertEqual(ds.head().shape, (20, ds.get_ncols()))
        self.assertEqual(ds.tail().shape, (20, ds.get_ncols()))
        self.assertEqual(ds.head(n=5).shape, (5, ds.get_ncols()))
        self.assertEqual(ds.tail(n=25).shape, (25, ds.get_ncols()))

    def test_drop_duplicates(self):
        import pandas as pd

        ds = Dataset(
            {
                'strcol': np.random.choice(['a', 'b', 'c', 'd'], 15),
                'intcol': np.random.randint(0, 3, 15),
                'rand': np.random.rand(15),
            }
        )
        df = pd.DataFrame(ds.asdict())

        rt_drop_first = ds.drop_duplicates(['strcol', 'intcol'], keep='first')
        pd_drop_first = df.drop_duplicates(['strcol', 'intcol'], keep='first')

        for colname in ds:
            rtcol = rt_drop_first[colname]
            pdcol = pd_drop_first[colname]

            self.assertTrue(bool(np.all(rtcol == pdcol.values)))

    def test_drop_duplicates_empty_dataset(self):
        empty_ds = Dataset({'a': [], 'b': []})
        new_ds = empty_ds.drop_duplicates('a')
        self.assertIsNot(new_ds, empty_ds)
        self_ds = empty_ds.drop_duplicates('a', inplace=True)
        self.assertIs(self_ds, empty_ds)

    def test_describe(self):
        ds = self.get_basic_dataset()
        # TODO NW Should be String, not FastArray
        ds.b = FastArray(ds.b.astype(str))
        labels = describe(None)
        col_a_res = np.array(
            [10.0, 10.0, 0.0, 4.5, 3.028, 0.0, 0.9, 2.25, 4.5, 6.75, 8.1, 9.0, 4.5]
        )
        col_c_res = col_a_res + 20.0
        col_c_res[:2] = col_a_res[:2]

        # TJD: test assumes intimate of describe display order, needs to be fixed
        col_c_res[2] = col_a_res[2]
        col_c_res[4] = col_a_res[4]

        def norm(v):
            return np.sqrt(v.dot(v))

        #
        res = ds.describe()
        self.assertIsInstance(res, Dataset)
        self.assertEqual(
            res.shape, (len(labels), ds.get_ncols())
        )  # one added, one dropped
        self.assertIsInstance(res.Stats, FastArray)
        # TODO NW Decode not needed if Stats were a String and not a FastArray
        self.assertEqual(
            [s.decode('unicode_escape') for s in res.Stats.tolist()], labels
        )
        self.assertAlmostEqual(norm(res.a - col_a_res), 0, places=3)
        self.assertNotIn('b', res)
        # print("***",res.c)
        # print("***",col_c_res)
        self.assertAlmostEqual(norm(res.c - col_c_res), 0, places=3)
        #
        # TODO NW When using Strings, fill_value should be Categorical.NODATA
        res = ds.describe(fill_value=INVALID_DICT[ds.b.dtype.num])
        self.assertIsInstance(res, Dataset)
        self.assertEqual(
            res.shape, (len(labels), ds.get_ncols() + 1)
        )  # one added, none dropped
        self.assertIsInstance(res.Stats, FastArray)
        # TODO NW Decode not needed if Stats were a String and not a FastArray
        self.assertEqual(
            [s.decode('unicode_escape') for s in res.Stats.tolist()], labels
        )
        self.assertAlmostEqual(norm(res.a - col_a_res), 0, places=3)
        # TODO NW Decode not needed if String used above for ds.b
        # TODO NW When using Strings, comparison value on right should be Categorical.NODATA
        # TJD: strings are not in describe
        self.assertEqual(res.b.tolist(), [INVALID_DICT[ds.b.dtype.num]] * len(labels))
        self.assertAlmostEqual(norm(res.c - col_c_res), 0, places=3)

        ds = Dataset({'arr': arange(10.0)})
        ds.arr = np.nan
        result = ds.describe().arr
        self.assertEqual(result[0], 10)
        self.assertEqual(result[1], 0)
        self.assertTrue(bool(np.all(isnan(result[2:]))))

    # Regression test for RIP-442 - Error displaying dataset with multiple 'key' columns with zero rows
    def test_repr_multikey_columns_with_zero_rows(self):
        ds = Dataset({
            'K1': Categorical(['A', 'A', 'A']),
            'K2': Categorical(['B', 'B', 'B']),
            'V': [1, 2, 3]
        })
        red_filtered = ds.filter(ds.V == 0)  # zero row Dataset
        # apply a reduction
        red = ds.cat(['K1', 'K2']).first(ds.V)

        # filter resulting in zero rows
        red_filtered = red.filter(red.V == 0)
        assert isinstance(repr(red_filtered), str)

    def test_footer_get_values(self):
        ds = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        nofooters = ds.footer_get_values()
        self.assertEqual(nofooters, {})

        ds.footer_set_values('sum', {'col_1': 10})
        with self.assertWarns(UserWarning):
            missing = ds.footer_get_values('mean')
            self.assertEqual(missing, {})

        allcols = ds.footer_get_values()
        self.assertEqual(len(allcols), 1)
        self.assertEqual(len(allcols['sum']), len(ds))
        self.assertTrue(allcols['sum'][0] is None)
        self.assertEqual(allcols['sum'][1], 10)

        allcols_fill = ds.footer_get_values(fill_value='INV')
        self.assertEqual(allcols_fill['sum'][0], 'INV')

        somecols = ds.footer_get_values(columns='col_0')
        self.assertTrue(somecols['sum'][0] is None)

        somecols = ds.footer_get_values(columns=['col_0', 'col_1'])
        self.assertEqual(somecols['sum'][1], 10)

    def test_footer_set_values(self):
        ds = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        ds.footer_set_values('sum', {'col_1': 10})
        footerdict = ds.col_get_attribute('col_1', 'Footer')
        self.assertEqual(footerdict['sum'], 10)

        ds.footer_set_values('mean', {'col_1': 2.0})
        footerdict = ds.col_get_attribute('col_1', 'Footer')
        self.assertEqual(footerdict['mean'], 2.0)

        footerdict = ds.col_get_attribute('col_0', 'Footer')
        self.assertTrue(footerdict is None)

    def test_footer_remove(self):
        ds = Dataset({'col_' + str(i): arange(5) for i in range(5)})
        ds.footer_set_values(
            'row1', dict(zip(ds, ['footer' + str(i) for i in range(len(ds))]))
        )
        with self.assertWarns(UserWarning):
            ds.footer_remove('row2')

        ds.footer_remove('row1', ['col_3', 'col_1'])
        footer3 = ds.col_get_attribute('col_3', 'Footer')
        self.assertEqual(len(footer3), 0)
        footer1 = ds.col_get_attribute('col_1', 'Footer')
        self.assertEqual(len(footer1), 0)
        footer2 = ds.col_get_attribute('col_2', 'Footer')
        self.assertEqual(footer2['row1'], 'footer2')

        with self.assertRaises(IndexError):
            ds.footer_remove('row1', 'col123')

        ds.footer_remove()
        self.assertTrue(ds.footers is None)

    def test_mask_or_isnan(self):
        d = {
            'intcol': FastArray([-128, 2, 3, -128, 1, 2, 3], dtype=np.int8),
            'fltcol': FastArray([np.nan, 2, np.nan, 4, 1, 2, 3], dtype=np.float32),
        }
        correct = FastArray([True, False, True, True, False, False, False])
        ds = Dataset(d)
        mask = ds.mask_or_isnan()
        self.assertTrue(bool(np.all(mask == correct)))

    def test_output(self):
        ds = self.get_basic_dataset()[:4, :3]

        self.assertEqual(
            str(ds),
            '''#   a    b    c
-   -   --   --
0   0   10   20
1   1   11   21
2   2   12   22
3   3   13   23''',
        )
        self.assertRegex(
            ds._last_row_stats(),
            r'\[4 rows x 3 columns\]\s+total\s+bytes:\s+[\d.]+\s+[KMGTPEZY]?B',
        )
        ds = self.get_arith_dataset(include_strings=True)
        self.assertEqual(
            str(ds),
            '#   A      B       G       C   S           U        \n-   -   ----   -----   -----   ---------   ---------\n0   0   1.20   -1.60    2.40   string_00   ℙƴ☂ℌøἤ_00\n1   6   3.10    2.70    6.20   string_01   ℙƴ☂ℌøἤ_01\n2   9   9.60    4.60   19.20   string_02   ℙƴ☂ℌøἤ_02',
        )
        self.assertRegex(
            ds._last_row_stats(),
            r'\[3 rows x 6 columns\]\s+total\s+bytes:\s+[\d.]+\s+[KMGTPEZY]?B',
        )

    def test_to_pandas(self):
        cols = ['c', 'b', 'a']
        ds = Dataset(dict([('c', [4, 5]), ('b', [2, 3]), ('a', [0, 1])]))
        self.assertEqual(list(ds.keys()), cols)
        df = ds.to_pandas()
        self.assertEqual(list(df.columns), cols)
        self.assertIsInstance(df.a, pd.Series)
        self.assertIsInstance(df.b, pd.Series)
        self.assertIsInstance(df.c, pd.Series)
        self.assertTrue((df.a == [0, 1]).all())
        self.assertTrue((df.b == [2, 3]).all())
        self.assertTrue((df.c == [4, 5]).all())

    def test_from_pandas(self):
        cols = ['c', 'b', 'a']
        df = pd.DataFrame({'a': [0, 1], 'b': [2, 3], 'c': [4, 5]}, columns=cols)
        ds = Dataset.from_pandas(df)
        self.assertEqual(list(ds.keys()), cols)
        check_type = FastArray if Dataset.UseFastArray else np.ndarray
        self.assertIsInstance(ds.a, check_type)
        self.assertIsInstance(ds.b, check_type)
        self.assertIsInstance(ds.c, check_type)
        self.assertTrue((ds.a == [0, 1]).all())
        self.assertTrue((ds.b == [2, 3]).all())
        self.assertTrue((ds.c == [4, 5]).all())

    def test_ds_df_roundtrip_with_categoricals_and_datetimes(self):
        from enum import IntEnum

        def _bytes_to_string(arr):
            if arr.dtype.char == 'S':
                arr = arr.astype('U')
            return arr

        def _datetime_to_int(arr):
            return arr.apply(lambda x: x.strftime('%Y%m%d')).astype(int)

        # Create a DataFrame
        df = pd.DataFrame({"A": ["a", "b", "c", "a"]})
        df["B"] = df["A"].astype('category')
        codes = [0, 1, 0, 2]
        df["C"] = pd.Categorical.from_codes(codes, list('qwe'))
        df["D"] = pd.Categorical.from_codes(codes, np.arange(3) ** 2)
        df["E"] = pd.Categorical.from_codes(codes, np.arange(0, 0.3, 0.1))
        df["F"] = pd.Categorical.from_codes(codes, np.arange(3, dtype='float32') ** 2)
        df["G"] = pd.Series(pd.date_range('20190101', periods=4))
        for (key, tz) in [
            ('H', 'UTC'),
            ('I', 'GMT'),
            ('J', 'US/Eastern'),
            ('K', 'Europe/Dublin'),
        ]:
            df[key] = pd.Series(pd.date_range('20190101', periods=4, tz=tz))

        # Create a Dataset from the DataFrame and compare
        ds = Dataset.from_pandas(df)
        self.assertIsInstance(ds.B, TypeRegister.Categorical)
        self.assertIsInstance(ds.C, TypeRegister.Categorical)
        self.assertTrue((df.A == ds.A.astype(str)).all())
        for key in 'BCDEF':
            self.assertTrue((df[key] == _bytes_to_string(ds[key].expand_array)).all())
        for (key, tz) in [
            ('G', 'UTC'),
            ('H', 'UTC'),
            ('I', 'GMT'),
            ('J', 'NYC'),
            ('K', 'DUBLIN'),
        ]:
            self.assertTrue((_datetime_to_int(df[key]) == ds[key].yyyymmdd).all())
            self.assertTrue(ds[key]._timezone._from_tz == 'UTC')
            self.assertTrue(ds[key]._timezone._to_tz == tz)

        # Make two of the categoricals into enum type
        class MyEnum(IntEnum):
            a = 0
            b = 1
            c = 2

        ds.B = TypeRegister.Categorical([0, 1, 2, 0], MyEnum)
        ds.C = TypeRegister.Categorical(codes, {'q': 0, 'w': 1, 'e': 2})

        # Create a DataFrame from the Dataset and compare with the original
        # DataFrame (ie. test the round-trip)
        df2 = ds.to_pandas()
        self.assertEqual(df2.B.dtype.name, 'category')
        self.assertEqual(df2.C.dtype.name, 'category')
        for key in df:
            # Timezone unaware pandas datetime should get converted to UTC in riptable
            if key == 'G':
                self.assertTrue(
                    (pd.DatetimeIndex(df[key]).tz_localize('UTC') == df2[key]).all()
                )
            else:
                self.assertTrue((df[key] == df2[key]).all())

    def test_to_pandas_categorical_mapping(self):
        country_map = {0: "USA", 2: "IRL", 4: "GBR", 8: "AUS", 16: "CHN", 32: "JPN"}
        k = "dict_cat"
        ds = Dataset({k: Categorical([0, 2, 32], categories=country_map)})
        df = ds.to_pandas()
        ds_from_pandas = Dataset.from_pandas(df)
        assert_array_equal(ds[k].as_singlekey().expand_array, ds_from_pandas[k].expand_array)
        assert_array_equal(ds[k].category_array, ds_from_pandas[k].category_array)

    def test_from_pandas_num_string(self):
        dates = ['20190101', '20190101']
        df = pd.DataFrame({'a': dates})
        ds = Dataset.from_pandas(df)
        assert_array_equal(ds['a'], np.array(dates, dtype='S'))

    def test_from_pandas_obj_number(self):
        df = pd.DataFrame({'a': np.array([1, 2, np.nan], dtype='O')})
        ds = Dataset.from_pandas(df)
        assert_array_almost_equal(ds['a'], np.array([1.0, 2.0, np.nan]))

    def test_to_pandas_timespan(self):
        dt_list = [11096000000000, 86401000000000]
        ds = Dataset({'a': TimeSpan(dt_list)})
        df = ds.to_pandas()
        assert_array_equal(df['a'], pd.to_timedelta(dt_list))

    def test_from_pandas_timespan(self):
        dt_list = [11096000000000, 86401000000000]
        df = pd.DataFrame({'a': pd.to_timedelta(dt_list)})
        ds = Dataset.from_pandas(df)
        assert_array_equal(ds['a'], TimeSpan(dt_list))

    def test_to_pandas_nullable_int(self):
        for dtype_str, pd_dtype_str in zip(
            ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'],
            ['Int8', 'Int16', 'Int32', 'Int64', 'UInt8', 'UInt16', 'UInt32', 'UInt64'],
        ):
            dtype = np.dtype(dtype_str)
            invalid_value = INVALID_DICT[dtype.num]
            ds = Dataset({'a': FastArray([1, 2, invalid_value], dtype=dtype)})
            # not all versions of pandas have .array
            if hasattr(pd, 'array'):
                df = ds.to_pandas()
                expected_arr = pd.Series([1, 2, np.nan], dtype=pd_dtype_str)
                self.assertTrue(
                    df['a'].equals(expected_arr),
                    msg=f'actual:\n{df.a}\nexpected:\n{expected_arr}',
                )

    def test_to_pandas_keep_invalid_sentinel(self):
        for dtype_str in ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8']:
            dtype = np.dtype(dtype_str)
            invalid_value = INVALID_DICT[dtype.num]
            ds = Dataset({'a': FastArray([1, 2, invalid_value], dtype=dtype)})
            df = ds.to_pandas(use_nullable=False)
            expected_arr = pd.Series([1, 2, invalid_value], dtype=dtype)
            self.assertTrue(
                df['a'].equals(expected_arr),
                msg=f'actual:\n{df.a}\nexpected:\n{expected_arr}',
            )

    def test_from_pandas_nullable_int(self):
        # not all versions of pandas have .array
        if hasattr(pd, 'array'):
            for dtype_str, pd_dtype_str in zip(
                [
                    'int8',
                    'int16',
                    'int32',
                    'int64',
                    'uint8',
                    'uint16',
                    'uint32',
                    'uint64',
                ],
                [
                    'Int8',
                    'Int16',
                    'Int32',
                    'Int64',
                    'UInt8',
                    'UInt16',
                    'UInt32',
                    'UInt64',
                ],
            ):
                df = pd.DataFrame({'a': pd.array([1, 2, np.nan], dtype=pd_dtype_str)})
                ds = Dataset.from_pandas(df)
                expected_arr = FastArray([1, 2, 3], dtype=dtype_str)
                expected_arr[2] = INVALID_DICT[expected_arr.dtype.num]
                assert_array_equal(ds['a'], expected_arr)

    def test_from_pandas_no_index_zero_string_column(self):
        df = pd.DataFrame({'a': [b'a', b'b']}, index=[1, 2])
        ds = Dataset.from_pandas(df)
        assert_array_equal(ds['a'], np.array([b'a', b'b']))

    def test_as_pandas_df_warn(self):
        ds = Dataset({'a': [1, 2, 3]})
        with self.assertWarns(FutureWarning):
            df = ds.as_pandas_df()

    def test_imatrix(self):
        arrsize = 3
        ds = Dataset({'time': arange(arrsize * 1.0), 'data': arange(arrsize)})
        im = ds.imatrix_make(dtype=np.int32)
        assert_array_equal(im[:, 0], arange(arrsize, dtype=np.int32))

        im = ds.imatrix_make(dtype=np.float32)
        assert_array_equal(im[:, 1], arange(arrsize, dtype=np.float32))

    def test_imatrix_isnan(self):
        ds = Dataset({'a': [np.nan, np.nan, np.nan], 'b': arange(3)})
        x = ds.imatrix_make()
        assert_array_equal(x.isnan(), np.isnan(x))

        ds = Dataset({'a': [np.nan, np.nan, np.nan], 'b': arange(3) + 1})
        x = ds.imatrix_make()
        result = ds.imatrix_make().fillna(0)

        assert_array_equal(x.isnan(), np.isnan(x))
        test = np.asfortranarray([[0, 1,], [0, 2], [0, 3]])
        self.assertTrue(np.all(test == result))

    def test_dataset_footer_copy(self):
        ds = Dataset({'x1': ['A', 'B']})
        ds.footer_set_values('sum', {'x1': 3})
        ds.footer_remove()
        ds.copy()

    def test_equals(self):
        ds = Dataset({'a': [True, True, False, True], 'b': [False, False, True, False]})
        ds2 = Dataset(
            {'a': [True, True, True, True], 'b': [False, False, False, False]}
        )
        self.assertTrue(ds.equals(ds))
        self.assertTrue(ds.equals(ds2) == False)

    def test_pivot2(self):
        ds2 = Dataset(
            {
                'date': [20190101] * 4,
                'symbol': ['AAPL', 'MSFT', 'EBAY', 'MSFT'],
                'test': ['n1', 'n2', 'n3', 'n2'],
                'shares': [100, 200, 300, 200],
            }
        )
        with self.assertRaises(ValueError):
            ds2.pivot('test', columns='symbol', values='shares')

        ds = Dataset(
            {
                'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                'baz': [1, 2, 3, 4, 5, 6],
                'zoo': ['x', 'y', 'z', 'q', 'w', 't'],
            }
        )

        ms = ds.pivot(labels='foo', columns='bar')
        newds = ds.pivot(labels='foo', columns='bar', values='baz')
        self.assertTrue(np.all(newds.A == FastArray([1, 4])))

    def test_allnames(self):
        previous = Struct.AllNames
        Struct.AllNames = True
        ds = Dataset({'a': arange(5), '_test': arange(5.0)})
        self.assertTrue(np.all(ds._test == arange(5.0)))
        ds['sum'] = 3
        self.assertTrue(np.all(ds['sum'] == FastArray([3, 3, 3, 3, 3])))
        Struct.AllNames = previous

    def test_scalar_expansion(self):
        ds = Dataset({'a': arange(5), 'b': arange(5.0), 'c': list('abcde')})
        x = utcnow(1)
        # test adding a scalar or array of 1 to dataset of row lwength 5
        ds.x = x
        y = x[0]
        ds.y = y
        self.assertTrue(np.all(ds.x == ds.y))
        z = x - utcnow(1)
        ds.z = -z
        ds.z1 = -z[0]
        self.assertTrue(np.all(ds.z == ds.z1))

    def test_pandas_objectarray(self):
        import pandas as pd

        df = pd.DataFrame({'test': np.arange(10).astype('U')})
        x = Dataset(df)
        self.assertTrue(x.test[1] == b'1')
        x = Dataset(df, unicode=True)
        self.assertTrue(x.test[1] == '1')

    def test_copyinplace(self):
        ds = Dataset()
        # also check that len returns 0 instead of None
        self.assertTrue(len(ds)==0)
        ds.MyStrKey = Categorical(list('ABCD'))
        ds.MyIntKey = [1, 2, 3, 4]
        ds.MyDate = Date([1, 2, 3, 4])
        ds.MyValue = 1, 2, 1, 2
        ds.filter(ds.MyValue == 1, inplace=True)
        result = ds.cat(['MyStrKey', 'MyDate']).null()
        self.assertTrue(isinstance(result['MyDate'], Date))

    def test_underscorename(self):
        ds=Dataset()
        ds.num=arange(10)
        with self.assertRaises(IndexError):
            ds['_num']=arange(10)

    def test_overwrite_column_with_scalar(self) -> None:
        # Test that a Dataset column can be overwritten with a scalar,
        # and that scalar is broadcast to the correct size to match the Dataset.
        # This is a regression test for some behavior that appears to be broken by
        # https://github.com/rtosholdings/riptable/commit/811960b3de521e19a1945602fb5a8b2193845b1d
        ds = rt.Dataset({
            'a': rt.full(20, 1.2345, dtype=np.float32),
            'b': rt.arange(20, dtype=np.uint64),
            'c': rt.FA([11, -13, -17, 19, 23]).tile(4)
        })
        orig_rowcount = ds.get_nrows()

        # Overwrite each of the columns with a scalar value.
        scalars = {
            'a': np.int16(12345), 'b': True, 'c': "hello"
        }
        assert set(ds.keys()) == set(scalars.keys())

        for col_name in ds.keys():
            ds[col_name] = scalars[col_name]

        assert ds.get_nrows() == orig_rowcount

        # Make sure each of the columns has the expected type and value
        # given the scalar that was assigned to it.
        for col_name in ds.keys():
            col = ds[col_name]
            scalar_value = scalars[col_name]
            expected_dtype = scalar_value.dtype if isinstance(scalar_value, np.generic) else np.min_scalar_type(scalar_value)
            assert col.dtype == expected_dtype
            assert_array_equal(col, scalar_value)



@pytest.mark.parametrize('categorical', get_all_categorical_data())
def test_dataset_to_dataframe_roundtripping(categorical):
    k = 'categorical'
    ds = Dataset({k: categorical})
    df = ds.to_pandas()
    ds_from_pandas = Dataset.from_pandas(df)
    assert_array_equal(ds[k].as_singlekey().expand_array, ds_from_pandas[k].expand_array)
    # add support for checking category arrays of multikey categoricals
    if categorical.category_mode != CategoryMode.MultiKey:
        assert_array_equal(ds[k].category_array, ds_from_pandas[k].category_array)


if __name__ == "__main__":
    tester = unittest.main()
