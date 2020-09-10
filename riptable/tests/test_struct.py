# $Id: //Depot/Source/SFW/riptable/Python/core/riptable/tests/test_struct.py#14 $


import unittest
import keyword
import sys
import re
from collections import OrderedDict

# from io import StringIO
import numpy as np
from riptable import Struct, FastArray, Dataset, Categorical
from riptable.rt_enum import TypeRegister
from riptable.rt_numpy import arange


class Struct_Test(unittest.TestCase):
    def test_col_ctor_01(self):
        dict1 = {_k: [_i] for _i, _k in enumerate('bac')}
        st = Struct(dict1)
        self.assertEqual(list(st.keys()), list(dict1))
        dict2 = {_k: [_i] for _i, _k in enumerate(['a', '_b', 'μεαν'])}
        with self.assertRaises(ValueError):
            st = Struct(dict2)
        with self.assertRaises(TypeError):
            _ = Struct([])
        with self.assertRaises(TypeError):
            _ = Struct(None)
        with self.assertRaises(TypeError):
            _ = Struct(np.array([1]))
        st = Struct()
        self.assertEqual(st.shape, (0, 0))

    # changed invalid name behavior
    # allowing all to pass through, no methods will be lost
    # 04/05/2019
    # def test_col_ctor_02(self):
    #    tempsave = Struct.AllowAnyName
    #    Struct.AllowAnyName=False
    #    hold_WarnOnInvalidNames = Struct.WarnOnInvalidNames
    #    Struct.WarnOnInvalidNames = False
    #    with self.assertRaises(ValueError):
    #        _ = Struct({'a': [0], '0': [1]})
    #    with self.assertRaises(ValueError):
    #        _ = Struct({'a': [0], 'a-b-c': [1]})
    #    with self.assertRaises(ValueError):
    #        # should never be called this way in normal course of events!
    #        struct_instance = Struct({})
    #        struct_instance._validate_names(['a', 'b', 'a'])
    #    # We auto captialize now
    #    #for kwd in keyword.kwlist:
    #    #    with self.assertRaises(ValueError):
    #    #        _ = Struct({'a': [0], kwd: [1]})
    #    #for memb in dir(Struct):
    #    #    with self.assertRaises(ValueError):
    #    #        _ = Struct({'a': [0], memb: [1]})
    #    Struct.WarnOnInvalidNames = hold_WarnOnInvalidNames
    #    Struct.AllowAnyName = tempsave

    def test_col_ctor_02(self):
        inv_keys = ['True', 'False', 'None']
        arr = arange(5)
        inv_dict = {k: arr for k in inv_keys}
        with self.assertWarns(UserWarning):
            st = Struct(inv_dict)

        self.assertTrue(bool(np.all(inv_keys == list(st))))
        for k in inv_keys:
            self.assertTrue(bool(np.all(st[k] == arr)))

    def test_col_ctor_03(self):
        tempsave = Struct.AllowAnyName
        Struct.AllowAnyName = False
        hold_WarnOnInvalidNames = Struct.WarnOnInvalidNames
        Struct.WarnOnInvalidNames = True
        with self.assertWarns(UserWarning):
            _ = Struct({'a': [0], '0': [1]})
        with self.assertWarns(UserWarning):
            _ = Struct({'a': [0], 'a-b-c': [1]})
        with self.assertRaises(ValueError):  # still should fail even here!
            # should never be called this way in normal course of events!
            struct_instance = Struct({})
            struct_instance._validate_names(['a', 'b', 'a'])
        for kwd in keyword.kwlist:
            with self.assertWarns(UserWarning):
                _ = Struct({'a': [0], kwd: [1]})
        for memb in dir(Struct):
            with self.assertWarns(UserWarning):
                # We expect really stupid errors resulting from not avoiding invalid names.
                try:
                    _ = Struct({'a': [0], memb: [1]})
                except (TypeError, AttributeError):
                    pass
        Struct.WarnOnInvalidNames = hold_WarnOnInvalidNames
        Struct.AllowAnyName = tempsave

    def test_col_ctor_04(self):
        hold_UseFastArray = Struct.UseFastArray
        arr = np.ones(1000).reshape(5, -1).T
        self.assertFalse(arr.flags.c_contiguous)
        Struct.set_fast_array(True)
        st1 = Struct({'a': arr})
        self.assertFalse(st1.a.flags.c_contiguous)
        Struct.set_fast_array(False)
        st2 = Struct({'a': arr})
        self.assertFalse(st2.a.flags.c_contiguous)
        Struct.set_fast_array(hold_UseFastArray)

    def test_lock(self):
        st1 = Struct({_k: [_i] for _i, _k in enumerate(['a', 'b', 'c'])})
        st1._lock()
        with self.assertRaises(AttributeError):
            st1.a = 1
        with self.assertRaises(AttributeError):
            st1.d = 1
        with self.assertRaises(AttributeError):
            st1['a'] = 1
        with self.assertRaises(AttributeError):
            st1['d'] = 1
        with self.assertRaises(AttributeError):
            del st1.b
        with self.assertRaises(AttributeError):
            st1.col_remove('c')
        with self.assertRaises(AttributeError):
            _ = st1.col_pop('c')
        with self.assertRaises(AttributeError):
            _ = st1.col_rename('c', 'C')
        with self.assertRaises(AttributeError):
            _ = st1.col_map({})
        with self.assertRaises(AttributeError):
            _ = st1.col_move_to_back('c')
        with self.assertRaises(AttributeError):
            _ = st1.col_move_to_front('c')
        st1._unlock()
        st1.a = 1
        st1.d = 1
        del st1.b
        self.assertEqual(list(st1.keys()), ['a', 'c', 'd'])
        self.assertEqual(st1.a, 1)
        self.assertEqual(st1.d, 1)

    def test_context_interface(self):
        with self.assertRaises(AttributeError):
            with Struct({_k: [_i] for _i, _k in enumerate(['a', 'b', 'c'])}) as st1:
                self.assertIsInstance(st1, Struct)
                self.assertEqual(list(st1.keys()), ['a', 'b', 'c'])
                st1._lock()
                st1.a = 'cannot modify when locked'
        self.assertEqual(st1.get_ncols(), 0)
        self.assertFalse(st1.is_locked())

    def test_basic_interface(self):
        cols = ['a', 'b', 'c', 'μεαν']
        dict1 = {_k: [_i] for _i, _k in enumerate(cols)}
        st1 = Struct(dict1)
        self.assertEqual(list(st1.keys()), cols)
        self.assertEqual(list(st1.keys()), cols)
        self.assertEqual(list(st1), cols)
        for _idx, (_k, _v) in enumerate(st1.items()):
            self.assertEqual(st1[_k], dict1[_k])
            self.assertEqual(cols[_idx], _k)
            self.assertEqual(st1[_idx], dict1[_k])
            self.assertEqual(getattr(st1, _k), dict1[_k])
        self.assertEqual(list(st1.keys()), cols)
        self.assertEqual(list(st1), cols)
        self.assertEqual(list(reversed(st1)), list(reversed(cols)))
        self.assertEqual(st1.get_nrows(), 0)
        self.assertEqual(st1.shape, (0, st1.get_ncols()))
        self.assertEqual(len(st1), st1.get_ncols())
        st1['a'] = -999
        self.assertEqual(st1.a, -999)
        st1['newcol'] = -999
        self.assertEqual(st1.newcol, -999)

    def test_col_indexing(self):
        cols = ['b', 'a', 'c', 'μεαν']
        dict1 = {_k: [_i] for _i, _k in enumerate(cols)}
        st = Struct(dict1)
        self.assertEqual(list(st.keys()), cols)
        self.assertEqual(st['b'], [0])
        self.assertEqual(st[1], [1])
        self.assertEqual(list(st[1:3]), cols[1:3])
        self.assertEqual(list(st[['b', 'c']]), ['b', 'c'])
        self.assertEqual(list(st[[True, False, True, False]]), ['b', 'c'])
        with self.assertRaises(IndexError):
            _ = st[[True, False, True]]
        with self.assertRaises(IndexError):
            _ = st[[True, False, True, True, True]]
        self.assertEqual(st.b, [0])
        self.assertEqual(st.a, [1])
        self.assertEqual(st.c, [2])
        self.assertEqual(st.μεαν, [3])
        # check shallow copy behavior
        st.d = 'stable'
        st1 = st[['a', 'b', 'd']]
        st1.a[0] = 'diff1'
        st1.b = 'diff2'
        st1.d = 'diff3'
        self.assertEqual(st1.tolist(), [['diff1'], 'diff2', 'diff3'])
        self.assertEqual(st.tolist(), [[0], ['diff1'], [2], [3], 'stable'])
        with self.assertRaises(AttributeError):
            _ = st.Q
        with self.assertRaises(IndexError):
            _ = st['Q']
        with self.assertRaises(IndexError):
            _ = st[st.get_ncols()]
        with self.assertRaises(IndexError):
            _ = st[0, 'a']
        with self.assertRaises(IndexError):
            _ = st[['a', 'a']]
        with self.assertRaises(TypeError):
            st[[complex(1)]] = 'complex is naughty'
        st1['a'] = 'now can do this'
        self.assertEqual(st1.a, 'now can do this')
        st1[['a', 'b']] = ['now can do this, too', 'as well as this']
        self.assertEqual(st1.a, 'now can do this, too')
        self.assertEqual(st1.b, 'as well as this')
        st1[['a', 'b']] = ('a', 'b')
        self.assertEqual(st1.a, 'a')
        self.assertEqual(st1.b, 'b')
        st1['now_there'] = 'and even this'
        self.assertEqual(st1.now_there, 'and even this')
        with self.assertRaises(IndexError):
            st1[['a', 'b']] = 'still cannot do this'
        with self.assertRaises(IndexError):
            st1[['a', 'b']] = np.array(['or', 'this'])
        with self.assertRaises(IndexError):
            st1[['a', 'b']] = 'AB'

    def test_col_rename(self):
        tempsave = Struct.AllowAnyName
        Struct.AllowAnyName = False
        dict1 = {_k: [_i] for _i, _k in enumerate('abc')}
        st = Struct(dict1)
        with self.assertRaises(ValueError):
            st.col_rename('b', 'no-longer-b')
        orig_cols = list(st.keys())
        st.col_rename('a', 'a')  # no-op
        with self.assertRaises(ValueError):
            st.col_rename('d', 'c')  # cannot rename non-existent column
        self.assertEqual(list(st.keys()), orig_cols)
        with self.assertRaises(ValueError):
            st.col_rename('a', 'c')
        for kwd in keyword.kwlist:
            with self.assertRaises(ValueError):
                st.col_rename('a', kwd)
        for memb in dir(Struct):
            with self.assertRaises(ValueError):
                st.col_rename('a', memb)
        st.col_rename('b', 'no_longer_b')
        self.assertEqual(list(st.keys()), ['a', 'no_longer_b', 'c'])
        with self.assertRaises(TypeError):
            st.col_rename('a')
        Struct.AllowAnyName = tempsave

    def test_col_map_01(self):
        orig_cols = ['a', 'b', 'c']
        dict1 = {_k: [_i] for _i, _k in enumerate(orig_cols)}
        st = Struct(dict1)
        with self.assertRaises(ValueError):
            st.col_map({'e': 'f'})  # cannot rename non-existent column
        self.assertEqual(list(st.keys()), orig_cols)
        st.col_map({_k: '{}_{}'.format(_k, _i) for _i, _k in enumerate(st)})
        dict2 = {'{}_{}'.format(_k, _i): [_i] for _i, _k in enumerate(orig_cols)}
        st2 = Struct(dict2)
        self.assertEqual(list(st.keys()), list(st2.keys()))
        # dicts no longer equal
        # for _k in st:
        #    self.assertEqual(st.__dict__[_k], st2.__dict__[_k])
        st = Struct(dict1)
        self.assertEqual(list(st.keys()), orig_cols)
        with self.assertRaises(ValueError):
            st.col_map({'a': 'e', 'b': 'd', 'c': 'd'})
            st.col_map({'q': 'z'})  # cannot rename non-existent column
            st.col_map({'q': 'a'})  # cannot rename non-existent column
        with self.assertRaises(TypeError):
            st.col_map({'a': 'e', 'b': 'd', 'c': 1})
        with self.assertRaises(TypeError):
            st.col_map({1: 'e', 'b': 'd', 'c': 1})
        with self.assertRaises(TypeError):
            st.col_map('a')
        self.assertEqual(list(st.keys()), orig_cols)
        st.col_map({'a': 'f', 'b': 'e', 'c': 'd'})
        self.assertEqual(list(st.keys()), ['f', 'e', 'd'])
        self.assertEqual(st.f, [0])
        self.assertEqual(st.e, [1])
        self.assertEqual(st.d, [2])

    def test_col_map_02(self):
        st = Struct(
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

    def test_col_map_03(self):
        # same as test_col_map_02 but w/ transitions in scrambled order
        st = Struct(
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
            'k': 'g',
            'h': 'j',
            'j': 'k',  # 4-cycle
            'n': 'o',
            'm': 'n',
            'o': 'y',  # 3-transition
            'q': 'z',
            'p': 'q',
            'z': 'r',
        }  # false 3-cycle, really a 2-transition
        st.col_map(cmap)
        for _k in 'abcdefghijklmnopqrst':
            self.assertEqual(st[cmap.get(_k, _k)][1], (ord(_k) - ord('a')) * 10 + 1)

    def test_col_swap(self):
        orig_cols = ['a', 'b', 'c', 'd']
        dict1 = {_k: [_i] for _i, _k in enumerate(orig_cols)}
        st = Struct(dict1)
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
        st = Struct(dict1)
        st.col_swap(['a', 'b', 'c', 'd'], ['c', 'b', 'a', 'd'])
        self.assertEqual(st['a'][0], 2)
        self.assertEqual(st['b'][0], 1)
        self.assertEqual(st['c'][0], 0)
        self.assertEqual(st['d'][0], 3)
        self.assertEqual(list(st.keys()), orig_cols)

    def test_col_moves(self):
        st = Struct(
            {
                _k: list(range(_i * 10, (_i + 1) * 10))
                for _i, _k in enumerate('abcdefghijklmnop')
            }
        )

        st.col_move_to_front(1)
        self.assertEqual(list(st), list('bacdefghijklmnop'))
        st.col_move_to_front(1)

        st.col_move_to_back(14)
        self.assertEqual(list(st), list('abcdefghijklmnpo'))
        st.col_move_to_back(14)

        with self.assertRaises(ValueError):
            st.col_move_to_front(arange(20))

        st.col_move_to_back(list('dgh'))
        self.assertEqual(list(st), list('abcefijklmnopdgh'))
        st.col_move_to_front(list('gpha'))
        self.assertEqual(list(st), list('gphabcefijklmnod'))
        st.col_move(list('cim'), list('hfo'))
        self.assertEqual(list(st), list('cimgpabejklndhfo'))
        st.col_move_to_front({'g': 1})
        st.col_move_to_front('h')
        with self.assertWarns(UserWarning):
            st.col_move_to_front('q')
        self.assertEqual(list(st), list('hgcimpabejklndfo'))
        st.col_move_to_back({'g': 1})
        st.col_move_to_back('h')
        with self.assertWarns(UserWarning):
            st.col_move_to_back('q')
        self.assertEqual(list(st), list('cimpabejklndfogh'))

    def test_col_remove(self):
        cols = ['aa', 'b', 'c', 'μεαν']
        dict1 = {_k: [_i] for _i, _k in enumerate(cols)}
        st1 = Struct(dict1)
        self.assertEqual(list(st1.keys()), cols)
        st1.col_remove('aa')
        self.assertEqual(list(st1.keys()), cols[1:])
        with self.assertRaises(IndexError):
            st1.col_remove('aa')
        st1.col_remove(['b', 'c'])
        self.assertEqual(list(st1.keys()), cols[3:])
        with self.assertRaises(IndexError):
            st1.col_remove(['μεαν', 'b', 'c'])
        with self.assertRaises(TypeError):
            st1.col_remove(2)
        self.assertEqual(list(st1.keys()), cols[3:])
        st1.col_remove(['μεαν'])
        self.assertEqual(list(st1.keys()), [])
        st1.col_remove([])
        self.assertEqual(list(st1.keys()), [])

    def test_col_pop(self):
        cols = ['aa', 'b', 'c', 'μεαν']
        dict1 = {_k: [_i] for _i, _k in enumerate(cols)}
        st1 = Struct(dict1)
        val = st1.col_pop('aa')
        self.assertEqual(val, [0])
        self.assertEqual(list(st1.keys()), cols[1:])
        with self.assertRaises(IndexError):
            st1.col_pop('aa')
        with self.assertRaises(IndexError):
            st1.col_pop(['aa'])
        st2 = st1.col_pop(['b', 'c'])
        self.assertEqual(list(st1.keys()), cols[3:])
        self.assertEqual(list(st2.keys()), ['b', 'c'])
        self.assertEqual(st2.b, [1])
        self.assertEqual(st2.c, [2])
        st3 = st1.col_pop(slice(None))
        self.assertEqual(st1.get_ncols(), 0)
        self.assertEqual(st3.get_ncols(), 1)
        self.assertEqual(st3.μεαν, [3])

    def test_col_str_match(self):
        cols = ['apple', 'orange', 'kumquat', 'koala', 'μεαν', 'two_kumquats_or_more']
        dict1 = {_k: [_i] for _i, _k in enumerate(cols)}
        st1 = Struct(dict1)
        self.assertEqual(st1.col_str_match(r'.*kum.*t').shape, (st1.get_ncols(),))
        self.assertEqual(st1.col_str_match(r'.*kum.*t').sum(), 2)
        self.assertEqual(
            list(st1[st1.col_str_match(r'.*kum.*t')]),
            ['kumquat', 'two_kumquats_or_more'],
        )
        self.assertEqual(
            list(st1[st1.col_str_match(r'.*KuM.*t', flags=re.IGNORECASE)]),
            ['kumquat', 'two_kumquats_or_more'],
        )

    @staticmethod
    def _sum_dict(dd):
        return {_k: sum(_v) for _k, _v in dd.items()}

    def test_as_dictionary(self):
        od = {
            _k: list(range(_i * 10, (_i + 1) * 10))
            for _i, _k in enumerate('abcdefghij')
        }
        st = Struct(od)
        dt1 = st.asdict()
        dt2 = st.asdict(copy=True)
        self.assertEqual(self._sum_dict(dt1), self._sum_dict(od))
        self.assertIsInstance(dt1, dict)
        self.assertEqual(
            list(st.asdict(sublist=['a', 'c', 'd']).keys()), ['a', 'c', 'd']
        )
        with self.assertRaises(AttributeError):
            _ = st.asdict(sublist=['a', 'c', 'q']).keys()
        # test for copy/no-copy?
        lst10 = list(range(10))
        self.assertEqual(st.a, lst10)
        self.assertEqual(dt1['a'], lst10)
        self.assertEqual(dt2['a'], lst10)
        st.a[5] = -1
        self.assertEqual(st.a[5], -1)
        self.assertEqual(dt1['a'][5], -1)
        self.assertEqual(dt2['a'], lst10)
        dt2['b'][3] = -2
        self.assertEqual(st.b[3], 13)
        self.assertEqual(dt1['b'][3], 13)

    def test_as_ordered_dictionary(self):
        od = {
            _k: list(range(_i * 10, (_i + 1) * 10))
            for _i, _k in enumerate('abcdefghij')
        }
        st = Struct(od)
        dt1 = st.as_ordered_dictionary()
        self.assertEqual(self._sum_dict(dt1), self._sum_dict(od))
        self.assertIsInstance(dt1, OrderedDict)
        self.assertEqual(
            list(st.as_ordered_dictionary(sublist=['a', 'c', 'd']).keys()),
            ['a', 'c', 'd'],
        )
        with self.assertRaises(ValueError):
            _ = st.as_ordered_dictionary(sublist=['a', 'c', 'q']).keys()

    def test_no_bool(self):
        with self.assertRaises(ValueError):
            _ = bool(Struct({'A': 5}))
        with self.assertRaises(ValueError):
            _ = bool(Struct({'A': [5.5], 'B': 'fish'}))
        with self.assertRaises(ValueError):
            _ = bool(Struct({'A': True}))
        with self.assertRaises(ValueError):
            _ = bool(Struct({'A': True, 'B': False}))

    def test_tolist(self):
        data = {'a': 5, 'b': 5.6, 'c': 'fish', 'd': 'μεαν', 'e': True, 'f': {'A': None}}
        st = Struct(data)
        self.assertEqual(st.tolist(), list(data.values()))

    def test_any(self):
        self.assertTrue(Struct({'A': True, 'B': True}).any())
        self.assertTrue(Struct({'A': True, 'B': False}).any())
        self.assertFalse(Struct({'A': False, 'B': False}).any())
        self.assertTrue(Struct({'A': True}).any())
        self.assertFalse(Struct({'A': False}).any())
        self.assertTrue(Struct({'A': [5.5], 'B': 'fish'}).any())
        self.assertTrue(Struct({'A': np.nan, 'B': False}).any())
        self.assertTrue(Struct({'A': ['A', 'B', ''], 'B': False}).any())
        self.assertTrue(Struct({'A': np.array(['A', 'B', '']), 'B': False}).any())
        self.assertFalse(Struct({}).any())  # there does not exist one which is true

    def test_all(self):
        self.assertTrue(Struct({'A': True, 'B': True}).all())
        self.assertFalse(Struct({'A': True, 'B': False}).all())
        self.assertFalse(Struct({'A': False, 'B': False}).all())
        self.assertTrue(Struct({'A': True}).all())
        self.assertFalse(Struct({'A': False}).all())
        self.assertTrue(Struct({'A': [5.5], 'B': 'fish'}).all())
        self.assertFalse(Struct({'A': np.nan, 'B': False}).all())
        self.assertFalse(Struct({'A': ['A', 'B', ''], 'B': False}).all())
        self.assertFalse(Struct({'A': np.array(['A', 'B', '']), 'B': False}).all())
        self.assertTrue(Struct({}).all())  # all that exist are true

    def test_comparison_01(self):
        st1 = Struct({'A': [5.5], 'B': 'fish'})
        st2 = Struct({'A': [5.5], 'B': 'fish', 'C': [380, 20]})
        st3 = Struct({'B': 'fish', 'C': [380, 20]})
        st4 = Struct({'A': [5.5], 'B': np.ones(4), 'C': [380, 20]})
        self.assertEqual((st1 == st1).asdict(), {'A': True, 'B': True})
        self.assertEqual((st1 == st2).asdict(), {'A': True, 'B': True, 'C': False})
        self.assertEqual((st2 == st1).asdict(), {'A': True, 'B': True, 'C': False})
        self.assertEqual(list((st2 == st3).keys()), ['A', 'B', 'C'])
        self.assertEqual(list((st3 == st2).keys()), ['B', 'C', 'A'])
        self.assertEqual((st4 == st4).asdict(), {'A': True, 'B': True, 'C': True})
        for comp in 'eq ne lt le gt ge'.split():
            self.assertIsInstance((getattr(st1, f'__{comp}__')(st2)).all(), bool)

    def test_comparison_02(self):
        st1 = Struct({'A': [5.5], 'B': 20})
        st2 = Struct({'A': [5.5], 'B': '20'})
        st3 = Struct({'A': [6.5], 'B': 30})
        self.assertEqual((st1 < st3).asdict(), {'A': True, 'B': True})
        with self.assertRaises(NotImplementedError):
            _ = st2 < st3
        with self.assertRaises(TypeError):
            _ = st2 < []

    def test_concat(self):
        st1 = Struct(
            {
                'ds': TypeRegister.Dataset(
                    {'col_' + str(i): np.random.rand(5) for i in range(5)}
                ),
                'arr': arange(5),
                'cat': TypeRegister.Categorical(['a', 'a', 'b', 'c', 'a']),
            }
        )
        st2 = Struct(
            {
                'ds': TypeRegister.Dataset(
                    {'col_' + str(i): np.random.rand(5) for i in range(5)}
                ),
                'arr': arange(5),
                'cat': TypeRegister.Categorical(['a', 'a', 'b', 'c', 'a']),
            }
        )

        result = Struct.concat_structs([st1, st2])
        self.assertTrue(isinstance(result.ds, TypeRegister.Dataset))
        self.assertTrue(isinstance(result.arr, TypeRegister.FastArray))
        self.assertTrue(isinstance(result.cat, TypeRegister.Categorical))

        correct_arr = np.hstack([st1.arr, st2.arr])
        self.assertTrue(bool(np.all(correct_arr == result.arr)))

        correct_cat = np.array(['a', 'a', 'b', 'c', 'a', 'a', 'a', 'b', 'c', 'a'])
        self.assertTrue(bool(np.all(correct_cat == result.cat)))

        for c in st1.ds:
            correct = np.hstack([st1.ds[c], st2.ds[c]])
            self.assertTrue(bool(np.all(correct == result.ds[c])))

        order = list(st1.keys())
        result = list(result.keys())
        self.assertTrue(bool(np.all(order == result)))

    def test_categorical(self):
        ds = Dataset({'A': ['a', 'b'], 'B': ['c', 'd'], 'C': ['a', 'μεαν']})
        # to_categorical called the categorical constructor
        # removed, because the user can simply say c=Categorical(ds.A)
        # self.assertIsInstance(Dataset.to_categorical(ds.A), Categorical)
        # self.assertNotIsInstance(ds.A, Categorical)
        ds.make_categoricals('A')
        self.assertIsInstance(ds.A, Categorical)
        ds = Dataset({'A': ['a', 'b'], 'B': ['c', 'd'], 'C': ['a', 'μεαν']})
        # 9/28/2018 SJK: dataset no longer flips unicode to categorical
        # self.assertIsInstance(ds.C, Categorical)
        ds.make_categoricals(['A', 'C'])
        self.assertIsInstance(ds.A, Categorical)
        self.assertIsInstance(ds.C, Categorical)
        # BUG: categorical is not retaining its base index for true unicode...
        self.assertEqual(ds.C[1], 'μεαν')
        ds = Dataset(
            {'A': ['a', 'b'], 'B': ['c', 'd'], 'C': ['a', 'μεαν'], 'D': [1, 2]}
        )
        types1 = {'A': FastArray, 'B': FastArray, 'C': FastArray, 'D': FastArray}
        for _k in ds:
            self.assertIsInstance(ds[_k], types1[_k])
        ds.make_categoricals()
        types2 = {'A': Categorical, 'B': Categorical, 'C': Categorical, 'D': FastArray}
        for _k in ds:
            self.assertIsInstance(ds[_k], types2[_k])

        ds1 = Dataset(
            {
                'a': np.random.choice(['a', 'b', 'c'], 5),
                'b': np.random.choice(['a', 'b', 'c'], 5),
            }
        )
        ds2 = Dataset(
            {
                'a': np.random.choice(['a', 'b', 'c'], 5),
                'b': np.random.choice(['a', 'b', 'c'], 5),
            }
        )
        ds3 = Dataset(
            {
                'a': np.random.choice(['a', 'b', 'c'], 5),
                'b': np.random.choice(['a', 'b', 'c'], 5),
            }
        )
        arr = np.random.choice(['a', 'b', 'c'], 5)
        st = Struct({'ds1': ds1, 'ds2': ds2, 'ds3': ds3, 'arr': arr})
        st.make_categoricals()
        for ds in [st.ds1, st.ds2, st.ds3]:
            for col in ds:
                self.assertTrue(isinstance(ds[col], Categorical))
        self.assertTrue(isinstance(st.arr, Categorical))

    def _test_output(self):
        if sys.platform != 'win32':
            intt_name = b'int64'
        else:
            intt_name = b'int32'

        class Dataset1(Struct):
            pass  # dummy for testing, mimics behavior of real Dataset

        st = Struct(
            {
                'a': Dataset1({'A': range(10), 'B': range(10, 20)}),
                'b': Struct({'C': 0, 'D': 1, 'E': 2}),
                'c': FastArray(np.arange(5)),
                'd': np.arange(5, 10),
                'e': ['abc', 'def', 'ghi'],
                'f': {'q': 1, 'r': 2},
                'g': 3.14,
                'h': 84,
                'i': None,
                'j': slice(None),
            }
        )
        headers, spec = st.get_table_data()
        self.assertEqual(len(headers), 1)
        self.assertEqual(
            [hd.col_name for hd in headers[0]], ['Name', 'Type', 'Rows', '0', '1', '2']
        )
        self.assertEqual(
            [_r.tolist() for _r in spec],
            [
                [b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j'],
                [
                    b'Dataset1',
                    b'Struct',
                    intt_name,
                    intt_name,
                    b'list',
                    b'dict',
                    b'float',
                    b'int',
                    b'NoneType',
                    b'slice',
                ],
                [b'2', b'3', b'5', b'5', b'3', b'2', b'0', b'0', b'0', b'0'],
                [b'A', b'', b'0', b'5', b'', b'', b'3.14', b'84', b'', b''],
                [b'B', b'', b'1', b'6', b'', b'', b'', b'', b'', b''],
                [b'', b'', b'2', b'7', b'', b'', b'', b'', b'', b''],
            ],
        )
        self.assertEqual(
            str(st),
            f'''#   Name   Type       Rows   0      1   2
-   ----   --------   ----   ----   -   -
0   a      Dataset1   2      A      B
1   b      Struct     3
2   c      {intt_name.decode()}      5      0      1   2
3   d      {intt_name.decode()}      5      5      6   7
4   e      list       3
5   f      dict       2
6   g      float      0      3.14
7   h      int        0      84
8   i      NoneType   0
9   j      slice      0                  ''',
        )
        self.assertEqual(Struct._sizeof_fmt(128), '128.0 B')
        tsize = 1280
        for unit in ['K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']:
            self.assertEqual(Struct._sizeof_fmt(tsize), f'1.2 {unit}B')
            tsize *= 1024
        self.assertEqual(st._last_row_stats(), '[10 columns]')

    def test_total_sizes(self):
        st = Struct(
            {
                'a': Dataset(
                    {
                        # 10x int32 => 40B
                        'A': range(10),
                        # 10x int32 => 40B
                        'B': range(10, 20),
                    }
                ),
                'b': Struct(
                    {
                        # 1x int32 => 4B
                        'C': 0,
                        # 1x int32 => 4B
                        'D': 1,
                        # 1x int32 => 4B
                        'E': 2,
                    }
                ),
                # 5x int32 => 20B
                'c': FastArray(np.arange(5)),
                # 5x int32 => 20B
                'd': np.arange(5, 10),
                # ???
                'e': ['abc', 'def', 'ghi'],
                'f': {
                    # 1x int32 => 4B
                    'q': 1,
                    # 1x int32 => 4B
                    'r': 2,
                },
                # 1x float64 => 8B
                'g': 3.14,
                # 1x int32 => 4B
                'h': 84,
                # ???
                'i': None,
                # ???
                'j': slice(None),
            }
        )

        # Create some duplicated/aliased data within the struct.
        st.z = st.c

        # Calculate the sizes of the Struct's data in bytes.
        (physical, logical) = st.total_sizes

        # For now, we only check that the logical size is larger than the physical size
        # (due to the presence of aliased array(s) somewhere within the Struct).
        # TODO: Strengthen this test by checking the actual computed sizes to make sure they're correct.
        self.assertLess(
            physical, logical, "The physical size is not less than the logical size."
        )

    def test_total_sizes_with_categorical(self):
        st = Struct({'c': Categorical(['aa', 'bb', 'cc', 'dd'])})
        st.d = st.c
        (physical, logical) = st.total_sizes
        self.assertEqual(physical, logical // 2)
        self.assertGreaterEqual(
            physical, np.asarray(st.c).nbytes + st.c.category_array.nbytes
        )


if __name__ == "__main__":
    tester = unittest.main()
