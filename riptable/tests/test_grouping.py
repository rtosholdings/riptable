import os
import unittest
import pytest
from numpy.testing import assert_array_equal

import riptable as rt
from riptable.rt_enum import GROUPBY_KEY_PREFIX
from riptable.rt_sds import SDSMakeDirsOn
from riptable import *

# change to true since we write into /tests directory
SDSMakeDirsOn()

# TODO: Replace these two functions with calls to assert_array_equal to get better diagnostics if the assertion fails.
def arr_all(a):
    return bool(np.all(a))

def arr_eq(a, b):
    return arr_all(a == b)

#
# TODO: Modify the tests in this file to -- at minimum -- generate the random data they use with a deterministic
#       seed, and also regenerate it within each test (to avoid potential issues sharing the data across tests).
#       It would be even better if we could generate the arrays used in the tests with hypothesis, since that'll
#       make it easier to describe the random arrays we want and it'd let us easily reproduce a random test case
#       if we find one that leads to a failure.
#

uniquelen = 7
arrlen = 50
modulo = 3
keyname = 'keycol'
d = {'data' + str(i): np.random.rand(arrlen) for i in range(7)}
f = logical(arange(arrlen) % modulo)

randi = np.random.randint(10000, 20000, uniquelen)
randi = np.random.choice(randi, arrlen)
rands = randi.astype('S')
randu = randi.astype('U')
randf = randi.astype(np.float32)


class Grouping_Test(unittest.TestCase):
    def test_head(self):
        pytest.skip("This test is not yet implemented.")

    def test_tail(self):
        pytest.skip("This test is not yet implemented.")

    def test_single_key(self):

        for vals in [randi, rands, randu, randf]:
            vals = vals.view(FA)
            vals.set_name(keyname)
            ds = Dataset({keyname: vals}, unicode=True)
            gb = ds.gb(keyname)
            gb_result = gb.sum(d)

            c = Categorical(vals)
            c_result = c.sum(d)

            ds_c = Dataset({keyname: c}, unicode=True)
            gbc = ds_c.gb(keyname)
            gbc_result = gbc.sum(d)

            self.assertTrue(
                gb_result.equals(c_result),
                msg=f'Groupby and categorical results did not match for single key dtype {vals.dtype}.',
            )
            assert_array_equal(gb_result.label_get_names(), c_result.label_get_names())

            self.assertTrue(
                c_result.equals(gbc_result),
                msg=f'Categorical and groupby categorical results did not match for single key dtype {vals.dtype}.',
            )
            assert_array_equal(c_result.label_get_names(), gbc_result.label_get_names())

    def test_single_key_filter(self):
        for vals in [randi, rands, randu, randf]:
            vals = vals.view(FA)
            vals.set_name(keyname)
            ds = Dataset({keyname: vals}, unicode=True)
            gb = ds.gb(keyname, filter=f)
            gb_result = gb.sum(d, showfilter=True)

            c = Categorical(vals, filter=f)
            c_result = c.sum(d, showfilter=True)

            # reset filter
            c = Categorical(vals)
            ds_c = Dataset({keyname: c}, unicode=True)
            gbc = ds_c.gb(keyname, filter=f)
            gbc_result = gbc.sum(d, showfilter=True)

            self.assertTrue(
                gb_result.equals(c_result),
                msg=f'Groupby and categorical results did not match for single key dtype {vals.dtype}.',
            )
            assert_array_equal(gb_result.label_get_names(), c_result.label_get_names())

            # TJD change
            for colname in c_result:
                if colname != keyname:
                    self.assertTrue(
                        arr_eq(c_result[colname], gbc_result[colname]),
                        msg=f'Categorical and groupby categorical results did not match for single key dtype {vals.dtype}.',
                    )
            self.assertTrue(
                c_result.equals(gbc_result),
                msg=f'Categorical and groupby categorical results did not match for single key dtype {vals.dtype}.',
            )
            assert_array_equal(c_result.label_get_names(), gbc_result.label_get_names())

    def test_multikey(self):
        keynames = ['keycol_0', 'keycol_1']
        for vals1 in [randi, rands, randu, randf]:
            vals1 = vals1.view(FA)
            vals1.set_name(keynames[0])
            for vals2 in [randi, rands, randu, randf]:
                if vals1.dtype == vals2.dtype:
                    continue
                vals2 = vals2.view(FA)
                vals2.set_name(keynames[1])

                vals = [vals1, vals2]

                ds = Dataset(dict(zip(keynames, vals)), unicode=True)
                gb = ds.gb(keynames)
                gb_result = gb.sum(d)

                c = Categorical(vals, sort_gb=True, unicode=True)
                c_result = c.sum(d)

                ds_c = Dataset({'keycol': c}, unicode=True)
                gbc = ds_c.gb('keycol')
                gbc_result = gbc.sum(d)

                assert_array_equal(gb_result.label_get_names(), c_result.label_get_names())
                self.assertTrue(
                    gb_result.equals(c_result),
                    msg=f'Groupby and categorical results did not match for multi key dtype {vals[0].dtype} {vals[1].dtype}.',
                )
                # assert_array_equal(c_result.label_get_names(), gbc_result.label_get_names())
                self.assertTrue(
                    c_result.equals(gbc_result),
                    msg=f'Categorical and groupby categorical results did not match for multi key dtype {c_result} \n {gbc_result}. {c_result} \n {gbc_result}',
                )

    def test_init_paths(self):
        #  empty
        with self.assertRaises(ValueError):
            g = Grouping({})

        # array mismatch
        with self.assertRaises(ValueError):
            g = Grouping({'a': arange(3), 'b': arange(4)})

        # non-fastarray
        g = Grouping([np.arange(3), np.arange(3)])
        gdict = g._grouping_dict
        for v in gdict.values():
            self.assertTrue(isinstance(v, FastArray))

        # named arrays
        ds = Dataset({'col1': arange(3), 'col2': arange(3)})
        g = Grouping([ds.col1, ds.col2])
        gdict = g._grouping_dict
        for k in gdict:
            self.assertTrue(k in ds)

        # no name arrays
        g = Grouping([arange(3), arange(3)])
        self.assertTrue(
            arr_eq(
                list(g._grouping_dict),
                [GROUPBY_KEY_PREFIX + '_0', GROUPBY_KEY_PREFIX + '_1'],
            )
        )

        # conflicting name arrays
        g = Grouping([ds.col1, ds.col1])
        self.assertTrue(
            arr_eq(list(g._grouping_dict), ['col1', GROUPBY_KEY_PREFIX + '_c1'])
        )

    def test_auto_naming(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        self.assertTrue(c.get_name() is None)
        ds = Dataset({'catcol': c})
        result = ds.catcol.sum(arange(5))
        self.assertTrue(arr_eq(['catcol'], result.label_get_names()))

        c = Categorical([FA(['a', 'a', 'b', 'c', 'a']), FA([1, 1, 2, 3, 1])])
        self.assertTrue(c.get_name() is None)
        ds = Dataset({'mkcol': c})
        result = ds.mkcol.sum(arange(5))
        self.assertTrue(arr_eq(['mkcol_0', 'mkcol_1'], result.label_get_names()))

        arr1 = FA(['a', 'a', 'b', 'c', 'a'])
        arr1.set_name('mystrings')
        arr2 = FA([1, 1, 2, 3, 1])
        arr2.set_name('myints')
        c = Categorical([arr1, arr2])
        ds = Dataset({'mkcol': c})
        result = ds.mkcol.sum(arange(5))
        self.assertTrue(arr_eq(['mystrings', 'myints'], result.label_get_names()))

    def test_make_ifirst(self):
        int_types = [np.int8, np.int16, np.int32, np.int64]
        for dt in int_types:
            arr = ones(174_763, dtype=dt)
            ifirst = makeifirst(arr, 1)
            assert ifirst[1] == 0
            assert isnan(ifirst)[0]

            ifirst = makeifirst(arr, 2)
            assert ifirst[1] == 0
            assert isnan(ifirst)[0]
            assert isnan(ifirst)[2]

        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        f = FastArray([True, True, False, True, True])
        cfilter = c.filter(f)
        ifirst = makeifirst(cfilter, 2)
        assert_array_equal(ifirst[1:], FA([0, 3]))
        assert isnan(ifirst)[0]

        ifirst = makeifirst(c, c.unique_count, filter=f)
        assert isnan(ifirst)[0]
        assert isnan(ifirst)[2]
        assert ifirst[1] == 0
        assert ifirst[-1] == 3

        c = Categorical([1, 1, 1, 1, 2, 1], ['a', 'b', 'c'])
        ifirst = makeifirst(c, c.unique_count)
        assert_array_equal(ifirst[1:3], FA([0, 4]))
        assert isnan(ifirst)[0]
        assert isnan(ifirst)[-1]

        strs = np.random.choice(arange(100, 200).astype('S'), 1287)

        ds = Dataset({'stringvals': strs})
        dsg = ds.gb('stringvals').grouping

        c = Categorical(ds.stringvals, ordered=False)
        assert_array_equal(makeifirst(c, c.unique_count)[1:], dsg.iFirstKey)
        assert_array_equal(c.grouping.ifirstkey, dsg.ifirstkey)

        c = Categorical(ds.stringvals, ordered=False, base_index=0)
        assert_array_equal(c.grouping.ifirstkey, dsg.ifirstkey)

    def test_makeilast(self):
        int_types = [np.int8, np.int16, np.int32, np.int64]
        for dt in int_types:
            arr = ones(174_763, dtype=dt)
            ilast = makeilast(arr, 1)
            assert ilast[1] == (len(arr) - 1)
            assert isnan(ilast)[0]

            ilast = makeilast(arr, 2)
            assert ilast[1] == (len(arr) - 1)
            assert isnan(ilast)[0]
            assert isnan(ilast)[2]

        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        f = FastArray([True, True, False, True, True])
        cfilter = c.filter(f)
        ilast = makeilast(cfilter, 2)
        assert_array_equal(ilast[1:], FA([4, 3]))
        assert isnan(ilast)[0]

        ilast = makeilast(c, c.unique_count, filter=f)
        assert isnan(ilast)[0]
        assert isnan(ilast)[2]
        assert ilast[1] == 4
        assert ilast[-1] == 3

        c = Categorical([1, 1, 1, 1, 2, 1], ['a', 'b', 'c'])
        ilast = makeilast(c, c.unique_count)
        assert_array_equal(ilast[1:3], FA([5, 4]))
        assert isnan(ilast)[0]
        assert isnan(ilast)[-1]

        strs = np.random.choice(arange(100, 200).astype('S'), 1287)

        ds = Dataset({'stringvals': strs})
        dsg = ds.gb('stringvals').grouping

        c = Categorical(ds.stringvals, ordered=False)
        assert_array_equal(makeilast(c, c.unique_count)[1:], dsg.ilastkey)
        assert_array_equal(c.grouping.ilastkey, dsg.ilastkey)

        c = Categorical(ds.stringvals, ordered=False, base_index=0)
        assert_array_equal(c.grouping.ilastkey, dsg.ilastkey)

    def test_isin(self):
        c = Categorical(
            [
                np.random.choice(['aa', 'b', 'bbb', 'c'], 40),
                np.random.choice([1, 2, 3], 40).astype(dtype=np.int8),
            ]
        )
        uniquelist = c.grouping.uniquelist
        tups = [
            t for t in zip(uniquelist[0].astype('U'), uniquelist[1].astype(np.float32))
        ]
        self.assertTrue(arr_all(c.grouping.isin(tups)))

        c = Categorical(
            [
                np.random.choice(['aa', 'b', 'bbb', 'c'], 40),
                np.random.choice([1, 2, 3], 40).astype(dtype=np.float32),
            ]
        )
        uniquelist = c.grouping.uniquelist
        tups = [
            t for t in zip(uniquelist[0].astype('U'), uniquelist[1].astype(np.int8))
        ]
        self.assertTrue(arr_all(c.grouping.isin(tups)))

    def test_isin_enum(self):
        c = Categorical(
            [10, 10, 10, 20, 30, 20, 10, 20, 20, 40], {'a': 30, 'b': 20, 'c': 10}
        )
        ten_mask = FastArray(
            [True, True, True, False, False, False, True, False, False, False]
        )
        self.assertTrue(arr_eq(c.isin(10), ten_mask))
        self.assertTrue(arr_eq(c.isin([10]), ten_mask))
        self.assertTrue(arr_eq(c.isin(FastArray(10)), ten_mask))

        self.assertTrue(arr_eq(c.isin('c'), ten_mask))
        self.assertTrue(arr_eq(c.isin(b'c'), ten_mask))
        self.assertTrue(arr_eq(c.isin(['c']), ten_mask))
        self.assertTrue(arr_eq(c.isin(np.array(['c'])), ten_mask))
        self.assertTrue(arr_eq(c.isin(FA('c')), ten_mask))

        with self.assertRaises(TypeError):
            _ = c.isin(1.23)

    def test_hstack_base_0(self):
        c7 = Categorical([1, 1], ['a', 'b'], base_index=0)
        c8 = Categorical([1, 0], ['b', 'c'], base_index=0)
        c9 = hstack([c7, c8])

        self.assertTrue(isinstance(c9, Categorical))
        self.assertEqual(c9.base_index, 0)
        self.assertTrue(arr_eq(c9._fa, FA([1, 1, 2, 1])))
        self.assertTrue(arr_eq(c9.expand_array, FA(['b', 'b', 'c', 'b'])))

    # TODO use pytest fixtures in place of 'paths' and move this and similar test cases into SDS save / load tests
    def test_stack_load_base_0(self):
        c1 = Categorical(
            np.random.choice(5, 30), ['a', 'b', 'c', 'd', 'e'], base_index=0
        )
        c2 = Categorical(
            np.random.choice(5, 30), ['e', 'f', 'g', 'a', 'b'], base_index=0
        )

        ds1 = Dataset({'catcol': c1})
        ds2 = Dataset({'catcol': c2})

        paths = [r'riptable/tests/temp/ds1.sds', r'riptable/tests/temp/ds2.sds']
        ds1.save(paths[0])
        ds2.save(paths[1])

        c_hstack = hstack([c1, c2])
        self.assertTrue(isinstance(c_hstack, Categorical))
        self.assertEqual(c_hstack.base_index, 0)

        pds = load_sds(paths, stack=True)
        self.assertTrue(isinstance(pds, PDataset))
        c = pds.catcol
        self.assertTrue(isinstance(c, Categorical))
        self.assertEqual(c.base_index, 0)
        self.assertTrue(arr_eq(c_hstack, c))

        for p in paths:
            os.remove(p)

    def test_hstack_ordered(self):
        c1 = Categorical([1, 2, 3], ['a', 'b', 'c'])
        self.assertTrue(c1.ordered)
        c2 = Categorical([1, 2, 3], ['b', 'a', 'z'])
        self.assertFalse(c2.ordered)

        st = hstack([c1, c2])
        self.assertTrue(isinstance(st, Categorical))
        self.assertTrue(st.ordered)

        st = hstack([c2, c1])
        self.assertTrue(isinstance(st, Categorical))
        self.assertFalse(st.ordered)

        st = hstack([c1, c2, c1])
        self.assertTrue(isinstance(st, Categorical))
        self.assertTrue(st.ordered)

        st = hstack([c2, c1, c2])
        self.assertTrue(isinstance(st, Categorical))
        self.assertFalse(st.ordered)

    def test_stack_load_ordered(self):
        c1 = Categorical([1, 2, 3], ['a', 'b', 'c'])
        self.assertTrue(c1.ordered)
        c2 = Categorical([1, 2, 3], ['b', 'a', 'z'])
        self.assertFalse(c2.ordered)

        ds1 = Dataset({'catcol': c1})
        ds2 = Dataset({'catcol': c2})
        paths = [r'riptable/tests/temp/ds1.sds', r'riptable/tests/temp/ds2.sds']
        ds1.save(paths[0])
        ds2.save(paths[1])

        ds1 = load_sds(paths[0])
        self.assertTrue(isinstance(ds1.catcol, Categorical))
        self.assertTrue(ds1.catcol.ordered)

        ds2 = load_sds(paths[1])
        self.assertTrue(isinstance(ds2.catcol, Categorical))
        self.assertFalse(ds2.catcol.ordered)

        ds3 = load_sds([paths[1], paths[0]], stack=True)
        self.assertTrue(isinstance(ds3.catcol, Categorical))
        self.assertFalse(ds3.catcol.ordered)

        ds4 = load_sds([paths[0], paths[1]], stack=True)
        self.assertTrue(isinstance(ds4.catcol, Categorical))
        self.assertTrue(ds4.catcol.ordered)

        for p in paths:
            os.remove(p)

    def test_stack_load_sort_display(self):
        c = Categorical([1, 1, 2, 3, 1], ['b', 'c', 'a'])
        self.assertFalse(c.ordered)
        self.assertFalse(c.grouping.isdisplaysorted)

        d = Categorical([1, 1, 2, 3, 1], ['b', 'c', 'a'], sort_gb=True)
        self.assertFalse(d.ordered)
        self.assertTrue(d.grouping.isdisplaysorted)

        ds1 = Dataset({'catcol': c})
        ds2 = Dataset({'catcol': d})
        paths = [r'riptable/tests/temp/ds1.sds', r'riptable/tests/temp/ds2.sds']

        ds1.save(paths[0])
        ds2.save(paths[1])

        ds3 = load_sds(paths, stack=True)
        gb_result = ds3.catcol.count()
        self.assertFalse(issorted(gb_result.catcol))

        ds4 = load_sds([paths[1], paths[0]], stack=True)
        gb_result = ds4.catcol.count()
        self.assertTrue(issorted(gb_result.catcol))

        for p in paths:
            os.remove(p)

    def testapply_reduce(self):
        c = Cat(ones(10))
        # make sure kwargs handled also
        ds = c.apply_reduce(np.std, arange(10), ddof=4)
        # first col, first val
        finalcalc = ds[0][0]
        self.assertTrue(finalcalc > 3.6 and finalcalc < 3.8)

    def testapply_nonreduce(self):
        # even / odd groups
        c = Cat(arange(10) % 2)

        ds = c.apply_nonreduce(np.cumsum, arange(10))

        finalcalc = ds[0]
        goodresult = FA([0, 1, 2, 4, 6, 9, 12, 16, 20, 25])

        self.assertTrue(np.all(finalcalc == goodresult))

        # try with a numba function
        a = arange(10.0)
        a[[3, 4, 7]] = nan

        finalcalc = c.apply_nonreduce(fill_forward, a)[0]
        goodresult = FA([0.0, 1.0, 2.0, 1.0, 2.0, 5.0, 6.0, 5.0, 8.0, 9.0])
        self.assertTrue(np.all(finalcalc == goodresult))

    def test_ema(self):
        arrsize = 100
        numrows = 20
        ds = Dataset({'time': arange(arrsize * 1.0)})
        ds.data = np.random.randint(numrows, size=arrsize)
        ds.data2 = np.random.randint(numrows, size=arrsize)
        symbols = ['ZYGO', 'YHOO', 'FB', 'GOOG', 'IBM']
        ds.symbol2 = Cat(
            1 + ds.data,
            [
                'A',
                'B',
                'C',
                'D',
                'E',
                'F',
                'G',
                'H',
                'I',
                'J',
                'K',
                'L',
                'M',
                'N',
                'O',
                'P',
                'Q',
                'R',
                'S',
                'T',
            ],
        )
        ds.symbol = Cat(1 + arange(arrsize) % len(symbols), symbols)
        result = ds.data.ema_decay(ds.time, 2.2)

        result = ds.data.ema_decay(ds.time, 2.2, filter=ds.symbol == 'YHOO')

        gb = ds.gb('symbol')
        prev = gb.grouping.iprevkey
        next = gb.grouping.inextkey

    def test_bincount(self):
        c = np.repeat(arange(1000), 3000)
        c1 = Cat(c)
        nCountGroup = rc.BinCount(c1._fa, c1.unique_count + 1)
        y = groupbypack(c1._fa, None, c1.unique_count + 1)
        # all of them should be 30000 except invalid bin
        x = (nCountGroup == 3000).sum()
        self.assertTrue(x == 1000)
        self.assertTrue(np.all(nCountGroup == y['nCountGroup']))
        z = bincount(c)
        self.assertTrue(np.all(nCountGroup[1:] == z))

        c = np.random.randint(0, 1000, 1_000_000)
        x = groupbyhash(c)
        ncountgroup = rc.BinCount(x['iKey'], x['unique_count'] + 1)
        y = groupbypack(x['iKey'], ncountgroup)
        self.assertTrue(np.all(ncountgroup == y['nCountGroup']))

        c1 = Cat(c)
        pack = groupbypack(x['iKey'], None, c1.unique_count + 1)
        self.assertTrue(np.all(ncountgroup == pack['nCountGroup']))
        pack2 = groupbypack(x['iKey'], ncountgroup, None)
        self.assertTrue(np.all(pack2['iFirstGroup'] == pack['iFirstGroup']))

        ## every 1 in 1000 is unique
        # c1=Cat(c)
        # nCat = c1.unique_count + 1
        # dCat = c1._fa
        # nCountGroup = rc.BinCount(c1._fa, c1.unique_count +1 )
        # x= rc.GroupFromBinCount(c1._fa, nCountGroup)

        ## highly unique count
        # c=np.random.randint(0, 300_000, 1_000_000)
        # c1=Cat(c)
        # nCat = c1.unique_count + 1
        # dCat = c1._fa

        ## Force int64
        # c=np.repeat(arange(1000),3000)
        # c1=Cat(c)
        # nCat = c1.unique_count + 1
        # dCat = c1._fa
        # nCountGroup = rc.BinCount(c1._fa, c1.unique_count +1 )

    def test_take_groups(self):
        ## Case 1: Basic operation.
        # Create a grouping from some data.
        key_data1 = rt.FA([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6])
        g1 = rt.Grouping(key_data1)

        # Create another data array the same length as the key for the Grouping.
        data1 = rt.arange(len(key_data1))

        # Extract elements from the data array where they correspond to even-numbered groups.
        result1 = Grouping.take_groups(data1, rt.FA([2, 4, 6]), g1.ncountgroup, g1.ifirstgroup)

        assert_array_equal(rt.FA([1, 2, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20]), result1)

    def test_extract_groups(self):
        ## Case 1: Basic operation.
        # Create a grouping from some data.
        key_data1 = rt.FA([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6])
        g1 = rt.Grouping(key_data1)

        # Create another data array the same length as the key for the Grouping.
        data1 = rt.arange(len(key_data1))

        # Create a mask which selects the even-numbered groups.
        group_mask1 = rt.arange(len(g1.ncountgroup)) % 2 == 0

        # Extract elements from the data array where they correspond to even-numbered groups.
        result1 = Grouping.extract_groups(group_mask1, data1, g1.ncountgroup, g1.ifirstgroup)

        assert_array_equal(rt.FA([1, 2, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20]), result1)


def test_merge_cats_stringcat_with_empty():
    cat_lens = [
        12,
        7,
    ]  # Arbitrary; specify the lengths of our test Cats here so we can re-use the lengths below for consistency.

    indices = np.hstack([np.full(cat_lens[0], 1), np.full(cat_lens[1], 2)])
    listcats = [rt.FA([b'2019/12/21', b'', b'2019/12/21'], dtype='|S32')]
    idx_cutoffs = np.cumsum(cat_lens)
    uniques_cutoffs = [np.array([1, 3], dtype=np.int64)]
    assert len(listcats) == len(uniques_cutoffs)

    fixed_indices, stacked_uniques = rt.merge_cats(
        indices, listcats, idx_cutoffs=idx_cutoffs, unique_cutoffs=uniques_cutoffs
    )

    assert len(stacked_uniques[0]) == 2


if __name__ == '__main__':
    tester = unittest.main()
