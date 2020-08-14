import unittest
from riptable import *
from riptable.rt_enum import GROUPBY_KEY_PREFIX

funcs = [np.sqrt, lambda x: x + 1]
reduce_funcs = [sum, np.min, np.max]


def arr_eq(a, b):
    return bool(np.all(a == b))


def arr_all(a):
    return bool(np.all(a))


class transform_apply_test(unittest.TestCase):
    def test_apply(self):
        ds = Dataset({'col_' + str(i): np.random.rand(30) for i in range(5)})
        ds.gb_col = np.random.choice(['a', 'b', 'c', 'd', 'e'], 30)
        ds_sorted = ds.sort_copy('gb_col')
        gb = ds_sorted.gb('gb_col')

        for f in reduce_funcs:
            gb_result = gb.apply(f)
            self.assertTrue(isinstance(gb_result, Dataset))
            for gb_rownum, gb_unique in enumerate(['a', 'b', 'c', 'd', 'e']):
                mask = ds_sorted.gb_col == gb_unique
                for i in range(5):
                    col_name = 'col_' + str(i)
                    col_slice = ds_sorted[col_name][mask]
                    ds_item = f(col_slice)
                    gb_item = gb_result[col_name][gb_rownum]
                    self.assertAlmostEqual(
                        ds_item, gb_item, msg=f"func:{f} {ds_item} {gb_item}"
                    )

    def test_apply_arrays(self):
        ds = Dataset({'col_' + str(i): np.random.rand(30) for i in range(5)})
        ds.gb_col = np.random.choice(['a', 'b', 'c', 'd', 'e'], 30)
        ds_sorted = ds.sort_copy('gb_col')
        gb = ds_sorted.gb('gb_col')

        for f in funcs:
            gb_result = gb.apply(f)
            # self.assertTrue(isinstance(gb_result, list))
            for i in range(5):
                colname = 'col_' + str(i)
                acol = f(ds_sorted[colname])
                bcol = gb_result[i]
                self.assertTrue(isinstance(bcol, np.ndarray))
                for idx, item in enumerate(acol):
                    self.assertAlmostEqual(
                        item, bcol[idx], msg=f"{item} {bcol[idx]} {f.__name__}"
                    )

    def test_docstring_examples(self):

        ds = Dataset({'A': 'a a b'.split(), 'B': [1, 2, 3], 'C': [4, 6, 5]})
        g = GroupBy(ds, 'A')

        result = g.apply(lambda x: x.sum())
        self.assertTrue(isinstance(result, Dataset))
        self.assertEqual(result._ncols, 3)
        self.assertTrue(arr_eq(result.A, FA(['a', 'b'])))
        self.assertTrue(arr_eq(result.B, FA([3, 3])))
        self.assertTrue(arr_eq(result.C, FA([10, 5])))

        result = g.apply(lambda x: {'B': x.B.sum()})
        self.assertTrue(isinstance(result, Dataset))
        self.assertEqual(result._ncols, 2)
        self.assertTrue(arr_eq(result.A, FA(['a', 'b'])))
        self.assertTrue(arr_eq(result.B, FA([3, 3])))

        result = g.apply(lambda x: x.max() - x.min())
        self.assertTrue(isinstance(result, Dataset))
        self.assertEqual(result._ncols, 3)
        self.assertTrue(arr_eq(result.A, FA(['a', 'b'])))
        self.assertTrue(arr_eq(result.B, FA([1, 0])))
        self.assertTrue(arr_eq(result.C, FA([2, 0])))

        result = g.apply(lambda x: Dataset({'val': [x.C.max() - x.B.min()]}))
        self.assertTrue(isinstance(result, Dataset))
        self.assertEqual(result._ncols, 2)
        self.assertTrue(arr_eq(result.A, FA(['a', 'b'])))
        self.assertTrue(arr_eq(result.val, FA([5, 2])))

        def userfunc(x):
            x.Sub = x.C - x.B
            return x

        result = g.apply(userfunc)
        self.assertTrue(isinstance(result, Dataset))
        self.assertEqual(result._ncols, 4)
        self.assertTrue(arr_eq(ds.A, result.A))
        self.assertTrue(arr_eq(ds.B, result.B))
        self.assertTrue(arr_eq(ds.C, result.C))
        self.assertTrue(arr_eq(result.Sub, FA([3, 4, 2])))

    def test_apply_catcol_filter(self):

        ds = Dataset(
            {'keycol': FA(['a', 'a', 'b', 'b', 'a']), 'data': np.random.rand(5)}
        )
        ds.keycol = Categorical(ds.keycol)
        f = FA([False, False, True, True, True])

        result_prefilter = ds.gb('keycol', filter=f).apply(sum)
        result_postfilter = ds.gb('keycol').apply(sum, filter=f)
        self.assertTrue(result_prefilter.equals(result_postfilter))

    def test_apply_userfunc(self):
        symbols = np.random.choice(['AAPL', 'AMZN', 'NFLX'], 20)
        spot = zeros(20)
        trade_size = np.random.choice([10.0, 25.0, 5.0], 20)
        ds = Dataset({'symbol': symbols, 'spot': spot, 'trade_size': trade_size})

        aapl_mask = ds.symbol == 'AAPL'
        tvals = (np.random.rand(sum(aapl_mask)) * 5) + 180
        ds.spot[aapl_mask] = tvals

        amzn_mask = ds.symbol == 'AMZN'
        tvals = (np.random.rand(sum(amzn_mask)) * 15) + 1830
        ds.spot[amzn_mask] = tvals

        nflx_mask = ds.symbol == 'NFLX'
        tvals = (np.random.rand(sum(nflx_mask)) * 15) + 1830
        ds.spot[nflx_mask] = tvals

    def test_apply_userfunc_kwargs(self):
        def addfunc(x, addent=0):
            x.data = x.data + 100
            return x

        arr = np.random.choice(['a', 'b', 'c'], 20)
        c = Categorical(arr)
        data = arange(20)

        ds_norm = Dataset({'keycol': arr, 'data': data})
        ds_result = ds_norm.gb('keycol').apply(addfunc, addent=100)

        c = Categorical(arr, ordered=False)
        c_result = c.apply(addfunc, ds_norm.data, addent=100)

        ds_cat = Dataset({'keycol': Categorical(arr, ordered=False), 'data': data})
        dsc_result = ds_cat.gb('keycol').apply(addfunc, addent=100)

        self.assertTrue(arr_eq(ds_result.data, c_result.data))
        self.assertTrue(arr_eq(c_result.data, dsc_result.data))
        self.assertTrue(arr_eq(np.sort(c_result.data) - 100, data))

    def test_apply_userfunc_multikey(self):
        st_arr = np.random.choice(['a', 'b', 'c'], 25)
        int_arr = np.random.choice([10, 20, 30], 25)
        data = arange(25)

        c = Categorical([st_arr, int_arr])

        ds_norm = Dataset(c.expand_dict)
        ds_norm.data = data

        ds_cat = Dataset({'keycol': Categorical([st_arr, int_arr]), 'data': data})

        c_result = c.apply(lambda x: x + 100, ds_norm.data)
        ds_result = ds_norm.gb(['key_0', 'key_1']).apply(lambda x: x + 100)
        dsc_result = ds_cat.gb('keycol').apply(lambda x: x + 100)

        self.assertTrue(arr_eq(c_result.data, ds_result.data))
        self.assertTrue(arr_eq(ds_result.data, dsc_result.data))

    def test_apply_filter(self):
        st_arr = np.random.choice(['a', 'b', 'c'], 25)
        data = arange(25)
        f = logical(data % 3)
        ds = Dataset({'strs': st_arr, 'data': data})
        ds_result = ds.gb('strs', lex=True).apply(sum, filter=f)

        c = Categorical(ds.strs)
        c_result = c.apply(sum, ds.data, filter=f)

        self.assertTrue(ds_result.equals(c_result))

    def test_apply_filter_multikey(self):
        st_arr = np.random.choice(['a', 'b', 'c'], 25).astype('S')
        int_arr = np.random.choice([10, 20, 30], 25)
        data = arange(25)
        f = logical(arange(25) % 3)

        c = Categorical([st_arr, int_arr])

        ds_norm = Dataset(c.expand_dict)
        ds_norm.data = data

        ds_cat = Dataset(
            {GROUPBY_KEY_PREFIX: Categorical([st_arr, int_arr]), 'data': data}
        )

        c_result = c.apply(np.sum, data, filter=f)
        ds_result = ds_norm.gbu(['key_0', 'key_1']).apply(np.sum, data, filter=f)
        dsc_result = ds_cat.gbu(GROUPBY_KEY_PREFIX).apply(np.sum, data, filter=f)

        self.assertTrue(c_result.equals(ds_result))
        self.assertTrue(ds_result.equals(dsc_result))

    def test_compare_agg(self):
        st_arr = np.random.choice(['a', 'b', 'c'], 25)
        data = arange(25)
        data.set_name('data')

        c = Categorical(st_arr)

        agg_result = c.agg({'data': [mean, sum]}, data)

        aggfunc = lambda x: {'Mean': x.data.mean(), 'Sum': x.data.sum()}
        app_result = c.apply(aggfunc, data)

        self.assertTrue(arr_eq(agg_result.Mean.data, app_result.Mean))
        self.assertTrue(arr_eq(agg_result.Sum.data, app_result.Sum))

    def test_apply_errors(self):

        st_arr = np.random.choice(['a', 'b', 'c'], 25)
        c = Categorical(st_arr)

        # no data
        with self.assertRaises(ValueError):
            _ = c.apply(sum)

        # not callable
        with self.assertRaises(TypeError):
            _ = c.apply('a', arange(25))

    def test_appply_reduce_sameid(self):
        data = Dataset({'id': np.array([0, 0, 1, 1]), 'val': np.array([0, 10, 20, 30])})

        def func_nr(x, y):
            return x.cumsum() + y

        def func_r(x, y):
            return x.sum() + y[0]

        result = Cat(data.id).apply_nonreduce(func_nr, (data.val, data.id))
        self.assertTrue(arr_eq(result['val'], [0, 10, 21, 51]))

        result = Cat(data.id).apply_reduce(func_r, (data.val, data.id))
        self.assertTrue(arr_eq(result['val'], [10, 51]))

    def test_apply_ema(self):

        correct = FastArray(
            [
                150.63090515,
                -3.11271882,
                0.0,
                -43.95726776,
                -36.96357727,
                -58.97815704,
                -80.67763519,
                33.27441788,
                32.07076263,
                -38.05633926,
                -31.82532883,
                505.58901978,
                -53.8103981,
                42.10805893,
                -289.54257202,
                -433.78964233,
                0.0,
                210.54029846,
                31.44789696,
                105.86235809,
                65.31745148,
                -111.80770111,
                22.91087723,
                288.18487549,
                -2536.56005859,
            ],
            dtype=np.float32,
        )

        sym = FastArray(
            # Top SPX constituents by weight as of 20200414
            [
                b'MSFT',
                b'AAPL',
                b'AMZN',
                b'FB',
                b'JNJ',
                b'GOOG',
                b'GOOGL',
                b'BRK.B',
                b'PG',
                b'JPM',
                b'V',
                b'INTC',
                b'UNH',
                b'VZ',
                b'MA',
                b'T',
                b'HD',
                b'MRK',
                b'PFE',
                b'PEP',
                b'BAC',
                b'DIS',
                b'KO',
                b'WMT',
                b'CSCO',
            ],
            dtype='|S8',
        )
        sym = Categorical(sym)

        org = FastArray(
            [
                0.0,
                150.63090515,
                -3.11271882,
                -80.67763519,
                505.58901978,
                33.27441788,
                32.07076263,
                -43.95726776,
                105.86235809,
                -53.8103981,
                22.91087723,
                -31.82532883,
                -111.80770111,
                288.18487549,
                -289.54257202,
                65.31745148,
                -38.05633926,
                -433.78964233,
                31.44789696,
                210.54029846,
                0.0,
                -58.97815704,
                42.10805893,
                -2536.56005859,
                -36.96357727,
            ]
        )

        time = FastArray(
            [
                3.98803711,
                10.36206055,
                25.64599609,
                26.74609375,
                27.3269043,
                27.71777344,
                27.88891602,
                27.88891602,
                30.08300781,
                30.32299805,
                31.14013672,
                32.32885742,
                32.96582031,
                35.02587891,
                35.02587891,
                35.30810547,
                37.74584961,
                38.34008789,
                38.37207031,
                38.37207031,
                39.13793945,
                40.09399414,
                41.65991211,
                42.22998047,
                47.26000977,
            ]
        )

        ds = Dataset({'sym': sym, 'org': org, 'time': time})

        def get_ema_userfunc(x, decay_rate=None):
            t = x.time
            v = x.org
            arrlen = len(v)

            ema = empty(arrlen)
            lastEma = 0
            lastTime = 0

            for i in range(arrlen):
                ema[i] = v[i] + lastEma * np.exp(-decay_rate * (t[i] - lastTime))
                lastEma = ema[i]
                lastTime = t[i]

            return {'ema': ema}

        # this is when two inputs specified via a tuple, but no output
        def get_ema_userfunc2(v, t, decay_rate=None):
            arrlen = len(v)
            ema = empty(arrlen)
            lastEma = 0
            lastTime = 0

            for i in range(arrlen):
                ema[i] = v[i] + lastEma * np.exp(-decay_rate * (t[i] - lastTime))
                lastEma = ema[i]
                lastTime = t[i]
            return ema

        # this is when two inputs specified via a tuple and...
        # this is when the output is preallocated via a dtype dictionary
        def get_ema_userfunc3(v, t, ema, decay_rate):
            lastEma = 0
            lastTime = 0
            for i in range(len(v)):
                ema[i] = v[i] + lastEma * np.exp(-decay_rate * (t[i] - lastTime))
                lastEma = ema[i]
                lastTime = t[i]

        decay = np.log(2) / (1e3 * 100)

        ema_result = ds.gb('sym').apply(get_ema_userfunc, decay_rate=decay)
        self.assertTrue(
            sum(correct - ema_result.ema) < 0.01,
            msg=f"got ema {ema_result['ema']}, wanted {correct}\nresult {sum(correct - ema_result.ema)}",
        )

        # here we use a tuple to pass in two values
        ema_result = ds.gb('sym').apply_nonreduce(
            get_ema_userfunc2, (ds.org,), decay_rate=decay
        )
        self.assertTrue(sum(correct - ema_result[0]) < 0.01)

        # here we use dtype with a dict to specify one output
        ema_result = ds.gb('sym').apply_nonreduce(
            get_ema_userfunc3, (ds.org,), decay_rate=decay, dtype={'ema': np.float64}
        )
        self.assertTrue(sum(correct - ema_result[0]) < 0.01)

        # we pass in tuple of a Dataset to specify the inputs
        # here we use dtype with a dict to specify one output
        ema_result = ds.sym.apply_nonreduce(
            get_ema_userfunc3,
            (ds[['org', 'time']],),
            decay_rate=decay,
            dtype={'ema': np.float64},
        )
        self.assertTrue(sum(correct - ema_result[0]) < 0.01)

        # we pass in tuple of two values
        # here we use dtype with a dict to specify one output
        # we change the output type to np.float32 and check the column name is 'ema'
        ema_result = ds.sym.apply_nonreduce(
            get_ema_userfunc3,
            (ds.org, ds.time),
            decay_rate=decay,
            dtype={'ema': np.float32},
        )
        self.assertTrue(
            sum(correct - ema_result['ema']) < 0.01,
            msg=f"got ema {ema_result['ema']}, wanted {correct}",
        )

        # create another org
        ds.org2 = ds.org + 1
        ema_result = ds.gb('sym')['org', 'org2'].apply_nonreduce(
            get_ema_userfunc2, (ds.time,), decay_rate=decay
        )
        self.assertTrue(sum(correct - ema_result['org']) < 0.01)

        ema_result = ds.sym.apply_nonreduce(
            get_ema_userfunc2, [ds.org, ds.org2], (ds.time,), decay_rate=decay
        )
        self.assertTrue(sum(correct - ema_result['org']) < 0.01)

    def test_apply_tuple(self):
        arrsize = 200
        numrows = 7

        ds = Dataset({'time': arange(arrsize * 1.0)})
        ds.data = arange(arrsize) % numrows
        ds.data2 = (arange(arrsize) + 3) % numrows
        symbols = [
            'AAPL',
            'AMZN',
            'FB',
            'GOOG',
            'IBM',
            '6',
            '7',
            '8',
            '9',
            '10',
            '11',
            '12',
            '13',
            '14',
            '15',
            '16',
            '17',
            '18',
        ]
        ds.symbol = Cat(1 + arange(arrsize) % len(symbols), symbols)

        # put the second argument in tuples
        result = ds.symbol.apply_reduce(
            lambda x, y: np.sum(np.minimum(x, y)), (ds.data, ds.data)
        )[0]

        self.assertTrue(arr_eq(result[:4], np.asarray([33, 38, 32, 29])))

        # now test nanmin on integers
        ds.data[5] = np.nan
        ds.datau = ds.data.astype(np.uint64)
        result = ds.symbol.nansum(ds.data)[0]
        self.assertTrue(sum((result >= 0) & (result <= 100)) == 18)
        result = ds.symbol.nanmin(ds.data)[0]
        self.assertTrue(sum(result) == 0)
        result = ds.symbol.nanmax(ds.data)[0]
        self.assertTrue(sum(result == 6) == 18)
        result = ds.symbol.nansum(ds.datau)[0]
        self.assertTrue(sum((result >= 0) & (result <= 100)) == 18)
        result = ds.symbol.nanmin(ds.datau)[0]
        self.assertTrue(sum(result) == 0)
        result = ds.symbol.nanmax(ds.datau)[0]
        self.assertTrue(sum(result == 6) == 18)

        # test passing transform to apply_reduce
        ds.gb('data').apply_reduce(sum, transform=True)

    def test_apply_gb_order(self):
        ds = Dataset(
            {
                'Type': ['Put', 'Call', 'Put', 'Put', 'Call', 'Call'],
                'Delta': [-0.7, 0.2, -0.4, -0.6, 0.3, 0.8],
                'PnL': [51, 3, 60, 529, 2, 36],
            }
        )
        maxpnl = ds.groupby(['Type'])['PnL'].max(transform=True)[0]
        self.assertTrue(maxpnl[0] == 529)
        self.assertTrue(maxpnl[1] == 36)

    def test_apply_gb_vs_cat(self):
        N = 100
        np.random.seed(1)
        ds = Dataset(
            dict(
                Symbol=Cat(np.random.choice(['SPY', 'IBM'], N)),
                Exchange=Cat(np.random.choice(['AMEX', 'NYSE'], N)),
                TradeSize=np.random.choice([1, 5, 10], N),
                TradePrice=np.random.choice([1.1, 2.2, 3.3], N),
            )
        )

        rds = ds.groupby(['Symbol', 'Exchange']).apply(
            lambda x: {
                'Count': len(x.TradeSize),
                'dollars_exec': (x.TradeSize * x.TradePrice).nansum(),
            }
        )

        c = ds.cat(['Symbol', 'Exchange'], ordered=True)
        newds = c.count()
        newds['dollars_exec'] = c.apply_reduce(
            lambda x, y: (x * y).nansum(), ds.TradeSize, (ds.TradePrice,)
        )
        newds
        self.assertTrue(newds.equals(rds))

        newds = ds.cat(['Symbol', 'Exchange'], ordered=True).apply(
            lambda x: {
                'Count': len(x.TradeSize),
                'dollars_exec': (x.TradeSize * x.TradePrice).nansum(),
            },
            ds,
        )
        self.assertTrue(newds.equals(rds))

        r1 = ds.groupby(['Symbol', 'Exchange'])['TradeSize'].apply_reduce(
            lambda x: len(x)
        )[0]
        r2 = ds.groupby(['Symbol', 'Exchange']).count()[0]
        self.assertTrue(arr_eq(r1, r2))

    def test_apply_computable(self):
        ds = Dataset(
            {'a': Cat(list('AABB')), 'b': ['foo', 'bar', 'baz', 'bazz']}, unicode=True
        )
        result = ds['a'].apply_nonreduce(FA.shift, ds['b'], computable=False)
        self.assertTrue(result['b'][1] == 'foo')

        ds.c = arange(4)
        ds['d'] = ds.a.copy()
        ds.c = arange(4)
        result = ds['a'].apply_nonreduce(
            FA.shift, ds[['b', 'c', 'd']], computable=False
        )
        self.assertTrue(result['d'][1] == 'A')


if __name__ == "__main__":
    tester = unittest.main()
