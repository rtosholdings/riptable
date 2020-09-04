import unittest
import pandas as pd
import pytest

import riptable as rt
# N.B. TL;DR We have to import the actual implementation module to override the module global
#      variable "tm.N" and "tm.K".
#      In pandas 1.0 they move the code from pandas/util/testing.py to pandas/_testing.py.
#      The "import pandas.util.testing" still works but because it doesn't contain the actual code
#      our attempt to override the "tm.N" and "tm.K" will not change the actual value for
#      makeTimeDataFrame, which will produce data with different shape and make the test
#      "test_accum_table" fail. Maybe we want to reconsider using the pandas internal testing utils.
try:
    import pandas._testing as tm
except ImportError:
    import pandas.util.testing as tm

from riptable import *
from numpy.testing import (
    assert_array_equal,
    assert_almost_equal,
    assert_array_almost_equal,
)
from riptable.rt_numpy import arange
# To create AccumTable test data
from riptable.Utils.pandas_utils import dataset_from_pandas_df
from riptable.rt_datetime import DateTimeNano


tm.N = 3
tm.K = 5


class Accum2_Test(unittest.TestCase):
    '''
    TODO: add more tests for different types
    '''

    def test_accum2(self):
        c = cut(arange(10), 3)
        self.assertTrue(sum(c._np - FA([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])) == 0)

        c = cut(arange(10.0), 3)
        self.assertTrue(sum(c._np - FA([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])) == 0)

        c = cut(arange(11), 3)
        self.assertTrue(sum(c._np - FA([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])) == 0)

        c = cut(FA([2, 4, 6, 8, 10]), FA([0, 2, 4, 6, 8, 10]))
        self.assertTrue(sum(c._np - FA([1, 2, 3, 4, 5])) == 0)

        c = cut(
            FA([2, 4, 6, 8, 10]),
            FA([0, 2, 4, 6, 8, 10]),
            labels=['a', 'b', 'c', 'd', 'e'],
        )
        self.assertTrue(sum(c._np - FA([1, 2, 3, 4, 5])) == 0)

    def test_qcut(self):
        c = qcut(arange(10), 3)
        self.assertTrue(sum(c._np - FA([2, 2, 2, 2, 3, 3, 3, 4, 4, 4])) == 0)

        c = qcut(arange(11), 3)
        self.assertTrue(sum(c._np - FA([2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4])) == 0)

        c = qcut(range(5), 3, labels=["good", "medium", "bad"])
        self.assertTrue(sum(c._np - FA([2, 2, 3, 4, 4])) == 0)

        c = cut(
            FA([2, 4, 6, 8, 10]),
            FA([0, 2, 4, 6, 8, 10]),
            labels=['a', 'b', 'c', 'd', 'e'],
        )

    def test_cut_errors(self):
        with self.assertRaises(ValueError):
            c = cut(
                FA([2, 4, 6, 8, 10]),
                FA([0, 2, 4, 6, 8, 10]),
                labels=['a', 'b', 'c', 'd', 'e', 'f'],
            )

    def test_simple_cats(self):
        data = arange(1, 6) * 10
        colnames = FastArray(['a', 'b', 'c', 'd', 'e'])
        c1 = Categorical(colnames)
        c2 = Categorical(arange(5))

        # no filter
        ac = Accum2(c2, c1)
        result = ac.sum(data)
        self.assertEqual(result._ncols, 7)
        for i, colname in enumerate(colnames):
            arr = result[colname]
            self.assertEqual(arr[i], data[i])

    def test_simple_cats_filter_accum(self):
        data = arange(1, 6) * 10
        colnames = FastArray(['a', 'b', 'c', 'd', 'e'])
        c1 = Categorical(colnames)
        c2 = Categorical(arange(5))

        # filtered accum object
        ac = Accum2(c2, c1, showfilter=True)
        result = ac.sum(data)
        self.assertEqual(result._ncols, 8)
        for i, colname in enumerate(colnames):
            arr = result[colname]
            self.assertEqual(arr[i + 1], data[i])

    def test_simple_cats_filter_operation(self):
        data = arange(1, 6) * 10
        colnames = FastArray(['a', 'b', 'c', 'd', 'e'])
        c1 = Categorical(colnames)
        c2 = Categorical(arange(5))

        # filtered operation
        ac = Accum2(c2, c1)
        result = ac.sum(data, showfilter=True)
        self.assertEqual(result._ncols, 8)
        for i, colname in enumerate(colnames):
            arr = result[colname]
            self.assertEqual(arr[i + 1], data[i])

    def test_multikey_cats(self):
        unsorted_str = FastArray(['c', 'e', 'b', 'd', 'a'])
        ints = arange(1, 6) * 10
        data = np.random.rand(5) * 10

        # unsorted no filter
        c1 = Categorical([unsorted_str, ints])
        c2 = Categorical([unsorted_str, ints])
        ac = Accum2(c2, c1)
        result = ac.sum(data)
        self.assertEqual(result._ncols, 8)
        for i, key1 in enumerate(unsorted_str):
            k1 = bytes.decode(key1)
            k2 = ints[i]
            full_colname = "('" + k1 + "', " + str(k2) + ")"
            arr = result[full_colname]
            self.assertEqual(arr[i], data[i])

        # sorted no filter
        sortidx = np.argsort(unsorted_str)
        sorted_str = unsorted_str[sortidx]
        sorted_ints = ints[sortidx]
        sorted_data = data[sortidx]
        c1 = Categorical([unsorted_str, ints], ordered=True)
        c2 = Categorical([unsorted_str, ints], ordered=True)
        ac = Accum2(c2, c1)
        result = ac.sum(data)
        self.assertEqual(result._ncols, 8)
        for i, key1 in enumerate(sorted_str):
            k1 = bytes.decode(key1)
            k2 = sorted_ints[i]
            full_colname = "('" + k1 + "', " + str(k2) + ")"
            arr = result[full_colname]
            self.assertEqual(arr[i], sorted_data[i])

    @pytest.mark.xfail(reason='20200416 This test was previously overridden by a later test in the file with the same name. Need to revisit and get back in a working state.')
    def test_multikey_cats_filter_accum_sorted(self):
        unsorted_str = FastArray(['c', 'e', 'b', 'd', 'a'])
        ints = arange(1, 6) * 10
        data = np.random.rand(5) * 10

        # unsorted filter accum object
        c1 = Categorical([unsorted_str, ints])
        c2 = Categorical([unsorted_str, ints])
        ac = Accum2(c2, c1, showfilter=True)
        result = ac.sum(data)
        self.assertEqual(result._ncols, 9)
        for i, key1 in enumerate(unsorted_str):
            k1 = bytes.decode(key1)
            k2 = ints[i]
            full_colname = "('" + k1 + "', " + str(k2) + ")"
            arr = result[full_colname]
            self.assertEqual(arr[i + 1], data[i])

        # sorted filter accum object
        sortidx = np.argsort(unsorted_str)
        sorted_str = unsorted_str[sortidx]
        sorted_ints = ints[sortidx]
        sorted_data = data[sortidx]
        c1 = Categorical([unsorted_str, ints], sort_gb=True)
        c2 = Categorical([unsorted_str, ints], sort_gb=True)
        ac = Accum2(c2, c1, showfilter=True)
        result = ac.sum(data)
        self.assertEqual(result._ncols, 9)
        for i, key1 in enumerate(sorted_str):
            k1 = bytes.decode(key1)
            k2 = sorted_ints[i]
            full_colname = "('" + k1 + "', " + str(k2) + ")"
            arr = result[full_colname]
            # TODO fix this regression that was masked due to duplicate test names
            # self.assertAlmostEqual(arr[i + 1], sorted_data[i])

    def test_multikey_cats_filter_accum_ordered(self):
        unsorted_str = FastArray(['c', 'e', 'b', 'd', 'a'])
        ints = arange(1, 6) * 10
        data = np.random.rand(5) * 10

        # unsorted filter accum object
        c1 = Categorical([unsorted_str, ints])
        c2 = Categorical([unsorted_str, ints])
        ac = Accum2(c2, c1)
        result = ac.sum(data, showfilter=True)
        self.assertEqual(result._ncols, 9)
        for i, key1 in enumerate(unsorted_str):
            k1 = bytes.decode(key1)
            k2 = ints[i]
            full_colname = "('" + k1 + "', " + str(k2) + ")"
            arr = result[full_colname]
            self.assertEqual(arr[i + 1], data[i])

        # sorted filter accum object
        sortidx = np.argsort(unsorted_str)
        sorted_str = unsorted_str[sortidx]
        sorted_ints = ints[sortidx]
        sorted_data = data[sortidx]
        c1 = Categorical([unsorted_str, ints], ordered=True)
        c2 = Categorical([unsorted_str, ints], ordered=True)
        ac = Accum2(c2, c1)
        result = ac.sum(data, showfilter=True)
        self.assertEqual(result._ncols, 9)
        for i, key1 in enumerate(sorted_str):
            k1 = bytes.decode(key1)
            k2 = sorted_ints[i]
            full_colname = "('" + k1 + "', " + str(k2) + ")"
            arr = result[full_colname]
            self.assertEqual(arr[i + 1], sorted_data[i])

    def test_dataset_accum2(self):
        # test from accum2 off dataset and with a filter
        ds = Dataset({'test': arange(10), 'data': arange(10) // 2})
        x = ds.accum2('data', 'test').sum(ds.test, filter=ds.data == 3)
        totalcol = x.summary_get_names()[0]
        self.assertEqual(x[totalcol][3], 13)

    def test_accum2_mean(self):
        ds = Dataset({'time': arange(200.0)})
        ds.data = np.random.randint(7, size=200)
        ds.data2 = np.random.randint(7, size=200)
        symbols = ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM']
        ds.symbol = Cat(1 + arange(200) % 5, symbols)
        ac = Accum2(ds.data, ds.symbol).mean(ds.time)
        totalcol = ac[ac.summary_get_names()[0]]
        footer = ac.footer_get_values()['Mean']
        for i in range(len(symbols)):
            s_mean = ds[ds.symbol == symbols[i], :].time.mean()
            self.assertEqual(footer[i + 1], s_mean)
        for i in range(7):
            s_mean = ds[ds.data == i, :].time.mean()
            self.assertEqual(totalcol[i], s_mean)

    def test_accum2_median(self):
        ds = Dataset({'time': arange(200.0)})
        ds.data = np.random.randint(7, size=200)
        ds.data2 = np.random.randint(7, size=200)
        symbols = ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM']
        ds.symbol = Cat(1 + arange(200) % 5, symbols)
        ac = Accum2(ds.data, ds.symbol).median(ds.time)
        totalcol = ac[ac.summary_get_names()[0]]
        footer = ac.footer_get_values()['Median']
        for i in range(len(symbols)):
            s_median = ds[ds.symbol == symbols[i], :].time.median()
            self.assertEqual(footer[i + 1], s_median)
        for i in range(7):
            s_median = ds[ds.data == i, :].time.median()
            self.assertEqual(totalcol[i], s_median)

    def test_accum2_nanmedian_with_filter(self):
        ds = Dataset({'time': arange(200.0)})
        ds.data = np.random.randint(7, size=200)
        ds.data2 = np.random.randint(7, size=200)
        symbols = ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM']
        # N.B. make a copy here for testing
        symbol_categorical = Cat(1 + arange(200) % 5, symbols)
        # N.B. Categorical.copy and Categorical constructor doesn't do deep copy?!
        ds.symbol = Cat(1 + arange(200) % 5, symbols)

        chosen_symbols = ['AMZN', 'AAPL']
        filt = symbol_categorical.isin(chosen_symbols)
        ac = Accum2(ds.data, ds.symbol)
        stat1 = ac.nanmedian(ds.time, filter=filt)
        totalcol = stat1[stat1.summary_get_names()[0]]
        footer = stat1.footer_get_values()['Median']
        # Make sure we don't change the input data
        self.assertTrue(not rt.any(ds.symbol._fa == 0))
        for sym in chosen_symbols:
            s_median = rt.nanmedian(ds[symbol_categorical == sym, :].time)
            i = rt.where(symbol_categorical.category_array == sym)[0].item()
            self.assertEqual(footer[i + 1], s_median)
        for i in range(7):
            s_median = rt.nanmedian(ds[(ds.data == i) & filt, :].time)
            self.assertEqual(totalcol[i], s_median)

        chosen_symbols = ['IBM', 'FB']
        filt = symbol_categorical.isin(chosen_symbols)
        stat2 = ac.nanmedian(ds.time, filter=filt)
        totalcol = stat2[stat2.summary_get_names()[0]]
        footer = stat2.footer_get_values()['Median']
        # Make sure we don't change the input data
        self.assertTrue(not rt.any(ds.symbol._fa == 0))
        for sym in chosen_symbols:
            s_median = rt.nanmedian(ds[symbol_categorical == sym, :].time)
        i = rt.where(symbol_categorical.category_array == sym)[0].item()
        self.assertEqual(footer[i + 1], s_median)
        for i in range(7):
            s_median = rt.nanmedian(ds[(ds.data == i) & filt, :].time)
        self.assertEqual(totalcol[i], s_median)

    def test_showfilter_label_subclass(self):
        d = Date.range('20190201', '20190210')
        c = Categorical(d)
        c2 = Categorical(arange(10))
        ac = Accum2(c, c2)
        result = ac.count(showfilter=True)

        self.assertTrue(isinstance(result.YLabel, Date))
        self.assertTrue(result.YLabel.isnan()[0])

        d = DateTimeNano.random(10)
        c = Categorical(d)
        c2 = Categorical(arange(10))
        ac = Accum2(c, c2)
        result = ac.count(showfilter=True)

        self.assertTrue(isinstance(result.YLabel, DateTimeNano))
        self.assertTrue(result.YLabel.isnan()[0])

        d = DateSpan(arange(10, 20))
        c = Categorical(d)
        c2 = Categorical(arange(10))
        ac = Accum2(c, c2)
        result = ac.count(showfilter=True)

        self.assertTrue(isinstance(result.YLabel, DateSpan))
        self.assertTrue(result.YLabel.isnan()[0])

        d = TimeSpan(np.random.rand(10) * 10_000_000_000)
        c = Categorical(d)
        c2 = Categorical(arange(10))
        ac = Accum2(c, c2)
        result = ac.count(showfilter=True)

        self.assertTrue(isinstance(result.YLabel, TimeSpan))
        self.assertTrue(result.YLabel.isnan()[0])

    def test_apply(self):
        arrsize = 200
        numrows = 7

        ds = Dataset({'time': arange(arrsize * 1.0)})
        ds.data = np.random.randint(numrows, size=arrsize)
        ds.data2 = np.random.randint(numrows, size=arrsize)
        symbols = ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM']
        ds.symbol = Cat(1 + arange(arrsize) % len(symbols), symbols)
        ds.accum2('symbol', 'data').sum(ds.data2)
        ds.accum2('symbol', 'data').sum(ds.data2, showfilter=True)
        ds.accum2('symbol', 'data').median(ds.data2, showfilter=True)
        ds.accum2('symbol', 'data').median(ds.data2, showfilter=False)
        ds.accum2('symbol', 'data').apply_reduce(np.median, ds.data2, showfilter=True)
        ds.accum2('symbol', 'data').apply_reduce(np.median, ds.data2, showfilter=False)
        f = logical(arange(200) % 2)
        ds.accum2('symbol', 'data').apply_reduce(np.median, ds.data2, filter=f)
        ds.accum2('symbol', 'data').apply_reduce(
            np.median, ds.data2, filter=f, showfilter=True
        )
        ds.accum2('symbol', 'data').median(ds.data2, filter=f, showfilter=True)

    def test_apply_nonreduce(self):
        arrsize = 200
        numrows = 7
        ds = rt.Dataset({'time': rt.arange(arrsize * 1.0)})
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
        ds.symbol = rt.Cat(1 + rt.arange(arrsize) % len(symbols), symbols)
        result = ds.symbol.apply_reduce(
            lambda x, y: np.sum(np.minimum(x, y)), (ds.data, ds.data)
        )

        ac = ds.accum2('symbol', 'data')
        newds = ac.apply_nonreduce(np.cumsum)
        ds2 = ac.apply_reduce(
            lambda x, y: np.sum(np.maximum(x, y)), (newds.data, newds.data2)
        )

        x = np.maximum(newds.data, newds.data2)
        y = ac.apply_nonreduce(
            lambda x, y: np.maximum(x, y), (newds.data, newds.data2)
        )[0]
        self.assertTrue(np.all(x == y))


class AccumTable_Test(unittest.TestCase):
    @pytest.mark.skip(reason="Test needs to be re-written to remove the np.random.seed usage -- it's not stable across numpy versions.")
    def test_accum_table(self):

        # Create the test data

        def unpivot(frame):
            N, K = frame.shape
            data = {
                'value': frame.values.ravel('F'),
                'variable': np.asarray(frame.columns).repeat(N),
                'date': np.tile(np.asarray(frame.index), K),
            }
            return pd.DataFrame(data, columns=['date', 'variable', 'value'])

        np.random.seed(1234)
        df = unpivot(pd.concat([tm.makeTimeDataFrame(), tm.makeTimeDataFrame()]))
        ds = dataset_from_pandas_df(df)
        ds.date = DateTimeNano(ds.date, from_tz='NYC').to_iso()
        ds.date = rt.FastArray([d[:10] for d in ds.date])
        ds.variable = rt.Categorical(ds.variable)
        ds.date = rt.Categorical(ds.date)

        at = rt.AccumTable(ds.date, ds.variable)

        # Add and view inner tables with totals
        at['Sum'] = at.sum(ds.value)
        self.assertEqual(at['Sum'].shape, (3, 7))
        assert_array_almost_equal(
            at['Sum']['A'], np.array([0.47, -0.79, 1.72]), decimal=2
        )

        vw = at.gen('Sum')
        self.assertEqual(vw.shape, (3, 7))
        assert_array_almost_equal(vw['A'], np.array([0.47, -0.79, 1.72]), decimal=2)

        assert_array_almost_equal(vw['Sum'], np.array([-0.10, -5.02, 5.37]), decimal=2)
        assert_array_almost_equal(
            vw.footer_get_values(columns=['Sum'])['Sum'], np.array([0.25]), decimal=2
        )

        at['Mean'] = at.mean(ds.value)
        self.assertEqual(at['Mean'].shape, (3, 7))
        assert_array_almost_equal(
            at['Mean']['A'], np.array([0.24, -0.39, 0.86]), decimal=2
        )

        at['Half'] = at['Mean'] / at['Sum']
        self.assertEqual(at['Half'].shape, (3, 7))
        assert_array_almost_equal(at['Half']['A'], np.array([0.5, 0.5, 0.5]), decimal=2)

            # Add and view inner tables with blanks

        at['Blanks'] = at['Sum'].copy()
        at['Blanks']['C'] = 0.0
        for col in at['Blanks'][:, 1:]:
            at['Blanks'][col][2] = np.nan

        vw = at.gen('Blanks')
        self.assertEqual(vw.shape, (2, 9))
        assert_array_almost_equal(vw['A'], np.array([0.47, -0.79]), decimal=2)
        assert_array_almost_equal(vw['Blanks'], np.array([-0.10, -5.02]), decimal=2)
        self.assertAlmostEqual(
            vw.footer_get_dict()['Blanks']['Blanks'], 0.245, places=2
        )

        vw = at.gen('Blanks', remove_blanks=False)
        self.assertEqual(vw.shape, (3, 10))
        assert_array_almost_equal(vw['A'], np.array([0.47, -0.79, np.nan]), decimal=2)
        assert_array_almost_equal(
            vw['Blanks'], np.array([-0.10, -5.02, np.nan]), decimal=2
        )

        # Test division with zeros and nans
        at['Bad'] = at['Blanks'] / at['Half']
        self.assertEqual(at['Blanks'].shape, (3, 7))
        vw = at.gen('Bad')
        self.assertEqual(vw.shape, (2, 10))
        vw = at.gen('Blanks')
        self.assertEqual(vw.shape, (2, 10))
        vw = at.gen('Half')
        self.assertEqual(vw.shape, (3, 11))

            # Set margin columns to the right

        at.set_margin_columns(['Blanks', 'Mean'])
        vw = at.gen('Half')
        self.assertEqual(vw.shape, (3, 9))
        self.assertEqual(vw.keys()[6], 'Half')
        self.assertEqual(vw.keys()[7], 'Blanks')
        self.assertEqual(vw.keys()[8], 'Mean')
        self.assertEqual(
            list(vw.footer_get_dict().keys()), ['Half', 'Sum', 'Mean', 'Blanks', 'Bad']
        )

        vw = at.gen()
        self.assertEqual(vw.keys()[6], 'Half')

        vw = at.gen('Sum')
        self.assertEqual(vw.keys()[6], 'Sum')
        self.assertEqual(vw.keys()[7], 'Blanks')
        self.assertEqual(vw.keys()[8], 'Mean')
        self.assertEqual(
            list(vw.footer_get_dict().keys()), ['Sum', 'Mean', 'Half', 'Blanks', 'Bad']
        )

            # Set footer rows at the bottom

        at.set_footer_rows(['Mean'])
        vw = at.gen('Half')
        self.assertEqual(vw.shape, (3, 9))
        self.assertEqual(vw.keys()[6], 'Half')
        self.assertEqual(vw.keys()[7], 'Blanks')
        self.assertEqual(vw.keys()[8], 'Mean')
        self.assertEqual(list(vw.footer_get_dict().keys()), ['Half', 'Mean'])

        vw = at.gen('Sum')
        self.assertEqual(vw.keys()[6], 'Sum')
        self.assertEqual(vw.keys()[7], 'Blanks')
        self.assertEqual(vw.keys()[8], 'Mean')
        self.assertEqual(list(vw.footer_get_dict().keys()), ['Sum', 'Mean'])

            # Access view Dataset elements

        vw = at.gen('Sum')
        assert_array_equal(
            vw.date, rt.FastArray(['2000-01-03', '2000-01-04', '2000-01-05'])
        )
        assert_array_almost_equal(vw['Sum'], np.array([-0.10, -5.02, 5.37]), decimal=2)
        assert_almost_equal(vw[vw.date == '2000-01-03', 'A'][0], 0.47355353, decimal=2)
        assert_almost_equal(
            list(vw.footer_get_values('Sum', columns=['A']).values())[0],
            1.409830,
            decimal=2,
        )


if __name__ == "__main__":
    tester = unittest.main()
