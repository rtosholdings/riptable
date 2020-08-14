import unittest
import os

from math import isclose
from riptable import *
from riptable import FastArray, Dataset, GroupBy, Categorical, arange, isnan, isnotnan
from riptable.rt_enum import (
    INVALID_DICT,
)
from riptable.Utils.pandas_utils import dataset_as_pandas_df
from riptable.tests.utils import LikertDecision


num_list = [3, 4, 5, 1, 2]
num_list_sorted = [1, 2, 3, 4, 5]
arr_types = [
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
    np.float32,
    np.float64,
]
arr_types_string = [np.bytes_, np.str_]
test_data = {'bool': np.array([True, False, True, False, True], dtype=np.bool)}
for dt in arr_types + arr_types_string:
    test_data[dt.__name__] = np.array(num_list, dtype=dt)
test_data['categorical'] = Categorical([str(i) for i in num_list])
all_headers = list(test_data.keys())
ds = Dataset(test_data)
gb_funcs = ['sum', 'mean', 'first', 'last', 'median', 'min', 'max', 'var']
gb_nan_funcs = ['nansum', 'nanmean', 'nanmedian', 'nanvar']  #'rolling', 'cumsum', 'nth'


class Groupby_Test(unittest.TestCase):
    def test_math_ops_same_return(self):
        result_dict = {
            'sum': [5, 10],
            'nansum': [5, 10],
            'median': [2.5, 3],
            # TODO: add support for min / max on strings
            'min': [1, 2],
            'max': [4, 5],
        }
        gb = ds.gb('bool')
        for name, correct in result_dict.items():
            call = getattr(gb, name)
            ds_result = call()
            for dt in arr_types:
                column = getattr(ds_result, dt.__name__)
                is_correct = bool(np.all(column == np.array(correct, dtype=dt)))
                self.assertTrue(
                    is_correct,
                    msg=f"Incorrect result for keys {ds_result.label_get_names()} in column with datatype {dt} after {name} operation.",
                )

            # make sure string columns are not included
            for dt in arr_types_string:
                was_included = hasattr(ds_result, dt.__name__)
                self.assertFalse(
                    was_included,
                    msg=f"String-like column {dt.__name__} was included in {name} math operation.",
                )

    def test_filter(self):
        correct = [1, 2]
        result = ds.gb('bool', filter=[False, False, False, True, True]).min()
        for dt in arr_types:
            column = getattr(result, dt.__name__)
            is_correct = bool(np.all(column == np.array(correct, dtype=dt)))
            self.assertTrue(
                is_correct,
                msg=f"Incorrect result in column of datatype {dt} when filter was applied on sorted groupby. {column} {correct}",
            )

    def test_math_ops_float_return(self):
        gb = ds.gb('bool')
        result_dict = {
            'mean': [2.5, 3.333333333],
            'std': [2.121320343560, 1.527525231652],
            'var': [4.5, 2.333333333333],
        }
        for name, correct in result_dict.items():
            call = getattr(gb, name)
            ds_result = call()
            for dt in arr_types:
                column = getattr(ds_result, dt.__name__)
                for idx, item in enumerate(column):
                    is_accurate = isclose(item, correct[idx], rel_tol=1e-06)
                    self.assertTrue(
                        is_accurate,
                        msg=f"Incorrect result for keys {ds_result.label_get_names()} in column with datatype {dt} after {name} operation. {item} != {correct[idx]}",
                    )

            # make sure string columns are not included
            for dt in arr_types_string:
                was_included = hasattr(ds_result, dt.__name__)
                self.assertFalse(
                    was_included,
                    msg=f"String-like column {dt.__name__} was included in {name} math operation.",
                )

    def test_keys_included(self):
        '''
        By default, groupby keys will not be included in the dataset's main data for the groupby calculation.
        The keyword include_keys can be set to include them.
        '''
        # now included
        ds_result = ds.gb('bool').sum()
        was_included = hasattr(ds_result, 'bool')
        self.assertTrue(
            was_included,
            msg=f"Groupby key columns were included in main data. (they should not be by default)",
        )

        # included (by keyword)
        ds_result = ds.gb('bool', include_keys=True).sum()
        was_included = hasattr(ds_result, 'bool')
        self.assertTrue(
            was_included,
            msg=f"Groupby key columns were NOT included in main data, even though the keyword was set.",
        )

        # included (by global flag)
        GroupBy.include_keys = True
        ds_result = ds.gb('bool').sum()
        was_included = hasattr(ds_result, 'bool')
        self.assertTrue(
            was_included,
            msg=f"Groupby key columns were NOT included in main data, even though the global flag was set.",
        )
        GroupBy.include_keys = False

    def test_ops_str_allowed(self):
        gb = ds.gb('bool')
        result_dict = {'first': [4, 3], 'last': [1, 2]}
        for name, correct in result_dict.items():
            call = getattr(gb, name)
            ds_result = call()
            # check numeric columns
            for dt in arr_types:
                column = getattr(ds_result, dt.__name__)
                is_correct = bool(np.all(column == np.array(correct, dtype=dt)))
                self.assertTrue(
                    is_correct,
                    msg=f"Incorrect result for keys {ds_result.label_get_names()} in column with datatype {dt} after {name} operation.",
                )

            # check string columns
            for dt_str in arr_types_string:
                column = getattr(ds_result, dt_str.__name__).astype(np.str_)
                is_correct = bool(np.all(column == np.array(correct, dtype=np.str_)))
                self.assertTrue(
                    is_correct,
                    msg=f"Incorrect result for keys {ds_result.label_get_names()} in column with datatype {dt_str} after {name} operation.",
                )

    def test_math_ops_gb_nums_same_return(self):
        GroupBy.include_keys = True
        test_funcs = ['sum', 'first', 'last', 'nansum']
        for name in test_funcs:
            # check sorted
            for dt in arr_types:
                call = getattr(ds.gb(dt.__name__), name)
                ds_result = call()
                for dt2 in arr_types:
                    column = getattr(ds_result, dt2.__name__)
                    is_correct = bool(
                        np.all(column == np.array(num_list_sorted, dtype=dt))
                    )
                    self.assertTrue(
                        is_correct,
                        msg=f"Incorrect result for column {dt2.__name__} after grouping by {dt.__name__} for {name} operation. (sorted)",
                    )

            # check unsorted
            for dt in arr_types:
                call = getattr(ds.gbu(dt.__name__), name)
                ds_result = call()
                for dt2 in arr_types:
                    column = getattr(ds_result, dt2.__name__)
                    is_correct = bool(np.all(column == np.array(num_list, dtype=dt)))
                    self.assertTrue(
                        is_correct,
                        msg=f"Incorrect result for column {dt2.__name__} after grouping by {dt.__name__} for {name} operation. (unsorted)",
                    )
        GroupBy.include_keys = False

    # funcs that return nan on single item
    def test_math_ops_nan_return(self):
        GroupBy.include_keys = True
        test_funcs = ['std', 'var', 'nanstd', 'nanvar']
        for dt in arr_types + arr_types_string:
            gb = ds.gb(dt.__name__)
            # print("***nan_return\n", ds, dt.__name__)
            for func_name in test_funcs:
                call = getattr(gb, func_name)
                ds_result = call()
                for dt2 in arr_types:
                    column = getattr(ds_result, dt2.__name__)
                    # skip over what we grouped by
                    if dt.__name__ != dt2.__name__:
                        for item in column:
                            self.assertNotEqual(
                                item,
                                item,
                                msg=f"Item was not NaN for groupby {dt} in column {dt2} for {func_name} operation. Got {item} instead.",
                            )
        GroupBy.include_keys = False

    def test_count(self):
        '''
        The result of count only returns a single column, which displays a count of the number of keys in that grouping.
        '''
        result_ds = ds.gb('bool').count()
        self.assertEqual(
            result_ds._ncols, 2, msg=f"Groupby count did not return a single column."
        )

        correct_count = np.array([2, 3])
        result_arr = result_ds.Count
        is_correct = bool(np.all(correct_count == result_arr))
        self.assertTrue(
            is_correct,
            msg=f"Groupby bool did not product the correct result for count. returned {result_arr}, expected {correct_count}",
        )

    def test_nan_funcs(self):
        GroupBy.include_keys = True
        nan_dict = {
            'f32': np.arange(5, dtype=np.float32),
            'f64': np.arange(5, dtype=np.float64),
        }
        nan_dict['f32'][0] = np.nan
        nan_dict['f64'][0] = np.nan
        nan_ds = Dataset(nan_dict)
        # TODO: add when more nan functions have been implemented
        correct = np.array([1.0, 2.0, 3.0, 4.0, 0.0])
        nan_funcs = ['nansum']
        col_names = ['f32', 'f64']

        for c_name in col_names:
            gb = nan_ds.gb(c_name)
            for f_name in nan_funcs:
                call = getattr(gb, f_name)
                result_ds = call()
                for result_name in col_names:
                    column = getattr(result_ds, result_name)
                    # skip groupby column
                    if c_name != result_name:
                        is_correct = bool(np.all(column == correct))
                        self.assertTrue(
                            is_correct,
                            msg=f"Incorrect result for groupby {c_name} in column {result_name} for {f_name} operation.",
                        )
        GroupBy.include_keys = False

    def test_not_supported(self):
        '''
        Remove these from the following list when they are implemented.
        '''
        not_implemented = [
            'sem',
            'ohlc',
            'describe',
            'expanding',
            'ngroup',
            'rank',
            'head',
            'tail',
            'stack',
            'unstack',
        ]
        # 8/28/2018 - SJK removed nth
        not_implemented_one_arg = ['resample']
        not_implemented_two_arg = ['transform']
        gb = ds.gb('bool')

        for func in not_implemented:
            call = getattr(gb, func)
            with self.assertRaises(
                NotImplementedError,
                msg=f"Failed to raise an error for {func}. If this HAS been implemented, remove test from list in test_not_supported in test_sfw_groupby.py",
            ):
                result = call()

        for func in not_implemented_one_arg:
            call = getattr(gb, func)
            with self.assertRaises(
                NotImplementedError,
                msg=f"Failed to raise an error for {func}. If this HAS been implemented, remove test from list in test_not_supported in test_sfw_groupby.py",
            ):
                result = call(1)

        for func in not_implemented_two_arg:
            with self.assertRaises(
                NotImplementedError,
                msg=f"Failed to raise an error for {func}. If this HAS been implemented, remove test from list in test_not_supported in test_sfw_groupby.py",
            ):
                result = call(1, 2)

    def array_equal(self, arr1, arr2):
        subr = arr1 - arr2
        sumr = sum(subr == 0)
        result = sumr == len(arr1)
        if not result:
            print("array comparison failed", arr1, arr2)
        return result

    def assertArrayEqual(self, ar1, ar2):
        assert (ar1 == ar2).all(axis=None), f'array equality failed\nar1 {ar1}\nar2 {ar2}'

    def assertArrayAlmostEqual(self, ar1, ar2, places=5):
        for v1, v2 in zip(ar1, ar2):
            self.assertAlmostEqual(v1, v2, places)

    def test_gb_categoricals(self):
        codes = [1, 44, 44, 133, 75, 75, 75, 1]
        stringlist = ['a', 'b', 'c', 'd', 'e', 'e', 'f', 'g']
        c1 = Categorical(codes, LikertDecision, sort_gb=True)
        c2 = Categorical(stringlist)
        d = {'nums': np.arange(8)}

        # from enum only
        d_enum = d.copy()
        d_enum['cat_from_enum'] = c1
        ds_enum = Dataset(d_enum)
        enum_result = ds_enum.gb('cat_from_enum').sum()
        correct = FastArray([3, 15, 3, 7], dtype=np.int64)
        self.assertTrue(
            self.array_equal(correct, enum_result.nums),
            msg=f"Incorrect sum when grouping by enum categorical.\nExpected {correct}\nActual {enum_result.nums}",
        )

        # from list only
        d_list = d.copy()
        d_list['cat_from_list'] = c2
        ds_list = Dataset(d_list)
        list_result = ds_list.gb('cat_from_list').sum()
        correct = FastArray([0, 1, 2, 3, 9, 6, 7], dtype=np.int64)
        self.assertTrue(
            self.array_equal(correct, list_result.nums),
            msg=f"Incorrect sum when grouping by list categorical.",
        )

        d_both = d_enum.copy()
        d_both['cat_from_list'] = c2
        ds_both = Dataset(d_both)

        # by enum, list
        result = ds_both.gb(['cat_from_enum', 'cat_from_list']).sum()
        num_result = result.nums
        correct = FastArray([0, 7, 1, 2, 9, 6, 3], dtype=np.int64)
        self.assertTrue(
            self.array_equal(correct, num_result),
            msg=f"Incorrect sum when grouping by enum, list categoricals.",
        )

        # by list, enum
        result = ds_both.gb(['cat_from_list', 'cat_from_enum']).sum()
        num_result = result.nums
        correct = FastArray([0, 1, 2, 3, 9, 6, 7], dtype=np.int64)
        self.assertTrue(
            self.array_equal(correct, num_result),
            msg=f"Incorrect sum when grouping by list, enum categoricals.",
        )

    def test_ops(self):
        ds = Dataset(
            {
                'test': arange(300000) % 3,
                'test2': arange(300000.0),
                'test2i': arange(300000),
                'test3': arange(300000) % 3,
            }
        )
        gb = ds.groupby('test')
        result = gb.mean()
        self.assertTrue(result.test2[0] == result.test2i[0])
        self.assertTrue(result.test2[1] == result.test2i[1])
        self.assertTrue(result.test3[1] == 1.0)
        result = gb.median()
        result = gb.trimbr()
        result = gb.nanmedian()

    def test_projections(self):
        num_rows_trade = 1_000_000
        num_symbols = 450
        Trade_Dates = ['20180602', '20180603', '20180604', '20180605', '20180606']
        Exchanges = np.array(['EXCH1', 'EXCH2', 'EXCH3'])
        np.random.seed(1234)
        ds = Dataset(
            {
                'SymbolID': np.random.randint(0, num_symbols, size=num_rows_trade),
                'Exchange': Exchanges[
                    np.random.randint(0, Exchanges.shape[0], size=num_rows_trade)
                ],
                'Trade_Date': [
                    Trade_Dates[int(i * len(Trade_Dates) / num_rows_trade)]
                    for i in range(num_rows_trade)
                ],
                'Time': [
                    int(i % (num_rows_trade / len(Trade_Dates)))
                    for i in range(num_rows_trade)
                ],
                'Price': 100 * (1.0 + 0.0005 * np.random.randn(num_rows_trade)),
                'Size': 10
                * np.array(1 + 30 * np.random.rand(num_rows_trade), dtype=np.int64),
            }
        )
        num_rows_quote = 1_000_000
        ds2 = Dataset(
            {
                'SymbolID': np.random.randint(0, num_symbols, size=num_rows_quote),
                'Exchange': Exchanges[
                    np.random.randint(0, Exchanges.shape[0], size=num_rows_quote)
                ],
                'Trade_Date': [
                    Trade_Dates[int(i * len(Trade_Dates) / num_rows_quote)]
                    for i in range(num_rows_quote)
                ],
                'Time': [
                    int(i % (num_rows_quote / len(Trade_Dates)))
                    for i in range(num_rows_quote)
                ],
                'Bid': 100 * (1.0 - 0.001 + 0.0005 * np.random.randn(num_rows_quote)),
                'Ask': 100 * (1.0 + 0.001 + 0.0005 * np.random.randn(num_rows_quote)),
            }
        )
        threshold = Dataset({'Is_Below_Thresdhold': np.random.rand(num_rows_quote) < 0.75})
        trade_time = Dataset({'time_2500': (ds.Time / 2500).astype(int)})
        trades = Dataset({}).concat_columns([ds, trade_time], do_copy=False)

        # Create GroupBy and corresponding Categorical
        trade_gb = trades.groupby(
            ['SymbolID', 'Exchange', 'Trade_Date', 'time_2500']
        )
        trade_cat = Categorical(
            [ds.SymbolID, ds.Exchange, ds.Trade_Date, trade_time.time_2500]
        )

        # Call sum() and count()
        self.assertEqual(trade_gb.sum().shape, (455654, 7))
        self.assertEqual(trade_cat.sum(ds).shape, (455654, 7))
        self.assertEqual(trade_gb.count().shape, (455654, 5))
        # 8/24/2018 SJK - multikey categorical groupby now returns multiple columns for groupby keys
        self.assertEqual(trade_cat.count().shape, (455654, 5))
        b1 = trade_gb.count().Count.mean()
        b1c = trade_cat.count().Count.mean()
        b2 = trade_gb.count().shape[0]
        self.assertAlmostEqual(ds.shape[0], b1 * b2, places=5)
        self.assertAlmostEqual(ds.shape[0], b1c * b2, places=5)

        # Create ds augmented with filtered ID
        trade_ds = Dataset({'ID': trade_gb.grouping.ikey})
        trade_ds_below_threshold = ds * threshold.Is_Below_Thresdhold
        trade_ds_below_thresholdb = Dataset.concat_columns([trade_ds_below_threshold, trade_ds], do_copy=False)

        # Create trade_ds size projection using GroupBy
        trade_gb_id = trade_ds_below_thresholdb.groupby('ID')
        trade_sizes_ds = trade_gb_id['Size'].sum()
        trade_size_ds = trade_sizes_ds.Size[trade_ds_below_thresholdb.ID - 1]
        self.assertEqual(trade_size_ds.shape[0], ds.shape[0])

        # Create trade_ds size projection using Categorical
        trade_sizes_cat_ds = trade_cat.sum(trade_ds_below_thresholdb.Size)
        trade_size_cat_ds = trade_sizes_cat_ds.Size[trade_cat - 1]
        self.assertArrayAlmostEqual(trade_size_ds, trade_size_cat_ds, places=6)

        # Create trade_ds size projection using Pandas groupby
        ptrade_ds_below_thresholdb = dataset_as_pandas_df(trade_ds_below_thresholdb)
        ptrade_gb_id = ptrade_ds_below_thresholdb.groupby('ID')
        trade_sizes_pd_ds = ptrade_gb_id.sum()
        trade_size_pd_ds = trade_sizes_pd_ds.Size.values[ptrade_gb_id.ngroup()]
        self.assertArrayAlmostEqual(trade_size_ds, trade_size_pd_ds, places=6)

    def test_reductions(self):
        message_types = [
            'CREATE',
            'RUN',
            'CREATE',
            'RUN',
            'RUN',
            'RUN',
            'RUN',
            'CANCEL',
            'RUN',
            'RUN',
            'RUN',
            'CANCEL',
        ]
        order_ids = [1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1]
        seconds = [50, 70, 72, 75, 90, 88, 95, 97, 98, 115, 116, 120]
        shares = [0, 200, 0, 500, 100, 400, 100, 0, 300, 150, 150, 0]
        d2 = dict(
            message_type=message_types,
            order_id=order_ids,
            second=seconds,
            shares=shares,
        )
        dat = Dataset(d2)
        dat = dat[['order_id', 'message_type', 'second', 'shares']]

        # Numeric reduction
        dsr = dat.groupby('order_id').sum()
        self.assertEqual(dsr.shape, (2, 3))
        self.assertArrayEqual(dsr.order_id, [1, 2])
        self.assertArrayEqual(dsr.second, [410, 676])
        self.assertArrayEqual(dsr.shares, [800, 1100])

        # Numeric reduction with all columns returned
        dsr = dat.groupby('order_id', return_all=True).sum()
        self.assertEqual(dsr.shape, (2, 4))
        self.assertEqual(dsr.keys()[1], 'message_type')

        # Order-based reduction
        dsr = dat.groupby('order_id').first()
        self.assertEqual(dsr.shape, (2, 4))
        self.assertArrayEqual(dsr.order_id, [1, 2])
        self.assertArrayEqual(dsr.message_type, ['CREATE', 'CREATE'])
        self.assertArrayEqual(dsr.second, [50, 72])
        self.assertArrayEqual(dsr.shares, [0, 0])

        # Order-based reduction, which returns all columns regardless
        dsr = dat.groupby('order_id', return_all=True).first()
        self.assertEqual(dsr.shape, (2, 4))

        # Order-based reduction with multiple keys
        dsr = dat.groupby(['order_id', 'message_type']).first()
        self.assertEqual(dsr.shape, (6, 4))
        self.assertArrayEqual(dsr.order_id, [1, 1, 1, 2, 2, 2])
        self.assertArrayEqual(
            dsr.message_type, ['CANCEL', 'CREATE', 'RUN', 'CANCEL', 'CREATE', 'RUN']
        )
        self.assertArrayEqual(dsr.second, [120, 50, 70, 97, 72, 90])
        self.assertArrayEqual(dsr.shares, [0, 0, 200, 0, 0, 100])

        # On a subset of columns
        gb = dat.groupby('order_id')
        dsr = gb['shares'].sum()
        self.assertEqual(dsr.shape, (2, 2))
        self.assertArrayEqual(dsr.shares, [800, 1100])

        # Accumulating function
        dsr = dat.groupby('order_id').cumsum()
        self.assertEqual(dsr.shape, (12, 2))
        self.assertArrayEqual(
            dsr.shares, [0, 200, 0, 700, 100, 500, 800, 500, 800, 950, 1100, 800]
        )

        # return_all has no effect with accumulating functions
        # 8/23/2018 SJK - changed behavior so return all shows the keys
        dsr = dat.groupby('order_id', return_all=True).cumsum()
        self.assertEqual(dsr.shape, (12, 3))

        # Add cum_shares back to a dataset
        dat['cum_shares'] = dat.groupby('order_id').shares.cumsum().shares
        self.assertEqual(dat.shape, (12, 5))
        self.assertArrayEqual(dat.cum_shares, gb.shares.cumsum().shares)

        # On a subset of columns
        dsr = dat.groupby('order_id')[['shares', 'second']].cumsum()
        self.assertEqual(dsr.shape, (12, 2))
        self.assertArrayEqual(
            dsr.shares, [0, 200, 0, 700, 100, 500, 800, 500, 800, 950, 1100, 800]
        )
        self.assertArrayEqual(
            dsr.second, [50, 120, 72, 195, 162, 250, 290, 347, 445, 560, 676, 410]
        )

        # On a subset of columns with a filter
        f = FastArray(
            [
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
            ]
        )
        dsr = dat.groupby('order_id')[['shares', 'second']].cumsum(filter=f)
        self.assertEqual(dsr.shape, (12, 2))
        self.assertArrayEqual(
            dsr.shares, [0, 0, 0, 0, 100, 100, 100, 100, 400, 400, 550, 100]
        )
        self.assertArrayEqual(
            dsr.second, [50, 50, 72, 50, 162, 162, 145, 162, 260, 260, 376, 145]
        )

        # On shares and second with filter at groupby construction
        dsr = dat.groupby('order_id', filter=f)[['shares', 'second']].cumsum()
        inv = INVALID_DICT[dsr.shares[0].dtype.num]
        self.assertEqual(dsr.shape, (12, 2))
        self.assertArrayEqual(
            dsr.shares, [0, inv, 0, inv, 100, inv, 100, inv, 400, inv, 550, inv]
        )
        self.assertArrayEqual(
            dsr.second, [50, inv, 72, inv, 162, inv, 145, inv, 260, inv, 376, inv]
        )

        # Using agg function
        dsr = gb[['second', 'shares']].agg(['sum', 'mean'])
        self.assertEqual(dsr.shape, (2, 2))
        self.assertArrayEqual(dsr.Sum.second, [410, 676])
        self.assertArrayEqual(dsr.Sum.shares, [800, 1100])
        self.assertArrayAlmostEqual(dsr.Mean.second, [82.00, 96.57], places=2)
        self.assertArrayAlmostEqual(dsr.Mean.shares, [160.00, 157.14], places=2)

        # Check for issue when bracket indexing on groupby
        f = open(os.devnull, 'w')
        print(gb, file=f)
        f.close()
        dsr = gb[['second', 'shares']].agg(['sum', 'mean'])

        # Using different functions on different columns
        dsr = gb.agg({'second': 'sum', 'shares': ['max', 'mean']})
        self.assertEqual(dsr.shape, (2, 3))
        self.assertArrayEqual(dsr.Sum.second, [410, 676])
        self.assertArrayEqual(dsr.Max.shares, [500, 400])
        self.assertArrayAlmostEqual(dsr.Mean.shares, [160.00, 157.14], places=2)

        # Using numpy functions
        dsr = gb.agg({'second': np.sum, 'shares': [np.max, np.mean]})
        self.assertEqual(dsr.shape, (2, 3))
        self.assertArrayEqual(dsr.Sum.second, [410, 676])
        self.assertArrayEqual(dsr.Max.shares, [500, 400])
        self.assertArrayAlmostEqual(dsr.Mean.shares, [160.00, 157.14], places=2)

        # Alternate way to add to multiset
        gb = dat.groupby('order_id')
        ms = gb[['shares']].agg(['max', 'mean'])
        ms.Sum = gb[['second']].sum()
        self.assertEqual(ms.shape, (2, 3))
        self.assertArrayEqual(ms.Sum.second, [410, 676])
        self.assertArrayEqual(ms.Max.shares, [500, 400])
        self.assertArrayAlmostEqual(ms.Mean.shares, [160.00, 157.14], places=2)

    def test_iter(self):
        correct_keys = FastArray(['e', 'd', 'b', 'c', 'a'])
        correct_idx = [[0, 1, 4, 7], [2, 9], [3], [5, 6], [8]]
        str_arr = FastArray(['e', 'e', 'd', 'b', 'e', 'c', 'c', 'e', 'a', 'd'])

        gb = Dataset({'keycol': str_arr, 'idxcol': arange(10)})
        gb = gb.gb('keycol')
        for i, tup in enumerate(gb):
            self.assertEqual(tup[0], correct_keys[i])
            self.assertTrue(bool(np.all(tup[1].idxcol == correct_idx[i])))

    def test_shift(self):
        import pandas as pd

        ds = Dataset({'col_' + str(i): np.random.rand(30) for i in range(5)})
        ds.keycol = np.random.choice(['a', 'b', 'c'], 30)
        df = pd.DataFrame(ds.asdict())

        rt_result = ds.gb('keycol').shift(periods=-2).trim()
        pd_result = df.groupby('keycol').shift(periods=-2).dropna(axis='rows')

        for k, v in rt_result.items():
            self.assertTrue(bool(np.all(v == pd_result[k])))

        rt_result = ds.gb('keycol').shift(periods=3).trim()
        pd_result = df.groupby('keycol').shift(periods=3).dropna(axis='rows')

        for k, v in rt_result.items():
            self.assertTrue(bool(np.all(v == pd_result[k])))

    def test_diff(self):
        import pandas as pd

        ds = Dataset({'col_' + str(i): np.random.rand(10) for i in range(5)})
        ds.keycol = np.random.choice(['a', 'b', 'c'], 10)
        df = pd.DataFrame(ds.asdict())

        rt_result = ds.gb('keycol').rolling_diff()
        pd_result = df.groupby('keycol').diff()

        for k, v in rt_result.items():
            pdc = pd_result[k]
            pdcnan = isnan(pdc)
            self.assertTrue(bool(np.all(isnan(v) == pdcnan)), msg=f'{v} {pdc}')

            masked_valid_pd = isnotnan(pdc)
            masked_valid_rt = isnotnan(v)

            self.assertTrue(bool(np.all(masked_valid_pd == masked_valid_rt)))

    def test_pad(self):
        arrsize = 100
        numrows = 20
        ds = Dataset({'time': arange(arrsize * 1.0)})
        ds.data = np.random.randint(numrows, size=arrsize)
        ds.data2 = np.random.randint(numrows, size=arrsize)
        symbols = ['ZYGO', 'YHOO', 'FB', 'GOOG', 'IBM']
        ds.symbol2 = Cat(1 + ds.data, list('ABCDEFGHIJKLMNOPQRST'))
        ds.symbol = Cat(1 + arange(arrsize) % len(symbols), symbols)
        ds.time[[3, 4, 7]] = nan
        newds = ds.gb('symbol').pad()
        self.assertTrue(newds.time[7] == 2.00)
        self.assertTrue(newds.time[3] != newds.time[3])
        newds = ds.gb('symbol').backfill()
        self.assertTrue(newds.time[7] == 12.00)

        # see if we can pull a group
        newds = ds.gb('symbol').get_group('YHOO')
        self.assertTrue(np.all(newds.symbol == 'YHOO'))

    def test_transform(self):

        arrsize = 200
        numrows = 7

        ds = Dataset({'time': arange(arrsize * 1.0)})
        ds.data = np.random.randint(numrows, size=arrsize)
        ds.data2 = np.random.randint(numrows, size=arrsize)
        symbols = ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM']
        ds.symbol = Cat(1 + arange(arrsize) % len(symbols), symbols)
        newds = ds.gb('symbol')['data'].sum(transform=True)

        # removed from test since gbkeys not returned in transform
        # self.assertTrue(np.all(newds.symbol == ds.symbol))
        catds = ds.symbol.sum(ds.data, transform=True)
        self.assertTrue(np.all(newds[0] == catds[0]))
        # test showfilter
        catds = ds.symbol.sum(ds.data, showfilter=True, transform=True)
        self.assertTrue(np.all(newds[0] == catds[0]))

        # test diff
        result1 = ds.gb('symbol').apply_nonreduce(TypeRegister.FastArray.diff)
        result2 = ds.gb('symbol').diff()
        self.assertTrue(result1.equals(result2))


if __name__ == "__main__":
    tester = unittest.main()
