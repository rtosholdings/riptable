import unittest
import pytest
from riptable import *
from riptable.rt_numpy import QUANTILE_METHOD_NP_KW, gb_np_quantile

from numpy.testing import (
    assert_array_equal,
    assert_almost_equal,
    assert_array_almost_equal,
)


class GroupByOps_Test(unittest.TestCase):
    def test_categorical_values(self):
        vals = np.random.choice(["a", "b", "c", "d", "e"], 50)
        ds = Dataset({"vals": vals})
        # TODO what assertions need to be done on 'c' and 'gb'
        c = Categorical(vals, ordered=False)
        gb = ds.gb("vals")

        vals = np.random.choice(["a", "b", "c", "d", "e"], 50)
        ds = Dataset({"vals": vals})
        c = Categorical(vals)
        gb = ds.gb("vals", lex=True)
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = FA(vals, unicode=True)
        ds = Dataset({"vals": vals})
        c = Categorical(vals, ordered=False, unicode=True)
        gb = ds.gb("vals")
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = np.random.randint(0, 1000, 50)
        ds = Dataset({"vals": vals})
        c = Categorical(vals, ordered=False)
        gb = ds.gb("vals")
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 50)
        ds = Dataset({"vals": vals})
        c = Categorical(vals, ordered=False)
        gb = ds.gb("vals")
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = np.random.choice(["a", "b", "c", "d", "e"], 50)
        ds = Dataset({"vals": vals})
        c = Categorical(vals, ordered=False, sort_gb=False)
        gb = ds.gb("vals")
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = FA(vals, unicode=True)
        ds = Dataset({"vals": vals})
        c = Categorical(vals, ordered=False, unicode=True, sort_gb=False)
        gb = ds.gb("vals")
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = np.random.randint(0, 1000, 50)
        ds = Dataset({"vals": vals})
        c = Categorical(vals, ordered=False, sort_gb=False)
        gb = ds.gb("vals")
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 50)
        ds = Dataset({"vals": vals})
        c = Categorical(vals, ordered=False, sort_gb=False)
        gb = ds.gb("vals")
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        # 4/29/2019 - iFirstKey gets generated later, uses a new routine
        # need to set ordered to False to correctly match groupby results
        # groupby always uses first-occurence
        # or ordered=True groupby(lex=True)

        vals = np.random.choice(["a", "b", "c", "d", "e"], 50)
        ds = Dataset({"vals": vals})
        c = Categorical(vals, ordered=True)
        gb = ds.gb("vals", lex=True)
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = FA(vals, unicode=True)
        ds = Dataset({"vals": vals})
        c = Categorical(vals, unicode=True, ordered=True)
        gb = ds.gb("vals", lex=True)
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = np.random.randint(0, 1000, 50)
        ds = Dataset({"vals": vals})
        c = Categorical(vals, ordered=True)
        gb = ds.gb("vals", lex=True)
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 50)
        ds = Dataset({"vals": vals})
        c = Categorical(vals, ordered=True)
        gb = ds.gb("vals", lex=True)
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

    def test_categorical_values_cats(self):
        pass

    def test_categorical_codes_cats(self):
        pass

    def test_categorical_mapping(self):
        pass

    def test_categorical_multikey(self):
        pass

    def test_gb_userfunc(self):
        ds = Dataset({"strs": np.random.choice(["a", "b", "c"], 20)})
        ds.data = arange(20)
        # make sure np.quantile stlye works where second argument is passed
        ds.gb("strs").apply_nonreduce(np.quantile, 0.5)

    def test_gb_extra_data(self):
        ds = Dataset({"strs": np.random.choice(["a", "b", "c"], 20)})
        ds.col_AAA = ones(20)
        ds.col_BBB = np.random.rand(20)
        extra = arange(20)
        extra.set_name("extra")
        ds2 = ds.gb("strs").sum(extra)

        final_cols = ["strs", "col_AAA", "col_BBB", "extra"]
        self.assertTrue(bool(np.all(list(ds2) == final_cols)))

    def test_cat_add_dataset(self):
        ds = Dataset({"strs": np.random.choice(["a", "b", "c"], 20)})
        ds.col_AAA = ones(20)
        ds.col_BBB = np.random.rand(20)
        c = Categorical(ds.strs)
        ds2 = c.sum(dataset=ds)
        final_cols = ["strs", "col_AAA", "col_BBB"]
        self.assertTrue(bool(np.all(list(ds2) == final_cols)))

        extra = arange(20)
        extra.set_name("extra")
        ds3 = c.sum(extra, dataset=ds)
        final_cols = ["strs", "col_AAA", "col_BBB", "extra"]
        self.assertTrue(bool(np.all(list(ds3) == final_cols)))

    def test_warn_conflicting_data(self):
        ds = Dataset({"strs": np.random.choice(["a", "b", "c"], 20)})
        ds.col_AAA = ones(20)
        ds.col_BBB = np.random.rand(20)
        c = Categorical(np.random.choice(["a", "b", "c"], 20))
        with self.assertWarns(UserWarning):
            ds2 = c.sum(ds, dataset=ds)

    def test_error_no_data(self):
        c = Categorical(np.random.choice(["a", "b", "c"], 20))
        with self.assertRaises(ValueError):
            _ = c.sum()

        ds = Dataset({"strs": np.random.choice(["a", "b", "c"], 20)})
        ds.col_AAA = ones(20)
        ds.col_BBB = np.random.rand(20)
        gb = ds.gb("strs")
        gb._dataset = None
        with self.assertRaises(ValueError):
            _ = gb.sum()

    def test_ema_funcs(self):
        ds = Dataset({"test": arange(10), "group2": arange(10) % 3})

        gb = ds.gb("group2")
        c = Categorical(ds.group2)

        # decay_rate now required to be filled in
        gb_result = gb.ema_normal(time=arange(10), decay_rate=1.0)
        c_result = c.ema_normal(ds.test, time=arange(10), decay_rate=1.0)

        for k, col in gb_result.items():
            self.assertTrue(bool(np.all(col == c_result[k])))

    def test_agg_funcs(self):
        ds = Dataset({"col_" + str(i): np.random.rand(20) for i in range(10)})
        ds.keycol = np.random.choice(["a", "b", "c"], 20)
        c = Categorical(ds.keycol)
        flt = np.random.rand(20).view(FA)
        flt.set_name("flt")

        gb_result = ds.gb("keycol").agg({"col_0": ["sum", np.mean], "col_5": ["min", np.var], "flt": "max"}, flt)
        c_result = c.agg(
            {"col_0": ["sum", np.mean], "col_5": ["min", np.var], "flt": "max"},
            flt,
            dataset=ds[:-1],
        )

        for dsname, d_total in gb_result.items():
            c_total = c_result[dsname]
            for k, v in d_total.items():
                self.assertTrue(bool(np.all(c_total[k] == v)))

    def test_agg_funcs_np(self):
        c = Categorical(np.random.choice(["a", "b", "c"], 20))

        rt_result = c.agg(["min", "max"], arange(20))
        np_result = c.agg([np.amin, np.amax], arange(20))

        for dsname, d_total in rt_result.items():
            c_total = np_result[dsname]
            for k, v in d_total.items():
                self.assertTrue(bool(np.all(c_total[k] == v)))

    def test_agg_funcs_error(self):
        c = Categorical(np.random.choice(["a", "b", "c"], 20))
        with self.assertRaises(ValueError):
            _ = c.agg({"badname": "min"}, arange(20))

        with self.assertRaises(ValueError):
            _ = c.agg(["min", "max", "badfunc"], arange(20))

    def test_groups_order(self):
        strings = np.array(["b", "a", "a", "z", "y", "b", "c"])
        ds = Dataset(
            {
                "col1": Categorical(strings, ordered=True),
                "col2": Categorical(strings, ordered=False),
                "col3": strings,
            }
        )
        d1 = ds.col1.groups
        d2 = ds.col2.groups
        d3 = ds.gb("col3").groups
        d4 = ds.gbu("col3").groups

        for colname, idx in d1.items():
            for d in (d2, d3, d4):
                self.assertTrue(bool(np.all(d[colname] == idx)))


class TestGoupByOps_functions:
    @pytest.mark.parametrize("N", [1, 2, 5, 100, 3467, 60_000, 120_000])
    @pytest.mark.parametrize("N_cat", [1, 2, 3, 10, 200, 5_000, 20_000])
    @pytest.mark.parametrize("nan_fraction", [0.0, 0.1, 0.3, 0.7, 0.9, 0.9999999, 1.0])
    @pytest.mark.parametrize("rep", [1, 2, 3])
    def test_groupbyops_all_min_max(self, N, N_cat, nan_fraction, rep):
        np.random.seed((N * N_cat * rep * int(nan_fraction * 1000)) % 10_000)
        ds = Dataset({"categ": np.random.randint(0, N_cat, size=N)})
        ds["categ"] = Cat(ds["categ"])
        categories = ds.categ._grouping.uniquelist[0]
        ds.data = np.random.normal(np.random.uniform(-10_000, 10_000), np.random.uniform(0.00, 10_000), size=N)
        ds.data[np.random.random(size=N) < nan_fraction] = np.nan

        ds.int_data = np.random.randint(-10_000, 10_000, size=N)

        int_nan_idxs = np.random.random(size=N) < nan_fraction
        ds.int_data[int_nan_idxs] = np.nan

        # transform into float and then into np.array to have proper np.nans there
        np_int_data_copy = ds.int_data.astype(np.float64)._np
        np_int_data_copy[int_nan_idxs] = np.nan

        cols = ["data", "int_data"]

        dsmin = ds.categ.min(ds)
        dsmax = ds.categ.max(ds)

        dsnanmin = ds.categ.nanmin(ds)
        dsnanmax = ds.categ.nanmax(ds)

        for column in cols:
            # check that the result on the whole dataset equals to the results on
            # individual columns
            assert_array_equal(dsmin[column], ds.categ.min(ds[column])[column])
            assert_array_equal(dsmax[column], ds.categ.max(ds[column])[column])

            assert_array_equal(dsnanmin[column], ds.categ.nanmin(ds[column])[column])
            assert_array_equal(dsnanmax[column], ds.categ.nanmax(ds[column])[column])

        def err_msg():
            return f"""
N = {N}
N_cat = {N_cat}
nan_frac = {nan_fraction}
column = {column}
category = {category}"""

        for column in cols:
            dtype = ds[column].dtype
            for category in np.random.choice(categories, min(len(categories), 60)):
                data_res_min = dsmin[column][dsmin.categ == category]
                data_res_max = dsmax[column][dsmax.categ == category]
                data_res_nanmin = dsnanmin[column][dsnanmin.categ == category]
                data_res_nanmax = dsnanmax[column][dsnanmax.categ == category]
                if dtype in [np.int32, np.int64]:
                    curr_slice = np_int_data_copy[ds.categ == category]
                else:
                    curr_slice = ds.filter(ds.categ == category)[column]._np
                data_res_min_np = np.min(curr_slice)
                data_res_max_np = np.max(curr_slice)
                data_res_nanmin_np = np.nanmin(curr_slice)
                data_res_nanmax_np = np.nanmax(curr_slice)
                # make answer float to transform integer invalids into np.nans for comparison
                assert_almost_equal(data_res_min.astype(float)[0], data_res_min_np, err_msg=err_msg())
                assert_almost_equal(data_res_max.astype(float)[0], data_res_max_np, err_msg=err_msg())
                assert_almost_equal(data_res_nanmin.astype(float)[0], data_res_nanmin_np, err_msg=err_msg())
                assert_almost_equal(data_res_nanmax.astype(float)[0], data_res_nanmax_np, err_msg=err_msg())

    @pytest.mark.parametrize("N", [1, 10, 123, 4_861, 89_931])
    @pytest.mark.parametrize("N_cat", [1, 3, 9, 391])
    @pytest.mark.parametrize("q", [0.0, 0.1298753, 0.5, 0.78, 1.0])
    @pytest.mark.parametrize("nan_fraction", [0.0, 0.15, 0.65, 1.0])
    @pytest.mark.parametrize("dtype", [np.int32, np.float64])
    def test_groupbyops_quantile(self, N, N_cat, q, nan_fraction, dtype):
        np.random.seed((N * N_cat * int(q * 1000) * int(nan_fraction * 1000)) % 10_000)
        kwargs = {QUANTILE_METHOD_NP_KW: "midpoint"}

        def err_msg():
            return f"""
q = {q}
N = {N}
N_cat = {N_cat}
nan_fraction = {nan_fraction}
dtype = {dtype}"""

        ds = Dataset({"categ": np.random.randint(0, N_cat, size=N)})
        categories = np.unique(ds.categ)
        ds["categ"] = Cat(ds["categ"])

        ds.data = np.random.normal(np.random.uniform(-10_000, 10_000), np.random.uniform(0.00, 10_000), size=N).astype(
            dtype
        )
        ds.data2 = np.random.normal(np.random.uniform(-10_000, 10_000), np.random.uniform(0.00, 10_000), size=N).astype(
            dtype
        )

        nan_idxs = np.random.random(size=N) < nan_fraction
        # need to convert to float for numpy to understand nans.
        # np_data_copy will only be used for numpy operations for checks.
        np_data_copy = ds.data.copy().astype(np.float64)
        np_data_copy[nan_idxs] = np.nan
        ds.data[nan_idxs] = np.nan

        data_quant = ds.categ.quantile(ds.data, q=q)
        data2_quant = ds.categ.quantile(ds.data2, q=q)
        both_quant = ds.categ.quantile((ds.data, ds.data2), q=q)

        assert_array_almost_equal(data_quant.data, both_quant.data, err_msg=err_msg())
        assert_array_almost_equal(data2_quant.data2, both_quant.data2, err_msg=err_msg())

        for category in np.random.choice(categories, min(len(categories), 15)):
            curr_filter = ds.categ == category
            curr_slice = ds.filter(curr_filter)
            data_res = data_quant.filter(data_quant.categ == category).data
            # converting this to float for nanmax and nanmean.
            if q * (1 - q) == 0:
                data_res = data_res.astype(np.float64)
            data_res_np = np.quantile(np_data_copy[curr_filter], q=q, **kwargs)
            assert_almost_equal(data_res, data_res_np, err_msg=err_msg())
            data2_res = data2_quant.filter(data2_quant.categ == category).data2
            data2_res_np = np.quantile(curr_slice.data2, q=q, **kwargs)
            assert_almost_equal(data2_res, data2_res_np, err_msg=err_msg())

        both_percentiles = ds.categ.percentile((ds.data, ds.data2), q=q * 100)
        assert_array_almost_equal(both_quant.data, both_percentiles.data, err_msg=err_msg())
        assert_array_almost_equal(both_quant.data2, both_percentiles.data2, err_msg=err_msg())

        # nanquantile
        data_nanquant = ds.categ.nanquantile(ds.data, q=q)
        data2_nanquant = ds.categ.nanquantile(ds.data2, q=q)
        both_nanquant = ds.categ.nanquantile((ds.data, ds.data2), q=q)

        assert_array_almost_equal(data2_nanquant.data2, data2_quant.data2, err_msg=err_msg())
        assert_array_almost_equal(data_nanquant.data, both_nanquant.data, err_msg=err_msg())
        assert_array_almost_equal(data2_nanquant.data2, both_nanquant.data2, err_msg=err_msg())

        for category in np.random.choice(categories, min(len(categories), 15)):
            curr_filter = ds.categ == category
            curr_slice = ds.filter(curr_filter)
            data_res = data_nanquant.filter(data_nanquant.categ == category).data
            # converting this to float for nanmax and nanmean.
            if q * (1 - q) == 0:
                data_res = data_res.astype(np.float64)
            data_res_np = np.nanquantile(np_data_copy[curr_filter], q=q, **kwargs)
            assert_almost_equal(data_res, data_res_np, err_msg=err_msg())
            data2_res = data2_nanquant.filter(data2_nanquant.categ == category).data2
            data2_res_np = np.nanquantile(curr_slice.data2, q=q, **kwargs)
            assert_almost_equal(data2_res, data2_res_np, err_msg=err_msg())

        both_nanpercentiles = ds.categ.nanpercentile((ds.data, ds.data2), q=q * 100)
        assert_array_almost_equal(both_nanquant.data, both_nanpercentiles.data, err_msg=err_msg())
        assert_array_almost_equal(both_nanquant.data2, both_nanpercentiles.data2, err_msg=err_msg())

    @pytest.mark.parametrize("N", [1, 5_678, 70_000])
    @pytest.mark.parametrize("N_cat", [1, 3, 9, 40])
    @pytest.mark.parametrize("q", [0.0, 0.3, 0.5, 0.7, 1.0])
    @pytest.mark.parametrize("nan_fraction", [0.0, 0.3])
    @pytest.mark.parametrize("inf_fraction", [0.0, 0.4, 1.0])
    @pytest.mark.parametrize("minus_inf_fraction", [0.0, 0.4, 1.0])
    def test_groupbyops_quantile_infs(self, N, N_cat, q, nan_fraction, inf_fraction, minus_inf_fraction):
        if nan_fraction + inf_fraction + minus_inf_fraction > 1:
            return

        def err_msg():
            return f"""
q = {q}
N = {N}
N_cat = {N_cat}
nan_fraction = {nan_fraction}
inf_fraction = {inf_fraction}
minus_inf_fraction = {minus_inf_fraction}"""

        np.random.seed((N * N_cat * int(q * 1000) * int(nan_fraction * 1000)) % 10_000)
        ds = Dataset({"categ": np.random.randint(0, N_cat, size=N)})
        categories = np.unique(ds.categ)
        ds["categ"] = Cat(ds["categ"])

        # infs and integers don't work normally, only check floats
        ds.data = np.random.normal(np.random.uniform(-10_000, 10_000), np.random.uniform(0.00, 10_000), size=N)
        ds.data2 = np.random.normal(np.random.uniform(-10_000, 10_000), np.random.uniform(0.00, 10_000), size=N)

        # notice that np.nanquantile doesn't handle infs properly
        # use gb_np_quantile for comparison

        rand_idxs = np.random.random(size=N)
        ds.data[rand_idxs < (nan_fraction + inf_fraction + minus_inf_fraction)] = -np.inf
        ds.data[rand_idxs < (nan_fraction + inf_fraction)] = np.inf
        ds.data[rand_idxs < (nan_fraction)] = np.nan

        data_quant = ds.categ.quantile(ds.data, q=q)
        data2_quant = ds.categ.quantile(ds.data2, q=q)
        both_quant = ds.categ.quantile((ds.data, ds.data2), q=q)

        assert_array_almost_equal(data_quant.data, both_quant.data, err_msg=err_msg())
        assert_array_almost_equal(data2_quant.data2, both_quant.data2, err_msg=err_msg())

        for category in np.random.choice(categories, min(len(categories), 15)):
            curr_filter = ds.categ == category
            curr_slice = ds.filter(curr_filter)
            data_res = data_quant.filter(data_quant.categ == category).data
            data_res_np = gb_np_quantile(ds.data[curr_filter], q=q, is_nan_function=False)
            assert_almost_equal(data_res, data_res_np, err_msg=err_msg())
            data2_res = data2_quant.filter(data2_quant.categ == category).data2
            data2_res_np = gb_np_quantile(curr_slice.data2, q=q, is_nan_function=False)
            assert_almost_equal(data2_res, data2_res_np, err_msg=err_msg())

        both_percentiles = ds.categ.percentile((ds.data, ds.data2), q=q * 100)
        assert_array_almost_equal(both_quant.data, both_percentiles.data, err_msg=err_msg())
        assert_array_almost_equal(both_quant.data2, both_percentiles.data2, err_msg=err_msg())

        # nanquantile
        data_nanquant = ds.categ.nanquantile(ds.data, q=q)
        data2_nanquant = ds.categ.nanquantile(ds.data2, q=q)
        both_nanquant = ds.categ.nanquantile((ds.data, ds.data2), q=q)

        assert_array_almost_equal(data2_nanquant.data2, data2_quant.data2, err_msg=err_msg())
        assert_array_almost_equal(data_nanquant.data, both_nanquant.data, err_msg=err_msg())
        assert_array_almost_equal(data2_nanquant.data2, both_nanquant.data2, err_msg=err_msg())

        for category in np.random.choice(categories, min(len(categories), 15)):
            curr_filter = ds.categ == category
            curr_slice = ds.filter(curr_filter)
            data_res = data_nanquant.filter(data_nanquant.categ == category).data
            data_res_np = gb_np_quantile(ds.data[curr_filter], q=q, is_nan_function=True)
            assert_almost_equal(data_res, data_res_np, err_msg=err_msg())
            data2_res = data2_nanquant.filter(data2_nanquant.categ == category).data2
            data2_res_np = gb_np_quantile(curr_slice.data2, q=q, is_nan_function=True)
            assert_almost_equal(data2_res, data2_res_np, err_msg=err_msg())

        both_nanpercentiles = ds.categ.nanpercentile((ds.data, ds.data2), q=q * 100)
        assert_array_almost_equal(both_nanquant.data, both_nanpercentiles.data, err_msg=err_msg())
        assert_array_almost_equal(both_nanquant.data2, both_nanpercentiles.data2, err_msg=err_msg())

    @pytest.mark.parametrize("N", [1, 300])
    @pytest.mark.parametrize("N_cat", [1, 5, 100])
    @pytest.mark.parametrize("q", [[0.0, 0.34, 0.81], [0.0, 0.5, 1.0, 0.2], [0.123, 0.12345, 0.1], 0.734])
    def test_groupbyops_quantile_multi_params(self, N, N_cat, q):
        def err_msg():
            return f"""
q = {q}
N = {N}
N_cat = {N_cat}"""

        np.random.seed((N * N_cat * int(np.sum(q) * 1000) % 10_000))
        ds = Dataset({"categ": np.random.randint(0, N_cat, size=N)})
        categories = np.unique(ds.categ)
        ds["categ"] = Cat(ds["categ"])

        ds.data = np.random.normal(np.random.uniform(-10_000, 10_000), np.random.uniform(0.00, 10_000), size=N)
        ds.data2 = np.random.normal(np.random.uniform(-10_000, 10_000), np.random.uniform(0.00, 10_000), size=N)
        ds.col_0 = np.random.normal(np.random.uniform(-10_000, 10_000), np.random.uniform(0.00, 10_000), size=N)
        ds.col_1 = np.random.normal(np.random.uniform(-10_000, 10_000), np.random.uniform(0.00, 10_000), size=N)
        ds.int_data = np.random.randint(-10_000, 10_000, size=N)
        ds.int_data2 = np.random.randint(0, 100, size=N)

        cols = ["data", "data2", "col_0", "col_1", "int_data", "int_data2"]

        quant_ds = ds.categ.quantile(ds, q=q)
        quant_sep = ds.categ.quantile((ds.data, ds.data2, ds.col_0, ds.col_1, ds.int_data, ds.int_data2), q=q)
        percentile_sep = ds.categ.percentile(
            (ds.data, ds.data2, ds.col_0, ds.col_1, ds.int_data, ds.int_data2), q=np.array(q) * 100
        )
        nanpercentile_sep = ds.categ.nanpercentile(
            (ds.data, ds.data2, ds.col_0, ds.col_1, ds.int_data, ds.int_data2), q=np.array(q) * 100
        )
        nanquantile_sep = ds.categ.nanquantile((ds.data, ds.data2, ds.col_0, ds.col_1, ds.int_data, ds.int_data2), q=q)

        assert_array_equal(quant_ds.imatrix_make(), quant_sep.imatrix_make())
        assert_array_equal(quant_ds.imatrix_make(), percentile_sep.imatrix_make())
        assert_array_equal(quant_ds.imatrix_make(), nanpercentile_sep.imatrix_make())
        assert_array_equal(quant_ds.imatrix_make(), nanquantile_sep.imatrix_make())

        q_cols = quant_ds.keys()

        if np.isscalar(q):
            q = [q]
            q_cols = cols

        # columns are ordered like [col + quantile for quantile in q for col in cols] if len(q) > 1

        for i_col, column in enumerate(cols):
            for i_q, quantile in enumerate(q):
                quant_col = q_cols[i_col * len(q) + i_q]
                for category in np.random.choice(categories, min(len(categories), 15)):
                    curr_filter = ds.categ == category
                    curr_slice = ds.filter(curr_filter)
                    data_res = quant_ds[quant_col][quant_ds.categ == category]
                    data_res_np = gb_np_quantile(curr_slice[column], q=quantile, is_nan_function=False)
                    assert_almost_equal(data_res, data_res_np, err_msg=err_msg() + f"\n{column} {quant_col}\n{q_cols}")

        # multi-key

        ds.cat2 = np.random.randint(0, N_cat, size=N)
        del ds.categ
        ds.categ = np.random.randint(0, N_cat, size=N)
        mk_cat = Cat([ds.categ, ds.cat2])
        categories = np.unique(mk_cat._fa)
        fa_cat_aligned = mk_cat.first(mk_cat._fa)["col_0"]

        quant_ds = mk_cat.quantile(ds, q=q)
        quant_sep = mk_cat.quantile((ds.data, ds.data2, ds.col_0, ds.col_1, ds.int_data, ds.int_data2), q=q)
        percentile_sep = mk_cat.percentile(
            (ds.data, ds.data2, ds.col_0, ds.col_1, ds.int_data, ds.int_data2), q=np.array(q) * 100
        )
        nanpercentile_sep = mk_cat.nanpercentile(
            (ds.data, ds.data2, ds.col_0, ds.col_1, ds.int_data, ds.int_data2), q=np.array(q) * 100
        )
        nanquantile_sep = mk_cat.nanquantile((ds.data, ds.data2, ds.col_0, ds.col_1, ds.int_data, ds.int_data2), q=q)

        assert_array_almost_equal(quant_ds.imatrix_make(), quant_sep.imatrix_make())
        assert_array_almost_equal(quant_ds.imatrix_make(), percentile_sep.imatrix_make())
        assert_array_almost_equal(quant_ds.imatrix_make(), nanpercentile_sep.imatrix_make())
        assert_array_almost_equal(quant_ds.imatrix_make(), nanquantile_sep.imatrix_make())

        q_cols = quant_ds.keys()

        if np.isscalar(q):
            q = [q]
            q_cols = cols

        # columns are ordered like [col + quantile for quantile in q for col in cols] if len(q) > 1

        for i_col, column in enumerate(cols):
            for i_q, quantile in enumerate(q):
                quant_col = q_cols[i_col * len(q) + i_q]
                for category in np.random.choice(categories, min(len(categories), 15)):
                    curr_filter = mk_cat._fa == category
                    curr_slice = ds.filter(curr_filter)
                    data_res = quant_ds[quant_col][fa_cat_aligned == category]
                    data_res_np = gb_np_quantile(curr_slice[column], q=quantile, is_nan_function=False)
                    assert_array_almost_equal(
                        data_res, data_res_np, err_msg=err_msg() + f"\n{column} {quant_col}\n{q_cols}"
                    )


if __name__ == "__main__":
    tester = unittest.main()
