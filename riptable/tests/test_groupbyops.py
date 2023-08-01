import unittest
import pytest
import itertools
import numpy as np
from numpy.random import default_rng

from riptable import *
import riptable as rt
from riptable.rt_numpy import QUANTILE_METHOD_NP_KW, gb_np_quantile, np_rolling_nanquantile
from riptable.tests.utils import get_rc_version, parse_version

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


def quantile_params_generator(dtypes, max_N, windowed, with_cats, with_nans, with_infs, seed, scrunity_level=1):
    rng = default_rng(seed)

    for dtype in dtypes:
        # N = 1 case
        N = 1
        N_cat = 1
        window = 1
        for quantile in [round(x, 4) for x in rng.random(3 * scrunity_level)]:
            curr_frac_options = [(0, 0, 0)]
            if with_nans:
                curr_frac_options += [(1, 0, 0)]
            if with_infs:
                curr_frac_options += [(0, 1, 0), (0, 0, 1)]
            for nan_fraction, inf_fraction, minus_inf_fraction in curr_frac_options:
                yield N, N_cat, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype

        frac_lists = [
            [0.0, 1] + [round(x, 4) for x in rng.random(1 * scrunity_level)],
            [0.0, 1] + [round(x, 4) for x in rng.random(1 * scrunity_level)],
            [0.0, 1] + [round(x, 4) for x in rng.random(1 * scrunity_level)],
        ]

        frac_options = [element for element in itertools.product(*frac_lists) if sum(element) <= 1]
        frac_options_only_nans = [(x, 0, 0) for x in frac_lists[0]]

        curr_frac_options = [(0, 0, 0)]
        if with_nans:
            curr_frac_options = frac_options_only_nans
        if with_infs:
            curr_frac_options = frac_options

        if with_cats:
            # N_cat = 1 case
            N = rng.integers(400, 600)
            N_cat = 1
            window = rng.integers(100, 200)
            for quantile in [round(x, 3) for x in rng.random(2 * scrunity_level)]:
                for nan_fraction, inf_fraction, minus_inf_fraction in curr_frac_options:
                    yield N, N_cat, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype

        # window = 1 case
        N = rng.integers(400, 600)
        N_cat = rng.integers(30, 50)
        window = 1
        for quantile in [round(x, 3) for x in rng.random(2 * scrunity_level)]:
            for nan_fraction, inf_fraction, minus_inf_fraction in curr_frac_options:
                yield N, N_cat, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype

        # window > N case
        N = rng.integers(400, 600)
        N_cat = rng.integers(30, 50)
        window = rng.integers(700, 900)
        for quantile in [round(x, 3) for x in rng.random(1 * scrunity_level)]:
            for nan_fraction, inf_fraction, minus_inf_fraction in curr_frac_options:
                yield N, N_cat, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype

        # quantile = 0., 1. cases
        N = rng.integers(400, 600)
        N_cat = rng.integers(10, 40)
        window_list = [1, 3, rng.integers(10, 40)]
        if not windowed:
            window_list = [1]
        for window in window_list:
            for quantile in [0.0, 1.0]:
                for nan_fraction, inf_fraction, minus_inf_fraction in curr_frac_options:
                    yield N, N_cat, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype

        for N in (
            rng.integers(20, 1000, size=3 * scrunity_level).tolist()
            + rng.integers(10_000, 50_000, size=2 * scrunity_level).tolist()
        ):
            if N > max_N:
                continue
            window_list = rng.integers(15, N, size=1 * scrunity_level)
            if not windowed:
                window_list = [1]
            for window in window_list:
                cat_list = rng.integers(2, N, size=1 * scrunity_level)
                if not with_cats:
                    cat_list = [1]
                for N_cat in cat_list:
                    for quantile in [round(x, 3) for x in rng.random(2 * scrunity_level)]:
                        for nan_fraction, inf_fraction, minus_inf_fraction in curr_frac_options:
                            yield N, N_cat, window, quantile, nan_fraction, inf_fraction, minus_inf_fraction, dtype


def fills_params_generator():
    float_types = [np.float32, np.float64]
    integral_types = [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ]

    all_types = float_types + integral_types

    common_types = [np.int64, np.float64]

    # N = 1 case
    N = 1
    N_cat = 1
    limit = 0
    fill_values = [None, 0, 15]
    nan_fractions = [0.5, 0.95]
    types = common_types

    for fill_val in fill_values:
        for nan_fraction in nan_fractions:
            yield N, N_cat, nan_fraction, types, limit, fill_val

    # types
    N = 1_000
    N_cat = 200
    limits = [0, 2]
    nan_fraction = 0.5
    types = all_types
    fill_val = None
    for limit in limits:
        yield N, N_cat, nan_fraction, types, limit, fill_val

    # limits
    N = 1_020
    N_cat = 210
    limits = [0, 1, 2, 4, 6, 10]
    nan_fractions = [0.5, 0.95]
    types = common_types
    fill_val = None
    for limit in limits:
        for nan_fraction in nan_fractions:
            yield N, N_cat, nan_fraction, types, limit, fill_val

    # fill_vals
    N = 965
    N_cat = 300
    limits = [0, 2]
    nan_fraction = 0.5
    types = common_types
    fill_values = [None, 0, -15, 100]
    for limit in limits:
        for fill_val in fill_values:
            yield N, N_cat, nan_fraction, types, limit, fill_val


class TestGoupByOps_functions:
    @pytest.mark.parametrize("N", [1, 2, 5, 100, 3467, 60_000, 120_000])
    @pytest.mark.parametrize("N_cat", [1, 2, 3, 10, 1_000, 20_000])
    @pytest.mark.parametrize("nan_fraction", [0.0, 0.1, 0.3, 0.7, 0.9, 0.9999999, 1.0])
    @pytest.mark.parametrize("rep", [1])
    def test_groupbyops_all_min_max(self, N, N_cat, nan_fraction, rep):
        this_seed = (N * N_cat * rep * int(nan_fraction * 1000)) % 10_000
        rng = default_rng(this_seed)

        ds = Dataset({"categ": rng.integers(0, N_cat, size=N)})
        ds["categ"] = Cat(ds["categ"])
        categories = ds.categ._grouping.uniquelist[0]
        ds.data = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)
        ds.data[rng.random(size=N) < nan_fraction] = np.nan

        ds.int_data = rng.integers(-10_000, 10_000, size=N)

        int_nan_idxs = rng.random(size=N) < nan_fraction
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
nan_fraction = {nan_fraction}
column = {column}
category = {category}"""

        for column in cols:
            dtype = ds[column].dtype
            for category in rng.choice(categories, min(len(categories), 60)):
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

    @pytest.mark.skipif(get_rc_version() < parse_version("1.13.2a"), reason="cummin/max not implemented", strict=False)
    @pytest.mark.parametrize("N", [1, 5, 97, 2643])
    @pytest.mark.parametrize("N_cat", [1, 10, 45])
    @pytest.mark.parametrize("nan_fraction", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("skipna", [True, False])
    def test_groupbyops_cum_min_max(self, N, N_cat, nan_fraction, skipna):
        this_seed = (N * N_cat * int(nan_fraction * 1000)) % 10_000
        rng = default_rng(this_seed)

        ds = Dataset({"categ": rng.integers(0, N_cat, size=N)})
        ds["categ"] = Cat(ds["categ"])
        categories = ds.categ._grouping.uniquelist[0]

        float_types = [np.float32, np.float64]
        integral_types = [
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
        ]
        bool_types = [np.bool_]

        all_types = float_types + integral_types + bool_types

        type_dict = {np.dtype(t).char: t for t in all_types}

        for char, data_type in type_dict.items():
            if data_type in float_types:
                ds[f"data_{char}"] = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)
                ds[f"data_{char}"] = ds[f"data_{char}"].astype(data_type)

                if nan_fraction > 0:
                    ds[f"data_{char}"][rng.random(size=N) < nan_fraction / 3] = np.nan
                    ds[f"data_{char}"][rng.random(size=N) < nan_fraction / 3] = np.inf
                    ds[f"data_{char}"][rng.random(size=N) < nan_fraction / 3] = -np.inf

            elif data_type in integral_types:
                ds[f"data_{char}"] = rng.integers(-10_000, 10_000, size=N)
                ds[f"data_{char}"] = ds[f"data_{char}"].astype(data_type)
                ds[f"data_{char}"][rng.random(size=N) < nan_fraction] = np.nan
            else:
                ds[f"data_{char}"] = rng.choice([True, False], size=N)
                ds[f"data_{char}"] = ds[f"data_{char}"].astype(data_type)

        dscummin = ds.categ.cummin(ds, skipna=skipna)
        dscummax = ds.categ.cummax(ds, skipna=skipna)

        cols = [c for c in ds if c.startswith("data_")]  # ["data_d", "data_l"]  # np.float64 and np.int64

        main_cols = ["data_d", "data_l", "data_?"]

        def err_msg():
            return f"""
N = {N}
N_cat = {N_cat}
nan_fraction = {nan_fraction}
column = {column}
category = {category}"""

        for column in cols:
            assert dscummin[column].dtype == ds[column].dtype
            assert dscummax[column].dtype == ds[column].dtype
            # if column not in main_cols:
            #     continue
            for category in rng.choice(categories, min(len(categories), 3)):
                data_res_cummin = dscummin[column][ds.categ == category]
                data_res_cummax = dscummax[column][ds.categ == category]
                curr_slice = ds.filter(ds.categ == category)[column].astype(float)._np
                if skipna:
                    data_res_cummin_np = np.fmin.accumulate(curr_slice)
                    data_res_cummax_np = np.fmax.accumulate(curr_slice)
                else:
                    data_res_cummin_np = np.minimum.accumulate(curr_slice)
                    data_res_cummax_np = np.maximum.accumulate(curr_slice)
                # make answer float to transform integer invalids into np.nans for comparison
                assert_array_almost_equal(data_res_cummin.astype(float), data_res_cummin_np, err_msg=err_msg())
                assert_array_almost_equal(data_res_cummax.astype(float), data_res_cummax_np, err_msg=err_msg())

    @pytest.mark.parametrize(
        "N, N_cat, window, q, nan_fraction, inf_fraction, minus_inf_fraction, dtype",
        quantile_params_generator(
            [np.int32, np.float64],
            max_N=100_000,
            windowed=False,
            with_cats=True,
            with_nans=True,
            with_infs=False,
            seed=100,
        ),
    )
    def test_groupbyops_quantile(self, N, N_cat, window, q, nan_fraction, inf_fraction, minus_inf_fraction, dtype):
        this_seed = (N * N_cat * int(q * 1000) * int(nan_fraction * 1000)) % 10_000
        rng = default_rng(this_seed)
        kwargs = {QUANTILE_METHOD_NP_KW: "midpoint"}

        def err_msg():
            return f"""
q = {q}
N = {N}
N_cat = {N_cat}
nan_fraction = {nan_fraction}
dtype = {dtype}"""

        ds = Dataset({"categ": rng.integers(0, N_cat, size=N)})
        categories = np.unique(ds.categ)
        ds["categ"] = Cat(ds["categ"])

        ds.data = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N).astype(dtype)
        ds.data2 = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N).astype(dtype)

        nan_idxs = rng.random(size=N) < nan_fraction
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

        for category in rng.choice(categories, min(len(categories), 15)):
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

        for category in rng.choice(categories, min(len(categories), 15)):
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

    @pytest.mark.parametrize(
        "N, N_cat, window, q, nan_fraction, inf_fraction, minus_inf_fraction, dtype",
        quantile_params_generator(
            [np.float64], max_N=100_000, windowed=False, with_cats=True, with_nans=True, with_infs=True, seed=200
        ),
    )
    def test_groupbyops_quantile_infs(self, N, N_cat, window, q, nan_fraction, inf_fraction, minus_inf_fraction, dtype):
        def err_msg():
            return f"""
q = {q}
N = {N}
N_cat = {N_cat}
nan_fraction = {nan_fraction}
inf_fraction = {inf_fraction}
minus_inf_fraction = {minus_inf_fraction}"""

        this_seed = (N * N_cat * int(q * 1000) * int(nan_fraction * 1000)) % 10_000
        rng = default_rng(this_seed)
        ds = Dataset({"categ": rng.integers(0, N_cat, size=N)})
        categories = np.unique(ds.categ)
        ds["categ"] = Cat(ds["categ"])

        # infs and integers don't work normally, only check floats
        ds.data = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)
        ds.data2 = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)

        # notice that np.nanquantile doesn't handle infs properly
        # use gb_np_quantile for comparison

        rand_idxs = rng.random(size=N)
        ds.data[rand_idxs < (nan_fraction + inf_fraction + minus_inf_fraction)] = -np.inf
        ds.data[rand_idxs < (nan_fraction + inf_fraction)] = np.inf
        ds.data[rand_idxs < (nan_fraction)] = np.nan

        data_quant = ds.categ.quantile(ds.data, q=q)
        data2_quant = ds.categ.quantile(ds.data2, q=q)
        both_quant = ds.categ.quantile((ds.data, ds.data2), q=q)

        assert_array_almost_equal(data_quant.data, both_quant.data, err_msg=err_msg())
        assert_array_almost_equal(data2_quant.data2, both_quant.data2, err_msg=err_msg())

        for category in rng.choice(categories, min(len(categories), 15)):
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

        for category in rng.choice(categories, min(len(categories), 15)):
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
    @pytest.mark.parametrize("N_cat", [20])
    @pytest.mark.parametrize("q", [[0.0, 0.34, 0.81], [0.0, 0.5, 1.0, 0.2], [0.123, 0.12345, 0.1], 0.734])
    def test_groupbyops_quantile_multi_params(self, N, N_cat, q):
        def err_msg():
            return f"""
q = {q}
N = {N}
N_cat = {N_cat}"""

        this_seed = N * N_cat * int(np.sum(q) * 1000) % 10_000
        rng = default_rng(this_seed)

        ds = Dataset({"categ": rng.integers(0, N_cat, size=N)})
        categories = np.unique(ds.categ)
        ds["categ"] = Cat(ds["categ"])

        ds.data = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)
        ds.data2 = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)
        ds.col_0 = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)
        ds.col_1 = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)
        ds.int_data = rng.integers(-10_000, 10_000, size=N)
        ds.int_data2 = rng.integers(0, 100, size=N)

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
                for category in rng.choice(categories, min(len(categories), 15)):
                    curr_filter = ds.categ == category
                    curr_slice = ds.filter(curr_filter)
                    data_res = quant_ds[quant_col][quant_ds.categ == category]
                    data_res_np = gb_np_quantile(curr_slice[column], q=quantile, is_nan_function=False)
                    assert_almost_equal(data_res, data_res_np, err_msg=err_msg() + f"\n{column} {quant_col}\n{q_cols}")

        # multi-key

        ds.cat2 = rng.integers(0, N_cat, size=N)
        del ds.categ
        ds.categ = rng.integers(0, N_cat, size=N)
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
                for category in rng.choice(categories, min(len(categories), 15)):
                    curr_filter = mk_cat._fa == category
                    curr_slice = ds.filter(curr_filter)
                    data_res = quant_ds[quant_col][fa_cat_aligned == category]
                    data_res_np = gb_np_quantile(curr_slice[column], q=quantile, is_nan_function=False)
                    assert_array_almost_equal(
                        data_res, data_res_np, err_msg=err_msg() + f"\n{column} {quant_col}\n{q_cols}"
                    )

    @pytest.mark.parametrize("N", [1, 10_000, 100_000])
    @pytest.mark.parametrize("N_cat", [1, 100])
    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
    def test_groupbyops_basic_functions(self, N, N_cat, dtype):
        this_seed = N * N_cat
        rng = default_rng(this_seed)

        def err_msg():
            return f"""
N = {N}
N_cat = {N_cat}
this_seed = {this_seed}
dtype = {dtype}
gb_function = {gb_function}
np_function = {np_function}"""

        ds = Dataset({"categ": rng.integers(0, N_cat, size=N)})
        categories = np.unique(ds.categ)
        ds["categ"] = Cat(ds["categ"])

        ds.data = rng.uniform(-100, 100, size=N).astype(dtype)

        if dtype in [np.int32, np.int64]:
            np_data_copy = np.array(ds.data.copy().astype(np.float64))
        else:
            np_data_copy = np.array(ds.data.copy())

        # np_data_copy[nan_idxs] = np.nan
        # ds.data[nan_idxs] = np.nan

        gb_functions = [
            ds.categ.sum,
            ds.categ.mean,
            ds.categ.var,
            ds.categ.std,
            ds.categ.nansum,
            ds.categ.nanmean,
            ds.categ.nanvar,
            ds.categ.nanstd,
        ]
        np_functions = [
            np.sum,
            np.mean,
            lambda x: np.var(x, ddof=1),
            lambda x: np.std(x, ddof=1),
            np.nansum,
            np.nanmean,
            lambda x: np.nanvar(x, ddof=1),
            lambda x: np.nanstd(x, ddof=1),
        ]

        def test_categories(ds, gb_result):
            for category in rng.choice(categories, min(len(categories), 7)):
                curr_filter = ds.categ == category
                if not curr_filter.sum():
                    return
                np_result = np_function(np_data_copy[curr_filter])

                if dtype in [np.int32, np.int64]:
                    categ_result = gb_result.data.astype(np.float64)[gb_result.categ == category][0]
                else:
                    categ_result = gb_result.data[gb_result.categ == category][0]

                if dtype not in [np.float32]:
                    assert_almost_equal(categ_result, np_result, err_msg=err_msg())
                else:
                    decimal = 2 if N <= 10_000 else 1
                    assert_almost_equal(categ_result, np_result, err_msg=err_msg(), decimal=decimal)

        for phase in range(2):
            if phase == 1:
                nan_idxs = rng.random(N) < 0.5
                ds.data[nan_idxs] = np.nan
                np_data_copy[nan_idxs] = np.nan

            for gb_function, np_function in zip(gb_functions, np_functions):
                gb_result = gb_function(ds.data)
                test_categories(ds, gb_result)

    @pytest.mark.parametrize(
        "N, N_cat, window, q, nan_fraction, inf_fraction, minus_inf_fraction, dtype",
        quantile_params_generator(
            [np.int32, np.float64],
            max_N=1_000,
            windowed=True,
            with_cats=True,
            with_nans=True,
            with_infs=False,
            seed=500,
        ),
    )
    def test_groupbyops_rolling_quantile(
        self, N, N_cat, window, q, nan_fraction, inf_fraction, minus_inf_fraction, dtype
    ):
        this_seed = (N * N_cat * int(q * 1000) * int(nan_fraction * 1000)) % 10_000
        rng = default_rng(this_seed)

        def err_msg():
            return f"""
N = {N}
N_cat = {N_cat}
window = {window}
q = {q}
nan_fraction = {nan_fraction}
dtype = {dtype}"""

        ds = Dataset({"categ": rng.integers(0, N_cat, size=N)})
        categories = np.unique(ds.categ)
        ds["categ"] = Cat(ds["categ"])

        ds.data = rng.normal(rng.uniform(-100, 100), rng.uniform(0.00, 100), size=N).astype(dtype)
        ds.data2 = rng.normal(rng.uniform(-100, 100), rng.uniform(0.00, 100), size=N).astype(dtype)

        nan_idxs = rng.random(size=N) < nan_fraction
        # need to convert to float for numpy to understand nans.
        # np_data_copy will only be used for numpy operations for checks.
        np_data_copy = ds.data.copy().astype(np.float64)
        np_data_copy[nan_idxs] = np.nan
        ds.data[nan_idxs] = np.nan

        np_data2_copy = ds.data2.copy().astype(np.float64)
        np_data2_copy[nan_idxs] = np.nan
        ds.data2[nan_idxs] = np.nan

        data_quant = ds.categ.rolling_quantile(ds.data, q=q, window=window)
        data2_quant = ds.categ.rolling_quantile(ds.data2, q=q, window=window)
        both_quant = ds.categ.rolling_quantile((ds.data, ds.data2), q=q, window=window)
        gb_quant = ds.groupby("categ").rolling_quantile(q=q, window=window)

        assert_array_almost_equal(data_quant.data, both_quant.data, err_msg=err_msg())
        assert_array_almost_equal(data2_quant.data2, both_quant.data2, err_msg=err_msg())
        assert_array_almost_equal(data_quant.data, gb_quant.data, err_msg=err_msg())

        for category in rng.choice(categories, min(len(categories), 15)):
            curr_filter = ds.categ == category
            curr_window = min(window, curr_filter.sum())
            if curr_window == 0:
                continue

            data_res = data_quant.filter(curr_filter).data
            curr_np_data = np.array(np_data_copy[curr_filter])
            data_res_np = np_rolling_nanquantile(a=curr_np_data, q=q, window=curr_window)
            assert_almost_equal(data_res[curr_window - 1 :], data_res_np, err_msg=err_msg())

            data2_res = data2_quant.filter(curr_filter).data2
            curr_np_data2 = np.array(np_data2_copy[curr_filter])
            data2_res_np = np_rolling_nanquantile(a=curr_np_data2, q=q, window=curr_window)
            assert_almost_equal(data2_res[curr_window - 1 :], data2_res_np, err_msg=err_msg())

    @pytest.mark.parametrize(
        "N, N_cat, window, q, nan_fraction, inf_fraction, minus_inf_fraction, dtype",
        quantile_params_generator(
            [np.float64], max_N=1_000, windowed=True, with_cats=True, with_nans=True, with_infs=True, seed=600
        ),
    )
    def test_groupbyops_rolling_quantile_infs(
        self, N, N_cat, window, q, nan_fraction, inf_fraction, minus_inf_fraction, dtype
    ):
        def err_msg():
            return f"""
N = {N}
N_cat = {N_cat}
q = {q}
window = {window}
nan_fraction = {nan_fraction}
inf_fraction = {inf_fraction}
minus_inf_fraction = {minus_inf_fraction}"""

        this_seed = (N * N_cat * int(q * 1000) * int(nan_fraction * 1000)) % 10_000
        rng = default_rng(this_seed)

        ds = Dataset({"categ": rng.integers(0, N_cat, size=N)})
        categories = np.unique(ds.categ)
        ds["categ"] = Cat(ds["categ"])

        data_quant = Dataset()
        # check that works fine with multiple data columns
        for i in range(6):
            ds[f"data{i}"] = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)

            rand_idxs = rng.random(size=N)
            ds[f"data{i}"][rand_idxs < (nan_fraction + inf_fraction + minus_inf_fraction)] = -np.inf
            ds[f"data{i}"][rand_idxs < (nan_fraction + inf_fraction)] = np.inf
            ds[f"data{i}"][rand_idxs < (nan_fraction)] = np.nan

            data_quant[f"data{i}"] = ds.categ.rolling_quantile(ds[f"data{i}"], q=q, window=window)[f"data{i}"]

        all_quant = ds.categ.rolling_quantile([ds[f"data{i}"] for i in range(6)], q=q, window=window)
        gb_quant = ds.groupby("categ").rolling_quantile(q=q, window=window)

        for i in range(6):
            assert_array_almost_equal(data_quant[f"data{i}"], all_quant[f"data{i}"], err_msg=err_msg())
            assert_array_almost_equal(data_quant[f"data{i}"], gb_quant[f"data{i}"], err_msg=err_msg())

        for i in range(6):
            for category in rng.choice(categories, min(len(categories), 1)):
                curr_filter = ds.categ == category
                curr_window = min(window, curr_filter.sum())
                if curr_window == 0:
                    continue

                data_res = data_quant.filter(curr_filter)[f"data{i}"]
                curr_np_data = np.array(ds[f"data{i}"][curr_filter])
                data_res_np = np_rolling_nanquantile(a=curr_np_data, q=q, window=curr_window)
                assert_almost_equal(data_res[curr_window - 1 :], data_res_np, err_msg=err_msg())

    @pytest.mark.parametrize("N", [300])
    @pytest.mark.parametrize("N_cat", [10])
    @pytest.mark.parametrize("q", [[0.0, 0.34, 0.81], [0.0, 0.5, 1.0, 0.24], [0.123, 0.314, 0.1], 0.734])
    @pytest.mark.parametrize("window", [20])
    def test_groupbyops_rolling_quantile_multi_params(self, N, N_cat, q, window):
        def err_msg():
            return f"""
N = {N}
N_cat = {N_cat}
q = {q}
window = {window}"""

        this_seed = N * N_cat * int(np.sum(q) * 1000) % 10_000
        rng = default_rng(this_seed)

        ds = Dataset({"categ": rng.integers(0, N_cat, size=N)})
        categories = np.unique(ds.categ)
        ds["categ"] = Cat(ds["categ"])

        ds.data = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)
        ds.data2 = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)
        ds.col_0 = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)
        ds.col_1 = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)
        ds.int_data = rng.integers(-10_000, 10_000, size=N)
        ds.int_data2 = rng.integers(0, 100, size=N)

        cols = ["data", "data2", "col_0", "col_1", "int_data", "int_data2"]

        quant_ds = ds.categ.rolling_quantile(ds, q=q, window=window)
        quant_sep = ds.categ.rolling_quantile(
            (ds.data, ds.data2, ds.col_0, ds.col_1, ds.int_data, ds.int_data2), q=q, window=window
        )

        assert_array_equal(quant_ds.imatrix_make(), quant_sep.imatrix_make())

        q_cols = quant_ds.keys()

        if np.isscalar(q):
            q = [q]
            q_cols = cols

        # columns are ordered like [col + quantile for quantile in q for col in cols] if len(q) > 1

        for i_col, column in enumerate(cols):
            for i_q, quantile in enumerate(q):
                quant_col = q_cols[i_col * len(q) + i_q]
                for category in rng.choice(categories, min(len(categories), 15)):
                    curr_filter = ds.categ == category
                    curr_window = min(window, curr_filter.sum())
                    if curr_window == 0:
                        continue
                    data_res = quant_ds.filter(curr_filter)[quant_col]
                    curr_np_data = np.array(ds[column][curr_filter])
                    data_res_np = np_rolling_nanquantile(a=curr_np_data, q=quantile, window=curr_window)
                    assert_almost_equal(data_res[curr_window - 1 :], data_res_np, err_msg=err_msg())

        # multi-key

        ds.cat2 = rng.integers(0, N_cat, size=N)
        del ds.categ
        ds.categ = rng.integers(0, N_cat, size=N)
        mk_cat = Cat([ds.categ, ds.cat2])
        categories = np.unique(mk_cat._fa)
        fa_cat_aligned = mk_cat.first(mk_cat._fa)["col_0"]

        quant_ds = mk_cat.rolling_quantile(ds, q=q, window=window)
        quant_sep = mk_cat.rolling_quantile(
            (ds.data, ds.data2, ds.col_0, ds.col_1, ds.int_data, ds.int_data2), q=q, window=window
        )

        assert_array_almost_equal(quant_ds.imatrix_make(), quant_sep.imatrix_make())

        q_cols = quant_ds.keys()

        if np.isscalar(q):
            q = [q]
            q_cols = cols

        for i_col, column in enumerate(cols):
            for i_q, quantile in enumerate(q):
                quant_col = q_cols[i_col * len(q) + i_q]
                for category in rng.choice(categories, min(len(categories), 15)):
                    curr_filter = mk_cat._fa == category
                    curr_window = min(window, curr_filter.sum())
                    if curr_window == 0:
                        continue
                    data_res = quant_ds[quant_col][curr_filter]
                    curr_np_data = np.array(ds[column][curr_filter])
                    data_res_np = np_rolling_nanquantile(a=curr_np_data, q=quantile, window=curr_window)
                    assert_almost_equal(data_res[curr_window - 1 :], data_res_np, err_msg=err_msg())

    @pytest.mark.skip("Old fill functions were replaced by numba implementations")
    @pytest.mark.parametrize("N", [1, 100, 1_000])
    @pytest.mark.parametrize("N_cat", [1, 10, 300])
    @pytest.mark.parametrize("limit", [0, 3, 10])
    @pytest.mark.parametrize("nan_fraction", [0.5, 0.95])
    @pytest.mark.parametrize("fill_value", [0, -15, 100, None])
    def test_groupbyops_compare_fill_nb_fill(self, N, N_cat, limit, nan_fraction, fill_value):
        def err_msg():
            return f"""
N = {N}
N_cat = {N_cat}
limit = {limit}"""

        this_seed = N * N_cat * np.sum(limit) % 10_000
        rng = default_rng(this_seed)

        ds = rt.Dataset()
        ds.categ = rt.Cat(rng.integers(0, N_cat, size=N))
        ds.categ2 = rng.integers(0, N_cat, size=N)
        multi_cat = rt.Cat([ds["categ"], ds["categ2"]])

        ds.gb_key1 = rng.integers(0, N_cat, size=N)
        ds.gb_key2 = rng.integers(0, N_cat, size=N)
        single_gb = ds.groupby("gb_key1")
        multi_gb = ds.groupby(["gb_key1", "gb_key2"])

        for data_type in [
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
        ]:
            temp_data = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)
            if type in [np.uint8, np.uint16, np.uint32, np.uint64]:
                temp_data = np.abs(temp_data)

            ds[f"data_{data_type.__name__}"] = rt.FA(temp_data).astype(data_type)
            ds[f"data_{data_type.__name__}"][rng.random(N) < nan_fraction] = np.nan

        # Cat

        old_cat_fill_f = ds.categ.fill_forward(ds, limit=limit, fill_val=fill_value)
        new_cat_fill_f = ds.categ.nb_fill_forward(ds, limit=limit, fill_val=fill_value)

        old_multi_cat_fill_f = multi_cat.fill_forward(ds, limit=limit, fill_val=fill_value)
        new_multi_cat_fill_f = multi_cat.nb_fill_forward(ds, limit=limit, fill_val=fill_value)

        assert old_cat_fill_f.equals(new_cat_fill_f, exact=True)
        assert old_multi_cat_fill_f.equals(new_multi_cat_fill_f)

        old_cat_fill_b = ds.categ.fill_backward(ds, limit=limit, fill_val=fill_value)
        new_cat_fill_b = ds.categ.nb_fill_backward(ds, limit=limit, fill_val=fill_value)

        old_multi_cat_fill_b = multi_cat.fill_backward(ds, limit=limit, fill_val=fill_value)
        # because of weirdness of names when doing operations on multi-key categoricals
        # exact=True always fails for multi-key categoricals if the keys have the same name
        old_multi_cat_fill_b.col_rename("gb_key_1", "gb_key_0")
        new_multi_cat_fill_b = multi_cat.nb_fill_backward(ds, limit=limit, fill_val=fill_value)

        assert old_cat_fill_b.equals(new_cat_fill_b, exact=True)
        assert old_multi_cat_fill_b.equals(new_multi_cat_fill_b)

        # GB

        old_gb_fill_f = single_gb.fill_forward(limit=limit, fill_val=fill_value)
        new_gb_fill_f = single_gb.nb_fill_forward(limit=limit, fill_val=fill_value)

        old_multi_gb_fill_f = multi_gb.fill_forward(limit=limit, fill_val=fill_value)
        new_multi_gb_fill_f = multi_gb.nb_fill_forward(limit=limit, fill_val=fill_value)

        assert old_gb_fill_f.equals(new_gb_fill_f, exact=True)
        assert old_multi_gb_fill_f.equals(new_multi_gb_fill_f, exact=True)

        old_gb_fill_b = single_gb.fill_backward(limit=limit, fill_val=fill_value)
        new_gb_fill_b = single_gb.nb_fill_backward(limit=limit, fill_val=fill_value)

        old_multi_gb_fill_b = multi_gb.fill_backward(limit=limit, fill_val=fill_value)
        new_multi_gb_fill_b = multi_gb.nb_fill_backward(limit=limit, fill_val=fill_value)

        assert old_gb_fill_b.equals(new_gb_fill_b, exact=True)
        assert old_multi_gb_fill_b.equals(new_multi_gb_fill_b, exact=True)

        # Make sure names are the same for multi-key categoricals
        ds = rt.Dataset()
        ds.categ = rt.Cat(rng.integers(0, N_cat, size=N))
        ds.categ2 = rng.integers(0, N_cat, size=N)
        multi_cat = rt.Cat([ds["categ"], ds["categ2"]])

        for data_type in [
            np.int64,
            np.float64,
        ]:
            temp_data = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)
            if type in [np.uint8, np.uint16, np.uint32, np.uint64]:
                temp_data = np.abs(temp_data)

            ds[f"data_{data_type.__name__}"] = rt.FA(temp_data).astype(data_type)
            ds[f"data_{data_type.__name__}"][rng.random(N) < nan_fraction] = np.nan

        # make copy of a categorical, as multiple applications of method to a categorical changes names of columns
        multi_cat2 = rt.Cat([ds["categ"], ds["categ2"]])

        # old
        old_multi_cat_fill_f = multi_cat.fill_forward(ds, limit=limit, fill_val=fill_value)
        old_multi_cat_fill_f_second_call = multi_cat.fill_forward(ds, limit=limit, fill_val=fill_value)

        # new
        new_multi_cat_fill_f = multi_cat2.fill_forward(ds, limit=limit, fill_val=fill_value)
        new_multi_cat_fill_f_second_call = multi_cat2.fill_forward(ds, limit=limit, fill_val=fill_value)

        # just check that column names are exactly the same, we checked that the values are equal in tests above
        assert old_multi_cat_fill_f.keys() == new_multi_cat_fill_f.keys()
        assert old_multi_cat_fill_f_second_call.keys() == new_multi_cat_fill_f_second_call.keys()

    @pytest.mark.parametrize(
        "N, N_cat, nan_fraction, types, limit, fill_val",
        fills_params_generator(),
    )
    def test_groupbyops_fills(self, N, N_cat, nan_fraction, types, limit, fill_val):
        this_seed = N * N_cat * int(nan_fraction * 1000) % 10_000
        rng = default_rng(this_seed)

        def err_msg(**kwargs):
            base = f"""
seed = {this_seed}
N = {N}
N_cat = {N_cat}
nan_fraction = {nan_fraction}
limit = {limit}
fill_val = {fill_val}"""
            for k, v in kwargs.items():
                base += f"""
{k} = {v}
"""
            return base

        ds = rt.Dataset()
        ds.categ = rt.Cat(rng.integers(0, N_cat, size=N))
        ds.categ2 = rng.integers(0, N_cat, size=N)
        multi_cat = rt.Cat([ds["categ"], ds["categ2"]])

        ds.gb_key1 = rng.integers(0, N_cat, size=N)
        ds.gb_key2 = rng.integers(0, N_cat, size=N)
        single_gb = ds.groupby("gb_key1")
        multi_gb = ds.groupby(["gb_key1", "gb_key2"])

        data_column_names = []

        for data_type in types:
            temp_data = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)
            if type in [np.uint8, np.uint16, np.uint32, np.uint64]:
                temp_data = np.abs(temp_data)

            col_name = f"data_{data_type.__name__}"
            ds[col_name] = rt.FA(temp_data).astype(data_type)
            ds[col_name][rng.random(N) < nan_fraction] = np.nan
            data_column_names.append(col_name)

        # Cat

        cat_ff = ds.categ.fill_forward(ds, limit=limit, fill_val=fill_val)
        cat_fb = ds.categ.fill_backward(ds, limit=limit, fill_val=fill_val)

        single_cat_result = {"ff": cat_ff, "fb": cat_fb}

        multi_cat_ff = multi_cat.fill_forward(ds, limit=limit, fill_val=fill_val)
        multi_cat_fb = multi_cat.fill_backward(ds, limit=limit, fill_val=fill_val)

        multi_cat_result = {"ff": multi_cat_ff, "fb": multi_cat_fb}

        for categorical, ds_result in zip([ds.categ, multi_cat], [single_cat_result, multi_cat_result]):
            unique_categories = np.unique(categorical._fa)
            for category in rng.choice(unique_categories, min(len(unique_categories), 8)):
                for col in data_column_names:
                    curr_filter = categorical._fa == category
                    curr_data = ds[col][curr_filter]
                    fa_ff = curr_data.fill_forward(limit=limit, fill_val=fill_val)
                    fa_fb = curr_data.fill_backward(limit=limit, fill_val=fill_val)

                    ds_ff = ds_result["ff"][col][curr_filter]
                    ds_fb = ds_result["fb"][col][curr_filter]

                    assert_array_equal(ds_ff, fa_ff, err_msg=err_msg(curr_data=curr_data))
                    assert_array_equal(ds_fb, fa_fb, err_msg=err_msg(curr_data=curr_data))

        # GB

        gb_ff = single_gb.fill_forward(limit=limit, fill_val=fill_val)
        gb_fb = single_gb.fill_backward(limit=limit, fill_val=fill_val)

        single_gb_result = {"ff": gb_ff, "fb": gb_fb}

        multi_cat_ff = multi_gb.fill_forward(limit=limit, fill_val=fill_val)
        multi_cat_fb = multi_gb.fill_backward(limit=limit, fill_val=fill_val)

        multi_gb_result = {"ff": multi_cat_ff, "fb": multi_cat_fb}

        for gb, ds_result in zip([single_gb, multi_gb], [single_gb_result, multi_gb_result]):
            unique_categories = np.unique(gb.grouping._iKey.unique())
            for category in rng.choice(unique_categories, min(len(unique_categories), 8)):
                for col in data_column_names:
                    curr_filter = gb.grouping._iKey == category
                    curr_data = ds[col][curr_filter]
                    fa_ff = curr_data.fill_forward(limit=limit, fill_val=fill_val)
                    fa_fb = curr_data.fill_backward(limit=limit, fill_val=fill_val)

                    ds_ff = ds_result["ff"][col][curr_filter]
                    ds_fb = ds_result["fb"][col][curr_filter]

                    assert_array_equal(ds_ff, fa_ff)
                    assert_array_equal(ds_fb, fa_fb)

    def test_groupbyops_fills_inplace(self):
        N = 1234
        N_cat = 56
        types = [
            np.float32,
            np.float64,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
        ]
        nan_fraction = 0.5

        this_seed = N * N_cat * int(nan_fraction * 1000) % 10_000
        rng = default_rng(this_seed)

        ds = rt.Dataset()
        ds.categ = rt.Cat(rng.integers(0, N_cat, size=N))
        ds.categ2 = rng.integers(0, N_cat, size=N)
        multi_cat = rt.Cat([ds["categ"], ds["categ2"]])

        ds.gb_key1 = rng.integers(0, N_cat, size=N)
        ds.gb_key2 = rng.integers(0, N_cat, size=N)
        single_gb = ds.groupby("gb_key1")
        multi_gb = ds.groupby(["gb_key1", "gb_key2"])

        data_column_names = []

        for data_type in types:
            temp_data = rng.normal(rng.uniform(-10_000, 10_000), rng.uniform(0.00, 10_000), size=N)
            if type in [np.uint8, np.uint16, np.uint32, np.uint64]:
                temp_data = np.abs(temp_data)

            col_name = f"data_{data_type.__name__}"
            ds[col_name] = rt.FA(temp_data).astype(data_type)
            ds[col_name][rng.random(N) < nan_fraction] = np.nan
            data_column_names.append(col_name)

        ds_copy = ds.copy(deep=True)

        for cat_object in [ds.categ, multi_cat]:
            # not inplace
            ds = ds_copy.copy(deep=True)
            ds_ff = cat_object.fill_forward(ds[data_column_names])
            assert ds.equals(ds_copy)

            # inplace
            cat_object.fill_forward(ds[data_column_names], inplace=True)
            assert not ds.equals(ds_copy)
            assert (ds[data_column_names]).equals(ds_ff[data_column_names])

        # not inplace
        ds = ds_copy.copy(deep=True)
        single_gb = ds.groupby("gb_key1")
        ds_ff = single_gb.fill_forward()
        assert ds.equals(ds_copy)

        # inplace
        single_gb.fill_forward(inplace=True)
        assert not ds.equals(ds_copy)
        assert (ds[data_column_names]).equals(ds_ff[data_column_names])

        # not inplace
        ds = ds_copy.copy(deep=True)
        multi_gb = ds.groupby(["gb_key1", "gb_key2"])

        ds_ff = multi_gb.fill_forward()
        assert ds.equals(ds_copy)

        # inplace
        multi_gb.fill_forward(inplace=True)
        assert not ds.equals(ds_copy)
        assert (ds[data_column_names]).equals(ds_ff[data_column_names])


if __name__ == "__main__":
    tester = unittest.main()
