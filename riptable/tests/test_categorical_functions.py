import numpy as np
import pandas as pd
import riptable as rt

import pytest
from numpy.testing import assert_array_almost_equal


def count(df, data=None):
    return df.count() if data is None else df.count()  ##this is different


def sum(df, data=None):
    return df.sum() if data is None else df.sum(data)


def mean(df, data=None):
    return df.mean() if data is None else df.mean(data)


def median(df, data=None):
    return df.median() if data is None else df.median(data)


def min(df, data=None):
    return df.min() if data is None else df.min(data)


def max(df, data=None):
    return df.max() if data is None else df.max(data)


def prod(df, data=None):
    return df.prod() if data is None else df.prod(data)


def var(df, data=None):
    return df.var() if data is None else df.var(data)


def quantile(df, data=None):
    return df.quantile() if data is None else df.quantile(data)


def cumsum(df, data=None):
    return df.cumsum() if data is None else df.cumsum(data)


def cumprod(df, data=None):
    return df.cumprod() if data is None else df.cumprod(data)


def cummax(df, data=None):
    return df.cummax() if data is None else df.cummax(data)


def cummin(df, data=None):
    return df.cummin() if data is None else df.cummin(data)


functions = [
    count,
    sum,
    mean,
    median,
    min,
    max,
    # prod,
    var,
    # quantile,
    cumsum,
    cumprod,
    # cummax,
    # cummin
]

functions_str = [
    'count',
    'sum',
    'mean',
    'median',
    'min',
    'max',
    # 'prod',
    'var',
    # 'quantile',
    'cumsum',
    'cumprod',
    # 'cummax',
    # 'cummin'
]

type_list = [
    # np.bool,          ## not a numeric type
    np.intc,
    np.intp,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    # np.float16,       ## not supported
    np.float32,
    np.float64,
    # np.complex64,     ## not supported
    # np.complex128     ## not supported
]

class TestCategoricalSimpleGroupBy:
    def groupby_func(self, df, fn, data):
        return fn(df, data)

    @pytest.mark.parametrize(('func', 'func_name'), [
        pytest.param(x, y, id=y) for (x, y) in zip(functions, functions_str)
    ])
    @pytest.mark.parametrize('data_dtype', type_list)
    def test_single_col_categoricals(self, func, func_name: str, data_dtype):
        values = [0, 1, 1, 2, 2, 2, 3, 3, 3, 4]
        bin_ids = ['a', 'b', 'c', 'd', 'e']
        #data = np.random.rand(10) + np.random.randint(0, 10, size=10)
        data = np.array([
            7.19200901, 0.14907245, 2.28258397, 5.07872708, 0.76125165,
            1.32797916, 3.40280423, 4.48942476, 6.98713656, 4.39541456])

        data = rt.FastArray(data, dtype=data_dtype)

        map = {'vs': data, 'ks': values}

        pd_data = pd.DataFrame(map).groupby(by='ks')
        rt_data = rt.Categorical(values=values, categories=bin_ids, base_index=0)

        pd_out = self.groupby_func(pd_data, func, None)
        rt_out = self.groupby_func(rt_data, func, data)

        col_index = 'Count' if func_name == 'count' else 0

        assert_array_almost_equal(rt_out[col_index], pd_out['vs'].values, decimal=3)
