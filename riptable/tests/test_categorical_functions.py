import pandas as pd
import riptable as rt


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
import numpy as np

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


def safe_equal(ary1, ary2):
    def isNaN(num):
        return num != num

    assert len(ary1) == len(ary2)

    for a, b in zip(ary1, ary2):
        epsilon = 0.001  ##high epsilon for comparing float32 and float64
        if not ((abs(a - b) < epsilon) or (isNaN(a) and isNaN(b))):
            return False
    return True


class TestCategoricalSimpleGroupBy:
    def groupby_func(self, df, fn, data=None):
        return functions[fn](df, data)

    # TODO pytest parameterize across type_list and functions
    def test_single_col_categoricals(self):
        values = [0, 1, 1, 2, 2, 2, 3, 3, 3, 4]
        bin_ids = ['a', 'b', 'c', 'd', 'e']
        data = np.random.rand(10) + np.random.randint(0, 10, size=10)

        for type_ in type_list:
            data = rt.FastArray(data, dtype=type_)

            map = {'vs': data, 'ks': values}

            pd_data = pd.DataFrame(map).groupby(by='ks')
            sfw_data = rt.Categorical(values=values, categories=bin_ids, base_index=0)

            for function in range(0, len(functions)):
                pd_out = self.groupby_func(pd_data, function)
                sfw_out = self.groupby_func(sfw_data, function, data)

                col_index = 'Count' if functions_str[function] == 'count' else 0

                if (
                    not safe_equal(list(sfw_out[col_index]), list(pd_out['vs']))
                    and function != 'median'
                    and not np.issubdtype(type_, np.integer)
                ):
                    print(sfw_out)
                    print(pd_out)

                    # TODO - rework this so assert messages are used
                    print(list(sfw_out[col_index]))
                    print(list(pd_out['vs']))
                    print('Function failed on - ', functions_str[function])
                    assert False
