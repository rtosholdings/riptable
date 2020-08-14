import riptable as rt
import random as rand
import pandas as pd
import unittest

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
    'first',
    'last',
    # 'mode'
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
    assert len(ary1) == len(ary2)

    def isNaN(num):
        return num != num

    for a, b in zip(ary1, ary2):
        if not (a == b or (isNaN(a) and isNaN(b))):
            return False
    return True


def min(a, b):
    return a if a < b else b


class GroupbyFunctions_Test(unittest.TestCase):
    def groupby_func(self, df, fn):
        return getattr(df, functions_str[fn])

    def test_single_col_groupby_tests(self):

        Values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        Keys = ['a', 'b', 'c', 'a', 'b', 'c', 'd', 'e', 'f']
        for type_ in type_list:

            data = {'Vs': rt.FastArray(Values, dtype=type_), 'Ks': Keys}

            pd_data = pd.DataFrame(data)
            sfw_data = rt.Dataset(data)

            key = 'Ks'
            val = 'Vs'

            pd_gb = pd_data.groupby(key)
            sfw_gb = sfw_data.groupby(key)

            for name in functions_str:

                pd_func = getattr(pd_gb, name)
                sfw_func = getattr(sfw_gb, name)

                pd_out = pd_func()
                sfw_out = sfw_func()

                pd_col = pd_out[val]._values
                if name == 'count':
                    sfw_col = sfw_out['Count']
                else:
                    sfw_col = sfw_out[val]

                is_integer_subttype = np.issubdtype(type_, np.integer)
                is_median = name != 'median'
                if not safe_equal(pd_col, sfw_col) and (
                    not is_integer_subttype and not is_median
                ):
                    print('data_type_t = ', type_)
                    print('function =', name)
                    print('pandas output =', pd_col)
                    print('sfw    output =', sfw_col)
                    # TODO move as error message following assert
                    self.assertTrue(False)

    # TODO pytest parameterize type_list
    def test_multi_col_groupby_tests(self, numb_keys_and_values=5, numb_rows=20):
        col_val_names = ['alpha', 'beta', 'gamma', 'sigma', 'zeta']
        col_key_names = ['lions', 'tigers', 'bears', 'oh', 'my']

        MAX_LENGTH = min(len(col_val_names), len(col_key_names))
        assert numb_keys_and_values <= MAX_LENGTH
        for type_ in type_list:

            vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
            keys = 'a b c d e f g'.split(' ')

            vs = []
            ks = []

            for i in range(0, numb_keys_and_values):
                vs.append(
                    [vals[rand.randint(0, len(vals) - 1)] for i in range(0, numb_rows)]
                )
                ks.append(
                    [keys[rand.randint(0, len(keys) - 1)] for i in range(0, numb_rows)]
                )

            data = {}

            for i in range(0, numb_keys_and_values):
                data[col_val_names[i]] = rt.FastArray(vs[i], dtype=type_)
                data[col_key_names[i]] = rt.FastArray(vs[i], dtype=type_)

            pd_data = pd.DataFrame(data)
            sfw_data = rt.Dataset(data)
            key = col_key_names[0:numb_keys_and_values]
            val = col_val_names[0:numb_keys_and_values]

            pd_gb = pd_data.groupby(key)
            sfw_gb = sfw_data.groupby(key)

            for name in functions_str:
                pd_out = getattr(pd_gb, name)()
                sfw_out = getattr(sfw_gb, name)()

                if name == 'count':
                    # only compare one column for count
                    pd_col = pd_out['alpha']
                    sfw_col = sfw_out.Count
                    if not safe_equal(pd_col, sfw_col):
                        print('function =', name)
                        print('pandas output =', pd_col)
                        print('sfw    output =', sfw_col)
                        self.assertTrue(False)
                else:
                    for val in col_val_names:
                        # extract array from pandas series
                        pd_col = pd_out[val]._values
                        sfw_col = sfw_out[val]

                        is_integer_subttype = np.issubdtype(type_, np.integer)
                        is_median = name != 'median'
                        if not safe_equal(pd_col, sfw_col) and (
                            not is_integer_subttype and not is_median
                        ):
                            print('function =', name)
                            print('pandas output =', pd_col)
                            assert False


if __name__ == "__main__":
    tester = unittest.main()
