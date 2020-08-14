__all__ = ['TestData', 'load_test_data']

import numpy as np
from ..rt_enum import TypeRegister as treg, NumpyCharTypes
from ..rt_numpy import *
from ..rt_timers import tt


class TestDataMeta(type):
    length = 20

    _dataset_numeric = None
    _dataset_strings = None
    _dataset_rt_types = None
    _dataset_for_gb = None
    _categorical_all = None
    _struct_nested = None

    _categorical_bytes = None
    _categorical_numeric = None
    _categorical_codes = None
    _categorical_mapping = None
    _categorical_multikey = None
    _categorical_matlab = None
    _categorical_pandas = None

    _categorical_firstorder = None
    _categorical_unicode = None
    _categorical_zero = None

    _int_fa = None
    _bytes_fa = None
    _unicode_fa = None
    _flt_fa = None
    _flt_fa2 = None
    _str_map = None

    _rand_bool = None
    # ------------------DATASETS-------------------------------------------
    # ---------------------------------------------------------------------
    @property
    def dataset_numeric(cls):
        # possibly reset dataset with new length
        if (
            cls._dataset_numeric is not None
            and cls._dataset_numeric._nrows != cls.length
        ):
            cls._dataset_numeric = None

        if cls._dataset_numeric is None:
            cls._dataset_numeric = treg.Dataset(
                {
                    'int8': np.random.randint(
                        low=-128, high=127, size=(cls.length), dtype=np.int8
                    ),
                    'uint8': np.random.randint(
                        low=0, high=255, size=(cls.length), dtype=np.uint8
                    ),
                    'int16': np.random.randint(
                        low=-32768, high=32767, size=(cls.length), dtype=np.int16
                    ),
                    'uint16': np.random.randint(
                        low=0, high=65535, size=(cls.length), dtype=np.uint16
                    ),
                    'int32': np.random.randint(
                        low=-2147483648,
                        high=2147483647,
                        size=(cls.length),
                        dtype=np.int32,
                    ),
                    'uint32': np.random.randint(
                        low=0, high=4294967295, size=(cls.length), dtype=np.uint32
                    ),
                    'int64': np.random.randint(
                        low=-9223372036854775808,
                        high=9223372036854775807,
                        size=(cls.length),
                        dtype=np.int64,
                    ),
                    'uint64': np.random.randint(
                        low=0,
                        high=18446744073709551615,
                        size=(cls.length),
                        dtype=np.uint64,
                    ),
                    'float32': np.random.rand(cls.length).astype(np.float32) * 10e9,
                    'float64': np.random.rand(cls.length).astype(np.float64) * 10e-9,
                }
            )
        return cls._dataset_numeric

    # ---------------------------------------------------------------------
    @property
    def dataset_strings(cls):
        # possibly reset dataset with new length
        if (
            cls._dataset_strings is not None
            and cls._dataset_strings._nrows != cls.length
        ):
            cls._dataset_strings = None

        if cls._dataset_strings is None:
            cls._dataset_strings = treg.Dataset(
                {
                    'unicode': treg.FastArray(
                        np.random.choice(
                            [
                                'peter',
                                'paul',
                                'mary ann',
                                'joe',
                                'john',
                                'ray',
                                'mary katherine',
                            ],
                            cls.length,
                        ),
                        unicode=True,
                    ),
                    'bytes': treg.FastArray(
                        np.random.choice(
                            ['greg', 'julia', 'paul', 'marc', 'chuck', 'mary'],
                            cls.length,
                        )
                    ),
                    'categorical': treg.Categorical(
                        np.random.randint(1, 8, cls.length),
                        ['mary ann', 'paul', 'sam', 'charlie', 'norm', 'lenny', 'tula'],
                    ),
                }
            )
        return cls._dataset_strings

    # ---------------------------------------------------------------------
    @property
    def dataset_for_gb(cls):
        '''
        This is a good one for testing groupby results/sorting/apply/etc.
        '''
        if cls._dataset_for_gb is not None and cls._dataset_for_gb._nrows != cls.length:
            cls._dataset_for_gb = None

        if cls._dataset_for_gb is None:
            cls._dataset_for_gb = treg.Dataset(
                {'col_' + str(i): np.random.rand(cls.length) for i in range(10)}
            )
            cls._dataset_for_gb.keycol = treg.FastArray(
                np.random.choice(
                    [
                        'peter',
                        'paul',
                        'mary ann',
                        'joe',
                        'john',
                        'ray',
                        'mary katherine',
                    ],
                    cls.length,
                ),
                unicode=True,
            )

        return cls._dataset_for_gb

    # ---------------------------------------------------------------------
    @property
    def dataset_rt_types(cls):
        if cls._dataset_rt_types is None:
            cls._dataset_rt_types = treg.Dataset(
                {
                    'CAT': cls.categorical_bytes,
                    'DTN': treg.DateTimeNano.random(cls.length),
                    'DATE': treg.Date(np.random.randint(15000, 20000, cls.length)),
                    'TSPAN': treg.TimeSpan(
                        np.random.randint(
                            0, 1_000_000_000 * 60 * 60 * 24, cls.length, dtype=np.int64
                        )
                    ),
                    'DSPAN': treg.DateSpan(np.random.randint(0, 365, cls.length)),
                }
            )
        return cls._dataset_rt_types

    # -----STRUCTS---------------------------------------------------------
    # ---------------------------------------------------------------------
    @property
    def struct_nested(cls):
        if cls._struct_nested is None:
            cls._struct_nested = treg.Struct(
                {
                    'cat1': treg.Categorical(['a', 'b', 'c', 'd']),
                    'arr1': arange(5),
                    'ds1': treg.Dataset(
                        {'col_' + str(i): np.random.rand(5) for i in range(5)}
                    ),
                    'arr2': arange(10),
                    'test_string': 'this is my string',
                    'array3': arange(30).astype(np.int8),
                    'nested_struct1': treg.Struct(
                        {
                            'set1': treg.Dataset(
                                {
                                    'col_' + str(i): np.random.randint(5)
                                    for i in range(3)
                                }
                            ),
                            'set2': treg.Dataset(
                                {
                                    'col_'
                                    + str(i): np.random.choice(['a', 'b', 'c'], 10)
                                    for i in range(2)
                                }
                            ),
                            'nested_2': treg.Struct(
                                {
                                    'c1': treg.Categorical(arange(5)),
                                    'c2': treg.Categorical(['aaa', 'bbbb', 'cccc']),
                                    'leaf_dataset': treg.Dataset(
                                        {'col_' + str(i): arange(5) for i in range(3)}
                                    ),
                                    'array4': arange(20).astype(np.uint64),
                                }
                            ),
                            'set3': treg.Dataset(
                                {
                                    'col_' + str(i): np.random.choice([True, False], 10)
                                    for i in range(4)
                                }
                            ),
                        }
                    ),
                    'ds2': treg.Dataset(
                        {'heading_' + str(i): np.random.rand(5) for i in range(3)}
                    ),
                    'int1': 5,
                    'float1': 7.0,
                    'cat2': treg.Categorical(['a', 'b', 'c', 'd']),
                }
            )
        return cls._struct_nested

    # -----CATEGORICALS----------------------------------------------------
    # ---------------------------------------------------------------------
    def _gen_categorical_data(cls):
        if cls._int_fa is not None and len(cls._int_fa) != cls.length:
            cls._int_fa = None

        if cls._int_fa is None:
            cls._int_fa = treg.FastArray(np.random.choice([10, 20, 30], cls.length))
            cls._bytes_fa = treg.FastArray(
                np.random.choice(
                    ['adam', 'brian', 'charlie', 'david', 'edward'], cls.length
                )
            )
            cls._unicode_fa = treg.FastArray(
                np.random.choice(
                    ['amanda', 'beth', 'catherine', 'danielle', 'emma'], cls.length
                ),
                unicode=True,
            )
            cls._flt_fa = treg.FastArray(np.random.rand(cls.length)) * 100
            cls._flt_fa2 = treg.FastArray(np.random.rand(cls.length)) * 100
            cls._str_map = {10: 'x', 20: 'y', 30: 'z', 40: 'zz'}

    def _gen_categorical(cls, name):
        cat = getattr(cls, name)
        if cat is not None and len(cat) != cls.length:
            cat = None

        if cat is None:
            cls._gen_categorical_data()

        return cat

    @property
    def categorical_all(cls):
        data = treg.Dataset(
            {
                'categorical_bytes': cls.categorical_bytes,
                'categorical_unicode': cls.categorical_unicode,
                'categorical_multikey': cls.categorical_multikey,
                'categorical_firstorder': cls.categorical_firstorder,
                'categorical_zero': cls.categorical_zero,
                'categorical_numeric': cls.categorical_numeric,
            }
        )
        return data

    @property
    def categorical_bytes(cls):
        cat = cls._gen_categorical('_categorical_bytes')
        if cat is None:
            cls._categorical_bytes = treg.Categorical(cls._bytes_fa)
        return cls._categorical_bytes

    @property
    def categorical_unicode(cls):
        cat = cls._gen_categorical('_categorical_unicode')
        if cat is None:
            cls._categorical_unicode = treg.Categorical(cls._unicode_fa, unicode=True)
        return cls._categorical_unicode

    @property
    def categorical_numeric(cls):
        cat = cls._gen_categorical('_categorical_numeric')
        if cat is None:
            cls._categorical_numeric = treg.Categorical(cls._int_fa)
        return cls._categorical_numeric

    @property
    def categorical_multikey(cls):
        cat = cls._gen_categorical('_categorical_multikey')
        if cat is None:
            cls._categorical_multikey = treg.Categorical([cls._bytes_fa, cls._int_fa])
        return cls._categorical_multikey

    @property
    def categorical_firstorder(cls):
        cat = cls._gen_categorical('_categorical_firstorder')
        if cat is None:
            cls._categorical_firstorder = treg.Categorical(cls._bytes_fa, ordered=False)
        return cls._categorical_firstorder

    @property
    def categorical_zero(cls):
        cat = cls._gen_categorical('_categorical_zero')
        if cat is None:
            cls._categorical_zero = treg.Categorical(cls._bytes_fa, base_index=0)
        return cls._categorical_zero

    # ----ARRAYS-----------------------------------------------------------
    # ---------------------------------------------------------------------
    @property
    def rand_bool(cls):
        if cls._rand_bool is not None and len(cls._rand_bool) != cls.length:
            cls._rand_bool = None
        if cls._rand_bool is None:
            cls._rand_bool = np.random.randint(0, 2, cls.length, dtype=np.bool).view(
                treg.FastArray
            )
        return cls._rand_bool

    # ---SETTINGS----------------------------------------------------------
    # ---------------------------------------------------------------------
    def reset(cls):
        cls._dataset_numeric = None
        cls._dataset_strings = None
        cls._dataset_rt_types = None
        cls._dataset_for_gb = None
        cls._categorical_all = None
        cls._struct_nested = None

        cls._categorical_bytes = None
        cls._categorical_numeric = None
        cls._categorical_codes = None
        cls._categorical_mapping = None
        cls._categorical_multikey = None
        cls._categorical_matlab = None
        cls._categorical_pandas = None

        cls._categorical_firstorder = None
        cls._categorical_unicode = None
        cls._categorical_zero = None

        _rand_bool = None

        cls._int_fa = None
        cls._bytes_fa = None
        cls._unicode_fa = None
        cls._flt_fa = None
        cls._flt_fa2 = None
        cls._str_map = None

    def set_array_size(cls, sz):
        if not isinstance(sz, (int, np.integer)):
            raise TypeError(f"Array length must be an integer, not {type(size)}")
        cls.length = sz

    # ----PERFORMANCE TESTS------------------------------------------------
    # ---------------------------------------------------------------------
    def _test_arr_from_dt(cls, nrows=1_000_000, dtype=np.dtype(np.float64)):
        if dtype.char in NumpyCharTypes.AllFloat:
            arr = np.asarray(np.random.rand(nrows), dtype=dtype)
        elif dtype.char in NumpyCharTypes.AllInteger:
            maxint = np.iinfo(dtype).max
            maxint = min(nrows, maxint)
            arr = np.random.randint(0, maxint, size=nrows, dtype=dtype)
        else:
            raise ValueError(
                f"Cannot perform reduce test with {dtype} dtype. Use integer or floating point type instead."
            )

        return arr

    # ---------------------------------------------------------------------
    def test_dataset_reduce(
        cls,
        nrows=1_000_000,
        ncols=10,
        dtype=np.float64,
        loops=2,
        transpose=False,
        apply=False,
    ):
        reduce_funcs = [
            'sum',
            'nansum',
            'mean',
            'nanmean',
            'std',
            'nanstd',
            'var',
            'nanvar',
            'min',
            'nanmin',
            'max',
            'nanmax',
        ]

        import pandas as pd

        if dtype is None:
            dtype = list(NumpyCharTypes.Computable)

        if not isinstance(dtype, list):
            dtype = [dtype]

        arrs = []
        if isinstance(dtype, list):
            dts = []
            # validate all dtypes
            for item in dtype:
                try:
                    dts.append(np.dtype(item))
                except:
                    raise ValueError(f"{item} was not a valid numpy dtype initializer.")

            # generate arrays
            for i in range(ncols):
                dt = np.random.choice(dts)
                arrs.append(cls._test_arr_from_dt(nrows, dt))

        ds = treg.Dataset({'c' + str(i): arrs[i] for i in range(ncols)})
        df = pd.DataFrame(ds.asdict())

        rt_times = []
        pd_times = []

        if apply:
            for i, func_str in enumerate(reduce_funcs):
                print(f"testing apply {func_str}")
                rt_str = f"ds.apply({func_str})"
                pd_str = f"df.apply(np.{func_str})"

                rt_times.append(tt(rt_str, loops=loops, return_time=True))
                pd_times.append(tt(pd_str, loops=loops, return_time=True))
        else:
            for i, func_str in enumerate(reduce_funcs):
                print(f"testing {func_str}")
                rt_str = f"ds.{func_str}()"
                if i % 2 == 1:
                    pd_str = f"df.{reduce_funcs[i-1]}()"
                else:
                    pd_str = f"df.{func_str}(skipna=False)"

                rt_times.append(tt(rt_str, loops=loops, return_time=True))
                pd_times.append(tt(pd_str, loops=loops, return_time=True))

        rt_times = treg.FastArray(rt_times)
        pd_times = treg.FastArray(pd_times)
        speedup = pd_times / rt_times

        result = treg.Dataset(
            {
                'operation': treg.FastArray(reduce_funcs),
                'pandas time (s)': pd_times,
                'riptable time (s)': rt_times,
                'speedup': speedup,
            }
        )

        result.sort_inplace('speedup', ascending=False)
        result.speedup = treg.FastArray(
            ["{0:.2f}".format(i) + 'x' for i in result.speedup]
        )

        return result._temp_display('PRECISION', 3)


class TestData(metaclass=TestDataMeta):
    pass


def load_test_data(generate=True):
    '''
    Return a Struct containing all of the test data in the TestData class (will all be generated if necessary)
    '''
    data = treg.Struct(
        {
            # FastArray test data
            'fastarray_bool': TestData.rand_bool,

            # Categorical test data
            # 'categorical_all': TestData.categorical_all,
            'categorical_bytes': TestData.categorical_bytes,
            'categorical_firstorder': TestData.categorical_firstorder,
            'categorical_multikey': TestData.categorical_multikey,
            'categorical_numeric': TestData.categorical_numeric,
            'categorical_unicode': TestData.categorical_unicode,
            'categorical_zero': TestData.categorical_zero,

            # Dataset test data
            'dataset_numeric': TestData.dataset_numeric,
            'dataset_strings': TestData.dataset_strings,
            'dataset_rt_types': TestData.dataset_rt_types,
            'dataset_for_gb': TestData.dataset_for_gb,

            # Struct test data
            # TODO enable struct_nested after fixing Struct.equality check
            # 'struct_nested': TestData.struct_nested,
        }
    )
    return data
