import warnings
import numpy as np
import unittest
import pytest
import sys
import riptable as rt
import riptide_cpp as rc
from numpy.testing import assert_equal
from contextlib import contextmanager
from typing import List, Callable
from riptable import FastArray
from riptable.rt_enum import (
    gBinaryUFuncs,
    gUnaryUFuncs,
    gBinaryLogicalUFuncs,
    INVALID_DICT,
    MATH_OPERATION,
)
from riptable.rt_utils import mbget
from riptable.rt_numpy import isnan, arange, issorted, nanmax, nanmin, isnotfinite, isnotnan
from riptable.tests.utils import new_array_function

NP_ARRAY_FUNCTION_PARAMS: List[Callable] = [
    np.argmax,
    np.nanargmax,
    np.argmin,
    np.nanargmin,
    np.max,
    np.nanmax,
    np.mean,
    np.nanmean,
    np.min,
    np.nanmin,
    np.std,
    np.nanstd,
    np.sum,
    np.nansum,
    np.var,
    np.nanvar,
]


# not used for these tests
# these sizes were chosen because of threading thresholds beneath the hood
# check array sizes right on the cusp of triggering a different path
array_sizes = [
    1,
    8,
    17,
    (2 * 65536) - 1,
    (2 * 65536),
    (2 * 65536) + 1,
    (65536 * 200),
    (65536 * 200) - 1,
    (65536 * 200) + 1,
]

# not used for these tests
# these will break a lot of ufuncs because of division by zero, nans, etc.
# TODO: add unit tests to make sure they break correctly
numeric_types_with_invalid = {
    'bool': FastArray([1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0], dtype=np.bool),
    'int8': FastArray(
        [
            26,
            1,
            -9,
            INVALID_DICT[np.dtype('int8').num],
            13,
            INVALID_DICT[np.dtype('int8').num],
            26,
            5,
            26,
            -4,
            27,
            28,
            30,
            -32,
            15,
        ],
        dtype=np.int8,
    ),
    'uint8': FastArray(
        [
            45,
            57,
            60,
            49,
            19,
            29,
            18,
            1,
            62,
            INVALID_DICT[np.dtype('uint8').num],
            55,
            47,
            31,
            INVALID_DICT[np.dtype('uint8').num],
            27,
        ],
        dtype=np.uint8,
    ),
    'int16': FastArray(
        [
            -601,
            -332,
            162,
            375,
            160,
            -357,
            -218,
            -673,
            INVALID_DICT[np.dtype('int16').num],
            378,
            -175,
            -529,
            INVALID_DICT[np.dtype('int16').num],
            -796,
            -365,
        ],
        dtype=np.int16,
    ),
    'uint16': FastArray(
        [
            1438,
            1723,
            433,
            1990,
            INVALID_DICT[np.dtype('uint16').num],
            1528,
            1124,
            42,
            1316,
            1003,
            1874,
            INVALID_DICT[np.dtype('uint16').num],
            1533,
            1443,
            1170,
        ],
        dtype=np.uint16,
    ),
    'int32': FastArray(
        [
            1896652134,
            -1424042309,
            INVALID_DICT[np.dtype('int32').num],
            503239478,
            1067866129,
            -1974125613,
            -1608929297,
            -301645171,
            1402604369,
            INVALID_DICT[np.dtype('int32').num],
            1080040975,
            -289078200,
            -823277029,
            -1383139138,
            978724859,
        ],
        dtype=np.int32,
    ),
    'uint32': FastArray(
        [
            337083591,
            688548370,
            INVALID_DICT[np.dtype('uint32').num],
            580206095,
            328423284,
            211281118,
            912676658,
            132565336,
            399918307,
            425384120,
            723039073,
            252319702,
            750186713,
            197297577,
            INVALID_DICT[np.dtype('uint32').num],
        ],
        dtype=np.uint32,
    ),
    'int64': FastArray(
        [
            INVALID_DICT[np.dtype('int64').num],
            -423272446,
            -235992796,
            -1442995093,
            INVALID_DICT[np.dtype('int64').num],
            109846344,
            -1458628816,
            232007889,
            -1671608168,
            1752532663,
            -1545252943,
            544588670,
            -1385051680,
            -137319813,
            -195616592,
        ],
        dtype=np.int64,
    ),
    'uint64': FastArray(
        [
            765232401,
            398653552,
            203749209,
            288238744,
            INVALID_DICT[np.dtype('uint64').num],
            271583764,
            985270266,
            391578626,
            196046134,
            916025221,
            694962930,
            34303390,
            647346354,
            INVALID_DICT[np.dtype('uint64').num],
            334977533,
        ],
        dtype=np.uint64,
    ),
    'float32': FastArray(
        [
            np.nan,
            0.6201803850883267,
            0.05285394972525459,
            0.1435023986327576,
            np.nan,
            0.32308353808130397,
            0.1861463881422203,
            0.6366386808076959,
            0.7703864299590418,
            0.8155206130668257,
            0.9588669164271945,
            0.2832984888482334,
            0.02662158289064087,
            0.2591740277624228,
            0.28945199094333374,
        ]
    ).astype(np.float32),
    'float64': FastArray(
        [
            0.264105510380617,
            np.nan,
            0.9094594817708785,
            0.13757414135018453,
            0.9997438463622871,
            0.1642171078246103,
            0.4883940875811662,
            0.2819313242616074,
            0.7868397473215173,
            0.8963052108412053,
            0.03571507605557389,
            0.6423436033517553,
            0.04402603090628798,
            0.5619514123321582,
            np.nan,
        ]
    ).astype(np.float64),
}

# not used for these tests
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=UserWarning)
    all_types = {
        'bool': FastArray([1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0], dtype=np.bool),
        'int8': FastArray(
            [
                26,
                1,
                -9,
                INVALID_DICT[np.dtype('int8').num],
                13,
                INVALID_DICT[np.dtype('int8').num],
                26,
                5,
                26,
                -4,
                27,
                28,
                30,
                -32,
                15,
            ],
            dtype=np.int8,
        ),
        'uint8': FastArray(
            [
                45,
                57,
                60,
                49,
                19,
                29,
                18,
                1,
                62,
                INVALID_DICT[np.dtype('uint8').num],
                55,
                47,
                31,
                INVALID_DICT[np.dtype('uint8').num],
                27,
            ],
            dtype=np.uint8,
        ),
        'int16': FastArray(
            [
                -601,
                -332,
                162,
                375,
                160,
                -357,
                -218,
                -673,
                INVALID_DICT[np.dtype('int16').num],
                378,
                -175,
                -529,
                INVALID_DICT[np.dtype('int16').num],
                -796,
                -365,
            ],
            dtype=np.int16,
        ),
        'uint16': FastArray(
            [
                1438,
                1723,
                433,
                1990,
                INVALID_DICT[np.dtype('uint16').num],
                1528,
                1124,
                42,
                1316,
                1003,
                1874,
                INVALID_DICT[np.dtype('uint16').num],
                1533,
                1443,
                1170,
            ],
            dtype=np.uint16,
        ),
        'int32': FastArray(
            [
                1896652134,
                -1424042309,
                INVALID_DICT[np.dtype('int32').num],
                503239478,
                1067866129,
                -1974125613,
                -1608929297,
                -301645171,
                1402604369,
                INVALID_DICT[np.dtype('int32').num],
                1080040975,
                -289078200,
                -823277029,
                -1383139138,
                978724859,
            ],
            dtype=np.int32,
        ),
        'uint32': FastArray(
            [
                337083591,
                688548370,
                INVALID_DICT[np.dtype('uint32').num],
                580206095,
                328423284,
                211281118,
                912676658,
                132565336,
                399918307,
                425384120,
                723039073,
                252319702,
                750186713,
                197297577,
                INVALID_DICT[np.dtype('uint32').num],
            ],
            dtype=np.uint32,
        ),
        'int64': FastArray(
            [
                INVALID_DICT[np.dtype('int64').num],
                -423272446,
                -235992796,
                -1442995093,
                INVALID_DICT[np.dtype('int64').num],
                109846344,
                -1458628816,
                232007889,
                -1671608168,
                1752532663,
                -1545252943,
                544588670,
                -1385051680,
                -137319813,
                -195616592,
            ],
            dtype=np.int64,
        ),
        'uint64': FastArray(
            [
                765232401,
                398653552,
                203749209,
                288238744,
                INVALID_DICT[np.dtype('uint64').num],
                271583764,
                985270266,
                391578626,
                196046134,
                916025221,
                694962930,
                34303390,
                647346354,
                INVALID_DICT[np.dtype('uint64').num],
                334977533,
            ],
            dtype=np.uint64,
        ),
        'float32': FastArray(
            [
                np.nan,
                0.6201803850883267,
                0.05285394972525459,
                0.1435023986327576,
                np.nan,
                0.32308353808130397,
                0.1861463881422203,
                0.6366386808076959,
                0.7703864299590418,
                0.8155206130668257,
                0.9588669164271945,
                0.2832984888482334,
                0.02662158289064087,
                0.2591740277624228,
                0.28945199094333374,
            ]
        ).astype(np.float32),
        'float64': FastArray(
            [
                0.264105510380617,
                np.nan,
                0.9094594817708785,
                0.13757414135018453,
                0.9997438463622871,
                0.1642171078246103,
                0.4883940875811662,
                0.2819313242616074,
                0.7868397473215173,
                0.8963052108412053,
                0.03571507605557389,
                0.6423436033517553,
                0.04402603090628798,
                0.5619514123321582,
                np.nan,
            ]
        ).astype(np.float64),
        'bytes': FastArray(
            [
                INVALID_DICT[np.dtype('bytes').num],
                b'12398dfkw',
                b'dlkv;lk3-2',
                b'111dkjfj3',
                b'e0383hjfns',
                b'qwernvldkj',
                b'abefgkejf',
                b'as777nrn',
                b'23dhsjkifuywfwefj',
                INVALID_DICT[np.dtype('bytes').num],
                b'zkdfjlw',
                b'a',
                br';][{}[\|||+=_-',
                b'qwernvldkj',
                b'abefgkejf',
            ],
            dtype=np.bytes_,
        ),
        'unicode': FastArray(
            [
                '\u2081asdf233rf',
                '12398dfkw',
                'dlkv;lk3-2',
                '111dkjfj3',
                'e038\u20813hjfns',
                INVALID_DICT[np.dtype('str_').num],
                'abefgkejf',
                'as777nrn',
                '23dhsjk\u2081ifuywfwefj',
                'rrrrn2fhfewl',
                'zkdfjlw',
                'a',
                r';][{}[\|||+=_-',
                'qwernvldkj',
                INVALID_DICT[np.dtype('str_').num],
            ],
            dtype=np.str_,
        ),
    }


# ------------------------DATA FOR UNIT TESTS----------------------------------
def static_unary(a, b, func):
    return func(a) + func(b)


def static_binary(a, b, func):
    return func(a, b)


func_sets = {
    "binary_funcs": [key for key, value in gBinaryUFuncs.items() if value is not None],
    "unary_funcs": [key for key, value in gUnaryUFuncs.items() if value is not None],
    "logical_funcs": list(gBinaryLogicalUFuncs.keys()),  # we hook all of these
}

#remove power for darwin - the test will fail because different compilers have different overflows for small ints like uint16
if sys.platform == 'darwin':
    bfuncs = func_sets["binary_funcs"]
    newbfuncs = [key for key in bfuncs if key is not np.power]
    warnings.warn('power removed for MacOS test')
    func_sets["binary_funcs"] = newbfuncs

drivers = {
    "binary_funcs": static_binary,
    "unary_funcs": static_unary,
    "logical_funcs": static_binary,
}

# size, lower bound for each array
arr_size = 100
low = 1
# upper bound for each type
max_int8 = 127
max_uint8 = 255
max_int16 = 32767
max_uint16 = 65535
max_int32 = 2147483647
max_uint32 = 0xFFFFFFFF
max_int64 = 9223372036854775807
max_uint64 = 0xFFFFFFFFFFFFFFFF
max_float32 = 2e50
max_float64 = 2e50

numeric_types_large = {
    'int8': np.random.randint(low, max_int8, arr_size, dtype=np.int8).view(FastArray),
    'uint8': np.random.randint(low, max_uint8, arr_size, dtype=np.uint8).view(
        FastArray
    ),
    'int16': np.random.randint(low, max_int16, arr_size, dtype=np.int16).view(
        FastArray
    ),
    'uint16': np.random.randint(low, max_uint16, arr_size, dtype=np.uint16).view(
        FastArray
    ),
    'int32': np.random.randint(low, max_int32, arr_size, dtype=np.int32).view(
        FastArray
    ),
    'uint32': np.random.randint(low, max_uint32, arr_size, dtype=np.uint32).view(
        FastArray
    ),
    'int64': np.random.randint(low, max_int64, arr_size, dtype=np.int64).view(
        FastArray
    ),
    'uint64': np.random.randint(low, max_uint64, arr_size, dtype=np.uint64).view(
        FastArray
    ),
    'float32': (max_uint64 * np.random.rand(arr_size))
    .astype(np.float32)
    .view(FastArray),
    'float64': (max_uint64 * np.random.rand(arr_size))
    .astype(np.float64)
    .view(FastArray),
}

small_val = 10
numeric_types_small = {
    'int8': np.random.randint(low, small_val, arr_size, dtype=np.int8).view(FastArray),
    'uint8': np.random.randint(low, small_val, arr_size, dtype=np.uint8).view(
        FastArray
    ),
    'int16': np.random.randint(low, small_val, arr_size, dtype=np.int16).view(
        FastArray
    ),
    'uint16': np.random.randint(low, small_val, arr_size, dtype=np.uint16).view(
        FastArray
    ),
    'int32': np.random.randint(low, small_val, arr_size, dtype=np.int32).view(
        FastArray
    ),
    'uint32': np.random.randint(low, small_val, arr_size, dtype=np.uint32).view(
        FastArray
    ),
    'int64': np.random.randint(low, small_val, arr_size, dtype=np.int64).view(
        FastArray
    ),
    'uint64': np.random.randint(low, small_val, arr_size, dtype=np.uint64).view(
        FastArray
    ),
    #'float32'   : (small_val * np.random.rand(arr_size)).astype(np.float32).view(FastArray),
    #'float64'   : (small_val * np.random.rand(arr_size)).astype(np.float64).view(FastArray)
}

numeric_types_small_vs_numpy = {
    'int8': np.random.randint(low, small_val, arr_size, dtype=np.int8).view(FastArray),
    'uint8': np.random.randint(low, small_val, arr_size, dtype=np.uint8).view(
        FastArray
    ),
    'int16': np.random.randint(low, small_val, arr_size, dtype=np.int16).view(
        FastArray
    ),
    'uint16': np.random.randint(low, small_val, arr_size, dtype=np.uint16).view(
        FastArray
    ),
    'int32': np.random.randint(low, small_val, arr_size, dtype=np.int32).view(
        FastArray
    ),
    'uint32': np.random.randint(low, small_val, arr_size, dtype=np.uint32).view(
        FastArray
    ),
    'int64': np.random.randint(low, small_val, arr_size, dtype=np.int64).view(
        FastArray
    ),
    'uint64': np.random.randint(low, small_val, arr_size, dtype=np.uint64).view(
        FastArray
    ),
    #'float32'   : (small_val * np.random.rand(arr_size)).astype(np.float32).view(FastArray),
    #'float64'   : (small_val * np.random.rand(arr_size)).astype(np.float64).view(FastArray)
}

# change this to set a dictionary of data
numeric_types = numeric_types_small


@contextmanager
def disable_class_member(kls: type, member_name: str) -> None:
    """
    Provides a context with the member removed from the class definition.

    Parameters
    ----------
    kls: type
        The class to remove the `member_name` from.
    member_name: str
        The member name to remove from the class definition.

    Raises
    ------
    AttributeError
        If the `member_name` is not defined in `kls`.
    """
    member = getattr(kls, member_name)
    try:
        # remove the class member and return control to caller
        delattr(kls, member_name)
        yield
    finally:
        # set the original class member after caller is finished doing work
        setattr(kls, member_name, member)

# -------------------------------------------------------------------------------------


class FastArray_UFunc_Test(unittest.TestCase):
    '''
    *** these tests DO NOT pass with every numeric_types dictionary above
    There are several overflow runtime warnings
    Bug in accuracy check
    '''

    def test_numeric_types(self):
        error_log = []
        # binary, unary, logical ufunc lists
        for set_name, ufunc_list in func_sets.items():
            driver = drivers[set_name]
            for f in ufunc_list:
                # mix/match all numeric datatypes
                for dt, arr in numeric_types.items():
                    for dt2, arr2 in numeric_types.items():
                        try:
                            driver(arr, arr2, f)
                        except Exception as e:
                            # log function name, both datatypes, exception name
                            error_log.append(
                                (f.__name__, dt, dt2, e.__class__.__name__)
                            )

        self.assertEqual(len(error_log), 0)
        # print(error_log)

    # -------------------------------------------------------------------------------------

    def test_vs_numpy(self):
        accuracy_threshold = 0.001
        crc_failures = []
        accuracy_failures = []
        # binary, unary, logical ufunc lists
        for set_name, ufunc_list in func_sets.items():
            driver = drivers[set_name]
            # print('ufunc_list was',ufunc_list)
            # continue
            for f in ufunc_list:
                # mix/match all numeric datatypes
                for dt, arr in numeric_types_small_vs_numpy.items():
                    # remove uints from negative operation
                    try:
                        ftest = gUnaryUFuncs[f]
                        if ftest == MATH_OPERATION.NEGATIVE:
                            if dt.startswith('u'):
                                continue
                    except:
                        pass
                    for dt2, arr2 in numeric_types_small_vs_numpy.items():
                        # remove uints from negative operation
                        try:
                            ftest = gUnaryUFuncs[f]
                            if ftest == MATH_OPERATION.NEGATIVE:
                                if dt2.startswith('u'):
                                    continue
                        except:
                            pass
                        fa_result = driver(arr, arr2, f)
                        np_result = driver(
                            arr.view(np.ndarray), arr2.view(np.ndarray), f
                        )

                        # first, perform a CRC check
                        fa_crc = rc.CalculateCRC(fa_result)
                        np_crc = rc.CalculateCRC(np_result)
                        if fa_crc != np_crc:
                            fail_tup = (f.__name__, dt, dt2)
                            crc_failures.append(fail_tup)
                            # sfw uses different casting rules than numpy
                            # CRC checks may fail becacuse of precision differences
                            percent_diff = np.nansum(fa_result) / np.nansum(np_result)
                            if abs(1 - percent_diff) > accuracy_threshold:
                                print(
                                    f'percent diff was',
                                    percent_diff,
                                    '  driver:',
                                    driver,
                                )
                                print(
                                    f"**Failure for func {f!r} {arr.dtype} {arr2.dtype} {arr} {arr2}"
                                )
                                print(f'\nFA:{fa_result}\n NP:{np_result}')
                                accuracy_failures.append(fail_tup)
        # TODO move print messages to assert messages
        self.assertEqual(len(accuracy_failures), 0)

        # print("CRC failures:")
        # for c in crc_failures: print(c)
        # print("Accuracy failures:")
        # for a in accuracy_failures: print(a)


class FastArray_Test(unittest.TestCase):
    # def test_col_ctor_01(self):
    #    for is_uc, arr1 in ((False, np.arange(-5, 22, 3)), (False, np.random.random(10)),
    #                        (False, np.random.random((5, 10))),
    #                        (False, np.array('banana apple pear kumquat'.split())),
    #                        (True, np.array(['a', 'μεαν'])),
    #                        (True, np.array(['μεαν', 'b']))):
    #        if is_uc:
    #            with self.assertWarns(UserWarning):
    #                fa1 = FastArray(arr1)
    #            with self.assertWarns(UserWarning):
    #                fa2 = FastArray(list(arr1))
    #        else:
    #            if arr1.ndim > 1:
    #                with self.assertWarns(UserWarning):
    #                    fa1 = FastArray(arr1)
    #                    fa2 = FastArray(list(arr1))
    #            else:
    #                fa1 = FastArray(arr1)
    #                fa2 = FastArray(list(arr1))
    #        if fa1.dtype.char == 'S' and arr1.dtype.char == 'U':
    #            with self.assertRaises(TypeError):
    #               _ = fa1 == arr1
    #        else:
    #            self.assertTrue((fa1 == arr1).all())

    #        if fa1.dtype.char == 'S' and arr1.dtype.char == 'U':
    #            with self.assertRaises(TypeError):
    #               _ = fa2 == arr1
    #        else:
    #            self.assertTrue((fa2 == arr1).all())
    #        self.assertEqual(fa1.shape, arr1.shape)
    #        self.assertEqual(fa2.shape, arr1.shape)

    # def test_col_ctor_02(self):
    #    for is_uc, val0 in ((False, 'a string'), (True, 'μεαν')):
    #        if is_uc:
    #            with self.assertWarns(UserWarning):
    #                fa0 = FastArray(val0)
    #        else:
    #            fa0 = FastArray(val0)
    #        self.assertEqual(bytes_to_str(fa0[0]), val0)
    #        self.assertEqual(fa0.shape, (1,))
    #    for val1 in (5, 5.6):
    #        fa1 = FastArray(val1)
    #        self.assertEqual(fa1[0], val1)
    #        self.assertEqual(fa1.shape, (1,))
    #    for val2 in (slice(None), None, {'A': 1, 'B': 2}, iter('an iterator')):
    #        with self.assertRaises(TypeError):
    #            fa2 = FastArray(val2)

    def test_col_ctor_03(self):
        # Want to make sure that it does not rely on first elements to decide on promotion.
        fa1 = FastArray([5.6, 'a string', 5])
        self.assertIsInstance(fa1[0], bytes)
        self.assertIsInstance(FastArray([5.6, 5])[0], np.floating)
        self.assertIsInstance(FastArray([5, 5.6])[0], np.floating)
        self.assertIsInstance(FastArray([5, 6])[0], np.integer)
        # Want to iterate over inputs different types to make sure the fastarrays are of the correct type
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            for xtype in (
                np.int8,
                np.int16,
                np.int32,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.float16,
                np.float32,
                np.float64,
            ):
                self.assertIsInstance(FastArray([xtype(5.67)])[0], xtype)

    def test_col_ctor_04(self):
        arr1 = np.random.random((5, 10))
        with self.assertWarns(UserWarning):
            fa1 = FastArray(arr1)
        self.assertTrue((fa1 == arr1).all())
        fa1[3, 4] *= -1
        self.assertTrue((fa1 == arr1).all())
        arr1[4, 3] *= -1
        self.assertTrue((fa1 == arr1).all())
        fa2 = fa1.copy()
        fa2[3, 4] *= -1
        self.assertFalse((fa2 == arr1).all())
        arr1[4, 3] *= -1
        self.assertFalse((fa2 == arr1).all())

    def test_astype(self):
        arr1 = np.random.random((5, 10))

        with self.assertWarns(UserWarning):
            fa1 = FastArray(arr1)
        self.assertTrue(issubclass(fa1.dtype.type, np.floating))

        fa2 = fa1.astype(int)
        self.assertIsInstance(fa2, FastArray)
        self.assertTrue(issubclass(fa2.dtype.type, np.integer))
        # Assert that .astype() preserves the C_CONTIGUOUS and F_CONTIGUOUS flags from the input if/when they're set.
        # This is a material conditional check (i.e. "p implies q")
        for flag_name in ['C_CONTIGUOUS', 'F_CONTIGUOUS']:
            assert not fa1.flags[flag_name] or fa2.flags[flag_name]
        self.assertEqual(int(arr1[3, 6]), fa2[3, 6])

        fa3 = fa1.astype(str)
        self.assertTrue(issubclass(fa3.dtype.type, (str, np.str_)))
        self.assertEqual(
            str(arr1[3, 6])[:8], fa3[3, 6][:8]
        )  # crude handling of rounding errors

    def test_astype_transpose_bug(self):
        # TODO: Also check that if we start with an F_CONTIGUOUS array, that flag is preserved in the output of astype().
        #       Also check that if we start with an array that's *both* C_CONTIGUOUS and F_CONTIGUOUS, those properties are preserved.
        orig = FastArray([[1, 2], [3, 4]])
        self.assertTrue(orig.flags['C_CONTIGUOUS'])

        result = orig.astype(np.float64)
        # Assert that .astype() preserves the C_CONTIGUOUS and F_CONTIGUOUS flags from the input if/when they're set.
        # This is a material conditional check (i.e. "p implies q")
        for flag_name in ['C_CONTIGUOUS', 'F_CONTIGUOUS']:
            assert not orig.flags[flag_name] or result.flags[flag_name]

    def test_indexing(self):
        arr1 = np.random.random((5, 10))
        with self.assertWarns(UserWarning):
            fa1 = FastArray(arr1)
        self.assertEqual(fa1[3, 6], arr1[3, 6])
        self.assertEqual(fa1[-3, 6], arr1[-3, 6])
        self.assertEqual(fa1[3, -6], arr1[3, -6])
        self.assertEqual(fa1[-3, -6], arr1[-3, -6])
        with self.assertRaises(IndexError):
            _r, _c = fa1.shape
            _ = fa1[_r, _c]
        arr2 = np.random.random(10)
        fa2 = FastArray(arr2)
        self.assertEqual(fa2[6], arr2[6])
        self.assertEqual(fa2[-6], arr2[-6])
        with self.assertRaises(IndexError):
            _r = fa2.shape
            _ = fa2[_r]

    def test_comparisons(self):
        arr1 = np.random.random(10)
        arr2 = np.random.random(10)
        fa1 = FastArray(arr1)
        fa2 = FastArray(arr2)
        for lhs, rhs in ((fa1, fa2), (fa1, arr2), (arr1, fa2)):
            self.assertTrue(((lhs < rhs) == (arr1 < arr2)).all())
            self.assertTrue(((lhs <= rhs) == (arr1 <= arr2)).all())
            self.assertTrue(((lhs == rhs) == (arr1 == arr2)).all())
            self.assertTrue(((lhs != rhs) == (arr1 != arr2)).all())
            self.assertTrue(((lhs >= rhs) == (arr1 >= arr2)).all())
            self.assertTrue(((lhs > rhs) == (arr1 > arr2)).all())

    def test_arithmetic(self):
        arr1 = np.random.random((5, 10))
        arr2 = np.random.random((5, 10))
        with self.assertWarns(UserWarning):
            fa1 = FastArray(arr1)
            fa2 = FastArray(arr2)
        for lhs, rhs in ((fa1, fa2), (fa1, arr2), (arr1, fa2)):
            self.assertTrue(((lhs + rhs) == (arr1 + arr2)).all())
            self.assertTrue(((lhs - rhs) == (arr1 - arr2)).all())
            self.assertTrue(((lhs * rhs) == (arr1 * arr2)).all())
            self.assertTrue(((lhs / rhs) == (arr1 / arr2)).all())
        self.assertTrue(((fa1 ** 5) == (arr1 ** 5)).all())
        self.assertTrue(((fa1 ** -0.5) == (arr1 ** -0.5)).all())

    def test_nanfunctions(self):
        accuracy = 7
        for func in (
            'sum',
            'mean',
            'min',
            'max',
        ):
            nn = np.random.random(10)
            ff = FastArray(nn)
            ff_func = getattr(ff, func)
            np_func = getattr(np, func)
            ff_nanfunc = getattr(ff, f'nan{func}')
            np_nanfunc = getattr(np, f'nan{func}')
            self.assertAlmostEqual(
                ff_func(), np_func(nn), places=accuracy, msg=f'Failure in {func}.'
            )
            self.assertAlmostEqual(
                np_func(ff), np_func(nn), places=accuracy, msg=f'Failure in {func}.'
            )
            self.assertAlmostEqual(
                np_nanfunc(ff),
                np_nanfunc(nn),
                places=accuracy,
                msg=f'Failure in {func}.',
            )
            self.assertAlmostEqual(
                ff_nanfunc(), np_nanfunc(nn), places=accuracy, msg=f'Failure in {func}.'
            )
            nn[5] = np.nan
            self.assertTrue(np.isnan(ff[5]))
            # ff = FastArray(nn)
            if func not in {'min', 'max'}:  # non-polluting
                self.assertTrue(np.isnan(np_func(ff)), msg=f'Failure in {func}.')
                self.assertTrue(np.isnan(ff_func()), msg=f'Failure in {func}.')
            self.assertAlmostEqual(
                np_nanfunc(ff),
                np_nanfunc(nn),
                places=accuracy,
                msg=f'Failure in {func}.',
            )
            self.assertAlmostEqual(
                ff_nanfunc(), np_nanfunc(nn), places=accuracy, msg=f'Failure in {func}.'
            )

        for func in (
            'var',
            'std',
        ):
            nn = np.random.random(10)
            ff = FastArray(nn)
            ff_func = getattr(ff, func)
            np_func = getattr(np, func)
            ff_nanfunc = getattr(ff, f'nan{func}')
            np_nanfunc = getattr(np, f'nan{func}')
            self.assertAlmostEqual(
                ff_func(),
                np_func(nn, ddof=1),
                places=accuracy,
                msg=f'Failure in {func}.',
            )
            self.assertAlmostEqual(
                np_func(ff), np_func(nn), places=accuracy, msg=f'Failure in {func}.'
            )
            self.assertAlmostEqual(
                np_nanfunc(ff),
                np_nanfunc(nn),
                places=accuracy,
                msg=f'Failure in {func}.',
            )
            self.assertAlmostEqual(
                ff_nanfunc(),
                np_nanfunc(nn, ddof=1),
                places=accuracy,
                msg=f'Failure in {func}.',
            )
            nn[5] = np.nan
            self.assertTrue(np.isnan(ff[5]))
            # ff = FastArray(nn)
            self.assertTrue(np.isnan(np_func(ff)), msg=f'Failure in {func}.')
            self.assertTrue(np.isnan(ff_func()), msg=f'Failure in {func}.')
            self.assertAlmostEqual(
                np_nanfunc(ff),
                np_nanfunc(nn),
                places=accuracy,
                msg=f'Failure in {func}.',
            )
            self.assertAlmostEqual(
                ff_nanfunc(),
                np_nanfunc(nn, ddof=1),
                places=accuracy,
                msg=f'Failure in {func}.',
            )

    def test_nanfunctions_var_bug(self):
        if FastArray.var == FastArray.nanvar:
            warnings.warn('FastArray.var is temp. hacked to map to FastArray.nanvar.')
        accuracy = 7
        nn = np.array(
            [
                0.18841,
                0.75549,
                0.14519,
                0.61485,
                0.52652,
                0.0,
                0.62643,
                0.49881,
                0.58673,
                0.53397,
            ]
        )
        ff = FastArray(nn)
        self.assertAlmostEqual(
            ((ff - ff.mean()) ** 2).sum() / (len(ff) - 1),
            np.var(nn, ddof=1),
            places=accuracy,
        )
        self.assertAlmostEqual(
            ((ff - ff.mean()) ** 2).sum() / (len(ff) - 1),
            ff.var(ddof=1),
            places=accuracy,
            msg='FastArray.var()',
        )

    def test_nanfunctions_var_bug_2(self):
        # In summary:
        accuracy = 7
        nn = np.array(
            [
                0.18841,
                0.75549,
                0.14519,
                0.61485,
                0.52652,
                0.0,
                0.62643,
                0.49881,
                0.58673,
                0.53397,
            ]
        )
        ff = FastArray(nn)
        self.assertAlmostEqual(
            np.var(ff, ddof=1), np.var(nn, ddof=1)
        )  # ddof=1 is hard-coded for fast-array
        self.assertAlmostEqual(ff.var(), ff.nanvar())

    def test_nanfunctions_std_bug(self):
        if FastArray.std == FastArray.nanstd:
            warnings.warn('FastArray.std is temp. hacked to map to FastArray.nanstd.')
        accuracy = 7
        nn = np.array(
            [
                0.18841,
                0.75549,
                0.14519,
                0.61485,
                0.52652,
                0.0,
                0.62643,
                0.49881,
                0.58673,
                0.53397,
            ]
        )
        ff = FastArray(nn)
        self.assertAlmostEqual(
            np.sqrt(((ff - ff.mean()) ** 2).sum() / (len(ff) - 1)),
            np.std(nn, ddof=1),
            places=accuracy,
        )
        self.assertAlmostEqual(
            np.sqrt(((ff - ff.mean()) ** 2).sum() / (len(ff) - 1)),
            ff.std(ddof=1),
            places=accuracy,
            msg='FastArray.std()',
        )

    def test_nanfunctions_std_bug_2(self):
        # In summary:
        accuracy = 7
        nn = np.array(
            [
                0.18841,
                0.75549,
                0.14519,
                0.61485,
                0.52652,
                0.0,
                0.62643,
                0.49881,
                0.58673,
                0.53397,
            ]
        )
        ff = FastArray(nn)
        self.assertAlmostEqual(
            np.std(ff, ddof=1), np.std(nn, ddof=1)
        )  # ddof=1 is hard-coded for fast-array
        self.assertAlmostEqual(ff.std(), ff.nanstd())

    def test_isnan_sentinel(self):
        dts = [
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
        arr = FastArray([-128, 2, 3, 4, 5], dtype=np.int8)
        correct = [True, False, False, False, False]

        for dt in dts:
            arr = arr.astype(dt)
            if not isnan(INVALID_DICT[arr.dtype.num]):
                self.assertEqual(INVALID_DICT[arr.dtype.num], arr[0])
            result = arr.isnan()
            self.assertTrue(
                bool(np.all(result == correct)),
                msg=f"isnan failed for dtype {np.dtype(dt)}",
            )

    def test_safe_conversions_uint(self):
        uint_types = [np.uint8, np.uint16, np.uint32, np.uint64]
        int_types = [np.int8, np.int16, np.int32, np.int64]
        # TEST REMOVED until flag can be set in C code to flip conversion
        # FastArray.SafeConversions = False
        # for udt in uint_types:
        #    dtnum = np.dtype(udt).num
        #    inv = INVALID_DICT[dtnum]
        #    arr = FastArray([inv], dtype=udt)
        #    for idt in int_types:
        #        dt2num = np.dtype(idt).num
        #        inv2 = INVALID_DICT[dt2num]
        #        arr2 = FastArray([inv2], dtype=idt)

        #    result_fa = np.negative(arr) + np.negative(arr2)
        #    result_np = np.negative(arr._np) + np.negative(arr2._np)

        #    self.assertEqual(result_fa[0], result_np[0])

        # FastArray.SafeConversions = True

    def test_empty(self):
        a = FastArray([])
        b = a * 100
        c = 100 * a
        self.assertEqual(b.shape[0], 0)
        self.assertEqual(c.shape[0], 0)

    def test_mbget_multidim(self):
        sz = 17 * 31
        arr = np.random.rand(sz)
        arr = arr.reshape((17, 31))
        idx = np.random.randint(0, 17, 100)

        result_np = arr[idx]
        result_rt = mbget(arr, idx)

        self.assertTrue(bool(np.all(result_np == result_rt)))
        self.assertTrue(isinstance(result_rt, FastArray))

    def test_mbget_negative(self):
        arr = arange(3)
        result = mbget(arr, [-1])[0]
        self.assertEqual(result, 2)

    def test_overlappedarrays(self):
        a = arange(21).reshape((7, 3), order='F')
        b = arange(21).reshape((7, 3), order='F')._np
        a[0][1:] -= a[0][:-1]
        b[0][1:] -= b[0][:-1]
        self.assertTrue(bool(np.all(a == b)))

    def test_fill_forward(self):
        arr = arange(10.0)
        arr[3] = np.nan
        arr[4] = np.nan
        arr[7] = np.nan
        a = arr.copy()
        b = arr.astype(np.int16)

        x = a.fill_forward(inplace=False)
        self.assertTrue(x[3] == 2.0)
        self.assertTrue(x[7] == 6.0)
        a.fill_forward(inplace=True, limit=1)
        self.assertTrue(a[3] == 2.0)
        self.assertTrue(a[4] != a[4])  # its a nan
        self.assertTrue(a[7] == 6.0)

        x = b.fill_forward(inplace=False)
        self.assertTrue(x[3] == 2)
        self.assertTrue(x[7] == 6)
        b.fill_forward(inplace=True, limit=1)
        self.assertTrue(b[3] == 2)
        self.assertTrue(b[4] == -32768)
        self.assertTrue(b[7] == 6)

    def test_invalid_setting(self):
        arr = arange(10, dtype=np.int16)
        arr[3] = np.nan
        arr[[5, 6, 7]] = [np.nan, 5, np.nan]
        self.assertTrue(arr[3] == -32768)
        self.assertTrue(arr[7] == -32768)
        self.assertTrue(arr[6] == 5)
        self.assertTrue(arr[9] == 9)
        self.assertTrue(arr.isnan().sum() == 3)

    def test_upcast_int64(self):
        x = np.int64(156950363900665450)
        a = arange(10, dtype=np.int32)
        y = a + x
        self.assertTrue(y.dtype.itemsize == 8)

    def test_strided_nan(self):
        a = arange(100.0)
        a = a.reshape(10, 10)
        a[2, 3] = np.nan
        x = a[:, 3]
        self.assertTrue(np.isnan(x[2]))

    def test_contiguous_isnan(self):
        a = arange(100.0)
        a[[2, 3, 23, 34]] = np.nan
        a[[57, 83]] = np.inf
        a[[67, 73]] = -np.inf

        # add some funky looking nans
        b = a.view(np.uint64)
        b[10] = 0x7FFFFFFFFFFFFFFF
        b[20] = 0x7FFFFFFF00000000
        b[30] = 0xFFFFFFFF00000001
        b[50] = 0xFFFFFFFFFFFFFFFF

        self.assertTrue(np.sum(a.isnan()), np.sum(np.isnan(a._np)))
        self.assertTrue(np.sum(a.isnotnan()), np.sum(~np.isnan(a._np)))
        self.assertTrue(np.sum(a.isfinite()), np.sum(np.isfinite(a._np)))
        self.assertTrue(np.sum(a.isnotfinite()), np.sum(~np.isfinite(a._np)))

        a = arange(100.0, dtype=np.float32)
        a[[2, 3, 23, 34]] = np.nan
        a[[57, 83]] = np.inf
        a[[67, 73]] = -np.inf
        # add some funky looking nans
        b = a.view(np.uint32)
        b[10] = 0x7FFFFFFF
        b[20] = 0x7FFFFFF0
        b[30] = 0xFFFFFFF1
        b[50] = 0xFFFFFFFF

        self.assertTrue(np.sum(a.isnan()), np.sum(np.isnan(a._np)))
        self.assertTrue(np.sum(a.isnotnan()), np.sum(~np.isnan(a._np)))
        self.assertTrue(np.sum(a.isfinite()), np.sum(np.isfinite(a._np)))
        self.assertTrue(np.sum(a.isnotfinite()), np.sum(~np.isfinite(a._np)))

    def test_rollingsum(self):
        x = FastArray([1, 2, 3, np.nan, np.nan, np.nan])
        result = x.rolling_nansum()
        goodresult = FastArray([1.0, 3.0, 6.0, 5.0, 3.0, 0.0])
        self.assertTrue(np.all(result == goodresult))

        # int based test of invalids
        result = x.astype(np.int64).rolling_nansum()
        goodresult = FastArray([1.0, 3.0, 6.0, 5.0, 3.0, 0.0]).astype(np.int64)
        self.assertTrue(np.all(result == goodresult))

    def test_sign(self):
        x = [1, -2, 3, np.nan, np.nan, np.nan]
        result = FastArray(x).sign().astype(np.int32)
        goodresult = np.sign(x).astype(np.int32)
        self.assertTrue(np.all(result == goodresult))

    def test_set_name(self):
        a = arange(100)
        x = a.set_name('test')
        # make sure same array
        self.assertTrue(id(x) == id(a))

    def test_init_pythonset(self):
        result = FastArray({3})
        self.assertTrue(result[0] == 3)
        result = FastArray({3, 4})
        self.assertTrue(result[1] == 4)
        result = FastArray({'3', '4'})
        self.assertTrue(result[1] == '4' or result[0] == '4')

    def test_nanarg(self):
        x1 = FastArray([np.nan, 1.0, 2.0, np.nan])
        result = np.nanargmin(x1._np)
        self.assertTrue(result == x1.nanargmin())
        self.assertTrue(result == x1.astype(np.int32).nanargmin())
        result = np.nanargmax(x1._np)
        self.assertTrue(result == x1.nanargmax())
        self.assertTrue(result == x1.astype(np.int32).nanargmax())

    def test_setitem(self):
        # test when the value of setitem is strided
        y = arange(9)
        y[y < 3] = y[::3]
        self.assertTrue(y[1] == 3)

    def test_isin(self):
        self.assertFalse(FastArray(['A']).isin(['AA'])[0])

    def test_between(self):
        # test endpoint configs
        x = FastArray([0, 1, 2, 3, 4])
        self.assertTrue((x.between(1, 3, False, False) == FastArray([False, False, True, False, False])).all())
        self.assertTrue((x.between(1, 3, False, True ) == FastArray([False, False, True, True,  False])).all())
        self.assertTrue((x.between(1, 3, True,  False) == FastArray([False, True,  True, False, False])).all())
        self.assertTrue((x.between(1, 3, True,  True ) == FastArray([False, True,  True, True,  False])).all())

        # test mixing of endpoint types (scalars vs arrays)
        y = [1, 1, 1, 1, 1]
        z = [3, 3, 3, 3, 3]
        self.assertTrue((x.between(1, 3) == FastArray([False, True, True, False, False])).all())
        self.assertTrue((x.between(1, z) == FastArray([False, True, True, False, False])).all())
        self.assertTrue((x.between(y, 3) == FastArray([False, True, True, False, False])).all())
        self.assertTrue((x.between(y, z) == FastArray([False, True, True, False, False])).all())

    def test_issorted(self):
        x = arange(100_000)
        self.assertTrue(x.issorted())
        x[50_000] = 4
        self.assertTrue(x.issorted() == False)
        x = arange(100_000.0)
        self.assertTrue(x.issorted())
        x[99_999] = np.nan
        self.assertTrue(x.issorted())
        x[0] = -np.inf
        x[99_998] = np.inf
        self.assertTrue(x.issorted())
        x[50_000] = 4.0
        self.assertTrue(x.issorted() == False)
        x = np.random.randint(0, 10, 100_000)
        self.assertTrue(issorted(x) == False)

    def test_allnans(self):
        allnans = FastArray([np.nan] * 5)
        result = nanmax(allnans)
        self.assertFalse(result == result)
        result = nanmin(allnans)
        self.assertFalse(result == result)
        morenans = allnans.repeat(50_000)
        morenans[249_000] = 3
        self.assertTrue(nanmin(morenans) == 3)
        self.assertTrue(nanmax(morenans) == 3)
        morenans = morenans.astype(np.int64)
        self.assertTrue(nanmin(morenans) == 3)
        self.assertTrue(nanmax(morenans) == 3)

    def test_lastminfloat(self):
        # test deliberately puts min on last value
        # to make sure it is carried forward during all the shifting in late stages
        # of min/max calculations
        a = 1000.0 - arange(256 * 256.0)
        self.assertTrue(min(a) == -64535.0)
        self.assertTrue(nanmin(a) == -64535.0)
        a = 1000.0 - arange(256 * 256.0, dtype=np.float32)
        self.assertTrue(min(a) == -64535.0)
        self.assertTrue(nanmin(a) == -64535.0)

    def test_twodim(self):
        a = np.asarray([[1, 2,], [3, 4]]).copy(order='F')
        b = np.asarray([[2, 2], [2, 2]]).copy(order='F')
        x = a == b
        y = FastArray(a) == FastArray(b)
        self.assertTrue(x.flags == y.flags)
        self.assertTrue(np.all(x == y._np))

    def test_isclose(self):
        a = FastArray([1, 2, 3, 999], dtype='f8')
        a[1] = np.nan
        x = np.isclose(a, 999)
        self.assertTrue(np.all(x == [False, False, False, True]))

    def test_unalignedmath(self):
        mydtypes = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.int64, np.float32, np.float64]
        for dt in mydtypes:
            for length in [1, 3, 10, 1000]:
                a = arange(length, dtype=dt) + 2
                # come in directly also
                self.assertTrue(np.all(abs(a) == abs(a._np)), msg=f'step 0 func was abs direct dtype:{a.dtype}  stride:{a.strides} {abs(a)} {abs(a._np)}')
                self.assertTrue(np.all(abs(a[1:]) == abs(a[1:]._np)), msg=f'step 0 func was abs direct dtype:{a.dtype}  stride:{a.strides}')
                funclist=[np.abs, np.floor, np.ceil, np.trunc, np.round, np.isnan, np.isfinite, np.negative]
                for func in funclist:
                    self.assertTrue(np.all(func(a) == func(a._np)), msg=f'step 1 func was {func} {length} dtype:{a.dtype}  stride:{a.strides}')
                    a=a[1:]
                    self.assertTrue(np.all(func(a) == func(a._np)), msg=f'step 2 func was {func} {length} dtype:{a.dtype}  stride:{a.strides}')
                    a=a[::2]
                    self.assertTrue(np.all(func(a) == func(a._np)), msg=f'step 3 func was {func} {length} dtype:{a.dtype}  stride:{a.strides}')
                    a=a[::-1]
                    self.assertTrue(np.all(func(a) == func(a._np)), msg=f'step 4 func was {func} {length} dtype:{a.dtype}  stride:{a.strides}')

        mydtypes = [np.float32, np.float64]
        for dt in mydtypes:
            a = arange(1000.0, dtype=dt) + 2
            funclist=[np.sqrt, isnotfinite, isnotnan]
            for func in funclist:
                self.assertTrue(np.all(func(a) == func(a._np)), msg=f'step 1 func was {func} dtype:{a.dtype}  stride:{a.strides}')
                a=a[1:]
                self.assertTrue(np.all(func(a) == func(a._np)), msg=f'step 2 func was {func} dtype:{a.dtype}  stride:{a.strides}')
                a=a[::2]
                self.assertTrue(np.all(func(a) == func(a._np)), msg=f'step 3 func was {func} dtype:{a.dtype}  stride:{a.strides}')
                a=a[::-1]
                self.assertTrue(np.all(func(a) == func(a._np)), msg=f'step 4 func was {func} dtype:{a.dtype}  stride:{a.strides}')

    def test_div_scalar_by_0d_array(self):
        result = FastArray([1])[0] / FastArray([2])
        self.assertEqual(result, 0.5)

    def test_int64_comparisons(self):
        x=701389446541966656
        y=rt.asarray([x-1,x,x+1,x,x,x]).astype(np.uint64)
        z=rt.asarray([-x-1,-x,-x+1,x-1,x,x+1]).astype(np.int64)
        r = y > z
        self.assertTrue(np.all(r == [ True,  True,  True, True, False, False]))
        r = y >= z
        self.assertTrue(np.all(r == [ True,  True,  True, True, True, False]))
        r = z < y
        self.assertTrue(np.all(r == [ True,  True,  True, True, False, False]))
        r = z == y
        self.assertTrue(np.all(r == [ False,  False, False, False, True, False]))
        
# TODO: Extend the tests in the TestFastArrayNanmax / TestFastArrayNanmin classes below to cover the following cases:
#   * non-array inputs (e.g. a list or set or scalar)
#   * other FastArray subclass, e.g. Date
#       * The tests for these should be implemented in the appropriate file for each class, e.g. test_date.py
#   * other unordered FastArray subclass, e.g. DateSpan (or at least, a subclass which doesn't
#       have a normal 3-way comparison because it's not a poset).
#       * For example, rt.DateSpan should probably use Allen's interval algebra which has 13 possible outcomes.
#       * Seems like we should disallow such classes; how should we check though?
#   * 2d and 3d arrays (these will probably punt to numpy, but verify)
#       * this won't work as expected on an integer dtype array; need to detect this case and just
#         disallow it rather than returning a bad result.
#   * strided/sliced arrays
#       * for sliced arrays, we're looking for issues related to data/pointer alignment in the riptide_cpp code;
#         use a reasonable-sized array (say, 1000 elements), and check we can call the function after slicing off
#         an element from the front of the array.
#   * test the above with float, integer, and string dtypes; make sure riptable invalid values (for integers; strings too?) are respected.
#   * check that any kwargs are validated (e.g. calling with a 1D array and axis=1 is not allowed).

class TestFastArrayNanmax:
    """Tests for the member/method FastArray.nanmax()."""

    @pytest.mark.parametrize(
        "arg",
        [
            pytest.param(rt.full(100, np.nan, dtype=np.float32), id='float', marks=pytest.mark.xfail(reason="RIP-417: Warning is not currently issued for the method version of nanmax, only the module function; fix this by moving the check + warning to a shared code path."))
            # TODO: Add cases for integer, string; consider just extending the test to cover all
            #  float/integer dtypes (that we support) + a bytestring example + unicode example.
        ]
    )
    def test_allnans(self, arg: FastArray):
        # Call FastArray.nanmax() with the test input.
        # It should raise a RuntimeWarning when given an input which
        # has all NaNs **on the specified axis**.
        with pytest.warns(RuntimeWarning):
            result = arg.nanmax()

        # If given a scalar or 1D array (or some collection converted to such)
        # the result should be a NaN; for higher-rank arrays, the result should
        # be an array where one of the dimensions was collapsed and if there were
        # all NaNs along the selected axis there'll be a NaN there in the result.
        # TODO: Need to fix this to assert correctly for when FastArray.nanmax() called on a higher-rank array.
        assert rt.isnan(result)

    @pytest.mark.parametrize(
        "arg",
        [
            pytest.param(
                rt.FastArray([], dtype=np.float32), id='float',
                marks=pytest.mark.xfail(
                    reason="RIP-417: The call to riptide_cpp via the ledger returns None, which then causes the isnan() to raise a TypeError. This needs to be fixed so we raise an error like numpy (either by checking for this and raising the exception, or fixing the way the function punts to numpy."))
            # TODO: Add cases for integer, string; consider just extending the test to cover all
            #  float/integer dtypes + a bytestring example + unicode example.
        ]
    )
    def test_empty(self, arg: FastArray):
        # Call FastArray.nanmax() on an empty input -- it should raise a ValueError.
        with pytest.raises(ValueError):
            arg.nanmax()


class TestFastArrayNanmin:
    """Tests for the member/method FastArray.nanmin()."""

    @pytest.mark.parametrize(
        "arg",
        [
            pytest.param(rt.full(100, np.nan, dtype=np.float32), id='float', marks=pytest.mark.xfail(reason="RIP-417: Warning is not currently issued for the method version of nanmax, only the module function; fix this by moving the check + warning to a shared code path."))
            # TODO: Add cases for integer, string; consider just extending the test to cover all
            #  float/integer dtypes (that we support) + a bytestring example + unicode example.
        ]
    )
    def test_allnans(self, arg: FastArray):
        # Call FastArray.nanmin() with the test input.
        # It should raise a RuntimeWarning when given an input which
        # has all NaNs **on the specified axis**.
        with pytest.warns(RuntimeWarning):
            result = arg.nanmin()

        # If given a scalar or 1D array (or some collection converted to such)
        # the result should be a NaN; for higher-rank arrays, the result should
        # be an array where one of the dimensions was collapsed and if there were
        # all NaNs along the selected axis there'll be a NaN there in the result.
        # TODO: Need to fix this to assert correctly for when FastArray.nanmin() called on a higher-rank array.
        assert rt.isnan(result)

    @pytest.mark.parametrize(
        "arg",
        [
            pytest.param(
                rt.FastArray([], dtype=np.float32), id='FastArray-float',
                marks=pytest.mark.xfail(
                    reason="RIP-417: The call to riptide_cpp via the ledger returns None, which then causes the isnan() to raise a TypeError. This needs to be fixed so we raise an error like numpy (either by checking for this and raising the exception, or fixing the way the function punts to numpy."))
            # TODO: Add cases for integer, string; consider just extending the test to cover all
            #  float/integer dtypes + a bytestring example + unicode example.
        ]
    )
    def test_empty(self, arg):
        # Call rt.nanmin with an empty input -- it should raise a ValueError.
        with pytest.raises(ValueError):
            rt.nanmin(arg)


# TODO parameterize over different array data types
@pytest.mark.parametrize("np_callable", NP_ARRAY_FUNCTION_PARAMS)
@pytest.mark.parametrize("np_array", [np.array([1,2,3])])
def test_array_function_matches_numpy(np_callable, np_array):
    # test equivalence between our array function implementation against numpy array function implementation
    np_result = np_callable(np_array)

    fast_array = np_array.view(FastArray)
    with new_array_function(FastArray):
        rt_result = np_callable(fast_array)

    # TODO assert dispatching of np.min in fast array returns a _min (by convention) method
    assert_equal(rt_result, np_result)

    if isinstance(np_result, np.ndarray):
        assert isinstance(rt_result, FastArray)
    else:
        if np_callable not in [np.sum, np.nansum]:  # riptable specific behavior
            assert type(rt_result) == type(np_result)


@pytest.mark.parametrize("np_callable", NP_ARRAY_FUNCTION_PARAMS)
@pytest.mark.parametrize("fast_array", [rt.FastArray([1,2,3])])
def test_new_array_function_matches_old_array_function(np_callable, fast_array):
    # test equivalence between our array function implementation against the old array function implementation
    with new_array_function(FastArray):
        actual = np_callable(fast_array)

    expected = np_callable(fast_array)

    assert_equal(expected, actual, f'new array function implementation is inconsistent with original implementation.')
    assert type(expected) == type(actual), f'new array function result type is inconsistent with original implementation.'


@pytest.mark.parametrize("np_callable", NP_ARRAY_FUNCTION_PARAMS + list(gUnaryUFuncs.keys()) + [np.around, np.round_])
@pytest.mark.parametrize("fast_array", [rt.FastArray([1,2,3])])
def test_new_array_function_matches_ufunc(np_callable, fast_array):
    # test equivalence between our array function implementation against our ufunc implementation
    with new_array_function(FastArray):
        actual = np_callable(fast_array)

    with disable_class_member(FastArray, '__array_function__'):
        expected = np_callable(fast_array)

    assert_equal(expected, actual, f'new array function implementation is inconsistent with ufunc implementation.')
    assert type(expected) == type(actual), f'new array function result type is inconsistent with ufunc implementation.'


@pytest.mark.parametrize("np_callable", [np.empty_like])
@pytest.mark.parametrize("fast_array", [rt.FastArray([1,2,3]), np.array([1,2,3])])
def test_array_function_empty_like(np_callable, fast_array):
    actual: FastArray
    with new_array_function(FastArray):
        actual = np_callable(fast_array)
    assert type(actual) == type(fast_array), 'type mismatch'
    assert actual.shape == fast_array.shape, 'shape mismatch'


if __name__ == "__main__":
    tester = unittest.main()
