import pytest
import unittest

from riptable import *
import riptide_cpp as rc
from riptable.rt_enum import (
    NumpyCharTypes,
    gBinaryUFuncs,
    INVALID_DICT,
    DisplayLength,
    DisplayJustification,
    DisplayColumnColors,
)
from riptable import FastArray as FA
from math import isclose

float_types = [np.float32, np.float64]
int_types = [
    np.bool,
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.intc,
    np.intp,
    np.int_,
    np.int64,
    np.uint64,
]
string_types = [np.bytes_, np.str_]
warning_types = [np.object, np.complex64, np.complex128, np.float16]

# nan_funcs   = [ np.nanargmax, np.nanargmin, np.nancumprod, np.nancumsum, np.nanmax,
#                np.nanmean, np.nanmedian, np.nanmin, np.nanpercentile, np.nanprod,
#                np.nanstd, np.nansum, np.nanvar ]
# nan_results = []

nan_funcs = [nansum, nanmean, nanmin, nanmax]
no_negatives = [np.log, np.log2, np.log10, np.sqrt]

num_list = list(range(10))
str_list = ['a', 'b', 'c', 'd', 'e']

# a set is now valid
invalid_data = [slice(1, 2, 3), ..., {'a': 1}]  # add type, elipses, dict

np.random.seed(1234)


def almost_eq(fa, arr, places=5, rtol=1e-5, atol=1e-5, equal_nan=True):
    epsilon = 10 ** (-places)
    rtol = epsilon

    if fa.shape != arr.shape:
        return False

    fa_arr = fa
    np_arr = arr
    if np.issubdtype(fa_arr.dtype, np.floating):
        # N.B.: Do not compare the elements that both numpy and riptable give nan or inf.
        #       In the case of modulo by zero, numpy gives nan but riptable gives inf.
        finite_mask = np.isfinite(fa_arr) | np.isfinite(np_arr)
        if not np.allclose(
            fa_arr[finite_mask], np_arr[finite_mask], rtol, atol, equal_nan
        ):
            return False
    else:
        try:
            if not (fa_arr == np_arr).all():
                return False
        except TypeError:
            return False
    return True


class FastArray_Test(unittest.TestCase):

    # --------VALID CONSTRUCTORS-------------------
    '''
    BUGS:
    when initializing from a scalar, the requested dtype is ignored
    it will default to the type of the scalar OR the default numpy type for the python type

    Numpy does not ignore the dtype keyword arg.
    '''

    def test_float_constructors(self):
        for dt in float_types:
            arr = FastArray(num_list, dtype=dt)
            self.assertEqual(arr.dtype, dt)

        # from scalars
        for dt in float_types:
            arr = FastArray(num_list[0], dtype=dt)
            self.assertEqual(arr.dtype, dt)

    def test_int_constructors(self):
        for dt in int_types:
            arr = FastArray(num_list, dtype=dt)
            self.assertEqual(arr.dtype, dt)
        # from scalars
        for dt in int_types:
            arr = FastArray(num_list[0], dtype=dt)
            self.assertEqual(arr.dtype, dt)

    # --------INVALID CONSTRUCTORS-----------------
    def test_invalid_types(self):
        for dt in warning_types:
            with self.assertWarns(UserWarning):
                _ = FastArray([1], dtype=dt)

    def test_invalid_data(self):
        for d in invalid_data:
            with self.assertRaises(TypeError):
                _ = FastArray(d)

    ## --------TEST SFW INTEGER NAN VALUES-----------
    #'''
    # BUGS:
    # nanstd: mean considers our custom nan values, variance does not
    #'''
    def test_nan_values_int8(self):
        arr = FastArray([1, INVALID_DICT[np.dtype(np.int8).num]], dtype=np.int8)
        for f in nan_funcs:
            result = f(arr)
            self.assertEqual(result, 1)

    def test_nan_values_uint8(self):
        arr = FastArray([1, INVALID_DICT[np.dtype(np.uint8).num]], dtype=np.uint8)
        for f in nan_funcs:
            result = f(arr)
            self.assertEqual(result, 1)

    def test_nan_values_int16(self):
        arr = FastArray([1, INVALID_DICT[np.dtype(np.int16).num]], dtype=np.int16)
        for f in nan_funcs:
            result = f(arr)
            self.assertEqual(result, 1)

    def test_nan_values_uint16(self):
        arr = FastArray([1, INVALID_DICT[np.dtype(np.uint16).num]], dtype=np.uint16)
        for f in nan_funcs:
            result = f(arr)
            self.assertEqual(result, 1)

    def test_nan_values_int32(self):
        arr = FastArray([1, INVALID_DICT[np.dtype(np.int32).num]], dtype=np.int32)
        for f in nan_funcs:
            result = f(arr)
            self.assertEqual(result, 1)

    def test_nan_values_uint32(self):
        arr = FastArray([1, INVALID_DICT[np.dtype(np.uint32).num]], dtype=np.uint32)
        for f in nan_funcs:
            result = f(arr)
            self.assertEqual(result, 1)

    def test_nan_values_int64(self):
        arr = FastArray([1, INVALID_DICT[np.dtype(np.int64).num]], dtype=np.int64)
        for f in nan_funcs:
            result = f(arr)
            self.assertEqual(result, 1)

    def test_nan_values_uint64(self):
        arr = FastArray([1, INVALID_DICT[np.dtype(np.uint64).num]], dtype=np.uint64)
        for f in nan_funcs:
            result = f(arr)
            self.assertEqual(result, 1)

    # --------OVERFLOW / CASTING TESTS-----------
    def test_overflow_int64(self):
        '''
        FastArray flips overflow values in integer types to floats.
        Numpy produces an incorrect result.
        '''
        a = 0xFFFFFFFFFFFFFFF
        arr_f = FastArray([a] * 20, dtype=np.int64)
        sum_f = arr_f.sum()

        # The FastArray sum is expected to detect the overflowing integer math,
        # continue accumulating as a float64, then return a float64.
        # Make sure it did.
        self.assertIsInstance(sum_f, np.float64)

        # Calculate the correct result using Python's arbitrary-precision integer support;
        # convert the result to a float64 value for comparison against the
        # sum returned by riptable.
        int_sum_f = a * 20
        self.assertEqual(sum_f, np.float64(int_sum_f))

    def test_overflow_uint64(self):
        '''
        FastArray flips overflow values in integer types to floats.
        Numpy produces an incorrect result.
        '''
        a = 0xFFFFFFFFFFFFFFF
        arr_f = FastArray([a] * 20, dtype=np.uint64)
        sum_f = arr_f.sum()

        # The FastArray sum is expected to detect the overflowing integer math,
        # continue accumulating as a float64, then return a float64.
        # Make sure it did.
        self.assertIsInstance(sum_f, np.float64)

        # Calculate the correct result using Python's arbitrary-precision integer support;
        # convert the result to a float64 value for comparison against the
        # sum returned by riptable.
        int_sum_f = a * 20
        self.assertEqual(sum_f, np.float64(int_sum_f))

    # ----------INPLACE OPERATIONS----------
    def test_inplace_int_float(self):
        '''
        Unlike numpy arrays, FastArray allows inplace operations between integer arrays and floating point scalars.
        The datatype of the array will remain the same. However currently division is not supported.

        Potential BUG: The floor division operator does not raise an error.
        '''
        nums = [1, 2, 3, 4, 5]
        for dt in int_types[1:]:  # dont include bool
            arr = FA(nums, dtype=dt)
            for dtf in float_types:
                scalar = dtf(1)
                arr += scalar
                self.assertEqual(
                    arr.dtype,
                    dt,
                    msg=f"Result datatype {arr.dtype} did not match original datatype {dt} after += operation",
                )
                arr -= scalar
                self.assertEqual(
                    arr.dtype,
                    dt,
                    msg=f"Result datatype {arr.dtype} did not match original datatype {dt} after -= operation",
                )
                arr *= scalar
                self.assertEqual(
                    arr.dtype,
                    dt,
                    msg=f"Result datatype {arr.dtype} did not match original datatype {dt} after *= operation",
                )
                arr /= scalar
                self.assertEqual(
                    arr.dtype,
                    dt,
                    msg=f"Result datatype {arr.dtype} did not match original datatype {dt} after /= opration",
                )
                arr //= scalar
                self.assertEqual(
                    arr.dtype,
                    dt,
                    msg=f"Result datatype {arr.dtype} did not match original datatype {dt} after //= operation",
                )

        for dt in int_types[1:]:  # dont include bool
            arr = FA(nums, dtype=dt)
            for dtf in float_types:
                arr2 = arr.astype(dtf)
                arr += arr2
                self.assertEqual(
                    arr.dtype,
                    dt,
                    msg=f"Result datatype {arr.dtype} did not match original datatype {dt} after += operation",
                )
                arr -= arr2
                self.assertEqual(
                    arr.dtype,
                    dt,
                    msg=f"Result datatype {arr.dtype} did not match original datatype {dt} after -= operation",
                )
                arr *= arr2
                self.assertEqual(
                    arr.dtype,
                    dt,
                    msg=f"Result datatype {arr.dtype} did not match original datatype {dt} after *= operation",
                )
                arr /= arr2
                self.assertEqual(
                    arr.dtype,
                    dt,
                    msg=f"Result datatype {arr.dtype} did not match original datatype {dt} after /= operation",
                )
                arr //= arr2
                self.assertEqual(
                    arr.dtype,
                    dt,
                    msg=f"Result datatype {arr.dtype} did not match original datatype {dt} after //= operation",
                )

    def test_forced_dtype_binary(self):
        '''
        Test to see that datatype is being correctly forced for result of binary ufuncs.
        You cannot force the type to be an unsigned integer.
        You cannot force the type for a division operation.
        '''
        nums = [1, 2, 3, 4, 5]
        types = int_types[1:] + float_types
        types = [i for i in types if i.__name__[0] != 'u']  # remove unsigned integers
        binary_funcs = [
            np.add,
            np.subtract,
            np.multiply,
        ]  # , np.divide, np.floor_divide ]
        for dt1 in types:
            arr1 = FA(nums, dtype=dt1)
            for dt2 in types:
                arr2 = FA(nums, dtype=dt2)
                for dt_forced in types:
                    for func in binary_funcs:
                        result = func(arr1, arr2, dtype=dt_forced)
                        self.assertEqual(
                            result.dtype,
                            dt_forced,
                            msg=f"Unable to force dtype {dt_forced} for binary func {func} between {dt1} and {dt2}",
                        )

    # def test_forced_dtype_unary(self):
    #    nums = [1,2,3,4,5]
    #    unary_funcs = [ np.sum, np.mean, np.std, np.var,
    #                    np.nansum, np.nanmean, np.nanstd, np.nanvar ]
    #    for dt in int_types[1:] + float_types:
    #        arr = FA(nums, dtype=dt)
    #        for dt_forced in int_types[1:] + float_types:
    #            for func in unary_funcs:
    #                result = func(arr, dtype=dt_forced)
    #                self.assertEqual(type(result), dt_forced, msg=f"Unable to force dtype {dt_forced} for unary func {func} on array of type {dt}")

    # def test_forced_dtype_unary(self):
    #    unary_funcs = [ np.absolute, np.abs, np.fabs, np.floor, np.ceil, np.trunc, np.round, np.rint ]
    #    nums = [1,2,3,4,5]
    #    for dt in int_types[1:] + float_types:
    #        arr = FA(nums, dtype=dt)
    #        for dt_forced in int_types[1:] + float_types:
    #            for func in unary_funcs:
    #                try:
    #                    result = func(arr, dtype=dt_forced)
    #                except:
    #                    print(f"unable to perform {func} on {dt}")
    #                #self.assertEqual(result.dtype, dt_forced, msg=f"Result datatype {result.dtype} did not match requested datatype {dt_forced} for input datatype {dt} for {func} operation")

    # --------REDUCE FUNCTION CASTING-----------
    '''
    For certain reduce functions on certain types, FastArray will return a different datatype than numpy
    '''

    def test_reduce_sums(self):
        reduce_funcs = [np.sum, np.nansum]
        usi64 = NumpyCharTypes.UnsignedInteger64
        si64 = NumpyCharTypes.SignedInteger64
        f64 = NumpyCharTypes.Float64
        input_types = [
            np.bool,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            np.intc,
            np.intp,
            np.int_,
            np.float32,
            np.float64,
        ]
        correct_list = [
            si64,
            si64,
            si64,
            si64,
            si64,
            si64,
            si64,
            si64,
            usi64,
            si64,
            si64,
            si64,
            f64,
            f64,
        ]
        correct_dict = dict(zip(input_types, correct_list))

        arr = FastArray(num_list)
        for dt in input_types:
            arr = arr.astype(dt)
            for f in reduce_funcs:
                reduce_dt = f(arr).dtype
                reduce_dt_char = reduce_dt.char
                self.assertIn(
                    reduce_dt_char,
                    correct_dict[dt],
                    msg=f"Reduce function {f} returned the wrong dtype {reduce_dt_char} ({reduce_dt}) for {dt} input.",
                )

    # -------------- REDUCE FUNCTION ACCURACY ---------
    def test_reduce_accuracy(self):
        '''
        FastArray often performs math operations at a higher level of precision than numpy. As a
        result, the answers may be slightly different.
        '''
        reduce_funcs = [
            np.sum,
            np.mean,
            np.std,
            np.var,
            np.nansum,
            np.nanmean,
            np.nanstd,
            np.nanvar,
        ]
        # use a large array to illustrate the differences
        num_list = (np.random.rand(1000000) * 100) + 1
        num_types = int_types + float_types

        for dt in num_types:
            fa_arr = FA(num_list, dtype=dt)
            np_arr = np.array(num_list, dtype=dt)
            for func in reduce_funcs:
                fa_result = func(fa_arr)
                np_result = func(np_arr)
                # 4 places of accuracy seems to solve differences in float32/64
                # TODO: Change to use self.assertAlmostEqual() instead of isclose?
                is_accurate = isclose(fa_result, np_result, rel_tol=1e-06)
                self.assertTrue(
                    is_accurate,
                    msg=f"Reduce function {func} produced fa: {fa_result} np: {np_result} for datatype {dt} with data: {fa_arr}",
                )

    # -------REDUCE FUNCTION PRESERVING---------
    def test_preserve_reduce_type(self):
        '''
        Certain reduce functions need to preserve their type. This test is to make sure FastArray
        is not performing unwanted casting.
        '''
        reduce_funcs = [np.min, np.max, np.nanmin, np.nanmax]
        arr = FastArray(num_list)
        for dt in int_types + float_types:
            arr = arr.astype(dt)
            for f in reduce_funcs:
                reduced = np.array(f(arr)).dtype
                self.assertEqual(reduced, dt)

    # --------WARN ON INVALID-------------------
    def test_no_negatives(self):
        '''
        FastArray does not warn on certain math functions that cannot take invalids.
        Numpy warns, but produces the same result.

        Make sure the result is NaN
        '''
        arr = FastArray([-50])
        for f in no_negatives:
            # DJC: unfortunate handling of neg. value warning for log2.
            if f == np.log2:
                with self.assertWarns(RuntimeWarning):
                    b = f(arr)
            else:
                b = f(arr)
            b = isnan(b)
            self.assertTrue(np.all(b))

    # --------FLIPPING DIMENSIONS-----------------
    def test_1_dimension(self):
        '''
        FastArray flips scalar initialization to 1 dimension (if the scalar is a supported type)
        Numpy creates an array of 0 dimensions

        Reduce functions on numpy arrays of 0 dimensions will produce a scalar.
        '''
        dim0 = FastArray(1).ndim
        dim1 = FastArray([1]).ndim
        self.assertEqual(dim0, dim1)

    # --------PASS SQUEEZE RESULT OF 0 DIMENSIONS TO NUMPY----
    def test_squeeze_zero(self):
        '''
        Because it does not support 0 dimensional arrays, FastArray will
        flip a 0 dimensional result of a squeeze() operation to Numpy.
        '''
        arr = FastArray([1]).squeeze()
        self.assertFalse(isinstance(arr, FastArray))

    # ----------ATTEMPT TO CONVERT STRINGS-------
    def test_string_convert(self):
        '''
        FastArray attempts to convert unicode strings to byte strings. If it can't, it will warn the user.
        '''
        u = "AAPL\u2080"
        b = 'AAPL'
        # with self.assertWarns(UserWarning):
        #    u_arr = FastArray(u)

        b_arr = FastArray([b])
        x = isinstance(b_arr[0], bytes)
        self.assertTrue(x)

    # --------TOO MANY DIMENSIONS > 1-----------
    def test_invalid_dims(self):
        '''
        FastArray only handles one dimension correctly, so it warns the user when the input is more than one dimension.
        '''
        multi_dims = np.empty((1, 20, 1, 40, 1))
        with self.assertWarns(UserWarning):
            _ = FastArray(multi_dims)

    # -------BINARY UFUNCS WITH SCALAR-------------------
    def test_binary_ufuncs_scalar(self):
        '''
        Binary math ufuncs between numeric types and scalars must match numpy exactly.
        In certain cases however, FastArray returns a different dtype, so the CRC check will fail.

        In floor_divide, numpy returns -0.0 for floats: ex. 0.0/1
        FastArray returns 0.0
        '''
        nonzero_nums = list(range(1, 10))

        num_types = int_types + float_types
        math_ufuncs = [
            np.add,
            np.subtract,
            np.multiply,
            np.divide,
            np.true_divide,
            np.floor_divide,
        ]
        for dt in num_types:
            fa_arr = FastArray(nonzero_nums, dtype=dt)
            np_arr = np.array(nonzero_nums, dtype=dt)
            for func in math_ufuncs:
                for scalar in (3, 2.1):
                    fa_result = func(fa_arr, scalar)
                    np_result = func(np_arr, scalar)
                    if fa_result.dtype == np_result.dtype:
                        fa_result = rc.CalculateCRC(fa_result)
                        np_result = rc.CalculateCRC(np_result)
                        self.assertEqual(
                            fa_result,
                            np_result,
                            msg=f"Test failed on function {func} with dtype {dt} and scalar {scalar}.",
                        )

    # -------BINARY UFUNCS WITH VECTOR-------------------
    def test_binary_ufuncs_vector(self):
        '''
        Binary math ufuncs between numeric types and scalars must match numpy exactly.
        In certain cases however, FastArray returns a different dtype, so the CRC check will fail.

        This mixes and matches every numeric datatype and performs every hooked binary ufunc. If the result dtypes match, a CRC check will
        be performed.
        '''

        num_types = int_types + float_types
        math_ufuncs = [func for func, v in gBinaryUFuncs.items() if v is not None]

        for dt1 in num_types:
            random_nums1 = (np.random.rand(10) * 100) + 1
            fa_nums1 = FastArray(random_nums1, dtype=dt1)
            np_nums1 = fa_nums1._np

            for dt2 in num_types:
                random_nums2 = (np.random.rand(10) * 100) + 1
                fa_nums2 = FastArray(random_nums2)
                np_nums2 = fa_nums2._np

                for func in math_ufuncs:
                    fa_result = func(fa_nums1, fa_nums2)
                    np_result = func(np_nums1, np_nums2)
                    if fa_result.dtype == np_result.dtype:
                        self.assertTrue(
                            almost_eq(fa_result, np_result, places=5),
                            msg=f"Test failed on function {func} with dtypes {fa_nums1.dtype} {fa_nums2.dtype}\n{fa_nums1}\n{fa_nums2}\n{fa_result}\n{np_result}",
                        )
                        # if func == np.power:
                        #    for i, _ in enumerate(fa_result):
                        #        fsub = np.log(fa_result[i]) - np.log(np_result[i])
                        #        self.assertAlmostEqual(fsub, 0, msg=f"Test failed on function {func} with dtypes {fa_nums1.dtype} {fa_nums2.dtype}\n{fa_nums1[i]}\n{fa_nums2[i]}\n{fa_result[i]}\n{np_result[i]}")
                        # else:

                        #    fa_resultcrc = rc.CalculateCRC(fa_result)
                        #    np_resultcrc = rc.CalculateCRC(np_result)
                        #    self.assertEqual(fa_resultcrc, np_resultcrc, msg=f"Test failed on function {func} with dtypes {fa_nums1.dtype} {fa_nums2.dtype}\n{fa_nums1}\n{fa_nums2}\n{fa_result}\n{np_result}")

    # ---------------UNARY UFUNCS-------------------
    def test_unary_ufuncs_scalar(self):
        '''
        ***incomplete test - need to handle the rest of the unary funcs for as many types as possible

        np.negative and np.positive both raise errors when applied to boolean arrays
        '''
        random_nums1 = (np.random.rand(10) * 100) + 1

        num_types = int_types + float_types
        # TODO: add separate test for boolean arrays to prevent natural crash
        # num_types.remove(np.bool)
        math_ufuncs = [
            np.absolute,
            np.abs,
            np.fabs,
            np.floor,
            np.ceil,
            np.trunc,
            np.round,
            np.rint,
        ]

        for dt in num_types:
            fa_arr = FastArray(random_nums1, dtype=dt)
            np_arr = np.array(random_nums1, dtype=dt)
            for func in math_ufuncs:
                fa_result = func(fa_arr)
                np_result = func(np_arr)
                if fa_result.dtype == np_result.dtype:
                    fa_result = rc.CalculateCRC(fa_result)
                    np_result = rc.CalculateCRC(np_result)
                    self.assertEqual(
                        fa_result,
                        np_result,
                        msg=f"Test failed on function {func} with dtype {dt}",
                    )
                else:
                    self.assertAlmostEqual(
                        fa_result,
                        np_result,
                        msg=f"Test failed on function {func} with dtype {dt}",
                    )

    # ----------------TEST BITWISE -------------------------
    def test_bitwise(self):
        '''
        These functions only apply to integer or boolean arrays.
        '''
        bitwise_funcs = [np.bitwise_and, np.bitwise_xor, np.bitwise_or]
        for dt in int_types:
            fa_arr = FastArray(num_list, dtype=dt)
            np_arr = fa_arr._np
            for func in bitwise_funcs:
                fa_result = func(fa_arr, fa_arr[::-1])
                np_result = func(np_arr, np_arr[::-1])
                if fa_result.dtype == np_result.dtype:
                    fa_result = rc.CalculateCRC(fa_result)
                    np_result = rc.CalculateCRC(np_result)
                    self.assertEqual(
                        fa_result,
                        np_result,
                        msg=f"Test failed on function {func} with dtype {dt}",
                    )

    # ----------------TEST LOGICAL -------------------------
    def test_compare(self):
        '''
        Compares FastArray results to numpy results for binary comparison and logical ufuncs.
        All results will be boolean arrays.

        There is a difference between calling a numpy ufunc and using a comparison operator,
        so the operators need to be checked separately.
        '''
        basic_types = [np.int32, np.int64, np.float32, np.float64]
        numeric_types = int_types + float_types
        comparison_ufuncs = [
            np.less_equal,
            np.less,
            np.equal,
            np.not_equal,
            np.greater,
            np.greater_equal,
        ]
        logical_ufuncs = [np.logical_and, np.logical_xor, np.logical_or]
        comparison_operators = [
            '__ne__',
            '__eq__',
            '__ge__',
            '__gt__',
            '__le__',
            '__lt__',
        ]
        all_funcs = comparison_ufuncs + logical_ufuncs

        for dt1 in numeric_types:
            for dt2 in numeric_types:
                fa_arr1 = FA(num_list, dtype=dt1)
                fa_arr2 = FA(list(reversed(num_list)), dtype=dt2)
                np_arr1 = np.array(num_list, dtype=dt1)
                np_arr2 = np.array(list(reversed(num_list)), dtype=dt2)
                for func in all_funcs:
                    fa_result = func(fa_arr1, fa_arr2)
                    np_result = func(np_arr1, np_arr2)
                    # check that result lengths are the same
                    self.assertEqual(
                        len(fa_result),
                        len(np_result),
                        msg=f"Result sizes did not match for {func} with dtypes {dt1} {dt2}",
                    )
                    # compare each result item
                    arr_size = len(fa_result)
                    for i in range(arr_size):
                        self.assertEqual(
                            fa_result[i],
                            np_result[i],
                            msg=f"Comparison result did not match for {func} with dtypes {dt1} {dt2}",
                        )

                for f_name in comparison_operators:
                    fa_func = fa_arr1.__getattribute__(f_name)
                    np_func = np_arr1.__getattribute__(f_name)
                    fa_result = fa_func(fa_arr2)
                    np_result = np_func(np_arr2)
                    # check that result lengths are the same
                    self.assertEqual(
                        len(fa_result),
                        len(np_result),
                        msg=f"Result sizes did not match for operator {f_name} with dtypes {dt1} {dt2}",
                    )
                    # compare each result item
                    arr_size = len(fa_result)
                    for i in range(arr_size):
                        self.assertEqual(
                            fa_result[i],
                            np_result[i],
                            msg=f"Comparison operator {f_name} failed with dtypes {dt1} {dt2}",
                        )

    # -------------------------------------------------------------
    def test_np_vs_member(self):
        import builtins

        '''
        Check to make sure the result is the same no matter how the ufunc is accessed.
        '''
        func_names = ['min', 'max', 'sum']
        arr = FA([1, 2, 3, 4, 5])
        num_types = int_types[1:] + float_types
        for dt in num_types:
            arr = arr.astype(dt)
            # print(dt)
            for name in func_names:
                member_func = None
                np_func = None
                builtin_func = None

                results = []
                member_func = arr.__getattribute__(name)
                results.append(member_func())
                if hasattr(np, name):
                    np_func = np.__getattribute__(name)
                    results.append(np_func(arr))
                if hasattr(builtins, name):
                    builtin_func = builtins.__getattribute__(name)
                    results.append(builtin_func(arr))

                self.assertEqual(
                    len(set(results)),
                    1,
                    msg=f"Results did not match for datatype {dt} and function {name}. Fastarray: {member_func} Numpy: {np_func} Builtin: {builtin_func}",
                )

                # we return different scalars than numpy, so the test below will fail
                # result_types = [ type(r) for r in results]
                # self.assertEqual(len(set(results)), 1, msg=f"Results did not match for datatype {dt} and function {name}. Fastarray: {member_func} Numpy: {np_func} Builtin: {builtin_func}" )

    # ---------------- UFUNCS ON EMPTY ----------------------------
    @pytest.mark.skip("Skipping until test is implemented.")
    def test_empty_ufuncs(self):
        n = np.array([])
        f = FA([])

        pass

    # ------------------------------------------------------
    def check_exception(self):
        '''
        Example of checking that we raise the same error as numpy
        '''
        try:
            n = np.array()
        except Exception as e:
            with self.assertRaises(e.__class__):
                f = FA()
        else:
            f = FA()
            self.assertEqual(
                f,
                n,
                msg="Empty constructor produced a different result for numpy and fastarray",
            )

    # ----------------TEST LOGICAL FOR STRING------------------------
    def test_string_compare(self):
        '''
        FastArray currently does not support bytestring array comparison with ufuncs - numpy also prints notimplemented
        However operators <=, <, ==, !=, >, >= will return the correct result (boolean array)
        '''
        f_arr = FA(['a', 'b', 'c'])
        invalid_funcs = [
            np.less_equal,
            np.less,
            np.equal,
            np.not_equal,
            np.greater,
            np.greater_equal,
        ]
        valid_func_names = ['__ne__', '__eq__', '__ge__', '__gt__', '__le__', '__lt__']
        correct_results = [
            [False, False, False],
            [True, True, True],
            [True, True, True],
            [False, False, False],
            [True, True, True],
            [False, False, False],
        ]
        correct_dict = dict(zip(valid_func_names, correct_results))

        # ufunc comparisons will not work for strings (should we implement this on our own?)
        for func in invalid_funcs:
            with self.assertRaises(
                TypeError, msg=f"String comparison did not raise TypeError for {func}"
            ):
                result = func(f_arr, f_arr)

        # strings need to be compared this way
        for f_name in valid_func_names:
            func = f_arr.__getattribute__(f_name)
            result = func(f_arr)
            correct = correct_dict[f_name]
            for i in range(len(result)):
                self.assertEqual(
                    result[i],
                    correct[i],
                    msg=f"String comparison failed for function {f_name}",
                )

    # ---------------ERROR FOR NP.POSITIVE, NP.NEGATIVE--------------
    def test_bool_bug1(self):
        '''
        Note: FastArray does not hook these ufuncs right now, but if they are taken over,
        the same error should be raised when these functions are called.
        '''
        arr = FastArray([True, False, True, False])

        with self.assertRaises(TypeError):
            _ = np.positive(arr)
        with self.assertRaises(TypeError):
            _ = np.negative(arr)

    # ------------------------------------------------------
    def test_display_properties(self):
        '''
        FastArrays of default types have default item formatting for display (see Utils.rt_display_properties)
        This checks to see that the correct item format is being returned from a FastArray
        '''
        f = FA(num_list, dtype=np.int32)
        item_format, convert_func = f.display_query_properties()
        self.assertEqual(
            item_format.length,
            DisplayLength.Short,
            msg=f"Incorrect length for item format.",
        )
        self.assertEqual(item_format.justification, DisplayJustification.Right)
        # self.assertEqual(item_format.invalid, None)
        self.assertEqual(item_format.can_have_spaces, False)
        self.assertEqual(item_format.color, DisplayColumnColors.Default)
        self.assertEqual(convert_func.__name__, 'convertInt')

    # ------------------------------------------------------
    def test_ddof_default(self):
        '''
        FastArray differs from numpy when calculating var, nanvar, std, and nanstd. We set ddof's default to 1 instead of 0
        '''
        arr = FA([1, 2])
        func_names = ['std', 'var', 'nanstd', 'nanvar']
        for name in func_names:
            # make sure different than numpy
            func = arr.__getattribute__(name)
            fa_default = func()
            np_default = func(ddof=0)
            self.assertNotAlmostEqual(
                fa_default,
                np_default,
                msg=f"Failed to set ddof to 1 for reduce function {name}",
            )

            # make sure ddof is 1 when sent to numpy reduce
            sent_to_numpy = func(keepdims=0)
            self.assertAlmostEqual(
                fa_default,
                sent_to_numpy,
                msg=f"Failed to set ddof to 1 before passing to numpy for function {name}",
            )

    # ------------------------------------------------------
    def test_dtype_reduce(self):
        '''
        If a dtype is passed to a reduce function, make sure the result is the correct dtype
        '''
        dt = np.int8
        arr = FA([1, 2])
        func_names = ['std', 'var', 'nanstd', 'nanvar']
        for name in func_names:
            func = arr.__getattribute__(name)
            result = func(dtype=dt)
            self.assertEqual(
                dt,
                result.dtype,
                msg=f"Dtypes did not match for func {name}. {dt} was the keyword, the output was {result.dtype}",
            )

    # ------------------------------------------------------
    def test_dtype_int_reduce(self):
        '''
        If the result of a reduce function is an integer, it will be cast to int64.
        If the input is uint64, the output will also be uint64
        '''
        func_names = ['nansum']
        unsigned_arr = FA(num_list, dtype=np.int32)
        signed_arr = FA(num_list, dtype=np.uint64)
        for name in func_names:
            us_func = unsigned_arr.__getattribute__(name)
            s_func = signed_arr.__getattribute__(name)
            us_dt = us_func().dtype
            s_dt = s_func().dtype
            self.assertEqual(us_dt, np.int64)
            self.assertEqual(s_dt, np.uint64)

    # ------------------------------------------------------
    def test_shift(self):
        '''
        Check to make sure FastArray's shift mimics pandas shift - not numpy roll.
        '''
        arr0 = FA([1, 2, 3, 4, 5])
        all_ints = int_types  # [1:] # bool is not included
        shift_dict = {
            tuple(float_types): np.nan,
            tuple([np.str_]): '',
            tuple([np.bytes_]): b'',
        }
        for tp in all_ints:
            if tp is bool:
                tp = np.bool_
            shift_dict[(tp,)] = INVALID_DICT[tp(1).dtype.num]
        for dtypes, invalid in shift_dict.items():
            for dt in dtypes:
                arr = arr0.astype(dt)
                pos_shift = arr.shift(1)
                neg_shift = arr.shift(-1)
                if invalid != invalid:
                    self.assertNotEqual(
                        pos_shift[0],
                        pos_shift[0],
                        msg=f"Positive shift on datatype {dt} did not fill with {invalid}.",
                    )
                    self.assertNotEqual(
                        neg_shift[-1],
                        neg_shift[-1],
                        msg=f"Negative shift on datatype {dt} did not fill with {invalid}.",
                    )
                else:
                    self.assertEqual(
                        pos_shift[0],
                        invalid,
                        msg=f"Positive shift on datatype {dt} did not fill with {invalid}.",
                    )
                    self.assertEqual(
                        neg_shift[-1],
                        invalid,
                        msg=f"Negative shift on datatype {dt} did not fill with {invalid}.",
                    )

                self.assertEqual(
                    pos_shift[1],
                    arr[0],
                    msg=f"Positive shift on datatype {dt} did not shift existing values to the correct place.",
                )
                self.assertEqual(
                    neg_shift[0],
                    arr[1],
                    msg=f"Negative shift on datatype {dt} did not shift existing values to the correct place.",
                )

    # FA supported
    def test_NotImplemented_not_returned(self):
        # See gh-5964 and gh-2091. Some of these functions are not operator
        # related and were fixed for other reasons in the past.

        # Cannot do string operation on
        binary_funcs = [
            np.power,
            np.subtract,
            np.multiply,
            np.divide,
            np.true_divide,
            np.floor_divide,
            np.bitwise_and,
            np.bitwise_or,
            np.bitwise_xor,
            np.left_shift,
            np.right_shift,
            np.fmax,
            np.fmin,
            np.fmod,
            np.hypot,
            np.logaddexp,
            np.logaddexp2,
            np.logical_and,
            np.logical_or,
            np.logical_xor,
            np.maximum,
            np.minimum,
            np.mod,
        ]

        # These functions still return NotImplemented. Will be fixed in
        # future.
        # bad = [np.greater, np.greater_equal, np.less, np.less_equal, np.not_equal]

        a = np.array('1')
        a = FA(a)
        b = 1
        for f in binary_funcs:
            with self.assertRaises(
                TypeError,
                msg=f"TypeError not raised for {f} with string and int input.",
            ):
                _ = f(a, b)


if __name__ == "__main__":
    tester = unittest.main()
