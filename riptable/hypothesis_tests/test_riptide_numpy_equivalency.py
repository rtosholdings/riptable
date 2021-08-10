# TODO add test cases for half, single, and longdouble functions.
# TODO add test cases for equivalent dtypes: bool_, bytes_, str_, int0, and uint0.
# TODO add test cases for mask operations: mask_or, mask_and, mask_xor, mask_andnot, mask_ori, mask_andi, mask_xori, and mask_andnoti.
# TODO fold all int, uint and float equivalence tests into one method that uses pytest parameters
import builtins
import operator
import random
import numpy as np
import riptable as rt
import pytest
import hypothesis
from typing import List
from riptable import FastArray, FA
from numpy.testing import assert_allclose
from hypothesis import assume, event, example, given, HealthCheck
from hypothesis.extra.numpy import (
    arrays,
    basic_indices,
    boolean_dtypes,
    floating_dtypes,
    integer_dtypes,
)
from hypothesis.strategies import (
    integers,
    booleans,
    shared,
    data,
    lists,
    floats,
    sampled_from,
    slices,
    text,
    just,
)

# riptable custom Hypothesis strategies
from .strategies.helper_strategies import (
    _MAX_FLOAT,
    _MAX_INT,
    _MAX_VALUE,
    floating_scalar,
    generate_array_and_axis,
    generate_array_and_where,
    generate_array_axis_and_ddof,
    generate_array_axis_and_repeats_array,
    generate_reshape_array_and_shape_strategy,
    generate_sample_test_floats,
    generate_sample_test_integers,
    generate_tuples_of_arrays,
    interpolation_data,
    ints_floats_complex_or_booleans,
    ints_floats_datetimes_and_timedeltas,
    ints_floats_or_complex_dtypes,
    ints_or_floats_dtypes,
    ints_or_floats_example,
    ndarray_shape_strategy,
    one_darray_shape_strategy,
    same_structure_ndarrays,
    start_stop_step_strategy,
    _MAX_SHAPE_SIZE,
)
from riptable.Utils.rt_testing import (
    assert_array_equal_,
    assert_equal_,
    _NDARRAY_FASTARRAY_COMMON_AND_DIFF,
)
from riptable.Utils.teamcity_helper import is_running_in_teamcity


"""
Commonalities between riptable and numpy
>> import numpy as np
>> import riptable as rt
>> sorted(list(set(dir(rt)).intersection(set(dir(np)))))
>> ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '__version__', 'abs', 'absolute', 'all', 'any', 'arange', 'argsort', 'bincount', 'ceil', 'concatenate', 'cumsum', 'diff', 'double', 'empty', 'empty_like', 'float32', 'float64', 'floor', 'full', 'hstack', 'int16', 'int32', 'int64', 'int8', 'interp', 'isfinite', 'isinf', 'isnan', 'lexsort', 'log', 'log10', 'max', 'maximum', 'mean', 'median', 'min', 'minimum', 'nan', 'nan_to_num', 'nanmax', 'nanmean', 'nanmedian', 'nanmin', 'nanpercentile', 'nanstd', 'nansum', 'nanvar', 'ones', 'ones_like', 'percentile', 'putmask', 'quantile', 'repeat', 'reshape', 'round', 'searchsorted', 'single', 'size', 'sort', 'std', 'sum', 'tile', 'transpose', 'trunc', 'uint16', 'uint32', 'uint64', 'uint8', 'unique', 'var', 'vstack', 'where', 'zeros', 'zeros_like']
"""


class TestRiptableNumpyEquivalency:
    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6479 abs calls np.abs instead of rt.absolute"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        arr_and_where=generate_array_and_where(
            shape=ndarray_shape_strategy(), dtype=ints_or_floats_dtypes()
        )
    )
    def test_abs(self, arr_and_where):
        arr, where_arr = arr_and_where

        # Test #1: Returns new array containing results; no optional parameters provided.
        rt_abs = rt.abs(arr)
        np_abs = np.abs(arr)
        assert isinstance(rt_abs, FastArray)
        assert_array_equal_(np.array(rt_abs), np_abs)

        # Test #2: Returns new array containing results; 'where' bitmask is provided.
        rt_abs = rt.abs(arr, where=where_arr)
        np_abs = np.abs(arr, where=where_arr)
        assert isinstance(rt_abs, FastArray)
        assert_array_equal_(np.array(rt_abs), np_abs)

        # Test #3: Results written to array specified in 'out' parameter; no other optional params.
        rt_abs = np.zeros_like(arr)
        np_abs = np.zeros_like(arr)
        rt_output = rt.abs(arr, out=rt_abs)
        np_output = np.abs(arr, out=np_abs)
        assert_array_equal_(np.array(rt_abs), np_abs)
        # TODO: Add assertions for rt_output and np_output -- what are they expected to return when the 'out' parameter is specified?
        # assert isinstance(rt_output, FastArray, msg="riptable.abs() did not return a FastArray")

        # Test #4: Results written to array specified in 'out' parameter; 'where' bitmask is provided.
        rt_abs = np.zeros_like(arr)
        np_abs = np.zeros_like(arr)
        rt_output = rt.abs(arr, where=where_arr, out=rt_abs)
        np_output = np.abs(arr, where=where_arr, out=np_abs)
        assert_array_equal_(np.array(rt_abs), np_abs)
        # TODO: Add assertions for rt_output and np_output -- what are they expected to return when the 'out' parameter is specified?
        # assert isinstance(rt_output, FastArray, msg="riptable.abs() did not return a FastArray")

    @pytest.mark.xfail(
        reason="https://jira/browse/RIP-357 discrepency between rt.absolute and np.absolute"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        arr_and_where=generate_array_and_where(
            shape=ndarray_shape_strategy(), dtype=ints_or_floats_dtypes()
        )
    )
    def test_absolute(self, arr_and_where):
        arr, where_arr = arr_and_where

        # Test #1: Returns new array containing results; no optional parameters provided.
        rt_output = rt.absolute(arr)
        np_output = np.absolute(arr)
        assert isinstance(rt_output, FastArray)
        assert_array_equal_(np.array(rt_output), np_output)

        # Test #2: Returns new array containing results; 'where' bitmask is provided.
        rt_output = rt.absolute(arr, where=where_arr)
        np_absolute = np.absolute(arr, where=where_arr)
        assert isinstance(rt_output, FastArray)
        assert_array_equal_(np.array(rt_output), np_output)

        # Test #3: Results written to array specified in 'out' parameter; no other optional params.
        rt_inplace = np.zeros_like(arr)
        np_inplace = np.zeros_like(arr)
        rt_output = rt.absolute(arr, out=rt_inplace)
        np_output = np.absolute(arr, out=np_inplace)
        assert_array_equal_(np.array(rt_inplace), np_inplace)
        # TODO: Add assertions for rt_output and np_output -- what are they expected to return when the 'out' parameter is specified?
        # assert isinstance(rt_output, FastArray, msg="riptable.absolute() did not return a FastArray")

        # Test #4: Results written to array specified in 'out' parameter; 'where' bitmask is provided.
        rt_inplace = np.zeros_like(arr)
        np_inplace = np.zeros_like(arr)
        rt_output = rt.absolute(arr, where=where_arr, out=rt_inplace)
        np_output = np.absolute(arr, where=where_arr, out=np_inplace)
        assert_array_equal_(np.array(rt_inplace), np_inplace)
        # TODO: Add assertions for rt_output and np_output -- what are they expected to return when the 'out' parameter is specified?
        # assert isinstance(rt_output, FastArray, msg="riptable.absolute() did not return a FastArray")

    @given(
        arr_and_axis=generate_array_and_axis(
            shape=ndarray_shape_strategy(), dtype=ints_floats_complex_or_booleans()
        )
    )
    def test_all(self, arr_and_axis):
        arr = arr_and_axis[0]
        axis = arr_and_axis[1]
        rt_all = rt.all(arr, axis=axis)
        np_all = np.all(arr, axis=axis)
        if axis is None:
            # If axis is None, all is performed on entire matrix, so it should have the same result as the builtin all()
            built_in_all = builtins.all(arr.flatten())
            assert rt_all == built_in_all
            assert rt_all == np_all
        else:
            # if performing all() over a certain axis, it will return an array/matrix
            assert_array_equal_(np.array(rt_all), np_all)

    @given(
        arr_and_axis=generate_array_and_axis(
            shape=ndarray_shape_strategy(), dtype=ints_floats_complex_or_booleans()
        )
    )
    def test_any(self, arr_and_axis):
        arr = arr_and_axis[0]
        axis = arr_and_axis[1]
        rt_any = rt.any(arr, axis=axis)
        np_any = np.any(arr, axis=axis)
        if axis is None:
            # If axis is None, any is performed on entire matrix, so it should have the same result as the builtin any()
            built_in_any = builtins.any(arr.flatten())
            assert rt_any == built_in_any
            assert rt_any == np_any
        else:
            # if performing any() over a certain axis, it will return an array/matrix
            assert_array_equal_(np.array(rt_any), np_any)

    @hypothesis.settings(suppress_health_check=[HealthCheck.too_slow])
    @example(
        start_stop_step={
            "start": (-1000000 - 1000000j),
            "stop": (1000000 + 1000000j),
            "step": (0.0000001 + 0.0000001j),
        }
    )
    @given(start_stop_step_strategy())
    def test_arange(self, start_stop_step):
        start = start_stop_step["start"]
        stop = start_stop_step["stop"]
        step = start_stop_step["step"]

        # This block is instead of the assume(length < MAX_VALUE) so the example can be used to test large
        # numbers and complex numbers.
        min_step = abs((stop - start) / _MAX_VALUE)
        if min_step > abs(step):
            step = (step / abs(step)) * min_step

        # Compare results of arange and exception message, if one is thrown.
        rt_error_msg = ""
        np_error_msg = ""
        try:
            rt_arange = rt.arange(start, stop, step)
        except Exception as e:
            rt_arange = None
            rt_error_msg = str(e)
        try:
            np_arange = np.arange(start, stop, step)
        except Exception as e:
            np_arange = None
            np_error_msg = str(e)

        if rt_error_msg and np_error_msg:
            assert rt_error_msg == np_error_msg
        else:
            assert_array_equal_(np.array(rt_arange), np_arange)
            assert isinstance(rt_arange, FastArray)

    # TODO: Delete this test once SOQTEST-6495 is resolved. This way arange's main functionality can still be tested
    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6495 kwargs not implemented in riptable.arange()"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(start_stop_step_strategy())
    def test_arange_kwargs(self, start_stop_step):
        start = start_stop_step["start"]
        stop = start_stop_step["stop"]
        step = start_stop_step["step"]
        # limit step size so lists don't get too long and use up too many resources
        # These can be smaller than in test_arange, because this one is just confirming the kwargs work
        max_length = 1000
        length = abs((stop - start) / step)
        # TODO directly draw from this range, instead of using hypothesis assume
        assume(-1 <= length < max_length)
        rt_error_msg = ""
        np_error_msg = ""
        try:
            rt_arange = rt.arange(start=start, stop=stop, step=step)
        except Exception as e:
            rt_arange = None
            rt_error_msg = str(e)
        try:
            np_arange = np.arange(start=start, stop=stop, step=step)
        except Exception as e:
            np_arange = None
            np_error_msg = str(e)

        if rt_error_msg and np_error_msg:
            assert rt_error_msg == np_error_msg
        else:
            assert_array_equal_(np.array(rt_arange), np_arange)
            assert isinstance(rt_arange, FastArray)

    @given(
        arrays(
            shape=ndarray_shape_strategy(), dtype=ints_floats_datetimes_and_timedeltas()
        )
    )
    @pytest.mark.skip(reason="Skip if riptable defers to numpy argsort implementation.")
    def test_argsort(self, arr):
        axis_list = [None]
        axis_list.extend(list(range(-1, len(arr.shape))))
        sort_algs = ["quicksort", "mergesort", "heapsort", "stable"]

        for axis in axis_list:
            for sort_alg in sort_algs:
                rt_argsort = rt.argsort(arr, axis=axis, kind=sort_alg)
                np_argsort = np.argsort(arr, axis=axis, kind=sort_alg)
                assert isinstance(rt_argsort, FastArray)
                assert_array_equal_(np.array(rt_argsort), np_argsort)

    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6497 riptable.bincount returns numpy array instead of FastArray"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        arr=arrays(dtype=integer_dtypes(), shape=ndarray_shape_strategy(max_rank=1)),
        use_weights=booleans(),
        min_length=integers(),
        data=data(),
    )
    def test_bincount(self, data, arr, use_weights, min_length):
        weights = None
        if use_weights:
            weights = data.draw(arrays(dtype=integer_dtypes(), shape=arr.shape))
        rt_err = np_err = rt_bincount = np_bincount = None

        try:
            rt_bincount = rt.bincount(arr, weights=weights, minlength=min_length)
        except Exception as e:
            rt_err = str(e)

        try:
            np_bincount = np.bincount(arr, weights=weights, minlength=min_length)
        except Exception as e:
            np_err = str(e)

        # TODO re-visit this implementation since if-clause will evaluate when there are no exception
        if rt_err is None or np_err is None:
            assert rt_err == np_err
        else:
            assert_array_equal_(np.array(rt_bincount), np_bincount)
            assert isinstance(rt_bincount, FastArray)

    @pytest.mark.xfail(
        reason="Related to https://jira/browse/SOQTEST-6478 different endiannesses not handled"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        data=data(),
        arr=arrays(
            shape=ndarray_shape_strategy(), dtype=floating_dtypes(sizes=(32, 64))
        ),
    )
    def test_ceil_array(self, data, arr):
        # TODO: Modify this to use the 'generate_array_and_where' strategy instead?
        where_arr = data.draw(arrays(shape=arr.shape, dtype=boolean_dtypes()))

        # Test #1: Returns new array containing results; no optional parameters provided.
        rt_output = rt.ceil(arr)
        np_output = np.ceil(arr)
        assert isinstance(rt_output, FastArray)
        assert_array_equal_(np.array(rt_output), np_output)

        # Test #2: Returns new array containing results; 'where' bitmask is provided.
        rt_output = rt.ceil(arr, where=where_arr)
        np_output = np.ceil(arr, where=where_arr)
        assert isinstance(rt_output, FastArray)
        assert_array_equal_(np.array(rt_output), np_output)

        # Test #3: Results written to array specified in 'out' parameter; no other optional params.
        rt_inplace = np.zeros_like(arr)
        np_inplace = np.zeros_like(arr)
        rt_output = rt.ceil(arr, out=rt_inplace)
        np_output = np.ceil(arr, out=np_inplace)
        assert_array_equal_(np.array(rt_inplace), np_inplace)
        # TODO: Add assertions for rt_output and np_output -- what are they expected to return when the 'out' parameter is specified?
        # assert isinstance(rt_output, FastArray, msg="riptable.ceil() did not return a FastArray")

        # Test #4: Results written to array specified in 'out' parameter; 'where' bitmask is provided.
        rt_inplace = np.zeros_like(arr)
        np_inplace = np.zeros_like(arr)
        rt_output = rt.ceil(arr, where=where_arr, out=rt_inplace)
        np_output = np.ceil(arr, where=where_arr, out=np_inplace)
        assert_array_equal_(np.array(rt_inplace), np_inplace)
        # TODO: Add assertions for rt_output and np_output -- what are they expected to return when the 'out' parameter is specified?
        # assert isinstance(rt_output, FastArray, msg="riptable.ceil() did not return a FastArray")

    @given(scalar=floating_scalar())
    def test_ceil_scalar(self, scalar):
        # test scalars
        rt_ceil_scalar = rt.ceil(scalar)
        np_ceil_scalar = np.ceil(scalar)
        assert_equal_(rt_ceil_scalar, np_ceil_scalar)

    # TODO fold the concatenate tests by using pytest params.
    # 20200303 N.B, rt.concatenate defers to np.concatenate.
    @hypothesis.settings(suppress_health_check=[HealthCheck.too_slow])
    @example(tuple_of_arrays=tuple())
    @given(tuple_of_arrays=generate_tuples_of_arrays(all_same_width=True))
    def test_concatenate_first_axis(self, tuple_of_arrays):
        rt_err_type = np_err_type = rt_concatenate = np_concatenate = None

        # Capture the output if evaluation succeeds, otherwise capture the error type.
        try:
            rt_concatenate = rt.concatenate(tuple_of_arrays, axis=0)
        except Exception as e:
            rt_err_type = type(e)
        try:
            np_concatenate = np.concatenate(tuple_of_arrays, axis=0)
        except Exception as e:
            np_err_type = type(e)

        # The concatenated arrays should be equal to a tolerance and if any exceptions were raised
        # they should be the same type.
        if rt_err_type and np_err_type:
            assert rt_err_type == np_err_type
        else:
            assert isinstance(rt_concatenate, FastArray)
            assert_array_equal_(np.array(rt_concatenate), np_concatenate)

    @hypothesis.settings(suppress_health_check=[HealthCheck.too_slow])
    @example(tuple_of_arrays=tuple())
    @given(tuple_of_arrays=generate_tuples_of_arrays(all_same_width=True))
    def test_concatenate_flatten(self, tuple_of_arrays):
        rt_err_type = np_err_type = rt_concatenate = np_concatenate = None

        # Capture the output if evaluation succeeds, otherwise capture the error type.
        try:
            rt_concatenate = rt.concatenate(tuple_of_arrays, axis=None)
        except Exception as e:
            rt_err_type = type(e)
        try:
            np_concatenate = np.concatenate(tuple_of_arrays, axis=None)
        except Exception as e:
            np_err_type = type(e)

        # The concatenated arrays should be equal to a tolerance and if any exceptions were raised
        # they should be the same type.
        if rt_err_type and np_err_type:
            assert rt_err_type == np_err_type, f"Exceptions should be the same type."
        else:
            assert isinstance(rt_concatenate, FastArray)
            assert_array_equal_(np.array(rt_concatenate), np_concatenate)

    @hypothesis.settings(suppress_health_check=[HealthCheck.too_slow])
    @example(tuple_of_arrays=tuple())
    @given(tuple_of_arrays=generate_tuples_of_arrays(all_same_height=True))
    def test_concatenate_second_axis(self, tuple_of_arrays):
        rt_err_type = np_err_type = rt_concatenate = np_concatenate = None

        # Capture the output if evaluation succeeds, otherwise capture the error type.
        try:
            rt_concatenate = rt.concatenate(tuple_of_arrays, axis=1)
        except ValueError as e:
            rt_err_type = ValueError
        try:
            np_concatenate = np.concatenate(tuple_of_arrays, axis=1)
        except ValueError as e:
            np_err_type = ValueError

        # The concatenated arrays should be equal to a tolerance and if any exceptions were raised
        # they should be the same type.
        if rt_err_type and np_err_type:
            assert rt_err_type == np_err_type
        else:
            assert isinstance(rt_concatenate, FastArray)
            assert_array_equal_(np.array(rt_concatenate), np_concatenate)

    @given(
        array_and_axis=generate_array_and_axis(
            shape=ndarray_shape_strategy(), dtype=ints_floats_or_complex_dtypes()
        ),
        output_dtype=ints_floats_or_complex_dtypes(),
    )
    def test_cumsum(self, array_and_axis, output_dtype):
        arr, axis = array_and_axis

        output_dtype = output_dtype
        rt_cumsum = rt.cumsum(arr, axis=axis)  # 20200303 defers to numpy
        np_cumsum = np.cumsum(arr, axis=axis)
        assert isinstance(rt_cumsum, FastArray)
        assert_array_equal_(np.array(rt_cumsum), np_cumsum)

        rt_cumsum = rt.cumsum(arr, axis=axis, dtype=output_dtype)
        np_cumsum = np.cumsum(arr, axis=axis, dtype=output_dtype)
        assert isinstance(rt_cumsum, FastArray)
        assert (
            rt_cumsum.dtype == output_dtype
        ), f"Dtype should be the same as input array."
        assert_array_equal_(np.array(rt_cumsum), np_cumsum)

    @given(
        array_and_axis=generate_array_and_axis(
            shape=ndarray_shape_strategy(),
            dtype=ints_floats_or_complex_dtypes(),
            # If not specified, np.diff uses -1 as the default axis.
            default_axis=-1
        ),
        # 'diff_iters': the number of differencing iterations the 'diff' function will perform.
        # This is kind of like the "divided differences" algorithm in that it's recursive, but
        # there's no division step. Specifying a large number of iterations for this is prohibitively
        # expensive and will cause the test to time out; we can test with a small-ish number of
        # iterations and still have good confidence we're covering all the important code paths.
        diff_iters=integers(min_value=0, max_value=8)
    )
    def test_diff(self, array_and_axis, diff_iters: int):
        arr, axis = array_and_axis

        # Record some events so when hypothesis reports test statistics we'll be able
        # to see roughly which parts of the search space were covered.
        event(f'dtype = {np.dtype(arr.dtype).name}')
        event(f'ndim = {len(arr.shape)}')

        # If we'll have to clamp the number of differences below for any of the array's axes,
        # record that as an event too -- so we'll know if we're running into that too often.
        if min(arr.shape) <= diff_iters:
            event("Clamped diff_iters for at least one axis of the array.")

        # We've drawn a random integer to use for the number of differencing steps to perform.
        # If the length of the current array axis is smaller, clamp it here to avoid an error.
        num_diffs = min(arr.shape[axis], diff_iters)

        # TODO parameterize over kwargs; but don't loop here, it'll be too slow.
        #  Draw the kwargs values (or None) from hypothesis strategies and let it
        #  decide how to search through the parameter space.

        rt_diff = rt.diff(arr, axis=axis, n=num_diffs)  # as of 20200303 defers to numpy (i.e. no riptable-specific implementation)
        np_diff = np.diff(arr, axis=axis, n=num_diffs)
        assert isinstance(rt_diff, FastArray)
        assert_array_equal_(np.array(rt_diff), np_diff)

    @given(examples=generate_sample_test_floats())
    def test_double(self, examples):
        for i in examples:
            rt_double = rt.double(i)
            np_double = np.double(i)
            assert_equal_(rt_double, np_double)
            assert isinstance(rt_double, type(np_double))

            rt_plus = rt_double + 1
            np_plus = np_double + 1
            assert_equal_(rt_plus, np_plus)
            assert isinstance(rt_plus, type(np_plus))

            rt_minus = rt_double - 1
            np_minus = np_double - 1
            assert_equal_(rt_minus, np_minus)
            assert isinstance(rt_minus, type(np_minus))

            rt_square = rt_double ** 2
            np_square = np_double ** 2
            assert_equal_(rt_square, np_square)
            assert isinstance(rt_square, type(np_square))

            rt_sqrt = np.sqrt(rt_double)
            np_sqrt = np.sqrt(np_double)
            assert_equal_(rt_sqrt, np_sqrt)
            isinstance(rt_sqrt, type(np_sqrt))

    @given(
        shape=ndarray_shape_strategy(),
        dtype=ints_floats_datetimes_and_timedeltas(),
        order=sampled_from(("F", "C")),
    )
    def test_empty(self, shape, dtype, order):
        # Empty does not initialize the values in the array, so it cannot be compared to a numpy array element-wise
        rt_empty = rt.empty(shape=shape, dtype=dtype, order=order)
        assert isinstance(rt_empty, FastArray)
        assert_equal_(shape, rt_empty.shape)
        assert_equal_(dtype.type, rt_empty.dtype.type)

        # 1-D arrays always use Column-order. Otherwise, use the order specified
        if len(rt_empty.shape) > 1 and min(rt_empty.shape) > 1:
            assert np.isfortran(rt_empty) == (order == "F")
        else:
            assert not np.isfortran(rt_empty)

    # TODO pull the non-subok parts so these are running tests
    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6563 riptable does not implement subok"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        arr=arrays(
            shape=ndarray_shape_strategy(), dtype=ints_floats_datetimes_and_timedeltas()
        ),
        dtype=ints_floats_or_complex_dtypes(),
        order=sampled_from(("C", "F", "K", "A")),
        subok=booleans(),
    )
    def test_empty_like(self, arr, dtype, order, subok):
        # TODO argument sweep
        rt_empty_like = rt.empty_like(arr, dtype=dtype, order=order, subok=subok)
        assert_equal_(dtype, rt_empty_like.dtype)
        assert_equal_(rt_empty_like.shape, arr.shape)

        # 1-D arrays always use Column-order. Otherwise, use the order specified
        if (
            len(rt_empty_like.shape) > 1
            and min(rt_empty_like.shape) > 1
            and (order == "F" or ((order == "A" or order == "K") and np.isfortran(arr)))
        ):
            assert np.isfortran(rt_empty_like)
        else:
            assert not np.isfortran(rt_empty_like)

        if subok:
            assert isinstance(rt_empty_like, FastArray)
        else:
            assert isinstance(rt_empty_like, np.ndarray)
            # FastArray is a subclass of np.ndarray, so also ensure it is not a FastArray
            assert not isinstance(rt_empty_like, FastArray)

    @given(examples=generate_sample_test_floats())
    def test_float32(self, examples):
        for i in examples:
            rt_float32 = rt.float32(i)
            np_float32 = np.float32(i)
            assert_equal_(rt_float32, np_float32)
            assert isinstance(rt_float32, type(np_float32))

            rt_plus = rt_float32 + 1
            np_plus = np_float32 + 1
            assert_equal_(rt_plus, np_plus)
            assert isinstance(rt_plus, type(np_plus))

            rt_minus = rt_float32 - 1
            np_minus = np_float32 - 1
            assert_equal_(rt_minus, np_minus)
            assert isinstance(rt_minus, type(np_minus))

            rt_square = rt_float32 ** 2
            np_square = np_float32 ** 2
            assert_equal_(rt_square, np_square)
            assert isinstance(rt_square, type(np_square))

            rt_sqrt = np.sqrt(rt_float32)
            np_sqrt = np.sqrt(np_float32)
            assert_equal_(rt_sqrt, np_sqrt)
            assert isinstance(rt_sqrt, type(np_sqrt))

    @given(examples=generate_sample_test_floats())
    def test_float64(self, examples):
        for i in examples:
            rt_float64 = rt.float64(i)
            np_float64 = np.float64(i)
            assert_equal_(rt_float64, np_float64)
            assert isinstance(rt_float64, type(np_float64))

            rt_plus = rt_float64 + 1
            np_plus = np_float64 + 1
            assert_equal_(rt_plus, np_plus)
            assert isinstance(rt_plus, type(np_plus))

            rt_minus = rt_float64 - 1
            np_minus = np_float64 - 1
            assert_equal_(rt_minus, np_minus)
            assert isinstance(rt_minus, type(np_minus))

            rt_square = rt_float64 ** 2
            np_square = np_float64 ** 2
            assert_equal_(rt_square, np_square)
            assert isinstance(rt_square, type(np_square))

            rt_sqrt = np.sqrt(rt_float64)
            np_sqrt = np.sqrt(np_float64)
            assert_equal_(rt_sqrt, np_sqrt)
            assert isinstance(rt_sqrt, type(np_sqrt))

    @pytest.mark.xfail(
        reason="Related to https://jira/browse/SOQTEST-6478 different endiannesses not handled"
    )
    @given(
        data=data(),
        arr=arrays(
            shape=ndarray_shape_strategy(), dtype=floating_dtypes(sizes=(32, 64))
        ),
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    def test_floor_array(self, data, arr):
        # TODO: Modify this to use the 'generate_array_and_where' strategy instead?
        where_arr = data.draw(arrays(shape=arr.shape, dtype=boolean_dtypes()))

        # Test #1: Returns new array containing results; no optional parameters provided.
        rt_output = rt.floor(arr)
        np_output = np.floor(arr)
        assert isinstance(rt_output, FastArray)
        assert_array_equal_(np.array(rt_output), np_output)

        # Test #2: Returns new array containing results; 'where' bitmask is provided.
        rt_output = rt.floor(arr, where=where_arr)
        np_output = np.floor(arr, where=where_arr)
        assert isinstance(rt_output, FastArray)
        assert_array_equal_(np.array(rt_output), np_output)

        # Test #3: Results written to array specified in 'out' parameter; no other optional params.
        rt_inplace = np.zeros_like(arr)
        np_inplace = np.zeros_like(arr)
        rt_output = rt.floor(arr, out=rt_inplace)
        np_output = np.floor(arr, out=np_inplace)
        assert_array_equal_(np.array(rt_inplace), np_inplace)
        # TODO: Add assertions for rt_output and np_output -- what are they expected to return when the 'out' parameter is specified?
        # assert isinstance(rt_output, FastArray, msg="riptable.floor() did not return a FastArray")

        # Test #4: Results written to array specified in 'out' parameter; 'where' bitmask is provided.
        rt_inplace = np.zeros_like(arr)
        np_inplace = np.zeros_like(arr)
        rt_output = rt.floor(arr, where=where_arr, out=rt_inplace)
        np_output = np.floor(arr, where=where_arr, out=np_inplace)
        assert_array_equal_(np.array(rt_inplace), np_inplace)
        # TODO: Add assertions for rt_output and np_output -- what are they expected to return when the 'out' parameter is specified?
        # assert isinstance(rt_output, FastArray, msg="riptable.floor() did not return a FastArray")

    @given(scalar=floating_scalar())
    def test_floor_scalar(self, scalar):
        # test scalars
        rt_floor_scalar = rt.floor(scalar)
        np_floor_scalar = np.floor(scalar)
        assert_equal_(np_floor_scalar, rt_floor_scalar)

    @given(
        shape=ndarray_shape_strategy(),
        fill_value=ints_or_floats_example(),
        dtype=ints_or_floats_dtypes(),
        order=sampled_from(("C", "F")),
    )
    def test_full(self, shape, fill_value, dtype, order):
        rt_err_type = np_err_type = rt_full = np_full = None
        try:
            rt_full = rt.full(shape, fill_value, dtype=dtype, order=order)
        except Exception as e:
            rt_err_type = type(e)
        try:
            np_full = np.full(
                shape=shape, fill_value=fill_value, dtype=dtype, order=order
            )
        except Exception as e:
            np_err_type = type(e)

        if rt_err_type and np_err_type:
            assert rt_err_type == np_err_type
        else:
            assert_array_equal_(np.array(rt_full), np_full)
            assert isinstance(rt_full, FastArray)
            assert np.isfortran(rt_full) == np.isfortran(np_full)

    @given(
        shape=ndarray_shape_strategy(),
        fill_value=text(),
        order=sampled_from(("C", "F")),
    )
    def test_full_str(self, shape, fill_value, order):
        rt_full = rt.full(shape, fill_value, order=order)
        np_full = np.full(shape, fill_value, order=order)
        assert_array_equal_(np.array(rt_full), np_full)
        assert isinstance(rt_full, FastArray)
        assert np.isfortran(rt_full) == np.isfortran(np_full)

    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6495 kwargs not implemented in riptable.full()",
        raises=ValueError,
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        shape=ndarray_shape_strategy(),
        fill_value=ints_or_floats_example(),
        dtype=ints_or_floats_dtypes(),
        order=sampled_from(("C", "F")),
    )
    def test_full_kwargs(self, shape, fill_value, dtype, order):
        rt_err_type = np_err_type = rt_full = np_full = None
        try:
            rt_full = rt.full(
                shape=shape, fill_value=fill_value, dtype=dtype, order=order
            )
        except Exception as e:
            rt_err_type = type(e)
        try:
            np_full = np.full(
                shape=shape, fill_value=fill_value, dtype=dtype, order=order
            )
        except Exception as e:
            np_err_type = type(e)

        if rt_err_type and np_err_type:
            assert rt_err_type == np_err_type
        else:
            assert_array_equal_(np.array(rt_full), np_full)
            assert isinstance(rt_full, FastArray)
            assert np.isfortran(rt_full) == np.isfortran(np_full)

    @hypothesis.settings(suppress_health_check=[HealthCheck.too_slow])
    @given(tuple_of_arrays=generate_tuples_of_arrays(all_same_height=True))
    def test_hstack(self, tuple_of_arrays):
        rt_err_type = np_err_type = rt_hstack = np_hstack = None
        try:
            rt_hstack = rt.hstack(tuple_of_arrays)
        except Exception as e:
            rt_err_type = type(e)
        try:
            np_hstack = np.hstack(tuple_of_arrays)
        except Exception as e:
            np_err_type = type(e)

        if rt_err_type and np_err_type:
            assert rt_err_type == np_err_type
        else:
            assert isinstance(rt_hstack, FastArray)
            assert_array_equal_(np.array(rt_hstack), np_hstack)

    @given(examples=generate_sample_test_integers(num_bits=8, signed=True))
    def test_int8(self, examples):
        for i in examples:
            rt_int8 = rt.int8(i)
            np_int8 = np.int8(i)
            assert_equal_(rt_int8, np_int8)
            assert isinstance(rt_int8, type(np_int8))

            rt_plus = rt_int8 + 1
            np_plus = np_int8 + 1
            assert_equal_(rt_plus, np_plus)
            assert isinstance(rt_plus, type(np_plus))

            rt_minus = rt_int8 - 1
            np_minus = np_int8 - 1
            assert_equal_(rt_minus, np_minus)
            assert isinstance(rt_minus, type(np_minus))

    @given(examples=generate_sample_test_integers(num_bits=16, signed=True))
    def test_int16(self, examples):
        for i in examples:
            rt_int16 = rt.int16(i)
            np_int16 = np.int16(i)
            assert_equal_(rt_int16, np_int16)
            assert isinstance(rt_int16, type(np_int16))

            rt_plus = rt_int16 + 1
            np_plus = np_int16 + 1
            assert_equal_(rt_plus, np_plus)
            assert isinstance(rt_plus, type(np_plus))

            rt_minus = rt_int16 - 1
            np_minus = np_int16 - 1
            assert_equal_(rt_minus, np_minus)
            assert isinstance(rt_minus, type(np_minus))

    @given(examples=generate_sample_test_integers(num_bits=32, signed=True))
    def test_int32(self, examples):
        for i in examples:
            rt_err = None
            np_err = None
            try:
                rt_int32 = rt.int32(i)
            except BaseException as e:
                rt_err = str(e)
            try:
                np_int32 = np.int32(i)
            except BaseException as e:
                np_err = str(e)
            if rt_err or np_err:
                assert rt_err == np_err
            else:
                assert_equal_(rt_int32, np_int32)
                assert isinstance(rt_int32, type(np_int32))

                rt_plus = rt_int32 + 1
                np_plus = np_int32 + 1
                assert_equal_(rt_plus, np_plus)
                assert isinstance(rt_plus, type(np_plus))

                rt_minus = rt_int32 - 1
                np_minus = np_int32 - 1
                assert_equal_(rt_minus, np_minus)
                assert isinstance(rt_minus, type(np_minus))

    @given(examples=generate_sample_test_integers(num_bits=64, signed=True))
    def test_int64(self, examples):
        for i in examples:
            rt_err = None
            np_err = None
            try:
                rt_int64 = rt.int64(i)
            except BaseException as e:
                rt_err = str(e)
            try:
                np_int64 = np.int64(i)
            except BaseException as e:
                np_err = str(e)

            if rt_err or np_err:
                assert rt_err == np_err
            else:
                assert_equal_(rt_int64, np_int64)
                assert isinstance(rt_int64, type(np_int64))

                rt_plus = rt_int64 + 1
                np_plus = np_int64 + 1
                assert_equal_(rt_plus, np_plus)
                assert isinstance(rt_plus, type(np_plus))

                rt_minus = rt_int64 - 1
                np_minus = np_int64 - 1
                assert_equal_(rt_minus, np_minus)
                assert isinstance(rt_minus, type(np_minus))

    @given(arg_dict=interpolation_data())
    def test_interp(self, arg_dict):
        rt_err = np_err = rt_interp = np_interp = None

        try:
            rt_interp = rt.interp(
                arg_dict["x"], arg_dict["xp"], arg_dict["fp"]
            )  # According to the documentation, Riptable
            # does not implement kwargs left, right,
            # and period
        except Exception as e:
            rt_err = e

        try:
            np_interp = np.interp(arg_dict["x"], arg_dict["xp"], arg_dict["fp"])
        except Exception as e:
            np_err = e

        if rt_err or np_err:
            assert rt_err == np_err
        else:
            assert_equal_(rt_interp.dtype.name, np_interp.dtype.name)
            assert_allclose(np.array(rt_interp), np_interp)
            assert isinstance(rt_interp, FastArray)

    @given(
        arr=arrays(
            shape=ndarray_shape_strategy(), dtype=floating_dtypes(endianness="=")
        )
    )
    def test_isfinite(self, arr):
        rt_isfinite = rt.isfinite(arr)
        np_isfinite = np.isfinite(arr)
        assert_array_equal_(rt_isfinite, np_isfinite)
        assert isinstance(rt_isfinite, FastArray)

    @given(
        arr=arrays(
            shape=ndarray_shape_strategy(), dtype=floating_dtypes(endianness="=")
        )
    )
    def test_isinf(self, arr):
        rt_isinf = rt.isinf(arr)
        np_isinf = np.isinf(arr)
        assert_array_equal_(rt_isinf, np_isinf)
        assert isinstance(rt_isinf, FastArray)

    @given(
        arr=arrays(
            shape=ndarray_shape_strategy(), dtype=floating_dtypes(endianness="=")
        )
    )
    def test_isnan(self, arr):
        rt_isnan = rt.isnan(arr)
        np_isnan = np.isnan(arr)
        assert_array_equal_(rt_isnan, np_isnan)
        assert isinstance(rt_isnan, FastArray)

    @pytest.mark.xfail(
        reason="RIP-339: lexsort broken for ndarray and FastArray input types"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Performance issues when running on TeamCity"
    )
    @given(data())
    def test_lexsort_basic(self, data):
        shape, dtype = (
            shared(ndarray_shape_strategy(max_rank=1)),
            shared(ints_floats_datetimes_and_timedeltas()),
        )
        a = data.draw(arrays(shape=shape, dtype=dtype))
        b = data.draw(arrays(shape=shape, dtype=dtype))

        assert_array_equal_(np.lexsort((b, a)), rt.lexsort((list(b), list(a))))

    @pytest.mark.xfail(
        reason="RIP-339: lexsort broken for ndarray and FastArray input types"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Performance issues when running on TeamCity"
    )
    @given(data())
    def test_lexsort_basic_fail(self, data):
        shape, dtype = (
            shared(ndarray_shape_strategy(max_rank=1)),
            shared(ints_or_floats_dtypes()),
        )
        a = data.draw(arrays(shape=shape, dtype=dtype))
        b = data.draw(arrays(shape=shape, dtype=dtype))

        rt_lexsort_args = [(FA(b), FA(a)), (FA(b), a), (b, FA(a)), (b, a)]
        for rt_arg in rt_lexsort_args:
            assert_array_equal_(np.lexsort((b, a)), rt.lexsort(rt_arg))

    @pytest.mark.xfail(
        reason="RIP-339: lexsort broken for ndarray and FastArray input types"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Performance issues when running on TeamCity"
    )
    @given(data())
    def test_lexsort_mixed(self, data):
        shape, dtype = (
            shared(ndarray_shape_strategy(max_rank=1)),
            ints_floats_datetimes_and_timedeltas(),
        )
        a = data.draw(arrays(shape=shape, dtype=dtype))
        b = data.draw(arrays(shape=shape, dtype=dtype))

        assert_array_equal_(np.lexsort((b, a)), rt.lexsort((list(b), list(a))))

    @pytest.mark.xfail(
        reason="RIP-339: lexsort broken for ndarray and FastArray input types"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Performance issues when running on TeamCity"
    )
    @given(data())
    def test_lexsort_mixed_fail(self, data):
        shape, dtype = (
            shared(ndarray_shape_strategy(max_rank=1)),
            ints_floats_datetimes_and_timedeltas(),
        )
        a = data.draw(arrays(shape=shape, dtype=dtype))
        b = data.draw(arrays(shape=shape, dtype=dtype))

        rt_lexsort_args = [(FA(b), FA(a)), (FA(b), a), (b, FA(a)), (b, a)]
        for rt_arg in rt_lexsort_args:
            assert_array_equal_(np.lexsort((b, a)), rt.lexsort(rt_arg))

    @pytest.mark.xfail(
        reason="RIP-339: lexsort broken for ndarray and FastArray input types"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Performance issues when running on TeamCity"
    )
    @given(data())
    def test_lexsort_many_columns(self, data):
        k = data.draw(integers(min_value=1, max_value=_MAX_SHAPE_SIZE // 10))
        shape, dtype = (
            shared(ndarray_shape_strategy(max_rank=1)),
            shared(integer_dtypes(endianness="=")),
        )

        keys: List[List[int]] = list()
        for _ in range(k):
            keys.append(list(data.draw(arrays(shape=shape, dtype=dtype))))

        assert_array_equal_(np.lexsort(keys), rt.lexsort(keys))

    @pytest.mark.xfail(
        reason="RIP-339: lexsort broken for ndarray and FastArray input types"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Performance issues when running on TeamCity"
    )
    @given(data())
    def test_lexsort_many_columns_fail(self, data):
        k = data.draw(integers(min_value=1, max_value=_MAX_SHAPE_SIZE // 10))
        shape, dtype = (
            shared(ndarray_shape_strategy(max_rank=1)),
            shared(integer_dtypes(endianness="=")),
        )

        keys: List[FastArray] = list()
        for _ in range(k):
            keys.append(FA(data.draw(arrays(shape=shape, dtype=dtype))))

        assert_array_equal_(np.lexsort(keys), rt.lexsort(keys))

    @given(
        arr=arrays(
            shape=ndarray_shape_strategy(), dtype=floating_dtypes(endianness="=")
        )
    )
    def test_log_array(self, arr):
        rt_log = rt.log(arr)
        np_log = np.log(arr)
        assert_allclose(np.array(rt_log), np_log, rtol=1e-6)
        assert isinstance(rt_log, FastArray)

    @given(scalar=floats())
    def test_log_scalar(self, scalar):
        rt_log = rt.log(scalar)
        np_log = np.log(scalar)
        assert_allclose(rt_log, np_log, rtol=1e-6)

    @given(
        arr=arrays(
            shape=ndarray_shape_strategy(), dtype=floating_dtypes(endianness="=")
        )
    )
    def test_log10_array(self, arr):
        rt_log10 = rt.log10(arr)
        np_log10 = np.log10(arr)
        assert_allclose(np.array(rt_log10), np_log10, rtol=1e-6)
        assert isinstance(rt_log10, FastArray)

    @given(scalar=floats())
    def test_log10_scalar(self, scalar):
        rt_log10 = rt.log10(scalar)
        np_log10 = np.log10(scalar)
        assert_allclose(rt_log10, np_log10, rtol=1e-6)

    @pytest.mark.xfail(
        reason="This test exposes a known bug around NaN-handling in 'max' that needs to be fixed."
    )
    @given(
        arr_axis_tuple=generate_array_and_axis(
            shape=ndarray_shape_strategy(), dtype=ints_floats_datetimes_and_timedeltas()
        )
    )
    @given(mats=same_structure_ndarrays())
    def test_maximum(self, mats):
        mat1, mat2 = mats

        np_output = np.maximum(mat1, mat2)
        rt_output = rt.maximum(mat1, mat2)

        assert_array_equal_(np.array(rt_output), np_output)

        # TODO: Enable this assertion once we've fixed riptable so rt.maximum returns a FastArray.
        # assert isinstance(rt_output, FastArray)

    @given(
        arr_axis_tuple=generate_array_and_axis(
            shape=ndarray_shape_strategy(), dtype=ints_floats_datetimes_and_timedeltas()
        )
    )
    def test_min(self, arr_axis_tuple):
        arr, axis = arr_axis_tuple

        rt_output = rt.min(arr, axis=axis)
        np_output = np.min(arr, axis=axis)

        assert_array_equal_(np.array(rt_output), np_output)

        # TODO: Enable this assertion once we've fixed riptable so rt.maximum returns a FastArray.
        # if axis: #if axis is None, it will return a scalar
        #     assert isinstance(rt_output, FastArray)

    @pytest.mark.parametrize(
        "func",
        [
            # TODO add params: nanmean, nansum, nanstd, nanvar, nanargmin, nanargmax
            # TODO add other missing array methods, then dynamically generate these functions from introspection of numpy and riptable
            # List of ndarray array methods: https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#array-methods
            "abs",
            "all",
            "any",
            "argmax",
            "argmin",
            "argsort",
            # TODO add cumprod
            "cumsum",
            "mean",
            "median",
            "nanmedian",
            "percentile",
            "std",
            "var",
            pytest.param(
                "sum",
                marks=[
                    pytest.mark.xfail(
                        reason="Need to investigate intermittent failures possibly due to roundoff errors."
                    )
                ],
            ),
            pytest.param(
                "nanmax",
                marks=[
                    pytest.mark.xfail(
                        reason="RIP-417: This test exposes a known bug around NaN-handling in 'nanmax' that needs to be fixed."
                    )
                ],
            ),
            pytest.param(
                "nanmin",
                marks=[
                    pytest.mark.xfail(
                        reason="RIP-417: This test exposes a known bug around NaN-handling in 'nanmin' that needs to be fixed."
                    )
                ],
            ),
            pytest.param(
                "max",
                marks=[
                    pytest.mark.xfail(
                        reason="Conflict in comparing FastArray with invalids that don't exist as ndarray invalids. Need to handle invalid assertion checks."
                    )
                ],
            ),
            pytest.param(
                "nanpercentile",
                marks=[
                    pytest.mark.xfail(
                        reason="Investigate frequent TeamCity hypothesis DeadlineExceeded exceptions that never occur on local runs"
                    )
                ],
            ),
        ],
    )
    @pytest.mark.parametrize(
        "include_invalid",
        [
            False
        ],  # TODO enable including invalids and use masked arrays to check valid and possibly invalid values
    )
    def test_array_methods(self, func, include_invalid):
        # TODO rework so the generated array and axis can be cached and reused, otherwise most of the time spent for this test is generating arrays and an axis
        # Attempted to use the hypothesis cacheable decorator, but it doesn't cache the strategy for reuse
        # Investigate reworking generate_array_and_axis by using the defines_strategy_with_reusable_values decorator
        @given(
            arr_axis_tuple=generate_array_and_axis(
                shape=ndarray_shape_strategy(),
                dtype=ints_or_floats_dtypes(),
                include_invalid=include_invalid,
            )
        )
        def inner(arr_axis_tuple):
            arr, axis = arr_axis_tuple

            # TODO enable this assertion for guarded funcs once we've fixed riptable so these return a fastarray.
            def _check_fastarray(inst):
                if (
                    axis
                ):  # need an axis to check if output is a FastArray, otherwise output is a scalar
                    assert isinstance(inst, rt.FastArray)

            args = (func, arr)
            kwargs = {}
            if func not in ["abs"]:
                kwargs["axis"] = axis
            # set up keyword arguments needed for some of the rt_numpy functions
            if "std" in func or "var" in func:
                kwargs["ddof"] = 1
            if "percentile" in func:
                kwargs["interpolation"] = "linear"
                kwargs["q"] = 50
            if "argsort" == func:
                kwargs["kind"] = "stable"

            # Test #1:  Check calling .func() as a module function in numpy vs. riptable.
            rt_output = operator.methodcaller(*args, **kwargs)(rt)
            np_output = operator.methodcaller(*args, **kwargs)(np)
            assert_array_equal_(rt_output, np_output)

            # Test #2:  Check calling .func() as a module function in numpy vs. riptable.
            #           This time, explicitly wrap the ndarray generated by hypothesis in a FastArray
            #           so we're sure to get the FastArray implementation.
            f_arr = rt.FA(arr)
            args = (func, f_arr)
            rt_output = operator.methodcaller(*args, **kwargs)(rt)
            np_output = operator.methodcaller(*args, **kwargs)(np)
            assert_array_equal_(rt_output, np_output)
            # _check_fastarray(rt_output)  # rt_output should be a FastArray, but is a ndarray

            # Test #3:  Call the .func() method on both the FastArray and the ndarray.
            # Either FastArray or ndarray does not support the guarded member names.
            common_members, _ = _NDARRAY_FASTARRAY_COMMON_AND_DIFF
            if func in common_members:
                rt_output = operator.methodcaller(func, **kwargs)(f_arr)
                np_output = operator.methodcaller(func, **kwargs)(arr)
                assert_array_equal_(rt_output, np_output)
                # _check_fastarray(rt_output)  # rt_output should be a FastArray, but is a ndarray

        inner()

    @given(arrs=sampled_from([[9, 8, 7], [[10, 1],[11, 2],[13, 3]]]), inds=just([False, True, False]))
    def test_array_boolean_indexing(self, arrs, inds):
        np_array = np.asfortranarray(arrs)
        rt_array = rt.FA(np_array)

        np_bool_indices = inds
        rt_bool_indices = rt.FA(np_bool_indices)

        nr = np_array[np_bool_indices]
        fr = rt_array[rt_bool_indices]

        assert_array_equal_(np.array(fr), nr)

    @given(mats=same_structure_ndarrays())
    def test_minimum(self, mats):
        mat1, mat2 = mats

        np_output = np.minimum(mat1, mat2)
        rt_output = rt.minimum(mat1, mat2)

        assert_array_equal_(np.array(rt_output), np_output)

        # TODO: Enable this assertion once we've fixed riptable so rt.minimum returns a FastArray.
        # assert isinstance(rt_output, FastArray)

    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6497 Riptable.nan_to_num() does not return a FastArray",
        raises=AssertionError,
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(array=arrays(shape=ndarray_shape_strategy(), dtype=floating_dtypes()))
    def test_nan_to_num_array(self, array):
        rt_nan_to_num = rt.nan_to_num(array)
        np_nan_to_num = np.nan_to_num(array)

        assert_allclose(rt_nan_to_num, np_nan_to_num)
        assert isinstance(rt_nan_to_num, FastArray)

    @given(scalar=floats())
    def test_nan_to_num_scalar(self, scalar):
        rt_nan_to_num = rt.nan_to_num(scalar)
        np_nan_to_num = np.nan_to_num(scalar)

        assert_allclose(rt_nan_to_num, np_nan_to_num)

    @given(
        arr=arrays(shape=ndarray_shape_strategy(), dtype=floating_dtypes()),
        q=floats(min_value=0, max_value=100),
    )
    def test_nanpercentile_single(self, arr, q):
        interpolations = ["linear", "lower", "higher", "midpoint", "nearest"]
        for interpolation in interpolations:
            # todo - rt_nanpercentile should come from riptable
            rt_nanpercentile = np.nanpercentile(arr, q=q, interpolation=interpolation)
            np_nanpercentile = np.nanpercentile(arr, q=q, interpolation=interpolation)
            assert_allclose(rt_nanpercentile, np_nanpercentile)

    # @given(arr=arrays(shape=ndarray_shape_strategy(), dtype=floating_dtypes()), q=floats(min_value=0, max_value=100))
    # def test_nanpercentile_single(self, arr, q):
    #     interpolations = ['linear', 'lower', 'higher', 'midpoint', 'nearest']
    #     for interpolation in interpolations:
    #         # todo - rt_nanpercentile should come from riptable
    #         rt_nanpercentile = np.nanpercentile(arr, q=q, interpolation=interpolation)
    #         np_nanpercentile = np.nanpercentile(arr, q=q, interpolation=interpolation)
    #         assert_allclose(rt_nanpercentile, np_nanpercentile)

    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6497 Riptable.nanpercentile() does not return a FastArray",
        raises=AssertionError,
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        arr=arrays(shape=ndarray_shape_strategy(), dtype=floating_dtypes()),
        q=lists(
            floats(min_value=0, max_value=100), min_size=10, max_size=50, unique=True
        ),
    )
    def test_nanpercentile_array(self, arr, q):
        interpolations = ["linear", "lower", "higher", "midpoint", "nearest"]
        for interpolation in interpolations:
            # todo - rt_nanpercentile should come from riptable
            rt_output = np.nanpercentile(arr, q=q, interpolation=interpolation)
            np_output = np.nanpercentile(arr, q=q, interpolation=interpolation)

            assert_allclose(rt_output, np_output)
            if len(arr.shape) > 1:
                assert isinstance(rt_output, FastArray)

    @pytest.mark.xfail(
        reason=
            "https://jira/browse/SOQTEST-6637 Riptable does not convert the array to a FastArray, so it does not guarantee nanstd will be an available attribute\n"
            "https://jira/browse/SOQTEST-6497 Riptable.nanstd() does not return a FastArray"
        ,
        raises=(AttributeError, AssertionError),
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(arr_axis_and_ddof=generate_array_axis_and_ddof())
    def test_nanstd(self, arr_axis_and_ddof):
        """
        This function has a different default behavior from numpy. Riptable, like pandas, has a default ddof of 1, while
        numpy uses 0.
        """
        arr = arr_axis_and_ddof["arr"]
        axis = arr_axis_and_ddof["axis"]
        ddof = arr_axis_and_ddof["ddof"]

        rt_output = rt.nanstd(arr, axis=axis, ddof=ddof)
        np_output = np.nanstd(arr, axis=axis, ddof=ddof)

        assert_allclose(rt_output, np_output)
        if axis and len(arr.shape) > 1:
            assert isinstance(rt_output, FastArray)

    @pytest.mark.xfail(
        reason=
            "https://jira/browse/SOQTEST-6637 Riptable does not convert the array to a FastArray, so it does not guarantee nansum will be an available attribute\n"
            "nansum does not implement the axis argument either",
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        array_and_axis=generate_array_and_axis(
            shape=ndarray_shape_strategy(max_rank=1), dtype=ints_or_floats_dtypes()
        )
    )
    def test_nansum(self, array_and_axis):
        """
        riptable.nansum() does not yet implement the axis argument, since riptable is mostly focused on 1D FastArrays.
        test_nansum_1D only selects arrays that are rank 0 or 1. Riptable's nansum, like nanmean, does not cast the array
        passed in to a FastArray--when a FastArray is not passed in an Attribute exception occurs because the function
        tries calling array.nansum().
        """
        arr, axis = array_and_axis
        arr = FastArray(arr)

        rt_output = rt.nansum(arr, axis=axis)
        np_output = np.nansum(arr, axis=axis)

        assert_allclose(rt_output, np_output)
        if axis:
            assert isinstance(rt_output, FastArray)

    # The floats should have the same byte-order as the machine at the moment (https://jira/browse/SOQTEST-6478) -> endianness="="
    # and riptable only officially supports float32 and float64 floating-point datatypes, so using other ones occasionally
    # leads to errors.
    # Split up ints and floats to ensure the same datatypes for widening. (int->int64, float->float64)
    @pytest.mark.skip(
        reason="When the array has very large numbers and numbers closed to zero, numpy and riptable can produce different results."
    )
    @given(
        arr=arrays(
            shape=ndarray_shape_strategy(max_rank=1),
            dtype=floating_dtypes(endianness="=", sizes=(32, 64)),
        )
    )
    def test_nansum_1D_float(self, arr):
        arr = FastArray(arr)
        # TODO: numpy is not widening the output array as expected to 64 bits all the time, so it is being forced here.
        rt_output = rt.nansum(arr, axis=None, dtype=np.float64)
        np_output = np.nansum(arr, axis=None, dtype=np.float64)
        assert_allclose(rt_output, np_output)

    # The floats should have the same byte-order as the machine at the moment (https://jira/browse/SOQTEST-6478) -> endianness="="
    # Split up ints and floats to ensure the same datatypes for widening. (int->int64, float->float64)
    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6670 Nansum does not follow the dtype argument and returns a float64"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        arr=arrays(
            shape=ndarray_shape_strategy(max_rank=1),
            dtype=integer_dtypes(endianness="="),
        )
    )
    def test_nansum_1D_int(self, arr):
        arr = FastArray(arr)
        # TODO: numpy is not widening the output array as expected to 64 bits all the time, so it is being forced here.
        rt_output = rt.nansum(arr, axis=None, dtype=np.int64)
        np_output = np.nansum(arr, axis=None, dtype=np.int64)
        assert rt_output.dtype.type == np.int64
        assert_allclose(rt_output, np_output)

    @pytest.mark.xfail(
        reason=
            "https://jira/browse/SOQTEST-6637 Riptable does not convert the array to a FastArray, so it does not guarantee nanvar will be an available attribute\n"
            "https://jira/browse/SOQTEST-6497 Riptable.nanvar() does not return a FastArray"
        ,
        raises=(AttributeError, AssertionError),
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(arr_axis_and_ddof=generate_array_axis_and_ddof())
    def test_nanvar(self, arr_axis_and_ddof):
        arr = arr_axis_and_ddof["arr"]
        axis = arr_axis_and_ddof["axis"]
        ddof = arr_axis_and_ddof["ddof"]

        rt_output = rt.nanvar(arr, axis=axis, ddof=ddof)
        np_output = np.nanvar(arr, axis=axis, ddof=ddof)

        assert_allclose(rt_output, np_output)
        if axis and len(arr.shape) > 1:
            assert isinstance(rt_output, FastArray)

    @given(
        shape=ndarray_shape_strategy(),
        dtype=ints_or_floats_dtypes(),
        order=sampled_from(("F", "C")),
    )
    def test_ones(self, shape, dtype, order):
        rt_ones = rt.ones(shape, dtype, order)
        np_ones = np.ones(shape=shape, dtype=dtype, order=order)
        assert isinstance(rt_ones, FastArray)

        assert_equal_(shape, rt_ones.shape)
        assert_equal_(dtype.type, rt_ones.dtype.type)
        assert_array_equal_(rt_ones, np_ones)

        # 1-D arrays always use Column-order. Otherwise, use the order specified
        if len(rt_ones.shape) > 1 and min(rt_ones.shape) > 1:
            assert np.isfortran(rt_ones) == (order == "F")
        else:
            assert not np.isfortran(rt_ones)

    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6495 kwargs not implemented in riptable.ones"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        shape=ndarray_shape_strategy(),
        dtype=ints_or_floats_dtypes(),
        order=sampled_from(("F", "C")),
    )
    def test_ones_kwargs(self, shape, dtype, order):
        rt_ones = rt.ones(shape=shape, dtype=dtype, order=order)
        np_ones = np.ones(shape=shape, dtype=dtype, order=order)
        assert isinstance(rt_ones, FastArray)

        assert_equal_(shape, rt_ones.shape)
        assert_equal_(dtype.type, rt_ones.dtype.type)
        assert_array_equal_(rt_ones, np_ones)

        # 1-D arrays always use Column-order. Otherwise, use the order specified
        if len(rt_ones.shape) > 1 and min(rt_ones.shape) > 1:
            assert np.isfortran(rt_ones) == (order == "F")
        else:
            assert not np.isfortran(rt_ones)

    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6563 riptable does not implement subok"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        arr=arrays(
            shape=ndarray_shape_strategy(), dtype=ints_floats_datetimes_and_timedeltas()
        ),
        dtype=ints_floats_or_complex_dtypes(),
        order=sampled_from(("C", "F", "K", "A")),
        subok=booleans(),
    )
    def test_ones_like(self, arr, dtype, order, subok):
        rt_ones_like = rt.ones_like(arr, dtype=dtype, order=order, subok=subok)
        np_ones_like = np.ones_like(arr, dtype=dtype, order=order, subok=subok)

        assert_equal_(dtype, rt_ones_like.dtype)
        assert_equal_(rt_ones_like.shape, arr.shape)
        assert_array_equal_(rt_ones_like, np_ones_like)

        # 1-D arrays always use Column-order. Otherwise, use the order specified
        if (
            len(rt_ones_like.shape) > 1
            and min(rt_ones_like.shape) > 1
            and (order == "F" or ((order == "A" or order == "K") and np.isfortran(arr)))
        ):
            assert np.isfortran(rt_ones_like)
        else:
            assert not np.isfortran(rt_ones_like)

        if subok:
            assert isinstance(rt_ones_like, FastArray)
        else:
            assert isinstance(rt_ones_like, np.ndarray)
            # FastArray is a subclass of np.ndarray, so also ensure it is not a FastArray
            assert not isinstance(rt_ones_like, FastArray)

    @given(
        arr=arrays(shape=ndarray_shape_strategy(), dtype=floating_dtypes()),
        q=floats(min_value=0, max_value=100),
    )
    def test_percentile_single(self, arr, q):
        interpolations = ["linear", "lower", "higher", "midpoint", "nearest"]
        for interpolation in interpolations:
            # todo - rt_percentile should come from riptable
            rt_output = np.percentile(arr, q=q, interpolation=interpolation, axis=None)
            np_output = np.percentile(arr, q=q, interpolation=interpolation, axis=None)
            assert_allclose(rt_output, np_output)

    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6497 Riptable.percentile() does not return a FastArray",
        raises=AssertionError,
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        array_and_axis=generate_array_and_axis(
            shape=ndarray_shape_strategy(), dtype=floating_dtypes()
        ),
        q=lists(
            floats(min_value=0, max_value=100), min_size=10, max_size=50, unique=True
        ),
    )
    def test_percentile_array(self, array_and_axis, q):
        arr, axis = array_and_axis
        interpolations = ["linear", "lower", "higher", "midpoint", "nearest"]
        for interpolation in interpolations:
            rt_output = rt.percentile(arr, q=q, interpolation=interpolation, axis=axis)
            np_output = np.percentile(arr, q=q, interpolation=interpolation, axis=axis)
            assert_allclose(rt_output, np_output)
            if axis and len(arr.shape) > 1:
                assert isinstance(rt_output, FastArray)

    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6497 Riptable.putmask does not cast input to a FastArray",
        raises=AssertionError,
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        array_and_mask=generate_array_and_where(
            shape=ndarray_shape_strategy(), dtype=floating_dtypes(endianness="=")
        ),
        rep_val=floats(),
    )
    def test_putmask(self, array_and_mask, rep_val):
        arr, mask = array_and_mask
        rt_err = None
        rt_putmask = arr.copy()
        np_err = None
        np_putmask = arr.copy()
        try:
            rt.putmask(rt_putmask, mask, rep_val)
        except OverflowError:
            rt_err = OverflowError

        try:
            np.putmask(np_putmask, mask, rep_val)
        except OverflowError:
            np_err = OverflowError

        if rt_err or np_err:
            assert rt_err == np_err
        else:
            assert_allclose(rt_putmask, np_putmask)
            assert isinstance(rt_putmask, FastArray)

    @pytest.mark.xfail(
        reason="RIP-341: quantile's _get_score raises an index error when getting the view from _val and using it to access the ndarray."
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        arr=arrays(
            shape=one_darray_shape_strategy(), dtype=floating_dtypes(sizes=(32, 64))
        )
    )
    def test_quantile(self, arr):
        # For a one to one equivalence of supported interpolation methods between numpy and riptable:
        # 1) np.quantile would need to support riptable's fraction method
        # 2) rt.quantile would need to support numpy's linear, midpoint, and nearest methods
        # Numpy quantile reference https://docs.scipy.org/doc/numpy/reference/generated/numpy.quantile.html
        # Tracked in Jira https://jira/browse/RIP-344
        interpolations = ["lower", "higher"]
        for interpolation in interpolations:
            q = random.random()
            rt_quantile, _, _ = rt.quantile(arr, q, interpolation)
            np_quantile = np.quantile(arr, q, interpolation=interpolation)
            assert_allclose(
                rt_quantile, np_quantile, err_msg=f"to reproduce use quantile {q}"
            )

    @given(
        arr=arrays(shape=ndarray_shape_strategy(), dtype=ints_or_floats_dtypes()),
        rep=integers(min_value=0, max_value=10),
        axis=sampled_from((None, 0, 1)),
    )
    def test_repeat_integer_repeats(self, arr, rep, axis):
        if axis == 1 and len(arr.shape) == 1:
            axis = 0

        rt_repeat = rt.repeat(arr, repeats=rep, axis=axis)
        np_repeat = np.repeat(arr, repeats=rep, axis=axis)
        assert_allclose(np.array(rt_repeat), np_repeat)

    @given(
        arr_axis_and_rep=generate_array_axis_and_repeats_array(
            dtype=ints_or_floats_dtypes(), shape=ndarray_shape_strategy()
        )
    )
    def test_repeat_array_repeats(self, arr_axis_and_rep):
        arr, axis, rep = arr_axis_and_rep

        rt_repeat = rt.repeat(arr, repeats=rep, axis=axis)
        np_repeat = np.repeat(arr, repeats=rep, axis=axis)

        assert_allclose(np.array(rt_repeat), np_repeat)
        assert isinstance(rt_repeat, FastArray)

    @given(
        arr_and_dim=generate_reshape_array_and_shape_strategy(
            shape=ndarray_shape_strategy(), dtype=ints_or_floats_dtypes()
        )
    )
    def test_reshape(self, arr_and_dim):
        arr, dim = arr_and_dim
        orders = ["C", "F", "A"]
        for order in orders:
            rt_reshape = rt.reshape(arr, dim, order=order)
            np_reshape = np.reshape(arr, dim, order=order)
            assert_allclose(np.array(rt_reshape), np_reshape)
            assert np.isfortran(rt_reshape) == np.isfortran(np_reshape)
            assert isinstance(rt_reshape, FastArray)

    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6530 riptable.round() does not return FastArray and decimals arg not yet implemented"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        arr=arrays(
            shape=ndarray_shape_strategy(), dtype=floating_dtypes(sizes=(32, 64))
        )
    )
    def test_round_array(self, arr):
        # TODO: Use range of decimals once https://jira/browse/SOQTEST-6530 is addressed
        decimals = [0]
        # decimals = range(-5, 100)

        for decimal in decimals:
            # test array
            rt_output = rt.round(arr, decimals=decimal)
            np_output = np.round(arr, decimals=decimal)

            assert_array_equal_(np.array(rt_output), np_output)
            assert isinstance(rt_output, FastArray)

    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6530 should fail on +/-inf, and [-0.5,-0], since riptable.round calls builtin round. Decimals arg also not yet implemented"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(scalar=floating_scalar())
    def test_round_scalar(self, scalar):
        # TODO: Use range of decimals once https://jira/browse/SOQTEST-6530 is addressed
        decimals = [0]
        # decimals = range(-5, 100)
        for decimal in decimals:
            # test scalars
            rt_round_scalar = rt.round(scalar, decimals=decimal)
            np_round_scalar = np.round(scalar, decimals=decimal)
            assert_equal_(
                rt_round_scalar,
                np_round_scalar,
                err_msg=f"Rounding error on {type(scalar)} {scalar}",
            )

    @pytest.mark.xfail(reason="RIP-345 - discrepancy between inserts position points")
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(arrays(shape=one_darray_shape_strategy(), dtype=integer_dtypes()))
    def test_searchsorted(self, arr):
        sides = {"left", "right"}

        arr.sort()
        v = arr[arr % 2 == 0]

        for side in sides:
            rt_indicies = rt.searchsorted(arr, v, side)
            np_indicies = np.searchsorted(arr, v, side)
            assert_array_equal_(
                rt_indicies,
                np_indicies,
                err_msg=f"using array {arr}\nvalues to insert {v}\nside {side}",
            )

    @pytest.mark.xfail(
        reason="RIP-350 - see Python/core/riptable/tests/test_base_function.TestStd.test_std for a materialized counterexample"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        arr=arrays(
            shape=one_darray_shape_strategy(max_shape_size=10),
            # do not consider overflow; numpy and riptable differ in how they behave on overflow
            elements=integers(-int(_MAX_INT / 10), int(_MAX_INT / 10)),
            dtype=integer_dtypes(endianness="=", sizes=(64,)),
        )
    )
    def test_std(self, arr):
        np_std = np.std(arr, ddof=1)
        rt_std = rt.std(rt.FastArray(arr))
        assert_equal_(rt_std, np_std, decimal=6)

    @given(
        arr=arrays(
            shape=one_darray_shape_strategy(max_shape_size=10),
            # do not consider overflow; numpy and riptable differ in how they behave on overflow
            elements=integers(-int(_MAX_INT / 10), int(_MAX_INT / 10)),
            dtype=integer_dtypes(endianness="=", sizes=(64,)),
        )
    )
    def test_sum_int(self, arr):
        np_sum = np.sum(arr)
        rt_sum = rt.sum(rt.FastArray(arr))
        assert_equal_(int(rt_sum), int(np_sum))

    @pytest.mark.xfail(
        reason="RIP-351 - see Python/core/riptable/tests/test_base_function.TestSum.test_sum_float for a materialized counterexample"
    )
    @given(
        arr=arrays(
            shape=one_darray_shape_strategy(max_shape_size=10),
            # do not consider overflow; numpy and riptable differ in how they behave on overflow
            elements=floats(-int(_MAX_FLOAT / 10), int(_MAX_FLOAT / 10)),
            dtype=floating_dtypes(endianness="=", sizes=(64,)),
        )
    )
    def test_sum_float(self, arr):
        np_sum = np.sum(arr)
        rt_sum = rt.sum(rt.FastArray(arr))
        assert_equal_(rt_sum, np_sum, decimal=6)

    @pytest.mark.skip(reason="Resolve DeadlineExceeded errors on workflow runs")
    @given(
        arr=arrays(
            shape=ndarray_shape_strategy(max_shape_size=5),
            dtype=floating_dtypes(endianness="=", sizes=(32, 64)),
        ),
        shape=ndarray_shape_strategy(),
    )
    def test_tile(self, arr, shape):
        np_result = np.tile(arr, shape)
        rt_result = rt.tile(arr, shape)
        assert_array_equal_(rt_result, np_result)

    @pytest.mark.xfail(
        reason="https://jira/browse/RIP-358 discrepency between rt.trunc and np.trunc"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        data=data(),
        arr=arrays(
            shape=ndarray_shape_strategy(), dtype=floating_dtypes(sizes=(32, 64))
        ),
    )
    def test_trunc(self, data, arr):
        # TODO: Modify this to use the 'generate_array_and_where' strategy instead?
        where_arr = data.draw(arrays(shape=arr.shape, dtype=boolean_dtypes()))

        # Test #1: Returns new array containing results; no optional parameters provided.
        rt_output = rt.trunc(arr)
        np_output = np.trunc(arr)
        assert isinstance(rt_output, FastArray)
        assert_array_equal_(np.array(rt_output), np_output)

        # Test #2: Returns new array containing results; 'where' bitmask is provided.
        rt_output = rt.trunc(arr, where=where_arr)
        np_output = np.trunc(arr, where=where_arr)
        assert isinstance(rt_output, FastArray)
        assert_array_equal_(np.array(rt_output), np_output)

        # Test #3: Results written to array specified in 'out' parameter; no other optional params.
        rt_inplace = np.zeros_like(arr)
        np_inplace = np.zeros_like(arr)
        rt_output = rt.trunc(arr, out=rt_inplace)
        np_output = np.trunc(arr, out=np_inplace)
        assert_array_equal_(np.array(rt_inplace), np_inplace)
        # TODO: Add assertions for rt_output and np_output -- what are they expected to return when the 'out' parameter is specified?
        # assert isinstance(rt_output, FastArray, msg="riptable.trunc() did not return a FastArray")

        # Test #4: Results written to array specified in 'out' parameter; 'where' bitmask is provided.
        rt_inplace = np.zeros_like(arr)
        np_inplace = np.zeros_like(arr)
        rt_output = rt.trunc(arr, where=where_arr, out=rt_inplace)
        np_output = np.trunc(arr, where=where_arr, out=np_inplace)
        assert_array_equal_(np.array(rt_inplace), np_inplace)
        # TODO: Add assertions for rt_output and np_output -- what are they expected to return when the 'out' parameter is specified?
        # assert isinstance(rt_output, FastArray, msg="riptable.trunc() did not return a FastArray")

    @given(scalar=floating_scalar())
    def test_trunc_scalar(self, scalar):
        # test scalars
        rt_trunc_scalar = rt.trunc(scalar)
        np_trunc_scalar = np.trunc(scalar)
        assert_equal_(rt_trunc_scalar, np_trunc_scalar)

    @given(examples=generate_sample_test_integers(num_bits=8, signed=False))
    def test_uint8(self, examples):
        for i in examples:
            rt_uint8 = rt.uint8(i)
            np_uint8 = np.uint8(i)
            assert_equal_(rt_uint8, np_uint8)
            assert isinstance(rt_uint8, type(np_uint8))

            rt_plus = rt_uint8 + 1
            np_plus = np_uint8 + 1
            assert_equal_(rt_plus, np_plus)
            assert isinstance(rt_plus, type(np_plus))

            rt_minus = rt_uint8 - 1
            np_minus = np_uint8 - 1
            assert_equal_(rt_minus, np_minus)
            assert isinstance(rt_minus, type(np_minus))

    @given(examples=generate_sample_test_integers(num_bits=16, signed=False))
    def test_uint16(self, examples):
        for i in examples:
            rt_uint16 = rt.uint16(i)
            np_uint16 = np.uint16(i)
            assert_equal_(rt_uint16, np_uint16)
            assert isinstance(rt_uint16, type(np_uint16))

            rt_plus = rt_uint16 + 1
            np_plus = np_uint16 + 1
            assert_equal_(rt_plus, np_plus)
            assert isinstance(rt_plus, type(np_plus))

            rt_minus = rt_uint16 - 1
            np_minus = np_uint16 - 1
            assert_equal_(rt_minus, np_minus)
            assert isinstance(rt_minus, type(np_minus))

    @given(examples=generate_sample_test_integers(num_bits=32, signed=False))
    def test_uint32(self, examples):
        for i in examples:
            rt_uint32 = rt.uint32(i)
            np_uint32 = np.uint32(i)
            assert_equal_(rt_uint32, np_uint32)
            assert isinstance(rt_uint32, type(np_uint32))

            rt_plus = rt_uint32 + 1
            np_plus = np_uint32 + 1
            assert_equal_(rt_plus, np_plus)
            assert isinstance(rt_plus, type(np_plus))

            rt_minus = rt_uint32 - 1
            np_minus = np_uint32 - 1
            assert_equal_(rt_minus, np_minus)
            assert isinstance(rt_minus, type(np_minus))

    @given(examples=generate_sample_test_integers(num_bits=64, signed=False))
    def test_uint64(self, examples):
        for i in examples:
            rt_uint64 = rt.uint64(i)
            np_uint64 = np.uint64(i)
            assert_equal_(rt_uint64, np_uint64)
            assert isinstance(rt_uint64, type(np_uint64))

            rt_plus = rt_uint64 + 1
            np_plus = np_uint64 + 1
            assert_equal_(rt_plus, np_plus)
            assert isinstance(rt_plus, type(np_plus))

            rt_minus = rt_uint64 - 1
            np_minus = np_uint64 - 1
            assert_equal_(rt_minus, np_minus)
            assert isinstance(rt_minus, type(np_minus))

    @given(
        arr=arrays(
            shape=one_darray_shape_strategy(),
            elements=integers(-_MAX_INT, _MAX_INT),
            dtype=integer_dtypes(endianness="=", sizes=(64,)),
        )
    )
    def test_unique(self, arr):
        rt_result = rt.unique(arr)
        np_result = np.unique(arr)
        assert_array_equal_(
            rt_result,
            np_result,
            err_msg=f"arr dtype {arr.dtype} {arr}\nrt {rt_result}\nnp {np_result}",
        )

    @hypothesis.settings(suppress_health_check=[HealthCheck.too_slow])
    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6548 Riptable.vstack transposes arrays with shape (x,) and multiple (1,1)s. vstack also does not convert non-numbers to a FastArray"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(tuple_of_arrays=generate_tuples_of_arrays(all_same_width=True))
    def test_vstack(self, tuple_of_arrays):
        rt_err_type = None
        np_err_type = None
        rt_vstack = None
        np_vstack = None
        try:
            rt_vstack = rt.vstack(tuple_of_arrays)
        except ValueError:
            rt_err_type = ValueError
        try:
            np_vstack = np.vstack(tuple_of_arrays)
        except ValueError:
            np_err_type = ValueError

        if rt_err_type and np_err_type:
            assert rt_err_type == np_err_type
        else:
            assert isinstance(rt_vstack, FastArray)
            assert_array_equal_(np.array(rt_vstack), np_vstack)

    @pytest.mark.xfail(
        reason="See TestWhere.test_where for a materialized counterexample"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        arr=arrays(
            shape=ndarray_shape_strategy(),
            dtype=integer_dtypes(endianness="=", sizes=(32, 64)),
        )
    )
    def test_where(self, arr):
        min, mean, max = arr.min(), arr.mean(), arr.max()
        # Break this out into pytest cases; RiptableNumpyEquivalencyTests inherit from unittest
        # which does not allow for pytest.mark.parametrize.
        np_gt_min, rt_gt_min = np.where(arr > min), rt.where(arr > min)
        assert_array_equal_(
            np_gt_min,
            rt_gt_min,
            err_msg=f"array elements greater than the minimum {min}",
        )

        np_lt_mean, np_lt_mean = np.where(arr < mean), rt.where(arr < mean)
        assert_array_equal_(
            np_lt_mean, np_lt_mean, err_msg=f"array elements less than the mean {mean}"
        )

        np_gt_mean, np_gt_mean = np.where(arr > mean), rt.where(arr > mean)
        assert_array_equal_(
            np_gt_mean,
            np_gt_mean,
            err_msg=f"array elements greater than the mean {mean}",
        )

        np_lt_max, np_lt_max = np.where(arr < max), rt.where(arr < max)
        assert_array_equal_(
            np_lt_max, np_lt_max, err_msg=f"array elements less than the max {max}"
        )

    @given(
        shape=ndarray_shape_strategy(),
        dtype=ints_or_floats_dtypes(),
        order=sampled_from(("F", "C")),
    )
    def test_zeros(self, shape, dtype, order):
        rt_zeros = rt.zeros(shape, dtype, order)
        np_zeros = np.zeros(shape=shape, dtype=dtype, order=order)
        assert isinstance(rt_zeros, FastArray)

        assert_equal_(shape, rt_zeros.shape)
        assert_equal_(dtype.type, rt_zeros.dtype.type)
        assert_array_equal_(rt_zeros, np_zeros)

        # 1-D arrays always use Column-order. Otherwise, use the order specified
        if len(rt_zeros.shape) > 1 and min(rt_zeros.shape) > 1:
            assert np.isfortran(rt_zeros) == (order == "F")
        else:
            assert not np.isfortran(rt_zeros)

    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6495 kwargs not implemented in riptable.zeros"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        shape=ndarray_shape_strategy(),
        dtype=ints_or_floats_dtypes(),
        order=sampled_from(("F", "C")),
    )
    def test_zeros_kwargs(self, shape, dtype, order):
        rt_zeros = rt.zeros(shape=shape, dtype=dtype, order=order)
        np_zeros = np.zeros(shape=shape, dtype=dtype, order=order)
        assert isinstance(rt_zeros, FastArray)

        assert_equal_(shape, rt_zeros.shape)
        assert_equal_(dtype.type, rt_zeros.dtype.type)
        assert_array_equal_(rt_zeros, np_zeros)

        # 1-D arrays always use Column-order. Otherwise, use the order specified
        if len(rt_zeros.shape) > 1 and min(rt_zeros.shape) > 1:
            assert np.isfortran(rt_zeros) == (order == "F")
        else:
            assert not np.isfortran(rt_zeros)

    @pytest.mark.xfail(
        reason="https://jira/browse/SOQTEST-6563 riptable does not implement subok"
    )
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Please remove alongside xfail removal."
    )
    @given(
        arr=arrays(
            shape=ndarray_shape_strategy(), dtype=ints_floats_datetimes_and_timedeltas()
        ),
        dtype=ints_floats_or_complex_dtypes(),
        order=sampled_from(("C", "F", "K", "A")),
        subok=booleans(),
    )
    def test_zeros_like(self, arr, dtype, order, subok):
        rt_zeros_like = rt.zeros_like(arr, dtype=dtype, order=order, subok=subok)
        np_zeros_like = np.zeros_like(arr, dtype=dtype, order=order, subok=subok)

        assert_equal_(dtype, rt_zeros_like.dtype)
        assert_equal_(rt_zeros_like.shape, arr.shape)
        assert_array_equal_(rt_zeros_like, np_zeros_like)

        # 1-D arrays always use Column-order. Otherwise, use the order specified
        if (
            len(rt_zeros_like.shape) > 1
            and min(rt_zeros_like.shape) > 1
            and (order == "F" or ((order == "A" or order == "K") and np.isfortran(arr)))
        ):
            assert np.isfortran(rt_zeros_like)
        else:
            assert not np.isfortran(rt_zeros_like)

        if subok:
            assert isinstance(rt_zeros_like, FastArray)
        else:
            assert isinstance(rt_zeros_like, np.ndarray)
            # FastArray is a subclass of np.ndarray, so also ensure it is not a FastArray
            assert not isinstance(rt_zeros_like, FastArray)
