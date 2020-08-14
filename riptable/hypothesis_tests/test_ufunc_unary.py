from math import nan
from typing import Any, List, Callable

import numpy as np
import riptable as rt
from riptable import FastArray, FA

import pytest
import unittest
from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal,
    assert_equal,
    assert_allclose,
    assert_almost_equal,
)

import hypothesis
from hypothesis import assume, event, example, given, HealthCheck
from hypothesis.extra.numpy import (
    arrays,
    boolean_dtypes,
    floating_dtypes,
    integer_dtypes,
    unsigned_integer_dtypes,
)
from hypothesis.strategies import one_of

# riptable custom Hypothesis strategies
from .strategies.helper_strategies import (
    generate_list_ndarrays,
    generate_lists,
    generate_ndarrays,
    ndarray_shape_strategy,
    one_darray_shape_strategy,
)

from riptable.Utils.teamcity_helper import is_running_in_teamcity


# TODO: Add tests for the following functions:
# - abs
# - mean/nanmean
# - median/nanmedian
# - std/nanstd
# - var/nanvar


class NanUnawareTestImpl:
    """
    Implementations of tests for "nan-unaware" unary ufunc functions,
    parameterized so the function under test can be passed in.
    """

    # TODO: Extend this to also check integer dtypes; need to use rt.isnan instead of np.isnan
    #       because rt.isnan will recognize the riptable invalid values.
    @staticmethod
    def test_isnan_implies_nan_result(
        func: Callable[[np.ndarray], Any], f_arr: rt.FastArray
    ):
        """
        Check that a nan-unaware unary ufunc propagates any NaNs in the input to the output.

        One or more NaNs in the input array should result in the function returning a NaN.

        Parameters
        ----------
        func : callable
        f_arr : rt.FastArray
            A FastArray to test the supplied function implementation with.
        """
        # Does the input array contain one or more NaNs?
        have_nans = np.any(np.isnan(f_arr._np))
        event(f'have NaNs = {have_nans}')

        # Call the function.
        result = func(f_arr)

        # If the input contained one or more NaNs, the result should have been a NaN too.
        assert have_nans == np.isnan(result)


class TestMax:
    # TODO: Extend this to also check integer dtypes (dtype=ints_or_floats_dtypes())
    @pytest.mark.xfail(
        reason="This test exposes a known bug around NaN-handling that needs to be fixed."
    )
    @given(
        arr=arrays(
            shape=one_darray_shape_strategy(),
            dtype=floating_dtypes(endianness="=", sizes=(32, 64)),
        )
    )
    @pytest.mark.parametrize("func_type", ['module', 'member'])
    def test_isnan_implies_nan_result(self, arr, func_type):
        """
        Check how :func:`rt.max` handles NaN values.

        One or more NaNs in the input array should result in the function returning a NaN.
        """
        # Get the function implementation based on how we want to call it.
        if func_type == 'module':
            test_func = lambda x: rt.max(x)
        elif func_type == 'member':
            test_func = lambda x: x.max()
        else:
            raise ValueError(
                f"Unhandled value '{func_type}' specified for the function type."
            )

        # Wrap the input as a FastArray to ensure we'll get the riptable implementation of the function.
        arr = rt.FA(arr)

        # Call the test implementation.
        NanUnawareTestImpl.test_isnan_implies_nan_result(test_func, arr)


class TestMin:
    # TODO: Extend this to also check integer dtypes (dtype=ints_or_floats_dtypes()).
    @pytest.mark.xfail(
        reason="This test exposes a known bug around NaN-handling that needs to be fixed."
    )
    @given(
        arr=arrays(
            shape=one_darray_shape_strategy(),
            dtype=floating_dtypes(endianness="=", sizes=(32, 64)),
        )
    )
    @pytest.mark.parametrize("func_type", ['module', 'member'])
    def test_isnan_implies_nan_result(self, arr, func_type):
        """
        Check how :func:`rt.min` handles NaN values.

        One or more NaNs in the input array should result in the function returning a NaN.
        """
        # Get the function implementation based on how we want to call it.
        if func_type == 'module':
            test_func = lambda x: rt.min(x)
        elif func_type == 'member':
            test_func = lambda x: x.min()
        else:
            raise ValueError(
                f"Unhandled value '{func_type}' specified for the function type."
            )

        # Wrap the input as a FastArray to ensure we'll get the riptable implementation of the function.
        arr = rt.FA(arr)

        # Call the test implementation.
        NanUnawareTestImpl.test_isnan_implies_nan_result(test_func, arr)


class TestSum:
    # TODO: Extend this to also check integer dtypes (dtype=ints_or_floats_dtypes()).
    @pytest.mark.xfail(
        reason="This test exposes a known bug around NaN-handling that needs to be fixed."
    )
    @given(
        arr=arrays(
            shape=one_darray_shape_strategy(),
            dtype=floating_dtypes(endianness="=", sizes=(32, 64)),
        )
    )
    @pytest.mark.parametrize("func_type", ['module', 'member'])
    def test_isnan_implies_nan_result(self, arr, func_type):
        """
        Check how :func:`rt.sum` handles NaN values.

        One or more NaNs in the input array should result in the function returning a NaN.
        """
        # Get the function implementation based on how we want to call it.
        if func_type == 'module':
            test_func = lambda x: rt.sum(x)
        elif func_type == 'member':
            test_func = lambda x: x.sum()
        else:
            raise ValueError(
                f"Unhandled value '{func_type}' specified for the function type."
            )

        # Wrap the input as a FastArray to ensure we'll get the riptable implementation of the function.
        arr = rt.FA(arr)

        # Call the test implementation.
        NanUnawareTestImpl.test_isnan_implies_nan_result(test_func, arr)


class NanAwareTestImpl:
    """
    Implementations of tests for "nan-aware" unary ufunc functions,
    parameterized so the function under test can be passed in.
    """

    @staticmethod
    # TODO: Extend this to also check integer dtypes; need to use rt.isnan instead of np.isnan
    # because rt.isnan will recognize the riptable invalid values.
    def test_nan_awareness(
        nan_aware_func: Callable[[np.ndarray], Any],
        nan_unaware_func: Callable[[np.ndarray], Any],
        f_arr: rt.FastArray,
    ):
        """
        Check how a nan-aware function handles NaN values by comparing it against the corresponding nan-unaware function.

        Call `nan_aware_func` with an array, then remove any NaNs from the array and call
        `nan_unaware_func` with the 'clean' array. The results should match.

        Parameters
        ----------
        nan_aware_func : callable
        nan_unaware_func : callable
        f_arr : rt.FastArray
        """
        # Determine which elements of the array are NaN.
        nan_mask = np.isnan(f_arr._np)

        # Does the input array contain one or more NaNs?
        have_nans = np.any(nan_mask)
        event(f'have NaNs = {have_nans}')

        # Call the nan-aware function.
        rt_nanresult = nan_aware_func(f_arr)

        # Remove any NaNs / invalids from the input array.
        f_arr_clean = f_arr[~nan_mask]

        # Until such time (if ever) we support calling reduction functions with zero-length
        # arrays, handle that case explicitly here so the test works as expected.
        if len(f_arr_clean) == 0:
            rt_result = nan
        else:
            # Call the nan-unaware function.
            rt_result = nan_unaware_func(f_arr_clean)

        # The result computed by the nan-aware function operating on the input array should match
        # that computed by the nan-unaware function operating on the cleaned array.
        assert (rt_nanresult == rt_result) or (
            np.isnan(rt_nanresult) and (np.isnan(rt_result))
        )


class TestNanMax:
    # TODO: Extend this to also check integer dtypes (dtype=ints_or_floats_dtypes());
    # need to use rt.isnan instead of np.isnan because it'll recognize the riptable invalid values.
    @hypothesis.settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        arr=arrays(
            shape=one_darray_shape_strategy(),
            dtype=floating_dtypes(endianness="=", sizes=(32, 64)),
        )
    )
    @pytest.mark.parametrize("func_type", ['module', 'member'])
    def test_nan_awareness(self, arr, func_type):
        """
        Check how :func:`rt.nanmax` handles NaN values by comparing it against :func:`rt.max`.

        Call :func:`rt.nanmax` with an array, then remove any NaNs from the array and call
        :func:`np.max` with the 'clean' array. The results should match.
        """
        # Get the function implementation based on how we want to call it.
        if func_type == 'module':
            test_func = lambda x: rt.nanmax(x)
        elif func_type == 'member':
            test_func = lambda x: x.nanmax()
        else:
            raise ValueError(
                f"Unhandled value '{func_type}' specified for the function type."
            )

        # Get the nan-unaware version of the function.
        nan_unaware_func = lambda x: rt.max(x)

        # Wrap the input as a FastArray to ensure we'll get the riptable implementation of the function.
        arr = rt.FA(arr)

        # Call the test implementation.
        NanAwareTestImpl.test_nan_awareness(test_func, nan_unaware_func, arr)


class TestNanMin:
    # TODO: Extend this to also check integer dtypes (dtype=ints_or_floats_dtypes());
    # need to use rt.isnan instead of np.isnan because it'll recognize the riptable invalid values.
    @hypothesis.settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        arr=arrays(
            shape=one_darray_shape_strategy(),
            dtype=floating_dtypes(endianness="=", sizes=(32, 64)),
        )
    )
    @pytest.mark.parametrize("func_type", ['module', 'member'])
    def test_nan_awareness(self, arr, func_type):
        """
        Check how :func:`rt.nanmin` handles NaN values by comparing it against :func:`rt.min`.

        Call :func:`rt.nanmin` with an array, then remove any NaNs from the array and call
        :func:`np.min` with the 'clean' array. The results should match.
        """
        # Get the function implementation based on how we want to call it.
        if func_type == 'module':
            test_func = lambda x: rt.nanmin(x)
        elif func_type == 'member':
            test_func = lambda x: x.nanmin()
        else:
            raise ValueError(
                f"Unhandled value '{func_type}' specified for the function type."
            )

        # Get the nan-unaware version of the function.
        nan_unaware_func = lambda x: rt.min(x)

        # Wrap the input as a FastArray to ensure we'll get the riptable implementation of the function.
        arr = rt.FA(arr)

        # Call the test implementation.
        NanAwareTestImpl.test_nan_awareness(test_func, nan_unaware_func, arr)


class TestNanSum:
    # TODO: Extend this to also check integer dtypes (dtype=ints_or_floats_dtypes());
    # need to use rt.isnan instead of np.isnan because it'll recognize the riptable invalid values.
    @pytest.mark.xfail(
        reason="Very small differences between sum and nansum; likely ignorable differences due to rounding, but let's investigate to be sure."
    )
    @given(
        arr=arrays(
            shape=one_darray_shape_strategy(),
            dtype=floating_dtypes(endianness="=", sizes=(32, 64)),
        )
    )
    @pytest.mark.parametrize("func_type", ['module', 'member'])
    def test_nan_awareness(self, arr, func_type):
        """
        Check how :func:`rt.nansum` handles NaN values by comparing it against :func:`rt.sum`.

        Call :func:`rt.nansum` with an array, then remove any NaNs from the array and call
        :func:`np.sum` with the 'clean' array. The results should match.
        """
        # Get the function implementation based on how we want to call it.
        if func_type == 'module':
            test_func = lambda x: rt.nansum(x)
        elif func_type == 'member':
            test_func = lambda x: x.nansum()
        else:
            raise ValueError(
                f"Unhandled value '{func_type}' specified for the function type."
            )

        # Get the nan-unaware version of the function.
        nan_unaware_func = lambda x: rt.sum(x)

        # Wrap the input as a FastArray to ensure we'll get the riptable implementation of the function.
        arr = rt.FA(arr)

        # Call the test implementation.
        NanAwareTestImpl.test_nan_awareness(test_func, nan_unaware_func, arr)
