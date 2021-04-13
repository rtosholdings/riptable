"""
Equality/comparison assertion functions for arrays, used when implementing riptable unit tests.
"""
__all__ = [
    'assert_array_or_cat_equal',
    'assert_categorical_equal'
]

from typing import Optional

import numpy as np
from numpy.testing import assert_array_equal

from .. import Categorical, DateTimeBase, FastArray
from riptable.tests.test_utils import verbose_categorical

def assert_categorical_equal(
    x: Categorical,
    y: Categorical,
    *,
    err_msg: Optional[str] = '',
    relaxed_check: bool = False,
    check_cat_names: bool = True,
    exact_dtype_match: bool = True,
) -> None:
    # TODO: Add optional flag parameters to customize the checks; e.g. we might want
    #       to have a flag controlling whether we require the category names to match,
    #       or the category dtypes to match, or the underlying dtype to match.
    # TODO: Would it make sense to have numpy assert_array_equal() be a @multidispatch function?
    #       Or even just a @singledispatch?
    #       Then we could just register this function as a handler for rt.Categorical.
    if not isinstance(x, Categorical):
        raise TypeError("'x' is not a Categorical.")
    elif not isinstance(y, Categorical):
        raise TypeError("'y' is not a Categorical.")

    if isinstance(x.category_array, DateTimeBase) or isinstance(y.category_array, DateTimeBase):
        relaxed_check = True
        check_cat_names = False
        exact_dtype_match = False

    extra_info = f"{x} {verbose_categorical(x)}\n\n{y} {verbose_categorical(y)}"

    # Require the Categoricals to have the same CategoryMode.
    if x.category_mode != y.category_mode:
        raise AssertionError(
            f"The Categoricals have different `CategoryMode` values: '{x.category_mode}' vs. '{y.category_mode}'.\n{extra_info}"
        )

    # Require the Categoricals to be the same shape (i.e., the shape of their underlying array data).
    if x._fa.shape != y._fa.shape:
        raise AssertionError(f"The Categoricals have different shapes: {x._fa.shape} vs. {y._fa.shape}.\n{extra_info}")

    # Do we have the same number of Categories in both?
    x_cat_dict = x.category_dict
    y_cat_dict = y.category_dict
    if len(x_cat_dict) != len(y_cat_dict):
        raise AssertionError(f"The Categoricals have different category arities (# of category columns): {len(x_cat_dict)} vs. {len(y_cat_dict)}.\n{extra_info}")

    # If we're doing the relaxed check, just use .expand_array to expand the Categoricals to arrays
    # (or tuples of normal arrays) then check whether the pairs of arrays are equal.
    if relaxed_check:
        x_expanded_arrays = x.expand_dict
        y_expanded_arrays = y.expand_dict

        for i, ((x_cat_name, x_exp_arr), (y_cat_name, y_exp_arr)) in enumerate(zip(x_expanded_arrays.items(), y_expanded_arrays.items())):
            # Check category names match, in the same order.
            if check_cat_names and x_cat_name != y_cat_name:
                raise AssertionError(
                    f"The category arrays at index {i} have different names: '{x_cat_name}' vs '{y_cat_name}'.\n{extra_info}")

            # rt.DateTimeNano category_array loses type info by transitioning from DateTimeNano to FastArray
            if not isinstance(x.category_array, DateTimeBase):
                # See the equivalent check below in the 'strict' version for a comment
                # on why the types must be exactly the same.
                assert type(x_exp_arr) == type(y_exp_arr), \
                    f"The category arrays at index {i} have different types: {type(x_exp_arr)} vs. {type(y_exp_arr)}.\n{extra_info}"

            assert_array_equal(x_exp_arr, y_exp_arr, err_msg=err_msg + extra_info)
            isnan_kinds = 'iuf'  # dtype 'kinds' for which FastArray supports .isnan()
            if np.dtype(x_exp_arr.dtype).kind in isnan_kinds:
                assert_array_equal(
                    x_exp_arr.isnan(), y_exp_arr.isnan(),
                    err_msg=f"Different NaN/invalid values between the expanded arrays (x='{x_cat_name}', y='{y_cat_name}').\n{extra_info}")

    else:
        # Check categories match.
        for i, ((x_cat_name, x_cat_arr), (y_cat_name, y_cat_arr)) in enumerate(zip(x_cat_dict.items(), y_cat_dict.items())):
            # Check category names match, in the same order.
            if check_cat_names and x_cat_name != y_cat_name:
                raise AssertionError(
                    f"The category arrays at index {i} have different names: '{x_cat_name}' vs '{y_cat_name}'.\n{extra_info}")

            # rt.DateTimeNano category_array loses type info by transitioning from DateTimeNano to FastArray
            if not isinstance(x.category_array, DateTimeBase):
                # The category array types must match exactly; if we only checked the dtypes (below) or that
                # one type was equal to or a subtype of the other, the semantic meaning of the data could be
                # lost. For example, consider if this array was a raw integer array in one Categorical and a
                # Date array in the other -- even if they contain the same underlying data, the Date type provides
                # some different interpretation/semantics on top of that data compared to a plain array.
                assert type(x_cat_arr) == type(y_cat_arr), \
                    f"The category arrays at index {i} have different types: {type(x_cat_arr)} vs. {type(y_cat_arr)}.\n{extra_info}"

            # Check category dtypes match. If we're doing the relaxed form of this check,
            # only require that the dtypes have the same 'kind'.
            x_cat_arr_dtype = np.dtype(x_cat_arr.dtype)
            y_cat_arr_dtype = np.dtype(y_cat_arr.dtype)
            if exact_dtype_match:
                assert x_cat_arr_dtype == y_cat_arr_dtype, extra_info
            else:
                assert x_cat_arr_dtype.kind == y_cat_arr_dtype.kind, extra_info
                # TODO: we don't currently allow categoricals to be built out of recarrays / structured dtypes;
                #       but in case we ever do, we should add a check here so that if x_cat_arr_dtype.kind == 'V'
                #       we'll recursively apply this check to the dtype fields.

            # Strict check (for now) -- the category arrays *must* match.
            # We also need to use .isnan() on the arrays here to handle nan-equality (we want to consider nans equal
            # to each other for the purposes of this comparison).
            # TODO: Add a flag to relax this check, for when we just want to make sure that one set of categories
            #       is a subset of the other, and that every category in their intersection is assigned to the same
            #       value in both.
            assert_array_equal(x_cat_arr, y_cat_arr, err_msg=err_msg + extra_info)
            isnan_kinds = 'iuf'  # dtype 'kinds' for which FastArray supports .isnan()
            if x_cat_arr_dtype.kind in isnan_kinds:
                assert_array_equal(
                    x_cat_arr.isnan(), y_cat_arr.isnan(),
                    err_msg=f"Different NaN/invalid values between the category arrays at index {i}.\n{extra_info}")


def assert_array_or_cat_equal(
    x: np.ndarray,
    y: np.ndarray,
    *,
    err_msg: Optional[str] = '',
    relaxed_cat_check: bool = False,
    check_cat_names: bool = True,
    exact_dtype_match: bool = True,
):
    # TODO: Add optional flag parameters to customize the checks; e.g. we might want
    #       to have a flag controlling whether we require the category names to match,
    #       or the category dtypes to match, or the underlying dtype to match.
    # TODO: Would it make sense to have numpy assert_array_equal() be a @multidispatch function?
    #       Or even just a @singledispatch?
    #       Then we could just register this function as a handler for rt.Categorical.

    assert type(x) == type(y)

    # Categorical needs special handling, otherwise we run into TypeError in
    # Categorical.match_str_to_category when assert_array_equal() tries to index
    # it with an inf/-inf.
    if isinstance(x, Categorical):
        assert_categorical_equal(
            x, y, err_msg=err_msg, relaxed_check=relaxed_cat_check,
            check_cat_names=check_cat_names, exact_dtype_match=exact_dtype_match)

    # TODO: Implement handler for DateTimeNano which checks the timezones match up for the two instances.

    elif isinstance(x, FastArray) and not type(x) == FastArray:
        # For FastArray-derived array types like Date, DateTimeNano, etc.
        # that don't otherwise have a handler defined above.
        # These don't work well with numpy's assert_array_equal when passed directly,
        # so we pass in views of the underlying numpy arrays.
        assert_array_equal(x._np, y._np)

    else:
        # For other array types, we can just defer to numpy array assertions.
        # assert_allclose with equal_nan to compare NaN values
        assert_array_equal(x, y, err_msg=err_msg)

