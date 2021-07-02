import unittest
import pytest
import numpy as np
from numpy.testing import assert_array_equal

import riptable as rt
from riptable import FastArray


rand_nums = np.random.choice(30, 100)
numeric_types = 'fd' + 'bB' + 'hH' + 'iIlL' + 'qQpP'  # + '?'


@pytest.mark.parametrize("dt_char", list(numeric_types))
@pytest.mark.parametrize("lex", [False, True])
def test_numeric_accuracy(dt_char, lex):
    target_dtype = np.dtype(dt_char)

    # N.B. It's important to use .view() instead of .astype() here so the case
    #      for np.bool_ is tested thoroughly; .astype(bool) will convert the
    #      values to 0/1, but .view(bool) will not -- so it will check that
    #      lower-level logic does proper C-style bool conversion (zero=false, non-zero=true).
    nums = rand_nums.view(target_dtype)
    np_un, np_idx, np_inv, np_cnt = np.unique(
        nums, return_index=True, return_inverse=True, return_counts=True
    )
    fa_un, fa_idx, fa_inv, fa_cnt = rt.unique(
        nums, return_index=True, return_inverse=True, return_counts=True, lex=lex
    )

    assert_array_equal(
        np_un,
        fa_un,
        err_msg=f"Results did not match for unique items for dtype {target_dtype.name}.",
    )
    assert_array_equal(
        np_idx,
        fa_idx,
        err_msg=f"Results did not match for unique fancy index for dtype {target_dtype.name}.",
    )
    assert_array_equal(
        np_inv,
        fa_inv,
        err_msg=f"Results did not match for inverse fancy index for dtype {target_dtype.name}.",
    )
    assert_array_equal(
        np_cnt,
        fa_cnt,
        err_msg=f"Results did not match for unique counts for dtype {target_dtype.name}.",
    )


@pytest.mark.parametrize("dt_char", list('US'))
@pytest.mark.parametrize("lex", [False, True])
def test_string_accuracy(dt_char, lex):
    target_dtype = np.dtype(dt_char)

    rand_strings = (
        np.random.choice(['test_string' + str(i) for i in range(30)], 100)
        .astype(target_dtype, copy=False)
        .view(FastArray)
    )
    np_un, np_idx, np_inv, np_cnt = np.unique(
        rand_strings, return_index=True, return_inverse=True, return_counts=True
    )
    fa_un, fa_idx, fa_inv, fa_cnt = rt.unique(
        rand_strings,
        return_index=True,
        return_inverse=True,
        return_counts=True,
        lex=lex,
    )

    assert_array_equal(
        np_un,
        fa_un,
        err_msg=f"Results did not match for unique items for dtype {target_dtype.name}.",
    )
    assert_array_equal(
        np_idx,
        fa_idx,
        err_msg=f"Results did not match for unique fancy index for dtype {target_dtype.name}.",
    )
    assert_array_equal(
        np_inv,
        fa_inv,
        err_msg=f"Results did not match for inverse fancy index for dtype {target_dtype.name}.",
    )
    assert_array_equal(
        np_cnt,
        fa_cnt,
        err_msg=f"Results did not match for unique counts for dtype {target_dtype.name}.",
    )


# TODO: Once the underlying logic in Grouping and/or rt.unique() has been fixed, add '?'
#       back to the 'numeric_types' variable above, then this test can be removed
#       (since bools will be included in the test_numeric_accuracy test).
@pytest.mark.xfail(reason="RIP-371: Grouping returns incorrect results for bool array")
@pytest.mark.parametrize("dt_char", list('?'))
@pytest.mark.parametrize("lex", [False, True])
def test_bool_accuracy(dt_char, lex):
    target_dtype = np.dtype(dt_char)

    # N.B. It's important to use .view() instead of .astype() here so the case
    #      for np.bool is tested thoroughly; .astype(bool) will convert the
    #      values to 0/1, but .view(bool) will not -- so it will check that
    #      lower-level logic does proper C-style bool conversion (zero=false, non-zero=true).
    nums = rand_nums.view(target_dtype)
    np_un, np_idx, np_inv, np_cnt = np.unique(
        nums, return_index=True, return_inverse=True, return_counts=True
    )
    fa_un, fa_idx, fa_inv, fa_cnt = rt.unique(
        nums, return_index=True, return_inverse=True, return_counts=True, lex=lex
    )

    assert_array_equal(
        np_un,
        fa_un,
        err_msg=f"Results did not match for unique items for dtype {target_dtype.name}.",
    )
    assert_array_equal(
        np_idx,
        fa_idx,
        err_msg=f"Results did not match for unique fancy index for dtype {target_dtype.name}.",
    )
    assert_array_equal(
        np_inv,
        fa_inv,
        err_msg=f"Results did not match for inverse fancy index for dtype {target_dtype.name}.",
    )
    assert_array_equal(
        np_cnt,
        fa_cnt,
        err_msg=f"Results did not match for unique counts for dtype {target_dtype.name}.",
    )


if __name__ == "__main__":
    tester = unittest.main()
