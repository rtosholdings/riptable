import pytest
import numpy as np
from numpy.testing import assert_array_equal

import riptable as rt
from riptable.rt_utils import crc_match

@pytest.mark.parametrize(
    "arrs,expected",
    [
        # Some basic positive test cases.
        pytest.param(
            [
                rt.FastArray([b'ABCDEFGHIJKL'], dtype='|S32'),
                rt.FastArray([b'ABCDEFGHIJKL'], dtype='|S32'),
            ],
            True,
            id="S32__S32",
        ),
        # Test cases for verifying array shape is checked even when
        # calculated CRC value of the raw data is the same.
        pytest.param(
            [np.arange(1, 10), np.arange(10)], False, id="int__int__leading-zero"
        ),
        pytest.param(
            [
                rt.FastArray([b'2019/12/21'], dtype='|S32'),
                rt.FastArray([b'', b'2019/12/21'], dtype='|S32'),
            ],
            False,
            id="ascii__ascii__leading-empty",
        ),
    ],
)
def test_crc_match(arrs, expected):
    result = crc_match(arrs)
    assert result == expected


def test_mbget_no_default_uses_invalid():
    data = np.arange(start=3, stop=53, dtype=np.int8).view(rt.FA)
    indices = rt.FA([0, 25, -40, 17, 100, -80, 50, -51, 35])

    valid_indices = np.logical_and(indices >= -50, indices < 50)

    # Call the 'mbget' function without providing an explicit default value.
    result = rt.mbget(data, indices)

    # The resulting array should have the same dtype as the values/data array.
    assert data.dtype == result.dtype, "The result has a different dtype than the values/data array."

    # The elements with out-of-bounds indices should have been assigned the
    # riptable NA/sentinel value because a default value was not explicitly specified.
    assert_array_equal(valid_indices, rt.isnotnan(result))

    # Check that the valid indices fetched the correct values.
    assert_array_equal(rt.FA([3, 28, 13, 20, 38]), result[valid_indices])


def test_mbget_with_explicit_default():
    data = np.arange(start=3, stop=53, dtype=np.int8).view(rt.FA)
    indices = rt.FA([0, 25, -40, 17, 100, -80, 50, -51, 35])

    valid_indices = np.logical_and(indices >= -50, indices < 50)

    # Call the 'mbget' function, providing an explicit default value.
    default_value = 123
    result = rt.mbget(data, indices, d=default_value)

    # The resulting array should have the same dtype as the values/data array.
    assert data.dtype == result.dtype, "The result has a different dtype than the values/data array."

    # The elements with out-of-bounds indices should have been assigned the
    # explicitly-specified default value.
    assert_array_equal(valid_indices, result != default_value)

    # Check that the valid indices fetched the correct values.
    assert_array_equal(rt.FA([3, 28, 13, 20, 38]), result[valid_indices])


@pytest.mark.xfail(
    reason="BUG mbget does not widen the output dtype to accommodate the default value, nor does it validate the "
           "default value will fit into the data/values dtype, so the default value is silently truncated.")
def test_mbget_with_too_large_explicit_default():
    data = np.arange(start=3, stop=53, dtype=np.int8).view(rt.FA)
    indices = rt.FA([0, 25, -40, 17, 100, -80, 50, -51, 35])

    valid_indices = np.logical_and(indices >= -50, indices < 50)

    # Call the 'mbget' function, providing an explicit default value
    # which is too large to be represented by the dtype of the data/values array.
    default_value = 1234
    result = rt.mbget(data, indices, d=default_value)

    # The resulting array will need to have a larger dtype than the original data array
    # to accommodate the explicit default value that was too large for the data's dtype.
    assert data.dtype != result.dtype, "The result has the same dtype as the values/data array."
    assert np.dtype(data.dtype).itemsize < np.dtype(result.dtype).itemsize, "The result dtype is not larger than the values/data dtype."

    # The elements with out-of-bounds indices should have been assigned the
    # explicitly-specified default value.
    assert_array_equal(valid_indices, result != default_value)

    # Check that the valid indices fetched the correct values.
    assert_array_equal(rt.FA([3, 28, 13, 20, 38]), result[valid_indices])
