import sys
import pytest
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal

import riptide_cpp as rc
import riptable as rt


# TODO: Implement tests for sorting, binning, ismember, hstack, etc.


class TestBasicMath:
    """Unit tests for the 'BasicMath' functions in riptide_cpp."""

    # TODO: Use pytest.parametrize() to make it easy to add additional cases for this test.
    #       Consider performing a combinatorial sweep over various inputs and functions (though maybe we'd run this under
    #       the long-running tests build instead then).
    def test_basicmath_two_inputs(self):
        arg0 = rt.FA([1, 2, 3, 4, 5]).tile(40)
        arg1 = np.array(1, dtype=np.int64)
        tupleargs = (arg0, arg1)

        final_num = 7
        fastfunction = rt.rt_enum.MATH_OPERATION.CMP_EQ

        # Try the operation.
        result = rc.BasicMathTwoInputs(tupleargs, fastfunction, final_num)

        assert_array_equal(result, rt.FA([True, False, False, False, False]).tile(40))


class TestIndexing:
    """Unit tests for various indexing functions in riptide_cpp."""

    def test_boolindex(self) -> None:
        # rc.BooleanIndex(...)
        pytest.skip("Test not yet implemented.")

    @pytest.mark.skipif(
        sys.platform != 'win32',
        reason="This test fails on Linux, perhaps due to int/long/longlong mismatch -- need to investigate.")
    def test_mbget_int32_int64(self) -> None:
        arg0 = rt.FA([1, 2, 3, 4, 5, 6], dtype=np.int32)
        arg1 = rt.FA(
            [-9223372036854775808, -9223372036854775808, -9223372036854775808, 0, 3, -9223372036854775808, 1, 4,
             -9223372036854775808, 2, 5, -9223372036854775808], dtype=np.int64)

        # Try the operation
        result = rc.MBGet(arg0, arg1)
        arr_inv = arg0.inv
        expected = rt.FA([arr_inv, arr_inv, arr_inv, 1, 4, arr_inv, 2, 5, arr_inv, 3, 6, arr_inv])
        assert_array_equal(result, expected)

    @pytest.mark.skipif(
        sys.platform != 'win32',
        reason="This test fails on Linux, perhaps due to int/long/longlong mismatch -- need to investigate.")
    def test_mbget_bytes_int64(self) -> None:
        arg0 = rt.FA(['x', 'y', 'z', 'q', 'w', 't'])
        arg1 = rt.FA(
            [rt.int64.inv, rt.int64.inv, rt.int64.inv, 0, 3, rt.int64.inv, 1, 4,
             rt.int64.inv, 2, 5, rt.int64.inv], dtype=np.int64)

        # Try the operation
        result = rc.MBGet(arg0, arg1)
        arr_inv = arg0.inv
        expected = rt.FA([arr_inv, arr_inv, arr_inv, b'x', b'q', arr_inv, b'y', b'w',
                          arr_inv, b'z', b't', arr_inv])
        assert_array_equal(result, expected)


class TestConversions:
    """Unit tests for conversion functions in riptide_cpp."""

    def test_convert_safe(self) -> None:
        # rc.ConvertSafe(...)
        pytest.skip("Test not yet implemented.")

    def test_convert_unsafe(self) -> None:
        # rc.ConvertUnsafe(...)
        pytest.skip("Test not yet implemented.")


class TestReductions:
    """Unit tests for reduction functions in riptide_cpp."""

    def test_reduce(self) -> None:
        # rc.Reduce(...)
        pytest.skip("Test not yet implemented.")


class TestLedgerFunctions:
    """Unit tests for ledger functions."""

    def test_foo(self) -> None:
        # rc.LedgerFunction(...)
        pytest.skip("Test not yet implemented.")


class TestCRC:
    """Unit tests for the CRC32/CRC64 calculation on arrays."""

    def test_raises_on_bad_arg_type(self) -> None:
        """Test that rc.CalculateCRC raises an error when called with the wrong argument type."""
        bytes = b'abcdefghi'
        with pytest.raises(TypeError):
            rc.CalculateCRC(bytes)

    def test_disallow_noncontig(self) -> None:
        """Test that rc.CalculateCRC raises an error when calling with a non-contiguous array."""
        arr = np.arange(100)
        with pytest.raises(ValueError):
            rc.CalculateCRC(arr[::2])

    def test_basic(self) -> None:
        # Array size (in bytes, since that's what CRC is going to look at).
        # This is a reasonably large, PRIME value so we make sure all code paths
        # are exercised (e.g. don't want to fall into some special case that
        # only handles arrays smaller than 1000 bytes or something).
        array_size_bytes = 82727

        # Create a "random" array (but seed the RNG so this test is deterministic).
        rng = default_rng(123456)
        bytes = np.frombuffer(rng.bytes(array_size_bytes), dtype=np.int8)

        # Calculate the CRC32C of several slices of the array.
        # We'll use these values to check if we're seeing hash-like cascading behavior;
        # this also allows us to avoid test breakage if the RNG algorithm changes
        # or otherwise has differences between numpy versions or systems.
        results = np.empty(shape=500, dtype=np.uint64)  # NOTE: dtype needs to be set based on the expected output type of the CRC
        for i in range(len(results)):
            # Calculate the CRC32C or CRC64 of the array.
            crc_result = rc.CalculateCRC(bytes[i:])
            results[i] = crc_result

        # What's the average hash value? It should be roughly half the range of a uint32.
        avg_hash = results.mean()
        uniform_mean = float(np.iinfo(results.dtype).max) / 2.0  # Mean of a discrete uniform distribution over [0, results.dtype.max]
        avg_hash_accuracy = avg_hash / uniform_mean

        # How far off is the hash avg from the midpoint of uint32?
        # It should only be by a few % (which should be lower / closer to the midpoint as the sample count increases).
        hash_avg_error = abs(1.0 - avg_hash_accuracy)
        assert hash_avg_error < 0.05,\
            f'The mean of the calculated hash values differs from the expected mean {np.dtype(results.dtype).name}.max/2.0 by {hash_avg_error * 100}%.'

        # Check the variance too to see if it's reasonably close to
        # what we'd expect from a uniform distribution.
        hash_var = results.astype(np.float64).var()
        # This is population variance (not sample var), but it's good enough for this test.
        uniform_variance = (float(np.iinfo(np.uint64).max) ** 2 - 1) / 12.0
        hash_var_accuracy = hash_var / uniform_variance

        hash_var_error = abs(1.0 - hash_var_accuracy)
        assert hash_var_error < 0.10,\
            f'The variance of the calculated hash values differs from the expected variance by {hash_var_error * 100}%.'
