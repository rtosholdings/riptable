import gc
import sys
import warnings

import numpy as np
import riptable as rt
from typing import Tuple, Set, TYPE_CHECKING, Optional
from collections import deque
from collections.abc import Iterable
from riptable import Categorical
from riptable.tests.test_utils import verbose_categorical
from numpy.testing import (
    assert_array_equal,
    assert_allclose,
    assert_equal,
    assert_almost_equal,
)


if TYPE_CHECKING:
    from ..rt_categorical import Categorical

name = lambda obj: obj.__class__.__name__


def assert_equal_(actual, expected, decimal=7, err_msg='', verbose=True):
    """
    A wrapper around numpy testing helpers for assertions that defer to an exact or approximate assertion
    depending on the type.

    Parameters
    ----------
    actual
        Object to check.
    expected
        Desired object.
    decimal
        Desired decimal precision (default 7).
    err_msg
        Error message in event of failure (default '').
    verbose
        In event of failure with verbose set to True, the actual and expected are part of the error message (default True).

    Raises
    ------
    AssertionError
        If actual not equal to expected up to specified precision depending if the expected type is inexact.

    See Also
    --------
    assert_equal, assert_almost_equal

    Note
    ----
    Default params resemble numpy assert_equal* default params.
    """
    # TODO delegate to corresponding assertion based on type of expected object; e.g.,
    # int types would defer to assert_equal, while float types would defer to assert_almost_equal
    # Need to consider objects and other types as opposed to just the numeric hierarchy.
    try:
        assert_equal(actual, expected, err_msg, verbose)
    except AssertionError:
        assert_almost_equal(actual, expected, decimal, err_msg, verbose)


def assert_array_equal_(actual, expected, decimal=6, err_msg='', verbose=True):
    """
    A wrapper around numpy testing helpers for array assertion that are aware of dtypes and dispatch
    to exact or approximate equality assertions.

    Parameters
    ----------
    actual: array_like
        The actual array-like object to check.
    expected: array_like
        The expected array-like object.
    decimal
        Desired decimal precision (default 7).
    err_msg
        Error message in event of failure (default '').
    verbose
        In event of failure with verbose set to True, the actual and expected are part of the error message (default True).

    Raises
    ------
    AssertionError
        If actual not equal to expected up to specified precision depending if the expected type is inexact.   ------

    See Also
    --------
    assert_array_equal, assert_allclose, assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Note
    ----
    Default params resemble numpy assert_array_equal* default params.
    """
    # TODO delegate to corresponding assertion based on dtype
    # E.g., exact equality for integer types and approximate equality with tolerance for floating types
    # Need to consider objects and other types as opposed to just the numeric hierarchy.
    try:
        assert_array_equal(np.array(actual), expected)
    except AssertionError:
        assert_allclose(np.array(actual), expected)


def assert_categorical_equal(
    actual: Categorical,
    expected: Categorical,
    check_expanded_array: Optional[bool] = True,
    verbose: Optional[bool] = False,
) -> None:
    """
    Assert ``actual`` and ``expected`` Categoricals, of the various ``CategoryModes``, have no mismatch.

    Parameters
    ----------
    actual: Categorical
    expected: Categorical
    check_expanded_array: bool, optional
        Check the expanded Categoricals are equal (default True).
    verbose: bool, optional
        Logs a verbose representation of ``actual`` and ``expected`` Categorical on failure.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ```actual`` or ``expected`` are not of type ``Categorical``.
    AssertionError
        If any mismatch in ``actual`` and ``expected`` exists.

    Note
    ----
    ``check_expanded_array`` is expensive since it will re-construct the array the Categorical
    is derived from. It may be sufficient to check the underlying ``FastArray`` as well as
    the categories.
    """
    fn = 'assert_categorical_equal'
    try:
        # check actual and expected types
        if not isinstance(expected, Categorical):
            raise ValueError(
                f'{fn}: expected type Categorical, got {type(expected)} for "expected" parameter'
            )
        if not isinstance(actual, Categorical):
            raise ValueError(
                f'{fn}: expected type Categorical, got {type(actual)} for "actual" parameter'
            )

        # check category_mode
        assert (
            actual.category_mode == expected.category_mode
        ), f'{fn}: category mode mismatch\nactual {repr(rt.rt_enum.CategoryMode(actual.category_mode))}\nexpected {repr(rt.rt_enum.CategoryMode(expected.category_mode))}'

        # check category_array
        if (
            expected.category_mode == rt.rt_enum.CategoryMode.StringArray
            or expected.category_mode == rt.rt_enum.CategoryMode.NumericArray
        ):
            assert_array_equal(
                actual.category_array, expected.category_array, err_msg=f'{fn}: mismatch in "category_array"'
            )
            assert_array_equal(
                actual.category_array, expected.category_array, err_msg=f'{fn}: mismatch in "category_array"'
            )
        else:  # TODO implement category_mode checks for MultiKey, Dict, and IntEnum.
            warnings.warn(
                f'{fn}: category_array checks not implemented for {repr(rt.rt_enum.CategoryMode(expected.category_mode))} '
            )

        # check underlying FastArray
        if (
            expected.category_mode == rt.rt_enum.CategoryMode.StringArray
            or expected.category_mode == rt.rt_enum.CategoryMode.NumericArray
        ):
            assert_array_equal(
                actual._fa, expected._fa, err_msg=f'{fn}: mismatch in underlying FastArray'
            )

        # check expand_array
        if check_expanded_array:
            assert_array_equal(
                actual.expand_array, expected.expand_array, err_msg=f'{fn}: mismatch in "expand_array"'
            )

        # check category_dict
        for k, fa in expected.category_dict.items():
            assert_array_equal(
                actual.category_dict.get(k, None), fa, err_msg=f'{fn}: mismatch "category_dict" for item "{k}"\nactual keys {actual.category_dict.keys()}\nexpected keys {expected.category_dict.keys()}'
            )

        # TODO check CategoryMode Dictionary category_mapping and category_codes

    except Exception:
        if verbose:
            print(
                f'{fn}: expected\n{verbose_categorical(expected)}\n'
                f'{fn}: actual\n{verbose_categorical(actual)}\n'
            )
        raise


def get_common_and_diff_members(
    obj_a: object, obj_b: object
) -> Tuple[Set[str], Set[str]]:
    """
    Return the commonalities and differences between the two objects public API.

    Parameters
    ----------
    obj_a: object
        Object to find commonalities and difference when compared with obj_b.
    obj_b: object
        Object to find commonalities and difference when compared with obj_a.

    Returns
    -------
    tuple
        A tuple of two sets of strings where the first is the commonalities and second is the differences.
    """
    is_public = lambda name: not (name.startswith('__') or name.startswith('_'))
    obj_a_dir, obj_b_dir = (
        set(filter(is_public, dir(obj_a))),
        set(filter(is_public, dir(obj_b))),
    )
    common_members = obj_a_dir.intersection(obj_b_dir)
    diff_members = obj_a_dir.symmetric_difference(obj_b_dir)
    return (common_members, diff_members)


# 20200304 / Riptable version 1.3.367 - Below are the common and differing members of ndarray vs FastArray:
# common members: {'flags', 'fill', 'partition', 'transpose', 'repeat', 'real', 'itemsize', 'ctypes', 'getfield', 'itemset', 'newbyteorder', 'tobytes', 'take', 'dumps', 'conjugate', 'flatten', 'nonzero', 'setflags', 'searchsorted', 'mean', 'dump', 'dtype', 'copy', 'data', 'imag', 'max', 'byteswap', 'astype', 'std', 'argsort', 'any', 'item', 'resize', 'shape', 'dot', 'compress', 'tostring', 'min', 'tofile', 'base', 'flat', 'tolist', 'ndim', 'cumprod', 'T', 'reshape', 'trace', 'ravel', 'squeeze', 'argmin', 'sum', 'sort', 'prod', 'diagonal', 'nbytes', 'argmax', 'strides', 'conj', 'ptp', 'round', 'all', 'put', 'size', 'clip', 'var', 'swapaxes', 'argpartition', 'cumsum', 'choose', 'view', 'setfield'}
# differing members: {'count', 'where', 'nanvar', 'WarningDict', 'clip_lower', 'str', 'isna', 'isnotnan', 'clip_upper', 'isin', 'Verbose', 'diff', 'push', 'median', 'differs', 'rolling_nanstd', 'move_mean', 'fill_backward', 'nanrankdata', 'describe', 'fill_invalid', 'MAX_DISPLAY_LEN', 'numbastring', 'SafeConversions', 'register_function', 'timewindow_sum', 'rolling_var', 'fillna', 'isfinite', 'duplicated', 'info', 'nanmean', 'ema_decay', 'nanargmax', 'save', 'rolling_nanvar', 'FasterUFunc', 'iscomputable', 'move_rank', 'tile', 'copy_invalid', 'sample', 'apply', 'move_argmax', 'get_name', 'move_median', 'cummax', 'Recycle', 'trunc', 'issorted', 'move_max', 'rolling_std', 'nunique', 'move_std', 'fill_forward', 'isnotnormal', 'replacena', 'crc', 'apply_pandas', 'notna', 'timewindow_prod', 'nanmax', 'CompressPickle', 'NoTolerance', 'abs', 'isinf', 'move_min', 'rolling_nansum', 'isnotinf', 'set_name', 'normalize_minmax', 'nansum', 'rolling_mean', 'isnan', 'isnotfinite', 'move_sum', 'argpartition2', 'inv', 'replace', 'rankdata', 'transitions', 'doc', 'normalize_zscore', 'rolling_nanmean', 'partition2', 'str_append', 'nanstd', 'display_query_properties', 'shift', 'move_var', 'unique', 'nanargmin', 'apply_schema', 'map', 'cummin', 'move_argmin', 'isnormal', 'map_old', 'nanmin', 'rolling_sum', 'WarningLevel', 'sign', 'isnanorzero', 'apply_numba'}
_NDARRAY_FASTARRAY_COMMON_AND_DIFF = get_common_and_diff_members(
    np.ndarray, rt.FastArray
)


def get_size(obj: object) -> int:
    """Returns the total size, in bytes, of the object and the object's referents using a
    breadth-first search sweep."""
    # TODO use the robust and well tests Pympler tool to measure the size of the objects
    # Pympler tool has a asizeof that calculates the combined size in bytes of an object.
    # Some shortcomings of this approach is that it gets all referents including of the object
    # class definition, decorators, and other referents that are not relevant to the objects
    # dynamic memory footprint.
    if not isinstance(obj, Iterable):
        obj = [obj]
    size: int = 0
    seen: Set[int] = set()
    queue: deque = deque(obj)
    while len(queue):
        u = queue.popleft()
        refs = gc.get_referents(u)
        for ref in refs:
            ref_id = id(ref)
            if ref_id not in seen:  # guard against cycles
                seen.add(ref_id)
                queue.append(ref)
        size += sys.getsizeof(u)
    return size
