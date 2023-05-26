"""
Time/ordering-based merge implementations.
"""
__all__ = ["merge_asof2"]

from datetime import timedelta
import logging
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union
import warnings

import numpy as np
import numpy.typing as npt
import numba as nb

from .config import get_global_settings
from .numba.indexing import scalar_or_lookup
from .numba.invalid_values import get_invalid, get_max_valid, get_min_valid, is_valid
from .rt_fastarray import FastArray
from .rt_grouping import Grouping
from .rt_datetime import Date, DateTimeNano, DateTimeNanoScalar, TimeSpanScalar
from .rt_numpy import empty, full, full_like, ismember, isnan, isnotnan
from .rt_merge import (
    JoinIndices,
    _construct_colname_mapping,
    _extract_on_columns,
    _gbkeys_extract,
    _get_or_create_keygroup,
    _normalize_selected_columns,
    _require_columns_present,
    _verify_join_keys_compat,
)

if TYPE_CHECKING:
    from .rt_dataset import Dataset


_logger = logging.getLogger(__name__)
"""Logger for this module."""


def _less_than_comparison(allow_exact_matches, x, y):
    """
    Numba-based wrapper for ``__lt__`` and ``__le__`` which allows them to be
    selected at compile-time using type-based dispatch.

    Parameters
    ----------
    allow_exact_matches : bool, optional
    x, y
        Values to compare.

    Returns
    -------
    bool
        ``x < y`` if `allow_exact_matches` is ``None``; otherwise ``x <= y``.
    """
    raise RuntimeError("Unexpected call")


@nb.extending.overload(_less_than_comparison, nopython=True)
def __less_than_comparison(allow_exact_matches, x, y):
    if isinstance(allow_exact_matches, nb.types.NoneType):
        return lambda allow_exact_matches, x, y: x < y
    else:
        return lambda allow_exact_matches, x, y: x <= y


def _greater_than_comparison(allow_exact_matches, x, y):
    """
    Numba-based wrapper for ``__gt__`` and ``__ge__`` which allows them to be
    selected at compile-time using type-based dispatch.

    Parameters
    ----------
    allow_exact_matches : bool, optional
    x, y
        Values to compare.

    Returns
    -------
    bool
        ``x > y`` if `allow_exact_matches` is ``None``; otherwise ``x >= y``.
    """
    raise RuntimeError("Unexpected call")


@nb.extending.overload(_greater_than_comparison, nopython=True)
def __greater_than_comparison(allow_exact_matches, x, y):
    if isinstance(allow_exact_matches, nb.types.NoneType):
        return lambda allow_exact_matches, x, y: x > y
    else:
        return lambda allow_exact_matches, x, y: x >= y


@nb.njit(cache=get_global_settings().enable_numba_cache)
def _merge_asof_backward(
    haystack_2nd_key: np.ndarray,
    needles_2nd_key: np.ndarray,
    needle_idx_in_haystack: npt.NDArray[np.integer],
    haystack_first_unsorted_idx: npt.NDArray[np.integer],
    needle_first_unsorted_idx: npt.NDArray[np.integer],
    equality_tolerance: Optional[Any],
    allow_exact_matches: Optional[Any],
) -> None:
    """
    Numba-based implementation of the 'backward'-mode 'merge_asof' without groups
    (when no 'by' column(s) are specified).

    Parameters
    ----------
    haystack_2nd_key : np.ndarray
        The 'on' column data from the 'right' `Dataset`.
    needles_2nd_key : np.ndarray
        The 'on' column data from the 'left' `Dataset`.
    needle_idx_in_haystack : np.ndarray of integer
        Output array, which, after this function returns, will contain an integer fancy index
        where each element indicates an index within `haystack` where
        The shape must be the same as `needles_igroup` and the dtype should match `haystack_igroup`.
    haystack_first_unsorted_idx, needle_first_unsorted_idx : np.ndarray of bool
        Single-element integer arrays (output) to hold indices within `haystack_2nd_key` and `needles_2nd_key`,
        respectively, where the first unsorted/out-of-order element was detected within that array (if any).
        Pre-initialize these arrays with invalid values so it's easy to detect when elements have been set.
        The arrays are checked after this function returns and are used to provide better error messages
        to users if unsorted values are detected.
    equality_tolerance : scalar, optional
    allow_exact_matches : bool, optional
        Indicates whether exact matches are allowed for the 'on' values.
        Specify ``None`` for ``False`` and ``True`` for ``True``;
        this leverages numba's type-based dispatch mechanism to specialize
        the compiled code with the appropriate comparison function.
    """
    # Get the 'invalid' value for the output array, to use when
    # we can't find a match.
    invalid_haystack_idx = get_invalid(needle_idx_in_haystack[0])

    # The lengths of the 'haystack' and 'needles' arrays.
    haystack_length = len(haystack_2nd_key)
    needles_length = len(needles_2nd_key)

    # Initialize needle_key2_curr, haystack_key2_curr
    # to minimum VALID value for their dtype (making sure not to set to riptable invalid).
    # If we try to be clever about declaring and initializing them in the loop below, numba isn't
    # able to infer the types and can't compile the function.
    # Initializing the values here -- to the same value for 'prev' and 'curr' -- also allows
    # some checks (e.g. 'if i_needle > 0') to be elided in the loop body.
    haystack_key2_curr = get_min_valid(haystack_2nd_key)
    needle_key2_curr = get_min_valid(needles_2nd_key)

    # Walk through the needle and haystack 2nd key ("on") values in a linear,
    # mergesort-style loop. This relies on both arrays (from their respective groups)
    # being sorted; if only the 'haystack' 2nd key values were sorted (within the group)
    # we'd have to implement this as a 'searchsorted'-style binary search
    # for each needle. (Which is faster for one element, but the linear style is much
    # faster for matching a whole array of elements.)
    i_needle = 0
    i_haystack = 0
    while i_needle < needles_length and i_haystack < haystack_length:
        # For convenience / conciseness in the rest of the loop.
        needle_curr_idx = i_needle
        haystack_curr_idx = i_haystack

        # Get the current 'needle' and 'haystack' elements from their 2nd keys (the 'on' arrays).
        # Check them against the previous 'needle' and 'haystack' values, respectively;
        # this in-line sortedness check on the data is more efficient (faster) than doing the
        # check separately, e.g. as a separate method.
        needle_key2_prev = needle_key2_curr
        needle_key2_curr = needles_2nd_key[needle_curr_idx]
        haystack_key2_prev = haystack_key2_curr
        haystack_key2_curr = haystack_2nd_key[haystack_curr_idx]

        # TODO: Should we also (maybe optionally?) recognize riptable integer invalid values
        #       here and consider them like NaN's w.r.t. sortedness?
        # TODO: Should we provide an option to users to ignore NaN (or any other unsorted value)?
        if not (needle_key2_prev <= needle_key2_curr):
            # Detected unsorted (and/or NaN) 'on' value in needle.
            # Save the current index of the needle (within the full array, not the group)
            # into the "unsorted index" array at this group's index,
            # then skip processing for the rest of this group because the results
            # wouldn't be meaningful anyway.
            needle_first_unsorted_idx[0] = needle_curr_idx
            i_needle = needles_length
            continue

        if not (haystack_key2_prev <= haystack_key2_curr):
            # Detected unsorted (and/or NaN) 'on' value in haystack.
            haystack_first_unsorted_idx[0] = haystack_curr_idx
            i_haystack = haystack_length
            continue

        # Compare the current values of the 2nd key for needle and haystack.
        if _less_than_comparison(allow_exact_matches, haystack_key2_curr, needle_key2_curr):
            # If we haven't reached the end of the haystack, check to see if the
            # next haystack value is *also* greater than (or equal to) the current needle.
            # If so, the *current* haystack value is the one we want the needle to match against,
            # because it's the largest haystack value that's still less than or equal to the needle.
            # Otherwise, if the next haystack is still less than the current needle,
            # advance the haystack pointer and recurse.
            if i_haystack + 1 < haystack_length and _less_than_comparison(
                allow_exact_matches, haystack_2nd_key[i_haystack + 1], needle_key2_curr
            ):
                i_haystack += 1

            else:
                # If the current value of `needles` 2nd key is greater than or equal to the current value of `haystack` 2nd key,
                # and their difference is *at most* the tolerance value (if specified), we found a match,
                # so assign the current index within the haystack array to the current index of the needles array.
                # Otherwise, the haystack value was outside the tolerance value so this isn't a match.
                needle_idx_in_haystack[needle_curr_idx] = (
                    haystack_curr_idx
                    # TODO: Do we need to use _less_than_comparison for the equality_tolerance check here?
                    #       Or does it always allow exact matches (i.e. the difference between the values is exactly
                    #       equal to the tolerance)?
                    if equality_tolerance is None or equality_tolerance >= (needle_key2_curr - haystack_key2_curr)
                    else invalid_haystack_idx
                )

                i_needle += 1

        else:
            # N.B. This assumes haystack_key2_curr > needle_key2_curr, but it might not
            # if the 2nd keys are floats and the current value is NaN.

            # This method implements the 'backward'-looking merge_asof, so if we're at
            # the start of the haystack, we know there's no match for this needle.
            if i_haystack < 1:
                needle_idx_in_haystack[needle_curr_idx] = invalid_haystack_idx
                i_needle += 1
            else:
                i_haystack += 1

    # If we reached the end of the haystack and have some remaining needles,
    # they should all be greater than (or equal to) the last/greatest haystack value.
    # Set their corresponding values in the output array to point to the last haystack value.
    if i_needle < needles_length:
        needle_idx_in_haystack[i_needle:] = haystack_length - 1 if haystack_length > 0 else invalid_haystack_idx


@nb.njit(parallel=True, cache=get_global_settings().enable_numba_cache)
def _merge_asof_grouped_backward(
    haystack_igroup: npt.NDArray[np.integer],
    haystack_ifirstgroup: npt.NDArray[np.integer],
    haystack_ncountgroup: npt.NDArray[np.integer],
    haystack_2nd_key: np.ndarray,
    needles_igroup: npt.NDArray[np.integer],
    needles_ifirstgroup: npt.NDArray[np.integer],
    needles_ncountgroup: npt.NDArray[np.integer],
    needles_2nd_key: np.ndarray,
    needle_key_to_haystack_key: npt.NDArray[np.integer],
    needle_idx_in_haystack: npt.NDArray[np.integer],
    haystack_first_unsorted_idx_in_group: npt.NDArray[np.integer],
    needle_first_unsorted_idx_in_group: npt.NDArray[np.integer],
    equality_tolerance: Optional[Any],
    allow_exact_matches: Optional[Any],
) -> None:
    """
    Numba-based implementation of the 'backward'-mode 'merge_asof' over groups
    (when 'by' columns are specified).

    Parameters
    ----------
    haystack_igroup, haystack_ifirstgroup, haystack_ncountgroup : np.ndarray of integer
        Arrays from the `Grouping` created for the 'by' column(s) of the 'right' `Dataset`.
    haystack_2nd_key : np.ndarray
        The 'on' column data from the 'right' `Dataset`.
    needles_igroup, needles_ifirstgroup, needles_ncountgroup : np.ndarray of integer
        Arrays from the `Grouping` created for the 'by' column(s) of the 'left' `Dataset`.
    needles_2nd_key : np.ndarray
        The 'on' column data from the 'left' `Dataset`.
    needle_key_to_haystack_key : np.ndarray of integer
        Array mapping indices of 'needle' "by" keys to the index of the same key
        from 'haystack', if the key also exists there.
    needle_idx_in_haystack : np.ndarray of integer
        Output array, which, after this function returns, will contain an integer fancy index
        where each element indicates an index within `haystack` where
        The shape must be the same as `needles_igroup` and the dtype should match `haystack_igroup`.
    haystack_first_unsorted_idx_in_group, needle_first_unsorted_idx_in_group : np.ndarray of integer
        Integer arrays (output) same length as `haystack_ncountgroup` and `needles_ncountgroup`.
        to hold indices within `haystack_2nd_key` and `needles_2nd_key`, respectively, where the
        first unsorted/out-of-order element was detected within that group (if any).
        Pre-initialize these arrays with invalid values so it's easy to detect when elements have been set.
        The arrays are checked after this function returns and are used to provide better error messages
        to users if unsorted values are detected.
    equality_tolerance : scalar, optional
    allow_exact_matches : bool, optional
        Indicates whether exact matches are allowed for the 'on' values.
        Specify ``None`` for ``False`` and ``True`` for ``True``;
        this leverages numba's type-based dispatch mechanism to specialize
        the compiled code with the appropriate comparison function.
    """
    # Get the 'invalid' value for the output array, to use when
    # we can't find a match.
    invalid_haystack_idx = get_invalid(needle_idx_in_haystack[0])

    for needles_key_idx in nb.prange(len(needle_key_to_haystack_key)):  # pylint: disable=not-an-iterable
        # Add one (1) to convert from the key index to the group index,
        # because group 0 represents the null/NA group.
        needle_group_offset = needles_ifirstgroup[needles_key_idx + 1]
        needle_group_length = needles_ncountgroup[needles_key_idx + 1]

        # Does this 'needles' KEY (1st key only) match a 'haystack' KEY (1st key only)?
        # If so, what's the corresponding 'haystack' group index?
        haystack_key_idx = needle_key_to_haystack_key[needles_key_idx]
        if haystack_key_idx < 0:
            # This needle doesn't exist in 'haystack'.
            # Set the values of all elements in 'needle_idx_in_haystack' that match this group to the invalid value for the array.
            needle_idx_in_haystack[
                needles_igroup[needle_group_offset : needle_group_offset + needle_group_length]
            ] = invalid_haystack_idx

            # Continue processing the next group.
            continue

        # Found a match on the first key between the 'needles' and 'haystack'.
        haystack_group_offset = haystack_ifirstgroup[haystack_key_idx + 1]
        haystack_group_length = haystack_ncountgroup[haystack_key_idx + 1]

        # Initialize needle_key2_curr, haystack_key2_curr
        # to minimum VALID value for their dtype (making sure not to set to riptable invalid).
        # If we try to be clever about declaring and initializing them in the loop below, numba isn't
        # able to infer the types and can't compile the function.
        # Initializing the values here -- to the same value for 'prev' and 'curr' -- also allows
        # some checks (e.g. 'if i_needle > 0') to be elided in the loop body.
        haystack_key2_curr = get_min_valid(haystack_2nd_key)
        needle_key2_curr = get_min_valid(needles_2nd_key)

        # Walk through the needle and haystack 2nd key ("on") values in a linear,
        # mergesort-style loop. This relies on both arrays (from their respective groups)
        # being sorted; if only the 'haystack' 2nd key values were sorted (within the group)
        # we'd have to implement this as a 'searchsorted'-style binary search
        # for each needle. (Which is faster for one element, but the linear style is much
        # faster for matching a whole array of elements.)
        i_needle = 0
        i_haystack = 0
        while i_needle < needle_group_length and i_haystack < haystack_group_length:
            # For convenience / conciseness in the rest of the loop.
            needle_curr_idx = needles_igroup[needle_group_offset + i_needle]
            haystack_curr_idx = haystack_igroup[haystack_group_offset + i_haystack]

            # Get the current 'needle' and 'haystack' elements from their 2nd keys (the 'on' arrays).
            # Check them against the previous 'needle' and 'haystack' values, respectively;
            # this in-line sortedness check on the data is more efficient (faster) than doing the
            # check separately, e.g. as a separate method.
            needle_key2_prev = needle_key2_curr
            needle_key2_curr = needles_2nd_key[needle_curr_idx]
            haystack_key2_prev = haystack_key2_curr
            haystack_key2_curr = haystack_2nd_key[haystack_curr_idx]

            # TODO: Should we also (maybe optionally?) recognize riptable integer invalid values
            #       here and consider them like NaN's w.r.t. sortedness?
            # TODO: Should we provide an option to users to ignore NaN (or any other unsorted value)?
            if not (needle_key2_prev <= needle_key2_curr):
                # Detected unsorted (and/or NaN) 'on' value in needle.
                # Save the current index of the needle (within the full array, not the group)
                # into the "unsorted index" array at this group's index,
                # then skip processing for the rest of this group because the results
                # wouldn't be meaningful anyway.
                needle_first_unsorted_idx_in_group[needles_key_idx + 1] = needle_curr_idx
                i_needle = needle_group_length
                continue

            if not (haystack_key2_prev <= haystack_key2_curr):
                # Detected unsorted (and/or NaN) 'on' value in haystack.
                haystack_first_unsorted_idx_in_group[haystack_key_idx + 1] = haystack_curr_idx
                i_haystack = haystack_group_length
                continue

            # Compare the current values of the 2nd key for needle and haystack.
            if _less_than_comparison(allow_exact_matches, haystack_key2_curr, needle_key2_curr):
                # If we haven't reached the end of the haystack, check to see if the
                # next haystack value is *also* greater than (or equal to) the current needle.
                # If so, the *current* haystack value is the one we want the needle to match against,
                # because it's the largest haystack value that's still less than or equal to the needle.
                # Otherwise, if the next haystack is still less than the current needle,
                # advance the haystack pointer and recurse.
                if i_haystack + 1 < haystack_group_length and _less_than_comparison(
                    allow_exact_matches,
                    haystack_2nd_key[haystack_igroup[haystack_group_offset + i_haystack + 1]],
                    needle_key2_curr,
                ):
                    i_haystack += 1

                else:
                    # If the current value of `needles` 2nd key is greater than or equal to the current value of `haystack` 2nd key,
                    # and their difference is *at most* the tolerance value (if specified), we found a match,
                    # so assign the current index within the haystack array to the current index of the needles array.
                    # Otherwise, the haystack value was outside the tolerance value so this isn't a match.
                    needle_idx_in_haystack[needle_curr_idx] = (
                        haystack_curr_idx
                        # TODO: Do we need to use _less_than_comparison for the equality_tolerance check here?
                        #       Or does it always allow exact matches (i.e. the difference between the values is exactly
                        #       equal to the tolerance)?
                        if equality_tolerance is None or equality_tolerance >= (needle_key2_curr - haystack_key2_curr)
                        else invalid_haystack_idx
                    )

                    i_needle += 1

            else:
                # N.B. This assumes haystack_key2_curr > needle_key2_curr, but it might not
                # if the 2nd keys are floats and the current value is NaN.

                # This method implements the 'backward'-looking merge_asof, so if we're at
                # the start of the haystack, we know there's no match for this needle.
                if i_haystack < 1:
                    needle_idx_in_haystack[needle_curr_idx] = invalid_haystack_idx
                    i_needle += 1
                else:
                    i_haystack += 1

        # If we reached the end of the haystack and have some remaining needles,
        # they should all be greater than (or equal to) the last/greatest haystack value.
        # Set their corresponding values in the output array to point to the last haystack value.
        if i_needle < needle_group_length:
            needle_idx_in_haystack[
                needles_igroup[needle_group_offset + i_needle : needle_group_offset + needle_group_length]
            ] = (haystack_group_length - 1 if haystack_group_length > 0 else invalid_haystack_idx)


@nb.njit(cache=get_global_settings().enable_numba_cache)
def _merge_asof_forward(
    haystack_2nd_key: np.ndarray,
    needles_2nd_key: np.ndarray,
    needle_idx_in_haystack: npt.NDArray[np.integer],
    haystack_first_unsorted_idx: npt.NDArray[np.integer],
    needle_first_unsorted_idx: npt.NDArray[np.integer],
    equality_tolerance: Optional[Any],
    allow_exact_matches: Optional[Any],
) -> None:
    """
    Numba-based implementation of the 'forward'-mode 'merge_asof' without groups
    (when no 'by' column(s) are specified).

    Parameters
    ----------
    haystack_2nd_key : np.ndarray
        The 'on' column data from the 'right' `Dataset`.
    needles_2nd_key : np.ndarray
        The 'on' column data from the 'left' `Dataset`.
    needle_idx_in_haystack : np.ndarray of integer
        Output array, which, after this function returns, will contain an integer fancy index
        where each element indicates an index within `haystack` where
        The shape must be the same as `needles_igroup` and the dtype should match `haystack_igroup`.
    haystack_first_unsorted_idx, needle_first_unsorted_idx : np.ndarray of integer
        Single-element integer arrays (output) to hold indices within `haystack_2nd_key` and `needles_2nd_key`,
        respectively, where the first unsorted/out-of-order element was detected within that group (if any).
        Pre-initialize these arrays with invalid values so it's easy to detect when elements have been set.
        The arrays are checked after this function returns and are used to provide better error messages
        to users if unsorted values are detected.
    equality_tolerance : scalar, optional
    allow_exact_matches : bool, optional
        Indicates whether exact matches are allowed for the 'on' values.
        Specify ``None`` for ``False`` and ``True`` for ``True``;
        this leverages numba's type-based dispatch mechanism to specialize
        the compiled code with the appropriate comparison function.
    """
    # Get the 'invalid' value for the output array, to use when
    # we can't find a match.
    invalid_haystack_idx = get_invalid(needle_idx_in_haystack[0])

    # The lengths of the 'haystack' and 'needles' arrays.
    haystack_length = len(haystack_2nd_key)
    needles_length = len(needles_2nd_key)

    # Initialize needle_key2_curr, haystack_key2_curr
    # to maximum VALID value for their dtype (making sure not to set to riptable invalid).
    # If we try to be clever about declaring and initializing them in the loop below, numba isn't
    # able to infer the types and can't compile the function.
    # Initializing the values here -- to the same value for 'prev' and 'curr' -- also allows
    # some checks (e.g. 'if i_needle > 0') to be elided in the loop body.
    haystack_key2_curr = get_max_valid(haystack_2nd_key)
    needle_key2_curr = get_max_valid(needles_2nd_key)

    # Walk through the needle and haystack 2nd key ("on") values in a linear,
    # mergesort-style loop. This relies on both arrays (from their respective groups)
    # being sorted; if only the 'haystack' 2nd key values were sorted (within the group)
    # we'd have to implement this as a 'searchsorted'-style binary search
    # for each needle. (Which is faster for one element, but the linear style is much
    # faster for matching a whole array of elements.)
    i_needle = needles_length - 1
    i_haystack = haystack_length - 1
    while i_needle >= 0 and i_haystack >= 0:
        # For convenience / conciseness in the rest of the loop.
        needle_curr_idx = i_needle
        haystack_curr_idx = i_haystack

        # Get the current 'needle' and 'haystack' elements from their 2nd keys (the 'on' arrays).
        # Check them against the previous 'needle' and 'haystack' values, respectively;
        # this in-line sortedness check on the data is more efficient (faster) than doing the
        # check separately, e.g. as a separate method.
        needle_key2_prev = needle_key2_curr
        needle_key2_curr = needles_2nd_key[needle_curr_idx]
        haystack_key2_prev = haystack_key2_curr
        haystack_key2_curr = haystack_2nd_key[haystack_curr_idx]

        # TODO: Should we also (maybe optionally?) recognize riptable integer invalid values
        #       here and consider them like NaN's w.r.t. sortedness?
        # TODO: Should we provide an option to users to ignore NaN (or any other unsorted value)?
        if not (needle_key2_prev >= needle_key2_curr):
            # Detected unsorted (and/or NaN) 'on' value in needle.
            # Save the current index of the needle (within the full array, not the group)
            # into the "unsorted index" array at this group's index,
            # then skip processing for the rest of this group because the results
            # wouldn't be meaningful anyway.
            needle_first_unsorted_idx[0] = needle_curr_idx
            i_needle = -1
            continue

        if not (haystack_key2_prev >= haystack_key2_curr):
            # Detected unsorted (and/or NaN) 'on' value in haystack.
            haystack_first_unsorted_idx[0] = haystack_curr_idx
            i_haystack = -1
            continue

        # Compare the current values of the 2nd key for needle and haystack.
        if _greater_than_comparison(allow_exact_matches, haystack_key2_curr, needle_key2_curr):
            # If we haven't reached the front of the haystack, check to see if the
            # previous haystack value is *also* less than (or equal to) the current needle.
            # If so, the *current* haystack value is the one we want the needle to match against,
            # because it's the smallest haystack value that's still greater than or equal to the needle.
            # Otherwise, if the previous haystack is still greater than the current needle,
            # decrement the haystack pointer and recurse.
            if i_haystack > 0 and _greater_than_comparison(
                allow_exact_matches, haystack_2nd_key[i_haystack - 1], needle_key2_curr
            ):
                i_haystack -= 1

            else:
                # If the current value of `needles` 2nd key is greater than or equal to the current value of `haystack` 2nd key,
                # and their difference is *at most* the tolerance value (if specified), we found a match,
                # so assign the current index within the haystack array to the current index of the needles array.
                # Otherwise, the haystack value was outside the tolerance value so this isn't a match.
                needle_idx_in_haystack[needle_curr_idx] = (
                    haystack_curr_idx
                    # TODO: Do we need to use _exact_than_comparison for the equality_tolerance check here?
                    #       Or does it always allow exact matches (i.e. the difference between the values is exactly
                    #       equal to the tolerance)?
                    if equality_tolerance is None or equality_tolerance >= (haystack_key2_curr - needle_key2_curr)
                    else invalid_haystack_idx
                )

                i_needle -= 1

        else:
            # N.B. This assumes haystack_key2_curr < needle_key2_curr, but it might not
            # if the 2nd keys are floats and the current value is NaN.

            # This method implements the 'forward'-looking merge_asof, so if we're still at
            # the end/back (greatest index) of the haystack, we know there's no match for this needle.
            if i_haystack + 1 == haystack_length:
                needle_idx_in_haystack[needle_curr_idx] = invalid_haystack_idx
                i_needle -= 1
            else:
                i_haystack -= 1

    # If we reached the end of the haystack and have some remaining needles,
    # they should all be greater than (or equal to) the last/greatest haystack value.
    # Set their corresponding values in the output array to point to the last haystack value.
    if i_needle >= 0:
        needle_idx_in_haystack[:i_needle] = 0 if haystack_length > 0 else invalid_haystack_idx


@nb.njit(parallel=True, cache=get_global_settings().enable_numba_cache)
def _merge_asof_grouped_forward(
    haystack_igroup: npt.NDArray[np.integer],
    haystack_ifirstgroup: npt.NDArray[np.integer],
    haystack_ncountgroup: npt.NDArray[np.integer],
    haystack_2nd_key: np.ndarray,
    needles_igroup: npt.NDArray[np.integer],
    needles_ifirstgroup: npt.NDArray[np.integer],
    needles_ncountgroup: npt.NDArray[np.integer],
    needles_2nd_key: np.ndarray,
    needle_key_to_haystack_key: npt.NDArray[np.integer],
    needle_idx_in_haystack: npt.NDArray[np.integer],
    haystack_first_unsorted_idx_in_group: npt.NDArray[np.integer],
    needle_first_unsorted_idx_in_group: npt.NDArray[np.integer],
    equality_tolerance: Optional[Any],
    allow_exact_matches: Optional[Any],
) -> None:
    """
    Numba-based implementation of the 'forward'-mode 'merge_asof' over groups
    (when 'by' columns are specified).

    Parameters
    ----------
    haystack_igroup, haystack_ifirstgroup, haystack_ncountgroup : np.ndarray of integer
        Arrays from the `Grouping` created for the 'by' column(s) of the 'right' `Dataset`.
    haystack_2nd_key : np.ndarray
        The 'on' column data from the 'right' `Dataset`.
    needles_igroup, needles_ifirstgroup, needles_ncountgroup : np.ndarray of integer
        Arrays from the `Grouping` created for the 'by' column(s) of the 'left' `Dataset`.
    needles_2nd_key : np.ndarray
        The 'on' column data from the 'left' `Dataset`.
    needle_key_to_haystack_key : np.ndarray of integer
        Array mapping indices of 'needle' "by" keys to the index of the same key
        from 'haystack', if the key also exists there.
    needle_idx_in_haystack : np.ndarray of integer
        Output array, which, after this function returns, will contain an integer fancy index
        where each element indicates an index within `haystack` where
        The shape must be the same as `needles_igroup` and the dtype should match `haystack_igroup`.
    haystack_first_unsorted_idx_in_group, needle_first_unsorted_idx_in_group : np.ndarray of integer
        Integer arrays (output) same length as `haystack_ncountgroup` and `needles_ncountgroup`.
        to hold indices within `haystack_2nd_key` and `needles_2nd_key`, respectively, where the
        first unsorted/out-of-order element was detected within that group (if any).
        Pre-initialize these arrays with invalid values so it's easy to detect when elements have been set.
        The arrays are checked after this function returns and are used to provide better error messages
        to users if unsorted values are detected.
    equality_tolerance : scalar, optional
    allow_exact_matches : bool, optional
        Indicates whether exact matches are allowed for the 'on' values.
        Specify ``None`` for ``False`` and ``True`` for ``True``;
        this leverages numba's type-based dispatch mechanism to specialize
        the compiled code with the appropriate comparison function.
    """
    # Get the 'invalid' value for the output array, to use when
    # we can't find a match.
    invalid_haystack_idx = get_invalid(needle_idx_in_haystack[0])

    for needles_key_idx in nb.prange(len(needle_key_to_haystack_key)):  # pylint: disable=not-an-iterable
        # Add one (1) to convert from the key index to the group index,
        # because group 0 represents the null/NA group.
        needle_group_offset = needles_ifirstgroup[needles_key_idx + 1]
        needle_group_length = needles_ncountgroup[needles_key_idx + 1]

        # Does this 'needles' KEY (1st key only) match a 'haystack' KEY (1st key only)?
        # If so, what's the corresponding 'haystack' group index?
        haystack_key_idx = needle_key_to_haystack_key[needles_key_idx]
        if haystack_key_idx < 0:
            # This needle doesn't exist in 'haystack'.
            # Set the values of all elements in 'needle_idx_in_haystack' that match this group to the invalid value for the array.
            needle_idx_in_haystack[
                needles_igroup[needle_group_offset : needle_group_offset + needle_group_length]
            ] = invalid_haystack_idx

            # Continue processing the next group.
            continue

        # Found a match on the first key between the 'needles' and 'haystack'.
        haystack_group_offset = haystack_ifirstgroup[haystack_key_idx + 1]
        haystack_group_length = haystack_ncountgroup[haystack_key_idx + 1]

        # Initialize needle_key2_curr, haystack_key2_curr
        # to maximum VALID value for their dtype (making sure not to set to riptable invalid).
        # If we try to be clever about declaring and initializing them in the loop below, numba isn't
        # able to infer the types and can't compile the function.
        # Initializing the values here -- to the same value for 'prev' and 'curr' -- also allows
        # some checks (e.g. 'if i_needle > 0') to be elided in the loop body.
        haystack_key2_curr = get_max_valid(haystack_2nd_key)
        needle_key2_curr = get_max_valid(needles_2nd_key)

        # Walk through the needle and haystack 2nd key ("on") values in a linear,
        # mergesort-style loop. This relies on both arrays (from their respective groups)
        # being sorted; if only the 'haystack' 2nd key values were sorted (within the group)
        # we'd have to implement this as a 'searchsorted'-style binary search
        # for each needle. (Which is faster for one element, but the linear style is much
        # faster for matching a whole array of elements.)
        i_needle = needle_group_length - 1
        i_haystack = haystack_group_length - 1
        while i_needle >= 0 and i_haystack >= 0:
            # For convenience / conciseness in the rest of the loop.
            needle_curr_idx = needles_igroup[needle_group_offset + i_needle]
            haystack_curr_idx = haystack_igroup[haystack_group_offset + i_haystack]

            # Get the current 'needle' and 'haystack' elements from their 2nd keys (the 'on' arrays).
            # Check them against the previous 'needle' and 'haystack' values, respectively;
            # this in-line sortedness check on the data is more efficient (faster) than doing the
            # check separately, e.g. as a separate method.
            needle_key2_prev = needle_key2_curr
            needle_key2_curr = needles_2nd_key[needle_curr_idx]
            haystack_key2_prev = haystack_key2_curr
            haystack_key2_curr = haystack_2nd_key[haystack_curr_idx]

            # TODO: Should we also (maybe optionally?) recognize riptable integer invalid values
            #       here and consider them like NaN's w.r.t. sortedness?
            # TODO: Should we provide an option to users to ignore NaN (or any other unsorted value)?
            if not (needle_key2_prev >= needle_key2_curr):
                # Detected unsorted (and/or NaN) 'on' value in needle.
                # Save the current index of the needle (within the full array, not the group)
                # into the "unsorted index" array at this group's index,
                # then skip processing for the rest of this group because the results
                # wouldn't be meaningful anyway.
                needle_first_unsorted_idx_in_group[needles_key_idx + 1] = needle_curr_idx
                i_needle = -1
                continue

            if not (haystack_key2_prev >= haystack_key2_curr):
                # Detected unsorted (and/or NaN) 'on' value in haystack.
                haystack_first_unsorted_idx_in_group[haystack_key_idx + 1] = haystack_curr_idx
                i_haystack = -1
                continue

            # Compare the current values of the 2nd key for needle and haystack.
            if _greater_than_comparison(allow_exact_matches, haystack_key2_curr, needle_key2_curr):
                # If we haven't reached the end of the haystack, check to see if the
                # next haystack value is *also* less than (or equal to) the current needle.
                # If so, the *current* haystack value is the one we want the needle to match against,
                # because it's the smallest haystack value that's still greater than or equal to the needle.
                # Otherwise, if the next haystack is still greater than the current needle,
                # advance the haystack pointer and recurse.
                if i_haystack > 0 and _greater_than_comparison(
                    allow_exact_matches,
                    haystack_2nd_key[haystack_igroup[haystack_group_offset + i_haystack - 1]],
                    needle_key2_curr,
                ):
                    i_haystack -= 1

                else:
                    # If the current value of `needles` 2nd key is greater than or equal to the current value of `haystack` 2nd key,
                    # and their difference is *at most* the tolerance value (if specified), we found a match,
                    # so assign the current index within the haystack array to the current index of the needles array.
                    # Otherwise, the haystack value was outside the tolerance value so this isn't a match.
                    needle_idx_in_haystack[needle_curr_idx] = (
                        haystack_curr_idx
                        # TODO: Do we need to use _less_than_comparison for the equality_tolerance check here?
                        #       Or does it always allow exact matches (i.e. the difference between the values is exactly
                        #       equal to the tolerance)?
                        if equality_tolerance is None or equality_tolerance >= (haystack_key2_curr - needle_key2_curr)
                        else invalid_haystack_idx
                    )

                    i_needle -= 1

            else:
                # N.B. This assumes haystack_key2_curr < needle_key2_curr, but it might not
                # if the 2nd keys are floats and the current value is NaN.

                # This method implements the 'forward'-looking merge_asof, so if we're still at
                # the end/back (greatest index) of the haystack, we know there's no match for this needle.
                if i_haystack + 1 == haystack_group_length:
                    needle_idx_in_haystack[needle_curr_idx] = invalid_haystack_idx
                    i_needle -= 1
                else:
                    i_haystack -= 1

        # If we reached the end of the haystack and have some remaining needles,
        # they should all be greater than (or equal to) the last/greatest haystack value.
        # Set their corresponding values in the output array to point to the last haystack value.
        if i_needle >= 0:
            needle_idx_in_haystack[needles_igroup[needle_group_offset : needle_group_offset + i_needle]] = (
                0 if haystack_group_length > 0 else invalid_haystack_idx
            )


@nb.njit(parallel=True, cache=get_global_settings().enable_numba_cache)
def _merge_asof_combine_into_nearest(
    haystack_2nd_key: np.ndarray,
    needles_2nd_key: np.ndarray,
    needle_idx_in_haystack_backward: npt.NDArray[np.integer],
    needle_idx_in_haystack_forward: npt.NDArray[np.integer],
) -> None:
    """
    Produce a 'nearest'-mode fancy index by combining fancy indices from 'backward' and 'forward'-mode merge_asof.

    This function can be used for both the "non-grouped" (no 'by' column(s) specified) and "grouped"
    merge_asof, because the 'backward' and 'forward'-mode fancy indices already incorporate the
    grouping information.

    Parameters
    ----------
    haystack_2nd_key : np.ndarray
    needles_2nd_key : np.ndarray
    needle_idx_in_haystack_backward : np.ndarray of integer
    needle_idx_in_haystack_forward : np.ndarray of integer
        Fancy index into `haystack_2nd_key` created by one of the forward-looking
        'merge_asof' implementations. This array is updated in-place and serves
        as both an input and the output of this function.

    Notes
    -----
    Requires ``len(needles_2nd_key) == len(needle_idx_in_haystack_backward) == len(needle_idx_in_haystack_forward)``.
    """
    for i in nb.prange(len(needles_2nd_key)):  # pylint: disable=not-an-iterable
        # The indices within 'haystack_2nd_key' for the backward- and forward-looking
        # matches of the current needle value.
        haystack_idx_backward = needle_idx_in_haystack_backward[i]
        haystack_idx_forward = needle_idx_in_haystack_forward[i]

        # For the current index / needle value, was there a backwards-looking match in 'haystack'?
        if is_valid(haystack_idx_backward):
            # If the index for the forward-looking match is invalid, use the backward-looking match index.
            if not is_valid(haystack_idx_forward):
                needle_idx_in_haystack_forward[i] = haystack_idx_backward
            else:
                # Matched both forwards and backwards, so determine which one is closer.
                needle_key2_curr = needles_2nd_key[i]

                # N.B. While computing the differences here, we use the contract between the
                # needle and haystack values in backwards (haystack <= needle) and
                # forwards (haystack >= needle) so both diffs will be >= 0.
                diff_backward = needle_key2_curr - haystack_2nd_key[haystack_idx_backward]
                diff_forward = haystack_2nd_key[haystack_idx_forward] - needle_key2_curr

                # If the 'haystack' value from the backwards-looking match is closer
                # to the 'needle' value than the 'haystack' value from the forwards-looking
                # match, update the fancy-index array to use the backwards-looking index
                # for this row.
                # NOTE: For backwards-compatibility, 'backwards' wins any ties,
                # so this comparison uses __le__ instead of __lt__.
                if diff_backward <= diff_forward:
                    needle_idx_in_haystack_forward[i] = haystack_idx_backward


def _fancyindex_dtype_from_shape(shape: Union[int, Tuple[int, ...]]) -> np.dtype:
    # Get max length of any dimension.
    max_dim_len = max(shape) if isinstance(shape, (tuple, list, np.ndarray)) else shape

    # Use -(... + 1) approach with np.min_scalar_type()
    return np.min_scalar_type(-(max_dim_len + 1))


def _create_merge_asof_fancy_indices(
    left_on: np.ndarray,
    right_on: np.ndarray,
    tolerance: Optional[Union[int, float, "timedelta", TimeSpanScalar]] = None,
    allow_exact_matches: bool = True,
    direction: Literal["backward", "forward", "nearest"] = "backward",
) -> JoinIndices:
    """
    Create two fancy indices -- for the left and right Datasets, respectively -- to be used to index the columns
    of those Datasets to produce the joined result Dataset.

    This function is for the non-grouped version of `merge_asof`; that is, where no 'by' column(s)
    have been specified.

    Parameters
    ----------
    left_on
    right_on
    tolerance : integer or float or Timedelta, optional, default None
        Tolerance allowed when performing the 'asof' part of the merge; whenever a row from
        `left` doesn't have a key in `right` within this distance or less, that row will have
        a null/missing/NA value for any columns from the `right` Dataset which appear in the
        merged result.
    allow_exact_matches : boolean, default True
        - If True, allow matching with the same 'on' value
          (i.e. less-than-or-equal-to / greater-than-or-equal-to)
        - If False, don't match the same 'on' value
          (i.e., strictly less-than / strictly greater-than)
    direction : {'backward', 'forward', or 'nearest'}, default 'backward'
        Whether to search for prior, subsequent, or closest matches.

    Returns
    -------
    JoinIndices
        A `JoinIndices` instance containing the results from the join algorithm.

    Notes
    -----
    No 'by' here -- this is a pure mergesort-type join along the 'on' columns.
    Basically, it's a specialized 'searchsorted' where BOTH arrays (needles and haystack)
    are sorted (not just haystack, as with normal searchsorted). This means the implementation
    can operate in O(n+m) time instead of O(n*log(m)).

    TODO: Consider allowing the ``tolerance`` parameter to also be a 1D array/sequence whose length is the same
          as the number of groups (or keys?) in the ``left`` dataset. That allows a per-group tolerance to be specified
          if needed.
    """
    # Allocate the output array.
    # TODO: Should be able to use rt.empty here and have it completely overwritten by the numba-based
    #       implementation, but for some reason it's not. So for now workaround by initializing the
    #       array with all elements set to the default value.
    right_fancyindex_dtype = _fancyindex_dtype_from_shape(right_on.shape)
    right_fancyindex = empty(left_on.shape, dtype=right_fancyindex_dtype)
    right_fancyindex[:] = right_fancyindex.inv

    # Allocate arrays the numba function can use to notify us that the
    # 'on' data isn't completely sorted within the arrays.
    haystack_first_unsorted_idx = empty(1, dtype=np.int64)
    haystack_first_unsorted_idx[0] = haystack_first_unsorted_idx.inv
    needle_first_unsorted_idx = empty(1, dtype=np.int64)
    needle_first_unsorted_idx[0] = needle_first_unsorted_idx.inv

    # Dispatch to numba-based implementation based on 'direction'.
    if direction == "backward":
        _merge_asof_backward(
            right_on,
            left_on,
            right_fancyindex,
            haystack_first_unsorted_idx,
            needle_first_unsorted_idx,
            tolerance,
            # Specify None if not allowing exact matches,
            # so numba can specialize the function with the correct operator.
            # (numba will specialize on types, but not on values.)
            True if allow_exact_matches else None,
        )

    elif direction == "forward":
        _merge_asof_forward(
            right_on,
            left_on,
            right_fancyindex,
            haystack_first_unsorted_idx,
            needle_first_unsorted_idx,
            tolerance,
            # Specify None if not allowing exact matches,
            # so numba can specialize the function with the correct operator.
            # (numba will specialize on types, but not on values.)
            True if allow_exact_matches else None,
        )

    elif direction == "nearest":
        # TEMP: For now, implement the 'nearest' mode by running both 'backwards' and 'forwards' modes
        #       then combining the results. We can implement a specialized function for it later to
        #       improve the performance.

        # Run both a backwards-looking and forwards-looking 'merge_asof'.
        right_fancyindex_backward = right_fancyindex
        _merge_asof_backward(
            right_on,
            left_on,
            right_fancyindex_backward,
            haystack_first_unsorted_idx,
            needle_first_unsorted_idx,
            tolerance,
            # Specify None if not allowing exact matches,
            # so numba can specialize the function with the correct operator.
            # (numba will specialize on types, but not on values.)
            True if allow_exact_matches else None,
        )

        right_fancyindex_forward = empty(right_fancyindex_backward.shape, dtype=right_fancyindex_backward.dtype)
        right_fancyindex_forward[:] = right_fancyindex_forward.inv

        _merge_asof_forward(
            right_on,
            left_on,
            right_fancyindex_forward,
            haystack_first_unsorted_idx,
            needle_first_unsorted_idx,
            tolerance,
            # Specify None if not allowing exact matches,
            # so numba can specialize the function with the correct operator.
            # (numba will specialize on types, but not on values.)
            True if allow_exact_matches else None,
        )

        # Combine the backwards-looking and forwards-looking results into a single fancy-index
        # based on which 'haystack' value is nearer for each row in 'needles'.
        # Note 'backward' wins ties.
        _merge_asof_combine_into_nearest(right_on, left_on, right_fancyindex_backward, right_fancyindex_forward)
        right_fancyindex = right_fancyindex_forward

    else:
        raise ValueError(f"Invalid direction value '{direction}' specified.")

    # Check whether any groups weren't sorted; if we find one or more (in either side),
    # raise an exception with the keys of some unsorted groups.
    if np.any(isnotnan(haystack_first_unsorted_idx)):
        raise ValueError("The 'on' column for the 'right' Dataset contains one or more unsorted elements.")
    elif np.any(isnotnan(needle_first_unsorted_idx)):
        raise ValueError("The 'on' column for the 'left' Dataset contains one or more unsorted elements.")

    # Create and return a JoinIndices with the right-fancyindex.
    return JoinIndices(None, right_fancyindex)


def _create_merge_asof_grouped_fancy_indices(
    left_by_keygroup: "Grouping",
    right_by_keygroup: "Grouping",
    left_on: np.ndarray,
    right_on: np.ndarray,
    tolerance: Optional[Union[int, float, "timedelta", TimeSpanScalar]] = None,
    allow_exact_matches: bool = True,
    direction: Literal["backward", "forward", "nearest"] = "backward",
) -> JoinIndices:
    """
    Create two fancy indices -- for the left and right Datasets, respectively -- to be used to index the columns
    of those Datasets to produce the joined result Dataset.

    Parameters
    ----------
    left_by_keygroup : Grouping
        A `Grouping` instance representing the 'by' key(s) from the 'left' Dataset.
    right_by_keygroup : Grouping
        A `Grouping` instance representing the 'by' key(s) from the 'right' Dataset.
    left_on
    right_on
    tolerance : integer or float or Timedelta, optional, default None
        Tolerance allowed when performing the 'asof' part of the merge; whenever a row from
        `left` doesn't have a key in `right` within this distance or less, that row will have
        a null/missing/NA value for any columns from the `right` Dataset which appear in the
        merged result.
    allow_exact_matches : boolean, default True
        - If True, allow matching with the same 'on' value
          (i.e. less-than-or-equal-to / greater-than-or-equal-to)
        - If False, don't match the same 'on' value
          (i.e., strictly less-than / strictly greater-than)
    direction : {'backward', 'forward', or 'nearest'}, default 'backward'
        Whether to search for prior, subsequent, or closest matches.

    Returns
    -------
    JoinIndices
        A `JoinIndices` instance containing the results from the join algorithm.

    Notes
    -----
    TODO: Consider allowing the ``tolerance`` parameter to also be a 1D array/sequence whose length is the same
          as the number of groups (or keys?) in the ``left`` dataset. That allows a per-group tolerance to be specified
          if needed.
    """
    # Get the unique values from the Groupings.
    # TODO: If any of the keys in the tuples below are Categorical, need to expand them back;
    #       current rt.ismember() won't handle it correctly, at least if they are one of
    #       multiple keys in the Grouping.
    left_grouping_gbkey = _gbkeys_extract(left_by_keygroup)
    right_grouping_gbkey = _gbkeys_extract(right_by_keygroup)

    # Determine which keys of the 'left_by' Grouping are also present in the 'right_by'
    # Grouping, and create a mapping from 'left_by' key indices to 'right_by' key indices.
    # N.B. we do this operation on the gbkeys of each grouping; the number of uniques in each
    # key will be smaller-than-or-equal to the length of the original data column, and in most
    # cases will be significantly smaller (so using ismember on the gbkeys will be significantly faster).
    # TODO: Pass 'hint_size' here to speed up the 'ismember' call?
    _, left_by_keyid_to_right_by_keyid_map = ismember(left_grouping_gbkey, right_grouping_gbkey)

    # Allocate the output array.
    # TODO: Should be able to use rt.empty here and have it completely overwritten by the numba-based
    #       implementation, but for some reason it's not. So for now workaround by initializing the
    #       array with all elements set to the default value.
    right_fancyindex = empty(left_by_keygroup.ikey.shape, dtype=right_by_keygroup.igroup.dtype)
    right_fancyindex[:] = right_fancyindex.inv

    # Allocate arrays the numba function can use to notify us that the
    # 'on' data isn't completely sorted within some group(s).
    haystack_first_unsorted_idx_in_group = full_like(right_by_keygroup.ifirstgroup, right_by_keygroup.ifirstgroup.inv)
    needle_first_unsorted_idx_in_group = full_like(left_by_keygroup.ifirstgroup, left_by_keygroup.ifirstgroup.inv)

    # Dispatch to numba-based implementation based on 'direction'.
    if direction == "backward":
        _merge_asof_grouped_backward(
            right_by_keygroup.igroup,
            right_by_keygroup.ifirstgroup,
            right_by_keygroup.ncountgroup,
            right_on,
            left_by_keygroup.igroup,
            left_by_keygroup.ifirstgroup,
            left_by_keygroup.ncountgroup,
            left_on,
            left_by_keyid_to_right_by_keyid_map,
            right_fancyindex,
            haystack_first_unsorted_idx_in_group,
            needle_first_unsorted_idx_in_group,
            tolerance,
            # Specify None if not allowing exact matches,
            # so numba can specialize the function with the correct operator.
            # (numba will specialize on types, but not on values.)
            True if allow_exact_matches else None,
        )

    elif direction == "forward":
        _merge_asof_grouped_forward(
            right_by_keygroup.igroup,
            right_by_keygroup.ifirstgroup,
            right_by_keygroup.ncountgroup,
            right_on,
            left_by_keygroup.igroup,
            left_by_keygroup.ifirstgroup,
            left_by_keygroup.ncountgroup,
            left_on,
            left_by_keyid_to_right_by_keyid_map,
            right_fancyindex,
            haystack_first_unsorted_idx_in_group,
            needle_first_unsorted_idx_in_group,
            tolerance,
            # Specify None if not allowing exact matches,
            # so numba can specialize the function with the correct operator.
            # (numba will specialize on types, but not on values.)
            True if allow_exact_matches else None,
        )

    elif direction == "nearest":
        # TEMP: For now, implement the 'nearest' mode by running both 'backwards' and 'forwards' modes
        #       then combining the results. We can implement a specialized function for it later to
        #       improve the performance.

        # Run both a backwards-looking and forwards-looking 'merge_asof'.
        right_fancyindex_backward = right_fancyindex
        _merge_asof_grouped_backward(
            right_by_keygroup.igroup,
            right_by_keygroup.ifirstgroup,
            right_by_keygroup.ncountgroup,
            right_on,
            left_by_keygroup.igroup,
            left_by_keygroup.ifirstgroup,
            left_by_keygroup.ncountgroup,
            left_on,
            left_by_keyid_to_right_by_keyid_map,
            right_fancyindex_backward,
            haystack_first_unsorted_idx_in_group,
            needle_first_unsorted_idx_in_group,
            tolerance,
            # Specify None if not allowing exact matches,
            # so numba can specialize the function with the correct operator.
            # (numba will specialize on types, but not on values.)
            True if allow_exact_matches else None,
        )

        right_fancyindex_forward = empty(right_fancyindex_backward.shape, dtype=right_fancyindex_backward.dtype)
        right_fancyindex_forward[:] = right_fancyindex_forward.inv

        _merge_asof_grouped_forward(
            right_by_keygroup.igroup,
            right_by_keygroup.ifirstgroup,
            right_by_keygroup.ncountgroup,
            right_on,
            left_by_keygroup.igroup,
            left_by_keygroup.ifirstgroup,
            left_by_keygroup.ncountgroup,
            left_on,
            left_by_keyid_to_right_by_keyid_map,
            right_fancyindex_forward,
            # Re-use these for the forward call. If unsortedness is detected, it'll be
            # at the same index in both forwards and backwards mode.
            haystack_first_unsorted_idx_in_group,
            needle_first_unsorted_idx_in_group,
            tolerance,
            # Specify None if not allowing exact matches,
            # so numba can specialize the function with the correct operator.
            # (numba will specialize on types, but not on values.)
            True if allow_exact_matches else None,
        )

        # Combine the backwards-looking and forwards-looking results into a single fancy-index
        # based on which 'haystack' value is nearer for each row in 'needles'.
        # Note 'backward' wins ties.
        _merge_asof_combine_into_nearest(right_on, left_on, right_fancyindex_backward, right_fancyindex_forward)
        right_fancyindex = right_fancyindex_forward

    else:
        raise ValueError(f"Invalid direction value '{direction}' specified.")

    # Check whether any groups weren't sorted; if we find one or more (in either side),
    # raise an exception with the keys of some unsorted groups.
    if np.any(isnotnan(haystack_first_unsorted_idx_in_group)):
        raise ValueError("The 'on' column isn't sorted within one or more unsorted 'by' groups of the 'right' Dataset.")
    elif np.any(isnotnan(needle_first_unsorted_idx_in_group)):
        raise ValueError("The 'on' column isn't sorted within one or more unsorted 'by' groups of the 'left' Dataset.")

    # Create and return a JoinIndices with the right-fancyindex.
    return JoinIndices(None, right_fancyindex)


def _validate_and_normalize_tolerance(
    left_on: np.ndarray, right_on: np.ndarray, tolerance: Union[int, float, "timedelta", TimeSpanScalar]
) -> Any:
    """
    Validate the tolerance value is compatible with the 'on' columns for an "as-of" merge.

    The tolerance value may be converted to another scalar type before returning, if it's
    necessary for compatibility with the 'on' columns in the generated code. For example,
    a `TimeSpanScalar` may be converted to a `DateTimeNanoScalar` to ensure the comparison
    works correctly.

    Parameters
    ----------
    left_on, right_on
    tolerance

    Returns
    -------
    validated_tolerance
    """
    # If 'tolerance' is a value of a signed type, don't allow negative values;
    # the 'as-of' logic won't work correctly if given a negative value.
    negative_tol_errmsg = "Negative tolerance values not supported for as-of merges."
    if isinstance(tolerance, float) or (
        isinstance(tolerance, np.generic) and np.issubdtype(tolerance.dtype, np.floating)
    ):
        if tolerance < 0 or isnan(tolerance):
            raise ValueError(negative_tol_errmsg)

        # Validate the 'left_on' and 'right_on' columns are np.ndarray or FastArray (but no derived FA types).
        if (type(left_on) != np.ndarray and type(left_on) != FastArray) or (
            type(right_on) != np.ndarray and type(right_on) != FastArray
        ):
            raise ValueError(f"`{type(tolerance)}`-typed tolerance values not supported for the given 'on' columns.")

        # Determine which dtype is compatible between the two 'on' columns,
        # then convert the tolerance value to a scalar of that type.
        # This prevents the tolerance value from changing the types that'd be inferred
        # by numba for the generated code; if the tolerance value is large enough to
        # require a larger type than the 'on' columns, it might make sense not to allow it?
        unified_on_type: np.dtype = np.promote_types(left_on.dtype, right_on.dtype)
        return np.array(tolerance, dtype=unified_on_type)[()]

    elif isinstance(tolerance, int) or (
        isinstance(tolerance, np.generic) and np.issubdtype(tolerance.dtype, np.integer)
    ):
        if tolerance < 0 or isnan(tolerance):
            raise ValueError(negative_tol_errmsg)

        # Validate the 'left_on' and 'right_on' columns are np.ndarray or FastArray (but no derived FA types)
        if (type(left_on) != np.ndarray and type(left_on) != FastArray) or (
            type(right_on) != np.ndarray and type(right_on) != FastArray
        ):
            raise ValueError(f"`{type(tolerance)}`-typed tolerance values not supported for the given 'on' columns.")

        # Determine which dtype is compatible between the two 'on' columns,
        # then convert the tolerance value to a scalar of that type.
        # This prevents the tolerance value from changing the types that'd be inferred
        # by numba for the generated code; if the tolerance value is large enough to
        # require a larger type than the 'on' columns, it might make sense not to allow it?
        unified_on_type: np.dtype = np.promote_types(left_on.dtype, right_on.dtype)
        return np.array(tolerance, dtype=unified_on_type)[()]

    elif isinstance(tolerance, TimeSpanScalar):
        if tolerance < TimeSpanScalar(0):
            raise ValueError(negative_tol_errmsg)

        if isinstance(left_on, DateTimeNano):
            # Convert to a DateTimeNanoScalar; this is a bit weird (since we'd usually want
            # the difference of two DateTimeNano values to be a TimeSpanScalar), but due to
            # the internal representation of these types we need a DateTimeNanoScalar for the
            # JIT-compiled code to work correctly.
            return DateTimeNanoScalar(tolerance.nanoseconds)

        else:
            raise ValueError(
                f"Can't determine how to use a `{type(tolerance).__qualname__}`-type tolerance value with an 'on' column of type '{type(left_on).__name__}' and dtype '{left_on.dtype}'."
            )

    elif isinstance(tolerance, timedelta):
        if tolerance < timedelta(0):
            raise ValueError(negative_tol_errmsg)

        if isinstance(left_on, DateTimeNano):
            # The range of datetime.timedelta is larger than can be represented
            # by a 64-bit integer (signed or not) of epoch-nanoseconds.
            # If outside the range, raise an exception since we can't convert.
            if tolerance.days > 106751:  # roughly 106751.9 days in 2**63 nanoseconds
                raise ValueError("Tolerance is too large to be used with `DateTimeNano`-typed 'on' columns.")

            # Convert to nanoseconds.
            # N.B. The conversion used here is only safe for non-negative timedelta values.
            return DateTimeNanoScalar(
                (((tolerance.days * 86_400) + tolerance.seconds) * 1_000_000_000) + (tolerance.microseconds * 1_000)
            )

        else:
            raise ValueError(
                f"Can't determine how to use a `{type(tolerance).__qualname__}`-type tolerance value with an 'on' column of type '{type(left_on).__name__}' and dtype '{left_on.dtype}'."
            )

    else:
        raise ValueError(f"Tolerance value has unsupported type '{type(tolerance).__qualname__}'.")


def merge_asof2(
    left: "Dataset",
    right: "Dataset",
    on: Optional[Union[str, Tuple[str, str]]] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    by: Optional[Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]]] = None,
    left_by: Optional[Union[str, List[str]]] = None,
    right_by: Optional[Union[str, List[str]]] = None,
    suffixes: Optional[Tuple[str, str]] = None,
    copy: bool = True,
    columns_left: Optional[Union[str, List[str]]] = None,
    columns_right: Optional[Union[str, List[str]]] = None,
    *,
    tolerance: Optional[Union[int, float, "timedelta", TimeSpanScalar]] = None,
    allow_exact_matches: bool = True,
    direction: Literal["backward", "forward", "nearest"] = "backward",
    matched_on: Union[bool, str] = False,
    **kwargs,
) -> "Dataset":
    """
    Perform an as-of merge. This is similar to a left-join except that we match on nearest key rather than equal keys.

    Both Datasets must be sorted (ascending) by the 'on' column. When 'by' columns are specified,
    the 'on' column for each Dataset only needs to be sorted (ascending) within each unique 'key'
    of the 'by' columns. Sorting the entire `Dataset` by the 'on' column also meets this requirement,
    but some `Datasets` may have an 'on' column which is already pre-sorted within each 'by' key,
    in which case no additional sorting is required.

    For each row in the left Dataset:
      - A "backward" search selects the last row in the right Dataset whose
        'on' key is less than or equal to the left's key.
      - A "forward" search selects the first row in the right Dataset whose
        'on' key is greater than or equal to the left's key.
      - A "nearest" search selects the row in the right Dataset whose 'on'
        key is closest in absolute distance to the left's key.

    Optionally match on equivalent keys with 'by' before searching with 'on'.

    Parameters
    ----------
    left : Dataset
        Left Dataset
    right : Dataset
        Right Dataset
    on : str
        Column name to join on. Must be found in both the `left` and `right` Datasets.
        This column in both left and right Datasets MUST be ordered.
        Furthermore this must be a numeric column, such as datetimelike,
        integer, or float. Either `on` or `left_on`/`right_on` must be specified.
    left_on : str or list of str, optional
        Column name to join on in `left` Dataset.
    right_on : str or list of str, optional
        Column name to join on in `right` Dataset.
    by : str or (str, str) or list of str or list of (str, str), optional
        Column name or list of column names. Match on these columns before
        performing merge operation.
    left_by : str or list of str, optional
        Column names to match on in the left Dataset.
    right_by : str or list of str, optional
        Column names to match on in the right Dataset.
    suffixes : (str, str), optional, default None
        Suffix to apply to overlapping column names in the left and right
        side, respectively.
    copy: bool, default True
        If False, avoid copying data when possible; this can reduce memory usage
        but users must be aware that data can be shared between `left` and/or `right`
        and the Dataset returned by this function.
    columns_left : str or list of str, optional
        Column names to include in the merge from `left`, defaults to None which causes all columns to be included.
    columns_right : str or list of str, optional
        Column names to include in the merge from `right`, defaults to None which causes all columns to be included.
    tolerance : integer or float or Timedelta, optional, default None
        Tolerance allowed when performing the 'asof' part of the merge; whenever a row from
        `left` doesn't have a key in `right` within this distance or less, that row will have
        a null/missing/NA value for any columns from the `right` Dataset which appear in the
        merged result.
    allow_exact_matches : boolean, default True
        - If True, allow matching with the same 'on' value
          (i.e. less-than-or-equal-to / greater-than-or-equal-to)
        - If False, don't match the same 'on' value
          (i.e., strictly less-than / strictly greater-than)
    direction : {'backward', 'forward', or 'nearest'}, default 'backward'
        Whether to search for prior, subsequent, or closest matches.
    matched_on : bool or str, default False
        If set to True or a string, an additional column is added to the result;
        for each row, it contains the value from the `on` column in `right` that was matched.
        When True, the column will use the default name 'matched_on'; specify a string
        to explicitly name the column.

    Returns
    -------
    merged : Dataset

    Raises
    ------
    ValueError
        The `on`, `left_on`, or `right_on` columns are not sorted in ascending order
        within one or more keys of the `by` columns.

    See Also
    --------
    riptable.merge_asof

    Notes
    -----
    TODO: Consider allowing the ``tolerance`` parameter to also be a 1D array/sequence whose length is the same
          as the number of groups (or keys?) in the ``left`` dataset. That allows a per-group tolerance to be specified
          if needed.

    Examples
    --------
    >>> left = rt.Dataset({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c']})
    >>> left
    #    a   left_val
    -   --   --------
    0    1   a
    1    5   b
    2   10   c
    >>> right = rt.Dataset({'a': [1, 2, 3, 6, 7],
    ...                       'right_val': [1, 2, 3, 6, 7]})
    >>> right
    #   a   right_val
    -   -   ---------
    0   1           1
    1   2           2
    2   3           3
    3   6           6
    4   7           7
    >>> rt.merge_asof(left, right, on='a')
    #   a_x   left_val   a_y   right_val
    -   ---   --------   ---   ---------
    0     1   a            1           1
    1     5   b            3           3
    2    10   c            7           7

    >>> rt.merge_asof(left, right, on='a', allow_exact_matches=False)
     #   a_x   left_val   a_y   right_val
    -   ---   --------   ---   ---------
    0     1   a          Inv         Inv
    1     5   b            3           3
    2    10   c            7           7

    >>> rt.merge_asof(left, right, on='a', direction='forward')
    #   a_x   left_val   a_y   right_val
    -   ---   --------   ---   ---------
    0     1   a            1           1
    1     5   b            6           6
    2    10   c          Inv         Inv

    Here is a real-world time-series example

    >>> quotes
    #                   time  ticker   Bid   Ask
    -   --------------------  ------  ----  ----
    0   20191015 09:45:57.09    AAPL  3.40  3.50
    1   20191015 11:35:09.76    AAPL  3.45  3.55
    2   20191015 12:02:27.11    AAPL  3.50  3.60
    3   20191015 12:43:13.73    MSFT  2.85  2.95
    4   20191015 14:32:11.18    MSFT  2.90  3.00

    >>> trades
    #                   time  ticker  TradePrice  TradeSize
    -   --------------------  ------  ----------  ---------
    0   20191015 10:03:24.73    AAPL        3.45       1.00
    1   20191015 10:41:22.79    MSFT        2.85       1.00
    2   20191015 10:41:35.69    MSFT        2.86       1.00
    3   20191015 11:04:32.55    AAPL        3.47       1.00
    4   20191015 11:44:35.63    MSFT        2.91       1.00
    5   20191015 12:26:17.68    AAPL        3.55       1.00
    6   20191015 14:24:10.93    MSFT        2.98       1.00
    7   20191015 15:45:13.41    AAPL        3.60       7.00
    8   20191015 15:50:42.53    AAPL        3.58       1.00
    9   20191015 15:53:59.60    AAPL        3.56       5.00

    >>> rt.merge_asof(trades, quotes, on='time', by='ticker')
    #                 time_x   ticker_x  TradePrice   TradeSize                 time_y   ticker_y    Bid    Ask
    -   --------------------  ---------  ----------   ---------   --------------------   --------   ----   ----
    0   20191015 10:03:24.73       AAPL        3.45        1.00   20191015 09:45:57.09       AAPL   3.40   3.50
    1   20191015 10:41:22.79       MSFT        2.85        1.00                    Inv   Filtered    nan    nan
    2   20191015 10:41:35.69       MSFT        2.86        1.00                    Inv   Filtered    nan    nan
    3   20191015 11:04:32.55       AAPL        3.47        1.00   20191015 09:45:57.09       AAPL   3.40   3.50
    4   20191015 11:44:35.63       MSFT        2.91        1.00                    Inv   Filtered    nan    nan
    5   20191015 12:26:17.68       AAPL        3.55        1.00   20191015 12:02:27.11       AAPL   3.50   3.60
    6   20191015 14:24:10.93       MSFT        2.98        1.00   20191015 12:43:13.73       MSFT   2.85   2.95
    7   20191015 15:45:13.41       AAPL        3.60        7.00   20191015 12:02:27.11       AAPL   3.50   3.60
    8   20191015 15:50:42.53       AAPL        3.58        1.00   20191015 12:02:27.11       AAPL   3.50   3.60
    9   20191015 15:53:59.60       AAPL        3.56        5.00   20191015 12:02:27.11       AAPL   3.50   3.60

    only merge with the forward quotes

    >>> rt.merge_asof(trades, quotes, on='time', by='ticker', direction='forward')
    #                 time_x  ticker_x  TradePrice  TradeSize                time_y  ticker_y   Bid   Ask
    -   --------------------  --------  ----------  ---------  --------------------  --------  ----  ----
    0   20191015 10:03:24.73      AAPL        3.45       1.00  20191015 11:35:09.76      AAPL  3.45  3.55
    1   20191015 10:41:22.79      MSFT        2.85       1.00  20191015 12:43:13.73      MSFT  2.85  2.95
    2   20191015 10:41:35.69      MSFT        2.86       1.00  20191015 12:43:13.73      MSFT  2.85  2.95
    3   20191015 11:04:32.55      AAPL        3.47       1.00  20191015 11:35:09.76      AAPL  3.45  3.55
    4   20191015 11:44:35.63      MSFT        2.91       1.00  20191015 12:43:13.73      MSFT  2.85  2.95
    6   20191015 14:24:10.93      MSFT        2.98       1.00  20191015 14:32:11.18      MSFT  2.90  3.00
    7   20191015 15:45:13.41      AAPL        3.60       7.00                   Inv  Filtered   nan   nan
    8   20191015 15:50:42.53      AAPL        3.58       1.00                   Inv  Filtered   nan   nan
    9   20191015 15:53:59.60      AAPL        3.56       5.00                   Inv  Filtered   nan   nan
    5   20191015 12:26:17.68      AAPL        3.55       1.00                   Inv  Filtered   nan   nan
    """
    # Process keyword arguments.
    left_index = bool(kwargs.pop("left_index", False))
    right_index = bool(kwargs.pop("right_index", False))
    _ = bool(kwargs.pop("check_sorted", True))
    _ = bool(kwargs.pop("verbose", False))
    if kwargs:
        # There were remaining keyword args passed here which we don't understand.
        first_kwarg = next(iter(kwargs.keys()))
        raise ValueError(f"This function does not support the kwarg '{first_kwarg}'.")

    if left_index or right_index:
        # Emit warning about 'left_index' and 'right_index' only being present for compatibility
        # with the pandas merge signature. They don't actually do anything in riptable since our
        # indexing is external (not internal) to Datasets.
        warnings.warn(
            "The 'left_index' and 'right_index' parameters are only present for pandas compatibility. They are not applicable for riptable and will have no effect."
        )

    # Validate the 'direction' argument.
    if direction not in ("forward", "backward", "nearest"):
        raise ValueError(f"`{direction}` is not a valid value for the 'direction' parameter.")

    # Validate and normalize the 'on' column name for each Dataset.
    if left_on is None:
        if on is None:
            raise ValueError("The `on` and `left_on` parameters cannot both be None.")
        elif isinstance(on, str):
            left_on = on
        elif isinstance(on, bytes):
            left_on = on.decode()
        else:
            # Assume `on` is a 2-tuple of strings.
            left_on = on[0]
    else:
        # 'on' and 'left_on' are not allowed to be specified together, as it's unclear which one should take precedence.
        # If we want to define which one does take precedence, we could drop this down to
        # a Warning to let the user know part of what they've specified will be ignored.
        if on is not None:
            raise ValueError(
                "The `on` and `left_on` parameters cannot be specified together; exactly one of them should be specified."
            )

    if right_on is None:
        if on is None:
            raise ValueError("The `on` and `right_on` parameters cannot both be None.")
        elif isinstance(on, str):
            right_on = on
        elif isinstance(on, bytes):
            right_on = on.decode()
        else:
            # Assume `on` is a 2-tuple of strings.
            right_on = on[1]
    else:
        # 'on' and 'right_on' are not allowed to be specified together, as it's unclear which one should take precedence.
        # If we want to define which one does take precedence, we could drop this down to
        # a Warning to let the user know part of what they've specified will be ignored.
        if on is not None:
            raise ValueError(
                "The `on` and `right_on` parameters cannot be specified together; exactly one of them should be specified."
            )

    # Validate and normalize the 'by' column name lists for each Dataset.
    # Note that for 'merge_asof', specifying any 'by' columns are optional -- unlike the 'on' columns, which are required.
    left_by = _extract_on_columns(by, left_by, True, "by", is_optional=True)
    right_by = _extract_on_columns(by, right_by, False, "by", is_optional=True)

    #
    # TODO: Disallow a column to be specified as both an 'on' and 'by' column (for each Dataset)?
    #       Using the same column for both means the 'on' column would effectively be ignored since
    #       the 'by' part is already doing an exact match (equi-join). If that's what someone wants,
    #       seems like they should be using merge_lookup() or merge2() instead.
    #

    # Normalize 'columns_left' and 'columns_right' first to simplify some logic later on
    # (by allowing us to assume they're a non-optional-but-maybe-empty List[str]).
    columns_left = _normalize_selected_columns(left, columns_left)
    columns_right = _normalize_selected_columns(right, columns_right)

    # PERF: Revisit this -- it could be made faster if ItemContainer.keys() returned a set-like object such as KeysView instead of a list; then we wouldn't need to create the sets here.
    left_keyset = set(left.keys())
    _require_columns_present(left_keyset, "left", "left_on", [left_on])
    _require_columns_present(left_keyset, "left", "left_by", left_by)
    _require_columns_present(
        left_keyset, "left", "columns_left", columns_left
    )  # PERF: Fix this -- if columns_left isn't populated initially it'll be normalized to the whole keyset above so this call is irrelevant
    right_keyset = set(right.keys())
    _require_columns_present(right_keyset, "right", "right_on", [right_on])
    _require_columns_present(right_keyset, "right", "right_by", right_by)
    _require_columns_present(
        right_keyset, "right", "columns_right", columns_right
    )  # PERF: Fix this -- if columns_right isn't populated initially it'll be normalized to the whole keyset above so this call is irrelevant

    # Make sure there aren't any column name collision _before_ we do the heavy lifting of merging;
    # if there are name collisions, attempt to resolve them by suffixing the colliding column names.
    # For the purposes of this validation, we combine the 'on' column and any 'by' column(s) since they're both
    # keys (it's just that the 'on' column is treated specially when the join is performed).
    left_keycol_names = [left_on]
    left_keycol_names.extend(left_by)
    right_keycol_names = [right_on]
    right_keycol_names.extend(right_by)
    col_left_tuple, col_right_tuple, intersection_cols = _construct_colname_mapping(
        left_keycol_names, right_keycol_names, suffixes=suffixes, columns_left=columns_left, columns_right=columns_right
    )

    # Validate the pair(s) of columns from the left and right join keys have compatible types.
    key_compat_errs = _verify_join_keys_compat(
        [left[col_name] for col_name in left_keycol_names], [right[col_name] for col_name in right_keycol_names]
    )

    if key_compat_errs:
        # If the list of errors is non-empty, we have some join-key compatibility issues.
        # Some "errors" may just be warnings; filter those out of the list and raise
        # them to notify the user.
        actual_errors: List[Exception] = []
        for err in key_compat_errs:
            if isinstance(err, Warning):
                warnings.warn(err)
            else:
                actual_errors.append(err)

        # If there are any remaining errors in the list, those are actual errors so we can't
        # proceed any further. Combine the remaining errors into a single error message then
        # raise a ValueError; in the future we might use something like a .NET AggregateException
        # to wrap multiple errors into one instead to allow for different error types.
        if actual_errors:
            flat_errs = "\n".join(
                map(str, actual_errors)
            )  # N.B. this is because it's disallowed to use backslashes inside f-string curly braces
            raise ValueError(f"Found one or more compatibility errors with the specified 'on' keys:\n{flat_errs}")

    #
    # TODO: Verify the 'on' columns are an ordered, comparable type (int, float, DateTimeNano, datetime, ordered Categorical, etc.)
    #       This is a requirement for them to be used with this function; without this, the 'asof' comparison can't be performed
    #       and will raise an exception later -- better to check here and give the user a better error message.
    #

    datasetclass = type(left)

    left_on_col = left[left_on]
    right_on_col = right[right_on]

    # If 'tolerance' is specified, verify it's a value of a type compatible with the 'on' columns.
    # The value may also be converted to a different scalar type before being passed to the numba-based impl.
    # so it'll be compatible in the JIT-compiled code.
    if tolerance is not None:
        tolerance = _validate_and_normalize_tolerance(left_on_col, right_on_col, tolerance)

    # Construct the Grouping object for each of the join keys.
    start = perf_counter_ns()

    # TODO: Should we always pass True for the 'force_invalids' argument below? The original intent was that we
    #       may want to be able to use the default (non-forced-invalids) behavior of Grouping for optional compatibility
    #       with rt.merge(), but that's no longer planned (users are just migrating to merge2/merge_lookup).
    #       Note, there *may* be something to passing False for the left and right grouping, when performing a
    #       left or right join, respectively -- before making the change to always pass True, verify all the
    #       optimized cases where we can return e.g. None for the fancy index aren't affected by the change.
    how = "left"
    high_card = False
    hint_size = None
    if left_by is None or len(left_by) == 0:
        left_grouping = None
    else:
        left_grouping, _ = _get_or_create_keygroup(
            [left[col_name] for col_name in left_by], 0, how != "left", False, high_card, hint_size
        )
    if right_by is None or len(right_by) == 0:
        right_grouping = None
    else:
        right_grouping, _ = _get_or_create_keygroup(
            [right[col_name] for col_name in right_by], 1, how != "right", how == "outer", high_card, hint_size
        )

    if _logger.isEnabledFor(logging.DEBUG):
        delta = perf_counter_ns() - start
        _logger.debug("Grouping creation complete.", extra={"elapsed_nanos": delta})

    # Construct fancy indices for the left/right Datasets; these will be used to index into
    # columns of the respective datasets to produce new arrays/columns for the merged Dataset.
    if left_grouping is None:
        join_indices = _create_merge_asof_fancy_indices(
            left_on_col,
            right_on_col,
            tolerance=tolerance,
            allow_exact_matches=allow_exact_matches,
            direction=direction,
        )
    else:
        join_indices = _create_merge_asof_grouped_fancy_indices(
            left_grouping,
            right_grouping,
            left_on_col,
            right_on_col,
            tolerance=tolerance,
            allow_exact_matches=allow_exact_matches,
            direction=direction,
        )

    left_fancyindex = join_indices.left_index  # Should always be None for this type of join
    right_fancyindex = join_indices.right_index

    # Begin creating the column data for the 'merged' Dataset.
    out: Dict[str, FastArray] = {}
    start = perf_counter_ns()

    def readonly_array_wrapper(arr: FastArray) -> FastArray:
        """Create a read-only view of an array."""
        new_arr = FastArray(arr)
        new_arr.flags.writeable = False
        return new_arr

    def array_copy(arr: FastArray) -> FastArray:
        return arr.copy()

    # Based on the 'copy' flag, determine which function to use in cases where
    # an input column will contain the same data in the output Dataset.
    array_copier = array_copy if copy else readonly_array_wrapper

    # Fetch/transform columns which overlap between the two Datasets (i.e. columns with these
    # names exist in both Datasets).
    if intersection_cols:
        # If we're missing one of the fancy indices, it means we can just copy the columns
        # over from the corresponding dataset to the result. We still allocate a new array
        # object as a read-only view of the original; this allows the name to be different
        # in the result (e.g. if changed later) than the original.
        if left_fancyindex is None:
            for field in intersection_cols:
                out[field] = array_copier(left[field])
        elif right_fancyindex is None:
            for field in intersection_cols:
                out[field] = array_copier(right[field])

        else:
            # For any columns which overlap (i.e. same name) in the two Datasets,
            # fetch them from the left Dataset; we don't support right-joins for merge_asof
            # so we don't need to handle that here as we do in merge2.
            # Applying the fancy index to transform each column to the correct shape for the resulting Dataset.
            # N.B. This is an arbitrary choice, we could just as easily default to fetching
            #      from the right Dataset instead.
            for field in intersection_cols:
                out[field] = left[field][left_fancyindex]

    if _logger.isEnabledFor(logging.DEBUG):
        delta = perf_counter_ns() - start
        _logger.debug("Transformed columns. cols='%s'", "intersection", extra={"elapsed_nanos": delta})

    # Transform the columns from the left Dataset and store to the new, merged Dataset.
    # If we don't have a left fancy index, it means the fancy index (if present) would simply create
    # a copy of the original column. Use this to optimize memory usage by just referencing (not copying)
    # the original column data in the new dataset.
    start = perf_counter_ns()
    if left_fancyindex is None:
        for old_name, new_name in zip(*col_left_tuple):
            # Wrap the original array in a read-only view; this is necessary to allow the "new" column to
            # have a different name than the original without interfering with the original array / source Dataset.
            # We make the view read-only to help protect the original data from being unexpectedly modified.
            out[new_name] = array_copier(left[old_name])
    else:
        for old_name, new_name in zip(*col_left_tuple):
            out[new_name] = left[old_name][left_fancyindex]

    if _logger.isEnabledFor(logging.DEBUG):
        delta = perf_counter_ns() - start
        _logger.debug("Transformed columns. cols='%s'", "left", extra={"elapsed_nanos": delta})

    # Transform the columns from the right Dataset and store to the new, merged Dataset.
    start = perf_counter_ns()
    if right_fancyindex is None:
        for old_name, new_name in zip(*col_right_tuple):
            # Wrap the original array in a read-only view; this is necessary to allow the "new" column to
            # have a different name than the original without interfering with the original array / source Dataset.
            # We make the view read-only to help protect the original data from being unexpectedly modified.
            out[new_name] = array_copier(right[old_name])
    else:
        for old_name, new_name in zip(*col_right_tuple):
            out[new_name] = right[old_name][right_fancyindex]

    if _logger.isEnabledFor(logging.DEBUG):
        delta = perf_counter_ns() - start
        _logger.debug("Transformed columns. cols='%s'", "right", extra={"elapsed_nanos": delta})

    # If the caller has asked for the 'matched_on' column, create it now.
    if matched_on:
        start = perf_counter_ns()

        if isinstance(matched_on, bool):
            matched_on = "matched_on"
        if matched_on in out:
            raise ValueError(f"`matched_on` column name collision with existing columns: {matched_on}")

        # Use the right_fancyindex to expand the 'on' column from the right Dataset and add it to the output.
        out[matched_on] = right_on_col[right_fancyindex]

        if _logger.isEnabledFor(logging.DEBUG):
            delta = perf_counter_ns() - start
            _logger.debug("matched_on column created.", extra={"elapsed_nanos": delta})

    return datasetclass(out)
