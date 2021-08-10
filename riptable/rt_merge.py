__all__ = [
    # Original merge implementation.
    'merge',
    # New merge implementation.
    'JoinIndices', 'merge2', 'merge_lookup', 'merge_indices',
    # as-of merge
    'merge_asof',
]

import logging
import warnings
from typing import TYPE_CHECKING, Collection, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import numba as nb

from .config import get_global_settings
from .rt_utils import alignmk, mbget, merge_prebinned, get_default_value
from .rt_timers import GetNanoTime
from .rt_numpy import (
    all, arange,
    bool_to_fancy,
    cumsum,
    empty, empty_like, full,
    hstack,
    ismember, isnan, isnotnan, issorted,
    mask_andi, maximum,
    putmask,
    sum, unique, unique32, where
    )
from .rt_categorical import Categorical
from .rt_fastarray import FastArray
from .rt_enum import TypeRegister, INVALID_DICT, int_dtype_from_len
from .rt_grouping import Grouping

if TYPE_CHECKING:
    from datetime import timedelta
    from .rt_dataset import Dataset


# Create a logger for this module.
logger = logging.getLogger(__name__)


def _create_noncompat_join_key_types_errmsg(
    left_key_name: str,
    left_key_type: type,
    right_key_name: str,
    right_key_type: type,
    reason: Optional[str] = None
) -> str:
    """Create an error message for two column types which are not compatible as join-keys."""

    # If the caller didn't supply a reason, provide a generic explanation
    # that's (maybe) still helpful for new users.
    _reason = reason if reason is not None else "You may need to convert one of them into a different type before joining."

    return f'The columns \'{left_key_name}\' ({left_key_type}) and \'{right_key_name}\' ({right_key_type}) are not compatible for use as join-keys. {_reason}'


def _safe_get_colname(column: FastArray) -> str:
    """Get the name of a FastArray or a default name if it has none."""
    column_name = column.get_name()
    return "UNNAMED-COLUMN" if column_name is None else column_name

# TODO: Starting in riptable 1.5, certain detected incompatibilities between join keys
#       will become errors (ValueError) instead of just warnings.
#       (Some "hard" incompatibilities such as different number of join keys from the
#       left and right Datasets are already errors since we cannot proceed in those cases.)
#_key_incompat_exn_type : type = ValueError
_key_incompat_exn_type : type = FutureWarning


def _verify_join_key_compat(left_key: FastArray, right_key: FastArray, arg_order_swapped:bool=False) -> Optional[List[Exception]]:
    """
    Verify whether two columns (ndarray or FastArray) are compatible for use as join-keys in a merge operation.

    By default, FastArray-derived array types like Categorical can only be joined against an array of the same type.
    Certain derived types have (or could have) slightly more relaxed join behavior; e.g. we could support joining a
    Date column with a DateTimeNano column.

    For standard (non-subclass) FastArray instances, we inspect the dtypes of the columns to determine their compatibility.
    The dtypes need not be identical so long as we can support upcasting from one type to the other, that is,
    without loss of information.

    Parameters
    ----------
    left_key : array-like
        The join key from the left Dataset.
    right_key : array-like
        The join key from the right Dataset.
    arg_order_swapped : bool
        Indicates whether `left_key` and `right_key` have been swapped through a self-recursive call to this function.

    Returns
    -------
    errors : list of Exception, optional
        If the columns are compatible for use as join-keys, None is returned. Otherwise, a non-empty list
        of Exceptions containing a diagnostic error message(s) for display to the user is returned.

    Notes
    -----
    TODO ideas for improvements:
      * For date/time-related array types like DateTimeNano -- should the timezone be considered when merging?
        If so, we'll need to write a function like alignmk for dates/times which also handles conversion to a specified timezone.
      * Can the logic in this class be split out into some ABC or mixin class and applied to individual array types?
        Although it'd be a bit more complex/verbose, it would also remove the need for the 'merge' code to know how to
        validate every specific array type; for complex cases like Categorical, there can be a large number of validation checks
        and it'd probably be better in the long run to move all of it inside Categorical itself or delegate it to some "CategoricalMerger" class.
        Another way of looking at this -- we basically want something like __array_function__ from numpy in the sense that we can
        define an operation on a class and override in subclasses, and when we call the method it'll use the most-derived subclasses'
        implementation.
    """
    # TEMP: The logic in the function below needs to be modified to handle cases where the order argument was swapped.
    #       Until then -- for safety/correctness -- raise an exception if the order swap is indicated.
    if arg_order_swapped:
        raise NotImplementedError("Argument order swapping is not implemented yet.")

    # NOTE: The general ordering of the checks below goes from more-specific to less-specific, so we make sure to
    #       use the most-specific handling available for a given array type(s).
    if isinstance(left_key, Categorical):
        # Categoricals are validated based on the type(s) of their column(s) from .category_dict
        # This means Categoricals are treated like we're merging on their .expand_array, which allows a merge to be performed
        # where the key from one side is a Categorical but the key from the other side is a normal (non-Categorical) array type
        # and we'll get the same result as if we were merging non-Categorical arrays.
        # TODO: For diagnostic error messages, may need to track the 'path' down through this Categorical (e.g. it's name)
        #       so we can provide a more-accurate message.
        left_key_cats = left_key.category_dict  # Use .category_dict instead of .categories(); we only care about the type info here, and .categories() allocates a new array
        left_key_cat_cols = left_key_cats.values() if isinstance(left_key_cats, dict) else [left_key_cats]

        # If both sides are Categoricals, save an extra recursive call by drilling down into it at the same time.
        if isinstance(right_key, Categorical):
            right_key_cats = right_key.category_dict
            right_key_cat_cols = right_key_cats.values() if isinstance(right_key_cats, dict) else [right_key_cats]
            return _verify_join_keys_compat(left_key_cat_cols, right_key_cat_cols)

        else:
            return _verify_join_keys_compat(left_key_cat_cols, [right_key])

    elif isinstance(right_key, Categorical):
        # TODO: Maybe replace the code below with a self-recursive call here where we utilize the order-swapping logic;
        #       this'll help keep the code simple/compact and means we have less logic to test/maintain.
        right_key_cats = right_key.category_dict
        right_key_cat_cols = right_key_cats.values() if isinstance(right_key_cats, dict) else [right_key_cats]

        return _verify_join_keys_compat([left_key], right_key_cat_cols)

    #
    # TODO: Implement some 'relaxed' checks e.g. for Date+DateTimeNano once we're ready to support that in the merge code.
    #

    # By default, if a FastArray subclass didn't get any special handling above, be strict and require both columns
    # to be the exact same type for compatibility.
    elif type(left_key) != type(right_key):
        left_key_name = _safe_get_colname(left_key)
        right_key_name = _safe_get_colname(right_key)
        errmsg = _create_noncompat_join_key_types_errmsg(left_key_name, type(left_key), right_key_name, type(right_key),
            reason="Columns must be the same Python __class__ to be mergeable.")
        return [ValueError(errmsg)]

    else:
        # Fallthrough case.
        # At this point, assume arrays are just FastArrays. If they're some more derived type, we don't care to do anything
        # special for that type (or we would have put in a special case for them above).
        # Consider the arrays to be compatible if they have the same dtype or we can _safely_
        # (without loss of information or semantic meaning) cast from one dtype to another.
        # TODO: Should we use the casting='same_kind' mode for the np.can_cast() call here? Doing so would prevent int->float conversions.
        # TODO: Refactor this logic so we can give more-specific error reasons when columns are not compatible? E.g. "integer columns cannot be joined against floating-point columns"
        if np.can_cast(left_key.dtype, right_key.dtype) or np.can_cast(right_key.dtype, left_key.dtype):
            return None
        elif issubclass(left_key.dtype.type, np.integer) and issubclass(right_key.dtype.type, np.integer):
            return None
        elif issubclass(left_key.dtype.type, np.floating) and issubclass(right_key.dtype.type, np.floating):
            return None
        elif left_key.dtype.char in ('S', 'U') and right_key.dtype.char in ('S', 'U'):
            # Allow string keys of differing character types (ASCII vs. Unicode) to be merged.
            # TODO: At present, `merge` is implemented in such a way (via rt.ismember) that this mismatch causes an implicit
            # conversion of one of the string columns to the type of the other (preferring U->S because the result is more compact),
            # and we may want to warn users about this due to the performance overhead / impact on memory utilization.
            return None
        else:
            left_key_name = _safe_get_colname(left_key)
            right_key_name = _safe_get_colname(right_key)
            return [
                _key_incompat_exn_type(f'The columns \'{left_key_name}\' (dtype={left_key.dtype}) and \'{right_key_name}\' (dtype={right_key.dtype}) are not compatible for use as join-keys. You may need to convert one of them into a different type before joining.')]


def _verify_join_keys_compat(left_keys: Collection[FastArray], right_keys: Collection[FastArray]) -> List[Exception]:
    """
    Inspect the list of column(s) being used as join keys from two Datasets to ensure they're compatible for joining.

    Parameters
    ----------
    left_keys : Collection of FastArray
        A list containing the join-key columns from the left Dataset.
    right_keys : Collection of FastArray
        A list containing the join-key columns from the right Dataset.

    Returns
    -------
    errors : list of Exception
        A list of Exceptions whose messages describe why the join-key columns aren't compatible.
        An empty list is returned when the join-key columns are compatible.
    """
    # The left and right side join-keys must be non-empty lists.
    if not left_keys or not right_keys:
        return [ValueError('The left and/or right join-key lists are empty.')]

    # We must have the same number of join keys on the left and right side.
    if len(left_keys) != len(right_keys):
        return [ValueError(f'Differing numbers of columns used as join-keys for the left ({len(left_keys)}) and right ({len(right_keys)}) Datasets.')]

    # Iterate over the pairs of keys, checking compatibility for each pair.
    errors : List[Exception] = []
    for left_key, right_key in zip(left_keys, right_keys):
        tmp = _verify_join_key_compat(left_key, right_key)
        if tmp is not None:
            errors.extend(tmp)

    return errors


def _construct_index(left, right, left_on: List[str], right_on: List[str], how: str, hint_size: int=10000
    ) -> Union[Tuple[FastArray, FastArray, FastArray], Tuple[tuple, tuple, tuple]]:
    """
    Construct fancy indices for the left and right Datasets to be used for creating new columns in the merged Dataset.

    Parameters
    ----------
    left : Dataset
        The left Dataset.
    right : Dataset
        The right Dataset.
    left_on : list of str
        The names of the columns to join on from the `left` Dataset.
    right_on : list of str
        The names of the columns to join on from the `right` Dataset.
    how : {'left', 'right', 'inner', 'outer'}
        A string indicating the type of join to perform.
    hint_size : int
        A hint about the approximate number of unique values within the join keys.
        Used by lower-level parts of the merging code to choose a code path that'll
        provide the best performance.

    Returns
    -------
    idx : FastArray or tuple of FastArray
    idx_left : FastArray or tuple of FastArray
        Fancy index used for indexing into columns of the `left` Dataset to create
        the new columns for the merged Dataset.
    idx_right : FastArray or tuple of FastArray
        Fancy index used for indexing into columns of the `right` Dataset to create
        the new columns for the merged Dataset.
    """
    if type(left_on) != type(right_on):
        raise ValueError("`left_on` and `right_on` must be the same type.")

    # If `left_on` and `right_on` are lists, they must have the same length and be non-empty.
    if isinstance(left_on, list):
        if len(left_on) != len(right_on):
            raise ValueError(f"`left_on` and `right_on` must have the same length ({len(left_on)} vs. {len(right_on)}).")
        elif not left_on:
            raise ValueError("When specified as lists, `left_on` and `right_on` must be non-empty.")

        # TEMP: Workaround until the code in this function can be adapted to better handle
        #       single-item lists. Before we started normalizing arguments being passed to this function,
        #       `left_on` and `right_on` could either be a str or a List[str]. They're always a List[str]
        #       now, so to preserve the existing behavior, unpack single-element lists for now.
        if len(left_on) == 1:
            left_on = left_on[0]
            right_on = right_on[0]

    idx_left = left[left_on]
    idx_right = right[right_on]
    if how == 'left':
        if isinstance(left_on, list):
            #tup_left = tuple([left[col].copy() for col in left_on])
            #tup_right = tuple([right[col].copy() for col in right_on])
            tup_left = tuple([left[col] for col in left_on])
            tup_right = tuple([right[col] for col in right_on])
            return tup_left, tup_left, tup_right
        else:
            return idx_left, idx_left, idx_right

    elif how == 'right':
        if isinstance(right_on, list):
            # TODO: Use the non-.copy()-based version here as for the 'left' join above?
            tup_left = tuple([left[col].copy() for col in left_on])
            tup_right = tuple([right[col].copy() for col in right_on])
            return tup_right, tup_left, tup_right
        else:
            return idx_right, idx_left, idx_right

    elif how == 'outer':
        if isinstance(left_on, list):
            idxList = []
            for col_num in range(len(left_on)):
                idxList.append(np.hstack([left[left_on[col_num]], right[right_on[col_num]]]))
            idxFilter, _ = unique32(idxList, hint_size)

            # TODO: Use the non-.copy()-based version here as for the 'left' join above?
            tup_left = tuple([left[col].copy() for col in left_on])
            tup_right = tuple([right[col].copy() for col in right_on])

            tup = tuple([idx[idxFilter] for idx in idxList])
            return tup, tup_left, tup_right
        else:
            stacked_idx = hstack((idx_left, idx_right))
            idx, _ = unique32([stacked_idx])
            idx = stacked_idx[idx]
            # TJD this code appears to be dependent on categoricals, filtering and re-expansion
            # idx = unique(stacked_idx, sorted=False)
            return idx, idx_left, idx_right

    elif how == 'inner':
        if isinstance(left_on, list):
            # TODO: Use the non-.copy()-based version here as for the 'left' join above?
            tup_left = tuple([left[col].copy() for col in left_on])
            tup_right = tuple([right[col].copy() for col in right_on])

            is_shared, _ = ismember(tup_left, tup_right)
            idx = idx_left[is_shared, :]
            tup = tuple([idx[col] for col in left_on])
            return tup, tup_left, tup_right
        else:
            is_shared, _ = ismember(idx_left, idx_right)
            idx = idx_left[is_shared]
            return idx, idx_left, idx_right

    else:
        raise ValueError(f'The value \'{how}\' is not valid for the \'how\' parameter.')


def _construct_colname_mapping(
    left_on: Collection[str],
    right_on: Collection[str],
    suffixes: Optional[Tuple[str, str]],
    columns_left: Collection[str],
    columns_right: Collection[str]
) -> Tuple[Tuple[Collection[str], Collection[str]], Tuple[Collection[str], Collection[str]], Set[str]]:
    """
    Resolve column name collisions in the set of columns being merged from the left and right Datasets.

    Check for any column name collisions (from the left and right datasets), and if present,
    attempt to resolve them by suffixing the colliding column names.

    Notes
    -----
    TODO: Consider using the bimap / bijective dictionary data structure here; there's a nice implementation
    in the Python 'bidict' package. Basically, the code below could be cleaned up and sped up if we could do
    set-like operations on both the keys and values in the left and right name-mappings.
    https://bidict.readthedocs.io/en/master/
    """
    # Alias these parameters for backwards-compatibility with the code below.
    col_left = columns_left
    col_right = columns_right

    left_on_set = set(left_on)
    right_on_set = set(right_on)

    # Names from the 'on' clause that are the same in both Datasets are not considered to conflict;
    # so filter those out of the list of other columns we want to be included in the merged Dataset.
    intersection_cols = left_on_set & right_on_set
    if intersection_cols:
        col_left_only = [c for c in col_left if c not in intersection_cols]
        col_right_only = [c for c in col_right if c not in intersection_cols]
    else:
        col_left_only = col_left
        col_right_only = col_right
    col_left_only_set = set(col_left_only)
    col_right_only_set = set(col_right_only)

    # TODO: Implement short-circuit for case when there is no overlap.
    col_overlap = col_left_only_set & col_right_only_set
    if col_overlap:
        if suffixes is None:
            raise ValueError(f'columns overlap but no suffix specified: {col_overlap}')
        if suffixes[0] == suffixes[1]:
            raise ValueError(f'columns overlap but suffixes are the same: {suffixes[0]}')

    ##
    # Column name collision can happen after we apply the suffixes.
    # It can happen within a dataset or between the two datasets so we must check for both.
    ##

    # Check for self-overlap within each Dataset.
    def check_self_overlap(renamed_cols: Set[str], original_cols: Set[str], dataset_name: str) -> None:
        if not renamed_cols.isdisjoint(original_cols):
            raise ValueError(
                f"Column name self-collision(s) in the '{dataset_name}' Dataset after applying suffixes. Columns={renamed_cols & original_cols}")

    def create_suffixed_remapped_cols(
        on_col_set: Set[str], only_cols: List[str], only_cols_set: Set[str],
        col_overlap: Set[str], dataset_name: str, suffix: Optional[str]
    ) -> List[str]:
        if suffix is None or not suffix:
            # Return the original column list, since we're not going to do any remapping for it.
            return only_cols

        overlap_renamed_cols = {col + suffix for col in col_overlap}

        check_self_overlap(overlap_renamed_cols, on_col_set, dataset_name)
        check_self_overlap(overlap_renamed_cols, only_cols_set, dataset_name)

        # Create the list of 'new' column names for this Dataset.
        # For each of the original column names selected, this is the suffixed name (if there is one)
        # or the original name.
        return [(col + suffix if col in col_overlap else col) for col in only_cols]

    left_suffix = '' if suffixes is None else suffixes[0]
    right_suffix = '' if suffixes is None else suffixes[1]

    new_col_left = create_suffixed_remapped_cols(left_on_set, col_left_only, col_left_only_set, col_overlap, 'left', left_suffix)
    new_col_right = create_suffixed_remapped_cols(right_on_set, col_right_only, col_right_only_set, col_overlap, 'right', right_suffix)

    # Are there still overlaps between the two sets of columns after renaming?
    new_col_left_set = set(new_col_left)
    new_col_right_set = set(new_col_right)
    if not new_col_left_set.isdisjoint(new_col_right_set):
        col_collision = new_col_left_set & new_col_right_set
        raise ValueError(f'column name collision after applying suffixes: {col_collision}')

    # TODO: Consider returning the left and right tuples as dictionaries instead (mapping new names to old names or vice versa).
    #       If we do this, we don't need to separately return 'intersection_cols', and the 'on' columns can be included
    #       in the dictionaries (they'll just be mapped to themselves like any other column that's not renamed).
    #       This will make it easier to restructure the code that consumes these results so that it places the columns
    #       in the correct/expected ordering in the merged (result) Dataset.
    return (col_left_only, new_col_left), (col_right_only, new_col_right), intersection_cols


def _get_keep_ifirstlastkey(grouping: 'Grouping', keep: Optional[str]) -> Optional[FastArray]:
    """
    Use the side-specific form of the 'keep' parameter to fetch ifirstkey or ilastkey from a Grouping instance.
    """
    if keep is None:
        return None
    elif keep == 'first':
        return grouping.ifirstkey
    elif keep == 'last':
        return grouping.ilastkey
    else:
        raise ValueError(f"Invalid argument value passed for the `keep` parameter.")


def _get_keep_ifirstlastkey_with_null(grouping: 'Grouping', keep: Optional[str]) -> Optional[FastArray]:
    """
    Use the side-specific form of the 'keep' parameter to fetch ifirstkey or ilastkey from a Grouping instance,
    or an ifirstkey/ilastkey-like array that also includes an entry for the invalid/NA group if any such entries
    exist in the data the Grouping was created from.
    """
    if keep is None:
        return None
    elif keep == 'first':
        # Does the original data contain any null/invalid/NA entries?
        if grouping.base_index > 0 and grouping.ncountgroup[0] > 0:
            # Create a new array like ifirstkey but with an additional entry for the invalid/NA group.
            # NOTE: It's not actually correct to use the original ifirstkey's dtype for the new array here; for example
            #       we could have a case where we have an (original data) array with 100k elements, and all the normal
            #       keys appear in the first ~30k rows so we use an int16 for the ifirstkey dtype, but there's an
            #       invalid/NA which first occurs in the last element of the array -- if we want to represent that
            #       index we'd need to use a wider dtype.
            orig_ifirstkey = grouping.ifirstkey
            invalid_key_first_idx = grouping.igroup[0]  # This works because the invalid group is always first (i.e. ifirstgroup[0] == 0).

            # Determine the dtype needed to hold the result.
            ifirstkey_dtype = np.promote_types(orig_ifirstkey.dtype, int_dtype_from_len(invalid_key_first_idx))

            ifirstkey = empty(len(orig_ifirstkey) + 1, dtype=ifirstkey_dtype)
            ifirstkey[1:] = orig_ifirstkey
            ifirstkey[0] = invalid_key_first_idx

            return ifirstkey
        else:
            return grouping.ifirstkey

    elif keep == 'last':
        # Does the original data contain any null/invalid/NA entries?
        if grouping.base_index > 0 and grouping.ncountgroup[0] > 0:
            # Create a new array like ilastkey but with an additional entry for the invalid/NA group.
            # NOTE: It's not actually correct to use the original ilastkey's dtype for the new array here; for example
            #       we could have a case where we have an (original data) array with 100k elements, and all the normal
            #       keys appear in the first ~30k rows so we use an int16 for the ilastkey dtype, but there's an invalid/NA
            #       in the last element of the array -- if we want to represent that index we'd need to use a wider dtype.
            orig_ilastkey = grouping.ilastkey
            invalid_key_last_idx = grouping.igroup[grouping.ncountgroup[0] - 1]  # This works because the invalid group is group 0 (as long as base_index > 0).

            # Determine the dtype needed to hold the result.
            ilastkey_dtype = np.promote_types(orig_ilastkey.dtype, int_dtype_from_len(invalid_key_last_idx))

            ilastkey = empty(len(orig_ilastkey) + 1, dtype=ilastkey_dtype)
            ilastkey[1:] = orig_ilastkey
            ilastkey[0] = invalid_key_last_idx

            return ilastkey
        else:
            return grouping.ilastkey

    else:
        raise ValueError(f"Invalid argument value passed for the `keep` parameter.")


def _get_keep_ikey(grouping: 'Grouping', keep: Optional[str]) -> FastArray:
    """
    TODO: Add description
    """
    if keep is None:
        return grouping.ikey
    else:
        # TODO: For the keep_left is not None case, we should post-process the result of indexing into the .ikey
        #       so the null/invalid case is sorted correctly w.r.t. the other elements (relative to the order
        #       in which they originally occurred).
        new_ikey = grouping.ikey[_get_keep_ifirstlastkey(grouping, keep)]
        return new_ikey


def _get_keep_ikey_with_null(grouping: 'Grouping', keep: Optional[str]) -> FastArray:
    """
    TODO: Add description
    """
    if keep is None:
        return grouping.ikey
    else:
        # TODO: For the keep_left is not None case, we should post-process the result of indexing into the .ikey
        #       so the null/invalid case is sorted correctly w.r.t. the other elements (relative to the order
        #       in which they originally occurred).
        new_ikey = grouping.ikey[_get_keep_ifirstlastkey_with_null(grouping, keep)]
        return new_ikey


def _get_keep_ncountgroup(grouping: 'Grouping', keep: Optional[str]) -> FastArray:
    """
    Use the side-specific form of the 'keep' parameter to fetch the ``Grouping.ncountgroup``
    from a Grouping instance and adjust it if necessary.

    Notes
    -----
    This function could be removed / optimized away in the future. Functions calling this
    could just build in their own logic for handling the 'keep' parameter, which would likely
    allow the array allocation in the "keep is not None" case to be eliminated.
    """
    if keep is None:
        return grouping.ncountgroup
    elif keep == 'first' or keep == 'last':
        return maximum(grouping.ifirstkey, 1)
    else:
        raise ValueError(f"Invalid argument value passed for the `keep` parameter.")


# TODO: There seems to be some flakiness or race condition when parallel=True for this function;
#       the output is sometimes different given the same inputs, which leads to correctness issues.
#       The issue does not occur when parallel=False or when NUMBA_DISABLE_JIT=1 is set (to disable jitting).
@nb.njit(parallel=False, cache=get_global_settings().enable_numba_cache, nogil=True)
def _build_right_fancyindex_impl(
    left_rows_with_right_keyids: np.ndarray,
    right_iFirstGroup: np.ndarray,
    right_iGroup: np.ndarray,
    row_cumcount: np.ndarray,
    invalid_index: int,
    right_fancyindex: np.ndarray):
    """
    Each row of 'left' is expanded in-place the number of times that
    key occurs in right; then we copy the entries for that key from the right-side iGroup to the array elements
    corresponding to the expanded left rows. In other words, we're (more or less) reordering the groups within the
    right iGroup (but leaving the elements _within_ the groups as-is), and dropping any groups which don't exist
    in left (for an inner join) or just putting an invalid in place (for left outer join).

    Parameters
    ----------
    left_rows_with_right_keyids : np.ndarray
    right_iFirstGroup : np.ndarray
    right_iGroup : np.ndarray
    row_cumcount : np.ndarray
    invalid_index : int
    right_fancyindex : np.ndarray

    Notes
    -----
    TODO: Add at least one example showing how to invoke this function (i.e. what goes into it) and the output.
    """
    output_length = len(right_fancyindex)
    group_count = len(right_iFirstGroup)
    for left_rowidx in nb.prange(len(left_rows_with_right_keyids)):
        # The starting index/offset for this row within the *output* fancy array.
        fancy_row_offset = 0 if left_rowidx == 0 else row_cumcount[left_rowidx - 1]

        # For an inner join, we only want rows which have keys belonging to both the left and right keygroup.
        # If the 'right' dataset has one or more keys at the end of it's array which aren't in 'left',
        # the output array won't have any entries to write to -- trying to do so here will cause an access violation
        # by writing past the end of the array.
        if fancy_row_offset < output_length:
            # If this row is an invalid value (i.e. will be a null/invalid in the final output),
            # just store it to the corresponding output array -- nothing else to do after that (for this row).
            keyid = left_rows_with_right_keyids[left_rowidx]
            if keyid <= 0 or keyid >= group_count:
                right_fancyindex[fancy_row_offset] = invalid_index
            else:
                # How many elements are in the group for the current row?
                curr_group_nCount = row_cumcount[left_rowidx] - fancy_row_offset

                # What's the start index for this group's row-indices within the iGroup array?
                curr_group_offset = right_iFirstGroup[keyid]

                # Copy the right-row-indices for this group from the iGroup to the fancy array.
                for i in range(curr_group_nCount):
                    right_fancyindex[fancy_row_offset + i] = right_iGroup[curr_group_offset + i]


@nb.njit(parallel=True, cache=get_global_settings().enable_numba_cache, nogil=True)
def _fancy_index_fetch_keymap(left_keyid_to_right_keyid_map: np.ndarray, left_ikey: np.ndarray, invalid_value: int, output: np.ndarray):
    """
    This function is a temporary, performance-oriented workaround for the left<->right key mappings being
    created as 0-indexed rather than 1-indexed (which would allow them to be used directly as fancy
    indices to e.g. ifirstkey/ilastkey/ncountgroup. When the code for creating the mappings
    is fixed to create them as 1-indexed (and always placing an invalid/NA for the invalid/NA bin),
    this function won't be needed any longer.

    Parameters
    ----------
    left_keyid_to_right_keyid_map : FastArray
        An array containing the right-ikeys of the gbkey values
        which also occur in the left-gbkeys. Used to transform the
        ikey array from `left_keygroup` into an array of the same
        length as the left ikey array but containing the corresponding
        right-ikeys.
    left_ikey : np.ndarray
    invalid_value : int
    output : np.ndarray
        Array that will be populated with the constructed fancy index.
        Expected to be the same length as `left_ikey`.
        The keyid values in this array are 1-based so the array can be used as
        a fancy index into e.g. an ifirstkey/ilastkey/ncountgroup/etc.

    Notes
    -----
    This function basically implements this:
        output = left_keyid_to_right_keyid_map[left_ikey - 1]

    However, the naive code (above) doesn't work correctly for Grouping objects where
    the underlying column(s) contained an invalid bin; those rows of the columns are assigned
    to the invalid bin (ikey = 0), so subtracting 1 means fancy indexing just wraps around
    and returns the last element of the array. Using the resulting array ends up giving the wrong
    results.

    We could fix the naive version like this:
        right_key_idx = left_ikey - 1
        rt.putmask(right_key_idx, right_key_idx == -1, rt.get_default_value(right_key_idx))
        output = left_keyid_to_right_keyid_map[right_key_idx]

    However, that version also allocates an extra boolean mask array. This function solves both issues
    by fusing the operations together.
    """
    keyid_map_len = len(left_keyid_to_right_keyid_map)

    for i in nb.prange(len(left_ikey)):
        raw_left_ikey = left_ikey[i]

        # If this element of the ikey array is 0, that means the corresponding row of the key column(s)
        # contained an invalid/NaN. Map those rows to the invalid_value for the fancy index we're constructing
        # so they'll be filtered out later.
        # For all other ikey values, subtract one (to account for ikey values being 1-indexed),
        # then fetch the element at that index from the left<->right key mapping array;
        # add one back to the element so it's a valid, 1-based "ikey" / keyid value.
        if raw_left_ikey == 0 or raw_left_ikey > keyid_map_len:
            output[i] = invalid_value
        else:
            curr_value = left_keyid_to_right_keyid_map[raw_left_ikey - 1]
            # Don't add one to negative numbers; these should all be invalids, and we don't want to
            # turn them into still-out-of-bounds-but-not-recognized-as-invalid values.
            output[i] = curr_value if curr_value < 0 else curr_value + 1


def _build_right_fancyindex(
    left_keyid_to_right_keyid_map: FastArray,
    left_keygroup: 'Grouping',
    right_keygroup: 'Grouping',
    is_inner_join: bool,
    keep: Tuple[Optional[str], Optional[str]]
) -> FastArray:
    """
    Build the fancy index array to be used for transforming columns of the 'right' Dataset
    in a join/merge operation into the corresponding columns for the output Dataset.

    Parameters
    ----------
    left_keyid_to_right_keyid_map : FastArray
        An array containing the right-ikeys of the gbkey values
        which also occur in the left-gbkeys. Used to transform the
        ikey array from `left_keygroup` into an array of the same
        length as the left ikey array but containing the corresponding
        right-ikeys.
    left_keygroup : Grouping
        The Grouping object representing the key(s) of the left Dataset in the join.
    right_keygroup : Grouping
        The Grouping object representing the key(s) of the right Dataset in the join.
    is_inner_join : bool
        Indicates whether the fancy index is being constructed for an inner join.
        If True, rows whose key only exists in one of `left_keygroup` or `right_keygroup`
        are dropped; when False, left outer join semantics are used.
    keep : 2-tuple of ({'first', 'last'}, optional)
        A 2-tuple of optional strings specifying that only the first or last occurrence
        of each unique key within `left_keygroup` and `right_keygroup` should be kept.
        In other words, resolves multiple occurrences of keys (multiplicity > 1) to a single occurrence.

    Returns
    -------
    right_fancyindex : FastArray
        An integer FastArray to be used as a fancy index for the columns
        of the right Dataset in the join.
    """
    # Unpack 'keep' for later use.
    keep_left, keep_right = keep

    left_ikey = left_keygroup.ikey if keep_left is None else left_keygroup.ikey[_get_keep_ifirstlastkey_with_null(left_keygroup, keep_left)]

    # Create an array with the same shape as the 'left' keygroup.
    # Populate each element with the 1-based keyid of the matching key from 'right';
    # if there is no matching key in 'right', the element is populated with the invalid value.
    left_rows_with_right_keyids = empty_like(left_ikey, dtype=left_keyid_to_right_keyid_map.dtype)
    invalid_index_val = get_default_value(left_rows_with_right_keyids)
    _fancy_index_fetch_keymap(left_keyid_to_right_keyid_map, left_ikey, invalid_index_val, left_rows_with_right_keyids)

    # When duplicate rows (per key) are being dropped from the *right* Dataset, it means we don't
    # need to expand the key multiplicities for the left Dataset; we're then either preserving or
    # reducing the key multiplicities in the left Dataset, and all of these cases can be handled
    # in optimized ways.
    if keep_right is not None:
        if is_inner_join:
            if keep_left is None:
                # Dropping duplicates in right, preserving key multiplicity from left, keeping only keys in both left and right.
                valid_left_row_mask = isnotnan(left_rows_with_right_keyids)
                valid_left_rows_with_right_keyids = left_rows_with_right_keyids[valid_left_row_mask]
                return _get_keep_ifirstlastkey(right_keygroup, keep_right)[valid_left_rows_with_right_keyids - 1]

            else:
                # Dropping duplicates in both left and right; key multiplicity is 1, keeping only keys in both left and right.
                valid_left_row_mask = isnotnan(left_rows_with_right_keyids)
                valid_left_rows_with_right_keyids = left_rows_with_right_keyids[valid_left_row_mask]
                return _get_keep_ifirstlastkey(right_keygroup, keep_right)[valid_left_rows_with_right_keyids - 1]

        else:
            if keep_left is None:
                # Dropping duplicates in right, preserving key multiplicity from left, keeping all keys in left (need to
                # put in invalids/NA in the fancyindex corresponding to the rows in left with keys that aren't in right).
                return _get_keep_ifirstlastkey(right_keygroup, keep_right)[left_rows_with_right_keyids - 1]

            else:
                # Dropping duplicates in both left and right; key multiplicity is 1, keeping all keys in left (need to
                # put in invalids/NA in the fancyindex corresponding to the rows in left with keys that aren't in right).
                return _get_keep_ifirstlastkey(right_keygroup, keep_right)[left_rows_with_right_keyids - 1]

    # Create an array which matches up to the left rows and contains for each row the right-multiplicity
    # for that row's key.
    # PERF: Worth implementing a "coalescing" fancy-index fetch operation for this? I.e. where we're doing arr[idx],
    #       but we can also supply a scalar value to fill in any elements (in the result) where 'idx' has an invalid value.
    #       If we had this, we could use it here to eliminate the need to have a separate .fillna() call after this.
    shuffled_group_count = right_keygroup.ncountgroup[left_rows_with_right_keyids]

    # For any rows in 'left' whose key didn't exist in 'right'
    # determine whether we'll create a corresponding row in the right fancy-index
    # (which will be filled with an invalid) or not (if we're just dropping the row,
    # the correct behavior for an inner join).
    nonmatched_row_group_count = 0 if is_inner_join else 1
    shuffled_group_count.fillna(nonmatched_row_group_count, inplace=True)

    # Determine the total number of rows (as a scalar) -- we use this to determine the most compact dtype
    # we can use for the cumulative row count array.
    total_row_group_count = sum(shuffled_group_count, dtype=np.int64)

    # NOTE: See comment in _build_left_fancyindex() for why we add one here; it's to handle the case where the row count
    #       is a riptable sentinel value.
    # TODO: Implement an rt.min_scalar_type() function to handle this automatically.
    shuffled_group_cumcount_dtype = np.min_scalar_type(total_row_group_count + 1)
    shuffled_group_cumcount = cumsum(shuffled_group_count, dtype=shuffled_group_cumcount_dtype)

    # Allocate the array which'll hold our fancy index for the 'right' Dataset.
    # Derive the dtype to use from the length of the 'right' Dataset; that's because
    # the values in the fancy index array can't be any larger than that (since they're
    # element/row indices into the 'right' Dataset).
    right_fancyindex_dtype = int_dtype_from_len(len(right_keygroup.ikey))
    right_fancyindex = empty(shuffled_group_cumcount[-1], dtype=right_fancyindex_dtype)

    # Call the numba function to create the fancy index for the 'right' Dataset.
    # Pass the invalid value for the fancy index so we ensure it always matches (e.g. don't want
    # to have an int32 invalid in an int64 array or vice versa).
    invalid_right_index_val = INVALID_DICT[right_fancyindex_dtype.num]
    _build_right_fancyindex_impl(left_rows_with_right_keyids, right_keygroup.ifirstgroup, right_keygroup.igroup, shuffled_group_cumcount, invalid_right_index_val, right_fancyindex)

    # Return the constructed fancy index.
    return right_fancyindex


@nb.njit(parallel=True, cache=get_global_settings().enable_numba_cache, nogil=True)
def _partial_repeat(arr: np.ndarray, repeats_cumsum: np.ndarray, output: np.ndarray):
    """
    Partial implementation of np.repeat.

    Takes an array created with cumsum(), and for each element, repeats that element the
    number of times specified by the diff between the element's corresponding value
    in the cumsum array and the previous element's corresponding value in the cumsum array.

    Parameters
    ----------
    arr : np.ndarray
        The array of elements to be repeated.
    repeats_cumsum : np.ndarray
        The 'repeats' array you'd pass to np.repeat(), except you've taken that array
        and called ``cumsum()`` on it. Having this array in cumsum form allows this function
        to be trivially parallelized.
        This array is expected to be the same length as `arr`.
    output : np.ndarray
        The output array where the results of this function are written to.
        The length of this array is expected to be the same as ``repeats_cumsum[-1]``.

    Examples
    --------
    >>> data = np.array([2, 3, 5, 7, 11, 13], dtype=np.int16)
    >>> repeat_count = np.array([2, 4, 1, 3, 0, 2])
    >>> total_count = repeat_count.sum(dtype=np.int64)
    >>> repeat_count_cumsum = repeat_count.cumsum(dtype=np.min_scalar_type(total_count))  # NOTE: For safety, you typically want to specify the dtype for cumsum here to avoid overflows.
    >>> result = np.empty_like(data, size=total_count)
    >>> _partial_repeat(repeat_count_cumsum, result)
    >>> result
    array([2, 2, 3, 3, 3, 3, 5, 7, 7, 7, 13, 13], dtype=int16)
    """
    element_count = len(arr)
    for i in nb.prange(element_count):
        # What's the starting index where we'll write this index to the output?
        output_start_index = 0 if i == 0 else repeats_cumsum[i - 1]

        # How many times should this element be repeated?
        repeat_count = repeats_cumsum[i] - output_start_index

        # Copy this element to the output array the specified number of times.
        for j in range(repeat_count):
            output[output_start_index + j] = arr[i]


@nb.njit(parallel=True, cache=get_global_settings().enable_numba_cache, nogil=True)
def _fused_arange_repeat(repeats_cumsum: np.ndarray, output: np.ndarray):
    """
    Fused implementation of np.arange and np.repeat.

    Takes an array created with cumsum(), and for each element, repeats that element's
    index the number of times specified by the diff between the element's corresponding value
    in the cumsum array and the previous element's corresponding value in the cumsum array.

    Equivalent to the following psuedo-code:
    ``np.repeat(np.arange(upper_bound), array_of_repeats_values)``

    Parameters
    ----------
    repeats_cumsum : np.ndarray
        The 'repeats' array you'd pass to np.repeat(), except you've taken that array
        and called ``cumsum()`` on it. Having this array in cumsum form allows this function
        to be trivially parallelized.
    output : np.ndarray
        The output array where the results of this function are written to.
        The length of this array is expected to be the same as ``repeats_cumsum[-1]``.

    Examples
    --------
    >>> repeat_count = np.array([2, 4, 1, 3, 0, 2])
    >>> total_count = repeat_count.sum(dtype=np.int64)
    >>> repeat_count_cumsum = repeat_count.cumsum(dtype=np.min_scalar_type(total_count))  # NOTE: For safety, you typically want to specify the dtype for cumsum here to avoid overflows.
    >>> result = np.empty(total_count, dtype=np.min_scalar_type(total_count))
    >>> _fused_arange_repeat(repeat_count_cumsum, result)
    >>> result
    array([0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 5, 5], dtype=uint8)
    """
    element_count = len(repeats_cumsum)
    for i in nb.prange(element_count):
        # What's the starting index where we'll write this index to the output?
        output_start_index = 0 if i == 0 else repeats_cumsum[i - 1]

        # How many times should this element be repeated?
        repeat_count = repeats_cumsum[i] - output_start_index

        # Write this index to the output array the specified number of times.
        for j in range(repeat_count):
            output[output_start_index + j] = i


def _build_left_fancyindex(
    right_keyid_to_left_keyid_map: FastArray,
    right_key_exists_in_left: FastArray,
    left_keygroup: 'Grouping',
    right_keygroup: 'Grouping',
    is_inner_join: bool,
    keep: Tuple[Optional[str], Optional[str]],
) -> Optional[FastArray]:
    """
    Build the fancy index array to be used for transforming columns of the 'left' Dataset
    in a join/merge operation into the corresponding columns for the output Dataset.

    Parameters
    ----------
    right_keyid_to_left_keyid_map : FastArray
        An array containing the left-ikeys of the gbkey values
        which also occur in the right-gbkeys. Used to transform the
        ikey array from `right_keygroup` into an array of the same
        length as the right ikey array but containing the corresponding
        left-ikeys.
    right_key_exists_in_left : FastArray
        A boolean FastArray of the same length as ``right_keygroup.ncountkey``,
        indicating which of the keys in `right_keygroup` also exist in `left_keygroup`.
    left_keygroup : Grouping
        The Grouping object representing the key(s) of the left Dataset in the join.
    right_keygroup : Grouping
        The Grouping object representing the key(s) of the right Dataset in the join.
    is_inner_join : bool
        Indicates whether the fancy index is being constructed for an inner join.
        If True, rows whose key only exists in one of `left_keygroup` or `right_keygroup`
        are dropped; when False, left outer join semantics are used.
    keep : 2-tuple of ({'first', 'last'}, optional)
        A 2-tuple of optional strings specifying that only the first or last occurrence
        of each unique key within `left_keygroup` and `right_keygroup` should be kept.
        In other words, resolves multiple occurrences of keys (multiplicity > 1) to a single occurrence.

    Returns
    -------
    left_fancyindex : FastArray, optional
        An integer FastArray to be used as a fancy index for the columns
        of the left Dataset in the join. For certain combinations of `is_inner_join`
        and `keep`, the fancy index is guaranteed to look something like ``arange(len(left_ds))``,
        i.e. we'd just be copying the columns of the left Dataset. In those cases,
        we return None from this function because creating and returning (and using)
        a fancy index like that is redundant; callers of this function can check for
        None and decide whether to make a copy of the columns of the left Dataset
        or to use them directly in the merged result (without making a copy).
    """
    # Unpack 'keep' for later use.
    keep_left, keep_right = keep

    # Check if the right grouping has any duplicate keys. If all rows have a unique key,
    # we can take the same optimized code path below (for left joins) as if the 'keep' parameter was specified.
    # PERF: This could be refined further if desired -- if doing a left join, we just need
    #       the rows with the keys from the left to be unique; so we can take the optimized
    #       code path even if there are other keys in right which appear in multiple rows.
    right_keys_all_unique = right_keygroup.unique_count == len(right_keygroup.ikey)

    # When duplicate rows (per key) are being dropped from the *right* Dataset, it means we don't
    # need to expand the key multiplicities for the left Dataset; we're then either preserving or
    # reducing the key multiplicities in the left Dataset, and all of these cases can be handled
    # in optimized ways.
    # This logic also applies if all rows in the right Dataset have unique keys (i.e. there are no
    # duplicate rows per key).
    if keep_right is not None or right_keys_all_unique:
        if is_inner_join:
            # TODO: As in code further down in this function, the creation of this array requires some workarounds to
            #       handle the difference between 0-based "key index" values and 1-based key ids that work more naturally
            #       with grouping arrays. This causes some additional allocations (and complexity) so should be cleaned up.
            left_group_exists_in_right = full(left_keygroup.ncountgroup.shape, False, dtype=bool)
            left_group_exists_in_right[1:][right_keyid_to_left_keyid_map[right_key_exists_in_left]] = True

            if keep_left is None:
                # We're not dropping duplicates from the left Dataset.
                # This means we're keeping just the keys shared by left and right but preserving
                # the key multiplicities from 'left' (since the key multiplicities in right are now 0 or 1).
                # We can achieve this by choosing for each row in the left Dataset a boolean indicating
                # whether that row's key also exists in right. When used to index columns from left, this'll
                # drop the rows whose keys don't exist in right and keep the rest (so the multiplicities of
                # remaining keys stays the same).
                # PERF: Benchmark whether it's faster to return this as a boolean mask or to pay the one-time cost
                #       of converting this to a fancy index using where() or bool_to_fancy(); it's possible the fancy
                #       index could be faster because for the boolean mask the boolsum operation will need to be performed
                #       each time it's applied to a column. Or, if it's sometimes faster one way and sometimes the other
                #       we can just keep returning the mask here and having the caller check the dtype of the array and
                #       deciding themselves whether to call where() or bool_to_fancy().
                left_row_mask = left_group_exists_in_right[left_keygroup.ikey]
                #return left_row_mask  # TODO: Some downstream stuff doesn't work properly with boolean masks; convert to a fancy index until they're fixed.
                return bool_to_fancy(left_row_mask)

            else:
                # We're dropping duplicates from the left Dataset too.
                # Using left_keygroup.ifirstkey/.ilastkey here handles the "drop duplicates" part,
                # and we use the mask indicating which keys/groups are shared between the left and
                # right Datasets to keep just those entries of the .ifirstkey/.ilastkey.
                # We have to slice off the first element of the mask (corresponding to the invalid bin) so it
                # matches the length of the ifirstkey/ilastkey.
                return _get_keep_ifirstlastkey(left_keygroup, keep_left)[left_group_exists_in_right[1:]]

        else:
            # This is a "left outer join" and we're not dropping duplicates from the left Dataset
            # and we *are* dropping duplicates from the right Dataset, so the fancy index created by this
            # function would be the same as arange(len(left_dataset)). Instead of creating that, short-circuit
            # and return None -- users of this function can decide how to handle the result.
            if keep_left is None:
                # Index here would be using (only) the keys from left with the same multiplicities as in left
                # (since we're dropping dups on the right, i.e. all keys shared by left/right will have the same
                # multiplicity in the merged result).
                # return arange(len(left_keygroup.ikey))
                return None

            else:
                # Since this is a left outer join we're only keeping the keys from left; when we're dropping
                # duplicates on both sides, the fancy index we return should give us just the keys from left
                # where each key occurs just once. This means we can simply return the .ifirstkey / .ilastkey.
                # We need to use _get_keep_ifirstlastkey_with_null here (instead of _get_keep_ifirstlastkey) because this is a
                # left outer join and we need to preserve an entry for null/invalid/NA (and a normal ikey does
                # not include an entry for the invalid/NA group).
                return _get_keep_ifirstlastkey_with_null(left_keygroup, keep_left)

    # For each gbkey value in 'right', get that value's multiplicity within 'right' if the value
    # also exists in left. For any gbkeys in 'left' which don't exist in 'right', store:
    #   * one (1) if performing a left join; this causes the rows in 'left' with that gbkey
    #     to be copied over 1-to-1 in the merged Dataset. The columns from 'right' will contain
    #     an invalid/null value for these elements in the merged Dataset.
    #   * zero (0) if performing an inner join; this causes the rows in 'left' with that gbkey
    #     to be eliminated so they don't appear in the merged Dataset.
    # The ncountgroup[1:] is to elide the count for the invalid bucket, since invalids aren't included
    # in the gbkeys (so they're zero-indexed, while ncountgroup is 1-indexed, with the 0 index being the
    # number of invalids in the original columns that Grouping was created from).
    # N.B. We offset the arrays on both the L.H.S. and R.H.S. here by one (1) because the indices in `right_keyid_to_left_keyid_map`
    #      are zero-indexed keyids (which don't consider the invalid bin); the right_key_multiplicity array is, however,
    #      aligned to the .ncountgroup from the left group which _does_ include the invalid count in the 0th bin.
    # TODO: Revisit to optimize dtype here (for memory usage and correctness); or, just use .ncountgroup.dtype and fix
    #       Grouping.ncountgroup to use the most compact dtype possible.
    # PERF: Consider writing a numba loop to fuse the next couple of lines together to avoid the intermediate array
    #       creation caused by the fancy-indexing.
    nonmatched_row_group_count = 0 if is_inner_join else 1
    right_group_multiplicity_as_left = full(left_keygroup.ncountgroup.shape, nonmatched_row_group_count, dtype=np.int64)
    right_group_multiplicity_as_left[1:][right_keyid_to_left_keyid_map[right_key_exists_in_left]] =\
        right_keygroup.ncountgroup[1:][right_key_exists_in_left]

    # Each row in 'left' (the original Dataset) having gbkey value K needs to be repeated in-place M times,
    # where M is the multiplicity of that row's key within the *right* Grouping. If the key K from the left Grouping
    # doesn't exist in the right Grouping, those rows get a repeat value of '1' (so they're kept as-is).
    left_ikey = _get_keep_ikey(left_keygroup, keep_left)
    left_repeats = right_group_multiplicity_as_left[left_ikey]
    assert all(isnotnan(left_repeats)), "Invalid entries found in 'left_repeats'."
    #assert not hasanynan(left_repeats), "Invalid entries found in 'left_repeats'."
    assert min(left_repeats) >= 0, "Negative entries found in 'left_repeats'."

    # EXPERIMENTAL :: For support of outer merge with keep='first'/'last' for left side.
    #left_ikey_with_null = _get_keep_ikey_with_null(left_keygroup, keep_left)
    #left_repeats_with_null = right_group_multiplicity_as_left[left_ikey_with_null]

    # Determine the total number of rows (as a scalar) -- we use this to determine the most compact dtype
    # we can use for the cumulative row count array.
    total_merged_rowcount = sum(left_repeats, dtype=np.int64)

    # N.B. We need to add one to the total rowcount here (just for the purposes of calling np.min_scalar_type).
    #      np.min_scalar_type does not understand riptable sentinel/NA values and it prefers to return an unsigned int
    #      dtype. If the total_merge_rowcount ends up being the riptable sentinel value for that dtype,
    #      the correctness assertions below break; adding one to the count fixes this by forcing the use of the next
    #      largest dtype (but only in that specific case).
    # TODO: Implement an rt.min_scalar_type() function which automatically handles this.
    left_repeats_cumsum_dtype = np.min_scalar_type(total_merged_rowcount + 1)

    # Perform a cumsum on the 'left_repeats' array, so we know for each element which index
    # (in the resulting array) to start repeating at.
    left_repeats_cumsum = cumsum(left_repeats, dtype=left_repeats_cumsum_dtype)
    assert np.all(isnotnan(left_repeats_cumsum)), "Invalid entries found in 'left_repeats_cumsum'."
    # assert not hasanynan(left_repeats_cumsum), "Invalid entries found in 'left_repeats_cumsum'."
    assert np.min(left_repeats_cumsum) >= 0, "Negative entries found in 'left_repeats_cumsum'."

    # Allocate the array which'll hold our fancy index for the 'left' Dataset.
    # Derive the dtype to use from the length of the 'left' Dataset; that's because
    # the values in the fancy index array can't be any larger than that (since they're
    # element/row indices into the 'left' Dataset).
    left_fancyindex_dtype = int_dtype_from_len(len(left_keygroup.ikey))     # Yes, use the full, real ikey here -- it's a proxy for the number of rows in the left Dataset.
    left_fancyindex = empty(left_repeats_cumsum[-1], dtype=left_fancyindex_dtype) # TODO: Just replace with dtype=left_keygroup.ikey.dtype if we can trust this to be the most-compact possible dtype.

    # Create the fancy index for columns of the 'left' Dataset; when used to index into the columns,
    # the result will be the corresponding column for the joined/merged Dataset.
    # This is done a bit differently based on whether we're dropping duplicate rows (per key) or not.
    if keep_left is None:
        # N.B. This used to be implemented with the following line, but it was much too slow (since it used np.repeats).
        #       left_fancy_index = repeat(FastArray(arange(len(left_keygroup.ikey))), left_repeats)
        _fused_arange_repeat(left_repeats_cumsum, left_fancyindex)
    else:
        # Get the ifirstkey/ilastkey for the left keygroup.
        left_ikey = _get_keep_ifirstlastkey(left_keygroup, keep_left)

        # When dropping duplicates, we're only keeping some of the rows; so in contrast to the
        # case above where we're (potentially) keeping all of the rows from the left Dataset,
        # we need to pass in the left ifirstkey/ilastkey here since they contain the indices of
        # just the rows we want to keep.
        _partial_repeat(left_ikey, left_repeats_cumsum, left_fancyindex)

    # The left fancy index should never have any invalid entries for inner and left_outer joins,
    # since we utilize the left join logic to perform right joins as well (by swapping the Datasets).
    # It can (correctly) only have invalids/NA values for a full outer join.
    return left_fancyindex


class JoinIndices(NamedTuple):
    """
    Holds fancy/logical indices into the left and right Datasets constructed by the join implementation,
    along with other relevant data needed to construct the resulting merged Dataset.
    """

    left_index: Optional[FastArray]
    """
    Integer fancy index or boolean mask for selecting data from the columns of the left Dataset
    to create the columns of the merged Dataset. This index is optional; when None, indicates that
    the columns from the left Dataset can be used directly in the resulting merged Dataset without
    needing to be filtered or otherwise transformed.
    """

    right_index: Optional[FastArray]
    """
    Integer fancy index or boolean mask for selecting data from the columns of the right Dataset
    to create the columns of the merged Dataset. This index is optional; when None, indicates that
    the columns from the right Dataset can be used directly in the resulting merged Dataset without
    needing to be filtered or otherwise transformed.
    """

    right_only_rowcount: Optional[int] = None
    """
    Only populated for outer merges. Indices the number of rows in the right Dataset whose keys
    do not occur in the left Dataset. This value can be used to slice `right_fancyindex` to get
    just the part at the end which represents these "right-only rows".
    """

    @staticmethod
    def result_rowcount(index_arr: Optional[np.ndarray], dset_rowcount: int) -> int:
        """
        Calculate the number of rows resulting from indexing into a `Dataset` with a fancy/logical index.

        Parameters
        ----------
        index_arr : np.ndarray, optional
            A fancy index or boolean mask array to be used to select rows from a `Dataset`.
        dset_rowcount : int
            The number of rows in the `Dataset` that `index_arr` will be applied to.

        Returns
        -------
        int
            The number of rows resulting from indexing into a `Dataset` with a fancy/logical index.
            Guaranteed to be non-negative.
        """
        if index_arr is None:
            return dset_rowcount
        elif index_arr.dtype == bool:
            return sum(index_arr)
        else:
            return len(index_arr)


def _create_merge_fancy_indices(
    left_keygroup: 'Grouping',
    right_keygroup: 'Grouping',
    right_groupby_keygroup: Optional['Grouping'],
    how: str,
    keep: Tuple[Optional[str], Optional[str]]
) -> JoinIndices:
    """
    Create two fancy indices -- for the left and right Datasets, respectively -- to be used to index the columns
    of those Datasets to produce the joined result Dataset.

    Parameters
    ----------
    left_keygroup : Grouping
        A `Grouping` instance representing the key(s) from the 'left' Dataset.
    right_keygroup : Grouping
        A `Grouping` instance representing the key(s) from the 'right' Dataset.
    right_groupby_keygroup : Grouping, optional
        A `Grouping` instance representing the key(s) from the 'right' Dataset;
        `right_keygroup` uses SQL "JOIN" semantics for determining valid (non-null/NA) tuples,
        but this instance uses SQL "GROUP BY" semantics instead, which is required to correctly
        implement the `keep` keyword for the 'right' Dataset in an outer merge.
        Only specify this for multi-key, 'outer' merges, where `keep` for the 'right' Dataset is not None;
        otherwise pass None.
    how : {'left','right','inner','outer'}
        Selects the type of join to perform.
    keep : 2-tuple of ({'first', 'last'}, optional)
        A 2-tuple of optional strings specifying that only the first or last occurrence
        of each unique key within `left_keygroup` and `right_keygroup` should be kept.
        In other words, resolves multiple occurrences of keys (multiplicity > 1) to a single occurrence.

    Returns
    -------
    JoinIndices
        A `JoinIndices` instance containing the results from the join algorithm.

    Notes
    -----
    The `right_groupby_keygroup` argument is required when performing an outer join (SQL: ``FULL OUTER JOIN``)
    with multiple key columns *and* the `keep` keyword is specified for the 'right' Dataset.
    `left_keygroup` and `right_keygroup` are created following the SQL ``JOIN`` semantics, where a tuple (row key)
    containing _any_ ``NULL``/NA values is considered to be ``NULL``/NA. However, the ``DISTINCT`` operator in
    SQL / relational algebra follows the semantics of SQL ``GROUP BY``, where a tuple (row key) is only considered
    ``NULL`` when _all_ values in the tuple are ``NULL``. An outer merge will include all rows with ``NULL`` keys
    from the 'right' Dataset in the results; when "keep" is not specified, finding those rows to include them in the
    merge results is straightforward using the ``JOIN`` `Grouping` (`right_keygroup`). However, when `keep` is
    specified for the 'right' Dataset, the "keep" / "drop duplicates" operation needs to choose the first/last row
    per tuple (row key) in the 'right' Dataset; that requires the unique keys to be determined according to
    SQL ``GROUP BY`` rather than ``JOIN`` semantics -- which is what `right_groupby_keygroup` provides.
    """
    def gbkeys_extract(g : 'Grouping') -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Extract unique groupby-keys values from a Grouping object for passing to ismember.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
        """
        # TODO: Use g.unique_dict here instead of g.gbkeys?
        gbkeys = g.gbkeys.values()
        if len(gbkeys) == 1:
            return next(iter(gbkeys))
        else:
            return tuple(gbkeys)

    def left_or_inner_join(left_keygroup: 'Grouping', right_keygroup: 'Grouping', is_inner_join: bool, keep: Tuple[Optional[str], Optional[str]]) -> JoinIndices:
        # TODO: Replace the next few lines here with a call to `Grouping.ismember`, it's implementing the same operation.
        #       However, Grouping.ismember doesn't currently support the values being passed in as another Grouping
        #       -- that would need to be implemented first in order to get the same result here.

        # Get the unique values from the Groupings.
        left_grouping_gbkey = gbkeys_extract(left_keygroup)
        right_grouping_gbkey = gbkeys_extract(right_keygroup)

        # Which keys in 'left' are also in 'right'?
        # N.B. we do this operation on the gbkeys of each grouping; the number of uniques in each
        # key will be smaller-than-or-equal to the length of the original data column, and in most
        # cases will be significantly smaller (so using ismember on the gbkeys will be significantly faster).
        # TODO: Pass 'hint_size' here to speed up the 'ismember' call?
        right_key_exists_in_left, right_keyid_to_left_keyid_map = ismember(right_grouping_gbkey, left_grouping_gbkey)

        # Determine which keys in 'right' are also in 'left'.
        # We can compute this relatively inexpensively from the boolean mask we created earlier
        # which told us which keys from 'left' are also in 'right'.
        # N.B. The left->right keyid map we produce here must be identical to what we'd get from
        #      calling ismember(left_grouping_gbkey, right_grouping_gbkey); the only reason we're
        #      not just calling that again is that we can just use fancy-indexing operations on the
        #      result of the first ismember call (plus the fact we know all the keys in gbkeys are unique)
        #      to create the same array without having to re-run the hashing/sorting code.
        # PERF: It seems like there should be a better, faster way to implement this fancy-indexing operation;
        #       if there's not (in the riptable / numpy API), consider moving this to a numba or C++ function.
        left_keyid_to_right_keyid_dtype: np.dtype = int_dtype_from_len(len(right_keygroup.ncountgroup))
        left_keyid_to_right_keyid_map = full(left_keygroup.ncountgroup.shape, INVALID_DICT[left_keyid_to_right_keyid_dtype.num], dtype=left_keyid_to_right_keyid_dtype)
        left_keyid_to_right_keyid_map[right_keyid_to_left_keyid_map[right_key_exists_in_left]] = bool_to_fancy(right_key_exists_in_left)

        # Create the fancy index for the left Dataset.
        left_fancy_index = _build_left_fancyindex(right_keyid_to_left_keyid_map, right_key_exists_in_left, left_keygroup, right_keygroup, is_inner_join, keep)

        # Use the information we've already computed + what's in the Grouping instances
        # to construct the fancy-index for the right Dataset.
        # We use a numba function for this because although the index construction is relatively
        # straightforward, there's no straightforward and/or efficient way to express it with the
        # numpy API.
        right_fancy_index = _build_right_fancyindex(left_keyid_to_right_keyid_map, left_keygroup, right_keygroup, is_inner_join, keep)

        # Return the constructed fancy indices for the left and right Datasets.
        return JoinIndices(left_fancy_index, right_fancy_index)

    def outer_join(
        left_keygroup: 'Grouping',
        right_keygroup: 'Grouping',
        right_groupby_keygroup: Optional['Grouping'],
        keep: Tuple[Optional[str], Optional[str]]
    ) -> JoinIndices:
        # TODO: The core logic in this function is the same as for the left_or_inner_join() function above.
        #       Once we have tests in place for exercising the logic and all code paths within this function
        #       (particularly around extending the left/right fancy indices created by the left join), this logic
        #       should be streamlined and merged into the left_or_inner_join() function -- probably at the same time
        #       we inline the "extension" logic into the _build_left_fancyindex/_build_right_fancyindex() functions.

        # Get the unique values from the Groupings.
        # The algorithm below assumes that the unique values here are exactly the set of
        # those values used within the data (i.e. the data here doesn't contain any _unused_ values).
        left_grouping_gbkey = gbkeys_extract(left_keygroup)
        right_grouping_gbkey = gbkeys_extract(right_keygroup)

        # Which keys in 'left' are also in 'right'?
        # N.B. we do this operation on the gbkeys of each grouping; the number of uniques in each
        # key will be smaller-than-or-equal to the length of the original data column, and in most
        # cases will be significantly smaller (so using ismember on the gbkeys will be significantly faster).
        # TODO: Pass 'hint_size' here to speed up the 'ismember' call?
        right_key_exists_in_left, right_keyid_to_left_keyid_map = ismember(right_grouping_gbkey, left_grouping_gbkey)

        # Determine which keys in 'right' are also in 'left'.
        # We can compute this relatively inexpensively from the boolean mask we created earlier
        # which told us which keys from 'left' are also in 'right'.
        # N.B. The left->right keyid map we produce here must be identical to what we'd get from
        #      calling ismember(left_grouping_gbkey, right_grouping_gbkey); the only reason we're
        #      not just calling that again is that we can just use fancy-indexing operations on the
        #      result of the first ismember call (plus the fact we know all the keys in gbkeys are unique)
        #      to create the same array without having to re-run the hashing/sorting code.
        # PERF: It seems like there should be a better, faster way to implement this fancy-indexing operation;
        #       if there's not (in the riptable / numpy API), consider moving this to a numba or C++ function.
        left_keyid_to_right_keyid_dtype: np.dtype = int_dtype_from_len(len(right_keygroup.ncountgroup))
        left_keyid_to_right_keyid_map = full(left_keygroup.ncountgroup.shape, INVALID_DICT[left_keyid_to_right_keyid_dtype.num], dtype=left_keyid_to_right_keyid_dtype)
        left_keyid_to_right_keyid_map[right_keyid_to_left_keyid_map[right_key_exists_in_left]] = bool_to_fancy(right_key_exists_in_left)

        # Get the rows from 'right' which have keys that aren't in 'left'.
        right_only_group_mask = empty(right_keygroup.ncountgroup.shape, dtype=bool)
        # Include the invalid/NA group from 'right', per definition of a full outer join.
        right_only_group_mask[0] = right_keygroup.ncountgroup[0] > 0
        right_only_group_mask[1:] = ~right_key_exists_in_left

        if keep[1] is None:
            right_only_group_rows = Grouping.extract_groups(
                right_only_group_mask, right_keygroup.igroup, right_keygroup.ncountgroup, ifirstgroup=right_keygroup.ifirstgroup)
        else:
            if right_groupby_keygroup is not None and right_keygroup.ncountgroup[0] > 0:
                # Find the rows in 'right' which have the null JOIN key.
                join_invalid_rows = right_keygroup.igroup[:right_keygroup.ncountgroup[0]]

                # What GROUP BY groups do each of these rows belong to?
                join_invalid_groupby_groups = right_groupby_keygroup.ikey[join_invalid_rows]

                # Keep only the unique GROUP BY groups.
                join_invalid_groupby_groups = unique(join_invalid_groupby_groups, sorted=False)

                # Get the first/last row for each group.
                keep_ifirstlastkey = _get_keep_ifirstlastkey_with_null(right_groupby_keygroup, keep[1])
                join_invalid_groupby_rows = keep_ifirstlastkey[join_invalid_groupby_groups]

                # Get the other right-only rows (those with valid join keys).
                ifirstlastkey_mask = right_only_group_mask[1:]
                keep_ifirstlastkey = _get_keep_ifirstlastkey(right_keygroup, keep[1])
                join_valid_groupby_rows = keep_ifirstlastkey[ifirstlastkey_mask]
                right_only_group_rows = hstack([join_invalid_groupby_rows, join_valid_groupby_rows])

            else:
                # _get_keep_ifirstlastkey_with_null returns an array which will include an entry for the
                # NA/invalid group if there are any rows assigned to that group; otherwise, it does not
                # include an entry for it -- so we need to slice the mask indicating the right-only groups
                # to account for that before using it to index the first/last key array.
                ifirstlastkey_mask = right_only_group_mask if right_only_group_mask[0] else right_only_group_mask[1:]
                keep_ifirstlastkey = _get_keep_ifirstlastkey_with_null(right_keygroup, keep[1])
                right_only_group_rows = keep_ifirstlastkey[ifirstlastkey_mask]

        # The rows (row indices) with right-only groups should always be in sorted
        # order to preserve the original ordering of the data. The row indices extracted
        # from the group information may be out of order, and if so, we need to sort the
        # array (in-place) to preserve the invariant for the ordering of the data.
        if not issorted(right_only_group_rows):
            right_only_group_rows.sort()

        # Create the fancy index for the left Dataset.
        left_fancy_index = _build_left_fancyindex(right_keyid_to_left_keyid_map, right_key_exists_in_left, left_keygroup, right_keygroup, False, keep)

        # Use the information we've already computed + what's in the Grouping instances
        # to construct the fancy-index for the right Dataset.
        # We use a numba function for this because although the index construction is relatively
        # straightforward, there's no straightforward and/or efficient way to express it with the
        # numpy API.
        right_fancy_index = _build_right_fancyindex(left_keyid_to_right_keyid_map, left_keygroup, right_keygroup, False, keep)

        # Only perform the additional post-processing on the fancy indices if needed -- when there's at least one
        # row with a right-only key. Otherwise skip it to avoid doing unnecessary work.
        if right_only_group_rows.size > 0:
            # Fill in the additional entries, if needed, in the fancy indices for the rows with right-only keys.
            # Make sure to account for the possibilities that the fancy indices could be e.g. None or a boolean mask,
            # we'll need to handle those specially.
            # PERF: Some or all of this logic can be moved into the _build_left_fancyindex() and _build_right_fancyindex()
            #       functions, which will allow most/all of the additional array allocations and data copying to be elided.
            if left_fancy_index is None:
                # Need to get every row of 'left' in order, but then also account for the rows of 'right'
                # which don't exist in 'left' whose keys don't exist in left. We get this by calling arange
                # with a value that covers the total number of rows, then overwriting the entries representing
                # the right-only rows with the invalid value.
                left_rowcount = len(left_keygroup.ikey)
                left_fancy_index = arange(left_rowcount + len(right_only_group_rows), dtype=left_keygroup.ikey.dtype)
                left_fancy_index[left_rowcount:] = left_fancy_index.inv

            else:
                if left_fancy_index.dtype.char == '?':
                    # The boolean mask must be converted to a fancy index, because a boolean mask must match
                    # the length of the array it's used to index into -- attempting to extend it here would make it
                    # incompatible for use with the 'left' Dataset.
                    # PERF: This could be more efficient if we had e.g. an argument in bool_to_fancy that would tell it
                    #       to allocate some additional, unused elements at the end of the array which we'd fill in later
                    #       without having to allocate yet another array and copy the data over to it.
                    left_fancy_index = bool_to_fancy(left_fancy_index)

                extended_left_rowcount = len(left_fancy_index) + len(right_only_group_rows)

                # Create an extended version of the fancy index for 'left'; for the new elements representing
                # rows with right-only keys, write invalid/NA values into them since they don't match any left rows.
                extended_left_fancy_index = empty(extended_left_rowcount, dtype=left_keygroup.ikey.dtype)
                extended_left_fancy_index[:len(left_fancy_index)] = left_fancy_index
                extended_left_fancy_index[len(left_fancy_index):] = extended_left_fancy_index.inv
                left_fancy_index = extended_left_fancy_index

            if right_fancy_index is None:
                # Similar to the approach for 'left' above, except w.r.t. how we fill in the elements
                # representing rows with right-only keys.
                right_rowcount = len(right_keygroup.ikey)
                right_fancy_index = arange(right_rowcount + len(right_only_group_rows))
                right_fancy_index[right_rowcount:] = right_only_group_rows

            else:
                if right_fancy_index.dtype.char == '?':
                    right_fancy_index = bool_to_fancy(right_fancy_index)

                extended_right_rowcount = len(right_fancy_index) + len(right_only_group_rows)

                # Create an extended version of the fancy index for 'right'; for the new elements representing
                # rows with right-only keys, we write in the indices of those rows (which we extracted earlier).
                extended_right_fancy_index = empty(extended_right_rowcount, dtype=right_keygroup.ikey.dtype)
                extended_right_fancy_index[:len(right_fancy_index)] = right_fancy_index
                extended_right_fancy_index[len(right_fancy_index):] = right_only_group_rows
                right_fancy_index = extended_right_fancy_index

        # Return the constructed fancy indices for the left and right Datasets.
        return JoinIndices(left_fancy_index, right_fancy_index, len(right_only_group_rows))


    # Calculate based on the join type.
    if how == 'left':
        # For left and right joins, we only use the keys from the left or right key, respectively.
        # All we need to do here is calculate the multiplicity for each value in the key column.
        return left_or_inner_join(left_keygroup, right_keygroup, False, keep)

    elif how == 'right':
        # Right join is just left join with the order of the arguments swapped.
        # We must also swap the order of the 'keep' tuple to match.
        keep = keep[1], keep[0]
        join_indices = left_or_inner_join(
            right_keygroup, left_keygroup, False, keep)
        return JoinIndices(join_indices.right_index, join_indices.left_index, None)

    elif how == 'inner':
        # Inner join computes the intersection of the left and right keysets and keeps only rows
        # whose key belongs to the intersection set.
        return left_or_inner_join(left_keygroup, right_keygroup, True, keep)

    elif how == 'outer':
        return outer_join(left_keygroup, right_keygroup, right_groupby_keygroup, keep)

    else:
        raise ValueError(f"Unsupported value ({how}) specified for the 'how' parameter.")


def _normalize_keep(keep: Optional[Union[str, Tuple[Optional[str], Optional[str]]]]) -> Tuple[Optional[str], Optional[str]]:
    """Convert a value passed for the 'keep' parameter to a normalized form."""
    if keep is None:
        return (None, None)
    elif isinstance(keep, (str, bytes)):
        if keep == 'first' or keep == 'last':
            return (keep, keep)
        else:
            raise ValueError(f"Unsupported value '{keep}' passed for the `keep` parameter.")
    elif isinstance(keep, tuple):
        if len(keep) == 2:
            return keep
        else:
            raise ValueError(f"Invalid tuple (length={len(keep)}) passed for the `keep` parameter.")
    else:
        raise ValueError(f"Invalid argument value passed for the `keep` parameter.")


def _extract_on_columns(
    on: Optional[Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]]],
    side_on: Optional[Union[str, List[str]]],
    for_left: bool,
    on_param_name: str,
    is_optional: bool
) -> List[str]:
    """
    Extract a list of column name(s) from either the 'on' or 'left_on'/'right_on' arguments
    as given to the 'merge' functions. (For the merge_asof function, this is the for the 'by'
    and 'left_by'/'right_by' arguments.)

    Parameters
    ----------
    on : str or (str, str) or list of str / (str, str), optional
        A string representing a column name in both Datasets to serve as a merge key; or,
        a 2-tuple of strings which are column names in the left and right Datasets, respectively,
        to be used together as a merge key; or, a list of strings and/or tuples.
    side_on : str or list of str, optional
    for_left : bool
    on_param_name : str
        The name of the parameter whose value is passed through the `on` parameter to this function.
        This string should typically be 'by' or 'on'.
    is_optional : bool
        Indicates whether the parameter is completely optional.
        When False, an exception is thrown if both `on` and `side_on` are None.

    Returns
    -------
    list of str
        Flat list of 'on' column names for the given side (left or right Dataset).

    Notes
    -----
    TODO: Implement a function that extracts the columns for the left or right datasets
          from the new-style 'on' parameter for _merge2, where 'on' can be a string, 2-tuple,
          or list of str and/or 2-tuples.
    """
    if side_on is None:
        if on is None:
            # Both 'on' and 'side_on' are missing. If this parameter is optional,
            # we just return an empty list; otherwise, this parameter is
            # required -- we cannot proceed so raise an exception.
            if is_optional:
                return []
            else:
                side_name = 'left' if for_left else 'right'
                raise ValueError(f"`{on_param_name}` and `{side_name}_{on_param_name}` cannot both be None.")
        else:
            # Parse the 'on' argument to extract a list of column names for this side.
            if isinstance(on, (bytes, str)):
                side_on = on

            elif isinstance(on, tuple):
                arity = len(on)
                if arity == 1:
                    side_on = on[0]
                elif arity == 2:
                    side_idx = 0 if for_left else 1
                    side_on = on[side_idx]
                else:
                    raise ValueError(f'Unsupported tuple of length {arity} provided for the `{on_param_name}` argument. Tuples must contain either one or two elements.')

            elif isinstance(on, list):
                side_on = []
                for idx, val in enumerate(on):
                    if isinstance(val, (bytes, str)):
                        side_on.append(val)
                    elif isinstance(val, tuple):
                        arity = len(val)
                        if arity == 1:
                            side_on.append(val[0])
                        elif arity == 2:
                            side_idx = 0 if for_left else 1
                            side_on.append(val[side_idx])
                        else:
                            raise ValueError(f'Unsupported tuple of length {arity} found at index {idx} of the list provided for the `{on_param_name}` argument. Tuples must contain either one or two elements.')
                    else:
                        raise ValueError(f'Unsupported argument type {type(val)} found at index {idx} of the list provided for the `{on_param_name}` argument.')

            else:
                raise ValueError(f'Unsupported argument type {type(on)} provided for the `{on_param_name}` argument.')
    else:
        # 'on' and 'left_on'/'right_on' are not allowed to be specified together,
        # as it's unclear which one should take precedence.
        # If we want to define which one does take precedence, we could drop this down to
        # a Warning to let the user know part of what they've specified will be ignored.
        if on is not None:
            side_name = 'left' if for_left else 'right'
            raise ValueError(f"The `{on_param_name}` and `{side_name}_{on_param_name}` parameters cannot be specified together; exactly one of them should be specified.")

    #
    # TODO: Handle case where 'side_on' is an empty string -- raise an Error
    #

    # Normalize 'side_on' -- if it's just a string/bytestring,
    # wrap it in a list so consumers can always just assume they'll
    # have a list of the arguments to work with.
    if isinstance(side_on, (bytes, str)):
        side_on = [side_on]

    # Before returning, verify the list has at least one entry.
    if side_on:
        return side_on
    else:
        side_name = 'left' if for_left else 'right'
        raise ValueError(f"Unable to extract column names for {side_name} Dataset from either `{on_param_name}` or `{side_name}_{on_param_name}`.")


def _validate_groupings(left_keygroup: 'Grouping', right_keygroup: 'Grouping', validate: str, keep: Tuple[Optional[str], Optional[str]]):
    """
    Perform the validation specified by the 'validate' parameter (for merge).

    Parameters
    ----------
    left_keygroup : Grouping
    right_keygroup : Grouping
    validate : {'one_to_one', '1:1', 'one_to_many', '1:m', 'many_to_one', 'm:1', 'many_to_many', 'm:m'}, optional
        Validate the uniqueness of the values in the columns specified by the `on`, `left_on`, `right_on`
        parameters. In other words, allows the _multiplicity_ of the keys to be checked so the user
        can prevent the merge if they want to ensure the uniqueness of the keys in one or both of the Datasets
        being merged.
    keep : 2-tuple of ({'first', 'last'}, optional)
        An optional string which specifies that only the first or last occurrence
        of each unique key within `left` and `right` should be kept. In other words,
        resolves multiple occurrences of keys (multiplicity > 1) to a single occurrence.
    """
    # Parse the 'keep' parameter into a (bool, bool) so we know if we're going to be dropping
    # duplicate keys in the left or right (in which case we can skip validating that side).
    keep_left, keep_right = keep
    drop_dups_left, drop_dups_right = (keep_left is not None), (keep_right is not None)

    if validate == '1:1' or validate == 'one_to_one':
        # 1-to-1

        # Only validate the key multiplicities when the user has *not* specified
        # they want to drop duplicate keys.
        require_unique_left = not drop_dups_left
        require_unique_right = not drop_dups_right

    elif validate == '1:m' or validate == 'one_to_many':
        # 1-to-many

        # Only validate the key multiplicities when the user has *not* specified
        # they want to drop duplicate keys.
        require_unique_left = not drop_dups_left
        require_unique_right = False

    elif validate == 'm:1' or validate == 'many_to_one':
        # many-to-1

        # Only validate the key multiplicities when the user has *not* specified
        # they want to drop duplicate keys.
        require_unique_left = False
        require_unique_right = not drop_dups_right

    elif validate == 'm:m' or validate == 'many_to_many':
        # many-to-many

        # At present, we don't perform any validation for this step.
        require_unique_left = False
        require_unique_right = False

    else:
        raise ValueError(f"Unsupported value '{validate}' specified for the `validate` parameter.")


    def _perform_validation(grouping:'Grouping', name:str):
        # N.B. It's important we use Grouping.ncountkey instead of .ncountgroup here;
        #      invalid/null values shouldn't be considered since they're always dropped.
        max_key_multiplicity = grouping.ncountkey.max()

        if max_key_multiplicity > 1:
            # TODO: Can we provide a better error message here, e.g. to provide the user with some
            #       examples of the keys with multiplicity > 1? Could do something like argmax() on Grouping.ncountkey,
            #       then use that index to fetch the gbkey value(s) for that index from the Grouping.gbkeys;
            #       based on the keyid, we could also get the Grouping.igroupfirst which'll tell us the first index
            #       (within the keycol(s)) where the key occurs.
            raise ValueError(f"Validation ({validate}) failed. The {name} Dataset has one or more keys which occur more than once.")


    if require_unique_left:
        _perform_validation(left_keygroup, 'left')

    if require_unique_right:
        _perform_validation(right_keygroup, 'right')


def _create_column_valid_mask(arr: FastArray) -> Optional[FastArray]:
    """
    Create a boolean mask from the given array indicating which of it's elements are valid.

    This is used to work around current issues related to how `Grouping` and `ismember` handle
    NaN/invalid; the mask created by this function can be passed in when creating a `Grouping`
    instance for `arr`.
    """
    # FIXME #1: rt.isnan/rt.isnotnan currently do not work as expected for Categoricals --
    # they'll always return an array of all False/True, respectively. Until they can be fixed to work
    # as expected, implement the equivalent operation by looking at a Categorical's underlying numpy
    # array and finding any entries containing the index for the invalid bin.
    if isinstance(arr, Categorical):
        # TODO: What's the correct way to handle the case of a base_index=0 Categorical?
        if arr.base_index == 0:
            raise NotImplementedError("This function does not yet support Categoricals with base_index=0.")
        else:
            # Assume this is a Categorical with base_index == 1, so a 0 in the underlying array indicates the invalid bin.
            return arr._np != 0

    else:
        # FIXME #2: rt.isnan/rt.isnotnan currently fail when invoked with a string array.
        # Until they can be fixed to return an array of all False/True or an array scalar,
        # check for them here and create the Grouping without the filter.
        return None if arr.dtype.char in 'US' else isnotnan(arr)


def _normalize_selected_columns(ds: 'Dataset', column_names) -> Collection[str]:
    if column_names is None:
        return ds.keys()
    elif column_names == '':
        return []
    elif isinstance(column_names, list):
        return column_names
    else:
        # Assume `column_names` is a str
        return [column_names]


def _require_columns_present(keyset: Set[str], dataset_name:str, param_name:str, col_names: Collection[str]) -> None:
    """Verify that column(s) specified in 'left_on', 'right_on', 'columns_left' and 'columns_right' are all present in their respective Datasets."""
    missing_cols: Optional[List[str]] = None
    for col_name in col_names:
        if col_name not in keyset:
            if missing_cols is None:
                missing_cols = [col_name]
            else:
                missing_cols.append(col_name)

    # If there were any missing columns, raise a ValueError whose message contains *all* of the
    # missing column names -- this provides more diagnostic info to the user compared to just
    # reporting the first missing column we found.
    if missing_cols is not None:
        joined_colnames = ", ".join(map(lambda x: f'\'{x}\'', missing_cols))
        raise ValueError(f'The column(s) {joined_colnames} specified in the `{param_name}` argument are not present in the `{dataset_name}` Dataset.')


def _get_perf_hint(hint, index: int, _default=None):
    """
    Extracts a "performance hint" value -- specified as either a scalar or 2-tuple -- for
    either the left or right Dataset in a merge.

    Parameters
    ----------
    hint : scalar or 2-tuple of scalars, optional
    index : int
        Indicates whether the hint value is being extracted for the left or right Dataset.
        0 = left, 1 = right.
    _default : optional
        Optional default value, returned if `hint` is None.

    Returns
    -------
    Any
        The extracted performance hint value.
    """
    if hint is None:
        return _default
    elif isinstance(hint, tuple):
        return hint[index]
    else:
        return hint


def _get_or_create_keygroup(
    keycols: List[FastArray],
    index: int,
    force_invalids: bool,
    create_groupby_grouping: bool,
    high_card,
    hint_size
) -> Tuple['Grouping', Optional[FastArray]]:
    """
    Given the key columns (i.e. the columns being joined on) from a Dataset, get or create a `Grouping`
    object to be passed to the join algorithm.

    Parameters
    ----------
    keycols : list of FastArray
    index : int
        Indicates whether the `Grouping` being constructed is for the left or right Dataset.
        0 = left, 1 = right. Used to parse performance hints such as `high_card` or `hint_size`.
    force_invalids : bool
        When specified, builds a mask indicating which values are 'valid' and uses it
        to force _invalid_ values into the invalid (0th) group in the constructed
        Grouping instance.
    create_groupby_grouping : bool
        When True and `keycols` contains more than one (1) column, causes this function to create
        and return a second `Grouping` which follows SQL "GROUP BY" semantics when determining
        which tuples are considered null/NA.
    high_card
        Optional hint indicating the key data has high cardinality (many unique values).
    hint_size
        Optional hint at the number of unique key values.

    Returns
    -------
    join_keygroup : Grouping
        A Grouping object constructed from the columns specified by `keycols`.
    groupby_keygroup : FastArray, optional
        A `Grouping` object constructed from the columns specified by `keycols`,
        following the semantics of SQL "GROUP BY" w.r.t. which tuples are considered non-null/NA.
        This is optional and only returned when requested through `create_outer_merge_mask` *and*
        it is necessary; otherwise, None is returned.

    Notes
    -----
    The `force_invalids` parameter is important for ensuring the join algorithm adheres to ANSI SQL semantics.
    The current (as of 2020-07-16) implementation for creating Grouping objects does not recognize invalid/NA values
    in some (all?) cases; if we used a `Grouping` object created that way, invalid/NA values would be matched
    against one another (during the join), which goes against how ANSI SQL semantics say NULL entries should be
    handled during a join. When `force_invalids` is specified, this function pre-screens the data in the key columns
    to locate invalid/NA values, builds a boolean mask of the *valid* values, and passes that mask when creating the
    `Grouping` object, effectively forcing `Grouping` to recognize the invalid/NA values as invalid/NA values and
    ensuring we'll get the correct (ANSI SQL) semantics from our join implementation.

    If `Grouping` is ever fixed to recognize invalid/NA values at construction time, this function can be simplified
    to simply parse the performance hints then invoke the `Grouping` creation code.

    In the special case of exactly one Categorical being provided as the key, we need to detect whether there
    are any unused categories in the Categorical (specifically, it's embedded Grouping instance); the presence
    of these unused categories break the join algorithm since it assumes the unique values it extracts from the
    Grouping are exactly the set which appear in the data. If unused categories are detected, the
    `Grouping.regroup()` method is called to consolidate the grouping data and remove the extra categories.

    Idea: for the multi-key (i.e. ``len(keycols) > 1``) and single-key non-`Categorical` cases, could we create the
    `Grouping` object then create a mask on the Grouping's keys (and return them both for use by the join algorithm)
    rather than building a mask that's the length of the whole key column(s)? That might save some time here but
    may also slow down the join algorithm if it requires more-complex fancy/mask-indexing operations. This could
    also simplify/streamline the way we need to create the "GROUP BY"-style validity mask for full outer joins.
    """
    # TODO If given a tuple of Categoricals, does Categorical.__init__() optimize the creation from them?
    #      If so -- let's check if all items in `keycols` are Categorical and if so, use this for better performance.
    #      Same with Grouping -- what if we have a key that's a tuple of a Categorical and a non-Categorical column?
    #      Can we find a way to optimize the creation of the combined Grouping, perhaps by creating a Grouping for the
    #      non-Categorical column and merging it with the .grouping from the Categorical rather than having to
    #      re-process all of the data all over again?

    # If any performance hints were specified, determine which values apply
    # for the key(s) of the given Dataset (specified by 'index'; 0=left, 1=right).
    _lex = _get_perf_hint(high_card, index, False)
    _hint_size = _get_perf_hint(hint_size, index, 0)

    # There's a simpler path for single-keys.
    if len(keycols) == 1:
        single_keycol = keycols[0]
        # If there's only one key and it's a Categorical, we can just use the existing Grouping object.
        if isinstance(single_keycol, Categorical):
            # TODO: Consider emitting a warning if any perf hints have been supplied (above), since in
            #       this case they're ignored (because we're using the existing Grouping instead of creating one).
            cat_grouping: 'Grouping' = single_keycol.grouping

            # If unused categories are present, call .regroup() to create a new Grouping with those
            # categories removed. (See the 'Notes' section in the docstring for details.)
            return (cat_grouping if all(cat_grouping.ncountkey > 0) else cat_grouping.regroup()), None

        else:
            # HACK: For non-Categorical single-key columns, we need to use rt.isnotnan to create
            #       a filter for the Grouping. This is due to rt.ismember not handling NaN/invalids
            #       in the expected way (i.e. where NaN != NaN).
            isvalid = _create_column_valid_mask(single_keycol) if force_invalids else None
            return TypeRegister.Grouping(keycols, lex=_lex, hint_size=_hint_size, filter=isvalid), None

    else:
        # HACK: For multiple keys, we apply the same approach as for single-key non-Categoricals where
        #       we use rt.isnotnan to get a mask indicating the value elements. We do this for all the
        #       columns then logical AND the masks together, since we want the behavior whereby a null in any
        #       one of the components (columns) of a composite key means the whole tuple is considered null.
        # PERF: We use the in-place version of logical AND here, so that no matter how many columns we have,
        #       we only ever allocate / reference a maximum of two mask arrays at any one time; this helps
        #       minimize peak memory utilization.
        # PERF: The base case / initial value for the mask is None so that if we have _only_ columns which
        #       don't have an invalid value -- such as strings -- we don't ever allocate a mask array.
        #       This saves memory both by not having a mask array and also within the `Grouping` __init__ function.

        valid_join_tuples_mask: Optional[FastArray] = None
        valid_groupby_tuples_mask: Optional[FastArray] = None
        if force_invalids:
            for keycol in keycols:
                isvalid = _create_column_valid_mask(keycol)

                # N.B. we elide a 3rd branch (for handling the 'isvalid is None' case) here
                #      because it'd be equivalent to logical-ANDing an array of all True elements,
                #      so it wouldn't have any effect.
                if isvalid is not None:
                    if valid_join_tuples_mask is None:
                        valid_join_tuples_mask = isvalid
                    else:
                        valid_join_tuples_mask &= isvalid

                # Only needed/applicable to outer merges.
                if create_groupby_grouping:
                    # If 'isvalid' is None, that's considered to be the same as a boolean array of the correct
                    # length with all True elements; since we logical-OR the masks together for the groupby-grouping,
                    # when 'isvalid' is None we set the valid groupby tuples mask to None.
                    if isvalid is None:
                        valid_groupby_tuples_mask = None
                    elif valid_groupby_tuples_mask is None:
                        # Need to copy 'isvalid' to avoid sharing the same instance with 'valid_join_tuples_mask',
                        # which'll lead to them incorrectly having the same data (since we're updating in-place).
                        valid_groupby_tuples_mask = isvalid.copy()
                    else:
                        valid_groupby_tuples_mask |= isvalid


        # Create the Grouping, passing in the mask we created so NaN/invalid values are handled
        # in the way we need them to be for merging.
        join_grouping = TypeRegister.Grouping(keycols, lex=_lex, hint_size=_hint_size, filter=valid_join_tuples_mask)
        groupby_grouping = TypeRegister.Grouping(keycols, lex=_lex, hint_size=_hint_size, filter=valid_groupby_tuples_mask) if create_groupby_grouping else None
        return join_grouping, groupby_grouping


def merge_indices(
    left: 'Dataset',
    right: 'Dataset',
    *,
    on: Optional[Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]]] = None,
    how: str = 'left',
    # TODO: Consider changing this to require_unique: Union[bool, Tuple[bool, bool]] -- the semantics would be clearer to users
    validate: Optional[str] = None,
    keep: Optional[Union[str, Tuple[Optional[str], Optional[str]]]] = None,
    high_card: Optional[Union[bool, Tuple[Optional[bool], Optional[bool]]]] = None,
    hint_size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    **kwargs
) -> JoinIndices:
    """
    Perform a join/merge of two `Dataset`s, returning the left/right indices created by the join engine.

    The returned indices can be used to index into the left and right `Dataset`s to construct a merged/joined `Dataset`.

    Parameters
    ----------
    left : Dataset
        Left Dataset
    right : Dataset
        Right Dataset
    on : str or (str, str) or list of str or list of (str, str), optional
        Column names to join on. Must be found in both `left` and `right`.
    how : {'left', 'right', 'inner', 'outer'}
        The type of merge to be performed.

        * left: use only keys from `left`, as in a SQL 'left join'. Preserves the ordering of keys.
        * right: use only keys from `right`, as in a SQL 'right join'. Preserves the ordering of keys.
        * inner: use intersection of keys from both Datasets, as in a SQL 'inner join'. Preserves the
          ordering of keys from `left`.
        * outer: use union of keys from both Datasets, as in a SQL 'full outer join'.
    validate : {'one_to_one', '1:1', 'one_to_many', '1:m', 'many_to_one', 'm:1', 'many_to_many', 'm:m'}, optional
        Validate the uniqueness of the values in the columns specified by the `on`, `left_on`, `right_on`
        parameters. In other words, allows the _multiplicity_ of the keys to be checked so the user
        can prevent the merge if they want to ensure the uniqueness of the keys in one or both of the Datasets
        being merged.
        Note: The `keep` parameter logically takes effect before `validate` when they're both specified.
    keep : {'first', 'last'} or (str, str), optional
        An optional string which specifies that only the first or last occurrence
        of each unique key within `left` and `right` should be kept. In other words,
        resolves multiple occurrences of keys (multiplicity > 1) to a single occurrence.
    high_card : bool or (bool, bool), optional
        Hint to low-level grouping implementation that the key(s) of `left` and/or `right`
        contain a high number of unique values (cardinality); the grouping logic *may* use
        this hint to select an algorithm that can provide better performance for such cases.
    hint_size : int or (int, int), optional
        An estimate of the number of unique keys used for the join. Used as a performance hint
        to the low-level grouping implementation.
        This hint is typically ignored when `high_card` is specified.

    Returns
    -------
    JoinIndices

    Examples
    --------
    >>> rt.merge_indices(ds_simple_1, ds_simple_2, on=('A', 'X'), how = 'inner')
    #   A      B   X       C
    -   -   ----   -   -----
    0   0   1.20   0    2.40
    1   1   3.10   1    6.20
    2   6   9.60   6   19.20
    <BLANKLINE>
    [3 rows x 4 columns] total bytes: 72.0 B

    Demonstrating a 'left' merge.

    >>> rt.merge_indices(ds_complex_1, ds_complex_2, on = ['A','B'], how = 'left')
    #   B    A       C       E
    -   -   --   -----   -----
    0   Q    0    2.40    1.50
    1   R    6    6.20   11.20
    2   S    9   19.20     nan
    3   T   11   25.90     nan
    <BLANKLINE>
    [4 rows x 4 columns] total bytes: 84.0 B

    See Also
    --------
    merge2
    """
    # Process keyword arguments.
    # NOTE: 'require_match' here isn't used yet; it is used in merge2, and the way it's used there is not as
    #       efficient as it could be. That check can be moved into the join algorithm where the gbkey<->gbkey
    #       mapping is created, at which point we'll pass in 'require_match' to specify when that check needs
    #       to be performed.
    require_match = bool(kwargs.pop("require_match")) if "require_match" in kwargs else False
    skip_column_presence_check =\
        bool(kwargs.pop("skip_column_presence_check")) if "skip_column_presence_check" in kwargs else False
    if kwargs:
        # There were remaining keyword args passed here which we don't understand.
        first_kwarg = next(iter(kwargs.keys()))
        raise ValueError(f"This function does not support the kwarg '{first_kwarg}'.")

    # Collect timing stats on how long various stages of the merge operation take.
    start = GetNanoTime()

    # Convert the 'keep' argument to a normalized form so it's easier to use later.
    # This also serves to validate the argument value.
    keep = _normalize_keep(keep)

    # TEMP: Disallow callers to pass a 'keep' value for the 'left' Dataset when performing
    #       an outer merge. There's a bug in the join algorithm which prevents this from working
    #       so it's better to give users an informative error message until it can be fixed.
    if how == 'outer' and keep[0] is not None:
        raise ValueError(
            "Specifying a 'keep' value for the left Dataset (or both) with an outer merge is "
            "temporarily not permitted due to a bug in the implementation.")

    # Validate and normalize the 'on' column lists for each Dataset.
    # TODO: Change this so the normalization step takes the 'on' parameter then returns a list of 2-tuples.
    #       We can consume that list just below here, and the normalized form of the list will allow that logic to be simplified.
    left_on = _extract_on_columns(on, None, True, 'on', is_optional=False)
    right_on = _extract_on_columns(on, None, False, 'on', is_optional=False)

    # If the column presence check is enabled (it is by default), check that all requested
    # 'on' columns are present. This check is disabled when calling from rt.merge2() since
    # the same check is already performed there.
    if not skip_column_presence_check:
        # PERF: Revisit this -- it could be made faster if ItemContainer.keys() returned a set-like object
        # such as KeysView instead of a list; then we wouldn't need to create the sets here.
        left_keyset = set(left.keys())
        _require_columns_present(left_keyset, 'left', 'left_on', left_on)
        right_keyset = set(right.keys())
        _require_columns_present(right_keyset, 'right', 'right_on', right_on)

    # Validate the pair(s) of columns from the left and right join keys have compatible types.
    left_on_arrs = [left[col_name] for col_name in left_on]
    right_on_arrs = [right[col_name] for col_name in right_on]
    key_compat_errs = _verify_join_keys_compat(left_on_arrs, right_on_arrs)
    if key_compat_errs:
        # If the list of errors is non-empty, we have some join-key compatibility issues.
        # Some of the "errors" may just be warnings; filter those out of the list and raise
        # them to notify the user.
        actual_errors : List[Exception] = []
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
            flat_errs = '\n'.join(map(str, actual_errors))  # N.B. this is because it's disallowed to use backslashes inside f-string curly braces
            raise ValueError(f"Found one or more compatibility errors with the specified 'on' keys:\n{flat_errs}")

    if logger.isEnabledFor(logging.DEBUG):
        delta = GetNanoTime() - start
        logger.debug("Validation complete.", extra={'elapsed_nanos': delta})

    # Construct the Grouping object for each of the join keys.
    start = GetNanoTime()

    # TODO: Should we always pass True for the 'force_invalids' argument below? The original intent was that we
    #       may want to be able to use the default (non-forced-invalids) behavior of Grouping for optional compatibility
    #       with rt.merge(), but that's no longer planned (users are just migrating to merge2/merge_lookup).
    #       Note, there *may* be something to passing False for the left and right grouping, when performing a
    #       left or right join, respectively -- before making the change to always pass True, verify all the
    #       optimized cases where we can return e.g. None for the fancy index aren't affected by the change.
    left_grouping, _ = _get_or_create_keygroup(left_on_arrs, 0, how != 'left', False, high_card, hint_size)
    right_grouping, right_groupby_grouping =\
        _get_or_create_keygroup(right_on_arrs, 1, how != 'right', how == 'outer', high_card, hint_size)

    if logger.isEnabledFor(logging.DEBUG):
        delta = GetNanoTime() - start
        logger.debug("Grouping creation complete.", extra={'elapsed_nanos': delta})

    # If the caller requested validation to be performed (to make sure the keys are
    # unique on one or both sides) do that now.
    if validate is not None:
        _validate_groupings(left_grouping, right_grouping, validate, keep)

    # Construct fancy indices for the left/right Datasets; these will be used to index into
    # columns of the respective datasets to produce new arrays/columns for the merged Dataset.
    start = GetNanoTime()
    join_indices = _create_merge_fancy_indices(left_grouping, right_grouping, right_groupby_grouping, how, keep)

    if logger.isEnabledFor(logging.DEBUG):
        delta = GetNanoTime() - start
        logger.debug("Join index creation complete.", extra={'elapsed_nanos': delta})

    # If running in debug mode (without specifying -O at the command-line),
    # verify the fancy indices will result in the same output lengths.
    # If they're not, we'll end up with an error right towards the end of this function when
    # everything's combined into the new Dataset instance.
#    if __debug__:
#        left_fancyindex_rowcount = JoinIndices.result_rowcount(join_indices.left_index, len(left))
#        right_fancyindex_rowcount = JoinIndices.result_rowcount(join_indices.right_index, len(right))
#        assert left_fancyindex_rowcount == right_fancyindex_rowcount,\
#            f'Left and right rowcount differ ({left_fancyindex_rowcount} vs. {right_fancyindex_rowcount}).'

#    # DEBUG: Print the constructed indices.
#    if logger.isEnabledFor(logging.DEBUG):
#        left_fancyindex = join_indices.left_index
#        right_fancyindex = join_indices.right_index
#        logger.debug(f'left_fancyindex ({type(left_fancyindex)}): {left_fancyindex}')
#        logger.debug(f'right_fancyindex ({type(right_fancyindex)}): {right_fancyindex}')

    return join_indices


#TODO: Clean-up column overlap
def merge2(
    left: 'Dataset',
    right: 'Dataset',
    on: Optional[Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]]] = None,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    how: str = 'left',
    suffixes: Optional[Tuple[str, str]] = None,
    copy: bool = True,
    indicator: Union[bool, str] = False,
    columns_left: Optional[Union[str, List[str]]] = None,
    columns_right: Optional[Union[str, List[str]]] = None,
    validate: Optional[str] = None,   # TODO: Consider changing this to require_unique: Union[bool, Tuple[bool, bool]] -- the semantics would be clearer to users
    keep: Optional[Union[str, Tuple[Optional[str], Optional[str]]]] = None,
    high_card: Optional[Union[bool, Tuple[Optional[bool], Optional[bool]]]] = None,
    hint_size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    **kwargs
) -> 'Dataset':
    """
    Merge Dataset by performing a database-style join operation by columns.

    Parameters
    ----------
    left : Dataset
        Left Dataset
    right : Dataset
        Right Dataset
    on : str or (str, str) or list of str or list of (str, str), optional
        Column names to join on. Must be found in both `left` and `right`.
    left_on : str or list of str, optional
        Column names from left Dataset to join on. When specified, overrides whatever is specified in `on`.
    right_on : str or list of str, optional
        Column names from right to join on. When specified, overrides whatever is specified in `on`.
    how : {'left', 'right', 'inner', 'outer'}
        The type of merge to be performed.

        * left: use only keys from `left`, as in a SQL 'left join'. Preserves the ordering of keys.
        * right: use only keys from `right`, as in a SQL 'right join'. Preserves the ordering of keys.
        * inner: use intersection of keys from both Datasets, as in a SQL 'inner join'. Preserves the
          ordering of keys from `left`.
        * outer: use union of keys from both Datasets, as in a SQL 'full outer join'.
    suffixes: tuple of (str, str), optional
        Suffix to apply to overlapping column names in the left and right side, respectively.
        The default (``None``) causes an exception to be raised for any overlapping columns.
    copy: bool, default True
        If False, avoid copying data when possible; this can reduce memory usage
        but users must be aware that data can be shared between `left` and/or `right`
        and the Dataset returned by this function.
    indicator : bool or str, default False
        If True, adds a column to output Dataset called "merge_indicator" with information on the
        source of each row. If string, column with information on source of each row will be added
        to output Dataset, and column will be named value of string. Information column is
        Categorical-type and takes on a value of "left_only" for observations whose merge key only
        appears in `left` Dataset, "right_only" for observations whose merge key only appears in
        `right` Dataset, and "both" if the observation's merge key is found in both.
    columns_left : str or list of str, optional
        Column names to include in the merge from `left`, defaults to None which causes all columns to be included.
    columns_right : str or list of str, optional
        Column names to include in the merge from `right`, defaults to None which causes all columns to be included.
    validate : {'one_to_one', '1:1', 'one_to_many', '1:m', 'many_to_one', 'm:1', 'many_to_many', 'm:m'}, optional
        Validate the uniqueness of the values in the columns specified by the `on`, `left_on`, `right_on`
        parameters. In other words, allows the _multiplicity_ of the keys to be checked so the user
        can prevent the merge if they want to ensure the uniqueness of the keys in one or both of the Datasets
        being merged.
        Note: The `keep` parameter logically takes effect before `validate` when they're both specified.
    keep : {'first', 'last'} or (str, str), optional
        An optional string which specifies that only the first or last occurrence
        of each unique key within `left` and `right` should be kept. In other words,
        resolves multiple occurrences of keys (multiplicity > 1) to a single occurrence.
    high_card : bool or (bool, bool), optional
        Hint to low-level grouping implementation that the key(s) of `left` and/or `right`
        contain a high number of unique values (cardinality); the grouping logic *may* use
        this hint to select an algorithm that can provide better performance for such cases.
    hint_size : int or (int, int), optional
        An estimate of the number of unique keys used for the join. Used as a performance hint
        to the low-level grouping implementation.
        This hint is typically ignored when `high_card` is specified.

    Returns
    -------
    merged : Dataset

    Examples
    --------
    >>> rt.merge2(ds_simple_1, ds_simple_2, left_on = 'A', right_on = 'X', how = 'inner')
    #   A      B   X       C
    -   -   ----   -   -----
    0   0   1.20   0    2.40
    1   1   3.10   1    6.20
    2   6   9.60   6   19.20
    <BLANKLINE>
    [3 rows x 4 columns] total bytes: 72.0 B

    Demonstrating a 'left' merge.

    >>> rt.merge2(ds_complex_1, ds_complex_2, on = ['A','B'], how = 'left')
    #   B    A       C       E
    -   -   --   -----   -----
    0   Q    0    2.40    1.50
    1   R    6    6.20   11.20
    2   S    9   19.20     nan
    3   T   11   25.90     nan
    <BLANKLINE>
    [4 rows x 4 columns] total bytes: 84.0 B

    See Also
    --------
    merge_asof
    """
    # Process keyword arguments.
    require_match = bool(kwargs.pop("require_match")) if "require_match" in kwargs else False
    if kwargs:
        # There were remaining keyword args passed here which we don't understand.
        first_kwarg = next(iter(kwargs.keys()))
        raise ValueError(f"This function does not support the kwarg '{first_kwarg}'.")

    # Collect timing stats on how long various stages of the merge operation take.
    start = GetNanoTime()

    # Normalize 'columns_left' and 'columns_right' first to simplify some logic later on
    # (by allowing us to assume they're a non-optional-but-maybe-empty List[str]).
    columns_left = _normalize_selected_columns(left, columns_left)
    columns_right = _normalize_selected_columns(right, columns_right)

    # Validate and normalize the 'on' column lists for each Dataset.
    left_on = _extract_on_columns(on, left_on, True, 'on', is_optional=False)
    right_on = _extract_on_columns(on, right_on, False, 'on', is_optional=False)

    # PERF: Revisit this -- it could be made faster if ItemContainer.keys() returned a set-like object such as KeysView instead of a list; then we wouldn't need to create the sets here.
    left_keyset = set(left.keys())
    _require_columns_present(left_keyset, 'left', 'left_on', left_on)
    _require_columns_present(left_keyset, 'left', 'columns_left', columns_left)  # PERF: Fix this -- if columns_left isn't populated initially it'll be normalized to the whole keyset above so this call is irrelevant
    right_keyset = set(right.keys())
    _require_columns_present(right_keyset, 'right', 'right_on', right_on)
    _require_columns_present(right_keyset, 'right', 'columns_right', columns_right)  # PERF: Fix this -- if columns_left isn't populated initially it'll be normalized to the whole keyset above so this call is irrelevant

    # Make sure there aren't any column name collision _before_ we do the heavy lifting of merging;
    # if there are name collisions, attempt to resolve them by suffixing the colliding column names.
    col_left_tuple, col_right_tuple, intersection_cols = \
        _construct_colname_mapping(left_on, right_on, suffixes=suffixes, columns_left=columns_left, columns_right=columns_right)

    if logger.isEnabledFor(logging.DEBUG):
        delta = GetNanoTime() - start
        logger.debug("Column validation complete.", extra={'elapsed_nanos': delta})

    # merge_indices only accepts the new-style (tuple-based) 'on' parameter, so we need to get
    # the parameters into that form.
    # TODO: If a caller is already specifying the new-style 'on' parameter, don't round-trip it through
    #       _extract_on_columns above -- let's simplify the logic so that above we either just re-use 'on'
    #       if it's specified (but not left_on/right_on) or we construct the new-style 'on' form directly there
    #       while validating so it doesn't need to be done as an extra step.
    if len(left_on) != len(right_on):
        ValueError(f'Differing numbers of columns used as join-keys for the left ({len(left_on)}) and right ({len(right_on)}) Datasets.')
    normalized_on = list(zip(left_on, right_on))

    # Call merge_indices to perform the join and return fancy indices we'll use to construct the merged Dataset.
    kwargs['skip_column_presence_check'] = True
    join_indices = merge_indices(
        left, right, on=normalized_on, how=how, validate=validate,
        keep=keep, high_card=high_card, hint_size=hint_size, **kwargs)
    left_fancyindex = join_indices.left_index
    right_fancyindex = join_indices.right_index

    #
    # TODO: Before using the constructed fancy indices to create the new "merged" columns
    #       by indexing into the columns of the left/right Datasets, examine the lengths of
    #       the indices, the selected output columns (+ their dtypes) -- use this to determine
    #       the size of the merged dataset. If the Dataset will be too large given the current
    #       free memory available in the machine (or user quota), don't proceed further because
    #       the Dataset construction would cause allocations which will lead to the process crashing.
    #       Instead, fail with an error -- at least the user can take action on that vs.
    #       e.g. a Jupyter kernel crash.
    #       This check can be done (if it'd be easier/faster) even before creating the fancy indices --
    #       all we need to know is the join type ('how') and the multiset (key * cardinality) of the join
    #       keys from the left and right datasets. Given we need those anyway to construct the fancy index
    #       and that construction logic will be well-optimized, there might not be much benefit to doing
    #       this check earlier, except in the case where the fancy indices themselves would be so large
    #       their construction itself would cause an out-of-memory crash.
    #

    if logger.isEnabledFor(logging.DEBUG):
        # The number of rows that'll be in the merged Dataset.
        result_num_rows = JoinIndices.result_rowcount(join_indices.left_index, len(left))

        # The estimated size, in bytes, of the merged Dataset.
        # "alloc size" here indicates the amount of newly-allocated memory that will be required for the result.
        # If column data from the original Datasets can be reused, this value will be lower than the virtual size.
        est_result_total_size = 0
        est_result_alloc_size = 0
        if intersection_cols:
            for field in intersection_cols:
                # TODO: Need to account for outer merge, where we'll hstack the intersection cols so the
                #       dtype will be the larger of the dtypes of the two columns (from the left/right Datasets).
                est_result_total_size += left[field].dtype.itemsize * result_num_rows

            if left_fancyindex is not None and right_fancyindex is not None:
                est_result_alloc_size += est_result_total_size

        est_result_total_size_tmp = 0
        for old_name in col_left_tuple[0]:
            est_result_total_size_tmp += left[old_name].dtype.itemsize * result_num_rows
        est_result_total_size += est_result_total_size_tmp
        if left_fancyindex is not None:
            est_result_alloc_size += est_result_total_size_tmp

        est_result_total_size_tmp = 0
        for old_name in col_right_tuple[0]:
            est_result_total_size_tmp += right[old_name].dtype.itemsize * result_num_rows
        est_result_total_size += est_result_total_size_tmp
        if right_fancyindex is not None:
            est_result_alloc_size += est_result_total_size_tmp

        # For now, just write the estimated size to the logger.
        # TODO: Check the estimated size against the current free memory size available
        #       and fail (or emit a warning) if it like's like we hit an OOM error and crash.
        logger.debug(
            "Estimated merged result size.",
            extra={'alloc_size': est_result_alloc_size, 'total_size': est_result_total_size}
        )

    def readonly_array_wrapper(arr: FastArray) -> FastArray:
        """Create a read-only view of an array."""
        new_arr = arr.view()
        new_arr.flags.writeable = False
        return new_arr

    def array_copy(arr: FastArray) -> FastArray:
        return arr.copy()

    # Based on the 'copy' flag, determine which function to use in cases where
    # an input column will contain the same data in the output Dataset.
    array_copier = array_copy if copy else readonly_array_wrapper

    # Workaround for rt.mbget not supporting Categoricals (so it ends up using the regular
    # integer invalids in the underlying array vs. the Categorical base index).
    # We still do want to use mbget() when possible since we need it to support the case where
    # we're trying to index into an np.ndarray, and eventually we want to use it to allow the
    # caller to provide their own per-column overrides for default values.
    def mbget_wrapper(arr: np.ndarray, index: np.ndarray, default_value = None) -> np.ndarray:
        return mbget(arr, index, default_value) if type(arr) in (FastArray, np.ndarray) else arr[index]

    # Begin creating the column data for the 'merged' Dataset.
    out: Dict[str, FastArray] = {}
    start = GetNanoTime()

    # Fetch/transform columns which overlap between the two Datasets (i.e. columns with these
    # names exist in both Datasets).
    if intersection_cols:
        # If we did an outer join but there aren't any rows whose keys only exist in the right Dataset,
        # we don't need special handling -- we can fall through to the code used for e.g. left join to
        # handle the intersecting columns.
        if how == 'outer' and join_indices.right_only_rowcount > 0:
            for field in intersection_cols:
                # Effectively, this is performing an hstack with the left 'on' column (after indexing into it with the
                # fancy index from the join) and the right 'on' column (after indexing into that with the right fancy
                # index from the join, sliced down to only the last N elements representing rows with right-only groups).
                # It's important we use hstack here to construct the resulting data column for the output;
                # hstack handles type unification of the two array types (in terms of both dtypes and Python classes)
                # to ensure the resulting type (for the output column) is able to hold data from both the left and right
                # source columns without truncating the data.
                # PERF: Consider implementing an 'hstack_fancy' function that accepts two lists -- one of arrays to
                #   hstack and another matching the length of the first where each element is a fancy index or None
                #   corresponding to the arrays of the first. We then go through the normal hstack logic to determine
                #   the output type but we compute the output length from the fancy indices; then, when copying to the
                #   result, we use the fancy indices to pull from the source arrays rather than a straight 1-to-1 copy.
                #   This function would allow for a few array allocations + operations to be elided here.
                left_data = left[field] if left_fancyindex is None else mbget_wrapper(left[field], left_fancyindex[:len(left_fancyindex) - join_indices.right_only_rowcount])
                right_data = mbget_wrapper(right[field], right_fancyindex[-join_indices.right_only_rowcount:])
                out[field] = hstack((left_data, right_data))

        # If we're missing one of the fancy indices, it means we can just copy the columns
        # over from the corresponding dataset to the result. We still allocate a new array
        # object as a read-only view of the original; this allows the name to be different
        # in the result (e.g. if changed later) than the original.
        elif left_fancyindex is None:
            for field in intersection_cols:
                out[field] = array_copier(left[field])
        elif right_fancyindex is None:
            for field in intersection_cols:
                out[field] = array_copier(right[field])

        else:
            # For any columns which overlap (i.e. same name) in the two Datasets,
            # fetch them from the left Dataset, except when performing a right join
            # in which case we need to fetch them from the right Dataset instead.
            # Applying the fancy index to transform each column to the correct shape for the resulting Dataset.
            # N.B. This is an arbitrary choice, we could just as easily default to fetching
            #      from the right Dataset instead.
            ds, fancyindex = (right, right_fancyindex) if how == 'right' else (left, left_fancyindex)
            for field in intersection_cols:
                out[field] = mbget_wrapper(ds[field], fancyindex)

    if logger.isEnabledFor(logging.DEBUG):
        delta = GetNanoTime() - start
        logger.debug("Transformed columns. cols='%s'", "intersection", extra={'elapsed_nanos': delta})

    # Transform the columns from the left Dataset and store to the new, merged Dataset.
    # If we don't have a left fancy index, it means the fancy index (if present) would simply create
    # a copy of the original column. Use this to optimize memory usage by just referencing (not copying)
    # the original column data in the new dataset.
    start = GetNanoTime()
    if left_fancyindex is None:
        for old_name, new_name in zip(*col_left_tuple):
            # Wrap the original array in a read-only view; this is necessary to allow the "new" column to
            # have a different name than the original without interfering with the original array / source Dataset.
            # We make the view read-only to help protect the original data from being unexpectedly modified.
            out[new_name] = array_copier(left[old_name])
    else:
        for old_name, new_name in zip(*col_left_tuple):
            out[new_name] = mbget_wrapper(left[old_name], left_fancyindex)

    if logger.isEnabledFor(logging.DEBUG):
        delta = GetNanoTime() - start
        logger.debug("Transformed columns. cols='%s'", "left", extra={'elapsed_nanos': delta})

    # Transform the columns from the right Dataset and store to the new, merged Dataset.
    start = GetNanoTime()
    if right_fancyindex is None:
        for old_name, new_name in zip(*col_right_tuple):
            # Wrap the original array in a read-only view; this is necessary to allow the "new" column to
            # have a different name than the original without interfering with the original array / source Dataset.
            # We make the view read-only to help protect the original data from being unexpectedly modified.
            out[new_name] = array_copier(right[old_name])
    else:
        # If the internal-use-only kwarg 'require_match' was specified and is True,
        # raise an exception if there were any keys from the 'left' Dataset that don't
        # have a matching key in 'right'. This is used in functions like merge_lookup().
        # PERF: Move this to the point where we build the gbkey<->gbkey lookup tables -- doing the check there will be
        #       cheaper/faster than doing it here since we only need to look at the key arrays instead of the larger
        #       fancy-index arrays. We can also use the information available at that point to provide a better error
        #       message (e.g. include the number of missing keys and an example of a missing key to help the user diagnose it).
        if require_match:
            # The index array could either be a boolean mask or a integer-based fancy index, so handle both.
            # PERF: For the fancy-index case, use the isallnan(...) function once it's implemented, it'll be faster.
            right_is_missing_keys = not (
                all(right_fancyindex) if right_fancyindex.dtype.char == '?' else all(isnotnan(right_fancyindex)))
            if right_is_missing_keys:
                raise ValueError('One or more keys from the left Dataset was missing from the right Dataset.')

        for old_name, new_name in zip(*col_right_tuple):
            out[new_name] = mbget_wrapper(right[old_name], right_fancyindex)

    if logger.isEnabledFor(logging.DEBUG):
        delta = GetNanoTime() - start
        logger.debug("Transformed columns. cols='%s'", "right", extra={'elapsed_nanos': delta})

    # If the caller has asked for an indicator column, create it now.
    if indicator:
        start = GetNanoTime()

        if indicator == True:
            indicator = 'merge_indicator'
        if indicator in out:
            raise ValueError(f'indicator column name collision with existing columns: {indicator}')

        # PERF: Revisit this code -- it works, but could be greatly optimized by constructing it
        #       from the left<->right key mappings we have available when building the left or right
        #       fancy indices.
        # TODO: Until we can implement the better approach described above, we can eliminate the 'where'
        #       calls and a couple of temporary array allocations by doing something like:
        #           indicator_category_index = isnotnan(left_fancyindex).view(np.int8)
        #           is_good_right = isnotnan(right_fancyindex).view(np.int8)
        #           indicator_category_index += is_good_right
        #           indicator_category_index += is_good_right  # add twice in-place; now we have the Categorical data
        is_good_left = True if left_fancyindex is None else isnotnan(left_fancyindex)
        is_good_right = True if right_fancyindex is None else isnotnan(right_fancyindex)

        indicator_category_index = where(is_good_left, 1, 0) + where(is_good_right, 2, 0)
        out[indicator] = Categorical(indicator_category_index, ['left_only', 'right_only', 'both'])

        if logger.isEnabledFor(logging.DEBUG):
            delta = GetNanoTime() - start
            logger.debug("merge_indicator created.", extra={'elapsed_nanos': delta})

    # Take the dictionary of column names we created, invoke the
    # selected dataset class constructor with it and return the new instance.
    datasetclass = type(left)
    return datasetclass(out)


#TODO: Clean-up column overlap and outer merge
def merge(
    left: 'Dataset',
    right: 'Dataset',
    on: Optional[Union[str, List[str]]] = None,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    how: str = 'left',
    suffixes: Tuple[str, str] = ('_x', '_y'),
    indicator: Union[bool, str] = False,
    columns_left: Optional[Union[str, List[str]]] = None,
    columns_right: Optional[Union[str, List[str]]] = None,
    verbose: bool = False,
    hint_size: int = 0):
    """
    Merge Dataset by performing a database-style join operation by columns.

    Parameters
    ----------
    left : Dataset
        Left Dataset
    right : Dataset
        Right Dataset
    on : str or list of str, optional
        Column names to join on. Must be found in both `left` and `right`.
    left_on : str or list of str, optional
        Column names from left Dataset to join on. When specified, overrides whatever is specified in `on`.
    right_on : str or list of str, optional
        Column names from right to join on. When specified, overrides whatever is specified in `on`.
    how : {'left','right', 'inner', 'outer'}
        - left: use only keys from the left. **The output rows will be in one-to-one correspondence with the left rows!** If multiple matches on the right occur, the last is taken.
        - right: use only keys from the right. **The output rows will be in one-to-one correspondence
            with the left rows!** If multiple matches on the left occur, the last is taken.
        - inner: use intersection of keys from both Datasets, similar to SQL inner join
        - outer: use union of keys from both Datasets, similar to SQL outer join
    suffixes: tuple of (str, str), default ('_x', '_y')
        Suffix to apply to overlapping column names in the left and right side, respectively.
        To raise an exception on overlapping columns use (False, False).
    indicator : bool or str, default False
        If True, adds a column to output Dataset called "merge_indicator" with information on the
        source of each row. If string, column with information on source of each row will be added
        to output Dataset, and column will be named value of string. Information column is
        Categorical-type and takes on a value of "left_only" for observations whose merge key only
        appears in `left` Dataset, "right_only" for observations whose merge key only appears in
        `right` Dataset, and "both" if the observation's merge key is found in both.
    columns_left : str or list of str, optional
        Column names to include in the merge from `left`, defaults to None which causes all columns to be included.
    columns_right : str or list of str, optional
        Column names to include in the merge from `right`, defaults to None which causes all columns to be included.
    verbose : boolean
        For the stdout debris, defaults to False
    hint_size : int
        An estimate of the number of unique keys used for the join, to optimize performance by
        pre-allocating memory for the key hash table.

    Returns
    -------
    merged : Dataset

    Examples
    --------
    >>> rt.merge(ds_simple_1, ds_simple_2, left_on = 'A', right_on = 'X', how = 'inner')
    #   A      B   X       C
    -   -   ----   -   -----
    0   0   1.20   0    2.40
    1   1   3.10   1    6.20
    2   6   9.60   6   19.20
    <BLANKLINE>
    [3 rows x 4 columns] total bytes: 72.0 B

    Demonstrating a 'left' merge.

    >>> rt.merge(ds_complex_1, ds_complex_2, on = ['A','B'], how = 'left')
    #   B    A       C       E
    -   -   --   -----   -----
    0   Q    0    2.40    1.50
    1   R    6    6.20   11.20
    2   S    9   19.20     nan
    3   T   11   25.90     nan
    <BLANKLINE>
    [4 rows x 4 columns] total bytes: 84.0 B

    See Also
    --------
    merge_asof
    """
    # Collect timing stats on how long various stages of the merge operation take.
    start=GetNanoTime()
    datasetclass = type(left)

    if (len(left) != len(right)) and how == 'outer':    # TODO: Is this check still functioning correctly after the change to len(Dataset) in riptable 1.2.x?
        if verbose:
            # TODO: Should this raise an actual Warning that's more visible to the end-user?
            print("Warning: outer merge produces unstable results for Datasets of different sizes.")

    if left_on is None:
        if on is None:
            raise ValueError("on and left_on cannot both be none")
        else:
            left_on = on
    else:
        # 'on' and 'left_on' are not allowed to be specified together, as it's unclear which one should take precedence.
        # If we want to define which one does take precedence, we could drop this down to
        # a Warning to let the user know part of what they've specified will be ignored.
        if on is not None:
            raise ValueError("The `on` and `left_on` parameters cannot be specified together; exactly one of them should be specified.")

    if right_on is None:
        if on is None:
            raise ValueError("on and right_on cannot both be none")
        else:
            right_on = on
    else:
        # 'on' and 'right_on' are not allowed to be specified together, as it's unclear which one should take precedence.
        # If we want to define which one does take precedence, we could drop this down to
        # a Warning to let the user know part of what they've specified will be ignored.
        if on is not None:
            raise ValueError("The `on` and `right_on` parameters cannot be specified together; exactly one of them should be specified.")

    #
    # TODO: Handle case where left_on / right_on was an empty string -- raise an Error
    #

    # Normalize 'left_on' and 'right_on' -- however they were specified,
    # list-ify them so they're easier to consume later.
    if isinstance(left_on, (bytes, str)):
        left_on = [left_on]
    if isinstance(right_on, (bytes, str)):
        right_on = [right_on]

    # Normalize 'columns_left' and 'columns_right' first to simplify some logic later on
    # (by allowing us to assume they're a non-optional-but-maybe-empty List[str]).
    columns_left = _normalize_selected_columns(left, columns_left)
    columns_right = _normalize_selected_columns(right, columns_right)

    # Verify that column(s) specified in 'left_on', 'right_on', 'columns_left' and 'columns_right' are all present in their respective Datasets.
    def require_columns_present(keyset: Set[str], dataset_name:str, param_name:str, col_names: Collection[str]) -> None:
        missing_cols: Optional[List[str]] = None
        for col_name in col_names:
            if col_name not in keyset:
                if missing_cols is None:
                    missing_cols = [col_name]
                else:
                    missing_cols.append(col_name)

        # If there were any missing columns, raise a ValueError whose message contains *all* of the
        # missing column names -- this provides more diagnostic info to the user compared to just
        # reporting the first missing column we found.
        if missing_cols is not None:
            joined_colnames = ", ".join(map(lambda x: f'\'{x}\'', missing_cols))
            raise ValueError(f'The column(s) {joined_colnames} specified in the `{param_name}` argument are not present in the `{dataset_name}` Dataset.')

    # PERF: Revisit this -- it could be made faster if ItemContainer.keys() returned a set-like object such as KeysView instead of a list; then we wouldn't need to create the sets here.
    left_keyset = set(left.keys())
    require_columns_present(left_keyset, 'left', 'left_on', left_on)
    require_columns_present(left_keyset, 'left', 'columns_left', columns_left)  # PERF: Fix this -- if columns_left isn't populated initially it'll be normalized to the whole keyset above so this call is irrelevant
    right_keyset = set(right.keys())
    require_columns_present(right_keyset, 'right', 'right_on', right_on)
    require_columns_present(right_keyset, 'right', 'columns_right', columns_right)  # PERF: Fix this -- if columns_left isn't populated initially it'll be normalized to the whole keyset above so this call is irrelevant

    # Make sure there aren't any column name collision _before_ we do the heavy lifting of merging;
    # if there are name collisions, attempt to resolve them by suffixing the colliding column names.
    _suffixes = None if (not suffixes[0]) and (not suffixes[1]) else suffixes
    col_left_tuple, col_right_tuple, intersection_cols = \
        _construct_colname_mapping(left_on, right_on, suffixes=_suffixes, columns_left=columns_left, columns_right=columns_right)

    # Validate the pair(s) of columns from the left and right join keys have compatible types.
    left_on_arrs = [left[col_name] for col_name in left_on]
    right_on_arrs = [right[col_name] for col_name in right_on]
    key_compat_errs = _verify_join_keys_compat(left_on_arrs, right_on_arrs)
    if key_compat_errs:
        # If the list of errors is non-empty, we have some join-key compatibility issues.
        # Some of the "errors" may just be warnings; filter those out of the list and raise
        # them to notify the user.
        actual_errors : List[Exception] = []
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
            flat_errs = '\n'.join(map(str, actual_errors))  # N.B. this is because it's disallowed to use backslashes inside f-string curly braces
            raise ValueError(f"Found one or more compatibility errors with the specified 'on' keys:\n{flat_errs}")

    if logger.isEnabledFor(logging.DEBUG):
        delta = GetNanoTime() - start
        logger.debug("validation took %d ns", delta, extra={'elapsed_nanos': delta})

    # Construct the join indices. That is, arrays for use as fancy indices into columns of the
    # left and right datasets to produce new arrays/columns for the merged Dataset.
    start=GetNanoTime()
    if hint_size <= 0:
        hint_size = max(left.shape[0], right.shape[0])
    idx, idx_left, idx_right = _construct_index(left, right, left_on, right_on, how, hint_size)

    if logger.isEnabledFor(logging.DEBUG):
        delta = GetNanoTime() - start
        logger.debug("_construct_index took %d ns", delta, extra={'elapsed_nanos': delta})

    # DEBUG: Print the constructed indices.
    #print(f'idx ({type(idx)}): {idx}')
    #print(f'idx_left ({type(idx_left)}): {idx_left}')
    #print(f'idx_right ({type(idx_right)}): {idx_right}')

    start=GetNanoTime()

    if how == 'left':
        #print('how was left')
        p_left = np.arange(left.shape[0])
        is_good_left = np.full(p_left.shape,True)
        #print('p_left',p_left)
        #print('is_good_left',is_good_left)
    else:
        is_good_left, p_left = ismember(idx, idx_left, hint_size=hint_size)
        #print('p_left',p_left)
        #print('is_good_left',is_good_left)

    if how == 'right':
        p_right = np.arange(right.shape[0])
        is_good_right = np.full(p_right.shape,True)
    else:
        is_good_right, p_right = ismember(idx, idx_right, hint_size=hint_size)
        #print('p_right',p_right)
        #print('is_good_right',is_good_right)

    if logger.isEnabledFor(logging.DEBUG):
        delta = GetNanoTime() - start
        logger.debug("ismember took %d ns", delta, extra={'elapsed_nanos': delta})

    # DEBUG: Print the constructed indices.
    #print(f'p_left ({type(p_left)}): {p_left}')
    #print(f'is_good_left ({type(is_good_left)}): {is_good_left}')
    #print()
    #print(f'p_right ({type(p_right)}): {p_right}')
    #print(f'is_good_right ({type(is_good_right)}): {is_good_right}')

    # Begin creating the column data for the 'merged' Dataset.
    out : Dict[str, FastArray] = {}
    start=GetNanoTime()

    if intersection_cols:
        if how == 'inner':
            for field in intersection_cols:
                out[field] = left[field][p_left]
        elif how == 'left':
            for field in intersection_cols:
                out[field] = left[field]
        elif how == 'right':
            for field in intersection_cols:
                out[field] = right[field]
        elif how == 'outer':
            in_right_not_in_left = mask_andi(isnan(p_left), isnotnan(p_right))
            for field in intersection_cols:
                out[field] = left[field][p_left]
                out[field][in_right_not_in_left] = right[field][p_right[in_right_not_in_left]]
        else:
            raise ValueError(f'The value \'{how}\' is not valid for the \'how\' parameter.')

    if logger.isEnabledFor(logging.DEBUG):
        delta = GetNanoTime() - start
        logger.debug("copying took %d ns", delta, extra={'elapsed_nanos': delta})

    start=GetNanoTime()

    for old_name, new_name in zip(*col_left_tuple):
        if how == "inner" or how == "right":
            #print('mbget hit in col_left.keys() right with')
            #print(left[col_left[fld]], p_left)
            out[new_name] = left[old_name][p_left]

        elif how == "outer":
            out[new_name] = left[old_name][p_left]

        else:
            #print('col left fld',col_left[fld])
            #print('left col left fld',left[col_left[fld]])
            out[new_name] = left[old_name]

    for old_name, new_name in zip(*col_right_tuple):
        #for fld2 in out.keys():
        #    print(fld2, 'address = ',out[fld2].__array_interface__['data'][0])
        if how == "inner" or how == "left":
            #print('pright from ismember',p_right)
            out[new_name] = right[old_name][p_right]
            #print('mbget', out[fld])

        elif how == "outer":
            out[new_name] = right[old_name][p_right]
        else:
            out[new_name] = right[old_name]

    if logger.isEnabledFor(logging.DEBUG):
        delta = GetNanoTime() - start
        logger.debug("indexing took %d ns", delta, extra={'elapsed_nanos': delta})

    if indicator:
        start=GetNanoTime()

        if indicator is True:
            indicator = 'merge_indicator'
        if indicator in out:
            raise ValueError(f'indicator column name collision with existing columns: {indicator}')

        indicator_category_index = where(is_good_left, 1, 0) + where(is_good_right, 2, 0)
        out[indicator] = Categorical(indicator_category_index, ['left_only', 'right_only', 'both'])

        if logger.isEnabledFor(logging.DEBUG):
            delta = GetNanoTime() - start
            logger.debug("merge_indicator created.", extra={'elapsed_nanos': delta})

    # Take the dictionary of column names we created, invoke the
    # selected dataset class constructor with it and return the new instance.
    return datasetclass(out)


def merge_lookup(
    left: 'Dataset',
    right: 'Dataset',
    on: Optional[Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]]] = None,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    require_match: bool = False,
    suffixes: Optional[Tuple[str, str]] = None,
    copy: bool = True,
    columns_left: Optional[Union[str, List[str]]] = None,
    columns_right: Optional[Union[str, List[str]]] = None,
    keep: Optional[str] = None,
    high_card: Optional[Union[bool, Tuple[Optional[bool], Optional[bool]]]] = None,
    hint_size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None
) -> 'Dataset':
    """
    Merge Dataset by performing a database-style left-join operation by columns.

    Parameters
    ----------
    left : Dataset
        Left Dataset
    right : Dataset
        Right Dataset
    on : str or (str, str) or list of str or list of (str, str), optional
        Column names to join on. Must be found in both `left` and `right`.
    left_on : str or list of str, optional
        Column names from left Dataset to join on. When specified, overrides whatever is specified in `on`.
    right_on : str or list of str, optional
        Column names from right to join on. When specified, overrides whatever is specified in `on`.
    require_match : bool, default False
        When True, all keys in `left` are required to have a matching key in `right`, and an exception
        is raised when this requirement is not met.
    suffixes: tuple of (str, str), optional
        Suffix to apply to overlapping column names in the left and right side, respectively.
        The default (``None``) causes an exception to be raised for any overlapping columns.
    copy: bool, default True
        If False, avoid copying data when possible; this can reduce memory usage
        but users must be aware that data can be shared between `left` and/or `right`
        and the Dataset returned by this function.
    columns_left : str or list of str, optional
        Column names to include in the merge from `left`, defaults to None which causes all columns to be included.
    columns_right : str or list of str, optional
        Column names to include in the merge from `right`, defaults to None which causes all columns to be included.
    keep : {'first', 'last'}, optional
        When `right` contains multiple rows with a given unique key from `left`, this function
        keeps only one such row and this parameter indicates whether it should be the first or
        last row with the given key; when the parameter is None (the default), an exception will
        be raised if there are any non-unique keys in `right`.
    high_card : bool or (bool, bool), optional
        Hint to low-level grouping implementation that the key(s) of `left` and/or `right`
        contain a high number of unique values (cardinality); the grouping logic *may* use
        this hint to select an algorithm that can provide better performance for such cases.
    hint_size : int or (int, int), optional
        An estimate of the number of unique keys used for the join. Used as a performance hint
        to the low-level grouping implementation.
        This hint is typically ignored when `high_card` is specified.

    Returns
    -------
    merged : Dataset

    Examples
    --------
    >>> rt.merge_lookup(ds_simple_1, ds_simple_2, left_on = 'A', right_on = 'X', how = 'inner')
    #   A      B   X       C
    -   -   ----   -   -----
    0   0   1.20   0    2.40
    1   1   3.10   1    6.20
    2   6   9.60   6   19.20
    <BLANKLINE>
    [3 rows x 4 columns] total bytes: 72.0 B


    Demonstrating a 'left' merge.

    >>> rt.merge_lookup(ds_complex_1, ds_complex_2, on = ['A','B'], how = 'left')
    #   B    A       C       E
    -   -   --   -----   -----
    0   Q    0    2.40    1.50
    1   R    6    6.20   11.20
    2   S    9   19.20     nan
    3   T   11   25.90     nan
    <BLANKLINE>
    [4 rows x 4 columns] total bytes: 84.0 B

    """
    # If keep is None, that means we want an exception to be raised if
    # there are any keys in 'right' occurring more than once. Do that via
    # merge2's 'validate' parameter.
    validate = 'm:1' if keep is None else None

    return merge2(
        left, right,
        on=on, left_on=left_on, right_on=right_on,
        how='left', suffixes=suffixes, copy=copy, indicator=False,
        columns_left=columns_left, columns_right=columns_right,
        validate=validate,
        keep=(None, keep), high_card=high_card, hint_size=hint_size,
        # kwargs
        require_match=require_match
    )


def merge_asof(
    left: 'Dataset',
    right: 'Dataset',
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
    tolerance: Optional[Union[int, 'timedelta']] = None,
    allow_exact_matches: bool = True,
    direction: str = "backward",
    verbose: bool = False,
    check_sorted: bool = True,
    matched_on: Union[bool, str] = False,
    **kwargs
) -> 'Dataset':
    """
    Perform an as-of merge. This is similar to a left-join except that we
    match on nearest key rather than equal keys.

    Both Datasets must be sorted by the key.

    For each row in the left Dataset:
      - A "backward" search selects the last row in the right Dataset whose
        'on' key is less than or equal to the left's key.
      - A "forward" search selects the first row in the right Dataset whose
        'on' key is greater than or equal to the left's key.
      - A "nearest" search selects the row in the right Dataset whose 'on'
        key is closest in absolute distance to the left's key.
        The 'nearest' search has not been implemented yet.

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
        **Not implemented yet.**
    total_unique: integer, optional, default None
        required in a faster version of merge_asof for the case
        where both left and right Datasets are prebinned. Not implemented yet.
    allow_exact_matches : boolean, default True
        - If True, allow matching with the same 'on' value
          (i.e. less-than-or-equal-to / greater-than-or-equal-to)
        - If False, don't match the same 'on' value
          (i.e., strictly less-than / strictly greater-than)
    direction : 'backward' (default), 'forward', or 'nearest'
        Whether to search for prior, subsequent, or closest matches.
        The option 'nearest' has not been implemented yet.
    verbose : bool, default False
        For the stdout debris; defaults to False.
    check_sorted : bool, default True
        Specifies whether a sortedness check should be performed on the
        `on` column(s). Defaults to True for safety, but users can override
        to False to avoid performance overhead if they're certain the data
        is already sorted.
    matched_on : bool or str, default False
        If set to True or a string, an additional column is added to the result;
        for each row, it contains the value from the `on` column in `right` that was matched.
        When True, the column will use the default name 'matched_on'; specify a string
        to explicitly name the column.

    Other Parameters
    ----------------
    left_index : boolean, optional, default False
        Unused. This parameter is only provided for compatibility with the pandas ``merge_asof``
        function and will be removed in a later version of riptable.
    right_index : boolean, optional, default False
        Unused. This parameter is only provided for compatibility with the pandas ``merge_asof``
        function and will be removed in a later version of riptable.

    Returns
    -------
    merged : Dataset

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

    See Also
    --------
    merge
    """
    # Process keyword arguments.
    left_index = bool(kwargs.pop("left_index", False))
    right_index = bool(kwargs.pop("right_index", False))
    if kwargs:
        # There were remaining keyword args passed here which we don't understand.
        first_kwarg = next(iter(kwargs.keys()))
        raise ValueError(f"This function does not support the kwarg '{first_kwarg}'.")

    if left_index or right_index:
        # Emit warning about 'left_index' and 'right_index' only being present for compatibility
        # with the pandas merge signature. They don't actually do anything in riptable since our
        # indexing is external (not internal) to Datasets.
        warnings.warn("The 'left_index' and 'right_index' parameters are only present for pandas compatibility. They are not applicable for riptable and will have no effect.")

    #
    # TODO: Validate the 'direction' -- needs to be in {'forward', 'backward', 'nearest'};
    #       Create the set once and expose it as a static property (not attribute) on _AsOfMerge so it's not reparsed and re-created each time.
    #

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
                "The `on` and `left_on` parameters cannot be specified together; exactly one of them should be specified.")

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
                "The `on` and `right_on` parameters cannot be specified together; exactly one of them should be specified.")

    # Validate and normalize the 'by' column name lists for each Dataset.
    # Note that for 'merge_asof', specifying any 'by' columns are optional -- unlike the 'on' columns, which are required.
    left_by = _extract_on_columns(by, left_by, True, 'by', is_optional=True)
    right_by = _extract_on_columns(by, right_by, False, 'by', is_optional=True)

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
    _require_columns_present(left_keyset, 'left', 'left_on', [left_on])
    _require_columns_present(left_keyset, 'left', 'left_by', left_by)
    _require_columns_present(left_keyset, 'left', 'columns_left', columns_left)  # PERF: Fix this -- if columns_left isn't populated initially it'll be normalized to the whole keyset above so this call is irrelevant
    right_keyset = set(right.keys())
    _require_columns_present(right_keyset, 'right', 'right_on', [right_on])
    _require_columns_present(right_keyset, 'right', 'right_by', right_by)
    _require_columns_present(right_keyset, 'right', 'columns_right', columns_right)  # PERF: Fix this -- if columns_right isn't populated initially it'll be normalized to the whole keyset above so this call is irrelevant

    # Validate the pair(s) of columns from the left and right join keys have compatible types.
    left_on_arrs = [left[left_on]]
    left_on_arrs.extend([left[col_name] for col_name in left_by])
    right_on_arrs = [right[right_on]]
    right_on_arrs.extend([right[col_name] for col_name in right_by])
    key_compat_errs = _verify_join_keys_compat(left_on_arrs, right_on_arrs)
    if key_compat_errs:
        # If the list of errors is non-empty, we have some join-key compatibility issues.
        # Some of the "errors" may just be warnings; filter those out of the list and raise
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
            flat_errs = '\n'.join(map(str, actual_errors))  # N.B. this is because it's disallowed to use backslashes inside f-string curly braces
            # TEMP: Until riptable 1.5, make this a warning instead of an error for backwards-compatibility with existing code -- users need time to fix any type mismatches.
            warnings.warn(f"Found one or more compatibility issues with the specified 'on' keys (these will become errors in a future major release):\n{flat_errs}")
            # raise ValueError(f"Found one or more compatibility errors with the specified 'on' keys:\n{flat_errs}")

    #
    # TODO: Verify the 'on' columns are an ordered, comparable type (int, float, DateTimeNano, datetime, ordered Categorical, etc.)
    #       This is a requirement for them to be used with this function; without this, the 'asof' comparison can't be performed
    #       and will raise an exception later -- better to check here and give the user a better error message.
    #

    return _AsOfMerge.get_result(
        left, right,
        left_on=left_on, right_on=right_on,
        left_by=left_by, right_by=right_by,
        suffixes=suffixes,
        copy=copy,
        columns_left=columns_left, columns_right=columns_right,
        tolerance=tolerance,
        allow_exact_matches=allow_exact_matches,
        direction=direction,
        verbose=verbose,
        check_sorted=check_sorted,
        matched_on=matched_on
    )


class _AsOfMerge:
    @staticmethod
    def get_result(
        left: 'Dataset',
        right: 'Dataset',
        left_on: str,
        right_on: str,
        left_by: Collection[str],
        right_by: Collection[str],
        suffixes: Optional[Tuple[str, str]],
        columns_left: Collection[str],
        columns_right: Collection[str],
        copy: bool,
        tolerance: Optional[Union[int, 'timedelta']],
        allow_exact_matches: bool,
        direction: str,
        verbose: bool,
        check_sorted: bool,
        matched_on: Union[bool, str]
    ) -> 'Dataset':
        # TEMP: Emit a warning if `tolerance` is specified until we actually implement it.
        if tolerance is not None:
            warnings.warn("merge_asof: The `tolerance` parameter is not implemented yet and will have no effect.")

        #FastArray._ROFF()  # temporary hack to protect against data corruption
        datasetclass = type(left)

        # Make sure there aren't any column name collision _before_ we do the heavy lifting of merging;
        # if there are name collisions, attempt to resolve them by suffixing the colliding column names.
        # For the purposes of this validation, we combine the 'on' column and any 'by' column(s) since they're both
        # keys (it's just that the 'on' column is treated specially when the join is performed).
        left_keys = [left_on]
        left_keys.extend(left_by)
        right_keys = [right_on]
        right_keys.extend(right_by)
        col_left_tuple, col_right_tuple, intersection_cols = \
            _construct_colname_mapping(left_keys, right_keys, suffixes=suffixes, columns_left=columns_left, columns_right=columns_right)

        left_on_col = left[left_on]
        right_on_col = right[right_on]

        # Before doing anything else, perform a sortedness check on the 'on' columns
        # unless the user has specified otherwise.
        if check_sorted:
            # Check whether the left_on column is sorted.
            start = GetNanoTime()
            left_on_sorted = issorted(left_on_col)

            if logger.isEnabledFor(logging.DEBUG):
                delta = GetNanoTime() - start
                logger.debug(f"merge_asof: issorted(%s).", "left_on", extra={'elapsed_nanos': delta})

            # If the left 'on' column isn't sorted, raise an exception.
            if not left_on_sorted:
                raise ValueError(f"The column '{left_on}' from the `left` Dataset is not sorted.")

            # Check whether the right_on column is sorted.
            start = GetNanoTime()
            right_on_sorted = issorted(right_on_col)

            if logger.isEnabledFor(logging.DEBUG):
                delta = GetNanoTime() - start
                logger.debug(f"merge_asof: issorted(%s).", "right_on", extra={'elapsed_nanos': delta})

            # If the right 'on' column isn't sorted, raise an exception.
            if not right_on_sorted:
                raise ValueError(f"The column '{right_on}' from the `right` Dataset is not sorted.")

        # Construct fancy indices for the left/right Datasets; these will be used to index into
        # columns of the respective datasets to produce new arrays/columns for the merged Dataset.
        start = GetNanoTime()
        if direction == 'nearest':
            # Recursively call forward/backward, choose the min. Note backward wins ties.
            # TODO: This could be done more efficiently (without having to run the join-index creation twice + combine the results)
            #       once the key-alignment / join index creation implements support for the 'nearest' direction.
            left_fancyindex_backward, right_fancyindex_backward = \
                _AsOfMerge._create_merge_fancy_indices(
                    left, right, left_on_col, right_on_col, left_by, right_by,
                    tolerance=tolerance, allow_exact_matches=allow_exact_matches,
                    direction='backward', verbose=verbose)

            left_fancyindex_forward, right_fancyindex_forward = \
                _AsOfMerge._create_merge_fancy_indices(
                    left, right, left_on_col, right_on_col, left_by, right_by,
                    tolerance=tolerance, allow_exact_matches=allow_exact_matches,
                    direction='forward', verbose=verbose)

            # Create a filter (bool array) indicating the rows where the 'forward' direction was closer ("nearest")
            # than the 'backward' direction, making sure to account for NaN/NA values.
            # We do this in a slightly unusual way -- inlined into the computation below -- for memory efficiency
            # and also to work around the __sub__ and __lt__ operations not correctly handling NaN/NA/invalid values in
            # integer arrays (as of 2020-05-05).
            # TODO: When that's fixed, the code could be simplified to:
            #     backward_distance = left_on_col[left_fancyindex_backward] - right_on_col[right_fancyindex_backward]
            #     forward_distance = left_on_col[left_fancyindex_forward] - right_on_col[right_fancyindex_forward]
            #     f_forward = isnan(backward_distance)
            #     f_forward |= (forward_distance < backward_distance)
            #     f_forward &= isnotnan(forward_distance)
            f_forward_closer = full(len(right_fancyindex_backward), False, dtype=bool)

            # Determine the distance between the left_on and right_on columns in both the forward and backwards directions.
            # TODO: What should we do when the 'on' columns are integers and operations below underflow/overflow?
            #       This is especially problematic if the 'on' columns are unsigned integer dtypes.
            # NOTE: Indexing into an ndarray/FastArray with None returns the original data but reshaped; so handle that case specially here.
            left_on_backward = left_on_col if left_fancyindex_backward is None else left_on_col[left_fancyindex_backward]
            f_forward_closer |= isnan(left_on_backward)
            right_on_backward = right_on_col if right_fancyindex_backward is None else right_on_col[right_fancyindex_backward]
            f_forward_closer |= isnan(right_on_backward)
            backward_distance = abs(left_on_backward - right_on_backward)
            del left_on_backward
            del right_on_backward
            f_forward_closer |= isnan(backward_distance)

            left_on_forward = left_on_col if left_fancyindex_forward is None else left_on_col[left_fancyindex_forward]
            right_on_forward = right_on_col if right_fancyindex_forward is None else right_on_col[right_fancyindex_forward]
            forward_distance = abs(left_on_forward - right_on_forward)

            f_forward_closer |= (forward_distance < backward_distance)
            del backward_distance
            f_forward_closer &= isnotnan(left_on_forward)
            f_forward_closer &= isnotnan(right_on_forward)
            f_forward_closer &= isnotnan(forward_distance)
            del left_on_forward
            del right_on_forward
            del forward_distance

            # Overwrite elements of the 'backward' fancy indices with the elements of the 'forward'
            # fancy indices where the 'forward' row from the right side was nearer.
            # NOTE: merge_asof is _always_ an "m:1" left join so the left fancy index should always be None;
            #       so we don't need to do any processing for it here.
            assert left_fancyindex_backward is None
            assert left_fancyindex_forward is None
            putmask(right_fancyindex_backward, f_forward_closer, right_fancyindex_forward)

            # Set the same variables for fancy indices as used by the normal 'forward'/'backward' code path.
            left_fancyindex = left_fancyindex_backward
            right_fancyindex = right_fancyindex_backward

        else:
            left_fancyindex, right_fancyindex = \
                _AsOfMerge._create_merge_fancy_indices(
                    left, right, left_on_col, right_on_col, left_by, right_by,
                    tolerance=tolerance, allow_exact_matches=allow_exact_matches,
                    direction=direction, verbose=verbose)

        if logger.isEnabledFor(logging.DEBUG):
            delta = GetNanoTime() - start
            logger.debug("Join index creation complete.", extra={'elapsed_nanos': delta})

        # Begin creating the column data for the 'merged' Dataset.
        out: Dict[str, FastArray] = {}
        start = GetNanoTime()

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

        if logger.isEnabledFor(logging.DEBUG):
            delta = GetNanoTime() - start
            logger.debug("Transformed columns. cols='%s'", "intersection", extra={'elapsed_nanos': delta})

        # Transform the columns from the left Dataset and store to the new, merged Dataset.
        # If we don't have a left fancy index, it means the fancy index (if present) would simply create
        # a copy of the original column. Use this to optimize memory usage by just referencing (not copying)
        # the original column data in the new dataset.
        start = GetNanoTime()
        if left_fancyindex is None:
            for old_name, new_name in zip(*col_left_tuple):
                # Wrap the original array in a read-only view; this is necessary to allow the "new" column to
                # have a different name than the original without interfering with the original array / source Dataset.
                # We make the view read-only to help protect the original data from being unexpectedly modified.
                out[new_name] = array_copier(left[old_name])
        else:
            for old_name, new_name in zip(*col_left_tuple):
                out[new_name] = left[old_name][left_fancyindex]

        if logger.isEnabledFor(logging.DEBUG):
            delta = GetNanoTime() - start
            logger.debug("Transformed columns. cols='%s'", "left", extra={'elapsed_nanos': delta})

        # Transform the columns from the right Dataset and store to the new, merged Dataset.
        start = GetNanoTime()
        if right_fancyindex is None:
            for old_name, new_name in zip(*col_right_tuple):
                # Wrap the original array in a read-only view; this is necessary to allow the "new" column to
                # have a different name than the original without interfering with the original array / source Dataset.
                # We make the view read-only to help protect the original data from being unexpectedly modified.
                out[new_name] = array_copier(right[old_name])
        else:
            for old_name, new_name in zip(*col_right_tuple):
                out[new_name] = right[old_name][right_fancyindex]

        if logger.isEnabledFor(logging.DEBUG):
            delta = GetNanoTime() - start
            logger.debug("Transformed columns. cols='%s'", "right", extra={'elapsed_nanos': delta})

        # If the caller has asked for the 'matched_on' column, create it now.
        if matched_on:
            start = GetNanoTime()

            if matched_on == True:
                matched_on = 'matched_on'
            if matched_on in out:
                raise ValueError(f'`matched_on` column name collision with existing columns: {matched_on}')

            # Use the right_fancyindex to expand the 'on' column from the right Dataset and add it to the output.
            out[matched_on] = right_on_col[right_fancyindex]

            if logger.isEnabledFor(logging.DEBUG):
                delta = GetNanoTime() - start
                logger.debug("matched_on column created.", extra={'elapsed_nanos': delta})

        #FastArray._RON()  # temporary hack to protect against data corruption
        return datasetclass(out)

    @staticmethod
    def _create_merge_fancy_indices(
        left: 'Dataset',
        right: 'Dataset',
        left_on_col: FastArray,
        right_on_col: FastArray,
        left_by: Collection[str],
        right_by: Collection[str],
        tolerance: Optional[Union[int, 'timedelta']],
        allow_exact_matches: bool,
        direction: str,
        verbose: bool
    ) -> Tuple[Optional[FastArray], Optional[FastArray]]:
        # check for categoricals first
        # TODO: Finish implementing/testing this. If we re-use the code from merge2() for creating the Grouping objects,
        #       it'd work for any number of columns and mixed Categorical / non-Categorical arrays passed for 'by', and
        #       null/missing/NA keys would also be handled correctly. We could probably also write a function that takes
        #       similar inputs to _build_right_fancyindex() (and either duplicates some of it's functionality, or calls
        #       that function then refines the result) -- basically do the 'by' grouping then refine that result using
        #       the 'on' columns. Basically, like a GroupByOps.searchsorted().
        if False:
            # TJD: when both left and right are prebinned and left_on_col, right_on_col are sorted and are int32/64/float32/float64
            # this is a future fast path to merge_asof
            start = GetNanoTime()
            #pright = merge_prebinned(left_by, right_by, left_on_col, right_on_col, total_unique + 1)

            if logger.isEnabledFor(logging.DEBUG):
                delta = GetNanoTime() - start
                logger.debug("merge_prebinned took %d ns", delta, extra={'elapsed_nanos': delta})

        else:
            # The list of 'by' column names is normalized earlier on in the 'merge_asof' logic,
            # so we can rely on it being an instance of collections.abc.Collection here.
            left_by_col_count = len(left_by)
            right_by_col_count = len(right_by)
            if left_by_col_count != right_by_col_count:
                raise ValueError(f"Different number of 'by' columns for the left and right Datasets. ({left_by_col_count} vs. {right_by_col_count})")

            if left_by_col_count == 0:
                # TODO: Can we modify alignmk so we can pass None (or an empty list) instead of allocating arrays here?
                left_by_cols = np.ones(left.shape[0])
                right_by_cols = np.ones(right.shape[0])
            elif left_by_col_count == 1:
                left_by_cols = left[left_by]
                right_by_cols = right[right_by]
            else:
                left_by_cols = tuple([left[col] for col in left_by])
                right_by_cols = tuple([right[col] for col in right_by])

            start = GetNanoTime()
            pright = alignmk(
                left_by_cols, right_by_cols,
                left_on_col, right_on_col,
                direction=direction,
                allow_exact_matches=allow_exact_matches,
                verbose=verbose)

            if logger.isEnabledFor(logging.DEBUG):
                delta = GetNanoTime() - start
                logger.debug("aligmmk took %d ns", delta, extra={'elapsed_nanos': delta})

        return None, pright
