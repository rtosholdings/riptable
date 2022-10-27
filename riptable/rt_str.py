__all__ = [
    "FAString",
    "CatString",
]

from functools import partial, wraps
from inspect import signature
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Union

if TYPE_CHECKING:
    from .rt_dataset import Dataset

import inspect
import re
import warnings
from functools import wraps

import numba as nb
import numpy as np
from numba.core.dispatcher import Dispatcher

from .config import get_global_settings
from .rt_enum import TypeRegister
from .rt_fastarray import FastArray
from .rt_numpy import empty, ones, unique, where, zeros
from .Utils.common import cached_property

# Partially-specialize the numba.njit decorator to simplify its use in the FAString class below.
_njit_serial = partial(nb.njit, parallel=False, cache=get_global_settings().enable_numba_cache, nogil=True)
_njit_par = partial(nb.njit, parallel=True, cache=get_global_settings().enable_numba_cache, nogil=True)


class FAStrDispatchPair(NamedTuple):
    """A pair of numba-based functions; one compiled for serial execution and the other for parallel execution."""

    serial: Dispatcher
    parallel: Dispatcher

    @staticmethod
    def create(py_func: callable) -> "FAStrDispatchPair":
        func_serial = _njit_serial(py_func)
        func_par = _njit_par(py_func)
        return FAStrDispatchPair(serial=func_serial, parallel=func_par)


def _warn_deprecated_naming(old_func, new_func):
    warnings.warn(
        f"`{old_func}` is now deprecated and has been renamed to `{new_func}`", DeprecationWarning, stacklevel=2
    )


# NOTE YOU MUST INSTALL tbb
# conda install tbb
# to confirm...
# >>> from numba import threading_layer()
# >>> threading_layer()
# >>> 'tbb'
#
# NOTE this class was first written reshaping the array as 2d
# ----
# if intype[1]=='S':
#    return self.view(np.uint8).reshape((len(self), self.itemsize))
# if intype[1]=='U':
#    return self.view(np.uint32).reshape((len(self), self.itemsize//4))
# ----
# then pass to numba as 2d using shape[0] outer loop, shape[1] inner loop with [row,col] accessors
# ---
# then flip back to string
# if self.itemsize == 4:
#    result=self.ravel().view('U'+str(self.shape[1]))
# else:
#    result=self.ravel().view('S'+str(self.shape[1]))
#
# Keeping it as 1d and remembering the itemsize is 10% faster on large arrays, and even faster on small arrays
# If recycling kicks in (only works on 1d arrays), it is 30% faster
# Also note: On Windows as of Aug 2019, prange is much slower without tbb installed
#
# subclass from FastArray


@nb.njit
def _str_equal(str1, str2):
    for char1, char2 in zip(str1, str2):
        if char1 != char2:
            return False
    return True


def _handle_apply_unique(func):
    sign = signature(func)

    if "apply_unique" not in sign.parameters:
        raise ValueError(f"apply_unique not found in the signature of {func}")

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        bound_args = sign.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        if bound_args.kwargs["apply_unique"]:
            new_kwargs = bound_args.kwargs.copy()
            new_kwargs["apply_unique"] = False
            unique_values, index = unique(self.backtostring, return_inverse=True)
            result = func(unique_values.str, *bound_args.args, **new_kwargs)
            return result[index] if isinstance(result, FastArray) else result[index, :]
        else:
            return func(self, *args, **kwargs)

    return wrapper


def _maybe_encode(x):
    if not isinstance(x, bytes):
        x = x.encode()
    return x


class FAString(FastArray):
    """
    String accessor class for `FastArray`.
    """

    _APPLY_PARALLEL_THRESHOLD = 10_000

    def __new__(cls, arr):
        # if data comes in list like, convert to an array
        if not isinstance(arr, np.ndarray):
            if np.isscalar(arr):
                arr = np.asanyarray([arr])
            else:
                arr = np.asanyarray(arr)

        intype = arr.dtype.char
        if intype == "O":
            # try to convert to string (might have come from pandas)
            # default to unicode (note: FastArray attempts 'S' first)
            arr = arr.astype("U")
            intype = arr.dtype.char

        itemsize = np.int64(arr.itemsize)

        if intype == "S":
            # convert to two dim one byte array
            instance = arr.view(np.uint8)
        elif intype == "U":
            # convert to two dim four byte array
            instance = arr.view(np.uint32)
            itemsize = itemsize // 4
        else:
            raise TypeError(f"FAString can only be used on byte string and unicode not {arr.dtype!r}")

        instance = instance.view(cls)

        # remember the intype and itemsize
        instance._intype = intype
        instance._itemsize = itemsize
        return instance

    # -----------------------------------------------------
    @property
    def str(self):
        # we already are a str
        return self

    # -----------------------------------------------------
    @property
    def backtostring(self):
        """
        convert back to FastArray or np.ndarray 'S' or 'U' string
        'S12'  or 'U40'
        """
        return FastArray(self._np).view(self._intype + str(self._itemsize))

    @property
    def n_elements(self):
        """
        The number of elements in the original string array
        """
        return len(self) // self._itemsize

    # -----------------------------------------------------
    def possibly_convert_tostr(self, arr):
        """
        converts list like or an array to the same string type
        """

        # if data comes in list like, convert to an array
        if not isinstance(arr, np.ndarray):
            if not isinstance(arr, (list, tuple)):
                arr = np.asanyarray([arr])
            else:
                arr = np.asanyarray(arr)

        if arr.dtype.char != self._intype:
            arr = arr.astype(self._intype)

        return arr

    def _validate_input(self, str2):
        str2 = self.possibly_convert_tostr(str2)
        if len(str2) != 1:
            raise TypeError(f"A single string must be passed for str2 not {str2!r}")
        str2 = FAString(str2)
        return str2

    # -----------------------------------------------------
    def _apply_func(self, func, funcp, *args, dtype=None, input=None):
        # can optionally pass in dtype
        # check when to flip into parallel mode.  > 10,000 go to parallel routine
        # TODO: Can't just check len(self) for parallelism here; if operating on a string-mode Categorical
        #       the parallelism decision should be based on e.g. len(my_cat.category_array), not the length
        #       of the categorical itself.
        if len(self) >= self._APPLY_PARALLEL_THRESHOLD and funcp is not None:
            func = funcp
        if dtype is None:
            dest = empty(len(self), self.dtype)
            dest._itemsize = self._itemsize
            dest._intype = self._intype
        else:
            # user requested specific output dtype
            dest = empty(self.n_elements, dtype)

        if input is None:
            func(self._itemsize, dest, *args)
        else:
            if not isinstance(input, FAString):
                raise TypeError(f"The input= value was not a FAString.  It is {input!r}.")
            func(input, input._itemsize, dest, *args)

        if dtype is None:
            dest = dest.view(dest._intype + str(dest._itemsize))

        return dest

    # -----------------------------------------------------
    def apply(self, func, *args, dtype=None):
        """
        Write your own string apply function
        NOTE: byte strings are passed as uint8
        NOTE: unicode strings are passed as uint32

        default signature must match

        @nb.njit(cache=get_global_settings().enable_numba_cache, nogil=True)
        def nb_upper(src, itemsize, dest):

        src: is uint array
        itemsize: is how wide the string is per row
        dest: is return uint array

        Other Parameters
        ----------------
        *args: pass in zero or more arguments (the arguments are always at the end)
        dtype: specify a different dtype

        Example
        -------
        >>> import numba as nb
        ... @nb.njit(cache=get_global_settings().enable_numba_cache, nogil=True)
        ... def nb_upper(src, itemsize, dest):
        ...     for i in nb.prange(len(src) / itemsize):
        ...         rowpos = i * itemsize
        ...        for j in range(itemsize):
        ...             c=src[rowpos+j]
        ...             if c >= 97 and c <= 122:
        ...                 # convert to ASCII upper
        ...                 dest[rowpos+j] = c-32
        ...             else:
        ...                dest[rowpos+j] = c

        >>> FAString(['this  ','that ','test']).apply(nb_upper)

        """
        return self._apply_func(func, func, *args, dtype=dtype, input=self)

    # -----------------------------------------------------
    def _nb_upper_inplace(src, itemsize):
        # loop over all rows
        for i in nb.prange(len(src) // itemsize):
            # loop over all chars in the string
            rowpos = i * itemsize
            # loop over all chars in the string
            for j in range(itemsize):
                c = src[rowpos + j]
                if c >= 97 and c <= 122:
                    # convert to ASCII upper
                    src[rowpos + j] = c - 32

    # -----------------------------------------------------
    def _nb_upper(src, itemsize, dest):
        # loop over all rows
        for i in nb.prange(np.int64(len(src) // itemsize)):
            rowpos = i * itemsize
            # loop over all chars in the string
            for j in range(itemsize):
                c = src[rowpos + j]
                if c >= 97 and c <= 122:
                    # convert to ASCII upper
                    dest[rowpos + j] = c - 32
                else:
                    dest[rowpos + j] = c

    # -----------------------------------------------------
    def _nb_lower(src, itemsize, dest):
        # loop over all rows
        for i in nb.prange(np.int64(len(src) // itemsize)):
            rowpos = i * itemsize
            # loop over all chars in the string
            for j in range(itemsize):
                c = src[rowpos + j]
                if c >= 65 and c <= 90:
                    # convert to ASCII lower
                    dest[rowpos + j] = c + 32
                else:
                    dest[rowpos + j] = c

    # -----------------------------------------------------
    def _nb_removetrailing(src, itemsize, dest, removechar):
        # loop over all rows
        for i in nb.prange(len(src) // itemsize):
            # loop over all chars in the string backwards
            rowpos = i * itemsize

            startpos = -1
            for startpos in range(itemsize - 1, -1, -1):
                c = src[rowpos + startpos]
                is_trailing_char = c == 0 or c == removechar
                out_char = 0 if is_trailing_char else c
                dest[rowpos + startpos] = out_char

                if not is_trailing_char:
                    break

            for pos in range(startpos - 1, -1, -1):
                dest[rowpos + pos] = src[rowpos + pos]

    # -----------------------------------------------------
    def _nb_reverse_inplace(src, itemsize):
        # loop over all rows
        for i in nb.prange(len(src) // itemsize):
            rowpos = i * itemsize
            # find length of string
            strlen = 0
            while strlen < itemsize:
                if src[rowpos + strlen] == 0:
                    break
                strlen += 1
            end = rowpos + strlen - 1
            start = rowpos
            while start < end:
                temp = src[end]
                src[end] = src[start]
                src[start] = temp
                start += 1
                end -= 1

    # -----------------------------------------------------
    def _nb_reverse(src, itemsize, dest):
        # loop over all rows
        for i in nb.prange(len(src) // itemsize):
            rowpos = i * itemsize
            # find length of string
            strlen = 0
            while strlen < itemsize:
                if src[rowpos + strlen] == 0:
                    break
                strlen += 1
            srcpos = 0
            while strlen > 0:
                strlen -= 1
                dest[rowpos + strlen] = src[rowpos + srcpos]
                srcpos += 1
            while srcpos < itemsize:
                dest[rowpos + srcpos] = 0
                srcpos += 1

    # -----------------------------------------------------
    def _nb_strlen(src, itemsize, dest):
        # loop over all rows
        for i in nb.prange(len(src) // itemsize):
            # loop over all chars in the string
            rowpos = i * itemsize
            strlen = 0
            # loop over all chars in the string
            for j in range(itemsize):
                if src[rowpos + j] == 0:
                    break
                strlen += 1
            # store length of string
            dest[i] = strlen

    # -----------------------------------------------------
    def _nb_index_any_of(src, itemsize, dest, str2):
        str2len = len(str2)
        # loop over all rows
        for i in nb.prange(len(src) // itemsize):
            # loop over all chars in the string
            rowpos = i * itemsize
            found = 0
            # loop over all chars in the string
            for j in range(itemsize):
                c = src[rowpos + j]
                for k in range(str2len):
                    if c == str2[k]:
                        # store location of match
                        dest[i] = j
                        found = 1
                        break
                if found == 1:
                    break
            if found == 0:
                dest[i] = -1

    # -----------------------------------------------------
    def _nb_index(src, itemsize, dest, str2):
        str2len = len(str2)
        # loop over all rows
        for i in nb.prange(len(src) // itemsize):
            # loop over all chars in the string
            rowpos = i * itemsize
            dest[i] = -1
            # loop over all substrings of sufficient length
            for j in range(itemsize - str2len + 1):
                if _str_equal(src[rowpos + j :], str2):
                    # store location of match
                    dest[i] = j
                    break

    # -----------------------------------------------------
    def _nb_contains(src, itemsize, dest, str2):
        str2len = len(str2)
        # loop over all rows
        for i in nb.prange(np.int64(len(src) // itemsize)):
            rowpos = i * itemsize
            dest[i] = False
            # loop over all substrings of sufficient length
            for j in range(itemsize - str2len + 1):
                if _str_equal(src[rowpos + j :], str2):
                    dest[i] = True
                    break

    def _nb_find(src, itemsize, dest, str2):
        """
        Searches src for occurrences of str2 and build a Boolean array
        with a row per string indicating indicating the starting points of all such occurrences.
        """
        str2len = len(str2)
        # loop over all rows
        for i in nb.prange(np.int64(len(src) // itemsize)):
            rowpos = i * itemsize
            # loop over all substrings of sufficient length
            str_pos = 0
            while str_pos <= itemsize - str2len:
                if _str_equal(src[rowpos + str_pos :], str2):
                    dest[i, str_pos] = True
                    str_pos += str2len
                else:
                    str_pos += 1

        return dest

    # -----------------------------------------------------

    def _nb_endswith(src, itemsize, dest, str2):
        str2len = len(str2)
        # loop over all rows
        for i in nb.prange(len(src) // itemsize):
            # loop over all chars in the string
            rowpos = i * itemsize
            dest[i] = False

            # loop over all chars in the string
            # check if enough space left
            if itemsize >= str2len:
                k = itemsize
                while (k > 0) and (src[rowpos + k - 1] == 0):
                    k -= 1

                # check if still enough space left
                if k >= str2len:

                    k2 = str2len
                    # check if only the end matches
                    while k2 > 0:
                        if src[rowpos + k - 1] != str2[k2 - 1]:
                            break
                        k -= 1
                        k2 -= 1
                    if k2 == 0:
                        # indicate we have a match
                        dest[i] = True

    # -----------------------------------------------------
    def _nb_startswith(src, itemsize, dest, str2):
        str2len = len(str2)
        # loop over all rows
        for i in nb.prange(np.int64(len(src) / itemsize)):
            # loop over all chars in the string
            rowpos = i * itemsize
            dest[i] = False
            # loop over all chars in the string
            # check if enough space left
            if itemsize >= str2len:
                k = 0
                # check if only the beginning matches
                while k < str2len:
                    if src[rowpos + k] != str2[k]:
                        break
                    k += 1
                if k == str2len:
                    # indicate we have a match
                    dest[i] = True

    # -----------------------------------------------------
    @property
    def upper(self):
        """
        upper case a string (bytes or unicode)
        makes a copy

        Examples
        --------
        >>> FAString(['this','that','test']).upper
        FastArray(['THIS','THAT','TEST'], dtype='<U4')

        """
        return self._apply_func(self.nb_upper, self.nb_upper_par)

    # -----------------------------------------------------
    @property
    def lower(self):
        """
        upper case a string (bytes or unicode)
        makes a copy

        Examples
        --------
        >>> FAString(['THIS','THAT','TEST']).lower
        FastArray(['this','that','test'], dtype='<U4')

        """
        return self._apply_func(self.nb_lower, self.nb_lower_par)

    # -----------------------------------------------------
    @property
    def upper_inplace(self):
        """
        upper case a string (bytes or unicode)
        does not make a copy

        Examples
        --------
        FAString(['this','that','test']).upper_inplace
        """
        # TODO: Enable parallel version + dispatching based on array length.
        self.nb_upper_inplace(self._itemsize)
        return self.backtostring

    # -----------------------------------------------------
    @property
    def reverse(self):
        """
        upper case a string (bytes or unicode)
        does not make a copy

        Examples
        --------
        FAString(['this','that','test']).reverse
        """
        return self._apply_func(self.nb_reverse, self.nb_reverse_par)

    # -----------------------------------------------------
    @property
    def reverse_inplace(self):
        """
        upper case a string (bytes or unicode)
        does not make a copy

        Examples
        --------
        FAString(['this','that','test']).reverse_inplace
        """
        # TODO: Enable parallel version + dispatching based on array length.
        self.nb_reverse_inplace(self._itemsize)
        return self.backtostring

    # -----------------------------------------------------
    def removetrailing(self, remove=32):
        """
        removes spaces at end of string (often to fixup matlab string)
        makes a copy

        Other Parameters
        ----------------
        remove=32.  defaults to removing ascii 32 (space character)

        Examples
        --------
        >>> FAString(['this  ','that ','test']).removetrailing()
        FastArray(['this','that','test'], dtype='<U6')
        """
        return self._apply_func(self.nb_removetrailing, self.nb_removetrailing_par, remove)

    # -----------------------------------------------------
    @cached_property  # only cached for Python 3.7 or higher
    def strlen(self):
        """
        return the string length of every string (bytes or unicode)

        Examples
        --------
        >>> FAString(['this  ','that ','test']).strlen
        FastArray([6, 5, 4])
        """
        return self._apply_func(self.nb_strlen, self.nb_strlen_par, dtype=np.int32)

    # -----------------------------------------------------
    def index_any_of(self, str2):
        """
        return the first index location any of the characters that are part of str2,
        or -1 if none of the characters match

        Parameters
        ----------
        str2 - a string with one or more characters to search for

        Examples
        --------
        >>> FAString(['this  ','that ','test']).index_any_of('ia')
        FastArray([2, 2, -1])
        """
        if not isinstance(str2, FAString):
            if str2 == "":
                return zeros(self.n_elements, dtype=np.int32)
            str2 = self._validate_input(str2)

        return self._apply_func(self.nb_index_any_of, self.nb_index_any_of_par, str2, dtype=np.int32)

    def strpbrk(self, str2):
        _warn_deprecated_naming("strpbrk", "index_any_of")
        return self.index_any_of(str2)

    # -----------------------------------------------------
    def index(self, str2):
        """
        return the first index location of the entire substring specified in str2,
        or -1 if the substring does not exist

        Parameters
        ----------
        str2 - a string with one or more characters to search for

        Examples
        --------
        >>> FAString(['this  ','that ','test']).index('at')
        FastArray([-1, 2, -1])
        """
        if not isinstance(str2, FAString):
            if str2 == "":
                return zeros(self.n_elements, dtype=np.int32)

            str2 = self._validate_input(str2)

        return self._apply_func(self.nb_index, self.nb_index_par, str2, dtype=np.int32)

    def strstr(self, str2):
        _warn_deprecated_naming("strstr", "index")
        return self.index(str2)

    # -----------------------------------------------------
    def contains(self, str2):
        """
        Return a boolean array that's True for each string element that contains the
        given substring, otherwise False.

        The entire substring must match.

        Parameters
        ----------
        str2 : str
            A string with one or more characters to search for. To search using regular
            expressions, use :meth:`FAString.regex_match`.

        Returns
        -------
        `FastArray`
            A boolean array where the value is True if the string contains the
            entire substring specified in `str2`, otherwise False.

        See Also
        --------
        FAString.startswith
        FAString.endswith
        FAString.regex_match

        Examples
        --------
        >>> FAString(['this  ','that ','test']).contains('at')
        FastArray([False, True, False])

        This can be called on a `FastArray` using ``.str.contains()``.

        >>> a = rt.FastArray(['this  ','that ','test'])
        >>> a.str.contains('at')
        FastArray([False,  True, False])
        """
        if not isinstance(str2, FAString):
            if str2 == "":
                return ones(self.n_elements, dtype=bool)

            str2 = self._validate_input(str2)

        return self._apply_func(self.nb_contains, self.nb_contains_par, str2, dtype=bool)

    def _find(self, str2):
        """
        Searches src for occurences of str2 and build a Boolean mask the same size
        as src indicating the starting point of all such occurences.

        Parameters
        ----------
        str2 - a string with one or more characters to search for

        Examples
        --------
        >>> FAString(['this','that','test']).find('t')
        FastArray([
            [True, False, False, False],
            [True, False, False, True],
            [True, False, False, True]
        ])
        """
        if not isinstance(str2, FAString):
            if str2 == "":
                return ones(len(self), dtype=bool)

            str2 = self._validate_input(str2)

        return self.nb_find(self._itemsize, str2=str2, dest=zeros((self.n_elements, self._itemsize), dtype=bool))

    def strstrb(self, str2):
        _warn_deprecated_naming("strstrb", "contains")
        return self.contains(str2)

    def _nb_replace(src, itemsize, dest, dest_itemsize, old, new, locations):
        old_len = len(old)
        new_len = len(new)

        new_is_empty = len(new) == 1 and new[0] == 0

        for row in nb.prange(np.int64(len(src) // itemsize)):
            rowpos = row * itemsize
            src_pos = 0
            dest_pos = row * dest_itemsize

            while src_pos < itemsize:
                if locations[row, src_pos]:
                    if new_is_empty:
                        src_pos += old_len
                    else:
                        dest[dest_pos : dest_pos + new_len] = new
                        src_pos += old_len
                        dest_pos += new_len
                else:
                    dest[dest_pos] = src[rowpos + src_pos]
                    src_pos += 1
                    dest_pos += 1
        return dest

    def replace(self, old: str, new: str) -> FastArray:
        """
        Replace all occurrences of `old` with `new`
        """
        if not isinstance(old, FAString):
            if old == "":
                raise ValueError("cannot replace the empty string")
            old = self._validate_input(old)

        new = self._validate_input(new)

        locations = self._find(old)
        if not locations.any():
            return self.backtostring

        char_diff = len(new) - len(old)
        if char_diff > 0:
            max_n_replacements = locations.sum(axis=1).max()
            dest_itemsize = self._itemsize + char_diff * max_n_replacements
        elif char_diff < 0:
            min_n_replacements = locations.sum(axis=1).min()
            dest_itemsize = self._itemsize - char_diff * min_n_replacements
        else:
            dest_itemsize = self._itemsize

        dest = zeros(self.n_elements * dest_itemsize, dtype=self.dtype)
        replace = self.nb_replace_par if len(self) >= self._APPLY_PARALLEL_THRESHOLD else self.nb_replace
        replaced = replace(
            itemsize=self._itemsize,
            dest=dest,
            dest_itemsize=dest_itemsize,
            old=old,
            new=new,
            locations=locations,
        )
        return replaced.view(self._intype + str(dest_itemsize))

    # -----------------------------------------------------
    def startswith(self, str2):
        """
        Return a boolean array that's True where the given substring matches the start
        of each string element, otherwise False.

        The entire substring must match.

        Parameters
        ----------
        str2 : str
            A string with one or more characters to search for. To search using regular
            expressions, use :meth:`FAString.regex_match`.

        Returns
        -------
        `FastArray`
            A boolean array where the value is True if the string starts with the
            entire substring specified in `str2`, otherwise False.

        See Also
        --------
        FAString.endswith
        FAString.contains
        FAString.regex_match

        Examples
        --------
        >>> FAString(['this  ','that ','test']).startswith('thi')
        FastArray([True, False, False])

        This can be called on a `FastArray` using ``.str.startswith()``.

        >>> a = rt.FastArray(['this  ','that ','test'])
        >>> a.str.startswith('thi')
        FastArray([True, False, False])
        """
        if not isinstance(str2, FAString):
            if str2 == "":
                return ones(self.n_elements, dtype=bool)

            str2 = self.possibly_convert_tostr(str2)
            if len(str2) != 1:
                return TypeError(f"A single string must be passed for str2 not {str2!r}")
            str2 = FAString(str2)

        return self._apply_func(self.nb_startswith, self.nb_startswith_par, str2, dtype=bool)

    # -----------------------------------------------------
    def endswith(self, str2):
        """
        Return a boolean array that's True where the given substring matches the end
        of each string element, otherwise False.

        The entire substring must match.

        Parameters
        ----------
        str2 : str
            A string with one or more characters to search for. To search using regular
            expressions, use :meth:`FAString.regex_match`.

        Returns
        -------
        `FastArray`
            A boolean array where the value is True if the string ends with the entire
            substring specified in `str2`, otherwise False.

        See Also
        --------
        FAString.startswith
        FAString.contains
        FAString.regex_match

        Examples
        --------
        >>> FAString(['abab','ababa','abababb']).endswith('ab')
        FastArray([True, False, False])

        This can be called on a `FastArray` using ``.str.endswith()``.

        >>> a = rt.FastArray(['abab','ababa','abababb'])
        >>> a.str.endswith('ab')
        FastArray([True, False, False])
        """
        if not isinstance(str2, FAString):
            if str2 == "":
                return ones(self.n_elements, dtype=bool)

            str2 = self.possibly_convert_tostr(str2)
            if len(str2) != 1:
                return TypeError(f"A single string must be passed for str2 not {str2!r}")
            str2 = FAString(str2)

        return self._apply_func(self.nb_endswith, self.nb_endswith_par, str2, dtype=bool)

    @_handle_apply_unique
    def regex_match(self, regex: Union["str", bytes], apply_unique: bool = True) -> FastArray:
        """
        Return a boolean array that's True where the given substring or regular
        expression pattern is contained in each string element, otherwise False.

        The entire substring or pattern must match.

        Applies :py:func:`re.search` on each element with `regex` as the pattern.

        Parameters
        ----------
        regex : str
            String or regular expression pattern to search for.
        apply_unique: bool, default True
            When True, the regex is applied to the unique values and then expanded
            using the reverse index (see :meth:`riptable.unique`). This is optimal
            for repetitive data and benign for unique or highly non-repetitive data.

        Returns
        -------
        `FastArray`
            A boolean array where the value is True if the string element contains the
            entire substring or regex pattern specified in `regex`, otherwise False.

        See Also
        --------
        FAString.regex_replace : Replace each instance of a specified substring or
            pattern.
        FAString.extract :
            Extract one or more pattern groups into a `Dataset` or `FastArray`.
        FAString.contains
        FAString.startswith
        FAString.endswith

        Examples
        --------
        Find any instance of 'ab' that appears at the end of a string:

        >>> FAString(['abab','ababa','abababb']).regex_match('ab$')
        FastArray([True, False, False])

        This can be called on a `FastArray` using ``.str.regex_match()``.

        >>> a = rt.FastArray(['abab','ababa','abababb'])
        >>> a.str.regex_match('ab$')
        FastArray([True, False, False])
        """
        if self._intype == "S":
            regex = _maybe_encode(regex)
        regex = re.compile(regex)
        vmatch = np.vectorize(lambda x: bool(regex.search(x)))
        bools = vmatch(self.backtostring)

        return bools

    @_handle_apply_unique
    def regex_replace(
        self, regex: Union["str", bytes], repl: Union["str", bytes], apply_unique: bool = True
    ) -> FastArray:
        """
        Replace each instance of a specified substring or pattern.

        The entire substring or pattern must match. If the substring or pattern isn't
        found, the original string is returned unchanged.

        The behavior is identical to that of :py:func:`re.sub`. In particular, the
        returned string is obtained by replacing the leftmost non-overlapping
        occurrences of the substring or pattern with the replacement string.

        Parameters
        ----------
        regex : str
            String or regular expression pattern to search for.
        repl : str
            The replacement string.
        apply_unique : bool, default True
            When True, the regex is applied to the unique values and then expanded
            using the reverse index (see :meth:`riptable.unique`). This is optimal
            for repetitive data and benign for unique or highly non-repetitive data.

        Returns
        -------
        `FastArray`
            An array with all occurrences of the substring or pattern replaced.

        See Also
        --------
        FAString.regex_match :
            Return a boolean array that indicates whether given substring or regular
            expression pattern is contained in each string element.
        FAString.extract :
            Extract one or more pattern groups into a `Dataset` or `FastArray`.
        FAString.contains
        FAString.startswith
        FAString.endswith

        Examples
        --------
        Replace instances of 'aa' with 'b'. All non-overlapping occurrences are
        replaced, starting from the left:

        >>> FAString(['aaa', 'aaaa', 'aaaaa']).regex_replace('aa', 'b')
        FastArray(['ba', 'bb', 'bba'], dtype='<U3>')

        Replace any instance of 'ab' that appears at the end of a string with 'b'.

        >>> FAString(['abab','ababa','abababb']).regex_replace('ab$', 'b')
        FastArray(['abb', 'ababa', 'abababb'], dtype='<U7')

        This can be called on a FastArray using ``.str.regex_replace()``. The
        returned `FastArray` elements are byte strings.

        >>> a = rt.FastArray(['abab','ababa','abababb'])
        >>> a.str.regex_replace('ab$', 'b')
        FastArray([b'abb', b'ababa', b'abababb'], dtype='|S7')
        """
        if self._intype == "S":
            regex, repl = map(_maybe_encode, (regex, repl))
        regex = re.compile(regex)
        vmatch = np.vectorize(lambda x: regex.sub(repl, x))
        return vmatch(self.backtostring)

    @_handle_apply_unique
    def extract(
        self, regex: str, expand: Optional[bool] = None, fillna: str = "", names=None, apply_unique: bool = True
    ) -> Union[FastArray, "Dataset"]:
        """
        Extract one or more pattern groups from each element of an array into a
        `FastArray` or `Dataset`.

        This is useful when you have pieces of data in a string that you want to split
        into separate elements.

        For one capture group, the default is to return a `FastArray`, but this can be
        overridden by setting `expand` to True or by providing a name of a `Dataset`
        column to populate. For more than one capture group, a `Dataset` is returned.

        Column names for the resulting `Dataset` can be specified within the regex
        using ``(?P<name>)`` in the capture group(s) or by passing the `names` argument,
        which may be more convenient.

        Parameters
        ----------
        regex : str
            The pattern(s) to search for. Define multiple capture groups using
            parentheses.
        expand : bool, default False
            Set to True to return a `Dataset` for a single capture group. If False, a
            `FastArray` is returned.
        fillna : str, default '' (empty string)
            For elements where there's no match, this is the fill value for the
            resulting `FastArray` or `Dataset` column.
        names : list of str, default None
            For more than one capture group, a `Dataset` is returned. Optionally, you
            can provide column names (keys) for the extracted data.
        apply_unique : bool
            When True, the regex is applied to the unique values and then expanded
            using the reverse index (see :meth:`riptable.unique`). This is optimal
            for repetitive data and benign for unique or highly non-repetitive data.

        Returns
        -------
        `FastArray` or `Dataset`
            For one capture group, a `FastArray` (or optionally a `Dataset`) is
            returned. For more than one capture group, a `Dataset` is returned.

        See Also
        --------
        FAString.regex_match :
            Return a boolean array that indicates whether given string or regular
            expression pattern is contained in each string element.
        FAString.regex_replace : Replace each instance of a specified string or pattern.

        Examples
        --------
        These examples use a `FastArray` containing OSI symbols.

        >>> osi = rt.FastArray(['SPX UO 12/15/23 C5700', 'SPXW UO 09/17/21 C3650'])

        Extract one substring:

        >>> osi.str.extract('\w+')
        FastArray([b'SPX', b'SPXW'], dtype='|S4')

        Provide a name for the resulting `Dataset` column:

        >>> osi.str.extract('(?P<root>\w+)')
        #   root
        -   ----
        0   SPX
        1   SPXW

        Define two capture groups and provide names for the resulting `Dataset` columns:

        >>> osi.str.extract('(\w+).* (\d{2}/\d{2}/\d{2})', names = ['root', 'expiration'])
        #   root   expiration
        -   ----   ----------
        0   SPX    12/15/23
        1   SPXW   09/17/21

        Extract one substring into a `Dataset` column using ``expand = True``. (Note
        that for the element with an unmatched pattern, an empty string is returned).

        >>> osi.str.extract('\w+W', expand = True)
        #   group_0
        -   -------
        0
        1   SPXW
        """
        kwargs = dict(expand=expand, fillna=fillna, names=names, apply_unique=apply_unique)

        if self._intype == "S":
            regex = _maybe_encode(regex)
        compiled = re.compile(regex)

        ngroups = compiled.groups
        if ngroups == 0:
            # convenience where we treat the entire pattern as a capture group
            return self.extract(f"({regex.decode()})", **kwargs)

        # expand defaults to False if we have one capture group and do not specify names
        if expand is None:
            expand = ngroups > 1 or names is not None or compiled.groupindex
        elif not expand and ngroups > 1:
            raise ValueError("expand cannot be False with multiple capture groups")

        if names is None:
            names = [f"group_{i}" for i in range(ngroups)]
            for name, index in compiled.groupindex.items():
                names[index - 1] = name
        elif len(names) != ngroups:
            raise ValueError(f"Number of names, {len(names)}, does not match number of groups, {ngroups}")

        strings = self.backtostring
        # we define a list containing an empty array for each group
        # use Python lists as we do not know how many chars will be in the resultant arrays.
        # Performance in comparable to unsing pre-allocated numpy arrays with ~600K unique elements
        out_arrs = [[fillna] * len(strings) for _ in range(ngroups)]

        for i, s in enumerate(strings):
            result = compiled.search(s)
            if result is not None:
                result = result.groups()
                for s, arr in zip(result, out_arrs):
                    arr[i] = s

        if expand:
            out = TypeRegister.Dataset(dict(zip(names, out_arrs)))
        else:
            out = FastArray(out_arrs[0])
        return out

    def _nb_substr(src, out, itemsize, start, stop, strlen):
        n_elements = len(out)
        max_chars = 0
        for elem in nb.prange(n_elements):
            elem_len = strlen[elem]
            i, j = start[elem], stop[elem]
            if i < 0:
                i += elem_len
            if j < 0:
                j += elem_len
            i = max(i, 0)
            j = min(j, elem_len)
            for out_pos, pos in enumerate(range(i, j)):
                char = src[itemsize * elem + pos]
                out[elem, out_pos] = char
                max_chars = max(max_chars, out_pos + 1)
        return out[:, :max_chars]

    def _substr(self, start: Union[int, np.ndarray], stop: Optional[Union[int, np.ndarray]] = None) -> FastArray:
        """
        Take a substring of each element using slice args.
        """
        if stop is None:
            # emulate behaviour of slice
            start, stop = 0, start

        def _bound_to_array(value, name):
            value = TypeRegister.FastArray(value)
            if len(value) == 1:
                value = np.full(self.n_elements, value[0])
            if value.shape != (self.n_elements,):
                raise ValueError(
                    f"{name} must be an integer or an array of length equal to the number of elements"
                    f" in self. Expected {(self.n_elements,)}, got {value.shape}"
                )
            return value

        start = _bound_to_array(start, "start")
        stop = _bound_to_array(stop, "stop")

        strlen = self.strlen

        out = zeros((self.n_elements, self._itemsize), self.dtype)

        if self.n_elements >= self._APPLY_PARALLEL_THRESHOLD:
            substr = self.nb_substr_par
        else:
            substr = self.nb_substr

        out = substr(out, self._itemsize, start, stop, strlen)

        n_chars = out.shape[1]
        if n_chars == 0:  # empty sub strings everywhere
            out = zeros(self.n_elements, self.dtype).view(f"{self._intype}1")
        else:
            out = out.ravel().view(f"<{self._intype}{n_chars}")
        return FastArray(out)

    @property
    def substr(self):
        return _SubStrAccessor(self)

    def substr_char_stop(self, stop: str, inclusive: bool = False) -> FastArray:
        """
        Take a substring of each element using characters as bounds.

        Parameters
        ----------
        stop:
            A string used to determine the start of the sub-string.
            Excluded from the result by default.
            We go to the end of the string where stop is not in found in the corresponding element
        inclusive: bool
            If True, include the stopping string in the result

        Examples
        --------
        >>> s = FastArray(['ABC', 'A_B', 'AB_C', 'AB_C_DD'])
        >>> s.str.substr_char_stop('_')
        FastArray([b'ABC', b'A', b'AB', b'AB'], dtype='|S2')
        >>> s.str.substr_char_stop('_', inclusive=True)
        FastArray([b'ABC', b'A_', b'AB_', b'AB_'], dtype='|S2')
        """
        int_stop = self.index(stop)
        int_stop[int_stop == -1] = self._itemsize  # return full string if stop not found
        if inclusive:
            int_stop += 1
        return self.substr(int_stop)

    def _nb_char(src, position, itemsize, strlen, out):
        broken_at = len(position)
        for i in nb.prange(len(position)):
            pos = position[i]
            if pos < 0:
                pos = strlen[i] + pos

            if pos >= itemsize or pos < 0:
                # Parallel reduction on this index.
                # Otherwise, returning here prevents the function from being parallelized.
                broken_at = np.minimum(broken_at, i)  # this triggers error below (in `char()`).

                # TODO: Set out[i] to some invalid value?
                out[i] = 0
            else:
                out[i] = src[itemsize * i + pos]

        return broken_at if broken_at < len(position) else -1

    def char(self, position: Union[int, List[int], np.ndarray]) -> FastArray:
        """
        Take a single character from each element.

        Parameters
        ----------
        position: int or list of int or np.ndarray
            The position of the character to be extracted. Negative values respect the
            length of the individual strings.
            If an array, the length must be equal to the number of strings.
            An error is raised if any positions are out of bounds (>= self._itemsize).
        """
        position = np.asanyarray(position)

        # Handle scalars
        if np.ndim(position) == 0:
            for size in [8, 16, 32, 64]:
                dtype = getattr(np, f"uint{size}")
                if self._itemsize <= np.iinfo(dtype).max:
                    break
            position = ones(self.n_elements, dtype) * position

        if len(position) != self.n_elements:
            raise ValueError("position must be a scalar or a vector of the same length as self")

        out = zeros(self.n_elements, self.dtype)
        broken_at = self._nb_char(position, self._itemsize, self.strlen, out)
        if broken_at >= 0:
            raise ValueError(f"Position {position[broken_at]} out of bounds " f"for string of length {self._itemsize}")
        out = out.view(f"{self._intype}1")
        return FastArray(out)

    # Use the specialized decorators to create both a serial and parallel version of each
    # numba function (so we only need one definition of each), then add it to FAString.
    nb_upper_inplace = _njit_serial(_nb_upper_inplace)
    nb_upper_inplace_par = _njit_par(_nb_upper_inplace)

    nb_upper = _njit_serial(_nb_upper)
    nb_upper_par = _njit_par(_nb_upper)

    # TODO: nb_lower_inplace

    nb_lower = _njit_serial(_nb_lower)
    nb_lower_par = _njit_par(_nb_lower)

    nb_removetrailing = _njit_serial(_nb_removetrailing)
    nb_removetrailing_par = _njit_par(_nb_removetrailing)

    nb_reverse_inplace = _njit_serial(_nb_reverse_inplace)
    nb_reverse_inplace_par = _njit_par(_nb_reverse_inplace)

    nb_reverse = _njit_serial(_nb_reverse)
    nb_reverse_par = _njit_par(_nb_reverse)

    nb_strlen = _njit_serial(_nb_strlen)
    nb_strlen_par = _njit_par(_nb_strlen)

    nb_index_any_of = _njit_serial(_nb_index_any_of)
    nb_index_any_of_par = _njit_par(_nb_index_any_of)

    nb_index = _njit_serial(_nb_index)
    nb_index_par = _njit_par(_nb_index)

    nb_contains = _njit_serial(_nb_contains)
    nb_contains_par = _njit_par(_nb_contains)

    nb_find = _njit_serial(_nb_find)
    nb_replace = _njit_serial(_nb_replace)
    nb_replace_par = _njit_serial(_nb_replace)

    nb_endswith = _njit_serial(_nb_endswith)
    nb_endswith_par = _njit_par(_nb_endswith)

    nb_startswith = _njit_serial(_nb_startswith)
    nb_startswith_par = _njit_par(_nb_startswith)

    nb_substr = _njit_serial(_nb_substr)
    nb_substr_par = _njit_par(_nb_substr)

    nb_char = _njit_serial(_nb_char)
    nb_char_par = _njit_par(_nb_char)


def _populate_wrappers(cls):
    """
    Decorator for the CatString class which populates the string methods and sets the defaults for
    filtered value fills.
    """
    functions = dict(inspect.getmembers(FAString, inspect.isfunction))
    properties = dict(inspect.getmembers(FAString, lambda o: isinstance(o, (cached_property, property))))
    for name in [
        "upper",
        "lower",
        "reverse",
        "removetrailing",
        "strlen",
        "index_any_of",
        "index",
        "contains",
        "startswith",
        "endswith",
        "regex_match",
        "regex_replace",
        "_substr",
        "char",
        "replace",
        "_find",
        # Deprecated methods.
        "strstr",
        "strstrb",
        "strpbrk",
    ]:
        if name in functions:
            wrapper = cls._build_method(functions[name])
        elif name in properties:
            wrapper = cls._build_property(name)
        else:
            raise RuntimeError(f"{name} is not defined on FAString as a function or property")
        setattr(cls, name, wrapper)
    return cls


class _SubStrAccessor:
    """
    Class for providing slicing on string arrays via FAString.substr
    """

    def __init__(self, fastring):
        self.fastring = fastring

    def __getitem__(self, y):
        if isinstance(y, slice):
            start = 0 if y.start is None else y.start
            return self.fastring._substr(start, y.stop)
        else:
            return self.fastring.char(y)

    def __call__(self, start, stop=None):
        return self.fastring._substr(start, stop)


@_populate_wrappers
class CatString:
    """
    Provides access to FAString methods for Categoricals.
    All string methods are wrappers of the FAString equivalent with
    categorical re-expansion and option for how to fill filtered elements.
    """

    def __init__(self, cat):
        from .rt_categorical import CategoryMode

        self.cat = cat
        if cat.category_mode == CategoryMode.StringArray:
            string_list = cat.category_array
            self.fastring = FAString(string_list)
        else:
            raise ValueError(f"Could not use .str in {CategoryMode(cat.category_mode).name}.")

    @cached_property
    def _isfiltered(self):
        return self.cat.isfiltered()

    def _convert_fastring_output(self, out):
        if out.dtype.kind in "SU":
            out = TypeRegister.Categorical._from_maybe_non_unique_labels(
                self.cat._fa, out, base_index=self.cat.base_index
            )
            return out
        else:
            return where(self._isfiltered, out.inv, out[self.cat.ikey - 1])

    @classmethod
    def _build_method(cls, method):
        """
        General purpose factory for FAString function wrappers.
        """

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            out = method(self.fastring, *args, **kwargs)
            return self._convert_fastring_output(out)

        return wrapper

    @classmethod
    def _build_property(cls, name):
        """
        General purpose factory for FAString property wrappers.
        """

        def wrapper(self):
            return self._convert_fastring_output(getattr(self.fastring, name))

        return property(wrapper)

    def extract(self, regex: str, expand: Optional[bool] = None, fillna: str = "", names=None):
        out = self.fastring.extract(regex, expand=expand, fillna=fillna, names=names, apply_unique=False)
        if isinstance(out, TypeRegister.Dataset):
            return TypeRegister.Dataset({key: self._convert_fastring_output(col) for key, col in out.items()})
        else:
            return self._convert_fastring_output(out)

    extract.__doc__ = FAString.__doc__  # might be misleading since we drop the apply_unique argument

    @property
    def substr(self):
        return _SubStrAccessor(self)


# keep as last line
TypeRegister.FAString = FAString
TypeRegister.CatString = CatString
