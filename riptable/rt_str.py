__all__ = [
    'FAString'
]

import warnings
try:
    # This will be used to cache strlen for version of Python 3.7 and higher
    from functools import cached_property
except ImportError:
    cached_property = property

import re
import numpy as np
import numba as nb

from .config import get_global_settings
from .rt_fastarray import FastArray
from .rt_numpy import empty_like, empty, where, ones, zeros
from .rt_enum import TypeRegister


# NOTE YOU MUST INSTALL tbb
# conda install tbb
# to confirm...
# >>> from numba import threading_layer()
# >>> threading_layer()
# >>> 'tbb'
#
# NOTE this class was first written reshaping the array as 2d
#----
#if intype[1]=='S':
#    return self.view(np.uint8).reshape((len(self), self.itemsize))
#if intype[1]=='U':
#    return self.view(np.uint32).reshape((len(self), self.itemsize//4))
#---- 
#then pass to numba as 2d using shape[0] outer loop, shape[1] inner loop with [row,col] accessors
#---
#then flip back to string
#if self.itemsize == 4:
#    result=self.ravel().view('U'+str(self.shape[1]))
#else:
#    result=self.ravel().view('S'+str(self.shape[1]))
#
# Keeping it as 1d and remembering the itemsize is 10% faster on large arrays, and even faster on small arrays
# If recycling kicks in (only works on 1d arrays), it is 30% faster
# Also note: On Windows as of Aug 2019, prange is much slower without tbb installed
#
# subclass from FastArray
class FAString(FastArray):

    _APPLY_PARALLEL_THRESHOLD = 10_000

    def __new__(cls, arr, ikey=None, **kwargs):
        # if data comes in list like, convert to an array
        if not isinstance(arr, np.ndarray):
            if np.isscalar(arr):
                arr=np.asanyarray([arr])
            else:
                arr=np.asanyarray(arr)

        intype=arr.dtype.char
        if intype=='O':
            # try to convert to string (might have come from pandas)
            # default to unicode (note: FastArray attempts 'S' first)
            arr = arr.astype('U')
            intype=arr.dtype.char

        itemsize = np.int64(arr.itemsize)

        if intype=='S':
            # convert to two dim one byte array
            instance = arr.view(np.uint8)
        elif intype=='U':
            # convert to two dim four byte array
            instance = arr.view(np.uint32)
            itemsize = itemsize //4
        else:
            raise TypeError(f"FAString can only be used on byte string and unicode not {arr.dtype!r}")

        instance = instance.view(cls)

        # remember the intype and itemsize
        instance._intype=intype
        instance._itemsize =itemsize
        instance._ikey = ikey
        return instance

    # -----------------------------------------------------
    @property
    def str(self):
        # we already are a str
        return self

    # -----------------------------------------------------
    @property
    def backtostring(self):
        '''
        convert back to FastArray or np.ndarray 'S' or 'U' string
        'S12'  or 'U40'
        '''
        return self.view(self._intype + str(self._itemsize))

    @property
    def n_elements(self):
        """
        The number of elements in the original string array
        """
        return len(self) // self._itemsize

    # -----------------------------------------------------
    def possibly_convert_tostr(self, arr):
        '''
        converts list like or an array to the same string type
        '''

        # if data comes in list like, convert to an array
        if not isinstance(arr, np.ndarray):
            if not isinstance(arr, (list, tuple)):
                arr=np.asanyarray([arr])
            else: 
                arr=np.asanyarray(arr)

        if arr.dtype.char != self._intype:
            arr = arr.astype(self._intype)

        return arr

    def _maybe_output_to_categorical(self, out):
        if self._ikey is not None:
            from .rt_categorical import Categorical
            out = Categorical(self._ikey + 1, out, base_index=1)
            out.category_make_unique(inplace=True)
        return out

    # -----------------------------------------------------
    def _apply_func(self, func, funcp, *args, dtype=None, input=None, filtered_fill_value=None):
        # can optionally pass in dtype
        # check when to flip into parallel mode.  > 10,000 go to parallel routine
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

        # check for categorical key re-expansion
        if self._ikey is not None:
            if dest.dtype.kind == 'S':
                return self._maybe_output_to_categorical(dest)
            else:
                unfiltered = self._ikey >= 0
                return where(unfiltered, dest[self._ikey], filtered_fill_value)
        return dest

    # -----------------------------------------------------
    def apply(self, func, *args, dtype=None):
        '''
        Write your own string apply function
        NOTE: byte strings are passed as uint8
        NOTE: unicode strings are passed as uint32

        default signature must match

        @nb.jit(nopython=True, cache=True)
        def nb_upper(src, itemsize, dest):

        src: is uint array
        itemsize: is how wide the string is per row
        dest: is return uint array

        Other Parameters
        ----------------
        *args: pass in zero or more arguments (the arguments are always at the end)
        dtype: specify a different dtype

        Example:
        -------
        import numba as nb
        @nb.jit(nopython=True, cache=True)
        def nb_upper(src, itemsize, dest):
            for i in range(len(src) / itemsize):
                rowpos = i * itemsize
                for j in range(itemsize):
                    c=src[rowpos+j]
                    if c >= 97 and c <= 122:
                        # convert to ASCII upper
                        dest[rowpos+j] = c-32
                    else:
                        dest[rowpos+j] = c

        FAString(['this  ','that ','test']).apply(nb_upper)

        '''
        return self._apply_func(func, func, *args, dtype=dtype, input=self)

    # -----------------------------------------------------
    @nb.njit(cache=get_global_settings().enable_numba_cache)
    def nb_upper_inplace(src, itemsize):
        # loop over all rows
        for i in range(len(src) / itemsize):
            # loop over all chars in the string
            rowpos = i * itemsize
            # loop over all chars in the string
            for j in range(itemsize):
                c=src[rowpos+j]
                if c >= 97 and c <= 122:
                    # convert to ASCII upper
                    src[rowpos+j] = c-32

    # -----------------------------------------------------
    @nb.njit(cache=get_global_settings().enable_numba_cache)
    def nb_upper(src, itemsize, dest):
        # loop over all rows
        for i in range(len(src) / itemsize):
            rowpos = i * itemsize
            # loop over all chars in the string
            for j in range(itemsize):
                c=src[rowpos+j]
                if c >= 97 and c <= 122:
                    # convert to ASCII upper
                    dest[rowpos+j] = c-32
                else:
                    dest[rowpos+j] = c

    # -----------------------------------------------------
    @nb.njit(parallel=True, cache=get_global_settings().enable_numba_cache)
    def nb_pupper(src, itemsize, dest):
        # loop over all rows
        for i in nb.prange(np.int64(len(src) / itemsize)):
            rowpos = i * itemsize
            # loop over all chars in the string
            for j in range(itemsize):
                c=src[rowpos+j]
                if c >= 97 and c <= 122:
                    # convert to ASCII upper
                    dest[rowpos+j] = c-32
                else:
                    dest[rowpos+j] = c

    # -----------------------------------------------------
    @nb.njit(cache=get_global_settings().enable_numba_cache)
    def nb_lower(src, itemsize, dest):
        # loop over all rows
        for i in range(len(src) / itemsize):
            rowpos = i * itemsize
            # loop over all chars in the string
            for j in range(itemsize):
                c=src[rowpos+j]
                if c >= 65 and c <= 90:
                    # convert to ASCII lower
                    dest[rowpos+j] = c+32
                else:
                    dest[rowpos+j] = c

    # -----------------------------------------------------
    @nb.njit(parallel=True, cache=get_global_settings().enable_numba_cache)
    def nb_plower(src, itemsize, dest):
        # loop over all rows
        for i in nb.prange(len(src) / itemsize):
            rowpos = i * itemsize
            # loop over all chars in the string
            for j in range(itemsize):
                c=src[rowpos+j]
                if c >= 65 and c <= 90:
                    # convert to ASCII lower
                    dest[rowpos+j] = c+32
                else:
                    dest[rowpos+j] = c

    # -----------------------------------------------------
    @nb.njit(cache=get_global_settings().enable_numba_cache)
    def nb_removetrailing(src, itemsize, dest, removechar):
        # loop over all rows
        for i in range(len(src) / itemsize):
            # loop over all chars in the string backwards
            rowpos = i * itemsize
            startpos = itemsize
            while (startpos >0):
                startpos-=1
                c=src[rowpos + startpos]
                if c == 0 or c==removechar:
                    dest[rowpos + startpos] = 0
                else:
                    dest[rowpos + startpos] = c
                    break
            while (startpos >0):
                startpos-=1
                c=src[rowpos + startpos]
                dest[rowpos + startpos] = c


    # -----------------------------------------------------
    @nb.njit(cache=get_global_settings().enable_numba_cache)
    def nb_reverse_inplace(src, itemsize):
        # loop over all rows
        for i in range(len(src) / itemsize):
            rowpos = i * itemsize
            # find length of string
            strlen=0
            while (strlen < itemsize):
                if src[rowpos + strlen] ==0: break
                strlen +=1
            end = rowpos + strlen -1
            start = rowpos
            while (start < end):
                temp = src[end]
                src[end] = src[start]
                src[start] = temp
                start += 1
                end -= 1

    # -----------------------------------------------------
    @nb.njit(cache=get_global_settings().enable_numba_cache)
    def nb_reverse(src, itemsize, dest):
        # loop over all rows
        for i in range(len(src) / itemsize):
            rowpos = i * itemsize
            # find length of string
            strlen=0
            while (strlen < itemsize):
                if src[rowpos + strlen] ==0: break
                strlen +=1
            srcpos=0
            while(strlen > 0):
                strlen -= 1
                dest[rowpos + strlen] = src[rowpos + srcpos]
                srcpos += 1
            while(srcpos < itemsize):
                dest[rowpos + srcpos] = 0
                srcpos +=1


    # -----------------------------------------------------
    @nb.njit(cache=get_global_settings().enable_numba_cache)
    def nb_strlen(src, itemsize, dest):
        # loop over all rows
        for i in nb.prange(len(src) // itemsize):
            # loop over all chars in the string
            rowpos = i * itemsize
            strlen= 0
            # loop over all chars in the string
            for j in range(itemsize):
                if src[rowpos + j] == 0:
                    break
                strlen += 1
            # store length of string
            dest[i] = strlen

    nb_pstrlen = nb.njit(cache=True, parallel=True)(nb_strlen.py_func)

    # -----------------------------------------------------
    @nb.njit(cache=get_global_settings().enable_numba_cache)
    def nb_strpbrk(src, itemsize, dest, str2):
        str2len = len(str2)
        # loop over all rows
        for i in range(len(src) / itemsize):
            # loop over all chars in the string
            rowpos = i * itemsize
            found =0
            # loop over all chars in the string
            for j in range(itemsize):
                c= src[rowpos + j]
                for k in range(str2len):
                    if c==str2[k]:
                        # store location of match
                        dest[i] = j
                        found = 1
                        break
                if found == 1:
                    break
            if found == 0:
                dest[i] = -1

    # -----------------------------------------------------
    @nb.njit(cache=get_global_settings().enable_numba_cache)
    def nb_strstr(src, itemsize, dest, str2):
        str2len = len(str2)
        # loop over all rows
        for i in range(len(src) / itemsize):
            # loop over all chars in the string
            rowpos = i * itemsize
            dest[i] = -1
            # loop over all substrings of sufficient length
            for j in range(itemsize - str2len + 1):
                # check if enough space left
                k = 0
                while k < str2len:
                    if src[rowpos + j + k] != str2[k]:
                        break
                    k += 1
                if k == str2len:
                    # store location of match
                    dest[i] = j
                    break

    # -----------------------------------------------------
    @nb.njit(cache=get_global_settings().enable_numba_cache)
    def nb_contains(src, itemsize, dest, str2):
        str2len = len(str2)
        # loop over all rows
        for i in range(len(src) / itemsize):
            rowpos = i * itemsize
            dest[i] = False
            # loop over all substrings of sufficient length
            for j in range(itemsize - str2len + 1):
                k = 0
                while (k < str2len):
                    if src[rowpos + j + k] != str2[k]:
                        break
                    k += 1
                if k==str2len:
                    # indicate we have a match
                    dest[i] = True
                    break

    # -----------------------------------------------------
    @nb.njit(parallel=True, cache=get_global_settings().enable_numba_cache)
    def nb_pcontains(src, itemsize, dest, str2):
        str2len = len(str2)
        # loop over all rows
        for i in nb.prange(np.int64(len(src) / itemsize)):
            rowpos = i * itemsize
            dest[i] = False
            # loop over all substrings of sufficient length
            for j in range(itemsize - str2len + 1):
                k = 0
                while (k < str2len):
                    if src[rowpos + j + k] != str2[k]:
                        break
                    k += 1
                if k==str2len:
                    # indicate we have a match
                    dest[i] = True
                    break

    # -----------------------------------------------------
    @nb.njit(cache=get_global_settings().enable_numba_cache)
    def nb_endswith(src, itemsize, dest, str2):
        str2len = len(str2)
        # loop over all rows
        for i in range(len(src) / itemsize):
            # loop over all chars in the string
            rowpos = i * itemsize
            dest[i] = False

            # loop over all chars in the string
            # check if enough space left
            if itemsize >= str2len: 
                k =itemsize
                while ((k > 0) and (src[rowpos + k -1]==0)):
                    k -= 1

                # check if still enough space left
                if k >= str2len: 

                    k2=str2len
                    # check if only the end matches
                    while (k2 > 0):
                        if src[rowpos + k -1] != str2[k2-1]:
                            break
                        k -= 1
                        k2 -=1
                    if k2==0:
                        # indicate we have a match
                        dest[i] = True
                    
    # -----------------------------------------------------
    @nb.njit(cache=get_global_settings().enable_numba_cache)
    def nb_startswith(src, itemsize, dest, str2):
        str2len = len(str2)
        # loop over all rows
        for i in range(len(src) / itemsize):
            # loop over all chars in the string
            rowpos = i * itemsize
            dest[i] = False
            # loop over all chars in the string
            # check if enough space left
            if itemsize >= str2len: 
                k =0
                # check if only the beginning matches
                while (k < str2len):
                    if src[rowpos + k] != str2[k]:
                        break
                    k += 1
                if k==str2len:
                    # indicate we have a match
                    dest[i] = True

    # -----------------------------------------------------
    @nb.njit(parallel=True, cache=get_global_settings().enable_numba_cache)
    def nb_pstartswith(src, itemsize, dest, str2):
        str2len = len(str2)
        # loop over all rows
        for i in nb.prange(np.int64(len(src) / itemsize)):
            # loop over all chars in the string
            rowpos = i * itemsize
            dest[i] = False
            # loop over all chars in the string
            # check if enough space left
            if itemsize >= str2len: 
                k =0
                # check if only the beginning matches
                while (k < str2len):
                    if src[rowpos + k] != str2[k]:
                        break
                    k += 1
                if k==str2len:
                    # indicate we have a match
                    dest[i] = True

    # -----------------------------------------------------
    @property
    def upper(self):
        '''
        upper case a string (bytes or unicode)
        makes a copy

        Examples
        --------
        >>> FAString(['this','that','test']).upper
        FastArray(['THIS','THAT','TEST'], dtype='<U4')
        
        '''
        return self._apply_func(self.nb_upper, self.nb_pupper)

    # -----------------------------------------------------
    @property
    def lower(self):
        '''
        upper case a string (bytes or unicode)
        makes a copy

        Examples
        --------
        >>> FAString(['THIS','THAT','TEST']).lower
        FastArray(['this','that','test'], dtype='<U4')

        '''
        return self._apply_func(self.nb_lower, self.nb_lower)

    # -----------------------------------------------------
    @property
    def upper_inplace(self):
        '''
        upper case a string (bytes or unicode)
        does not make a copy

        Examples
        --------
        FAString(['this','that','test']).upper_inplace
        '''
        self.nb_upper_inplace(self._itemsize)
        return self.backtostring

    # -----------------------------------------------------
    @property
    def reverse(self):
        '''
        upper case a string (bytes or unicode)
        does not make a copy

        Examples
        --------
        FAString(['this','that','test']).reverse
        '''
        return self._apply_func(self.nb_reverse, self.nb_reverse)

    # -----------------------------------------------------
    @property
    def reverse_inplace(self):
        '''
        upper case a string (bytes or unicode)
        does not make a copy

        Examples
        --------
        FAString(['this','that','test']).reverse_inplace
        '''
        self.nb_reverse_inplace(self._itemsize)
        return self.backtostring

    # -----------------------------------------------------
    def removetrailing(self, remove=32):
        '''
        removes spaces at end of string (often to fixup matlab string)
        makes a copy

        Other Parameters
        ----------------
        remove=32.  defaults to removing ascii 32 (space character)

        Examples
        --------
        >>> FAString(['this  ','that ','test']).removetrailing()
        FastArray(['this','that','test'], dtype='<U6')

        '''
        return self._apply_func(self.nb_removetrailing, self.nb_removetrailing, remove)

    # -----------------------------------------------------
    @cached_property     # only cached for Python 3.7 or higher
    def strlen(self):
        '''
        return the string length of every string (bytes or unicode)

        Examples
        --------
        >>> FAString(['this  ','that ','test']).strlen
        FastArray([6, 5, 4])
        '''
        return self._apply_func(self.nb_strlen, self.nb_pstrlen, dtype=np.int32,
                                filtered_fill_value=np.iinfo(np.int32).min)

    # -----------------------------------------------------
    def strpbrk(self, str2):
        '''
        return the first index location any of the characters that are part of str2,
        or -1 if none of the characters match

        Parameters
        ----------
        str2 - a string with one or more characters to search for
        
        Examples
        --------
        >>> FAString(['this  ','that ','test']).strpbrk('ia')
        FastArray([2, 2, -1])
        '''
        if not isinstance(str2, FAString):
            str2 = self.possibly_convert_tostr(str2)
            if len(str2) != 1:
                return TypeError(f"A single string must be passed for str2 not {str2!r}")
            str2 = FAString(str2)
       
        return self._apply_func(self.nb_strpbrk, self.nb_strpbrk, str2, dtype=np.int32,
                                filtered_fill_value=np.iinfo(np.int32).min)

    # -----------------------------------------------------
    def strstr(self, str2):
        '''
        return the first index location of the entire substring specified in str2,
        or -1 if the substring does not exist

        Parameters
        ----------
        str2 - a string with one or more characters to search for
        
        Examples
        --------
        >>> FAString(['this  ','that ','test']).strstr('at')
        FastArray([-1, 2, -1])
        '''
        if not isinstance(str2, FAString):
            if str2 == '':
                return zeros(self.n_elements, dtype=np.int32)

            str2 = self.possibly_convert_tostr(str2)
            if len(str2) != 1:
                return TypeError(f"A single string must be passed for str2 not {str2!r}")
            str2 = FAString(str2)

        return self._apply_func(self.nb_strstr, self.nb_strstr, str2, dtype=np.int32,
                                filtered_fill_value=np.iinfo(np.int32).min)

    # -----------------------------------------------------
    def contains(self, str2):
        '''
        Return a boolean array where the value is set True if str2 is a substring of the element
        or False otherwise. Note this does not support regex like in Pandas.
        Please use regex_match for that.

        Parameters
        ----------
        str2 - a string with one or more characters to search for
        
        Examples
        --------
        >>> FAString(['this  ','that ','test']).contains('at')
        FastArray([False, True, False])
        '''
        if not isinstance(str2, FAString):
            if str2 == '':
                return ones(self.n_elements, dtype=bool)

            str2 = self.possibly_convert_tostr(str2)
            if len(str2) != 1:
                return TypeError(f"A single string must be passed for str2 not {str2!r}")
            str2 = FAString(str2)

        return self._apply_func(self.nb_contains, self.nb_pcontains, str2, dtype=np.bool,
                                filtered_fill_value=False)

    def strstrb(self, str2):
        """
        Deprecated. Please see .contains.
        """
        warnings.warn("strstrb is now deprecated and has been renamed to `contains`", DeprecationWarning)
        return self.contains(str2)

    # -----------------------------------------------------
    def startswith(self, str2):
        '''
        return a boolean array where the value is set True if the string starts with the entire substring specified in str2,
        or False if the substring does not start with the entire substring

        Parameters
        ----------
        str2 - a string with one or more characters to search for
        
        Examples
        --------
        >>> FAString(['this  ','that ','test']).startswith('thi')
        FastArray([True, False, False])
        '''
        if not isinstance(str2, FAString):
            if str2 == '':
                return ones(self.n_elements, dtype=bool)

            str2 = self.possibly_convert_tostr(str2)
            if len(str2) != 1:
                return TypeError(f"A single string must be passed for str2 not {str2!r}")
            str2 = FAString(str2)
       
        return self._apply_func(self.nb_startswith, self.nb_pstartswith, str2, dtype=np.bool,
                                filtered_fill_value=False)

    # -----------------------------------------------------
    def endswith(self, str2):
        '''
        return a boolean array where the value is set True if the string ends with the entire substring specified in str2,
        or False if the substring does not end with the entire substring

        Parameters
        ----------
        str2 - a string with one or more characters to search for
        
        Examples
        --------
        >>> FAString(['abab','ababa','abababb']).endswith('ab')
        FastArray([True, False, False])
        '''
        if not isinstance(str2, FAString):
            if str2 == '':
                return ones(self.n_elements, dtype=bool)

            str2 = self.possibly_convert_tostr(str2)
            if len(str2) != 1:
                return TypeError(f"A single string must be passed for str2 not {str2!r}")
            str2 = FAString(str2)

        return self._apply_func(self.nb_endswith, self.nb_endswith, str2, dtype=np.bool,
                                filtered_fill_value=False)

    def regex_match(self, regex):
        '''
        Return a Boolean array where the value is set True if the string contains str2.
        str2 may be a normal string or a regular expression.
        Applies re.search on each element with str2 as the pattern.

        Parameters
        ----------
        regex - Perform element-wise regex matching to this regex

        Examples
        --------
        >>> FAString(['abab','ababa','abababb']).regex_match('ab$')
        FastArray([True, False, False])
        '''
        if not isinstance(regex, bytes):
            regex = bytes(regex, 'utf-8')
        regex = re.compile(regex)
        vmatch = np.vectorize(lambda x: bool(regex.search(x)))
        bools = vmatch(self.backtostring)
        if self._ikey is not None:
            bools = bools[self._ikey] & (self._ikey >= 0)
        return bools

    @staticmethod
    @nb.njit(parallel=True)
    def _nb_substr(src, out, itemsize, start, stop, strlen):
        n_elements = len(out)
        max_chars = 0
        for elem in nb.prange(n_elements):
            elem_len = strlen[elem]
            i, j = start, stop
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

    @cached_property
    def _cat_strlen(self):
        """
        Same as strlen except for Categoricals it is aligned with the categories
        as opposed to the full array. Used for substring methods.
        """
        if self._ikey is None:
            return self.strlen
        else:
            return FAString(self.backtostring).strlen

    def substr(self, start, stop=None):
        """
        Take a substring of each element using slice args.
        """
        if stop is None:
            stop = self._itemsize

        strlen = self._cat_strlen

        if start < 0:
            if stop < 0:
                n_chars = stop - start
            else:
                n_chars = stop  # we can't tell what the max length at this point
        elif stop < 0:
            pos_stop = self._itemsize - stop
            n_chars = pos_stop - start
        else:
            n_chars = stop - start
        out = zeros((self.n_elements, n_chars), self.dtype)
        out = self._nb_substr(self, out, self._itemsize, start, stop, strlen)
        n_chars = out.shape[1]
        if n_chars == 0:    # empty sub strings everywhere
            out = zeros(self.n_elements, self.dtype).view(f'{self._intype}1')
        else:
            out = out.ravel().view(f'<{self._intype}{n_chars}')
        out = self._maybe_output_to_categorical(out)
        return out

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def _nb_char(src, position, itemsize, strlen, out):
        broken_at = -1
        for i in nb.prange(len(position)):
            pos = position[i]
            if pos < 0:
                pos = strlen[i] + pos
            if pos >= itemsize or pos < 0:
                return i    # this triggers error below
            out[i] = src[itemsize * i + pos]

        return broken_at

    def char(self, position):
        """
        Take a single character from each element.

        Parameters
        ----------
        position: int / np.ndarray
            The position of the character to be extracted. Negative values respect the
            length of the individual strings.
            If an array, the length must be equal to the number of strings.
            An error is raised if any positions are out of bounds (>= self._itemsize).
        """
        if np.ndim(position) == 0:
            for size in [8, 16, 32, 64]:
                dtype = getattr(np, f'uint{size}')
                if self._itemsize <= np.iinfo(dtype).max:
                    break
            position = ones(self.n_elements, dtype) * position

        if len(position) != self.n_elements:
            raise ValueError("position must be a scalar or a vector of the same length as self")

        out = zeros(self.n_elements, self.dtype)
        strlen = self._cat_strlen
        broken_at = self._nb_char(self, position, self._itemsize, strlen, out)
        if broken_at >= 0:
            raise ValueError(f"Position {position[broken_at]} out of bounds "
                             f"for string of length {self._itemsize}")
        out = out.view(f'{self._intype}1')
        return self._maybe_output_to_categorical(out)


# keep as last line
TypeRegister.FAString=FAString
