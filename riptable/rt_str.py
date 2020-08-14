__all__ = ['FAString', ]

import numpy as np
import numba as nb
from .rt_fastarray import FastArray
from .rt_numpy import empty_like, empty
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

    # -----------------------------------------------------
    def _apply_func(self, func, funcp, *args, dtype=None, input=None):
        # can optionally pass in dtype
        # check when to flip into parallel mode.  > 10,000 go to parallel routine
        if len(self) > 10_000 and funcp is not None:
            func = funcp
        if dtype is None:
            dest = empty(len(self), self.dtype)
            dest._itemsize = self._itemsize
            dest._intype = self._intype
        else:
            # user requested specific output dtype
            arrlen = len(self) // self._itemsize
            dest = empty(arrlen, dtype)

        if input is None:
            func(self._itemsize, dest, *args)
        else:
            if not isinstance(input, FAString):
                raise TypeError(f"The input= value was not a FAString.  It is {input!r}.")
            func(input, input._itemsize, dest, *args)

        if dtype is None:
            dest =dest.view(dest._intype + str(dest._itemsize))

        # check for categorical key re-expansion
        if self._ikey is not None:
            return dest[self._ikey]
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
    @nb.jit(nopython=True, cache=True)
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
    @nb.jit(nopython=True, cache=True)
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
    @nb.jit(parallel=True, nopython=True, cache=True)
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
    @nb.jit(nopython=True, cache=True)
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
    @nb.jit(parallel=True, nopython=True, cache=True)
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
    @nb.jit(nopython=True, cache=True)
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
    @nb.jit(nopython=True, cache=True)
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
    @nb.jit( cache=True)
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
    @nb.jit(nopython=True, cache=True)
    def nb_strlen(src, itemsize, dest):
        # loop over all rows
        for i in range(len(src) / itemsize):
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

    # -----------------------------------------------------
    @nb.jit(nopython=True, cache=True)
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
    @nb.jit(nopython=True, cache=True)
    def nb_strstr(src, itemsize, dest, str2):
        str2len = len(str2)
        # loop over all rows
        for i in range(len(src) / itemsize):
            # loop over all chars in the string
            rowpos = i * itemsize
            dest[i] = -1
            # loop over all chars in the string
            for j in range(itemsize):
                # check if enough space left
                if (itemsize - j) < str2len: 
                    break
                c= src[rowpos + j]
                k =0
                while (k < str2len):
                    if src[rowpos + j + k] != str2[k]:
                        break
                    k += 1
                if k==str2len:
                    # store location of match
                    dest[i] = j
                    break

    # -----------------------------------------------------
    @nb.jit(nopython=True, cache=True)
    def nb_strstrb(src, itemsize, dest, str2):
        str2len = len(str2)
        # loop over all rows
        for i in range(len(src) / itemsize):
            rowpos = i * itemsize
            dest[i] = False
            # loop over all chars in the string
            for j in range(itemsize):
                # check if enough space left
                if (itemsize - j) < str2len: 
                    break
                k =0
                while (k < str2len):
                    if src[rowpos + j + k] != str2[k]:
                        break
                    k += 1
                if k==str2len:
                    # indicate we have a match
                    dest[i] = True
                    break

    # -----------------------------------------------------
    @nb.jit(parallel=True, nopython=True, cache=True)
    def nb_pstrstrb(src, itemsize, dest, str2):
        str2len = len(str2)
        # loop over all rows
        for i in nb.prange(np.int64(len(src) / itemsize)):
            rowpos = i * itemsize
            dest[i] = False
            # loop over all chars in the string
            for j in range(itemsize):
                # check if enough space left
                if (itemsize - j) < str2len: 
                    break
                k =0
                while (k < str2len):
                    if src[rowpos + j + k] != str2[k]:
                        break
                    k += 1
                if k==str2len:
                    # indicate we have a match
                    dest[i] = True
                    break

    # -----------------------------------------------------
    @nb.jit(nopython=True, cache=True)
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
    @nb.jit(nopython=True, cache=True)
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
    @nb.jit(parallel=True, nopython=True, cache=True)
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
        return self.backtostring

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
    @property
    def strlen(self):
        '''
        return the string length of every string (bytes or unicode)

        Examples
        --------
        >>> FAString(['this  ','that ','test']).strlen
        FastArray([6, 5, 4])
        '''
        return self._apply_func(self.nb_strlen, self.nb_strlen, dtype=np.int32)

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
       
        return self._apply_func(self.nb_strpbrk, self.nb_strpbrk, str2, dtype=np.int32)


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
            str2 = self.possibly_convert_tostr(str2)
            if len(str2) != 1:
                return TypeError(f"A single string must be passed for str2 not {str2!r}")
            str2 = FAString(str2)
       
        return self._apply_func(self.nb_strstr, self.nb_strstr, str2, dtype=np.int32)

    # -----------------------------------------------------
    def strstrb(self, str2):
        '''
        return a boolean array where the value is set True if the first index location of the entire substring specified in str2,
        or False if the substring does not exist

        Parameters
        ----------
        str2 - a string with one or more characters to search for
        
        Examples
        --------
        >>> FAString(['this  ','that ','test']).strstrb('at')
        FastArray([False, True, False])
        '''
        if not isinstance(str2, FAString):
            str2 = self.possibly_convert_tostr(str2)
            if len(str2) != 1:
                return TypeError(f"A single string must be passed for str2 not {str2!r}")
            str2 = FAString(str2)
       
        return self._apply_func(self.nb_strstrb, self.nb_pstrstrb, str2, dtype=np.bool)

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
            str2 = self.possibly_convert_tostr(str2)
            if len(str2) != 1:
                return TypeError(f"A single string must be passed for str2 not {str2!r}")
            str2 = FAString(str2)
       
        return self._apply_func(self.nb_startswith, self.nb_pstartswith, str2, dtype=np.bool)

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
            str2 = self.possibly_convert_tostr(str2)
            if len(str2) != 1:
                return TypeError(f"A single string must be passed for str2 not {str2!r}")
            str2 = FAString(str2)

        return self._apply_func(self.nb_endswith, self.nb_endswith, str2, dtype=np.bool)


# keep as last line
TypeRegister.FAString=FAString
