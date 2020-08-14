__all__ = ['FastArray', 'Threading', 'Recycle','Ledger']

import numpy as np
import warnings
import os

from numpy.core.numeric import ScalarType

import riptide_cpp as rc
from .rt_enum import gBinaryUFuncs, gBinaryLogicalUFuncs, gBinaryBitwiseUFuncs, gBinaryBitwiseMonoUFuncs, gUnaryUFuncs, gReduceUFuncs
from .rt_enum import TypeRegister, ROLLING_FUNCTIONS, TIMEWINDOW_FUNCTIONS, REDUCE_FUNCTIONS, gNumpyScalarType, DisplayLength, NumpyCharTypes, MATH_OPERATION, INVALID_DICT
from .Utils.rt_display_properties import ItemFormat, DisplayConvert, default_item_formats
from .rt_mlutils import normalize_minmax, normalize_zscore
from .rt_numpy import ismember, ones, unique, sort, full, empty, empty_like, searchsorted, _searchsorted, bool_to_fancy, issorted, repeat, tile, where, groupbyhash, asanyarray
from .rt_sds import save_sds
from .rt_utils import  sample, describe
from .rt_grouping import Grouping
try:
    # optional extra routines if bottleneck installed
    import bottleneck as bn
except Exception:
    pass

NUMPY_CONVERSION_TABLE = {
    np.sum: REDUCE_FUNCTIONS.REDUCE_SUM,
    np.nansum: REDUCE_FUNCTIONS.REDUCE_NANSUM,
    np.amin: REDUCE_FUNCTIONS.REDUCE_MIN,
    np.nanmin: REDUCE_FUNCTIONS.REDUCE_NANMIN,
    np.amax: REDUCE_FUNCTIONS.REDUCE_MAX,
    np.nanmax: REDUCE_FUNCTIONS.REDUCE_NANMAX,
    np.var: REDUCE_FUNCTIONS.REDUCE_VAR,
    np.nanvar: REDUCE_FUNCTIONS.REDUCE_NANVAR,
    np.mean: REDUCE_FUNCTIONS.REDUCE_MEAN,
    np.nanmean: REDUCE_FUNCTIONS.REDUCE_NANMEAN,
    np.std: REDUCE_FUNCTIONS.REDUCE_STD,
    np.nanstd: REDUCE_FUNCTIONS.REDUCE_NANSTD,
    np.argmin: REDUCE_FUNCTIONS.REDUCE_ARGMIN,
    np.nanargmin: REDUCE_FUNCTIONS.REDUCE_NANARGMIN,
    np.argmax: REDUCE_FUNCTIONS.REDUCE_ARGMAX,
    np.nanargmax: REDUCE_FUNCTIONS.REDUCE_NANARGMAX,
    # np.any: REDUCE_FUNCTIONS.REDUCE_ANY,
    # np.all: REDUCE_FUNCTIONS.REDUCE_ALL,
    }

#--------------------------------------------------------------
def FA_FROM_UINT8(uint8arr):
    '''
    Used in de-pickling
    '''
    return rc.CompressDecompressArrays([uint8arr],1)[0]

#--------------------------------------------------------------
def FA_FROM_BYTESTRING(bytestring):
    '''
    Used in de-pickling when tostring() used (currently disabled)
    '''
    return FA_FROM_UINT8(np.frombuffer(bytestring, dtype=np.uint8))

#--------------------------------------------------------------
def logical_find_common_type(arraytypes,scalartypes, scalarval):
    '''
    assumes one scalar and one array

    '''
    scalar = scalartypes[0]
    array = arraytypes[0]

    unsigned = False
    isinteger = False

    # TJD this routine needs to be rewritten
    # can check isinstance(scalar,(np.integer, int))

    # if this comes in as np.int64 and not a dtype, we convert to a dtype
    if not hasattr(scalar, 'char'):
        scalar = np.dtype(scalar)

    if scalar.char in NumpyCharTypes.UnsignedInteger:
        unsigned = True
        isinteger = True
    if scalar.char in NumpyCharTypes.Integer:
        isinteger = True

    if not isinteger:
        # go by numpy upscale rules
        # NOTE: should consider allowing integer ^ True -- or changing a bool scalar to an int
        #print("punting not integer scalar", scalar)
        return np.find_common_type(arraytypes,scalartypes)

    unsigned = False
    isinteger = False

    try:
        if array.char in NumpyCharTypes.UnsignedInteger:
            unsigned = True
            isinteger = True
        if array.char in NumpyCharTypes.Integer:
            isinteger = True
    except:
        pass

    #if isinstance(array, int):
    #    isinteger = True

    # IF ARRAY IS UNSIGNED BY SCALAR IS SIGNED upcast

    if not isinteger:
        # go by numpy upscale rules
        # NOTE: should consider allowing integer ^ True -- or changing a bool scalar to an int
        #print("punting not integer array", array)
        return np.find_common_type(arraytypes,scalartypes)

    final = None

    scalarval=int(scalarval)

    # Determine the possible integer upscaling based on the scalar value
    if unsigned:
        if scalarval <= 255: final=np.uint8
        elif scalarval <= 65535: final=np.uint16
        elif scalarval <= (2**32-1): final=np.uint32
        elif scalarval <= (2**64-1): final=np.uint64
        else: final=np.float64
    else:
        if scalarval >= -128 and scalarval <= 127: final=np.int8
        elif scalarval >= -32768 and scalarval <= 32767: final=np.int16
        elif scalarval >= -(2**31) and scalarval <= (2**31-1): final=np.int32
        elif scalarval >= -(2**63) and scalarval <= (2**63-1): final=np.int64
        else: final=np.float64

    final = np.dtype(final)

    # do not allow downcasting
    if array.num < final.num:
        #print("returning final", final)
        return final

    return array

    #if type(args[0]) in ScalarType:
    #    print("converting arg2 to ", final_dtype)
    #    args[1] = args[1].astype(final_dtype);
    #else:
    #    print("converting arg1 to ", final_dtype)
    #    args[0] = args[0].astype(final_dtype);

#--------------------------------------------------------------
def _ASTYPE(self, dtype):
    ''' internal call from array_ufunc to convert arrays.  returns numpy arrays'''
    #return self.astype(dtype)
    to_num = dtype.num
    if self.dtype.num <= 13 and to_num <= 13:
        if FastArray.SafeConversions:
            # perform a safe conversion understanding sentinels
            return TypeRegister.MathLedger._AS_FA_TYPE(self, to_num)._np
        else:
            # perform unsafe conversion NOT understanding sentinels
            return TypeRegister.MathLedger._AS_FA_TYPE_UNSAFE(self, to_num)._np

    return self.astype(dtype)

#--------------------------------------------------------------
#--------------------------------------------------------------
class FastArray(np.ndarray):
    '''
    Class FastArray
    replaces a numpy array for 1 dimensional arrays
    arrays with more than 1 dimension are often punted back to numpy for calculations

    example usage:
       arr = FastArray([1,2,3,4,5])
       arr = FastArray(np.arange(100))
       arr = FastArray(list('abc'), unicode=True)

    to flip an existing numpy array such as nparray use the view method
       fa = nparray.view(FastArray)

    to change it back
       fa.view(np.ndarray) or fa._np

    FastArray will take over many numpy ufuncs, can recycle arrays, and use multiple threads

    How to subclass FastArray:
    --------------------------
    Required class definition:

    class TestSubclass(FastArray):
        def __new__(cls, arr, **args):
            # before arr this call, arr needs to be a np.ndarray instance
            return arr.view(cls)

        def __init__(self, arr, **args):
            pass

    If the subclass is computable, you might define your own math operations.
    In these operations, you might define what the subclass can be computed with. DateTimeNano is a good example.
    Common operations to hook are comparisons:
    __eq__(), __ne__(), __gt__(), __lt__(), __le__(), __ge__()
    Basic math functions:
    __add__(), __sub__(), __mul__(), etc.

    Bracket indexing operations are very common. If the subclass needs to set or return a value
    other than that in the underlying array, you need to take over:
    __getitem__(), __setitem__()

    Indexing is also used in display.
    For regular console/notebook display, you need to take over:
    __repr__():
    >>> arr
    __str__():
    >>> print(arr)
    _repr_html_() *for Jupyter Lab/Notebook

    If the array is being displayed in a Dataset, and you require certain formatting you need to define two more methods:
    display_query_properties() - returns an ItemFormat object (see Utils.rt_display_properties), and a conversion function
    display_convert_func() - the conversion function returned by display_query_properties(), must return a string. each item being
                            displayed will go through this function individually, accompanied by an ItemFormat object.
                            The item going through this is the result of __getitem__() at a single index.

    Many riptable operations need to return arrays of the same class they received. To ensure that your
    subclass will retain its special properties, you need to take over newclassfrominstance().
    Failure to take this over will often result in an object with uninitialized variables.

    copy() is another method that is called generically in riptable routines, and needs to be taken
    over to retain subclass properties.

    For a view of the underlying FastArray, you can use the _fa property.

    TODO: Need more text
    '''
    # Defines a generic np.ndarray subclass, that can cache numpy arrays
    # Static Class VARIABLES

    # change this to show or less values on __repr__
    MAX_DISPLAY_LEN = 10

    # set to 2 or 3 for extra debug information
    Verbose = 1

    # set to true for reusing numpy arrays instead of deleting them completely
    Recycle = True

    # set to true to preserve sentinels during internal array_ufunc calculations
    SafeConversions = True

    #set to false to be just normal numpy
    FasterUFunc = True

    # 0=Quiet, 1=Warn, 2=Exception
    WarningLevel = 1

    # set to true to not allow ararys we do not support
    NoTolerance = False

    # set to false to not compress when pickling
    CompressPickle = True

    # a dictionary to avoid repeating warnings in multiple places
    # TODO: wrap this in a class so that warnings can be turned on/off
    WarningDict = {
        "multiple_dimensions" : "FastArray contains two or more dimensions greater than one - shape:{}.  Problems may occur."
    }

    #--------------------------------------------------------------------------
    @classmethod
    def _possibly_warn(cls, warning_string):
        if cls.WarningLevel ==0:
            return False
        if cls.WarningLevel ==1:
            warnings.warn(warning_string)
            return True
        raise TypeError(warning_string)

    #--------------------------------------------------------------------------
    def __new__(cls,arr,**kwargs):

        allow_unicode = kwargs.get('unicode', False)
        try:
            del kwargs['unicode']
        except:
            pass
        # If already a numpy array no need to call asany
        if isinstance(arr, np.ndarray) and len(kwargs) == 0:
            instance = arr
            if isinstance(instance, cls) and instance.dtype.char != 'U':
                if instance.dtype.char not in NumpyCharTypes.Supported:
                    cls._possibly_warn(f"FastArray contains an unsupported type '{instance.dtype}'.  Problems may occur.  Consider categoricals.")
                # if already a FastArray, do not rewrap this
                return instance
        else:
            # flip the list or other object to a numpy array
            instance= np.asanyarray(arr,**kwargs)

        if not allow_unicode and instance.dtype.char == 'U':
            try:
                instance = np.asarray(instance, dtype='S')
            except:
                pass

        if len(instance.shape) == 0:
            if instance.dtype.char in NumpyCharTypes.Supported:
                instance = np.asanyarray([instance], **kwargs)
            else:
                # np.asarray on a set will return an object of 1
                if isinstance(arr, set):
                    instance = np.asarray(list(arr), **kwargs)
                else:
                    raise TypeError(f"FastArray cannot initialize {arr}")

        if instance.ndim > 1:
            # only one dimension can be greater than one
            if cls._check_ndim(instance) > 1:
                cls._possibly_warn(FastArray.WarningDict["multiple_dimensions"].format(instance.shape))
                #warnings.warn(f"FastArray contains two or more dimensions greater than one - shape:{instance.shape}.  Problems may occur.")
        elif not (instance.flags.f_contiguous or instance.flags.c_contiguous):
                # copy should eliminate strides problem
                instance=instance.copy()
                cls._possibly_warn(f"FastArray initialized with strides.")

        # for arrays that can cause problems but we allow now
        if cls.NoTolerance:
            if not (instance.flags.f_contiguous or instance.flags.c_contiguous):
                # copy should eliminate strides problem
                instance=instance.copy()
                cls._possibly_warn(f"FastArray initialized with strides.")

        if instance.dtype.char not in NumpyCharTypes.Supported:
            cls._possibly_warn(f"FastArray contains an unsupported type '{instance.dtype}'.  Problems may occur.  Consider categoricals.")

        return instance.view(cls)


    #--------------------------------------------------------------------------
    def __reduce__(self):
        '''
        Used for pickling.
        For just a FastArray we pass back the view of the np.ndarray, which then knows how to pickle itself.
        NOTE: I think there is a faster way.. possible returning a byte string.
        '''
        cls = type(self)

        # check if subclassed routine knows how to serialize itself
        if hasattr(self, '_build_sds_meta_data'):
            try:
                name=self._name
            except:
                name='unknown'

            tups =self._build_sds_meta_data(name)
            return (cls._load_from_sds_meta_data, (name, self.view(FastArray), tups[1], tups[0]))

        # set to true to turn compression on
        if cls.CompressPickle and len(self) > 0:
            # create a single compressed array of uint8
            carr = rc .CompressDecompressArrays([self], 0)[0]
            return (FA_FROM_UINT8, (carr.view(np.ndarray),))
        else:
            return (cls.__new__, (cls, self.view(np.ndarray),))

    #--------------------------------------------------------------------------
    @classmethod
    def _check_ndim(cls, instance):
        '''
        Iterates through dimensions of an array, counting how many dimensions have values greater than 1.
        Problems may occure with multidimensional FastArrays, and the user will be warned.
        '''
        index=0
        aboveone=0
        while index < instance.ndim:
            if instance.shape[index] > 1:
                aboveone+=1
            index += 1
        return aboveone

    #--------------------------------------------------------------------------
    def get_name(self):
        '''
        FastArray can hold a name.  When a Dataset puts a FastArray into a column, it may receive a name.

        Returns
        -------
        The array name or if the name does not exist, None is retuned.

        See Also
        --------
        set_name
        '''
        name=None
        try:
            name=self._name
        except:
            pass
        return name

    #--------------------------------------------------------------------------
    def set_name(self, name):
        '''
        FastArray can hold a name.  Use set_name to assign a name.
        When a Dataset puts a FastArray into a named column, it may call set_name().
        If the same FastArray is in two datasets, with two different column names,
        another FastArray wrapper object will be created to hold the different name,
        however the underlying array will remain the same in both datasets.

        Returns
        -------
        FastArray (or subclass)

        Examples
        --------
        >>> a=rt.arange(100)
        >>> a.set_name('test')

        See Also
        --------
        FastArray.get_name
        '''
        self._name = name
        return self

    #--------------------------------------------------------------------------
    @staticmethod
    def _FastFunctionsOn():
        if FastArray.Verbose > 0: print(f"FASTFUNC ON: fastfunc was {FastArray.FasterUFunc}")
        FastArray.FasterUFunc = True

    @staticmethod
    def _FastFunctionsOff():
        if FastArray.Verbose > 0: print(f"FASTFUNC OFF: fastfunc was {FastArray.FasterUFunc}")
        FastArray.FasterUFunc = False

    @property
    def _np(self):
        '''
        quick way to return a numpy array instead of fast array
        '''
        return self.view(np.ndarray)

    @staticmethod
    def _V0():
        print("setting verbose level to 0")
        FastArray.Verbose=0
        return FastArray.Verbose

    @staticmethod
    def _V1():
        print("setting verbose level to 1")
        FastArray.Verbose=1
        return FastArray.Verbose

    @staticmethod
    def _V2():
        print("setting verbose level to 2")
        FastArray.Verbose=2
        return FastArray.Verbose

    @staticmethod
    def _ON():
        '''
        enable intercepting array ufunc
        '''
        return FastArray._FastFunctionsOn()

    @staticmethod
    def _OFF():
        '''
        disable intercepting of array ufunc
        '''
        return FastArray._FastFunctionsOff()

    @staticmethod
    def _TON():
        print("Threading on")
        return rc.ThreadingMode(0)

    @staticmethod
    def _TOFF():
        print("Threading off")
        return rc.ThreadingMode(1)

    @staticmethod
    def _RON(quiet=False):
        '''
        Turn on recycling.

        Parameters
        ----------
        quiet: bool, optional

        Returns
        -------
        True if recycling was previously on, else False
        '''
        if not quiet:
            print("Recycling numpy arrays on")
        result = rc.SetRecycleMode(0)
        FastArray.Recycle=True
        return result

    @staticmethod
    def _ROFF(quiet=False):
        '''
        Turn off recycling.

        Parameters
        ----------
        quiet: bool, optional

        Returns
        -------
        True if recycling was previously on, else False
        '''
        if not quiet:
            print("Recycling numpy arrays off")
        result = rc.SetRecycleMode(1)
        FastArray.Recycle=False
        return result

    @staticmethod
    def _RDUMP():
        '''
        Displays to server's stdout

        Returns
        -------
        Total size of items not in use
        '''
        return rc.RecycleDump()

    @staticmethod
    def _GCNOW(timeout:int = 0):
        '''
        Pass the garbage collector timeout value to cleanup.
        Passing 0 will force an immediate garbage collection.

        Returns
        -------
        Dictionary of memory heuristics including 'TotalDeleted'
        '''
        import gc
        gc.collect()
        result= rc.RecycleGarbageCollectNow(timeout)
        totalDeleted = result['TotalDeleted']
        if totalDeleted > 0:
            FastArray._GCNOW(timeout)
        return result

    @staticmethod
    def _GCSET(timeout:int = 100):
        '''
        Pass the garbage collector timeout value to expire
        The timeout value is roughly in 2/5 secs
        A value of 100 is usually about 40 seconds

        Returns
        -------
        Previous timespan
        '''
        return rc.RecycleSetGarbageCollectTimeout(timeout)

    @staticmethod
    def _LON():
        '''Turn the math ledger on to record all array math routines'''
        return TypeRegister.MathLedger._LedgerOn()

    @staticmethod
    def _LOFF():
        '''Turn the math ledger off'''
        return TypeRegister.MathLedger._LedgerOff()

    @staticmethod
    def _LDUMP(dataset=True):
        '''Print out the math ledger'''
        return TypeRegister.MathLedger._LedgerDump(dataset=dataset)

    @staticmethod
    def _LDUMPF(filename):
        '''Save the math ledger to a file'''
        return TypeRegister.MathLedger._LedgerDumpFile(filename)

    @staticmethod
    def _LCLEAR():
        '''Clear all the entries in the math ledger'''
        return TypeRegister.MathLedger._LedgerClear()


    #--------------------------------------------------------------------------
    def __setitem__(self, fld, value):
        """
        Used on the left hand side of
        arr[fld] = value

        This routine tries to convert invalid dtypes to that invalids are preserved when setting
        The mbset portion of this is no written (which will not raise an indexerror on out of bounds)

        Parameters
        ----------
        fld: scalar, boolean, fancy index mask, slice, sequence, or list
        value: scalar, sequence or dataset value as follows
                sequence can be list, tuple, np.ndarray, FastArray

        Raises
        -------
        IndexError

        """
        newvalue = None

        # try to make an array, even if array of 1
        if np.isscalar(value):
           if not isinstance(value,(str, bytes, np.bytes_, np.str_)):
                # convert to array of 1 item
                newvalue = FastArray([value])
        elif isinstance(value, (list, tuple)):
            # convert to numpy array
            newvalue = FastArray(value, unicode=True)
        elif isinstance(value, np.ndarray):
            # just reference it
            newvalue = value

        if newvalue is not None:

            # now we have a numpy array.. convert the dtype to match us
            # this should take care of invalids
            # convert first 14 common types (bool, ints, floats)
            if newvalue.dtype != self.dtype and newvalue.dtype.num <= 13:
                newvalue = newvalue.astype(self.dtype)

            # check for boolean array since we do not handle fancy index yet
            if isinstance(fld, np.ndarray) and fld.dtype.num ==0:
                if self._is_not_supported(newvalue):
                    # make it contiguous
                    newvalue = newvalue.copy()

                # call our setitem, it will return False if it fails
                if rc.SetItem(self, fld, newvalue):
                    return
            try:
                np.ndarray.__setitem__(self, fld, newvalue)
            except Exception:
                # odd ball cases handled here like ufunc tests
                np.ndarray.__setitem__(self, fld, value)
            return

        # punt to normal numpy
        np.ndarray.__setitem__(self, fld, value)

    #--------------------------------------------------------------------------
    def __getitem__(self, fld):
        '''
        riptable has special routines to handle array input in the indexer.
        Everything else will go to numpy getitem.
        '''
        if isinstance(fld, np.ndarray):
            #result= super(FastArray, self).__getitem__(fld).view(FastArray)
            if fld.dtype == np.bool:
                # make sure no striding
                # NOTE: will fail on self.dtype.byteorder as little endian
                if self.flags.f_contiguous:
                    return TypeRegister.MathLedger._INDEX_BOOL(self,fld)

            # if we have fancy indexing and we support the array type, make sure we do not have stride problem
            if fld.dtype.char in NumpyCharTypes.AllInteger and self.dtype.char in NumpyCharTypes.SupportedAlternate:
                if self.flags.f_contiguous and fld.flags.f_contiguous:
                    if len(self.shape) ==1:
                        return TypeRegister.MathLedger._MBGET(self,fld)

            result= TypeRegister.MathLedger._GETITEM(super(FastArray, self),fld)
            return result.view(FastArray)
        else:
            # could be a list which is often converted to an array

            # This assumes that FastArray has a sole parent, np.ndarray
            # If this changes, the super() call needs to be used
            return np.ndarray.__getitem__(self, fld)
            #return super(FastArray, self).__getitem__(fld)

    #--------------------------------------------------------------------------
    def display_query_properties(self):
        '''
        Returns an ItemFormat object and a function for converting the FastArrays items to strings.
        Basic types: Bool, Int, Float, Bytes, String all have default formats / conversion functions.
        (see Utils.rt_display_properties)

        If a new type is a subclass of FastArray and needs to be displayed in format
        different from its underlying type, it will need to take over this routine.
        '''
        arr_type, convert_func = DisplayConvert.get_display_convert(self)
        display_format = default_item_formats.get(arr_type, ItemFormat())
        if len(self.shape) > 1:
            display_format.convert = convert_func
            convert_func = DisplayConvert.convertMultiDims

        # add sentinel value for integer
        if display_format.invalid is None:
            display_format = display_format.copy()
            if self.dtype.char in NumpyCharTypes.AllInteger:
                display_format.invalid = INVALID_DICT[self.dtype.num]
        return display_format, convert_func

    #--------------------------------------------------------------------------
    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        #result= super(FastArray, self).astype(dtype, order,casting,subok,copy)
        # 17 is object
        # 18 = ASCII string
        # 19 = UNICODE string
        to_num=np.dtype(dtype).num

        # check for contiguous in one or two dimensions
        if self.flags.f_contiguous or self.flags.c_contiguous:
            if order=='K' and subok and copy and self.dtype.num <= 13 and to_num <= 13:
                # perform a safe conversion understanding sentinels
                return TypeRegister.MathLedger._AS_FA_TYPE(self, to_num)

        # punt to numpy
        result=TypeRegister.MathLedger._ASTYPE(super(FastArray, self), dtype, order,casting,subok,copy)
        return result.view(FastArray)

    #--------------------------------------------------------------------------
    def _view_internal(self, type=None):
        '''
        FastArray subclasses need to take this over if they want to make a shallow copy of
        a fastarray instead of viewing themselves as a fastarray (which drops their other properties).
        Taking over view directly may have a lot of unintended consequences.
        '''
        if type is not FastArray or type is not None:
            newarr=self.view(type)
            # copy all the properties
            newarr.__dict__ = self.__dict__.copy()
            return newarr
        return self.view(FastArray)

    #--------------------------------------------------------------------------
    def copy(self, order='K'):
        #result= super(FastArray, self).copy(order)
        if self.flags.f_contiguous or self.flags.c_contiguous:
            if order=='K' and self.dtype.num <= 13:
                # perform a faster multithreaded copy
                return TypeRegister.MathLedger._AS_FA_TYPE(self, self.dtype.num)

        result= TypeRegister.MathLedger._COPY(super(FastArray, self), order)
        return result.view(FastArray)

    #--------------------------------------------------------------------------
    def copy_invalid(self):
        '''
        Makes a copy of the array filled with invalids.

        Examples
        --------
        >>> rt.arange(5).copy_invalid()
        FastArray([-2147483648, -2147483648, -2147483648, -2147483648, -2147483648])

        >>> rt.arange(5).copy_invalid().astype(np.float32)
        FastArray([nan, nan, nan, nan, nan], dtype=float32)

        See Also:
        ---------
        FastArray.inv
        FastArray.fill_invalid
        '''
        return self.fill_invalid(inplace=False)

    #--------------------------------------------------------------------------
    @property
    def inv(self):
        '''
        Returns the invalid value for the array.
        np.int8: -128
        np.uint8: 255
        np.int16: -32768
        ...and so on..

        Examples
        --------
        >>> rt.arange(5).inv
        -2147483648

        See Also:
        ---------
        FastArray.copy_invalid
        FastArray.fill_invalid
        INVALID_DICT
        '''
        return INVALID_DICT[self.dtype.num]

    #--------------------------------------------------------------------------
    def fill_invalid(self, shape=None, dtype=None, inplace=True):
        '''
        Fills array or returns copy of array with invalid value of array's dtype or a specified one.
        Warning: by default this operation is inplace.

        Examples
        --------
        >>> a=rt.arange(5).fill_invalid()
        >>> a
        FastArray([-2147483648, -2147483648, -2147483648, -2147483648, -2147483648])

        See Also:
        ---------
        FastArray.inv
        FastArray.fill_invalid
        '''
        return self._fill_invalid_internal(shape=shape, dtype=dtype, inplace=inplace)

    def _fill_invalid_internal(self, shape=None, dtype=None, inplace=True, fill_val=None):
        if dtype is None:
            dtype = self.dtype
        if shape is None:
            shape = self.shape
        elif not isinstance(shape, tuple):
            shape = (shape,)

        if fill_val is None:
            inv = INVALID_DICT[dtype.num]
        else:
            inv = fill_val

        if inplace is True:
            if shape != self.shape:
                raise ValueError(f"Inplace fill invalid cannot be different number of rows than existing array. Got {shape} vs. length {len(self)}")
            if dtype != self.dtype:
                raise ValueError(f"Inplace fill invalid cannot be different dtype than existing categorical. Got {dtype} vs. {len(self.dtype)}")

            self.fill(inv)
        else:
            arr = full(shape, inv, dtype=dtype)
            return arr

    # -------------------------------------------------------------------------
    def isin(self, test_elements, assume_unique=False, invert=False):
        '''
        Calculates `self in test_elements`, broadcasting over `self` only.
        Returns a boolean array of the same shape as `self` that is True
        where an element of `self` is in `test_elements` and False otherwise.

        Parameters
        ----------
        test_elements : array_like
            The values against which to test each value of `element`.
            This argument is flattened if it is an array or array_like.
            See notes for behavior with non-array-like parameters.
        assume_unique : bool, optional
            If True, the input arrays are both assumed to be unique, which
            can speed up the calculation.  Default is False.
        invert : bool, optional
            If True, the values in the returned array are inverted, as if
            calculating `element not in test_elements`. Default is False.
            ``np.isin(a, b, invert=True)`` is equivalent to (but faster
            than) ``np.invert(np.isin(a, b))``.

        Returns
        -------
        isin : ndarray, bool
            Has the same shape as `element`. The values `element[isin]`
            are in `test_elements`.

        Note: behavior differs from pandas
        - Riptable favors bytestrings, and will make conversions from unicode/bytes to match for operations as necessary.
        - We will also accept single scalars for values.
        - Pandas series will return another series - we have no series, and will return a FastArray

        Examples
        --------
        >>> from riptable import *
        >>> a = FA(['a','b','c','d','e'], unicode=False)
        >>> a.isin(['a','b'])
        FastArray([ True,  True, False, False, False])
        >>> a.isin('a')
        FastArray([ True,  False, False, False, False])
        >>> a.isin({'b'})
        FastArray([ False, True, False, False, False])
        '''
        if isinstance(test_elements, set):
            test_elements = list(test_elements)

        if not isinstance(test_elements, np.ndarray):
            # align byte string vs unicode
            if self.dtype.char in 'SU':
                if np.isscalar(test_elements):
                    test_elements = np.asarray([test_elements], dtype=self.dtype.char)
                else:
                    test_elements = np.asarray(test_elements, dtype=self.dtype.char)
            else:
                if isinstance(test_elements, tuple):
                    raise ValueError('isin does not currently support tuples.  In the future a tuple will be used to represent a multi-key.')
                test_elements = np.atleast_1d(test_elements)

        try:
            # optimization: if we have just one element, we can just parallel compare that one element
            if len(test_elements) ==1:
                # string comparison to int will fail
                result = self == test_elements[0]
                # check for failed result
                if np.isscalar(result):
                    result = ismember(self, test_elements)[0]
            else:
                result = ismember(self, test_elements)[0]
            if invert:
                np.logical_not(result, out=result)
            return result
        except Exception:
            # punt non-supported types to numpy
            return np.isin(self._np, test_elements, assume_unique=assume_unique, invert=invert)

    # -------------------------------------------------------------------------
    def between(self, low, high, include_low:bool=True, include_high:bool=False):
        """
        Determine which elements of the array are in a a given interval.

        Return a boolean mask indicating which elements are between `low` and `high` (including/excluding endpoints
        can be controlled by the `include_low` and `include_high` arguments).

        Default behaviour is equivalent to (self >= low) & (self < high).
        
        Parameters
        ----------
        low: scalar, array_like
            Lower bound for test interval.  If array, should have the same size as `self` and comparisons are done elementwise.
        high: scalar, array_like
            Upper bound for test interval.  If array, should have the same size as `self` and comparisons are done elementwise.
        include_low: bool
            Should the left endpoint included in the test interval
        include_high: bool
            Should the right endpoint included in the test interval

        Returns
        -------
        array_like[bool]
            An boolean mask indicating if the associated elements are in the test interval
        """
        low = asanyarray(low)
        high = asanyarray(high)
        
        if include_low:
            ret = self >= low
        else:
            ret = self > low
        if include_high:
            ret &= self <= high
        else:
            ret &= self < high
        return ret

    #--------------------------------------------------------------------------
    def sample(self, N=10, filter=None):
        '''
        Examples
        --------
        >>> a=rt.arange(10)
        >>> a.sample(3)
        FastArray([0, 4, 9])
        '''
        return sample(self, N=N, filter=filter)

    #--------------------------------------------------------------------------
    def duplicated(self, keep='first', high_unique=False):
        '''
        See pandas.Series.duplicated

        Duplicated values are indicated as True values in the resulting
        FastArray. Either all duplicates, all except the first or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            - 'first' : Mark duplicates as True except for the first occurrence.
            - 'last' : Mark duplicates as True except for the last occurrence.
            - False : Mark values with just one occurrence as False.

        '''
        arr = self

        if keep == 'last':
            arr = arr[::-1].copy()

        elif keep is not False and keep != 'first':
            raise ValueError(f'keep must be either "first", "last" or False')

        # create an return array all set to True
        result = ones(len(arr), dtype=np.bool)

        g = Grouping(arr._fa if hasattr(arr,'_fa') else arr)

        if keep is False:
            # search for groups with a count of 1
            result[g.ifirstkey[g.ncountgroup[1:]==1]] = False
        else:
            result[g.ifirstkey] = False

            if keep == 'last':
                result= result[::-1].copy()
        return result

    #--------------------------------------------------------------------------
    def save(self, filepath, share=None, compress=True, overwrite=True, name=None):
        '''
        Save a single array in an .sds file
        See save_sds()
        '''
        save_sds(filepath, self, share=share, compress=compress, overwrite=overwrite, name=name)

    #--------------------------------------------------------------------------
    def reshape(self, *args, **kwargs):
        result = super(FastArray, self).reshape(*args, **kwargs)
        # this warning happens too much now
        #if FastArray._check_ndim(result) != 1:
        #    warnings.warn(FastArray.WarningDict["multiple_dimensions"].format(result.shape))

        if not (result.flags.c_contiguous or result.flags.f_contiguous):
            # fix strides problem
            return result.copy()
        return result

    #--------------------------------------------------------------------------
    def repeat(self, repeats, axis=None):
        ''' see rt.repeat '''
        return repeat(self, repeats, axis=axis)

    #--------------------------------------------------------------------------
    def tile(self, reps):
        ''' see rt.tile '''
        return tile(self, reps)

    #--------------------------------------------------------------------------
    def _kwarg_check(self, *args, **kwargs):
        # we handle dtype
        if ( "ddof" in kwargs and kwargs["ddof"]!=1 ) or "axis" in kwargs or "keepdims" in kwargs:
            return True

    #--------------------------------------------------------------------------
    def _reduce_check(self, reduceFunc, npFunc, *args, **kwargs):
        '''
        Arg2: npFunc pass in None if no numpy equivalent function
        '''
        if npFunc is not None and self._kwarg_check(*args, **kwargs):
            # TODO: add to math ledger
            # set ddof=1 if NOT set which is FastArray default to match matlab/pandas
            if 'ddof' not in kwargs and reduceFunc in [
                REDUCE_FUNCTIONS.REDUCE_VAR,
                REDUCE_FUNCTIONS.REDUCE_NANVAR,
                REDUCE_FUNCTIONS.REDUCE_STD,
                REDUCE_FUNCTIONS.REDUCE_NANSTD]:
                kwargs['ddof']=1

            result = npFunc(self._np, *args, **kwargs)
            return result

        result = TypeRegister.MathLedger._REDUCE(self, reduceFunc)

        dtype = kwargs.get('dtype', None)
        if dtype is not None:
            # user forced dtype return value
            return dtype(result)

        #preserve type for min/max/nanmin/nanmax
        if reduceFunc in [
                REDUCE_FUNCTIONS.REDUCE_MIN,
                REDUCE_FUNCTIONS.REDUCE_NANMIN,
                REDUCE_FUNCTIONS.REDUCE_MAX,
                REDUCE_FUNCTIONS.REDUCE_NANMAX]:
            return self.dtype.type(result)

        #internally numpy expects a dtype returned for nanstd and other calculations
        if isinstance(result,(int, np.integer)):
            # for uint64, the high bit must be preserved
            if self.dtype.char in NumpyCharTypes.UnsignedInteger64:
                return np.uint64(result)
            return np.int64(result)

        return np.float64(result)
    #---------------------------------------------------------------------------
    def _compare_check(self, func, other):
        # a user might type in a string and we want a bytes string
        if self.dtype.char in 'SU':
            if isinstance(other, str):
                if (self.dtype.char=='S'):
                    # we are byte strings but scalar unicode passed in
                    other = str.encode(other)

            if isinstance(other, list):
                # convert the list so a comparison can be made to the byte string array
                other = FastArray(other)

            result= func(other)

            #NOTE: numpy does call FA ufunc for strings
            if not isinstance(result, FastArray) and isinstance(result,np.ndarray):
                result = result.view(FastArray)
            return result

        result= func(other)
        return result

    def __ne__(self,other):   return self._compare_check(super().__ne__,other)
    def __eq__(self, other):  return self._compare_check(super().__eq__,other)
    def __ge__(self, other):  return self._compare_check(super().__ge__,other)
    def __gt__(self, other):  return self._compare_check(super().__gt__,other)
    def __le__(self, other):  return self._compare_check(super().__le__,other)
    def __lt__(self, other):  return self._compare_check(super().__lt__,other)

    #---------------------------------------------------------------------------
    def str_append(self, other):
        if self.dtype.num == other.dtype.num:
            func=TypeRegister.MathLedger._BASICMATH_TWO_INPUTS
            return func((self, other), MATH_OPERATION.ADD, self.dtype.num)
        raise TypeError("cannot concat")

    #---------------------------------------------------------------------------
    def squeeze(self, *args, **kwargs):
        return self._np.squeeze(*args, **kwargs)

    #---------------------------------------------------------------------------
    def iscomputable(self):
        return TypeRegister.is_computable(self)


    #############################################
    # Helper section
    #############################################
    def abs(self, *args, **kwargs):            return np.abs(self, *args, **kwargs)
    def median(self, *args, **kwargs):         return np.median(self, *args, **kwargs)
    def unique(self,  *args, **kwargs):        return unique(self, *args, **kwargs)
    def clip_lower(self, a_min, **kwargs):     return self.clip(a_min, None, **kwargs)
    def clip_upper(self, a_max, **kwargs):     return self.clip(None, a_max, **kwargs)
    def sign(self, **kwargs):                  return np.sign(self, **kwargs)
    def trunc(self, **kwargs):                 return np.trunc(self, **kwargs)
    def where(self, condition, y=np.nan, **kwargs):      return where(condition, self, y, **kwargs)

    def count(self, sorted=True):
        '''
        Returns the unique counts
        Same as calling.unique(return_counts=True)

        Other Parameters
        ----------------
        sorted=True, set to False for first appearance

        Examples
        --------
        >>> a=arange(10) %3
        >>> a.count()
        *Unique   Count
        -------   -----
              0       4
              1       3
              2       3
        '''
        unique_counts= unique(self, sorted=sorted, return_counts=True)
        name=self.get_name()
        if name is None: name = 'Unique'
        ds= TypeRegister.Dataset({name: unique_counts[0], 'Count': unique_counts[1]})
        ds.label_set_names([name])
        return ds

    #############################################
    # Rolling section (cannot handle strides)
    #############################################
    def rolling_sum(self, window:int = 3):     return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_SUM, window)
    def rolling_nansum(self, window:int = 3):  return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_NANSUM, window)
    def rolling_mean(self, window:int = 3):    return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_MEAN, window)
    def rolling_nanmean(self, window:int = 3): return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_NANMEAN, window)
    def rolling_var(self, window:int = 3):     return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_VAR, window)
    def rolling_nanvar(self, window:int = 3):  return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_NANVAR, window)
    def rolling_std(self, window:int = 3):     return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_STD, window)
    def rolling_nanstd(self, window:int = 3):  return rc.Rolling(self, ROLLING_FUNCTIONS.ROLLING_NANSTD, window)


    #############################################
    # TimeWindow section (cannot handle strides), time_array must be INT64
    #############################################
    def timewindow_sum(self, time_array, time_dist):
        '''
        The input array must be int64 and sorted with ever increasing values.
        Sums up the values for a given time window.

        Parameters
        ----------
        time_array: sorted integer array of timestamps
        time_dist: integer value of the time window size

        Examples
        --------
        >>> a=rt.arange(10, dtype=rt.int64)
        >>> a.timewindow_sum(a,5)
        FastArray([ 0,  1,  3,  6, 10, 15, 21, 27, 33, 39], dtype=int64)

        '''
        return rc.TimeWindow(self, time_array, TIMEWINDOW_FUNCTIONS.TIMEWINDOW_SUM, time_dist)

    def timewindow_prod(self, time_array, time_dist):
        '''
        The input array must be int64 and sorted with ever increasing values.
        Multiplies up the values for a given time window.

        Parameters
        ----------
        time_array: sorted integer array of timestamps
        time_dist: integer value of the time window size

        Examples
        --------
        >>> a=rt.arange(10, dtype=rt.int64)
        >>> a.timewindow_prod(a,5)
        FastArray([    0,     0,     0,     0,     0,     0,   720,  5040, 20160, 60480], dtype=int64)
        '''
        return rc.TimeWindow(self, time_array, TIMEWINDOW_FUNCTIONS.TIMEWINDOW_PROD, time_dist)

    #############################################
    # Bottleneck section (only handles int32/int64/float32/float64)
    # bottleneck is optional
    #############################################
    def move_sum(self,  *args, **kwargs):      return bn.move_sum(self, *args, **kwargs)
    def move_mean(self, *args, **kwargs):      return bn.move_mean(self, *args, **kwargs)
    def move_std(self,  *args, **kwargs):      return bn.move_std(self, *args, **kwargs)
    def move_var(self,  *args, **kwargs):      return bn.move_var(self, *args, **kwargs)
    def move_min(self,  *args, **kwargs):      return bn.move_min(self, *args, **kwargs)
    def move_max(self,  *args, **kwargs):      return bn.move_max(self, *args, **kwargs)
    def move_argmin(self, *args, **kwargs):    return bn.move_argmin(self, *args, **kwargs)
    def move_argmax(self, *args, **kwargs):    return bn.move_argmax(self, *args, **kwargs)
    def move_median(self, *args, **kwargs):    return bn.move_median(self, *args, **kwargs)
    def move_rank(self, *args, **kwargs):      return bn.move_rank(self, *args, **kwargs)
    #---------------------------------------------------------------------------
    def replace(self, old, new):               return bn.replace(self, old, new)
    def partition2(self, *args, **kwargs):     return bn.partition(self, *args, **kwargs)
    def argpartition2(self, *args, **kwargs):  return bn.argpartition(self, *args, **kwargs)
    def rankdata(self, *args, **kwargs):       return bn.rankdata(self, *args, **kwargs)
    def nanrankdata(self, *args, **kwargs):    return bn.nanrankdata(self, *args, **kwargs)
    def push(self, *args, **kwargs):           return bn.push(self, *args, **kwargs)

   #---------------------------------------------------------------------------
    def issorted(self):
        ''' returns True if the array is sorted otherwise False
        If the data is likely to be sorted, call the issorted property to check.
        '''
        return issorted(self)
    #---------------------------------------------------------------------------
    def _unary_op(self, funcnum, fancy=False):
        if self._is_not_supported(self):
            # make it contiguous
            arr = self.copy()
        else:
            arr = self

        func=TypeRegister.MathLedger._BASICMATH_ONE_INPUT
        result = func(arr, funcnum, 0)

        if result is None:
            raise TypeError(f'Could not perform operation {funcnum} on FastArray of dtype {arr.dtype}')
        if fancy:
            result = bool_to_fancy(result)
        return result

    #############################################
    # Boolean section
    #############################################
    def isnan(self, fancy=False):       return self._unary_op(MATH_OPERATION.ISNAN, fancy=fancy)
    def isnotnan(self, fancy=False):    return self._unary_op(MATH_OPERATION.ISNOTNAN, fancy=fancy)
    def isnanorzero(self, fancy=False): return self._unary_op(MATH_OPERATION.ISNANORZERO, fancy=fancy)
    def isfinite(self, fancy=False):    return self._unary_op(MATH_OPERATION.ISFINITE, fancy=fancy)
    def isnotfinite(self, fancy=False): return self._unary_op(MATH_OPERATION.ISNOTFINITE, fancy=fancy)
    def isinf(self, fancy=False):       return self._unary_op(MATH_OPERATION.ISINF, fancy=fancy)
    def isnotinf(self, fancy=False):    return self._unary_op(MATH_OPERATION.ISNOTINF, fancy=fancy)
    def isnormal(self, fancy=False):    return self._unary_op(MATH_OPERATION.ISNORMAL, fancy=fancy)
    def isnotnormal(self, fancy=False): return self._unary_op(MATH_OPERATION.ISNOTNORMAL, fancy=fancy)

    #############################################
    # Reduce section
    #############################################
    def nansum(self, *args, **kwargs):         return self._reduce_check( REDUCE_FUNCTIONS.REDUCE_NANSUM, np.nansum,  *args, **kwargs)
    def mean(self, *args, **kwargs):           return self._reduce_check( REDUCE_FUNCTIONS.REDUCE_MEAN,   np.mean,    *args, **kwargs)
    def nanmean(self, *args, **kwargs):        return self._reduce_check( REDUCE_FUNCTIONS.REDUCE_NANMEAN,np.nanmean, *args, **kwargs)
    #---------------------------------------------------------------------------
    # these function take a ddof kwarg
    def var(self, *args, **kwargs):            return self._reduce_check( REDUCE_FUNCTIONS.REDUCE_VAR,    np.var,     *args, **kwargs)
    def nanvar(self, *args, **kwargs):         return self._reduce_check( REDUCE_FUNCTIONS.REDUCE_NANVAR, np.nanvar,  *args, **kwargs)
    def std(self, *args, **kwargs):            return self._reduce_check( REDUCE_FUNCTIONS.REDUCE_STD,    np.std,     *args, **kwargs)
    def nanstd(self, *args, **kwargs):         return self._reduce_check( REDUCE_FUNCTIONS.REDUCE_NANSTD, np.nanstd,  *args, **kwargs)
    #---------------------------------------------------------------------------
    def nanmin(self, *args, **kwargs):         return self._reduce_check( REDUCE_FUNCTIONS.REDUCE_NANMIN, np.nanmin,  *args, **kwargs)
    def nanmax(self, *args, **kwargs):         return self._reduce_check( REDUCE_FUNCTIONS.REDUCE_NANMAX, np.nanmax,  *args, **kwargs)
    #---------------------------------------------------------------------------
    def argmin(self, *args, **kwargs):         return self._reduce_check( REDUCE_FUNCTIONS.REDUCE_ARGMIN, np.argmin,  *args, **kwargs)
    def argmax(self, *args, **kwargs):         return self._reduce_check( REDUCE_FUNCTIONS.REDUCE_ARGMAX, np.argmax,  *args, **kwargs)
    def nanargmin(self, *args, **kwargs):      return self._reduce_check( REDUCE_FUNCTIONS.REDUCE_NANARGMIN, np.nanargmin,  *args, **kwargs)
    def nanargmax(self, *args, **kwargs):      return self._reduce_check( REDUCE_FUNCTIONS.REDUCE_NANARGMAX, np.nanargmax,  *args, **kwargs)

    #############################################
    # Stats/ML section
    #############################################
    def normalize_zscore(self):                return normalize_zscore(self)
    def normalize_minmax(self):                return normalize_minmax(self)

    #############################################
    # BasicMath section (to be hooked at C level now)
    #############################################
    #def __add__(self, value):   result=rc.BasicMathTwoInputs((self, value), 1, 0); result= result if result is not None else np.add(self,value); return result
    #def __add__(self, value):   return rc.BasicMathTwoInputs((self, value), 1, 0)

    @property
    def crc(self):
        '''
        performs a 32 bit CRC algo, returning a 64 bit value

        Examples
        --------

        can be used to compare two arrays for structural equality
        >>> a = arange(100)
        >>> b = arange(100.0)
        >>> a.crc == b.crc
        False
        '''
        return rc.CalculateCRC(self)

    #todo: range/nanrange
    #todo: stats/nanstats

    #-------------------------------------------------------
    def nunique(self):
        '''
        Returns number of unique values in array. Does not include nan or sentinel values in the count.

        Examples
        --------

        Float with nan:

        >>> a = FastArray([1.,2.,3.,np.nan])
        >>> a.nunique()
        3

        Signed integer with sentinel:

        >>> a = FastArray([-128, 2, 3], dtype=np.int8)
        >>> a.nunique()
        2

        Unsigned integer with sentinel:

        >>> a = FastArray([255, 2, 3], dtype=np.uint8)
        >>> a.nunique()
        2

        '''
        un = unique(self)
        count = len(un)
        if count > 0:
            # unique is sorted, so check for sentinel based on dtype
            inv = INVALID_DICT[self.dtype.num]
            if self.dtype.char in NumpyCharTypes.AllFloat:
                # check if last item is nan
                if un[count-1] != un[count-1]:
                    count -= 1
            # unsigned int uses high number as sentinel
            elif self.dtype.char in NumpyCharTypes.UnsignedInteger:
                if un[count-1] == inv:
                    count -= 1
            # all other sentinels are lowest number
            else:
                if un[0] == inv:
                    count -=1
        return count

    #-------------------------------------------------------
    def searchsorted(self, v, side='left', sorter=None):
        return _searchsorted(self, v, side=side, sorter=sorter)

    #---------------------------------------------------------------------------
    def map_old(self, npdict:dict):
        '''
        d = {1:10, 2:20}
        dat['c'] = dat.a.map(d)
        print(dat)
           a  b   cb   c
        0  1  0  0.0  10
        1  1  1  1.0  10
        2  1  2  3.0  10
        3  2  3  5.0  20
        4  2  4  7.0  20
        5  2  5  9.0  20
        '''
        outArray = self.copy()
        for k,v in npdict.items():
            outArray[self==k]=v
        return outArray

    def map(self, npdict:dict):
        '''
        Notes
        -----
        Uses ismember and can handle large dictionaries

        Examples
        --------
        >>> a=arange(3)
        >>> a.map({1: 'a', 2:'b', 3:'c'})
        FastArray(['', 'a', 'b'], dtype='<U1')
        >>> a=arange(3)+1
        >>> a.map({1: 'a', 2:'b', 3:'c'})
        FastArray(['a', 'b', 'c'], dtype='<U1')
        '''
        orig = FastArray([*npdict], unicode=True)
        replace = FastArray([*npdict.values()], unicode=True)
        outArray = self.fill_invalid(self.shape, dtype=replace.dtype, inplace=False)
        found, idx = ismember(self, orig)
        outArray[found] = replace[idx[found]]
        return outArray

    #---------------------------------------------------------------------------
    def shift(self, periods=1, invalid=None):
        """
        Modeled on pandas.shift.
        Values in the array will be shifted to the right for positive, to the left for negative.
        Spaces at either end will be filled with an invalid based on the datatype.
        If abs(periods) >= the length of the FastArray, it will return a FastArray full of invalid
        will be returned.

        Parameters
        ----------
        periods: int, 1
             number of elements to shift right (if positive) or left (if negative), defaults to 1
        invalid: None, default
             optional invalid value to fill

        Returns
        -------
        FastArray shifted right or left by number of periods

        Examples
        --------
        >>> arange(5).shift(2)
        FastArray([-2147483648, -2147483648,           0,           1,            2])
        """

        if periods == 0:
            return self

        if invalid is None:
            if isinstance(self, TypeRegister.Categorical):
                invalid =0
            else:
                try:
                    invalid = INVALID_DICT[self.dtype.num]
                except Exception:
                    raise TypeError(f"shift does not support the dtype {self.dtype.name!r}")

        # we know that this is a simple vector: shape == (len, )
        # TODO: get recycled
        temp = empty_like(self)
        if abs(periods) >= len(self):
            temp.fill(invalid)
        elif periods > 0:
            temp[:periods] = invalid
            temp[periods:] = self[:-periods]
        else:
            temp[:periods] = self[-periods:]
            temp[periods:] = invalid

        # to rewrap categoricals or datelike
        if hasattr(self, 'newclassfrominstance'):
            temp = self.newclassfrominstance(temp, self)

        return temp

    #-------------------------------------------------------
    def _internal_self_compare(self, math_op, periods=1, fancy=False):
        ''' internal routine used for differs and transitions '''
        result = empty_like(self, dtype=np.bool)

        if periods == 0:
            raise ValueError("periods of 0 is invalid for transitions")

        if periods > 0:
            TypeRegister.MathLedger._BASICMATH_TWO_INPUTS((self[periods:], self[:-periods], result[periods:]), math_op, 0)
            # fill upfront with invalids
            result[:periods] = False
        else:
            TypeRegister.MathLedger._BASICMATH_TWO_INPUTS((self[:periods], self[-periods:], result[:periods]), math_op, 0)
            # fill back with invalids (periods is negative)
            result[periods:] = False

        if fancy:
            return bool_to_fancy(result)
        return result

    #-------------------------------------------------------
    def differs(self, periods=1, fancy=False):
        """
        Returns a boolean array.
        The boolean array is set to True when the previous item in the array equals the current.
        Use -1 instead of 1 if you want True set when the next item in the array equals the previous.
        See also: ``transitions``

        ::param periods: The number of elements to look ahead (or behind), defaults to 1
        :type periods: int
        :param fancy: Indicates whether to return a fancy_index instead of a boolean array, defaults to False.
        :type fancy: bool
        :return: boolean ``FastArray``, or fancyIndex (see: `fancy` kwarg)
        """
        if self.dtype.num > 13:
            result = self != self.shift(periods)
            if fancy:
                return bool_to_fancy(result)
            return result
        return self._internal_self_compare(MATH_OPERATION.CMP_EQ, periods=periods, fancy=fancy)

    #---------------------------------------------------------------------------
    def transitions(self, periods=1, fancy=False):
        """
        Returns a boolean array.
        The boolean array is set to True when the previous item in the array does not equal the current.
        Use -1 instead of 1 if you want True set when the next item in the array does not equal the previous.
        See also: ``differs``

        :param periods: The number of elements to look ahead (or behind), defaults to 1
        :type periods: int
        :param fancy: Indicates whether to return a fancy_index instead of a boolean array, defaults to False.
        :type fancy: bool
        :return: boolean ``FastArray``, or fancyIndex (see: `fancy` kwarg)

        >>> a = FastArray([0,1,2,3,3,3,4])
        >>> a.transitions(periods=1)
        FastArray([False, True, True, True, False, False, True])

        >>> a.transitions(periods=2)
        FastArray([False, False, True, True, True, False, True])

        >>> a.transitions(periods=-1)
        FastArray([ True, True, True, False, False, True, False])
        """
        if self.dtype.num > 13:
            result = self != self.shift(periods)
            if fancy:
                return bool_to_fancy(result)
            return result
        return self._internal_self_compare(MATH_OPERATION.CMP_NE, periods=periods, fancy=fancy)

    #-------------------------------------------------------
    def diff(self, periods=1):
        """
        Only works for integers and floats.

        Parameters
        ----------
        periods: int, defaults to 1.  How many elements to shift the data before subtracting.

        Returns
        -------
        FastArray same length as current array.  Invalids will fill the beginning based on the periods.

        Examples
        --------
        >>> a=rt.arange(3, dtype=rt.int32); a.diff()
        FastArray([-2147483648,           1,           1])

        """
        try:
            invalid = INVALID_DICT[self.dtype.num]
        except:
            raise TypeError(f"shift does not support the dtype {self.dtype.name!r}")

        temp = empty(self.shape, dtype=self.dtype)
        if abs(periods) >= len(self):
            temp.fill(invalid)
        elif periods > 0:
            temp[:periods] = invalid

            #output into the empty array we created, np.subtract will call FastArray's subtract
            np.subtract(self[periods:], self[:-periods], out=temp[periods:])
        else:
            temp[periods:] = invalid
            np.subtract(self[:periods], self[-periods:], out= temp[:periods])
        return temp

    #-------------------------------------------------------
    def isna(self):
        '''
        isnan is mapped directly to isnan()
        Categoricals and DateTime take over isnan.
        FastArray handles sentinels.

        >>> a=arange(100.0)
        >>> a[5]=np.nan
        >>> a[87]=np.nan
        >>> sum(a.isna())
        2
        >>> sum(a.astype(np.int32).isna())
        2
        '''
        return self.isnan()

    def notna(self):
        '''
        notna is mapped directly to isnotnan()
        Categoricals and DateTime take over isnotnan.
        FastArray handles sentinels.

        >>> a=arange(100.0)
        >>> a[5]=np.nan
        >>> a[87]=np.nan
        >>> sum(a.notna())
        98
        >>> sum(a.astype(np.int32).notna())
        98
        '''
        return self.isnotnan()

    def replacena(self, value, inplace=False):
        """
        Returns a copy with all invalid values set to the given value.
        Optionally modify the original, this might fail if locked.

        Parameters
        ----------
        value: a replacement value
        inplace: defaults False. If True modify original and return None

        Returns
        -------
        FastArray (size and dtype == original) or None
        """
        inst = self if inplace else self.copy()
        isna = inst.isna()
        if isna.any():
            inst[isna] = value
        if inplace:
            return None
        return inst

    def fillna(self, value=None, method=None, inplace=False, limit=None):
        """
        Returns a copy with all invalid values set to the given value.
        Optionally modify the original, this might fail if locked.

        Parameters
        ----------
        value: a replacement value
        method : {'backfill', 'bfill', 'pad', 'ffill', None},
             backfill/bfill: call fill_backward
             pad/ffill: calls fill_forward
             None: calls replacena

        inplace: defaults False. If True modify original and return None
        limit: only valid when method is not None

        Returns
        -------
        FastArray (size and dtype == original) or None

        Examples
        --------
        >>> ds = rt.Dataset({'A': arange(3), 'B': arange(3.0)})^M
        >>> ds.A[2]=ds.A.inv; ds.B[1]=np.nan;
        >>> ds.fillna(FastArray.fillna, 0)
        #   A      B
        -   -   ----
        0   0   0.00
        1   1   0.00
        2   0   2.00

        """
        if method is not None:
            if method in ['backfill','bfill']:
                return self.fill_backward(value, inplace=inplace, limit=limit)
            if method in ['pad','ffill']:
                return self.fill_forward(value, inplace=inplace, limit=limit)
            raise KeyError(f"fillna: The method {method!r} must be 'backfill', 'bfill', 'pad', 'ffill'")

        if value is None:
            raise ValueError(f"fillna: Must specify either a 'value' that is not None or a 'method' that is not None.")

        if limit is not None:
            raise KeyError(f"fillna: There is no limit when method is None")

        return self.replacena(value, inplace=inplace)

    #---------------------------------------------------------------------------
    def _is_not_supported(self, arr):
        ''' returns True if a numpy array is not FastArray internally supported '''
        if not (arr.flags.c_contiguous or arr.flags.f_contiguous):
            return True
        if arr.dtype.char not in NumpyCharTypes.Supported:
            return True
        if len(arr.strides) == 0:
            return True
        return False

    #---------------------------------------------------------------------------
    def __array_function__(self, func, types, args, kwargs):
        '''
        Called before array_ufunc.
        Does not get called for every function np.isnan/trunc/true_divide for instance.
        '''
        reduceFunc=NUMPY_CONVERSION_TABLE.get(func, None)

        # TODO:
        # kwargs of 'axis': None  'out': None should be accepted

        if reduceFunc is not None and len(kwargs)==0:
            # speed path (todo add call to ledger)
            # default to ddof=0 when no kwargs passed
            result =rc.Reduce(args[0], reduceFunc, 0)

            if result is not None:
                # TypeRegister.MathLedger._REDUCE(args[0], newfunc)

                dtype = kwargs.get('dtype', None)
                if dtype is not None:
                    # user forced dtype return value
                    return dtype(result)

                #preserve type for min/max/nanmin/nanmax
                if reduceFunc in [
                        REDUCE_FUNCTIONS.REDUCE_MIN,
                        REDUCE_FUNCTIONS.REDUCE_NANMIN,
                        REDUCE_FUNCTIONS.REDUCE_MAX,
                        REDUCE_FUNCTIONS.REDUCE_NANMAX]:
                    return self.dtype.type(result)

                #internally numpy expects a dtype returned for nanstd and other calculations
                if isinstance(result,(int, np.integer)):
                    # for uint64, the high bit must be preserved
                    if self.dtype.char in NumpyCharTypes.UnsignedInteger64:
                        return np.uint64(result)
                    return np.int64(result)

                return np.float64(result)
        # call the version numpy wanted use to
        return super(FastArray, self).__array_function__(func, types, args, kwargs)

    #---------------------------------------------------------------------------
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''
        - *ufunc* is the ufunc object that was called.
        - *method* is a string indicating how the Ufunc was called, either
          ``"__call__"`` to indicate it was called directly, or one of its
          :ref:methods<ufuncs.methods>: "reduce", "accumulate"`,
          ``"reduceat"``, ``"outer"``, or ``"at"``.
        - *inputs* is a tuple of the input arguments to the ``ufunc``
        - *kwargs* contains any optional or keyword arguments passed to the
          function. This includes any ``out`` arguments, which are always
          contained in a tuple.
        '''
        toplevel_abort = False

        if FastArray.Verbose > 2:
            print("*** top level array_ufunc", ufunc, method, *inputs, kwargs)

        # flip any inputs that are fastarrays back to an ndarray...
        args = []
        for input in inputs:
            if isinstance(input, np.ndarray):
                toplevel_abort |= self._is_not_supported(input)
            args.append(input)

        # Check for numpy rules that we cannot handle.
        for _kw in ['casting', 'sig', 'signature', 'core_signature']:
            if _kw in kwargs: toplevel_abort |= True

        # Check for numpy rules that we partially support; that is, where we only support
        # the keyword if the value is some default value and otherwise punt to numpy.
        # The value associated with each keyword in the dictionary is the only value we'll
        # support for that keyword.
        # For example, in numpy 1.17 the sum() function passes where=True by default.
        for _kw, default_val in {
            'where': True,
            'axis': None,
            'axes': None
            }.items():
            if _kw in kwargs:
                # Use a type check before equality here to avoid errors caused
                # by checking equality between bools and arrays.
                kwarg_val = kwargs[_kw]
                if type(default_val) != type(kwarg_val) or kwarg_val != default_val:
                    toplevel_abort |= True

        dtype=kwargs.get('dtype',None)

        hasOutputs = False
        out_args = []

        #flip any outputs to ndarray...
        outputs = kwargs.pop('out', None)
        if outputs:
            hasOutputs = True
            for output in outputs:
                if isinstance(output, np.ndarray):
                    toplevel_abort |= self._is_not_supported(output)
                out_args.append(output)
            #replace out
            kwargs['out'] = tuple(out_args)
        else:
            # TJD - here outputs was not specified
            # now if UFunc.nout ==1, this function requires an output
            outputs = (None,) * ufunc.nout

        # See https://docs.python.org/3/c-api/typeobj.html
        # See Number Object Structures and Mapping Object Structure for indexing

        #ufunc.nin	    The number of inputs.
        #ufunc.nout	    The number of outputs.
        #ufunc.nargs	The number of arguments.
        #ufunc.ntypes	The number of types.
        #ufunc.types	Returns a list with types grouped input->output.
        #ufunc.identity	The identity value.

        final_dtype = None
        mode = None
        fastFunction = None
        reduceFunc = None

        # note: when method is 'at' this is an inplace unbuffered operation
        # this can speed up routines that use heavy masked operations
        if method == 'reduce' and FastArray.FasterUFunc and not toplevel_abort:

            # a.any() and a.all() are logical reduce operations
            # Examples
            # Look for axis:None -- otherwise ABORT
            # Then look for Keepdims wihch means to wrap result in list/array?
            # Then check dtype also
            #
            #In [22]: t=FA([[3,4,5],[6,7,8]])
            #In [23]: np.add.reduce(t)
            #!!reduce  reduce nin: 2 1 <ufunc 'add'> [array([[3, 4, 5],
            #       [6, 7, 8]])] out: (None,) kwargs: {}
            #results [ 9 11 13]
            #Out[23]: array([ 9, 11, 13])
            #In [24]: np.add.reduce(t, axis=None)
            #!!reduce  reduce nin: 2 1 <ufunc 'add'> [array([[3, 4, 5],
            #       [6, 7, 8]])] out: (None,) kwargs: {'axis': None}
            #results 33
            #Out[24]: 33
            #In [25]: np.add.reduce(t, axis=None, keepdims=True)
            #!!reduce  reduce nin: 2 1 <ufunc 'add'> [array([[3, 4, 5],
            #       [6, 7, 8]])] out: (None,) kwargs: {'axis': None, 'keepdims': True}
            #results [[33]]
            #Out[25]: array([[33]])
            #In [26]: np.add.reduce(t, axis=None, keepdims=True, dtype=np.float32)
            #!!reduce  reduce nin: 2 1 <ufunc 'add'> [array([[3, 4, 5],
            #       [6, 7, 8]])] out: (None,) kwargs: {'axis': None, 'keepdims': True, 'dtype': <class 'numpy.float32'>}
            #results [[33.]]
            #Out[26]: array([[33.]], dtype=float32)
            #print("!!reduce ", method, 'nin:', ufunc.nin, ufunc.nout, ufunc,  args, 'out:', outputs,  'kwargs:', kwargs,'ndim', args[0].ndim)
            #resultN = super(FastArray, self).__array_ufunc__(ufunc, method, *args, **kwargs)
            #print("!!result numpy", resultN, type(resultN))
            # NOTE:
            # look for reduce logical_or
            # look for reduce_logical_and   (used with np.fmin for instance)

            reduceFunc=gReduceUFuncs.get(ufunc,None)


        # check if we can proceed to calculate a faster way
        if method == '__call__' and FastArray.FasterUFunc and not toplevel_abort:

            # check for binary ufunc
            if len(args)==2 and ufunc.nout==1:

                ###########################################################################
                ## BINARY
                ###########################################################################
                arraytypes = []
                scalartypes = []
                anyTypes=[]

                scalars = 0
                abort=0
                for arr in args:
                    arrType = type(arr)
                    if arrType in ScalarType:
                        scalars += 1
                        scalartypes.append(arrType)
                    else:
                        try:
                            arraytypes.append(arr.dtype)
                            # check for non contingous arrays
                            if arr.itemsize != arr.strides[0]: abort =1
                        except:
                            abort=1
                            # can happen when None or a python list is passed
                            if (FastArray.Verbose > 1):
                                print(f"**dont know how to handle array {arr} args: {args}")

                if abort == 0:
                    if scalars < 2:
                        isLogical = 0
                        # check for add, sub, mul, divide, power
                        fastFunction= gBinaryUFuncs.get(ufunc, None)
                        if fastFunction is None:
                            #check for comparison and logical or/and functions
                            fastFunction = gBinaryLogicalUFuncs.get(ufunc, None)
                            if fastFunction is not None:
                                if (FastArray.Verbose > 2):
                                    print(f"**logical function called {ufunc} args: {args}")
                                isLogical = 1
                                final_dtype = np.bool


                        if fastFunction is None:
                            #check for bitwise functions? (test this)
                            fastFunction = gBinaryBitwiseUFuncs.get(ufunc, None)

                        if fastFunction is not None:
                            if hasOutputs and isLogical == 0:
                                # have to conform to output
                                final_dtype = out_args[0].dtype
                            else:
                                if isLogical == 1 and scalars == 1:
                                # NOTE: scalar upcast rules -- just apply to logicals so that arr < 5 does not upcast?
                                #       or globally apply this rule so that arr = arr + 5
                                #if scalars == 1:
                                    #special case have to see if scalar is in range
                                    if type(args[0]) in ScalarType:
                                        scalarval = args[0]
                                    else:
                                        scalarval = args[1]

                                    final_dtype = logical_find_common_type(arraytypes, scalartypes, scalarval)

                                else:
                                    print
                                    # TODO: check for bug where np.int32 type 7 gets flipped to np.int32 type 5
                                    if scalars ==0 and len(arraytypes)==2 and (arraytypes[0] == arraytypes[1]):
                                        final_dtype = arraytypes[0]
                                    else:
                                        # check for int scalar against int
                                        # bug where np.int8 and then add +1999 or larger number.  need to upcast
                                        if scalars == 1 and arraytypes[0].num <=10:
                                            if type(args[0]) in ScalarType:
                                                scalarval = args[0]
                                            else:
                                                scalarval = args[1]

                                            final_dtype = logical_find_common_type(arraytypes, scalartypes, scalarval)
                                        else:
                                            final_dtype = np.find_common_type(arraytypes, scalartypes)

                            # if we are adding two strings or unicode, special case
                            # if we think the final dtype is an object, check if this is really two strings
                            if fastFunction == MATH_OPERATION.ADD and (arraytypes[0].num == 18 or arraytypes[0].num == 19) :
                                # assume addition of two strings
                                final_dtype = arraytypes[0]
                                if scalars != 0:
                                    # we have a scalar... make sure we convert it
                                    if type(args[0]) in ScalarType:
                                        # fix scalar type make sure string or unicode
                                        if arraytypes[0].num == 18:
                                            args[0] = str.encode(str(args[0]))
                                        if arraytypes[0].num == 19:
                                            args[0] = str(args[0])
                                    else:
                                        if arraytypes[0].num == 18:
                                            args[1] = str.encode(str(args[1]))
                                        if arraytypes[0].num == 19:
                                            args[1] = str(args[1])
                                else:
                                    # we have two arrays, if one array is not proper string type, convert it
                                    if arraytypes[1] != final_dtype:
                                        if arraytypes[0].num == 18:
                                            args[1] = args[1].astype('S')
                                        if arraytypes[0].num == 19:
                                            args[1] = args[1].astype('U')

                                if (FastArray.Verbose > 2):
                                    print("ADD string operation", arraytypes, scalartypes)

                            elif scalars ==0:
                                if arraytypes[0] != arraytypes[1]:
                                    # UPCAST RULES
                                    if arraytypes[0] == final_dtype and arraytypes[1] != final_dtype:
                                        #print("!!!upcast rules second", arraytypes[0], arraytypes[1], final_dtype)
                                        #convert to the proper type befor calculation
                                        args[1] = _ASTYPE(args[1], final_dtype)

                                    elif arraytypes[0] != final_dtype and arraytypes[1] == final_dtype:
                                        #print("!!!upcast rules first", arraytypes[0], arraytypes[1], final_dtype)
                                        #convert to the proper type befor calculation
                                        args[0] = _ASTYPE(args[0], final_dtype)
                                    else:
                                        # sometimes both of them must be upcast...
                                        # consider  int8 * uint8 ==> will upcast to int16
                                        #print("!!!cannot understand upcast rules", arraytypes[0], arraytypes[1], final_dtype)
                                        args[0] = _ASTYPE(args[0], final_dtype)
                                        args[1] = _ASTYPE(args[1], final_dtype)
                                        #TJD check logic here... what does numpy when int* * uint8 ? speed test
                                        ##UseNumpy = True
                            else:
                                # UPCAST RULES when one is a scalar
                                if arraytypes[0] != final_dtype:
                                    # which argument is the scalar?  convert the other one
                                    if type(args[0]) in ScalarType:
                                        #print("converting arg2 from", args[1], final_dtype)
                                        args[1] = _ASTYPE(args[1], final_dtype)
                                    else:
                                        #print("converting arg1 from ", args[0], final_dtype)
                                        args[0] = _ASTYPE(args[0], final_dtype)



            # not a binary ufunc, check for unary ufunc
            # check for just 1 input (unary)

            elif ((ufunc.nin==1) and (ufunc.nout==1)):
                ###########################################################################
                ## UNARY
                ###########################################################################
                fastFunction= gUnaryUFuncs.get(ufunc, None)
            else:
                if (FastArray.Verbose > 1):
                    print("***unknown ufunc arg style: ", ufunc.nin, ufunc.nout, ufunc, args, kwargs)


        # -------------------------------------------------------------------------------------------------------------
        if not FastArray.FasterUFunc:
            fastFunction = None
            reduceFunc= None

        #arrType = input[0].dtype
        #shape = numel(input[0])
        outArray = None

        # check for a reduce func like sum or min
        if reduceFunc is not None:
            keepdims = kwargs.get('keepdims',False)
            if dtype is None: dtype = args[0].dtype

            #MathLedger
            result= TypeRegister.MathLedger._REDUCE(args[0], reduceFunc)
            char = np.dtype(dtype).char
            if FastArray.Verbose > 1:
                print("***result from reduce", result, type(result), dtype, char)

            if result is not None:
                #print("reduce called", ufunc, keepdims, dtype)
                if reduceFunc in [REDUCE_FUNCTIONS.REDUCE_SUM, REDUCE_FUNCTIONS.REDUCE_NANSUM] and isinstance(result, float):
                    result = np.float64(result)
                elif (dtype != np.float32 and dtype != np.float64):
                    # preserve integers
                    if char in NumpyCharTypes.UnsignedInteger64:
                        # preserve high bit
                        result = np.uint64(result)
                    else:
                        result = np.int64(result)
                else:
                    result=np.float64(result)

                # MIN/MAX need to return same type
                if (reduceFunc >= REDUCE_FUNCTIONS.REDUCE_MIN):
                    # min max not allowed on empty array per unit test
                    if (len(args[0])==0): raise ValueError("min/max arg is an empty sequence.")

                    # min/max/nanmin/nanmax -- same result
                    if dtype == np.bool:
                        result =np.bool(result)
                    else:
                        result=dtype.type(result)

                    if (keepdims):
                        result= FastArray([result]).astype(dtype)
                elif (keepdims):
                    # force back into an array from scalar
                    result= FastArray([result])


                # we did the reduce, now return the result
                return result

        # check for normal call function
        elif fastFunction is not None:
            # Call the FastArray APIs instead of numpy
            #callmode = 'f'
            results=None
            if ufunc.nin==2:
                final_num=-1
                if final_dtype is not None:
                    if final_dtype == np.bool:
                        final_num=0
                    else:
                        final_num=final_dtype.num

                # because scalars can be passed as np.int64(864000)
                if type(args[0]) in gNumpyScalarType:
                    #print('converting arg1', args[0])
                    args[0]=np.asarray(args[0]);

                if type(args[1]) in gNumpyScalarType:
                    #print('converting arg2', args[1])
                    args[1]=np.asarray(args[1]);


                if FastArray.Verbose > 2:
                    print("*** binary think we can call", fastFunction, ufunc.nin, ufunc.nout, "arg1", args[0], "arg2", args[1], "out", out_args, "final", final_num)
                if len(out_args)==1:
                    results = TypeRegister.MathLedger._BASICMATH_TWO_INPUTS((args[0], args[1], out_args[0]), fastFunction, final_num)
                else:
                    results = TypeRegister.MathLedger._BASICMATH_TWO_INPUTS((args[0], args[1]), fastFunction, final_num)
            else:
                #for conversion functions
                #dtype=kwargs.get('dtype',None)
                if FastArray.Verbose > 2:
                    print("*** unary think we can call", fastFunction, ufunc.nin, ufunc.nout, "arg1", args[0], "out", out_args)

                if len(out_args)==1:
                    results = TypeRegister.MathLedger._BASICMATH_ONE_INPUT((args[0], out_args[0]), fastFunction,0)
                else:
                    results = TypeRegister.MathLedger._BASICMATH_ONE_INPUT((args[0]), fastFunction,0)

            if results is not None and len(out_args)==1:
                # when the output argument is forced but we calculate it into another array we need to copy the result into the output
                if not rc.CompareNumpyMemAddress(out_args[0], results):
                    if FastArray.Verbose > 2:
                        print("*** performing an extra copy to match output request", id(out_args[0]), id(results), out_args[0], results)
                    out_args[0][...]=results
                    results = out_args[0]

            if results is None:
                #punted
                #callmode='p'
                if (FastArray.Verbose > 1):
                    print("***punted ufunc: ", ufunc.nin, ufunc.nout, ufunc, args, kwargs)
                fastFunction =None
                # fall to "if fastFunction is None" and run through numpy...

            # respect dtype
            elif dtype is not None and isinstance(results, np.ndarray):
                if dtype is not results.dtype:
                    if FastArray.Verbose > 1:  print("***result from reduce", results, results.dtype, dtype)
                    # convert
                    results = results.astype(dtype)

        if fastFunction is None:
            # Call the numpy APIs
            # Check if we can use the recycled arrays to avoid an allocation for the output array

            if FastArray.Verbose > 1:
                print("**punted on numpy!", ufunc)

            # NOTE: We are going to let numpy process it
            # We must change all FastArrays to normal numpy arrays
            args = []
            for input in inputs:
                #flip back to numpy to avoid errors when numpy calculates
                if isinstance(input, FastArray):
                    args.append(input.view(np.ndarray))
                else:
                    args.append(input)

            if hasOutputs:
                outputs = kwargs.pop('out', None)
                if outputs:
                    out_args=[]
                    for output in outputs:
                        if isinstance(output, FastArray):
                            out_args.append(output.view(np.ndarray))
                        else:
                            out_args.append(output)
                    #replace out
                    kwargs['out'] = tuple(out_args)

            # NOTE: If the specified ufunc + inputs combination isn't supported by numpy either,
            #       as of numpy 1.17.x this call will end up raising a UFuncTypeError so the rest
            #       of the FastArray.__array_ufunc__ body (below) won't end up executing.
            results = TypeRegister.MathLedger._ARRAY_UFUNC(super(FastArray, self),ufunc, method, *args, **kwargs)

        # If riptable has not implemented a certain ufunc (or doesn't support it for the given arguments),
        # emit a warning about it to let the user know.
        # When numpy does not support the ufunc+inputs either, we won't reach this point (as of numpy 1.17.x),
        # since numpy will raise a UFuncTypeError earlier (before this point) rather than after we return NotImplemented.
        if results is NotImplemented:
            warnings.warn(f"***ufunc {ufunc} {args} {kwargs} is not implemented")
            return NotImplemented

        #Ufuncs also have a fifth method that allows in place operations to be performed using fancy indexing.
        #No buffering is used on the dimensions where fancy indexing is used, so the fancy index can list an item more than once
        #     and the operation will be performed on the result of the previous operation for that item.
        #ufunc.reduce(a[, axis, dtype, out, keepdims])	Reduces a's dimension by one, by applying ufunc along one axis.
        #ufunc.accumulate(array[, axis, dtype, out])	Accumulate the result of applying the operator to all elements.
        #ufunc.reduceat(a, indices[, axis, dtype, out])	Performs a (local) reduce with specified slices over a single axis.
        #ufunc.outer(A, B)	Apply the ufunc op to all pairs (a, b) with a in A and b in B.
        #ufunc.at(a, indices[, b])	Performs unbuffered in place operation on operand 'a' for elements specified by 'indices'.

        if method == 'at':
            return

        if ufunc.nout == 1:
            #check if we used our own output

            #if isinstance(outArray, np.ndarray):
            #    return outArray.view(FastArray)

            #if (final_dtype != None and final_dtype != results.dtype):
            #    print("****** mispredicted final", final_dtype, results.dtype, ufunc, scalartypes, args, outputs, kwargs);
            #results = (results,)

            if not isinstance(results,FastArray) and isinstance(results,np.ndarray):
                return results.view(FastArray)

            # think hit here for sum wihch does not return an array, just a number
            return results

        # more than one item, so we are making a tuple
        # can result in __array_finalize__ being called
        results = tuple((np.asarray(result).view(FastArray)
                         if output is None else output)
                        for result, output in zip(results, outputs))

        # check if we have a tuple of one item, if so just return the one item
        if len(results) == 1:
            results = results[0]
        return results

    @property
    def numbastring(self):
        '''
        converts byte string and unicode strings to a 2dimensional array
        so that numba can process it correctly

        Examples
        --------
        >>> @numba.jit(nopython=True)
        ... def numba_str(txt):
        ...     x=0
        ...     for i in range(txt.shape[0]):
        ...         if (txt[i,0]==116 and  # 't'
        ...             txt[i,1]==101 and  # 'e'
        ...             txt[i,2]==120 and  # 'x'
        ...             txt[i,3]==116):    # 't'
        ...             x += 1
        ...     return x
        >>>
        >>> x=FastArray(['some','text','this','is'])
        >>> numba_str(x.view(np.uint8).reshape((len(x), x.itemsize)))
        >>> numba_str(x.numbastring)
        '''

        intype=self.dtype.__str__()
        if intype[0]=='|' or intype[0]=='<':
            if intype[1]=='S':
                return self.view(np.uint8).reshape((len(self), self.itemsize))
            if intype[1]=='U':
                return self.view(np.uint32).reshape((len(self), self.itemsize//4))
        return self

    #-----------------------------------------------------------
    def apply_numba(self, *args, otype=None, myfunc="myfunc",name=None):
        '''
        Usage:
        -----
        Prints to screen an example numba signature for the array.
        You can then copy this example to build your own numba function.

        Inputs:
        ------
        Can pass in multiple test arguments.

        kwargs
        ------
        otype: specify a different output type
        myfunc: specify a string to call the function
        name: specify a string to name the array

        Example using numba
        -------------------
        >>> import numba
        >>> @numba.guvectorize(['void(int64[:], int64[:])'], '(n)->(n)')
        ... def squarev(x,out):
        ...     for i in range(len(x)):
        ...         out[i]=x[i]**2
        ...
        >>> a=arange(1_000_000).astype(np.int64)
        >>> squarev(a)
        FastArray([           0,            1,            4, ..., 999994000009,
                   999996000004, 999998000001], dtype=int64)
        '''
        if name is None:
            # try first to get the name
            name=self.get_name()

            if name is None:
                name="a"

        intype=self.dtype.__str__()

        if otype is None:
            outtype=self.dtype.__str__()
        else:
            outtype=np.dtype(otype).__str__()

        # TODO: what if unicode or string?  .frombuffer/.view(np.uint8)

        preamble = "import numba\n@numba.guvectorize([\n"

        middle=f"'void({intype}[:], {outtype}[:])',       # <-- can stack multiple different dtypes  x.view(np.uint8).reshape(-1, x.itemsize)\n"

        postamble="    ], '(n)->(n)', target='cpu')\n"
        code=f"def {myfunc}(data_in, data_out):\n    for i in range(len(data_in)):\n        data_out[i]=data_in[i]   #<-- put your code here\n"
        exec = preamble+middle+postamble+code

        print("Copy the code snippet below and rename myfunc")
        print("---------------------------------------------")
        print(exec)
        print("---------------------------------------------")
        if intype[0]=='|' or intype[0]=='<':
            if intype[1]=='S':
                print(f"Then call {myfunc}({name}.numbastring,empty_like({name}).numbastring) where {name} is the input array")
            elif intype[1]=='U':
                print(f"Then call {myfunc}({name}.numbastring,empty_like({name}).numbastring) where {name} is the input array")
        else:
            print(f"Then call {myfunc}({name},empty_like({name})) where {name} is the input array")
        #return exec

    def apply(self, pyfunc, *args, otypes=None, doc=None, excluded =None, cache=False, signature=None):
        """
        Generalized function class.  see: np.vectorize

        Creates and then applies a vectorized function which takes a nested sequence of objects or
        numpy arrays as inputs and returns an single or tuple of numpy array as
        output. The vectorized function evaluates `pyfunc` over successive tuples
        of the input arrays like the python map function, except it uses the
        broadcasting rules of numpy.

        The data type of the output of `vectorized` is determined by calling
        the function with the first element of the input.  This can be avoided
        by specifying the `otypes` argument.

        Parameters
        ----------
        pyfunc : callable
            A python function or method.
        otypes : str or list of dtypes, optional
            The output data type. It must be specified as either a string of
            typecode characters or a list of data type specifiers. There should
            be one data type specifier for each output.
        doc : str, optional
            The docstring for the function. If `None`, the docstring will be the
            ``pyfunc.__doc__``.
        excluded : set, optional
            Set of strings or integers representing the positional or keyword
            arguments for which the function will not be vectorized.  These will be
            passed directly to `pyfunc` unmodified.

            .. versionadded:: 1.7.0

        cache : bool, optional
           If `True`, then cache the first function call that determines the number
           of outputs if `otypes` is not provided.

            .. versionadded:: 1.7.0

        signature : string, optional
            Generalized universal function signature, e.g., ``(m,n),(n)->(m)`` for
            vectorized matrix-vector multiplication. If provided, ``pyfunc`` will
            be called with (and expected to return) arrays with shapes given by the
            size of corresponding core dimensions. By default, ``pyfunc`` is
            assumed to take scalars as input and output.

            .. versionadded:: 1.12.0

        Returns
        -------
        vectorized : callable
            Vectorized function.

        See Also
        --------
        FastArray.apply_numba
        FastArray.apply_pandas

        Examples
        --------
        >>> def myfunc(a, b):
        ...     "Return a-b if a>b, otherwise return a+b"
        ...     if a > b:
        ...         return a - b
        ...     else:
        ...         return a + b
        >>>
        >>> a=arange(10)
        >>> b=arange(10)+1
        >>> a.apply(myfunc,b)
        FastArray([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19])

        Example with one input array

        >>> def square(x):
        ...     return x**2
        >>>
        >>> a=arange(10)
        >>> a.apply(square)
        FastArray([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])

        Example with lambda

        >>> a=arange(10)
        >>> a.apply(lambda x: x**2)
        FastArray([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])

        Example with numba

        >>> from numba import jit
        >>> @jit
        ... def squareit(x):
        ...     return x**2
        >>> a.apply(squareit)
        FastArray([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])

        Examples to use existing builtin oct function but change the output from string, to unicode, to object

        >>> a=arange(10)
        >>> a.apply(oct, otypes=['S'])
        FastArray([b'0o0', b'0o1', b'0o2', b'0o3', b'0o4', b'0o5', b'0o6', b'0o7', b'0o10', b'0o11'], dtype='|S4')

        >>> a=arange(10)
        >>> a.apply(oct, otypes=['U'])
        FastArray(['0o0', '0o1', '0o2', '0o3', '0o4', '0o5', '0o6', '0o7', '0o10', '0o11'], dtype='<U4')

        >>> a=arange(10)
        >>> a.apply(oct, otypes=['O'])
        FastArray(['0o0', '0o1', '0o2', '0o3', '0o4', '0o5', '0o6', '0o7', '0o10', '0o11'], dtype=object)

        """

        vfunc = np.vectorize(pyfunc, otypes=otypes, doc=doc, excluded=excluded, cache=cache, signature=signature)
        result=vfunc(self, *args)
        return result

    #-----------------------------------------------------------
    def apply_pandas(self, func, convert_dtype=True, args=(), **kwds):
        """
        Invoke function on values of FastArray. Can be ufunc (a NumPy function
        that applies to the entire FastArray) or a Python function that only works
        on single values

        Parameters
        ----------
        func : function
        convert_dtype : boolean, default True
            Try to find better dtype for elementwise function results. If
            False, leave as dtype=object
        args : tuple
            Positional arguments to pass to function in addition to the value
        Additional keyword arguments will be passed as keywords to the function

        Returns
        -------
        y : FastArray or Dataset if func returns a FastArray

        See Also
        --------
        FastArray.map: For element-wise operations
        FastArray.agg: only perform aggregating type operations
        FastArray.transform: only perform transforming type operations

        Examples
        --------
        Create a FastArray with typical summer temperatures for each city.

        >>> fa = rt.FastArray([20, 21, 12], index=['London', 'New York','Helsinki'])
        >>> fa
        London      20
        New York    21
        Helsinki    12
        dtype: int64

        Square the values by defining a function and passing it as an
        argument to ``apply()``.

        >>> def square(x):
        ...     return x**2
        >>> fa.apply(square)
        London      400
        New York    441
        Helsinki    144
        dtype: int64

        Square the values by passing an anonymous function as an
        argument to ``apply()``.

        >>> fa.apply(lambda x: x**2)
        London      400
        New York    441
        Helsinki    144
        dtype: int64

        Define a custom function that needs additional positional
        arguments and pass these additional arguments using the
        ``args`` keyword.

        >>> def subtract_custom_value(x, custom_value):
        ...     return x-custom_value
        >>> fa.apply(subtract_custom_value, args=(5,))
        London      15
        New York    16
        Helsinki     7
        dtype: int64

        Define a custom function that takes keyword arguments
        and pass these arguments to ``apply``.

        >>> def add_custom_values(x, **kwargs):
        ...     for month in kwargs:
        ...         x+=kwargs[month]
        ...     return x
        >>> fa.apply(add_custom_values, june=30, july=20, august=25)
        London      95
        New York    96
        Helsinki    87
        dtype: int64

        Use a function from the Numpy library.

        >>> fa.apply(np.log)
        London      2.995732
        New York    3.044522
        Helsinki    2.484907
        dtype: float64
        """
        import pandas as pd

        series = pd.Series(self)
        result = series.apply(func, convert_dtype=convert_dtype, args=args, **kwds)
        return result.values

    #-----------------------------------------------------------
    @property
    def str(self):
        r"""Casts an array of byte strings or unicode as ``FAString``.

        Enables a variety of useful string manipulation methods.

        Returns
        -------
        FAString

        Raises
        ------
        TypeError
            If the FastArray is of dtype other than byte string or unicode

        See Also
        --------
        np.chararray
        np.char
        rt.FAString.apply

        Examples
        --------
        >>> s=FA(['this','that','test ']*100_000)
        >>> s.str.upper
        FastArray([b'THIS', b'THAT', b'TEST ', ..., b'THIS', b'THAT', b'TEST '],
                  dtype='|S5')
        >>> s.str.lower
        FastArray([b'this', b'that', b'test ', ..., b'this', b'that', b'test '],
                  dtype='|S5')
        >>> s.str.removetrailing()
        FastArray([b'this', b'that', b'test', ..., b'this', b'that', b'test'],
                  dtype='|S5')

        """
        if self.dtype.char in 'US':
            return TypeRegister.FAString(self)
        if self.dtype.char == 'O':
            # try to convert to string (might have come from pandas)
            try:
                conv = self.astype('S')
            except:
                conv = self.astype('U')
            return TypeRegister.FAString(conv)

        raise TypeError(f"The .str function can only be used on byte string and unicode not {self.dtype!r}")

    #-----------------------------------------------------------
    @classmethod
    def register_function(cls, name, func):
        '''
        Used to register functions to FastArray.
        Used by rt_fastarraynumba
        '''
        setattr(cls, name, func)

    def apply_schema(self, schema):
        """
        Apply a schema containing descriptive information to the FastArray

        :param schema: dict
        :return: dictionary of deviations from the schema
        """
        from .rt_meta import apply_schema as _apply_schema
        return _apply_schema(self, schema)

    def info(self, **kwargs):
        """
        Print a description of the FastArray's contents
        """
        from .rt_meta import info as _info
        return _info(self, **kwargs)

    @property
    def doc(self):
        """
        The Doc object for the structure
        """
        from .rt_meta import doc as _doc
        return _doc(self)
    # ====================== END OF CLASS DEFINITION ===============================

#-----------------------------------------------------------
def _setfastarrayview(arr):
    '''
    Call from CPP into python to flip array view
    '''
    if isinstance(arr, FastArray):
        if FastArray.Verbose > 2:
            print("no need to setfastarrayview", arr.dtype, len(arr))
        return arr

    if FastArray.Verbose > 2:
        print("setfastarrayview", arr.dtype, len(arr))

    return arr.view(FastArray)


#-----------------------------------------------------------
def _setfastarraytype():
    #-----------------------------------------------------------
    # calling this function will force fm to return FastArray subclass
    #rc.BasicMathHook(FastArray, np.ndarray)
    # Coming next build
    fa=np.arange(1).view(FastArray)
    rc.SetFastArrayType(fa, _setfastarrayview)
    rc.BasicMathHook(fa, fa._np)

#-----------------------------------------------------------
def _FixupDocStrings():
    """
    Load all the member function of this module
    Load all the member functions of the np module
    If we find a match, copy over the doc strings
    """
    import inspect
    import sys
    mymodule=sys.modules[__name__]
    all_myfunctions = inspect.getmembers(FastArray, inspect.isfunction)

    try:
        # bottleneck is optional
        all_bnfunctions = inspect.getmembers(bn, inspect.isfunction)
        all_bnfunctions += inspect.getmembers(bn, inspect.isbuiltin)

        # build dictionary of bottleneck docs
        bndict={}
        for funcs in all_bnfunctions:
            bndict[funcs[0]]=funcs[1]

        # now for each function that has an bn flavor, copy over the doc strings
        for funcs in all_myfunctions:
            if funcs[0] in bndict:
                funcs[1].__doc__ = bndict[funcs[0]].__doc__
    except Exception:
        pass

    all_npfunctions = [func for func in inspect.getmembers(np.ndarray)
                       if not func[0].startswith('_')]

    # build dictionary of np.ndarray docs
    npdict={}
    for funcs in all_npfunctions:
        npdict[funcs[0]]=funcs[1]

    # now for each function that has an np flavor, copy over the doc strings
    for funcs in all_myfunctions:
        if funcs[0] in npdict:
            funcs[1].__doc__ = npdict[funcs[0]].__doc__

    # now do just plain np
    all_npfunctions = [func for func in inspect.getmembers(np)
                       if '__' not in funcs[0]]

    # build dictionary of np docs
    npdict={}
    for funcs in all_npfunctions:
        #print("getting doc string for ", funcs[0])
        npdict[funcs[0]]=funcs[1]

    # now for each function that has an np flavor, copy over the doc strings
    for funcs in all_myfunctions:
        if funcs[0] in npdict:
            funcs[1].__doc__ = npdict[funcs[0]].__doc__


#----------------------------------------------------------
class Threading():
    @staticmethod
    def on():
        '''
        Turn riptable threading on.
        Used only when riptable threading was turned off.

        Example
        -------
        a=rt.arange(1_000_00)
        Threading.off()
        %time a+=1
        Threading.on()
        %time a+=1

        Returns
        -------
        Previously whether threading was on or not. 0 or 1. 0=threading was off before.

        '''
        return FastArray._TON()

    @staticmethod
    def off():
        '''
        Turn riptable threading off.
        Useful for when the system has other processes using other threads
        or to limit threading resources.

        Example
        -------
        a=rt.arange(1_000_00)
        Threading.off()
        %time a+=1
        Threading.on()
        %time a+=1

        Returns
        -------
        Previously whether threading was on or not. 0 or 1. 0=threading was off before.
        '''
        return FastArray._TOFF()

    @staticmethod
    def threads(threadcount):
        '''
        Set how many worker threads riptable can use.
        Often defaults to 12 and cannot be set below 1 or > 31.

        To turn riptable threading off completely use Threading.off()
        Useful for when the system has other processes using other threads
        or to limit threading resources.

        Example
        -------
        Threading.threads(8)

        Returns
        -------
        number of threads previously used
        '''
        return rc.SetThreadWakeUp(threadcount)


#----------------------------------------------------------
class Recycle():
    @staticmethod
    def on():
        '''
        Turn riptable recycling on.
        Used only when riptable recycling was turned off.

        Example
        -------
        a=arange(1_000_00)
        Recycle.off()
        %timeit a=a + 1
        Recycle.on()
        %timeit a=a + 1

        '''
        return FastArray._RON()

    @staticmethod
    def off():
        return FastArray._ROFF()

    @staticmethod
    def now(timeout:int = 0):
        '''
        Pass the garbage collector timeout value to cleanup.
        Also calls the python garbage collector.

        Parameters
        ----------
        timeout: default to 0.  0 will not set a timeout

        Returns
        -------
        total arrays deleted
        '''
        import gc
        gc.collect()
        result= rc.RecycleGarbageCollectNow(timeout)['TotalDeleted']
        if result > 0:
            rc.RecycleGarbageCollectNow(timeout)
        return result

    @staticmethod
    def timeout(timeout:int = 100):
        '''
        Pass the garbage collector timeout value to expire.
        The timeout value is roughly in 2/5 secs.
        A value of 100 is usually about 40 seconds.
        If an array has not been reused by the timeout, it is permanently deleted.

        Returns
        -------
        previous timespan
        '''
        return rc.RecycleSetGarbageCollectTimeout(timeout)

#----------------------------------------------------------
class Ledger():
    @staticmethod
    def on():
        '''Turn the math ledger on to record all array math routines'''
        return TypeRegister.MathLedger._LedgerOn()

    @staticmethod
    def off():
        '''Turn the math ledger off'''
        return TypeRegister.MathLedger._LedgerOff()

    @staticmethod
    def dump(dataset=True):
        '''Print out the math ledger'''
        return TypeRegister.MathLedger._LedgerDump(dataset=dataset)

    @staticmethod
    def to_file(filename):
        '''Save the math ledger to a file'''
        return TypeRegister.MathLedger._LedgerDumpFile(filename)

    @staticmethod
    def clear():
        '''Clear all the entries in the math ledger'''
        return TypeRegister.MathLedger._LedgerClear()

#----------------------------------------------------------
# this is called when the module is loaded
_FixupDocStrings()

# NOTE: Keep this at the end of the file
#-----------------------------------------------------------
# calling this function will force fm to return FastArray subclass
_setfastarraytype()

TypeRegister.FastArray=FastArray

FastArray.register_function('describe', describe)


