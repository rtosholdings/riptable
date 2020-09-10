__all__ = [
    'DATETIME_TYPES',
    'DS_DISPLAY_TYPES',
    'GB_FUNCTIONS',
    'MATH_OPERATION',
    'INVALID_DICT',
    'TIMEWINDOW_FUNCTIONS',
    'NumpyCharTypes',
    'REDUCE_FUNCTIONS',
    'ROLLING_FUNCTIONS',
    'SD_TYPES',
    'SM_DTYPES',
    'TypeRegister',
    'DisplayJustification',
    'DisplayColumnColors',
    'DisplayArrayTypes',
    'DisplayDetectModes',
    'DisplayLength',
    'ColHeader',
]


import sys
from collections import namedtuple
from typing import Optional, Callable, Mapping, Any, Tuple, Type, List
from enum import IntEnum
import numpy as np

# common strings used to indicate special columns or items
INVALID_SHORT_NAME: str = 'Inv'
INVALID_LONG_NAME: str = 'Invalid'
TOTAL_LONG_NAME: str = 'Total'
CLIPPED_LONG_NAME: str = 'Clipped'
FILTERED_LONG_NAME: str = 'Filtered'
GROUPBY_KEY_PREFIX: str = 'key'
"""Default groupby key name - followed by _n for n"""

INVALID_POINTER_32: int = -2147483648
INVALID_POINTER_U32: int = 0xFFFFFFFF  # 4294967295
# INVALID_POINTER_64  = 0x8000000000000000  # -9223372036854775808
INVALID_POINTER_64: int = -9223372036854775808
INVALID_POINTER_U64: int = 0xFFFFFFFFFFFFFFFF  # 18446744073709551615


INVALID_DICT: Mapping[int, Any] = {
    # keys in this dict can be generated with ndarray.dtype.num
    0: False,  # np.bool
    1: -128,  # np.int8
    2: 255,  # numpy.uint8,
    3: -32768,  # numpy.int16
    4: 65535,  # numpy.uint16
    5: INVALID_POINTER_32,  # numpy.int32
    6: INVALID_POINTER_U32,  # numpy.uint32
    7: INVALID_POINTER_32,  # numpy.int_  # default for arange on linux and windows
    8: INVALID_POINTER_U32,  # numpy.uint_
    9: INVALID_POINTER_64,  # numpy.int64
    10: INVALID_POINTER_U64,  # np.uint64
    11: np.nan,  # np.float32
    12: np.nan,  # np.float64
    13: np.nan,  # np.float64
    # 14: numpy.complex64,   # np.complex64
    # 15: numpy.complex128,  # np.complex128
    # 16: numpy.complex128,  # np.complex128
    17: None,  # np.object_
    18: b'',  # np.bytes_
    19: "",  # np.str_
    # 20:                    # numpy.void
    # 21:                    # numpy.datetime64
    # 22:                    # numpy.timedelta64
    23: np.nan,  # numpy.float16
}

if sys.platform != 'win32':
    INVALID_DICT: Mapping[int, Any] = {
        # keys in this dict can be generated with ndarray.dtype.num
        0: False,  # np.bool
        1: -128,  # np.int8
        2: 255,  # numpy.uint8,
        3: -32768,  # numpy.int16
        4: 65535,  # numpy.uint16
        5: INVALID_POINTER_32,  # numpy.int32
        6: INVALID_POINTER_U32,  # numpy.uint32
        7: INVALID_POINTER_64,  # numpy.int_  # default for arange on linux and windows
        8: INVALID_POINTER_U64,  # numpy.uint_
        9: INVALID_POINTER_64,  # numpy.int64
        10: INVALID_POINTER_U64,  # np.uint64
        11: np.nan,  # np.float32
        12: np.nan,  # np.float64
        13: np.nan,  # np.float64
        # 14: numpy.complex64,   # np.complex64
        # 15: numpy.complex128,  # np.complex128
        # 16: numpy.complex128,  # np.complex128
        17: None,  # np.object_
        18: b'',  # np.bytes_
        19: "",  # np.str_
        # 20:                    # numpy.void
        # 21:                    # numpy.datetime64
        # 22:                    # numpy.timedelta64
        23: np.nan,  # numpy.float16
    }


# See DatasetRW.h
# Official list of data types
class SM_DTYPES(IntEnum):
    DT_INVALID = 0
    DT_BOOL = 1
    DT_BYTE = 2
    DT_INT8 = 3
    DT_INT16 = 4
    DT_INT32 = 5
    DT_INT64 = 6
    DT_UINT8 = 7
    DT_UINT16 = 8
    DT_UINT32 = 9
    DT_UINT64 = 10
    DT_FLOAT16 = 11
    DT_FLOAT32 = 12
    DT_FLOAT64 = 13
    DT_OBJECT = 17
    DT_BYTES = 18  # fixed size ascii string uses this
    DT_UNICODE = 19
    DT_NPVOID = 20
    DT_DATETIME64 = 21
    DT_TIMEDELTA64 = 22
    DT_HALF = 23
    DT_CHARARRAY = 24  # how we like to store a fixed size ascii


class SD_TYPES(IntEnum):
    SD_UNKNOWN = 0

    # Matlab based
    SD_FUNCTIONH = 1
    SD_DATASET = 2
    SD_CLASS = 3
    SD_STRUCT = 4
    SD_SCALAR = 5
    SD_CHAR = 6
    SD_LOGICAL = 7
    SD_CELL = 8
    SD_NUMERIC = 9
    SD_VECTOR = 10

    # Python based
    SD_PANDAS = 20
    SD_NUMPY = 21


class MATH_OPERATION(IntEnum):
    """
    MATH_OPERATION is the encoding of the Riptable implemented mathematical operations.
    """
    # MATH_OPERATION is how Riptable communicates with RiptideCPP and this enumeration
    # is repeated in RiptideCPP CommonInc.h. Any changes made to either location must be
    # reflected in both places.
    # Two ops, returns same type
    ADD = 1
    SUB = 2
    MUL = 3
    MOD = 5
    MIN = 6
    MAX = 7
    NANMIN = 8
    NANMAX = 9
    FLOORDIV = 10
    POWER = 11
    REMAINDER = 12
    FMOD = 13

    # Two ops always return a float
    DIV = 101
    SUBDATETIMES = 102  # will check both sides for INV/ZEROS input:int64/int32 (returns double)
    # SUBDATETIMESRIGHT = 103,  # will check right hand side for ZEROS input:int64/int32 (returns double)
    SUBDATES = 103  # will check both sides for INV/ZEROS same dtype in/out (int32/int64)
    # ADDDATES = 105           # will check left hand side for ZEROS same dtype in/out (int32/int64)

    # One input returns same data type
    ABS = 201
    NEG = 202
    FABS = 203
    INVERT = 204
    FLOOR = 205
    CEIL = 206
    TRUNC = 207
    ROUND = 208
    NEGATIVE = 212
    POSITIVE = 213
    SIGN = 214
    RINT = 215
    EXP = 216
    EXP2 = 217
    # does not allow floats
    BITWISE_NOT = 218

    # One input always return a float one input
    SQRT = 301
    LOG = 302
    LOG2 = 303
    LOG10 = 304
    EXPM1 = 305
    LOG1P = 306
    SQUARE = 307
    CBRT = 308
    RECIPROCAL = 309

    # Two inputs Always return a bool
    CMP_EQ = 401
    CMP_NE = 402
    CMP_LT = 403
    CMP_GT = 404
    CMP_LTE = 405
    CMP_GTE = 406
    LOGICAL_AND = 407
    LOGICAL_XOR = 408
    LOGICAL_OR = 409

    # Two inputs
    BITWISE_LSHIFT = 501
    BITWISE_RSHIFT = 502
    BITWISE_AND = 503
    BITWISE_XOR = 504
    BITWISE_OR = 505
    BITWISE_ANDNOT = 506
    BITWISE_NOTAND = 507

    BITWISE_XOR_SPECIAL = 550

    # one input output bool
    LOGICAL_NOT = 601
    ISINF = 603
    ISNAN = 604
    ISFINITE = 605
    ISNORMAL = 606
    ISNOTINF = 607
    ISNOTNAN = 608
    ISNOTFINITE = 609
    ISNOTNORMAL = 610
    ISNANORZERO = 611
    SIGNBIT = 612


gBinaryUFuncs: Mapping[Callable, Optional[MATH_OPERATION]] = {
    # math ops
    np.add: MATH_OPERATION.ADD,
    np.subtract: MATH_OPERATION.SUB,
    np.multiply: MATH_OPERATION.MUL,
    np.matmul: None,
    np.divide: MATH_OPERATION.DIV,
    np.true_divide: MATH_OPERATION.DIV,
    np.floor_divide: MATH_OPERATION.FLOORDIV,
    np.remainder: MATH_OPERATION.REMAINDER,
    np.fmod: MATH_OPERATION.FMOD,
    np.mod: MATH_OPERATION.MOD,
    np.power: MATH_OPERATION.POWER,
    np.minimum: MATH_OPERATION.MIN,
    np.maximum: MATH_OPERATION.MAX,
    np.fmin: MATH_OPERATION.NANMIN,
    np.fmax: MATH_OPERATION.NANMAX,
}
"""
The mapping of Numpy to Riptable arithmetic binary operator overrides.

See Also
--------
MATH_OPERATION : the full set of mathematical operations supported by Riptable.
"""


gBinaryLogicalUFuncs: Mapping[Callable, MATH_OPERATION] = {
    # comparisons
    np.less_equal: MATH_OPERATION.CMP_LTE,
    np.less: MATH_OPERATION.CMP_LT,
    np.equal: MATH_OPERATION.CMP_EQ,
    np.not_equal: MATH_OPERATION.CMP_NE,
    np.greater: MATH_OPERATION.CMP_GT,
    np.greater_equal: MATH_OPERATION.CMP_GTE,
    np.logical_and: MATH_OPERATION.LOGICAL_AND,
    np.logical_xor: MATH_OPERATION.LOGICAL_XOR,
    np.logical_or: MATH_OPERATION.LOGICAL_OR,
}
"""
The mapping of Numpy to Riptable comparison function overrides.

See Also
--------
MATH_OPERATION : the full set of mathematical operations supported by Riptable.
"""


gBinaryBitwiseUFuncs: Mapping[Callable, Optional[MATH_OPERATION]] = {
    # bitwise operations only apply to bool and integers
    np.left_shift: None,
    np.right_shift: None,
    np.bitwise_and: MATH_OPERATION.BITWISE_AND,
    np.bitwise_xor: MATH_OPERATION.BITWISE_XOR,
    np.bitwise_or: MATH_OPERATION.BITWISE_OR,
}
"""
The mapping of Numpy to Riptable bit-twiddling binary operator overrides.

See Also
--------
MATH_OPERATION : the full set of mathematical operations supported by Riptable.
"""


gBinaryBitwiseMonoUFuncs: Mapping[Callable, Optional[MATH_OPERATION]] = {
    # bitwise operations only apply to bool and integers
    np.invert: None
}
"""
The mapping of Numpy to Riptable bit-twiddling unary operator overrides.

See Also
--------
MATH_OPERATION : the full set of mathematical operations supported by Riptable.
"""


gUnaryUFuncs: Mapping[Callable, Optional[MATH_OPERATION]] = {
    # math ops
    np.absolute: MATH_OPERATION.ABS,
    np.abs: MATH_OPERATION.ABS,
    np.fabs: MATH_OPERATION.FABS,
    np.invert: MATH_OPERATION.INVERT,
    np.floor: MATH_OPERATION.FLOOR,
    np.ceil: MATH_OPERATION.CEIL,
    np.trunc: MATH_OPERATION.TRUNC,
    np.round: MATH_OPERATION.ROUND,
    np.rint: MATH_OPERATION.ROUND,
    np.isinf: MATH_OPERATION.ISINF,
    np.isnan: MATH_OPERATION.ISNAN,
    np.isfinite: MATH_OPERATION.ISFINITE,
    np.signbit: MATH_OPERATION.SIGNBIT,
    np.negative: MATH_OPERATION.NEGATIVE,
    np.positive: MATH_OPERATION.POSITIVE,
    np.sign: MATH_OPERATION.SIGN,
    np.exp: MATH_OPERATION.EXP,
    np.exp2: MATH_OPERATION.EXP2,
    np.log: MATH_OPERATION.LOG,
    np.log2: MATH_OPERATION.LOG2,
    np.log10: MATH_OPERATION.LOG10,
    np.expm1: None,
    np.log1p: None,
    np.sqrt: MATH_OPERATION.SQRT,
    np.square: None,
    np.cbrt: MATH_OPERATION.CBRT,
    np.reciprocal: None,
    np.logical_not: MATH_OPERATION.LOGICAL_NOT,
    np.bitwise_not: MATH_OPERATION.BITWISE_NOT,
    np.signbit: MATH_OPERATION.SIGNBIT,
}
"""
The mapping of Numpy to Riptable arithmetic unary operator overrides.

See Also
--------
MATH_OPERATION : the full set of mathematical operations supported by Riptable.
"""


gNanFuncs: Mapping[Callable, Optional[Callable]] = {
    np.nanargmax: None,
    np.nanargmin: None,
    np.nancumprod: None,
    np.nancumsum: None,
    np.nanmax: None,
    np.nanmean: None,
    np.nanmedian: None,  # bug in numpy for floats?
    np.nanmin: None,
    np.nanpercentile: None,
    np.nanprod: None,
    np.nanstd: None,
    np.nansum: None,
    np.nanvar: None,
}


# See GroupBy.h
# Official list of funcss
class ROLLING_FUNCTIONS(IntEnum):
    ROLLING_SUM = 0
    ROLLING_NANSUM = 1
    ROLLING_MEAN = 102
    ROLLING_NANMEAN = 103
    ROLLING_VAR = 106
    ROLLING_NANVAR = 107
    ROLLING_STD = 108
    ROLLING_NANSTD = 109


class REDUCE_FUNCTIONS(IntEnum):
    REDUCE_SUM = 0
    REDUCE_NANSUM = 1
    REDUCE_MEAN = 102
    REDUCE_NANMEAN = 103
    REDUCE_VAR = 106
    REDUCE_NANVAR = 107
    REDUCE_STD = 108
    REDUCE_NANSTD = 109
    REDUCE_MIN = 200
    REDUCE_NANMIN = 201
    REDUCE_MAX = 202
    REDUCE_NANMAX = 203
    REDUCE_ARGMIN = 204
    REDUCE_NANARGMIN = 205
    REDUCE_ARGMAX = 206
    REDUCE_NANARGMAX = 207

    # for Jack TODO
    REDUCE_ANY = 208
    REDUCE_ALL = 209


gReduceUFuncs: Mapping[Callable, REDUCE_FUNCTIONS] = {
    np.add: REDUCE_FUNCTIONS.REDUCE_SUM,
    np.minimum: REDUCE_FUNCTIONS.REDUCE_MIN,
    np.maximum: REDUCE_FUNCTIONS.REDUCE_MAX
    ## TODO add support for the following reduce ufuncs
    #   np.logical_and: RIPTIDE NP.ALL
    #   np.logical_or:  RIPTIDE NP.ANY
}
"""
The mapping of Numpy to Riptable reduce function overrides.

See Also
--------
REDUCE_FUNCTIONS encoding of all reduce functions that riptable supports
"""


class TIMEWINDOW_FUNCTIONS(IntEnum):
    TIMEWINDOW_SUM = 0
    TIMEWINDOW_PROD = 1


# See GroupBy.h
# Official list of funcss
class GB_FUNCTIONS(IntEnum):
    GB_SUM = 0
    GB_MEAN = 1
    GB_MIN = 2
    GB_MAX = 3

    # STD uses VAR with the param set to 1
    GB_VAR = 4
    GB_STD = 5

    GB_NANSUM = 50
    GB_NANMEAN = 51
    GB_NANMIN = 52
    GB_NANMAX = 53
    GB_NANVAR = 54
    GB_NANSTD = 55

    GB_FIRST = 100
    GB_NTH = 101
    GB_LAST = 102
    GB_MEDIAN = 103  # auto handles nan
    GB_MODE = 104  # auto handles nan
    GB_TRIMBR = 105  # auto handles nan

    # All int/uints output upgraded to INT64
    # Output is all elements (not just grouped)
    GB_ROLLING_SUM = 200
    GB_ROLLING_NANSUM = 201
    GB_ROLLING_DIFF = 202
    GB_ROLLING_SHIFT = 203
    GB_ROLLING_COUNT = 204
    GB_ROLLING_MEAN = 205
    GB_ROLLING_NANMEAN = 206

    # In ema.cpp
    GB_CUMSUM = 300
    GB_EMADECAY = 301
    GB_CUMPROD = 302
    GB_FINDNTH = 303
    GB_EMANORMAL = 304
    GB_EMAWEIGHTED = 305


# some groupby functions will work for strings
GB_STRING_ALLOWED = [
    GB_FUNCTIONS.GB_FIRST,
    GB_FUNCTIONS.GB_NTH,
    GB_FUNCTIONS.GB_LAST,
    GB_FUNCTIONS.GB_ROLLING_SHIFT,
    GB_FUNCTIONS.GB_MODE,
]
GB_DATE_ALLOWED = [
    GB_FUNCTIONS.GB_FIRST,
    GB_FUNCTIONS.GB_NTH,
    GB_FUNCTIONS.GB_LAST,
    GB_FUNCTIONS.GB_MEAN,
    GB_FUNCTIONS.GB_MIN,
    GB_FUNCTIONS.GB_MAX,
    GB_FUNCTIONS.GB_NANMEAN,
    GB_FUNCTIONS.GB_NANMIN,
    GB_FUNCTIONS.GB_NANMAX,
    GB_FUNCTIONS.GB_MEDIAN,
    GB_FUNCTIONS.GB_MODE,
    GB_FUNCTIONS.GB_ROLLING_DIFF,
    GB_FUNCTIONS.GB_ROLLING_SHIFT,
]

GB_FUNC_COUNT = -1
GB_FUNC_USER = 900
GB_FUNC_NUMBA = 1000


######################################################
# Numba groupby enums
######################################################
class GB_PACKUNPACK(IntEnum):
    UNPACK = 0
    PACK = 1


class NumpyCharTypes:
    All = '?bhilqpBHILQPefdgFDGSUVOMm'
    AllFloat = 'efdgFDG'
    AllInteger = 'bBhHiIlLqQpP'
    Computable = 'fdgbBhHiIlLqQpP'  # does not include boolean or strings
    Noncomputable = 'SeFDGUVOMm'
    Unsupported = 'eFDGVOMm'  # unsupported in riptable world
    Supported = '?fdgbBhHiIlLqQpPSUV'
    SupportedFloat = 'fdg'
    SupportedAlternate = '?fdgbBhHiIlLqQpPSU'
    Character = 'c'
    Complex = 'FDG'
    Datetime = 'Mm'
    Float = 'efdg'
    Float64 = 'dg'
    Integer = 'bhilqp'
    UnsignedInteger = 'BHILQP'
    UnsignedInteger64 = 'QP'
    SignedInteger64 = 'qp'

    # linux gcc compiler long is int64, msvc long is int32
    if sys.platform != 'win32':
        UnsignedInteger64 = 'LQP'
        SignedInteger64 = 'lqp'


gScalarType: Tuple[type, ...] = (
    int,
    float,
    complex,
    bool,
    bytes,
    str,
    memoryview,
    np.bool_,
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    np.object_,
    np.bytes_,
    np.str_,
    np.void,
    np.datetime64,
    np.timedelta64,
)


gNumpyScalarType: Tuple[type, ...] = (
    memoryview,
    np.bool_,
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    np.object_,
    np.bytes_,
    np.str_,
    np.void,
    np.datetime64,
    np.timedelta64,
)


def int_dtype_from_len(newlen: int) -> np.dtype:
    """Returns minimum itemsize unsigned integer dtype for given array length.
    Assumes that numbers up to the length will need to be stored by the returned dtype.
    Used by Grouping and Categorical.
    """
    if newlen < 100:
        dt = np.int8
    elif newlen < 30_000:
        dt = np.int16
    elif newlen < 2_000_000_000:
        dt = np.int32
    else:
        dt = np.int64
    return np.dtype(dt)


gAnsiColors: Mapping[str, str] = {
    'Black': '\x1b[0;30m',
    'BlinkBlack': '\x1b[5;30m',
    'BlinkBlue': '\x1b[5;34m',
    'BlinkCyan': '\x1b[5;36m',
    'BlinkGreen': '\x1b[5;32m',
    'BlinkLightGray': '\x1b[5;37m',
    'BlinkPurple': '\x1b[5;35m',
    'BlinkRed': '\x1b[5;31m',
    'BlinkYellow': '\x1b[5;33m',
    'Blue': '\x1b[0;34m',
    'Brown': '\x1b[0;33m',
    'Cyan': '\x1b[0;36m',
    'DarkGray': '\x1b[1;30m',
    'Green': '\x1b[0;32m',
    'LightBlue': '\x1b[1;34m',
    'LightCyan': '\x1b[1;36m',
    'LightGray': '\x1b[0;37m',
    'LightGreen': '\x1b[1;32m',
    'LightPurple': '\x1b[1;35m',
    'LightRed': '\x1b[1;31m',
    'NoColor': '',
    'Normal': '\x1b[0m',
    'Purple': '\x1b[0;35m',
    'Red': '\x1b[0;31m',
    'White': '\x1b[1;37m',
    'Yellow': '\x1b[1;33m',
}


class DS_DISPLAY_TYPES(IntEnum):
    HTML = 1
    REPR = 2
    STR = 3


class DATETIME_TYPES(IntEnum):
    ORDINAL_DATE = 1


DateTimeFormats: Mapping[str, str] = {'day': "%d-%b-%Y"}


class DisplayDetectModes(IntEnum):
    Jupyter = 1
    Ipython = 2
    Console = 3
    HTML = 5


class DisplayArrayTypes(IntEnum):
    Bool = 0
    Integer = 1
    Float = 2
    Bytes = 3
    Categorical = 4
    String = 5
    DateTime = 6
    DateTimeBase = 7
    DateTimeNano = 9
    TimeSpan = 10
    Record = 11


class DisplayLength(IntEnum):
    Undefined = 0
    Short = 1
    Medium = 2
    Long = 3


class TimeFormat(IntEnum):
    Clock = 1
    YearMonthDay = 2
    SIGNano = 3


class DisplayJustification(IntEnum):
    Undefined = 0
    Left = 1
    Right = 2
    Center = 3


class DisplayTextDecoration(IntEnum):
    Undefined = 0
    Bold = 1
    Italic = 2
    Underline = 3
    Strikethrough = 4


class DisplayNumberSeparator:
    NoSeparator = ""
    Comma = ","
    # Period      = "." #BUG, fix later
    Underscore = "_"


class DisplayColumnColors(IntEnum):
    Default = 0  # no styling
    Rownum = 1  # row numbers / default header color
    Sort = 2  # regular sort header and column data
    Groupby = 3  # groupby header and column data
    Multiset_head_a = 4  # comparison color for multiset columns headers
    Multiset_head_b = 5  # comparison color for multiset columns headers
    Multiset_col_a = 6  # comparison color for multiset column data
    Multiset_col_b = 7  # comparison color for multiset column data
    Accum2t = 8
    Purple = 9
    Pink = 10
    Red = 11
    GrayItalic = 12
    DarkBlue = 13
    BGColor = 14
    FGColor = 15


class ColumnStyle:
    """
    Holds display styles for entire columns or individual cells.
    These styles will override the defaults from the _display_query_properties callback in FastArray
    See also: DisplayColumn, DisplayCell, ItemFormat

    properties:
    color      : DisplayColumnColors
    align      : DisplayJustification
    decoration : DisplayTextDecoration
    width      : DisplayLength (default) OR can be set to new max width for array item's string repr

    """

    def __init__(
        self,
        color=DisplayColumnColors.Default,
        align=DisplayJustification.Right,
        decoration=DisplayTextDecoration.Undefined,
        width=None,
    ):
        self.color = color
        self.align = align
        self.decoration = decoration
        self.width = width

    def _build_string(self):
        repr_str = []
        repr_str.append(f"     {self.__class__.__name__}")
        repr_str.append(f"     color: {DisplayColumnColors(self.color).name}")
        repr_str.append(f"     align: {DisplayJustification(self.align).name}")
        repr_str.append(f"decoration: {DisplayTextDecoration(self.decoration).name}")
        repr_str.append(f"     width: {self.width}")
        return "\n".join(repr_str)

    def __repr__(self):
        return self._build_string()

    def __str__(self):
        return self._build_string()


class DisplayColorMode(IntEnum):
    NoColors = 0
    Light = 1
    Dark = 2


class CategoryMode(IntEnum):
    Default = 0
    StringArray = 1
    IntEnum = 2
    Dictionary = 3
    NumericArray = 4
    MultiKey = 5


class CategoricalOrigin(IntEnum):
    CategoricalView = 0
    CategoricalCopy = 1
    StringList = 2
    StringListWithCategories = 3
    NumericList = 4
    IndexWithCategories = 5
    CodeMapping = 6
    Multikey = 7
    Matlab = 8
    Pandas = 9
    SDSFile = 10


class CategoryStringMode(IntEnum):
    Default = 0
    Bytes = 1
    Unicode = 2


class CategoricalConstructor(IntEnum):
    EmptyValues = 0
    IntegerValues = 1
    FloatValues = 2
    StringValues = 3
    MultikeyListValues = 4
    MultikeyDictValues = 5


class ApplyType(IntEnum):
    Invalid = 0
    ReduceDataset = 1
    ReduceList = 2
    Dataset = 3
    Arrays = 4


class CompressionMode(IntEnum):
    Compress = 0
    Decompress = 1
    CompressFile = 2
    DecompressFile = 3
    SharedMemory = 4
    Info = 5


class CompressionType(IntEnum):
    Uncompressed = 0
    ZStd = 1


class ColumnAttribute(IntEnum):
    Default = 0
    Left = 1
    Right = 2


class SDSFlag(IntEnum):
    OriginalContainer = 0x01
    Stackable = 0x02
    Scalar = 0x04
    Nested = 0x08
    Meta = 0x10


class SDSFileType(IntEnum):
    Unknown = 0
    Struct = 1
    Dataset = 2
    Table = 3
    Array = 4
    OneFile = 5  # new for one file


class DayOfWeek(IntEnum):
    Monday = 0
    Tuesday = 1
    Wednesday = 2
    Thursday = 3
    Friday = 4
    Saturday = 5
    Sunday = 6


# allowing / now for denest
INVALID_FILE_CHARS: Tuple[str, ...] = ('\\', ':', '<', '>', '!', '|', '*', '?')


gBasicStats: Mapping[str, str] = {
    'count': 'Number of non-null observations',
    'sum': 'Sum of values',
    'mean': 'Mean of values',
    'mad': 'Mean absolute deviation',  # median(abs(a - median(a)))
    'median': 'Arithmetic median of values',
    'min': 'Minimum',
    'max': 'Maximum',
    'std': 'Unbiased standard deviation',
    'var': 'Unbiased variance',
    'nansum': 'Sum of values',
    'nanmean': 'Mean of values',
    'nanmad': 'Mean absolute deviation',
    'nanmedian': 'Arithmetic median of values',
    'nanmin': 'Minimum',
    'nanmax': 'Maximum',
    'nanstd': 'Unbiased standard deviation',
    'nanvar': 'Unbiased variance',
    'mode': 'Mode',
    'abs': 'Absolute Value',
    'prod': 'Product of values',
    'sem': 'Unbiased standard error of the mean',
    'skew': 'Unbiased skewness (3rd moment)',
    'kurt': 'Unbiased kurtosis (4th moment)',
    'quantile': 'Sample quantile (value at %)',
    'cumsum': 'Cumulative sum',
    'cumprod': 'Cumulative product',
    'cummax': 'Cumulative maximum',
    'cummin': 'Cumulative minimum',
}


#####################################################################################
# Structs begin
#####################################################################################


# Used in display table for multi-line column headers
# color_group indexing starts at 0
# cell_span cannot be 0.  a cell_span of 1 indicates 1 cell wide.
ColHeader = namedtuple('ColHeader', ['col_name', 'cell_span', 'color_group'])


###################################
## TJD NOTE: Need to use strings instead of enum here
##################################
class TypeId(IntEnum):
    Default = 0
    Struct = 1
    Dataset = 2
    Multiset = 3
    GroupBy = 4
    Grouping = 5
    FastArray = 6
    MathLedger = 7
    Categorical = 8
    Categories = 9
    Accum2 = 10
    DisplayDetect = 11
    DisplayOptions = 12
    DisplayTable = 13
    SortCache = 14
    DateTimeBase = 15
    DateTimeNano = 17
    TimeSpan = 18
    TimeZone = 19
    Calendar = 20
    Date = 21
    DateSpan = 22
    PDataset = 23


######################################################
# SDS File Header order
######################################################
gSDSFileHeader: List[str] = [
    'SDSHeaderMagic',
    'VersionHigh',
    'VersionLow',
    'CompMode',
    'CompType',
    'CompLevel',
    # ----- offset 16 -----
    'NameBlockSize',
    'NameBlockOffset',
    'NameBlockCount',
    'FileType',  # struct, dataset
    'AuthorId',  # python, matlab
    # ----- offset 48 -----
    'MetaBlockSize',
    'MetaBlockOffset',
    # ----- offset 64 -----
    'TotalMetaCompressedSize',
    'TotalMetaUncompressedSize',
    # ----- offset 80 -----
    'ArrayBlockSize',
    'ArrayBlockOffset',
    # ----- offset 96 -----
    'ArraysWritten',
    'ArrayFirstOffset',
    # ----- offset 112 -----
    'TotalArrayCompressedSize',
    'TotalArrayUncompressedSize',
]
SDS_EXTENSION: str = '.sds'
SDS_EXTENSION_BYTES: bytes = b'.sds'


# please keep the TypeRegister at the end of the file
class TypeRegister:
    '''
    When special classes are loaded, they register with this class to avoid cyclical dependencies
    '''

    Struct = None
    Dataset = None
    Multiset = None
    GroupBy = None
    Grouping = None
    FastArray = None
    MathLedger = None
    Categorical = None
    Categories = None
    Accum2 = None
    DisplayDetect = None
    DisplayOptions = None
    DisplayTable = None
    DisplayString = None
    DisplayAttributes = None
    DisplayText = None
    SortCache = None
    DateTimeBase = None
    DateTimeNano = None
    TimeSpan = None
    SharedMemory = None
    TimeZone = None
    Calendar = None
    DateBase = None
    Date = None
    DateSpan = None
    PDataset = None

    @classmethod
    def validate_registry(cls):
        missing = set()
        for _nm in dir(cls):
            if not _nm.startswith('_') and getattr(TypeRegister, _nm) is None:
                missing.add(_nm)
        if len(missing) > 0:
            msg = ', '.join(sorted(missing))
            raise RuntimeError(f'riptable: Improper initialization!  Missing: {msg}')

    @classmethod
    def is_computable(cls, other):
        if not (other.dtype.char in NumpyCharTypes.Noncomputable or isinstance(other, cls.Categorical)):
            return True
        return False

    @classmethod
    def is_array_subclass(cls, arr):
        '''
        Certain routines can be sped up by skipping the logic before falling back on a numpy call.
        Note: this is different than using python's issubclass(), which returns True if the classes are the same.
        Returns True if the item is an instance of a FastArray or numpy array subclass.
        '''
        if isinstance(arr, np.ndarray):
            if type(arr) == np.ndarray or type(arr) == cls.FastArray:
                return False
            return True
        else:
            return False

    @classmethod
    def is_binned_array(cls, arr):
        '''
        Use this instead of checking isinstance(item, TypeRegister.Categorical). For other binned 
        types in the future.

        Called by:
        Dataset.melt()             -re-expands
        Dataset.from_jagged_rows() -re-expands
        GroupBy.__init__           -calls grouping, gb_keychain properties to borrow bins

        '''
        return isinstance(arr, cls.Categorical)

    @classmethod
    def is_binned_type(cls, arrtype):
        '''
        Check the type rather than the instance.
        See also is_binned_array()

        Called by:
        rt_utils._multistack_items()
        '''
        return arrtype == cls.Categorical

    # ---------------------------------------------------------------
    @classmethod
    def is_spanlike(cls, arr: np.ndarray):
        ''' return True if it is a datespan or timespan '''
        # TODO: datetime/span are computable sometimes... need a way to distinguish from other FA subclasses
        # simple math works, but not larger groupby operations like sum
        result = False
        if isinstance(arr, (TypeRegister.TimeSpan, TypeRegister.DateSpan)):
            result = True
        return result

    # ---------------------------------------------------------------
    @classmethod
    def is_datelike(cls, arr: np.ndarray):
        ''' return True if it is a date or time '''
        # TODO: datetime/span are computable sometimes... need a way to distinguish from other FA subclasses
        # simple math works, but not larger groupby operations like sum
        result = False
        if isinstance(
            arr, (TypeRegister.DateTimeNano, TypeRegister.TimeSpan, TypeRegister.Date, TypeRegister.DateSpan)
        ):
            result = True
        return result

    @classmethod
    def is_string_or_object(cls, arr):
        return cls.is_array_subclass(arr) or arr.dtype.char in 'OSU'

    @classmethod
    def newclassfrominstance(cls, instance, origin):
        """After slicing or an array routine, return a new instance of a FastArray subclass.
        If the array was not a subclass, instance is unchanged.

        Parameters
        ----------
        instance : ndarray
            Array generated from operation.
        origin : ndarray
            Array, possibly a FastArray subclass.

        Returns
        -------
        instance : ndarray
            Array of the same class as origin if the origin class has a newclassfrominstance defined.
        """
        # FastArray subclasses should define this classmethod to return
        # a new object with a different instance array
        if hasattr(origin, 'newclassfrominstance'):
            instance = origin.newclassfrominstance(instance, origin)
        return instance

    # put these here for now
    # where should they live?
    @classmethod
    def as_meta_data(cls, obj):
        pass

    @classmethod
    def from_meta_data(cls, itemdict: Optional[dict] = None, flags: Optional[list] = None, meta: str = ""):
        from .Utils.rt_metadata import MetaData

        if itemdict is None:
            itemdict = dict()
        if flags is None:
            flags = list()
        meta = MetaData(meta)
        iclass = meta.itemclass
        return iclass._from_meta_data(itemdict, flags, meta)
