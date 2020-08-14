import numpy as np
from ..rt_enum import (
    TypeRegister,
    DisplayJustification,
    DisplayLength,
    DisplayArrayTypes,
    DisplayColumnColors,
    DisplayTextDecoration,
    NumpyCharTypes,
    DisplayNumberSeparator,
    INVALID_DICT,
    INVALID_SHORT_NAME,
    INVALID_LONG_NAME,
)


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
class ItemFormat:
    '''
    A container for display options for different data types in FastArray and numpy arrays.
    Basic numpy types have defaults. (see below)
    New types (subclassed from FastArray) will be queried to get formatting options.
    New types have the option of overwriting display_query_properties to set their own defaults.
    '''

    def __init__(
        self,
        length=DisplayLength.Short,
        justification=DisplayJustification.Right,
        invalid=None,
        can_have_spaces=False,
        html=False,
        color=DisplayColumnColors.Default,
        decoration=None,
        convert=None,
        convert_format=None,
        format_string=None,
        timezone_str=None,
        maxwidth=None,
    ):

        self.length = length
        self.justification = justification
        self.invalid = invalid
        self.can_have_spaces = can_have_spaces
        self.html = html
        self.color = color
        self.decoration = decoration
        self.convert = convert
        self.convert_format = convert_format
        self.format_string = format_string
        self.timezone_str = timezone_str
        self.maxwidth = maxwidth

    # copy needs to be made for correct integer invalid
    def copy(self):
        new_format = ItemFormat(
            length=self.length,
            justification=self.justification,
            invalid=self.invalid,
            can_have_spaces=self.can_have_spaces,
            html=self.html,
            color=self.color,
            decoration=self.decoration,
            convert=self.convert,
            convert_format=self.convert_format,
            format_string=self.format_string,
        )
        return new_format

    # dump formatting options for debugging
    def __repr__(self):
        return self.summary()

    def __str__(self):
        return self.summary()

    def summary(self):
        info_str = ["\n     ItemFormat:"]
        info_str.append("         Length: " + str(self.length))
        info_str.append("  Justification: " + str(self.justification))
        info_str.append("        Invalid: " + str(self.invalid))
        info_str.append("Can have spaces: " + str(self.can_have_spaces))
        info_str.append("          Color: " + str(self.color))
        info_str.append("     Decoration: " + str(self.decoration))
        return "\n".join(info_str)


# ---------------------------------------------------------------------
default_item_formats = {
    DisplayArrayTypes.Bool: ItemFormat(
        length=DisplayLength.Long,
        justification=DisplayJustification.Right,
        invalid=None,
        can_have_spaces=False,
        html=False,
        color=DisplayColumnColors.Default,
        decoration=None,
    ),
    DisplayArrayTypes.Integer: ItemFormat(
        length=DisplayLength.Short,
        justification=DisplayJustification.Right,
        invalid=None,
        can_have_spaces=False,
        html=False,
        color=DisplayColumnColors.Default,
        decoration=None,
    ),
    DisplayArrayTypes.Float: ItemFormat(
        length=DisplayLength.Short,
        justification=DisplayJustification.Right,
        invalid=None,
        can_have_spaces=False,
        html=False,
        color=DisplayColumnColors.Default,
        decoration=None,
    ),
    DisplayArrayTypes.Bytes: ItemFormat(
        length=DisplayLength.Short,
        justification=DisplayJustification.Left,
        invalid=None,
        can_have_spaces=True,
        html=False,
        color=DisplayColumnColors.Default,
        decoration=None,
    ),
    DisplayArrayTypes.String: ItemFormat(
        length=DisplayLength.Short,
        justification=DisplayJustification.Left,
        invalid=None,
        can_have_spaces=True,
        html=False,
        color=DisplayColumnColors.Default,
        decoration=None,
    ),
    DisplayArrayTypes.Record: ItemFormat(
        length=DisplayLength.Short,
        justification=DisplayJustification.Left,
        invalid=None,
        can_have_spaces=True,
        html=False,
        color=DisplayColumnColors.Default,
        decoration=None,
    ),
}


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
class DisplayConvert:
    '''
    Will analyze an array of a default type and return the appropriate conversion function.
    Anything that subclasses from FastArray will always fall back on the dtype of its underlying array.
    '''

    convert_func_dict = {}

    # cache the array's dtype char to save type and conversion function
    # this will also be saved for any other occurrences of the same array type in a different dataset
    ConvertTypeCache = {}
    ConvertFuncCache = {}

    Verbose = 0

    def __init__(self):
        pass

    @staticmethod
    def get_display_convert(arr):
        if hasattr(arr, 'dtype'):
            # check to see if the conversion function has been cached
            if arr.dtype in DisplayConvert.ConvertTypeCache:
                if DisplayConvert.Verbose > 1:
                    print("pulling conversion function from cache")
                arr_type = DisplayConvert.ConvertTypeCache[arr.dtype.char]
                convert_func = DisplayConvert.ConvertFuncCache[arr.dtype.char]
            else:
                arr_type = DisplayConvert.get_display_array_type(arr.dtype)
                convert_func = DisplayConvert.convert_func_dict.get(
                    arr_type, DisplayConvert.convertDefault
                )
                # store the array type and conversion function into a cache so the lookup only happens once
                if DisplayConvert.Verbose > 1:
                    print("adding new conversion function to cache")
                DisplayConvert.ConvertTypeCache[arr.dtype.char] = arr_type
                DisplayConvert.ConvertFuncCache[arr.dtype.char] = convert_func
        else:
            # TODO: expand to handle python list-like objects
            arr_type = -1
            convert_func = DisplayConvert.convertDefault
        return arr_type, convert_func

    # ---------------------------------------------------------------------
    @staticmethod
    def get_display_array_type(dtype):
        '''
        For FastArray and numpy array of basic type (not for objects that are subclasses of FastArray)
        '''
        dtype_char = dtype.char
        if dtype_char == '?':
            return DisplayArrayTypes.Bool

        elif dtype_char in NumpyCharTypes.AllInteger:
            return DisplayArrayTypes.Integer

        elif dtype_char in NumpyCharTypes.AllFloat:
            return DisplayArrayTypes.Float

        elif dtype_char == 'S':
            return DisplayArrayTypes.Bytes

        elif dtype_char == 'U':
            return DisplayArrayTypes.String

        elif dtype_char == 'V':
            return DisplayArrayTypes.Record

        # will use overall defaults
        else:
            return -1

    # ------------DEFAULT FORMATTING FUNCTIONS-----------------------------
    # ---------------------------------------------------------------------
    @staticmethod
    def convertBool(b, itemformat: ItemFormat):
        if DisplayConvert.Verbose > 0:
            print("***convert bool")
        result = str(b)
        if itemformat.length == DisplayLength.Long:
            return result
        else:
            return result[0]  # T/F

    # ---------------------------------------------------------------------
    @staticmethod
    def convertInt(i, itemformat: ItemFormat):
        if DisplayConvert.Verbose > 0:
            print("***convert int")
        if itemformat.invalid is not None:
            if i == itemformat.invalid:
                if DisplayConvert.Verbose > 0:
                    print("***found invalid int")
                if itemformat.length == DisplayLength.Short:
                    return INVALID_SHORT_NAME
                else:
                    return INVALID_LONG_NAME
        if TypeRegister.DisplayOptions.NUMBER_SEPARATOR:
            separator_string = (
                '{:' + TypeRegister.DisplayOptions.NUMBER_SEPARATOR_CHAR + '}'
            )
            return separator_string.format(i)
        else:
            return str(i)

    # ---------------------------------------------------------------------
    @staticmethod
    def convertFloat(f, itemformat: ItemFormat):
        if DisplayConvert.Verbose > 0:
            print("***convert float")
        # potential for long/short format
        separator = ""
        if TypeRegister.DisplayOptions.NUMBER_SEPARATOR:
            separator = TypeRegister.DisplayOptions.NUMBER_SEPARATOR_CHAR

        reg_precision = TypeRegister.DisplayOptions.PRECISION
        use_sci = False
        if f == f:
            f_test = abs(f)
            # lower bound for switching to scientific notation
            e_min = TypeRegister.DisplayOptions.e_min()
            # upper bound
            e_max = TypeRegister.DisplayOptions.e_max()
            if f_test != 0:
                if (f_test <= e_min) or (f_test >= e_max):
                    use_sci = True
                # switch to scientific notation if non-zero will appear as zero
                elif f_test < TypeRegister.DisplayOptions.p_threshold():
                    use_sci = True
        if use_sci:
            precision_str = (
                "{:"
                + separator
                + "."
                + str(TypeRegister.DisplayOptions.E_PRECISION)
                + "e}"
            )
        else:
            precision_str = "{:" + separator + "." + str(reg_precision) + "f}"

        f = precision_str.format(f)
        return f

    # ---------------------------------------------------------------------
    @staticmethod
    def convertBytes(i, itemformat: ItemFormat):
        if DisplayConvert.Verbose > 0:
            print("***convert bytes")
        # need to know long or short format
        # possibly enforce global maximum string
        if len(i) > 0:
            i = bytes.decode(i, errors='ignore')
            return trim_string(i, itemformat)
        else:
            return ""

    # ---------------------------------------------------------------------
    @staticmethod
    def convertString(i, itemformat: ItemFormat):
        if DisplayConvert.Verbose > 0:
            print("***convert string")
        return trim_string(i, itemformat)

    # ---------------------------------------------------------------------
    @staticmethod
    def convertDefault(i, itemformat: ItemFormat):
        if DisplayConvert.Verbose > 0:
            print("***convert default")
        i = str(i)
        return trim_string(i, itemformat)

    # ---------------------------------------------------------------------
    @staticmethod
    def convertMultiDims(i, itemformat: ItemFormat):
        '''
        For displaying multi-dimensional arrays (currently only supports 2-dims).
        ItemFormat object contains a convert function for the dtype of the multidimensional array.
        '''
        # for lots of dims, add lots of brackets
        final_list = []
        dims = i.ndim - 1
        # reduce to single dimension
        while i.ndim > 1:
            i = i[0]
        for item in i:
            final_list.append(itemformat.convert(item, itemformat))
        final_str = '[' * dims + str(final_list).replace("'", "")
        return final_str[: TypeRegister.DisplayOptions.MAX_STRING_WIDTH]

    # ---------------------------------------------------------------------
    @staticmethod
    def convertRecord(i, itemformat: ItemFormat):
        return str(i)


def trim_string(i, itemformat: ItemFormat):
    '''
    If maxwidth was specified and is larger than the global default, it will be used instead.
    See also ColumnStyle()
    '''
    if (
        itemformat.maxwidth is not None
        and itemformat.maxwidth > TypeRegister.DisplayOptions.MAX_STRING_WIDTH
    ):
        return i[: itemformat.maxwidth]
    if itemformat.length == DisplayLength.Short:
        i = i[: TypeRegister.DisplayOptions.MAX_STRING_WIDTH]
    return i


DisplayConvert.convert_func_dict = {
    DisplayArrayTypes.Bool: DisplayConvert.convertBool,
    DisplayArrayTypes.Integer: DisplayConvert.convertInt,
    DisplayArrayTypes.Float: DisplayConvert.convertFloat,
    DisplayArrayTypes.Bytes: DisplayConvert.convertBytes,
    DisplayArrayTypes.String: DisplayConvert.convertString,
    DisplayArrayTypes.Record: DisplayConvert.convertRecord,
}


def get_array_formatter(arr):
    '''
    FastArray/subclasses have display_query_properties defined for custom string formatting.
    Numpy arrays have defaults in DisplayConvert.
    Returns ItemFormat object and display function for items in array based on type.
    '''
    if hasattr(arr, 'display_query_properties'):
        display_format, func = arr.display_query_properties()
    else:
        arr_type, func = DisplayConvert.get_display_convert(arr)
        display_format = default_item_formats.get(arr_type, ItemFormat())
    return display_format, func


def format_scalar(sc):
    """Convert a scalar to a string for display - the same way it would be if contained in a FastArray.
    Returns the converted scalar.
    """
    # consider writing a path just for scalars, wrapping in a FastArray
    arr = TypeRegister.FastArray(sc)
    display_format, func = get_array_formatter(arr)
    return func(arr[0], display_format)
