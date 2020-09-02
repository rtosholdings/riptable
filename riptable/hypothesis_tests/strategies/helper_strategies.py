import warnings
import sys
import numpy as np
import riptable as rt
from typing import Any, List, Optional, Tuple, Union
import hypothesis.extra.pandas as pdst
from hypothesis import assume
from hypothesis.strategies import (
    booleans,
    characters,
    complex_numbers,
    composite,
    floats,
    integers,
    just,
    lists,
    none,
    one_of,
    sampled_from,
    shared,
    slices,
    text,
    SearchStrategy,
)
from hypothesis.extra.numpy import (
    arrays,
    basic_indices,
    boolean_dtypes,
    byte_string_dtypes,
    complex_number_dtypes,
    datetime64_dtypes,
    floating_dtypes,
    integer_dtypes,
    timedelta64_dtypes,
    unicode_string_dtypes,
    unsigned_integer_dtypes,
)
from riptable.Utils.common import _dtypes_by_group, integer_range, dtypes_by_group
from riptable.Utils.rt_testing import name


_MAX_SHAPE_SIZE: int = 270_000  # maximum shape size (this is an arbitrary value >250k, which should put it above various length cutoffs we use)
_MAX_VALUE: int = 1_000_000  # numeric maximum and minimum magnitude value
_MAX_INT: int = sys.maxsize
_MAX_FLOAT: float = sys.float_info.max


@composite
def ndarray_shape_strategy(
    draw,
    min_rank: int = 0,
    max_rank: int = 2,
    max_shape_size: Optional[int] = None,
    max_size_multiplier: Optional[float] = None
):
    rank = draw(integers(min_value=min_rank, max_value=max_rank))

    if max_shape_size is None:
        # The max shape size is meant to be a maximum number of elements, so for higher-rank
        # arrays we need to use a smaller size-per-axis. This is only an upper bound, so for higher-rank
        # arrays this approach will typically generate arrays whose total element count is somewhat smaller
        # than the max size.
        max_shape_size = _MAX_SHAPE_SIZE if rank <= 1 else int(np.float_power(_MAX_SHAPE_SIZE, 1 / rank))

        # If a max size multiplier was specified, apply it here.
        if max_size_multiplier is not None:
            # Validate the parameter value; this check is implemented to disallow NaNs.
            if max_size_multiplier > 0.:
                # Round upwards to avoid any case where we round down to zero.
                max_shape_size = int(np.ceil(max_shape_size * max_size_multiplier))
            else:
                raise ValueError("The 'max_size_multiplier' argument must be a strictly-positive value.")

    if rank == 0:
        x_dim = y_dim = 1
    elif rank == 1:
        col_vector = draw(booleans())
        min_shape_size = 2
        if col_vector:
            x_dim = 1
            y_dim = draw(integers(min_value=min_shape_size, max_value=max_shape_size))
        else:
            x_dim = draw(integers(min_value=min_shape_size, max_value=max_shape_size))
            y_dim = 1
            if draw(booleans()):
                return (x_dim,)
    elif rank == 2:
        min_shape_size = 3
        x_dim = draw(integers(min_value=min_shape_size, max_value=max_shape_size))
        y_dim = draw(integers(min_value=min_shape_size, max_value=max_shape_size))
    else:
        raise ValueError(f"Unsupported rank value. (rank = {rank})")

    return x_dim, y_dim


@composite
def one_darray_shape_strategy(draw, max_shape_size: Optional[int] = None):
    x_dim = draw(integers(min_value=1, max_value=max_shape_size or _MAX_SHAPE_SIZE))
    return (x_dim,)


@composite
def ints_floats_datetimes_and_timedeltas(draw):
    dtypes = (
        unsigned_integer_dtypes(endianness="="),
        integer_dtypes(endianness="="),
        floating_dtypes(endianness="="),
        datetime64_dtypes(endianness="="),
        timedelta64_dtypes(endianness="="),
    )
    return draw(one_of(dtypes))


@composite
def ints_or_floats_example(draw):
    dtypes = (
        integers(),
        floats(),
    )
    return draw(one_of(dtypes))


@composite
def ints_floats_complex_or_booleans(draw):
    dtypes = (
        unsigned_integer_dtypes(endianness="="),
        integer_dtypes(endianness="="),
        floating_dtypes(endianness="="),
        # complex_number_dtypes(endianness="="),
        boolean_dtypes(),
    )
    return draw(one_of(dtypes))


@composite
def ints_floats_and_complex_numbers(draw):
    dtypes = (
        integers(min_value=-40000000, max_value=40000000),
        floats(min_value=-40000000, max_value=40000000),
        # complex_numbers(min_magnitude=0, max_magnitude=40000000),
    )
    return draw(one_of(dtypes))


@composite
def ints_floats_or_complex_dtypes(draw):
    # Endianness needs to be specified for now, otherwise the byte-order may get flipped
    # https://jira/browse/SOQTEST-6478
    dtypes = (
        unsigned_integer_dtypes(endianness="="),
        integer_dtypes(endianness="="),
        floating_dtypes(endianness="="),
        # complex_number_dtypes(endianness="="),
    )
    return draw(one_of(dtypes))


@composite
def ints_or_floats_dtypes(draw):
    # Endianness needs to be specified for now, otherwise the byte-order may get flipped
    # https://jira/browse/SOQTEST-6478
    # Half floats are not supported.
    dtypes = (
        unsigned_integer_dtypes(endianness="="),
        integer_dtypes(endianness="="),
        floating_dtypes(endianness="=", sizes=(32, 64)),
    )
    return draw(one_of(dtypes))


@composite
def same_structure_ndarrays(draw):
    shape = draw(ndarray_shape_strategy())
    # The datatypes for both arrays should be the same. Since ints_floats_datetimes_and_timedeltas() returns a strategy
    # rather than a draw(strategy()) a different value could be returned from the one_of. Shared prevents this and
    # ensures the same datatype is used for both arrays on each run.
    dtype = shared(ints_floats_datetimes_and_timedeltas())
    array1 = draw(arrays(dtype=dtype, shape=shape))
    array2 = draw(arrays(dtype=dtype, shape=shape))
    assert array1.dtype == array2.dtype
    assert array1.shape == array2.shape
    return array1, array2


@composite
def generate_array_and_where(draw, shape, dtype):
    arr = draw(arrays(shape=shape, dtype=dtype))
    where = draw(arrays(shape=arr.shape, dtype=boolean_dtypes()))
    return arr, where


@composite
def generate_ndarray_of_int_float_datetime_or_timedelta(draw):
    shape = draw(ndarray_shape_strategy())
    dtype = draw(ints_floats_datetimes_and_timedeltas())
    return draw(arrays(dtype=dtype, shape=shape))


@composite
def select_axis(draw, arr: np.ndarray, default_axis: Optional[int] = None):
    """Strategy for generating a random axis value based on the shape of an array."""
    max_axis = len(arr.shape) - 1
    min_axis = 0
    return draw(
        one_of((just(default_axis), integers(min_value=min_axis, max_value=max_axis))))


@composite
def generate_array(draw, shape, dtype, include_invalid: bool = True):
    if isinstance(dtype, SearchStrategy):
        dtype = draw(shared(dtype))
    arr = draw(
        arrays(
            dtype=dtype,
            shape=shape,
            elements=rt_element_strategy(dtype, include_invalid=include_invalid),
        )
    )
    return arr


@composite
def generate_array_and_axis(draw, shape, dtype, include_invalid: bool = True, default_axis: Optional[int] = None):
    arr = draw(generate_array(shape, dtype, include_invalid))
    axis = draw(select_axis(arr, default_axis=default_axis))
    return arr, axis


@composite
def floating_scalar(draw):
    scalar = draw(floats())
    scalars = (
        scalar,
        np.array(scalar),
        np.float64(scalar),
        np.float32(scalar),
    )
    return draw(sampled_from(scalars))


@composite
def start_stop_step_strategy(draw):
    dtype = ints_floats_and_complex_numbers()
    l = np.sort(draw(lists(dtype, min_size=3, max_size=3)))
    params = {
        'start': l[0],
        'stop': l[2],
        'step': l[1],
    }
    return params


@composite
def generate_tuples_of_arrays(
    draw,
    dtype_strategy=ints_floats_datetimes_and_timedeltas(),
    max_tuple_length=30,
    max_array_size=200,
    all_same_width=False,
    all_same_height=False,
):
    shape_type = draw(
        integers(min_value=0, max_value=4)
    )  # 0: (1,1), 1: (x,), 2: (x,1), 3: (1,x), 4: (x,y)
    if shape_type == 1 and all_same_height:
        shape_type = 3
    dtype = shared(dtype_strategy)
    # NOTE: shape=(height, width) or shape=(width,)
    starting_size = (
        draw(integers(min_value=2, max_value=max_array_size)),
        draw(integers(min_value=2, max_value=max_array_size)),
    )
    arr_list = []
    arr_length = draw(shared(integers(min_value=1, max_value=max_tuple_length)))
    if shape_type == 0:  # (1,1)
        for i in range(arr_length):
            arr_list.append(draw(arrays(dtype=dtype, shape=(1, 1))))
    elif shape_type == 1:  # (x,)
        if all_same_width:
            for i in range(arr_length):
                arr_list.append(draw(arrays(dtype=dtype, shape=(starting_size[1],))))
        else:
            for i in range(arr_length):
                x = draw(integers(min_value=1, max_value=max_array_size))
                arr_list.append(draw(arrays(dtype=dtype, shape=(x,))))
    elif shape_type == 2:  # (x,1)
        if all_same_height:
            for i in range(arr_length):
                arr_list.append(draw(arrays(dtype=dtype, shape=(starting_size[0], 1))))
        else:
            for i in range(arr_length):
                x = draw(integers(min_value=1, max_value=max_array_size))
                arr_list.append(draw(arrays(dtype=dtype, shape=(x, 1))))
    elif shape_type == 3:  # (1,x)
        if all_same_width:
            for i in range(arr_length):
                arr_list.append(draw(arrays(dtype=dtype, shape=(1, starting_size[1]))))
        else:
            for i in range(arr_length):
                x = draw(integers(min_value=1, max_value=max_array_size))
                arr_list.append(draw(arrays(dtype=dtype, shape=(1, x))))
    elif shape_type == 4:  # (x,y)
        if all_same_width and all_same_height:
            for i in range(arr_length):
                arr_list.append(
                    draw(
                        arrays(dtype=dtype, shape=(starting_size[0], starting_size[1]))
                    )
                )
        elif all_same_height:
            for i in range(arr_length):
                x = draw(integers(min_value=1, max_value=max_array_size))
                arr_list.append(draw(arrays(dtype=dtype, shape=(starting_size[0], x))))
        elif all_same_width:
            for i in range(arr_length):
                x = draw(integers(min_value=1, max_value=max_array_size))
                arr_list.append(draw(arrays(dtype=dtype, shape=(x, starting_size[1]))))
        else:
            for i in range(arr_length):
                x = draw(integers(min_value=1, max_value=max_array_size))
                y = draw(integers(min_value=1, max_value=max_array_size))
                arr_list.append(draw(arrays(dtype=dtype, shape=(x, y))))

    return tuple(arr_list)


@composite
def generate_array_axis_and_ddof(draw):
    arr, axis = draw(
        generate_array_and_axis(
            shape=draw(ndarray_shape_strategy()), dtype=draw(ints_or_floats_dtypes())
        )
    )
    N = np.count_nonzero(~np.isnan(arr), axis=axis)
    ddof = draw(integers(min_value=0, max_value=np.min(N)))
    return {
        'arr': arr,
        'axis': axis,
        'ddof': ddof,
    }


@composite
def interpolation_data(draw):
    length = draw(integers(min_value=2, max_value=10000))
    min_xp = draw(integers())
    xp = np.linspace(min_xp, min_xp + length, length)
    x = np.random.uniform(min_xp - 10, min_xp + length + 10, 1000)
    if draw(booleans()):
        m = draw(floats(min_value=-_MAX_VALUE, max_value=_MAX_VALUE))
        b = draw(floats(min_value=-_MAX_VALUE, max_value=_MAX_VALUE))
        fp = m * xp + b
    else:
        m = draw(floats(min_value=-_MAX_VALUE, max_value=_MAX_VALUE))
        vx = draw(floats(min_value=-_MAX_VALUE, max_value=_MAX_VALUE))
        vy = draw(floats(min_value=-_MAX_VALUE, max_value=_MAX_VALUE))
        fp = m * (xp - vx) ** 2 + vy

    return {
        'x': x,
        'xp': xp,
        'fp': fp,
    }


@composite
def generate_array_axis_and_repeats_array(draw, dtype, shape):
    arr = draw(arrays(shape=shape, dtype=dtype))
    axes = list(range(-1, len(arr.shape)))
    axis = draw(sampled_from(tuple(axes)))

    if axis == -1:
        axis = None
        repeats_length = arr.size
    elif axis == 0:
        repeats_length = arr.shape[0]
    elif axis == 1:
        repeats_length = arr.shape[1]
    else:
        repeats_length = 0

    repeats = draw(
        lists(
            integers(min_value=0, max_value=10),
            min_size=repeats_length,
            max_size=repeats_length,
        )
    )
    return arr, axis, repeats


@composite
def generate_reshape_array_and_shape_strategy(draw, shape, dtype):
    arr = draw(arrays(shape=shape, dtype=dtype))

    if draw(booleans()):
        return arr, -1  # make it a 1D array

    new_dim = draw(integers(min_value=1, max_value=arr.size))
    assume(arr.size % new_dim == 0)
    if draw(booleans()):
        return arr, (new_dim, -1)
    return arr, (-1, new_dim)


@composite
def generate_sample_test_integers(draw, num_bits, signed):
    if signed:
        min_value = -(2 ** (num_bits - 1))
        max_value = 2 ** (num_bits - 1) - 1
    else:
        min_value = 0
        max_value = (2 ** num_bits) - 1

    l = (
        0,
        -1,  # FF..FF
        (2 ** num_bits) - 1,  # FF..FF, -1 for signed, max for unsigned
        2
        ** (
            num_bits - 1
        ),  # 80..00, smallest negative for signed, convert to 64 bits first to prevent overflow exceptions
        ~(np.int64(1) << (num_bits - 1)),  # 7F..FF, max for signed
        np.int64(1) << (num_bits - 1),  # 80..00, done differently
        draw(integers(min_value=min_value, max_value=max_value)),
    )
    return l


@composite
def generate_sample_test_floats(draw):
    l = (
        0,
        -0.0,
        1,
        -1,
        np.inf,
        -np.inf,
        np.nan,
        draw(floats(min_value=0, max_value=1)),  # [0, 1]
        draw(floats(min_value=1)),  # [1, inf]
        draw(floats(min_value=-1, max_value=0)),  # [-1, 0]
        draw(floats(max_value=1)),  # [-inf, 1]
        draw(integers(min_value=0)),  # Z & (-inf, 0]
        draw(integers(max_value=0)),  # Z & [0, inf)
    )

    return l


@composite
def one_of_supported_dtypes(draw):
    # A strategy that selects a dtype that riptable is known to handle.
    # dtype size 16-bit is not supported
    # little endian is not supported
    return one_of(
        boolean_dtypes(),
        integer_dtypes(endianness="=", sizes=(8, 32, 64)),
        unsigned_integer_dtypes(endianness="=", sizes=(8, 32, 64)),
        floating_dtypes(endianness="=", sizes=(32, 64)),
        byte_string_dtypes(endianness="="),
        unicode_string_dtypes(endianness="="),
        # the following dtypes are not supported
        # complex_number_dtypes(),
        # datetime64_dtypes(),
        # timedelta64_dtypes(),
    )


@composite
def generate_lists(draw) -> Tuple[List[str], List[str]]:
    # A strategy that generates a tuple of two lists of the same length for text
    # of non-ASCII characters, nothing on the high end, that exclude control characters
    # and surrogates.
    size = draw(shared(integers(min_value=1, max_value=100)))
    lst1 = draw(
        lists(
            text(
                characters(
                    min_codepoint=1,
                    max_codepoint=1000,
                    blacklist_categories=("Cc", "Cs"),
                )
            ),
            min_size=size,
            max_size=size,
            unique=True,
        )
    )
    lst2 = draw(
        lists(
            text(
                characters(
                    min_codepoint=1,
                    max_codepoint=1000,
                    blacklist_categories=("Cc", "Cs"),
                )
            ),
            min_size=size,
            max_size=size,
            unique=True,
        )
    )
    return (lst1, lst2)


@composite
def generate_ndarrays(draw):
    shape = draw(shared(one_darray_shape_strategy(max_shape_size=10)))
    dtype = draw(one_of_supported_dtypes())
    arr1 = draw(arrays(shape=shape, dtype=dtype, unique=True))
    arr2 = draw(arrays(shape=shape, dtype=dtype, unique=True))
    return (arr1, arr2)


@composite
def generate_list_ndarrays(draw):
    # A strategy that generates a tuple of two lists of the same length that
    # contain ndarrays of possibly different dtypes from the supported dtypes and
    # are the same shape.
    size = draw(shared(integers(min_value=1, max_value=100)))
    shape = draw(shared(one_darray_shape_strategy(max_shape_size=10)))
    dtype = draw(one_of_supported_dtypes())
    lst1 = draw(
        lists(
            arrays(shape=shape, dtype=dtype, unique=True), min_size=size, max_size=size
        )
    )
    lst2 = draw(
        lists(
            arrays(shape=shape, dtype=dtype, unique=True), min_size=size, max_size=size
        )
    )
    return (lst1, lst2)


def rt_element_strategy(
    dtype: Union[np.dtype, type], include_invalid: bool = True
) -> Optional[SearchStrategy[Union[int, float, complex]]]:
    """
    Given a numpy dtype and whether to include riptable invalid values, return a strategy that
    can be used in generating the set of elements when constructing a ndarray or FastArray.

    Parameters
    ----------
    dtype: numpy.dtype or type
        The numpy type or numpy dtype of elements to generate for the search strategy.
    include_invalid: bool, optional
        If True, include riptable specific invalid values, otherwise do not include them (default True).

    Returns
    -------
    SearchStrategy
        A strategy that can be used to limit the scope of elements to generate for an ndarray or FastArray.
        If the dtype is not supported, returns None and warns the dtype is unsupported.

    Examples
    --------
    Generate an example ndarray of dtype int8 with invalid values:

    >>> arrays(np.int8, (2, 3), elements=rt_element_strategy(np.dtype(np.int8)), include_invalid=True).example()
    array([[-8,  6,  3],
           [-6,  4,  -128]], dtype=int8)

    Note, the invalid value, -128, is permissible in this example.

    Generate an example ndarray of dtype int8 without invalid values:

    >>> arrays(np.int8, (2, 3), elements=rt_element_strategy(np.dtype(np.int8)), include_invalid=False).example()
    array([[-8,  6,  3],
           [-6,  4,  -127]], dtype=int8)

    Note that the invalid value, -128, will not show in this array.

    Show all possible integer, float, and complex SearchStrategies:

    >>> for dtype in _dtypes_by_group["AllInteger"] + _dtypes_by_group["Float"] + _dtypes_by_group["Complex"]:
    >>>     print(f"{dtype} with invalid: {repr(rt_element_strategy(dtype, include_invalid=True))}")
    >>>     print(f"{dtype} without invalid: {repr(rt_element_strategy(dtype, include_invalid=False))}")
    int8 with invalid: integers(min_value=-128, max_value=127)
    int8 without invalid: integers(min_value=-127, max_value=127)
    ...

    """
    # Attempt to create a dtype from type.
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)

    elements_strategy: SearchStrategy = none()
    if dtype in _dtypes_by_group["AllInteger"]:
        min_value, max_value = integer_range(dtype, include_invalid=include_invalid)
        elements_strategy = integers(min_value=min_value, max_value=max_value)
    elif dtype in _dtypes_by_group["Float"]:
        width = dtype.itemsize * 8
        dtype_info = np.finfo(dtype)
        if include_invalid:
            elements_strategy = floats(
                width=width, allow_nan=True, allow_infinity=True,
            )
        else:
            elements_strategy = floats(
                width=width, allow_nan=False, allow_infinity=True,
            )
    elif dtype in _dtypes_by_group["Complex"]:
        # invalid value is not specified
        elements_strategy = complex_numbers(allow_nan=True, allow_infinity=True)
    else:
        # TODO handle invalids for np.sctypes["others"] [bool, object, bytes, str, numpy.void]
        warnings.warn(f"unhandled dtype '{dtype}'")
        return None
    return elements_strategy


# column strategies and helper utilities for Datasets
# get array columns (ndarray + fastarray)
# get struct columns (one of each)
# get dataset columns (copy of itself for two layer embedding)
# get categorical column (using our categorical strategy)
def column_by_dtypes(dtype_group: Optional[str] = "RiptableNumeric") -> List[pdst.column]:
    """Returns a list of columns from a dtype group for generation of primitive data types wrapped in columns for DataFrame strategies."""
    return [
        pdst.column(str(dtype), dtype=np.dtype(dtype))
        for dtype in set(dtypes_by_group[dtype_group])
    ]


def column_arrays(draw) -> List[Union[np.ndarray, rt.FastArray]]:
    """Returns a list of numpy ndarray and riptide FastArray wrapped in columns for DataFrame strategies."""
    # todo add strategy to generate FastArray to the return list
    arr = draw(
        generate_array(
            shape=ndarray_shape_strategy(),
            dtype=ints_or_floats_dtypes(),
            include_invalid=False,
        )
    )
    # f_arr = rt.FastArray(arr)
    return [pdst.column(name(arr), elements=arr)]
