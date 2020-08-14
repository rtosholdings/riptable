import unittest
import string
import riptable as rt

from typing import Optional, List, Any, Union
from riptable.rt_enum import CategoryMode

#
# TODO: Consider moving these functions into a 'testing' submodule of riptable, a la numpy.testing.
#


_N = 10

# The following helper methods are used to generate the underlying data a Categorical of the various
# category modes will use when constructed.


def get_categorical_base_data() -> List[List[str]]:
    # This acts a a base for the different edge cases considered for the type of data that is used when constructing
    # a Categorical. The other Categorical data functions, such as the one for bytes and numerics, derive from this table.
    # If new edge cases are considered, this table can be extended which will extend tables for the other data types.
    return [
        # [],  # empty list; commented out since this case throws a ValueError in Categorical constructor.
        ["a"],  # one element list
        ["a"] * _N,  # list of single repeated element
        list(string.ascii_letters),  # list of all unique elements
        ["a"] * _N + ["z"] + ["a"] * _N,  # repeats surrounding single unique
        ["a"] + ["b"] * _N + ["c"],  # single unique surrounded by repeats
    ]


def _get_categorical_byte_data() -> List[List[bytes]]:
    bytes = []
    for values in get_categorical_base_data():
        b = [s.encode("utf-8") for s in values]
        bytes.extend(b)
    return bytes


def _get_categorical_numeric_data() -> List[List[int]]:
    numerics = []
    for values in get_categorical_base_data():
        numerics.append([ord(c) for c in values])
    return numerics


def _get_categorical_multikey_data() -> List[List[rt.FastArray]]:
    # Create data that can be used to construct MultiKey Categoricals where both keys are
    # strings, numerics, and combination of the two.
    strings = get_categorical_base_data()
    numerics = _get_categorical_numeric_data()

    results = []
    # consider parameterizing over the number of keys instead of literal handling of up to four keys
    for values in strings + numerics:  # two keys of same dtype and value
        results.append([rt.FA(values), rt.FA(values)])
    for values, values1 in zip(strings, numerics):  # two keys of different dtypes
        results.append([rt.FA(values), rt.FA(values1)])
    for values, values1, values2 in zip(
        strings, strings, numerics
    ):  # three keys of different dtypes
        results.append([rt.FA(values), rt.FA(values1), rt.FA(values2)])
    for values, values1, values2, values3 in zip(
        strings, strings, strings, numerics
    ):  # four keys of different dtypes
        results.append([rt.FA(values), rt.FA(values1), rt.FA(values2), rt.FA(values3)])
    return results


def verbose_categorical(categorical: rt.Categorical) -> str:
    lst = []
    lst.append(f"Categorical:\n{repr(categorical)}")
    lst.append(f"Type:\n{type(categorical)}")
    lst.append(
        f"Category mode:\n{repr(rt.rt_enum.CategoryMode(categorical.category_mode))}"
    )
    lst.append(f"Lock:\n{categorical._locked}")
    # add support for multikey categoricals
    if categorical.category_mode != rt.rt_enum.CategoryMode.MultiKey:
        lst.append(f"_fa\n{repr(categorical._fa)}")
        lst.append(f"category_array\n{repr(categorical.category_array)}")
        lst.append(f"category_dict\n{categorical.category_dict}")
        if categorical.category_mode == rt.rt_enum.CategoryMode.Dictionary:
            lst.append(f"category_mapping\n{categorical.category_mapping}")
            lst.append(f"category_codes\n{categorical.category_codes}")
    return "\n\n".join(lst)


def get_categorical_data_factory_method(
    category_modes: Optional[Union[CategoryMode, List[CategoryMode]]] = None
) -> List[List[Any]]:
    """Get test data used to construct Categoricals.

    Parameters
    ----------
    category_modes : CategoryMode, list of CategoryMode, optional
        Category modes that are used for getting the appropriate data to construct a Categorical. If no category_mode is specified, data for all supported CategoryModes will be returned, otherwise select the type of data by specifying the CategoryMode or a list of CategoryModes.

    Returns
    -------
    data : List[List[Any]]
        - The list of values that can be used to construct a Categorical.

    Examples
    --------
    >>> data = get_categorical_data_factory_method([CategoryMode.StringArray, CategoryMode.NumericArray])
    >>> cats = [Categorical(values) for values in data]
    """
    if category_modes is None:  # return all types of categories
        return (
            get_categorical_base_data()
            + _get_categorical_numeric_data()
            + _get_categorical_multikey_data()
        )
    if not isinstance(category_modes, list):  # wrap single category mode in a list
        category_modes = [category_modes]

    underlying_data = []
    if rt.rt_enum.CategoryMode.StringArray in category_modes:
        underlying_data.extend(get_categorical_base_data())
    if rt.rt_enum.CategoryMode.NumericArray in category_modes:
        underlying_data.extend(_get_categorical_numeric_data())
    if rt.rt_enum.CategoryMode.MultiKey in category_modes:
        underlying_data.extend(_get_categorical_multikey_data())
    return underlying_data


def get_all_categorical_data() -> List[rt.Categorical]:
    """Returns a list of all the Categorical test data of all supported CategoryModes."""
    return [rt.Categorical(data) for data in get_categorical_data_factory_method()]


if __name__ == "__main__":
    tester = unittest.main()
