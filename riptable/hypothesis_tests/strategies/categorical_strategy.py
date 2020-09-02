from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import riptable as rt
from riptable import Categorical
from riptable.rt_enum import CategoryMode

from hypothesis import event
import hypothesis.strategies as st
from hypothesis.extra.numpy import (
    arrays,
    array_shapes,
    byte_string_dtypes,
    integer_array_indices,
    integer_dtypes,
    unicode_string_dtypes,
    unsigned_integer_dtypes
)

# TODO
#   * Implement a way to generate Categoricals of derived FastArray classes, such as rt.Date.
#     Could just implement strategies for each of them, then use .map(lambda x: rt.Cat(x)) when we need
#     a Categorical of that type.
#   * Implement a strategy which just creates a random Categorical of any type -- single/multikey, incl.
#     derived FastArray types. This could be our general go-to when we're implementing tests where we want
#     to ensure all Categorical cases are fully covered -- e.g. when a Categorical column in a Dataset is
#     round-tripped through SDS.
#   * Implement another strategy accepting one or more strategies and creates a single or multikey Categorical
#     with them. One difficulty with this _may_ be structuring this both so we allow for various strategies to be used
#     for the categorical data while also enforcing that each time we create an example, the array(s) produced
#     by the strategies must have the same length.
#   * Maybe implement (and contribute to hypothesis) a strategy like hypothesis.strategy.text but which generates
#     numpy arrays; this strategy should also have the unique= kwarg like 'arrays' does. If this strategy was available
#     it'd be useful as a building block for some of the Categorical strategies below -- it'd simplify some things
#     while (probably) also improving performance.


def category_labels(
    min_str_len: int,
    max_str_len: int,
    unicode: bool
) -> st.SearchStrategy:
    """Creates a `hypothesis.strategies.SearchStrategy` instance for generating category labels."""
    category_label_chars = \
        st.characters(
            whitelist_categories=('L', 'Nd', 'Nl', 'Pc', 'Pd', 'Zs'),
            max_codepoint=None if unicode else 127  # Max codepoint of 127 limits to ASCII codepoints.
        )
    return st.text(alphabet=category_label_chars, min_size=min_str_len, max_size=max_str_len)


@st.composite
def categorical_stringarray(
    draw,
    max_length: int,
    max_categories: int,
    *,
    endianness: str = '=',
    min_str_len: int = 1,
    max_str_len: int = 16,
    unicode: Optional[bool] = None,
    ordered: Optional[bool] = None,
) -> Categorical:
    """
    Strategy for creating StringArray-mode Categoricals.

    Parameters
    ----------
    draw
    max_length : int
    max_categories : int
    endianness : str
    min_str_len : int
    max_str_len : int
    unicode : bool, optional
    ordered : bool, optional

    Examples
    --------
    >>> array_strategy = arrays(integer_dtypes(endianness="=", sizes=(64,)), (5,))
    arrays(dtype=integer_dtypes(endianness='=', sizes=(64,)), shape=(5,))
    >>> categorical_stringarray(array_strategy, with_categories=True).example()
    0, 0, 0, 0, 0

    Notes
    -----
    TODO: Make sure to include the case where we have category values (in the underlying integer array)
          past the end of the categories array. (Or is that only for a Dictionary mode categorical?)
          To clarify -- this is the behavior where, when we print the Categorical, we get entries like <!456>.

    TODO: Also exercise (in one way or another) the following arguments to the Categorical constructor:
        * base_index
            Add an optional boolean parameter. When None, draw a boolean to fill it in.
            When the bool is false, call rt.Cat() with base_index=0.
            When True, call rt.Cat() with base_index=1.
        * dtype
            Call the ctor with dtype=None or a signed integer dtype that's either the min size given the
            number of categories or any larger signed integer dtype.
            E.g. if len(categories) == 1000, draw from { None, np.int16, np.int32, np.int64 }
        * filter
            Add an optional boolean param to the strategy which defaults to None, in which case we'll fill it by drawing a boolean.
            When the bool is false we we call rt.Cat() with filter=None.
            When True, we create a boolean array the same length as our values or fancy index and pass that as the filter.

    TODO: Support slicing/strides on the values/categories arrays passed to the Categorical constructor.

    TODO: When creating the fancy index array and we've drawn 'explicit_categories=True', allow the fancy index to be created
          with any applicable integer type (signed or unsigned) whose range is large enough to index into the categories array.
          (Or, should we just allow _any_ integer dtype, even if too small? We wouldn't be able to index categories past the
          range of the dtype, but maybe that's an interesting thing to test? Especially around cases like having auto_add=True.)
    """
    # Draw a boolean indicating how the data will be passed to the Categorical constructor later.
    # This is done first since it's one of the most likely things to affect the behavior of the Categorical,
    # and shrinking (in some cases) works better when such values are drawn earlier in strategy.
    explicit_categories: bool = draw(st.booleans())
    if explicit_categories:
        event('Categorical created from unique category array and fancy index.')
    else:
        event('Categorical created from non-unique array of strings.')

    # Draw the string dtype based on whether we want a byte (ascii) string or Unicode.
    is_unicode: bool = draw(st.booleans()) if unicode is None else unicode
    if is_unicode:
        labels_dtype = draw(unicode_string_dtypes(endianness=endianness, min_len=min_str_len, max_len=max_str_len))
    else:
        labels_dtype = draw(byte_string_dtypes(endianness=endianness, min_len=min_str_len, max_len=max_str_len))

    # Create an array of unique category labels.
    cats_shapes = array_shapes(max_dims=1, max_side=max_categories)
    category_label_strat = category_labels(min_str_len, max_str_len, unicode=is_unicode)
    unique_labels = draw(arrays(dtype=labels_dtype, shape=cats_shapes, elements=category_label_strat, unique=True))

    # Use basic_indices to create a fancy index into the array of unique category labels.
    # Apply it to expand the array of unique labels into an array where those labels may occur zero or more times.
    fancy_index_shapes = array_shapes(max_dims=1, max_side=max_length)
    fancy_index = draw(integer_array_indices(shape=unique_labels.shape, result_shape=fancy_index_shapes))

    # If the 'ordered' flag is not set, draw a boolean for it now so we have a concrete value
    # to use when creating the categorical.
    is_ordered = draw(st.booleans()) if ordered is None else ordered

    # If the 'explicit_categories' flag is set, create the Categorical by passing in the
    # unique values and fancy index separately.
    # Otherwise, apply the fancy index to the array of unique category values to produce an
    # array where each category appears zero or more times; then create the Categorical from that.
    if explicit_categories:
        return Categorical(fancy_index, categories=unique_labels, ordered=is_ordered, unicode=is_unicode)

    else:
        values = unique_labels[fancy_index]
        return Categorical(values, ordered=is_ordered, unicode=is_unicode)


def bimap(
    keys: st.SearchStrategy,
    values: st.SearchStrategy,
    *,
    min_size: int = 0,
    max_size: Optional[int] = None
) -> st.SearchStrategy:
    """
    Strategy similar to the hypothesis 'dictionaries' strategy, but which ensures both keys and values are unique.
    """
    return st.lists(
        st.tuples(keys, values),
        min_size=min_size,
        max_size=max_size,
        unique=True
    ).map(dict)


@st.composite
def categorical_dictmode(
    draw,
    max_length: int,
    max_categories: int,
    *,
    endianness: str = '=',
    min_str_len: int = 1,
    max_str_len: int = 16,
    unicode: Optional[bool] = None,
    ordered: Optional[bool] = None,
) -> Categorical:
    """
    Strategy for creating Dictionary-mode Categoricals.

    This strategy currently only covers creating `Categorical` instances with
    string-typed category labels.

    Parameters
    ----------
    draw
    max_length : int
    max_categories : int
    endianness : str
    min_str_len : int
    max_str_len : int
    unicode : bool, optional
    ordered : bool, optional

    Examples
    --------
    >>> categorical_dictmode(10_000, 1_000, max_str_len=20).example()
    0, 0, 0, 0, 0

    Notes
    -----
    TODO: Make sure to include the case where we have category values (in the underlying integer array)
          past the end of the categories array. (Or is that only for a Dictionary mode categorical?)
          To clarify -- this is the behavior where, when we print the Categorical, we get entries like <!456>.

    TODO: Also exercise (in one way or another) the following arguments to the Categorical constructor:
        * base_index
            Add an optional boolean parameter. When None, draw a boolean to fill it in.
            When the bool is false, call rt.Cat() with base_index=0.
            When True, call rt.Cat() with base_index=1.
        * dtype
            Call the ctor with dtype=None or a signed integer dtype that's either the min size given the
            number of categories or any larger signed integer dtype.
            E.g. if len(categories) == 1000, draw from { None, np.int16, np.int32, np.int64 }
        * filter
            Add an optional boolean param to the strategy which defaults to None, in which case we'll fill it by drawing a boolean.
            When the bool is false we we call rt.Cat() with filter=None.
            When True, we create a boolean array the same length as our values or fancy index and pass that as the filter.

    TODO: Support slicing/strides on the values/categories arrays passed to the Categorical constructor.

    TODO: Does a Dictionary-mode Categorical allow any other types (e.g. rt.Date) to be used for the category labels?
        If so, these should also be covered by this strategy (though changes will needed to allow a variety of
        types to be used for category labels).

    TODO: Any possible issues (that we might want to exercise in this strategy) between the string used when displaying
        the invalid category (e.g. 'Inv') and category labels? What happens if we have a category label using the same string?
    """
    # Draw a boolean indicating whether we'll use a signed or unsigned integer dtype.
    use_signed_integer_dtype: bool = draw(st.booleans())

    # If using a signed integer dtype, draw another boolean indicating whether we'll
    # generate negative category values.
    allow_negative_category_values: bool = draw(st.booleans()) if use_signed_integer_dtype else False
    if use_signed_integer_dtype:
        if allow_negative_category_values:
            event('Categorical may have a mix of negative, zero, and positive category values.')
        else:
            event('Categorical has only non-negative category values.')

    # If the 'unicode' flag is not set, draw a boolean to fill it in.
    is_unicode: bool = draw(st.booleans()) if unicode is None else unicode
    event(f'Category labels are {"unicode" if is_unicode else "ascii"} strings.')

    # If the 'ordered' flag is not set, draw a boolean for it now so we have a concrete value
    # to use when creating the categorical.
    is_ordered = draw(st.booleans()) if ordered is None else ordered
    event(f'ordered = {is_ordered}')

    # Draw the dtype for the category values.
    # TODO: Draw a signed or unsigned integer dtype here which is at least as large as needed, but perhaps larger
    #       than needed.
    #       For now, we just use the smallest dtype large enough to fit the max number of categories; but allowing for
    #       larger (randomly-selected) dtypes later will help ensure we test cases where there are non-consecutive
    #       category values even when the max_categories value is near the max value of a dtype.
    values_dtype = np.min_scalar_type(max_categories)

    # Create the strategy for the category values (integer values representing the categories).
    values_dtype_info = np.iinfo(values_dtype)
    values_strat =\
        st.integers(
            min_value=(values_dtype_info.min if allow_negative_category_values else 0),
            max_value=values_dtype_info.max)

    # Create an array of unique category values/codes.
    cats_shapes = array_shapes(max_dims=1, max_side=max_categories)
    unique_cat_values = draw(arrays(dtype=values_dtype, shape=cats_shapes, elements=values_strat, unique=True))

    # Draw the string dtype for the labels based on whether we want a byte (ascii) string or Unicode.
    is_unicode: bool = draw(st.booleans()) if unicode is None else unicode
    if is_unicode:
        labels_dtype = draw(unicode_string_dtypes(endianness=endianness, min_len=min_str_len, max_len=max_str_len))
    else:
        labels_dtype = draw(byte_string_dtypes(endianness=endianness, min_len=min_str_len, max_len=max_str_len))

    # Create an array of unique category labels; this must be the same shape as the unique category values array.
    category_label_strat = category_labels(min_str_len, max_str_len, unicode=is_unicode)
    unique_labels =\
        draw(arrays(dtype=labels_dtype, shape=unique_cat_values.shape, elements=category_label_strat, unique=True))

    # TODO: Draw a slice (or None) that we'll apply to both arrays of uniques (the labels and values)
    #   before using them to create the category dictionary.
    #   This allows us to cover cases where a category value isn't in the dictionary.

    # Combine the unique category labels and values to create a dictionary.
    category_dict = dict(zip(unique_labels, unique_cat_values))

    # Use basic_indices to create a fancy index into the array of unique values.
    # Apply it to expand the array of unique values into an array where those values may occur zero or more times.
    fancy_index_shapes = array_shapes(max_dims=1, max_side=max_length)
    fancy_index = draw(integer_array_indices(shape=unique_cat_values.shape, result_shape=fancy_index_shapes))

    # Apply the fancy index to the array of unique category values to produce an
    # array where each category appears zero or more times; then create the Categorical from that.
    cat_values = unique_cat_values[fancy_index]
    return Categorical(cat_values, categories=category_dict, ordered=is_ordered, unicode=is_unicode)


@st.composite
def categorical_intenum(
    draw
) -> Categorical:
    """Strategy for creating IntEnum-mode Categoricals."""
    raise NotImplementedError()


@st.composite
def categorical_numericarray(
    draw
) -> Categorical:
    """Strategy for creating NumericArray-mode Categoricals."""
    raise NotImplementedError()


class CategoricalStrategy(st.SearchStrategy):
    """
    A strategy for generating `Categoricals` across the `CategoryMode` \s using a
    hypothesis `SearchStrategy` for generating the values and categories.

    Parameters
    ----------
    value_strategy : SearchStrategy
        A strategy that generates :class:`numpy:numpy.ndarray` s or list-like values that
        the `Categorical` constructor accepts.
    with_categories : bool
        Whether to generate and pass categories when constructing the ``Categorical``.
    category_mode : ``CategoryMode``
        The type of `Categorical` to construct.
    ordered : bool
        Corresponds to ``Categorical`` ordered parameter when initializing.

    Examples
    --------
    >>> array_strategy = arrays(integer_dtypes(endianness="=", sizes=(64,)), (5,))
    arrays(dtype=integer_dtypes(endianness='=', sizes=(64,)), shape=(5,))
    >>> CategoricalStrategy(array_strategy, with_categories=True, category_mode=CategoryMode.StringArray).example()
    0, 0, 0, 0, 0
    """

    _CN = "CategoricalStrategy"

    def __init__(
        self,
        values_strategy: st.SearchStrategy,
        with_categories: bool = False,
        category_mode: Optional[CategoryMode] = None,
        ordered: Optional[bool] = None,
    ):
        super().__init__()
        self.value_strategy = values_strategy
        self.with_categories = with_categories
        self.category_mode = category_mode
        self.ordered = ordered

    def _construct_dict(self, data, values) -> Union[Dict[str, int], Dict[int, str]]:
        """Return a dictionary that will be used for Dictionary Categorymodes."""
        # add support for Dict[str, int] mode
        unique_values = np.unique(values)
        n = len(unique_values)
        keys = data.draw(
            st.lists(
                st.integers(min_value=1, max_value=n * 10),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
        return {k: str(v) for k, v in zip(keys, unique_values)}

    def do_draw(self, data):
        # categories will be set if either:
        # - with_categories parameter is set to True, or
        # - CategoryMode or category_mode designates a dictionary Categorical.
        values, categories, cat = None, None, None
        if self.category_mode == CategoryMode.StringArray:
            values = list(map(str, data.draw(self.value_strategy)))
            if self.with_categories:
                categories = list(map(str, set(values)))
            cat = Categorical(values, categories=categories, ordered=self.ordered)
        elif self.category_mode == CategoryMode.Dictionary:
            values = data.draw(self.value_strategy)
            category_dict = self._construct_dict(data, values)
            cat = Categorical(values, categories=category_dict, ordered=self.ordered)
        else:
            raise ValueError(
                f"{self._CN}.do_draw: unhandled category mode {self.category_mode}\n\t{self}"
            )
        return cat

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{self._CN}({self.value_strategy}, with_categories={self.with_categories}, category_mode={self.category_mode} ordered={self.ordered})"


if __name__ == "__main__":
    from hypothesis.extra.numpy import arrays, integer_dtypes

    def print_cat(c):
        print(
            f"c\n{repr(c)}\n\ntype(c), mode\n{type(c)}, {rt.rt_enum.CategoryMode(c.category_mode)}\n"
            f"\nc._fa\n{repr(c._fa)}\n\nc.category_array\n{repr(c.category_array)}\n\nc.category_dict\n{c.category_dict}\n\n"
        )
        if c.category_mode == rt.rt_enum.CategoryMode.Dictionary:
            print(
                f"*c.category_mapping*\n{c.category_mapping}\n\n*c.category_codes*\n{c.category_codes}"
            )

    cat_strat = categorical_dictmode(10_000, 1_000, max_str_len=20)
    print(f"cat_strat {cat_strat}\n")
    print_cat(cat_strat.example())
