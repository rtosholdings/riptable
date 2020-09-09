from collections import Counter
from typing import List

import hypothesis
import pytest
import numpy as np

from hypothesis import given, HealthCheck
from hypothesis.strategies import (
    composite,
    shared,
    integers,
    lists,
    booleans,
    one_of,
)
from hypothesis.extra.numpy import (
    arrays,
    integer_dtypes,
    unsigned_integer_dtypes,
    datetime64_dtypes,
    timedelta64_dtypes,
    byte_string_dtypes,
    unicode_string_dtypes,
)
from hypothesis.strategies import data

from .strategies.categorical_strategy import CategoricalStrategy
from .strategies.helper_strategies import one_darray_shape_strategy
from riptable import Categorical, hstack, FastArray
from riptable.rt_enum import CategoryMode
from riptable.Utils.teamcity_helper import is_running_in_teamcity


_MAX_SIZE = 1_000


def _get_category_to_count(categoricals) -> Counter:
    if not isinstance(categoricals, list):
        categorical = categoricals
        # create a new list since list constructor with categoricals will return the underlying representation as a list
        categoricals = list()
        categoricals.append(categorical)

    category_to_count = Counter()
    for categorical in categoricals:
        categories = categorical.categories()
        multiplicities = categorical.grouping.ncountgroup
        for category, multiplicity in zip(categories, multiplicities):
            category_to_count[category] += multiplicity

    return category_to_count


def _check_categorical(categorical: Categorical) -> (bool, str):
    valid: bool = True
    errors: List[str] = list()
    if not isinstance(categorical, Categorical):
        valid = False
        errors.append(
            f"Categorical {categorical} should be of type {type(Categorical)}"
        )
    return valid, "\n".join(errors)


@composite
def one_of_categorical_values(draw):
    cat_values = integers(
        min_value=1, max_value=np.iinfo(np.int64).max
    )  # add - bytes(), characters(),
    return draw(one_of(cat_values))


@pytest.mark.skipif(reason="Categorical generator needs to be rewritten for better performance before re-enabling this test to run in TeamCity builds.")
# 2020-09-09T21:02:22.3088485Z ================================== FAILURES ===================================
# 2020-09-09T21:02:22.3088746Z ________ test_categorical_ctor[CategoryMode.StringArray-integer_dtype] ________
# 2020-09-09T21:02:22.3088996Z
# 2020-09-09T21:02:22.3089415Z value_strategy = arrays(dtype=integer_dtypes(endianness='=', sizes=(64,)), shape=one_darray_shape_strategy(), elements=integers(min_value=1, max_value=9223372036854775807))
# 2020-09-09T21:02:22.3089925Z category_mode = <CategoryMode.StringArray: 1>
# 2020-09-09T21:02:22.3090026Z
# 2020-09-09T21:02:22.3090166Z >   ???
# 2020-09-09T21:02:22.3090416Z E   hypothesis.errors.FailedHealthCheck: Data generation is extremely slow: Only produced 9 valid examples in 1.13 seconds (0 invalid ones and 4 exceeded maximum size). Try decreasing size of the data you're generating (with e.g.max_size or max_leaves parameters).
# 2020-09-09T21:02:22.3091373Z E   See https://hypothesis.readthedocs.io/en/latest/healthchecks.html for more information about this. If you want to disable just this health check, add HealthCheck.too_slow to the suppress_health_check settings for this test.
@given(data())
@pytest.mark.parametrize(
    "value_strategy",
    [
        # Categorical values must be nonempty
        pytest.param(
            lists(one_of_categorical_values(), min_size=5, max_size=10),
            id="list",
        ),
        pytest.param(
            lists(
                one_of_categorical_values(), min_size=1, unique=True, max_size=10
            ),
            id="unique_list",
        ),
        pytest.param(
            arrays(
                shape=one_darray_shape_strategy(max_shape_size=10),
                dtype=integer_dtypes(endianness="=", sizes=(64,)),
                elements=integers(min_value=1, max_value=np.iinfo(np.int64).max),
            ),
            id="integer_dtype",
        ),
        pytest.param(
            arrays(
                shape=one_darray_shape_strategy(),
                dtype=integer_dtypes(endianness="=", sizes=(64,)),
                elements=integers(min_value=1, max_value=np.iinfo(np.int64).max),
                fill=integers(min_value=0, max_value=np.iinfo(np.int64).max),
                unique=True
            ),
            id="integer_dtype_unique",
            marks=[
                pytest.mark.skip,
                pytest.mark.xfail(reason='Now throws a hypothesis.errors.InvalidArgument: Cannot fill unique array with non-NaN value 1'),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "category_mode", [CategoryMode.StringArray, CategoryMode.Dictionary]
)
def test_categorical_ctor(value_strategy, category_mode, data):
    # cat is drawn from CategoricalStrategy
    ordered: bool = data.draw(booleans())
    cat: Categorical = data.draw(
        CategoricalStrategy(
            value_strategy, category_mode=category_mode, ordered=ordered
        )
    )
    assert _check_categorical(cat)

    # Validate properties on constructing a Categorical from a Categorical's values and categories.
    values, categories = cat.expand_array, cat._categories
    # For Dictionary Categoricals, 'categories' should be the original Categorical's category_map.
    if category_mode == CategoryMode.Dictionary:
        categories = cat.category_mapping
    cat2 = Categorical(values, categories=categories, ordered=ordered)
    assert _check_categorical(cat2)

    # Validate properties on constructing a Categorical given a Categorical.
    cat3 = Categorical(cat2)
    assert _check_categorical(cat3)

    # Validate properties on constructing a Categorical using _from_categorical which is a fast path
    # that skips internal routine checks, sorting, or making values unique, but should be identical to
    # the original Categorical.
    from_categorical = cat._categories_wrap
    cat4 = Categorical(
        values,
        categories=categories,
        _from_categorical=from_categorical,
        ordered=ordered,
    )
    assert _check_categorical(cat4)

    # TODO: add equality checks for the Categoricals above since they should all be equivalent.


# TODO remove hypothesis suppress_health_check after investigating FailedHealthCheck for test_categorical_property.test_hstack[CategoryMode_StringArray-unsigned_integer_dtype]
# E   hypothesis.errors.FailedHealthCheck: Data generation is extremely slow: Only produced 7 valid examples in 1.06 seconds (0 invalid ones and 5 exceeded maximum size). Try decreasing size of the data you're generating (with e.g.max_size or max_leaves parameters).
# As is, the unsigned_integer_dtype case uses min and max values for data generation.
@pytest.mark.skip(reason="Categorical generator needs to be rewritten for better performance before re-enabling this test to run in TeamCity builds.")
@hypothesis.settings(suppress_health_check=[HealthCheck.too_slow])
@given(data())
@pytest.mark.parametrize(
    "datatype, elements",
    [
        pytest.param(
            integer_dtypes(endianness="=", sizes=(64,)),
            integers(min_value=1, max_value=np.iinfo(np.int64).max),
            id="integer_dtype",
        ),
        pytest.param(
            unsigned_integer_dtypes(endianness="=", sizes=(64,)),
            integers(min_value=1, max_value=np.iinfo(np.int64).max),
            id="unsigned_integer_dtype",
        ),
        pytest.param(byte_string_dtypes(endianness="="), None, id="byte_string_dtype"),
        pytest.param(
            datetime64_dtypes(endianness="="),
            None,
            id="datetime64_dtype",
            marks=[
                pytest.mark.xfail(reason="RIP-375 - Categorical unsupported dtypes"),
                pytest.mark.skip,
            ],
        ),
        pytest.param(
            timedelta64_dtypes(endianness="="),
            None,
            id="timedelta64_dtype",
            marks=[
                pytest.mark.xfail(reason="RIP-375 - Categorical unsupported dtypes"),
                pytest.mark.skip,
            ],
        ),
        pytest.param(
            unicode_string_dtypes(endianness="="),
            None,
            id="unicode_string_dtype",
            marks=[
                pytest.mark.xfail(reason="RIP-375 - Categorical unsupported dtypes"),
                pytest.mark.skip,
            ],
        ),
    ],
)
@pytest.mark.parametrize("category_mode", [CategoryMode.StringArray])
def test_hstack(datatype, elements, category_mode, data):
    shape = one_darray_shape_strategy()
    dtype = shared(datatype)
    msg = f"Using dtype {dtype}\nUsing elements {elements}\n"

    # Increasing the maximum number of runs by a 10x magnitude will result in FailedHealthCheck errors with slow data generation.
    max = data.draw(integers(min_value=1, max_value=5))
    categoricals: List[Categorical] = list()
    for i in range(max):
        value_strategy = arrays(dtype, shape, elements=elements)
        with_categories: bool = data.draw(booleans())
        categoricals.append(
            data.draw(
                CategoricalStrategy(
                    value_strategy,
                    with_categories=with_categories,
                    category_mode=category_mode,
                )
            )
        )

    # Test #1: Length of hstacked categoricals should be the sum of the aggregate categoricals.
    output = hstack(tuple(categoricals))
    assert isinstance(output, Categorical)

    assert len(output) == sum(map(len, categoricals)), (
        f"Length of hstacked categoricals should be the sum of the aggregate categoricals\n"
        + msg
        + f"actual:\n{output}\nexpected:\n{categoricals}"
    )

    # Test #2: The hstacked categories should be equivalent to the set of aggregate categories.
    expected_counts = _get_category_to_count(categoricals)
    actual_counts = _get_category_to_count(output)

    assert not set(actual_counts.elements()).symmetric_difference(
        set(expected_counts.elements())
    ), (
        f"The hstacked categories should be equivalent to the set of aggregate categories\n"
        + msg
        + f"actual {set(actual_counts.elements())}\nexpected {set(expected_counts.elements())}"
    )

    # Test #3: The hstacked multiplicity of categories should be equivalent to the multiplicity of aggregate categories.
    # Test (2) is a subset of this equality check, but remains for clarity reasons when investigating failures.
    assert expected_counts == actual_counts, (
        f"The hstacked multiplicity of categories should be equivalent to the multiplicity of aggregate categories\n"
        + msg
        + f"actual {actual_counts}\nexpected {expected_counts}"
    )


@pytest.mark.xfail(reason="RIP-375 - Categorical unsupported dtypes")
@pytest.mark.skipif(
    is_running_in_teamcity(), reason="Please remove alongside xfail removal."
)
@pytest.mark.parametrize(
    "data",
    [
        # ValueError: BuildArrayInfo array has bad dtype of 21
        FastArray(["1970"], dtype="datetime64[Y]"),
        # ValueError: BuildArrayInfo array has bad dtype of 22
        FastArray([0], dtype="timedelta64[Y]"),
    ],
)
def test_falsifying_categorical_ctor(data):
    Categorical(data)


@pytest.mark.skipif(True, reason="RIP-452: Mutikey Categorical isin is consistent with its single key isin alternative")
def test_multikey_categorical_isin():
    # See Python/core/riptable/tests/test_categorical.py test_multikey_categorical_isin as an example
    pass
