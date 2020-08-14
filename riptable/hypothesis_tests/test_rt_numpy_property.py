from math import nan
from typing import List

import numpy as np
import riptable as rt
from riptable import FastArray, FA

import pytest
import unittest
from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal,
    assert_equal,
    assert_allclose,
    assert_almost_equal,
)

import hypothesis
from hypothesis import assume, event, example, given, HealthCheck
from hypothesis.extra.numpy import (
    arrays,
    boolean_dtypes,
    floating_dtypes,
    integer_dtypes,
    unsigned_integer_dtypes,
)
from hypothesis.strategies import one_of

# riptable custom Hypothesis strategies
from .strategies.helper_strategies import (
    generate_list_ndarrays,
    generate_lists,
    generate_ndarrays,
    ndarray_shape_strategy,
    one_darray_shape_strategy,
)

from riptable.Utils.teamcity_helper import is_running_in_teamcity


class TestCat2Keys:
    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Hypothesis generation taking too long."
    )
    @given(keys=generate_list_ndarrays())
    def test_cat2keys_nested_array(self, keys):
        key1, key2 = keys
        multi_cat = rt.cat2keys(key1, key2)
        print(f"key1 {repr(key1)}\nkey2: {repr(key2)}")

    @pytest.mark.skipif(
        is_running_in_teamcity(), reason="Hypothesis generation taking too long."
    )
    @given(keys=one_of(generate_lists(), generate_ndarrays()))
    def test_cat2keys(self, keys):
        key1, key2 = keys

        multi_cat = rt.cat2keys(key1, key2)
        assert len(key1) == len(key2)  # add test to check different length lists

        # these are the expected entries in the multi key categorical dictionary
        n = len(key1)
        expected_key1 = set(rt.FastArray([k for _ in range(n) for k in key1]))
        expected_key2 = set(rt.FastArray([k for k in key2 for _ in range(n)]))

        key_itr = iter(multi_cat.category_dict)
        actual_key1 = set(multi_cat.category_dict[next(key_itr)])
        actual_key2 = set(multi_cat.category_dict[next(key_itr)])

        not_nan = lambda x: not np.isnan(x)
        assert not set(
            filter(not_nan, expected_key1.symmetric_difference(actual_key1))
        ), f"\nexpected {expected_key1}\nactual {actual_key1}"
        assert not set(
            filter(not_nan, expected_key2.symmetric_difference(actual_key2))
        ), f"\nexpected {expected_key2}\nactual {actual_key2}"

        # need to handle tuple ordering and string dtype discrepancies
        # Taking the entries one by one of expected_key1 and expected_key2 should produce the
        # cartesian product of key1 and key2.
        # expected_product = {(k1, k2) for k1, k2 in itertools.product(key1, key2)}
        # actual_product = {(k1, k2) for k1, k2 in zip(actual_key1, actual_key2)}

        # not_nan = lambda tup: not np.isnan(tup[0]) or not np.isnan(tup[1])
        # assert not set(filter(not_nan, expected_product.symmetric_difference(actual_product))), f"expected {expected_product}\nactual {actual_product}\nmulti_cat {self.print_cat(multi_cat)}\n{self.print_keys(keys)}"
