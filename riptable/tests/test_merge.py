from collections import Counter
import operator
import unittest
import pytest
import itertools
from typing import Tuple, List, NamedTuple, Optional, Union
import numpy
import numpy as np
from numpy.testing import assert_array_equal, assert_array_compare

import riptable as rt
from riptable.rt_utils import normalize_keys
from riptable import arange, Cat, FA, stack_rows



# TODO: Implement test for merge2 that uses collections.Counter to check the gbkeys in the 'on' cols of the output Dataset
#       have the correct multiplicity. Use np.nditer(keycol) for iterating over the key column array (to pass to the Counter constructor)?
#         * Make sure to check the behavior with and without invalids in the key column(s).

# TODO: Implement tests for merge2 that check it's behavior when the keycols from 'left' and 'right' have different dtypes.
#         * What's the type of the keycol(s) in the merged Dataset?

# TODO: Implement tests for merge2 to check how it behaves when the keycols from 'left' and/or 'right' have a Grouping
#       containing groups with 0 elements, such as with a filtered Categorical.

# TODO: Implement test that verifies invalids are handled correctly (in the transformed columns) for a left/right/outer join
#       when the keycol(s) are something like an int32 (or Categorical with backing int8/int16/int32) and we have enough duplication
#       of keys (between 'left' and 'right') that the joined fancy index needs to be an int64.
#       * Check that the invalids in the int64 fancy index are correct (and not the invalid for int8/int16/int32).
#         * This doesn't actually matter due to the behavior of mbget (riptable fancy indexing), but might be important later
#           if numpy adopts the 'invalid flag' at the C level -- i.e. where numpy fancy indexing might recognize an invalid value
#           and propagate it, but may reject other negative values (like an int32 invalid in an int64 array).

# TODO: Implement test to verify we're able to handle the case where a merge (merge2, merge_lookup, etc.) is performed
#       with multiple key columns, and the same key is specified multiple times for one or both Datasets.
#       For example: rt.merge2(left_ds, right_ds, on=['a', 'b', ('c', 'a')], how='left')


xfail_rip260_outermerge_left_keep = pytest.mark.xfail(
    reason="RIP-260: 'keep' for the 'left' Dataset in a multi-column outer merge is broken and needs to be fixed.",
    raises=ValueError,
    #strict=True
)


class ConversionUtilityTest:
    def test_normalize(self):
        c1 = Cat(['A', 'B', 'C'])
        c2 = Cat(arange(3) + 1, ['A', 'B', 'C'])

        [d1], [d2] = normalize_keys(c1, c2)
        assert d1.dtype == d2.dtype

        [d1], [d2] = normalize_keys(
            arange(10, dtype=np.int32), arange(3, dtype=np.int64)
        )
        assert d1.dtype == d2.dtype

        # should force float upcasting
        [d1], [d2] = normalize_keys(
            arange(10, dtype=np.int32), arange(3, dtype=np.uint64)
        )
        assert d1.dtype == d2.dtype

        # should convert string to categorical
        [d1], [d2] = normalize_keys([c1], ['A', 'B'])
        assert d1.dtype == d2.dtype

        # should convert S1 to U4
        s1 = FA(['this', 'that'], unicode=True)
        s2 = FA(['t', 't'])
        [d1], [d2] = normalize_keys(s1, s2)
        assert d1.dtype == d2.dtype

    def test_stack_rows(self):
        d = {'test1': arange(3), 'test2': arange(1), 'test3': arange(2)}
        arr, cat = stack_rows(d)
        assert cat[5] == 'test3'
        assert arr[5] == 1


class MergeTest(unittest.TestCase):
    def test_merge_single(self):
        error_tol = 0.00001
        ds1 = rt.Dataset({'A': [0, 1, 6, 7], 'B': [1.2, 3.1, 9.6, 21]})
        ds2 = rt.Dataset({'X': [0, 1, 6, 9], 'C': [2.4, 6.2, 19.2, 53]})

        ds = rt.merge(ds1, ds2, left_on='A', right_on='X', how='inner')

        self.assertIsInstance(ds, rt.Dataset)
        self.assertEqual(ds.shape[0], 3)
        self.assertEqual(ds.shape[1], 4)
        self.assertTrue((ds.A._np == [0, 1, 6]).all())
        self.assertTrue((ds.X._np == [0, 1, 6]).all())
        self.assertTrue(
            (numpy.abs(ds.B._np - numpy.array([1.2, 3.1, 9.6])) < error_tol).all()
        )
        self.assertTrue(
            (numpy.abs(ds.C._np - numpy.array([2.4, 6.2, 19.2])) < error_tol).all()
        )

        ds1 = rt.Dataset(
            {'A': [0, 6, 9, 11], 'B': ['Q', 'R', 'S', 'T'], 'C': [2.4, 6.2, 19.2, 25.9]}
        )
        ds2 = rt.Dataset(
            {
                'A': [0, 1, 6, 10],
                'B': ['Q', 'R', 'R', 'T'],
                'E': [1.5, 3.75, 11.2, 13.1],
            }
        )

        ds = rt.merge(ds1, ds2, on=['A', 'B'], how='left')
        self.assertIsInstance(ds, rt.Dataset)
        self.assertEqual(ds.shape[0], 4)
        self.assertEqual(ds.shape[1], 4)
        self.assertTrue((ds.A._np == [0, 6, 9, 11]).all())
        self.assertTrue((ds.B._np == numpy.array([b'Q', b'R', b'S', b'T'])).all())
        self.assertTrue(
            (
                numpy.abs(ds.C._np - numpy.array([2.4, 6.2, 19.2, 25.9])) < error_tol
            ).all()
        )
        self.assertTrue(
            (numpy.abs(ds.E._np[0:2] - numpy.array([1.5, 11.2])) < error_tol).all()
        )
        self.assertTrue(numpy.isnan(ds.E._np[2:4]).all())

        ds = rt.merge(ds1, ds2, on=['A', 'B'], how='outer')
        self.assertIsInstance(ds, rt.Dataset)
        self.assertEqual(ds.shape[0], 6)
        self.assertEqual(ds.shape[1], 4)
        self.assertTrue((ds.A._np == [0, 6, 9, 11, 1, 10]).all())
        self.assertTrue(
            (ds.B._np == numpy.array([b'Q', b'R', b'S', b'T', b'R', b'T'])).all()
        )
        self.assertTrue(
            (
                numpy.abs(ds.C._np[0:4] - numpy.array([2.4, 6.2, 19.2, 25.9]))
                < error_tol
            ).all()
        )
        self.assertTrue(numpy.isnan(ds.C._np[4:6]).all())
        self.assertTrue(
            (
                numpy.abs(
                    ds.E._np[[0, 1, 4, 5]] - numpy.array([1.5, 11.2, 3.75, 13.10])
                )
                < error_tol
            ).all()
        )
        self.assertTrue(numpy.isnan(ds.E._np[2:4]).all())

        ds1 = rt.Dataset(
            {
                'A': [0, 6, 9, 11],
                'B': ['Q', 'R', 'S', 'T'],
                'Cc': [2.4, 6.2, 19.2, 25.9],
            }
        )
        ds2 = rt.Dataset(
            {
                'A': [0, 1, 6, 10],
                'B': ['Q', 'R', 'R', 'T'],
                'Ee': [1.5, 3.75, 11.2, 13.1],
            }
        )
        ds = rt.merge(ds1, ds2, on='A', columns_left=['Cc'])
        self.assertEqual(ds.shape[0], 4)
        self.assertEqual(ds.shape[1], 4)
        ds = rt.merge(ds1, ds2, on='A', columns_left='Cc')
        self.assertEqual(ds.shape[0], 4)
        self.assertEqual(ds.shape[1], 4)
        ds = rt.merge(ds1, ds2, on='A', columns_right=['Ee'])
        self.assertEqual(ds.shape[0], 4)
        self.assertEqual(ds.shape[1], 4)
        ds = rt.merge(ds1, ds2, on='A', columns_right='Ee')
        self.assertEqual(ds.shape[0], 4)
        self.assertEqual(ds.shape[1], 4)
        ds = rt.merge(ds1, ds2, on='A', columns_left='Cc', columns_right='B')
        self.assertEqual(ds.shape[0], 4)
        self.assertEqual(ds.shape[1], 3)

    def test_merge_different_left_right_on(self):
        dtype = np.dtype('i8')

        def _make_array(l):
            return rt.FastArray(l, dtype=dtype)

        inv = rt.INVALID_DICT[dtype.num]
        ds1 = rt.Dataset({'a': _make_array([5, 10, 15]), 'aa': _make_array([1, 2, 3])})
        ds2 = rt.Dataset(
            {'b': _make_array([10, 15, 20]), 'bb': _make_array([100, 200, 300])}
        )

        merged = rt.merge(ds1, ds2, left_on='a', right_on='b', how='inner')
        expected = rt.Dataset(
            {
                'a': _make_array([10, 15]),
                'b': _make_array([10, 15]),
                'aa': _make_array([2, 3]),
                'bb': _make_array([100, 200]),
            }
        )
        for col in expected.keys():
            assert_array_equal(merged[col], expected[col])

        merged = rt.merge(ds1, ds2, left_on='a', right_on='b', how='left')
        expected = rt.Dataset(
            {
                'a': _make_array([5, 10, 15]),
                'b': _make_array([inv, 10, 15]),
                'aa': _make_array([1, 2, 3]),
                'bb': _make_array([inv, 100, 200]),
            }
        )
        for col in expected.keys():
            assert_array_equal(merged[col], expected[col])

        merged = rt.merge(ds1, ds2, left_on='a', right_on='b', how='right')
        expected = rt.Dataset(
            {
                'a': _make_array([10, 15, inv]),
                'b': _make_array([10, 15, 20]),
                'aa': _make_array([2, 3, inv]),
                'bb': _make_array([100, 200, 300]),
            }
        )
        for col in expected.keys():
            assert_array_equal(merged[col], expected[col])

        merged = rt.merge(ds1, ds2, left_on='a', right_on='b', how='outer')
        expected = rt.Dataset(
            {
                'a': _make_array([5, 10, 15, inv]),
                'b': _make_array([inv, 10, 15, 20]),
                'aa': _make_array([1, 2, 3, inv]),
                'bb': _make_array([inv, 100, 200, 300]),
            }
        )
        for col in expected.keys():
            assert_array_equal(merged[col], expected[col])

        # Columns overlapping
        ds1 = rt.Dataset({'a': _make_array([1, 2, 3]), 'b': _make_array([5, 10, 15])})
        ds2 = rt.Dataset(
            {'a': _make_array([100, 200, 300]), 'b': _make_array([2, 3, 4])}
        )
        merged = rt.merge(ds1, ds2, left_on='a', right_on='b', how='outer')
        expected = rt.Dataset(
            {
                'a_x': _make_array([1, 2, 3, inv]),
                'b_x': _make_array([5, 10, 15, inv]),
                'a_y': _make_array([inv, 100, 200, 300]),
                'b_y': _make_array([inv, 2, 3, 4]),
            }
        )
        for col in expected.keys():
            assert_array_equal(merged[col], expected[col])

        # Multiple columns
        ds1 = rt.Dataset(
            {
                'a': _make_array([5, 10, 15]),
                'c': _make_array([1, 2, 2]),
                'aa': _make_array([1, 2, 3]),
            }
        )
        ds2 = rt.Dataset(
            {
                'b': _make_array([10, 15, 20]),
                'c': _make_array([2, 2, 3]),
                'bb': _make_array([100, 200, 300]),
            }
        )
        merged = rt.merge(
            ds1, ds2, left_on=['a', 'c'], right_on=['b', 'c'], how='outer'
        )
        expected = rt.Dataset(
            {
                'a': _make_array([5, 10, 15, inv]),
                'b': _make_array([inv, 10, 15, 20]),
                'c': _make_array([1, 2, 2, 3]),
                'aa': _make_array([1, 2, 3, inv]),
                'bb': _make_array([inv, 100, 200, 300]),
            }
        )
        for col in expected.keys():
            assert_array_equal(merged[col], expected[col])

    def test_merge_suffixes(self):
        ds1 = rt.Dataset({'A': [1, 2, 3], 'B': [10, 20, 30], 'B_1': [11, 21, 31]})
        ds2 = rt.Dataset({'A': [1, 2, 3], 'B': [100, 200, 300]})

        # Test default value
        ds = ds1.merge(ds2, on='A')
        self.assertEqual(ds.shape, (3, 4))
        self.assertEqual({'A', 'B_x', 'B_1', 'B_y'}, set(ds.keys()))

        # Test custom value
        ds = ds1.merge(ds2, on='A', suffixes=('_l', '_r'))
        self.assertEqual(ds.shape, (3, 4))
        self.assertEqual({'A', 'B_l', 'B_1', 'B_r'}, set(ds.keys()))

        # Test exception enabling
        with pytest.raises(ValueError):
            ds1.merge(ds2, on='A', suffixes=(False, False))

    def test_merge_indicator(self):
        ds1 = rt.Dataset({'A': [1, 2], 'B': [10, 20]})
        ds2 = rt.Dataset({'A': [1, 3], 'C': [10, 30]})

        # Test default value
        ds = ds1.merge(ds2, on='A', how='left', indicator=True)
        self.assertEqual({'A', 'B', 'C', 'merge_indicator'}, set(ds.keys()))
        self.assertTrue(
            (ds['merge_indicator'].expand_array == ['both', 'left_only']).all()
        )

        # Test custom value
        ds = ds1.merge(ds2, on='A', how='left', indicator='source')
        self.assertEqual({'A', 'B', 'C', 'source'}, set(ds.keys()))
        self.assertTrue((ds['source'].expand_array == ['both', 'left_only']).all())

        # Test "right" merge
        ds = ds1.merge(ds2, on='A', how='right', indicator=True)
        self.assertTrue(
            (ds['merge_indicator'].expand_array == ['both', 'right_only']).all()
        )

        # Test "inner" merge
        ds = ds1.merge(ds2, on='A', how='inner', indicator=True)
        self.assertTrue((ds['merge_indicator'].expand_array == ['both']).all())

        # Test "outer" merge
        ds = ds1.merge(ds2, on='A', how='outer', indicator=True)
        self.assertTrue(
            (
                ds['merge_indicator'].expand_array
                == ['both', 'left_only', 'right_only']
            ).all()
        )

        # Test name collision
        with pytest.raises(ValueError):
            ds1.merge(ds2, on='A', indicator='B')

    def test_outer_merge_categorical(self):

        cats = rt.FastArray([b'none', b'added', b'removed', b'all'], dtype='|S7')

        c1 = rt.Categorical([4, 3], cats)
        c2 = rt.Categorical([4, 2], cats)

        pnl1 = rt.FastArray([1724000.0, -184349.71])
        pnl2 = rt.FastArray([1711000.0, 12122.53])

        ds1 = rt.Dataset({'send_activity': c1, 'd2_pnl': pnl1})
        ds2 = rt.Dataset({'send_activity': c2, 'd2_pnl': pnl2})

        ds3 = rt.Dataset({'send_activity': c1.expand_array, 'd2_pnl': pnl1})
        ds4 = rt.Dataset({'send_activity': c2.expand_array, 'd2_pnl': pnl2})

        merge_cats = rt.merge(ds1, ds2, on='send_activity', how='outer')
        merge_strings = rt.merge(ds3, ds4, on='send_activity', how='outer')

        assert_array_equal(
            merge_cats.send_activity.expand_array, merge_strings.send_activity
        )
        notnanmask_cat1 = merge_cats.d2_pnl_x.isnotnan()
        notnanmask_cat2 = merge_cats.d2_pnl_y.isnotnan()
        notnanmask_str1 = merge_strings.d2_pnl_x.isnotnan()
        notnanmask_str2 = merge_strings.d2_pnl_y.isnotnan()

        assert_array_equal(notnanmask_cat1, notnanmask_str1)
        assert_array_equal(notnanmask_cat2, notnanmask_str2)

        masked_c1 = merge_cats.d2_pnl_x[notnanmask_cat1]
        masked_c2 = merge_cats.d2_pnl_y[notnanmask_cat2]
        masked_s1 = merge_strings.d2_pnl_x[notnanmask_str1]
        masked_s2 = merge_strings.d2_pnl_y[notnanmask_str2]

        assert_array_equal(masked_c1, masked_s1)
        assert_array_equal(masked_c2, masked_s2)

    def test_merge_dict_backed_categorical(self):
        c1 = rt.Categorical([1, 2, 3], categories={1: 'a', 2: 'b', 3: 'c'})
        c2 = c1[[True, False, True]]
        b_arr = rt.FastArray([1, 2, 3])
        c_arr = rt.FastArray([10, 30])
        ds1 = rt.Dataset({'a': c1, 'b': b_arr})
        ds2 = rt.Dataset({'a': c2, 'c': c_arr})
        merged = rt.merge(ds1, ds2, on='a', how='left')
        assert_array_equal(b_arr, merged['b'])
        expected_c_arr = rt.FastArray([10, 0, 30])
        expected_c_arr[1] = expected_c_arr.inv
        assert_array_equal(expected_c_arr, merged['c'])


class BagHelpers:
    """
    Test helper methods for bags/multisets (based on the `Counter` class).

    These methods implement relational/algebraic operations on bags/multisets; they're used
    for verifying that join results have the correct/expected multiplicities for the keys.
    """

    @staticmethod
    def from_grouping(grouping: rt.Grouping) -> Counter:
        """Create a Counter from the keys and their multiplicities in a Grouping object."""

        # Get the keys (as tuples, if a multi-key Grouping).
        gbkeys = grouping.gbkeys.values()
        keys = next(iter(gbkeys)) if len(gbkeys) == 1 else zip(*tuple(gbkeys))

        # Zip each key with it's multiplicity, then add them to the Counter.
        keys_and_counts = zip(keys, grouping.ncountkey)
        result = Counter(keys_and_counts)

        # If the Grouping has an 'invalid' element, add the number of
        # invalid elements to the Counter.
        if grouping.base_index > 0:
            result[None] = grouping.ncountkey[0]

        return result

    @staticmethod
    def left_join(left: Counter, right: Counter) -> Counter:
        """Perform a 'left join' of two bags."""

        # Iterate over the 'left' bag. For any keys there which are
        # also in 'right we'll multiply the counts together and add them to the result;
        # any keys which aren't in right are just copied over with their counts.
        result = left.copy()
        for key, rcount in right.items():
            # Ignore the None key -- nulls/invalids aren't joined on (for SQL semantics).
            if key is None:
                continue
            elif key in left:
                # Get this key's multiplicity in 'left'.
                lcount = left[key]

                # The multiplicity for this key in the joined result
                # is the product of their individual multiplicities.
                result[key] = lcount * rcount

        return result

    @staticmethod
    def inner_join(left: Counter, right: Counter) -> Counter:
        """Perform a 'inner join' of two bags."""

        # Find the keys which exist in both bags.
        shared_keys = left.keys() & right.keys()

        result: Counter = Counter()
        for key in shared_keys:
            # Ignore the None key -- nulls/invalids aren't joined on (for SQL semantics).
            if key is None:
                continue

            # The multiplicity of the key in the joined bag
            # is the product of the left/right multiplicities.
            result[key] = left[key] * right[key]

        return result

    @staticmethod
    def outer_join(left: Counter, right: Counter) -> Counter:
        """Perform a 'full outer join' of two bags."""

        # Iterate over the 'left' bag. For any keys there which are
        # also in 'right we'll multiply the counts together and add them to the result;
        # any keys which aren't in right are just copied over with their counts.
        result = left.copy()
        for key, rcount in right.items():
            # For the None key -- representing null/invalid -- we don't join / multiply out
            # the rows, but we do copy them over to the joined result.
            if key is None:
                lcount = left.get(key)
                result[key] = rcount if lcount is None else lcount + rcount
            elif key in left:
                # Get this key's multiplicity in 'left'.
                lcount = left[key]

                # The multiplicity for this key in the joined result
                # is the product of their individual multiplicities.
                result[key] = lcount * rcount
            else:
                # Just copy the key and it's count over.
                result[key] = rcount

        return result


def _create_merge_paramverify_dsets() -> Tuple[rt.Dataset, rt.Dataset]:
    ds1 = rt.Dataset()
    ds1['Foo'] = [0, 1, 4, 6, 8, 9, 11, 16, 19, 30]
    ds1['Bar'] = [10, 12, 15, 11, 10, 9, 13, 7, 9, 10]

    # ds2 purposely has a different number of rows from ds1;
    # this ensures we're not making assumptions that the Datasets have the same
    # number of rows, since that's going to be an infrequent occurrence in practice.
    ds2 = rt.Dataset()
    ds2['Bar'] = [0, 0, 5, 7, 8, 10, 12, 15]
    ds2['Baz'] = [20, 21, 22, 23, 24, 25, 26, 27]

    return ds1, ds2


def test_merge_on_and_left_on_disallowed():
    # Test that `merge` raises a ValueError if a user calls it by specifying both
    # the 'on' and 'left_on' parameters.
    (ds1, ds2) = _create_merge_paramverify_dsets()

    with pytest.raises(ValueError):
        rt.merge(ds1, ds2, on='Bar', left_on='Foo')


def test_merge_on_and_right_on_disallowed():
    # Test that `merge` raises a ValueError if a user calls it by specifying both
    # the 'on' and 'right_on' parameters.
    (ds1, ds2) = _create_merge_paramverify_dsets()

    with pytest.raises(ValueError):
        rt.merge(ds1, ds2, on='Bar', right_on='Baz')


@pytest.mark.parametrize("right_on", ['Baz', ['Baz']])
def test_merge_on_xor_left_on_required(right_on):
    # Test that `merge` raises a ValueError if a user calls it without specifying
    # either the 'on' or 'left_on' parameters.
    (ds1, ds2) = _create_merge_paramverify_dsets()

    with pytest.raises(ValueError):
        rt.merge(ds1, ds2, right_on=right_on)


@pytest.mark.parametrize("left_on", ['Foo', ['Foo']])
def test_merge_on_xor_right_on_required(left_on):
    # Test that `merge` raises a ValueError if a user calls it without specifying
    # either the 'on' or 'right_on' parameters.
    (ds1, ds2) = _create_merge_paramverify_dsets()

    with pytest.raises(ValueError):
        rt.merge(ds1, ds2, left_on=left_on)


@pytest.mark.parametrize("right_on", ['Bar', ['Bar']])
def test_merge_requires_left_on_columns_present(right_on):
    # Test that `merge` raises a ValueError if a user calls it and specifies
    # a column name in 'left_on' which does not exist in the 'left' Dataset.
    (ds1, ds2) = _create_merge_paramverify_dsets()

    bad_colname = 'Hello'
    assert (
        bad_colname not in ds1.keys()
    )  # Make sure the bad key really isn't in the Dataset

    with pytest.raises(ValueError):
        rt.merge(ds1, ds2, left_on=bad_colname, right_on=right_on)


@pytest.mark.parametrize("left_on", ['Bar', ['Bar']])
def test_merge_requires_right_on_columns_present(left_on):
    # Test that `merge` raises a ValueError if a user calls it and specifies
    # a column name in 'right_on' which does not exist in the 'right' Dataset.
    (ds1, ds2) = _create_merge_paramverify_dsets()

    bad_colname = 'Hello'
    assert (
        bad_colname not in ds2.keys()
    )  # Make sure the bad key really isn't in the Dataset

    with pytest.raises(ValueError):
        rt.merge(ds1, ds2, left_on=left_on, right_on=bad_colname)

    with pytest.raises(ValueError):
        rt.merge(ds1, ds2, left_on=left_on, right_on=[bad_colname])


@pytest.mark.parametrize("left_on", ['Foo', ['Foo']])
@pytest.mark.parametrize("right_on", ['Baz', ['Baz']])
def test_merge_requires_columns_left_columns_present(left_on, right_on):
    # Test that `merge` raises a ValueError if a user calls it and specifies
    # a column name in 'columns_left' which does not exist in the 'left' Dataset.
    (ds1, ds2) = _create_merge_paramverify_dsets()

    bad_colname = 'Hello'
    assert (
        bad_colname not in ds1.keys()
    )  # Make sure the bad key really isn't in the Dataset

    with pytest.raises(ValueError):
        rt.merge(ds1, ds2, left_on=left_on, right_on=right_on, columns_left=bad_colname)

    with pytest.raises(ValueError):
        rt.merge(
            ds1, ds2, left_on=left_on, right_on=right_on, columns_left=[bad_colname]
        )

    with pytest.raises(ValueError):
        rt.merge(
            ds1, ds2, left_on=left_on, right_on=right_on, columns_left={bad_colname}
        )


@pytest.mark.parametrize("left_on", ['Foo', ['Foo']])
@pytest.mark.parametrize("right_on", ['Baz', ['Baz']])
def test_merge_requires_columns_right_columns_present(left_on, right_on):
    # Test that `merge` raises a ValueError if a user calls it and specifies
    # a column name in 'columns_right' which does not exist in the 'left' Dataset.
    (ds1, ds2) = _create_merge_paramverify_dsets()

    bad_colname = 'Hello'
    assert (
        bad_colname not in ds2.keys()
    )  # Make sure the bad key really isn't in the Dataset

    with pytest.raises(ValueError):
        rt.merge(
            ds1, ds2, left_on=left_on, right_on=right_on, columns_right=bad_colname
        )

    with pytest.raises(ValueError):
        rt.merge(
            ds1, ds2, left_on=left_on, right_on=right_on, columns_right=[bad_colname]
        )

    with pytest.raises(ValueError):
        rt.merge(
            ds1, ds2, left_on=left_on, right_on=right_on, columns_right={bad_colname}
        )


@pytest.mark.parametrize("left_on", ['Foo', ['Foo']])
@pytest.mark.parametrize("right_on", ['Baz', ['Baz']])
def test_merge_columns_overlap_no_suffix(left_on, right_on):
    # Test that `merge` raises a ValueError if it detects a naming collision between
    # the left and right datasets and no suffix was specified for resolving conflicts.
    (ds1, ds2) = _create_merge_paramverify_dsets()

    with pytest.raises(ValueError):
        rt.merge(ds1, ds2, left_on=left_on, right_on=right_on, suffixes=('', ''))


@pytest.mark.parametrize("left_on", ['Foo', ['Foo']])
@pytest.mark.parametrize("right_on", ['Baz', ['Baz']])
def test_merge_columns_overlap_same_suffixes(left_on, right_on):
    # Test that `merge` raises a ValueError if it detects a naming collision between
    # the left and right datasets and the same suffix was specified for both sides.
    (ds1, ds2) = _create_merge_paramverify_dsets()

    with pytest.raises(ValueError):
        rt.merge(ds1, ds2, left_on=left_on, right_on=right_on, suffixes=('_x', '_x'))


@pytest.mark.parametrize("left_on", ['Foo', ['Foo']])
@pytest.mark.parametrize("right_on", ['Baz', ['Baz']])
def test_merge_columns_overlap_errorif_left_suffix_collision(left_on, right_on):
    # Test that `merge` raises a ValueError if it detects a naming collision remaining
    # in the non-key columns being selected from either the left or right dataset even
    # after applying the left suffix to attempt to resolve a naming collision.
    (ds1, ds2) = _create_merge_paramverify_dsets()

    overlapping_colname = 'Bar'
    left_suffix = '_x'
    colliding_name = f'{overlapping_colname}{left_suffix}'
    ds1[colliding_name] = rt.arange(len(ds1.Bar))

    # Verify the test data is self-consistent before running the code we want to test.
    assert overlapping_colname in ds1.keys()
    assert overlapping_colname in ds2.keys()
    assert colliding_name in ds1.keys()

    with pytest.raises(ValueError):
        rt.merge(
            ds1, ds2, left_on=left_on, right_on=right_on, suffixes=(left_suffix, '_y')
        )

    # Demonstrate that the collision detection doesn't care which Dataset the
    # 2nd colliding column name is found in -- it should still raise an error.
    del ds1[colliding_name]
    ds2[colliding_name] = rt.arange(len(ds2.Bar))
    assert colliding_name in ds2.keys()

    with pytest.raises(ValueError):
        rt.merge(
            ds1, ds2, left_on=left_on, right_on=right_on, suffixes=(left_suffix, '_y')
        )


@pytest.mark.parametrize("left_on", ['Foo', ['Foo']])
@pytest.mark.parametrize("right_on", ['Baz', ['Baz']])
def test_merge_columns_overlap_errorif_right_suffix_collision(left_on, right_on):
    # Test that `merge` raises a ValueError if it detects a naming collision remaining
    # in the non-key columns being selected from either the left or right dataset even
    # after applying the right suffix to attempt to resolve a naming collision.
    (ds1, ds2) = _create_merge_paramverify_dsets()

    overlapping_colname = 'Bar'
    right_suffix = '_y'
    colliding_name = f'{overlapping_colname}{right_suffix}'
    ds2[colliding_name] = rt.arange(len(ds2.Bar))

    # Verify the test data is self-consistent before running the code we want to test.
    assert overlapping_colname in ds1.keys()
    assert overlapping_colname in ds2.keys()
    assert colliding_name in ds2.keys()

    with pytest.raises(ValueError):
        rt.merge(
            ds1, ds2, left_on=left_on, right_on=right_on, suffixes=('_x', right_suffix)
        )

    # Demonstrate that the collision detection doesn't care which Dataset the
    # 2nd colliding column name is found in -- it should still raise an error.
    del ds2[colliding_name]
    ds1[colliding_name] = rt.arange(len(ds1.Bar))
    assert colliding_name in ds1.keys()

    with pytest.raises(ValueError):
        rt.merge(
            ds1, ds2, left_on=left_on, right_on='Baz', suffixes=('_x', right_suffix)
        )


def _create_merge_column_compat_rules_integer_cases(
    left_rowcount: int, right_rowcount: int
) -> list:
    cases = []
    types = [np.dtype(x) for x in np.typecodes['AllInteger']]
    for x in types:
        for y in types:
            x_step, y_step = 2, 3
            cases.append(
                pytest.param(
                    rt.FA(
                        np.arange(
                            start=1,
                            stop=1 + (x_step * left_rowcount),
                            step=x_step,
                            dtype=x,
                        )
                    ),
                    rt.FA(
                        np.arange(
                            start=1, stop=1 + (y_step * right_rowcount), step=y_step
                        ),
                        dtype=y,
                    ),
                    True,
                    False,
                    id=f"integers--{x}__{y}",
                )
            )
    return cases


_merge_column_compat_rules_cases = [
    # Test cases for obviously malformed inputs.
    pytest.param(
        [],
        rt.FA(np.arange(start=1, stop=25, step=3)),
        False,
        True,
        id="left_nullary_key",
    ),
    pytest.param(
        rt.FA(np.arange(start=1, stop=21, step=2)),
        [],
        False,
        True,
        id="right_nullary_key",
    ),
    # Test case: Disallow a Date column to be joined with a normal FastArray containing integer data.
    pytest.param(
        rt.Date(
            [
                '2019-03-15',
                '2019-04-18',
                '2019-05-17',
                '2019-06-21',
                '2019-07-19',
                '2019-08-16',
                '2019-09-20',
                '2019-10-18',
                '2019-11-15',
                '2019-12-20',
            ]
        ),
        rt.arange(8),
        False,
        True,
        id="Date__int32",
    ),
    pytest.param(
        [
            rt.FA(np.arange(start=1, stop=21, step=2, dtype=np.int8)),
            rt.FA(np.arange(start=1, stop=21, step=2, dtype=np.int64)),
        ],
        [
            rt.FA(np.arange(start=1, stop=25, step=3, dtype=np.uint16)),
            rt.FA(np.arange(start=1, stop=25, step=3, dtype=np.int32)),
        ],
        True,
        False,
        id="(int8, int64)__(uint16, int32)--mixed-size-sign-integers-multicol",
    ),
    # This is a special case -- when we're merging on columns where one side is a
    # signed integer and the other is the largest unsigned integer size (e.g. uint64)
    # we want to *carefully* allow this to work by e.g. checking whether the signed column(s)
    # have any negative values, and if not we'll use uint64 as the joined column type in the output.
    pytest.param(
        rt.FA(np.arange(start=1, stop=21, step=2, dtype=np.int32)),
        rt.FA(np.arange(start=1, stop=25, step=3, dtype=np.uint64)),
        True,
        False,
        id="int32__uint64--signed-vs-max-unsigned-size-integers",
    ),
    # Test case: Disallow mismatched cardinality of join keys/cols
    pytest.param(
        (
            rt.FA(np.arange(start=1, stop=21, step=2)),
            rt.Date(
                [
                    '2019-03-15',
                    '2019-04-18',
                    '2019-05-17',
                    '2019-06-21',
                    '2019-07-19',
                    '2019-08-16',
                    '2019-09-20',
                    '2019-10-18',
                    '2019-11-15',
                    '2019-12-20',
                ]
            ),
        ),
        rt.FA(np.arange(start=1, stop=25, step=3)),
        False,
        True,
        id="int32_Date__int32--mismatched-cardinality",
    ),
    # Test case: Disallow merging on a mix of integer and floating-point columns.
    pytest.param(
        rt.FA(np.linspace(0.1, 1.0, 10, dtype=np.float32)),
        rt.arange(8),
        False,
        False,  # TODO: Change to True for riptable 1.4.x, when this will be disallowed.
        id="float32__int32",
    ),
    # Merging on string columns of different lengths should be allowed.
    pytest.param(
        rt.FastArray(
            ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM', 'AAPL', 'IBM', 'AMZN', 'AAPL', 'FB']
        ),  # dtype='|S4'
        rt.FastArray(['T', 'AA', 'FB', 'F', 'KO', 'FB', 'FB', 'KO']),  # dtype='|S2'
        True,
        False,
        id="|S4__|S2",
    ),
    pytest.param(
        rt.FastArray(
            ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM', 'AAPL', 'IBM', 'AMZN', 'AAPL', 'FB']
        ).astype(
            np.unicode_
        ),  # dtype='<U4'
        rt.FastArray(['T', 'AA', 'FB', 'F', 'KO', 'FB', 'FB', 'KO']).astype(
            np.unicode_
        ),  # dtype='<U2'
        True,
        False,
        id="<U4__<U2",
    ),
    # Allow merging on string columns which are a mix of ASCII and Unicode.
    # As of 2020-Jan-10, merge is implemented in such a way (using ismember) that passing
    # string columns of different types like this causes an implicit conversion (in one direction
    # or the other, preferring the U->S conversion since it's more compact in memory);
    # we might want to make this and any other cases which could cause implicit allocations/conversions
    # issue detect it and emit a warning to let the user know about the potential performance impact.
    pytest.param(
        rt.FastArray(
            ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM', 'AAPL', 'IBM', 'AMZN', 'AAPL', 'FB']
        ),  # dtype='|S4'
        rt.FastArray(
            ['T', 'AA', 'FB', 'F', 'KO', 'FB', 'FB', 'KO'], unicode=True
        ),  # dtype='<U2'
        True,
        False,
        id="|S4__<U2",
    ),
    #######################################################################
    # Test cases for Categoricals -- they're treated specially since they have
    # their own special semantics for merging.
    #######################################################################
    pytest.param(
        rt.Categorical(
            [4, 3, 3, 2, 4, 1, 1, 1, 3, 2],
            # We make sure here the category values are different (but compatible);
            # some later test for the actual merge functionality would want to verify
            # the two Categoricals are merged so that the resulting Categorical key
            # in the output Dataset has a merged set of category values.
            rt.FastArray(
                [b'none', b'added', b'all', b'modified', b'verified'], dtype='|S8'
            ),
        ),
        rt.Categorical(
            [4, 2, 1, 1, 4, 3, 2, 3],
            rt.FastArray([b'none', b'added', b'removed', b'all'], dtype='|S7'),
        ),
        True,
        False,
        id="Cat[|S8]__Cat[|S7]",
    ),
    # Allow merging Categoricals of strings with different character types (ASCII vs. Unicode).
    # See the comment for the non-Categorical version of this test above for more details on this situation.
    pytest.param(
        rt.Cat(
            rt.FastArray(
                [
                    'AAPL',
                    'AMZN',
                    'FB',
                    'GOOG',
                    'IBM',
                    'AAPL',
                    'IBM',
                    'AMZN',
                    'AAPL',
                    'FB',
                ]
            )
        ),  # dtype='|S4'
        rt.Cat(
            rt.FastArray(['T', 'AA', 'FB', 'F', 'KO', 'FB', 'FB', 'KO'], unicode=True)
        ),  # dtype='<U2'
        True,
        False,
        id="Cat[|S4]__Cat[<U2]",
    ),
    # Merging on keys where one side is a Categorical and the other side is
    # not a Categorical but has an array type compatible with the Categorical's
    # categories is allowed.
    pytest.param(
        rt.FastArray(
            ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM', 'AAPL', 'IBM', 'AMZN', 'AAPL', 'FB']
        ),  # categories dtype='|S4'
        rt.Categorical(['T', 'AA', 'FB', 'F', 'KO', 'FB', 'FB', 'KO']),  # dtype='|S2'
        True,
        False,
        id="|S4__Cat[|S2]",
    ),
    pytest.param(
        rt.Categorical(
            rt.Date(
                [
                    '2019-03-15',
                    '2019-04-18',
                    '2019-05-17',
                    '2019-06-21',
                    '2019-07-19',
                    '2019-08-16',
                    '2019-09-20',
                    '2019-10-18',
                    '2019-11-15',
                    '2019-12-20',
                ]
            )
        ),
        rt.Date(
            [
                '2019-03-15',
                '2019-05-17',
                '2019-06-21',
                '2019-07-19',
                '2019-08-16',
                '2019-09-20',
                '2019-10-18',
                '2019-12-20',
            ]
        ),
        True,
        False,
        id="Cat[Date]__Date",
    ),
    # Disallow merging where both sides are Categoricals but their categories have
    # types that'd be incompatible for merging.
    pytest.param(
        rt.Categorical(
            rt.Date(
                [
                    '2019-03-15',
                    '2019-04-18',
                    '2019-05-17',
                    '2019-06-21',
                    '2019-07-19',
                    '2019-08-16',
                    '2019-09-20',
                    '2019-10-18',
                    '2019-11-15',
                    '2019-12-20',
                ]
            )
        ),
        rt.Categorical(
            rt.FastArray(
                [
                    '2019-03-15',
                    '2019-05-17',
                    '2019-06-21',
                    '2019-07-19',
                    '2019-08-16',
                    '2019-09-20',
                    '2019-10-18',
                    '2019-12-20',
                ]
            )
        ),
        False,
        True,
        id="Cat[Date]__Cat[|S10]",
    ),
    # Disallow merging on Categoricals which have different ranks.
    pytest.param(
        rt.Cat(
            [
                rt.Date(
                    [
                        '2019-03-15',
                        '2019-04-18',
                        '2019-05-17',
                        '2019-06-21',
                        '2019-07-19',
                        '2019-08-16',
                        '2019-09-20',
                        '2019-10-18',
                        '2019-11-15',
                        '2019-12-20',
                    ]
                ),
                rt.FA(
                    [
                        'aaaaaaaaaa',
                        'bbbbbbbbbb',
                        'cccccccccc',
                        'dddddddddd',
                        'eeeeeeeeee',
                        'ffffffffff',
                        'gggggggggg',
                        'hhhhhhhhhh',
                        'iiiiiiiiii',
                        'jjjjjjjjjj',
                    ]
                ),
            ]
        ),
        rt.Cat(
            rt.Date(
                [
                    '2019-03-15',
                    '2019-05-17',
                    '2019-06-21',
                    '2019-07-19',
                    '2019-08-16',
                    '2019-09-20',
                    '2019-10-18',
                    '2019-12-20',
                ]
            )
        ),
        False,
        True,
        id="Cat[Date_|S10]__Cat[Date]",
    )
    # TODO: Additional test cases
    #   * What about multi-key categoricals?
    #       * Should the user be allowed to pass a multi-key Cat for one Dataset's merge key, and on the other side pass a
    #         tuple/list of FastArray/Date/DateTimeNano/etc. as long as they match the types of the Categorical's individual keys?
    #         How does this mesh with using multiple columns in the join key?
    #
]


@pytest.mark.parametrize(
    "left_keycols,right_keycols,is_allowed,is_hard_error",
    itertools.chain.from_iterable(
        [
            _create_merge_column_compat_rules_integer_cases(10, 8),
            _merge_column_compat_rules_cases,
        ]
    ),
)
def test_merge_columns_compat_rules(
    left_keycols, right_keycols, is_allowed, is_hard_error
):
    # Test that `merge` disallows incompatible column types to be used as join keys,
    # such as Date and a plain integer FastArray.
    (ds1, ds2) = _create_merge_paramverify_dsets()

    # Add the left key and right key column(s) to the corresponding Datasets.
    def add_keycols_to_dataset(ds, colname_base: str, keycols) -> List[str]:
        if isinstance(keycols, (list, tuple)):
            colnames = []
            for idx, keycol in enumerate(keycols):
                colname = f'{colname_base}{idx}'
                colnames.append(colname)
                ds[colname] = keycol
            return colnames
        else:
            # Assume this is a single column that's not in a list/tuple
            ds[colname_base] = keycols
            return [colname_base]

    left_keynames = add_keycols_to_dataset(ds1, 'LKey', left_keycols)
    right_keynames = add_keycols_to_dataset(ds2, 'RKey', right_keycols)

    # If the merge is expected to be allowed for these keys,
    # just try to run it -- if there's an exception the test will fail.
    # The merge is not allowed, verify an exception is raised as expected.
    if is_allowed:
        rt.merge(ds1, ds2, left_on=left_keynames, right_on=right_keynames)
    elif is_hard_error:
        with pytest.raises(ValueError):
            rt.merge(ds1, ds2, left_on=left_keynames, right_on=right_keynames)
    else:
        # Check for 'soft' errors, which we currently manifest as warnings.
        with pytest.warns(Warning):
            rt.merge(ds1, ds2, left_on=left_keynames, right_on=right_keynames)


@pytest.mark.parametrize(
    "on,side_on,for_left,is_error,expected",
    [
        # Cases covering the "old-style" 'on'/'left_on'/'right_on' parameter parsing.
        # Note that the 'for_left' parameter is irrelevant for this parsing mode.
        pytest.param('foo', None, True, False, ['foo'], id="on_str"),
        pytest.param(None, 'foo', True, False, ['foo'], id="side_on_str"),
        pytest.param('foo', 'foo', True, True, None, id="on_and_side_on"),
        pytest.param(
            ['foo', 'bar'], None, True, False, ['foo', 'bar'], id="on_List[str]"
        ),
        pytest.param(
            None, ['foo', 'bar'], True, False, ['foo', 'bar'], id="side_on_List[str]"
        ),
        # Cases for the "new-style" 'on' parameter where tuples can also be passed
        # within the list to provide separate left/right column names.
        pytest.param(('foo',), None, True, False, ['foo'], id="on_Tuple[str]"),
        pytest.param(
            ('foo', 'bar'), None, True, False, ['foo'], id="on_Tuple[str, str]"
        ),
        pytest.param(
            ('foo', 'bar', 'baz'), None, True, True, None, id="on_Tuple[str, str, str]"
        ),
        pytest.param([('foo',)], None, True, False, ['foo'], id="on_List[Tuple[str]]"),
        pytest.param(
            ['foo', ('bar', 'baz')],
            None,
            True,
            False,
            ['foo', 'bar'],
            id="on_List[str; Tuple[str, str]]--left",
        ),
        pytest.param(
            ['foo', ('bar', 'baz')],
            None,
            False,
            False,
            ['foo', 'baz'],
            id="on_List[str; Tuple[str, str]]--right",
        ),
    ],
)
def test_merge_extract_on_columns(
    on, side_on, for_left: bool, is_error: bool, expected: List[str]
):
    """
    Basic input/output checks for the private _extract_on_columns() function.
    """
    if is_error:
        with pytest.raises(ValueError):
            rt.rt_merge._extract_on_columns(on, side_on, for_left, 'on', is_optional=False)

    else:
        result = rt.rt_merge._extract_on_columns(on, side_on, for_left, 'on', is_optional=False)
        assert result == expected


class TestDataset:
    """
    Helper methods for creating Dataset instances used by merge / merge_asof tests.
    """

    class HockeyDataset(NamedTuple):
        """
        Test data taken from: http://doc.nuodb.com/Latest/Default.htm#About-Join-Operations-Supported-by-NuoDB.htm
        """

        teams: rt.Dataset
        players: rt.Dataset

        teams_grp: rt.Grouping
        players_grp: rt.Grouping

        on: str

    @staticmethod
    def hockey() -> HockeyDataset:
        # Test data taken from: http://doc.nuodb.com/Latest/Default.htm#About-Join-Operations-Supported-by-NuoDB.htm

        test_teams = rt.Dataset()
        test_teams.team_id = rt.Cat(
            [
                'BOS',
                'BUF',
                'CAR',
                'FLO',
                'MTL',
                'NJD',
                'NYI',
                'NYR',
                'OTT',
                'PHI',
                'PIT',
                'TBL',
                'TOR',
                'WAS',
                'WPG',
            ]
        )
        test_teams.team_name = rt.FA(
            [
                "Boston Bruins",
                "Buffalo Sabres",
                "Carolina Hurricanes",
                "Florida Panthers",
                "Montreal Canadiens",
                "New Jersey Devils",
                "New York Islanders",
                "New York Rangers",
                "Ottawa Senators",
                "Philadelphia Flyers",
                "Pittsburgh Penguins",
                "Tampa Bay Lightning",
                "Toronto Maple Leafs",
                "Washington Capitals",
                "Winnipeg Jets",
            ]
        )
        test_teams.wins = rt.FA(
            [49, 39, 33, 38, 31, 48, 34, 51, 41, 47, 51, 38, 35, 42, 37]
        )
        test_teams.losses = rt.FA(
            [29, 32, 33, 26, 35, 28, 37, 24, 31, 26, 25, 36, 37, 32, 35]
        )
        test_teams.division_id = rt.Cat(
            [
                'NE',
                'NE',
                'SE',
                'SE',
                'NE',
                'AT',
                'AT',
                'AT',
                'NE',
                'AT',
                'AT',
                'SE',
                'NE',
                'SE',
                'SE',
            ]
        )

        test_players = rt.Dataset()
        test_players.playerid = rt.Cat(
            [
                "bachmri01",
                "bouchbr01",
                "conklty01",
                "crawfco01",
                "drouije01",
                "ellisda01",
                "emeryra01",
                "enrotjh01",
                "gigueje01",
                "hillejo01",
                "howarji01",
                "hunwish01",
                "irvinle01",
                "karlshe01",
                "khudoan01",
                "kiprumi01",
                "lehtoka01",
                "macdojo01",
                "macindr01",
                "masonst01",
                "millery01",
                "murphmi02",
                "peterju01",
                "rasktu01",
                "raycran01",
                "sanfocu01",
                "tarkkii01",
                "thomati01",
                "turcoma01",
                "varlasi01",
                "wardca01",
                "yorkal01",
            ]
        )
        test_players.player_name = rt.FA(
            [
                "Richard Bachman",
                "Brian Boucher",
                "Ty Conklin",
                "Corey Crawford",
                "Jeff Drouin-Deslauriers",
                "Dan Ellis",
                "Ray Emery",
                "Jhonas Enroth",
                "Jean-Sebastien Giguere",
                "Jonas Hiller",
                "Jimmy Howard",
                "Shawn Hunwick",
                "Leland Irving",
                "Henrik Karlsson",
                "Anton Khudobin",
                "Miikka Kiprusoff",
                "Kari Lehtonen",
                "Joey MacDonald",
                "Drew MacIntyre",
                "Steve Mason",
                "Ryan Miller",
                "Mike Murphy",
                "Justin Peters",
                "Tuukka Rask",
                "Andrew Raycroft",
                "Curtis Sanford",
                "Iiro Tarkki",
                "Tim Thomas",
                "Marty Turco",
                "Semyon Varlamov",
                "Cam Ward",
                "Allen York",
            ]
        )
        test_players.year = rt.FA(
            [
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
                2011,
            ]
        )
        test_players.team_id = rt.Cat(
            [
                "DAL",
                "CAR",
                "DET",
                "CHI",
                "AND",
                "AND",
                "CHI",
                "BUF",
                "COL",
                "AND",
                "DET",
                "CBS",
                "CAL",
                "CAL",
                "BOS",
                "CAL",
                "DAL",
                "DET",
                "BUF",
                "CBS",
                "BUF",
                "CAR",
                "CAR",
                "BOS",
                "DAL",
                "CBS",
                "AND",
                "BOS",
                "BOS",
                "COL",
                "CAR",
                "CBS",
            ]
        )
        test_players.position = rt.Cat(["G"]).tile(32)

        return TestDataset.HockeyDataset(
            test_teams,
            test_players,
            test_teams.team_id.grouping,
            test_players.team_id.grouping,
            'team_id',
        )

    @staticmethod
    def sql_semantics() -> Tuple[rt.Dataset, rt.Dataset]:
        inv = rt.int32.inv

        foo = rt.Dataset()
        foo.id = rt.FA([1, 2, 3, 4, 5, 6], dtype=np.int32)
        foo.col1 = rt.FA([5, 5, 8, inv, 10, inv], dtype=np.int32)
        foo.col2 = rt.FA([inv, 5, inv, 1, 1, 4], dtype=np.int32)

        bar = rt.Dataset()
        bar.id = rt.FA([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
        bar.col1 = rt.FA([10, 10, 8, inv, inv, inv, inv, 5, 5], dtype=np.int32)
        bar.col2 = rt.FA([4, inv, inv, 3, inv, inv, 1, 5, inv], dtype=np.int32)
        bar.strcol = rt.FA(
            [
                b'Chestnut',
                b'Pine',
                b'Walnut',
                b'Locust',
                b'Cherry',
                b'Spruce',
                b'Cypress',
                b'Lombard',
                b'Sansom'
            ]
        )

        return foo, bar

    # TODO: This data covers a few scenarios not covered by the original data. merge2_sql_semantics tests
    #       should be migrated to use this data instead to make the tests more robust.
    @staticmethod
    def sql_semantics2() -> Tuple[rt.Dataset, rt.Dataset]:
        inv = rt.int32.inv

        foo = rt.Dataset()
        foo.id = rt.FA([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        foo.col1 = rt.FA([5, 5, 8, inv, 10, inv, -1, 11], dtype=np.int32)
        foo.col2 = rt.FA([inv, 5, inv, 1, 1, 4, 22, 9], dtype=np.int32)
        foo.team_name = rt.FA(
            [
                'Phillies',
                'Eagles',
                '76ers',
                'Flyers',
                'Union',
                'Wings',
                'Fusion',
                'Fight'
            ]
        )

        bar = rt.Dataset()
        bar.id = rt.FA([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int32)
        bar.col1 = rt.FA([10, 10, 8, inv, inv, inv, inv, 5, 5, 14, 5, -15], dtype=np.int32)
        bar.col2 = rt.FA([4, inv, inv, 3, inv, inv, 1, 5, inv, 9, 5, 13], dtype=np.int32)
        bar.strcol = rt.FA(
            [
                b'Chestnut',
                b'Pine',
                b'Walnut',
                b'Locust',
                b'Cherry',
                b'Spruce',
                b'Cypress',
                b'Lombard',
                b'Sansom',
                b'Market',
                b'Arch',
                b'Vine'
            ]
        )

        return foo, bar

    @staticmethod
    def sql_semantics_data(on_col_name: str) -> HockeyDataset:
        foo, bar = TestDataset.sql_semantics()

        foo_grouping = rt.Grouping(foo[on_col_name])
        bar_grouping = rt.Grouping(bar[on_col_name])

        return TestDataset.HockeyDataset(
            foo, bar, foo_grouping, bar_grouping, on_col_name
        )


@pytest.mark.parametrize(
    "left,right,left_grp,right_grp,on",
    [pytest.param(*TestDataset.hockey(), id="hockey")],
)
@pytest.mark.parametrize(
    "how", [
        'left', 'right', 'inner',
        pytest.param('outer', marks=xfail_rip260_outermerge_left_keep)
    ]
)
@pytest.mark.parametrize("keep", [None, pytest.param('first'), pytest.param('last')])
def test_merge2_result_matches_computed_fancyindex(
    left: rt.Dataset,
    right: rt.Dataset,
    left_grp: rt.Grouping,
    right_grp: rt.Grouping,
    on,
    how: str,
    keep: Optional[str],
):
    """
    Verify the output (merged) Dataset returned by merge2 has the same
    number of rows as the computed fancy indexes for the join.
    """
    from riptable.rt_merge import JoinIndices, _create_merge_fancy_indices, _normalize_keep

    # Test #1: Calculate the fancy-indices for the join.
    _keep = _normalize_keep(keep)
    join_indices = _create_merge_fancy_indices(
        left_grp, right_grp, None, how, _keep
    )

    # The fancy indices must be the same length so that when they're used
    # to transform the columns from the left and right Datasets, all the
    # transformed columns have the same length.
    assert JoinIndices.result_rowcount(join_indices.left_index, left.get_nrows()) \
        == JoinIndices.result_rowcount(join_indices.right_index, right.get_nrows())

    # Test #2: Call the merge function with the datasets.
    merge_result = rt.merge2(
        left, right, on=on, how=how, keep=keep, suffixes=('_x', '_y')
    )

    # Does the merged Dataset have the correct number of rows?
    assert len(merge_result) == JoinIndices.result_rowcount(join_indices.left_index, left.get_nrows())

    # Test #3: Try merging again, this time using 'normal' columns created using .expand_array on the Categoricals.
    expanded_on = f'{on}_expanded'
    left[expanded_on] = left[on].expand_array
    right[expanded_on] = right[on].expand_array

    expand_merge_result = rt.merge2(
        left, right, on=expanded_on, how=how, keep=keep, suffixes=('_x', '_y')
    )

    # Does the merged Dataset have the same number of rows as the original result?
    assert len(expand_merge_result) == len(merge_result)


@pytest.mark.parametrize(
    "left,right,left_grp,right_grp,on",
    [
        pytest.param(*TestDataset.hockey(), id="hockey"),
        pytest.param(
            *TestDataset.sql_semantics_data('col1'), id="sql_semantics_singlekey"
        ),
    ],
)
@pytest.mark.parametrize(
    "how", [
        'left', 'right', 'inner',
        pytest.param(
            'outer',
            marks=pytest.mark.xfail(
                reason="RIP-260: Outer merge handling of 'keep' for the left Dataset is broken. This test also needs "
                       "to be fixed for outer merge, since the bounds on how many rows it can return are different "
                       "than the other merge types."
            )
        )
    ]
)
@pytest.mark.parametrize("keep", [None, pytest.param('first'), pytest.param('last')])
def test_merge_create_fancy_indices_with_invalids(
    left: rt.Dataset,
    right: rt.Dataset,
    left_grp: rt.Grouping,
    right_grp: rt.Grouping,
    on,
    how: str,
    keep: Optional[str],
):
    """
    Verify the output (merged) Dataset returned by merge2 has the same
    number of rows as the computed fancy indexes for the join.
    """
    from riptable.rt_merge import (
        JoinIndices,
        _create_merge_fancy_indices,
        _create_column_valid_mask,
        _normalize_keep,
    )

    # Assert preconditions/assumptions for this test.
    assert (
        left_grp.ncountgroup[0] == 0
    )  # left Grouping should have no invalids in the underlying cols it was created from
    assert (
        right_grp.ncountgroup[0] == 0
    )  # right Grouping should have no invalids in the underlying cols it was created from

    # Calculate the fancy-indices for the join. These are used as a baseline for comparison below.
    _keep = _normalize_keep(keep)
    join_indices = _create_merge_fancy_indices(
        # TODO: Need to create and pass the 'right_groupby_keygroup' here for 'outer' when keep='first'/'last'.
        left_grp, right_grp, None, how, _keep
    )

    # The fancy indices must be the same length so that when they're used
    # to transform the columns from the left and right Datasets, all the
    # transformed columns have the same length.
    left_fancyindex_len = JoinIndices.result_rowcount(join_indices.left_index, left.get_nrows())
    assert left_fancyindex_len == JoinIndices.result_rowcount(join_indices.right_index, right.get_nrows())

    left_on = left[on]
    right_on = right[on]

    # Test #1: Filter every other row in both Categoricals. Use this as a rough bounds-check by comparing
    # the number of rows in this result to the original merge_result -- it should have at most the same
    # number of rows as the original.
    left_filt = (np.arange(len(left_on)) % 2).astype(np.bool)
    right_filt = (np.arange(len(right_on)) % 2).astype(np.bool)

    filtered_on = f'{on}_filtered'

    left[filtered_on] = (
        left_on.filter(left_filt)
        if isinstance(left_on, rt.Categorical)
        else rt.where(left_filt, left_on, rt.INVALID_DICT[left_on.dtype.num])
    )
    right[filtered_on] = (
        right_on.filter(right_filt)
        if isinstance(right_on, rt.Categorical)
        else rt.where(right_filt, right_on, rt.INVALID_DICT[right_on.dtype.num])
    )

    left_filt_grp = (
        left[filtered_on].grouping
        if isinstance(left_on, rt.Categorical)
        else rt.Grouping(
            left[filtered_on], filter=_create_column_valid_mask(left[filtered_on])
        )
    )
    right_filt_grp = (
        right[filtered_on].grouping
        if isinstance(right_on, rt.Categorical)
        else rt.Grouping(
            right[filtered_on], filter=_create_column_valid_mask(right[filtered_on])
        )
    )

    def create_and_verify_join_indices(
        left_grp, right_grp, how: str, keep, baseline_length: int, strict: bool
    ) -> int:
        filt_join_indices = _create_merge_fancy_indices(
            # TODO: Need to create and pass the 'right_groupby_keygroup' here for 'outer' when keep='first'/'last'.
            left_grp, right_grp, None, how, keep
        )

        # The returned fancy indices should ALWAYS produce the same output length (no matter what).
        # They can be different lengths if one is an integer fancy index and the other is a boolean mask,
        # as long as they'd produce the same output length when applied.
        left_filt_fancyindex_len = JoinIndices.result_rowcount(filt_join_indices.left_index, len(left_grp.ikey))
        assert left_filt_fancyindex_len == JoinIndices.result_rowcount(
            filt_join_indices.right_index, len(right_grp.ikey)
        )

        # The fancy indices created for the join where the left and/or right key column(s)
        # contained one or more invalids should have a length less than or equal to the length
        # of the fancy indices we created as a baseline (from the unfiltered columns).
        if strict:
            assert left_filt_fancyindex_len < baseline_length
        else:
            assert left_filt_fancyindex_len <= baseline_length

        return left_filt_fancyindex_len

    # Test 1.1: Filtered left, unfiltered right.
    strict_check = False
    filt_unfilt_len = create_and_verify_join_indices(
        left_filt_grp, right_grp, how, _keep, left_fancyindex_len, strict_check
    )

    # Test 1.2: Unfiltered left, filtered right.
    unfilt_filt_len = create_and_verify_join_indices(
        left_grp, right_filt_grp, how, _keep, left_fancyindex_len, strict_check
    )

    # Test 1.3: Filtered left, filtered right.
    filt_filt_len = create_and_verify_join_indices(
        left_filt_grp, right_filt_grp, how, _keep, left_fancyindex_len, strict_check
    )

    # Test 1.3.1: With both columns filtered, the length of the join indices should be less than or equal to
    # the length of the join indices created when just one of the keys was filtered.
    # The bounds are different for how='outer' because an outer join can actually have *more* results when
    # data is filtered out.
    if how == 'outer':
        # TODO: Need to determine bounds for outer joins.
        pytest.skip("Need to determine bounds for outer joins.")
        #assert filt_filt_len <= filt_unfilt_len
        #assert filt_filt_len <= unfilt_filt_len
    else:
        assert filt_filt_len <= filt_unfilt_len
        assert filt_filt_len <= unfilt_filt_len


@pytest.mark.parametrize(
    "left,right,left_grp,right_grp,on",
    [
        pytest.param(*TestDataset.hockey(), id="hockey"),
        pytest.param(
            *TestDataset.sql_semantics_data('col1'), id="sql_semantics_singlekey"
        ),
    ],
)
@pytest.mark.parametrize(
    "how", [
        'left', 'right', 'inner',
        pytest.param('outer', marks=xfail_rip260_outermerge_left_keep)
    ]
)
@pytest.mark.parametrize("keep", [None, pytest.param('first'), pytest.param('last')])
def test_merge2_handles_invalids(
    left: rt.Dataset,
    right: rt.Dataset,
    left_grp: rt.Grouping,
    right_grp: rt.Grouping,
    on,
    how: str,
    keep: Optional[str],
):
    """
    Verify the output (merged) Dataset returned by merge2 has the same
    number of rows as the computed fancy indexes for the join.
    """
    # Call the merge function with the Datasets to give us a baseline for comparison.
    merge_result = rt.merge2(
        left, right, on=on, how=how, keep=keep, suffixes=('_x', '_y')
    )

    left_on = left[on]
    right_on = right[on]

    # Test #1: Filter every other row in both Categoricals. We use this as a rough bounds-check by comparing
    # the number of rows in this result to the original merge_result -- it should have at most the same
    # number of rows as the original.
    left_filt = (np.arange(len(left_on)) % 2).astype(np.bool)
    right_filt = (np.arange(len(right_on)) % 2).astype(np.bool)

    filtered_on = f'{on}_filtered'

    left[filtered_on] = (
        left_on.filter(left_filt)
        if isinstance(left_on, rt.Categorical)
        else rt.where(left_filt, left_on, rt.INVALID_DICT[left_on.dtype.num])
    )
    right[filtered_on] = (
        right_on.filter(right_filt)
        if isinstance(right_on, rt.Categorical)
        else rt.where(right_filt, right_on, rt.INVALID_DICT[right_on.dtype.num])
    )

    filt_left_merge_result = rt.merge2(
        left,
        right,
        left_on=filtered_on,
        right_on=on,
        how=how,
        keep=keep,
        suffixes=('_x', '_y'),
    )
    filt_right_merge_result = rt.merge2(
        left,
        right,
        left_on=on,
        right_on=filtered_on,
        how=how,
        keep=keep,
        suffixes=('_x', '_y'),
    )
    filt_both_merge_result = rt.merge2(
        left, right, on=filtered_on, how=how, keep=keep, suffixes=('_x', '_y')
    )

    # 'outer' merge follows some different rules so needs to be handled specially.
    # For example, filtering out rows (where they'll be assigned the invalid/NA key)
    # can actually _increase_ the number of rows in the output.
    if how == 'outer':
        # TODO: Need to determine bounds for outer joins -- they're different than for other join types;
        #       e.g. in the case where a key exists in exactly one row in the left and right Datasets,
        #       filtering out either or both of those rows (in the
        pytest.skip("Need to determine bounds for outer joins.")
        #assert len(filt_left_merge_result) <= len(merge_result)
        #assert len(filt_right_merge_result) <= len(merge_result)
        #assert len(filt_both_merge_result) <= len(filt_left_merge_result)
        #assert len(filt_both_merge_result) <= len(filt_right_merge_result)

    else:
        assert len(filt_left_merge_result) <= len(merge_result)
        assert len(filt_right_merge_result) <= len(merge_result)
        assert len(filt_both_merge_result) <= len(filt_left_merge_result)
        assert len(filt_both_merge_result) <= len(filt_right_merge_result)


def test_merge2_multikey_join_with_cats():
    """
    Test how merge2 operates on multi-key-column joins with one or more Categorical columns.
    """
    left_ds = rt.Dataset()
    left_ds['InkColor'] = rt.Cat(['Cyan', 'Magenta', 'Yellow', 'Black', 'Magenta', 'Cyan', 'Black', 'Yellow'])
    left_ds['CartridgeInstallDate'] = rt.Date(['2019-06-19', '2019-06-19', '2020-01-15', '2020-05-22', '2020-02-10', '2020-02-10', '2020-03-17', '2020-03-17'])

    right_ds = rt.Dataset()
    right_ds['InkColor'] = rt.Cat(['Cyan', 'Magenta', 'Yellow', 'Black']).tile(4)
    right_ds['PurchaseDate'] = rt.Cat(rt.Date(np.repeat(np.array(['2019-06-19', '2020-02-10', '2020-03-17', '2019-12-01']), repeats=4)))

    result = rt.merge2(left_ds, right_ds, on=['InkColor', ('CartridgeInstallDate', 'PurchaseDate')], suffixes=('_installed', '_purchased'))
    assert len(result) == 8


def test_merge2_nonempty_symmetric_diff():
    ds1 = rt.Dataset({'a': [1, 2, 3]})
    ds2 = rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]})
    ds = rt.merge2(ds1, ds2, on='a', how='left')
    assert_array_equal(ds['a'], rt.FastArray([1, 2, 3]))
    expected_b_arr = rt.FastArray([10, 0, 30])
    expected_b_arr[1] = rt.nan
    assert_array_equal(ds['b'], expected_b_arr)


@pytest.mark.parametrize("copy", [False, True])
def test_merge2_copy_data_sharing(copy):
    """Test how the 'copy' parameter of merge2 influences data-sharing between the input and output Datasets."""
    foo, bar = TestDataset.sql_semantics()

    # By definition, a left join/merge where we keep exactly one row per key
    # on the right side means the output columns from the left Dataset will be
    # exactly the same as the corresponding input columns.
    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='left',
        suffixes=('_foo', '_bar'),
        copy=copy,
        keep=(None, 'first'),
    )

    # The input and output columns from the left Dataset should overlap
    # (physically, in memory) iff (if and only if) 'copy=False'.
    left_inout_colnames = set(result.keys()) & set(foo.keys())
    for col_name in left_inout_colnames:
        if copy:
            assert not np.shares_memory(foo[col_name], result[col_name])
        else:
            assert np.shares_memory(foo[col_name], result[col_name])


@pytest.mark.parametrize(
    "left,right,on,validate,keep,should_succeed",
    [
        # Validate = m:m, Keep = None
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            'm:m',
            None,
            True,
            id="validate=m:m;keep=None;key=single;invalids=no;dups=None",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 3], 'b': [10, 30, 40]}),
            'a',
            'm:m',
            None,
            True,
            id="validate=m:m;keep=None;key=single;invalids=no;dups=right",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            'm:m',
            None,
            True,
            id="validate=m:m;keep=None;key=single;invalids=no;dups=left",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [2, 4, 4], 'b': [10, 30, 40]}),
            'a',
            'm:m',
            None,
            True,
            id="validate=m:m;keep=None;key=single;invalids=no;dups=both",
        ),
        # Validate = 1:m, Keep = None
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            '1:m',
            None,
            True,
            id="validate=1:m;keep=None;key=single;invalids=no;dups=None",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 3], 'b': [10, 30, 40]}),
            'a',
            '1:m',
            None,
            True,
            id="validate=1:m;keep=None;key=single;invalids=no;dups=right",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            '1:m',
            None,
            False,
            id="validate=1:m;keep=None;key=single;invalids=no;dups=left",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [2, 4, 4], 'b': [10, 30, 40]}),
            'a',
            '1:m',
            None,
            False,
            id="validate=1:m;keep=None;key=single;invalids=no;dups=both",
        ),
        # Validate = m:1, Keep = None
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            'm:1',
            None,
            True,
            id="validate=m:1;keep=None;key=single;invalids=no;dups=None",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 3], 'b': [10, 30, 40]}),
            'a',
            'm:1',
            None,
            False,
            id="validate=m:1;keep=None;key=single;invalids=no;dups=right",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            'm:1',
            None,
            True,
            id="validate=m:1;keep=None;key=single;invalids=no;dups=left",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [2, 4, 4], 'b': [10, 30, 40]}),
            'a',
            'm:1',
            None,
            False,
            id="validate=m:1;keep=None;key=single;invalids=no;dups=both",
        ),
        # Validate = 1:1, Keep = None
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            '1:1',
            None,
            True,
            id="validate=1:1;keep=None;key=single;invalids=no;dups=None",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 3], 'b': [10, 30, 40]}),
            'a',
            '1:1',
            None,
            False,
            id="validate=1:1;keep=None;key=single;invalids=no;dups=right",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            '1:1',
            None,
            False,
            id="validate=1:1;keep=None;key=single;invalids=no;dups=left",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [2, 4, 4], 'b': [10, 30, 40]}),
            'a',
            '1:1',
            None,
            False,
            id="validate=1:1;keep=None;key=single;invalids=no;dups=both",
        ),
        # Validate = m:m, Keep = 'first'
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            'm:m',
            'first',
            True,
            id="validate=m:m;keep=first;key=single;invalids=no;dups=None",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 3], 'b': [10, 30, 40]}),
            'a',
            'm:m',
            'first',
            True,
            id="validate=m:m;keep=first;key=single;invalids=no;dups=right",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            'm:m',
            'first',
            True,
            id="validate=m:m;keep=first;key=single;invalids=no;dups=left",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [2, 4, 4], 'b': [10, 30, 40]}),
            'a',
            'm:m',
            'first',
            True,
            id="validate=m:m;keep=first;key=single;invalids=no;dups=both",
        ),
        # Validate = 1:m, Keep = 'first'
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            '1:m',
            'first',
            True,
            id="validate=1:m;keep=first;key=single;invalids=no;dups=None",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 3], 'b': [10, 30, 40]}),
            'a',
            '1:m',
            'first',
            True,
            id="validate=1:m;keep=first;key=single;invalids=no;dups=right",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            '1:m',
            'first',
            True,
            id="validate=1:m;keep=first;key=single;invalids=no;dups=left",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [2, 4, 4], 'b': [10, 30, 40]}),
            'a',
            '1:m',
            'first',
            True,
            id="validate=1:m;keep=first;key=single;invalids=no;dups=both",
        ),
        # Validate = m:1, Keep = 'first'
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            'm:1',
            'first',
            True,
            id="validate=m:1;keep=first;key=single;invalids=no;dups=None",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 3], 'b': [10, 30, 40]}),
            'a',
            'm:1',
            'first',
            True,
            id="validate=m:1;keep=first;key=single;invalids=no;dups=right",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            'm:1',
            'first',
            True,
            id="validate=m:1;keep=first;key=single;invalids=no;dups=left",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [2, 4, 4], 'b': [10, 30, 40]}),
            'a',
            'm:1',
            'first',
            True,
            id="validate=m:1;keep=first;key=single;invalids=no;dups=both",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [2, 4, 4], 'b': [10, 30, 40]}),
            'a',
            'm:1',
            (None, 'first'),  # Test the tuple-parsing logic for 'keep'
            True,
            id="validate=m:1;keep=None,first;key=single;invalids=no;dups=both",
        ),
        # Validate = 1:1, Keep = 'first'
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            '1:1',
            'first',
            True,
            id="validate=1:1;keep=first;key=single;invalids=no;dups=None",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 2, 3]}),
            rt.Dataset({'a': [1, 3, 3], 'b': [10, 30, 40]}),
            'a',
            '1:1',
            'first',
            True,
            id="validate=1:1;keep=first;key=single;invalids=no;dups=right",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [1, 3, 4], 'b': [10, 30, 40]}),
            'a',
            '1:1',
            'first',
            True,
            id="validate=1:1;keep=first;key=single;invalids=no;dups=left",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 1, 3]}),
            rt.Dataset({'a': [2, 4, 4], 'b': [10, 30, 40]}),
            'a',
            '1:1',
            'first',
            True,
            id="validate=1:1;keep=first;key=single;invalids=no;dups=both",
        ),
        pytest.param(
            rt.Dataset({'a': [1, 2, 3], 'b': [10, 30, 40], 'c': ['x', 'y', 'z'], 'd': [3.1, -0.3, 2.2]}),
            rt.Dataset({'x': [2, 4, 3], 'y': [30, 40, 50], 'z': ['', 'x', ''], 'w': [1.0, 1.1, 1.11]}),
            # Test parsing of the 'on' parameter when given a list with more than 3 elements,
            # and there's at least one tuple at index >=2. This makes sure the parser for the 'on'
            # parameter is checking the length of each tuple (if applicable) rather than the list length.
            [('a', 'x'), ('b', 'y'), ('c', 'z')],
            None,
            None,
            True,
            id="validate=None;keep=None;key=multi;invalids=yes;dups=None"
        )
        # TODO: Extend the test cases for merge2 w.r.t. the behavior of the 'validate' kwarg. This test should include
        #       cases for single-key, multi-key, single-key-with-invalids, multi-key-with-invalids. For each case, test both
        #       passing the key(s) as 'normal' columns and as Categoricals; for the multi-key cases, try passing:
        #         * a tuple/list of normal (non-Cat) columns
        #         * a single multi-key Categorical
        #         * a combination of normal (non-Cat) column(s) and single-key Cats (to verify the Grouping-merging code used by merge works correctly).
    ],
)
def test_merge2_validate(
    left: rt.Dataset, right: rt.Dataset, on, validate: str, keep, should_succeed: bool
):
    """
    Test cases for how the validation logic used by 'merge2' works.
    """

    # Call the validation function, checking for expected (and unexpected) failure.
    if should_succeed:
        rt.merge2(left, right, on=on, validate=validate, keep=keep)

    else:
        with pytest.raises(ValueError):
            rt.merge2(left, right, on=on, validate=validate, keep=keep)


def test_merge2_sql_semantics_leftjoin_single():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    left join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """
    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='left',
        suffixes=('_foo', '_bar'),
        indicator=True,
    )

    assert result.get_nrows() == 9

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(
        result.col1, rt.FA([5, 5, 5, 5, 8, inv, 10, 10, inv], dtype=np.int32)
    )

    # Cols from the left Dataset.
    assert_array_equal(
        result.id_foo, rt.FA([1, 1, 2, 2, 3, 4, 5, 5, 6], dtype=np.int32)
    )
    assert_array_equal(
        result.col2_foo, rt.FA([inv, inv, 5, 5, inv, 1, 1, 1, 4], dtype=np.int32)
    )

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([8, 9, 8, 9, 3, inv, 1, 2, inv], dtype=np.int32)
    )
    assert_array_equal(
        result.col2_bar, rt.FA([5, inv, 5, inv, inv, inv, 4, inv, inv], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol,
        rt.FA(
            [
                b'Lombard',
                b'Sansom',
                b'Lombard',
                b'Sansom',
                b'Walnut',
                b'',
                b'Chestnut',
                b'Pine',
                b'',
            ]
        ),
    )


def test_merge2_sql_semantics_rightjoin_single():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    right join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
    order by b.id, f.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='right',
        suffixes=('_foo', '_bar'),
        indicator=True,
    )

    assert result.get_nrows() == 11

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(
        result.col1, rt.FA([10, 10, 8, inv, inv, inv, inv, 5, 5, 5, 5], dtype=np.int32)
    )

    # Cols from the left Dataset.
    assert_array_equal(
        result.id_foo, rt.FA([5, 5, 3, inv, inv, inv, inv, 1, 2, 1, 2], dtype=np.int32)
    )
    assert_array_equal(
        result.col2_foo,
        rt.FA([1, 1, inv, inv, inv, inv, inv, inv, 5, inv, 5], dtype=np.int32),
    )

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9], dtype=np.int32)
    )
    assert_array_equal(
        result.col2_bar,
        rt.FA([4, inv, inv, 3, inv, inv, 1, 5, 5, inv, inv], dtype=np.int32),
    )
    assert_array_equal(
        result.strcol,
        rt.FA(
            [
                b'Chestnut',
                b'Pine',
                b'Walnut',
                b'Locust',
                b'Cherry',
                b'Spruce',
                b'Cypress',
                b'Lombard',
                b'Lombard',
                b'Sansom',
                b'Sansom',
            ]
        ),
    )


def test_merge2_sql_semantics_innerjoin_single():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    inner join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='inner',
        suffixes=('_foo', '_bar'),
        indicator=True,
    )

    assert result.get_nrows() == 7

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 5, 5, 8, 10, 10], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 1, 2, 2, 3, 5, 5], dtype=np.int32))
    assert_array_equal(
        result.col2_foo, rt.FA([inv, inv, 5, 5, inv, 1, 1], dtype=np.int32)
    )

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8, 9, 8, 9, 3, 1, 2], dtype=np.int32))
    assert_array_equal(
        result.col2_bar, rt.FA([5, inv, 5, inv, inv, 4, inv], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol,
        rt.FA(
            [
                b'Lombard',
                b'Sansom',
                b'Lombard',
                b'Sansom',
                b'Walnut',
                b'Chestnut',
                b'Pine',
            ]
        ),
    )


def test_merge2_sql_semantics_outerjoin_single():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        f.team_name as foo_teamname,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    full outer join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics2()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='outer',
        suffixes=('_foo', '_bar'),
        indicator=True,
    )

    assert result.get_nrows() == 19

    # "foo_id","foo_col1","foo_col2","foo_teamname","bar_id","bar_col1","bar_col2","bar_strcol"
    # 1,5,NULL,"Phillies",8,5,5,"Lombard"
    # 1,5,NULL,"Phillies",9,5,NULL,"Sansom"
    # 1,5,NULL,"Phillies",11,5,5,"Arch"
    # 2,5,5,"Eagles",8,5,5,"Lombard"
    # 2,5,5,"Eagles",9,5,NULL,"Sansom"
    # 2,5,5,"Eagles",11,5,5,"Arch"
    # 3,8,NULL,"76ers",3,8,NULL,"Walnut"
    # 4,NULL,1,"Flyers",NULL,NULL,NULL,NULL
    # 5,10,1,"Union",1,10,4,"Chestnut"
    # 5,10,1,"Union",2,10,NULL,"Pine"
    # 6,NULL,4,"Wings",NULL,NULL,NULL,NULL
    # 7,-1,22,"Fusion",NULL,NULL,NULL,NULL
    # 8,11,9,"Fight",NULL,NULL,NULL,NULL
    # NULL,NULL,NULL,NULL,4,NULL,3,"Locust"
    # NULL,NULL,NULL,NULL,5,NULL,NULL,"Cherry"
    # NULL,NULL,NULL,NULL,6,NULL,NULL,"Spruce"
    # NULL,NULL,NULL,NULL,7,NULL,1,"Cypress"
    # NULL,NULL,NULL,NULL,10,14,9,"Market"
    # NULL,NULL,NULL,NULL,12,-15,13,"Vine"

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 5, 5, 5, 5, 8, inv, 10, 10, inv, -1, 11, inv, inv, inv, inv, 14, -15], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 1, 1, 2, 2, 2, 3, 4, 5, 5, 6, 7, 8, inv, inv, inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(
        result.col2_foo, rt.FA([inv, inv, inv, 5, 5, 5, inv, 1, 1, 1, 4, 22, 9, inv, inv, inv, inv, inv, inv], dtype=np.int32)
    )
    assert_array_equal(
        result.team_name,
        rt.FA(
            [
                b'Phillies',
                b'Phillies',
                b'Phillies',
                b'Eagles',
                b'Eagles',
                b'Eagles',
                b'76ers',
                b'Flyers',
                b'Union',
                b'Union',
                b'Wings',
                b'Fusion',
                b'Fight',
                b'',
                b'',
                b'',
                b'',
                b'',
                b'',
            ]
        ),
    )

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8, 9, 11, 8, 9, 11, 3, inv, 1, 2, inv, inv, inv, 4, 5, 6, 7, 10, 12], dtype=np.int32))
    assert_array_equal(
        result.col2_bar, rt.FA([5, inv, 5, 5, inv, 5, inv, inv, 4, inv, inv, inv, inv, 3, inv, inv, 1, 9, 13], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol,
        rt.FA(
            [
                b'Lombard',
                b'Sansom',
                b'Arch',
                b'Lombard',
                b'Sansom',
                b'Arch',
                b'Walnut',
                b'',
                b'Chestnut',
                b'Pine',
                b'',
                b'',
                b'',
                b'Locust',
                b'Cherry',
                b'Spruce',
                b'Cypress',
                b'Market',
                b'Vine'
            ]
        ),
    )


def test_merge2_sql_semantics_leftjoin_single_keep_Nonefirst():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    left join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct FIRST_VALUE(id) over w as `first_row_id`
            from sql_semantics.bar
            window w as (partition by bar.col1 order by bar.id)
        ) as bar_ids
        on
            bar.id = bar_ids.first_row_id
    ) as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='left',
        suffixes=('_foo', '_bar'),
        keep=(None, 'first'),
        indicator=True,
    )

    assert result.get_nrows() == 6

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      1 |        5 |     NULL |      8 |        5 |        5 | Lombard    |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      4 |     NULL |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      5 |       10 |        1 |      1 |       10 |        4 | Chestnut   |
    # |      6 |     NULL |        4 |   NULL |     NULL |     NULL | NULL       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 6 rows in set (0.0009 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, inv, 10, inv], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 3, 4, 5, 6], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([inv, 5, inv, 1, 1, 4], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8, 8, 3, inv, 1, inv], dtype=np.int32))
    assert_array_equal(result.col2_bar, rt.FA([5, 5, inv, inv, 4, inv], dtype=np.int32))
    assert_array_equal(
        result.strcol, rt.FA([b'Lombard', b'Lombard', b'Walnut', b'', b'Chestnut', b''])
    )


def test_merge2_sql_semantics_leftjoin_single_keep_Nonelast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    left join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over w as `last_row_id`
            from sql_semantics.bar
            window w as (partition by bar.col1 order by bar.id desc)
        ) as bar_ids
        on
            bar.id = bar_ids.last_row_id
    ) as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='left',
        suffixes=('_foo', '_bar'),
        keep=(None, 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 6

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      1 |        5 |     NULL |      9 |        5 |     NULL | Sansom     |
    # |      2 |        5 |        5 |      9 |        5 |     NULL | Sansom     |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      4 |     NULL |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      5 |       10 |        1 |      2 |       10 |     NULL | Pine       |
    # |      6 |     NULL |        4 |   NULL |     NULL |     NULL | NULL       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 6 rows in set (0.0009 sec)

    print(repr(result))

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, inv, 10, inv], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 3, 4, 5, 6], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([inv, 5, inv, 1, 1, 4], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([9, 9, 3, inv, 2, inv], dtype=np.int32))
    assert_array_equal(
        result.col2_bar, rt.FA([inv, inv, inv, inv, inv, inv], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol, rt.FA([b'Sansom', b'Sansom', b'Walnut', b'', b'Pine', b''])
    )


def test_merge2_sql_semantics_rightjoin_single_keep_Nonefirst():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    right join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct FIRST_VALUE(id) over w as `first_row_id`
            from sql_semantics.bar
            window w as (partition by bar.col1 order by bar.id)
        ) as bar_ids
        on
            bar.id = bar_ids.first_row_id
    ) as b
    on
        f.col1 = b.col1
    order by b.id, f.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='right',
        suffixes=('_foo', '_bar'),
        keep=(None, 'first'),
        indicator=True,
    )

    assert result.get_nrows() == 5

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |   NULL |     NULL |     NULL |      4 |     NULL |        3 | Locust     |
    # |      1 |        5 |     NULL |      8 |        5 |        5 | Lombard    |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      5 |       10 |        1 |      1 |       10 |        4 | Chestnut   |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 5 rows in set (0.0010 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([10, 8, inv, 5, 5], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([5, 3, inv, 1, 2], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([1, inv, inv, inv, 5], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([1, 3, 4, 8, 8], dtype=np.int32))
    assert_array_equal(result.col2_bar, rt.FA([4, inv, 3, 5, 5], dtype=np.int32))
    assert_array_equal(
        result.strcol,
        rt.FA([b'Chestnut', b'Walnut', b'Locust', b'Lombard', b'Lombard']),
    )


def test_merge2_sql_semantics_rightjoin_single_keep_Nonelast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    right join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over w as `last_row_id`
            from sql_semantics.bar
            window w as (partition by bar.col1 order by bar.id desc)
        ) as bar_ids
        on
            bar.id = bar_ids.last_row_id
    ) as b
    on
        f.col1 = b.col1
    order by b.id, f.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='right',
        suffixes=('_foo', '_bar'),
        keep=(None, 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 5

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |   NULL |     NULL |     NULL |      7 |     NULL |        1 | Cypress    |
    # |      1 |        5 |     NULL |      9 |        5 |     NULL | Sansom     |
    # |      2 |        5 |        5 |      9 |        5 |     NULL | Sansom     |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      5 |       10 |        1 |      2 |       10 |     NULL | Pine       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 5 rows in set (0.0009 sec)

    print(repr(result))

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([10, 8, inv, 5, 5], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([5, 3, inv, 1, 2], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([1, inv, inv, inv, 5], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([2, 3, 7, 9, 9], dtype=np.int32))
    assert_array_equal(result.col2_bar, rt.FA([inv, inv, 1, inv, inv], dtype=np.int32))
    assert_array_equal(
        result.strcol, rt.FA([b'Pine', b'Walnut', b'Cypress', b'Sansom', b'Sansom'])
    )


def test_merge2_sql_semantics_innerjoin_single_keep_Nonefirst():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    inner join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct FIRST_VALUE(id) over w as `first_row_id`
            from sql_semantics.bar
            window w as (partition by bar.col1 order by bar.id)
        ) as bar_ids
        on
            bar.id = bar_ids.first_row_id
    ) as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='inner',
        suffixes=('_foo', '_bar'),
        keep=(None, 'first'),
        indicator=True,
    )

    assert result.get_nrows() == 4

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      1 |        5 |     NULL |      8 |        5 |        5 | Lombard    |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      5 |       10 |        1 |      1 |       10 |        4 | Chestnut   |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 4 rows in set (0.0010 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, 10], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 3, 5], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([inv, 5, inv, 1], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8, 8, 3, 1], dtype=np.int32))
    assert_array_equal(result.col2_bar, rt.FA([5, 5, inv, 4], dtype=np.int32))
    assert_array_equal(
        result.strcol, rt.FA([b'Lombard', b'Lombard', b'Walnut', b'Chestnut'])
    )


def test_merge2_sql_semantics_innerjoin_single_keep_Nonelast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    inner join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over w as `last_row_id`
            from sql_semantics.bar
            window w as (partition by bar.col1 order by bar.id desc)
        ) as bar_ids
        on
            bar.id = bar_ids.last_row_id
    ) as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='inner',
        suffixes=('_foo', '_bar'),
        keep=(None, 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 4

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      1 |        5 |     NULL |      9 |        5 |     NULL | Sansom     |
    # |      2 |        5 |        5 |      9 |        5 |     NULL | Sansom     |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      5 |       10 |        1 |      2 |       10 |     NULL | Pine       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 4 rows in set (0.0010 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, 10], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 3, 5], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([inv, 5, inv, 1], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([9, 9, 3, 2], dtype=np.int32))
    assert_array_equal(result.col2_bar, rt.FA([inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(result.strcol, rt.FA([b'Sansom', b'Sansom', b'Walnut', b'Pine']))


def test_merge2_sql_semantics_outerjoin_single_keep_Nonefirst():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        f.team_name as foo_teamname,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    full outer join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct FIRST_VALUE(id) over w as firstlast_row_id
            from sql_semantics.bar
            window w as (
                partition by bar.col1
                order by bar.id asc
            )
        ) as bar_ids
        on
            bar.id = bar_ids.firstlast_row_id
    ) as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics2()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='outer',
        suffixes=('_foo', '_bar'),
        keep=(None, 'first'),
        indicator=True,
    )

    assert result.get_nrows() == 11

    # "foo_id","foo_col1","foo_col2","foo_teamname","bar_id","bar_col1","bar_col2","bar_strcol"
    # 1,5,NULL,"Phillies",8,5,5,"Lombard"
    # 2,5,5,"Eagles",8,5,5,"Lombard"
    # 3,8,NULL,"76ers",3,8,NULL,"Walnut"
    # 4,NULL,1,"Flyers",NULL,NULL,NULL,NULL
    # 5,10,1,"Union",1,10,4,"Chestnut"
    # 6,NULL,4,"Wings",NULL,NULL,NULL,NULL
    # 7,-1,22,"Fusion",NULL,NULL,NULL,NULL
    # 8,11,9,"Fight",NULL,NULL,NULL,NULL
    # NULL,NULL,NULL,NULL,4,NULL,3,"Locust"
    # NULL,NULL,NULL,NULL,10,14,9,"Market"
    # NULL,NULL,NULL,NULL,12,-15,13,"Vine"

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, inv, 10, inv, -1, 11, inv, 14, -15], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 3, 4, 5, 6, 7, 8, inv, inv, inv], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([inv, 5, inv, 1, 1, 4, 22, 9, inv, inv, inv], dtype=np.int32))
    assert_array_equal(
        result.team_name, rt.FA([b'Phillies', b'Eagles', b'76ers', b'Flyers', b'Union', b'Wings', b'Fusion', b'Fight', b'', b'', b''])
    )

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8, 8, 3, inv, 1, inv, inv, inv, 4, 10, 12], dtype=np.int32))
    assert_array_equal(result.col2_bar, rt.FA([5, 5, inv, inv, 4, inv, inv, inv, 3, 9, 13], dtype=np.int32))
    assert_array_equal(
        result.strcol, rt.FA([b'Lombard', b'Lombard', b'Walnut', b'', b'Chestnut', b'', b'', b'', b'Locust', b'Market', b'Vine'])
    )


def test_merge2_sql_semantics_outerjoin_single_keep_Nonelast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        f.team_name as foo_teamname,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    full outer join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over w as firstlast_row_id
            from sql_semantics.bar
            window w as (
                partition by bar.col1
                order by bar.id
                rows between unbounded preceding and unbounded following
            )
        ) as bar_ids
        on
            bar.id = bar_ids.firstlast_row_id
    ) as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics2()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='outer',
        suffixes=('_foo', '_bar'),
        keep=(None, 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 11

    # "foo_id","foo_col1","foo_col2","foo_teamname","bar_id","bar_col1","bar_col2","bar_strcol"
    # 1,5,NULL,"Phillies",11,5,5,"Arch"
    # 2,5,5,"Eagles",11,5,5,"Arch"
    # 3,8,NULL,"76ers",3,8,NULL,"Walnut"
    # 4,NULL,1,"Flyers",NULL,NULL,NULL,NULL
    # 5,10,1,"Union",2,10,NULL,"Pine"
    # 6,NULL,4,"Wings",NULL,NULL,NULL,NULL
    # 7,-1,22,"Fusion",NULL,NULL,NULL,NULL
    # 8,11,9,"Fight",NULL,NULL,NULL,NULL
    # NULL,NULL,NULL,NULL,7,NULL,1,"Cypress"
    # NULL,NULL,NULL,NULL,10,14,9,"Market"
    # NULL,NULL,NULL,NULL,12,-15,13,"Vine"

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, inv, 10, inv, -1, 11, inv, 14, -15], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 3, 4, 5, 6, 7, 8, inv, inv, inv], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([inv, 5, inv, 1, 1, 4, 22, 9, inv, inv, inv], dtype=np.int32))
    assert_array_equal(result.team_name, rt.FA([b'Phillies', b'Eagles', b'76ers', b'Flyers', b'Union', b'Wings', b'Fusion', b'Fight', b'', b'', b'']))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([11, 11, 3, inv, 2, inv, inv, inv, 7, 10, 12], dtype=np.int32))
    assert_array_equal(result.col2_bar, rt.FA([5, 5, inv, inv, inv, inv, inv, inv, 1, 9, 13], dtype=np.int32))
    assert_array_equal(result.strcol, rt.FA([b'Arch', b'Arch', b'Walnut', b'', b'Pine', b'', b'', b'', b'Cypress', b'Market', b'Vine']))


def test_merge2_sql_semantics_leftjoin_single_keep_firstNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over w as `first_row_id`
            from sql_semantics.foo
            window w as (partition by foo.col1 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.first_row_id
    ) as f
    left join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='left',
        suffixes=('_foo', '_bar'),
        keep=('first', None),
        indicator=True,
    )

    assert result.get_nrows() == 6

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      1 |        5 |     NULL |      8 |        5 |        5 | Lombard    |
    # |      1 |        5 |     NULL |      9 |        5 |     NULL | Sansom     |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      4 |     NULL |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      5 |       10 |        1 |      1 |       10 |        4 | Chestnut   |
    # |      5 |       10 |        1 |      2 |       10 |     NULL | Pine       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 6 rows in set (0.0017 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, inv, 10, 10], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 1, 3, 4, 5, 5], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([inv, inv, inv, 1, 1, 1], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8, 9, 3, inv, 1, 2], dtype=np.int32))
    assert_array_equal(
        result.col2_bar, rt.FA([5, inv, inv, inv, 4, inv], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol,
        rt.FA([b'Lombard', b'Sansom', b'Walnut', b'', b'Chestnut', b'Pine']),
    )


@pytest.mark.xfail(
    reason="RIP-260: This test fails due to a difference in the ordering of the returned rows as compared to the test data. Need to verify the test data to determine which one is correct."
)
def test_merge2_sql_semantics_leftjoin_single_keep_lastNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct LAST_VALUE(id) over w as `last_row_id`
            from sql_semantics.foo
            window w as (partition by foo.col1 order by foo.id desc)
        ) as foo_ids
        on
            foo.id = foo_ids.last_row_id
    ) as f
    left join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='left',
        suffixes=('_foo', '_bar'),
        keep=('last', None),
        indicator=True,
    )

    assert result.get_nrows() == 6

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # |      2 |        5 |        5 |      9 |        5 |     NULL | Sansom     |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      5 |       10 |        1 |      1 |       10 |        4 | Chestnut   |
    # |      5 |       10 |        1 |      2 |       10 |     NULL | Pine       |
    # |      6 |     NULL |        4 |   NULL |     NULL |     NULL | NULL       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 6 rows in set (0.0006 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, 10, 10, inv], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([2, 2, 3, 5, 5, 6], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([5, 5, inv, 1, 1, 4], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8, 9, 3, 1, 2, inv], dtype=np.int32))
    assert_array_equal(
        result.col2_bar, rt.FA([5, inv, inv, 4, inv, inv], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol,
        rt.FA([b'Lombard', b'Sansom', b'Walnut', b'Chestnut', b'Pine', b'']),
    )


def test_merge2_sql_semantics_rightjoin_single_keep_firstNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over w as `first_row_id`
            from sql_semantics.foo
            window w as (partition by foo.col1 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.first_row_id
    ) as f
    right join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
    order by b.id, f.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='right',
        suffixes=('_foo', '_bar'),
        keep=('first', None),
        indicator=True,
    )

    assert result.get_nrows() == 9

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |   NULL |     NULL |     NULL |      4 |     NULL |        3 | Locust     |
    # |   NULL |     NULL |     NULL |      5 |     NULL |     NULL | Cherry     |
    # |   NULL |     NULL |     NULL |      6 |     NULL |     NULL | Spruce     |
    # |   NULL |     NULL |     NULL |      7 |     NULL |        1 | Cypress    |
    # |      1 |        5 |     NULL |      8 |        5 |        5 | Lombard    |
    # |      1 |        5 |     NULL |      9 |        5 |     NULL | Sansom     |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      5 |       10 |        1 |      1 |       10 |        4 | Chestnut   |
    # |      5 |       10 |        1 |      2 |       10 |     NULL | Pine       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 9 rows in set (0.0006 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(
        result.col1, rt.FA([10, 10, 8, inv, inv, inv, inv, 5, 5], dtype=np.int32)
    )

    # Cols from the left Dataset.
    assert_array_equal(
        result.id_foo, rt.FA([5, 5, 3, inv, inv, inv, inv, 1, 1], dtype=np.int32)
    )
    assert_array_equal(
        result.col2_foo,
        rt.FA([1, 1, inv, inv, inv, inv, inv, inv, inv], dtype=np.int32),
    )

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    )
    assert_array_equal(
        result.col2_bar, rt.FA([4, inv, inv, 3, inv, inv, 1, 5, inv], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol,
        rt.FA(
            [
                b'Chestnut',
                b'Pine',
                b'Walnut',
                b'Locust',
                b'Cherry',
                b'Spruce',
                b'Cypress',
                b'Lombard',
                b'Sansom',
            ]
        ),
    )


def test_merge2_sql_semantics_rightjoin_single_keep_lastNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct LAST_VALUE(id) over w as `last_row_id`
            from sql_semantics.foo
            window w as (partition by foo.col1 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.last_row_id
    ) as f
    right join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
    order by b.id, f.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='right',
        suffixes=('_foo', '_bar'),
        keep=('last', None),
        indicator=True,
    )

    assert result.get_nrows() == 9

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |   NULL |     NULL |     NULL |      4 |     NULL |        3 | Locust     |
    # |   NULL |     NULL |     NULL |      5 |     NULL |     NULL | Cherry     |
    # |   NULL |     NULL |     NULL |      6 |     NULL |     NULL | Spruce     |
    # |   NULL |     NULL |     NULL |      7 |     NULL |        1 | Cypress    |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # |      2 |        5 |        5 |      9 |        5 |     NULL | Sansom     |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      5 |       10 |        1 |      1 |       10 |        4 | Chestnut   |
    # |      5 |       10 |        1 |      2 |       10 |     NULL | Pine       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 9 rows in set (0.0006 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(
        result.col1, rt.FA([10, 10, 8, inv, inv, inv, inv, 5, 5], dtype=np.int32)
    )

    # Cols from the left Dataset.
    assert_array_equal(
        result.id_foo, rt.FA([5, 5, 3, inv, inv, inv, inv, 2, 2], dtype=np.int32)
    )
    assert_array_equal(
        result.col2_foo, rt.FA([1, 1, inv, inv, inv, inv, inv, 5, 5], dtype=np.int32)
    )

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    )
    assert_array_equal(
        result.col2_bar, rt.FA([4, inv, inv, 3, inv, inv, 1, 5, inv], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol,
        rt.FA(
            [
                b'Chestnut',
                b'Pine',
                b'Walnut',
                b'Locust',
                b'Cherry',
                b'Spruce',
                b'Cypress',
                b'Lombard',
                b'Sansom',
            ]
        ),
    )


def test_merge2_sql_semantics_innerjoin_single_keep_firstNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over w as `first_row_id`
            from sql_semantics.foo
            window w as (partition by foo.col1 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.first_row_id
    ) as f
    inner join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='inner',
        suffixes=('_foo', '_bar'),
        keep=('first', None),
        indicator=True,
    )

    assert result.get_nrows() == 5

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      1 |        5 |     NULL |      8 |        5 |        5 | Lombard    |
    # |      1 |        5 |     NULL |      9 |        5 |     NULL | Sansom     |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      5 |       10 |        1 |      1 |       10 |        4 | Chestnut   |
    # |      5 |       10 |        1 |      2 |       10 |     NULL | Pine       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 5 rows in set (0.0010 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, 10, 10], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 1, 3, 5, 5], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([inv, inv, inv, 1, 1], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8, 9, 3, 1, 2], dtype=np.int32))
    assert_array_equal(result.col2_bar, rt.FA([5, inv, inv, 4, inv], dtype=np.int32))
    assert_array_equal(
        result.strcol, rt.FA([b'Lombard', b'Sansom', b'Walnut', b'Chestnut', b'Pine'])
    )


def test_merge2_sql_semantics_innerjoin_single_keep_lastNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct LAST_VALUE(id) over w as `last_row_id`
            from sql_semantics.foo
            window w as (partition by foo.col1 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.last_row_id
    ) as f
    inner join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='inner',
        suffixes=('_foo', '_bar'),
        keep=('last', None),
        indicator=True,
    )

    assert result.get_nrows() == 5

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # |      2 |        5 |        5 |      9 |        5 |     NULL | Sansom     |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      5 |       10 |        1 |      1 |       10 |        4 | Chestnut   |
    # |      5 |       10 |        1 |      2 |       10 |     NULL | Pine       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 5 rows in set (0.0011 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, 10, 10], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([2, 2, 3, 5, 5], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([5, 5, inv, 1, 1], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8, 9, 3, 1, 2], dtype=np.int32))
    assert_array_equal(result.col2_bar, rt.FA([5, inv, inv, 4, inv], dtype=np.int32))
    assert_array_equal(
        result.strcol, rt.FA([b'Lombard', b'Sansom', b'Walnut', b'Chestnut', b'Pine'])
    )


@xfail_rip260_outermerge_left_keep
def test_merge2_sql_semantics_outerjoin_single_keep_firstNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        f.team_name as foo_teamname,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over w as `first_row_id`
            from sql_semantics.foo
            window w as (partition by foo.col1 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.first_row_id
    ) as f
    full outer join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics2()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='outer',
        suffixes=('_foo', '_bar'),
        keep=('first', None),
        indicator=True,
    )

    assert result.get_nrows() == 15

    # "foo_id","foo_col1","foo_col2","foo_teamname","bar_id","bar_col1","bar_col2","bar_strcol"
    # 1,5,NULL,"Phillies",8,5,5,"Lombard"
    # 1,5,NULL,"Phillies",9,5,NULL,"Sansom"
    # 1,5,NULL,"Phillies",11,5,5,"Arch"
    # 3,8,NULL,"76ers",3,8,NULL,"Walnut"
    # 4,NULL,1,"Flyers",NULL,NULL,NULL,NULL
    # 5,10,1,"Union",1,10,4,"Chestnut"
    # 5,10,1,"Union",2,10,NULL,"Pine"
    # 7,-1,22,"Fusion",NULL,NULL,NULL,NULL
    # 8,11,9,"Fight",NULL,NULL,NULL,NULL
    # NULL,NULL,NULL,NULL,4,NULL,3,"Locust"
    # NULL,NULL,NULL,NULL,5,NULL,NULL,"Cherry"
    # NULL,NULL,NULL,NULL,6,NULL,NULL,"Spruce"
    # NULL,NULL,NULL,NULL,7,NULL,1,"Cypress"
    # NULL,NULL,NULL,NULL,10,14,9,"Market"
    # NULL,NULL,NULL,NULL,12,-15,13,"Vine"

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 5, 8, inv, 10, 10, -1, 11, inv, inv, inv, inv, 14, -15], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 1, 1, 3, 4, 5, 5, 7, 8, inv, inv, inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([inv, inv, inv, inv, 1, 1, 1, 22, 9, inv, inv, inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(
        result.team_name, rt.FA([b'Phillies', b'Phillies', b'Phillies', b'76ers', b'Flyers', b'Union', b'Union', b'Fusion', b'Fight', b'', b'', b'', b'', b'', b''])
    )

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8, 9, 11, 3, inv, 1, 2, inv, inv, 4, 5, 6, 7, 10, 12], dtype=np.int32))
    assert_array_equal(result.col2_bar, rt.FA([5, 5, 5, 8, inv, 10, 10, inv, inv, inv, inv, inv, inv, 14, -15], dtype=np.int32))
    assert_array_equal(
        result.strcol, rt.FA([b'Lombard', b'Sansom', b'Arch', b'Walnut', b'', b'Chestnut', b'Pine', b'', b'', b'Locust', b'Cherry', b'Spruce', b'Cypress', b'Market', b'Vine'])
    )


@xfail_rip260_outermerge_left_keep
def test_merge2_sql_semantics_outerjoin_single_keep_lastNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        f.team_name as foo_teamname,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct LAST_VALUE(id) over w as last_row_id
            from sql_semantics.foo
            window w as (
                partition by foo.col1
                order by foo.id
                rows between unbounded preceding and unbounded following
            )
        ) as foo_ids
        on
            foo.id = foo_ids.last_row_id
    ) as f
    full outer join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics2()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='outer',
        suffixes=('_foo', '_bar'),
        keep=('last', None),
        indicator=True,
    )

    assert result.get_nrows() == 19

    # "foo_id","foo_col1","foo_col2","foo_teamname","bar_id","bar_col1","bar_col2","bar_strcol"
    # 1,5,NULL,"Phillies",8,5,5,"Lombard"
    # 1,5,NULL,"Phillies",9,5,NULL,"Sansom"
    # 1,5,NULL,"Phillies",11,5,5,"Arch"
    # 2,5,5,"Eagles",8,5,5,"Lombard"
    # 2,5,5,"Eagles",9,5,NULL,"Sansom"
    # 2,5,5,"Eagles",11,5,5,"Arch"
    # 3,8,NULL,"76ers",3,8,NULL,"Walnut"
    # 4,NULL,1,"Flyers",NULL,NULL,NULL,NULL
    # 5,10,1,"Union",1,10,4,"Chestnut"
    # 5,10,1,"Union",2,10,NULL,"Pine"
    # 6,NULL,4,"Wings",NULL,NULL,NULL,NULL
    # 7,-1,22,"Fusion",NULL,NULL,NULL,NULL
    # 8,11,9,"Fight",NULL,NULL,NULL,NULL
    # NULL,NULL,NULL,NULL,4,NULL,3,"Locust"
    # NULL,NULL,NULL,NULL,5,NULL,NULL,"Cherry"
    # NULL,NULL,NULL,NULL,6,NULL,NULL,"Spruce"
    # NULL,NULL,NULL,NULL,7,NULL,1,"Cypress"
    # NULL,NULL,NULL,NULL,10,14,9,"Market"
    # NULL,NULL,NULL,NULL,12,-15,13,"Vine"

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 5, 5, 5, 5, 8, inv, 10, 10, inv, -1, 11, inv, inv, inv, inv, 14, -15], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 1, 1, 2, 2, 2, 3, 4, 5, 5, 6, 7, 8, inv, inv, inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([inv, inv, inv, 5, 5, 5, inv, 1, 1, 1, 4, 22, 9, inv, inv, inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(
        result.team_name, rt.FA([b'Phillies', b'Phillies', b'Phillies', b'Eagles', b'Eagles', b'Eagles', b'76ers', b'Flyers', b'Union', b'Union', b'Wings', b'Fusion', b'Fight', b'', b'', b'', b'', b'', b''])
    )

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8, 9, 11, 8, 9, 11, 3, inv, 1, 2, inv, inv, inv, 4, 5, 6, 7, 10, 12], dtype=np.int32))
    assert_array_equal(result.col2_bar, rt.FA([5, inv, 5, 5, inv, 5, inv, inv, 4, inv, inv, inv, inv, 3, inv, inv, 1, 9, 13], dtype=np.int32))
    assert_array_equal(
        result.strcol, rt.FA([b'Lombard', b'Sansom', b'Arch', b'Lombard', b'Sansom', b'Arch', b'Walnut', b'', b'Chestnut', b'Pine', b'', b'', b'', b'Locust', b'Cherry', b'Spruce', b'Cypress', b'Market', b'Vine'])
    )


def test_merge2_sql_semantics_leftjoin_single_keep_firstlast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over wf as `first_row_id`
            from sql_semantics.foo
            window wf as (partition by foo.col1 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.first_row_id
    ) as f
    left join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over wb as `last_row_id`
            from sql_semantics.bar
            window wb as (partition by bar.col1 order by bar.id)
        ) as bar_ids
        on
            bar.id = bar_ids.last_row_id
    ) as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='left',
        suffixes=('_foo', '_bar'),
        keep=('first', 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 4

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      1 |        5 |     NULL |      9 |        5 |     NULL | Sansom     |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      5 |       10 |        1 |      2 |       10 |     NULL | Pine       |
    # |      6 |     NULL |        4 |   NULL |     NULL |     NULL | NULL       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 4 rows in set (0.0016 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 8, inv, 10], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 3, 4, 5], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([inv, inv, 1, 1], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([9, 3, inv, 2], dtype=np.int32))
    assert_array_equal(result.col2_bar, rt.FA([inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(result.strcol, rt.FA([b'Sansom', b'Walnut', b'', b'Pine']))


def test_merge2_sql_semantics_rightjoin_single_keep_firstlast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over wf as `first_row_id`
            from sql_semantics.foo
            window wf as (partition by foo.col1 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.first_row_id
    ) as f
    right join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over wb as `last_row_id`
            from sql_semantics.bar
            window wb as (partition by bar.col1 order by bar.id)
        ) as bar_ids
        on
            bar.id = bar_ids.last_row_id
    ) as b
    on
        f.col1 = b.col1
    order by b.id, f.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='right',
        suffixes=('_foo', '_bar'),
        keep=('first', 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 4

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |   NULL |     NULL |     NULL |      4 |     NULL |        3 | Locust     |
    # |      1 |        5 |     NULL |      9 |        5 |     NULL | Sansom     |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      5 |       10 |        1 |      2 |       10 |     NULL | Pine       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 4 rows in set (0.0011 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([10, 8, inv, 5], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([5, 3, inv, 1], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([1, inv, inv, inv], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([2, 3, 7, 9], dtype=np.int32))
    assert_array_equal(result.col2_bar, rt.FA([inv, inv, 1, inv], dtype=np.int32))
    assert_array_equal(
        result.strcol, rt.FA([b'Pine', b'Walnut', b'Cypress', b'Sansom'])
    )


def test_merge2_sql_semantics_innerjoin_single_keep_firstlast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over wf as `first_row_id`
            from sql_semantics.foo
            window wf as (partition by foo.col1 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.first_row_id
    ) as f
    inner join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over wb as `last_row_id`
            from sql_semantics.bar
            window wb as (partition by bar.col1 order by bar.id)
        ) as bar_ids
        on
            bar.id = bar_ids.last_row_id
    ) as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='inner',
        suffixes=('_foo', '_bar'),
        keep=('first', 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 3

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      1 |        5 |     NULL |      9 |        5 |     NULL | Sansom     |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      5 |       10 |        1 |      2 |       10 |     NULL | Pine       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 3 rows in set (0.0011 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 8, 10], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 3, 5], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([inv, inv, 1], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([9, 3, 2], dtype=np.int32))
    assert_array_equal(result.col2_bar, rt.FA([inv, inv, inv], dtype=np.int32))
    assert_array_equal(result.strcol, rt.FA([b'Sansom', b'Walnut', b'Pine']))


@xfail_rip260_outermerge_left_keep
def test_merge2_sql_semantics_outerjoin_single_keep_firstlast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        f.team_name as foo_teamname,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over wf as firstlast_row_id
            from sql_semantics.foo
            window wf as (partition by foo.col1 order by foo.id asc)
        ) as foo_ids
        on
            foo.id = foo_ids.firstlast_row_id
    ) as f
    full outer join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over wb as firstlast_row_id
            from sql_semantics.bar
            window wb as (
                partition by bar.col1
                order by bar.id
                rows between unbounded preceding and unbounded following
            )
        ) as bar_ids
        on
            bar.id = bar_ids.firstlast_row_id
    ) as b
    on
        f.col1 = b.col1
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics2()

    result = rt.merge2(
        foo,
        bar,
        on=('col1', 'col1'),
        how='outer',
        suffixes=('_foo', '_bar'),
        keep=('first', 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 15

    # "foo_id","foo_col1","foo_col2","foo_teamname","bar_id","bar_col1","bar_col2","bar_strcol"
    # 1,5,NULL,"Phillies",8,5,5,"Lombard"
    # 1,5,NULL,"Phillies",9,5,NULL,"Sansom"
    # 1,5,NULL,"Phillies",11,5,5,"Arch"
    # 3,8,NULL,"76ers",3,8,NULL,"Walnut"
    # 4,NULL,1,"Flyers",NULL,NULL,NULL,NULL
    # 5,10,1,"Union",1,10,4,"Chestnut"
    # 5,10,1,"Union",2,10,NULL,"Pine"
    # 7,-1,22,"Fusion",NULL,NULL,NULL,NULL
    # 8,11,9,"Fight",NULL,NULL,NULL,NULL
    # NULL,NULL,NULL,NULL,4,NULL,3,"Locust"
    # NULL,NULL,NULL,NULL,5,NULL,NULL,"Cherry"
    # NULL,NULL,NULL,NULL,6,NULL,NULL,"Spruce"
    # NULL,NULL,NULL,NULL,7,NULL,1,"Cypress"
    # NULL,NULL,NULL,NULL,10,14,9,"Market"
    # NULL,NULL,NULL,NULL,12,-15,13,"Vine"

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 5, 8, inv, 10, 10, -1, 11, inv, inv, inv, inv, 14, -15], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 1, 1, 3, 4, 5, 5, 7, 8, inv, inv, inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(result.col2_foo, rt.FA([inv, inv, inv, inv, 1, 1, 1, 22, 9, inv, inv, inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(result.team_name, rt.FA([b'Phillies', b'Phillies', b'Phillies', b'76ers', b'Flyers', b'Union', b'Union', b'Fusion', b'Fight', b'', b'', b'', b'', b'', b'']))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8, 9, 11, 3, inv, 1, 2, inv, inv, 4, 5, 6, 7, 10, 12], dtype=np.int32))
    assert_array_equal(result.col2_bar, rt.FA([5, 5, 5, 8, inv, 10, 10, inv, inv, inv, inv, inv, inv, 14, -15], dtype=np.int32))
    assert_array_equal(result.strcol, rt.FA([b'Lombard', b'Sansom', b'Arch', b'Walnut', b'', b'Chestnut', b'Pine', b'', b'', b'Locust', b'Cherry', b'Spruce', b'Cypress', b'Market', b'Vine']))


#########################
# Multi-key merge tests
#########################


def test_merge2_sql_semantics_leftjoin_multi():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    left join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='left',
        suffixes=('_foo', '_bar'),
        indicator=True,
    )

    assert result.get_nrows() == 6

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      1 |        5 |     NULL |   NULL |     NULL |     NULL | NULL       |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # |      3 |        8 |     NULL |   NULL |     NULL |     NULL | NULL       |
    # |      4 |     NULL |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      5 |       10 |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      6 |     NULL |        4 |   NULL |     NULL |     NULL | NULL       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 6 rows in set (0.0007 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, inv, 10, inv], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([inv, 5, inv, 1, 1, 4], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 3, 4, 5, 6], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([inv, 8, inv, inv, inv, inv], dtype=np.int32)
    )
    assert_array_equal(result.strcol, rt.FA([b'', b'Lombard', b'', b'', b'', b'']))


def test_merge2_sql_semantics_rightjoin_multi():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    right join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by b.id, f.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='right',
        suffixes=('_foo', '_bar'),
        indicator=True,
    )

    assert result.get_nrows() == 9

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |   NULL |     NULL |     NULL |      1 |       10 |        4 | Chestnut   |
    # |   NULL |     NULL |     NULL |      2 |       10 |     NULL | Pine       |
    # |   NULL |     NULL |     NULL |      3 |        8 |     NULL | Walnut     |
    # |   NULL |     NULL |     NULL |      4 |     NULL |        3 | Locust     |
    # |   NULL |     NULL |     NULL |      5 |     NULL |     NULL | Cherry     |
    # |   NULL |     NULL |     NULL |      6 |     NULL |     NULL | Spruce     |
    # |   NULL |     NULL |     NULL |      7 |     NULL |        1 | Cypress    |
    # |   NULL |     NULL |     NULL |      9 |        5 |     NULL | Sansom     |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 9 rows in set (0.0009 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(
        result.col1, rt.FA([10, 10, 8, inv, inv, inv, inv, 5, 5], dtype=np.int32)
    )
    assert_array_equal(
        result.col2, rt.FA([4, inv, inv, 3, inv, inv, 1, 5, inv], dtype=np.int32)
    )

    # Cols from the left Dataset.
    assert_array_equal(
        result.id_foo,
        rt.FA([inv, inv, inv, inv, inv, inv, inv, 2, inv], dtype=np.int32),
    )

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol,
        rt.FA(
            [
                b'Chestnut',
                b'Pine',
                b'Walnut',
                b'Locust',
                b'Cherry',
                b'Spruce',
                b'Cypress',
                b'Lombard',
                b'Sansom',
            ]
        ),
    )


def test_merge2_sql_semantics_innerjoin_multi():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    inner join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='inner',
        suffixes=('_foo', '_bar'),
        indicator=True,
    )

    assert result.get_nrows() == 1

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 1 row in set (0.0006 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([5], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([2], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8], dtype=np.int32))
    assert_array_equal(result.strcol, rt.FA([b'Lombard']))


def test_merge2_sql_semantics_outerjoin_multi():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        f.team_name as foo_teamname,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    full outer join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics2()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='outer',
        suffixes=('_foo', '_bar'),
        indicator=True,
    )

    assert result.get_nrows() == 19

    # "foo_id","foo_col1","foo_col2","foo_teamname","bar_id","bar_col1","bar_col2","bar_strcol"
    # 1,5,NULL,"Phillies",NULL,NULL,NULL,NULL
    # 2,5,5,"Eagles",8,5,5,"Lombard"
    # 2,5,5,"Eagles",11,5,5,"Arch"
    # 3,8,NULL,"76ers",NULL,NULL,NULL,NULL
    # 4,NULL,1,"Flyers",NULL,NULL,NULL,NULL
    # 5,10,1,"Union",NULL,NULL,NULL,NULL
    # 6,NULL,4,"Wings",NULL,NULL,NULL,NULL
    # 7,-1,22,"Fusion",NULL,NULL,NULL,NULL
    # 8,11,9,"Fight",NULL,NULL,NULL,NULL
    # NULL,NULL,NULL,NULL,1,10,4,"Chestnut"
    # NULL,NULL,NULL,NULL,2,10,NULL,"Pine"
    # NULL,NULL,NULL,NULL,3,8,NULL,"Walnut"
    # NULL,NULL,NULL,NULL,4,NULL,3,"Locust"
    # NULL,NULL,NULL,NULL,5,NULL,NULL,"Cherry"
    # NULL,NULL,NULL,NULL,6,NULL,NULL,"Spruce"
    # NULL,NULL,NULL,NULL,7,NULL,1,"Cypress"
    # NULL,NULL,NULL,NULL,9,5,NULL,"Sansom"
    # NULL,NULL,NULL,NULL,10,14,9,"Market"
    # NULL,NULL,NULL,NULL,12,-15,13,"Vine"

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 5, 8, inv, 10, inv, -1, 11, 10, 10, 8, inv, inv, inv, inv, 5, 14, -15], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([inv, 5, 5, inv, 1, 1, 4, 22, 9, 4, inv, inv, 3, inv, inv, 1, inv, 9, 13], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 2, 3, 4, 5, 6, 7, 8, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(
        result.team_name,
        rt.FA([b'Phillies', b'Eagles', b'Eagles', b'76ers', b'Flyers', b'Union', b'Wings', b'Fusion', b'Fight', b'', b'', b'', b'', b'', b'', b'', b'', b'', b''])
    )

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([inv, 8, 11, inv, inv, inv, inv, inv, inv, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol,
        rt.FA([b'', b'Lombard', b'Arch', b'', b'', b'', b'', b'', b'', b'Chestnut', b'Pine', b'Walnut', b'Locust', b'Cherry', b'Spruce', b'Cypress', b'Sansom', b'Market', b'Vine'])
    )


def test_merge2_sql_semantics_leftjoin_multi_keep_Nonefirst():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    left join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct FIRST_VALUE(id) over w as `first_row_id`
            from sql_semantics.bar
            window w as (partition by bar.col1, bar.col2 order by bar.id)
        ) as bar_ids
        on
            bar.id = bar_ids.first_row_id
    ) as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='left',
        suffixes=('_foo', '_bar'),
        keep=(None, 'first'),
        indicator=True,
    )

    assert result.get_nrows() == 6

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      1 |        5 |     NULL |   NULL |     NULL |     NULL | NULL       |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # |      3 |        8 |     NULL |   NULL |     NULL |     NULL | NULL       |
    # |      4 |     NULL |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      5 |       10 |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      6 |     NULL |        4 |   NULL |     NULL |     NULL | NULL       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 6 rows in set (0.0016 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, inv, 10, inv], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([inv, 5, inv, 1, 1, 4], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 3, 4, 5, 6], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([inv, 8, inv, inv, inv, inv], dtype=np.int32)
    )
    assert_array_equal(result.strcol, rt.FA([b'', b'Lombard', b'', b'', b'', b'']))


def test_merge2_sql_semantics_leftjoin_multi_keep_Nonelast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    left join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over w as `last_row_id`
            from sql_semantics.bar
            window w as (partition by bar.col1, bar.col2 order by bar.id)
        ) as bar_ids
        on
            bar.id = bar_ids.last_row_id
    ) as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='left',
        suffixes=('_foo', '_bar'),
        keep=(None, 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 6

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      1 |        5 |     NULL |   NULL |     NULL |     NULL | NULL       |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # |      3 |        8 |     NULL |   NULL |     NULL |     NULL | NULL       |
    # |      4 |     NULL |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      5 |       10 |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      6 |     NULL |        4 |   NULL |     NULL |     NULL | NULL       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 6 rows in set (0.0011 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, inv, 10, inv], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([inv, 5, inv, 1, 1, 4], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 3, 4, 5, 6], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([inv, 8, inv, inv, inv, inv], dtype=np.int32)
    )
    assert_array_equal(result.strcol, rt.FA([b'', b'Lombard', b'', b'', b'', b'']))


@pytest.mark.xfail(
    reason="RIP-260: This test currently fails, but the test data used here may itself be incorrect. Need to verify the test data to determine how to proceed.",
    strict=True
)
def test_merge2_sql_semantics_rightjoin_multi_keep_Nonefirst():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    right join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct FIRST_VALUE(id) over w as `first_row_id`
            from sql_semantics.bar
            window w as (partition by bar.col1, bar.col2 order by bar.id)
        ) as bar_ids
        on
            bar.id = bar_ids.first_row_id
    ) as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by b.id, f.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='right',
        suffixes=('_foo', '_bar'),
        keep=(None, 'first'),
        indicator=True,
    )

    assert result.get_nrows() == 8

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |   NULL |     NULL |     NULL |      1 |       10 |        4 | Chestnut   |
    # |   NULL |     NULL |     NULL |      2 |       10 |     NULL | Pine       |
    # |   NULL |     NULL |     NULL |      3 |        8 |     NULL | Walnut     |
    # |   NULL |     NULL |     NULL |      4 |     NULL |        3 | Locust     |
    # |   NULL |     NULL |     NULL |      6 |     NULL |     NULL | Spruce     |
    # |   NULL |     NULL |     NULL |      7 |     NULL |        1 | Cypress    |
    # |   NULL |     NULL |     NULL |      9 |        5 |     NULL | Sansom     |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 8 rows in set (0.0010 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(
        result.col1, rt.FA([10, 10, 8, inv, inv, inv, 5, 5], dtype=np.int32)
    )
    assert_array_equal(
        result.col2, rt.FA([4, inv, inv, 3, inv, 1, 5, inv], dtype=np.int32)
    )

    # Cols from the left Dataset.
    assert_array_equal(
        result.id_foo, rt.FA([inv, inv, inv, inv, inv, inv, 2, inv], dtype=np.int32)
    )

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([1, 2, 3, 4, 6, 7, 8, 9], dtype=np.int32))
    assert_array_equal(
        result.strcol,
        rt.FA(
            [
                b'Chestnut',
                b'Pine',
                b'Walnut',
                b'Locust',
                b'Spruce',
                b'Cypress',
                b'Lombard',
                b'Sansom',
            ]
        ),
    )


def test_merge2_sql_semantics_rightjoin_multi_keep_Nonelast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    right join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over w as `last_row_id`
            from sql_semantics.bar
            window w as (partition by bar.col1, bar.col2 order by bar.id)
        ) as bar_ids
        on
            bar.id = bar_ids.last_row_id
    ) as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by b.id, f.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='right',
        suffixes=('_foo', '_bar'),
        keep=(None, 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 8

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |   NULL |     NULL |     NULL |      1 |       10 |        4 | Chestnut   |
    # |   NULL |     NULL |     NULL |      2 |       10 |     NULL | Pine       |
    # |   NULL |     NULL |     NULL |      3 |        8 |     NULL | Walnut     |
    # |   NULL |     NULL |     NULL |      4 |     NULL |        3 | Locust     |
    # |   NULL |     NULL |     NULL |      6 |     NULL |     NULL | Spruce     |
    # |   NULL |     NULL |     NULL |      7 |     NULL |        1 | Cypress    |
    # |   NULL |     NULL |     NULL |      9 |        5 |     NULL | Sansom     |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 8 rows in set (0.0012 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(
        result.col1, rt.FA([10, 10, 8, inv, inv, inv, 5, 5], dtype=np.int32)
    )
    assert_array_equal(
        result.col2, rt.FA([4, inv, inv, 3, inv, 1, 5, inv], dtype=np.int32)
    )

    # Cols from the left Dataset.
    assert_array_equal(
        result.id_foo, rt.FA([inv, inv, inv, inv, inv, inv, 2, inv], dtype=np.int32)
    )

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([1, 2, 3, 4, 6, 7, 8, 9], dtype=np.int32))
    assert_array_equal(
        result.strcol,
        rt.FA(
            [
                b'Chestnut',
                b'Pine',
                b'Walnut',
                b'Locust',
                b'Spruce',
                b'Cypress',
                b'Lombard',
                b'Sansom',
            ]
        ),
    )


def test_merge2_sql_semantics_innerjoin_multi_keep_Nonefirst():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    inner join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct FIRST_VALUE(id) over w as `first_row_id`
            from sql_semantics.bar
            window w as (partition by bar.col1, bar.col2 order by bar.id)
        ) as bar_ids
        on
            bar.id = bar_ids.first_row_id
    ) as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='inner',
        suffixes=('_foo', '_bar'),
        keep=(None, 'first'),
        indicator=True,
    )

    assert result.get_nrows() == 1

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 1 row in set (0.0031 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([5], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([2], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8], dtype=np.int32))
    assert_array_equal(result.strcol, rt.FA([b'Lombard']))


def test_merge2_sql_semantics_innerjoin_multi_keep_Nonelast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    inner join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over w as `last_row_id`
            from sql_semantics.bar
            window w as (partition by bar.col1, bar.col2 order by bar.id)
        ) as bar_ids
        on
            bar.id = bar_ids.last
            _row_id
    ) as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='inner',
        suffixes=('_foo', '_bar'),
        keep=(None, 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 1

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 1 row in set (0.0009 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([5], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([2], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8], dtype=np.int32))
    assert_array_equal(result.strcol, rt.FA([b'Lombard']))


def test_merge2_sql_semantics_outerjoin_multi_keep_Nonefirst():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        f.team_name as foo_teamname,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    full outer join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct FIRST_VALUE(id) over w as firstlast_row_id
            from sql_semantics.bar
            window w as (
                partition by bar.col1, bar.col2
                order by bar.id asc
            )
        ) as bar_ids
        on
            bar.id = bar_ids.firstlast_row_id
    ) as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics2()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='outer',
        suffixes=('_foo', '_bar'),
        keep=(None, 'first'),
        indicator=True,
    )

    assert result.get_nrows() == 17

    # "foo_id","foo_col1","foo_col2","foo_teamname","bar_id","bar_col1","bar_col2","bar_strcol"
    # 1,5,NULL,"Phillies",NULL,NULL,NULL,NULL
    # 2,5,5,"Eagles",8,5,5,"Lombard"
    # 3,8,NULL,"76ers",NULL,NULL,NULL,NULL
    # 4,NULL,1,"Flyers",NULL,NULL,NULL,NULL
    # 5,10,1,"Union",NULL,NULL,NULL,NULL
    # 6,NULL,4,"Wings",NULL,NULL,NULL,NULL
    # 7,-1,22,"Fusion",NULL,NULL,NULL,NULL
    # 8,11,9,"Fight",NULL,NULL,NULL,NULL
    # NULL,NULL,NULL,NULL,1,10,4,"Chestnut"
    # NULL,NULL,NULL,NULL,2,10,NULL,"Pine"
    # NULL,NULL,NULL,NULL,3,8,NULL,"Walnut"
    # NULL,NULL,NULL,NULL,4,NULL,3,"Locust"
    # NULL,NULL,NULL,NULL,5,NULL,NULL,"Cherry"
    # NULL,NULL,NULL,NULL,7,NULL,1,"Cypress"
    # NULL,NULL,NULL,NULL,9,5,NULL,"Sansom"
    # NULL,NULL,NULL,NULL,10,14,9,"Market"
    # NULL,NULL,NULL,NULL,12,-15,13,"Vine"

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, inv, 10, inv, -1, 11, 10, 10, 8, inv, inv, inv, 5, 14, -15], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([inv, 5, inv, 1, 1, 4, 22, 9, 4, inv, inv, 3, inv, 1, inv, 9, 13], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 3, 4, 5, 6, 7, 8, inv, inv, inv, inv, inv, inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(
        result.team_name,
        rt.FA([b'Phillies', b'Eagles', b'76ers', b'Flyers', b'Union', b'Wings', b'Fusion', b'Fight', b'', b'', b'', b'', b'', b'', b'', b'', b'']))

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([inv, 8, inv, inv, inv, inv, inv, inv, 1, 2, 3, 4, 5, 7, 9, 10, 12], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol,
        rt.FA([b'', b'Lombard', b'', b'', b'', b'', b'', b'', b'Chestnut', b'Pine', b'Walnut', b'Locust', b'Cherry', b'Cypress', b'Sansom', b'Market', b'Vine']))


def test_merge2_sql_semantics_outerjoin_multi_keep_Nonelast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        f.team_name as foo_teamname,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from
        sql_semantics.foo as f
    full outer join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over w as firstlast_row_id
            from sql_semantics.bar
            window w as (
                partition by bar.col1, bar.col2
                order by bar.id
                rows between unbounded preceding and unbounded following
            )
        ) as bar_ids
        on
            bar.id = bar_ids.firstlast_row_id
    ) as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics2()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='outer',
        suffixes=('_foo', '_bar'),
        keep=(None, 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 17

    # "foo_id","foo_col1","foo_col2","foo_teamname","bar_id","bar_col1","bar_col2","bar_strcol"
    # 1,5,NULL,"Phillies",NULL,NULL,NULL,NULL
    # 2,5,5,"Eagles",11,5,5,"Arch"
    # 3,8,NULL,"76ers",NULL,NULL,NULL,NULL
    # 4,NULL,1,"Flyers",NULL,NULL,NULL,NULL
    # 5,10,1,"Union",NULL,NULL,NULL,NULL
    # 6,NULL,4,"Wings",NULL,NULL,NULL,NULL
    # 7,-1,22,"Fusion",NULL,NULL,NULL,NULL
    # 8,11,9,"Fight",NULL,NULL,NULL,NULL
    # NULL,NULL,NULL,NULL,1,10,4,"Chestnut"
    # NULL,NULL,NULL,NULL,2,10,NULL,"Pine"
    # NULL,NULL,NULL,NULL,3,8,NULL,"Walnut"
    # NULL,NULL,NULL,NULL,4,NULL,3,"Locust"
    # NULL,NULL,NULL,NULL,6,NULL,NULL,"Spruce"
    # NULL,NULL,NULL,NULL,7,NULL,1,"Cypress"
    # NULL,NULL,NULL,NULL,9,5,NULL,"Sansom"
    # NULL,NULL,NULL,NULL,10,14,9,"Market"
    # NULL,NULL,NULL,NULL,12,-15,13,"Vine"

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, inv, 10, inv, -1, 11, 10, 10, 8, inv, inv, inv, 5, 14, -15], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([inv, 5, inv, 1, 1, 4, 22, 9, 4, inv, inv, 3, inv, 1, inv, 9, 13], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 3, 4, 5, 6, 7, 8, inv, inv, inv, inv, inv, inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(
        result.team_name,
        rt.FA([b'Phillies', b'Eagles', b'76ers', b'Flyers', b'Union', b'Wings', b'Fusion', b'Fight', b'', b'', b'', b'', b'', b'', b'', b'', b'']))

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([inv, 11, inv, inv, inv, inv, inv, inv, 1, 2, 3, 4, 6, 7, 9, 10, 12], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol,
        rt.FA([b'', b'Arch', b'', b'', b'', b'', b'', b'', b'Chestnut', b'Pine', b'Walnut', b'Locust', b'Spruce', b'Cypress', b'Sansom', b'Market', b'Vine']))


def test_merge2_sql_semantics_leftjoin_multi_keep_firstNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over w as `first_row_id`
            from sql_semantics.foo
            window w as (partition by foo.col1, foo.col2 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.first_row_id
    ) as f
    left join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='left',
        suffixes=('_foo', '_bar'),
        keep=('first', None),
        indicator=True,
    )

    print(repr(result))

    assert result.get_nrows() == 6

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      1 |        5 |     NULL |   NULL |     NULL |     NULL | NULL       |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # |      3 |        8 |     NULL |   NULL |     NULL |     NULL | NULL       |
    # |      4 |     NULL |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      5 |       10 |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      6 |     NULL |        4 |   NULL |     NULL |     NULL | NULL       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 6 rows in set (0.0019 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, inv, 10, inv], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([inv, 5, inv, 1, 1, 4], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 3, 4, 5, 6], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([inv, 8, inv, inv, inv, inv], dtype=np.int32)
    )
    assert_array_equal(result.strcol, rt.FA([b'', b'Lombard', b'', b'', b'', b'']))


def test_merge2_sql_semantics_leftjoin_multi_keep_lastNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct LAST_VALUE(id) over w as `last_row_id`
            from sql_semantics.foo
            window w as (partition by foo.col1, foo.col2 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.last_row_id
    ) as f
    left join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='left',
        suffixes=('_foo', '_bar'),
        keep=('last', None),
        indicator=True,
    )

    assert result.get_nrows() == 6

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      1 |        5 |     NULL |   NULL |     NULL |     NULL | NULL       |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # |      3 |        8 |     NULL |   NULL |     NULL |     NULL | NULL       |
    # |      4 |     NULL |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      5 |       10 |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      6 |     NULL |        4 |   NULL |     NULL |     NULL | NULL       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 6 rows in set (0.0009 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, inv, 10, inv], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([inv, 5, inv, 1, 1, 4], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 3, 4, 5, 6], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([inv, 8, inv, inv, inv, inv], dtype=np.int32)
    )
    assert_array_equal(result.strcol, rt.FA([b'', b'Lombard', b'', b'', b'', b'']))


def test_merge2_sql_semantics_rightjoin_multi_keep_firstNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over w as `first_row_id`
            from sql_semantics.foo
            window w as (partition by foo.col1, foo.col2 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.first_row_id
    ) as f
    right join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by b.id, f.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='right',
        suffixes=('_foo', '_bar'),
        keep=('first', None),
        indicator=True,
    )

    assert result.get_nrows() == 9

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |   NULL |     NULL |     NULL |      1 |       10 |        4 | Chestnut   |
    # |   NULL |     NULL |     NULL |      2 |       10 |     NULL | Pine       |
    # |   NULL |     NULL |     NULL |      3 |        8 |     NULL | Walnut     |
    # |   NULL |     NULL |     NULL |      4 |     NULL |        3 | Locust     |
    # |   NULL |     NULL |     NULL |      5 |     NULL |     NULL | Cherry     |
    # |   NULL |     NULL |     NULL |      6 |     NULL |     NULL | Spruce     |
    # |   NULL |     NULL |     NULL |      7 |     NULL |        1 | Cypress    |
    # |   NULL |     NULL |     NULL |      9 |        5 |     NULL | Sansom     |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 9 rows in set (0.0009 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(
        result.col1, rt.FA([10, 10, 8, inv, inv, inv, inv, 5, 5], dtype=np.int32)
    )
    assert_array_equal(
        result.col2, rt.FA([4, inv, inv, 3, inv, inv, 1, 5, inv], dtype=np.int32)
    )

    # Cols from the left Dataset.
    assert_array_equal(
        result.id_foo,
        rt.FA([inv, inv, inv, inv, inv, inv, inv, 2, inv], dtype=np.int32),
    )

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol,
        rt.FA(
            [
                b'Chestnut',
                b'Pine',
                b'Walnut',
                b'Locust',
                b'Cherry',
                b'Spruce',
                b'Cypress',
                b'Lombard',
                b'Sansom',
            ]
        ),
    )


def test_merge2_sql_semantics_rightjoin_multi_keep_lastNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct LAST_VALUE(id) over w as `last_row_id`
            from sql_semantics.foo
            window w as (partition by foo.col1, foo.col2 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.last_row_id
    ) as f
    right join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by b.id, f.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='right',
        suffixes=('_foo', '_bar'),
        keep=('last', None),
        indicator=True,
    )

    assert result.get_nrows() == 9

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |   NULL |     NULL |     NULL |      1 |       10 |        4 | Chestnut   |
    # |   NULL |     NULL |     NULL |      2 |       10 |     NULL | Pine       |
    # |   NULL |     NULL |     NULL |      3 |        8 |     NULL | Walnut     |
    # |   NULL |     NULL |     NULL |      4 |     NULL |        3 | Locust     |
    # |   NULL |     NULL |     NULL |      5 |     NULL |     NULL | Cherry     |
    # |   NULL |     NULL |     NULL |      6 |     NULL |     NULL | Spruce     |
    # |   NULL |     NULL |     NULL |      7 |     NULL |        1 | Cypress    |
    # |   NULL |     NULL |     NULL |      9 |        5 |     NULL | Sansom     |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 9 rows in set (0.0009 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(
        result.col1, rt.FA([10, 10, 8, inv, inv, inv, inv, 5, 5], dtype=np.int32)
    )
    assert_array_equal(
        result.col2, rt.FA([4, inv, inv, 3, inv, inv, 1, 5, inv], dtype=np.int32)
    )

    # Cols from the left Dataset.
    assert_array_equal(
        result.id_foo,
        rt.FA([inv, inv, inv, inv, inv, inv, inv, 2, inv], dtype=np.int32),
    )

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol,
        rt.FA(
            [
                b'Chestnut',
                b'Pine',
                b'Walnut',
                b'Locust',
                b'Cherry',
                b'Spruce',
                b'Cypress',
                b'Lombard',
                b'Sansom',
            ]
        ),
    )


def test_merge2_sql_semantics_innerjoin_multi_keep_firstNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over w as `first_row_id`
            from sql_semantics.foo
            window w as (partition by foo.col1, foo.col2 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.first_row_id
    ) as f
    inner join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='inner',
        suffixes=('_foo', '_bar'),
        keep=('first', None),
        indicator=True,
    )

    assert result.get_nrows() == 1

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 1 row in set (0.0009 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([5], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([2], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8], dtype=np.int32))
    assert_array_equal(result.strcol, rt.FA([b'Lombard']))


def test_merge2_sql_semantics_innerjoin_multi_keep_lastNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct LAST_VALUE(id) over w as `last_row_id`
            from sql_semantics.foo
            window w as (partition by foo.col1, foo.col2 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.last_row_id
    ) as f
    inner join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='inner',
        suffixes=('_foo', '_bar'),
        keep=('last', None),
        indicator=True,
    )

    assert result.get_nrows() == 1

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 1 row in set (0.0009 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([5], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([2], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([8], dtype=np.int32))
    assert_array_equal(result.strcol, rt.FA([b'Lombard']))


@xfail_rip260_outermerge_left_keep
def test_merge2_sql_semantics_outerjoin_multi_keep_firstNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        f.team_name as foo_teamname,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over w as firstlast_row_id
            from sql_semantics.foo
            window w as (partition by foo.col1, foo.col2 order by foo.id asc)
        ) as foo_ids
        on
            foo.id = foo_ids.firstlast_row_id
    ) as f
    full outer join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics2()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='outer',
        suffixes=('_foo', '_bar'),
        keep=('first', None),
        indicator=True,
    )

    assert result.get_nrows() == 19

    # "foo_id","foo_col1","foo_col2","foo_teamname","bar_id","bar_col1","bar_col2","bar_strcol"
    # 1,5,NULL,"Phillies",NULL,NULL,NULL,NULL
    # 2,5,5,"Eagles",8,5,5,"Lombard"
    # 2,5,5,"Eagles",11,5,5,"Arch"
    # 3,8,NULL,"76ers",NULL,NULL,NULL,NULL
    # 4,NULL,1,"Flyers",NULL,NULL,NULL,NULL
    # 5,10,1,"Union",NULL,NULL,NULL,NULL
    # 6,NULL,4,"Wings",NULL,NULL,NULL,NULL
    # 7,-1,22,"Fusion",NULL,NULL,NULL,NULL
    # 8,11,9,"Fight",NULL,NULL,NULL,NULL
    # NULL,NULL,NULL,NULL,1,10,4,"Chestnut"
    # NULL,NULL,NULL,NULL,2,10,NULL,"Pine"
    # NULL,NULL,NULL,NULL,3,8,NULL,"Walnut"
    # NULL,NULL,NULL,NULL,4,NULL,3,"Locust"
    # NULL,NULL,NULL,NULL,5,NULL,NULL,"Cherry"
    # NULL,NULL,NULL,NULL,6,NULL,NULL,"Spruce"
    # NULL,NULL,NULL,NULL,7,NULL,1,"Cypress"
    # NULL,NULL,NULL,NULL,9,5,NULL,"Sansom"
    # NULL,NULL,NULL,NULL,10,14,9,"Market"
    # NULL,NULL,NULL,NULL,12,-15,13,"Vine"

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 5, 8, inv, 10, inv, -1, 11, 10, 10, 8, inv, inv, inv, inv, 5, 14, -15], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([inv, 5, 5, inv, 1, 1, 4, 22, 9, 4, inv, inv, 3, inv, inv, 1, inv, 9, 13], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 2, 3, 4, 5, 6, 7, 8, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(
        result.team_name,
        rt.FA([b'Phillies', b'Eagles', b'Eagles', b'76ers', b'Flyers', b'Union', b'Wings', b'Fusion', b'Fight', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'']))

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([inv, 8, 11, inv, inv, inv, inv, inv, inv, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol,
        rt.FA([b'', b'Lombard', b'Arch', b'', b'', b'', b'', b'', b'', b'Chestnut', b'Pine', b'Walnut', b'Locust', b'Cherry', b'Spruce', b'Cypress', b'Sansom', b'Market', b'Vine']))


@xfail_rip260_outermerge_left_keep
def test_merge2_sql_semantics_outerjoin_multi_keep_lastNone():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        f.team_name as foo_teamname,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct LAST_VALUE(id) over w as last_row_id
            from sql_semantics.foo
            window w as (
                partition by foo.col1, foo.col2
                order by foo.id
                rows between unbounded preceding and unbounded following
            )
        ) as foo_ids
        on
            foo.id = foo_ids.last_row_id
    ) as f
    full outer join
        sql_semantics.bar as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics2()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='outer',
        suffixes=('_foo', '_bar'),
        keep=('last', None),
        indicator=True,
    )

    assert result.get_nrows() == 19

    # "foo_id","foo_col1","foo_col2","foo_teamname","bar_id","bar_col1","bar_col2","bar_strcol"
    # 1,5,NULL,"Phillies",NULL,NULL,NULL,NULL
    # 2,5,5,"Eagles",8,5,5,"Lombard"
    # 2,5,5,"Eagles",11,5,5,"Arch"
    # 3,8,NULL,"76ers",NULL,NULL,NULL,NULL
    # 4,NULL,1,"Flyers",NULL,NULL,NULL,NULL
    # 5,10,1,"Union",NULL,NULL,NULL,NULL
    # 6,NULL,4,"Wings",NULL,NULL,NULL,NULL
    # 7,-1,22,"Fusion",NULL,NULL,NULL,NULL
    # 8,11,9,"Fight",NULL,NULL,NULL,NULL
    # NULL,NULL,NULL,NULL,1,10,4,"Chestnut"
    # NULL,NULL,NULL,NULL,2,10,NULL,"Pine"
    # NULL,NULL,NULL,NULL,3,8,NULL,"Walnut"
    # NULL,NULL,NULL,NULL,4,NULL,3,"Locust"
    # NULL,NULL,NULL,NULL,5,NULL,NULL,"Cherry"
    # NULL,NULL,NULL,NULL,6,NULL,NULL,"Spruce"
    # NULL,NULL,NULL,NULL,7,NULL,1,"Cypress"
    # NULL,NULL,NULL,NULL,9,5,NULL,"Sansom"
    # NULL,NULL,NULL,NULL,10,14,9,"Market"
    # NULL,NULL,NULL,NULL,12,-15,13,"Vine"

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 5, 8, inv, 10, inv, -1, 11, 10, 10, 8, inv, inv, inv, inv, 5, 14, -15], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([inv, 5, 5, inv, 1, 1, 4, 22, 9, 4, inv, inv, 3, inv, inv, 1, inv, 9, 13], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 2, 3, 4, 5, 6, 7, 8, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(
        result.team_name,
        rt.FA([b'Phillies', b'Eagles', b'Eagles', b'76ers', b'Flyers', b'Union', b'Wings', b'Fusion', b'Fight', b'', b'', b'', b'', b'', b'', b'', b'', b'', b'']))

    # Cols from the right Dataset.
    assert_array_equal(
        result.id_bar, rt.FA([inv, 8, 11, inv, inv, inv, inv, inv, inv, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12], dtype=np.int32)
    )
    assert_array_equal(
        result.strcol,
        rt.FA([b'', b'Lombard', b'Arch', b'', b'', b'', b'', b'', b'', b'Chestnut', b'Pine', b'Walnut', b'Locust', b'Cherry', b'Spruce', b'Cypress', b'Sansom', b'Market', b'Vine']))


def test_merge2_sql_semantics_leftjoin_multi_keep_firstlast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        f.team_name as foo_teamname,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over wf as first_row_id
            from sql_semantics.foo
            window wf as (
                partition by foo.col1, foo.col2
                order by foo.id
            )
        ) as foo_ids
        on
            foo.id = foo_ids.first_row_id
    ) as f
    left join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over wb as last_row_id
            from sql_semantics.bar
            window wb as (
                partition by bar.col1, bar.col2
                order by bar.id
                rows between unbounded preceding and unbounded following
            )
        ) as bar_ids
        on
            bar.id = bar_ids.last_row_id
    ) as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics2()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='left',
        suffixes=('_foo', '_bar'),
        keep=('first', 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 8

    # "foo_id","foo_col1","foo_col2","foo_teamname","bar_id","bar_col1","bar_col2","bar_strcol"
    # 1,5,NULL,"Phillies",NULL,NULL,NULL,NULL
    # 2,5,5,"Eagles",11,5,5,"Arch"
    # 3,8,NULL,"76ers",NULL,NULL,NULL,NULL
    # 4,NULL,1,"Flyers",NULL,NULL,NULL,NULL
    # 5,10,1,"Union",NULL,NULL,NULL,NULL
    # 6,NULL,4,"Wings",NULL,NULL,NULL,NULL
    # 7,-1,22,"Fusion",NULL,NULL,NULL,NULL
    # 8,11,9,"Fight",NULL,NULL,NULL,NULL

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, inv, 10, inv, -1, 11], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([inv, 5, inv, 1, 1, 4, 22, 9], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32))
    assert_array_equal(
        result.team_name,
        rt.FA([b'Phillies', b'Eagles', b'76ers', b'Flyers', b'Union', b'Wings', b'Fusion', b'Fight'])
    )

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([inv, 11, inv, inv, inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(
        result.strcol, rt.FA([b'', b'Arch', b'', b'', b'', b'', b'', b''])
    )


@pytest.mark.xfail(
    reason="RIP-260: Result of this test is generally correct, but the ordering of the rows is "
           "slightly different (and incorrect) so even though the data is correct the test fails.",
    strict=True
)
def test_merge2_sql_semantics_rightjoin_multi_keep_firstlast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        f.team_name as foo_teamname,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over wf as first_row_id
            from sql_semantics.foo
            window wf as (partition by foo.col1, foo.col2 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.first_row_id
    ) as f
    right join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over wb as last_row_id
            from sql_semantics.bar
            window wb as (
                partition by bar.col1, bar.col2
                order by bar.id
                rows between unbounded preceding and unbounded following
            )
        ) as bar_ids
        on
            bar.id = bar_ids.last_row_id
    ) as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by b.id, f.id asc;
    """

    foo, bar = TestDataset.sql_semantics2()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='right',
        suffixes=('_foo', '_bar'),
        keep=('first', 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 10

    # "foo_id","foo_col1","foo_col2","foo_teamname","bar_id","bar_col1","bar_col2","bar_strcol"
    # NULL,NULL,NULL,NULL,1,10,4,"Chestnut"
    # NULL,NULL,NULL,NULL,2,10,NULL,"Pine"
    # NULL,NULL,NULL,NULL,3,8,NULL,"Walnut"
    # NULL,NULL,NULL,NULL,4,NULL,3,"Locust"
    # NULL,NULL,NULL,NULL,6,NULL,NULL,"Spruce"
    # NULL,NULL,NULL,NULL,7,NULL,1,"Cypress"
    # NULL,NULL,NULL,NULL,9,5,NULL,"Sansom"
    # NULL,NULL,NULL,NULL,10,14,9,"Market"
    # 2,5,5,"Eagles",11,5,5,"Arch"
    # NULL,NULL,NULL,NULL,12,-15,13,"Vine"

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([10, 10, 8, inv, inv, inv, 5, 14, 5, -15], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([4, inv, inv, 3, inv, 1, inv, 9, 5, 13], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([inv, inv, inv, inv, inv, inv, inv, inv, 2, inv], dtype=np.int32))
    assert_array_equal(
        result.team_name,
        rt.FA([b'', b'', b'', b'', b'', b'', b'', b'', b'Eagles', b''])
    )

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([1, 2, 3, 4, 6, 7, 9, 10, 11, 12], dtype=np.int32))
    assert_array_equal(
        result.strcol,
        rt.FA([b'Chestnut', b'Pine', b'Walnut', b'Locust', b'Spruce', b'Cypress', b'Sansom', b'Market', b'Arch', b'Vine'])
    )


def test_merge2_sql_semantics_innerjoin_multi_keep_firstlast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        f.team_name as foo_teamname,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over wf as first_row_id
            from sql_semantics.foo
            window wf as (partition by foo.col1, foo.col2 order by foo.id)
        ) as foo_ids
        on
            foo.id = foo_ids.first_row_id
    ) as f
    inner join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over wb as last_row_id
            from sql_semantics.bar
            window wb as (
                partition by bar.col1, bar.col2
                order by bar.id
                rows between unbounded preceding and unbounded following
            )
        ) as bar_ids
        on
            bar.id = bar_ids.last_row_id
    ) as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics2()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='inner',
        suffixes=('_foo', '_bar'),
        keep=('first', 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 1

    # "foo_id","foo_col1","foo_col2","foo_teamname","bar_id","bar_col1","bar_col2","bar_strcol"
    # 2,5,5,"Eagles",11,5,5,"Arch"

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([5], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([2], dtype=np.int32))
    assert_array_equal(
        result.team_name, rt.FA([b'Eagles'])
    )

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([11], dtype=np.int32))
    assert_array_equal(result.strcol, rt.FA([b'Arch']))


@xfail_rip260_outermerge_left_keep
def test_merge2_sql_semantics_outerjoin_multi_keep_firstlast():
    """
    Test that merge2 matches the following SQL query:

    select
        f.id as foo_id,
        f.col1 as foo_col1,
        f.col2 as foo_col2,
        f.team_name as foo_teamname,
        b.id as bar_id,
        b.col1 as bar_col1,
        b.col2 as bar_col2,
        b.strcol as bar_strcol
    from (
        select *
        from sql_semantics.foo
        inner join (
            select distinct FIRST_VALUE(id) over wf as first_row_id
            from sql_semantics.foo
            window wf as (
                partition by foo.col1, foo.col2
                order by foo.id
            )
        ) as foo_ids
        on
            foo.id = foo_ids.first_row_id
    ) as f
    full outer join (
        select *
        from sql_semantics.bar
        inner join (
            select distinct LAST_VALUE(id) over wb as last_row_id
            from sql_semantics.bar
            window wb as (
                partition by bar.col1, bar.col2
                order by bar.id
                rows between unbounded preceding and unbounded following
            )
        ) as bar_ids
        on
            bar.id = bar_ids.last_row_id
    ) as b
    on
        f.col1 = b.col1
        and
        f.col2 = b.col2
    order by f.id, b.id asc;
    """

    foo, bar = TestDataset.sql_semantics2()

    result = rt.merge2(
        foo,
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        how='outer',
        suffixes=('_foo', '_bar'),
        keep=('first', 'last'),
        indicator=True,
    )

    assert result.get_nrows() == 17

    # "foo_id","foo_col1","foo_col2","foo_teamname","bar_id","bar_col1","bar_col2","bar_strcol"
    # 1,5,NULL,"Phillies",NULL,NULL,NULL,NULL
    # 2,5,5,"Eagles",11,5,5,"Arch"
    # 3,8,NULL,"76ers",NULL,NULL,NULL,NULL
    # 4,NULL,1,"Flyers",NULL,NULL,NULL,NULL
    # 5,10,1,"Union",NULL,NULL,NULL,NULL
    # 6,NULL,4,"Wings",NULL,NULL,NULL,NULL
    # 7,-1,22,"Fusion",NULL,NULL,NULL,NULL
    # 8,11,9,"Fight",NULL,NULL,NULL,NULL
    # NULL,NULL,NULL,NULL,1,10,4,"Chestnut"
    # NULL,NULL,NULL,NULL,2,10,NULL,"Pine"
    # NULL,NULL,NULL,NULL,3,8,NULL,"Walnut"
    # NULL,NULL,NULL,NULL,4,NULL,3,"Locust"
    # NULL,NULL,NULL,NULL,6,NULL,NULL,"Spruce"
    # NULL,NULL,NULL,NULL,7,NULL,1,"Cypress"
    # NULL,NULL,NULL,NULL,9,5,NULL,"Sansom"
    # NULL,NULL,NULL,NULL,10,14,9,"Market"
    # NULL,NULL,NULL,NULL,12,-15,13,"Vine"

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(result.col1, rt.FA([5, 5, 8, inv, 10, inv, -1, 11, 10, 10, 8, inv, inv, inv, 5, 14, -15], dtype=np.int32))
    assert_array_equal(result.col2, rt.FA([inv, 5, inv, 1, 1, 4, 22, 9, 4, inv, inv, 3, inv, 1, inv, 9, 13], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(result.id_foo, rt.FA([1, 2, 3, 4, 5, 6, 7, 8, inv, inv, inv, inv, inv, inv, inv, inv, inv], dtype=np.int32))
    assert_array_equal(
        result.team_name,
        rt.FA([b'Phillies', b'Eagles', b'76ers', b'Flyers', b'Union', b'Wings', b'Fusion', b'Fight', b'', b'', b'', b'', b'', b'', b'', b'', b''])
    )

    # Cols from the right Dataset.
    assert_array_equal(result.id_bar, rt.FA([inv, 11, inv, inv, inv, inv, inv, inv, 1, 2, 3, 4, 6, 7, 9, 10, 12], dtype=np.int32))
    assert_array_equal(
        result.strcol,
        rt.FA([b'', b'Arch', b'', b'', b'', b'', b'', b'', b'Chestnut', b'Pine', b'Walnut', b'Locust', b'Spruce', b'Cypress', b'Sansom', b'Market', b'Vine'])
    )


@pytest.mark.parametrize("how", [
    'left', 'right', 'inner',
    pytest.param('outer', marks=xfail_rip260_outermerge_left_keep)
])
@pytest.mark.parametrize("keep_left", [None, pytest.param('first'), pytest.param('last')])
@pytest.mark.parametrize("keep_right", [None, pytest.param('first'), pytest.param('last')])
def test_merge2_with_unused_categories(how, keep_left, keep_right):
    x = rt.Dataset({
        'k1': rt.Categorical(['a', 'b', 'c', 'b', 'a', 'a', 'c'], ['a', 'b', 'c', 'd']),
        'a': [1, 2, 3, 4, 5, 6, 7]
    })
    y = rt.Dataset({
        'k1': rt.Categorical(['b', 'c', 'd', 'd', 'b', 'd', 'c', 'b', 'f', 'c'], ['a', 'b', 'c', 'd', 'e', 'f']),
        'b': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    })

    # For now, just check the function succeeds without failing.
    # TODO: Add more-robust tests that verify the result to make sure it's correct.
    _ = rt.merge2(x, y, on=['k1'], how=how, keep=(keep_left, keep_right))


def test_merge2_outer_with_unused_categories():
    x = rt.Dataset({
        'k1': rt.Categorical(['a', 'b', 'c', 'b', 'a', 'a', 'c'], ['a', 'b', 'c', 'd']),
        'a': [1, 2, 3, 4, 5, 6, 7]
    })
    y = rt.Dataset({
        'k1': rt.Categorical(['b', 'c', 'd', 'd', 'b', 'd', 'c', 'b', 'f', 'c'], ['a', 'b', 'c', 'd', 'e', 'f']),
        'b': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    })

    result = rt.merge2(x, y, on=['k1'], how='outer')
    assert result.get_nrows() == 19


def test_merge2_outer_overlapping_on_column_has_common_type():
    x = rt.Dataset({'k1': rt.FA(['a', 'b', 'c'], unicode=True), 'a': [1, 2, 3]})
    y = rt.Dataset({'k1': rt.FA(['b', 'c', 'dd']), 'b': [2, 3, 4]})

    assert x.k1.dtype.kind != y.k1.dtype.kind
    assert x.k1.dtype.itemsize != y.k1.dtype.itemsize

    result = rt.merge2(x, y, on=['k1'], how='outer')
    assert result.get_nrows() == 4

    assert result.k1.dtype == np.dtype('<U2')


def test_merge2_rowcount_is_sentinel():
    # Test case from RIP-466.

    N = rt.uint8.inv
    symbols = N * ['A']
    left_ds = rt.Dataset(dict(symbol=symbols))
    right_ds = rt.Dataset(dict(symbol=sorted(set(symbols))))
    right_ds['foo'] = 1.0

    # This should succeed without raising an exception.
    # Prior to a fix being implemented for this case, _build_left_fancyindex() broke here
    # due to incorrectly handling the case of the row count being a sentinel value.
    result = rt.merge2(left_ds, right_ds, on=['symbol'], validate='m:1')

    # Verify the result has the expected number of rows.
    assert len(result) == N

    # Verify the 'foo' column from the RIGHT dataset doesn't have any NaNs.
    # This is a check to make sure _build_right_fancyindex() also handled this case correctly.
    # It's failure would be a bit more subtle there because for a left join it's allowed (and typical)
    # that the right fancy index will have invalids due to missing keys from the left Dataset.
    # This already worked (prior to _build_right_fancyindex() getting the fix) but would have been
    # broken if/when cumsum was modified to respect integer invalids/NA values.
    assert rt.all(result['foo'].isnotnan())


@pytest.mark.parametrize("on", [
    ['f', 'g'],
    ['g', 'f']
])
def test_merge2_outer_on_multikey_with_string(on):
    ds1 = rt.Dataset({'f': ['1', '2', '3', '4'], 'g': [1, 2, 3, 4], 'h1': [1, 1, 1, 1]})
    ds2 = rt.Dataset({'f': ['2', '3', '4', '5'], 'g': [2, 3, 4, 5], 'h2': [2, 2, 2, 2]})

    # Just verifying this succeeds (and does not fail with an exception/error).
    # In riptable 1.0.21 and earlier, this failed due to the _create_column_valid_mask() function
    # returning None; the None was passed into get_or_create_keygroup(), which then choked
    # when trying to call .copy() on the None.
    _ = ds1.merge2(ds2, on=on, how='outer')


def test_lookup_copy_when_right_already_unique():
    """
    Test case for RIP-491.

    Verifies that the 'copy' parameter behaves as expected when the right
    Dataset is already a unique key (i.e. keep does not need to be specified).
    """

    left_ds = rt.Dataset()
    left_ds['Symbol'] = rt.FA(['AAPL', 'AMZN', 'AA', 'AAN', 'AAPL', 'AMZN', 'AAP', 'AMZN'])
    # "SpecialDate" isn't special; it could have been called Date but I
    # didn't want anyone to confuse the use of the Date array type with
    # the column name here but I couldn't come up with a better name.
    left_ds['SpecialDate'] = rt.Date(['2020-06-22', '2020-06-22', '2020-06-22', '2020-06-22', '2020-06-23', '2020-06-23', '2020-06-23', '2020-06-24'])

    right_ds = rt.Dataset()
    right_ds['Symbol'] = rt.FA(['AA', 'AAN', 'AAP', 'AAPL', 'AMZN'])
    right_ds['SharesOutstanding'] = rt.FA([185.92e6, 67.57e6, 69.1e6, 4.334e9, 498.8e6])

    # Call merge_lookup, using the Symbol column as the merge key;
    # specify copy=False so the columns from left_ds should be copied over without being modified.
    result = rt.merge_lookup(left_ds, right_ds, on='Symbol', copy=False)

    # The columns from left_ds present in the result should have all been copied over.
    # That may mean they've been wrapped in a read-only view, so do the check in a way
    # that accounts for that possibility.
    for col_name in left_ds.keys():
        left_col = left_ds[col_name]
        result_col = result[col_name]

        # Check that the original and result columns point to the same physical memory.
        assert np.may_share_memory(left_col, result_col),\
            f"Column '{col_name}' does not share the same physical memory as the original array."

        # Also verify that the logic used for copy=False to create the array view
        # preserves the array subclass (e.g. rt.Date).
        assert type(left_col) == type(result_col)


def test_merge_lookup_fails_with_nonunique_right_keys():
    """Check that merge_lookup() fails when the right Dataset has keys with multiplicity > 1."""
    foo, bar = TestDataset.sql_semantics()

    with pytest.raises(ValueError):
        rt.rt_merge.merge_lookup(
            foo,
            bar,
            on=('col1', 'col1'),
            require_match=True,
            suffixes=('_foo', '_bar'),
        )


def test_merge_lookup_require_match():
    """Check that the 'require_match' parameter of merge_lookup() is respected when specified."""
    foo, bar = TestDataset.sql_semantics()

    with pytest.raises(ValueError):
        rt.rt_merge.merge_lookup(
            foo,
            bar,
            on=('col1', 'col1'),
            require_match=True,
            suffixes=('_foo', '_bar'),
            keep='first'
        )


def test_merge_lookup_inplace_single():
    foo, bar = TestDataset.sql_semantics()

    foo.merge_lookup(
        bar,
        on=('col1', 'col1'),
        suffix='_bar',
        keep='first',
        inplace=True
    )

    assert foo.get_nrows() == 6

    # +--------+----------+----------+--------+----------+----------+------------+
    # |     id |     col1 |     col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      1 |        5 |     NULL |      8 |        5 |        5 | Lombard    |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # |      3 |        8 |     NULL |      3 |        8 |     NULL | Walnut     |
    # |      4 |     NULL |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      5 |       10 |        1 |      1 |       10 |        4 | Chestnut   |
    # |      6 |     NULL |        4 |   NULL |     NULL |     NULL | NULL       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 6 rows in set (0.0009 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(foo.col1, rt.FA([5, 5, 8, inv, 10, inv], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(foo.id, rt.FA([1, 2, 3, 4, 5, 6], dtype=np.int32))
    assert_array_equal(foo.col2, rt.FA([inv, 5, inv, 1, 1, 4], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(foo.id_bar, rt.FA([8, 8, 3, inv, 1, inv], dtype=np.int32))
    assert_array_equal(foo.col2_bar, rt.FA([5, 5, inv, inv, 4, inv], dtype=np.int32))
    assert_array_equal(
        foo.strcol, rt.FA([b'Lombard', b'Lombard', b'Walnut', b'', b'Chestnut', b'']))


def test_merge_lookup_inplace_multi():
    foo, bar = TestDataset.sql_semantics()

    foo.merge_lookup(
        bar,
        on=[('col1', 'col1'), ('col2', 'col2')],
        suffix='_bar',
        keep='first',
        inplace=True
    )

    assert foo.get_nrows() == 6

    # +--------+----------+----------+--------+----------+----------+------------+
    # | foo_id | foo_col1 | foo_col2 | bar_id | bar_col1 | bar_col2 | bar_strcol |
    # +--------+----------+----------+--------+----------+----------+------------+
    # |      1 |        5 |     NULL |   NULL |     NULL |     NULL | NULL       |
    # |      2 |        5 |        5 |      8 |        5 |        5 | Lombard    |
    # |      3 |        8 |     NULL |   NULL |     NULL |     NULL | NULL       |
    # |      4 |     NULL |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      5 |       10 |        1 |   NULL |     NULL |     NULL | NULL       |
    # |      6 |     NULL |        4 |   NULL |     NULL |     NULL | NULL       |
    # +--------+----------+----------+--------+----------+----------+------------+
    # 6 rows in set (0.0016 sec)

    inv = rt.int32.inv
    # Intersection cols (the 'on' cols)
    assert_array_equal(foo.col1, rt.FA([5, 5, 8, inv, 10, inv], dtype=np.int32))
    assert_array_equal(foo.col2, rt.FA([inv, 5, inv, 1, 1, 4], dtype=np.int32))

    # Cols from the left Dataset.
    assert_array_equal(foo.id, rt.FA([1, 2, 3, 4, 5, 6], dtype=np.int32))

    # Cols from the right Dataset.
    assert_array_equal(
        foo.id_bar, rt.FA([inv, 8, inv, inv, inv, inv], dtype=np.int32)
    )
    assert_array_equal(foo.strcol, rt.FA([b'', b'Lombard', b'', b'', b'', b'']))


class MergeAsofTest(unittest.TestCase):
    def test_merge_asof(self):
        def check_merge_asof(ds, ds1, ds2):
            # print(ds)
            self.assertIsInstance(ds, rt.Dataset)
            self.assertEqual(ds.shape[0], ds1.shape[0])
            self.assertEqual(ds.shape[1], ds1.shape[1] + ds2.shape[1])
            assert_array_equal(ds.A._np, ds1.A._np)
            assert_array_equal(ds.left_val._np, ds1.left_val._np)
            assert_array_equal(ds.right_val._np, ds.X._np)

        ds1 = rt.Dataset(
            {'A': [1, 5, 10], 'left_val': ['a', 'b', 'c'], 'left_grp': [1, 1, 1]}
        )
        ds2 = rt.Dataset(
            {
                'X': [1, 2, 3, 6, 7],
                'right_val': [1, 2, 3, 6, 7],
                'right_grp': [1, 1, 1, 1, 1],
            }
        )

        ds = rt.Dataset.merge_asof(
            ds1,
            ds2,
            left_on='A',
            right_on='X',
            left_by='left_grp',
            right_by='right_grp',
        )
        check_merge_asof(ds, ds1, ds2)
        assert_array_equal(ds.left_grp._np, ds.right_grp._np)
        assert_array_compare(operator.__ge__, ds.A, ds.X)

        ds = rt.Dataset.merge_asof(
            ds1,
            ds2,
            left_on='A',
            right_on='X',
            left_by='left_grp',
            right_by='right_grp',
            direction='backward',
        )
        check_merge_asof(ds, ds1, ds2)
        assert_array_equal(ds.left_grp._np, ds.right_grp._np)
        assert_array_compare(operator.__ge__, ds.A, ds.X)

        ds = rt.Dataset.merge_asof(
            ds1,
            ds2,
            left_on='A',
            right_on='X',
            left_by='left_grp',
            right_by='right_grp',
            allow_exact_matches=False,
        )
        check_merge_asof(ds, ds1, ds2)
        assert_array_equal(ds.left_grp._np[1:], ds.right_grp._np[1:])
        assert_array_compare(operator.__gt__, ds.A[1:], ds.X[1:])

        ds = rt.Dataset.merge_asof(
            ds1,
            ds2,
            left_on='A',
            right_on='X',
            left_by='left_grp',
            right_by='right_grp',
            direction='forward',
        )
        check_merge_asof(ds, ds1, ds2)
        assert_array_equal(ds.left_grp._np[:-1], ds.right_grp._np[:-1])
        assert_array_compare(operator.__le__, ds.A[:-1], ds.X[:-1])

        ds = rt.Dataset.merge_asof(
            ds1,
            ds2,
            left_on='A',
            right_on='X',
            left_by='left_grp',
            right_by='right_grp',
            allow_exact_matches=False,
            direction='forward',
        )
        check_merge_asof(ds, ds1, ds2)
        assert_array_equal(ds.left_grp._np[:-1], ds.right_grp._np[:-1])
        assert_array_compare(operator.__lt__, ds.A[:-1], ds.X[:-1])

        # ds = rt.Dataset.merge_asof(ds1, ds2, left_on='A', right_on='X', left_by='left_grp',
        #                            right_by='right_grp', direction='nearest')
        # check_merge_asof(ds, ds1, ds2)
        # for i in range(0, ds.shape[0]):
        #    diff = abs(ds.A[i] - ds.X[i])
        #    mindiff = min(ds.A - ds.X)
        #    self.assertEqual(diff, mindiff, diff + ", " + mindiff);
        # self.assertTrue((ds.left_grp._np == ds.right_grp._np).all())

    def test_merge_asof_categorical_and_string_keys(self):
        ds1 = rt.Dataset()
        ds1['Time'] = [0, 1, 4, 6, 8, 9, 11, 16, 19, 30]
        ds1['Px'] = [10, 12, 15, 11, 10, 9, 13, 7, 9, 10]

        ds2 = rt.Dataset()
        ds2['Time'] = [0, 0, 5, 7, 8, 10, 12, 15, 17, 20]
        ds2['Vols'] = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

        target = rt.Dataset()
        target.Time = ds1.Time
        target.Px = ds1.Px
        target.Vols = [20, 20, 20, 22, 24, 24, 24, 26, 28, 28]
        target.matched_on = [0, 0, 0, 5, 8, 8, 8, 12, 17, 17]   # For seeing which time was matched against (from the right side) for each row

        # Categorical keys
        ds1['Ticker'] = rt.Categorical(['Test'] * 10)
        ds2['Ticker'] = rt.Categorical(['Test', 'Blah'] * 5)
        target.Ticker = ds1.Ticker

        ds = ds1.merge_asof(ds2, on='Time', by='Ticker', matched_on=True)

        for key in ds.keys():
            # TODO: Switch to assert_array_equal here once a string-based Categorical allows np.inf to be used in an equality check
            #assert_array_equal(ds[key], target[key], err_msg=f"Column '{key}' differs between the actual and expected.")
            self.assertTrue((ds[key] == target[key]).all())

        # String keys
        ds1['Ticker'] = rt.FastArray(['Test'] * 10)
        ds2['Ticker'] = rt.FastArray(['Test', 'Blah'] * 5)
        target.Ticker = ds1.Ticker

        ds = ds1.merge_asof(ds2, on='Time', by='Ticker')

        for key in ds.keys():
            assert_array_equal(ds[key], target[key], err_msg=f"Column '{key}' differs between the actual and expected.")

    def test_merge_asof_nearest(self):
        ds1 = rt.Dataset()
        ds1['Time'] = [0, 1, 4, 6, 8, 9, 11, 16, 19, 30]
        ds1['Px'] = [10, 12, 15, 11, 10, 9, 13, 7, 9, 10]

        ds2 = rt.Dataset()
        ds2['Time'] = [0, 5, 7, 8, 10, 12, 15, 17, 20]
        ds2['Vols'] = [20, 22, 23, 24, 25, 26, 27, 28, 29]

        target = rt.Dataset()
        target.Time = ds1.Time
        target.Px = ds1.Px
        target.Vols = [20, 20, 22, 22, 24, 24, 25, 27, 29, 29]
        target.Time_y = [0, 0, 5, 5, 8, 8, 10, 15, 20, 20]  # For seeing which time was matched against (from the right side) for each row

        merged = ds1.merge_asof(ds2, on='Time', direction='nearest', matched_on='Time_y')

        for key in merged.keys():
            assert_array_equal(merged[key], target[key], err_msg=f"Column '{key}' differs between the actual and expected.")


@pytest.mark.parametrize(
    "left,right,on,check_sorted,should_fail",
    [
        pytest.param(
            rt.Dataset(
                {'A': [1, 5, 10], 'left_val': ['a', 'b', 'c'], 'left_grp': [1, 1, 1]}
            ),
            rt.Dataset(
                {
                    'X': [1, 2, 3, 6, 7],
                    'right_val': [1, 2, 3, 6, 7],
                    'right_grp': [1, 1, 1, 1, 1],
                }
            ),
            ('A', 'X'),
            True,
            False,
            id="sorted,sorted,check",
        ),
        pytest.param(
            rt.Dataset(
                {'A': [1, 5, 10], 'left_val': ['a', 'b', 'c'], 'left_grp': [1, 1, 1]}
            ),
            rt.Dataset(
                {
                    'X': [1, 2, 3, 6, 7],
                    'right_val': [1, 2, 3, 6, 7],
                    'right_grp': [1, 1, 1, 1, 1],
                }
            ),
            ('A', 'X'),
            False,
            False,
            id="sorted,sorted,nocheck",
        ),
        pytest.param(
            rt.Dataset(
                {'A': [1, 10, 5], 'left_val': ['a', 'b', 'c'], 'left_grp': [1, 1, 1]}
            ),
            rt.Dataset(
                {
                    'X': [1, 2, 3, 6, 7],
                    'right_val': [1, 2, 3, 6, 7],
                    'right_grp': [1, 1, 1, 1, 1],
                }
            ),
            ('A', 'X'),
            True,
            True,
            id="unsorted,sorted,check",
        ),
        pytest.param(
            rt.Dataset(
                {'A': [1, 10, 5], 'left_val': ['a', 'b', 'c'], 'left_grp': [1, 1, 1]}
            ),
            rt.Dataset(
                {
                    'X': [1, 2, 3, 6, 7],
                    'right_val': [1, 2, 3, 6, 7],
                    'right_grp': [1, 1, 1, 1, 1],
                }
            ),
            ('A', 'X'),
            False,
            False,
            id="unsorted,sorted,nocheck",
        ),
        pytest.param(
            rt.Dataset(
                {'A': [1, 5, 10], 'left_val': ['a', 'b', 'c'], 'left_grp': [1, 1, 1]}
            ),
            rt.Dataset(
                {
                    'X': [1, 2, 3, 6, 4],
                    'right_val': [1, 2, 3, 6, 7],
                    'right_grp': [1, 1, 1, 1, 1],
                }
            ),
            ('A', 'X'),
            True,
            True,
            id="sorted,unsorted,check",
        ),
        pytest.param(
            rt.Dataset(
                {'A': [1, 5, 10], 'left_val': ['a', 'b', 'c'], 'left_grp': [1, 1, 1]}
            ),
            rt.Dataset(
                {
                    'X': [1, 2, 3, 6, 4],
                    'right_val': [1, 2, 3, 6, 7],
                    'right_grp': [1, 1, 1, 1, 1],
                }
            ),
            ('A', 'X'),
            False,
            False,
            id="sorted,unsorted,nocheck",
        ),
    ],
)
def test_merge_asof_unsorted_fails(
    left: rt.Dataset,
    right: rt.Dataset,
    on: Union[str, Tuple[str, str]],
    check_sorted: bool,
    should_fail: bool,
):
    """
    Test for verifying 'merge_asof' checks the sortedness of the 'on' columns before merging.
    """
    # Unpack the 'on' parameter (if needed) until 'merge_asof' has been
    # modified to allow a tuple to be passed for the 'on' parameter (instead of left_on+right_on).
    left_on, right_on = on if isinstance(on, tuple) else (on, on)

    if should_fail:
        with pytest.raises(ValueError):
            rt.merge_asof(
                left,
                right,
                left_on=left_on,
                right_on=right_on,
                check_sorted=check_sorted,
                verbose=True,
            )
    else:
        # Just check that this succeeds without raising any errors;
        # we otherwise don't care about the result of the operation.
        rt.merge_asof(
            left,
            right,
            left_on=left_on,
            right_on=right_on,
            check_sorted=check_sorted,
            verbose=True,
        )

if __name__ == "__main__":
    tester = unittest.main()
