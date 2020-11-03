"""Unit tests for conversions of arrays and Dataset/tables between riptable and pyarrow."""

import numpy as np
import riptable as rt

import pytest
from riptable.testing.array_assert import assert_array_equal, assert_array_or_cat_equal

# pyarrow is still an optional dependency;
# use the functionality of pytest to skip all tests in this module if pyarrow can't be imported.
# https://docs.pytest.org/en/latest/how-to/skipping.html#skipping-on-a-missing-import-dependency
_ = pytest.importorskip('pyarrow', minversion='4.0.0')
import pyarrow as pa
import pyarrow.types as pat
import pyarrow.compute as pc


class TestPyarrowConvertFastArray:
    @pytest.mark.parametrize(('rt_farr',), [
        pytest.param(rt.FA([], dtype=np.int8), id='empty(int8)'),
        pytest.param(rt.FA([-120, rt.int8.inv, -1, 0, 1, 101, 127], dtype=np.int8), id='int8'),
        pytest.param(rt.FA([0.01, -0.0, np.nan, 1e10, -1e10, np.inf, np.pi], dtype=np.float32), id='float32'),
        # bool
        # ascii string
        # unicode string
    ])
    def test_roundtrip_rt_pa_rt(self, rt_farr: rt.FastArray) -> None:
        """Test round-tripping from rt.FastArray to pyarrow.Array and back."""
        result_pa_arr = rt_farr.to_arrow()
        result_farr = rt.FastArray.from_arrow(result_pa_arr, zero_copy_only=False)
        assert_array_equal(rt_farr, result_farr)


class TestPyarrowConvertDate:
    @pytest.mark.parametrize(('rt_date_arr',), [
        pytest.param(rt.Date([]), id='empty'),
        # TODO: Add test cases
    ])
    def test_roundtrip_rt_pa_rt(self, rt_date_arr: rt.Date) -> None:
        """Test round-tripping from rt.Date to pyarrow.Array and back."""
        result_pa_arr = rt_date_arr.to_arrow()
        result_date_arr = rt.Date.from_arrow(result_pa_arr, zero_copy_only=False)
        assert_array_equal(rt_date_arr, result_date_arr)


class TestPyarrowConvertDateTimeNano:
    @pytest.mark.parametrize(('rt_dtn_arr',), [
        pytest.param(rt.DateTimeNano([]), id='empty'),
        # TODO: Add test cases -- including various timezones
    ])
    def test_roundtrip_rt_pa_rt(self, rt_dtn_arr: rt.Date) -> None:
        """Test round-tripping from rt.Date to pyarrow.Array and back."""
        result_pa_arr = rt_dtn_arr.to_arrow()
        result_dtn_arr = rt.DateTimeNano.from_arrow(result_pa_arr, zero_copy_only=False)
        assert_array_equal(rt_dtn_arr, result_dtn_arr)


class TestPyarrowConvertTimeSpan:
    @pytest.mark.parametrize(('rt_tsp_arr',), [
        pytest.param(rt.TimeSpan([]), id='empty'),
        # TODO: Add test cases
    ])
    def test_roundtrip_rt_pa_rt(self, rt_tsp_arr: rt.Date) -> None:
        """Test round-tripping from rt.Date to pyarrow.Array and back."""
        result_pa_arr = rt_tsp_arr.to_arrow()
        result_tsp_arr = rt.TimeSpan.from_arrow(result_pa_arr, zero_copy_only=False)
        assert_array_equal(rt_tsp_arr, result_tsp_arr)


class TestPyarrowConvertCategorical:
    @pytest.mark.parametrize(('rt_cat',), [
        # TODO: Add test cases for CategoryMode.IntEnum; at present, it appears IntEnum support is broken, can't seem to create a Categorical in that mode.
        # pytest.param(rt.Categorical([]), id='empty', marks=pytest.mark.skip(reason="rt.Categorical does not support creation from an empty list/array.")),
        pytest.param(rt.Categorical(['red', 'red', 'green', 'blue', 'green', 'red', 'blue'], ordered=False), id='CategoryMode.StringArray'),
        pytest.param(rt.Categorical(['red', 'red', 'green', 'blue', 'green', 'red', 'blue'], ordered=True), id='CategoryMode.StringArray--ordered'),
        pytest.param(rt.Categorical(['red', 'red', 'green', 'blue', 'green', 'red', 'blue'], dtype=np.int8, ordered=False), id='CategoryMode.StringArray;int8;ordered=False'),
        pytest.param(rt.Categorical(['red', 'red', 'green', 'blue', 'green', 'red', 'blue'], dtype=np.int8, ordered=True), id='CategoryMode.StringArray;int8;ordered=True'),
        pytest.param(rt.Categorical([f"x{i}" for i in range(0, 127)], dtype=np.int8), id="max number of categories for a signed int backing array without causing overflow"),
        # N.B. The test cases below for Categorical[Date] require pyarrow 5.0.0 or higher; dictionary-encoded date32() arrays didn't work before then.
        pytest.param(
            rt.Categorical(
                rt.Date(['2019-06-19', '2019-06-19', '2020-01-15', '2020-05-22', '2020-02-10', '2020-02-10', '2020-03-17', '2020-03-17']),
                ordered=False),
            id="Categorical[Date];ordered=False"),
        pytest.param(
            rt.Categorical(
                rt.Date(['2019-06-19', '2019-06-19', '2020-01-15', '2020-05-22', '2020-02-10', '2020-02-10', '2020-03-17', '2020-03-17']),
                ordered=True),
            id="Categorical[Date];ordered=True"
        ),
        pytest.param(rt.Categorical(
            # Country codes -- adapted from TestCategorical.test_hstack_fails_for_different_mode_cats.
            [36, 36, 344, 840, 840, 372, 840, 372, 840, 124, 840, 124, 36, 484],
            {
                'IRL': 372, 'USA': 840, 'AUS': 36, 'HKG': 344, 'JPN': 392,
                'MEX': 484, 'KHM': 116, 'THA': 764, 'JAM': 388, 'ARM': 51
            },
            ordered=False
        ), id="CategoryMode.Dictionary;ordered=False"),
        pytest.param(rt.Categorical(
            # Country codes -- adapted from TestCategorical.test_hstack_fails_for_different_mode_cats.
            [36, 36, 344, 840, 840, 372, 840, 372, 840, 124, 840, 124, 36, 484],
            {
                'IRL': 372, 'USA': 840, 'AUS': 36, 'HKG': 344, 'JPN': 392,
                'MEX': 484, 'KHM': 116, 'THA': 764, 'JAM': 388, 'ARM': 51
            },
            ordered=True
        ), id="CategoryMode.Dictionary;ordered=True"),
        pytest.param(rt.Categorical(
            [
                rt.FastArray(['Cyan', 'Magenta', 'Yellow', 'Black', 'Magenta', 'Cyan', 'Black', 'Yellow']).set_name('InkColor'),
                rt.Date(['2019-06-19', '2019-06-19', '2020-01-15', '2020-05-22', '2020-02-10', '2020-02-10', '2020-03-17', '2020-03-17']).set_name('CartridgeInstallDate')
            ]
        ), id="CategoryMode.MultiKey")
    ])
    @pytest.mark.parametrize('output_writable', [False, True])
    @pytest.mark.parametrize('have_nulls', [False, True])
    def test_roundtrip_rt_pa_rt(self, rt_cat: rt.Categorical, output_writable: bool, have_nulls: bool) -> None:
        """Test round-tripping from rt.Categorical to pyarrow.Array/pyarrow.Table and back."""
        orig_cat_shape = rt_cat.shape
        if have_nulls:
            # riptable's filtering/masking uses a valid mask (where False means null/NA).
            indices = np.arange(len(rt_cat))
            valid_mask = indices % 3 != 1
            rt_cat = rt_cat.filter(valid_mask)
            assert rt_cat.shape == orig_cat_shape

            # isfiltered() doesn't work as expected for Dictionary/IntEnum-mode Categorical as of riptable 1.1.0.
            filtered_element_count = (rt.isnan(rt_cat._fa) if rt_cat.category_mode in (rt.rt_enum.CategoryMode.Dictionary, rt.rt_enum.CategoryMode.IntEnum) else rt_cat.isfiltered()).sum()
            assert filtered_element_count == (len(rt_cat) - valid_mask.sum())

        result_pa_arr = rt_cat.to_arrow()

        # Verify the pyarrow array has the correct length, number of categories, etc.
        assert len(rt_cat) == len(result_pa_arr)
        assert pat.is_dictionary(result_pa_arr.type)
        assert len(result_pa_arr.dictionary) >= len(next(iter(rt_cat.category_dict.values()))), \
            "The number of categories in the pyarrow array's dictionary is smaller than the number of categories in the input Categorical."

        if have_nulls:
            assert valid_mask.sum() > 0
            assert (len(rt_cat) - valid_mask.sum()) == result_pa_arr.null_count

        # TEMP: Certain cases are marked as XFAIL here due to issues in Categorical.
        #         * Cannot create a pre-filtered (i.e. filtered at construction time) Dictionary- or IntEnum-mode Categorical.
        #         * Filtering a Dictionary- or IntEnum-mode Categorical causes unused categories to be dropped,
        #           which is not the same behavior as for other Categorical modes.
        #         * MultiKey Categoricals can't be created with an explicit list of category arrays + an index array,
        #           like what is supported for other Categorical modes.
        if rt_cat.category_mode == rt.rt_enum.CategoryMode.MultiKey or (have_nulls and rt_cat.category_mode == rt.rt_enum.CategoryMode.Dictionary):
            pytest.xfail("Expected failure due to issues with the Categorical constructor and/or filtering.")

        result_cat = rt.Categorical.from_arrow(result_pa_arr, zero_copy_only=False, writable=output_writable)

        # relaxed_cat_check <==> rt_cat.ordered, because if the categories are ordered, we expect them to be
        # in the same position after being roundtripped, so they should be mapped to the same integer before/after.
        # multi-key cats always seem to be ordered, even if ordered=False is specified when creating them.
        # TODO: Remove CategoryMode.Dictionary from the relaxed_cat_check here -- it's failing because our encoding in
        #       pyarrow doesn't currenly preserve unused entries from the name <-> code mapping. Once that's fixed
        #       we should be able to use the stronger equality check.
        assert_array_or_cat_equal(rt_cat, result_cat, relaxed_cat_check=rt_cat.ordered or rt_cat.category_mode == rt.rt_enum.CategoryMode.MultiKey or rt_cat.category_mode == rt.rt_enum.CategoryMode.Dictionary)

    @pytest.mark.parametrize(('num_cats', 'dtype'), [
        pytest.param(127, np.uint8),
        pytest.param(128, np.uint8),
        pytest.param(129, np.uint8),
        pytest.param(32769, np.uint16)
    ])
    @pytest.mark.parametrize('ordered', [False, True])
    @pytest.mark.parametrize('output_writable', [False, True])
    @pytest.mark.parametrize('have_nulls', [False, True])
    def test_pa_to_rt_unsigned(self, num_cats, dtype, ordered: bool, output_writable: bool, have_nulls: bool) -> None:
        # Create a numpy array containing `num_cats` distinct strings.
        cat_labels = np.array([f"x{i}" for i in range(0, num_cats)])
        indices = np.arange(num_cats, dtype=dtype)

        # Create the pyarrow dict-encoded array.
        if have_nulls:
            # pyarrow uses an INvalid mask (where True means null/NA).
            invalid_mask = indices % 7 == 3
            pa_indices = pa.array(indices, mask=pa.array(invalid_mask))
            pa_arr = pa.DictionaryArray.from_arrays(pa_indices, cat_labels, ordered=ordered)
        else:
            pa_arr = pa.DictionaryArray.from_arrays(indices, cat_labels, ordered=ordered)

        assert len(pa_arr.dictionary) == num_cats

        # Create the Categorical from the pyarrow array.
        result_cat = rt.Categorical.from_arrow(pa_arr, zero_copy_only=False, writable=output_writable)

        if have_nulls:
            result_invalid_mask = result_cat.isfiltered()
            assert_array_equal(result_invalid_mask, invalid_mask)

        # TODO: Add assertions here to verify correctness of `result_cat`.
        #   * Make sure we have same number of categories as input pyarrow arrray, even if not all categories (entries in the pyarrow array.dictionary) are used.


class TestPyarrowConvertDataset:
    @pytest.mark.parametrize(('rt_dset',), [
        pytest.param(rt.Dataset({}), id='empty'),
        # TODO: Add test cases
    ])
    def test_roundtrip_rt_pa_rt(self, rt_dset: rt.Dataset) -> None:
        """Test round-tripping from rt.Dataset to pyarrow.Table and back."""
        result_pa_tbl = rt_dset.to_arrow()
        result_rt_dset = rt.Dataset.from_arrow(result_pa_tbl, zero_copy_only=False)

        assert rt_dset.keys() == result_rt_dset.keys()
        for col_name in rt_dset.keys():
            # relaxed_cat_check=True, because we're not trying to test specific details of Categorical conversion
            # here, we're more interested in the dataset-level stuff.
            assert_array_or_cat_equal(rt_dset[col_name], result_rt_dset[col_name], relaxed_cat_check=True)
