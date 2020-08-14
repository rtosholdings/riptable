import pytest
import numpy as np
import riptable as rt
from riptable.rt_enum import CategoryMode
from riptable import save_sds, load_sds
from riptable.Utils.rt_testing import assert_array_equal_, name, get_size
from hypothesis import given, settings, Verbosity
from hypothesis.strategies import integers
import hypothesis.extra.pandas as pdst
from hypothesis.extra.numpy import arrays
from .strategies.helper_strategies import (
    ndarray_shape_strategy,
    ints_or_floats_dtypes,
    generate_array,
    column_by_dtypes,
    one_of_supported_dtypes,
)
from .strategies.categorical_strategy import CategoricalStrategy


def assert_save_load(loaded: object, saved: object, err_msg: str = "") -> None:
    """Common assertions across saved and loaded objects.

    Assertions:
    The identity of the saved and loaded objects should be unique.
    The size, in bytes, of the saved and loaded object should be the same.
    The shape and size should be the same.
    """

    try:
        assert id(loaded) != id(
            saved
        ), f"Identity of saved {name(saved)} should be different from the loaded."
        # TODO - follow up with addition of xfail for falsifying examples that do not preserve the object types
        # Saved ndarray gets loaded as a FastArray
        # Saved Multiset gets loaded as a Struct

        if not isinstance(saved, (np.ndarray, rt.Multiset,)):
            assert type(loaded) == type(
                saved
            ), f"Identity of saved {name(saved)} should be different from the loaded {name(loaded)}."
            assert get_size(loaded) == get_size(
                saved
            ), f"Size in bytes of saved {name(saved)} should equal size of the loaded {name(loaded)}."
            if hasattr(saved, "shape"):
                assert (
                    loaded.shape == saved.shape
                ), f"Shapes should be the same.\nSaved {name(saved)}\n{repr(saved)}\nLoaded {name(loaded)}\n{loaded}"
            if hasattr(saved, "size"):
                assert (
                    loaded.size == saved.size
                ), f"Size of saved {name(saved)} should equal size of loaded {name(loaded)}."
    except AssertionError:
        # Display all columns of the saved and loaded objects
        rt.rt_enum.TypeRegister.DisplayOptions.COL_ALL = True
        err_msg += f"\nsaved {name(saved)}\n{repr(saved)}\nloaded {name(loaded)}\n{repr(loaded)}\n"
        print(f"{err_msg}")
        raise


# perform a parameter sweep across shared memory and other arguments
class TestSaveLoad:
    # Some of the following tests have deadline checks turned off since these tests involve reading, writing,
    # doing comprehensive size validations using all the referents, and containing multiple test
    # cases under one test method.
    # The reason we have multiple test cases in one test method is so re-use of Hypothesis generated data.
    # Alternatives to this approach would be to cache the generated data and split these test cases out which
    # would remove the need for turning off the deadline setting.
    @settings(deadline=None)
    @given(
        arr=generate_array(
            shape=ndarray_shape_strategy(),
            dtype=ints_or_floats_dtypes(),
            include_invalid=False,
        )
    )
    def test_save_load_array(self, arr, tmpdir):
        # Test #1: save and load of ndarray
        fn = str(tmpdir.join(name(arr)))
        save_sds(fn, arr)
        arr2 = load_sds(fn)

        assert_save_load(arr2, arr)
        assert_array_equal_(arr2, arr)

        # Test #2: save and load of FastArray derived from ndarray
        f_arr = rt.FA(arr)
        save_sds(fn, f_arr)
        f_arr2 = load_sds(fn)

        assert_array_equal_(f_arr2, f_arr)

    # TODO fold this test case in the Dataset test case below
    @settings(deadline=None)
    @given(
        arr=generate_array(
            shape=ndarray_shape_strategy(),
            dtype=ints_or_floats_dtypes(),
            include_invalid=False,
        )
    )
    def test_save_load_dataset_array(self, arr, tmpdir):
        # Test #1: save and load of ndarray within Dataset
        fn = str(tmpdir.join(name(arr)))

        ds = rt.Dataset({name(arr): arr})

        ds.save(fn)
        ds2 = rt.Dataset.load(fn)

        assert_save_load(ds2, ds)
        assert_array_equal_(ds2[name(arr)], ds[name(arr)])

        # Test #2: save and load of FastArray derived from ndarray within Dataset
        f_arr = rt.FA(arr)
        fn = str(tmpdir.join(name(f_arr)))

        ds = rt.Dataset({name(f_arr): f_arr})
        ds.save(fn)
        ds2 = rt.Dataset.load(fn)

        assert_save_load(ds2, ds)
        assert_array_equal_(ds[name(f_arr)], ds2[name(f_arr)])

    # TODO add columns for ndarray, FastArray, Categorical, Struct, and nested DataSets
    @settings(deadline=None)
    @given(dataframe=pdst.data_frames(column_by_dtypes()))
    def test_save_load_datasets(self, dataframe, tmpdir):
        # generate a dataframe of all the dtypes
        # all array types
        # copy itself and create nested datasets and sibling datasets

        # Test #1: save and load of DataFrame
        fn = str(tmpdir.join(name(dataframe)))
        # save_sds(fn, dataframe)
        # dataframe2 = load_sds(fn)
        # assert dataframe2 == dataframe
        # E TypeError: save_sds() can only save Structs, Datasets, or single arrays. Got <class 'pandas.core.frame.DataFrame'>
        # ..\rt_sds.py:470: TypeError

        # Test #2: save and load of Dataset created from DataFrame
        dataset = rt.Dataset(dataframe)

        save_sds(fn, dataset)
        dataset2 = load_sds(fn)

        assert_save_load(dataset2, dataset)
        for f_arr1, f_arr2 in zip(dataset.values(), dataset2.values()):
            assert_array_equal_(f_arr2._np, f_arr1._np)

        # Test #3: save and load nested Dataset within a Multiset
        # This also tests that shallow and deep copies that are saved and loaded from SDS
        # are both unique objects with the same size footprint.
        multiset = rt.Multiset()
        shallow_copy_name, deep_copy_name = "dataset_shallow_copy", "dataset_deep_copy"
        dataset_shallow_copy, dataset_deep_copy = (
            dataset.copy(deep=False),
            dataset.copy(deep=True),
        )
        multiset[shallow_copy_name], multiset[deep_copy_name] = (
            dataset_shallow_copy,
            dataset_deep_copy,
        )

        fn = str(tmpdir.join(name(multiset)))
        save_sds(fn, multiset)
        multiset2 = load_sds(fn)

        assert_save_load(multiset2, multiset)
        # Shallow copy assertions
        assert id(multiset[shallow_copy_name]) != id(
            multiset2[shallow_copy_name]
        ), f"Identity of saved object should be different from the loaded object."
        for f_arr1, f_arr2 in zip(
            multiset[shallow_copy_name].values(), multiset2[shallow_copy_name].values()
        ):
            # Convert these to ndarrays so we don't need to consider Riptable invalid checks.
            # This test is concerned with ensuring the same data is loaded as saved.
            assert_save_load(f_arr2, f_arr1)
            assert_array_equal_(f_arr2._np, f_arr2._np)

        # Deep copy assertions
        assert id(multiset[deep_copy_name]) != id(
            multiset2[deep_copy_name]
        ), f"Identity of saved object should be different from the loaded object."
        for f_arr1, f_arr2 in zip(
            multiset[deep_copy_name].values(), multiset2[deep_copy_name].values()
        ):
            assert_save_load(f_arr2, f_arr1)
            assert_array_equal_(f_arr2._np, f_arr2._np)

    @settings(deadline=None, verbosity=Verbosity.verbose)
    @given(
        dataframe=pdst.data_frames(column_by_dtypes()),
        stack_count=integers(min_value=0, max_value=10),
    )
    @pytest.mark.parametrize("stack", [True, False])
    def test_stack_save_load(self, dataframe, stack_count, tmpdir, stack):
        def assert_stack_equal(pds, ds, num_stack=1):
            assert id(pds) != id(
                ds
            ), f"Identity of saved {name(ds)} should be different from the loaded {name(ds)}."
            assert isinstance(pds, rt.PDataset), f"got type {type(pds)}"
            assert pds.shape == (
                num_stack * ds.shape[0],
                ds.shape[1],
            ), f"Shapes should be the same.\n{name(ds)}\n{repr(ds)}\n{name(pds)}\n{pds}"
            # TODO consider stacking
            # for f_arr1, f_arr2 in zip(pds.values(), ds.values()):
            #     assert_array_equal_(f_arr2._np, f_arr1._np)

        fn = str(tmpdir.join(name(dataframe)))

        ds = rt.Dataset(dataframe)
        save_sds(fn, ds)

        for i in range(stack_count):
            # expectations for empty input
            if i == 0:
                if stack:
                    with pytest.raises(ValueError):
                        _ = load_sds([fn] * i, stack=stack)
                else:
                    pds = load_sds([fn] * i, stack=stack)
                    assert isinstance(pds, type(None)), f"got type {type(pds)}"
                continue

            # expectations for n+1 input where n is a positive nonzero integer
            pds = load_sds([fn] * i, stack=stack)

            if stack:
                assert_stack_equal(pds, ds, num_stack=i)
            else:
                # handle expectations for non-stacked load
                assert isinstance(pds, list), f"got type {type(pds)}"

    # Categorical parameter sweep
    @pytest.mark.parametrize(
        "category_mode", [CategoryMode.StringArray, CategoryMode.Dictionary]
    )
    @pytest.mark.parametrize("with_categories", [True, False])
    @pytest.mark.parametrize("ordered", [True, False])
    def test_save_load_categorical(
        self, category_mode, with_categories, ordered, tmpdir
    ):
        # Categorical of with a value strategy from on of the supported data types
        @given(
            cat=CategoricalStrategy(
                arrays(shape=ndarray_shape_strategy(), dtype=one_of_supported_dtypes()),
                category_mode=category_mode,
                with_categories=with_categories,
                ordered=ordered,
            )
        )
        def inner(cat):
            # Test #1: save and load Categorical
            fn = str(tmpdir.join(name(cat)))

            save_sds(fn, cat)
            cat2 = load_sds(fn)

            assert_save_load(cat2, cat)
            assert cat == cat2

            # Test #2: save and load Categorical from within Dataset
            ds = rt.Dataset({name(cat): cat})

            ds.save(fn)
            ds2 = rt.Dataset.load(fn)

            assert_save_load(ds2, ds)
            assert ds[name(cat)] == ds2[name(cat)]
