import os
import pathlib
import sys
from typing import Optional, Sequence

from numpy.random import default_rng
import riptide_cpp as rc
import riptable as rt

from riptable.testing.array_assert import assert_array_or_cat_equal
from riptable.testing.randgen import create_test_dataset

import pytest

class TestSDS:
    """Unit tests for functions in the `rt_sds` module."""

    @pytest.mark.parametrize('rng_seed', [458973497])
    @pytest.mark.parametrize('include_extension', [False, True])
    def test_sds_info_single(self, tmp_path, rng_seed: int, include_extension: bool) -> None:
        """Test calling `rt.sds_info()` with a single path."""
        rng = default_rng(rng_seed)

        # Generate a test dataset.
        test_ds = create_test_dataset(rng=rng, rowcount=19)

        # Save the test dataset.
        test_ds_path = tmp_path / 'test_ds.sds'
        test_ds.save(test_ds_path)

        # Call rt.sds_info(), passing in the path where the test dataset was saved.
        if not include_extension:
            test_ds_path = test_ds_path.parent / test_ds_path.stem
        test_ds_infos = rt.sds_info(test_ds_path)
        assert isinstance(test_ds_infos, Sequence)
        assert len(test_ds_infos) == 1
        # TEMP: Disabled until riptide_cpp CI pypi builds are fixed, otherwise this next line breaks in CI
        #       when running with riptide_cpp <1.6.28
        #assert isinstance(test_ds_infos[0], rc.sds_file_info)

        # TODO: Add some additional assertions -- check that `test_ds_info` has info for
        #       all of the columns in the test dataset, that the column shapes are correct
        #       (i.e. match the test dataset's .get_nrows()).

    @pytest.mark.parametrize('rng_seed', [199457824])
    def test_sds_info_multi(self, tmp_path, rng_seed: int) -> None:
        """Test calling `rt.sds_info()` with a sequence of multiple paths."""
        rng = default_rng(rng_seed)

        # Choose some number of datasets to generate, then generate a rowcount for
        # each of the datasets that'll be generated.
        dataset_count = rng.integers(low=3, high=11)
        dataset_rowcounts = [int(x) for x in rng.integers(low=7, high=23, size=dataset_count)]

        # Generate the test datasets.
        test_dss = [create_test_dataset(rng=rng, rowcount=rowcount) for rowcount in dataset_rowcounts]

        # Save the test datasets.
        test_ds_paths = [tmp_path / f'test_ds-{i}.sds' for i in range(dataset_count)]
        for test_ds, test_ds_path in zip(test_dss, test_ds_paths):
            test_ds.save(test_ds_path)

        # Call rt.sds_info(), passing in the paths to the test datasets on-disk.
        test_ds_infos = rt.sds_info(test_ds_paths)
        assert isinstance(test_ds_infos, Sequence)
        assert len(test_ds_infos) == dataset_count
        # TEMP: Disabled until riptide_cpp CI pypi builds are fixed, otherwise this next line breaks in CI
        #       when running with riptide_cpp <1.6.28
        #assert all([isinstance(x, rc.sds_file_info) for x in test_ds_infos])

        # TODO: Add some additional assertions -- check that each info in `test_ds_infos` corresponds
        #       to the same test dataset (in the same position in the input path list), that it has info for
        #       all of the columns in the corresponding test dataset, that the column shapes are correct
        #       (i.e. match the corresponding test dataset's .get_nrows()).
