import os
import pathlib
import shutil
import sys
import unittest
import tempfile
from typing import Optional

import pytest
from numpy.random import default_rng

import riptable as rt
from riptable import *
from riptable.rt_enum import CategoryMode, SDSFlag
from riptable.rt_sds import SDSMakeDirsOn
from riptable.testing.array_assert import assert_array_or_cat_equal
from riptable.testing.randgen import create_test_dataset
from riptable.tests.test_utils import get_all_categorical_data
from riptable.Utils.rt_metadata import MetaData
from riptable.Utils.rt_testdata import load_test_data
from riptable.Utils.rt_testing import (
    assert_array_equal_,
    assert_categorical_equal,
    name,
)

_TESTDIR = os.path.join(os.path.dirname(__file__), "test_files")

# change to true since we write into /tests directory
SDSMakeDirsOn()

arr_types = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64]


def arr_eq(a, b):
    return bool(np.all(a == b))


def arr_all(a):
    return bool(np.all(a))


class SaveLoad_Test(unittest.TestCase):
    def setUp(self) -> None:
        # TODO enable SDS verbose logging for local SDS pytest runs if the verbose flag is passed to pytest
        # See the pytest doc for how to do this dynamically
        # https://docs.pytest.org/en/latest/example/simple.html#pass-different-values-to-a-test-function-depending-on-command-line-options
        # SDSVerboseOn()
        pass

    def test_save_arrays(self):
        pass

    def test_save_numeric(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({dt.__name__: arange(10, dtype=dt) for dt in arr_types})
            ds.save(f"{tmpdirname}/tempsave")
            ds2 = Dataset.load(f"{tmpdirname}/tempsave")

            # name, column order matches
            loadkeys = list(ds2.keys())
            for i, k in enumerate(ds.keys()):
                self.assertEqual(loadkeys[i], k)

            # dtype
            for i, c in enumerate(ds2.values()):
                self.assertEqual(arr_types[i], c.dtype)

    def test_save_strings(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            uni = FastArray(["a", "b", "c", "d", "e"], unicode=True)
            b = FastArray(["a", "b", "c", "d", "e"])
            ds = Dataset({"unicode": uni, "bytes": b}, unicode=True)
            ds.save(f"{tmpdirname}/tempsave")
            ds2 = Dataset.load(f"{tmpdirname}/tempsave")

            self.assertTrue(ds2.unicode.dtype.char, "U")
            self.assertTrue(ds2.bytes.dtype.char, "S")

    def test_save_categorical(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # stringlike
            cstr = Categorical(["a", "b", "c", "d", "e"])

            # numeric
            cnum = Categorical(FastArray([1, 2, 3, 4, 5]), FastArray([10, 20, 30, 40, 50]))

            # enum
            cmap = Categorical([10, 20, 30, 40, 50], {10: "a", 20: "b", 30: "c", 40: "d", 50: "e"})

            # multikey
            cmk = Categorical([FastArray(["a", "b", "c", "d", "e"]), arange(5)])

            ds = Dataset({"cstr": cstr, "cnum": cnum, "cmap": cmap, "cmk": cmk})
            ds.save(f"{tmpdirname}/tempsave")
            ds2 = Dataset.load(f"{tmpdirname}/tempsave")

            self.assertEqual(ds2.cstr.category_mode, CategoryMode.StringArray)
            self.assertEqual(ds2.cnum.category_mode, CategoryMode.NumericArray)
            self.assertEqual(ds2.cmap.category_mode, CategoryMode.Dictionary)
            self.assertEqual(ds2.cmk.category_mode, CategoryMode.MultiKey)

    def test_load_single_item(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            st = Struct(
                {"ds": Dataset({dt.__name__: arange(10, dtype=dt) for dt in arr_types}), "col1": arange(5), "num": 13}
            )

            st.save(f"{tmpdirname}/tempsavestruct")
            ds = Struct.load(f"{tmpdirname}/tempsavestruct", name="ds")

            for k, model_col in st.ds.items():
                self.assertTrue(bool(np.all(model_col == ds[k])))

    def test_name_kwarg(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            arr = rt.FA([1, 2, 3])
            arr.save(tmpdirname, name="tempsave_fa")
            arr2 = rt.load_sds(f"{tmpdirname}/tempsave_fa.sds")
            assert_array_or_cat_equal(arr, arr2)

            ds = rt.Dataset({"a": ["a", "b", "c"]})
            ds.save(tmpdirname, name="tempsave_ds")
            ds2 = rt.load_sds(f"{tmpdirname}/tempsave_ds.sds")
            assert_array_or_cat_equal(ds.a, ds2.a)

            # Test .sds
            arr3 = rt.FA(["a", "b", "c"])
            arr3.save(f"{tmpdirname}/test.sds", name="tempsave_fa2")
            arr4 = rt.load_sds(f"{tmpdirname}/test.sds/tempsave_fa2.sds")
            assert_array_or_cat_equal(arr3, arr4)

    def test_sharedmem_save(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # need to write different routine to remove temporary save from linux shared memory
            if False:
                if sys.platform == "windows":
                    ds = Dataset({"col_" + str(i): np.random.rand(5) for i in range(5)})
                    ds.save(f"{tmpdirname}/temp_save_shared", share="test_save_shared")

                    # make sure the dataset is only shared in shared memory
                    self.assertFalse(os.path.exists(f"{tmpdirname}/temp_save_shared.sds"))

                    # after loading, compare dataset to original
                    ds2 = Dataset.load(f"{tmpdirname}/temp_save_shared", share="test_save_shared")
                    for k, v in ds2.items():
                        self.assertTrue(bool(np.all(v == ds[k])))

                    # test with a garbage filepath
                    ds.save(f"Z:::/riptable/tests/temp_save_shared", share="test_save_shared")
                    ds2 = Dataset.load(r"Y::::::/differentfilepath/samename/temp_save_shared", share="test_save_shared")
                    for k, v in ds2.items():
                        self.assertTrue(bool(np.all(v == ds[k])))

    def test_sharedmem_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"col_" + str(i): np.random.rand(5) for i in range(5)})
            ds.save(f"{tmpdirname}/ds_to_file")

            # NOTE: shared memory on Windows requires admin privs and currently disabled
            if False:
                ds2 = load_sds(f"{tmpdirname}/ds_to_file", share="test_load_shared")
                # remove file on disk

                # load only from share
                ds3 = load_sds(f"{tmpdirname}/ds_to_file", share="test_load_shared")

                for d in [ds, ds3]:
                    for k, v in d.items():
                        self.assertTrue(bool(np.all(v == ds[k])))

    def test_uncompressed_save(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"col_" + str(i): zeros(100_000, dtype=np.int32) for i in range(50)})
            ds.save(f"{tmpdirname}/tempsave_compressed.sds")
            ds.save(f"{tmpdirname}/tempsave_uncompressed.sds", compress=False)

            compsize = os.stat(f"{tmpdirname}/tempsave_compressed.sds").st_size
            uncompsize = os.stat(f"{tmpdirname}/tempsave_uncompressed.sds").st_size

            # make sure the file was smaller (this is full of zeros, so compression should be extreme)
            self.assertTrue(compsize < uncompsize)

            compds = Dataset.load(f"{tmpdirname}/tempsave_compressed.sds")
            uncompds = Dataset.load(f"{tmpdirname}/tempsave_uncompressed.sds")

            for ds2 in [compds, uncompds]:
                for k, v in ds2.items():
                    self.assertTrue(bool(np.all(v == ds[k])))

    def test_meta_tuples(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # test meta tuples for all fastarray / subclasses
            dtn = DateTimeNano(["2018-01-09", "2000-02-29", "2000-03-01", "2019-12-31"], from_tz="NYC")
            span = dtn.hour_span
            c = Categorical(["b", "a", "c", "a"])
            norm = arange(4)

            ds = Dataset(
                {"norm": arange(4), "dtn": dtn, "Cat": Categorical(["b", "a", "c", "a"]), "span": dtn.hour_span}
            )
            ds.save(f"{tmpdirname}/tempsave_ds.sds")
            # only load the meta tuples
            correct_ds = [(b"norm", 3), (b"dtn", 3), (b"Cat", 1), (b"span", 3), (b"Cat!col_0", 2)]
            ds_tups = decompress_dataset_internal(f"{tmpdirname}/tempsave_ds.sds")[0][2]
            for idx, correct in enumerate(correct_ds):
                result_tup = ds_tups[idx]
                self.assertEqual(correct[0], result_tup[0])
                self.assertEqual(correct[1], result_tup[1])

            st = Struct(ds.asdict())
            st.save(f"{tmpdirname}/tempsave_st")
            correct_st = [(b"norm", 1), (b"dtn", 1), (b"Cat", 1), (b"span", 1), (b"Cat!col_0", 0)]
            st_tups = decompress_dataset_internal(f"{tmpdirname}/tempsave_st.sds")[0][2]
            for idx, correct in enumerate(correct_st):
                result_tup = st_tups[idx]
                self.assertEqual(correct[0], result_tup[0])
                self.assertEqual(correct[1], result_tup[1])

    def test_struct_no_arrays(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds1 = Dataset({"col_" + str(i): arange(5) for i in range(3)})
            ds2 = Dataset({"col_" + str(i): arange(5) for i in range(3)})
            ds3 = Dataset({"col_" + str(i): arange(5) for i in range(3)})

            st = Struct({"ds1": ds1, "ds2": ds2, "ds3": ds3})
            st.save(f"{tmpdirname}/tempsave_st")
            st2 = Struct.load(f"{tmpdirname}/tempsave_st")

    def test_corrupt_sds(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            f = open(f"{tmpdirname}/garbage.sds", "w")
            f.close()
            with self.assertRaises(ValueError):
                ds = Dataset.load(f"{tmpdirname}/garbage.sds")

    def test_scalar_overflow(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            large_val = 0xFFFFFFFFFFFFFFFFF
            large_val_arr = np.asarray([large_val])
            # will make an object when put in array
            self.assertEqual(large_val_arr.dtype.char, "O")

            st = Struct({"val": large_val})
            with self.assertWarns(UserWarning):
                st.save(f"{tmpdirname}/tempsave_st")

            st2 = Struct.load(f"{tmpdirname}/tempsave_st")
            val = st2.val
            self.assertTrue(isinstance(val, bytes))

    def test_meta_tuples_new(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            st = Struct(
                {
                    "sc1": 1,
                    "arr1": arange(5),
                    "ds1": Dataset({"col1": arange(5)}),
                    "cat1": Categorical(["a", "b", "c"]),
                    "arr2": arange(5),
                    "sc2": 2,
                    "ds2": Struct({"test": 1}),
                }
            )
            correct_order = ["sc1", "arr1", "ds1", "cat1", "arr2", "sc2", "ds2"]
            correct_tuples = [
                (b"sc1", (SDSFlag.OriginalContainer | SDSFlag.Scalar)),
                (b"arr1", (SDSFlag.OriginalContainer)),
                (b"ds1", (SDSFlag.OriginalContainer | SDSFlag.Nested)),
                (b"cat1", (SDSFlag.OriginalContainer)),
                (b"arr2", (SDSFlag.OriginalContainer)),
                (b"sc2", (SDSFlag.OriginalContainer | SDSFlag.Scalar)),
                (b"ds2", (SDSFlag.OriginalContainer | SDSFlag.Nested)),
                (b"cat1!col_0", 0),
            ]

            st.save(f"{tmpdirname}/tempsave_st")

            _, _, tups, _ = decompress_dataset_internal(f"{tmpdirname}/tempsave_st/_root.sds")[0]
            for t, correct_t in zip(tups, correct_tuples):
                self.assertEqual(t[0], correct_t[0])
                self.assertEqual(t[1], correct_t[1])

    def test_load_extra_files(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"col_" + str(i): arange(5) for i in range(5)})
            st = Struct({"ds1": ds, "ds2": ds, "ds3": ds})

            st.save(f"{tmpdirname}/tempsave_st")
            ds.save(f"{tmpdirname}/tempsave_st/ds4")

            st2 = Struct.load(f"{tmpdirname}/tempsave_st", include_all_sds=True)
            self.assertTrue("ds4" in st2)

            st2 = load_sds(f"{tmpdirname}/tempsave_st", include_all_sds=True)
            self.assertTrue("ds4" in st2)

    def test_missing_item(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"col_" + str(i): arange(5) for i in range(5)})
            st = Struct({"ds1": ds, "ds2": ds, "ds3": ds})

            st.save(f"{tmpdirname}/tempsave_st")
            os.remove(f"{tmpdirname}/tempsave_st/ds2.sds")

            with self.assertWarns(UserWarning):
                st2 = load_sds(f"{tmpdirname}/tempsave_st")
            self.assertTrue("ds2" not in st2)

    def test_single_items(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            arr = arange(5)
            save_sds(pathlib.PurePath(f"{tmpdirname}/temparray"), arr)
            arr2 = load_sds(pathlib.PurePath(f"{tmpdirname}/temparray"))
            self.assertTrue(bool(np.all(arr == arr2)))

            cat = Categorical(["a", "a", "b", "c", "a"])
            save_sds(f"{tmpdirname}/tempcat", cat)
            cat2 = load_sds(f"{tmpdirname}/tempcat")
            self.assertTrue(isinstance(cat2, Categorical))
            self.assertTrue(bool(np.all(cat._fa == cat2._fa)))
            self.assertTrue(bool(np.all(cat.category_array == cat2.category_array)))

            dtn = DateTimeNano(
                ["1992-02-01 12:34", "1995-05-12 12:34", "1956-02-07 12:34", "1959-12-30 12:34"],
                from_tz="NYC",
                to_tz="NYC",
            )
            save_sds(f"{tmpdirname}/tempdtn", dtn)
            dtn2 = load_sds(f"{tmpdirname}/tempdtn")
            self.assertTrue(isinstance(dtn2, DateTimeNano))
            self.assertTrue(bool(np.all(dtn._fa == dtn2._fa)))
            self.assertTrue(dtn._timezone._to_tz == dtn2._timezone._to_tz)

            arr = arange(24)
            arr = arr.reshape((2, 2, 2, 3))
            save_sds(f"{tmpdirname}/temp4dims", arr)
            arr2 = load_sds(f"{tmpdirname}/temp4dims")
            self.assertTrue(arr.dtype == arr2.dtype)
            self.assertTrue(arr.shape == arr2.shape)
            self.assertTrue(arr[1][1][1][1] == arr2[1][1][1][1])

    def test_shared_mem_errors(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(ValueError):
                ds = load_sds(f"{tmpdirname}/invalidfile.sds", share="invalidname")

            with self.assertRaises(ValueError):
                ds = load_sds_mem(f"{tmpdirname}/invalidfile.sds", "invalidname")

    def test_include_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"col_" + str(i): arange(5) for i in range(5)})
            inc = ["col_1", "col_3"]
            ds.save(f"{tmpdirname}/tempds")
            ds2 = load_sds(f"{tmpdirname}/tempds", include=inc)
            self.assertEqual(ds2.shape[1], 2)
            self.assertTrue(list(ds2) == inc)
            self.assertTrue(isinstance(ds2, Dataset))

    def test_multistack_string_width(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds1 = Dataset({"strings": FastArray(["a", "b", "c", "d", "e"])})
            ds2 = Dataset({"strings": FastArray(["aa", "bb", "cc", "dd", "ee"])})
            files = [f"{tmpdirname}/len1.sds", f"{tmpdirname}/len2.sds"]
            ds1.save(files[0])
            ds2.save(files[1])

            arrays, _, _, _, _, _ = ds3 = rc.MultiStackFiles(files)
            stacked = arrays[0]
            self.assertTrue(isinstance(stacked, FastArray))
            self.assertEqual(stacked.itemsize, 2)
            self.assertEqual(stacked.dtype.name, "bytes16")

    def test_stack_missing_string(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds1 = Dataset({"col1": arange(5), "strings": FA(["a", "b", "c", "d", "e"], unicode=True)})
            ds2 = Dataset({"col1": arange(5)})
            files = [f"{tmpdirname}/ds1.sds", f"{tmpdirname}/ds2.sds"]
            ds1.save(files[0])
            ds2.save(files[1])

            ds3 = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds3, Dataset))
            self.assertTrue(ds3._nrows, 10)
            self.assertTrue(ds3._ncols, 2)
            self.assertTrue(ds3.strings.dtype.char == "U")
            self.assertTrue(ds3.strings.itemsize == 4)

    # def test_stack_files(self):
    #    ds1 = Dataset({'col_'+str(i):np.random.rand(5) for i in range(5)})
    #    ds2 = Dataset({'col_'+str(i):np.random.rand(5) for i in range(5)})
    #    files = [ r'riptable/tests/ds1.sds', r'riptable/tests/ds2.sds' ]
    #    ds1.save(files[0])
    #    ds2.save(files[1])

    #    ds3 = load_sds(files, stack=True)
    #    self.assertTrue(isinstance(ds3, Dataset))
    #    self.assertEqual(ds3._nrows, 10)
    #    self.assertEqual(ds3._ncols, 5)

    #    top = ds3[:5,:]
    #    eq = top == ds1
    #    for col in eq.values():
    #        self.assertTrue(bool(np.all(col)))

    #    btm = ds3[5:,:]
    #    eq = btm == ds2
    #    for col in eq.values():
    #        self.assertTrue(bool(np.all(col)))

    def test_stack_files_include(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds1 = Dataset({"col_" + str(i): np.random.rand(5) for i in range(5)})
            ds2 = Dataset({"col_" + str(i): np.random.rand(5) for i in range(5)})
            files = [f"{tmpdirname}/ds1.sds", f"{tmpdirname}/ds2.sds"]
            ds1.save(files[0])
            ds2.save(files[1])

            inc = ["col_3", "col_4"]

            ds3 = load_sds(files, include=inc, stack=True)
            self.assertTrue(isinstance(ds3, Dataset))
            self.assertEqual(ds3._nrows, 10)
            self.assertEqual(ds3._ncols, 2)
            self.assertTrue(bool(np.all(inc == list(ds3))))

    # def test_stack_files_error(self):
    #    ds1 = Dataset({'nums':arange(5)})
    #    ds2 = Dataset({'nums':arange(5,dtype=np.int8)})
    #    files = [ r'riptable/tests/ds1.sds', r'riptable/tests/ds2.sds' ]
    #    ds1.save(files[0])
    #    ds2.save(files[1])

    #    with self.assertRaises(ValueError):
    #        ds3 = load_sds(files, stack=True)

    #    with self.assertRaises(TypeError):
    #        ds3 = load_sds( [ r'riptable/tests/ds1.sds', r'riptable/tests/temp' ], stack=True )

    #    for f in files:
    #        os.remove(f)

    def test_stack_dir(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds1 = Dataset({"col_" + str(i): np.random.rand(5) for i in range(5)})
            ds2 = Dataset({"col_" + str(i): np.random.rand(5) for i in range(5)})
            files1 = [f"{tmpdirname}/dir1/ds1.sds", f"{tmpdirname}/dir2/ds1.sds"]
            files2 = [f"{tmpdirname}/dir1/ds2.sds", f"{tmpdirname}/dir2/ds2.sds"]
            for f in files1:
                ds1.save(f)
            for f in files2:
                ds2.save(f)

            dirs = [f"{tmpdirname}/dir1", f"{tmpdirname}/dir2"]
            inc = ["ds1", "ds2"]

            st = load_sds(dirs, include=inc, stack=True)

            self.assertTrue(isinstance(st, Struct))
            self.assertTrue(bool(np.all(inc == list(st))))
            for v in st.values():
                self.assertTrue(isinstance(v, Dataset))
                self.assertEqual(v._nrows, 10)
                self.assertEqual(v._ncols, 5)

    def test_stack_dir_error(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds1 = Dataset({"nums": arange(5)})
            ds2 = Dataset({"nums": arange(5, dtype=np.int8)})
            files = [f"{tmpdirname}/temp/ds1.sds", f"{tmpdirname}/temp/ds2.sds"]
            ds1.save(files[0])
            ds2.save(files[1])

            with self.assertRaises(ValueError):
                _ = load_sds([f"{tmpdirname}/temp"], stack=True)

    def test_stack_upcast(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            correct = tile(arange(5), 2)

            for dt in arr_types:
                ds = Dataset({"nums": arange(5, dtype=dt)})
                ds.save(f"{tmpdirname}/upcast/" + np.dtype(dt).name)

            int_dt = [np.int8, np.int16, np.int32, np.int64]
            uint_dt = [np.uint8, np.uint16, np.uint32, np.uint64]
            flt_dt = [np.float32, np.float64]

            # float64
            flt64 = f"{tmpdirname}/upcast/float64"
            for dt in int_dt + uint_dt + flt_dt:
                files = [flt64, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: float64 and {np.dtype(dt).name}"
                )
                self.assertTrue(
                    ds.nums.dtype == np.float64, msg=f"Failure in stacking: float64 and {np.dtype(dt).name}"
                )

            # float32
            flt32 = f"{tmpdirname}/upcast/float32"
            for dt in int_dt[:2] + uint_dt[:2]:
                files = [flt32, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: float32 and {np.dtype(dt).name}"
                )
                self.assertTrue(
                    ds.nums.dtype == np.float32, msg=f"Failure in stacking: float32 and {np.dtype(dt).name}"
                )
            for dt in int_dt[2:] + uint_dt[2:]:
                files = [flt32, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: float32 and {np.dtype(dt).name}"
                )
                self.assertTrue(
                    ds.nums.dtype == np.float64, msg=f"Failure in stacking: float32 and {np.dtype(dt).name}"
                )
            for dt in [int_dt[-1], uint_dt[-1], flt_dt[-1]]:
                files = [flt32, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: float32 and {np.dtype(dt).name}"
                )
                self.assertTrue(
                    ds.nums.dtype == np.float64, msg=f"Failure in stacking: float32 and {np.dtype(dt).name}"
                )

            # int64
            i64 = f"{tmpdirname}/upcast/int64"
            for dt in int_dt + uint_dt[:-1]:
                files = [i64, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: int64 and {np.dtype(dt).name}"
                )
                self.assertTrue(ds.nums.dtype == np.int64, msg=f"Failure in stacking: int64 and {np.dtype(dt).name}")
            for dt in flt_dt:
                files = [i64, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: int64 and {np.dtype(dt).name}"
                )
                self.assertTrue(ds.nums.dtype == np.float64, msg=f"Failure in stacking: int64 and {np.dtype(dt).name}")

            # int32
            i32 = f"{tmpdirname}/upcast/int32"
            for dt in int_dt[:-1] + uint_dt[:2]:
                files = [i32, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: int32 and {np.dtype(dt).name}"
                )
                self.assertTrue(ds.nums.dtype == np.int32, msg=f"Failure in stacking: int32 and {np.dtype(dt).name}")
            for dt in flt_dt:
                files = [i32, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: int32 and {np.dtype(dt).name}"
                )
                self.assertTrue(ds.nums.dtype == np.float64, msg=f"Failure in stacking: int32 and {np.dtype(dt).name}")

            # int16
            i16 = f"{tmpdirname}/upcast/int16"
            for dt in int_dt[:2] + uint_dt[:1]:
                files = [i16, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: int16 and {np.dtype(dt).name}"
                )
                self.assertTrue(ds.nums.dtype == np.int16, msg=f"Failure in stacking: int16 and {np.dtype(dt).name}")
            for dt in flt_dt:
                files = [i16, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: int16 and {np.dtype(dt).name}"
                )
                self.assertTrue(ds.nums.dtype == dt, msg=f"Failure in stacking: int16 and {np.dtype(dt).name}")

            # int8
            i8 = f"{tmpdirname}/upcast/int8"
            for dt in int_dt + flt_dt:
                files = [i8, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: int8 and {np.dtype(dt).name}"
                )
                self.assertTrue(ds.nums.dtype == dt, msg=f"Failure in stacking: int8 and {np.dtype(dt).name}")

            # uint64
            ui64 = f"{tmpdirname}/upcast/uint64"
            for dt in uint_dt:
                files = [ui64, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: uint64 and {np.dtype(dt).name}"
                )
                self.assertTrue(ds.nums.dtype == np.uint64, msg=f"Failure in stacking: uint64 and {np.dtype(dt).name}")
            for dt in int_dt[:-1]:
                files = [ui64, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: uint64 and {np.dtype(dt).name}"
                )
                self.assertTrue(ds.nums.dtype == np.int64, msg=f"Failure in stacking: uint64 and {np.dtype(dt).name}")

            # uint32
            ui32 = f"{tmpdirname}/upcast/uint32"
            for dt in int_dt[:-2]:
                files = [ui32, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: uint32 and {np.dtype(dt).name}"
                )
                self.assertTrue(ds.nums.dtype == np.int64, msg=f"Failure in stacking: uint32 and {np.dtype(dt).name}")
            for dt in uint_dt[:-1]:
                files = [ui32, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: uint32 and {np.dtype(dt).name}"
                )
                self.assertTrue(ds.nums.dtype == np.uint32, msg=f"Failure in stacking: uint32 and {np.dtype(dt).name}")

            # uint16
            ui16 = f"{tmpdirname}/upcast/uint16"
            for dt in int_dt[:-3]:
                files = [ui16, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: uint16 and {np.dtype(dt).name}"
                )
                self.assertTrue(ds.nums.dtype == np.int32, msg=f"Failure in stacking: uint16 and {np.dtype(dt).name}")
            for dt in uint_dt[:2]:
                files = [ui16, f"{tmpdirname}/upcast/" + np.dtype(dt).name]
                ds = load_sds(files, stack=True)
                self.assertTrue(isinstance(ds, Dataset))
                self.assertTrue(
                    bool(np.all(ds.nums == correct)), msg=f"Failure in stacking: uint16 and {np.dtype(dt).name}"
                )
                self.assertTrue(ds.nums.dtype == np.uint16, msg=f"Failure in stacking: uint16 and {np.dtype(dt).name}")

            # int + uint to larger itemsize
            # 32 -> 64
            files = [i32, ui32]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f"Failed to upcast int32/uint32 -> int64")
            self.assertTrue(ds.nums.dtype == np.int64, msg="Failed to upcast int32/uint32 -> int64")

            # 16 -> 32
            files = [i16, ui16]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f"Failed to upcast int16/uint16 -> int32")
            self.assertTrue(ds.nums.dtype == np.int32, msg="Failed to upcast int16/uint16 -> int32")

            ui8 = f"{tmpdirname}/upcast/uint8"

            # 8 -> 16
            files = [i8, ui8]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f"Failed to upcast int8/uint8 -> int16")
            self.assertTrue(ds.nums.dtype == np.int16, msg="Failed to upcast int8/uint8 -> int16")

    def test_save_classname(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            files = [
                f"{tmpdirname}/tempsave/cat",
                f"{tmpdirname}/tempsave/dtn",
                f"{tmpdirname}/tempsave/ts",
                f"{tmpdirname}/tempsave/d",
                f"{tmpdirname}/tempsave/dspan",
            ]
            c = Categorical(["a", "a", "b", "c", "a"])
            c.save(files[0])
            dtn = DateTimeNano.random(5)
            dtn.save(files[1])
            ts = TimeSpan(arange(100))
            ts.save(files[2])
            d = Date.range(20190202, 20191231)
            d.save(files[3])
            dspan = DateSpan(arange(100))
            dspan.save(files[4])

            types = [Categorical, DateTimeNano, TimeSpan, Date, DateSpan]
            typenames = [cls.__name__ for cls in types]

            # pull all the single items, compare typenames
            for i, t_name in enumerate(typenames):
                meta = MetaData(decompress_dataset_internal(files[i])[0][0])
                meta = MetaData(meta["item_meta"][0])
                self.assertEqual(meta["classname"], t_name)

            files = [
                f"{tmpdirname}/tempsave/ds",
                f"{tmpdirname}/tempsave/st",
                f"{tmpdirname}/tempsave/pds",
            ]

            # test containers
            ds = Dataset({dt.__name__: arange(10, dtype=dt) for dt in arr_types})
            ds.save(files[0])
            st = Struct({"singlearr": arange(5)})
            st.save(files[1])
            pds = PDataset([ds, ds])
            pds.save(files[2])

            typenames = ["Dataset", "Struct", "PDataset"]
            for i, t_name in enumerate(typenames):
                meta = MetaData(decompress_dataset_internal(files[i])[0][0])
                self.assertEqual(typenames[i], meta["classname"])

    def test_load_list(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            files = [
                pathlib.PurePath(x)
                for x in [
                    f"{tmpdirname}/tempsave/ds",
                    f"{tmpdirname}/tempsave/st",
                    f"{tmpdirname}/tempsave/cat",
                    f"{tmpdirname}/tempsave/fa",
                    f"{tmpdirname}/tempsave/dtn",
                    f"{tmpdirname}/tempsave/ts",
                    f"{tmpdirname}/tempsave/d",
                    f"{tmpdirname}/tempsave/dspan",
                ]
            ]

            ds = Dataset({dt.__name__: arange(10, dtype=dt) for dt in arr_types})
            ds.save(files[0])
            st = Struct({"ds1": ds, "ds2": ds})
            st.save(files[1])
            c = Categorical(["a", "a", "b", "c", "a"])
            c.save(files[2])
            a = arange(1000)
            a.save(files[3])
            dtn = DateTimeNano.random(5)
            dtn.save(files[4])
            ts = TimeSpan(arange(100))
            ts.save(files[5])
            d = Date.range(20190202, 20191231)
            d.save(files[6])
            dspan = DateSpan(arange(100))
            dspan.save(files[7])

            reload = load_sds(files)
            types = [Dataset, Struct, Categorical, FastArray, DateTimeNano, TimeSpan, Date, DateSpan]

            for i, item in enumerate(reload):
                self.assertTrue(isinstance(item, types[i]), msg=f"expected {types[i]} got {type(item)}")

    def test_load_list_include(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"col_" + str(i): arange(5) for i in range(5)})
            ds.save(f"{tmpdirname}/tempsave/ds1")
            ds.save(f"{tmpdirname}/tempsave/ds2")
            files = [pathlib.PurePath(x) for x in [f"{tmpdirname}/tempsave/ds1", f"{tmpdirname}/tempsave/ds2"]]
            inc = ["col_2", "col_3"]
            reload = load_sds(files, include=inc)
            for d in reload:
                self.assertTrue(isinstance(d, Dataset))
                self.assertTrue(bool(np.all(list(d) == inc)))

    def test_fix_strides(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            arr = arange(20, dtype=np.int32)
            arr = arr[::2]
            self.assertEqual(arr.strides[0], 8)
            arr.save(f"{tmpdirname}/arr")

            arr2 = load_sds(f"{tmpdirname}/arr")
            self.assertEqual(arr2.strides[0], 4, msg=f"{arr2}")

            self.assertTrue(bool(np.all(arr == arr2)))

    def test_stack_fa_subclass(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset(
                {
                    "CAT": Categorical(np.random.choice(["a", "b", "c"], 1000)),
                    "DTN": DateTimeNano.random(1000),
                    "DATE": Date(np.random.randint(15000, 20000, 1000)),
                    "TSPAN": TimeSpan(np.random.randint(0, 1_000_000_000 * 60 * 60 * 24, 1000, dtype=np.int64)),
                    "DSPAN": DateSpan(np.random.randint(0, 365, 1000)),
                }
            )
            files = [f"{tmpdirname}/ds" + str(i) + ".sds" for i in range(3)]
            for f in files:
                ds.save(f)

            pds = load_sds(files, stack=True)
            self.assertTrue(isinstance(pds, PDataset))
            for k, v in pds.items():
                self.assertEqual(type(v), type(ds[k]))

    def test_stack_multikey(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            paths = [f"{tmpdirname}/ds1.sds", f"{tmpdirname}/ds2.sds"]

            a_strings = FA(["b", "a", "c", "a"])
            a_nums = FA([1, 1, 3, 1])
            mkcat_a = Cat([a_strings, a_nums])
            single_a = Cat(FA(["b1", "a1", "c3", "a1"]))

            b_strings = FA(["a", "b", "a", "c"])
            b_nums = FA([1, 2, 1, 3])
            single_b = Cat(FA(["a1", "b2", "a1", "c3"]))
            mkcat_b = Cat([b_strings, b_nums])

            ds1 = Dataset({"mkcat": mkcat_a, "singlecat": single_a})
            ds1.save(paths[0])

            ds2 = Dataset({"mkcat": mkcat_b, "singlecat": single_b})
            ds2.save(paths[1])

            ds3 = load_sds(paths, stack=True)

            self.assertTrue(isinstance(ds3, PDataset))

            correct_single = FA(["b1", "a1", "c3", "a1", "a1", "b2", "a1", "c3"])
            self.assertTrue(arr_eq(correct_single, ds3.singlecat.expand_array))

            correct_mk_ikey = FastArray([1, 2, 3, 2, 2, 4, 2, 3])
            self.assertTrue(arr_eq(correct_mk_ikey, ds3.mkcat._fa))

            correct_mk_strings = FA(["b", "a", "c", "b"])
            correct_mk_nums = FA([1, 1, 3, 2])
            mk_cols = [correct_mk_strings, correct_mk_nums]
            result_cols = list(ds3.mkcat.category_dict.values())
            for i, col in enumerate(mk_cols):
                self.assertTrue(arr_eq(result_cols[i], col))

    # def test_include_dataset_shared(self):
    #    ds = Dataset({'col_'+str(i):arange(5) for i in range(5)})
    #    inc = ['col_1', 'col_3']
    #    ds.save(r'riptable/tests/tempds', share='testshare1')
    #    ds2 = load_sds(r'riptable/tests/tempds', share='testshare1', include=inc)
    #    self.assertEqual(len(ds2), 2)
    #    self.assertTrue(list(ds2)==inc)
    #    self.assertTrue(isinstance(ds2, Dataset))
    #    with self.assertRaises(FileNotFoundError):
    #        os.remove(r'riptable/tests/tempds.sds')

    # include keyword works differently with multi-file load
    # def test_include_struct(self):
    #    ds = Dataset({'col_'+str(i):arange(5) for i in range(5)})
    #    st = Struct({'ds1':ds, 'ds2':ds, 'ds3':ds, 'arr':arange(5)})
    #    inc = ['ds1','ds3','col_0','col_1','col_2','col_3','col_4']
    #    st.save(r'riptable/tests/temp_st')
    #    st2 = load_sds(r'riptable/tests/temp_st', include=inc)
    #    self.assertEqual(len(st2), 2)
    #    self.assertTrue(list(st2)==inc)
    #    self.assertTrue(isinstance(st2, Struct))
    #    self.assertFalse(isinstance(st2, Dataset))
    #    self.assertTrue(list(st.ds1) == list(st2.ds1))

    #    inc = ['ds2', 'arr','col_0','col_1','col_2','col_3','col_4']
    #    st2 = load_sds(r'riptable/tests/temp_st', include=inc)
    #    self.assertEqual(len(st2), 2)
    #    self.assertTrue(list(st2)==inc)
    #    self.assertTrue(isinstance(st2, Struct))
    #    self.assertFalse(isinstance(st2, Dataset))
    #    self.assertTrue(list(st.ds2) == list(st2.ds2))

    # def test_include_struct_shared(self):
    #    ds = Dataset({'col_'+str(i):arange(5) for i in range(5)})
    #    st = Struct({'ds1':ds, 'ds2':ds, 'ds3':ds, 'arr':arange(5)})
    #    inc = ['ds1','ds3']
    #    st.save(r'riptable/tests/temp_st', share='testshare2')
    #    st2 = load_sds(r'riptable/tests/temp_st', share='testshare2', include=inc)
    #    self.assertEqual(len(st2), 2)
    #    self.assertTrue(list(st2)==inc)
    #    self.assertTrue(isinstance(st2, Struct))
    #    self.assertFalse(isinstance(st2, Dataset))
    #    self.assertTrue(list(st.ds1) == list(st2.ds1))

    #    inc = ['ds2', 'arr']
    #    st2 = load_sds(r'riptable/tests/temp_st', share='testshare2', include=inc)
    #    self.assertEqual(len(st2), 2)
    #    self.assertTrue(list(st2)==inc)
    #    self.assertTrue(isinstance(st2, Struct))
    #    self.assertFalse(isinstance(st2, Dataset))
    #    self.assertTrue(list(st.ds2) == list(st2.ds2))

    def test_include_categorical(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"arr": arange(5), "cat_col": Categorical(["a", "a", "b", "c", "a"])})
            ds.save(f"{tmpdirname}/temp_cat")

            ds2 = load_sds(f"{tmpdirname}/temp_cat", include=["cat_col"])
            self.assertTrue(list(ds2) == ["cat_col"])
            self.assertTrue(isinstance(ds2.cat_col, Categorical))
            self.assertTrue(bool(np.all(ds.cat_col == ds2.cat_col)))

    def test_include_categorical_multi(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"arr": arange(5), "cat_col": Categorical([np.array(["a", "a", "b", "c", "a"]), arange(5)])})
            ds.save(f"{tmpdirname}/temp_cat")

            ds2 = load_sds(f"{tmpdirname}/temp_cat", include=["cat_col"])
            self.assertTrue(list(ds2) == ["cat_col"])
            self.assertTrue(isinstance(ds2.cat_col, Categorical))

    def test_sections(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            f = f"{tmpdirname}/temp_cat.sds"
            ds = Dataset({"arr": arange(5), "cat_col": Categorical([np.array(["a", "a", "b", "c", "a"]), arange(5)])})
            for i in range(3):
                ds.save(f, append=str(i))
                ds.arr += 5
            ds2 = load_sds(f, stack=True)
            self.assertTrue(np.all(ds2.arr == arange(15)))

            # now add the section later (save first with no section name)
            ds = Dataset({"arr": arange(5), "cat_col": Categorical([np.array(["a", "a", "b", "c", "a"]), arange(5)])})
            ds.save(f)
            # remove catcol, change dataset length
            ds = Dataset({"arr": arange(15) + 5})
            ds.save(f, append="test")
            ds.arr += 15
            ds.save(f, append="test")

            # now read the data vertically using stack=True
            # can it auto gap fill a categorical?
            ds2 = load_sds(f, stack=True)
            self.assertTrue(np.all(ds2.arr == arange(35)))

            ds2 = load_sds(f, include=["cat_col"], stack=True)
            self.assertTrue(list(ds2) == ["cat_col"])
            self.assertTrue(isinstance(ds2.cat_col, Categorical))
            # make sure it dropped arr
            self.assertFalse(list(ds2) == ["arr"])

            # now do multiple files that have append section
            ds2 = load_sds([f, f], stack=True)
            # it should have stacked 6 sections total
            self.assertTrue(np.all(ds2.arr == hstack([np.arange(35), np.arange(35)])))

            # now read the data horizontal by removing stack=True
            ds2 = load_sds(f, include="arr", stack=False)
            # make sure there are 3 datasets
            self.assertTrue(len(ds2) == 3, msg=f"got {ds2}")

            # make sure the first dataset is the first dataset we saved
            self.assertTrue(np.all(ds2[0].arr == arange(5)))

    def test_fix_path(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"col_" + str(i): arange(5) for i in range(5)})
            st = Struct({"ds1": ds, "ds2": ds, "ds3": ds, "arr": arange(5)})
            st.save(f"{tmpdirname}/save_st_dir")
            # auto remove .sds extension
            st2 = Struct.load(f"{tmpdirname}/save_st_dir.sds")
            self.assertTrue(isinstance(st2, Struct))
            st2 = load_sds(f"{tmpdirname}/save_st_dir.sds")
            self.assertTrue(isinstance(st2, Struct))

            # auto add .sds extension
            ds.save(f"{tmpdirname}/temp_save_ds")
            ds2 = Dataset.load(f"{tmpdirname}/temp_save_ds")
            self.assertTrue(isinstance(ds2, Dataset))
            ds2 = load_sds(f"{tmpdirname}/temp_save_ds")
            self.assertTrue(isinstance(ds2, Dataset))
            os.remove(f"{tmpdirname}/temp_save_ds.sds")

            # not file or directory (.sds added)
            with self.assertRaises(ValueError):
                ds = Dataset.load(f"{tmpdirname}/tempgarbagenamewithext.sds")
            with self.assertRaises(ValueError):
                st = Struct.load(f"{tmpdirname}/tempgarbagenamewithext.sds")

            # not file or directory (without .sds)
            with self.assertRaises(ValueError):
                ds = Dataset.load(f"{tmpdirname}/tempgarbagenamewihtoutext")
            with self.assertRaises(ValueError):
                ds = Struct.load(f"{tmpdirname}/tempgarbagenamewihtoutext")

        # st.save(r'riptable/tests/struct_with_ext.sds')
        # self.assertTrue(os.path.isdir(r'riptable/tests/struct_with_ext'))
        #

    def test_double_nested(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            st = Struct({"nested": Struct({"ds": Dataset({"arr": arange(5)})})})
            st.save(f"{tmpdirname}/save_st")
            st2 = load_sds(f"{tmpdirname}/save_st")
            self.assertTrue(isinstance(st2, Struct))
            # struct, not dictionary
            self.assertTrue(isinstance(st2.nested, Struct))
            self.assertTrue(isinstance(st2.nested.ds, Dataset))
            # also make sure no file was created
            self.assertFalse(os.path.exists(f"{tmpdirname}/save_st/nested.sds"))

    def test_sds_flatten(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"col_" + str(i): arange(5) for i in range(5)})
            st = Struct({"ds1": ds, "ds2": ds, "ds3": ds, "arr": arange(5)})
            st.save(f"{tmpdirname}/root_temp")
            st.save(f"{tmpdirname}/root_temp/st1")

            # garbage directory in nest
            st.save(f"{tmpdirname}/root_temp/st2")

            try:
                os.makedirs(f"{tmpdirname}/root_temp/st2/garbagedir")
            except FileExistsError:
                pass

            # garbage file in nest
            st.save(f"{tmpdirname}/root_temp/st3")
            ds.save(f"{tmpdirname}/root_temp/st3/garbagefile.sds")

            try:
                os.rename(f"{tmpdirname}/root_temp/st3/garbagefile.sds", f"{tmpdirname}/root_temp/st3/garbagefile")
            except FileExistsError:
                pass

            sds_flatten(f"{tmpdirname}/root_temp")
            self.assertFalse(os.path.isdir(f"{tmpdirname}/root_temp/st1"))
            self.assertTrue(os.path.isfile(f"{tmpdirname}/root_temp/st1.sds"))
            self.assertTrue(os.path.isfile(f"{tmpdirname}/root_temp/st1!ds1.sds"))
            self.assertTrue(os.path.isfile(f"{tmpdirname}/root_temp/st1!ds2.sds"))
            self.assertTrue(os.path.isfile(f"{tmpdirname}/root_temp/st1!ds3.sds"))
            self.assertTrue(os.path.isdir(f"{tmpdirname}/root_temp/st2"))
            self.assertTrue(os.path.isdir(f"{tmpdirname}/root_temp/st3"))

            new_st = load_sds(f"{tmpdirname}/root_temp", include_all_sds=True)

            s = Struct()
            s["a"] = Dataset({"a": [1, 2, 3]})
            s["b"] = Struct()
            s.save(f"{tmpdirname}/st4.sds", onefile=True)
            s4 = load_sds([f"{tmpdirname}/st4.sds", f"{tmpdirname}/st4.sds"], stack=True)
            self.assertTrue(np.all(s4.a.a == [1, 2, 3, 1, 2, 3]))

        # ADD TESTS FOR:
        # conflict with temp root
        # conflict with existing file

    # def test_load_h5(self):
    #    try:
    #        import hdf5
    #    except:
    #        print('Could not import hdf5 module. Skipping load_h5 test in test_saveload.py')
    #    else:
    #        # load everything
    #        st = load_h5(r'riptable/tests/test_files/h5data.h5', name='/')
    #        self.assertTrue(isinstance(st, Struct))
    #        self.assertEqual(len(st), 1)

    #        # load 1 struct
    #        xtra = load_h5(r'riptable/tests/test_files/h5data.h5', name='xtra')
    #        self.assertTrue(isinstance(xtra, Struct))
    #        self.assertTrue(len(xtra), 4)

    #        # roundtrip h5 -> riptable -> sds -> riptable
    #        xtra.save(r'riptable/tests/testsaveh5')
    #        xtra2 = Struct.load(r'riptable/tests/testsaveh5')
    #        for k,v in xtra.items():
    #            self.assertTrue(bool(np.all(v == xtra2[k])))
    #

    #        # drop short columns
    #        with self.assertWarns(UserWarning):
    #            xtra_ds = load_h5(r'riptable/tests/test_files/h5data.h5', name='xtra', drop_short=True)
    #            self.assertTrue(isinstance(xtra_ds, Dataset))
    #            self.assertTrue(len(xtra_ds), 1)

    def test_all_types(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_len = 20_000

            ds = Dataset(
                {
                    "int8": np.random.randint(low=-128, high=127, size=(test_len), dtype=np.int8),
                    "uint8": np.random.randint(low=0, high=255, size=(test_len), dtype=np.uint8),
                    "int16": np.random.randint(low=-32768, high=32767, size=(test_len), dtype=np.int16),
                    "uint16": np.random.randint(low=0, high=65535, size=(test_len), dtype=np.uint16),
                    "int32": np.random.randint(low=-2147483648, high=2147483647, size=(test_len), dtype=np.int32),
                    "uint32": np.random.randint(low=0, high=4294967295, size=(test_len), dtype=np.uint32),
                    "int64": np.random.randint(
                        low=-9223372036854775808, high=9223372036854775807, size=(test_len), dtype=np.int64
                    ),
                    "uint64": np.random.randint(low=0, high=18446744073709551615, size=(test_len), dtype=np.uint64),
                    "float32": np.random.rand(test_len).astype(np.float32) * 10e9,
                    "float64": np.random.rand(test_len).astype(np.float64) * 10e-9,
                    "unicode_str": FastArray(np.random.choice(["a", "b", "c"], test_len), unicode=True),
                    "bytes_str": np.random.choice([b"a", b"b", b"c"], test_len),
                }
            )

            path = f"{tmpdirname}/alltypes.sds"

            ds.save(path)
            ds2 = load_sds(path)
            self.assertTrue(list(ds) == list(ds2))
            for k, col in ds2.items():
                self.assertTrue(col.dtype == ds[k].dtype)
                self.assertTrue(bool(np.all(col == ds[k])))

            os.remove(path)

            ds.save(path, compress=False)
            ds2 = load_sds(path)
            self.assertTrue(list(ds) == list(ds2))
            for k, col in ds2.items():
                self.assertTrue(col.dtype == ds[k].dtype)
                self.assertTrue(bool(np.all(col == ds[k])))

    def test_type_flip(self):
        # unless they can't be converted to a non-object single-item numpy array, all python
        # scalars will flip to numpy scalars.
        # any objects will be stored as their string representation.

        with tempfile.TemporaryDirectory() as tmpdirname:
            st = Struct(
                {
                    "empty_unicode": "",
                    "empty_bytes": b"",
                    "empty_unicode_arr": np.array([""]),
                    "empty_bytes_arr": np.array([b""]),
                    "nan": np.nan,
                    "empty_numeric_array": arange(0),
                    "npint8": np.int8(0),
                    "npint8": np.uint8(0),
                    "npint16": np.int16(0),
                    "npuint16": np.uint16(0),
                    "npint32": np.int32(0),
                    "npuint32": np.uint32(0),
                    "npint64": np.int64(0),
                    "npuint64": np.uint64(0),
                    "npfloat32": np.float32(0),
                    "npfloat64": np.float64(0),
                    "pyint": 0xFFFFFFFFFFFFFFFFF,  # stored as string, warns
                    "pyflt": 3.14,
                    "bool": True,
                    "npbool": np.bool_(True),
                    "npstr_scalar": np.str_("test"),
                    "npbytes_scalar": np.bytes_("test"),
                    "none": None,
                }
            )
            path = f"{tmpdirname}/alltypes"
            with self.assertWarns(UserWarning):
                save_sds(path, st)
            st2 = load_sds(path)

            s = st2.empty_unicode
            self.assertEqual(s.__class__.__name__, "str_")

            s = st2.empty_bytes
            self.assertEqual(s.__class__.__name__, "bytes_")

            s = st2.pyflt
            self.assertEqual(s.__class__, np.float64)

            s = st2.bool
            self.assertEqual(s.__class__, np.bool_)

            s = st2.pyint
            self.assertEqual(s, b"295147905179352825855")

    # leave this test here until pdataset has its own save / load
    def test_pdataset_save_warning(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"col_" + str(i): np.random.rand(5) for i in range(7)})
            pds = PDataset([ds, ds])
            path = f"{tmpdirname}/pds.sds"
            with self.assertWarns(UserWarning):
                pds.save(path)

            pds2 = load_sds(path)
            self.assertTrue(pds2.__class__ == Dataset)

    def test_stack_cat_invalid_index(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            c = Categorical(["a", "a", "b", "c", "a"])
            c._fa[0] = -128
            ds = Dataset({"catcol": c})
            path = f"{tmpdirname}/ds1.sds"
            ds.save(path)

            ds2 = load_sds([path, path], stack=True)

            self.assertTrue(isinstance(ds2.catcol, Categorical))
            self.assertEqual(ds2.catcol._fa[0], 0)

    def test_load_multikey_base_0(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            arr = np.random.choice(["a", "b", "c"], 10)
            c = Categorical([arr, arr], base_index=0)
            self.assertTrue(isinstance(c, Categorical))
            self.assertTrue(c.ismultikey)
            self.assertEqual(c.base_index, 0)

            path = f"{tmpdirname}/ds1.sds"
            ds1 = Dataset({"catcol": c})
            ds1.save(path)

            ds2 = load_sds(path)
            c2 = ds2.catcol
            self.assertTrue(isinstance(c2, Categorical))
            self.assertTrue(c2.ismultikey)
            self.assertEqual(c2.base_index, 0)

            self.assertTrue(ds1.catcol.count().equals(ds2.catcol.count()))

    def test_onefile(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"arr1": arange(5), "arr2": arange(5)})
            st = Struct({"ds": ds, "arr3": arange(5)})

            folderpath = f"{tmpdirname}/stfolder"
            st.save(folderpath, onefile=False)
            self.assertTrue(os.path.isdir(folderpath))

            singlepath = f"{tmpdirname}/stfile.sds"
            st.save(singlepath, onefile=True)
            self.assertTrue(os.path.isfile(singlepath))

    def test_onefile_folder(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"arr1": arange(5), "arr2": arange(5)})
            st = Struct({"ds1": ds, "arr3": arange(5)})

            singlepath = f"{tmpdirname}/stfile.sds"
            st.save(singlepath, onefile=True)
            ds1 = load_sds(singlepath, folders=["ds1/"])
            self.assertTrue(isinstance(ds1, Dataset))
            self.assertTrue(ds1.equals(ds))

    # these tests are muted until root rebuild is committed
    def _test_rebuild_root(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"arr1": arange(5), "arr2": arange(5)})
            st = Struct({"ds1": ds, "arr3": arange(5)})

            folderpath = f"{tmpdirname}/stfolder"
            st.save(folderpath)
            ds.save(os.path.join(folderpath, "ds2"))

            st2 = load_sds(folderpath)

            self.assertTrue("ds2" in st2)
            self.assertTrue(isinstance(st2.ds2, Dataset))

    def _test_rebuild_root_onefile(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"arr1": arange(5), "arr2": arange(5)})
            st = Struct({"ds1": ds, "arr3": arange(5)})

            folderpath = f"{tmpdirname}/stfolder"
            st.save(folderpath)
            st.save(os.path.join(folderpath, "st2"), onefile=True)

            st2 = load_sds(folderpath)

            self.assertTrue("st2" in st2)
            self.assertTrue(isinstance(st2.st2, Struct))

    def _test_rebuild_root_name(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({"arr1": arange(5), "arr2": arange(5)})
            st = Struct({"ds1": ds, "arr3": arange(5)})

            folderpath = f"{tmpdirname}/stfolder"
            st.save(folderpath)
            st.save(folderpath, name="st2")

            st2 = load_sds(folderpath)

            self.assertTrue("st2" in st2)
            self.assertTrue(isinstance(st2.st2, Struct))

    def test_load_filter(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds = Dataset({_k: list(range(_i * 10, (_i + 1) * 10)) for _i, _k in enumerate(["a", "b", "c", "d", "e"])})
            singlepath = f"{tmpdirname}/stfile.sds"
            ds.save(singlepath)
            ds1 = load_sds(singlepath, filter=[False, False, True, True, True, False, False, False, False, False])
            ds2 = load_sds(singlepath, filter=arange(2, 5))
            self.assertTrue(np.all(ds1.crc.imatrix_make() == ds2.crc.imatrix_make()))

    # def test_onefile_stack(self):
    #    ds = Dataset({'arr1':arange(5), 'arr2':arange(5)})
    #    st = Struct({'ds1':ds, 'arr3':arange(5)})

    #    paths = [ r'riptable/tests/stfile'+str(i)+'.sds' for i in range(3) ]
    #    for p in paths:
    #        st.save(p, onefile=True)

    #    # disabled, names don't match after stack

    #    stacked = load_sds(paths, folders=['ds1/'], stack=True)
    #    # are we popping this item automatically?
    #    stacked = stacked.ds1
    #    self.assertTrue(isinstance(stacked, PDataset))

    #    pds = PDataset([ds for _ in range(len(paths))])
    #    self.assertTrue(pds.equals(stacked))

    #    for p in paths:
    #        os.remove(p)


# TODO fold test_sds_stack_with_categorical into the more general test_sds_stack
# We will still want to test across various container types, but rt_test_data module should be responsible for that detail
@pytest.mark.parametrize("container_type", [Dataset, Struct])
@pytest.mark.parametrize("stack", [True, False])
@pytest.mark.parametrize("stack_count", [1])  # TODO consider stack_count 0 and 2
@pytest.mark.parametrize("data", [get_all_categorical_data()])
def test_sds_stack_with_categorical(container_type, data, stack, stack_count, tmpdir):
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = "test_sds_stack_with_categorical"

        # one SDS file per each container per each data element
        for i, exp in enumerate(data):
            key = f"{container_type.__name__}_{name(exp)}_{i}"
            filename = str(tmpdir.join(key))

            expected = container_type({key: exp})
            save_sds(filename, expected)

            # TODO stack across SDS files that are not identical
            # TODO add case where data is sliced to same length and stored in one container
            actual: Optional[PDataset] = None
            if stack:
                actual = load_sds([filename] * stack_count, stack=stack)
            else:
                actual = load_sds(filename, stack=stack)

            act = actual[key]
            if isinstance(exp, Categorical):
                assert_categorical_equal(act, exp, verbose=True)
            elif isinstance(exp, Struct):
                assert act.equals(exp)
            else:
                pytest.fail(
                    f"{fn}: assertions not implemented for data type {type(data)}\n"
                    + f"expected of type {type(exp)}\n{repr(exp)}"
                    + f"actual of type {type(act)}\n{repr(act)}"
                )


@pytest.mark.parametrize(
    "stack",
    [
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                reason="RIP-483 - Raises 'ValueError: SDS stacking only implemented for Datasets. Must provide folders list if loading from multiple Struct directories.'"
            ),
        ),
        False,
        pytest.param(
            None,
            marks=pytest.mark.xfail(
                reason="RIP-482 - Saving a Struct that contains a Categorical with stacking set to `None` and `stack_count=1` modifies the publically accessible category_dict by modifying the key from `CAT` to `categorical_bytes`. To reproduce, remove this `pytest.mark.xfail`."
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "stack_count",
    [
        1,
        pytest.param(0, marks=pytest.mark.xfail(reason="Same as reason for param `2` below")),
        pytest.param(
            2,
            marks=pytest.mark.xfail(
                reason="RIP-482 - Saving a Struct that contains a Categorical with stacking set to `False` and `stack_count=2` modifies the publically accessible category_dict by modifying the key from `CAT` to `categorical_bytes`. To reproduce, remove this `pytest.mark.xfail`."
            ),
        ),
    ],
)
@pytest.mark.parametrize("data", [load_test_data()])
def test_sds_stack(data, stack, stack_count, tmpdir):
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = "test_sds_stack"

        # one SDS file per each container per each data element
        key = f"{fn}_{id(data)}"
        filename = str(tmpdir.join(key))

        save_sds(filename, data)

        # TODO stack across SDS files that are not identical
        # TODO add case where data is sliced to same length and stored in one container
        actual: Optional[PDataset] = None
        if stack:
            actual = load_sds([filename] * stack_count, stack=stack)
        else:
            actual = load_sds(filename, stack=stack)

        for k, v in data.items():
            exp, act = data[k], actual[k]
            err_msg = (
                f"{fn}: failed equality check for {k}\n"
                + f"expected of type {type(data)}\n{repr(data)}"
                + f"actual of type {type(act)}\n{repr(act)}"
            )
            if isinstance(exp, Categorical):
                assert_categorical_equal(act, exp, verbose=True)
            elif isinstance(exp, FastArray):
                assert_array_equal_(act, exp, err_msg=err_msg)
            elif isinstance(data, Struct):
                assert act.equals(exp), err_msg
            else:
                pytest.fail(f"{fn}: assertions not implemented for data type {type(exp)}\n" + err_msg)


@pytest.mark.parametrize(
    "data",
    [
        Struct(),
        Struct(Struct()),
        Struct(Multiset()),
        Struct({"s": Struct()}),
        Struct({"d": Dataset({})}),
        pytest.param(
            Struct({"m": Multiset()}),
            marks=pytest.mark.xfail(reason="_sds_from_tree: 'NoneType' object cannot be interpreted as an integer"),
        ),
        pytest.param(
            Multiset({"d": Dataset()}),
            marks=pytest.mark.xfail(reason="_sds_from_tree: 'NoneType' object cannot be interpreted as an integer"),
        ),
    ],
)
def test_empty_roundtrip(data, tmpdir):
    with tempfile.TemporaryDirectory() as tmpdirname:
        dir = tmpdir.mkdir("test_empty_roundtrip")
        p = str(dir.join(f"tmp_{id(data)}.sds"))
        save_sds(p, data)
        data2 = load_sds(p)
        assert isinstance(data2, data.__class__)
        assert data.shape == data2.shape, f"saved shape should equal original shape"


@pytest.mark.parametrize("rng_seed", [590518374])
@pytest.mark.parametrize(
    "include_date",
    [
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason="Test fails due to the issue described in https://github.com/rtosholdings/riptable/issues/138"
            ),
        ),
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                reason="Categorical with Date-typed category does not currently preserve the type (Date) of the category data during round-tripping through SDS."
            ),
        ),
    ],
)
@pytest.mark.parametrize("explicit_col_load_list", [False, True])
def test_multi_section_sds_stacked_load_of_single_cat(
    tmp_path, request, rng_seed: int, include_date: bool, explicit_col_load_list: bool
) -> None:
    """
    Test that loading a column(s) from a single SDS file containing multiple 'sections'
    (due to appending or using rt.sds_concat()) works as expected.
    The column(s) should have the correct length and be gap-filled wherever the column(s) do(es)
    not appear in some section(s) of the file.

    This test is a repro/test for https://github.com/rtosholdings/riptable/issues/138
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        rng = default_rng(rng_seed)

        # Randomly generate the rowcounts for the test datasets we'll generate.
        ds_count = 10
        min_rowcount = 5
        max_rowcount = 20
        ds_rowcounts = rng.integers(low=min_rowcount, high=max_rowcount, size=ds_count, endpoint=True)

        # Generate the test datasets.
        test_dss = [
            create_test_dataset(rng=rng, rowcount=int(x), include_dict_cat=True, include_date=include_date)
            for x in ds_rowcounts
        ]
        test_ds_keys = set(test_dss[0].keys())

        # For some of the test datasets, drop one or more columns --
        # this lets us test SDS performs gap-filling as expected when loading.
        drop_cat_col_prob = 0.3
        fa_col_name = "open_price"
        cat_col_name = "deliverable_symbol"
        dropped_col_names = [fa_col_name, cat_col_name]
        for dropped_col_name in dropped_col_names:
            assert (
                dropped_col_name in test_dss[0]
            )  # sanity check to make sure we don't attempt to drop a non-existent column.
        drop_cat_col = rng.choice(
            2, size=len(test_dss), replace=True, p=[drop_cat_col_prob, 1.0 - drop_cat_col_prob]
        ).astype(np.bool_)
        assert 0 < drop_cat_col.sum() < len(test_dss), "Bad random data for drop-column flags."

        for do_drop_cat, test_ds in zip(drop_cat_col, test_dss):
            if do_drop_cat:
                test_ds.col_remove(dropped_col_names)

        # Save the datasets to a single SDS file by creating the file then appending to it.
        data_path = tmp_path / f"{request.node.originalname}-{rng_seed}.sds"
        test_dss[0].save(data_path, overwrite=True)
        for i in range(1, len(test_dss)):
            test_dss[i].save(data_path, append=f"test_section_{i}")

        # Row-concat the in-memory Datasets so we have an expected result
        # to compare the output of rt.load_sds(..., stack=True) against.
        inmem_stacked_ds = rt.Dataset.concat_rows(test_dss)
        assert test_ds_keys == set(inmem_stacked_ds.keys())

        # First, check whether the dropped column(s) is gap-filled as expected when we load the entire Dataset.
        include_keys = inmem_stacked_ds.keys() if explicit_col_load_list else None
        sds_stacked_allcols_ds = rt.load_sds(data_path, include=include_keys, stack=True)
        assert test_ds_keys == set(sds_stacked_allcols_ds.keys())
        assert inmem_stacked_ds.get_nrows() == sds_stacked_allcols_ds.get_nrows()
        for col_name in test_ds_keys:
            inmem_col = inmem_stacked_ds[col_name]
            sds_loaded_col = sds_stacked_allcols_ds[col_name]

            # Verify the columns match up.
            assert_array_or_cat_equal(
                inmem_col,
                sds_loaded_col,
                err_msg=f"The array data for column '{col_name}' does not match the expected result.",
                # Categorical equality comparison options.
                relaxed_cat_check=True,
            )

        # For debugging purposes, determine how many rows are in the datasets where the column was dropped.
        dropped_row_count = sum(map(lambda do_drop, ds: ds.get_nrows() if do_drop else 0, drop_cat_col, test_dss))
        print(f"Datasets where the target column was dropped account for {dropped_row_count} rows.")

        # Try to load the column(s) we randomly dropped from some of the Datasets.
        dropped_col_name_lists = []
        # First try loading all of the dropped columns plus one column present in all sections.
        dropped_col_name_lists.append(["close_price", *dropped_col_names])
        dropped_col_name_lists.append(dropped_col_names)
        dropped_col_name_lists.extend([[x] for x in dropped_col_names])
        for dropped_col_name_list in dropped_col_name_lists:
            with rt.load_sds(data_path, include=dropped_col_name_list, stack=True) as sds_stacked_dropcol_ds:
                assert sds_stacked_dropcol_ds.get_ncols() == len(dropped_col_name_list)
                for dropped_col_name in dropped_col_name_list:
                    assert dropped_col_name in sds_stacked_dropcol_ds
                for dropped_col_name in dropped_col_name_list:
                    assert_array_or_cat_equal(
                        inmem_stacked_ds[dropped_col_name],
                        sds_stacked_dropcol_ds[dropped_col_name],
                        err_msg=f"The array data for column '{dropped_col_name}' does not match the expected result.\tDropped columns: {dropped_col_name_list}",
                        # Categorical equality comparison options.
                        relaxed_cat_check=True,
                    )


@pytest.mark.parametrize("rng_seed", [558902840])
@pytest.mark.parametrize(
    "include_date",
    [
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason="Test fails due to the issue described in https://github.com/rtosholdings/riptable/issues/163"
            ),
        )
    ],
)
@pytest.mark.parametrize("explicit_col_load_list", [False, True])
@pytest.mark.parametrize("randomize_start_mask", [False, True])
@pytest.mark.parametrize(
    "mask_dropped_section_prop",
    [
        pytest.param(None, id="No dropped sections completely masked out"),
        pytest.param(0.01, id="One dropped section completely masked out."),
        pytest.param(1.0, id="All dropped sections completely masked out."),
    ],
)
@pytest.mark.parametrize(
    "mask_undropped_section_prop",
    [
        pytest.param(None, id="No undropped sections completely masked out"),
        pytest.param(0.01, id="One undropped section completely masked out."),
        pytest.param(1.0, id="All undropped sections completely masked out."),
    ],
)
def test_multi_section_sds_stacked_load_single_array_filtered(
    tmp_path,
    request,
    rng_seed: int,
    include_date: bool,
    explicit_col_load_list: bool,
    randomize_start_mask: bool,
    mask_dropped_section_prop: Optional[float],
    mask_undropped_section_prop: Optional[float],
) -> None:
    """
    Test that loading a column(s) from a single SDS file containing multiple 'sections'
    (due to appending or using rt.sds_concat()) works as expected when a mask/filter is provided
    to `rt.load_sds()`.
    The column(s) should have the correct length and be gap-filled wherever the column(s) do(es)
    not appear in some section(s) of the file.

    This test is a repro/test for https://github.com/rtosholdings/riptable/issues/163
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        rng = default_rng(rng_seed)

        # Randomly generate the rowcounts for the test datasets we'll generate.
        ds_count = 10
        min_rowcount = 5
        max_rowcount = 20
        ds_rowcounts = rng.integers(low=min_rowcount, high=max_rowcount, size=ds_count, endpoint=True)

        # Generate the test datasets.
        test_dss = [
            create_test_dataset(rng=rng, rowcount=int(x), include_dict_cat=True, include_date=include_date)
            for x in ds_rowcounts
        ]
        test_ds_keys = set(test_dss[0].keys())

        # For some of the test datasets, drop one or more columns --
        # this lets us test SDS performs gap-filling as expected when loading.
        drop_cat_col_prob = 0.3
        fa_col_name = "open_price"
        cat_col_name = "deliverable_symbol"
        dropped_col_names = [fa_col_name, cat_col_name]
        for dropped_col_name in dropped_col_names:
            assert (
                dropped_col_name in test_dss[0]
            )  # sanity check to make sure we don't attempt to drop a non-existent column.
        drop_cat_col = rng.choice(
            2, size=len(test_dss), replace=True, p=[drop_cat_col_prob, 1.0 - drop_cat_col_prob]
        ).astype(np.bool_)
        assert 0 < drop_cat_col.sum() < len(test_dss), "Bad random data for drop-column flags."

        for do_drop_cat, test_ds in zip(drop_cat_col, test_dss):
            if do_drop_cat:
                test_ds.col_remove(dropped_col_names)

        # Save the datasets to a single SDS file by creating the file then appending to it.
        data_path = tmp_path / f"{request.node.originalname}-{rng_seed}.sds"
        test_dss[0].save(data_path, overwrite=True)
        for i in range(1, len(test_dss)):
            test_dss[i].save(data_path, append=f"test_section_{i}")

        # Create the mask we'll use to filter the data -- both in-memory and when we call rt.load_sds().
        fully_masked_section_flags = rt.ones(len(test_dss), dtype=np.bool_)
        if mask_dropped_section_prop is not None:
            dropped_section_idxs = np.where(drop_cat_col)
            maskout_dropped_section_count = int(np.ceil(mask_dropped_section_prop * len(dropped_section_idxs)))
            maskout_dropped_section_idxs = rng.choice(dropped_section_idxs, size=maskout_dropped_section_count)
            fully_masked_section_flags[maskout_dropped_section_idxs] = False

        if mask_undropped_section_prop is not None:
            undropped_section_idxs = np.where(np.logical_not(drop_cat_col))
            maskout_undropped_section_count = int(np.ceil(mask_undropped_section_prop * len(undropped_section_idxs)))
            maskout_undropped_section_idxs = rng.choice(undropped_section_idxs, size=maskout_undropped_section_count)
            fully_masked_section_flags[maskout_undropped_section_idxs] = False

        data_mask = np.repeat(fully_masked_section_flags, ds_rowcounts)
        if randomize_start_mask:
            total_rowcount = sum(ds.get_nrows() for ds in test_dss)
            row_maskout_prob = 0.2  # probability for each row that it's masked OUT
            overlay_mask = rng.choice(
                2, size=total_rowcount, replace=True, p=[row_maskout_prob, 1.0 - row_maskout_prob]
            ).astype(np.bool_)
            rt.mask_andi(data_mask, overlay_mask)

        # Row-concat the in-memory Datasets so we have an expected result
        # to compare the output of rt.load_sds(..., stack=True) against.
        inmem_stacked_ds = rt.Dataset.concat_rows(test_dss)
        inmem_stacked_ds = inmem_stacked_ds[data_mask, :]
        assert test_ds_keys == set(inmem_stacked_ds.keys())

        # First, check whether the dropped column(s) is gap-filled as expected when we load the entire Dataset.
        include_keys = inmem_stacked_ds.keys() if explicit_col_load_list else None
        sds_stacked_allcols_ds = rt.load_sds(data_path, include=include_keys, stack=True, filter=data_mask)
        assert test_ds_keys == set(sds_stacked_allcols_ds.keys())
        assert inmem_stacked_ds.get_nrows() == sds_stacked_allcols_ds.get_nrows()
        for col_name in test_ds_keys:
            inmem_col = inmem_stacked_ds[col_name]
            sds_loaded_col = sds_stacked_allcols_ds[col_name]

            # Verify the columns match up.
            assert_array_or_cat_equal(
                inmem_col,
                sds_loaded_col,
                err_msg=f"The array data for column '{col_name}' does not match the expected result.",
                # Categorical equality comparison options.
                relaxed_cat_check=True,
                # Don't bother checking category names until SDS round-trips them.
                check_cat_names=False,
                # TEMP: assert_array_or_cat_equal() still checks names right now, unless check_cat_types is disabled.
                check_cat_types=False,
            )

        # For debugging purposes, determine how many rows are in the datasets where the column was dropped.
        dropped_row_count = sum(map(lambda do_drop, ds: ds.get_nrows() if do_drop else 0, drop_cat_col, test_dss))
        print(
            f"Datasets where the target column was dropped account for {dropped_row_count}/{ds_rowcounts.sum()} rows. (Mask selected {data_mask.sum()}/{len(data_mask)} rows.)"
        )

        # Try to load the column(s) we randomly dropped from some of the Datasets.
        dropped_col_name_lists = []
        # First try loading all of the dropped columns plus one column present in all sections.
        dropped_col_name_lists.append(["close_price", *dropped_col_names])
        dropped_col_name_lists.append(dropped_col_names)
        dropped_col_name_lists.extend([[x] for x in dropped_col_names])
        for dropped_col_name_list in dropped_col_name_lists:
            with rt.load_sds(
                data_path, include=dropped_col_name_list, stack=True, filter=data_mask
            ) as sds_stacked_dropcol_ds:
                assert sds_stacked_dropcol_ds.get_ncols() == len(dropped_col_name_list)
                for dropped_col_name in dropped_col_name_list:
                    assert dropped_col_name in sds_stacked_dropcol_ds
                for dropped_col_name in dropped_col_name_list:
                    assert_array_or_cat_equal(
                        inmem_stacked_ds[dropped_col_name],
                        sds_stacked_dropcol_ds[dropped_col_name],
                        err_msg=f"The array data for column '{dropped_col_name}' does not match the expected result.\tDropped columns: {dropped_col_name_list}",
                        # Categorical equality comparison options.
                        relaxed_cat_check=True,
                        # Don't bother checking category names until SDS round-trips them.
                        check_cat_names=False,
                        # TEMP: assert_array_or_cat_equal() still checks names right now, unless check_cat_types is disabled.
                        check_cat_types=False,
                    )


@pytest.mark.parametrize("kind", ["old", "new"])
def test_save_load_timezone_names(kind: str):
    old_std_names = [
        # These are old names that are standard backwards-compatible tzdb names
        "GMT",
        "UTC",
    ]

    old_names = [
        # These are all the old ("short") names stored in SDS files prior to v1.12.0
        # when we switched to storing tzdb names.
        "NYC",
        "DUBLIN",
    ] + old_std_names

    testpath = os.path.join(_TESTDIR, f"{kind}_tz_names.sds")

    # Ensure the sds file actually contains the expected _to_tz names
    sdsinfo = rt.sds_info(testpath)

    info = sdsinfo[0]
    meta_str = info[0]
    meta = MetaData(meta_str)
    i_meta_strs = meta["item_meta"]
    assert len(i_meta_strs) == len(old_names)
    for i_meta_str in i_meta_strs:
        i_meta = MetaData(i_meta_str)
        vars = i_meta["instance_vars"]
        totz = vars["_to_tz"]
        if kind == "old":
            assert totz in old_names
        else:
            assert totz in old_std_names or totz not in old_names

    # Load the old tz names, which should be converted into tzdb names
    ds = rt.load_sds(testpath)
    assert len(ds.keys()) == len(old_names)

    for name in old_names:
        val = ds[name]
        tzdb_name = rt.TimeZone.normalize_tz_to_tzdb_name(name)
        actual_tz = val._timezone
        assert actual_tz._from_tz == "GMT"  # converted on input to GMT
        assert actual_tz._to_tz == tzdb_name


if __name__ == "__main__":
    tester = unittest.main()
