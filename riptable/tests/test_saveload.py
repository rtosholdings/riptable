import unittest
import shutil
import os
import sys
import pytest
from riptable import *
from typing import Optional
from riptable.rt_enum import CategoryMode, SDSFlag
from riptable.Utils.rt_metadata import MetaData
from riptable.rt_sds import SDSMakeDirsOn
from riptable.tests.test_utils import get_all_categorical_data
from riptable.Utils.rt_testing import assert_array_equal_, assert_categorical_equal, name
from riptable.Utils.rt_testdata import load_test_data


# change to true since we write into /tests directory
SDSMakeDirsOn()

arr_types = [ np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64 ]

def arr_eq(a,b):
    return bool(np.all(a==b))
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
        ds = Dataset({dt.__name__:arange(10, dtype=dt) for dt in arr_types})
        ds.save(r'riptable/tests/temp/tempsave')
        ds2 = Dataset.load(r'riptable/tests/temp/tempsave')
        
        # name, column order matches
        loadkeys = list(ds2.keys())
        for i, k in enumerate(ds.keys()):
            self.assertEqual(loadkeys[i], k)

        # dtype
        for i, c in enumerate(ds2.values()):
            self.assertEqual(arr_types[i], c.dtype)

        os.remove(r'riptable/tests/temp/tempsave.sds')

    def test_save_strings(self):
        uni = FastArray(['a','b','c','d','e'],unicode=True)
        b = FastArray(['a','b','c','d','e'])
        ds = Dataset({'unicode':uni, 'bytes':b},unicode=True)
        ds.save(r'riptable/tests/temp/tempsave')
        ds2 = Dataset.load(r'riptable/tests/temp/tempsave')

        self.assertTrue(ds2.unicode.dtype.char, 'U')
        self.assertTrue(ds2.bytes.dtype.char, 'S')

        os.remove(r'riptable/tests/temp/tempsave.sds')

    def test_save_categorical(self):
        # stringlike
        cstr = Categorical(['a','b','c','d','e'])

        # numeric
        cnum = Categorical(FastArray([1,2,3,4,5]), FastArray([10,20,30,40,50]))

        # enum
        cmap = Categorical([10,20,30,40,50], {10:'a',20:'b',30:'c',40:'d',50:'e'})

        # multikey
        cmk = Categorical([ FastArray(['a','b','c','d','e']), arange(5) ])

        ds = Dataset({'cstr': cstr, 'cnum':cnum, 'cmap':cmap, 'cmk':cmk})
        ds.save(r'riptable/tests/temp/tempsave')
        ds2 = Dataset.load(r'riptable/tests/temp/tempsave')

        self.assertEqual(ds2.cstr.category_mode, CategoryMode.StringArray)
        self.assertEqual(ds2.cnum.category_mode, CategoryMode.NumericArray)
        self.assertEqual(ds2.cmap.category_mode, CategoryMode.Dictionary)
        self.assertEqual(ds2.cmk.category_mode, CategoryMode.MultiKey)

        os.remove(r'riptable/tests/temp/tempsave.sds')

    def test_load_single_item(self):
        st = Struct({
            'ds' : Dataset({dt.__name__:arange(10, dtype=dt) for dt in arr_types}),
            'col1' : arange(5),
            'num' : 13
        })

        st.save(r'riptable/tests/temp/tempsavestruct')
        ds = Struct.load(r'riptable/tests/temp/tempsavestruct', name='ds')
        
        for k, model_col in st.ds.items():
            self.assertTrue(bool(np.all(model_col == ds[k])))

        shutil.rmtree(r'riptable/tests/temp/tempsavestruct')

    def test_sharedmem_save(self):
        # need to write different routine to remove temporary save from linux shared memory
        if False:
            if sys.platform == 'windows':
                ds = Dataset({'col_'+str(i):np.random.rand(5) for i in range(5)})
                ds.save(r'riptable/tests/temp/temp_save_shared', share='test_save_shared')

                # make sure the dataset is only shared in shared memory
                self.assertFalse(os.path.exists(r'riptable/tests/temp/temp_save_shared.sds'))

                # after loading, compare dataset to original
                ds2 = Dataset.load(r'riptable/tests/temp/temp_save_shared', share='test_save_shared')
                for k,v in ds2.items():
                    self.assertTrue(bool(np.all(v == ds[k])))

                # test with a garbage filepath
                ds.save(r'Z:::/riptable/tests/temp/temp_save_shared', share='test_save_shared')
                ds2 = Dataset.load(r'Y::::::/differentfilepath/samename/temp_save_shared', share='test_save_shared')
                for k,v in ds2.items():
                    self.assertTrue(bool(np.all(v == ds[k])))

    def test_sharedmem_load(self):
        ds = Dataset({'col_'+str(i):np.random.rand(5) for i in range(5)})
        ds.save(r'riptable/tests/temp/ds_to_file')

        # NOTE: shared memory on Windows requires admin privs and currently disabled
        if False:
            ds2 = load_sds(r'riptable/tests/temp/ds_to_file', share='test_load_shared')
            # remove file on disk
            os.remove(r'riptable/tests/temp/ds_to_file.sds')
            # load only from share
            ds3 = load_sds(r'riptable/tests/temp/ds_to_file', share='test_load_shared')

            for d in [ds, ds3]:
                for k,v in d.items():
                    self.assertTrue(bool(np.all(v == ds[k])))

    def test_uncompressed_save(self):
        ds = Dataset({'col_'+str(i):zeros(100_000, dtype=np.int32) for i in range(50)})
        ds.save(r'riptable/tests/temp/tempsave_compressed.sds')
        ds.save(r'riptable/tests/temp/tempsave_uncompressed.sds', compress=False)

        compsize = os.stat(r'riptable/tests/temp/tempsave_compressed.sds').st_size
        uncompsize = os.stat(r'riptable/tests/temp/tempsave_uncompressed.sds').st_size

        # make sure the file was smaller (this is full of zeros, so compression should be extreme)
        self.assertTrue(compsize < uncompsize)

        compds = Dataset.load(r'riptable/tests/temp/tempsave_compressed.sds')
        uncompds = Dataset.load(r'riptable/tests/temp/tempsave_uncompressed.sds')

        for ds2 in [compds, uncompds]:
            for k,v in ds2.items():
                self.assertTrue(bool(np.all(v == ds[k])))

        os.remove(r'riptable/tests/temp/tempsave_compressed.sds')
        os.remove(r'riptable/tests/temp/tempsave_uncompressed.sds')

    def test_meta_tuples(self):
        # test meta tuples for all fastarray / subclasses
        dtn = DateTimeNano(['2018-01-09', '2000-02-29', '2000-03-01', '2019-12-31'], from_tz='NYC')
        span = dtn.hour_span
        c = Categorical(['b','a','c','a'])
        norm = arange(4)

        ds = Dataset({'norm':arange(4), 'dtn':dtn, 'Cat':Categorical(['b','a','c','a']), 'span':dtn.hour_span})
        ds.save(r'riptable/tests/temp/tempsave_ds.sds')
        # only load the meta tuples
        correct_ds = [(b'norm', 3), (b'dtn', 3), (b'Cat', 1), (b'span', 3), (b'Cat!col_0', 2)]
        ds_tups = decompress_dataset_internal(r'riptable/tests/temp/tempsave_ds.sds')[0][2]
        for idx, correct in enumerate(correct_ds):
            result_tup = ds_tups[idx]
            self.assertEqual(correct[0], result_tup[0])
            self.assertEqual(correct[1], result_tup[1])

        os.remove(r'riptable/tests/temp/tempsave_ds.sds')

        st = Struct(ds.asdict())
        st.save(r'riptable/tests/temp/tempsave_st')
        correct_st = [(b'norm', 1), (b'dtn', 1), (b'Cat', 1), (b'span', 1), (b'Cat!col_0', 0)]
        st_tups = decompress_dataset_internal(r'riptable/tests/temp/tempsave_st.sds')[0][2]
        for idx, correct in enumerate(correct_st):
            result_tup = st_tups[idx]
            self.assertEqual(correct[0], result_tup[0])
            self.assertEqual(correct[1], result_tup[1])

        os.remove(r'riptable/tests/temp/tempsave_st.sds')

    def test_struct_no_arrays(self):
        ds1 = Dataset({'col_'+str(i):arange(5) for i in range(3)})
        ds2 = Dataset({'col_'+str(i):arange(5) for i in range(3)})
        ds3 = Dataset({'col_'+str(i):arange(5) for i in range(3)})

        st = Struct({'ds1':ds1, 'ds2':ds2, 'ds3':ds3})
        st.save(r'riptable/tests/temp/tempsave_st')
        st2 = Struct.load(r'riptable/tests/temp/tempsave_st')

        shutil.rmtree(r'riptable/tests/temp/tempsave_st')

    def test_corrupt_sds(self):
        f = open(r'riptable/tests/temp/garbage.sds', 'w')
        f.close()
        with self.assertRaises(ValueError):
            ds = Dataset.load(r'riptable/tests/temp/garbage.sds')
        os.remove(r'riptable/tests/temp/garbage.sds')


    def test_scalar_overflow(self):
        large_val = 0xFFFFFFFFFFFFFFFFF
        large_val_arr = np.asarray([large_val])
        # will make an object when put in array
        self.assertEqual(large_val_arr.dtype.char, 'O')

        st = Struct({'val':large_val})
        with self.assertWarns(UserWarning):
            st.save(r'riptable/tests/temp/tempsave_st')

        st2 = Struct.load(r'riptable/tests/temp/tempsave_st')
        val = st2.val
        self.assertTrue(isinstance(val, bytes))

        os.remove(r'riptable/tests/temp/tempsave_st.sds')
        #shutil.rmtree(r'riptable/tests/temp/tempsave_st')


    def test_meta_tuples_new(self):
        st = Struct({'sc1':1, 'arr1':arange(5), 'ds1':Dataset({'col1':arange(5)}), 'cat1': Categorical(['a','b','c']), 'arr2':arange(5), 'sc2':2, 'ds2':Struct({'test':1})})
        correct_order = ['sc1', 'arr1', 'ds1', 'cat1', 'arr2', 'sc2', 'ds2']
        correct_tuples = [
            (b'sc1',  (SDSFlag.OriginalContainer | SDSFlag.Scalar)),
            (b'arr1', (SDSFlag.OriginalContainer)),
            (b'ds1',  (SDSFlag.OriginalContainer | SDSFlag.Nested)),
            (b'cat1', (SDSFlag.OriginalContainer)),
            (b'arr2', (SDSFlag.OriginalContainer)),
            (b'sc2',  (SDSFlag.OriginalContainer | SDSFlag.Scalar)),
            (b'ds2',  (SDSFlag.OriginalContainer | SDSFlag.Nested)),
            (b'cat1!col_0', 0)
        ]

        st.save(r'riptable/tests/temp/tempsave_st')

        _, _, tups, _ = decompress_dataset_internal(r'riptable/tests/temp/tempsave_st/_root.sds')[0]
        for t, correct_t in zip(tups, correct_tuples):
            self.assertEqual(t[0], correct_t[0])
            self.assertEqual(t[1], correct_t[1])

        shutil.rmtree(r'riptable/tests/temp/tempsave_st')

    def test_load_extra_files(self):
        ds = Dataset({'col_'+str(i):arange(5) for i in range(5)})
        st = Struct({'ds1':ds, 'ds2':ds, 'ds3':ds})

        st.save(r'riptable/tests/temp/tempsave_st')
        ds.save(r'riptable/tests/temp/tempsave_st/ds4')

        st2 = Struct.load(r'riptable/tests/temp/tempsave_st', include_all_sds=True)
        self.assertTrue('ds4' in st2)

        st2 = load_sds(r'riptable/tests/temp/tempsave_st', include_all_sds=True)
        self.assertTrue('ds4' in st2)

        shutil.rmtree(r'riptable/tests/temp/tempsave_st')

    def test_missing_item(self):
        ds = Dataset({'col_'+str(i):arange(5) for i in range(5)})
        st = Struct({'ds1':ds, 'ds2':ds, 'ds3':ds})

        st.save(r'riptable/tests/temp/tempsave_st')
        os.remove(r'riptable/tests/temp/tempsave_st/ds2.sds')

        with self.assertWarns(UserWarning):
            st2 = load_sds(r'riptable/tests/temp/tempsave_st')
        self.assertTrue('ds2' not in st2)

        shutil.rmtree(r'riptable/tests/temp/tempsave_st')


    def test_single_items(self):
        arr = arange(5)
        save_sds(r'riptable/tests/temp/temparray', arr)
        arr2 = load_sds(r'riptable/tests/temp/temparray')
        self.assertTrue(bool(np.all(arr == arr2)))
        os.remove(r'riptable/tests/temp/temparray.sds')

        cat = Categorical(['a','a','b','c','a'])
        save_sds(r'riptable/tests/temp/tempcat', cat)
        cat2 = load_sds(r'riptable/tests/temp/tempcat')
        self.assertTrue(isinstance(cat2, Categorical))
        self.assertTrue(bool(np.all(cat._fa == cat2._fa)))
        self.assertTrue(bool(np.all(cat.category_array == cat2.category_array)))
        os.remove(r'riptable/tests/temp/tempcat.sds')

        dtn = DateTimeNano(['1992-02-01 12:34', '1995-05-12 12:34', '1956-02-07 12:34', '1959-12-30 12:34'],from_tz='NYC', to_tz='NYC')
        save_sds(r'riptable/tests/temp/tempdtn', dtn)
        dtn2 = load_sds(r'riptable/tests/temp/tempdtn')
        self.assertTrue(isinstance(dtn2, DateTimeNano))
        self.assertTrue(bool(np.all(dtn._fa == dtn2._fa)))
        self.assertTrue(dtn._timezone._to_tz == dtn2._timezone._to_tz)
        os.remove(r'riptable/tests/temp/tempdtn.sds')

        arr = arange(24)
        arr = arr.reshape((2,2,2,3))
        save_sds(r'riptable/tests/temp/temp4dims',arr)
        arr2 = load_sds(r'riptable/tests/temp/temp4dims')
        self.assertTrue(arr.dtype == arr2.dtype)
        self.assertTrue(arr.shape == arr2.shape)
        self.assertTrue(arr[1][1][1][1] == arr2[1][1][1][1])
        os.remove(r'riptable/tests/temp/temp4dims.sds')

    def test_shared_mem_errors(self):
        with self.assertRaises(ValueError):
            ds = load_sds(r'riptable/tests/temp/invalidfile.sds',share='invalidname')

        with self.assertRaises(ValueError):
            ds = load_sds_mem(r'riptable/tests/temp/invalidfile.sds', 'invalidname')

    def test_include_dataset(self):
        ds = Dataset({'col_'+str(i):arange(5) for i in range(5)})
        inc = ['col_1', 'col_3']
        ds.save(r'riptable/tests/temp/tempds')
        ds2 = load_sds(r'riptable/tests/temp/tempds', include=inc)
        self.assertEqual(ds2.shape[1], 2)
        self.assertTrue(list(ds2)==inc)
        self.assertTrue(isinstance(ds2, Dataset))
        os.remove(r'riptable/tests/temp/tempds.sds')

    def test_multistack_string_width(self):
        ds1 = Dataset({'strings':FastArray(['a','b','c','d','e'])})
        ds2 = Dataset({'strings':FastArray(['aa','bb','cc','dd','ee'])})
        files = [ r'riptable/tests/temp/len1.sds', r'riptable/tests/temp/len2.sds' ]
        ds1.save(files[0])
        ds2.save(files[1])

        arrays, _, _, _, _, _ = ds3 = rc.MultiStackFiles(files)
        stacked = arrays[0]
        self.assertTrue(isinstance(stacked, FastArray))
        self.assertEqual(stacked.itemsize, 2)
        self.assertEqual(stacked.dtype.name, 'bytes16')

        for f in files:
            os.remove(f)

    def test_stack_missing_string(self):
        ds1 = Dataset({'col1':arange(5), 'strings':FA(['a','b','c','d','e'], unicode=True)})
        ds2 = Dataset({'col1': arange(5)})
        files = [ r'riptable/tests/temp/ds1.sds', r'riptable/tests/temp/ds2.sds' ]
        ds1.save(files[0])
        ds2.save(files[1])

        ds3 = load_sds(files, stack=True)
        self.assertTrue(isinstance(ds3, Dataset))
        self.assertTrue(ds3._nrows, 10)
        self.assertTrue(ds3._ncols, 2)
        self.assertTrue(ds3.strings.dtype.char == 'U')
        self.assertTrue(ds3.strings.itemsize == 4)

        for f in files:
            os.remove(f)


    #def test_stack_files(self):
    #    ds1 = Dataset({'col_'+str(i):np.random.rand(5) for i in range(5)})
    #    ds2 = Dataset({'col_'+str(i):np.random.rand(5) for i in range(5)})
    #    files = [ r'riptable/tests/temp/ds1.sds', r'riptable/tests/temp/ds2.sds' ]
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

    #    for f in files:
    #        os.remove(f)

    def test_stack_files_include(self):
        ds1 = Dataset({'col_'+str(i):np.random.rand(5) for i in range(5)})
        ds2 = Dataset({'col_'+str(i):np.random.rand(5) for i in range(5)})
        files = [ r'riptable/tests/temp/ds1.sds', r'riptable/tests/temp/ds2.sds' ]
        ds1.save(files[0])
        ds2.save(files[1])

        inc = ['col_3','col_4']

        ds3 = load_sds(files, include=inc, stack=True)
        self.assertTrue(isinstance(ds3, Dataset))
        self.assertEqual(ds3._nrows, 10)
        self.assertEqual(ds3._ncols, 2)
        self.assertTrue(bool(np.all(inc == list(ds3))))
        
        for f in files:
            os.remove(f)

    #def test_stack_files_error(self):
    #    ds1 = Dataset({'nums':arange(5)})
    #    ds2 = Dataset({'nums':arange(5,dtype=np.int8)})
    #    files = [ r'riptable/tests/temp/ds1.sds', r'riptable/tests/temp/ds2.sds' ]
    #    ds1.save(files[0])
    #    ds2.save(files[1])

    #    with self.assertRaises(ValueError):
    #        ds3 = load_sds(files, stack=True)

    #    with self.assertRaises(TypeError):
    #        ds3 = load_sds( [ r'riptable/tests/temp/ds1.sds', r'riptable/tests/temp' ], stack=True )

    #    for f in files:
    #        os.remove(f)

    def test_stack_dir(self):
        ds1 = Dataset({'col_'+str(i):np.random.rand(5) for i in range(5)})
        ds2 = Dataset({'col_'+str(i):np.random.rand(5) for i in range(5)})
        files1 = [ r'riptable/tests/temp/dir1/ds1.sds', r'riptable/tests/temp/dir2/ds1.sds' ]
        files2 = [ r'riptable/tests/temp/dir1/ds2.sds', r'riptable/tests/temp/dir2/ds2.sds' ]
        for f in files1: ds1.save(f)
        for f in files2: ds2.save(f)

        dirs = [ r'riptable/tests/temp/dir1', r'riptable/tests/temp/dir2' ]
        inc = ['ds1', 'ds2']

        st = load_sds(dirs, include=inc, stack=True)

        self.assertTrue(isinstance(st, Struct))
        self.assertTrue(bool(np.all(inc == list(st))))
        for v in st.values():
            self.assertTrue(isinstance(v, Dataset))
            self.assertEqual(v._nrows, 10)
            self.assertEqual(v._ncols, 5)

        for d in dirs:
            shutil.rmtree(d)


    def test_stack_dir_error(self):
        ds1 = Dataset({'nums':arange(5)})
        ds2 = Dataset({'nums':arange(5,dtype=np.int8)})
        files = [ r'riptable/tests/temp/ds1.sds', r'riptable/tests/temp/ds2.sds' ]
        ds1.save(files[0])
        ds2.save(files[1])

        with self.assertRaises(ValueError):
            ds3 = load_sds([r'riptable/tests/temp'], stack=True)

        for f in files:
            os.remove(f)

    def test_stack_upcast(self):

        correct = tile(arange(5),2)

        for dt in arr_types:
            ds = Dataset({'nums':arange(5, dtype=dt)})
            ds.save(r'riptable/tests/temp/upcast/'+np.dtype(dt).name)

        int_dt = [ np.int8, np.int16, np.int32, np.int64 ]
        uint_dt = [ np.uint8, np.uint16, np.uint32, np.uint64 ]
        flt_dt = [ np.float32, np.float64 ]

        # float64
        flt64 = r'riptable/tests/temp/upcast/float64'
        for dt in int_dt+uint_dt+flt_dt:
            files = [ flt64, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: float64 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == np.float64, msg=f'Failure in stacking: float64 and {np.dtype(dt).name}')

        # float32
        flt32 = r'riptable/tests/temp/upcast/float32'
        for dt in int_dt[:2] + uint_dt[:2]:
            files = [ flt32, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: float32 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == np.float32, msg=f'Failure in stacking: float32 and {np.dtype(dt).name}')
        for dt in int_dt[2:] + uint_dt[2:]:
            files = [ flt32, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: float32 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == np.float64, msg=f'Failure in stacking: float32 and {np.dtype(dt).name}')
        for dt in [ int_dt[-1], uint_dt[-1], flt_dt[-1] ]:
            files = [ flt32, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: float32 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == np.float64, msg=f'Failure in stacking: float32 and {np.dtype(dt).name}')

        # int64
        i64 = r'riptable/tests/temp/upcast/int64'
        for dt in int_dt + uint_dt[:-1]:
            files = [ i64, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: int64 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == np.int64, msg=f'Failure in stacking: int64 and {np.dtype(dt).name}')
        for dt in flt_dt:
            files = [ i64, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: int64 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == np.float64, msg=f'Failure in stacking: int64 and {np.dtype(dt).name}')

        # int32
        i32 = r'riptable/tests/temp/upcast/int32'
        for dt in int_dt[:-1] + uint_dt[:2]:
            files = [ i32, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: int32 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == np.int32, msg=f'Failure in stacking: int32 and {np.dtype(dt).name}')
        for dt in flt_dt:
            files = [ i32, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: int32 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == np.float64, msg=f'Failure in stacking: int32 and {np.dtype(dt).name}')

        # int16
        i16 = r'riptable/tests/temp/upcast/int16'
        for dt in int_dt[:2] + uint_dt[:1]:
            files = [ i16, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: int16 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == np.int16, msg=f'Failure in stacking: int16 and {np.dtype(dt).name}')
        for dt in flt_dt:
            files = [ i16, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: int16 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == dt, msg=f'Failure in stacking: int16 and {np.dtype(dt).name}')

        # int8
        i8 = r'riptable/tests/temp/upcast/int8'
        for dt in int_dt + flt_dt:
            files = [ i8, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: int8 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == dt, msg=f'Failure in stacking: int8 and {np.dtype(dt).name}')

        # uint64
        ui64 = r'riptable/tests/temp/upcast/uint64'
        for dt in uint_dt:
            files = [ ui64, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: uint64 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == np.uint64, msg=f'Failure in stacking: uint64 and {np.dtype(dt).name}')
        for dt in int_dt[:-1]:
            files = [ ui64, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: uint64 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == np.int64, msg=f'Failure in stacking: uint64 and {np.dtype(dt).name}')

        # uint32
        ui32 = r'riptable/tests/temp/upcast/uint32'
        for dt in int_dt[:-2]:
            files = [ ui32, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: uint32 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == np.int64, msg=f'Failure in stacking: uint32 and {np.dtype(dt).name}')
        for dt in uint_dt[:-1]:
            files = [ ui32, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: uint32 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == np.uint32, msg=f'Failure in stacking: uint32 and {np.dtype(dt).name}')

        # uint16
        ui16 = r'riptable/tests/temp/upcast/uint16'
        for dt in int_dt[:-3]:
            files = [ ui16, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: uint16 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == np.int32, msg=f'Failure in stacking: uint16 and {np.dtype(dt).name}')
        for dt in uint_dt[:2]:
            files = [ ui16, r'riptable/tests/temp/upcast/'+np.dtype(dt).name ]
            ds = load_sds(files, stack=True)
            self.assertTrue(isinstance(ds, Dataset))
            self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failure in stacking: uint16 and {np.dtype(dt).name}')
            self.assertTrue(ds.nums.dtype == np.uint16, msg=f'Failure in stacking: uint16 and {np.dtype(dt).name}')

        # int + uint to larger itemsize
        # 32 -> 64
        files = [ i32, ui32 ]
        ds = load_sds(files, stack=True)
        self.assertTrue(isinstance(ds, Dataset))
        self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failed to upcast int32/uint32 -> int64')
        self.assertTrue(ds.nums.dtype == np.int64, msg='Failed to upcast int32/uint32 -> int64')

        # 16 -> 32
        files = [ i16, ui16 ]
        ds = load_sds(files, stack=True)
        self.assertTrue(isinstance(ds, Dataset))
        self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failed to upcast int16/uint16 -> int32')
        self.assertTrue(ds.nums.dtype == np.int32, msg='Failed to upcast int16/uint16 -> int32')

        ui8 = r'riptable/tests/temp/upcast/uint8'

        # 8 -> 16
        files = [ i8, ui8 ]
        ds = load_sds(files, stack=True)
        self.assertTrue(isinstance(ds, Dataset))
        self.assertTrue(bool(np.all(ds.nums == correct)), msg=f'Failed to upcast int8/uint8 -> int16')
        self.assertTrue(ds.nums.dtype == np.int16, msg='Failed to upcast int8/uint8 -> int16')

        shutil.rmtree(r'riptable/tests/temp/upcast')

    def test_save_classname(self):
        files = [
            r'riptable/tests/temp/tempsave/cat',
            r'riptable/tests/temp/tempsave/dtn',
            r'riptable/tests/temp/tempsave/ts',
            r'riptable/tests/temp/tempsave/d',
            r'riptable/tests/temp/tempsave/dspan',
        ]
        c = Categorical(['a','a','b','c','a'])
        c.save(files[0])
        dtn = DateTimeNano.random(5)
        dtn.save(files[1])
        ts = TimeSpan(arange(100))
        ts.save(files[2])
        d = Date.range(20190202,20191231)
        d.save(files[3])
        dspan = DateSpan(arange(100))
        dspan.save(files[4])

        types = [Categorical, DateTimeNano, TimeSpan, Date, DateSpan]
        typenames = [ cls.__name__ for cls in types ]

        # pull all the single items, compare typenames
        for i, t_name in enumerate(typenames):
            meta = MetaData( decompress_dataset_internal(files[i])[0][0] )
            meta = MetaData( meta['item_meta'][0] )
            self.assertEqual(meta['classname'], t_name)

        files = [
            r'riptable/tests/temp/tempsave/ds',
            r'riptable/tests/temp/tempsave/st',
            r'riptable/tests/temp/tempsave/pds'
        ]

        # test containers
        ds = Dataset({dt.__name__:arange(10, dtype=dt) for dt in arr_types})
        ds.save(files[0])
        st = Struct({'singlearr':arange(5)})
        st.save(files[1])
        pds = PDataset([ds,ds])
        pds.save(files[2])

        typenames = ['Dataset', 'Struct', 'PDataset']
        for i, t_name in enumerate(typenames):
            meta = MetaData( decompress_dataset_internal(files[i])[0][0] )
            self.assertEqual( typenames[i], meta['classname'] )

        shutil.rmtree(r'riptable/tests/temp/tempsave')

    def test_load_list(self):
        files = [
            r'riptable/tests/temp/tempsave/ds',
            r'riptable/tests/temp/tempsave/st',
            r'riptable/tests/temp/tempsave/cat',
            r'riptable/tests/temp/tempsave/fa',
            r'riptable/tests/temp/tempsave/dtn',
            r'riptable/tests/temp/tempsave/ts',
            r'riptable/tests/temp/tempsave/d',
            r'riptable/tests/temp/tempsave/dspan',
        ]

        ds = Dataset({dt.__name__:arange(10, dtype=dt) for dt in arr_types})
        ds.save(files[0])
        st = Struct({'ds1':ds,'ds2':ds})
        st.save(files[1])
        c = Categorical(['a','a','b','c','a'])
        c.save(files[2])
        a = arange(1000)
        a.save(files[3])
        dtn = DateTimeNano.random(5)
        dtn.save(files[4])
        ts = TimeSpan(arange(100))
        ts.save(files[5])
        d = Date.range(20190202,20191231)
        d.save(files[6])
        dspan = DateSpan(arange(100))
        dspan.save(files[7])

        reload = load_sds(files)
        types = [Dataset, Struct, Categorical, FastArray, DateTimeNano, TimeSpan, Date, DateSpan]

        for i, item in enumerate(reload):
            self.assertTrue(isinstance(item, types[i]), msg=f'expected {types[i]} got {type(item)}')

        shutil.rmtree(r'riptable/tests/temp/tempsave')

    def test_load_list_include(self):
        ds = Dataset({'col_'+str(i):arange(5) for i in range(5)})
        ds.save(r'riptable/tests/temp/tempsave/ds1')
        ds.save(r'riptable/tests/temp/tempsave/ds2')
        files = [r'riptable/tests/temp/tempsave/ds1', r'riptable/tests/temp/tempsave/ds2']
        inc = ['col_2','col_3']
        reload = load_sds(files, include=inc)
        for d in reload:
            self.assertTrue(isinstance(d, Dataset))
            self.assertTrue(bool(np.all(list(d) == inc)))

        shutil.rmtree(r'riptable/tests/temp/tempsave')

    def test_fix_strides(self):
        arr = arange(20, dtype=np.int32)
        arr = arr[::2]
        self.assertEqual(arr.strides[0],8)
        arr.save(r'riptable/tests/temp/arr')

        arr2 = load_sds(r'riptable/tests/temp/arr')
        self.assertEqual(arr2.strides[0],4, msg=f'{arr2}')

        self.assertTrue(bool(np.all(arr == arr2)))

        os.remove(r'riptable/tests/temp/arr.sds')

    def test_stack_fa_subclass(self):
        ds = Dataset({
            'CAT' : Categorical(np.random.choice(['a','b','c'],1000)),
            'DTN' : DateTimeNano.random(1000),
            'DATE' : Date(np.random.randint(15000, 20000, 1000)),
            'TSPAN' : TimeSpan(np.random.randint(0, 1_000_000_000*60*60*24, 1000, dtype=np.int64)),
            'DSPAN' : DateSpan(np.random.randint(0, 365, 1000))
        })
        files = [r'riptable/tests/temp/ds'+str(i)+'.sds' for i in range(3)]
        for f in files:
            ds.save(f)

        pds = load_sds(files, stack=True)
        self.assertTrue(isinstance(pds,PDataset))
        for k,v in pds.items():
            self.assertEqual(type(v), type(ds[k]))

        for f in files:
            os.remove(f)

    def test_stack_multikey(self):
        paths = [ r'riptable/tests/temp/ds1.sds',
                  r'riptable/tests/temp/ds2.sds' ]

        a_strings = FA(['b','a','c','a'])
        a_nums = FA([1,1,3,1])
        mkcat_a = Cat([a_strings, a_nums])
        single_a = Cat(FA(['b1','a1', 'c3', 'a1']))

        b_strings = FA(['a','b','a','c'])
        b_nums = FA([1,2,1,3])
        single_b = Cat(FA(['a1','b2','a1','c3']))
        mkcat_b = Cat([b_strings, b_nums])

        ds1 = Dataset({'mkcat':mkcat_a, 'singlecat':single_a})
        ds1.save(paths[0])

        ds2 = Dataset({'mkcat':mkcat_b, 'singlecat':single_b})
        ds2.save(paths[1])

        ds3 = load_sds(paths, stack=True)

        self.assertTrue(isinstance(ds3, PDataset))

        correct_single = FA(['b1','a1','c3','a1','a1','b2','a1','c3'])
        self.assertTrue(arr_eq(correct_single, ds3.singlecat.expand_array))

        correct_mk_ikey = FastArray([1, 2, 3, 2, 2, 4, 2, 3])
        self.assertTrue(arr_eq(correct_mk_ikey, ds3.mkcat._fa))

        correct_mk_strings = FA(['b','a','c','b'])
        correct_mk_nums = FA([1,1,3,2])
        mk_cols = [correct_mk_strings, correct_mk_nums]
        result_cols = list(ds3.mkcat.category_dict.values())
        for i, col in enumerate(mk_cols):
            self.assertTrue(arr_eq( result_cols[i], col ))

        for p in paths:
            os.remove(p)

    #def test_include_dataset_shared(self):
    #    ds = Dataset({'col_'+str(i):arange(5) for i in range(5)})
    #    inc = ['col_1', 'col_3']
    #    ds.save(r'riptable/tests/temp/tempds', share='testshare1')
    #    ds2 = load_sds(r'riptable/tests/temp/tempds', share='testshare1', include=inc)
    #    self.assertEqual(len(ds2), 2)
    #    self.assertTrue(list(ds2)==inc)
    #    self.assertTrue(isinstance(ds2, Dataset))
    #    with self.assertRaises(FileNotFoundError):
    #        os.remove(r'riptable/tests/temp/tempds.sds')

    # include keyword works differently with multi-file load
    #def test_include_struct(self):
    #    ds = Dataset({'col_'+str(i):arange(5) for i in range(5)})
    #    st = Struct({'ds1':ds, 'ds2':ds, 'ds3':ds, 'arr':arange(5)})
    #    inc = ['ds1','ds3','col_0','col_1','col_2','col_3','col_4']
    #    st.save(r'riptable/tests/temp/temp_st')
    #    st2 = load_sds(r'riptable/tests/temp/temp_st', include=inc)
    #    self.assertEqual(len(st2), 2)
    #    self.assertTrue(list(st2)==inc)
    #    self.assertTrue(isinstance(st2, Struct))
    #    self.assertFalse(isinstance(st2, Dataset))
    #    self.assertTrue(list(st.ds1) == list(st2.ds1))

    #    inc = ['ds2', 'arr','col_0','col_1','col_2','col_3','col_4']
    #    st2 = load_sds(r'riptable/tests/temp/temp_st', include=inc)
    #    self.assertEqual(len(st2), 2)
    #    self.assertTrue(list(st2)==inc)
    #    self.assertTrue(isinstance(st2, Struct))
    #    self.assertFalse(isinstance(st2, Dataset))
    #    self.assertTrue(list(st.ds2) == list(st2.ds2))

    #    shutil.rmtree(r'riptable/tests/temp/temp_st')

    #def test_include_struct_shared(self):
    #    ds = Dataset({'col_'+str(i):arange(5) for i in range(5)})
    #    st = Struct({'ds1':ds, 'ds2':ds, 'ds3':ds, 'arr':arange(5)})
    #    inc = ['ds1','ds3']
    #    st.save(r'riptable/tests/temp/temp_st', share='testshare2')
    #    st2 = load_sds(r'riptable/tests/temp/temp_st', share='testshare2', include=inc)
    #    self.assertEqual(len(st2), 2)
    #    self.assertTrue(list(st2)==inc)
    #    self.assertTrue(isinstance(st2, Struct))
    #    self.assertFalse(isinstance(st2, Dataset))
    #    self.assertTrue(list(st.ds1) == list(st2.ds1))

    #    inc = ['ds2', 'arr']
    #    st2 = load_sds(r'riptable/tests/temp/temp_st', share='testshare2', include=inc)
    #    self.assertEqual(len(st2), 2)
    #    self.assertTrue(list(st2)==inc)
    #    self.assertTrue(isinstance(st2, Struct))
    #    self.assertFalse(isinstance(st2, Dataset))
    #    self.assertTrue(list(st.ds2) == list(st2.ds2))
    #    with self.assertRaises(FileNotFoundError):
    #        shutil.rmtree(r'riptable/tests/temp/temp_st')

    def test_include_categorical(self):
        ds = Dataset({'arr':arange(5), 'cat_col':Categorical(['a','a','b','c','a'])})
        ds.save(r'riptable/tests/temp/temp_cat')
        
        ds2 = load_sds(r'riptable/tests/temp/temp_cat', include=['cat_col'])
        self.assertTrue(list(ds2)==['cat_col'])
        self.assertTrue(isinstance(ds2.cat_col, Categorical))
        self.assertTrue(bool(np.all(ds.cat_col == ds2.cat_col)))

        os.remove(r'riptable/tests/temp/temp_cat.sds')
            
    def test_include_categorical_multi(self):
        ds = Dataset({'arr':arange(5), 'cat_col':Categorical([ np.array(['a','a','b','c','a']), arange(5) ])})
        ds.save(r'riptable/tests/temp/temp_cat')
        
        ds2 = load_sds(r'riptable/tests/temp/temp_cat', include=['cat_col'])
        self.assertTrue(list(ds2)==['cat_col'])
        self.assertTrue(isinstance(ds2.cat_col, Categorical))

        os.remove(r'riptable/tests/temp/temp_cat.sds')

    def test_sections(self):
        f=r'riptable/tests/temp/temp_cat.sds'
        ds = Dataset({'arr':arange(5), 'cat_col':Categorical([ np.array(['a','a','b','c','a']), arange(5) ])})
        for i in range(3):
            ds.save(f, append=str(i))
            ds.arr+=5
        ds2 = load_sds(f, stack=True)
        self.assertTrue(np.all(ds2.arr == arange(15)))

        # now add the section later (save first with no section name)
        ds = Dataset({'arr':arange(5), 'cat_col':Categorical([ np.array(['a','a','b','c','a']), arange(5) ])})
        ds.save(f)
        # remove catcol, change dataset length
        ds = Dataset({'arr':arange(15)+5})
        ds.save(f, append='test')
        ds.arr+=15
        ds.save(f, append='test')

        # now read the data vertically using stack=True
        # can it auto gap fill a categorical?
        ds2 = load_sds(f, stack=True)
        self.assertTrue(np.all(ds2.arr == arange(35)))

        ds2 = load_sds(f, include=['cat_col'], stack=True)
        self.assertTrue(list(ds2)==['cat_col'])
        self.assertTrue(isinstance(ds2.cat_col, Categorical))
        # make sure it dropped arr
        self.assertFalse(list(ds2)==['arr'])

        # now do multiple files that have append section
        ds2 = load_sds([f,f], stack=True)
        # it should have stacked 6 sections total
        self.assertTrue(np.all(ds2.arr == hstack([np.arange(35), np.arange(35)])))

        # now read the data horizontal by removing stack=True
        ds2 = load_sds(f, include='arr', stack=False)
        # make sure there are 3 datasets 
        self.assertTrue(len(ds2) == 3, msg=f'got {ds2}')
        
        # make sure the first dataset is the first dataset we saved
        self.assertTrue(np.all(ds2[0].arr == arange(5)))

        os.remove(f)

    def test_fix_path(self):
        ds = Dataset({'col_'+str(i):arange(5) for i in range(5)})
        st = Struct({'ds1':ds, 'ds2':ds, 'ds3':ds, 'arr':arange(5)})
        st.save(r'riptable/tests/temp/save_st_dir')
        # auto remove .sds extension
        st2 = Struct.load(r'riptable/tests/temp/save_st_dir.sds')
        self.assertTrue(isinstance(st2, Struct))
        st2 = load_sds(r'riptable/tests/temp/save_st_dir.sds')
        self.assertTrue(isinstance(st2, Struct))
        shutil.rmtree(r'riptable/tests/temp/save_st_dir')
        
        # auto add .sds extension
        ds.save(r'riptable/tests/temp/temp_save_ds')
        ds2 = Dataset.load(r'riptable/tests/temp/temp_save_ds')
        self.assertTrue(isinstance(ds2, Dataset))
        ds2 = load_sds(r'riptable/tests/temp/temp_save_ds')
        self.assertTrue(isinstance(ds2, Dataset))
        os.remove(r'riptable/tests/temp/temp_save_ds.sds')

        # not file or directory (.sds added)
        with self.assertRaises(ValueError):
            ds = Dataset.load(r'riptable/tests/temp/tempgarbagenamewithext.sds')
        with self.assertRaises(ValueError):
            st = Struct.load(r'riptable/tests/temp/tempgarbagenamewithext.sds')

        # not file or directory (without .sds)
        with self.assertRaises(ValueError):
            ds = Dataset.load(r'riptable/tests/temp/tempgarbagenamewihtoutext')
        with self.assertRaises(ValueError):
            ds = Struct.load(r'riptable/tests/temp/tempgarbagenamewihtoutext')

        #st.save(r'riptable/tests/temp/struct_with_ext.sds')
        #self.assertTrue(os.path.isdir(r'riptable/tests/temp/struct_with_ext'))
        #shutil.rmtree(r'riptable/tests/temp/struct_with_ext')

    def test_double_nested(self):
        st = Struct({'nested':Struct({'ds':Dataset({'arr':arange(5)})})})
        st.save(r'riptable/tests/temp/save_st')
        st2 = load_sds(r'riptable/tests/temp/save_st')
        self.assertTrue(isinstance(st2, Struct))
        # struct, not dictionary
        self.assertTrue(isinstance(st2.nested, Struct))
        self.assertTrue(isinstance(st2.nested.ds, Dataset))
        # also make sure no file was created
        self.assertFalse(os.path.exists(r'riptable/tests/temp/save_st/nested.sds'))

        shutil.rmtree(r'riptable/tests/temp/save_st')

    def test_sds_flatten(self):
        ds = Dataset({'col_'+str(i):arange(5) for i in range(5)})
        st = Struct({'ds1':ds, 'ds2':ds, 'ds3':ds, 'arr':arange(5)})
        shutil.rmtree(r'riptable/tests/temp')
        os.makedirs(r'riptable/tests/temp')
        st.save(r'riptable/tests/temp/root_temp')
        st.save(r'riptable/tests/temp/root_temp/st1')

        # garbage directory in nest
        st.save(r'riptable/tests/temp/root_temp/st2')
        
        try:
            os.makedirs(r'riptable/tests/temp/root_temp/st2/garbagedir')
        except FileExistsError:
            pass

        # garbage file in nest
        st.save(r'riptable/tests/temp/root_temp/st3')
        ds.save(r'riptable/tests/temp/root_temp/st3/garbagefile.sds')

        try:
            os.rename(r'riptable/tests/temp/root_temp/st3/garbagefile.sds', r'riptable/tests/temp/root_temp/st3/garbagefile')
        except FileExistsError:
            pass

        sds_flatten(r'riptable/tests/temp/root_temp')
        self.assertFalse( os.path.isdir(r'riptable/tests/temp/root_temp/st1') )
        self.assertTrue( os.path.isfile(r'riptable/tests/temp/root_temp/st1.sds') )
        self.assertTrue( os.path.isfile(r'riptable/tests/temp/root_temp/st1!ds1.sds') )
        self.assertTrue( os.path.isfile(r'riptable/tests/temp/root_temp/st1!ds2.sds') )
        self.assertTrue( os.path.isfile(r'riptable/tests/temp/root_temp/st1!ds3.sds') )
        self.assertTrue( os.path.isdir(r'riptable/tests/temp/root_temp/st2') )
        self.assertTrue( os.path.isdir(r'riptable/tests/temp/root_temp/st3') )

        new_st = load_sds(r'riptable/tests/temp/root_temp', include_all_sds=True)

        s = Struct()
        s['a'] = Dataset({'a': [1,2,3]})
        s['b'] = Struct()
        s.save(r'riptable/tests/temp/st4.sds', onefile=True)
        s4 = load_sds([r'riptable/tests/temp/st4.sds',r'riptable/tests/temp/st4.sds'], stack=True)
        self.assertTrue(np.all(s4.a.a == [1,2,3,1,2,3]))

        shutil.rmtree(r'riptable/tests/temp/root_temp')

        # ADD TESTS FOR:
        # conflict with temp root
        # conflict with existing file


    #def test_load_h5(self):
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
    #        xtra.save(r'riptable/tests/temp/testsaveh5')
    #        xtra2 = Struct.load(r'riptable/tests/temp/testsaveh5')
    #        for k,v in xtra.items():
    #            self.assertTrue(bool(np.all(v == xtra2[k])))
    #        shutil.rmtree(r'riptable/tests/temp/testsaveh5')

    #        # drop short columns
    #        with self.assertWarns(UserWarning):
    #            xtra_ds = load_h5(r'riptable/tests/test_files/h5data.h5', name='xtra', drop_short=True)
    #            self.assertTrue(isinstance(xtra_ds, Dataset))
    #            self.assertTrue(len(xtra_ds), 1)

    def test_all_types(self):

        test_len = 20_000

        ds = Dataset({
	        'int8'   : np.random.randint(low=-128,high=127,size=(test_len),dtype=np.int8),
	        'uint8'  : np.random.randint(low=0,high=255,size=(test_len),dtype=np.uint8),
	        'int16'  : np.random.randint(low=-32768,high=32767,size=(test_len),dtype=np.int16),
	        'uint16' : np.random.randint(low=0,high=65535,size=(test_len),dtype=np.uint16),
	        'int32'  : np.random.randint(low=-2147483648,high=2147483647,size=(test_len),dtype=np.int32),
	        'uint32' : np.random.randint(low=0,high=4294967295,size=(test_len),dtype=np.uint32),
	        'int64'  : np.random.randint(low=-9223372036854775808,high=9223372036854775807,size=(test_len),dtype=np.int64),
	        'uint64' : np.random.randint(low=0,high=18446744073709551615,size=(test_len),dtype=np.uint64),
	        'float32': np.random.rand(test_len).astype(np.float32)*10e9,
	        'float64': np.random.rand(test_len).astype(np.float64)*10e-9,
            'unicode_str': FastArray(np.random.choice(['a','b','c'],test_len),unicode=True),
            'bytes_str' : np.random.choice([b'a',b'b',b'c'],test_len)
        })
        
        path = r'riptable/tests/temp/alltypes.sds'

        ds.save(path)
        ds2 = load_sds(path)
        self.assertTrue(list(ds)==list(ds2))
        for k, col in ds2.items():
            self.assertTrue(col.dtype == ds[k].dtype)
            self.assertTrue(bool(np.all(col == ds[k])))

        os.remove(path)

        ds.save(path, compress=False)
        ds2 = load_sds(path)
        self.assertTrue(list(ds)==list(ds2))
        for k, col in ds2.items():
            self.assertTrue(col.dtype == ds[k].dtype)
            self.assertTrue(bool(np.all(col == ds[k])))

        os.remove(path)

    def test_type_flip(self):
        # unless they can't be converted to a non-object single-item numpy array, all python 
        # scalars will flip to numpy scalars.
        # any objects will be stored as their string representation.
        st = Struct({
            'empty_unicode' : "",
            'empty_bytes' : b'',
            'empty_unicode_arr' : np.array([""]),
            'empty_bytes_arr' : np.array([b'']),
            'nan' : np.nan,
            'empty_numeric_array' : arange(0),
            'npint8' : np.int8(0),
            'npint8' : np.uint8(0),
            'npint16' : np.int16(0),
            'npuint16' : np.uint16(0),
            'npint32' : np.int32(0),
            'npuint32' : np.uint32(0),
            'npint64' : np.int64(0),
            'npuint64' : np.uint64(0),
            'npfloat32' : np.float32(0),
            'npfloat64' : np.float64(0),
            'pyint' : 0xFFFFFFFFFFFFFFFFF, # stored as string, warns
            'pyflt' : 3.14,
            'bool' : True,
            'npbool' : np.bool_(True),
            'npstr_scalar' : np.str_('test'),
            'npbytes_scalar' : np.bytes_('test'),
            'none' : None,
        })
        path = r'riptable/tests/temp/alltypes'
        with self.assertWarns(UserWarning):
            save_sds(path, st)
        st2 = load_sds(path)

        s = st2.empty_unicode
        self.assertEqual(s.__class__.__name__, 'str_')

        s = st2.empty_bytes
        self.assertEqual(s.__class__.__name__, 'bytes_')

        s = st2.pyflt
        self.assertEqual(s.__class__, np.float64)

        s = st2.bool
        self.assertEqual(s.__class__, np.bool_)

        s = st2.pyint
        self.assertEqual(s, b'295147905179352825855')
        
        os.remove(r'riptable/tests/temp/alltypes.sds')

    # leave this test here until pdataset has its own save / load
    def test_pdataset_save_warning(self):
        ds = Dataset({'col_'+str(i):np.random.rand(5) for i in range(7)})
        pds = PDataset([ds,ds])
        path = r'riptable/tests/temp/pds.sds'
        with self.assertWarns(UserWarning):
            pds.save(path)

        pds2 = load_sds(path)
        self.assertTrue(pds2.__class__ == Dataset)

        os.remove(path)

    def test_stack_cat_invalid_index(self):
        c = Categorical(['a','a','b','c','a'])
        c._fa[0] = -128
        ds = Dataset({'catcol':c})
        path = r'riptable/tests/temp/ds1.sds'
        ds.save(path)

        ds2 = load_sds([path,path],stack=True)

        self.assertTrue(isinstance(ds2.catcol, Categorical))
        self.assertEqual(ds2.catcol._fa[0], 0)

        os.remove(path)

    def test_load_multikey_base_0(self):
        arr = np.random.choice(['a','b','c'],10)
        c = Categorical([arr,arr], base_index=0)
        self.assertTrue(isinstance(c, Categorical))
        self.assertTrue(c.ismultikey)
        self.assertEqual(c.base_index, 0)
        
        path = r'riptable/tests/temp/ds1.sds'
        ds1 = Dataset({'catcol':c})
        ds1.save(path)

        ds2 = load_sds(path)
        c2 = ds2.catcol
        self.assertTrue(isinstance(c2,Categorical))
        self.assertTrue(c2.ismultikey)
        self.assertEqual(c2.base_index, 0)

        self.assertTrue(ds1.catcol.count().equals(ds2.catcol.count()))

        os.remove(path)

    def test_onefile(self):
        ds = Dataset({'arr1':arange(5), 'arr2':arange(5)})
        st = Struct({'ds':ds, 'arr3':arange(5)})

        folderpath = r'riptable/tests/temp/stfolder'
        st.save(folderpath, onefile=False)
        self.assertTrue(os.path.isdir(folderpath))
        shutil.rmtree(folderpath)

        singlepath = r'riptable/tests/temp/stfile.sds'
        st.save(singlepath, onefile=True)
        self.assertTrue(os.path.isfile(singlepath))
        os.remove(singlepath)

    def test_onefile_folder(self):
        ds = Dataset({'arr1':arange(5), 'arr2':arange(5)})
        st = Struct({'ds1':ds, 'arr3':arange(5)})

        singlepath = r'riptable/tests/temp/stfile.sds'
        st.save(singlepath, onefile=True)
        ds1 = load_sds(singlepath, folders=['ds1/'])
        self.assertTrue(isinstance(ds1, Dataset))
        self.assertTrue(ds1.equals(ds))

        os.remove(singlepath)

    # these tests are muted until root rebuild is committed
    def _test_rebuild_root(self):
        ds = Dataset({'arr1':arange(5), 'arr2':arange(5)})
        st = Struct({'ds1':ds, 'arr3':arange(5)})

        folderpath = r'riptable/tests/temp/stfolder'
        st.save(folderpath)
        ds.save(os.path.join(folderpath, 'ds2'))

        st2 = load_sds(folderpath)

        self.assertTrue('ds2' in st2)
        self.assertTrue(isinstance(st2.ds2, Dataset))

        shutil.rmtree(folderpath)

    def _test_rebuild_root_onefile(self):
        ds = Dataset({'arr1':arange(5), 'arr2':arange(5)})
        st = Struct({'ds1':ds, 'arr3':arange(5)})

        folderpath = r'riptable/tests/temp/stfolder'
        st.save(folderpath)
        st.save(os.path.join(folderpath, 'st2'), onefile=True)

        st2 = load_sds(folderpath)

        self.assertTrue('st2' in st2)
        self.assertTrue(isinstance(st2.st2, Struct))

        shutil.rmtree(folderpath)

    def _test_rebuild_root_name(self):
        ds = Dataset({'arr1':arange(5), 'arr2':arange(5)})
        st = Struct({'ds1':ds, 'arr3':arange(5)})

        folderpath = r'riptable/tests/temp/stfolder'
        st.save(folderpath)
        st.save(folderpath, name='st2')

        st2 = load_sds(folderpath)

        self.assertTrue('st2' in st2)
        self.assertTrue(isinstance(st2.st2, Struct))

        shutil.rmtree(folderpath)

    def test_load_filter(self):
        ds = Dataset({_k: list(range(_i *10, (_i +1) *10)) for _i, _k in enumerate(['a','b','c','d','e'])})
        singlepath = r'riptable/tests/temp/stfile.sds'
        ds.save(singlepath)
        ds1=load_sds(singlepath, filter=[False, False, True, True, True, False, False, False, False, False])
        ds2=load_sds(singlepath, filter=arange(2,5))
        self.assertTrue(np.all(ds1.crc.imatrix_make() == ds2.crc.imatrix_make()))
        os.remove(singlepath)

    #def test_onefile_stack(self):
    #    ds = Dataset({'arr1':arange(5), 'arr2':arange(5)})
    #    st = Struct({'ds1':ds, 'arr3':arange(5)})

    #    paths = [ r'riptable/tests/temp/stfile'+str(i)+'.sds' for i in range(3) ]
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
@pytest.mark.parametrize("data", [ get_all_categorical_data() ])
def test_sds_stack_with_categorical(container_type, data, stack, stack_count, tmpdir):
    fn = 'test_sds_stack_with_categorical'

    # one SDS file per each container per each data element
    for i, exp in enumerate(data):
        key = f'{container_type.__name__}_{name(exp)}_{i}'
        filename = str(tmpdir.join(key))

        expected = container_type({key: exp})
        save_sds(filename, expected)

        # TODO stack across SDS files that are not identical
        # TODO add case where data is sliced to same length and stored in one container
        actual: Optional[PDataset] = None
        if stack:
            actual = load_sds([filename]*stack_count, stack=stack)
        else:
            actual = load_sds(filename, stack=stack)

        act = actual[key]
        if isinstance(exp, Categorical):
            assert_categorical_equal(act, exp, verbose=True)
        elif isinstance(exp, Struct):
            assert act.equals(exp)
        else:
            pytest.fail(
                f'{fn}: assertions not implemented for data type {type(data)}\n' + \
                f'expected of type {type(exp)}\n{repr(exp)}' + \
                f'actual of type {type(act)}\n{repr(act)}'
            )


@pytest.mark.parametrize(
    "stack",
    [
        pytest.param(True, marks=pytest.mark.xfail(reason="RIP-483 - Raises 'ValueError: SDS stacking only implemented for Datasets. Must provide folders list if loading from multiple Struct directories.'")),
        False,
        pytest.param(None, marks=pytest.mark.xfail(reason="RIP-482 - Saving a Struct that contains a Categorical with stacking set to `None` and `stack_count=1` modifies the publically accessible category_dict by modifying the key from `CAT` to `categorical_bytes`. To reproduce, remove this `pytest.mark.xfail`.")),
    ]
)
@pytest.mark.parametrize(
    "stack_count",
    [
        1,
        pytest.param(0, marks=pytest.mark.xfail(reason="Same as reason for param `2` below")),
        pytest.param(2, marks=pytest.mark.xfail(reason="RIP-482 - Saving a Struct that contains a Categorical with stacking set to `False` and `stack_count=2` modifies the publically accessible category_dict by modifying the key from `CAT` to `categorical_bytes`. To reproduce, remove this `pytest.mark.xfail`."))
    ]
)
@pytest.mark.parametrize("data", [ load_test_data() ])
def test_sds_stack(data, stack, stack_count, tmpdir):
    fn = 'test_sds_stack'

    # one SDS file per each container per each data element
    key = f'{fn}_{id(data)}'
    filename = str(tmpdir.join(key))

    save_sds(filename, data)

    # TODO stack across SDS files that are not identical
    # TODO add case where data is sliced to same length and stored in one container
    actual: Optional[PDataset] = None
    if stack:
        actual = load_sds([filename]*stack_count, stack=stack)
    else:
        actual = load_sds(filename, stack=stack)

    for k, v in data.items():
        exp, act = data[k], actual[k]
        err_msg = f'{fn}: failed equality check for {k}\n' + \
                  f'expected of type {type(data)}\n{repr(data)}' + \
                  f'actual of type {type(act)}\n{repr(act)}'
        if isinstance(exp, Categorical):
            assert_categorical_equal(act, exp, verbose=True)
        elif isinstance(exp, FastArray):
            assert_array_equal_(act, exp, err_msg=err_msg)
        elif isinstance(data, Struct):
            assert act.equals(exp), err_msg
        else:
            pytest.fail(f'{fn}: assertions not implemented for data type {type(exp)}\n' + err_msg)


@pytest.mark.parametrize(
    'data',
    [
        Struct(),
        Struct(Struct()),
        Struct(Multiset()),
        Struct({'s': Struct()}),

        pytest.param(
            Struct({'d': Dataset({})}),
            marks=pytest.mark.xfail(reason="_sds_from_tree: 'NoneType' object cannot be interpreted as an integer"),
        ),
        pytest.param(
            Struct({'m': Multiset()}),
            marks=pytest.mark.xfail(reason="_sds_from_tree: 'NoneType' object cannot be interpreted as an integer"),
        ),
        pytest.param(
            Multiset({'d': Dataset()}),
            marks=pytest.mark.xfail(reason="_sds_from_tree: 'NoneType' object cannot be interpreted as an integer")
        ),
    ]
)
def test_empty_roundtrip(data, tmpdir):
    dir = tmpdir.mkdir("test_empty_roundtrip")
    p = str(dir.join(f'tmp_{id(data)}.sds'))
    save_sds(p, data)
    data2 = load_sds(p)
    assert isinstance(data2, data.__class__)
    assert data.shape == data2.shape, f'saved shape should equal original shape'


if __name__ == "__main__":
    tester = unittest.main()
