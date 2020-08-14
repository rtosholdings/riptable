import unittest
from riptable import *


def arr_eq(a, b):
    return bool(np.all(a == b))


def arr_all(a):
    return bool(np.all(a))


class Lexsort_Test(unittest.TestCase):
    '''
    TODO: add more tests for different types
    also include string types
    '''

    # TODO pytest parameterize arr_end
    def test_numeric_multikey(self):

        arr_len = 1_000_000
        d = {
            'int8': np.random.randint(0, 120, arr_len, dtype=np.int8),
            'int16': np.random.randint(0, 32000, arr_len, dtype=np.int16),
            'int32': np.random.randint(0, arr_len, arr_len, dtype=np.int32),
            'int64': np.random.randint(0, arr_len, arr_len, dtype=np.int64),
            'float32': np.random.rand(arr_len).astype(np.float64),
            'float64': np.random.rand(arr_len),
            'bytes': np.random.choice(
                ["AAPL\u2080", "AMZN\u2082", "IBM\u2081"], arr_len
            ),
        }

        for arr_end in [100, 1000, 10_000, 100_000, 1_000_000]:

            arr = list(a[:arr_end] for a in d.values())
            np_result = np.lexsort(arr)
            sfw_result = lexsort(arr)
            diff = np_result - sfw_result
            self.assertEqual(
                0,
                sum(diff),
                msg=f"Lexsort results for length {arr_end} did not match numpy. Total off by {sum(diff)}",
            )

    def test_low_unique(self):
        # toggle threading
        for i in range(2):
            if i == 0:
                FA._TOFF()

            arr_len = 300_000
            ds = Dataset(
                {
                    'flts': np.random.choice([1.11, 2.22, 3.33], arr_len),
                    'bytes': full(arr_len, 'a'),
                    'ints': np.random.randint(4, 7, arr_len),
                    'strs': np.random.choice(['700', '800', '900'], arr_len).view(FA),
                }
            )
            # regular
            rt_lex = lexsort(list(ds.values()))
            np_lex = np.lexsort(list(ds.values()))
            self.assertEqual(sum(rt_lex - np_lex), 0)

            # record array
            rec_forward = ds.as_recordarray()
            rt_lex_f = lexsort([rec_forward])
            np_lex_f = lexsort([rec_forward])
            self.assertEqual(sum(rt_lex_f - np_lex_f), 0)

            # record array
            ds2 = ds[list(ds)[::-1]]
            rec_backward = ds2.as_recordarray()
            rt_lex_b = lexsort([rec_backward])
            np_lex_b = lexsort([rec_backward])
            self.assertEqual(sum(rt_lex_b - np_lex_b), 0)

            FA._TON()
            # self.assertEqual( sum(rt_lex - np_lex), 0 )

    def test_record_array(self):
        # toggle threading
        for i in range(2):
            if i == 0:
                FA._TOFF()

            arr_len = 300_000
            ds = Dataset(
                {
                    'uint64': np.random.randint(0, 1000, arr_len, dtype=np.uint64),
                    'uint32': np.random.randint(0, 1000, arr_len, dtype=np.uint32),
                    'uint16': np.random.randint(0, 1000, arr_len, dtype=np.uint16),
                    'uint8': np.random.randint(0, 200, arr_len, dtype=np.uint8),
                }
            )
            # if the arrays are in this order (large itemsize -> small, record array results will compare correctly)
            rec = ds.as_recordarray()
            rt_lex = lexsort([rec])
            np_lex = np.lexsort([rec])
            self.assertEqual(sum(rt_lex - np_lex), 0)

            FA._TON()

    def test_gb_lex(self):
        length = 30
        int_dt = [
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
        ]
        flt_dt = [np.float32, np.float64]
        str_dt = ['S', 'U']

        vals = [1, 2, 3]
        for idt in int_dt:
            arr = np.random.choice(vals, length).astype(idt)
            arr[0] = vals[0]
            arr[1] = vals[1]
            arr[2] = vals[2]
            gbh = groupbyhash(arr)
            gblex = groupbylex(arr)
            self.assertTrue(
                bool(np.all(gbh['iKey'] == gblex['iKey'])), msg=f'failed on {arr.dtype}'
            )

        vals = [1.1, 2.2, 3.3]
        for fdt in flt_dt:
            arr = np.random.choice(vals, length).astype(fdt)
            arr[0] = vals[0]
            arr[1] = vals[1]
            arr[2] = vals[2]
            gbh = groupbyhash(arr)
            gblex = groupbylex(arr)
            self.assertTrue(
                bool(np.all(gbh['iKey'] == gblex['iKey'])), msg=f'failed on {arr.dtype}'
            )

        vals = ['a', 'b', 'c']
        for sdt in str_dt:
            arr = np.random.choice(vals, length).astype(sdt)
            arr[0] = vals[0]
            arr[1] = vals[1]
            arr[2] = vals[2]
            gbh = groupbyhash(arr)
            gblex = groupbylex(arr)
            self.assertTrue(
                bool(np.all(gbh['iKey'] == gblex['iKey'])), msg=f'failed on {arr.dtype}'
            )

    def test_igroup_ifirst_ncount(self):
        vals = [1, 2, 3]
        vals2 = [1.1, 2.2, 3.3]
        vals3 = [b'a', b'b', b'c']
        vals4 = ['a', 'b', 'c']
        for vals in [vals, vals2, vals3, vals4]:
            arr = np.random.choice(vals, 30)
            for i in range(len(vals)):
                arr[i] = vals[i]

            ds = Dataset({'keycol': arr, 'data': np.random.rand(30)})
            gbh = ds.gb('keycol')
            gbh.grouping.pack_by_group()
            gblex = groupbylex(arr)

            self.assertTrue(
                bool(np.all(gblex['iGroup'] == gbh.grouping.iGroup)),
                msg='failed on {arr.dtype}',
            )
            self.assertTrue(
                bool(np.all(gblex['iFirstGroup'] == gbh.grouping.iFirstGroup)),
                msg='failed on {arr.dtype}',
            )
            self.assertTrue(
                bool(np.all(gblex['nCountGroup'] == gbh.grouping.nCountGroup)),
                msg='failed on {arr.dtype}',
            )

    def test_gb_lex_multikey(self):
        vals_numeric = np.random.randint(0, 3, 100_000)
        vals_str = np.random.choice(['a', 'b', 'c'], 100_000)
        vals_numeric[:3] = 0
        vals_numeric[3:6] = 1
        vals_numeric[6:9] = 2
        vals_str[[0, 3, 6]] = 'a'
        vals_str[[1, 4, 7]] = 'b'
        vals_str[[2, 5, 8]] = 'c'

        gbh = groupbyhash([vals_numeric, vals_str], pack=True)
        gblex = groupbylex([vals_numeric, vals_str], rec=False)

        self.assertTrue(
            bool(np.all(gblex['iKey'] == gbh['iKey'])),
            msg=f'failed on int string multikey',
        )
        self.assertTrue(
            bool(np.all(gblex['iFirstKey'] == gbh['iFirstKey'])),
            msg=f'failed on int string multikey',
        )
        self.assertTrue(
            bool(np.all(gblex['unique_count'] == gbh['unique_count'])),
            msg=f'failed on int string multikey',
        )
        self.assertTrue(
            bool(np.all(gblex['iGroup'] == gbh['iGroup'])),
            msg=f'failed on int string multikey',
        )
        self.assertTrue(
            bool(np.all(gblex['iFirstGroup'] == gbh['iFirstGroup'])),
            msg=f'failed on int string multikey',
        )
        self.assertTrue(
            bool(np.all(gblex['nCountGroup'] == gbh['nCountGroup'])),
            msg=f'failed on int string multikey',
        )

    def test_rt_np_igroup(self):
        vals_numeric = np.random.randint(0, 5, 100_000)

        gbh = groupbyhash(vals_numeric)
        gblex = groupbylex(vals_numeric)
        nplex = np.lexsort([vals_numeric])

        self.assertTrue(bool(np.all(gblex['iGroup'] == nplex)))

    def test_lex_nans(self):
        arr = np.random.choice([np.nan, 1.11, 2.22, 3.33], 50)
        arr[0] = 1.11
        arr[1] = 2.22
        arr[2] = 3.33
        arr[3] = np.nan
        gbh = groupbyhash(arr, pack=True)
        gblex = groupbylex(arr)
        self.assertTrue(
            bool(np.all(gblex['iKey'] == gbh['iKey'])),
            msg=f'failed on int single float with nans',
        )
        self.assertTrue(
            bool(np.all(gblex['iFirstKey'] == gbh['iFirstKey'])),
            msg=f'failed on int single float with nans',
        )
        self.assertTrue(
            bool(np.all(gblex['unique_count'] == gbh['unique_count'])),
            msg=f'failed on int single float with nans',
        )
        self.assertTrue(
            bool(np.all(gblex['iGroup'] == gbh['iGroup'])),
            msg=f'failed on int single float with nans',
        )
        self.assertTrue(
            bool(np.all(gblex['iFirstGroup'] == gbh['iFirstGroup'])),
            msg=f'failed on int single float with nans',
        )
        self.assertTrue(
            bool(np.all(gblex['nCountGroup'] == gbh['nCountGroup'])),
            msg=f'failed on int single float with nans',
        )

    def test_all_unique(self):
        arr = np.random.choice(100_000, 50_000, replace=False)

        int_dt = [np.int32, np.uint32, np.int64, np.uint64]
        flt_dt = [np.float32, np.float64]
        str_dt = ['S', 'U']
        for dt in int_dt + flt_dt + str_dt:
            a = arr.astype(dt)
            sortidx = np.lexsort([a])
            gbh = groupbyhash(arr, pack=True)
            gblex = groupbylex(a)
            self.assertTrue(
                bool(np.all(gblex['iFirstKey'] == sortidx)),
                msg=f'failed on int all unique with dtype {a.dtype}',
            )
            self.assertTrue(
                bool(np.all(gblex['unique_count'] == gbh['unique_count'])),
                msg=f'failed on int all unique with dtype {a.dtype}',
            )
            self.assertTrue(
                bool(np.all(gblex['iGroup'] == sortidx)),
                msg=f'failed on int all unique with dtype {a.dtype}',
            )

        arr.astype('S')

    def test_lex_hash_categorical(self):
        arr = np.random.choice(['a', 'b', 'c'], 20)
        c_lex = Categorical(arr, lex=True)
        c_hash = Categorical(arr, lex=False)
        self.assertTrue(arr_eq(c_lex._fa, c_hash._fa))
        self.assertTrue(arr_eq(c_lex.expand_array, c_hash.expand_array))
        self.assertEqual(c_lex.base_index, c_hash.base_index)

        c_lex_zero = Categorical(arr, lex=True, base_index=0)
        c_hash_zero = Categorical(arr, lex=False, base_index=0)
        self.assertTrue(arr_eq(c_lex_zero._fa, c_hash_zero._fa))
        self.assertTrue(arr_eq(c_lex_zero.expand_array, c_hash_zero.expand_array))
        self.assertEqual(c_lex_zero.base_index, c_hash_zero.base_index)

        self.assertTrue(arr_eq(c_lex_zero.expand_array, c_lex.expand_array))

    def test_lex_categorical_error(self):
        with self.assertRaises(TypeError):
            c = Categorical([1, 2, 3], {1: 'a', 2: 'b', 3: 'c'}, lex=True)
        with self.assertRaises(TypeError):
            c = Categorical(['a', 'a', 'b', 'c', 'a'], ['a', 'b', 'c'], lex=True)

    def test_lex_filter(self):
        arr = np.random.choice(['a', 'b', 'c'], 20)
        f = logical(arange(20) % 2)

        c_lex = Categorical(arr, filter=f, lex=True)
        c_hash = Categorical(arr, filter=f, lex=False)
        # ikeys will be different because combine filter uses first occurence numbering
        # self.assertTrue(arr_eq(c_lex._fa,c_hash._fa))
        self.assertTrue(arr_eq(c_lex.expand_array, c_hash.expand_array))

        arr = FA(['a', 'a', 'b', 'c', 'a'])
        f = FA([True, True, False, True, True])

        c_lex = Categorical(arr, filter=f, lex=True)
        c_hash = Categorical(arr, filter=f, lex=False)
        self.assertEqual(c_lex.unique_count, c_hash.unique_count)
        # self.assertTrue(arr_eq(c_lex._fa,c_hash._fa))
        self.assertTrue(arr_eq(c_lex.expand_array, c_hash.expand_array))

    def test_reverse_shuffle(self):
        arr_len = 300_000
        values = FA(np.random.randint(1, 7, arr_len))
        sorted_idx = lexsort(values)
        reverse_sort = rc.ReverseShuffle(sorted_idx)
        sorted_vals = values[sorted_idx]
        unsorted_vals = sorted_vals[reverse_sort]
        self.assertTrue(arr_eq(unsorted_vals, values))


if __name__ == "__main__":
    tester = unittest.main()
