import unittest

from riptable import *


class PGroupby_Test(unittest.TestCase):
    def test_groupbyhash_cutoffs(self):
        arr = np.random.randint(0, 5, 15)
        arr2 = np.random.randint(0, 5, 15)
        arr3 = hstack([arr, arr2])

        gb_arr = groupbyhash([arr])
        gb_arr2 = groupbyhash([arr2])
        gb_arr3 = groupbyhash([arr3], cutoffs=FA([15, 30], dtype=np.int64))

        ikey = gb_arr3['iKey']
        ifirst_vals = gb_arr3['iFirstKey'][0]
        ifirst_cutoffs = gb_arr3['iFirstKey'][1]

        self.assertTrue(bool(np.all(gb_arr['iKey'] == ikey[:15])))
        self.assertTrue(
            bool(np.all(gb_arr['iFirstKey'] == ifirst_vals[: ifirst_cutoffs[0]]))
        )

        self.assertTrue(bool(np.all(gb_arr2['iKey'] == ikey[15:])))
        self.assertTrue(
            bool(np.all(gb_arr2['iFirstKey'] == ifirst_vals[ifirst_cutoffs[0] :]))
        )
        self.assertEqual(
            gb_arr3['unique_count'], (gb_arr['unique_count'] + gb_arr2['unique_count'])
        )

    def test_cutoffs_vs_multikey(self):
        arr = np.random.randint(0, 5, 15)
        arr2 = np.random.randint(0, 5, 15)
        arr3 = hstack([arr, arr2])
        mk_col = hstack([zeros(15), ones(15)])

        gb_mk = groupbyhash([mk_col, arr3])
        gb_arr3 = groupbyhash([arr3], cutoffs=FA([15, 30], dtype=np.int64))
        ikey = gb_arr3['iKey']
        ifirst_vals = gb_arr3['iFirstKey'][0]
        ifirst_cutoffs = gb_arr3['iFirstKey'][1]

        self.assertTrue(
            bool(
                np.all(
                    gb_mk['iFirstKey'][: ifirst_cutoffs[0]]
                    == ifirst_vals[: ifirst_cutoffs[0]]
                )
            )
        )
        self.assertTrue(
            bool(
                np.all(
                    gb_mk['iFirstKey'][ifirst_cutoffs[0] :]
                    == ifirst_vals[ifirst_cutoffs[0] :] + 15
                )
            )
        )
        self.assertEqual(gb_mk['unique_count'], gb_arr3['unique_count'])

    def test_single_unique_per_group(self):
        mk_col = hstack([zeros(15), ones(15)])
        pgb = groupbyhash([mk_col], cutoffs=FA([15, 30], dtype=np.int64))
        self.assertTrue(bool(np.all(pgb['iKey'] == 1)))
        self.assertTrue(bool(np.all(pgb['iFirstKey'][0] == 0)))
        self.assertEqual(pgb['unique_count'], 2)

        gb1 = groupbyhash([mk_col[:15]])
        gb2 = groupbyhash([mk_col[15:]])
        self.assertEqual(gb1['unique_count'], 1)
        self.assertEqual(gb2['unique_count'], 1)

    def test_single_cutoff(self):
        arr = np.random.randint(0, 10, 15)
        gb = groupbyhash([arr])
        pgb = groupbyhash([arr], cutoffs=FA([15], dtype=np.int64))

        self.assertTrue(bool(np.all(gb['iKey'] == pgb['iKey'])))
        self.assertTrue(bool(np.all(gb['iFirstKey'] == pgb['iFirstKey'][0])))
        self.assertEqual(len(pgb['iFirstKey'][1]), 1)
        self.assertEqual(pgb['iFirstKey'][1][0], pgb['unique_count'])

    def test_single_item_per_group(self):
        cutoffs = arange(10, dtype=np.int64) + 1
        gb = groupbyhash([arange(10)])
        pgb = groupbyhash([ones(10)], cutoffs=cutoffs)

        self.assertTrue(bool(np.all(gb['iKey'] == pgb['iFirstKey'][1])))
        self.assertEqual(gb['unique_count'], pgb['unique_count'])
        self.assertTrue(bool(np.all(pgb['iKey'] == 1)))
        self.assertTrue(bool(np.all(pgb['iFirstKey'][0] == 0)))

    def test_empty_cutoffs(self):
        a = arange(10)
        cutoffs = a.astype(np.int64) + 1
        f = logical(a % 2)
        no_change = FastArray([0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=np.int64)

        pgb = groupbyhash([ones(10)], filter=f, cutoffs=cutoffs)
        self.assertTrue(bool(np.all(pgb['iKey'] == f)))
        self.assertTrue(bool(np.all(pgb['iFirstKey'][1] == no_change)))


if __name__ == '__main__':
    tester = unittest.main()
