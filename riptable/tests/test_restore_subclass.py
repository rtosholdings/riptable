import unittest

from riptable import *
from riptable.rt_datetime import NANOS_PER_DAY


def arr_eq(a, b):
    return arr_all(a == b)


def arr_all(a):
    return bool(np.all(a))


class RestoreSubclass_Test(unittest.TestCase):
    def test_tile(self):
        dtn = DateTimeNano.random(5)
        dtn2 = tile(dtn, 2)
        nptile = tile(dtn._np, 2)
        self.assertTrue(isinstance(dtn2, DateTimeNano))
        self.assertTrue(bool(np.all(dtn2._fa == nptile)))

        ts = TimeSpan(np.random.randint(0, NANOS_PER_DAY, 5, dtype=np.int64))
        ts2 = tile(ts, 2)
        nptile = tile(ts._np, 2)
        self.assertTrue(isinstance(dtn2, DateTimeNano))
        self.assertTrue(bool(np.all(ts2._fa == nptile)))

        d = Date(np.random.randint(15_000, 20_000, 5))
        d2 = tile(d, 2)
        nptile = tile(d._np, 2)
        self.assertTrue(isinstance(d2, Date))
        self.assertTrue(arr_eq(nptile, d2._fa))

        ds = DateSpan(np.random.randint(0, 365, 5))
        ds2 = tile(ds, 2)
        nptile = tile(ds._np, 2)
        self.assertTrue(isinstance(ds2, DateSpan))
        self.assertTrue(arr_eq(nptile, ds2._fa))

        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        c2 = tile(c, 2)
        nptile = tile(c._np, 2)
        self.assertTrue(isinstance(c2, Categorical))
        self.assertTrue(arr_eq(nptile, c2._fa))

    def test_repeat(self):

        dtn = DateTimeNano.random(5)
        dtn2 = repeat(dtn, 2)
        nprep = repeat(dtn._np, 2)
        self.assertTrue(isinstance(dtn2, DateTimeNano))
        self.assertTrue(bool(np.all(dtn2._fa == nprep)))
        dtn2 = dtn.repeat(2)
        nprep = dtn._np.repeat(2)
        self.assertTrue(isinstance(dtn2, DateTimeNano))
        self.assertTrue(bool(np.all(dtn2._fa == nprep)))

        ts = TimeSpan(np.random.randint(0, NANOS_PER_DAY, 5, dtype=np.int64))
        ts2 = repeat(ts, 2)
        nprep = repeat(ts._np, 2)
        self.assertTrue(isinstance(dtn2, DateTimeNano))
        self.assertTrue(bool(np.all(ts2._fa == nprep)))
        ts2 = ts.repeat(2)
        nprep = ts._np.repeat(2)
        self.assertTrue(isinstance(dtn2, DateTimeNano))
        self.assertTrue(bool(np.all(ts2._fa == nprep)))

        d = Date(np.random.randint(15_000, 20_000, 5))
        d2 = repeat(d, 2)
        nprep = repeat(d._np, 2)
        self.assertTrue(isinstance(d2, Date))
        self.assertTrue(arr_eq(nprep, d2._fa))
        d2 = d.repeat(2)
        nprep = d._np.repeat(2)
        self.assertTrue(isinstance(d2, Date))
        self.assertTrue(arr_eq(nprep, d2._fa))

        ds = DateSpan(np.random.randint(0, 365, 5))
        ds2 = repeat(ds, 2)
        nprep = repeat(ds._np, 2)
        self.assertTrue(isinstance(ds2, DateSpan))
        self.assertTrue(arr_eq(nprep, ds2._fa))
        ds2 = ds.repeat(2)
        nprep = ds._np.repeat(2)
        self.assertTrue(isinstance(ds2, DateSpan))
        self.assertTrue(arr_eq(nprep, ds2._fa))

        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        c2 = repeat(c, 2)
        nprep = repeat(c._np, 2)
        self.assertTrue(isinstance(c2, Categorical))
        self.assertTrue(arr_eq(nprep, c2._fa))
        c2 = c.repeat(2)
        nprep = c._np.repeat(2)
        self.assertTrue(isinstance(c2, Categorical))
        self.assertTrue(arr_eq(nprep, c2._fa))

    def test_gbkeys(self):
        length = 20
        ds = Dataset(
            {
                'CAT': Categorical(np.random.choice(['a', 'b', 'c'], length)),
                'DTN': DateTimeNano.random(length),
                'DATE': Date(np.random.randint(15000, 20000, length)),
                'TSPAN': TimeSpan(
                    np.random.randint(
                        0, 1_000_000_000 * 60 * 60 * 24, length, dtype=np.int64
                    )
                ),
                'DSPAN': DateSpan(np.random.randint(0, 365, length)),
            }
        )

        for k, v in ds.items():
            result = ds.gb(k).count()
            if k != 'CAT':
                self.assertEqual(type(result[k]), type(v))
            self.assertTrue(arr_eq([k], result.label_get_names()))

    def test_mbget(self):
        length = 20
        ds = Dataset(
            {
                'CAT': Categorical(np.random.choice(['a', 'b', 'c'], length)),
                'DTN': DateTimeNano.random(length),
                'DATE': Date(np.random.randint(15000, 20000, length)),
                'TSPAN': TimeSpan(
                    np.random.randint(
                        0, 1_000_000_000 * 60 * 60 * 24, length, dtype=np.int64
                    )
                ),
                'DSPAN': DateSpan(np.random.randint(0, 365, length)),
            }
        )
        for k, v in ds.items():
            result = mbget(v, [1, 2, 3])
            self.assertEqual(type(result), type(v))

    def test_cat_grouping(self):
        c = Categorical(['a', 'a', 'b', 'c', 'a'])
        ds = Dataset({'catcol': c, 'data': FA([1, 1, 2, 3, 1])})
        ds2 = ds.drop_duplicates('data')

        c2 = ds2.catcol
        self.assertTrue(isinstance(c2, Categorical))
        self.assertTrue(arr_eq(['a', 'b', 'c'], c2))
        self.assertTrue(arr_eq([1, 2, 3], c2.grouping.ikey))
        self.assertTrue(c2.grouping.isdirty)
        self.assertTrue(arr_eq(c2[[0]], ['a']))


if __name__ == '__main__':
    tester = unittest.main()
