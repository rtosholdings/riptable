import unittest

from riptable import *
from riptable.rt_datetime import (
    NANOS_PER_DAY,
    YDAY_SPLITS,
    YDAY_SPLITS_LEAP,
)

numeric_dt = [
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
    np.float32,
    np.float64,
]


class Date_Test(unittest.TestCase):
    def test_constructor_numeric(self):
        # python scalars
        d = Date(1)
        self.assertTrue(isinstance(d, Date))
        self.assertEqual(d._fa[0], 1)
        self.assertEqual(d.day_of_year[0], 2)
        self.assertEqual(d.year[0], 1970)
        self.assertEqual(d.month[0], 1)

        d = Date(1.0)
        self.assertTrue(isinstance(d, Date))
        self.assertEqual(d._fa[0], 1)
        self.assertEqual(d.day_of_year[0], 2)
        self.assertEqual(d.year[0], 1970)
        self.assertEqual(d.month[0], 1)

        # python lists
        d = Date([1])
        self.assertTrue(isinstance(d, Date))
        self.assertEqual(d._fa[0], 1)
        self.assertEqual(d.day_of_year[0], 2)
        self.assertEqual(d.year[0], 1970)
        self.assertEqual(d.month[0], 1)

        d = Date([1.0])
        self.assertTrue(isinstance(d, Date))
        self.assertEqual(d._fa[0], 1)
        self.assertEqual(d.day_of_year[0], 2)
        self.assertEqual(d.year[0], 1970)
        self.assertEqual(d.month[0], 1)

        # numeric numpy arrays
        for dt in numeric_dt:
            d = Date(FA(1, dtype=dt))
            self.assertEqual(d._fa[0], 1)
            self.assertEqual(d.day_of_year[0], 2)
            self.assertEqual(d.year[0], 1970)
            self.assertEqual(d.month[0], 1)

    def test_constructor_string(self):
        yeardays = [46, 365, 158]
        months = [2, 12, 6]
        years = [2016, 2017, 2018]
        d = Date(['2016-02-15', '2017-12-31', '2018-06-07'])
        self.assertTrue(isinstance(d, Date))
        self.assertTrue(bool(np.all(yeardays == d.day_of_year)))
        self.assertTrue(bool(np.all(months == d.month)))
        self.assertTrue(bool(np.all(years == d.year)))

    def test_constructor_format(self):
        yeardays = [46, 365, 158]
        months = [2, 12, 6]
        years = [2016, 2017, 2018]
        d = Date(['02_15_2016', '12_31_2017', '06_07_2018'], format='%m_%d_%Y')
        self.assertTrue(isinstance(d, Date))
        self.assertTrue(bool(np.all(yeardays == d.day_of_year)))
        self.assertTrue(bool(np.all(months == d.month)))
        self.assertTrue(bool(np.all(years == d.year)))

    def test_constructor_categorical(self):
        yeardays = [46, 365, 158]
        months = [2, 12, 6]
        years = [2016, 2017, 2018]
        c = Categorical(['2016-02-15', '2017-12-31', '2018-06-07'])
        d = Date(c)
        self.assertTrue(isinstance(d, Date))
        self.assertTrue(bool(np.all(yeardays == d.day_of_year)))
        self.assertTrue(bool(np.all(months == d.month)))
        self.assertTrue(bool(np.all(years == d.year)))

    def test_constructor_nano(self):
        pass

    def test_constructor_matlab_datenum(self):
        yeardays = arange(1, 6)
        months = 1
        years = 2018
        d = FA([737061.0, 737062.0, 737063.0, 737064.0, 737065.0])
        d = Date(d, from_matlab=True)
        self.assertTrue(isinstance(d, Date))
        self.assertTrue(bool(np.all(yeardays == d.day_of_year)))
        self.assertTrue(bool(np.all(months == d.month)))
        self.assertTrue(bool(np.all(years == d.year)))

    def test_constructor_empty(self):
        d = Date([])
        self.assertTrue(len(d) == 0)
        self.assertTrue(isinstance(d, Date))
        self.assertTrue(d.dtype == np.int32)

    def test_subtract(self):
        # TODO: add more assertions
        # only return type is checked now
        # most combinations of not allowed types will be caught by FastArray

        d = Date(10)
        val = 1
        result = d - val
        self.assertTrue(isinstance(result, Date))

        val = [1]
        result = d - val
        self.assertTrue(isinstance(result, Date))

        val = FastArray(1)
        result = d - val
        self.assertTrue(isinstance(result, Date))

        val = Date(1)
        result = d - val
        self.assertTrue(isinstance(result, DateSpan))

        val = DateSpan(1)
        result = d - val
        self.assertTrue(isinstance(result, Date))

        val = DateTimeNano(['1970-01-10'], from_tz='GMT')
        result = d - val
        self.assertTrue(isinstance(result, TimeSpan))

        val = TimeSpan(1, unit='h')
        result = d - val
        self.assertTrue(isinstance(result, DateTimeNano))

    def test_add(self):
        d = Date(10)
        val = 1
        result = d + val
        self.assertTrue(isinstance(result, Date))

        val = 1.0
        result = d + val
        self.assertTrue(isinstance(result, Date))

        val = [1]
        result = d + val
        self.assertTrue(isinstance(result, Date))

        val = FastArray(1)
        result = d + val
        self.assertTrue(isinstance(result, Date))

        val = DateSpan(1)
        result = d + val
        self.assertTrue(isinstance(result, Date))

        val = TimeSpan(1, unit='h')
        result = d + val
        self.assertTrue(isinstance(result, DateTimeNano))

    def test_add_datespan(self):
        d = DateSpan(10)
        val = 1
        result = d + val
        self.assertTrue(isinstance(result, DateSpan))

        val = [1]
        result = d + val
        self.assertTrue(isinstance(result, DateSpan))

        val = FastArray(1)
        result = d + val
        self.assertTrue(isinstance(result, DateSpan))

        val = DateSpan(1)
        result = d + val
        self.assertTrue(isinstance(result, DateSpan))

        val = DateTimeNano(NANOS_PER_DAY, from_tz='GMT')
        result = d + val
        self.assertTrue(isinstance(result, DateTimeNano))

        val = TimeSpan(1, unit='h')
        result = d + val
        self.assertTrue(isinstance(result, TimeSpan))

    def test_math_errors(self):

        d = Date(1)
        val = Date(1)
        with self.assertRaises(TypeError):
            result = d + val
        with self.assertRaises(TypeError):
            d += val
        with self.assertRaises(TypeError):
            d -= val

        val = DateTimeNano(NANOS_PER_DAY, from_tz='GMT')
        with self.assertRaises(TypeError):
            result = d + val
        with self.assertRaises(TypeError):
            d += val
        with self.assertRaises(TypeError):
            d -= val

        # with self.assertRaises(TypeError):
        #    result = abs(d)

    def test_hstack(self):
        d = Date(1)
        result = hstack([d, d])
        self.assertTrue(isinstance(result, Date))
        self.assertEqual(len(result), 2)

        d = DateSpan(1)
        result = hstack([d, d])
        self.assertTrue(isinstance(result, DateSpan))
        self.assertEqual(len(result), 2)

    # removed Date, DateSpan from allowed groupby operation types
    # need a better list of computable functions
    # SJK 3/13/2019
    # def test_groupby(self):
    #    ds = Dataset({'col_'+str(i): np.random.rand(5) for i in range(5)})
    #    ds.dates = Date(DateTimeNano.random(5))
    #    ds.datespans = DateSpan(np.random.randint(0,100,5))

    #    result = ds.gb('dates').sum()
    #    self.assertTrue(isinstance(result.dates, Date))
    #    self.assertTrue(isinstance(result.datespans, DateSpan))
    #    self.assertTrue(issorted(result.dates))

    #    result = ds.gb('datespans').sum()
    #    self.assertTrue(isinstance(result.dates, Date))
    #    self.assertTrue(isinstance(result.datespans, DateSpan))
    #    self.assertTrue(issorted(result.datespans))

    def test_comparisons_scalar(self):
        d = Date(10)
        cmp_val = 1
        self.assertTrue((d > cmp_val)[0])
        self.assertTrue((d != cmp_val)[0])
        self.assertTrue((d >= cmp_val)[0])
        self.assertFalse((d < cmp_val)[0])
        self.assertFalse((d == cmp_val)[0])
        self.assertFalse((d <= cmp_val)[0])
        cmp_val = 10
        self.assertTrue((d == cmp_val)[0])

        cmp_val = '1970-01-02'
        self.assertTrue((d > cmp_val)[0])
        self.assertTrue((d != cmp_val)[0])
        self.assertTrue((d >= cmp_val)[0])
        self.assertFalse((d < cmp_val)[0])
        self.assertFalse((d == cmp_val)[0])
        self.assertFalse((d <= cmp_val)[0])
        cmp_val = '1970-01-11'
        self.assertTrue((d == cmp_val)[0])

        cmp_val = b'1970-01-02'
        self.assertTrue((d > cmp_val)[0])
        self.assertTrue((d != cmp_val)[0])
        self.assertTrue((d >= cmp_val)[0])
        self.assertFalse((d < cmp_val)[0])
        self.assertFalse((d == cmp_val)[0])
        self.assertFalse((d <= cmp_val)[0])
        cmp_val = b'1970-01-11'
        self.assertTrue((d == cmp_val)[0])

    def test_comparisons_categorical(self):
        d = Date(10)
        cmp_val = Categorical([b'1970-01-02'])
        self.assertTrue((d > cmp_val)[0])
        self.assertTrue((d != cmp_val)[0])
        self.assertTrue((d >= cmp_val)[0])
        self.assertFalse((d < cmp_val)[0])
        self.assertFalse((d == cmp_val)[0])
        self.assertFalse((d <= cmp_val)[0])
        cmp_val = Categorical([b'1970-01-11'])
        self.assertTrue((d == cmp_val)[0])

    def test_comparisons_nano(self):
        d = Date(10)
        cmp_val = DateTimeNano('1970-01-02', from_tz='NYC', to_tz='NYC')
        self.assertTrue((d > cmp_val)[0])
        self.assertTrue((d != cmp_val)[0])
        self.assertTrue((d >= cmp_val)[0])
        self.assertFalse((d < cmp_val)[0])
        self.assertFalse((d == cmp_val)[0])
        self.assertFalse((d <= cmp_val)[0])
        cmp_val = DateTimeNano('1970-01-11', from_tz='NYC', to_tz='NYC')
        self.assertTrue((d == cmp_val)[0])

    def test_comparisons_error(self):
        d = Date(10)
        with self.assertRaises(TypeError):
            _ = d > DateSpan(1)
        with self.assertRaises(TypeError):
            _ = d < TimeSpan(1)

    def test_comparisons_datespan_scalar(self):
        d = DateSpan(10)
        cmp_val = 1
        self.assertTrue((d > cmp_val)[0])
        self.assertTrue((d != cmp_val)[0])
        self.assertTrue((d >= cmp_val)[0])
        self.assertFalse((d < cmp_val)[0])
        self.assertFalse((d == cmp_val)[0])
        self.assertFalse((d <= cmp_val)[0])
        cmp_val = 10
        self.assertTrue((d == cmp_val)[0])

        cmp_val = 1.0
        self.assertTrue((d > cmp_val)[0])
        self.assertTrue((d != cmp_val)[0])
        self.assertTrue((d >= cmp_val)[0])
        self.assertFalse((d < cmp_val)[0])
        self.assertFalse((d == cmp_val)[0])
        self.assertFalse((d <= cmp_val)[0])
        cmp_val = 10.0
        self.assertTrue((d == cmp_val)[0])

    def test_comparisons_datespan_nano(self):
        d = DateSpan(10)
        cmp_val = TimeSpan(24, 'h')
        self.assertTrue((d > cmp_val)[0])
        self.assertTrue((d != cmp_val)[0])
        self.assertTrue((d >= cmp_val)[0])
        self.assertFalse((d < cmp_val)[0])
        self.assertFalse((d == cmp_val)[0])
        self.assertFalse((d <= cmp_val)[0])
        cmp_val = TimeSpan(240, 'h')
        self.assertTrue((d == cmp_val)[0])

    def test_comparisons_datespan_error(self):
        d = DateSpan(10)
        with self.assertRaises(TypeError):
            _ = d < Date(1)
        with self.assertRaises(TypeError):
            _ = d > DateTimeNano('1970-01-02', from_tz='NYC', to_tz='NYC')
        with self.assertRaises(TypeError):
            _ = d == Categorical(['1970-01-02'])

    def test_year(self):
        arr = full(50, 50) + (arange(50) * 365)
        d = Date(arr)
        result = d.year
        correct = arange(1970, 2020)
        self.assertTrue(bool(np.all(result == correct)))

    def test_month(self):
        d = Date(arange(12) * 32)
        result = d.month
        correct = arange(1, 13)
        self.assertTrue(bool(np.all(result[1:] == correct[1:])))

    def test_day_of_year(self):
        d = Date(arange(365) + 365)
        result = d.day_of_year
        correct = arange(1, 366)
        self.assertTrue(bool(np.all(result == correct)))

    def test_day_of_week(self):
        d = Date.range('2019-02-11', '2019-02-18', closed='left')
        result = d.day_of_week
        correct = arange(7)
        self.assertTrue(bool(np.all(result == correct)))

    def test_is_weekend(self):
        d = Date.range('2019-02-11', '2019-02-18', closed='left')
        result = d.is_weekend
        correct = full(7, False)
        correct[-2:] = True
        self.assertTrue(bool(np.all(result == correct)))

    def test_is_weekday(self):
        d = Date.range('2019-02-11', '2019-02-18', closed='left')
        result = d.is_weekday
        correct = full(7, True)
        correct[-2:] = False
        self.assertTrue(bool(np.all(result == correct)))

    def test_is_leapyear(self):
        d = Date.range('1992-06-01', '2031-06-01', step=365)
        result = d.is_leapyear
        correct = tile([True, False, False, False], 10)
        self.assertTrue(bool(np.all(result == correct)))

    def test_day_of_month(self):
        d = Date.range('2019-01-01', '2020-01-01', closed='left')
        daynums = []
        for idx, i in enumerate(YDAY_SPLITS[:-1]):
            arr = arange(1, YDAY_SPLITS[idx + 1] - i + 1)
            daynums.append(arr)
        daynums.append(arange(1, 32))
        result = d.day_of_month
        correct = hstack(daynums)

        d_leap = Date.range('2020-01-01', '2021-01-01', closed='left')
        daynums = []
        for idx, i in enumerate(YDAY_SPLITS_LEAP[:-1]):
            arr = arange(1, YDAY_SPLITS_LEAP[idx + 1] - i + 1)
            daynums.append(arr)
        daynums.append(arange(1, 32))
        result = d.day_of_month
        correct = hstack(daynums)

    def test_fill_invalid(self):
        dtn = Date.range(20190201, 20190207)
        dtn2 = dtn.fill_invalid(inplace=False)
        self.assertTrue(isinstance(dtn2, Date))
        self.assertTrue(bool(np.all(dtn2.isnan())))

        dtn2 = dtn.fill_invalid(inplace=False, shape=3)
        self.assertTrue(isinstance(dtn2, Date))
        self.assertTrue(bool(np.all(dtn2.isnan())))
        self.assertEqual(len(dtn2), 3)

        with self.assertRaises(ValueError):
            dtn.fill_invalid(inplace=True, shape=3)

        dtn.fill_invalid(inplace=True)
        self.assertTrue(bool(np.all(dtn.isnan())))

    def test_fill_invalid_span(self):
        dtn = DateSpan(arange(5))
        dtn2 = dtn.fill_invalid(inplace=False)
        self.assertTrue(isinstance(dtn2, DateSpan))
        self.assertTrue(bool(np.all(dtn2.isnan())))

        dtn2 = dtn.fill_invalid(inplace=False, shape=3)
        self.assertTrue(isinstance(dtn2, DateSpan))
        self.assertTrue(bool(np.all(dtn2.isnan())))
        self.assertEqual(len(dtn2), 3)

        with self.assertRaises(ValueError):
            dtn.fill_invalid(inplace=True, shape=3)

        dtn.fill_invalid(inplace=True)
        self.assertTrue(bool(np.all(dtn.isnan())))

    def test_min(self):
        d = Date(['20190201', '20190207', '20191230', '20190101'])
        m = d.min()
        self.assertTrue(isinstance(m, Date))
        self.assertTrue(m[0] == d[-1])

        ds = DateSpan([10, 5, 2, 3, 12])
        m = ds.min()
        self.assertTrue(isinstance(m, DateSpan))
        self.assertTrue(m[0] == ds[2])

    def test_max(self):
        d = Date(['20190201', '20190207', '20191230', '20190101'])
        m = d.max()
        self.assertTrue(isinstance(m, Date))
        self.assertTrue(m[0] == d[2])

        ds = DateSpan([10, 5, 2, 3, 12])
        m = ds.max()
        self.assertTrue(isinstance(m, DateSpan))
        self.assertTrue(m[0] == ds[-1])

    def test_range_closed(self):
        both = Date(['20190201', '20190202', '20190203', '20190204'])
        d_both = Date.range('20190201', '20190204')
        self.assertTrue(isinstance(d_both, Date))
        self.assertTrue(bool(np.all(both == d_both)))

        d_left = Date.range('20190201', '20190204', closed='left')
        self.assertTrue(isinstance(d_left, Date))
        self.assertTrue(bool(np.all(both[:-1] == d_left)))

        d_right = Date.range('20190201', '20190204', closed='right')
        self.assertTrue(isinstance(d_right, Date))
        self.assertTrue(bool(np.all(both[1:] == d_right)))

        with self.assertRaises(ValueError):
            d_error = Date.range('20190201', '20190204', closed='garbage')
        pass

    def test_start_of_week(self):
        d = Date(['20200507', '20200508'])
        d2 = Date(['20200504', '20200504'])
        self.assertTrue((d.start_of_week==d2).all())

    def test_start_of_month(self):
        d = Date(['20200507', '20200508'])
        d2 = Date(['20200501', '20200501'])
        self.assertTrue((d.start_of_month==d2).all())




if __name__ == '__main__':
    tester = unittest.main()
