import unittest

from riptable import *
from riptable.rt_datetime import (
    NANOS_PER_HOUR,
)


class DateTimeMath_Test(unittest.TestCase):
    def test_add_scalar(self):
        dtn = DateTimeNano(
            [
                'Inv',
                '19810421 09:06:41.220080800',
                '19980125 03:02:11.912821440',
                'Inv',
                'Inv',
            ],
            from_tz='NYC',
        )
        result = dtn + 1
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)

        result = dtn + 1.0
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)

        result = dtn + [1]
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)

        result = dtn + [1.0]
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)

    def test_iadd_scalar(self):
        dtn = DateTimeNano(
            [
                'Inv',
                '19810421 09:06:41.220080800',
                '19980125 03:02:11.912821440',
                'Inv',
                'Inv',
            ],
            from_tz='NYC',
        )
        dtn += 1
        result = dtn
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)

        dtn = DateTimeNano(
            [
                'Inv',
                '19810421 09:06:41.220080800',
                '19980125 03:02:11.912821440',
                'Inv',
                'Inv',
            ],
            from_tz='NYC',
        )
        dtn += 1.0
        result = dtn
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)

        dtn = DateTimeNano(
            [
                'Inv',
                '19810421 09:06:41.220080800',
                '19980125 03:02:11.912821440',
                'Inv',
                'Inv',
            ],
            from_tz='NYC',
        )
        dtn += [1]
        result = dtn
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)

        dtn = DateTimeNano(
            [
                'Inv',
                '19810421 09:06:41.220080800',
                '19980125 03:02:11.912821440',
                'Inv',
                'Inv',
            ],
            from_tz='NYC',
        )
        dtn += [1.0]
        result = dtn
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)

    def test_add_array(self):
        dtn = DateTimeNano(
            [
                '19881114 22:58:18.875374947',
                'Inv',
                '19781014 09:09:50.558413019',
                'Inv',
                '20180606 14:45:20.799048559',
            ],
            from_tz='NYC',
        )
        result = dtn + full(5, 500)
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[1], 0)

        val = full(5, 500, dtype=np.float32)
        result = dtn + val
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[1], 0)
        val[0] = np.nan
        result = dtn + val
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)
        self.assertEqual(result._fa[1], 0)

    def test_iadd_array(self):
        dtn = DateTimeNano(
            [
                '19881114 22:58:18.875374947',
                'Inv',
                '19781014 09:09:50.558413019',
                'Inv',
                '20180606 14:45:20.799048559',
            ],
            from_tz='NYC',
        )
        dtn += full(5, 500)
        result = dtn
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[1], 0)

        dtn = DateTimeNano(
            [
                '19881114 22:58:18.875374947',
                'Inv',
                '19781014 09:09:50.558413019',
                'Inv',
                '20180606 14:45:20.799048559',
            ],
            from_tz='NYC',
        )
        val = full(5, 500, dtype=np.float32)
        dtn += val
        result = dtn

        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[1], 0)
        val[0] = np.nan
        dtn = DateTimeNano(
            [
                '19881114 22:58:18.875374947',
                'Inv',
                '19781014 09:09:50.558413019',
                'Inv',
                '20180606 14:45:20.799048559',
            ],
            from_tz='NYC',
        )
        dtn += val
        result = dtn
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)
        self.assertEqual(result._fa[1], 0)

    def test_add_dtclass(self):
        dtn = DateTimeNano(
            [
                '20110719 06:30:28.899581754',
                '19800121 20:54:42.763693173',
                'Inv',
                'Inv',
                '19820625 05:59:45.813906699',
            ],
            from_tz='NYC',
        )
        with self.assertRaises(TypeError):
            dtn2 = dtn + dtn

        with self.assertRaises(TypeError):
            dtn2 = dtn + Date(1)

        result = dtn + TimeSpan(1)
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[2], 0)

        result = dtn + DateSpan(1)
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[2], 0)

    def test_sub_dtclass(self):
        # INT64 - INT64 => FLOAT64 (both checked, nan fill)
        dtn = DateTimeNano(
            [
                'Inv',
                'Inv',
                '20010911 05:25:55.782094544',
                '20150217 21:44:37.149415405',
                'Inv',
            ],
            from_tz='NYC',
        )
        val = DateTimeNano(
            [
                'Inv',
                'Inv',
                '20010911 04:25:55.782094544',
                '20150217 20:44:37.149415405',
                'Inv',
            ],
            from_tz='NYC',
        )
        result = dtn - val
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(isnan(result._fa[0]))

        # INT64 - INT64 => FLOAT64 (both checked for 0, nan fill)
        val = Date(1)
        result = dtn - val
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(isnan(result._fa[0]))

        # INT64 - INT64 => INT64 (left checked for 0, right checked for INT64 sentinel, 0 fill)
        val = DateSpan(1)
        result = dtn - val
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)

        # INT64 - INT64 (upcast to int) => INT64 (left checked for 0, right checked for INT64 sentinel, 0 fill)
        val = TimeSpan(1)
        result = dtn - val
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)

    def test_sub_scalar(self):
        dtn = DateTimeNano(
            [
                'Inv',
                '19810421 09:06:41.220080800',
                '19980125 03:02:11.912821440',
                'Inv',
                'Inv',
            ],
            from_tz='NYC',
        )
        result = dtn - 1
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)
        self.assertEqual(dtn._fa[1] - 1, result._fa[1])

        result = dtn - 1.0
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)
        self.assertEqual(dtn._fa[1] - 1, result._fa[1])

        result = dtn - [1]
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)
        self.assertEqual(dtn._fa[1] - 1, result._fa[1])

        result = dtn - [1.0]
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)
        self.assertEqual(dtn._fa[1] - 1, result._fa[1])

    def test_isub_scalar(self):
        dtn = DateTimeNano(
            [
                'Inv',
                '19810421 09:06:41.220080800',
                '19980125 03:02:11.912821440',
                'Inv',
                'Inv',
            ],
            from_tz='NYC',
        )
        dtn -= 1
        result = dtn
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)

        dtn = DateTimeNano(
            [
                'Inv',
                '19810421 09:06:41.220080800',
                '19980125 03:02:11.912821440',
                'Inv',
                'Inv',
            ],
            from_tz='NYC',
        )
        dtn -= 1.0
        result = dtn
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)

        dtn = DateTimeNano(
            [
                'Inv',
                '19810421 09:06:41.220080800',
                '19980125 03:02:11.912821440',
                'Inv',
                'Inv',
            ],
            from_tz='NYC',
        )
        dtn -= [1]
        result = dtn
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)

        dtn = DateTimeNano(
            [
                'Inv',
                '19810421 09:06:41.220080800',
                '19980125 03:02:11.912821440',
                'Inv',
                'Inv',
            ],
            from_tz='NYC',
        )
        dtn -= [1.0]
        result = dtn
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)

    def test_sub_array(self):
        dtn = DateTimeNano(
            [
                '19881114 22:58:18.875374947',
                'Inv',
                '19781014 09:09:50.558413019',
                'Inv',
                '20180606 14:45:20.799048559',
            ],
            from_tz='NYC',
        )
        result = dtn - full(5, 500)
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[1], 0)

        val = full(5, 500, dtype=np.float32)
        result = dtn - val
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[1], 0)
        val[0] = np.nan
        result = dtn - val
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)
        self.assertEqual(result._fa[1], 0)

    def test_isub_array(self):
        dtn = DateTimeNano(
            [
                '19881114 22:58:18.875374947',
                'Inv',
                '19781014 09:09:50.558413019',
                'Inv',
                '20180606 14:45:20.799048559',
            ],
            from_tz='NYC',
        )
        dtn -= full(5, 500)
        result = dtn
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[1], 0)

        dtn = DateTimeNano(
            [
                '19881114 22:58:18.875374947',
                'Inv',
                '19781014 09:09:50.558413019',
                'Inv',
                '20180606 14:45:20.799048559',
            ],
            from_tz='NYC',
        )
        val = full(5, 500, dtype=np.float32)
        dtn -= val
        result = dtn
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[1], 0)
        val[0] = np.nan
        dtn = DateTimeNano(
            [
                '19881114 22:58:18.875374947',
                'Inv',
                '19781014 09:09:50.558413019',
                'Inv',
                '20180606 14:45:20.799048559',
            ],
            from_tz='NYC',
        )
        dtn -= val
        result = dtn
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], 0)
        self.assertEqual(result._fa[1], 0)

    def test_timespan_add_scalar(self):
        ts = TimeSpan([10, 300, 456, 20, np.nan], unit='m')
        result = ts + 1
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(isnan(result._fa[4]))

        result = ts + [1]
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(isnan(result._fa[4]))

        result = ts + FA(1)
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(isnan(result._fa[4]))

        result = ts + 1.0
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(isnan(result._fa[4]))

        result = ts + [1.0]
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(isnan(result._fa[4]))

        result = ts + FA(1.0)
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(isnan(result._fa[4]))

    def test_timespan_add_array(self):
        ts = TimeSpan([10, 300, 456, 20, np.nan], unit='m')
        arr = full(5, 1_000_000_000, dtype=np.float64)
        result = ts + arr
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(isnan(result._fa[4]))
        arr[0] = np.nan
        result = ts + arr
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(isnan(result._fa[4]))
        self.assertTrue(isnan(result._fa[0]))
        arr = arr.astype(np.int64)
        result = ts + arr
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(isnan(result._fa[4]))
        self.assertTrue(isnan(result._fa[0]))

    def test_dtn_compare_date(self):

        dtn = DateTimeNano(['20190227'], from_tz='NYC', to_tz='NYC').repeat(24)
        dtn = dtn + TimeSpan(arange(24), unit='h')
        self.assertTrue(bool(np.all(dtn.hour == arange(24))))

        LT = dtn < Date('20190228')
        self.assertTrue(bool(np.all(LT)))

        GT = dtn > Date('20190227')
        self.assertTrue(bool(np.all(GT[1:])))
        self.assertFalse(GT[0])

        dtn = dtn + NANOS_PER_HOUR
        EQ = dtn == Date('20190228')
        self.assertTrue(EQ[-1])

        dtn = DateTimeNano(['20190227'], from_tz='GMT', to_tz='GMT').repeat(24)
        dtn = dtn + TimeSpan(arange(24), unit='h')
        self.assertTrue(bool(np.all(dtn.hour == arange(24))))

        LT = dtn < Date('20190228')
        self.assertTrue(bool(np.all(LT)))

        GT = dtn > Date('20190227')
        self.assertTrue(bool(np.all(GT[1:])))
        self.assertFalse(GT[0])

        dtn = dtn + NANOS_PER_HOUR
        EQ = dtn == Date('20190228')
        self.assertTrue(EQ[-1])

    def test_diff(self):
        dt = Date(['201910%02d' % i for i in range(1, 31)])
        d = dt.diff()
        self.assertTrue(isinstance(d, DateSpan))
        dt = utcnow(30)
        d = dt.diff()
        self.assertTrue(isinstance(d, TimeSpan))


if __name__ == '__main__':
    tester = unittest.main()
