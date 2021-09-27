import unittest
import pytest
import warnings
import numpy as np
import os
import datetime

from dateutil import tz
from riptable import *
from riptable.rt_datetime import (
    NANOS_PER_HOUR,
    NANOS_PER_SECOND,
    NANOS_PER_DAY,
    NANOS_PER_YEAR,
)
from riptable.rt_sds import SDSMakeDirsOn

# change to true since we write into /tests directory
SDSMakeDirsOn()


start = 1539611143000000000
step = 3_600_000_000_000
span_min = 60_000_000_000
span_max = 300_000_000_000
dtinv = INVALID_DICT[np.dtype(np.int64).num]


def random_dtn(sz, to_tz='NYC', from_tz='NYC', inv_mask=None):
    arr = np.random.randint(NANOS_PER_YEAR, NANOS_PER_YEAR * 40, sz, dtype=np.int64)
    if inv_mask is not None:
        putmask(arr, inv_mask, 0)
    return DateTimeNano(arr, to_tz=to_tz, from_tz=from_tz)


def arr_eq(a, b):
    return bool(np.all(a == b))


def arr_all(a):
    return bool(np.all(a))


class DateTime_Test(unittest.TestCase):
    def test_nano_add(self):
        a = DateTimeNano(
            FA([start + step * i for i in range(7)], dtype=np.int64),
            from_tz='GMT',
            to_tz='NYC',
        )
        result = a + 400
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], start + 400)
        self.assertTrue(result.dtype == np.int64)

        result = 400 + a
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], start + 400)
        self.assertTrue(result.dtype == np.int64)

        result = a + 400.0
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], start + 400.0)
        self.assertTrue(result.dtype == np.int64)

        result = 400.0 + a
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], start + 400.0)
        self.assertTrue(result.dtype == np.int64)

        result = a + np.full(7, 400)
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], start + 400)
        self.assertTrue(result.dtype == np.int64)

        result = np.full(7, 400) + a
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], start + 400)
        self.assertTrue(result.dtype == np.int64)

        result = a + np.full(7, 400.0)
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], start + 400.0)
        self.assertTrue(result.dtype == np.int64)

        result = np.full(7, 400.0) + a
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], start + 400.0)
        self.assertTrue(result.dtype == np.int64)

        with self.assertRaises(TypeError):
            result = a + a

        b = TimeSpan(FA(np.full(7, step)))
        result = a + b
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], start + step)
        self.assertTrue(result.dtype == np.int64)

    def test_nano_sub(self):
        a = DateTimeNano(
            FA([start + step * i for i in range(7)], dtype=np.int64),
            from_tz='GMT',
            to_tz='NYC',
        )
        b = a - span_min
        result = a - b
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertEqual(result._fa[0], span_min)
        self.assertTrue(result.dtype == np.float64)

        result = a - 400
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertTrue(result._fa[0] == start - 400)
        self.assertTrue(result.dtype == np.int64)

        result = a - 400.0
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], start - 400.0)
        self.assertTrue(result.dtype == np.int64)

        result = a - np.full(7, 400)
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertTrue(result._fa[0] == start - 400)
        self.assertTrue(result.dtype == np.int64)

        result = a - np.full(7, 400.0)
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertEqual(result._fa[0], start - 400.0)
        self.assertTrue(result.dtype == np.int64)

    def test_nano_math_errors(self):
        a = DateTimeNano(
            FA([start + step * i for i in range(7)], dtype=np.int64),
            from_tz='GMT',
            to_tz='NYC',
        )

        with self.assertRaises(TypeError):
            _ = a * a
        with self.assertRaises(TypeError):
            _ = a / a
        with self.assertRaises(TypeError):
            _ = a // a

    def test_span_add(self):
        b = TimeSpan(
            np.random.randint(-span_max, span_max, 7, dtype=np.int64).astype(np.float64)
        )

        c = TimeSpan(
            np.random.randint(-span_max, span_max, 7, dtype=np.int64).astype(np.float64)
        )
        result = b + c
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(result.dtype == np.float64)

        c = DateTimeNano(
            FA([start + step * i for i in range(7)], dtype=np.int64),
            from_tz='GMT',
            to_tz='NYC',
        )
        result = b + c
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertTrue(result.dtype == np.int64)

        result = b + 400
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(result.dtype == np.float64)

        result = b + 400.0
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(result.dtype == np.float64)

        result = b + np.full(7, 400)
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(result.dtype == np.float64)

        result = b + np.full(7, 400.0)
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(result.dtype == np.float64)

    def test_span_sub(self):
        b = TimeSpan(
            np.random.randint(-span_max, span_max, 7, dtype=np.int64).astype(np.float64)
        )

        c = TimeSpan(
            np.random.randint(-span_max, span_max, 7, dtype=np.int64).astype(np.float64)
        )
        result = b - c
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(result.dtype == np.float64)

        result = b - 400
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(result.dtype == np.float64)

        result = b - 400.0
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(result.dtype == np.float64)

        result = b - np.full(7, 400)
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(result.dtype == np.float64)

        result = b - np.full(7, 400.0)
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(result.dtype == np.float64)

    def test_span_unary(self):
        b = TimeSpan(
            np.random.randint(-span_max, span_max, 7, dtype=np.int64).astype(np.float64)
        )

    def test_save_load(self):
        temp_name = 'dtset' + str(np.random.randint(1, 1_000_000))
        temp_path = (
            os.path.dirname(os.path.abspath(__file__))
            + os.path.sep
            + 'temp'
            + os.path.sep
            + temp_name
            + '.sds'
        )

        a = DateTimeNano(
            FA([start + step * i for i in range(7)], dtype=np.int64),
            from_tz='GMT',
            to_tz='NYC',
        )
        b = TimeSpan(
            np.random.randint(-span_max, span_max, 7, dtype=np.int64).astype(np.float64)
        )

        ds1 = Dataset({'dt': a, 'dtspan': b})
        ds1.save(temp_path)

        ds2 = Dataset.load(temp_path)

        self.assertTrue(isinstance(ds2.dt, DateTimeNano))
        match = bool(np.all(ds1.dt._fa == ds2.dt._fa))
        self.assertTrue(match)

        self.assertTrue(isinstance(ds2.dtspan, TimeSpan))
        match = bool(np.all(ds1.dtspan._fa == ds2.dtspan._fa))
        self.assertTrue(match)

        os.remove(temp_path)

    def test_nano_index(self):
        a = DateTimeNano(
            FA([start + step * i for i in range(7)], dtype=np.int64),
            from_tz='GMT',
            to_tz='NYC',
        )

        f = np.array([True, False, True, True, True, False, False])
        result = a[f]
        self.assertTrue(isinstance(result, DateTimeNano))
        fa_result = a._fa[f]
        match = bool(np.all(result._fa == fa_result))
        self.assertTrue(match)

        idx = [1, 5]
        result = a[idx]
        self.assertTrue(isinstance(result, DateTimeNano))
        fa_result = a._fa[idx]
        match = bool(np.all(result._fa == fa_result))
        self.assertTrue(match)

        slc = slice(None, 3, None)
        result = a[slc]
        self.assertTrue(isinstance(result, DateTimeNano))
        fa_result = a._fa[slc]
        match = bool(np.all(result._fa == fa_result))
        self.assertTrue(match)

        result = a[0]
        self.assertTrue(start == result)

    def test_init_strings(self):
        result = DateTimeNano(
            [
                '2018-11-02 09:30:00.177080',
                '2018-11-02 09:30:00.228403',
                '2018-11-02 09:30:00.228458',
                '2018-11-02 09:30:00.228977',
                '2018-11-02 09:30:00.229061',
            ],
            from_tz='NYC',
            to_tz='NYC',
        )
        correct = FastArray(
            [
                1541165400177080000,
                1541165400228403000,
                1541165400228458000,
                1541165400228977000,
                1541165400229061000,
            ],
            dtype=np.int64,
        )
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertTrue(result.dtype == np.int64)
        self.assertTrue(bool(np.all(result._fa == correct)))

    def test_init_python_dt(self):
        pdts = [datetime.datetime(2018, 11, i) for i in range(1, 6)]
        result = DateTimeNano(pdts, from_tz='NYC', to_tz='NYC')
        correct = FastArray(
            [
                1541044800000000000,
                1541131200000000000,
                1541217600000000000,
                1541304000000000000,
                1541394000000000000,
            ],
            dtype=np.int64,
        )
        self.assertTrue(isinstance(result, DateTimeNano))
        self.assertTrue(result.dtype == np.int64)
        self.assertTrue(bool(np.all(result._fa == correct)))

    def test_convert_matlab(self):
        ds = Dataset(
            {
                'dtcol': FastArray(
                    [
                        1541044800000000000,
                        1541131200000000000,
                        1541217600000000000,
                        1541304000000000000,
                        1541394000000000000,
                    ],
                    dtype=np.int64,
                )
            }
        )

        ds.make_matlab_datetimes('dtcol')
        self.assertTrue(isinstance(ds.dtcol, DateTimeNano))
        self.assertEqual(1541044800000000000, ds.dtcol[0])

    def test_to_iso(self):
        dtn = DateTimeNano(
            [
                1541044800000000000,
                1541131200000000000,
                1541217600000000000,
                1541304000000000000,
                1541394000000000000,
            ],
            from_tz='NYC',
            to_tz='NYC',
        )
        result = dtn.to_iso()
        correct = FastArray(
            [
                '2018-11-01T04:00:00.000000000',
                '2018-11-02T04:00:00.000000000',
                '2018-11-03T04:00:00.000000000',
                '2018-11-04T04:00:00.000000000',
                '2018-11-05T05:00:00.000000000',
            ]
        )
        self.assertTrue(bool(np.all(result == correct)))
        self.assertEqual(result.dtype.char, 'S')

    def test_year(self):
        dtn = DateTimeNano(
            [1546297200000000000, 1546304400000000000], from_tz='NYC', to_tz='NYC'
        )
        correct = FastArray([2018, 2019])
        result = dtn.year()
        self.assertTrue(isinstance(result, FastArray))
        self.assertTrue(bool(np.all(result == correct)))

        dtn = DateTimeNano(
            [1546315200000000000, 1546322400000000000], from_tz='GMT', to_tz='NYC'
        )
        result = dtn.year()
        self.assertTrue(isinstance(result, FastArray))
        self.assertTrue(bool(np.all(result == correct)))

    def test_month(self):
        dtn = DateTimeNano(
            [1546297200000000000, 1546304400000000000], from_tz='NYC', to_tz='NYC'
        )
        correct = FastArray([12, 1])
        result = dtn.month()
        self.assertTrue(isinstance(result, FastArray))
        self.assertTrue(bool(np.all(result == correct)))

        dtn = DateTimeNano(
            [1546315200000000000, 1546322400000000000], from_tz='GMT', to_tz='NYC'
        )
        self.assertTrue(isinstance(result, FastArray))
        self.assertTrue(bool(np.all(result == correct)))

    def test_day(self):
        dtn = DateTimeNano(
            ['2018-12-28 06:00:00', '2018-12-28 12:00:00', '2018-12-28 18:00:00'],
            from_tz='NYC',
            to_tz='NYC',
        )
        correct = FastArray([0.25, 0.5, 0.75])
        result = dtn.day
        self.assertTrue(isinstance(result, FastArray))
        self.assertTrue(bool(np.all(result == correct)))

    def test_days(self):
        dtn1 = DateTimeNano(['2019-01-08'], from_tz='NYC', to_tz='NYC')
        dtn2 = DateTimeNano(['2019-01-05'], from_tz='NYC', to_tz='NYC')
        difference = dtn1 - dtn2
        self.assertTrue(isinstance(difference, TimeSpan))
        self.assertEqual(difference.days[0], 3)

    def test_hour(self):
        dtn = DateTimeNano(
            [1546305300000000000, 1546307100000000000], from_tz='NYC', to_tz='NYC'
        )
        correct = FastArray([1.25, 1.75])
        result = dtn.hour
        self.assertTrue(isinstance(result, FastArray))
        self.assertTrue(bool(np.all(result == correct)))

        result = dtn.hour_span
        self.assertTrue(isinstance(result, TimeSpan))
        result = result.hours
        self.assertTrue(bool(np.all(result == correct)))

        dtn = DateTimeNano(
            [1546323300000000000, 1546325100000000000], from_tz='GMT', to_tz='NYC'
        )
        result = dtn.hour
        self.assertTrue(isinstance(result, FastArray))
        self.assertTrue(bool(np.all(result == correct)))

        result = dtn.hour_span
        self.assertTrue(isinstance(result, TimeSpan))
        result = result.hours
        self.assertTrue(bool(np.all(result == correct)))

    def test_minute(self):
        dtn = DateTimeNano(
            [1546305315000000000, 1546307145000000000], from_tz='NYC', to_tz='NYC'
        )
        correct = FastArray([15.25, 45.75])
        result = dtn.minute
        self.assertTrue(isinstance(result, FastArray))
        self.assertTrue(bool(np.all(result == correct)))

        result = dtn.minute_span
        self.assertTrue(isinstance(result, TimeSpan))
        result = result.minutes
        self.assertTrue(bool(np.all(result == correct)))

        dtn = DateTimeNano(
            [1546323315000000000, 1546325145000000000], from_tz='GMT', to_tz='NYC'
        )
        result = dtn.minute
        self.assertTrue(isinstance(result, FastArray))
        self.assertTrue(bool(np.all(result == correct)))

        result = dtn.minute_span
        self.assertTrue(isinstance(result, TimeSpan))
        result = result.minutes
        self.assertTrue(bool(np.all(result == correct)))

    def test_second(self):
        dtn = DateTimeNano(
            [1546307145250000000, 1546307146750000000], from_tz='GMT', to_tz='NYC'
        )
        correct = FastArray([45.25, 46.75])
        result = dtn.second
        self.assertTrue(isinstance(result, FastArray))
        self.assertTrue(bool(np.all(result == correct)))

        result = dtn.second_span
        self.assertTrue(isinstance(result, TimeSpan))
        result = result.seconds
        self.assertTrue(bool(np.all(result == correct)))

    def test_second_fraction(self):
        dts = DateTimeNano(
            [1546307145250000000, 1546307146750000000], from_tz='GMT', to_tz='NYC'
        ).second_span

        correct_ms = FastArray([45250.0, 46750.0])
        result = dts.milliseconds
        self.assertTrue(isinstance(result, FastArray))
        self.assertTrue(bool(np.all(result == correct_ms)))

        correct_us = FastArray([45250000.0, 46750000.0])
        result = dts.microseconds
        self.assertTrue(isinstance(result, FastArray))
        self.assertTrue(bool(np.all(result == correct_us)))

        correct_ns = FastArray([45250000000.0, 46750000000.0])
        result = dts.nanoseconds
        self.assertTrue(isinstance(result, FastArray))
        self.assertTrue(bool(np.all(result == correct_ns)))

    def test_dst_fall(self):
        dtn = DateTimeNano(
            [1541239200000000000, 1541325600000000000], from_tz='NYC', to_tz='NYC'
        )
        correct = FastArray([10.0, 10.0])
        result = dtn.hour
        self.assertTrue(bool(np.all(correct == result)))

        dtn = DateTimeNano(
            [1541239200000000000, 1541325600000000000], from_tz='GMT', to_tz='NYC'
        )
        correct = FastArray([6.0, 5.0])
        result = dtn.hour
        self.assertTrue(bool(np.all(correct == result)))

    def test_to_iso_dst_fall(self):

        # test a daylight savings day change
        # 3 hours were added to the underlying array, but only two hours changed because of time change

        # NYC
        correct1 = b'2018-11-03T23:59:00.000000000'
        dtn = DateTimeNano(['2018-11-03 23:59'], from_tz='NYC', to_tz='NYC')
        stamp1 = dtn.to_iso()[0]
        self.assertEqual(correct1, stamp1)

        # these strings will be the same
        correct2 = b'2018-11-04T01:59:00.000000000'
        dtn2 = DateTimeNano(dtn._fa + NANOS_PER_HOUR * 2, from_tz='GMT', to_tz='NYC')
        stamp2 = dtn2.to_iso()[0]
        self.assertEqual(correct2, stamp2)

        dtn3 = DateTimeNano(dtn._fa + NANOS_PER_HOUR * 3, from_tz='GMT', to_tz='NYC')
        stamp3 = dtn3.to_iso()[0]
        self.assertEqual(correct2, stamp3)

        # UTC nano will be different
        self.assertNotEqual(dtn2._fa[0], dtn3._fa[0])

        # fix dublin tests...

        # DUBLIN
        correct1 = b'2019-10-26T23:59:00.000000000'
        dtn = DateTimeNano(['2019-10-26 23:59'], from_tz='DUBLIN', to_tz='DUBLIN')
        stamp1 = dtn.to_iso()[0]
        self.assertEqual(correct1, stamp1)

        correct2 = b'2019-10-27T01:59:00.000000000'
        dtn2 = DateTimeNano(dtn._fa + NANOS_PER_HOUR * 3, from_tz='GMT', to_tz='DUBLIN')
        stamp2 = dtn2.to_iso()[0]
        self.assertEqual(correct2, stamp2)

        # test a normal day change
        # 3 hours added to underlying array, 3 hour change displayed

        # NYC
        correct1 = b'2018-06-06T23:59:00.000000000'
        dtn = DateTimeNano(['2018-06-06 23:59'], from_tz='NYC')
        stamp1 = dtn.to_iso()[0]
        self.assertEqual(correct1, stamp1)

        correct2 = b'2018-06-07T02:59:00.000000000'
        dtn2 = DateTimeNano(dtn._fa + NANOS_PER_HOUR * 3, from_tz='GMT', to_tz='NYC')
        stamp2 = dtn2.to_iso()[0]
        self.assertEqual(correct2, stamp2)

        # DUBLIN
        dtn = DateTimeNano(['2018-06-06 23:59'], from_tz='DUBLIN', to_tz='DUBLIN')
        stamp1 = dtn.to_iso()[0]
        self.assertEqual(correct1, stamp1)

        dtn2 = DateTimeNano(dtn._fa + NANOS_PER_HOUR * 3, from_tz='GMT', to_tz='DUBLIN')
        stamp2 = dtn2.to_iso()[0]
        self.assertEqual(correct2, stamp2)

    def test_to_iso_dst_spring(self):
        correct1 = b'2018-03-10T23:59:00.000000000'
        dtn = DateTimeNano(['2018-03-10 23:59'], from_tz='NYC', to_tz='NYC')
        stamp1 = dtn.to_iso()[0]
        self.assertEqual(correct1, stamp1)

        correct2 = b'2018-03-11T03:59:00.000000000'
        dtn2 = DateTimeNano(dtn._fa + NANOS_PER_HOUR * 3, from_tz='GMT', to_tz='NYC')
        stamp2 = dtn2.to_iso()[0]
        self.assertEqual(correct2, stamp2)

        correct1 = b'2019-03-30T23:59:00.000000000'
        dtn = DateTimeNano(['2019-03-30 23:59'], from_tz='DUBLIN', to_tz='DUBLIN')
        stamp1 = dtn.to_iso()[0]
        self.assertEqual(correct1, stamp1)

        correct2 = b'2019-03-31T03:59:00.000000000'
        dtn2 = DateTimeNano(dtn._fa + NANOS_PER_HOUR * 3, from_tz='GMT', to_tz='DUBLIN')
        stamp2 = dtn2.to_iso()[0]
        self.assertEqual(correct2, stamp2)

    def set_timezone(self):
        correct_utcnano = 1546875360000000000

        correct_nyc = b'2019-01-07T10:36:00.000000000'
        dtn = DateTimeNano(['2019-01-07 10:36'], from_tz='NYC', to_tz='NYC')
        stamp_nyc = dtn.to_iso()[0]
        self.assertEqual(stamp_nyc, correct_nyc)
        self.assertEqual(dtn._fa[0], correct_utcnano)

        dtn.set_timezone('DUBLIN')

        correct_dublin = b'2019-01-07T15:36:00.000000000'
        stamp_dublin = dtn.to_iso()[0]
        self.assertEqual(stamp_dublin, correct_dublin)
        self.assertEqual(dtn._fa[0], correct_utcnano)

        self.assertEqual(dtn._timezone._timezone_str, 'Europe/Dublin')
        self.assertEqual(dtn._timezone._to_tz, 'DUBLIN')

        dtn.set_timezone('GMT')

        correct_gmt = b'2019-01-07T15:36:00.000000000'
        stamp_gmt = dtn.to_iso()[0]
        self.assertEqual(stamp_gmt, correct_gmt)
        self.assertEqual(dtn._fa[0], correct_utcnano)

        self.assertEqual(dtn._timezone._timezone_str, 'GMT')
        self.assertEqual(dtn._timezone._to_tz, 'GMT')

        with self.assertRaises(ValueError):
            dtn.set_timezone('JUNK')

    def test_shift(self):
        dtn = DateTimeNano(
            [
                '2018-11-02 09:30:00.177080',
                '2018-11-02 09:30:00.228403',
                '2018-11-02 09:30:00.228458',
                '2018-11-02 09:30:00.228977',
                '2018-11-02 09:30:00.229061',
            ],
            from_tz='NYC',
            to_tz='NYC',
        )
        dtnfa = dtn._fa

        pos_shift = dtn.shift(2)
        pos_shift_fa = dtnfa.shift(2)
        self.assertTrue(isinstance(pos_shift, DateTimeNano))
        pos_shift = pos_shift._fa
        self.assertTrue(bool(np.all(pos_shift_fa == pos_shift)))

        neg_shift = dtn.shift(-2)
        neg_shift_fa = dtnfa.shift(-2)
        self.assertTrue(isinstance(neg_shift, DateTimeNano))
        neg_shift = neg_shift._fa
        self.assertTrue(bool(np.all(neg_shift_fa == neg_shift)))

    def test_date(self):
        dtn = DateTimeNano(
            [
                '2018-11-02 00:30:00.177080',
                '2018-11-02 01:30:00.228403',
                '2018-11-02 02:30:00.228458',
                '2018-11-02 03:30:00.228977',
                '2018-11-02 04:30:00.229061',
                '2018-11-02 05:30:00.177080',
                '2018-11-02 06:30:00.228403',
                '2018-11-02 07:30:00.228458',
                '2018-11-02 08:30:00.228977',
                '2018-11-02 09:30:00.229061',
                '2018-11-02 10:30:00.177080',
                '2018-11-02 11:30:00.228403',
                '2018-11-02 12:30:00.228458',
                '2018-11-02 13:30:00.228977',
                '2018-11-02 14:30:00.229061',
                '2018-11-02 15:30:00.177080',
                '2018-11-02 16:30:00.228403',
                '2018-11-02 17:30:00.228458',
                '2018-11-02 18:30:00.228977',
                '2018-11-02 19:30:00.229061',
                '2018-11-02 20:30:00.177080',
                '2018-11-02 21:30:00.228403',
                '2018-11-02 22:30:00.228458',
                '2018-11-02 23:30:00.228977',
            ],
            from_tz='NYC',
            to_tz='NYC',
        )

        d = dtn.date()
        self.assertTrue(isinstance(d, DateTimeNano))
        d = d._fa
        self.assertTrue(bool(np.all(d == 1541131200000000000)))

    def test_days_since_epoch(self):
        dtn = DateTimeNano(
            [
                '2018-11-01T00:00:00.000000000',
                '2018-11-02T00:00:00.000000000',
                '2018-11-03T00:00:00.000000000',
                '2018-11-03T23:00:00.000000000',
                '2018-11-05T00:00:00.000000000',
            ],
            from_tz='NYC',
            to_tz='NYC',
        )
        result = dtn.days_since_epoch
        correct = FastArray([17836, 17837, 17838, 17838, 17840], dtype=np.int64)
        self.assertTrue(bool(np.all(result == correct)))

        dtn = DateTimeNano([1, 2, 3], from_tz='GMT', to_tz='GMT')
        result = dtn.days_since_epoch
        correct = FastArray([0, 0, 0], dtype=np.int64)
        self.assertTrue(bool(np.all(result == correct)))

    def test_timestrings(self):
        ts = FastArray(
            [
                '1:30:00',
                '01:30:00',
                '1:30:00.000000',
                '01.30.00',
                '01:30:00.0',
                '1:30:00.00000000000',
            ]
        )
        result = timestring_to_nano(ts, from_tz='NYC', to_tz='NYC')
        self.assertTrue(isinstance(result, TimeSpan))
        result = result.astype(np.int64)
        self.assertTrue(bool(np.all(result == 5400000000000)))

        dstring = '2018-01-01'
        result = timestring_to_nano(ts, date=dstring, from_tz='NYC', to_tz='NYC')
        self.assertTrue(isinstance(result, DateTimeNano))
        result = result._fa
        self.assertTrue(bool(np.all(result == 1514788200000000000)))

        dstrings = full(6, dstring)
        result = timestring_to_nano(ts, date=dstring, from_tz='NYC', to_tz='NYC')
        self.assertTrue(isinstance(result, DateTimeNano))
        result = result._fa
        self.assertTrue(bool(np.all(result == 1514788200000000000)))

    def test_datestrings(self):
        dates = FastArray(['2018-01-01', '20180101', '2018.01.01'])
        result = datestring_to_nano(dates, from_tz='NYC', to_tz='NYC')
        self.assertTrue(isinstance(result, DateTimeNano))
        result = result._fa
        self.assertTrue(bool(np.all(result == 1514782800000000000)))

        tstring = '1:30:00'
        result = datestring_to_nano(dates, time=tstring, from_tz='NYC', to_tz='NYC')
        self.assertTrue(isinstance(result, DateTimeNano))
        result = result._fa
        self.assertTrue(bool(np.all(result == 1514788200000000000)))

        tstrings = full(3, tstring)
        result = datestring_to_nano(dates, time=tstrings, from_tz='NYC', to_tz='NYC')
        self.assertTrue(isinstance(result, DateTimeNano))
        result = result._fa
        self.assertTrue(bool(np.all(result == 1514788200000000000)))

    def test_datetimestrings(self):
        dtstrings = FastArray(
            [
                '2018-01-01 12:45:30.123456',
                '2018-01-01 12:45:30.123456000',
                '20180101 12:45:30.123456',
            ]
        )
        result = datetimestring_to_nano(dtstrings, from_tz='NYC', to_tz='NYC')
        self.assertTrue(isinstance(result, DateTimeNano))
        result = result._fa
        self.assertTrue(bool(np.all(result == 1514828730123456000)))

    def test_timesubbug(self):
        time1 = utcnow(1)
        time2 = DateTimeNano(time1.astype('q'))
        x = time1 - time2
        self.assertTrue(x._np[0] == 0.0)

    def test_timespan_unit(self):
        b = 1_000_000_000
        unit_dict = {
            'Y': b * 365 * 24 * 60 * 60,
            'W': b * 7 * 24 * 60 * 60,
            'D': b * 24 * 60 * 60,
            'h': b * 60 * 60,
            'm': b * 60,
            's': b,
            'ms': b / 1000,
            'us': b / 1_000_000,
            'ns': 1,
            b'Y': b * 365 * 24 * 60 * 60,
            b'W': b * 7 * 24 * 60 * 60,
            b'D': b * 24 * 60 * 60,
            b'h': b * 60 * 60,
            b'm': b * 60,
            b's': b,
            b'ms': b / 1000,
            b'us': b / 1_000_000,
            b'ns': 1,
            None: 1,
        }
        for unit, val in unit_dict.items():
            ts = TimeSpan(1, unit=unit)
            self.assertEqual(ts._fa[0], val)

        with self.assertRaises(ValueError):
            ts = TimeSpan(1, unit='junk')

    def test_math_with_invalid(self):
        # TODO: also add sentinels to this test?
        dtn = DateTimeNano(arange(10), from_tz='GMT', to_tz='GMT')
        self.assertEqual(dtn[0], 0)
        dtn2 = dtn + 1
        self.assertTrue(isinstance(dtn2, DateTimeNano))
        self.assertEqual(dtn[0], 0)

    def test_constructor_with_invalid(self):
        dtn = DateTimeNano([0, 10_000_000_000_000], from_tz='NYC', to_tz='NYC')
        self.assertEqual(dtn[0], 0)

    def test_date_tz_combos(self):
        dtn = DateTimeNano(
            [
                '2018-11-01 22:00:00',
                '2018-11-01 23:00:00',
                '2018-11-02 00:00:00',
                '2018-11-02 01:00:00',
                '2018-11-02 02:00:00',
            ],
            from_tz='GMT',
            to_tz='GMT',
        )
        date_arr = dtn.date()
        date_fa = FastArray(
            [
                1541030400000000000,
                1541030400000000000,
                1541116800000000000,
                1541116800000000000,
                1541116800000000000,
            ],
            dtype=np.int64,
        )
        self.assertTrue(bool(np.all(date_arr._fa == date_fa)))
        date_hour = date_arr.hour
        self.assertTrue(bool(np.all(date_hour == 0)))

        dtn = DateTimeNano(
            [
                '2018-11-01 22:00:00',
                '2018-11-01 23:00:00',
                '2018-11-02 00:00:00',
                '2018-11-02 01:00:00',
                '2018-11-02 02:00:00',
            ],
            from_tz='GMT',
            to_tz='NYC',
        )
        date_arr = dtn.date()
        date_fa = FastArray(
            [
                1541044800000000000,
                1541044800000000000,
                1541044800000000000,
                1541044800000000000,
                1541044800000000000,
            ],
            dtype=np.int64,
        )

        self.assertTrue(bool(np.all(date_arr._fa == date_fa)))
        date_hour = date_arr.hour
        self.assertTrue(bool(np.all(date_hour == 0)))

        # 04/24/2019 fixed bug, this combination now adds to the original time to bring into GMT
        # now displays in GMT (will display 4 hours AHEAD of these strings)
        dtn = DateTimeNano(
            [
                '2018-11-01 22:00:00',
                '2018-11-01 23:00:00',
                '2018-11-02 00:00:00',
                '2018-11-02 01:00:00',
                '2018-11-02 02:00:00',
            ],
            from_tz='NYC',
            to_tz='GMT',
        )
        date_arr = dtn.date()
        date_fa = FastArray(
            [
                1541116800000000000,
                1541116800000000000,
                1541116800000000000,
                1541116800000000000,
                1541116800000000000,
            ],
            dtype=np.int64,
        )
        self.assertTrue(bool(np.all(date_arr._fa == date_fa)))
        date_hour = date_arr.hour
        self.assertTrue(bool(np.all(date_hour == 0)))

        dtn = DateTimeNano(
            [
                '2018-11-01 22:00:00',
                '2018-11-01 23:00:00',
                '2018-11-02 00:00:00',
                '2018-11-02 01:00:00',
                '2018-11-02 02:00:00',
            ],
            from_tz='NYC',
            to_tz='NYC',
        )
        date_arr = dtn.date()
        date_fa = FastArray(
            [
                1541044800000000000,
                1541044800000000000,
                1541131200000000000,
                1541131200000000000,
                1541131200000000000,
            ],
            dtype=np.int64,
        )
        self.assertTrue(bool(np.all(date_arr._fa == date_fa)))
        date_hour = date_arr.hour
        self.assertTrue(bool(np.all(date_hour == 0)))

    def test_day_of_week(self):
        dtn2 = DateTimeNano(
            ['1970-01-02 00:00:00', '1970-01-02 00:00:00', '1970-01-03 00:00:01'],
            from_tz='NYC',
            to_tz='NYC',
        )
        dayofweek = dtn2.day_of_week
        self.assertTrue(dayofweek.dtype == np.int64)
        self.assertTrue(bool(np.all(dayofweek == [4, 4, 5])))

        isweekday = dtn2.is_weekday
        self.assertTrue(bool(np.all(isweekday == [True, True, False])))

        isweekend = dtn2.is_weekend
        self.assertTrue(bool(np.all(isweekend == [False, False, True])))

        dtn = DateTimeNano(['1970-01-01 12:00:00'], from_tz='GMT', to_tz='GMT')
        dayofweek = dtn.day_of_week
        self.assertTrue(dayofweek.dtype == np.int64)
        self.assertEqual(dayofweek[0], 3)

    def test_day_of_month(self):
        correct = FastArray([9, 29, 1, 31])
        dtn = DateTimeNano(
            ['2018-01-09', '2000-02-29', '2000-03-01', '2019-12-31'], from_tz='NYC'
        )
        dom = dtn.day_of_month
        self.assertTrue(bool(np.all(dom == correct)))

    def test_timestamp_from_string(self):
        correct = 45240000000000.0

        dts = TimeSpan('12:34')
        self.assertEqual(dts[0], correct)

        dts = TimeSpan(b'12:34')
        self.assertEqual(dts[0], correct)

        dts = TimeSpan(np.array(['12:34', '12:34']))
        for i in dts:
            self.assertEqual(correct, i)

        dts = TimeSpan(np.array([b'12:34', b'12:34']))
        for i in dts:
            self.assertEqual(correct, i)

    def test_nanos_since_year(self):
        dtn = DateTimeNano(['2018-01-01 00:00:00.000123456'], from_tz='NYC')
        since_year = dtn.nanos_since_start_of_year()
        correct = 123456
        self.assertEqual(since_year[0], correct)
        self.assertTrue(since_year.dtype == np.int64)
        self.assertTrue(isinstance(since_year, FastArray))

        dtn = DateTimeNano(
            ['2018-01-01 00:00:00.000123456'], from_tz='GMT', to_tz='GMT'
        )
        since_year = dtn.nanos_since_start_of_year()
        correct = 123456
        self.assertEqual(since_year[0], correct)
        self.assertTrue(since_year.dtype == np.int64)
        self.assertTrue(isinstance(since_year, FastArray))

    def test_nanos_since_midnight(self):
        dtn = DateTimeNano(['2018-02-01 00:00:00.000123456'], from_tz='NYC')
        since_mn = dtn.nanos_since_midnight()
        correct = 123456
        self.assertEqual(since_mn[0], correct)
        self.assertTrue(since_mn.dtype == np.int64)
        self.assertTrue(isinstance(since_mn, FastArray))

        dtn = DateTimeNano(
            ['2018-02-01 00:00:00.000123456'], from_tz='GMT', to_tz='GMT'
        )
        since_mn = dtn.nanos_since_midnight()
        correct = 123456
        self.assertEqual(since_mn[0], correct)
        self.assertTrue(since_mn.dtype == np.int64)
        self.assertTrue(isinstance(since_mn, FastArray))

    def test_time_since_year(self):
        dtn = DateTimeNano(['2018-01-01 00:00:00.000123456'], from_tz='NYC')
        since_year = dtn.time_since_start_of_year()
        correct = TimeSpan([123456])
        self.assertTrue(bool(np.all(since_year == correct)))
        self.assertTrue(isinstance(since_year, TimeSpan))

        dtn = DateTimeNano(
            ['2018-01-01 00:00:00.000123456'], from_tz='GMT', to_tz='GMT'
        )
        since_year = dtn.time_since_start_of_year()
        correct = TimeSpan([123456])
        self.assertTrue(bool(np.all(since_year == correct)))
        self.assertTrue(isinstance(since_year, TimeSpan))

    def test_time_since_midnight(self):
        dtn = DateTimeNano(['2018-02-01 00:00:00.000123456'], from_tz='NYC')
        since_mn = dtn.time_since_midnight()
        correct = TimeSpan([123456])
        self.assertTrue(bool(np.all(since_mn == correct)))
        self.assertTrue(isinstance(since_mn, TimeSpan))

        dtn = DateTimeNano(
            ['2018-02-01 00:00:00.000123456'], from_tz='GMT', to_tz='GMT'
        )
        since_mn = dtn.time_since_midnight()
        correct = TimeSpan([123456])
        self.assertTrue(bool(np.all(since_mn == correct)))
        self.assertTrue(isinstance(since_mn, TimeSpan))

    def test_save_all_tz_combos(self):
        timestring = b'1992-02-01T19:48:30.000000000'
        as_gmt = b'1992-02-02T00:48:30.000000000'

        dtn_nyc_nyc = DateTimeNano(['1992-02-01 19:48:30'], from_tz='NYC', to_tz='NYC')
        dtn_gmt_nyc = DateTimeNano(['1992-02-02 00:48:30'], from_tz='GMT', to_tz='NYC')
        dtn_nyc_gmt = DateTimeNano(['1992-02-01 14:48:30'], from_tz='NYC', to_tz='GMT')
        dtn_gmt_gmt = DateTimeNano(['1992-02-01 19:48:30'], from_tz='GMT', to_tz='GMT')

        ds1 = Dataset(
            {
                'dtn_nyc_nyc': dtn_nyc_nyc,
                'dtn_gmt_nyc': dtn_gmt_nyc,
                'dtn_nyc_gmt': dtn_nyc_gmt,
                'dtn_gmt_gmt': dtn_gmt_gmt,
            }
        )

        for dt in ds1.values():
            self.assertEqual(dt.to_iso()[0], timestring)

        ds1.save(r'riptable/tests/temp/tempsave')

        ds2 = Dataset.load(r'riptable/tests/temp/tempsave')

        for dt, totz in zip(ds2.values(), ['NYC', 'NYC', 'GMT', 'GMT']):
            self.assertEqual(dt._timezone._from_tz, 'GMT')
            self.assertEqual(dt._timezone._to_tz, totz)
            self.assertEqual(dt.to_iso()[0], timestring)

        os.remove(r'riptable/tests/temp/tempsave.sds')

    def test_timespan_nano_extension(self):
        # make sure previous bug with absolute value / mod was fixed

        # NOTE: this relates to display - not usually tested if str/repr changes, this test needs to be modified
        correct_positive = "'00:00:00.000000100'"
        ts = TimeSpan(100)
        d = str(ts)
        self.assertEqual(d, correct_positive)

        correct_negative = "'-00:00:00.000000100'"
        ts = TimeSpan(-100)
        d = str(ts)
        self.assertEqual(d, correct_negative)

    def test_timezone_errors(self):
        with self.assertRaises(ValueError):
            tz = TimeZone(from_tz=None, to_tz=None)

        with self.assertRaises(ValueError):
            _ = TimeZone._init_from_tz('JUNK')

        with self.assertRaises(ValueError):
            _, _, _ = TimeZone._init_to_tz('JUNK')

    def test_mask_no_cutoffs(self):
        tz = TimeZone(from_tz='GMT', to_tz='GMT')
        mask = tz._mask_dst(arange(5))
        self.assertTrue(bool(np.all(~mask)))
        self.assertEqual(len(mask), 5)

    def test_calendar(self):
        with self.assertRaises(NotImplementedError):
            c = Calendar()

    def test_internal_set_tz(self):
        tz = TimeZone(from_tz='NYC', to_tz='NYC')
        tz._set_timezone('GMT')
        self.assertEqual(tz._to_tz, 'GMT')
        self.assertEqual(tz._timezone_str, 'GMT')

    def test_vs_python_dst_fall(self):
        format_str = "%Y-%m-%dT%H:%M:%S.000000000"
        zone = tz.gettz('America/New_York')

        pdt_first = datetime.datetime(
            2018, 11, 4, 5, 1, 0, tzinfo=datetime.timezone.utc
        )
        dtn_first = DateTimeNano(['2018-11-04 01:01'], from_tz='NYC')
        pdt_utc_first = pdt_first.timestamp() * NANOS_PER_SECOND
        dtn_utc_first = dtn_first._fa[0]
        self.assertTrue(pdt_utc_first == dtn_utc_first)

        pdt_last = datetime.datetime(2018, 11, 4, 6, 1, 0, tzinfo=datetime.timezone.utc)
        dtn_last = DateTimeNano(dtn_first._fa + NANOS_PER_HOUR, from_tz='GMT')
        pdt_utc_last = pdt_last.timestamp() * NANOS_PER_SECOND
        dtn_utc_last = dtn_last._fa[0]
        self.assertTrue(pdt_utc_last == dtn_utc_last)

        # assert that the UTC timestamps are different for different hours
        self.assertNotEqual(pdt_utc_first, pdt_utc_last)
        self.assertNotEqual(dtn_utc_first, dtn_utc_last)

        # because a timechange happens, the timestring will appear the same because of timezone adjustment
        pdt_str_first = pdt_first.astimezone(zone).strftime(format_str)
        dtn_str_first = dtn_first.to_iso()[0].decode()
        self.assertEqual(pdt_str_first, dtn_str_first)

        pdt_str_last = pdt_last.astimezone(zone).strftime(format_str)
        dtn_str_last = dtn_last.to_iso()[0].decode()
        self.assertEqual(pdt_str_last, dtn_str_last)

        self.assertEqual(pdt_str_first, dtn_str_last)

    def test_vs_python_dst_spring(self):
        format_str = "%Y-%m-%dT%H:%M:%S.000000000"
        zone = tz.gettz('America/New_York')

        pdt_first = datetime.datetime(
            2019, 3, 10, 6, 1, 0, tzinfo=datetime.timezone.utc
        )
        pdt_last = datetime.datetime(2019, 3, 10, 7, 1, 0, tzinfo=datetime.timezone.utc)

        dtn_first = DateTimeNano(['2019-03-10 01:01'], from_tz='NYC')
        dtn_last = DateTimeNano(dtn_first._fa + NANOS_PER_HOUR, from_tz='GMT')

        correct_first = '2019-03-10T01:01:00.000000000'
        pdt_str_first = pdt_first.astimezone(zone).strftime(format_str)
        dtn_str_first = dtn_first.to_iso()[0].decode()
        self.assertEqual(pdt_str_first, dtn_str_first)
        self.assertEqual(dtn_str_first, correct_first)

        correct_last = '2019-03-10T03:01:00.000000000'
        pdt_str_last = pdt_last.astimezone(zone).strftime(format_str)
        dtn_str_last = dtn_last.to_iso()[0].decode()
        self.assertEqual(pdt_str_last, dtn_str_last)
        self.assertEqual(dtn_str_last, correct_last)

    def test_vs_python_dst_fall_dublin(self):
        format_str = "%Y-%m-%dT%H:%M:%S.000000000"
        zone = tz.gettz('Europe/Dublin')

        pdt_first = datetime.datetime(
            2018, 10, 28, 0, 1, 0, tzinfo=datetime.timezone.utc
        )
        pdt_last = datetime.datetime(
            2018, 10, 28, 1, 1, 0, tzinfo=datetime.timezone.utc
        )
        t1 = pdt_first.timestamp()
        t2 = pdt_last.timestamp()
        t1 = int(NANOS_PER_SECOND * t1)
        t2 = int(NANOS_PER_SECOND * t2)

        dtn_first = DateTimeNano([t1], from_tz='GMT', to_tz='DUBLIN')
        dtn_last = DateTimeNano([t2], from_tz='GMT', to_tz='DUBLIN')

        pdt_str_first = pdt_first.astimezone(zone).strftime(format_str)
        dtn_str_first = dtn_first.to_iso()[0].decode()
        self.assertEqual(pdt_str_first, dtn_str_first)

        pdt_str_last = pdt_last.astimezone(zone).strftime(format_str)
        dtn_str_last = dtn_last.to_iso()[0].decode()
        self.assertEqual(pdt_str_last, dtn_str_last)

        self.assertEqual(pdt_str_first, dtn_str_last)

    def test_dst_fall_hour(self):
        '''
        When daylight savings time ends, clocks go back one hour. However, UTC is
        does not change. Initializing times in different hours in UTC on the changing hour
        will yield the same result in a specific timezone.
        '''

        zone = tz.gettz('Europe/Dublin')

        pdt1 = datetime.datetime(2018, 10, 28, 0, 1, 0, tzinfo=datetime.timezone.utc)
        pdt2 = datetime.datetime(2018, 10, 28, 1, 1, 0, tzinfo=datetime.timezone.utc)
        pdt1_dub = pdt1.astimezone(zone)
        pdt2_dub = pdt2.astimezone(zone)

        dtn1 = DateTimeNano(['2018-10-28 00:01'], from_tz='GMT', to_tz='DUBLIN')
        dtn2 = DateTimeNano(['2018-10-28 01:01'], from_tz='GMT', to_tz='DUBLIN')

        dtn1_hour = int(dtn1.hour[0])
        dtn2_hour = int(dtn2.hour[0])

        self.assertEqual(pdt1_dub.hour, dtn1_hour)
        self.assertEqual(pdt2_dub.hour, dtn2_hour)
        self.assertEqual(pdt1_dub.hour, dtn2_hour)

        zone = tz.gettz('America/New_York')

        pdt1 = datetime.datetime(2018, 11, 4, 5, 1, 0, tzinfo=datetime.timezone.utc)
        pdt2 = datetime.datetime(2018, 11, 4, 6, 1, 0, tzinfo=datetime.timezone.utc)
        pdt1_nyc = pdt1.astimezone(zone)
        pdt2_nyc = pdt2.astimezone(zone)

        dtn1 = DateTimeNano(['2018-11-04 05:01'], from_tz='GMT', to_tz='NYC')
        dtn2 = DateTimeNano(['2018-11-04 06:01'], from_tz='GMT', to_tz='NYC')

        dtn1_hour = int(dtn1.hour[0])
        dtn2_hour = int(dtn2.hour[0])

        self.assertEqual(pdt1_nyc.hour, dtn1_hour)
        self.assertEqual(pdt2_nyc.hour, dtn2_hour)
        self.assertEqual(pdt1_nyc.hour, dtn2_hour)

    def test_dst_spring_hour(self):
        '''
        When daylight savings time begins, an hour is skipped.
        NYC changes over at 2am local time
        DUBLIN changes over at 1am local time
        '''
        zone = tz.gettz('Europe/Dublin')

        pdt0 = datetime.datetime(2018, 3, 25, 0, 59, 0, tzinfo=datetime.timezone.utc)
        pdt1 = datetime.datetime(2018, 3, 25, 1, 59, 0, tzinfo=datetime.timezone.utc)
        pdt0_dub = pdt0.astimezone(zone)
        pdt1_dub = pdt1.astimezone(zone)

        dtn = DateTimeNano(
            ['2018-03-25 00:59', '2018-03-25 01:59'], from_tz='GMT', to_tz='DUBLIN'
        )
        dtn_hour = dtn.hour.astype(np.int32)

        self.assertEqual(pdt0_dub.hour, dtn_hour[0])
        self.assertEqual(pdt1_dub.hour, dtn_hour[1])
        self.assertEqual(dtn_hour[1] - dtn_hour[0], 2)

        zone = tz.gettz('America/New_York')

        pdt0 = datetime.datetime(2019, 3, 10, 6, 1, 0, tzinfo=datetime.timezone.utc)
        pdt1 = datetime.datetime(2019, 3, 10, 7, 1, 0, tzinfo=datetime.timezone.utc)
        pdt0_nyc = pdt0.astimezone(zone)
        pdt1_nyc = pdt1.astimezone(zone)

        dtn = DateTimeNano(
            ['2019-03-10 06:01', '2019-03-10 07:01'], from_tz='GMT', to_tz='NYC'
        )
        dtn_hour = dtn.hour.astype(np.int32)

        self.assertEqual(pdt0_nyc.hour, dtn_hour[0])
        self.assertEqual(pdt1_nyc.hour, dtn_hour[1])
        self.assertEqual(dtn_hour[1] - dtn_hour[0], 2)

    def test_dst_spring_constructor(self):
        '''
        Ensures that DateTimeNano is correctly converting timezone specific stamps to UTC.
        '''
        zone = tz.gettz('Europe/Dublin')

        pdt0 = datetime.datetime(2018, 3, 25, 0, 59, 0, tzinfo=zone)
        pdt1 = datetime.datetime(2018, 3, 25, 2, 59, 0, tzinfo=zone)
        pdt_dublin_diff = pdt1.hour - pdt0.hour
        pdt0_utc = pdt0.astimezone(datetime.timezone.utc)
        pdt1_utc = pdt1.astimezone(datetime.timezone.utc)

        dtn = DateTimeNano(
            ['2018-03-25 00:59', '2018-03-25 02:59'], from_tz='DUBLIN', to_tz='DUBLIN'
        )
        dtn_dublin_hour = dtn.hour.astype(np.int32)
        dtn_dublin_diff = dtn_dublin_hour[1] - dtn_dublin_hour[0]
        dtn.set_timezone('GMT')  # view as UTC
        dtn_hour = dtn.hour.astype(np.int32)

        self.assertEqual(pdt_dublin_diff, dtn_dublin_diff)
        self.assertEqual(pdt0_utc.hour, dtn_hour[0])
        self.assertEqual(pdt1_utc.hour, dtn_hour[1])
        self.assertEqual(dtn_hour[1] - dtn_hour[0], 1)

        zone = tz.gettz('America/New_York')

        pdt0 = datetime.datetime(2019, 3, 10, 1, 59, 0, tzinfo=zone)
        pdt1 = datetime.datetime(2019, 3, 10, 3, 59, 0, tzinfo=zone)
        pdt_nyc_diff = pdt1.hour - pdt0.hour
        pdt0_utc = pdt0.astimezone(datetime.timezone.utc)
        pdt1_utc = pdt1.astimezone(datetime.timezone.utc)

        dtn = DateTimeNano(
            ['2019-03-10 01:59', '2019-03-10 03:59'], from_tz='NYC', to_tz='NYC'
        )
        dtn_nyc_hour = dtn.hour.astype(np.int32)
        dtn_nyc_diff = dtn_nyc_hour[1] - dtn_nyc_hour[0]
        dtn.set_timezone('GMT')  # view as UTC
        dtn_hour = dtn.hour.astype(np.int32)

        self.assertEqual(pdt_nyc_diff, dtn_nyc_diff)
        self.assertEqual(pdt0_utc.hour, dtn_hour[0])
        self.assertEqual(pdt1_utc.hour, dtn_hour[1])
        self.assertEqual(dtn_hour[1] - dtn_hour[0], 1)

    def test_dst_fall_constructor(self):
        zone = tz.gettz('America/New_York')
        pdt0 = datetime.datetime(2018, 11, 4, 1, 59, 0, tzinfo=zone)
        pdt1 = datetime.datetime(2018, 11, 4, 2, 59, 0, tzinfo=zone)
        pdt_nyc_diff = pdt1.hour - pdt0.hour
        pdt0_utc = pdt0.astimezone(datetime.timezone.utc)
        pdt1_utc = pdt1.astimezone(datetime.timezone.utc)

        dtn = DateTimeNano(
            ['2018-11-04 01:59', '2018-11-04 02:59'], from_tz='NYC', to_tz='NYC'
        )
        dtn_nyc_hour = dtn.hour.astype(np.int32)
        dtn_nyc_diff = dtn_nyc_hour[1] - dtn_nyc_hour[0]
        dtn.set_timezone('GMT')  # view as UTC
        dtn_hour = dtn.hour.astype(np.int32)

        self.assertEqual(pdt_nyc_diff, dtn_nyc_diff)
        self.assertEqual(pdt0_utc.hour, dtn_hour[0])
        self.assertEqual(pdt1_utc.hour, dtn_hour[1])
        self.assertEqual(dtn_hour[1] - dtn_hour[0], 2)

    def test_constructor_errors(self):
        with self.assertRaises(TypeError):
            dtn = DateTimeNano({1, 2, 3}, from_tz='NYC')
        with self.assertRaises(TypeError):
            dtn = DateTimeNano(zeros(5, dtype=bool), from_tz='NYC')

    def test_classname(self):
        dtn = DateTimeNano(['2000-01-01 00:00:00'], from_tz='NYC')
        self.assertEqual(dtn.get_classname(), dtn.__class__.__name__)

        ts = TimeSpan(100, unit='s')
        self.assertEqual(ts.get_classname(), ts.__class__.__name__)

    def test_matlab_datenum(self):
        d = FA([730545.00])
        dtn = DateTimeNano(d, from_matlab=True, from_tz='NYC')
        self.assertEqual(dtn.to_iso()[0], b'2000-02-29T00:00:00.000000000')

        # test precision too
        d = FA([730545.00], dtype=np.float32)
        dtn = DateTimeNano(d, from_matlab=True, from_tz='NYC')
        self.assertEqual(dtn.to_iso()[0], b'2000-02-29T00:00:00.000000000')

    def test_hstack_errors(self):
        c = Categorical(['a', 'a', 'b', 'c'])
        dtn = DateTimeNano(['2000-01-01 00:00:00'], from_tz='NYC')
        with self.assertRaises(TypeError):
            _ = DateTimeNano.hstack([dtn, c])

        dtn2 = DateTimeNano(['2000-01-01 00:00:00'], from_tz='NYC', to_tz='GMT')
        with self.assertRaises(NotImplementedError):
            _ = DateTimeNano.hstack([dtn, dtn2])

    def test_inplace_subtract(self):
        dtn = DateTimeNano(['2000-01-01'], from_tz='NYC', to_tz='GMT')
        start = dtn.days_since_epoch[0]
        dtn -= NANOS_PER_DAY
        end = dtn.days_since_epoch[0]
        self.assertEqual(end - start, -1)

    def test_diff(self):
        dtn = DateTimeNano(
            [
                '2018-11-01 22:00:00',
                '2018-11-01 23:00:00',
                '2018-11-02 00:00:00',
                '2018-11-02 01:00:00',
                '2018-11-02 02:00:00',
            ],
            from_tz='NYC',
            to_tz='NYC',
        )

        dtndiff = dtn.diff()
        self.assertTrue(isinstance(dtndiff, TimeSpan))
        self.assertTrue(isnan(dtndiff[0]))

        hour_diff = dtn.diff()[1:].hours
        self.assertTrue(bool(np.all(hour_diff == 1)))

    def test_math_errors(self):
        dtn = DateTimeNano(['2000-01-01 00:00:00'], from_tz='NYC')
        with self.assertRaises(TypeError):
            a = dtn.__abs__()

        # modula now allowed
        # with self.assertRaises(TypeError):
        #    a = dtn % 7

    def test_timespan_true_divide(self):
        ts = TimeSpan(3, unit='m')
        ts2 = ts / 3
        self.assertTrue(isinstance(ts2, TimeSpan))
        self.assertEqual(ts2.minutes[0], 1)

        ts = TimeSpan(3, unit='m')
        ts2 = ts / TimeSpan(3, unit='m')
        self.assertFalse(isinstance(ts2, TimeSpan))
        self.assertEqual(ts2[0], 1)

    def test_timespan_floor_divide(self):
        ts = TimeSpan(5.5, unit='m')
        ts2 = ts // TimeSpan(1, unit='m')
        self.assertFalse(isinstance(ts2, TimeSpan))
        self.assertEqual(ts2[0], 5)

        with self.assertRaises(TypeError):
            ts2 = ts // 3

    def test_timespan_unit_display(self):
        d = {'ns': 1, 'us': 1000, 'ms': 1_000_000, 's': 2_000_000_000}
        for k, v in d.items():
            result = TimeSpan.display_item_unit(v)
            self.assertTrue(result.endswith(k))

    def test_timespan_hhhhmmss(self):
        timespan = TimeSpan([
            '09:30:17.557593707',
            '15:31:32.216792000',
            '11:28:23.519020994',
            '19:46:10.838007105',
            '09:30:29.999999999',
            '10:40:00.000000000',
            '00:00:00.999999999',
            '23:59:59.999999999',
        ])
        expected = FastArray([93017, 153132, 112823, 194610, 93029, 104000, 0, 235959])
        actual = timespan.hhmmss
        self.assertTrue(np.all(expected == actual))

    def test_round_nano_time(self):
        correct_str = "'20181231 23:59:59.999999999'"
        correct_iso = b'2018-12-31T23:59:59.999999999'

        # ensures that previous python rounding error was fixed
        dtn = DateTimeNano(['2018-12-31 23:59:59.999999999'], from_tz='NYC')
        repr_str = dtn._build_string()
        iso = dtn.to_iso()[0]

        self.assertEqual(repr_str, correct_str)
        self.assertEqual(iso, correct_iso)

    def test_day_of_year(self):
        dtn = DateTimeNano(
            ['2019-01-01', '2019-02-01', '2019-12-31 23:59', '2000-12-31 23:59'],
            from_tz='NYC',
        )
        dayofyear = dtn.day_of_year
        correct = FastArray([1, 32, 365, 366])
        self.assertTrue(bool(np.all(dayofyear == correct)))

    def test_month_edge(self):
        # ensure that searchsorted goes to the right for matching value
        dtn = DateTimeNano(['2000-02-01', '2019-02-01'], from_tz='NYC')
        m = dtn.month()
        self.assertTrue(bool(np.all(m == 2)))

    def test_datetime_string_invalid(self):
        # 1 nanosecond from epoch
        dtn = DateTimeNano(
            ['1970-01-01 00:00:00.000000001'], from_tz='GMT', to_tz='GMT'
        )
        self.assertEqual(dtn._fa[0], 1)

        # before epoch time (invalid)
        dtn = DateTimeNano(['1969-12-31'], from_tz='NYC')
        self.assertEqual(dtn._fa[0], 0)

        dtn = DateTimeNano(['2000-13-01'], from_tz='NYC')
        self.assertEqual(dtn._fa[0], 0)

        dtn = DateTimeNano(['2000-12-40'], from_tz='NYC')
        self.assertEqual(dtn._fa[0], 0)

    def test_yyyymmdd(self):
        correct = FastArray([20180109, 20000229, 20000301, 20191231])
        dtn = DateTimeNano(
            ['2018-01-09 23:59:59.999999999', '2000-02-29', '2000-03-01', '2019-12-31'],
            from_tz='NYC',
        )
        ymd = dtn.yyyymmdd
        self.assertTrue(bool(np.all(correct == ymd)))

    def test_seconds_since_epoch(self):
        seconds_per_day = 86400
        dtn = DateTimeNano(['1970-01-02'], from_tz='NYC')
        result = dtn.seconds_since_epoch

        self.assertTrue(result.dtype == np.int64)
        self.assertEqual(seconds_per_day, result[0])

    def test_millisecond(self):
        dtn = DateTimeNano(['1992-02-01 12:00:01.123000000'], from_tz='NYC')

        f = dtn.millisecond
        self.assertTrue(f.dtype == np.float64)
        self.assertEqual(f[0], 123.0)

        s = dtn.millisecond_span
        self.assertTrue(isinstance(s, TimeSpan))
        self.assertEqual(s._fa[0], 123000000.0)

    def test_microsecond(self):
        dtn = DateTimeNano(['1992-02-01 12:00:01.000123000'], from_tz='NYC')

        f = dtn.microsecond
        self.assertTrue(f.dtype == np.float64)
        self.assertEqual(f[0], 123.0)

        s = dtn.microsecond_span
        self.assertTrue(isinstance(s, TimeSpan))
        self.assertEqual(s._fa[0], 123000.0)

    def test_nanosecond(self):
        dtn = DateTimeNano(['1992-02-01 12:00:01.000000123'], from_tz='NYC')

        f = dtn.nanosecond
        self.assertTrue(f.dtype == np.float64)
        self.assertEqual(f[0], 123.0)

        s = dtn.nanosecond_span
        self.assertTrue(isinstance(s, TimeSpan))
        self.assertEqual(s._fa[0], 123.0)

    def test_millis_since_midnight(self):
        dtn = DateTimeNano(['1992-02-01 00:00:01.002003004'], from_tz='NYC')
        result = dtn.millis_since_midnight()

        self.assertTrue(result.dtype == np.float64)
        self.assertEqual(result[0], 1002.003004)

        # check scalar
        self.assertEqual(result[0], dtn[0].millis_since_midnight())

    def test_is_dst_nyc(self):
        dtn = DateTimeNano(
            ['2018-11-03 12:34', '2018-11-04 12:34'], from_tz='NYC', to_tz='NYC'
        )
        result = dtn.is_dst
        correct = FastArray([True, False])
        self.assertTrue(bool(np.all(result == correct)))

    def test_is_dst_dublin(self):
        dtn = DateTimeNano(
            ['2019-03-30 12:34', '2019-03-31 12:34'], from_tz='DUBLIN', to_tz='DUBLIN'
        )
        result = dtn.is_dst
        correct = FastArray([False, True])
        self.assertTrue(bool(np.all(result == correct)))

    def test_is_dst_gmt(self):
        dtn = DateTimeNano(['2019-01-01'], from_tz='GMT', to_tz='GMT')
        start = dtn._fa[0]
        daystamps = arange(start, start + NANOS_PER_YEAR, NANOS_PER_DAY)
        dtn = DateTimeNano(daystamps, from_tz='GMT', to_tz='GMT')
        result = dtn.is_dst
        self.assertFalse(bool(np.any(result)))

    def test_tz_offset_nyc(self):
        dtn = DateTimeNano(
            ['2018-11-03 12:34', '2018-11-04 12:34'], from_tz='NYC', to_tz='NYC'
        )
        result = dtn.tz_offset
        correct = FastArray([-4, -5])
        self.assertTrue(bool(np.all(result == correct)))

    def test_is_offset_dublin(self):
        dtn = DateTimeNano(
            ['2019-03-30 12:34', '2019-03-31 12:34'], from_tz='DUBLIN', to_tz='DUBLIN'
        )
        result = dtn.tz_offset
        correct = FastArray([0, 1])
        self.assertTrue(bool(np.all(result == correct)))

    def test_is_offset_gmt(self):
        dtn = DateTimeNano(['2019-01-01'], from_tz='GMT', to_tz='GMT')
        start = dtn._fa[0]
        daystamps = arange(start, start + NANOS_PER_YEAR, NANOS_PER_DAY)
        dtn = DateTimeNano(daystamps, from_tz='GMT', to_tz='GMT')
        result = dtn.tz_offset
        self.assertFalse(bool(np.any(result)))

    def test_strptime_date(self):
        fmt = '%m/%d/%Y'
        t_strings = ['02/01/1992', '2/1/1992', '2/29/2000']
        dtn = FA(t_strings)
        dtn = strptime_to_nano(dtn, fmt, from_tz='NYC')
        pdt = [datetime.datetime.strptime(t, fmt) for t in t_strings]

        rt_year = dtn.year()
        py_year = [t.year for t in pdt]
        self.assertTrue(bool(np.all(rt_year == py_year)))

        rt_month = dtn.month()
        py_month = [t.month for t in pdt]
        self.assertTrue(bool(np.all(rt_month == py_month)))

        rt_day = dtn.day_of_month
        py_day = [t.day for t in pdt]
        self.assertTrue(bool(np.all(rt_day == py_day)))

        # also test with constructor
        dtn = DateTimeNano(t_strings, from_tz='NYC', format=fmt)
        rt_year = dtn.year()
        rt_month = dtn.month()
        rt_day = dtn.day_of_month

        self.assertTrue(bool(np.all(rt_year == py_year)))
        self.assertTrue(bool(np.all(rt_month == py_month)))
        self.assertTrue(bool(np.all(rt_day == py_day)))

    def test_strptime_time(self):
        fmt = '%m/%d/%Y %H:%M:%S'
        t_strings = ['02/01/1992 12:15:11', '2/1/1992 5:01:09', '2/29/2000 12:39:59']
        dtn = FA(t_strings)
        dtn = strptime_to_nano(dtn, fmt, from_tz='NYC')
        pdt = [datetime.datetime.strptime(t, fmt) for t in t_strings]

        rt_year = dtn.year()
        py_year = [t.year for t in pdt]
        self.assertTrue(bool(np.all(rt_year == py_year)))

        rt_month = dtn.month()
        py_month = [t.month for t in pdt]
        self.assertTrue(bool(np.all(rt_month == py_month)))

        rt_day = dtn.day_of_month
        py_day = [t.day for t in pdt]
        self.assertTrue(bool(np.all(rt_day == py_day)))

        rt_hour = np.int64(dtn.hour)
        py_hour = [t.hour for t in pdt]
        self.assertTrue(bool(np.all(rt_hour == py_hour)))

        rt_min = np.int64(dtn.minute)
        py_min = [t.minute for t in pdt]
        self.assertTrue(bool(np.all(rt_min == py_min)))

        rt_sec = np.int64(dtn.second)
        py_sec = [t.second for t in pdt]
        self.assertTrue(bool(np.all(rt_sec == py_sec)))

        # also test with constructor
        dtn = DateTimeNano(t_strings, from_tz='NYC', format=fmt)
        rt_year = dtn.year()
        rt_month = dtn.month()
        rt_day = dtn.day_of_month
        rt_hour = np.int64(dtn.hour)
        rt_min = np.int64(dtn.minute)
        rt_sec = np.int64(dtn.second)

        self.assertTrue(bool(np.all(rt_year == py_year)))
        self.assertTrue(bool(np.all(rt_month == py_month)))
        self.assertTrue(bool(np.all(rt_day == py_day)))
        self.assertTrue(bool(np.all(rt_hour == py_hour)))
        self.assertTrue(bool(np.all(rt_min == py_min)))
        self.assertTrue(bool(np.all(rt_sec == py_sec)))

    def test_strptime_ampm(self):
        fmt = '%m/%d/%Y %I:%M:%S %p'
        t_strings = [
            '02/01/1992 7:15:11 AM',
            '2/1/1992 5:01:09 PM',
            '2/29/2000 6:39:59 AM',
        ]
        dtn = FA(t_strings)
        dtn = strptime_to_nano(dtn, fmt, from_tz='NYC')
        pdt = [datetime.datetime.strptime(t, fmt) for t in t_strings]

        rt_year = dtn.year()
        py_year = [t.year for t in pdt]
        self.assertTrue(bool(np.all(rt_year == py_year)))

        rt_month = dtn.month()
        py_month = [t.month for t in pdt]
        self.assertTrue(bool(np.all(rt_month == py_month)))

        rt_day = dtn.day_of_month
        py_day = [t.day for t in pdt]
        self.assertTrue(bool(np.all(rt_day == py_day)))

        rt_hour = np.int64(dtn.hour)
        py_hour = [t.hour for t in pdt]
        self.assertTrue(bool(np.all(rt_hour == py_hour)))

        rt_min = np.int64(dtn.minute)
        py_min = [t.minute for t in pdt]
        self.assertTrue(bool(np.all(rt_min == py_min)))

        rt_sec = np.int64(dtn.second)
        py_sec = [t.second for t in pdt]
        self.assertTrue(bool(np.all(rt_sec == py_sec)))

        # also test with constructor
        dtn = DateTimeNano(t_strings, from_tz='NYC', format=fmt)
        rt_year = dtn.year()
        rt_month = dtn.month()
        rt_day = dtn.day_of_month
        rt_hour = np.int64(dtn.hour)
        rt_min = np.int64(dtn.minute)
        rt_sec = np.int64(dtn.second)

        self.assertTrue(bool(np.all(rt_year == py_year)))
        self.assertTrue(bool(np.all(rt_month == py_month)))
        self.assertTrue(bool(np.all(rt_day == py_day)))
        self.assertTrue(bool(np.all(rt_hour == py_hour)))
        self.assertTrue(bool(np.all(rt_min == py_min)))
        self.assertTrue(bool(np.all(rt_sec == py_sec)))

    def test_strptime_monthname(self):
        names = [
            'January',
            'February',
            'March',
            'April',
            'May',
            'June',
            'July',
            'August',
            'September',
            'October',
            'November',
            'December',
        ]
        t_strings = [f'10 {n} 2018' for n in names]
        fmt = '%d %B %Y'
        dtn = DateTimeNano(t_strings, from_tz='NYC', format=fmt)
        self.assertTrue(bool(np.all(dtn.month() == arange(1, 13))))

    def test_strptime_monthname_short(self):
        names = [
            'Jan',
            'Feb',
            'Mar',
            'Apr',
            'May',
            'Jun',
            'Jul',
            'Aug',
            'Sep',
            'Oct',
            'Nov',
            'Dec',
        ]
        t_strings = [f'10 {n} 2018' for n in names]
        fmt = '%d %b %Y'
        dtn = DateTimeNano(t_strings, from_tz='NYC', format=fmt)
        self.assertTrue(bool(np.all(dtn.month() == arange(1, 13))))

    def test_strptime_frac(self):
        fmt = '%m/%d/%Y %H:%M:%S'
        t_strings = [
            '02/01/1992 12:15:11.123567',
            '2/1/1992 05:01:09.888777',
            '2/29/2000 12:39:59.999999',
        ]
        correct = [123.567, 888.777, 999.999]
        dtn = FA(t_strings)
        dtn = strptime_to_nano(dtn, fmt, from_tz='NYC')
        result = dtn.millisecond
        self.assertTrue(bool(np.all(correct == result)))

    def test_strptime_invalid(self):
        # no date in format
        fmt = '%H:%M'
        t_strings = ['12:34', '15:59']
        dtn = DateTimeNano(t_strings, from_tz='NYC', format=fmt)
        self.assertTrue(bool(np.all(dtn.isnan())))

        # invalid date
        fmt = '%Y/%m/%d'
        t_strings = ['2010/00/30', '2010/13/31']
        dtn = DateTimeNano(t_strings, from_tz='NYC', format=fmt)
        self.assertTrue(bool(np.all(dtn.isnan())))

    def test_strptime_scrambled(self):
        fmt = '%H %Y/%m/%d'
        t_strings = ['12 1992/02/01', '13 1992/02/01']
        dtn = DateTimeNano(t_strings, from_tz='NYC', format=fmt)
        correct = [12, 13]
        result = dtn.hour
        self.assertTrue(bool(np.all(correct == result)))

    def test_invalid_constructor_str(self):
        dtn = DateTimeNano(
            ['2018-02-01 12:34', 'inv'], from_tz='NYC', format='%Y-%m-%d %H:%M'
        )
        self.assertEqual(dtn[1], DateTimeNano.NAN_TIME)
        dtn = DateTimeNano(
            ['2018-02-01 12:34', 'inv'], from_tz='GMT', format='%Y-%m-%d %H:%M'
        )
        self.assertEqual(dtn[1], DateTimeNano.NAN_TIME)

    def test_invalid_constructor_matlab(self):
        d = FA([730545.00, np.nan])
        dtn = DateTimeNano(d, from_matlab=True, from_tz='NYC')
        self.assertEqual(dtn._fa[1], 0)

    def test_invalid_to_iso(self):
        d = FA([730545.00, np.nan])
        dtn = DateTimeNano(d, from_matlab=True, from_tz='NYC')
        result = dtn.to_iso()
        self.assertEqual(result[1], b'NaT')

    def test_invalid_date(self):
        dtn = DateTimeNano(
            ['2018-02-01 12:34', 'inv'], from_tz='NYC', format='%Y-%m-%d %H:%M'
        ).date()
        self.assertEqual(dtn[1], DateTimeNano.NAN_TIME)
        dtn = DateTimeNano(
            ['2018-02-01 12:34', 'inv'], from_tz='GMT', format='%Y-%m-%d %H:%M'
        ).date()
        self.assertEqual(dtn[1], DateTimeNano.NAN_TIME)

    def test_invalid_day(self):
        dtn = DateTimeNano.random_invalid(50)
        mask = dtn._fa == 0
        result = dtn.day
        self.assertTrue(
            bool(np.all(mask == result.isnan())),
            f'Did not match at time: \n{dtn[mask!=result.isnan()]}',
        )

    def test_invalid_hour(self):
        dtn = DateTimeNano.random_invalid(50)
        mask = dtn._fa == 0
        result = dtn.hour
        self.assertTrue(
            bool(np.all(mask == result.isnan())),
            f'Did not match at time: \n{dtn[mask!=result.isnan()]}',
        )

        result = dtn.hour_span
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(
            bool(np.all(mask == result.isnan())),
            f'Did not match at time: \n{dtn[mask!=result.isnan()]}',
        )

    def test_invalid_minute(self):
        dtn = DateTimeNano.random_invalid(50)
        mask = dtn._fa == 0
        result = dtn.minute
        self.assertTrue(
            bool(np.all(mask == result.isnan())),
            f'Did not match at time: \n{dtn[mask!=result.isnan()]}',
        )

        result = dtn.minute_span
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(
            bool(np.all(mask == result.isnan())),
            f'Did not match at time: \n{dtn[mask!=result.isnan()]}',
        )

    def test_invalid_second(self):
        dtn = DateTimeNano.random_invalid(50)
        mask = dtn._fa == 0
        result = dtn.second
        self.assertTrue(
            bool(np.all(mask == result.isnan())),
            f'Did not match at time: \n{dtn[mask!=result.isnan()]}',
        )

        result = dtn.second_span
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(
            bool(np.all(mask == result.isnan())),
            f'Did not match at time: \n{dtn[mask!=result.isnan()]}',
        )

    def test_invalid_millisecond(self):
        dtn = DateTimeNano.random_invalid(50)
        mask = dtn._fa == 0
        result = dtn.millisecond
        self.assertTrue(
            bool(np.all(mask == result.isnan())),
            f'Did not match at time: \n{dtn[mask!=result.isnan()]}',
        )

        result = dtn.millisecond_span
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(
            bool(np.all(mask == result.isnan())),
            f'Did not match at time: \n{dtn[mask!=result.isnan()]}',
        )

    def test_invalid_microsecond(self):
        dtn = DateTimeNano.random_invalid(50)
        mask = dtn._fa == 0
        result = dtn.microsecond
        self.assertTrue(
            bool(np.all(mask == result.isnan())),
            f'Did not match at time: \n{dtn[mask!=result.isnan()]}',
        )

        result = dtn.microsecond_span
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(
            bool(np.all(mask == result.isnan())),
            f'Did not match at time: \n{dtn[mask!=result.isnan()]}',
        )

    def test_invalid_nanosecond(self):
        dtn = DateTimeNano.random_invalid(50)
        mask = dtn._fa == 0
        result = dtn.nanosecond
        self.assertTrue(
            bool(np.all(mask == result.isnan())),
            f'Did not match at time: \n{dtn[mask!=result.isnan()]}',
        )

        result = dtn.nanosecond_span
        self.assertTrue(isinstance(result, TimeSpan))
        self.assertTrue(
            bool(np.all(mask == result.isnan())),
            f'Did not match at time: \n{dtn[mask!=result.isnan()]}',
        )

    def test_groupby_restore(self):
        dtn = DateTimeNano(
            [
                '2000-01-01',
                '2000-01-02',
                '2000-01-03',
                '2000-01-01',
                '2000-01-02',
                '2000-01-03',
            ],
            from_tz='NYC',
        )
        ds = Dataset({'dtn': dtn, 'data': arange(6)})
        result = ds.gb('dtn').sum()
        self.assertTrue(isinstance(result.dtn, DateTimeNano))

    def test_subtract(self):
        d = DateTimeNano.random(5)
        val = 1
        result = d - val
        self.assertTrue(isinstance(result, DateTimeNano))

        # val = [1]
        # result = d - val
        # self.assertTrue(isinstance(result, DateTimeNano))

        val = FastArray(1)
        result = d - val
        self.assertTrue(isinstance(result, DateTimeNano))

        # val = Date(1)
        # result = d - val
        # self.assertTrue(isinstance(result, TimeSpan))

        # val = DateSpan(1)
        # result = d - val
        # self.assertTrue(isinstance(result, DateTimeNano))

        val = DateTimeNano(['1970-01-10'], from_tz='GMT')
        result = d - val
        self.assertTrue(isinstance(result, TimeSpan))

        val = TimeSpan(1, unit='h')
        result = d - val
        self.assertTrue(isinstance(result, DateTimeNano))

        x = utcnow(5)
        result = x - x[0]
        self.assertTrue(isinstance(result, TimeSpan))

    def test_add(self):
        d = DateTimeNano.random(10)
        val = 1
        result = d + val
        self.assertTrue(isinstance(result, DateTimeNano))

        val = 1.0
        result = d + val
        self.assertTrue(isinstance(result, DateTimeNano))

        # val = [1]
        # result = d + val
        # self.assertTrue(isinstance(result, DateTimeNano))

        val = FastArray(1)
        result = d + val
        self.assertTrue(isinstance(result, DateTimeNano))

        val = TimeSpan(1)
        result = d + val
        self.assertTrue(isinstance(result, DateTimeNano))

        val = TimeSpan(1, unit='h')
        result = d + val
        self.assertTrue(isinstance(result, DateTimeNano))

    def test_add_invalid(self):
        dtn = DateTimeNano.random_invalid(5)

    # still deciding what to return from min/max - DateTimeNano scalar?
    # def test_min_max(self):
    #    dtn = DateTimeNano(['2018-11-01 22:00:00', '2018-11-01 23:00:00', '2018-11-02 00:00:00', '2018-11-02 01:00:00', '2018-11-02 02:00:00'], from_tz='NYC', to_tz='NYC')

    #    mintime = dtn.min()
    #    self.assertTrue(isinstance(mintime, DateTimeNano))
    #    self.assertEqual(len(mintime),1)
    #    self.assertTrue(bool(np.all(mintime==dtn[0])))

    #    maxtime = dtn.max()
    #    self.assertTrue(isinstance(maxtime, DateTimeNano))
    #    self.assertEqual(len(maxtime),1)
    #    self.assertTrue(bool(np.all(maxtime==dtn[4])))

    def test_start_date(self):

        dtn = DateTimeNano(
            NANOS_PER_HOUR * arange(5),
            from_tz='NYC',
            to_tz='NYC',
            start_date='20190201',
        )
        self.assertTrue(bool(np.all(dtn.hour == arange(5))))
        self.assertTrue(bool(np.all(dtn.yyyymmdd == 20190201)))

        dtn = DateTimeNano(
            TimeSpan(NANOS_PER_HOUR * arange(5)),
            from_tz='NYC',
            to_tz='NYC',
            start_date='20190201',
        )
        self.assertTrue(bool(np.all(dtn.hour == arange(5))))
        self.assertTrue(bool(np.all(dtn.yyyymmdd == 20190201)))

        dtn = DateTimeNano(
            ['00:00', '01:00', '02:00', '03:00', '04:00'],
            from_tz='NYC',
            to_tz='NYC',
            start_date='20190201',
        )
        self.assertTrue(bool(np.all(dtn.hour == arange(5))))
        self.assertTrue(bool(np.all(dtn.yyyymmdd == 20190201)))

        with self.assertRaises(TypeError):
            dtn = DateTimeNano(arange(5), start_date=1234, to_tz='NYC', from_tz='NYC')

        with self.assertRaises(TypeError):
            dtn = DateTimeNano(
                TimeSpan(NANOS_PER_HOUR * arange(5)), from_tz='NYC', to_tz='NYC'
            )

        # dates
        now = utcnow(50)
        today = Date(now)

        # create different dates
        today += arange(50)

        # create future timestamps from start dates
        future = DateTimeNano(arange(50), start_date=today, from_tz='NYC', to_tz='NYC')
        future = Date(future)
        self.assertTrue(np.all(today == future))

    def test_copy(self):

        dtn = DateTimeNano.random(5)
        dtn2 = dtn.copy()
        self.assertTrue(isinstance(dtn2, DateTimeNano))

        d = Date(5)
        d2 = d.copy()
        self.assertTrue(isinstance(d2, Date))

        ts = TimeSpan(arange(10))
        ts2 = ts.copy()
        self.assertTrue(isinstance(ts2, TimeSpan))

        ds = DateSpan(arange(5))
        ds2 = ds.copy()
        self.assertTrue(isinstance(ds2, DateSpan))

    def test_string_from_tz_error(self):
        with self.assertRaises(ValueError):
            dtn = DateTimeNano(['20190201', '20190202'])

        dtn = DateTimeNano([1, 2, 3])
        self.assertEqual(dtn._timezone._from_tz, 'UTC')

    def test_fill_invalid(self):
        dtn = DateTimeNano.random(5)
        dtn2 = dtn.fill_invalid(inplace=False)
        self.assertTrue(isinstance(dtn2, DateTimeNano))
        self.assertTrue(bool(np.all(dtn2.isnan())))

        dtn2 = dtn.fill_invalid(inplace=False, shape=3)
        self.assertTrue(isinstance(dtn2, DateTimeNano))
        self.assertTrue(bool(np.all(dtn2.isnan())))
        self.assertEqual(len(dtn2), 3)

        with self.assertRaises(ValueError):
            dtn.fill_invalid(inplace=True, shape=3)

        dtn.fill_invalid(inplace=True)
        self.assertTrue(bool(np.all(dtn.isnan())))

    def test_fill_invalid_span(self):
        dtn = TimeSpan(arange(5))
        dtn2 = dtn.fill_invalid(inplace=False)
        self.assertTrue(isinstance(dtn2, TimeSpan))
        self.assertTrue(bool(np.all(dtn2.isnan())))

        dtn2 = dtn.fill_invalid(inplace=False, shape=3)
        self.assertTrue(isinstance(dtn2, TimeSpan))
        self.assertTrue(bool(np.all(dtn2.isnan())))
        self.assertEqual(len(dtn2), 3)

        with self.assertRaises(ValueError):
            dtn.fill_invalid(inplace=True, shape=3)

        dtn.fill_invalid(inplace=True)
        self.assertTrue(bool(np.all(dtn.isnan())))

    def test_resample_interval(self):
        import pandas as pd

        dtn = DateTimeNano(
            [
                '2015-04-15 14:26:54.735321368',
                '2015-04-20 07:30:00.858219615',
                '2015-04-23 13:15:24.526871083',
                '2015-04-21 02:25:11.768548100',
                '2015-04-24 07:47:54.737776979',
                '2015-04-10 23:59:59.376589955',
            ],
            from_tz='UTC',
            to_tz='UTC',
        )
        pdt = pd.DatetimeIndex(dtn._fa)
        self.assertTrue(bool(np.all(dtn._fa == pdt.values.view(np.int64))))

        df = pd.DataFrame({'pdt': pdt})
        df = df.set_index('pdt')

        rule = '1H'
        pd_result = df.resample(rule).count().index.values.view(np.int64)
        rt_result = dtn.resample(rule)
        self.assertTrue(
            bool(np.all(pd_result == rt_result)),
            msg=f'Failed to resample with rule {rule}',
        )

        rule = '30T'
        pd_result = df.resample(rule).count().index.values.view(np.int64)
        rt_result = dtn.resample(rule)
        self.assertTrue(
            bool(np.all(pd_result == rt_result)),
            msg=f'Failed to resample with rule {rule}',
        )

        rule = 'S'
        pd_result = df.resample(rule).count().index.values.view(np.int64)
        rt_result = dtn.resample(rule)
        self.assertTrue(
            bool(np.all(pd_result == rt_result)),
            msg=f'Failed to resample with rule {rule}',
        )

        rule = '1H'
        pd_result = df.resample(rule).count().index.values.view(np.int64)
        rt_result = dtn.resample(rule)
        self.assertTrue(
            bool(np.all(pd_result == rt_result)),
            msg=f'Failed to resample with rule {rule}',
        )

    def test_resample_mod(self):
        dtn = DateTimeNano(
            [
                '2015-04-15 14:26:54.735321368',
                '2015-04-20 07:30:00.858219615',
                '2015-04-23 13:15:24.526871083',
                '2015-04-21 02:25:11.768548100',
                '2015-04-24 07:47:54.737776979',
                '2015-04-10 23:59:59.376589955',
            ],
            from_tz='UTC',
            to_tz='UTC',
        )
        correct_hour = DateTimeNano(
            [
                '2015-04-15 14:00:00',
                '2015-04-20 07:00:00',
                '2015-04-23 13:00:00',
                '2015-04-21 02:00:00',
                '2015-04-24 07:00:00',
                '2015-04-10 23:00:00',
            ],
            from_tz='UTC',
            to_tz='UTC',
        )
        h_result = dtn.resample('H', dropna=True)
        self.assertTrue(arr_eq(correct_hour, h_result))

        correct_minute = DateTimeNano(
            [
                '2015-04-15 14:26:00',
                '2015-04-20 07:30:00',
                '2015-04-23 13:15:00',
                '2015-04-21 02:25:00',
                '2015-04-24 07:47:00',
                '2015-04-10 23:59:00',
            ],
            from_tz='UTC',
            to_tz='UTC',
        )
        m_result = dtn.resample('T', dropna=True)
        self.assertTrue(arr_eq(correct_minute, m_result))

        correct_second = DateTimeNano(
            [
                '2015-04-15 14:26:54',
                '2015-04-20 07:30:00',
                '2015-04-23 13:15:24',
                '2015-04-21 02:25:11',
                '2015-04-24 07:47:54',
                '2015-04-10 23:59:59',
            ],
            from_tz='UTC',
            to_tz='UTC',
        )
        s_result = dtn.resample('S', dropna=True)
        self.assertTrue(arr_eq(correct_second, s_result))

        correct_ms = DateTimeNano(
            [
                '2015-04-15 14:26:54.735',
                '2015-04-20 07:30:00.858',
                '2015-04-23 13:15:24.526',
                '2015-04-21 02:25:11.768',
                '2015-04-24 07:47:54.737',
                '2015-04-10 23:59:59.376',
            ],
            from_tz='UTC',
            to_tz='UTC',
        )
        ms_result = dtn.resample('L', dropna=True)
        self.assertTrue(arr_eq(correct_ms, ms_result))

        correct_us = DateTimeNano(
            [
                '2015-04-15 14:26:54.735321',
                '2015-04-20 07:30:00.858219',
                '2015-04-23 13:15:24.526871',
                '2015-04-21 02:25:11.768548',
                '2015-04-24 07:47:54.737776',
                '2015-04-10 23:59:59.376589',
            ],
            from_tz='UTC',
            to_tz='UTC',
        )
        us_result = dtn.resample('U', dropna=True)
        self.assertTrue(arr_eq(correct_us, us_result))

    # def test_resample_float(self):
    #    pass

    # def test_resample_units(self):
    #    pass

    def test_resample_errors(self):
        dtn = DateTimeNano.random(5)

        # non string
        with self.assertRaises(TypeError):
            dtn.resample(1)

        # bad float
        with self.assertRaises(ValueError):
            dtn.resample('123..T')

        # no freq string
        with self.assertRaises(ValueError):
            dtn.resample('123')

        # invalid freq string
        with self.assertRaises(ValueError):
            dtn.resample('2BAD')

    def test_copy_to_tz(self):
        dtn = DateTimeNano.random(5, from_tz='GMT', to_tz='NYC')
        dtn2 = dtn.copy()
        self.assertEqual(dtn2._timezone._timezone_str, dtn._timezone._timezone_str)
        self.assertEqual(dtn2._timezone._to_tz, dtn._timezone._to_tz)
        self.assertEqual(dtn2._timezone._from_tz, dtn._timezone._from_tz)

        dtn3 = dtn[[1, 2, 3]]
        self.assertEqual(dtn3._timezone._timezone_str, dtn._timezone._timezone_str)
        self.assertEqual(dtn3._timezone._to_tz, dtn._timezone._to_tz)
        self.assertEqual(dtn3._timezone._from_tz, dtn._timezone._from_tz)

    def test_start_of_week(self):
        x = DateTimeNano(
            ['2019-11-05', '2019-11-04', '2019-11-10 23:00:00', '2019-11-11'],
            from_tz='NYC',
        )
        y = x.start_of_week
        z = Date(['2019-11-04', '2019-11-04', '2019-11-04', '2019-11-11'])
        self.assertTrue(arr_eq(z, y))

    def test_cat_expansion(self):
        x = DateTimeNano(['20191126', '20191125', '20191126'], from_tz='NYC')
        c = Cat(x)
        y = DateTimeNano(c)
        self.assertTrue(arr_eq(x, y))

        c1 = Cat(['20191126', '20191125', '20191126'])
        z = DateTimeNano(c1, format='%Y%m%d', from_tz="NYC")
        self.assertTrue(arr_eq(z, y))

    def test_cut_time(self):
        import riptable as rt
        import numpy as np

        arr1 = DateTimeNano(
            [
                '20191119 09:30:17.557593707',
                '20191119 15:31:32.216792000',
                '20191121 11:28:23.519020994',
                '20191121 11:28:56.822878000',
                '20191121 14:01:39.112893000',
                '20191121 15:46:10.838007105',
                '20191122 11:53:05.974525000',
                '20191125 10:40:32.079135847',
                '20191126 10:00:43.232329062',
                '20191126 14:04:31.421071000',
            ],
            from_tz='NYC',
            to_tz='NYC',
        )

        assert_cat_true = lambda left, right: self.assertTrue(arr_eq(left, right))

        # test 1, some starting point
        test_result = arr1.cut_time(rt.TimeSpan(1, "h"), label="left", nyc=True)
        test_expect = rt.Categorical(
            np.array([1, 7, 2, 2, 5, 7, 3, 2, 1, 5]),
            rt.FastArray(
                [b'09:30', b'10:30', b'11:30', b'12:30', b'13:30', b'14:30', b'15:30'],
                dtype='|S5',
            ),
            base_index=1,
        )
        assert_cat_true(test_result, test_expect)

        # test 2, start_time overridden
        test_result = arr1.cut_time(
            rt.TimeSpan(1, "h"), start_time=(9, 0), label="left", nyc=True
        )
        test_expect = rt.Categorical(
            np.array([1, 7, 3, 3, 6, 7, 3, 2, 2, 6]),
            rt.FastArray(
                [
                    b'09:00',
                    b'10:00',
                    b'11:00',
                    b'12:00',
                    b'13:00',
                    b'14:00',
                    b'15:00',
                    b'16:00',
                ],
                dtype='|S5',
            ),
            base_index=1,
        )
        assert_cat_true(test_result, test_expect)

        # test 3, right end point labeling
        test_result = arr1.cut_time(
            rt.TimeSpan(1, "h"), start_time=(9, 0), label="right", nyc=True
        )
        test_expect = rt.Categorical(
            np.array([1, 7, 3, 3, 6, 7, 3, 2, 2, 6]),
            rt.FastArray(
                [
                    b'10:00',
                    b'11:00',
                    b'12:00',
                    b'13:00',
                    b'14:00',
                    b'15:00',
                    b'16:00',
                    b'16:15',
                ],
                dtype='|S5',
            ),
            base_index=1,
        )
        assert_cat_true(test_result, test_expect)

        # test 4, add pre and post buckets
        test_result = arr1.cut_time(
            rt.TimeSpan(10, "m"),
            start_time=(10, 0),
            end_time=(11, 0),
            add_pre_bucket=True,
            add_post_bucket=True,
            label="left",
        )
        test_expect = rt.Categorical(
            np.array([1, 8, 8, 8, 8, 8, 8, 6, 2, 8]),
            rt.FastArray(
                [
                    b'pre',
                    b'10:00',
                    b'10:10',
                    b'10:20',
                    b'10:30',
                    b'10:40',
                    b'10:50',
                    b'post',
                ],
                dtype='|S5',
            ),
            base_index=1,
        )
        assert_cat_true(test_result, test_expect)

        # test 5, use label_fmt for matlab style time cut
        test_result = arr1.cut_time(
            [(9, 30), (9, 45), (10, 0), (11, 0), (12, 0), (16, 0)],
            label="right",
            label_fmt="to %H:%M",
            nyc=True,
        )
        test_expect = rt.Categorical(
            np.array([1, 5, 4, 4, 5, 5, 4, 3, 3, 5]),
            rt.FastArray(
                [b'to 09:45', b'to 10:00', b'to 11:00', b'to 12:00', b'to 16:00'],
                dtype='|S8',
            ),
            base_index=1,
        )
        assert_cat_true(test_result, test_expect)

        # test 6, same as 1 but without nyc=True
        test_result = arr1.cut_time(
            rt.TimeSpan(1, "h"), label="left", start_time=(9, 30), end_time=(16, 15)
        )
        test_expect = rt.Categorical(
            np.array([1, 7, 2, 2, 5, 7, 3, 2, 1, 5]),
            rt.FastArray(
                [b'09:30', b'10:30', b'11:30', b'12:30', b'13:30', b'14:30', b'15:30'],
                dtype='|S5',
            ),
            base_index=1,
        )
        assert_cat_true(test_result, test_expect)

    def test_ravel(self):
        z = Date(arange(1, 1000, 10)).ravel()[0]

    def test_timespanscalar(self):
        self.assertTrue(TimeSpan([100], unit='s')[0].seconds == 100)

    def test_datescalar_strftime(self):
        ret = Date(utcnow(4))[0].strftime('%D')
        # make sure a string was returned
        self.assertTrue(isinstance(ret, str))

    def test_timespandivision(self):
        x = TimeSpan('00:30:00')[0] / TimeSpan('01:00:00')[0]
        self.assertTrue(x == 0.5)
        x = TimeSpan('00:30:00') / TimeSpan('01:00:00')
        self.assertTrue(x[0] == 0.5)
        x = TimeSpanScalar('00:30:00') / TimeSpan('01:00:00')[0]
        self.assertTrue(x == 0.5)

    def test_maximumdatetime(self):
        x = utcnow(30)
        y = utcnow(30)
        z = maximum(x, y)
        self.assertTrue(isinstance(x, DateTimeNano))
        z = minimum(x, y)
        self.assertTrue(isinstance(x, DateTimeNano))

    def test_timespan_string_cmp(self):
        self.assertTrue( (TimeSpan(['12:00:00']) == '12:00:00').all() )
        self.assertTrue((TimeSpan(['12:00:00'])[0] == '12:00:00').all())

    def test_datetimenano_view_unspecified_type(self) -> None:
        """Test how DateTimeNano.view() behaves when an output type is not explicitly specified."""
        arr = DateTimeNano(
            [
                '20191119 09:30:17.557593707',
                '20191119 15:31:32.216792000',
                '20191121 11:28:23.519020994',
                '20191121 11:28:56.822878000',
                '20191121 14:01:39.112893000',
                '20191121 15:46:10.838007105',
                '20191122 11:53:05.974525000',
                '20191125 10:40:32.079135847',
                '20191126 10:00:43.232329062',
                '20191126 14:04:31.421071000',
            ],
            from_tz='NYC',
            to_tz='NYC',
        )
        arr.set_name(f"my_test_{type(arr).__name__}")
        arr_view = arr.view()
        self.assertEqual(type(arr), type(arr_view))
        self.assertEqual(arr.shape, arr_view.shape)
        self.assertFalse(arr_view.flags.owndata)
        self.assertEqual(arr.get_name(), arr_view.get_name())
        self.assertTrue((arr == arr_view).all())
        # DateTimeNano-specific checks
        self.assertEqual(arr._timezone, arr_view._timezone)

    def test_date_view_unspecified_type(self) -> None:
        """Test how Date.view() behaves when an output type is not explicitly specified."""
        arr = Date(['2019-11-04', '2019-11-04', '2019-11-04', '2019-11-11'])
        arr.set_name(f"my_test_{type(arr).__name__}")
        arr_view = arr.view()
        self.assertEqual(type(arr), type(arr_view))
        self.assertEqual(arr.shape, arr_view.shape)
        self.assertFalse(arr_view.flags.owndata)
        self.assertEqual(arr.get_name(), arr_view.get_name())
        self.assertTrue((arr == arr_view).all())
        # Date-specific checks
        # (None)

    def test_timespan_view_unspecified_type(self) -> None:
        """Test how Date.view() behaves when an output type is not explicitly specified."""
        arr = TimeSpan([
            '09:30:17.557593707',
            '15:31:32.216792000',
            '11:28:23.519020994',
            '19:46:10.838007105',
            '09:30:29.999999999',
            '10:40:00.000000000',
            '00:00:00.999999999',
            '23:59:59.999999999',
        ])
        arr.set_name(f"my_test_{type(arr).__name__}")
        arr_view = arr.view()
        self.assertEqual(type(arr), type(arr_view))
        self.assertEqual(arr.shape, arr_view.shape)
        self.assertFalse(arr_view.flags.owndata)
        self.assertEqual(arr.get_name(), arr_view.get_name())
        self.assertTrue((arr == arr_view).all())
        # TimeSpan-specific checks
        # (None)

class TestTimeZone(unittest.TestCase):
    def test_equal_positive(self) -> None:
        tz1 = TimeZone(from_tz='NYC', to_tz='GMT')
        tz2 = TimeZone(from_tz='NYC', to_tz='GMT')

        # The two instances should be considered equal since they have
        # equal `from_tz` and `to_tz` inputs.
        self.assertEqual(tz1, tz2)

    def test_equal_negative_diff_fromtz(self) -> None:
        tz1 = TimeZone(from_tz='GMT', to_tz='NYC')
        tz2 = TimeZone(from_tz='NYC', to_tz='NYC')

        # The two instances should be considered unequal since they have
        # unequal `from_tz` inputs.
        self.assertNotEqual(tz1, tz2)

    def test_equal_negative_diff_totz(self) -> None:
        tz1 = TimeZone(from_tz='NYC', to_tz='NYC')
        tz2 = TimeZone(from_tz='NYC', to_tz='GMT')

        # The two instances should be considered unequal since they have
        # unequal `to_tz` inputs.
        self.assertNotEqual(tz1, tz2)

    def test_repr(self) -> None:
        test_tz = TimeZone(from_tz='NYC', to_tz='GMT')

        # Quick check that we can call repr() on the object (without some kind of
        # exception being raised) and that it returns a reasonable result.
        test_tz_repr = repr(test_tz)
        self.assertEqual(test_tz_repr, "TimeZone(from_tz='NYC', to_tz='GMT')")



# TODO RIP-486 add tests for other date types from rt_datetime
@pytest.mark.parametrize(
    "typ,arrays",
    [
        pytest.param(
            DateTimeNano,
            (
                    DateTimeNano([], from_tz='NYC', to_tz='NYC'),
            ),
        ),
        (
            DateTimeNano,
            (
                    DateTimeNano([], from_tz='NYC', to_tz='NYC'),
                    DateTimeNano(['20191126 10:00:43.232329062'], from_tz='NYC', to_tz='NYC')
            )
        ),
        (
            DateTimeNano,
            (
                    DateTimeNano(['20191126 14:04:31.421071000',], from_tz='NYC', to_tz='NYC'),
                    DateTimeNano([], from_tz='NYC', to_tz='NYC')
            )
        ),
        (
            DateTimeNano,
            (
                    DateTimeNano(['20191119 09:30:17.557593707', '20191119 15:31:32.216792000', '20191121 11:28:23.519020994'], from_tz='NYC', to_tz='NYC'),
                    DateTimeNano(['20191119 09:30:17.557593707', '20191119 15:31:32.216792000', '20191121 11:28:23.519020994'], from_tz='NYC', to_tz='NYC'),
                    DateTimeNano(['20191126 10:00:43.232329062', '20191126 14:04:31.421071000',], from_tz='NYC', to_tz='NYC'),
                    DateTimeNano(['20191126 10:00:43.232329062', '20191126 14:04:31.421071000',], from_tz='NYC', to_tz='NYC'),
            )
        ),
    ]
)
def test_concatenate_preserves_datetime_type_regression(typ, arrays):
    fn = 'test_concatenate_preserves_datetime_type_regression'
    # assumption - tuple of arrays are same type; catch tester errors
    # TODO test scenarios where this assumption does not hold
    assert all([isinstance(a, typ) for a in arrays]), f'{fn}: expected tuple of arrays to be the same type'

    result = concatenate(arrays)
    assert isinstance(result, typ)

@pytest.mark.parametrize(
    "obj",
    [
        pytest.param(Date(['2018-02-01']), id='Date'),
        pytest.param(DateSpan([123]), id='DateSpan'),
        pytest.param(DateTimeNano(['19900401 02:00:00'], from_tz='NYC'), id='DateTimeNano_NYC'),
        pytest.param(DateTimeNano(['20210921 02:21:21'], from_tz='GMT'), id='DateTimeNano_GMT'),
        pytest.param(TimeSpan(['12.34']), id='TimeSpan'),
        # the following don't return the same type from self.view()
        #pytest.param(DateScalar(34567), id='DateScalar'),
        #pytest.param(DateSpanScalar(12345), id='DateSpanScalar'),
        #pytest.param(DateTimeNanoScalar(87654321), id='DateTimeNanoScalar'),
        #pytest.param(TimeSpanScalar(8.76543), id='TimeSpanScalar'),
    ]
)
def test_new_from_template(obj):
    objView = obj.view()
    assert repr(objView) == repr(obj)
    members = dir(obj)
    viewMembers = dir(objView)
    missing = [m for m in members if m not in viewMembers]
    assert len(missing) == 0, f"Not found: {missing}"

@pytest.mark.parametrize(
    "cls,arr",
    [
        pytest.param(Date, np.array(['2018-02-01']), id='Date'),
        pytest.param(DateSpan, np.array([123]), id='DateSpan'),
        pytest.param(DateTimeNano, np.array([1632164888]), id='DateTimeNano'),
        pytest.param(TimeSpan, np.array(['12.34']), id='TimeSpan'),
    ]
)
def test_view_casting(cls, arr):
    obj = cls(arr)
    objView = obj.view()
    assert repr(objView) == repr(obj)
    members = dir(obj)
    viewMembers = dir(objView)
    missing = [m for m in members if m not in viewMembers]
    assert len(missing) == 0, f"Not found: {missing}"

if __name__ == '__main__':
    tester = unittest.main()
