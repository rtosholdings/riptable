from typing import List, Optional
import abc
import datetime
import pytz
import dateutil.rrule
from matplotlib import units, ticker, dates

import riptable as rt

__all__ = []

sizes_ns = {
    'MICROSECONDLY': 1_000,
    'SECONDLY': 1_000_000_000,
    'MINUTELY': 60 * 1_000_000_000,
    'HOURLY': 60 * 60 * 1_000_000_000,
    'DAILY': 24 * 60 * 60 * 1_000_000_000,
    'MONTHLY': 30 * 24 * 60 * 60 * 1_000_000_000,
    'YEARLY': 365 * 24 * 60 * 60 * 1_000_000_000,
}

NANOS_PER_SECOND = 1_000_000_000
MICROS_PER_SECOND = 1_000_000


def dts_to_ns(dts):
    return [dt_to_ns(dt) for dt in dts]


def td_to_ns(td):
    ns = td.days * sizes_ns['DAILY'] + td.seconds * sizes_ns['SECONDLY'] + td.microseconds * 1_000
    return ns


def dt_to_ns(dt):
    return int(dt.timestamp() * NANOS_PER_SECOND)


class DTTicker(abc.ABC):
    tz: pytz.tzfile.DstTzInfo
    interva: int
    start_ns: int
    end_ns: int

    start_utc: datetime.datetime
    start_local: datetime.datetime
    end_utc: datetime.datetime
    end_local: datetime.datetime
    aligned_utc: datetime.datetime
    aligned_local: datetime.datetime

    microsecond_aligned_ticks: List[datetime.datetime]
    millisecond_aligned_ticks: List[datetime.datetime]
    second_aligned_ticks: List[datetime.datetime]
    minute_aligned_ticks: List[datetime.datetime]
    hour_aligned_ticks: List[datetime.datetime]
    day_aligned_ticks: List[datetime.datetime]
    month_aligned_ticks: List[datetime.datetime]
    year_aligned_ticks: List[datetime.datetime]

    ticks_utc: List[datetime.datetime]
    ticks_local: List[datetime.datetime]
    ticks_ns: List[int]

    offset_local: Optional[datetime.datetime]
    tick_format: str
    offset_format: str

    def __init__(self, tz, interval, start_ns, end_ns):
        self.tz = tz
        self.interval = interval
        self.start_ns = start_ns
        self.end_ns = end_ns

        self.make_times()
        self.make_ticks()
        self.find_aligned_ticks()
        self.make_formats()
        return

    def make_times(self):
        self.start_utc = datetime.datetime.fromtimestamp(self.start_ns / NANOS_PER_SECOND, tz=datetime.timezone.utc)
        self.start_local = self.start_utc.astimezone(self.tz)

        self.end_utc = datetime.datetime.fromtimestamp(self.end_ns / NANOS_PER_SECOND, tz=datetime.timezone.utc)
        self.end_local = self.end_utc.astimezone(self.tz)

        self.aligned_local = self.get_aligned()
        self.aligned_utc = self.aligned_local.astimezone(datetime.timezone.utc)
        return

    @abc.abstractmethod
    def get_aligned(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def make_ticks(self):
        raise NotImplementedError()

    def find_aligned_ticks(self):
        self.microsecond_aligned_ticks = [tick for tick in self.ticks_local if tick.microsecond % 1 == 0]
        self.millisecond_aligned_ticks = [
            tick for tick in self.microsecond_aligned_ticks if tick.microsecond % 1_000 == 0
        ]
        self.second_aligned_ticks = [tick for tick in self.millisecond_aligned_ticks if tick.microsecond == 0]
        self.minute_aligned_ticks = [tick for tick in self.second_aligned_ticks if tick.second == 0]
        self.hour_aligned_ticks = [tick for tick in self.minute_aligned_ticks if tick.minute == 0]
        self.day_aligned_ticks = [tick for tick in self.hour_aligned_ticks if tick.hour == 0]
        self.month_aligned_ticks = [tick for tick in self.day_aligned_ticks if tick.day == 0]
        self.year_aligned_ticks = [tick for tick in self.month_aligned_ticks if tick.month == 0]
        return

    @abc.abstractmethod
    def make_formats(self):
        raise NotImplementedError()


class LinTicker(DTTicker):
    step_us = None

    def make_ticks(self):
        pos = self.aligned_utc
        step = datetime.timedelta(microseconds=self.interval * self.step_us)

        ticks = list()
        while pos < self.end_utc:
            if pos >= self.start_utc:
                ticks.append(pos)
            pos += step
        self.ticks_utc = ticks
        self.ticks_local = [tick.astimezone(self.tz) for tick in self.ticks_utc]
        self.ticks_ns = dts_to_ns(self.ticks_utc)
        return


class MicrosecondTicker(LinTicker):
    step_us = 1

    def get_aligned(self):
        aligned = self.tz.localize(self.start_local.replace(microsecond=0, tzinfo=None))
        return aligned

    def make_formats(self):
        if len(self.second_aligned_ticks) == 0:
            first_tick = self.ticks_local[0]
            self.offset_local = first_tick.replace(second=0, microsecond=0)
            self.tick_format = '%fus'
            self.offset_format = '%Y-%m-%d %H:%M:%S'

        elif len(self.second_aligned_ticks) == 1:
            self.offset_local = self.second_aligned_ticks[0]
            self.tick_format = '%fus'
            self.offset_format = '%Y-%m-%d %H:%M:%S'

        else:
            first_tick = self.ticks_local[0]
            self.offset_local = first_tick.replace(hour=0, minute=0, second=0, microsecond=0)
            self.tick_format = '%H:%M:%S.%f'
            self.offset_format = '%Y-%m-%d'
        return


class RRTicker(DTTicker):
    rr: dateutil.rrule.rrule

    freq = None

    def make_ticks(self):
        self.rr = dateutil.rrule.rrule(
            self.freq,
            self.aligned_utc,
            self.interval,
        )
        self.ticks_utc = self.rr.between(self.start_utc, self.end_utc)
        self.ticks_local = [tick.astimezone(self.tz) for tick in self.ticks_utc]
        self.ticks_ns = dts_to_ns(self.ticks_utc)
        return


class SecondTicker(RRTicker):
    freq = dateutil.rrule.SECONDLY

    def get_aligned(self):
        aligned = self.tz.localize(self.start_local.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None))
        return aligned

    def make_formats(self):
        if len(self.day_aligned_ticks) == 0:
            first_tick = self.ticks_local[0]
            self.offset_local = first_tick.replace(hour=0, minute=0, second=0, microsecond=0)
            self.tick_format = '%H:%M:%S'
            self.offset_format = '%Y-%m-%d'
        elif len(self.day_aligned_ticks) == 1:
            self.offset_local = self.day_aligned_ticks[0]
            self.tick_format = '%H:%M:%S'
            self.offset_format = '%Y-%m-%d'
        else:
            self.offset_local = None
            self.tick_format = '%H:%M:%S'
            self.offset_format = '%Y-%m-%d'
        return


class MinuteTicker(RRTicker):
    freq = dateutil.rrule.MINUTELY

    def get_aligned(self):
        aligned = self.tz.localize(self.start_local.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None))
        return aligned

    def make_formats(self):
        if len(self.day_aligned_ticks) == 0:
            first_tick = self.ticks_local[0]
            self.offset_local = first_tick.replace(hour=0, minute=0, second=0, microsecond=0)
            self.tick_format = '%H:%M:%S'
            self.offset_format = '%Y-%m-%d'
        elif len(self.day_aligned_ticks) == 1:
            self.offset_local = self.day_aligned_ticks[0]
            self.tick_format = '%H:%M:%S'
            self.offset_format = '%Y-%m-%d'
        else:
            self.offset_local = None
            self.tick_format = '%H:%M:%S'
            self.offset_format = '%Y-%m-%d'
        return


class HourTicker(RRTicker):
    freq = dateutil.rrule.HOURLY

    def get_aligned(self):
        aligned = self.tz.localize(self.start_local.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None))
        return aligned

    def make_formats(self):
        if len(self.day_aligned_ticks) == 0:
            first_tick = self.ticks_local[0]
            self.offset_local = first_tick.replace(hour=0, minute=0, second=0, microsecond=0)
            self.tick_format = '%H:%M:%S'
            self.offset_format = '%Y-%m-%d'
        elif len(self.day_aligned_ticks) == 1:
            self.offset_local = self.day_aligned_ticks[0]
            self.tick_format = '%H:%M:%S'
            self.offset_format = '%Y-%m-%d'
        else:
            self.offset_local = None
            self.tick_format = '%Y-%m-%d %H:%M:%S'
            self.offset_format = ''
        return


class DayTicker(RRTicker):
    freq = dateutil.rrule.DAILY

    def get_aligned(self):
        aligned = self.tz.localize(
            self.start_local.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        )
        return aligned

    def make_formats(self):
        self.offset_local = None
        self.tick_format = '%Y-%m-%d'
        self.offset_format = ''
        return


class MonthTicker(RRTicker):
    freq = dateutil.rrule.MONTHLY

    def get_aligned(self):
        aligned = self.tz.localize(
            self.start_local.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        )
        return aligned

    def make_formats(self):
        self.offset_local = None
        self.tick_format = '%Y-%m'
        self.offset_format = ''
        return


class YearTicker(RRTicker):
    freq = dateutil.rrule.YEARLY

    def get_aligned(self):
        aligned = self.tz.localize(
            self.start_local.replace(year=1950, month=1, day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        )
        return aligned

    def make_formats(self):
        self.offset_local = None
        self.tick_format = '%Y'
        self.offset_format = ''
        return


class TickFormatter(ticker.Formatter):
    def __init__(self, locator):
        self.locator = locator
        return

    def __call__(self, x, pos=None):
        return self.format_tick(x)

    def format_tick(self, tick_ns):
        tick_utc = datetime.datetime.fromtimestamp(tick_ns / NANOS_PER_SECOND, tz=datetime.timezone.utc)
        tick_local = tick_utc.astimezone(self.locator.ticker.tz)
        name = tick_local.strftime(self.locator.ticker.tick_format)
        if tick_local == self.locator.ticker.offset_local:
            name = '*' + name
        return name

    def format_offset(self):
        if self.locator.ticker.offset_local is None:
            name = ''
        else:
            name = self.locator.ticker.offset_local.strftime(self.locator.ticker.offset_format)
        self.offset_str = name

    def get_offset(self):
        self.format_offset()
        return self.offset_str


class DateTimeNanoLocator(ticker.Locator):
    """
    Locator for axes whose values are given in nanoseconds.
    """

    intervals = {
        'MICROSECONDLY': [1, 2, 5, 10, 25, 50, 100, 250, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000],
        'SECONDLY': [1, 2, 5, 10, 15, 30],
        'MINUTELY': [1, 2, 5, 10, 15, 30],
        'HOURLY': [1, 3, 4, 6, 12],
        'DAILY': [1, 2, 3, 7, 14, 21],
        'MONTHLY': [1, 2, 3, 4, 6],
        'YEARLY': [1, 2, 4, 5, 10, 20, 40, 50],
    }

    max_ticks = {
        'MICROSECONDLY': 10,
        'SECONDLY': 10,
        'MINUTELY': 10,
        'HOURLY': 10,
        'DAILY': 10,
        'MONTHLY': 10,
        'YEARLY': 10,
    }

    def __init__(self, tz):
        self.scale = None
        self.interval = None
        self.tz = tz
        return

    def __call__(self):
        (vmin, vmax) = self.axis.get_view_interval()
        self.ticks = self.tick_values(vmin, vmax)
        return self.ticks

    def tick_values(self, vmin, vmax):
        # Compute how wide the view is.
        width_nanos = abs(vmax - vmin)
        (self.scale, self.interval) = self.get_tick_interval(width_nanos)
        ticks = self.get_ticks(vmin, vmax)
        return ticks

    def get_tick_interval(self, width):
        for (scale, sizes) in self.intervals.items():
            nanos = sizes_ns[scale]
            max_tick = self.max_ticks[scale]
            for size in sizes:
                if width / (nanos * size) < max_tick:
                    return (scale, size)
        else:
            raise ValueError("Could not determine tick interval")

    def get_ticks(self, start, end):
        start = int(start)
        end = int(end)
        if self.scale == 'MICROSECONDLY':
            self.ticker = MicrosecondTicker(self.tz, self.interval, start, end)
        elif self.scale == 'SECONDLY':
            self.ticker = SecondTicker(self.tz, self.interval, start, end)
        elif self.scale == 'MINUTELY':
            self.ticker = MinuteTicker(self.tz, self.interval, start, end)
        elif self.scale == 'HOURLY':
            self.ticker = HourTicker(self.tz, self.interval, start, end)
        elif self.scale == 'DAILY':
            self.ticker = DayTicker(self.tz, self.interval, start, end)
        elif self.scale == 'MONTHLY':
            self.ticker = MonthTicker(self.tz, self.interval, start, end)
        elif self.scale == 'YEARLY':
            self.ticker = YearTicker(self.tz, self.interval, start, end)
        else:
            raise ValueError('Scale not recognized.')

        return self.ticker.ticks_ns


class DateTimeNanoScalarConverter(units.ConversionInterface):
    @staticmethod
    def default_units(x, axis):
        return ('ns', x._timezone._to_tz)

    @staticmethod
    def axisinfo(unit, axis):
        (u, rt_tz) = unit
        if rt_tz == 'NYC':
            tz = pytz.timezone('America/New_York')
        else:
            tz = pytz.utc

        majloc = DateTimeNanoLocator(tz)
        majfmt = TickFormatter(majloc)
        info = units.AxisInfo(
            majloc=majloc,
            majfmt=majfmt,
            label='DateTime',
        )
        return info


class DateTimeScalarConverter(units.ConversionInterface):
    @staticmethod
    def default_units(x, axis):
        return 'date'

    @staticmethod
    def axisinfo(unit, axis):
        majloc = dates.AutoDateLocator()
        majfmt = dates.AutoDateFormatter(majloc)
        info = units.AxisInfo(
            majloc=majloc,
            majfmt=majfmt,
            label='Date',
        )
        return info


class TSTicker(abc.ABC):
    step_us = None

    def __init__(
        self,
        interval,
        start_ns,
        end_ns,
    ):
        self.interval = interval
        self.start_ns = start_ns
        self.end_ns = end_ns

        self.make_time_deltas()
        self.make_ticks()
        self.find_aligned_ticks()
        self.make_formats()
        return

    def make_time_deltas(self):
        self.start_td = datetime.timedelta(microseconds=self.start_ns / 1_000)
        self.end_td = datetime.timedelta(microseconds=self.end_ns / 1_000)
        self.aligned_td = self.get_aligned()
        return

    @abc.abstractmethod
    def get_aligned(self):
        raise NotImplementedError()

    def make_ticks(self):
        pos = self.aligned_td
        step = datetime.timedelta(microseconds=self.interval * self.step_us)
        ticks = list()
        while pos < self.end_td:
            if pos > self.start_td:
                ticks.append(pos)
            pos += step
        self.ticks_td = ticks
        self.ticks_ns = [td_to_ns(td) for td in self.ticks_td]

    def find_aligned_ticks(self):
        self.microsecond_aligned_ticks = [tick for tick in self.ticks_td if tick.microseconds % 1 == 0]
        self.millisecond_aligned_ticks = [
            tick for tick in self.microsecond_aligned_ticks if tick.microseconds % 1_000 == 0
        ]
        self.second_aligned_ticks = [tick for tick in self.millisecond_aligned_ticks if tick.microseconds == 0]
        self.minute_aligned_ticks = [tick for tick in self.second_aligned_ticks if tick.seconds % 60 == 0]
        self.hour_aligned_ticks = [tick for tick in self.minute_aligned_ticks if tick.seconds % 3600 == 0]
        self.day_aligned_ticks = [tick for tick in self.hour_aligned_ticks if tick.seconds == 0]

    @abc.abstractmethod
    def make_formats(self):
        raise NotImplementedError()


class MicrosecondTSTicker(TSTicker):
    step_us = 1

    def get_aligned(self):
        aligned = self.start_td - datetime.timedelta(microseconds=self.start_td.microseconds)
        return aligned

    def make_formats(self):
        if len(self.day_aligned_ticks) == 0:
            first_tick = self.ticks_td[0]
            self.offset = first_tick - datetime.timedelta(microseconds=first_tick.microseconds)
            self.tick_format = '{F:06}us'
            self.offset_format = '{D} days, {H:02}:{M:02}:{S:02}'

        elif len(self.day_aligned_ticks) == 1:
            self.offset = self.day_aligned_ticks[0]
            self.tick_format = '{F:06}us'
            self.offset_format = '{D} days, {H:02}:{M:02}:{S:02}'

        else:
            self.offset = None
            self.tick_format = '{D} days, {H:02}:{M:02}:{S:02}'
            self.offset_format = ''
        return


class SecondTSTicker(TSTicker):
    step_us = 1_000_000

    def get_aligned(self):
        aligned = self.start_td - datetime.timedelta(
            seconds=self.start_td.seconds, microseconds=self.start_td.microseconds
        )
        return aligned

    def make_formats(self):
        if len(self.day_aligned_ticks) == 0:
            first_tick = self.ticks_td[0]
            self.offset = first_tick - datetime.timedelta(first_tick.seconds)
            self.tick_format = '{H:02}:{M:02}:{S:02}'
            self.offset_format = '{D} days'

        elif len(self.day_aligned_ticks) == 1:
            self.offset = self.day_aligned_ticks[0]
            self.tick_format = '{H:02}:{M:02}:{S:02}'
            self.offset_format = '{D} days'

        else:
            self.offset = None
            self.tick_format = '{D} days, {H:02}:{M:02}:{S:02}'
            self.offset_format = ''
        return


class MinuteTSTicker(TSTicker):
    step_us = 60 * 1_000_000

    def get_aligned(self):
        aligned = self.start_td - datetime.timedelta(
            seconds=self.start_td.seconds, microseconds=self.start_td.microseconds
        )
        return aligned

    def make_formats(self):
        if len(self.day_aligned_ticks) == 0:
            first_tick = self.ticks_td[0]
            self.offset = first_tick - datetime.timedelta(
                seconds=first_tick.seconds, microseconds=first_tick.microseconds
            )
            self.tick_format = '{H:02}:{M:02}:{S:02}'
            self.offset_format = '{D} days'

        elif len(self.day_aligned_ticks) == 1:
            self.offset = self.day_aligned_ticks[0]
            self.tick_format = '{H:02}:{M:02}:{S:02}'
            self.offset_format = '{D} days'

        else:
            self.offset = None
            self.tick_format = '{D} days, {H:02}:{M:02}:{S:02}'
            self.offset_format = ''
        return


class HourTSTicker(TSTicker):
    step_us = 60 * 60 * 1_000_000

    def get_aligned(self):
        aligned = self.start_td - datetime.timedelta(
            seconds=self.start_td.seconds, microseconds=self.start_td.microseconds
        )
        return aligned

    def make_formats(self):
        if len(self.day_aligned_ticks) == 0:
            first_tick = self.ticks_td[0]
            self.offset = first_tick - datetime.timedelta(
                seconds=first_tick.seconds, microseconds=first_tick.microseconds
            )
            self.tick_format = '{H:02}:{M:02}:{S:02}'
            self.offset_format = '{D} days'

        elif len(self.day_aligned_ticks) == 1:
            self.offset = self.day_aligned_ticks[0]
            self.tick_format = '{H:02}:{M:02}:{S:02}'
            self.offset_format = '{D} days'

        else:
            self.offset = None
            self.tick_format = '{D} days, {H:02}:{M:02}:{S:02}'
            self.offset_format = ''
        return


class DayTSTicker(TSTicker):
    step_us = 24 * 60 * 60 * 1_000_000

    def get_aligned(self):
        aligned = self.start_td - datetime.timedelta(
            seconds=self.start_td.seconds, microseconds=self.start_td.microseconds
        )
        return aligned

    def make_formats(self):
        self.offset = None
        self.tick_format = '{D} days'
        self.offset_format = ''
        return


class TimeSpanLocator(ticker.Locator):
    intervals = {
        'MICROSECONDLY': [1, 2, 5, 10, 25, 50, 100, 250, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000],
        'SECONDLY': [1, 2, 5, 10, 15, 30],
        'MINUTELY': [1, 2, 5, 10, 15, 30],
        'HOURLY': [1, 3, 4, 6, 12],
        'DAILY': [1, 2, 3, 7, 14, 21, 30, 60, 90, 180, 365, 730, 3650],
    }

    max_ticks = {
        'MICROSECONDLY': 10,
        'SECONDLY': 10,
        'MINUTELY': 10,
        'HOURLY': 10,
        'DAILY': 10,
    }

    def __init__(self):
        self.scale = None
        self.interval = None
        return

    def __call__(self):
        (vmin, vmax) = self.axis.get_view_interval()
        self.ticks = self.tick_values(vmin, vmax)
        return self.ticks

    def tick_values(self, vmin, vmax):
        # Compute how wide the view is.
        width_nanos = abs(vmax - vmin)
        (self.scale, self.interval) = self.get_tick_interval(width_nanos)
        ticks = self.get_ticks(vmin, vmax)
        return ticks

    def get_tick_interval(self, width):
        for (scale, sizes) in self.intervals.items():
            nanos = sizes_ns[scale]
            max_tick = self.max_ticks[scale]
            for size in sizes:
                if width / (nanos * size) < max_tick:
                    return (scale, size)
        else:
            raise ValueError("Could not determine tick interval")

    def get_ticks(self, start, end):
        start = int(start)
        end = int(end)
        if self.scale == 'MICROSECONDLY':
            self.ticker = MicrosecondTSTicker(self.interval, start, end)
        elif self.scale == 'SECONDLY':
            self.ticker = SecondTSTicker(self.interval, start, end)
        elif self.scale == 'MINUTELY':
            self.ticker = MinuteTSTicker(self.interval, start, end)
        elif self.scale == 'HOURLY':
            self.ticker = HourTSTicker(self.interval, start, end)
        elif self.scale == 'DAILY':
            self.ticker = DayTSTicker(self.interval, start, end)
        else:
            raise ValueError('Scale not recognized.')

        return self.ticker.ticks_ns


def td_to_dhmsf(td):
    d = td.days
    sec_rem = td.seconds
    (h, sec_rem) = divmod(sec_rem, 3600)
    (m, s) = divmod(sec_rem, 60)
    f = td.microseconds
    return (d, h, m, s, f)


class TimeSpanFormatter(ticker.Formatter):
    def __init__(self, locator):
        self.locator = locator
        return

    def __call__(self, x, pos=None):
        return self.format_tick(x)

    def format_tick(self, tick_ns):
        tick_td = datetime.timedelta(microseconds=tick_ns / 1_000)
        (days, hours, minutes, seconds, microseconds) = td_to_dhmsf(tick_td)
        name = self.locator.ticker.tick_format.format(
            D=days,
            H=hours,
            M=minutes,
            S=seconds,
            F=microseconds,
        )
        if tick_td == self.locator.ticker.offset:
            name = '*' + name
        return name

    def format_offset(self):
        if self.locator.ticker.offset is None:
            name = ''
        else:
            offset = self.locator.ticker.offset
            (days, hours, minutes, seconds, microseconds) = td_to_dhmsf(offset)
            name = self.locator.ticker.offset_format.format(
                D=days,
                H=hours,
                M=minutes,
                S=seconds,
                F=microseconds,
            )
        self.offset_str = name

    def get_offset(self):
        self.format_offset()
        return self.offset_str


class TimeSpanConverter(units.ConversionInterface):
    @staticmethod
    def default_units(x, axis):
        return 'ns'

    @staticmethod
    def axisinfo(unit, axis):
        majloc = TimeSpanLocator()
        majfmt = TimeSpanFormatter(majloc)
        info = units.AxisInfo(
            majloc=majloc,
            majfmt=majfmt,
            label='TimeSpan',
        )
        return info


units.registry[rt.DateTimeNanoScalar] = DateTimeNanoScalarConverter
units.registry[rt.DateScalar] = DateTimeScalarConverter
units.registry[rt.TimeSpanScalar] = TimeSpanConverter
