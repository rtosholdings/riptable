import unittest
from riptable import *


class TimeWindow_Test(unittest.TestCase):
    def test_time_window(self):
        a = arange(100)
        r = rc.TimeWindow(uint64(a), int64(a), 0, 0)
        self.assertEqual(r[99], 99, msg=f"Wrong result produced for timewindow {r}")

        r = rc.TimeWindow(single(a), int64(a), 0, 0)
        self.assertEqual(r[99], 99.0, msg=f"Wrong result produced for timewindow {r}")

        r = rc.TimeWindow(int64(a), int64(a), 0, 0)
        self.assertEqual(r[99], 99, msg=f"Wrong result produced for timewindow {r}")

        r = rc.TimeWindow(int64(a), int64(a), 0, 3)
        self.assertEqual(
            r[99], 99 + 98 + 97 + 96, msg=f"Wrong result produced for timewindow {r}"
        )


if __name__ == "__main__":
    tester = unittest.main()
