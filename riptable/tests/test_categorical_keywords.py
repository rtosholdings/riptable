import unittest
import pandas as pd
from riptable import *


class Categorical_keyword_Test(unittest.TestCase):
    def test_invalid(self):
        c = Categorical(["b", "a", "a", "Inv", "c", "a", "b"], invalid="Inv")
        self.assertTrue(b"Inv" in c.category_array)
        self.assertEqual(c.base_index, 1)

        # with self.assertRaises(ValueError):
        #    c = Categorical(['b','a','a','Inv','c','a','b'], invalid='Inv', base_index=0)

        c = Categorical(["b", "a", "a", "c", "a", "b"], invalid="Inv", base_index=0)
        self.assertTrue(b"Inv" not in c.category_array)

        with self.assertRaises(ValueError):
            c = Categorical(["b", "a", "a", "Inv", "c", "a", "b"], ["a", "b", "c"])

        # 5/19/2019 invalid category must be in uniques
        with self.assertRaises(ValueError):
            c = Categorical(
                ["b", "a", "a", "Inv", "c", "a", "b"], ["a", "b", "c"], invalid="Inv"
            )
            self.assertTrue("Inv" not in c.category_array)
            self.assertTrue(b"Inv" not in c.category_array)
            self.assertEqual(c.base_index, 1)
            self.assertEqual(c._fa[3], 0)
        with self.assertRaises(ValueError):
            c = Categorical(
                ["b", "a", "a", "Inv", "c", "a", "b", "d", "e"],
                ["a", "b", "c"],
                invalid="Invalid",
            )
            self.assertTrue("Invalid" not in c.category_array)
            self.assertTrue(b"Invalid" not in c.category_array)
            self.assertEqual(c.base_index, 1)
            for i in [3, 7, 8]:
                self.assertEqual(c._fa[i], 0)

        c = Categorical([2, 1, 1, 0, 1, 2, 3], ["c", "a", "b"], invalid="Invalid")
        self.assertTrue("Invalid" not in c.category_array)
        self.assertTrue(b"Invalid" not in c.category_array)
        self.assertEqual(c.base_index, 1)
        self.assertEqual(c._fa[3], 0)

        # with self.assertRaises(ValueError):
        #    c = Categorical([2,1,1,0,1,2], ['c','a','b'], invalid='Invalid', base_index=0)

    def test_base_index_zero_not_allowed(self):
        with self.assertRaises(ValueError):
            c = Categorical(
                ["b", "a", "a", "c", "a", "b"],
                filter=FA([True, True, False, True, True, True]),
                base_index=0,
            )
        with self.assertRaises(ValueError):
            c = Categorical(
                ["b", "a", "a", "c", "a", "b"],
                ["a", "b", "c"],
                filter=FA([True, True, False, True, True, True]),
                base_index=0,
            )
        with self.assertRaises(ValueError):
            c = Categorical(
                [1, 0, 0, 2, 0, 1],
                ["a", "b", "c"],
                filter=FA([True, True, False, True, True, True]),
                base_index=0,
            )
        # with self.assertRaises(ValueError):
        #    c = Categorical(['b','a','a','c','a','b'], ['a','b','c'], invalid='c', base_index=0)
        with self.assertRaises(ValueError):
            pdc = pd.Categorical(["b", "a", "a", "c", "a", "b"])
            c = Categorical(pdc, base_index=0)
        with self.assertRaises(ValueError):
            c = Categorical(
                [2.0, 1.0, 1.0, 3.0, 1.0, 2.0],
                ["a", "b", "c"],
                from_matlab=True,
                base_index=0,
            )
        # muted this warning to simplify categorical constructor
        # with self.assertWarns(UserWarning):
        #    c = Categorical([2,1,1,3,1,2], {'a':1, 'b':2, 'c':3}, base_index=0)


if __name__ == "__main__":
    tester = unittest.main()
