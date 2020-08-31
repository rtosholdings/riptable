import unittest

from riptable import *


def arr_eq(a, b):
    return bool(np.all(a == b))


def arr_all(a):
    return bool(np.all(a))


class CategoricalSetItem_Test(unittest.TestCase):
    def test_setitem_array_bool(self):
        c = Categorical(["a", "a", "b", "c", "a"])
        b = logical(arange(5) % 2)
        c[b] = "b"
        self.assertTrue(arr_eq(c._fa, FastArray([1, 2, 2, 2, 1])))
        with self.assertRaises(ValueError):
            c[b] = "f"
        c.auto_add_on()
        c[b] = "f"
        self.assertTrue(arr_eq(c._fa, FastArray([1, 4, 2, 4, 1])))

    def test_setitem_array_singleint(self):
        c = Categorical(["a", "a", "b", "c", "a"])
        c[1] = "b"
        c[3] = "b"
        self.assertTrue(arr_eq(c._fa, FastArray([1, 2, 2, 2, 1])))
        with self.assertRaises(ValueError):
            c[1] = "f"
        c.auto_add_on()
        c[1] = "f"
        c[3] = "f"
        self.assertTrue(arr_eq(c._fa, FastArray([1, 4, 2, 4, 1])))

    def test_setitem_array_index(self):
        c = Categorical(["a", "a", "b", "c", "a"])
        b = bool_to_fancy(logical(arange(5) % 2))
        c[b] = "b"
        self.assertTrue(arr_eq(c._fa, FastArray([1, 2, 2, 2, 1])))
        with self.assertRaises(ValueError):
            c[b] = "f"
        c.auto_add_on()
        c[b] = "f"
        self.assertTrue(arr_eq(c._fa, FastArray([1, 4, 2, 4, 1])))

    def test_setitem_array_string(self):
        c = Categorical(["a", "a", "b", "c", "a"])
        c["a"] = "b"
        self.assertTrue(arr_eq(c._fa, FastArray([2, 2, 2, 3, 2])))
        with self.assertRaises(ValueError):
            c["b"] = "f"
        c.auto_add_on()
        c["b"] = "f"
        self.assertTrue(arr_eq(c._fa, FastArray([4, 4, 4, 3, 4])))

    def test_setitem_array_slice(self):
        c = Categorical(["a", "a", "b", "c", "a"])
        c[:2] = "b"
        self.assertTrue(arr_eq(c._fa, FastArray([2, 2, 2, 3, 1])))
        with self.assertRaises(ValueError):
            c[:2] = "f"
        c.auto_add_on()
        c[:2] = "f"
        self.assertTrue(arr_eq(c._fa, FastArray([4, 4, 2, 3, 1])))

    def test_setitem_mapping_bool(self):
        idx = FastArray([1, 1, 2, 3, 1])
        codes = {"a": 1, "b": 2, "c": 3}
        c = Categorical(idx, codes)
        b = logical(arange(5) % 2)
        c[b] = "b"
        self.assertTrue(arr_eq(c._fa, FastArray([1, 2, 2, 2, 1])))
        with self.assertRaises(ValueError):
            c[b] = "f"
        c.mapping_add(4, "f")
        c[b] = "f"
        self.assertTrue(arr_eq(c._fa, FastArray([1, 4, 2, 4, 1])))

    def test_setitem_mapping_singleint(self):
        idx = FastArray([1, 1, 2, 3, 1])
        codes = {"a": 1, "b": 2, "c": 3}
        c = Categorical(idx, codes)
        c[1] = "b"
        c[3] = "b"
        self.assertTrue(arr_eq(c._fa, FastArray([1, 2, 2, 2, 1])))
        with self.assertRaises(ValueError):
            c[1] = "f"
        c.mapping_add(4, "f")
        c[1] = "f"
        c[3] = "f"
        self.assertTrue(arr_eq(c._fa, FastArray([1, 4, 2, 4, 1])))

    def test_setitem_mapping_index(self):
        idx = FastArray([1, 1, 2, 3, 1])
        codes = {"a": 1, "b": 2, "c": 3}
        c = Categorical(idx, codes)
        b = bool_to_fancy(logical(arange(5) % 2))
        c[b] = "b"
        self.assertTrue(arr_eq(c._fa, FastArray([1, 2, 2, 2, 1])))
        with self.assertRaises(ValueError):
            c[b] = "f"
        c.mapping_add(4, "f")
        c[b] = "f"
        self.assertTrue(arr_eq(c._fa, FastArray([1, 4, 2, 4, 1])))

    def test_setitem_mapping_string(self):
        idx = FastArray([1, 1, 2, 3, 1])
        codes = {"a": 1, "b": 2, "c": 3}
        c = Categorical(idx, codes)
        c["a"] = "b"
        self.assertTrue(arr_eq(c._fa, FastArray([2, 2, 2, 3, 2])))
        with self.assertRaises(ValueError):
            c["b"] = "f"
        c.mapping_add(4, "f")
        c["b"] = "f"
        self.assertTrue(arr_eq(c._fa, FastArray([4, 4, 4, 3, 4])))

    def test_setitem_numeric_bool(self):
        c = Categorical([1.23, 1.23, 4.56, 7.89, 1.23])
        b = logical(arange(5) % 2)
        c[b] = 4.56
        self.assertTrue(arr_eq(c._fa, FastArray([1, 2, 2, 2, 1])))
        with self.assertRaises(ValueError):
            c[b] = 10.1112
        c.auto_add_on()
        c[b] = 10.1112
        self.assertTrue(arr_eq(c._fa, FastArray([1, 4, 2, 4, 1])))

    def test_setitem_numeric_singleint(self):
        c = Categorical([1.23, 1.23, 4.56, 7.89, 1.23])
        c[1] = 4.56
        c[3] = 4.56
        self.assertTrue(arr_eq(c._fa, FastArray([1, 2, 2, 2, 1])))
        with self.assertRaises(ValueError):
            c[1] = 10.1112
        c.auto_add_on()
        c[1] = 10.1112
        c[3] = 10.1112
        self.assertTrue(arr_eq(c._fa, FastArray([1, 4, 2, 4, 1])))

    def test_setitem_numeric_index(self):
        c = Categorical([1.23, 1.23, 4.56, 7.89, 1.23])
        b = bool_to_fancy(logical(arange(5) % 2))
        c[b] = 4.56
        self.assertTrue(arr_eq(c._fa, FastArray([1, 2, 2, 2, 1])))
        with self.assertRaises(ValueError):
            c[b] = 10.1112
        c.auto_add_on()
        c[b] = 10.1112
        self.assertTrue(arr_eq(c._fa, FastArray([1, 4, 2, 4, 1])))

    def test_setitem_numeric_string(self):
        c = Categorical([1.23, 1.23, 4.56, 7.89, 1.23])
        c[1.23] = 4.56
        self.assertTrue(arr_eq(c._fa, FastArray([2, 2, 2, 3, 2])))
        with self.assertRaises(ValueError):
            c[4.56] = 10.1112
        c.auto_add_on()
        c[4.56] = 10.1112
        self.assertTrue(arr_eq(c._fa, FastArray([4, 4, 4, 3, 4])))

    def test_setitem_filter(self):
        data = Dataset({"First": Cat("A B A B A B".split())})
        data.Second = Cat("B A B A B A".split())
        data.Third = data.First.copy()
        f = data.Third == "A"
        data.First[0] = 2
        self.assertFalse(arr_eq(data.First._fa, data.Third._fa))
        data.First[f] = data.Second[f]
        self.assertFalse(arr_eq(data.First._fa, data.Third._fa))
        self.assertTrue(arr_eq(data.First._fa, FastArray([2, 2, 2, 2, 2, 2])))


if __name__ == "__main__":
    tester = unittest.main()
