import unittest

from riptable import *


def arr_eq(a, b):
    return bool(np.all(a == b))


def cat_eq(c1, c2):
    # for single key
    # TODO: compare other attributes
    inst_eq = arr_eq(c1._fa, c2._fa)
    cats_eq = arr_eq(c1.category_array, c2.category_array)
    return inst_eq and cats_eq


class Flatten_Test(unittest.TestCase):
    # TODO pull nested Struct into testing data for reuse in tooling integration tests
    def test_nested(self):
        st1 = Struct(
            {
                'cat1': Categorical(['a', 'b', 'c', 'd']),
                'arr1': arange(5),
                'ds1': Dataset({'col_' + str(i): np.random.rand(5) for i in range(5)}),
                'arr2': arange(10),
                'test_string': 'this is my string',
                'array3': arange(30).astype(np.int8),
                'nested_struct1': Struct(
                    {
                        'set1': Dataset(
                            {'col_' + str(i): np.random.randint(5) for i in range(3)}
                        ),
                        'set2': Dataset(
                            {
                                'col_' + str(i): np.random.choice(['a', 'b', 'c'], 10)
                                for i in range(2)
                            }
                        ),
                        'nested_2': Struct(
                            {
                                'c1': Categorical(arange(5)),
                                'c2': Categorical(['aaa', 'bbbb', 'cccc']),
                                'leaf_dataset': Dataset(
                                    {'col_' + str(i): arange(5) for i in range(3)}
                                ),
                                'array4': arange(20).astype(np.uint64),
                                'dtn1': DateTimeNano.random(20),
                                'nested_3': Dataset(
                                    {
                                        'CAT': Categorical(
                                            np.random.choice(['X', 'Y', 'Z'], 17)
                                        ),
                                        'DTN': DateTimeNano.random(17),
                                        'DATE': Date(
                                            np.random.randint(15000, 20000, 17)
                                        ),
                                        'TSPAN': TimeSpan(
                                            np.random.randint(
                                                0,
                                                1_000_000_000 * 60 * 60 * 24,
                                                17,
                                                dtype=np.int64,
                                            )
                                        ),
                                        'DSPAN': DateSpan(
                                            np.random.randint(0, 365, 17)
                                        ),
                                    }
                                ),
                            }
                        ),
                        'set3': Dataset(
                            {
                                'col_' + str(i): np.random.choice([True, False], 10)
                                for i in range(4)
                            }
                        ),
                    }
                ),
                'ds2': Dataset(
                    {'heading_' + str(i): np.random.rand(5) for i in range(3)}
                ),
                'int1': 5,
                'float1': 7.0,
                'cat2': Categorical(['a', 'b', 'c', 'd']),
            }
        )

        st_flattened = st1.flatten()
        st2 = st_flattened.flatten_undo()

        # top level keys are the same, in the same order
        self.assertTrue(arr_eq(list(st1), list(st2)))

        # single datasets of normal arrays
        self.assertTrue(isinstance(st2.ds1, Dataset))
        self.assertTrue(st1.ds1.equals(st2.ds1))
        self.assertTrue(isinstance(st2.ds2, Dataset))
        self.assertTrue(st1.ds2.equals(st2.ds2))

        # top level arrays
        self.assertTrue(isinstance(st2.cat1, Categorical))
        self.assertTrue(cat_eq(st1.cat1, st2.cat1))
        self.assertTrue(arr_eq(st1.arr1, st2.arr1))
        self.assertTrue(arr_eq(st1.arr2, st2.arr2))
        self.assertTrue(isinstance(st2.test_string, str))
        self.assertEqual(st1.test_string, st2.test_string)
        self.assertTrue(arr_eq(st1.array3, st2.array3))
        self.assertTrue(isinstance(st2.cat2, Categorical))
        self.assertTrue(cat_eq(st1.cat2, st2.cat2))

        # top level scalars
        self.assertTrue(isinstance(st2.int1, (int, np.integer)))
        self.assertEqual(st1.int1, st2.int1)
        self.assertTrue(isinstance(st2.float1, (float, np.floating)))
        self.assertEqual(st1.float1, st2.float1)

        # main nested struct
        self.assertTrue(isinstance(st2.nested_struct1, Struct))
        self.assertTrue(arr_eq(list(st1.nested_struct1), list(st2.nested_struct1)))

        # dataset of ints
        self.assertTrue(isinstance(st2.nested_struct1.set1, Dataset))
        self.assertTrue(st1.nested_struct1.set1.equals(st2.nested_struct1.set1))

        # dataset of bytes
        self.assertTrue(isinstance(st2.nested_struct1.set2, Dataset))
        self.assertTrue(st1.nested_struct1.set2.equals(st2.nested_struct1.set2))

        # dataset of bools
        self.assertTrue(isinstance(st2.nested_struct1.set2, Dataset))
        self.assertTrue(st1.nested_struct1.set2.equals(st2.nested_struct1.set2))

        # another nested level
        nest1 = st1.nested_struct1.nested_2
        nest2 = st2.nested_struct1.nested_2
        self.assertTrue(isinstance(nest2, Struct))
        self.assertTrue(arr_eq(list(nest1), list(nest2)))

        # arrays
        self.assertTrue(cat_eq(nest1.c1, nest2.c1))
        self.assertTrue(cat_eq(nest1.c2, nest2.c2))
        self.assertTrue(arr_eq(nest1.array4, nest2.array4))
        self.assertTrue(isinstance(nest2.dtn1, DateTimeNano))
        self.assertTrue(arr_eq(nest1.dtn1, nest2.dtn1))

        # datasets
        self.assertTrue(isinstance(nest2.leaf_dataset, Dataset))
        self.assertTrue(nest1.leaf_dataset.equals(nest2.leaf_dataset))

        # final nesting, all special types
        DS1 = nest1.nested_3
        DS2 = nest2.nested_3
        self.assertTrue(isinstance(DS2, Dataset))
        self.assertTrue(arr_eq(list(DS1), list(DS2)))

        self.assertTrue(cat_eq(DS1.CAT, DS2.CAT))
        self.assertTrue(isinstance(DS2.DTN, DateTimeNano))
        self.assertTrue(arr_eq(DS1.DTN, DS2.DTN))

        # TODO: compare tz info, etc.
        self.assertTrue(isinstance(DS2.DATE, Date))
        self.assertTrue(arr_eq(DS1.DATE, DS2.DATE))
        self.assertTrue(isinstance(DS2.TSPAN, TimeSpan))
        self.assertTrue(arr_eq(DS1.TSPAN, DS2.TSPAN))
        self.assertTrue(isinstance(DS2.DSPAN, DateSpan))
        self.assertTrue(arr_eq(DS1.DSPAN, DS2.DSPAN))

    # def test_separator(self):
    #    ds = Dataset({'colA':arange(3), 'colB':arange(10,13)})
    #    st = Struct({'arr':arange(10), 'ds1':ds})
    #    flat = st.flatten()
    #    flatcols = ['arr', 'ds1/', 'ds1/colA', 'ds1/colB']
    #    self.assertTrue(arr_eq(list(flat), flatcols))


if __name__ == "__main__":
    tester = unittest.main()
