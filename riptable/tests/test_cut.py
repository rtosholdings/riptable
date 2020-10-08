import unittest
import numpy as np

from riptable import cut, qcut, FA, Categorical, FastArray, arange


class Cut_Test(unittest.TestCase):
    '''
    TODO: add more tests for different types
    also include string types
    '''

    def test_cut(self):
        c = cut(arange(10), 3)
        self.assertTrue(sum(c._np - FA([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])) == 0)

        c = cut(arange(10.0), 3)
        self.assertTrue(sum(c._np - FA([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])) == 0)

        c = cut(arange(11), 3)
        self.assertTrue(sum(c._np - FA([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])) == 0)

        c = cut(FA([2, 4, 6, 8, 10]), FA([0, 2, 4, 6, 8, 10]))
        self.assertTrue(sum(c._np - FA([1, 2, 3, 4, 5])) == 0)

        c = cut(
            FA([2, 4, 6, 8, 10]),
            FA([0, 2, 4, 6, 8, 10]),
            labels=['a', 'b', 'c', 'd', 'e'],
        )
        self.assertTrue(sum(c._np - FA([1, 2, 3, 4, 5])) == 0)

        a = np.array([1, 7, 5, 4, 6, 3])
        l = FA([b'1.0->3.0', b'3.0->5.0', b'5.0->7.0'])

        c = cut(a, 3)
        self.assertIsInstance(c, Categorical)
        self.assertTrue(sum(c._np - FA([1, 3, 2, 2, 3, 1])) == 0)
        self.assertTrue((c.category_array == l).all())

        c = cut(a, 3, labels=True)
        self.assertIsInstance(c, Categorical)
        self.assertTrue(sum(c._np - FA([1, 3, 2, 2, 3, 1])) == 0)
        self.assertTrue((c.category_array == l).all())

        c = cut(a, 3, labels=None)
        self.assertIsInstance(c, Categorical)
        self.assertTrue(sum(c._np - FA([1, 3, 2, 2, 3, 1])) == 0)
        self.assertTrue((c.category_array == l).all())

        c = cut(a, 3, labels=False)
        self.assertIsInstance(c, FastArray)
        self.assertTrue(sum(c._np - FA([1, 3, 2, 2, 3, 1])) == 0)

        c, b = cut(a, 3, retbins=True)
        self.assertIsInstance(c, Categorical)
        self.assertIsInstance(b, np.ndarray)
        self.assertTrue(sum(c._np - FA([1, 3, 2, 2, 3, 1])) == 0)
        self.assertTrue((c.category_array == l).all())
        self.assertTrue(sum(b - FA([1.0, 3.0, 5.0, 7.0])) == 0)

        l = ["bad", "medium", "good"]
        c = cut(a, 3, labels=l)
        self.assertIsInstance(c, Categorical)
        self.assertTrue(sum(c._np - FA([1, 3, 2, 2, 3, 1])) == 0)
        self.assertTrue((c.category_array == l).all())

        # contiguous test
        x = arange(4).reshape(2, 2)
        knots = [-0.5, 0.5, 1.5, 2.5, 3.5]
        c = cut(x[:, 1], knots)
        l = FastArray([b'-0.5->0.5', b'0.5->1.5', b'1.5->2.5', b'2.5->3.5'])
        self.assertTrue((c.category_array == l).all())

        # inf upcast test
        x = np.array([0, 1, 10, 100, 5])
        knots = [-np.inf, 2, 11, 50, np.inf]
        c = cut(x, knots)
        self.assertTrue((c._fa == FA([1,1,2,4,2])).all())

    def test_qcut(self):
        c = qcut(arange(10), 3)
        self.assertTrue(sum(c._np - FA([2, 2, 2, 2, 3, 3, 3, 4, 4, 4])) == 0)

        c = qcut(arange(11), 3)
        self.assertTrue(sum(c._np - FA([2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4])) == 0)

        c = qcut(range(5), 3, labels=["good", "medium", "bad"])
        self.assertTrue(sum(c._np - FA([2, 2, 3, 4, 4])) == 0)

        c = qcut(arange(100.0), [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        self.assertTrue(c._np[0] == 1)
        self.assertTrue(c._np[5] == 2)
        self.assertTrue(c._np[94] == 7)
        self.assertTrue(c._np[95] == 1)

        c = qcut(arange(100.0), [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        self.assertTrue(c._np[0] == 2)
        self.assertTrue(c._np[5] == 2)
        self.assertTrue(c._np[94] == 7)
        self.assertTrue(c._np[95] == 1)

        c = qcut(range(5), 4)
        self.assertIsInstance(c, Categorical)
        self.assertTrue(sum(c._np - FA([2, 2, 3, 4, 5])) == 0)
        self.assertTrue(
            (
                c.category_array
                == [b'Clipped', b'0.0->1.0', b'1.0->2.0', b'2.0->3.0', b'3.0->4.0']
            ).all()
        )

        c = qcut(range(5), 4, labels=True)
        self.assertIsInstance(c, Categorical)
        self.assertTrue(sum(c._np - FA([2, 2, 3, 4, 5])) == 0)
        self.assertTrue(
            (
                c.category_array
                == [b'Clipped', b'0.0->1.0', b'1.0->2.0', b'2.0->3.0', b'3.0->4.0']
            ).all()
        )

        c = qcut(range(5), 4, labels=None)
        self.assertIsInstance(c, Categorical)
        self.assertTrue(sum(c._np - FA([2, 2, 3, 4, 5])) == 0)
        self.assertTrue(
            (
                c.category_array
                == [b'Clipped', b'0.0->1.0', b'1.0->2.0', b'2.0->3.0', b'3.0->4.0']
            ).all()
        )

        c = qcut(range(5), 3, labels=["good", "medium", "bad"])
        self.assertIsInstance(c, Categorical)
        self.assertTrue(sum(c._np - FA([2, 2, 3, 4, 4])) == 0)
        self.assertTrue(
            np.array(
                [
                    (h == t)
                    for (h, t) in zip(
                        c.expand_array.astype('U'),
                        ['good', 'good', 'medium', 'bad', 'bad'],
                    )
                ]
            ).all()
        )
        self.assertTrue(
            (c.category_array == [[b'Clipped', b'good', b'medium', b'bad']]).all()
        )

        c = qcut(range(5), 4, labels=False)
        self.assertIsInstance(c, FastArray)
        self.assertTrue(sum(c._np - FA([2, 2, 3, 4, 5])) == 0)

    def test_cut_errors(self):
        with self.assertRaises(ValueError):
            c = cut(
                FA([2, 4, 6, 8, 10]),
                FA([0, 2, 4, 6, 8, 10]),
                labels=['a', 'b', 'c', 'd', 'e', 'f'],
            )


if __name__ == "__main__":
    tester = unittest.main()
