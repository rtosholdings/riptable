from riptable import *


class TestCategoricalBaseIndex:
    def testbase_index_from_codes(self):
        base_0_codes = FastArray([0, 0, 1, 2, 1])
        base_1_codes = FastArray([1, 1, 2, 3, 1])
        cats = FastArray(['a', 'b', 'c'])

        # default this constructor to base 0 if not set
        # c = Categorical(base_0_codes, cats)
        # self.assertEqual(c.base_index, 0)
        # self.assertEqual(c._fa[0], 0)
        # self.assertEqual(c[2], 'b')

        # still use specified base index
        c = Categorical(base_0_codes, cats, base_index=0)
        assert c.base_index == 0
        assert c._fa[4] == 1
        assert c[3] == 'c'

        # even if base index is 1
        c = Categorical(base_1_codes, cats, base_index=1)
        assert c.base_index == 1
        assert c._fa[3] == 3
        assert c[1] == 'a'

    # ------------------------------------------------------

    def test_strings(self):
        c = Categorical(['b', 'c', 'a', 'a', 'b'])
        assert c._fa[0] == 2
        assert c.base_index == 1

        c = Categorical(['b', 'c', 'a', 'a', 'b'], base_index=0)
