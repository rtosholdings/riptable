import pytest
from enum import IntEnum
from riptable import *
from riptable.tests.utils import LikertDecision


class TestCategoricalChangeCategories:
    def test_add_mapping(self):
        codes = [44, 133, 75, 1, 144, 144, 1, 44, 75]
        c = Categorical(codes, LikertDecision)
        c.mapping_add(50, 'Fifty')
        new_mapping = c.category_mapping
        result = new_mapping.get(50, False)
        assert result == 'Fifty'

        with pytest.raises(ValueError):
            c.mapping_add(1, 'One')

        with pytest.raises(TypeError):
            c.mapping_add('notanint', 'Two')

        codes = [33, 133, 75, 1, 144, 144, 1, 33, 75]
        c = Categorical(codes, LikertDecision)
        assert c[0] == '!<33>'
        c.mapping_add(33, 'ThirtyThree')
        assert c[0] == 'ThirtyThree'

    def test_remove_mapping(self):
        codes = [44, 133, 75, 1, 144, 144, 1, 44, 75]
        c = Categorical(codes, LikertDecision)
        c.mapping_remove(44)
        assert c[0] == '!<44>'
        assert c[7] == '!<44>'
        new_mapping = c.category_mapping
        removed = new_mapping.get(44, True)
        assert removed

        with pytest.raises(ValueError):
            c.mapping_remove(33)

    def test_replace_mapping(self):
        codes = [44, 133, 75, 1, 144, 144, 1, 44, 75]
        c = Categorical(codes, LikertDecision)
        c.mapping_replace(1, 'One')
        assert c[3] == 'One'
        new_mapping = c.category_mapping
        assert new_mapping[1] == 'One'

        with pytest.raises(ValueError):
            c.mapping_replace(50, 'Fifty')

    def test_new_mapping(self):
        class MappingNew(IntEnum):
            FourFour = 44
            OneThreeThree = 133
            SevenFive = 75
            ThreeSix = 36
            OneFourFour = 144

        codes = [44, 133, 75, 1, 144, 144, 1, 44, 75]
        c = Categorical(codes, LikertDecision)
        assert c[3] == 'StronglyDisagree'
        c.mapping_new(MappingNew)
        assert c[0] == 'FourFour'
        assert c[3] == '!<1>'

        with pytest.raises(TypeError):
            c.mapping_new(np.arange(10))

    # TODO: add more tests for different types of stringlist categoricals
    def test_category_add(self):
        c = Categorical(['b', 'b', 'c', 'd', 'e', 'b', 'c'])
        orig_fa = c._fa.copy()
        c.category_add('a')
        first_match = bool(np.all(orig_fa == c._fa))
        assert first_match
        new_cat_array = c.category_array
        assert len(new_cat_array) == 5
        assert new_cat_array[-1] == b'a'

        with pytest.raises(ValueError):
            c.category_add('a')

    def test_category_remove(self):
        c = Categorical(['b', 'b', 'c', 'd', 'e', 'b', 'c'])
        c.category_remove('b')
        c.filtered_set_name('Inv')
        assert c[1] == 'Inv'
        assert c._fa[1] == 0
        assert c._fa[2] == 1
        assert len(c.category_array) == 3

        with pytest.raises(ValueError):
            c.category_remove('b')

    def test_category_replace(self):
        c = Categorical(['b', 'b', 'c', 'd', 'e', 'b', 'c'])
        c.category_replace('b', 'a')
        assert c.category_array[0] == b'a'
        assert c[0] == 'a'
        assert not (b'b' in c.category_array)
        c.category_replace('a', 'd')
        assert len(c.category_array) == 4
        assert not c._ordered

        with pytest.raises(ValueError):
            c.category_replace('z', 'a')
