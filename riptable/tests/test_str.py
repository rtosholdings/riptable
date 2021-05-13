import math
from typing import Tuple

import pytest
parametrize = pytest.mark.parametrize

import riptable as rt
from riptable import *


from numpy.testing import assert_array_equal
from ..testing.array_assert import assert_array_or_cat_equal



SYMBOLS = ['AAPL', 'AMZN', 'FB', 'GOOG', 'IBM']
PARALLEL_MULTIPLIER = 2000
NB_PARALLEL_SYMBOLS = SYMBOLS * PARALLEL_MULTIPLIER
assert len(NB_PARALLEL_SYMBOLS) >= FAString._APPLY_PARALLEL_THRESHOLD


_top_25_US_cities_by_population_2019_state_names = [
    'New York', 'California', 'Illinois', 'Texas', 'Arizona',
    'Pennsylvania', 'Texas', 'California', 'Texas', 'California',
    'Texas', 'Florida', 'Texas', 'Ohio', 'North Carolina',
    'California', 'Indiana', 'Washington', 'Colorado', 'District of Columbia',
    'Massachusetts', 'Texas', 'Tennessee', 'Michigan', 'Oklahoma',
]
"""
The state names (not abbreviations) of the top 25 US cities by population
in 2019 according to the US Census Bureau. Used to provide test data for
unit tests of string functions.
https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population
"""

def _tile_array(arr: np.ndarray, tile_count: int) -> np.ndarray:
    return arr.tile(tile_count) if hasattr(arr, 'tile') else np.tile(arr, tile_count)

def _make_parallelizable_array(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    # assumes C-contiguous layout here
    rowcount = arr.shape[0]
    tile_count = int(math.ceil((FAString._APPLY_PARALLEL_THRESHOLD + 1) / rowcount))
    tiled_arr = _tile_array(arr, tile_count)
    return tiled_arr, tile_count


class TestFAString:
    """
    Unit tests for the FAString class (via the ``FastArray.str`` accessor).
    """

    # NOTE: When implementing new unit tests (e.g. for new additions to FAString), make sure to cover
    #       all *combinations* of the following * each applicable input to the function:
    #           * array type: np.ndarray, FastArray, Categorical
    #           * encoding: ascii (bytes), Unicode (str)
    #           * parallelism: serial, parallel (based on the array length and FAString._APPLY_PARALLEL_THRESHOLD).
    #           * empty vs. non-empty array
    #               * for Categorical, this could mean either or both of the following:
    #                   * the Categorical has length zero (e.g. len(my_cat) == 0).
    #                   * the Categorical has no categories, possibly because they were all filtered out.

    cat_symbol = Cat(np.tile(np.arange(len(SYMBOLS) + 1), 3), SYMBOLS)


    @parametrize('position', [0, 1, -1, -2, 3])
    def test_char(self, position):
        result = FAString(SYMBOLS).char(position)
        expected = [s[position] if position < len(s) else '' for s in SYMBOLS]
        assert result.tolist() == expected

    @parametrize('position', [0, 1, -1, -2])
    def test_char_cat(self, position):
        result = self.cat_symbol.str.char(position)
        expected = Categorical(self.cat_symbol.ikey, [s[position] for s in SYMBOLS], base_index=1)
        assert_array_or_cat_equal(expected, result, relaxed_cat_check=True)
        # check categories are unique
        assert len(set(result.category_array)) == len(result.category_array)

    def test_char_array_position(self):
        position = [-1, 2, 0, 1, 2]
        result = FAString(SYMBOLS).char(position)
        expected = [s[pos] for s, pos in zip(SYMBOLS, position)]
        assert result.tolist() == expected

    @parametrize('position', [
        -3, 6, [0, 0, 0, 0, -5]
    ])
    def test_char_failure(self, position):
        with pytest.raises(ValueError, match='Position -?\d out of bounds'):
            FAString(SYMBOLS).char(position)

    @parametrize("str2, expected", [
        ('A', [True, True, False, False, False]),
        ('AA', [True, False, False, False, False]),
        ('', [True] * 5),
        ('AAA', [False] * 5),
        ('AAPL', [True] + [False] * 4)
    ])
    def test_contains(self, str2, expected):
        result = FAString(SYMBOLS).contains(str2)
        assert_array_equal(result, expected)

        result = FAString(NB_PARALLEL_SYMBOLS).contains(str2)
        assert_array_equal(result, expected * PARALLEL_MULTIPLIER)

    @parametrize("str2, expected", [
        ('bb', [False, False, True]),
        ('ba', [False, True, False]),
    ])
    def test_endswith(self, str2, expected):
        # TODO: Expand test to include when the underlying array is a Categorical.
        #       Also test parallel case.
        result = FAString(['abab', 'ababa', 'abababb']).endswith(str2)
        assert_array_equal(result, expected)

    def test_lower(self):
        result = FAString(SYMBOLS).lower
        assert (result.tolist() == [s.lower() for s in SYMBOLS])

    def test_lower_cat(self):
        result = self.cat_symbol.str.lower
        expected = Cat(self.cat_symbol.ikey, [s.lower() for s in SYMBOLS])
        assert_array_or_cat_equal(result, expected, relaxed_cat_check=True)

    regex_match_test_cases = parametrize('regex, expected', [
        ('.', [True] * 5),
        ('\.', [False] * 5),
        ('A', [True, True, False, False, False]),
        ('[A|B]', [True, True, True, False, True]),
        ('B$', [False, False, True, False, False]),
    ])

    @regex_match_test_cases
    def test_regex_match(self, regex, expected):
        fa = FA(SYMBOLS)
        assert_array_equal(fa.str.regex_match(regex), expected)

    @regex_match_test_cases
    def test_regex_match_cat(self, regex, expected):
        cat = Cat(SYMBOLS * 2)  # introduce duplicity to test ikey properly
        assert_array_equal(cat.str.regex_match(regex), expected * 2)

    def test_removetrailing_empty(self) -> None:
        arr = rt.FA([], dtype='S11')  # empty array
        result = arr.str.removetrailing()
        assert_array_equal(arr, result)

    @pytest.mark.parametrize('strings', [
        pytest.param(
            [(x + (' ' * (((len(x) * 17) + idx) % 3)) if idx % 3 != 0 else x) for idx, x in enumerate(_top_25_US_cities_by_population_2019_state_names)],
            id="modified US state names"
        )
    ])
    @pytest.mark.parametrize('arr_type_factory', [
        pytest.param(rt.FastArray, id='FastArray'),
        pytest.param(rt.Categorical, id='Categorical')
    ])
    @pytest.mark.parametrize('unicode', [pytest.param(False, id='ascii'), pytest.param(True, id='unicode')])
    @pytest.mark.parametrize('parallel', [pytest.param(False, id='serial'), pytest.param(True, id='parallel')])
    @pytest.mark.parametrize('chars', [pytest.param(' ', id='space'), 's'])
    def test_removetrailing(self, strings: list, arr_type_factory: callable, unicode: bool, parallel: bool, chars) -> None:
        dtype_str = '<U' if unicode else '|S'
        if not unicode:
            strings = [x.encode() for x in strings]
        ndarray = np.array(strings, dtype=dtype_str)
        arr = arr_type_factory(ndarray)
        arr, tile_count = (arr, None) if not parallel else _make_parallelizable_array(arr)

        # Perform the operation on the string array.
        result = arr.str.removetrailing(remove=ord(chars))

        # Perform the equivalent operation in pure Python.
        chars = chars if unicode else chars.encode()
        expected_strings = [x.strip(chars) for x in strings]
        expected_ndarray = np.array(expected_strings, dtype=dtype_str)
        expected = arr_type_factory(expected_ndarray)
        if tile_count is not None:
            expected = _tile_array(expected, tile_count)

        # Verify the output matches the expected result.
        assert_array_or_cat_equal(
            result, expected,
            # TEMP: The function doesn't shrink the dtype of the category array in
            #       an output categorical. So the dtype of the output cat will be the
            #       same as that of the output cat, even if trailing character(s) have
            #       been removed and the strings could fit in a smaller dtype.
            exact_dtype_match=False
        )

    #
    # TODO: reverse tests
    #

    #
    # TODO: reverse_inplace tests
    #

    def test_startswith(self):
        arrsize = 200
        symbol = Cat(1 + arange(arrsize) % len(SYMBOLS), SYMBOLS)
        assert_array_equal(symbol.expand_array.str.startswith('AAPL'), symbol.str.startswith(
            'AAPL'
        ))

    def test_startswith_cat(self):
        arrsize = 200
        symbol = Cat(1 + arange(arrsize) % len(SYMBOLS), SYMBOLS)
        assert_array_equal(symbol.expand_array.str.startswith('AAPL'), symbol.str.startswith(
            'AAPL'
        ))

    def test_startswith_cat_filtered(self):
        assert_array_equal(self.cat_symbol.expand_array.str.startswith('IBM'), self.cat_symbol.str.startswith(
            'IBM'
        ))

    def test_strlen_cat(self):
        result = self.cat_symbol.str.strlen
        inv = rt.INVALID_DICT[np.dtype(result.dtype).num]
        expected = rt.FA([inv, 4, 4, 2, 4, 3], dtype=result.dtype).tile(3)
        assert_array_equal(result, expected)

    def test_strlen_parallel(self):
        result = rt.FAString(NB_PARALLEL_SYMBOLS).strlen
        expected = rt.FA([4, 4, 2, 4, 3], dtype=result.dtype).tile(PARALLEL_MULTIPLIER)
        assert_array_equal(result, expected)

    def test_strpbrk_cat(self):
        result = self.cat_symbol.str.strpbrk('PZG')
        inv = rt.INVALID_DICT[np.dtype(result.dtype).num]
        expected = rt.FA([inv, 2, 2, -1, 0, -1], dtype=result.dtype).tile(3)
        assert_array_equal(result, expected)

    @parametrize("str2, expected", [
        ('A', [0, 0, -1, -1, -1]),
        ('AA', [0, -1, -1, -1, -1]),
        ('AAPL', [0, -1, -1, -1, -1]),
        ('', [0] * 5),
        ('AAA', [-1] * 5),
        ('B', [-1, -1, 1, -1, 1])
    ])
    def test_strstr(self, str2, expected):
        result = FAString(SYMBOLS).strstr(str2)
        assert_array_equal(result, expected)

        result = FAString(NB_PARALLEL_SYMBOLS).strstr(str2)
        assert_array_equal(result, expected * PARALLEL_MULTIPLIER)

    def test_strstr_cat(self):
        result = self.cat_symbol.str.strstr('A')
        inv = rt.INVALID_DICT[np.dtype(result.dtype).num]
        expected = rt.FA([inv, 0, 0, -1, -1, -1], dtype=result.dtype).tile(3)
        assert_array_equal(result, expected)

    substr_test_cases = parametrize("start_stop", [
        (0, 1),
        (0, 2),
        (1, 3),
        (1, -1),
        (-2, 3),
        (2,),
        (-1,),
        (-3, -1),
        (-1, 1),
    ])

    @substr_test_cases
    def test_substr(self, start_stop):
        expected = [s[slice(*start_stop)] for s in SYMBOLS]
        result = FAString(SYMBOLS).substr(*start_stop)
        assert (expected == result.tolist())

    @substr_test_cases
    def test_substr_bytes(self, start_stop):
        expected = [s[slice(*start_stop)] for s in SYMBOLS]
        result = FAString(FastArray(SYMBOLS)).substr(*start_stop)
        assert_array_equal(FastArray(expected), result)

    @substr_test_cases
    def test_substr_cat(self, start_stop):
        result = self.cat_symbol.str.substr(*start_stop)
        expected = [s[slice(*start_stop)] for s in SYMBOLS]
        expected = Categorical(self.cat_symbol.ikey, expected, base_index=1)
        assert_array_or_cat_equal(expected, result, relaxed_cat_check=True)
        # check categories are unique
        assert len(set(result.category_array)) == len(result.category_array)

    def test_upper(self):
        result = FAString(SYMBOLS).upper
        assert (result.tolist() == [s.upper() for s in SYMBOLS])

    def test_upper_cat(self):
        result = self.cat_symbol.str.upper
        expected = Cat(self.cat_symbol.ikey, [s.upper() for s in SYMBOLS])
        assert_array_or_cat_equal(result, expected, relaxed_cat_check=True)

    #
    # TODO: upper_inplace tests
    #

