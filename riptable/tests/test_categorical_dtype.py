import pytest
import pandas as pd

from riptable import *


sorted_codes = FastArray(
    [
        2,
        4,
        4,
        3,
        2,
        1,
        3,
        2,
        0,
        1,
        3,
        4,
        2,
        0,
        4,
        3,
        1,
        0,
        1,
        2,
        3,
        1,
        4,
        2,
        2,
        3,
        4,
        2,
        0,
        2,
    ]
)
str_fa = FastArray(
    [
        'c',
        'e',
        'e',
        'd',
        'c',
        'b',
        'd',
        'c',
        'a',
        'b',
        'd',
        'e',
        'c',
        'a',
        'e',
        'd',
        'b',
        'a',
        'b',
        'c',
        'd',
        'b',
        'e',
        'c',
        'c',
        'd',
        'e',
        'c',
        'a',
        'c',
    ]
)
complete_unique_cats = FastArray(['a', 'b', 'c', 'd', 'e'])
big_cats = FastArray(['string' + str(i) for i in range(2000)])
matlab_codes = (sorted_codes + 1).astype(np.float32)


# TODO pytest parameterize expectations for dtypes
class TestCategoricalDtype:
    def test_codes(self):
        c = Categorical(sorted_codes.astype(np.int16), complete_unique_cats)
        assert c._fa.dtype == np.int16

        c = Categorical(sorted_codes.astype(np.uint64), complete_unique_cats)
        assert c._fa.dtype == np.int8

        c = Categorical(sorted_codes, complete_unique_cats, dtype=np.int64)
        assert c._fa.dtype == np.int64

        c = Categorical(sorted_codes, big_cats, dtype=np.int8)
        assert c._fa.dtype == np.int16

    def test_matlab(self):
        c = Categorical(matlab_codes, complete_unique_cats, from_matlab=True)
        assert c._fa.dtype == np.int8

        c = Categorical(
            matlab_codes, complete_unique_cats, from_matlab=True, dtype=np.int64
        )
        assert c._fa.dtype == np.int64

    def test_strings(self):
        c = Categorical(str_fa)
        assert c._fa.dtype == np.int8

        c = Categorical(str_fa, dtype=np.int64)
        assert c._fa.dtype == np.int64

    def test_strings_cats(self):
        c = Categorical(str_fa, complete_unique_cats)
        assert c._fa.dtype == np.int8

        c = Categorical(str_fa, complete_unique_cats, dtype=np.int64)
        assert c._fa.dtype == np.int64

    def test_multikey(self):
        c = Categorical([str_fa, sorted_codes])
        assert c._fa.dtype == np.int8

        c = Categorical([str_fa, sorted_codes], dtype=np.int64)
        assert c._fa.dtype == np.int64

    def test_pandas(self):
        pdc = pd.Categorical(str_fa)
        c = Categorical(str_fa)
        assert c._fa.dtype == np.int8

        pdc._codes = pdc._codes.astype(np.int32)
        c = Categorical(pdc)
        assert c._fa.dtype == np.int8

        c = Categorical(pdc, dtype=np.int64)
        assert c._fa.dtype == np.int64

    def test_errors(self):
        with pytest.raises(TypeError):
            c = Categorical(str_fa, dtype=np.uint8)

        with pytest.warns(UserWarning):
            c = Categorical(sorted_codes, big_cats, dtype=np.int8)
