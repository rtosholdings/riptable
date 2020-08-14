import unittest
import pytest

from numpy.testing import assert_array_equal
from riptable import *
from riptable.rt_enum import INVALID_DICT, TypeRegister
from riptable.rt_numpy import ismember, arange


# TODO: Use pytest.mark.parametrize() here to iterate over the different dtypes
#       rather than looping over them within the test function.
def test_ismember_nums():
    allowed_int = 'bhilqpBHILQP'
    allowed_float = 'fd'
    allowed_types = allowed_int + allowed_float
    # allowed_types = 'lqfd'

    a_nums_list = [1, 2, 3, 4, 5]
    a_nums_tup = tuple(a_nums_list)

    b_nums_list = [1, 2]
    b_nums_tup = tuple(b_nums_list)

    a_nums = (
        [a_nums_list, a_nums_tup]
        + [np.array(a_nums_list, dtype=np.dtype(i)) for i in allowed_types]
        + [FA(a_nums_list, dtype=np.dtype(i)) for i in allowed_types]
    )
    b_nums = (
        [b_nums_list, b_nums_tup]
        + [np.array(b_nums_list, dtype=np.dtype(i)) for i in allowed_types]
        + [FA(b_nums_list, dtype=np.dtype(i)) for i in allowed_types]
    )

    correct_bool = [True, True, False, False, False]
    correct_idx = [0, 1]

    for a in a_nums:
        adt = None
        if hasattr(a, 'dtype'):
            adt = a.dtype
        else:
            adt = type(a)
        for b in b_nums:
            bdt = None
            if hasattr(b, 'dtype'):
                bdt = b.dtype
            else:
                bdt = type(b)

            bool_arr, idx_arr = ismember(a, b)

            assert_array_equal(
                bool_arr,
                correct_bool,
                err_msg=f"Boolean return was incorrect for ismember with a type {adt} b type {bdt}",
            )
            assert_array_equal(
                idx_arr[:2],
                correct_idx,
                err_msg=f"Index was incorrect for ismember with a type {adt} b type {bdt}",
            )

            # TODO: Use rt.isnan here to compare idx_arr[2:] (or just idx_arr) to an expected array; then we can use assert_array_equal()
            for idx in idx_arr[2:]:
                assert (
                    idx == INVALID_DICT[idx_arr.dtype.num]
                ), f"Index element was not invalid at index {idx} for ismember with a type {adt} b type {bdt}"


def test_ismember_strings():
    correct_bool = [True, True, False, False, False]
    correct_idx = [0, 1]

    bytes_list_a = ['a', 'b', 'c', 'd', 'e']
    a_strings = [
        bytes_list_a,
        tuple(bytes_list_a),
        np.array(bytes_list_a),
        FA(bytes_list_a),
    ]
    # np.chararray(bytes_list_a) ]

    bytes_list_b = ['a', 'b']
    b_strings = [
        bytes_list_b,
        tuple(bytes_list_b),
        np.array(bytes_list_b),
        FA(bytes_list_b),
    ]
    # np.chararray(bytes_list_b) ]

    true_unicode_a = np.array(
        [u'a\u2082', u'b\u2082', u'c\u2082', u'd\u2082', u'e\u2082']
    )
    true_unicode_b = np.array([u'a\u2082', u'b\u2082'])

    # valid inputs, conversion happens when necessary
    for a in a_strings:
        adt = None
        if hasattr(a, 'dtype'):
            adt = a.dtype
        else:
            adt = type(a)
        for b in b_strings:
            bdt = None
            if hasattr(b, 'dtype'):
                bdt = b.dtype
            else:
                bdt = type(b)

            bool_arr, idx_arr = ismember(a, b)

            assert_array_equal(
                bool_arr,
                correct_bool,
                err_msg=f"Boolean return was incorrect for ismember with a type {adt} b type {bdt}",
            )
            assert_array_equal(
                idx_arr[:2],
                correct_idx,
                err_msg=f"Index was incorrect for ismember with a type {adt} b type {bdt}",
            )

            # TODO: Use rt.isnan here to compare idx_arr[2:] (or just idx_arr) to an expected array; then we can use assert_array_equal()
            for idx in idx_arr[2:]:
                assert (
                    idx == INVALID_DICT[idx_arr.dtype.num]
                ), f"Index element was not invalid at index {idx} for ismember with a type {adt} b type {bdt}"

    ## invalid first input
    # a = true_unicode_a
    # adt = "REAL unicode"
    # for b in b_strings:
    #    bdt = None
    #    if hasattr(b,'dtype'):
    #        bdt=b.dtype
    #    else:
    #        bdt = type(b)
    #    with pytest.raises(TypeError, msg=f"Failed to raise error between {adt} and {bdt} when unable to convert unicode."):
    #        bool_arr, idx_arr = ismember(a,b)

    ## invalid second input
    # b = true_unicode_b
    # bdt = "REAL unicode"
    # for a in a_strings:
    #    bdt = None
    #    if hasattr(a,'dtype'):
    #        adt=a.dtype
    #    else:
    #        adt = type(a)
    #    with pytest.raises(TypeError, msg=f"Failed to raise error between {adt} and {bdt} when unable to convert unicode."):
    #        bool_arr, idx_arr = ismember(a,b)

    ## both inputs invalid
    # a = true_unicode_a
    # b = true_unicode_b
    # with pytest.raises(TypeError, msg=f"Failed to raise error between {a.dtype} and {b.dtype} when unable to convert unicode."):
    #    bool_arr, idx_arr = ismember(a,b)


def test_ismember_categorical():
    for b_index_c in [0, 1]:
        for b_index_d in [0, 1]:

            # string values, both base indices
            c = TypeRegister.Categorical(
                np.random.choice(['a', 'b', 'c', 'd', 'e', 'f'], 15),
                base_index=b_index_c,
            )
            d = TypeRegister.Categorical(
                np.random.choice(['a', 'b', 'c'], 10), base_index=b_index_d
            )
            cs, ds = c.as_string_array, d.as_string_array

            b, f = ismember(c, d)
            bs, fs = ismember(cs, ds)
            assert_array_equal(b, bs)
            assert_array_equal(int8(f), fs)

            b, f = ismember(d, c)
            bs, fs = ismember(ds, cs)
            assert_array_equal(b, bs)
            assert_array_equal(int8(f), fs)

            # codes, string values, both base indices

        c = TypeRegister.Categorical(
            np.random.choice(['a', 'b', 'c', 'd', 'e', 'f'], 15), base_index=b_index_c
        )
        d = TypeRegister.Categorical(
            np.random.choice(['a', 'b', 'c'], 10), ['a', 'b', 'c'], base_index=1
        )
        cs, ds = c.as_string_array, d.as_string_array
        b, f = ismember(c, d)
        bs, fs = ismember(cs, ds)
        assert_array_equal(b, bs)
        assert_array_equal(int8(f), fs)

        b, f = ismember(d, c)
        bs, fs = ismember(ds, cs)
        assert_array_equal(b, bs)
        assert_array_equal(int8(f), fs)

    c = Categorical(np.random.choice(['a', 'b', 'c'], 15))
    with pytest.raises(TypeError):
        b, idx = ismember(c, arange(3))
    # TypeRegister.Categorical.TestIsMemberVerbose = True
    # c = TypeRegister.Categorical(np.random.choice(['a','b','c', 'd', 'e', 'f'],15), base_index=1)
    # _ = ismember(c, ['a','b'])
    # self.assertEqual("newcat", TypeRegister.Categorical._test_cat_ismember)


class Ismember_Test(unittest.TestCase):
    # TODO: Convert this test to a module-level, pytest-style test.
    def test_ismember_categorical_numeric(self):
        c = Categorical([1, 2, 3, 1, 2, 3, 1, 2, 4])
        f = FastArray([1, 2, 3], dtype=np.int64)
        b, idx = ismember(c, f)
        self.assertTrue(bool(np.all(b[:-1])))
        self.assertFalse(b[-1], False)
        self.assertTrue(bool(np.all(idx[:-1] == tile(FA([0, 1, 2]), 3)[:-1])))
        self.assertTrue(idx.isnan()[-1])
        f = FastArray(['a', 'b', 'c'])
        with pytest.raises(TypeError):
            b, idx = ismember(c, f)

        c = Categorical([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 4.0])
        f = FastArray([1, 2, 3], dtype=np.float64)
        b, idx = ismember(c, f)
        self.assertTrue(bool(np.all(b[:-1])))
        self.assertFalse(b[-1], False)
        self.assertTrue(bool(np.all(idx[:-1] == tile(FA([0, 1, 2]), 3)[:-1])))
        self.assertTrue(idx.isnan()[-1])
        f = FastArray(['a', 'b', 'c'])
        with pytest.raises(TypeError):
            b, idx = ismember(c, f)

        c = Categorical([np.random.choice(['a', 'b', 'c'], 10), arange(10)])
        with pytest.raises(TypeError):
            b, idx = ismember(c, f)


@pytest.mark.skip("RIP-364: This test is empty and needs to be implemented.")
def test_ismember_categorical_codes():
    pass
    # c_codes = np.random.randint(0,1000,1000)
    # d_codes = np.random.randint(0,1000,500)

    # for_cats = arange(5000,6000)
    # cats = (for_cats.astype('S'), True)
    # unicats = (for_cats.astype('U'), False)

    # for c1 in [cats, unicats]:
    #    c = Categorical(c_codes, c1[0], convert_string_to_bytes=c1[1])
    #    cs = c.as_string_array

    #    for c2 in [cats, unicats]:
    #        d = Categorical(d_codes, c2[0], convert_string_to_bytes=c2[1])
    #        ds = d.as_string_array

    #        b, f = ismember(c, d)
    #        bs, fs = ismember(cs, ds)
    #        self.assertTrue(bool(np.all(b==bs)))
    #        self.assertTrue(bool(np.all(int8(f)==fs)))

    #        b, f = ismember(d, c)
    #        bs, fs = ismember(ds, cs)
    #        self.assertTrue(bool(np.all(b==bs)))
    #        self.assertTrue(bool(np.all(int8(f)==fs)))


def test_ismember_index_casting():
    casting_dict = {
        5: np.dtype(np.int8),
        29_000: np.dtype(np.int16),
        60_000: np.dtype(np.int32),
    }
    for sz, correct_type in casting_dict.items():
        a = np.full(sz, b'a').view(FA)
        b = FA(['a', 'b'])
        _, idx = ismember(b, a)
        assert (
            idx.dtype == correct_type
        ), f"ismember with {sz} strings did not properly downcast index array to {correct_type}, got {idx.dtype} instead."


# TODO: RIP-364: This test doesn't have any assertions, so it's currently only checking whether the function excepts. Add assertions to check the ismember output for correctness."
def test_ismember_int_edges():
    # hit thresholds for a previous bug
    for a_size in [127, 129, 254, 256]:
        a = arange(a_size)
        for b_size in range(129):
            _, _ = ismember(a, arange(b_size))


# TODO: RIP-364: This test doesn't have any assertions, so it's currently only checking whether the function excepts. Add assertions to check the ismember output for correctness."
def test_ismember_diff_itemsize():
    a = FA([b'b', b'a', b'a', b'Inv', b'c', b'a', b'b'], dtype='S')
    b = FA([b'a', b'b', b'c'], dtype='S')
    bl, idx = ismember(a, b)
    correctbool = [True, True, True, False, True, True, True]
    correctidx = FA([1, 0, 0, int8.inv, 2, 0, 1], dtype=np.int8)
    boolmatch = bool(np.all(bl == [False, True, True]))
    idxmatch = bool(np.all(idx == [int8.inv, 1, 2]))


@pytest.mark.skip("RIP-364: This test is empty and needs to be implemented.")
def test_ismember_empty():
    """Test which checks how ismember handles empty input(s)."""
    pass


def test_ismember_multikey_errors():
    # types didn't match
    a_keys = [FA([1, 2, 3]), FA([1, 2, 3], dtype=np.float32)]
    b_keys = [FA([1, 2, 3]), FA(['a', 'b', 'c'])]
    with pytest.raises(TypeError):
        _, _ = ismember(a_keys, b_keys)

    # different number of key columns
    a_keys = [FA([1, 2, 3])]
    b_keys = [FA([1, 2, 3]), FA([1, 2, 3], dtype=np.float64)]
    with pytest.raises(ValueError):
        _, _ = ismember(a_keys, b_keys)


def test_ismember_multikey_single_key():
    # send single columns through regular ismember
    a_keys = [FA([1, 2, 3])]
    b_keys = [FA([1, 2])]
    c, d = ismember(a_keys, b_keys)
    correct_bool = [True, True, False]
    correct_idx = FastArray([0, 1, INVALID_DICT[d.dtype.num]], dtype=d.dtype)

    assert_array_equal(
        c, correct_bool, err_msg=f"Incorrect boolean array for single-key ismembermk"
    )
    assert_array_equal(
        d, correct_idx, err_msg=f"Incorrect index array for single-key ismembermk"
    )


def test_ismember_multikey_non_ndarray():
    # lists
    correct_bool = [True, True, False]
    correct_idx = FastArray([0, 1, INVALID_DICT[np.dtype(np.int8).num]], dtype=np.int8)

    a_keys = [[1, 2, 3], ['a', 'b', 'c']]
    b_keys = [[1, 2], ['a', 'b']]
    c, d = ismember(a_keys, b_keys)
    assert_array_equal(
        c, correct_bool, err_msg=f"Incorrect boolean array for keys as lists ismembermk"
    )
    assert_array_equal(
        d, correct_idx, err_msg=f"Incorrect index array for keys as lists ismembermk"
    )

    # tuples
    a_keys = [tuple(key) for key in a_keys]
    b_keys = [tuple(key) for key in b_keys]
    c, d = ismember(a_keys, b_keys)
    assert_array_equal(
        c,
        correct_bool,
        err_msg=f"Incorrect boolean array for keys as tuples ismembermk",
    )
    assert_array_equal(
        d, correct_idx, err_msg=f"Incorrect index array for keys as tuples ismembermk"
    )

    # nparray
    a_keys = [np.array(key) for key in a_keys]
    b_keys = [np.array(key) for key in b_keys]
    c, d = ismember(a_keys, b_keys)
    assert_array_equal(
        c,
        correct_bool,
        err_msg=f"Incorrect boolean array for keys as np arrays ismembermk",
    )
    assert_array_equal(
        d,
        correct_idx,
        err_msg=f"Incorrect index array for keys as np arrays ismembermk",
    )


# TODO: RIP-364: This test doesn't have any assertions, so it's currently only checking whether the function excepts. Add assertions to check the ismember output for correctness."
def test_ismember_multikey_unicode():
    a_keys = [
        np.array([u'a\u2082', u'b\u2082', u'c\u2082', u'd\u2082', u'e\u2082']),
        [1, 2, 3, 4, 5],
    ]
    b_keys = [np.array([u'a\u2082', u'b\u2082']), [1, 2]]
    b, idx = ismember(a_keys, b_keys)
    # with pytest.raises(TypeError):
    #    _, _ = ismember(a_keys, b_keys)


def test_ismember_align_multikey():
    correct_bool = FastArray([True, True, True, False, False])
    correct_idx = FastArray([0, 1, 2, int8.inv, int8.inv], dtype=np.int8)

    # bytes / unicode both upcast
    a_keys = [arange(5), FastArray([b'a', b'b', b'c', b'd', b'e'], dtype='S5')]
    b_keys = [arange(3), FastArray(['a', 'b', 'c'], dtype='U4', unicode=True)]
    b, idx = ismember(a_keys, b_keys)
    assert_array_equal(b, correct_bool)
    # NOTE: flip to numpy because FastArray is sentinel-aware
    assert_array_equal(idx._np, correct_idx._np)
    assert a_keys[1].dtype.char == 'S'

    # bytes / Categorical unicode
    a_keys = [arange(5), FastArray(['a', 'b', 'c', 'd', 'e'], dtype='S5')]
    b_keys = [
        arange(3),
        Categorical(FastArray(['a', 'b', 'c'], dtype='U4', unicode=True), unicode=True),
    ]
    b, idx = ismember(a_keys, b_keys)
    assert_array_equal(b, correct_bool)
    # NOTE: flip to numpy because FastArray is sentinel-aware
    assert_array_equal(idx._np, correct_idx._np)

    # unicode / Categorical
    a_keys = [arange(5), FastArray(['a', 'b', 'c', 'd', 'e'], dtype='U5', unicode=True)]
    b_keys = [
        arange(3),
        Categorical(FastArray(['a', 'b', 'c'], dtype='U4', unicode=True), unicode=True),
    ]
    b, idx = ismember(a_keys, b_keys)
    assert_array_equal(b, correct_bool)
    # NOTE: flip to numpy because FastArray is sentinel-aware
    assert_array_equal(idx._np, correct_idx._np)

    # different numeric types
    a_keys = [
        arange(5, dtype=np.float64),
        FastArray(['a', 'b', 'c', 'd', 'e'], dtype='U5', unicode=True),
    ]
    b_keys = [
        arange(3),
        Categorical(FastArray(['a', 'b', 'c'], dtype='U4', unicode=True), unicode=True),
    ]
    b, idx = ismember(a_keys, b_keys)
    assert_array_equal(b, correct_bool)
    # NOTE: flip to numpy because FastArray is sentinel-aware
    assert_array_equal(idx._np, correct_idx._np)

    # string / non-string
    a_keys = [
        arange(5).astype('S'),
        FastArray(['a', 'b', 'c', 'd', 'e'], dtype='U5', unicode=True),
    ]
    b_keys = [
        arange(3),
        Categorical(FastArray(['a', 'b', 'c'], dtype='U4', unicode=True), unicode=True),
    ]
    with pytest.raises(TypeError):
        b, idx = ismember(a_keys, b_keys)

    # multikey categorical, no expand array
    a_keys = [
        arange(5).astype('S'),
        FastArray(['a', 'b', 'c', 'd', 'e'], dtype='U5', unicode=True),
    ]
    b_keys = [
        arange(3),
        Categorical(
            [FastArray(['a', 'b', 'c'], dtype='U4', unicode=True), arange(3)],
            unicode=True,
        ),
    ]
    with pytest.raises(TypeError):
        b, idx = ismember(a_keys, b_keys)
    with pytest.raises(TypeError):
        b, idx = ismember(b_keys, a_keys)

    # unsupported object array
    a_keys = [arange(5).astype('O'), FastArray(['a', 'b', 'c', 'd', 'e'], dtype='S5')]
    b_keys = [arange(3), FastArray(['a', 'b', 'c'], dtype='U4', unicode=True)]

    with pytest.raises(TypeError):
        b, idx = ismember(a_keys, b_keys)


def test_ismember_categorical_with_invalid():
    c1 = Categorical(['a', 'b', 'c'])
    c2 = Categorical(['a', 'b', 'c', 'd', 'c'])
    c1[0] = 0
    c2[3] = 0
    # This is a little hacky because it relies on the implementation detail
    np.asarray(c1)[1] = c1.inv
    np.asarray(c2)[2] = c2.inv
    is_in, idx = ismember(c1, c2)
    expected_is_in = FastArray([False, False, True])
    assert_array_equal(is_in, expected_is_in)
    expected_idx = FastArray([0, 0, 4], dtype=idx.dtype)
    expected_idx[0:2] = expected_idx.inv
    assert_array_equal(idx, expected_idx)


if __name__ == "__main__":
    tester = unittest.main()
