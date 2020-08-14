import numpy as np
import random as rnd
import unittest

from riptable import Dataset, Cat
from riptable.rt_enum import INVALID_DICT


master_type_dict = {
    'bool': np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0], dtype=np.bool),
    'int8': np.array(
        [
            26,
            1,
            -9,
            INVALID_DICT[np.dtype('int8').num],
            13,
            INVALID_DICT[np.dtype('int8').num],
            26,
            5,
            26,
            -4,
            27,
            28,
            30,
            -32,
            15,
        ],
        dtype=np.int8,
    ),
    'uint8': np.array(
        [
            45,
            57,
            60,
            49,
            19,
            29,
            18,
            1,
            62,
            INVALID_DICT[np.dtype('uint8').num],
            55,
            47,
            31,
            INVALID_DICT[np.dtype('uint8').num],
            27,
        ],
        dtype=np.uint8,
    ),
    'int16': np.array(
        [
            -601,
            -332,
            162,
            375,
            160,
            -357,
            -218,
            -673,
            INVALID_DICT[np.dtype('int16').num],
            378,
            -175,
            -529,
            INVALID_DICT[np.dtype('int16').num],
            -796,
            -365,
        ],
        dtype=np.int16,
    ),
    'uint16': np.array(
        [
            1438,
            1723,
            433,
            1990,
            INVALID_DICT[np.dtype('uint16').num],
            1528,
            1124,
            42,
            1316,
            1003,
            1874,
            INVALID_DICT[np.dtype('uint16').num],
            1533,
            1443,
            1170,
        ],
        dtype=np.uint16,
    ),
    'int32': np.array(
        [
            1896652134,
            -1424042309,
            INVALID_DICT[np.dtype('int32').num],
            503239478,
            1067866129,
            -1974125613,
            -1608929297,
            -301645171,
            1402604369,
            INVALID_DICT[np.dtype('int32').num],
            1080040975,
            -289078200,
            -823277029,
            -1383139138,
            978724859,
        ],
        dtype=np.int32,
    ),
    'uint32': np.array(
        [
            337083591,
            688548370,
            INVALID_DICT[np.dtype('uint32').num],
            580206095,
            328423284,
            211281118,
            912676658,
            132565336,
            399918307,
            425384120,
            723039073,
            252319702,
            750186713,
            197297577,
            INVALID_DICT[np.dtype('uint32').num],
        ],
        dtype=np.uint32,
    ),
    'int64': np.array(
        [
            INVALID_DICT[np.dtype('int64').num],
            -423272446,
            -235992796,
            -1442995093,
            INVALID_DICT[np.dtype('int64').num],
            109846344,
            -1458628816,
            232007889,
            -1671608168,
            1752532663,
            -1545252943,
            544588670,
            -1385051680,
            -137319813,
            -195616592,
        ],
        dtype=np.int64,
    ),
    'uint64': np.array(
        [
            765232401,
            398653552,
            203749209,
            288238744,
            INVALID_DICT[np.dtype('uint64').num],
            271583764,
            985270266,
            391578626,
            196046134,
            916025221,
            694962930,
            34303390,
            647346354,
            INVALID_DICT[np.dtype('uint64').num],
            334977533,
        ],
        dtype=np.uint64,
    ),
    'float32': np.array(
        [
            np.nan,
            0.6201803850883267,
            0.05285394972525459,
            0.1435023986327576,
            np.nan,
            0.32308353808130397,
            0.1861463881422203,
            0.6366386808076959,
            0.7703864299590418,
            0.8155206130668257,
            0.9588669164271945,
            0.2832984888482334,
            0.02662158289064087,
            0.2591740277624228,
            0.28945199094333374,
        ]
    ).astype(np.float32),
    'float64': np.array(
        [
            0.264105510380617,
            np.nan,
            0.9094594817708785,
            0.13757414135018453,
            0.9997438463622871,
            0.1642171078246103,
            0.4883940875811662,
            0.2819313242616074,
            0.7868397473215173,
            0.8963052108412053,
            0.03571507605557389,
            0.6423436033517553,
            0.04402603090628798,
            0.5619514123321582,
            np.nan,
        ]
    ).astype(np.float64),
    'bytes': np.array(
        [
            INVALID_DICT[np.dtype('bytes').num],
            b'12398dfkw',
            b'dlkv;lk3-2',
            b'111dkjfj3',
            b'e0383hjfns',
            b'qwernvldkj',
            b'abefgkejf',
            b'as777nrn',
            b'23dhsjkifuywfwefj',
            INVALID_DICT[np.dtype('bytes').num],
            b'zkdfjlw',
            b'a',
            b';][{}[\|||+=_-',
            b'qwernvldkj',
            b'abefgkejf',
        ],
        dtype=np.bytes_,
    ),
    'unicode': np.array(
        [
            'asdf233rf',
            '12398dfkw',
            'dlkv;lk3-2',
            '111dkjfj3',
            'e0383hjfns',
            INVALID_DICT[np.dtype('str_').num],
            'abefgkejf',
            'as777nrn',
            '23dhsjkifuywfwefj',
            'rrrrn2fhfewl',
            'zkdfjlw',
            'a',
            ';][{}[\|||+=_-',
            'qwernvldkj',
            INVALID_DICT[np.dtype('str_').num],
        ],
        dtype=np.str_,
    ),
}

simple_keys = dict(
    zip(
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'],
        list(master_type_dict.values()),
    )
)
ds = Dataset(simple_keys)
num_cols = len(simple_keys)
num_rows = len(simple_keys['a'])
dict_key_list = list(simple_keys.keys())
dict_col_names = np.array(dict_key_list)
# --------------------SLICE DATA---------------------------------------------
# ---------------------------------------------------------------------------
single_slices = {
    ":2": slice(None, 2, None),
    "-2:": slice(-2, None, None),
    "2:5": slice(2, 5, None),
    "5:": slice(5, None, None),
    ":": slice(None, None, None),
}

# ---------------------------------------------------------------------------
row_bool_arrays = {
    "python_bool": [
        True,
        False,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
    ],
    "numpy_bool": np.array(
        [
            True,
            False,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
        ]
    ),
}

# ---------------------------------------------------------------------------
col_bool_arrays = {
    "python_bool": [
        True,
        False,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
        True,
        False,
        False,
        True,
    ],
    "numpy_bool": np.array(
        [
            True,
            False,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
            True,
            False,
            False,
            True,
        ]
    ),
}

# ---------------------------------------------------------------------------
col_string_lists = {
    "col_names_size"
    + str(sample_size): [
        dict_key_list[i] for i in rnd.sample(range(1, num_cols), sample_size)
    ]
    for sample_size in range(1, num_cols)
}

# ---------------------------------------------------------------------------
row_idx_arrays = {
    "int_idx_size"
    + str(idx_size): np.random.randint(low=0, high=num_rows, size=idx_size)
    for idx_size in range(1, num_rows)
}

# ---------------------------------------------------------------------------
col_idx_arrays = {
    "int_idx_size"
    + str(idx_size): np.random.randint(low=0, high=num_cols, size=idx_size)
    for idx_size in range(1, num_cols)
}

# change to True for printouts of slice input
ShowSliceInfo = False


class Dataset_Slice_Accuracy(unittest.TestCase):

    # ------------------GENERAL COMPARE FUNCTIONS-------------------------------
    # --------------------------------------------------------------------------
    def try_conversions(self, ds_item, dict_item):
        # nan != nan, and strings get converted from unicode to bytes
        # make sure these aren't the same values in a different form
        nans = False
        try:
            if np.isnan(ds_item) and np.isnan(dict_item):
                nans = True
        except TypeError:
            pass
        if nans:
            pass

        elif isinstance(ds_item, bytes):
            ds_item_str = ds_item.decode()
            if ds_item_str == dict_item:
                pass
        else:
            if ShowSliceInfo is True:
                print("Items did not match!")
                print("incorrect:", ds_item, "correct:", dict_item)
            return 1
        return 0

    # --------------------------------------------------------------------------
    def match_lists(self, ds_list, dict_list):
        # compares items in lists
        for idx, ds_item in enumerate(ds_list):
            dict_item = dict_list[idx]

            if ds_item != dict_item:
                # further test for nans and string type conversions
                try_convert = self.try_conversions(ds_item, dict_item)
                self.assertEqual(try_convert, 0)

    # -------------------------ROW ONLY-----------------------------------------
    # --------------------------------------------------------------------------
    def rsingle(self, slice_dict):
        # checks list of lists
        for input_str, input in slice_dict.items():
            ds1 = ds[input, :]
            ds_list = [getattr(ds1, col) for col in ds1]
            dict_list = [val[input] for val in simple_keys.values()]

            if ShowSliceInfo:
                print("Checking ds[" + str(input_str) + "]")
            for idx, ds_section in enumerate(ds_list):
                dict_section = dict_list[idx]
                self.match_lists(ds_section, dict_section)

    def row_single(self):
        # [int]
        dict_size = len(simple_keys['a'])
        for input in range(dict_size):
            # TODO: fix bug where slice is [-1] in ds._getitem how single ints are handled
            ds1 = ds[input, :]
            ds_list = [getattr(ds1, col)[0] for col in ds1]
            dict_list = [val[input] for val in simple_keys.values()]

            if ShowSliceInfo:
                print("Checking ds[" + str(input) + "]")
            self.match_lists(ds_list, dict_list)

    # --------------------------------------------------------------------------
    def test_row_masks(self):
        # [int:int]
        # [[bool, bool, bool]]
        # [[int, int, int]]
        errors = 0
        for row_param in [single_slices, row_bool_arrays, row_idx_arrays]:
            self.rsingle(row_param)

    # ----------------------------COLUMN ONLY-----------------------------------
    # --------------------------------------------------------------------------
    def col_string(self):
        # ['col']
        for input, dict_list in simple_keys.items():
            if ShowSliceInfo:
                print("Checking ds['" + str(input) + "']")
            ds_list = ds[input]
            self.match_lists(ds_list, dict_list)

    # --------------------------------------------------------------------------
    def col_string_list(self):
        # [['col1', 'col2', 'col3']]
        slice_dict = col_string_lists

        for input_str, input in slice_dict.items():
            ds1 = ds[input]
            ds_list = [getattr(ds1, col) for col in ds1]
            dict_list = [simple_keys[i] for i in input]

            if ShowSliceInfo:
                print("Checking ds[" + str(input_str) + "]")
            for idx, ds_section in enumerate(ds_list):
                dict_section = dict_list[idx]
                self.match_lists(ds_section, dict_section)

    # --------------------COMBINED SLICING--------------------------------------
    # --------------------------------------------------------------------------
    def rsingle_cmulti(self, slice_dict):
        # [int, slice/bool/idx]

        num_rows = len(simple_keys['a'])
        for col_slice_str, col_slice in slice_dict.items():
            dict_sliced_col_names = dict_col_names[col_slice]
            for row_num in range(num_rows):
                try:
                    ds1 = ds[row_num, col_slice]
                except IndexError as e:
                    self.assertEqual(e.args[0], 'Cannot index cols with duplicates.')
                    continue
                if ShowSliceInfo:
                    print(f"Checking ds[{row_num}, {col_slice_str}]")
                dict_list = [
                    simple_keys[dict_name][row_num]
                    for dict_name in dict_sliced_col_names
                ]
                ds_list = [
                    getattr(ds1, dict_name)[0] for dict_name in dict_sliced_col_names
                ]
                self.match_lists(ds_list, dict_list)

    # --------------------------------------------------------------------------
    def rmulti_cmulti(self, row_multi_arrays, col_multi_arrays):
        # [slice/bool/idx, slice/bool/idx]
        for col_slice_str, col_slice in col_multi_arrays.items():
            dict_sliced_col_names = dict_col_names[col_slice]
            for row_slice_str, row_slice in row_multi_arrays.items():
                try:
                    ds1 = ds[row_slice, col_slice]
                except IndexError as e:
                    self.assertEqual(e.args[0], 'Cannot index cols with duplicates.')
                    continue
                if ShowSliceInfo:
                    print(f"Checking ds[{row_slice_str}, {col_slice_str}]")
                if ShowSliceInfo:
                    print("Checking ds[" + row_slice_str + "," + col_slice_str + "]")
                for dict_name in dict_sliced_col_names:
                    dict_list = simple_keys[dict_name][row_slice]
                    ds_list = getattr(ds1, dict_name)
                    self.match_lists(ds_list, dict_list)

    # --------------------------------------------------------------------------
    def rsingle_csingle(self):
        # [int, int]
        errors = 0
        for col_idx, (col_name, dict_list) in enumerate(simple_keys.items()):
            if ShowSliceInfo:
                print("Checking rows in column", col_name)
            num_rows = len(dict_list)
            for row_idx in range(num_rows):
                ds1 = ds[row_idx, col_idx]
                ds_value = getattr(ds1, col_name)[0]
                dict_value = dict_list[row_idx]
                if ds_value != dict_value:
                    try_convert = self.try_conversions(ds_value, dict_value)
                    self.assertEqual(try_convert, 0)

    # --------------------------------------------------------------------------
    def rsingle_cstringlist(self):
        # [int, ['col1', 'col2', 'col3']]

        for col_str, col_stringlist in col_string_lists.items():
            for row_num in range(num_rows):
                ds1 = ds[row_num, col_stringlist]

                if ShowSliceInfo:
                    print("Checking ds[" + str(row_num) + "," + col_str + "]")
                dict_list = [
                    simple_keys[dict_name][row_num] for dict_name in col_stringlist
                ]
                ds_list = [getattr(ds1, dict_name)[0] for dict_name in col_stringlist]
                self.match_lists(ds_list, dict_list)

    # --------------------------------------------------------------------------
    def rmulti_cstringlist(self, row_multi_arrays):
        # [slice/bool/idx, ['col1', 'col2', 'col3']]
        for col_str, col_stringlist in col_string_lists.items():
            for row_slice_str, row_slice in row_multi_arrays.items():
                if ShowSliceInfo:
                    print("Checking ds[" + row_slice_str + "," + col_str + "]")
                ds1 = ds[row_slice, col_stringlist]
                for dict_name in col_stringlist:
                    dict_list = simple_keys[dict_name][row_slice]
                    ds_list = getattr(ds1, dict_name)
                    self.match_lists(ds_list, dict_list)

    # --------------------------------------------------------------------------
    def test_rsingle_cmask_combos(self):
        # [int, int:int]
        # [int, [bool, bool, bool]]
        # [int, [int, int, int]]
        for col_param in [single_slices, col_bool_arrays, col_idx_arrays]:
            self.rsingle_cmulti(col_param)

    # --------------------------------------------------------------------------
    def test_rmask_cstringlist_combos(self):
        # [int:int, ['col1', 'col2', 'col3']]
        # [[bool, bool, bool], ['col1', 'col2', 'col3']]
        # [[int, int, int], ['col1', 'col2', 'col3']]
        for row_param in [single_slices, row_bool_arrays, row_idx_arrays]:
            self.rmulti_cstringlist(row_param)

    # --------------------------------------------------------------------------
    def test_all_mask_combos(self):
        # [int:int, int:int]
        # [int:int, [bool, bool, bool]]
        # [int:int, [int, int, int]]
        # [[bool, bool, bool], int:int]
        # [[bool, bool, bool], [bool, bool, bool]]
        # [[bool, bool, bool], [int, int, int]]
        # [[int, int, int], int:int]
        # [[int, int, int], [bool, bool, bool]]
        # [[int, int, int], [int, int, int]]
        for row_param in [single_slices, row_bool_arrays, row_idx_arrays]:
            for col_param in [single_slices, col_bool_arrays, col_idx_arrays]:
                self.rmulti_cmulti(row_param, col_param)

    # --------------------------------------------------------------------------
    def test_slice_accuracy(self):
        self.row_single()
        self.col_string()
        self.col_string_list()

    def test_add_dataset(self):
        arrsize = 200
        numrows = 7

        ds = Dataset({'time': np.arange(arrsize * 1.0)})
        ds.data = np.random.randint(numrows, size=arrsize)
        ds.data2 = np.random.randint(numrows, size=arrsize)
        symbols = [
            'AAPL',
            'AMZN',
            'FB',
            'GOOG',
            'IBM',
            '6',
            '7',
            '8',
            '9',
            '10',
            '11',
            '12',
            '13',
            '14',
            '15',
            '16',
            '17',
            '18',
        ]
        symbol2 = ['A', 'X', 'P', 'C', 'D', 'E', 'F', 'G', 'G', 'I', 'J', 'K']
        ds.symbol2 = Cat(1 + np.arange(arrsize) % len(symbol2), symbol2)
        ds.symbol = Cat(1 + np.arange(arrsize) % len(symbols), symbols)

        x = ds.copy()
        del x.symbol
        del x.data
        del x.time
        x.label_set_names('data2')

        # now x has two columns, and one is labelled so adding an entire dataset should just add x.symbol2
        ds.junk = x


if __name__ == '__main__':
    tester = unittest.main()
