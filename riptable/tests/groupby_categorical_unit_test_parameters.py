import numpy as np
import random

functions_str = [
    # 'count',
    'sum',
    'mean',
    'median',
    'min',
    'max',
    # 'prod',
    'var',
    # 'quantile',
    # 'cumsum',
    # 'cumprod',
    # 'cummax',
    # 'cummin'
]

VAL_COLUMNS_MAX = 8
VAL_COLUMNS_MIN = 1
VAL_COLUMNS_INC = 1

SYMBOL_RATIO_MAX = 0.5
SYMBOL_RATIO_MIN = 0.1
SYMBOL_RATIO_INC = 0.15

SIZE_DEFAULT = 100  ## to ensure that multithreading is enabled


def isNaN(num):
    return num != num


def safe_assert(ary1, ary2):

    if len(ary1) != len(ary2):
        print('mismatch lengths-', len(ary1), len(ary2))
        assert len(ary1) == len(ary2)
    epsilon = 0.00000005  ##diference must be in range of this

    for a, b in zip(ary1, ary2):
        assert abs(a - b) < epsilon or (isNaN(a) and isNaN(b))


def remove_nan(ary):
    return [x for x in ary if not x == 0 and not isNaN(x)]


class categorical_parameters:
    def __init__(self):
        self.val_cols = VAL_COLUMNS_MIN
        self.symbol_ratio = SYMBOL_RATIO_MIN
        self.numb_keys = int(SIZE_DEFAULT * self.symbol_ratio)
        self.aggs = ''
        self.bin_ids = []
        self.aggs_id = 0

        self.update('aggs')

    def update(self, parameter=None, increment=None):
        def update_aggs(self):
            self.aggs = functions_str[self.aggs_id % len(functions_str)]
            self.aggs_id += 1

        def update_symbs(self):
            inc = SYMBOL_RATIO_INC if increment is None else increment

            self.symbol_ratio += inc
            value = self.symbol_ratio
            self.symbol_ratio %= SYMBOL_RATIO_MAX
            if self.symbol_ratio < SYMBOL_RATIO_MIN:
                self.symbol_ratio = SYMBOL_RATIO_MIN

            return value != self.symbol_ratio

        def update_val_cols(self):
            inc = VAL_COLUMNS_INC if increment is None else increment

            self.val_cols += inc
            value = self.val_cols
            self.val_cols %= VAL_COLUMNS_MAX
            self.val_cols = (
                self.val_cols if self.val_cols > VAL_COLUMNS_MIN else VAL_COLUMNS_MIN
            )
            return value != self.val_cols

        if parameter is None:
            if update_val_cols(self):
                update_symbs(self)
            update_aggs(self)
        else:
            switch = {
                'aggs': update_aggs,
                'syms': update_symbs,
                'vals': update_val_cols,
            }
            switch[parameter](self)


class categorical_base:
    def __init__(self, val_cols, symbol_ratio, aggs):

        self.val_cols = val_cols
        self.symbol_ratio = symbol_ratio
        self.numb_keys = int(SIZE_DEFAULT * self.symbol_ratio)
        self.aggs = aggs
        self.generate_bin_data()
        self.data = self.generate_dummy_data()

    def generate_bin_data(self):
        self.numb_keys = int(SIZE_DEFAULT * self.symbol_ratio)
        self.keys = ['label' + str(x) for x in range(0, self.numb_keys)]
        self.bin_ids = [
            random.randint(0, self.numb_keys - 1) for x in range(0, SIZE_DEFAULT)
        ]

    import numpy as np

    def generate_dummy_data(self):
        col_names = ['bin' + char.upper() for char in 'abcdefghijklmnopqrstuviwxyz']
        col_names = col_names[0 : self.val_cols]

        data = {}

        for name in col_names:
            data[name] = [0.5 + x for x in np.random.rand(SIZE_DEFAULT)]

        # self.GB_KEYS = self.bin_ids
        # data['KEYS'] = self.bin_ids

        return data
