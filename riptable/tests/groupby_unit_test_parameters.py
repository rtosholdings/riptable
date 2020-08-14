import numpy as np
import random as rand

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

KEY_COLUMN_NAMES = 'a b c d e g h i j k l m n o p r s t u v w x y z'.split(' ')
VAL_COLUMN_NAMES = ['val' + char.upper() for char in KEY_COLUMN_NAMES]

VAL_COLUMNS_MAX = 8
VAL_COLUMNS_MIN = 1
VAL_COLUMNS_INC = 3

KEY_COLUMNS_MAX = 4
KEY_COLUMNS_MIN = 1
KEY_COLUMNS_INC = 1

SYMBOL_RATIO_MAX = 0.5
SYMBOL_RATIO_MIN = 0.1
SYMBOL_RATIO_INC = 0.2

AGGREGATION_MAX = 5
AGGREGATION_MIN = 1
AGGREGATION_INC = 1

SIZE_DEFAULT = 100  ## to ensure that multithreading is enabled


def safe_assert(self, ary1, ary2):
    epsilon = 0.00000005

    def isNaN(num):
        return num != num

    assert len(ary1) == len(ary2)

    for a, b in zip(ary1, ary2):
        assert abs(a - b) < epsilon or (isNaN(a) and isNaN(b))


class groupby_parameters:
    def random_agg_list(self, numb_funcs):
        lst = []
        while len(lst) < numb_funcs:
            val = rand.randint(0, len(functions_str) - 1)
            val = functions_str[val]

            if not val in lst:
                lst.append(val)

        return lst

    def __init__(self):
        self.val_cols = VAL_COLUMNS_MIN
        self.key_cols = KEY_COLUMNS_MIN
        self.symbol_ratio = SYMBOL_RATIO_MIN
        self.aggs = AGGREGATION_MIN
        self.agg_list = self.random_agg_list(
            self.aggs
        )  ##itertools.combinations(functions_str, PARAMETERS.aggs).

    def update(self, parameter=None):
        def update_aggs(self):
            self.aggs += AGGREGATION_INC
            value = self.aggs
            self.aggs %= AGGREGATION_MAX

            if self.aggs < AGGREGATION_MIN:
                self.aggs = AGGREGATION_MIN

            self.agg_list = self.random_agg_list(self.aggs)
            return value != self.aggs

        def update_symbs(self):
            self.symbol_ratio += SYMBOL_RATIO_INC
            value = self.symbol_ratio
            self.symbol_ratio %= SYMBOL_RATIO_MAX
            if self.symbol_ratio < SYMBOL_RATIO_MIN:
                self.symbol_ratio = SYMBOL_RATIO_MIN
            return value != self.symbol_ratio

        def update_val_cols(self):
            self.val_cols += VAL_COLUMNS_INC
            value = self.val_cols
            self.val_cols %= VAL_COLUMNS_MAX
            self.val_cols = (
                self.val_cols if self.val_cols > VAL_COLUMNS_MIN else VAL_COLUMNS_MIN
            )
            return value != self.val_cols

        def update_key_cols(self):
            self.key_cols += KEY_COLUMNS_INC
            value = self.key_cols
            self.key_cols %= KEY_COLUMNS_MAX
            self.key_cols = (
                self.key_cols if self.key_cols > KEY_COLUMNS_MIN else KEY_COLUMNS_MIN
            )
            return value != self.key_cols

        if parameter is None:
            if update_val_cols(self):
                if update_key_cols(self):
                    if update_symbs(self):
                        update_aggs(self)
        else:
            switch = {
                'aggs': update_aggs,
                'syms': update_symbs,
                'vals': update_val_cols,
                'keys': update_key_cols,
            }

            switch[parameter]()
            ###---------------------------------------------


###---------------------------------------------------------------------------


class groupby_everything:

    val_columns = VAL_COLUMNS_MIN
    key_columns = KEY_COLUMNS_MIN
    size = SIZE_DEFAULT
    symbol_ratio = SYMBOL_RATIO_MIN

    aggregation_functions_count = AGGREGATION_MIN
    symbol_ratio_count = int(
        (size * symbol_ratio) ** (1 / val_columns)
    )  ###this equation ensures an equal distribution in respect to numb columns
    aggregation_functions = []
    data = {}

    def init_dummy_data(self):
        for i in range(0, self.val_columns):
            self.data[VAL_COLUMN_NAMES[i]] = np.random.rand(self.size)

        for i in range(0, self.key_columns):
            self.data[KEY_COLUMN_NAMES[i]] = [
                'k' + str(rand.randint(0, self.symbol_ratio_count))
                for i in np.random.randint(self.size, size=self.size)
            ]

    def __init__(
        self, val_cols, key_cols, symbs, agg_list, sz=SIZE_DEFAULT,
    ):
        self.val_columns = val_cols  ##integer number of value columns
        self.key_columns = key_cols  ##integer number of key   columns
        self.size = sz  ###integer number of rows
        self.aggregation_functions = agg_list
        self.symbol_ratio = symbs  ###float   ratio of symbols to size
        self.symbol_ratio_count = int(
            (self.size * self.symbol_ratio) ** (1 / self.val_columns)
        )  ###number of expected groups

        self.data = {}  ###the groupby data
        self.init_dummy_data()
