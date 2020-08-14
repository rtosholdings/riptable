import unittest
from riptable import *


class GroupByOps_Test(unittest.TestCase):
    def test_categorical_values(self):
        vals = np.random.choice(['a', 'b', 'c', 'd', 'e'], 50)
        ds = Dataset({'vals': vals})
        # TODO what assertions need to be done on 'c' and 'gb'
        c = Categorical(vals, ordered=False)
        gb = ds.gb('vals')

        vals = np.random.choice(['a', 'b', 'c', 'd', 'e'], 50)
        ds = Dataset({'vals': vals})
        c = Categorical(vals)
        gb = ds.gb('vals', lex=True)
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = FA(vals, unicode=True)
        ds = Dataset({'vals': vals})
        c = Categorical(vals, ordered=False, unicode=True)
        gb = ds.gb('vals')
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = np.random.randint(0, 1000, 50)
        ds = Dataset({'vals': vals})
        c = Categorical(vals, ordered=False)
        gb = ds.gb('vals')
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 50)
        ds = Dataset({'vals': vals})
        c = Categorical(vals, ordered=False)
        gb = ds.gb('vals')
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = np.random.choice(['a', 'b', 'c', 'd', 'e'], 50)
        ds = Dataset({'vals': vals})
        c = Categorical(vals, ordered=False, sort_gb=False)
        gb = ds.gb('vals')
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = FA(vals, unicode=True)
        ds = Dataset({'vals': vals})
        c = Categorical(vals, ordered=False, unicode=True, sort_gb=False)
        gb = ds.gb('vals')
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = np.random.randint(0, 1000, 50)
        ds = Dataset({'vals': vals})
        c = Categorical(vals, ordered=False, sort_gb=False)
        gb = ds.gb('vals')
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 50)
        ds = Dataset({'vals': vals})
        c = Categorical(vals, ordered=False, sort_gb=False)
        gb = ds.gb('vals')
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        # 4/29/2019 - iFirstKey gets generated later, uses a new routine
        # need to set ordered to False to correctly match groupby results
        # groupby always uses first-occurence
        # or ordered=True groupby(lex=True)

        vals = np.random.choice(['a', 'b', 'c', 'd', 'e'], 50)
        ds = Dataset({'vals': vals})
        c = Categorical(vals, ordered=True)
        gb = ds.gb('vals', lex=True)
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = FA(vals, unicode=True)
        ds = Dataset({'vals': vals})
        c = Categorical(vals, unicode=True, ordered=True)
        gb = ds.gb('vals', lex=True)
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = np.random.randint(0, 1000, 50)
        ds = Dataset({'vals': vals})
        c = Categorical(vals, ordered=True)
        gb = ds.gb('vals', lex=True)
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

        vals = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 50)
        ds = Dataset({'vals': vals})
        c = Categorical(vals, ordered=True)
        gb = ds.gb('vals', lex=True)
        self.assertTrue(bool(np.all(c.ifirstkey == gb.ifirstkey)))

    def test_categorical_values_cats(self):
        pass

    def test_categorical_codes_cats(self):
        pass

    def test_categorical_mapping(self):
        pass

    def test_categorical_multikey(self):
        pass

    def test_gb_userfunc(self):
        ds = Dataset({'strs': np.random.choice(['a', 'b', 'c'], 20)})
        ds.data = arange(20)
        # make sure np.quantile stlye works where second argument is passed
        ds.gb('strs').apply_nonreduce(np.quantile, 0.5)

    def test_gb_extra_data(self):
        ds = Dataset({'strs': np.random.choice(['a', 'b', 'c'], 20)})
        ds.col_AAA = ones(20)
        ds.col_BBB = np.random.rand(20)
        extra = arange(20)
        extra.set_name('extra')
        ds2 = ds.gb('strs').sum(extra)

        final_cols = ['strs', 'col_AAA', 'col_BBB', 'extra']
        self.assertTrue(bool(np.all(list(ds2) == final_cols)))

    def test_cat_add_dataset(self):
        ds = Dataset({'strs': np.random.choice(['a', 'b', 'c'], 20)})
        ds.col_AAA = ones(20)
        ds.col_BBB = np.random.rand(20)
        c = Categorical(ds.strs)
        ds2 = c.sum(dataset=ds)
        final_cols = ['strs', 'col_AAA', 'col_BBB']
        self.assertTrue(bool(np.all(list(ds2) == final_cols)))

        extra = arange(20)
        extra.set_name('extra')
        ds3 = c.sum(extra, dataset=ds)
        final_cols = ['strs', 'col_AAA', 'col_BBB', 'extra']
        self.assertTrue(bool(np.all(list(ds3) == final_cols)))

    def test_warn_conflicting_data(self):
        ds = Dataset({'strs': np.random.choice(['a', 'b', 'c'], 20)})
        ds.col_AAA = ones(20)
        ds.col_BBB = np.random.rand(20)
        c = Categorical(np.random.choice(['a', 'b', 'c'], 20))
        with self.assertWarns(UserWarning):
            ds2 = c.sum(ds, dataset=ds)

    def test_error_no_data(self):
        c = Categorical(np.random.choice(['a', 'b', 'c'], 20))
        with self.assertRaises(ValueError):
            _ = c.sum()

        ds = Dataset({'strs': np.random.choice(['a', 'b', 'c'], 20)})
        ds.col_AAA = ones(20)
        ds.col_BBB = np.random.rand(20)
        gb = ds.gb('strs')
        gb._dataset = None
        with self.assertRaises(ValueError):
            _ = gb.sum()

    def test_ema_funcs(self):
        ds = Dataset({'test': arange(10), 'group2': arange(10) % 3})

        gb = ds.gb('group2')
        c = Categorical(ds.group2)

        # decay_rate now required to be filled in
        gb_result = gb.ema_normal(time=arange(10), decay_rate=1.0)
        c_result = c.ema_normal(ds.test, time=arange(10), decay_rate=1.0)

        for k, col in gb_result.items():
            self.assertTrue(bool(np.all(col == c_result[k])))

    def test_agg_funcs(self):
        ds = Dataset({'col_' + str(i): np.random.rand(20) for i in range(10)})
        ds.keycol = np.random.choice(['a', 'b', 'c'], 20)
        c = Categorical(ds.keycol)
        flt = np.random.rand(20).view(FA)
        flt.set_name('flt')

        gb_result = ds.gb('keycol').agg(
            {'col_0': ['sum', np.mean], 'col_5': ['min', np.var], 'flt': 'max'}, flt
        )
        c_result = c.agg(
            {'col_0': ['sum', np.mean], 'col_5': ['min', np.var], 'flt': 'max'},
            flt,
            dataset=ds[:-1],
        )

        for dsname, d_total in gb_result.items():
            c_total = c_result[dsname]
            for k, v in d_total.items():
                self.assertTrue(bool(np.all(c_total[k] == v)))

    def test_agg_funcs_np(self):
        c = Categorical(np.random.choice(['a', 'b', 'c'], 20))

        rt_result = c.agg(['min', 'max'], arange(20))
        np_result = c.agg([np.amin, np.amax], arange(20))

        for dsname, d_total in rt_result.items():
            c_total = np_result[dsname]
            for k, v in d_total.items():
                self.assertTrue(bool(np.all(c_total[k] == v)))

    def test_agg_funcs_error(self):
        c = Categorical(np.random.choice(['a', 'b', 'c'], 20))
        with self.assertRaises(ValueError):
            _ = c.agg({'badname': 'min'}, arange(20))

        with self.assertRaises(ValueError):
            _ = c.agg(['min', 'max', 'badfunc'], arange(20))

    def test_groups_order(self):
        strings = np.array(['b', 'a', 'a', 'z', 'y', 'b', 'c'])
        ds = Dataset(
            {
                'col1': Categorical(strings, ordered=True),
                'col2': Categorical(strings, ordered=False),
                'col3': strings,
            }
        )
        d1 = ds.col1.groups
        d2 = ds.col2.groups
        d3 = ds.gb('col3').groups
        d4 = ds.gbu('col3').groups

        for colname, idx in d1.items():
            for d in (d2, d3, d4):
                self.assertTrue(bool(np.all(d[colname] == idx)))


if __name__ == "__main__":
    tester = unittest.main()
