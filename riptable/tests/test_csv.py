# $Id: //Depot/Source/SFW/riptable/Python/core/riptable/tests/test_csv.py#8 $
import unittest
import sys, os
import numpy as np
from riptable import rt_csv


testdir = os.getenv('TEST_DIR', os.path.join(os.path.dirname(__file__), 'test_files'))


def sfloat(x):
    try:
        return float(x)
    except:
        return np.nan


def sint(x):
    try:
        return int(x)
    except:
        return sys.maxsize


def scfloat(x):
    if type(x) is str:
        x = x.replace(',', '')
    return sfloat(x)


def scint(x):
    if type(x) is str:
        x = x.replace(',', '')
    return sint(x)


class Utility_Test(unittest.TestCase):
    def test_00(self, fname=None):
        if fname is None:
            fname = os.path.join(testdir, 'unicode_ex3.csv')

        with open(fname, encoding='utf-8') as infile:
            headers = [_tok.strip('"') for _tok in infile.readline().strip().split(',')]
        ident = lambda x: x
        conv0 = {_k: ident for _k in headers}
        # conv1 = {_k: str for _k in headers}
        dsets = {}
        for ii in range(4):
            dsets[f'v{ii}'] = rt_csv.load_csv_as_dataset(
                fname, headers, conv0, version=ii, skip_rows=1
            )
        for _k in dsets:
            self.assertTrue((dsets['v0'] == dsets[_k]).all(axis=None))
            self.assertEqual(dsets[_k].shape, (35, 5))

        for ii in range(4):
            dsets[f'v{ii}'] = rt_csv.load_csv_as_dataset(
                fname, headers, conv0, version=ii, skip_rows=6
            )
        for _k in dsets:
            self.assertTrue((dsets['v0'] == dsets[_k]).all(axis=None))
            self.assertEqual(dsets[_k].shape, (30, 5))

    # BUG in this test. calling pd.read_csv corrupts memory and changes tests that follow
    # def test_02(self):
    #    cnv = dict(KEY=str, Region=str, Total=scint, Hombre=scint, Mujer=scint)
    #    fname1 = os.path.join(testdir, 'unicode_ex1.csv')
    #    fname2 = os.path.join(testdir, 'unicode_ex2.csv')
    #    fname3 = os.path.join(testdir, 'unicode_ex3.csv')

    #    df1 = pd.read_csv(fname1, encoding='latin-1', converters=cnv)
    #    df2 = pd.read_csv(fname2, encoding='utf-8', converters=cnv)

    #    self.assertTrue((df1 == df2).all().all())
    #    df3 = pd.read_csv(fname3, encoding='utf-8', converters=cnv)
    #    ds1 = dataset_from_pandas_df(df1)
    #    ds2 = dataset_from_pandas_df(df2)
    #    ds3 = dataset_from_pandas_df(df3)
    #    self.assertEqual(ds1.shape, (33, 5))
    #    self.assertEqual(ds2.shape, (33, 5))
    #    self.assertEqual(ds3.shape, (35, 5))
    #    with open(fname3, encoding='utf-8') as infile:
    #        ds4 = rt_csv.load_csv_as_dataset(infile, None, cnv)
    #    self.assertTrue((ds4 == ds3).all(axis=None))
    #    with open(fname3, encoding='utf-8') as infile:
    #        ds5 = rt_csv.load_csv_as_dataset(infile, list(ds3.keys()), cnv, skip_rows=1)
    #    self.assertTrue((ds5 == ds3).all(axis=None))
    #    sum1 = ds3[ds3.Region > 'Zacatecas', :][['Total', 'Hombre', 'Mujer']].sum()
    #    sum2 = dataset_as_pandas_df(ds3[ds3.Region > 'Zacatecas', :][['Total', 'Hombre', 'Mujer']]).sum()
    #    # ^-- reorders my fields alphabetically
    #    self.assertEqual(sum1.tolist()[0], sum2.values.tolist())
    #    subdf1 = dataset_as_pandas_df(ds3[ds3.Region < 'Puebla', :])
    #    subdf2 = dataset_as_pandas_df(ds2[ds2.Region < 'Puebla', :])
    #    self.assertTrue((subdf1 == subdf2).all().all())
    #    self.assertTrue((ds3[ds3.Region < 'Puebla', :] == ds2[ds2.Region < 'Puebla', :]).all(axis=None))
    #    return (df1, df2, df3), (ds1, ds2, ds3)

    # def test_03(self):
    #    cnv = dict(KEY=str, Region=str, Total=scint, Hombre=scint, Mujer=scint)
    #    fname3 = os.path.join(testdir, 'unicode_ex3.csv')
    #    df3 = pd.read_csv(fname3, encoding='utf-8', converters=cnv)
    #    ds3 = dataset_from_pandas_df(df3)
    #    ds3['Cs'] = [_s.startswith('C') for _s in ds3.Region]
    #    ds3['l1'] = [_s[0] for _s in ds3.Region]
    #    gb3_sum = ds3.groupby(['l1', 'Cs']).sum()
    #    result = gb3_sum.asdict()
    #    fname4 = os.path.join(testdir, 'groupby1_ex3.pickle')
    #    with open(fname4, 'rb') as infile:
    #        comp = pickle.load(infile)

    #    # asdict() now returns groupby columns in addition to the rest of the columns
    #    # had to remove groupby columns from asdict() result for correct comparison
    #    gb_keys = gb3_sum.label_get_names()
    #    correct_headers = [ h for h in list(result) if h not in gb_keys ]
    #    self.assertEqual(list(comp), correct_headers)

    #    correct_values = [ v for k,v in result.items() if k not in gb_keys ]
    #    for _v1, _v2 in zip(comp.values(), correct_values):
    #        self.assertTrue((_v1 ==_v2).all())


if __name__ == "__main__":
    tester = unittest.main()
