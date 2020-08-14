import unittest
from riptable import Dataset
import pandas as pd
from riptable.Utils.pandas_utils import dataset_as_pandas_df, dataset_from_pandas_df

# Parse the pandas library version into a tuple of integers.
pd_version = tuple(map(int, pd.__version__.split('.')))

# Well-known pandas version(s) that we may need to recognize
# within the tests below.
pd_ver_0_24 = (0, 24, 0)


class TestPandasUtils(unittest.TestCase):
    def test_dataset_from_pandas_df_warn(self):
        df = pd.DataFrame({'a': [1, 2, 3]})

        if pd_version >= pd_ver_0_24:
            with self.assertWarns(FutureWarning):
                ds = dataset_from_pandas_df(df)
        else:
            ds = dataset_from_pandas_df(df)

    def test_dataset_as_pandas_df_warn(self):
        ds = Dataset({'a': [1, 2, 3]})

        if pd_version >= pd_ver_0_24:
            with self.assertWarns(FutureWarning):
                df = dataset_as_pandas_df(ds)
        else:
            df = dataset_as_pandas_df(ds)


if __name__ == "__main__":
    tester = unittest.main()
