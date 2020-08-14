__all__ = ['normalize_zscore', 'normalize_minmax']

import warnings
import numpy as np

from .rt_enum import TypeRegister

# Routines to normalize data for machine learing

def normalize_zscore(arr):
    _mean = arr.nanmean()
    _std = arr.nanstd()
    return (arr - _mean) / _std

def normalize_minmax(arr):
    _min = arr.nanmin()
    _max = arr.nanmax()
    _range = _max - _min
    return (arr - _min) / _range

# TODO seaborn is no longer a dependency, nor imported so this will break when called
# def plot_features(ds, features, row_mask=None):
#     ''' uses seaborn lmplot '''
#     import pandas as pd
#     if row_mask is None:
#         g=ds[features]
#     else:
#         g=ds[features][row_mask,:]
#
#     g['opt']=y_train
#     g['symbol']=foo.pSym[goodsymbol_mask]
#
#     pd_data= pd.DataFrame(g.asdict())
#
#     #sns.relplot(x="date", y="opt",  col="symbol", kind="line", data=pd.DataFrame(g.asdict()))
#     for c in features:
#         sns.lmplot(x=c, y="opt",  col="symbol",  data=pd_data)

