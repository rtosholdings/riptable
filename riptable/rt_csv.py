from typing import Optional, List

__all__ = ['load_csv_as_dataset', ]


import csv
import numpy as np
from .rt_dataset import Dataset


def load_csv_as_dataset(path_or_file, column_names: Optional[List[str]] = None, converters: Optional[dict] = None, skip_rows: int = 0, version: Optional[int] = None, encoding: str = 'utf-8', **kwargs) -> Dataset:
    """
    Load a Dataset from a comma-separated value (CSV) file.

    Parameters
    ----------
    path_or_file
        A filename or a file-like object (from open() or StringIO());
        if you need a non-standard encoding, do the open yourself.
    column_names : list of str, optional
        List of column names (must be legal python var names), or None for
        'use first row read from file'. Defaults to None.
    converters : dict
        {column_name -> str2type-converters}, do your own error handling,
        should return uniform types, and handle bad/missing data as desired
        missing converter will default to 'leave as string'.
    skip_rows : int
        Number of rows to skip before processing, defaults to 0.
    version : int, optional
        Selects the implementation of the CSV parser used to read the input file.
        Defaults to None, in which case the function chooses the best available implementation.
    encoding : str
        The text encoding of the CSV file, defaults to 'utf-8'.
    kwargs
        Any csv 'dialect' params you like.

    Returns
    -------
    Dataset

    Notes
    -----
    For a dataset of shape (459302, 15) (all strings) the timings are roughly:
    (version=0) 6.195947s
    (version=1) 5.605156s (default if pandas not available)
    (version=2) 8.370234s
    (version=3) 6.994191s
    (version=4) 3.642205s (only available if pandas is available, default if so)
    """
    try:
        import pandas as pd
        from .Utils.pandas_utils import dataset_from_pandas_df
    except ImportError:
        pd = None
    if converters is None:
        converters = dict()
    if version is None:
        version = 4 if pd is not None else 1
    if pd is None and version == 4:
        raise RuntimeError('load_csv_as_dataset(version=4) is not allowed if pandas is not available.')
    if version == 4:
        # BUG: pd.read_csv does some sort of import that breaks the unit tester. the csv test succeeds but the next test that runs will raise an error.
        return _load_rows_via_pandas(pd, path_or_file, column_names, converters, skip_rows, encoding=encoding)
    if hasattr(path_or_file, 'read'):
        infile = path_or_file
    else:
        infile = open(path_or_file, 'r', encoding=encoding)
    for _ in range(skip_rows): _ = infile.readline()
    reader = csv.reader(infile, **kwargs)
    if column_names is None or len(column_names) == 0:
        column_names = list(next(reader))
    if not all(_k.isidentifier() for _k in column_names):
        raise ValueError('load_csv_as_dataset: column names must be legal python identifiers')
    if version == 0:
        data = _load_rows_to_dict_conv_by_col(reader, column_names, converters)
    elif version == 1:
        data = _load_rows_to_dict_conv_by_row(reader, column_names, converters)
    elif version == 2:
        data = _load_rows_to_tagged_rows(reader, column_names, converters)
    elif version == 3:
        data = _load_rows_to_rows_and_cols(reader, column_names, converters)
    else:
        raise NotImplementedError('load_csv_as_dataset(version=[0|1|2|3|4]) only.')
    if infile != path_or_file:
        infile.close()
    return data


def _load_rows_to_dict_conv_by_col(reader, column_names, converters):
    # 865ms, all strings
    rawd = [_r for _r in reader]
    data = {}
    for _i, _cname in enumerate(column_names):
        _conv = converters.get(_cname)
        if _conv is None or type(_conv) is str:
            data[_cname] = np.array([_e[_i] for _e in rawd])
        else:
            data[_cname] = np.array([_conv(_e[_i]) for _e in rawd])
    return Dataset(data)


def _load_rows_to_dict_conv_by_row(reader, column_names, converters):
    # 1930ms, all strings
    _ident = lambda _x: _x
    convs = [converters.get(_cname, _ident) for _cname in column_names]
    rawd = [[] for _ in column_names]
    for _r in reader:
        for _v, _c, _l in zip(_r, convs, rawd):
            _l.append(_c(_v))
    data = {_k: np.array(rawd[_i]) for _i, _k in enumerate(column_names)}
    return Dataset(data)


def _load_rows_to_tagged_rows(reader, column_names, converters):
    # 1930ms, all strings
    _ident = lambda _x: _x
    convs = [converters.get(_cname, _ident) for _cname in column_names]
    rawd = []
    for _r in reader:
        rawd.append({_n: _c(_v) for _v, _c, _n in zip(_r, convs, column_names)})
    ds = Dataset.from_tagged_rows(rawd)
    ds.col_move_to_front(column_names)
    return ds


def _load_rows_to_rows_and_cols(reader, column_names, converters):
    # 1930ms, all strings
    _ident = lambda _x: _x
    convs = [converters.get(_cname, _ident) for _cname in column_names]
    rawd = []
    for _r in reader:
        rawd.append([_c(_v) for _v, _c in zip(_r, convs)])
    return Dataset.from_rows(rawd, column_names)


def _load_rows_via_pandas(pd, fname, column_names, converters, skip_rows, encoding='utf-8'):
    if column_names is None or len(column_names) == 0:
        column_names = None
        convs = converters
    else:
        _ident = lambda _x: _x
        convs = {_cname: converters.get(_cname, _ident) for _cname in column_names}
    df = pd.read_csv(fname, converters=convs, names=column_names, skiprows=skip_rows, encoding=encoding)
    return Dataset(df)
