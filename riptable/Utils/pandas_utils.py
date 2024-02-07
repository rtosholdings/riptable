"""
Utility function for rt.
These functions (may) have dependence on additional libraries and therefore should
_NOT_ be imported in __init__.py or any other such core (like rt_appconfig.py).
"""

__all__ = [
    "dataset_as_pandas_df",
    "dataset_from_pandas_df",
    "fastarray_to_pandas_series",
    "pandas_series_to_riptable",
]

import warnings
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    # pandas is an optional dependency.
    try:
        import pandas as pd
    except ImportError:
        pass

from .. import TypeRegister, INVALID_DICT
from ..rt_dataset import Dataset
from ..rt_enum import CategoryMode


def dataset_from_pandas_df(df, tz="UTC"):
    """
    This function is deprecated, please use riptable.Dataset.from_pandas.

    Creates a riptable Dataset from a pandas DataFrame. Pandas categoricals
    and datetime arrays are converted to their riptable counterparts.
    Any timezone-unaware datetime arrays (or those using a timezone not
    recognized by riptable) are localized to the timezone specified by the
    tz parameter.

    Recognized pandas timezones:
        UTC, GMT, US/Eastern, and Europe/Dublin

    Parameters
    ----------
    df: DataFrame
        The pandas DataFrame to be converted
    tz: string
        A riptable-supported timezone ('UTC', 'NYC', 'DUBLIN', 'GMT')

    Returns
    -------
    Dataset

    See Also
    --------
    riptable.Dataset.from_pandas
    riptable.Dataset.to_pandas
    """
    warnings.warn(
        "dataset_from_pandas_df is deprecated and will be removed in future release, "
        "please use riptable.Dataset.from_pandas method",
        FutureWarning,
        stacklevel=2,
    )
    return Dataset.from_pandas(df, tz)


def dataset_as_pandas_df(ds):
    """
    This function is deprecated, please use riptable.Dataset.as_pandas_df method.

    Create a pandas DataFrame from a riptable Dataset.
    Will attempt to preserve single-key categoricals, otherwise will appear as
    an index array. Any bytestrings will be converted to unicode.

    Parameters
    ----------
    ds : Dataset
        The riptable Dataset to be converted.

    Returns
    -------
    DataFrame

    See Also
    --------
    riptable.Dataset.to_pandas
    """
    warnings.warn(
        "dataset_as_pandas_df is deprecated and will be removed in future release, "
        "please use riptable.Dataset.to_pandas method",
        FutureWarning,
        stacklevel=2,
    )
    return ds.to_pandas()


def pandas_series_to_riptable(series: Union["pd.Series", "pd.Categorical"], tz: str = "UTC") -> TypeRegister.FastArray:
    import pandas as pd

    dtype = series.dtype
    dtype_kind = dtype.kind
    if hasattr(pd, "CategoricalDtype"):
        iscat = isinstance(dtype, pd.CategoricalDtype)
    else:
        iscat = dtype.num == 100

    iscat = iscat or isinstance(series, pd.Categorical)

    if iscat:
        if isinstance(series, pd.Categorical):
            cat = series
        else:
            cat = series.cat

        codes = cat.codes
        categories = cat.categories

        # check for newer version of pandas
        if hasattr(codes, "to_numpy"):
            codes = codes.to_numpy()
            categories = categories.to_numpy()
        else:
            codes = np.asarray(codes)
            categories = np.asarray(categories)

        return TypeRegister.Categorical(codes + 1, categories=categories)
    elif hasattr(pd, "Int8Dtype") and isinstance(
        dtype,
        (
            pd.Int8Dtype,
            pd.Int16Dtype,
            pd.Int32Dtype,
            pd.Int64Dtype,
            pd.UInt8Dtype,
            pd.UInt16Dtype,
            pd.UInt32Dtype,
            pd.UInt64Dtype,
        ),
    ):
        sentinel = INVALID_DICT[dtype.numpy_dtype.num]
        return TypeRegister.FastArray(series.fillna(sentinel), dtype=dtype.numpy_dtype)
    elif dtype_kind == "M":
        try:
            _tz = str(dtype.tz)
        except AttributeError:
            _tz = tz
        return TypeRegister.DateTimeNano(np.asarray(series, dtype="i8"), from_tz="UTC", to_tz=_tz)
    elif dtype_kind == "m":
        arr = TypeRegister.FastArray(series, dtype="i8")
        arr = TypeRegister.TimeSpan(arr)
        arr[arr == pd.NaT.value] = arr.inv
        return arr
    elif dtype_kind == "O":
        if len(series) > 0:
            notnull = np.where(series.notnull())[0]
            all_null = len(notnull) == 0
            first_element = np.nan if all_null else series.iloc[notnull[0]]
            if isinstance(first_element, (int, float, np.number)):
                # An object array with number (int or float) in it probably means there is
                # NaN in it so convert to float64.
                new_col = np.asarray(series, dtype="f8")
            else:
                try:
                    new_col = np.asarray(series, dtype="S")
                except UnicodeEncodeError:
                    new_col = np.asarray(series, dtype="U")
        else:
            new_col = np.asarray(series, dtype="S")
        return TypeRegister.FastArray(new_col)
    else:
        return TypeRegister.FastArray(series)


def fastarray_to_pandas_series(
    arr: Union[TypeRegister.FastArray, TypeRegister.Categorical], unicode: bool = True, use_nullable: bool = True
) -> "pd.Series":
    import pandas as pd

    def _to_unicode_if_string(arr):
        if arr.dtype.char == "S":
            arr = arr.astype("U")
        return arr

    dtype = arr.dtype
    if isinstance(arr, TypeRegister.Categorical):
        if arr.category_mode in (CategoryMode.Default, CategoryMode.StringArray, CategoryMode.NumericArray):
            pass  # already compatible with pandas; no special handling needed
        elif arr.category_mode in (CategoryMode.Dictionary, CategoryMode.MultiKey, CategoryMode.IntEnum):
            # Pandas does not have a notion of a IntEnum, Dictionary, and Multikey category mode.
            # Encode dictionary codes to a monotonically increasing sequence and construct
            # pandas Categorical as if it was a string or numeric array category mode.
            old_category_mode = arr.category_mode
            arr = arr.as_singlekey()
            warnings.warn(
                f"Series  converted from {repr(CategoryMode(old_category_mode))} to {repr(CategoryMode(arr.category_mode))}.",
                stacklevel=2,
            )
        else:
            raise NotImplementedError(
                f"Dataset.to_pandas: Unhandled category mode {repr(CategoryMode(arr.category_mode))}"
            )

        base_index = 0 if arr.base_index is None else arr.base_index
        codes = np.asarray(arr) - base_index
        categories = _to_unicode_if_string(arr.category_array) if unicode else arr.category_array
        out = pd.Series(pd.Categorical.from_codes(codes, categories=categories))
    elif isinstance(arr, TypeRegister.DateTimeNano):
        utc_datetime = pd.DatetimeIndex(arr._np, tz="UTC")
        tz_datetime = utc_datetime.tz_convert(arr._timezone._to_tz)
        out = pd.Series(tz_datetime)
    elif isinstance(arr, TypeRegister.TimeSpan):
        out = pd.Series(arr._np, dtype="timedelta64[ns]")
    # TODO: riptable.DateSpan doesn't have a counterpart in pandas, what do we want to do?
    elif use_nullable and np.issubdtype(dtype, np.integer):
        # N.B. Has to use the same dtype for `isin` otherwise riptable will convert the dtype
        #      and the invalid value.
        is_invalid = arr.isin(TypeRegister.FastArray([INVALID_DICT[dtype.num]], dtype=dtype))
        # N.B. Have to make a copy of the array to numpy array otherwise pandas seg
        #      fault in DataFrame.
        # NOTE: not all versions of pandas have pd.arrays
        if hasattr(pd, "arrays"):
            arr = pd.arrays.IntegerArray(np.array(arr), mask=is_invalid)
        else:
            arr = np.array(arr)
        out = pd.Series(arr)
    elif unicode and arr.dtype.char == "S":
        out = pd.Series(arr.astype("U"))
    else:
        out = pd.Series(arr)

    out.name = getattr(arr, "_name", None)
    return out
