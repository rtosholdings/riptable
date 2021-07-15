"""
rt: RIPTABLE
"""
__author__ = "SIG"
from ._version import __version__

# Cannot simply remove imports w/o resolving the TypeRegister initialization
# at the end of each module.  Note the call to validate_registry()!
# Fundamental?
from .rt_fastarray import FastArray, Threading, Recycle, Ledger
from .rt_struct import Struct
from .rt_ledger import MathLedger
from .rt_dataset import Dataset
from .rt_groupby import GroupBy
from .rt_pgroupby import PGroupBy
from .rt_groupbyops import GroupByOps
from .rt_groupbynumba import GroupbyNumba
from .rt_multiset import Multiset
from .rt_categorical import Categorical, categorical_convert, CatZero
from .rt_accum2 import Accum2
from .rt_accumtable import AccumTable, accum_ratio, accum_ratiop, accum_cols
from .rt_sharedmemory import SharedMemory
from .rt_timezone import TimeZone, Calendar
from .rt_pdataset import PDataset


from .rt_enum import (
    DATETIME_TYPES,
    DS_DISPLAY_TYPES,
    GB_FUNCTIONS,
    MATH_OPERATION,
    INVALID_DICT,
    NumpyCharTypes,
    TIMEWINDOW_FUNCTIONS,
    REDUCE_FUNCTIONS,
    ROLLING_FUNCTIONS,
    SD_TYPES,
    SM_DTYPES,
    TypeRegister,
    DisplayJustification,
    DisplayColumnColors,
    DisplayArrayTypes,
    DisplayDetectModes,
    DisplayLength,
)


TypeRegister.validate_registry()
del TypeRegister
SharedMemory.check_shared_memory_limit()

# Not fundamental yet of first-rank?
from .rt_merge import merge, merge2, merge_asof, merge_lookup

# Second-rank?
from .rt_timers import (
    GetNanoTime,
    GetTSC,
    tic,
    toc,
    ticx,
    tocx,
    tt,
    ttx,
    ticp,
    tocp,
    ticf,
    tocf,
    utcnow,
)
from .Utils.rt_testdata import TestData as td, load_test_data

# Perhaps convenience?
from .rt_numpy import (
    get_dtype,
    get_common_dtype,
    asanyarray,
    asarray,
    abs,
    all,
    any,
    arange,
    argmax,
    argmin,
    argsort,
    bincount,
    bitcount,
    ceil,
    concatenate,
    crc32c,
    crc64,
    cumprod,
    cumsum,
    diff,
    double,
    empty,
    empty_like,
    searchsorted,
    vstack,
    tile,
    repeat,
    floor,
    full,
    groupby,
    groupbyhash,
    groupbylex,
    groupbypack,
    half,
    int16,
    int32,
    int64,
    int8,
    int0,
    uint0,
    bool_,
    isfinite,
    isnotfinite,
    isinf,
    isnotinf,
    ismember,
    isnan,
    isnanorzero,
    isnotnan,
    issorted,
    lexsort,
    logical,
    longdouble,
    max,
    mean,
    median,
    min,
    multikeyhash,
    nan_to_num,
    nan_to_zero,
    nanargmax,
    nanargmin,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanpercentile,
    nanstd,
    nansum,
    nanvar,
    ones,
    ones_like,
    percentile,
    putmask,
    reindex_fast,
    reshape,
    round,
    single,
    sort,
    sortinplaceindirect,
    std,
    sum,
    transpose,
    trunc,
    uint16,
    uint32,
    uint64,
    uint8,
    float32,
    float64,
    bytes_,
    str_,
    unique,
    unique32,
    var,
    where,
    zeros,
    zeros_like,
    mask_and,
    mask_andnot,
    mask_or,
    mask_xor,
    mask_andi,
    mask_andnoti,
    mask_ori,
    mask_xori,
    combine_filter,
    combine_accum1_filter,
    combine_accum2_filter,
    putmask,
    bool_to_fancy,
    interp,
    interp_extrap,
    minimum,
    maximum,
    hstack,
    makeifirst,
    makeilast,
    makeinext,
    makeiprev,
    combine2keys,
    cat2keys,
    assoc_copy,
    assoc_index,
    log,
    log10,
    absolute,
    power,
)

from .rt_grouping import Grouping, hstack_groupings, combine2groups, merge_cats
from .rt_bin import cut, qcut, quantile
from .rt_csv import load_csv_as_dataset
from .rt_hstack import hstack_any, stack_rows
from .rt_display import DisplayTable as Display

from .Utils.rt_display_nested import treedir

from .rt_utils import (
    get_default_value,
    merge_prebinned,
    alignmk,
    normalize_keys,
    bytes_to_str,
    findTrueWidth,
    ischararray,
    islogical,
    mbget,
    str_to_bytes,
    to_str,
    describe,
    h5io_to_struct,
    load_h5,
)

from .rt_sds import (
    compress_dataset_internal,
    decompress_dataset_internal,
    save_struct,
    load_sds,
    save_sds,
    load_sds_mem,
    sds_tree,
    sds_info,
    sds_flatten,
    sds_concat,
    SDSVerboseOn,
    SDSVerboseOff,
)

from .rt_mlutils import normalize_zscore, normalize_minmax

from .rt_misc import (
    output_cache_none,
    output_cache_setsize,
    output_cache_flush,
    profile_func,
    autocomplete,
)
from .rt_datetime import (
    DateTimeNano,
    DateTimeNanoScalar,
    DateTimeBase,
    TimeSpan,
    TimeSpanScalar,
    Date,
    DateScalar,
    DateSpan,
    DateSpanScalar,
    DateBase,
    timestring_to_nano,
    datestring_to_nano,
    datetimestring_to_nano,
    strptime_to_nano,
)

from .rt_str import FAString
from .benchmarks import *

# short hand class names
from numpy import ndarray as NPA
from numpy import nan

import numpy as np

from .rt_fastarray import FastArray as FA
from .rt_fastarraynumba import fill_forward, fill_backward
from .rt_categorical import Categorical as Cat
import riptide_cpp as rc

from .rt_enum import TypeRegister

# matplotlib may not be installed.
try:
    from . import rt_matplotlib
except ModuleNotFoundError:
    pass

# if environment is IPython/Jupyter, force config options here
try:
    # disable outputcache only for ipython by default
    __IPYTHON__
    output_cache_none()

    # turn greedy on for better autocompletions on riptable classes
    ipconfig = get_ipython()
    ipconfig.Completer.greedy = True

    # as of Dec 16, 2019 we let user control jedi
    # if jedi is False, then _def_complete will be called
    # ipconfig.Completer.use_jedi = False

    # as of Jan 15, 2020 we are turning on our autocompletion engine
    autocomplete()

    # If DisplayOptions.CUSTOM_COMPLETION is toggled then use custom attribute completion for riptable data objects.
    from .Utils import ipython_utils
except Exception:
    pass
