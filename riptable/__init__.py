"""
rt: RIPTABLE
"""
__author__ = "SIG"
from ._version import __version__
from .rt_accum2 import Accum2
from .rt_accumtable import AccumTable, accum_cols, accum_ratio, accum_ratiop
from .rt_categorical import Categorical, CatZero, categorical_convert
from .rt_dataset import Dataset
from .rt_enum import (
    DATETIME_TYPES,
    DS_DISPLAY_TYPES,
    GB_FUNCTIONS,
    INVALID_DICT,
    MATH_OPERATION,
    REDUCE_FUNCTIONS,
    ROLLING_FUNCTIONS,
    SD_TYPES,
    SM_DTYPES,
    TIMEWINDOW_FUNCTIONS,
    DisplayArrayTypes,
    DisplayColumnColors,
    DisplayDetectModes,
    DisplayJustification,
    DisplayLength,
    NumpyCharTypes,
    TypeRegister,
)

# Cannot simply remove imports w/o resolving the TypeRegister initialization
# at the end of each module.  Note the call to validate_registry()!
# Fundamental?
from .rt_fastarray import FastArray, Ledger, Recycle, Threading
from .rt_groupby import GroupBy
from .rt_groupbynumba import GroupbyNumba
from .rt_groupbyops import GroupByOps
from .rt_ledger import MathLedger
from .rt_multiset import Multiset
from .rt_pdataset import PDataset
from .rt_pgroupby import PGroupBy
from .rt_sharedmemory import SharedMemory
from .rt_struct import Struct
from .rt_timezone import Calendar, TimeZone

TypeRegister.validate_registry()
del TypeRegister
SharedMemory.check_shared_memory_limit()

import numpy as np
import riptide_cpp as rc

# short hand class names
from numpy import nan
from numpy import ndarray as NPA

from .benchmarks import *
from .rt_bin import cut, qcut, quantile
from .rt_categorical import Categorical as Cat
from .rt_csv import load_csv_as_dataset
from .rt_datetime import (
    Date,
    DateBase,
    DateScalar,
    DateSpan,
    DateSpanScalar,
    DateTimeBase,
    DateTimeNano,
    DateTimeNanoScalar,
    TimeSpan,
    TimeSpanScalar,
    datestring_to_nano,
    datetimestring_to_nano,
    strptime_to_nano,
    timestring_to_nano,
)
from .rt_display import DisplayTable as Display
from .rt_enum import TypeRegister
from .rt_fastarray import FastArray as FA
from .rt_fastarraynumba import fill_backward, fill_forward
from .rt_grouping import Grouping, combine2groups, hstack_groupings, merge_cats
from .rt_hstack import hstack_any, stack_rows

# Not fundamental yet of first-rank?
from .rt_merge import merge, merge2, merge_asof, merge_lookup
from .rt_misc import (
    autocomplete,
    output_cache_flush,
    output_cache_none,
    output_cache_setsize,
    profile_func,
)
from .rt_mlutils import normalize_minmax, normalize_zscore

# Perhaps convenience?
from .rt_numpy import (
    abs,
    absolute,
    all,
    any,
    arange,
    argmax,
    argmin,
    argsort,
    asanyarray,
    asarray,
    assoc_copy,
    assoc_index,
    bincount,
    bitcount,
    bool_,
    bool_to_fancy,
    bytes_,
    cat2keys,
    ceil,
    combine2keys,
    combine_accum1_filter,
    combine_accum2_filter,
    combine_filter,
    concatenate,
    crc32c,
    crc64,
    cumprod,
    cumsum,
    diff,
    double,
    empty,
    empty_like,
    float32,
    float64,
    floor,
    full,
    get_common_dtype,
    get_dtype,
    groupby,
    groupbyhash,
    groupbylex,
    groupbypack,
    half,
    hstack,
    int0,
    int8,
    int16,
    int32,
    int64,
    interp,
    interp_extrap,
    isfinite,
    isinf,
    ismember,
    isnan,
    isnanorzero,
    isnotfinite,
    isnotinf,
    isnotnan,
    issorted,
    lexsort,
    log,
    log10,
    logical,
    longdouble,
    makeifirst,
    makeilast,
    makeinext,
    makeiprev,
    mask_and,
    mask_andi,
    mask_andnot,
    mask_andnoti,
    mask_or,
    mask_ori,
    mask_xor,
    mask_xori,
    max,
    maximum,
    mean,
    median,
    min,
    minimum,
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
    power,
    putmask,
    reindex_fast,
    repeat,
    reshape,
    round,
    searchsorted,
    single,
    sort,
    sortinplaceindirect,
    std,
    str_,
    sum,
    tile,
    transpose,
    trunc,
    uint0,
    uint8,
    uint16,
    uint32,
    uint64,
    unique,
    unique32,
    var,
    vstack,
    where,
    zeros,
    zeros_like,
)
from .rt_sds import (
    SDSVerboseOff,
    SDSVerboseOn,
    compress_dataset_internal,
    decompress_dataset_internal,
    load_sds,
    load_sds_mem,
    save_sds,
    save_struct,
    sds_concat,
    sds_flatten,
    sds_info,
    sds_tree,
)
from .rt_str import FAString

# Second-rank?
from .rt_timers import (
    GetNanoTime,
    GetTSC,
    tic,
    ticf,
    ticp,
    ticx,
    toc,
    tocf,
    tocp,
    tocx,
    tt,
    ttx,
    utcnow,
)
from .rt_utils import (
    alignmk,
    bytes_to_str,
    describe,
    findTrueWidth,
    get_default_value,
    h5io_to_struct,
    ischararray,
    islogical,
    load_h5,
    mbget,
    merge_prebinned,
    normalize_keys,
    str_to_bytes,
    to_str,
)
from .Utils.rt_display_nested import treedir
from .Utils.rt_testdata import TestData as td
from .Utils.rt_testdata import load_test_data

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
