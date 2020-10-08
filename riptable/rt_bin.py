__all__ = ['cut', 'qcut', 'quantile' ]


import numpy as np
import riptide_cpp as rc
from .rt_enum import NumpyCharTypes, TypeRegister, CLIPPED_LONG_NAME
from .rt_numpy import lexsort, unique
from .rt_categorical import Categorical


# ------------------------------------------------------------------------------------
def quantile(x, q, interpolation_method: str='fraction'):
    """
    Compute sample quantile or quantiles of the input array. For example, q=0.5 computes the median.

    The `interpolation_method` parameter supports three values, namely
    `fraction` (default), `lower` and `higher`. Interpolation is done only,
    if the desired quantile lies between two data points `i` and `j`. For
    `fraction`, the result is an interpolated value between `i` and `j`;
    for `lower`, the result is `i`, for `higher` the result is `j`.

    Parameters
    ----------
    x : ndarray
        Values from which to extract score.
    q : scalar or array
        Percentile at which to extract score.
    interpolation_method : {'fraction', 'lower', 'higher'}, optional
        This optional parameter specifies the interpolation method to use,
        when the desired quantile lies between two data points `i` and `j`:
        - fraction: `i + (j - i)*fraction`, where `fraction` is the fractional part of the index surrounded by `i` and `j`.
        - lower: `i`.
        - higher: `j`.

    Returns
    -------
    score : float
        Score at percentile.

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.arange(100)
    >>> stats.scoreatpercentile(a, 50)
    49.5
    """
    if not isinstance(x, np.ndarray):
        x = TypeRegister.FastArray(x)

    # make sure contiguous
    if not x.flags.contiguous:
        x=np.ascontiguousarray(x).view(TypeRegister.FastArray)

    # indirect sort and pass this back...
    # print("lexsort", x, x.shape)
    lsort = lexsort([x])

    # once indirectly sorted, we can count the bad values since inf,-inf,nan move to ends
    counts = rc.NanInfCountFromSort(x, lsort)

    nancount = counts[0]
    infcount = counts[1]
    neginfcount = counts[2]

    #print("**counts", nancount, infcount, neginfcount)

    # now make a window into the section of the array that contains good values
    # windowX = x[neginfcount:(x.shape[0] - (nancount + infcount))]
    windowI = lsort
    datalen = x.shape[0]

    if (neginfcount + nancount + infcount) > 0:
        # count how many good values
        datalen = x.shape[0] - (nancount + infcount + neginfcount)

        #print(nancount, infcount, neginfcount)
        #print("**windowI", windowI)
        windowI = lsort[neginfcount:(x.shape[0] - (nancount + infcount))]
        #print("**windowI", windowI)

    #print("**count result", counts)

    ## TJD NOTE:
    ## do not need to mask -- a sort will put nans at the end
    ##get rid of bad values (like isnan)
    # mask = np.isnan(x)

    # x = x[~mask]

    # print("sorting", x, x.shape[0])

    # values = np.sort(x)
    def _val(index):
        # use the view or window since
        return x[windowI[index]]

    def _interpolate(a, b, fraction):
        """Return the point at the given fraction between a and b, where 'fraction' must be between 0 and 1."""
        return a + (b - a) * fraction

    def _get_score(at):
        if datalen == 0:
            return np.nan

        idx = at * (datalen - 1)
        #print("***idx",idx)
        if idx % 1 == 0:
            score = _val(int(idx))
        else:
            if interpolation_method == 'fraction':
                score = _interpolate(_val(int(idx)), _val(int(idx) + 1), idx % 1)
            elif interpolation_method == 'lower':
                score = _val(np.floor(idx))
            elif interpolation_method == 'higher':
                score = _val(np.ceil(idx))
            else:
                raise ValueError("interpolation_method can only be 'fraction', 'lower' or 'higher'")

        #print("score", score)
        return score

    if np.isscalar(q):
        # they can pass in a number like 4
        return _get_score(q), lsort, counts
    else:
        q = np.asarray(q, np.float64)
        # print("***",q)
        length = q.shape[0]
        i = 0

        result = np.empty(length, dtype=np.object_)
        for i in range(length):
            result[i] = _get_score(q[i])
        return result, lsort, counts



def _cut_result(fac, labels, bins, retbins):
    """Final cut / qcut return."""
    if labels is not False:
        labels = TypeRegister.FastArray(labels)
        grp = TypeRegister.Grouping(fac, labels, base_index=1, ordered=False, sort_display=False, _trusted=True)
        cat = Categorical(grp)
        if not retbins:
            return cat
        return cat, bins

    if not retbins:
        return fac
    return fac, bins


# ------------------------------------------------------------------------------------
def qcut(x, q, labels=True, retbins=False, precision=3, duplicates='raise'):
    """
    Quantile-based discretization function.

    Discretize variable into equal-sized buckets based on rank or based on sample quantiles.
    For example, 1000 values for 10 quantiles would produce a Categorical object indicating
    quantile membership for each data point.

    Parameters
    ----------
    x : 1d ndarray
    q : integer or array of quantiles
        Number of quantiles. 10 for deciles, 4 for quartiles, etc. Alternately,
        array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles
    labels : boolean, array, or None
        Used as labels for the resulting bins. If an array, must be of the same
        length as the resulting bins. If False, returns only integer indicators
        of the bins.  If None or True, the labels are created based on the bins.
        This affects the type of the output container (see below).
    retbins : bool, optional
        Whether to return the (bins, labels) or not.
    precision : int, optional
        The precision at which to store and display the bins labels
    duplicates : {default 'raise', 'drop'}, optional
        If bin edges are not unique, raise ValueError or drop non-uniques.

    Returns
    -------
    out : Categorical or FastArray
        An array-like object representing the respective bin for each value
        of `x`. The type depends on the value of `labels`:
            * False : returns a FastArray of integers
            * array, True, or None : returns a Categorical
    bins : ndarray of floats
        The computed or specified bins. Only returned when `retbins=True`.

    Notes
    -----
    Out of bounds values will be represented as 'Clipped' in the resulting
    Categorical object

    See Also
    --------
    :meth:`~rt.rt_bin.cut` :
        Bin values into discrete intervals.
    :class:`~rt.rt_categorical.Categorical` :
        Array type for storing data that come from a fixed set of values.

    Examples
    --------
    >>> rt.qcut(range(5), 4)
    Categorical([0.0->1.0, 0.0->1.0, 1.0->2.0, 2.0->3.0, 3.0->4.0]) Length: 5
      FastArray([2, 2, 3, 4, 5], dtype=int8) Base Index: 1
      FastArray([b'Clipped', b'0.0->1.0', b'1.0->2.0', b'2.0->3.0', b'3.0->4.0'], dtype='|S8') Unique count: 5

    >>> rt.qcut(range(5), 3, labels=["good", "medium", "bad"])
    Categorical([good, good, medium, bad, bad]) Length: 5
      FastArray([2, 2, 3, 4, 4], dtype=int8) Base Index: 1
      FastArray([b'Clipped', b'good', b'medium', b'bad'], dtype='|S7') Unique count: 4

    >>> rt.qcut(range(5), 4, labels=False)
    FastArray([2, 2, 3, 4, 5], dtype=int8)
    """
    if not isinstance(x, np.ndarray):
        x = TypeRegister.FastArray(x)

    # make sure contiguous
    if not x.flags.contiguous:
        x=np.ascontiguousarray(x).view(TypeRegister.FastArray)

    dtype = x.dtype

    if not isinstance(q, np.ndarray):
        q = np.asarray(q)

    if q.dtype.char in NumpyCharTypes.AllInteger:
        # return evenly spaced numbers  start, stop, num
        quantiles = np.linspace(0, 1, q + 1)
    else:
        quantiles = q

    labels = None if labels is True else labels

    # get the bins on the WINDOWED data
    bins, lsort, counts = quantile(x, quantiles)

    #print("**bins", bins, '**lsort', lsort, "**counts", counts, x, x[lsort])
    fac, bins, ret_labels = _bins_to_cuts_new(x, bins, lsort, counts, labels=labels,
                                          precision=precision, include_lowest=True,
                                          dtype=dtype, duplicates=duplicates)

    labels = ret_labels if labels is not False else labels
    return _cut_result(fac, labels, bins, retbins)


# ------------------------------------------------------------------------------------
def _round_frac(x, precision):
    """Round the fractional part of the given number."""
    if not np.isfinite(x) or x == 0:
        return x
    else:
        frac, whole = np.modf(x)
        if whole == 0:
            digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
        else:
            digits = precision
        return np.around(x, digits)


# ------------------------------------------------------------------------------------
def _infer_precision(base_precision, bins):
    """Infer an appropriate precision for _round_frac."""
    for precision in range(base_precision, 20):
        levels = [_round_frac(b, precision) for b in bins]
        if np.unique(levels).size == bins.size:
            return precision
    return base_precision  # default


# ------------------------------------------------------------------------------------
# will always create a clipped
def _format_labels(bins, precision, right=True,
                   include_lowest=False, dtype=None, clipped=False):
    """Based on the dtype, return our labels."""
    closed = 'right' if right else 'left'

    precision = _infer_precision(precision, bins)
    formatter = lambda x: _round_frac(x, precision)
    adjust = lambda x: x - 10 ** (-precision)

    breaks = [formatter(b) for b in bins]
    # print("**breaks:", breaks)


    if clipped:
        # first label is always clipped
        labels = [CLIPPED_LONG_NAME]
    else:
        labels = []

    for i in range(1, len(breaks)):
        #labels.append(str(breaks[i - 1]) + "\U00002192" + str(breaks[i]))
        #labels.append(str(breaks[i - 1]) + "\U000021E8" + str(breaks[i]))
        #labels.append(str(breaks[i - 1]) + "\U000027A1" + str(breaks[i]))
        labels.append(str(breaks[i - 1]) + "->" + str(breaks[i]))

    # labels = IntervalIndex.from_breaks(breaks, closed=closed)

    # if right and include_lowest:
    #    # we will adjust the left hand side by precision to
    #    # account that we are all right closed
    #    v = adjust(labels[0].left)

    #    i = IntervalIndex([Interval(v, labels[0].right, closed='right')])
    #    labels = i.append(labels[1:])

    return labels


# ------------------------------------------------------------------------------------
# TJD: Needs to be rewritten
# windowX == view into valid X data
# windowI == view into sorted indexing of X
def _bins_to_cuts_new(x, bins, lsort=None, counts=None, right=True, labels=None,
                      precision=3, include_lowest=False,
                      dtype=None, duplicates='raise'):
    if duplicates not in ['raise', 'drop']:
        raise ValueError("invalid value for 'duplicates' parameter, valid options are: raise, drop")

    # TOD=O: if sorted, can take fast path
    unique_bins = unique(bins)

    #print("after unique", bins, unique_bins)

    if len(unique_bins) < len(bins) != 2:
        if duplicates == 'raise':
            raise ValueError(
                "Bin edges must be unique: {bins!r}.\nYou can drop duplicate edges by setting the 'duplicates' kwarg".format(
                    bins=bins))
        else:
            bins = unique_bins

    # right then mode =0, else mode =1
    mode = 0 if right else 1

    bins = bins.astype(np.float64)

    if lsort is not None:
        ids = rc.BinsToCutsSorted(x, bins, lsort, counts, mode)
        #print("bins sorted",ids, x, bins)
    else:
        # check for int compaing to inf, if so force upcast
        if (bins[0]==-np.inf or bins[-1]==np.inf):
            x=x.astype(np.float64)
        ids = rc.BinsToCutsBSearch(x, bins, mode)
        #print("bins bsearch",ids, x, bins)

    # if include_lowest:
    #    # Numpy 1.9 support: ensure this mask is a Numpy array
    #    ids[np.asarray(x == bins[0])] = 1

    if labels is not False:
        if labels is None:
            labels = _format_labels(bins, precision, right=right,
                                    include_lowest=include_lowest,
                                    dtype=dtype, clipped=lsort is not None)
        else:
            if len(labels) != len(bins) - 1:
                raise ValueError('Bin labels must be one fewer than the number of bin edges')

            # first label is always clipped
            if lsort is not None:
                temp = [CLIPPED_LONG_NAME]
                labels = temp + labels

        # if not is_categorical_dtype(labels):
        #    labels = Categorical(labels, categories=labels, ordered=True)

        # np.putmask(ids, na_mask, 0)
        # result = algos.take_nd(labels, ids - 1)

        result = ids

        # TJD Not sure?
        # make a categorical from result
        return result, bins, labels

    else:
        result = ids

    return result, bins, None


# ------------------------------------------------------------------------------------
def cut(x, bins, labels=True, right=True, retbins=False, precision=3,
        include_lowest=False):
    """
    Bin values into discrete intervals.

    Use `cut` when you need to segment and sort data values into bins. This
    function is also useful for going from a continuous variable to a
    categorical variable. For example, `cut` could convert ages to groups of
    age ranges. Supports binning into an equal number of bins or a
    pre-specified array of bins.

    Parameters
    ----------
    x : array-like
        The input array to be binned. Must be 1-dimensional.
    bins : int, sequence of scalars
        The criteria to bin by:
            * int: Defines the number of equal-width bins in the range of `x`.
              The range of `x` is extended by .1% on each side to include the minimum
              and maximum values of `x`.
            * sequence of scalars: Defines the bin edges allowing for non-uniform width.
              No extension of the range of `x` is done.
    right : bool, default True
        Indicates whether `bins` includes the rightmost edge or not. If
        ``right == True`` (the default), then the `bins` ``[1, 2, 3, 4]``
        indicate (1,2], (2,3], (3,4].
    labels : boolean, array, or None
        Specifies the labels for the returned bins. If an array, must be the same
        length as the resulting bins. If False, returns only integer indicators of the
        bins. If None or True, the labels are created based on the bins.
        This affects the type of the output container (see below).
    retbins : bool, default False
        Whether to return the bins or not. Useful when bins is provided
        as a scalar.
    precision : int, default 3
        The precision at which to store and display the bins labels.
    include_lowest : bool, default False
        Whether the first interval should be left-inclusive or not.

    Returns
    -------
    out : Categorical or FastArray
        An array-like object representing the respective bin for each value
        of `x`. The type depends on the value of `labels`:
            * False : returns a FastArray of integers
            * array, True, or None : returns a Categorical
    bins : ndarray of floats
        The computed or specified bins. Only returned when `retbins=True`.

    See Also
    --------
    :meth:`~rt.rt_bin.qcut` :
        Discretize variable into equal-sized buckets based on rank or based on sample quantiles.
    :class:`~rt.rt_categorical.Categorical` :
        Array type for storing data that come from a fixed set of values.

    Examples
    --------
    Discretize into three equal-sized bins.

    >>> rt.cut(np.array([1, 7, 5, 4, 6, 3]), 3)
    Categorical([1.0->3.0, 5.0->7.0, 3.0->5.0, 3.0->5.0, 5.0->7.0, 1.0->3.0]) Length: 6
      FastArray([1, 3, 2, 2, 3, 1], dtype=int8) Base Index: 1
      FastArray([b'1.0->3.0', b'3.0->5.0', b'5.0->7.0'], dtype='|S8') Unique count: 3

    Also return the bins.

    >>> rt.cut(np.array([1, 7, 5, 4, 6, 3]), 3, retbins=True)
    (Categorical([1.0->3.0, 5.0->7.0, 3.0->5.0, 3.0->5.0, 5.0->7.0, 1.0->3.0]) Length: 6
      FastArray([1, 3, 2, 2, 3, 1], dtype=int8) Base Index: 1
      FastArray([b'1.0->3.0', b'3.0->5.0', b'5.0->7.0'], dtype='|S8') Unique count: 3,
     array([1., 3., 5., 7.]))

    Return just an array of integers.

    >>> rt.cut(np.array([1, 7, 5, 4, 6, 3]), 3, labels=False)
    FastArray([1, 3, 2, 2, 3, 1], dtype=int8)

    Discovers the same bins, but assign them specific labels. Notice that
    the returned Categorical's categories are `labels` and it is ordered.

    >>> rt.cut(np.array([1, 7, 5, 4, 6, 3]),
    ...        3, labels=["bad", "medium", "good"])
    Categorical([bad, good, medium, medium, good, bad]) Length: 6
      FastArray([1, 3, 2, 2, 3, 1], dtype=int8) Base Index: 1
      FastArray([b'bad', b'medium', b'good'], dtype='|S6') Unique count: 3
    """
    # NOTE: this binning code is changed a bit from histogram for var(x) == 0

    # for handling the cut for datetime and timedelta objects
    # x_is_series, series_index, name, x = _preprocess_for_cut(x)

    if not isinstance(x, np.ndarray):
        x = TypeRegister.FastArray(x)

    # make sure contiguous
    if not x.flags.contiguous:
        x=np.ascontiguousarray(x).view(TypeRegister.FastArray)

    if not np.iterable(bins):
        if np.isscalar(bins) and bins < 1:
            raise ValueError("`bins` should be a positive integer.")

        if x.size == 0:
            raise ValueError('Cannot cut empty array')

        # todo range function
        rng = (0, 0)
        if x.dtype.char in NumpyCharTypes.Noncomputable:
            rng = (np.nanmin(x), np.nanmax(x))
        else:
            x = x.view(TypeRegister.FastArray)
            rng = (x.nanmin(), x.nanmax())

        mn, mx = [mi + 0.0 for mi in rng]

        # TJD Note: the 0.001 is from pandas and not sure if needed
        tolerance = 0.001

        if mn == mx:  # adjust end points before binning
            mn -= tolerance * abs(mn) if mn != 0 else tolerance
            mx += tolerance * abs(mx) if mx != 0 else tolerance
            bins = np.linspace(mn, mx, bins + 1, endpoint=True)
        else:  # adjust end points after binning
            bins = np.linspace(mn, mx, bins + 1, endpoint=True)
            adj = (mx - mn) * tolerance  # 0.1% of the range

    else:
        bins = np.asarray(bins)
        # bins = _convert_bin_to_numeric_type(bins, dtype)
        if (np.diff(bins) < 0).any():
            raise ValueError('bins must increase monotonically.')

    dtype = x.dtype

    labels = None if labels is True else labels

    fac, bins, ret_labels = _bins_to_cuts_new(x, bins, right=right, labels=labels,
                                          precision=precision,
                                          include_lowest=include_lowest,
                                          dtype=dtype)

    labels = ret_labels if labels is not False else labels

    # if they want labels, make a categorical

    return _cut_result(fac, labels, bins, retbins)
