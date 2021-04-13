"""
Helper functions for generating random data for use in implementing unit tests for riptable.
"""
__all__ = [
    'create_test_dataset'
]

import datetime as dt

import numpy as np
from numpy.random import Generator

from .. import Categorical, Dataset, Date, DateTimeNano, TimeSpan


def create_test_dataset(
    *,
    rng: Generator,
    rowcount: int,
    include_dict_cat: bool = False,
    include_date: bool = False,
    include_datetime_nano: bool = True,
    include_timespan: bool = True
) -> Dataset:
    test_ds = Dataset()

    # Add deliverable_symbol, drawn as a subset of a few (say, 3-5) symbols from a larger set of symbols.
    # Tile it, truncate to the desired rowcount, then turn it into a Categorical
    # TODO: Add more symbols to this list for better testing; the symbols themselves don't matter,
    #       it's more of a set theory thing -- we want to get coverage around cases where there's
    #       (no, partial, complete) overlap in symbols between sections, etc.
    deliv_symbols = np.array(['CBO', 'CBX', 'ZBZX', 'ZJZZT', 'ZTEST', 'ZVV', 'ZVZZT', 'ZWZZT', 'ZXZZT'])
    symbol_count = 3
    # Choose N unique symbols
    attempts = 100
    symbol_idxs = None
    for _ in range(attempts):
        # Generate N random indices in the range of the symbol set.
        rand_idxs = rng.integers(0, len(deliv_symbols), size=symbol_count)

        # If the generated indices are all unique, save this array and break out of the loop.
        if len(np.unique(rand_idxs)) == symbol_count:
            symbol_idxs = rand_idxs
            break

    # If we couldn't get a unique set of indices, we can't proceed.
    if symbol_idxs is None:
        raise Exception("Unable to draw a unique set of indices.")

    # Draw a fancy index (that we'll then use to fancy-index our symbol indices) from a distribution that makes
    # some elements more likely to appear than others.
    bucket_weights = np.arange(symbol_count + 1, 1, -1)
    # TODO: If we generalize this to a function, add a check here for negative bucket weights
    total_bucket_weight = np.sum(bucket_weights)
    cum_bucket_weights = bucket_weights.cumsum(dtype=np.min_scalar_type(total_bucket_weight))
    # 'endpoint=False' here because we specify 'side=right' for searchsorted below.
    # TODO: This won't be correct if we have any zero-weighted buckets, it could mean we draw from a zero-weighted bucket
    #       instead of an adjacent non-zero-weighted bucket.
    samples = rng.integers(0, total_bucket_weight, endpoint=False, size=rowcount)
    sampled_buckets = np.searchsorted(cum_bucket_weights, samples, side='right')

    # Get the set of unique symbols then create the Categorical column.
    # N.B. If desired, it'd be a fairly simple change to randomly include more of the symbols
    #      so they'll be included in the Categorical but not actually be used -- might be useful
    #      for testing loading/hstacking.
    test_ds.deliverable_symbol = Categorical(sampled_buckets, categories=deliv_symbols[symbol_idxs])

    # Create a dictionary-mode Categorical (from ISO3166 data).
    if include_dict_cat:
        category_dict = {
            'IRL': 372, 'USA': 840, 'AUS': 36, 'HKG': 344, 'JPN': 392,
            'MEX': 484, 'KHM': 116, 'THA': 764, 'JAM': 388, 'ARM': 51
        }
        # The values for the Categorical's backing array.
        # This includes some value(s) not in the dictionary and not all values in the dictionary are used here.
        cat_values = [36, 36, 344, 840, 840, 372, 840, 372, 840, 124, 840, 124, 36, 484]
        # calculate cat length
        tile_num = np.math.ceil(rowcount / len(cat_values))
        cat_values = np.tile(cat_values, tile_num)[:rowcount]
        test_ds.country = Categorical(cat_values, categories=category_dict)

    # Add dates and time spans from rowcount days ago
    offset_day_range = 1_095  # offset day range by three years
    offset_days = rng.integers(-offset_day_range, offset_day_range)
    start_date = dt.date.today() + dt.timedelta(days=int(offset_days - rowcount))
    end_date = start_date + dt.timedelta(days=rowcount)

    np_date = np.arange(start_date, end_date, dt.timedelta(days=1)).astype('<M8[D]')
    dtns = DateTimeNano(np_date, from_tz='GMT')  # TODO: exercise other timezones

    if include_datetime_nano:
        test_ds.datetime_nano = dtns
        # TODO enable DTN categoricals after resolving rt.Cat inconsistencies
        # test_ds.datetime_nano_cat = dtns_cat = rt.Categorical(test_ds.datetime_nano)
        # dtns_cat[0] = dtns_cat[1] = dtns_cat[rowcount-1] = 0  # set invalids

    if include_date:
        test_ds.np_date = np_date
        test_ds.date = Date(test_ds.np_date)
        test_ds.date_cat = Categorical(test_ds.date)

    if include_timespan:
        test_ds.time_span = TimeSpan([dtns[i - 1] - dtns[1] for i, _ in enumerate(dtns)])
        # TODO enable TS categoricals after resolving rt.Cat inconsistencies
        # test_ds.time_span_cat = ts_cat = rt.Categorical(test_ds.time_span)
        # ts_cat[0] = ts_cat[1] = ts_cat[rowcount-1] = 0  # set invalids

    # Add opening price, drawn from non-negative floats.
    test_ds.open_price = rng.uniform(10.0, 1000.0, size=rowcount)

    # Add closing price; draw from lognormal distribution then multiply with opening price (and max(..., 0)).
    test_ds.close_price = np.maximum(
        0.0,
        test_ds.open_price * rng.lognormal(sigma=0.04, size=rowcount)
    )

    # Add a volume column as uint32
    test_ds.volume = rng.integers(0, 10_000_000, size=rowcount, dtype=np.uint32)

    return test_ds

