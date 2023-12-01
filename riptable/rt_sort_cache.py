__all__ = [
    "SortCache",
]

import numpy as np

from .rt_enum import TypeRegister
from .rt_numpy import crc64, lexsort


class SortCache(object):
    """
    Global sort cache for uid - unique ids which are often generated from GetTSC (CPU time stamp counter)

    to ensure that the values have not changed underneath, it performs a crc check and compares via the crc of a known sorted index array
    """

    _cache = {}
    _logging = False

    @classmethod
    def logging_on(cls):
        cls._logging = True

    @classmethod
    def logging_off(cls):
        cls._logging = False

    @classmethod
    def store_sort(cls, uid, sortlist, sortidx, ascending):
        """
        Restore a sort index from file.
        """
        crcvals = []
        for c in sortlist:
            crcvals.append(crc64(c))
        cls._cache[uid] = (crcvals, sortidx, len(sortidx), ascending)

    @classmethod
    def get_sorted_row_index(cls, uid, nrows, sortdict, ascending):
        if sortdict is not None and len(sortdict) > 0:
            sortlist = list(sortdict.values())

            # perform a crc on known sorted values and remember the crc
            crcvals = [crc64(vals) for vals in sortlist]

            def _cached():
                if not (uid in cls._cache):
                    return None

                checkvals, sort_idx, checkrows, checkasc = cls._cache[uid]

                # compare all multikey sort values to see if a match
                # check if crc values match
                if (
                    type(checkasc) != type(ascending)
                    or len(checkvals) != len(crcvals)
                    or checkrows != nrows
                    or any(oldcrc != newcrc for oldcrc, newcrc in zip(checkvals, crcvals))
                ):
                    return None

                if isinstance(checkasc, bool):
                    return None if checkasc != ascending else sort_idx

                if len(checkasc) != len(ascending) or any(old != new for old, new in zip(checkasc, ascending)):
                    return None

                return sort_idx

            sort_idx = _cached()

            if sort_idx is None:
                if cls._logging:
                    print("performing lexsort on columns:", list(sortdict.keys()))
                sortlist.reverse()
                sort_idx = lexsort(sortlist, ascending=ascending)
                cls._cache[uid] = (crcvals, sort_idx, nrows, ascending)
            else:
                if cls._logging:
                    print("NOT performing lexsort on columns:", list(sortdict.keys()))

            return sort_idx
        else:
            return None
            # NOTE: arange too costly, disabling this path for now
            # TODO: if nrows under max int32, return 32 bit version to save memory
            # if nrows is None: nrows = 0
            # sort_idx = np.arange(nrows,dtype=np.int64)
            # cls._cache[uid] = ([], sort_idx, nrows)

    @classmethod
    def invalidate(cls, uid):
        if uid in cls._cache:
            del cls._cache[uid]

    @classmethod
    def invalidate_all(cls):
        to_delete = [*cls._cache.keys()]
        for uid in to_delete:
            del cls._cache[uid]


TypeRegister.SortCache = SortCache
