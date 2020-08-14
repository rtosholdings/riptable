__all__ = ['SortCache', ]

import numpy as np

from .rt_numpy import lexsort, crc64
from .rt_enum import TypeRegister

class SortCache(object):
    '''
    Global sort cache for uid - unique ids which are often generated from GetTSC (CPU time stamp counter)

    to ensure that the values have not changed underneath, it performs a crc check and compares via the crc of a known sorted index array
    '''
    _cache = {}
    _logging = False

    @classmethod
    def logging_on(cls):
        cls._logging=True

    @classmethod
    def logging_off(cls):
        cls._logging=False

    @classmethod
    def store_sort(cls, uid, sortlist, sortidx):
        '''
        Restore a sort index from file.
        '''
        crcvals = []
        for c in sortlist:
            crcvals.append(crc64(c))
        cls._cache[uid] = (crcvals, sortidx, len(sortidx))

    @classmethod
    def get_sorted_row_index(cls, uid, nrows, sortdict):
        if sortdict is not None and len(sortdict) > 0:
            crcvals=[]
            sortlist=list(sortdict.values())

            for vals in sortlist:
                # perform a crc on known sorted values and remember the crc
                crcvals.append(crc64(vals))

            updateCache=True

            sort_idx = None

            if uid in cls._cache:
                checkvals, sort_idx, checkrows  = cls._cache[uid]

                if len(checkvals) == len(crcvals) and checkrows == nrows:
                    updateCache = False

                    # compare all multikey sort values to see if a match
                    for i in range(len(checkvals)):
                        if checkvals[i] != crcvals[i]:
                            updateCache = True
                            break

            if updateCache:    
                if cls._logging: print("performing lexsort on columns:",list(sortdict.keys()))
                sortlist.reverse()
                sort_idx = lexsort(sortlist)
                cls._cache[uid] = (crcvals, sort_idx, nrows)
            else:
                if cls._logging: print("NOT performing lexsort on columns:",list(sortdict.keys()))

            return sort_idx

        else:
            return None
            # NOTE: arange too costly, disabling this path for now
            # TODO: if nrows under max int32, return 32 bit version to save memory
            #if nrows is None: nrows = 0
            #sort_idx = np.arange(nrows,dtype=np.int64)
            #cls._cache[uid] = ([], sort_idx, nrows)

        return sort_idx

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

