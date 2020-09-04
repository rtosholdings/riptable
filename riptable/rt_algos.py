from typing import List
import numpy as np

from .rt_enum import TypeRegister
from .rt_numpy import unique, ismember, hstack
from .rt_utils import crc_match


# ------------------------------------------------------------
def merge_index(indices, listcats, idx_cutoffs=None, unique_cutoffs=None, from_mapping=False, stack=True):
    """
    For hstacking Categoricals or fixing indices in a categorical from a stacked .sds load.

    Supports categoricals from single array or dictionary mapping.

    Parameters
    ----------
    indices : single stacked array or list of indices
        if single array, needs idx_cutoffs for slicing
    listcats : list
        list of stacked unique category arrays (needs unique_cutoffs) or list of lists of uniques
    idx_cutoffs
        (TODO)
    unique_cutoffs
        (TODO)
    from_mapping : bool
        (TODO)
    stack : bool
        (TODO)

    Returns
    -------
    Tuple containing two items:
    - list of fixed indices, or array of fixed contiguous indices.
    - stacked unique values
    """
    # ------------------------------------------------------------
    def index_fixups(oldcats, newcats):
        # funnel for ismember in merge_index
        fixups = [ismember(oc, newcats, base_index=1)[1] for oc in oldcats]
        fixups = [ hstack([TypeRegister.FastArray([0]), f]) for f in fixups ]
        return fixups
    # ------------------------------------------------------------

    if unique_cutoffs is not None:
        oldcats = []
        # all combined uniques will have the same cutoff points
        unique_cutoffs = unique_cutoffs[0]
        match = True
        for cat_array in listcats:
            oc = []
            start = 0
            for end in unique_cutoffs:
                oc.append(cat_array[start:end])
                start = end
            # build list of slices
            oldcats.append(oc)
    # single list of unique categories (not from slices)
    else:
        oldcats = [listcats]

    # check to see if all categories were the same
    match = True
    for oc in oldcats:
        match = crc_match(oc)
        if not match:
            break

    if match:
        # first uniques are the same for all
        newcats = [oc[0] for oc in oldcats]
        if from_mapping:
            newcats[1] = newcats[1].astype('U', copy=False)
        else:
            newcats = newcats[0]
        return indices, newcats


    if from_mapping:
        # listcats is two arrays:
        # the first is the combined uniques of codes
        # the second is the combined uniques of names
        codes, uidx = unique(listcats[0], return_index=True, sorted=False)
        names = listcats[1][uidx].astype('U', copy=False)
        newcats = [codes, names]
        # use first occurance of codes to get uniques for both codes and names
        return indices, newcats

    # need to perform own hstack
    # this will get hit by Categorical.hstack() for single/multikey
    # nothing has been stacked
    if unique_cutoffs is None:
        newcats = [ unique(hstack(oc)) for oc in oldcats ][0]
        oldcats = oldcats[0]
        fixups = index_fixups(oldcats, newcats)

        # use masks
        indices = [ fixups[i][c] for i, c in enumerate(indices) ]

        return indices, newcats

    # otherwise, already stacked
    else:
        # TODO: fix for multikey unique
        newcats = [ unique(oc) for oc in listcats ][0]
        oldcats = oldcats[0]
        fixups = index_fixups(oldcats, newcats)

        #use slices
        start = 0
        for i, end in enumerate(idx_cutoffs):
            indices[start:end] = fixups[i][indices[start:end]]
            start = end

        return indices, newcats
