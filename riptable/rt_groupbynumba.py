__all__ = ['GroupbyNumba' ]

import numpy as np
import numba as nb
from .rt_fastarray import FastArray
from .rt_numpy import empty_like, empty
from .rt_enum import TypeRegister, GB_FUNC_NUMBA, GB_PACKUNPACK, GB_FUNCTIONS, INVALID_DICT
from .rt_groupbyops import GroupByOps

# NOTE YOU MUST INSTALL tbb
# conda install tbb
# to confirm...
# >>> from numba import threading_layer()
# >>> threading_layer()
# >>> 'tbb'
#
# See Table at end
#-------------------------------------------------------------------------------------------------
@nb.jit(nopython=True, cache=True)
def build_core_list(cores, unique_rows, binLow, binHigh):
        dividend = unique_rows // cores
        remainder = unique_rows % cores
        high =0
        low =0

        for i in range(cores):
            # Calculate band range
            high = low + dividend

            # add in any remainder until nothing left
            if (remainder > 0):
                high+=1
                remainder-=1

            binLow[i] = low
            binHigh[i] = high

            # next low bin is the previous high bin
            low = high

class GroupbyNumba(GroupByOps):

    # how many cores to cap the computation at
    # NOTE: this is not how many cores the system has but a number where
    # we believe thrashing takes place.  This number could be dynamic per algo in the future.
    CORE_COUNT = 12

    #-------------------------------------------------------------------------------------------------
    def _nb_groupbycalculateall(
        values, 
        ikey, 
        unique_rows, 
        funcList, 
        binLowList, 
        binHighList, 
        func_param):
        results= []

        unique_rows += 1
        corecount = min (GroupbyNumba.CORE_COUNT, unique_rows)

        binLow = np.empty(corecount, dtype=np.int32)
        binHigh = np.empty(corecount, dtype=np.int32)

        build_core_list(corecount, unique_rows, binLow, binHigh)
        
        for funcnum, inputdata in zip(funcList, values):
            nbrev = NUMBA_REVERSE_TABLE[funcnum]

            # lookup function to call
            nbfunc = nbrev['func_back']

            # lookup return dtype requested
            dtypefunc = nbrev['func_dtype']
            if dtypefunc is None:
                # output is same dtype as input
                dtype = inputdata.dtype
            else:
                dtype = dtypefunc(inputdata.dtype)

            # allocate for numba 
            ret = empty( unique_rows, dtype=dtype)

            nbfunc(ikey, unique_rows, binLow, binHigh, inputdata, ret, *func_param)
            results.append(ret)
        return results


    #-------------------------------------------------------------------------------------------------
    # This routine is called before the numba routines
    def _nb_groupbycalculateallpack(
        values,     # list of arrays (the data to be calculated)
        ikey,       # bin numbers (integer array)
        iGroup,     # used to go over
        iFirstGroup, 
        nCountGroup, 
        unique_rows, # often the same as len(iFirstGroup)
        funcList,    # support aggregation
        binLowList,  # start bin to work on for prange
        binHighList, # high bin to work on for prange
        func_param): # parameters

        results= []

        # TODO: add enumerate here
        for funcnum, inputdata in zip(funcList, values):
            nbrev = NUMBA_REVERSE_TABLE[funcnum]

            # lookup function to call
            nbfunc = nbrev['func_back']

            # lookup return dtype requested
            dtypefunc = nbrev['func_dtype']
            if dtypefunc is None:
                # output is same dtype as input
                dtype = inputdata.dtype
            else:
                dtype = dtypefunc(inputdata.dtype)

            # allocate for numba 
            ret = empty( len(inputdata), dtype=dtype)

            #print("sending data", inputdata)
            #print("binlow", binLowList[0])
            nbfunc(iGroup, iFirstGroup, nCountGroup, binLowList[0], binHighList[0], inputdata, ret, *func_param)
            results.append(ret)
        return results
    
    #-------------------------------------------------------------------------------------------------
    @nb.jit(parallel=True, nopython=True, cache=True)
    def _numbasum(ikey, unique_rows, binLow, binHigh, data, ret):
        datacount = len(ikey)

        # binLow and binHigh are arrays (same length)
        # they divide up the work for prange while also allowing selective group filtering
        for core in nb.prange(len(binLow)):
            binlow = binLow[core]
            binhigh = binHigh[core]

            # zero out summation counters before we begin
            for i in range(binlow, binhigh):
                ret[i] = 0

            # concurrently loop over all the data
            for index in range(datacount):
                grpIdx = ikey[index]

                # make sure assigned to our range (concurrency issue)
                if grpIdx >= binlow and grpIdx < binhigh:
                    ret[grpIdx] += data[index]

    #-------------------------------------------------------------------------------------------------
    @nb.jit(parallel=True, cache=True)
    def _numbamin(ikey, unique_rows, binLow, binHigh, data, ret):
        inv = INVALID_DICT[ret.dtype.num]
        datacount = len(ikey)

        for core in nb.prange(len(binLow)):
            binlow = binLow[core]
            binhigh = binHigh[core]

            # mark all initial values as invalid we begin
            for i in range(binlow, binhigh):
                ret[i] = inv

            # concurrently loop over all the data
            for index in range(datacount):
                grpIdx = ikey[index]

                # make sure assigned to our range (concurrency issue)
                if grpIdx >= binlow and grpIdx < binhigh:
                    val = data[index]

                    # set the min, use not >= to handle nan comparison
                    if ret[grpIdx] == inv or not val >= ret[grpIdx]:
                        ret[grpIdx] = val


    #-------------------------------------------------------------------------------------------------
    @nb.jit(parallel=True, nopython=True, cache=True)
    def _numbaEMA(iGroup, iFirstGroup, nCountGroup, binLow, binHigh, data, ret, time, decayRate):
        for grpIdx in nb.prange(binLow, binHigh):
            start = iFirstGroup[grpIdx]
            last = start + nCountGroup[grpIdx]

            # init per group data
            lastEma = 0.0
            lastTime = time[iGroup[start]]
            for index in range(start, last):
                rowIdx=iGroup[index]

                # ema calculation
                timeDelta = time[rowIdx]-lastTime
                lastTime = time[rowIdx]
                lastEma = data[rowIdx] + lastEma * np.exp(-decayRate * timeDelta)

                # store the return result
                ret[rowIdx]=lastEma

    #-------------------------------------------------------------------------------------------------#
    @nb.njit(parallel=True)
    def _numbaEMA2(iGroup, iFirstGroup, nCountGroup, data, ret, time, decayRate):
        '''
        For each group defined by the grouping arguments, sets 'ret' to a true EMA of the 'data'
        argument using the time argument as the time and the 'decayRate' as the decay rate.
    
        Arguments:
        iGroup, iFirstGroup, nCountGroup:  from a groupby object's 'get_groupings' method
        data:  the original data to be opperated on
        ret:  a blank array the same size as 'data' which will return the processed data
        time: a list of times associated to the rows of data
        decayRate: the decay rate (e based)
    
        TODO:  Error checking.
        '''
        for grpIdx in nb.prange(1, iFirstGroup.shape[0]):
            startIdx = iFirstGroup[grpIdx]
            nInGrp = nCountGroup[grpIdx]
            endIdx = startIdx + nInGrp
            rowIdx = iGroup[startIdx : endIdx]
        
            if nInGrp > 0:
                rows = data[ rowIdx ]
                times = time[ rowIdx ]
                totalWeight = 0.0
                totalValues = 0.0
                pEMA = np.nan
                pTime = times[0]
                for (idx, (t, v)) in enumerate(zip(times, rows)):
                    if not np.isnan(v):
                        deltaT = t - pTime
                        decay = np.exp(-decayRate*deltaT)
                        totalWeight = totalWeight*decay + 1
                        totalValues = totalValues*decay + v
                        pTime = t
                        pEMA = totalValues / totalWeight
                    rows[idx] = pEMA
                
                ret[rowIdx] = rows
        return


    ### Trim (an example which returns a dataset the same size as the original) ###
    #-------------------------------------------------------------------------------------------------#
    @nb.njit(parallel=True)
    def _numbaTrim(iGroup, iFirstGroup, nCountGroup, data, ret,   x, y):
        '''
        For each group defined by the grouping arguments, sets 'ret' to be a copy of the 'data'
        with elements below the 'x'th percentile or above the 'y'th percentile of the group set to nan.
    
        Arguments:
        iGroup, iFirstGroup, nCountGroup:  from a groupby object's 'get_groupings' method
        data:  the original data to be opperated on
        ret:  a blank array the same size as 'data' which will return the processed data
        x:  the lower percentile bound
        y:  the upper percentile bound
        '''
        for grpIdx in nb.prange(1, iFirstGroup.shape[0]):
            startIdx = iFirstGroup[grpIdx]
            endIdx = startIdx + nCountGroup[grpIdx]
            rowIdx = iGroup[startIdx : endIdx]
            rows = data[ rowIdx ]
            (a, b) = np.nanpercentile(rows, [x,y])
            mask = (rows <= a) | (rows >= b)
            rows[mask] = np.nan
            ret[ rowIdx ] = rows
        return


    def grpTrim(grp, x, y):
        '''
        For each column, for each group, determine the x'th and y'th percentile of the data
        and set data below the x'th percentile or above the y'th percentile to nan.
    
        Arguments:
        grp:  a groupby object
        x:  lower percentile
        y:  uppper percentile
    
        Return:  a dataset with the values outside the given percentiles set to np.nan
    
        TODO:  Test column types to make sure that the numba code will work nicely
        '''
        g = grp.get_groupings()
        iGroup = g['iGroup']
        iFirstGroup = g['iFirstGroup']
        nCountGroup = g['nCountGroup']
    
        #retData = rt.Dataset(tmp.grp.gbkeys)
        retData = grp._dataset[ list(grp.gbkeys.keys()) ]
        for colName in grp._dataset:
            if colName not in grp.gbkeys.keys():
                ret = empty( grp._dataset.shape[0] )
                grp._numbaTrim(iGroup, iFirstGroup, nCountGroup, grp._dataset[colName], ret, x, y)
                retData[colName] = ret
        return retData
    #-------------------------------------------------------------------------------------------------#

    # FillForward
    #-------------------------------------------------------------------------------------------------#
    @nb.njit(parallel=True)
    def _numbaFillForward(iGroup, iFirstGroup, nCountGroup, data, ret):
        '''
        propogate forward non-NaN values within a group, overwriting NaN values.
        TODO:  better documentation
        '''
        for grpIdx in nb.prange(1, iFirstGroup.shape[0]):
            startIdx = iFirstGroup[grpIdx]
            endIdx = startIdx + nCountGroup[grpIdx]
            rowIdx = iGroup[startIdx : endIdx]
            rows = data[rowIdx]
        
            fill = np.nan
            for idx in range(rows.shape[0]):
                if np.isnan(rows[idx]):
                    rows[idx] = fill
                else:
                    fill = rows[idx]
            ret[rowIdx] = rows
        return

    @nb.njit(parallel=True)
    def _numbaFillBackward(iGroup, iFirstGroup, nCountGroup, data, ret):
        '''
        propogate backward non-NaN values within a group, overwriting NaN values.
        TODO:  better documentation
        '''
        for grpIdx in nb.prange(1, iFirstGroup.shape[0]):
            startIdx = iFirstGroup[grpIdx]
            endIdx = startIdx + nCountGroup[grpIdx]
            rowIdx = iGroup[startIdx : endIdx]
            rows = data[ rowIdx ]
        
            fill = np.nan
            for idx in range(rows.shape[0]):
                if np.isnan(rows[-idx-1]):
                    rows[-idx-1] = fill
                else:
                    fill = rows[-idx-1]
            ret[rowIdx] = rows
        return

    def grpFillForward(grp):
        '''
        propogate forward non-NaN values within a group, overwriting NaN values.
        TODO:  better documentation
        '''
        g = grp.get_groupings()
        iGroup = g['iGroup']
        iFirstGroup = g['iFirstGroup']
        nCountGroup = g['nCountGroup']
    
        #retData = rt.Dataset(tmp.grp.gbkeys)
        retData = grp._dataset[ list(grp.gbkeys.keys()) ]
        for colName in grp._dataset:
            if colName not in grp.gbkeys.keys():
                ret = empty( grp._dataset.shape[0] )
                grp._numbaFillForward(iGroup, iFirstGroup, nCountGroup, grp._dataset[colName], ret)
                retData[colName] = ret
        return retData

    def grpFillBackward(grp):
        '''
        propogate backward non-NaN values within a group, overwriting NaN values.
        TODO:  better documentation
        '''
        g = grp.get_groupings()
        iGroup = g['iGroup']
        iFirstGroup = g['iFirstGroup']
        nCountGroup = g['nCountGroup']
    
        #retData = rt.Dataset(tmp.grp.gbkeys)
        retData = grp._dataset[ list(grp.gbkeys.keys()) ]
        for colName in grp._dataset:
            if colName not in grp.gbkeys.keys():
                ret = empty( grp._dataset.shape[0] )
                grp._numbaFillBackward(iGroup, iFirstGroup, nCountGroup, grp._dataset[colName], ret)
                retData[colName] = ret
        return retData

    def grpFillForwardBackward(grp):
        '''
        propogate forward, then backward, non-NaN values within a group, overwriting NaN values.
        TODO:  better documentation
        '''
        g = grp.get_groupings()
        iGroup = g['iGroup']
        iFirstGroup = g['iFirstGroup']
        nCountGroup = g['nCountGroup']
    
        #retData = rt.Dataset(tmp.grp.gbkeys)
        retData = grp._dataset[ list(grp.gbkeys.keys()) ]
        for colName in grp._dataset:
            if colName not in grp.gbkeys.keys():
                forwardFilled = empty( grp._dataset.shape[0] )
                grp._numbaFillForward(iGroup, iFirstGroup, nCountGroup, grp._dataset[colName], forwardFilled)
                ret = empty( grp._dataset.shape[0] )
                grp._numbaFillBackward(iGroup, iFirstGroup, nCountGroup, forwardFilled, ret)
                retData[colName] = ret
        return retData

    #---------------------------------------------------------------
    def nb_ema(self, *args, time=None, decay_rate = None, **kwargs):
        '''
        Other Parameters
        ----------------
        time: an array of times (often in nanoseconds) associated to the rows of data
        decayRate: the scalar decay rate (e based)
        '''
        if time is None:
            raise KeyError("time cannot be none")
        if len(time) != self._dataset.shape[0]:
            raise TypeError(f"time array must be the same size as the dataset")

        if decay_rate is None:
            raise KeyError("decay_rate cannot be none")
        if not np.isscalar(decay_rate):
            raise TypeError(f"decay_rate must be a scalar not type {type(decay_rate)}")

        # Lookup our function to get a function_number
        return self._calculate_all(NUMBA_REVERSE_FUNC[GroupbyNumba.nb_ema], *args, func_param=(time, decay_rate), **kwargs)

    #---------------------------------------------------------------
    def nb_sum_punt_test(self, *args, **kwargs):
        """Compute sum of group"""
        return self._calculate_all(GB_FUNCTIONS.GB_SUM, *args, **kwargs)

    #---------------------------------------------------------------
    def nb_sum(self, *args, **kwargs):
        """Compute sum of group"""
        return self._calculate_all(NUMBA_REVERSE_FUNC[GroupbyNumba.nb_sum], *args, **kwargs)

    #---------------------------------------------------------------
    def nb_min(self, *args, **kwargs):
        """Compute sum of group"""
        return self._calculate_all(NUMBA_REVERSE_FUNC[GroupbyNumba.nb_min], *args, **kwargs)


#----------------------------------------------------
# add more routines here to determine the output dtype from the input dtype
def NUMBA_DTYPE_FLOATS(dtype):
    if isinstance(dtype, np.float64):
        return np.float64
    return np.float32

def NUMBA_DTYPE_SUM(dtype):
    #upcast most ints to int64
    if isinstance(dtype, np.uint64):
        return np.uint64
    if dtype.num <=10:
        return np.int64
    return dtype

CALC_PACK = GroupbyNumba._nb_groupbycalculateallpack
CALC_UNPACK = GroupbyNumba._nb_groupbycalculateall

#------------------------------------------------------------
#     name:       (basic/packing,       func_frontend,                   func_backend,          gb_function,  dtype,              return_full True/False)
#     ------      -------------------   ------------------------         ------------------     -----------   ----------------    ------------------
NUMBA_GB_TABLE= {
    'nb_ema' :     (GB_PACKUNPACK.PACK,   GroupbyNumba.nb_ema,           GroupbyNumba._numbaEMA, CALC_PACK,    NUMBA_DTYPE_FLOATS, True),
    'nb_sum' :     (GB_PACKUNPACK.UNPACK, GroupbyNumba.nb_sum,           GroupbyNumba._numbasum, CALC_UNPACK,  NUMBA_DTYPE_SUM,    False),
    'nb_min' :     (GB_PACKUNPACK.UNPACK, GroupbyNumba.nb_min,           GroupbyNumba._numbamin, CALC_UNPACK,  None,               False),
    'nb_sum_punt' :(GB_PACKUNPACK.UNPACK, GroupbyNumba.nb_sum_punt_test, None, None, None, False),
}

# key is a function number
# key : (funcname, requirespacking, frontend, backend, grouper)

NUMBA_REVERSE_TABLE={}
NUMBA_REVERSE_FUNC={}

# start assigning funcnum values at 1000
for i,(k,v)in enumerate(NUMBA_GB_TABLE.items()):
    NUMBA_REVERSE_TABLE[i + GB_FUNC_NUMBA]={'name': k,  'packing': v[0], 'func_front': v[1], 'func_back': v[2],  'func_gb':v[3],  'func_dtype': v[4], 'return_full': v[5]}
    NUMBA_REVERSE_FUNC[v[1]] = i + GB_FUNC_NUMBA

# register our custom functions
GroupByOps.register_functions(NUMBA_REVERSE_TABLE)
TypeRegister.Grouping.register_functions(NUMBA_REVERSE_TABLE)
