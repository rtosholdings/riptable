# __all__ = ['rolling_window', ]

#import numba

## int32 must appear before int64
## float32 signature should appear before float64 signatures
#@numba.guvectorize([
#    'void(int32[:], int32[:], int32[:], int32[:], int32[:], int64, int64, int64)',
#    'void(int32[:], int32[:], int32[:], int64[:], int64[:], int64, int64, int64)',
#    'void(int32[:], int32[:], int32[:], float32[:], float32[:], int64, int64, int64)',
#    'void(int32[:], int32[:], int32[:], float64[:], float64[:], int64, int64, int64)'
#    ], '(m),(n),(n),(o),(o),(),(),()', target='cpu')
#def rolling_window(pGroup, pFirst, pCount, pAccumBin, pSrc, numUnique, totalInputRows, param):

#    # dummy assignment 
#    currentSum = pSrc[0]
#    windowSize = param

#    #print('rolling windows', numUnique, totalInputRows, windowSize)

#    for i in range(numUnique):
#        start = pFirst[i]
#        last = start + pCount[i]

#        currentSum = 0

#        # this is the loop for a given group
#        for j in range(start,last):
#            if j >= (start + windowSize):
#                break

#            index = pGroup[j]
#            currentSum  += pSrc[index]
#            pAccumBin[index] = currentSum
            
#        for j in range(start + windowSize,last):
#            index = pGroup[j]
#            currentSum  += pSrc[index]
#            currentSum  -= pSrc[pGroup[j - windowSize]]
#            pAccumBin[index] = currentSum

#def test_rolling(gb):
#    return gb.calculate_custom_packed(rolling_window, 2)


#@njit(parallel=True)
#def ema_decay(iGroup, nFirstGroup, nCountGroup, time, data, ema, decay):
#    # decay = np.log(2)/(1e3*100)
#    # start on group 1 to skip over 0 bin
#    for i in prange(1, nFirstGroup.shape[0]):
#        start=nFirstGroup[i]
#        last=start + nCountGroup[i]
#        lastEma = 0
#        lastTime = 0
#        for j in range(start,last):
#            index=iGroup[j]
#            lastTime = time[i]-lastTime
#            lastEma = data[j] + lastEma * np.exp(-decay * lastTime)
#        #update output
#        ema[i]=lastEma


