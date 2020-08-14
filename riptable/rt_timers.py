__all__ = ['GetNanoTime', 'GetTSC', 'tic', 'ticx', 'toc', 'tocx', 'tt', 'ttx','ticp','tocp','ticf','tocf','utcnow']

import sys
import numpy as np
import riptide_cpp as rc
from riptable.rt_enum import TypeRegister

'''
Timer functionality
'''

def GetNanoTime():
    '''
    Returns: a long integer in unix epoch nanoseconds
    Note: this function is written as fast as possible for both Windows and Linux
    '''
    return rc.GetNanoTime()

def GetTSC():
    '''
    Returns: a long integer from the CPUs's current time stamp counter

    time stamp counter (TSC) are based on the CPUs's clock cycle, which is often above 1GHz
    thus GetTSC return values are guaranteed to be both unique and subsample below 1 nanosecond

    Note: this function is written as fast as possible for both Windows and Linux
    '''
    return rc.GetTSC()

def utcnow(count=1):
    '''
    Returns: a DateTimeNano array with one item representing the current time in utc nanos.

    import riptable as rt
    >>> rt.utcnow()
    DateTimeNano([20190215 11:29:44.022382600])
    >>> rt.utcnow()._fa
    FastArray([1550248297734812800], dtype=int64)

    To make a Dataset full of timestamps:
    >>> Dataset({'localtime':utcnow(1_000_000)})

    See also: rc.GetNanoTime()
    Compare with: datetime.datetime.utcnow()
    '''
    if count ==1:
        return TypeRegister.DateTimeNano([rc.GetNanoTime()], from_tz='GMT')
    else:
        x=[rc.GetNanoTime() for i in range(count)]
        return TypeRegister.DateTimeNano(x, from_tz='GMT')

# Timing code below
def tic():
    '''
    Call tic() folowed by code followed by toc() to time a routine in nanoseconds

    See also: toc, ticx, ticp, ticf
    '''
    global TicStartTime
    TicStartTime = GetNanoTime()


def toc():
    '''
    Call tic() folowed by code followed by toc() to time a routine

    See also: toc, ticx, ticp, ticf
    '''
    global TicStartTime
    global TocEndTime
    TocEndTime = GetNanoTime()
    delta = (TocEndTime - TicStartTime) / 1000000000.0
    deltaTime = float("{0:.6f}".format(delta))
    print("Elapsed time",deltaTime,"seconds.")

# even more accurate cycle counting
def ticx():
    '''
    Call ticx() folowed by code followed by tocx() to time a routine in TSC

    See also: toc, ticx, ticp, ticf
    '''
    global TicStartTimeX
    TicStartTimeX = GetTSC()


def tocx():
    '''
    Call ticx() folowed by code followed by tocx() to time a routine in TSC

    See also: toc, ticx, ticp, ticf
    '''
    global TicStartTimeX
    global TocEndTimeX
    TocEndTimeX = GetTSC()
    delta = (TocEndTimeX - TicStartTimeX)
    print("Elapsed time", delta,"cycles.")

def ticf():
    '''
    Call ticf() folowed by code followed by tocf() to time fastarrays

    See also: toc, ticx, ticp, ticf
    '''
    FA=TypeRegister.FastArray
    FA._LCLEAR()
    FA._LON()

def tocf(dataset=True):
    '''
    Call ticf() folowed by code followed by tocf() to time fastarrays

    Other Parameters
    ----------------
    dataset: defaults to True.  Returns a dataset.  Set to False to print out instead.
    '''
    FA=TypeRegister.FastArray
    FA._LOFF()
    return TypeRegister.MathLedger._LDUMP()

def ticp():
    '''
    Call ticp() folowed by code followed by tocp() to profile function calls

    See also: toc, ticx, ticp, ticf
    '''
    import cProfile
    global pr
    pr = cProfile.Profile()
    pr.enable()

def tocp(dataset=True, logfile=None, sort='time', strip=True, stats=False, calls=False, find=None):
    '''
    Call ticp() folowed by code followed by tocp() to profile anything between the ticp/tocp
    tocp() may be called again to retrieve data in a different manner

    Examples
    --------
    ticp(); ds.sort_copy(by='Symbol'); tocp()._H
    ticp(); ds.sort_copy(by='Symbol'); tocp().sort_view('cumtime')._A
    ticp(); ds.sort_copy(by='Symbol'); tocp(find='rt_fastarray.py:332')._H
    ticp(); ds.sort_copy(by='Symbol'); tocp(find='rt_fastarray.py')._H
    ticp(); ds.sort_copy(by='Symbol'); ds=tocp(calls=True); ds.gb('filepath').sum()._H
    tocp(calls=True).gb(['function','filepath'])['tottime'].sum().sort_view('tottime')._A
    ticp(); ds.sort_copy(by='Symbol'); stats=tocp(stats=True);
    ticp(); ds.sort_copy(by='Symbol'); tocp(False);
    ticp(); ds.sort_copy(by='Symbol'); tocp(False, strip=False);
    ticp(); ds.sort_copy(by='Symbol'); tocp(False, sort='cumtime');

    Other Parameters
    ----------------
    dataset=False.  set to True to return a Dataset otherwise use pstats output
    logfile=None.   set to filename to save the Dataset in SDS format
                    NOTE: consider pickling the result when stats=True to save for later analysis
    strip=True.     set to False to return full path for the filename when dataset=False
    calls=False.    set to True to include 'callee' and 'filepath' to determine caller info
    find=None.      set to a string with 'filename:lineno' to drill into those specific calls
    sort='time'     by default when dataset=False, other options include
        "calls"     --> "call count"
        "ncalls"    --> "call count"
        "cumtime"   --> "cumulative time"
        "cumulative"--> "cumulative time"
        "file"      --> "file name"
        "filename"  --> "file name"
        "line"      --> "line number"
        "module"    --> "file name"
        "name"      --> "function name"
        "nfl"       --> "name/file/line"
        "pcalls"    --> "primitive call count"
        "stdname"   --> "standard name"
        "time"      --> "internal time"
        "tottime"   --> "internal time"

    stats=False.    set to True to return all stats collected by _lsprof.c
        return all information collected by the profiler.
        Each profiler_entry is a tuple-like object with the
        following attributes:
    
            code          code object
            callcount     how many times this was called
            reccallcount  how many times called recursively
            totaltime     total time in this entry
            inlinetime    inline time in this entry (not in subcalls)
            calls         details of the calls
    
        The calls attribute is either None or a list of
        profiler_subentry objects:
    
            code          called code object
            callcount     how many times this is called
            reccallcount  how many times this is called recursively
            totaltime     total time spent in this call
            inlinetime    inline time (not in further subcalls)
    '''
    global pr
    pr.disable()
    if stats:
        return pr.getstats()

    if dataset:
        if sort == 'time': sort='tottime'
        ds = snapshot_stats(pr, sort=sort, calls=calls, findfunc=find)
        if logfile is not None:
            ds.save(logfile)
        return ds
    else:
        import pstats
        if strip:
            pstats.Stats(pr).strip_dirs().sort_stats(sort).print_stats()
        else:
            pstats.Stats(pr).sort_stats(sort).print_stats()


def snapshot_stats(pr, sort='tottime', calls=True, findfunc=None):
    '''
    Parameters
    ----------
    pr:

    Other Parameters
    ----------------
    sort:
    calls:
    findfunc: must be in form filename:lineno such as 'rt_fastarray:423'
              or in the form 'rt_fastarray'

    Returns
    -------
    a Dataset
    '''
    import os

    if findfunc is not None:
        try:
            funcname, linenum = findfunc.split(':')
            linenum = int(linenum)
        except Exception:
            funcname = findfunc
            linenum = None

    def parse_func_info(tup):
        func_str = []

        filepath = '~'
        # module
        if tup[0] != '~':
            # parse file name
            normpath = os.path.normpath(tup[0])
            basename = os.path.basename(normpath)
            func_str.append(basename)
            filepath=normpath[:-len(basename)]

        # line_number 
        if tup[1] != 0:
            func_str.append(':'+str(tup[1]))

        # func name
        if len(func_str) != 0:
            func_str.append('('+tup[2]+')')

        # python func
        else:
            func_str.append(tup[2])
            # to match pstats display
            func_str[0].replace('<','{')
            func_str[0].replace('>','}')

        return "".join(func_str), filepath


    entries = pr.getstats()
    stats = {}
    callersdicts = {}

    #def get_top_level_stats(self):
    #    for func, (cc, nc, tt, ct, callers) in self.stats.items():
    #        self.total_calls += nc
    #        self.prim_calls  += cc
    #        self.total_tt    += tt

    # call information
    # NOTE consider cython or C since this can be huge
    for entry in entries:

        code = entry.code
        if isinstance(code, str):
            func= ('~', 0, code)    # built-in functions ('~' sorts at the end)
        else:
            func= (code.co_filename, code.co_firstlineno, code.co_name)

        nc = entry.callcount         # ncalls column of pstats (before '/')s
        cc = nc - entry.reccallcount # ncalls column of pstats (after '/')
        tt = entry.inlinetime        # tottime column of pstats
        ct = entry.totaltime         # cumtime column of pstats
        callers = {}
        callersdicts[id(entry.code)] = callers
        stats[func] = cc, nc, tt, ct, callers

    # subcall information
    for entry in entries:
        if entry.calls:
            code = entry.code
            if isinstance(code, str):
                func= ('~', 0, code)    # built-in functions ('~' sorts at the end)
            else:
                func= (code.co_filename, code.co_firstlineno, code.co_name)

            for subentry in entry.calls:
                try:
                    callers = callersdicts[id(subentry.code)]
                except KeyError:
                    continue

                nc = subentry.callcount
                cc = nc - subentry.reccallcount
                tt = subentry.inlinetime
                ct = subentry.totaltime

                if func in callers:
                    prev = callers[func]
                    nc += prev[0]
                    cc += prev[1]
                    tt += prev[2]
                    ct += prev[3]

                callers[func] = nc, cc, tt, ct

    if findfunc is not None:
        # this path is taken when user wants to drill into
        callcount = []
        ncalls = []
        tottime = []
        cumtime = []
        names = []

        for func_info, (cc, nc, tt, ct, callers) in stats.items():
            callercount = len(callers)
            if callercount > 0:
                name, filepath =(parse_func_info(func_info))
                if name[0] != '<':
                    funcn, stuff = name.split(':')
                    lineno, stuff = stuff.split('(')
                    lineno = int(lineno)
                    if (linenum is None or lineno == linenum) and funcn == funcname:
                        # NOTE: not sure this is
                        for k,v in callers.items():
                            name, filepathc = (parse_func_info(k))
                            cc1, nc1, tt1, ct1 = (v)
                            
                            callcount.append(cc1)
                            ncalls.append(nc1)
                            tottime.append(tt1)
                            cumtime.append(ct1)
                            names.append(name)

        ds = TypeRegister.Dataset({
            'ncalls' : ncalls,
            'tottime' : tottime,
            'cumtime' : cumtime,
            'callers' : callcount,
            'function' : names})

    else:
        ncalls = []
        tottime = []
        cumtime = []
        names = []
        callcount = []
        ncallers = []
        firstcaller = []
        path = []
        pathc = []


        for func_info, (cc, nc, tt, ct, callers) in stats.items():
            ncalls.append(nc)
            tottime.append(tt)
            cumtime.append(ct)
            callcount.append(cc)
            callercount = len(callers)
            ncallers.append(callercount)
            name, filepath =(parse_func_info(func_info))
            names.append(name)

            # does user want more information?
            if calls:
                filepathc = '~'
                if callercount > 0:
                    firstcall = next(iter(callers))
                    firstcall, filepathc =(parse_func_info(firstcall))
                    #firstcall = f'{firstcall[0]}:{firstcall[1]}({firstcall[2]})'
                else:
                    firstcall = '~'
                firstcaller.append(firstcall)
                path.append(filepath)
                pathc.append(filepathc)

        ds = TypeRegister.Dataset({
            'ncalls' : ncalls,
            'tottime' : tottime,
            'cumtime' : cumtime,
            'callers' : ncallers,
            'function' : names})

        if calls:
            ds['filepath'] = path

        arr_callcount=np.asanyarray(callcount)
        ds.percallT = ds.tottime / ds.ncalls
        ds.percallC = ds.cumtime / arr_callcount

        total_tt = ds.tottime.sum()

        # check if they want information on the caller
        if calls:
            ds['callee'] = firstcaller
            ds['filepathc'] = pathc

    # NOTE: need an option for this to not SORT because that is the order
    return ds.sort_inplace(sort, ascending=False)

def tt(expression:str, loops=1, return_time=False):
    '''
    tictoc time an expression in nanoseconds.  use ; to separate lines

    Args:
        arg1 is a string of code to execute
        arg2 is optional and is how many loops to execute

    '''
    #import __builtin__
    #__builtin__.__dict__.update(locals())
    import inspect
    frame = inspect.currentframe()

    # allow callee to use ; for new lines
    codestr=expression.replace('; ','\n')
    codestr=codestr.replace(';','\n')

    # compile to byte code first to eliminate compile time in calculation
    code=compile(codestr,'<string>','exec')

    #preallocate array of floats
    aTimers=loops*[0.0]

    for i in range(loops):
        startTime = GetNanoTime()
        exec(code, frame.f_back.f_globals, frame.f_back.f_locals)
        endTime = GetNanoTime()
        aTimers[i]=(endTime - startTime) / 1000000000.0

    if loops==1:
        deltaTime = float("{0:.6f}".format(aTimers[0]))
        if return_time:
            return deltaTime
        print("Elapsed time",deltaTime,"seconds.")
    else:
        mTime=np.median(aTimers)
        deltaTime = float("{0:.6f}".format(mTime))
        if return_time:
            return deltaTime
        print("Median",loops,"runs",deltaTime,"seconds.")


def ttx(expression:str, loops=1):
    '''
    tictoc time an expression in TSC (time stamp counters).  use ; to separate lines

    Args:
        arg1 is a string of code to execute
        arg2 is optional and is how many loops to execute

    '''
    import inspect
    frame = inspect.currentframe()

    # allow callee to use ; for new lines
    codestr=expression.replace(';','\n')

    # compile to byte code first to eliminate compile time in calculation
    code=compile(codestr,'<string>','exec')

    #preallocate array of floats
    aTimers=loops*[0]

    for i in range(loops):
        startTime = GetTSC()
        exec(code, frame.f_back.f_globals, frame.f_back.f_locals)
        endTime = GetTSC()
        aTimers[i]=(endTime - startTime)

    if loops==1:
        deltaTime = aTimers[0]
        print("Elapsed time",deltaTime,"cycles.")
    else:
        mTime=np.median(aTimers)
        deltaTime = mTime
        print("Median",loops,"runs",deltaTime,"cycles.")

