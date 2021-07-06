__all__ = ["MathLedger"]

import numpy as np
import riptide_cpp as rc

from typing import List
from .rt_struct import Struct
from .rt_timers import GetNanoTime
from .rt_enum import TypeRegister, MATH_OPERATION


# =====================================================================================
class MathLedger(Struct):
    """
    MathLedger class for tracking math calculations and times
    """

    # Defines a generic np.ndarray subclass, that can cache numpy arrays
    # Static Class VARIABLES

    # set Verbose > 1 for extra print debug information
    Verbose: int = 1

    # set VerboseConversion > 1 for extra print debug information
    VerboseConversion: int = 1

    # set to TRUE to calculate CRC
    DOCRC: bool = False

    # Ledger on/off
    DebugUFunc: bool = False

    # ------------------------------------------------------------
    def __init__(self):
        super().__init__()
        # this list grows
        MathLedger._LedgerClear()

    @classmethod
    def _LedgerOn(cls) -> bool:
        print(f"LEDGER ON: ledger was {cls.DebugUFunc}")
        cls.DebugUFunc = True
        try:
            temp = cls.TotalDeletes
        except:
            cls._LedgerClear()
        return True

    @classmethod
    def _LedgerOff(cls) -> bool:
        print(f"LEDGER OFF: ledger was {cls.DebugUFunc}")
        cls.DebugUFunc = False
        return True

    @classmethod
    def _LedgerClear(cls) -> bool:
        cls.UFuncLedger: List[str] = []
        cls.UFuncName: List[str] = []
        cls.USubName: List[str] = []
        cls.UDeltaTime: List[float] = []
        cls.UArgLen: List[int] = []
        cls.UResultLen: List[int] = []
        cls.UKwargs: List[str] = []
        cls.UCRC: List[int] = []
        cls.UArgs1Type: List[str] = []
        cls.UArgs1Len: List[int] = []
        cls.UArgs1: List[str] = []
        cls.UArgs2: List[str] = []
        cls.UArgs3: List[str] = []
        cls.UResult: List[str] = []
        cls.UResultType: List[str] = []
        cls.UResultLen: List[int] = []

        cls.TotalDeletes: int = 0
        cls.TotalTime: float = 0.0
        cls.TotalOperations: int = 0
        cls.TotalConversions: int = 0
        cls.TotalRecycled: int = 0
        return True

    @classmethod
    def _Ledger(cls) -> List[str]:
        return cls.UFuncLedger

    @classmethod
    def _LedgerDump(cls, dataset=True):
        print("--- UFUNC LEDGER ----")
        try:
            print("--- TOTAL DELETES ", cls.TotalDeletes)
        except:
            print(
                "Nothing to show.  Please turn on ledger first with FA._LON or ticf()"
            )
            return

        print("--- TOTAL CONVERSIONS ", cls.TotalConversions)
        print("--- TOTAL RECYCLED ", cls.TotalRecycled)
        print("--- TOTAL TIME ", cls.TotalTime)

        if cls.TotalOperations > 0:
            print("--- AVG TIME ", cls.TotalTime / cls.TotalOperations)

        if dataset:
            ds_dict = {
                "Name": cls.UFuncName,
                "SubName": cls.USubName,
                "Time": cls.UDeltaTime,
                "Args1": cls.UArgs1,
                "Args1Len": cls.UArgs1Len,
                "Args1Type": cls.UArgs1Type,
                "Args2": cls.UArgs2,
                "Args3": cls.UArgs3,
                "Args": cls.UArgLen,
                "Result": cls.UResult,
                "ResultLen": cls.UResultLen,
                "ResultType": cls.UResultType,
            }

            return TypeRegister.Dataset(ds_dict)

        else:
            cls.UFuncLedger = []
            for i in range(len(cls.UFuncName)):
                if cls.DOCRC:
                    ledger = f"{cls.UFuncName[i]}\t{i}\t{cls.UDeltaTime[i]}\t{cls.UCRC[i]}\t{cls.UArgs1[i]}\t{cls.UKwargs[i]}\t-->\t{cls.UResult[i]}"
                else:
                    ledger = f"{cls.UFuncName[i]}\t{i}\t{cls.UDeltaTime[i]}\t{cls.UArgs1[i]}\t{cls.UKwargs[i]}\t-->\t{cls.UResult[i]}"

                cls.UFuncLedger.append(ledger)

            for val in cls.UFuncLedger:
                val = val.replace("\n", "")
                val = val.replace("\r", "")
                print(val)

    @classmethod
    def _LedgerDumpFile(cls, filename):
        import sys

        old_stdout = sys.stdout
        try:
            sys.stdout = open(filename, "w")
            cls._LedgerDump()
        except:
            pass
        sys.stdout = old_stdout

    @classmethod
    def _LON(cls):
        return cls._LedgerOn()

    @classmethod
    def _LOFF(cls):
        return cls._LedgerOff()

    @classmethod
    def _LDUMP(cls, dataset=True):
        return cls._LedgerDump(dataset=dataset)

    @classmethod
    def _LCLEAR(cls):
        return cls._LedgerClear()

    @classmethod
    def _TRACEBACK(cls, func):
        """ print the callback stack to help with debugging """
        import traceback

        for line in traceback.format_stack():
            print(line.strip())

    @classmethod
    def _ASTYPE(cls, func, dtype, *args, **kwargs):
        """ use numpy astype """
        if cls.VerboseConversion > 1:
            print(f"astype ledger {func} {dtype} {args}")
            if cls.VerboseConversion > 2:
                cls._TRACEBACK(func)
        return cls._FUNNEL_ALL(func.astype, dtype, *args, **kwargs)

    @classmethod
    def _AS_FA_TYPE(cls, faself, dtypenum, *args, **kwargs):
        """ use multithreaded conversion preserving sentinels"""
        if cls.VerboseConversion > 1:
            print(f"as fa type ledger {faself.dtype.num} {dtypenum}")
            if cls.VerboseConversion > 2:
                cls._TRACEBACK(rc.ConvertSafe)
        return cls._FUNNEL_ALL(rc.ConvertSafe, faself, dtypenum)

    @classmethod
    def _AS_FA_TYPE_UNSAFE(cls, faself, dtypenum, *args, **kwargs):
        """ use multithreaded conversion NOT preserving sentinels"""
        if cls.VerboseConversion > 1:
            print(f"as fa type unsafe ledger {faself.dtype.num} {dtypenum}")
            if cls.VerboseConversion > 2:
                cls._TRACEBACK(rc.ConvertUnsafe)
        return cls._FUNNEL_ALL(rc.ConvertUnsafe, faself, dtypenum)

    @classmethod
    def _COPY(cls, func, *args, **kwargs):
        if cls.Verbose > 1:
            print(f"copy ledger {func} {args}")
        return cls._FUNNEL_ALL(func.copy, *args, **kwargs)

    @classmethod
    def _INDEX_BOOL(cls, *args):
        if cls.Verbose > 1:
            print(f"index_bool ledger {args}")
        return cls._FUNNEL_ALL(rc.BooleanIndex, *args)

    @classmethod
    def _GETITEM(cls, func, *args):
        if cls.Verbose > 1:
            print(f"getitem ledger {func} {args}")
        return cls._FUNNEL_ALL(func.__getitem__, *args)

    @classmethod
    def _MBGET(cls, *args):
        if cls.Verbose > 1:
            print(f"mbget ledger {args}")
        return cls._FUNNEL_ALL(rc.MBGet, *args)

    @classmethod
    def _REDUCE(cls, arg1, reduceFunc):
        if cls.Verbose > 1:
            print(f"reduce ledger {arg1} {reduceFunc}")
            # if cls.Verbose > 2:   cls._TRACEBACK(rc.Reduce)
        return cls._FUNNEL_ALL(rc.Reduce, arg1, reduceFunc)

    @classmethod
    def _ARRAY_UFUNC(cls, func, ufunc, method, *args, **kwargs):
        if cls.Verbose > 1:
            print(f"array ufunc ledger {func} {ufunc} {method} {args}")
            if cls.Verbose > 2:
                cls._TRACEBACK(ufunc)
        return cls._FUNNEL_ALL(func.__array_ufunc__, ufunc, method, *args, **kwargs)

    @classmethod
    def _BASICMATH_ONE_INPUT(cls, tupleargs, fastfunction, final_num):

        if fastfunction == MATH_OPERATION.BITWISE_NOT:
            # speed up this common operation
            if isinstance(tupleargs, tuple) and tupleargs[0].dtype == bool:
                return cls._BASICMATH_TWO_INPUTS(
                    (tupleargs[0], True, tupleargs[1]),
                    MATH_OPERATION.BITWISE_XOR,
                    final_num,
                )
            elif tupleargs.dtype == bool:
                return cls._BASICMATH_TWO_INPUTS(
                    (tupleargs, True), MATH_OPERATION.BITWISE_XOR, final_num
                )

        if cls.Verbose > 1:
            print(f"BasicMathOneInput {tupleargs} {fastfunction} {final_num}")
        # speedup
        if cls.DebugUFunc is False:
            return rc.BasicMathOneInput(tupleargs, fastfunction, final_num)
        return cls._FUNNEL_ALL(rc.BasicMathOneInput, tupleargs, fastfunction, final_num)

    @classmethod
    def _BASICMATH_TWO_INPUTS(cls, tupleargs, fastfunction, final_num):
        if cls.Verbose > 1:
            print(f"BasicMathTwoInputs {tupleargs} {fastfunction} {final_num}")
        # speedup
        if cls.DebugUFunc is False:
            return rc.BasicMathTwoInputs(tupleargs, fastfunction, final_num)
        return cls._FUNNEL_ALL(
            rc.BasicMathTwoInputs, tupleargs, fastfunction, final_num
        )

    @classmethod
    def _FUNNEL_ALL(cls, func, *args, **kwargs):
        if cls.DebugUFunc is False:
            return func(*args, **kwargs)

        #############################################
        # We are in debug mode below, time and record
        #############################################
        startTime = GetNanoTime()
        result = func(*args, **kwargs)

        delta = (GetNanoTime() - startTime) / 1000000000.0

        cls.TotalTime += delta
        cls.TotalOperations += 1
        deltaTime = float("{0:.9f}".format(delta))

        # add to the ledger
        # cls.UFuncLedger.append(f"{func.__name__}\t{cls.TotalOperations}\t{deltaTime}\t{args}\t{kwargs}")
        # check if CRC option set
        if cls.DOCRC and isinstance(result, np.ndarray):
            crc = rc.CalculateCRC(result)
            cls.UCRC.append(crc)
        else:
            cls.UCRC.append(0)

        # sub name for math functions
        if func == rc.BasicMathTwoInputs or func == rc.BasicMathOneInput:
            cls.USubName.append(MATH_OPERATION(args[1]).name)
        else:
            cls.USubName.append("")

        # cls.UFuncLedger.append(ledger)
        cls.UFuncName.append(func.__name__)
        cls.UDeltaTime.append(deltaTime)
        if len(args) > 0:
            args0 = args[0]
            if isinstance(args0, tuple):
                args0 = args0[0]

            cls.UArgs1.append(f"{args0}")

            try:
                cls.UArgs1Type.append(f"{args0.dtype}")
            except:
                cls.UArgs1Type.append(f"{type(args0)}")

            try:
                cls.UArgs1Len.append(len(args0))
            except:
                cls.UArgs1Len.append(0)

            if len(args) > 1:
                cls.UArgs2.append(f"{args[1]}")
                if len(args) > 2:
                    cls.UArgs3.append(f"{args[2]}")
                else:
                    cls.UArgs3.append("")
            else:
                cls.UArgs2.append("")
                cls.UArgs3.append("")
        else:
            cls.UArgs1Type.append(f"{type(None)}")
            cls.UArgs1Len.append(0)
            cls.UArgs1.append("")
            cls.UArgs2.append("")
            cls.UArgs3.append("")

        cls.UKwargs.append(f"{kwargs}")
        cls.UArgLen.append(len(args))

        if result is not None:
            r0 = result
            if isinstance(r0, tuple):
                r0 = r0[0]

            cls.UResult.append(f"{r0}")

            try:
                cls.UResultLen.append(len(r0))
            except:
                cls.UResultLen.append(0)

            try:
                cls.UResultType.append(f"{r0.dtype}")
            except:
                cls.UResultType.append(f"{type(r0)}")

        else:
            # nothing was returned
            cls.UResult.append("None")
            cls.UResultLen.append(0)
            cls.UResultType.append("")

        return result


# keep as last line
TypeRegister.MathLedger = MathLedger
