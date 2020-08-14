# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
import enum
import logging
import sys
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional
from ..rt_enum import TypeRegister

if TYPE_CHECKING:
    from ..rt_struct import Struct

# The following import imports all the benchmarks exposed by this module which
# is run inside main. Please do not remove this import unless changes are made to main.
from ..benchmarks import *
from .runner import quick_analysis

_logger = logging.getLogger(__name__)
"""Module-level logger."""
_logger.setLevel(logging.INFO)


class ExitCode(enum.IntEnum):
    """Encode the benchmark driver valid exit codes."""

    OK = 0
    FAILED = 1
    INTERRUPTED = 2
    USAGE_ERROR = 3


class UsageFailure(Exception):
    """Raised on benchmarks invalid usage which should be handled before performing
    any further argument processing or benchmark invocations."""

    pass


def collect_benchmarks(
    benchmark_names: Optional[List[str]] = None,
    collect_comparisons: bool = False
) -> Mapping[str, callable]:
    """Returns the list of benchmark functions for invocation."""
    # HACK HACK HACK - Run the benchmarks via globals()
    # This requires newly added benchmarks to be exposed at the benchmarks module level
    # and imported in this global namespace.
    g = globals()

    benchmarks: Dict[str, callable] = dict()
    for k, v in g.items():
        # Only consider benchmarks specified, if any.
        if benchmark_names and k not in benchmark_names:
            continue

        # Look for functions whose name matches our convention(s) for benchmarks.
        func: Optional[callable] = None
        if k.startswith("benchmark_") or k.startswith("bench_"):
            func = v
        if collect_comparisons and k.startswith("compare_"):
            func = v

        # If this function is a benchmark, add it to the dictionary of benchmarks to run.
        if func is not None:
            benchmarks[k] = func

    return benchmarks


def _capture_benchmark_metadata() -> 'Struct':
    """Capture contextual metadata for a benchmark run e.g. the current date/time, riptable version, and machine information."""
    import platform
    from .. import __version__ as rt_version

    benchmark_metadata = TypeRegister.Struct()
    benchmark_metadata['timestamp_utc_start'] = datetime.utcnow().isoformat()

    benchmark_metadata['python_version'] = platform.python_version_tuple()
    benchmark_metadata['python_implementation'] = platform.python_implementation()

    benchmark_metadata['riptable_version'] = rt_version

    benchmark_metadata['sysname'] = platform.system()
    benchmark_metadata['nodename'] = platform.node()
    benchmark_metadata['platform'] = platform.platform()
    benchmark_metadata['platform_release'] = platform.release()
    benchmark_metadata['platform_version'] = platform.version()
    benchmark_metadata['platform_machine'] = platform.machine()
    benchmark_metadata['platform_processor'] = platform.processor()

    try:
        from cpuinfo import get_cpu_info
        benchmark_metadata['cpuinfo'] = TypeRegister.Struct(get_cpu_info())
    except:
        _logger.warning("Unable to import 'cpuinfo' package and/or read CPU information.")

    return benchmark_metadata


def _set_win_process_priority() -> Optional[bool]:
    """
    Sets the process priority class to an elevated value.

    Microbenchmarks are typically very short in duration and therefore are prone
    to noise from other code running on the same machine. Setting the process priority
    to an elevated level while running the benchmarks helps mitigate that noise
    so the results are more accurate (and also more consistent between runs).

    Returns
    -------
    success : bool, optional
        Indication of whether (or not) setting the process priority class succeeded.
        If the priority did not need to be elevated (because it was already), None is returned.
    """
    import win32api, win32process

    # Psuedo-handle for the current process.
    # Because this is a psuedo-handle (i.e. isn't a real handle), it doesn't need to be cleaned up.
    curr_proc_hnd = win32api.GetCurrentProcess()

    # We use the 'ABOVE_NORMAL_PRIORITY_CLASS' here, as that should be good enough to reduce general noise;
    # if necessary, we can try the 'HIGH_PRIORITY_CLASS' but that class and higher can begin to cause the system
    # to become unresponsive so we'll avoid it unless needed; or we can control it with something like a
    # 'strong_hint' bool parameter which chooses between ABOVE_NORMAL_PRIORITY_CLASS and HIGH_PRIORITY.
    target_priority_class: int = win32process.ABOVE_NORMAL_PRIORITY_CLASS

    try:
        # Get the current process priority class. If it's already equal to or higher than the class
        # we were going to set to, don't bother -- we don't want to lower it.
        current_priority_class = win32process.GetPriorityClass(curr_proc_hnd)
        if current_priority_class >= target_priority_class:
            return None

        else:
            # Try to set the priority level for the current process.
            # It can fail if the user (or process) hasn't been granted the PROCESS_SET_INFORMATION right.
            # https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setpriorityclass
            return win32process.SetPriorityClass(curr_proc_hnd, target_priority_class)
    except:
        return False


def _setup(cli_args: argparse.Namespace) -> None:
    """Perform any pre-processing work such as configuring display settings prior to benchmark function invocations."""
    # Display all rows/columns in the resulting Datasets/Multisets.
    TypeRegister.DisplayOptions.COL_ALL = True
    TypeRegister.DisplayOptions.ROW_ALL = True

    # Set process priority to cut down on noise from other processes.
    # TODO: Implement this for Linux and macOS/Darwin too.
    #       https://linux.die.net/man/3/setpriority
    if sys.platform.startswith('win'):
        priority_was_elevated = _set_win_process_priority()
        if priority_was_elevated is None: pass
        elif priority_was_elevated:
            _logger.debug("Increased process priority level to reduce measurement noise.")
        else:
            _logger.warning("Unable to set the process priority class. Benchmark results may be noisier as a result.")

    else:
        _logger.warning(f"Setting the process priority class is not yet supported for platform '{sys.platform}'. "
                        f"Benchmark results may be noisier as a result.",
                        extra={'platform': sys.platform})


def make_argparser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(description="Riptable benchmark runner.")

    arg_parser.add_argument(
        dest="benchmarks",
        metavar="benchmark",
        nargs="*",
        help="space delimited benchmark names to run",
    )
    arg_parser.add_argument(
        "--comparison",
        dest="comparison",
        action="store_true",
        help="additionally collect and run comparison benchmarks",
    )
    arg_parser.add_argument(
        "-c", dest="comparison", action="store_true", help="additionally collect and run comparison benchmarks"
    )
    arg_parser.add_argument(
        "--debug", dest="debug", action="store_true", help="debug level logging"
    )
    arg_parser.add_argument(
        "--d", dest="debug", action="store_true", help="debug level logging"
    )
    arg_parser.add_argument(
        '--out-file',
        dest='output_filename',
        help='The filename where benchmark results will be saved in SDS format.')

    # Should there be a verbose flag? does that log INFO level logging which is the current default?
    # arg_parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose mode with INFO level logging')

    return arg_parser


def main() -> ExitCode:
    # TODO: Further improvements:
    #   - use python logging facilities and indentation for more readable display results
    #   - maybe add a benchmark progress record after each benchmark step
    #     - (optionally) dump each benchmark result data to SDS and flush before proceeding to the next trial
    #     - this way we can restart the run and get partial results (i.e. results for just some of the benchmarks)
    #       if one of the benchmarks crashes e.g. due to OOM on a particular machine

    # Configure the benchmark logger to log to 'benchmarks.log' and to console.
    # Table of log record attributes: https://docs.python.org/3.9/library/logging.html#logrecord-attributes
    # Table of logging levels: https://docs.python.org/3.9/library/logging.html#levels
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    # set up root logger which writes all logs
    logging.basicConfig(
        filename="benchmarks.log", level=logging.INFO, format=log_format
    )
    formatter = logging.Formatter(log_format)

    # set up module logger that only writes benchmark related logs
    root = logging.getLogger()

    # set up console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    _logger.addHandler(ch)

    # Parse CLI arguments.
    arg_parser = make_argparser()
    args = arg_parser.parse_args()
    _logger.info(f"main: parsed args {args}")

    if args.debug:
        root.setLevel(logging.DEBUG)
        _logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)

    # Perform other setup/configuration tasks.
    _setup(args)

    # Switch to control whether we run comparisons or not,
    # since as things are structured now they'll be doing redundant work.
    run_comparisons = args.comparison
    runnable_benchmarks: Optional[List[str]] = args.benchmarks if args.benchmarks else None

    bench_map = collect_benchmarks(
        benchmark_names=runnable_benchmarks, collect_comparisons=run_comparisons
    )
    if bench_map is None:
        raise UsageFailure(f"No benchmarks found with arguments {args}.")
    if not all(bench_map.values()):
        raise UsageFailure(
            f"Error collecting benchmarks.\nPassed arguments {args}\nCollected {bench_map}"
        )

    # Struct to hold benchmark results if the option has been specified
    # to save the data.
    benchmark_results: Optional['Struct'] = TypeRegister.Struct() if args.output_filename else None

    # If saving out benchmark results, capture some additional metadata about the benchmark run,
    # e.g. the current date/time, riptable version, and machine information.
    benchmark_metadata: Optional['Struct'] = None
    if benchmark_results is not None:
        benchmark_metadata = _capture_benchmark_metadata()
        benchmark_results['meta'] = benchmark_metadata

    for name, func in bench_map.items():
        _logger.info(f"Running {name}")

        # TODO: Reset/GC recycler before each benchmark to keep peak mem usage as low as possible?

        # Run the benchmark function; it will return a Dataset containing the
        # raw results over multiple runs.
        benchmark_result = func()

        # If saving the results to SDS, save the raw/complete data for analysis --
        # we want all the data, rather than the rolled-up data we'd display to the console.
        if benchmark_results is not None:
            benchmark_results[name] = benchmark_result

        # If the benchmark is one of the new-style benchmarks, we can automatically
        # perform a quick analysis to summarize the results for display to the console.
        if name.startswith("bench_") or name.startswith("compare_"):
            benchmark_result = quick_analysis(benchmark_result)

        # Write the result out to the logger (e.g. so it's written to a file and the console).
        _logger.info(f"\n{benchmark_result}")

    # If the option was specified to write the benchmark data out to a file,
    # do that now.
    if args.output_filename:
        # Capture some additional metadata about the benchmark run before saving.
        benchmark_metadata['timestamp_utc_end'] = datetime.utcnow().isoformat()

        # TODO: Capture additional metadata
        #   * loaded package versions of numpy, pandas, numba, tbb
        #   * numba threading engine (only available once we've executed at least one parallel numba function)
        #   * peak memory usage for this process?

        _logger.info("Writing results to: %s", args.output_filename, extra={'output_filename': args.output_filename})
        benchmark_results.save(args.output_filename, overwrite=True)
        _logger.info("Finished writing results to disk.")

    _logger.info(f"main: finished with exit code {ExitCode.OK}")
    return ExitCode.OK
