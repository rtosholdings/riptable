import threading
import logging
from attr import dataclass
from datetime import timedelta
from enum import Enum
from logging import Logger

from collections.abc import Callable

import riptide_cpp as rc


@dataclass(slots=True, frozen=True)
class RiptideLogConfig:
    batch_size: int = 50
    flush_interval: timedelta = timedelta(milliseconds=1000)
    max_concurrent_logs: int = 1_000_000
    on_exception: Callable[[str], None] = None


def enable_riptide_logs(config: RiptideLogConfig = RiptideLogConfig()) -> None:
    rc.EnableLogging(
        int(config.flush_interval.total_seconds() * 1000),
        config.batch_size,
        config.max_concurrent_logs,
        config.on_exception,
    )


def disable_riptide_logs(timeout: int | None = None) -> None:
    rc.DisableLogging(timeout if timeout else -1)
