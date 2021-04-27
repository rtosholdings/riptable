"""
General-purpose settings for configuring riptable behavior.

This module provides a class encapsulating riptable settings and
feature flags, along with functions for retrieving a top-level,
process-wide instance of the class.
"""
__all__ = [
    'Settings',
    'get_global_settings'
]

from typing import NamedTuple


# TODO: Make this a frozen @dataclass once riptable only supports Python 3.7+.
#       Once the change is made, we can also implement a function which looks for environment
#       variables (e.g. "RIPTABLE_ENABLE_NUMBA_CACHE", "RIPTABLE_MAX_NUM_THREADS") specifying
#       overrides for default values, then creates the global Settings instance accordingly.
class Settings(NamedTuple):
    """
    Encapsulates process-wide settings and feature-flags for riptable.
    """

    enable_numba_cache: bool = False
    """
    Controls whether the numba JIT cache is enabled for functions within riptable.
    This is disabled (False) by default because the caching can lead to occasional
    segfaults in numba-compiled code for some users, possibly caused by a race condition
    or filesystem non-atomicity.
    """


__global_settings = Settings()
"""Global (process-wide) `Settings` instance."""


def get_global_settings() -> Settings:
    """
    Get the global (process-wide) `Settings` instance.

    Returns
    -------
    Settings
        Global (process-wide) settings.
    """
    return __global_settings
