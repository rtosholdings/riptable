"""TeamCity utility that wraps TeamCity build metadata.
TeamCity server predefined build parameters can be found here:
https://www.jetbrains.com/help/teamcity/predefined-build-parameters.html
"""
import os
from typing import Optional


__all__ = ["is_running_in_teamcity", "get_build_conf_name"]


def is_running_in_teamcity() -> bool:
    return get_build_conf_name() is not None


def get_build_conf_name() -> Optional[str]:
    return os.environ.get('TEAMCITY_BUILDCONF_NAME', None)
