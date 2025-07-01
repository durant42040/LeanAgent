"""
Utility functions for Lean version management and compatibility checking.
"""

import re
from typing import List


# Regex pattern for parsing Lean 4 toolchain versions
_LEAN4_VERSION_REGEX = re.compile(
    r"leanprover/lean4:(?P<version>v\d+\.\d+\.\d+(?:-rc\d+)?)"
)


def get_lean4_version_from_config(toolchain: str) -> str:
    """Return the required Lean version given a ``lean-toolchain`` config."""
    m = _LEAN4_VERSION_REGEX.fullmatch(toolchain.strip())
    assert m is not None, "Invalid config."
    return m["version"]


def is_supported_version(v: str) -> bool:
    """
    Check if ``v`` is at least `v4.3.0-rc2` and at most `v4.18.0-rc1`.
    Note: Lean versions are generally not backwards-compatible. Also, the Lean FRO
    keeps bumping the default versions of repos to the latest version, which is
    not necessarily the latest stable version. So, we need to be careful about
    what we choose to support.
    """
    max_version = 18
    min_version = 3
    if not v.startswith("v"):
        return False
    v = v[1:]
    major, minor, patch = [int(_) for _ in v.split("-")[0].split(".")]
    if (
        major < 4
        or (major == 4 and minor < min_version)
        or (major == 4 and minor > max_version)
        or (major == 4 and minor == max_version and patch > 1)
    ):
        return False
    if (
        major > 4
        or (major == 4 and minor > min_version)
        or (major == 4 and minor == min_version and patch > 0)
    ):
        return True
    assert major == 4 and minor == min_version and patch == 0
    return True
