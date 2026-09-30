"""``__version__`` is derived from the installed distribution, not hardcoded."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version

import sofias_sdk_lite


def test_version_matches_installed_metadata_or_falls_back() -> None:
    try:
        expected = version("sofias-sdk-lite")
    except PackageNotFoundError:
        expected = "0.0.0"
    assert sofias_sdk_lite.__version__ == expected


def test_version_is_pep440_like() -> None:
    assert re.match(r"^\d+\.\d+\.\d+", sofias_sdk_lite.__version__)


def test_new_public_symbols_are_exported() -> None:
    for name in ("BaseStreamingResponseWorkflow", "usage_state_from_list", "__version__"):
        assert name in sofias_sdk_lite.__all__
        assert hasattr(sofias_sdk_lite, name)
