"""Webitel Knowledge Base indexer."""

from importlib.metadata import PackageNotFoundError, version

SERVICE_NAME = "webitel-kb-indexer"

try:
    SERVICE_VERSION = version("webitel-kb-indexer")
except PackageNotFoundError:  # running from a source tree without an install
    SERVICE_VERSION = "0.0.0"

__all__ = ["SERVICE_NAME", "SERVICE_VERSION"]
