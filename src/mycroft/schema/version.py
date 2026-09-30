"""Current and backward-compatible Mycroft event schema versions."""

from __future__ import annotations


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, SCHEMA_VERSION})

__all__ = ["SCHEMA_VERSION", "SUPPORTED_SCHEMA_VERSIONS"]
