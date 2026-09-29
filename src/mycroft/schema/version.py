"""Version constants for the unified Mycroft event schema."""

from __future__ import annotations


SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

__all__ = ["SCHEMA_VERSION", "SUPPORTED_SCHEMA_VERSIONS"]
