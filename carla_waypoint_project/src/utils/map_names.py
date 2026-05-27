"""CARLA map name normalization and compatibility helpers."""

from __future__ import annotations

import re
from typing import Optional


_KNOWN_PREFIXES = (
    "/Game/Carla/Maps/",
    "Game/Carla/Maps/",
    "/Carla/Maps/",
    "Carla/Maps/",
)


def display_map_name(map_name: Optional[str]) -> str:
    """Return a compact user-facing map name."""
    if not map_name:
        return "unknown"
    value = str(map_name).strip().replace("\\", "/")
    for prefix in _KNOWN_PREFIXES:
        if value.lower().startswith(prefix.lower()):
            value = value[len(prefix) :]
            break
    value = value.rstrip("/")
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    return value or "unknown"


def normalize_map_name(map_name: Optional[str]) -> Optional[str]:
    """Return a stable path-stripped map identifier, preserving meaningful suffixes."""
    display = display_map_name(map_name)
    if display == "unknown":
        return None
    return display


def map_slug(map_name: Optional[str]) -> str:
    """Return a filesystem-safe map slug."""
    normalized = normalize_map_name(map_name) or "unknown_map"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized.strip())
    return slug.strip("_") or "unknown_map"


def maps_compatible(a: Optional[str], b: Optional[str]) -> bool:
    """Return whether two CARLA map names can safely share route endpoints."""
    a_id = normalize_map_name(a)
    b_id = normalize_map_name(b)
    if not a_id or not b_id:
        return False
    if a_id.casefold() == b_id.casefold():
        return True
    return _compatibility_key(a_id) == _compatibility_key(b_id)


def _compatibility_key(map_id: str) -> str:
    value = map_id.casefold()
    match = re.fullmatch(r"(town\d+)(?:_opt)?", value)
    if match:
        return match.group(1)
    return value
