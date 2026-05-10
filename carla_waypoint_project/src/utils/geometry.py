"""Geometry helpers shared across modules."""

from __future__ import annotations

import numpy as np

from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


def location_to_homogeneous(location: "carla.Location") -> np.ndarray:
    """Convert a CARLA location to homogeneous world coordinates."""
    return np.array([location.x, location.y, location.z, 1.0], dtype=np.float32)


def with_height_offset(location: "carla.Location", z_offset: float) -> "carla.Location":
    """Return a new CARLA location with a vertical offset."""
    return carla.Location(x=location.x, y=location.y, z=location.z + z_offset)

