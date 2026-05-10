"""Coordinate transform helpers for CARLA camera projection."""

from __future__ import annotations

import numpy as np

from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


def world_to_camera_matrix(camera_transform: "carla.Transform") -> np.ndarray:
    """Build world-to-camera transform matrix from a camera transform."""
    return np.array(camera_transform.get_inverse_matrix(), dtype=np.float32)


def ue_to_cv_camera_axes(point_camera_ue: np.ndarray) -> np.ndarray:
    """Convert Unreal Engine camera axes to standard pinhole camera axes."""
    if point_camera_ue.shape[0] == 4:
        point_camera_ue = point_camera_ue[:3]
    return np.array(
        [point_camera_ue[1], -point_camera_ue[2], point_camera_ue[0]],
        dtype=np.float32,
    )

