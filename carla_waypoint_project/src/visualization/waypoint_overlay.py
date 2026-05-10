"""Waypoint projection and pygame overlay rendering."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pygame

from config.settings import WAYPOINT
from src.utils.carla_import import ensure_carla_import
from src.utils.geometry import location_to_homogeneous, with_height_offset
from src.utils.transforms import ue_to_cv_camera_axes, world_to_camera_matrix

carla = ensure_carla_import()

IntrinsicsCache = Dict[int, Tuple[np.ndarray, int, int]]


def build_camera_intrinsics(camera: "carla.Sensor") -> Tuple[np.ndarray, int, int]:
    """Build camera intrinsics matrix from CARLA camera attributes."""
    image_w = int(camera.attributes["image_size_x"])
    image_h = int(camera.attributes["image_size_y"])
    fov_deg = float(camera.attributes["fov"])

    focal = image_w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    intrinsics = np.array(
        [
            [focal, 0.0, image_w * 0.5],
            [0.0, focal, image_h * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return intrinsics, image_w, image_h


def project_world_to_image(
    point: "carla.Location",
    camera_transform,
    camera_intrinsics: np.ndarray,
) -> Optional[Tuple[int, int, float]]:
    """Project a 3D world point to 2D image coordinates."""
    if isinstance(camera_transform, np.ndarray):
        world_to_camera = camera_transform
    else:
        world_to_camera = world_to_camera_matrix(camera_transform)

    point_world = location_to_homogeneous(point)
    point_camera_ue = world_to_camera @ point_world
    point_camera_cv = ue_to_cv_camera_axes(point_camera_ue)

    depth = float(point_camera_cv[2])
    if depth <= 0.0:
        return None

    pixel = camera_intrinsics @ point_camera_cv
    u = int(pixel[0] / depth)
    v = int(pixel[1] / depth)
    return u, v, depth


def draw_waypoints_on_image(
    surface: pygame.Surface,
    waypoints: Sequence["carla.Waypoint"],
    camera: "carla.Sensor",
    vehicle: "carla.Vehicle",
    target_waypoint: Optional["carla.Waypoint"] = None,
    intrinsics_cache: Optional[IntrinsicsCache] = None,
    surface_offset: tuple[int, int] = (0, 0),
) -> None:
    """Draw projected waypoints on top of camera image."""
    _ = vehicle  # Vehicle kept in signature for future camera/pose-dependent extensions.
    if not waypoints:
        return

    if intrinsics_cache is None:
        intrinsics_cache = {}

    cache_key = camera.id
    cached_intrinsics = intrinsics_cache.get(cache_key)
    if cached_intrinsics is None:
        cached_intrinsics = build_camera_intrinsics(camera)
        intrinsics_cache[cache_key] = cached_intrinsics

    camera_intrinsics, image_w, image_h = cached_intrinsics
    world_to_camera = world_to_camera_matrix(camera.get_transform())

    for waypoint in waypoints:
        point = with_height_offset(waypoint.transform.location, WAYPOINT.height_offset_m)
        projected = project_world_to_image(point, world_to_camera, camera_intrinsics)
        if projected is None:
            continue

        u, v, _ = projected
        if 0 <= u < image_w and 0 <= v < image_h:
            pygame.draw.circle(
                surface,
                WAYPOINT.full_path_color,
                (u + surface_offset[0], v + surface_offset[1]),
                WAYPOINT.full_path_radius_px,
            )

    if target_waypoint is None:
        target_idx = min(WAYPOINT.target_index, len(waypoints) - 1)
        target_waypoint = waypoints[target_idx]

    target_point = with_height_offset(target_waypoint.transform.location, WAYPOINT.height_offset_m)
    target_pixel = project_world_to_image(target_point, world_to_camera, camera_intrinsics)
    if target_pixel is None:
        return

    target_u, target_v, _ = target_pixel
    if 0 <= target_u < image_w and 0 <= target_v < image_h:
        pygame.draw.circle(
            surface,
            WAYPOINT.target_color,
            (target_u + surface_offset[0], target_v + surface_offset[1]),
            WAYPOINT.target_radius_px,
        )


class WaypointOverlayRenderer:
    """Render waypoint overlays into the pygame display surface."""

    def __init__(self) -> None:
        self._intrinsics_cache: IntrinsicsCache = {}

    def draw(
        self,
        surface: pygame.Surface,
        waypoints: Sequence["carla.Waypoint"],
        camera: "carla.Sensor",
        vehicle: "carla.Vehicle",
        target_waypoint: Optional["carla.Waypoint"] = None,
        surface_offset: tuple[int, int] = (0, 0),
    ) -> None:
        draw_waypoints_on_image(
            surface=surface,
            waypoints=waypoints,
            camera=camera,
            vehicle=vehicle,
            target_waypoint=target_waypoint,
            intrinsics_cache=self._intrinsics_cache,
            surface_offset=surface_offset,
        )
