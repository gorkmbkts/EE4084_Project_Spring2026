"""LiDAR sensor wrapper for live point-cloud monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional

import numpy as np

from config.settings import LIDAR
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


@dataclass(frozen=True)
class LidarMeasurement:
    """Latest LiDAR point cloud parsed from CARLA raw data."""

    points: np.ndarray
    frame: int
    timestamp: float

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])


class LidarSensor:
    """Manage a CARLA ray-cast LiDAR actor and its latest point cloud."""

    def __init__(self, world: "carla.World", blueprint_library: "carla.BlueprintLibrary") -> None:
        self._world = world
        self._blueprint_library = blueprint_library
        self._actor: Optional["carla.Sensor"] = None
        self._latest_measurement: Optional[LidarMeasurement] = None
        self._lock = Lock()

    @property
    def actor(self) -> "carla.Sensor":
        if self._actor is None:
            raise RuntimeError("LiDAR actor is not spawned yet.")
        return self._actor

    def spawn(self, attach_to: "carla.Actor") -> None:
        """Spawn the LiDAR sensor and start listening for point clouds."""
        lidar_bp = self._blueprint_library.find(LIDAR.blueprint_id)
        attributes = {
            "sensor_tick": LIDAR.sensor_tick,
            "channels": LIDAR.channels,
            "range": LIDAR.range_m,
            "points_per_second": LIDAR.points_per_second,
            "rotation_frequency": LIDAR.rotation_frequency_hz,
            "upper_fov": LIDAR.upper_fov_deg,
            "lower_fov": LIDAR.lower_fov_deg,
        }
        for name, value in attributes.items():
            if lidar_bp.has_attribute(name):
                lidar_bp.set_attribute(name, str(value))

        transform = carla.Transform(
            carla.Location(
                x=LIDAR.relative_x,
                y=LIDAR.relative_y,
                z=LIDAR.relative_z,
            )
        )
        self._actor = self._world.spawn_actor(lidar_bp, transform, attach_to=attach_to)
        self._actor.listen(self._on_measurement)

    def _on_measurement(self, measurement: "carla.LidarMeasurement") -> None:
        points = self._parse_points(measurement.raw_data)
        latest = LidarMeasurement(
            points=points,
            frame=int(measurement.frame),
            timestamp=float(measurement.timestamp),
        )
        with self._lock:
            self._latest_measurement = latest

    @staticmethod
    def _parse_points(raw_data: bytes) -> np.ndarray:
        points = np.frombuffer(raw_data, dtype=np.float32)
        if points.size == 0 or points.size % 4 != 0:
            return np.empty((0, 4), dtype=np.float32)
        return points.reshape((-1, 4)).copy()

    def get_latest_measurement(self) -> Optional[LidarMeasurement]:
        """Return a copy of the latest LiDAR measurement, if one has arrived."""
        with self._lock:
            if self._latest_measurement is None:
                return None
            latest = self._latest_measurement
            return LidarMeasurement(
                points=latest.points.copy(),
                frame=latest.frame,
                timestamp=latest.timestamp,
            )

    def destroy(self) -> None:
        """Stop and destroy LiDAR actor safely."""
        if self._actor is not None:
            self._actor.stop()
            self._actor.destroy()
            self._actor = None
