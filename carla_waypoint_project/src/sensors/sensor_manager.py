"""Sensor factory and lifecycle registry."""

from __future__ import annotations

from typing import List, Protocol

from src.evaluation.benchmark_config import SensorNoiseConfig
from src.sensors.camera_sensor import CameraSensor
from src.sensors.gnss_sensor import GnssSensor
from src.sensors.imu_sensor import ImuSensor
from src.sensors.lidar_sensor import LidarSensor
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class ManagedSensor(Protocol):
    def destroy(self) -> None:
        ...


class SensorManager:
    """Create and destroy project sensor wrappers."""

    def __init__(self, world: "carla.World", blueprint_library: "carla.BlueprintLibrary") -> None:
        self._world = world
        self._blueprint_library = blueprint_library
        self._active_sensors: List[ManagedSensor] = []

    def create_rgb_camera(self, attach_to: "carla.Actor") -> CameraSensor:
        """Create and spawn one RGB camera attached to an actor."""
        camera = CameraSensor(self._world, self._blueprint_library, attach_to=attach_to)
        camera.spawn()
        self._active_sensors.append(camera)
        return camera

    def create_gnss(self, attach_to: "carla.Actor", config: SensorNoiseConfig | None = None) -> GnssSensor:
        """Create and spawn one GNSS sensor attached to an actor."""
        gnss = GnssSensor(self._world, self._blueprint_library)
        if config is not None:
            gnss.apply_config(config, respawn=False)
        gnss.spawn(attach_to=attach_to)
        self._active_sensors.append(gnss)
        return gnss

    def create_imu(self, attach_to: "carla.Actor", config: SensorNoiseConfig | None = None) -> ImuSensor:
        """Create and spawn one IMU sensor attached to an actor."""
        imu = ImuSensor(self._world, self._blueprint_library)
        if config is not None:
            imu.apply_config(config, respawn=False)
        imu.spawn(attach_to=attach_to)
        self._active_sensors.append(imu)
        return imu

    def create_lidar(self, attach_to: "carla.Actor") -> LidarSensor:
        """Create and spawn one LiDAR sensor attached to an actor."""
        lidar = LidarSensor(self._world, self._blueprint_library)
        lidar.spawn(attach_to=attach_to)
        self._active_sensors.append(lidar)
        return lidar

    def destroy_all(self) -> None:
        """Destroy all sensors currently owned by this manager."""
        for sensor in reversed(self._active_sensors):
            sensor.destroy()
        self._active_sensors.clear()
