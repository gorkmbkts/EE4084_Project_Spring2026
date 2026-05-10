"""Sensor factory and lifecycle registry."""

from __future__ import annotations

from typing import List

from src.sensors.camera_sensor import CameraSensor
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class SensorManager:
    """Create and destroy project sensor wrappers."""

    def __init__(self, world: "carla.World", blueprint_library: "carla.BlueprintLibrary") -> None:
        self._world = world
        self._blueprint_library = blueprint_library
        self._active_cameras: List[CameraSensor] = []

    def create_rgb_camera(self, attach_to: "carla.Actor") -> CameraSensor:
        """Create and spawn one RGB camera attached to an actor."""
        camera = CameraSensor(self._world, self._blueprint_library, attach_to=attach_to)
        camera.spawn()
        self._active_cameras.append(camera)
        return camera

    def destroy_all(self) -> None:
        """Destroy all sensors currently owned by this manager."""
        for camera in reversed(self._active_cameras):
            camera.destroy()
        self._active_cameras.clear()

