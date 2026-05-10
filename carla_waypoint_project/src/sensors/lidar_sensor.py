"""LiDAR sensor skeleton for future perception/localization integration."""

from __future__ import annotations

from typing import Optional

from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class LidarSensor:
    """Placeholder wrapper for a CARLA LiDAR sensor actor."""

    def __init__(self, world: "carla.World", blueprint_library: "carla.BlueprintLibrary") -> None:
        self._world = world
        self._blueprint_library = blueprint_library
        self._actor: Optional["carla.Sensor"] = None

    def spawn(self, attach_to: "carla.Actor") -> None:
        """Spawn LiDAR sensor.

        TODO: Configure LiDAR blueprint, attach callback, and store point-cloud frames.
        """
        _ = attach_to

    def destroy(self) -> None:
        """Destroy LiDAR actor when implemented."""
        if self._actor is not None:
            self._actor.stop()
            self._actor.destroy()
            self._actor = None

