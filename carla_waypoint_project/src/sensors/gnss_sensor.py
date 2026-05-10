"""GNSS sensor skeleton for future localization integration."""

from __future__ import annotations

from typing import Optional

from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class GnssSensor:
    """Placeholder wrapper for a CARLA GNSS sensor actor."""

    def __init__(self, world: "carla.World", blueprint_library: "carla.BlueprintLibrary") -> None:
        self._world = world
        self._blueprint_library = blueprint_library
        self._actor: Optional["carla.Sensor"] = None

    def spawn(self, attach_to: "carla.Actor") -> None:
        """Spawn GNSS sensor.

        TODO: Configure GNSS blueprint, attach callback, and store latest measurement.
        """
        _ = attach_to

    def destroy(self) -> None:
        """Destroy GNSS actor when implemented."""
        if self._actor is not None:
            self._actor.stop()
            self._actor.destroy()
            self._actor = None

