"""GNSS sensor wrapper for live pre-fusion monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional

from config.settings import GNSS
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


@dataclass(frozen=True)
class GnssMeasurement:
    """Latest GNSS reading from CARLA."""

    latitude: float
    longitude: float
    altitude: float
    frame: int
    timestamp: float


class GnssSensor:
    """Manage a CARLA GNSS sensor actor and its latest measurement."""

    def __init__(self, world: "carla.World", blueprint_library: "carla.BlueprintLibrary") -> None:
        self._world = world
        self._blueprint_library = blueprint_library
        self._actor: Optional["carla.Sensor"] = None
        self._latest_measurement: Optional[GnssMeasurement] = None
        self._lock = Lock()

    @property
    def actor(self) -> "carla.Sensor":
        if self._actor is None:
            raise RuntimeError("GNSS actor is not spawned yet.")
        return self._actor

    def spawn(self, attach_to: "carla.Actor") -> None:
        """Spawn the GNSS sensor and start listening for measurements."""
        gnss_bp = self._blueprint_library.find(GNSS.blueprint_id)
        if gnss_bp.has_attribute("sensor_tick"):
            gnss_bp.set_attribute("sensor_tick", str(GNSS.sensor_tick))

        transform = carla.Transform(
            carla.Location(
                x=GNSS.relative_x,
                y=GNSS.relative_y,
                z=GNSS.relative_z,
            )
        )
        self._actor = self._world.spawn_actor(gnss_bp, transform, attach_to=attach_to)
        self._actor.listen(self._on_measurement)

    def _on_measurement(self, measurement: "carla.GnssMeasurement") -> None:
        latest = GnssMeasurement(
            latitude=float(measurement.latitude),
            longitude=float(measurement.longitude),
            altitude=float(measurement.altitude),
            frame=int(measurement.frame),
            timestamp=float(measurement.timestamp),
        )
        with self._lock:
            self._latest_measurement = latest

    def get_latest_measurement(self) -> Optional[GnssMeasurement]:
        """Return the latest GNSS measurement, if one has arrived."""
        with self._lock:
            return self._latest_measurement

    def destroy(self) -> None:
        """Stop and destroy GNSS actor safely."""
        if self._actor is not None:
            self._actor.stop()
            self._actor.destroy()
            self._actor = None
