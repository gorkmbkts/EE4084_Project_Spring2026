"""Ground-truth ego state provider for the pre-localization route-following stage."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol

from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


@dataclass(frozen=True)
class EgoState:
    """Common ego-state abstraction used by tracking and control."""

    x: float
    y: float
    z: float
    yaw: float
    speed: float
    timestamp: float

    def distance_xy_to(self, location: "carla.Location") -> float:
        """Return planar distance from this state to a CARLA location."""
        return math.hypot(location.x - self.x, location.y - self.y)


class EgoStateProvider(Protocol):
    """Interface for current and future localization-backed state providers."""

    def get_state(self) -> EgoState:
        """Return the current ego state."""
        ...


class GroundTruthStateProvider:
    """Read ego pose and velocity directly from CARLA actor ground truth."""

    def __init__(self, vehicle: "carla.Vehicle") -> None:
        self._vehicle = vehicle

    def get_state(self) -> EgoState:
        """Convert CARLA vehicle transform and velocity to ``EgoState``."""
        transform = self._vehicle.get_transform()
        velocity = self._vehicle.get_velocity()
        speed = math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)

        try:
            timestamp = float(self._vehicle.get_world().get_snapshot().timestamp.elapsed_seconds)
        except RuntimeError:
            timestamp = time.monotonic()

        return EgoState(
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(transform.location.z),
            yaw=float(transform.rotation.yaw),
            speed=float(speed),
            timestamp=timestamp,
        )
