"""CARLA-backed state providers."""

from __future__ import annotations

import math
import time

from src.core.vehicle_state import VehicleState
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class GroundTruthStateProvider:
    """Read ego pose and velocity directly from a CARLA vehicle actor."""

    def __init__(self, vehicle: "carla.Vehicle") -> None:
        self._vehicle = vehicle

    def get_state(self) -> VehicleState:
        transform = self._vehicle.get_transform()
        velocity = self._vehicle.get_velocity()
        speed = math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)
        try:
            timestamp = float(self._vehicle.get_world().get_snapshot().timestamp.elapsed_seconds)
        except RuntimeError:
            timestamp = time.monotonic()
        return VehicleState(
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(transform.location.z),
            yaw=float(transform.rotation.yaw),
            speed=float(speed),
            timestamp=timestamp,
            vx_mps=float(velocity.x),
            vy_mps=float(velocity.y),
            source_filter_id="ground_truth",
            model_type="GROUND_TRUTH",
            safe_for_autonomous_control=True,
        )
