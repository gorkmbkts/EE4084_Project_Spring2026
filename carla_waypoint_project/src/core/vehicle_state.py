"""Shared vehicle-state value used across filters, control, UI, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class VehicleState:
    """Application-level state estimate.

    Yaw is in degrees in the project/CARLA convention.  Speed is in m/s.
    Optional fields are capabilities: consumers must check availability before
    using them and fall back to the mandatory fields when absent.
    """

    x: float
    y: float
    z: float
    yaw: float
    speed: float
    timestamp: float
    vx_mps: Optional[float] = None
    vy_mps: Optional[float] = None
    acceleration_mps2: Optional[float] = None
    longitudinal_accel_mps2: Optional[float] = None
    lateral_accel_mps2: Optional[float] = None
    yaw_rate_radps: Optional[float] = None
    curvature_1pm: Optional[float] = None
    covariance_diagonal: Optional[tuple[float, ...]] = None
    position_covariance_2x2: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
    confidence: Optional[float] = None
    source_filter_id: str = ""
    model_type: str = ""
    raw_state_vector: Optional[tuple[float, ...]] = None
    diagnostics_summary: Mapping[str, Any] = field(default_factory=dict)
    safe_for_autonomous_control: bool = True
    active_tracking_supported: bool = False

    def __post_init__(self) -> None:
        for name in ("x", "y", "z", "yaw", "speed", "timestamp"):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in (
            "vx_mps",
            "vy_mps",
            "acceleration_mps2",
            "longitudinal_accel_mps2",
            "lateral_accel_mps2",
            "yaw_rate_radps",
            "curvature_1pm",
            "confidence",
        ):
            object.__setattr__(self, name, finite_or_none(getattr(self, name)))
        object.__setattr__(self, "covariance_diagonal", finite_tuple_or_none(self.covariance_diagonal))
        object.__setattr__(self, "position_covariance_2x2", finite_2x2_or_none(self.position_covariance_2x2))
        object.__setattr__(self, "raw_state_vector", finite_tuple_or_none(self.raw_state_vector))
        object.__setattr__(self, "source_filter_id", str(self.source_filter_id or ""))
        object.__setattr__(self, "model_type", str(self.model_type or ""))
        object.__setattr__(self, "safe_for_autonomous_control", bool(self.safe_for_autonomous_control))
        object.__setattr__(self, "active_tracking_supported", bool(self.active_tracking_supported))
        object.__setattr__(self, "diagnostics_summary", dict(self.diagnostics_summary or {}))

    @property
    def yaw_rad(self) -> float:
        return math.radians(float(self.yaw))

    @property
    def has_velocity(self) -> bool:
        return self.vx_mps is not None and self.vy_mps is not None

    @property
    def has_yaw_rate(self) -> bool:
        return self.yaw_rate_radps is not None

    @property
    def has_curvature(self) -> bool:
        return self.curvature_1pm is not None

    @property
    def has_acceleration(self) -> bool:
        return (
            self.acceleration_mps2 is not None
            or self.longitudinal_accel_mps2 is not None
            or self.lateral_accel_mps2 is not None
        )

    def capabilities(self) -> tuple[str, ...]:
        fields: list[str] = []
        if self.has_velocity:
            fields.append("velocity")
        if self.has_yaw_rate:
            fields.append("yaw_rate")
        if self.has_curvature:
            fields.append("curvature")
        if self.has_acceleration:
            fields.append("acceleration")
        if self.covariance_diagonal is not None:
            fields.append("covariance")
        if self.position_covariance_2x2 is not None:
            fields.append("position_covariance_2x2")
        return tuple(fields)

    def distance_xy_to(self, location: object) -> float:
        """Return planar distance to any object with CARLA-like x/y attrs."""
        return math.hypot(float(getattr(location, "x")) - self.x, float(getattr(location, "y")) - self.y)


def finite_or_none(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def finite_tuple_or_none(value: object) -> Optional[tuple[float, ...]]:
    if not isinstance(value, (tuple, list)):
        return None
    result: list[float] = []
    for item in value:
        number = finite_or_none(item)
        if number is None:
            return None
        result.append(number)
    return tuple(result)


def finite_2x2_or_none(value: object) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    rows: list[tuple[float, float]] = []
    for row in value[:2]:
        if not isinstance(row, (tuple, list)) or len(row) < 2:
            return None
        first = finite_or_none(row[0])
        second = finite_or_none(row[1])
        if first is None or second is None:
            return None
        rows.append((first, second))
    return (rows[0], rows[1])
