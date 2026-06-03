"""Small command-to-motion helper used by active-tracking filters."""

from __future__ import annotations

from dataclasses import dataclass
import math

from config.settings import AUTONOMOUS_CONTROL
from src.KalmanLab.filter_base import FilterControlInput


@dataclass(frozen=True)
class CommandMotionEstimate:
    """World-frame acceleration and yaw-rate implied by one control command."""

    acceleration_xy: tuple[float, float]
    yaw_rate_dps: float
    longitudinal_accel_mps2: float
    lateral_accel_mps2: float


def estimate_command_motion(
    control_input: FilterControlInput,
    speed_mps: float,
    yaw_deg: float,
    tune: dict[str, object],
    dt_s: float = 0.0,
) -> CommandMotionEstimate:
    """Convert applied CARLA control into a bounded world-frame acceleration."""
    throttle = _clamp(float(control_input.throttle), 0.0, 1.0)
    brake = _clamp(float(control_input.brake), 0.0, 1.0)
    steer = _clamp(float(control_input.steer), -1.0, 1.0)
    speed = max(0.0, float(speed_mps))

    throttle_gain = _float_tune(tune, "command_throttle_accel_gain_mps2", 3.0)
    brake_gain = _float_tune(tune, "command_brake_decel_gain_mps2", 6.0)
    max_accel = max(0.1, _float_tune(tune, "command_max_accel_mps2", 8.0))
    max_yaw_rate_dps = max(1.0, _float_tune(tune, "command_max_yaw_rate_dps", 90.0))

    longitudinal = throttle * max(0.0, throttle_gain) - brake * max(0.0, brake_gain)
    if control_input.reverse:
        longitudinal = -longitudinal
    longitudinal = _clamp(longitudinal, -max_accel, max_accel)

    steer_angle_rad = steer * math.radians(AUTONOMOUS_CONTROL.max_steer_angle_deg)
    yaw_rate_rad_s = 0.0
    wheel_base = max(0.1, float(AUTONOMOUS_CONTROL.wheel_base_m))
    if abs(steer_angle_rad) > 1.0e-6 and speed > 0.05:
        yaw_rate_rad_s = speed / wheel_base * math.tan(steer_angle_rad)
    yaw_rate_dps = _clamp(math.degrees(yaw_rate_rad_s), -max_yaw_rate_dps, max_yaw_rate_dps)
    yaw_rate_rad_s = math.radians(yaw_rate_dps)

    lateral = _clamp(speed * yaw_rate_rad_s, -max_accel, max_accel)
    yaw_mid_rad = math.radians(_normalize_angle_deg(yaw_deg + 0.5 * yaw_rate_dps * max(0.0, dt_s)))
    cos_yaw = math.cos(yaw_mid_rad)
    sin_yaw = math.sin(yaw_mid_rad)

    world_ax = cos_yaw * longitudinal - sin_yaw * lateral
    world_ay = sin_yaw * longitudinal + cos_yaw * lateral
    magnitude = math.hypot(world_ax, world_ay)
    if magnitude > max_accel:
        scale = max_accel / magnitude
        world_ax *= scale
        world_ay *= scale

    return CommandMotionEstimate(
        acceleration_xy=(float(world_ax), float(world_ay)),
        yaw_rate_dps=float(yaw_rate_dps),
        longitudinal_accel_mps2=float(longitudinal),
        lateral_accel_mps2=float(lateral),
    )


def _float_tune(tune: dict[str, object], key: str, default: float) -> float:
    try:
        value = float(tune.get(key, default))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return float(value)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _normalize_angle_deg(angle_deg: float) -> float:
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return float(angle_deg)
