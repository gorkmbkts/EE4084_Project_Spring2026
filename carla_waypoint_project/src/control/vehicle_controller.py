"""Ground-truth route-following vehicle controller."""

from __future__ import annotations

import math
from typing import Optional

from config.settings import AUTONOMOUS_CONTROL
from src.localization.state_estimator import EgoState
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class VehicleController:
    """Pure Pursuit lateral control with proportional speed control."""

    def __init__(
        self,
        target_speed_mps: float = AUTONOMOUS_CONTROL.target_speed_mps,
        turn_speed_mps: float = AUTONOMOUS_CONTROL.turn_speed_mps,
        wheel_base_m: float = AUTONOMOUS_CONTROL.wheel_base_m,
        max_steer: float = AUTONOMOUS_CONTROL.max_steer,
        max_steer_angle_deg: float = AUTONOMOUS_CONTROL.max_steer_angle_deg,
        steering_gain: float = AUTONOMOUS_CONTROL.steering_gain,
        turn_slowdown_steer_threshold: float = AUTONOMOUS_CONTROL.turn_slowdown_steer_threshold,
        sharp_turn_steer_threshold: float = AUTONOMOUS_CONTROL.sharp_turn_steer_threshold,
        max_throttle: float = AUTONOMOUS_CONTROL.max_throttle,
        max_brake: float = AUTONOMOUS_CONTROL.max_brake,
        speed_kp: float = AUTONOMOUS_CONTROL.speed_kp,
        brake_kp: float = AUTONOMOUS_CONTROL.brake_kp,
    ) -> None:
        self._target_speed_mps = target_speed_mps
        self._turn_speed_mps = turn_speed_mps
        self._wheel_base_m = wheel_base_m
        self._max_steer = max_steer
        self._max_steer_angle_rad = math.radians(max_steer_angle_deg)
        self._steering_gain = steering_gain
        self._turn_slowdown_steer_threshold = turn_slowdown_steer_threshold
        self._sharp_turn_steer_threshold = sharp_turn_steer_threshold
        self._max_throttle = max_throttle
        self._max_brake = max_brake
        self._speed_kp = speed_kp
        self._brake_kp = brake_kp

    def compute_control(
        self,
        state: EgoState,
        target_waypoint: Optional["carla.Waypoint"],
        route_completed: bool = False,
    ) -> "carla.VehicleControl":
        """Compute a CARLA control command for the current route target."""
        if route_completed or target_waypoint is None:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=False)

        steer = self._compute_steer(state, target_waypoint)
        throttle, brake = self._compute_speed_control(state, steer)
        return carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )

    def _compute_steer(self, state: EgoState, target_waypoint: "carla.Waypoint") -> float:
        target_location = target_waypoint.transform.location
        dx = target_location.x - state.x
        dy = target_location.y - state.y
        yaw_rad = math.radians(state.yaw)

        local_x = math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy
        local_y = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
        lookahead_distance = max(0.1, math.hypot(local_x, local_y))

        if local_x < 0.1:
            if abs(local_y) < 0.1:
                return 0.0
            return self._max_steer if local_y > 0.0 else -self._max_steer

        alpha = math.atan2(local_y, local_x)
        curvature = 2.0 * math.sin(alpha) / lookahead_distance
        steer_angle = math.atan(self._wheel_base_m * curvature)
        normalized = self._steering_gain * steer_angle / self._max_steer_angle_rad
        return self._clamp(normalized, -self._max_steer, self._max_steer)

    def _compute_speed_control(self, state: EgoState, steer: float) -> tuple[float, float]:
        target_speed = self._target_speed_for_steer(steer)
        speed_error = target_speed - state.speed
        if speed_error >= 0.0:
            throttle = self._clamp(self._speed_kp * speed_error, 0.0, self._max_throttle)
            return throttle, 0.0

        brake = self._clamp(self._brake_kp * abs(speed_error), 0.0, self._max_brake)
        return 0.0, brake

    def _target_speed_for_steer(self, steer: float) -> float:
        abs_steer = abs(steer)
        low = self._turn_slowdown_steer_threshold
        high = max(low + 0.01, self._sharp_turn_steer_threshold)
        if abs_steer <= low:
            return self._target_speed_mps
        if abs_steer >= high:
            return self._turn_speed_mps

        ratio = (abs_steer - low) / (high - low)
        return self._target_speed_mps + ratio * (self._turn_speed_mps - self._target_speed_mps)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))
