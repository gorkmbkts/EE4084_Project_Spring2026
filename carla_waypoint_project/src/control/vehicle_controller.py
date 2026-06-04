"""Route-following vehicle controller for an abstract ego state."""

from __future__ import annotations

import math
from typing import Optional

from config.settings import AUTONOMOUS_CONTROL
from src.localization.motion_info import MotionInfo
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
        behavior_config: Optional[object] = None,
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
        self._behavior_config = behavior_config
        self._filtered_motion_curvature: Optional[float] = None
        self._latest_model_control_diagnostics: dict[str, object] = {
            "model_aware_control_enabled": False,
            "motion_info_used": False,
            "motion_info_ignored_reason": "not evaluated",
        }

    @property
    def latest_model_control_diagnostics(self) -> dict[str, object]:
        """Return diagnostics for the optional model-aware control path."""
        return dict(self._latest_model_control_diagnostics)

    def compute_control(
        self,
        state: EgoState,
        target_waypoint: Optional["carla.Waypoint"],
        route_completed: bool = False,
        target_speed_mps: Optional[float] = None,
        motion_info: Optional[MotionInfo] = None,
    ) -> "carla.VehicleControl":
        """Compute a CARLA control command for the current route target."""
        if route_completed or target_waypoint is None:
            self._latest_model_control_diagnostics = {
                "model_aware_control_enabled": self._model_aware_enabled(),
                "motion_info_used": False,
                "motion_info_ignored_reason": "route completed or no target waypoint",
            }
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=False)

        pure_pursuit_steer = self._compute_steer(state, target_waypoint)
        steer, motion_curvature, diagnostics = self._apply_model_aware_steering(
            state=state,
            pure_pursuit_steer=pure_pursuit_steer,
            motion_info=motion_info,
        )
        guarded_target_speed = self._model_guarded_target_speed(
            state=state,
            steer=steer,
            target_speed_mps=target_speed_mps,
            motion_curvature=motion_curvature,
            diagnostics=diagnostics,
        )
        throttle, brake = self._compute_speed_control(state, steer, guarded_target_speed)
        diagnostics.update(
            {
                "requested_target_speed_mps": target_speed_mps,
                "effective_target_speed_mps": guarded_target_speed,
                "throttle": throttle,
                "brake": brake,
                "final_steer": steer,
            }
        )
        self._latest_model_control_diagnostics = diagnostics
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

    def _compute_speed_control(
        self,
        state: EgoState,
        steer: float,
        target_speed_mps: Optional[float],
    ) -> tuple[float, float]:
        target_speed = self._target_speed_for_steer(steer) if target_speed_mps is None else max(0.0, target_speed_mps)
        speed_error = target_speed - state.speed
        if speed_error >= 0.0:
            throttle = self._clamp(self._speed_kp * speed_error, 0.0, self._max_throttle)
            return throttle, 0.0

        brake = self._clamp(self._brake_kp * abs(speed_error), 0.0, self._max_brake)
        return 0.0, brake

    def _apply_model_aware_steering(
        self,
        state: EgoState,
        pure_pursuit_steer: float,
        motion_info: Optional[MotionInfo],
    ) -> tuple[float, Optional[float], dict[str, object]]:
        enabled = self._model_aware_enabled()
        diagnostics: dict[str, object] = {
            "model_aware_control_enabled": enabled,
            "motion_info_available": motion_info is not None,
            "motion_info_used": False,
            "motion_info_ignored_reason": "",
            "source_filter_id": motion_info.source_filter_id if motion_info is not None else "",
            "model_type": motion_info.model_type if motion_info is not None else "",
            "pure_pursuit_steer": pure_pursuit_steer,
            "model_steer_correction": 0.0,
            "yaw_rate_feedback_active": False,
            "yaw_rate_feedback_note": "yaw-rate feedback is not active without a route desired yaw-rate signal",
        }
        if not enabled:
            diagnostics["motion_info_ignored_reason"] = "model-aware control disabled"
            return pure_pursuit_steer, None, diagnostics

        curvature, reason = self._motion_curvature_for_control(state, motion_info)
        if curvature is None:
            diagnostics["motion_info_ignored_reason"] = reason
            return pure_pursuit_steer, None, diagnostics

        alpha = self._clamp(self._config_value("motion_info_lowpass_alpha", 0.25), 0.02, 1.0)
        if self._filtered_motion_curvature is None:
            filtered_curvature = curvature
        else:
            filtered_curvature = self._filtered_motion_curvature + alpha * (curvature - self._filtered_motion_curvature)
        self._filtered_motion_curvature = filtered_curvature

        steer_ff_angle = math.atan(self._wheel_base_m * filtered_curvature)
        steer_ff_normalized = steer_ff_angle / max(1.0e-6, self._max_steer_angle_rad)
        correction = self._config_value("yaw_rate_feedforward_gain", 0.25) * steer_ff_normalized
        max_correction = max(0.0, self._config_value("max_model_steer_correction", 0.15))
        correction = self._clamp(correction, -max_correction, max_correction)
        steer = self._clamp(pure_pursuit_steer + correction, -self._max_steer, self._max_steer)

        diagnostics.update(
            {
                "motion_info_used": True,
                "motion_info_ignored_reason": "",
                "motion_curvature_1pm": curvature,
                "filtered_motion_curvature_1pm": filtered_curvature,
                "model_steer_ff_angle_rad": steer_ff_angle,
                "model_steer_ff_normalized": steer_ff_normalized,
                "model_steer_correction": correction,
            }
        )
        return steer, filtered_curvature, diagnostics

    def _model_guarded_target_speed(
        self,
        state: EgoState,
        steer: float,
        target_speed_mps: Optional[float],
        motion_curvature: Optional[float],
        diagnostics: dict[str, object],
    ) -> Optional[float]:
        diagnostics["model_speed_guard_enabled"] = self._flag_value("enable_model_speed_guard", False)
        diagnostics["model_speed_guard_applied"] = False
        diagnostics["model_speed_safe_mps"] = None
        if not self._flag_value("enable_model_speed_guard", False):
            return target_speed_mps
        if motion_curvature is None or not math.isfinite(float(motion_curvature)):
            return target_speed_mps

        abs_curvature = abs(float(motion_curvature))
        if abs_curvature < 1.0e-6:
            return target_speed_mps

        base_target = self._target_speed_for_steer(steer) if target_speed_mps is None else max(0.0, float(target_speed_mps))
        safe_cornering = max(0.1, self._config_value("safe_cornering_factor", 1.0))
        lateral_accel = max(0.25, self._config_value("max_forward_accel_mps2", 1.6) * safe_cornering)
        speed_factor = max(0.1, self._config_value("model_curvature_speed_factor", 0.5))
        safe_speed = math.sqrt(lateral_accel / abs_curvature) * speed_factor
        guarded = min(base_target, safe_speed)
        diagnostics["model_speed_safe_mps"] = safe_speed
        diagnostics["model_speed_guard_applied"] = guarded < base_target
        return guarded

    def _motion_curvature_for_control(
        self,
        state: EgoState,
        motion_info: Optional[MotionInfo],
    ) -> tuple[Optional[float], str]:
        if motion_info is None:
            return None, "motion info missing"
        if motion_info.confidence is not None and motion_info.confidence < 0.2:
            return None, "motion info confidence below threshold"

        min_speed = max(0.0, self._config_value("min_model_control_speed_mps", 1.0))
        if state.speed < min_speed:
            return None, "speed below model-aware control threshold"

        yaw_rate = motion_info.yaw_rate_radps
        yaw_rate_cap = max(0.01, self._config_value("max_abs_motion_yaw_rate_radps", 2.5))
        if yaw_rate is not None:
            if not math.isfinite(float(yaw_rate)):
                return None, "non-finite motion yaw-rate"
            if abs(float(yaw_rate)) > yaw_rate_cap:
                return None, "motion yaw-rate exceeds configured cap"

        curvature = motion_info.curvature_1pm
        if curvature is None and yaw_rate is not None:
            curvature = float(yaw_rate) / max(float(state.speed), min_speed, 1.0e-6)
        if curvature is None:
            return None, "motion curvature unavailable"
        if not math.isfinite(float(curvature)):
            return None, "non-finite motion curvature"
        return float(curvature), ""

    def _model_aware_enabled(self) -> bool:
        return self._flag_value("enable_model_aware_control", False)

    def _flag_value(self, attribute: str, default: bool) -> bool:
        value = self._config_value(attribute, 1.0 if default else 0.0)
        return float(value) >= 0.5

    def _config_value(self, attribute: str, default: float) -> float:
        if self._behavior_config is None:
            return float(default)
        try:
            value = float(getattr(self._behavior_config, attribute))
        except (AttributeError, TypeError, ValueError):
            return float(default)
        return value if math.isfinite(value) else float(default)

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
