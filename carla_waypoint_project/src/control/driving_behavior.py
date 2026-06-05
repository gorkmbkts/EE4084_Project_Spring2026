"""Curvature-aware speed planning and actuator realism for autonomous driving."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Deque, Optional, Sequence

from config.settings import AUTONOMOUS_CONTROL
from src.core.vehicle_state import VehicleState
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


@dataclass
class DrivingBehaviorConfig:
    """Mutable live-tuning values shared by planner, actuator, and UI."""

    max_speed_mps: float = 8.0
    min_curve_speed_mps: float = AUTONOMOUS_CONTROL.turn_speed_mps
    max_forward_accel_mps2: float = 1.6
    max_braking_decel_mps2: float = 3.2
    max_steer_rate_per_s: float = 1.8
    throttle_smoothing: float = 0.35
    brake_smoothing: float = 0.28
    steering_smoothing: float = 0.30
    curve_lookahead_m: float = 28.0
    curvature_sensitivity: float = 1.15
    speed_change_aggressiveness: float = 1.0
    actuator_noise: float = 0.008
    actuator_delay_s: float = 0.08
    throttle_response_gain: float = 1.0
    brake_response_gain: float = 1.0
    steering_response_gain: float = 1.0
    safe_cornering_factor: float = 1.0
    max_throttle_rate_per_s: float = 1.8
    max_brake_rate_per_s: float = 2.4
    enable_model_aware_control: float = 0.0
    yaw_rate_feedforward_gain: float = 0.25
    yaw_rate_feedback_gain: float = 0.0
    max_model_steer_correction: float = 0.15
    min_model_control_speed_mps: float = 1.0
    model_state_lowpass_alpha: float = 0.25
    max_abs_motion_yaw_rate_radps: float = 2.5
    enable_model_speed_guard: float = 0.0
    model_curvature_speed_factor: float = 0.5
    enable_acceleration_feedforward: float = 0.0
    acceleration_feedforward_gain: float = 0.0
    max_acceleration_feedforward_delta: float = 0.15


@dataclass(frozen=True)
class SpeedPlan:
    """Latest speed-planner output for control and diagnostics."""

    target_speed_mps: float
    raw_target_speed_mps: float
    curvature_score: float
    curvature_rad_per_m: float
    lookahead_distance_m: float
    mode: str


class CurvatureSpeedPlanner:
    """Plan smooth target speed from upcoming waypoint curvature."""

    def __init__(self, config: DrivingBehaviorConfig) -> None:
        self._config = config
        self._smoothed_target_speed_mps = max(0.0, float(config.max_speed_mps))
        self._latest_plan = SpeedPlan(
            target_speed_mps=self._smoothed_target_speed_mps,
            raw_target_speed_mps=self._smoothed_target_speed_mps,
            curvature_score=0.0,
            curvature_rad_per_m=0.0,
            lookahead_distance_m=0.0,
            mode="STRAIGHT",
        )

    @property
    def latest_plan(self) -> SpeedPlan:
        return self._latest_plan

    def reset(self, initial_speed_mps: float = 0.0) -> None:
        self._smoothed_target_speed_mps = self._clamp(
            initial_speed_mps,
            0.0,
            max(0.0, self._config.max_speed_mps),
        )
        self._latest_plan = SpeedPlan(
            target_speed_mps=self._smoothed_target_speed_mps,
            raw_target_speed_mps=self._smoothed_target_speed_mps,
            curvature_score=0.0,
            curvature_rad_per_m=0.0,
            lookahead_distance_m=0.0,
            mode="STRAIGHT",
        )

    def plan(
        self,
        state: VehicleState,
        preview_waypoints: Sequence["carla.Waypoint"],
        route_completed: bool,
        dt_seconds: float,
    ) -> SpeedPlan:
        """Return a smoothed target speed for the current route preview."""
        if route_completed:
            raw_target = 0.0
            curvature_score = 1.0
            curvature_rad_per_m = 0.0
            lookahead_distance_m = 0.0
            mode = "STOPPING"
        else:
            curvature_score, curvature_rad_per_m, lookahead_distance_m = self._curvature_score(
                state,
                preview_waypoints,
            )
            severity = self._clamp(curvature_score * self._config.safe_cornering_factor, 0.0, 1.0)
            max_speed = max(0.1, float(self._config.max_speed_mps))
            min_curve_speed = self._clamp(
                self._config.min_curve_speed_mps,
                0.0,
                max_speed,
            )
            raw_target = max_speed - severity * (max_speed - min_curve_speed)
            mode = self._mode_for_curvature(curvature_score)

        smoothed_target = self._move_towards_speed(
            current=self._smoothed_target_speed_mps,
            target=raw_target,
            dt_seconds=dt_seconds,
        )
        self._smoothed_target_speed_mps = smoothed_target
        self._latest_plan = SpeedPlan(
            target_speed_mps=smoothed_target,
            raw_target_speed_mps=raw_target,
            curvature_score=curvature_score,
            curvature_rad_per_m=curvature_rad_per_m,
            lookahead_distance_m=lookahead_distance_m,
            mode=mode,
        )
        return self._latest_plan

    def _curvature_score(
        self,
        state: VehicleState,
        preview_waypoints: Sequence["carla.Waypoint"],
    ) -> tuple[float, float, float]:
        del state
        max_distance = max(3.0, float(self._config.curve_lookahead_m))
        points: list[tuple[float, float]] = []
        distance = 0.0

        for waypoint in preview_waypoints:
            location = waypoint.transform.location
            x = float(location.x)
            y = float(location.y)
            if not points:
                points.append((x, y))
                continue

            previous_x, previous_y = points[-1]
            step = math.hypot(x - previous_x, y - previous_y)
            if step < 0.05:
                continue
            distance += step
            points.append((x, y))
            if distance >= max_distance:
                break

        if len(points) < 4 or distance <= 0.5:
            return 0.0, 0.0, distance

        headings: list[float] = []
        segment_lengths: list[float] = []
        for first, second in zip(points, points[1:]):
            dx = second[0] - first[0]
            dy = second[1] - first[1]
            length = math.hypot(dx, dy)
            if length < 0.25:
                continue
            headings.append(math.atan2(dy, dx))
            segment_lengths.append(length)

        if len(headings) < 2:
            return 0.0, 0.0, distance

        total_heading_change = 0.0
        max_local_curvature = 0.0
        for index, (first_heading, second_heading) in enumerate(zip(headings, headings[1:])):
            delta = abs(self._normalize_angle_rad(second_heading - first_heading))
            total_heading_change += delta
            local_distance = max(0.5, 0.5 * (segment_lengths[index] + segment_lengths[index + 1]))
            max_local_curvature = max(max_local_curvature, delta / local_distance)

        average_curvature = total_heading_change / max(distance, 1.0)
        combined_curvature = max(average_curvature, 0.55 * max_local_curvature)
        score_from_total_turn = total_heading_change / (math.pi * 0.65)
        score_from_curvature = combined_curvature * max_distance / (math.pi * 0.65)
        score = max(score_from_total_turn, score_from_curvature)
        score *= max(0.05, float(self._config.curvature_sensitivity))
        return self._clamp(score, 0.0, 1.0), combined_curvature, distance

    def _move_towards_speed(
        self,
        current: float,
        target: float,
        dt_seconds: float,
    ) -> float:
        dt = max(0.001, float(dt_seconds))
        aggressiveness = max(0.05, float(self._config.speed_change_aggressiveness))
        if target >= current:
            max_delta = max(0.05, self._config.max_forward_accel_mps2) * dt * aggressiveness
        else:
            max_delta = max(0.05, self._config.max_braking_decel_mps2) * dt * aggressiveness
        delta = self._clamp(target - current, -max_delta, max_delta)
        return self._clamp(current + delta, 0.0, max(0.1, self._config.max_speed_mps))

    @staticmethod
    def _mode_for_curvature(curvature_score: float) -> str:
        if curvature_score < 0.18:
            return "STRAIGHT"
        if curvature_score < 0.55:
            return "APPROACHING CURVE"
        return "CURVE"

    @staticmethod
    def _normalize_angle_rad(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))


class ActuatorRealism:
    """Apply actuator delay, smoothing, rate limits, and subtle noise."""

    def __init__(self, config: DrivingBehaviorConfig, seed: int = 4084) -> None:
        self._config = config
        self._rng = random.Random(seed)
        self._command_buffer: Deque["carla.VehicleControl"] = deque()
        self._applied_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        self._last_requested_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        self._noise_throttle = 0.0
        self._noise_brake = 0.0
        self._noise_steer = 0.0

    @property
    def latest_applied_control(self) -> "carla.VehicleControl":
        return self._clone_control(self._applied_control)

    @property
    def latest_requested_control(self) -> "carla.VehicleControl":
        return self._clone_control(self._last_requested_control)

    def reset(self, control: Optional["carla.VehicleControl"] = None) -> None:
        base = control if control is not None else carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        self._command_buffer.clear()
        self._applied_control = self._clone_control(base)
        self._last_requested_control = self._clone_control(base)
        self._noise_throttle = 0.0
        self._noise_brake = 0.0
        self._noise_steer = 0.0

    def apply(
        self,
        requested_control: "carla.VehicleControl",
        dt_seconds: float,
    ) -> "carla.VehicleControl":
        """Transform an ideal control request into a believable applied command."""
        dt = max(0.001, float(dt_seconds))
        self._last_requested_control = self._clone_control(requested_control)
        delayed = self._delayed_command(requested_control, dt)

        desired_throttle = self._clamp(
            delayed.throttle * max(0.0, self._config.throttle_response_gain),
            0.0,
            AUTONOMOUS_CONTROL.max_throttle,
        )
        desired_brake = self._clamp(
            delayed.brake * max(0.0, self._config.brake_response_gain),
            0.0,
            AUTONOMOUS_CONTROL.max_brake,
        )
        desired_steer = self._clamp(
            delayed.steer * max(0.0, self._config.steering_response_gain),
            -AUTONOMOUS_CONTROL.max_steer,
            AUTONOMOUS_CONTROL.max_steer,
        )

        if desired_brake > 0.02:
            desired_throttle = 0.0

        throttle = self._smooth(self._applied_control.throttle, desired_throttle, self._config.throttle_smoothing)
        brake = self._smooth(self._applied_control.brake, desired_brake, self._config.brake_smoothing)
        steer = self._smooth(self._applied_control.steer, desired_steer, self._config.steering_smoothing)

        throttle = self._rate_limit(
            self._applied_control.throttle,
            throttle,
            max(0.05, self._config.max_throttle_rate_per_s),
            dt,
        )
        brake = self._rate_limit(
            self._applied_control.brake,
            brake,
            max(0.05, self._config.max_brake_rate_per_s),
            dt,
        )
        steer = self._rate_limit(
            self._applied_control.steer,
            steer,
            max(0.05, self._config.max_steer_rate_per_s),
            dt,
        )

        throttle, brake, steer = self._add_smooth_noise(throttle, brake, steer)
        if brake > 0.02:
            throttle = min(throttle, 0.02)

        applied = carla.VehicleControl(
            throttle=self._clamp(throttle, 0.0, AUTONOMOUS_CONTROL.max_throttle),
            steer=self._clamp(steer, -AUTONOMOUS_CONTROL.max_steer, AUTONOMOUS_CONTROL.max_steer),
            brake=self._clamp(brake, 0.0, AUTONOMOUS_CONTROL.max_brake),
            hand_brake=bool(delayed.hand_brake),
            reverse=bool(delayed.reverse),
            manual_gear_shift=bool(delayed.manual_gear_shift),
        )
        self._applied_control = self._clone_control(applied)
        return applied

    def _delayed_command(
        self,
        requested_control: "carla.VehicleControl",
        dt_seconds: float,
    ) -> "carla.VehicleControl":
        delay_steps = max(0, int(round(max(0.0, self._config.actuator_delay_s) / dt_seconds)))
        command = self._clone_control(requested_control)
        if delay_steps <= 0:
            self._command_buffer.clear()
            return command

        self._command_buffer.append(command)
        while len(self._command_buffer) > delay_steps + 1:
            self._command_buffer.popleft()
        if len(self._command_buffer) <= delay_steps:
            return self._clone_control(self._applied_control)
        return self._clone_control(self._command_buffer[0])

    def _add_smooth_noise(self, throttle: float, brake: float, steer: float) -> tuple[float, float, float]:
        magnitude = max(0.0, float(self._config.actuator_noise))
        if magnitude <= 0.0:
            return throttle, brake, steer

        self._noise_throttle = self._smooth_noise_step(self._noise_throttle, magnitude)
        self._noise_brake = self._smooth_noise_step(self._noise_brake, magnitude * 0.6)
        self._noise_steer = self._smooth_noise_step(self._noise_steer, magnitude)
        return throttle + self._noise_throttle, brake + self._noise_brake, steer + self._noise_steer

    def _smooth_noise_step(self, current: float, magnitude: float) -> float:
        target = self._rng.uniform(-magnitude, magnitude)
        return current + 0.08 * (target - current)

    @staticmethod
    def _smooth(current: float, target: float, smoothing: float) -> float:
        lag = max(0.0, min(0.95, float(smoothing)))
        alpha = max(0.02, 1.0 - lag)
        return current + alpha * (target - current)

    @staticmethod
    def _rate_limit(current: float, target: float, max_rate_per_s: float, dt_seconds: float) -> float:
        max_delta = max(0.0, max_rate_per_s) * max(0.001, dt_seconds)
        return current + ActuatorRealism._clamp(target - current, -max_delta, max_delta)

    @staticmethod
    def _clone_control(control: "carla.VehicleControl") -> "carla.VehicleControl":
        return carla.VehicleControl(
            throttle=float(getattr(control, "throttle", 0.0)),
            steer=float(getattr(control, "steer", 0.0)),
            brake=float(getattr(control, "brake", 0.0)),
            hand_brake=bool(getattr(control, "hand_brake", False)),
            reverse=bool(getattr(control, "reverse", False)),
            manual_gear_shift=bool(getattr(control, "manual_gear_shift", False)),
            gear=int(getattr(control, "gear", 0)),
        )

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))
