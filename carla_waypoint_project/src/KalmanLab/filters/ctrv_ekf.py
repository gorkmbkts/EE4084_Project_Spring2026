"""Constant Turn Rate and Velocity EKF plugin for KalmanLab."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, TYPE_CHECKING

import numpy as np

from config.settings import AUTONOMOUS_CONTROL
from src.KalmanLab.filter_base import FilterControlInput, TRACKING_MODE_ACTIVE, normalize_tracking_mode
from src.core.vehicle_state import VehicleState
from src.evaluation.benchmark_config import ParameterSpec
from src.localization.gnss_projection import GnssLocalProjector, LocalGnssMeasurement

if TYPE_CHECKING:
    from src.sensors.gnss_sensor import GnssMeasurement
    from src.sensors.imu_sensor import ImuMeasurement


FILTER_INFO = {
    "id": "ctrv_ekf",
    "name": "CTRV EKF",
    "type": "Extended Kalman Filter",
    "state_vector": "[px, py, yaw, speed, yaw_rate]^T",
    "process_model": "Constant Turn Rate and Velocity",
    "measurement_model": "GNSS position x/y + IMU compass yaw + IMU gyro z yaw-rate",
    "description": "Nonlinear CTRV EKF using projected GNSS and raw IMU heading/yaw-rate measurements.",
    "model_type": "CTRV",
    "provided_state_fields": (
        "x",
        "y",
        "z",
        "yaw",
        "speed",
        "timestamp",
        "vx_mps",
        "vy_mps",
        "yaw_rate_radps",
        "curvature_1pm",
        "covariance_diagonal",
        "raw_state_vector",
    ),
    "safe_for_autonomous_control": True,
    "active_tracking_supported": True,
    "benchmark_selectable": True,
    "experimental": True,
    "requires_raw_imu": True,
    "autonomous_control_note": "CTRV EKF is experimental; verify gyro_z_sign and compass_yaw_offset_deg on turns before relying on benchmark scores.",
}


TUNE = {
    "process_accel_stddev_mps2": 3.0,
    "process_yaw_accel_stddev_radps2": 0.7,
    "gnss_position_stddev_m": 1.25,
    "imu_yaw_stddev_deg": 5.0,
    "imu_yaw_rate_stddev_radps": 0.08,
    "initial_position_stddev_m": 4.0,
    "initial_yaw_stddev_deg": 25.0,
    "initial_speed_stddev_mps": 4.0,
    "initial_yaw_rate_stddev_radps": 0.5,
    "yaw_from_velocity_min_speed_mps": 0.35,
    "min_prediction_dt_s": 1.0e-4,
    "max_prediction_dt_s": 0.20,
    "turn_rate_epsilon_radps": 1.0e-4,
    "gyro_z_sign": 1.0,
    "compass_yaw_offset_deg": -90.0,
    "max_abs_yaw_rate_radps": 2.5,
    "max_speed_mps": 50.0,
    "enable_control_input_prediction": 1.0,
    "control_accel_gain_mps2": 1.2,
    "control_brake_decel_gain_mps2": 2.4,
    "control_steer_to_yaw_rate_gain": 0.25,
    "control_input_timeout_s": 0.35,
    "max_control_yaw_rate_delta_radps": 0.12,
    "max_control_speed_delta_mps": 0.35,
}


TUNE_SPECS = (
    ParameterSpec("process_accel_stddev_mps2", "Process accel", 0.05, 12.0, "m/s2", 2, "Noise"),
    ParameterSpec("process_yaw_accel_stddev_radps2", "Process yaw accel", 0.02, 4.0, "rad/s2", 2, "Noise"),
    ParameterSpec("gnss_position_stddev_m", "GNSS position", 0.10, 12.0, "m", 2, "Noise"),
    ParameterSpec("imu_yaw_stddev_deg", "IMU yaw", 0.5, 45.0, "deg", 1, "Noise"),
    ParameterSpec("imu_yaw_rate_stddev_radps", "IMU yaw rate", 0.005, 1.0, "rad/s", 3, "Noise"),
    ParameterSpec("initial_position_stddev_m", "Initial pos", 0.25, 25.0, "m", 2, "Initialization"),
    ParameterSpec("initial_yaw_stddev_deg", "Initial yaw", 1.0, 90.0, "deg", 1, "Initialization"),
    ParameterSpec("initial_speed_stddev_mps", "Initial speed", 0.10, 20.0, "m/s", 2, "Initialization"),
    ParameterSpec("initial_yaw_rate_stddev_radps", "Initial yaw rate", 0.01, 2.5, "rad/s", 2, "Initialization"),
    ParameterSpec("yaw_from_velocity_min_speed_mps", "Yaw min speed", 0.05, 3.0, "m/s", 2, "Yaw"),
    ParameterSpec("min_prediction_dt_s", "Min dt", 0.00001, 0.02, "s", 5, "Prediction"),
    ParameterSpec("max_prediction_dt_s", "Max dt", 0.02, 0.60, "s", 2, "Prediction"),
    ParameterSpec("turn_rate_epsilon_radps", "Turn eps", 0.000001, 0.02, "rad/s", 5, "Prediction"),
    ParameterSpec("gyro_z_sign", "Gyro z sign", -1.0, 1.0, "", 0, "IMU convention"),
    ParameterSpec("compass_yaw_offset_deg", "Compass offset", -180.0, 180.0, "deg", 0, "IMU convention"),
    ParameterSpec("max_abs_yaw_rate_radps", "Yaw-rate cap", 0.2, 5.0, "rad/s", 2, "Guards"),
    ParameterSpec("max_speed_mps", "Speed cap", 5.0, 80.0, "m/s", 1, "Guards"),
    ParameterSpec("enable_control_input_prediction", "Use control input", 0.0, 1.0, "", 0, "Active tracking"),
    ParameterSpec("control_accel_gain_mps2", "Control accel", 0.0, 5.0, "m/s2", 2, "Active tracking"),
    ParameterSpec("control_brake_decel_gain_mps2", "Control brake", 0.0, 8.0, "m/s2", 2, "Active tracking"),
    ParameterSpec("control_steer_to_yaw_rate_gain", "Steer yaw gain", 0.0, 1.0, "x", 2, "Active tracking"),
    ParameterSpec("control_input_timeout_s", "Control timeout", 0.02, 1.0, "s", 2, "Active tracking"),
    ParameterSpec("max_control_yaw_rate_delta_radps", "Control yaw delta", 0.0, 1.0, "rad/s", 2, "Active tracking"),
    ParameterSpec("max_control_speed_delta_mps", "Control speed delta", 0.0, 2.0, "m/s", 2, "Active tracking"),
)


AUTO_TUNE_PROFILE = {
    "enabled": True,
    "primary": [
        {"key": "process_accel_stddev_mps2", "scale": "log", "min": 0.20, "max": 8.0},
        {"key": "process_yaw_accel_stddev_radps2", "scale": "log", "min": 0.05, "max": 2.5},
        {"key": "gnss_position_stddev_m", "scale": "log", "min": 0.40, "max": 6.0},
        {"key": "imu_yaw_rate_stddev_radps", "scale": "log", "min": 0.01, "max": 0.5},
    ],
    "secondary": [
        {"key": "imu_yaw_stddev_deg", "scale": "log", "min": 1.0, "max": 20.0},
    ],
    "search": {
        "default_trials": 30,
        "strategy": "random_plus_coordinate_refinement",
    },
    "objective": "rmse_consistency",
}


def normalize_angle_rad(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi]."""
    if not math.isfinite(float(angle)):
        return 0.0
    wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped == -math.pi and angle > 0.0:
        return math.pi
    return float(wrapped)


def normalize_angle_deg(angle: float) -> float:
    """Wrap an angle in degrees to [-180, 180]."""
    if not math.isfinite(float(angle)):
        return 0.0
    wrapped = (float(angle) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0 and angle > 0.0:
        return 180.0
    return float(wrapped)


def yaw_rad_from_compass(compass_rad: float, offset_deg: float = -90.0) -> Optional[float]:
    """Convert CARLA IMU compass radians to this project's yaw convention."""
    if not math.isfinite(float(compass_rad)):
        return None
    return normalize_angle_rad(float(compass_rad) + math.radians(float(offset_deg)))


def process_model(
    state_vector: np.ndarray,
    dt: float,
    turn_rate_epsilon_radps: float = 1.0e-4,
) -> np.ndarray:
    """CTRV discrete process model for [px, py, yaw, speed, yaw_rate]."""
    state = np.asarray(state_vector, dtype=float).reshape(-1)
    px, py, yaw, speed, yaw_rate = (float(value) for value in state[:5])
    dt = max(0.0, float(dt))
    yaw = normalize_angle_rad(yaw)

    if abs(yaw_rate) > max(1.0e-12, float(turn_rate_epsilon_radps)):
        next_yaw = yaw + yaw_rate * dt
        px += speed / yaw_rate * (math.sin(next_yaw) - math.sin(yaw))
        py += speed / yaw_rate * (-math.cos(next_yaw) + math.cos(yaw))
        yaw = next_yaw
    else:
        px += speed * math.cos(yaw) * dt
        py += speed * math.sin(yaw) * dt

    return np.array(
        [px, py, normalize_angle_rad(yaw), speed, yaw_rate],
        dtype=float,
    )


def numerical_jacobian(
    function: object,
    state_vector: np.ndarray,
    step: float = 1.0e-5,
) -> np.ndarray:
    """Return a central-difference Jacobian for a small nonlinear state."""
    state = np.asarray(state_vector, dtype=float).reshape(-1)
    base = np.asarray(function(state), dtype=float).reshape(-1)
    jacobian = np.zeros((base.size, state.size), dtype=float)
    for index in range(state.size):
        delta = float(step) * max(1.0, abs(float(state[index])))
        plus = state.copy()
        minus = state.copy()
        plus[index] += delta
        minus[index] -= delta
        y_plus = np.asarray(function(plus), dtype=float).reshape(-1)
        y_minus = np.asarray(function(minus), dtype=float).reshape(-1)
        diff = y_plus - y_minus
        if diff.size >= 3:
            diff[2] = normalize_angle_rad(float(y_plus[2] - y_minus[2]))
        jacobian[:, index] = diff / (2.0 * delta)
    return jacobian


@dataclass(frozen=True)
class _CtrvSnapshot:
    px: float
    py: float
    yaw_rad: float
    speed: float
    yaw_rate_radps: float
    timestamp: float


class _CtrvEkfCore:
    """EKF core for [px, py, yaw, speed, yaw_rate]^T."""

    def __init__(self, tune: dict[str, float]) -> None:
        self._accel_var = float(tune["process_accel_stddev_mps2"]) ** 2
        self._yaw_accel_var = float(tune["process_yaw_accel_stddev_radps2"]) ** 2
        self._position_var = float(tune["gnss_position_stddev_m"]) ** 2
        self._yaw_var = math.radians(float(tune["imu_yaw_stddev_deg"])) ** 2
        self._yaw_rate_var = float(tune["imu_yaw_rate_stddev_radps"]) ** 2
        self._initial_position_var = float(tune["initial_position_stddev_m"]) ** 2
        self._initial_yaw_var = math.radians(float(tune["initial_yaw_stddev_deg"])) ** 2
        self._initial_speed_var = float(tune["initial_speed_stddev_mps"]) ** 2
        self._initial_yaw_rate_var = float(tune["initial_yaw_rate_stddev_radps"]) ** 2
        self._turn_rate_epsilon = float(tune["turn_rate_epsilon_radps"])
        self._max_abs_yaw_rate = max(0.01, float(tune["max_abs_yaw_rate_radps"]))
        self._max_speed = max(0.1, float(tune["max_speed_mps"]))

        self._x = np.zeros((5, 1), dtype=float)
        self._p = np.eye(5, dtype=float)
        self._timestamp: Optional[float] = None

        self.initialized = False
        self.last_update_type: Optional[str] = None
        self.last_innovation: Optional[list[float]] = None
        self.last_nis: Optional[float] = None
        self.last_runtime_warning: Optional[str] = None
        self.latest_predicted_state: Optional[list[float]] = None
        self.innovations_by_type: dict[str, list[float]] = {}
        self.nis_by_type: dict[str, float] = {}

    @property
    def timestamp(self) -> Optional[float]:
        return self._timestamp

    @property
    def state_vector(self) -> np.ndarray:
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._p.copy()

    def reset(self) -> None:
        self._x = np.zeros((5, 1), dtype=float)
        self._p = np.eye(5, dtype=float)
        self._timestamp = None
        self.initialized = False
        self.last_update_type = None
        self.last_innovation = None
        self.last_nis = None
        self.last_runtime_warning = None
        self.latest_predicted_state = None
        self.innovations_by_type.clear()
        self.nis_by_type.clear()

    def initialize(
        self,
        position_xy: tuple[float, float],
        yaw_rad: float,
        speed_mps: float,
        yaw_rate_radps: float,
        timestamp: float,
    ) -> None:
        self._x = np.array(
            [
                [float(position_xy[0])],
                [float(position_xy[1])],
                [normalize_angle_rad(float(yaw_rad))],
                [self._clamp(float(speed_mps), 0.0, self._max_speed)],
                [self._clamp(float(yaw_rate_radps), -self._max_abs_yaw_rate, self._max_abs_yaw_rate)],
            ],
            dtype=float,
        )
        self._p = np.diag(
            [
                self._initial_position_var,
                self._initial_position_var,
                self._initial_yaw_var,
                self._initial_speed_var,
                self._initial_yaw_rate_var,
            ]
        ).astype(float)
        self._timestamp = float(timestamp)
        self.initialized = True
        self.last_update_type = "initialize"
        self.last_innovation = None
        self.last_nis = None
        self.last_runtime_warning = None
        self.latest_predicted_state = [float(value) for value in self._x.reshape(-1)]

    def predict(self, dt: float, timestamp: Optional[float] = None) -> None:
        if not self.initialized:
            return

        dt = max(0.0, float(dt))
        if dt <= 0.0:
            if timestamp is not None:
                self._timestamp = float(timestamp)
            return

        flat_state = self._x.reshape(-1)
        transition = lambda state: process_model(state, dt, self._turn_rate_epsilon)
        predicted = transition(flat_state).reshape(5, 1)
        f = numerical_jacobian(transition, flat_state)
        q = self._process_noise(float(self._x[2, 0]), dt)
        p = f @ self._p @ f.T + q

        if not self._set_state_and_covariance(predicted, p, "predict"):
            return

        self.latest_predicted_state = [float(value) for value in self._x.reshape(-1)]
        if timestamp is not None:
            self._timestamp = float(timestamp)
        elif self._timestamp is not None:
            self._timestamp += dt

    def update_gnss_position(self, position_xy: tuple[float, float]) -> None:
        if not self.initialized:
            return
        if not all(math.isfinite(float(value)) for value in position_xy):
            self._skip_update("gnss_position", "non-finite GNSS position")
            return

        z = np.array([[float(position_xy[0])], [float(position_xy[1])]], dtype=float)
        h = np.zeros((2, 5), dtype=float)
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        r = np.eye(2, dtype=float) * self._position_var
        self.safe_update(z, h, r, "gnss_position")

    def update_yaw(self, yaw_rad: float) -> None:
        if not self.initialized:
            return
        if not math.isfinite(float(yaw_rad)):
            self._skip_update("imu_yaw", "non-finite IMU yaw")
            return

        z = np.array([[normalize_angle_rad(float(yaw_rad))]], dtype=float)
        h = np.zeros((1, 5), dtype=float)
        h[0, 2] = 1.0
        r = np.array([[self._yaw_var]], dtype=float)
        self.safe_update(z, h, r, "imu_yaw", wrap_yaw_innovation=True)

    def update_yaw_rate(self, yaw_rate_radps: float) -> None:
        if not self.initialized:
            return
        if not math.isfinite(float(yaw_rate_radps)):
            self._skip_update("imu_yaw_rate", "non-finite IMU yaw-rate")
            return

        clamped = self._clamp(float(yaw_rate_radps), -self._max_abs_yaw_rate, self._max_abs_yaw_rate)
        z = np.array([[clamped]], dtype=float)
        h = np.zeros((1, 5), dtype=float)
        h[0, 4] = 1.0
        r = np.array([[self._yaw_rate_var]], dtype=float)
        self.safe_update(z, h, r, "imu_yaw_rate")

    def apply_control_prediction(self, speed_delta_mps: float, yaw_rate_delta_radps: float) -> bool:
        """Apply a bounded active-tracking prediction nudge, not a measurement."""
        if not self.initialized:
            return False
        if not math.isfinite(float(speed_delta_mps)) or not math.isfinite(float(yaw_rate_delta_radps)):
            self.last_runtime_warning = "control prediction skipped: non-finite delta"
            return False

        next_x = self._x.copy()
        next_x[3, 0] = float(next_x[3, 0]) + float(speed_delta_mps)
        next_x[4, 0] = float(next_x[4, 0]) + float(yaw_rate_delta_radps)
        next_x = self._sanitize_state(next_x)
        next_p = self._p.copy()
        next_p[3, 3] += max(1.0e-9, abs(float(speed_delta_mps)) * 0.05)
        next_p[4, 4] += max(1.0e-9, abs(float(yaw_rate_delta_radps)) * 0.05)
        return self._set_state_and_covariance(next_x, next_p, "control_prediction")

    def snapshot(self) -> Optional[_CtrvSnapshot]:
        if not self.initialized or self._timestamp is None:
            return None
        return _CtrvSnapshot(
            px=float(self._x[0, 0]),
            py=float(self._x[1, 0]),
            yaw_rad=normalize_angle_rad(float(self._x[2, 0])),
            speed=self._clamp(float(self._x[3, 0]), 0.0, self._max_speed),
            yaw_rate_radps=self._clamp(float(self._x[4, 0]), -self._max_abs_yaw_rate, self._max_abs_yaw_rate),
            timestamp=float(self._timestamp),
        )

    def safe_update(
        self,
        z: np.ndarray,
        h: np.ndarray,
        r: np.ndarray,
        update_type: str,
        wrap_yaw_innovation: bool = False,
    ) -> None:
        prediction = h @ self._x
        innovation = z - prediction
        if wrap_yaw_innovation:
            innovation[0, 0] = normalize_angle_rad(float(innovation[0, 0]))

        innovation_cov = h @ self._p @ h.T + r
        try:
            solved_innovation = np.linalg.solve(innovation_cov, innovation)
            gain = np.linalg.solve(innovation_cov.T, (self._p @ h.T).T).T
        except (np.linalg.LinAlgError, ValueError) as exc:
            self._skip_update(update_type, f"innovation solve failed: {exc}")
            return

        next_x = self._x + gain @ innovation
        next_x = self._sanitize_state(next_x)

        identity = np.eye(self._p.shape[0], dtype=float)
        residual_transform = identity - gain @ h
        next_p = residual_transform @ self._p @ residual_transform.T + gain @ r @ gain.T

        if not self._set_state_and_covariance(next_x, next_p, update_type):
            return

        innovation_list = [float(value) for value in innovation.reshape(-1)]
        nis = float((innovation.T @ solved_innovation)[0, 0])
        self.last_update_type = update_type
        self.last_innovation = innovation_list
        self.last_nis = nis
        self.last_runtime_warning = None
        self.innovations_by_type[update_type] = innovation_list
        self.nis_by_type[update_type] = nis

    def _process_noise(self, yaw: float, dt: float) -> np.ndarray:
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        accel_gain = np.array(
            [
                [0.5 * cos_yaw * dt * dt],
                [0.5 * sin_yaw * dt * dt],
                [0.0],
                [dt],
                [0.0],
            ],
            dtype=float,
        )
        yaw_accel_gain = np.array(
            [[0.0], [0.0], [0.5 * dt * dt], [0.0], [dt]],
            dtype=float,
        )
        q = self._accel_var * (accel_gain @ accel_gain.T)
        q += self._yaw_accel_var * (yaw_accel_gain @ yaw_accel_gain.T)
        q[0, 0] += 0.25 * self._accel_var * dt**4
        q[1, 1] += 0.25 * self._accel_var * dt**4
        q += np.eye(5, dtype=float) * 1.0e-12
        return self._symmetrize(q)

    def _set_state_and_covariance(self, state: np.ndarray, covariance: np.ndarray, context: str) -> bool:
        state = self._sanitize_state(state)
        covariance = self._symmetrize(covariance)

        if not np.all(np.isfinite(state)):
            self.last_runtime_warning = f"{context} skipped: non-finite state"
            return False
        if not np.all(np.isfinite(covariance)):
            self.last_runtime_warning = f"{context} skipped: non-finite covariance"
            return False

        diagonal = np.diag(covariance)
        if np.any(diagonal < 0.0):
            covariance = covariance.copy()
            for index, value in enumerate(np.diag(covariance)):
                covariance[index, index] = max(1.0e-9, float(value))

        self._x = state
        self._p = self._symmetrize(covariance)
        return True

    def _sanitize_state(self, state: np.ndarray) -> np.ndarray:
        result = np.asarray(state, dtype=float).reshape(5, 1).copy()
        result[2, 0] = normalize_angle_rad(float(result[2, 0]))
        result[3, 0] = self._clamp(float(result[3, 0]), 0.0, self._max_speed)
        result[4, 0] = self._clamp(float(result[4, 0]), -self._max_abs_yaw_rate, self._max_abs_yaw_rate)
        return result

    def _skip_update(self, update_type: str, reason: str) -> None:
        self.last_update_type = f"{update_type}_skipped"
        self.last_innovation = None
        self.last_nis = None
        self.last_runtime_warning = reason

    @staticmethod
    def _symmetrize(matrix: np.ndarray) -> np.ndarray:
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))


class Filter:
    """Self-contained CTRV EKF plugin with GNSS and raw IMU updates."""

    def __init__(
        self,
        gnss_projector: GnssLocalProjector,
        tune: Optional[dict[str, object]] = None,
        tracking_mode: str = "passive",
    ) -> None:
        self._gnss_projector = gnss_projector
        self._tune = dict(TUNE)
        if tune:
            self._tune.update(dict(tune))
        self._tracking_mode = normalize_tracking_mode(tracking_mode)
        self._filter = _CtrvEkfCore(self._tune)
        self._min_prediction_dt_s = float(self._tune["min_prediction_dt_s"])
        self._max_prediction_dt_s = float(self._tune["max_prediction_dt_s"])
        self._yaw_speed_threshold = float(self._tune["yaw_from_velocity_min_speed_mps"])
        self._gyro_z_sign = float(self._tune["gyro_z_sign"])
        self._compass_yaw_offset_deg = float(self._tune["compass_yaw_offset_deg"])
        self._max_abs_yaw_rate = float(self._tune["max_abs_yaw_rate_radps"])
        self._enable_control_input_prediction = bool(float(self._tune["enable_control_input_prediction"]) >= 0.5)
        self._control_timeout_s = float(self._tune["control_input_timeout_s"])
        self._control_accel_gain = float(self._tune["control_accel_gain_mps2"])
        self._control_brake_decel_gain = float(self._tune["control_brake_decel_gain_mps2"])
        self._control_steer_yaw_gain = float(self._tune["control_steer_to_yaw_rate_gain"])
        self._max_control_yaw_rate_delta = float(self._tune["max_control_yaw_rate_delta_radps"])
        self._max_control_speed_delta = float(self._tune["max_control_speed_delta_mps"])

        self._latest_state: Optional[VehicleState] = None
        self._latest_gnss_local: Optional[LocalGnssMeasurement] = None
        self._latest_compass_yaw_rad: Optional[float] = None
        self._latest_compass_yaw_deg: Optional[float] = None
        self._latest_gyro_z_radps: Optional[float] = None
        self._latest_signed_gyro_z_radps: Optional[float] = None
        self._gyro_yaw_rate_clamped = False
        self._latest_control_input: Optional[FilterControlInput] = None
        self._active_command_used_latest_prediction = False
        self._control_predicted_accel_mps2: Optional[float] = None
        self._control_predicted_yaw_rate_radps: Optional[float] = None
        self._control_input_age_s: Optional[float] = None
        self._control_prediction_reason = "waiting for active prediction"
        self._last_imu_frame: Optional[int] = None
        self._last_gnss_frame: Optional[int] = None

    @property
    def initialized(self) -> bool:
        return self._filter.initialized

    @property
    def latest_gnss_local(self) -> Optional[LocalGnssMeasurement]:
        return self._latest_gnss_local

    def reset(self) -> None:
        self._filter.reset()
        self._latest_state = None
        self._latest_gnss_local = None
        self._latest_compass_yaw_rad = None
        self._latest_compass_yaw_deg = None
        self._latest_gyro_z_radps = None
        self._latest_signed_gyro_z_radps = None
        self._gyro_yaw_rate_clamped = False
        self._latest_control_input = None
        self._active_command_used_latest_prediction = False
        self._control_predicted_accel_mps2 = None
        self._control_predicted_yaw_rate_radps = None
        self._control_input_age_s = None
        self._control_prediction_reason = "reset"
        self._last_imu_frame = None
        self._last_gnss_frame = None

    def process_imu(self, imu: "ImuMeasurement") -> Optional[VehicleState]:
        yaw_rad = yaw_rad_from_compass(float(imu.compass), self._compass_yaw_offset_deg)
        if yaw_rad is not None:
            self._latest_compass_yaw_rad = yaw_rad
            self._latest_compass_yaw_deg = normalize_angle_deg(math.degrees(yaw_rad))

        yaw_rate_radps = self._yaw_rate_measurement_from_imu(imu)
        self._last_imu_frame = int(imu.frame)

        if not self._filter.initialized:
            return self._latest_state

        self._predict_to(float(imu.timestamp))
        if yaw_rad is not None:
            self._filter.update_yaw(yaw_rad)
        if yaw_rate_radps is not None:
            self._filter.update_yaw_rate(yaw_rate_radps)
        return self._refresh_state_from_filter()

    def process_gnss(self, gnss: "GnssMeasurement") -> Optional[VehicleState]:
        local = self._gnss_projector.project(gnss)
        if local is None:
            return self._latest_state

        self._latest_gnss_local = local
        self._last_gnss_frame = int(gnss.frame)

        if not self._filter.initialized:
            yaw_rad = self._latest_compass_yaw_rad if self._latest_compass_yaw_rad is not None else 0.0
            yaw_rate = self._latest_signed_gyro_z_radps if self._latest_signed_gyro_z_radps is not None else 0.0
            self._filter.initialize(
                position_xy=(local.x, local.y),
                yaw_rad=yaw_rad,
                speed_mps=0.0,
                yaw_rate_radps=yaw_rate,
                timestamp=local.timestamp,
            )
            return self._refresh_state_from_filter()

        self._predict_to(local.timestamp)
        self._filter.update_gnss_position((local.x, local.y))
        return self._refresh_state_from_filter()

    def process_control(self, control_input: FilterControlInput) -> bool:
        self._latest_control_input = control_input
        return self._tracking_mode == TRACKING_MODE_ACTIVE and self._enable_control_input_prediction

    def get_state(self) -> Optional[VehicleState]:
        return self._latest_state

    def get_diagnostics(self) -> dict[str, object]:
        snapshot = self._filter.snapshot()
        covariance = self._filter.covariance
        state_vector = [float(value) for value in self._filter.state_vector.reshape(-1)]
        yaw_rate = snapshot.yaw_rate_radps if snapshot is not None else None
        speed = snapshot.speed if snapshot is not None else None
        curvature = self._curvature(yaw_rate, speed)
        innovations = self._filter.innovations_by_type
        yaw_innovation = self._first_innovation(innovations.get("imu_yaw"))

        return {
            "filter_id": FILTER_INFO["id"],
            "model_type": "CTRV",
            "initialized": self.initialized,
            "safe_for_autonomous_control": bool(FILTER_INFO["safe_for_autonomous_control"]),
            "state_vector": state_vector,
            "covariance_diagonal": [float(value) for value in np.diag(covariance)],
            "yaw_deg": normalize_angle_deg(math.degrees(snapshot.yaw_rad)) if snapshot is not None else None,
            "speed_mps": speed,
            "yaw_rate_radps": yaw_rate,
            "yaw_rate_dps": math.degrees(yaw_rate) if yaw_rate is not None else None,
            "curvature_1pm": curvature,
            "latest_compass_yaw_deg": self._latest_compass_yaw_deg,
            "latest_compass_yaw_rad": self._latest_compass_yaw_rad,
            "latest_gyro_z_radps": self._latest_gyro_z_radps,
            "latest_signed_gyro_z_radps": self._latest_signed_gyro_z_radps,
            "applied_gyro_z_sign": self._gyro_z_sign,
            "gyro_yaw_rate_clamped": self._gyro_yaw_rate_clamped,
            "yaw_innovation_deg": math.degrees(yaw_innovation) if yaw_innovation is not None else None,
            "yaw_rate_innovation_radps": self._first_innovation(innovations.get("imu_yaw_rate")),
            "gnss_innovation": innovations.get("gnss_position"),
            "innovation": self._filter.last_innovation,
            "innovations_by_type": dict(innovations),
            "nis": self._filter.last_nis,
            "nis_by_type": dict(self._filter.nis_by_type),
            "last_update_type": self._filter.last_update_type,
            "latest_predicted_state": self._filter.latest_predicted_state,
            "latest_gnss_local": self._local_gnss_dict(self._latest_gnss_local),
            "last_gnss_frame": self._last_gnss_frame,
            "last_imu_frame": self._last_imu_frame,
            "tracking_mode": self._tracking_mode,
            "active_tracking_supported": True,
            "active_command_used_latest_prediction": self._active_command_used_latest_prediction,
            "control_predicted_accel_mps2": self._control_predicted_accel_mps2,
            "control_predicted_yaw_rate_radps": self._control_predicted_yaw_rate_radps,
            "control_input_age_s": self._control_input_age_s,
            "control_prediction_reason": self._control_prediction_reason,
            "runtime_warning": self._filter.last_runtime_warning,
            "yaw_sign_diagnostic": self._yaw_sign_diagnostic(yaw_innovation),
            "timestamp": snapshot.timestamp if snapshot is not None else None,
            "note": "CTRV EKF. Uses linear GNSS x/y, compass yaw, and raw gyro z yaw-rate. Experimental: verify IMU sign tuning before relying on benchmark scores.",
        }

    def _predict_to(self, timestamp: float) -> None:
        current_timestamp = self._filter.timestamp
        if current_timestamp is None:
            return

        dt = float(timestamp) - float(current_timestamp)
        if dt <= self._min_prediction_dt_s:
            return

        clipped_dt = min(dt, self._max_prediction_dt_s)
        self._filter.predict(dt=clipped_dt, timestamp=timestamp)
        self._apply_control_prediction(clipped_dt, timestamp)

    def _apply_control_prediction(self, dt: float, timestamp: float) -> None:
        self._active_command_used_latest_prediction = False
        self._control_predicted_accel_mps2 = None
        self._control_predicted_yaw_rate_radps = None
        self._control_input_age_s = None
        if self._tracking_mode != TRACKING_MODE_ACTIVE:
            self._control_prediction_reason = "passive tracking mode"
            return
        if not self._enable_control_input_prediction:
            self._control_prediction_reason = "control input prediction disabled"
            return
        if self._latest_control_input is None:
            self._control_prediction_reason = "no control input"
            return

        age = max(0.0, float(timestamp) - float(self._latest_control_input.timestamp))
        self._control_input_age_s = age
        if age > max(0.0, self._control_timeout_s):
            self._control_prediction_reason = "control input timed out"
            return

        snapshot = self._filter.snapshot()
        if snapshot is None:
            self._control_prediction_reason = "filter not initialized"
            return

        throttle = self._clamp(float(self._latest_control_input.throttle), 0.0, 1.0)
        brake = self._clamp(float(self._latest_control_input.brake), 0.0, 1.0)
        steer = self._clamp(float(self._latest_control_input.steer), -1.0, 1.0)
        accel = throttle * max(0.0, self._control_accel_gain) - brake * max(0.0, self._control_brake_decel_gain)
        if self._latest_control_input.reverse:
            accel = -accel
        speed_delta = self._clamp(accel * max(0.0, dt), -self._max_control_speed_delta, self._max_control_speed_delta)

        steer_angle_rad = steer * math.radians(AUTONOMOUS_CONTROL.max_steer_angle_deg)
        yaw_rate_target = 0.0
        if abs(steer_angle_rad) > 1.0e-6 and snapshot.speed > 0.05:
            yaw_rate_target = snapshot.speed / max(0.1, float(AUTONOMOUS_CONTROL.wheel_base_m)) * math.tan(steer_angle_rad)
        yaw_rate_delta = (yaw_rate_target - snapshot.yaw_rate_radps) * max(0.0, self._control_steer_yaw_gain)
        yaw_rate_delta = self._clamp(
            yaw_rate_delta,
            -self._max_control_yaw_rate_delta,
            self._max_control_yaw_rate_delta,
        )

        if self._filter.apply_control_prediction(speed_delta, yaw_rate_delta):
            self._active_command_used_latest_prediction = True
            self._control_predicted_accel_mps2 = accel
            self._control_predicted_yaw_rate_radps = snapshot.yaw_rate_radps + yaw_rate_delta
            self._control_prediction_reason = "control prediction applied"
        else:
            self._control_prediction_reason = "control prediction rejected by filter core"

    def _refresh_state_from_filter(self) -> Optional[VehicleState]:
        snapshot = self._filter.snapshot()
        if snapshot is None:
            self._latest_state = None
            return None

        z = self._latest_gnss_local.z if self._latest_gnss_local is not None else 0.0
        yaw_deg = normalize_angle_deg(math.degrees(snapshot.yaw_rad))
        speed = max(0.0, float(snapshot.speed))
        curvature = self._curvature(snapshot.yaw_rate_radps, speed)
        state_vector = self._filter.state_vector.reshape(-1)
        covariance = self._filter.covariance
        self._latest_state = VehicleState(
            x=float(snapshot.px),
            y=float(snapshot.py),
            z=float(z),
            yaw=yaw_deg,
            speed=speed,
            timestamp=float(snapshot.timestamp),
            vx_mps=speed * math.cos(snapshot.yaw_rad),
            vy_mps=speed * math.sin(snapshot.yaw_rad),
            yaw_rate_radps=float(snapshot.yaw_rate_radps),
            curvature_1pm=curvature,
            covariance_diagonal=tuple(float(value) for value in np.diag(covariance)),
            source_filter_id=FILTER_INFO["id"],
            model_type=FILTER_INFO["model_type"],
            raw_state_vector=tuple(float(value) for value in state_vector),
            diagnostics_summary={
                "last_update_type": self._filter.last_update_type,
                "active_command_used": self._active_command_used_latest_prediction,
                "control_prediction_reason": self._control_prediction_reason,
            },
            safe_for_autonomous_control=bool(FILTER_INFO["safe_for_autonomous_control"]),
            active_tracking_supported=bool(FILTER_INFO["active_tracking_supported"]),
        )
        return self._latest_state

    def _yaw_rate_measurement_from_imu(self, imu: "ImuMeasurement") -> Optional[float]:
        raw_gyro_z = getattr(imu, "gyro_z_radps", None)
        if raw_gyro_z is None:
            raw_gyro_z = imu.gyroscope[2]
        try:
            raw_value = float(raw_gyro_z)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(raw_value):
            return None

        signed = self._gyro_z_sign * raw_value
        self._latest_gyro_z_radps = raw_value
        self._gyro_yaw_rate_clamped = abs(signed) > self._max_abs_yaw_rate
        signed = self._clamp(signed, -self._max_abs_yaw_rate, self._max_abs_yaw_rate)
        self._latest_signed_gyro_z_radps = signed
        return signed

    def _curvature(self, yaw_rate: Optional[float], speed: Optional[float]) -> Optional[float]:
        if yaw_rate is None or speed is None or not math.isfinite(yaw_rate) or not math.isfinite(speed):
            return None
        if abs(speed) < max(1.0e-6, self._yaw_speed_threshold):
            return None
        return float(yaw_rate / speed)

    def _yaw_sign_diagnostic(self, yaw_innovation: Optional[float]) -> str:
        if self._latest_compass_yaw_deg is None or self._latest_signed_gyro_z_radps is None:
            return "waiting for compass and gyro"
        if yaw_innovation is None:
            return "waiting for yaw innovation"
        if abs(yaw_innovation) > math.radians(45.0) and abs(self._latest_signed_gyro_z_radps) > 0.15:
            return "large yaw innovation while turning; verify gyro_z_sign and compass_yaw_offset_deg"
        return "gyro z sign and compass yaw appear bounded"

    @staticmethod
    def _first_innovation(values: Optional[list[float]]) -> Optional[float]:
        if not values:
            return None
        try:
            number = float(values[0])
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _local_gnss_dict(local: Optional[LocalGnssMeasurement]) -> Optional[dict[str, object]]:
        if local is None:
            return None
        return {
            "x": local.x,
            "y": local.y,
            "z": local.z,
            "latitude": local.latitude,
            "longitude": local.longitude,
            "altitude": local.altitude,
            "frame": local.frame,
            "timestamp": local.timestamp,
        }

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))
