"""Constant Turn Rate and Acceleration EKF plugin for KalmanLab."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, TYPE_CHECKING

import numpy as np

from config.settings import AUTONOMOUS_CONTROL
from src.KalmanLab.filter_base import FilterControlInput, TRACKING_MODE_ACTIVE, normalize_tracking_mode
from src.evaluation.benchmark_config import ParameterSpec
from src.localization.gnss_projection import GnssLocalProjector, LocalGnssMeasurement
from src.localization.state_estimator import EgoState

if TYPE_CHECKING:
    from src.sensors.gnss_sensor import GnssMeasurement
    from src.sensors.imu_sensor import ImuMeasurement


FILTER_INFO = {
    "id": "ctra_ekf",
    "name": "CTRA EKF",
    "type": "Extended Kalman Filter",
    "state_vector": "[px, py, yaw, speed, acceleration, yaw_rate]^T",
    "process_model": "Constant Turn Rate and Acceleration with RK4 integration",
    "measurement_model": "GNSS position x/y + IMU compass yaw + IMU gyro z yaw-rate + raw longitudinal acceleration",
    "description": "Nonlinear CTRA EKF using projected GNSS and raw IMU heading, yaw-rate, and longitudinal acceleration.",
    "model_type": "CTRA",
    "motion_info_fields": ("yaw_rate_radps", "longitudinal_accel_mps2", "curvature_1pm"),
    "safe_for_autonomous_control": True,
    "active_tracking_supported": True,
    "benchmark_selectable": True,
    "experimental": True,
    "requires_raw_imu": True,
    "autonomous_control_note": "CTRA EKF is experimental; verify gyro_z_sign, compass_yaw_offset_deg, and acceleration axis/sign before relying on benchmark scores.",
}


TUNE = {
    "process_jerk_stddev_mps3": 2.0,
    "process_yaw_accel_stddev_radps2": 0.7,
    "gnss_position_stddev_m": 1.25,
    "imu_yaw_stddev_deg": 5.0,
    "imu_yaw_rate_stddev_radps": 0.08,
    "imu_accel_stddev_mps2": 0.8,
    "imu_accel_bias_mps2": 0.0,
    "initial_position_stddev_m": 4.0,
    "initial_yaw_stddev_deg": 25.0,
    "initial_speed_stddev_mps": 4.0,
    "initial_accel_stddev_mps2": 2.0,
    "initial_yaw_rate_stddev_radps": 0.5,
    "yaw_from_velocity_min_speed_mps": 0.35,
    "min_prediction_dt_s": 1.0e-4,
    "max_prediction_dt_s": 0.20,
    "turn_rate_epsilon_radps": 1.0e-4,
    "gyro_z_sign": 1.0,
    "compass_yaw_offset_deg": -90.0,
    "imu_longitudinal_accel_axis": 0,
    "imu_longitudinal_accel_sign": 1.0,
    "max_abs_yaw_rate_radps": 2.5,
    "max_abs_accel_mps2": 12.0,
    "max_speed_mps": 50.0,
    "enable_control_input_prediction": 1.0,
    "control_accel_gain_mps2": 1.2,
    "control_brake_decel_gain_mps2": 2.4,
    "control_steer_to_yaw_rate_gain": 0.25,
    "control_input_timeout_s": 0.35,
    "max_control_accel_delta_mps2": 0.35,
    "max_control_yaw_rate_delta_radps": 0.12,
}


TUNE_SPECS = (
    ParameterSpec("process_jerk_stddev_mps3", "Process jerk", 0.05, 12.0, "m/s3", 2, "Noise"),
    ParameterSpec("process_yaw_accel_stddev_radps2", "Process yaw accel", 0.02, 4.0, "rad/s2", 2, "Noise"),
    ParameterSpec("gnss_position_stddev_m", "GNSS position", 0.10, 12.0, "m", 2, "Noise"),
    ParameterSpec("imu_yaw_stddev_deg", "IMU yaw", 0.5, 45.0, "deg", 1, "Noise"),
    ParameterSpec("imu_yaw_rate_stddev_radps", "IMU yaw rate", 0.005, 1.0, "rad/s", 3, "Noise"),
    ParameterSpec("imu_accel_stddev_mps2", "IMU accel", 0.02, 5.0, "m/s2", 2, "Noise"),
    ParameterSpec("imu_accel_bias_mps2", "IMU accel bias", -5.0, 5.0, "m/s2", 2, "IMU convention"),
    ParameterSpec("initial_position_stddev_m", "Initial pos", 0.25, 25.0, "m", 2, "Initialization"),
    ParameterSpec("initial_yaw_stddev_deg", "Initial yaw", 1.0, 90.0, "deg", 1, "Initialization"),
    ParameterSpec("initial_speed_stddev_mps", "Initial speed", 0.10, 20.0, "m/s", 2, "Initialization"),
    ParameterSpec("initial_accel_stddev_mps2", "Initial accel", 0.05, 8.0, "m/s2", 2, "Initialization"),
    ParameterSpec("initial_yaw_rate_stddev_radps", "Initial yaw rate", 0.01, 2.5, "rad/s", 2, "Initialization"),
    ParameterSpec("yaw_from_velocity_min_speed_mps", "Yaw min speed", 0.05, 3.0, "m/s", 2, "Yaw"),
    ParameterSpec("min_prediction_dt_s", "Min dt", 0.00001, 0.02, "s", 5, "Prediction"),
    ParameterSpec("max_prediction_dt_s", "Max dt", 0.02, 0.60, "s", 2, "Prediction"),
    ParameterSpec("turn_rate_epsilon_radps", "Turn eps", 0.000001, 0.02, "rad/s", 5, "Prediction"),
    ParameterSpec("gyro_z_sign", "Gyro z sign", -1.0, 1.0, "", 0, "IMU convention"),
    ParameterSpec("compass_yaw_offset_deg", "Compass offset", -180.0, 180.0, "deg", 0, "IMU convention"),
    ParameterSpec("imu_longitudinal_accel_axis", "Accel axis", 0.0, 2.0, "", 0, "IMU convention"),
    ParameterSpec("imu_longitudinal_accel_sign", "Accel sign", -1.0, 1.0, "", 0, "IMU convention"),
    ParameterSpec("max_abs_yaw_rate_radps", "Yaw-rate cap", 0.2, 5.0, "rad/s", 2, "Guards"),
    ParameterSpec("max_abs_accel_mps2", "Accel cap", 1.0, 25.0, "m/s2", 1, "Guards"),
    ParameterSpec("max_speed_mps", "Speed cap", 5.0, 80.0, "m/s", 1, "Guards"),
    ParameterSpec("enable_control_input_prediction", "Use control input", 0.0, 1.0, "", 0, "Active tracking"),
    ParameterSpec("control_accel_gain_mps2", "Control accel", 0.0, 5.0, "m/s2", 2, "Active tracking"),
    ParameterSpec("control_brake_decel_gain_mps2", "Control brake", 0.0, 8.0, "m/s2", 2, "Active tracking"),
    ParameterSpec("control_steer_to_yaw_rate_gain", "Steer yaw gain", 0.0, 1.0, "x", 2, "Active tracking"),
    ParameterSpec("control_input_timeout_s", "Control timeout", 0.02, 1.0, "s", 2, "Active tracking"),
    ParameterSpec("max_control_accel_delta_mps2", "Control accel delta", 0.0, 2.0, "m/s2", 2, "Active tracking"),
    ParameterSpec("max_control_yaw_rate_delta_radps", "Control yaw delta", 0.0, 1.0, "rad/s", 2, "Active tracking"),
)


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
    """CTRA discrete process model for [px, py, yaw, speed, acceleration, yaw_rate]."""
    state = np.asarray(state_vector, dtype=float).reshape(-1).astype(float)
    dt = max(0.0, float(dt))
    if dt <= 0.0:
        state[2] = normalize_angle_rad(float(state[2]))
        return state

    px, py, yaw, speed, acceleration, yaw_rate = (float(value) for value in state[:6])
    yaw = normalize_angle_rad(yaw)
    if abs(yaw_rate) <= max(1.0e-12, float(turn_rate_epsilon_radps)):
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        px += speed * cos_yaw * dt + 0.5 * acceleration * cos_yaw * dt * dt
        py += speed * sin_yaw * dt + 0.5 * acceleration * sin_yaw * dt * dt
        speed += acceleration * dt
        return np.array([px, py, yaw, speed, acceleration, yaw_rate], dtype=float)

    result = np.array([px, py, yaw, speed, acceleration, yaw_rate], dtype=float)
    steps = max(1, int(math.ceil(dt / 0.02)))
    h = dt / steps
    for _ in range(steps):
        result = _rk4_step(result, h)
        result[2] = normalize_angle_rad(float(result[2]))
    return result


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


def _rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = _derivative(state)
    k2 = _derivative(state + 0.5 * dt * k1)
    k3 = _derivative(state + 0.5 * dt * k2)
    k4 = _derivative(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _derivative(state: np.ndarray) -> np.ndarray:
    yaw = float(state[2])
    speed = float(state[3])
    acceleration = float(state[4])
    yaw_rate = float(state[5])
    return np.array(
        [
            speed * math.cos(yaw),
            speed * math.sin(yaw),
            yaw_rate,
            acceleration,
            0.0,
            0.0,
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class _CtraSnapshot:
    px: float
    py: float
    yaw_rad: float
    speed: float
    acceleration_mps2: float
    yaw_rate_radps: float
    timestamp: float


class _CtraEkfCore:
    """EKF core for [px, py, yaw, speed, acceleration, yaw_rate]^T."""

    def __init__(self, tune: dict[str, float]) -> None:
        self._jerk_var = float(tune["process_jerk_stddev_mps3"]) ** 2
        self._yaw_accel_var = float(tune["process_yaw_accel_stddev_radps2"]) ** 2
        self._position_var = float(tune["gnss_position_stddev_m"]) ** 2
        self._yaw_var = math.radians(float(tune["imu_yaw_stddev_deg"])) ** 2
        self._yaw_rate_var = float(tune["imu_yaw_rate_stddev_radps"]) ** 2
        self._accel_var = float(tune["imu_accel_stddev_mps2"]) ** 2
        self._initial_position_var = float(tune["initial_position_stddev_m"]) ** 2
        self._initial_yaw_var = math.radians(float(tune["initial_yaw_stddev_deg"])) ** 2
        self._initial_speed_var = float(tune["initial_speed_stddev_mps"]) ** 2
        self._initial_accel_var = float(tune["initial_accel_stddev_mps2"]) ** 2
        self._initial_yaw_rate_var = float(tune["initial_yaw_rate_stddev_radps"]) ** 2
        self._turn_rate_epsilon = float(tune["turn_rate_epsilon_radps"])
        self._max_abs_yaw_rate = max(0.01, float(tune["max_abs_yaw_rate_radps"]))
        self._max_abs_accel = max(0.1, float(tune["max_abs_accel_mps2"]))
        self._max_speed = max(0.1, float(tune["max_speed_mps"]))

        self._x = np.zeros((6, 1), dtype=float)
        self._p = np.eye(6, dtype=float)
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
        self._x = np.zeros((6, 1), dtype=float)
        self._p = np.eye(6, dtype=float)
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
        acceleration_mps2: float,
        yaw_rate_radps: float,
        timestamp: float,
    ) -> None:
        self._x = np.array(
            [
                [float(position_xy[0])],
                [float(position_xy[1])],
                [normalize_angle_rad(float(yaw_rad))],
                [self._clamp(float(speed_mps), 0.0, self._max_speed)],
                [self._clamp(float(acceleration_mps2), -self._max_abs_accel, self._max_abs_accel)],
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
                self._initial_accel_var,
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
        predicted = transition(flat_state).reshape(6, 1)
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
        h = np.zeros((2, 6), dtype=float)
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
        h = np.zeros((1, 6), dtype=float)
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
        h = np.zeros((1, 6), dtype=float)
        h[0, 5] = 1.0
        r = np.array([[self._yaw_rate_var]], dtype=float)
        self.safe_update(z, h, r, "imu_yaw_rate")

    def update_acceleration(self, acceleration_mps2: float) -> None:
        if not self.initialized:
            return
        if not math.isfinite(float(acceleration_mps2)):
            self._skip_update("imu_longitudinal_accel", "non-finite IMU acceleration")
            return

        clamped = self._clamp(float(acceleration_mps2), -self._max_abs_accel, self._max_abs_accel)
        z = np.array([[clamped]], dtype=float)
        h = np.zeros((1, 6), dtype=float)
        h[0, 4] = 1.0
        r = np.array([[self._accel_var]], dtype=float)
        self.safe_update(z, h, r, "imu_longitudinal_accel")

    def apply_control_prediction(self, accel_delta_mps2: float, yaw_rate_delta_radps: float) -> bool:
        """Apply a bounded active-tracking prediction nudge, not a measurement."""
        if not self.initialized:
            return False
        if not math.isfinite(float(accel_delta_mps2)) or not math.isfinite(float(yaw_rate_delta_radps)):
            self.last_runtime_warning = "control prediction skipped: non-finite delta"
            return False

        next_x = self._x.copy()
        next_x[4, 0] = float(next_x[4, 0]) + float(accel_delta_mps2)
        next_x[5, 0] = float(next_x[5, 0]) + float(yaw_rate_delta_radps)
        next_x = self._sanitize_state(next_x)
        next_p = self._p.copy()
        next_p[4, 4] += max(1.0e-9, abs(float(accel_delta_mps2)) * 0.05)
        next_p[5, 5] += max(1.0e-9, abs(float(yaw_rate_delta_radps)) * 0.05)
        return self._set_state_and_covariance(next_x, next_p, "control_prediction")

    def snapshot(self) -> Optional[_CtraSnapshot]:
        if not self.initialized or self._timestamp is None:
            return None
        return _CtraSnapshot(
            px=float(self._x[0, 0]),
            py=float(self._x[1, 0]),
            yaw_rad=normalize_angle_rad(float(self._x[2, 0])),
            speed=self._clamp(float(self._x[3, 0]), 0.0, self._max_speed),
            acceleration_mps2=self._clamp(float(self._x[4, 0]), -self._max_abs_accel, self._max_abs_accel),
            yaw_rate_radps=self._clamp(float(self._x[5, 0]), -self._max_abs_yaw_rate, self._max_abs_yaw_rate),
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
        jerk_gain = np.array(
            [
                [(1.0 / 6.0) * cos_yaw * dt**3],
                [(1.0 / 6.0) * sin_yaw * dt**3],
                [0.0],
                [0.5 * dt * dt],
                [dt],
                [0.0],
            ],
            dtype=float,
        )
        yaw_accel_gain = np.array(
            [[0.0], [0.0], [0.5 * dt * dt], [0.0], [0.0], [dt]],
            dtype=float,
        )
        q = self._jerk_var * (jerk_gain @ jerk_gain.T)
        q += self._yaw_accel_var * (yaw_accel_gain @ yaw_accel_gain.T)
        q[0, 0] += (1.0 / 36.0) * self._jerk_var * dt**6
        q[1, 1] += (1.0 / 36.0) * self._jerk_var * dt**6
        q += np.eye(6, dtype=float) * 1.0e-12
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
        result = np.asarray(state, dtype=float).reshape(6, 1).copy()
        result[2, 0] = normalize_angle_rad(float(result[2, 0]))
        result[3, 0] = self._clamp(float(result[3, 0]), 0.0, self._max_speed)
        result[4, 0] = self._clamp(float(result[4, 0]), -self._max_abs_accel, self._max_abs_accel)
        result[5, 0] = self._clamp(float(result[5, 0]), -self._max_abs_yaw_rate, self._max_abs_yaw_rate)
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
    """Self-contained CTRA EKF plugin with GNSS and raw IMU updates."""

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
        self._filter = _CtraEkfCore(self._tune)
        self._min_prediction_dt_s = float(self._tune["min_prediction_dt_s"])
        self._max_prediction_dt_s = float(self._tune["max_prediction_dt_s"])
        self._yaw_speed_threshold = float(self._tune["yaw_from_velocity_min_speed_mps"])
        self._gyro_z_sign = float(self._tune["gyro_z_sign"])
        self._compass_yaw_offset_deg = float(self._tune["compass_yaw_offset_deg"])
        self._accel_axis_index = self._axis_index(self._tune["imu_longitudinal_accel_axis"])
        self._accel_axis_name = ("x", "y", "z")[self._accel_axis_index]
        self._accel_sign = float(self._tune["imu_longitudinal_accel_sign"])
        self._accel_bias = float(self._tune["imu_accel_bias_mps2"])
        self._max_abs_yaw_rate = float(self._tune["max_abs_yaw_rate_radps"])
        self._max_abs_accel = float(self._tune["max_abs_accel_mps2"])
        self._enable_control_input_prediction = bool(float(self._tune["enable_control_input_prediction"]) >= 0.5)
        self._control_timeout_s = float(self._tune["control_input_timeout_s"])
        self._control_accel_gain = float(self._tune["control_accel_gain_mps2"])
        self._control_brake_decel_gain = float(self._tune["control_brake_decel_gain_mps2"])
        self._control_steer_yaw_gain = float(self._tune["control_steer_to_yaw_rate_gain"])
        self._max_control_accel_delta = float(self._tune["max_control_accel_delta_mps2"])
        self._max_control_yaw_rate_delta = float(self._tune["max_control_yaw_rate_delta_radps"])

        self._latest_state: Optional[EgoState] = None
        self._latest_gnss_local: Optional[LocalGnssMeasurement] = None
        self._latest_compass_yaw_rad: Optional[float] = None
        self._latest_compass_yaw_deg: Optional[float] = None
        self._latest_gyro_z_radps: Optional[float] = None
        self._latest_signed_gyro_z_radps: Optional[float] = None
        self._gyro_yaw_rate_clamped = False
        self._latest_raw_accelerometer: Optional[tuple[float, float, float]] = None
        self._latest_longitudinal_accel_mps2: Optional[float] = None
        self._longitudinal_accel_clamped = False
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
        self._latest_raw_accelerometer = None
        self._latest_longitudinal_accel_mps2 = None
        self._longitudinal_accel_clamped = False
        self._latest_control_input = None
        self._active_command_used_latest_prediction = False
        self._control_predicted_accel_mps2 = None
        self._control_predicted_yaw_rate_radps = None
        self._control_input_age_s = None
        self._control_prediction_reason = "reset"
        self._last_imu_frame = None
        self._last_gnss_frame = None

    def process_imu(self, imu: "ImuMeasurement") -> Optional[EgoState]:
        yaw_rad = yaw_rad_from_compass(float(imu.compass), self._compass_yaw_offset_deg)
        if yaw_rad is not None:
            self._latest_compass_yaw_rad = yaw_rad
            self._latest_compass_yaw_deg = normalize_angle_deg(math.degrees(yaw_rad))

        yaw_rate_radps = self._yaw_rate_measurement_from_imu(imu)
        acceleration_mps2 = self._longitudinal_accel_measurement_from_imu(imu)
        self._last_imu_frame = int(imu.frame)

        if not self._filter.initialized:
            return self._latest_state

        self._predict_to(float(imu.timestamp))
        if yaw_rad is not None:
            self._filter.update_yaw(yaw_rad)
        if yaw_rate_radps is not None:
            self._filter.update_yaw_rate(yaw_rate_radps)
        if acceleration_mps2 is not None:
            self._filter.update_acceleration(acceleration_mps2)
        return self._refresh_state_from_filter()

    def process_gnss(self, gnss: "GnssMeasurement") -> Optional[EgoState]:
        local = self._gnss_projector.project(gnss)
        if local is None:
            return self._latest_state

        self._latest_gnss_local = local
        self._last_gnss_frame = int(gnss.frame)

        if not self._filter.initialized:
            yaw_rad = self._latest_compass_yaw_rad if self._latest_compass_yaw_rad is not None else 0.0
            yaw_rate = self._latest_signed_gyro_z_radps if self._latest_signed_gyro_z_radps is not None else 0.0
            acceleration = self._latest_longitudinal_accel_mps2 if self._latest_longitudinal_accel_mps2 is not None else 0.0
            self._filter.initialize(
                position_xy=(local.x, local.y),
                yaw_rad=yaw_rad,
                speed_mps=0.0,
                acceleration_mps2=acceleration,
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

    def get_state(self) -> Optional[EgoState]:
        return self._latest_state

    def get_diagnostics(self) -> dict[str, object]:
        snapshot = self._filter.snapshot()
        covariance = self._filter.covariance
        state_vector = [float(value) for value in self._filter.state_vector.reshape(-1)]
        yaw_rate = snapshot.yaw_rate_radps if snapshot is not None else None
        speed = snapshot.speed if snapshot is not None else None
        acceleration = snapshot.acceleration_mps2 if snapshot is not None else None
        curvature = self._curvature(yaw_rate, speed)
        innovations = self._filter.innovations_by_type
        yaw_innovation = self._first_innovation(innovations.get("imu_yaw"))

        return {
            "filter_id": FILTER_INFO["id"],
            "model_type": "CTRA",
            "initialized": self.initialized,
            "safe_for_autonomous_control": False,
            "state_vector": state_vector,
            "covariance_diagonal": [float(value) for value in np.diag(covariance)],
            "yaw_deg": normalize_angle_deg(math.degrees(snapshot.yaw_rad)) if snapshot is not None else None,
            "speed_mps": speed,
            "acceleration_mps2": acceleration,
            "longitudinal_accel_mps2": acceleration,
            "yaw_rate_radps": yaw_rate,
            "yaw_rate_dps": math.degrees(yaw_rate) if yaw_rate is not None else None,
            "curvature_1pm": curvature,
            "imu_longitudinal_accel_axis": self._accel_axis_index,
            "imu_longitudinal_accel_axis_name": self._accel_axis_name,
            "imu_longitudinal_accel_sign": self._accel_sign,
            "imu_accel_bias_mps2": self._accel_bias,
            "latest_raw_accelerometer": self._latest_raw_accelerometer,
            "latest_longitudinal_accel_measurement_mps2": self._latest_longitudinal_accel_mps2,
            "longitudinal_accel_clamped": self._longitudinal_accel_clamped,
            "latest_compass_yaw_deg": self._latest_compass_yaw_deg,
            "latest_compass_yaw_rad": self._latest_compass_yaw_rad,
            "latest_gyro_z_radps": self._latest_gyro_z_radps,
            "latest_signed_gyro_z_radps": self._latest_signed_gyro_z_radps,
            "applied_gyro_z_sign": self._gyro_z_sign,
            "gyro_yaw_rate_clamped": self._gyro_yaw_rate_clamped,
            "yaw_innovation_deg": math.degrees(yaw_innovation) if yaw_innovation is not None else None,
            "yaw_rate_innovation_radps": self._first_innovation(innovations.get("imu_yaw_rate")),
            "accel_innovation_mps2": self._first_innovation(innovations.get("imu_longitudinal_accel")),
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
            "note": "CTRA EKF. Uses linear GNSS x/y, compass yaw, raw gyro z yaw-rate, and raw selected accelerometer axis. Autonomous-safe flag is false until validated in turns.",
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
        accel_target = throttle * max(0.0, self._control_accel_gain) - brake * max(0.0, self._control_brake_decel_gain)
        if self._latest_control_input.reverse:
            accel_target = -accel_target
        accel_delta = self._clamp(
            accel_target - snapshot.acceleration_mps2,
            -self._max_control_accel_delta,
            self._max_control_accel_delta,
        )

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

        if self._filter.apply_control_prediction(accel_delta, yaw_rate_delta):
            self._active_command_used_latest_prediction = True
            self._control_predicted_accel_mps2 = snapshot.acceleration_mps2 + accel_delta
            self._control_predicted_yaw_rate_radps = snapshot.yaw_rate_radps + yaw_rate_delta
            self._control_prediction_reason = "control prediction applied"
        else:
            self._control_prediction_reason = "control prediction rejected by filter core"

    def _refresh_state_from_filter(self) -> Optional[EgoState]:
        snapshot = self._filter.snapshot()
        if snapshot is None:
            self._latest_state = None
            return None

        z = self._latest_gnss_local.z if self._latest_gnss_local is not None else 0.0
        self._latest_state = EgoState(
            x=float(snapshot.px),
            y=float(snapshot.py),
            z=float(z),
            yaw=normalize_angle_deg(math.degrees(snapshot.yaw_rad)),
            speed=max(0.0, float(snapshot.speed)),
            timestamp=float(snapshot.timestamp),
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

    def _longitudinal_accel_measurement_from_imu(self, imu: "ImuMeasurement") -> Optional[float]:
        raw = getattr(imu, "raw_accelerometer", None)
        if raw is None:
            raw = imu.accelerometer
        try:
            raw_tuple = tuple(float(value) for value in raw)
        except (TypeError, ValueError):
            return None
        if len(raw_tuple) < 3 or not all(math.isfinite(value) for value in raw_tuple[:3]):
            return None

        self._latest_raw_accelerometer = (raw_tuple[0], raw_tuple[1], raw_tuple[2])
        measured = self._accel_sign * raw_tuple[self._accel_axis_index] - self._accel_bias
        self._longitudinal_accel_clamped = abs(measured) > self._max_abs_accel
        measured = self._clamp(measured, -self._max_abs_accel, self._max_abs_accel)
        self._latest_longitudinal_accel_mps2 = measured
        return measured

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
    def _axis_index(value: object) -> int:
        try:
            index = int(round(float(value)))
        except (TypeError, ValueError):
            return 0
        return max(0, min(2, index))

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
