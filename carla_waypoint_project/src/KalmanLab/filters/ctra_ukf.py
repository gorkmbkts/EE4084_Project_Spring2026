"""Constant Turn Rate and Acceleration UKF plugin for KalmanLab."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, TYPE_CHECKING

import numpy as np

from src.KalmanLab.filter_base import normalize_tracking_mode
from src.KalmanLab.filters import ctra_ekf as _ctra_shared
from src.core.vehicle_state import VehicleState
from src.evaluation.benchmark_config import ParameterSpec
from src.localization.gnss_projection import GnssLocalProjector, LocalGnssMeasurement

if TYPE_CHECKING:
    from src.sensors.gnss_sensor import GnssMeasurement
    from src.sensors.imu_sensor import ImuMeasurement


normalize_angle_rad = _ctra_shared.normalize_angle_rad
normalize_angle_deg = _ctra_shared.normalize_angle_deg
yaw_rad_from_compass = _ctra_shared.yaw_rad_from_compass


FILTER_INFO = {
    "id": "ctra_ukf",
    "name": "CTRA UKF",
    "type": "Unscented Kalman Filter",
    "state_vector": "[px, py, yaw, speed, acceleration, yaw_rate]^T",
    "process_model": "Nonlinear Constant Turn Rate and Acceleration sigma-point propagation",
    "measurement_model": "GNSS position x/y + optional IMU compass yaw + IMU gyro z yaw-rate + raw longitudinal acceleration",
    "description": "CTRA Unscented Kalman Filter using projected GNSS and optional raw IMU heading, yaw-rate, and longitudinal acceleration.",
    "model_type": "CTRA",
    "provided_state_fields": (
        "x",
        "y",
        "z",
        "yaw",
        "speed",
        "timestamp",
        "vx_mps",
        "vy_mps",
        "acceleration_mps2",
        "longitudinal_accel_mps2",
        "yaw_rate_radps",
        "curvature_1pm",
        "covariance_diagonal",
        "position_covariance_2x2",
        "raw_state_vector",
    ),
    "safe_for_autonomous_control": True,
    "active_tracking_supported": True,
    "benchmark_selectable": True,
    "experimental": True,
    "requires_raw_imu": True,
    "autonomous_control_note": "CTRA UKF is experimental; verify gyro_z_sign, compass_yaw_offset_deg, and acceleration axis/sign before relying on benchmark scores.",
}


TUNE = {
    "process_position_stddev_m": 0.20,
    "process_jerk_stddev_mps3": 2.5,
    "process_accel_stddev_mps2": 0.80,
    "process_yaw_rate_stddev_radps": 0.08,
    "process_yaw_accel_stddev_radps2": 0.90,
    "gnss_position_stddev_m": 1.75,
    "imu_yaw_stddev_deg": 6.0,
    "imu_yaw_rate_stddev_radps": 0.10,
    "imu_accel_stddev_mps2": 1.0,
    "gnss_R_multiplier": 1.0,
    "imu_yaw_R_multiplier": 1.0,
    "imu_yaw_rate_R_multiplier": 1.0,
    "imu_accel_R_multiplier": 1.0,
    "process_noise_multiplier": 1.0,
    "covariance_inflation": 1.02,
    "imu_accel_bias_mps2": 0.0,
    "initial_position_stddev_m": 5.0,
    "initial_yaw_stddev_deg": 30.0,
    "initial_speed_stddev_mps": 5.0,
    "initial_accel_stddev_mps2": 2.5,
    "initial_yaw_rate_stddev_radps": 0.6,
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
    "ukf_alpha": 0.35,
    "ukf_beta": 2.0,
    "ukf_kappa": 0.0,
    "min_covariance_diagonal": 1.0e-9,
    "enable_control_input_prediction": 1.0,
    "control_accel_gain_mps2": 4.5,
    "control_brake_decel_gain_mps2": 6.0,
    "control_coast_decel_mps2": 0.0,
    "control_steer_to_yaw_rate_gain": 0.25,
    "control_input_timeout_s": 0.35,
    "max_control_accel_delta_mps2": 0.10,
    "max_control_yaw_rate_delta_radps": 0.12,
}


TUNE_SPECS = (
    ParameterSpec("process_position_stddev_m", "Process position", 0.001, 3.0, "m", 3, "Noise"),
    ParameterSpec("process_jerk_stddev_mps3", "Process jerk", 0.05, 14.0, "m/s3", 2, "Noise"),
    ParameterSpec("process_accel_stddev_mps2", "Process accel", 0.01, 8.0, "m/s2", 2, "Noise"),
    ParameterSpec("process_yaw_rate_stddev_radps", "Process yaw rate", 0.001, 1.0, "rad/s", 3, "Noise"),
    ParameterSpec("process_yaw_accel_stddev_radps2", "Process yaw accel", 0.02, 5.0, "rad/s2", 2, "Noise"),
    ParameterSpec("gnss_position_stddev_m", "GNSS position", 0.10, 15.0, "m", 2, "Noise"),
    ParameterSpec("imu_yaw_stddev_deg", "IMU yaw", 0.5, 60.0, "deg", 1, "Noise"),
    ParameterSpec("imu_yaw_rate_stddev_radps", "IMU yaw rate", 0.005, 1.2, "rad/s", 3, "Noise"),
    ParameterSpec("imu_accel_stddev_mps2", "IMU accel", 0.02, 6.0, "m/s2", 2, "Noise"),
    ParameterSpec("gnss_R_multiplier", "Effective GNSS R", 0.10, 30.0, "x", 2, "Effective uncertainty"),
    ParameterSpec("imu_yaw_R_multiplier", "Effective yaw R", 0.10, 30.0, "x", 2, "Effective uncertainty"),
    ParameterSpec("imu_yaw_rate_R_multiplier", "Effective yaw-rate R", 0.10, 30.0, "x", 2, "Effective uncertainty"),
    ParameterSpec("imu_accel_R_multiplier", "Effective accel R", 0.10, 30.0, "x", 2, "Effective uncertainty"),
    ParameterSpec("process_noise_multiplier", "Process Q inflation", 0.10, 30.0, "x", 2, "Effective uncertainty"),
    ParameterSpec("covariance_inflation", "Covariance inflation", 1.0, 5.0, "x", 2, "Effective uncertainty"),
    ParameterSpec("imu_accel_bias_mps2", "IMU accel bias", -5.0, 5.0, "m/s2", 2, "IMU convention"),
    ParameterSpec("initial_position_stddev_m", "Initial pos", 0.25, 30.0, "m", 2, "Initialization"),
    ParameterSpec("initial_yaw_stddev_deg", "Initial yaw", 1.0, 120.0, "deg", 1, "Initialization"),
    ParameterSpec("initial_speed_stddev_mps", "Initial speed", 0.10, 25.0, "m/s", 2, "Initialization"),
    ParameterSpec("initial_accel_stddev_mps2", "Initial accel", 0.05, 10.0, "m/s2", 2, "Initialization"),
    ParameterSpec("initial_yaw_rate_stddev_radps", "Initial yaw rate", 0.01, 3.0, "rad/s", 2, "Initialization"),
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
    ParameterSpec("ukf_alpha", "UKF alpha", 0.001, 1.0, "", 3, "UKF"),
    ParameterSpec("ukf_beta", "UKF beta", 0.0, 4.0, "", 2, "UKF"),
    ParameterSpec("ukf_kappa", "UKF kappa", -5.0, 5.0, "", 2, "UKF"),
    ParameterSpec("min_covariance_diagonal", "Min covariance", 0.0, 0.001, "", 8, "UKF"),
    ParameterSpec("enable_control_input_prediction", "Use control input", 0.0, 1.0, "", 0, "Active tracking"),
    ParameterSpec("control_accel_gain_mps2", "Control accel", 0.0, 5.0, "m/s2", 2, "Active tracking"),
    ParameterSpec("control_brake_decel_gain_mps2", "Control brake", 0.0, 8.0, "m/s2", 2, "Active tracking"),
    ParameterSpec("control_coast_decel_mps2", "Control coast", 0.0, 3.0, "m/s2", 2, "Active tracking"),
    ParameterSpec("control_steer_to_yaw_rate_gain", "Steer yaw gain", 0.0, 1.0, "x", 2, "Active tracking"),
    ParameterSpec("control_input_timeout_s", "Control timeout", 0.02, 1.0, "s", 2, "Active tracking"),
    ParameterSpec("max_control_accel_delta_mps2", "Control accel delta", 0.0, 2.0, "m/s2", 2, "Active tracking"),
    ParameterSpec("max_control_yaw_rate_delta_radps", "Control yaw delta", 0.0, 1.0, "rad/s", 2, "Active tracking"),
)


AUTO_TUNE_PROFILE = {
    "enabled": True,
    "primary": [
        {"key": "process_position_stddev_m", "scale": "log", "min": 0.01, "max": 1.5},
        {"key": "process_jerk_stddev_mps3", "scale": "log", "min": 0.20, "max": 10.0},
        {"key": "process_accel_stddev_mps2", "scale": "log", "min": 0.05, "max": 4.0},
        {"key": "process_yaw_rate_stddev_radps", "scale": "log", "min": 0.005, "max": 0.5},
        {"key": "process_yaw_accel_stddev_radps2", "scale": "log", "min": 0.05, "max": 3.0},
        {"key": "gnss_position_stddev_m", "scale": "log", "min": 0.40, "max": 8.0},
        {"key": "imu_yaw_rate_stddev_radps", "scale": "log", "min": 0.01, "max": 0.6},
        {"key": "gnss_R_multiplier", "scale": "log", "min": 0.25, "max": 20.0},
        {"key": "imu_yaw_rate_R_multiplier", "scale": "log", "min": 0.25, "max": 20.0},
        {"key": "imu_accel_R_multiplier", "scale": "log", "min": 0.25, "max": 20.0},
        {"key": "process_noise_multiplier", "scale": "log", "min": 0.25, "max": 16.0},
    ],
    "secondary": [
        {"key": "imu_accel_stddev_mps2", "scale": "log", "min": 0.05, "max": 4.0},
        {"key": "imu_yaw_stddev_deg", "scale": "log", "min": 1.0, "max": 25.0},
        {"key": "imu_yaw_R_multiplier", "scale": "log", "min": 0.25, "max": 20.0},
        {"key": "covariance_inflation", "scale": "log", "min": 1.0, "max": 3.5},
        {"key": "ukf_alpha", "scale": "linear", "min": 0.10, "max": 0.90},
    ],
    "search": {
        "default_trials": 36,
        "strategy": "random_plus_coordinate_refinement",
    },
    "objective": "rmse_consistency",
}


def process_model(
    state_vector: np.ndarray,
    dt: float,
    turn_rate_epsilon_radps: float = 1.0e-4,
) -> np.ndarray:
    """Closed-form CTRA model for [px, py, yaw, speed, acceleration, yaw_rate]."""
    state = np.asarray(state_vector, dtype=float).reshape(-1).astype(float)
    dt = max(0.0, float(dt))
    if dt <= 0.0:
        state[2] = normalize_angle_rad(float(state[2]))
        return state

    px, py, yaw, speed, acceleration, yaw_rate = (float(value) for value in state[:6])
    yaw = normalize_angle_rad(yaw)
    epsilon = max(1.0e-12, float(turn_rate_epsilon_radps))
    if abs(yaw_rate) <= epsilon:
        distance = speed * dt + 0.5 * acceleration * dt * dt
        px += distance * math.cos(yaw)
        py += distance * math.sin(yaw)
        speed += acceleration * dt
        return np.array([px, py, yaw, speed, acceleration, yaw_rate], dtype=float)

    next_yaw = yaw + yaw_rate * dt
    sin_yaw = math.sin(yaw)
    cos_yaw = math.cos(yaw)
    sin_next = math.sin(next_yaw)
    cos_next = math.cos(next_yaw)
    omega = yaw_rate
    omega2 = omega * omega

    px += speed / omega * (sin_next - sin_yaw)
    px += acceleration * (dt * sin_next / omega + (cos_next - cos_yaw) / omega2)
    py += speed / omega * (-cos_next + cos_yaw)
    py += acceleration * (-dt * cos_next / omega + (sin_next - sin_yaw) / omega2)
    speed += acceleration * dt

    return np.array([px, py, normalize_angle_rad(next_yaw), speed, acceleration, yaw_rate], dtype=float)


@dataclass(frozen=True)
class _CtraSnapshot:
    px: float
    py: float
    yaw_rad: float
    speed: float
    acceleration_mps2: float
    yaw_rate_radps: float
    timestamp: float


class _CtraUkfCore:
    """Scaled UKF core for [px, py, yaw, speed, acceleration, yaw_rate]^T."""

    _STATE_DIM = 6
    _YAW_INDEX = 2

    def __init__(self, tune: dict[str, float]) -> None:
        process_multiplier = max(1.0e-6, float(tune.get("process_noise_multiplier", 1.0)))
        self._position_process_var = float(tune["process_position_stddev_m"]) ** 2 * process_multiplier
        self._jerk_var = float(tune["process_jerk_stddev_mps3"]) ** 2 * process_multiplier
        self._accel_process_var = float(tune["process_accel_stddev_mps2"]) ** 2 * process_multiplier
        self._yaw_rate_process_var = float(tune["process_yaw_rate_stddev_radps"]) ** 2 * process_multiplier
        self._yaw_accel_var = float(tune["process_yaw_accel_stddev_radps2"]) ** 2 * process_multiplier
        self._position_var = float(tune["gnss_position_stddev_m"]) ** 2 * max(1.0e-6, float(tune.get("gnss_R_multiplier", 1.0)))
        self._yaw_var = math.radians(float(tune["imu_yaw_stddev_deg"])) ** 2 * max(1.0e-6, float(tune.get("imu_yaw_R_multiplier", 1.0)))
        self._yaw_rate_var = float(tune["imu_yaw_rate_stddev_radps"]) ** 2 * max(1.0e-6, float(tune.get("imu_yaw_rate_R_multiplier", 1.0)))
        self._accel_var = float(tune["imu_accel_stddev_mps2"]) ** 2 * max(1.0e-6, float(tune.get("imu_accel_R_multiplier", 1.0)))
        self._covariance_inflation = max(1.0, float(tune.get("covariance_inflation", 1.0)))
        self._initial_position_var = float(tune["initial_position_stddev_m"]) ** 2
        self._initial_yaw_var = math.radians(float(tune["initial_yaw_stddev_deg"])) ** 2
        self._initial_speed_var = float(tune["initial_speed_stddev_mps"]) ** 2
        self._initial_accel_var = float(tune["initial_accel_stddev_mps2"]) ** 2
        self._initial_yaw_rate_var = float(tune["initial_yaw_rate_stddev_radps"]) ** 2
        self._turn_rate_epsilon = float(tune["turn_rate_epsilon_radps"])
        self._max_abs_yaw_rate = max(0.01, float(tune["max_abs_yaw_rate_radps"]))
        self._max_abs_accel = max(0.1, float(tune["max_abs_accel_mps2"]))
        self._max_speed = max(0.1, float(tune["max_speed_mps"]))
        self._min_covariance_diagonal = max(0.0, float(tune.get("min_covariance_diagonal", 1.0e-9)))

        self._alpha = max(1.0e-3, min(1.0, float(tune.get("ukf_alpha", 0.35))))
        self._beta = max(0.0, min(4.0, float(tune.get("ukf_beta", 2.0))))
        self._kappa = float(tune.get("ukf_kappa", 0.0))
        if self._STATE_DIM + self._kappa <= 1.0e-6:
            self._kappa = 0.0
        self._lambda = self._alpha * self._alpha * (self._STATE_DIM + self._kappa) - self._STATE_DIM
        if self._STATE_DIM + self._lambda <= 1.0e-9:
            self._alpha = 1.0
            self._kappa = 0.0
            self._lambda = 0.0
        self._sigma_scale = self._STATE_DIM + self._lambda
        self._weights_mean, self._weights_cov = self._weights()

        self._x = np.zeros((self._STATE_DIM, 1), dtype=float)
        self._p = np.eye(self._STATE_DIM, dtype=float)
        self._timestamp: Optional[float] = None

        self.initialized = False
        self.last_update_type: Optional[str] = None
        self.last_innovation: Optional[list[float]] = None
        self.last_nis: Optional[float] = None
        self.last_runtime_warning: Optional[str] = None
        self.latest_predicted_state: Optional[list[float]] = None
        self.innovations_by_type: dict[str, list[float]] = {}
        self.nis_by_type: dict[str, float] = {}
        self.nis_update_counts_by_type: dict[str, int] = {}

    @property
    def timestamp(self) -> Optional[float]:
        return self._timestamp

    @property
    def state_vector(self) -> np.ndarray:
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._p.copy()

    @property
    def ukf_parameters(self) -> dict[str, float]:
        return {
            "alpha": float(self._alpha),
            "beta": float(self._beta),
            "kappa": float(self._kappa),
            "lambda": float(self._lambda),
            "sigma_point_count": float(2 * self._STATE_DIM + 1),
        }

    def reset(self) -> None:
        self._x = np.zeros((self._STATE_DIM, 1), dtype=float)
        self._p = np.eye(self._STATE_DIM, dtype=float)
        self._timestamp = None
        self.initialized = False
        self.last_update_type = None
        self.last_innovation = None
        self.last_nis = None
        self.last_runtime_warning = None
        self.latest_predicted_state = None
        self.innovations_by_type.clear()
        self.nis_by_type.clear()
        self.nis_update_counts_by_type.clear()

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

        try:
            sigma = self._sigma_points()
        except ValueError as exc:
            self.last_runtime_warning = f"predict skipped: sigma point generation failed: {exc}"
            return

        predicted_sigma = np.array(
            [
                self._sanitize_state_vector(process_model(point, dt, self._turn_rate_epsilon))
                for point in sigma
            ],
            dtype=float,
        )
        q = self._process_noise(float(self._x[self._YAW_INDEX, 0]), dt)
        predicted, covariance = self._recover_mean_and_covariance(predicted_sigma, q)
        if self._covariance_inflation > 1.0:
            covariance *= self._covariance_inflation

        if not self._set_state_and_covariance(predicted.reshape(self._STATE_DIM, 1), covariance, "predict"):
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

        z = np.array([float(position_xy[0]), float(position_xy[1])], dtype=float)
        r = np.eye(2, dtype=float) * self._position_var
        self._unscented_update(z, lambda state: np.array([state[0], state[1]], dtype=float), r, "gnss_position")

    def update_yaw(self, yaw_rad: float) -> None:
        if not self.initialized:
            return
        if not math.isfinite(float(yaw_rad)):
            self._skip_update("imu_yaw", "non-finite IMU yaw")
            return

        z = np.array([normalize_angle_rad(float(yaw_rad))], dtype=float)
        r = np.array([[self._yaw_var]], dtype=float)
        self._unscented_update(z, lambda state: np.array([state[2]], dtype=float), r, "imu_yaw", angle_measurement_indices=(0,))

    def update_yaw_rate(self, yaw_rate_radps: float) -> None:
        if not self.initialized:
            return
        if not math.isfinite(float(yaw_rate_radps)):
            self._skip_update("imu_yaw_rate", "non-finite IMU yaw-rate")
            return

        clamped = self._clamp(float(yaw_rate_radps), -self._max_abs_yaw_rate, self._max_abs_yaw_rate)
        z = np.array([clamped], dtype=float)
        r = np.array([[self._yaw_rate_var]], dtype=float)
        self._unscented_update(z, lambda state: np.array([state[5]], dtype=float), r, "imu_yaw_rate")

    def update_acceleration(self, acceleration_mps2: float) -> None:
        if not self.initialized:
            return
        if not math.isfinite(float(acceleration_mps2)):
            self._skip_update("imu_longitudinal_accel", "non-finite IMU acceleration")
            return

        clamped = self._clamp(float(acceleration_mps2), -self._max_abs_accel, self._max_abs_accel)
        z = np.array([clamped], dtype=float)
        r = np.array([[self._accel_var]], dtype=float)
        self._unscented_update(z, lambda state: np.array([state[4]], dtype=float), r, "imu_longitudinal_accel")

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

    def _unscented_update(
        self,
        z: np.ndarray,
        measurement_function: object,
        r: np.ndarray,
        update_type: str,
        angle_measurement_indices: tuple[int, ...] = (),
    ) -> None:
        try:
            sigma = self._sigma_points()
            z_sigma = np.array([np.asarray(measurement_function(point), dtype=float).reshape(-1) for point in sigma], dtype=float)
        except (TypeError, ValueError) as exc:
            self._skip_update(update_type, f"sigma measurement failed: {exc}")
            return

        z = np.asarray(z, dtype=float).reshape(-1)
        if z_sigma.ndim != 2 or z_sigma.shape[1] != z.size:
            self._skip_update(update_type, "measurement dimension mismatch")
            return

        z_pred = self._weighted_measurement_mean(z_sigma, angle_measurement_indices)
        innovation = z - z_pred
        for index in angle_measurement_indices:
            innovation[index] = normalize_angle_rad(float(innovation[index]))

        state_mean = self._x.reshape(-1)
        innovation_cov = np.asarray(r, dtype=float).reshape(z.size, z.size).copy()
        cross_cov = np.zeros((self._STATE_DIM, z.size), dtype=float)
        for point, measurement, weight in zip(sigma, z_sigma, self._weights_cov):
            state_residual = np.asarray(point, dtype=float).reshape(-1) - state_mean
            state_residual[self._YAW_INDEX] = normalize_angle_rad(float(state_residual[self._YAW_INDEX]))
            measurement_residual = measurement - z_pred
            for index in angle_measurement_indices:
                measurement_residual[index] = normalize_angle_rad(float(measurement_residual[index]))
            innovation_cov += float(weight) * np.outer(measurement_residual, measurement_residual)
            cross_cov += float(weight) * np.outer(state_residual, measurement_residual)

        innovation_cov = self._symmetrize(innovation_cov)
        try:
            solved_innovation = np.linalg.solve(innovation_cov, innovation.reshape(-1, 1))
            gain = np.linalg.solve(innovation_cov.T, cross_cov.T).T
        except (np.linalg.LinAlgError, ValueError) as exc:
            self._skip_update(update_type, f"innovation solve failed: {exc}")
            return

        next_x = self._x + gain @ innovation.reshape(-1, 1)
        next_p = self._p - gain @ innovation_cov @ gain.T

        if not self._set_state_and_covariance(next_x, next_p, update_type):
            return

        innovation_list = [float(value) for value in innovation.reshape(-1)]
        nis = float((innovation.reshape(1, -1) @ solved_innovation)[0, 0])
        self.last_update_type = update_type
        self.last_innovation = innovation_list
        self.last_nis = nis
        self.last_runtime_warning = None
        self.innovations_by_type[update_type] = innovation_list
        self.nis_by_type[update_type] = nis
        self.nis_update_counts_by_type[update_type] = self.nis_update_counts_by_type.get(update_type, 0) + 1

    def _sigma_points(self) -> np.ndarray:
        covariance = self._symmetrize(self._p)
        diagonal = np.diag(covariance).copy()
        for index, value in enumerate(diagonal):
            if value < self._min_covariance_diagonal:
                covariance[index, index] = self._min_covariance_diagonal
        scaled = self._sigma_scale * covariance
        jitter = max(self._min_covariance_diagonal, 1.0e-10)
        for _ in range(6):
            try:
                root = np.linalg.cholesky(self._symmetrize(scaled))
                break
            except np.linalg.LinAlgError:
                scaled = scaled + np.eye(self._STATE_DIM, dtype=float) * jitter
                jitter *= 10.0
        else:
            eigvals, eigvecs = np.linalg.eigh(self._symmetrize(scaled))
            if not np.all(np.isfinite(eigvals)):
                raise ValueError("non-finite covariance eigenvalues")
            eigvals = np.maximum(eigvals, max(self._min_covariance_diagonal, 1.0e-9))
            root = eigvecs @ np.diag(np.sqrt(eigvals))

        mean = self._x.reshape(-1)
        sigma = [mean]
        for index in range(self._STATE_DIM):
            delta = root[:, index]
            sigma.append(self._sanitize_state_vector(mean + delta))
            sigma.append(self._sanitize_state_vector(mean - delta))
        return np.array(sigma, dtype=float)

    def _recover_mean_and_covariance(self, sigma: np.ndarray, additive_covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(self._STATE_DIM, dtype=float)
        for index in range(self._STATE_DIM):
            if index == self._YAW_INDEX:
                mean[index] = self._weighted_angle_mean(sigma[:, index])
            else:
                mean[index] = float(np.dot(self._weights_mean, sigma[:, index]))
        mean = self._sanitize_state_vector(mean)

        covariance = np.asarray(additive_covariance, dtype=float).reshape(self._STATE_DIM, self._STATE_DIM).copy()
        for point, weight in zip(sigma, self._weights_cov):
            residual = np.asarray(point, dtype=float).reshape(-1) - mean
            residual[self._YAW_INDEX] = normalize_angle_rad(float(residual[self._YAW_INDEX]))
            covariance += float(weight) * np.outer(residual, residual)
        return mean, self._symmetrize(covariance)

    def _weighted_measurement_mean(self, z_sigma: np.ndarray, angle_indices: tuple[int, ...]) -> np.ndarray:
        mean = np.zeros(z_sigma.shape[1], dtype=float)
        angle_set = set(angle_indices)
        for index in range(z_sigma.shape[1]):
            if index in angle_set:
                mean[index] = self._weighted_angle_mean(z_sigma[:, index])
            else:
                mean[index] = float(np.dot(self._weights_mean, z_sigma[:, index]))
        return mean

    def _weighted_angle_mean(self, values: np.ndarray) -> float:
        sin_sum = float(np.dot(self._weights_mean, np.sin(values)))
        cos_sum = float(np.dot(self._weights_mean, np.cos(values)))
        return normalize_angle_rad(math.atan2(sin_sum, cos_sum))

    def _process_noise(self, yaw: float, dt: float) -> np.ndarray:
        dt = max(0.0, float(dt))
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
        q[0, 0] += self._position_process_var * dt + (1.0 / 36.0) * self._jerk_var * dt**6
        q[1, 1] += self._position_process_var * dt + (1.0 / 36.0) * self._jerk_var * dt**6
        q[2, 2] += self._yaw_rate_process_var * dt * dt
        q[4, 4] += self._accel_process_var * dt
        q[5, 5] += self._yaw_rate_process_var * dt
        q += np.eye(self._STATE_DIM, dtype=float) * max(self._min_covariance_diagonal, 1.0e-12)
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

        covariance = covariance.copy()
        for index, value in enumerate(np.diag(covariance)):
            if value < self._min_covariance_diagonal:
                covariance[index, index] = self._min_covariance_diagonal

        self._x = state
        self._p = self._symmetrize(covariance)
        return True

    def _sanitize_state(self, state: np.ndarray) -> np.ndarray:
        return self._sanitize_state_vector(state).reshape(self._STATE_DIM, 1)

    def _sanitize_state_vector(self, state: np.ndarray) -> np.ndarray:
        result = np.asarray(state, dtype=float).reshape(self._STATE_DIM).copy()
        result[2] = normalize_angle_rad(float(result[2]))
        result[3] = self._clamp(float(result[3]), 0.0, self._max_speed)
        result[4] = self._clamp(float(result[4]), -self._max_abs_accel, self._max_abs_accel)
        result[5] = self._clamp(float(result[5]), -self._max_abs_yaw_rate, self._max_abs_yaw_rate)
        return result

    def _skip_update(self, update_type: str, reason: str) -> None:
        self.last_update_type = f"{update_type}_skipped"
        self.last_innovation = None
        self.last_nis = None
        self.last_runtime_warning = reason

    def _weights(self) -> tuple[np.ndarray, np.ndarray]:
        count = 2 * self._STATE_DIM + 1
        mean = np.full(count, 0.5 / self._sigma_scale, dtype=float)
        cov = np.full(count, 0.5 / self._sigma_scale, dtype=float)
        mean[0] = self._lambda / self._sigma_scale
        cov[0] = mean[0] + (1.0 - self._alpha * self._alpha + self._beta)
        return mean, cov

    @staticmethod
    def _symmetrize(matrix: np.ndarray) -> np.ndarray:
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))


class Filter(_ctra_shared.Filter):
    """CTRA UKF plugin with the same sensor/control surface as the CTRA EKF."""

    def __init__(
        self,
        gnss_projector: GnssLocalProjector,
        tune: Optional[dict[str, object]] = None,
        tracking_mode: str = "passive",
    ) -> None:
        super().__init__(gnss_projector, tune=tune, tracking_mode=tracking_mode)
        self._tune = dict(TUNE)
        if tune:
            self._tune.update(dict(tune))
        self._tracking_mode = normalize_tracking_mode(tracking_mode)
        self._filter = _CtraUkfCore(self._tune)
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
        self._control_coast_decel = float(self._tune["control_coast_decel_mps2"])
        self._control_steer_yaw_gain = float(self._tune["control_steer_to_yaw_rate_gain"])
        self._max_control_accel_delta = float(self._tune["max_control_accel_delta_mps2"])
        self._max_control_yaw_rate_delta = float(self._tune["max_control_yaw_rate_delta_radps"])

    def get_diagnostics(self) -> dict[str, object]:
        snapshot = self._filter.snapshot()
        covariance = self._filter.covariance
        position_covariance_2x2 = [
            [float(covariance[0, 0]), float(covariance[0, 1])],
            [float(covariance[1, 0]), float(covariance[1, 1])],
        ]
        state_vector = [float(value) for value in self._filter.state_vector.reshape(-1)]
        yaw_rate = snapshot.yaw_rate_radps if snapshot is not None else None
        speed = snapshot.speed if snapshot is not None else None
        acceleration = snapshot.acceleration_mps2 if snapshot is not None else None
        curvature = self._curvature(yaw_rate, speed)
        innovations = self._filter.innovations_by_type
        yaw_innovation = self._first_innovation(innovations.get("imu_yaw"))

        return {
            "filter_id": FILTER_INFO["id"],
            "model_type": FILTER_INFO["model_type"],
            "filter_family": "UKF",
            "initialized": self.initialized,
            "safe_for_autonomous_control": bool(FILTER_INFO["safe_for_autonomous_control"]),
            "state_vector": state_vector,
            "covariance_diagonal": [float(value) for value in np.diag(covariance)],
            "position_covariance_2x2": position_covariance_2x2,
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
            "nis_update_counts_by_type": dict(self._filter.nis_update_counts_by_type),
            "nis_expected_dimensions_by_type": {
                "gnss_position": 2,
                "imu_yaw": 1,
                "imu_yaw_rate": 1,
                "imu_longitudinal_accel": 1,
            },
            "effective_uncertainty_multipliers": {
                "gnss_R_multiplier": float(self._tune.get("gnss_R_multiplier", 1.0)),
                "imu_yaw_R_multiplier": float(self._tune.get("imu_yaw_R_multiplier", 1.0)),
                "imu_yaw_rate_R_multiplier": float(self._tune.get("imu_yaw_rate_R_multiplier", 1.0)),
                "imu_accel_R_multiplier": float(self._tune.get("imu_accel_R_multiplier", 1.0)),
                "process_noise_multiplier": float(self._tune.get("process_noise_multiplier", 1.0)),
                "covariance_inflation": float(self._tune.get("covariance_inflation", 1.0)),
            },
            "ukf_parameters": self._filter.ukf_parameters,
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
            "note": "CTRA UKF. Uses nonlinear sigma-point prediction plus GNSS x/y, compass yaw, raw gyro z yaw-rate, and raw selected accelerometer axis.",
        }

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
        position_covariance_2x2 = (
            (float(covariance[0, 0]), float(covariance[0, 1])),
            (float(covariance[1, 0]), float(covariance[1, 1])),
        )
        self._latest_state = VehicleState(
            x=float(snapshot.px),
            y=float(snapshot.py),
            z=float(z),
            yaw=yaw_deg,
            speed=speed,
            timestamp=float(snapshot.timestamp),
            vx_mps=speed * math.cos(snapshot.yaw_rad),
            vy_mps=speed * math.sin(snapshot.yaw_rad),
            acceleration_mps2=float(snapshot.acceleration_mps2),
            longitudinal_accel_mps2=float(snapshot.acceleration_mps2),
            yaw_rate_radps=float(snapshot.yaw_rate_radps),
            curvature_1pm=curvature,
            covariance_diagonal=tuple(float(value) for value in np.diag(covariance)),
            position_covariance_2x2=position_covariance_2x2,
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
