"""Constant-acceleration linear Kalman filter plugin for KalmanLab."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, TYPE_CHECKING

import numpy as np

from src.KalmanLab.control_model import estimate_command_motion
from src.KalmanLab.filter_base import FilterControlInput, TRACKING_MODE_ACTIVE, normalize_tracking_mode
from src.evaluation.benchmark_config import ParameterSpec
from src.localization.gnss_projection import GnssLocalProjector, LocalGnssMeasurement
from src.localization.state_estimator import EgoState

if TYPE_CHECKING:
    from src.sensors.gnss_sensor import GnssMeasurement
    from src.sensors.imu_sensor import ImuMeasurement


FILTER_INFO = {
    "id": "ca_kf",
    "name": "CA-KF",
    "type": "Linear Kalman Filter",
    "state_vector": "[px, py, vx, vy, ax, ay]^T",
    "process_model": "Constant Acceleration",
    "measurement_model": "GNSS position x/y + IMU acceleration x/y",
    "description": "Linear constant-acceleration Kalman filter using noisy GNSS and IMU acceleration.",
    "safe_for_autonomous_control": True,
    "active_tracking_supported": True,
}


TUNE = {
    "process_jerk_stddev_mps3": 1.2,
    "gnss_position_stddev_m": 1.25,
    "imu_accel_stddev_mps2": 0.45,
    "initial_position_stddev_m": 4.0,
    "initial_velocity_stddev_mps": 3.0,
    "initial_accel_stddev_mps2": 1.5,
    "yaw_from_velocity_min_speed_mps": 0.35,
    "min_prediction_dt_s": 1.0e-4,
    "max_prediction_dt_s": 0.20,
    "command_accel_stddev_mps2": 1.25,
    "command_throttle_accel_gain_mps2": 3.0,
    "command_brake_decel_gain_mps2": 6.0,
    "command_max_accel_mps2": 8.0,
    "command_yaw_rate_stddev_dps": 18.0,
    "command_max_yaw_rate_dps": 90.0,
}


TUNE_SPECS = (
    ParameterSpec("process_jerk_stddev_mps3", "Process jerk", 0.05, 8.0, "m/s3", 2, "Noise"),
    ParameterSpec("gnss_position_stddev_m", "GNSS position", 0.10, 12.0, "m", 2, "Noise"),
    ParameterSpec("imu_accel_stddev_mps2", "IMU accel", 0.02, 5.0, "m/s2", 2, "Noise"),
    ParameterSpec("initial_position_stddev_m", "Initial pos", 0.25, 25.0, "m", 2, "Initialization"),
    ParameterSpec("initial_velocity_stddev_mps", "Initial speed", 0.10, 15.0, "m/s", 2, "Initialization"),
    ParameterSpec("initial_accel_stddev_mps2", "Initial accel", 0.05, 8.0, "m/s2", 2, "Initialization"),
    ParameterSpec("yaw_from_velocity_min_speed_mps", "Yaw min speed", 0.05, 3.0, "m/s", 2, "Yaw"),
    ParameterSpec("min_prediction_dt_s", "Min dt", 0.00001, 0.02, "s", 5, "Prediction"),
    ParameterSpec("max_prediction_dt_s", "Max dt", 0.02, 0.60, "s", 2, "Prediction"),
    ParameterSpec("command_accel_stddev_mps2", "Command accel", 0.10, 6.0, "m/s2", 2, "Active tracking"),
    ParameterSpec("command_throttle_accel_gain_mps2", "Throttle accel", 0.2, 8.0, "m/s2", 2, "Active tracking"),
    ParameterSpec("command_brake_decel_gain_mps2", "Brake decel", 0.5, 12.0, "m/s2", 2, "Active tracking"),
    ParameterSpec("command_max_accel_mps2", "Cmd accel cap", 0.5, 15.0, "m/s2", 2, "Active tracking"),
    ParameterSpec("command_yaw_rate_stddev_dps", "Yaw-rate trust", 1.0, 90.0, "deg/s", 1, "Active tracking"),
    ParameterSpec("command_max_yaw_rate_dps", "Yaw-rate cap", 10.0, 180.0, "deg/s", 1, "Active tracking"),
)


@dataclass(frozen=True)
class _CAStateSnapshot:
    px: float
    py: float
    vx: float
    vy: float
    ax: float
    ay: float
    timestamp: float


class _CAFilterCore:
    """Linear 2D constant-acceleration Kalman filter."""

    def __init__(self, tune: dict[str, float]) -> None:
        self._process_jerk_var = float(tune["process_jerk_stddev_mps3"]) ** 2
        self._position_var = float(tune["gnss_position_stddev_m"]) ** 2
        self._accel_var = float(tune["imu_accel_stddev_mps2"]) ** 2
        self._initial_position_var = float(tune["initial_position_stddev_m"]) ** 2
        self._initial_velocity_var = float(tune["initial_velocity_stddev_mps"]) ** 2
        self._initial_accel_var = float(tune["initial_accel_stddev_mps2"]) ** 2

        self._x = np.zeros((6, 1), dtype=float)
        self._p = np.eye(6, dtype=float)
        self._timestamp: Optional[float] = None
        self.initialized = False
        self.last_update_type: Optional[str] = None
        self.last_innovation: Optional[list[float]] = None
        self.last_nis: Optional[float] = None

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

    def initialize(
        self,
        position_xy: tuple[float, float],
        timestamp: float,
        velocity_xy: tuple[float, float] = (0.0, 0.0),
        acceleration_xy: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self._x = np.array(
            [
                [float(position_xy[0])],
                [float(position_xy[1])],
                [float(velocity_xy[0])],
                [float(velocity_xy[1])],
                [float(acceleration_xy[0])],
                [float(acceleration_xy[1])],
            ],
            dtype=float,
        )
        self._p = np.diag(
            [
                self._initial_position_var,
                self._initial_position_var,
                self._initial_velocity_var,
                self._initial_velocity_var,
                self._initial_accel_var,
                self._initial_accel_var,
            ]
        ).astype(float)
        self._timestamp = float(timestamp)
        self.initialized = True

    def predict(self, dt: float, timestamp: Optional[float] = None) -> None:
        if not self.initialized:
            return

        dt = max(0.0, float(dt))
        if dt <= 0.0:
            if timestamp is not None:
                self._timestamp = float(timestamp)
            return

        f = self._transition_matrix(dt)
        q = self._process_noise(dt)
        self._x = f @ self._x
        self._p = f @ self._p @ f.T + q
        self._p = self._symmetrize(self._p)
        if timestamp is not None:
            self._timestamp = float(timestamp)
        elif self._timestamp is not None:
            self._timestamp += dt

    def update_position(self, position_xy: tuple[float, float]) -> None:
        if not self.initialized:
            return

        z = np.array([[float(position_xy[0])], [float(position_xy[1])]], dtype=float)
        h = np.zeros((2, 6), dtype=float)
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        r = np.eye(2, dtype=float) * self._position_var
        self._update(z, h, r, "gnss_position")

    def update_acceleration(
        self,
        acceleration_xy: tuple[float, float],
        stddev_mps2: Optional[float] = None,
        update_type: str = "imu_acceleration",
    ) -> None:
        if not self.initialized:
            return

        z = np.array([[float(acceleration_xy[0])], [float(acceleration_xy[1])]], dtype=float)
        h = np.zeros((2, 6), dtype=float)
        h[0, 4] = 1.0
        h[1, 5] = 1.0
        accel_var = self._accel_var if stddev_mps2 is None else max(1.0e-8, float(stddev_mps2) ** 2)
        r = np.eye(2, dtype=float) * accel_var
        self._update(z, h, r, update_type)

    def snapshot(self) -> Optional[_CAStateSnapshot]:
        if not self.initialized or self._timestamp is None:
            return None
        return _CAStateSnapshot(
            px=float(self._x[0, 0]),
            py=float(self._x[1, 0]),
            vx=float(self._x[2, 0]),
            vy=float(self._x[3, 0]),
            ax=float(self._x[4, 0]),
            ay=float(self._x[5, 0]),
            timestamp=float(self._timestamp),
        )

    def _update(self, z: np.ndarray, h: np.ndarray, r: np.ndarray, update_type: str) -> None:
        innovation = z - h @ self._x
        innovation_cov = h @ self._p @ h.T + r
        innovation_cov_inv = np.linalg.inv(innovation_cov)
        gain = self._p @ h.T @ innovation_cov_inv
        identity = np.eye(self._p.shape[0], dtype=float)

        self._x = self._x + gain @ innovation
        residual_transform = identity - gain @ h
        self._p = residual_transform @ self._p @ residual_transform.T + gain @ r @ gain.T
        self._p = self._symmetrize(self._p)

        self.last_update_type = update_type
        self.last_innovation = [float(value) for value in innovation.reshape(-1)]
        self.last_nis = float((innovation.T @ innovation_cov_inv @ innovation)[0, 0])

    @staticmethod
    def _transition_matrix(dt: float) -> np.ndarray:
        half_dt2 = 0.5 * dt * dt
        return np.array(
            [
                [1.0, 0.0, dt, 0.0, half_dt2, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, half_dt2],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _process_noise(self, dt: float) -> np.ndarray:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        single_axis = self._process_jerk_var * np.array(
            [
                [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
                [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
                [dt3 / 6.0, dt2 / 2.0, dt],
            ],
            dtype=float,
        )

        q = np.zeros((6, 6), dtype=float)
        x_indices = [0, 2, 4]
        y_indices = [1, 3, 5]
        for row in range(3):
            for col in range(3):
                q[x_indices[row], x_indices[col]] = single_axis[row, col]
                q[y_indices[row], y_indices[col]] = single_axis[row, col]
        return q

    @staticmethod
    def _symmetrize(matrix: np.ndarray) -> np.ndarray:
        return 0.5 * (matrix + matrix.T)


class Filter:
    """Self-contained CA-KF plugin with IMU/GNSS preprocessing."""

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
        self._filter = _CAFilterCore(self._tune)
        self._yaw_speed_threshold = float(self._tune["yaw_from_velocity_min_speed_mps"])
        self._min_prediction_dt_s = float(self._tune["min_prediction_dt_s"])
        self._max_prediction_dt_s = float(self._tune["max_prediction_dt_s"])
        self._command_accel_stddev = float(self._tune["command_accel_stddev_mps2"])

        self._latest_state: Optional[EgoState] = None
        self._latest_gnss_local: Optional[LocalGnssMeasurement] = None
        self._pending_acceleration_xy: Optional[tuple[float, float]] = None
        self._latest_imu_yaw_deg: Optional[float] = None
        self._latest_control_input: Optional[FilterControlInput] = None
        self._latest_command_acceleration_xy: Optional[tuple[float, float]] = None
        self._latest_command_yaw_rate_dps: Optional[float] = None
        self._latest_command_longitudinal_accel_mps2: Optional[float] = None
        self._active_command_used_latest_prediction = False
        self._last_valid_yaw_deg = 0.0
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
        self._pending_acceleration_xy = None
        self._latest_imu_yaw_deg = None
        self._latest_control_input = None
        self._latest_command_acceleration_xy = None
        self._latest_command_yaw_rate_dps = None
        self._latest_command_longitudinal_accel_mps2 = None
        self._active_command_used_latest_prediction = False
        self._last_valid_yaw_deg = 0.0
        self._last_imu_frame = None
        self._last_gnss_frame = None

    def process_control(self, control_input: FilterControlInput) -> bool:
        self._latest_control_input = control_input
        return self._tracking_mode == TRACKING_MODE_ACTIVE

    def process_imu(self, imu: "ImuMeasurement") -> Optional[EgoState]:
        yaw_deg = self._yaw_deg_from_compass(imu.compass)
        if yaw_deg is not None:
            self._latest_imu_yaw_deg = yaw_deg

        acceleration_xy = self._imu_acceleration_to_world_xy(imu, yaw_deg)
        self._pending_acceleration_xy = acceleration_xy
        self._last_imu_frame = int(imu.frame)

        if not self._filter.initialized:
            return self._latest_state

        self._predict_to(float(imu.timestamp))
        self._filter.update_acceleration(acceleration_xy)
        return self._refresh_state_from_filter()

    def process_gnss(self, gnss: "GnssMeasurement") -> Optional[EgoState]:
        local = self._gnss_projector.project(gnss)
        if local is None:
            return self._latest_state

        self._latest_gnss_local = local
        self._last_gnss_frame = int(gnss.frame)
        acceleration_xy = self._pending_acceleration_xy or (0.0, 0.0)

        if not self._filter.initialized:
            self._filter.initialize(
                position_xy=(local.x, local.y),
                timestamp=local.timestamp,
                acceleration_xy=acceleration_xy,
            )
            return self._refresh_state_from_filter()

        self._predict_to(local.timestamp)
        self._filter.update_position((local.x, local.y))
        return self._refresh_state_from_filter()

    def get_state(self) -> Optional[EgoState]:
        return self._latest_state

    def get_diagnostics(self) -> dict[str, object]:
        snapshot = self._filter.snapshot()
        covariance = self._filter.covariance
        return {
            "filter_id": FILTER_INFO["id"],
            "initialized": self.initialized,
            "state_vector": [float(value) for value in self._filter.state_vector.reshape(-1)],
            "covariance_diagonal": [float(value) for value in np.diag(covariance)],
            "last_update_type": self._filter.last_update_type,
            "innovation": self._filter.last_innovation,
            "nis": self._filter.last_nis,
            "pending_acceleration_xy": self._pending_acceleration_xy,
            "latest_imu_yaw_deg": self._latest_imu_yaw_deg,
            "tracking_mode": self._tracking_mode,
            "latest_control_input": self._control_input_dict(self._latest_control_input),
            "active_command_used_latest_prediction": self._active_command_used_latest_prediction,
            "latest_command_acceleration_xy": self._latest_command_acceleration_xy,
            "latest_command_yaw_rate_dps": self._latest_command_yaw_rate_dps,
            "latest_command_longitudinal_accel_mps2": self._latest_command_longitudinal_accel_mps2,
            "last_gnss_frame": self._last_gnss_frame,
            "last_imu_frame": self._last_imu_frame,
            "timestamp": snapshot.timestamp if snapshot is not None else None,
        }

    def _predict_to(self, timestamp: float) -> None:
        current_timestamp = self._filter.timestamp
        if current_timestamp is None:
            return

        dt = float(timestamp) - float(current_timestamp)
        if dt <= self._min_prediction_dt_s:
            return

        clipped_dt = min(dt, self._max_prediction_dt_s)
        self._filter.predict(
            dt=clipped_dt,
            timestamp=timestamp,
        )
        self._update_from_command_control(clipped_dt)

    def _update_from_command_control(self, dt: float) -> None:
        self._active_command_used_latest_prediction = False
        if self._tracking_mode != TRACKING_MODE_ACTIVE or self._latest_control_input is None:
            return
        snapshot = self._filter.snapshot()
        if snapshot is None:
            return
        speed = math.hypot(snapshot.vx, snapshot.vy)
        yaw_deg = self._yaw_for_command(snapshot.vx, snapshot.vy, speed)
        command = estimate_command_motion(
            control_input=self._latest_control_input,
            speed_mps=speed,
            yaw_deg=yaw_deg,
            tune=self._tune,
            dt_s=dt,
        )
        self._latest_command_acceleration_xy = command.acceleration_xy
        self._latest_command_yaw_rate_dps = command.yaw_rate_dps
        self._latest_command_longitudinal_accel_mps2 = command.longitudinal_accel_mps2
        self._filter.update_acceleration(
            command.acceleration_xy,
            stddev_mps2=self._command_accel_stddev,
            update_type="command_acceleration",
        )
        self._active_command_used_latest_prediction = True

    def _refresh_state_from_filter(self) -> Optional[EgoState]:
        snapshot = self._filter.snapshot()
        if snapshot is None:
            self._latest_state = None
            return None

        speed = math.hypot(snapshot.vx, snapshot.vy)
        yaw = self._derive_yaw_deg(snapshot.vx, snapshot.vy, speed)
        z = self._latest_gnss_local.z if self._latest_gnss_local is not None else 0.0
        self._latest_state = EgoState(
            x=snapshot.px,
            y=snapshot.py,
            z=float(z),
            yaw=yaw,
            speed=float(speed),
            timestamp=snapshot.timestamp,
        )
        return self._latest_state

    def _derive_yaw_deg(self, vx: float, vy: float, speed: float) -> float:
        if speed >= self._yaw_speed_threshold:
            yaw = math.degrees(math.atan2(vy, vx))
            self._last_valid_yaw_deg = self._normalize_angle_deg(yaw)
            return self._last_valid_yaw_deg

        if self._latest_imu_yaw_deg is not None:
            self._last_valid_yaw_deg = self._latest_imu_yaw_deg
        return self._last_valid_yaw_deg

    def _yaw_for_command(self, vx: float, vy: float, speed: float) -> float:
        if speed >= self._yaw_speed_threshold:
            return self._normalize_angle_deg(math.degrees(math.atan2(vy, vx)))
        if self._latest_imu_yaw_deg is not None:
            return self._latest_imu_yaw_deg
        return self._last_valid_yaw_deg

    def _imu_acceleration_to_world_xy(
        self,
        imu: "ImuMeasurement",
        yaw_deg: Optional[float],
    ) -> tuple[float, float]:
        body_ax, body_ay, _ = imu.accelerometer
        yaw_rad = math.radians(yaw_deg if yaw_deg is not None else self._last_valid_yaw_deg)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        world_ax = cos_yaw * body_ax - sin_yaw * body_ay
        world_ay = sin_yaw * body_ax + cos_yaw * body_ay
        return float(world_ax), float(world_ay)

    @staticmethod
    def _yaw_deg_from_compass(compass_rad: float) -> Optional[float]:
        if not math.isfinite(compass_rad):
            return None
        return Filter._normalize_angle_deg(math.degrees(compass_rad) - 90.0)

    @staticmethod
    def _normalize_angle_deg(angle_deg: float) -> float:
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return float(angle_deg)

    @staticmethod
    def _control_input_dict(control_input: Optional[FilterControlInput]) -> Optional[dict[str, object]]:
        if control_input is None:
            return None
        return {
            "timestamp": control_input.timestamp,
            "throttle": control_input.throttle,
            "steer": control_input.steer,
            "brake": control_input.brake,
            "hand_brake": control_input.hand_brake,
            "reverse": control_input.reverse,
            "source": control_input.source,
            "speed_mps": control_input.speed_mps,
            "yaw_deg": control_input.yaw_deg,
        }
