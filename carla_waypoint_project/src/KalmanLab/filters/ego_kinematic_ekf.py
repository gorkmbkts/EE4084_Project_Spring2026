"""Ego-state kinematic EKF plugin for KalmanLab.

This is an intermediate nonlinear EKF before a CTRV model. It estimates:

    x = [px, py, yaw, speed]^T

Yaw is stored internally in radians. The public ``EgoState`` output uses
degrees, matching the rest of the localization/control interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, TYPE_CHECKING

import numpy as np

from src.KalmanLab.filter_base import FilterControlInput, normalize_tracking_mode
from src.evaluation.benchmark_config import ParameterSpec
from src.localization.gnss_projection import GnssLocalProjector, LocalGnssMeasurement
from src.localization.state_estimator import EgoState

if TYPE_CHECKING:
    from src.sensors.gnss_sensor import GnssMeasurement
    from src.sensors.imu_sensor import ImuMeasurement


FILTER_INFO = {
    "id": "ego_kinematic_ekf",
    "name": "Ego Kinematic EKF",
    "type": "Extended Kalman Filter",
    "state_vector": "[px, py, yaw, speed]^T",
    "process_model": "Nonlinear constant-heading constant-speed kinematic model",
    "measurement_model": "GNSS position x/y + IMU compass yaw",
    "description": "EKF using an EgoState-like state with nonlinear x/y propagation from yaw and speed.",
    "safe_for_autonomous_control": True,
    "active_tracking_supported": False,
}


TUNE = {
    "process_accel_stddev_mps2": 3.0,
    "process_yaw_rate_stddev_dps": 15.0,
    "gnss_position_stddev_m": 1.25,
    "imu_yaw_stddev_deg": 5.0,
    "initial_position_stddev_m": 4.0,
    "initial_yaw_stddev_deg": 20.0,
    "initial_speed_stddev_mps": 4.0,
    "yaw_from_velocity_min_speed_mps": 0.35,
    "min_prediction_dt_s": 1.0e-4,
    "max_prediction_dt_s": 0.20,
}


TUNE_SPECS = (
    ParameterSpec("process_accel_stddev_mps2", "Process accel", 0.05, 12.0, "m/s2", 2, "Noise"),
    ParameterSpec("process_yaw_rate_stddev_dps", "Process yaw rate", 0.5, 90.0, "deg/s", 1, "Noise"),
    ParameterSpec("gnss_position_stddev_m", "GNSS position", 0.10, 12.0, "m", 2, "Noise"),
    ParameterSpec("imu_yaw_stddev_deg", "IMU yaw", 0.5, 45.0, "deg", 1, "Noise"),
    ParameterSpec("initial_position_stddev_m", "Initial pos", 0.25, 25.0, "m", 2, "Initialization"),
    ParameterSpec("initial_yaw_stddev_deg", "Initial yaw", 1.0, 90.0, "deg", 1, "Initialization"),
    ParameterSpec("initial_speed_stddev_mps", "Initial speed", 0.10, 15.0, "m/s", 2, "Initialization"),
    ParameterSpec("yaw_from_velocity_min_speed_mps", "Yaw min speed", 0.05, 3.0, "m/s", 2, "Yaw"),
    ParameterSpec("min_prediction_dt_s", "Min dt", 0.00001, 0.02, "s", 5, "Prediction"),
    ParameterSpec("max_prediction_dt_s", "Max dt", 0.02, 0.60, "s", 2, "Prediction"),
)


def normalize_angle_rad(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi]."""
    if not math.isfinite(angle):
        return 0.0
    wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped == -math.pi and angle > 0.0:
        return math.pi
    return float(wrapped)


def normalize_angle_deg(angle: float) -> float:
    """Wrap an angle in degrees to [-180, 180]."""
    if not math.isfinite(angle):
        return 0.0
    wrapped = (float(angle) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0 and angle > 0.0:
        return 180.0
    return float(wrapped)


def yaw_deg_from_compass(compass_rad: float) -> Optional[float]:
    """Convert CARLA IMU compass radians to the yaw convention used by CA/CV filters."""
    if not math.isfinite(compass_rad):
        return None
    return normalize_angle_deg(math.degrees(float(compass_rad)) - 90.0)


@dataclass(frozen=True)
class _EgoKinematicSnapshot:
    px: float
    py: float
    yaw_rad: float
    speed: float
    timestamp: float


class _EgoKinematicEkfCore:
    """Small EKF core for [px, py, yaw, speed]^T."""

    def __init__(self, tune: dict[str, float]) -> None:
        self._accel_var = float(tune["process_accel_stddev_mps2"]) ** 2
        self._yaw_rate_var = math.radians(float(tune["process_yaw_rate_stddev_dps"])) ** 2
        self._position_var = float(tune["gnss_position_stddev_m"]) ** 2
        self._yaw_var = math.radians(float(tune["imu_yaw_stddev_deg"])) ** 2
        self._initial_position_var = float(tune["initial_position_stddev_m"]) ** 2
        self._initial_yaw_var = math.radians(float(tune["initial_yaw_stddev_deg"])) ** 2
        self._initial_speed_var = float(tune["initial_speed_stddev_mps"]) ** 2

        self._x = np.zeros((4, 1), dtype=float)
        self._p = np.eye(4, dtype=float)
        self._timestamp: Optional[float] = None

        self.initialized = False
        self.last_update_type: Optional[str] = None
        self.last_innovation: Optional[list[float]] = None
        self.last_nis: Optional[float] = None
        self.last_runtime_warning: Optional[str] = None
        self.latest_predicted_state: Optional[list[float]] = None

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
        self._x = np.zeros((4, 1), dtype=float)
        self._p = np.eye(4, dtype=float)
        self._timestamp = None

        self.initialized = False
        self.last_update_type = None
        self.last_innovation = None
        self.last_nis = None
        self.last_runtime_warning = None
        self.latest_predicted_state = None

    def initialize(
        self,
        position_xy: tuple[float, float],
        yaw_rad: float,
        timestamp: float,
        speed_mps: float = 0.0,
    ) -> None:
        self._x = np.array(
            [
                [float(position_xy[0])],
                [float(position_xy[1])],
                [normalize_angle_rad(float(yaw_rad))],
                [max(0.0, float(speed_mps))],
            ],
            dtype=float,
        )
        self._p = np.diag(
            [
                self._initial_position_var,
                self._initial_position_var,
                self._initial_yaw_var,
                self._initial_speed_var,
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

        px = float(self._x[0, 0])
        py = float(self._x[1, 0])
        yaw = float(self._x[2, 0])
        speed = max(0.0, float(self._x[3, 0]))

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        predicted = np.array(
            [
                [px + speed * cos_yaw * dt],
                [py + speed * sin_yaw * dt],
                [normalize_angle_rad(yaw)],
                [speed],
            ],
            dtype=float,
        )

        f = self._jacobian(yaw, speed, dt)
        q = self._process_noise(yaw, dt)
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
        h = np.zeros((2, 4), dtype=float)
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        r = np.eye(2, dtype=float) * self._position_var
        self._update(z, h, r, "gnss_position")

    def update_yaw(self, yaw_rad: float) -> None:
        if not self.initialized:
            return
        if not math.isfinite(yaw_rad):
            self._skip_update("imu_yaw", "non-finite IMU yaw")
            return

        z = np.array([[normalize_angle_rad(float(yaw_rad))]], dtype=float)
        h = np.zeros((1, 4), dtype=float)
        h[0, 2] = 1.0
        r = np.array([[self._yaw_var]], dtype=float)
        self._update(z, h, r, "imu_yaw", wrap_yaw_innovation=True)

    def snapshot(self) -> Optional[_EgoKinematicSnapshot]:
        if not self.initialized or self._timestamp is None:
            return None
        return _EgoKinematicSnapshot(
            px=float(self._x[0, 0]),
            py=float(self._x[1, 0]),
            yaw_rad=normalize_angle_rad(float(self._x[2, 0])),
            speed=max(0.0, float(self._x[3, 0])),
            timestamp=float(self._timestamp),
        )

    def _update(
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
        next_x[2, 0] = normalize_angle_rad(float(next_x[2, 0]))
        next_x[3, 0] = max(0.0, float(next_x[3, 0]))

        identity = np.eye(self._p.shape[0], dtype=float)
        residual_transform = identity - gain @ h
        next_p = residual_transform @ self._p @ residual_transform.T + gain @ r @ gain.T

        if not self._set_state_and_covariance(next_x, next_p, update_type):
            return

        self.last_update_type = update_type
        self.last_innovation = [float(value) for value in innovation.reshape(-1)]
        self.last_nis = float((innovation.T @ solved_innovation)[0, 0])
        self.last_runtime_warning = None

    def _set_state_and_covariance(self, state: np.ndarray, covariance: np.ndarray, context: str) -> bool:
        state[2, 0] = normalize_angle_rad(float(state[2, 0]))
        state[3, 0] = max(0.0, float(state[3, 0]))
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

    def _skip_update(self, update_type: str, reason: str) -> None:
        self.last_update_type = f"{update_type}_skipped"
        self.last_innovation = None
        self.last_nis = None
        self.last_runtime_warning = reason

    @staticmethod
    def _jacobian(yaw: float, speed: float, dt: float) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, -speed * math.sin(yaw) * dt, math.cos(yaw) * dt],
                [0.0, 1.0, speed * math.cos(yaw) * dt, math.sin(yaw) * dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _process_noise(self, yaw: float, dt: float) -> np.ndarray:
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        accel_gain = np.array(
            [
                [0.5 * cos_yaw * dt * dt],
                [0.5 * sin_yaw * dt * dt],
                [0.0],
                [dt],
            ],
            dtype=float,
        )
        yaw_rate_gain = np.array([[0.0], [0.0], [dt], [0.0]], dtype=float)
        q = self._accel_var * (accel_gain @ accel_gain.T)
        q += self._yaw_rate_var * (yaw_rate_gain @ yaw_rate_gain.T)

        position_process_var = 0.25 * self._accel_var * dt**4
        q[0, 0] += position_process_var
        q[1, 1] += position_process_var
        q += np.eye(4, dtype=float) * 1.0e-12
        return self._symmetrize(q)

    @staticmethod
    def _symmetrize(matrix: np.ndarray) -> np.ndarray:
        return 0.5 * (matrix + matrix.T)


class Filter:
    """Self-contained Ego Kinematic EKF plugin with GNSS and IMU yaw updates."""

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
        self._filter = _EgoKinematicEkfCore(self._tune)
        self._min_prediction_dt_s = float(self._tune["min_prediction_dt_s"])
        self._max_prediction_dt_s = float(self._tune["max_prediction_dt_s"])
        self._yaw_speed_threshold = float(self._tune["yaw_from_velocity_min_speed_mps"])

        self._latest_state: Optional[EgoState] = None
        self._latest_gnss_local: Optional[LocalGnssMeasurement] = None
        self._latest_imu_yaw_rad: Optional[float] = None
        self._latest_imu_yaw_deg: Optional[float] = None
        self._latest_control_input: Optional[FilterControlInput] = None
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
        self._latest_imu_yaw_rad = None
        self._latest_imu_yaw_deg = None
        self._latest_control_input = None
        self._last_imu_frame = None
        self._last_gnss_frame = None

    def process_imu(self, imu: "ImuMeasurement") -> Optional[EgoState]:
        yaw_deg = yaw_deg_from_compass(float(imu.compass))
        yaw_rad = None if yaw_deg is None else math.radians(yaw_deg)
        if yaw_rad is not None:
            self._latest_imu_yaw_rad = yaw_rad
            self._latest_imu_yaw_deg = yaw_deg

        self._last_imu_frame = int(imu.frame)
        if not self._filter.initialized:
            return self._latest_state

        self._predict_to(float(imu.timestamp))
        if yaw_rad is not None:
            self._filter.update_yaw(yaw_rad)
        return self._refresh_state_from_filter()

    def process_gnss(self, gnss: "GnssMeasurement") -> Optional[EgoState]:
        local = self._gnss_projector.project(gnss)
        if local is None:
            return self._latest_state

        self._latest_gnss_local = local
        self._last_gnss_frame = int(gnss.frame)

        if not self._filter.initialized:
            yaw_rad = self._latest_imu_yaw_rad if self._latest_imu_yaw_rad is not None else 0.0
            self._filter.initialize(
                position_xy=(local.x, local.y),
                yaw_rad=yaw_rad,
                timestamp=local.timestamp,
                speed_mps=0.0,
            )
            return self._refresh_state_from_filter()

        self._predict_to(local.timestamp)
        self._filter.update_gnss_position((local.x, local.y))
        return self._refresh_state_from_filter()

    def process_control(self, control_input: FilterControlInput) -> bool:
        self._latest_control_input = control_input
        return False

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
            "latest_predicted_state": self._filter.latest_predicted_state,
            "latest_imu_yaw_deg": self._latest_imu_yaw_deg,
            "latest_gnss_local": self._local_gnss_dict(self._latest_gnss_local),
            "last_gnss_frame": self._last_gnss_frame,
            "last_imu_frame": self._last_imu_frame,
            "tracking_mode": self._tracking_mode,
            "active_tracking_supported": False,
            "latest_control_input": self._control_input_dict(self._latest_control_input),
            "yaw_from_velocity_min_speed_mps": self._yaw_speed_threshold,
            "runtime_warning": self._filter.last_runtime_warning,
            "timestamp": snapshot.timestamp if snapshot is not None else None,
            "note": "Non-CTRV kinematic EKF: constant heading, constant speed, nonlinear x/y propagation.",
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
