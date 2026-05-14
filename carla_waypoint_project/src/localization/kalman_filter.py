"""Linear Kalman filters used by localization estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config.settings import LOCALIZATION


@dataclass(frozen=True)
class KalmanStateSnapshot:
    """Read-only view of the current CA filter state."""

    px: float
    py: float
    vx: float
    vy: float
    ax: float
    ay: float
    timestamp: float


class ConstantAccelerationKalmanFilter:
    """Linear 2D constant-acceleration Kalman filter.

    State vector:
        [px, py, vx, vy, ax, ay]^T

    GNSS local x/y is fused as a position measurement. IMU acceleration is
    rotated into the local map frame by the state estimator and fused as a
    direct acceleration measurement.
    """

    def __init__(
        self,
        process_jerk_stddev_mps3: float = LOCALIZATION.process_jerk_stddev_mps3,
        gnss_position_stddev_m: float = LOCALIZATION.gnss_position_stddev_m,
        imu_accel_stddev_mps2: float = LOCALIZATION.imu_accel_stddev_mps2,
        initial_position_stddev_m: float = LOCALIZATION.initial_position_stddev_m,
        initial_velocity_stddev_mps: float = LOCALIZATION.initial_velocity_stddev_mps,
        initial_accel_stddev_mps2: float = LOCALIZATION.initial_accel_stddev_mps2,
    ) -> None:
        self._process_jerk_var = float(process_jerk_stddev_mps3) ** 2
        self._position_var = float(gnss_position_stddev_m) ** 2
        self._accel_var = float(imu_accel_stddev_mps2) ** 2
        self._initial_position_var = float(initial_position_stddev_m) ** 2
        self._initial_velocity_var = float(initial_velocity_stddev_mps) ** 2
        self._initial_accel_var = float(initial_accel_stddev_mps2) ** 2

        self._x = np.zeros((6, 1), dtype=float)
        self._p = np.eye(6, dtype=float)
        self._timestamp: Optional[float] = None
        self.initialized = False

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
        """Clear the filter so the next GNSS local position reinitializes it."""
        self._x = np.zeros((6, 1), dtype=float)
        self._p = np.eye(6, dtype=float)
        self._timestamp = None
        self.initialized = False

    def initialize(
        self,
        position_xy: tuple[float, float],
        timestamp: float,
        velocity_xy: tuple[float, float] = (0.0, 0.0),
        acceleration_xy: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Initialize state mean and covariance from the first local GNSS fix."""
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
        """Advance the CA process model by ``dt`` seconds."""
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
        """Fuse a GNSS local x/y position measurement."""
        if not self.initialized:
            return

        z = np.array([[float(position_xy[0])], [float(position_xy[1])]], dtype=float)
        h = np.zeros((2, 6), dtype=float)
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        r = np.eye(2, dtype=float) * self._position_var
        self._update(z, h, r)

    def update_acceleration(self, acceleration_xy: tuple[float, float]) -> None:
        """Fuse a world-frame x/y acceleration measurement from the IMU."""
        if not self.initialized:
            return

        z = np.array([[float(acceleration_xy[0])], [float(acceleration_xy[1])]], dtype=float)
        h = np.zeros((2, 6), dtype=float)
        h[0, 4] = 1.0
        h[1, 5] = 1.0
        r = np.eye(2, dtype=float) * self._accel_var
        self._update(z, h, r)

    def snapshot(self) -> Optional[KalmanStateSnapshot]:
        """Return the current state as scalar values."""
        if not self.initialized or self._timestamp is None:
            return None
        return KalmanStateSnapshot(
            px=float(self._x[0, 0]),
            py=float(self._x[1, 0]),
            vx=float(self._x[2, 0]),
            vy=float(self._x[3, 0]),
            ax=float(self._x[4, 0]),
            ay=float(self._x[5, 0]),
            timestamp=float(self._timestamp),
        )

    def _update(self, z: np.ndarray, h: np.ndarray, r: np.ndarray) -> None:
        innovation = z - h @ self._x
        innovation_cov = h @ self._p @ h.T + r
        gain = self._p @ h.T @ np.linalg.inv(innovation_cov)
        identity = np.eye(self._p.shape[0], dtype=float)

        self._x = self._x + gain @ innovation
        # Joseph form keeps covariance positive semi-definite under rounding.
        residual_transform = identity - gain @ h
        self._p = residual_transform @ self._p @ residual_transform.T + gain @ r @ gain.T
        self._p = self._symmetrize(self._p)

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


class KalmanFilter(ConstantAccelerationKalmanFilter):
    """Backward-compatible name for the active linear CA Kalman filter."""
