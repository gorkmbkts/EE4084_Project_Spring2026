"""Ego-state providers and IMU/GNSS localization estimators."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional, Protocol, TYPE_CHECKING

from config.settings import LOCALIZATION
from src.localization.gnss_projection import GnssLocalProjector, LocalGnssMeasurement
from src.localization.kalman_filter import ConstantAccelerationKalmanFilter
from src.utils.carla_import import ensure_carla_import

if TYPE_CHECKING:
    from src.sensors.gnss_sensor import GnssMeasurement, GnssSensor
    from src.sensors.imu_sensor import ImuMeasurement, ImuSensor

carla = ensure_carla_import()


@dataclass(frozen=True)
class EgoState:
    """Common ego-state abstraction used by tracking and control."""

    x: float
    y: float
    z: float
    yaw: float
    speed: float
    timestamp: float

    def distance_xy_to(self, location: "carla.Location") -> float:
        """Return planar distance from this state to a CARLA location."""
        return math.hypot(location.x - self.x, location.y - self.y)


@dataclass(frozen=True)
class LocalizationStatus:
    """Latest estimator state and diagnostics for UI/debugging."""

    filter_name: str
    initialized: bool
    estimated_state: Optional[EgoState]
    ground_truth_state: Optional[EgoState]
    gnss_local: Optional[LocalGnssMeasurement]
    position_error_m: Optional[float]
    last_gnss_frame: Optional[int]
    last_imu_frame: Optional[int]


class EgoStateProvider(Protocol):
    """Interface for state providers consumed by tracking and control."""

    def get_state(self) -> EgoState:
        """Return the current ego state."""
        ...


class StateEstimator(Protocol):
    """Estimator interface intended to stay stable for later EKF work."""

    @property
    def name(self) -> str:
        ...

    @property
    def initialized(self) -> bool:
        ...

    @property
    def latest_gnss_local(self) -> Optional[LocalGnssMeasurement]:
        ...

    def reset(self) -> None:
        ...

    def process_imu(self, imu: "ImuMeasurement") -> Optional[EgoState]:
        ...

    def process_gnss(self, gnss: "GnssMeasurement") -> Optional[EgoState]:
        ...

    def get_state(self) -> Optional[EgoState]:
        ...


class GroundTruthStateProvider:
    """Read ego pose and velocity directly from CARLA actor ground truth."""

    def __init__(self, vehicle: "carla.Vehicle") -> None:
        self._vehicle = vehicle

    def get_state(self) -> EgoState:
        """Convert CARLA vehicle transform and velocity to ``EgoState``."""
        transform = self._vehicle.get_transform()
        velocity = self._vehicle.get_velocity()
        speed = math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)

        try:
            timestamp = float(self._vehicle.get_world().get_snapshot().timestamp.elapsed_seconds)
        except RuntimeError:
            timestamp = time.monotonic()

        return EgoState(
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(transform.location.z),
            yaw=float(transform.rotation.yaw),
            speed=float(speed),
            timestamp=timestamp,
        )


class KalmanStateEstimator:
    """IMU/GNSS localization estimator backed by a linear CA Kalman filter."""

    def __init__(
        self,
        gnss_projector: GnssLocalProjector,
        kalman_filter: Optional[ConstantAccelerationKalmanFilter] = None,
        name: str = LOCALIZATION.estimator_name,
        yaw_from_velocity_min_speed_mps: float = LOCALIZATION.yaw_from_velocity_min_speed_mps,
        min_prediction_dt_s: float = LOCALIZATION.min_prediction_dt_s,
        max_prediction_dt_s: float = LOCALIZATION.max_prediction_dt_s,
    ) -> None:
        self._gnss_projector = gnss_projector
        self._filter = kalman_filter if kalman_filter is not None else ConstantAccelerationKalmanFilter()
        self._name = name
        self._yaw_speed_threshold = float(yaw_from_velocity_min_speed_mps)
        self._min_prediction_dt_s = float(min_prediction_dt_s)
        self._max_prediction_dt_s = float(max_prediction_dt_s)

        self._latest_state: Optional[EgoState] = None
        self._latest_gnss_local: Optional[LocalGnssMeasurement] = None
        self._pending_acceleration_xy: Optional[tuple[float, float]] = None
        self._latest_imu_yaw_deg: Optional[float] = None
        self._last_valid_yaw_deg = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def initialized(self) -> bool:
        return self._filter.initialized

    @property
    def latest_gnss_local(self) -> Optional[LocalGnssMeasurement]:
        return self._latest_gnss_local

    def reset(self) -> None:
        """Reset filter and estimator-side derived state."""
        self._filter.reset()
        self._latest_state = None
        self._latest_gnss_local = None
        self._pending_acceleration_xy = None
        self._latest_imu_yaw_deg = None
        self._last_valid_yaw_deg = 0.0

    def process_imu(self, imu: "ImuMeasurement") -> Optional[EgoState]:
        """Fuse one IMU frame as a world-frame acceleration measurement."""
        yaw_deg = self._yaw_deg_from_compass(imu.compass)
        if yaw_deg is not None:
            self._latest_imu_yaw_deg = yaw_deg

        acceleration_xy = self._imu_acceleration_to_world_xy(imu, yaw_deg)
        self._pending_acceleration_xy = acceleration_xy

        if not self._filter.initialized:
            return self._latest_state

        self._predict_to(float(imu.timestamp))
        self._filter.update_acceleration(acceleration_xy)
        return self._refresh_state_from_filter()

    def process_gnss(self, gnss: "GnssMeasurement") -> Optional[EgoState]:
        """Fuse one GNSS frame after projecting lat/lon into local x/y."""
        local = self._gnss_projector.project(gnss)
        if local is None:
            return self._latest_state

        self._latest_gnss_local = local
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
        """Return the latest estimated state, if the filter has initialized."""
        return self._latest_state

    def _predict_to(self, timestamp: float) -> None:
        current_timestamp = self._filter.timestamp
        if current_timestamp is None:
            return

        dt = float(timestamp) - float(current_timestamp)
        if dt <= self._min_prediction_dt_s:
            return

        self._filter.predict(
            dt=min(dt, self._max_prediction_dt_s),
            timestamp=timestamp,
        )

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
        """Derive heading from velocity, with stable low-speed fallback.

        Velocity direction becomes noisy near zero speed. Below the configured
        speed threshold, the estimator uses IMU compass yaw when available, then
        falls back to the last valid yaw.
        """
        if speed >= self._yaw_speed_threshold:
            yaw = math.degrees(math.atan2(vy, vx))
            self._last_valid_yaw_deg = self._normalize_angle_deg(yaw)
            return self._last_valid_yaw_deg

        if self._latest_imu_yaw_deg is not None:
            self._last_valid_yaw_deg = self._latest_imu_yaw_deg
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
        return KalmanStateEstimator._normalize_angle_deg(math.degrees(compass_rad) - 90.0)

    @staticmethod
    def _normalize_angle_deg(angle_deg: float) -> float:
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return float(angle_deg)


class EstimatedStateProvider:
    """Adapter that feeds latest sensor frames into a localization estimator."""

    def __init__(
        self,
        estimator: StateEstimator,
        gnss_sensor: "GnssSensor",
        imu_sensor: "ImuSensor",
    ) -> None:
        self._estimator = estimator
        self._gnss_sensor = gnss_sensor
        self._imu_sensor = imu_sensor
        self._last_gnss_frame: Optional[int] = None
        self._last_imu_frame: Optional[int] = None

    @property
    def initialized(self) -> bool:
        return self._estimator.initialized

    @property
    def estimator_name(self) -> str:
        return self._estimator.name

    def reset(self, skip_current_sensor_frames: bool = True) -> None:
        """Reset estimator, optionally ignoring already-buffered stale frames."""
        if skip_current_sensor_frames:
            gnss = self._gnss_sensor.get_latest_measurement()
            imu = self._imu_sensor.get_latest_measurement()
            self._last_gnss_frame = gnss.frame if gnss is not None else None
            self._last_imu_frame = imu.frame if imu is not None else None
        else:
            self._last_gnss_frame = None
            self._last_imu_frame = None
        self._estimator.reset()

    def update(self) -> Optional[EgoState]:
        """Process any new sensor frames and return the latest estimate."""
        imu = self._imu_sensor.get_latest_measurement()
        if imu is not None and imu.frame != self._last_imu_frame:
            self._estimator.process_imu(imu)
            self._last_imu_frame = imu.frame

        gnss = self._gnss_sensor.get_latest_measurement()
        if gnss is not None and gnss.frame != self._last_gnss_frame:
            self._estimator.process_gnss(gnss)
            self._last_gnss_frame = gnss.frame

        return self._estimator.get_state()

    def get_state(self) -> EgoState:
        """Return the latest estimated state or raise until GNSS initializes it."""
        state = self.update()
        if state is None:
            raise RuntimeError("Estimated localization state is not initialized yet.")
        return state

    def build_status(self, ground_truth_state: Optional[EgoState]) -> LocalizationStatus:
        """Build UI/debug status without mutating estimator state."""
        estimated_state = self._estimator.get_state()
        position_error_m = None
        if estimated_state is not None and ground_truth_state is not None:
            position_error_m = math.hypot(
                estimated_state.x - ground_truth_state.x,
                estimated_state.y - ground_truth_state.y,
            )

        return LocalizationStatus(
            filter_name=self._estimator.name,
            initialized=self._estimator.initialized,
            estimated_state=estimated_state,
            ground_truth_state=ground_truth_state,
            gnss_local=self._estimator.latest_gnss_local,
            position_error_m=position_error_m,
            last_gnss_frame=self._last_gnss_frame,
            last_imu_frame=self._last_imu_frame,
        )
