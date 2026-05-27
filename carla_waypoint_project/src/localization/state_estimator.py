"""Ego-state providers and shared localization status types."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional, Protocol, TYPE_CHECKING

from src.localization.gnss_projection import LocalGnssMeasurement
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
