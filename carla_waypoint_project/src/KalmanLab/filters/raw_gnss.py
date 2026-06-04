"""Raw GNSS localization plugin for KalmanLab baseline comparisons."""

from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

from src.KalmanLab.filter_base import normalize_tracking_mode
from src.core.vehicle_state import VehicleState
from src.evaluation.benchmark_config import ParameterSpec
from src.localization.gnss_projection import GnssLocalProjector, LocalGnssMeasurement

if TYPE_CHECKING:
    from src.sensors.gnss_sensor import GnssMeasurement
    from src.sensors.imu_sensor import ImuMeasurement


FILTER_INFO = {
    "id": "raw_gnss",
    "name": "Raw GNSS",
    "type": "Baseline",
    "state_vector": "[px, py, speed, yaw]^T",
    "process_model": "None",
    "measurement_model": "Projected GNSS latitude/longitude",
    "description": "Projected raw GNSS position with speed and yaw estimated from successive fixes.",
    "model_type": "RAW_GNSS",
    "provided_state_fields": ("x", "y", "z", "yaw", "speed", "timestamp"),
    "safe_for_autonomous_control": False,
    "active_tracking_supported": False,
    "benchmark_selectable": False,
    "experimental": False,
    "requires_raw_imu": False,
}


TUNE = {
    "yaw_from_velocity_min_speed_mps": 0.35,
    "max_speed_step_mps": 80.0,
}


TUNE_SPECS = (
    ParameterSpec("yaw_from_velocity_min_speed_mps", "Yaw min speed", 0.05, 3.0, "m/s", 2, "Raw GNSS"),
    ParameterSpec("max_speed_step_mps", "Max speed step", 5.0, 160.0, "m/s", 1, "Raw GNSS"),
)


class Filter:
    """Use raw projected GNSS as a state-producing baseline filter."""

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
        self._latest_state: Optional[VehicleState] = None
        self._latest_gnss_local: Optional[LocalGnssMeasurement] = None
        self._previous_gnss_local: Optional[LocalGnssMeasurement] = None
        self._latest_imu_yaw_deg: Optional[float] = None
        self._last_valid_yaw_deg = 0.0
        self._speed_mps = 0.0
        self._last_gnss_frame: Optional[int] = None
        self._last_imu_frame: Optional[int] = None

    @property
    def initialized(self) -> bool:
        return self._latest_state is not None

    @property
    def latest_gnss_local(self) -> Optional[LocalGnssMeasurement]:
        return self._latest_gnss_local

    def reset(self) -> None:
        self._latest_state = None
        self._latest_gnss_local = None
        self._previous_gnss_local = None
        self._latest_imu_yaw_deg = None
        self._last_valid_yaw_deg = 0.0
        self._speed_mps = 0.0
        self._last_gnss_frame = None
        self._last_imu_frame = None

    def process_imu(self, imu: "ImuMeasurement") -> Optional[VehicleState]:
        yaw_deg = self._yaw_deg_from_compass(imu.compass)
        if yaw_deg is not None:
            self._latest_imu_yaw_deg = yaw_deg
            if self._speed_mps < float(self._tune["yaw_from_velocity_min_speed_mps"]):
                self._last_valid_yaw_deg = yaw_deg
        self._last_imu_frame = int(imu.frame)
        return self._latest_state

    def process_gnss(self, gnss: "GnssMeasurement") -> Optional[VehicleState]:
        local = self._gnss_projector.project(gnss)
        if local is None:
            return self._latest_state

        speed, yaw = self._speed_and_yaw_from_gnss(local)
        self._previous_gnss_local = self._latest_gnss_local
        self._latest_gnss_local = local
        self._speed_mps = speed
        self._last_gnss_frame = int(gnss.frame)
        self._latest_state = VehicleState(
            x=float(local.x),
            y=float(local.y),
            z=float(local.z),
            yaw=float(yaw),
            speed=float(speed),
            timestamp=float(local.timestamp),
            source_filter_id=FILTER_INFO["id"],
            model_type=FILTER_INFO["model_type"],
            diagnostics_summary={"warning": "Raw GNSS is noisy and may be unsafe for closed-loop control."},
            safe_for_autonomous_control=False,
            active_tracking_supported=False,
        )
        return self._latest_state

    def get_state(self) -> Optional[VehicleState]:
        return self._latest_state

    def get_diagnostics(self) -> dict[str, object]:
        return {
            "filter_id": FILTER_INFO["id"],
            "initialized": self.initialized,
            "safe_for_autonomous_control": False,
            "tracking_mode": self._tracking_mode,
            "active_tracking_supported": False,
            "warning": "Raw GNSS is noisy and may be unsafe for closed-loop control.",
            "latest_speed_mps": self._speed_mps,
            "latest_yaw_deg": self._last_valid_yaw_deg,
            "last_gnss_frame": self._last_gnss_frame,
            "last_imu_frame": self._last_imu_frame,
        }

    def _speed_and_yaw_from_gnss(self, local: LocalGnssMeasurement) -> tuple[float, float]:
        if self._latest_gnss_local is None:
            yaw = self._latest_imu_yaw_deg if self._latest_imu_yaw_deg is not None else self._last_valid_yaw_deg
            self._last_valid_yaw_deg = float(yaw)
            return 0.0, self._last_valid_yaw_deg

        dt = float(local.timestamp) - float(self._latest_gnss_local.timestamp)
        if dt <= 1.0e-6:
            return self._speed_mps, self._last_valid_yaw_deg

        dx = float(local.x) - float(self._latest_gnss_local.x)
        dy = float(local.y) - float(self._latest_gnss_local.y)
        speed = min(math.hypot(dx, dy) / dt, float(self._tune["max_speed_step_mps"]))
        if speed >= float(self._tune["yaw_from_velocity_min_speed_mps"]):
            self._last_valid_yaw_deg = self._normalize_angle_deg(math.degrees(math.atan2(dy, dx)))
        elif self._latest_imu_yaw_deg is not None:
            self._last_valid_yaw_deg = self._latest_imu_yaw_deg
        return float(speed), self._last_valid_yaw_deg

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
