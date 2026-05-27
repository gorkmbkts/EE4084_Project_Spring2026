"""Lightweight shared types for KalmanLab filter plugins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, TYPE_CHECKING

from src.localization.gnss_projection import LocalGnssMeasurement
from src.localization.state_estimator import EgoState

if TYPE_CHECKING:
    from src.sensors.gnss_sensor import GnssMeasurement
    from src.sensors.imu_sensor import ImuMeasurement


REQUIRED_FILTER_INFO_FIELDS = (
    "id",
    "name",
    "type",
    "state_vector",
    "process_model",
    "measurement_model",
    "description",
)


@dataclass(frozen=True)
class FilterPluginRecord:
    """Discovery result for one file in ``src/KalmanLab/filters``."""

    module_name: str
    file_path: Path
    valid: bool
    filter_info: dict[str, Any]
    tune: dict[str, Any]
    filter_class: Optional[type]
    error: Optional[str] = None
    template: bool = False

    @property
    def filter_id(self) -> str:
        return str(self.filter_info.get("id") or self.module_name)

    @property
    def display_name(self) -> str:
        return str(self.filter_info.get("name") or self.filter_id)

    @property
    def safe_for_autonomous_control(self) -> bool:
        return bool(self.filter_info.get("safe_for_autonomous_control", True))


class LocalizationFilter(Protocol):
    """Duck-typed plugin surface consumed by ``FilterManager``."""

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

    def get_diagnostics(self) -> dict[str, Any]:
        ...
