"""Application-level localization status shared with UI and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from src.core.vehicle_state import VehicleState

if TYPE_CHECKING:
    from src.localization.gnss_projection import LocalGnssMeasurement


@dataclass(frozen=True)
class LocalizationStatus:
    """Latest estimator state and diagnostics for UI/debugging."""

    filter_name: str
    initialized: bool
    estimated_state: Optional[VehicleState]
    ground_truth_state: Optional[VehicleState]
    gnss_local: Optional["LocalGnssMeasurement"]
    position_error_m: Optional[float]
    last_gnss_frame: Optional[int]
    last_imu_frame: Optional[int]
