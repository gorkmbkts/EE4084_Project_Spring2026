"""Runtime manager for active KalmanLab localization filters."""

from __future__ import annotations

import math
from typing import Any, Optional

from src.KalmanLab.filter_base import FilterPluginRecord, LocalizationFilter
from src.KalmanLab.registry import discover_filters
from src.localization.gnss_projection import GnssLocalProjector, LocalGnssMeasurement
from src.localization.state_estimator import EgoState, LocalizationStatus


class FilterManager:
    """Discover filters, own the active filter, and feed it sensor frames."""

    def __init__(
        self,
        gnss_projector: GnssLocalProjector,
        gnss_sensor: object,
        imu_sensor: object,
        default_filter_id: str = "ca_kf",
    ) -> None:
        self._gnss_projector = gnss_projector
        self._gnss_sensor = gnss_sensor
        self._imu_sensor = imu_sensor
        self._records = discover_filters()
        self._valid_records = {record.filter_id: record for record in self._records if record.valid}
        self._active_filter_id: Optional[str] = None
        self._active_filter: Optional[LocalizationFilter] = None
        self._last_gnss_frame: Optional[int] = None
        self._last_imu_frame: Optional[int] = None
        self._runtime_error: Optional[str] = None
        self._last_switch_message = ""

        initial_id = default_filter_id if default_filter_id in self._valid_records else self._first_valid_filter_id()
        if initial_id is not None:
            self.switch_filter(initial_id, skip_current_sensor_frames=False)

    @property
    def active_filter_id(self) -> Optional[str]:
        return self._active_filter_id

    @property
    def last_switch_message(self) -> str:
        return self._last_switch_message

    @property
    def runtime_error(self) -> Optional[str]:
        return self._runtime_error

    def available_filters(self) -> tuple[FilterPluginRecord, ...]:
        return tuple(record for record in self._records if record.valid)

    def invalid_filters(self, include_templates: bool = False) -> tuple[FilterPluginRecord, ...]:
        return tuple(
            record
            for record in self._records
            if not record.valid and (include_templates or not record.template)
        )

    def all_records(self) -> tuple[FilterPluginRecord, ...]:
        return tuple(self._records)

    def switch_filter(self, filter_id: str, skip_current_sensor_frames: bool = True) -> tuple[bool, str]:
        """Switch to a valid plugin, resetting frame bookkeeping and state."""
        record = self._valid_records.get(filter_id)
        if record is None or record.filter_class is None:
            message = f"Filter not available: {filter_id}"
            self._last_switch_message = message
            return False, message

        try:
            filter_instance = record.filter_class(self._gnss_projector)
        except Exception as exc:
            message = f"Failed to instantiate {record.display_name}: {exc}"
            self._last_switch_message = message
            return False, message

        self._active_filter_id = record.filter_id
        self._active_filter = filter_instance
        self._runtime_error = None
        self._set_skipped_sensor_frames(skip_current_sensor_frames)
        message = f"Active filter: {record.display_name}"
        self._last_switch_message = message
        return True, message

    def reset(self, skip_current_sensor_frames: bool = True) -> None:
        """Reset the active filter and optionally ignore buffered stale frames."""
        self._set_skipped_sensor_frames(skip_current_sensor_frames)
        self._runtime_error = None
        if self._active_filter is not None:
            try:
                self._active_filter.reset()
            except Exception as exc:
                self._runtime_error = f"Reset failed: {exc}"

    def update(self) -> Optional[EgoState]:
        """Process any new sensor frames and return the latest active-filter state."""
        if self._active_filter is None:
            return None

        imu = self._latest_sensor_measurement(self._imu_sensor)
        if imu is not None and getattr(imu, "frame", None) != self._last_imu_frame:
            if not self._call_filter_method("process_imu", imu):
                return self.get_state()
            self._last_imu_frame = int(imu.frame)

        gnss = self._latest_sensor_measurement(self._gnss_sensor)
        if gnss is not None and getattr(gnss, "frame", None) != self._last_gnss_frame:
            if not self._call_filter_method("process_gnss", gnss):
                return self.get_state()
            self._last_gnss_frame = int(gnss.frame)

        return self.get_state()

    def get_state(self) -> Optional[EgoState]:
        if self._active_filter is None:
            return None
        try:
            state = self._active_filter.get_state()
        except Exception as exc:
            self._runtime_error = f"get_state failed: {exc}"
            return None
        return state if isinstance(state, EgoState) else state

    def get_status(self, ground_truth_state: Optional[EgoState]) -> LocalizationStatus:
        estimated_state = self.get_state()
        position_error_m = None
        if estimated_state is not None and ground_truth_state is not None:
            position_error_m = math.hypot(
                estimated_state.x - ground_truth_state.x,
                estimated_state.y - ground_truth_state.y,
            )

        return LocalizationStatus(
            filter_name=self.get_active_filter_name(),
            initialized=self.initialized,
            estimated_state=estimated_state,
            ground_truth_state=ground_truth_state,
            gnss_local=self.latest_gnss_local,
            position_error_m=position_error_m,
            last_gnss_frame=self._last_gnss_frame,
            last_imu_frame=self._last_imu_frame,
        )

    @property
    def initialized(self) -> bool:
        if self._active_filter is None:
            return False
        return bool(getattr(self._active_filter, "initialized", False))

    @property
    def latest_gnss_local(self) -> Optional[LocalGnssMeasurement]:
        if self._active_filter is None:
            return None
        value = getattr(self._active_filter, "latest_gnss_local", None)
        return value if isinstance(value, LocalGnssMeasurement) or value is None else None

    def get_active_filter_name(self) -> str:
        info = self.get_active_filter_info()
        return str(info.get("name") or self._active_filter_id or "No filter")

    def get_active_filter_info(self) -> dict[str, Any]:
        if self._active_filter_id is None:
            return {}
        record = self._valid_records.get(self._active_filter_id)
        return dict(record.filter_info) if record is not None else {}

    def get_active_filter_tune(self) -> dict[str, Any]:
        if self._active_filter_id is None:
            return {}
        record = self._valid_records.get(self._active_filter_id)
        return dict(record.tune) if record is not None else {}

    def active_filter_safe_for_autonomous_control(self) -> bool:
        info = self.get_active_filter_info()
        return bool(info.get("safe_for_autonomous_control", True))

    def get_diagnostics(self) -> dict[str, Any]:
        if self._active_filter is None:
            return {"runtime_error": self._runtime_error} if self._runtime_error else {}
        try:
            diagnostics = self._active_filter.get_diagnostics()
        except AttributeError:
            diagnostics = {}
        except Exception as exc:
            self._runtime_error = f"get_diagnostics failed: {exc}"
            diagnostics = {}
        if not isinstance(diagnostics, dict):
            diagnostics = {"diagnostics": diagnostics}
        if self._runtime_error:
            diagnostics = dict(diagnostics)
            diagnostics["runtime_error"] = self._runtime_error
        diagnostics.setdefault("last_gnss_frame", self._last_gnss_frame)
        diagnostics.setdefault("last_imu_frame", self._last_imu_frame)
        return diagnostics

    def _call_filter_method(self, method_name: str, measurement: object) -> bool:
        if self._active_filter is None:
            return False
        try:
            getattr(self._active_filter, method_name)(measurement)
        except Exception as exc:
            self._runtime_error = f"{method_name} failed: {exc}"
            return False
        return True

    def _set_skipped_sensor_frames(self, skip_current_sensor_frames: bool) -> None:
        if skip_current_sensor_frames:
            gnss = self._latest_sensor_measurement(self._gnss_sensor)
            imu = self._latest_sensor_measurement(self._imu_sensor)
            self._last_gnss_frame = int(gnss.frame) if gnss is not None else None
            self._last_imu_frame = int(imu.frame) if imu is not None else None
        else:
            self._last_gnss_frame = None
            self._last_imu_frame = None

    @staticmethod
    def _latest_sensor_measurement(sensor: object) -> object | None:
        getter = getattr(sensor, "get_latest_measurement", None)
        if getter is None:
            return None
        return getter()

    def _first_valid_filter_id(self) -> Optional[str]:
        for record in self._records:
            if record.valid:
                return record.filter_id
        return None
