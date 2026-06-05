"""Runtime manager for active KalmanLab localization filters."""

from __future__ import annotations

import math
import inspect
from typing import Any, Optional

from src.KalmanLab.filter_base import (
    FilterControlInput,
    FilterPluginRecord,
    LocalizationFilter,
    TRACKING_MODE_ACTIVE,
    TRACKING_MODE_PASSIVE,
    normalize_tracking_mode,
)
from src.KalmanLab.registry import discover_filters
from src.core.localization_status import LocalizationStatus
from src.core.vehicle_state import VehicleState
from src.localization.gnss_projection import GnssLocalProjector, LocalGnssMeasurement


class FilterManager:
    """Discover filters, own the active filter, and feed it sensor frames."""

    def __init__(
        self,
        gnss_projector: GnssLocalProjector,
        gnss_sensor: object,
        imu_sensor: object,
        default_filter_id: str = "ca_kf",
        default_tune_overrides: Optional[dict[str, dict[str, object]]] = None,
        tracking_mode: str = TRACKING_MODE_PASSIVE,
    ) -> None:
        self._gnss_projector = gnss_projector
        self._gnss_sensor = gnss_sensor
        self._imu_sensor = imu_sensor
        self._records = discover_filters()
        self._valid_records = {record.filter_id: record for record in self._records if record.valid}
        self._runtime_tunes: dict[str, dict[str, Any]] = {
            filter_id: self._clamped_tune_values(record, dict(record.tune))
            for filter_id, record in self._valid_records.items()
        }
        for filter_id, values in (default_tune_overrides or {}).items():
            if filter_id in self._valid_records and isinstance(values, dict):
                self._runtime_tunes[filter_id] = self._merged_clamped_tune(filter_id, values)
        self._tracking_mode = normalize_tracking_mode(tracking_mode)
        self._active_filter_id: Optional[str] = None
        self._active_filter: Optional[LocalizationFilter] = None
        self._last_gnss_frame: Optional[int] = None
        self._last_imu_frame: Optional[int] = None
        self._runtime_error: Optional[str] = None
        self._last_switch_message = ""
        self._tracking_mode_message = ""
        self._last_control_input: Optional[FilterControlInput] = None
        self._active_control_input_used = False

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

    @property
    def tracking_mode(self) -> str:
        return self._tracking_mode

    @property
    def tracking_mode_message(self) -> str:
        return self._tracking_mode_message

    @property
    def active_control_input_used(self) -> bool:
        return self._active_control_input_used

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
            effective_mode = self._effective_tracking_mode(record)
            filter_instance = self._instantiate_filter(record, effective_mode)
        except Exception as exc:
            message = f"Failed to instantiate {record.display_name}: {exc}"
            self._last_switch_message = message
            return False, message

        if self._tracking_mode == TRACKING_MODE_ACTIVE and effective_mode != TRACKING_MODE_ACTIVE:
            self._tracking_mode = TRACKING_MODE_PASSIVE
            self._tracking_mode_message = f"Active mode unsupported by {record.display_name}; using passive tracking."
        elif self._tracking_mode == TRACKING_MODE_ACTIVE:
            self._tracking_mode_message = f"Active tracking enabled for {record.display_name}."
        else:
            self._tracking_mode_message = "Passive tracking mode."

        self._active_filter_id = record.filter_id
        self._active_filter = filter_instance
        self._runtime_error = None
        self._last_control_input = None
        self._active_control_input_used = False
        self._set_skipped_sensor_frames(skip_current_sensor_frames)
        message = f"Active filter: {record.display_name}"
        if self._tracking_mode_message:
            message = f"{message} | {self._tracking_mode_message}"
        self._last_switch_message = message
        return True, message

    def set_tracking_mode(self, tracking_mode: str, reset_active: bool = True) -> tuple[bool, str]:
        """Set passive/active tracking and optionally rebuild the active filter."""
        requested = normalize_tracking_mode(tracking_mode)
        previous_mode = self._tracking_mode
        if requested == TRACKING_MODE_ACTIVE and not self.active_filter_supports_control_input():
            self._tracking_mode = TRACKING_MODE_PASSIVE
            name = self.get_active_filter_name()
            message = f"Active mode unsupported by {name}; using passive tracking."
            self._tracking_mode_message = message
            self._last_switch_message = message
            return False, message

        self._tracking_mode = requested
        self._tracking_mode_message = (
            "Active tracking enabled." if requested == TRACKING_MODE_ACTIVE else "Passive tracking mode."
        )
        if reset_active and self._active_filter_id is not None:
            ok, switch_message = self.switch_filter(self._active_filter_id, skip_current_sensor_frames=True)
            if not ok:
                self._tracking_mode = previous_mode
                self._tracking_mode_message = f"Tracking mode change failed: {switch_message}"
                self._last_switch_message = self._tracking_mode_message
            return ok, switch_message
        self._last_switch_message = self._tracking_mode_message
        return True, self._tracking_mode_message

    def reset(self, skip_current_sensor_frames: bool = True) -> None:
        """Reset the active filter and optionally ignore buffered stale frames."""
        self._set_skipped_sensor_frames(skip_current_sensor_frames)
        self._runtime_error = None
        if self._active_filter is not None:
            try:
                self._active_filter.reset()
                self._last_control_input = None
                self._active_control_input_used = False
            except Exception as exc:
                self._runtime_error = f"Reset failed: {exc}"

    def update(self) -> Optional[VehicleState]:
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

    def get_state(self) -> Optional[VehicleState]:
        if self._active_filter is None:
            return None
        try:
            state = self._active_filter.get_state()
        except Exception as exc:
            self._runtime_error = f"get_state failed: {exc}"
            return None
        return state if isinstance(state, VehicleState) or state is None else state

    def get_status(self, ground_truth_state: Optional[VehicleState]) -> LocalizationStatus:
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
        return self.get_filter_runtime_tune(self._active_filter_id)

    def get_filter_tune_specs(self, filter_id: str) -> tuple[Any, ...]:
        record = self._valid_records.get(filter_id)
        return tuple(record.tune_specs) if record is not None else ()

    def get_filter_runtime_tune(self, filter_id: str) -> dict[str, Any]:
        record = self._valid_records.get(filter_id)
        if record is None:
            return {}
        return dict(self._runtime_tunes.get(filter_id, record.tune))

    def update_filter_tune(
        self,
        filter_id: str,
        values: dict[str, object],
        reset_active: bool = True,
    ) -> tuple[bool, str]:
        """Apply runtime tune values, clamped through specs, preserving them by filter id."""
        record = self._valid_records.get(filter_id)
        if record is None:
            return False, f"Filter not available: {filter_id}"
        self._runtime_tunes[filter_id] = self._merged_clamped_tune(filter_id, values)
        if reset_active and filter_id == self._active_filter_id:
            return self.switch_filter(filter_id, skip_current_sensor_frames=True)
        return True, f"Tune updated for {record.display_name}"

    def active_filter_safe_for_autonomous_control(self) -> bool:
        info = self.get_active_filter_info()
        return bool(info.get("safe_for_autonomous_control", True))

    def active_filter_supports_control_input(self) -> bool:
        if self._active_filter_id is None:
            return False
        record = self._valid_records.get(self._active_filter_id)
        return self._record_supports_control_input(record)

    def process_control(self, control_input: FilterControlInput) -> bool:
        """Feed latest applied vehicle control to the active filter when supported."""
        self._last_control_input = control_input
        self._active_control_input_used = False
        if self._tracking_mode != TRACKING_MODE_ACTIVE or self._active_filter is None:
            return False
        method = getattr(self._active_filter, "process_control", None)
        if method is None:
            self._tracking_mode_message = "Active command input unavailable for this filter."
            return False
        try:
            used = method(control_input)
        except Exception as exc:
            self._runtime_error = f"process_control failed: {exc}"
            return False
        self._active_control_input_used = bool(True if used is None else used)
        return self._active_control_input_used

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
        diagnostics.setdefault("tracking_mode", self._tracking_mode)
        diagnostics.setdefault("tracking_mode_message", self._tracking_mode_message)
        diagnostics.setdefault("active_control_input_supported", self.active_filter_supports_control_input())
        diagnostics.setdefault("active_control_input_used", self._active_control_input_used)
        diagnostics.setdefault("latest_control_input", self._control_input_dict(self._last_control_input))
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

    def _instantiate_filter(self, record: FilterPluginRecord, tracking_mode: str) -> LocalizationFilter:
        assert record.filter_class is not None
        tune = self.get_filter_runtime_tune(record.filter_id)
        kwargs = self._constructor_kwargs(
            record.filter_class,
            tune=tune,
            tracking_mode=tracking_mode,
        )
        return record.filter_class(self._gnss_projector, **kwargs)

    @staticmethod
    def _constructor_kwargs(filter_class: type, **candidate_kwargs: object) -> dict[str, object]:
        try:
            signature = inspect.signature(filter_class)
        except (TypeError, ValueError):
            return dict(candidate_kwargs)
        parameters = signature.parameters
        accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        if accepts_kwargs:
            return dict(candidate_kwargs)
        return {key: value for key, value in candidate_kwargs.items() if key in parameters}

    def _effective_tracking_mode(self, record: FilterPluginRecord) -> str:
        if self._tracking_mode == TRACKING_MODE_ACTIVE and self._record_supports_control_input(record):
            return TRACKING_MODE_ACTIVE
        return TRACKING_MODE_PASSIVE

    @staticmethod
    def _record_supports_control_input(record: Optional[FilterPluginRecord]) -> bool:
        if record is None or record.filter_class is None:
            return False
        if record.filter_info.get("active_tracking_supported") is True:
            return True
        return hasattr(record.filter_class, "process_control")

    def _merged_clamped_tune(self, filter_id: str, values: dict[str, object]) -> dict[str, Any]:
        record = self._valid_records[filter_id]
        merged = dict(self._runtime_tunes.get(filter_id, record.tune))
        for key, value in values.items():
            if key in record.tune or any(getattr(spec, "key", None) == key for spec in record.tune_specs):
                merged[key] = value
        return self._clamped_tune_values(record, merged)

    @staticmethod
    def _clamped_tune_values(record: FilterPluginRecord, values: dict[str, object]) -> dict[str, Any]:
        result = dict(record.tune)
        specs_by_key = {getattr(spec, "key", ""): spec for spec in record.tune_specs}
        keys = set(result) | set(specs_by_key)
        for key in keys:
            current = values.get(key, result.get(key))
            spec = specs_by_key.get(key)
            if spec is not None:
                current = spec.clamp(current)
            default = record.tune.get(key)
            if isinstance(default, bool):
                result[key] = bool(float(current) >= 0.5)
            elif isinstance(default, int) and not isinstance(default, bool):
                result[key] = int(round(float(current)))
            else:
                try:
                    number = float(current)
                except (TypeError, ValueError):
                    number = float(default) if isinstance(default, (int, float)) else 0.0
                result[key] = number
        return result

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
