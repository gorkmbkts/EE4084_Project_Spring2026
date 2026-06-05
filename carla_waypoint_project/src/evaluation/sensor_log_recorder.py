"""Physical CARLA sensor-log recording for offline localization replay."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import csv
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from pathlib import Path
import time
from typing import Optional

from config.settings import BENCHMARK
from src.core.vehicle_state import VehicleState
from src.evaluation.benchmark_config import SensorNoiseConfig, project_commit_hash
from src.evaluation.evaluation_artifacts import (
    OFFLINE_LOCALIZATION_EXPLANATION,
    OFFLINE_MODE_NAME,
    OFFLINE_REPORT_NAME,
    RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
    recordings_root,
    slugify,
    timestamp_id,
    unique_folder,
    write_json,
)
from src.evaluation.test_route_store import SavedTestRoute, TestRouteStore
from src.localization.gnss_projection import GnssLocalProjector
from src.utils.carla_import import ensure_carla_import
from src.utils.map_names import display_map_name, maps_compatible, normalize_map_name

carla = ensure_carla_import()


SENSOR_LOG_FIELDNAMES = [
    "timestamp",
    "frame",
    "dt",
    "map_name",
    "route_name",
    "route_index",
    "recording_driver",
    "phase",
    "valid_for_metrics",
    "seconds_since_teleport",
    "seconds_since_recording_start",
    "fresh_gnss_after_teleport_count",
    "fresh_imu_after_teleport_count",
    "teleport_frame",
    "warmup_excluded_reason",
    "ground_truth_x",
    "ground_truth_y",
    "ground_truth_z",
    "ground_truth_yaw",
    "ground_truth_speed",
    "ground_truth_vx_mps",
    "ground_truth_vy_mps",
    "ground_truth_ax_mps2",
    "ground_truth_ay_mps2",
    "ground_truth_yaw_rate_radps",
    "gnss_latitude",
    "gnss_longitude",
    "gnss_altitude",
    "gnss_local_x",
    "gnss_local_y",
    "gnss_local_z",
    "gnss_frame",
    "gnss_timestamp",
    "imu_accel_x",
    "imu_accel_y",
    "imu_accel_z",
    "imu_gyro_x",
    "imu_gyro_y",
    "imu_gyro_z",
    "imu_compass",
    "imu_frame",
    "imu_timestamp",
    "control_throttle",
    "control_brake",
    "control_steer",
    "control_hand_brake",
    "control_reverse",
]


PHASE_TELEPORT_SETTLING = "TELEPORT_SETTLING"
PHASE_SENSOR_WARMUP = "SENSOR_WARMUP"
PHASE_FILTER_WARMUP = "FILTER_WARMUP"
PHASE_EVALUATION_ACTIVE = "EVALUATION_ACTIVE"

TELEPORT_TRANSIENT_EXPLANATION = (
    "Saved route tests begin by relocating the ego vehicle to the route start. "
    "This teleportation creates non-physical transient samples in CARLA physics "
    "and GNSS/IMU sensors. These samples are kept for diagnostics but excluded "
    "from reported evaluation metrics using valid_for_metrics=false."
)


class SensorLogRecorderState(Enum):
    IDLE = "IDLE"
    INITIALIZING_ROUTE = "INITIALIZING_ROUTE"
    RUNNING_ROUTE = "RUNNING_ROUTE"
    SWITCHING_MAP = "SWITCHING_MAP"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


@dataclass
class OfflineRecordingConfig:
    """Configuration for recording reusable offline replay sensor logs."""

    selected_routes: tuple[SavedTestRoute, ...]
    sensor_noise_config: SensorNoiseConfig
    vehicle_behavior_config: dict[str, object]
    sensor_noise_preset: str = "Medium Noise"
    vehicle_behavior_preset: str = "Balanced"
    random_seed: int = 4084
    output_root: str = "benchmark_results"
    run_id: str = ""
    created_at: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = timestamp_id()
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_routes": [route.to_dict() for route in self.selected_routes],
            "sensor_noise_config": self.sensor_noise_config.to_dict(),
            "vehicle_behavior_config": dict(self.vehicle_behavior_config),
            "sensor_noise_preset": self.sensor_noise_preset,
            "vehicle_behavior_preset": self.vehicle_behavior_preset,
            "random_seed": self.random_seed,
            "output_root": self.output_root,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
            "metadata": dict(self.metadata),
        }


class SensorLogRecorder:
    """Record GNSS/IMU logs while the route is driven from ground-truth state."""

    def __init__(
        self,
        world_map: "carla.Map",
        route_store: TestRouteStore,
        begin_route_callback: Callable[["carla.Waypoint", "carla.Waypoint", Sequence["carla.Waypoint"]], None],
        plan_route_callback: Callable[["carla.Waypoint", "carla.Waypoint"], Sequence["carla.Waypoint"]],
        weather_callback: Callable[[], Optional[dict[str, object]]],
        vehicle_blueprint_callback: Callable[[], Optional[str]],
        selected_map_load_name: Optional[str] = None,
    ) -> None:
        self._world_map = world_map
        self._route_store = route_store
        self._begin_route_callback = begin_route_callback
        self._plan_route_callback = plan_route_callback
        self._weather_callback = weather_callback
        self._vehicle_blueprint_callback = vehicle_blueprint_callback
        self._selected_map_load_name = selected_map_load_name

        self._active = False
        self._route_running = False
        self._state = SensorLogRecorderState.IDLE
        self._routes: list[SavedTestRoute] = []
        self._current_route: Optional[SavedTestRoute] = None
        self._current_route_index = 0
        self._config: Optional[OfflineRecordingConfig] = None
        self._run_folder: Optional[Path] = None
        self._route_folder: Optional[Path] = None
        self._csv_file = None
        self._writer: Optional[csv.DictWriter] = None
        self._sample_count = 0
        self._started_monotonic: Optional[float] = None
        self._route_started_timestamp: Optional[float] = None
        self._last_sample_timestamp: Optional[float] = None
        self._last_ground_truth: Optional[VehicleState] = None
        self._last_exported_summary: Optional[dict[str, object]] = None
        self._route_summaries: list[dict[str, object]] = []
        self._teleport_frame: Optional[int] = None
        self._fresh_gnss_after_teleport_count = 0
        self._fresh_imu_after_teleport_count = 0
        self._last_fresh_gnss_key: Optional[tuple[object, object]] = None
        self._last_fresh_imu_key: Optional[tuple[object, object]] = None
        self._valid_for_metrics_sample_count = 0
        self._warmup_excluded_sample_count = 0
        self._warmup_excluded_s = 0.0
        self._phase_counts: dict[str, int] = {}
        self._current_phase = PHASE_TELEPORT_SETTLING
        self._status_text = "Offline recording idle"
        self._last_failure_reason = ""

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def route_running(self) -> bool:
        return self._route_running

    @property
    def state(self) -> SensorLogRecorderState:
        return self._state

    @property
    def status_text(self) -> str:
        return self._status_text

    @property
    def current_route_name(self) -> str:
        return self._current_route.name if self._current_route is not None else ""

    @property
    def current_route_index(self) -> int:
        return self._current_route_index

    @property
    def total_routes(self) -> int:
        return len(self._routes)

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def current_phase(self) -> str:
        return self._current_phase

    @property
    def controller_enabled(self) -> bool:
        return self._route_running and self._current_phase == PHASE_EVALUATION_ACTIVE

    @property
    def warmup_excluded_seconds(self) -> float:
        return self._warmup_excluded_s

    @property
    def run_folder(self) -> Optional[Path]:
        return self._run_folder

    @property
    def last_exported_summary(self) -> Optional[dict[str, object]]:
        return self._last_exported_summary

    @property
    def last_failure_reason(self) -> str:
        return self._last_failure_reason

    def update_world_context(
        self,
        world_map: "carla.Map",
        route_store: TestRouteStore,
        selected_map_load_name: Optional[str] = None,
    ) -> None:
        self._world_map = world_map
        self._route_store = route_store
        self._selected_map_load_name = selected_map_load_name

    def start_recording(self, config: OfflineRecordingConfig, active_map_name: Optional[str]) -> bool:
        if self._active:
            self._status_text = "Offline recording already running"
            return False
        if not config.selected_routes:
            self._status_text = "Offline recording blocked: no selected routes"
            return False

        root = recordings_root(config.output_root)
        run_folder = unique_folder(root, config.run_id)
        run_folder.mkdir(parents=True, exist_ok=False)
        config.run_id = run_folder.name
        run_metadata = {
            **config.to_dict(),
            "mode": OFFLINE_MODE_NAME,
            "report_name": OFFLINE_REPORT_NAME,
            "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
            "warmup_config": _warmup_config_dict(),
            "teleport_transient_handling": TELEPORT_TRANSIENT_EXPLANATION,
            "project_commit": project_commit_hash(),
            "explanation": OFFLINE_LOCALIZATION_EXPLANATION,
        }
        write_json(run_folder / "metadata.json", run_metadata)

        self._active = True
        self._route_running = False
        self._state = SensorLogRecorderState.INITIALIZING_ROUTE
        self._routes = list(config.selected_routes)
        self._current_route = None
        self._current_route_index = 0
        self._config = config
        self._run_folder = run_folder
        self._route_folder = None
        self._sample_count = 0
        self._route_summaries = []
        self._last_exported_summary = None
        self._last_failure_reason = ""
        self._reset_route_counters()
        self._status_text = f"Offline recording ready: {len(self._routes)} route(s)"
        return self.begin_current_route(active_map_name)

    def begin_current_route(self, active_map_name: Optional[str]) -> bool:
        route = self._pending_route()
        if not self._active or route is None:
            self._finalize()
            return False
        if not maps_compatible(active_map_name, route.map_name):
            self._state = SensorLogRecorderState.SWITCHING_MAP
            self._status_text = f"Switching map for recording: {display_map_name(route.map_name)}"
            return False
        return self._start_route(route, self._current_route_index)

    def needs_map_switch(self, active_map_name: Optional[str]) -> bool:
        if not self._active or self._route_running:
            return False
        route = self._pending_route()
        return route is not None and not maps_compatible(active_map_name, route.map_name)

    def required_map_name(self) -> Optional[str]:
        route = self._pending_route()
        return route.map_name if route is not None else None

    def stop(self, aborted: bool = True, reason: str = "Offline recording stopped") -> Optional[Path]:
        if not self._active:
            self._status_text = reason
            return self._run_folder
        if self._route_running:
            self._finish_current_route(aborted=aborted, reason=reason)
        self._record_incomplete_pending_routes(reason)
        self._finalize()
        self._state = SensorLogRecorderState.ERROR if aborted else SensorLogRecorderState.FINISHED
        self._status_text = reason
        return self._run_folder

    def update(
        self,
        route_completed: bool,
        route_failed: bool,
        active_map_name: Optional[str],
        ground_truth_state: Optional[VehicleState],
        gnss_measurement: object | None,
        imu_measurement: object | None,
        gnss_projector: Optional[GnssLocalProjector],
        applied_control: object | None,
        frame_index: Optional[int] = None,
        failure_reason: Optional[str] = None,
    ) -> Optional[Path]:
        if not self._active:
            return None
        if not self._route_running:
            if self.needs_map_switch(active_map_name):
                self._state = SensorLogRecorderState.SWITCHING_MAP
                return None
            self.begin_current_route(active_map_name)
            return None
        if route_failed:
            self._finish_current_route(aborted=True, reason=failure_reason or "Route unavailable before completion")
            self._advance(active_map_name)
            return self._run_folder
        self._collect_sample(
            ground_truth_state=ground_truth_state,
            gnss_measurement=gnss_measurement,
            imu_measurement=imu_measurement,
            gnss_projector=gnss_projector,
            applied_control=applied_control,
            frame_index=frame_index,
            active_map_name=active_map_name,
        )
        if self._timed_out():
            self._finish_current_route(aborted=True, reason=f"Route timeout after {BENCHMARK.max_pass_duration_s:.0f}s")
            self._advance(active_map_name)
            return self._run_folder
        if route_completed:
            self._finish_current_route(aborted=False, reason="Recording route completed")
            self._advance(active_map_name)
            return self._run_folder
        return None

    def elapsed_route_seconds(self) -> float:
        if self._started_monotonic is None:
            return 0.0
        return max(0.0, time.monotonic() - self._started_monotonic)

    def _start_route(self, route: SavedTestRoute, route_index: int) -> bool:
        active_map_name = getattr(self._world_map, "name", None)
        if not self._route_store.route_is_compatible(route):
            self._record_route_error(route, "Route map is incompatible with active map")
            self._advance(active_map_name)
            return False
        resolved = self._route_store.resolve_route_to_waypoints(self._world_map, route)
        if resolved is None:
            self._record_route_error(route, "Failed to resolve saved test route")
            self._advance(active_map_name)
            return False
        start_waypoint, goal_waypoint = resolved
        route_waypoints = list(self._plan_route_callback(start_waypoint, goal_waypoint))
        if not route_waypoints:
            self._record_route_error(route, "Route planner returned empty route")
            self._advance(active_map_name)
            return False

        assert self._run_folder is not None
        route_folder = self._run_folder / f"route_{route_index + 1:03d}_{slugify(route.name, 'route')}"
        route_folder.mkdir(parents=True, exist_ok=True)
        route_metadata = self._route_metadata(route, route_index, active_map_name, route_waypoints)
        write_json(route_folder / "route_metadata.json", route_metadata)

        self._close_writer()
        csv_file = (route_folder / "sensor_log.csv").open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=SENSOR_LOG_FIELDNAMES)
        writer.writeheader()
        self._csv_file = csv_file
        self._writer = writer

        self._current_route = route
        self._current_route_index = route_index
        self._route_folder = route_folder
        self._sample_count = 0
        self._started_monotonic = time.monotonic()
        self._route_started_timestamp = None
        self._last_sample_timestamp = None
        self._last_ground_truth = None
        self._reset_route_counters()
        self._route_running = True
        self._state = SensorLogRecorderState.RUNNING_ROUTE
        self._status_text = f"Recording sensor log: {route.name} ({PHASE_TELEPORT_SETTLING})"
        self._begin_route_callback(start_waypoint, goal_waypoint, route_waypoints)
        return True

    def _collect_sample(
        self,
        ground_truth_state: Optional[VehicleState],
        gnss_measurement: object | None,
        imu_measurement: object | None,
        gnss_projector: Optional[GnssLocalProjector],
        applied_control: object | None,
        frame_index: Optional[int],
        active_map_name: Optional[str],
    ) -> None:
        if self._writer is None or self._current_route is None or ground_truth_state is None:
            return
        timestamp = float(ground_truth_state.timestamp)
        if self._route_started_timestamp is None:
            self._route_started_timestamp = timestamp
        if self._teleport_frame is None and frame_index is not None:
            self._teleport_frame = int(frame_index)
        dt = 0.0 if self._last_sample_timestamp is None else max(0.0, timestamp - self._last_sample_timestamp)
        self._last_sample_timestamp = timestamp
        seconds_since_teleport = max(0.0, timestamp - self._route_started_timestamp)
        seconds_since_recording_start = seconds_since_teleport
        self._update_fresh_sensor_counts(gnss_measurement, imu_measurement)
        phase, valid_for_metrics, warmup_reason = self._sample_phase(seconds_since_teleport)
        self._current_phase = phase
        self._phase_counts[phase] = self._phase_counts.get(phase, 0) + 1
        if valid_for_metrics:
            self._valid_for_metrics_sample_count += 1
        else:
            self._warmup_excluded_sample_count += 1
            self._warmup_excluded_s += dt
        gt_ax, gt_ay, gt_yaw_rate = self._derived_ground_truth_motion(ground_truth_state, dt)
        local = gnss_projector.project(gnss_measurement) if gnss_projector is not None and gnss_measurement is not None else None
        imu_accel = getattr(imu_measurement, "accelerometer", (None, None, None)) if imu_measurement is not None else (None, None, None)
        imu_gyro = getattr(imu_measurement, "gyroscope", (None, None, None)) if imu_measurement is not None else (None, None, None)
        row = {
            "timestamp": timestamp,
            "frame": frame_index if frame_index is not None else self._sample_count,
            "dt": dt,
            "map_name": active_map_name or getattr(self._world_map, "name", None),
            "route_name": self._current_route.name,
            "route_index": self._current_route_index + 1,
            "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
            "phase": phase,
            "valid_for_metrics": valid_for_metrics,
            "seconds_since_teleport": seconds_since_teleport,
            "seconds_since_recording_start": seconds_since_recording_start,
            "fresh_gnss_after_teleport_count": self._fresh_gnss_after_teleport_count,
            "fresh_imu_after_teleport_count": self._fresh_imu_after_teleport_count,
            "teleport_frame": self._teleport_frame,
            "warmup_excluded_reason": warmup_reason,
            "ground_truth_x": ground_truth_state.x,
            "ground_truth_y": ground_truth_state.y,
            "ground_truth_z": ground_truth_state.z,
            "ground_truth_yaw": ground_truth_state.yaw,
            "ground_truth_speed": ground_truth_state.speed,
            "ground_truth_vx_mps": ground_truth_state.vx_mps,
            "ground_truth_vy_mps": ground_truth_state.vy_mps,
            "ground_truth_ax_mps2": gt_ax,
            "ground_truth_ay_mps2": gt_ay,
            "ground_truth_yaw_rate_radps": gt_yaw_rate,
            "gnss_latitude": getattr(gnss_measurement, "latitude", None),
            "gnss_longitude": getattr(gnss_measurement, "longitude", None),
            "gnss_altitude": getattr(gnss_measurement, "altitude", None),
            "gnss_local_x": local.x if local is not None else None,
            "gnss_local_y": local.y if local is not None else None,
            "gnss_local_z": local.z if local is not None else None,
            "gnss_frame": getattr(gnss_measurement, "frame", None),
            "gnss_timestamp": getattr(gnss_measurement, "timestamp", None),
            "imu_accel_x": _tuple_item(imu_accel, 0),
            "imu_accel_y": _tuple_item(imu_accel, 1),
            "imu_accel_z": _tuple_item(imu_accel, 2),
            "imu_gyro_x": _tuple_item(imu_gyro, 0),
            "imu_gyro_y": _tuple_item(imu_gyro, 1),
            "imu_gyro_z": _tuple_item(imu_gyro, 2),
            "imu_compass": getattr(imu_measurement, "compass", None),
            "imu_frame": getattr(imu_measurement, "frame", None),
            "imu_timestamp": getattr(imu_measurement, "timestamp", None),
            "control_throttle": getattr(applied_control, "throttle", None),
            "control_brake": getattr(applied_control, "brake", None),
            "control_steer": getattr(applied_control, "steer", None),
            "control_hand_brake": getattr(applied_control, "hand_brake", None),
            "control_reverse": getattr(applied_control, "reverse", None),
        }
        self._writer.writerow({key: "" if value is None else value for key, value in row.items()})
        self._sample_count += 1
        self._last_ground_truth = ground_truth_state
        self._status_text = f"Recording sensor log: {self._current_route.name} ({phase})"

    def _finish_current_route(self, aborted: bool, reason: str) -> None:
        route = self._current_route
        route_folder = self._route_folder
        self._close_writer()
        if route is None or route_folder is None:
            self._route_running = False
            return
        duration = None
        if self._route_started_timestamp is not None and self._last_sample_timestamp is not None:
            duration = max(0.0, self._last_sample_timestamp - self._route_started_timestamp)
        summary = {
            "route_name": route.name,
            "map_name": getattr(self._world_map, "name", None),
            "route_map_name": route.map_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sample_count": self._sample_count,
            "valid_for_metrics_sample_count": self._valid_for_metrics_sample_count,
            "warmup_excluded_sample_count": self._warmup_excluded_sample_count,
            "warmup_excluded_s": self._warmup_excluded_s,
            "phase_counts": dict(self._phase_counts),
            "duration_s": duration,
            "success": not aborted,
            "failure_reason": reason if aborted else "",
            "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
            "teleport_frame": self._teleport_frame,
            "fresh_gnss_after_teleport_count": self._fresh_gnss_after_teleport_count,
            "fresh_imu_after_teleport_count": self._fresh_imu_after_teleport_count,
            "warmup_config": _warmup_config_dict(),
            "teleport_transient_handling": TELEPORT_TRANSIENT_EXPLANATION,
            "sensor_log_path": str(route_folder / "sensor_log.csv"),
            "route_folder": str(route_folder),
        }
        write_json(route_folder / "recording_summary.json", summary)
        self._route_summaries.append(summary)
        self._last_exported_summary = summary
        self._route_running = False
        self._state = SensorLogRecorderState.INITIALIZING_ROUTE
        self._status_text = f"Recorded {route.name}: {self._sample_count} samples"
        if aborted:
            self._last_failure_reason = reason

    def _advance(self, active_map_name: Optional[str]) -> None:
        self._current_route_index += 1
        if self._current_route_index >= len(self._routes):
            self._finalize()
            return
        self.begin_current_route(active_map_name)

    def _finalize(self) -> None:
        self._close_writer()
        if self._run_folder is not None:
            summary = {
                "mode": OFFLINE_MODE_NAME,
                "report_name": OFFLINE_REPORT_NAME,
                "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
                "route_count": len(self._routes),
                "completed_route_count": sum(1 for item in self._route_summaries if item.get("success")),
                "failed_route_count": sum(1 for item in self._route_summaries if not item.get("success")),
                "valid_for_metrics_sample_count": sum(int(item.get("valid_for_metrics_sample_count") or 0) for item in self._route_summaries),
                "warmup_excluded_sample_count": sum(int(item.get("warmup_excluded_sample_count") or 0) for item in self._route_summaries),
                "warmup_excluded_s": sum(float(item.get("warmup_excluded_s") or 0.0) for item in self._route_summaries),
                "warmup_config": _warmup_config_dict(),
                "route_summaries": self._route_summaries,
                "teleport_transient_handling": TELEPORT_TRANSIENT_EXPLANATION,
                "explanation": OFFLINE_LOCALIZATION_EXPLANATION,
            }
            write_json(self._run_folder / "recording_summary.json", summary)
        self._active = False
        self._route_running = False
        self._state = SensorLogRecorderState.FINISHED
        self._status_text = f"Offline recording finished: {self._run_folder.name if self._run_folder else 'none'}"

    def _record_route_error(self, route: SavedTestRoute, reason: str) -> None:
        if self._run_folder is None:
            return
        route_folder = self._run_folder / f"route_{self._current_route_index + 1:03d}_{slugify(route.name, 'route')}"
        route_folder.mkdir(parents=True, exist_ok=True)
        summary = {
            "route_name": route.name,
            "map_name": getattr(self._world_map, "name", None),
            "route_map_name": route.map_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sample_count": 0,
            "duration_s": None,
            "success": False,
            "failure_reason": reason,
            "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
            "route_folder": str(route_folder),
        }
        write_json(route_folder / "recording_summary.json", summary)
        self._route_summaries.append(summary)
        self._last_exported_summary = summary
        self._last_failure_reason = reason
        self._status_text = reason

    def _record_incomplete_pending_routes(self, reason: str) -> None:
        for index in range(self._current_route_index, len(self._routes)):
            if any(item.get("route_name") == self._routes[index].name for item in self._route_summaries):
                continue
            self._current_route_index = index
            self._record_route_error(self._routes[index], reason)

    def _route_metadata(
        self,
        route: SavedTestRoute,
        route_index: int,
        active_map_name: Optional[str],
        route_waypoints: Sequence["carla.Waypoint"],
    ) -> dict[str, object]:
        config = self._config
        route_length = _route_length(route_waypoints)
        return {
            "mode": OFFLINE_MODE_NAME,
            "report_name": OFFLINE_REPORT_NAME,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "map_name": active_map_name,
            "active_map_id": normalize_map_name(active_map_name),
            "selected_map_load_name": self._selected_map_load_name,
            "route_index": route_index + 1,
            "route_count": len(self._routes),
            "route": route.to_dict(),
            "route_length_m": route_length,
            "route_waypoint_count": len(route_waypoints),
            "route_waypoints": [_waypoint_location_dict(waypoint) for waypoint in route_waypoints],
            "sensor_noise_profile": config.sensor_noise_preset if config is not None else None,
            "sensor_noise_config": config.sensor_noise_config.to_dict() if config is not None else None,
            "vehicle_behavior_profile": config.vehicle_behavior_preset if config is not None else None,
            "vehicle_behavior_config": dict(config.vehicle_behavior_config) if config is not None else None,
            "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
            "weather": self._weather_callback(),
            "vehicle_blueprint": self._vehicle_blueprint_callback(),
            "random_seed": config.random_seed if config is not None else None,
            "teleport_settle_seconds": BENCHMARK.teleport_settle_seconds,
            "sensor_warmup_seconds": BENCHMARK.sensor_warmup_seconds,
            "filter_warmup_seconds": BENCHMARK.filter_warmup_seconds,
            "min_fresh_sensor_frames_after_teleport": BENCHMARK.min_fresh_sensor_frames_after_teleport,
            "warmup_config": _warmup_config_dict(),
            "teleport_transient_handling": TELEPORT_TRANSIENT_EXPLANATION,
            "project_commit": project_commit_hash(),
            "explanation": OFFLINE_LOCALIZATION_EXPLANATION,
            "recording_flow": (
                "Saved route -> GroundTruthStateProvider -> waypoint controller -> physical CARLA vehicle "
                "motion -> noisy GNSS/IMU logging."
            ),
        }

    def _pending_route(self) -> Optional[SavedTestRoute]:
        if self._current_route_index < 0 or self._current_route_index >= len(self._routes):
            return None
        return self._routes[self._current_route_index]

    def _timed_out(self) -> bool:
        if self._started_monotonic is None:
            return False
        return time.monotonic() - self._started_monotonic >= BENCHMARK.max_pass_duration_s

    def _derived_ground_truth_motion(self, state: VehicleState, dt: float) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if self._last_ground_truth is None or dt <= 1.0e-6:
            return None, None, None
        previous = self._last_ground_truth
        ax = None
        ay = None
        if state.vx_mps is not None and state.vy_mps is not None and previous.vx_mps is not None and previous.vy_mps is not None:
            ax = (state.vx_mps - previous.vx_mps) / dt
            ay = (state.vy_mps - previous.vy_mps) / dt
        yaw_delta = math.radians(_normalize_angle_deg(state.yaw - previous.yaw))
        return ax, ay, yaw_delta / dt

    def _reset_route_counters(self) -> None:
        self._teleport_frame = None
        self._fresh_gnss_after_teleport_count = 0
        self._fresh_imu_after_teleport_count = 0
        self._last_fresh_gnss_key = None
        self._last_fresh_imu_key = None
        self._valid_for_metrics_sample_count = 0
        self._warmup_excluded_sample_count = 0
        self._warmup_excluded_s = 0.0
        self._phase_counts = {}
        self._current_phase = PHASE_TELEPORT_SETTLING

    def _update_fresh_sensor_counts(self, gnss_measurement: object | None, imu_measurement: object | None) -> None:
        gnss_key = self._fresh_sensor_key(gnss_measurement)
        if gnss_key is not None and gnss_key != self._last_fresh_gnss_key:
            self._fresh_gnss_after_teleport_count += 1
            self._last_fresh_gnss_key = gnss_key
        imu_key = self._fresh_sensor_key(imu_measurement)
        if imu_key is not None and imu_key != self._last_fresh_imu_key:
            self._fresh_imu_after_teleport_count += 1
            self._last_fresh_imu_key = imu_key

    def _fresh_sensor_key(self, measurement: object | None) -> Optional[tuple[object, object]]:
        if measurement is None:
            return None
        frame = getattr(measurement, "frame", None)
        timestamp = getattr(measurement, "timestamp", None)
        if self._teleport_frame is not None and frame is not None:
            try:
                if int(frame) < self._teleport_frame:
                    return None
            except (TypeError, ValueError):
                pass
        if frame is None and timestamp is None:
            return None
        return frame, timestamp

    def _sample_phase(self, seconds_since_teleport: float) -> tuple[str, bool, str]:
        settle_s = max(0.0, float(BENCHMARK.teleport_settle_seconds))
        sensor_s = max(0.0, float(BENCHMARK.sensor_warmup_seconds))
        filter_s = max(0.0, float(BENCHMARK.filter_warmup_seconds))
        min_frames = max(0, int(BENCHMARK.min_fresh_sensor_frames_after_teleport))
        if seconds_since_teleport < settle_s:
            return PHASE_TELEPORT_SETTLING, False, "teleport_settling"
        if (
            seconds_since_teleport < settle_s + sensor_s
            or self._fresh_gnss_after_teleport_count < min_frames
            or self._fresh_imu_after_teleport_count < min_frames
        ):
            return PHASE_SENSOR_WARMUP, False, "sensor_warmup"
        if seconds_since_teleport < settle_s + sensor_s + filter_s:
            return PHASE_FILTER_WARMUP, False, "filter_warmup"
        return PHASE_EVALUATION_ACTIVE, True, ""

    def _close_writer(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
        self._csv_file = None
        self._writer = None


def _tuple_item(value: object, index: int) -> object:
    if isinstance(value, (tuple, list)) and index < len(value):
        return value[index]
    return None


def _warmup_config_dict() -> dict[str, object]:
    return {
        "teleport_settle_seconds": BENCHMARK.teleport_settle_seconds,
        "sensor_warmup_seconds": BENCHMARK.sensor_warmup_seconds,
        "filter_warmup_seconds": BENCHMARK.filter_warmup_seconds,
        "offline_metric_warmup_seconds": BENCHMARK.offline_metric_warmup_seconds,
        "min_fresh_sensor_frames_after_teleport": BENCHMARK.min_fresh_sensor_frames_after_teleport,
        "max_valid_imu_accel_mps2": BENCHMARK.max_valid_imu_accel_mps2,
        "divergence_error_threshold_m": BENCHMARK.divergence_error_threshold_m,
    }


def _waypoint_location_dict(waypoint: "carla.Waypoint") -> dict[str, float]:
    location = waypoint.transform.location
    return {"x": float(location.x), "y": float(location.y), "z": float(location.z)}


def _route_length(route_waypoints: Sequence["carla.Waypoint"]) -> Optional[float]:
    if len(route_waypoints) < 2:
        return None
    total = 0.0
    for previous, current in zip(route_waypoints, route_waypoints[1:]):
        total += previous.transform.location.distance(current.transform.location)
    return float(total)


def _normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle
