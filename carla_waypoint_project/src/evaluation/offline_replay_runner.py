"""Offline replay runner for identical GNSS/IMU sensor logs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import inspect
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Optional

from config.settings import BENCHMARK
from src.KalmanLab.filter_base import TRACKING_MODE_PASSIVE
from src.KalmanLab.registry import discover_filters
from src.core.vehicle_state import VehicleState
from src.evaluation.benchmark_config import project_commit_hash
from src.evaluation.evaluation_artifacts import (
    OFFLINE_LOCALIZATION_EXPLANATION,
    OFFLINE_MODE_NAME,
    OFFLINE_REPORT_NAME,
    evaluations_root,
    read_json,
    slugify,
    timestamp_id,
    unique_folder,
    write_json,
)
from src.evaluation.localization_metrics import (
    compute_localization_metrics,
    nees_xy,
    position_error,
    yaw_error_deg,
)
from src.evaluation.offline_plots import generate_aggregate_rmse_plot, generate_offline_route_plots
from src.localization.gnss_projection import LocalGnssMeasurement


ESTIMATE_FIELDNAMES = [
    "timestamp",
    "sample_timestamp",
    "metric_timestamp",
    "sensor_time_offset_s",
    "ground_truth_alignment",
    "frame",
    "dt",
    "phase",
    "valid_for_metrics",
    "seconds_since_teleport",
    "seconds_since_recording_start",
    "seconds_since_replay_start",
    "fresh_gnss_after_teleport_count",
    "fresh_imu_after_teleport_count",
    "teleport_frame",
    "warmup_excluded_reason",
    "filter_id",
    "ground_truth_x",
    "ground_truth_y",
    "ground_truth_z",
    "ground_truth_yaw",
    "ground_truth_speed",
    "ground_truth_vx_mps",
    "ground_truth_vy_mps",
    "sample_ground_truth_x",
    "sample_ground_truth_y",
    "sample_ground_truth_yaw",
    "sample_ground_truth_speed",
    "estimate_x",
    "estimate_y",
    "estimate_z",
    "estimate_yaw",
    "estimate_speed",
    "estimate_vx_mps",
    "estimate_vy_mps",
    "position_error_m",
    "x_error_m",
    "y_error_m",
    "yaw_error_deg",
    "speed_error_mps",
    "velocity_error_mps",
    "nis",
    "nees",
]

AUTO_TUNE_REPLAY_CONTEXT = "auto_tune_trial"
WINDOWS_PATH_LENGTH_GUARD = 240


@dataclass(frozen=True)
class OfflineReplayRequest:
    """Configuration for one offline localization evaluation run."""

    sensor_log_paths: tuple[Path, ...]
    selected_filter_ids: tuple[str, ...]
    filter_tunes: dict[str, dict[str, object]] = field(default_factory=dict)
    output_root: str = "benchmark_results"
    include_raw_gnss_baseline: bool = True
    initial_condition_policy: str = "first_valid_gnss_initializes_each_filter"
    run_folder_override: Optional[Path] = None
    generate_plots: bool = True
    replay_context: str = "normal_evaluation"


@dataclass(frozen=True)
class OfflineReplayResult:
    """Summary returned after an offline replay run."""

    output_folder: Path
    route_count: int
    selected_filters: tuple[str, ...]
    best_filter_id: Optional[str]
    best_position_rmse_m: Optional[float]
    raw_gnss_rmse_m: Optional[float]
    warmup_excluded_s: float
    failures: tuple[dict[str, object], ...]


class ReplayGnssProjector:
    """Project replay GNSS objects by using logged local coordinates."""

    available = True
    projection_error = None

    def project(self, gnss: object) -> Optional[LocalGnssMeasurement]:
        local_x = _optional_float(getattr(gnss, "local_x", None))
        local_y = _optional_float(getattr(gnss, "local_y", None))
        if local_x is None or local_y is None:
            return None
        return LocalGnssMeasurement(
            x=local_x,
            y=local_y,
            z=_optional_float(getattr(gnss, "local_z", None)) or 0.0,
            latitude=_optional_float(getattr(gnss, "latitude", None)) or 0.0,
            longitude=_optional_float(getattr(gnss, "longitude", None)) or 0.0,
            altitude=_optional_float(getattr(gnss, "altitude", None)) or 0.0,
            frame=int(_optional_float(getattr(gnss, "frame", None)) or 0),
            timestamp=_optional_float(getattr(gnss, "timestamp", None)) or 0.0,
        )


class OfflineReplayRunner:
    """Evaluate multiple filters against the exact same saved sensor logs."""

    def __init__(self) -> None:
        self._records = {record.filter_id: record for record in discover_filters() if record.valid}

    def run(self, request: OfflineReplayRequest) -> OfflineReplayResult:
        if not request.sensor_log_paths:
            raise ValueError("Select at least one recorded sensor log.")

        filter_ids = list(dict.fromkeys(str(item) for item in request.selected_filter_ids if str(item).strip()))
        if request.include_raw_gnss_baseline and "raw_gnss" not in filter_ids:
            filter_ids.insert(0, "raw_gnss")
        if not filter_ids:
            raise ValueError("Select at least one filter for offline replay.")
        resolved_filter_tunes = self._resolved_filter_tunes(filter_ids, request.filter_tunes)

        if request.run_folder_override is not None:
            run_folder = Path(request.run_folder_override)
        else:
            run_folder = unique_folder(evaluations_root(request.output_root), timestamp_id())
        _validate_windows_path_length(run_folder, "Offline replay output folder")
        run_folder.mkdir(parents=True, exist_ok=False)
        metadata = {
            "mode": OFFLINE_MODE_NAME,
            "report_name": OFFLINE_REPORT_NAME,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "replay_context": request.replay_context,
            "output_folder": str(run_folder),
            "selected_filters": filter_ids,
            "filter_tunes": resolved_filter_tunes,
            "sensor_log_paths": [str(path) for path in request.sensor_log_paths],
            "initial_condition_policy": request.initial_condition_policy,
            "generate_plots": request.generate_plots,
            "replay_mode": "passive_offline_replay",
            "control_commands_used": False,
            "control_command_policy": "Logged throttle/brake/steer are preserved in sensor logs but are not fed to passive replay filters.",
            "warmup_exclusion_policy": (
                "Use recorded valid_for_metrics when present; otherwise exclude the first "
                f"{BENCHMARK.offline_metric_warmup_seconds:.1f}s by timestamp fallback."
            ),
            "divergence_error_threshold_m": BENCHMARK.divergence_error_threshold_m,
            "metric_definitions": {
                "full_position_rmse_m": "Position RMSE over every finite replay sample, including startup diagnostics.",
                "eval_position_rmse_m": "Position RMSE over valid_for_metrics=true samples only. This is the fair comparison metric.",
                "position_rmse_m": "Backward-compatible alias for eval_position_rmse_m.",
                "metric_ground_truth_alignment": "Estimate errors are computed against ground truth interpolated at the sensor/estimate timestamp.",
                "improvement_over_raw_gnss_percent": "100 * (raw_gnss_eval_rmse - filter_eval_rmse) / raw_gnss_eval_rmse",
                "nis": "reported only when a filter exposes innovation statistics",
                "nees": "computed only when a comparable position covariance is exposed",
            },
            "explanation": OFFLINE_LOCALIZATION_EXPLANATION,
            "project_commit": project_commit_hash(),
            "failures": [],
        }
        write_json(run_folder / "metadata.json", metadata)

        aggregate_rows: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        best_filter_id: Optional[str] = None
        best_rmse: Optional[float] = None
        first_raw_rmse: Optional[float] = None
        total_warmup_excluded_s = 0.0

        for route_index, log_path in enumerate(request.sensor_log_paths, start=1):
            route_result = self._run_one_log(
                log_path=Path(log_path),
                run_folder=run_folder,
                route_index=route_index,
                filter_ids=filter_ids,
                filter_tunes=resolved_filter_tunes,
                initial_condition_policy=request.initial_condition_policy,
                replay_context=request.replay_context,
                generate_plots=request.generate_plots,
            )
            failures.extend(route_result["failures"])
            aggregate_rows.extend(route_result["aggregate_rows"])
            total_warmup_excluded_s += float(route_result.get("warmup_excluded_s") or 0.0)
            if first_raw_rmse is None:
                first_raw_rmse = route_result.get("raw_gnss_rmse_m")
            for row in route_result["aggregate_rows"]:
                rmse = _optional_float(row.get("eval_position_rmse_m") or row.get("position_rmse_m"))
                filter_id = str(row.get("filter_id") or "")
                if filter_id == "raw_gnss" or rmse is None:
                    continue
                if best_rmse is None or rmse < best_rmse:
                    best_rmse = rmse
                    best_filter_id = filter_id

        best_filter_id, best_rmse = _best_filter_by_mean_eval_rmse(aggregate_rows)
        _write_csv(run_folder / "aggregate_summary.csv", _aggregate_fieldnames(), aggregate_rows)
        aggregate_summary = {
            "route_count": len(request.sensor_log_paths),
            "selected_filters": filter_ids,
            "best_filter_id": best_filter_id,
            "best_position_rmse_m": best_rmse,
            "raw_gnss_rmse_m": first_raw_rmse,
            "warmup_excluded_s": total_warmup_excluded_s,
            "failures": failures,
            "aggregate_rows": aggregate_rows,
            "explanation": OFFLINE_LOCALIZATION_EXPLANATION,
        }
        write_json(run_folder / "aggregate_summary.json", aggregate_summary)
        if request.generate_plots:
            generate_aggregate_rmse_plot(run_folder)
        metadata["failures"] = failures
        write_json(run_folder / "metadata.json", metadata)

        return OfflineReplayResult(
            output_folder=run_folder,
            route_count=len(request.sensor_log_paths),
            selected_filters=tuple(filter_ids),
            best_filter_id=best_filter_id,
            best_position_rmse_m=best_rmse,
            raw_gnss_rmse_m=first_raw_rmse,
            warmup_excluded_s=total_warmup_excluded_s,
            failures=tuple(failures),
        )

    def _run_one_log(
        self,
        log_path: Path,
        run_folder: Path,
        route_index: int,
        filter_ids: list[str],
        filter_tunes: dict[str, dict[str, object]],
        initial_condition_policy: str,
        replay_context: str,
        generate_plots: bool,
    ) -> dict[str, object]:
        rows, warmup_policy = _annotated_log_rows(log_path)
        if len(rows) < 2:
            failure = {
                "sensor_log_path": str(log_path),
                "reason": "Sensor log is empty or too short.",
            }
            return {"failures": [failure], "aggregate_rows": [], "raw_gnss_rmse_m": None}
        metric_alignment = _annotate_metric_ground_truth(rows)

        source_metadata = _source_metadata(log_path)
        route_name = _route_name(source_metadata, log_path)
        compact = replay_context == AUTO_TUNE_REPLAY_CONTEXT
        route_folder_name = f"r{route_index:03d}" if compact else f"route_{route_index:03d}_{slugify(route_name, 'route')}"
        result_dir_name = "res" if compact else "replay_results"
        metrics_dir_name = "met" if compact else "metrics"
        plots_dir_name = "plt" if compact else "plots"
        route_folder = run_folder / route_folder_name
        result_dir = route_folder / result_dir_name
        metrics_dir = route_folder / metrics_dir_name
        plots_dir = route_folder / plots_dir_name
        _validate_windows_path_length(route_folder, "Offline replay route output folder")
        _validate_windows_path_length(metrics_dir / "summary_metrics.json", "Offline replay metrics file")
        result_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        if generate_plots:
            plots_dir.mkdir(parents=True, exist_ok=True)

        raw_estimates = _raw_gnss_estimates(rows)
        raw_metrics = compute_localization_metrics(
            raw_estimates,
            raw_gnss_rmse_m=None,
            divergence_error_threshold_m=BENCHMARK.divergence_error_threshold_m,
        )
        raw_rmse = _optional_float(raw_metrics.get("eval_position_rmse_m"))
        _write_csv(result_dir / "raw_gnss_estimates.csv", ESTIMATE_FIELDNAMES, raw_estimates)
        write_json(metrics_dir / "raw_gnss_metrics.json", raw_metrics)

        summary_rows = [_summary_row(route_name, "raw_gnss", raw_metrics, replay_runtime_s=0.0)]
        failures: list[dict[str, object]] = []
        route_metadata = {
            "mode": OFFLINE_MODE_NAME,
            "report_name": OFFLINE_REPORT_NAME,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "replay_context": replay_context,
            "sensor_log_path": str(log_path),
            "route_name": route_name,
            "route_folder_name": route_folder.name,
            "map_name": source_metadata.get("map_name"),
            "artifact_layout": {
                "compact": compact,
                "result_dir": result_dir.name,
                "metrics_dir": metrics_dir.name,
                "plots_dir": plots_dir.name if generate_plots else None,
            },
            "selected_filters": filter_ids,
            "filter_tunes": {
                filter_id: dict(filter_tunes.get(filter_id, {}))
                for filter_id in filter_ids
                if filter_id != "raw_gnss"
            },
            "initial_condition_policy": initial_condition_policy,
            "generate_plots": generate_plots,
            "replay_mode": "passive_offline_replay",
            "control_commands_used": False,
            "control_command_policy": "Logged controls are not fed to passive offline replay filters.",
            "sample_count": len(rows),
            "valid_for_metrics_sample_count": sum(1 for row in rows if _bool_value(row.get("valid_for_metrics"), default=False)),
            "warmup_excluded_sample_count": sum(1 for row in rows if not _bool_value(row.get("valid_for_metrics"), default=False)),
            "warmup_excluded_s": raw_metrics.get("warmup_excluded_s"),
            "warmup_exclusion_policy": warmup_policy,
            "metric_ground_truth_alignment": metric_alignment,
            "divergence_error_threshold_m": BENCHMARK.divergence_error_threshold_m,
            "recording_metadata": source_metadata,
            "explanation": OFFLINE_LOCALIZATION_EXPLANATION,
            "teleport_transient_handling": (
                "Warm-up and teleport transient samples are replayed through filters for stabilization, "
                "but eval_* metrics only use valid_for_metrics=true samples."
            ),
            "failures": failures,
        }
        write_json(route_folder / "metadata.json", route_metadata)

        for filter_id in filter_ids:
            if filter_id == "raw_gnss":
                continue
            started = time.perf_counter()
            try:
                estimates, diagnostics = self._replay_filter(filter_id, rows, filter_tunes.get(filter_id, {}))
                runtime_s = time.perf_counter() - started
                metrics = compute_localization_metrics(
                    estimates,
                    raw_gnss_rmse_m=raw_rmse,
                    divergence_error_threshold_m=BENCHMARK.divergence_error_threshold_m,
                )
                metrics["replay_runtime_s"] = runtime_s
                metrics["diagnostics"] = diagnostics
                _write_csv(result_dir / f"{slugify(filter_id, 'filter')}_estimates.csv", ESTIMATE_FIELDNAMES, estimates)
                write_json(metrics_dir / f"{slugify(filter_id, 'filter')}_metrics.json", metrics)
                summary_rows.append(_summary_row(route_name, filter_id, metrics, replay_runtime_s=runtime_s))
            except Exception as exc:
                failure = {
                    "route_name": route_name,
                    "sensor_log_path": str(log_path),
                    "filter_id": filter_id,
                    "reason": str(exc),
                }
                failures.append(failure)
                write_json(metrics_dir / f"{slugify(filter_id, 'filter')}_failure.json", failure)

        _write_csv(metrics_dir / "summary_metrics.csv", _summary_fieldnames(), summary_rows)
        summary_json = {
            "route_name": route_name,
            "sensor_log_path": str(log_path),
            "raw_gnss_rmse_m": raw_rmse,
            "raw_gnss_eval_rmse_m": raw_rmse,
            "warmup_exclusion_policy": warmup_policy,
            "metric_ground_truth_alignment": metric_alignment,
            "warmup_excluded_s": raw_metrics.get("warmup_excluded_s"),
            "rows": summary_rows,
            "failures": failures,
        }
        write_json(metrics_dir / "summary_metrics.json", summary_json)
        route_metadata["failures"] = failures
        write_json(route_folder / "metadata.json", route_metadata)
        if generate_plots:
            generate_offline_route_plots(route_folder)
        aggregate_rows = [
            {"route_index": route_index, **row, "sensor_log_path": str(log_path)}
            for row in summary_rows
        ]
        return {
            "failures": failures,
            "aggregate_rows": aggregate_rows,
            "raw_gnss_rmse_m": raw_rmse,
            "warmup_excluded_s": raw_metrics.get("warmup_excluded_s"),
        }

    def _replay_filter(
        self,
        filter_id: str,
        rows: list[dict[str, object]],
        tune_override: dict[str, object],
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        record = self._records.get(filter_id)
        if record is None or record.filter_class is None:
            raise ValueError(f"Filter is not available: {filter_id}")
        tune = dict(record.tune)
        tune.update(dict(tune_override or {}))
        filter_instance = _instantiate_filter(record.filter_class, ReplayGnssProjector(), tune)
        reset = getattr(filter_instance, "reset", None)
        if reset is not None:
            reset()

        estimates: list[dict[str, object]] = []
        last_gnss_frame: Optional[int] = None
        last_imu_frame: Optional[int] = None
        failures = 0
        last_diagnostics: dict[str, object] = {}
        for row in rows:
            imu = _imu_from_row(row)
            gnss = _gnss_from_row(row)
            try:
                if imu is not None and imu.frame != last_imu_frame:
                    getattr(filter_instance, "process_imu")(imu)
                    last_imu_frame = int(imu.frame)
                if gnss is not None and gnss.frame != last_gnss_frame:
                    getattr(filter_instance, "process_gnss")(gnss)
                    last_gnss_frame = int(gnss.frame)
                state = getattr(filter_instance, "get_state")()
                diagnostics = _filter_diagnostics(filter_instance)
                last_diagnostics = diagnostics
            except Exception:
                failures += 1
                state = None
                diagnostics = {}
            estimates.append(_estimate_row(row, filter_id, state, diagnostics))
        return estimates, {
            "failed_sample_updates": failures,
            "last_gnss_frame": last_gnss_frame,
            "last_imu_frame": last_imu_frame,
            "last_filter_diagnostics": last_diagnostics,
        }

    def _resolved_filter_tunes(
        self,
        filter_ids: list[str],
        tune_overrides: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        resolved: dict[str, dict[str, object]] = {}
        for filter_id in filter_ids:
            if filter_id == "raw_gnss":
                continue
            record = self._records.get(filter_id)
            if record is None:
                resolved[filter_id] = dict(tune_overrides.get(filter_id, {}))
                continue
            tune = dict(record.tune)
            tune.update(dict(tune_overrides.get(filter_id, {})))
            resolved[filter_id] = tune
        return resolved


def _instantiate_filter(filter_class: type, projector: ReplayGnssProjector, tune: dict[str, object]) -> object:
    kwargs = {"tune": tune, "tracking_mode": TRACKING_MODE_PASSIVE}
    try:
        signature = inspect.signature(filter_class)
    except (TypeError, ValueError):
        return filter_class(projector, **kwargs)
    parameters = signature.parameters
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    if not accepts_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return filter_class(projector, **kwargs)


def _raw_gnss_estimates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    estimates: list[dict[str, object]] = []
    previous: Optional[dict[str, object]] = None
    previous_yaw = 0.0
    for row in rows:
        x = _optional_float(row.get("gnss_local_x"))
        y = _optional_float(row.get("gnss_local_y"))
        z = _optional_float(row.get("gnss_local_z"))
        timestamp = _first_float(row.get("gnss_timestamp"), row.get("timestamp"))
        speed = None
        yaw = previous_yaw
        if previous is not None and timestamp is not None and x is not None and y is not None:
            prev_t = _first_float(previous.get("gnss_timestamp"), previous.get("timestamp"))
            prev_x = _optional_float(previous.get("gnss_local_x"))
            prev_y = _optional_float(previous.get("gnss_local_y"))
            if prev_t is not None and prev_x is not None and prev_y is not None:
                dt = timestamp - prev_t
                if dt > 1.0e-6:
                    dx = x - prev_x
                    dy = y - prev_y
                    speed = math.hypot(dx, dy) / dt
                    if speed > 0.35:
                        yaw = _normalize_angle_deg(math.degrees(math.atan2(dy, dx)))
                        previous_yaw = yaw
        state = VehicleState(
            x=x if x is not None else float("nan"),
            y=y if y is not None else float("nan"),
            z=z if z is not None else 0.0,
            yaw=yaw,
            speed=speed if speed is not None else 0.0,
            timestamp=timestamp if timestamp is not None else 0.0,
            source_filter_id="raw_gnss",
            model_type="RAW_GNSS",
            safe_for_autonomous_control=False,
        ) if x is not None and y is not None else None
        estimates.append(_estimate_row(row, "raw_gnss", state, {}))
        if x is not None and y is not None:
            previous = row
    return estimates


def _estimate_row(
    row: dict[str, object],
    filter_id: str,
    state: Optional[VehicleState],
    diagnostics: dict[str, object],
) -> dict[str, object]:
    gt_x = _first_float(row.get("metric_ground_truth_x"), row.get("ground_truth_x"))
    gt_y = _first_float(row.get("metric_ground_truth_y"), row.get("ground_truth_y"))
    gt_z = _first_float(row.get("metric_ground_truth_z"), row.get("ground_truth_z"))
    gt_yaw = _first_float(row.get("metric_ground_truth_yaw"), row.get("ground_truth_yaw"))
    gt_speed = _first_float(row.get("metric_ground_truth_speed"), row.get("ground_truth_speed"))
    gt_vx = _first_float(row.get("metric_ground_truth_vx_mps"), row.get("ground_truth_vx_mps"))
    gt_vy = _first_float(row.get("metric_ground_truth_vy_mps"), row.get("ground_truth_vy_mps"))
    est_x = state.x if state is not None else None
    est_y = state.y if state is not None else None
    x_error = (est_x - gt_x) if est_x is not None and gt_x is not None and math.isfinite(est_x) else None
    y_error = (est_y - gt_y) if est_y is not None and gt_y is not None and math.isfinite(est_y) else None
    speed_error = None
    if state is not None and gt_speed is not None:
        speed_error = state.speed - gt_speed
    velocity_error = None
    if state is not None and state.vx_mps is not None and state.vy_mps is not None and gt_vx is not None and gt_vy is not None:
        velocity_error = math.hypot(state.vx_mps - gt_vx, state.vy_mps - gt_vy)
    covariance = diagnostics.get("covariance_diagonal")
    if covariance is None and state is not None:
        covariance = state.covariance_diagonal
    return {
        "timestamp": row.get("timestamp"),
        "sample_timestamp": row.get("timestamp"),
        "metric_timestamp": row.get("metric_timestamp"),
        "sensor_time_offset_s": row.get("sensor_time_offset_s"),
        "ground_truth_alignment": row.get("ground_truth_alignment"),
        "frame": row.get("frame"),
        "dt": row.get("dt"),
        "phase": row.get("phase"),
        "valid_for_metrics": row.get("valid_for_metrics"),
        "seconds_since_teleport": row.get("seconds_since_teleport"),
        "seconds_since_recording_start": row.get("seconds_since_recording_start"),
        "seconds_since_replay_start": row.get("seconds_since_replay_start"),
        "fresh_gnss_after_teleport_count": row.get("fresh_gnss_after_teleport_count"),
        "fresh_imu_after_teleport_count": row.get("fresh_imu_after_teleport_count"),
        "teleport_frame": row.get("teleport_frame"),
        "warmup_excluded_reason": row.get("warmup_excluded_reason"),
        "filter_id": filter_id,
        "ground_truth_x": gt_x,
        "ground_truth_y": gt_y,
        "ground_truth_z": gt_z,
        "ground_truth_yaw": gt_yaw,
        "ground_truth_speed": gt_speed,
        "ground_truth_vx_mps": gt_vx,
        "ground_truth_vy_mps": gt_vy,
        "sample_ground_truth_x": row.get("ground_truth_x"),
        "sample_ground_truth_y": row.get("ground_truth_y"),
        "sample_ground_truth_yaw": row.get("ground_truth_yaw"),
        "sample_ground_truth_speed": row.get("ground_truth_speed"),
        "estimate_x": est_x,
        "estimate_y": est_y,
        "estimate_z": state.z if state is not None else None,
        "estimate_yaw": state.yaw if state is not None else None,
        "estimate_speed": state.speed if state is not None else None,
        "estimate_vx_mps": state.vx_mps if state is not None else None,
        "estimate_vy_mps": state.vy_mps if state is not None else None,
        "position_error_m": position_error(est_x, est_y, gt_x, gt_y),
        "x_error_m": x_error,
        "y_error_m": y_error,
        "yaw_error_deg": yaw_error_deg(state.yaw if state is not None else None, gt_yaw),
        "speed_error_mps": speed_error,
        "velocity_error_mps": velocity_error,
        "nis": diagnostics.get("nis"),
        "nees": nees_xy(x_error, y_error, covariance),
    }


def _gnss_from_row(row: dict[str, object]) -> Optional[SimpleNamespace]:
    local_x = _optional_float(row.get("gnss_local_x"))
    local_y = _optional_float(row.get("gnss_local_y"))
    if local_x is None or local_y is None:
        return None
    return SimpleNamespace(
        latitude=_optional_float(row.get("gnss_latitude")) or 0.0,
        longitude=_optional_float(row.get("gnss_longitude")) or 0.0,
        altitude=_optional_float(row.get("gnss_altitude")) or 0.0,
        local_x=local_x,
        local_y=local_y,
        local_z=_optional_float(row.get("gnss_local_z")) or 0.0,
        frame=int(_optional_float(row.get("gnss_frame")) or _optional_float(row.get("frame")) or 0),
        timestamp=_optional_float(row.get("gnss_timestamp")) or _optional_float(row.get("timestamp")) or 0.0,
    )


def _imu_from_row(row: dict[str, object]) -> Optional[SimpleNamespace]:
    accel_x = _optional_float(row.get("imu_accel_x"))
    gyro_z = _optional_float(row.get("imu_gyro_z"))
    if accel_x is None and gyro_z is None:
        return None
    return SimpleNamespace(
        accelerometer=(
            accel_x or 0.0,
            _optional_float(row.get("imu_accel_y")) or 0.0,
            _optional_float(row.get("imu_accel_z")) or 0.0,
        ),
        gyroscope=(
            _optional_float(row.get("imu_gyro_x")) or 0.0,
            _optional_float(row.get("imu_gyro_y")) or 0.0,
            gyro_z or 0.0,
        ),
        compass=_optional_float(row.get("imu_compass")) or 0.0,
        frame=int(_optional_float(row.get("imu_frame")) or _optional_float(row.get("frame")) or 0),
        timestamp=_optional_float(row.get("imu_timestamp")) or _optional_float(row.get("timestamp")) or 0.0,
    )


def _filter_diagnostics(filter_instance: object) -> dict[str, object]:
    getter = getattr(filter_instance, "get_diagnostics", None)
    if getter is None:
        return {}
    try:
        diagnostics = getter()
    except Exception:
        return {}
    return diagnostics if isinstance(diagnostics, dict) else {}


def _annotated_log_rows(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = _read_log_rows(path)
    if not rows:
        return rows, {
            "valid_for_metrics_source": "none",
            "warnings": ["Sensor log is empty."],
        }

    valid_values = [str(row.get("valid_for_metrics") or "").strip() for row in rows]
    has_valid_column = "valid_for_metrics" in rows[0]
    has_usable_valid_column = has_valid_column and any(valid_values)
    first_timestamp = next((_optional_float(row.get("timestamp")) for row in rows if _optional_float(row.get("timestamp")) is not None), 0.0)
    first_timestamp = first_timestamp or 0.0
    warnings: list[str] = []
    if not has_usable_valid_column:
        warnings.append("valid_for_metrics column missing; timestamp-based warm-up fallback was used.")

    annotated: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        timestamp = _optional_float(item.get("timestamp"))
        replay_seconds = _first_float(
            item.get("seconds_since_recording_start"),
            item.get("seconds_since_teleport"),
        )
        if replay_seconds is None and timestamp is not None:
            replay_seconds = max(0.0, timestamp - first_timestamp)
        if replay_seconds is None:
            replay_seconds = 0.0
        item["seconds_since_replay_start"] = replay_seconds

        if has_usable_valid_column:
            valid = _bool_value(item.get("valid_for_metrics"), default=False)
            reason = str(item.get("warmup_excluded_reason") or "")
        else:
            valid = replay_seconds >= float(BENCHMARK.offline_metric_warmup_seconds)
            reason = "" if valid else "offline_metric_warmup_fallback"
            item["phase"] = item.get("phase") or ("EVALUATION_ACTIVE" if valid else "OFFLINE_METRIC_WARMUP")
            item["warmup_excluded_reason"] = reason
        item["valid_for_metrics"] = valid
        if not valid and not str(item.get("warmup_excluded_reason") or ""):
            item["warmup_excluded_reason"] = reason or "warmup_excluded"
        annotated.append(item)

    return annotated, {
        "valid_for_metrics_source": "sensor_log" if has_usable_valid_column else "timestamp_fallback",
        "offline_metric_warmup_seconds": BENCHMARK.offline_metric_warmup_seconds,
        "warnings": warnings,
    }


def _annotate_metric_ground_truth(rows: list[dict[str, object]]) -> dict[str, object]:
    timeline = _ground_truth_timeline(rows)
    offsets: list[float] = []
    for row in rows:
        sample_timestamp = _optional_float(row.get("timestamp"))
        metric_timestamp = _first_float(row.get("gnss_timestamp"), row.get("imu_timestamp"), row.get("timestamp"))
        if metric_timestamp is None:
            metric_timestamp = sample_timestamp
        if metric_timestamp is not None:
            row["metric_timestamp"] = metric_timestamp
        if sample_timestamp is not None and metric_timestamp is not None:
            offset = max(0.0, sample_timestamp - metric_timestamp)
            row["sensor_time_offset_s"] = offset
            offsets.append(offset)
        metric_gt = _interpolated_ground_truth(timeline, metric_timestamp)
        if metric_gt is None:
            row["ground_truth_alignment"] = "sample_timestamp_fallback"
            continue
        row["ground_truth_alignment"] = "sensor_timestamp_interpolated"
        for key, value in metric_gt.items():
            row[f"metric_ground_truth_{key}"] = value
    return {
        "policy": "errors_compare_estimates_to_ground_truth_interpolated_at_sensor_timestamp",
        "mean_sensor_time_offset_s": _mean(offsets),
        "max_sensor_time_offset_s": max(offsets) if offsets else None,
        "sample_count": len(rows),
    }


def _ground_truth_timeline(rows: list[dict[str, object]]) -> list[dict[str, float]]:
    timeline: list[dict[str, float]] = []
    for row in rows:
        timestamp = _optional_float(row.get("timestamp"))
        x = _optional_float(row.get("ground_truth_x"))
        y = _optional_float(row.get("ground_truth_y"))
        if timestamp is None or x is None or y is None:
            continue
        item = {
            "timestamp": timestamp,
            "x": x,
            "y": y,
        }
        for source_key, target_key in (
            ("ground_truth_z", "z"),
            ("ground_truth_yaw", "yaw"),
            ("ground_truth_speed", "speed"),
            ("ground_truth_vx_mps", "vx_mps"),
            ("ground_truth_vy_mps", "vy_mps"),
        ):
            value = _optional_float(row.get(source_key))
            if value is not None:
                item[target_key] = value
        timeline.append(item)
    return timeline


def _interpolated_ground_truth(
    timeline: list[dict[str, float]],
    timestamp: Optional[float],
) -> Optional[dict[str, float]]:
    if not timeline or timestamp is None:
        return None
    if timestamp <= timeline[0]["timestamp"]:
        return dict(timeline[0])
    if timestamp >= timeline[-1]["timestamp"]:
        return dict(timeline[-1])
    left_index = 0
    right_index = len(timeline) - 1
    while left_index + 1 < right_index:
        mid = (left_index + right_index) // 2
        if timeline[mid]["timestamp"] <= timestamp:
            left_index = mid
        else:
            right_index = mid
    left = timeline[left_index]
    right = timeline[right_index]
    t0 = left["timestamp"]
    t1 = right["timestamp"]
    if t1 <= t0:
        return dict(left)
    alpha = max(0.0, min(1.0, (timestamp - t0) / (t1 - t0)))
    result = {"timestamp": timestamp}
    for key in ("x", "y", "z", "speed", "vx_mps", "vy_mps"):
        if key in left and key in right:
            result[key] = float(left[key] + alpha * (right[key] - left[key]))
    if "yaw" in left and "yaw" in right:
        result["yaw"] = _normalize_angle_deg(left["yaw"] + alpha * _normalize_angle_deg(right["yaw"] - left["yaw"]))
    return result


def _read_log_rows(path: Path) -> list[dict[str, object]]:
    with Path(path).open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _source_metadata(log_path: Path) -> dict[str, object]:
    route_metadata = read_json(log_path.parent / "route_metadata.json")
    summary = read_json(log_path.parent / "recording_summary.json")
    merged = dict(route_metadata)
    merged.update({f"recording_summary_{key}": value for key, value in summary.items()})
    if "map_name" not in merged:
        merged["map_name"] = summary.get("map_name")
    return merged


def _route_name(metadata: dict[str, object], log_path: Path) -> str:
    route = metadata.get("route")
    if isinstance(route, dict) and route.get("name"):
        return str(route["name"])
    value = metadata.get("recording_summary_route_name")
    return str(value or log_path.parent.name)


def _summary_row(
    route_name: str,
    filter_id: str,
    metrics: dict[str, object],
    replay_runtime_s: float,
) -> dict[str, object]:
    return {
        "route_name": route_name,
        "filter_id": filter_id,
        "full_position_rmse_m": metrics.get("full_position_rmse_m"),
        "full_position_mae_m": metrics.get("full_position_mae_m"),
        "full_mean_position_error_m": metrics.get("full_mean_position_error_m"),
        "full_max_position_error_m": metrics.get("full_max_position_error_m"),
        "full_final_position_error_m": metrics.get("full_final_position_error_m"),
        "eval_position_rmse_m": metrics.get("eval_position_rmse_m"),
        "eval_position_mae_m": metrics.get("eval_position_mae_m"),
        "eval_mean_position_error_m": metrics.get("eval_mean_position_error_m"),
        "eval_max_position_error_m": metrics.get("eval_max_position_error_m"),
        "eval_final_position_error_m": metrics.get("eval_final_position_error_m"),
        "eval_position_error_std_m": metrics.get("eval_position_error_std_m"),
        "position_rmse_m": metrics.get("position_rmse_m"),
        "position_mae_m": metrics.get("position_mae_m"),
        "mean_position_error_m": metrics.get("mean_position_error_m"),
        "position_error_std_m": metrics.get("position_error_std_m"),
        "max_position_error_m": metrics.get("max_position_error_m"),
        "final_position_error_m": metrics.get("final_position_error_m"),
        "median_position_error_m": metrics.get("median_position_error_m"),
        "p95_position_error_m": metrics.get("p95_position_error_m"),
        "p99_position_error_m": metrics.get("p99_position_error_m"),
        "divergence_error_threshold_m": metrics.get("divergence_error_threshold_m"),
        "divergence_event_count": metrics.get("divergence_event_count"),
        "divergence_duration_s": metrics.get("divergence_duration_s"),
        "improvement_over_raw_gnss_percent": metrics.get("improvement_over_raw_gnss_percent"),
        "valid_estimate_count": metrics.get("valid_estimate_count"),
        "missing_or_invalid_estimate_count": metrics.get("missing_or_invalid_estimate_count"),
        "valid_for_metrics_sample_count": metrics.get("valid_for_metrics_sample_count"),
        "warmup_excluded_sample_count": metrics.get("warmup_excluded_sample_count"),
        "warmup_excluded_s": metrics.get("warmup_excluded_s"),
        "total_sample_count": metrics.get("total_sample_count"),
        "yaw_rmse_deg": metrics.get("yaw_rmse_deg"),
        "speed_rmse_mps": metrics.get("speed_rmse_mps"),
        "velocity_rmse_mps": metrics.get("velocity_rmse_mps"),
        "mean_nis": metrics.get("mean_nis"),
        "mean_nees": metrics.get("mean_nees"),
        "nis_available": metrics.get("nis_available"),
        "nees_available": metrics.get("nees_available"),
        "replay_runtime_s": replay_runtime_s,
    }


def _summary_fieldnames() -> list[str]:
    return [
        "route_name",
        "filter_id",
        "full_position_rmse_m",
        "full_position_mae_m",
        "full_mean_position_error_m",
        "full_max_position_error_m",
        "full_final_position_error_m",
        "eval_position_rmse_m",
        "eval_position_mae_m",
        "eval_mean_position_error_m",
        "eval_max_position_error_m",
        "eval_final_position_error_m",
        "eval_position_error_std_m",
        "position_rmse_m",
        "position_mae_m",
        "mean_position_error_m",
        "position_error_std_m",
        "max_position_error_m",
        "final_position_error_m",
        "median_position_error_m",
        "p95_position_error_m",
        "p99_position_error_m",
        "divergence_error_threshold_m",
        "divergence_event_count",
        "divergence_duration_s",
        "improvement_over_raw_gnss_percent",
        "valid_estimate_count",
        "missing_or_invalid_estimate_count",
        "valid_for_metrics_sample_count",
        "warmup_excluded_sample_count",
        "warmup_excluded_s",
        "total_sample_count",
        "yaw_rmse_deg",
        "speed_rmse_mps",
        "velocity_rmse_mps",
        "mean_nis",
        "mean_nees",
        "nis_available",
        "nees_available",
        "replay_runtime_s",
    ]


def _aggregate_fieldnames() -> list[str]:
    return ["route_index", *_summary_fieldnames(), "sensor_log_path"]


def _best_filter_by_mean_eval_rmse(rows: list[dict[str, object]]) -> tuple[Optional[str], Optional[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        filter_id = str(row.get("filter_id") or "")
        if not filter_id or filter_id == "raw_gnss":
            continue
        rmse = _optional_float(row.get("eval_position_rmse_m") or row.get("position_rmse_m"))
        if rmse is not None:
            grouped.setdefault(filter_id, []).append(rmse)
    best_filter_id = None
    best_rmse = None
    for filter_id, values in grouped.items():
        if not values:
            continue
        mean_rmse = sum(values) / len(values)
        if best_rmse is None or mean_rmse < best_rmse:
            best_filter_id = filter_id
            best_rmse = mean_rmse
    return best_filter_id, best_rmse


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return value


def _optional_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validate_windows_path_length(path: Path, label: str) -> None:
    if os.name != "nt":
        return
    absolute = str(Path(path).resolve())
    if len(absolute) < WINDOWS_PATH_LENGTH_GUARD:
        return
    raise ValueError(
        f"{label} is likely too long for Windows ({len(absolute)} characters): {absolute}. "
        "Auto-tune trials should use compact output paths; if this still occurs, move the project "
        "or benchmark output root to a shorter directory."
    )


def _first_float(*values: object) -> Optional[float]:
    for value in values:
        number = _optional_float(value)
        if number is not None:
            return number
    return None


def _bool_value(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return default


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _normalize_angle_deg(angle_deg: float) -> float:
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg
