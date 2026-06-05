"""Offline replay runner for identical GNSS/IMU sensor logs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import inspect
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Optional

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
    "frame",
    "filter_id",
    "ground_truth_x",
    "ground_truth_y",
    "ground_truth_z",
    "ground_truth_yaw",
    "ground_truth_speed",
    "ground_truth_vx_mps",
    "ground_truth_vy_mps",
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


@dataclass(frozen=True)
class OfflineReplayRequest:
    """Configuration for one offline localization evaluation run."""

    sensor_log_paths: tuple[Path, ...]
    selected_filter_ids: tuple[str, ...]
    filter_tunes: dict[str, dict[str, object]] = field(default_factory=dict)
    output_root: str = "benchmark_results"
    include_raw_gnss_baseline: bool = True
    initial_condition_policy: str = "first_valid_gnss_initializes_each_filter"


@dataclass(frozen=True)
class OfflineReplayResult:
    """Summary returned after an offline replay run."""

    output_folder: Path
    route_count: int
    selected_filters: tuple[str, ...]
    best_filter_id: Optional[str]
    best_position_rmse_m: Optional[float]
    raw_gnss_rmse_m: Optional[float]
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

        run_folder = unique_folder(evaluations_root(request.output_root), timestamp_id())
        run_folder.mkdir(parents=True, exist_ok=False)
        metadata = {
            "mode": OFFLINE_MODE_NAME,
            "report_name": OFFLINE_REPORT_NAME,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "selected_filters": filter_ids,
            "sensor_log_paths": [str(path) for path in request.sensor_log_paths],
            "initial_condition_policy": request.initial_condition_policy,
            "metric_definitions": {
                "position_rmse_m": "sqrt(mean((estimated_xy - ground_truth_xy)^2))",
                "position_mae_m": "mean absolute horizontal position error",
                "improvement_over_raw_gnss_percent": "100 * (raw_gnss_rmse - filter_rmse) / raw_gnss_rmse",
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

        for route_index, log_path in enumerate(request.sensor_log_paths, start=1):
            route_result = self._run_one_log(
                log_path=Path(log_path),
                run_folder=run_folder,
                route_index=route_index,
                filter_ids=filter_ids,
                filter_tunes=request.filter_tunes,
            )
            failures.extend(route_result["failures"])
            aggregate_rows.extend(route_result["aggregate_rows"])
            if first_raw_rmse is None:
                first_raw_rmse = route_result.get("raw_gnss_rmse_m")
            for row in route_result["aggregate_rows"]:
                rmse = _optional_float(row.get("position_rmse_m"))
                filter_id = str(row.get("filter_id") or "")
                if filter_id == "raw_gnss" or rmse is None:
                    continue
                if best_rmse is None or rmse < best_rmse:
                    best_rmse = rmse
                    best_filter_id = filter_id

        _write_csv(run_folder / "aggregate_summary.csv", _aggregate_fieldnames(), aggregate_rows)
        aggregate_summary = {
            "route_count": len(request.sensor_log_paths),
            "selected_filters": filter_ids,
            "best_filter_id": best_filter_id,
            "best_position_rmse_m": best_rmse,
            "raw_gnss_rmse_m": first_raw_rmse,
            "failures": failures,
            "aggregate_rows": aggregate_rows,
            "explanation": OFFLINE_LOCALIZATION_EXPLANATION,
        }
        write_json(run_folder / "aggregate_summary.json", aggregate_summary)
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
            failures=tuple(failures),
        )

    def _run_one_log(
        self,
        log_path: Path,
        run_folder: Path,
        route_index: int,
        filter_ids: list[str],
        filter_tunes: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        rows = _read_log_rows(log_path)
        if len(rows) < 2:
            failure = {
                "sensor_log_path": str(log_path),
                "reason": "Sensor log is empty or too short.",
            }
            return {"failures": [failure], "aggregate_rows": [], "raw_gnss_rmse_m": None}

        source_metadata = _source_metadata(log_path)
        route_name = _route_name(source_metadata, log_path)
        route_folder = run_folder / f"route_{route_index:03d}_{slugify(route_name, 'route')}"
        result_dir = route_folder / "replay_results"
        metrics_dir = route_folder / "metrics"
        plots_dir = route_folder / "plots"
        result_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        raw_estimates = _raw_gnss_estimates(rows)
        raw_metrics = compute_localization_metrics(raw_estimates, raw_gnss_rmse_m=None)
        raw_rmse = _optional_float(raw_metrics.get("position_rmse_m"))
        _write_csv(result_dir / "raw_gnss_estimates.csv", ESTIMATE_FIELDNAMES, raw_estimates)
        write_json(metrics_dir / "raw_gnss_metrics.json", raw_metrics)

        summary_rows = [_summary_row(route_name, "raw_gnss", raw_metrics, replay_runtime_s=0.0)]
        failures: list[dict[str, object]] = []
        route_metadata = {
            "mode": OFFLINE_MODE_NAME,
            "report_name": OFFLINE_REPORT_NAME,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sensor_log_path": str(log_path),
            "route_name": route_name,
            "map_name": source_metadata.get("map_name"),
            "selected_filters": filter_ids,
            "filter_tunes": {
                filter_id: dict(filter_tunes.get(filter_id, self._records.get(filter_id).tune if self._records.get(filter_id) else {}))
                for filter_id in filter_ids
                if filter_id != "raw_gnss"
            },
            "initial_condition_policy": "first_valid_gnss_initializes_each_filter",
            "sample_count": len(rows),
            "recording_metadata": source_metadata,
            "explanation": OFFLINE_LOCALIZATION_EXPLANATION,
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
                metrics = compute_localization_metrics(estimates, raw_gnss_rmse_m=raw_rmse)
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
            "rows": summary_rows,
            "failures": failures,
        }
        write_json(metrics_dir / "summary_metrics.json", summary_json)
        route_metadata["failures"] = failures
        write_json(route_folder / "metadata.json", route_metadata)
        generate_offline_route_plots(route_folder)
        aggregate_rows = [
            {"route_index": route_index, **row, "sensor_log_path": str(log_path)}
            for row in summary_rows
        ]
        return {
            "failures": failures,
            "aggregate_rows": aggregate_rows,
            "raw_gnss_rmse_m": raw_rmse,
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
            except Exception:
                failures += 1
                state = None
                diagnostics = {}
            estimates.append(_estimate_row(row, filter_id, state, diagnostics))
        return estimates, {
            "failed_sample_updates": failures,
            "last_gnss_frame": last_gnss_frame,
            "last_imu_frame": last_imu_frame,
        }


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
        timestamp = _optional_float(row.get("timestamp"))
        speed = None
        yaw = previous_yaw
        if previous is not None and timestamp is not None and x is not None and y is not None:
            prev_t = _optional_float(previous.get("timestamp"))
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
    gt_x = _optional_float(row.get("ground_truth_x"))
    gt_y = _optional_float(row.get("ground_truth_y"))
    gt_vx = _optional_float(row.get("ground_truth_vx_mps"))
    gt_vy = _optional_float(row.get("ground_truth_vy_mps"))
    est_x = state.x if state is not None else None
    est_y = state.y if state is not None else None
    x_error = (est_x - gt_x) if est_x is not None and gt_x is not None and math.isfinite(est_x) else None
    y_error = (est_y - gt_y) if est_y is not None and gt_y is not None and math.isfinite(est_y) else None
    speed_error = None
    if state is not None:
        gt_speed = _optional_float(row.get("ground_truth_speed"))
        if gt_speed is not None:
            speed_error = state.speed - gt_speed
    velocity_error = None
    if state is not None and state.vx_mps is not None and state.vy_mps is not None and gt_vx is not None and gt_vy is not None:
        velocity_error = math.hypot(state.vx_mps - gt_vx, state.vy_mps - gt_vy)
    covariance = diagnostics.get("covariance_diagonal")
    if covariance is None and state is not None:
        covariance = state.covariance_diagonal
    return {
        "timestamp": row.get("timestamp"),
        "frame": row.get("frame"),
        "filter_id": filter_id,
        "ground_truth_x": row.get("ground_truth_x"),
        "ground_truth_y": row.get("ground_truth_y"),
        "ground_truth_z": row.get("ground_truth_z"),
        "ground_truth_yaw": row.get("ground_truth_yaw"),
        "ground_truth_speed": row.get("ground_truth_speed"),
        "ground_truth_vx_mps": row.get("ground_truth_vx_mps"),
        "ground_truth_vy_mps": row.get("ground_truth_vy_mps"),
        "estimate_x": est_x,
        "estimate_y": est_y,
        "estimate_z": state.z if state is not None else None,
        "estimate_yaw": state.yaw if state is not None else None,
        "estimate_speed": state.speed if state is not None else None,
        "estimate_vx_mps": state.vx_mps if state is not None else None,
        "estimate_vy_mps": state.vy_mps if state is not None else None,
        "position_error_m": position_error(est_x, est_y, row.get("ground_truth_x"), row.get("ground_truth_y")),
        "x_error_m": x_error,
        "y_error_m": y_error,
        "yaw_error_deg": yaw_error_deg(state.yaw if state is not None else None, row.get("ground_truth_yaw")),
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
        "position_rmse_m": metrics.get("position_rmse_m"),
        "position_mae_m": metrics.get("position_mae_m"),
        "mean_position_error_m": metrics.get("mean_position_error_m"),
        "position_error_std_m": metrics.get("position_error_std_m"),
        "max_position_error_m": metrics.get("max_position_error_m"),
        "final_position_error_m": metrics.get("final_position_error_m"),
        "improvement_over_raw_gnss_percent": metrics.get("improvement_over_raw_gnss_percent"),
        "valid_estimate_count": metrics.get("valid_estimate_count"),
        "missing_or_invalid_estimate_count": metrics.get("missing_or_invalid_estimate_count"),
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
        "position_rmse_m",
        "position_mae_m",
        "mean_position_error_m",
        "position_error_std_m",
        "max_position_error_m",
        "final_position_error_m",
        "improvement_over_raw_gnss_percent",
        "valid_estimate_count",
        "missing_or_invalid_estimate_count",
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


def _normalize_angle_deg(angle_deg: float) -> float:
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg
