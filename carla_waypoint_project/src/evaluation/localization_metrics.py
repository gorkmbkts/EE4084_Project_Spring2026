"""Metrics for offline localization replay results."""

from __future__ import annotations

import math
from typing import Iterable, Optional

from config.settings import BENCHMARK
from src.evaluation.consistency_metrics import (
    consistency_report_from_summaries,
    position_nees,
    summarize_nis_by_type,
    summarize_position_nees,
)


def compute_localization_metrics(
    samples: Iterable[dict[str, object]],
    raw_gnss_rmse_m: Optional[float] = None,
    divergence_error_threshold_m: Optional[float] = None,
) -> dict[str, object]:
    """Compute report-friendly localization metrics from estimate samples."""
    rows = list(samples)
    threshold = (
        float(divergence_error_threshold_m)
        if divergence_error_threshold_m is not None
        else float(BENCHMARK.divergence_error_threshold_m)
    )
    eval_rows = [row for row in rows if _bool_value(row.get("valid_for_metrics"), default=True)]
    full = _metric_group(rows, "full")
    eval_metrics = _metric_group(eval_rows, "eval")
    eval_position_errors = _finite_values(row.get("position_error_m") for row in eval_rows)
    full_position_errors = _finite_values(row.get("position_error_m") for row in rows)
    valid_count = len(eval_position_errors)
    eval_sample_count = len(eval_rows)
    total_count = len(rows)
    eval_rmse = eval_metrics["eval_position_rmse_m"]
    improvement = None
    if raw_gnss_rmse_m is not None and raw_gnss_rmse_m > 0.0 and eval_rmse is not None:
        improvement = 100.0 * (raw_gnss_rmse_m - eval_rmse) / raw_gnss_rmse_m
    divergence_rows = [
        row for row in eval_rows
        if (error := _optional_float(row.get("position_error_m"))) is not None and error > threshold
    ]

    metrics = {
        **full,
        **eval_metrics,
        "position_rmse_m": eval_metrics["eval_position_rmse_m"],
        "position_mae_m": eval_metrics["eval_position_mae_m"],
        "max_position_error_m": eval_metrics["eval_max_position_error_m"],
        "final_position_error_m": eval_metrics["eval_final_position_error_m"],
        "mean_position_error_m": eval_metrics["eval_mean_position_error_m"],
        "position_error_std_m": eval_metrics["eval_position_error_std_m"],
        "improvement_over_raw_gnss_percent": improvement,
        "eval_improvement_over_raw_gnss_percent": improvement,
        "valid_estimate_count": valid_count,
        "missing_or_invalid_estimate_count": max(0, eval_sample_count - valid_count),
        "full_valid_estimate_count": len(full_position_errors),
        "full_missing_or_invalid_estimate_count": max(0, total_count - len(full_position_errors)),
        "valid_for_metrics_sample_count": eval_sample_count,
        "warmup_excluded_sample_count": max(0, total_count - eval_sample_count),
        "warmup_excluded_s": _excluded_duration_s(rows),
        "total_sample_count": total_count,
        "median_position_error_m": _percentile(eval_position_errors, 50.0),
        "p95_position_error_m": _percentile(eval_position_errors, 95.0),
        "p99_position_error_m": _percentile(eval_position_errors, 99.0),
        "full_median_position_error_m": _percentile(full_position_errors, 50.0),
        "full_p95_position_error_m": _percentile(full_position_errors, 95.0),
        "full_p99_position_error_m": _percentile(full_position_errors, 99.0),
        "divergence_error_threshold_m": threshold,
        "divergence_event_count": len(divergence_rows),
        "divergence_duration_s": _duration_for_rows(divergence_rows, eval_rows),
        "yaw_rmse_deg": eval_metrics["eval_yaw_rmse_deg"],
        "speed_rmse_mps": eval_metrics["eval_speed_rmse_mps"],
        "velocity_rmse_mps": eval_metrics["eval_velocity_rmse_mps"],
        "mean_nis": eval_metrics["eval_mean_nis"],
        "mean_nees": eval_metrics["eval_mean_nees"],
        "legacy_mean_nis_mixed": eval_metrics["eval_mean_nis"],
        "legacy_mean_nis_mixed_note": "Legacy mixed scalar NIS; prefer nis_by_type_summary.",
        "nis_by_type_summary": eval_metrics["eval_nis_by_type_summary"],
        "position_nees_summary": eval_metrics["eval_position_nees_summary"],
        "position_nees_diagonal_approx_summary": eval_metrics["eval_position_nees_diagonal_approx_summary"],
        "position_nees_source": eval_metrics["eval_position_nees_source"],
        "mean_position_nees": eval_metrics["eval_mean_position_nees"],
        "mean_position_nees_diagonal_approx": eval_metrics["eval_mean_position_nees_diagonal_approx"],
        "nis_available": bool(eval_metrics["eval_nis_available"]),
        "nees_available": bool(eval_metrics["eval_nees_available"]),
        "position_nees_available": bool(eval_metrics["eval_position_nees_available"]),
        "position_nees_diagonal_approx_available": bool(eval_metrics["eval_position_nees_diagonal_approx_available"]),
    }
    metrics["consistency_report"] = consistency_report_from_summaries(
        nis_by_type_summary=metrics["nis_by_type_summary"],
        mean_position_nees=metrics["mean_position_nees"],
        mean_position_nees_diagonal_approx=metrics["mean_position_nees_diagonal_approx"],
        position_nees_source=metrics["position_nees_source"],
    )
    return metrics


def _metric_group(rows: list[dict[str, object]], prefix: str) -> dict[str, object]:
    position_errors = _finite_values(row.get("position_error_m") for row in rows)
    yaw_errors = [abs(value) for value in _finite_values(row.get("yaw_error_deg") for row in rows)]
    speed_errors = _finite_values(row.get("speed_error_mps") for row in rows)
    velocity_errors = _finite_values(row.get("velocity_error_mps") for row in rows)
    nis_values = _finite_values(row.get("nis") for row in rows)
    nees_summary = summarize_position_nees(rows)
    nis_by_type = summarize_nis_by_type(rows)
    return {
        f"{prefix}_position_rmse_m": _rmse(position_errors),
        f"{prefix}_position_mae_m": _mean(position_errors),
        f"{prefix}_mean_position_error_m": _mean(position_errors),
        f"{prefix}_max_position_error_m": max(position_errors) if position_errors else None,
        f"{prefix}_final_position_error_m": _last_finite(row.get("position_error_m") for row in rows),
        f"{prefix}_position_error_std_m": _stddev(position_errors),
        f"{prefix}_yaw_rmse_deg": _rmse(yaw_errors),
        f"{prefix}_speed_rmse_mps": _rmse(speed_errors),
        f"{prefix}_velocity_rmse_mps": _rmse(velocity_errors),
        f"{prefix}_mean_nis": _mean(nis_values) if nis_values else None,
        f"{prefix}_mean_nees": nees_summary["mean_nees"],
        f"{prefix}_nis_by_type_summary": nis_by_type,
        f"{prefix}_position_nees_summary": nees_summary["position_nees_summary"],
        f"{prefix}_position_nees_diagonal_approx_summary": nees_summary["position_nees_diagonal_approx_summary"],
        f"{prefix}_legacy_nees_summary": nees_summary["legacy_nees_summary"],
        f"{prefix}_position_nees_source": nees_summary["position_nees_source"],
        f"{prefix}_mean_position_nees": nees_summary["mean_position_nees"],
        f"{prefix}_mean_position_nees_diagonal_approx": nees_summary["mean_position_nees_diagonal_approx"],
        f"{prefix}_nis_available": bool(nis_values),
        f"{prefix}_nees_available": bool(nees_summary["nees_available"]),
        f"{prefix}_position_nees_available": bool(nees_summary["position_nees_available"]),
        f"{prefix}_position_nees_diagonal_approx_available": bool(nees_summary["position_nees_diagonal_approx_available"]),
    }


def position_error(x_value: object, y_value: object, gt_x: object, gt_y: object) -> Optional[float]:
    x = _optional_float(x_value)
    y = _optional_float(y_value)
    x_gt = _optional_float(gt_x)
    y_gt = _optional_float(gt_y)
    if x is None or y is None or x_gt is None or y_gt is None:
        return None
    return math.hypot(x - x_gt, y - y_gt)


def yaw_error_deg(yaw_value: object, gt_yaw_value: object) -> Optional[float]:
    yaw = _optional_float(yaw_value)
    gt_yaw = _optional_float(gt_yaw_value)
    if yaw is None or gt_yaw is None:
        return None
    delta = yaw - gt_yaw
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def nees_xy(
    x_error_m: object,
    y_error_m: object,
    covariance_diagonal: object,
    position_covariance_2x2: object = None,
) -> Optional[float]:
    result = position_nees(
        x_error_m=x_error_m,
        y_error_m=y_error_m,
        position_covariance_2x2=position_covariance_2x2,
        covariance_diagonal=covariance_diagonal,
    )
    return _optional_float(result.get("nees"))


def _finite_values(values: Iterable[object]) -> list[float]:
    result: list[float] = []
    for value in values:
        number = _optional_float(value)
        if number is not None:
            result.append(number)
    return result


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


def _optional_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rmse(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _stddev(values: list[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _last_finite(values: Iterable[object]) -> Optional[float]:
    latest = None
    for value in values:
        number = _optional_float(value)
        if number is not None:
            latest = number
    return latest


def _percentile(values: list[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    index = int(math.ceil((percentile / 100.0) * len(sorted_values))) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def _excluded_duration_s(rows: list[dict[str, object]]) -> float:
    excluded = [row for row in rows if not _bool_value(row.get("valid_for_metrics"), default=True)]
    return _duration_for_rows(excluded, rows)


def _duration_for_rows(rows: list[dict[str, object]], reference_rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    dt_values = [_optional_float(row.get("dt")) for row in rows]
    finite_dt = [value for value in dt_values if value is not None and value >= 0.0]
    if finite_dt:
        return float(sum(finite_dt))
    timestamps = [
        _first_float(
            row.get("seconds_since_replay_start"),
            row.get("seconds_since_recording_start"),
            row.get("seconds_since_teleport"),
        )
        for row in rows
    ]
    finite_times = [value for value in timestamps if value is not None]
    if not finite_times:
        median_dt = _median_sample_dt(reference_rows)
        return float(len(rows) * median_dt) if median_dt is not None else 0.0
    median_dt = _median_sample_dt(reference_rows) or 0.0
    return float(max(finite_times) - min(finite_times) + median_dt)


def _median_sample_dt(rows: list[dict[str, object]]) -> Optional[float]:
    previous = None
    deltas: list[float] = []
    for row in rows:
        timestamp = (
            _first_float(
                row.get("seconds_since_replay_start"),
                row.get("seconds_since_recording_start"),
                row.get("seconds_since_teleport"),
                row.get("timestamp"),
            )
        )
        if timestamp is None:
            continue
        if previous is not None and timestamp > previous:
            deltas.append(timestamp - previous)
        previous = timestamp
    return _percentile(deltas, 50.0)


def _first_float(*values: object) -> Optional[float]:
    for value in values:
        number = _optional_float(value)
        if number is not None:
            return number
    return None
