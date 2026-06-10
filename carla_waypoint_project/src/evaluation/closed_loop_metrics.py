"""Paper-oriented utility metrics for closed-loop benchmark summaries."""

from __future__ import annotations

import json
import math
from typing import Optional

from src.evaluation.consistency_metrics import (
    consistency_report_from_summaries,
    severity_rank,
)


EPSILON = 1.0e-9
FALLBACK_ROUTE_TIMEOUT_S = 120.0

CLOSED_LOOP_UTILITY_SCORE_FORMULA = (
    "difficulty_factor * route_completed_factor * attempt_factor * "
    "(1 + localization_improvement_ratio) / "
    "(1 + time_norm + cte_norm + p95_cte_norm + yaw_norm + "
    "consistency_penalty + divergence_penalty)"
)
CLOSED_LOOP_UTILITY_SCORE_NOTES = (
    "Primary localization and tracking inputs prefer evaluation/driving-phase metrics. "
    "Missing optional consistency and divergence inputs contribute zero penalty; "
    "paper_ready_score_available identifies whether all core paper inputs were present."
)

UTILITY_COMPONENT_FIELDS = (
    "route_completed_factor",
    "attempt_factor",
    "difficulty_factor",
    "localization_improvement_ratio",
    "time_norm",
    "cte_norm",
    "p95_cte_norm",
    "yaw_norm",
    "consistency_penalty",
    "divergence_penalty",
    "closed_loop_completion_efficiency",
    "route_tracking_quality_score",
    "closed_loop_utility_score",
)

PAPER_METRIC_FIELDS = (
    "raw_gnss_position_rmse_m",
    "filtered_position_rmse_m",
    *UTILITY_COMPONENT_FIELDS,
    "paper_ready_score_available",
    "consistency_status",
    "consistency_penalty_source",
    "divergence_penalty_source",
    "difficulty_factor_source",
    "route_timeout_s",
    "route_timeout_source",
    "time_norm_source",
    "paper_ready_score_missing_fields",
    "closed_loop_utility_score_formula",
    "closed_loop_utility_score_notes",
)


def compute_closed_loop_utility_metrics(
    metrics: dict[str, object],
    *,
    sensor_noise_profile: object = None,
    actuator_realism_profile: object = None,
    route_timeout_s: object = None,
) -> dict[str, object]:
    """Compute finite, JSON-serializable closed-loop paper metrics."""
    filtered_rmse = _first_float(
        metrics,
        "eval_filtered_rmse_m",
        "driving_filtered_rmse_m",
        "filtered_position_rmse_m",
        "filtered_rmse_m",
        "mean_filtered_rmse_m",
    )
    raw_rmse = _first_float(
        metrics,
        "eval_raw_gnss_rmse_m",
        "driving_raw_gnss_rmse_m",
        "raw_gnss_position_rmse_m",
        "raw_gnss_rmse_m",
        "mean_raw_gnss_rmse_m",
    )
    mean_cte = _first_float(
        metrics,
        "driving_mean_cross_track_error_m",
        "mean_cross_track_error_m",
    )
    p95_cte = _first_float(
        metrics,
        "driving_p95_cross_track_error_m",
        "p95_cross_track_error_m",
    )
    yaw_rmse = _first_float(
        metrics,
        "eval_yaw_rmse_deg",
        "driving_yaw_rmse_deg",
        "yaw_rmse_deg",
    )
    completion_time = _first_float(metrics, "completion_time_s")
    attempts_used = _first_float(metrics, "attempts_used", "successful_attempt")
    attempts = max(1.0, attempts_used if attempts_used is not None else 1.0)

    route_completed = _as_bool(
        metrics.get("route_completion_success")
        if metrics.get("route_completion_success") is not None
        else metrics.get("route_completed")
    )
    progress_ratio = _progress_ratio(metrics)
    completion_factor = 1.0 if route_completed else 0.2 * progress_ratio if progress_ratio is not None else 0.0
    attempt_factor = 1.0 / attempts

    localization_improvement = 0.0
    if raw_rmse is not None and filtered_rmse is not None:
        localization_improvement = max(0.0, raw_rmse / max(filtered_rmse, EPSILON) - 1.0)

    timeout_value = _optional_float(route_timeout_s)
    timeout_source = "argument"
    if timeout_value is None:
        timeout_value = _first_float(metrics, "route_timeout_s", "max_pass_duration_s")
        timeout_source = "summary"
    if timeout_value is None:
        timeout_value = _nested_route_timeout(metrics)
        timeout_source = "metadata"
    if timeout_value is None or timeout_value <= 0.0:
        timeout_value = FALLBACK_ROUTE_TIMEOUT_S
        timeout_source = f"fallback_{FALLBACK_ROUTE_TIMEOUT_S:g}_s"

    effective_completion_time = completion_time if completion_time is not None else timeout_value
    time_norm_source = (
        "completion_time_s"
        if completion_time is not None
        else "route_timeout_assumed_when_completion_time_missing"
    )
    time_norm = max(0.0, effective_completion_time) / max(timeout_value, EPSILON)
    cte_norm = max(0.0, mean_cte or 0.0) / 1.0
    p95_cte_norm = max(0.0, p95_cte or 0.0) / 2.0
    yaw_norm = max(0.0, yaw_rmse or 0.0) / 5.0

    divergence, divergence_source = _first_float_with_source(
        metrics,
        (
            "eval_divergence_event_count",
            "driving_divergence_event_count",
            "divergence_event_count",
            "filter_divergence_event_count",
            "estimator_divergence_count",
        ),
    )
    divergence_penalty = max(0.0, divergence or 0.0)

    consistency_report, consistency_source = _consistency_report(metrics)
    consistency_status = str(consistency_report.get("overall_status") or "unavailable")
    consistency_error = _optional_float(consistency_report.get("consistency_error")) or 0.0
    consistency_penalty = 0.0
    if consistency_status != "unavailable":
        consistency_penalty = consistency_error + max(0, severity_rank(consistency_status) - 1)

    difficulty_factor, difficulty_source = _difficulty_factor(
        metrics,
        sensor_noise_profile=sensor_noise_profile,
        actuator_realism_profile=actuator_realism_profile,
    )
    completion_efficiency = completion_factor * attempt_factor / (1.0 + time_norm)
    tracking_quality = 1.0 / (
        1.0 + cte_norm + p95_cte_norm + yaw_norm + divergence_penalty
    )
    denominator = (
        1.0
        + time_norm
        + cte_norm
        + p95_cte_norm
        + yaw_norm
        + consistency_penalty
        + divergence_penalty
    )
    utility = (
        difficulty_factor
        * completion_factor
        * attempt_factor
        * (1.0 + localization_improvement)
        / max(denominator, EPSILON)
    )

    required = {
        "route_completion_success": _has_any(metrics, "route_completion_success", "route_completed"),
        "completion_time_s": completion_time is not None,
        "raw_gnss_position_rmse_m": raw_rmse is not None,
        "filtered_position_rmse_m": filtered_rmse is not None,
        "mean_cross_track_error_m": mean_cte is not None,
        "p95_cross_track_error_m": p95_cte is not None,
        "yaw_rmse_deg": yaw_rmse is not None,
    }
    missing_fields = [key for key, available in required.items() if not available]

    return {
        "raw_gnss_position_rmse_m": raw_rmse,
        "filtered_position_rmse_m": filtered_rmse,
        "route_completed_factor": _finite_float(completion_factor),
        "attempt_factor": _finite_float(attempt_factor),
        "difficulty_factor": _finite_float(difficulty_factor),
        "localization_improvement_ratio": _finite_float(localization_improvement),
        "time_norm": _finite_float(time_norm),
        "cte_norm": _finite_float(cte_norm),
        "p95_cte_norm": _finite_float(p95_cte_norm),
        "yaw_norm": _finite_float(yaw_norm),
        "consistency_penalty": _finite_float(consistency_penalty),
        "divergence_penalty": _finite_float(divergence_penalty),
        "closed_loop_completion_efficiency": _finite_float(completion_efficiency),
        "route_tracking_quality_score": _finite_float(tracking_quality),
        "closed_loop_utility_score": _finite_float(utility),
        "paper_ready_score_available": not missing_fields,
        "consistency_status": consistency_status,
        "consistency_penalty_source": consistency_source,
        "divergence_penalty_source": divergence_source,
        "difficulty_factor_source": difficulty_source,
        "route_timeout_s": _finite_float(timeout_value),
        "route_timeout_source": timeout_source,
        "time_norm_source": time_norm_source,
        "paper_ready_score_missing_fields": missing_fields,
        "closed_loop_utility_score_formula": CLOSED_LOOP_UTILITY_SCORE_FORMULA,
        "closed_loop_utility_score_notes": CLOSED_LOOP_UTILITY_SCORE_NOTES,
    }


def _consistency_report(metrics: dict[str, object]) -> tuple[dict[str, object], str]:
    existing = _dict_value(metrics.get("consistency_report"))
    if existing and str(existing.get("overall_status") or "unavailable") != "unavailable":
        return existing, "existing_consistency_report"

    nis_summary = _first_dict(
        metrics,
        "driving_nis_by_type_summary",
        "eval_nis_by_type_summary",
        "nis_by_type_summary",
    )
    mean_nees = _first_float(
        metrics,
        "driving_mean_position_nees",
        "eval_mean_position_nees",
        "mean_position_nees",
    )
    mean_nees_approx = _first_float(
        metrics,
        "driving_mean_position_nees_diagonal_approx",
        "eval_mean_position_nees_diagonal_approx",
        "mean_position_nees_diagonal_approx",
    )
    nees_source = _first_object(
        metrics,
        "driving_position_nees_source",
        "eval_position_nees_source",
        "position_nees_source",
    )
    if not nis_summary and mean_nees is None and mean_nees_approx is None:
        return {"overall_status": "unavailable", "consistency_error": 0.0}, "unavailable"
    report = consistency_report_from_summaries(
        nis_by_type_summary=nis_summary,
        mean_position_nees=mean_nees,
        mean_position_nees_diagonal_approx=mean_nees_approx,
        position_nees_source=nees_source,
    )
    return report, "derived_from_nis_nees"


def _difficulty_factor(
    metrics: dict[str, object],
    *,
    sensor_noise_profile: object,
    actuator_realism_profile: object,
) -> tuple[float, dict[str, str]]:
    noise = _profile_text(
        sensor_noise_profile,
        metrics.get("sensor_noise_profile"),
        metrics.get("sensor_noise_preset"),
        metrics.get("sensor_noise_config"),
    )
    actuator = _profile_text(
        actuator_realism_profile,
        metrics.get("actuator_realism_profile"),
        metrics.get("actuator_realism_preset"),
        metrics.get("actuator_realism_config"),
    )

    factor = 1.0
    noise_class = "unknown"
    noise_lower = noise.lower()
    if any(token in noise_lower for token in ("high", "degraded", "harsh")):
        factor += 0.25
        noise_class = "high"
    elif "medium" in noise_lower:
        factor += 0.10
        noise_class = "medium"
    elif noise:
        noise_class = "known_other"

    actuator_class = "unknown"
    actuator_lower = actuator.lower()
    if "perfect" in actuator_lower:
        actuator_class = "perfect"
    elif "realistic" in actuator_lower:
        factor += 0.15
        actuator_class = "realistic"
    elif actuator:
        actuator_class = "known_other"

    return factor, {
        "sensor_noise_profile": noise or "unknown",
        "sensor_noise_class": noise_class,
        "actuator_realism_profile": actuator or "unknown",
        "actuator_realism_class": actuator_class,
    }


def _nested_route_timeout(metrics: dict[str, object]) -> Optional[float]:
    for container_key in ("general", "metadata", "benchmark_metadata"):
        container = _dict_value(metrics.get(container_key))
        settings = _dict_value(container.get("benchmark_settings"))
        value = _optional_float(settings.get("max_pass_duration_s"))
        if value is not None:
            return value
    return None


def _progress_ratio(metrics: dict[str, object]) -> Optional[float]:
    value = _first_float(
        metrics,
        "route_progress_ratio",
        "route_progress_percent",
        "route_completion_percent",
        "progress_percent",
    )
    if value is None:
        return None
    if value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))


def _profile_text(*values: object) -> str:
    for value in values:
        if isinstance(value, dict):
            text = str(
                value.get("preset_name")
                or value.get("profile")
                or value.get("name")
                or ""
            ).strip()
        else:
            text = str(value or "").strip()
        if text:
            return text
    return ""


def _has_any(metrics: dict[str, object], *keys: str) -> bool:
    return any(key in metrics and metrics.get(key) is not None for key in keys)


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "completed", "success"}
    return bool(value)


def _first_float(metrics: dict[str, object], *keys: str) -> Optional[float]:
    for key in keys:
        value = _optional_float(metrics.get(key))
        if value is not None:
            return value
    return None


def _first_float_with_source(
    metrics: dict[str, object],
    keys: tuple[str, ...],
) -> tuple[Optional[float], str]:
    for key in keys:
        value = _optional_float(metrics.get(key))
        if value is not None:
            return value, key
    return None, "unavailable"


def _first_dict(metrics: dict[str, object], *keys: str) -> dict[str, object]:
    for key in keys:
        value = _dict_value(metrics.get(key))
        if value:
            return value
    return {}


def _first_object(metrics: dict[str, object], *keys: str) -> object:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def _dict_value(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _optional_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_float(value: object) -> float:
    number = _optional_float(value)
    return number if number is not None else 0.0
