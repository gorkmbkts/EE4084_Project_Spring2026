"""Closed-loop paper metric and summary dashboard tests."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark_plotter import _plot_summary_dashboard  # noqa: E402
from src.evaluation.closed_loop_metrics import compute_closed_loop_utility_metrics  # noqa: E402
from src.evaluation.route_test_runner import RouteTestRunner  # noqa: E402


def _successful_metrics() -> dict[str, object]:
    return {
        "route_completion_success": True,
        "attempts_used": 1,
        "completion_time_s": 60.0,
        "route_timeout_s": 120.0,
        "eval_raw_gnss_rmse_m": 4.0,
        "eval_filtered_rmse_m": 2.0,
        "driving_mean_cross_track_error_m": 0.5,
        "driving_p95_cross_track_error_m": 1.0,
        "eval_yaw_rmse_deg": 2.0,
        "driving_divergence_event_count": 0,
        "driving_nis_by_type_summary": {
            "gnss_position": {
                "mean": 2.0,
                "sample_count": 20,
                "expected_dimension": 2,
            }
        },
        "driving_mean_position_nees": 2.0,
        "driving_position_nees_source": "full_2x2",
        "sensor_noise_profile": "Medium Noise",
        "actuator_realism_profile": "Realistic",
    }


def _score(overrides: dict[str, object] | None = None) -> dict[str, object]:
    metrics = _successful_metrics()
    metrics.update(overrides or {})
    return compute_closed_loop_utility_metrics(metrics)


def test_closed_loop_utility_score_is_finite_for_successful_route() -> None:
    result = _score()
    assert math.isfinite(float(result["closed_loop_utility_score"]))
    assert float(result["closed_loop_utility_score"]) > 0.0
    assert result["paper_ready_score_available"] is True
    assert result["difficulty_factor"] == 1.25


def test_failed_route_has_zero_or_heavily_reduced_utility() -> None:
    success = _score()
    failure = _score({"route_completion_success": False})
    assert failure["closed_loop_utility_score"] == 0.0
    assert float(failure["closed_loop_utility_score"]) < 0.2 * float(success["closed_loop_utility_score"])


def test_more_attempts_reduce_closed_loop_utility() -> None:
    one_attempt = _score({"attempts_used": 1})
    three_attempts = _score({"attempts_used": 3})
    assert float(three_attempts["closed_loop_utility_score"]) < float(one_attempt["closed_loop_utility_score"])


def test_higher_cross_track_error_reduces_closed_loop_utility() -> None:
    low_cte = _score({"driving_mean_cross_track_error_m": 0.2})
    high_cte = _score({"driving_mean_cross_track_error_m": 2.0})
    assert float(high_cte["closed_loop_utility_score"]) < float(low_cte["closed_loop_utility_score"])


def test_better_filtered_rmse_increases_closed_loop_utility() -> None:
    better = _score({"eval_filtered_rmse_m": 1.0})
    worse = _score({"eval_filtered_rmse_m": 3.5})
    assert float(better["closed_loop_utility_score"]) > float(worse["closed_loop_utility_score"])


def test_missing_optional_fields_do_not_crash_metric_computation() -> None:
    result = compute_closed_loop_utility_metrics({"route_completion_success": False})
    assert math.isfinite(float(result["closed_loop_utility_score"]))
    assert result["closed_loop_utility_score"] == 0.0
    assert result["consistency_penalty"] == 0.0
    assert result["consistency_penalty_source"] == "unavailable"
    assert result["divergence_penalty_source"] == "unavailable"
    assert result["paper_ready_score_available"] is False


def test_paper_csv_contains_new_metric_columns(tmp_path: Path) -> None:
    summary = {
        **_successful_metrics(),
        **_score(),
        "run_id": "run_001",
        "benchmark_id": "route_001",
        "route_index": 1,
        "route_name": "paper route",
        "map_name": "Town01",
        "selected_filter": "ctra_ukf",
        "tracking_mode": "active",
        "tune_algorithm": "direct_closed_loop_optuna_tpe",
        "divergence_event_count": 0,
    }
    runner = RouteTestRunner.__new__(RouteTestRunner)
    runner._route_summaries = [summary]
    output = tmp_path / "paper_metrics.csv"
    runner._write_paper_metrics_csv(output)

    with output.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows[0]["closed_loop_utility_score"]
    assert rows[0]["localization_improvement_ratio"]
    assert rows[0]["filter_id"] == "ctra_ukf"
    assert rows[0]["tune_algorithm"] == "direct_closed_loop_optuna_tpe"


def test_legacy_summary_still_generates_dashboard(tmp_path: Path) -> None:
    output = tmp_path / "summary_dashboard.png"
    legacy_summary = {
        "route_completion_success": True,
        "completion_time_s": 20.0,
        "filtered_rmse_m": 1.0,
        "raw_gnss_rmse_m": 2.0,
        "mean_cross_track_error_m": 0.3,
        "p95_cross_track_error_m": 0.7,
        "yaw_rmse_deg": 1.5,
    }
    metadata = {
        "active_filter": {"id": "ca_kf", "name": "CA KF", "tracking_mode": "passive"},
        "general": {"route_name": "legacy route", "map_name": "Town01"},
    }
    _plot_summary_dashboard(output, metadata, legacy_summary, [])
    assert output.exists()
    assert output.stat().st_size > 0
