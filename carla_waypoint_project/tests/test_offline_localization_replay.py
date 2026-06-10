"""Offline localization replay artifact and fairness checks."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pygame
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluation_artifacts import (  # noqa: E402
    DEFAULT_OFFLINE_OUTPUT_ROOT,
    OFFLINE_LOCALIZATION_EXPLANATION,
    RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
    list_recorded_logs,
    recordings_root,
    write_json,
)
from src.evaluation.localization_metrics import compute_localization_metrics  # noqa: E402
from src.evaluation.offline_replay_runner import (  # noqa: E402
    OfflineReplayRequest,
    OfflineReplayRunner,
    _annotate_metric_ground_truth,
    _raw_gnss_estimates,
)
import src.evaluation.offline_replay_runner as offline_replay_runner_module  # noqa: E402
from src.evaluation.filter_auto_tuner import AutoTuneRequest, FilterAutoTuner, objective_score  # noqa: E402
import src.evaluation.filter_auto_tuner as filter_auto_tuner_module  # noqa: E402
from src.evaluation.sensor_log_recorder import SENSOR_LOG_FIELDNAMES  # noqa: E402
from src.evaluation.test_route_store import TestRouteStore  # noqa: E402
from src.control.driving_behavior import ActuatorRealism, CurvatureSpeedPlanner, DrivingBehaviorConfig  # noqa: E402
from src.control.vehicle_controller import VehicleController  # noqa: E402
from src.evaluation.benchmark_config import (  # noqa: E402
    BEHAVIOR_PRESETS,
    BenchmarkConfig,
    ParameterSpec,
    SENSOR_NOISE_PRESETS,
    sensor_noise_config_from_values,
    validate_benchmark_config,
)
from src.evaluation.closed_loop_auto_tune import (  # noqa: E402
    ClosedLoopAutoTuneRequest,
    ClosedLoopFinalist,
    ClosedLoopValidationRequest,
    ClosedLoopValidationRoute,
    PendingClosedLoopAutoTuneSession,
)
from src.evaluation.consistency_metrics import consistency_report_from_summaries, position_nees, summarize_nis_by_type  # noqa: E402
from src.evaluation.sensor_noise_tune_mapper import (  # noqa: E402
    SensorNoiseTuneMapper,
    noise_signature,
    process_only_auto_tune_profile,
)
from src.evaluation.tune_config_schema import (  # noqa: E402
    BENCHMARK_MODE_CLOSED_LOOP,
    BENCHMARK_MODE_OFFLINE,
    SCHEMA_VERSION,
    TRACKING_ACTIVE,
    TRACKING_PASSIVE,
    TUNE_SCOPE_CLOSED_LOOP_VALIDATED,
    TUNE_SCOPE_OFFLINE,
    TuneCompatibility,
    closed_loop_tune_context,
    offline_tune_context,
)
from src.visualization.ui.parameter_controls import ParameterEditor  # noqa: E402
from src.KalmanLab.registry import discover_filters  # noqa: E402
from src.KalmanLab.filter_base import TRACKING_MODE_ACTIVE, TRACKING_MODE_PASSIVE  # noqa: E402
import src.visualization.startup_map_selector as startup_map_selector_module  # noqa: E402
from src.visualization.startup_map_selector import (  # noqa: E402
    CLOSED_LOOP_SUBTABS,
    OFFLINE_SUBTABS,
    OFFLINE_TEST_SETUP_SUBTABS,
    TOP_LEVEL_TABS,
    StartupMapSelector,
)
from src.core.app import AppClosedLoopValidationRunner, DriveMode, RouteActivationState, SimulationApp  # noqa: E402


def test_offline_replay_runs_on_short_saved_route_log(tmp_path: Path) -> None:
    route = _short_saved_route()
    output_root = tmp_path / "benchmark_results"
    route_folder = output_root / "offline_localization" / "recordings" / "20260605_000000" / "route_001_sadece_viraj"
    route_folder.mkdir(parents=True)
    log_path = route_folder / "sensor_log.csv"
    _write_synthetic_log(log_path, route.name, route.map_name or "Carla/Maps/Town01_Opt")
    write_json(
        route_folder / "route_metadata.json",
        {
            "route": route.to_dict(),
            "map_name": route.map_name,
            "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
            "sensor_noise_config": {"preset_name": "Synthetic"},
            "vehicle_behavior_config": {"preset_name": "Synthetic"},
            "explanation": OFFLINE_LOCALIZATION_EXPLANATION,
        },
    )
    write_json(
        route_folder / "recording_summary.json",
        {
            "route_name": route.name,
            "map_name": route.map_name,
            "sample_count": 24,
            "duration_s": 2.3,
            "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
        },
    )

    discovered = list_recorded_logs(str(output_root))
    if len(discovered) != 1:
        raise AssertionError(f"expected one discovered log, got {len(discovered)}")
    if discovered[0].recording_driver != RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER:
        raise AssertionError("recording driver metadata was not preserved")

    result = OfflineReplayRunner().run(
        OfflineReplayRequest(
            sensor_log_paths=(log_path,),
            selected_filter_ids=("ca_kf",),
            filter_tunes={"ca_kf": {"gnss_position_stddev_m": 2.75}},
            output_root=str(output_root),
        )
    )
    if result.route_count != 1:
        raise AssertionError("offline replay did not evaluate one route")
    if result.failures:
        raise AssertionError(f"offline replay had failures: {result.failures}")
    if result.output_folder.parent.name != "evaluations":
        raise AssertionError(f"normal offline replay should write under evaluations, got {result.output_folder}")

    route_result = next((result.output_folder / "route_001_sadece_viraj").glob("metrics/summary_metrics.csv"))
    rows = list(csv.DictReader(route_result.open("r", newline="", encoding="utf-8")))
    filter_ids = {row["filter_id"] for row in rows}
    if {"raw_gnss", "ca_kf"} - filter_ids:
        raise AssertionError(f"missing expected metrics rows: {filter_ids}")

    metadata = json.loads((result.output_folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("explanation") != OFFLINE_LOCALIZATION_EXPLANATION:
        raise AssertionError("offline replay metadata explanation missing")
    if metadata.get("filter_tunes", {}).get("ca_kf", {}).get("gnss_position_stddev_m") != 2.75:
        raise AssertionError("offline replay run metadata did not preserve requested CA-KF tune")
    if "raw_gnss" in metadata.get("filter_tunes", {}):
        raise AssertionError("raw_gnss should be baseline-only and not have tune metadata")
    for relative in (
        "replay_results/raw_gnss_estimates.csv",
        "replay_results/ca_kf_estimates.csv",
        "plots/trajectory_comparison.png",
        "plots/position_error_over_time.png",
        "plots/rmse_comparison.png",
        "plots/rmse_comparison_full_window.png",
        "plots/rmse_comparison_eval_window.png",
    ):
        if not (result.output_folder / "route_001_sadece_viraj" / relative).exists():
            raise AssertionError(f"missing offline artifact: {relative}")
    ca_metrics = json.loads(
        (result.output_folder / "route_001_sadece_viraj" / "metrics" / "ca_kf_metrics.json").read_text(encoding="utf-8")
    )
    route_metadata = json.loads(
        (result.output_folder / "route_001_sadece_viraj" / "metadata.json").read_text(encoding="utf-8")
    )
    if route_metadata.get("filter_tunes", {}).get("ca_kf", {}).get("gnss_position_stddev_m") != 2.75:
        raise AssertionError("offline replay route metadata did not preserve requested CA-KF tune")
    if ca_metrics.get("eval_position_rmse_m") is None:
        raise AssertionError("CA-KF eval RMSE missing")
    if "full_position_rmse_m" not in ca_metrics:
        raise AssertionError("CA-KF full RMSE missing")
    aggregate = json.loads((result.output_folder / "aggregate_summary.json").read_text(encoding="utf-8"))
    if aggregate.get("best_position_rmse_m") != ca_metrics.get("eval_position_rmse_m"):
        raise AssertionError("aggregate best RMSE does not use eval_position_rmse_m")


def test_localization_metrics_split_full_and_eval_windows() -> None:
    samples = [
        {"position_error_m": 20.0, "timestamp": 0.0, "valid_for_metrics": False, "dt": 0.0},
        {"position_error_m": 1.0, "timestamp": 1.0, "valid_for_metrics": True, "dt": 1.0},
        {"position_error_m": 2.0, "timestamp": 2.0, "valid_for_metrics": True, "dt": 1.0},
    ]
    metrics = compute_localization_metrics(samples, raw_gnss_rmse_m=3.0, divergence_error_threshold_m=10.0)
    if metrics["full_position_rmse_m"] <= metrics["eval_position_rmse_m"]:
        raise AssertionError("full RMSE should include the startup spike")
    if metrics["warmup_excluded_sample_count"] != 1:
        raise AssertionError("warm-up exclusion count incorrect")
    if metrics["divergence_event_count"] != 0:
        raise AssertionError("divergence should be counted on eval samples only")


def test_offline_replay_errors_use_sensor_timestamp_ground_truth() -> None:
    rows = [
        {
            "timestamp": 0.0,
            "gnss_timestamp": 0.0,
            "ground_truth_x": 0.0,
            "ground_truth_y": 0.0,
            "ground_truth_yaw": 0.0,
            "ground_truth_speed": 10.0,
            "gnss_local_x": 0.0,
            "gnss_local_y": 0.0,
            "valid_for_metrics": True,
        },
        {
            "timestamp": 1.0,
            "gnss_timestamp": 0.0,
            "ground_truth_x": 10.0,
            "ground_truth_y": 0.0,
            "ground_truth_yaw": 0.0,
            "ground_truth_speed": 10.0,
            "gnss_local_x": 0.0,
            "gnss_local_y": 0.0,
            "valid_for_metrics": True,
        },
    ]
    _annotate_metric_ground_truth(rows)
    estimates = _raw_gnss_estimates(rows)
    if estimates[1]["position_error_m"] != 0.0:
        raise AssertionError("raw GNSS was scored against sample-time GT instead of sensor-time GT")
    if estimates[1]["sensor_time_offset_s"] != 1.0:
        raise AssertionError("sensor time offset was not recorded")


def test_old_logs_without_valid_for_metrics_use_timestamp_fallback(tmp_path: Path) -> None:
    route = _short_saved_route()
    output_root = tmp_path / "benchmark_results"
    route_folder = output_root / "offline_localization" / "recordings" / "20260605_000001" / "route_001_old_log"
    route_folder.mkdir(parents=True)
    log_path = route_folder / "sensor_log.csv"
    _write_old_synthetic_log(log_path, route.name, route.map_name or "Carla/Maps/Town01_Opt")
    write_json(route_folder / "route_metadata.json", {"route": route.to_dict(), "map_name": route.map_name})
    write_json(route_folder / "recording_summary.json", {"route_name": route.name, "map_name": route.map_name, "sample_count": 140})

    result = OfflineReplayRunner().run(
        OfflineReplayRequest(
            sensor_log_paths=(log_path,),
            selected_filter_ids=("cv_kf",),
            output_root=str(output_root),
        )
    )
    metadata = json.loads((result.output_folder / "route_001_sadece_viraj" / "metadata.json").read_text(encoding="utf-8"))
    policy = metadata.get("warmup_exclusion_policy") or {}
    if policy.get("valid_for_metrics_source") != "timestamp_fallback":
        raise AssertionError("old log did not use timestamp fallback")
    if not policy.get("warnings"):
        raise AssertionError("old log fallback warning missing")


def test_auto_tune_replay_context_uses_compact_route_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "benchmark_results"
    long_route_name = "very_long_recorded_route_name_" + "segment_" * 12
    route_folder = output_root / "offline_localization" / "recordings" / "run_001" / "route_001_long"
    route_folder.mkdir(parents=True)
    log_path = route_folder / "sensor_log.csv"
    _write_synthetic_log(log_path, long_route_name, "Carla/Maps/Town01_Opt")
    write_json(
        route_folder / "route_metadata.json",
        {
            "route": {"name": long_route_name, "map_name": "Carla/Maps/Town01_Opt"},
            "map_name": "Carla/Maps/Town01_Opt",
            "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
            "sensor_noise_config": {"preset_name": "Synthetic"},
            "vehicle_behavior_config": {"preset_name": "Synthetic"},
        },
    )
    write_json(
        route_folder / "recording_summary.json",
        {
            "route_name": long_route_name,
            "map_name": "Carla/Maps/Town01_Opt",
            "sample_count": 24,
            "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
        },
    )
    run_folder = output_root / "offline_localization" / "auto_tune" / "ca_kf" / "a_test" / "t001"
    result = OfflineReplayRunner().run(
        OfflineReplayRequest(
            sensor_log_paths=(log_path,),
            selected_filter_ids=("ca_kf",),
            output_root=str(output_root),
            run_folder_override=run_folder,
            generate_plots=False,
            replay_context="auto_tune_trial",
        )
    )
    if result.output_folder != run_folder:
        raise AssertionError("auto-tune replay did not use requested compact output folder")
    route_output = run_folder / "r001"
    if not (route_output / "res" / "raw_gnss_estimates.csv").exists():
        raise AssertionError("auto-tune replay did not write compact result directory")
    if not (route_output / "met" / "summary_metrics.json").exists():
        raise AssertionError("auto-tune replay did not write compact metrics directory")
    for forbidden in ("replay_results", "metrics", "plots"):
        if (route_output / forbidden).exists():
            raise AssertionError(f"auto-tune replay should not create verbose {forbidden} directory")
    if any(path.name.startswith("route_001_") for path in run_folder.iterdir() if path.is_dir()):
        raise AssertionError("auto-tune replay used descriptive route folder name")
    metadata = json.loads((route_output / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("route_name") != long_route_name:
        raise AssertionError("compact auto-tune route metadata lost the descriptive route name")
    if metadata.get("artifact_layout", {}).get("compact") is not True:
        raise AssertionError("compact auto-tune route metadata did not describe compact artifact layout")


def test_windows_path_length_guard_has_clear_error(tmp_path: Path) -> None:
    original_os_name = offline_replay_runner_module.os.name
    offline_replay_runner_module.os.name = "nt"
    try:
        try:
            offline_replay_runner_module._validate_windows_path_length(tmp_path / ("x" * 260), "Synthetic output")
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("Windows path length guard did not raise for a long path")
    finally:
        offline_replay_runner_module.os.name = original_os_name
    if "likely too long for Windows" not in message or "shorter directory" not in message:
        raise AssertionError(f"Windows path length guard message is not clear: {message}")


def test_offline_recording_does_not_feed_filter_control() -> None:
    app_source = (PROJECT_ROOT / "src" / "core" / "app.py").read_text(encoding="utf-8")
    state_start = app_source.index("def _state_for_tracking_and_control")
    state_end = app_source.index("    def _set_latest_control", state_start)
    state_body = app_source[state_start:state_end]
    if "if self._offline_recording_active()" not in state_body:
        raise AssertionError("offline recording does not override control state")
    if "return self._latest_ground_truth_state" not in state_body:
        raise AssertionError("offline recording does not use ground truth for control")

    if 'source="autonomous_applied"' not in app_source:
        raise AssertionError("autonomous control feed path not found")
    guard = "if not self._offline_recording_active():\n                            self._feed_filter_control_input"
    if guard not in app_source:
        raise AssertionError("filter control input is not guarded during offline recording")


def test_offline_recording_warmup_does_not_trigger_route_failure() -> None:
    app_source = (PROJECT_ROOT / "src" / "core" / "app.py").read_text(encoding="utf-8")
    start = app_source.index("def _update_offline_recording")
    end = app_source.index("    def _current_world_frame", start)
    body = app_source[start:end]
    if "if not recorder.controller_enabled:" not in body:
        raise AssertionError("offline recording does not guard warm-up from route failure checks")
    warmup_guard = body[body.index("if not recorder.controller_enabled:") : body.index("failure_reason", body.index("if not recorder.controller_enabled:"))]
    if "_reset_benchmark_failure_monitor()" not in warmup_guard:
        raise AssertionError("offline warm-up does not reset the route failure monitor")
    if "route_failed=False" not in warmup_guard:
        raise AssertionError("offline warm-up can still mark route_failed")


def test_startup_gui_uses_mode_based_tabs() -> None:
    startup_source = (PROJECT_ROOT / "src" / "visualization" / "startup_map_selector.py").read_text(encoding="utf-8")
    if TOP_LEVEL_TABS != ("Demo", "Closed Loop Benchmark", "Offline Localization Benchmark"):
        raise AssertionError(f"unexpected top-level tabs: {TOP_LEVEL_TABS}")
    if CLOSED_LOOP_SUBTABS != ("Filters", "Sensor Noise", "Vehicle Behavior", "Routes"):
        raise AssertionError(f"unexpected closed-loop subtabs: {CLOSED_LOOP_SUBTABS}")
    if OFFLINE_SUBTABS != ("Record Sensor Data", "Test Setup"):
        raise AssertionError(f"unexpected offline subtabs: {OFFLINE_SUBTABS}")
    if OFFLINE_TEST_SETUP_SUBTABS != ("Select Route", "Filters"):
        raise AssertionError(f"unexpected offline setup subtabs: {OFFLINE_TEST_SETUP_SUBTABS}")
    for text in ("Record Selected Route Log", "Run Offline Localization Benchmark", "Raw GNSS is always included as the baseline."):
        if text not in startup_source:
            raise AssertionError(f"startup setup missing {text!r}")
    for removed in ('"Map Selection"', '"Evaluation Mode"', '"Offline Replay"'):
        if removed in startup_source:
            raise AssertionError(f"startup setup still exposes removed label {removed}")


def test_closed_loop_tracking_buttons_visible_with_long_filter_info() -> None:
    pygame.init()
    selector = StartupMapSelector.__new__(StartupMapSelector)
    selector._surface = pygame.Surface((760, 420))
    selector._init_fonts()
    selector._tracking_mode = TRACKING_MODE_PASSIVE
    selector._tracking_button_rects = {}
    selector._setup_filter_buttons = {}
    long_info = {
        "model_type": "CTRA",
        "type": "Unscented Kalman Filter",
        "state_vector": "[px, py, yaw, speed, acceleration, yaw_rate]^T",
        "process_model": "Constant Turn Rate and Acceleration with a deliberately long explanation",
        "measurement_model": "GNSS position x/y plus IMU yaw and yaw-rate with a deliberately long explanation",
        "autonomous_control_note": "Long note that previously pushed the tracking controls below the visible panel.",
    }
    record = SimpleNamespace(
        filter_id="ctra_ukf",
        display_name="CTRA UKF",
        filter_info=long_info,
        safe_for_autonomous_control=True,
        active_tracking_supported=True,
        benchmark_selectable=True,
        experimental=True,
    )
    selector._setup_filter_records = [record]
    selector._selected_filter_id = "ctra_ukf"

    selector._draw_closed_loop_filter_selection(pygame.Rect(0, 0, 760, 220))

    if TRACKING_MODE_PASSIVE not in selector._tracking_button_rects or TRACKING_MODE_ACTIVE not in selector._tracking_button_rects:
        raise AssertionError(f"tracking buttons were not drawn: {selector._tracking_button_rects}")
    for mode, rect in selector._tracking_button_rects.items():
        if rect.top < 0 or rect.bottom > 220:
            raise AssertionError(f"{mode} tracking button is outside the panel: {rect}")


def test_startup_tune_storage_feeds_closed_loop_and_offline_requests() -> None:
    selector = StartupMapSelector.__new__(StartupMapSelector)
    selector._setup_filter_records = [record for record in discover_filters() if record.valid]
    selector._selected_filter_tunes = {
        "ca_kf": {"gnss_position_stddev_m": 3.21},
        "ctra_ekf": {"process_jerk_stddev_mps3": 4.56},
    }
    selector._filter_tune_editor = None
    selector._filter_tune_editor_filter_id = ""
    selector._selected_filter_id = "ca_kf"
    selector._recommendation_applied_by_filter = {}

    closed_loop_tune = selector._current_filter_tune_values()
    if closed_loop_tune.get("gnss_position_stddev_m") != 3.21:
        raise AssertionError("closed-loop selected filter tune did not come from startup storage")
    config = BenchmarkConfig(
        selected_filter="ca_kf",
        selected_routes=(_short_saved_route(),),
        sensor_noise_config=sensor_noise_config_from_values(SENSOR_NOISE_PRESETS["Medium Noise"], preset_name="Medium Noise"),
        vehicle_behavior_config=BEHAVIOR_PRESETS["Balanced"],
        selected_filter_tune=closed_loop_tune,
        tracking_mode=TRACKING_MODE_PASSIVE,
    )
    if config.selected_filter_tune.get("gnss_position_stddev_m") != 3.21:
        raise AssertionError("closed-loop BenchmarkConfig did not receive GUI-edited tune values")

    offline_tunes = selector._included_offline_filter_tunes(("ca_kf", "ctra_ekf", "raw_gnss"))
    if offline_tunes.get("ca_kf", {}).get("gnss_position_stddev_m") != 3.21:
        raise AssertionError("offline CA-KF tune was not preserved")
    if offline_tunes.get("ctra_ekf", {}).get("process_jerk_stddev_mps3") != 4.56:
        raise AssertionError("offline CTRA-EKF tune was not preserved")
    if "raw_gnss" in offline_tunes:
        raise AssertionError("raw_gnss should not require startup tune values")


def test_registry_reads_optional_auto_tune_profile() -> None:
    records = {record.filter_id: record for record in discover_filters() if record.valid}
    ca = records.get("ca_kf")
    raw = records.get("raw_gnss")
    if ca is None or not ca.auto_tune_enabled:
        raise AssertionError("CA-KF should expose AUTO_TUNE_PROFILE through registry")
    if not ca.auto_tune_profile or not ca.auto_tune_profile.get("primary"):
        raise AssertionError("CA-KF auto tune primary parameters missing")
    if raw is None or raw.auto_tune_profile is not None or raw.auto_tune_enabled:
        raise AssertionError("raw_gnss should remain valid but not auto-tuneable")


def test_type_aware_nis_and_position_nees_helpers() -> None:
    nis = summarize_nis_by_type(
        [
            {"nis_by_type": {"gnss_position": 2.0, "imu_yaw": 1.0}},
            {"nis_by_type": json.dumps({"gnss_position": 6.0, "imu_yaw_rate": 4.0})},
        ]
    )
    if nis["gnss_position"]["mean"] != 4.0 or nis["gnss_position"]["sample_count"] != 2:
        raise AssertionError("GNSS position NIS was not summarized separately")
    if nis["gnss_position"]["expected_dimension"] != 2:
        raise AssertionError("GNSS position NIS expected dimension was not reported")
    if nis["imu_yaw"]["expected_dimension"] != 1 or nis["imu_yaw_rate"]["expected_dimension"] != 1:
        raise AssertionError("IMU scalar NIS dimensions were not reported")

    full = position_nees(1.0, 2.0, position_covariance_2x2=((4.0, 1.0), (1.0, 9.0)), covariance_diagonal=(4.0, 9.0))
    expected_full = (9.0 - 4.0 + 16.0) / 35.0
    if not math.isclose(float(full["position_nees"]), expected_full, rel_tol=1.0e-9):
        raise AssertionError("full 2x2 position NEES was not computed with covariance cross-term")
    if full["position_nees_source"] != "full_2x2" or full["position_nees_diagonal_approx"] is not None:
        raise AssertionError("full covariance NEES source was not labeled correctly")

    approx = position_nees(1.0, 2.0, covariance_diagonal=(4.0, 9.0))
    if approx["position_nees"] is not None or approx["position_nees_source"] != "diagonal_approx":
        raise AssertionError("diagonal fallback was not clearly labeled as approximate")

    report = consistency_report_from_summaries(
        nis_by_type_summary={
            "gnss_position": {"mean": 2.0, "sample_count": 5, "expected_dimension": 2},
            "imu_yaw_rate": {"mean": 8.0, "sample_count": 5, "expected_dimension": 1},
        },
        mean_position_nees=0.2,
        position_nees_source="full_2x2",
    )
    if report["nis_by_type"]["gnss_position"]["status"] != "good":
        raise AssertionError("GNSS NIS status should be good near its expected dimension")
    if report["nis_by_type"]["imu_yaw_rate"]["status"] != "severe":
        raise AssertionError("IMU yaw-rate NIS status should flag severe overconfidence")
    if report["position_nees"]["behavior"] != "underconfident":
        raise AssertionError("low position NEES should explain underconfidence")


def test_auto_tune_objective_modes_apply_consistency_dimension_penalties() -> None:
    good_metrics = {
        "mean_eval_position_rmse_m": 1.0,
        "mean_yaw_rmse_deg": 0.0,
        "divergence_event_count": 0,
        "nis_by_type_summary": {"gnss_position": {"mean": 2.0, "sample_count": 10, "expected_dimension": 2}},
        "mean_position_nees": 2.0,
        "position_nees_source": "full_2x2",
    }
    severe_metrics = {
        **good_metrics,
        "mean_eval_position_rmse_m": 0.9,
        "nis_by_type_summary": {"imu_yaw_rate": {"mean": 10.0, "sample_count": 10, "expected_dimension": 1}},
        "mean_position_nees": 12.0,
    }
    if objective_score(severe_metrics, "min_eval_rmse") >= objective_score(good_metrics, "min_eval_rmse"):
        raise AssertionError("pure RMSE objective should prefer the lower-RMSE candidate")
    for mode in ("min_rmse_with_consistency_guard", "consistency_first", "balanced_score"):
        if objective_score(severe_metrics, mode) <= objective_score(good_metrics, mode):
            raise AssertionError(f"{mode} should penalize severe type-aware NIS/NEES inconsistency")


def test_localization_metrics_expose_type_aware_nis_and_corrected_nees() -> None:
    metrics = compute_localization_metrics(
        [
            {
                "valid_for_metrics": True,
                "position_error_m": 1.0,
                "nis": 99.0,
                "nis_by_type": {"gnss_position": 2.0, "imu_yaw": 1.0},
                "nees": 3.0,
                "position_nees": 2.5,
                "position_nees_source": "full_2x2",
            }
        ]
    )
    if metrics["legacy_mean_nis_mixed"] != 99.0:
        raise AssertionError("legacy mixed NIS scalar was not preserved")
    if metrics["nis_by_type_summary"]["gnss_position"]["mean"] != 2.0:
        raise AssertionError("type-aware NIS summary was not exposed")
    if metrics["mean_position_nees"] != 2.5 or metrics["position_nees_source"] != "full_2x2":
        raise AssertionError("corrected position NEES summary was not exposed")


def test_sensor_noise_locking_removes_r_keys_from_process_search() -> None:
    profile = {
        "enabled": True,
        "primary": [
            {"key": "process_jerk_stddev_mps3", "min": 0.1, "max": 5.0},
            {"key": "gnss_position_stddev_m", "min": 0.4, "max": 6.0},
            {"key": "imu_accel_stddev_mps2", "min": 0.05, "max": 3.0},
        ],
    }
    process_only = process_only_auto_tune_profile(profile)
    keys = {item["key"] for item in process_only["primary"]}
    if keys != {"process_jerk_stddev_mps3"}:
        raise AssertionError(f"process-only profile still contains measurement-noise keys: {keys}")

    locked = SensorNoiseTuneMapper.locked_values(
        "ca_kf",
        {"gnss_position_stddev_m": 5.0, "imu_accel_stddev_mps2": 5.0, "process_jerk_stddev_mps3": 1.2},
        {"preset_name": "Synthetic", "gnss_position_stddev_m": 1.25, "imu_accel_stddev_mps2": 0.45},
    )
    if locked.values.get("gnss_position_stddev_m") != 1.25 or locked.values.get("imu_accel_stddev_mps2") != 0.45:
        raise AssertionError("sensor-noise tune values were not locked from the selected profile")
    merged = SensorNoiseTuneMapper.apply_locked_values("ca_kf", {"gnss_position_stddev_m": 5.0}, locked.representative_config)
    if merged.get("gnss_position_stddev_m") != 1.25:
        raise AssertionError("locked sensor noise was not applied to the tune dictionary")


def test_parameter_editor_mousewheel_uses_mouse_position_without_event_pos() -> None:
    pygame.font.init()
    surface = pygame.Surface((360, 160))
    specs = tuple(
        ParameterSpec(f"value_{index}", f"Value {index}", 0.0, 10.0, "", 1, "Group")
        for index in range(18)
    )
    editor = ParameterEditor(specs=specs, values={}, presets={}, active_preset="Custom", title="Many Values")
    editor.draw(surface, pygame.Rect(0, 0, 360, 120))
    before = editor._scroll_y
    event = pygame.event.Event(pygame.MOUSEWHEEL, {"y": -1})
    if not editor.handle_event(event):
        raise AssertionError("ParameterEditor did not consume MOUSEWHEEL without event.pos")
    if editor._scroll_y <= before:
        raise AssertionError("ParameterEditor did not scroll downward")
    for _ in range(100):
        editor.handle_event(event)
    if editor._scroll_y > editor._max_scroll_y():
        raise AssertionError("ParameterEditor scroll was not clamped")


def test_filter_auto_tuner_passes_candidate_tunes_and_saves_best_config(tmp_path: Path) -> None:
    output_root = tmp_path / "benchmark_results"
    logs = []
    for index in range(2):
        route_folder = output_root / "offline_localization" / "recordings" / "run_001" / f"route_{index + 1:03d}"
        route_folder.mkdir(parents=True)
        log_path = route_folder / "sensor_log.csv"
        log_path.write_text("timestamp\n0.0\n", encoding="utf-8")
        write_json(
            route_folder / "route_metadata.json",
            {
                "route": {"name": f"route_{index + 1}", "map_name": "Town01"},
                "map_name": "Town01",
                "sensor_noise_config": {"preset_name": "Synthetic", "gnss_position_stddev_m": 1.25},
                "vehicle_behavior_config": {"preset_name": "Balanced"},
                "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
            },
        )
        write_json(
            route_folder / "recording_summary.json",
            {
                "route_name": f"route_{index + 1}",
                "map_name": "Town01",
                "sample_count": 1,
                "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
            },
        )
        logs.append(log_path)

    calls: list[OfflineReplayRequest] = []

    class FakeRunner:
        def run(self, request: OfflineReplayRequest) -> SimpleNamespace:
            calls.append(request)
            trial_index = len(calls)
            if request.run_folder_override is None:
                raise AssertionError("auto tuner did not pass a trial output override")
            folder = Path(request.run_folder_override)
            folder.mkdir(parents=True)
            tune = request.filter_tunes["ca_kf"]
            rmse = float(tune.get("process_jerk_stddev_mps3", 9.0))
            write_json(
                folder / "aggregate_summary.json",
                {
                    "route_count": len(request.sensor_log_paths),
                    "aggregate_rows": [
                        {
                            "filter_id": "ca_kf",
                            "eval_position_rmse_m": rmse,
                            "yaw_rmse_deg": 0.0,
                            "divergence_event_count": 0,
                        }
                    ],
                },
            )
            return SimpleNamespace(output_folder=folder, failures=())

    request = AutoTuneRequest(
        filter_id="ca_kf",
        sensor_log_paths=tuple(logs),
        base_tune={"gnss_position_stddev_m": 1.25, "process_jerk_stddev_mps3": 1.2},
        auto_tune_profile={
            "enabled": True,
            "primary": [
                {"key": "process_jerk_stddev_mps3", "scale": "log", "min": 0.5, "max": 2.0},
                {"key": "gnss_position_stddev_m", "scale": "log", "min": 0.5, "max": 2.0},
            ],
            "search": {"default_trials": 2, "strategy": "random_plus_coordinate_refinement"},
            "objective": "rmse_consistency",
        },
        max_trials=2,
        output_root=str(output_root),
        keep_only_best_trial_output=True,
        generate_trial_plots=False,
    )
    result = FilterAutoTuner(runner_factory=FakeRunner).run(request)
    if not calls:
        raise AssertionError("auto tuner did not run OfflineReplayRunner")
    for call in calls:
        if call.sensor_log_paths != tuple(logs):
            raise AssertionError("auto tuner did not pass all selected logs")
        if call.selected_filter_ids != ("ca_kf",):
            raise AssertionError("auto tuner should tune exactly one filter")
        if "ca_kf" not in call.filter_tunes:
            raise AssertionError("candidate tune was not passed through filter_tunes")
        if call.filter_tunes["ca_kf"].get("gnss_position_stddev_m") != 1.25:
            raise AssertionError("auto tuner changed locked GNSS measurement noise")
        if call.replay_context != "auto_tune_trial":
            raise AssertionError("auto tuner did not mark replay context")
        if call.generate_plots:
            raise AssertionError("auto-tune trials should disable replay plots by default")
        path_text = str(call.run_folder_override).replace("\\", "/")
        if "/_tmp/at/" not in path_text or not path_text.rsplit("/", 1)[-1].startswith("t"):
            raise AssertionError(f"auto-tune trial output override did not use compact staging: {call.run_folder_override}")
        metrics_path = Path(call.run_folder_override) / "r001" / "met" / "summary_metrics.json"
        if len(str(metrics_path.resolve())) >= offline_replay_runner_module.WINDOWS_PATH_LENGTH_GUARD:
            raise AssertionError(f"auto-tune trial metrics path is too long: {metrics_path}")
        run_id = Path(call.run_folder_override).parent.name
        if not run_id.startswith("at") or len(run_id) > 18:
            raise AssertionError(f"auto-tune staging run id is not compact: {run_id}")
    evaluations = output_root / "offline_localization" / "evaluations"
    if evaluations.exists() and any(evaluations.iterdir()):
        raise AssertionError("auto tuner polluted normal offline evaluations output")
    if result.saved_config_path is None or not result.saved_config_path.exists():
        raise AssertionError("best tune config was not saved")
    run_path_text = str(result.output_folder).replace("\\", "/")
    if "/_at/o/ca_kf/" not in run_path_text:
        raise AssertionError(f"offline auto-tune final output did not use compact physical folder: {result.output_folder}")
    summary_path = result.output_folder / "auto_tune_summary.json"
    if len(str(summary_path.resolve())) >= offline_replay_runner_module.WINDOWS_PATH_LENGTH_GUARD:
        raise AssertionError(f"offline auto-tune summary path is too long: {summary_path}")
    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    if config.get("filter_id") != "ca_kf" or "best_tune" not in config:
        raise AssertionError("saved best tune config is incomplete")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError("saved best tune config did not use schema version 2")
    if config.get("benchmark_mode") != BENCHMARK_MODE_OFFLINE or config.get("tracking_mode") != TRACKING_PASSIVE:
        raise AssertionError("offline auto tuner did not save an offline passive tune config")
    if config.get("tune_scope") != TUNE_SCOPE_OFFLINE or not config.get("process_only_tune"):
        raise AssertionError("offline auto tuner did not mark process-only offline tuning scope")
    if config.get("locked_sensor_noise_values", {}).get("gnss_position_stddev_m") != 1.25:
        raise AssertionError("saved config did not record locked GNSS measurement noise")
    saved_primary_keys = {item.get("key") for item in (config.get("auto_tune_profile") or {}).get("primary", [])}
    if "gnss_position_stddev_m" in saved_primary_keys:
        raise AssertionError("schema v2 saved process-only search profile still includes GNSS measurement noise")
    if len(config.get("selected_logs") or []) != 2:
        raise AssertionError("saved best tune config did not preserve selected logs")
    best_output = str((config.get("best_metrics") or {}).get("output_folder") or "").replace("\\", "/")
    if "/_tmp/at/" not in best_output or not best_output.rsplit("/", 1)[-1].startswith("t"):
        raise AssertionError("best tune metadata did not reference auto_tune trial output")
    if not str(config.get("noise_signature") or ""):
        raise AssertionError("saved config did not preserve full noise_signature")
    if config.get("output_folder") != str(result.output_folder) or config.get("physical_output_folder") != str(result.output_folder):
        raise AssertionError("saved config did not record compact physical output folder")
    if "offline_localization/auto_tune/offline_passive/ca_kf/" not in str(config.get("logical_output_group") or ""):
        raise AssertionError("saved config did not record logical offline output group")
    listed = filter_auto_tuner_module.list_saved_tune_configs(
        "ca_kf",
        output_root=str(output_root),
        context=offline_tune_context("ca_kf", sensor_noise_config=config.get("representative_sensor_noise_config")),
    )
    if not any(str(item.get("path")) == str(result.saved_config_path) for item in listed):
        raise AssertionError(f"saved tune browser did not find compact offline config: {listed}")
    for key in ("objective_name", "score_formula", "score_notes", "nis_nees_policy", "unavailable_metrics_policy"):
        if not config.get(key):
            raise AssertionError(f"saved best tune config missing objective metadata: {key}")
    summary = json.loads((result.output_folder / "auto_tune_summary.json").read_text(encoding="utf-8"))
    staging_folder = str((summary.get("metadata") or {}).get("offline_candidate_staging_folder") or "").replace("\\", "/")
    if "/_tmp/at/" not in staging_folder:
        raise AssertionError("auto-tune summary did not record compact candidate staging folder")
    if summary.get("trial_output_policy", {}).get("normal_evaluations_directory_used") is not False:
        raise AssertionError("auto-tune summary did not document trial output policy")
    kept_outputs = [trial.output_folder for trial in result.trial_results if trial.output_folder is not None and trial.output_folder.exists()]
    if len(kept_outputs) != 1:
        raise AssertionError(f"keep_only_best_trial_output should keep exactly one replay output, got {kept_outputs}")
    if str(kept_outputs[0]).replace("\\", "/") != best_output:
        raise AssertionError("retention did not keep the best trial output")


def test_filter_auto_tuner_uses_compact_final_output_under_long_root(tmp_path: Path) -> None:
    output_root = tmp_path / "benchmark_results"
    old_run_name = "a260607_120000"
    noise_slug = "n_1234567890"
    while True:
        old_summary_path = (
            output_root
            / "offline_localization"
            / "auto_tune"
            / "offline_passive"
            / "ca_kf"
            / noise_slug
            / old_run_name
            / "auto_tune_summary.json"
        )
        compact_summary_path = output_root / "_at" / "o" / "ca_kf" / old_run_name / "auto_tune_summary.json"
        if len(str(old_summary_path.resolve())) >= offline_replay_runner_module.WINDOWS_PATH_LENGTH_GUARD:
            break
        output_root = output_root.parent / "long_project_segment" / "benchmark_results"
    if len(str(compact_summary_path.resolve())) >= offline_replay_runner_module.WINDOWS_PATH_LENGTH_GUARD:
        pytest.skip("temporary directory root is too long to test compact auto-tune outputs")

    route_folder = tmp_path / "logs" / "run_001" / "route_001"
    route_folder.mkdir(parents=True)
    log_path = route_folder / "sensor_log.csv"
    log_path.write_text("timestamp\n0.0\n", encoding="utf-8")
    sensor = {"preset_name": "Synthetic", "gnss_position_stddev_m": 1.25, "imu_accel_stddev_mps2": 0.45}
    write_json(route_folder / "route_metadata.json", {"map_name": "Town01", "sensor_noise_config": sensor})
    write_json(route_folder / "recording_summary.json", {"route_name": "route_001", "map_name": "Town01", "sample_count": 1})

    calls: list[OfflineReplayRequest] = []

    class FakeRunner:
        def run(self, request: OfflineReplayRequest) -> SimpleNamespace:
            calls.append(request)
            folder = Path(request.run_folder_override)
            folder.mkdir(parents=True, exist_ok=True)
            rmse = float(request.filter_tunes["ca_kf"].get("process_jerk_stddev_mps3", 1.2))
            write_json(
                folder / "aggregate_summary.json",
                {
                    "route_count": 1,
                    "aggregate_rows": [
                        {
                            "filter_id": "ca_kf",
                            "eval_position_rmse_m": rmse,
                            "yaw_rmse_deg": 0.0,
                            "divergence_event_count": 0,
                        }
                    ],
                },
            )
            return SimpleNamespace(output_folder=folder, failures=())

    request = AutoTuneRequest(
        filter_id="ca_kf",
        sensor_log_paths=(log_path,),
        base_tune={"gnss_position_stddev_m": 1.25, "process_jerk_stddev_mps3": 1.2},
        auto_tune_profile={
            "enabled": True,
            "primary": [{"key": "process_jerk_stddev_mps3", "scale": "log", "min": 0.5, "max": 2.0}],
        },
        max_trials=1,
        output_root=str(output_root),
        generate_trial_plots=False,
        metadata={"candidate_generation_strategy": "random_plus_coordinate_refinement"},
    )
    result = FilterAutoTuner(runner_factory=FakeRunner).run(request)
    summary_path = result.output_folder / "auto_tune_summary.json"
    if "/_at/o/ca_kf/" not in str(summary_path).replace("\\", "/"):
        raise AssertionError(f"auto-tune summary was not written under compact output root: {summary_path}")
    if len(str(summary_path.resolve())) >= offline_replay_runner_module.WINDOWS_PATH_LENGTH_GUARD:
        raise AssertionError(f"compact auto-tune summary path is too long: {summary_path}")
    trial_metrics_path = Path(calls[0].run_folder_override) / "r001" / "met" / "summary_metrics.json"
    if len(str(trial_metrics_path.resolve())) >= offline_replay_runner_module.WINDOWS_PATH_LENGTH_GUARD:
        raise AssertionError(f"compact trial staging path is too long: {trial_metrics_path}")
    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    if config.get("benchmark_mode") != BENCHMARK_MODE_OFFLINE or config.get("tracking_mode") != TRACKING_PASSIVE:
        raise AssertionError("compact offline output did not preserve offline passive metadata")
    if config.get("noise_signature") != noise_signature(sensor):
        raise AssertionError("compact offline output did not preserve full noise signature")
    if config.get("representative_sensor_noise_config") != sensor:
        raise AssertionError("compact offline output did not preserve representative sensor config")
    if "offline_localization/auto_tune/offline_passive/ca_kf/" not in str(config.get("logical_output_group") or ""):
        raise AssertionError("compact offline output did not record logical output group")
    listed = filter_auto_tuner_module.list_saved_tune_configs(
        "ca_kf",
        output_root=str(output_root),
        context=offline_tune_context("ca_kf", sensor_noise_config=sensor),
    )
    if not any(str(item.get("path")) == str(result.saved_config_path) for item in listed):
        raise AssertionError("saved tune browser did not find compact offline config under long output root")


def test_filter_auto_tuner_reports_no_improvement_and_does_not_save_when_candidates_are_worse(tmp_path: Path) -> None:
    output_root = tmp_path / "benchmark_results"
    route_folder = output_root / "offline_localization" / "recordings" / "run_001" / "route_001"
    route_folder.mkdir(parents=True)
    log_path = route_folder / "sensor_log.csv"
    log_path.write_text("timestamp\n0.0\n", encoding="utf-8")
    write_json(
        route_folder / "route_metadata.json",
        {
            "route": {"name": "route_1", "map_name": "Town01"},
            "map_name": "Town01",
            "sensor_noise_config": {"preset_name": "Synthetic", "gnss_position_stddev_m": 1.25},
            "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
        },
    )
    write_json(route_folder / "recording_summary.json", {"route_name": "route_1", "map_name": "Town01", "sample_count": 1})

    class FakeRunner:
        def run(self, request: OfflineReplayRequest) -> SimpleNamespace:
            folder = Path(request.run_folder_override)
            folder.mkdir(parents=True)
            process_noise = float(request.filter_tunes["ca_kf"].get("process_jerk_stddev_mps3", 1.2))
            rmse = 1.0 if math.isclose(process_noise, 1.2, rel_tol=1.0e-12, abs_tol=1.0e-12) else 1.3
            write_json(
                folder / "aggregate_summary.json",
                {
                    "route_count": 1,
                    "aggregate_rows": [
                        {
                            "filter_id": "ca_kf",
                            "route_name": "route_1",
                            "eval_position_rmse_m": rmse,
                            "yaw_rmse_deg": 0.0,
                            "divergence_event_count": 0,
                            "nis_by_type_summary": {
                                "gnss_position": {"mean": 2.0, "sample_count": 1, "expected_dimension": 2}
                            },
                            "mean_position_nees": 2.0,
                            "position_nees_source": "full_2x2",
                        }
                    ],
                },
            )
            return SimpleNamespace(output_folder=folder, failures=())

    result = FilterAutoTuner(runner_factory=FakeRunner).run(
        AutoTuneRequest(
            filter_id="ca_kf",
            sensor_log_paths=(log_path,),
            base_tune={"gnss_position_stddev_m": 1.25, "process_jerk_stddev_mps3": 1.2},
            auto_tune_profile={"enabled": True, "primary": [{"key": "process_jerk_stddev_mps3", "min": 0.5, "max": 2.0}]},
            max_trials=2,
            output_root=str(output_root),
        )
    )
    if result.saved_config_path is not None or (result.output_folder / "best_tune.json").exists():
        raise AssertionError("auto tuner saved a best_tune.json even though every generated candidate was worse")
    if result.best_tune or result.best_score is not None or result.improved_over_baseline:
        raise AssertionError("auto tuner reported an improvement when baseline won verification")
    if result.recommendation_status != "no_improved_tune_found":
        raise AssertionError("auto tuner did not clearly report no improvement")
    for name in (
        "auto_tune_summary.json",
        "candidates.json",
        "trials.csv",
        "verification_leaderboard.csv",
        "per_route_metrics.csv",
        "consistency_report.json",
    ):
        if not (result.output_folder / name).exists():
            raise AssertionError(f"auto tuner did not write transparent artifact: {name}")
    summary = json.loads((result.output_folder / "auto_tune_summary.json").read_text(encoding="utf-8"))
    if summary.get("improved_over_baseline") is not False or summary.get("message") != "No improved tune found.":
        raise AssertionError("auto-tune summary did not record the no-improvement result")
    winner = summary.get("verification_winner") or {}
    if winner.get("candidate_type") not in {"default_base", "current_ui"}:
        raise AssertionError("verification should have selected the baseline when candidates were worse")


def test_filter_auto_tuner_rejects_candidate_that_fails_verification(tmp_path: Path) -> None:
    output_root = tmp_path / "benchmark_results"
    route_folder = output_root / "offline_localization" / "recordings" / "run_001" / "route_001"
    route_folder.mkdir(parents=True)
    log_path = route_folder / "sensor_log.csv"
    log_path.write_text("timestamp\n0.0\n", encoding="utf-8")
    write_json(
        route_folder / "route_metadata.json",
        {
            "route": {"name": "route_1", "map_name": "Town01"},
            "map_name": "Town01",
            "sensor_noise_config": {"preset_name": "Synthetic", "gnss_position_stddev_m": 1.25},
            "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
        },
    )
    write_json(route_folder / "recording_summary.json", {"route_name": "route_1", "map_name": "Town01", "sample_count": 1})
    evaluations_by_process_noise: dict[float, int] = {}

    class FakeRunner:
        def run(self, request: OfflineReplayRequest) -> SimpleNamespace:
            folder = Path(request.run_folder_override)
            folder.mkdir(parents=True)
            process_noise = round(float(request.filter_tunes["ca_kf"].get("process_jerk_stddev_mps3", 1.2)), 12)
            evaluations_by_process_noise[process_noise] = evaluations_by_process_noise.get(process_noise, 0) + 1
            if math.isclose(process_noise, 1.2, rel_tol=1.0e-12, abs_tol=1.0e-12):
                rmse = 1.0
            elif evaluations_by_process_noise[process_noise] == 1:
                rmse = 0.5
            else:
                rmse = 1.4
            write_json(
                folder / "aggregate_summary.json",
                {
                    "route_count": 1,
                    "aggregate_rows": [
                        {
                            "filter_id": "ca_kf",
                            "route_name": "route_1",
                            "eval_position_rmse_m": rmse,
                            "yaw_rmse_deg": 0.0,
                            "divergence_event_count": 0,
                        }
                    ],
                },
            )
            return SimpleNamespace(output_folder=folder, failures=())

    result = FilterAutoTuner(runner_factory=FakeRunner).run(
        AutoTuneRequest(
            filter_id="ca_kf",
            sensor_log_paths=(log_path,),
            base_tune={"gnss_position_stddev_m": 1.25, "process_jerk_stddev_mps3": 1.2},
            auto_tune_profile={"enabled": True, "primary": [{"key": "process_jerk_stddev_mps3", "min": 0.5, "max": 2.0}]},
            max_trials=1,
            output_root=str(output_root),
        )
    )
    search_results = [trial for trial in result.trial_results if trial.stage == "search"]
    verification_results = [trial for trial in result.verification_results if trial.candidate_type.startswith("generated")]
    if not search_results or not verification_results:
        raise AssertionError("test did not exercise both search and verification candidates")
    if float(search_results[0].score) >= 1.0:
        raise AssertionError("test setup expected the generated candidate to look better during search")
    if float(verification_results[0].score) <= 1.0:
        raise AssertionError("test setup expected the generated candidate to fail verification")
    if result.saved_config_path is not None or result.best_tune or result.improved_over_baseline:
        raise AssertionError("auto tuner recommended a tune that verification proved worse than baseline")


def test_filter_auto_tuner_rejects_mixed_noise_signatures_by_default(tmp_path: Path) -> None:
    output_root = tmp_path / "benchmark_results"
    logs = []
    for index, gnss_stddev in enumerate((1.25, 2.5)):
        route_folder = output_root / "offline_localization" / "recordings" / "run_001" / f"route_{index + 1:03d}"
        route_folder.mkdir(parents=True)
        log_path = route_folder / "sensor_log.csv"
        log_path.write_text("timestamp\n0.0\n", encoding="utf-8")
        write_json(
            route_folder / "route_metadata.json",
            {
                "route": {"name": f"route_{index + 1}", "map_name": "Town01"},
                "map_name": "Town01",
                "sensor_noise_config": {"preset_name": "Synthetic", "gnss_position_stddev_m": gnss_stddev},
                "vehicle_behavior_config": {"preset_name": "Balanced"},
                "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
            },
        )
        write_json(route_folder / "recording_summary.json", {"route_name": f"route_{index + 1}", "map_name": "Town01"})
        logs.append(log_path)

    request = AutoTuneRequest(
        filter_id="ca_kf",
        sensor_log_paths=tuple(logs),
        base_tune={"gnss_position_stddev_m": 1.25, "process_jerk_stddev_mps3": 1.2},
        auto_tune_profile={"enabled": True, "primary": [{"key": "process_jerk_stddev_mps3", "min": 0.5, "max": 2.0}]},
        max_trials=1,
        output_root=str(output_root),
    )
    with pytest.raises(ValueError, match="mixed sensor noise signatures"):
        FilterAutoTuner().run(request)


def test_filter_auto_tuner_uses_fallback_when_optuna_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "benchmark_results"
    route_folder = output_root / "offline_localization" / "recordings" / "run_001" / "route_001"
    route_folder.mkdir(parents=True)
    log_path = route_folder / "sensor_log.csv"
    log_path.write_text("timestamp\n0.0\n", encoding="utf-8")
    write_json(
        route_folder / "route_metadata.json",
        {
            "route": {"name": "route_1", "map_name": "Town01"},
            "map_name": "Town01",
            "sensor_noise_config": {"preset_name": "Synthetic", "gnss_position_stddev_m": 1.25},
            "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
        },
    )
    write_json(route_folder / "recording_summary.json", {"route_name": "route_1", "map_name": "Town01", "sample_count": 1})

    class FakeRunner:
        def run(self, request: OfflineReplayRequest) -> SimpleNamespace:
            folder = Path(request.run_folder_override)
            folder.mkdir(parents=True)
            write_json(
                folder / "aggregate_summary.json",
                {
                    "route_count": len(request.sensor_log_paths),
                    "aggregate_rows": [
                        {
                            "filter_id": "ca_kf",
                            "eval_position_rmse_m": float(request.filter_tunes["ca_kf"].get("process_jerk_stddev_mps3", 1.0)),
                            "yaw_rmse_deg": 0.0,
                            "divergence_event_count": 0,
                        }
                    ],
                },
            )
            return SimpleNamespace(output_folder=folder, failures=())

    monkeypatch.setattr(filter_auto_tuner_module, "optuna", None)
    result = FilterAutoTuner(runner_factory=FakeRunner).run(
        AutoTuneRequest(
            filter_id="ca_kf",
            sensor_log_paths=(log_path,),
            base_tune={"gnss_position_stddev_m": 1.25, "process_jerk_stddev_mps3": 1.2},
            auto_tune_profile={"enabled": True, "primary": [{"key": "process_jerk_stddev_mps3", "min": 0.5, "max": 2.0}]},
            max_trials=1,
            output_root=str(output_root),
        )
    )
    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    if config.get("candidate_generation_strategy") != "random_plus_coordinate_refinement":
        raise AssertionError("auto tuner did not record fallback search strategy when Optuna was unavailable")
    if config.get("optuna_available") is not False:
        raise AssertionError("saved config did not record Optuna availability correctly")


def test_saved_tune_apply_updates_only_selected_filter() -> None:
    selector = StartupMapSelector.__new__(StartupMapSelector)
    selector._setup_filter_records = [record for record in discover_filters() if record.valid]
    selector._selected_filter_tunes = {
        "ca_kf": {"gnss_position_stddev_m": 1.0},
        "ctra_ekf": {"process_jerk_stddev_mps3": 2.0},
    }
    selector._filter_tune_editor_filter_id = ""
    selector._filter_tune_editor = None
    selector._apply_tune_to_filter("ca_kf", {"gnss_position_stddev_m": 4.4})
    if selector._selected_filter_tunes["ca_kf"].get("gnss_position_stddev_m") != 4.4:
        raise AssertionError("saved tune did not update selected filter")
    if selector._selected_filter_tunes["ctra_ekf"].get("process_jerk_stddev_mps3") != 2.0:
        raise AssertionError("saved tune apply leaked into another filter")


def test_closed_loop_saved_tune_config_apply_updates_selected_filter() -> None:
    selector = StartupMapSelector.__new__(StartupMapSelector)
    selector._setup_filter_records = [record for record in discover_filters() if record.valid]
    selector._selected_filter_tunes = {
        "ca_kf": {"gnss_position_stddev_m": 1.0},
        "ctra_ekf": {"process_jerk_stddev_mps3": 2.0},
    }
    selector._filter_tune_editor_filter_id = ""
    selector._filter_tune_editor = None
    selector._closed_loop_saved_tune_status = ""
    selector._closed_loop_filter_saved_tune_mode = True
    selector._tracking_mode = TRACKING_MODE_PASSIVE
    selector._sensor_editor = None
    selector._sensor_preset = "Medium Noise"
    selector._behavior_editor = None
    selector._behavior_preset = "Balanced"
    selector._selected_recorded_log_index = None
    selector._recorded_logs = []
    config_path = Path("synthetic_auto_tune") / "best_tune.json"
    sensor_config = sensor_noise_config_from_values(SENSOR_NOISE_PRESETS["Medium Noise"], preset_name="Medium Noise")
    original_list = startup_map_selector_module.list_saved_tune_configs
    original_load = startup_map_selector_module.load_saved_tune_config
    startup_map_selector_module.list_saved_tune_configs = lambda filter_id, **kwargs: [{"path": str(config_path)}] if filter_id == "ca_kf" else []
    startup_map_selector_module.load_saved_tune_config = lambda path: {
        "schema_version": SCHEMA_VERSION,
        "filter_id": "ca_kf",
        "benchmark_mode": BENCHMARK_MODE_CLOSED_LOOP,
        "tracking_mode": TRACKING_PASSIVE,
        "tune_scope": TUNE_SCOPE_CLOSED_LOOP_VALIDATED,
        "noise_signature": noise_signature(sensor_config),
        "best_tune": {"gnss_position_stddev_m": 4.4},
    }
    try:
        selector._apply_saved_tune_config("ca_kf", 0, context="closed_loop")
    finally:
        startup_map_selector_module.list_saved_tune_configs = original_list
        startup_map_selector_module.load_saved_tune_config = original_load
    if selector._selected_filter_tunes["ca_kf"].get("gnss_position_stddev_m") != 4.4:
        raise AssertionError("closed-loop saved config did not update selected filter tune")
    if selector._selected_filter_tunes["ctra_ekf"].get("process_jerk_stddev_mps3") != 2.0:
        raise AssertionError("closed-loop saved config leaked into another filter")
    if selector._closed_loop_filter_saved_tune_mode:
        raise AssertionError("closed-loop saved tune browser did not close after applying")
    if "ca_kf" not in selector._closed_loop_saved_tune_status:
        raise AssertionError("closed-loop saved tune apply status did not name target filter")


def test_saved_tune_apply_rejects_incompatible_tracking_mode() -> None:
    selector = StartupMapSelector.__new__(StartupMapSelector)
    selector._setup_filter_records = [record for record in discover_filters() if record.valid]
    selector._selected_filter_tunes = {"ca_kf": {"gnss_position_stddev_m": 1.0}}
    selector._filter_tune_editor_filter_id = ""
    selector._filter_tune_editor = None
    selector._closed_loop_saved_tune_status = ""
    selector._closed_loop_filter_saved_tune_mode = True
    selector._tracking_mode = TRACKING_MODE_ACTIVE
    selector._sensor_editor = None
    selector._sensor_preset = "Medium Noise"
    selector._behavior_editor = None
    selector._behavior_preset = "Balanced"
    selector._selected_recorded_log_index = None
    selector._recorded_logs = []
    sensor_config = sensor_noise_config_from_values(SENSOR_NOISE_PRESETS["Medium Noise"], preset_name="Medium Noise")
    config_path = Path("synthetic_auto_tune") / "best_tune.json"
    original_list = startup_map_selector_module.list_saved_tune_configs
    original_load = startup_map_selector_module.load_saved_tune_config
    startup_map_selector_module.list_saved_tune_configs = lambda filter_id, **kwargs: [{"path": str(config_path)}] if filter_id == "ca_kf" else []
    startup_map_selector_module.load_saved_tune_config = lambda path: {
        "schema_version": SCHEMA_VERSION,
        "filter_id": "ca_kf",
        "benchmark_mode": BENCHMARK_MODE_CLOSED_LOOP,
        "tracking_mode": TRACKING_PASSIVE,
        "tune_scope": TUNE_SCOPE_CLOSED_LOOP_VALIDATED,
        "noise_signature": noise_signature(sensor_config),
        "best_tune": {"gnss_position_stddev_m": 4.4},
    }
    try:
        selector._apply_saved_tune_config("ca_kf", 0, context="closed_loop")
    finally:
        startup_map_selector_module.list_saved_tune_configs = original_list
        startup_map_selector_module.load_saved_tune_config = original_load
    if selector._selected_filter_tunes["ca_kf"].get("gnss_position_stddev_m") != 1.0:
        raise AssertionError("incompatible passive tune was applied in active tracking mode")
    if "tracking_mode" not in selector._closed_loop_saved_tune_status:
        raise AssertionError("incompatible apply rejection did not explain tracking mismatch")


def test_saved_tune_browse_filters_mode_tracking_noise_and_legacy(tmp_path: Path) -> None:
    output_root = tmp_path / "benchmark_results"
    sensor_config = {"preset_name": "Synthetic", "gnss_position_stddev_m": 1.25}
    other_sensor_config = {"preset_name": "Synthetic", "gnss_position_stddev_m": 2.5}
    signature = noise_signature(sensor_config)
    other_signature = noise_signature(other_sensor_config)
    offline_dir = output_root / "offline_localization" / "auto_tune" / "offline_passive" / "ca_kf"
    closed_passive_dir = output_root / "closed_loop" / "auto_tune" / "passive" / "ca_kf"
    closed_active_dir = output_root / "closed_loop" / "auto_tune" / "active" / "ca_kf"
    offline_dir.mkdir(parents=True)
    closed_passive_dir.mkdir(parents=True)
    closed_active_dir.mkdir(parents=True)

    def write_config(directory: Path, name: str, mode: str, tracking: str, scope: str, noise_sig: str) -> Path:
        config_path = directory / f"{name}.json"
        write_json(
            config_path,
            {
                "schema_version": SCHEMA_VERSION,
                "filter_id": "ca_kf",
                "benchmark_mode": mode,
                "tracking_mode": tracking,
                "tune_scope": scope,
                "noise_signature": noise_sig,
                "best_tune": {"process_jerk_stddev_mps3": 1.2},
            },
        )
        return config_path

    offline_path = write_config(offline_dir, "offline", BENCHMARK_MODE_OFFLINE, TRACKING_PASSIVE, TUNE_SCOPE_OFFLINE, signature)
    offline_other_noise_path = write_config(offline_dir, "offline_other_noise", BENCHMARK_MODE_OFFLINE, TRACKING_PASSIVE, TUNE_SCOPE_OFFLINE, other_signature)
    legacy_path = offline_dir / "legacy.json"
    write_json(legacy_path, {"filter_id": "ca_kf", "best_tune": {"process_jerk_stddev_mps3": 9.9}})
    closed_passive_path = write_config(
        closed_passive_dir,
        "closed_passive",
        BENCHMARK_MODE_CLOSED_LOOP,
        TRACKING_PASSIVE,
        TUNE_SCOPE_CLOSED_LOOP_VALIDATED,
        signature,
    )
    closed_active_path = write_config(
        closed_active_dir,
        "closed_active",
        BENCHMARK_MODE_CLOSED_LOOP,
        TRACKING_ACTIVE,
        TUNE_SCOPE_CLOSED_LOOP_VALIDATED,
        signature,
    )
    write_json(
        offline_dir / "saved_tune_configs.json",
        {
            "configs": [
                {"path": str(offline_path), "score": 1.0},
                {"path": str(offline_other_noise_path), "score": 2.0},
                {"path": str(legacy_path), "score": 3.0},
            ]
        },
    )
    write_json(closed_passive_dir / "saved_tune_configs.json", {"configs": [{"path": str(closed_passive_path), "score": 4.0}]})
    write_json(closed_active_dir / "saved_tune_configs.json", {"configs": [{"path": str(closed_active_path), "score": 5.0}]})

    offline_items = filter_auto_tuner_module.list_saved_tune_configs(
        "ca_kf",
        output_root=str(output_root),
        context=offline_tune_context("ca_kf", sensor_noise_config=sensor_config),
    )
    if [Path(str(item["path"])).name for item in offline_items] != ["offline.json"]:
        raise AssertionError(f"offline browse returned incompatible or legacy configs: {offline_items}")

    passive_items = filter_auto_tuner_module.list_saved_tune_configs(
        "ca_kf",
        output_root=str(output_root),
        context=closed_loop_tune_context("ca_kf", TRACKING_PASSIVE, sensor_noise_config=sensor_config),
    )
    if [Path(str(item["path"])).name for item in passive_items] != ["closed_passive.json"]:
        raise AssertionError("closed-loop passive browse did not isolate passive configs")

    active_items = filter_auto_tuner_module.list_saved_tune_configs(
        "ca_kf",
        output_root=str(output_root),
        context=closed_loop_tune_context("ca_kf", TRACKING_ACTIVE, sensor_noise_config=sensor_config),
    )
    if [Path(str(item["path"])).name for item in active_items] != ["closed_active.json"]:
        raise AssertionError("closed-loop active browse did not isolate active configs")

    incompatible = {
        "schema_version": SCHEMA_VERSION,
        "filter_id": "ca_kf",
        "benchmark_mode": BENCHMARK_MODE_CLOSED_LOOP,
        "tracking_mode": TRACKING_PASSIVE,
        "tune_scope": TUNE_SCOPE_CLOSED_LOOP_VALIDATED,
        "noise_signature": signature,
    }
    result = TuneCompatibility.check(incompatible, closed_loop_tune_context("ca_kf", TRACKING_ACTIVE, sensor_noise_config=sensor_config))
    if result.compatible:
        raise AssertionError("compatibility helper allowed passive config in active context")


def test_closed_loop_apply_recommended_still_updates_tune() -> None:
    pygame.font.init()
    selector = StartupMapSelector.__new__(StartupMapSelector)
    selector._setup_filter_records = [record for record in discover_filters() if record.valid]
    selector._selected_filter_id = "ca_kf"
    selector._selected_filter_tunes = {"ca_kf": {"gnss_position_stddev_m": 9.0, "process_jerk_stddev_mps3": 1.0}}
    selector._filter_tune_editor_filter_id = ""
    selector._filter_tune_editor = None
    selector._sensor_editor = None
    selector._sensor_preset = "Medium Noise"
    selector._tracking_mode = TRACKING_MODE_PASSIVE
    selector._recommendation_applied_by_filter = {}
    selector._apply_recommended_setup_tune("ca_kf")
    if not selector._recommendation_applied_by_filter.get("ca_kf"):
        raise AssertionError("closed-loop Apply Recommended did not mark recommendation as applied")
    if selector._selected_filter_tunes["ca_kf"].get("gnss_position_stddev_m") == 9.0:
        raise AssertionError("closed-loop Apply Recommended did not update CA-KF tune values")


def test_closed_loop_benchmark_config_rejects_multiple_routes() -> None:
    route_a = _short_saved_route()
    config = BenchmarkConfig(
        selected_filter="ca_kf",
        selected_routes=(route_a, route_a),
        sensor_noise_config=sensor_noise_config_from_values(SENSOR_NOISE_PRESETS["Medium Noise"], preset_name="Medium Noise"),
        vehicle_behavior_config=BEHAVIOR_PRESETS["Balanced"],
        selected_filter_tune={"process_jerk_stddev_mps3": 1.2},
        tracking_mode=TRACKING_MODE_PASSIVE,
    )
    errors = validate_benchmark_config(config, valid_filter_ids=("ca_kf",), available_maps=(route_a.map_name or "",))
    if not any("exactly one" in error for error in errors):
        raise AssertionError(f"multiple closed-loop routes were not rejected: {errors}")


def test_closed_loop_route_selection_stores_single_route() -> None:
    selector = StartupMapSelector.__new__(StartupMapSelector)
    selector._auto_tune_modal_open = False
    selector._closed_loop_subtab_rects = {}
    selector._active_closed_loop_subtab = "Routes"
    selector._closed_loop_filter_saved_tune_mode = False
    selector._setup_filter_buttons = {}
    selector._tracking_button_rects = {}
    selector._filter_tune_editor = None
    selector._sensor_editor = None
    selector._behavior_editor = None
    selector._select_all_routes_rect = pygame.Rect(0, 0, 1, 1)
    selector._clear_routes_rect = pygame.Rect(0, 0, 1, 1)
    selector._route_rects = {1: pygame.Rect(10, 10, 80, 20), 2: pygame.Rect(10, 40, 80, 20)}
    selector._selected_route_indices = {1, 2}
    selector._start_benchmark_rect = pygame.Rect(500, 500, 1, 1)
    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (20, 45)})
    selector._handle_closed_loop_event(event, client=None)
    if selector._selected_route_indices != {2}:
        raise AssertionError("closed-loop route click did not reduce selection to exactly one route")


def test_pending_closed_loop_auto_tune_session_saves_explicit_handoff(tmp_path: Path) -> None:
    handoff_path = tmp_path / "closed_loop" / "auto_tune" / "active" / "ca_kf" / "run_001" / "pending_session.json"
    session = PendingClosedLoopAutoTuneSession(
        selected_filter="ca_kf",
        tracking_mode=TRACKING_ACTIVE,
        offline_log_paths=("log_a.csv", "log_b.csv"),
        noise_signature="noise_sig",
        validation_route_name="route_1",
        validation_route_map="Town01",
        validation_route_id="route_1@Town01",
        sensor_config={"preset_name": "Synthetic"},
        vehicle_behavior_config={"preset_name": "Balanced"},
        actuator_realism_config={"enabled": True},
        trial_count=20,
        finalist_count=3,
        strategy="optuna_tpe",
        output_root=str(tmp_path),
    )
    session.save(handoff_path)
    payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    if payload.get("tracking_mode") != TRACKING_ACTIVE or payload.get("finalist_count") != 3:
        raise AssertionError("pending closed-loop auto tune handoff did not serialize required fields")
    if len(payload.get("offline_log_paths") or []) != 2:
        raise AssertionError("pending closed-loop auto tune handoff did not preserve offline log paths")


def test_closed_loop_route_data_survives_pending_session_and_app_reconstruction(tmp_path: Path) -> None:
    route = _short_saved_route()
    route_data = route.to_dict()
    session = PendingClosedLoopAutoTuneSession(
        selected_filter="ca_kf",
        tracking_mode=TRACKING_ACTIVE,
        offline_log_paths=("log_a.csv",),
        noise_signature="noise_sig",
        validation_route_name=route.name,
        validation_route_map=route.map_name or "",
        validation_route_id="route_identity",
        sensor_config={"preset_name": "Synthetic"},
        vehicle_behavior_config={"preset_name": "Balanced"},
        actuator_realism_config={"enabled": True},
        trial_count=5,
        finalist_count=1,
        strategy="optuna_tpe",
        output_root=str(tmp_path),
        validation_route_data=route_data,
    )
    request = ClosedLoopAutoTuneRequest.from_pending_session(
        session,
        auto_tune_profile={"enabled": True, "primary": [{"key": "process_jerk_stddev_mps3", "min": 0.5, "max": 2.0}]},
    )
    validation_route = request.validation_routes[0]
    if validation_route.route_data != route_data:
        raise AssertionError("validation route data was not preserved from pending session")
    if request.to_dict()["validation_routes"][0]["route_data"] != route_data:
        raise AssertionError("validation route data was not preserved in request serialization")

    from_dict_route = ClosedLoopValidationRoute.from_object({"route_data": route_data, "name": route.name, "map_name": route.map_name})
    if from_dict_route.route_data != route_data:
        raise AssertionError("ClosedLoopValidationRoute.from_object did not preserve route_data")

    app = SimulationApp.__new__(SimulationApp)
    app._test_route_store = None
    finalist = ClosedLoopFinalist(
        rank=1,
        candidate_tune={},
        offline_score=1.0,
        offline_metrics={},
        trial_index=1,
        source_output_folder=None,
    )
    validation_request = ClosedLoopValidationRequest(
        filter_id="ca_kf",
        tracking_mode=TRACKING_ACTIVE,
        finalist=finalist,
        validation_route=validation_route,
        sensor_noise_config={},
        vehicle_behavior_config={},
        actuator_realism_config={},
        output_folder=tmp_path,
    )
    reconstructed = app._saved_route_from_validation_request(validation_request)
    if reconstructed is None or reconstructed.to_dict() != route_data:
        raise AssertionError("app-side validation route reconstruction did not use preserved route_data")


def test_route_tab_lines_do_not_require_metrics() -> None:
    app = SimulationApp.__new__(SimulationApp)
    app._test_route_store = None
    app._map_selector = None
    app.route_planner = None
    app._active_map_name = "Town01"
    app._drive_mode = SimpleNamespace(value="MANUAL")
    app._map_selection_active = False
    app._test_route_authoring_active = False
    app._route_activation_state = SimpleNamespace(value="IDLE")
    app._planner_status = "Planner idle"
    app._control_status_text = "Control idle"
    lines = app._route_tab_lines()
    if not lines or lines[0] != "Route:":
        raise AssertionError("route tab lines did not render basic route state")


def test_live_evaluation_lines_do_not_require_position_nees() -> None:
    class EmptyLogger:
        current_raw_gnss_error_m = None
        current_position_error_m = None

        def running_sample_count(self, phases=None) -> int:
            return 0

        def running_metrics(self) -> dict[str, object]:
            return {
                "filtered_rmse_m": None,
                "raw_gnss_rmse_m": None,
                "mean_position_nees": None,
                "mean_position_nees_diagonal_approx": None,
            }

    app = SimulationApp.__new__(SimulationApp)
    app._active_performance_logger = lambda: EmptyLogger()
    app._filter_manager = None
    app._latest_ground_truth_state = None
    app._latest_estimated_state = None
    lines = app._live_evaluation_lines()
    if not any(line.startswith("Position NEES approx:") for line in lines):
        raise AssertionError(f"live evaluation lines did not render guarded NEES line: {lines}")


def test_closed_loop_auto_tune_builder_requires_one_validation_route(tmp_path: Path) -> None:
    selector = _closed_loop_autotune_selector(tmp_path)
    selector._closed_loop_auto_tune_validation_route_index = None
    try:
        selector._build_closed_loop_auto_tune_request_from_modal()
    except ValueError as exc:
        if "exactly one validation route" not in str(exc):
            raise AssertionError(f"wrong validation-route rejection: {exc}") from exc
    else:
        raise AssertionError("closed-loop autotune builder accepted missing validation route")


def test_closed_loop_auto_tune_builder_ignores_offline_log_selection(tmp_path: Path) -> None:
    selector = _closed_loop_autotune_selector(tmp_path, include_mixed_noise=True)
    selector._closed_loop_auto_tune_selected_log_indices = {0, 1}
    request = selector._build_closed_loop_auto_tune_request_from_modal()
    if request.offline_log_paths:
        raise AssertionError("direct closed-loop request included UI-selected offline logs")


def test_closed_loop_auto_tune_builder_includes_explicit_session_fields(tmp_path: Path) -> None:
    selector = _closed_loop_autotune_selector(tmp_path)
    request = selector._build_closed_loop_auto_tune_request_from_modal()
    if request.filter_id != "ca_kf" or request.tracking_mode != TRACKING_MODE_ACTIVE:
        raise AssertionError("closed-loop autotune request did not preserve selected filter/tracking mode")
    if request.offline_log_paths:
        raise AssertionError("direct closed-loop autotune request should not require offline logs")
    if len(request.validation_routes) != 1:
        raise AssertionError("closed-loop autotune request did not preserve exactly one validation route")
    route = request.validation_routes[0]
    if not route.route_data or route.name != _short_saved_route().name:
        raise AssertionError("closed-loop autotune request did not serialize validation route data")
    if request.sensor_noise_profile != "Medium Noise":
        raise AssertionError("closed-loop autotune request did not preserve sensor preset")
    if request.vehicle_behavior_config.get("preset_name") != "Balanced":
        raise AssertionError("closed-loop autotune request did not preserve behavior config")
    if not request.actuator_realism_config.get("enabled"):
        raise AssertionError("closed-loop autotune request did not preserve actuator realism config")
    if request.trial_count != 12 or request.finalist_count != 1:
        raise AssertionError("closed-loop autotune request did not preserve direct trial count/finalist compatibility value")
    pending = request.metadata.get("pending_session")
    if not isinstance(pending, dict) or pending.get("validation_route_data") != route.route_data:
        raise AssertionError("closed-loop autotune request did not include explicit pending-session handoff metadata")
    if pending.get("offline_log_paths") != ():
        raise AssertionError("direct pending-session metadata did not preserve an empty optional log list")
    if request.strategy != "random_plus_coordinate_refinement" or pending.get("strategy") != request.strategy:
        raise AssertionError("closed-loop modal did not preserve the default adaptive/random algorithm")


def test_closed_loop_auto_tune_builder_propagates_optuna_strategy(tmp_path: Path) -> None:
    selector = _closed_loop_autotune_selector(tmp_path)
    selector._closed_loop_auto_tune_strategy = "optuna_tpe"
    request = selector._build_closed_loop_auto_tune_request_from_modal()
    pending = request.metadata.get("pending_session")
    if request.strategy != "optuna_tpe":
        raise AssertionError("closed-loop modal did not propagate the selected Optuna strategy")
    if not isinstance(pending, dict) or pending.get("strategy") != "optuna_tpe":
        raise AssertionError("pending closed-loop session did not preserve the selected Optuna strategy")


def test_closed_loop_auto_tune_builder_uses_current_gui_sensor_noise(tmp_path: Path) -> None:
    selector = _closed_loop_autotune_selector(tmp_path)
    custom_values = dict(SENSOR_NOISE_PRESETS["High Noise"])
    first_key = next(iter(custom_values))
    custom_values[first_key] = float(custom_values[first_key]) * 1.1
    selector._sensor_editor = SimpleNamespace(values=lambda: dict(custom_values), active_preset="Custom")
    selector._sensor_preset = "Custom"

    request = selector._build_closed_loop_auto_tune_request_from_modal()
    expected = sensor_noise_config_from_values(custom_values, preset_name="Custom").to_dict()
    if request.sensor_noise_config != expected:
        raise AssertionError("direct request did not use the current Sensor Noise tab values")
    if request.metadata.get("selected_sensor_noise_signature") != noise_signature(expected):
        raise AssertionError("direct request did not derive its noise signature from current GUI values")


def test_closed_loop_auto_tune_modal_not_available_for_raw_gnss() -> None:
    selector = StartupMapSelector.__new__(StartupMapSelector)
    selector._setup_filter_records = [
        SimpleNamespace(filter_id="raw_gnss", auto_tune_enabled=False, auto_tune_profile=None)
    ]
    selector._closed_loop_saved_tune_status = ""
    selector._closed_loop_auto_tune_modal_open = False
    selector._open_closed_loop_auto_tune_modal("raw_gnss")
    if selector._closed_loop_auto_tune_modal_open:
        raise AssertionError("closed-loop autotune modal should not open for raw_gnss")
    if "unavailable" not in selector._closed_loop_saved_tune_status:
        raise AssertionError("raw_gnss closed-loop autotune guard did not show a clear status")


def test_direct_closed_loop_trial_reset_starts_two_trials_cleanly() -> None:
    class FakeRoutePlanner:
        planner_error = ""

        def __init__(self) -> None:
            self.route = ["stale"]
            self.generated_count = 0

        def clear_route(self) -> None:
            self.route = []

        def generate_route(self, start: object, goal: object) -> list[object]:
            self.generated_count += 1
            self.route = [start, goal]
            return list(self.route)

        def get_route(self) -> list[object]:
            return list(self.route)

    class FakeTracker:
        def __init__(self) -> None:
            self.completed = True
            self.routes: list[list[object]] = []

        def clear_route(self) -> None:
            self.completed = False

        def set_route(self, route: list[object]) -> None:
            self.completed = False
            self.routes.append(list(route))

    class FakeVehicle:
        def __init__(self) -> None:
            self.controls: list[object] = []
            self.autopilot_values: list[bool] = []

        def set_autopilot(self, enabled: bool) -> None:
            self.autopilot_values.append(bool(enabled))

        def apply_control(self, control: object) -> None:
            self.controls.append(control)

    class FakeVehicleManager:
        def __init__(self) -> None:
            self.teleport_count = 0

        def teleport_to_waypoint(self, waypoint: object) -> None:
            self.teleport_count += 1

    app = SimulationApp.__new__(SimulationApp)
    app._test_runner = None
    app._offline_recorder = None
    app._test_route_authoring_active = True
    app._map_selection_active = True
    app._route_activation_state = RouteActivationState.ROUTE_ACTIVE
    app._pending_start_waypoint = None
    app._pending_goal_waypoint = None
    app._pending_start_autonomous = False
    app._stabilization_started_monotonic = None
    app._stabilization_elapsed_seconds = 0.0
    app._stabilization_stable_ticks = 0
    app._stabilization_error_m = None
    app._stabilization_timed_out = False
    app._route_generation_blocked = False
    app._planner_status = ""
    app._lightweight_closed_loop_auto_tune = True
    app._drive_mode = DriveMode.AUTONOMOUS
    app._latest_state = None
    app._latest_ground_truth_state = SimpleNamespace(speed=7.0)
    app._latest_estimated_state = None
    app._latest_localization_status = None
    app._latest_gnss_diagnostics = None
    app._latest_gnss_frame = None
    app._gnss_trail_xy = []
    app._failure_monitor_last_progress_time = 1.0
    app._failure_monitor_last_position = (1.0, 2.0)
    app._failure_monitor_last_distance_to_goal = 3.0
    app._failure_monitor_last_closest_index = 4
    app._failure_monitor_deviation_started = 5.0
    app._last_benchmark_failure_reason = "Vehicle stuck"
    app.route_planner = FakeRoutePlanner()
    app.waypoint_tracker = FakeTracker()
    app.driving_behavior_config = DrivingBehaviorConfig()
    app.autonomous_controller = VehicleController(behavior_config=app.driving_behavior_config)
    app.speed_planner = CurvatureSpeedPlanner(app.driving_behavior_config)
    app.actuator_realism = ActuatorRealism(app.driving_behavior_config)
    stale_brake = SimulationApp._neutral_vehicle_control()
    stale_brake.brake = 1.0
    app._latest_requested_control = stale_brake
    app._latest_applied_control = stale_brake
    app._vehicle = FakeVehicle()
    app._vehicle_manager = FakeVehicleManager()
    app._filter_manager = None
    app._ground_truth_provider = None

    start = SimpleNamespace()
    goal = SimpleNamespace()
    for _trial in range(2):
        app._latest_tracking = app._empty_tracking_status()
        app._latest_tracking = app._latest_tracking.__class__(
            target_waypoint=None,
            closest_index=99,
            target_index=99,
            cross_track_error_m=0.0,
            distance_to_goal_m=0.0,
            heading_error_deg=None,
            completed=True,
        )
        app._latest_applied_control = stale_brake
        app._latest_requested_control = stale_brake
        app._last_benchmark_failure_reason = "Vehicle stuck"

        app._reset_direct_closed_loop_trial_lifecycle()
        app._begin_route_initialization(start, goal, start_autonomous=True)
        app._activate_pending_route_after_stabilization()

        if app._route_activation_state != RouteActivationState.ROUTE_ACTIVE:
            raise AssertionError("direct trial did not activate a fresh route")
        if not app.route_planner.get_route():
            raise AssertionError("direct trial did not generate a fresh route")
        if app._latest_tracking.completed:
            raise AssertionError("direct trial inherited completed tracking state")
        if app._latest_applied_control.brake != 0.0 or app._latest_requested_control.brake != 0.0:
            raise AssertionError("direct trial inherited a stale brake command")
        if app.actuator_realism.latest_applied_control.brake != 0.0:
            raise AssertionError("direct trial actuator model started from stale braking")
        if app._last_benchmark_failure_reason:
            raise AssertionError("direct trial inherited previous failure reason")
        if app._failure_monitor_last_progress_time is not None:
            raise AssertionError("direct trial inherited stuck detector progress state")

    if app._vehicle_manager.teleport_count != 2:
        raise AssertionError("two direct trial starts did not teleport to the route start twice")
    if len(app.waypoint_tracker.routes) != 2:
        raise AssertionError("two direct trial starts did not install two fresh tracker routes")


def test_app_closed_loop_validation_runner_delegates_to_app() -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.requests = []

        def _run_closed_loop_auto_tune_validation(self, request: object) -> dict[str, object]:
            self.requests.append(request)
            return {"route_completion_success": True, "eval_filtered_rmse_m": 1.0}

    app = FakeApp()
    finalist = ClosedLoopFinalist(
        rank=1,
        candidate_tune={"process_jerk_stddev_mps3": 1.2},
        offline_score=0.5,
        offline_metrics={},
        trial_index=7,
        source_output_folder=None,
    )
    request = ClosedLoopValidationRequest(
        filter_id="ca_kf",
        tracking_mode=TRACKING_MODE_PASSIVE,
        finalist=finalist,
        validation_route=ClosedLoopValidationRoute.from_object(_short_saved_route()),
        sensor_noise_config={},
        vehicle_behavior_config={},
        actuator_realism_config={},
        output_folder=Path("validation_output"),
    )
    result = AppClosedLoopValidationRunner(app).run(request)
    if result.get("route_completion_success") is not True:
        raise AssertionError("app closed-loop validation runner did not return app metrics")
    if app.requests != [request]:
        raise AssertionError("app closed-loop validation runner did not pass through the validation request")


def test_auto_tune_modal_requires_primary_profile() -> None:
    selector = StartupMapSelector.__new__(StartupMapSelector)
    raw = SimpleNamespace(
        filter_id="raw_gnss",
        auto_tune_enabled=False,
        auto_tune_profile=None,
    )
    empty = SimpleNamespace(
        filter_id="empty_profile_filter",
        auto_tune_enabled=True,
        auto_tune_profile={"enabled": True, "primary": []},
    )
    selector._setup_filter_records = [raw, empty]
    selector._offline_saved_tune_status = ""
    selector._auto_tune_modal_open = False
    selector._open_auto_tune_modal("raw_gnss")
    if selector._auto_tune_modal_open:
        raise AssertionError("raw_gnss should not open auto tune modal")
    if selector._offline_saved_tune_status != "No auto-tune profile with primary parameters for this filter.":
        raise AssertionError("raw_gnss modal guard did not show clear status")
    selector._open_auto_tune_modal("empty_profile_filter")
    if selector._auto_tune_modal_open:
        raise AssertionError("empty primary profile should not open auto tune modal")


def test_offline_recording_discovery_defaults_to_benchmark_results() -> None:
    expected = PROJECT_ROOT / DEFAULT_OFFLINE_OUTPUT_ROOT / "offline_localization" / "recordings"
    if recordings_root() != expected:
        raise AssertionError(f"offline recordings default path changed: {recordings_root()} != {expected}")


def _closed_loop_autotune_selector(tmp_path: Path, include_mixed_noise: bool = False) -> StartupMapSelector:
    selector = StartupMapSelector.__new__(StartupMapSelector)
    selector._setup_filter_records = [record for record in discover_filters() if record.valid]
    selector._selected_filter_id = "ca_kf"
    selector._closed_loop_auto_tune_filter_id = "ca_kf"
    selector._tracking_mode = TRACKING_MODE_ACTIVE
    selector._selected_filter_tunes = {
        "ca_kf": {"gnss_position_stddev_m": 1.0, "process_jerk_stddev_mps3": 1.2}
    }
    selector._filter_tune_editor = None
    selector._filter_tune_editor_filter_id = ""
    selector._sensor_editor = None
    selector._sensor_preset = "Medium Noise"
    selector._behavior_editor = None
    selector._behavior_preset = "Balanced"
    selector._closed_loop_auto_tune_trials = 12
    selector._closed_loop_auto_tune_finalists = 1
    selector._closed_loop_auto_tune_strategy = "random_plus_coordinate_refinement"
    selector._closed_loop_auto_tune_status_lines = []
    selector._recommendation_applied_by_filter = {}
    selector._selected_route_indices = set()

    route = _short_saved_route()
    selector._route_items = [
        SimpleNamespace(index=0, route=route, straight_line_length_m=100.0, compatible_with_available_maps=True)
    ]
    selector._closed_loop_auto_tune_validation_route_index = 0

    sensor = sensor_noise_config_from_values(SENSOR_NOISE_PRESETS["Medium Noise"], preset_name="Medium Noise").to_dict()
    log_a = _recorded_log_info(tmp_path, "run_001", "route_001", sensor)
    logs = [log_a]
    if include_mixed_noise:
        other_sensor = dict(sensor)
        other_sensor["gnss_noise_lat_stddev_deg"] = float(other_sensor.get("gnss_noise_lat_stddev_deg", 0.0)) + 0.00001
        other_sensor["preset_name"] = "Different Noise"
        logs.append(_recorded_log_info(tmp_path, "run_001", "route_002", other_sensor))
    selector._recorded_logs = logs
    selector._closed_loop_auto_tune_selected_log_indices = {0}
    return selector


def _recorded_log_info(
    tmp_path: Path,
    run_id: str,
    route_folder_name: str,
    sensor_config: dict[str, object],
) -> object:
    route = _short_saved_route()
    run_folder = tmp_path / "benchmark_results" / "offline_localization" / "recordings" / run_id
    route_folder = run_folder / route_folder_name
    route_folder.mkdir(parents=True, exist_ok=True)
    log_path = route_folder / "sensor_log.csv"
    log_path.write_text("timestamp\n0.0\n", encoding="utf-8")
    write_json(
        route_folder / "route_metadata.json",
        {
            "route": route.to_dict(),
            "map_name": route.map_name,
            "sensor_noise_config": dict(sensor_config),
            "vehicle_behavior_config": {"preset_name": "Balanced"},
            "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
        },
    )
    write_json(
        route_folder / "recording_summary.json",
        {
            "route_name": route.name,
            "map_name": route.map_name,
            "sample_count": 1,
            "duration_s": 0.1,
            "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
        },
    )
    return SimpleNamespace(
        route_folder=route_folder,
        sensor_log_path=log_path,
        run_folder=run_folder,
        recording_id=run_id,
        route_name=route.name,
        map_name=route.map_name or "",
        sample_count=1,
        duration_s=0.1,
        recording_driver=RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
        sensor_noise_preset=str(sensor_config.get("preset_name") or ""),
        vehicle_behavior_preset="Balanced",
        created_at="2026-06-07T00:00:00",
        failure_reason="",
    )


def _short_saved_route():
    store = TestRouteStore()
    for route in store.all_routes:
        if route.name == "sadece_viraj":
            return route
    if not store.all_routes:
        raise AssertionError("config/test_routes.json has no saved routes")
    return store.all_routes[0]


def _write_synthetic_log(path: Path, route_name: str, map_name: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SENSOR_LOG_FIELDNAMES)
        writer.writeheader()
        for frame in range(24):
            timestamp = frame * 0.1
            gt_x = frame * 0.7
            gt_y = 0.15 * math.sin(frame * 0.2)
            gnss_x = gt_x + 0.35
            gnss_y = gt_y - 0.18
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "frame": frame,
                    "dt": 0.1 if frame else 0.0,
                    "map_name": map_name,
                    "route_name": route_name,
                    "route_index": 1,
                    "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
                    "phase": "EVALUATION_ACTIVE" if frame >= 12 else "FILTER_WARMUP",
                    "valid_for_metrics": frame >= 12,
                    "seconds_since_teleport": timestamp,
                    "seconds_since_recording_start": timestamp,
                    "fresh_gnss_after_teleport_count": frame + 1,
                    "fresh_imu_after_teleport_count": frame + 1,
                    "teleport_frame": 0,
                    "warmup_excluded_reason": "" if frame >= 12 else "synthetic_warmup",
                    "ground_truth_x": gt_x,
                    "ground_truth_y": gt_y,
                    "ground_truth_z": 0.0,
                    "ground_truth_yaw": 0.0,
                    "ground_truth_speed": 7.0,
                    "ground_truth_vx_mps": 7.0,
                    "ground_truth_vy_mps": 0.0,
                    "ground_truth_ax_mps2": 0.0,
                    "ground_truth_ay_mps2": 0.0,
                    "ground_truth_yaw_rate_radps": 0.0,
                    "gnss_latitude": 0.0,
                    "gnss_longitude": 0.0,
                    "gnss_altitude": 0.0,
                    "gnss_local_x": gnss_x,
                    "gnss_local_y": gnss_y,
                    "gnss_local_z": 0.0,
                    "gnss_frame": frame,
                    "gnss_timestamp": timestamp,
                    "imu_accel_x": 0.0,
                    "imu_accel_y": 0.0,
                    "imu_accel_z": 0.0,
                    "imu_gyro_x": 0.0,
                    "imu_gyro_y": 0.0,
                    "imu_gyro_z": 0.0,
                    "imu_compass": math.pi / 2.0,
                    "imu_frame": frame,
                    "imu_timestamp": timestamp,
                    "control_throttle": 0.35,
                    "control_brake": 0.0,
                    "control_steer": 0.0,
                    "control_hand_brake": False,
                    "control_reverse": False,
                }
            )


def _write_old_synthetic_log(path: Path, route_name: str, map_name: str) -> None:
    old_fieldnames = [field for field in SENSOR_LOG_FIELDNAMES if field not in {
        "phase",
        "valid_for_metrics",
        "seconds_since_teleport",
        "seconds_since_recording_start",
        "fresh_gnss_after_teleport_count",
        "fresh_imu_after_teleport_count",
        "teleport_frame",
        "warmup_excluded_reason",
    }]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=old_fieldnames)
        writer.writeheader()
        for frame in range(140):
            timestamp = frame * 0.1
            gt_x = frame * 0.5
            row = {
                "timestamp": timestamp,
                "frame": frame,
                "dt": 0.1 if frame else 0.0,
                "map_name": map_name,
                "route_name": route_name,
                "route_index": 1,
                "recording_driver": RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER,
                "ground_truth_x": gt_x,
                "ground_truth_y": 0.0,
                "ground_truth_z": 0.0,
                "ground_truth_yaw": 0.0,
                "ground_truth_speed": 5.0,
                "ground_truth_vx_mps": 5.0,
                "ground_truth_vy_mps": 0.0,
                "ground_truth_ax_mps2": 0.0,
                "ground_truth_ay_mps2": 0.0,
                "ground_truth_yaw_rate_radps": 0.0,
                "gnss_latitude": 0.0,
                "gnss_longitude": 0.0,
                "gnss_altitude": 0.0,
                "gnss_local_x": gt_x + 0.2,
                "gnss_local_y": 0.0,
                "gnss_local_z": 0.0,
                "gnss_frame": frame,
                "gnss_timestamp": timestamp,
                "imu_accel_x": 0.0,
                "imu_accel_y": 0.0,
                "imu_accel_z": 0.0,
                "imu_gyro_x": 0.0,
                "imu_gyro_y": 0.0,
                "imu_gyro_z": 0.0,
                "imu_compass": math.pi / 2.0,
                "imu_frame": frame,
                "imu_timestamp": timestamp,
                "control_throttle": 0.3,
                "control_brake": 0.0,
                "control_steer": 0.0,
                "control_hand_brake": False,
                "control_reverse": False,
            }
            writer.writerow({key: row.get(key, "") for key in old_fieldnames})


def run_all() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_offline_replay_runs_on_short_saved_route_log(Path(directory))
    test_localization_metrics_split_full_and_eval_windows()
    test_offline_replay_errors_use_sensor_timestamp_ground_truth()
    with tempfile.TemporaryDirectory() as directory:
        test_old_logs_without_valid_for_metrics_use_timestamp_fallback(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_auto_tune_replay_context_uses_compact_route_artifacts(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_windows_path_length_guard_has_clear_error(Path(directory))
    test_offline_recording_does_not_feed_filter_control()
    test_offline_recording_warmup_does_not_trigger_route_failure()
    test_startup_gui_uses_mode_based_tabs()
    test_startup_tune_storage_feeds_closed_loop_and_offline_requests()
    test_registry_reads_optional_auto_tune_profile()
    test_parameter_editor_mousewheel_uses_mouse_position_without_event_pos()
    with tempfile.TemporaryDirectory() as directory:
        test_filter_auto_tuner_passes_candidate_tunes_and_saves_best_config(Path(directory))
    test_saved_tune_apply_updates_only_selected_filter()
    test_closed_loop_saved_tune_config_apply_updates_selected_filter()
    test_closed_loop_apply_recommended_still_updates_tune()
    test_auto_tune_modal_requires_primary_profile()
    test_offline_recording_discovery_defaults_to_benchmark_results()


if __name__ == "__main__":
    run_all()
    print("offline localization replay checks passed")
