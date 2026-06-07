"""Offline localization replay artifact and fairness checks."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

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
from src.evaluation.sensor_log_recorder import SENSOR_LOG_FIELDNAMES  # noqa: E402
from src.evaluation.test_route_store import TestRouteStore  # noqa: E402
from src.KalmanLab.registry import discover_filters  # noqa: E402
from src.visualization.startup_map_selector import (  # noqa: E402
    CLOSED_LOOP_SUBTABS,
    OFFLINE_SUBTABS,
    OFFLINE_TEST_SETUP_SUBTABS,
    TOP_LEVEL_TABS,
    StartupMapSelector,
)


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

    offline_tunes = selector._included_offline_filter_tunes(("ca_kf", "ctra_ekf", "raw_gnss"))
    if offline_tunes.get("ca_kf", {}).get("gnss_position_stddev_m") != 3.21:
        raise AssertionError("offline CA-KF tune was not preserved")
    if offline_tunes.get("ctra_ekf", {}).get("process_jerk_stddev_mps3") != 4.56:
        raise AssertionError("offline CTRA-EKF tune was not preserved")
    if "raw_gnss" in offline_tunes:
        raise AssertionError("raw_gnss should not require startup tune values")


def test_offline_recording_discovery_defaults_to_benchmark_results() -> None:
    expected = PROJECT_ROOT / DEFAULT_OFFLINE_OUTPUT_ROOT / "offline_localization" / "recordings"
    if recordings_root() != expected:
        raise AssertionError(f"offline recordings default path changed: {recordings_root()} != {expected}")


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
    test_offline_recording_does_not_feed_filter_control()
    test_offline_recording_warmup_does_not_trigger_route_failure()
    test_startup_gui_uses_mode_based_tabs()
    test_startup_tune_storage_feeds_closed_loop_and_offline_requests()
    test_offline_recording_discovery_defaults_to_benchmark_results()


if __name__ == "__main__":
    run_all()
    print("offline localization replay checks passed")
