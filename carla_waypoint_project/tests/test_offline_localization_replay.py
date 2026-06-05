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
from src.evaluation.offline_replay_runner import OfflineReplayRequest, OfflineReplayRunner  # noqa: E402
from src.evaluation.sensor_log_recorder import SENSOR_LOG_FIELDNAMES  # noqa: E402
from src.evaluation.test_route_store import TestRouteStore  # noqa: E402


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
    for relative in (
        "replay_results/raw_gnss_estimates.csv",
        "replay_results/ca_kf_estimates.csv",
        "plots/trajectory_comparison.png",
        "plots/position_error_over_time.png",
        "plots/rmse_comparison.png",
    ):
        if not (result.output_folder / "route_001_sadece_viraj" / relative).exists():
            raise AssertionError(f"missing offline artifact: {relative}")


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


def test_startup_test_setup_has_nested_offline_tabs() -> None:
    startup_source = (PROJECT_ROOT / "src" / "visualization" / "startup_map_selector.py").read_text(encoding="utf-8")
    for text in (
        "Evaluation Mode",
        "Sensor Noise",
        "Vehicle Behavior",
        "Offline Replay",
        "Record Sensor Logs from Selected Routes",
        "Run Offline Replay Evaluation",
    ):
        if text not in startup_source:
            raise AssertionError(f"startup setup missing {text!r}")


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


def run_all() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_offline_replay_runs_on_short_saved_route_log(Path(directory))
    test_offline_recording_does_not_feed_filter_control()
    test_startup_test_setup_has_nested_offline_tabs()
    test_offline_recording_discovery_defaults_to_benchmark_results()


if __name__ == "__main__":
    run_all()
    print("offline localization replay checks passed")
