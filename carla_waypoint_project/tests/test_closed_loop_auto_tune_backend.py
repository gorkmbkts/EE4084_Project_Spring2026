"""Closed-loop auto-tune backend tests with fake runners."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import filter_auto_tuner as filter_auto_tuner_module  # noqa: E402
from src.evaluation.closed_loop_auto_tune import (  # noqa: E402
    ClosedLoopAutoTuneRequest,
    ClosedLoopBenchmarkAutoTuner,
    ClosedLoopValidationRoute,
    closed_loop_objective_score,
)
from src.evaluation.evaluation_artifacts import write_json  # noqa: E402
from src.evaluation.filter_auto_tuner import AutoTuneResult, AutoTuneTrialResult, OfflineBenchmarkAutoTuner  # noqa: E402
import src.evaluation.offline_replay_runner as offline_replay_runner_module  # noqa: E402
from src.evaluation.sensor_noise_tune_mapper import noise_signature  # noqa: E402
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


def test_closed_loop_autotuner_rejects_zero_validation_routes(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    request = _request(tmp_path, (log_path,), sensor, validation_routes=())
    with pytest.raises(ValueError, match="exactly one validation route"):
        ClosedLoopBenchmarkAutoTuner(offline_tuner=_FakeOfflineTuner(), validation_runner=_FakeValidationRunner()).run(request)


def test_closed_loop_autotuner_rejects_multiple_validation_routes(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    routes = (
        ClosedLoopValidationRoute("route_1", "Town01", "route_1"),
        ClosedLoopValidationRoute("route_2", "Town01", "route_2"),
    )
    request = _request(tmp_path, (log_path,), sensor, validation_routes=routes)
    with pytest.raises(ValueError, match="exactly one validation route"):
        ClosedLoopBenchmarkAutoTuner(offline_tuner=_FakeOfflineTuner(), validation_runner=_FakeValidationRunner()).run(request)


def test_closed_loop_autotuner_rejects_mixed_noise_logs_by_default(tmp_path: Path) -> None:
    log_a, sensor = _recorded_log(tmp_path, "route_001", gnss_stddev=1.25)
    log_b, _other_sensor = _recorded_log(tmp_path, "route_002", gnss_stddev=2.5)
    offline = _FakeOfflineTuner()
    request = _request(tmp_path, (log_a, log_b), sensor)
    with pytest.raises(ValueError, match="mixed sensor noise"):
        ClosedLoopBenchmarkAutoTuner(offline_tuner=offline, validation_runner=_FakeValidationRunner()).run(request)
    if offline.calls:
        raise AssertionError("mixed-noise rejection should happen before offline candidate generation")


def test_closed_loop_autotuner_selects_top_finalists_and_validates_only_them(tmp_path: Path) -> None:
    log_a, sensor = _recorded_log(tmp_path, "route_001")
    log_b, _sensor_b = _recorded_log(tmp_path, "route_002")
    offline = _FakeOfflineTuner(
        scores=(3.0, 1.0, 2.0, 4.0),
        output_root=tmp_path / "offline_trials",
    )
    runner = _FakeValidationRunner(
        {
            1: {"route_completion_success": True, "eval_filtered_rmse_m": 5.0, "mean_cross_track_error_m": 1.0, "max_cross_track_error_m": 2.0},
            2: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0, "mean_cross_track_error_m": 0.2, "max_cross_track_error_m": 0.5},
        }
    )
    request = _request(
        tmp_path,
        (log_a, log_b),
        sensor,
        finalist_count=2,
        trial_count=4,
        strategy="random_plus_coordinate_refinement",
    )
    result = ClosedLoopBenchmarkAutoTuner(offline_tuner=offline, validation_runner=runner).run(request)

    if len(offline.calls) != 1:
        raise AssertionError("offline candidate generation was not delegated to the offline tuner")
    offline_request = offline.calls[0]
    if offline_request.sensor_log_paths != (log_a, log_b) or offline_request.max_trials != 4:
        raise AssertionError("offline tuner did not receive selected logs and trial count")
    if len(runner.requests) != 2:
        raise AssertionError("closed-loop validation should run only finalist_count finalists")
    if [call.finalist.trial_index for call in runner.requests] != [2, 3]:
        raise AssertionError("closed-loop validation did not use the top offline-score finalists")
    if result.best_tune.get("process_jerk_stddev_mps3") != 3.0:
        raise AssertionError("final best tune should be selected by closed-loop score, not offline rank only")

    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError("closed-loop best tune did not save schema v2")
    if config.get("benchmark_mode") != BENCHMARK_MODE_CLOSED_LOOP or config.get("tracking_mode") != TRACKING_PASSIVE:
        raise AssertionError("passive closed-loop backend did not save passive closed-loop metadata")
    if config.get("tune_scope") != TUNE_SCOPE_CLOSED_LOOP_VALIDATED or not config.get("validated_in_closed_loop"):
        raise AssertionError("closed-loop backend did not mark validated tune scope")
    path_text = str(result.saved_config_path).replace("\\", "/")
    if "/_at/cl/p/ca_kf/" not in path_text:
        raise AssertionError(f"passive closed-loop tune was saved to the wrong folder: {path_text}")
    if "closed_loop/auto_tune/passive/ca_kf/" not in str(config.get("logical_output_group") or ""):
        raise AssertionError("passive closed-loop tune did not preserve logical output group")
    listed = filter_auto_tuner_module.list_saved_tune_configs(
        "ca_kf",
        output_root=request.output_root,
        context=closed_loop_tune_context(
            "ca_kf",
            TRACKING_PASSIVE,
            sensor_noise_config=request.sensor_noise_config,
            vehicle_behavior_config=request.vehicle_behavior_config,
            actuator_realism_config=request.actuator_realism_config,
        ),
    )
    if not any(str(item.get("path")) == str(result.saved_config_path) for item in listed):
        raise AssertionError(f"saved tune browser did not find compact closed-loop config: {listed}")
    if len(config.get("closed_loop_validation_results") or []) != 2:
        raise AssertionError("saved config did not preserve finalist validation results")


def test_closed_loop_autotuner_candidate_generation_uses_compact_replay_staging(tmp_path: Path) -> None:
    long_root = tmp_path / "long_project_segment" / "benchmark_results"
    log_path, sensor = _recorded_log(long_root.parent, "route_001")
    replay_calls = []

    class FakeReplayRunner:
        def run(self, request: object) -> object:
            replay_calls.append(request)
            folder = Path(request.run_folder_override)
            folder.mkdir(parents=True, exist_ok=True)
            write_json(
                folder / "aggregate_summary.json",
                {
                    "route_count": len(request.sensor_log_paths),
                    "best_filter_id": "ca_kf",
                    "aggregate_rows": [
                        {
                            "filter_id": "ca_kf",
                            "eval_position_rmse_m": 1.0,
                            "yaw_rmse_deg": 0.0,
                            "divergence_event_count": 0,
                        }
                    ],
                },
            )
            return SimpleNamespace(output_folder=folder, failures=())

    runner = _FakeValidationRunner({1: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0}})
    request = _request(
        long_root.parent,
        (log_path,),
        sensor,
        finalist_count=1,
        trial_count=1,
        strategy="random_plus_coordinate_refinement",
    )
    request = ClosedLoopAutoTuneRequest(
        **{**request.to_dict(), "offline_log_paths": request.offline_log_paths, "validation_routes": request.validation_routes, "output_root": str(long_root)}
    )
    result = ClosedLoopBenchmarkAutoTuner(
        offline_tuner=OfflineBenchmarkAutoTuner(runner_factory=FakeReplayRunner),
        validation_runner=runner,
    ).run(request)

    if len(replay_calls) != 1:
        raise AssertionError("closed-loop candidate generation did not run one offline replay trial")
    replay_path = Path(replay_calls[0].run_folder_override)
    path_text = str(replay_path).replace("\\", "/")
    if "/_tmp/at/" not in path_text:
        raise AssertionError(f"closed-loop candidate replay did not use compact staging: {replay_path}")
    metrics_path = replay_path / "r001" / "met" / "summary_metrics.json"
    if len(str(metrics_path.resolve())) >= offline_replay_runner_module.WINDOWS_PATH_LENGTH_GUARD:
        raise AssertionError(f"closed-loop candidate replay metrics path is too long: {metrics_path}")

    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    final_summary_path = result.output_folder / "closed_loop_auto_tune_summary.json"
    if "/_at/cl/p/ca_kf/" not in str(final_summary_path).replace("\\", "/"):
        raise AssertionError(f"closed-loop final output did not use compact physical folder: {result.saved_config_path}")
    if len(str(final_summary_path.resolve())) >= offline_replay_runner_module.WINDOWS_PATH_LENGTH_GUARD:
        raise AssertionError(f"closed-loop final summary path is too long: {final_summary_path}")
    if "closed_loop/auto_tune/passive/ca_kf/" not in str(config.get("logical_output_group") or ""):
        raise AssertionError("closed-loop saved config did not preserve logical output group")
    if config.get("noise_signature") != noise_signature(sensor):
        raise AssertionError("closed-loop saved config did not preserve full noise_signature")
    if "/_tmp/at/" not in str(config.get("offline_candidate_staging_folder") or "").replace("\\", "/"):
        raise AssertionError("closed-loop saved config did not link compact offline candidate staging folder")


def test_closed_loop_autotuner_saves_active_config_separately(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    offline = _FakeOfflineTuner(scores=(1.0, 2.0), output_root=tmp_path / "offline_trials")
    runner = _FakeValidationRunner({1: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0}})
    request = _request(tmp_path, (log_path,), sensor, tracking_mode=TRACKING_ACTIVE, finalist_count=1)
    result = ClosedLoopBenchmarkAutoTuner(offline_tuner=offline, validation_runner=runner).run(request)
    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    if config.get("tracking_mode") != TRACKING_ACTIVE or config.get("recommended_usage") != "closed_loop_active":
        raise AssertionError("active closed-loop backend did not save active usage metadata")
    if "passive sensor-log replay" not in str(config.get("active_control_parameter_policy") or ""):
        raise AssertionError("active config did not document passive offline candidate-generation limitation")
    path_text = str(result.saved_config_path).replace("\\", "/")
    if "/_at/cl/a/ca_kf/" not in path_text:
        raise AssertionError(f"active closed-loop tune was saved to the wrong folder: {path_text}")
    if "closed_loop/auto_tune/active/ca_kf/" not in str(config.get("logical_output_group") or ""):
        raise AssertionError("active config did not preserve active logical output group")


def test_backend_created_configs_cannot_mix_contexts(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    behavior = {"preset_name": "Balanced", "target_speed": 8.0}
    actuator = {"preset_name": "Realistic", "actuator_delay_s": 0.08}

    passive_result = ClosedLoopBenchmarkAutoTuner(
        offline_tuner=_FakeOfflineTuner(scores=(1.0,)),
        validation_runner=_FakeValidationRunner({1: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0}}),
    ).run(_request(tmp_path, (log_path,), sensor, tracking_mode=TRACKING_PASSIVE, behavior=behavior, actuator=actuator))
    active_result = ClosedLoopBenchmarkAutoTuner(
        offline_tuner=_FakeOfflineTuner(scores=(1.0,)),
        validation_runner=_FakeValidationRunner({1: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0}}),
    ).run(_request(tmp_path, (log_path,), sensor, tracking_mode=TRACKING_ACTIVE, behavior=behavior, actuator=actuator))

    passive_config = json.loads(passive_result.saved_config_path.read_text(encoding="utf-8"))
    active_config = json.loads(active_result.saved_config_path.read_text(encoding="utf-8"))
    passive_context = closed_loop_tune_context("ca_kf", TRACKING_PASSIVE, sensor, behavior, actuator)
    active_context = closed_loop_tune_context("ca_kf", TRACKING_ACTIVE, sensor, behavior, actuator)
    offline_context = offline_tune_context("ca_kf", sensor)

    if TuneCompatibility.check(passive_config, active_context).compatible:
        raise AssertionError("passive closed-loop config was compatible with active context")
    if TuneCompatibility.check(active_config, passive_context).compatible:
        raise AssertionError("active closed-loop config was compatible with passive context")
    if TuneCompatibility.check(passive_config, offline_context).compatible:
        raise AssertionError("closed-loop config was compatible with offline context")

    offline_config = {
        "schema_version": SCHEMA_VERSION,
        "filter_id": "ca_kf",
        "benchmark_mode": BENCHMARK_MODE_OFFLINE,
        "tracking_mode": TRACKING_PASSIVE,
        "tune_scope": TUNE_SCOPE_OFFLINE,
        "noise_signature": noise_signature(sensor),
    }
    if TuneCompatibility.check(offline_config, passive_context).compatible:
        raise AssertionError("offline config was compatible with closed-loop context")

    mismatched_behavior = {"preset_name": "Aggressive", "target_speed": 15.0}
    bad_context = closed_loop_tune_context("ca_kf", TRACKING_PASSIVE, sensor, mismatched_behavior, actuator)
    if TuneCompatibility.check(passive_config, bad_context).compatible:
        raise AssertionError("closed-loop config ignored incompatible behavior signature")


def test_closed_loop_autotuner_records_fallback_when_optuna_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    monkeypatch.setattr(filter_auto_tuner_module, "optuna", None)
    result = ClosedLoopBenchmarkAutoTuner(
        offline_tuner=_FakeOfflineTuner(scores=(1.0,)),
        validation_runner=_FakeValidationRunner({1: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0}}),
    ).run(_request(tmp_path, (log_path,), sensor, strategy="optuna_tpe"))
    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    if config.get("candidate_generation_strategy") != "random_plus_coordinate_refinement":
        raise AssertionError("closed-loop backend did not record fallback strategy when Optuna was unavailable")
    if config.get("optuna_available") is not False:
        raise AssertionError("closed-loop backend did not record Optuna availability")


def test_closed_loop_objective_penalizes_failure_and_timeout() -> None:
    success = closed_loop_objective_score(
        {
            "route_completion_success": True,
            "eval_filtered_rmse_m": 1.0,
            "mean_cross_track_error_m": 0.2,
            "max_cross_track_error_m": 0.5,
            "completion_time_s": 20.0,
            "driving_nis_by_type_summary": {"gnss_position": {"mean": 2.0}},
            "driving_mean_position_nees": 2.0,
        }
    )
    failure = closed_loop_objective_score(
        {
            "route_completion_success": False,
            "route_aborted": True,
            "timeout": True,
            "abort_reason": "timeout",
            "eval_filtered_rmse_m": 1.0,
            "mean_cross_track_error_m": 0.2,
            "max_cross_track_error_m": 0.5,
        }
    )
    if failure <= success + 100000.0:
        raise AssertionError("closed-loop objective did not heavily penalize failure/timeout")


class _FakeOfflineTuner:
    def __init__(self, scores: tuple[float, ...] = (1.0,), output_root: Path | None = None) -> None:
        self.scores = scores
        self.output_root = output_root or Path("fake_offline")
        self.calls: list[object] = []

    def run(self, request: object, progress_callback: object = None, stop_requested: object = None) -> AutoTuneResult:
        self.calls.append(request)
        trials = []
        for index, score in enumerate(self.scores, start=1):
            trials.append(
                AutoTuneTrialResult(
                    trial_index=index,
                    candidate_tune={
                        **dict(getattr(request, "base_tune", {})),
                        "process_jerk_stddev_mps3": float(index),
                    },
                    score=float(score),
                    metrics={"mean_eval_position_rmse_m": float(score), "candidate_index": index},
                    output_folder=self.output_root / f"t{index:03d}",
                    failed=False,
                )
            )
        return AutoTuneResult(
            filter_id=str(getattr(request, "filter_id", "ca_kf")),
            best_tune=dict(trials[0].candidate_tune),
            best_score=float(trials[0].score),
            best_metrics=dict(trials[0].metrics),
            selected_logs=tuple(getattr(request, "sensor_log_paths", ())),
            trial_results=tuple(trials),
            output_folder=self.output_root,
            saved_config_path=None,
        )


class _FakeValidationRunner:
    def __init__(self, metrics_by_rank: dict[int, dict[str, object]] | None = None) -> None:
        self.metrics_by_rank = metrics_by_rank or {1: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0}}
        self.requests: list[object] = []

    def run(self, request: object) -> dict[str, object]:
        self.requests.append(request)
        rank = int(request.finalist.rank)
        metrics = dict(self.metrics_by_rank.get(rank, {"route_completion_success": True, "eval_filtered_rmse_m": float(rank)}))
        metrics.setdefault("route_aborted", False)
        metrics.setdefault("timeout", False)
        metrics.setdefault("abort_reason", "")
        metrics.setdefault("completion_time_s", 20.0 + rank)
        metrics.setdefault("mean_cross_track_error_m", 0.5)
        metrics.setdefault("max_cross_track_error_m", 1.0)
        metrics.setdefault("driving_nis_by_type_summary", {"gnss_position": {"mean": 2.0, "sample_count": 10, "expected_dimension": 2}})
        metrics.setdefault("driving_mean_position_nees", 2.0)
        metrics.setdefault("driving_position_nees_source", "full_2x2")
        metrics["output_folder"] = str(request.output_folder)
        return metrics


def _request(
    tmp_path: Path,
    log_paths: tuple[Path, ...],
    sensor: dict[str, object],
    *,
    validation_routes: tuple[ClosedLoopValidationRoute, ...] | None = None,
    tracking_mode: str = TRACKING_PASSIVE,
    finalist_count: int = 1,
    trial_count: int = 3,
    strategy: str = "optuna_tpe",
    behavior: dict[str, object] | None = None,
    actuator: dict[str, object] | None = None,
) -> ClosedLoopAutoTuneRequest:
    return ClosedLoopAutoTuneRequest(
        filter_id="ca_kf",
        tracking_mode=tracking_mode,
        offline_log_paths=log_paths,
        validation_routes=validation_routes if validation_routes is not None else (ClosedLoopValidationRoute("route_1", "Town01", "route_1"),),
        sensor_noise_config=dict(sensor),
        vehicle_behavior_config=dict(behavior or {"preset_name": "Balanced"}),
        actuator_realism_config=dict(actuator or {"preset_name": "Realistic"}),
        base_tune={"gnss_position_stddev_m": 1.25, "imu_accel_stddev_mps2": 0.45, "process_jerk_stddev_mps3": 1.2},
        auto_tune_profile={
            "enabled": True,
            "primary": [
                {"key": "process_jerk_stddev_mps3", "scale": "log", "min": 0.5, "max": 3.0},
                {"key": "gnss_position_stddev_m", "scale": "log", "min": 0.5, "max": 3.0},
            ],
        },
        sensor_noise_profile=str(sensor.get("preset_name") or "Custom"),
        vehicle_behavior_profile=str((behavior or {}).get("preset_name") or "Balanced"),
        actuator_realism_profile=str((actuator or {}).get("preset_name") or "Realistic"),
        trial_count=trial_count,
        finalist_count=finalist_count,
        strategy=strategy,
        output_root=str(tmp_path / "benchmark_results"),
    )


def _recorded_log(tmp_path: Path, route_folder_name: str, gnss_stddev: float = 1.25) -> tuple[Path, dict[str, object]]:
    output_root = tmp_path / "benchmark_results"
    route_folder = output_root / "offline_localization" / "recordings" / "run_001" / route_folder_name
    route_folder.mkdir(parents=True)
    log_path = route_folder / "sensor_log.csv"
    log_path.write_text("timestamp\n0.0\n", encoding="utf-8")
    sensor = {"preset_name": "Synthetic", "gnss_position_stddev_m": gnss_stddev, "imu_accel_stddev_mps2": 0.45}
    write_json(
        route_folder / "route_metadata.json",
        {
            "route": {"name": route_folder_name, "map_name": "Town01"},
            "map_name": "Town01",
            "sensor_noise_config": sensor,
            "vehicle_behavior_config": {"preset_name": "Balanced"},
            "recording_driver": "ground_truth_controller",
        },
    )
    write_json(
        route_folder / "recording_summary.json",
        {
            "route_name": route_folder_name,
            "map_name": "Town01",
            "sample_count": 1,
            "recording_driver": "ground_truth_controller",
        },
    )
    return log_path, sensor
