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

from config.settings import BENCHMARK  # noqa: E402
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
from src.evaluation.route_test_runner import (  # noqa: E402
    MAX_ROUTE_ATTEMPTS,
    RouteTestRunner,
    _configured_route_attempt_limit,
)
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


def test_closed_loop_autotuner_runs_without_offline_logs_using_request_noise(tmp_path: Path) -> None:
    sensor = {"preset_name": "High Noise", "gnss_position_stddev_m": 2.75, "imu_accel_stddev_mps2": 0.8}
    runner = _FakeValidationRunner({1: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0}})
    request = _request(tmp_path, (), sensor, trial_count=1)
    result = ClosedLoopBenchmarkAutoTuner(validation_runner=runner).run(request)

    if len(runner.requests) != 1:
        raise AssertionError("zero-log direct request did not run its CARLA route trial")
    if runner.requests[0].sensor_noise_config != sensor:
        raise AssertionError("direct route trial did not receive request.sensor_noise_config")

    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    if config.get("selected_logs") != [] or config.get("selected_offline_logs") != []:
        raise AssertionError("zero-log direct config did not serialize empty selected log lists")
    if config.get("noise_signature") != noise_signature(sensor):
        raise AssertionError("direct config did not derive compatibility signature from request.sensor_noise_config")
    if config.get("sensor_noise_config") != sensor or config.get("sensor_noise_profile") != "High Noise":
        raise AssertionError("direct config did not preserve the selected sensor noise profile/config")
    if config.get("route_attempt_policy") != "one_attempt_per_candidate_trial":
        raise AssertionError("direct config did not record the one-attempt trial policy")


def test_closed_loop_autotuner_runs_direct_closed_loop_trials_and_selects_best(tmp_path: Path) -> None:
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

    if offline.calls:
        raise AssertionError("direct closed-loop auto tune must not delegate candidate scoring to the offline tuner")
    if len(runner.requests) != 4:
        raise AssertionError("closed-loop auto tune should run one real route trial per search trial")
    if [call.finalist.trial_index for call in runner.requests] != [1, 2, 3, 4]:
        raise AssertionError("closed-loop validation did not run direct sequential trial candidates")
    validation_output_folders = [str(call.output_folder).replace("\\", "/") for call in runner.requests]
    if not all("/trials/t" in folder for folder in validation_output_folders):
        raise AssertionError(f"closed-loop trial output did not use compact direct trial folders: {validation_output_folders}")
    best_validation = min(result.validation_results, key=lambda item: item.closed_loop_score)
    if best_validation.finalist_rank != 2:
        raise AssertionError("final best tune should be selected by direct closed-loop score")

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
    if config.get("candidate_generation_stage") != "direct_closed_loop_route_trials":
        raise AssertionError("saved config did not record direct closed-loop candidate generation")
    if len(config.get("closed_loop_validation_results") or []) != 4:
        raise AssertionError("saved config did not preserve all direct closed-loop trial results")
    if config.get("direct_closed_loop_trial_count") != 4:
        raise AssertionError("saved config did not record direct trial count")


def test_closed_loop_autotuner_direct_trials_use_compact_output_without_replay_staging(tmp_path: Path) -> None:
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

    if replay_calls:
        raise AssertionError("direct closed-loop auto tune should not run offline replay staging")

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
    if config.get("offline_candidate_staging_folder"):
        raise AssertionError("direct closed-loop saved config should not link offline candidate staging")
    direct_results = config.get("direct_closed_loop_trial_results") or []
    if len(direct_results) != 1:
        raise AssertionError("direct closed-loop saved config did not preserve trial results")


def test_closed_loop_autotuner_saves_active_config_separately(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    offline = _FakeOfflineTuner(scores=(1.0, 2.0), output_root=tmp_path / "offline_trials")
    runner = _FakeValidationRunner({1: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0}})
    request = _request(tmp_path, (log_path,), sensor, tracking_mode=TRACKING_ACTIVE, finalist_count=1)
    result = ClosedLoopBenchmarkAutoTuner(offline_tuner=offline, validation_runner=runner).run(request)
    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    if config.get("tracking_mode") != TRACKING_ACTIVE or config.get("recommended_usage") != "closed_loop_active":
        raise AssertionError("active closed-loop backend did not save active usage metadata")
    if "Active tracking tune search includes" not in str(config.get("active_control_parameter_policy") or ""):
        raise AssertionError("active config did not document active-control direct search policy")
    path_text = str(result.saved_config_path).replace("\\", "/")
    if "/_at/cl/a/ca_kf/" not in path_text:
        raise AssertionError(f"active closed-loop tune was saved to the wrong folder: {path_text}")
    if "closed_loop/auto_tune/active/ca_kf/" not in str(config.get("logical_output_group") or ""):
        raise AssertionError("active config did not preserve active logical output group")


def test_closed_loop_autotuner_respects_staged_trial_budgets(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    runner = _FakeValidationRunner(
        {
            1: {"route_completion_success": True, "eval_filtered_rmse_m": 3.0},
            2: {"route_completion_success": True, "eval_filtered_rmse_m": 2.0},
            3: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0},
        }
    )
    request = _with_stage_budgets(
        _request(tmp_path, (log_path,), sensor, tracking_mode=TRACKING_PASSIVE),
        passive=2,
        active=4,
        joint=1,
    )

    result = ClosedLoopBenchmarkAutoTuner(validation_runner=runner).run(request)
    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))

    if len(runner.requests) != 3:
        raise AssertionError("staged closed-loop tuner did not honor passive+joint trial budgets")
    if [call.stage for call in runner.requests] != [
        "stage0_context_baseline",
        "stage1_passive_q_model",
        "stage3_joint_local",
    ]:
        raise AssertionError("staged closed-loop tuner did not run expected passive stages")
    budgets = config.get("stage_budgets") if isinstance(config.get("stage_budgets"), dict) else {}
    if budgets.get("total_planned_trials") != 3 or config.get("direct_closed_loop_trial_count") != 3:
        raise AssertionError("saved config did not preserve staged trial budget metadata")


def test_closed_loop_autotuner_passive_mode_skips_active_control_stage(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    runner = _FakeValidationRunner({index: {"route_completion_success": True, "eval_filtered_rmse_m": float(index)} for index in range(1, 6)})
    request = _with_stage_budgets(
        _request(tmp_path, (log_path,), sensor, tracking_mode=TRACKING_PASSIVE),
        passive=2,
        active=3,
        joint=1,
    )

    result = ClosedLoopBenchmarkAutoTuner(validation_runner=runner).run(request)
    stages = [call.stage for call in runner.requests]
    if "stage2_active_control" in stages:
        raise AssertionError("passive closed-loop auto tune ran active-control tuning")
    if any(call.tracking_mode != TRACKING_PASSIVE for call in runner.requests):
        raise AssertionError("passive closed-loop auto tune used active tracking during staged trials")
    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    budgets = config.get("stage_budgets") if isinstance(config.get("stage_budgets"), dict) else {}
    if budgets.get("active_control_trials") != 0:
        raise AssertionError("passive saved config did not mark active-control trials as skipped")


def test_closed_loop_autotuner_active_mode_includes_active_control_stage(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    runner = _FakeValidationRunner({index: {"route_completion_success": True, "eval_filtered_rmse_m": float(index)} for index in range(1, 5)})
    request = _with_stage_budgets(
        _request(tmp_path, (log_path,), sensor, tracking_mode=TRACKING_ACTIVE),
        passive=1,
        active=2,
        joint=0,
    )

    result = ClosedLoopBenchmarkAutoTuner(validation_runner=runner).run(request)
    if [call.stage for call in runner.requests] != [
        "stage0_context_baseline",
        "stage2_active_control",
        "stage2_active_control",
    ]:
        raise AssertionError("active closed-loop auto tune did not run active-control stage")
    if [call.tracking_mode for call in runner.requests] != [TRACKING_PASSIVE, TRACKING_ACTIVE, TRACKING_ACTIVE]:
        raise AssertionError("active staged auto tune did not use passive baseline before active tuning")
    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    if not config.get("best_active_control_tune"):
        raise AssertionError("active saved config did not store best active-control tune metadata")


def test_active_ctra_search_runs_explicit_control_ablations(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    runner = _FakeValidationRunner(
        {
            1: {"route_completion_success": True, "eval_filtered_rmse_m": 2.0},
            2: {"route_completion_success": True, "eval_filtered_rmse_m": 0.5},
            3: {"route_completion_success": True, "eval_filtered_rmse_m": 1.5},
            4: {"route_completion_success": True, "eval_filtered_rmse_m": 1.6},
            5: {"route_completion_success": True, "eval_filtered_rmse_m": 1.7},
        }
    )
    base_request = _request(
        tmp_path,
        (log_path,),
        sensor,
        tracking_mode=TRACKING_ACTIVE,
        filter_id="ctra_ukf",
        strategy="random_plus_coordinate_refinement",
    )
    request = _with_stage_budgets(base_request, passive=1, active=4, joint=0)

    result = ClosedLoopBenchmarkAutoTuner(validation_runner=runner).run(request)
    active_results = [
        item for item in result.validation_results if item.stage == "stage2_active_control"
    ]
    if [item.raw_metrics.get("candidate_type") for item in active_results] != [
        "active_ablation_disabled",
        "active_ablation_accel_only",
        "active_ablation_yaw_only",
        "active_ablation_default_combined",
    ]:
        raise AssertionError("CTRA active search did not run the four explicit control ablations")
    disabled, accel_only, yaw_only, combined = [item.candidate_tune for item in active_results]
    if float(disabled.get("enable_control_input_prediction", 1.0)) != 0.0:
        raise AssertionError("disabled ablation left active prediction enabled")
    if float(accel_only.get("control_steer_to_yaw_rate_gain", 1.0)) != 0.0:
        raise AssertionError("acceleration-only ablation left yaw control enabled")
    if float(yaw_only.get("control_accel_gain_mps2", 1.0)) != 0.0:
        raise AssertionError("yaw-only ablation left acceleration control enabled")
    if float(combined.get("enable_control_input_prediction", 0.0)) != 1.0:
        raise AssertionError("combined ablation did not enable active prediction")
    if float(result.best_tune.get("enable_control_input_prediction", 1.0)) != 0.0:
        raise AssertionError("disabled ablation was best but was not selected")

    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    if config.get("best_active_candidate_uses_control") is not False:
        raise AssertionError("saved metadata did not identify the disabled active winner")


def test_joint_phase_selects_only_final_phase_validations(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    runner = _FakeValidationRunner(
        {
            1: {"route_completion_success": True, "eval_filtered_rmse_m": 5.0},
            2: {"route_completion_success": True, "eval_filtered_rmse_m": 0.1},
            3: {"route_completion_success": True, "eval_filtered_rmse_m": 3.0},
            4: {"route_completion_success": True, "eval_filtered_rmse_m": 2.0},
            5: {"route_completion_success": True, "eval_filtered_rmse_m": 2.5},
        }
    )
    request = _with_stage_budgets(
        _request(
            tmp_path,
            (log_path,),
            sensor,
            tracking_mode=TRACKING_PASSIVE,
            strategy="random_plus_coordinate_refinement",
        ),
        passive=2,
        active=0,
        joint=3,
    )

    result = ClosedLoopBenchmarkAutoTuner(validation_runner=runner).run(request)
    if result.validation_results[1].closed_loop_score >= result.best_score:
        raise AssertionError("test setup did not make an earlier-stage candidate look better")
    selected = next(
        item
        for item in result.validation_results
        if item.closed_loop_score == result.best_score and item.stage == "stage3_joint_local"
    )
    if selected.finalist_rank != 4:
        raise AssertionError("final tune was not the best candidate validated in the final phase")
    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    if config.get("final_phase_candidate_count") != 3:
        raise AssertionError("saved config did not record the final-phase candidate count")
    if config.get("final_tuning_stage") != "stage3_joint_local":
        raise AssertionError("saved final tune did not come from the joint phase")
    if config.get("active_vs_passive_improvement") is not None:
        raise AssertionError("passive run should not report active-versus-passive improvement")


def test_disabled_active_winner_stays_disabled_in_joint_phase(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    runner = _FakeValidationRunner(
        {
            1: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0},
            2: {"route_completion_success": True, "eval_filtered_rmse_m": 0.4},
            3: {"route_completion_success": True, "eval_filtered_rmse_m": 2.0},
            4: {"route_completion_success": True, "eval_filtered_rmse_m": 0.5},
        }
    )
    request = _with_stage_budgets(
        _request(
            tmp_path,
            (log_path,),
            sensor,
            tracking_mode=TRACKING_ACTIVE,
            strategy="random_plus_coordinate_refinement",
        ),
        passive=1,
        active=2,
        joint=1,
    )

    result = ClosedLoopBenchmarkAutoTuner(validation_runner=runner).run(request)
    final_request = runner.requests[-1]
    if float(final_request.finalist.candidate_tune.get("enable_control_input_prediction", 1.0)) != 0.0:
        raise AssertionError("joint phase re-enabled a disabled active winner")
    if float(result.best_tune.get("enable_control_input_prediction", 1.0)) != 0.0:
        raise AssertionError("final saved tune did not preserve the disabled winner")
    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    if config.get("final_tune_uses_control") is not False:
        raise AssertionError("saved metadata did not preserve the disabled final tune")
    expected_improvement = float(config["best_passive_score"]) - float(config["final_objective_score"])
    if float(config.get("active_vs_passive_improvement") or 0.0) != pytest.approx(expected_improvement):
        raise AssertionError("active-versus-passive improvement did not use the final saved tune")


def test_closed_loop_autotuner_stores_failure_classes_in_trial_results(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    runner = _FakeValidationRunner(
        {
            1: {
                "route_completion_success": False,
                "route_aborted": True,
                "abort_reason": "Vehicle stuck: speed 0.00 m/s, no route progress for 12.0s",
                "eval_filtered_rmse_m": 1.0,
            }
        }
    )
    request = _with_stage_budgets(
        _request(tmp_path, (log_path,), sensor, tracking_mode=TRACKING_PASSIVE),
        passive=1,
        active=0,
        joint=0,
    )

    result = ClosedLoopBenchmarkAutoTuner(validation_runner=runner).run(request)
    validation = result.validation_results[0]
    if validation.failure_class != "vehicle_stuck_no_progress":
        raise AssertionError(f"failure class was not stored on validation result: {validation.failure_class}")
    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    direct_results = config.get("direct_closed_loop_trial_results") or []
    if not direct_results or direct_results[0].get("failure_class") != "vehicle_stuck_no_progress":
        raise AssertionError("saved direct trial result did not include failure class")


def test_adaptive_search_probes_one_family_and_protects_best_baseline(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    runner = _FakeValidationRunner(
        {
            1: {"route_completion_success": True, "eval_filtered_rmse_m": 0.5},
            2: {"route_completion_success": True, "eval_filtered_rmse_m": 4.0},
            3: {"route_completion_success": True, "eval_filtered_rmse_m": 4.5},
            4: {"route_completion_success": True, "eval_filtered_rmse_m": 5.0},
            5: {"route_completion_success": True, "eval_filtered_rmse_m": 5.5},
        }
    )
    request = _with_stage_budgets(
        _request(tmp_path, (log_path,), sensor, tracking_mode=TRACKING_PASSIVE, strategy="random_plus_coordinate_refinement"),
        passive=5,
        active=0,
        joint=0,
    )

    result = ClosedLoopBenchmarkAutoTuner(validation_runner=runner).run(request)
    for trial_request in runner.requests[1:]:
        if len(trial_request.affected_families) > 1:
            raise AssertionError("early adaptive trial changed more than one parameter family")
        families = {
            change.get("parameter_family")
            for change in trial_request.changed_parameters.values()
            if isinstance(change, dict)
        }
        if families != set(trial_request.affected_families):
            raise AssertionError("trial changed-parameter attribution did not match selected family")

    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    history = config.get("bound_adaptation_history") or []
    if not any("protected current best" in " ".join(item.get("decisions") or []) for item in history):
        raise AssertionError("repeated degradation did not tighten bounds around the protected baseline")
    if config.get("final_objective_score") != config.get("baseline_objective_score"):
        raise AssertionError("degraded adaptive trials displaced the protected context baseline")
    direct_results = config.get("direct_closed_loop_trial_results") or []
    attributed = next((item for item in direct_results if item.get("parameter_attribution")), None)
    if attributed is None:
        raise AssertionError("adaptive trial did not save per-parameter attribution records")
    record = attributed["parameter_attribution"][0]
    required = {
        "baseline_value",
        "trial_value",
        "absolute_change",
        "relative_change_percent",
        "parameter_family",
        "stage",
        "score",
        "rmse_m",
        "nis",
        "nees",
        "mean_cte_m",
        "max_cte_m",
        "failure_class",
        "route_completion_success",
    }
    if not required.issubset(record):
        raise AssertionError(f"parameter attribution record is incomplete: {record}")


def test_high_nees_biases_process_uncertainty_upward(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    runner = _FakeValidationRunner(
        {
            1: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0, "driving_mean_position_nees": 2.0},
            2: {"route_completion_success": True, "eval_filtered_rmse_m": 2.0, "driving_mean_position_nees": 20.0},
            3: {"route_completion_success": True, "eval_filtered_rmse_m": 1.5, "driving_mean_position_nees": 2.0},
        }
    )
    request = _with_stage_budgets(
        _request(tmp_path, (log_path,), sensor, tracking_mode=TRACKING_PASSIVE, strategy="random_plus_coordinate_refinement"),
        passive=3,
        active=0,
        joint=0,
    )

    result = ClosedLoopBenchmarkAutoTuner(validation_runner=runner).run(request)
    if runner.requests[2].affected_families != ("Q_acceleration_jerk",):
        raise AssertionError("adaptive family scheduler did not continue the available process-Q family")
    process_change = runner.requests[2].changed_parameters.get("process_jerk_stddev_mps3")
    if not isinstance(process_change, dict) or float(process_change.get("delta") or 0.0) <= 0.0:
        raise AssertionError("high NEES did not bias the next process-noise probe upward")
    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    history_text = json.dumps(config.get("bound_adaptation_history") or [])
    if "high NEES" not in history_text or "biased" not in history_text:
        raise AssertionError("high-NEES adaptation decision was not saved")


def test_control_instability_shrinks_active_parameter_families(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    runner = _FakeValidationRunner(
        {
            1: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0},
            2: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0},
            3: {"route_completion_success": True, "eval_filtered_rmse_m": 1.1},
            4: {"route_completion_success": True, "eval_filtered_rmse_m": 1.15},
            5: {"route_completion_success": True, "eval_filtered_rmse_m": 1.2},
            6: {
                "route_completion_success": False,
                "route_aborted": True,
                "abort_reason": "Control instability and steering oscillation",
                "control_instability_score": 5.0,
            },
        }
    )
    request = _with_stage_budgets(
        _request(tmp_path, (log_path,), sensor, tracking_mode=TRACKING_ACTIVE, strategy="random_plus_coordinate_refinement"),
        passive=1,
        active=5,
        joint=0,
    )

    result = ClosedLoopBenchmarkAutoTuner(validation_runner=runner).run(request)
    config = json.loads(result.saved_config_path.read_text(encoding="utf-8"))
    diagnostics = config.get("parameter_family_diagnostics") or {}
    active_scales = [
        float(item.get("final_bound_scale"))
        for family, item in diagnostics.items()
        if family.startswith("active_") and isinstance(item, dict) and item.get("final_bound_scale") is not None
    ]
    if not active_scales or max(active_scales) >= 0.5:
        raise AssertionError("control instability did not shrink active-control family bounds")
    if "control instability: shrunk all active-control families" not in json.dumps(config.get("bound_adaptation_history") or []):
        raise AssertionError("active-control shrink decision was not recorded")


def test_context_baseline_does_not_use_route_geometry(tmp_path: Path) -> None:
    log_path, sensor = _recorded_log(tmp_path, "route_001")
    base_request = _with_stage_budgets(
        _request(tmp_path, (log_path,), sensor, tracking_mode=TRACKING_PASSIVE),
        passive=1,
        active=0,
        joint=0,
    )
    route_a = ClosedLoopValidationRoute(
        "route_a",
        "Town01",
        "route_a",
        route_data={"start": {"x": 0.0, "y": 0.0}, "goal": {"x": 10.0, "y": 0.0}},
    )
    route_b = ClosedLoopValidationRoute(
        "route_b",
        "Town01",
        "route_b",
        route_data={
            "start": {"x": -100.0, "y": 50.0},
            "goal": {"x": 900.0, "y": -700.0},
            "waypoints": [{"x": index, "y": index * index} for index in range(20)],
        },
    )
    request_a = ClosedLoopAutoTuneRequest(
        **{**base_request.to_dict(), "offline_log_paths": base_request.offline_log_paths, "validation_routes": (route_a,)}
    )
    request_b = ClosedLoopAutoTuneRequest(
        **{**base_request.to_dict(), "offline_log_paths": base_request.offline_log_paths, "validation_routes": (route_b,)}
    )
    runner_a = _FakeValidationRunner()
    runner_b = _FakeValidationRunner()

    result_a = ClosedLoopBenchmarkAutoTuner(validation_runner=runner_a).run(request_a)
    result_b = ClosedLoopBenchmarkAutoTuner(validation_runner=runner_b).run(request_b)

    if runner_a.requests[0].finalist.candidate_tune != runner_b.requests[0].finalist.candidate_tune:
        raise AssertionError("context baseline changed after only route geometry changed")
    config = json.loads(result_a.saved_config_path.read_text(encoding="utf-8"))
    policy = config.get("adaptive_search_policy") if isinstance(config.get("adaptive_search_policy"), dict) else {}
    if policy.get("route_geometry_preanalysis") is not False:
        raise AssertionError("saved adaptive policy did not explicitly disable route geometry pre-analysis")


def test_closed_loop_validation_route_runner_uses_compact_route_artifact_folders(tmp_path: Path) -> None:
    runner = RouteTestRunner.__new__(RouteTestRunner)
    runner._run_folder = tmp_path / "run"
    runner._config = SimpleNamespace(metadata={"compact_route_output": True})
    route = SimpleNamespace(name="mahalle validation route")

    route_folder = runner._route_output_folder(route, 0)
    if route_folder != tmp_path / "run" / "routes" / "r001":
        raise AssertionError(f"closed-loop validation route artifacts were not compacted: {route_folder}")

    runner._config = SimpleNamespace(metadata={})
    regular_folder = runner._route_output_folder(route, 0)
    if regular_folder.name != "route_001_mahalle_validation_route":
        raise AssertionError(f"regular benchmark route folder naming changed unexpectedly: {regular_folder}")


def test_direct_closed_loop_route_runner_disables_automatic_plots() -> None:
    runner = RouteTestRunner.__new__(RouteTestRunner)
    runner._config = SimpleNamespace(metadata={"direct_closed_loop_mode": True})
    if runner._automatic_plot_generation_enabled():
        raise AssertionError("direct closed-loop trials re-enabled automatic route or aggregate plots")

    runner._config = SimpleNamespace(metadata={})
    if runner._automatic_plot_generation_enabled() != bool(BENCHMARK.generate_plots_on_completion):
        raise AssertionError("normal benchmark automatic plot policy changed")


def test_route_runner_one_attempt_policy_fails_without_retrying_candidate() -> None:
    if _configured_route_attempt_limit(SimpleNamespace(metadata={"max_route_attempts": 1})) != 1:
        raise AssertionError("direct benchmark metadata did not configure one route attempt")
    if _configured_route_attempt_limit(SimpleNamespace(metadata={})) != MAX_ROUTE_ATTEMPTS:
        raise AssertionError("normal visual benchmark retry default changed")

    runner = RouteTestRunner.__new__(RouteTestRunner)
    runner._active = True
    runner._automated = True
    runner._route_running = True
    runner._current_route = SimpleNamespace(name="direct_trial_route")
    runner._current_route_index = 0
    runner._current_attempt = 1
    runner._max_route_attempts = 1
    runner._attempt_failures_by_route = {}
    runner._last_failure_reason = ""
    runner._route_status = "running"
    runner._started_monotonic = None
    runner._status_text = ""

    finish_calls: list[dict[str, object]] = []
    advance_calls: list[object] = []
    retry_calls: list[object] = []

    def finish_current_route(**kwargs: object) -> Path:
        finish_calls.append(dict(kwargs))
        runner._route_running = False
        return Path("failed_trial")

    runner._finish_current_route = finish_current_route
    runner._advance_after_route = lambda active_map_name: advance_calls.append(active_map_name)
    runner.begin_current_route = lambda active_map_name: retry_calls.append(active_map_name) or True

    runner.fail_current_attempt(
        reason="Vehicle stuck: no route progress",
        simulation_time_s=12.0,
        active_map_name="Town01",
    )

    if retry_calls:
        raise AssertionError("one-attempt direct trial retried the same candidate")
    if len(finish_calls) != 1 or finish_calls[0].get("record_result") is not True:
        raise AssertionError("one-attempt direct failure was not recorded as the final trial result")
    if finish_calls[0].get("final_status") != "TEST_NOT_COMPLETED":
        raise AssertionError("one-attempt direct failure did not end as TEST_NOT_COMPLETED")
    if len(advance_calls) != 1:
        raise AssertionError("one-attempt direct failure did not advance to the next candidate")


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
    if config.get("candidate_generation_strategy") != "direct_closed_loop_random":
        raise AssertionError("closed-loop backend did not record fallback strategy when Optuna was unavailable")
    if config.get("optuna_available") is not False:
        raise AssertionError("closed-loop backend did not record Optuna availability")


def test_closed_loop_request_defaults_to_legacy_adaptive_random_strategy() -> None:
    if ClosedLoopAutoTuneRequest.__dataclass_fields__["strategy"].default != "random_plus_coordinate_refinement":
        raise AssertionError("closed-loop request default no longer preserves the legacy adaptive/random strategy")


def test_optuna_tpe_uses_real_closed_loop_scores_and_changes_candidates(tmp_path: Path) -> None:
    sensor = {"preset_name": "Synthetic", "gnss_position_stddev_m": 1.25, "imu_accel_stddev_mps2": 0.45}
    optuna_runner = _FakeValidationRunner(
        {
            1: {"route_completion_success": True, "eval_filtered_rmse_m": 3.0},
            2: {"route_completion_success": True, "eval_filtered_rmse_m": 2.0},
            3: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0},
        }
    )
    legacy_runner = _FakeValidationRunner(
        {
            1: {"route_completion_success": True, "eval_filtered_rmse_m": 3.0},
            2: {"route_completion_success": True, "eval_filtered_rmse_m": 2.0},
            3: {"route_completion_success": True, "eval_filtered_rmse_m": 1.0},
        }
    )
    optuna_request = _with_stage_budgets(
        _request(tmp_path / "optuna", (), sensor, strategy="optuna_tpe"),
        passive=3,
        active=0,
        joint=0,
    )
    legacy_request = _with_stage_budgets(
        _request(tmp_path / "legacy", (), sensor, strategy="random_plus_coordinate_refinement"),
        passive=3,
        active=0,
        joint=0,
    )

    optuna_result = ClosedLoopBenchmarkAutoTuner(validation_runner=optuna_runner).run(optuna_request)
    ClosedLoopBenchmarkAutoTuner(validation_runner=legacy_runner).run(legacy_request)

    if optuna_runner.requests[1].finalist.candidate_tune == legacy_runner.requests[1].finalist.candidate_tune:
        raise AssertionError("Optuna selection did not change closed-loop candidate generation")
    config = json.loads(optuna_result.saved_config_path.read_text(encoding="utf-8"))
    if config.get("candidate_generation_strategy") != "direct_closed_loop_optuna_tpe":
        raise AssertionError("Optuna strategy was not recorded in the saved tune config")
    optimizer_summaries = config.get("optimizer_stage_summaries") or []
    if not optimizer_summaries:
        raise AssertionError("Optuna stage diagnostics were not saved")
    reported_values = [
        float(trial["value"])
        for summary in optimizer_summaries
        for trial in summary.get("trials", [])
        if trial.get("value") is not None
    ]
    actual_scores = [float(item.closed_loop_score) for item in optuna_result.validation_results]
    if not set(actual_scores).issubset(set(reported_values)):
        raise AssertionError(f"Optuna did not receive real completed closed-loop scores: {reported_values}")


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
    filter_id: str = "ca_kf",
) -> ClosedLoopAutoTuneRequest:
    return ClosedLoopAutoTuneRequest(
        filter_id=filter_id,
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


def _with_stage_budgets(
    request: ClosedLoopAutoTuneRequest,
    *,
    passive: int,
    active: int,
    joint: int,
) -> ClosedLoopAutoTuneRequest:
    return ClosedLoopAutoTuneRequest(
        **{
            **request.to_dict(),
            "offline_log_paths": request.offline_log_paths,
            "validation_routes": request.validation_routes,
            "trial_count": passive + active + joint,
            "passive_model_trials": passive,
            "active_control_trials": active,
            "joint_fine_tune_trials": joint,
        }
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
