"""Closed-loop benchmark auto-tuning backend.

This module intentionally has no pygame/UI dependency and does not launch
CARLA by itself.  Closed-loop candidates are scored by an injected route
runner; each candidate score comes from a real controlled CARLA route trial.
Offline replay auto-tuning remains implemented separately in filter_auto_tuner.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import json
import math
from pathlib import Path
import random
from typing import Callable, Optional

from src.KalmanLab.registry import discover_filters
from src.evaluation import filter_auto_tuner as offline_auto_tuner_module
from src.evaluation.consistency_metrics import consistency_report_from_summaries
from src.evaluation.evaluation_artifacts import benchmark_root, read_json, slugify, unique_folder, write_json
from src.evaluation.filter_auto_tuner import (
    AutoTuneRequest,
    AutoTuneResult,
    AutoTuneTrialResult,
    OfflineBenchmarkAutoTuner,
    noise_profile_summary,
)
from src.evaluation.offline_replay_runner import _validate_windows_path_length
from src.evaluation.sensor_noise_tune_mapper import SensorNoiseTuneMapper, noise_signature, process_only_auto_tune_profile
from src.evaluation.tune_config_schema import (
    BENCHMARK_MODE_CLOSED_LOOP,
    SCHEMA_VERSION,
    TRACKING_ACTIVE,
    TRACKING_PASSIVE,
    TUNE_SCOPE_CLOSED_LOOP_VALIDATED,
    TuneCompatibility,
    build_closed_loop_schema_v2_config,
    closed_loop_tune_context,
    config_signature,
    noise_signature_slug,
)


ProgressCallback = Callable[[str, dict[str, object]], None]

ACTIVE_CONTROL_TUNE_KEYS = {
    "enable_control_input_prediction",
    "control_accel_gain_mps2",
    "control_brake_decel_gain_mps2",
    "control_steer_to_yaw_rate_gain",
    "control_input_timeout_s",
    "max_control_accel_delta_mps2",
    "max_control_yaw_rate_delta_radps",
    "max_control_speed_delta_mps",
    "imu_accel_control_stddev_mps2",
    "command_accel_stddev_mps2",
    "command_throttle_accel_gain_mps2",
    "command_brake_decel_gain_mps2",
    "command_max_accel_mps2",
    "command_yaw_rate_stddev_dps",
    "command_max_yaw_rate_dps",
}

PASSIVE_MODEL_STAGE = "stage1_passive_q_model"
CONTEXT_BASELINE_STAGE = "stage0_context_baseline"
ACTIVE_CONTROL_STAGE = "stage2_active_control"
JOINT_FINE_TUNE_STAGE = "stage3_joint_local"


@dataclass(frozen=True)
class ClosedLoopStageBudgets:
    passive_model_trials: int
    active_control_trials: int
    joint_fine_tune_trials: int

    @property
    def total_trials(self) -> int:
        return self.passive_model_trials + self.active_control_trials + self.joint_fine_tune_trials

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_passive_q_model_trials": self.passive_model_trials,
            "active_control_trials": self.active_control_trials,
            "joint_local_fine_tune_trials": self.joint_fine_tune_trials,
            "total_planned_trials": self.total_trials,
        }


@dataclass
class ParameterFamilyStats:
    trials: int = 0
    successes: int = 0
    failures: int = 0
    improvements: int = 0
    degradations: int = 0
    stable_trials: int = 0
    high_nis_count: int = 0
    high_nees_count: int = 0
    large_cte_count: int = 0
    upward_improvements: int = 0
    downward_improvements: int = 0
    upward_degradations: int = 0
    downward_degradations: int = 0
    score_delta_sum: float = 0.0
    failure_classes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AdaptiveStageState:
    stage: str
    baseline_tune: dict[str, object]
    params: list[dict[str, object]]
    best_tune: dict[str, object]
    best_score: Optional[float]
    baseline_score: Optional[float]
    family_by_key: dict[str, str]
    initial_bounds: dict[str, tuple[float, float]]
    family_scales: dict[str, float]
    family_direction_bias: dict[str, float]
    family_stats: dict[str, ParameterFamilyStats]
    parameter_attempts: dict[str, int]
    adaptation_history: list[dict[str, object]]
    non_improving_trials: int = 0
    completed_trials: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "baseline_score": self.baseline_score,
            "best_score": self.best_score,
            "completed_trials": self.completed_trials,
            "non_improving_trials": self.non_improving_trials,
            "family_scales": dict(self.family_scales),
            "family_direction_bias": dict(self.family_direction_bias),
            "parameter_family_stats": {
                family: stats.to_dict()
                for family, stats in sorted(self.family_stats.items())
            },
            "adaptation_history": list(self.adaptation_history),
        }


@dataclass(frozen=True)
class AdaptiveCandidatePlan:
    tune: dict[str, object]
    anchor_tune: dict[str, object]
    affected_families: tuple[str, ...]
    candidate_rationale: str


@dataclass(frozen=True)
class PendingClosedLoopAutoTuneSession:
    selected_filter: str
    tracking_mode: str
    offline_log_paths: tuple[str, ...]
    noise_signature: str
    validation_route_name: str
    validation_route_map: str
    validation_route_id: str
    sensor_config: dict[str, object]
    vehicle_behavior_config: dict[str, object]
    actuator_realism_config: dict[str, object]
    trial_count: int
    finalist_count: int
    strategy: str
    output_root: str
    passive_model_trials: Optional[int] = None
    active_control_trials: Optional[int] = None
    joint_fine_tune_trials: Optional[int] = None
    created_at: str = ""
    handoff_path: Optional[str] = None
    base_tune: dict[str, object] = field(default_factory=dict)
    auto_tune_profile: dict[str, object] = field(default_factory=dict)
    validation_route_data: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


@dataclass(frozen=True)
class ClosedLoopValidationRoute:
    name: str
    map_name: str
    route_id: str = ""
    route_signature: str = ""
    route_data: dict[str, object] = field(default_factory=dict)

    @staticmethod
    def from_object(value: object) -> "ClosedLoopValidationRoute":
        if isinstance(value, ClosedLoopValidationRoute):
            return value
        if isinstance(value, dict):
            name = str(value.get("name") or value.get("route_name") or "")
            map_name = str(value.get("map_name") or value.get("map") or "")
            route_id = str(value.get("route_id") or value.get("id") or "")
            route_data = value.get("route_data")
            if not isinstance(route_data, dict):
                route_data = {
                    key: value[key]
                    for key in ("name", "start", "goal", "map_name", "created_from")
                    if key in value
                }
            route_signature = str(value.get("route_signature") or value.get("signature") or config_signature(route_data))
            return ClosedLoopValidationRoute(
                name=name,
                map_name=map_name,
                route_id=route_id,
                route_signature=route_signature,
                route_data=dict(route_data),
            )
        name = str(getattr(value, "name", "") or getattr(value, "route_name", ""))
        map_name = str(getattr(value, "map_name", "") or getattr(value, "map", ""))
        route_id = str(getattr(value, "route_id", "") or getattr(value, "id", ""))
        route_data: dict[str, object] = {}
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, dict):
                route_data = dict(payload)
        route_signature = str(getattr(value, "route_signature", "") or getattr(value, "signature", "") or config_signature(route_data))
        return ClosedLoopValidationRoute(
            name=name,
            map_name=map_name,
            route_id=route_id,
            route_signature=route_signature,
            route_data=route_data,
        )

    def identity(self) -> str:
        return self.route_id or self.route_signature or f"{self.name}@{self.map_name}"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "map_name": self.map_name,
            "route_id": self.route_id,
            "route_signature": self.route_signature,
            "route_data": dict(self.route_data),
        }


@dataclass(frozen=True)
class ClosedLoopAutoTuneRequest:
    filter_id: str
    tracking_mode: str
    validation_routes: tuple[ClosedLoopValidationRoute, ...]
    sensor_noise_config: dict[str, object]
    vehicle_behavior_config: dict[str, object]
    actuator_realism_config: dict[str, object]
    offline_log_paths: tuple[Path, ...] = ()
    base_tune: dict[str, object] = field(default_factory=dict)
    auto_tune_profile: dict[str, object] = field(default_factory=dict)
    sensor_noise_profile: str = "Custom"
    vehicle_behavior_profile: str = "Custom"
    actuator_realism_enabled: bool = True
    actuator_realism_profile: str = "Custom"
    trial_count: int = 30
    passive_model_trials: Optional[int] = None
    active_control_trials: Optional[int] = None
    joint_fine_tune_trials: Optional[int] = None
    finalist_count: int = 3
    strategy: str = "optuna_tpe"
    output_root: str = "benchmark_results"
    random_seed: int = 4084
    allow_mixed_noise_logs: bool = False
    keep_trial_outputs: bool = True
    generate_trial_plots: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    @staticmethod
    def from_pending_session(
        session: PendingClosedLoopAutoTuneSession,
        *,
        base_tune: Optional[dict[str, object]] = None,
        auto_tune_profile: Optional[dict[str, object]] = None,
    ) -> "ClosedLoopAutoTuneRequest":
        return ClosedLoopAutoTuneRequest(
            filter_id=session.selected_filter,
            tracking_mode=session.tracking_mode,
            offline_log_paths=tuple(Path(path) for path in session.offline_log_paths),
            validation_routes=(
                ClosedLoopValidationRoute(
                    name=session.validation_route_name,
                    map_name=session.validation_route_map,
                    route_id=session.validation_route_id,
                    route_data=dict(session.validation_route_data),
                ),
            ),
            sensor_noise_config=dict(session.sensor_config),
            vehicle_behavior_config=dict(session.vehicle_behavior_config),
            actuator_realism_config=dict(session.actuator_realism_config),
            base_tune=dict(base_tune or session.base_tune),
            auto_tune_profile=dict(auto_tune_profile or session.auto_tune_profile),
            sensor_noise_profile=str(session.sensor_config.get("preset_name") or "Custom"),
            vehicle_behavior_profile=str(session.vehicle_behavior_config.get("preset_name") or "Custom"),
            actuator_realism_enabled=bool(session.actuator_realism_config),
            actuator_realism_profile=str(session.actuator_realism_config.get("preset_name") or "Custom"),
            trial_count=int(session.trial_count),
            passive_model_trials=session.passive_model_trials,
            active_control_trials=session.active_control_trials,
            joint_fine_tune_trials=session.joint_fine_tune_trials,
            finalist_count=int(session.finalist_count),
            strategy=session.strategy,
            output_root=session.output_root,
            metadata={"pending_session": session.to_dict()},
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["offline_log_paths"] = [str(path) for path in self.offline_log_paths]
        data["validation_routes"] = [route.to_dict() for route in self.validation_routes]
        return data


@dataclass(frozen=True)
class ClosedLoopFinalist:
    rank: int
    candidate_tune: dict[str, object]
    offline_score: float
    offline_metrics: dict[str, object]
    trial_index: int
    source_output_folder: Optional[Path]

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "candidate_tune": dict(self.candidate_tune),
            "offline_score": self.offline_score,
            "offline_metrics": dict(self.offline_metrics),
            "trial_index": self.trial_index,
            "source_output_folder": str(self.source_output_folder) if self.source_output_folder else None,
        }


@dataclass(frozen=True)
class ClosedLoopValidationRequest:
    filter_id: str
    tracking_mode: str
    finalist: ClosedLoopFinalist
    validation_route: ClosedLoopValidationRoute
    sensor_noise_config: dict[str, object]
    vehicle_behavior_config: dict[str, object]
    actuator_realism_config: dict[str, object]
    output_folder: Path
    stage: str = ""
    stage_trial_index: int = 0
    stage_trial_total: int = 0
    changed_parameters: dict[str, object] = field(default_factory=dict)
    changed_parameters_summary: str = ""
    affected_families: tuple[str, ...] = ()
    adaptation_decision: str = ""


@dataclass(frozen=True)
class ClosedLoopValidationResult:
    finalist_rank: int
    candidate_tune: dict[str, object]
    route_completion_success: bool
    route_aborted: bool
    timeout: bool
    abort_reason: str
    completion_time_s: Optional[float]
    eval_filtered_rmse_m: Optional[float]
    filtered_rmse_m: Optional[float]
    mean_cross_track_error_m: Optional[float]
    max_cross_track_error_m: Optional[float]
    nis_by_type_summary: dict[str, object]
    position_nees_source: str
    mean_position_nees: Optional[float]
    mean_position_nees_diagonal_approx: Optional[float]
    raw_metrics: dict[str, object]
    output_folder: Optional[Path]
    closed_loop_score: float
    stage: str = ""
    stage_trial_index: int = 0
    stage_trial_total: int = 0
    failure_class: str = ""
    changed_parameters: dict[str, object] = field(default_factory=dict)
    changed_parameters_summary: str = ""
    affected_families: tuple[str, ...] = ()
    adaptation_decision: str = ""
    outcome_class: str = ""
    parameter_attribution: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "finalist_rank": self.finalist_rank,
            "candidate_tune": dict(self.candidate_tune),
            "route_completion_success": self.route_completion_success,
            "route_aborted": self.route_aborted,
            "timeout": self.timeout,
            "abort_reason": self.abort_reason,
            "completion_time_s": self.completion_time_s,
            "eval_filtered_rmse_m": self.eval_filtered_rmse_m,
            "filtered_rmse_m": self.filtered_rmse_m,
            "mean_cross_track_error_m": self.mean_cross_track_error_m,
            "max_cross_track_error_m": self.max_cross_track_error_m,
            "nis_by_type_summary": self.nis_by_type_summary,
            "position_nees_source": self.position_nees_source,
            "mean_position_nees": self.mean_position_nees,
            "mean_position_nees_diagonal_approx": self.mean_position_nees_diagonal_approx,
            "raw_metrics": dict(self.raw_metrics),
            "output_folder": str(self.output_folder) if self.output_folder else None,
            "closed_loop_score": self.closed_loop_score,
            "stage": self.stage,
            "stage_trial_index": self.stage_trial_index,
            "stage_trial_total": self.stage_trial_total,
            "failure_class": self.failure_class,
            "changed_parameters": dict(self.changed_parameters),
            "changed_parameters_summary": self.changed_parameters_summary,
            "affected_families": list(self.affected_families),
            "adaptation_decision": self.adaptation_decision,
            "outcome_class": self.outcome_class,
            "parameter_attribution": [dict(item) for item in self.parameter_attribution],
        }


@dataclass(frozen=True)
class ClosedLoopAutoTuneResult:
    filter_id: str
    tracking_mode: str
    best_tune: dict[str, object]
    best_score: Optional[float]
    best_metrics: dict[str, object]
    finalists: tuple[ClosedLoopFinalist, ...]
    validation_results: tuple[ClosedLoopValidationResult, ...]
    output_folder: Path
    saved_config_path: Optional[Path]
    offline_auto_tune_result: AutoTuneResult


class ClosedLoopBenchmarkRunnerAdapter:
    """Placeholder adapter; UI/app integration will provide a real runner later."""

    def run(self, request: ClosedLoopValidationRequest) -> dict[str, object]:
        raise NotImplementedError("Closed-loop validation runner is not wired yet. Provide a runner dependency.")


class ClosedLoopBenchmarkAutoTuner:
    """Direct closed-loop auto tuner backend.

    Each search trial applies one candidate tune and delegates a real
    closed-loop CARLA route trial to the injected runner.  When Optuna is
    available, the study is told the closed-loop objective after each trial.
    """

    def __init__(
        self,
        *,
        offline_tuner: Optional[object] = None,
        validation_runner: Optional[object] = None,
    ) -> None:
        self._records = {record.filter_id: record for record in discover_filters() if record.valid}
        self._offline_tuner = offline_tuner or OfflineBenchmarkAutoTuner()
        self._validation_runner = validation_runner or ClosedLoopBenchmarkRunnerAdapter()

    def run(
        self,
        request_or_session: ClosedLoopAutoTuneRequest | PendingClosedLoopAutoTuneSession,
        progress_callback: Optional[ProgressCallback] = None,
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> ClosedLoopAutoTuneResult:
        request = self._normalize_request(request_or_session)
        record = self._records.get(request.filter_id)
        if record is None or record.filter_class is None:
            raise ValueError(f"Filter is not available: {request.filter_id}")
        request = self._fill_defaults_from_record(request, record)
        self._validate_request(request)

        validation_route = request.validation_routes[0]
        logs = [_offline_log_metadata(path) for path in request.offline_log_paths]
        log_noise = noise_profile_summary(logs)
        if log_noise.get("mixed") and not request.allow_mixed_noise_logs:
            raise ValueError(
                "Selected closed-loop auto-tune logs use mixed sensor noise signatures. "
                "Select logs with the same noise profile/signature."
            )
        selected_noise_signature = noise_signature(request.sensor_noise_config)
        if log_noise.get("signature") and log_noise.get("signature") != selected_noise_signature:
            raise ValueError("Selected offline logs are incompatible with the selected sensor noise signature.")

        locked = SensorNoiseTuneMapper.locked_values(
            request.filter_id,
            dict(request.base_tune),
            request.sensor_noise_config,
            tuple(record.tune_specs),
        )
        base_tune = dict(request.base_tune)
        base_tune.update(locked.values)
        context_baseline = _context_aware_baseline_tune(
            base_tune,
            request,
            tuple(record.tune_specs),
        )
        stage_budgets = closed_loop_stage_budgets(request)

        run_folder = unique_folder(
            _closed_loop_physical_root(request.output_root, request.filter_id, request.tracking_mode),
            _run_folder_name(),
        )
        _validate_windows_path_length(
            run_folder / "closed_loop_auto_tune_summary.json",
            "Closed-loop auto-tune summary file",
        )
        run_folder.mkdir(parents=True, exist_ok=False)
        write_json(run_folder / "closed_loop_auto_tune_request.json", request.to_dict())

        strategy = _resolve_candidate_generation_strategy(request.strategy)
        search_profile = _closed_loop_search_profile(
            request.auto_tune_profile,
            TRACKING_PASSIVE,
            request.actuator_realism_config,
        )
        passive_params = _passive_model_search_params(
            _closed_loop_search_params(search_profile),
            context_baseline,
        )
        active_params = _active_control_search_params(
            context_baseline,
            tuple(record.tune_specs),
            request.actuator_realism_config,
        )
        total_trials = stage_budgets.total_trials
        _emit(
            progress_callback,
            "search_started",
            {
                "trial_count": total_trials,
                "stage_budgets": stage_budgets.to_dict(),
                "strategy": strategy,
                "search_param_count": len(passive_params) + len(active_params),
                "passive_search_param_count": len(passive_params),
                "active_search_param_count": len(active_params),
                "tracking_mode": request.tracking_mode,
                "actuator_realism_profile": request.actuator_realism_profile,
                "actuator_search_policy": _actuator_search_policy(request.actuator_realism_config),
            },
        )
        validation_results: list[ClosedLoopValidationResult] = []
        rng = random.Random(int(request.random_seed))
        failure_counts: dict[str, int] = {}
        seen_candidates: set[str] = set()
        stage_summaries: list[dict[str, object]] = []
        adaptive_stage_states: list[AdaptiveStageState] = []
        best_score_so_far: Optional[float] = None
        global_trial_index = 0

        def run_stage(
            *,
            stage: str,
            stage_label: str,
            stage_trial_count: int,
            stage_tracking_mode: str,
            stage_baseline: dict[str, object],
            params: list[dict[str, object]],
            evaluate_baseline_first: bool,
            protected_result: Optional[ClosedLoopValidationResult] = None,
        ) -> list[ClosedLoopValidationResult]:
            nonlocal best_score_so_far, global_trial_index
            if stage_trial_count <= 0:
                return []
            if not params and not evaluate_baseline_first:
                return []
            if not params and evaluate_baseline_first:
                stage_trial_count = 1
            _emit(
                progress_callback,
                "stage_started",
                {
                    "stage": stage,
                    "stage_label": stage_label,
                    "stage_trial_count": stage_trial_count,
                    "trial_total": total_trials,
                    "tracking_mode": stage_tracking_mode,
                    "search_param_count": len(params),
                },
            )
            stage_results: list[ClosedLoopValidationResult] = []
            adaptive_state = _adaptive_stage_state(
                stage=stage,
                stage_baseline=stage_baseline,
                params=params,
                protected_result=protected_result,
            )
            adaptive_stage_states.append(adaptive_state)
            for stage_trial_index in range(1, stage_trial_count + 1):
                if stop_requested is not None and stop_requested():
                    break
                global_trial_index += 1
                baseline_trial = evaluate_baseline_first and stage_trial_index == 1
                plan = _adaptive_candidate_plan(
                    adaptive_state,
                    rng,
                    baseline_trial=baseline_trial,
                    failure_counts=failure_counts,
                )
                candidate_tune = _ensure_unique_candidate(
                    plan.tune,
                    plan.anchor_tune,
                    [
                        param
                        for param in params
                        if adaptive_state.family_by_key.get(str(param.get("key") or "")) in plan.affected_families
                    ],
                    stage_tracking_mode,
                    seen_candidates,
                    rng,
                    allow_existing=baseline_trial,
                )
                if candidate_tune is not plan.tune:
                    plan = replace(plan, tune=candidate_tune)
                changed_parameters = _changed_parameters(plan.anchor_tune, candidate_tune, adaptive_state.family_by_key)
                changed_summary = _changed_parameters_summary(changed_parameters)
                candidate_type = "context_baseline" if stage == CONTEXT_BASELINE_STAGE else "adaptive_family_local"
                finalist = ClosedLoopFinalist(
                    rank=global_trial_index,
                    candidate_tune=candidate_tune,
                    offline_score=0.0,
                    offline_metrics={
                        "candidate_generation_stage": stage,
                        "candidate_type": candidate_type,
                        "tracking_mode": stage_tracking_mode,
                        "actuator_search_policy": _actuator_search_policy(request.actuator_realism_config),
                        "changed_parameters": changed_parameters,
                        "changed_parameters_summary": changed_summary,
                        "affected_families": list(plan.affected_families),
                        "candidate_rationale": plan.candidate_rationale,
                    },
                    trial_index=global_trial_index,
                    source_output_folder=None,
                )
                validation_request = ClosedLoopValidationRequest(
                    filter_id=request.filter_id,
                    tracking_mode=stage_tracking_mode,
                    finalist=finalist,
                    validation_route=validation_route,
                    sensor_noise_config=dict(request.sensor_noise_config),
                    vehicle_behavior_config=dict(request.vehicle_behavior_config),
                    actuator_realism_config=dict(request.actuator_realism_config),
                    output_folder=run_folder / "trials" / f"t{global_trial_index:03d}",
                    stage=stage,
                    stage_trial_index=stage_trial_index,
                    stage_trial_total=stage_trial_count,
                    changed_parameters=changed_parameters,
                    changed_parameters_summary=changed_summary,
                    affected_families=plan.affected_families,
                    adaptation_decision=plan.candidate_rationale,
                )
                _emit(
                    progress_callback,
                    "trial_started",
                    {
                        "trial_index": global_trial_index,
                        "trial_total": total_trials,
                        "stage": stage,
                        "stage_label": stage_label,
                        "stage_trial_index": stage_trial_index,
                        "stage_trial_total": stage_trial_count,
                        "candidate_tune": dict(candidate_tune),
                        "changed_parameters": changed_parameters,
                        "changed_parameters_summary": changed_summary,
                        "affected_families": list(plan.affected_families),
                        "adaptation_decision": plan.candidate_rationale,
                        "route_name": validation_route.name,
                        "tracking_mode": stage_tracking_mode,
                        "actuator_realism_profile": request.actuator_realism_profile,
                        "best_score": best_score_so_far,
                    },
                )
                runner_result = self._run_validation(validation_request)
                validation = _validation_result_from_runner(runner_result, validation_request)
                outcome_class = _trial_outcome_class(validation, adaptive_state.best_score)
                adaptation_decisions = _update_adaptive_stage_state(
                    adaptive_state,
                    validation,
                    changed_parameters,
                    outcome_class,
                    plan.affected_families,
                )
                adaptation_decision = "; ".join(adaptation_decisions) if adaptation_decisions else plan.candidate_rationale
                parameter_attribution = _parameter_attribution_records(
                    changed_parameters,
                    validation,
                    outcome_class,
                    adaptation_decision,
                )
                raw_metrics = dict(validation.raw_metrics)
                raw_metrics.update(
                    {
                        "tuning_stage": stage,
                        "stage_trial_index": stage_trial_index,
                        "stage_trial_total": stage_trial_count,
                        "failure_class": validation.failure_class,
                        "changed_parameters": changed_parameters,
                        "changed_parameters_summary": changed_summary,
                        "affected_families": list(plan.affected_families),
                        "adaptation_decision": adaptation_decision,
                        "outcome_class": outcome_class,
                        "parameter_attribution": [dict(item) for item in parameter_attribution],
                        "candidate_type": candidate_type,
                    }
                )
                validation = replace(
                    validation,
                    stage=stage,
                    stage_trial_index=stage_trial_index,
                    stage_trial_total=stage_trial_count,
                    changed_parameters=changed_parameters,
                    changed_parameters_summary=changed_summary,
                    affected_families=plan.affected_families,
                    adaptation_decision=adaptation_decision,
                    outcome_class=outcome_class,
                    parameter_attribution=parameter_attribution,
                    raw_metrics=raw_metrics,
                )
                validation_results.append(validation)
                stage_results.append(validation)
                failure_class = validation.failure_class
                if failure_class:
                    failure_counts[failure_class] = failure_counts.get(failure_class, 0) + 1
                if best_score_so_far is None or validation.closed_loop_score < best_score_so_far:
                    best_score_so_far = float(validation.closed_loop_score)
                    _emit(
                        progress_callback,
                        "new_search_best",
                        {
                            "trial_index": global_trial_index,
                            "stage": stage,
                            "stage_label": stage_label,
                            "score": best_score_so_far,
                            "metrics": dict(validation.raw_metrics),
                            "candidate_tune": dict(candidate_tune),
                            "changed_parameters_summary": changed_summary,
                            "affected_families": list(plan.affected_families),
                            "adaptation_decision": adaptation_decision,
                        },
                    )
                write_json(run_folder / f"trial_{global_trial_index:03d}.json", validation.to_dict())
                _emit(
                    progress_callback,
                    "trial_finished",
                    {
                        "trial_index": global_trial_index,
                        "trial_total": total_trials,
                        "stage": stage,
                        "stage_label": stage_label,
                        "stage_trial_index": stage_trial_index,
                        "stage_trial_total": stage_trial_count,
                        "score": validation.closed_loop_score,
                        "best_score": best_score_so_far,
                        "failed": not validation.route_completion_success,
                        "failure_class": validation.failure_class,
                        "failure_reason": validation.abort_reason,
                        "changed_parameters": changed_parameters,
                        "changed_parameters_summary": changed_summary,
                        "affected_families": list(plan.affected_families),
                        "adaptation_decision": adaptation_decision,
                        "outcome_class": outcome_class,
                        "metrics": dict(validation.raw_metrics),
                        "route_completion_success": validation.route_completion_success,
                        "tracking_mode": stage_tracking_mode,
                        "actuator_realism_profile": request.actuator_realism_profile,
                    },
                )
            if stage_results:
                stage_best = min(stage_results, key=lambda item: item.closed_loop_score)
                stage_summaries.append(
                    {
                        "stage": stage,
                        "stage_label": stage_label,
                        "tracking_mode": stage_tracking_mode,
                        "planned_trials": stage_trial_count,
                        "completed_trials": len(stage_results),
                        "baseline_score": adaptive_state.baseline_score,
                        "best_trial_index": (
                            stage_best.finalist_rank
                            if adaptive_state.best_score == stage_best.closed_loop_score
                            else protected_result.finalist_rank if protected_result is not None else stage_best.finalist_rank
                        ),
                        "best_score": adaptive_state.best_score,
                        "stage_improvement": (
                            adaptive_state.baseline_score - adaptive_state.best_score
                            if adaptive_state.baseline_score is not None and adaptive_state.best_score is not None
                            else None
                        ),
                        "failure_classes": _failure_class_counts(stage_results),
                        "adaptive_state": adaptive_state.to_dict(),
                    }
                )
            return stage_results

        context_results = run_stage(
            stage=CONTEXT_BASELINE_STAGE,
            stage_label="Context baseline",
            stage_trial_count=1,
            stage_tracking_mode=TRACKING_PASSIVE,
            stage_baseline=context_baseline,
            params=[],
            evaluate_baseline_first=True,
            protected_result=None,
        )
        passive_search_results = run_stage(
            stage=PASSIVE_MODEL_STAGE,
            stage_label="Passive Q/model",
            stage_trial_count=max(0, stage_budgets.passive_model_trials - 1),
            stage_tracking_mode=TRACKING_PASSIVE,
            stage_baseline=context_baseline,
            params=passive_params,
            evaluate_baseline_first=False,
            protected_result=context_results[0] if context_results else None,
        )
        passive_results = context_results + passive_search_results
        best_passive = min(passive_results, key=lambda item: item.closed_loop_score) if passive_results else None

        active_results: list[ClosedLoopValidationResult] = []
        best_active: Optional[ClosedLoopValidationResult] = None
        if request.tracking_mode == TRACKING_ACTIVE and best_passive is not None:
            active_baseline = _active_control_baseline(
                best_passive.candidate_tune,
                context_baseline,
                tuple(record.tune_specs),
            )
            active_results = run_stage(
                stage=ACTIVE_CONTROL_STAGE,
                stage_label="Active-control",
                stage_trial_count=stage_budgets.active_control_trials,
                stage_tracking_mode=TRACKING_ACTIVE,
                stage_baseline=active_baseline,
                params=active_params,
                evaluate_baseline_first=True,
                protected_result=None,
            )
            if active_results:
                best_active = min(active_results, key=lambda item: item.closed_loop_score)

        stage3_baseline_result = best_active if request.tracking_mode == TRACKING_ACTIVE else best_passive
        joint_results: list[ClosedLoopValidationResult] = []
        if stage3_baseline_result is not None and stage_budgets.joint_fine_tune_trials > 0:
            joint_params = _joint_local_search_params(
                passive_params,
                active_params,
                stage3_baseline_result.candidate_tune,
                request.tracking_mode,
                request.actuator_realism_config,
            )
            joint_results = run_stage(
                stage=JOINT_FINE_TUNE_STAGE,
                stage_label="Joint local fine-tune",
                stage_trial_count=stage_budgets.joint_fine_tune_trials,
                stage_tracking_mode=request.tracking_mode,
                stage_baseline=stage3_baseline_result.candidate_tune,
                params=joint_params,
                evaluate_baseline_first=False,
                protected_result=stage3_baseline_result,
            )
        if not validation_results:
            raise ValueError("No closed-loop auto-tune trials completed.")

        final_candidates = (
            active_results + joint_results
            if request.tracking_mode == TRACKING_ACTIVE
            else passive_results + joint_results
        )
        if not final_candidates:
            final_candidates = validation_results
        best_validation = min(final_candidates, key=lambda item: item.closed_loop_score)
        best_tune = dict(best_validation.candidate_tune)
        best_score = float(best_validation.closed_loop_score)
        best_metrics = dict(best_validation.raw_metrics)
        best_metrics["closed_loop_score"] = best_score
        direct_trial_results = _direct_auto_tune_trial_results(validation_results)
        offline_result = AutoTuneResult(
            filter_id=request.filter_id,
            best_tune=best_tune,
            best_score=best_score,
            best_metrics=best_metrics,
            selected_logs=tuple(Path(path) for path in request.offline_log_paths),
            trial_results=tuple(direct_trial_results),
            output_folder=run_folder,
            saved_config_path=None,
            baseline_score=direct_trial_results[0].score if direct_trial_results else None,
            final_score=best_score,
            improved_over_baseline=(
                bool(direct_trial_results)
                and direct_trial_results[0].score is not None
                and best_score < float(direct_trial_results[0].score)
            ),
            recommendation_status=(
                "improved"
                if direct_trial_results and direct_trial_results[0].score is not None and best_score < float(direct_trial_results[0].score)
                else "baseline_kept"
            ),
            verification_results=(),
        )
        validation_dicts = [result.to_dict() for result in validation_results]
        finalists = _direct_finalists_from_validations(validation_results, request.finalist_count)
        finalist_dicts = [finalist.to_dict() for finalist in finalists]
        _write_finalists_csv(run_folder / "finalists.csv", finalists)
        _write_validations_csv(run_folder / "validations.csv", validation_results)

        config = build_closed_loop_schema_v2_config(
            filter_id=request.filter_id,
            filter_display_name=record.display_name,
            tracking_mode=request.tracking_mode,
            sensor_noise_profile=request.sensor_noise_profile,
            noise_sig=selected_noise_signature,
            representative_sensor_noise_config=dict(locked.representative_config),
            vehicle_behavior_profile=request.vehicle_behavior_profile,
            vehicle_behavior_config=dict(request.vehicle_behavior_config),
            actuator_realism_enabled=request.actuator_realism_enabled,
            actuator_realism_profile=request.actuator_realism_profile,
            actuator_realism_config=dict(request.actuator_realism_config),
            validation_route_name=validation_route.name,
            validation_route_map=validation_route.map_name,
            validation_route_id=validation_route.identity(),
            selected_logs=logs,
            candidate_generation_strategy=strategy,
            optuna_available=offline_auto_tuner_module.optuna is not None,
            optuna_study_path=None,
            finalist_count=len(finalists),
            offline_candidate_results=[],
            closed_loop_validation_results=validation_dicts,
            score=best_score,
            best_metrics=best_metrics,
            best_tune=best_tune,
            base_tune=base_tune,
            locked_sensor_noise_values=dict(locked.values),
            output_folder=run_folder,
            extra=_closed_loop_extra_metadata(
                request,
                selected_noise_signature,
                run_folder,
                locked.sources,
                search_profile,
                validation_dicts,
                finalist_dicts,
                stage_budgets=stage_budgets,
                stage_summaries=stage_summaries,
                context_baseline=context_baseline,
                best_passive=best_passive,
                best_active=best_active,
                final_validation=best_validation,
                adaptive_stage_states=adaptive_stage_states,
                baseline_validation=context_results[0] if context_results else None,
            ),
        )
        _ensure_backend_config_compatible(config, request)
        saved_config_path = save_closed_loop_best_tune(request.output_root, request.filter_id, request.tracking_mode, run_folder, config)
        summary = dict(config)
        summary.update(
            {
                "saved_config_path": str(saved_config_path),
            }
        )
        write_json(run_folder / "closed_loop_auto_tune_summary.json", summary)
        _emit(
            progress_callback,
            "completed",
            {
                "best_score": best_score,
                "saved_config_path": str(saved_config_path),
                "stage_budgets": stage_budgets.to_dict(),
                "stage_summaries": stage_summaries,
            },
        )
        return ClosedLoopAutoTuneResult(
            filter_id=request.filter_id,
            tracking_mode=request.tracking_mode,
            best_tune=best_tune,
            best_score=best_score,
            best_metrics=best_metrics,
            finalists=tuple(finalists),
            validation_results=tuple(validation_results),
            output_folder=run_folder,
            saved_config_path=saved_config_path,
            offline_auto_tune_result=offline_result,
        )

    def _normalize_request(
        self,
        request_or_session: ClosedLoopAutoTuneRequest | PendingClosedLoopAutoTuneSession,
    ) -> ClosedLoopAutoTuneRequest:
        if isinstance(request_or_session, ClosedLoopAutoTuneRequest):
            return request_or_session
        if isinstance(request_or_session, PendingClosedLoopAutoTuneSession):
            return ClosedLoopAutoTuneRequest.from_pending_session(request_or_session)
        raise TypeError("ClosedLoopBenchmarkAutoTuner.run expects ClosedLoopAutoTuneRequest or PendingClosedLoopAutoTuneSession.")

    def _fill_defaults_from_record(self, request: ClosedLoopAutoTuneRequest, record: object) -> ClosedLoopAutoTuneRequest:
        base_tune = dict(getattr(record, "tune", {}) or {})
        base_tune.update(dict(request.base_tune or {}))
        auto_tune_profile = dict(request.auto_tune_profile or getattr(record, "auto_tune_profile", {}) or {})
        routes = tuple(ClosedLoopValidationRoute.from_object(route) for route in request.validation_routes)
        return ClosedLoopAutoTuneRequest(
            **{
                **request.to_dict(),
                "offline_log_paths": tuple(Path(path) for path in request.offline_log_paths),
                "validation_routes": routes,
                "base_tune": base_tune,
                "auto_tune_profile": auto_tune_profile,
            }
        )

    @staticmethod
    def _validate_request(request: ClosedLoopAutoTuneRequest) -> None:
        if request.tracking_mode not in {TRACKING_PASSIVE, TRACKING_ACTIVE}:
            raise ValueError(f"Unsupported tracking mode: {request.tracking_mode}")
        if len(request.validation_routes) != 1:
            raise ValueError("Closed-loop auto tune requires exactly one validation route.")
        route = request.validation_routes[0]
        if not route.name or not route.map_name:
            raise ValueError("Validation route must include a route name and map.")
        if not request.auto_tune_profile.get("primary"):
            raise ValueError(f"No auto-tune profile for filter: {request.filter_id}")
        if int(request.finalist_count or 0) < 1:
            raise ValueError("finalist_count must be at least 1.")

    def _run_validation(self, request: ClosedLoopValidationRequest) -> object:
        runner = self._validation_runner
        if hasattr(runner, "run"):
            return runner.run(request)
        if callable(runner):
            return runner(request)
        raise TypeError("Closed-loop validation runner must be callable or expose run(request).")


def closed_loop_objective_score(metrics: dict[str, object]) -> float:
    success = bool(metrics.get("route_completion_success"))
    aborted = bool(metrics.get("route_aborted"))
    timeout = bool(metrics.get("timeout"))
    failure_class = _classify_failure(metrics)
    failure_penalty = 0.0
    if not success:
        failure_penalty += 75000.0
    if aborted:
        failure_penalty += 25000.0
    if timeout:
        failure_penalty += 50000.0
    if metrics.get("abort_reason"):
        failure_penalty += 10000.0
    failure_penalty += {
        "failed_before_route_activation": 35000.0,
        "localization_stability_timeout": 40000.0,
        "no_target_waypoint": 30000.0,
        "vehicle_stuck_no_progress": 25000.0,
        "large_lateral_deviation": 30000.0,
        "control_instability": 35000.0,
        "timeout": 30000.0,
        "route_failure": 20000.0,
    }.get(failure_class, 0.0)
    progress = _first_float(metrics, "route_progress_percent", "route_completion_percent", "progress_percent")
    if not success and progress is not None:
        normalized_progress = progress * 100.0 if 0.0 <= progress <= 1.0 else progress
        failure_penalty += 120.0 * max(0.0, 100.0 - min(100.0, normalized_progress))

    rmse = _first_float(metrics, "eval_filtered_rmse_m", "filtered_rmse_m", "mean_filtered_rmse_m")
    mean_cte = _first_float(metrics, "driving_mean_cross_track_error_m", "mean_cross_track_error_m")
    max_cte = _first_float(metrics, "driving_max_cross_track_error_m", "max_cross_track_error_m")
    speed_rmse = _first_float(metrics, "eval_speed_rmse_mps", "driving_speed_rmse_mps", "speed_rmse_mps")
    yaw_rmse = _first_float(metrics, "eval_yaw_rmse_deg", "driving_yaw_rmse_deg", "yaw_rmse_deg")
    completion_time = _first_float(metrics, "completion_time_s")
    score = failure_penalty
    score += 8.0 * (rmse if rmse is not None else 1000.0)
    score += 4.0 * (mean_cte if mean_cte is not None else 100.0)
    score += 1.5 * (max_cte if max_cte is not None else 100.0)
    score += 0.5 * (speed_rmse if speed_rmse is not None else 0.0)
    score += 0.02 * (yaw_rmse if yaw_rmse is not None else 0.0)
    score += 0.02 * (completion_time if completion_time is not None else 0.0)

    nis = _max_nis_type_mean(
        _first_object(metrics, "driving_nis_by_type_summary", "eval_nis_by_type_summary", "nis_by_type_summary")
    )
    if nis is None:
        nis = _first_float(metrics, "driving_legacy_mean_nis_mixed", "eval_legacy_mean_nis_mixed", "legacy_mean_nis_mixed", "mean_nis")
    nees = _first_float(metrics, "driving_mean_position_nees", "eval_mean_position_nees", "mean_position_nees")
    if nees is None:
        nees = _first_float(
            metrics,
            "driving_mean_position_nees_diagonal_approx",
            "eval_mean_position_nees_diagonal_approx",
            "mean_position_nees_diagonal_approx",
        )
    if nees is None:
        nees = _first_float(metrics, "driving_mean_nees", "eval_mean_nees", "mean_nees")
    consistency = consistency_report_from_summaries(
        nis_by_type_summary=_first_object(
            metrics,
            "driving_nis_by_type_summary",
            "eval_nis_by_type_summary",
            "nis_by_type_summary",
        ),
        mean_position_nees=nees,
        mean_position_nees_diagonal_approx=_first_float(
            metrics,
            "driving_mean_position_nees_diagonal_approx",
            "eval_mean_position_nees_diagonal_approx",
            "mean_position_nees_diagonal_approx",
        ),
        position_nees_source=_first_object(
            metrics,
            "driving_position_nees_source",
            "eval_position_nees_source",
            "position_nees_source",
        ),
    )
    score += 8.0 * float(consistency.get("consistency_error") or 0.0)
    if nis is not None and nis > 18.0:
        score += 2.0 * (nis - 18.0)
    if nees is not None and nees > 20.0:
        score += 2.0 * (nees - 20.0)
    if max_cte is not None and max_cte > 4.0:
        score += 25.0 * (max_cte - 4.0)

    oscillation = _first_float(
        metrics,
        "control_instability_score",
        "control_oscillation_score",
        "control_oscillation_count",
        "mean_abs_steer_rate",
    )
    if oscillation is not None:
        score += 2.0 * max(0.0, oscillation)
    return score if math.isfinite(score) else 1.0e12


def save_closed_loop_best_tune(
    output_root: str,
    filter_id: str,
    tracking_mode: str,
    run_folder: Path,
    config: dict[str, object],
) -> Path:
    config_path = Path(run_folder) / "best_tune.json"
    write_json(config_path, config)
    _update_closed_loop_saved_config_index(output_root, filter_id, tracking_mode, config_path, config)
    return config_path


def closed_loop_stage_budgets(request: ClosedLoopAutoTuneRequest) -> ClosedLoopStageBudgets:
    requested = (
        request.passive_model_trials,
        request.active_control_trials,
        request.joint_fine_tune_trials,
    )
    if any(value is not None for value in requested):
        passive = max(1, int(request.passive_model_trials or 1))
        active = max(0, int(request.active_control_trials or 0))
        joint = max(0, int(request.joint_fine_tune_trials or 0))
        if request.tracking_mode == TRACKING_PASSIVE:
            active = 0
        else:
            active = max(1, active)
        return ClosedLoopStageBudgets(
            passive_model_trials=passive,
            active_control_trials=active,
            joint_fine_tune_trials=joint,
        )
    return closed_loop_default_stage_budgets(request.trial_count, request.tracking_mode)


def closed_loop_default_stage_budgets(total_trials: object, tracking_mode: str) -> ClosedLoopStageBudgets:
    total = max(1, int(total_trials or 1))
    if str(tracking_mode or TRACKING_PASSIVE) != TRACKING_ACTIVE:
        joint = 0 if total < 5 else max(1, int(round(total * 0.20)))
        passive = max(1, total - joint)
        return ClosedLoopStageBudgets(
            passive_model_trials=passive,
            active_control_trials=0,
            joint_fine_tune_trials=max(0, total - passive),
        )
    if total == 1:
        return ClosedLoopStageBudgets(1, 1, 0)
    passive = max(1, int(round(total * 0.50)))
    active = max(1, int(round(total * 0.30)))
    joint = max(0, total - passive - active)
    while passive + active + joint > total and joint > 0:
        joint -= 1
    while passive + active + joint > total and passive > 1:
        passive -= 1
    while passive + active + joint < total:
        joint += 1
    return ClosedLoopStageBudgets(passive, active, joint)


def _validation_result_from_runner(result: object, request: ClosedLoopValidationRequest) -> ClosedLoopValidationResult:
    if isinstance(result, ClosedLoopValidationResult):
        raw = dict(result.raw_metrics)
    elif isinstance(result, dict):
        raw = dict(result)
    else:
        to_dict = getattr(result, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            raw = dict(payload) if isinstance(payload, dict) else {}
        else:
            raw = {
                key: getattr(result, key)
                for key in dir(result)
                if not key.startswith("_") and not callable(getattr(result, key))
            }
    output_folder = _path_or_none(raw.get("output_folder") or raw.get("route_folder") or request.output_folder)
    raw["failure_class"] = _classify_failure(raw)
    score = closed_loop_objective_score(raw)
    return ClosedLoopValidationResult(
        finalist_rank=request.finalist.rank,
        candidate_tune=dict(request.finalist.candidate_tune),
        route_completion_success=bool(raw.get("route_completion_success")),
        route_aborted=bool(raw.get("route_aborted")),
        timeout=bool(raw.get("timeout")),
        abort_reason=str(raw.get("abort_reason") or ""),
        completion_time_s=_first_float(raw, "completion_time_s"),
        eval_filtered_rmse_m=_first_float(raw, "eval_filtered_rmse_m"),
        filtered_rmse_m=_first_float(raw, "filtered_rmse_m"),
        mean_cross_track_error_m=_first_float(raw, "driving_mean_cross_track_error_m", "mean_cross_track_error_m"),
        max_cross_track_error_m=_first_float(raw, "driving_max_cross_track_error_m", "max_cross_track_error_m"),
        nis_by_type_summary=_dict_value(_first_object(raw, "driving_nis_by_type_summary", "eval_nis_by_type_summary", "nis_by_type_summary")),
        position_nees_source=str(_first_object(raw, "driving_position_nees_source", "eval_position_nees_source", "position_nees_source") or "unavailable"),
        mean_position_nees=_first_float(raw, "driving_mean_position_nees", "eval_mean_position_nees", "mean_position_nees"),
        mean_position_nees_diagonal_approx=_first_float(
            raw,
            "driving_mean_position_nees_diagonal_approx",
            "eval_mean_position_nees_diagonal_approx",
            "mean_position_nees_diagonal_approx",
        ),
        raw_metrics=raw,
        output_folder=output_folder,
        closed_loop_score=score,
        stage=request.stage,
        stage_trial_index=request.stage_trial_index,
        stage_trial_total=request.stage_trial_total,
        failure_class=str(raw.get("failure_class") or ""),
        changed_parameters=dict(request.changed_parameters),
        changed_parameters_summary=request.changed_parameters_summary,
        affected_families=tuple(request.affected_families),
        adaptation_decision=request.adaptation_decision,
    )


def _select_finalists(offline_result: AutoTuneResult, finalist_count: int) -> list[ClosedLoopFinalist]:
    successful_all = [
        trial
        for trial in offline_result.trial_results
        if not trial.failed and trial.score is not None and math.isfinite(float(trial.score))
    ]
    non_baseline = [
        trial
        for trial in successful_all
        if getattr(trial, "candidate_type", "") not in {"default_base", "current_ui"}
    ]
    verified = [trial for trial in non_baseline if getattr(trial, "stage", "") == "verification"]
    search = [trial for trial in non_baseline if getattr(trial, "stage", "search") == "search"]
    successful = verified or search or non_baseline or successful_all
    ordered = sorted(successful, key=lambda trial: float(trial.score))
    finalists: list[ClosedLoopFinalist] = []
    for rank, trial in enumerate(ordered[: max(1, int(finalist_count))], start=1):
        finalists.append(
            ClosedLoopFinalist(
                rank=rank,
                candidate_tune=dict(trial.candidate_tune),
                offline_score=float(trial.score),
                offline_metrics=dict(trial.metrics),
                trial_index=trial.trial_index,
                source_output_folder=trial.output_folder,
            )
        )
    return finalists


def _context_aware_baseline_tune(
    base_tune: dict[str, object],
    request: ClosedLoopAutoTuneRequest,
    tune_specs: tuple[object, ...],
) -> dict[str, object]:
    baseline = dict(base_tune)
    behavior_scale = _behavior_process_scale(request.vehicle_behavior_config)
    actuator_scale = 1.0 + 0.18 * _actuator_imperfection_score(request.actuator_realism_config)
    dynamic_scale = max(0.75, min(1.35, behavior_scale * actuator_scale))
    for key in list(baseline.keys()):
        if _is_process_model_key(key):
            value = _optional_float(baseline.get(key))
            if value is not None:
                baseline[key] = value * dynamic_scale
        elif key.endswith("_R_multiplier"):
            baseline[key] = 1.0
    return _clamp_tune_to_specs(baseline, tune_specs)


def _behavior_process_scale(config: dict[str, object]) -> float:
    speed = _optional_float(config.get("max_speed_mps")) or 8.0
    accel = _optional_float(config.get("max_forward_accel_mps2")) or 1.6
    aggressiveness = _optional_float(config.get("speed_change_aggressiveness")) or 1.0
    profile = str(config.get("preset_name") or "").lower()
    scale = 0.85 + 0.08 * (speed / 8.0) + 0.05 * (accel / 1.6) + 0.08 * aggressiveness
    if "aggressive" in profile:
        scale += 0.08
    elif "conservative" in profile:
        scale -= 0.06
    return max(0.75, min(1.35, scale))


def _is_process_model_key(key: str) -> bool:
    text = str(key)
    return (
        text.startswith("process_")
        or text in {"covariance_inflation", "yaw_from_velocity_min_speed_mps"}
        or text.startswith("initial_")
    )


def _clamp_tune_to_specs(tune: dict[str, object], tune_specs: tuple[object, ...]) -> dict[str, object]:
    result = dict(tune)
    specs = {str(getattr(spec, "key", "")): spec for spec in tune_specs if getattr(spec, "key", "")}
    for key, value in list(result.items()):
        spec = specs.get(key)
        if spec is not None and hasattr(spec, "clamp"):
            result[key] = float(spec.clamp(value))
    return result


def _adaptive_stage_state(
    *,
    stage: str,
    stage_baseline: dict[str, object],
    params: list[dict[str, object]],
    protected_result: Optional[ClosedLoopValidationResult],
) -> AdaptiveStageState:
    family_by_key: dict[str, str] = {}
    initial_bounds: dict[str, tuple[float, float]] = {}
    for param in params:
        key = str(param.get("key") or "")
        if not key:
            continue
        family_by_key[key] = _parameter_family(key)
        initial_bounds[key] = _param_bounds(param, stage_baseline.get(key))
    families = sorted(set(family_by_key.values()))
    initial_scale = {
        PASSIVE_MODEL_STAGE: 0.55,
        ACTIVE_CONTROL_STAGE: 0.50,
        JOINT_FINE_TUNE_STAGE: 0.32,
    }.get(stage, 0.0)
    protected_tune = dict(protected_result.candidate_tune) if protected_result is not None else dict(stage_baseline)
    protected_score = float(protected_result.closed_loop_score) if protected_result is not None else None
    return AdaptiveStageState(
        stage=stage,
        baseline_tune=dict(stage_baseline),
        params=[dict(param) for param in params],
        best_tune=protected_tune,
        best_score=protected_score,
        baseline_score=protected_score,
        family_by_key=family_by_key,
        initial_bounds=initial_bounds,
        family_scales={family: initial_scale for family in families},
        family_direction_bias={family: 0.0 for family in families},
        family_stats={family: ParameterFamilyStats() for family in families},
        parameter_attempts={key: 0 for key in family_by_key},
        adaptation_history=[],
    )


def _adaptive_candidate_plan(
    state: AdaptiveStageState,
    rng: random.Random,
    *,
    baseline_trial: bool,
    failure_counts: dict[str, int],
) -> AdaptiveCandidatePlan:
    if baseline_trial or not state.params:
        return AdaptiveCandidatePlan(
            tune=dict(state.baseline_tune),
            anchor_tune=dict(state.baseline_tune),
            affected_families=(),
            candidate_rationale="protected context baseline",
        )
    family_params: dict[str, list[dict[str, object]]] = {}
    for param in state.params:
        key = str(param.get("key") or "")
        family = state.family_by_key.get(key)
        if key and family:
            family_params.setdefault(family, []).append(param)
    if not family_params:
        return AdaptiveCandidatePlan(
            tune=dict(state.best_tune),
            anchor_tune=dict(state.best_tune),
            affected_families=(),
            candidate_rationale="no supported adaptive parameters",
        )

    family = _select_adaptive_family(state, family_params)
    candidates = sorted(
        family_params[family],
        key=lambda param: (
            state.parameter_attempts.get(str(param.get("key") or ""), 0),
            str(param.get("key") or ""),
        ),
    )
    chosen = candidates[:1]
    stats = state.family_stats[family]
    if stats.improvements >= 2 and stats.failures == 0 and len(candidates) > 1:
        chosen = candidates[:2]
    anchor = dict(state.best_tune)
    tune = dict(anchor)
    direction_bias = state.family_direction_bias.get(family, 0.0)
    for param in chosen:
        key = str(param.get("key") or "")
        direction = _adaptive_direction(direction_bias, state.parameter_attempts.get(key, 0), rng)
        tune[key] = _adaptive_sample_value(state, param, anchor.get(key), family, direction, rng)
        state.parameter_attempts[key] = state.parameter_attempts.get(key, 0) + 1
    scale = state.family_scales.get(family, 0.0)
    direction_text = "upward" if direction_bias > 0.20 else "downward" if direction_bias < -0.20 else "two-sided"
    protection = " protected-best local" if state.non_improving_trials >= 3 else ""
    failure_note = ""
    if failure_counts.get("control_instability", 0) and family.startswith("active_"):
        failure_note = " after control failures"
    return AdaptiveCandidatePlan(
        tune=tune,
        anchor_tune=anchor,
        affected_families=(family,),
        candidate_rationale=(
            f"probe {family} near current best; {direction_text} bias; "
            f"bound scale {scale:.2f}{protection}{failure_note}"
        ),
    )


def _select_adaptive_family(
    state: AdaptiveStageState,
    family_params: dict[str, list[dict[str, object]]],
) -> str:
    families = sorted(family_params)
    if state.non_improving_trials >= 3:
        improving = [
            family
            for family in families
            if state.family_stats[family].improvements > 0
        ]
        if improving:
            return min(
                improving,
                key=lambda family: (
                    -state.family_stats[family].improvements,
                    state.family_stats[family].failures,
                    state.family_stats[family].trials,
                    family,
                ),
            )
    return min(
        families,
        key=lambda family: (
            state.family_stats[family].trials,
            state.family_stats[family].failures + state.family_stats[family].degradations
            - state.family_stats[family].improvements,
            -state.family_scales.get(family, 0.0),
            family,
        ),
    )


def _adaptive_direction(bias: float, attempt_count: int, rng: random.Random) -> int:
    if bias > 0.20:
        return 1 if rng.random() < 0.80 else -1
    if bias < -0.20:
        return -1 if rng.random() < 0.80 else 1
    return 1 if attempt_count % 2 == 0 else -1


def _adaptive_sample_value(
    state: AdaptiveStageState,
    param: dict[str, object],
    anchor_value: object,
    family: str,
    direction: int,
    rng: random.Random,
) -> float:
    key = str(param.get("key") or "")
    anchor = _optional_float(anchor_value)
    if anchor is None:
        low, high = state.initial_bounds[key]
        anchor = low + 0.5 * (high - low)
    low, high = _adaptive_effective_bounds(state, key, anchor, family)
    target = high if direction > 0 else low
    if abs(target - anchor) <= 1.0e-12:
        target = low if direction > 0 else high
    fraction = rng.uniform(0.25, 0.62)
    if str(param.get("scale") or "").lower() == "log" and anchor > 0.0 and target > 0.0:
        value = math.exp(math.log(anchor) + fraction * (math.log(target) - math.log(anchor)))
    else:
        value = anchor + fraction * (target - anchor)
    if abs(value - anchor) <= max(1.0e-9, abs(anchor) * 1.0e-6):
        value = target
    return max(low, min(high, value))


def _adaptive_effective_bounds(
    state: AdaptiveStageState,
    key: str,
    anchor: float,
    family: str,
) -> tuple[float, float]:
    initial_low, initial_high = state.initial_bounds[key]
    scale = max(0.06, min(1.0, state.family_scales.get(family, 0.5)))
    if anchor > 0.0 and initial_low > 0.0 and initial_high > 0.0:
        low = math.exp(math.log(anchor) + scale * (math.log(initial_low) - math.log(anchor)))
        high = math.exp(math.log(anchor) + scale * (math.log(initial_high) - math.log(anchor)))
    else:
        low = anchor + scale * (initial_low - anchor)
        high = anchor + scale * (initial_high - anchor)
    low = max(initial_low, min(initial_high, low))
    high = max(initial_low, min(initial_high, high))
    if high <= low:
        return _expand_degenerate_bounds(anchor, initial_low, initial_high)
    return low, high


def _trial_outcome_class(
    validation: ClosedLoopValidationResult,
    reference_score: Optional[float],
) -> str:
    if not validation.route_completion_success:
        return "failed"
    if reference_score is None:
        return "baseline"
    tolerance = max(0.02, abs(reference_score) * 0.01)
    if validation.closed_loop_score < reference_score - tolerance:
        return "improved"
    if validation.closed_loop_score > reference_score + max(0.10, abs(reference_score) * 0.05):
        return "degraded"
    return "stable"


def _update_adaptive_stage_state(
    state: AdaptiveStageState,
    validation: ClosedLoopValidationResult,
    changed_parameters: dict[str, object],
    outcome_class: str,
    affected_families: tuple[str, ...],
) -> list[str]:
    state.completed_trials += 1
    reference_score = state.best_score
    if state.baseline_score is None:
        state.baseline_score = validation.closed_loop_score
    diagnostics = _trial_diagnostic_flags(validation)
    decisions: list[str] = []
    if not affected_families:
        if state.best_score is None or validation.closed_loop_score < state.best_score:
            state.best_score = validation.closed_loop_score
            state.best_tune = dict(validation.candidate_tune)
        state.adaptation_history.append(
            {
                "trial_index": validation.finalist_rank,
                "outcome_class": outcome_class,
                "affected_families": [],
                "decisions": ["protected baseline evaluated"],
                "diagnostics": diagnostics,
            }
        )
        return ["protected baseline evaluated"]

    for family in affected_families:
        stats = state.family_stats.setdefault(family, ParameterFamilyStats())
        stats.trials += 1
        direction = _family_change_direction(changed_parameters, family)
        if validation.route_completion_success:
            stats.successes += 1
        else:
            stats.failures += 1
            if validation.failure_class:
                stats.failure_classes[validation.failure_class] = stats.failure_classes.get(validation.failure_class, 0) + 1
        if reference_score is not None:
            stats.score_delta_sum += validation.closed_loop_score - reference_score
        if diagnostics["high_nis"]:
            stats.high_nis_count += 1
        if diagnostics["high_nees"]:
            stats.high_nees_count += 1
        if diagnostics["large_cte"]:
            stats.large_cte_count += 1

        if outcome_class == "improved":
            stats.improvements += 1
            state.family_direction_bias[family] = _clamp_bias(
                state.family_direction_bias.get(family, 0.0) + 0.35 * direction
            )
            if direction > 0:
                stats.upward_improvements += 1
            elif direction < 0:
                stats.downward_improvements += 1
            decisions.append(f"biased {family} {'upward' if direction > 0 else 'downward' if direction < 0 else 'locally'}")
            if stats.improvements >= 2 and stats.failures == 0:
                old_scale = state.family_scales.get(family, 0.5)
                state.family_scales[family] = min(1.0, old_scale * 1.12)
                if state.family_scales[family] > old_scale:
                    decisions.append(f"expanded {family} bounds after repeated stable improvements")
        elif outcome_class == "stable":
            stats.stable_trials += 1
        else:
            stats.degradations += 1
            if direction > 0:
                stats.upward_degradations += 1
            elif direction < 0:
                stats.downward_degradations += 1
            shrink = 0.52 if outcome_class == "failed" else 0.72
            if validation.failure_class == "control_instability" and family.startswith("active_"):
                shrink = 0.42
            state.family_scales[family] = max(0.06, state.family_scales.get(family, 0.5) * shrink)
            state.family_direction_bias[family] = _clamp_bias(
                state.family_direction_bias.get(family, 0.0) - 0.30 * direction
            )
            decisions.append(f"shrunk {family} bounds after {outcome_class}")

    if outcome_class == "improved":
        state.best_score = validation.closed_loop_score
        state.best_tune = dict(validation.candidate_tune)
        state.non_improving_trials = 0
    else:
        state.non_improving_trials += 1

    decisions.extend(_diagnostic_adaptation_decisions(state, validation, diagnostics, changed_parameters))
    if state.non_improving_trials >= 3 and state.non_improving_trials % 3 == 0:
        for family in state.family_scales:
            state.family_scales[family] = max(0.06, state.family_scales[family] * 0.70)
        decisions.append("protected current best and tightened all stage bounds")

    decisions = _dedupe_text(decisions)
    state.adaptation_history.append(
        {
            "trial_index": validation.finalist_rank,
            "score": validation.closed_loop_score,
            "reference_best_score": reference_score,
            "outcome_class": outcome_class,
            "failure_class": validation.failure_class,
            "affected_families": list(affected_families),
            "decisions": decisions,
            "diagnostics": diagnostics,
            "family_scales_after": dict(state.family_scales),
            "family_direction_bias_after": dict(state.family_direction_bias),
        }
    )
    return decisions


def _diagnostic_adaptation_decisions(
    state: AdaptiveStageState,
    validation: ClosedLoopValidationResult,
    diagnostics: dict[str, object],
    changed_parameters: dict[str, object],
) -> list[str]:
    decisions: list[str] = []
    if diagnostics.get("high_nees"):
        targets = _existing_families(
            state,
            {"Q_position", "Q_acceleration_jerk", "Q_yaw", "Q_process_general", "covariance_initialization"},
        )
        for family in targets:
            state.family_direction_bias[family] = max(
                0.35,
                _clamp_bias(state.family_direction_bias.get(family, 0.0) + 0.25),
            )
        if targets:
            decisions.append(f"high NEES: biased {', '.join(targets)} upward")
    if diagnostics.get("high_nis"):
        targets = _existing_families(state, {"R_GNSS_effective", "R_IMU_effective"})
        for family in targets:
            state.family_direction_bias[family] = max(
                0.30,
                _clamp_bias(state.family_direction_bias.get(family, 0.0) + 0.20),
            )
        if targets:
            decisions.append(f"high NIS: avoided lower measurement uncertainty and biased {', '.join(targets)} upward")
    failure_class = validation.failure_class
    if failure_class in {"failed_before_route_activation", "localization_stability_timeout"}:
        targets = _existing_families(
            state,
            {"covariance_initialization", "Q_position", "Q_acceleration_jerk", "Q_yaw", "Q_process_general"},
        )
        _shrink_families(state, targets, 0.68)
        if targets:
            decisions.append(f"{failure_class}: shrunk initialization/process families")
    elif failure_class == "vehicle_stuck_no_progress":
        targets = _existing_families(state, {"Q_position", "Q_acceleration_jerk", "Q_process_general"})
        downward = any(_family_change_direction(changed_parameters, family) < 0 for family in targets)
        if downward:
            for family in targets:
                state.family_direction_bias[family] = _clamp_bias(state.family_direction_bias.get(family, 0.0) + 0.22)
            decisions.append("stuck outcome after reduced model flexibility: biased process uncertainty upward")
    elif failure_class == "large_lateral_deviation":
        targets = _existing_families(state, {"Q_yaw", "active_yaw_steer_control"})
        _shrink_families(state, targets, 0.55)
        if targets:
            decisions.append(f"large CTE: shrunk {', '.join(targets)} bounds")
    elif failure_class == "control_instability":
        targets = _existing_families(
            state,
            {
                "active_acceleration_control",
                "active_yaw_steer_control",
                "active_timeout_input_memory",
                "active_delta_limits",
            },
        )
        _shrink_families(state, targets, 0.48)
        if targets:
            decisions.append("control instability: shrunk all active-control families")
    if diagnostics.get("divergence"):
        targets = _existing_families(state, set(state.family_scales))
        _shrink_families(state, targets, 0.65)
        decisions.append("divergence indicator: tightened current stage around the best tune")
    return decisions


def _trial_diagnostic_flags(validation: ClosedLoopValidationResult) -> dict[str, object]:
    consistency = consistency_report_from_summaries(
        nis_by_type_summary=validation.nis_by_type_summary,
        mean_position_nees=validation.mean_position_nees,
        mean_position_nees_diagonal_approx=validation.mean_position_nees_diagonal_approx,
        position_nees_source=validation.position_nees_source,
    )
    nis_report = consistency.get("nis_by_type") if isinstance(consistency.get("nis_by_type"), dict) else {}
    high_nis = any(
        isinstance(item, dict) and str(item.get("status") or "") in {"warning", "severe"}
        and str(item.get("behavior") or "") == "overconfident"
        for item in nis_report.values()
    )
    nees_report = consistency.get("position_nees") if isinstance(consistency.get("position_nees"), dict) else {}
    high_nees = (
        str(nees_report.get("status") or "") in {"warning", "severe"}
        and str(nees_report.get("behavior") or "") == "overconfident"
    )
    raw = validation.raw_metrics
    divergence_count = _first_float(
        raw,
        "divergence_event_count",
        "filter_divergence_event_count",
        "estimator_divergence_count",
    ) or 0.0
    covariance_growth = _first_float(
        raw,
        "covariance_growth_ratio",
        "max_covariance_growth_ratio",
        "position_covariance_growth_ratio",
    ) or 0.0
    oscillation = _first_float(
        raw,
        "control_instability_score",
        "control_oscillation_score",
        "control_oscillation_count",
    ) or 0.0
    return {
        "high_nis": bool(high_nis),
        "high_nees": bool(high_nees),
        "large_cte": bool(
            validation.failure_class == "large_lateral_deviation"
            or (
                validation.max_cross_track_error_m is not None
                and validation.max_cross_track_error_m > 4.0
            )
        ),
        "divergence": bool(divergence_count > 0.0 or covariance_growth > 8.0),
        "control_instability": bool(
            validation.failure_class == "control_instability"
            or oscillation > 0.0
        ),
        "consistency_status": consistency.get("overall_status"),
        "consistency_error": consistency.get("consistency_error"),
    }


def _parameter_family(key: str) -> str:
    text = str(key or "").lower()
    if text == "enable_control_input_prediction":
        return "active_enable"
    if text in ACTIVE_CONTROL_TUNE_KEYS or any(
        token in text
        for token in ("control_", "command_", "input_timeout", "input_memory")
    ):
        if "timeout" in text or "memory" in text:
            return "active_timeout_input_memory"
        if any(token in text for token in ("delta", "limit", "max_")):
            return "active_delta_limits"
        if any(token in text for token in ("yaw", "steer", "turn")):
            return "active_yaw_steer_control"
        if any(token in text for token in ("accel", "brake", "throttle")):
            return "active_acceleration_control"
        return "active_control_other"
    if any(token in text for token in ("sigma", "alpha", "beta", "kappa", "lambda")) and "imu" not in text:
        return "UKF_sigma_point"
    if text.startswith("initial_") or text.startswith("startup_") or "covariance_inflation" in text:
        return "covariance_initialization"
    if text.endswith("_r_multiplier") or "_r_" in text:
        if "gnss" in text or "position" in text:
            return "R_GNSS_effective"
        if any(token in text for token in ("imu", "yaw", "gyro", "accel")):
            return "R_IMU_effective"
        return "R_measurement_effective"
    if text.startswith("process_") or text == "process_noise_multiplier":
        if any(token in text for token in ("yaw", "turn", "heading", "gyro")):
            return "Q_yaw"
        if any(token in text for token in ("accel", "jerk", "velocity", "speed")):
            return "Q_acceleration_jerk"
        if "position" in text:
            return "Q_position"
        return "Q_process_general"
    if any(token in text for token in ("prediction_dt", "turn_rate_epsilon", "yaw_from_velocity")):
        return "model_dynamics"
    return "estimator_specific"


def _family_change_direction(changes: dict[str, object], family: str) -> int:
    directions: list[int] = []
    for raw in changes.values():
        if not isinstance(raw, dict) or str(raw.get("parameter_family") or "") != family:
            continue
        delta = _optional_float(raw.get("delta"))
        if delta is not None:
            directions.append(1 if delta > 0.0 else -1 if delta < 0.0 else 0)
    total = sum(directions)
    return 1 if total > 0 else -1 if total < 0 else 0


def _existing_families(state: AdaptiveStageState, requested: set[str]) -> list[str]:
    return sorted(family for family in requested if family in state.family_scales)


def _shrink_families(state: AdaptiveStageState, families: list[str], factor: float) -> None:
    for family in families:
        state.family_scales[family] = max(0.06, state.family_scales.get(family, 0.5) * factor)


def _clamp_bias(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _dedupe_text(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _parameter_attribution_records(
    changes: dict[str, object],
    validation: ClosedLoopValidationResult,
    outcome_class: str,
    adaptation_decision: str,
) -> tuple[dict[str, object], ...]:
    nis = _max_nis_type_mean(validation.nis_by_type_summary)
    nees = validation.mean_position_nees
    if nees is None:
        nees = validation.mean_position_nees_diagonal_approx
    records: list[dict[str, object]] = []
    for key, raw in changes.items():
        change = raw if isinstance(raw, dict) else {}
        records.append(
            {
                "parameter": key,
                "parameter_family": change.get("parameter_family") or _parameter_family(key),
                "baseline_value": change.get("baseline"),
                "trial_value": change.get("candidate"),
                "absolute_change": change.get("absolute_change"),
                "relative_change_percent": change.get("relative_change_percent"),
                "direction": change.get("direction"),
                "stage": validation.stage,
                "score": validation.closed_loop_score,
                "rmse_m": validation.eval_filtered_rmse_m or validation.filtered_rmse_m,
                "nis": nis,
                "nees": nees,
                "mean_cte_m": validation.mean_cross_track_error_m,
                "max_cte_m": validation.max_cross_track_error_m,
                "failure_class": validation.failure_class,
                "route_completion_success": validation.route_completion_success,
                "outcome_class": outcome_class,
                "adaptation_decision": adaptation_decision,
            }
        )
    return tuple(records)


def _passive_model_search_params(
    params: list[dict[str, object]],
    baseline: dict[str, object],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for param in params:
        key = str(param.get("key") or "")
        if not key or key in ACTIVE_CONTROL_TUNE_KEYS:
            continue
        result.append(_local_param_bounds(param, baseline, stage="passive_model"))
    return result


def _active_control_search_params(
    baseline: dict[str, object],
    tune_specs: tuple[object, ...],
    actuator_realism_config: dict[str, object],
) -> list[dict[str, object]]:
    severity = _actuator_imperfection_score(actuator_realism_config)
    radius = 0.45 if severity <= 0.08 else 0.28 if severity <= 0.35 else 0.16
    result: list[dict[str, object]] = []
    for spec in tune_specs:
        key = str(getattr(spec, "key", "") or "")
        group = str(getattr(spec, "group", "") or "")
        if not key or key == "enable_control_input_prediction":
            continue
        if key not in ACTIVE_CONTROL_TUNE_KEYS and group.lower() != "active tracking":
            continue
        base_value = _optional_float(baseline.get(key))
        if base_value is None:
            continue
        spec_low = _optional_float(getattr(spec, "minimum", None))
        spec_high = _optional_float(getattr(spec, "maximum", None))
        if spec_low is None or spec_high is None or spec_high <= spec_low:
            continue
        span = spec_high - spec_low
        low = max(spec_low, base_value - span * radius)
        high = min(spec_high, base_value + span * radius)
        if key in {"control_steer_to_yaw_rate_gain", "max_control_yaw_rate_delta_radps", "command_yaw_rate_stddev_dps", "command_max_yaw_rate_dps"}:
            high = min(high, base_value + span * radius * 0.75)
        if high <= low:
            low, high = _expand_degenerate_bounds(base_value, spec_low, spec_high)
        result.append(
            _actuator_adjusted_param(
                {
                    "key": key,
                    "scale": "log" if base_value > 0.0 and ("stddev" in key or key.endswith("_gain_mps2")) else "linear",
                    "min": low,
                    "max": high,
                },
                actuator_realism_config,
                TRACKING_ACTIVE,
            )
        )
    return result


def _joint_local_search_params(
    passive_params: list[dict[str, object]],
    active_params: list[dict[str, object]],
    baseline: dict[str, object],
    tracking_mode: str,
    actuator_realism_config: dict[str, object],
) -> list[dict[str, object]]:
    candidates = passive_params + (active_params if tracking_mode == TRACKING_ACTIVE else [])
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for param in candidates:
        key = str(param.get("key") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        local = _local_param_bounds(param, baseline, stage="joint_local")
        if tracking_mode == TRACKING_ACTIVE and key in ACTIVE_CONTROL_TUNE_KEYS:
            local = _actuator_adjusted_param(local, actuator_realism_config, TRACKING_ACTIVE)
        result.append(local)
    return result


def _local_param_bounds(param: dict[str, object], baseline: dict[str, object], *, stage: str) -> dict[str, object]:
    result = dict(param)
    key = str(result.get("key") or "")
    base = _optional_float(baseline.get(key))
    if base is None:
        return result
    low, high = _param_bounds(result, base)
    if key.endswith("_R_multiplier"):
        factor = 0.15 if stage == "joint_local" else 0.45
        result["min"] = max(low, 1.0 - factor)
        result["max"] = min(high, 1.0 + factor)
        return result
    if str(result.get("scale") or "").lower() == "log" and base > 0.0:
        factor = 1.25 if stage == "joint_local" else 2.4
        result["min"] = max(low, base / factor)
        result["max"] = min(high, base * factor)
    else:
        span = high - low
        fraction = 0.12 if stage == "joint_local" else 0.32
        result["min"] = max(low, base - span * fraction)
        result["max"] = min(high, base + span * fraction)
    if _optional_float(result.get("max")) is not None and _optional_float(result.get("min")) is not None:
        if float(result["max"]) <= float(result["min"]):
            result["min"], result["max"] = _expand_degenerate_bounds(base, low, high)
    return result


def _expand_degenerate_bounds(base: float, low: float, high: float) -> tuple[float, float]:
    epsilon = max(1.0e-6, abs(base) * 0.05, (high - low) * 0.05)
    return max(low, base - epsilon), min(high, base + epsilon)


def _active_control_baseline(
    passive_best_tune: dict[str, object],
    context_baseline: dict[str, object],
    tune_specs: tuple[object, ...],
) -> dict[str, object]:
    tune = dict(passive_best_tune)
    for key in ACTIVE_CONTROL_TUNE_KEYS:
        if key in context_baseline:
            tune[key] = context_baseline[key]
    if "enable_control_input_prediction" in context_baseline:
        tune["enable_control_input_prediction"] = 1.0
    return _clamp_tune_to_specs(tune, tune_specs)


def _failure_aware_candidate(
    candidate: dict[str, object],
    baseline: dict[str, object],
    params: list[dict[str, object]],
    stage: str,
    failure_counts: dict[str, int],
) -> dict[str, object]:
    result = dict(candidate)
    activation_failures = (
        failure_counts.get("failed_before_route_activation", 0)
        + failure_counts.get("localization_stability_timeout", 0)
    )
    active_failures = (
        failure_counts.get("control_instability", 0)
        + failure_counts.get("vehicle_stuck_no_progress", 0)
    )
    lateral_failures = failure_counts.get("large_lateral_deviation", 0)
    for param in params:
        key = str(param.get("key") or "")
        if not key or key not in result:
            continue
        if activation_failures >= 2 and (_is_initialization_key(key) or key in {"min_prediction_dt_s", "max_prediction_dt_s"}):
            result[key] = baseline.get(key, result[key])
            continue
        if lateral_failures >= 1 and _is_yaw_or_steer_key(key):
            result[key] = _blend_value(result.get(key), baseline.get(key), 0.35)
            continue
        if stage == ACTIVE_CONTROL_STAGE and active_failures >= 1 and key in ACTIVE_CONTROL_TUNE_KEYS:
            result[key] = _blend_value(result.get(key), baseline.get(key), 0.50 if active_failures == 1 else 0.25)
            continue
        if activation_failures >= 3:
            result[key] = _blend_value(result.get(key), baseline.get(key), 0.50)
    return result


def _blend_value(value: object, baseline: object, candidate_weight: float) -> object:
    current = _optional_float(value)
    base = _optional_float(baseline)
    if current is None or base is None:
        return baseline if baseline is not None else value
    weight = max(0.0, min(1.0, candidate_weight))
    return base + weight * (current - base)


def _is_initialization_key(key: str) -> bool:
    return str(key).startswith("initial_") or str(key).startswith("startup_")


def _is_yaw_or_steer_key(key: str) -> bool:
    text = str(key).lower()
    return any(token in text for token in ("yaw", "steer", "turn", "gyro"))


def _ensure_unique_candidate(
    candidate: dict[str, object],
    baseline: dict[str, object],
    params: list[dict[str, object]],
    tracking_mode: str,
    seen: set[str],
    rng: random.Random,
    *,
    allow_existing: bool = False,
) -> dict[str, object]:
    signature = _candidate_signature(candidate, tracking_mode)
    if allow_existing or signature not in seen:
        seen.add(signature)
        return candidate
    result = dict(candidate)
    for _attempt in range(8):
        for param in params:
            key = str(param.get("key") or "")
            if key:
                result[key] = _sample_param(rng, param, baseline.get(key))
        signature = _candidate_signature(result, tracking_mode)
        if signature not in seen:
            seen.add(signature)
            return result
    if params:
        key = str(params[0].get("key") or "")
        if key:
            low, high = _param_bounds(params[0], baseline.get(key))
            current = _optional_float(result.get(key))
            if current is not None:
                result[key] = max(low, min(high, current + 0.03 * (high - low)))
    seen.add(_candidate_signature(result, tracking_mode))
    return result


def _candidate_signature(candidate: dict[str, object], tracking_mode: str) -> str:
    rounded: dict[str, object] = {}
    for key, value in sorted(candidate.items()):
        number = _optional_float(value)
        rounded[key] = round(number, 9) if number is not None else value
    return f"{tracking_mode}:{json.dumps(rounded, sort_keys=True, separators=(',', ':'))}"


def _changed_parameters(
    baseline: dict[str, object],
    candidate: dict[str, object],
    family_by_key: Optional[dict[str, str]] = None,
) -> dict[str, object]:
    changes: dict[str, object] = {}
    for key in sorted(set(baseline) | set(candidate)):
        base = baseline.get(key)
        value = candidate.get(key)
        if not _values_differ(base, value):
            continue
        base_num = _optional_float(base)
        value_num = _optional_float(value)
        change: dict[str, object] = {
            "baseline": base,
            "candidate": value,
            "parameter_family": (family_by_key or {}).get(key) or _parameter_family(key),
        }
        if base_num is not None and value_num is not None:
            signed_change = value_num - base_num
            relative_change = None if abs(base_num) <= 1.0e-12 else 100.0 * signed_change / base_num
            change["delta"] = signed_change
            change["delta_percent"] = relative_change
            change["absolute_change"] = abs(signed_change)
            change["relative_change_percent"] = relative_change
            change["direction"] = "up" if signed_change > 0.0 else "down" if signed_change < 0.0 else "unchanged"
        changes[key] = change
    return changes


def _changed_parameters_summary(changes: dict[str, object], max_items: int = 4) -> str:
    items: list[str] = []
    for key, raw in list(changes.items())[:max_items]:
        change = raw if isinstance(raw, dict) else {}
        value = change.get("candidate")
        base = change.get("baseline")
        value_num = _optional_float(value)
        base_num = _optional_float(base)
        if value_num is not None and base_num is not None:
            items.append(f"{key} {base_num:.3g}->{value_num:.3g}")
        else:
            items.append(f"{key} changed")
    remaining = max(0, len(changes) - len(items))
    if remaining:
        items.append(f"+{remaining} more")
    return ", ".join(items) if items else "baseline"


def _values_differ(left: object, right: object) -> bool:
    left_num = _optional_float(left)
    right_num = _optional_float(right)
    if left_num is not None and right_num is not None:
        return abs(left_num - right_num) > max(1.0e-9, 1.0e-6 * max(abs(left_num), abs(right_num), 1.0))
    return left != right


def _failure_class_counts(results: list[ClosedLoopValidationResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        failure_class = result.failure_class
        if failure_class:
            counts[failure_class] = counts.get(failure_class, 0) + 1
    return counts


def _aggregate_parameter_family_diagnostics(
    states: list[AdaptiveStageState],
) -> dict[str, object]:
    aggregate: dict[str, ParameterFamilyStats] = {}
    final_scales: dict[str, float] = {}
    final_biases: dict[str, float] = {}
    for state in states:
        for family, stats in state.family_stats.items():
            target = aggregate.setdefault(family, ParameterFamilyStats())
            target.trials += stats.trials
            target.successes += stats.successes
            target.failures += stats.failures
            target.improvements += stats.improvements
            target.degradations += stats.degradations
            target.stable_trials += stats.stable_trials
            target.high_nis_count += stats.high_nis_count
            target.high_nees_count += stats.high_nees_count
            target.large_cte_count += stats.large_cte_count
            target.upward_improvements += stats.upward_improvements
            target.downward_improvements += stats.downward_improvements
            target.upward_degradations += stats.upward_degradations
            target.downward_degradations += stats.downward_degradations
            target.score_delta_sum += stats.score_delta_sum
            for failure_class, count in stats.failure_classes.items():
                target.failure_classes[failure_class] = target.failure_classes.get(failure_class, 0) + count
            final_scales[family] = state.family_scales.get(family, final_scales.get(family, 0.0))
            final_biases[family] = state.family_direction_bias.get(family, final_biases.get(family, 0.0))
    families: dict[str, object] = {}
    for family, stats in sorted(aggregate.items()):
        mean_score_delta = stats.score_delta_sum / stats.trials if stats.trials else None
        families[family] = {
            **stats.to_dict(),
            "mean_score_delta_vs_best_at_trial": mean_score_delta,
            "final_bound_scale": final_scales.get(family),
            "final_direction_bias": final_biases.get(family),
            "diagnostic_guidance": _family_diagnostic_guidance(
                family,
                stats,
                final_scales.get(family),
                final_biases.get(family),
            ),
        }
    changed_families = [
        family
        for family in families
        if int(dict(families[family]).get("trials") or 0) > 0
    ]
    best = sorted(
        changed_families,
        key=lambda family: (
            -int(dict(families[family]).get("improvements") or 0),
            int(dict(families[family]).get("failures") or 0),
            float(dict(families[family]).get("mean_score_delta_vs_best_at_trial") or 0.0),
            family,
        ),
    )
    worst = sorted(
        changed_families,
        key=lambda family: (
            -(
                int(dict(families[family]).get("failures") or 0)
                + int(dict(families[family]).get("degradations") or 0)
            ),
            int(dict(families[family]).get("improvements") or 0),
            family,
        ),
    )
    return {
        "families": families,
        "best_families": best[:3],
        "worst_families": worst[:3],
    }


def _family_diagnostic_guidance(
    family: str,
    stats: ParameterFamilyStats,
    bound_scale: Optional[float],
    direction_bias: Optional[float],
) -> str:
    if stats.failures + stats.degradations > stats.improvements:
        return f"{family} is associated with degradation/failure; keep bounds tight"
    if stats.improvements > 0:
        direction = "upward" if (direction_bias or 0.0) > 0.20 else "downward" if (direction_bias or 0.0) < -0.20 else "locally"
        return f"{family} is associated with improvements; bias {direction}"
    if bound_scale is not None and bound_scale < 0.30:
        return f"{family} remains uncertain; protected-best bounds were tightened"
    return f"{family} has limited evidence; continue local probing"


def _classify_failure(metrics: dict[str, object]) -> str:
    explicit = str(metrics.get("failure_class") or "").strip()
    if explicit:
        return explicit
    if bool(metrics.get("route_completion_success")):
        return ""
    reason = str(
        metrics.get("abort_reason")
        or metrics.get("last_failure_reason")
        or metrics.get("error")
        or ""
    ).lower()
    debug = metrics.get("trial_start_debug") if isinstance(metrics.get("trial_start_debug"), dict) else {}
    if "localization" in reason and ("timeout" in reason or "stability" in reason or "stable" in reason):
        return "localization_stability_timeout"
    if "no target" in reason or "target waypoint" in reason:
        return "no_target_waypoint"
    if "stuck" in reason or "no route progress" in reason or "no progress" in reason:
        return "vehicle_stuck_no_progress"
    if "lateral deviation" in reason or "cross track" in reason or "cross-track" in reason:
        return "large_lateral_deviation"
    if "control instability" in reason or "oscillation" in reason:
        return "control_instability"
    if bool(metrics.get("timeout")) or "timeout" in reason:
        return "timeout"
    activation_state = str(debug.get("route_activation_state") or metrics.get("route_activation_state") or "").lower()
    progress = _first_float(metrics, "route_progress_percent", "route_completion_percent", "progress_percent")
    if (
        "waiting" in activation_state
        or "activation" in reason
        or "did not start" in reason
        or "benchmark runner unavailable" in reason
        or "filter manager unavailable" in reason
        or (progress is not None and progress <= 0.0)
    ):
        return "failed_before_route_activation"
    return "route_failure"


def _closed_loop_extra_metadata(
    request: ClosedLoopAutoTuneRequest,
    noise_sig: str,
    run_folder: Path,
    locked_sensor_noise_sources: dict[str, str],
    search_profile: dict[str, object],
    direct_trial_results: list[dict[str, object]],
    direct_finalists: list[dict[str, object]],
    *,
    stage_budgets: ClosedLoopStageBudgets,
    stage_summaries: list[dict[str, object]],
    context_baseline: dict[str, object],
    best_passive: Optional[ClosedLoopValidationResult],
    best_active: Optional[ClosedLoopValidationResult],
    final_validation: ClosedLoopValidationResult,
    adaptive_stage_states: list[AdaptiveStageState],
    baseline_validation: Optional[ClosedLoopValidationResult],
) -> dict[str, object]:
    family_diagnostics = _aggregate_parameter_family_diagnostics(adaptive_stage_states)
    adaptation_history = [
        {"stage": state.stage, **dict(entry)}
        for state in adaptive_stage_states
        for entry in state.adaptation_history
    ]
    return {
        "source": "closed_loop_auto_tune",
        "direct_closed_loop_mode": True,
        "physical_output_folder": str(run_folder),
        "logical_output_group": _closed_loop_logical_group(request.filter_id, request.tracking_mode, noise_sig),
        "logical_output_root": str(_closed_loop_noise_root(request.output_root, request.filter_id, request.tracking_mode, noise_sig)),
        "logical_index_path": str(_closed_loop_index_path(request.output_root, request.filter_id, request.tracking_mode)),
        "candidate_generation_stage": "direct_closed_loop_route_trials",
        "closed_loop_validation_stage": "every_search_trial",
        "offline_auto_tune_output_folder": "",
        "offline_candidate_staging_folder": "",
        "direct_closed_loop_trial_results": direct_trial_results,
        "direct_closed_loop_finalists": direct_finalists,
        "direct_closed_loop_trial_count": len(direct_trial_results),
        "stage_budgets": stage_budgets.to_dict(),
        "tuning_stage_summary": list(stage_summaries),
        "context_baseline": dict(context_baseline),
        "best_passive_baseline": dict(best_passive.candidate_tune) if best_passive is not None else {},
        "best_passive_score": best_passive.closed_loop_score if best_passive is not None else None,
        "best_active_control_tune": dict(best_active.candidate_tune) if best_active is not None else {},
        "best_active_control_score": best_active.closed_loop_score if best_active is not None else None,
        "final_tune": dict(final_validation.candidate_tune),
        "final_objective_score": final_validation.closed_loop_score,
        "final_tuning_stage": final_validation.stage,
        "baseline_objective_score": (
            baseline_validation.closed_loop_score if baseline_validation is not None else None
        ),
        "baseline_to_final_improvement": (
            baseline_validation.closed_loop_score - final_validation.closed_loop_score
            if baseline_validation is not None
            else None
        ),
        "parameter_family_diagnostics": family_diagnostics["families"],
        "best_changed_parameter_families": family_diagnostics["best_families"],
        "worst_changed_parameter_families": family_diagnostics["worst_families"],
        "bound_adaptation_history": adaptation_history,
        "adaptive_stage_diagnostics": [state.to_dict() for state in adaptive_stage_states],
        "adaptive_search_policy": {
            "candidate_scope": "one parameter family or a small same-family group per trial",
            "anchor_policy": "all perturbations are anchored to the current protected best tune",
            "baseline_protection": "after repeated non-improvements all remaining stage bounds tighten around the best tune",
            "expansion_policy": "a family expands only after repeated stable improvements without failures",
            "route_geometry_preanalysis": False,
        },
        "sensor_noise_profile": request.sensor_noise_profile,
        "sensor_noise_config": dict(request.sensor_noise_config),
        "sensor_noise_signature": noise_sig,
        "tracking_mode": request.tracking_mode,
        "vehicle_behavior_profile": request.vehicle_behavior_profile,
        "vehicle_behavior_config": dict(request.vehicle_behavior_config),
        "locked_sensor_noise_sources": dict(locked_sensor_noise_sources),
        "metadata": dict(request.metadata),
        "auto_tune_profile_used": dict(search_profile),
        "actuator_search_policy": _actuator_search_policy(request.actuator_realism_config),
        "actuator_model_profile": request.actuator_realism_profile,
        "actuator_model_config": dict(request.actuator_realism_config),
        "route_attempt_policy": "one_attempt_per_candidate_trial",
        "max_route_attempts_per_trial": 1,
        "active_control_parameter_policy": (
            "Active tracking tune search includes supported active-control prediction parameters. "
            "The staged tuner first builds a passive Q/model baseline, then uses actuator-aware bounds "
            "before a narrow joint local fine-tune."
            if request.tracking_mode == TRACKING_ACTIVE
            else "Passive mode builds and locally refines only the passive Q/model baseline; active-control tuning is skipped."
        ),
        "performance_mode": {
            "no_rendering_mode": True,
            "fixed_delta_seconds_policy": "CARLA fixed_delta_seconds is not changed by the auto tuner.",
            "visual_workload_policy": "Direct closed-loop auto tune uses only a lightweight Pygame text progress monitor.",
        },
    }


def _closed_loop_search_profile(
    profile: dict[str, object],
    tracking_mode: str,
    actuator_realism_config: dict[str, object],
) -> dict[str, object]:
    result = process_only_auto_tune_profile(dict(profile))
    for group in ("primary", "secondary"):
        params = result.get(group)
        if not isinstance(params, list):
            result[group] = []
            continue
        filtered: list[dict[str, object]] = []
        for param in params:
            if not isinstance(param, dict):
                continue
            key = str(param.get("key") or "")
            if tracking_mode != TRACKING_ACTIVE and key in ACTIVE_CONTROL_TUNE_KEYS:
                continue
            filtered.append(_actuator_adjusted_param(dict(param), actuator_realism_config, tracking_mode))
        result[group] = filtered
    result["closed_loop_direct"] = True
    result["tracking_mode"] = tracking_mode
    result["actuator_search_policy"] = _actuator_search_policy(actuator_realism_config)
    return result


def _closed_loop_search_params(profile: dict[str, object]) -> list[dict[str, object]]:
    params: list[dict[str, object]] = []
    seen: set[str] = set()
    for group in ("primary", "secondary"):
        for param in list(profile.get(group) or []):
            if not isinstance(param, dict):
                continue
            key = str(param.get("key") or "")
            if not key or key in seen:
                continue
            low = _optional_float(param.get("min"))
            high = _optional_float(param.get("max"))
            if low is None and high is None:
                continue
            if low is not None and high is not None and high <= low:
                continue
            seen.add(key)
            params.append(dict(param))
    return params


def _actuator_adjusted_param(
    param: dict[str, object],
    actuator_realism_config: dict[str, object],
    tracking_mode: str,
) -> dict[str, object]:
    key = str(param.get("key") or "")
    if tracking_mode != TRACKING_ACTIVE or key not in ACTIVE_CONTROL_TUNE_KEYS:
        return param
    severity = _actuator_imperfection_score(actuator_realism_config)
    low = _optional_float(param.get("min"))
    high = _optional_float(param.get("max"))
    if high is None:
        return param
    if severity <= 0.08:
        param["search_policy"] = "perfect_actuator_confident"
        if key in {"control_accel_gain_mps2", "control_brake_decel_gain_mps2", "control_steer_to_yaw_rate_gain"}:
            param["max"] = high
        return param
    param["search_policy"] = "realistic_actuator_conservative"
    if key in {"control_accel_gain_mps2", "control_brake_decel_gain_mps2", "control_steer_to_yaw_rate_gain"}:
        param["max"] = max(low if low is not None else 0.0, high * 0.75)
    elif key in {"max_control_accel_delta_mps2", "max_control_yaw_rate_delta_radps", "max_control_speed_delta_mps"}:
        param["max"] = max(low if low is not None else 0.0, high * 0.65)
    elif key == "control_input_timeout_s":
        param["min"] = max(low if low is not None else 0.02, min(high, 0.08 + 0.20 * severity))
    return param


def _actuator_search_policy(actuator_realism_config: dict[str, object]) -> str:
    severity = _actuator_imperfection_score(actuator_realism_config)
    if severity <= 0.08:
        return "perfect_actuator_confident_active_search"
    if severity <= 0.35:
        return "mild_realistic_balanced_active_search"
    return "delayed_or_noisy_conservative_active_search"


def _actuator_imperfection_score(config: dict[str, object]) -> float:
    profile = str(config.get("preset_name") or "").lower()
    if "perfect" in profile:
        return 0.0
    delay = _optional_float(config.get("actuator_delay_s")) or 0.0
    noise = _optional_float(config.get("actuator_noise")) or 0.0
    smoothing = max(
        _optional_float(config.get("throttle_smoothing")) or 0.0,
        _optional_float(config.get("brake_smoothing")) or 0.0,
        _optional_float(config.get("steering_smoothing")) or 0.0,
    )
    score = 1.5 * delay + 8.0 * noise + 0.6 * smoothing
    if "delayed" in profile or "harsh" in profile:
        score += 0.25
    return max(0.0, min(1.0, score))


def _create_direct_optuna_study(
    strategy: str,
    request: ClosedLoopAutoTuneRequest,
    params: list[dict[str, object]],
    base_tune: dict[str, object],
    *,
    enqueue_baseline: bool = True,
) -> object:
    optuna_module = offline_auto_tuner_module.optuna
    if strategy != "direct_closed_loop_optuna_tpe" or optuna_module is None or not params:
        return None
    study = optuna_module.create_study(  # type: ignore[union-attr]
        direction="minimize",
        sampler=optuna_module.samplers.TPESampler(seed=int(request.random_seed)),  # type: ignore[union-attr]
    )
    if enqueue_baseline:
        study.enqueue_trial(
            {
                str(param.get("key")): _param_bound_or_base_value(param, base_tune)
                for param in params
                if str(param.get("key") or "")
            }
        )
    return study


def _direct_candidate_tune(
    *,
    base_tune: dict[str, object],
    params: list[dict[str, object]],
    optuna_trial: object,
    rng: random.Random,
    trial_index: int,
    sample_first: bool = False,
) -> dict[str, object]:
    candidate = dict(base_tune)
    if not params:
        return candidate
    if optuna_trial is not None:
        suggest_float = getattr(optuna_trial, "suggest_float")
        for param in params:
            key = str(param.get("key") or "")
            if not key:
                continue
            low, high = _param_bounds(param, base_tune.get(key))
            log = str(param.get("scale") or "").lower() == "log" and low > 0.0 and high > low
            candidate[key] = suggest_float(key, low, high, log=log)
        return candidate
    if trial_index == 1 and not sample_first:
        return candidate
    for param in params:
        key = str(param.get("key") or "")
        if key:
            candidate[key] = _sample_param(rng, param, base_tune.get(key))
    return candidate


def _param_bound_or_base_value(param: dict[str, object], base_tune: dict[str, object]) -> float:
    key = str(param.get("key") or "")
    low, high = _param_bounds(param, base_tune.get(key))
    base = _optional_float(base_tune.get(key))
    if base is None:
        return low + 0.5 * (high - low)
    return max(low, min(high, base))


def _sample_param(rng: random.Random, param: dict[str, object], base_value: object) -> float:
    low, high = _param_bounds(param, base_value)
    if str(param.get("scale") or "").lower() == "log" and low > 0.0 and high > low:
        return math.exp(rng.uniform(math.log(low), math.log(high)))
    return rng.uniform(low, high)


def _param_bounds(param: dict[str, object], base_value: object) -> tuple[float, float]:
    low = _optional_float(param.get("min"))
    high = _optional_float(param.get("max"))
    base = _optional_float(base_value)
    if low is None:
        low = max(1.0e-9, (base or 1.0) * 0.25)
    if high is None:
        high = max(low, (base or low) * 4.0)
    if high <= low:
        high = low + max(1.0e-6, abs(low) * 0.1)
    return float(low), float(high)


def _direct_auto_tune_trial_results(validations: list[ClosedLoopValidationResult]) -> list[AutoTuneTrialResult]:
    results: list[AutoTuneTrialResult] = []
    for validation in validations:
        metrics = dict(validation.raw_metrics)
        metrics["closed_loop_score"] = validation.closed_loop_score
        metrics["failure_class"] = validation.failure_class
        metrics["tuning_stage"] = validation.stage
        metrics["changed_parameters_summary"] = validation.changed_parameters_summary
        metrics["affected_families"] = list(validation.affected_families)
        metrics["adaptation_decision"] = validation.adaptation_decision
        metrics["outcome_class"] = validation.outcome_class
        metrics["parameter_attribution"] = [dict(item) for item in validation.parameter_attribution]
        results.append(
            AutoTuneTrialResult(
                trial_index=validation.finalist_rank,
                candidate_tune=dict(validation.candidate_tune),
                score=float(validation.closed_loop_score),
                metrics=metrics,
                output_folder=validation.output_folder,
                failed=not validation.route_completion_success,
                failure_reason=validation.failure_class or validation.abort_reason,
                candidate_id=f"{validation.stage or 'direct'}_trial_{validation.finalist_rank:03d}",
                candidate_type=str(validation.raw_metrics.get("candidate_type") or "direct_closed_loop"),
                stage=validation.stage or "closed_loop_search",
            )
        )
    return results


def _direct_finalists_from_validations(
    validations: list[ClosedLoopValidationResult],
    finalist_count: int,
) -> list[ClosedLoopFinalist]:
    ordered = sorted(validations, key=lambda item: item.closed_loop_score)
    finalists: list[ClosedLoopFinalist] = []
    for rank, validation in enumerate(ordered[: max(1, int(finalist_count or 1))], start=1):
        finalists.append(
            ClosedLoopFinalist(
                rank=rank,
                candidate_tune=dict(validation.candidate_tune),
                offline_score=float(validation.closed_loop_score),
                offline_metrics=dict(validation.raw_metrics),
                trial_index=validation.finalist_rank,
                source_output_folder=validation.output_folder,
            )
        )
    return finalists


def _ensure_backend_config_compatible(config: dict[str, object], request: ClosedLoopAutoTuneRequest) -> None:
    context = closed_loop_tune_context(
        filter_id=request.filter_id,
        tracking_mode=request.tracking_mode,
        sensor_noise_config=request.sensor_noise_config,
        vehicle_behavior_config=request.vehicle_behavior_config,
        actuator_realism_config=request.actuator_realism_config,
    )
    result = TuneCompatibility.check(config, context)
    if not result.compatible:
        raise ValueError(f"Generated closed-loop tune config failed compatibility check: {result.reason}")


def _offline_log_metadata(path: Path) -> dict[str, object]:
    path = Path(path)
    route_metadata = read_json(path.parent / "route_metadata.json")
    summary = read_json(path.parent / "recording_summary.json")
    sensor_noise_config = route_metadata.get("sensor_noise_config")
    vehicle_behavior_config = route_metadata.get("vehicle_behavior_config")
    return {
        "route_name": str(summary.get("route_name") or route_metadata.get("route_name") or path.parent.name),
        "map_name": str(summary.get("map_name") or route_metadata.get("map_name") or ""),
        "sensor_log_path": str(path),
        "sample_count": summary.get("sample_count"),
        "sensor_noise_preset": _profile_name(sensor_noise_config),
        "sensor_noise_config": sensor_noise_config if isinstance(sensor_noise_config, dict) else {},
        "recording_driver": str(summary.get("recording_driver") or route_metadata.get("recording_driver") or ""),
        "behavior_preset": _profile_name(vehicle_behavior_config),
        "created_at": str(summary.get("created_at") or route_metadata.get("created_at") or ""),
        "recording_id": path.parent.parent.name,
    }


def _closed_loop_filter_root(output_root: str, filter_id: str, tracking_mode: str) -> Path:
    return benchmark_root(output_root) / "closed_loop" / "auto_tune" / str(tracking_mode or TRACKING_PASSIVE) / slugify(filter_id, "filter")


def _closed_loop_noise_root(output_root: str, filter_id: str, tracking_mode: str, noise_sig: str) -> Path:
    return _closed_loop_filter_root(output_root, filter_id, tracking_mode) / noise_signature_slug(noise_sig)


def _closed_loop_physical_root(output_root: str, filter_id: str, tracking_mode: str) -> Path:
    tracking = "a" if str(tracking_mode or TRACKING_PASSIVE) == TRACKING_ACTIVE else "p"
    return benchmark_root(output_root) / "_at" / "cl" / tracking / _short_filter_slug(filter_id)


def _closed_loop_logical_group(filter_id: str, tracking_mode: str, noise_sig: str) -> str:
    return "/".join(
        (
            "closed_loop",
            "auto_tune",
            str(tracking_mode or TRACKING_PASSIVE),
            slugify(filter_id, "filter"),
            noise_signature_slug(noise_sig),
        )
    )


def _closed_loop_index_path(output_root: str, filter_id: str, tracking_mode: str) -> Path:
    return _closed_loop_filter_root(output_root, filter_id, tracking_mode) / "saved_tune_configs.json"


def _short_filter_slug(filter_id: str) -> str:
    return slugify(filter_id, "f")[:18]


def _update_closed_loop_saved_config_index(
    output_root: str,
    filter_id: str,
    tracking_mode: str,
    config_path: Path,
    config: dict[str, object],
) -> None:
    index_path = _closed_loop_index_path(output_root, filter_id, tracking_mode)
    index = read_json(index_path)
    configs = index.get("configs")
    items = [dict(item) for item in configs if isinstance(item, dict)] if isinstance(configs, list) else []
    best_metrics = config.get("best_metrics") if isinstance(config.get("best_metrics"), dict) else {}
    entry = {
        "path": str(config_path),
        "filter_id": filter_id,
        "schema_version": config.get("schema_version"),
        "benchmark_mode": config.get("benchmark_mode"),
        "tracking_mode": config.get("tracking_mode"),
        "tune_scope": config.get("tune_scope"),
        "recommended_usage": config.get("recommended_usage"),
        "created_at": config.get("created_at"),
        "noise_profile_label": config.get("sensor_noise_profile"),
        "noise_signature": config.get("noise_signature"),
        "vehicle_behavior_signature": config.get("vehicle_behavior_signature"),
        "actuator_realism_signature": config.get("actuator_realism_signature"),
        "validation_route_name": config.get("validation_route_name"),
        "score": config.get("score"),
        "mean_eval_position_rmse_m": best_metrics.get("eval_filtered_rmse_m") or best_metrics.get("filtered_rmse_m"),
        "log_count": len(config.get("selected_offline_logs") or []),
        "output_folder": config.get("output_folder"),
    }
    items = [item for item in items if item.get("path") != str(config_path)]
    items.insert(0, entry)
    write_json(index_path, {"configs": items})


def _write_finalists_csv(path: Path, finalists: list[ClosedLoopFinalist]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("rank", "trial_index", "offline_score", "candidate_tune", "offline_metrics", "source_output_folder"),
        )
        writer.writeheader()
        for finalist in finalists:
            writer.writerow(
                {
                    "rank": finalist.rank,
                    "trial_index": finalist.trial_index,
                    "offline_score": finalist.offline_score,
                    "candidate_tune": json.dumps(finalist.candidate_tune, sort_keys=True),
                    "offline_metrics": json.dumps(finalist.offline_metrics, sort_keys=True),
                    "source_output_folder": str(finalist.source_output_folder) if finalist.source_output_folder else "",
                }
            )


def _write_validations_csv(path: Path, validations: list[ClosedLoopValidationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "stage",
                "stage_trial_index",
                "finalist_rank",
                "closed_loop_score",
                "route_completion_success",
                "route_aborted",
                "timeout",
                "failure_class",
                "abort_reason",
                "outcome_class",
                "affected_families",
                "adaptation_decision",
                "completion_time_s",
                "eval_filtered_rmse_m",
                "filtered_rmse_m",
                "mean_cross_track_error_m",
                "max_cross_track_error_m",
                "position_nees_source",
                "mean_position_nees",
                "mean_position_nees_diagonal_approx",
                "changed_parameters_summary",
                "candidate_tune",
                "changed_parameters",
                "parameter_attribution",
                "raw_metrics",
                "output_folder",
            ),
        )
        writer.writeheader()
        for validation in validations:
            writer.writerow(
                {
                    "stage": validation.stage,
                    "stage_trial_index": validation.stage_trial_index,
                    "finalist_rank": validation.finalist_rank,
                    "closed_loop_score": validation.closed_loop_score,
                    "route_completion_success": validation.route_completion_success,
                    "route_aborted": validation.route_aborted,
                    "timeout": validation.timeout,
                    "failure_class": validation.failure_class,
                    "abort_reason": validation.abort_reason,
                    "outcome_class": validation.outcome_class,
                    "affected_families": ",".join(validation.affected_families),
                    "adaptation_decision": validation.adaptation_decision,
                    "completion_time_s": validation.completion_time_s,
                    "eval_filtered_rmse_m": validation.eval_filtered_rmse_m,
                    "filtered_rmse_m": validation.filtered_rmse_m,
                    "mean_cross_track_error_m": validation.mean_cross_track_error_m,
                    "max_cross_track_error_m": validation.max_cross_track_error_m,
                    "position_nees_source": validation.position_nees_source,
                    "mean_position_nees": validation.mean_position_nees,
                    "mean_position_nees_diagonal_approx": validation.mean_position_nees_diagonal_approx,
                    "changed_parameters_summary": validation.changed_parameters_summary,
                    "candidate_tune": json.dumps(validation.candidate_tune, sort_keys=True),
                    "changed_parameters": json.dumps(validation.changed_parameters, sort_keys=True),
                    "parameter_attribution": json.dumps(validation.parameter_attribution, sort_keys=True),
                    "raw_metrics": json.dumps(validation.raw_metrics, sort_keys=True),
                    "output_folder": str(validation.output_folder) if validation.output_folder else "",
                }
            )


def _resolve_candidate_generation_strategy(requested_strategy: str) -> str:
    requested = str(requested_strategy or "").strip().lower()
    if requested in {"random_plus_coordinate_refinement", "random", "direct_closed_loop_random"}:
        return "direct_closed_loop_random"
    if offline_auto_tuner_module.optuna is not None:
        return "direct_closed_loop_optuna_tpe"
    return "direct_closed_loop_random"


def _run_folder_name() -> str:
    return "cl" + datetime.now().strftime("%y%m%d_%H%M%S")


def _profile_name(config: object) -> str:
    if isinstance(config, dict):
        return str(config.get("preset_name") or "Custom")
    return "Custom"


def _emit(callback: Optional[ProgressCallback], event: str, payload: dict[str, object]) -> None:
    if callback is not None:
        callback(event, payload)


def _first_object(metrics: dict[str, object], *keys: str) -> object:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def _first_float(metrics: dict[str, object], *keys: str) -> Optional[float]:
    for key in keys:
        value = _optional_float(metrics.get(key))
        if value is not None:
            return value
    return None


def _optional_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _max_nis_type_mean(summary: object) -> Optional[float]:
    summary = _dict_value(summary)
    values = []
    for stats in summary.values():
        if isinstance(stats, dict):
            value = _optional_float(stats.get("mean"))
            if value is not None:
                values.append(value)
    return max(values) if values else None


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


def _path_or_none(value: object) -> Optional[Path]:
    if value is None:
        return None
    return Path(str(value))
