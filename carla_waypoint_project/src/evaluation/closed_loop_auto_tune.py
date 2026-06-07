"""Closed-loop benchmark auto-tuning backend.

This module intentionally has no pygame/UI dependency and does not launch
CARLA by itself.  Candidate generation is delegated to the offline replay
auto-tuner, and closed-loop finalist validation is delegated to an injected
runner so unit tests can exercise the flow without Unreal/CARLA.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Callable, Optional

from src.KalmanLab.registry import discover_filters
from src.evaluation import filter_auto_tuner as offline_auto_tuner_module
from src.evaluation.evaluation_artifacts import benchmark_root, read_json, slugify, unique_folder, write_json
from src.evaluation.filter_auto_tuner import (
    AutoTuneRequest,
    AutoTuneResult,
    OfflineBenchmarkAutoTuner,
    noise_profile_summary,
)
from src.evaluation.offline_replay_runner import _validate_windows_path_length
from src.evaluation.sensor_noise_tune_mapper import SensorNoiseTuneMapper, noise_signature
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
    offline_log_paths: tuple[Path, ...]
    validation_routes: tuple[ClosedLoopValidationRoute, ...]
    sensor_noise_config: dict[str, object]
    vehicle_behavior_config: dict[str, object]
    actuator_realism_config: dict[str, object]
    base_tune: dict[str, object] = field(default_factory=dict)
    auto_tune_profile: dict[str, object] = field(default_factory=dict)
    sensor_noise_profile: str = "Custom"
    vehicle_behavior_profile: str = "Custom"
    actuator_realism_enabled: bool = True
    actuator_realism_profile: str = "Custom"
    trial_count: int = 30
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
    """Two-stage closed-loop auto tuner backend.

    Stage 1 runs offline replay candidate generation. Stage 2 validates only
    finalists through the injected closed-loop runner.
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
        offline_request = AutoTuneRequest(
            filter_id=request.filter_id,
            sensor_log_paths=tuple(request.offline_log_paths),
            base_tune=base_tune,
            auto_tune_profile=dict(request.auto_tune_profile),
            max_trials=max(1, int(request.trial_count)),
            output_root=request.output_root,
            keep_trial_outputs=request.keep_trial_outputs,
            keep_only_best_trial_output=False,
            generate_trial_plots=request.generate_trial_plots,
            allow_mixed_noise_logs=request.allow_mixed_noise_logs,
            metadata={
                **dict(request.metadata),
                "random_seed": int(request.random_seed),
                "candidate_generation_strategy": request.strategy,
                "closed_loop_candidate_generation": True,
                "tracking_mode_for_final_validation": request.tracking_mode,
            },
        )
        _emit(progress_callback, "candidate_generation_started", {"trial_count": offline_request.max_trials, "strategy": strategy})
        offline_result = self._offline_tuner.run(
            offline_request,
            progress_callback=progress_callback,
            stop_requested=stop_requested,
        )
        finalists = _select_finalists(offline_result, request.finalist_count)
        if not finalists:
            raise ValueError("Offline candidate generation produced no successful finalist tunes.")
        _write_finalists_csv(run_folder / "finalists.csv", finalists)

        validation_results: list[ClosedLoopValidationResult] = []
        for finalist in finalists:
            if stop_requested is not None and stop_requested():
                break
            validation_request = ClosedLoopValidationRequest(
                filter_id=request.filter_id,
                tracking_mode=request.tracking_mode,
                finalist=finalist,
                validation_route=validation_route,
                sensor_noise_config=dict(request.sensor_noise_config),
                vehicle_behavior_config=dict(request.vehicle_behavior_config),
                actuator_realism_config=dict(request.actuator_realism_config),
                output_folder=run_folder / "validations" / f"f{finalist.rank:03d}",
            )
            _emit(progress_callback, "finalist_validation_started", {"finalist_rank": finalist.rank})
            runner_result = self._run_validation(validation_request)
            validation = _validation_result_from_runner(runner_result, validation_request)
            validation_results.append(validation)
            _emit(
                progress_callback,
                "finalist_validation_finished",
                {"finalist_rank": finalist.rank, "score": validation.closed_loop_score},
            )
        if not validation_results:
            raise ValueError("No closed-loop finalist validations completed.")

        best_validation = min(validation_results, key=lambda item: item.closed_loop_score)
        best_tune = dict(best_validation.candidate_tune)
        best_score = float(best_validation.closed_loop_score)
        best_metrics = dict(best_validation.raw_metrics)
        best_metrics["closed_loop_score"] = best_score
        validation_dicts = [result.to_dict() for result in validation_results]
        finalist_dicts = [finalist.to_dict() for finalist in finalists]
        _write_validations_csv(run_folder / "validations.csv", validation_results)

        config = build_closed_loop_schema_v2_config(
            filter_id=request.filter_id,
            filter_display_name=record.display_name,
            tracking_mode=request.tracking_mode,
            sensor_noise_profile=request.sensor_noise_profile,
            noise_sig=locked.signature,
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
            offline_candidate_results=finalist_dicts,
            closed_loop_validation_results=validation_dicts,
            score=best_score,
            best_metrics=best_metrics,
            best_tune=best_tune,
            base_tune=base_tune,
            locked_sensor_noise_values=dict(locked.values),
            output_folder=run_folder,
            extra=_closed_loop_extra_metadata(request, offline_result, locked.signature, run_folder, locked.sources),
        )
        _ensure_backend_config_compatible(config, request)
        saved_config_path = save_closed_loop_best_tune(request.output_root, request.filter_id, request.tracking_mode, run_folder, config)
        summary = dict(config)
        summary.update(
            {
                "offline_auto_tune_output_folder": str(offline_result.output_folder),
                "saved_config_path": str(saved_config_path),
            }
        )
        write_json(run_folder / "closed_loop_auto_tune_summary.json", summary)
        _emit(progress_callback, "completed", {"best_score": best_score, "saved_config_path": str(saved_config_path)})
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
        base_tune = dict(request.base_tune or getattr(record, "tune", {}) or {})
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
        if not request.offline_log_paths:
            raise ValueError("Select at least one offline sensor log for candidate generation.")
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
    failure_penalty = 0.0
    if not success:
        failure_penalty += 100000.0
    if aborted:
        failure_penalty += 50000.0
    if timeout:
        failure_penalty += 50000.0
    if metrics.get("abort_reason"):
        failure_penalty += 10000.0

    rmse = _first_float(metrics, "eval_filtered_rmse_m", "filtered_rmse_m", "mean_filtered_rmse_m")
    mean_cte = _first_float(metrics, "driving_mean_cross_track_error_m", "mean_cross_track_error_m")
    max_cte = _first_float(metrics, "driving_max_cross_track_error_m", "max_cross_track_error_m")
    completion_time = _first_float(metrics, "completion_time_s")
    score = failure_penalty
    score += 5.0 * (rmse if rmse is not None else 1000.0)
    score += 2.0 * (mean_cte if mean_cte is not None else 100.0)
    score += 0.5 * (max_cte if max_cte is not None else 100.0)
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
    if nis is not None and nis > 9.0:
        score += 2.0 * (nis - 9.0)
    if nees is not None and nees > 12.0:
        score += 2.0 * (nees - 12.0)

    oscillation = _first_float(metrics, "control_oscillation_score", "control_oscillation_count", "mean_abs_steer_rate")
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
    )


def _select_finalists(offline_result: AutoTuneResult, finalist_count: int) -> list[ClosedLoopFinalist]:
    successful = [
        trial
        for trial in offline_result.trial_results
        if not trial.failed and trial.score is not None and math.isfinite(float(trial.score))
    ]
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


def _closed_loop_extra_metadata(
    request: ClosedLoopAutoTuneRequest,
    offline_result: AutoTuneResult,
    noise_sig: str,
    run_folder: Path,
    locked_sensor_noise_sources: dict[str, str],
) -> dict[str, object]:
    offline_summary = read_json(Path(offline_result.output_folder) / "auto_tune_summary.json")
    offline_metadata = offline_summary.get("metadata") if isinstance(offline_summary.get("metadata"), dict) else {}
    return {
        "source": "closed_loop_auto_tune",
        "physical_output_folder": str(run_folder),
        "logical_output_group": _closed_loop_logical_group(request.filter_id, request.tracking_mode, noise_sig),
        "logical_output_root": str(_closed_loop_noise_root(request.output_root, request.filter_id, request.tracking_mode, noise_sig)),
        "logical_index_path": str(_closed_loop_index_path(request.output_root, request.filter_id, request.tracking_mode)),
        "candidate_generation_stage": "offline_replay_process_only",
        "closed_loop_validation_stage": "finalists_only",
        "offline_auto_tune_output_folder": str(offline_result.output_folder),
        "offline_candidate_staging_folder": str(offline_metadata.get("offline_candidate_staging_folder") or ""),
        "locked_sensor_noise_sources": dict(locked_sensor_noise_sources),
        "metadata": dict(request.metadata),
        "active_control_parameter_policy": (
            "Offline candidate generation uses passive sensor-log replay and does not validate active-control-specific "
            "parameters. Active tracking finalists are evaluated only during closed-loop validation."
            if request.tracking_mode == TRACKING_ACTIVE
            else "Passive closed-loop finalists are generated from passive offline replay and validated in passive closed-loop mode."
        ),
    }


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
                "finalist_rank",
                "closed_loop_score",
                "route_completion_success",
                "route_aborted",
                "timeout",
                "abort_reason",
                "completion_time_s",
                "eval_filtered_rmse_m",
                "filtered_rmse_m",
                "mean_cross_track_error_m",
                "max_cross_track_error_m",
                "position_nees_source",
                "mean_position_nees",
                "mean_position_nees_diagonal_approx",
                "candidate_tune",
                "raw_metrics",
                "output_folder",
            ),
        )
        writer.writeheader()
        for validation in validations:
            writer.writerow(
                {
                    "finalist_rank": validation.finalist_rank,
                    "closed_loop_score": validation.closed_loop_score,
                    "route_completion_success": validation.route_completion_success,
                    "route_aborted": validation.route_aborted,
                    "timeout": validation.timeout,
                    "abort_reason": validation.abort_reason,
                    "completion_time_s": validation.completion_time_s,
                    "eval_filtered_rmse_m": validation.eval_filtered_rmse_m,
                    "filtered_rmse_m": validation.filtered_rmse_m,
                    "mean_cross_track_error_m": validation.mean_cross_track_error_m,
                    "max_cross_track_error_m": validation.max_cross_track_error_m,
                    "position_nees_source": validation.position_nees_source,
                    "mean_position_nees": validation.mean_position_nees,
                    "mean_position_nees_diagonal_approx": validation.mean_position_nees_diagonal_approx,
                    "candidate_tune": json.dumps(validation.candidate_tune, sort_keys=True),
                    "raw_metrics": json.dumps(validation.raw_metrics, sort_keys=True),
                    "output_folder": str(validation.output_folder) if validation.output_folder else "",
                }
            )


def _resolve_candidate_generation_strategy(requested_strategy: str) -> str:
    requested = str(requested_strategy or "").strip().lower()
    if requested == "random_plus_coordinate_refinement":
        return "random_plus_coordinate_refinement"
    if offline_auto_tuner_module.optuna is not None:
        return "optuna_tpe"
    return "random_plus_coordinate_refinement"


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
