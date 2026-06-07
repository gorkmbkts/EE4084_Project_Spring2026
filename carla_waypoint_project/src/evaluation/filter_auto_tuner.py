"""Offline auto-tuning for one filter over multiple recorded sensor logs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import math
from pathlib import Path
import random
import shutil
from typing import Callable, Optional

try:
    import optuna
except ImportError:  # pragma: no cover - depends on optional environment package.
    optuna = None

from src.KalmanLab.registry import discover_filters
from src.evaluation.benchmark_config import project_commit_hash
from src.evaluation.consistency_metrics import MEASUREMENT_NOISE_TUNE_KEYS
from src.evaluation.evaluation_artifacts import (
    RecordedLogInfo,
    list_recorded_logs,
    offline_root,
    read_json,
    slugify,
    unique_folder,
    write_json,
)
from src.evaluation.offline_replay_runner import OfflineReplayRequest, OfflineReplayRunner, _validate_windows_path_length
from src.evaluation.sensor_noise_tune_mapper import SensorNoiseTuneMapper, noise_signature, process_only_auto_tune_profile
from src.evaluation.tune_config_schema import (
    BENCHMARK_MODE_CLOSED_LOOP,
    BENCHMARK_MODE_OFFLINE,
    SCHEMA_VERSION,
    TRACKING_ACTIVE,
    TRACKING_PASSIVE,
    TUNE_SCOPE_CLOSED_LOOP_CANDIDATE,
    TUNE_SCOPE_CLOSED_LOOP_VALIDATED,
    TUNE_SCOPE_OFFLINE,
    TuneCompatibility,
    TuneContext,
    build_offline_schema_v2_config,
    closed_loop_tune_context,
    noise_signature_slug,
    offline_tune_context,
)


@dataclass(frozen=True)
class AutoTuneRequest:
    """Configuration for one offline auto-tuning run."""

    filter_id: str
    sensor_log_paths: tuple[Path, ...]
    base_tune: dict[str, object]
    auto_tune_profile: dict[str, object]
    max_trials: int = 30
    objective_name: str = "rmse_consistency"
    output_root: str = "benchmark_results"
    keep_trial_outputs: bool = True
    keep_only_best_trial_output: bool = False
    generate_trial_plots: bool = False
    allow_mixed_noise_logs: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AutoTuneTrialResult:
    """Result for one candidate tune."""

    trial_index: int
    candidate_tune: dict[str, object]
    score: Optional[float]
    metrics: dict[str, object]
    output_folder: Optional[Path]
    failed: bool
    failure_reason: str = ""


@dataclass(frozen=True)
class AutoTuneResult:
    """Summary returned after auto-tuning finishes."""

    filter_id: str
    best_tune: dict[str, object]
    best_score: Optional[float]
    best_metrics: dict[str, object]
    selected_logs: tuple[Path, ...]
    trial_results: tuple[AutoTuneTrialResult, ...]
    output_folder: Path
    saved_config_path: Optional[Path] = None


ProgressCallback = Callable[[str, dict[str, object]], None]


class FilterAutoTuner:
    """Run a small framework-safe offline tune search for one filter."""

    def __init__(self, runner_factory: Callable[[], OfflineReplayRunner] = OfflineReplayRunner) -> None:
        self._records = {record.filter_id: record for record in discover_filters() if record.valid}
        self._runner_factory = runner_factory

    def run(
        self,
        request: AutoTuneRequest,
        progress_callback: Optional[ProgressCallback] = None,
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> AutoTuneResult:
        if request.filter_id == "raw_gnss":
            raise ValueError("Raw GNSS is a baseline and cannot be auto-tuned.")
        if not request.sensor_log_paths:
            raise ValueError("Select at least one recorded sensor log for auto tune.")
        record = self._records.get(request.filter_id)
        if record is None or record.filter_class is None:
            raise ValueError(f"Filter is not available: {request.filter_id}")
        if not record.auto_tune_enabled or not request.auto_tune_profile.get("primary"):
            raise ValueError(f"No auto-tune profile for filter: {request.filter_id}")

        selected_logs_metadata = [_log_metadata(path, request.output_root) for path in request.sensor_log_paths]
        noise = noise_profile_summary(selected_logs_metadata)
        if noise.get("mixed") and not request.allow_mixed_noise_logs:
            raise ValueError(
                "Selected auto-tune logs use mixed sensor noise signatures. "
                "Select logs with the same noise profile/signature; mixed/general-purpose tuning is not enabled."
            )

        process_only_profile = process_only_auto_tune_profile(dict(request.auto_tune_profile))
        locked = SensorNoiseTuneMapper.locked_values(
            request.filter_id,
            dict(request.base_tune),
            noise.get("representative_config") or {},
            tuple(record.tune_specs),
        )
        base_tune = dict(request.base_tune)
        base_tune.update(locked.values)
        effective_metadata = dict(request.metadata)
        effective_metadata.update(
            {
                "selected_log_noise_summary": noise,
                "sensor_noise_locked_from_profile": True,
                "process_only_tune": True,
                "locked_sensor_noise_values": dict(locked.values),
                "locked_sensor_noise_sources": dict(locked.sources),
                "noise_signature": locked.signature,
                "representative_sensor_noise_config": dict(locked.representative_config),
            }
        )
        request = replace(
            request,
            base_tune=base_tune,
            auto_tune_profile=process_only_profile,
            metadata=effective_metadata,
        )
        max_trials = max(1, int(request.max_trials or 1))
        run_folder = unique_folder(
            _offline_auto_tune_noise_root(request.output_root, request.filter_id, locked.signature),
            _run_folder_name(request.filter_id, request.sensor_log_paths, request.output_root),
        )
        _validate_windows_path_length(run_folder / "auto_tune_summary.json", "Auto-tune summary file")
        run_folder.mkdir(parents=True, exist_ok=False)
        trial_staging_folder = _auto_tune_trial_staging_folder(request, run_folder)
        _validate_windows_path_length(
            trial_staging_folder / "t001" / "r001" / "met" / "summary_metrics.json",
            "Auto-tune trial metrics file",
        )
        trial_staging_folder.mkdir(parents=True, exist_ok=True)
        request = replace(
            request,
            metadata={
                **dict(request.metadata),
                "offline_candidate_staging_folder": str(trial_staging_folder),
                "auto_tune_final_output_folder": str(run_folder),
            },
        )
        trial_results: list[AutoTuneTrialResult] = []
        best_tune: dict[str, object] = dict(request.base_tune)
        best_score: Optional[float] = None
        best_metrics: dict[str, object] = {}
        strategy = _resolve_candidate_generation_strategy(request)

        self._emit(
            progress_callback,
            "started",
            {
                "output_folder": str(run_folder),
                "trials": max_trials,
                "candidate_generation_strategy": strategy,
                "optuna_available": optuna is not None,
            },
        )
        trial_index = 1
        if strategy == "optuna_tpe":
            study = optuna.create_study(  # type: ignore[union-attr]
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=int(request.metadata.get("random_seed", 4084))),  # type: ignore[union-attr]
            )
            for _ in range(max_trials):
                if stop_requested is not None and stop_requested():
                    self._emit(progress_callback, "stopped", {"trial_index": trial_index})
                    break
                optuna_trial = study.ask()
                candidate = self._optuna_candidate(request, optuna_trial)
                result = self._run_trial(request, run_folder, trial_index, max_trials, candidate, progress_callback)
                trial_results.append(result)
                study.tell(optuna_trial, float(result.score) if result.score is not None else 1.0e9)
                if not result.failed and result.score is not None and (best_score is None or result.score < best_score):
                    best_score = result.score
                    best_tune = dict(result.candidate_tune)
                    best_metrics = dict(result.metrics)
                    self._emit(progress_callback, "new_best", {"trial_index": trial_index, "score": best_score, "metrics": best_metrics})
                self._emit_trial_finished(progress_callback, request, result, max_trials, best_score)
                trial_index += 1
            request.metadata["optuna_study_trial_count"] = len(study.trials)
        else:
            candidates = self._random_candidates(request, max_trials=max_trials)
            for candidate in candidates:
                if stop_requested is not None and stop_requested():
                    self._emit(progress_callback, "stopped", {"trial_index": trial_index})
                    break
                result = self._run_trial(request, run_folder, trial_index, max_trials, candidate, progress_callback)
                trial_results.append(result)
                if not result.failed and result.score is not None and (best_score is None or result.score < best_score):
                    best_score = result.score
                    best_tune = dict(result.candidate_tune)
                    best_metrics = dict(result.metrics)
                    self._emit(progress_callback, "new_best", {"trial_index": trial_index, "score": best_score, "metrics": best_metrics})
                self._emit_trial_finished(progress_callback, request, result, max_trials, best_score)
                trial_index += 1

            for candidate in self._coordinate_candidates(request, best_tune, remaining=max_trials - len(trial_results)):
                if stop_requested is not None and stop_requested():
                    self._emit(progress_callback, "stopped", {"trial_index": trial_index})
                    break
                result = self._run_trial(request, run_folder, trial_index, max_trials, candidate, progress_callback)
                trial_results.append(result)
                if not result.failed and result.score is not None and (best_score is None or result.score < best_score):
                    best_score = result.score
                    best_tune = dict(result.candidate_tune)
                    best_metrics = dict(result.metrics)
                    self._emit(progress_callback, "new_best", {"trial_index": trial_index, "score": best_score, "metrics": best_metrics})
                self._emit_trial_finished(progress_callback, request, result, max_trials, best_score)
                trial_index += 1

        self._write_trials_csv(run_folder / "trials.csv", trial_results)
        summary = self._summary_dict(request, record.display_name, run_folder, best_tune, best_score, best_metrics, trial_results)
        write_json(run_folder / "auto_tune_summary.json", summary)
        saved_config_path = None
        if best_score is not None:
            saved_config_path = self.save_best_tune(request, record.display_name, run_folder, best_tune, best_score, best_metrics, trial_results)
        self._apply_retention_policy(request, trial_results, best_score)
        self._emit(
            progress_callback,
            "completed",
            {
                "best_score": best_score,
                "best_metrics": best_metrics,
                "saved_config_path": str(saved_config_path) if saved_config_path else "",
            },
        )
        return AutoTuneResult(
            filter_id=request.filter_id,
            best_tune=best_tune,
            best_score=best_score,
            best_metrics=best_metrics,
            selected_logs=tuple(Path(path) for path in request.sensor_log_paths),
            trial_results=tuple(trial_results),
            output_folder=run_folder,
            saved_config_path=saved_config_path,
        )

    def _run_trial(
        self,
        request: AutoTuneRequest,
        run_folder: Path,
        trial_index: int,
        trial_total: int,
        candidate_tune: dict[str, object],
        progress_callback: Optional[ProgressCallback],
    ) -> AutoTuneTrialResult:
        self._emit(
            progress_callback,
            "trial_started",
            {
                "trial_index": trial_index,
                "trial_total": trial_total,
                "filter_id": request.filter_id,
                "candidate_tune": dict(candidate_tune),
                "log_count": len(request.sensor_log_paths),
            },
        )
        staging_folder = Path(str(request.metadata.get("offline_candidate_staging_folder") or run_folder))
        replay_folder = staging_folder / f"t{trial_index:03d}"
        try:
            result = self._runner_factory().run(
                OfflineReplayRequest(
                    sensor_log_paths=tuple(Path(path) for path in request.sensor_log_paths),
                    selected_filter_ids=(request.filter_id,),
                    filter_tunes={request.filter_id: dict(candidate_tune)},
                    output_root=request.output_root,
                    include_raw_gnss_baseline=True,
                    run_folder_override=replay_folder,
                    generate_plots=request.generate_trial_plots,
                    replay_context="auto_tune_trial",
                )
            )
            metrics = self._trial_metrics(result.output_folder, request.filter_id)
            metrics["failure_count"] = len(result.failures)
            score = objective_score(metrics, objective_name=request.objective_name, failure_count=len(result.failures))
            failed = not math.isfinite(score)
            failure_reason = "" if not failed else "Objective score was not finite."
            trial_result = AutoTuneTrialResult(
                trial_index=trial_index,
                candidate_tune=dict(candidate_tune),
                score=score,
                metrics=metrics,
                output_folder=result.output_folder,
                failed=failed,
                failure_reason=failure_reason,
            )
        except Exception as exc:
            trial_result = AutoTuneTrialResult(
                trial_index=trial_index,
                candidate_tune=dict(candidate_tune),
                score=None,
                metrics={},
                output_folder=None,
                failed=True,
                failure_reason=str(exc),
            )
        write_json(run_folder / f"trial_{trial_index:03d}.json", _trial_to_dict(trial_result))
        return trial_result

    def save_best_tune(
        self,
        request: AutoTuneRequest,
        filter_display_name: str,
        run_folder: Path,
        best_tune: dict[str, object],
        best_score: float,
        best_metrics: dict[str, object],
        trial_results: list[AutoTuneTrialResult],
    ) -> Path:
        config = self._saved_config_dict(
            request=request,
            filter_display_name=filter_display_name,
            run_folder=run_folder,
            best_tune=best_tune,
            best_score=best_score,
            best_metrics=best_metrics,
            trial_count=len(trial_results),
        )
        config_path = run_folder / "best_tune.json"
        write_json(config_path, config)
        _update_saved_config_index(request.output_root, request.filter_id, config_path, config)
        return config_path

    def _random_candidates(self, request: AutoTuneRequest, max_trials: int) -> list[dict[str, object]]:
        primary = list(request.auto_tune_profile.get("primary") or [])
        random_budget = max(1, max_trials - len(primary) * 5)
        rng = random.Random(int(request.metadata.get("random_seed", 4084)))
        candidates = [dict(request.base_tune)]
        while len(candidates) < random_budget:
            candidate = dict(request.base_tune)
            for param in primary:
                key = str(param.get("key") or "")
                if not key or key in MEASUREMENT_NOISE_TUNE_KEYS:
                    continue
                candidate[key] = _sample_param(rng, param, request.base_tune.get(key))
            candidates.append(candidate)
        return candidates[:max_trials]

    def _coordinate_candidates(
        self,
        request: AutoTuneRequest,
        best_tune: dict[str, object],
        remaining: int,
    ) -> list[dict[str, object]]:
        if remaining <= 0:
            return []
        candidates: list[dict[str, object]] = []
        for param in list(request.auto_tune_profile.get("primary") or []):
            key = str(param.get("key") or "")
            base_value = _optional_float(best_tune.get(key))
            if not key or key in MEASUREMENT_NOISE_TUNE_KEYS or base_value is None:
                continue
            for multiplier in (0.5, 0.75, 1.25, 1.5, 2.0):
                candidate = dict(best_tune)
                candidate[key] = _clamp_value(base_value * multiplier, param)
                candidates.append(candidate)
                if len(candidates) >= remaining:
                    return candidates
        return candidates

    def _optuna_candidate(self, request: AutoTuneRequest, trial: object) -> dict[str, object]:
        candidate = dict(request.base_tune)
        suggest_float = getattr(trial, "suggest_float")
        for param in list(request.auto_tune_profile.get("primary") or []):
            if not isinstance(param, dict):
                continue
            key = str(param.get("key") or "")
            if not key or key in MEASUREMENT_NOISE_TUNE_KEYS:
                continue
            low = _optional_float(param.get("min"))
            high = _optional_float(param.get("max"))
            base = _optional_float(request.base_tune.get(key))
            if low is None:
                low = max(1.0e-9, (base or 1.0) * 0.25)
            if high is None:
                high = max(low, (base or low) * 4.0)
            log = str(param.get("scale") or "").lower() == "log" and low > 0.0 and high > low
            candidate[key] = suggest_float(key, low, high, log=log)
        return candidate

    @staticmethod
    def _trial_metrics(output_folder: Path, filter_id: str) -> dict[str, object]:
        aggregate = read_json(Path(output_folder) / "aggregate_summary.json")
        rows = aggregate.get("aggregate_rows")
        filter_rows = [row for row in rows if isinstance(row, dict) and row.get("filter_id") == filter_id] if isinstance(rows, list) else []
        eval_rmses = [_optional_float(row.get("eval_position_rmse_m") or row.get("position_rmse_m")) for row in filter_rows]
        yaw_rmses = [_optional_float(row.get("yaw_rmse_deg")) for row in filter_rows]
        divergence_counts = [_optional_float(row.get("divergence_event_count")) for row in filter_rows]
        nis_values = [_optional_float(row.get("mean_nis")) for row in filter_rows]
        nees_values = [_optional_float(row.get("mean_nees")) for row in filter_rows]
        position_nees_values = [_optional_float(row.get("mean_position_nees")) for row in filter_rows]
        position_nees_approx_values = [_optional_float(row.get("mean_position_nees_diagonal_approx")) for row in filter_rows]
        return {
            "route_count": aggregate.get("route_count"),
            "mean_eval_position_rmse_m": _mean([value for value in eval_rmses if value is not None]),
            "mean_yaw_rmse_deg": _mean([value for value in yaw_rmses if value is not None]),
            "divergence_event_count": sum(value for value in divergence_counts if value is not None),
            "mean_nis": _mean([value for value in nis_values if value is not None]),
            "mean_nees": _mean([value for value in nees_values if value is not None]),
            "legacy_mean_nis_mixed": _mean([value for value in nis_values if value is not None]),
            "mean_position_nees": _mean([value for value in position_nees_values if value is not None]),
            "mean_position_nees_diagonal_approx": _mean([value for value in position_nees_approx_values if value is not None]),
            "position_nees_source": _aggregate_position_nees_source(filter_rows),
            "nis_by_type_summary": _aggregate_nis_by_type_summary(filter_rows),
            "best_filter_id": aggregate.get("best_filter_id"),
            "raw_gnss_rmse_m": aggregate.get("raw_gnss_rmse_m"),
            "aggregate_summary_path": str(Path(output_folder) / "aggregate_summary.json"),
            "output_folder": str(output_folder),
        }

    def _summary_dict(
        self,
        request: AutoTuneRequest,
        filter_display_name: str,
        run_folder: Path,
        best_tune: dict[str, object],
        best_score: Optional[float],
        best_metrics: dict[str, object],
        trial_results: list[AutoTuneTrialResult],
    ) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "filter_id": request.filter_id,
            "filter_display_name": filter_display_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "tuner_kind": "offline_benchmark_autotuner",
            "benchmark_mode": BENCHMARK_MODE_OFFLINE,
            "tracking_mode": TRACKING_PASSIVE,
            "tune_scope": TUNE_SCOPE_OFFLINE,
            "sensor_noise_locked_from_profile": True,
            "process_only_tune": True,
            "candidate_generation_strategy": _resolve_candidate_generation_strategy(request),
            "optuna_available": optuna is not None,
            "objective": request.objective_name,
            "objective_name": request.objective_name,
            "score_formula": _score_formula_description(),
            "score_notes": _score_notes(),
            "nis_nees_policy": _nis_nees_policy(),
            "unavailable_metrics_policy": _unavailable_metrics_policy(),
            "best_score": best_score,
            "best_metrics": best_metrics,
            "best_tune": dict(best_tune),
            "base_tune": dict(request.base_tune),
            "locked_sensor_noise_values": dict(request.metadata.get("locked_sensor_noise_values") or {}),
            "auto_tune_profile": dict(request.auto_tune_profile),
            "trial_output_policy": _trial_output_policy(request),
            "selected_logs": [_log_metadata(path, request.output_root) for path in request.sensor_log_paths],
            "trial_count": len(trial_results),
            "project_commit": project_commit_hash(),
            "output_folder": str(run_folder),
            "metadata": dict(request.metadata),
        }

    def _saved_config_dict(
        self,
        request: AutoTuneRequest,
        filter_display_name: str,
        run_folder: Path,
        best_tune: dict[str, object],
        best_score: float,
        best_metrics: dict[str, object],
        trial_count: int,
    ) -> dict[str, object]:
        logs = [_log_metadata(path, request.output_root) for path in request.sensor_log_paths]
        noise = noise_profile_summary(logs)
        extra = {
            "source": "offline_auto_tune",
            "noise_profile_label": noise["label"],
            "sensor_noise_config": noise.get("representative_config") or {},
            "objective": request.objective_name,
            "objective_name": request.objective_name,
            "score_formula": _score_formula_description(),
            "score_notes": _score_notes(),
            "nis_nees_policy": _nis_nees_policy(),
            "unavailable_metrics_policy": _unavailable_metrics_policy(),
            "auto_tune_profile": dict(request.auto_tune_profile),
            "trial_output_policy": _trial_output_policy(request),
            "trial_count": trial_count,
        }
        return build_offline_schema_v2_config(
            filter_id=request.filter_id,
            filter_display_name=filter_display_name,
            sensor_noise_profile=str(noise["label"]),
            noise_sig=str(request.metadata.get("noise_signature") or noise["signature"]),
            representative_sensor_noise_config=dict(noise.get("representative_config") or {}),
            selected_logs=logs,
            candidate_generation_strategy=_resolve_candidate_generation_strategy(request),
            optuna_available=optuna is not None,
            optuna_study_path=None,
            score=best_score,
            best_metrics=best_metrics,
            best_tune=best_tune,
            base_tune=request.base_tune,
            locked_sensor_noise_values=dict(request.metadata.get("locked_sensor_noise_values") or {}),
            output_folder=run_folder,
            extra=extra,
        )

    @staticmethod
    def _write_trials_csv(path: Path, trial_results: list[AutoTuneTrialResult]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=("trial_index", "score", "failed", "failure_reason", "candidate_tune", "metrics", "output_folder"),
            )
            writer.writeheader()
            for result in trial_results:
                writer.writerow(
                    {
                        "trial_index": result.trial_index,
                        "score": result.score,
                        "failed": result.failed,
                        "failure_reason": result.failure_reason,
                        "candidate_tune": json.dumps(result.candidate_tune, sort_keys=True),
                        "metrics": json.dumps(result.metrics, sort_keys=True),
                        "output_folder": str(result.output_folder) if result.output_folder else "",
                    }
                )

    @staticmethod
    def _emit(callback: Optional[ProgressCallback], event: str, payload: dict[str, object]) -> None:
        if callback is not None:
            callback(event, payload)

    @staticmethod
    def _emit_trial_finished(
        callback: Optional[ProgressCallback],
        request: AutoTuneRequest,
        result: AutoTuneTrialResult,
        trial_total: int,
        best_score: Optional[float],
    ) -> None:
        FilterAutoTuner._emit(
            callback,
            "trial_finished",
            {
                "trial_index": result.trial_index,
                "trial_total": trial_total,
                "score": result.score,
                "best_score": best_score,
                "failed": result.failed,
                "failure_reason": result.failure_reason,
                "metrics": dict(result.metrics),
                "log_count": len(request.sensor_log_paths),
            },
        )

    @staticmethod
    def _apply_retention_policy(
        request: AutoTuneRequest,
        trial_results: list[AutoTuneTrialResult],
        best_score: Optional[float],
    ) -> None:
        if request.keep_trial_outputs and not request.keep_only_best_trial_output:
            return
        best_trial_index: Optional[int] = None
        if request.keep_only_best_trial_output and best_score is not None:
            for result in trial_results:
                if result.score == best_score and not result.failed:
                    best_trial_index = result.trial_index
                    break
        for result in trial_results:
            output_folder = result.output_folder
            if output_folder is None:
                continue
            if request.keep_only_best_trial_output and result.trial_index == best_trial_index:
                continue
            shutil.rmtree(output_folder, ignore_errors=True)
            parent = output_folder.parent
            try:
                parent.rmdir()
            except OSError:
                pass


class OfflineBenchmarkAutoTuner(FilterAutoTuner):
    """Mode-explicit name for the offline replay auto tuner."""


def objective_score(metrics: dict[str, object], objective_name: str = "rmse_consistency", failure_count: int = 0) -> float:
    """Compute a simple explainable score where lower is better."""
    rmse = _optional_float(metrics.get("mean_eval_position_rmse_m"))
    if rmse is None:
        return 1.0e9
    yaw = _optional_float(metrics.get("mean_yaw_rmse_deg")) or 0.0
    divergence = _optional_float(metrics.get("divergence_event_count")) or 0.0
    nis_penalty_source = _max_nis_type_mean(metrics.get("nis_by_type_summary"))
    nis = nis_penalty_source if nis_penalty_source is not None else _optional_float(metrics.get("legacy_mean_nis_mixed"))
    nees = _optional_float(metrics.get("mean_position_nees"))
    if nees is None:
        nees = _optional_float(metrics.get("mean_position_nees_diagonal_approx"))
    consistency_penalty = 0.0
    if nis is not None and nis > 9.0:
        consistency_penalty += 0.05 * (nis - 9.0)
    if nees is not None and nees > 12.0:
        consistency_penalty += 0.05 * (nees - 12.0)
    return rmse + 0.01 * yaw + 10.0 * divergence + 100.0 * max(0, failure_count) + consistency_penalty


def list_saved_tune_configs(
    filter_id: str,
    output_root: str = "benchmark_results",
    context: str | TuneContext | None = None,
    tracking_mode: str = TRACKING_PASSIVE,
    sensor_noise_config: object | None = None,
    vehicle_behavior_config: object | None = None,
    actuator_realism_config: object | None = None,
    include_legacy: bool = False,
) -> list[dict[str, object]]:
    tune_context = _context_from_args(
        filter_id=filter_id,
        context=context,
        tracking_mode=tracking_mode,
        sensor_noise_config=sensor_noise_config,
        vehicle_behavior_config=vehicle_behavior_config,
        actuator_realism_config=actuator_realism_config,
        include_legacy=include_legacy,
    )
    items: list[dict[str, object]] = []
    for index_path in _saved_config_index_paths(output_root, filter_id, tune_context, include_legacy=include_legacy):
        index = read_json(index_path)
        configs = index.get("configs")
        if isinstance(configs, list):
            items.extend(dict(item) for item in configs if isinstance(item, dict))
    filtered: list[dict[str, object]] = []
    for item in items:
        path = Path(str(item.get("path") or ""))
        config = load_saved_tune_config(path)
        compatibility = TuneCompatibility.check(config, tune_context)
        if not compatibility.compatible:
            continue
        merged = dict(item)
        merged.update(
            {
                "schema_version": config.get("schema_version"),
                "benchmark_mode": config.get("benchmark_mode"),
                "tracking_mode": config.get("tracking_mode"),
                "tune_scope": config.get("tune_scope"),
                "recommended_usage": config.get("recommended_usage"),
                "validation_route_name": config.get("validation_route_name"),
            }
        )
        filtered.append(merged)
    return filtered


def load_saved_tune_config(path: Path) -> dict[str, object]:
    return read_json(Path(path))


def tune_config_compatibility(config: dict[str, object], context: TuneContext) -> tuple[bool, str]:
    result = TuneCompatibility.check(config, context)
    return result.compatible, result.reason


def noise_profile_summary(logs: list[dict[str, object]]) -> dict[str, object]:
    labels = {str(log.get("sensor_noise_preset") or "Custom") for log in logs}
    signatures = {noise_signature(log.get("sensor_noise_config") or {}) for log in logs}
    if not logs:
        label = "Unknown"
    elif len(labels) == 1:
        label = next(iter(labels))
    else:
        label = "Mixed Noise"
    mixed = len(signatures) > 1
    signature = "mixed" if mixed else (next(iter(signatures)) if signatures else "")
    representative_config = logs[0].get("sensor_noise_config") if logs and not mixed else {}
    return {
        "label": label,
        "signature": signature,
        "representative_config": representative_config if isinstance(representative_config, dict) else {},
        "mixed": mixed,
    }


def _context_from_args(
    *,
    filter_id: str,
    context: str | TuneContext | None,
    tracking_mode: str,
    sensor_noise_config: object | None,
    vehicle_behavior_config: object | None,
    actuator_realism_config: object | None,
    include_legacy: bool,
) -> TuneContext:
    if isinstance(context, TuneContext):
        return context
    if context == "closed_loop":
        return closed_loop_tune_context(
            filter_id=filter_id,
            tracking_mode=tracking_mode,
            sensor_noise_config=sensor_noise_config,
            vehicle_behavior_config=vehicle_behavior_config,
            actuator_realism_config=actuator_realism_config,
            include_legacy=include_legacy,
        )
    return offline_tune_context(
        filter_id=filter_id,
        sensor_noise_config=sensor_noise_config,
        include_legacy=include_legacy,
    )


def _auto_tune_filter_root(output_root: str, filter_id: str) -> Path:
    return _offline_auto_tune_filter_root(output_root, filter_id)


def _offline_auto_tune_filter_root(output_root: str, filter_id: str) -> Path:
    return offline_root(output_root) / "auto_tune" / "offline_passive" / slugify(filter_id, "filter")


def _offline_auto_tune_noise_root(output_root: str, filter_id: str, noise_signature: str) -> Path:
    return _offline_auto_tune_filter_root(output_root, filter_id) / noise_signature_slug(noise_signature)


def _auto_tune_trial_staging_root(output_root: str) -> Path:
    root = Path(output_root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    return root / "_tmp" / "at"


def _auto_tune_trial_staging_folder(request: AutoTuneRequest, run_folder: Path) -> Path:
    configured = request.metadata.get("offline_candidate_staging_folder")
    if configured:
        return Path(str(configured))
    name = run_folder.name
    suffix = name[1:] if name.startswith("a") else name
    return _auto_tune_trial_staging_root(request.output_root) / slugify(f"at{suffix}", "at")[:18]


def _closed_loop_auto_tune_filter_root(output_root: str, filter_id: str, tracking_mode: str) -> Path:
    root = Path(output_root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    return root / "closed_loop" / "auto_tune" / str(tracking_mode or TRACKING_PASSIVE) / slugify(filter_id, "filter")


def _saved_config_index_path(output_root: str, filter_id: str) -> Path:
    return _auto_tune_filter_root(output_root, filter_id) / "saved_tune_configs.json"


def _legacy_saved_config_index_path(output_root: str, filter_id: str) -> Path:
    return offline_root(output_root) / "auto_tune" / slugify(filter_id, "filter") / "saved_tune_configs.json"


def _saved_config_index_paths(
    output_root: str,
    filter_id: str,
    context: TuneContext,
    include_legacy: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    if context.benchmark_mode == BENCHMARK_MODE_CLOSED_LOOP:
        paths.append(_closed_loop_auto_tune_filter_root(output_root, filter_id, context.tracking_mode) / "saved_tune_configs.json")
    else:
        paths.append(_saved_config_index_path(output_root, filter_id))
    if include_legacy:
        paths.append(_legacy_saved_config_index_path(output_root, filter_id))
    return paths


def _update_saved_config_index(output_root: str, filter_id: str, config_path: Path, config: dict[str, object]) -> None:
    index_path = _saved_config_index_path(output_root, filter_id)
    index = read_json(index_path)
    configs = index.get("configs")
    items = [dict(item) for item in configs if isinstance(item, dict)] if isinstance(configs, list) else []
    entry = {
        "path": str(config_path),
        "filter_id": filter_id,
        "schema_version": config.get("schema_version"),
        "benchmark_mode": config.get("benchmark_mode"),
        "tracking_mode": config.get("tracking_mode"),
        "tune_scope": config.get("tune_scope"),
        "recommended_usage": config.get("recommended_usage"),
        "created_at": config.get("created_at"),
        "noise_profile_label": config.get("noise_profile_label") or config.get("sensor_noise_profile"),
        "noise_signature": config.get("noise_signature"),
        "score": config.get("score"),
        "mean_eval_position_rmse_m": (config.get("best_metrics") or {}).get("mean_eval_position_rmse_m")
        if isinstance(config.get("best_metrics"), dict)
        else None,
        "log_count": len(config.get("selected_logs") or []),
        "output_folder": config.get("output_folder"),
    }
    items = [item for item in items if item.get("path") != str(config_path)]
    items.insert(0, entry)
    write_json(index_path, {"configs": items})


def _run_folder_name(filter_id: str, sensor_log_paths: tuple[Path, ...], output_root: str) -> str:
    return "a" + datetime.now().strftime("%y%m%d_%H%M%S")


def _log_metadata(path: Path, output_root: str) -> dict[str, object]:
    path = Path(path)
    matching = _recorded_info_by_path(output_root).get(str(path.resolve()))
    route_metadata = read_json(path.parent / "route_metadata.json")
    sensor_noise_config = route_metadata.get("sensor_noise_config")
    if matching is not None:
        return {
            "route_name": matching.route_name,
            "map_name": matching.map_name,
            "sensor_log_path": str(matching.sensor_log_path),
            "sample_count": matching.sample_count,
            "sensor_noise_preset": matching.sensor_noise_preset,
            "sensor_noise_config": sensor_noise_config if isinstance(sensor_noise_config, dict) else {},
            "recording_driver": matching.recording_driver,
            "behavior_preset": matching.vehicle_behavior_preset,
            "created_at": matching.created_at,
            "recording_id": matching.recording_id,
        }
    summary = read_json(path.parent / "recording_summary.json")
    return {
        "route_name": str(summary.get("route_name") or path.parent.name),
        "map_name": str(summary.get("map_name") or route_metadata.get("map_name") or ""),
        "sensor_log_path": str(path),
        "sample_count": summary.get("sample_count"),
        "sensor_noise_preset": _preset_name(sensor_noise_config),
        "sensor_noise_config": sensor_noise_config if isinstance(sensor_noise_config, dict) else {},
        "recording_driver": str(summary.get("recording_driver") or route_metadata.get("recording_driver") or ""),
        "behavior_preset": _preset_name(route_metadata.get("vehicle_behavior_config")),
        "created_at": str(summary.get("created_at") or route_metadata.get("created_at") or ""),
        "recording_id": path.parent.parent.name,
    }


def _recorded_info_by_path(output_root: str) -> dict[str, RecordedLogInfo]:
    result: dict[str, RecordedLogInfo] = {}
    for info in list_recorded_logs(output_root):
        try:
            result[str(info.sensor_log_path.resolve())] = info
        except OSError:
            result[str(info.sensor_log_path)] = info
    return result


def _trial_to_dict(result: AutoTuneTrialResult) -> dict[str, object]:
    return {
        "trial_index": result.trial_index,
        "candidate_tune": dict(result.candidate_tune),
        "score": result.score,
        "metrics": dict(result.metrics),
        "output_folder": str(result.output_folder) if result.output_folder else None,
        "failed": result.failed,
        "failure_reason": result.failure_reason,
    }


def _resolve_candidate_generation_strategy(request: AutoTuneRequest) -> str:
    requested = str(request.metadata.get("candidate_generation_strategy") or "").strip().lower()
    if requested == "random_plus_coordinate_refinement":
        return "random_plus_coordinate_refinement"
    if optuna is not None:
        return "optuna_tpe"
    return "random_plus_coordinate_refinement"


def _aggregate_position_nees_source(rows: list[dict[str, object]]) -> str:
    sources = {str(row.get("position_nees_source") or "") for row in rows if row.get("position_nees_source")}
    if "full_2x2" in sources:
        return "full_2x2"
    if "diagonal_approx" in sources:
        return "diagonal_approx"
    return "unavailable"


def _aggregate_nis_by_type_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        summary = row.get("nis_by_type_summary")
        if isinstance(summary, str):
            try:
                parsed = json.loads(summary)
            except json.JSONDecodeError:
                parsed = {}
            summary = parsed if isinstance(parsed, dict) else {}
        if not isinstance(summary, dict):
            continue
        for update_type, stats in summary.items():
            if isinstance(stats, dict):
                grouped.setdefault(str(update_type), []).append(stats)
    result: dict[str, object] = {}
    for update_type, stats_list in grouped.items():
        means = [_optional_float(stats.get("mean")) for stats in stats_list]
        p95s = [_optional_float(stats.get("p95")) for stats in stats_list]
        p99s = [_optional_float(stats.get("p99")) for stats in stats_list]
        counts = [_optional_float(stats.get("sample_count")) for stats in stats_list]
        result[update_type] = {
            "mean": _mean([value for value in means if value is not None]),
            "p95": max((value for value in p95s if value is not None), default=None),
            "p99": max((value for value in p99s if value is not None), default=None),
            "sample_count": int(sum(value for value in counts if value is not None)),
            "expected_dimension": next((stats.get("expected_dimension") for stats in stats_list if stats.get("expected_dimension")), None),
        }
    return result


def _max_nis_type_mean(summary: object) -> Optional[float]:
    if isinstance(summary, str):
        try:
            parsed = json.loads(summary)
        except json.JSONDecodeError:
            parsed = {}
        summary = parsed if isinstance(parsed, dict) else {}
    if not isinstance(summary, dict):
        return None
    values = []
    for stats in summary.values():
        if isinstance(stats, dict):
            value = _optional_float(stats.get("mean"))
            if value is not None:
                values.append(value)
    return max(values) if values else None


def _sample_param(rng: random.Random, param: dict[str, object], fallback: object) -> float:
    low = _optional_float(param.get("min"))
    high = _optional_float(param.get("max"))
    base = _optional_float(fallback)
    if low is None:
        low = max(1.0e-9, (base or 1.0) * 0.25)
    if high is None:
        high = max(low, (base or low) * 4.0)
    if str(param.get("scale") or "").lower() == "log" and low > 0.0 and high > low:
        return math.exp(rng.uniform(math.log(low), math.log(high)))
    return rng.uniform(low, high)


def _clamp_value(value: float, param: dict[str, object]) -> float:
    low = _optional_float(param.get("min"))
    high = _optional_float(param.get("max"))
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _score_formula_description() -> str:
    return (
        "score = mean_eval_position_rmse_m + 0.01 * mean_yaw_rmse_deg + "
        "10 * divergence_event_count + 100 * failures + consistency_penalty; "
        "NIS/NEES penalties are skipped when unavailable."
    )


def _score_notes() -> str:
    return (
        "rmse_consistency is a heuristic score. RMSE is the dominant term. "
        "Yaw, divergence, failures, and large NIS/NEES values are penalized when available."
    )


def _nis_nees_policy() -> str:
    return (
        "NIS/NEES thresholds are fixed heuristic thresholds used for ranking only, "
        "not formal chi-square acceptance bounds."
    )


def _unavailable_metrics_policy() -> str:
    return (
        "Unavailable yaw, NIS, or NEES metrics do not add penalty. Missing eval RMSE makes the trial noncompetitive."
    )


def _trial_output_policy(request: AutoTuneRequest) -> dict[str, object]:
    return {
        "trial_outputs_root": str(request.metadata.get("offline_candidate_staging_folder") or "."),
        "trial_output_folder_pattern": "t###",
        "compact_trial_paths": True,
        "keep_trial_outputs": bool(request.keep_trial_outputs),
        "keep_only_best_trial_output": bool(request.keep_only_best_trial_output),
        "generate_trial_plots": bool(request.generate_trial_plots),
        "normal_evaluations_directory_used": False,
    }


def _preset_name(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("preset_name") or value.get("profile") or "")
    return ""


def _optional_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)
