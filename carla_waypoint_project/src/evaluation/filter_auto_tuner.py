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
from src.evaluation.consistency_metrics import (
    MEASUREMENT_NOISE_TUNE_KEYS,
    consistency_report_from_summaries,
    severity_rank,
)
from src.evaluation.evaluation_artifacts import (
    RecordedLogInfo,
    benchmark_root,
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
    candidate_id: str = ""
    candidate_type: str = "generated"
    stage: str = "search"


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
    baseline_score: Optional[float] = None
    final_score: Optional[float] = None
    improved_over_baseline: bool = False
    recommendation_status: str = "not_run"
    verification_results: tuple[AutoTuneTrialResult, ...] = ()


@dataclass(frozen=True)
class AutoTuneCandidate:
    """A named tune candidate evaluated by the offline tuner."""

    candidate_id: str
    candidate_type: str
    tune: dict[str, object]
    source: str
    parent_candidate_id: str = ""


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
        objective_name = _normalize_objective_name(
            request.objective_name or request.auto_tune_profile.get("objective") or "balanced_score"
        )

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
        default_tune = dict(record.tune)
        default_tune.update(locked.values)
        current_tune = dict(record.tune)
        current_tune.update(dict(request.base_tune))
        current_tune.update(locked.values)
        effective_metadata = dict(request.metadata)
        effective_metadata.update(
            {
                "selected_log_noise_summary": noise,
                "sensor_noise_locked_from_profile": True,
                "process_only_tune": True,
                "effective_uncertainty_multiplier_policy": (
                    "Physical sensor noise stddev values are locked from the selected recorded-log profile. "
                    "Effective *_R_multiplier/process/covariance inflation tune keys may adjust filter uncertainty without "
                    "changing the recorded sensor-noise profile."
                ),
                "locked_sensor_noise_values": dict(locked.values),
                "locked_sensor_noise_sources": dict(locked.sources),
                "noise_signature": locked.signature,
                "representative_sensor_noise_config": dict(locked.representative_config),
                "objective_mode": objective_name,
            }
        )
        request = replace(
            request,
            base_tune=current_tune,
            auto_tune_profile=process_only_profile,
            objective_name=objective_name,
            metadata=effective_metadata,
        )
        max_trials = max(1, int(request.max_trials or 1))
        run_folder = unique_folder(
            _offline_auto_tune_physical_root(request.output_root, request.filter_id),
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
                "auto_tune_physical_output_folder": str(run_folder),
                "logical_output_group": _offline_auto_tune_logical_group(request.filter_id, locked.signature),
                "logical_output_root": str(_offline_auto_tune_noise_root(request.output_root, request.filter_id, locked.signature)),
                "logical_index_path": str(_saved_config_index_path(request.output_root, request.filter_id)),
            },
        )
        strategy = _resolve_candidate_generation_strategy(request)

        self._emit(
            progress_callback,
            "started",
            {
                "output_folder": str(run_folder),
                "trials": max_trials,
                "objective_mode": objective_name,
                "candidate_generation_strategy": strategy,
                "optuna_available": optuna is not None,
            },
        )
        evaluation_index = 1
        all_results: list[AutoTuneTrialResult] = []
        candidate_catalog: list[AutoTuneCandidate] = []

        baseline_candidates = self._baseline_candidates(
            request=request,
            record_default_tune=default_tune,
            current_tune=current_tune,
        )
        previous_candidates = self._previous_best_candidates(request, record.display_name)
        baseline_and_reference_candidates = _dedupe_candidates([*baseline_candidates, *previous_candidates])
        candidate_catalog.extend(baseline_and_reference_candidates)

        self._emit(progress_callback, "baseline_started", {"candidate_count": len(baseline_and_reference_candidates)})
        for candidate in baseline_and_reference_candidates:
            if stop_requested is not None and stop_requested():
                self._emit(progress_callback, "stopped", {"trial_index": evaluation_index})
                break
            result = self._run_trial(
                request,
                run_folder,
                evaluation_index,
                len(baseline_and_reference_candidates),
                candidate,
                progress_callback,
                stage="baseline",
            )
            all_results.append(result)
            self._emit_trial_finished(progress_callback, request, result, len(baseline_and_reference_candidates), None)
            evaluation_index += 1
        baseline_reference = _best_result(
            result
            for result in all_results
            if result.stage == "baseline" and result.candidate_type in {"default_base", "current_ui"}
        )
        self._emit(
            progress_callback,
            "baseline_finished",
            {
                "baseline_score": baseline_reference.score if baseline_reference is not None else None,
                "baseline_candidate_id": baseline_reference.candidate_id if baseline_reference is not None else "",
            },
        )

        generated_candidates = self._generated_candidates(request, max_trials=max_trials, strategy=strategy)
        candidate_catalog.extend(generated_candidates)
        search_best: Optional[AutoTuneTrialResult] = None
        self._emit(progress_callback, "search_started", {"candidate_count": len(generated_candidates), "strategy": strategy})
        for search_offset, candidate in enumerate(generated_candidates, start=1):
            if stop_requested is not None and stop_requested():
                self._emit(progress_callback, "stopped", {"trial_index": evaluation_index})
                break
            result = self._run_trial(
                request,
                run_folder,
                evaluation_index,
                len(generated_candidates),
                candidate,
                progress_callback,
                stage="search",
            )
            all_results.append(result)
            if not result.failed and result.score is not None and (search_best is None or float(result.score) < float(search_best.score)):
                search_best = result
                self._emit(
                    progress_callback,
                    "new_search_best",
                    {
                        "trial_index": search_offset,
                        "score": result.score,
                        "metrics": dict(result.metrics),
                        "candidate_id": result.candidate_id,
                        "candidate_type": result.candidate_type,
                    },
                )
            self._emit_trial_finished(
                progress_callback,
                request,
                result,
                len(generated_candidates),
                search_best.score if search_best is not None else None,
            )
            evaluation_index += 1

        verification_candidates = self._verification_candidates(
            baseline_and_reference_candidates,
            generated_candidates,
            all_results,
            max_candidates=int(request.metadata.get("verification_candidate_count") or 5),
        )
        self._emit(progress_callback, "verification_started", {"candidate_count": len(verification_candidates)})
        verification_results: list[AutoTuneTrialResult] = []
        for candidate in verification_candidates:
            if stop_requested is not None and stop_requested():
                self._emit(progress_callback, "stopped", {"trial_index": evaluation_index})
                break
            result = self._run_trial(
                request,
                run_folder,
                evaluation_index,
                len(verification_candidates),
                candidate,
                progress_callback,
                stage="verification",
            )
            verification_results.append(result)
            all_results.append(result)
            self._emit_trial_finished(progress_callback, request, result, len(verification_candidates), None)
            evaluation_index += 1

        baseline_verification = _best_result(
            result
            for result in verification_results
            if result.candidate_type in {"default_base", "current_ui"}
        )
        verification_winner = _best_result(verification_results)
        baseline_score = baseline_verification.score if baseline_verification is not None else (
            baseline_reference.score if baseline_reference is not None else None
        )
        final_score = verification_winner.score if verification_winner is not None else None
        improved = _verified_improvement(verification_winner, baseline_verification or baseline_reference)
        recommendation_status = "improved" if improved else "no_improved_tune_found"
        best_tune: dict[str, object] = dict(verification_winner.candidate_tune) if improved and verification_winner is not None else {}
        best_score: Optional[float] = float(verification_winner.score) if improved and verification_winner is not None and verification_winner.score is not None else None
        best_metrics: dict[str, object] = dict(verification_winner.metrics) if improved and verification_winner is not None else {}

        self._write_candidates_json(run_folder / "candidates.json", candidate_catalog, all_results, verification_winner, baseline_verification)
        self._write_trials_csv(run_folder / "trials.csv", all_results)
        self._write_verification_leaderboard(run_folder / "verification_leaderboard.csv", verification_results, baseline_verification)
        self._write_per_route_metrics_csv(run_folder / "per_route_metrics.csv", all_results)
        consistency_report = _combined_consistency_report(verification_results, verification_winner, baseline_verification)
        write_json(run_folder / "consistency_report.json", consistency_report)

        summary = self._summary_dict(
            request,
            record.display_name,
            run_folder,
            best_tune,
            best_score,
            best_metrics,
            all_results,
            baseline_score=baseline_score,
            final_score=final_score,
            verification_winner=verification_winner,
            baseline_result=baseline_verification or baseline_reference,
            improved=improved,
            recommendation_status=recommendation_status,
            consistency_report=consistency_report,
        )
        write_json(run_folder / "auto_tune_summary.json", summary)
        saved_config_path = None
        if improved and best_score is not None:
            saved_config_path = self.save_best_tune(request, record.display_name, run_folder, best_tune, best_score, best_metrics, all_results)
        self._apply_retention_policy(request, all_results, verification_winner if improved else None)
        self._emit(
            progress_callback,
            "completed",
            {
                "best_score": best_score,
                "best_metrics": best_metrics,
                "baseline_score": baseline_score,
                "final_score": final_score,
                "improved_over_baseline": improved,
                "recommendation_status": recommendation_status,
                "message": "Verified improved tune found." if improved else "No improved tune found.",
                "verification_winner": _trial_to_dict(verification_winner) if verification_winner is not None else {},
                "saved_config_path": str(saved_config_path) if saved_config_path else "",
            },
        )
        return AutoTuneResult(
            filter_id=request.filter_id,
            best_tune=best_tune,
            best_score=best_score,
            best_metrics=best_metrics,
            selected_logs=tuple(Path(path) for path in request.sensor_log_paths),
            trial_results=tuple(all_results),
            output_folder=run_folder,
            saved_config_path=saved_config_path,
            baseline_score=float(baseline_score) if baseline_score is not None else None,
            final_score=float(final_score) if final_score is not None else None,
            improved_over_baseline=improved,
            recommendation_status=recommendation_status,
            verification_results=tuple(verification_results),
        )

    def _run_trial(
        self,
        request: AutoTuneRequest,
        run_folder: Path,
        trial_index: int,
        trial_total: int,
        candidate: AutoTuneCandidate,
        progress_callback: Optional[ProgressCallback],
        stage: str = "search",
    ) -> AutoTuneTrialResult:
        self._emit(
            progress_callback,
            "trial_started",
            {
                "trial_index": trial_index,
                "trial_total": trial_total,
                "stage": stage,
                "candidate_id": candidate.candidate_id,
                "candidate_type": candidate.candidate_type,
                "filter_id": request.filter_id,
                "candidate_tune": dict(candidate.tune),
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
                    filter_tunes={request.filter_id: dict(candidate.tune)},
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
                candidate_tune=dict(candidate.tune),
                score=score,
                metrics=metrics,
                output_folder=result.output_folder,
                failed=failed,
                failure_reason=failure_reason,
                candidate_id=candidate.candidate_id,
                candidate_type=candidate.candidate_type,
                stage=stage,
            )
        except Exception as exc:
            trial_result = AutoTuneTrialResult(
                trial_index=trial_index,
                candidate_tune=dict(candidate.tune),
                score=None,
                metrics={},
                output_folder=None,
                failed=True,
                failure_reason=str(exc),
                candidate_id=candidate.candidate_id,
                candidate_type=candidate.candidate_type,
                stage=stage,
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
            trial_results=trial_results,
        )
        config_path = run_folder / "best_tune.json"
        write_json(config_path, config)
        _update_saved_config_index(request.output_root, request.filter_id, config_path, config)
        return config_path

    def _baseline_candidates(
        self,
        request: AutoTuneRequest,
        record_default_tune: dict[str, object],
        current_tune: dict[str, object],
    ) -> list[AutoTuneCandidate]:
        candidates = [
            AutoTuneCandidate(
                candidate_id="base_default",
                candidate_type="default_base",
                tune=dict(record_default_tune),
                source="filter_default_with_recorded_sensor_noise_lock",
            )
        ]
        if not _tunes_equivalent(record_default_tune, current_tune):
            candidates.append(
                AutoTuneCandidate(
                    candidate_id="current_ui",
                    candidate_type="current_ui",
                    tune=dict(current_tune),
                    source="current_ui_tune_with_recorded_sensor_noise_lock",
                )
            )
        return candidates

    def _previous_best_candidates(self, request: AutoTuneRequest, filter_display_name: str) -> list[AutoTuneCandidate]:
        representative = request.metadata.get("representative_sensor_noise_config") or {}
        context = offline_tune_context(request.filter_id, sensor_noise_config=representative)
        candidates: list[AutoTuneCandidate] = []
        for index, item in enumerate(list_saved_tune_configs(request.filter_id, output_root=request.output_root, context=context), start=1):
            path = Path(str(item.get("path") or ""))
            config = load_saved_tune_config(path)
            best_tune = config.get("best_tune")
            if not isinstance(best_tune, dict) or not best_tune:
                continue
            tune = dict(best_tune)
            tune.update(dict(request.metadata.get("locked_sensor_noise_values") or {}))
            candidates.append(
                AutoTuneCandidate(
                    candidate_id=f"previous_best_{index:02d}",
                    candidate_type="previous_compatible_best",
                    tune=tune,
                    source=str(path),
                )
            )
            if len(candidates) >= int(request.metadata.get("previous_best_candidate_count") or 1):
                break
        return candidates

    def _generated_candidates(self, request: AutoTuneRequest, max_trials: int, strategy: str) -> list[AutoTuneCandidate]:
        params = _search_params(request)
        if not params:
            return [
                AutoTuneCandidate(
                    candidate_id="generated_001",
                    candidate_type="generated",
                    tune=dict(request.base_tune),
                    source="base_fallback_no_search_params",
                )
            ][:max_trials]
        candidates: list[AutoTuneCandidate] = []
        for tune, source in self._designed_candidate_tunes(request, params):
            candidates.append(
                AutoTuneCandidate(
                    candidate_id=f"generated_{len(candidates) + 1:03d}",
                    candidate_type="generated_designed",
                    tune=tune,
                    source=source,
                )
            )
            if len(candidates) >= max_trials:
                return _dedupe_candidates(candidates)[:max_trials]

        remaining = max(0, max_trials - len(_dedupe_candidates(candidates)))
        if remaining <= 0:
            return _dedupe_candidates(candidates)[:max_trials]

        if strategy == "optuna_tpe" and optuna is not None:
            study = optuna.create_study(  # type: ignore[union-attr]
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=int(request.metadata.get("random_seed", 4084))),  # type: ignore[union-attr]
            )
            for _ in range(remaining):
                trial = study.ask()
                tune = self._optuna_candidate(request, trial, params)
                study.tell(trial, 1.0)
                candidates.append(
                    AutoTuneCandidate(
                        candidate_id=f"generated_{len(candidates) + 1:03d}",
                        candidate_type="generated_optuna",
                        tune=tune,
                        source="optuna_tpe_broad_range_sample",
                    )
                )
        else:
            rng = random.Random(int(request.metadata.get("random_seed", 4084)))
            while len(_dedupe_candidates(candidates)) < max_trials:
                candidates.append(
                    AutoTuneCandidate(
                        candidate_id=f"generated_{len(candidates) + 1:03d}",
                        candidate_type="generated_random",
                        tune=self._random_candidate_tune(request, params, rng),
                        source="random_broad_range_sample",
                    )
                )
                if len(candidates) >= max_trials * 4:
                    break
        return _renumber_candidates(_dedupe_candidates(candidates)[:max_trials])

    def _designed_candidate_tunes(
        self,
        request: AutoTuneRequest,
        params: list[dict[str, object]],
    ) -> list[tuple[dict[str, object], str]]:
        designed: list[tuple[dict[str, object], str]] = []
        for label, selector in (
            ("all_min", lambda param: _param_bound_or_sample_base(param, request.base_tune, "min")),
            ("all_mid", lambda param: _param_midpoint(param, request.base_tune)),
            ("all_max", lambda param: _param_bound_or_sample_base(param, request.base_tune, "max")),
        ):
            tune = dict(request.base_tune)
            for param in params:
                key = str(param.get("key") or "")
                tune[key] = selector(param)
            designed.append((tune, f"designed_{label}_full_range"))
        for param in params:
            key = str(param.get("key") or "")
            if not key:
                continue
            for bound_name in ("min", "max"):
                value = _param_bound_or_sample_base(param, request.base_tune, bound_name)
                tune = dict(request.base_tune)
                tune[key] = value
                designed.append((tune, f"designed_{key}_{bound_name}"))
        return designed

    def _random_candidate_tune(
        self,
        request: AutoTuneRequest,
        params: list[dict[str, object]],
        rng: random.Random,
    ) -> dict[str, object]:
        candidate = dict(request.base_tune)
        for param in params:
            key = str(param.get("key") or "")
            if not key:
                continue
            candidate[key] = _sample_param(rng, param, request.base_tune.get(key))
        return candidate

    def _optuna_candidate(self, request: AutoTuneRequest, trial: object, params: list[dict[str, object]]) -> dict[str, object]:
        candidate = dict(request.base_tune)
        suggest_float = getattr(trial, "suggest_float")
        for param in params:
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

    def _verification_candidates(
        self,
        baseline_candidates: list[AutoTuneCandidate],
        generated_candidates: list[AutoTuneCandidate],
        all_results: list[AutoTuneTrialResult],
        max_candidates: int,
    ) -> list[AutoTuneCandidate]:
        by_id = {candidate.candidate_id: candidate for candidate in [*baseline_candidates, *generated_candidates]}
        search_results = [
            result
            for result in all_results
            if result.stage == "search" and not result.failed and result.score is not None and math.isfinite(float(result.score))
        ]
        ordered_search = sorted(search_results, key=lambda result: float(result.score))
        selected = list(baseline_candidates)
        for result in ordered_search[: max(1, max_candidates)]:
            candidate = by_id.get(result.candidate_id)
            if candidate is not None:
                selected.append(candidate)
        return _dedupe_candidates(selected)

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
        nis_by_type_summary = _aggregate_nis_by_type_summary(filter_rows)
        mean_position_nees = _mean([value for value in position_nees_values if value is not None])
        mean_position_nees_approx = _mean([value for value in position_nees_approx_values if value is not None])
        position_nees_source = _aggregate_position_nees_source(filter_rows)
        consistency_report = consistency_report_from_summaries(
            nis_by_type_summary=nis_by_type_summary,
            mean_position_nees=mean_position_nees,
            mean_position_nees_diagonal_approx=mean_position_nees_approx,
            position_nees_source=position_nees_source,
        )
        return {
            "route_count": aggregate.get("route_count"),
            "mean_eval_position_rmse_m": _mean([value for value in eval_rmses if value is not None]),
            "mean_yaw_rmse_deg": _mean([value for value in yaw_rmses if value is not None]),
            "divergence_event_count": sum(value for value in divergence_counts if value is not None),
            "mean_nis": _mean([value for value in nis_values if value is not None]),
            "mean_nees": _mean([value for value in nees_values if value is not None]),
            "legacy_mean_nis_mixed": _mean([value for value in nis_values if value is not None]),
            "mean_position_nees": mean_position_nees,
            "mean_position_nees_diagonal_approx": mean_position_nees_approx,
            "position_nees_source": position_nees_source,
            "nis_by_type_summary": nis_by_type_summary,
            "consistency_report": consistency_report,
            "per_route_metrics": [_compact_route_metric(row) for row in filter_rows],
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
        *,
        baseline_score: Optional[float],
        final_score: Optional[float],
        verification_winner: Optional[AutoTuneTrialResult],
        baseline_result: Optional[AutoTuneTrialResult],
        improved: bool,
        recommendation_status: str,
        consistency_report: dict[str, object],
    ) -> dict[str, object]:
        improvement_percent = _improvement_percent(baseline_score, best_score if improved else final_score)
        winner_metrics = dict(verification_winner.metrics) if verification_winner is not None else {}
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
            "objective_mode": request.objective_name,
            "score_formula": _score_formula_description(),
            "score_notes": _score_notes(),
            "nis_nees_policy": _nis_nees_policy(),
            "unavailable_metrics_policy": _unavailable_metrics_policy(),
            "best_score": best_score,
            "best_metrics": best_metrics,
            "best_tune": dict(best_tune),
            "baseline_score": baseline_score,
            "baseline_metrics": dict(baseline_result.metrics) if baseline_result is not None else {},
            "baseline_candidate_id": baseline_result.candidate_id if baseline_result is not None else "",
            "final_score": final_score,
            "final_metrics": winner_metrics,
            "final_tune": dict(verification_winner.candidate_tune) if verification_winner is not None else {},
            "improvement_percent": improvement_percent,
            "improved_over_baseline": bool(improved),
            "recommendation_status": recommendation_status,
            "message": "Verified improved tune found." if improved else "No improved tune found.",
            "verification_winner": _trial_to_dict(verification_winner) if verification_winner is not None else {},
            "verification_winner_candidate_type": verification_winner.candidate_type if verification_winner is not None else "",
            "verification_winner_rmse_m": winner_metrics.get("mean_eval_position_rmse_m"),
            "verification_winner_nis_by_type": winner_metrics.get("nis_by_type_summary"),
            "verification_winner_position_nees": {
                "mean_position_nees": winner_metrics.get("mean_position_nees"),
                "mean_position_nees_diagonal_approx": winner_metrics.get("mean_position_nees_diagonal_approx"),
                "position_nees_source": winner_metrics.get("position_nees_source"),
            },
            "consistency_report": consistency_report,
            "per_route_metrics": _all_per_route_metrics(trial_results),
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
        trial_results: list[AutoTuneTrialResult],
    ) -> dict[str, object]:
        logs = [_log_metadata(path, request.output_root) for path in request.sensor_log_paths]
        noise = noise_profile_summary(logs)
        verified_result = next(
            (
                result
                for result in trial_results
                if result.stage == "verification"
                and not result.failed
                and result.score is not None
                and math.isclose(float(result.score), float(best_score), rel_tol=1.0e-12, abs_tol=1.0e-12)
                and _tunes_equivalent(result.candidate_tune, best_tune)
            ),
            None,
        )
        extra = {
            "source": "offline_auto_tune",
            "verified_recommendation": True,
            "recommendation_status": "improved",
            "verification_winner_candidate_id": verified_result.candidate_id if verified_result is not None else "",
            "verification_winner_candidate_type": verified_result.candidate_type if verified_result is not None else "",
            "noise_profile_label": noise["label"],
            "sensor_noise_config": noise.get("representative_config") or {},
            "physical_output_folder": str(run_folder),
            "logical_output_group": str(request.metadata.get("logical_output_group") or ""),
            "logical_output_root": str(request.metadata.get("logical_output_root") or ""),
            "logical_index_path": str(request.metadata.get("logical_index_path") or ""),
            "objective": request.objective_name,
            "objective_name": request.objective_name,
            "score_formula": _score_formula_description(),
            "score_notes": _score_notes(),
            "nis_nees_policy": _nis_nees_policy(),
            "unavailable_metrics_policy": _unavailable_metrics_policy(),
            "auto_tune_profile": dict(request.auto_tune_profile),
            "trial_output_policy": _trial_output_policy(request),
            "trial_count": len(trial_results),
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
                fieldnames=(
                    "trial_index",
                    "stage",
                    "candidate_id",
                    "candidate_type",
                    "score",
                    "failed",
                    "failure_reason",
                    "candidate_tune",
                    "metrics",
                    "output_folder",
                ),
            )
            writer.writeheader()
            for result in trial_results:
                writer.writerow(
                    {
                        "trial_index": result.trial_index,
                        "stage": result.stage,
                        "candidate_id": result.candidate_id,
                        "candidate_type": result.candidate_type,
                        "score": result.score,
                        "failed": result.failed,
                        "failure_reason": result.failure_reason,
                        "candidate_tune": json.dumps(result.candidate_tune, sort_keys=True),
                        "metrics": json.dumps(result.metrics, sort_keys=True),
                        "output_folder": str(result.output_folder) if result.output_folder else "",
                    }
                )

    @staticmethod
    def _write_candidates_json(
        path: Path,
        candidates: list[AutoTuneCandidate],
        results: list[AutoTuneTrialResult],
        verification_winner: Optional[AutoTuneTrialResult],
        baseline_result: Optional[AutoTuneTrialResult],
    ) -> None:
        result_groups: dict[str, list[dict[str, object]]] = {}
        for result in results:
            result_groups.setdefault(result.candidate_id, []).append(_trial_to_dict(result))
        data = {
            "candidate_count": len(candidates),
            "verification_winner_candidate_id": verification_winner.candidate_id if verification_winner is not None else "",
            "baseline_candidate_id": baseline_result.candidate_id if baseline_result is not None else "",
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_type": candidate.candidate_type,
                    "source": candidate.source,
                    "parent_candidate_id": candidate.parent_candidate_id,
                    "candidate_tune": dict(candidate.tune),
                    "evaluations": result_groups.get(candidate.candidate_id, []),
                }
                for candidate in candidates
            ],
        }
        write_json(path, data)

    @staticmethod
    def _write_verification_leaderboard(
        path: Path,
        verification_results: list[AutoTuneTrialResult],
        baseline_result: Optional[AutoTuneTrialResult],
    ) -> None:
        rows = sorted(
            [result for result in verification_results if not result.failed and result.score is not None],
            key=lambda result: float(result.score),
        )
        baseline_score = baseline_result.score if baseline_result is not None else None
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=(
                    "rank",
                    "candidate_id",
                    "candidate_type",
                    "score",
                    "improvement_percent_vs_baseline",
                    "mean_eval_position_rmse_m",
                    "mean_yaw_rmse_deg",
                    "consistency_status",
                    "position_nees_source",
                    "mean_position_nees",
                    "mean_position_nees_diagonal_approx",
                    "output_folder",
                    "candidate_tune",
                ),
            )
            writer.writeheader()
            for rank, result in enumerate(rows, start=1):
                metrics = result.metrics
                consistency = metrics.get("consistency_report") if isinstance(metrics.get("consistency_report"), dict) else {}
                writer.writerow(
                    {
                        "rank": rank,
                        "candidate_id": result.candidate_id,
                        "candidate_type": result.candidate_type,
                        "score": result.score,
                        "improvement_percent_vs_baseline": _improvement_percent(baseline_score, result.score),
                        "mean_eval_position_rmse_m": metrics.get("mean_eval_position_rmse_m"),
                        "mean_yaw_rmse_deg": metrics.get("mean_yaw_rmse_deg"),
                        "consistency_status": consistency.get("overall_status"),
                        "position_nees_source": metrics.get("position_nees_source"),
                        "mean_position_nees": metrics.get("mean_position_nees"),
                        "mean_position_nees_diagonal_approx": metrics.get("mean_position_nees_diagonal_approx"),
                        "output_folder": str(result.output_folder) if result.output_folder else "",
                        "candidate_tune": json.dumps(result.candidate_tune, sort_keys=True),
                    }
                )

    @staticmethod
    def _write_per_route_metrics_csv(path: Path, results: list[AutoTuneTrialResult]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            "stage",
            "candidate_id",
            "candidate_type",
            "trial_index",
            "route_index",
            "route_name",
            "filter_id",
            "sensor_log_path",
            "eval_position_rmse_m",
            "yaw_rmse_deg",
            "divergence_event_count",
            "mean_nis",
            "mean_position_nees",
            "mean_position_nees_diagonal_approx",
            "position_nees_source",
        )
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                for route_metric in _per_route_metrics(result):
                    writer.writerow({key: _csv_value(route_metric.get(key)) for key in fieldnames})

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
                "stage": result.stage,
                "candidate_id": result.candidate_id,
                "candidate_type": result.candidate_type,
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
        keep_result: Optional[AutoTuneTrialResult],
    ) -> None:
        if request.keep_trial_outputs and not request.keep_only_best_trial_output:
            return
        keep_trial_index = keep_result.trial_index if request.keep_only_best_trial_output and keep_result is not None else None
        for result in trial_results:
            output_folder = result.output_folder
            if output_folder is None:
                continue
            if request.keep_only_best_trial_output and result.trial_index == keep_trial_index:
                continue
            shutil.rmtree(output_folder, ignore_errors=True)
            parent = output_folder.parent
            try:
                parent.rmdir()
            except OSError:
                pass


class OfflineBenchmarkAutoTuner(FilterAutoTuner):
    """Mode-explicit name for the offline replay auto tuner."""


def objective_score(metrics: dict[str, object], objective_name: str = "balanced_score", failure_count: int = 0) -> float:
    """Compute an explainable score where lower is better."""
    objective = _normalize_objective_name(objective_name)
    rmse = _optional_float(metrics.get("mean_eval_position_rmse_m"))
    if rmse is None:
        return 1.0e9
    yaw = _optional_float(metrics.get("mean_yaw_rmse_deg")) or 0.0
    divergence = _optional_float(metrics.get("divergence_event_count")) or 0.0
    consistency = _consistency_report(metrics)
    consistency_error = _optional_float(consistency.get("consistency_error")) or 0.0
    consistency_status = str(consistency.get("overall_status") or "unavailable")
    failure_penalty = 100.0 * max(0, failure_count) + 10.0 * divergence

    if objective == "min_eval_rmse":
        return rmse + failure_penalty
    if objective == "min_rmse_with_consistency_guard":
        guard_penalty = 0.0
        if consistency_status == "warning":
            guard_penalty = max(1.0, 0.25 * rmse + consistency_error)
        elif consistency_status == "severe":
            guard_penalty = max(1000.0, 10.0 * rmse + 100.0 * consistency_error)
        return rmse + failure_penalty + guard_penalty
    if objective == "consistency_first":
        return 100.0 * severity_rank(consistency_status) + 25.0 * consistency_error + 0.1 * rmse + failure_penalty
    return rmse + 0.01 * yaw + failure_penalty + 2.0 * consistency_error + 5.0 * max(0, severity_rank(consistency_status) - 1)


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


def _offline_auto_tune_physical_root(output_root: str, filter_id: str) -> Path:
    return benchmark_root(output_root) / "_at" / "o" / _short_filter_slug(filter_id)


def _offline_auto_tune_logical_group(filter_id: str, noise_signature: str) -> str:
    return "/".join(
        (
            "offline_localization",
            "auto_tune",
            "offline_passive",
            slugify(filter_id, "filter"),
            noise_signature_slug(noise_signature),
        )
    )


def _auto_tune_trial_staging_root(output_root: str) -> Path:
    return benchmark_root(output_root) / "_tmp" / "at"


def _auto_tune_trial_staging_folder(request: AutoTuneRequest, run_folder: Path) -> Path:
    configured = request.metadata.get("offline_candidate_staging_folder")
    if configured:
        return Path(str(configured))
    name = run_folder.name
    suffix = name[1:] if name.startswith("a") else name
    return _auto_tune_trial_staging_root(request.output_root) / slugify(f"at{suffix}", "at")[:18]


def _closed_loop_auto_tune_filter_root(output_root: str, filter_id: str, tracking_mode: str) -> Path:
    return benchmark_root(output_root) / "closed_loop" / "auto_tune" / str(tracking_mode or TRACKING_PASSIVE) / slugify(filter_id, "filter")


def _short_filter_slug(filter_id: str) -> str:
    return slugify(filter_id, "f")[:18]


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
        "stage": result.stage,
        "candidate_id": result.candidate_id,
        "candidate_type": result.candidate_type,
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


def _search_params(request: AutoTuneRequest) -> list[dict[str, object]]:
    params: list[dict[str, object]] = []
    seen: set[str] = set()
    for group in ("primary", "secondary"):
        for param in list(request.auto_tune_profile.get(group) or []):
            if not isinstance(param, dict):
                continue
            key = str(param.get("key") or "")
            if not key or key in seen or key in MEASUREMENT_NOISE_TUNE_KEYS:
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


def _dedupe_candidates(candidates: list[AutoTuneCandidate]) -> list[AutoTuneCandidate]:
    result: list[AutoTuneCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        signature = _tune_signature(candidate.tune)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(candidate)
    return result


def _renumber_candidates(candidates: list[AutoTuneCandidate]) -> list[AutoTuneCandidate]:
    result: list[AutoTuneCandidate] = []
    for index, candidate in enumerate(candidates, start=1):
        if not candidate.candidate_id.startswith("generated_"):
            result.append(candidate)
            continue
        result.append(
            AutoTuneCandidate(
                candidate_id=f"generated_{index:03d}",
                candidate_type=candidate.candidate_type,
                tune=dict(candidate.tune),
                source=candidate.source,
                parent_candidate_id=candidate.parent_candidate_id,
            )
        )
    return result


def _tunes_equivalent(left: dict[str, object], right: dict[str, object]) -> bool:
    return _tune_signature(left) == _tune_signature(right)


def _tune_signature(tune: dict[str, object]) -> str:
    normalized: dict[str, object] = {}
    for key, value in sorted(tune.items()):
        number = _optional_float(value)
        normalized[str(key)] = round(number, 12) if number is not None else value
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _best_result(results: object) -> Optional[AutoTuneTrialResult]:
    successful = [
        result
        for result in list(results)
        if isinstance(result, AutoTuneTrialResult)
        and not result.failed
        and result.score is not None
        and math.isfinite(float(result.score))
    ]
    if not successful:
        return None
    return min(successful, key=lambda result: float(result.score))


def _verified_improvement(
    winner: Optional[AutoTuneTrialResult],
    baseline: Optional[AutoTuneTrialResult],
) -> bool:
    if winner is None or baseline is None or winner.score is None or baseline.score is None:
        return False
    if winner.candidate_type in {"default_base", "current_ui"}:
        return False
    baseline_score = float(baseline.score)
    winner_score = float(winner.score)
    min_delta = max(1.0e-9, abs(baseline_score) * 0.001)
    return winner_score < baseline_score - min_delta


def _improvement_percent(baseline_score: object, final_score: object) -> Optional[float]:
    baseline = _optional_float(baseline_score)
    final = _optional_float(final_score)
    if baseline is None or final is None or abs(baseline) <= 1.0e-12:
        return None
    return 100.0 * (baseline - final) / abs(baseline)


def _param_bound_or_sample_base(param: dict[str, object], base_tune: dict[str, object], bound_name: str) -> float:
    bound = _optional_float(param.get(bound_name))
    if bound is not None:
        return bound
    return _param_midpoint(param, base_tune)


def _param_midpoint(param: dict[str, object], base_tune: dict[str, object]) -> float:
    low = _optional_float(param.get("min"))
    high = _optional_float(param.get("max"))
    base = _optional_float(base_tune.get(str(param.get("key") or "")))
    if low is None:
        low = max(1.0e-9, (base or 1.0) * 0.25)
    if high is None:
        high = max(low, (base or low) * 4.0)
    if str(param.get("scale") or "").lower() == "log" and low > 0.0 and high > low:
        return math.exp((math.log(low) + math.log(high)) / 2.0)
    return 0.5 * (low + high)


def _compact_route_metric(row: dict[str, object]) -> dict[str, object]:
    return {
        "route_index": row.get("route_index"),
        "route_name": row.get("route_name"),
        "filter_id": row.get("filter_id"),
        "sensor_log_path": row.get("sensor_log_path"),
        "eval_position_rmse_m": row.get("eval_position_rmse_m") or row.get("position_rmse_m"),
        "yaw_rmse_deg": row.get("yaw_rmse_deg"),
        "divergence_event_count": row.get("divergence_event_count"),
        "mean_nis": row.get("mean_nis"),
        "legacy_mean_nis_mixed": row.get("legacy_mean_nis_mixed"),
        "mean_position_nees": row.get("mean_position_nees"),
        "mean_position_nees_diagonal_approx": row.get("mean_position_nees_diagonal_approx"),
        "position_nees_source": row.get("position_nees_source"),
        "nis_by_type_summary": row.get("nis_by_type_summary"),
    }


def _per_route_metrics(result: AutoTuneTrialResult) -> list[dict[str, object]]:
    metrics = result.metrics.get("per_route_metrics") if isinstance(result.metrics, dict) else None
    if not isinstance(metrics, list):
        return []
    rows: list[dict[str, object]] = []
    for row in metrics:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "stage": result.stage,
                "candidate_id": result.candidate_id,
                "candidate_type": result.candidate_type,
                "trial_index": result.trial_index,
                **dict(row),
            }
        )
    return rows


def _all_per_route_metrics(results: list[AutoTuneTrialResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        rows.extend(_per_route_metrics(result))
    return rows


def _combined_consistency_report(
    verification_results: list[AutoTuneTrialResult],
    verification_winner: Optional[AutoTuneTrialResult],
    baseline_result: Optional[AutoTuneTrialResult],
) -> dict[str, object]:
    return {
        "winner_candidate_id": verification_winner.candidate_id if verification_winner is not None else "",
        "winner_candidate_type": verification_winner.candidate_type if verification_winner is not None else "",
        "winner": _consistency_report(verification_winner.metrics) if verification_winner is not None else {},
        "baseline_candidate_id": baseline_result.candidate_id if baseline_result is not None else "",
        "baseline": _consistency_report(baseline_result.metrics) if baseline_result is not None else {},
        "by_candidate": {
            result.candidate_id: {
                "candidate_type": result.candidate_type,
                "score": result.score,
                "consistency_report": _consistency_report(result.metrics),
            }
            for result in verification_results
        },
    }


def _consistency_report(metrics: dict[str, object]) -> dict[str, object]:
    existing = metrics.get("consistency_report") if isinstance(metrics, dict) else None
    if isinstance(existing, dict):
        return existing
    return consistency_report_from_summaries(
        nis_by_type_summary=metrics.get("nis_by_type_summary") if isinstance(metrics, dict) else {},
        mean_position_nees=metrics.get("mean_position_nees") if isinstance(metrics, dict) else None,
        mean_position_nees_diagonal_approx=metrics.get("mean_position_nees_diagonal_approx") if isinstance(metrics, dict) else None,
        position_nees_source=metrics.get("position_nees_source") if isinstance(metrics, dict) else None,
    )


def _normalize_objective_name(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "rmse_consistency": "balanced_score",
        "balanced": "balanced_score",
        "rmse": "min_eval_rmse",
        "min_rmse": "min_eval_rmse",
        "consistency_guard": "min_rmse_with_consistency_guard",
    }
    text = aliases.get(text, text)
    if text in {"min_eval_rmse", "min_rmse_with_consistency_guard", "consistency_first", "balanced_score"}:
        return text
    return "balanced_score"


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


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
        "Objective modes: min_eval_rmse uses eval RMSE plus failure/divergence penalties; "
        "min_rmse_with_consistency_guard adds large penalties for warning/severe NIS/NEES; "
        "consistency_first ranks consistency severity/error ahead of RMSE; "
        "balanced_score blends RMSE, yaw, divergence/failures, and type-aware NIS/NEES consistency."
    )


def _score_notes() -> str:
    return (
        "Lower score is better. The final recommendation is chosen from the verification leaderboard, "
        "not from raw search trials. The legacy rmse_consistency objective name maps to balanced_score."
    )


def _nis_nees_policy() -> str:
    return (
        "NIS/NEES are classified by expected dimension: GNSS position NIS dim=2, IMU yaw/yaw-rate/accel NIS dim=1, "
        "and position NEES dim=2. Means above dimension indicate overconfidence; means below dimension indicate underconfidence."
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
