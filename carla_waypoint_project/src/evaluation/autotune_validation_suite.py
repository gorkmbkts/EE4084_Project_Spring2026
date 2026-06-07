"""Recorded-log smoke validation for the offline auto tuner."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Sequence

from src.KalmanLab.registry import discover_filters
from src.evaluation.evaluation_artifacts import benchmark_root, list_recorded_logs, read_json, timestamp_id, write_json
from src.evaluation.filter_auto_tuner import AutoTuneRequest, OfflineBenchmarkAutoTuner
from src.evaluation.sensor_noise_tune_mapper import noise_signature


DEFAULT_FILTERS = ("ca_kf", "ctra_ekf")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline autotune smoke validation on recorded logs.")
    parser.add_argument("--output-root", default="benchmark_results")
    parser.add_argument("--max-trials", type=int, default=4)
    parser.add_argument("--logs-per-group", type=int, default=2)
    parser.add_argument("--filters", nargs="*", default=list(DEFAULT_FILTERS))
    parser.add_argument("--objective", default="min_rmse_with_consistency_guard")
    args = parser.parse_args(argv)

    run_id = timestamp_id("av_")
    report_root = benchmark_root(args.output_root) / "autotune_validation" / run_id
    report_root.mkdir(parents=True, exist_ok=False)

    records = {record.filter_id: record for record in discover_filters() if record.valid}
    logs = list_recorded_logs(args.output_root)
    groups = _group_logs_by_noise_signature(logs)
    report_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    if not groups:
        write_json(
            report_root / "validation_summary.json",
            {
                "run_id": run_id,
                "status": "no_recorded_logs",
                "message": "No recorded logs found under benchmark_results/offline_localization/recordings.",
                "groups": [],
                "failures": [],
            },
        )
        _write_csv(report_root / "validation_runs.csv", report_rows)
        return 0

    tuner = OfflineBenchmarkAutoTuner()
    for group_key, group_logs in groups.items():
        selected = group_logs[: max(1, int(args.logs_per_group))]
        group_folder = report_root / _safe_group_name(group_key)
        group_folder.mkdir(parents=True, exist_ok=True)
        write_json(
            group_folder / "group_logs.json",
            {
                "group_key": group_key,
                "selected_log_count": len(selected),
                "selected_logs": [_recorded_info_to_dict(info) for info in selected],
            },
        )
        for filter_id in args.filters:
            record = records.get(str(filter_id))
            if record is None or not record.auto_tune_enabled or not isinstance(record.auto_tune_profile, dict):
                failures.append({"group_key": group_key, "filter_id": filter_id, "reason": "filter unavailable or not autotuneable"})
                continue
            try:
                result = tuner.run(
                    AutoTuneRequest(
                        filter_id=record.filter_id,
                        sensor_log_paths=tuple(info.sensor_log_path for info in selected),
                        base_tune=dict(record.tune),
                        auto_tune_profile=dict(record.auto_tune_profile),
                        max_trials=max(1, int(args.max_trials)),
                        objective_name=str(args.objective),
                        output_root=str(args.output_root),
                        keep_trial_outputs=False,
                        generate_trial_plots=False,
                        metadata={
                            "startup_mode": "autotune_validation_suite",
                            "validation_suite_run_id": run_id,
                            "candidate_generation_strategy": "random_plus_coordinate_refinement",
                        },
                    )
                )
                row = {
                    "run_id": run_id,
                    "group_key": group_key,
                    "filter_id": record.filter_id,
                    "log_count": len(selected),
                    "objective": args.objective,
                    "improved_over_baseline": result.improved_over_baseline,
                    "recommendation_status": result.recommendation_status,
                    "baseline_score": result.baseline_score,
                    "final_score": result.final_score,
                    "best_score": result.best_score,
                    "saved_config_path": str(result.saved_config_path) if result.saved_config_path else "",
                    "output_folder": str(result.output_folder),
                }
                report_rows.append(row)
                write_json(group_folder / f"{record.filter_id}_result.json", row)
            except Exception as exc:
                failure = {"group_key": group_key, "filter_id": filter_id, "reason": str(exc)}
                failures.append(failure)
                write_json(group_folder / f"{filter_id}_failure.json", failure)

    _write_csv(report_root / "validation_runs.csv", report_rows)
    write_json(
        report_root / "validation_summary.json",
        {
            "run_id": run_id,
            "status": "completed_with_failures" if failures else "completed",
            "output_folder": str(report_root),
            "filters": [str(item) for item in args.filters],
            "max_trials": int(args.max_trials),
            "logs_per_group": int(args.logs_per_group),
            "group_count": len(groups),
            "run_count": len(report_rows),
            "runs": report_rows,
            "failures": failures,
        },
    )
    return 1 if failures and not report_rows else 0


def _group_logs_by_noise_signature(logs: object) -> dict[str, list[object]]:
    groups: dict[str, list[object]] = {}
    for info in logs:
        metadata = read_json(Path(info.route_folder) / "route_metadata.json")
        sensor = metadata.get("sensor_noise_config")
        signature = noise_signature(sensor if isinstance(sensor, dict) else {})
        preset = str(getattr(info, "sensor_noise_preset", "") or "Custom")
        groups.setdefault(f"{preset}|{signature}", []).append(info)
    return groups


def _safe_group_name(group_key: str) -> str:
    return "group_" + hashlib.sha1(group_key.encode("utf-8")).hexdigest()[:12]


def _recorded_info_to_dict(info: object) -> dict[str, object]:
    return {
        "route_folder": str(getattr(info, "route_folder", "")),
        "sensor_log_path": str(getattr(info, "sensor_log_path", "")),
        "run_folder": str(getattr(info, "run_folder", "")),
        "recording_id": getattr(info, "recording_id", ""),
        "route_name": getattr(info, "route_name", ""),
        "map_name": getattr(info, "map_name", ""),
        "sample_count": getattr(info, "sample_count", None),
        "duration_s": getattr(info, "duration_s", None),
        "recording_driver": getattr(info, "recording_driver", ""),
        "sensor_noise_preset": getattr(info, "sensor_noise_preset", ""),
        "vehicle_behavior_preset": getattr(info, "vehicle_behavior_preset", ""),
        "created_at": getattr(info, "created_at", ""),
        "failure_reason": getattr(info, "failure_reason", ""),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "run_id",
        "group_key",
        "filter_id",
        "log_count",
        "objective",
        "improved_over_baseline",
        "recommendation_status",
        "baseline_score",
        "final_score",
        "best_score",
        "saved_config_path",
        "output_folder",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
