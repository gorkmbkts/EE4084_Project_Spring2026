"""Plot generation for offline localization replay outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_offline_route_plots(route_folder: Path) -> list[Path]:
    """Create required plots for one offline replay route folder."""
    route_folder = Path(route_folder)
    plots_dir = route_folder / ("plt" if (route_folder / "res").exists() or (route_folder / "met").exists() else "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    result_dir = route_folder / ("res" if (route_folder / "res").exists() else "replay_results")
    metrics_dir = route_folder / ("met" if (route_folder / "met").exists() else "metrics")
    estimate_files = sorted(result_dir.glob("*_estimates.csv"))
    outputs = [
        _plot_trajectory(plots_dir / "trajectory_comparison.png", estimate_files),
        _plot_position_error(plots_dir / "position_error_over_time.png", estimate_files),
        _plot_rmse_bar(
            plots_dir / "rmse_comparison_full_window.png",
            metrics_dir / "summary_metrics.csv",
            "full_position_rmse_m",
            "Full-Window Position RMSE",
        ),
        _plot_rmse_bar(
            plots_dir / "rmse_comparison_eval_window.png",
            metrics_dir / "summary_metrics.csv",
            "eval_position_rmse_m",
            "Evaluation-Window Position RMSE",
        ),
        _plot_rmse_bar(
            plots_dir / "rmse_comparison.png",
            metrics_dir / "summary_metrics.csv",
            "eval_position_rmse_m",
            "Evaluation-Window Position RMSE",
        ),
    ]
    nis = _plot_metric_if_available(plots_dir / "nis_comparison.png", estimate_files, "nis", "NIS")
    nees = _plot_metric_if_available(plots_dir / "nees_comparison.png", estimate_files, "position_nees", "Position NEES (full 2x2)")
    if nees is None:
        nees = _plot_metric_if_available(
            plots_dir / "nees_comparison.png",
            estimate_files,
            "position_nees_diagonal_approx",
            "Position NEES (diagonal approx)",
        )
    legacy_nees = _plot_metric_if_available(plots_dir / "legacy_nees_comparison.png", estimate_files, "nees", "Legacy NEES")
    if nis is not None:
        outputs.append(nis)
    if nees is not None:
        outputs.append(nees)
    if legacy_nees is not None:
        outputs.append(legacy_nees)
    return outputs


def generate_aggregate_rmse_plot(evaluation_folder: Path) -> Optional[Path]:
    path = Path(evaluation_folder) / "aggregate_summary.csv"
    rows = _read_csv(path)
    if not rows:
        return None
    output = Path(evaluation_folder) / "aggregate_rmse_comparison.png"
    grouped: dict[str, list[float]] = {}
    for row in rows:
        filter_id = str(row.get("filter_id") or "")
        rmse = _float_or_none(row.get("eval_position_rmse_m") or row.get("position_rmse_m"))
        if filter_id and rmse is not None:
            grouped.setdefault(filter_id, []).append(rmse)
    if not grouped:
        return None
    labels = sorted(grouped)
    values = [sum(grouped[label]) / len(grouped[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#2d8a63")
    ax.set_title("Aggregate Evaluation Position RMSE")
    ax.set_ylabel("RMSE (m)")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def _plot_trajectory(output_path: Path, estimate_files: list[Path]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))
    for path in estimate_files:
        rows = _read_csv(path)
        label = _label_from_estimate_path(path)
        if label == "raw_gnss":
            _plot_xy(ax, rows, "estimate_x", "estimate_y", "Raw GNSS", linestyle="--", alpha=0.8)
        else:
            _plot_xy(ax, rows, "estimate_x", "estimate_y", label, alpha=0.9)
    all_rows = _read_first_nonempty(estimate_files)
    _plot_xy(ax, all_rows, "ground_truth_x", "ground_truth_y", "Ground truth", color="black", linewidth=2.0)
    ax.set_title("Trajectory Comparison")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_position_error(output_path: Path, estimate_files: list[Path]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    warmup_end_s = None
    for path in estimate_files:
        rows = _read_csv(path)
        if warmup_end_s is None:
            warmup_end_s = _warmup_end_seconds(rows)
        label = "Raw GNSS" if _label_from_estimate_path(path) == "raw_gnss" else _label_from_estimate_path(path)
        _plot_time_series(ax, rows, "position_error_m", label)
    if warmup_end_s is not None and warmup_end_s > 0.0:
        ax.axvspan(0.0, warmup_end_s, color="#d99a2b", alpha=0.18, label="Excluded warm-up")
        ax.axvline(warmup_end_s, color="#9a6a1a", linestyle="--", linewidth=1.2)
    ax.set_title("Position Error Over Time")
    ax.set_xlabel("Time since replay start (s)")
    ax.set_ylabel("Position error (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_rmse_bar(output_path: Path, summary_path: Path, metric_field: str, title: str) -> Path:
    rows = _read_csv(summary_path)
    labels = [str(row.get("filter_id") or "") for row in rows]
    values = [_float_or_none(row.get(metric_field)) for row in rows]
    pairs = [(label, value) for label, value in zip(labels, values) if label and value is not None]
    fig, ax = plt.subplots(figsize=(10, 5))
    if pairs:
        ax.bar([item[0] for item in pairs], [item[1] for item in pairs], color="#2d8a63")
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.text(
            0.5,
            0.5,
            _empty_rmse_message(rows, metric_field),
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_title(title)
    ax.set_ylabel("RMSE (m)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_metric_if_available(
    output_path: Path,
    estimate_files: list[Path],
    field: str,
    label: str,
) -> Optional[Path]:
    has_data = False
    fig, ax = plt.subplots(figsize=(10, 5))
    for path in estimate_files:
        rows = _read_csv(path)
        if any(_float_or_none(row.get(field)) is not None for row in rows):
            has_data = True
            _plot_time_series(ax, rows, field, _label_from_estimate_path(path))
    if not has_data:
        plt.close(fig)
        return None
    ax.set_title(f"{label} Over Time")
    ax.set_xlabel("Time since replay start (s)")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_xy(
    ax: object,
    rows: list[dict[str, object]],
    x_key: str,
    y_key: str,
    label: str,
    **kwargs: object,
) -> None:
    points = [
        (x, y)
        for x, y in ((_float_or_none(row.get(x_key)), _float_or_none(row.get(y_key))) for row in rows)
        if x is not None and y is not None
    ]
    if not points:
        return
    xs, ys = zip(*points)
    ax.plot(xs, ys, label=label, **kwargs)


def _plot_time_series(ax: object, rows: list[dict[str, object]], value_key: str, label: str) -> None:
    if not rows:
        return
    t0 = _row_time_seconds(rows[0]) or 0.0
    points = []
    for row in rows:
        timestamp = _row_time_seconds(row)
        value = _float_or_none(row.get(value_key))
        if timestamp is not None and value is not None:
            points.append((timestamp - t0, value))
    if not points:
        return
    xs, ys = zip(*points)
    ax.plot(xs, ys, label=label)


def _read_first_nonempty(paths: list[Path]) -> list[dict[str, object]]:
    for path in paths:
        rows = _read_csv(path)
        if rows:
            return rows
    return []


def _read_csv(path: Path) -> list[dict[str, object]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as csv_file:
            return list(csv.DictReader(csv_file))
    except OSError:
        return []


def _label_from_estimate_path(path: Path) -> str:
    name = path.stem
    suffix = "_estimates"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _empty_rmse_message(rows: list[dict[str, object]], metric_field: str) -> str:
    if metric_field == "eval_position_rmse_m" and rows:
        valid_counts = [
            int(value)
            for value in (_float_or_none(row.get("valid_for_metrics_sample_count")) for row in rows)
            if value is not None
        ]
        excluded_counts = [
            int(value)
            for value in (_float_or_none(row.get("warmup_excluded_sample_count")) for row in rows)
            if value is not None
        ]
        valid_total = max(valid_counts) if valid_counts else 0
        excluded_total = max(excluded_counts) if excluded_counts else 0
        if valid_total <= 0:
            return (
                "No eval-window samples\n"
                f"valid_for_metrics=0, warm-up/excluded samples={excluded_total}"
            )
    return "No RMSE values"


def _float_or_none(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _row_time_seconds(row: dict[str, object]) -> Optional[float]:
    for key in ("seconds_since_replay_start", "seconds_since_recording_start", "seconds_since_teleport", "timestamp"):
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _warmup_end_seconds(rows: list[dict[str, object]]) -> Optional[float]:
    first_time = _row_time_seconds(rows[0]) if rows else None
    for row in rows:
        if _bool_value(row.get("valid_for_metrics"), default=True):
            current = _row_time_seconds(row)
            if current is None:
                return None
            return current - (first_time or 0.0)
    return None


def _bool_value(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return default
