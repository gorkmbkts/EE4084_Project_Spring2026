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
    plots_dir = route_folder / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    result_dir = route_folder / "replay_results"
    metrics_dir = route_folder / "metrics"
    estimate_files = sorted(result_dir.glob("*_estimates.csv"))
    outputs = [
        _plot_trajectory(plots_dir / "trajectory_comparison.png", estimate_files),
        _plot_position_error(plots_dir / "position_error_over_time.png", estimate_files),
        _plot_rmse_bar(plots_dir / "rmse_comparison.png", metrics_dir / "summary_metrics.csv"),
    ]
    nis = _plot_metric_if_available(plots_dir / "nis_comparison.png", estimate_files, "nis", "NIS")
    nees = _plot_metric_if_available(plots_dir / "nees_comparison.png", estimate_files, "nees", "NEES")
    if nis is not None:
        outputs.append(nis)
    if nees is not None:
        outputs.append(nees)
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
        rmse = _float_or_none(row.get("position_rmse_m"))
        if filter_id and rmse is not None:
            grouped.setdefault(filter_id, []).append(rmse)
    if not grouped:
        return None
    labels = sorted(grouped)
    values = [sum(grouped[label]) / len(grouped[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#2d8a63")
    ax.set_title("Aggregate Position RMSE")
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
    for path in estimate_files:
        rows = _read_csv(path)
        label = "Raw GNSS" if _label_from_estimate_path(path) == "raw_gnss" else _label_from_estimate_path(path)
        _plot_time_series(ax, rows, "position_error_m", label)
    ax.set_title("Position Error Over Time")
    ax.set_xlabel("Time since replay start (s)")
    ax.set_ylabel("Position error (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_rmse_bar(output_path: Path, summary_path: Path) -> Path:
    rows = _read_csv(summary_path)
    labels = [str(row.get("filter_id") or "") for row in rows]
    values = [_float_or_none(row.get("position_rmse_m")) for row in rows]
    pairs = [(label, value) for label, value in zip(labels, values) if label and value is not None]
    fig, ax = plt.subplots(figsize=(10, 5))
    if pairs:
        ax.bar([item[0] for item in pairs], [item[1] for item in pairs], color="#2d8a63")
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.text(0.5, 0.5, "No RMSE values", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Position RMSE Comparison")
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
    t0 = _float_or_none(rows[0].get("timestamp")) or 0.0
    points = []
    for row in rows:
        timestamp = _float_or_none(row.get("timestamp"))
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


def _float_or_none(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None
