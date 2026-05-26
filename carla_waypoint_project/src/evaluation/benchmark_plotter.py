"""Matplotlib plot generation for single-run Kalman benchmarks."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_benchmark_plots(benchmark_folder: Path) -> list[Path]:
    """Generate benchmark plots from samples.csv, metadata.json, and summary.json."""
    benchmark_folder = Path(benchmark_folder)
    plots_dir = benchmark_folder / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metadata = _read_json(benchmark_folder / "metadata.json")
    summary = _read_json(benchmark_folder / "summary.json")
    samples = _read_samples(benchmark_folder / "samples.csv")

    outputs = [
        _plot_trajectory(plots_dir / "trajectory_comparison.png", metadata, samples),
        _plot_localization_error(plots_dir / "localization_error_over_time.png", samples),
        _plot_cross_track_error(plots_dir / "cross_track_error_over_time.png", samples),
        _plot_summary_dashboard(plots_dir / "summary_dashboard.png", metadata, summary, samples),
    ]
    return outputs


def _plot_trajectory(output_path: Path, metadata: dict[str, object], samples: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_trajectory_panel(ax, metadata, samples)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_localization_error(output_path: Path, samples: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_localization_error_panel(ax, samples)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_cross_track_error(output_path: Path, samples: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_series(ax, samples, "cross_track_error_m", "Cross-track error")
    ax.set_title("Cross-Track Error Over Time")
    ax.set_xlabel("Time since benchmark start (s)")
    ax.set_ylabel("Cross-track error (m)")
    ax.grid(True, alpha=0.3)
    _legend_if_labels(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_summary_dashboard(
    output_path: Path,
    metadata: dict[str, object],
    summary: dict[str, object],
    samples: list[dict[str, object]],
) -> Path:
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(3, 2, width_ratios=(2.2, 1.0), height_ratios=(1.25, 1.0, 1.0))
    trajectory_ax = fig.add_subplot(grid[0, 0])
    localization_ax = fig.add_subplot(grid[1, 0])
    cross_track_ax = fig.add_subplot(grid[2, 0])
    text_ax = fig.add_subplot(grid[:, 1])

    _draw_trajectory_panel(trajectory_ax, metadata, samples)
    _draw_localization_error_panel(localization_ax, samples)
    _plot_series(cross_track_ax, samples, "cross_track_error_m", "Cross-track error")
    cross_track_ax.set_title("Cross-Track Error")
    cross_track_ax.set_xlabel("Time since benchmark start (s)")
    cross_track_ax.set_ylabel("m")
    cross_track_ax.grid(True, alpha=0.3)
    _legend_if_labels(cross_track_ax, fontsize=8)

    text_ax.axis("off")
    text_ax.text(
        0.0,
        1.0,
        _summary_text(metadata, summary),
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.5,
        linespacing=1.25,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _draw_trajectory_panel(ax, metadata: dict[str, object], samples: list[dict[str, object]]) -> None:
    route_points = _route_points(metadata)
    if route_points:
        ax.plot(
            [point["x"] for point in route_points],
            [point["y"] for point in route_points],
            color="black",
            linewidth=2,
            linestyle="--",
            label="Planned route",
        )

    _plot_xy(ax, samples, "ground_truth_x", "ground_truth_y", "Ground truth trajectory", color="tab:green")
    _plot_xy(ax, samples, "kalman_x", "kalman_y", "Kalman estimated trajectory", color="tab:blue", linestyle=":")
    _scatter_xy(ax, samples, "gnss_x", "gnss_y", "Raw noisy GNSS", color="tab:orange")

    general = metadata.get("general", {}) if isinstance(metadata.get("general"), dict) else {}
    start = general.get("route_start") if isinstance(general.get("route_start"), dict) else None
    goal = general.get("route_goal") if isinstance(general.get("route_goal"), dict) else None
    if start:
        ax.scatter([start["x"]], [start["y"]], marker="o", s=60, color="lime", edgecolors="black", label="Start")
    if goal:
        ax.scatter([goal["x"]], [goal["y"]], marker="X", s=70, color="deepskyblue", edgecolors="black", label="Goal")

    ax.set_title("Trajectory Comparison")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    _legend_if_labels(ax, fontsize=8)


def _draw_localization_error_panel(ax, samples: list[dict[str, object]]) -> None:
    _plot_series(ax, samples, "kalman_position_error_m", "Kalman position error")
    _plot_series(ax, samples, "raw_gnss_error_m", "Raw GNSS position error", linestyle=":")
    ax.set_title("Localization Error Over Time")
    ax.set_xlabel("Time since benchmark start (s)")
    ax.set_ylabel("Position error (m)")
    ax.grid(True, alpha=0.3)
    _legend_if_labels(ax, fontsize=8)


def _summary_text(metadata: dict[str, object], summary: dict[str, object]) -> str:
    general = metadata.get("general", {}) if isinstance(metadata.get("general"), dict) else {}
    kf = metadata.get("kalman_filter", {}) if isinstance(metadata.get("kalman_filter"), dict) else {}
    sensors = metadata.get("sensor_configuration", {}) if isinstance(metadata.get("sensor_configuration"), dict) else {}
    gnss = sensors.get("gnss", {}) if isinstance(sensors.get("gnss"), dict) else {}
    imu = sensors.get("imu", {}) if isinstance(sensors.get("imu"), dict) else {}

    lines = [
        "Kalman Benchmark Summary",
        f"Route: {general.get('route_name')}",
        f"Map: {general.get('map_name')}",
        f"Benchmark: {general.get('benchmark_id')}",
        "",
        "Kalman Filter",
        f"Type: {kf.get('filter_type')}",
        f"State: {kf.get('state_vector')}",
        f"Process jerk: {_fmt(_nested(kf, 'process_noise_parameters', 'process_jerk_stddev_mps3'))} m/s^3",
        f"GNSS pos std: {_fmt(_nested(kf, 'measurement_noise_parameters', 'gnss_position_stddev_m'))} m",
        f"IMU accel std: {_fmt(_nested(kf, 'measurement_noise_parameters', 'imu_accel_stddev_mps2'))} m/s^2",
        "",
        "Configured Sensor Noise",
        f"GNSS lat/lon/alt std:",
        f"  {_fmt(gnss.get('noise_lat_stddev_deg'))} / {_fmt(gnss.get('noise_lon_stddev_deg'))} / {_fmt(gnss.get('noise_alt_stddev_m'))}",
        f"IMU accel std:",
        f"  {_fmt(imu.get('noise_accel_stddev_x'))} / {_fmt(imu.get('noise_accel_stddev_y'))} / {_fmt(imu.get('noise_accel_stddev_z'))}",
        f"IMU gyro std:",
        f"  {_fmt(imu.get('noise_gyro_stddev_x'))} / {_fmt(imu.get('noise_gyro_stddev_y'))} / {_fmt(imu.get('noise_gyro_stddev_z'))}",
        "",
        "Localization",
        f"Kalman RMSE: {_fmt(summary.get('kalman_rmse_m'))} m",
        f"Raw GNSS RMSE: {_fmt(summary.get('raw_gnss_rmse_m'))} m",
        f"Improvement ratio: {_fmt(summary.get('kalman_improvement_ratio'))}x",
        "",
        "Route Tracking",
        f"Mean CTE: {_fmt(summary.get('mean_cross_track_error_m'))} m",
        f"Max CTE: {_fmt(summary.get('max_cross_track_error_m'))} m",
        "",
        "Completion",
        f"Success: {summary.get('route_completion_success')}",
        f"Aborted: {summary.get('route_aborted')}",
        f"Timeout: {summary.get('timeout')}",
        f"Completion time: {_fmt(summary.get('completion_time_s'))} s",
    ]
    return "\n".join(lines)


def _read_samples(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return [{key: _convert(value) for key, value in row.items()} for row in reader]


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _route_points(metadata: dict[str, object]) -> list[dict[str, float]]:
    general = metadata.get("general")
    if not isinstance(general, dict):
        return []
    points = general.get("route_points")
    if not isinstance(points, list):
        return []
    result = []
    for point in points:
        if isinstance(point, dict) and isinstance(point.get("x"), (int, float)) and isinstance(point.get("y"), (int, float)):
            result.append({"x": float(point["x"]), "y": float(point["y"])})
    return result


def _plot_series(
    ax,
    samples: list[dict[str, object]],
    field: str,
    label: str,
    linestyle: str = "-",
) -> None:
    times, values = _time_series(samples, field)
    if not times:
        return
    ax.plot(times, values, linestyle=linestyle, linewidth=1.7, label=label)


def _plot_xy(
    ax,
    samples: list[dict[str, object]],
    x_field: str,
    y_field: str,
    label: str,
    color: str,
    linestyle: str = "-",
) -> None:
    xs, ys = _xy_series(samples, x_field, y_field)
    if not xs:
        return
    ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.8, label=label)


def _scatter_xy(
    ax,
    samples: list[dict[str, object]],
    x_field: str,
    y_field: str,
    label: str,
    color: str,
) -> None:
    xs, ys = _xy_series(samples, x_field, y_field)
    if not xs:
        return
    stride = max(1, len(xs) // 500)
    ax.scatter(xs[::stride], ys[::stride], s=5, alpha=0.35, color=color, label=label)


def _time_series(samples: list[dict[str, object]], field: str) -> tuple[list[float], list[float]]:
    base_time: Optional[float] = None
    times: list[float] = []
    values: list[float] = []
    for sample in samples:
        timestamp = sample.get("timestamp")
        value = sample.get(field)
        if not _finite(timestamp) or not _finite(value):
            continue
        timestamp_f = float(timestamp)
        if base_time is None:
            base_time = timestamp_f
        times.append(timestamp_f - base_time)
        values.append(float(value))
    return times, values


def _xy_series(samples: list[dict[str, object]], x_field: str, y_field: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for sample in samples:
        x_value = sample.get(x_field)
        y_value = sample.get(y_field)
        if not _finite(x_value) or not _finite(y_value):
            continue
        xs.append(float(x_value))
        ys.append(float(y_value))
    return xs, ys


def _convert(value: str) -> object:
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _nested(data: dict[str, object], first: str, second: str) -> object:
    nested = data.get(first)
    if not isinstance(nested, dict):
        return None
    return nested.get(second)


def _fmt(value: object) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "n/a"
    value_f = float(value)
    if abs(value_f) >= 1000.0 or (0.0 < abs(value_f) < 0.001):
        return f"{value_f:.2e}"
    return f"{value_f:.3f}"


def _legend_if_labels(ax, fontsize: Optional[int] = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(fontsize=fontsize)
