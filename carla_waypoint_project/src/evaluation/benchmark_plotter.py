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

from config.settings import BENCHMARK

DRIVING_PHASES = ("driving", "completed")
STABILIZATION_PHASES = ("stabilization",)
RouteBounds = tuple[float, float, float, float]


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
    metric_samples = _samples_for_metric_plots(samples)
    _plot_series(ax, metric_samples, "cross_track_error_m", "Cross-track error")
    title_suffix = " (Driving Phase)" if BENCHMARK.metrics_use_driving_phase_only else ""
    ax.set_title(f"Cross-Track Error Over Time{title_suffix}")
    ax.set_xlabel("Time since benchmark start (s)")
    ax.set_ylabel("Cross-track error (m)")
    ax.grid(True, alpha=0.3)
    _warn_if_no_series(ax, metric_samples, "cross_track_error_m", "No cross-track samples to plot")
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

    trajectory_info = _filtered_samples_for_trajectory(metadata, samples)
    _draw_trajectory_panel(trajectory_ax, metadata, samples, trajectory_info=trajectory_info)
    _draw_localization_error_panel(localization_ax, samples)

    metric_samples = _samples_for_metric_plots(samples)
    _plot_series(cross_track_ax, metric_samples, "cross_track_error_m", "Cross-track error")
    title_suffix = " (Driving Phase)" if BENCHMARK.metrics_use_driving_phase_only else " (All Phases)"
    cross_track_ax.set_title(f"Cross-Track Error{title_suffix}")
    cross_track_ax.set_xlabel("Time since benchmark start (s)")
    cross_track_ax.set_ylabel("m")
    cross_track_ax.grid(True, alpha=0.3)
    _warn_if_no_series(cross_track_ax, metric_samples, "cross_track_error_m", "No cross-track samples to plot")
    _legend_if_labels(cross_track_ax, fontsize=8)

    text_ax.axis("off")
    text_ax.text(
        0.0,
        1.0,
        _summary_text(metadata, summary, trajectory_info),
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.3,
        linespacing=1.22,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _draw_trajectory_panel(
    ax,
    metadata: dict[str, object],
    samples: list[dict[str, object]],
    trajectory_info: Optional[dict[str, object]] = None,
) -> None:
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

    info = trajectory_info or _filtered_samples_for_trajectory(metadata, samples)
    ground_truth_samples = _sample_list(info.get("ground_truth"))
    kalman_samples = _sample_list(info.get("kalman"))
    gnss_samples = _sample_list(info.get("gnss"))

    _plot_xy(ax, ground_truth_samples, "ground_truth_x", "ground_truth_y", "Ground truth trajectory", color="tab:green")
    _plot_xy(ax, kalman_samples, "kalman_x", "kalman_y", "Kalman estimated trajectory", color="tab:blue", linestyle=":")
    _scatter_xy(ax, gnss_samples, "gnss_x", "gnss_y", "Raw noisy GNSS", color="tab:orange")

    if not kalman_samples:
        ax.text(
            0.02,
            0.03,
            "No valid Kalman trajectory samples after filtering",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            color="darkred",
            bbox={"facecolor": "white", "edgecolor": "darkred", "alpha": 0.85},
        )

    general = metadata.get("general", {}) if isinstance(metadata.get("general"), dict) else {}
    start = general.get("route_start") if isinstance(general.get("route_start"), dict) else None
    goal = general.get("route_goal") if isinstance(general.get("route_goal"), dict) else None
    if start and _finite(start.get("x")) and _finite(start.get("y")):
        ax.scatter([start["x"]], [start["y"]], marker="o", s=60, color="lime", edgecolors="black", label="Start")
    if goal and _finite(goal.get("x")) and _finite(goal.get("y")):
        ax.scatter([goal["x"]], [goal["y"]], marker="X", s=70, color="deepskyblue", edgecolors="black", label="Goal")

    bounds = info.get("route_bounds")
    if isinstance(bounds, tuple) and len(bounds) == 4:
        min_x, max_x, min_y, max_y = bounds
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    ax.set_title("Trajectory Comparison")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    _legend_if_labels(ax, fontsize=8)


def _draw_localization_error_panel(ax, samples: list[dict[str, object]]) -> None:
    driving_samples = _filter_by_phase(samples, DRIVING_PHASES)
    if BENCHMARK.metrics_use_driving_phase_only:
        base_samples = driving_samples
        if BENCHMARK.collect_stabilization_samples:
            stabilization_samples = _filter_by_phase(samples, STABILIZATION_PHASES)
            base_time = _base_timestamp(samples)
            _plot_series(
                ax,
                stabilization_samples,
                "kalman_position_error_m",
                "Kalman position error (stabilization)",
                linestyle="--",
                alpha=0.22,
                base_time=base_time,
            )
            _plot_series(
                ax,
                stabilization_samples,
                "raw_gnss_error_m",
                "Raw GNSS position error (stabilization)",
                linestyle=":",
                alpha=0.18,
                base_time=base_time,
            )
        else:
            base_time = _base_timestamp(base_samples)

        _plot_series(
            ax,
            driving_samples,
            "kalman_position_error_m",
            "Kalman position error (driving)",
            base_time=base_time,
        )
        _plot_series(
            ax,
            driving_samples,
            "raw_gnss_error_m",
            "Raw GNSS position error (driving)",
            linestyle=":",
            base_time=base_time,
        )
        title_suffix = " (Driving Phase)"
        warning_samples = driving_samples
    else:
        _plot_series(ax, samples, "kalman_position_error_m", "Kalman position error")
        _plot_series(ax, samples, "raw_gnss_error_m", "Raw GNSS position error", linestyle=":")
        title_suffix = " (All Phases)"
        warning_samples = samples

    ax.set_title(f"Localization Error Over Time{title_suffix}")
    ax.set_xlabel("Time since benchmark start (s)")
    ax.set_ylabel("Position error (m)")
    ax.grid(True, alpha=0.3)
    _warn_if_no_series(ax, warning_samples, "kalman_position_error_m", "No Kalman error samples to plot")
    _legend_if_labels(ax, fontsize=8)


def _summary_text(
    metadata: dict[str, object],
    summary: dict[str, object],
    trajectory_info: Optional[dict[str, object]] = None,
) -> str:
    general = metadata.get("general", {}) if isinstance(metadata.get("general"), dict) else {}
    kf = metadata.get("kalman_filter", {}) if isinstance(metadata.get("kalman_filter"), dict) else {}
    sensors = metadata.get("sensor_configuration", {}) if isinstance(metadata.get("sensor_configuration"), dict) else {}
    gnss = sensors.get("gnss", {}) if isinstance(sensors.get("gnss"), dict) else {}
    imu = sensors.get("imu", {}) if isinstance(sensors.get("imu"), dict) else {}
    no_valid_kalman = bool(trajectory_info and trajectory_info.get("no_valid_kalman"))
    excluded_plot_samples = summary.get("excluded_kalman_plot_sample_count")
    if trajectory_info and trajectory_info.get("excluded_kalman_plot_sample_count") is not None:
        excluded_plot_samples = trajectory_info.get("excluded_kalman_plot_sample_count")

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
        "GNSS lat/lon/alt std:",
        f"  {_fmt(gnss.get('noise_lat_stddev_deg'))} / {_fmt(gnss.get('noise_lon_stddev_deg'))} / {_fmt(gnss.get('noise_alt_stddev_m'))}",
        "IMU accel std:",
        f"  {_fmt(imu.get('noise_accel_stddev_x'))} / {_fmt(imu.get('noise_accel_stddev_y'))} / {_fmt(imu.get('noise_accel_stddev_z'))}",
        "IMU gyro std:",
        f"  {_fmt(imu.get('noise_gyro_stddev_x'))} / {_fmt(imu.get('noise_gyro_stddev_y'))} / {_fmt(imu.get('noise_gyro_stddev_z'))}",
        "",
        "Localization (Driving Phase)",
        f"Kalman RMSE: {_fmt(_primary(summary, 'driving_kalman_rmse_m', 'kalman_rmse_m'))} m",
        f"Raw GNSS RMSE: {_fmt(_primary(summary, 'driving_raw_gnss_rmse_m', 'raw_gnss_rmse_m'))} m",
        f"Improvement ratio: {_fmt(_primary(summary, 'driving_kalman_improvement_ratio', 'kalman_improvement_ratio'))}x",
        f"Driving samples: {summary.get('driving_sample_count')} / {summary.get('sample_count')}",
        f"Kalman plot samples excluded: {excluded_plot_samples}",
        "",
        "Route Tracking (Driving Phase)",
        f"Mean CTE: {_fmt(_primary(summary, 'driving_mean_cross_track_error_m', 'mean_cross_track_error_m'))} m",
        f"Max CTE: {_fmt(_primary(summary, 'driving_max_cross_track_error_m', 'max_cross_track_error_m'))} m",
        "",
        "Completion",
        f"Success: {summary.get('route_completion_success')}",
        f"Aborted: {summary.get('route_aborted')}",
        f"Timeout: {summary.get('timeout')}",
        f"Completion time: {_fmt(summary.get('completion_time_s'))} s",
        "",
        "Note:",
        "Raw GNSS is evaluated as a",
        "localization baseline only.",
        "Vehicle control uses the",
        "Kalman estimate.",
    ]
    if no_valid_kalman:
        lines.extend(["", "Warning:", "No valid Kalman trajectory", "samples after filtering."])
    return "\n".join(lines)


def _filtered_samples_for_trajectory(
    metadata: dict[str, object],
    samples: list[dict[str, object]],
) -> dict[str, object]:
    bounds = _route_bounds_from_metadata(metadata)
    ground_truth_samples = _filter_xy_by_phase(
        samples,
        "ground_truth_x",
        "ground_truth_y",
        phases=DRIVING_PHASES,
        bounds=bounds,
    )
    gnss_samples = _filter_xy_by_phase(
        samples,
        "gnss_x",
        "gnss_y",
        phases=DRIVING_PHASES,
        bounds=bounds,
    )

    kalman_xy_candidates = _filter_xy_by_phase(
        samples,
        "kalman_x",
        "kalman_y",
        phases=DRIVING_PHASES,
        bounds=None,
    )
    kalman_error_candidates = []
    for sample in kalman_xy_candidates:
        if not _finite(sample.get("ground_truth_x")) or not _finite(sample.get("ground_truth_y")):
            continue
        if not _finite(sample.get("kalman_position_error_m")):
            continue
        if float(sample["kalman_position_error_m"]) > BENCHMARK.max_kalman_plot_error_m:
            continue
        if not _inside_route_bounds(sample.get("kalman_x"), sample.get("kalman_y"), bounds):
            continue
        if not _inside_route_bounds(sample.get("ground_truth_x"), sample.get("ground_truth_y"), bounds):
            continue
        kalman_error_candidates.append(sample)

    kalman_samples = _remove_unrealistic_jumps(
        kalman_error_candidates,
        "kalman_x",
        "kalman_y",
        max_jump_m=BENCHMARK.max_trajectory_jump_m,
    )
    excluded_count = max(0, len(kalman_xy_candidates) - len(kalman_samples))
    return {
        "ground_truth": ground_truth_samples,
        "kalman": kalman_samples,
        "gnss": gnss_samples,
        "route_bounds": bounds,
        "excluded_kalman_plot_sample_count": excluded_count,
        "no_valid_kalman": not kalman_samples,
    }


def _filter_xy_by_phase(
    samples: list[dict[str, object]],
    x_field: str,
    y_field: str,
    phases: tuple[str, ...] = DRIVING_PHASES,
    bounds: Optional[RouteBounds] = None,
) -> list[dict[str, object]]:
    result = []
    for sample in samples:
        if sample.get("phase") not in phases:
            continue
        x_value = sample.get(x_field)
        y_value = sample.get(y_field)
        if not _finite(x_value) or not _finite(y_value):
            continue
        if not _inside_route_bounds(x_value, y_value, bounds):
            continue
        result.append(sample)
    return result


def _remove_unrealistic_jumps(
    samples: list[dict[str, object]],
    x_field: str,
    y_field: str,
    max_jump_m: float = BENCHMARK.max_trajectory_jump_m,
) -> list[dict[str, object]]:
    if max_jump_m <= 0.0:
        return list(samples)

    filtered = []
    previous_xy: Optional[tuple[float, float]] = None
    for sample in samples:
        x_value = sample.get(x_field)
        y_value = sample.get(y_field)
        if not _finite(x_value) or not _finite(y_value):
            continue
        current_xy = (float(x_value), float(y_value))
        if previous_xy is not None:
            jump_m = math.hypot(current_xy[0] - previous_xy[0], current_xy[1] - previous_xy[1])
            if jump_m > max_jump_m:
                continue
        filtered.append(sample)
        previous_xy = current_xy
    return filtered


def _route_bounds_from_metadata(metadata: dict[str, object]) -> Optional[RouteBounds]:
    points = _route_points(metadata)
    general = metadata.get("general")
    if isinstance(general, dict):
        for key in ("route_start", "route_goal"):
            point = general.get(key)
            if isinstance(point, dict) and _finite(point.get("x")) and _finite(point.get("y")):
                points.append({"x": float(point["x"]), "y": float(point["y"])})

    if not points:
        return None

    margin = max(0.0, float(BENCHMARK.route_bounds_margin_m))
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


def _inside_route_bounds(
    x_value: object,
    y_value: object,
    bounds: Optional[RouteBounds],
) -> bool:
    if bounds is None:
        return True
    if not _finite(x_value) or not _finite(y_value):
        return False
    min_x, max_x, min_y, max_y = bounds
    x_float = float(x_value)
    y_float = float(y_value)
    return min_x <= x_float <= max_x and min_y <= y_float <= max_y


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
        if (
            isinstance(point, dict)
            and isinstance(point.get("x"), (int, float))
            and isinstance(point.get("y"), (int, float))
            and math.isfinite(float(point["x"]))
            and math.isfinite(float(point["y"]))
        ):
            result.append({"x": float(point["x"]), "y": float(point["y"])})
    return result


def _samples_for_metric_plots(samples: list[dict[str, object]]) -> list[dict[str, object]]:
    if BENCHMARK.metrics_use_driving_phase_only:
        return _filter_by_phase(samples, DRIVING_PHASES)
    return samples


def _filter_by_phase(samples: list[dict[str, object]], phases: tuple[str, ...]) -> list[dict[str, object]]:
    return [sample for sample in samples if sample.get("phase") in phases]


def _plot_series(
    ax,
    samples: list[dict[str, object]],
    field: str,
    label: str,
    linestyle: str = "-",
    alpha: float = 1.0,
    base_time: Optional[float] = None,
) -> None:
    times, values = _time_series(samples, field, base_time=base_time)
    if not times:
        return
    ax.plot(times, values, linestyle=linestyle, linewidth=1.7, alpha=alpha, label=label)


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


def _time_series(
    samples: list[dict[str, object]],
    field: str,
    base_time: Optional[float] = None,
) -> tuple[list[float], list[float]]:
    if base_time is None:
        base_time = _base_timestamp(samples)
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


def _base_timestamp(samples: list[dict[str, object]]) -> Optional[float]:
    for sample in samples:
        timestamp = sample.get("timestamp")
        if _finite(timestamp):
            return float(timestamp)
    return None


def _warn_if_no_series(ax, samples: list[dict[str, object]], field: str, message: str) -> None:
    if any(_finite(sample.get(field)) for sample in samples):
        return
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="darkred",
    )


def _sample_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [sample for sample in value if isinstance(sample, dict)]


def _primary(summary: dict[str, object], primary_key: str, fallback_key: str) -> object:
    value = summary.get(primary_key)
    return value if value is not None else summary.get(fallback_key)


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
