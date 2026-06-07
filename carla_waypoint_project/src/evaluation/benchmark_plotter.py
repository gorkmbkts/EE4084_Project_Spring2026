"""Matplotlib plot generation for single-run localization filter benchmarks."""

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
LOCALIZATION_ERROR_FIELDS = ("filtered_position_error_m", "raw_gnss_error_m")
RouteBounds = tuple[float, float, float, float]


def generate_benchmark_plots(benchmark_folder: Path) -> list[Path]:
    """Generate benchmark plots from route benchmark CSV/JSON files."""
    benchmark_folder = Path(benchmark_folder)
    plots_dir = benchmark_folder / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metadata = _read_json(benchmark_folder / "metadata.json")
    summary = _read_json(_first_existing(benchmark_folder / "route_summary.json", benchmark_folder / "summary.json"))
    samples = _read_samples(_first_existing(benchmark_folder / "timeseries.csv", benchmark_folder / "samples.csv"))

    outputs = [
        _plot_trajectory(plots_dir / "trajectory_comparison.png", metadata, samples),
        _plot_position_error(plots_dir / "position_error_over_time.png", metadata, samples),
        _plot_raw_vs_filtered_error(plots_dir / "raw_gnss_vs_filtered_position_error.png", metadata, samples),
        _plot_speed_comparison(plots_dir / "ground_truth_vs_estimated_speed.png", metadata, samples),
        _plot_yaw_comparison(plots_dir / "ground_truth_vs_estimated_yaw.png", metadata, samples),
        _plot_metric_over_time(plots_dir / "nis_over_time.png", metadata, samples, "nis", "NIS"),
        _plot_position_nees_over_time(plots_dir / "nees_over_time.png", metadata, samples),
        _plot_metric_over_time(plots_dir / "legacy_nees_over_time.png", metadata, samples, "nees", "Legacy NEES"),
        _plot_sigma_bounds(plots_dir / "estimation_error_2sigma_bounds.png", metadata, samples),
        _plot_segment_rmse(plots_dir / "segment_rmse_bar_chart.png", metadata, summary),
        _plot_curvature_vs_error(plots_dir / "curvature_severity_vs_position_error.png", metadata, samples),
        _plot_localization_error(plots_dir / "localization_error_over_time.png", samples),
        _plot_cross_track_error(plots_dir / "cross_track_error_over_time.png", samples),
        _plot_summary_dashboard(plots_dir / "summary_dashboard.png", metadata, summary, samples),
    ]
    return outputs


def generate_aggregate_benchmark_plots(run_folder: Path) -> list[Path]:
    """Generate aggregate plots for a multi-route automated benchmark run."""
    run_folder = Path(run_folder)
    plots_dir = run_folder / "aggregate_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv_dicts(run_folder / "aggregate_summary.csv")
    outputs = [
        _plot_aggregate_bar(plots_dir / "position_rmse_per_route.png", rows, "filtered_rmse_m", "Position RMSE (m)"),
        _plot_aggregate_raw_vs_filtered(plots_dir / "raw_gnss_rmse_vs_filtered_rmse.png", rows),
        _plot_aggregate_bar(plots_dir / "improvement_percentage_per_route.png", rows, "improvement_percent", "Improvement (%)"),
        _plot_aggregate_dual_metric(
            plots_dir / "mean_nees_nis_per_route.png",
            rows,
            "mean_position_nees",
            "legacy_mean_nis_mixed",
            "Position NEES",
            "Legacy Mixed NIS",
        ),
        _plot_aggregate_segments(plots_dir / "segment_rmse_summary.png", run_folder),
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


def _plot_position_error(output_path: Path, metadata: dict[str, object], samples: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_samples = _samples_for_metric_plots(samples)
    _plot_series(ax, metric_samples, "filtered_position_error_m", "Filtered position error")
    ax.set_title(_title_with_context("Position Error Over Time", metadata))
    ax.set_xlabel("Time since benchmark start (s)")
    ax.set_ylabel("Position error (m)")
    ax.grid(True, alpha=0.3)
    _warn_if_no_series(ax, metric_samples, "filtered_position_error_m", "No filtered position error samples")
    _legend_if_labels(ax, fontsize=8)
    _add_metadata_box(ax, metadata)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_raw_vs_filtered_error(output_path: Path, metadata: dict[str, object], samples: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_localization_error_panel(ax, samples)
    ax.set_title(_title_with_context("Raw GNSS vs Filtered Position Error", metadata))
    _add_metadata_box(ax, metadata)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_speed_comparison(output_path: Path, metadata: dict[str, object], samples: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_samples = _samples_for_metric_plots(samples)
    _plot_series(ax, metric_samples, "ground_truth_speed", "Ground truth speed")
    _plot_series(ax, metric_samples, "filtered_speed", "Estimated speed", linestyle=":")
    ax.set_title(_title_with_context("Ground Truth Speed vs Estimated Speed", metadata))
    ax.set_xlabel("Time since benchmark start (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, alpha=0.3)
    _warn_if_no_series(ax, metric_samples, "filtered_speed", "No estimated speed samples")
    _legend_if_labels(ax, fontsize=8)
    _add_metadata_box(ax, metadata)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_yaw_comparison(output_path: Path, metadata: dict[str, object], samples: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_samples = _samples_for_metric_plots(samples)
    _plot_series(ax, metric_samples, "ground_truth_yaw", "Ground truth yaw")
    _plot_series(ax, metric_samples, "filtered_yaw", "Estimated yaw", linestyle=":")
    ax.set_title(_title_with_context("Ground Truth Yaw vs Estimated Yaw", metadata))
    ax.set_xlabel("Time since benchmark start (s)")
    ax.set_ylabel("Yaw (deg)")
    ax.grid(True, alpha=0.3)
    _warn_if_no_series(ax, metric_samples, "filtered_yaw", "No estimated yaw samples")
    _legend_if_labels(ax, fontsize=8)
    _add_metadata_box(ax, metadata)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_metric_over_time(
    output_path: Path,
    metadata: dict[str, object],
    samples: list[dict[str, object]],
    field: str,
    label: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_samples = _samples_for_metric_plots(samples)
    _plot_series(ax, metric_samples, field, label)
    ax.set_title(_title_with_context(f"{label} Over Time", metadata))
    ax.set_xlabel("Time since benchmark start (s)")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    _warn_if_no_series(ax, metric_samples, field, f"No {label} samples available")
    _legend_if_labels(ax, fontsize=8)
    _add_metadata_box(ax, metadata)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_position_nees_over_time(output_path: Path, metadata: dict[str, object], samples: list[dict[str, object]]) -> Path:
    if _has_metric_values(samples, "position_nees"):
        return _plot_metric_over_time(output_path, metadata, samples, "position_nees", "Position NEES (full 2x2)")
    if _has_metric_values(samples, "position_nees_diagonal_approx"):
        return _plot_metric_over_time(output_path, metadata, samples, "position_nees_diagonal_approx", "Position NEES (diagonal approx)")
    return _plot_metric_over_time(output_path, metadata, samples, "nees", "Legacy NEES")


def _has_metric_values(samples: list[dict[str, object]], field: str) -> bool:
    return any(_to_float(sample.get(field)) is not None for sample in samples)


def _plot_sigma_bounds(output_path: Path, metadata: dict[str, object], samples: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_samples = _samples_for_metric_plots(samples)
    times, x_errors = _time_series(metric_samples, "x_error_m")
    _times_y, y_errors = _time_series(metric_samples, "y_error_m", base_time=_base_timestamp(metric_samples))
    _times_sx, sigma_x = _time_series(metric_samples, "covariance_x_std_m", base_time=_base_timestamp(metric_samples))
    _times_sy, sigma_y = _time_series(metric_samples, "covariance_y_std_m", base_time=_base_timestamp(metric_samples))
    if times and x_errors:
        ax.plot(times, x_errors, label="X error")
    if _times_y and y_errors:
        ax.plot(_times_y, y_errors, label="Y error")
    if _times_sx and sigma_x:
        ax.plot(_times_sx, [2.0 * value for value in sigma_x], linestyle="--", color="tab:blue", label="+2sigma X")
        ax.plot(_times_sx, [-2.0 * value for value in sigma_x], linestyle="--", color="tab:blue", alpha=0.55, label="-2sigma X")
    if _times_sy and sigma_y:
        ax.plot(_times_sy, [2.0 * value for value in sigma_y], linestyle="--", color="tab:orange", label="+2sigma Y")
        ax.plot(_times_sy, [-2.0 * value for value in sigma_y], linestyle="--", color="tab:orange", alpha=0.55, label="-2sigma Y")
    ax.set_title(_title_with_context("Estimation Error with +/-2sigma Bounds", metadata))
    ax.set_xlabel("Time since benchmark start (s)")
    ax.set_ylabel("Error (m)")
    ax.grid(True, alpha=0.3)
    if not times and not _times_y:
        ax.text(0.5, 0.5, "No covariance/error samples available", transform=ax.transAxes, ha="center", va="center")
    _legend_if_labels(ax, fontsize=8)
    _add_metadata_box(ax, metadata)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_segment_rmse(output_path: Path, metadata: dict[str, object], summary: dict[str, object]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    segments = summary.get("driving_segment_metrics") or summary.get("segment_metrics")
    if isinstance(segments, dict) and segments:
        labels = list(segments.keys())
        values = [
            _to_float(segments[label].get("position_rmse_m")) if isinstance(segments[label], dict) else None
            for label in labels
        ]
        ax.bar(labels, [value if value is not None else 0.0 for value in values], color="tab:blue")
        ax.set_ylabel("Position RMSE (m)")
    else:
        ax.text(0.5, 0.5, "No segment metrics available", transform=ax.transAxes, ha="center", va="center")
    ax.set_title(_title_with_context("Segment-Based Position RMSE", metadata))
    ax.grid(True, axis="y", alpha=0.3)
    _add_metadata_box(ax, metadata)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_curvature_vs_error(output_path: Path, metadata: dict[str, object], samples: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_samples = _samples_for_metric_plots(samples)
    xs = []
    ys = []
    for sample in metric_samples:
        curvature = sample.get("curvature_score")
        error = sample.get("filtered_position_error_m")
        if _finite(curvature) and _finite(error):
            xs.append(float(curvature))
            ys.append(float(error))
    if xs:
        ax.scatter(xs, ys, s=8, alpha=0.45, color="tab:purple")
    else:
        ax.text(0.5, 0.5, "No curvature/error samples available", transform=ax.transAxes, ha="center", va="center")
    ax.set_title(_title_with_context("Curvature Severity vs Position Error", metadata))
    ax.set_xlabel("Curvature severity score")
    ax.set_ylabel("Position error (m)")
    ax.grid(True, alpha=0.3)
    _add_metadata_box(ax, metadata)
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
    filtered_samples = _sample_list(info.get("filtered"))
    gnss_samples = _sample_list(info.get("gnss"))
    active_filter_name = _active_filter_name(metadata)

    _plot_xy(ax, ground_truth_samples, "ground_truth_x", "ground_truth_y", "Ground truth trajectory", color="tab:green")
    _plot_xy(
        ax,
        filtered_samples,
        "filtered_x",
        "filtered_y",
        f"{active_filter_name} estimated trajectory",
        color="tab:blue",
        linestyle=":",
    )
    _scatter_xy(ax, gnss_samples, "gnss_x", "gnss_y", "Raw GNSS measurements", color="tab:orange")

    if not filtered_samples:
        ax.text(
            0.02,
            0.03,
            "No valid filtered trajectory samples after filtering",
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
    plotted_samples = samples
    if BENCHMARK.metrics_use_driving_phase_only:
        base_samples = driving_samples
        stabilization_samples: list[dict[str, object]] = []
        if BENCHMARK.collect_stabilization_samples:
            stabilization_samples = _filter_by_phase(samples, STABILIZATION_PHASES)
            base_time = _base_timestamp(samples)
            _plot_series(
                ax,
                stabilization_samples,
                "filtered_position_error_m",
                "Filtered position error (stabilization)",
                linestyle="--",
                alpha=0.22,
                base_time=base_time,
            )
            _plot_series(
                ax,
                stabilization_samples,
                "raw_gnss_error_m",
                "Raw GNSS baseline error (stabilization)",
                linestyle=":",
                alpha=0.18,
                base_time=base_time,
            )
        else:
            base_time = _base_timestamp(base_samples)

        _plot_series(
            ax,
            driving_samples,
            "filtered_position_error_m",
            "Filtered position error (driving)",
            base_time=base_time,
        )
        _plot_series(
            ax,
            driving_samples,
            "raw_gnss_error_m",
            "Raw GNSS baseline error (driving)",
            linestyle=":",
            base_time=base_time,
        )
        title_suffix = " (Driving Phase)"
        warning_samples = driving_samples
        plotted_samples = stabilization_samples + driving_samples
    else:
        _plot_series(ax, samples, "filtered_position_error_m", "Filtered position error")
        _plot_series(ax, samples, "raw_gnss_error_m", "Raw GNSS baseline error", linestyle=":")
        title_suffix = " (All Phases)"
        warning_samples = samples

    robust_ymax = _robust_error_ymax(driving_samples, plotted_samples)
    if robust_ymax is not None:
        ax.set_ylim(bottom=0.0, top=robust_ymax)
        _annotate_clipped_error_samples(ax, plotted_samples, robust_ymax)

    ax.set_title(f"Localization Error Over Time{title_suffix}")
    ax.set_xlabel("Time since benchmark start (s)")
    ax.set_ylabel("Position error (m)")
    ax.grid(True, alpha=0.3)
    _warn_if_no_series(ax, warning_samples, "filtered_position_error_m", "No filtered error samples to plot")
    _legend_if_labels(ax, fontsize=8)


def _summary_text(
    metadata: dict[str, object],
    summary: dict[str, object],
    trajectory_info: Optional[dict[str, object]] = None,
) -> str:
    general = metadata.get("general", {}) if isinstance(metadata.get("general"), dict) else {}
    active_filter = _active_filter_info(metadata)
    sensors = metadata.get("sensor_configuration", {}) if isinstance(metadata.get("sensor_configuration"), dict) else {}
    gnss = sensors.get("gnss", {}) if isinstance(sensors.get("gnss"), dict) else {}
    imu = sensors.get("imu", {}) if isinstance(sensors.get("imu"), dict) else {}
    no_valid_filtered = bool(trajectory_info and trajectory_info.get("no_valid_filtered"))
    excluded_plot_samples = _primary(summary, "excluded_filtered_plot_sample_count", "excluded_kalman_plot_sample_count")
    if trajectory_info and trajectory_info.get("excluded_filtered_plot_sample_count") is not None:
        excluded_plot_samples = trajectory_info.get("excluded_filtered_plot_sample_count")
    tune = active_filter.get("tune") if isinstance(active_filter.get("tune"), dict) else {}
    tune_lines = [f"{key}: {_fmt(value)}" for key, value in list(tune.items())[:6]]
    tracking_mode = _tracking_mode_label(metadata, active_filter, summary)
    active_control_used = _first(
        summary,
        "active_control_input_used_by_filter",
        "active_control_input_used",
    )
    if active_control_used is None:
        active_control_used = _first(
            metadata,
            "active_control_input_used_by_filter",
            "active_control_input_used",
        )
    if active_control_used is None:
        active_control_used = active_filter.get("active_control_input_used")
    driving_legacy_nis = _first(summary, "driving_legacy_mean_nis_mixed", "legacy_mean_nis_mixed", "driving_mean_nis", "mean_nis")
    driving_position_nees = _first(summary, "driving_mean_position_nees", "mean_position_nees")
    driving_position_nees_label = "Driving Position NEES"
    if driving_position_nees is None:
        driving_position_nees = _first(summary, "driving_mean_position_nees_diagonal_approx", "mean_position_nees_diagonal_approx")
        driving_position_nees_label = "Driving Position NEES (diag approx)"
    driving_legacy_nees = _primary(summary, "driving_mean_nees", "mean_nees")
    position_nees_source = _first(summary, "driving_position_nees_source", "position_nees_source") or "unavailable"
    nis_by_type_line = _nis_by_type_line(_first(summary, "driving_nis_by_type_summary", "nis_by_type_summary"))
    consistency_warning = _consistency_warning(driving_legacy_nis, driving_position_nees)

    lines = [
        "KalmanLab Benchmark Summary",
        f"Route: {general.get('route_name')}",
        f"Map: {general.get('map_name')}",
        f"Benchmark: {general.get('benchmark_id')}",
        "",
        "Selected Filter",
        f"Name: {active_filter.get('name')}",
        f"Tracking mode: {tracking_mode}",
        f"Active control input used: {_fmt_bool(active_control_used)}",
        f"Type: {active_filter.get('type')}",
        f"State: {active_filter.get('state_vector')}",
        f"Process: {active_filter.get('process_model')}",
        f"Measurement: {active_filter.get('measurement_model')}",
        "Tune:",
        *(tune_lines or ["n/a"]),
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
        f"Filtered RMSE: {_fmt(_first(summary, 'driving_filtered_rmse_m', 'filtered_rmse_m', 'driving_kalman_rmse_m', 'kalman_rmse_m'))} m",
        f"Raw GNSS RMSE: {_fmt(_first(summary, 'driving_raw_gnss_rmse_m', 'raw_gnss_rmse_m'))} m",
        f"Improvement ratio: {_fmt(_first(summary, 'driving_filtered_improvement_ratio', 'filtered_improvement_ratio', 'driving_kalman_improvement_ratio', 'kalman_improvement_ratio'))}x",
        f"Driving samples: {summary.get('driving_sample_count')} / {summary.get('sample_count')}",
        f"Filtered plot samples excluded: {excluded_plot_samples}",
        "",
        "Route Tracking (Driving Phase)",
        f"Mean CTE: {_fmt(_primary(summary, 'driving_mean_cross_track_error_m', 'mean_cross_track_error_m'))} m",
        f"Max CTE: {_fmt(_primary(summary, 'driving_max_cross_track_error_m', 'max_cross_track_error_m'))} m",
        "",
        "Consistency (Driving Phase)",
        f"NIS by type: {nis_by_type_line}",
        f"Legacy mixed NIS: {_fmt(driving_legacy_nis)}",
        f"{driving_position_nees_label}: {_fmt(driving_position_nees)}",
        f"Position NEES source: {position_nees_source}",
        f"Legacy Mean NEES: {_fmt(driving_legacy_nees)}",
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
        "selected filter estimate.",
    ]
    if no_valid_filtered:
        lines.extend(["", "Warning:", "No valid filtered trajectory", "samples after filtering."])
    if consistency_warning:
        lines.extend(["", "Warning:", "High NIS/NEES suggests", "covariance/noise tuning", "inconsistency."])
    diagnostic_notes = summary.get("diagnostic_notes")
    if isinstance(diagnostic_notes, list) and diagnostic_notes:
        lines.extend(["", "Diagnostic note:", str(diagnostic_notes[0])[:38]])
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

    filtered_xy_candidates = _filter_xy_by_phase(
        samples,
        "filtered_x",
        "filtered_y",
        phases=DRIVING_PHASES,
        bounds=None,
    )
    filtered_error_candidates = []
    for sample in filtered_xy_candidates:
        if not _finite(sample.get("ground_truth_x")) or not _finite(sample.get("ground_truth_y")):
            continue
        if not _finite(sample.get("filtered_position_error_m")):
            continue
        if float(sample["filtered_position_error_m"]) > BENCHMARK.max_kalman_plot_error_m:
            continue
        if not _inside_route_bounds(sample.get("filtered_x"), sample.get("filtered_y"), bounds):
            continue
        if not _inside_route_bounds(sample.get("ground_truth_x"), sample.get("ground_truth_y"), bounds):
            continue
        filtered_error_candidates.append(sample)

    filtered_samples = _remove_unrealistic_jumps(
        filtered_error_candidates,
        "filtered_x",
        "filtered_y",
        max_jump_m=BENCHMARK.max_trajectory_jump_m,
    )
    excluded_count = max(0, len(filtered_xy_candidates) - len(filtered_samples))
    return {
        "ground_truth": ground_truth_samples,
        "filtered": filtered_samples,
        "kalman": filtered_samples,
        "gnss": gnss_samples,
        "route_bounds": bounds,
        "excluded_filtered_plot_sample_count": excluded_count,
        "excluded_kalman_plot_sample_count": excluded_count,
        "no_valid_filtered": not filtered_samples,
        "no_valid_kalman": not filtered_samples,
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
        return [_normalize_sample_fields({key: _convert(value) for key, value in row.items()}) for row in reader]


def _read_csv_dicts(path: Path) -> list[dict[str, object]]:
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


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


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


def _finite_error_values(samples: list[dict[str, object]], fields: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for sample in samples:
        for field in fields:
            value = sample.get(field)
            if _finite(value):
                values.append(float(value))
    return values


def _percentile_linear_or_existing(values: list[float], percentile: float) -> Optional[float]:
    finite_values = sorted(value for value in values if math.isfinite(value))
    if not finite_values:
        return None
    if len(finite_values) == 1:
        return finite_values[0]
    percentile = min(100.0, max(0.0, float(percentile)))
    position = (len(finite_values) - 1) * percentile / 100.0
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return finite_values[lower_index]
    fraction = position - lower_index
    lower_value = finite_values[lower_index]
    upper_value = finite_values[upper_index]
    return lower_value + (upper_value - lower_value) * fraction


def _nice_ceiling(value: float) -> Optional[float]:
    if not math.isfinite(value) or value <= 0.0:
        return None
    magnitude = 10.0 ** math.floor(math.log10(value))
    normalized = value / magnitude
    for step in (1.0, 1.5, 2.0, 2.5, 5.0, 7.5, 10.0):
        if normalized <= step:
            return step * magnitude
    return 10.0 * magnitude


def _robust_error_ymax(
    driving_samples: list[dict[str, object]],
    fallback_samples: list[dict[str, object]],
) -> Optional[float]:
    values = _finite_error_values(driving_samples, LOCALIZATION_ERROR_FIELDS)
    if not values:
        values = _finite_error_values(fallback_samples, LOCALIZATION_ERROR_FIELDS)
    if not values:
        return None

    if len(values) >= 20:
        p95 = _percentile_linear_or_existing(values, 95.0)
        p99 = _percentile_linear_or_existing(values, 99.0)
        if p95 is not None and p99 is not None:
            ymax = max(p95 * 1.35, p99 * 1.15)
        else:
            ymax = max(values) * 1.10
    else:
        ymax = max(values) * 1.10
    if ymax <= 0.0:
        ymax = 1.0
    return _nice_ceiling(ymax)


def _annotate_clipped_error_samples(ax, samples: list[dict[str, object]], ymax: float) -> None:
    values = _finite_error_values(samples, LOCALIZATION_ERROR_FIELDS)
    if not values or not math.isfinite(ymax):
        return
    clipped = [value for value in values if value > ymax]
    if not clipped:
        return
    message = f"Outliers clipped visually: {len(clipped)} samples; max error {max(clipped):.2f} m"
    if len(clipped) / len(values) > 0.10:
        message += "\nMany samples exceed visible range; filter may be unstable."
    ax.text(
        0.02,
        0.96,
        message,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="darkred",
        bbox={"facecolor": "white", "edgecolor": "darkred", "alpha": 0.85},
    )


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


def _first(data: dict[str, object], *keys: str) -> object:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _nis_by_type_line(value: object) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = {}
        value = parsed
    if not isinstance(value, dict) or not value:
        return "n/a"
    parts = []
    for update_type, stats in list(sorted(value.items()))[:3]:
        if not isinstance(stats, dict):
            continue
        mean_value = _fmt(stats.get("mean"))
        count = stats.get("sample_count")
        parts.append(f"{update_type} mean {mean_value} n={count}")
    return "; ".join(parts) if parts else "n/a"


def _tracking_mode_label(
    metadata: dict[str, object],
    active_filter: dict[str, object],
    summary: Optional[dict[str, object]] = None,
) -> str:
    general = metadata.get("general") if isinstance(metadata.get("general"), dict) else {}
    kalman_filter = metadata.get("kalman_filter") if isinstance(metadata.get("kalman_filter"), dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    value = (
        summary.get("tracking_mode")
        or metadata.get("tracking_mode")
        or active_filter.get("tracking_mode")
        or general.get("tracking_mode")
        or kalman_filter.get("tracking_mode")
    )
    text = str(value or "passive").strip().lower()
    if text == "active":
        return "Active"
    if text == "passive":
        return "Passive"
    return str(value)


def _active_filter_name(metadata: dict[str, object]) -> str:
    info = _active_filter_info(metadata)
    return str(info.get("name") or info.get("id") or "Selected filter")


def _active_filter_info(metadata: dict[str, object]) -> dict[str, object]:
    active_filter = metadata.get("active_filter")
    if isinstance(active_filter, dict):
        return active_filter

    legacy = metadata.get("kalman_filter")
    active_control_used = _first(metadata, "active_control_input_used_by_filter", "active_control_input_used")
    if isinstance(legacy, dict):
        return {
            "id": metadata.get("active_filter_id", "legacy_filter"),
            "name": metadata.get("active_filter_name") or legacy.get("filter_type") or "Kalman filter",
            "type": metadata.get("active_filter_type") or legacy.get("filter_type"),
            "state_vector": metadata.get("active_filter_state_vector") or legacy.get("state_vector"),
            "process_model": metadata.get("active_filter_process_model") or legacy.get("process_model"),
            "measurement_model": metadata.get("active_filter_measurement_model") or legacy.get("measurement_models"),
            "tracking_mode": metadata.get("tracking_mode") or legacy.get("tracking_mode"),
            "active_control_input_used": active_control_used
            if active_control_used is not None
            else legacy.get("active_control_input_used"),
            "tune": legacy.get("tune", {}),
        }

    return {
        "id": metadata.get("active_filter_id", "active_filter"),
        "name": metadata.get("active_filter_name", "Active filter"),
        "type": metadata.get("active_filter_type", "unknown"),
        "state_vector": metadata.get("active_filter_state_vector", "n/a"),
        "process_model": metadata.get("active_filter_process_model", "n/a"),
        "measurement_model": metadata.get("active_filter_measurement_model", "n/a"),
        "tracking_mode": metadata.get("tracking_mode"),
        "active_control_input_used": active_control_used,
        "tune": metadata.get("active_filter_tune", {}),
    }


def _normalize_sample_fields(sample: dict[str, object]) -> dict[str, object]:
    if "filtered_x" not in sample:
        sample["filtered_x"] = sample.get("kalman_x")
    if "filtered_y" not in sample:
        sample["filtered_y"] = sample.get("kalman_y")
    if "filtered_yaw" not in sample:
        sample["filtered_yaw"] = sample.get("kalman_yaw")
    if "filtered_speed" not in sample:
        sample["filtered_speed"] = sample.get("kalman_speed")
    if "filtered_position_error_m" not in sample:
        sample["filtered_position_error_m"] = sample.get("kalman_position_error_m")
    return sample


def _title_with_context(title: str, metadata: dict[str, object]) -> str:
    general = metadata.get("general", {}) if isinstance(metadata.get("general"), dict) else {}
    active_filter = _active_filter_info(metadata)
    route = general.get("route_name") or "route"
    map_name = general.get("map_name") or general.get("active_carla_map_name") or "map"
    filter_name = active_filter.get("name") or active_filter.get("id") or "filter"
    return f"{title}\n{filter_name} | {route} | {map_name}"


def _add_metadata_box(ax, metadata: dict[str, object]) -> None:
    general = metadata.get("general", {}) if isinstance(metadata.get("general"), dict) else {}
    sensors = metadata.get("sensor_configuration", {}) if isinstance(metadata.get("sensor_configuration"), dict) else {}
    behavior = metadata.get("vehicle_behavior_config", {}) if isinstance(metadata.get("vehicle_behavior_config"), dict) else {}
    gnss = sensors.get("gnss", {}) if isinstance(sensors.get("gnss"), dict) else {}
    text = (
        f"run: {general.get('benchmark_id') or metadata.get('run_id', 'n/a')}\n"
        f"GNSS std: {_fmt(gnss.get('noise_lat_stddev_deg'))}/{_fmt(gnss.get('noise_lon_stddev_deg'))} deg\n"
        f"max/min speed: {_fmt(behavior.get('max_speed_mps'))}/{_fmt(behavior.get('min_curve_speed_mps'))} m/s"
    )
    ax.text(
        0.99,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        family="monospace",
        bbox={"facecolor": "white", "edgecolor": "0.65", "alpha": 0.78},
    )


def _plot_aggregate_bar(output_path: Path, rows: list[dict[str, object]], field: str, ylabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [_route_label(row, index) for index, row in enumerate(rows)]
    values = [_to_float(row.get(field)) for row in rows]
    if labels:
        ax.bar(labels, [value if value is not None else 0.0 for value in values], color="tab:blue")
        ax.tick_params(axis="x", rotation=25)
    else:
        ax.text(0.5, 0.5, "No aggregate rows available", transform=ax.transAxes, ha="center", va="center")
    ax.set_title(ylabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_aggregate_raw_vs_filtered(output_path: Path, rows: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [_route_label(row, index) for index, row in enumerate(rows)]
    raw = [_to_float(row.get("raw_gnss_rmse_m")) or 0.0 for row in rows]
    filtered = [_to_float(row.get("filtered_rmse_m")) or 0.0 for row in rows]
    x_positions = list(range(len(labels)))
    width = 0.38
    if labels:
        ax.bar([x - width / 2 for x in x_positions], raw, width=width, label="Raw GNSS")
        ax.bar([x + width / 2 for x in x_positions], filtered, width=width, label="Filtered")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=25, ha="right")
    else:
        ax.text(0.5, 0.5, "No aggregate rows available", transform=ax.transAxes, ha="center", va="center")
    ax.set_title("Raw GNSS RMSE vs Filtered RMSE")
    ax.set_ylabel("RMSE (m)")
    ax.grid(True, axis="y", alpha=0.3)
    _legend_if_labels(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_aggregate_dual_metric(
    output_path: Path,
    rows: list[dict[str, object]],
    first_field: str,
    second_field: str,
    first_label: str,
    second_label: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [_route_label(row, index) for index, row in enumerate(rows)]
    first_values = [_to_float(row.get(first_field)) for row in rows]
    second_values = [_to_float(row.get(second_field)) for row in rows]
    x_positions = list(range(len(labels)))
    if labels:
        ax.plot(x_positions, [value if value is not None else float("nan") for value in first_values], marker="o", label=first_label)
        ax.plot(x_positions, [value if value is not None else float("nan") for value in second_values], marker="s", label=second_label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=25, ha="right")
    else:
        ax.text(0.5, 0.5, "No aggregate rows available", transform=ax.transAxes, ha="center", va="center")
    ax.set_title(f"{first_label} / {second_label} per Route")
    ax.grid(True, alpha=0.3)
    _legend_if_labels(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_aggregate_segments(output_path: Path, run_folder: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    aggregate = _read_json(run_folder / "aggregate_summary.json")
    segments = aggregate.get("segment_rmse_summary") if isinstance(aggregate, dict) else None
    if isinstance(segments, dict) and segments:
        labels = list(segments.keys())
        values = [_to_float(segments[label].get("position_rmse_m")) if isinstance(segments[label], dict) else None for label in labels]
        ax.bar(labels, [value if value is not None else 0.0 for value in values], color="tab:green")
    else:
        ax.text(0.5, 0.5, "No aggregate segment metrics available", transform=ax.transAxes, ha="center", va="center")
    ax.set_title("Segment-Based Position RMSE Across Routes")
    ax.set_ylabel("Position RMSE (m)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _route_label(row: dict[str, object], index: int) -> str:
    name = str(row.get("route_name") or f"route_{index + 1}")
    if len(name) > 20:
        name = name[:17] + "..."
    return name


def _to_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


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


def _fmt_bool(value: object) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return "n/a"
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return "Yes"
    if text in ("false", "0", "no"):
        return "No"
    return str(value)


def _consistency_warning(mean_nis: object, mean_nees: object) -> bool:
    nis = _to_float(mean_nis)
    nees = _to_float(mean_nees)
    return (nis is not None and nis > 10.0) or (nees is not None and nees > 10.0)


def _legend_if_labels(ax, fontsize: Optional[int] = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(fontsize=fontsize)
