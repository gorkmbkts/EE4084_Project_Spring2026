"""Main simulation application orchestrator."""

from __future__ import annotations

from collections import deque
from enum import Enum
import math
import time
from typing import Optional, Sequence

import pygame

from config.settings import BENCHMARK, DASHBOARD, ROUTE_INITIALIZATION, TOPDOWN_MAP
from src.KalmanLab.filter_base import (
    FilterControlInput,
    TRACKING_MODE_ACTIVE,
    TRACKING_MODE_PASSIVE,
)
from src.KalmanLab.filter_manager import FilterManager
from src.KalmanLab.tune_advisor import TuneRecommendation, recommend_filter_tune
from src.control.driving_behavior import (
    ActuatorRealism,
    CurvatureSpeedPlanner,
    DrivingBehaviorConfig,
    SpeedPlan,
)
from src.control.vehicle_controller import VehicleController
from src.control.waypoint_tracker import TrackingStatus, WaypointTracker
from src.core.carla_client import CarlaClientManager
from src.core.localization_status import LocalizationStatus
from src.core.simulation import SimulationClock
from src.core.state_providers import GroundTruthStateProvider
from src.core.vehicle_state import VehicleState
from src.evaluation.benchmark_config import (
    ACTUATOR_REALISM_PRESETS,
    BEHAVIOR_PRESETS,
    BenchmarkConfig,
    SENSOR_NOISE_PRESETS,
    SENSOR_NOISE_SPECS,
    SensorNoiseConfig,
    actuator_realism_from_values,
    apply_actuator_values,
    apply_behavior_values,
    default_sensor_noise_values,
    sensor_noise_config_from_values,
)
from src.evaluation.closed_loop_auto_tune import (
    ClosedLoopAutoTuneRequest,
    ClosedLoopAutoTuneResult,
    ClosedLoopBenchmarkAutoTuner,
    ClosedLoopValidationRequest,
    closed_loop_stage_budgets,
)
from src.evaluation.filter_performance import FilterPerformanceLogger
from src.evaluation.plot_job_worker import BenchmarkPlotJobWorker
from src.evaluation.route_test_runner import RouteTestRunner
from src.evaluation.sensor_log_recorder import OfflineRecordingConfig, SensorLogRecorder
from src.evaluation.test_route_store import SavedTestRoute, TestRouteStore
from src.localization.gnss_projection import GnssDiagnostics, GnssLocalProjector
from src.planning.map_selector import MapSelector
from src.planning.route_planner import RoutePlanner
from src.planning.waypoint_manager import WaypointManager
from src.sensors.camera_sensor import CameraSensor
from src.sensors.gnss_sensor import GnssSensor
from src.sensors.imu_sensor import ImuSensor
from src.sensors.lidar_sensor import LidarSensor
from src.sensors.sensor_manager import SensorManager
from src.vehicle.manual_controller import ManualController
from src.vehicle.vehicle_manager import VehicleManager
from src.visualization.lidar_panel import LidarPanelRenderer
from src.visualization.pygame_display import PygameDisplay
from src.visualization.topdown_map import TopDownHudData, TopDownMapRenderer
from src.visualization.ui.driving_behavior_widgets import (
    BehaviorTuningPanel,
    ControlVisualizationWidget,
    DrivingDiagnosticsWidget,
)
from src.visualization.ui.benchmark_panels import LiveEvaluationPanel, TestProgressPanel
from src.visualization.ui.parameter_controls import ParameterEditor
from src.visualization.ui.status_bar import StatusBar
from src.visualization.ui.tabbed_panel import TabbedPanel
from src.visualization.waypoint_overlay import WaypointOverlayRenderer
from src.utils.carla_import import ensure_carla_import
from src.utils.map_names import display_map_name, maps_compatible, normalize_map_name

carla = ensure_carla_import()

BENCHMARK_STUCK_SPEED_MPS = 0.15
BENCHMARK_STUCK_SECONDS = 8.0
BENCHMARK_NO_PROGRESS_SECONDS = 14.0
BENCHMARK_PROGRESS_DISTANCE_M = 0.8
BENCHMARK_GOAL_DISTANCE_PROGRESS_M = 1.0
BENCHMARK_LATERAL_DEVIATION_M = 15.0
BENCHMARK_LATERAL_DEVIATION_SECONDS = 3.0
CLOSED_LOOP_AUTOTUNE_PROGRESS_REFRESH_S = 1.0


class DriveMode(Enum):
    """High-level ego driving mode."""

    MANUAL = "MANUAL"
    AUTONOMOUS = "AUTO"


class RouteActivationState(Enum):
    """Lifecycle state for route creation after A/B selection."""

    IDLE = "IDLE"
    WAITING_FOR_LOCALIZATION_STABILITY = "WAITING_FOR_LOCALIZATION_STABILITY"
    ROUTE_ACTIVE = "ROUTE_ACTIVE"


class WorldSwitchResult(Enum):
    """Outcome of an automated benchmark map switch."""

    NOOP = "NOOP"
    SWITCHED = "SWITCHED"
    FAILED = "FAILED"


class SimulationApp:
    """Coordinate CARLA, sensors, display, route selection, and control."""

    def __init__(
        self,
        requested_map_name: Optional[str] = None,
        selected_map_load_name: Optional[str] = None,
        existing_display_surface: Optional[pygame.Surface] = None,
        benchmark_config: Optional[BenchmarkConfig] = None,
        offline_recording_config: Optional[OfflineRecordingConfig] = None,
        closed_loop_auto_tune_request: Optional[ClosedLoopAutoTuneRequest] = None,
    ) -> None:
        pygame.init()
        self._requested_map_name = requested_map_name
        self._selected_map_load_name = selected_map_load_name or requested_map_name
        self._existing_display_surface = existing_display_surface
        self._startup_benchmark_config = benchmark_config
        self._startup_offline_recording_config = offline_recording_config
        self._startup_closed_loop_auto_tune_request = closed_loop_auto_tune_request
        self._lightweight_closed_loop_auto_tune = closed_loop_auto_tune_request is not None
        self._client_manager = CarlaClientManager(requested_map_name=requested_map_name)
        self._clock = SimulationClock()
        self._display: Optional[PygameDisplay] = None
        self._control_panel: Optional[TabbedPanel] = None

        self._vehicle_manager: Optional[VehicleManager] = None
        self._sensor_manager: Optional[SensorManager] = None
        self._waypoint_manager: Optional[WaypointManager] = None
        self._manual_controller: Optional[ManualController] = None
        self._camera_sensor: Optional[CameraSensor] = None
        self._gnss_sensor: Optional[GnssSensor] = None
        self._imu_sensor: Optional[ImuSensor] = None
        self._lidar_sensor: Optional[LidarSensor] = None
        self._vehicle: Optional["carla.Vehicle"] = None

        self._overlay_renderer = WaypointOverlayRenderer()
        self._lidar_renderer = LidarPanelRenderer()
        self._status_bar = StatusBar()
        self._ground_truth_provider: Optional[GroundTruthStateProvider] = None
        self._filter_manager: Optional[FilterManager] = None
        self._gnss_projector: Optional[GnssLocalProjector] = None
        self._map_selector: Optional[MapSelector] = None
        self._topdown_renderer: Optional[TopDownMapRenderer] = None
        self._test_route_store: Optional[TestRouteStore] = None
        self._test_runner: Optional[RouteTestRunner] = None
        self._offline_recorder: Optional[SensorLogRecorder] = None
        self._plot_worker = BenchmarkPlotJobWorker()
        self._performance_logger = FilterPerformanceLogger()
        self.route_planner: Optional[RoutePlanner] = None
        self.waypoint_tracker = WaypointTracker()
        self.driving_behavior_config = DrivingBehaviorConfig()
        self.sensor_noise_config = self._initial_sensor_noise_config(
            benchmark_config,
            closed_loop_auto_tune_request,
        )
        if benchmark_config is not None:
            apply_behavior_values(self.driving_behavior_config, benchmark_config.vehicle_behavior_config)
            apply_actuator_values(self.driving_behavior_config, benchmark_config.actuator_realism_config or {})
        elif closed_loop_auto_tune_request is not None:
            apply_behavior_values(self.driving_behavior_config, closed_loop_auto_tune_request.vehicle_behavior_config)
            apply_actuator_values(self.driving_behavior_config, closed_loop_auto_tune_request.actuator_realism_config)
        self.autonomous_controller = VehicleController(behavior_config=self.driving_behavior_config)
        self.speed_planner = CurvatureSpeedPlanner(self.driving_behavior_config)
        self.actuator_realism = ActuatorRealism(self.driving_behavior_config)

        self._drive_mode = DriveMode.MANUAL
        self._map_selection_active = False
        self._test_route_authoring_active = False
        self._latest_state: Optional[VehicleState] = None
        self._latest_ground_truth_state: Optional[VehicleState] = None
        self._latest_estimated_state: Optional[VehicleState] = None
        self._latest_localization_status: Optional[LocalizationStatus] = None
        self._latest_tracking = self._empty_tracking_status()
        self._latest_speed_plan: SpeedPlan = self.speed_planner.latest_plan
        self._latest_requested_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        self._latest_applied_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        self._planner_status = ""
        self._latest_gnss_diagnostics: Optional[GnssDiagnostics] = None
        self._latest_gnss_frame: Optional[int] = None
        self._gnss_trail_xy: deque[tuple[float, float]] = deque(maxlen=TOPDOWN_MAP.gnss_trail_length)
        self._active_map_name: Optional[str] = None
        self._active_map_id: Optional[str] = None
        self._route_activation_state = RouteActivationState.IDLE
        self._pending_start_waypoint: Optional["carla.Waypoint"] = None
        self._pending_goal_waypoint: Optional["carla.Waypoint"] = None
        self._pending_start_autonomous = False
        self._stabilization_started_monotonic: Optional[float] = None
        self._stabilization_elapsed_seconds = 0.0
        self._stabilization_stable_ticks = 0
        self._stabilization_error_m: Optional[float] = None
        self._stabilization_timed_out = False
        self._route_generation_blocked = False
        self._control_status_text = "Test idle"
        self._behavior_tuning_panel: Optional[BehaviorTuningPanel] = None
        self._sensor_noise_panel: Optional[ParameterEditor] = None
        self._filter_tune_panel: Optional[ParameterEditor] = None
        self._filter_tune_panel_filter_id = ""
        self._filter_tracking_rects: dict[str, pygame.Rect] = {}
        self._filter_apply_recommended_rect = pygame.Rect(0, 0, 1, 1)
        self._last_filter_tune_status = "Tune changes reset the active estimator."
        self._filter_recommendation_applied_by_filter: dict[str, bool] = {}
        self._filter_overlay_font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size)
        self._filter_overlay_small_font = pygame.font.SysFont("consolas", 11)
        self._filter_overlay_bold_font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size, bold=True)
        self._test_progress_panel: Optional[TestProgressPanel] = None
        self._live_evaluation_panel: Optional[LiveEvaluationPanel] = None
        self._control_visual_widget: Optional[ControlVisualizationWidget] = None
        self._driving_diagnostics_widget: Optional[DrivingDiagnosticsWidget] = None
        self._benchmark_start_attempted = False
        self._offline_recording_start_attempted = False
        self._closed_loop_auto_tune_start_attempted = False
        self._closed_loop_auto_tune_result: Optional[ClosedLoopAutoTuneResult] = None
        self._closed_loop_auto_tune_cancel_requested = False
        self._closed_loop_auto_tune_status_lines: list[str] = []
        self._closed_loop_auto_tune_live_line = ""
        self._closed_loop_auto_tune_context: dict[str, object] = {}
        self._closed_loop_auto_tune_live_metrics: dict[str, object] = {}
        self._closed_loop_auto_tune_trial_rows: list[dict[str, object]] = []
        self._closed_loop_auto_tune_current_trial = 0
        self._closed_loop_auto_tune_total_trials = 0
        self._closed_loop_auto_tune_current_stage = ""
        self._closed_loop_auto_tune_current_status = "idle"
        self._closed_loop_auto_tune_background_surface: Optional[pygame.Surface] = (
            existing_display_surface.copy()
            if self._lightweight_closed_loop_auto_tune and existing_display_surface is not None
            else None
        )
        self._closed_loop_auto_tune_progress_dirty = True
        self._closed_loop_auto_tune_last_draw_monotonic = 0.0
        self._closed_loop_auto_tune_last_live_update_monotonic = 0.0
        self._closed_loop_auto_tune_trial_start_debug: dict[str, object] = {}
        self._closed_loop_auto_tune_best_score: Optional[float] = None
        self._closed_loop_auto_tune_previous_no_rendering: Optional[bool] = None
        self._closed_loop_auto_tune_trial_wall_start: Optional[float] = None
        self._closed_loop_auto_tune_trial_sim_start: Optional[float] = None
        self._closed_loop_auto_tune_trial_tick_count = 0
        self._last_sensor_apply_status = "Sensor noise changes respawn GNSS/IMU safely."
        self._failure_monitor_last_progress_time: Optional[float] = None
        self._failure_monitor_last_position: Optional[tuple[float, float]] = None
        self._failure_monitor_last_distance_to_goal: Optional[float] = None
        self._failure_monitor_last_closest_index = 0
        self._failure_monitor_deviation_started: Optional[float] = None
        self._last_benchmark_failure_reason = ""
        self._world_context_generation = 0
        self._world_reload_in_progress = False
        self._skip_frames_after_world_reload = 0

    @staticmethod
    def _initial_sensor_noise_config(
        benchmark_config: Optional[BenchmarkConfig],
        closed_loop_auto_tune_request: Optional[ClosedLoopAutoTuneRequest],
    ) -> SensorNoiseConfig:
        if benchmark_config is not None:
            return benchmark_config.sensor_noise_config
        if closed_loop_auto_tune_request is not None:
            return SensorNoiseConfig.from_dict(closed_loop_auto_tune_request.sensor_noise_config)
        return SensorNoiseConfig()

    def _setup(self) -> None:
        """Initialize CARLA world, vehicle, sensors, route tools, and visualization."""
        self._client_manager.connect()

        self._active_map_name = getattr(self._client_manager.world_map, "name", None)
        self._active_map_id = normalize_map_name(self._active_map_name)

        self._vehicle_manager = VehicleManager(
            world=self._client_manager.world,
            world_map=self._client_manager.world_map,
            blueprint_library=self._client_manager.blueprint_library,
        )
        self._vehicle = self._vehicle_manager.spawn_vehicle()

        self._sensor_manager = SensorManager(
            world=self._client_manager.world,
            blueprint_library=self._client_manager.blueprint_library,
        )
        if self._lightweight_closed_loop_auto_tune:
            self._camera_sensor = None
        else:
            self._camera_sensor = self._sensor_manager.create_rgb_camera(attach_to=self._vehicle)
        self._gnss_sensor = self._sensor_manager.create_gnss(
            attach_to=self._vehicle,
            config=self.sensor_noise_config,
        )
        self._imu_sensor = self._sensor_manager.create_imu(
            attach_to=self._vehicle,
            config=self.sensor_noise_config,
        )
        if self._lightweight_closed_loop_auto_tune:
            self._lidar_sensor = None
        else:
            self._lidar_sensor = self._sensor_manager.create_lidar(attach_to=self._vehicle)

        world_map = self._client_manager.world_map
        self._waypoint_manager = WaypointManager(world_map=world_map)
        self._manual_controller = ManualController(vehicle=self._vehicle)
        self._ground_truth_provider = GroundTruthStateProvider(vehicle=self._vehicle)
        self._gnss_projector = GnssLocalProjector(world_map=world_map)
        self._filter_manager = FilterManager(
            gnss_projector=self._gnss_projector,
            gnss_sensor=self._gnss_sensor,
            imu_sensor=self._imu_sensor,
        )
        self._map_selector = MapSelector(world_map=world_map)
        self.route_planner = RoutePlanner(world_map=world_map)
        self._topdown_renderer = (
            None
            if self._lightweight_closed_loop_auto_tune
            else TopDownMapRenderer(world_map=world_map)
        )
        self._test_route_store = TestRouteStore(map_name=self._active_map_name)
        self._test_runner = RouteTestRunner(
            world_map=world_map,
            route_store=self._test_route_store,
            begin_route_callback=lambda start, goal: self._begin_route_initialization(
                start_waypoint=start,
                goal_waypoint=goal,
                start_autonomous=True,
            ),
            reset_estimator_callback=self._reset_estimator,
            plan_route_callback=self._planned_route_for_metadata,
            weather_callback=self._weather_metadata,
            vehicle_blueprint_callback=self._vehicle_blueprint_metadata,
            active_filter_info_callback=self._active_filter_info_for_metadata,
            active_filter_tune_callback=self._active_filter_tune_for_metadata,
            tracking_mode_callback=self._tracking_mode_for_metadata,
            active_control_used_callback=self._active_control_used_for_metadata,
            enqueue_route_plots_callback=self._plot_worker.enqueue_route_plots,
            enqueue_aggregate_plots_callback=self._plot_worker.enqueue_aggregate_plots,
            selected_map_load_name=self._selected_map_load_name,
        )
        self._offline_recorder = SensorLogRecorder(
            world_map=world_map,
            route_store=self._test_route_store,
            begin_route_callback=self._begin_offline_recording_route,
            plan_route_callback=self._planned_route_for_metadata,
            weather_callback=self._weather_metadata,
            vehicle_blueprint_callback=self._vehicle_blueprint_metadata,
            selected_map_load_name=self._selected_map_load_name,
        )

        if self.route_planner.planner_error:
            self._planner_status = "Planner: fallback"
        else:
            self._planner_status = "Planner: CARLA"

        self._initialize_display()
        if self._startup_closed_loop_auto_tune_request is not None:
            self._apply_startup_closed_loop_auto_tune_request()
        elif self._startup_offline_recording_config is not None:
            self._apply_startup_offline_recording_config()
        else:
            self._apply_startup_benchmark_config()

    def _ensure_ready(self) -> None:
        required = {
            "display": self._display,
            "control panel": self._control_panel,
            "vehicle": self._vehicle,
            "GNSS sensor": self._gnss_sensor,
            "IMU sensor": self._imu_sensor,
            "waypoint manager": self._waypoint_manager,
            "manual controller": self._manual_controller,
            "ground-truth provider": self._ground_truth_provider,
            "filter manager": self._filter_manager,
            "GNSS projector": self._gnss_projector,
            "map selector": self._map_selector,
            "route planner": self.route_planner,
            "test route store": self._test_route_store,
            "test runner": self._test_runner,
            "offline recorder": self._offline_recorder,
        }
        if not self._lightweight_closed_loop_auto_tune:
            required["camera sensor"] = self._camera_sensor
            required["LiDAR sensor"] = self._lidar_sensor
            required["top-down renderer"] = self._topdown_renderer
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(f"Application is not initialized: {', '.join(missing)}.")

    def _initialize_display(self) -> None:
        self._display = PygameDisplay(existing_surface=self._existing_display_surface)
        self._existing_display_surface = None
        self._control_panel = TabbedPanel(
            self._display.control_panel_rect,
            tabs=("Route", "Filters", "Benchmark", "Sensors", "Debug"),
        )
        self._behavior_tuning_panel = BehaviorTuningPanel(
            self._display.behavior_tuning_rect,
            self.driving_behavior_config,
        )
        self._sensor_noise_panel = ParameterEditor(
            specs=SENSOR_NOISE_SPECS,
            values=default_sensor_noise_values()
            | {key: float(value) for key, value in self.sensor_noise_config.to_dict().items() if isinstance(value, (int, float))},
            presets=SENSOR_NOISE_PRESETS,
            active_preset=self.sensor_noise_config.preset_name,
            title="Live Sensor Noise/Error Tuning",
            on_commit=self._commit_live_sensor_noise,
        )
        self._sensor_noise_panel.status_text = self._last_sensor_apply_status
        self._test_progress_panel = TestProgressPanel(self._display.behavior_tuning_rect)
        self._live_evaluation_panel = LiveEvaluationPanel(self._display.control_panel_rect)
        self._control_visual_widget = ControlVisualizationWidget(self._display.control_visual_rect)
        self._driving_diagnostics_widget = DrivingDiagnosticsWidget(
            self._display.driving_state_rect,
            self.driving_behavior_config,
        )
        self._build_control_panel()
        self._sync_filter_tune_panel()
        self._control_status_text = "Dashboard ready"
        if not self._lightweight_closed_loop_auto_tune:
            self._draw_startup_frame()

    def _draw_startup_frame(self) -> None:
        if self._display is None or self._control_panel is None:
            return
        self._display.begin_frame(None)
        self._update_control_panel_state()
        self._control_panel.draw(self._display.surface)
        self._draw_driving_behavior_panels()
        self._draw_status_bar()
        self._display.set_test_mode_titles(self._test_mode_active())
        self._display.end_frame()

    def _planned_route_for_metadata(
        self,
        start_waypoint: "carla.Waypoint",
        goal_waypoint: "carla.Waypoint",
    ) -> list["carla.Waypoint"]:
        if self.route_planner is None:
            return []
        route = self.route_planner.generate_route(start_waypoint, goal_waypoint)
        planned_route = list(route)
        self.route_planner.clear_route()
        self.waypoint_tracker.clear_route()
        self._latest_tracking = self._empty_tracking_status()
        return planned_route

    def _weather_metadata(self) -> Optional[dict[str, object]]:
        try:
            weather = self._client_manager.world.get_weather()
        except RuntimeError:
            return None
        keys = [
            "cloudiness",
            "precipitation",
            "precipitation_deposits",
            "wind_intensity",
            "sun_azimuth_angle",
            "sun_altitude_angle",
            "fog_density",
            "fog_distance",
            "wetness",
        ]
        return {
            key: getattr(weather, key)
            for key in keys
            if hasattr(weather, key)
        }

    def _vehicle_blueprint_metadata(self) -> Optional[str]:
        if self._vehicle is None:
            return None
        return getattr(self._vehicle, "type_id", None)

    def _active_filter_info_for_metadata(self) -> dict[str, object]:
        if self._filter_manager is None:
            return {}
        info = self._filter_manager.get_active_filter_info()
        filter_id = str(info.get("id") or self._filter_manager.active_filter_id or "")
        info["tracking_mode"] = self._filter_manager.tracking_mode
        info["active_control_input_used"] = self._filter_manager.active_control_input_used
        info["recommendation_applied"] = bool(self._filter_recommendation_applied_by_filter.get(filter_id, False))
        return info

    def _active_filter_tune_for_metadata(self) -> dict[str, object]:
        if self._filter_manager is None:
            return {}
        return self._filter_manager.get_active_filter_tune()

    def _tracking_mode_for_metadata(self) -> str:
        if self._filter_manager is None:
            return TRACKING_MODE_PASSIVE
        return self._filter_manager.tracking_mode

    def _active_control_used_for_metadata(self) -> bool:
        if self._filter_manager is None:
            return False
        diagnostics = self._filter_manager.get_diagnostics()
        return bool(
            diagnostics.get("active_command_used_latest_prediction")
            or diagnostics.get("active_control_input_used")
        )

    def _apply_startup_benchmark_config(self) -> None:
        config = self._startup_benchmark_config
        if config is None or self._benchmark_start_attempted:
            return
        self._benchmark_start_attempted = True
        self.sensor_noise_config = config.sensor_noise_config
        apply_behavior_values(self.driving_behavior_config, config.vehicle_behavior_config)
        apply_actuator_values(self.driving_behavior_config, config.actuator_realism_config or {})
        if self._sensor_noise_panel is not None:
            self._sensor_noise_panel.set_values(
                {
                    key: value
                    for key, value in self.sensor_noise_config.to_dict().items()
                    if isinstance(value, (int, float))
                },
                active_preset=config.sensor_noise_preset,
                commit=False,
            )

        if self._filter_manager is not None:
            if config.selected_filter_tune:
                self._filter_manager.update_filter_tune(
                    config.selected_filter,
                    dict(config.selected_filter_tune),
                    reset_active=False,
                )
            ok, message = self._filter_manager.switch_filter(config.selected_filter, skip_current_sensor_frames=True)
            if not ok:
                self._planner_status = message
                self._control_status_text = message
                return
            self._filter_manager.set_tracking_mode(config.tracking_mode, reset_active=True)
            config.selected_filter_tune = self._filter_manager.get_active_filter_tune()
            config.tracking_mode = self._filter_manager.tracking_mode
            self._sync_filter_tune_panel()
            if not self._filter_manager.active_filter_safe_for_autonomous_control():
                self._planner_status = "Benchmark blocked: selected filter is unsafe for autonomous control"
                self._control_status_text = self._planner_status
                return

        if self._test_runner is None:
            self._planner_status = "Benchmark runner unavailable"
            self._control_status_text = self._planner_status
            return
        self._test_route_authoring_active = False
        self._map_selection_active = False
        self._drive_mode = DriveMode.AUTONOMOUS
        started = self._test_runner.start_configured_benchmark(config, self._active_map_name)
        self._planner_status = self._test_runner.status_text
        self._control_status_text = self._test_runner.status_text
        if not started and self._test_runner.needs_map_switch(self._active_map_name):
            self._switch_map_for_benchmark()

    def _apply_startup_closed_loop_auto_tune_request(self) -> None:
        request = self._startup_closed_loop_auto_tune_request
        if request is None or self._closed_loop_auto_tune_start_attempted:
            return
        self._closed_loop_auto_tune_start_attempted = True
        if len(request.validation_routes) != 1:
            self._planner_status = "Closed-loop auto tune blocked: select exactly one validation route"
            self._control_status_text = self._planner_status
            return
        validation_route = request.validation_routes[0]
        if not maps_compatible(self._active_map_name, validation_route.map_name):
            self._planner_status = (
                "Closed-loop auto tune blocked: validation route map "
                f"{display_map_name(validation_route.map_name)} is not the active map "
                f"{display_map_name(self._active_map_name)}"
            )
            self._control_status_text = self._planner_status
            return
        self.sensor_noise_config = SensorNoiseConfig.from_dict(request.sensor_noise_config)
        apply_behavior_values(self.driving_behavior_config, request.vehicle_behavior_config)
        apply_actuator_values(self.driving_behavior_config, request.actuator_realism_config)
        if self._sensor_noise_panel is not None:
            self._sensor_noise_panel.set_values(
                {
                    key: value
                    for key, value in self.sensor_noise_config.to_dict().items()
                    if isinstance(value, (int, float))
                },
                active_preset=str(request.sensor_noise_config.get("preset_name") or "Custom"),
                commit=False,
            )
        try:
            if self._gnss_sensor is not None:
                self._gnss_sensor.apply_config(self.sensor_noise_config, respawn=True)
            if self._imu_sensor is not None:
                self._imu_sensor.apply_config(self.sensor_noise_config, respawn=True)
        except Exception as exc:
            self._planner_status = f"Closed-loop auto tune sensor setup failed: {exc}"
            self._control_status_text = self._planner_status
            return
        self._test_route_authoring_active = False
        self._map_selection_active = False
        self._drive_mode = DriveMode.AUTONOMOUS
        self._closed_loop_auto_tune_cancel_requested = False
        self._closed_loop_auto_tune_best_score = None
        self._closed_loop_auto_tune_live_line = ""
        self._closed_loop_auto_tune_current_trial = 0
        stage_budgets = closed_loop_stage_budgets(request)
        self._closed_loop_auto_tune_total_trials = stage_budgets.total_trials
        self._closed_loop_auto_tune_current_stage = "Context baseline"
        self._closed_loop_auto_tune_current_status = "Starting staged CARLA route trials"
        self._closed_loop_auto_tune_live_metrics = {}
        self._closed_loop_auto_tune_trial_rows = []
        self._closed_loop_auto_tune_trial_start_debug = {}
        self._closed_loop_auto_tune_context = {
            "filter": request.filter_id,
            "tracking_mode": request.tracking_mode,
            "route": validation_route.name,
            "map": display_map_name(validation_route.map_name),
            "sensor_noise": request.sensor_noise_profile or request.sensor_noise_config.get("preset_name") or "Custom",
            "behavior": request.vehicle_behavior_profile or request.vehicle_behavior_config.get("preset_name") or "Custom",
            "actuator": request.actuator_realism_profile or request.actuator_realism_config.get("preset_name") or "Custom",
            "trial_count": self._closed_loop_auto_tune_total_trials,
            "stage_budgets": stage_budgets.to_dict(),
            "active_control_policy": (
                "active-control params may be tuned"
                if request.tracking_mode == TRACKING_MODE_ACTIVE
                else "active-control params are not tuned"
            ),
        }
        self._closed_loop_auto_tune_status_lines = [
            "Staged mode: context baseline, passive Q/model, active-control when selected, then local fine-tune.",
            "Each candidate gets one route attempt; a route failure ends that trial.",
            "CARLA no-rendering mode enabled; fixed_delta_seconds is unchanged.",
        ]
        self._closed_loop_auto_tune_previous_no_rendering = self._client_manager.no_rendering_mode
        self._client_manager.set_no_rendering_mode(True)
        self._planner_status = "Closed-loop auto tune: staged trial search starting"
        self._control_status_text = self._planner_status
        self._capture_closed_loop_auto_tune_background()
        self._mark_closed_loop_auto_tune_progress_dirty()
        self._draw_closed_loop_auto_tune_progress_frame(force=True)
        runner = AppClosedLoopValidationRunner(self)
        try:
            result = ClosedLoopBenchmarkAutoTuner(validation_runner=runner).run(
                request,
                progress_callback=self._closed_loop_auto_tune_progress_callback,
                stop_requested=lambda: self._closed_loop_auto_tune_cancel_requested,
            )
        except Exception as exc:
            self._planner_status = f"Closed-loop auto tune failed: {exc}"
            self._control_status_text = self._planner_status
            self._closed_loop_auto_tune_status_lines.append(self._planner_status)
            self._closed_loop_auto_tune_current_status = self._planner_status
            self._drive_mode = DriveMode.MANUAL
            self._restore_closed_loop_auto_tune_rendering()
            self._mark_closed_loop_auto_tune_progress_dirty()
            self._draw_closed_loop_auto_tune_progress_frame(force=True)
            return
        self._closed_loop_auto_tune_result = result
        self._restore_closed_loop_auto_tune_rendering()
        if self._filter_manager is not None and result.best_tune:
            self._filter_manager.update_filter_tune(result.filter_id, result.best_tune, reset_active=True)
            self._filter_manager.switch_filter(result.filter_id, skip_current_sensor_frames=True)
            self._filter_manager.set_tracking_mode(result.tracking_mode, reset_active=True)
            self._sync_filter_tune_panel()
        self._planner_status = f"Closed-loop auto tune complete: {result.saved_config_path}"
        self._control_status_text = self._planner_status
        self._closed_loop_auto_tune_status_lines.append(self._planner_status)
        self._closed_loop_auto_tune_current_status = "Complete"
        self._drive_mode = DriveMode.MANUAL
        self._mark_closed_loop_auto_tune_progress_dirty()
        self._draw_closed_loop_auto_tune_progress_frame(force=True)

    def _closed_loop_auto_tune_progress_callback(self, event_name: str, payload: dict[str, object]) -> None:
        def number_text(value: object, digits: int = 3) -> str:
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return f"{float(value):.{digits}f}"
            return "n/a"

        if event_name == "search_started":
            self._closed_loop_auto_tune_total_trials = max(1, int(payload.get("trial_count") or self._closed_loop_auto_tune_total_trials or 1))
            self._closed_loop_auto_tune_context.update(
                {
                    "strategy": payload.get("strategy") or "direct",
                    "search_param_count": payload.get("search_param_count"),
                    "actuator_search_policy": payload.get("actuator_search_policy"),
                    "stage_budgets": payload.get("stage_budgets"),
                }
            )
            self._closed_loop_auto_tune_current_status = "Staged search started"
            text = (
                f"Staged search started | trials {payload.get('trial_count')} | {payload.get('strategy')} | "
                f"params {payload.get('search_param_count')} | actuator {payload.get('actuator_realism_profile')}"
            )
        elif event_name == "stage_started":
            self._closed_loop_auto_tune_current_stage = str(payload.get("stage_label") or payload.get("stage") or "")
            self._closed_loop_auto_tune_current_status = f"Running {self._closed_loop_auto_tune_current_stage}"
            text = (
                f"Stage started | {self._closed_loop_auto_tune_current_stage} | "
                f"{payload.get('stage_trial_count')} trials | tracking {payload.get('tracking_mode')}"
            )
        elif event_name == "trial_started":
            self._closed_loop_auto_tune_current_trial = int(payload.get("trial_index") or 0)
            self._closed_loop_auto_tune_current_stage = str(payload.get("stage_label") or payload.get("stage") or "")
            self._closed_loop_auto_tune_total_trials = max(
                self._closed_loop_auto_tune_total_trials,
                int(payload.get("trial_total") or self._closed_loop_auto_tune_total_trials or 1),
            )
            self._closed_loop_auto_tune_current_status = f"{self._closed_loop_auto_tune_current_stage}: trial running"
            self._upsert_closed_loop_auto_tune_trial_row(
                {
                    "trial": self._closed_loop_auto_tune_current_trial,
                    "stage": self._closed_loop_auto_tune_current_stage,
                    "status": "running",
                    "score": None,
                    "rmse": None,
                    "mean_cte": None,
                    "max_cte": None,
                    "nis": None,
                    "nees": None,
                    "reason": "",
                    "families": ", ".join(str(item) for item in (payload.get("affected_families") or [])),
                    "changes": payload.get("changed_parameters_summary") or "",
                    "adaptation": payload.get("adaptation_decision") or "",
                }
            )
            text = (
                f"{self._closed_loop_auto_tune_current_stage} | trial "
                f"{payload.get('stage_trial_index')}/{payload.get('stage_trial_total')} "
                f"(overall {payload.get('trial_index')}/{payload.get('trial_total')}) | "
                f"route {payload.get('route_name')} | best {number_text(payload.get('best_score'))}"
            )
        elif event_name == "trial_finished":
            metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
            self._closed_loop_auto_tune_best_score = (
                float(payload["best_score"])
                if isinstance(payload.get("best_score"), (int, float)) and math.isfinite(float(payload["best_score"]))
                else self._closed_loop_auto_tune_best_score
            )
            rmse = self._closed_loop_auto_tune_metric(metrics, "eval_filtered_rmse_m", "filtered_rmse_m")
            mean_cte = self._closed_loop_auto_tune_metric(
                metrics,
                "driving_mean_cross_track_error_m",
                "mean_cross_track_error_m",
            )
            max_cte = self._closed_loop_auto_tune_metric(
                metrics,
                "driving_max_cross_track_error_m",
                "max_cross_track_error_m",
            )
            nis = self._closed_loop_auto_tune_nis_metric(metrics)
            nees = self._closed_loop_auto_tune_metric(
                metrics,
                "driving_mean_position_nees",
                "eval_mean_position_nees",
                "mean_position_nees",
                "driving_mean_nees",
                "eval_mean_nees",
                "mean_nees",
            )
            failure_class = str(payload.get("failure_class") or "")
            reason = failure_class or str(payload.get("failure_reason") or "")
            self._closed_loop_auto_tune_current_stage = str(payload.get("stage_label") or payload.get("stage") or "")
            self._closed_loop_auto_tune_current_status = "Trial failed" if payload.get("failed") else "Trial finished"
            self._upsert_closed_loop_auto_tune_trial_row(
                {
                    "trial": int(payload.get("trial_index") or 0),
                    "stage": self._closed_loop_auto_tune_current_stage,
                    "status": "failed" if payload.get("failed") else "ok",
                    "score": payload.get("score"),
                    "rmse": rmse,
                    "mean_cte": mean_cte,
                    "max_cte": max_cte,
                    "nis": nis,
                    "nees": nees,
                    "reason": reason,
                    "families": ", ".join(str(item) for item in (payload.get("affected_families") or [])),
                    "changes": payload.get("changed_parameters_summary") or "",
                    "adaptation": payload.get("adaptation_decision") or "",
                }
            )
            text = (
                f"{self._closed_loop_auto_tune_current_stage} | trial "
                f"{payload.get('stage_trial_index')}/{payload.get('stage_trial_total')} finished | "
                f"score {number_text(payload.get('score'))} | best {number_text(self._closed_loop_auto_tune_best_score)} | "
                f"RMSE {number_text(rmse)} m | "
                f"CTE {number_text(mean_cte)} m | "
                f"success {'yes' if payload.get('route_completion_success') else 'no'}"
            )
            if reason:
                text += f" | reason {reason}"
        elif event_name == "new_search_best":
            self._closed_loop_auto_tune_best_score = (
                float(payload["score"])
                if isinstance(payload.get("score"), (int, float)) and math.isfinite(float(payload["score"]))
                else self._closed_loop_auto_tune_best_score
            )
            text = (
                f"New best | {payload.get('stage_label') or payload.get('stage')} | trial {payload.get('trial_index')} | "
                f"score {number_text(self._closed_loop_auto_tune_best_score)}"
            )
        elif event_name == "completed":
            self._closed_loop_auto_tune_current_status = "Complete"
            text = (
                f"Closed-loop auto tune complete | best {number_text(payload.get('best_score'))} | "
                f"saved {payload.get('saved_config_path')}"
            )
        else:
            text = f"Closed-loop auto tune: {event_name}"
        self._planner_status = text
        self._control_status_text = text
        self._closed_loop_auto_tune_status_lines.append(text)
        self._closed_loop_auto_tune_status_lines = self._closed_loop_auto_tune_status_lines[-16:]
        self._mark_closed_loop_auto_tune_progress_dirty()
        self._draw_closed_loop_auto_tune_progress_frame(force=True)
        self._pump_closed_loop_auto_tune_events()

    def _upsert_closed_loop_auto_tune_trial_row(self, row: dict[str, object]) -> None:
        trial = int(row.get("trial") or 0)
        if trial <= 0:
            return
        for index, existing in enumerate(self._closed_loop_auto_tune_trial_rows):
            if int(existing.get("trial") or 0) == trial:
                merged = dict(existing)
                merged.update(row)
                self._closed_loop_auto_tune_trial_rows[index] = merged
                return
        self._closed_loop_auto_tune_trial_rows.append(dict(row))
        self._closed_loop_auto_tune_trial_rows = self._closed_loop_auto_tune_trial_rows[-12:]

    @staticmethod
    def _closed_loop_auto_tune_metric(metrics: dict[str, object], *keys: str) -> Optional[float]:
        for key in keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
        return None

    @classmethod
    def _closed_loop_auto_tune_nis_metric(cls, metrics: dict[str, object]) -> Optional[float]:
        direct = cls._closed_loop_auto_tune_metric(
            metrics,
            "driving_legacy_mean_nis_mixed",
            "eval_legacy_mean_nis_mixed",
            "legacy_mean_nis_mixed",
            "mean_nis",
        )
        if direct is not None:
            return direct
        values: list[float] = []
        for key in ("driving_nis_by_type_summary", "eval_nis_by_type_summary", "nis_by_type_summary"):
            summary = metrics.get(key)
            if not isinstance(summary, dict):
                continue
            for item in summary.values():
                if not isinstance(item, dict):
                    continue
                value = cls._closed_loop_auto_tune_metric(item, "mean")
                if value is not None:
                    values.append(value)
        return max(values) if values else None

    def _restore_closed_loop_auto_tune_rendering(self) -> None:
        if self._closed_loop_auto_tune_previous_no_rendering is None:
            return
        self._client_manager.set_no_rendering_mode(self._closed_loop_auto_tune_previous_no_rendering)
        self._closed_loop_auto_tune_previous_no_rendering = None

    def _capture_closed_loop_auto_tune_background(self) -> None:
        if self._display is None:
            self._closed_loop_auto_tune_background_surface = None
            return
        if (
            self._closed_loop_auto_tune_background_surface is not None
            and self._closed_loop_auto_tune_background_surface.get_size() == self._display.surface.get_size()
        ):
            return
        try:
            self._closed_loop_auto_tune_background_surface = self._display.surface.copy()
        except pygame.error:
            self._closed_loop_auto_tune_background_surface = None

    def _mark_closed_loop_auto_tune_progress_dirty(self) -> None:
        self._closed_loop_auto_tune_progress_dirty = True

    def _pump_closed_loop_auto_tune_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._closed_loop_auto_tune_cancel_requested = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._closed_loop_auto_tune_cancel_requested = True
        pygame.event.pump()

    def _draw_closed_loop_auto_tune_progress_frame(self, force: bool = False) -> None:
        if self._display is None:
            return
        now = time.monotonic()
        if not force and not self._closed_loop_auto_tune_progress_dirty:
            return
        surface = self._display.surface
        width, height = surface.get_size()
        background = self._closed_loop_auto_tune_background_surface
        if background is not None and background.get_size() == surface.get_size():
            surface.blit(background, (0, 0))
        else:
            surface.fill((8, 12, 18))
        dim = pygame.Surface((width, height), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 165))
        surface.blit(dim, (0, 0))

        modal_w = min(width - 50, 1280)
        modal_h = min(height - 50, 650)
        modal = pygame.Rect(0, 0, modal_w, modal_h)
        modal.center = (width // 2, height // 2)
        pygame.draw.rect(surface, DASHBOARD.panel_background_color, modal, border_radius=8)
        pygame.draw.rect(surface, DASHBOARD.success_color, modal, width=1, border_radius=8)
        content = modal.inflate(-30, -26)
        self._draw_overlay_text(surface, "Closed-loop Auto Tune", content.topleft, self._filter_overlay_bold_font, DASHBOARD.title_color, content.width - 120)
        self._draw_overlay_text(
            surface,
            "Direct no-render CARLA route trials. ESC or window close requests safe cancellation.",
            (content.left, content.top + 25),
            self._filter_overlay_small_font,
            DASHBOARD.muted_text_color,
            content.width,
        )

        context = self._closed_loop_auto_tune_context
        trial_text = f"{self._closed_loop_auto_tune_current_trial}/{self._closed_loop_auto_tune_total_trials or '?'}"
        stage_budgets = context.get("stage_budgets") if isinstance(context.get("stage_budgets"), dict) else {}
        budget_text = (
            f"P {stage_budgets.get('baseline_passive_q_model_trials', '?')} | "
            f"A {stage_budgets.get('active_control_trials', '?')} | "
            f"J {stage_budgets.get('joint_local_fine_tune_trials', '?')}"
        )
        left_x = content.left
        right_x = content.left + content.width // 2 + 14
        y = content.top + 58
        context_lines = [
            f"Trial: {trial_text} | Stage budgets: {budget_text}",
            f"Filter: {context.get('filter', 'n/a')}",
            f"Tracking: {context.get('tracking_mode', 'n/a')} ({context.get('active_control_policy', 'n/a')})",
            f"Route: {context.get('route', 'n/a')} | Map: {context.get('map', 'n/a')}",
        ]
        context_lines_right = [
            f"Sensor noise: {context.get('sensor_noise', 'n/a')}",
            f"Behavior: {context.get('behavior', 'n/a')}",
            f"Actuator: {context.get('actuator', 'n/a')}",
            f"Stage: {self._closed_loop_auto_tune_current_stage} | Status: {self._closed_loop_auto_tune_current_status}",
        ]
        for offset, line in enumerate(context_lines):
            self._draw_overlay_text(surface, line, (left_x, y + offset * 18), self._filter_overlay_small_font, DASHBOARD.text_color, content.width // 2 - 22)
        for offset, line in enumerate(context_lines_right):
            color = DASHBOARD.warning_color if line.startswith("Status:") and any(word in line.lower() for word in ("fail", "cancel", "abort")) else DASHBOARD.text_color
            self._draw_overlay_text(surface, line, (right_x, y + offset * 18), self._filter_overlay_small_font, color, content.right - right_x)

        metrics_rect = pygame.Rect(content.left, y + 86, content.width, 92)
        pygame.draw.rect(surface, DASHBOARD.panel_inner_color, metrics_rect, border_radius=6)
        pygame.draw.rect(surface, DASHBOARD.panel_border_color, metrics_rect, width=1, border_radius=6)
        metrics = self._closed_loop_auto_tune_live_metrics
        metric_rows = [
            ("Sim", self._format_metric_seconds(metrics.get("sim_time_s")), "Wall", self._format_metric_seconds(metrics.get("wall_time_s")), "Speedup", self._format_metric_ratio(metrics.get("speedup"))),
            ("Ticks/s", self._format_metric_number(metrics.get("ticks_per_second"), 1), "Progress", self._format_progress_percent(self._finite_metric(metrics.get("route_progress_percent"))), "Best", self._format_metric_number(self._closed_loop_auto_tune_best_score, 3)),
            ("RMSE", self._format_metric_number(metrics.get("rmse_m"), 3), "Mean CTE", self._format_metric_number(metrics.get("mean_cte_m"), 3), "Max CTE", self._format_metric_number(metrics.get("max_cte_m"), 3)),
            ("NIS", self._format_metric_number(metrics.get("nis"), 3), "NEES", self._format_metric_number(metrics.get("nees"), 3), "Failure", str(metrics.get("failure_reason") or "none")),
        ]
        col_w = metrics_rect.width // 3
        for row_index, row in enumerate(metric_rows):
            row_y = metrics_rect.top + 10 + row_index * 19
            for col_index in range(3):
                label = row[col_index * 2]
                value = row[col_index * 2 + 1]
                text = f"{label}: {value}"
                color = DASHBOARD.warning_color if label == "Failure" and value != "none" else DASHBOARD.text_color
                self._draw_overlay_text(surface, text, (metrics_rect.left + 12 + col_index * col_w, row_y), self._filter_overlay_small_font, color, col_w - 18)

        table_top = metrics_rect.bottom + 16
        self._draw_overlay_text(surface, "Trial Results", (content.left, table_top), self._filter_overlay_bold_font, DASHBOARD.title_color, content.width)
        header_y = table_top + 27
        headers = ("Stage", "Trial", "Status", "Score", "RMSE", "Mean CTE", "Max CTE", "NIS", "NEES", "Failure", "Family", "Changed", "Decision")
        fixed_widths = (110, 42, 56, 68, 58, 66, 62, 50, 50, 112, 100, 150)
        col_widths = fixed_widths + (max(110, content.width - sum(fixed_widths)),)
        x = content.left
        for header, col_width in zip(headers, col_widths):
            self._draw_overlay_text(surface, header, (x, header_y), self._filter_overlay_small_font, DASHBOARD.muted_text_color, col_width - 6)
            x += col_width
        row_y = header_y + 22
        row_h = 22
        max_rows = max(1, (content.bottom - row_y - 54) // row_h)
        rows = self._closed_loop_auto_tune_trial_rows[-max_rows:]
        for row in rows:
            status = str(row.get("status") or "")
            color = DASHBOARD.warning_color if status == "failed" else DASHBOARD.text_color
            values = (
                str(row.get("stage") or ""),
                str(row.get("trial") or ""),
                status,
                self._format_metric_number(row.get("score"), 3),
                self._format_metric_number(row.get("rmse"), 3),
                self._format_metric_number(row.get("mean_cte"), 3),
                self._format_metric_number(row.get("max_cte"), 3),
                self._format_metric_number(row.get("nis"), 3),
                self._format_metric_number(row.get("nees"), 3),
                str(row.get("reason") or ""),
                str(row.get("families") or ""),
                str(row.get("changes") or ""),
                str(row.get("adaptation") or ""),
            )
            x = content.left
            for value, col_width in zip(values, col_widths):
                self._draw_overlay_text(surface, value, (x, row_y), self._filter_overlay_small_font, color, col_width - 6)
                x += col_width
            row_y += row_h

        notes_rect = pygame.Rect(content.left, content.bottom - 44, content.width, 44)
        pygame.draw.rect(surface, (14, 18, 24), notes_rect, border_radius=5)
        note_y = notes_rect.top + 6
        for line in self._closed_loop_auto_tune_status_lines[-2:]:
            color = DASHBOARD.warning_color if any(word in str(line).lower() for word in ("failed", "cancel", "reason", "blocked")) else DASHBOARD.muted_text_color
            self._draw_overlay_text(surface, line, (notes_rect.left + 10, note_y), self._filter_overlay_small_font, color, notes_rect.width - 20)
            note_y += 17
        pygame.display.flip()
        self._closed_loop_auto_tune_last_draw_monotonic = now
        self._closed_loop_auto_tune_progress_dirty = False

    def _update_closed_loop_auto_tune_live_line(self) -> None:
        now = time.monotonic()
        wall = self._closed_loop_auto_tune_trial_wall_seconds()
        sim = self._closed_loop_auto_tune_trial_sim_seconds()
        speedup = sim / wall if wall > 1.0e-9 else 0.0
        tps = self._closed_loop_auto_tune_trial_tick_count / wall if wall > 1.0e-9 else 0.0
        progress = self._route_completion_percent()
        cte = self._latest_tracking.cross_track_error_m
        rmse = self._active_performance_logger().current_position_error_m
        nis = None
        nees = None
        if self._filter_manager is not None:
            diagnostics = self._filter_manager.get_diagnostics()
            nis = diagnostics.get("latest_nis")
            nees = diagnostics.get("latest_nees")
        failure = self._last_benchmark_failure_reason or (self._test_runner.last_failure_reason if self._test_runner is not None else "")
        self._closed_loop_auto_tune_live_metrics = {
            "wall_time_s": wall,
            "sim_time_s": sim,
            "speedup": speedup,
            "ticks_per_second": tps,
            "route_progress_percent": progress,
            "rmse_m": rmse,
            "mean_cte_m": cte,
            "max_cte_m": None,
            "nis": nis,
            "nees": nees,
            "failure_reason": failure,
        }
        self._closed_loop_auto_tune_live_line = (
            f"route {self._format_progress_percent(progress)} | "
            f"wall {wall:.1f}s | sim {sim:.1f}s | speedup {speedup:.2f}x | {tps:.1f} tick/s | "
            f"best {self._format_live_number(self._closed_loop_auto_tune_best_score)} | "
            f"RMSE {self._format_live_number(rmse)}m | NIS {self._format_live_number(nis)} | "
            f"NEES {self._format_live_number(nees)} | CTE {self._format_live_number(cte)}m | "
            f"failure {failure or 'none'}"
        )
        if now - self._closed_loop_auto_tune_last_live_update_monotonic >= CLOSED_LOOP_AUTOTUNE_PROGRESS_REFRESH_S:
            self._closed_loop_auto_tune_last_live_update_monotonic = now
            self._mark_closed_loop_auto_tune_progress_dirty()

    def _closed_loop_auto_tune_trial_wall_seconds(self) -> float:
        if self._closed_loop_auto_tune_trial_wall_start is None:
            return 0.0
        return max(0.0, time.monotonic() - self._closed_loop_auto_tune_trial_wall_start)

    def _closed_loop_auto_tune_trial_sim_seconds(self) -> float:
        if self._latest_ground_truth_state is None or self._closed_loop_auto_tune_trial_sim_start is None:
            return 0.0
        return max(0.0, float(self._latest_ground_truth_state.timestamp) - self._closed_loop_auto_tune_trial_sim_start)

    @staticmethod
    def _format_live_number(value: object) -> str:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return f"{float(value):.3g}"
        return "n/a"

    @staticmethod
    def _format_progress_percent(value: Optional[float]) -> str:
        if value is None or not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.1f}%"

    @staticmethod
    def _finite_metric(value: object) -> Optional[float]:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    @staticmethod
    def _format_metric_number(value: object, digits: int = 2) -> str:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return f"{float(value):.{digits}g}"
        return "n/a"

    @staticmethod
    def _format_metric_seconds(value: object) -> str:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return f"{float(value):.1f}s"
        return "n/a"

    @staticmethod
    def _format_metric_ratio(value: object) -> str:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return f"{float(value):.2f}x"
        return "n/a"

    def _reset_direct_closed_loop_trial_lifecycle(self) -> None:
        """Clear app-side route/control state before one direct auto-tune trial."""
        if self._test_runner is not None and self._test_runner.is_active:
            self._test_runner.stop(aborted=True, reason="Direct auto-tune trial reset")
        self._test_route_authoring_active = False
        self._map_selection_active = False
        self._cancel_route_activation()
        if self.route_planner is not None:
            self.route_planner.clear_route()
        self.waypoint_tracker.clear_route()
        self._latest_tracking = self._empty_tracking_status()
        self._latest_state = None
        self._latest_ground_truth_state = None
        self._latest_estimated_state = None
        self._latest_localization_status = None
        self._latest_gnss_diagnostics = None
        self._latest_gnss_frame = None
        self._gnss_trail_xy.clear()
        self._last_benchmark_failure_reason = ""
        self._reset_benchmark_failure_monitor()
        neutral = self._neutral_vehicle_control()
        self._set_latest_control(neutral, neutral)
        self.autonomous_controller = VehicleController(behavior_config=self.driving_behavior_config)
        self.speed_planner = CurvatureSpeedPlanner(self.driving_behavior_config)
        self.actuator_realism = ActuatorRealism(self.driving_behavior_config)
        self._reset_driving_behavior(initial_speed_mps=0.0, actuator_control=neutral)
        self._drive_mode = DriveMode.MANUAL
        if self._vehicle is not None:
            try:
                self._vehicle.set_autopilot(False)
                self._vehicle.apply_control(neutral)
            except RuntimeError:
                pass

    def _reset_direct_closed_loop_trial_after_finish(self) -> None:
        """Leave no completed-route or brake state for the next direct trial."""
        neutral = self._neutral_vehicle_control()
        if self._vehicle is not None:
            try:
                self._vehicle.set_autopilot(False)
                self._vehicle.apply_control(neutral)
            except RuntimeError:
                pass
        self._set_latest_control(neutral, neutral)
        self.actuator_realism.reset(neutral)
        self.speed_planner.reset(initial_speed_mps=0.0)
        self._latest_speed_plan = self.speed_planner.latest_plan
        if self.route_planner is not None:
            self.route_planner.clear_route()
        self.waypoint_tracker.clear_route()
        self._latest_tracking = self._empty_tracking_status()
        self._cancel_route_activation()
        self._reset_benchmark_failure_monitor()
        self._last_benchmark_failure_reason = ""
        self._drive_mode = DriveMode.MANUAL

    @staticmethod
    def _neutral_vehicle_control() -> "carla.VehicleControl":
        return carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=0.0,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )

    def _closed_loop_trial_debug_snapshot(self, route: SavedTestRoute) -> dict[str, object]:
        start_distance = None
        if self._latest_ground_truth_state is not None:
            route_data = route.to_dict()
            start = route_data.get("start") if isinstance(route_data.get("start"), dict) else {}
            x = start.get("x")
            y = start.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                start_distance = math.hypot(float(self._latest_ground_truth_state.x) - float(x), float(self._latest_ground_truth_state.y) - float(y))
        requested = self._control_debug_dict(self._latest_requested_control)
        applied = self._control_debug_dict(self._latest_applied_control)
        return {
            "route_activation_state": self._route_activation_state.value,
            "target_waypoint_exists": self._latest_tracking.target_waypoint is not None,
            "tracker_completed": bool(self._latest_tracking.completed),
            "requested_control": requested,
            "applied_control": applied,
            "vehicle_speed_mps": self._latest_ground_truth_state.speed if self._latest_ground_truth_state is not None else None,
            "route_progress_percent": self._route_completion_percent(),
            "distance_from_route_start_m": start_distance,
        }

    @staticmethod
    def _control_debug_dict(control: "carla.VehicleControl") -> dict[str, float]:
        return {
            "throttle": float(getattr(control, "throttle", 0.0)),
            "brake": float(getattr(control, "brake", 0.0)),
            "steer": float(getattr(control, "steer", 0.0)),
        }

    def _format_closed_loop_trial_debug_line(self, debug: dict[str, object]) -> str:
        requested = debug.get("requested_control") if isinstance(debug.get("requested_control"), dict) else {}
        applied = debug.get("applied_control") if isinstance(debug.get("applied_control"), dict) else {}
        return (
            "Trial start debug | "
            f"activation {debug.get('route_activation_state')} | "
            f"target {'yes' if debug.get('target_waypoint_exists') else 'no'} | "
            f"completed {'yes' if debug.get('tracker_completed') else 'no'} | "
            f"req T/B/S {self._format_live_number(requested.get('throttle'))}/"
            f"{self._format_live_number(requested.get('brake'))}/"
            f"{self._format_live_number(requested.get('steer'))} | "
            f"app T/B/S {self._format_live_number(applied.get('throttle'))}/"
            f"{self._format_live_number(applied.get('brake'))}/"
            f"{self._format_live_number(applied.get('steer'))} | "
            f"speed {self._format_live_number(debug.get('vehicle_speed_mps'))} | "
            f"progress {self._format_progress_percent(self._finite_metric(debug.get('route_progress_percent')))} | "
            f"start_dist {self._format_live_number(debug.get('distance_from_route_start_m'))}m"
        )

    def _run_closed_loop_auto_tune_validation(self, request: ClosedLoopValidationRequest) -> dict[str, object]:
        route = self._saved_route_from_validation_request(request)
        if route is None:
            return self._closed_loop_validation_failure_metrics(request, "Validation route data is unavailable")
        if not maps_compatible(self._active_map_name, route.map_name):
            return self._closed_loop_validation_failure_metrics(
                request,
                (
                    "Validation route map "
                    f"{display_map_name(route.map_name)} is not active map {display_map_name(self._active_map_name)}"
                ),
            )
        if self._test_runner is None:
            return self._closed_loop_validation_failure_metrics(request, "Benchmark runner unavailable")
        if self._filter_manager is None:
            return self._closed_loop_validation_failure_metrics(request, "Filter manager unavailable")
        if self._test_runner.is_active:
            self._test_runner.stop(aborted=True, reason="Resetting benchmark runner for next direct auto-tune trial")

        sensor_config = SensorNoiseConfig.from_dict(request.sensor_noise_config)
        self.sensor_noise_config = sensor_config
        apply_behavior_values(self.driving_behavior_config, request.vehicle_behavior_config)
        apply_actuator_values(self.driving_behavior_config, request.actuator_realism_config)
        self._reset_direct_closed_loop_trial_lifecycle()
        try:
            if self._gnss_sensor is not None:
                self._gnss_sensor.apply_config(sensor_config, respawn=True)
            if self._imu_sensor is not None:
                self._imu_sensor.apply_config(sensor_config, respawn=True)
        except Exception as exc:
            return self._closed_loop_validation_failure_metrics(request, f"Sensor setup failed: {exc}")

        ok, message = self._filter_manager.update_filter_tune(request.filter_id, request.finalist.candidate_tune, reset_active=True)
        if not ok:
            return self._closed_loop_validation_failure_metrics(request, message)
        ok, message = self._filter_manager.switch_filter(request.filter_id, skip_current_sensor_frames=True)
        if not ok:
            return self._closed_loop_validation_failure_metrics(request, message)
        ok, message = self._filter_manager.set_tracking_mode(request.tracking_mode, reset_active=True)
        if not ok:
            return self._closed_loop_validation_failure_metrics(request, message)
        if not self._filter_manager.active_filter_safe_for_autonomous_control():
            return self._closed_loop_validation_failure_metrics(request, "Selected filter is unsafe for autonomous control")
        self._sync_filter_tune_panel()
        self._clear_localization_after_filter_reset()
        self._reset_driving_behavior(actuator_control=self._neutral_vehicle_control())

        config = BenchmarkConfig(
            selected_filter=request.filter_id,
            selected_routes=(route,),
            sensor_noise_config=sensor_config,
            vehicle_behavior_config=dict(request.vehicle_behavior_config),
            actuator_realism_config=dict(request.actuator_realism_config),
            selected_filter_tune=dict(self._filter_manager.get_active_filter_tune()),
            tracking_mode=request.tracking_mode,
            sensor_noise_preset=str(request.sensor_noise_config.get("preset_name") or "Custom"),
            vehicle_behavior_preset=str(request.vehicle_behavior_config.get("preset_name") or "Custom"),
            actuator_realism_preset=str(request.actuator_realism_config.get("preset_name") or "Custom"),
            output_root=str(request.output_folder),
            run_id="v",
            metadata={
                "startup_mode": "closed_loop_auto_tune_validation",
                "compact_route_output": True,
                "direct_closed_loop_mode": True,
                "no_rendering_mode": True,
                "max_route_attempts": 1,
                "route_attempt_policy": "one_attempt_per_candidate_trial",
                "finalist_rank": request.finalist.rank,
                "offline_score": request.finalist.offline_score,
                "offline_trial_index": request.finalist.trial_index,
                "candidate_tune": dict(request.finalist.candidate_tune),
                "tuning_stage": request.stage,
                "stage_trial_index": request.stage_trial_index,
                "stage_trial_total": request.stage_trial_total,
                "changed_parameters": dict(request.changed_parameters),
                "changed_parameters_summary": request.changed_parameters_summary,
            },
        )
        self._test_route_authoring_active = False
        self._map_selection_active = False
        self._drive_mode = DriveMode.MANUAL
        started = self._test_runner.start_configured_benchmark(config, self._active_map_name)
        self._planner_status = self._test_runner.status_text
        self._control_status_text = self._test_runner.status_text
        if not started:
            reason = self._test_runner.status_text or "Validation benchmark did not start"
            if self._test_runner.needs_map_switch(self._active_map_name):
                reason = "Validation blocked by route map mismatch"
            self._test_runner.stop(aborted=True, reason=reason)
            return self._closed_loop_validation_failure_metrics(request, reason)

        self._closed_loop_auto_tune_trial_start_debug = self._closed_loop_trial_debug_snapshot(route)
        self._closed_loop_auto_tune_status_lines.append(self._format_closed_loop_trial_debug_line(self._closed_loop_auto_tune_trial_start_debug))
        self._closed_loop_auto_tune_status_lines = self._closed_loop_auto_tune_status_lines[-16:]
        self._mark_closed_loop_auto_tune_progress_dirty()
        self._draw_closed_loop_auto_tune_progress_frame(force=True)

        self._closed_loop_auto_tune_trial_wall_start = time.monotonic()
        self._closed_loop_auto_tune_trial_sim_start = None
        self._closed_loop_auto_tune_trial_tick_count = 0
        while self._test_runner is not None and self._test_runner.is_active:
            if self._closed_loop_auto_tune_cancel_requested:
                self._test_runner.stop(aborted=True, reason="Closed-loop auto tune cancelled")
                break
            running = self._run_closed_loop_validation_frame()
            if not running:
                self._closed_loop_auto_tune_cancel_requested = True
                if self._test_runner is not None and self._test_runner.is_active:
                    self._test_runner.stop(aborted=True, reason="Closed-loop auto tune cancelled by window close")
                break

        summary = dict(self._test_runner.last_exported_summary or {}) if self._test_runner is not None else {}
        if not summary:
            self._reset_direct_closed_loop_trial_after_finish()
            return self._closed_loop_validation_failure_metrics(request, "Validation produced no route summary")
        summary.setdefault("route_completion_success", bool(summary.get("route_completion_success")))
        summary.setdefault("route_aborted", not bool(summary.get("route_completion_success")))
        summary.setdefault("timeout", False)
        summary.setdefault("abort_reason", summary.get("last_failure_reason") or summary.get("error") or "")
        summary["output_folder"] = summary.get("route_folder") or str(self._test_runner.benchmark_folder or request.output_folder)
        summary["validation_output_folder"] = str(self._test_runner.benchmark_folder or request.output_folder)
        wall_time = self._closed_loop_auto_tune_trial_wall_seconds()
        sim_time = self._closed_loop_auto_tune_trial_sim_seconds()
        summary["auto_tune_wall_time_s"] = wall_time
        summary["auto_tune_sim_time_s"] = sim_time
        summary["auto_tune_speedup"] = sim_time / wall_time if wall_time > 1.0e-9 else None
        summary["auto_tune_ticks_per_second"] = self._closed_loop_auto_tune_trial_tick_count / wall_time if wall_time > 1.0e-9 else None
        summary["actuator_realism_config"] = dict(request.actuator_realism_config)
        summary["trial_start_debug"] = dict(self._closed_loop_auto_tune_trial_start_debug)
        self._reset_direct_closed_loop_trial_after_finish()
        return summary

    def _run_closed_loop_validation_frame(self) -> bool:
        self._pump_closed_loop_auto_tune_events()
        if self._closed_loop_auto_tune_cancel_requested:
            return False
        self._client_manager.tick()
        self._closed_loop_auto_tune_trial_tick_count += 1
        if self._world_reload_in_progress or self._skip_frames_after_world_reload > 0:
            self._skip_frames_after_world_reload = max(0, self._skip_frames_after_world_reload - 1)
            self._update_closed_loop_auto_tune_live_line()
            self._draw_closed_loop_auto_tune_progress_frame()
            self._tick_closed_loop_auto_tune_pygame()
            return True
        if self._ground_truth_provider is None or self._filter_manager is None:
            self._update_closed_loop_auto_tune_live_line()
            self._draw_closed_loop_auto_tune_progress_frame()
            self._tick_closed_loop_auto_tune_pygame()
            return True
        try:
            self._latest_ground_truth_state = self._ground_truth_provider.get_state()
            if self._closed_loop_auto_tune_trial_sim_start is None:
                self._closed_loop_auto_tune_trial_sim_start = float(self._latest_ground_truth_state.timestamp)
            self._latest_estimated_state = self._filter_manager.update()
            self._latest_localization_status = self._filter_manager.get_status(self._latest_ground_truth_state)
        except RuntimeError as exc:
            self._planner_status = f"World context unavailable: {exc}"
            self._control_status_text = self._planner_status
            self._update_closed_loop_auto_tune_live_line()
            self._draw_closed_loop_auto_tune_progress_frame()
            self._tick_closed_loop_auto_tune_pygame()
            return True
        self._update_route_activation_state()
        self._latest_state = self._state_for_tracking_and_control()
        if self._can_update_route_tracking() and self._latest_state is not None:
            self._latest_tracking = self.waypoint_tracker.update(self._latest_state)
        self._update_sensor_diagnostics()
        world_reloaded = self._update_test_performance()
        if world_reloaded:
            self._tick_closed_loop_auto_tune_pygame()
            return True
        self._apply_closed_loop_validation_control()
        self._update_closed_loop_auto_tune_live_line()
        self._draw_closed_loop_auto_tune_progress_frame()
        self._tick_closed_loop_auto_tune_pygame()
        return True

    def _tick_closed_loop_auto_tune_pygame(self) -> None:
        self._clock.tick_pygame_unthrottled()

    def _apply_closed_loop_validation_control(self) -> None:
        if self._vehicle is None:
            return
        dt_seconds = self._clock.fixed_delta_seconds
        if self._route_activation_state == RouteActivationState.WAITING_FOR_LOCALIZATION_STABILITY:
            control = carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=ROUTE_INITIALIZATION.hold_brake,
                hand_brake=False,
            )
            if self._apply_vehicle_control_safely(control):
                self._set_latest_control(control, control)
                # Closed-loop tuning keeps active prediction dormant until route
                # activation so startup brake commands cannot bias initialization.
                self.actuator_realism.reset(control)
            return
        control_state = self._state_for_tracking_and_control()
        if control_state is None:
            control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
            if self._apply_vehicle_control_safely(control):
                self._set_latest_control(control, control)
                self._feed_filter_control_input(control, source="autonomous_safety_brake")
                self.actuator_realism.reset(control)
            return
        preview_waypoints = self.waypoint_tracker.get_preview_waypoints(max_count=90)
        no_active_target = self._latest_tracking.target_waypoint is None
        self._latest_speed_plan = self.speed_planner.plan(
            state=control_state,
            preview_waypoints=preview_waypoints,
            route_completed=self._latest_tracking.completed or no_active_target,
            dt_seconds=dt_seconds,
        )
        control = self.autonomous_controller.compute_control(
            state=control_state,
            target_waypoint=self._latest_tracking.target_waypoint,
            route_completed=self._latest_tracking.completed,
            target_speed_mps=self._latest_speed_plan.target_speed_mps,
        )
        applied_control = self.actuator_realism.apply(control, dt_seconds)
        if self._apply_vehicle_control_safely(applied_control):
            self._set_latest_control(control, applied_control)
            self._feed_filter_control_input(applied_control, source="autonomous_applied")

    def _saved_route_from_validation_request(self, request: ClosedLoopValidationRequest) -> Optional[SavedTestRoute]:
        route_info = request.validation_route
        if route_info.route_data:
            try:
                return SavedTestRoute.from_dict(route_info.route_data, route_info.map_name)
            except (KeyError, TypeError, ValueError):
                return None
        if self._test_route_store is not None:
            for route in self._test_route_store.all_routes:
                if route.name == route_info.name and maps_compatible(route.map_name, route_info.map_name):
                    return route
        return None

    def _closed_loop_validation_failure_metrics(
        self,
        request: ClosedLoopValidationRequest,
        reason: str,
    ) -> dict[str, object]:
        return {
            "route_completion_success": False,
            "route_aborted": True,
            "timeout": "timeout" in reason.lower(),
            "abort_reason": reason,
            "error": reason,
            "selected_filter": request.filter_id,
            "tracking_mode": request.tracking_mode,
            "route_name": request.validation_route.name,
            "map_name": self._active_map_name,
            "route_map_name": request.validation_route.map_name,
            "output_folder": str(request.output_folder),
        }

    def _apply_startup_offline_recording_config(self) -> None:
        config = self._startup_offline_recording_config
        if config is None or self._offline_recording_start_attempted:
            return
        self._offline_recording_start_attempted = True
        self.sensor_noise_config = config.sensor_noise_config
        apply_behavior_values(self.driving_behavior_config, config.vehicle_behavior_config)
        apply_actuator_values(
            self.driving_behavior_config,
            actuator_realism_from_values(
                ACTUATOR_REALISM_PRESETS["Realistic"],
                preset_name="Realistic",
            ),
        )
        if self._sensor_noise_panel is not None:
            self._sensor_noise_panel.set_values(
                {
                    key: value
                    for key, value in self.sensor_noise_config.to_dict().items()
                    if isinstance(value, (int, float))
                },
                active_preset=config.sensor_noise_preset,
                commit=False,
            )
        try:
            if self._gnss_sensor is not None:
                self._gnss_sensor.apply_config(self.sensor_noise_config, respawn=True)
            if self._imu_sensor is not None:
                self._imu_sensor.apply_config(self.sensor_noise_config, respawn=True)
        except Exception as exc:
            self._planner_status = f"Offline recording sensor setup failed: {exc}"
            self._control_status_text = self._planner_status
            return
        if self._offline_recorder is None:
            self._planner_status = "Offline recorder unavailable"
            self._control_status_text = self._planner_status
            return
        self._test_route_authoring_active = False
        self._map_selection_active = False
        self._drive_mode = DriveMode.AUTONOMOUS
        started = self._offline_recorder.start_recording(config, self._active_map_name)
        self._planner_status = self._offline_recorder.status_text
        self._control_status_text = self._offline_recorder.status_text
        if not started and self._offline_recorder.needs_map_switch(self._active_map_name):
            self._switch_map_for_offline_recording()

    def _benchmark_output_status(self) -> str:
        runner = self._test_runner
        if runner is None or runner.benchmark_folder is None:
            return "Output: none"
        return f"Output: {runner.benchmark_folder.name}"

    def _commit_live_sensor_noise(self, values: dict[str, float], preset_name: str) -> None:
        if self._test_mode_active():
            self._last_sensor_apply_status = "Sensor changes locked during benchmark test mode."
            if self._sensor_noise_panel is not None:
                self._sensor_noise_panel.status_text = self._last_sensor_apply_status
            return
        config = sensor_noise_config_from_values(values, preset_name=preset_name)
        self.sensor_noise_config = config
        try:
            if self._gnss_sensor is not None:
                self._gnss_sensor.apply_config(config, respawn=True)
            if self._imu_sensor is not None:
                self._imu_sensor.apply_config(config, respawn=True)
            self._reset_estimator()
            self._last_sensor_apply_status = "Applied: GNSS/IMU sensors respawned with updated noise."
        except Exception as exc:
            self._last_sensor_apply_status = f"Sensor noise apply failed: {exc}"
        if self._sensor_noise_panel is not None:
            self._sensor_noise_panel.status_text = self._last_sensor_apply_status

    def _active_performance_logger(self) -> FilterPerformanceLogger:
        if self._test_runner is not None and self._test_runner.current_logger is not None:
            return self._test_runner.current_logger
        return self._performance_logger

    def _build_control_panel(self) -> None:
        assert self._control_panel is not None
        for label, callback in (
            ("Map Select ON/OFF", self._toggle_map_selection),
            ("Test Route Mode ON/OFF", self._toggle_test_route_authoring),
            ("Save A/B as Test Route", self._save_current_ab_as_test_route),
            ("Previous Test Route", self._select_previous_test_route),
            ("Next Test Route", self._select_next_test_route),
            ("Load Selected Test Route", self._load_selected_test_route),
            ("Reset Selection", self._reset_selection_and_route),
            ("Clear Route", self._clear_route),
            ("Manual Mode", self._set_manual_mode),
            ("Autonomous Mode", self._try_enable_autonomous_mode),
            ("Emergency Brake", self._emergency_brake),
        ):
            self._control_panel.add_button("Route", label, callback)

        if self._filter_manager is not None:
            for record in self._filter_manager.available_filters():
                self._control_panel.add_button(
                    "Filters",
                    record.display_name,
                    lambda filter_id=record.filter_id: self._select_filter(filter_id),
                )

        for label, callback in (
            ("Start Test Run", self._start_test_run),
            ("Stop Test Run", self._stop_test_run),
            ("Save Test Report", self._save_test_report),
            ("Regenerate Plots", self._regenerate_plots),
        ):
            self._control_panel.add_button("Benchmark", label, callback)

        self._control_panel.add_button("Debug", "Reset Estimator", self._reset_estimator)

    def _update_control_panel_state(self) -> None:
        assert self._display is not None
        assert self._control_panel is not None
        store = self._test_route_store
        runner = self._test_runner
        logger = self._active_performance_logger()
        test_active = runner.is_active if runner is not None else False
        endpoints_ready = self._map_selector is not None and self._map_selector.endpoints is not None
        filter_switch_enabled = (
            not test_active
            and self._drive_mode == DriveMode.MANUAL
            and self._route_activation_state != RouteActivationState.WAITING_FOR_LOCALIZATION_STABILITY
        )

        self._control_panel.set_rect(self._display.control_panel_rect)
        self._control_panel.set_button_state("Manual Mode", active=self._drive_mode == DriveMode.MANUAL)
        self._control_panel.set_button_state("Autonomous Mode", active=self._drive_mode == DriveMode.AUTONOMOUS)
        self._control_panel.set_button_state("Map Select ON/OFF", active=self._map_selection_active)
        self._control_panel.set_button_state("Test Route Mode ON/OFF", active=self._test_route_authoring_active)
        self._control_panel.set_button_state("Save A/B as Test Route", active=endpoints_ready)
        self._control_panel.set_button_state(
            "Start Test Run",
            enabled=not test_active and self._active_filter_safe_for_autonomous(),
        )
        self._control_panel.set_button_state("Stop Test Run", enabled=test_active, active=test_active)
        self._control_panel.set_button_state(
            "Save Test Report",
            enabled=bool(logger.samples) or (runner is not None and runner.benchmark_folder is not None),
        )
        self._control_panel.set_button_state(
            "Regenerate Plots",
            enabled=runner is not None and runner.benchmark_folder is not None,
        )

        if self._filter_manager is not None:
            active_id = self._filter_manager.active_filter_id
            for record in self._filter_manager.available_filters():
                self._control_panel.set_button_state(
                    record.display_name,
                    enabled=filter_switch_enabled or record.filter_id == active_id,
                    active=record.filter_id == active_id,
                )

        active_tab = self._control_panel.active_tab
        if active_tab == "Route":
            self._control_panel.set_text_lines("Route", self._route_tab_lines())
        elif active_tab == "Filters":
            self._control_panel.set_text_lines("Filters", ())
        elif active_tab == "Benchmark":
            self._control_panel.set_text_lines("Benchmark", self._benchmark_tab_lines())
        elif active_tab == "Sensors":
            self._control_panel.set_text_lines("Sensors", self._sensors_tab_lines())
        elif active_tab == "Debug":
            self._control_panel.set_text_lines("Debug", self._debug_tab_lines())

        self._status_bar.set_text(self._status_bar_text())

    def _select_filter(self, filter_id: str) -> None:
        if self._filter_manager is None:
            self._control_status_text = "Filter manager unavailable"
            return
        if self._test_runner is not None and self._test_runner.is_active:
            self._control_status_text = "Filter switching blocked during active benchmark"
            self._planner_status = self._control_status_text
            return
        if self._drive_mode == DriveMode.AUTONOMOUS:
            self._control_status_text = "Filter switching blocked during autonomous driving"
            self._planner_status = self._control_status_text
            return
        if self._route_activation_state == RouteActivationState.WAITING_FOR_LOCALIZATION_STABILITY:
            self._control_status_text = "Filter switching blocked during route initialization"
            self._planner_status = self._control_status_text
            return

        ok, message = self._filter_manager.switch_filter(filter_id, skip_current_sensor_frames=True)
        self._sync_filter_tune_panel()
        self._clear_localization_after_filter_reset()
        warning = self._active_filter_warning()
        self._planner_status = message
        self._control_status_text = warning or message
        if not ok:
            self._planner_status = message

    def _handle_filters_tab_event(self, event: pygame.event.Event) -> bool:
        if self._filter_tune_panel is not None and self._filter_tune_panel.handle_event(event):
            return True
        if event.type != pygame.MOUSEBUTTONDOWN or getattr(event, "button", None) != 1 or not hasattr(event, "pos"):
            return False
        position = event.pos
        for mode, rect in self._filter_tracking_rects.items():
            if rect.collidepoint(position):
                self._set_live_tracking_mode(mode)
                return True
        if self._filter_apply_recommended_rect.collidepoint(position):
            self._apply_recommended_filter_tune()
            return True
        return False

    def _sync_filter_tune_panel(self) -> None:
        manager = self._filter_manager
        if manager is None or manager.active_filter_id is None:
            self._filter_tune_panel = None
            self._filter_tune_panel_filter_id = ""
            return
        filter_id = manager.active_filter_id
        specs = manager.get_filter_tune_specs(filter_id)
        if not specs:
            self._filter_tune_panel = None
            self._filter_tune_panel_filter_id = filter_id
            return
        values = manager.get_filter_runtime_tune(filter_id)
        if self._filter_tune_panel is None or self._filter_tune_panel_filter_id != filter_id:
            self._filter_tune_panel = ParameterEditor(
                specs=specs,
                values={key: float(value) for key, value in values.items() if isinstance(value, (int, float, bool))},
                presets={},
                active_preset="Custom",
                title="Tune Parameters",
                on_commit=self._commit_live_filter_tune,
            )
            self._filter_tune_panel_filter_id = filter_id
        else:
            self._filter_tune_panel.set_values(values, active_preset="Custom", commit=False)
        self._filter_tune_panel.status_text = self._last_filter_tune_status

    def _commit_live_filter_tune(self, values: dict[str, float], _preset_name: str) -> None:
        manager = self._filter_manager
        if manager is None or manager.active_filter_id is None:
            self._last_filter_tune_status = "Filter manager unavailable."
            return
        if self._test_mode_active():
            self._last_filter_tune_status = "Tune edits locked during automated benchmark."
            if self._filter_tune_panel is not None:
                self._filter_tune_panel.status_text = self._last_filter_tune_status
            return
        filter_id = manager.active_filter_id
        ok, message = manager.update_filter_tune(filter_id, values, reset_active=True)
        self._last_filter_tune_status = "Applied and estimator reset." if ok else message
        self._filter_recommendation_applied_by_filter[filter_id] = False
        self._sync_filter_tune_panel()
        if ok:
            self._clear_localization_after_filter_reset()
            self._hold_vehicle_after_filter_reset()
        self._planner_status = message
        self._control_status_text = self._last_filter_tune_status

    def _set_live_tracking_mode(self, mode: str) -> None:
        manager = self._filter_manager
        if manager is None:
            self._control_status_text = "Filter manager unavailable"
            return
        if self._test_mode_active():
            self._control_status_text = "Tracking mode locked during automated benchmark"
            self._planner_status = self._control_status_text
            return
        ok, message = manager.set_tracking_mode(mode, reset_active=True)
        self._last_filter_tune_status = message
        self._sync_filter_tune_panel()
        if ok:
            self._clear_localization_after_filter_reset()
            self._hold_vehicle_after_filter_reset()
        self._planner_status = message
        self._control_status_text = message

    def _current_filter_recommendation(self) -> TuneRecommendation:
        manager = self._filter_manager
        if manager is None or manager.active_filter_id is None:
            return TuneRecommendation("", TRACKING_MODE_PASSIVE, {}, ("Filter manager unavailable.",))
        filter_id = manager.active_filter_id
        return recommend_filter_tune(
            filter_id=filter_id,
            sensor_noise_config=self.sensor_noise_config,
            tracking_mode=manager.tracking_mode,
            current_tune=manager.get_filter_runtime_tune(filter_id),
            tune_specs=manager.get_filter_tune_specs(filter_id),
        )

    def _apply_recommended_filter_tune(self) -> None:
        manager = self._filter_manager
        if manager is None or manager.active_filter_id is None:
            return
        if self._test_mode_active():
            self._last_filter_tune_status = "Recommendations locked during automated benchmark."
            return
        recommendation = self._current_filter_recommendation()
        if not recommendation.values:
            self._last_filter_tune_status = "No recommendation available for this filter."
            return
        filter_id = manager.active_filter_id
        ok, message = manager.update_filter_tune(filter_id, recommendation.values, reset_active=True)
        self._last_filter_tune_status = "Recommended tune applied; estimator reset." if ok else message
        self._filter_recommendation_applied_by_filter[filter_id] = bool(ok)
        self._sync_filter_tune_panel()
        if ok:
            self._clear_localization_after_filter_reset()
            self._hold_vehicle_after_filter_reset()
        self._planner_status = message
        self._control_status_text = self._last_filter_tune_status

    def _draw_filters_tab_overlay(self) -> None:
        if self._display is None or self._control_panel is None:
            return
        manager = self._filter_manager
        if manager is None:
            return
        surface = self._display.surface
        panel = self._display.control_panel_rect
        padding = DASHBOARD.panel_padding_px
        y = self._filters_button_bottom(panel) + 8
        bottom = panel.bottom - padding
        content_width = panel.width - 2 * padding
        self._filter_tracking_rects.clear()
        self._filter_apply_recommended_rect = pygame.Rect(0, 0, 1, 1)

        mode_rect = pygame.Rect(panel.left + padding, y, content_width, 26)
        self._draw_filter_tracking_buttons(surface, mode_rect, manager.tracking_mode)
        y = mode_rect.bottom + 8

        summary_height = 48
        summary_rect = pygame.Rect(panel.left + padding, y, content_width, summary_height)
        pygame.draw.rect(surface, (17, 22, 29), summary_rect, border_radius=4)
        pygame.draw.rect(surface, DASHBOARD.panel_border_color, summary_rect, width=1, border_radius=4)
        info = manager.get_active_filter_info()
        warning = self._active_filter_warning()
        summary_lines = [
            f"Active: {info.get('name', 'none')} ({info.get('id', 'n/a')}) | Mode: {manager.tracking_mode}",
            warning or manager.tracking_mode_message or self._last_filter_tune_status,
        ]
        line_y = summary_rect.top + 7
        for line in summary_lines:
            lower_line = line.lower()
            color = (
                DASHBOARD.warning_color
                if "unsafe" in lower_line or "unsupported" in lower_line or "experimental" in lower_line
                else DASHBOARD.text_color
            )
            self._draw_overlay_text(surface, line, (summary_rect.left + 8, line_y), self._filter_overlay_small_font, color, summary_rect.width - 16)
            line_y += 17
        y = summary_rect.bottom + 8

        recommendation_height = 72
        recommendation_rect = pygame.Rect(panel.left + padding, max(y + 82, bottom - recommendation_height), content_width, recommendation_height)
        tune_rect = pygame.Rect(panel.left + padding, y, content_width, max(64, recommendation_rect.top - y - 8))
        if self._filter_tune_panel is not None:
            self._filter_tune_panel.status_text = self._last_filter_tune_status
            self._filter_tune_panel.draw(surface, tune_rect)
        else:
            pygame.draw.rect(surface, DASHBOARD.panel_inner_color, tune_rect, border_radius=4)
            pygame.draw.rect(surface, DASHBOARD.panel_border_color, tune_rect, width=1, border_radius=4)
            self._draw_overlay_text(
                surface,
                "Selected filter exposes no editable tune specs.",
                (tune_rect.left + 8, tune_rect.top + 10),
                self._filter_overlay_small_font,
                DASHBOARD.muted_text_color,
                tune_rect.width - 16,
            )
        self._draw_live_recommendation_card(surface, recommendation_rect)

    def _draw_filter_tracking_buttons(self, surface: pygame.Surface, rect: pygame.Rect, active_mode: str) -> None:
        gap = 6
        button_width = max(60, (rect.width - gap) // 2)
        for index, (mode, label) in enumerate(((TRACKING_MODE_PASSIVE, "Passive"), (TRACKING_MODE_ACTIVE, "Active"))):
            button = pygame.Rect(rect.left + index * (button_width + gap), rect.top, button_width, rect.height)
            self._filter_tracking_rects[mode] = button
            active = active_mode == mode
            hovered = button.collidepoint(pygame.mouse.get_pos())
            background = (35, 73, 53) if active else ((38, 47, 61) if hovered else (24, 30, 39))
            border = DASHBOARD.success_color if active else DASHBOARD.panel_border_color
            pygame.draw.rect(surface, background, button, border_radius=4)
            pygame.draw.rect(surface, border, button, width=1, border_radius=4)
            rendered = self._filter_overlay_bold_font.render(label, True, DASHBOARD.title_color if active else DASHBOARD.text_color)
            surface.blit(rendered, rendered.get_rect(center=button.center))

    def _draw_live_recommendation_card(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        recommendation = self._current_filter_recommendation()
        pygame.draw.rect(surface, (18, 23, 30), rect, border_radius=4)
        pygame.draw.rect(surface, DASHBOARD.panel_border_color, rect, width=1, border_radius=4)
        title_y = rect.top + 7
        self._draw_overlay_text(surface, "Recommendation", (rect.left + 8, title_y), self._filter_overlay_bold_font, DASHBOARD.title_color, rect.width - 154)
        self._filter_apply_recommended_rect = pygame.Rect(rect.right - 136, rect.top + 6, 128, 24)
        self._draw_overlay_button(surface, self._filter_apply_recommended_rect, "Apply Recommended", enabled=recommendation.has_values)
        lines = list(recommendation.messages[:2])
        lines.extend(recommendation.warnings[:1])
        if not lines:
            lines = ["No recommendation available for this filter."]
        y = rect.top + 34
        for line in lines[:2]:
            color = DASHBOARD.warning_color if line in recommendation.warnings else DASHBOARD.muted_text_color
            self._draw_overlay_text(surface, line, (rect.left + 8, y), self._filter_overlay_small_font, color, rect.width - 16)
            y += 15

    def _draw_overlay_button(self, surface: pygame.Surface, rect: pygame.Rect, label: str, enabled: bool = True) -> None:
        hovered = rect.collidepoint(pygame.mouse.get_pos())
        if enabled:
            background = (32, 88, 63) if hovered else (24, 64, 48)
            border = DASHBOARD.success_color
            text_color = DASHBOARD.title_color
        else:
            background = (38, 42, 50)
            border = (56, 62, 72)
            text_color = DASHBOARD.muted_text_color
        pygame.draw.rect(surface, background, rect, border_radius=4)
        pygame.draw.rect(surface, border, rect, width=1, border_radius=4)
        rendered = self._filter_overlay_small_font.render(self._fit_overlay_text(label, self._filter_overlay_small_font, rect.width - 8), True, text_color)
        surface.blit(rendered, rendered.get_rect(center=rect.center))

    def _filters_button_bottom(self, panel: pygame.Rect) -> int:
        manager = self._filter_manager
        button_count = len(manager.available_filters()) if manager is not None else 0
        if button_count <= 0:
            return panel.top + 40
        columns = 1 if panel.width < 380 else 2
        rows = (button_count + columns - 1) // columns
        return panel.top + 40 + rows * 28 - 4

    def _draw_overlay_text(
        self,
        surface: pygame.Surface,
        text: object,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
        max_width: int,
    ) -> None:
        rendered = font.render(self._fit_overlay_text(str(text), font, max_width), True, color)
        surface.blit(rendered, position)

    @staticmethod
    def _fit_overlay_text(text: str, font: pygame.font.Font, max_width: int) -> str:
        if font.size(text)[0] <= max_width:
            return text
        ellipsis = "..."
        available = max(0, max_width - font.size(ellipsis)[0])
        fitted = ""
        for char in text:
            if font.size(fitted + char)[0] > available:
                break
            fitted += char
        return fitted.rstrip() + ellipsis

    def _route_tab_lines(self) -> list[str]:
        store = self._test_route_store
        route_count = store.route_count() if store is not None else 0
        other_map_count = store.other_map_route_count() if store is not None else 0
        endpoints_ready = self._map_selector is not None and self._map_selector.endpoints is not None
        route = self.route_planner.get_route() if self.route_planner is not None else []
        lines = [
            "Route:",
            f"Active map: {self._active_map_display_name()}",
            f"Mode: {self._drive_mode.value}",
            f"Map select: {'ON' if self._map_selection_active else 'OFF'}",
            f"Test route mode: {'ON' if self._test_route_authoring_active else 'OFF'}",
            f"A/B selection: {'READY' if endpoints_ready else 'not set'}",
            f"Compatible saved routes: {route_count}",
            f"Selected route: {self._selected_route_label()}",
            f"Active route waypoints: {len(route)}",
            f"Route activation: {self._route_activation_state.value}",
            self._planner_status,
            self._control_status_text,
        ]
        if other_map_count > 0:
            lines.insert(7, "Other-map routes hidden")
        if route_count == 0:
            lines.insert(7, "No saved routes for this map")
        return lines

    def _filters_tab_lines(self) -> list[str]:
        manager = self._filter_manager
        if manager is None:
            return ["Filters:", "Filter manager unavailable"]
        info = manager.get_active_filter_info()
        tune = manager.get_active_filter_tune()
        estimate = self._latest_estimated_state
        provided_fields = tuple(info.get("provided_state_fields", ()))
        state_caps = estimate.capabilities() if estimate is not None else ()
        lines = [
            "Filters:",
            f"Active: {info.get('name', 'none')} ({info.get('id', 'n/a')})",
            f"Model type: {info.get('model_type', 'n/a')}",
            f"State source: {estimate.source_filter_id if estimate is not None else 'n/a'}",
            f"State caps: {', '.join(state_caps) if state_caps else 'basic'}",
            f"Provided fields: {', '.join(provided_fields[:6]) if provided_fields else 'basic'}",
            f"Type: {info.get('type', 'n/a')}",
            f"Safe for autonomous: {'YES' if info.get('safe_for_autonomous_control', True) else 'NO'}",
            f"Benchmark selectable: {'YES' if info.get('benchmark_selectable', info.get('safe_for_autonomous_control', True)) else 'NO'}",
            f"Active tracking supported: {'YES' if info.get('active_tracking_supported', False) else 'NO'}",
            f"Experimental: {'YES' if info.get('experimental', False) else 'NO'}",
        ]
        warning = self._active_filter_warning()
        if warning:
            lines.append(f"Warning: {warning}")
        lines.extend(
            [
                f"State: {info.get('state_vector', 'n/a')}",
                f"Process: {info.get('process_model', 'n/a')}",
                f"Measurement: {info.get('measurement_model', 'n/a')}",
                f"Description: {info.get('description', 'n/a')}",
                "TUNE:",
            ]
        )
        for key, value in tune.items():
            lines.append(f"{key}: {value}")

        invalid = manager.invalid_filters(include_templates=False)
        if invalid:
            lines.append("Invalid plugins:")
            for record in invalid:
                lines.append(f"{record.file_path.name}: {record.error}")
        templates = [record for record in manager.all_records() if record.template]
        for record in templates:
            lines.append(f"Template: {record.file_path.name}")
        return lines

    def _benchmark_tab_lines(self) -> list[str]:
        runner = self._test_runner
        logger = self._active_performance_logger()
        ratio = self._ratio(logger.current_raw_gnss_error_m, logger.current_position_error_m)
        lines = [
            "Benchmark:",
            f"Active map: {self._active_map_display_name()}",
            f"State: {'ACTIVE' if runner is not None and runner.is_active else 'inactive'}",
            runner.status_text if runner is not None else "Benchmark idle",
            f"Output folder: {runner.benchmark_folder.name if runner is not None and runner.benchmark_folder is not None else 'none'}",
            f"Filtered error: {self._format_optional_metric(logger.current_position_error_m, 'm')}",
            f"Raw GNSS error: {self._format_optional_metric(logger.current_raw_gnss_error_m, 'm')}",
            f"Improvement ratio: {self._format_optional_metric(ratio, 'x')}",
            f"Cross-track error: {self._format_optional_metric(logger.current_cross_track_error_m, 'm')}",
            f"Running RMSE: {self._format_optional_metric(logger.running_rmse_m(), 'm')}",
            f"Completion: {'YES' if self._latest_tracking.completed else 'NO'}",
        ]
        lines.extend(self._plot_job_status_lines())
        if not self._active_filter_safe_for_autonomous():
            lines.append("Warning: active filter is unsafe for autonomous benchmark control")
        return lines

    def _plot_job_status_lines(self) -> list[str]:
        status = self._plot_worker.poll_status()
        lines = [self._plot_worker.status_text()]
        if status.latest_error:
            lines.append(f"Latest plot error: {status.latest_error}")
        return lines

    def _sensors_tab_lines(self) -> list[str]:
        gnss = self._gnss_sensor.get_latest_measurement() if self._gnss_sensor is not None else None
        imu = self._imu_sensor.get_latest_measurement() if self._imu_sensor is not None else None
        lidar = self._lidar_sensor.get_latest_measurement() if self._lidar_sensor is not None else None
        status = self._latest_localization_status
        lines = [
            "Sensors:",
            self._client_manager.sync_status,
            f"Fixed dt: {self._client_manager.fixed_delta_seconds:.3f}s",
            f"Pygame dt: {self._clock.last_frame_dt_seconds:.3f}s",
            f"Localization: {status.filter_name if status is not None else 'waiting'}",
            f"Initialized: {'YES' if status is not None and status.initialized else 'NO'}",
            f"Position error: {self._format_optional_metric(status.position_error_m if status else None, 'm')}",
        ]
        if gnss is None:
            lines.append("GNSS: waiting")
        else:
            lines.extend(
                [
                    f"GNSS frame: {gnss.frame}",
                    f"Lat/Lon: {gnss.latitude:.8f}, {gnss.longitude:.8f}",
                    f"Alt: {gnss.altitude:.2f} m",
                ]
            )
            if self._latest_gnss_diagnostics is not None:
                diag = self._latest_gnss_diagnostics
                lines.append(f"GNSS local x/y: {diag.local_x:.2f}, {diag.local_y:.2f}")
                lines.append(f"Raw GNSS error: {diag.horizontal_error_m:.2f} m")
            elif self._gnss_projector is not None and self._gnss_projector.projection_error:
                lines.append(self._gnss_projector.projection_error)
        if imu is None:
            lines.append("IMU: waiting")
        else:
            ax, ay, az = imu.accelerometer
            gx, gy, gz = imu.gyroscope
            lines.extend(
                [
                    f"IMU frame: {imu.frame}",
                    f"Accel: {ax:+.2f}, {ay:+.2f}, {az:+.2f}",
                    f"Gyro: {gx:+.3f}, {gy:+.3f}, {gz:+.3f}",
                    f"Compass: {math.degrees(imu.compass):.2f} deg",
                ]
            )
        if lidar is None:
            lines.append("LiDAR: waiting")
        else:
            lines.extend([f"LiDAR frame: {lidar.frame}", f"LiDAR points: {lidar.point_count}"])
        return lines

    def _debug_tab_lines(self) -> list[str]:
        diagnostics = self._filter_manager.get_diagnostics() if self._filter_manager is not None else {}
        lines = [
            "Debug:",
            f"Route activation: {self._route_activation_state.value}",
            f"Stabilization ticks: {self._stabilization_stable_ticks}/{ROUTE_INITIALIZATION.stable_ticks_required}",
            f"Stabilization err: {self._format_optional_metric(self._stabilization_error_m, 'm')}",
            f"Stabilization time: {self._stabilization_elapsed_seconds:.1f}/{ROUTE_INITIALIZATION.max_wait_seconds:.1f}s",
            f"Route blocked: {'YES' if self._route_generation_blocked else 'NO'}",
            f"Planner: {self._planner_status}",
            f"Tracker closest/target: {self._latest_tracking.closest_index}/{self._latest_tracking.target_index}",
            f"Tracker search: {self._latest_tracking.search_start_index}-{self._latest_tracking.search_end_index}",
            f"Distance to goal: {self._format_optional_metric(self._latest_tracking.distance_to_goal_m, 'm')}",
            f"Heading error: {self._format_optional_metric(self._latest_tracking.heading_error_deg, 'deg')}",
            "Filter diagnostics:",
        ]
        for key, value in diagnostics.items():
            if key == "covariance_diagonal" and isinstance(value, list):
                shown = ", ".join(self._format_debug_value(item) for item in value[:6])
                lines.append(f"covariance diag: {shown}")
            elif key == "state_vector" and isinstance(value, list):
                shown = ", ".join(self._format_debug_value(item) for item in value[:6])
                lines.append(f"state: {shown}")
            else:
                lines.append(f"{key}: {self._format_debug_value(value)}")
        lines.append("Model-aware control diagnostics:")
        for key, value in self.autonomous_controller.latest_model_control_diagnostics.items():
            lines.append(f"{key}: {self._format_debug_value(value)}")
        return lines

    def _status_bar_text(self) -> str:
        logger = self._active_performance_logger()
        runner = self._test_runner
        benchmark_state = "active" if runner is not None and runner.is_active else "inactive"
        offline_state = "recording" if self._offline_recording_active() else "inactive"
        output = ""
        if runner is not None and runner.benchmark_folder is not None and not runner.is_active:
            output = f" | OUTPUT {runner.benchmark_folder.name}"
        if self._offline_recorder is not None and self._offline_recorder.run_folder is not None and not self._offline_recorder.is_active:
            output = f" | OUTPUT {self._offline_recorder.run_folder.name}"
        return (
            f"MODE {self._drive_mode.value} | "
            f"FILTER {self._active_filter_name()} | "
            f"TRACK {self._tracking_mode_for_metadata()} | "
            f"WORLD {self._active_map_display_name()} | "
            f"MAP {'ON' if self._map_selection_active else 'OFF'} | "
            f"ROUTE {self._selected_route_label()} | "
            f"ERR {self._format_optional_metric(logger.current_position_error_m, 'm')} | "
            f"GNSS {self._format_optional_metric(logger.current_raw_gnss_error_m, 'm')} | "
            f"CTE {self._format_optional_metric(logger.current_cross_track_error_m, 'm')} | "
            f"BENCH {benchmark_state}"
            f" | OFFLINE {offline_state}"
            f"{output}"
        )

    def _selected_route_label(self) -> str:
        store = self._test_route_store
        if store is None:
            return "none"
        route = store.get_current_route()
        if route is None:
            return "none"
        return f"{store.current_index + 1}/{store.route_count()} {route.name}"

    def _active_filter_name(self) -> str:
        if self._filter_manager is None:
            return "none"
        return self._filter_manager.get_active_filter_name()

    def _active_map_display_name(self) -> str:
        return display_map_name(self._active_map_name)

    def _active_filter_safe_for_autonomous(self) -> bool:
        if self._filter_manager is None:
            return False
        return self._filter_manager.active_filter_safe_for_autonomous_control()

    def _test_mode_active(self) -> bool:
        benchmark_active = bool(
            self._test_runner is not None
            and self._test_runner.is_active
            and self._test_runner.is_automated
        )
        return benchmark_active or self._offline_recording_active()

    def _offline_recording_active(self) -> bool:
        return bool(self._offline_recorder is not None and self._offline_recorder.is_active)

    def _offline_recording_warmup_active(self) -> bool:
        return bool(
            self._offline_recorder is not None
            and self._offline_recorder.is_active
            and self._offline_recorder.route_running
            and not self._offline_recorder.controller_enabled
        )

    def _active_filter_warning(self) -> str:
        if self._filter_manager is None:
            return ""
        tracking_message = self._filter_manager.tracking_mode_message
        if "unsupported" in tracking_message.lower():
            return tracking_message
        info = self._filter_manager.get_active_filter_info()
        note = str(info.get("autonomous_control_note") or "")
        if self._active_filter_safe_for_autonomous() and bool(info.get("experimental", False)):
            return note or "Selected filter is experimental; validate signs/tuning before relying on benchmark scores."
        if self._active_filter_safe_for_autonomous():
            return ""
        if note:
            return note
        filter_name = str(info.get("name") or self._filter_manager.active_filter_id or "Active filter")
        return f"{filter_name} is not marked safe for closed-loop autonomous control."

    @staticmethod
    def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        if numerator is None or denominator is None or denominator <= 0.0:
            return None
        return numerator / denominator

    @staticmethod
    def _format_debug_value(value: object) -> str:
        if isinstance(value, float):
            if not math.isfinite(value):
                return "n/a"
            return f"{value:.3g}"
        if isinstance(value, (list, tuple)):
            return ", ".join(SimulationApp._format_debug_value(item) for item in value[:4])
        if value is None:
            return "n/a"
        return str(value)

    def run(self) -> None:
        """Run the camera, route-selection, and route-following application loop."""
        try:
            self._setup()
            self._ensure_ready()

            running = True
            while running:
                running = self._process_events()
                if not running:
                    break
                self._client_manager.tick()
                frame_generation = self._world_context_generation
                if self._world_reload_in_progress or self._skip_frames_after_world_reload > 0:
                    self._skip_frames_after_world_reload = max(0, self._skip_frames_after_world_reload - 1)
                    self._draw_frame_without_camera()
                    self._clock.tick_pygame()
                    continue

                if self._ground_truth_provider is None or self._filter_manager is None:
                    self._draw_frame_without_camera()
                    self._clock.tick_pygame()
                    continue

                try:
                    self._latest_ground_truth_state = self._ground_truth_provider.get_state()
                    self._latest_estimated_state = self._filter_manager.update()
                    self._latest_localization_status = self._filter_manager.get_status(self._latest_ground_truth_state)
                except RuntimeError as exc:
                    self._planner_status = f"World context unavailable: {exc}"
                    self._control_status_text = self._planner_status
                    self._draw_frame_without_camera()
                    self._clock.tick_pygame()
                    continue
                self._update_route_activation_state()
                self._latest_state = self._state_for_tracking_and_control()
                if self._can_update_route_tracking() and self._latest_state is not None:
                    self._latest_tracking = self.waypoint_tracker.update(self._latest_state)
                self._update_sensor_diagnostics()
                if self._offline_recording_active():
                    world_reloaded = self._update_offline_recording()
                else:
                    world_reloaded = self._update_test_performance()
                if world_reloaded or frame_generation != self._world_context_generation:
                    self._clock.tick_pygame()
                    continue

                if (
                    self._vehicle is None
                    or self._manual_controller is None
                    or self._camera_sensor is None
                    or self._waypoint_manager is None
                    or self._display is None
                ):
                    self._draw_frame_without_camera()
                    self._clock.tick_pygame()
                    continue

                dt_seconds = self._clock.fixed_delta_seconds
                if self._route_activation_state == RouteActivationState.WAITING_FOR_LOCALIZATION_STABILITY:
                    control = carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=ROUTE_INITIALIZATION.hold_brake,
                        hand_brake=False,
                    )
                    if not self._apply_vehicle_control_safely(control):
                        self._draw_frame_without_camera()
                        self._clock.tick_pygame()
                        continue
                    self._set_latest_control(control, control)
                    # Keep active prediction dormant until route activation so
                    # the stationary hold brake cannot bias filter startup.
                    self.actuator_realism.reset(control)
                elif self._drive_mode == DriveMode.AUTONOMOUS:
                    if self._offline_recording_warmup_active():
                        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=False)
                        if not self._apply_vehicle_control_safely(control):
                            self._draw_frame_without_camera()
                            self._clock.tick_pygame()
                            continue
                        self._set_latest_control(control, control)
                        self.actuator_realism.reset(control)
                        camera_surface = self._camera_sensor.get_latest_surface()
                        self._display.begin_frame(camera_surface)
                        self._draw_topdown_map()
                        self._draw_lidar_panel()
                        self._draw_driving_behavior_panels()
                        self._draw_control_panel()
                        self._draw_status_bar()
                        self._display.set_test_mode_titles(self._test_mode_active())
                        self._display.end_frame()
                        self._clock.tick_pygame()
                        continue
                    control_state = self._state_for_tracking_and_control()
                    if control_state is None:
                        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
                        if not self._apply_vehicle_control_safely(control):
                            self._draw_frame_without_camera()
                            self._clock.tick_pygame()
                            continue
                        self._set_latest_control(control, control)
                        if not self._offline_recording_active():
                            self._feed_filter_control_input(control, source="autonomous_safety_brake")
                        self.actuator_realism.reset(control)
                    else:
                        preview_waypoints = self.waypoint_tracker.get_preview_waypoints(max_count=90)
                        no_active_target = self._latest_tracking.target_waypoint is None
                        self._latest_speed_plan = self.speed_planner.plan(
                            state=control_state,
                            preview_waypoints=preview_waypoints,
                            route_completed=self._latest_tracking.completed or no_active_target,
                            dt_seconds=dt_seconds,
                        )
                        control = self.autonomous_controller.compute_control(
                            state=control_state,
                            target_waypoint=self._latest_tracking.target_waypoint,
                            route_completed=self._latest_tracking.completed,
                            target_speed_mps=self._latest_speed_plan.target_speed_mps,
                        )
                        applied_control = self.actuator_realism.apply(control, dt_seconds)
                        if not self._apply_vehicle_control_safely(applied_control):
                            self._draw_frame_without_camera()
                            self._clock.tick_pygame()
                            continue
                        self._set_latest_control(control, applied_control)
                        if not self._offline_recording_active():
                            self._feed_filter_control_input(applied_control, source="autonomous_applied")
                else:
                    try:
                        control = self._manual_controller.apply_control()
                    except RuntimeError as exc:
                        self._planner_status = f"Manual control skipped: {exc}"
                        self._control_status_text = self._planner_status
                        self._draw_frame_without_camera()
                        self._clock.tick_pygame()
                        continue
                    self._set_latest_control(control, control)
                    self.actuator_realism.reset(control)

                camera_surface = self._camera_sensor.get_latest_surface()
                self._display.begin_frame(camera_surface)
                try:
                    self._draw_camera_waypoints()
                except RuntimeError as exc:
                    self._planner_status = f"Camera overlay skipped: {exc}"
                    self._control_status_text = self._planner_status
                self._draw_topdown_map()
                self._draw_lidar_panel()
                self._draw_driving_behavior_panels()
                self._draw_control_panel()
                self._draw_status_bar()
                self._display.set_test_mode_titles(self._test_mode_active())
                self._display.end_frame()
                self._clock.tick_pygame()
        finally:
            self.shutdown()

    def _draw_frame_without_camera(self) -> None:
        """Render UI panels without touching camera, vehicle, or sensor actors."""
        if self._display is None:
            return
        self._display.begin_frame(None)
        self._draw_topdown_map(update_sensor_diagnostics=False)
        self._draw_driving_behavior_panels()
        self._draw_control_panel()
        self._draw_status_bar()
        self._display.set_test_mode_titles(self._test_mode_active())
        self._display.end_frame()

    def _apply_vehicle_control_safely(
        self,
        control: "carla.VehicleControl",
    ) -> bool:
        if self._world_reload_in_progress or self._vehicle is None:
            self._planner_status = "Vehicle control skipped during world reload"
            self._control_status_text = self._planner_status
            return False
        try:
            self._vehicle.apply_control(control)
            return True
        except RuntimeError as exc:
            self._planner_status = f"Vehicle control skipped: {exc}"
            self._control_status_text = self._planner_status
            return False

    def _state_for_tracking_and_control(self) -> Optional[VehicleState]:
        """Return GT in manual mode and the active filter state in autonomous mode."""
        if self._offline_recording_active():
            return self._latest_ground_truth_state
        if self._drive_mode == DriveMode.AUTONOMOUS:
            return self._latest_estimated_state
        return self._latest_ground_truth_state

    def _set_latest_control(
        self,
        requested_control: "carla.VehicleControl",
        applied_control: "carla.VehicleControl",
    ) -> None:
        self._latest_requested_control = carla.VehicleControl(
            throttle=float(getattr(requested_control, "throttle", 0.0)),
            steer=float(getattr(requested_control, "steer", 0.0)),
            brake=float(getattr(requested_control, "brake", 0.0)),
            hand_brake=bool(getattr(requested_control, "hand_brake", False)),
            reverse=bool(getattr(requested_control, "reverse", False)),
            manual_gear_shift=bool(getattr(requested_control, "manual_gear_shift", False)),
        )
        self._latest_applied_control = carla.VehicleControl(
            throttle=float(getattr(applied_control, "throttle", 0.0)),
            steer=float(getattr(applied_control, "steer", 0.0)),
            brake=float(getattr(applied_control, "brake", 0.0)),
            hand_brake=bool(getattr(applied_control, "hand_brake", False)),
            reverse=bool(getattr(applied_control, "reverse", False)),
            manual_gear_shift=bool(getattr(applied_control, "manual_gear_shift", False)),
        )

    def _feed_filter_control_input(self, applied_control: "carla.VehicleControl", source: str) -> None:
        if self._filter_manager is None:
            return
        # Avoid ground-truth leakage: active-tracking filters may use this as a
        # control input, so speed/yaw come only from the current estimate.
        state = self._latest_estimated_state
        timestamp = self._filter_control_timestamp(state)
        control_input = FilterControlInput(
            timestamp=float(timestamp),
            throttle=float(getattr(applied_control, "throttle", 0.0)),
            steer=float(getattr(applied_control, "steer", 0.0)),
            brake=float(getattr(applied_control, "brake", 0.0)),
            hand_brake=bool(getattr(applied_control, "hand_brake", False)),
            reverse=bool(getattr(applied_control, "reverse", False)),
            source=source,
            speed_mps=float(state.speed) if state is not None else None,
            yaw_deg=float(state.yaw) if state is not None else None,
        )
        self._filter_manager.process_control(control_input)

    def _filter_control_timestamp(self, estimated_state: Optional[VehicleState]) -> float:
        if estimated_state is not None:
            return float(estimated_state.timestamp)
        try:
            return float(self._client_manager.world.get_snapshot().timestamp.elapsed_seconds)
        except Exception:
            return time.monotonic()

    def _reset_driving_behavior(
        self,
        initial_speed_mps: Optional[float] = None,
        actuator_control: Optional["carla.VehicleControl"] = None,
    ) -> None:
        initial_speed = (
            float(initial_speed_mps)
            if initial_speed_mps is not None
            else self._latest_ground_truth_state.speed
            if self._latest_ground_truth_state is not None
            else 0.0
        )
        self.speed_planner.reset(initial_speed_mps=initial_speed)
        self._latest_speed_plan = self.speed_planner.latest_plan
        self.actuator_realism.reset(actuator_control if actuator_control is not None else self._latest_applied_control)

    def _can_update_route_tracking(self) -> bool:
        if self._route_activation_state == RouteActivationState.WAITING_FOR_LOCALIZATION_STABILITY:
            return False
        return self.route_planner is not None and bool(self.route_planner.get_route())

    def _process_events(self) -> bool:
        assert self._control_panel is not None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.VIDEORESIZE:
                self._handle_video_resize(event)
                continue
            if (
                not self._test_mode_active()
                and self._control_panel.active_tab == "Filters"
                and self._handle_filters_tab_event(event)
            ):
                continue
            if (
                not self._test_mode_active()
                and self._control_panel.active_tab == "Sensors"
                and self._sensor_noise_panel is not None
                and self._sensor_noise_panel.handle_event(event)
            ):
                continue
            if (
                not self._test_mode_active()
                and self._behavior_tuning_panel is not None
                and self._behavior_tuning_panel.handle_event(event)
            ):
                continue
            if event.type == pygame.KEYDOWN:
                if not self._handle_key_down(event):
                    return False
            elif self._test_mode_active():
                continue
            elif self._control_panel.handle_event(event):
                continue
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_button_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._handle_mouse_button_up(event)
            elif event.type == pygame.MOUSEMOTION:
                self._handle_mouse_motion(event)
            elif event.type == pygame.MOUSEWHEEL:
                self._handle_mouse_wheel(event)
        return True

    def _handle_video_resize(self, event: pygame.event.Event) -> None:
        assert self._display is not None
        assert self._control_panel is not None
        self._display.resize(event.w, event.h)
        self._control_panel.set_rect(self._display.control_panel_rect)
        if self._behavior_tuning_panel is not None:
            self._behavior_tuning_panel.set_rect(self._display.behavior_tuning_rect)
        if self._test_progress_panel is not None:
            self._test_progress_panel.set_rect(self._display.behavior_tuning_rect)
        if self._live_evaluation_panel is not None:
            self._live_evaluation_panel.set_rect(self._display.control_panel_rect)
        if self._control_visual_widget is not None:
            self._control_visual_widget.set_rect(self._display.control_visual_rect)
        if self._driving_diagnostics_widget is not None:
            self._driving_diagnostics_widget.set_rect(self._display.driving_state_rect)

    def _handle_key_down(self, event: pygame.event.Event) -> bool:
        if event.key == pygame.K_ESCAPE:
            return False
        if event.key == pygame.K_m:
            self._set_manual_mode()
            return True
        if event.key == pygame.K_p:
            self._try_enable_autonomous_mode()
            return True
        if event.key == pygame.K_t:
            self._toggle_map_selection()
            return True
        if event.key == pygame.K_r:
            self._reset_selection_and_route()
            return True
        if event.key == pygame.K_c:
            self._clear_route()
            return True
        if event.key == pygame.K_g:
            self._generate_route_from_selection(teleport_to_start=True, start_autonomous=True)
            return True
        return True

    def _set_manual_mode(self) -> None:
        if self._test_runner is not None and self._test_runner.is_active:
            self._test_runner.stop(aborted=True, reason="Test aborted: manual mode")
        self._stop_offline_recording_if_active("Offline recording aborted: manual mode")
        self._cancel_route_activation()
        self._drive_mode = DriveMode.MANUAL
        if self._vehicle is not None:
            self._vehicle.set_autopilot(False)
        self._planner_status = "Mode: manual"
        self._control_status_text = "Manual mode"
        self._reset_driving_behavior()

    def _toggle_map_selection(self) -> None:
        self._map_selection_active = not self._map_selection_active
        self._control_status_text = f"Map select {'ON' if self._map_selection_active else 'OFF'}"

    def _toggle_test_route_authoring(self) -> None:
        self._test_route_authoring_active = not self._test_route_authoring_active
        if self._test_route_authoring_active:
            self._map_selection_active = True
            self._clear_route()
            self._planner_status = "Test route mode: select A/B"
            self._control_status_text = "Test route mode: select A/B"
        else:
            self._planner_status = "Test route mode off"
            self._control_status_text = "Test route mode off"

    def _save_current_ab_as_test_route(self) -> None:
        if self._map_selector is None or self._test_route_store is None:
            self._planner_status = "Test route store unavailable"
            self._control_status_text = self._planner_status
            return

        endpoints = self._map_selector.endpoints
        if endpoints is None:
            self._planner_status = "Select A/B before saving test route"
            self._control_status_text = self._planner_status
            return

        route = self._test_route_store.add_route_from_endpoints(None, endpoints)
        self._planner_status = f"Saved test route: {route.name}"
        self._control_status_text = self._planner_status

    def _select_next_test_route(self) -> None:
        if self._test_route_store is None:
            self._control_status_text = "Test route store unavailable"
            return
        route = self._test_route_store.next_route()
        if route is None:
            self._planner_status = "No saved test routes"
            self._control_status_text = self._planner_status
            return
        self._planner_status = f"Selected test route: {route.name}"
        self._control_status_text = self._planner_status

    def _select_previous_test_route(self) -> None:
        if self._test_route_store is None:
            self._control_status_text = "Test route store unavailable"
            return
        route = self._test_route_store.previous_route()
        if route is None:
            self._planner_status = "No saved test routes"
            self._control_status_text = self._planner_status
            return
        self._planner_status = f"Selected test route: {route.name}"
        self._control_status_text = self._planner_status

    def _load_selected_test_route(self) -> None:
        if self._test_route_store is None or self._map_selector is None:
            self._planner_status = "Test route tools unavailable"
            self._control_status_text = self._planner_status
            return

        route = self._test_route_store.get_current_route()
        if route is None:
            self._planner_status = "No saved test routes"
            self._control_status_text = self._planner_status
            return
        if not self._test_route_store.route_is_compatible(route):
            self._planner_status = "Saved route blocked: wrong map"
            self._control_status_text = self._planner_status
            return

        resolved = self._test_route_store.resolve_route_to_waypoints(
            self._client_manager.world_map,
            route,
        )
        if resolved is None:
            self._planner_status = "Failed to resolve saved test route"
            self._control_status_text = self._planner_status
            return

        start_waypoint, goal_waypoint = resolved
        self._clear_route()
        self._map_selector.set_endpoints(start_waypoint, goal_waypoint)
        self._map_selection_active = True
        self._planner_status = f"Loaded test route: {route.name}"
        self._control_status_text = self._planner_status

    def _start_test_run(self) -> None:
        if self._test_route_store is None or self._test_runner is None:
            self._planner_status = "Test runner unavailable"
            self._control_status_text = self._planner_status
            return
        if not self._active_filter_safe_for_autonomous():
            self._planner_status = "Active filter is unsafe for autonomous benchmark control"
            self._control_status_text = self._active_filter_warning()
            return

        route = self._test_route_store.get_current_route()
        if route is None:
            self._planner_status = "Select or create a test route first"
            self._control_status_text = self._planner_status
            return
        if not self._test_route_store.route_is_compatible(route):
            self._planner_status = "Benchmark blocked: selected route is for another map"
            self._control_status_text = self._planner_status
            return

        self._test_route_authoring_active = False
        self._map_selection_active = False
        started = self._test_runner.start_selected_route(route)
        self._planner_status = self._test_runner.status_text
        self._control_status_text = self._test_runner.status_text
        if not started:
            self._drive_mode = DriveMode.MANUAL

    def _stop_test_run(self) -> None:
        if self._test_runner is None or not self._test_runner.is_active:
            self._planner_status = "No active test run"
            self._control_status_text = self._planner_status
            return

        self._test_runner.stop(aborted=True, reason="Test stopped")
        self._clear_route(stop_test=False)
        self._planner_status = "Test stopped"
        self._control_status_text = "Test stopped"

    def _reset_estimator(self) -> None:
        if self._filter_manager is None:
            self._control_status_text = "Estimator not initialized"
            return

        self._filter_manager.reset(skip_current_sensor_frames=True)
        self._clear_localization_after_filter_reset()
        self._hold_vehicle_after_filter_reset()
        self._planner_status = "Estimator reset"
        self._control_status_text = "Estimator reset"

    def _clear_localization_after_filter_reset(self) -> None:
        self._latest_estimated_state = None
        self._latest_localization_status = None
        self._latest_gnss_diagnostics = None
        self._latest_gnss_frame = None
        self._gnss_trail_xy.clear()

    def _hold_vehicle_after_filter_reset(self) -> None:
        if self._drive_mode != DriveMode.AUTONOMOUS or self._vehicle is None:
            return
        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=False)
        try:
            self._vehicle.apply_control(control)
        except RuntimeError:
            return
        self._set_latest_control(control, control)
        self.actuator_realism.reset(control)

    def _respawn_benchmark_localization_sensors(self) -> None:
        """Recreate GNSS/IMU actors so every benchmark attempt starts from the selected noise config."""
        if self._world_reload_in_progress or self._skip_frames_after_world_reload > 0:
            return
        try:
            if self._gnss_sensor is not None:
                self._gnss_sensor.apply_config(self.sensor_noise_config, respawn=True)
            if self._imu_sensor is not None:
                self._imu_sensor.apply_config(self.sensor_noise_config, respawn=True)
            self._latest_gnss_diagnostics = None
            self._latest_gnss_frame = None
            self._gnss_trail_xy.clear()
        except RuntimeError as exc:
            self._last_sensor_apply_status = f"Benchmark sensor respawn failed: {exc}"

    def _reset_benchmark_failure_monitor(self) -> None:
        self._failure_monitor_last_progress_time = None
        self._failure_monitor_last_position = None
        self._failure_monitor_last_distance_to_goal = None
        self._failure_monitor_last_closest_index = 0
        self._failure_monitor_deviation_started = None

    def _emergency_brake(self) -> None:
        if self._test_runner is not None and self._test_runner.is_active:
            self._test_runner.stop(aborted=True, reason="Test aborted: emergency brake")
        self._stop_offline_recording_if_active("Offline recording aborted: emergency brake")
        self._cancel_route_activation()
        self._drive_mode = DriveMode.MANUAL
        if self._vehicle is not None:
            self._vehicle.set_autopilot(False)
            self._vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0,
                    hand_brake=True,
                )
            )
        self._planner_status = "Emergency brake"
        self._control_status_text = "Emergency brake"
        self._reset_driving_behavior()

    def _save_test_report(self) -> None:
        if self._test_runner is not None and self._test_runner.benchmark_folder is not None:
            queued = self._plot_worker.enqueue_route_plots(self._test_runner.benchmark_folder)
            self._planner_status = "Plots queued" if queued else "Plot queue unavailable"
            self._control_status_text = self._planner_status
            return

        logger = self._active_performance_logger()
        if not logger.samples:
            self._planner_status = "No test samples to export"
            self._control_status_text = self._planner_status
            return

        _csv_path, json_path = logger.export()
        queued = self._plot_worker.enqueue_route_plots(json_path.parent)
        suffix = ", plots queued" if queued else ", plot queue unavailable"
        self._planner_status = f"Saved test report: {json_path.name}{suffix}"
        self._control_status_text = self._planner_status

    def _regenerate_plots(self) -> None:
        if self._test_runner is None:
            self._planner_status = "Test runner unavailable"
            self._control_status_text = self._planner_status
            return
        if self._test_runner.regenerate_plots():
            self._planner_status = self._test_runner.status_text
        else:
            self._planner_status = self._test_runner.status_text
        self._control_status_text = self._planner_status

    def _handle_mouse_button_down(self, event: pygame.event.Event) -> None:
        assert self._display is not None
        renderer = self._topdown_renderer
        if renderer is None or not self._map_selection_active:
            return

        consumed = renderer.handle_mouse_button_down(
            self._display.surface,
            event,
            panel_rect=self._display.map_rect,
        )
        if consumed:
            return

        if event.button != 1:
            return

        world_location = renderer.screen_to_world(
            self._display.surface,
            event.pos,
            panel_rect=self._display.map_rect,
        )
        if world_location is None:
            return

        assert self._map_selector is not None
        self._map_selector.select_world_location(world_location)
        self._drive_mode = DriveMode.MANUAL
        self._clear_route()
        if self._test_route_authoring_active:
            if self._map_selector.endpoints is not None:
                self._planner_status = "Test route ready: click Save A/B"
                self._control_status_text = "Test route ready: click Save A/B"
            else:
                self._planner_status = "Test route mode: select A/B"
                self._control_status_text = "Test route mode: select A/B"
            return
        if self._map_selector.endpoints is not None:
            self._begin_route_initialization_from_selection(start_autonomous=True)

    def _handle_mouse_button_up(self, event: pygame.event.Event) -> None:
        if self._topdown_renderer is not None:
            self._topdown_renderer.handle_mouse_button_up(event)

    def _handle_mouse_motion(self, event: pygame.event.Event) -> None:
        assert self._display is not None
        if self._topdown_renderer is not None and self._map_selection_active:
            self._topdown_renderer.handle_mouse_motion(
                self._display.surface,
                event,
                panel_rect=self._display.map_rect,
            )

    def _handle_mouse_wheel(self, event: pygame.event.Event) -> None:
        assert self._display is not None
        if self._topdown_renderer is None or not self._map_selection_active:
            return
        self._topdown_renderer.handle_mouse_wheel(
            surface=self._display.surface,
            position=pygame.mouse.get_pos(),
            wheel_y=event.y,
            panel_rect=self._display.map_rect,
        )

    def _try_enable_autonomous_mode(self) -> None:
        if self.route_planner is None:
            return
        if not self._active_filter_safe_for_autonomous():
            self._planner_status = "Autonomous blocked: active filter is unsafe"
            self._control_status_text = self._active_filter_warning()
            return
        if self._route_activation_state == RouteActivationState.WAITING_FOR_LOCALIZATION_STABILITY:
            self._pending_start_autonomous = True
            self._drive_mode = DriveMode.AUTONOMOUS
            return
        if not self.route_planner.get_route():
            self._begin_route_initialization_from_selection(start_autonomous=True)
            return
        if self.route_planner.get_route():
            self._drive_mode = DriveMode.AUTONOMOUS
            if self._vehicle is not None:
                self._vehicle.set_autopilot(False)
            self._reset_driving_behavior()

    def _begin_route_initialization_from_selection(self, start_autonomous: bool) -> None:
        if self._map_selector is None:
            return
        if start_autonomous and not self._active_filter_safe_for_autonomous():
            self._planner_status = "Autonomous blocked: active filter is unsafe"
            self._control_status_text = self._active_filter_warning()
            return

        endpoints = self._map_selector.endpoints
        if endpoints is None:
            self._planner_status = "Planner: select A/B"
            return

        self._begin_route_initialization(
            start_waypoint=endpoints.start,
            goal_waypoint=endpoints.goal,
            start_autonomous=start_autonomous,
        )

    def _begin_route_initialization(
        self,
        start_waypoint: "carla.Waypoint",
        goal_waypoint: "carla.Waypoint",
        start_autonomous: bool,
    ) -> None:
        if self.route_planner is not None:
            self.route_planner.clear_route()
        self.waypoint_tracker.clear_route()
        self._latest_tracking = self._empty_tracking_status()
        self._reset_benchmark_failure_monitor()
        if self._test_mode_active():
            self._respawn_benchmark_localization_sensors()

        self._pending_start_waypoint = start_waypoint
        self._pending_goal_waypoint = goal_waypoint
        self._pending_start_autonomous = start_autonomous
        self._route_activation_state = RouteActivationState.WAITING_FOR_LOCALIZATION_STABILITY
        self._drive_mode = DriveMode.AUTONOMOUS if start_autonomous else DriveMode.MANUAL
        self._stabilization_started_monotonic = time.monotonic()
        self._stabilization_elapsed_seconds = 0.0
        self._stabilization_stable_ticks = 0
        self._stabilization_error_m = None
        self._stabilization_timed_out = False
        self._route_generation_blocked = True
        self._planner_status = "Planner: waiting localization stability"
        self._teleport_vehicle_to_route_start(start_waypoint)
        self._reset_driving_behavior()

    def _begin_offline_recording_route(
        self,
        start_waypoint: "carla.Waypoint",
        goal_waypoint: "carla.Waypoint",
        route_waypoints: Sequence["carla.Waypoint"],
    ) -> None:
        del goal_waypoint
        if self.route_planner is not None:
            self.route_planner.set_route(route_waypoints)
        self.waypoint_tracker.set_route(route_waypoints)
        self._latest_tracking = self._empty_tracking_status()
        self._reset_benchmark_failure_monitor()
        self._pending_start_waypoint = None
        self._pending_goal_waypoint = None
        self._pending_start_autonomous = False
        self._route_activation_state = RouteActivationState.ROUTE_ACTIVE
        self._drive_mode = DriveMode.AUTONOMOUS
        self._route_generation_blocked = False
        self._planner_status = "Offline recording: ground-truth controller active"
        self._respawn_benchmark_localization_sensors()
        self._teleport_vehicle_to_route_start(start_waypoint)
        if self.route_planner is not None:
            self.route_planner.set_route(route_waypoints)
        self.waypoint_tracker.set_route(route_waypoints)
        self._route_activation_state = RouteActivationState.ROUTE_ACTIVE
        self._reset_driving_behavior()

    def _generate_route_from_selection(
        self,
        teleport_to_start: bool = True,
        start_autonomous: bool = False,
    ) -> None:
        if teleport_to_start:
            self._begin_route_initialization_from_selection(start_autonomous=start_autonomous)
            return
        if self._map_selector is None or self.route_planner is None:
            return

        endpoints = self._map_selector.endpoints
        if endpoints is None:
            self._planner_status = "Planner: select A/B"
            return

        route = self.route_planner.generate_route(endpoints.start, endpoints.goal)
        self.waypoint_tracker.set_route(route)
        if route:
            if start_autonomous:
                self._drive_mode = DriveMode.AUTONOMOUS
            self._planner_status = f"Planner: {len(route)} wp, driving A->B"
        elif self.route_planner.planner_error:
            self._planner_status = self.route_planner.planner_error
        else:
            self._planner_status = "Planner: no route"

    def _update_route_activation_state(self) -> None:
        if self._route_activation_state != RouteActivationState.WAITING_FOR_LOCALIZATION_STABILITY:
            self._route_generation_blocked = False
            return

        self._route_generation_blocked = True
        self._stabilization_elapsed_seconds = self._elapsed_stabilization_seconds()
        self._stabilization_error_m = self._latest_position_error_m()

        stable_now = self._localization_is_stable_for_route_start()
        if stable_now:
            self._stabilization_stable_ticks += 1
        else:
            self._stabilization_stable_ticks = 0

        stable_enough = self._stabilization_stable_ticks >= ROUTE_INITIALIZATION.stable_ticks_required
        timeout_ready = self._stabilization_timeout_ready()
        if stable_enough or timeout_ready:
            self._stabilization_timed_out = timeout_ready and not stable_enough
            self._activate_pending_route_after_stabilization()

    def _activate_pending_route_after_stabilization(self) -> None:
        if (
            self.route_planner is None
            or self._pending_start_waypoint is None
            or self._pending_goal_waypoint is None
        ):
            self._planner_status = "Planner: stabilization missing endpoints"
            self._route_activation_state = RouteActivationState.IDLE
            self._route_generation_blocked = False
            return

        route = self.route_planner.generate_route(
            self._pending_start_waypoint,
            self._pending_goal_waypoint,
        )
        self.waypoint_tracker.set_route(route)
        self._route_generation_blocked = False
        if route:
            self._route_activation_state = RouteActivationState.ROUTE_ACTIVE
            if self._pending_start_autonomous:
                self._drive_mode = DriveMode.AUTONOMOUS
                if self._vehicle is not None:
                    self._vehicle.set_autopilot(False)
                if self._lightweight_closed_loop_auto_tune:
                    neutral = self._neutral_vehicle_control()
                    self._set_latest_control(neutral, neutral)
                    self._reset_driving_behavior(actuator_control=neutral)
                else:
                    self._reset_driving_behavior()
            status_suffix = "timeout" if self._stabilization_timed_out else "stable"
            self._planner_status = f"Planner: {len(route)} wp, {status_suffix}, driving A->B"
        elif self.route_planner.planner_error:
            self._route_activation_state = RouteActivationState.IDLE
            self._planner_status = self.route_planner.planner_error
        else:
            self._route_activation_state = RouteActivationState.IDLE
            self._planner_status = "Planner: no route"

        self._pending_start_waypoint = None
        self._pending_goal_waypoint = None

    def _localization_is_stable_for_route_start(self) -> bool:
        error = self._latest_position_error_m()
        estimate = self._latest_estimated_state
        status = self._latest_localization_status
        if error is None or estimate is None or status is None or not status.initialized:
            return False
        return (
            error <= ROUTE_INITIALIZATION.position_error_threshold_m
            and estimate.speed <= ROUTE_INITIALIZATION.estimated_speed_threshold_mps
        )

    def _stabilization_timeout_ready(self) -> bool:
        error = self._latest_position_error_m()
        estimate = self._latest_estimated_state
        if error is None or estimate is None:
            return False
        return (
            self._stabilization_elapsed_seconds >= ROUTE_INITIALIZATION.max_wait_seconds
            and error <= ROUTE_INITIALIZATION.timeout_position_error_threshold_m
        )

    def _elapsed_stabilization_seconds(self) -> float:
        if self._stabilization_started_monotonic is None:
            return 0.0
        return max(0.0, time.monotonic() - self._stabilization_started_monotonic)

    def _latest_position_error_m(self) -> Optional[float]:
        if self._latest_localization_status is None:
            return None
        return self._latest_localization_status.position_error_m

    def _cancel_route_activation(self) -> None:
        self._route_activation_state = RouteActivationState.IDLE
        self._pending_start_waypoint = None
        self._pending_goal_waypoint = None
        self._pending_start_autonomous = False
        self._stabilization_started_monotonic = None
        self._stabilization_elapsed_seconds = 0.0
        self._stabilization_stable_ticks = 0
        self._stabilization_error_m = None
        self._stabilization_timed_out = False
        self._route_generation_blocked = False

    def _teleport_vehicle_to_route_start(self, start_waypoint: "carla.Waypoint") -> None:
        if self._vehicle_manager is None:
            return
        self._vehicle_manager.teleport_to_waypoint(start_waypoint)
        if self._filter_manager is not None:
            self._filter_manager.reset(skip_current_sensor_frames=True)
            self._clear_localization_after_filter_reset()
        if self._ground_truth_provider is not None:
            self._latest_ground_truth_state = self._ground_truth_provider.get_state()
            self._latest_state = self._latest_ground_truth_state
        if self._filter_manager is not None:
            self._latest_localization_status = self._filter_manager.get_status(self._latest_ground_truth_state)

    def _reset_selection_and_route(self) -> None:
        if self._map_selector is not None:
            self._map_selector.reset()
        self._clear_route()
        self._planner_status = "Planner: reset"

    def _clear_route(self, stop_test: bool = True) -> None:
        if stop_test and self._test_runner is not None and self._test_runner.is_active:
            self._test_runner.stop(aborted=True, reason="Test aborted: route cleared")
        if stop_test:
            self._stop_offline_recording_if_active("Offline recording aborted: route cleared")
        self._cancel_route_activation()
        if self.route_planner is not None:
            self.route_planner.clear_route()
        self.waypoint_tracker.clear_route()
        self._latest_tracking = self._empty_tracking_status()
        self._reset_benchmark_failure_monitor()
        self._drive_mode = DriveMode.MANUAL
        self._reset_driving_behavior()

    def _stop_offline_recording_if_active(self, reason: str) -> None:
        if self._offline_recorder is not None and self._offline_recorder.is_active:
            self._offline_recorder.stop(aborted=True, reason=reason)

    def _switch_map_for_benchmark(self) -> WorldSwitchResult:
        runner = self._test_runner
        if runner is None or not runner.is_automated:
            return WorldSwitchResult.NOOP
        map_name = runner.required_map_name()
        if not map_name:
            return WorldSwitchResult.NOOP
        load_name = self._client_manager.resolve_map_load_name(map_name)
        runner_state_text = f"Loading map: {display_map_name(load_name)}"
        self._planner_status = runner_state_text
        self._control_status_text = runner_state_text
        self._world_context_generation += 1
        self._world_reload_in_progress = True
        try:
            self._destroy_world_actors_for_reload()
            self._client_manager.load_world(load_name)
            self._rebuild_world_context_after_map_load(selected_map_load_name=load_name)
            self._world_context_generation += 1
            runner.update_world_context(
                world_map=self._client_manager.world_map,
                route_store=self._test_route_store,
                selected_map_load_name=map_name,
            )
            self._drive_mode = DriveMode.AUTONOMOUS
            self._skip_frames_after_world_reload = 1
            self._planner_status = runner.status_text
            self._control_status_text = runner.status_text
            return WorldSwitchResult.SWITCHED
        except Exception as exc:
            runner.stop(aborted=True, reason=f"Map switch failed: {exc}")
            self._planner_status = runner.status_text
            self._control_status_text = runner.status_text
            return WorldSwitchResult.FAILED
        finally:
            self._world_reload_in_progress = False

    def _switch_map_for_offline_recording(self) -> WorldSwitchResult:
        recorder = self._offline_recorder
        if recorder is None or not recorder.is_active:
            return WorldSwitchResult.NOOP
        map_name = recorder.required_map_name()
        if not map_name:
            return WorldSwitchResult.NOOP
        load_name = self._client_manager.resolve_map_load_name(map_name)
        status = f"Loading map for offline recording: {display_map_name(load_name)}"
        self._planner_status = status
        self._control_status_text = status
        self._world_context_generation += 1
        self._world_reload_in_progress = True
        try:
            self._destroy_world_actors_for_reload()
            self._client_manager.load_world(load_name)
            self._rebuild_world_context_after_map_load(selected_map_load_name=load_name)
            self._world_context_generation += 1
            recorder.update_world_context(
                world_map=self._client_manager.world_map,
                route_store=self._test_route_store,
                selected_map_load_name=map_name,
            )
            self._drive_mode = DriveMode.AUTONOMOUS
            self._skip_frames_after_world_reload = 1
            self._planner_status = recorder.status_text
            self._control_status_text = recorder.status_text
            return WorldSwitchResult.SWITCHED
        except Exception as exc:
            recorder.stop(aborted=True, reason=f"Offline recording map switch failed: {exc}")
            self._planner_status = recorder.status_text
            self._control_status_text = recorder.status_text
            return WorldSwitchResult.FAILED
        finally:
            self._world_reload_in_progress = False

    def _destroy_world_actors_for_reload(self) -> None:
        if self._vehicle is not None:
            try:
                self._vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            except RuntimeError:
                pass
        if self._sensor_manager is not None:
            self._sensor_manager.destroy_all()
        if self._vehicle_manager is not None:
            self._vehicle_manager.destroy()
        self._manual_controller = None
        self._waypoint_manager = None
        self._ground_truth_provider = None
        self._filter_manager = None
        self._gnss_projector = None
        self._map_selector = None
        self.route_planner = None
        self._topdown_renderer = None
        self._test_route_store = None
        self._sensor_manager = None
        self._camera_sensor = None
        self._gnss_sensor = None
        self._imu_sensor = None
        self._lidar_sensor = None
        self._vehicle_manager = None
        self._vehicle = None
        self._latest_state = None
        self._latest_ground_truth_state = None
        self._latest_estimated_state = None
        self._latest_localization_status = None
        self._latest_gnss_diagnostics = None
        self._latest_gnss_frame = None
        self._gnss_trail_xy.clear()
        self.waypoint_tracker.clear_route()
        self._latest_tracking = self._empty_tracking_status()
        self._cancel_route_activation()
        self._reset_benchmark_failure_monitor()

    def _rebuild_world_context_after_map_load(self, selected_map_load_name: Optional[str]) -> None:
        self._selected_map_load_name = selected_map_load_name
        self._active_map_name = getattr(self._client_manager.world_map, "name", None)
        self._active_map_id = normalize_map_name(self._active_map_name)
        self._vehicle_manager = VehicleManager(
            world=self._client_manager.world,
            world_map=self._client_manager.world_map,
            blueprint_library=self._client_manager.blueprint_library,
        )
        self._vehicle = self._vehicle_manager.spawn_vehicle()
        self._sensor_manager = SensorManager(
            world=self._client_manager.world,
            blueprint_library=self._client_manager.blueprint_library,
        )
        if self._lightweight_closed_loop_auto_tune:
            self._camera_sensor = None
        else:
            self._camera_sensor = self._sensor_manager.create_rgb_camera(attach_to=self._vehicle)
        self._gnss_sensor = self._sensor_manager.create_gnss(
            attach_to=self._vehicle,
            config=self.sensor_noise_config,
        )
        self._imu_sensor = self._sensor_manager.create_imu(
            attach_to=self._vehicle,
            config=self.sensor_noise_config,
        )
        if self._lightweight_closed_loop_auto_tune:
            self._lidar_sensor = None
        else:
            self._lidar_sensor = self._sensor_manager.create_lidar(attach_to=self._vehicle)
        self._waypoint_manager = WaypointManager(world_map=self._client_manager.world_map)
        self._manual_controller = ManualController(vehicle=self._vehicle)
        self._ground_truth_provider = GroundTruthStateProvider(vehicle=self._vehicle)
        self._gnss_projector = GnssLocalProjector(world_map=self._client_manager.world_map)
        tune_overrides = {}
        if self._startup_benchmark_config is not None:
            tune_overrides[self._startup_benchmark_config.selected_filter] = dict(
                self._startup_benchmark_config.selected_filter_tune or {}
            )
        elif self._startup_closed_loop_auto_tune_request is not None:
            tune_overrides[self._startup_closed_loop_auto_tune_request.filter_id] = dict(
                self._startup_closed_loop_auto_tune_request.base_tune or {}
            )
        default_filter_id = "ca_kf"
        tracking_mode = TRACKING_MODE_PASSIVE
        if self._startup_benchmark_config is not None:
            default_filter_id = self._startup_benchmark_config.selected_filter
            tracking_mode = self._startup_benchmark_config.tracking_mode
        elif self._startup_closed_loop_auto_tune_request is not None:
            default_filter_id = self._startup_closed_loop_auto_tune_request.filter_id
            tracking_mode = self._startup_closed_loop_auto_tune_request.tracking_mode
        self._filter_manager = FilterManager(
            gnss_projector=self._gnss_projector,
            gnss_sensor=self._gnss_sensor,
            imu_sensor=self._imu_sensor,
            default_filter_id=default_filter_id,
            default_tune_overrides=tune_overrides,
            tracking_mode=tracking_mode,
        )
        if self._startup_benchmark_config is not None:
            self._filter_manager.switch_filter(self._startup_benchmark_config.selected_filter, skip_current_sensor_frames=True)
            self._filter_manager.set_tracking_mode(self._startup_benchmark_config.tracking_mode, reset_active=True)
            self._startup_benchmark_config.selected_filter_tune = self._filter_manager.get_active_filter_tune()
            self._startup_benchmark_config.tracking_mode = self._filter_manager.tracking_mode
        elif self._startup_closed_loop_auto_tune_request is not None:
            self._filter_manager.switch_filter(self._startup_closed_loop_auto_tune_request.filter_id, skip_current_sensor_frames=True)
            self._filter_manager.set_tracking_mode(self._startup_closed_loop_auto_tune_request.tracking_mode, reset_active=True)
        self._sync_filter_tune_panel()
        self._map_selector = MapSelector(world_map=self._client_manager.world_map)
        self.route_planner = RoutePlanner(world_map=self._client_manager.world_map)
        self._topdown_renderer = (
            None
            if self._lightweight_closed_loop_auto_tune
            else TopDownMapRenderer(world_map=self._client_manager.world_map)
        )
        self._test_route_store = TestRouteStore(map_name=self._active_map_name)
        self.waypoint_tracker.clear_route()
        self._latest_tracking = self._empty_tracking_status()
        self._latest_state = None
        self._latest_ground_truth_state = None
        self._latest_estimated_state = None
        self._latest_localization_status = None
        self._latest_gnss_diagnostics = None
        self._latest_gnss_frame = None
        self._gnss_trail_xy.clear()
        self._cancel_route_activation()
        self._reset_benchmark_failure_monitor()
        self._reset_driving_behavior()

    @staticmethod
    def _empty_tracking_status() -> TrackingStatus:
        return TrackingStatus(
            target_waypoint=None,
            closest_index=0,
            target_index=0,
            cross_track_error_m=float("inf"),
            distance_to_goal_m=float("inf"),
            heading_error_deg=None,
            completed=False,
        )

    def _draw_camera_waypoints(self) -> None:
        if (
            self._world_reload_in_progress
            or self._display is None
            or self._waypoint_manager is None
            or self._camera_sensor is None
            or self._vehicle is None
        ):
            return
        if self.route_planner is not None and self.route_planner.get_route():
            overlay_waypoints = self.waypoint_tracker.get_preview_waypoints()
            target_waypoint = self._latest_tracking.target_waypoint
        else:
            overlay_waypoints = self._waypoint_manager.get_future_waypoints(self._vehicle)
            target_waypoint = None

        self._overlay_renderer.draw(
            surface=self._display.surface,
            waypoints=overlay_waypoints,
            camera=self._camera_sensor.actor,
            vehicle=self._vehicle,
            target_waypoint=target_waypoint,
            camera_content_rect=self._display.camera_content_rect,
        )

    def _draw_topdown_map(self, update_sensor_diagnostics: bool = True) -> None:
        assert self._display is not None
        if self._world_reload_in_progress:
            return
        if self._topdown_renderer is None:
            return
        if update_sensor_diagnostics:
            self._update_sensor_diagnostics()

        route = self.route_planner.get_route() if self.route_planner is not None else []
        start = self._map_selector.start if self._map_selector is not None else None
        goal = self._map_selector.goal if self._map_selector is not None else None
        speed = self._latest_state.speed if self._latest_state is not None else 0.0
        hud = TopDownHudData(
            drive_mode=self._drive_mode.value,
            selection_active=self._map_selection_active,
            route_size=len(route),
            closest_index=self._latest_tracking.closest_index,
            target_index=self._latest_tracking.target_index,
            route_completed=self._latest_tracking.completed,
            speed_mps=speed,
            planner_status=self._planner_status,
        )
        self._topdown_renderer.draw(
            surface=self._display.surface,
            hud=hud,
            ego_state=self._latest_ground_truth_state,
            estimated_state=self._latest_estimated_state,
            start_waypoint=start,
            goal_waypoint=goal,
            route=route,
            target_waypoint=self._latest_tracking.target_waypoint,
            panel_rect=self._display.map_rect,
            gnss_position_xy=self._latest_gnss_position_xy(),
            gnss_trail_xy=tuple(self._gnss_trail_xy),
        )

    def _draw_lidar_panel(self) -> None:
        assert self._display is not None
        measurement = None
        if self._lidar_sensor is not None:
            measurement = self._lidar_sensor.get_latest_measurement()
        self._lidar_renderer.draw(
            surface=self._display.surface,
            rect=self._display.lidar_rect,
            measurement=measurement,
        )

    def _draw_driving_behavior_panels(self) -> None:
        assert self._display is not None
        if self._test_mode_active() and self._test_progress_panel is not None:
            self._test_progress_panel.draw(self._display.surface, self._test_progress_lines())
        elif self._behavior_tuning_panel is not None:
            self._behavior_tuning_panel.draw(self._display.surface)
        if self._control_visual_widget is not None:
            self._control_visual_widget.draw(
                self._display.surface,
                applied_control=self._latest_applied_control,
                requested_control=self._latest_requested_control,
            )
        if self._driving_diagnostics_widget is not None:
            self._driving_diagnostics_widget.draw(
                self._display.surface,
                state=self._latest_ground_truth_state,
                speed_plan=self._latest_speed_plan,
                applied_control=self._latest_applied_control,
            )

    def _draw_control_panel(self) -> None:
        assert self._display is not None
        assert self._control_panel is not None
        if self._offline_recording_active() and self._live_evaluation_panel is not None:
            self._live_evaluation_panel.draw(self._display.surface, self._offline_recording_lines())
            return
        if self._test_mode_active() and self._live_evaluation_panel is not None:
            self._update_live_evaluation_history()
            self._live_evaluation_panel.draw(self._display.surface, self._live_evaluation_lines())
            return
        self._update_control_panel_state()
        self._control_panel.draw(self._display.surface)
        if self._control_panel.active_tab == "Filters":
            if (
                self._filter_manager is not None
                and self._filter_manager.active_filter_id != self._filter_tune_panel_filter_id
            ):
                self._sync_filter_tune_panel()
            self._draw_filters_tab_overlay()
        if self._control_panel.active_tab == "Sensors" and self._sensor_noise_panel is not None:
            self._sensor_noise_panel.status_text = self._last_sensor_apply_status
            sensor_rect = self._display.control_panel_rect.copy()
            sensor_rect.top += 126
            sensor_rect.height = max(40, sensor_rect.height - 126)
            self._sensor_noise_panel.draw(self._display.surface, sensor_rect)

    def _draw_status_bar(self) -> None:
        assert self._display is not None
        self._status_bar.draw(self._display.surface, self._display.status_bar_rect)

    def _test_progress_lines(self) -> list[str]:
        if self._offline_recording_active():
            return self._offline_recording_lines()
        runner = self._test_runner
        config = runner.config if runner is not None else None
        route = runner.current_route_name if runner is not None else ""
        next_route = self._next_benchmark_route_name()
        completion = self._route_completion_percent()
        lines = [
            "Test Mode: ON",
            f"Route: {(runner.current_route_index + 1) if runner is not None else 0}/{runner.total_routes if runner is not None else 0}",
            f"Current: {route or 'initializing'}",
            f"Map: {self._active_map_display_name()}",
            f"Filter: {self._active_filter_name()}",
            f"Tracking mode: {self._tracking_mode_for_metadata()}",
            f"State: {runner.state.value if runner is not None else 'n/a'}",
            f"Route status: {runner.route_status if runner is not None else 'n/a'}",
            f"Attempt: {runner.current_attempt if runner is not None and runner.current_attempt else 0}/{runner.max_attempts if runner is not None else 0}",
            f"Test time: {runner.elapsed_test_seconds():.1f}s" if runner is not None else "Test time: n/a",
            f"Route time: {runner.elapsed_route_seconds():.1f}s" if runner is not None else "Route time: n/a",
            f"Distance to goal: {self._format_optional_metric(self._latest_tracking.distance_to_goal_m, 'm')}",
            f"Completion: {completion:.0f}%" if completion is not None else "Completion: n/a",
            f"Next: {next_route or 'none'}",
        ]
        if config is not None:
            lines.insert(5, f"Sensor preset: {config.sensor_noise_preset}")
            lines.insert(6, f"Behavior preset: {config.vehicle_behavior_preset}")
        if runner is not None and runner.status_text:
            lines.append(runner.status_text)
        lines.extend(self._plot_job_status_lines())
        last_failure = runner.last_failure_reason if runner is not None else self._last_benchmark_failure_reason
        if last_failure:
            lines.append(f"Last failure: {last_failure}")
        return lines

    def _offline_recording_lines(self) -> list[str]:
        recorder = self._offline_recorder
        config = self._startup_offline_recording_config
        completion = self._route_completion_percent()
        lines = [
            "Offline Localization Replay: recording",
            "Driver: ground_truth_controller",
            f"Route: {(recorder.current_route_index + 1) if recorder is not None else 0}/{recorder.total_routes if recorder is not None else 0}",
            f"Current: {recorder.current_route_name if recorder is not None else 'initializing'}",
            f"Map: {self._active_map_display_name()}",
            f"State: {recorder.state.value if recorder is not None else 'n/a'}",
            f"Phase: {recorder.current_phase if recorder is not None else 'n/a'}",
            f"Samples: {recorder.sample_count if recorder is not None else 0}",
            f"Warm-up excluded: {recorder.warmup_excluded_seconds:.1f}s" if recorder is not None else "Warm-up excluded: n/a",
            f"Route time: {recorder.elapsed_route_seconds():.1f}s" if recorder is not None else "Route time: n/a",
            f"Distance to goal: {self._format_optional_metric(self._latest_tracking.distance_to_goal_m, 'm')}",
            f"Completion: {completion:.0f}%" if completion is not None else "Completion: n/a",
            "Control state: ground truth",
            "Candidate filters do not control this pass.",
        ]
        if config is not None:
            lines.insert(6, f"Sensor preset: {config.sensor_noise_preset}")
            lines.insert(7, f"Behavior preset: {config.vehicle_behavior_preset}")
        if recorder is not None and recorder.status_text:
            lines.append(recorder.status_text)
        if recorder is not None and recorder.run_folder is not None:
            lines.append(f"Output folder: {recorder.run_folder}")
        if recorder is not None and recorder.last_failure_reason:
            lines.append(f"Last failure: {recorder.last_failure_reason}")
        return lines

    def _live_evaluation_lines(self) -> list[str]:
        logger = self._active_performance_logger()
        driving_sample_count = logger.running_sample_count(phases=("driving", "completed"))
        if driving_sample_count > 0:
            metrics = logger.running_driving_metrics()
            metric_phase_label = f"driving phase ({driving_sample_count} samples)"
            waiting_line = None
        else:
            metrics = logger.running_metrics()
            metric_phase_label = "all-phase/stabilization"
            waiting_line = "Waiting for driving phase..."
        diagnostics = self._filter_manager.get_diagnostics() if self._filter_manager is not None else {}
        ratio = self._ratio(logger.current_raw_gnss_error_m, logger.current_position_error_m)
        improvement = None
        raw_rmse = metrics.get("raw_gnss_rmse_m")
        filtered_rmse = metrics.get("filtered_rmse_m")
        if raw_rmse is not None and raw_rmse > 0.0 and filtered_rmse is not None:
            improvement = 100.0 * (raw_rmse - filtered_rmse) / raw_rmse
        position_nees = metrics.get("mean_position_nees")
        position_nees_label = "Position NEES"
        if position_nees is None:
            position_nees = metrics.get("mean_position_nees_diagonal_approx")
            position_nees_label = "Position NEES approx"
        lines = [
            f"Metric source: {metric_phase_label}",
            f"Current position error: {self._format_optional_metric(logger.current_position_error_m, 'm')}",
            f"Current speed error: {self._format_optional_metric(self._current_speed_error(), 'm/s')}",
            f"Current yaw error: {self._format_optional_metric(self._current_yaw_error(), 'deg')}",
            f"Primary position RMSE: {self._format_optional_metric(metrics.get('filtered_rmse_m'), 'm')}",
            f"Primary speed RMSE: {self._format_optional_metric(metrics.get('speed_rmse_mps'), 'm/s')}",
            f"Primary yaw RMSE: {self._format_optional_metric(metrics.get('yaw_rmse_deg'), 'deg')}",
            f"Raw GNSS RMSE: {self._format_optional_metric(metrics.get('raw_gnss_rmse_m'), 'm')}",
            f"Filtered RMSE: {self._format_optional_metric(metrics.get('filtered_rmse_m'), 'm')}",
            f"Improvement: {self._format_optional_metric(improvement, '%')}",
            f"Legacy mixed NIS: {self._format_optional_metric(metrics.get('legacy_mean_nis_mixed') or metrics.get('mean_nis'), '')}",
            f"{position_nees_label}: {self._format_optional_metric(position_nees, '')}",
            f"Legacy NEES: {self._format_optional_metric(metrics.get('mean_nees'), '')}",
            f"P95 position error: {self._format_optional_metric(metrics.get('filtered_p95_error_m'), 'm')}",
            f"Max position error: {self._format_optional_metric(metrics.get('filtered_max_error_m'), 'm')}",
            f"Innovation norm: {self._format_optional_metric(metrics.get('innovation_mean'), '')}",
            f"GNSS updates: {self._format_debug_value(diagnostics.get('last_gnss_frame'))}",
            f"IMU updates: {self._format_debug_value(diagnostics.get('last_imu_frame'))}",
            f"Instant improvement ratio: {self._format_optional_metric(ratio, 'x')}",
        ]
        if waiting_line is not None:
            lines.insert(1, waiting_line)
        return lines

    def _update_live_evaluation_history(self) -> None:
        if self._live_evaluation_panel is None:
            return
        self._live_evaluation_panel.update_histories(
            position_error_m=self._active_performance_logger().current_position_error_m,
            raw_error_m=self._active_performance_logger().current_raw_gnss_error_m,
            actual_speed_mps=self._latest_ground_truth_state.speed if self._latest_ground_truth_state else None,
            estimated_speed_mps=self._latest_estimated_state.speed if self._latest_estimated_state else None,
        )

    def _current_speed_error(self) -> Optional[float]:
        if self._latest_ground_truth_state is None or self._latest_estimated_state is None:
            return None
        return self._latest_estimated_state.speed - self._latest_ground_truth_state.speed

    def _current_yaw_error(self) -> Optional[float]:
        if self._latest_ground_truth_state is None or self._latest_estimated_state is None:
            return None
        delta = self._latest_estimated_state.yaw - self._latest_ground_truth_state.yaw
        while delta > 180.0:
            delta -= 360.0
        while delta < -180.0:
            delta += 360.0
        return delta

    def _route_completion_percent(self) -> Optional[float]:
        route = self.route_planner.get_route() if self.route_planner is not None else []
        if not route:
            return None
        return 100.0 * min(1.0, max(0.0, self._latest_tracking.closest_index / max(1, len(route) - 1)))

    def _next_benchmark_route_name(self) -> str:
        runner = self._test_runner
        if runner is None or not runner.is_automated or runner.config is None:
            return ""
        next_index = runner.current_route_index + (1 if runner.route_running else 0)
        routes = runner.config.selected_routes
        if 0 <= next_index < len(routes):
                return routes[next_index].name
        return ""

    def _update_offline_recording(self) -> bool:
        recorder = self._offline_recorder
        if recorder is None or not recorder.is_active:
            return False
        if not recorder.route_running:
            if recorder.needs_map_switch(self._active_map_name):
                return self._switch_map_for_offline_recording() != WorldSwitchResult.NOOP
            recorder.update(
                route_completed=False,
                route_failed=False,
                active_map_name=self._active_map_name,
                ground_truth_state=self._latest_ground_truth_state,
                gnss_measurement=self._gnss_sensor.get_latest_measurement() if self._gnss_sensor is not None else None,
                imu_measurement=self._imu_sensor.get_latest_measurement() if self._imu_sensor is not None else None,
                gnss_projector=self._gnss_projector,
                applied_control=self._latest_applied_control,
                frame_index=self._current_world_frame(),
            )
            self._control_status_text = recorder.status_text
            self._planner_status = recorder.status_text
            return False

        route = self.route_planner.get_route() if self.route_planner is not None else []
        route_failed = (
            self._route_activation_state == RouteActivationState.IDLE
            and not route
            and not self._latest_tracking.completed
        )
        if not recorder.controller_enabled:
            self._reset_benchmark_failure_monitor()
            recorder.update(
                route_completed=False,
                route_failed=False,
                active_map_name=self._active_map_name,
                ground_truth_state=self._latest_ground_truth_state,
                gnss_measurement=self._gnss_sensor.get_latest_measurement() if self._gnss_sensor is not None else None,
                imu_measurement=self._imu_sensor.get_latest_measurement() if self._imu_sensor is not None else None,
                gnss_projector=self._gnss_projector,
                applied_control=self._latest_applied_control,
                frame_index=self._current_world_frame(),
            )
            self._control_status_text = recorder.status_text
            self._planner_status = recorder.status_text
            return False

        failure_reason = self._offline_recording_failure_reason(route_failed=route_failed)
        if failure_reason:
            if self._vehicle is not None:
                try:
                    self._vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                except RuntimeError:
                    pass
            recorder.update(
                route_completed=False,
                route_failed=True,
                active_map_name=self._active_map_name,
                ground_truth_state=self._latest_ground_truth_state,
                gnss_measurement=self._gnss_sensor.get_latest_measurement() if self._gnss_sensor is not None else None,
                imu_measurement=self._imu_sensor.get_latest_measurement() if self._imu_sensor is not None else None,
                gnss_projector=self._gnss_projector,
                applied_control=self._latest_applied_control,
                frame_index=self._current_world_frame(),
            )
            self._last_benchmark_failure_reason = failure_reason
            self._reset_benchmark_failure_monitor()
        else:
            recorder.update(
                route_completed=self._latest_tracking.completed,
                route_failed=route_failed,
                active_map_name=self._active_map_name,
                ground_truth_state=self._latest_ground_truth_state,
                gnss_measurement=self._gnss_sensor.get_latest_measurement() if self._gnss_sensor is not None else None,
                imu_measurement=self._imu_sensor.get_latest_measurement() if self._imu_sensor is not None else None,
                gnss_projector=self._gnss_projector,
                applied_control=self._latest_applied_control,
                frame_index=self._current_world_frame(),
            )
        self._control_status_text = recorder.status_text
        self._planner_status = recorder.status_text
        if recorder.needs_map_switch(self._active_map_name):
            return self._switch_map_for_offline_recording() != WorldSwitchResult.NOOP
        return False

    def _current_world_frame(self) -> Optional[int]:
        try:
            return int(self._client_manager.world.get_snapshot().frame)
        except Exception:
            return None

    def _update_test_performance(self) -> bool:
        runner = self._test_runner
        if runner is None or not runner.is_active:
            return False
        logger = runner.current_logger
        if logger is None:
            if runner.needs_map_switch(self._active_map_name):
                return self._switch_map_for_benchmark() != WorldSwitchResult.NOOP
            runner.update(route_completed=False, route_failed=False, active_map_name=self._active_map_name)
            self._control_status_text = runner.status_text
            self._planner_status = runner.status_text
            return False

        route_name = runner.current_route_name or "test_route"
        phase = self._benchmark_phase()
        if BENCHMARK.collect_stabilization_samples or phase != "stabilization":
            logger.collect_sample(
                phase=phase,
                route_name=route_name,
                ground_truth_state=self._latest_ground_truth_state,
                filtered_state=self._latest_estimated_state,
                gnss_diagnostics=self._latest_gnss_diagnostics,
                tracking=self._latest_tracking,
                route_completed=self._latest_tracking.completed,
                filter_diagnostics=self._filter_manager.get_diagnostics() if self._filter_manager is not None else None,
                speed_plan=self._latest_speed_plan,
            )

        route = self.route_planner.get_route() if self.route_planner is not None else []
        route_failed = (
            self._route_activation_state == RouteActivationState.IDLE
            and not route
            and not self._latest_tracking.completed
        )
        failure_reason = self._benchmark_failure_reason(route_failed=route_failed)
        if failure_reason:
            self._last_benchmark_failure_reason = failure_reason
            if self._vehicle is not None:
                try:
                    self._vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                except RuntimeError:
                    pass
            paths = runner.fail_current_attempt(
                reason=failure_reason,
                simulation_time_s=self._latest_ground_truth_state.timestamp
                if self._latest_ground_truth_state is not None
                else None,
                active_map_name=self._active_map_name,
            )
            self._reset_benchmark_failure_monitor()
        else:
            paths = runner.update(
                route_completed=self._latest_tracking.completed,
                route_failed=route_failed,
                active_map_name=self._active_map_name,
            )
        if paths is not None:
            self._control_status_text = runner.status_text
            self._planner_status = runner.status_text
        if runner.needs_map_switch(self._active_map_name):
            return self._switch_map_for_benchmark() != WorldSwitchResult.NOOP
        return False

    def _benchmark_phase(self) -> str:
        if self._latest_tracking.completed:
            return "completed"
        if self._route_activation_state == RouteActivationState.WAITING_FOR_LOCALIZATION_STABILITY:
            return "stabilization"
        if self._route_activation_state == RouteActivationState.ROUTE_ACTIVE:
            return "driving"
        return "idle"

    def _benchmark_failure_reason(self, route_failed: bool) -> str:
        runner = self._test_runner
        if runner is None or not runner.route_running or not self._test_mode_active():
            self._reset_benchmark_failure_monitor()
            return ""
        if route_failed:
            return "Route unavailable before completion"
        if self._latest_tracking.completed:
            self._reset_benchmark_failure_monitor()
            return ""
        if self._route_activation_state != RouteActivationState.ROUTE_ACTIVE:
            self._reset_benchmark_failure_monitor()
            return ""
        state = self._latest_ground_truth_state
        if state is None:
            return ""

        now = time.monotonic()
        position = (float(state.x), float(state.y))
        speed = abs(float(state.speed))
        tracking = self._latest_tracking
        cte = tracking.cross_track_error_m
        if isinstance(cte, (int, float)) and math.isfinite(cte) and cte >= BENCHMARK_LATERAL_DEVIATION_M:
            if self._failure_monitor_deviation_started is None:
                self._failure_monitor_deviation_started = now
            elif now - self._failure_monitor_deviation_started >= BENCHMARK_LATERAL_DEVIATION_SECONDS:
                return f"Large lateral deviation ({cte:.1f} m from route)"
        else:
            self._failure_monitor_deviation_started = None

        if self._failure_monitor_last_progress_time is None:
            self._failure_monitor_last_progress_time = now
            self._failure_monitor_last_position = position
            self._failure_monitor_last_distance_to_goal = self._finite_float(tracking.distance_to_goal_m)
            self._failure_monitor_last_closest_index = tracking.closest_index
            return ""

        moved_m = 0.0
        if self._failure_monitor_last_position is not None:
            moved_m = math.hypot(
                position[0] - self._failure_monitor_last_position[0],
                position[1] - self._failure_monitor_last_position[1],
            )
        current_goal_distance = self._finite_float(tracking.distance_to_goal_m)
        previous_goal_distance = self._failure_monitor_last_distance_to_goal
        goal_distance_progress = (
            previous_goal_distance is not None
            and current_goal_distance is not None
            and previous_goal_distance - current_goal_distance >= BENCHMARK_GOAL_DISTANCE_PROGRESS_M
        )
        index_progress = tracking.closest_index > self._failure_monitor_last_closest_index
        movement_progress = moved_m >= BENCHMARK_PROGRESS_DISTANCE_M

        if movement_progress or goal_distance_progress or index_progress:
            self._failure_monitor_last_progress_time = now
            self._failure_monitor_last_position = position
            self._failure_monitor_last_distance_to_goal = current_goal_distance
            self._failure_monitor_last_closest_index = tracking.closest_index
            return ""

        stalled_seconds = now - self._failure_monitor_last_progress_time
        if speed <= BENCHMARK_STUCK_SPEED_MPS and stalled_seconds >= BENCHMARK_STUCK_SECONDS:
            return f"Vehicle stuck: speed {speed:.2f} m/s, no route progress for {stalled_seconds:.1f}s"
        if stalled_seconds >= BENCHMARK_NO_PROGRESS_SECONDS:
            return f"No route progress for {stalled_seconds:.1f}s"
        return ""

    def _offline_recording_failure_reason(self, route_failed: bool) -> str:
        recorder = self._offline_recorder
        if recorder is None or not recorder.route_running:
            self._reset_benchmark_failure_monitor()
            return ""
        if route_failed:
            return "Route unavailable before completion"
        if self._latest_tracking.completed:
            self._reset_benchmark_failure_monitor()
            return ""
        if self._route_activation_state != RouteActivationState.ROUTE_ACTIVE:
            self._reset_benchmark_failure_monitor()
            return ""
        state = self._latest_ground_truth_state
        if state is None:
            return ""

        now = time.monotonic()
        position = (float(state.x), float(state.y))
        speed = abs(float(state.speed))
        tracking = self._latest_tracking
        cte = tracking.cross_track_error_m
        if isinstance(cte, (int, float)) and math.isfinite(cte) and cte >= BENCHMARK_LATERAL_DEVIATION_M:
            if self._failure_monitor_deviation_started is None:
                self._failure_monitor_deviation_started = now
            elif now - self._failure_monitor_deviation_started >= BENCHMARK_LATERAL_DEVIATION_SECONDS:
                return f"Large lateral deviation ({cte:.1f} m from route)"
        else:
            self._failure_monitor_deviation_started = None

        if self._failure_monitor_last_progress_time is None:
            self._failure_monitor_last_progress_time = now
            self._failure_monitor_last_position = position
            self._failure_monitor_last_distance_to_goal = self._finite_float(tracking.distance_to_goal_m)
            self._failure_monitor_last_closest_index = tracking.closest_index
            return ""

        moved_m = 0.0
        if self._failure_monitor_last_position is not None:
            moved_m = math.hypot(
                position[0] - self._failure_monitor_last_position[0],
                position[1] - self._failure_monitor_last_position[1],
            )
        current_goal_distance = self._finite_float(tracking.distance_to_goal_m)
        previous_goal_distance = self._failure_monitor_last_distance_to_goal
        goal_distance_progress = (
            previous_goal_distance is not None
            and current_goal_distance is not None
            and previous_goal_distance - current_goal_distance >= BENCHMARK_GOAL_DISTANCE_PROGRESS_M
        )
        index_progress = tracking.closest_index > self._failure_monitor_last_closest_index
        movement_progress = moved_m >= BENCHMARK_PROGRESS_DISTANCE_M

        if movement_progress or goal_distance_progress or index_progress:
            self._failure_monitor_last_progress_time = now
            self._failure_monitor_last_position = position
            self._failure_monitor_last_distance_to_goal = current_goal_distance
            self._failure_monitor_last_closest_index = tracking.closest_index
            return ""

        stalled_seconds = now - self._failure_monitor_last_progress_time
        if speed <= BENCHMARK_STUCK_SPEED_MPS and stalled_seconds >= BENCHMARK_STUCK_SECONDS:
            return f"Vehicle stuck: speed {speed:.2f} m/s, no route progress for {stalled_seconds:.1f}s"
        if stalled_seconds >= BENCHMARK_NO_PROGRESS_SECONDS:
            return f"No route progress for {stalled_seconds:.1f}s"
        return ""

    def _update_sensor_diagnostics(self) -> None:
        gnss = self._gnss_sensor.get_latest_measurement() if self._gnss_sensor is not None else None
        if self._gnss_projector is None:
            self._latest_gnss_diagnostics = None
            return

        diagnostics = self._gnss_projector.diagnostics(gnss, self._latest_ground_truth_state)
        self._latest_gnss_diagnostics = diagnostics
        if gnss is None or diagnostics is None:
            return
        if self._latest_gnss_frame == gnss.frame:
            return
        self._latest_gnss_frame = gnss.frame
        self._gnss_trail_xy.append((diagnostics.local_x, diagnostics.local_y))

    def _latest_gnss_position_xy(self) -> Optional[tuple[float, float]]:
        if self._latest_gnss_diagnostics is None:
            return None
        return self._latest_gnss_diagnostics.local_x, self._latest_gnss_diagnostics.local_y

    @staticmethod
    def _finite_float(value: object) -> Optional[float]:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    @staticmethod
    def _format_optional_metric(value: Optional[float], suffix: str) -> str:
        if value is None or not isinstance(value, (int, float)):
            return "n/a"
        if value != value or value in (float("inf"), float("-inf")):
            return "n/a"
        return f"{value:.2f} {suffix}"

    def shutdown(self) -> None:
        """Destroy actors and close pygame resources."""
        if self._test_runner is not None and self._test_runner.is_active:
            self._test_runner.stop(aborted=True, reason="Application shutdown")
        if self._offline_recorder is not None and self._offline_recorder.is_active:
            self._offline_recorder.stop(aborted=True, reason="Application shutdown")

        self._plot_worker.shutdown(wait=True, timeout_s=2.0)

        if self._vehicle is not None:
            try:
                self._vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            except RuntimeError:
                pass

        if self._sensor_manager is not None:
            self._sensor_manager.destroy_all()
            self._sensor_manager = None
            self._camera_sensor = None
            self._gnss_sensor = None
            self._imu_sensor = None
            self._lidar_sensor = None

        if self._vehicle_manager is not None:
            self._vehicle_manager.destroy()
            self._vehicle_manager = None

        self._client_manager.restore_world_settings()
        if self._display is not None:
            self._display.shutdown()
            self._display = None
        else:
            pygame.quit()


class AppClosedLoopValidationRunner:
    """Adapter from the backend validation-runner interface to SimulationApp."""

    def __init__(self, app: SimulationApp) -> None:
        self._app = app

    def run(self, request: ClosedLoopValidationRequest) -> dict[str, object]:
        return self._app._run_closed_loop_auto_tune_validation(request)
