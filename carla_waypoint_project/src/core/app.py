"""Main simulation application orchestrator."""

from __future__ import annotations

from collections import deque
from enum import Enum
import time
from typing import Optional

import pygame

from config.settings import BENCHMARK, ROUTE_INITIALIZATION, TOPDOWN_MAP
from src.control.vehicle_controller import VehicleController
from src.control.waypoint_tracker import TrackingStatus, WaypointTracker
from src.core.carla_client import CarlaClientManager
from src.core.simulation import SimulationClock
from src.evaluation.filter_performance import FilterPerformanceLogger
from src.evaluation.route_test_runner import RouteTestRunner
from src.evaluation.test_route_store import TestRouteStore
from src.localization.gnss_projection import GnssDiagnostics, GnssLocalProjector
from src.localization.state_estimator import (
    EgoState,
    EstimatedStateProvider,
    GroundTruthStateProvider,
    KalmanStateEstimator,
    LocalizationStatus,
)
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
from src.visualization.sensor_panel import SensorPanelData, SensorPanelRenderer
from src.visualization.topdown_map import TopDownHudData, TopDownMapRenderer
from src.visualization.ui.control_panel import ControlPanel
from src.visualization.waypoint_overlay import WaypointOverlayRenderer
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class DriveMode(Enum):
    """High-level ego driving mode."""

    MANUAL = "MANUAL"
    AUTONOMOUS = "AUTO"


class RouteActivationState(Enum):
    """Lifecycle state for route creation after A/B selection."""

    IDLE = "IDLE"
    WAITING_FOR_LOCALIZATION_STABILITY = "WAITING_FOR_LOCALIZATION_STABILITY"
    ROUTE_ACTIVE = "ROUTE_ACTIVE"


class SimulationApp:
    """Coordinate CARLA, sensors, display, route selection, and control."""

    def __init__(self) -> None:
        pygame.init()
        self._client_manager = CarlaClientManager()
        self._clock = SimulationClock()
        self._display: Optional[PygameDisplay] = None
        self._control_panel: Optional[ControlPanel] = None

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
        self._sensor_panel_renderer = SensorPanelRenderer()
        self._ground_truth_provider: Optional[GroundTruthStateProvider] = None
        self._estimated_state_provider: Optional[EstimatedStateProvider] = None
        self._gnss_projector: Optional[GnssLocalProjector] = None
        self._map_selector: Optional[MapSelector] = None
        self._topdown_renderer: Optional[TopDownMapRenderer] = None
        self._test_route_store: Optional[TestRouteStore] = None
        self._test_runner: Optional[RouteTestRunner] = None
        self._performance_logger = FilterPerformanceLogger()
        self.route_planner: Optional[RoutePlanner] = None
        self.waypoint_tracker = WaypointTracker()
        self.autonomous_controller = VehicleController()

        self._drive_mode = DriveMode.MANUAL
        self._map_selection_active = False
        self._test_route_authoring_active = False
        self._latest_state: Optional[EgoState] = None
        self._latest_ground_truth_state: Optional[EgoState] = None
        self._latest_estimated_state: Optional[EgoState] = None
        self._latest_localization_status: Optional[LocalizationStatus] = None
        self._latest_tracking = self._empty_tracking_status()
        self._planner_status = ""
        self._latest_gnss_diagnostics: Optional[GnssDiagnostics] = None
        self._latest_gnss_frame: Optional[int] = None
        self._gnss_trail_xy: deque[tuple[float, float]] = deque(maxlen=TOPDOWN_MAP.gnss_trail_length)
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

    def _setup(self) -> None:
        """Initialize CARLA world, vehicle, sensors, route tools, and visualization."""
        self._client_manager.connect()

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
        self._camera_sensor = self._sensor_manager.create_rgb_camera(attach_to=self._vehicle)
        self._gnss_sensor = self._sensor_manager.create_gnss(attach_to=self._vehicle)
        self._imu_sensor = self._sensor_manager.create_imu(attach_to=self._vehicle)
        self._lidar_sensor = self._sensor_manager.create_lidar(attach_to=self._vehicle)

        world_map = self._client_manager.world_map
        self._waypoint_manager = WaypointManager(world_map=world_map)
        self._manual_controller = ManualController(vehicle=self._vehicle)
        self._ground_truth_provider = GroundTruthStateProvider(vehicle=self._vehicle)
        self._gnss_projector = GnssLocalProjector(world_map=world_map)
        self._estimated_state_provider = EstimatedStateProvider(
            estimator=KalmanStateEstimator(gnss_projector=self._gnss_projector),
            gnss_sensor=self._gnss_sensor,
            imu_sensor=self._imu_sensor,
        )
        self._map_selector = MapSelector(world_map=world_map)
        self.route_planner = RoutePlanner(world_map=world_map)
        self._topdown_renderer = TopDownMapRenderer(world_map=world_map)
        self._test_route_store = TestRouteStore(map_name=getattr(world_map, "name", None))
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
        )

        if self.route_planner.planner_error:
            self._planner_status = "Planner: fallback"
        else:
            self._planner_status = "Planner: CARLA"

        self._initialize_display()

    def _ensure_ready(self) -> None:
        required = {
            "display": self._display,
            "control panel": self._control_panel,
            "vehicle": self._vehicle,
            "camera sensor": self._camera_sensor,
            "GNSS sensor": self._gnss_sensor,
            "IMU sensor": self._imu_sensor,
            "LiDAR sensor": self._lidar_sensor,
            "waypoint manager": self._waypoint_manager,
            "manual controller": self._manual_controller,
            "ground-truth provider": self._ground_truth_provider,
            "estimated-state provider": self._estimated_state_provider,
            "GNSS projector": self._gnss_projector,
            "map selector": self._map_selector,
            "route planner": self.route_planner,
            "top-down renderer": self._topdown_renderer,
            "test route store": self._test_route_store,
            "test runner": self._test_runner,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(f"Application is not initialized: {', '.join(missing)}.")

    def _initialize_display(self) -> None:
        self._display = PygameDisplay()
        self._control_panel = ControlPanel(self._display.control_panel_rect)
        self._build_control_panel()
        self._control_status_text = "Dashboard ready"
        self._draw_startup_frame()

    def _draw_startup_frame(self) -> None:
        if self._display is None or self._control_panel is None:
            return
        self._display.begin_frame(None)
        self._update_control_panel_state()
        self._control_panel.draw(self._display.surface)
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

    def _benchmark_output_status(self) -> str:
        runner = self._test_runner
        if runner is None or runner.benchmark_folder is None:
            return "Output: none"
        return f"Output: {runner.benchmark_folder.name}"

    def _active_performance_logger(self) -> FilterPerformanceLogger:
        if self._test_runner is not None and self._test_runner.current_logger is not None:
            return self._test_runner.current_logger
        return self._performance_logger

    def _build_control_panel(self) -> None:
        assert self._control_panel is not None
        button_specs = [
            ("Manual Mode", self._set_manual_mode),
            ("Autonomous Mode", self._try_enable_autonomous_mode),
            ("Map Select ON/OFF", self._toggle_map_selection),
            ("Test Route Mode ON/OFF", self._toggle_test_route_authoring),
            ("Save A/B as Test Route", self._save_current_ab_as_test_route),
            ("Previous Test Route", self._select_previous_test_route),
            ("Next Test Route", self._select_next_test_route),
            ("Load Selected Test Route", self._load_selected_test_route),
            ("Start Test Run", self._start_test_run),
            ("Stop Test Run", self._stop_test_run),
            ("Reset Selection", self._reset_selection_and_route),
            ("Clear Route", self._clear_route),
            ("Reset Estimator", self._reset_estimator),
            ("Emergency Brake", self._emergency_brake),
            ("Save Test Report", self._save_test_report),
        ]
        for label, callback in button_specs:
            self._control_panel.add_button(label, callback)

    def _update_control_panel_state(self) -> None:
        assert self._display is not None
        assert self._control_panel is not None
        store = self._test_route_store
        runner = self._test_runner
        logger = self._active_performance_logger()
        route_count = store.route_count() if store is not None else 0
        current_route = store.get_current_route() if store is not None else None
        test_active = runner.is_active if runner is not None else False
        endpoints_ready = self._map_selector is not None and self._map_selector.endpoints is not None

        self._control_panel.set_rect(self._display.control_panel_rect)
        self._control_panel.set_button_state("Manual Mode", active=self._drive_mode == DriveMode.MANUAL)
        self._control_panel.set_button_state("Autonomous Mode", active=self._drive_mode == DriveMode.AUTONOMOUS)
        self._control_panel.set_button_state("Map Select ON/OFF", active=self._map_selection_active)
        self._control_panel.set_button_state("Test Route Mode ON/OFF", active=self._test_route_authoring_active)
        self._control_panel.set_button_state("Save A/B as Test Route", active=endpoints_ready)
        self._control_panel.set_button_state("Start Test Run", enabled=not test_active)
        self._control_panel.set_button_state("Stop Test Run", enabled=test_active, active=test_active)
        self._control_panel.set_button_state(
            "Save Test Report",
            enabled=bool(logger.samples) or (runner is not None and runner.benchmark_folder is not None),
        )

        route_label = "none"
        if current_route is not None and store is not None:
            route_label = f"{store.current_index + 1}/{route_count} {current_route.name}"

        status_lines = [
            (
                f"Map: {'ON' if self._map_selection_active else 'OFF'}  "
                f"Author: {'ON' if self._test_route_authoring_active else 'OFF'}"
            ),
            f"A/B: {'READY' if endpoints_ready else 'not set'}  Saved: {route_count}",
            f"Selected: {route_label}",
            f"Benchmark: {'ACTIVE' if test_active else 'inactive'}",
            runner.status_text if runner is not None else "Test idle",
            (
                f"Err/GNSS: {self._format_optional_metric(logger.current_position_error_m, 'm')} / "
                f"{self._format_optional_metric(logger.current_raw_gnss_error_m, 'm')}"
            ),
            (
                f"CTE/RMSE: {self._format_optional_metric(logger.current_cross_track_error_m, 'm')} / "
                f"{self._format_optional_metric(logger.running_rmse_m(), 'm')}"
            ),
            self._benchmark_output_status(),
            self._control_status_text,
        ]
        self._control_panel.set_status_lines(status_lines)

    def run(self) -> None:
        """Run the camera, route-selection, and route-following application loop."""
        try:
            self._setup()
            self._ensure_ready()

            display = self._display
            assert display is not None

            manual_controller = self._manual_controller
            camera_sensor = self._camera_sensor
            waypoint_manager = self._waypoint_manager
            vehicle = self._vehicle
            ground_truth_provider = self._ground_truth_provider
            estimated_state_provider = self._estimated_state_provider

            assert manual_controller is not None
            assert camera_sensor is not None
            assert waypoint_manager is not None
            assert vehicle is not None
            assert ground_truth_provider is not None
            assert estimated_state_provider is not None

            running = True
            while running:
                running = self._process_events()
                if not running:
                    break
                self._client_manager.tick()
                self._latest_ground_truth_state = ground_truth_provider.get_state()
                self._latest_estimated_state = estimated_state_provider.update()
                self._latest_localization_status = estimated_state_provider.build_status(
                    self._latest_ground_truth_state
                )
                self._update_route_activation_state()
                self._latest_state = self._state_for_tracking_and_control()
                if self._can_update_route_tracking() and self._latest_state is not None:
                    self._latest_tracking = self.waypoint_tracker.update(self._latest_state)
                self._update_sensor_diagnostics()
                self._update_test_performance()

                if self._route_activation_state == RouteActivationState.WAITING_FOR_LOCALIZATION_STABILITY:
                    control = carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=ROUTE_INITIALIZATION.hold_brake,
                        hand_brake=False,
                    )
                    vehicle.apply_control(control)
                elif self._drive_mode == DriveMode.AUTONOMOUS:
                    control_state = self._state_for_tracking_and_control()
                    if control_state is None:
                        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
                    else:
                        control = self.autonomous_controller.compute_control(
                            state=control_state,
                            target_waypoint=self._latest_tracking.target_waypoint,
                            route_completed=self._latest_tracking.completed,
                        )
                    vehicle.apply_control(control)
                else:
                    manual_controller.apply_control()

                camera_surface = camera_sensor.get_latest_surface()
                display.begin_frame(camera_surface)
                self._draw_camera_waypoints(waypoint_manager, camera_sensor, vehicle)
                self._draw_topdown_map()
                self._draw_lidar_panel()
                self._draw_sensor_panel()
                self._draw_control_panel()
                display.end_frame()
                self._clock.tick_pygame()
        finally:
            self.shutdown()

    def _state_for_tracking_and_control(self) -> Optional[EgoState]:
        """Return GT in manual mode and Kalman state in autonomous mode."""
        if self._drive_mode == DriveMode.AUTONOMOUS:
            return self._latest_estimated_state
        return self._latest_ground_truth_state

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
            if event.type == pygame.KEYDOWN:
                if not self._handle_key_down(event):
                    return False
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
        self._cancel_route_activation()
        self._drive_mode = DriveMode.MANUAL
        if self._vehicle is not None:
            self._vehicle.set_autopilot(False)
        self._planner_status = "Mode: manual"
        self._control_status_text = "Manual mode"

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

        route = self._test_route_store.get_current_route()
        if route is None:
            self._planner_status = "Select or create a test route first"
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
        if self._estimated_state_provider is None:
            self._control_status_text = "Estimator not initialized"
            return

        self._estimated_state_provider.reset(skip_current_sensor_frames=True)
        self._latest_estimated_state = None
        self._latest_localization_status = None
        self._latest_gnss_diagnostics = None
        self._latest_gnss_frame = None
        self._gnss_trail_xy.clear()
        self._planner_status = "Estimator reset"
        self._control_status_text = "Estimator reset"

    def _emergency_brake(self) -> None:
        if self._test_runner is not None and self._test_runner.is_active:
            self._test_runner.stop(aborted=True, reason="Test aborted: emergency brake")
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

    def _save_test_report(self) -> None:
        if self._test_runner is not None and self._test_runner.benchmark_folder is not None:
            try:
                from src.evaluation.benchmark_plotter import generate_benchmark_plots

                paths = generate_benchmark_plots(self._test_runner.benchmark_folder)
                self._planner_status = f"Regenerated plots: {len(paths)} files"
                self._control_status_text = self._planner_status
            except Exception as exc:
                self._planner_status = f"Plot generation failed: {exc}"
                self._control_status_text = self._planner_status
            return

        logger = self._active_performance_logger()
        if not logger.samples:
            self._planner_status = "No test samples to export"
            self._control_status_text = self._planner_status
            return

        _csv_path, json_path = logger.export()
        self._planner_status = f"Saved test report: {json_path.name}"
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

    def _begin_route_initialization_from_selection(self, start_autonomous: bool) -> None:
        if self._map_selector is None:
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
        if self._estimated_state_provider is not None:
            self._estimated_state_provider.reset(skip_current_sensor_frames=True)
            self._latest_estimated_state = None
            self._latest_gnss_diagnostics = None
            self._latest_gnss_frame = None
            self._gnss_trail_xy.clear()
        if self._ground_truth_provider is not None:
            self._latest_ground_truth_state = self._ground_truth_provider.get_state()
            self._latest_state = self._latest_ground_truth_state
        if self._estimated_state_provider is not None:
            self._latest_localization_status = self._estimated_state_provider.build_status(
                self._latest_ground_truth_state
            )

    def _reset_selection_and_route(self) -> None:
        if self._map_selector is not None:
            self._map_selector.reset()
        self._clear_route()
        self._planner_status = "Planner: reset"

    def _clear_route(self, stop_test: bool = True) -> None:
        if stop_test and self._test_runner is not None and self._test_runner.is_active:
            self._test_runner.stop(aborted=True, reason="Test aborted: route cleared")
        self._cancel_route_activation()
        if self.route_planner is not None:
            self.route_planner.clear_route()
        self.waypoint_tracker.clear_route()
        self._latest_tracking = self._empty_tracking_status()
        self._drive_mode = DriveMode.MANUAL

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

    def _draw_camera_waypoints(
        self,
        waypoint_manager: WaypointManager,
        camera_sensor: CameraSensor,
        vehicle: "carla.Vehicle",
    ) -> None:
        if self.route_planner is not None and self.route_planner.get_route():
            overlay_waypoints = self.waypoint_tracker.get_preview_waypoints()
            target_waypoint = self._latest_tracking.target_waypoint
        else:
            overlay_waypoints = waypoint_manager.get_future_waypoints(vehicle)
            target_waypoint = None

        self._overlay_renderer.draw(
            surface=self._display.surface,
            waypoints=overlay_waypoints,
            camera=camera_sensor.actor,
            vehicle=vehicle,
            target_waypoint=target_waypoint,
            surface_offset=self._display.main_view_rect.topleft,
        )

    def _draw_topdown_map(self) -> None:
        assert self._display is not None
        if self._topdown_renderer is None:
            return
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

    def _draw_sensor_panel(self) -> None:
        assert self._display is not None
        route = self.route_planner.get_route() if self.route_planner is not None else []
        gnss = self._gnss_sensor.get_latest_measurement() if self._gnss_sensor is not None else None
        imu = self._imu_sensor.get_latest_measurement() if self._imu_sensor is not None else None
        lidar = self._lidar_sensor.get_latest_measurement() if self._lidar_sensor is not None else None
        data = SensorPanelData(
            drive_mode=self._drive_mode.value,
            map_selection_active=self._map_selection_active,
            sync_status=self._client_manager.sync_status,
            fixed_delta_seconds=self._client_manager.fixed_delta_seconds,
            pygame_frame_dt_seconds=self._clock.last_frame_dt_seconds,
            planner_status=self._planner_status,
            route_size=len(route),
            ego_state=self._latest_state,
            ground_truth_state=self._latest_ground_truth_state,
            estimated_state=self._latest_estimated_state,
            localization_status=self._latest_localization_status,
            route_activation_state=self._route_activation_state.value,
            stabilization_active=(
                self._route_activation_state == RouteActivationState.WAITING_FOR_LOCALIZATION_STABILITY
            ),
            stabilization_error_m=self._stabilization_error_m,
            stabilization_stable_ticks=self._stabilization_stable_ticks,
            stabilization_required_ticks=ROUTE_INITIALIZATION.stable_ticks_required,
            stabilization_elapsed_seconds=self._stabilization_elapsed_seconds,
            stabilization_timeout_seconds=ROUTE_INITIALIZATION.max_wait_seconds,
            route_generation_blocked=self._route_generation_blocked,
            tracking=self._latest_tracking,
            gnss=gnss,
            gnss_diagnostics=self._latest_gnss_diagnostics,
            imu=imu,
            lidar=lidar,
        )
        self._sensor_panel_renderer.draw(
            surface=self._display.surface,
            rect=self._display.sensor_panel_rect,
            data=data,
        )

    def _draw_control_panel(self) -> None:
        assert self._display is not None
        assert self._control_panel is not None
        self._update_control_panel_state()
        self._control_panel.draw(self._display.surface)

    def _update_test_performance(self) -> None:
        runner = self._test_runner
        if runner is None or not runner.is_active:
            return
        logger = runner.current_logger
        if logger is None:
            return

        route_name = runner.current_route_name or "test_route"
        phase = self._benchmark_phase()
        if BENCHMARK.collect_stabilization_samples or phase != "stabilization":
            logger.collect_sample(
                phase=phase,
                route_name=route_name,
                ground_truth_state=self._latest_ground_truth_state,
                kalman_state=self._latest_estimated_state,
                gnss_diagnostics=self._latest_gnss_diagnostics,
                tracking=self._latest_tracking,
                route_completed=self._latest_tracking.completed,
            )

        route = self.route_planner.get_route() if self.route_planner is not None else []
        route_failed = (
            self._route_activation_state == RouteActivationState.IDLE
            and not route
            and not self._latest_tracking.completed
        )
        paths = runner.update(
            route_completed=self._latest_tracking.completed,
            route_failed=route_failed,
        )
        if paths is not None:
            self._control_status_text = runner.status_text
            self._planner_status = runner.status_text

    def _benchmark_phase(self) -> str:
        if self._latest_tracking.completed:
            return "completed"
        if self._route_activation_state == RouteActivationState.WAITING_FOR_LOCALIZATION_STABILITY:
            return "stabilization"
        if self._route_activation_state == RouteActivationState.ROUTE_ACTIVE:
            return "driving"
        return "idle"

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

        if self._vehicle is not None:
            self._vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

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
