"""Main simulation application orchestrator."""

from __future__ import annotations

from collections import deque
from enum import Enum
import time
from typing import Optional

import pygame

from config.settings import ROUTE_INITIALIZATION, TOPDOWN_MAP
from src.control.vehicle_controller import VehicleController
from src.control.waypoint_tracker import TrackingStatus, WaypointTracker
from src.core.carla_client import CarlaClientManager
from src.core.simulation import SimulationClock
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
        self._client_manager = CarlaClientManager()
        self._display = PygameDisplay()
        self._clock = SimulationClock()

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
        self.route_planner: Optional[RoutePlanner] = None
        self.waypoint_tracker = WaypointTracker()
        self.autonomous_controller = VehicleController()

        self._drive_mode = DriveMode.MANUAL
        self._map_selection_active = False
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

        if self.route_planner.planner_error:
            self._planner_status = "Planner: fallback"
        else:
            self._planner_status = "Planner: CARLA"

    def _ensure_ready(self) -> None:
        required = {
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
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(f"Application is not initialized: {', '.join(missing)}.")

    def run(self) -> None:
        """Run the camera, route-selection, and route-following application loop."""
        try:
            self._setup()
            self._ensure_ready()

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

                if self._route_activation_state == RouteActivationState.WAITING_FOR_LOCALIZATION_STABILITY:
                    control = carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=ROUTE_INITIALIZATION.hold_brake,
                        hand_brake=False,
                    )
                    vehicle.apply_control(control)
                elif self._drive_mode == DriveMode.AUTONOMOUS:
                    if self._latest_estimated_state is None:
                        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
                    else:
                        control = self.autonomous_controller.compute_control(
                            state=self._latest_estimated_state,
                            target_waypoint=self._latest_tracking.target_waypoint,
                            route_completed=self._latest_tracking.completed,
                        )
                    vehicle.apply_control(control)
                else:
                    manual_controller.apply_control()

                camera_surface = camera_sensor.get_latest_surface()
                self._display.begin_frame(camera_surface)
                self._draw_camera_waypoints(waypoint_manager, camera_sensor, vehicle)
                self._draw_topdown_map()
                self._draw_lidar_panel()
                self._draw_sensor_panel()
                self._display.end_frame()
                self._clock.tick_pygame()
        finally:
            self.shutdown()

    def _state_for_tracking_and_control(self) -> Optional[EgoState]:
        """Return GT in manual mode and estimated state in autonomous mode."""
        if self._drive_mode == DriveMode.AUTONOMOUS:
            return self._latest_estimated_state
        return self._latest_ground_truth_state

    def _can_update_route_tracking(self) -> bool:
        if self._route_activation_state == RouteActivationState.WAITING_FOR_LOCALIZATION_STABILITY:
            return False
        return self.route_planner is not None and bool(self.route_planner.get_route())

    def _process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if not self._handle_key_down(event):
                    return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_button_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._handle_mouse_button_up(event)
            elif event.type == pygame.MOUSEMOTION:
                self._handle_mouse_motion(event)
            elif event.type == pygame.MOUSEWHEEL:
                self._handle_mouse_wheel(event)
        return True

    def _handle_key_down(self, event: pygame.event.Event) -> bool:
        if event.key == pygame.K_ESCAPE:
            return False
        if event.key == pygame.K_m:
            self._cancel_route_activation()
            self._drive_mode = DriveMode.MANUAL
            return True
        if event.key == pygame.K_p:
            self._try_enable_autonomous_mode()
            return True
        if event.key == pygame.K_t:
            self._map_selection_active = not self._map_selection_active
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

    def _handle_mouse_button_down(self, event: pygame.event.Event) -> None:
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
        if self._map_selector.endpoints is not None:
            self._begin_route_initialization_from_selection(start_autonomous=True)

    def _handle_mouse_button_up(self, event: pygame.event.Event) -> None:
        if self._topdown_renderer is not None:
            self._topdown_renderer.handle_mouse_button_up(event)

    def _handle_mouse_motion(self, event: pygame.event.Event) -> None:
        if self._topdown_renderer is not None and self._map_selection_active:
            self._topdown_renderer.handle_mouse_motion(
                self._display.surface,
                event,
                panel_rect=self._display.map_rect,
            )

    def _handle_mouse_wheel(self, event: pygame.event.Event) -> None:
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

    def _clear_route(self) -> None:
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
        measurement = None
        if self._lidar_sensor is not None:
            measurement = self._lidar_sensor.get_latest_measurement()
        self._lidar_renderer.draw(
            surface=self._display.surface,
            rect=self._display.lidar_rect,
            measurement=measurement,
        )

    def _draw_sensor_panel(self) -> None:
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

    def shutdown(self) -> None:
        """Destroy actors and close pygame resources."""
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
        self._display.shutdown()
