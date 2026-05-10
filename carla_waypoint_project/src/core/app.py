"""Main simulation application orchestrator."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import pygame

from src.control.vehicle_controller import VehicleController
from src.control.waypoint_tracker import TrackingStatus, WaypointTracker
from src.core.carla_client import CarlaClientManager
from src.core.simulation import SimulationClock
from src.localization.state_estimator import EgoState, GroundTruthStateProvider
from src.planning.map_selector import MapSelector
from src.planning.route_planner import RoutePlanner
from src.planning.waypoint_manager import WaypointManager
from src.sensors.camera_sensor import CameraSensor
from src.sensors.sensor_manager import SensorManager
from src.vehicle.manual_controller import ManualController
from src.vehicle.vehicle_manager import VehicleManager
from src.visualization.pygame_display import PygameDisplay
from src.visualization.topdown_map import TopDownHudData, TopDownMapRenderer
from src.visualization.waypoint_overlay import WaypointOverlayRenderer
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class DriveMode(Enum):
    """High-level ego driving mode."""

    MANUAL = "MANUAL"
    AUTONOMOUS = "AUTO"


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
        self._vehicle: Optional["carla.Vehicle"] = None

        self._overlay_renderer = WaypointOverlayRenderer()
        self._state_provider: Optional[GroundTruthStateProvider] = None
        self._map_selector: Optional[MapSelector] = None
        self._topdown_renderer: Optional[TopDownMapRenderer] = None
        self.route_planner: Optional[RoutePlanner] = None
        self.waypoint_tracker = WaypointTracker()
        self.autonomous_controller = VehicleController()

        self._drive_mode = DriveMode.MANUAL
        self._map_selection_active = False
        self._latest_state: Optional[EgoState] = None
        self._latest_tracking = TrackingStatus(
            target_waypoint=None,
            closest_index=0,
            target_index=0,
            cross_track_error_m=float("inf"),
            completed=False,
        )
        self._planner_status = ""

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

        world_map = self._client_manager.world_map
        self._waypoint_manager = WaypointManager(world_map=world_map)
        self._manual_controller = ManualController(vehicle=self._vehicle)
        self._state_provider = GroundTruthStateProvider(vehicle=self._vehicle)
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
            "waypoint manager": self._waypoint_manager,
            "manual controller": self._manual_controller,
            "state provider": self._state_provider,
            "map selector": self._map_selector,
            "route planner": self.route_planner,
            "top-down renderer": self._topdown_renderer,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(f"Application is not initialized: {', '.join(missing)}.")

    def run(self) -> None:
        """Run the camera, route-selection, and route-following application loop."""
        self._setup()
        self._ensure_ready()

        manual_controller = self._manual_controller
        camera_sensor = self._camera_sensor
        waypoint_manager = self._waypoint_manager
        vehicle = self._vehicle
        state_provider = self._state_provider

        assert manual_controller is not None
        assert camera_sensor is not None
        assert waypoint_manager is not None
        assert vehicle is not None
        assert state_provider is not None

        running = True
        try:
            while running:
                running = self._process_events()
                self._latest_state = state_provider.get_state()
                self._latest_tracking = self.waypoint_tracker.update(self._latest_state)

                if self._drive_mode == DriveMode.AUTONOMOUS:
                    control = self.autonomous_controller.compute_control(
                        state=self._latest_state,
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
                self._display.end_frame()
                self._clock.tick()
        finally:
            self.shutdown()

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

        consumed = renderer.handle_mouse_button_down(self._display.surface, event)
        if consumed:
            return

        if event.button != 1:
            return

        world_location = renderer.screen_to_world(self._display.surface, event.pos)
        if world_location is None:
            return

        assert self._map_selector is not None
        self._map_selector.select_world_location(world_location)
        self._drive_mode = DriveMode.MANUAL
        self._clear_route()
        if self._map_selector.endpoints is not None:
            self._generate_route_from_selection(teleport_to_start=True, start_autonomous=True)

    def _handle_mouse_button_up(self, event: pygame.event.Event) -> None:
        if self._topdown_renderer is not None:
            self._topdown_renderer.handle_mouse_button_up(event)

    def _handle_mouse_motion(self, event: pygame.event.Event) -> None:
        if self._topdown_renderer is not None and self._map_selection_active:
            self._topdown_renderer.handle_mouse_motion(self._display.surface, event)

    def _handle_mouse_wheel(self, event: pygame.event.Event) -> None:
        if self._topdown_renderer is None or not self._map_selection_active:
            return
        self._topdown_renderer.handle_mouse_wheel(
            surface=self._display.surface,
            position=pygame.mouse.get_pos(),
            wheel_y=event.y,
        )

    def _try_enable_autonomous_mode(self) -> None:
        if self.route_planner is None:
            return
        if not self.route_planner.get_route():
            self._generate_route_from_selection(teleport_to_start=True, start_autonomous=False)
        if self.route_planner.get_route():
            self._drive_mode = DriveMode.AUTONOMOUS
            if self._vehicle is not None:
                self._vehicle.set_autopilot(False)

    def _generate_route_from_selection(
        self,
        teleport_to_start: bool = True,
        start_autonomous: bool = False,
    ) -> None:
        if self._map_selector is None or self.route_planner is None:
            return

        endpoints = self._map_selector.endpoints
        if endpoints is None:
            self._planner_status = "Planner: select A/B"
            return

        route = self.route_planner.generate_route(endpoints.start, endpoints.goal)
        self.waypoint_tracker.set_route(route)
        if route:
            if teleport_to_start:
                self._teleport_vehicle_to_route_start(endpoints.start)
            if start_autonomous:
                self._drive_mode = DriveMode.AUTONOMOUS
            self._planner_status = f"Planner: {len(route)} wp, driving A->B"
        elif self.route_planner.planner_error:
            self._planner_status = self.route_planner.planner_error
        else:
            self._planner_status = "Planner: no route"

    def _teleport_vehicle_to_route_start(self, start_waypoint: "carla.Waypoint") -> None:
        if self._vehicle_manager is None:
            return
        self._vehicle_manager.teleport_to_waypoint(start_waypoint)
        if self._state_provider is not None:
            self._latest_state = self._state_provider.get_state()
            self._latest_tracking = self.waypoint_tracker.update(self._latest_state)

    def _reset_selection_and_route(self) -> None:
        if self._map_selector is not None:
            self._map_selector.reset()
        self._clear_route()
        self._planner_status = "Planner: reset"

    def _clear_route(self) -> None:
        if self.route_planner is not None:
            self.route_planner.clear_route()
        self.waypoint_tracker.clear_route()
        self._drive_mode = DriveMode.MANUAL

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
        )

    def _draw_topdown_map(self) -> None:
        if not self._map_selection_active or self._topdown_renderer is None:
            return

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
            ego_state=self._latest_state,
            start_waypoint=start,
            goal_waypoint=goal,
            route=route,
            target_waypoint=self._latest_tracking.target_waypoint,
        )

    def shutdown(self) -> None:
        """Destroy actors and close pygame resources."""
        if self._vehicle is not None:
            self._vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

        if self._sensor_manager is not None:
            self._sensor_manager.destroy_all()
            self._sensor_manager = None

        if self._vehicle_manager is not None:
            self._vehicle_manager.destroy()
            self._vehicle_manager = None

        self._display.shutdown()
