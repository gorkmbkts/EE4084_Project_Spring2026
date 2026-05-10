"""Top-down pygame route map rendering and screen/world conversion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

import pygame

from config.settings import ROUTE_PLANNER, TOPDOWN_MAP
from src.localization.state_estimator import EgoState
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class TopDownHudData:
    """Debug values rendered inside the map panel."""

    drive_mode: str
    selection_active: bool
    route_size: int
    closest_index: int
    target_index: int
    route_completed: bool
    speed_mps: float
    planner_status: str


class TopDownMapRenderer:
    """Render CARLA road topology, route state, and selection markers."""

    def __init__(self, world_map: "carla.Map") -> None:
        self._world_map = world_map
        self._road_polylines = self._build_road_polylines()
        self._bounds = self._compute_bounds(self._road_polylines)
        self._center_x = (self._bounds[0] + self._bounds[1]) * 0.5
        self._center_y = (self._bounds[2] + self._bounds[3]) * 0.5
        self._zoom = 1.0
        self._is_panning = False
        self._last_pan_pos: Optional[tuple[int, int]] = None
        self._font = pygame.font.SysFont("consolas", 15)
        self._small_font = pygame.font.SysFont("consolas", 13)

    def get_panel_rect(self, surface: pygame.Surface) -> pygame.Rect:
        """Return the current top-down panel rectangle."""
        width = min(TOPDOWN_MAP.panel_width, surface.get_width() - 2 * TOPDOWN_MAP.margin_px)
        height = min(TOPDOWN_MAP.panel_height, surface.get_height() - 2 * TOPDOWN_MAP.margin_px)
        return pygame.Rect(
            surface.get_width() - width - TOPDOWN_MAP.margin_px,
            TOPDOWN_MAP.margin_px,
            width,
            height,
        )

    def contains_screen_point(
        self,
        surface: pygame.Surface,
        position: tuple[int, int],
        panel_rect: Optional[pygame.Rect] = None,
    ) -> bool:
        """Return whether a screen point is inside the top-down map panel."""
        rect = panel_rect if panel_rect is not None else self.get_panel_rect(surface)
        return rect.collidepoint(position)

    def screen_to_world(
        self,
        surface: pygame.Surface,
        position: tuple[int, int],
        panel_rect: Optional[pygame.Rect] = None,
    ) -> Optional["carla.Location"]:
        """Convert a point in the pygame window to a CARLA world location."""
        rect = panel_rect if panel_rect is not None else self.get_panel_rect(surface)
        if not rect.collidepoint(position):
            return None

        scale = self._scale_for_rect(rect)
        x = self._center_x + (position[0] - rect.centerx) / scale
        y = self._center_y - (position[1] - rect.centery) / scale
        return carla.Location(x=float(x), y=float(y), z=0.0)

    def handle_mouse_button_down(
        self,
        surface: pygame.Surface,
        event: pygame.event.Event,
        panel_rect: Optional[pygame.Rect] = None,
    ) -> bool:
        """Handle map zoom or pan start. Returns True if event was consumed."""
        if not self.contains_screen_point(surface, event.pos, panel_rect):
            return False

        if event.button == 4:
            self._zoom_at(surface, event.pos, TOPDOWN_MAP.zoom_step, panel_rect)
            return True
        if event.button == 5:
            self._zoom_at(surface, event.pos, 1.0 / TOPDOWN_MAP.zoom_step, panel_rect)
            return True
        if event.button in (2, 3):
            self._is_panning = True
            self._last_pan_pos = event.pos
            return True
        return False

    def handle_mouse_button_up(self, event: pygame.event.Event) -> None:
        """Stop active mouse panning."""
        if event.button in (2, 3):
            self._is_panning = False
            self._last_pan_pos = None

    def handle_mouse_motion(
        self,
        surface: pygame.Surface,
        event: pygame.event.Event,
        panel_rect: Optional[pygame.Rect] = None,
    ) -> bool:
        """Pan the map while right or middle mouse is held."""
        if not self._is_panning or self._last_pan_pos is None:
            return False

        rect = panel_rect if panel_rect is not None else self.get_panel_rect(surface)
        scale = self._scale_for_rect(rect)
        dx = event.pos[0] - self._last_pan_pos[0]
        dy = event.pos[1] - self._last_pan_pos[1]
        self._center_x -= dx / scale
        self._center_y += dy / scale
        self._last_pan_pos = event.pos
        return True

    def handle_mouse_wheel(
        self,
        surface: pygame.Surface,
        position: tuple[int, int],
        wheel_y: int,
        panel_rect: Optional[pygame.Rect] = None,
    ) -> bool:
        """Zoom the map around the mouse position."""
        if not self.contains_screen_point(surface, position, panel_rect):
            return False

        zoom_factor = TOPDOWN_MAP.zoom_step if wheel_y > 0 else 1.0 / TOPDOWN_MAP.zoom_step
        self._zoom_at(surface, position, zoom_factor, panel_rect)
        return True

    def draw(
        self,
        surface: pygame.Surface,
        hud: TopDownHudData,
        ego_state: Optional[EgoState],
        start_waypoint: Optional["carla.Waypoint"],
        goal_waypoint: Optional["carla.Waypoint"],
        route: Sequence["carla.Waypoint"],
        target_waypoint: Optional["carla.Waypoint"],
        panel_rect: Optional[pygame.Rect] = None,
    ) -> None:
        """Draw the complete top-down map panel."""
        rect = panel_rect if panel_rect is not None else self.get_panel_rect(surface)
        pygame.draw.rect(surface, TOPDOWN_MAP.background_color, rect)
        pygame.draw.rect(surface, TOPDOWN_MAP.border_color, rect, width=1)

        old_clip = surface.get_clip()
        surface.set_clip(rect)
        for polyline in self._road_polylines:
            points = [self._world_to_screen(rect, point[0], point[1]) for point in polyline]
            if len(points) >= 2:
                pygame.draw.lines(surface, TOPDOWN_MAP.road_color, False, points, 2)

        self._draw_route(surface, rect, route)
        self._draw_endpoint(surface, rect, start_waypoint, TOPDOWN_MAP.start_color, "A")
        self._draw_endpoint(surface, rect, goal_waypoint, TOPDOWN_MAP.goal_color, "B")
        self._draw_target(surface, rect, target_waypoint)
        self._draw_vehicle(surface, rect, ego_state)
        self._draw_hud(surface, rect, hud, start_waypoint, goal_waypoint)
        surface.set_clip(old_clip)

    def _build_road_polylines(self) -> List[List[Point2D]]:
        polylines: List[List[Point2D]] = []
        sample_distance = max(ROUTE_PLANNER.sampling_resolution_m * 2.0, 4.0)

        for entry_wp, exit_wp in self._world_map.get_topology():
            path = [entry_wp]
            current_wp = entry_wp
            exit_location = exit_wp.transform.location
            for _ in range(1500):
                if current_wp.transform.location.distance(exit_location) <= sample_distance:
                    break
                next_wps = current_wp.next(sample_distance)
                if not next_wps:
                    break
                current_wp = next_wps[0]
                path.append(current_wp)
            path.append(exit_wp)
            polylines.append([(wp.transform.location.x, wp.transform.location.y) for wp in path])

        return polylines

    @staticmethod
    def _compute_bounds(polylines: Sequence[Sequence[Point2D]]) -> tuple[float, float, float, float]:
        xs = [point[0] for polyline in polylines for point in polyline]
        ys = [point[1] for polyline in polylines for point in polyline]
        if not xs or not ys:
            return -100.0, 100.0, -100.0, 100.0

        margin = TOPDOWN_MAP.world_margin_m
        return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin

    def _scale_for_rect(self, rect: pygame.Rect) -> float:
        min_x, max_x, min_y, max_y = self._bounds
        world_width = max(1.0, max_x - min_x)
        world_height = max(1.0, max_y - min_y)
        base_scale = min(rect.width / world_width, rect.height / world_height)
        return base_scale * self._zoom

    def _world_to_screen(self, rect: pygame.Rect, x: float, y: float) -> tuple[int, int]:
        scale = self._scale_for_rect(rect)
        screen_x = rect.centerx + (x - self._center_x) * scale
        screen_y = rect.centery - (y - self._center_y) * scale
        return int(screen_x), int(screen_y)

    def _zoom_at(
        self,
        surface: pygame.Surface,
        position: tuple[int, int],
        zoom_factor: float,
        panel_rect: Optional[pygame.Rect] = None,
    ) -> None:
        before = self.screen_to_world(surface, position, panel_rect)
        self._zoom = max(TOPDOWN_MAP.min_zoom, min(TOPDOWN_MAP.max_zoom, self._zoom * zoom_factor))
        after = self.screen_to_world(surface, position, panel_rect)
        if before is None or after is None:
            return
        self._center_x += before.x - after.x
        self._center_y += before.y - after.y

    def _draw_route(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        route: Sequence["carla.Waypoint"],
    ) -> None:
        if len(route) < 2:
            return
        points = [
            self._world_to_screen(rect, waypoint.transform.location.x, waypoint.transform.location.y)
            for waypoint in route
        ]
        pygame.draw.lines(surface, TOPDOWN_MAP.route_color, False, points, 3)

    def _draw_endpoint(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        waypoint: Optional["carla.Waypoint"],
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        if waypoint is None:
            return
        location = waypoint.transform.location
        position = self._world_to_screen(rect, location.x, location.y)
        pygame.draw.circle(surface, color, position, 7)
        label_surface = self._font.render(label, True, color)
        surface.blit(label_surface, (position[0] + 8, position[1] - 9))

    def _draw_target(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        waypoint: Optional["carla.Waypoint"],
    ) -> None:
        if waypoint is None:
            return
        location = waypoint.transform.location
        position = self._world_to_screen(rect, location.x, location.y)
        pygame.draw.circle(surface, TOPDOWN_MAP.target_color, position, 6, width=2)

    def _draw_vehicle(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        ego_state: Optional[EgoState],
    ) -> None:
        if ego_state is None:
            return

        yaw = math.radians(ego_state.yaw)
        size_m = 4.5
        half_width_m = 1.7
        points_world = [
            (ego_state.x + math.cos(yaw) * size_m, ego_state.y + math.sin(yaw) * size_m),
            (
                ego_state.x + math.cos(yaw + 2.45) * half_width_m,
                ego_state.y + math.sin(yaw + 2.45) * half_width_m,
            ),
            (
                ego_state.x + math.cos(yaw - 2.45) * half_width_m,
                ego_state.y + math.sin(yaw - 2.45) * half_width_m,
            ),
        ]
        points = [self._world_to_screen(rect, x, y) for x, y in points_world]
        pygame.draw.polygon(surface, TOPDOWN_MAP.vehicle_color, points)

    def _draw_hud(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        hud: TopDownHudData,
        start_waypoint: Optional["carla.Waypoint"],
        goal_waypoint: Optional["carla.Waypoint"],
    ) -> None:
        rows = [
            f"Mode: {hud.drive_mode}",
            f"Map select: {'ON' if hud.selection_active else 'OFF'}",
            f"A: {'set' if start_waypoint else 'not set'}  B: {'set' if goal_waypoint else 'not set'}",
            f"Route: {hud.route_size} wp",
            f"Closest: {hud.closest_index}  Target: {hud.target_index}",
            f"Speed: {hud.speed_mps:.1f} m/s",
            f"Done: {'yes' if hud.route_completed else 'no'}",
        ]
        if hud.planner_status:
            rows.append(hud.planner_status[:42])

        x = rect.left + 8
        y = rect.top + 26
        for row in rows:
            text_surface = self._small_font.render(row, True, TOPDOWN_MAP.text_color)
            surface.blit(text_surface, (x, y))
            y += 17
