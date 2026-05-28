"""Pre-dashboard pygame UI for CARLA startup status and map selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Optional

import pygame

from config.settings import CARLA, DASHBOARD, DISPLAY
from src.KalmanLab.registry import discover_filters
from src.evaluation.benchmark_config import (
    BEHAVIOR_PRESETS,
    BEHAVIOR_SPECS,
    BenchmarkConfig,
    SENSOR_NOISE_PRESETS,
    SENSOR_NOISE_SPECS,
    behavior_values_from_config,
    driving_behavior_from_values,
    load_available_test_routes,
    sensor_noise_config_from_values,
    validate_benchmark_config,
)
from src.control.driving_behavior import DrivingBehaviorConfig
from src.evaluation.test_route_store import SavedTestRoute
from src.utils.map_names import display_map_name, maps_compatible, normalize_map_name
from src.visualization.ui.parameter_controls import ParameterEditor
from src.visualization.windowing import create_display_surface, display_flags_from_settings


_COMMON_FALLBACK_MAPS = (
    "Town01",
    "Town01_Opt",
    "Town02",
    "Town02_Opt",
    "Town03",
    "Town03_Opt",
    "Town04",
    "Town04_Opt",
    "Town05",
    "Town05_Opt",
    "Town10HD",
    "Town10HD_Opt",
)


@dataclass(frozen=True)
class StartupMapSelection:
    """Result returned after a startup map is selected and loaded."""

    selected_map_load_name: Optional[str]
    active_map_name: Optional[str]
    used_current_map: bool
    benchmark_config: Optional[BenchmarkConfig] = None


@dataclass(frozen=True)
class _MapOption:
    load_name: Optional[str]
    display_name: str
    detail: str
    guaranteed: bool
    is_current: bool = False


class StartupMapSelector:
    """Full-window startup status and map-selection screen."""

    def __init__(self, width: int = DISPLAY.width, height: int = DISPLAY.height) -> None:
        pygame.init()
        self._surface = create_display_surface(width=width, height=height, title=DISPLAY.title)
        self._clock = pygame.time.Clock()
        self._init_fonts()
        self._status = "Not running"
        self._detail = ""
        self._error = ""
        self._executable_path = ""
        self._current_map_name: Optional[str] = None
        self._available_count = 0
        self._options: list[_MapOption] = []
        self._selected_index = 0
        self._scroll_offset = 0
        self._hovered_index: Optional[int] = None
        self._row_rects: dict[int, pygame.Rect] = {}
        self._map_list_top = 0
        self._start_button_rect = pygame.Rect(0, 0, 1, 1)
        self._use_current_button_rect = pygame.Rect(0, 0, 1, 1)
        self._refresh_button_rect = pygame.Rect(0, 0, 1, 1)
        self._project_root = Path(__file__).resolve().parents[2]
        self._runtime_state_path = self._project_root / "config" / "runtime_state.json"
        self._active_tab = "Map Selection"
        self._tab_rects: dict[str, pygame.Rect] = {}
        self._setup_filter_records = []
        self._setup_filter_buttons: dict[str, pygame.Rect] = {}
        self._selected_filter_id = ""
        self._sensor_editor: Optional[ParameterEditor] = None
        self._behavior_editor: Optional[ParameterEditor] = None
        self._sensor_preset = "Medium Noise"
        self._behavior_preset = "Balanced"
        self._route_items = []
        self._selected_route_indices: set[int] = set()
        self._route_scroll = 0
        self._route_rects: dict[int, pygame.Rect] = {}
        self._select_all_routes_rect = pygame.Rect(0, 0, 1, 1)
        self._clear_routes_rect = pygame.Rect(0, 0, 1, 1)
        self._start_benchmark_rect = pygame.Rect(0, 0, 1, 1)
        self._setup_summary_lines: list[str] = []

    @property
    def surface(self) -> pygame.Surface:
        return self._surface

    def show_status(
        self,
        status: str,
        detail: Optional[str] = None,
        executable_path: Optional[Path] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Draw a startup status frame and return False if the user cancels."""
        self._ensure_display_ready()
        self._status = status
        self._detail = detail or ""
        if executable_path is not None:
            self._executable_path = str(executable_path)
        self._error = error or self._error
        if not self._pump_basic_events():
            return False
        self._draw(status_only=True)
        self._clock.tick(30)
        return True

    def wait_for_error_ack(self, message: str) -> None:
        """Show a startup error until the user closes the window or presses ESC."""
        self._ensure_display_ready()
        self._error = message
        self._detail = "Press ESC or close the window to quit."
        running = True
        while running:
            running = self._pump_basic_events()
            self._draw(status_only=True)
            self._clock.tick(30)

    def choose_map(
        self,
        client: object,
        executable_path: Optional[Path] = None,
    ) -> Optional[StartupMapSelection]:
        """Run the selector until a map is loaded, current map is accepted, or user quits."""
        self._ensure_display_ready()
        self._status = "Connected"
        self._detail = "Select a map before the dashboard starts."
        if executable_path is not None:
            self._executable_path = str(executable_path)
        self._refresh_options(client, preserve_selection=False)
        self._refresh_test_setup()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.VIDEORESIZE:
                    self._resize(event.w, event.h)
                    continue
                if self._active_tab == "Test Setup":
                    result = self._handle_test_setup_event(event, client)
                    if result is not _NoSelection:
                        return result
                    continue
                if event.type == pygame.KEYDOWN:
                    result = self._handle_key_down(event, client)
                    if result is not _NoSelection:
                        return result
                elif event.type == pygame.MOUSEWHEEL:
                    self._scroll_by(-event.y * 3)
                elif event.type == pygame.MOUSEMOTION:
                    self._update_hover(event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    result = self._handle_left_click(event, client)
                    if result is not _NoSelection:
                        return result

            self._draw(status_only=False)
            self._clock.tick(30)

        return None

    def _handle_key_down(self, event: pygame.event.Event, client: object) -> object:
        if event.key == pygame.K_ESCAPE:
            return None
        if event.key == pygame.K_TAB:
            self._active_tab = "Test Setup" if self._active_tab == "Map Selection" else "Map Selection"
            return _NoSelection
        if event.key == pygame.K_UP:
            self._move_selection(-1)
            return _NoSelection
        if event.key == pygame.K_DOWN:
            self._move_selection(1)
            return _NoSelection
        if event.key == pygame.K_PAGEUP:
            self._move_selection(-8)
            return _NoSelection
        if event.key == pygame.K_PAGEDOWN:
            self._move_selection(8)
            return _NoSelection
        if event.key == pygame.K_u:
            return self._use_current_map(client)
        if event.key == pygame.K_r:
            self._refresh_options(client, preserve_selection=True)
            return _NoSelection
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return self._load_selected_option(client)
        return _NoSelection

    def _use_current_map(self, client: object) -> Optional[StartupMapSelection]:
        current_map = self._read_valid_current_map(client)
        if current_map is None:
            self._error = "Current CARLA world is not ready."
            return _NoSelection
        self._write_runtime_state(selected_load_name=None, active_map_name=current_map)
        return StartupMapSelection(
            selected_map_load_name=None,
            active_map_name=current_map,
            used_current_map=True,
        )

    def _load_selected_option(self, client: object) -> object:
        if not self._options:
            self._error = "No maps are available. Use U if the current map is readable."
            return _NoSelection

        option = self._options[self._selected_index]
        if option.is_current or option.load_name is None:
            return self._use_current_map(client)

        self._status = "Loading map"
        self._detail = f"Loading {option.display_name}..."
        self._error = ""
        self._draw(status_only=False)
        pygame.display.flip()
        pygame.event.pump()

        try:
            setter = getattr(client, "set_timeout", None)
            if setter is not None:
                setter(max(float(CARLA.timeout_seconds), 60.0))
            world = client.load_world(option.load_name)
            world_map = world.get_map()
            blueprint_library = world.get_blueprint_library()
            if world_map is None or blueprint_library is None:
                raise RuntimeError("Loaded world did not expose map metadata.")
            active_map_name = getattr(world_map, "name", None)
        except Exception as exc:
            self._status = "Connected"
            self._detail = "Select a map before the dashboard starts."
            self._error = f"Map load failed: {exc}"
            self._refresh_options(client, preserve_selection=True)
            return _NoSelection

        self._write_runtime_state(
            selected_load_name=option.load_name,
            active_map_name=active_map_name,
        )
        return StartupMapSelection(
            selected_map_load_name=option.load_name,
            active_map_name=active_map_name,
            used_current_map=False,
        )

    def _refresh_options(self, client: object, preserve_selection: bool) -> None:
        previous_load_name = None
        previous_display = None
        if preserve_selection and self._options:
            previous = self._options[self._selected_index]
            previous_load_name = previous.load_name
            previous_display = previous.display_name

        self._current_map_name = self._read_current_map(client)
        available_maps, listing_error = self._read_available_maps(client)
        self._available_count = len(available_maps)
        self._options = self._build_options(self._current_map_name, available_maps)

        if listing_error:
            self._error = listing_error
        elif self._error.startswith("Map list"):
            self._error = ""

        self._selected_index = self._preferred_index(
            previous_load_name=previous_load_name,
            previous_display=previous_display,
        )
        self._scroll_offset = 0
        self._ensure_selection_visible()

    def _build_options(
        self,
        current_map_name: Optional[str],
        available_maps: list[str],
    ) -> list[_MapOption]:
        options: list[_MapOption] = []
        if current_map_name:
            options.append(
                _MapOption(
                    load_name=None,
                    display_name=f"Use currently loaded map ({display_map_name(current_map_name)})",
                    detail="Current world, no load_world call",
                    guaranteed=True,
                    is_current=True,
                )
            )

        seen_load_names: set[str] = set()
        for map_name in sorted(available_maps, key=lambda item: display_map_name(item).casefold()):
            if map_name in seen_load_names:
                continue
            seen_load_names.add(map_name)
            options.append(
                _MapOption(
                    load_name=map_name,
                    display_name=display_map_name(map_name),
                    detail=map_name,
                    guaranteed=True,
                )
            )

        if not available_maps:
            for map_name in _COMMON_FALLBACK_MAPS:
                options.append(
                    _MapOption(
                        load_name=map_name,
                        display_name=map_name,
                        detail="Fallback/manual hint, not guaranteed",
                        guaranteed=False,
                    )
                )

        return options

    def _preferred_index(
        self,
        previous_load_name: Optional[str],
        previous_display: Optional[str],
    ) -> int:
        if not self._options:
            return 0

        if previous_load_name or previous_display:
            for index, option in enumerate(self._options):
                if previous_load_name and option.load_name == previous_load_name:
                    return index
                if previous_display and option.display_name == previous_display:
                    return index

        runtime_state = self._read_runtime_state()
        last_load_name = runtime_state.get("last_map_load_name")
        last_active_map_name = runtime_state.get("last_active_map_name")
        for index, option in enumerate(self._options):
            if isinstance(last_load_name, str) and option.load_name == last_load_name:
                return index
        for index, option in enumerate(self._options):
            if isinstance(last_active_map_name, str) and maps_compatible(option.display_name, last_active_map_name):
                return index
        return 0

    def _read_current_map(self, client: object) -> Optional[str]:
        try:
            world = client.get_world()
            world_map = world.get_map()
            return getattr(world_map, "name", None)
        except Exception as exc:
            self._error = f"Current map read failed: {exc}"
            return None

    def _read_valid_current_map(self, client: object) -> Optional[str]:
        try:
            world = client.get_world()
            world_map = world.get_map()
            blueprint_library = world.get_blueprint_library()
            if world_map is None or blueprint_library is None:
                raise RuntimeError("World metadata is incomplete.")
            return getattr(world_map, "name", None)
        except Exception as exc:
            self._error = f"Current world validation failed: {exc}"
            return None

    def _read_available_maps(self, client: object) -> tuple[list[str], Optional[str]]:
        getter = getattr(client, "get_available_maps", None)
        if getter is None:
            return [], "Map list unavailable: CARLA client does not expose get_available_maps()."
        try:
            raw_maps = getter()
        except Exception as exc:
            return [], f"Map list unavailable: {exc}"
        maps = [str(item) for item in raw_maps or [] if str(item).strip()]
        if not maps:
            return [], "Map list unavailable: get_available_maps() returned no maps."
        return maps, None

    def _handle_left_click(self, event: pygame.event.Event, client: object) -> object:
        position = event.pos
        tab = self._tab_at_position(position)
        if tab is not None:
            self._active_tab = tab
            return _NoSelection
        if self._start_button_rect.collidepoint(position):
            return self._load_selected_option(client)
        if self._use_current_button_rect.collidepoint(position):
            return self._use_current_map(client)
        if self._refresh_button_rect.collidepoint(position):
            self._refresh_options(client, preserve_selection=True)
            return _NoSelection

        for index, rect in self._row_rects.items():
            if rect.collidepoint(position):
                already_selected = self._selected_index == index
                self._selected_index = index
                self._ensure_selection_visible()
                if already_selected and getattr(event, "clicks", 1) >= 2:
                    return self._load_selected_option(client)
                return _NoSelection
        return _NoSelection

    def _update_hover(self, position: tuple[int, int]) -> None:
        self._hovered_index = None
        for index, rect in self._row_rects.items():
            if rect.collidepoint(position):
                self._hovered_index = index
                return

    def _move_selection(self, delta: int) -> None:
        if not self._options:
            return
        self._selected_index = max(0, min(len(self._options) - 1, self._selected_index + delta))
        self._ensure_selection_visible()

    def _scroll_by(self, rows: int) -> None:
        if not self._options:
            return
        visible = self._visible_row_count()
        max_offset = max(0, len(self._options) - visible)
        self._scroll_offset = max(0, min(max_offset, self._scroll_offset + rows))

    def _ensure_selection_visible(self) -> None:
        visible = self._visible_row_count()
        if self._selected_index < self._scroll_offset:
            self._scroll_offset = self._selected_index
        elif self._selected_index >= self._scroll_offset + visible:
            self._scroll_offset = self._selected_index - visible + 1
        self._scroll_offset = max(0, self._scroll_offset)

    def _visible_row_count(self) -> int:
        _rect, columns, card_height, gap = self._card_layout()
        rows = max(1, (_rect.height + gap) // (card_height + gap))
        return max(1, rows * columns)

    def _pump_basic_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
            if event.type == pygame.VIDEORESIZE:
                self._resize(event.w, event.h)
        return True

    def _resize(self, width: int, height: int) -> None:
        width = max(760, int(width))
        height = max(520, int(height))
        self._surface = pygame.display.set_mode((width, height), display_flags_from_settings())

    def _ensure_display_ready(self) -> None:
        if not pygame.get_init():
            pygame.init()
            self._clock = pygame.time.Clock()
        if not pygame.font.get_init():
            pygame.font.init()
        if not pygame.display.get_init():
            pygame.display.init()
        if pygame.display.get_surface() is None:
            self._surface = create_display_surface(title=DISPLAY.title)
            self._init_fonts()

    def _init_fonts(self) -> None:
        self._title_font = pygame.font.SysFont("consolas", 34, bold=True)
        self._subtitle_font = pygame.font.SysFont("consolas", 20, bold=True)
        self._font = pygame.font.SysFont("consolas", 16)
        self._small_font = pygame.font.SysFont("consolas", 13)
        self._button_font = pygame.font.SysFont("consolas", 15, bold=True)

    def _draw(self, status_only: bool) -> None:
        width, height = self._surface.get_size()
        self._surface.fill(DISPLAY.clear_color)
        margin = max(28, min(54, width // 28))
        header_top = max(28, height // 24)

        self._draw_text(
            "KalmanLab CARLA Localization Dashboard",
            (margin, header_top),
            self._title_font,
            (239, 243, 250),
            max_width=width - 2 * margin,
        )
        self._draw_text(
            "Select the CARLA world before entering the live simulation dashboard.",
            (margin, header_top + 46),
            self._font,
            DASHBOARD.muted_text_color,
            max_width=width - 2 * margin,
        )
        status_rect = pygame.Rect(margin, header_top + 86, width - 2 * margin, 128)
        self._draw_status_panel(status_rect)

        if status_only:
            self._draw_text(
                "Startup is preparing CARLA. Press ESC to quit safely.",
                (margin, status_rect.bottom + 26),
                self._font,
                (185, 194, 210),
                max_width=width - 2 * margin,
            )
            pygame.display.flip()
            return

        tab_top = status_rect.bottom + 18
        self._draw_startup_tabs(pygame.Rect(margin, tab_top, width - 2 * margin, 34))
        if self._active_tab == "Test Setup":
            self._draw_test_setup(pygame.Rect(margin, tab_top + 48, width - 2 * margin, height - tab_top - 70))
        else:
            selector_header = pygame.Rect(margin, tab_top + 48, width - 2 * margin, 58)
            self._draw_selector_header(selector_header)
            self._map_list_top = selector_header.bottom + 12
            self._draw_map_list()
            self._draw_controls(pygame.Rect(margin, height - 78, width - 2 * margin, 52))
        pygame.display.flip()

    def _draw_startup_tabs(self, rect: pygame.Rect) -> None:
        self._tab_rects.clear()
        gap = 8
        tab_width = min(190, max(130, (rect.width - gap) // 4))
        x = rect.left
        for tab in ("Map Selection", "Test Setup"):
            tab_rect = pygame.Rect(x, rect.top, tab_width, 30)
            self._tab_rects[tab] = tab_rect
            active = tab == self._active_tab
            hovered = tab_rect.collidepoint(pygame.mouse.get_pos())
            background = (35, 73, 53) if active else ((34, 42, 54) if hovered else (24, 30, 39))
            border = DASHBOARD.success_color if active else (DASHBOARD.panel_border_color if not hovered else (116, 188, 255))
            pygame.draw.rect(self._surface, background, tab_rect, border_radius=5)
            pygame.draw.rect(self._surface, border, tab_rect, width=1, border_radius=5)
            rendered = self._button_font.render(tab, True, DASHBOARD.title_color if active else DASHBOARD.text_color)
            self._surface.blit(rendered, rendered.get_rect(center=tab_rect.center))
            x += tab_width + gap

    def _tab_at_position(self, position: tuple[int, int]) -> Optional[str]:
        for tab, rect in self._tab_rects.items():
            if rect.collidepoint(position):
                return tab
        return None

    def _refresh_test_setup(self) -> None:
        self._setup_filter_records = [
            record for record in discover_filters() if record.valid and record.safe_for_autonomous_control
        ]
        if self._setup_filter_records and not self._selected_filter_id:
            self._selected_filter_id = self._setup_filter_records[0].filter_id
        self._route_items = load_available_test_routes([option.load_name or option.detail for option in self._options])
        if self._sensor_editor is None:
            self._sensor_editor = ParameterEditor(
                specs=SENSOR_NOISE_SPECS,
                values=SENSOR_NOISE_PRESETS["Medium Noise"],
                presets=SENSOR_NOISE_PRESETS,
                active_preset="Medium Noise",
                title="Sensor Noise/Error Settings",
            )
        if self._behavior_editor is None:
            self._behavior_editor = ParameterEditor(
                specs=BEHAVIOR_SPECS,
                values=behavior_values_from_config(DrivingBehaviorConfig()),
                presets=BEHAVIOR_PRESETS,
                active_preset="Balanced",
                title="Vehicle Behavior Settings",
            )

    def _handle_test_setup_event(self, event: pygame.event.Event, client: object) -> object:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return None
            if event.key == pygame.K_TAB:
                self._active_tab = "Map Selection"
                return _NoSelection
        if hasattr(event, "pos"):
            tab = self._tab_at_position(event.pos)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and tab is not None:
                self._active_tab = tab
                return _NoSelection

        if self._sensor_editor is not None and self._sensor_editor.handle_event(event):
            self._sensor_preset = self._sensor_editor.active_preset
            return _NoSelection
        if self._behavior_editor is not None and self._behavior_editor.handle_event(event):
            self._behavior_preset = self._behavior_editor.active_preset
            return _NoSelection

        if event.type == pygame.MOUSEWHEEL:
            self._route_scroll = max(0, self._route_scroll - event.y * 2)
            return _NoSelection
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and hasattr(event, "pos"):
            position = event.pos
            for filter_id, rect in self._setup_filter_buttons.items():
                if rect.collidepoint(position):
                    self._selected_filter_id = filter_id
                    return _NoSelection
            if self._select_all_routes_rect.collidepoint(position):
                self._selected_route_indices = {item.index for item in self._route_items}
                return _NoSelection
            if self._clear_routes_rect.collidepoint(position):
                self._selected_route_indices.clear()
                return _NoSelection
            for index, rect in self._route_rects.items():
                if rect.collidepoint(position):
                    if index in self._selected_route_indices:
                        self._selected_route_indices.remove(index)
                    else:
                        self._selected_route_indices.add(index)
                    return _NoSelection
            if self._start_benchmark_rect.collidepoint(position):
                return self._start_benchmark_from_setup(client)
        return _NoSelection

    def _draw_test_setup(self, rect: pygame.Rect) -> None:
        self._refresh_test_setup()
        gap = 12
        column_width = max(260, (rect.width - 2 * gap) // 3)
        left = pygame.Rect(rect.left, rect.top, column_width, rect.height - 58)
        middle = pygame.Rect(left.right + gap, rect.top, column_width, rect.height - 58)
        right = pygame.Rect(middle.right + gap, rect.top, rect.right - middle.right - gap, rect.height - 58)
        bottom = pygame.Rect(rect.left, rect.bottom - 46, rect.width, 42)

        self._draw_filter_selection(left)
        if self._sensor_editor is not None:
            self._sensor_editor.draw(self._surface, middle)
        if self._behavior_editor is not None:
            behavior_rect = pygame.Rect(right.left, right.top, right.width, max(180, int(right.height * 0.56)))
            routes_rect = pygame.Rect(right.left, behavior_rect.bottom + gap, right.width, right.bottom - behavior_rect.bottom - gap)
            self._behavior_editor.draw(self._surface, behavior_rect)
            self._draw_route_selection(routes_rect)
        else:
            self._draw_route_selection(right)
        self._draw_test_setup_footer(bottom)

    def _draw_filter_selection(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self._surface, DASHBOARD.panel_inner_color, rect, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, rect, width=1, border_radius=6)
        content = rect.inflate(-2 * DASHBOARD.panel_padding_px, -2 * DASHBOARD.panel_padding_px)
        self._draw_text("Filter Selection", content.topleft, self._subtitle_font, DASHBOARD.title_color, content.width)
        y = content.top + 34
        self._setup_filter_buttons.clear()
        if not self._setup_filter_records:
            self._draw_text("No autonomous-safe filters found.", (content.left, y), self._font, DASHBOARD.warning_color, content.width)
            return
        for record in self._setup_filter_records:
            button = pygame.Rect(content.left, y, content.width, 34)
            self._setup_filter_buttons[record.filter_id] = button
            active = record.filter_id == self._selected_filter_id
            pygame.draw.rect(self._surface, (35, 73, 53) if active else (24, 30, 39), button, border_radius=5)
            pygame.draw.rect(self._surface, DASHBOARD.success_color if active else DASHBOARD.panel_border_color, button, width=1, border_radius=5)
            label = f"{record.display_name} ({record.filter_id})"
            self._draw_text(label, (button.left + 10, button.top + 8), self._button_font, DASHBOARD.title_color, button.width - 20)
            y += 42
        y += 8
        active = next((record for record in self._setup_filter_records if record.filter_id == self._selected_filter_id), None)
        if active is not None:
            lines = [
                f"Type: {active.filter_info.get('type')}",
                f"State: {active.filter_info.get('state_vector')}",
                f"Model: {active.filter_info.get('process_model')}",
                f"Measurement: {active.filter_info.get('measurement_model')}",
            ]
            for line in lines:
                if y + 16 > content.bottom:
                    break
                self._draw_text(line, (content.left, y), self._small_font, DASHBOARD.text_color, content.width)
                y += 18

    def _draw_route_selection(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self._surface, DASHBOARD.panel_inner_color, rect, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, rect, width=1, border_radius=6)
        content = rect.inflate(-2 * DASHBOARD.panel_padding_px, -2 * DASHBOARD.panel_padding_px)
        title = f"Saved Test Routes ({len(self._selected_route_indices)} selected)"
        self._draw_text(title, content.topleft, self._subtitle_font, DASHBOARD.title_color, content.width)
        self._select_all_routes_rect = pygame.Rect(content.left, content.top + 30, 92, 24)
        self._clear_routes_rect = pygame.Rect(self._select_all_routes_rect.right + 8, content.top + 30, 86, 24)
        self._draw_button(self._select_all_routes_rect, "Select All")
        self._draw_button(self._clear_routes_rect, "Clear")
        list_rect = pygame.Rect(content.left, content.top + 62, content.width, content.height - 62)
        pygame.draw.rect(self._surface, (14, 18, 24), list_rect, border_radius=4)
        self._route_rects.clear()
        if not self._route_items:
            self._draw_text("No saved routes in config/test_routes.json.", (list_rect.left + 8, list_rect.top + 10), self._small_font, DASHBOARD.warning_color, list_rect.width - 16)
            return
        row_h = 56
        visible = max(1, list_rect.height // row_h)
        self._route_scroll = min(self._route_scroll, max(0, len(self._route_items) - visible))
        for visible_index, item in enumerate(self._route_items[self._route_scroll : self._route_scroll + visible]):
            row = pygame.Rect(list_rect.left + 5, list_rect.top + 5 + visible_index * row_h, list_rect.width - 10, row_h - 6)
            self._route_rects[item.index] = row
            selected = item.index in self._selected_route_indices
            pygame.draw.rect(self._surface, (35, 73, 53) if selected else (24, 30, 39), row, border_radius=4)
            pygame.draw.rect(self._surface, DASHBOARD.success_color if selected else DASHBOARD.panel_border_color, row, width=1, border_radius=4)
            mark = "[x]" if selected else "[ ]"
            length = f"{item.straight_line_length_m:.0f}m" if item.straight_line_length_m is not None else "n/a"
            status = "available" if item.compatible_with_available_maps else "map not listed"
            self._draw_text(f"{mark} {item.route.name}", (row.left + 8, row.top + 6), self._font, DASHBOARD.title_color, row.width - 16)
            self._draw_text(f"{display_map_name(item.route.map_name)} | {length} | {status}", (row.left + 8, row.top + 29), self._small_font, DASHBOARD.muted_text_color, row.width - 16)

    def _draw_test_setup_footer(self, rect: pygame.Rect) -> None:
        self._start_benchmark_rect = pygame.Rect(rect.right - 214, rect.top + 3, 214, 34)
        self._draw_button(self._start_benchmark_rect, "Start Benchmark Test", primary=True)
        summary = self._setup_summary_text()
        self._draw_text(summary, (rect.left, rect.top + 10), self._small_font, DASHBOARD.muted_text_color, self._start_benchmark_rect.left - rect.left - 12)

    def _setup_summary_text(self) -> str:
        routes = [item.route for item in self._route_items if item.index in self._selected_route_indices]
        maps = sorted({display_map_name(route.map_name) for route in routes})
        filter_label = self._selected_filter_id or "none"
        map_text = ", ".join(maps[:3]) + ("..." if len(maps) > 3 else "")
        return f"Filter {filter_label} | Routes {len(routes)} | Maps {map_text or 'none'} | Sensor {self._sensor_preset} | Behavior {self._behavior_preset}"

    def _start_benchmark_from_setup(self, client: object) -> object:
        routes = [item.route for item in self._route_items if item.index in self._selected_route_indices]
        sensor_values = self._sensor_editor.values() if self._sensor_editor is not None else SENSOR_NOISE_PRESETS["Medium Noise"]
        behavior_values = self._behavior_editor.values() if self._behavior_editor is not None else BEHAVIOR_PRESETS["Balanced"]
        if self._sensor_editor is not None:
            self._sensor_preset = self._sensor_editor.active_preset
        if self._behavior_editor is not None:
            self._behavior_preset = self._behavior_editor.active_preset
        config = BenchmarkConfig(
            selected_filter=self._selected_filter_id,
            selected_routes=tuple(routes),
            sensor_noise_config=sensor_noise_config_from_values(sensor_values, preset_name=self._sensor_preset),
            vehicle_behavior_config=driving_behavior_from_values(behavior_values, preset_name=self._behavior_preset),
            sensor_noise_preset=self._sensor_preset,
            vehicle_behavior_preset=self._behavior_preset,
            metadata={"startup_mode": "test_setup"},
        )
        errors = validate_benchmark_config(
            config,
            valid_filter_ids=[record.filter_id for record in self._setup_filter_records],
            available_maps=[option.load_name or option.detail for option in self._options],
        )
        if errors:
            self._error = " ".join(errors[:3])
            return _NoSelection

        first_route = routes[0]
        return self._load_map_for_benchmark(client, first_route, config)

    def _load_map_for_benchmark(
        self,
        client: object,
        first_route: SavedTestRoute,
        config: BenchmarkConfig,
    ) -> object:
        current_map = self._read_current_map(client)
        if maps_compatible(current_map, first_route.map_name):
            self._write_runtime_state(selected_load_name=None, active_map_name=current_map)
            return StartupMapSelection(
                selected_map_load_name=None,
                active_map_name=current_map,
                used_current_map=True,
                benchmark_config=config,
            )
        if not first_route.map_name:
            self._error = "First selected route has no map metadata."
            return _NoSelection
        load_name = self._load_name_for_map(first_route.map_name)
        self._status = "Loading map"
        self._detail = f"Loading {display_map_name(load_name)} for benchmark..."
        self._error = ""
        self._draw(status_only=False)
        pygame.display.flip()
        pygame.event.pump()
        try:
            setter = getattr(client, "set_timeout", None)
            if setter is not None:
                setter(max(float(CARLA.timeout_seconds), 60.0))
            world = client.load_world(load_name)
            world_map = world.get_map()
            active_map_name = getattr(world_map, "name", None)
        except Exception as exc:
            self._status = "Connected"
            self._detail = "Select a map before the dashboard starts."
            self._error = f"Benchmark map load failed: {exc}"
            return _NoSelection
        self._write_runtime_state(selected_load_name=load_name, active_map_name=active_map_name)
        return StartupMapSelection(
            selected_map_load_name=load_name,
            active_map_name=active_map_name,
            used_current_map=False,
            benchmark_config=config,
        )

    def _load_name_for_map(self, map_name: str) -> str:
        for option in self._options:
            if option.load_name and maps_compatible(option.load_name, map_name):
                return option.load_name
        return display_map_name(map_name)

    def _draw_status_panel(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self._surface, DASHBOARD.panel_background_color, rect, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, rect, width=1, border_radius=6)

        column_gap = 28
        left_width = max(260, int(rect.width * 0.35))
        left = pygame.Rect(rect.left + 18, rect.top + 14, left_width, rect.height - 28)
        right = pygame.Rect(left.right + column_gap, rect.top + 14, rect.right - left.right - column_gap - 18, rect.height - 28)

        self._draw_text("CARLA server", left.topleft, self._subtitle_font, DASHBOARD.title_color, left.width)
        status_color = DASHBOARD.success_color if self._status.lower() in ("connected", "running") else (116, 188, 255)
        self._draw_text(f"Status: {self._status}", (left.left, left.top + 34), self._font, status_color, left.width)
        current_map = display_map_name(self._current_map_name) if self._current_map_name else "unread"
        self._draw_text(f"Current map: {current_map}", (left.left, left.top + 62), self._small_font, DASHBOARD.text_color, left.width)

        exe_text = self._executable_path or "not detected yet"
        self._draw_text("Executable", right.topleft, self._small_font, DASHBOARD.muted_text_color, right.width)
        self._draw_text(exe_text, (right.left, right.top + 20), self._small_font, DASHBOARD.text_color, right.width)
        message = self._error or self._detail
        color = DASHBOARD.warning_color if self._error else DASHBOARD.muted_text_color
        self._draw_text(message, (right.left, right.top + 52), self._small_font, color, right.width)

    def _draw_selector_header(self, rect: pygame.Rect) -> None:
        available = f"{self._available_count} available map(s)"
        normalized = normalize_map_name(self._current_map_name) or "unknown"
        self._draw_text("Startup map selection", (rect.left, rect.top), self._subtitle_font, DASHBOARD.title_color, rect.width)
        self._draw_text(
            f"{available} from CARLA RPC | current map id: {normalized}",
            (rect.left, rect.top + 30),
            self._font,
            DASHBOARD.muted_text_color,
            rect.width,
        )

    def _draw_map_list(self) -> None:
        rect, columns, card_height, gap = self._card_layout()
        pygame.draw.rect(self._surface, (14, 18, 24), rect, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, rect, width=1, border_radius=6)
        self._row_rects.clear()

        if not self._options:
            self._draw_text("No map options available.", (rect.left + 16, rect.top + 16), self._font, DASHBOARD.text_color, rect.width - 32)
            return

        visible = self._visible_row_count()
        end_index = min(len(self._options), self._scroll_offset + visible)
        card_width = max(120, (rect.width - 2 * DASHBOARD.panel_padding_px - (columns - 1) * gap) // columns)
        start_x = rect.left + DASHBOARD.panel_padding_px
        start_y = rect.top + DASHBOARD.panel_padding_px
        for index in range(self._scroll_offset, end_index):
            option = self._options[index]
            visible_index = index - self._scroll_offset
            row_index = visible_index // columns
            column_index = visible_index % columns
            card = pygame.Rect(
                start_x + column_index * (card_width + gap),
                start_y + row_index * (card_height + gap),
                card_width,
                card_height,
            )
            self._row_rects[index] = card
            selected = index == self._selected_index
            hovered = index == self._hovered_index
            card_color = (33, 48, 63) if selected else ((28, 34, 44) if hovered else (21, 26, 34))
            border_color = (116, 188, 255) if selected else ((122, 139, 164) if hovered else (58, 67, 82))
            pygame.draw.rect(self._surface, card_color, card, border_radius=6)
            pygame.draw.rect(self._surface, border_color, card, width=1, border_radius=6)
            if selected:
                accent = pygame.Rect(card.left, card.top + 10, 4, card.height - 20)
                pygame.draw.rect(self._surface, DASHBOARD.success_color, accent, border_radius=2)
            title_color = (248, 250, 253) if selected else (222, 228, 238)
            detail_color = DASHBOARD.muted_text_color if option.guaranteed else DASHBOARD.warning_color
            badge = "CURRENT" if option.is_current else ("AVAILABLE" if option.guaranteed else "FALLBACK")
            badge_color = DASHBOARD.success_color if option.is_current else ((116, 188, 255) if option.guaranteed else DASHBOARD.warning_color)
            self._draw_text(option.display_name, (card.left + 16, card.top + 13), self._font, title_color, card.width - 32)
            self._draw_text(option.detail, (card.left + 16, card.top + 38), self._small_font, detail_color, card.width - 32)
            self._draw_text(badge, (card.left + 16, card.bottom - 22), self._small_font, badge_color, card.width - 32)

        if len(self._options) > visible:
            scroll_text = f"{self._scroll_offset + 1}-{end_index} / {len(self._options)}"
            self._draw_text(scroll_text, (rect.right - 130, rect.bottom - 24), self._small_font, DASHBOARD.muted_text_color, 120)

    def _draw_controls(self, rect: pygame.Rect) -> None:
        button_gap = 10
        start_width = 180
        secondary_width = 136
        self._start_button_rect = pygame.Rect(rect.right - start_width, rect.top, start_width, 36)
        self._refresh_button_rect = pygame.Rect(
            self._start_button_rect.left - button_gap - secondary_width,
            rect.top,
            secondary_width,
            36,
        )
        self._use_current_button_rect = pygame.Rect(
            self._refresh_button_rect.left - button_gap - secondary_width,
            rect.top,
            secondary_width,
            36,
        )

        controls = "Arrow keys select | Mouse wheel scrolls | Enter starts | Double-click starts | ESC quits"
        available_width = max(80, self._use_current_button_rect.left - rect.left - 12)
        self._draw_text(controls, (rect.left, rect.top + 9), self._small_font, DASHBOARD.muted_text_color, available_width)
        self._draw_button(self._use_current_button_rect, "Use Current", muted=False)
        self._draw_button(self._refresh_button_rect, "Refresh", muted=False)
        self._draw_button(self._start_button_rect, "Start Dashboard", muted=not bool(self._options), primary=True)

    def _list_geometry(self) -> tuple[int, int, int, int]:
        width, height = self._surface.get_size()
        margin = max(28, min(54, width // 28))
        top = max(330, height // 3, self._map_list_top)
        bottom_margin = 94
        return margin, top, width - 2 * margin, max(120, height - top - bottom_margin)

    def _card_layout(self) -> tuple[pygame.Rect, int, int, int]:
        left, top, width, height = self._list_geometry()
        columns = 3 if width >= 1240 else (2 if width >= 820 else 1)
        gap = 12
        card_height = 86
        return pygame.Rect(left, top, width, height), columns, card_height, gap

    def _draw_button(self, rect: pygame.Rect, label: str, muted: bool = False, primary: bool = False) -> None:
        mouse_pos = pygame.mouse.get_pos()
        hovered = rect.collidepoint(mouse_pos)
        if muted:
            background = (38, 42, 50)
            border = (56, 62, 72)
            text_color = DASHBOARD.muted_text_color
        elif primary:
            background = (32, 88, 63) if not hovered else (39, 105, 75)
            border = DASHBOARD.success_color
            text_color = DASHBOARD.title_color
        else:
            background = (24, 30, 39) if not hovered else (38, 47, 61)
            border = DASHBOARD.panel_border_color if not hovered else (122, 139, 164)
            text_color = DASHBOARD.text_color
        pygame.draw.rect(self._surface, background, rect, border_radius=5)
        pygame.draw.rect(self._surface, border, rect, width=1, border_radius=5)
        rendered = self._button_font.render(label, True, text_color)
        self._surface.blit(rendered, rendered.get_rect(center=rect.center))

    def _draw_text(
        self,
        text: object,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
        max_width: int,
    ) -> None:
        fitted = self._fit_text(str(text), font, max_width)
        rendered = font.render(fitted, True, color)
        self._surface.blit(rendered, position)

    @staticmethod
    def _fit_text(text: str, font: pygame.font.Font, max_width: int) -> str:
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

    def _read_runtime_state(self) -> dict[str, object]:
        try:
            return json.loads(self._runtime_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_runtime_state(
        self,
        selected_load_name: Optional[str],
        active_map_name: Optional[str],
    ) -> None:
        data = {
            "last_map_load_name": selected_load_name,
            "last_active_map_name": active_map_name,
            "last_active_map_id": normalize_map_name(active_map_name),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            self._runtime_state_path.parent.mkdir(parents=True, exist_ok=True)
            self._runtime_state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass


class _NoSelectionType:
    pass


_NoSelection = _NoSelectionType()
