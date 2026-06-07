"""Pre-dashboard pygame UI for CARLA startup status and map selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Optional

import pygame

from config.settings import CARLA, DASHBOARD, DISPLAY
from src.KalmanLab.filter_base import TRACKING_MODE_ACTIVE, TRACKING_MODE_PASSIVE
from src.KalmanLab.registry import discover_filters
from src.KalmanLab.tune_advisor import TuneRecommendation, recommend_filter_tune
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
from src.evaluation.evaluation_artifacts import (
    RecordedLogInfo,
    list_recorded_logs,
    read_json,
)
from src.evaluation.filter_auto_tuner import (
    AutoTuneRequest,
    AutoTuneResult,
    FilterAutoTuner,
    list_saved_tune_configs,
    load_saved_tune_config,
    noise_profile_summary,
)
from src.evaluation.offline_replay_runner import OfflineReplayRequest, OfflineReplayRunner
from src.evaluation.sensor_log_recorder import OfflineRecordingConfig
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

TOP_LEVEL_TABS = (
    "Demo",
    "Closed Loop Benchmark",
    "Offline Localization Benchmark",
)
CLOSED_LOOP_SUBTABS = (
    "Filters",
    "Sensor Noise",
    "Vehicle Behavior",
    "Routes",
)
OFFLINE_SUBTABS = (
    "Record Sensor Data",
    "Test Setup",
)
OFFLINE_TEST_SETUP_SUBTABS = (
    "Select Route",
    "Filters",
)


@dataclass(frozen=True)
class StartupMapSelection:
    """Result returned after a startup map is selected and loaded."""

    selected_map_load_name: Optional[str]
    active_map_name: Optional[str]
    used_current_map: bool
    benchmark_config: Optional[BenchmarkConfig] = None
    offline_recording_config: Optional[OfflineRecordingConfig] = None


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
        self._active_tab = "Demo"
        self._tab_rects: dict[str, pygame.Rect] = {}
        self._active_closed_loop_subtab = "Filters"
        self._closed_loop_subtab_rects: dict[str, pygame.Rect] = {}
        self._active_offline_subtab = "Record Sensor Data"
        self._offline_subtab_rects: dict[str, pygame.Rect] = {}
        self._active_offline_setup_subtab = "Select Route"
        self._offline_setup_subtab_rects: dict[str, pygame.Rect] = {}
        self._setup_filter_records = []
        self._setup_filter_buttons: dict[str, pygame.Rect] = {}
        self._offline_filter_tab_rects: dict[str, pygame.Rect] = {}
        self._offline_filter_include_rects: dict[str, pygame.Rect] = {}
        self._selected_filter_id = ""
        self._active_offline_filter_id = ""
        self._offline_filter_ids: set[str] = set()
        self._selected_filter_tunes: dict[str, dict[str, object]] = {}
        self._filter_tune_editor: Optional[ParameterEditor] = None
        self._filter_tune_editor_filter_id = ""
        self._filter_tune_panel_rect = pygame.Rect(0, 0, 1, 1)
        self._tracking_mode = TRACKING_MODE_PASSIVE
        self._tracking_button_rects: dict[str, pygame.Rect] = {}
        self._apply_recommended_rect = pygame.Rect(0, 0, 1, 1)
        self._recommendation_applied_by_filter: dict[str, bool] = {}
        self._sensor_editor: Optional[ParameterEditor] = None
        self._behavior_editor: Optional[ParameterEditor] = None
        self._sensor_preset = "Medium Noise"
        self._behavior_preset = "Balanced"
        self._route_items = []
        self._selected_route_indices: set[int] = set()
        self._recording_route_index: Optional[int] = None
        self._closed_loop_route_scroll = 0
        self._recording_route_scroll = 0
        self._route_rects: dict[int, pygame.Rect] = {}
        self._select_all_routes_rect = pygame.Rect(0, 0, 1, 1)
        self._clear_routes_rect = pygame.Rect(0, 0, 1, 1)
        self._start_benchmark_rect = pygame.Rect(0, 0, 1, 1)
        self._record_sensor_logs_rect = pygame.Rect(0, 0, 1, 1)
        self._run_offline_replay_rect = pygame.Rect(0, 0, 1, 1)
        self._refresh_recorded_logs_rect = pygame.Rect(0, 0, 1, 1)
        self._recorded_logs: list[RecordedLogInfo] = []
        self._selected_recorded_log_index: Optional[int] = None
        self._recorded_log_scroll = 0
        self._recorded_log_rects: dict[int, pygame.Rect] = {}
        self._offline_status_lines: list[str] = ["Recorded logs are loaded from benchmark_results/offline_localization/recordings."]
        self._setup_summary_lines: list[str] = []
        self._offline_auto_tune_rect = pygame.Rect(0, 0, 1, 1)
        self._offline_saved_tunes_rect = pygame.Rect(0, 0, 1, 1)
        self._offline_back_to_sliders_rect = pygame.Rect(0, 0, 1, 1)
        self._offline_saved_tune_apply_rects: dict[int, pygame.Rect] = {}
        self._offline_filter_saved_tune_mode = False
        self._offline_saved_tune_status = ""
        self._auto_tune_modal_open = False
        self._auto_tune_filter_id = ""
        self._auto_tune_selected_log_indices: set[int] = set()
        self._auto_tune_log_scroll = 0
        self._auto_tune_log_rects: dict[int, pygame.Rect] = {}
        self._auto_tune_trials = 0
        self._auto_tune_running = False
        self._auto_tune_stop_requested = False
        self._auto_tune_status_lines: list[str] = []
        self._auto_tune_result: Optional[AutoTuneResult] = None
        self._auto_tune_start_rect = pygame.Rect(0, 0, 1, 1)
        self._auto_tune_cancel_rect = pygame.Rect(0, 0, 1, 1)
        self._auto_tune_save_rect = pygame.Rect(0, 0, 1, 1)
        self._auto_tune_apply_rect = pygame.Rect(0, 0, 1, 1)
        self._auto_tune_close_rect = pygame.Rect(0, 0, 1, 1)
        self._auto_tune_refresh_logs_rect = pygame.Rect(0, 0, 1, 1)
        self._auto_tune_select_all_noise_rect = pygame.Rect(0, 0, 1, 1)
        self._auto_tune_clear_logs_rect = pygame.Rect(0, 0, 1, 1)

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
                if self._active_tab != "Demo":
                    result = self._handle_benchmark_mode_event(event, client)
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
            self._cycle_top_level_tab()
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

    def _cycle_top_level_tab(self) -> None:
        try:
            index = TOP_LEVEL_TABS.index(self._active_tab)
        except ValueError:
            self._active_tab = TOP_LEVEL_TABS[0]
            return
        self._active_tab = TOP_LEVEL_TABS[(index + 1) % len(TOP_LEVEL_TABS)]

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
        content_rect = pygame.Rect(margin, tab_top + 48, width - 2 * margin, height - tab_top - 70)
        if self._active_tab == "Demo":
            selector_header = pygame.Rect(margin, tab_top + 48, width - 2 * margin, 58)
            self._draw_selector_header(selector_header)
            self._map_list_top = selector_header.bottom + 12
            self._draw_map_list()
            self._draw_controls(pygame.Rect(margin, height - 78, width - 2 * margin, 52))
        elif self._active_tab == "Closed Loop Benchmark":
            self._draw_closed_loop_benchmark(content_rect)
        else:
            self._draw_offline_localization_benchmark(content_rect)
        if self._auto_tune_modal_open:
            self._draw_auto_tune_modal()
        pygame.display.flip()

    def _draw_startup_tabs(self, rect: pygame.Rect) -> None:
        self._tab_rects.clear()
        gap = 8
        tab_width = max(148, (rect.width - gap * (len(TOP_LEVEL_TABS) - 1)) // len(TOP_LEVEL_TABS))
        x = rect.left
        for tab in TOP_LEVEL_TABS:
            tab_rect = pygame.Rect(x, rect.top, min(tab_width, rect.right - x), 30)
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
        self._setup_filter_records = [record for record in discover_filters() if record.valid]
        benchmark_records = [
            record
            for record in self._setup_filter_records
            if record.benchmark_selectable and record.filter_id != "raw_gnss"
        ]
        replay_records = [
            record
            for record in self._setup_filter_records
            if record.benchmark_selectable and record.filter_id != "raw_gnss"
        ]
        if benchmark_records and not any(record.filter_id == self._selected_filter_id for record in benchmark_records):
            self._selected_filter_id = benchmark_records[0].filter_id
        elif not benchmark_records and self._setup_filter_records and not self._selected_filter_id:
            self._selected_filter_id = self._setup_filter_records[0].filter_id
        if replay_records and not any(record.filter_id == self._active_offline_filter_id for record in replay_records):
            self._active_offline_filter_id = replay_records[0].filter_id
        if not self._offline_filter_ids:
            self._offline_filter_ids = {
                record.filter_id
                for record in replay_records
                if record.benchmark_selectable and record.filter_id != "raw_gnss"
            }
            if not self._offline_filter_ids and self._active_offline_filter_id:
                self._offline_filter_ids.add(self._active_offline_filter_id)
        self._offline_filter_ids = {
            filter_id
            for filter_id in self._offline_filter_ids
            if any(record.filter_id == filter_id and record.benchmark_selectable for record in replay_records)
        }
        visible_filter_id = self._visible_tune_filter_id()
        if visible_filter_id:
            self._ensure_filter_tune_editor(visible_filter_id)
        self._route_items = load_available_test_routes([option.load_name or option.detail for option in self._options])
        if not self._recorded_logs:
            self._refresh_recorded_logs()
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

    def _refresh_recorded_logs(self) -> None:
        self._recorded_logs = list_recorded_logs()
        if self._selected_recorded_log_index is not None and self._selected_recorded_log_index >= len(self._recorded_logs):
            self._selected_recorded_log_index = None
        self._recorded_log_scroll = min(self._recorded_log_scroll, max(0, len(self._recorded_logs) - 1))

    def _selected_filter_record(self) -> object | None:
        return next(
            (record for record in self._setup_filter_records if record.filter_id == self._selected_filter_id),
            None,
        )

    def _filter_record(self, filter_id: str) -> object | None:
        return next((record for record in self._setup_filter_records if record.filter_id == filter_id), None)

    def _visible_tune_filter_id(self) -> str:
        if self._active_tab == "Closed Loop Benchmark" and self._active_closed_loop_subtab == "Filters":
            return self._selected_filter_id
        if (
            self._active_tab == "Offline Localization Benchmark"
            and self._active_offline_subtab == "Test Setup"
            and self._active_offline_setup_subtab == "Filters"
        ):
            return self._active_offline_filter_id
        return ""

    def _commit_filter_tune_editor(self) -> None:
        if self._filter_tune_editor is None or not self._filter_tune_editor_filter_id:
            return
        self._commit_filter_tune_values(self._filter_tune_editor_filter_id, self._filter_tune_editor.values())

    def _ensure_filter_tune_editor(self, filter_id: str) -> None:
        if self._filter_tune_editor is not None and self._filter_tune_editor_filter_id != filter_id:
            self._commit_filter_tune_editor()
        record = self._filter_record(filter_id)
        if record is None:
            self._filter_tune_editor = None
            self._filter_tune_editor_filter_id = ""
            return
        if record.filter_id not in self._selected_filter_tunes:
            self._selected_filter_tunes[record.filter_id] = dict(record.tune)
        specs = tuple(getattr(record, "tune_specs", ()))
        if not specs:
            self._filter_tune_editor = None
            self._filter_tune_editor_filter_id = record.filter_id
            return
        values = self._selected_filter_tunes.get(record.filter_id, dict(record.tune))
        if self._filter_tune_editor is None or self._filter_tune_editor_filter_id != record.filter_id:
            self._filter_tune_editor = ParameterEditor(
                specs=specs,
                values={key: float(value) for key, value in values.items() if isinstance(value, (int, float, bool))},
                presets={},
                active_preset="Custom",
                title="Tune Parameters",
                on_commit=lambda editor_values, _preset_name, filter_id=record.filter_id: self._commit_filter_tune_values(filter_id, editor_values),
            )
            self._filter_tune_editor_filter_id = record.filter_id
        else:
            self._filter_tune_editor.set_values(values, active_preset="Custom", commit=False)

    def _commit_filter_tune_values(self, filter_id: str, values: dict[str, float]) -> None:
        record = self._filter_record(filter_id)
        if record is None:
            return
        merged = dict(self._selected_filter_tunes.get(record.filter_id, record.tune))
        for key, value in values.items():
            default = record.tune.get(key)
            merged[key] = bool(value >= 0.5) if isinstance(default, bool) else float(value)
        self._selected_filter_tunes[record.filter_id] = merged
        self._recommendation_applied_by_filter[record.filter_id] = False

    def _current_filter_tune_values(self, filter_id: Optional[str] = None) -> dict[str, object]:
        selected_filter_id = filter_id or self._selected_filter_id
        record = self._filter_record(selected_filter_id)
        if record is None:
            return {}
        if self._filter_tune_editor is not None and self._filter_tune_editor_filter_id == record.filter_id:
            self._commit_filter_tune_editor()
        return dict(self._selected_filter_tunes.get(record.filter_id, record.tune))

    def _included_offline_filter_tunes(self, selected_filters: tuple[str, ...]) -> dict[str, dict[str, object]]:
        self._commit_filter_tune_editor()
        return {
            filter_id: self._current_filter_tune_values(filter_id)
            for filter_id in selected_filters
            if filter_id != "raw_gnss"
        }

    def _current_sensor_values(self) -> dict[str, object]:
        if self._sensor_editor is not None:
            return self._sensor_editor.values()
        return SENSOR_NOISE_PRESETS["Medium Noise"]

    def _current_recommendation(self, filter_id: Optional[str] = None, force_passive: bool = False) -> TuneRecommendation:
        record = self._filter_record(filter_id or self._selected_filter_id)
        if record is None:
            return TuneRecommendation("", self._tracking_mode, {}, ("Select a filter to see recommendations.",))
        tracking_mode = TRACKING_MODE_PASSIVE if force_passive else self._tracking_mode
        return recommend_filter_tune(
            filter_id=record.filter_id,
            sensor_noise_config=sensor_noise_config_from_values(self._current_sensor_values(), preset_name=self._sensor_preset),
            tracking_mode=tracking_mode,
            current_tune=dict(self._selected_filter_tunes.get(record.filter_id, record.tune)),
            tune_specs=getattr(record, "tune_specs", ()),
        )

    def _apply_recommended_setup_tune(self, filter_id: Optional[str] = None, force_passive: bool = False) -> None:
        record = self._filter_record(filter_id or self._selected_filter_id)
        if record is None:
            return
        recommendation = self._current_recommendation(record.filter_id, force_passive=force_passive)
        if not recommendation.values:
            return
        merged = dict(self._selected_filter_tunes.get(record.filter_id, record.tune))
        merged.update(recommendation.values)
        self._selected_filter_tunes[record.filter_id] = merged
        self._recommendation_applied_by_filter[record.filter_id] = True
        self._ensure_filter_tune_editor(record.filter_id)

    def _handle_benchmark_mode_event(self, event: pygame.event.Event, client: object) -> object:
        if self._auto_tune_modal_open:
            return self._handle_auto_tune_modal_event(event)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return None
            if event.key == pygame.K_TAB:
                self._commit_filter_tune_editor()
                self._cycle_top_level_tab()
                return _NoSelection
        if hasattr(event, "pos"):
            tab = self._tab_at_position(event.pos)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and tab is not None:
                self._commit_filter_tune_editor()
                self._active_tab = tab
                return _NoSelection

        if self._active_tab == "Closed Loop Benchmark":
            return self._handle_closed_loop_event(event, client)
        if self._active_tab == "Offline Localization Benchmark":
            return self._handle_offline_localization_event(event, client)
        return _NoSelection

    def _handle_closed_loop_event(self, event: pygame.event.Event, client: object) -> object:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and hasattr(event, "pos"):
            position = event.pos
            for subtab, rect in self._closed_loop_subtab_rects.items():
                if rect.collidepoint(position):
                    self._commit_filter_tune_editor()
                    self._active_closed_loop_subtab = subtab
                    return _NoSelection
            for filter_id, rect in self._setup_filter_buttons.items():
                if rect.collidepoint(position):
                    record = self._filter_record(filter_id)
                    if record is None or not record.benchmark_selectable or filter_id == "raw_gnss":
                        self._error = f"Filter is not benchmark-selectable: {filter_id}."
                        return _NoSelection
                    self._commit_filter_tune_editor()
                    self._selected_filter_id = filter_id
                    self._ensure_filter_tune_editor(filter_id)
                    self._error = ""
                    return _NoSelection
            for mode, rect in self._tracking_button_rects.items():
                if rect.collidepoint(position):
                    self._tracking_mode = mode
                    return _NoSelection
            if self._active_closed_loop_subtab == "Filters" and self._apply_recommended_rect.collidepoint(position):
                self._commit_filter_tune_editor()
                self._apply_recommended_setup_tune(self._selected_filter_id)
                return _NoSelection

        if (
            self._active_closed_loop_subtab == "Filters"
            and self._filter_tune_editor is not None
            and self._filter_tune_editor.handle_event(event)
        ):
            return _NoSelection
        if (
            self._active_closed_loop_subtab == "Sensor Noise"
            and self._sensor_editor is not None
            and self._sensor_editor.handle_event(event)
        ):
            self._sensor_preset = self._sensor_editor.active_preset
            return _NoSelection
        if (
            self._active_closed_loop_subtab == "Vehicle Behavior"
            and self._behavior_editor is not None
            and self._behavior_editor.handle_event(event)
        ):
            self._behavior_preset = self._behavior_editor.active_preset
            return _NoSelection

        if event.type == pygame.MOUSEWHEEL:
            if self._active_closed_loop_subtab == "Routes":
                self._closed_loop_route_scroll = max(0, self._closed_loop_route_scroll - event.y * 2)
            return _NoSelection
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and hasattr(event, "pos"):
            position = event.pos
            if self._active_closed_loop_subtab == "Routes" and self._select_all_routes_rect.collidepoint(position):
                self._selected_route_indices = {item.index for item in self._route_items}
                return _NoSelection
            if self._active_closed_loop_subtab == "Routes" and self._clear_routes_rect.collidepoint(position):
                self._selected_route_indices.clear()
                return _NoSelection
            if self._active_closed_loop_subtab == "Routes":
                for index, rect in self._route_rects.items():
                    if not rect.collidepoint(position):
                        continue
                    if index in self._selected_route_indices:
                        self._selected_route_indices.remove(index)
                    else:
                        self._selected_route_indices.add(index)
                    return _NoSelection
            if self._start_benchmark_rect.collidepoint(position):
                return self._start_benchmark_from_setup(client)
        return _NoSelection

    def _handle_offline_localization_event(self, event: pygame.event.Event, client: object) -> object:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and hasattr(event, "pos"):
            position = event.pos
            for subtab, rect in self._offline_subtab_rects.items():
                if rect.collidepoint(position):
                    self._commit_filter_tune_editor()
                    self._active_offline_subtab = subtab
                    return _NoSelection
            if self._active_offline_subtab == "Test Setup":
                for subtab, rect in self._offline_setup_subtab_rects.items():
                    if rect.collidepoint(position):
                        self._commit_filter_tune_editor()
                        self._active_offline_setup_subtab = subtab
                        return _NoSelection
            if self._active_offline_subtab == "Test Setup" and self._active_offline_setup_subtab == "Select Route":
                if self._refresh_recorded_logs_rect.collidepoint(position):
                    self._refresh_recorded_logs()
                    self._offline_status_lines = [f"Found {len(self._recorded_logs)} recorded route log(s)."]
                    return _NoSelection
            if self._active_offline_subtab == "Test Setup" and self._active_offline_setup_subtab == "Filters":
                if self._offline_filter_saved_tune_mode:
                    if self._offline_back_to_sliders_rect.collidepoint(position):
                        self._offline_filter_saved_tune_mode = False
                        return _NoSelection
                    for index, rect in self._offline_saved_tune_apply_rects.items():
                        if rect.collidepoint(position):
                            self._apply_saved_tune_config(index)
                            return _NoSelection
                for filter_id, rect in self._offline_filter_include_rects.items():
                    if rect.collidepoint(position):
                        record = self._filter_record(filter_id)
                        if record is not None and record.benchmark_selectable:
                            if filter_id in self._offline_filter_ids:
                                self._offline_filter_ids.remove(filter_id)
                            else:
                                self._offline_filter_ids.add(filter_id)
                        return _NoSelection
                for filter_id, rect in self._offline_filter_tab_rects.items():
                    if rect.collidepoint(position):
                        self._commit_filter_tune_editor()
                        self._active_offline_filter_id = filter_id
                        self._ensure_filter_tune_editor(filter_id)
                        self._offline_filter_saved_tune_mode = False
                        return _NoSelection
                if self._offline_auto_tune_rect.collidepoint(position):
                    self._commit_filter_tune_editor()
                    self._open_auto_tune_modal(self._active_offline_filter_id)
                    return _NoSelection
                if self._offline_saved_tunes_rect.collidepoint(position):
                    self._commit_filter_tune_editor()
                    self._offline_filter_saved_tune_mode = True
                    return _NoSelection

        if (
            self._active_offline_subtab == "Record Sensor Data"
            and self._sensor_editor is not None
            and self._sensor_editor.handle_event(event)
        ):
            self._sensor_preset = self._sensor_editor.active_preset
            return _NoSelection
        if (
            self._active_offline_subtab == "Test Setup"
            and self._active_offline_setup_subtab == "Filters"
            and not self._offline_filter_saved_tune_mode
            and self._filter_tune_editor is not None
            and self._filter_tune_editor.handle_event(event)
        ):
            return _NoSelection

        if event.type == pygame.MOUSEWHEEL:
            if self._active_offline_subtab == "Record Sensor Data":
                self._recording_route_scroll = max(0, self._recording_route_scroll - event.y * 2)
            elif self._active_offline_subtab == "Test Setup" and self._active_offline_setup_subtab == "Select Route":
                self._recorded_log_scroll = max(0, self._recorded_log_scroll - event.y * 2)
            return _NoSelection

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and hasattr(event, "pos"):
            position = event.pos
            if self._active_offline_subtab == "Record Sensor Data":
                for index, rect in self._route_rects.items():
                    if rect.collidepoint(position):
                        self._recording_route_index = index
                        return _NoSelection
                if self._record_sensor_logs_rect.collidepoint(position):
                    return self._start_offline_recording_from_setup(client)
            elif self._active_offline_subtab == "Test Setup" and self._active_offline_setup_subtab == "Select Route":
                for index, rect in self._recorded_log_rects.items():
                    if rect.collidepoint(position):
                        self._selected_recorded_log_index = index
                        return _NoSelection
            elif self._active_offline_subtab == "Test Setup" and self._active_offline_setup_subtab == "Filters":
                if self._run_offline_replay_rect.collidepoint(position):
                    if self._auto_tune_running:
                        self._error = "Offline replay is disabled while auto tune is running."
                        return _NoSelection
                    self._run_offline_replay_from_setup()
                    return _NoSelection
        return _NoSelection

    def _draw_closed_loop_benchmark(self, rect: pygame.Rect) -> None:
        self._refresh_test_setup()
        gap = 12
        bottom = pygame.Rect(rect.left, rect.bottom - 46, rect.width, 42)
        subtab_bar = pygame.Rect(rect.left, rect.top, rect.width, 34)
        self._draw_subtabs(subtab_bar, CLOSED_LOOP_SUBTABS, self._active_closed_loop_subtab, self._closed_loop_subtab_rects)
        work = pygame.Rect(rect.left, subtab_bar.bottom + gap, rect.width, rect.height - 58 - subtab_bar.height - gap)
        self._setup_filter_buttons.clear()
        self._tracking_button_rects.clear()
        self._route_rects.clear()
        self._recorded_log_rects.clear()
        self._offline_filter_tab_rects.clear()
        self._offline_filter_include_rects.clear()
        self._refresh_recorded_logs_rect = pygame.Rect(0, 0, 1, 1)
        if self._active_closed_loop_subtab == "Filters":
            if work.width >= 980:
                left = pygame.Rect(work.left, work.top, max(320, int(work.width * 0.42)), work.height)
                right = pygame.Rect(left.right + gap, work.top, work.right - left.right - gap, work.height)
                self._draw_closed_loop_filter_selection(left)
                self._draw_filter_tune_panel(right, self._selected_filter_id)
            else:
                top = pygame.Rect(work.left, work.top, work.width, max(220, int(work.height * 0.48)))
                lower = pygame.Rect(work.left, top.bottom + gap, work.width, work.bottom - top.bottom - gap)
                self._draw_closed_loop_filter_selection(top)
                self._draw_filter_tune_panel(lower, self._selected_filter_id)
        elif self._active_closed_loop_subtab == "Sensor Noise":
            if self._sensor_editor is not None:
                self._sensor_editor.draw(self._surface, work)
        elif self._active_closed_loop_subtab == "Vehicle Behavior":
            if self._behavior_editor is not None:
                self._behavior_editor.draw(self._surface, work)
        elif self._active_closed_loop_subtab == "Routes":
            self._draw_route_selection(
                work,
                selected_indices=set(self._selected_route_indices),
                multi_select=True,
                title="Saved Test Routes",
            )
        self._draw_closed_loop_footer(bottom)

    def _draw_offline_localization_benchmark(self, rect: pygame.Rect) -> None:
        self._refresh_test_setup()
        gap = 12
        workflow = pygame.Rect(rect.left, rect.top, rect.width, 32)
        self._draw_workflow_strip(workflow)
        subtab_bar = pygame.Rect(rect.left, workflow.bottom + 8, rect.width, 34)
        self._draw_subtabs(subtab_bar, OFFLINE_SUBTABS, self._active_offline_subtab, self._offline_subtab_rects)
        bottom = pygame.Rect(rect.left, rect.bottom - 46, rect.width, 42)
        work = pygame.Rect(rect.left, subtab_bar.bottom + gap, rect.width, rect.height - workflow.height - subtab_bar.height - 8 - gap - 58)
        self._setup_filter_buttons.clear()
        self._tracking_button_rects.clear()
        self._route_rects.clear()
        self._recorded_log_rects.clear()
        self._offline_filter_tab_rects.clear()
        self._offline_filter_include_rects.clear()
        self._refresh_recorded_logs_rect = pygame.Rect(0, 0, 1, 1)

        show_record_button = False
        show_run_button = False
        if self._active_offline_subtab == "Record Sensor Data":
            self._draw_record_sensor_data_panel(work)
            show_record_button = True
        else:
            nested_bar = pygame.Rect(work.left, work.top, work.width, 30)
            self._draw_subtabs(
                nested_bar,
                OFFLINE_TEST_SETUP_SUBTABS,
                self._active_offline_setup_subtab,
                self._offline_setup_subtab_rects,
                small=True,
            )
            nested_work = pygame.Rect(work.left, nested_bar.bottom + gap, work.width, work.bottom - nested_bar.bottom - gap)
            if self._active_offline_setup_subtab == "Select Route":
                self._draw_recorded_log_list(nested_work)
            else:
                self._draw_offline_filter_setup(nested_work)
                show_run_button = True
        self._draw_offline_footer(bottom, show_record_button=show_record_button, show_run_button=show_run_button)

    def _draw_subtabs(
        self,
        rect: pygame.Rect,
        tabs: tuple[str, ...],
        active_tab: str,
        target: dict[str, pygame.Rect],
        small: bool = False,
    ) -> None:
        target.clear()
        gap = 6
        x = rect.left
        available = rect.width - gap * (len(tabs) - 1)
        tab_width = max(92 if small else 118, available // max(1, len(tabs)))
        for tab in tabs:
            button = pygame.Rect(x, rect.top, min(tab_width, rect.right - x), rect.height)
            target[tab] = button
            active = tab == active_tab
            hovered = button.collidepoint(pygame.mouse.get_pos())
            background = (35, 73, 53) if active else ((34, 42, 54) if hovered else (24, 30, 39))
            border = DASHBOARD.success_color if active else DASHBOARD.panel_border_color
            pygame.draw.rect(self._surface, background, button, border_radius=5)
            pygame.draw.rect(self._surface, border, button, width=1, border_radius=5)
            rendered = self._small_font.render(tab, True, DASHBOARD.title_color if active else DASHBOARD.text_color)
            self._surface.blit(rendered, rendered.get_rect(center=button.center))
            x += tab_width + gap

    def _draw_workflow_strip(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self._surface, (16, 21, 28), rect, border_radius=5)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, rect, width=1, border_radius=5)
        text = "1. Record sensor data -> 2. Select recorded route -> 3. Tune filters -> 4. Run comparison"
        self._draw_text(text, (rect.left + 12, rect.top + 8), self._small_font, DASHBOARD.muted_text_color, rect.width - 24)

    def _draw_recorded_log_list(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self._surface, DASHBOARD.panel_inner_color, rect, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, rect, width=1, border_radius=6)
        content = rect.inflate(-2 * DASHBOARD.panel_padding_px, -2 * DASHBOARD.panel_padding_px)
        selected_count = 1 if self._selected_recorded_log_index is not None else 0
        title = f"Recorded Logs ({selected_count} selected)"
        self._draw_text(title, content.topleft, self._subtitle_font, DASHBOARD.title_color, content.width)
        self._refresh_recorded_logs_rect = pygame.Rect(content.right - 136, content.top, 136, 26)
        self._draw_button(self._refresh_recorded_logs_rect, "Refresh Logs")
        list_rect = pygame.Rect(content.left, content.top + 36, content.width, max(80, int(content.height * 0.68)))
        pygame.draw.rect(self._surface, (14, 18, 24), list_rect, border_radius=4)
        self._recorded_log_rects.clear()
        if not self._recorded_logs:
            self._draw_text(
                "No recorded logs found. Record a sensor log first from Record Sensor Data.",
                (list_rect.left + 8, list_rect.top + 10),
                self._small_font,
                DASHBOARD.warning_color,
                list_rect.width - 16,
            )
        else:
            row_h = 58
            visible = max(1, list_rect.height // row_h)
            self._recorded_log_scroll = min(self._recorded_log_scroll, max(0, len(self._recorded_logs) - visible))
            for visible_index, info in enumerate(self._recorded_logs[self._recorded_log_scroll : self._recorded_log_scroll + visible]):
                index = self._recorded_log_scroll + visible_index
                row = pygame.Rect(list_rect.left + 5, list_rect.top + 5 + visible_index * row_h, list_rect.width - 10, row_h - 6)
                self._recorded_log_rects[index] = row
                selected = index == self._selected_recorded_log_index
                pygame.draw.rect(self._surface, (35, 73, 53) if selected else (24, 30, 39), row, border_radius=4)
                pygame.draw.rect(self._surface, DASHBOARD.success_color if selected else DASHBOARD.panel_border_color, row, width=1, border_radius=4)
                mark = "[x]" if selected else "[ ]"
                heading = f"{mark} {info.route_name} | {info.created_at or info.recording_id}"
                self._draw_text(heading, (row.left + 8, row.top + 6), self._font, DASHBOARD.title_color, row.width - 16)
                detail = (
                    f"{display_map_name(info.map_name)} | {info.sample_count or 'n/a'} samples | "
                    f"Sensor {info.sensor_noise_preset or 'n/a'} | Driver {info.recording_driver or 'unknown'} | "
                    f"Behavior {info.vehicle_behavior_preset or 'n/a'}"
                )
                self._draw_text(detail, (row.left + 8, row.top + 30), self._small_font, DASHBOARD.muted_text_color, row.width - 16)
        status_top = list_rect.bottom + 10
        status_rect = pygame.Rect(content.left, status_top, content.width, content.bottom - status_top)
        pygame.draw.rect(self._surface, (14, 18, 24), status_rect, border_radius=4)
        y = status_rect.top + 8
        for line in self._offline_status_lines[:4]:
            self._draw_text(line, (status_rect.left + 8, y), self._small_font, DASHBOARD.text_color, status_rect.width - 16)
            y += 18

    def _draw_record_sensor_data_panel(self, rect: pygame.Rect) -> None:
        gap = 12
        if rect.width >= 980:
            left = pygame.Rect(rect.left, rect.top, max(360, int(rect.width * 0.44)), rect.height)
            right = pygame.Rect(left.right + gap, rect.top, rect.right - left.right - gap, rect.height)
            self._draw_route_selection(
                left,
                selected_indices={self._recording_route_index} if self._recording_route_index is not None else set(),
                multi_select=False,
                title="Route to Record",
            )
            self._draw_recording_sensor_panel(right)
        else:
            top = pygame.Rect(rect.left, rect.top, rect.width, max(180, int(rect.height * 0.44)))
            lower = pygame.Rect(rect.left, top.bottom + gap, rect.width, rect.bottom - top.bottom - gap)
            self._draw_route_selection(
                top,
                selected_indices={self._recording_route_index} if self._recording_route_index is not None else set(),
                multi_select=False,
                title="Route to Record",
            )
            self._draw_recording_sensor_panel(lower)

    def _draw_recording_sensor_panel(self, rect: pygame.Rect) -> None:
        note_height = 62
        editor_rect = pygame.Rect(rect.left, rect.top, rect.width, max(90, rect.height - note_height - 10))
        if self._sensor_editor is not None:
            self._sensor_editor.draw(self._surface, editor_rect)
        note = pygame.Rect(rect.left, editor_rect.bottom + 10, rect.width, min(note_height, rect.bottom - editor_rect.bottom - 10))
        pygame.draw.rect(self._surface, (18, 23, 30), note, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, note, width=1, border_radius=6)
        content = note.inflate(-14, -10)
        self._draw_text("Recording driver: ground_truth_controller", content.topleft, self._small_font, DASHBOARD.text_color, content.width)
        self._draw_text("Vehicle behavior: Balanced default", (content.left, content.top + 20), self._small_font, DASHBOARD.muted_text_color, content.width)

    def _draw_closed_loop_filter_selection(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self._surface, DASHBOARD.panel_inner_color, rect, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, rect, width=1, border_radius=6)
        content = rect.inflate(-2 * DASHBOARD.panel_padding_px, -2 * DASHBOARD.panel_padding_px)
        self._tracking_button_rects.clear()
        self._draw_text("Filter Selection", content.topleft, self._subtitle_font, DASHBOARD.title_color, content.width)
        y = content.top + 34
        self._setup_filter_buttons.clear()
        if not self._setup_filter_records:
            self._draw_text("No valid filters found.", (content.left, y), self._font, DASHBOARD.warning_color, content.width)
            return
        for record in self._setup_filter_records:
            button = pygame.Rect(content.left, y, content.width, 44)
            self._setup_filter_buttons[record.filter_id] = button
            active = record.filter_id == self._selected_filter_id
            selectable = record.benchmark_selectable and record.filter_id != "raw_gnss"
            pygame.draw.rect(self._surface, (35, 73, 53) if active else (24, 30, 39), button, border_radius=5)
            border = DASHBOARD.success_color if active else (DASHBOARD.panel_border_color if selectable else DASHBOARD.warning_color)
            pygame.draw.rect(self._surface, border, button, width=1, border_radius=5)
            label = f"{record.display_name} ({record.filter_id})"
            self._draw_text(label, (button.left + 10, button.top + 6), self._button_font, DASHBOARD.title_color, button.width - 20)
            status = self._filter_capability_summary(record)
            if not selectable:
                status = f"{status} | disabled for closed-loop benchmark"
            status_color = DASHBOARD.warning_color if "experimental" in status.lower() or "not safe" in status.lower() or "disabled" in status.lower() else DASHBOARD.muted_text_color
            self._draw_text(status, (button.left + 10, button.top + 25), self._small_font, status_color, button.width - 20)
            y += 52
        y += 8
        active = next((record for record in self._setup_filter_records if record.filter_id == self._selected_filter_id), None)
        if active is not None:
            lines = self._filter_info_lines(active, include_filter_id=False, include_autonomous=True)
            for line in lines:
                if y + 16 > content.bottom:
                    break
                color = DASHBOARD.warning_color if "NO" in line or "Experimental: YES" in line else DASHBOARD.text_color
                self._draw_text(line, (content.left, y), self._small_font, color, content.width)
                y += 18
        if y + 58 <= content.bottom:
            y += 8
            self._draw_text("Tracking Mode", (content.left, y), self._small_font, DASHBOARD.muted_text_color, content.width)
            y += 20
            self._draw_tracking_mode_buttons(pygame.Rect(content.left, y, content.width, 28))

    def _draw_offline_filter_setup(self, rect: pygame.Rect) -> None:
        gap = 12
        if rect.width >= 980:
            left = pygame.Rect(rect.left, rect.top, max(300, int(rect.width * 0.34)), rect.height)
            right = pygame.Rect(left.right + gap, rect.top, rect.right - left.right - gap, rect.height)
        else:
            left = pygame.Rect(rect.left, rect.top, rect.width, max(180, int(rect.height * 0.36)))
            right = pygame.Rect(rect.left, left.bottom + gap, rect.width, rect.bottom - left.bottom - gap)
        self._draw_offline_filter_tabs(left)
        self._draw_offline_filter_detail(right)

    def _draw_offline_filter_tabs(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self._surface, DASHBOARD.panel_inner_color, rect, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, rect, width=1, border_radius=6)
        content = rect.inflate(-2 * DASHBOARD.panel_padding_px, -2 * DASHBOARD.panel_padding_px)
        self._offline_filter_tab_rects.clear()
        self._draw_text("Replay Filters", content.topleft, self._subtitle_font, DASHBOARD.title_color, content.width)
        self._draw_text("Raw GNSS is always included as the baseline.", (content.left, content.top + 28), self._small_font, DASHBOARD.muted_text_color, content.width)
        y = content.top + 58
        records = [record for record in self._setup_filter_records if record.filter_id != "raw_gnss"]
        if not records:
            self._draw_text("No replay filters found.", (content.left, y), self._font, DASHBOARD.warning_color, content.width)
            return
        row_h = 48
        for record in records:
            if y + row_h > content.bottom:
                break
            row = pygame.Rect(content.left, y, content.width, row_h - 6)
            self._offline_filter_tab_rects[record.filter_id] = row
            active = record.filter_id == self._active_offline_filter_id
            included = record.filter_id in self._offline_filter_ids
            selectable = bool(record.benchmark_selectable)
            background = (35, 73, 53) if active else (24, 30, 39)
            border = DASHBOARD.success_color if active else (DASHBOARD.panel_border_color if selectable else DASHBOARD.warning_color)
            pygame.draw.rect(self._surface, background, row, border_radius=5)
            pygame.draw.rect(self._surface, border, row, width=1, border_radius=5)
            mark = "[x]" if included else "[ ]"
            toggle = pygame.Rect(row.left + 8, row.top + 7, 28, 22)
            self._offline_filter_include_rects[record.filter_id] = toggle
            self._draw_button(toggle, mark, muted=not selectable)
            label = f"{record.display_name} ({record.filter_id})"
            self._draw_text(label, (toggle.right + 8, row.top + 6), self._button_font, DASHBOARD.title_color, row.right - toggle.right - 16)
            status = "included" if included else "not included"
            if not selectable:
                status = "not benchmark-selectable"
            elif record.experimental:
                status = f"{status} | experimental"
            self._draw_text(status, (toggle.right + 8, row.top + 25), self._small_font, DASHBOARD.warning_color if not selectable or record.experimental else DASHBOARD.muted_text_color, row.right - toggle.right - 16)
            y += row_h

    def _draw_offline_filter_detail(self, rect: pygame.Rect) -> None:
        record = self._filter_record(self._active_offline_filter_id)
        if record is None:
            pygame.draw.rect(self._surface, DASHBOARD.panel_inner_color, rect, border_radius=6)
            pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, rect, width=1, border_radius=6)
            content = rect.inflate(-2 * DASHBOARD.panel_padding_px, -2 * DASHBOARD.panel_padding_px)
            self._draw_text("Select a filter tab.", content.topleft, self._subtitle_font, DASHBOARD.title_color, content.width)
            return
        top_h = min(198, max(152, int(rect.height * 0.36)))
        info_rect = pygame.Rect(rect.left, rect.top, rect.width, top_h)
        tune_rect = pygame.Rect(rect.left, info_rect.bottom + 10, rect.width, rect.bottom - info_rect.bottom - 10)
        pygame.draw.rect(self._surface, DASHBOARD.panel_inner_color, info_rect, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, info_rect, width=1, border_radius=6)
        content = info_rect.inflate(-2 * DASHBOARD.panel_padding_px, -2 * DASHBOARD.panel_padding_px)
        self._draw_text(record.display_name, content.topleft, self._subtitle_font, DASHBOARD.title_color, content.width)
        include_rect = pygame.Rect(content.left, content.top + 30, 26, 24)
        self._offline_filter_include_rects[record.filter_id] = include_rect
        included = record.filter_id in self._offline_filter_ids
        selectable = bool(record.benchmark_selectable)
        self._draw_button(include_rect, "[x]" if included else "[ ]", muted=not selectable)
        self._draw_text("Include this filter in offline replay", (include_rect.right + 8, include_rect.top + 5), self._small_font, DASHBOARD.text_color if selectable else DASHBOARD.warning_color, content.width - include_rect.width - 12)
        y = include_rect.bottom + 8
        for line in self._filter_info_lines(record, include_filter_id=True, include_autonomous=False):
            if y + 16 > content.bottom:
                break
            color = DASHBOARD.warning_color if "NO" in line or "Experimental: YES" in line else DASHBOARD.text_color
            self._draw_text(line, (content.left, y), self._small_font, color, content.width)
            y += 17
        if self._offline_filter_saved_tune_mode:
            self._draw_saved_tune_browser(tune_rect, record.filter_id)
        else:
            self._draw_filter_tune_panel(tune_rect, record.filter_id, force_passive=True)

    def _filter_info_lines(self, record: object, include_filter_id: bool, include_autonomous: bool) -> list[str]:
        lines = []
        if include_filter_id:
            lines.extend(
                [
                    f"Filter id: {record.filter_id}",
                    f"Display name: {record.display_name}",
                ]
            )
        lines.extend(
            [
                f"Model type: {record.filter_info.get('model_type', 'n/a')}",
                f"Type: {record.filter_info.get('type')}",
                f"State: {record.filter_info.get('state_vector')}",
                f"Process model: {record.filter_info.get('process_model')}",
                f"Measurement model: {record.filter_info.get('measurement_model')}",
            ]
        )
        if include_autonomous:
            lines.extend(
                [
                    f"Safe autonomous: {'YES' if record.safe_for_autonomous_control else 'NO'}",
                    f"Active tracking: {'YES' if record.active_tracking_supported else 'NO'}",
                ]
            )
        lines.extend(
            [
                f"Benchmark selectable: {'YES' if record.benchmark_selectable else 'NO'}",
                f"Experimental: {'YES' if record.experimental else 'NO'}",
            ]
        )
        note = str(record.filter_info.get("autonomous_control_note") or "")
        if include_autonomous and note:
            lines.append(note)
        return lines

    @staticmethod
    def _filter_capability_summary(record: object) -> str:
        safe = "safe" if getattr(record, "safe_for_autonomous_control", False) else "not safe"
        tracking = "active" if getattr(record, "active_tracking_supported", False) else "passive"
        selectable = "bench" if getattr(record, "benchmark_selectable", False) else "no bench"
        experimental = "experimental" if getattr(record, "experimental", False) else "stable"
        return f"{safe} | {tracking} | {selectable} | {experimental}"

    def _draw_filter_tune_panel(self, rect: pygame.Rect, filter_id: str, force_passive: bool = False) -> None:
        self._filter_tune_panel_rect = rect.copy()
        recommendation_height = min(112, max(86, int(rect.height * 0.24)))
        recommendation_rect = pygame.Rect(rect.left, rect.bottom - recommendation_height, rect.width, recommendation_height)
        editor_rect = pygame.Rect(rect.left, rect.top, rect.width, max(80, recommendation_rect.top - rect.top - 10))
        if self._filter_tune_editor is not None and self._filter_tune_editor_filter_id == filter_id:
            self._filter_tune_editor.draw(self._surface, editor_rect)
        else:
            pygame.draw.rect(self._surface, DASHBOARD.panel_inner_color, editor_rect, border_radius=6)
            pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, editor_rect, width=1, border_radius=6)
            content = editor_rect.inflate(-2 * DASHBOARD.panel_padding_px, -2 * DASHBOARD.panel_padding_px)
            self._draw_text("Tune Parameters", content.topleft, self._subtitle_font, DASHBOARD.title_color, content.width)
            self._draw_text(
                "Selected filter exposes no editable tune specs.",
                (content.left, content.top + 34),
                self._small_font,
                DASHBOARD.muted_text_color,
                content.width,
            )
        if force_passive:
            self._draw_auto_tune_card(recommendation_rect, filter_id)
        else:
            self._draw_recommendation_card(recommendation_rect, filter_id, force_passive=force_passive)

    def _draw_tracking_mode_buttons(self, rect: pygame.Rect) -> None:
        self._tracking_button_rects.clear()
        gap = 8
        button_width = max(60, (rect.width - gap) // 2)
        for index, (mode, label) in enumerate(((TRACKING_MODE_PASSIVE, "Passive"), (TRACKING_MODE_ACTIVE, "Active"))):
            button = pygame.Rect(rect.left + index * (button_width + gap), rect.top, button_width, rect.height)
            self._tracking_button_rects[mode] = button
            active = self._tracking_mode == mode
            hovered = button.collidepoint(pygame.mouse.get_pos())
            background = (35, 73, 53) if active else ((38, 47, 61) if hovered else (24, 30, 39))
            border = DASHBOARD.success_color if active else DASHBOARD.panel_border_color
            pygame.draw.rect(self._surface, background, button, border_radius=5)
            pygame.draw.rect(self._surface, border, button, width=1, border_radius=5)
            rendered = self._button_font.render(label, True, DASHBOARD.title_color if active else DASHBOARD.text_color)
            self._surface.blit(rendered, rendered.get_rect(center=button.center))

    def _draw_recommendation_card(self, rect: pygame.Rect, filter_id: str, force_passive: bool = False) -> None:
        pygame.draw.rect(self._surface, (18, 23, 30), rect, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, rect, width=1, border_radius=6)
        content = rect.inflate(-2 * DASHBOARD.panel_padding_px, -2 * DASHBOARD.panel_padding_px)
        self._draw_text("Recommendation", content.topleft, self._subtitle_font, DASHBOARD.title_color, content.width)
        recommendation = self._current_recommendation(filter_id, force_passive=force_passive)
        line_y = content.top + 30
        shown_lines = list(recommendation.messages[:2])
        shown_lines.extend(recommendation.warnings[:1])
        if not shown_lines:
            shown_lines = ["No recommendation for this filter."]
        for line in shown_lines[:3]:
            color = DASHBOARD.warning_color if line in recommendation.warnings else DASHBOARD.muted_text_color
            self._draw_text(line, (content.left, line_y), self._small_font, color, content.width)
            line_y += 17
        self._apply_recommended_rect = pygame.Rect(content.right - 142, content.bottom - 28, 142, 24)
        self._draw_button(self._apply_recommended_rect, "Apply Recommended", muted=not recommendation.has_values)

    def _draw_auto_tune_card(self, rect: pygame.Rect, filter_id: str) -> None:
        pygame.draw.rect(self._surface, (18, 23, 30), rect, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, rect, width=1, border_radius=6)
        content = rect.inflate(-2 * DASHBOARD.panel_padding_px, -2 * DASHBOARD.panel_padding_px)
        record = self._filter_record(filter_id)
        profile = record.auto_tune_profile if record is not None else None
        available = bool(record is not None and record.auto_tune_enabled and isinstance(profile, dict) and profile.get("primary"))
        self._draw_text("Auto Tune", content.topleft, self._subtitle_font, DASHBOARD.title_color, content.width)
        if available:
            text = "Auto Tune tunes this selected filter using multiple recorded sensor logs. Raw GNSS is not tuned."
            color = DASHBOARD.muted_text_color
        else:
            text = "This filter does not expose AUTO_TUNE_PROFILE. Manual tuning is still available."
            color = DASHBOARD.warning_color
        self._draw_wrapped_text(text, pygame.Rect(content.left, content.top + 28, content.width, 36), self._small_font, color)
        saved = list_saved_tune_configs(filter_id)
        status = self._offline_saved_tune_status or (f"Saved configs: {len(saved)}" if saved else "No saved tune config yet.")
        self._draw_text(status, (content.left, content.bottom - 26), self._small_font, DASHBOARD.text_color, content.width - 300)
        self._offline_auto_tune_rect = pygame.Rect(content.right - 310, content.bottom - 30, 132, 24)
        self._offline_saved_tunes_rect = pygame.Rect(content.right - 170, content.bottom - 30, 170, 24)
        self._draw_button(self._offline_auto_tune_rect, "Start Auto Tune", muted=not available or self._auto_tune_running)
        self._draw_button(self._offline_saved_tunes_rect, "Browse Saved Tunes", muted=self._auto_tune_running)

    def _draw_saved_tune_browser(self, rect: pygame.Rect, filter_id: str) -> None:
        pygame.draw.rect(self._surface, DASHBOARD.panel_inner_color, rect, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, rect, width=1, border_radius=6)
        content = rect.inflate(-2 * DASHBOARD.panel_padding_px, -2 * DASHBOARD.panel_padding_px)
        self._offline_saved_tune_apply_rects.clear()
        self._draw_text("Saved Tune Configs", content.topleft, self._subtitle_font, DASHBOARD.title_color, content.width)
        self._offline_back_to_sliders_rect = pygame.Rect(content.right - 210, content.top, 210, 26)
        self._draw_button(self._offline_back_to_sliders_rect, "Back to Manual Tune Sliders")
        configs = list_saved_tune_configs(filter_id)
        if not configs:
            self._draw_text("No saved tune configs for this filter.", (content.left, content.top + 42), self._font, DASHBOARD.warning_color, content.width)
            return
        row_h = 62
        y = content.top + 38
        for index, item in enumerate(configs[: max(1, (content.height - 38) // row_h)]):
            row = pygame.Rect(content.left, y, content.width, row_h - 8)
            pygame.draw.rect(self._surface, (24, 30, 39), row, border_radius=5)
            pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, row, width=1, border_radius=5)
            apply_rect = pygame.Rect(row.right - 78, row.top + 14, 68, 26)
            self._offline_saved_tune_apply_rects[index] = apply_rect
            name = Path(str(item.get("path") or "")).parent.name or str(item.get("created_at") or "saved tune")
            score = item.get("score")
            rmse = item.get("mean_eval_position_rmse_m")
            score_text = f"score {float(score):.3f}" if isinstance(score, (int, float)) else "score n/a"
            rmse_text = f"RMSE {float(rmse):.3f}m" if isinstance(rmse, (int, float)) else "RMSE n/a"
            title = f"{name} | {item.get('noise_profile_label') or 'Unknown Noise'} | {score_text} | {rmse_text}"
            detail = f"{item.get('log_count') or 0} log(s) | {item.get('created_at') or ''}"
            self._draw_text(title, (row.left + 10, row.top + 7), self._small_font, DASHBOARD.title_color, row.width - 100)
            self._draw_text(detail, (row.left + 10, row.top + 30), self._small_font, DASHBOARD.muted_text_color, row.width - 100)
            self._draw_button(apply_rect, "Apply")
            y += row_h

    def _draw_route_selection(
        self,
        rect: pygame.Rect,
        selected_indices: set[int],
        multi_select: bool,
        title: str,
    ) -> None:
        pygame.draw.rect(self._surface, DASHBOARD.panel_inner_color, rect, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, rect, width=1, border_radius=6)
        content = rect.inflate(-2 * DASHBOARD.panel_padding_px, -2 * DASHBOARD.panel_padding_px)
        self._select_all_routes_rect = pygame.Rect(0, 0, 1, 1)
        self._clear_routes_rect = pygame.Rect(0, 0, 1, 1)
        selected_count = len(selected_indices)
        self._draw_text(f"{title} ({selected_count} selected)", content.topleft, self._subtitle_font, DASHBOARD.title_color, content.width)
        list_top = content.top + 36
        if multi_select:
            self._select_all_routes_rect = pygame.Rect(content.left, content.top + 30, 92, 24)
            self._clear_routes_rect = pygame.Rect(self._select_all_routes_rect.right + 8, content.top + 30, 86, 24)
            self._draw_button(self._select_all_routes_rect, "Select All")
            self._draw_button(self._clear_routes_rect, "Clear")
            list_top = content.top + 62
        list_rect = pygame.Rect(content.left, list_top, content.width, content.bottom - list_top)
        pygame.draw.rect(self._surface, (14, 18, 24), list_rect, border_radius=4)
        self._route_rects.clear()
        if not self._route_items:
            self._draw_text("No saved routes in config/test_routes.json.", (list_rect.left + 8, list_rect.top + 10), self._small_font, DASHBOARD.warning_color, list_rect.width - 16)
            return
        row_h = 56
        visible = max(1, list_rect.height // row_h)
        if multi_select:
            self._closed_loop_route_scroll = min(self._closed_loop_route_scroll, max(0, len(self._route_items) - visible))
            scroll = self._closed_loop_route_scroll
        else:
            self._recording_route_scroll = min(self._recording_route_scroll, max(0, len(self._route_items) - visible))
            scroll = self._recording_route_scroll
        for visible_index, item in enumerate(self._route_items[scroll : scroll + visible]):
            row = pygame.Rect(list_rect.left + 5, list_rect.top + 5 + visible_index * row_h, list_rect.width - 10, row_h - 6)
            self._route_rects[item.index] = row
            selected = item.index in selected_indices
            pygame.draw.rect(self._surface, (35, 73, 53) if selected else (24, 30, 39), row, border_radius=4)
            pygame.draw.rect(self._surface, DASHBOARD.success_color if selected else DASHBOARD.panel_border_color, row, width=1, border_radius=4)
            mark = "[x]" if selected else "[ ]"
            length = f"{item.straight_line_length_m:.0f}m" if item.straight_line_length_m is not None else "n/a"
            status = "available" if item.compatible_with_available_maps else "map not listed"
            self._draw_text(f"{mark} {item.route.name}", (row.left + 8, row.top + 6), self._font, DASHBOARD.title_color, row.width - 16)
            self._draw_text(f"{display_map_name(item.route.map_name)} | {length} | {status}", (row.left + 8, row.top + 29), self._small_font, DASHBOARD.muted_text_color, row.width - 16)

    def _draw_closed_loop_footer(self, rect: pygame.Rect) -> None:
        self._start_benchmark_rect = pygame.Rect(0, 0, 1, 1)
        self._record_sensor_logs_rect = pygame.Rect(0, 0, 1, 1)
        self._run_offline_replay_rect = pygame.Rect(0, 0, 1, 1)
        self._start_benchmark_rect = pygame.Rect(rect.right - 250, rect.top + 3, 250, 34)
        self._draw_button(self._start_benchmark_rect, "Start Closed-Loop Benchmark", primary=True)
        summary = self._closed_loop_summary_text()
        summary_right = self._start_benchmark_rect.left - 12
        self._draw_text(summary, (rect.left, rect.top + 10), self._small_font, DASHBOARD.muted_text_color, summary_right - rect.left)

    def _draw_offline_footer(self, rect: pygame.Rect, show_record_button: bool, show_run_button: bool) -> None:
        self._start_benchmark_rect = pygame.Rect(0, 0, 1, 1)
        self._record_sensor_logs_rect = pygame.Rect(0, 0, 1, 1)
        self._run_offline_replay_rect = pygame.Rect(0, 0, 1, 1)
        summary_right = rect.right
        if show_record_button:
            self._record_sensor_logs_rect = pygame.Rect(rect.right - 236, rect.top + 3, 236, 34)
            self._draw_button(self._record_sensor_logs_rect, "Record Selected Route Log", primary=True)
            summary_right = self._record_sensor_logs_rect.left - 12
        elif show_run_button:
            self._run_offline_replay_rect = pygame.Rect(rect.right - 282, rect.top + 3, 282, 34)
            self._draw_button(self._run_offline_replay_rect, "Run Offline Localization Benchmark", muted=self._auto_tune_running, primary=True)
            summary_right = self._run_offline_replay_rect.left - 12
        self._draw_text(self._offline_summary_text(), (rect.left, rect.top + 10), self._small_font, DASHBOARD.muted_text_color, summary_right - rect.left)

    def _closed_loop_summary_text(self) -> str:
        routes = [item.route for item in self._route_items if item.index in self._selected_route_indices]
        maps = sorted({display_map_name(route.map_name) for route in routes})
        filter_label = self._selected_filter_id or "none"
        map_text = ", ".join(maps[:3]) + ("..." if len(maps) > 3 else "")
        return (
            f"Closed-loop | Filter {filter_label} | Tracking {self._tracking_mode} | Routes {len(routes)} | "
            f"Maps {map_text or 'none'} | Sensor {self._sensor_preset} | Behavior {self._behavior_preset}"
        )

    def _offline_summary_text(self) -> str:
        log_label = "none"
        sensor = self._sensor_preset
        if self._selected_recorded_log_index is not None and self._selected_recorded_log_index < len(self._recorded_logs):
            info = self._recorded_logs[self._selected_recorded_log_index]
            log_label = f"{info.route_name}/{info.recording_id}"
            sensor = info.sensor_noise_preset or sensor
        filters = ", ".join(sorted(self._offline_filter_ids)) or "none"
        return f"Offline localization | Log {log_label} | Filters {filters} + raw_gnss | Sensor {sensor}"

    def _apply_saved_tune_config(self, config_index: int) -> None:
        configs = list_saved_tune_configs(self._active_offline_filter_id)
        if config_index < 0 or config_index >= len(configs):
            return
        config_path = Path(str(configs[config_index].get("path") or ""))
        config = load_saved_tune_config(config_path)
        best_tune = config.get("best_tune")
        if not isinstance(best_tune, dict):
            self._offline_saved_tune_status = "Saved tune config is missing best_tune."
            return
        self._apply_tune_to_filter(self._active_offline_filter_id, best_tune)
        name = config_path.parent.name or config_path.name
        self._offline_saved_tune_status = f"Applied saved tune config {name} to {self._active_offline_filter_id}."
        self._offline_filter_saved_tune_mode = False

    def _apply_tune_to_filter(self, filter_id: str, tune: dict[str, object]) -> None:
        record = self._filter_record(filter_id)
        if record is None:
            return
        merged = dict(self._selected_filter_tunes.get(filter_id, record.tune))
        for key, value in tune.items():
            if key in record.tune or any(getattr(spec, "key", None) == key for spec in getattr(record, "tune_specs", ())):
                merged[str(key)] = value
        self._selected_filter_tunes[filter_id] = merged
        if self._filter_tune_editor_filter_id == filter_id:
            self._ensure_filter_tune_editor(filter_id)

    def _open_auto_tune_modal(self, filter_id: str) -> None:
        record = self._filter_record(filter_id)
        if record is None or not record.auto_tune_enabled or not isinstance(record.auto_tune_profile, dict):
            self._offline_saved_tune_status = "No auto-tune profile for this filter."
            return
        self._auto_tune_modal_open = True
        self._auto_tune_filter_id = filter_id
        search = record.auto_tune_profile.get("search") if isinstance(record.auto_tune_profile.get("search"), dict) else {}
        self._auto_tune_trials = int(search.get("default_trials") or 30)
        self._auto_tune_selected_log_indices = set()
        self._auto_tune_log_scroll = 0
        self._auto_tune_stop_requested = False
        self._auto_tune_result = None
        self._auto_tune_status_lines = [
            f"Auto tune ready for {record.display_name}.",
            "Select multiple recorded logs, then press Start.",
        ]

    def _handle_auto_tune_modal_event(self, event: pygame.event.Event) -> object:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self._auto_tune_running:
                self._auto_tune_stop_requested = True
                self._append_auto_tune_status("Stop requested. Auto tune will stop after the current trial.")
            else:
                self._auto_tune_modal_open = False
            return _NoSelection
        if event.type == pygame.MOUSEWHEEL:
            self._auto_tune_log_scroll = max(0, self._auto_tune_log_scroll - event.y * 2)
            return _NoSelection
        if event.type != pygame.MOUSEBUTTONDOWN or getattr(event, "button", None) != 1 or not hasattr(event, "pos"):
            return _NoSelection
        position = event.pos
        if self._auto_tune_close_rect.collidepoint(position):
            if self._auto_tune_running:
                self._auto_tune_stop_requested = True
                self._append_auto_tune_status("Stop requested. Auto tune will stop after the current trial.")
            else:
                self._auto_tune_modal_open = False
            return _NoSelection
        if self._auto_tune_refresh_logs_rect.collidepoint(position):
            self._refresh_recorded_logs()
            self._append_auto_tune_status(f"Found {len(self._recorded_logs)} recorded log(s).")
            return _NoSelection
        if self._auto_tune_clear_logs_rect.collidepoint(position):
            self._auto_tune_selected_log_indices.clear()
            return _NoSelection
        if self._auto_tune_select_all_noise_rect.collidepoint(position):
            self._select_all_auto_tune_matching_noise()
            return _NoSelection
        for index, rect in self._auto_tune_log_rects.items():
            if rect.collidepoint(position):
                if index in self._auto_tune_selected_log_indices:
                    self._auto_tune_selected_log_indices.remove(index)
                else:
                    self._auto_tune_selected_log_indices.add(index)
                return _NoSelection
        if self._auto_tune_start_rect.collidepoint(position):
            if not self._auto_tune_running:
                self._start_auto_tune_from_modal()
            return _NoSelection
        if self._auto_tune_cancel_rect.collidepoint(position):
            if self._auto_tune_running:
                self._auto_tune_stop_requested = True
                self._append_auto_tune_status("Stop requested. Auto tune will stop after the current trial.")
            return _NoSelection
        if self._auto_tune_save_rect.collidepoint(position):
            if self._auto_tune_result is not None and self._auto_tune_result.saved_config_path is not None:
                self._append_auto_tune_status(f"Best tune already saved: {self._auto_tune_result.saved_config_path}")
            return _NoSelection
        if self._auto_tune_apply_rect.collidepoint(position):
            if self._auto_tune_result is not None and self._auto_tune_result.best_tune:
                self._apply_tune_to_filter(self._auto_tune_result.filter_id, self._auto_tune_result.best_tune)
                self._offline_saved_tune_status = f"Applied auto-tuned values to {self._auto_tune_result.filter_id}."
                self._append_auto_tune_status(self._offline_saved_tune_status)
            return _NoSelection
        return _NoSelection

    def _start_auto_tune_from_modal(self) -> None:
        record = self._filter_record(self._auto_tune_filter_id)
        if record is None or not isinstance(record.auto_tune_profile, dict):
            self._append_auto_tune_status("No auto-tune profile for this filter.")
            return
        selected_logs = [
            info.sensor_log_path
            for index, info in enumerate(self._recorded_logs)
            if index in self._auto_tune_selected_log_indices
        ]
        if not selected_logs:
            self._append_auto_tune_status("Select at least one recorded log before starting.")
            return
        self._commit_filter_tune_editor()
        self._auto_tune_running = True
        self._auto_tune_stop_requested = False
        self._auto_tune_result = None
        self._append_auto_tune_status(f"Starting auto tune for {record.display_name} on {len(selected_logs)} log(s).")
        request = AutoTuneRequest(
            filter_id=record.filter_id,
            sensor_log_paths=tuple(selected_logs),
            base_tune=self._current_filter_tune_values(record.filter_id),
            auto_tune_profile=dict(record.auto_tune_profile),
            max_trials=max(1, int(self._auto_tune_trials or 30)),
            objective_name=str(record.auto_tune_profile.get("objective") or "rmse_consistency"),
            metadata={"startup_mode": "offline_auto_tune", "random_seed": 4084},
        )
        try:
            self._auto_tune_result = FilterAutoTuner().run(
                request,
                progress_callback=self._auto_tune_progress_callback,
                stop_requested=lambda: self._auto_tune_stop_requested,
            )
        except Exception as exc:
            self._append_auto_tune_status(f"Auto tune failed: {exc}")
        finally:
            self._auto_tune_running = False
            if self._auto_tune_result is not None:
                score = self._auto_tune_result.best_score
                score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
                self._append_auto_tune_status(f"Auto tune complete. Best score: {score_text}")

    def _auto_tune_progress_callback(self, event_name: str, payload: dict[str, object]) -> None:
        if event_name == "trial_started":
            trial = payload.get("trial_index")
            tune = payload.get("candidate_tune") if isinstance(payload.get("candidate_tune"), dict) else {}
            shown = ", ".join(f"{key}={float(value):.3g}" for key, value in list(tune.items())[:4] if isinstance(value, (int, float)))
            self._append_auto_tune_status(f"Trial {trial}/{self._auto_tune_trials} | {self._auto_tune_filter_id} | {shown}", redraw=False)
        elif event_name == "trial_finished":
            score = payload.get("score")
            failed = bool(payload.get("failed"))
            if failed:
                self._append_auto_tune_status(f"Trial {payload.get('trial_index')} failed: {payload.get('failure_reason')}", redraw=False)
            else:
                score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
                self._append_auto_tune_status(f"Trial {payload.get('trial_index')} score: {score_text}", redraw=False)
        elif event_name == "new_best":
            score = payload.get("score")
            score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
            self._append_auto_tune_status(f"New best tune found. Score: {score_text}", redraw=False)
        elif event_name == "completed":
            self._append_auto_tune_status(f"Saved best tune: {payload.get('saved_config_path')}", redraw=False)
        self._redraw_auto_tune_modal_frame()
        self._pump_auto_tune_cancel_events()

    def _append_auto_tune_status(self, line: str, redraw: bool = True) -> None:
        self._auto_tune_status_lines.append(str(line))
        self._auto_tune_status_lines = self._auto_tune_status_lines[-60:]
        if redraw:
            self._redraw_auto_tune_modal_frame()

    def _redraw_auto_tune_modal_frame(self) -> None:
        if not self._auto_tune_modal_open:
            return
        self._draw(status_only=False)
        pygame.display.flip()
        pygame.event.pump()

    def _pump_auto_tune_cancel_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._auto_tune_stop_requested = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._auto_tune_stop_requested = True
            elif event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", None) == 1 and hasattr(event, "pos"):
                if self._auto_tune_cancel_rect.collidepoint(event.pos) or self._auto_tune_close_rect.collidepoint(event.pos):
                    self._auto_tune_stop_requested = True

    def _select_all_auto_tune_matching_noise(self) -> None:
        if not self._recorded_logs:
            return
        reference = None
        if self._selected_recorded_log_index is not None and self._selected_recorded_log_index < len(self._recorded_logs):
            reference = self._recorded_logs[self._selected_recorded_log_index].sensor_noise_preset
        if reference is None and self._auto_tune_selected_log_indices:
            first = min(self._auto_tune_selected_log_indices)
            if first < len(self._recorded_logs):
                reference = self._recorded_logs[first].sensor_noise_preset
        if reference is None:
            reference = self._recorded_logs[0].sensor_noise_preset
        self._auto_tune_selected_log_indices = {
            index
            for index, info in enumerate(self._recorded_logs)
            if info.sensor_noise_preset == reference
        }
        self._append_auto_tune_status(f"Selected {len(self._auto_tune_selected_log_indices)} log(s) matching noise: {reference or 'n/a'}")

    def _draw_auto_tune_modal(self) -> None:
        width, height = self._surface.get_size()
        dim = pygame.Surface((width, height), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        self._surface.blit(dim, (0, 0))
        modal_w = min(width - 80, 980)
        modal_h = min(height - 70, 700)
        modal = pygame.Rect(0, 0, modal_w, modal_h)
        modal.center = (width // 2, height // 2)
        pygame.draw.rect(self._surface, DASHBOARD.panel_background_color, modal, border_radius=8)
        pygame.draw.rect(self._surface, DASHBOARD.success_color, modal, width=1, border_radius=8)
        content = modal.inflate(-24, -22)
        record = self._filter_record(self._auto_tune_filter_id)
        title = f"Auto Tune {record.display_name if record is not None else self._auto_tune_filter_id}"
        self._draw_text(title, content.topleft, self._subtitle_font, DASHBOARD.title_color, content.width - 90)
        self._auto_tune_close_rect = pygame.Rect(content.right - 74, content.top, 74, 26)
        self._draw_button(self._auto_tune_close_rect, "Close", muted=self._auto_tune_running)
        self._draw_text(
            "Tunes one selected filter over multiple recorded sensor logs.",
            (content.left, content.top + 30),
            self._small_font,
            DASHBOARD.muted_text_color,
            content.width,
        )

        list_top = content.top + 58
        footer_h = 42
        if content.width >= 900:
            left = pygame.Rect(content.left, list_top, int(content.width * 0.54), content.height - 58 - footer_h - 12)
            right = pygame.Rect(left.right + 12, list_top, content.right - left.right - 12, left.height)
        else:
            left = pygame.Rect(content.left, list_top, content.width, max(180, int((content.height - 58 - footer_h) * 0.48)))
            right = pygame.Rect(content.left, left.bottom + 10, content.width, content.bottom - footer_h - left.bottom - 10)
        self._draw_auto_tune_log_selection(left)
        self._draw_auto_tune_settings_and_console(right)
        self._draw_auto_tune_footer(pygame.Rect(content.left, content.bottom - footer_h, content.width, footer_h))

    def _draw_auto_tune_log_selection(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self._surface, DASHBOARD.panel_inner_color, rect, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, rect, width=1, border_radius=6)
        content = rect.inflate(-2 * DASHBOARD.panel_padding_px, -2 * DASHBOARD.panel_padding_px)
        self._draw_text(
            f"Recorded Sensor Logs ({len(self._auto_tune_selected_log_indices)} selected)",
            content.topleft,
            self._subtitle_font,
            DASHBOARD.title_color,
            content.width,
        )
        button_y = content.top + 30
        self._auto_tune_refresh_logs_rect = pygame.Rect(content.left, button_y, 112, 24)
        self._auto_tune_select_all_noise_rect = pygame.Rect(self._auto_tune_refresh_logs_rect.right + 8, button_y, 172, 24)
        self._auto_tune_clear_logs_rect = pygame.Rect(self._auto_tune_select_all_noise_rect.right + 8, button_y, 70, 24)
        self._draw_button(self._auto_tune_refresh_logs_rect, "Refresh Logs")
        self._draw_button(self._auto_tune_select_all_noise_rect, "Select All Matching Noise")
        self._draw_button(self._auto_tune_clear_logs_rect, "Clear")
        list_rect = pygame.Rect(content.left, button_y + 32, content.width, content.bottom - button_y - 32)
        pygame.draw.rect(self._surface, (14, 18, 24), list_rect, border_radius=4)
        self._auto_tune_log_rects.clear()
        if not self._recorded_logs:
            self._draw_text(
                "No recorded logs found. Record sensor data first.",
                (list_rect.left + 8, list_rect.top + 10),
                self._small_font,
                DASHBOARD.warning_color,
                list_rect.width - 16,
            )
            return
        row_h = 70
        visible = max(1, list_rect.height // row_h)
        self._auto_tune_log_scroll = min(self._auto_tune_log_scroll, max(0, len(self._recorded_logs) - visible))
        for visible_index, info in enumerate(self._recorded_logs[self._auto_tune_log_scroll : self._auto_tune_log_scroll + visible]):
            index = self._auto_tune_log_scroll + visible_index
            row = pygame.Rect(list_rect.left + 5, list_rect.top + 5 + visible_index * row_h, list_rect.width - 10, row_h - 6)
            self._auto_tune_log_rects[index] = row
            selected = index in self._auto_tune_selected_log_indices
            pygame.draw.rect(self._surface, (35, 73, 53) if selected else (24, 30, 39), row, border_radius=4)
            pygame.draw.rect(self._surface, DASHBOARD.success_color if selected else DASHBOARD.panel_border_color, row, width=1, border_radius=4)
            mark = "[x]" if selected else "[ ]"
            title = f"{mark} {info.route_name} | {display_map_name(info.map_name)} | {info.sample_count or 'n/a'} samples"
            detail = (
                f"Sensor {info.sensor_noise_preset or 'n/a'} {self._compact_log_noise_details(info)} | "
                f"Driver {info.recording_driver or 'unknown'} | Behavior {info.vehicle_behavior_preset or 'n/a'}"
            )
            stamp = info.created_at or info.recording_id
            self._draw_text(title, (row.left + 8, row.top + 6), self._small_font, DASHBOARD.title_color, row.width - 16)
            self._draw_text(detail, (row.left + 8, row.top + 27), self._small_font, DASHBOARD.muted_text_color, row.width - 16)
            self._draw_text(stamp, (row.left + 8, row.top + 48), self._small_font, DASHBOARD.muted_text_color, row.width - 16)

    def _draw_auto_tune_settings_and_console(self, rect: pygame.Rect) -> None:
        settings_h = min(116, max(92, int(rect.height * 0.24)))
        settings = pygame.Rect(rect.left, rect.top, rect.width, settings_h)
        console = pygame.Rect(rect.left, settings.bottom + 10, rect.width, rect.bottom - settings.bottom - 10)
        pygame.draw.rect(self._surface, DASHBOARD.panel_inner_color, settings, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, settings, width=1, border_radius=6)
        content = settings.inflate(-2 * DASHBOARD.panel_padding_px, -2 * DASHBOARD.panel_padding_px)
        record = self._filter_record(self._auto_tune_filter_id)
        profile = record.auto_tune_profile if record is not None and isinstance(record.auto_tune_profile, dict) else {}
        search = profile.get("search") if isinstance(profile.get("search"), dict) else {}
        objective = str(profile.get("objective") or "rmse_consistency")
        lines = [
            f"Trials: {self._auto_tune_trials or search.get('default_trials') or 30}",
            f"Strategy: {search.get('strategy') or 'random_plus_coordinate_refinement'}",
            f"Objective: {objective}",
            "Use current GUI tune as base: ON",
        ]
        self._draw_text("Auto Tune Settings", content.topleft, self._subtitle_font, DASHBOARD.title_color, content.width)
        y = content.top + 28
        for line in lines:
            self._draw_text(line, (content.left, y), self._small_font, DASHBOARD.text_color, content.width)
            y += 17

        pygame.draw.rect(self._surface, (14, 18, 24), console, border_radius=6)
        pygame.draw.rect(self._surface, DASHBOARD.panel_border_color, console, width=1, border_radius=6)
        body = console.inflate(-12, -10)
        self._draw_text("Progress", body.topleft, self._subtitle_font, DASHBOARD.title_color, body.width)
        line_y = body.top + 30
        line_h = 17
        max_lines = max(1, (body.bottom - line_y) // line_h)
        for line in self._auto_tune_status_lines[-max_lines:]:
            color = DASHBOARD.warning_color if "failed" in line.lower() or "stop" in line.lower() else DASHBOARD.text_color
            self._draw_text(line, (body.left, line_y), self._small_font, color, body.width)
            line_y += line_h

    def _draw_auto_tune_footer(self, rect: pygame.Rect) -> None:
        self._auto_tune_start_rect = pygame.Rect(rect.right - 438, rect.top + 4, 82, 30)
        self._auto_tune_cancel_rect = pygame.Rect(self._auto_tune_start_rect.right + 8, rect.top + 4, 96, 30)
        self._auto_tune_save_rect = pygame.Rect(self._auto_tune_cancel_rect.right + 8, rect.top + 4, 118, 30)
        self._auto_tune_apply_rect = pygame.Rect(self._auto_tune_save_rect.right + 8, rect.top + 4, 126, 30)
        self._draw_button(self._auto_tune_start_rect, "Start", muted=self._auto_tune_running, primary=True)
        self._draw_button(self._auto_tune_cancel_rect, "Cancel / Stop", muted=not self._auto_tune_running)
        self._draw_button(self._auto_tune_save_rect, "Save Best Tune", muted=self._auto_tune_result is None)
        self._draw_button(self._auto_tune_apply_rect, "Apply Best Tune", muted=self._auto_tune_result is None)
        selected_paths = [info.sensor_log_path for index, info in enumerate(self._recorded_logs) if index in self._auto_tune_selected_log_indices]
        logs = [
            {
                "sensor_noise_preset": info.sensor_noise_preset,
                "sensor_noise_config": read_json(info.route_folder / "route_metadata.json").get("sensor_noise_config"),
            }
            for index, info in enumerate(self._recorded_logs)
            if index in self._auto_tune_selected_log_indices
        ]
        noise = noise_profile_summary(logs)
        summary = f"Selected logs: {len(selected_paths)} | Noise: {noise.get('label')}"
        if noise.get("mixed"):
            summary += " | Selected logs use mixed sensor noise profiles."
        self._draw_text(summary, (rect.left, rect.top + 12), self._small_font, DASHBOARD.warning_color if noise.get("mixed") else DASHBOARD.muted_text_color, self._auto_tune_start_rect.left - rect.left - 10)

    def _compact_log_noise_details(self, info: RecordedLogInfo) -> str:
        metadata = read_json(info.route_folder / "route_metadata.json")
        config = metadata.get("sensor_noise_config")
        if not isinstance(config, dict):
            return ""
        pairs = []
        for key in ("gnss_position_stddev_m", "imu_accel_stddev_mps2", "imu_gyro_stddev_radps", "imu_compass_stddev_deg"):
            value = config.get(key)
            if isinstance(value, (int, float)):
                pairs.append(f"{key}={float(value):.2g}")
        return f"({', '.join(pairs[:3])})" if pairs else ""

    def _start_benchmark_from_setup(self, client: object) -> object:
        self._commit_filter_tune_editor()
        routes = [item.route for item in self._route_items if item.index in self._selected_route_indices]
        sensor_values = self._sensor_editor.values() if self._sensor_editor is not None else SENSOR_NOISE_PRESETS["Medium Noise"]
        behavior_values = self._behavior_editor.values() if self._behavior_editor is not None else BEHAVIOR_PRESETS["Balanced"]
        if self._sensor_editor is not None:
            self._sensor_preset = self._sensor_editor.active_preset
        if self._behavior_editor is not None:
            self._behavior_preset = self._behavior_editor.active_preset
        selected_filter_tune = self._current_filter_tune_values()
        recommendation_applied = bool(self._recommendation_applied_by_filter.get(self._selected_filter_id, False))
        config = BenchmarkConfig(
            selected_filter=self._selected_filter_id,
            selected_routes=tuple(routes),
            sensor_noise_config=sensor_noise_config_from_values(sensor_values, preset_name=self._sensor_preset),
            vehicle_behavior_config=driving_behavior_from_values(behavior_values, preset_name=self._behavior_preset),
            selected_filter_tune=selected_filter_tune,
            tracking_mode=self._tracking_mode,
            sensor_noise_preset=self._sensor_preset,
            vehicle_behavior_preset=self._behavior_preset,
            metadata={
                "startup_mode": "closed_loop_benchmark",
                "filter_tune_recommendation_applied": recommendation_applied,
                "filter_tune_recommendation_applied_by_filter": dict(self._recommendation_applied_by_filter),
                "selected_filter_tune": dict(selected_filter_tune),
            },
        )
        errors = validate_benchmark_config(
            config,
            valid_filter_ids=[record.filter_id for record in self._setup_filter_records if record.benchmark_selectable],
            available_maps=[option.load_name or option.detail for option in self._options],
        )
        if errors:
            self._error = " ".join(errors[:3])
            return _NoSelection

        first_route = routes[0]
        return self._load_map_for_benchmark(client, first_route, config)

    def _start_offline_recording_from_setup(self, client: object) -> object:
        routes = [item.route for item in self._route_items if item.index == self._recording_route_index]
        if len(routes) != 1:
            self._error = "Select exactly one saved route before recording a sensor log."
            self._active_offline_subtab = "Record Sensor Data"
            return _NoSelection
        sensor_values = self._sensor_editor.values() if self._sensor_editor is not None else SENSOR_NOISE_PRESETS["Medium Noise"]
        if self._sensor_editor is not None:
            self._sensor_preset = self._sensor_editor.active_preset
        behavior_preset = "Balanced"
        behavior_values = BEHAVIOR_PRESETS[behavior_preset]
        config = OfflineRecordingConfig(
            selected_routes=tuple(routes),
            sensor_noise_config=sensor_noise_config_from_values(sensor_values, preset_name=self._sensor_preset),
            vehicle_behavior_config=driving_behavior_from_values(behavior_values, preset_name=behavior_preset),
            sensor_noise_preset=self._sensor_preset,
            vehicle_behavior_preset=behavior_preset,
            metadata={
                "startup_mode": "offline_localization_recording",
                "recording_driver": "ground_truth_controller",
            },
        )
        available_maps = [option.load_name or option.detail for option in self._options]
        missing = [
            route
            for route in routes
            if route.map_name and available_maps and not any(maps_compatible(candidate, route.map_name) for candidate in available_maps)
        ]
        if missing:
            self._error = f"Route map unavailable: {missing[0].name} -> {missing[0].map_name}."
            return _NoSelection
        return self._load_map_for_offline_recording(client, routes[0], config)

    def _run_offline_replay_from_setup(self) -> None:
        self._commit_filter_tune_editor()
        if self._selected_recorded_log_index is None or self._selected_recorded_log_index >= len(self._recorded_logs):
            self._error = "Select exactly one recorded sensor log."
            self._active_offline_subtab = "Test Setup"
            self._active_offline_setup_subtab = "Select Route"
            return
        selected_log = self._recorded_logs[self._selected_recorded_log_index]
        selected_filters = tuple(sorted(filter_id for filter_id in self._offline_filter_ids if filter_id != "raw_gnss"))
        if not selected_filters:
            self._error = "Select at least one replay filter. Raw GNSS is only the baseline."
            self._active_offline_subtab = "Test Setup"
            self._active_offline_setup_subtab = "Filters"
            return
        filter_tunes = self._included_offline_filter_tunes(selected_filters)
        self._offline_status_lines = [
            f"Running replay with custom tunes for: {', '.join(selected_filters)}",
            "Raw GNSS baseline is included automatically.",
        ]
        try:
            result = OfflineReplayRunner().run(
                OfflineReplayRequest(
                    sensor_log_paths=(selected_log.sensor_log_path,),
                    selected_filter_ids=selected_filters,
                    filter_tunes=filter_tunes,
                    include_raw_gnss_baseline=True,
                )
            )
        except Exception as exc:
            self._error = f"Offline replay failed: {exc}"
            self._offline_status_lines = [self._error]
            return
        self._error = ""
        raw = f"{result.raw_gnss_rmse_m:.3f} m" if result.raw_gnss_rmse_m is not None else "n/a"
        best = (
            f"{result.best_filter_id} ({result.best_position_rmse_m:.3f} m)"
            if result.best_filter_id is not None and result.best_position_rmse_m is not None
            else "n/a"
        )
        self._offline_status_lines = [
            f"Results: {result.output_folder}",
            f"Best eval RMSE filter: {best}",
            f"Raw GNSS eval RMSE: {raw}",
            f"Warm-up excluded: {result.warmup_excluded_s:.1f}s",
            f"Failures: {len(result.failures)}",
            f"Custom tunes used: {', '.join(selected_filters)}",
        ]

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

    def _load_map_for_offline_recording(
        self,
        client: object,
        first_route: SavedTestRoute,
        config: OfflineRecordingConfig,
    ) -> object:
        current_map = self._read_current_map(client)
        if maps_compatible(current_map, first_route.map_name):
            self._write_runtime_state(selected_load_name=None, active_map_name=current_map)
            return StartupMapSelection(
                selected_map_load_name=None,
                active_map_name=current_map,
                used_current_map=True,
                offline_recording_config=config,
            )
        if not first_route.map_name:
            self._error = "First selected route has no map metadata."
            return _NoSelection
        load_name = self._load_name_for_map(first_route.map_name)
        self._status = "Loading map"
        self._detail = f"Loading {display_map_name(load_name)} for offline recording..."
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
            self._error = f"Offline recording map load failed: {exc}"
            return _NoSelection
        self._write_runtime_state(selected_load_name=load_name, active_map_name=active_map_name)
        return StartupMapSelection(
            selected_map_load_name=load_name,
            active_map_name=active_map_name,
            used_current_map=False,
            offline_recording_config=config,
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

    def _draw_wrapped_text(
        self,
        text: object,
        rect: pygame.Rect,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        line_gap: int = 4,
    ) -> None:
        words = str(text).split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if not current or font.size(candidate)[0] <= rect.width:
                current = candidate
                continue
            lines.append(current)
            current = word
        if current:
            lines.append(current)
        y = rect.top
        line_height = font.get_linesize() + line_gap
        for line in lines:
            if y + font.get_linesize() > rect.bottom:
                break
            rendered = font.render(self._fit_text(line, font, rect.width), True, color)
            self._surface.blit(rendered, (rect.left, y))
            y += line_height

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
