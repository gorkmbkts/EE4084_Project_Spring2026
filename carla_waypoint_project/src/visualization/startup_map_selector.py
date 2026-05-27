"""Pre-dashboard pygame UI for CARLA startup status and map selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Optional

import pygame

from config.settings import CARLA, DISPLAY
from src.utils.map_names import display_map_name, maps_compatible, normalize_map_name


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


@dataclass(frozen=True)
class _MapOption:
    load_name: Optional[str]
    display_name: str
    detail: str
    guaranteed: bool
    is_current: bool = False


class StartupMapSelector:
    """Small responsive pygame screen shown before dashboard initialization."""

    def __init__(self, width: int = 1000, height: int = 720) -> None:
        pygame.init()
        self._surface = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("KalmanLab CARLA Startup")
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
        self._row_rects: dict[int, pygame.Rect] = {}
        self._project_root = Path(__file__).resolve().parents[2]
        self._runtime_state_path = self._project_root / "config" / "runtime_state.json"

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

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.VIDEORESIZE:
                    self._resize(event.w, event.h)
                    continue
                if event.type == pygame.KEYDOWN:
                    result = self._handle_key_down(event, client)
                    if result is not _NoSelection:
                        return result
                elif event.type == pygame.MOUSEWHEEL:
                    self._scroll_by(-event.y * 3)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_left_click(event.pos)

            self._draw(status_only=False)
            self._clock.tick(30)

        return None

    def _handle_key_down(self, event: pygame.event.Event, client: object) -> object:
        if event.key == pygame.K_ESCAPE:
            return None
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

    def _handle_left_click(self, position: tuple[int, int]) -> None:
        for index, rect in self._row_rects.items():
            if rect.collidepoint(position):
                self._selected_index = index
                self._ensure_selection_visible()
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
        _left, top, _width, height = self._list_geometry()
        return max(1, height // 42)

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
        self._surface = pygame.display.set_mode((width, height), pygame.RESIZABLE)

    def _ensure_display_ready(self) -> None:
        if not pygame.get_init():
            pygame.init()
            self._clock = pygame.time.Clock()
        if not pygame.font.get_init():
            pygame.font.init()
        if not pygame.display.get_init():
            pygame.display.init()
        if pygame.display.get_surface() is None:
            self._surface = pygame.display.set_mode((1000, 720), pygame.RESIZABLE)
            pygame.display.set_caption("KalmanLab CARLA Startup")
            self._init_fonts()

    def _init_fonts(self) -> None:
        self._title_font = pygame.font.SysFont("consolas", 26, bold=True)
        self._subtitle_font = pygame.font.SysFont("consolas", 18, bold=True)
        self._font = pygame.font.SysFont("consolas", 16)
        self._small_font = pygame.font.SysFont("consolas", 14)

    def _draw(self, status_only: bool) -> None:
        width, height = self._surface.get_size()
        self._surface.fill((12, 15, 20))
        margin = 32

        self._draw_text(
            "KalmanLab CARLA Localization Dashboard",
            (margin, 24),
            self._title_font,
            (239, 243, 250),
            max_width=width - 2 * margin,
        )
        self._draw_status_panel(pygame.Rect(margin, 76, width - 2 * margin, 146))

        if status_only:
            self._draw_text(
                "Startup is preparing CARLA. Press ESC to quit safely.",
                (margin, 244),
                self._font,
                (185, 194, 210),
                max_width=width - 2 * margin,
            )
            pygame.display.flip()
            return

        self._draw_selector_header(pygame.Rect(margin, 244, width - 2 * margin, 70))
        self._draw_map_list()
        self._draw_controls(pygame.Rect(margin, height - 64, width - 2 * margin, 40))
        pygame.display.flip()

    def _draw_status_panel(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self._surface, (23, 28, 36), rect, border_radius=6)
        pygame.draw.rect(self._surface, (79, 91, 112), rect, width=1, border_radius=6)

        y = rect.top + 14
        self._draw_text("CARLA server", (rect.left + 16, y), self._subtitle_font, (235, 238, 244), rect.width - 32)
        y += 34
        self._draw_text(f"Status: {self._status}", (rect.left + 16, y), self._font, (119, 220, 156), rect.width - 32)
        y += 26
        exe_text = self._executable_path or "not detected yet"
        self._draw_text(f"Executable: {exe_text}", (rect.left + 16, y), self._small_font, (190, 198, 212), rect.width - 32)
        y += 24
        current_map = display_map_name(self._current_map_name) if self._current_map_name else "unread"
        self._draw_text(f"Currently loaded map: {current_map}", (rect.left + 16, y), self._small_font, (190, 198, 212), rect.width - 32)
        y += 24
        message = self._error or self._detail
        color = (255, 198, 94) if self._error else (170, 178, 194)
        self._draw_text(message, (rect.left + 16, y), self._small_font, color, rect.width - 32)

    def _draw_selector_header(self, rect: pygame.Rect) -> None:
        available = f"{self._available_count} available map(s)"
        normalized = normalize_map_name(self._current_map_name) or "unknown"
        self._draw_text("Startup map selection", (rect.left, rect.top), self._subtitle_font, (235, 238, 244), rect.width)
        self._draw_text(
            f"{available} from CARLA RPC. Active map id if using current: {normalized}",
            (rect.left, rect.top + 30),
            self._font,
            (178, 187, 204),
            rect.width,
        )

    def _draw_map_list(self) -> None:
        left, top, width, height = self._list_geometry()
        rect = pygame.Rect(left, top, width, height)
        pygame.draw.rect(self._surface, (17, 21, 28), rect, border_radius=6)
        pygame.draw.rect(self._surface, (77, 89, 111), rect, width=1, border_radius=6)
        self._row_rects.clear()

        if not self._options:
            self._draw_text("No map options available.", (left + 16, top + 16), self._font, (230, 230, 230), width - 32)
            return

        row_h = 42
        visible = self._visible_row_count()
        end_index = min(len(self._options), self._scroll_offset + visible)
        y = top + 8
        for index in range(self._scroll_offset, end_index):
            option = self._options[index]
            row = pygame.Rect(left + 8, y, width - 16, row_h - 4)
            self._row_rects[index] = row
            selected = index == self._selected_index
            row_color = (42, 63, 84) if selected else (24, 29, 38)
            pygame.draw.rect(self._surface, row_color, row, border_radius=4)
            if selected:
                pygame.draw.rect(self._surface, (116, 188, 255), row, width=1, border_radius=4)

            title_color = (248, 250, 253) if selected else (222, 228, 238)
            detail_color = (182, 193, 211) if option.guaranteed else (255, 198, 94)
            self._draw_text(option.display_name, (row.left + 12, row.top + 5), self._font, title_color, row.width - 24)
            self._draw_text(option.detail, (row.left + 12, row.top + 23), self._small_font, detail_color, row.width - 24)
            y += row_h

        if len(self._options) > visible:
            scroll_text = f"{self._scroll_offset + 1}-{end_index} / {len(self._options)}"
            self._draw_text(scroll_text, (rect.right - 130, rect.bottom - 24), self._small_font, (160, 170, 188), 120)

    def _draw_controls(self, rect: pygame.Rect) -> None:
        controls = "Up/Down: move | Mouse wheel: scroll | Click: select | Enter: load | U: use current | R: refresh | ESC: quit"
        self._draw_text(controls, (rect.left, rect.top), self._small_font, (192, 201, 216), rect.width)

    def _list_geometry(self) -> tuple[int, int, int, int]:
        width, height = self._surface.get_size()
        margin = 32
        top = 320
        bottom_margin = 84
        return margin, top, width - 2 * margin, max(120, height - top - bottom_margin)

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
