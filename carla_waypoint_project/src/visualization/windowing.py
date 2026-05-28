"""Shared pygame window setup helpers."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys

import pygame

from config.settings import DISPLAY


def display_flags_from_settings() -> int:
    """Return pygame display flags from centralized display settings."""
    flags = 0
    if DISPLAY.fullscreen and not DISPLAY.maximized:
        flags |= pygame.FULLSCREEN
    if DISPLAY.resizable or DISPLAY.maximized:
        flags |= pygame.RESIZABLE
    return flags


def create_display_surface(
    width: int = DISPLAY.width,
    height: int = DISPLAY.height,
    title: str = DISPLAY.title,
) -> pygame.Surface:
    """Create or update the shared pygame display surface."""
    pygame.init()
    surface = pygame.display.set_mode((width, height), display_flags_from_settings())
    pygame.display.set_caption(title)
    configure_native_window()
    return pygame.display.get_surface() or surface


def configure_native_window() -> None:
    """Apply Windows window styles and maximize without pygame fullscreen."""
    if sys.platform != "win32":
        return

    window_info = pygame.display.get_wm_info()
    hwnd = window_info.get("window")
    if not hwnd:
        return

    hwnd_handle = wintypes.HWND(hwnd)
    if DISPLAY.borderless:
        _make_window_borderless_resizable(hwnd_handle)
    if DISPLAY.maximized:
        ctypes.windll.user32.ShowWindow(hwnd_handle, 3)
    pygame.time.wait(50)
    pygame.event.pump()


def _make_window_borderless_resizable(hwnd: wintypes.HWND) -> None:
    user32 = ctypes.windll.user32
    gwl_style = -16
    swp_nosize = 0x0001
    swp_nomove = 0x0002
    swp_nozorder = 0x0004
    swp_framechanged = 0x0020
    ws_caption = 0x00C00000
    ws_sysmenu = 0x00080000
    ws_thickframe = 0x00040000
    ws_minimizebox = 0x00020000
    ws_maximizebox = 0x00010000

    style = user32.GetWindowLongW(hwnd, gwl_style)
    style &= ~ws_caption
    style |= ws_sysmenu | ws_thickframe | ws_minimizebox | ws_maximizebox
    user32.SetWindowLongW(hwnd, gwl_style, style)
    user32.SetWindowPos(
        hwnd,
        wintypes.HWND(0),
        0,
        0,
        0,
        0,
        swp_nomove | swp_nosize | swp_nozorder | swp_framechanged,
    )
