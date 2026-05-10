"""Lightweight simulation loop helpers."""

from __future__ import annotations

import pygame

from config.settings import DISPLAY, SIMULATION


class SimulationClock:
    """Wrap pygame clock behavior for a target frame rate."""

    def __init__(self, target_fps: int = DISPLAY.fps) -> None:
        self._clock = pygame.time.Clock()
        self._target_fps = target_fps
        self._last_frame_dt_seconds = 0.0

    @property
    def target_fps(self) -> int:
        return self._target_fps

    @property
    def fixed_delta_seconds(self) -> float:
        return SIMULATION.fixed_delta_seconds

    @property
    def last_frame_dt_seconds(self) -> float:
        return self._last_frame_dt_seconds

    def tick_pygame(self) -> float:
        """Sleep if needed to keep the pygame frame rate and return wall dt."""
        elapsed_ms = self._clock.tick(self._target_fps)
        self._last_frame_dt_seconds = elapsed_ms * 0.001
        return self._last_frame_dt_seconds
