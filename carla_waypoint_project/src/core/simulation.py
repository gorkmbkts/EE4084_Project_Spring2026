"""Lightweight simulation loop helpers."""

from __future__ import annotations

import pygame

from config.settings import DISPLAY


class SimulationClock:
    """Wrap pygame clock behavior for a target frame rate."""

    def __init__(self, target_fps: int = DISPLAY.fps) -> None:
        self._clock = pygame.time.Clock()
        self._target_fps = target_fps

    def tick(self) -> None:
        """Sleep if needed to keep the target FPS."""
        self._clock.tick(self._target_fps)

