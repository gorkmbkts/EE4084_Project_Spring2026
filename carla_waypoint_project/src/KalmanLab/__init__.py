"""KalmanLab plugin framework for localization filter benchmarking."""

from .filter_manager import FilterManager
from .filter_base import FilterPluginRecord
from .registry import discover_filters

__all__ = [
    "FilterManager",
    "FilterPluginRecord",
    "discover_filters",
]
