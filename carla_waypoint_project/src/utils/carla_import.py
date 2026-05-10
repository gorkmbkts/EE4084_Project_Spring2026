"""Helpers for importing CARLA in local project environments."""

from __future__ import annotations

import glob
import importlib
import sys
from pathlib import Path
from types import ModuleType


def _candidate_carla_roots() -> list[Path]:
    project_root = Path(__file__).resolve().parents[2]
    return [
        project_root / "CARLA_0.9.16",
        project_root.parent / "CARLA_0.9.16",
    ]


def add_carla_egg_to_path() -> bool:
    """Try to locate and append a CARLA wheel or egg path to ``sys.path``."""
    dist_candidates = [root / "PythonAPI" / "carla" / "dist" for root in _candidate_carla_roots()]

    for dist_dir in dist_candidates:
        packages = sorted(glob.glob(str(dist_dir / "carla-*.egg")))
        packages.extend(sorted(glob.glob(str(dist_dir / "carla-*.whl"))))
        if not packages:
            continue

        package_path = packages[0]
        if package_path not in sys.path:
            sys.path.append(package_path)
        return True

    return False


def add_carla_python_api_to_path() -> bool:
    """Add CARLA's PythonAPI/carla folder so ``agents.*`` imports work."""
    added = False
    for carla_root in _candidate_carla_roots():
        api_path = carla_root / "PythonAPI" / "carla"
        if not api_path.exists():
            continue
        api_path_str = str(api_path)
        if api_path_str not in sys.path:
            sys.path.append(api_path_str)
        added = True
    return added


def ensure_carla_import() -> ModuleType:
    """Import ``carla`` with an egg-path fallback."""
    try:
        import carla  # type: ignore
    except ModuleNotFoundError:
        add_carla_egg_to_path()
        add_carla_python_api_to_path()
        import carla  # type: ignore
    return carla


def ensure_carla_agents_import(module_name: str) -> ModuleType:
    """Import a CARLA agents module with robust local PythonAPI path handling."""
    add_carla_python_api_to_path()
    return importlib.import_module(module_name)
