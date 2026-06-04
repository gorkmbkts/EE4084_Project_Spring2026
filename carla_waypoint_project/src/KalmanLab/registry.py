"""Discovery and validation for KalmanLab localization filter plugins."""

from __future__ import annotations

import importlib
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

from .filter_base import FilterPluginRecord, REQUIRED_FILTER_INFO_FIELDS

FILTERS_PACKAGE = "src.KalmanLab.filters"


def discover_filters(filters_dir: Path | None = None) -> list[FilterPluginRecord]:
    """Scan the filters package and return valid and invalid plugin records."""
    package_dir = filters_dir if filters_dir is not None else Path(__file__).resolve().parent / "filters"
    records: list[FilterPluginRecord] = []
    if not package_dir.exists():
        return records

    importlib.invalidate_caches()
    for path in sorted(package_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        records.append(_load_plugin(path))
    return records


def _load_plugin(path: Path) -> FilterPluginRecord:
    module_name = path.stem
    if module_name == "filter_template":
        return FilterPluginRecord(
            module_name=module_name,
            file_path=path,
            valid=False,
            filter_info={"id": module_name, "name": "Filter Template"},
            tune={},
            tune_specs=(),
            filter_class=None,
            error="Template file; copy and rename it to create a filter plugin.",
            template=True,
        )

    qualified_name = f"{FILTERS_PACKAGE}.{module_name}"
    try:
        module = importlib.import_module(qualified_name)
    except Exception as exc:
        return _invalid_record(path, module_name, f"Import failed: {exc}", exc)

    return _validate_module(path, module_name, module)


def _validate_module(path: Path, module_name: str, module: ModuleType) -> FilterPluginRecord:
    filter_info = getattr(module, "FILTER_INFO", None)
    tune = getattr(module, "TUNE", None)
    tune_specs = getattr(module, "TUNE_SPECS", ())
    filter_class = getattr(module, "Filter", None)

    if not isinstance(filter_info, dict):
        return _invalid_record(path, module_name, "FILTER_INFO must be a dictionary.")
    if not isinstance(tune, dict):
        return _invalid_record(path, module_name, "TUNE must be a dictionary.")
    if filter_class is None or not isinstance(filter_class, type):
        return _invalid_record(path, module_name, "Filter must be a class.")

    missing = [field for field in REQUIRED_FILTER_INFO_FIELDS if not filter_info.get(field)]
    if missing:
        return _invalid_record(path, module_name, f"FILTER_INFO missing fields: {', '.join(missing)}.")

    normalized_info: dict[str, Any] = dict(filter_info)
    safe_for_autonomous = bool(normalized_info.get("safe_for_autonomous_control", True))
    normalized_info["safe_for_autonomous_control"] = safe_for_autonomous
    normalized_info.setdefault("active_tracking_supported", hasattr(filter_class, "process_control"))
    # For older filters, benchmark selection followed autonomous safety.  New
    # experimental filters can opt in explicitly without changing UI code.
    normalized_info.setdefault("benchmark_selectable", safe_for_autonomous)
    normalized_info.setdefault("experimental", False)
    normalized_info.setdefault("requires_raw_imu", False)
    normalized_info.setdefault("motion_info_fields", ())
    normalized_info.setdefault("model_type", str(normalized_info.get("id") or module_name))
    return FilterPluginRecord(
        module_name=module_name,
        file_path=path,
        valid=True,
        filter_info=normalized_info,
        tune=dict(tune),
        tune_specs=_normalize_tune_specs(tune_specs),
        filter_class=filter_class,
    )


def _normalize_tune_specs(value: object) -> tuple[Any, ...]:
    """Return optional tune specs without making them mandatory plugin metadata."""
    if value is None:
        return ()
    if not isinstance(value, (tuple, list)):
        return ()

    specs = []
    for spec in value:
        if hasattr(spec, "key") and hasattr(spec, "clamp"):
            specs.append(spec)
    return tuple(specs)


def _invalid_record(
    path: Path,
    module_name: str,
    message: str,
    exc: Exception | None = None,
) -> FilterPluginRecord:
    if exc is not None:
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        if detail and detail not in message:
            message = f"{message} ({detail})"
    return FilterPluginRecord(
        module_name=module_name,
        file_path=path,
        valid=False,
        filter_info={"id": module_name, "name": module_name},
        tune={},
        tune_specs=(),
        filter_class=None,
        error=message,
    )
