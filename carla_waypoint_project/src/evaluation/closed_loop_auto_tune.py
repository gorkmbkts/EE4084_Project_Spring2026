"""Deferred closed-loop auto-tune session handoff types.

The first auto-tuning redesign batch intentionally does not run closed-loop
auto tuning.  This explicit handoff object is the future contract between the
startup UI candidate-generation step and the in-app closed-loop validation
runner, avoiding hidden global state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PendingClosedLoopAutoTuneSession:
    selected_filter: str
    tracking_mode: str
    offline_log_paths: tuple[str, ...]
    noise_signature: str
    validation_route_name: str
    validation_route_map: str
    validation_route_id: str
    sensor_config: dict[str, object]
    vehicle_behavior_config: dict[str, object]
    actuator_realism_config: dict[str, object]
    trial_count: int
    finalist_count: int
    strategy: str
    output_root: str
    created_at: str = ""
    handoff_path: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
