"""Helpers for comparing GNSS geodetic readings with CARLA local coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from src.localization.state_estimator import EgoState
from src.sensors.gnss_sensor import GnssMeasurement
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()

METERS_PER_DEGREE_LAT = 111_320.0


@dataclass(frozen=True)
class GnssDiagnostics:
    """GNSS reading projected into CARLA local coordinates and compared to GT."""

    local_x: float
    local_y: float
    dx_m: float
    dy_m: float
    dz_m: float
    horizontal_error_m: float


class GnssLocalProjector:
    """Approximate inverse of CARLA map geolocation near the map origin."""

    def __init__(self, world_map: "carla.Map") -> None:
        self._world_map = world_map
        self._origin_geo = world_map.transform_to_geolocation(carla.Location(x=0.0, y=0.0, z=0.0))
        self._meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(self._origin_geo.latitude))

        x_geo = world_map.transform_to_geolocation(carla.Location(x=1.0, y=0.0, z=0.0))
        y_geo = world_map.transform_to_geolocation(carla.Location(x=0.0, y=1.0, z=0.0))
        self._basis_x = self._geo_meter_delta(x_geo.latitude, x_geo.longitude)
        self._basis_y = self._geo_meter_delta(y_geo.latitude, y_geo.longitude)

        self._determinant = (
            self._basis_x[0] * self._basis_y[1] - self._basis_y[0] * self._basis_x[1]
        )

    def to_local_xy(self, latitude: float, longitude: float) -> Optional[tuple[float, float]]:
        """Convert geodetic lat/lon to approximate CARLA local x/y meters."""
        if abs(self._determinant) < 1.0e-9:
            return None

        east_m, north_m = self._geo_meter_delta(latitude, longitude)
        local_x = (east_m * self._basis_y[1] - self._basis_y[0] * north_m) / self._determinant
        local_y = (self._basis_x[0] * north_m - east_m * self._basis_x[1]) / self._determinant
        return local_x, local_y

    def diagnostics(
        self,
        gnss: Optional[GnssMeasurement],
        state: Optional[EgoState],
    ) -> Optional[GnssDiagnostics]:
        """Build GNSS-vs-ground-truth diagnostics if both values are available."""
        if gnss is None or state is None:
            return None

        local_xy = self.to_local_xy(gnss.latitude, gnss.longitude)
        if local_xy is None:
            return None

        local_x, local_y = local_xy
        gt_location = carla.Location(x=state.x, y=state.y, z=state.z)
        gt_geo = self._world_map.transform_to_geolocation(gt_location)
        dx_m = local_x - state.x
        dy_m = local_y - state.y
        dz_m = gnss.altitude - float(gt_geo.altitude)
        return GnssDiagnostics(
            local_x=float(local_x),
            local_y=float(local_y),
            dx_m=float(dx_m),
            dy_m=float(dy_m),
            dz_m=float(dz_m),
            horizontal_error_m=float(math.hypot(dx_m, dy_m)),
        )

    def _geo_meter_delta(self, latitude: float, longitude: float) -> tuple[float, float]:
        east_m = (float(longitude) - float(self._origin_geo.longitude)) * self._meters_per_degree_lon
        north_m = (float(latitude) - float(self._origin_geo.latitude)) * METERS_PER_DEGREE_LAT
        return east_m, north_m
