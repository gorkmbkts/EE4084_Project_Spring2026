"""GNSS geodetic-to-local conversion and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, TYPE_CHECKING

from src.sensors.gnss_sensor import GnssMeasurement
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()

if TYPE_CHECKING:
    from src.localization.state_estimator import EgoState

METERS_PER_DEGREE_LAT = 111_320.0


@dataclass(frozen=True)
class LocalGnssMeasurement:
    """GNSS fix projected into the CARLA local map frame."""

    x: float
    y: float
    z: float
    latitude: float
    longitude: float
    altitude: float
    frame: int
    timestamp: float


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
        self._origin_geo = None
        self._meters_per_degree_lon = 0.0
        self._basis_x = (0.0, 0.0)
        self._basis_y = (0.0, 0.0)
        self._determinant = 0.0
        self._projection_error: Optional[str] = None

        try:
            self._origin_geo = world_map.transform_to_geolocation(carla.Location(x=0.0, y=0.0, z=0.0))
            self._meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(self._origin_geo.latitude))

            x_geo = world_map.transform_to_geolocation(carla.Location(x=1.0, y=0.0, z=0.0))
            y_geo = world_map.transform_to_geolocation(carla.Location(x=0.0, y=1.0, z=0.0))
            self._basis_x = self._geo_meter_delta(x_geo.latitude, x_geo.longitude)
            self._basis_y = self._geo_meter_delta(y_geo.latitude, y_geo.longitude)

            self._determinant = (
                self._basis_x[0] * self._basis_y[1] - self._basis_y[0] * self._basis_x[1]
            )
            if not math.isfinite(self._determinant) or abs(self._determinant) < 1.0e-9:
                self._projection_error = "Invalid GNSS georeference basis"
        except Exception as exc:
            self._projection_error = f"GNSS georeference unavailable: {exc}"

    @property
    def available(self) -> bool:
        return self._projection_error is None and self._origin_geo is not None

    @property
    def projection_error(self) -> Optional[str]:
        return self._projection_error

    def project(self, gnss: Optional[GnssMeasurement]) -> Optional[LocalGnssMeasurement]:
        """Convert a raw GNSS reading to local map x/y/z coordinates."""
        if gnss is None or not self.available or self._origin_geo is None:
            return None

        local_xy = self.to_local_xy(gnss.latitude, gnss.longitude)
        if local_xy is None:
            return None

        local_x, local_y = local_xy
        local_z = float(gnss.altitude) - float(self._origin_geo.altitude)
        return LocalGnssMeasurement(
            x=float(local_x),
            y=float(local_y),
            z=float(local_z),
            latitude=float(gnss.latitude),
            longitude=float(gnss.longitude),
            altitude=float(gnss.altitude),
            frame=int(gnss.frame),
            timestamp=float(gnss.timestamp),
        )

    def to_local_xy(self, latitude: float, longitude: float) -> Optional[tuple[float, float]]:
        """Convert geodetic lat/lon to approximate CARLA local x/y meters."""
        if not self.available:
            return None
        if not math.isfinite(self._determinant) or abs(self._determinant) < 1.0e-9:
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

        local = self.project(gnss)
        if local is None:
            return None

        gt_location = carla.Location(x=state.x, y=state.y, z=state.z)
        try:
            gt_geo = self._world_map.transform_to_geolocation(gt_location)
        except Exception:
            return None
        dx_m = local.x - state.x
        dy_m = local.y - state.y
        dz_m = gnss.altitude - float(gt_geo.altitude)
        return GnssDiagnostics(
            local_x=float(local.x),
            local_y=float(local.y),
            dx_m=float(dx_m),
            dy_m=float(dy_m),
            dz_m=float(dz_m),
            horizontal_error_m=float(math.hypot(dx_m, dy_m)),
        )

    def _geo_meter_delta(self, latitude: float, longitude: float) -> tuple[float, float]:
        if self._origin_geo is None:
            return 0.0, 0.0
        east_m = (float(longitude) - float(self._origin_geo.longitude)) * self._meters_per_degree_lon
        north_m = (float(latitude) - float(self._origin_geo.latitude)) * METERS_PER_DEGREE_LAT
        return east_m, north_m
