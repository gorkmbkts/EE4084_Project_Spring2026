"""IMU sensor wrapper for live pre-fusion monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional

from config.settings import IMU
from src.evaluation.benchmark_config import SensorNoiseConfig
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


@dataclass(frozen=True)
class ImuMeasurement:
    """Latest IMU reading from CARLA."""

    accelerometer: tuple[float, float, float]
    gyroscope: tuple[float, float, float]
    compass: float
    frame: int
    timestamp: float


class ImuSensor:
    """Manage a CARLA IMU sensor actor and its latest measurement."""

    def __init__(self, world: "carla.World", blueprint_library: "carla.BlueprintLibrary") -> None:
        self._world = world
        self._blueprint_library = blueprint_library
        self._actor: Optional["carla.Sensor"] = None
        self._attach_to: Optional["carla.Actor"] = None
        self._config = SensorNoiseConfig()
        self._latest_measurement: Optional[ImuMeasurement] = None
        self._lock = Lock()

    @property
    def actor(self) -> "carla.Sensor":
        if self._actor is None:
            raise RuntimeError("IMU actor is not spawned yet.")
        return self._actor

    def spawn(self, attach_to: "carla.Actor") -> None:
        """Spawn the IMU sensor and start listening for measurements."""
        self._attach_to = attach_to
        imu_bp = self._blueprint_library.find(IMU.blueprint_id)
        config = self._config
        attributes = {
            "sensor_tick": config.imu_sensor_tick,
            "noise_accel_stddev_x": config.imu_noise_accel_stddev_x,
            "noise_accel_stddev_y": config.imu_noise_accel_stddev_y,
            "noise_accel_stddev_z": config.imu_noise_accel_stddev_z,
            "noise_gyro_stddev_x": config.imu_noise_gyro_stddev_x,
            "noise_gyro_stddev_y": config.imu_noise_gyro_stddev_y,
            "noise_gyro_stddev_z": config.imu_noise_gyro_stddev_z,
            "noise_gyro_bias_x": config.imu_noise_gyro_bias_x,
            "noise_gyro_bias_y": config.imu_noise_gyro_bias_y,
            "noise_gyro_bias_z": config.imu_noise_gyro_bias_z,
            "noise_seed": config.imu_noise_seed,
        }
        for name, value in attributes.items():
            if imu_bp.has_attribute(name):
                imu_bp.set_attribute(name, str(value))

        transform = carla.Transform(
            carla.Location(
                x=IMU.relative_x,
                y=IMU.relative_y,
                z=IMU.relative_z,
            )
        )
        self._actor = self._world.spawn_actor(imu_bp, transform, attach_to=attach_to)
        self._actor.listen(self._on_measurement)

    @property
    def config(self) -> SensorNoiseConfig:
        return self._config

    def apply_config(self, config: SensorNoiseConfig, respawn: bool = True) -> None:
        """Apply IMU blueprint noise settings, respawning when the actor is live."""
        self._config = config
        if not respawn or self._attach_to is None:
            return
        self.destroy(clear_attachment=False)
        with self._lock:
            self._latest_measurement = None
        self.spawn(self._attach_to)

    def _on_measurement(self, measurement: "carla.IMUMeasurement") -> None:
        latest = ImuMeasurement(
            accelerometer=(
                float(measurement.accelerometer.x),
                float(measurement.accelerometer.y),
                float(measurement.accelerometer.z),
            ),
            gyroscope=(
                float(measurement.gyroscope.x),
                float(measurement.gyroscope.y),
                float(measurement.gyroscope.z),
            ),
            compass=float(measurement.compass),
            frame=int(measurement.frame),
            timestamp=float(measurement.timestamp),
        )
        with self._lock:
            self._latest_measurement = latest

    def get_latest_measurement(self) -> Optional[ImuMeasurement]:
        """Return the latest IMU measurement, if one has arrived."""
        with self._lock:
            return self._latest_measurement

    def destroy(self, clear_attachment: bool = True) -> None:
        """Stop and destroy IMU actor safely."""
        if self._actor is not None:
            self._actor.stop()
            self._actor.destroy()
            self._actor = None
        if clear_attachment:
            self._attach_to = None
