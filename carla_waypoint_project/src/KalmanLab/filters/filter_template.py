"""Template for adding a new KalmanLab localization filter.

Copy this file into ``src/KalmanLab/filters/`` with a new filename. The
registry scans Python files in that directory. A plugin is valid when it
defines:

``FILTER_INFO``
    Dictionary with these required fields:
    ``id``, ``name``, ``type``, ``state_vector``, ``process_model``,
    ``measurement_model``, and ``description``.

``TUNE``
    Dictionary containing all filter-specific tuning parameters. Do not put
    filter tuning in ``config/settings.py``.

``class Filter``
    Class constructed as ``Filter(gnss_projector)``. It should expose:
    ``reset()``, ``process_imu(imu)``, ``process_gnss(gnss)``,
    ``get_state()``, and ``get_diagnostics()``.

``get_state()`` should return ``src.core.vehicle_state.VehicleState`` or
``None`` until initialized. Diagnostics should be a small dictionary. Include
covariance diagonals, innovation vectors, NIS, or other filter-specific details
when available.

Put control-relevant estimated quantities such as yaw rate, acceleration,
curvature, covariance, model type, and source filter id on ``VehicleState``.
Keep only verbose debug details in diagnostics.

Set ``"safe_for_autonomous_control": False`` in ``FILTER_INFO`` for filters
that should be benchmark baselines only, such as raw GNSS.
"""

from __future__ import annotations


FILTER_INFO = {
    "id": "my_filter",
    "name": "My Filter",
    "type": "EKF/UKF/Particle Filter/etc.",
    "state_vector": "[document your state]^T",
    "process_model": "Describe prediction model",
    "measurement_model": "Describe fused sensors",
    "description": "One sentence explaining the filter.",
    "model_type": "MY_MODEL",
    "provided_state_fields": ("x", "y", "z", "yaw", "speed", "timestamp"),
    "safe_for_autonomous_control": True,
}


TUNE = {
    "example_parameter": 1.0,
}


class Filter:
    def __init__(self, gnss_projector):
        self._gnss_projector = gnss_projector
        self._latest_state = None
        self._latest_gnss_local = None
        self.initialized = False

    @property
    def latest_gnss_local(self):
        return self._latest_gnss_local

    def reset(self):
        self._latest_state = None
        self._latest_gnss_local = None
        self.initialized = False

    def process_imu(self, imu):
        return self._latest_state

    def process_gnss(self, gnss):
        local = self._gnss_projector.project(gnss)
        if local is None:
            return self._latest_state
        self._latest_gnss_local = local
        # Build and assign a VehicleState here.
        self.initialized = True
        return self._latest_state

    def get_state(self):
        return self._latest_state

    def get_diagnostics(self):
        return {"initialized": self.initialized}
