# control/vehicle_state.py
from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class VehicleState:
    """Current state of the vehicle."""

    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]
    orientation: np.ndarray  # [roll, pitch, yaw]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    timestamp: float


class StateEstimator:
    """Estimates vehicle state from sensors."""

    def __init__(self):
        self.last_state: Optional[VehicleState] = None

    def update(self, sensor_data: Dict[str, Any]) -> VehicleState:
        """Update state estimate with new sensor data."""
        pass
