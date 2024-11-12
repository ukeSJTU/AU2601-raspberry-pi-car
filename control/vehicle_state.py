# control/vehicle_state.py

from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, Optional, Tuple
import time
from scipy.spatial.transform import Rotation


@dataclass
class VehicleState:
    """Current state of the vehicle."""

    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]
    orientation: np.ndarray  # [roll, pitch, yaw]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    timestamp: float

    @property
    def speed(self) -> float:
        """Get current speed magnitude."""
        return float(np.linalg.norm(self.velocity))

    @property
    def heading(self) -> float:
        """Get current heading angle in radians."""
        return float(self.orientation[2])  # yaw angle


class StateEstimator:
    """Estimates vehicle state from sensors using Kalman filtering."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.last_state: Optional[VehicleState] = None

        # Kalman filter state: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        self.state = np.zeros(12)
        self.covariance = np.eye(12) * 1000  # Initial uncertainty

        # Process noise (tune these based on vehicle dynamics)
        self.Q = np.eye(12)
        self.Q[0:3, 0:3] *= 0.1  # position
        self.Q[3:6, 3:6] *= 1.0  # velocity
        self.Q[6:9, 6:9] *= 0.1  # orientation
        self.Q[9:12, 9:12] *= 1.0  # angular velocity

        # Measurement noise (tune based on sensor characteristics)
        self.R_gps = np.eye(3) * 2.0  # GPS position noise
        self.R_imu = np.eye(6) * 0.1  # IMU noise (acceleration & angular velocity)

        self.last_update = time.time()

    def predict(self, dt: float) -> None:
        """Predict step of the Kalman filter."""
        # State transition matrix
        F = np.eye(12)
        F[0:3, 3:6] = np.eye(3) * dt  # position += velocity * dt
        F[6:9, 9:12] = np.eye(3) * dt  # orientation += angular_velocity * dt

        # Predict state
        self.state = F @ self.state

        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.Q * dt

    def update_gps(self, position: np.ndarray) -> None:
        """Update state with GPS measurement."""
        H = np.zeros((3, 12))
        H[0:3, 0:3] = np.eye(3)  # Measure position

        self._update(position, H, self.R_gps)

    def update_imu(
        self, acceleration: np.ndarray, angular_velocity: np.ndarray
    ) -> None:
        """Update state with IMU measurement."""
        measurement = np.concatenate([acceleration, angular_velocity])

        H = np.zeros((6, 12))
        # Convert body acceleration to world frame
        R = Rotation.from_euler("xyz", self.state[6:9]).as_matrix()
        H[0:3, 3:6] = R  # Measure acceleration in world frame
        H[3:6, 9:12] = np.eye(3)  # Measure angular velocity

        self._update(measurement, H, self.R_imu)

    def _update(self, measurement: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Kalman filter update step."""
        # Calculate Kalman gain
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        innovation = measurement - H @ self.state
        self.state = self.state + K @ innovation

        # Update covariance
        self.covariance = (np.eye(12) - K @ H) @ self.covariance

    def update(self, sensor_data: Dict[str, Any]) -> VehicleState:
        """
        Update state estimate with new sensor data.

        Args:
            sensor_data: Dictionary containing sensor measurements
                - 'gps': [x, y, z] position
                - 'imu': {'acceleration': [ax, ay, az],
                         'angular_velocity': [wx, wy, wz]}
                - 'wheel_speeds': [fl, fr, rl, rr] wheel speeds
                - timestamp: measurement timestamp
        """
        current_time = sensor_data.get("timestamp", time.time())
        dt = current_time - self.last_update

        # Prediction step
        self.predict(dt)

        # Update with measurements
        if "gps" in sensor_data:
            self.update_gps(np.array(sensor_data["gps"]))

        if "imu" in sensor_data:
            imu_data = sensor_data["imu"]
            self.update_imu(
                np.array(imu_data["acceleration"]),
                np.array(imu_data["angular_velocity"]),
            )

        # Create new vehicle state
        new_state = VehicleState(
            position=self.state[0:3].copy(),
            velocity=self.state[3:6].copy(),
            acceleration=self.state[3:6].copy() / dt,  # approximate
            orientation=self.state[6:9].copy(),
            angular_velocity=self.state[9:12].copy(),
            timestamp=current_time,
        )

        self.last_state = new_state
        self.last_update = current_time

        return new_state

    def get_state(self) -> Optional[VehicleState]:
        """Get current vehicle state."""
        return self.last_state

    def reset(self) -> None:
        """Reset the estimator."""
        self.state = np.zeros(12)
        self.covariance = np.eye(12) * 1000
        self.last_state = None
        self.last_update = time.time()

    def extrapolate_state(self, dt: float) -> VehicleState:
        """
        Extrapolate state forward in time without measurements.
        Useful for prediction and planning.
        """
        if self.last_state is None:
            raise ValueError("No state to extrapolate from")

        # Simple linear extrapolation
        new_position = self.last_state.position + self.last_state.velocity * dt
        new_orientation = (
            self.last_state.orientation + self.last_state.angular_velocity * dt
        )

        return VehicleState(
            position=new_position,
            velocity=self.last_state.velocity,
            acceleration=self.last_state.acceleration,
            orientation=new_orientation,
            angular_velocity=self.last_state.angular_velocity,
            timestamp=self.last_state.timestamp + dt,
        )
