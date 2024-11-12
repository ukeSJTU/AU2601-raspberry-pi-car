# control/trajectory_generator.py

from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline
from planning.path_planner import PathPlan


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""

    x: float  # x position
    y: float  # y position
    theta: float  # heading angle
    v: float  # linear velocity
    w: float  # angular velocity
    t: float  # timestamp


class TrajectoryGenerator:
    """Generates smooth trajectory from path plan."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Motion constraints
        self.max_velocity = self.config.get("max_velocity", 50.0)  # km/h
        self.max_acceleration = self.config.get("max_acceleration", 2.0)  # m/s^2
        self.max_deceleration = self.config.get("max_deceleration", -4.0)  # m/s^2
        self.max_angular_velocity = self.config.get(
            "max_angular_velocity", 1.0
        )  # rad/s

        # Temporal parameters
        self.timestep = self.config.get("timestep", 0.1)  # seconds
        self.planning_horizon = self.config.get("planning_horizon", 3.0)  # seconds

    def generate(self, path: PathPlan) -> List[TrajectoryPoint]:
        """
        Generate trajectory points from path plan.

        Args:
            path: Path plan with waypoints, speeds, and curvatures

        Returns:
            List of trajectory points
        """
        # Validate input
        if not path.waypoints or len(path.waypoints) < 2:
            return []

        # Generate time vector
        total_distance = self._calculate_path_length(path.waypoints)
        average_speed = np.mean(path.speeds)
        estimated_duration = total_distance / max(0.1, average_speed)
        num_points = int(min(estimated_duration, self.planning_horizon) / self.timestep)

        # Generate temporal trajectory
        trajectory = self._generate_temporal_trajectory(path, num_points)

        # Apply dynamic constraints
        trajectory = self._apply_dynamic_constraints(trajectory)

        return trajectory

    def _calculate_path_length(self, waypoints: List[tuple]) -> float:
        """Calculate total path length."""
        length = 0.0
        for i in range(len(waypoints) - 1):
            dx = waypoints[i + 1][0] - waypoints[i][0]
            dy = waypoints[i + 1][1] - waypoints[i][1]
            length += np.sqrt(dx * dx + dy * dy)
        return length

    def _generate_temporal_trajectory(
        self, path: PathPlan, num_points: int
    ) -> List[TrajectoryPoint]:
        """Generate initial trajectory with timing."""
        # Extract coordinates
        x_coords = [p[0] for p in path.waypoints]
        y_coords = [p[1] for p in path.waypoints]

        # Create path parameter (0 to 1)
        path_param = np.linspace(0, 1, len(path.waypoints))

        # Create splines
        x_spline = CubicSpline(path_param, x_coords)
        y_spline = CubicSpline(path_param, y_coords)

        # Generate finer trajectory
        trajectory = []
        times = np.linspace(0, self.planning_horizon, num_points)
        params = np.linspace(0, 1, num_points)

        for t, s in zip(times, params):
            # Position
            x = float(x_spline(s))
            y = float(y_spline(s))

            # Derivatives
            dx = float(x_spline.derivative()(s))
            dy = float(y_spline.derivative()(s))

            # Calculate heading and velocities
            theta = np.arctan2(dy, dx)
            v = np.sqrt(dx * dx + dy * dy)  # linear velocity

            # Angular velocity from path curvature
            if len(path.curvatures) > 1:
                # Interpolate curvature
                curvature = np.interp(s, path_param, path.curvatures)
                w = v * curvature
            else:
                w = 0.0

            trajectory.append(TrajectoryPoint(x=x, y=y, theta=theta, v=v, w=w, t=t))

        return trajectory

    def _apply_dynamic_constraints(
        self, trajectory: List[TrajectoryPoint]
    ) -> List[TrajectoryPoint]:
        """Apply velocity and acceleration constraints."""
        if len(trajectory) < 2:
            return trajectory

        # Convert to numpy arrays for vectorized operations
        times = np.array([p.t for p in trajectory])
        velocities = np.array([p.v for p in trajectory])
        angular_velocities = np.array([p.w for p in trajectory])

        # Apply velocity constraints
        velocities = np.clip(velocities, 0, self.max_velocity)
        angular_velocities = np.clip(
            angular_velocities, -self.max_angular_velocity, self.max_angular_velocity
        )

        # Calculate accelerations
        dt = times[1:] - times[:-1]
        accelerations = np.diff(velocities) / dt
        angular_accelerations = np.diff(angular_velocities) / dt

        # Apply acceleration constraints
        max_dv = self.max_acceleration * dt
        min_dv = self.max_deceleration * dt

        for i in range(len(accelerations)):
            if accelerations[i] > self.max_acceleration:
                velocities[i + 1] = velocities[i] + max_dv[i]
            elif accelerations[i] < self.max_deceleration:
                velocities[i + 1] = velocities[i] + min_dv[i]

        # Create constrained trajectory
        constrained_trajectory = []
        for i in range(len(trajectory)):
            constrained_trajectory.append(
                TrajectoryPoint(
                    x=trajectory[i].x,
                    y=trajectory[i].y,
                    theta=trajectory[i].theta,
                    v=float(velocities[i]),
                    w=float(angular_velocities[i]),
                    t=float(times[i]),
                )
            )

        return constrained_trajectory

    def _smooth_velocity_profile(self, velocities: np.ndarray) -> np.ndarray:
        """Apply smoothing to velocity profile."""
        window_size = 5
        window = np.ones(window_size) / window_size

        # Pad velocity array
        padded_velocities = np.pad(
            velocities, (window_size // 2, window_size // 2), mode="edge"
        )

        # Apply moving average
        smoothed = np.convolve(padded_velocities, window, mode="valid")

        return smoothed
