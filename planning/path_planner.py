# planning/path_planner.py

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interp1d
from .behavior_planner import BehaviorCommand


@dataclass
class PathPlan:
    """Planned path with associated metadata."""

    waypoints: List[Tuple[float, float]]  # List of (x,y) coordinates
    speeds: List[float]  # Target speed at each waypoint
    curvatures: List[float]  # Path curvature at each waypoint


class PathPlanner:
    """Enhanced path planning with smoothing and constraints."""

    def __init__(self):
        # Planning parameters
        self.lookahead_distance = 5.0  # meters
        self.point_spacing = 0.5  # meters between waypoints
        self.max_curvature = 0.5  # maximum allowed curvature
        self.max_speed = 50.0  # km/h
        self.min_speed = 5.0  # km/h

        # Vehicle parameters
        self.wheelbase = 2.8  # meters
        self.max_steering_angle = np.radians(35)  # radians

    def plan_path(self, command: BehaviorCommand, features: Dict[str, Any]) -> PathPlan:
        """
        Generate detailed path plan based on behavior command.

        Args:
            command: Behavior command to execute
            features: Current perception features

        Returns:
            Detailed path plan
        """
        # Get reference path based on action type
        raw_path = self._generate_reference_path(command, features)

        # Smooth the path
        smoothed_path = self._smooth_path(raw_path)

        # Calculate path properties
        curvatures = self._calculate_curvatures(smoothed_path)

        # Generate speed profile
        speeds = self._generate_speed_profile(smoothed_path, curvatures, command)

        return PathPlan(waypoints=smoothed_path, speeds=speeds, curvatures=curvatures)

    def _generate_reference_path(
        self, command: BehaviorCommand, features: Dict[str, Any]
    ) -> List[Tuple[float, float]]:
        """Generate initial reference path."""
        action = command.action
        params = command.parameters

        if action == "cruise":
            return self._generate_cruise_path(features, params)
        elif action == "turn":
            return self._generate_turn_path(features, params)
        elif action == "stop":
            return self._generate_stop_path(features, params)
        elif action == "park":
            return self._generate_parking_path(features, params)
        else:
            # Default to cruise path
            return self._generate_cruise_path(features, params)

    def _generate_cruise_path(
        self, features: Dict[str, Any], params: Dict[str, Any]
    ) -> List[Tuple[float, float]]:
        """Generate path for cruising."""
        lane_feature = features.get("lane_feature")

        if not lane_feature:
            # If no lane detected, go straight
            return [
                (x, 0)
                for x in np.arange(0, self.lookahead_distance, self.point_spacing)
            ]

        # Generate path following lane center
        points = []
        x_coords = np.arange(0, self.lookahead_distance, self.point_spacing)

        for x in x_coords:
            # Calculate lateral offset using lane curvature
            y = lane_feature.center_offset + lane_feature.curvature * x * x / 2
            points.append((x, y))

        return points

    def _generate_turn_path(
        self, features: Dict[str, Any], params: Dict[str, Any]
    ) -> List[Tuple[float, float]]:
        """Generate turning path."""
        turn_direction = params.get("turn_direction", "right")
        turn_radius = self.wheelbase / np.tan(self.max_steering_angle)

        points = []
        if turn_direction == "right":
            center = (0, -turn_radius)  # Center of turning circle
            angles = np.linspace(np.pi / 2, 0, num=20)
        else:
            center = (0, turn_radius)
            angles = np.linspace(-np.pi / 2, 0, num=20)

        for angle in angles:
            x = center[0] + turn_radius * np.cos(angle)
            y = center[1] + turn_radius * np.sin(angle)
            points.append((x, y))

        return points

    def _generate_stop_path(
        self, features: Dict[str, Any], params: Dict[str, Any]
    ) -> List[Tuple[float, float]]:
        """Generate stopping path."""
        # Similar to cruise path but shorter
        points = self._generate_cruise_path(features, params)

        # Truncate path based on stopping distance
        current_speed = params.get("current_speed", 30.0)
        decel = abs(params.get("required_acceleration", -2.0))
        stop_distance = current_speed * current_speed / (2 * decel)

        # Keep only points within stopping distance
        truncated_points = []
        accumulated_distance = 0
        prev_point = points[0]

        for point in points:
            dx = point[0] - prev_point[0]
            dy = point[1] - prev_point[1]
            accumulated_distance += np.sqrt(dx * dx + dy * dy)

            if accumulated_distance > stop_distance:
                break

            truncated_points.append(point)
            prev_point = point

        return truncated_points

    def _generate_parking_path(
        self, features: Dict[str, Any], params: Dict[str, Any]
    ) -> List[Tuple[float, float]]:
        """Generate parking maneuver path."""
        parking_type = params.get("parking_type", "parallel")
        side = params.get("side", "right")

        if parking_type == "parallel":
            return self._generate_parallel_parking_path(side)
        else:
            return self._generate_perpendicular_parking_path(side)

    def _smooth_path(
        self, path: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Apply path smoothing using spline interpolation."""
        if len(path) < 2:
            return path

        # Extract x and y coordinates
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]

        # Create parameterization for spline
        t = np.linspace(0, 1, len(path))

        # Fit splines
        x_spline = interp1d(t, x_coords, kind="cubic")
        y_spline = interp1d(t, y_coords, kind="cubic")

        # Generate smoothed points
        t_new = np.linspace(0, 1, int(len(path) * 1.5))
        smoothed_path = [(x_spline(t), y_spline(t)) for t in t_new]

        return smoothed_path

    def _calculate_curvatures(self, path: List[Tuple[float, float]]) -> List[float]:
        """Calculate curvature at each path point."""
        if len(path) < 3:
            return [0.0] * len(path)

        curvatures = []
        for i in range(len(path)):
            if i == 0 or i == len(path) - 1:
                curvatures.append(0.0)
                continue

            # Get three consecutive points
            p1 = np.array(path[i - 1])
            p2 = np.array(path[i])
            p3 = np.array(path[i + 1])

            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2

            # Calculate curvature using cross product
            cross_prod = np.cross(v1, v2)
            curvature = (2 * cross_prod) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(p3 - p1)
            )

            # Limit maximum curvature
            curvature = np.clip(curvature, -self.max_curvature, self.max_curvature)
            curvatures.append(curvature)

        return curvatures

    def _generate_speed_profile(
        self,
        path: List[Tuple[float, float]],
        curvatures: List[float],
        command: BehaviorCommand,
    ) -> List[float]:
        """Generate speed profile along the path."""
        target_speed = command.parameters.get("target_speed", 30.0)

        speeds = []
        for curvature in curvatures:
            # Reduce speed in curves based on curvature
            curvature_factor = 1.0 - min(1.0, abs(curvature) / self.max_curvature)
            speed = target_speed * curvature_factor

            # Ensure speed limits
            speed = np.clip(speed, self.min_speed, self.max_speed)
            speeds.append(speed)

        # Smooth speed profile
        speeds = self._smooth_speed_profile(speeds)

        return speeds

    def _smooth_speed_profile(self, speeds: List[float]) -> List[float]:
        """Apply smoothing to speed profile."""
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size

        # Pad speeds for convolution
        padded_speeds = np.pad(
            speeds, (kernel_size // 2, kernel_size // 2), mode="edge"
        )

        # Apply smoothing
        smoothed = np.convolve(padded_speeds, kernel, mode="valid")

        return smoothed.tolist()

    def _generate_parallel_parking_path(self, side: str) -> List[Tuple[float, float]]:
        """Generate path for parallel parking."""
        # Simplified parallel parking trajectory
        points = []
        if side == "right":
            points.extend([(x, 0) for x in np.linspace(0, 2, 5)])  # Move forward
            points.extend(
                [(2 - x, -x) for x in np.linspace(0, 1, 5)]
            )  # Reverse + turn right
            points.extend([(1 + x, -1) for x in np.linspace(0, 1, 5)])  # Straighten
        else:
            # Mirror for left side parking
            points.extend([(x, 0) for x in np.linspace(0, 2, 5)])
            points.extend([(2 - x, x) for x in np.linspace(0, 1, 5)])
            points.extend([(1 + x, 1) for x in np.linspace(0, 1, 5)])

        return points

    def _generate_perpendicular_parking_path(
        self, side: str
    ) -> List[Tuple[float, float]]:
        """Generate path for perpendicular parking."""
        points = []
        if side == "right":
            points.extend([(x, 0) for x in np.linspace(0, 1, 5)])  # Approach
            points.extend([(1, -x) for x in np.linspace(0, 2, 10)])  # Turn into spot
        else:
            points.extend([(x, 0) for x in np.linspace(0, 1, 5)])
            points.extend([(1, x) for x in np.linspace(0, 2, 10)])

        return points
