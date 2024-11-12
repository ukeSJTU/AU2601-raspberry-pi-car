# control/controller.py

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from .trajectory_generator import TrajectoryPoint
from .vehicle_state import VehicleState
from .input_manager import InputManager, ManualCommand, InputMode


@dataclass
class ControlCommand:
    """Low-level control command."""

    forward_speed: float  # Forward speed in m/s
    steering_angle: float  # Steering angle in radians
    lateral_speed: float = 0.0  # Lateral speed for holonomic vehicles
    brake: bool = False  # Brake flag


class Controller:
    """Enhanced controller with pure pursuit and PID control."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.input_manager = InputManager(config.get("input_manager", {}))

        # Controller parameters
        self.wheelbase = config.get("wheelbase", 2.8)  # meters
        self.lookahead_distance = config.get("lookahead_distance", 5.0)  # meters

        # Speed PID parameters
        self.speed_kp = config.get("speed_kp", 1.0)
        self.speed_ki = config.get("speed_ki", 0.1)
        self.speed_kd = config.get("speed_kd", 0.05)

        # Steering PID parameters
        self.steering_kp = config.get("steering_kp", 1.5)
        self.steering_ki = config.get("steering_ki", 0.0)
        self.steering_kd = config.get("steering_kd", 0.1)

        # Control limits
        self.max_speed = config.get("max_speed", 50.0)  # km/h
        self.max_steering = np.radians(config.get("max_steering", 35))  # radians

        # Control state
        self.speed_error_integral = 0.0
        self.speed_error_previous = 0.0
        self.steering_error_integral = 0.0
        self.steering_error_previous = 0.0
        self.last_command: Optional[ControlCommand] = None

    async def start(self):
        """Start the controller and input management."""
        # Register callback for manual inputs
        self.input_manager.register_callback("control", self._handle_manual_input)
        # Start keyboard monitoring
        await self.input_manager.start_keyboard_monitoring()

    def compute_control(
        self,
        current_state: VehicleState,
        target_trajectory: List[TrajectoryPoint],
        manual_command: Optional[ManualCommand] = None,
        dt: float = 0.1,
    ) -> ControlCommand:
        """
        Compute control commands with manual input fusion.

        Args:
            current_state: Current vehicle state
            target_trajectory: Target trajectory points
            manual_command: Optional manual control input
            dt: Time step
        """
        mode = self.input_manager.get_current_mode()

        if mode == InputMode.MANUAL:
            return self._compute_manual_control(manual_command)

        elif mode == InputMode.HYBRID:
            auto_command = self._compute_autonomous_control(
                current_state, target_trajectory, dt
            )
            return self._fuse_commands(auto_command, manual_command)

        else:  # AUTO mode
            return self._compute_autonomous_control(
                current_state, target_trajectory, dt
            )

    def _compute_manual_control(
        self, manual_command: Optional[ManualCommand]
    ) -> ControlCommand:
        """Compute control for manual mode."""
        if not manual_command:
            return ControlCommand(forward_speed=0.0, steering_angle=0.0)

        # Convert manual inputs to control command
        forward_speed = manual_command.speed_delta * self.max_speed
        steering_angle = manual_command.steering_delta * self.max_steering

        return ControlCommand(
            forward_speed=forward_speed,
            steering_angle=steering_angle,
            brake=manual_command.brake,
        )

    def _compute_autonomous_control(
        self,
        current_state: VehicleState,
        target_trajectory: List[TrajectoryPoint],
        dt: float,
    ) -> ControlCommand:
        """Compute control for autonomous mode."""
        if not target_trajectory:
            return ControlCommand(forward_speed=0.0, steering_angle=0.0)

        # Find target point using pure pursuit
        target_point = self._find_target_point(
            current_state.position[:2], current_state.heading, target_trajectory
        )

        # Compute controls
        speed_command = self._compute_speed_control(
            current_state.speed, target_point.v, dt
        )

        steering_command = self._compute_steering_control(
            current_state.position[:2], current_state.heading, target_point, dt
        )

        return ControlCommand(
            forward_speed=float(speed_command), steering_angle=float(steering_command)
        )

    def _find_target_point(
        self,
        current_pos: np.ndarray,
        current_heading: float,
        trajectory: List[TrajectoryPoint],
    ) -> TrajectoryPoint:
        """Find target point on trajectory using pure pursuit."""
        # Transform trajectory points to vehicle frame
        cos_h = np.cos(-current_heading)
        sin_h = np.sin(-current_heading)
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        # Find closest point exceeding lookahead distance
        min_dist = float("inf")
        target_point = trajectory[0]

        for point in trajectory:
            # Transform to vehicle frame
            delta = np.array([point.x, point.y]) - current_pos
            local_point = R @ delta

            dist = np.linalg.norm(local_point)
            if dist >= self.lookahead_distance and dist < min_dist:
                min_dist = dist
                target_point = point

        return target_point

    def _compute_speed_control(
        self, current_speed: float, target_speed: float, dt: float
    ) -> float:
        """Compute speed control using PID."""
        # Calculate error
        speed_error = target_speed - current_speed

        # Update integral term
        self.speed_error_integral += speed_error * dt

        # Calculate derivative term
        speed_error_derivative = (speed_error - self.speed_error_previous) / dt
        self.speed_error_previous = speed_error

        # PID control
        control = (
            self.speed_kp * speed_error
            + self.speed_ki * self.speed_error_integral
            + self.speed_kd * speed_error_derivative
        )

        # Apply limits
        control = np.clip(control, -self.max_speed, self.max_speed)

        return control

    def _compute_steering_control(
        self,
        current_pos: np.ndarray,
        current_heading: float,
        target_point: TrajectoryPoint,
        dt: float,
    ) -> float:
        """Compute steering control using pure pursuit and PID."""
        # Calculate target heading
        delta = np.array([target_point.x, target_point.y]) - current_pos
        target_heading = np.arctan2(delta[1], delta[0])

        # Calculate heading error
        heading_error = self._normalize_angle(target_heading - current_heading)

        # Update integral term
        self.steering_error_integral += heading_error * dt

        # Calculate derivative term
        steering_error_derivative = (heading_error - self.steering_error_previous) / dt
        self.steering_error_previous = heading_error

        # PID control
        control = (
            self.steering_kp * heading_error
            + self.steering_ki * self.steering_error_integral
            + self.steering_kd * steering_error_derivative
        )

        # Apply limits
        control = np.clip(control, -self.max_steering, self.max_steering)

        return control

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _fuse_commands(
        self, auto_cmd: ControlCommand, manual_cmd: Optional[ManualCommand]
    ) -> ControlCommand:
        """Fuse autonomous and manual commands."""
        if not manual_cmd:
            return auto_cmd

        # Simple additive fusion with manual override for brake
        fused_cmd = ControlCommand(
            forward_speed=auto_cmd.forward_speed
            + manual_cmd.speed_delta * self.max_speed,
            steering_angle=auto_cmd.steering_angle
            + manual_cmd.steering_delta * self.max_steering,
            brake=manual_cmd.brake or auto_cmd.brake,
        )

        # Apply limits
        fused_cmd.forward_speed = np.clip(
            fused_cmd.forward_speed, -self.max_speed, self.max_speed
        )
        fused_cmd.steering_angle = np.clip(
            fused_cmd.steering_angle, -self.max_steering, self.max_steering
        )

        return fused_cmd

    def reset(self) -> None:
        """Reset controller state."""
        self.speed_error_integral = 0.0
        self.speed_error_previous = 0.0
        self.steering_error_integral = 0.0
        self.steering_error_previous = 0.0
        self.last_command = None
