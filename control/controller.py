# control/controller.py

from dataclasses import dataclass
from typing import Dict, Any, Optional
from control.trajectory_generator import TrajectoryPoint
from input_manager import InputManager, ManualCommand, InputMode


@dataclass
class ControlCommand:
    """Low-level control command."""

    forward_speed: float
    steering_angle: float
    lateral_speed: float = 0.0


class Controller:
    """Enhanced controller with manual input support."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.input_manager = InputManager(config.get("input_manager", {}))
        self.current_speed = 0
        self.current_steering = 0

    async def start(self):
        """Start the controller and input management."""
        # Register callback for manual inputs
        self.input_manager.register_callback("control", self._handle_manual_input)
        # Start keyboard monitoring
        await self.input_manager.start_keyboard_monitoring()

    def _handle_manual_input(self, command: ManualCommand):
        """Handle manual control inputs."""
        # Update internal state based on manual command
        if command.brake:
            self.current_speed = 0
        else:
            self.current_speed += command.speed_delta
        self.current_steering += command.steering_delta

    def compute_control(
        self,
        current: TrajectoryPoint,
        target: TrajectoryPoint,
        manual_command: Optional[ManualCommand] = None,
    ) -> ControlCommand:
        """Compute control commands with manual input fusion."""
        mode = self.input_manager.get_current_mode()

        if mode == InputMode.MANUAL:
            # Pure manual control
            return ControlCommand(
                forward_speed=self.current_speed, steering_angle=self.current_steering
            )

        elif mode == InputMode.HYBRID:
            # Fuse autonomous and manual control
            auto_command = self._compute_autonomous_control(current, target)
            return self._fuse_commands(auto_command, manual_command)

        else:  # AUTO mode
            return self._compute_autonomous_control(current, target)

    def _compute_autonomous_control(
        self, current: TrajectoryPoint, target: TrajectoryPoint
    ) -> ControlCommand:
        """Compute autonomous control command."""
        # Original autonomous control logic here
        pass

    def _fuse_commands(
        self, auto_cmd: ControlCommand, manual_cmd: Optional[ManualCommand]
    ) -> ControlCommand:
        """Fuse autonomous and manual commands."""
        if not manual_cmd:
            return auto_cmd

        # Simple additive fusion
        return ControlCommand(
            forward_speed=auto_cmd.forward_speed + manual_cmd.speed_delta,
            steering_angle=auto_cmd.steering_angle + manual_cmd.steering_delta,
        )
