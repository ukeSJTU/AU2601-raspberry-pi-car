# control/input_manager.py
from dataclasses import dataclass
import asyncio
from enum import Enum
import keyboard
from typing import Dict, Any, Optional, Callable


class InputMode(Enum):
    AUTO = "auto"
    MANUAL = "manual"
    HYBRID = "hybrid"


@dataclass
class ManualCommand:
    """Manual control command from keyboard/joystick."""

    speed_delta: float = 0.0
    steering_delta: float = 0.0
    brake: bool = False
    override: bool = False  # Whether to completely override autonomous control


class InputManager:
    """Manages manual inputs and control mode switching."""

    def __init__(self, config: Dict[str, Any]):
        self.mode = InputMode.AUTO
        self.manual_command = ManualCommand()
        self.base_speed = config.get("base_speed", 70)
        self.speed_increment = config.get("speed_increment", 10)
        self.steering_increment = config.get("steering_increment", 5)
        self.callbacks: Dict[str, Callable] = {}

    async def start_keyboard_monitoring(self):
        """Start asynchronous keyboard monitoring."""

        def on_key_event(event):
            if event.event_type == "down":
                if event.name == "w":
                    self.manual_command.speed_delta += self.speed_increment
                elif event.name == "s":
                    self.manual_command.speed_delta -= self.speed_increment
                elif event.name == "a":
                    self.manual_command.steering_delta -= self.steering_increment
                elif event.name == "d":
                    self.manual_command.steering_delta += self.steering_increment
                elif event.name == "space":
                    self.manual_command.brake = True
                elif event.name == "m":
                    self._toggle_mode()

                # Notify callbacks
                for callback in self.callbacks.values():
                    callback(self.manual_command)

            elif event.event_type == "up":
                if event.name in ["w", "s"]:
                    self.manual_command.speed_delta = 0
                elif event.name in ["a", "d"]:
                    self.manual_command.steering_delta = 0
                elif event.name == "space":
                    self.manual_command.brake = False

        keyboard.hook(on_key_event)

    def _toggle_mode(self):
        """Toggle between control modes."""
        if self.mode == InputMode.AUTO:
            self.mode = InputMode.MANUAL
        elif self.mode == InputMode.MANUAL:
            self.mode = InputMode.HYBRID
        else:
            self.mode = InputMode.AUTO

    def register_callback(self, name: str, callback: Callable):
        """Register a callback for manual control updates."""
        self.callbacks[name] = callback

    def get_current_mode(self) -> InputMode:
        """Get current control mode."""
        return self.mode
