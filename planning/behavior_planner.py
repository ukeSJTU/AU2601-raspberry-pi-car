# planning/behavior_planner.py

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time
from .scene_manager import SceneType


@dataclass
class BehaviorCommand:
    """Command from behavior planner."""

    action: str  # Primary action to take
    parameters: Dict[str, Any]  # Action parameters
    priority: int  # Command priority (1-10)
    timestamp: float  # Command generation time


class BehaviorPlanner:
    """Enhanced behavior planning with decision history."""

    def __init__(self):
        self.decision_history: List[BehaviorCommand] = []

        # Define actions for each scene type
        self.scene_actions = {
            SceneType.CRUISE: self._plan_cruise,
            SceneType.TURN: self._plan_turn,
            SceneType.STOP: self._plan_stop,
            SceneType.PARK: self._plan_park,
        }

        # Emergency actions take priority
        self.emergency_actions = {
            "emergency_stop": {"priority": 10, "parameters": {"deceleration": -3.0}}
        }

    def plan(
        self, scene: SceneType, features: Dict[str, Any], context: Dict[str, Any]
    ) -> BehaviorCommand:
        """
        Generate behavior command based on current situation.

        Args:
            scene: Current scene type
            features: Perception features
            context: Driving context

        Returns:
            Behavior command
        """
        # Check for emergency situations first
        emergency_command = self._check_emergency(features, context)
        if emergency_command:
            return emergency_command

        # Get scene-specific planning function
        plan_function = self.scene_actions.get(scene, self._plan_cruise)

        # Generate command
        command = plan_function(features, context)

        # Update history
        self.decision_history.append(command)
        if len(self.decision_history) > 50:  # Keep last 50 decisions
            self.decision_history.pop(0)

        return command

    def _check_emergency(
        self, features: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[BehaviorCommand]:
        """Check for emergency situations requiring immediate action."""
        # Check for imminent collision
        if context.get("collision_warning", False):
            return BehaviorCommand(
                action="emergency_stop",
                parameters=self.emergency_actions["emergency_stop"]["parameters"],
                priority=self.emergency_actions["emergency_stop"]["priority"],
                timestamp=time.time(),
            )

        return None

    def _plan_cruise(
        self, features: Dict[str, Any], context: Dict[str, Any]
    ) -> BehaviorCommand:
        """Plan normal cruising behavior."""
        lane_feature = features.get("lane_feature")
        scene_features = features.get("scene_features", {})

        # Default cruise parameters
        params = {
            "target_speed": context.get("cruise_speed", 30),  # km/h
            "target_lane_offset": 0.0,
            "required_acceleration": 0.0,
            "steering_angle": 0.0,
        }

        # Adjust for lane offset
        if lane_feature:
            params["target_lane_offset"] = -lane_feature.center_offset
            params["steering_angle"] = -0.5 * lane_feature.center_offset

            # Reduce speed in curves
            if abs(lane_feature.curvature) > 0.2:
                curve_factor = 1.0 - min(1.0, abs(lane_feature.curvature))
                params["target_speed"] *= curve_factor

        return BehaviorCommand(
            action="cruise", parameters=params, priority=5, timestamp=time.time()
        )

    def _plan_turn(
        self, features: Dict[str, Any], context: Dict[str, Any]
    ) -> BehaviorCommand:
        """Plan turning behavior."""
        lane_feature = features.get("lane_feature")

        params = {
            "turn_direction": (
                "left" if (lane_feature and lane_feature.curvature > 0) else "right"
            ),
            "target_speed": 15.0,  # Reduced speed for turning
            "required_acceleration": -0.5,  # Slight deceleration
            "steering_angle": 35.0 if params["turn_direction"] == "left" else -35.0,
        }

        # Adjust steering based on curve sharpness
        if lane_feature:
            steering_factor = min(1.0, abs(lane_feature.curvature) * 2)
            params["steering_angle"] *= steering_factor

        return BehaviorCommand(
            action="turn", parameters=params, priority=6, timestamp=time.time()
        )

    def _plan_stop(
        self, features: Dict[str, Any], context: Dict[str, Any]
    ) -> BehaviorCommand:
        """Plan stopping behavior."""
        scene_features = features.get("scene_features", {})
        stop_sign = scene_features.get("highest_confidence_sign", {})

        params = {
            "target_speed": 0.0,
            "required_acceleration": -2.0,
            "stop_duration": 3.0 if stop_sign.get("type") == "stop" else 1.0,
            "maintain_position": True,
        }

        return BehaviorCommand(
            action="stop", parameters=params, priority=8, timestamp=time.time()
        )

    def _plan_park(
        self, features: Dict[str, Any], context: Dict[str, Any]
    ) -> BehaviorCommand:
        """Plan parking behavior."""
        params = {
            "parking_type": context.get("parking_type", "parallel"),
            "side": context.get("parking_side", "right"),
            "target_speed": 5.0,
            "required_acceleration": -1.0,
            "steering_sequence": self._generate_parking_sequence(
                context.get("parking_type", "parallel"),
                context.get("parking_side", "right"),
            ),
        }

        return BehaviorCommand(
            action="park", parameters=params, priority=7, timestamp=time.time()
        )

    def _generate_parking_sequence(self, parking_type: str, side: str) -> List[float]:
        """Generate sequence of steering angles for parking maneuver."""
        if parking_type == "parallel":
            return (
                [30 if side == "right" else -30] * 10
                + [-30 if side == "right" else 30] * 10
                + [0] * 5
            )
        else:  # perpendicular
            return [35 if side == "right" else -35] * 15 + [0] * 5

    def get_decision_history(self) -> List[BehaviorCommand]:
        """Get recent decision history."""
        return self.decision_history
