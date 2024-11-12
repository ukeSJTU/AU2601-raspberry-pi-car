# planning/behavior_planner.py

from typing import Dict, Any
from dataclasses import dataclass
from planning.scene import SceneType


@dataclass
class BehaviorCommand:
    """Command from behavior planner."""

    action: str
    parameters: Dict[str, Any]
    priority: int
    timestamp: float


class BehaviorPlanner:
    """Plans vehicle behavior based on scene and context."""

    def __init__(self):
        self.decision_history = []

    def plan(
        self, scene: SceneType, features: Dict[str, Any], context: Dict[str, Any]
    ) -> BehaviorCommand:
        """Generate behavior command based on current situation."""
        pass
