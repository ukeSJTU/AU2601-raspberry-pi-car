# planning/path_planner.py
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from planning.behavior_planner import BehaviorCommand


@dataclass
class PathPlan:
    """Planned path with associated metadata."""

    waypoints: List[Tuple[float, float]]
    speeds: List[float]
    curvatures: List[float]


class PathPlanner:
    """Plans detailed path based on behavior command."""

    def plan_path(self, command: BehaviorCommand, features: Dict[str, Any]) -> PathPlan:
        """Generate detailed path plan."""
        pass
