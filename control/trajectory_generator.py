# control/trajectory_generator.py


from typing import List
from dataclasses import dataclass
from planning.path_planner import PathPlan


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""

    x: float
    y: float
    theta: float
    v: float
    w: float
    t: float


class TrajectoryGenerator:
    """Generates smooth trajectory from path plan."""

    def generate(self, path: PathPlan) -> List[TrajectoryPoint]:
        """Generate trajectory points."""
        pass
