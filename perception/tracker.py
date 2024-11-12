# perception/tracker.py
from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np


@dataclass
class TrackedObject:
    """Object being tracked across frames."""

    id: int
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    class_label: str
    confidence: float
    history: List[np.ndarray]  # Position history


class SignTracker:
    """Tracks detected signs across frames."""

    def __init__(self, max_age: int = 10):
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 0
        self.max_age = max_age

    def update(self, detections: List[Dict[str, Any]]):
        """Update tracks with new detections."""
        pass
