# perception/vision.py
from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VisionFeature:
    """Base class for all vision features."""

    timestamp: float
    confidence: float


@dataclass
class LaneFeature(VisionFeature):
    """Lane detection features."""

    lane_type: str  # 'solid', 'dashed', 'double', 'none'
    center_offset: float
    curvature: float
    is_left_detected: bool
    is_right_detected: bool


@dataclass
class SignFeature(VisionFeature):
    """Traffic sign features."""

    sign_type: str
    position: Tuple[int, int]  # (x, y) in image
    size: Tuple[int, int]  # width, height
    orientation: float  # estimated orientation in radians


class VisionModule(ABC):
    """Abstract base class for vision processing modules."""

    @abstractmethod
    def process(self, frame: np.ndarray) -> Dict[str, VisionFeature]:
        """Process a frame and return detected features."""
        pass

    @abstractmethod
    def visualize(
        self, frame: np.ndarray, features: Dict[str, VisionFeature]
    ) -> np.ndarray:
        """Visualize detected features on frame."""
        pass
