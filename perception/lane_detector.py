# perception/lane_detector.py

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from .vision import VisionModule, VisionFeature, LaneFeature


@dataclass
class LaneParameters:
    """Lane detection parameters."""

    roi_width: int = 400
    roi_height: int = 100
    threshold: int = 200
    yellow_lower: Tuple[int, int, int] = (20, 100, 100)
    yellow_upper: Tuple[int, int, int] = (30, 255, 255)


class LaneDetector(VisionModule):
    """Enhanced lane detection implementation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.params = LaneParameters(**config.get("lane_params", {}))
        self.last_valid_feature: Optional[LaneFeature] = None

    def process(self, frame: np.ndarray) -> Dict[str, VisionFeature]:
        """
        Process a frame to detect lane features.
        Enhanced with both white and yellow lane detection.
        """
        frame_height, frame_width = frame.shape[:2]
        roi_x = (frame_width - self.params.roi_width) // 2
        roi_y = frame_height - self.params.roi_height - 20

        # Extract ROI
        roi = frame[
            roi_y : roi_y + self.params.roi_height,
            roi_x : roi_x + self.params.roi_width,
        ]

        # Detect white lanes
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(
            gray_roi, self.params.threshold, 255, cv2.THRESH_BINARY
        )

        # Detect yellow lanes
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(
            hsv_roi,
            np.array(self.params.yellow_lower),
            np.array(self.params.yellow_upper),
        )

        # Combine masks
        combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

        # Find lane edges
        edges = self._detect_edges(combined_mask)

        # Find lane lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=100
        )

        if lines is None:
            return self._create_empty_feature()

        # Separate left and right lane lines
        left_lines, right_lines = self._separate_lines(lines)

        # Calculate lane properties
        lane_feature = self._calculate_lane_features(
            left_lines, right_lines, self.params.roi_width
        )

        self.last_valid_feature = lane_feature

        return {"lanes": lane_feature}

    def _detect_edges(self, mask: np.ndarray) -> np.ndarray:
        """Detect edges in the binary mask."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)

        # Detect edges using Canny
        edges = cv2.Canny(blurred, 50, 150)

        return edges

    def _separate_lines(
        self, lines: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Separate detected lines into left and right lanes."""
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:  # Vertical line
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Filter by slope
            if abs(slope) < 0.3:  # Too horizontal
                continue

            if slope < 0:  # Negative slope = left lane
                left_lines.append(line)
            else:  # Positive slope = right lane
                right_lines.append(line)

        return left_lines, right_lines

    def _calculate_lane_features(
        self,
        left_lines: List[np.ndarray],
        right_lines: List[np.ndarray],
        roi_width: int,
    ) -> LaneFeature:
        """Calculate lane features from detected lines."""
        left_detected = len(left_lines) > 0
        right_detected = len(right_lines) > 0

        # Calculate average lane positions
        left_x = (
            0 if not left_detected else np.mean([line[0][0] for line in left_lines])
        )
        right_x = (
            roi_width
            if not right_detected
            else np.mean([line[0][0] for line in right_lines])
        )

        # Calculate center offset
        center_x = (left_x + right_x) / 2
        center_offset = center_x - (roi_width / 2)

        # Calculate lane curvature (simplified)
        curvature = 0.0
        if left_detected and right_detected:
            left_slope = np.mean(
                [
                    (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])
                    for line in left_lines
                ]
            )
            right_slope = np.mean(
                [
                    (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])
                    for line in right_lines
                ]
            )
            curvature = (right_slope - left_slope) / 2

        # Determine lane type
        lane_type = "none"
        if left_detected or right_detected:
            lane_type = "solid" if abs(curvature) < 0.1 else "curved"

        return LaneFeature(
            timestamp=0.0,  # Should be set from frame metadata
            confidence=0.8 if (left_detected and right_detected) else 0.4,
            lane_type=lane_type,
            center_offset=float(center_offset),
            curvature=float(curvature),
            is_left_detected=left_detected,
            is_right_detected=right_detected,
        )

    def _create_empty_feature(self) -> Dict[str, VisionFeature]:
        """Create empty feature when no lanes are detected."""
        return {
            "lanes": LaneFeature(
                timestamp=0.0,
                confidence=0.0,
                lane_type="none",
                center_offset=0.0,
                curvature=0.0,
                is_left_detected=False,
                is_right_detected=False,
            )
        }

    def visualize(
        self, frame: np.ndarray, features: Dict[str, VisionFeature]
    ) -> np.ndarray:
        """Visualize detected lane features on frame."""
        lane_feature = features.get("lanes")
        if not lane_feature:
            return frame

        frame_height, frame_width = frame.shape[:2]
        roi_x = (frame_width - self.params.roi_width) // 2
        roi_y = frame_height - self.params.roi_height - 20

        # Draw ROI
        cv2.rectangle(
            frame,
            (roi_x, roi_y),
            (roi_x + self.params.roi_width, roi_y + self.params.roi_height),
            (0, 255, 0),
            2,
        )

        # Draw center line
        center_x = int(roi_x + self.params.roi_width / 2 + lane_feature.center_offset)
        cv2.line(
            frame,
            (center_x, roi_y + self.params.roi_height),
            (center_x, roi_y + self.params.roi_height - 20),
            (0, 0, 255),
            2,
        )

        # Add text info
        text = f"Type: {lane_feature.lane_type} | Confidence: {lane_feature.confidence:.2f}"
        cv2.putText(
            frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        return frame
