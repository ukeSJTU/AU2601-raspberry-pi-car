# perception/feature_fusion.py

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import deque
from .vision import VisionFeature, LaneFeature, SignFeature


@dataclass
class FusedFeatures:
    """Container for fused perception features."""

    lanes: Optional[LaneFeature] = None
    signs: List[SignFeature] = None
    timestamp: float = 0.0


class FeatureFusion:
    """Enhanced feature fusion with temporal filtering."""

    def __init__(self):
        self.history_size = 5
        self.lane_history = deque(maxlen=self.history_size)
        self.sign_history = deque(maxlen=self.history_size)

    def fuse_features(
        self, features: Dict[str, Dict[str, VisionFeature]]
    ) -> Dict[str, Any]:
        """
        Combine features from multiple sources with temporal filtering.

        Args:
            features: Dictionary with feature type as key and features dict as value
                     e.g. {"lane": {...}, "sign": {...}}
        """
        # Extract lane features
        lane_features = None
        if "lane" in features:
            lane_features = features["lane"].get("lanes")
            if lane_features:
                self.lane_history.append(lane_features)

        # Extract sign features
        sign_features = []
        if "sign" in features:
            sign_features = [
                f for f in features["sign"].values() if isinstance(f, SignFeature)
            ]
            if sign_features:
                self.sign_history.append(sign_features)

        # Perform temporal fusion
        fused_lane = self._fuse_lane_features()
        fused_signs = self._fuse_sign_features()

        # Combine all features
        fused_features = {
            "lane_feature": fused_lane,
            "sign_features": fused_signs,
            "scene_features": self._extract_scene_features(fused_lane, fused_signs),
        }

        return fused_features

    def _fuse_lane_features(self) -> Optional[LaneFeature]:
        """Fuse lane features with temporal filtering."""
        if not self.lane_history:
            return None

        # Weight recent features more heavily
        weights = np.exp(np.linspace(-1, 0, len(self.lane_history)))
        weights /= weights.sum()

        # Compute weighted average of numeric properties
        offset_sum = 0
        curvature_sum = 0
        confidence_sum = 0

        for w, feature in zip(weights, self.lane_history):
            offset_sum += w * feature.center_offset
            curvature_sum += w * feature.curvature
            confidence_sum += w * feature.confidence

        # Use most recent feature for boolean properties
        recent = self.lane_history[-1]

        return LaneFeature(
            timestamp=recent.timestamp,
            confidence=confidence_sum,
            lane_type=recent.lane_type,
            center_offset=offset_sum,
            curvature=curvature_sum,
            is_left_detected=recent.is_left_detected,
            is_right_detected=recent.is_right_detected,
        )

    def _fuse_sign_features(self) -> List[SignFeature]:
        """Fuse sign features with temporal filtering."""
        if not self.sign_history:
            return []

        # Get most recent signs
        recent_signs = self.sign_history[-1]

        # Find persistent signs across history
        persistent_signs = []
        for sign in recent_signs:
            # Count occurrences of this sign type in history
            occurrences = sum(
                1
                for hist in self.sign_history
                for old_sign in hist
                if old_sign.sign_type == sign.sign_type
                and self._is_similar_position(old_sign.position, sign.position)
            )

            # Only keep signs that appear in multiple frames
            if occurrences >= 2:
                # Boost confidence for persistent signs
                sign.confidence = min(1.0, sign.confidence * (1.0 + 0.1 * occurrences))
                persistent_signs.append(sign)

        return persistent_signs

    def _is_similar_position(
        self, pos1: tuple, pos2: tuple, threshold: int = 50
    ) -> bool:
        """Check if two positions are similar within threshold."""
        return abs(pos1[0] - pos2[0]) < threshold and abs(pos1[1] - pos2[1]) < threshold

    def _extract_scene_features(
        self, lane: Optional[LaneFeature], signs: List[SignFeature]
    ) -> Dict[str, Any]:
        """Extract high-level scene features from fused perceptions."""
        scene_features = {
            "has_lanes": lane is not None and lane.confidence > 0.5,
            "lane_offset": lane.center_offset if lane else 0.0,
            "is_curved": lane.lane_type == "curved" if lane else False,
            "detected_signs": [sign.sign_type for sign in signs],
            "highest_confidence_sign": None,
        }

        # Find highest confidence sign
        if signs:
            max_conf_sign = max(signs, key=lambda x: x.confidence)
            scene_features["highest_confidence_sign"] = {
                "type": max_conf_sign.sign_type,
                "confidence": max_conf_sign.confidence,
            }

        return scene_features
