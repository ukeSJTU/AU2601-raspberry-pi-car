# planning/scene_manager.py

from enum import Enum
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import time


class SceneType(Enum):
    """Possible driving scenes."""

    CRUISE = "cruise"  # Normal cruising
    TURN = "turn"  # Turning at intersection
    STOP = "stop"  # Stopping (traffic sign, obstacle, etc)
    PARK = "park"  # Parking maneuver


@dataclass
class SceneContext:
    """Additional context for current scene."""

    duration: float = 0.0  # How long in this scene
    confidence: float = 0.0  # Confidence in scene classification
    metadata: Dict[str, Any] = None  # Additional scene-specific data


class SceneManager:
    """Enhanced scene understanding and management."""

    def __init__(self):
        self.current_scene = SceneType.CRUISE
        self.scene_history: List[SceneType] = []
        self.scene_context = SceneContext()
        self.scene_start_time = time.time()

        # Scene transition rules
        self.valid_transitions: Dict[SceneType, Set[SceneType]] = {
            SceneType.CRUISE: {SceneType.TURN, SceneType.STOP, SceneType.PARK},
            SceneType.TURN: {SceneType.CRUISE, SceneType.STOP},
            SceneType.STOP: {SceneType.CRUISE, SceneType.TURN, SceneType.PARK},
            SceneType.PARK: {SceneType.CRUISE},
        }

        # Minimum duration for each scene (seconds)
        self.min_durations = {
            SceneType.CRUISE: 2.0,
            SceneType.TURN: 3.0,
            SceneType.STOP: 1.0,
            SceneType.PARK: 5.0,
        }

    def update_scene(
        self, features: Dict[str, Any], context: Dict[str, Any]
    ) -> SceneType:
        """
        Update current scene based on perceived features and context.

        Args:
            features: Fused perception features
            context: Current driving context

        Returns:
            Updated scene type
        """
        # Extract relevant features
        lane_feature = features.get("lane_feature")
        sign_features = features.get("sign_features", [])
        scene_features = features.get("scene_features", {})

        # Calculate current scene duration
        current_duration = time.time() - self.scene_start_time

        # Get candidate scene based on current features
        candidate_scene = self._determine_scene(features, context)

        # Check if scene transition is valid and minimum duration is met
        if (
            candidate_scene != self.current_scene
            and candidate_scene in self.valid_transitions[self.current_scene]
            and current_duration >= self.min_durations[self.current_scene]
        ):

            # Update scene history
            self.scene_history.append(self.current_scene)
            if len(self.scene_history) > 10:  # Keep last 10 scenes
                self.scene_history.pop(0)

            # Update scene
            self.current_scene = candidate_scene
            self.scene_start_time = time.time()

            # Initialize new scene context
            self.scene_context = SceneContext(
                duration=0.0,
                confidence=self._calculate_scene_confidence(features, candidate_scene),
                metadata=self._extract_scene_metadata(features, context),
            )
        else:
            # Update current scene context
            self.scene_context.duration = current_duration
            self.scene_context.confidence = self._calculate_scene_confidence(
                features, self.current_scene
            )

        return self.current_scene

    def _determine_scene(
        self, features: Dict[str, Any], context: Dict[str, Any]
    ) -> SceneType:
        """Determine appropriate scene based on current features."""
        scene_features = features.get("scene_features", {})

        # Check for stop signs or traffic lights
        sign_types = set(scene_features.get("detected_signs", []))
        if "stop" in sign_types and scene_features.get("has_lanes", False):
            return SceneType.STOP

        # Check for turn scene
        lane_feature = features.get("lane_feature")
        if lane_feature and abs(lane_feature.curvature) > context.get(
            "turn_threshold", 0.3
        ):
            return SceneType.TURN

        # Check for parking
        if "park" in sign_types or context.get("parking_requested", False):
            return SceneType.PARK

        # Default to cruise
        return SceneType.CRUISE

    def _calculate_scene_confidence(
        self, features: Dict[str, Any], scene: SceneType
    ) -> float:
        """Calculate confidence in scene classification."""
        scene_features = features.get("scene_features", {})

        # Base confidence on lane detection quality
        base_confidence = 0.5
        if scene_features.get("has_lanes"):
            lane_feature = features.get("lane_feature")
            if lane_feature:
                base_confidence = lane_feature.confidence

        # Adjust based on scene-specific factors
        if scene == SceneType.STOP:
            # Higher confidence if stop sign is detected with high confidence
            sign_conf = scene_features.get("highest_confidence_sign", {}).get(
                "confidence", 0
            )
            return min(1.0, base_confidence + 0.3 * sign_conf)

        elif scene == SceneType.TURN:
            # Higher confidence if consistent curve detection
            curve_confidence = 0.3
            if features.get("lane_feature"):
                if abs(features["lane_feature"].curvature) > 0.3:
                    curve_confidence = 0.8
            return min(1.0, base_confidence + curve_confidence)

        return base_confidence

    def _extract_scene_metadata(
        self, features: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract relevant metadata for current scene."""
        metadata = {}

        if self.current_scene == SceneType.TURN:
            if features.get("lane_feature"):
                metadata["turn_direction"] = (
                    "left" if features["lane_feature"].curvature > 0 else "right"
                )
                metadata["curvature"] = abs(features["lane_feature"].curvature)

        elif self.current_scene == SceneType.STOP:
            sign_feature = features.get("scene_features", {}).get(
                "highest_confidence_sign"
            )
            if sign_feature:
                metadata["stop_sign_confidence"] = sign_feature.get("confidence", 0)

        elif self.current_scene == SceneType.PARK:
            metadata["parking_side"] = context.get("parking_side", "right")

        return metadata

    def get_scene_context(self) -> SceneContext:
        """Get current scene context."""
        return self.scene_context
