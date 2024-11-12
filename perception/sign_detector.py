# perception/sign_detector.py

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from .vision import VisionModule, VisionFeature, SignFeature
from .tracker import SignTracker, TrackedObject


class SignDetector(VisionModule):
    """Enhanced traffic sign detection and recognition."""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Initialize tracker
        self.tracker = SignTracker(max_age=config.get("max_track_age", 10))

        # Setup image transformation
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        # Define sign classes
        self.classes = ["forward", "left", "right", "stop"]

        # Color detection parameters
        self.sign_color_ranges = {
            "blue": {
                "lower": np.array([100, 150, 0]),
                "upper": np.array([140, 255, 255]),
            }
        }

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load and prepare the CNN model."""
        model = torch.load(model_path, map_location=self.device)
        model.to(self.device)
        return model

    def process(self, frame: np.ndarray) -> Dict[str, VisionFeature]:
        """
        Process a frame to detect and recognize traffic signs.
        """
        # Convert to HSV for color detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect signs by color
        sign_regions = self._detect_sign_regions(hsv_frame)

        # Recognize signs in detected regions
        detected_signs = []
        for region in sign_regions:
            sign_type, confidence = self._recognize_sign(frame, region)
            if confidence > self.config.get("confidence_threshold", 0.7):
                detected_signs.append(
                    {
                        "type": sign_type,
                        "position": (
                            region["x"] + region["width"] // 2,
                            region["y"] + region["height"] // 2,
                        ),
                        "size": (region["width"], region["height"]),
                        "confidence": confidence,
                    }
                )

        # Update tracker
        self.tracker.update(detected_signs)

        # Create SignFeatures from tracked objects
        features = {}
        for track_id, track in self.tracker.tracks.items():
            features[f"sign_{track_id}"] = SignFeature(
                timestamp=0.0,  # Should be set from frame metadata
                confidence=track.confidence,
                sign_type=track.class_label,
                position=tuple(track.position[:2]),  # x, y
                size=(int(track.size[0]), int(track.size[1])),
                orientation=0.0,  # Not implemented yet
            )

        return features

    def _detect_sign_regions(self, hsv_frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect potential sign regions based on color."""
        regions = []

        for color_name, color_range in self.sign_color_ranges.items():
            # Create color mask
            mask = cv2.inRange(hsv_frame, color_range["lower"], color_range["upper"])

            # Find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter and process contours
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.config.get("min_sign_area", 500):
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                # Filter by aspect ratio
                if not (0.7 <= aspect_ratio <= 1.3):  # Expect roughly square signs
                    continue

                regions.append(
                    {"x": x, "y": y, "width": w, "height": h, "color": color_name}
                )

        return regions

    def _recognize_sign(
        self, frame: np.ndarray, region: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Recognize the type of sign in the given region."""
        # Extract region
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]
        sign_image = frame[y : y + h, x : x + w]

        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB))

        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            sign_type = self.classes[predicted.item()]

        return sign_type, confidence.item()

    def visualize(
        self, frame: np.ndarray, features: Dict[str, VisionFeature]
    ) -> np.ndarray:
        """Visualize detected signs on frame."""
        for feature_name, feature in features.items():
            if not isinstance(feature, SignFeature):
                continue

            # Draw bounding box
            pos = feature.position
            size = feature.size
            confidence = feature.confidence

            cv2.rectangle(
                frame,
                (int(pos[0] - size[0] // 2), int(pos[1] - size[1] // 2)),
                (int(pos[0] + size[0] // 2), int(pos[1] + size[1] // 2)),
                (0, 255, 255),
                2,
            )

            # Add text info
            text = f"{feature.sign_type}: {confidence:.2f}"
            cv2.putText(
                frame,
                text,
                (int(pos[0] - size[0] // 2), int(pos[1] - size[1] // 2 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        return frame
