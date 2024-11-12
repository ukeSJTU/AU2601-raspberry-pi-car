# main.py


import cv2
import numpy as np
from typing import Any, Dict
from perception.lane_detector import LaneDetector
from perception.sign_detector import SignDetector
from perception.feature_fusion import FeatureFusion
from planning.scene_manager import SceneManager
from planning.behavior_planner import BehaviorPlanner
from planning.path_planner import PathPlanner
from control.trajectory_generator import TrajectoryGenerator
from control.controller import Controller
from context.context_manager import ContextManager
from debug.data_logger import DataLogger
from control.controller import ControlCommand
from new_driver import driver
import argparse
import asyncio


class AutonomousDrivingSystem:
    """Main system class."""

    def __init__(self, config: Dict[str, Any]):
        # Initialize all components
        self.lane_detector = LaneDetector(config["lane_detector"])
        self.sign_detector = SignDetector(
            config["sign_detector"]["model_path"], config["sign_detector"]
        )
        self.feature_fusion = FeatureFusion()
        self.scene_manager = SceneManager()
        self.behavior_planner = BehaviorPlanner()
        self.path_planner = PathPlanner()
        self.trajectory_generator = TrajectoryGenerator()
        self.controller = Controller(config["controller"])
        self.context_manager = ContextManager()
        self.data_logger = DataLogger(config["log_dir"])

    def process_frame(self, frame: np.ndarray) -> ControlCommand:
        # Perception
        lane_features = self.lane_detector.process(frame)
        sign_features = self.sign_detector.process(frame)
        fused_features = self.feature_fusion.fuse_features(
            {"lane": lane_features, "sign": sign_features}
        )

        # Planning
        context = self.context_manager.get_context()
        scene = self.scene_manager.update_scene(fused_features, context)
        behavior = self.behavior_planner.plan(scene, fused_features, context)
        path = self.path_planner.plan_path(behavior, fused_features)

        # Control
        trajectory = self.trajectory_generator.generate(path)
        control = self.controller.compute_control(
            trajectory[0], trajectory[1]  # current point  # next target point
        )

        # Update context
        self.context_manager.update_context(fused_features, scene, behavior)

        # Debug logging
        self.data_logger.log_data(
            {
                "features": fused_features,
                "scene": scene,
                "behavior": behavior,
                "control": control,
            }
        )

        return control


def apply_control(control: ControlCommand, car: driver) -> None:
    """
    Apply control command to the car.

    Args:
        control: ControlCommand containing forward_speed, steering_angle and lateral_speed
        car: driver instance for controlling the car
    """
    # Convert control command to x, y, w format
    x = int(control.forward_speed)
    y = int(control.lateral_speed)
    w = int(control.steering_angle)

    # Apply speed command to car
    car.set_speed(x, y, w)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Autonomous Driving System")

    # Original CLI arguments
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--video_source", default=0, help="Video source (0 for webcam, or file path)"
    )
    parser.add_argument(
        "--max_offset",
        type=int,
        default=50,
        help="Maximum allowed offset before warning",
    )
    parser.add_argument(
        "--threshold", type=int, default=200, help="Threshold for binary image"
    )
    parser.add_argument(
        "--strategy",
        choices=["single", "double", "manual", "cnn"],
        default="single",
        help="Driving strategy",
    )
    parser.add_argument("--record", action="store_true", help="Record video")
    parser.add_argument(
        "--record_filename", type=str, help="Filename for recorded video"
    )
    parser.add_argument("--model_path", type=str, help="Path to the trained CNN model")

    # New CLI arguments
    parser.add_argument(
        "--input_mode",
        choices=["auto", "manual", "hybrid"],
        default="auto",
        help="Initial input control mode",
    )
    parser.add_argument(
        "--base_speed", type=int, default=70, help="Base speed for manual control"
    )
    parser.add_argument(
        "--speed_increment",
        type=float,
        default=10.0,
        help="Speed increment for manual control",
    )
    parser.add_argument(
        "--steering_increment",
        type=float,
        default=5.0,
        help="Steering increment for manual control",
    )

    return parser.parse_args()


async def main():
    """Main function with async support."""
    args = parse_args()

    # Convert args to config dict
    config = {
        "lane_detector": {
            "threshold": args.threshold,
            "roi_width": 400,
            "roi_height": 100,
        },
        "sign_detector": {"model_path": args.model_path, "confidence_threshold": 0.8},
        "controller": {
            "max_speed": 100,
            "max_steering": 45,
            "input_manager": {
                "base_speed": args.base_speed,
                "speed_increment": args.speed_increment,
                "steering_increment": args.steering_increment,
            },
        },
        "log_dir": "logs",
        "debug": args.debug,
        "record": args.record,
        "record_filename": args.record_filename,
        "video_source": args.video_source,
        "max_offset": args.max_offset,
        "strategy": args.strategy,
    }

    # Initialize system
    system = AutonomousDrivingSystem(config)

    # Start controller with input management
    await system.controller.start()

    # Initialize video
    video = cv2.VideoCapture(config["video_source"])

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Process frame
            control = system.process_frame(frame)

            # Apply control
            apply_control(control)

            # Display debug visualization
            cv2.imshow("Debug View", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
