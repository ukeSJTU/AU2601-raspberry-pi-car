# debug/data_logger.py

import numpy as np
from typing import Dict, Any


class DataLogger:
    """Records system data for debugging."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir

    def log_data(self, data: Dict[str, Any]):
        """Log data point."""
        pass

    def save_video(self, frame: np.ndarray, features: Dict[str, Any]):
        """Save annotated video frame."""
        pass
