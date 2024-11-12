# context/context_manager.py

from typing import Dict, Any
from planning.scene import SceneType, BehaviorCommand


class ContextManager:
    """Manages system context and history."""

    def __init__(self):
        self.context = {}
        self.history = []

    def update_context(
        self, features: Dict[str, Any], scene: SceneType, command: BehaviorCommand
    ):
        """Update context with new information."""
        pass

    def get_context(self) -> Dict[str, Any]:
        """Get current context."""
        return self.context
