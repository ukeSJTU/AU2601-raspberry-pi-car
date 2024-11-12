# config/config_manager.py
import yaml
from typing import Dict, Any


class ConfigManager:
    """Manages system configuration."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """Get configuration for specific module."""
        return self.config.get(module_name, {})
