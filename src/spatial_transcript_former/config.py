import yaml
import os
from typing import Any, Dict, List, Optional


class ProjectConfig:
    """
    Singleton wrapper for project-wide configuration.
    Loads settings from config.yaml in the project root.
    """

    _config: Dict[str, Any] = {}
    _loaded: bool = False

    @classmethod
    def load(cls, config_path: Optional[str] = None):
        """Load configuration from a YAML file."""
        if config_path is None:
            # Default to root of the project
            root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            config_path = os.path.join(root, "config.yaml")

        if not os.path.exists(config_path):
            # Fallback for when running from scripts/ or tests/
            config_path = "config.yaml"

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                try:
                    cls._config = yaml.safe_load(f) or {}
                except yaml.YAMLError as e:
                    print(f"Warning: Failed to parse config file: {e}")
                    cls._config = {}
        else:
            print(
                f"Warning: Config file not found at {config_path}. Using hardcoded defaults."
            )
            cls._config = {}

        cls._loaded = True

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a value from the configuration using dot notation (e.g., 'training.lr')."""
        if not cls._loaded:
            cls.load()

        parts = key.split(".")
        val = cls._config
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return default
        return val


def get_config(key: str, default: Any = None) -> Any:
    """Helper function to access configuration values."""
    return ProjectConfig.get(key, default)
