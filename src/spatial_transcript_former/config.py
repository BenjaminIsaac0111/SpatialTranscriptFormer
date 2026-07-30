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

        # Overlay an untracked local override, if present. Keeps
        # machine-specific absolute paths (e.g. a Windows data drive) out of the
        # tracked config, where they become every contributor's -- and CI's --
        # default. On Linux a path like "A:\hest_data" is a legal *filename*,
        # so it fails silently by creating a strangely-named directory rather
        # than erroring.
        local_path = os.path.join(os.path.dirname(config_path), "config.local.yaml")
        if os.path.exists(local_path):
            try:
                with open(local_path, "r") as f:
                    overrides = yaml.safe_load(f) or {}
                cls._config = cls._deep_merge(cls._config, overrides)
            except yaml.YAMLError as e:
                print(f"Warning: Failed to parse config.local.yaml: {e}")

        cls._loaded = True

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge ``override`` into ``base`` (override wins)."""
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = ProjectConfig._deep_merge(out[k], v)
            else:
                out[k] = v
        return out

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
