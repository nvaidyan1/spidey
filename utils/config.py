"""
Configuration Loader
====================
Loads settings from config.yaml
"""

import os
import yaml
from typing import Any, Dict, Optional


_config: Optional[Dict[str, Any]] = None


def load_config(path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    Caches the result for subsequent calls.
    """
    global _config
    
    if _config is not None:
        return _config
    
    if path is None:
        # Default to config.yaml in project root
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    
    if not os.path.exists(path):
        print(f"Warning: Config file not found at {path}, using defaults")
        _config = {}
        return _config
    
    with open(path, 'r') as f:
        _config = yaml.safe_load(f)
    
    return _config


def get(key: str, default: Any = None) -> Any:
    """
    Get a config value using dot notation.
    Example: get('terrain.chunk_size', 32)
    """
    config = load_config()
    
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value


def reload():
    """Force reload of config file."""
    global _config
    _config = None
    load_config()