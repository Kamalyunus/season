"""
Configuration Loader
Loads and validates configuration from config.yaml
"""

import yaml
from pathlib import Path


class Config:
    """Configuration handler for the forecasting pipeline"""

    def __init__(self, config_path='config.yaml'):
        """Load configuration from YAML file"""
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path, default=None):
        """
        Get config value using dot notation
        Example: config.get('forecast.horizon_days')
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def __getitem__(self, key):
        """Allow dict-like access: config['forecast']"""
        return self._config[key]

    def update(self, key_path, value):
        """Update a config value"""
        keys = key_path.split('.')
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def save(self, output_path=None):
        """Save current config to file"""
        path = output_path or self.config_path

        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def __repr__(self):
        return f"Config(file='{self.config_path}')"


# Global config instance
_config = None


def load_config(config_path='config.yaml'):
    """Load global configuration"""
    global _config
    _config = Config(config_path)
    return _config


def get_config():
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
