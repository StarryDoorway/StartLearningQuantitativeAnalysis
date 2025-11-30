"""
Configuration loader module for the quantitative trading framework.

This module provides functionality to load and manage configuration settings
from YAML files, with support for environment-specific configurations.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """
    Configuration loader class that handles loading and accessing configuration settings.
    
    This class supports loading configuration from YAML files, with support for
    environment-specific overrides and secret management.
    """
    
    def __init__(self, config_dir: str = "config", environment: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Environment name (dev, test, prod) for environment-specific configs
        """
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv("QUANT_ENV", "dev")
        self._config = {}
        self._secrets = {}
        
        # Load configuration
        self._load_config()
        self._load_secrets()
    
    def _load_config(self) -> None:
        """Load the main configuration file."""
        config_path = self.config_dir / "trading_config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            self._config = yaml.safe_load(file)
        
        # Load environment-specific overrides if they exist
        env_config_path = self.config_dir / f"config.{self.environment}.yaml"
        if env_config_path.exists():
            with open(env_config_path, 'r') as file:
                env_config = yaml.safe_load(file)
                self._merge_config(self._config, env_config)
    
    def _load_secrets(self) -> None:
        """Load the secrets configuration file."""
        secrets_path = self.config_dir / "secrets.yaml"
        
        if not secrets_path.exists():
            # Try to load from example file if secrets.yaml doesn't exist
            secrets_path = self.config_dir / "config.yaml"
            
            if not secrets_path.exists():
                print(f"Warning: Secrets file not found. Using empty secrets.")
                self._secrets = {}
                return
            
            print(f"Warning: Using example secrets file. Please create a proper secrets.yaml file.")
        
        with open(secrets_path, 'r') as file:
            self._secrets = yaml.safe_load(file)
    
    def _merge_config(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> None:
        """
        Recursively merge override configuration into base configuration.
        
        Args:
            base_config: Base configuration dictionary to merge into
            override_config: Override configuration dictionary
        """
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'data.base_dir')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_secret(self, key: str, default: Any = None) -> Any:
        """
        Get a secret value by key.
        
        Args:
            key: Secret key (supports dot notation, e.g., 'clients.okx.api_key')
            default: Default value if key is not found
            
        Returns:
            Secret value or default
        """
        keys = key.split('.')
        value = self._secrets
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_exchange_config(self, exchange_name: str) -> Dict[str, Any]:
        """
        Get exchange-specific configuration.
        
        Args:
            exchange_name: Name of the exchange (e.g., 'okx', 'binance')
            
        Returns:
            Dictionary containing exchange configuration
        """
        exchange_config = self.get_secret(f"clients.{exchange_name}", {})
        
        # Add common exchange settings from main config if available
        common_config = self.get("clients.common", {})
        
        # Merge common config with exchange-specific config
        result = {**common_config, **exchange_config}
        
        return result
    
    def validate_config(self) -> bool:
        """
        Validate the configuration for required fields.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_keys = [
            "framework.version",
            "data.base_dir",
            "backtest.initial_cash",
            "risk.max_daily_loss_percent"
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                print(f"Error: Required configuration key missing: {key}")
                return False
        
        return True
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._config
    
    @property
    def secrets(self) -> Dict[str, Any]:
        """Get the full secrets dictionary."""
        return self._secrets


# Global configuration instance
_config_loader = None


def get_config(config_dir: str = "config", environment: Optional[str] = None) -> ConfigLoader:
    """
    Get the global configuration loader instance.
    
    Args:
        config_dir: Directory containing configuration files
        environment: Environment name (dev, test, prod)
        
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir, environment)
    
    return _config_loader


def reset_config() -> None:
    """Reset the global configuration loader instance."""
    global _config_loader
    _config_loader = None