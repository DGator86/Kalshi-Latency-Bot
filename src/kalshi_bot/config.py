"""Configuration management."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load()
        
    def load(self):
        """Load configuration from file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            logger.warning("Using default configuration")
            self.config = self.get_default_config()
            return
            
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = self.get_default_config()
            
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "kalshi": {
                "api_base": "https://trading-api.kalshi.com",
                "api_key": "",
                "api_secret": ""
            },
            "trading": {
                "edge_threshold": 0.03,
                "kelly_fraction": 0.25,
                "max_daily_loss": 1000,
                "max_position_size": 100,
                "target_markets": ["BTC", "ETH"]
            },
            "exchanges": {
                "binance": {
                    "enabled": True,
                    "ws_url": "wss://stream.binance.com:9443/ws"
                },
                "coinbase": {
                    "enabled": True,
                    "ws_url": "wss://ws-feed.exchange.coinbase.com"
                },
                "kraken": {
                    "enabled": True,
                    "ws_url": "wss://ws.kraken.com"
                }
            },
            "monitoring": {
                "dashboard_port": 8080,
                "log_level": "INFO"
            }
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def save(self):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
