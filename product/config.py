"""
Configuration module for the product matching workflow.
"""

import os
from typing import Dict, Any

# Default configuration values
DEFAULT_CONFIG = {
    # URL Processing
    'url_processing': {
        'max_depth': 5,
        'max_workers': 8,
        'chunk_size': 10000,
        'min_url_length': 5,
    },
    
    # Data Retrieval
    'data_retrieval': {
        'clickhouse_timeout': 30,
        'retry_attempts': 3,
        'batch_size': 1000,
    },
    
    # Product Classification
    'product_classification': {
        'n_workers': 4,
        'chunk_size': 10000,
        'prefer': 'productGroupId',
    },
    
    # Metrics Coalescing
    'metrics_coalescing': {
        'tolerance': 1e-9,
        'residual_product_id': '__unmapped__',
    },
    
    # Ad Targeting
    'ad_targeting': {
        'min_token_len': 3,
        'fuzzy_threshold': 85,
        'fuzzy_limit': 30,
        'keep_top_n': 50,
    },
    
    # Memory Management
    'memory_management': {
        'enable_gc': True,
        'gc_frequency': 1000,  # Perform GC every N operations
    },
    
    # Logging
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'enable_file_logging': False,
        'log_file': 'product_matching.log',
    }
}


class Config:
    """Configuration manager for the product matching workflow."""
    
    def __init__(self):
        self._config = DEFAULT_CONFIG.copy()
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # URL Processing
        self._update_config_value('url_processing.max_depth', 'URL_MAX_DEPTH', int)
        self._update_config_value('url_processing.max_workers', 'URL_MAX_WORKERS', int)
        self._update_config_value('url_processing.chunk_size', 'URL_CHUNK_SIZE', int)
        
        # Data Retrieval
        self._update_config_value('data_retrieval.clickhouse_timeout', 'CLICKHOUSE_TIMEOUT', int)
        self._update_config_value('data_retrieval.retry_attempts', 'RETRY_ATTEMPTS', int)
        
        # Product Classification
        self._update_config_value('product_classification.n_workers', 'CLASSIFICATION_WORKERS', int)
        self._update_config_value('product_classification.chunk_size', 'CLASSIFICATION_CHUNK_SIZE', int)
        
        # Metrics Coalescing
        self._update_config_value('metrics_coalescing.tolerance', 'COALESCING_TOLERANCE', float)
        
        # Ad Targeting
        self._update_config_value('ad_targeting.min_token_len', 'MIN_TOKEN_LEN', int)
        self._update_config_value('ad_targeting.fuzzy_threshold', 'FUZZY_THRESHOLD', int)
        
        # Logging
        self._update_config_value('logging.level', 'LOG_LEVEL', str)
        self._update_config_value('logging.enable_file_logging', 'ENABLE_FILE_LOGGING', bool)
    
    def _update_config_value(self, path: str, env_var: str, converter: type):
        """Update a config value from environment variable."""
        value = os.getenv(env_var)
        if value is not None:
            try:
                # Navigate to the nested dictionary
                keys = path.split('.')
                config_section = self._config
                for key in keys[:-1]:
                    config_section = config_section[key]
                
                # Convert and set the value
                converted_value = converter(value)
                config_section[keys[-1]] = converted_value
            except (ValueError, TypeError):
                # If conversion fails, keep the default value
                pass
    
    def get(self, path: str, default=None):
        """Get a configuration value using dot notation."""
        keys = path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, path: str, value: Any):
        """Set a configuration value using dot notation."""
        keys = path.split('.')
        config_section = self._config
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        config_section[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with a dictionary of changes."""
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
        deep_update(self._config, updates)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config