"""
Configuration for Feature Store implementation.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any
from datetime import datetime


@dataclass
class FeatureStoreConfig:
    """Configuration for feature store settings."""
    
    # Base directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRANSFORMED_DATA_DIR: str = os.path.join(BASE_DIR, "transformed_data")
    FEATURE_STORE_DIR: str = os.path.join(BASE_DIR, "feature_store")
    FEATURE_METADATA_DIR: str = os.path.join(FEATURE_STORE_DIR, "metadata")
    FEATURE_REGISTRY_DIR: str = os.path.join(FEATURE_STORE_DIR, "registry")
    
    # Database settings for feature store
    DATABASE_CONFIG: Dict[str, Any] = None
    
    # Feature versioning settings
    VERSIONING_CONFIG: Dict[str, Any] = None
    
    # Feature serving settings
    SERVING_CONFIG: Dict[str, Any] = None
    
    # Data quality rules for features
    FEATURE_QUALITY_RULES: Dict[str, Dict] = None
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.DATABASE_CONFIG is None:
            self.DATABASE_CONFIG = {
                "type": "sqlite",
                "database_path": os.path.join(self.FEATURE_STORE_DIR, "feature_store.db"),
                "connection_pool_size": 5,
                "timeout": 30
            }
        
        if self.VERSIONING_CONFIG is None:
            self.VERSIONING_CONFIG = {
                "versioning_strategy": "timestamp",  # timestamp, semantic, hash
                "max_versions_per_feature": 10,
                "auto_cleanup": True,
                "retention_days": 90
            }
        
        if self.SERVING_CONFIG is None:
            self.SERVING_CONFIG = {
                "serving_types": ["batch", "online", "streaming"],
                "default_serving_type": "batch",
                "cache_enabled": True,
                "cache_ttl_seconds": 3600,
                "api_enabled": True,
                "api_port": 8080
            }
        
        if self.FEATURE_QUALITY_RULES is None:
            self.FEATURE_QUALITY_RULES = {
                "data_quality_checks": {
                    "completeness": {"min_threshold": 0.95},
                    "uniqueness": {"max_duplicate_ratio": 0.05},
                    "consistency": {"data_type_stability": True},
                    "validity": {"range_checks": True}
                },
                "statistical_monitoring": {
                    "distribution_drift": {"enabled": True, "threshold": 0.1},
                    "statistical_tests": ["ks_test", "chi_square"],
                    "outlier_detection": {"method": "iqr", "threshold": 3.0}
                },
                "business_rules": {
                    "domain_constraints": True,
                    "logical_constraints": True,
                    "temporal_constraints": True
                }
            }
        
        # Create directories
        for directory in [
            self.FEATURE_STORE_DIR, 
            self.FEATURE_METADATA_DIR, 
            self.FEATURE_REGISTRY_DIR
        ]:
            os.makedirs(directory, exist_ok=True)
            os.makedirs(os.path.join(directory, "tables"), exist_ok=True)
            os.makedirs(os.path.join(directory, "views"), exist_ok=True)
            os.makedirs(os.path.join(directory, "versions"), exist_ok=True)


# Global configuration instance
feature_store_config = FeatureStoreConfig()
