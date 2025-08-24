"""
Configuration for raw data storage with partitioning strategy.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


@dataclass
class StorageConfig:
    """Configuration for data lake storage organization."""
    
    # Base directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR: str = os.path.join(BASE_DIR, "raw_data")
    PARTITIONED_STORAGE: str = os.path.join(BASE_DIR, "data_lake")
    
    # Partition structure: source/type/year/month/day/
    PARTITION_LEVELS: List[str] = None
    
    # Data source mappings
    SOURCE_MAPPINGS: Dict[str, str] = None
    
    # File format preferences
    STORAGE_FORMAT: str = "parquet"  # parquet, csv, json
    COMPRESSION: str = "snappy"
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.PARTITION_LEVELS is None:
            self.PARTITION_LEVELS = ["source", "data_type", "year", "month", "day"]
        
        if self.SOURCE_MAPPINGS is None:
            self.SOURCE_MAPPINGS = {
                "kaggle_telco_churn.csv": {
                    "source": "kaggle",
                    "data_type": "customer_churn",
                    "original_name": "telco_customer_churn"
                },
                "huggingface_census.csv": {
                    "source": "huggingface", 
                    "data_type": "demographics",
                    "original_name": "adult_census_income"
                }
            }
        
        # Create directories
        os.makedirs(self.PARTITIONED_STORAGE, exist_ok=True)


# Global configuration instance
storage_config = StorageConfig()
