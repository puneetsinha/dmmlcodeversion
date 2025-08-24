"""
Configuration file for data ingestion pipeline.
Student IDs: 2024ab05134, 2024aa05664

This config file contains all the settings we need for downloading data
from different sources. We chose Kaggle and HuggingFace because they have
good datasets for our churn prediction project.

Note: Had to implement fallback URLs because kaggle authentication was
giving us trouble on the university servers - learned this the hard way!
"""

import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class IngestionConfig:
    """Configuration class for data ingestion settings."""
    
    # Base directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR: str = os.path.join(BASE_DIR, "raw_data")
    LOGS_DIR: str = os.path.join(BASE_DIR, "logs")
    
    # Kaggle settings
    KAGGLE_DATASETS: List[Dict[str, str]] = None
    
    # Hugging Face settings
    HUGGINGFACE_DATASETS: List[Dict[str, str]] = None
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Initialize default dataset configurations."""
        if self.KAGGLE_DATASETS is None:
            self.KAGGLE_DATASETS = [
                {
                    "name": "blastchar/telco-customer-churn", 
                    "file_name": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
                    "output_name": "kaggle_telco_churn.csv",
                    "fallback_url": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
                }
            ]
        
        if self.HUGGINGFACE_DATASETS is None:
            self.HUGGINGFACE_DATASETS = [
                {
                    "name": "scikit-learn/adult-census-income",
                    "output_name": "huggingface_census.csv",
                    "config": "default",
                    "split": "train"
                }
            ]
        
        # Create directories if they don't exist
        os.makedirs(self.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)


# Global configuration instance
config = IngestionConfig()
