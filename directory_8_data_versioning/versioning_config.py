"""
Configuration for data versioning with DVC.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass 
class VersioningConfig:
    """Configuration for data versioning settings."""
    
    # Base directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VERSIONING_DIR: str = os.path.join(BASE_DIR, "data_versions")
    
    # Data directories to version
    DATA_DIRECTORIES: List[str] = None
    
    # DVC configuration
    DVC_CONFIG: Dict[str, Any] = None
    
    # Git configuration
    GIT_CONFIG: Dict[str, Any] = None
    
    # Version metadata configuration
    VERSION_METADATA: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.DATA_DIRECTORIES is None:
            self.DATA_DIRECTORIES = [
                "raw_data",
                "processed_data", 
                "transformed_data",
                "feature_store",
                "models"
            ]
        
        if self.DVC_CONFIG is None:
            self.DVC_CONFIG = {
                "cache_dir": os.path.join(self.BASE_DIR, ".dvc", "cache"),
                "remote_storage": {
                    "name": "local_remote",
                    "type": "local",
                    "path": os.path.join(self.VERSIONING_DIR, "remote_storage")
                },
                "metrics_file": "metrics.json",
                "plots_dir": "plots",
                "experiments_dir": "experiments"
            }
        
        if self.GIT_CONFIG is None:
            self.GIT_CONFIG = {
                "user_name": "ML Pipeline User",
                "user_email": "ml-pipeline@dmml.local",
                "default_branch": "main",
                "commit_message_template": "DVC: Version {version} - {description}"
            }
        
        if self.VERSION_METADATA is None:
            self.VERSION_METADATA = {
                "version_format": "v{year}.{month}.{day}.{hour}{minute}",
                "metadata_file": "version_metadata.json",
                "changelog_file": "CHANGELOG.md",
                "version_tags": ["data", "model", "experiment", "feature"]
            }
        
        # Create directories
        os.makedirs(self.VERSIONING_DIR, exist_ok=True)
        os.makedirs(self.DVC_CONFIG["remote_storage"]["path"], exist_ok=True)


# Global configuration instance
versioning_config = VersioningConfig()
