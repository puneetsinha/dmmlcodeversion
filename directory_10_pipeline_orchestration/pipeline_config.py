"""
Configuration for pipeline orchestration.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class PipelineConfig:
    """Configuration for pipeline orchestration settings."""
    
    # Base directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PIPELINE_LOGS_DIR: str = os.path.join(BASE_DIR, "pipeline_logs")
    PIPELINE_REPORTS_DIR: str = os.path.join(BASE_DIR, "pipeline_reports")
    
    # Pipeline steps configuration
    PIPELINE_STEPS: List[Dict[str, Any]] = None
    
    # Execution settings
    EXECUTION_CONFIG: Dict[str, Any] = None
    
    # Monitoring and alerting
    MONITORING_CONFIG: Dict[str, Any] = None
    
    # Scheduling configuration
    SCHEDULING_CONFIG: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.PIPELINE_STEPS is None:
            self.PIPELINE_STEPS = [
                {
                    "step_id": "data_ingestion",
                    "name": "Data Ingestion",
                    "description": "Ingest data from Kaggle and Hugging Face",
                    "module_path": "directory_2_data_ingestion.main_ingestion",
                    "function": "main",
                    "enabled": True,
                    "dependencies": [],
                    "timeout_minutes": 30,
                    "retry_count": 3,
                    "critical": True
                },
                {
                    "step_id": "raw_data_storage",
                    "name": "Raw Data Storage",
                    "description": "Organize data into data lake structure",
                    "module_path": "directory_3_raw_data_storage.data_lake_organizer",
                    "function": "main",
                    "enabled": True,
                    "dependencies": ["data_ingestion"],
                    "timeout_minutes": 15,
                    "retry_count": 2,
                    "critical": True
                },
                {
                    "step_id": "data_validation",
                    "name": "Data Validation",
                    "description": "Validate data quality and generate reports",
                    "module_path": "directory_4_data_validation.validation_orchestrator",
                    "function": "main",
                    "enabled": True,
                    "dependencies": ["raw_data_storage"],
                    "timeout_minutes": 20,
                    "retry_count": 2,
                    "critical": True
                },
                {
                    "step_id": "data_preparation",
                    "name": "Data Preparation",
                    "description": "Clean data and perform EDA",
                    "module_path": "directory_5_data_preparation.preparation_orchestrator",
                    "function": "main",
                    "enabled": True,
                    "dependencies": ["data_validation"],
                    "timeout_minutes": 45,
                    "retry_count": 2,
                    "critical": True
                },
                {
                    "step_id": "data_transformation",
                    "name": "Data Transformation",
                    "description": "Feature engineering and transformation",
                    "module_path": "directory_6_data_transformation.transformation_orchestrator",
                    "function": "main",
                    "enabled": True,
                    "dependencies": ["data_preparation"],
                    "timeout_minutes": 60,
                    "retry_count": 2,
                    "critical": True
                },
                {
                    "step_id": "feature_store",
                    "name": "Feature Store",
                    "description": "Register features in centralized store",
                    "module_path": "directory_7_feature_store.feature_store_orchestrator",
                    "function": "main",
                    "enabled": True,
                    "dependencies": ["data_transformation"],
                    "timeout_minutes": 30,
                    "retry_count": 2,
                    "critical": True
                },
                {
                    "step_id": "data_versioning",
                    "name": "Data Versioning",
                    "description": "Version control with DVC and Git",
                    "module_path": "directory_8_data_versioning.versioning_orchestrator",
                    "function": "main",
                    "enabled": False,  # Optional due to system dependencies
                    "dependencies": ["feature_store"],
                    "timeout_minutes": 20,
                    "retry_count": 1,
                    "critical": False
                },
                {
                    "step_id": "model_building",
                    "name": "Model Building",
                    "description": "Train and evaluate ML models",
                    "module_path": "directory_9_model_building.model_orchestrator",
                    "function": "main",
                    "enabled": True,
                    "dependencies": ["feature_store"],
                    "timeout_minutes": 180,  # 3 hours for model training
                    "retry_count": 1,
                    "critical": True
                }
            ]
        
        if self.EXECUTION_CONFIG is None:
            self.EXECUTION_CONFIG = {
                "parallel_execution": False,  # Sequential by default
                "max_parallel_workers": 2,
                "continue_on_non_critical_failure": True,
                "save_intermediate_results": True,
                "cleanup_on_failure": False,
                "log_level": "INFO",
                "enable_profiling": True
            }
        
        if self.MONITORING_CONFIG is None:
            self.MONITORING_CONFIG = {
                "enable_monitoring": True,
                "metrics_collection": True,
                "performance_tracking": True,
                "resource_monitoring": True,
                "alert_on_failure": True,
                "alert_on_long_execution": True,
                "max_execution_hours": 6
            }
        
        if self.SCHEDULING_CONFIG is None:
            self.SCHEDULING_CONFIG = {
                "enable_scheduling": False,
                "schedule_type": "cron",  # cron, interval
                "cron_expression": "0 2 * * 1",  # Weekly on Monday at 2 AM
                "interval_hours": 168,  # Weekly
                "timezone": "UTC",
                "enable_manual_trigger": True
            }
        
        # Create directories
        for directory in [self.PIPELINE_LOGS_DIR, self.PIPELINE_REPORTS_DIR]:
            os.makedirs(directory, exist_ok=True)
            os.makedirs(os.path.join(directory, "step_logs"), exist_ok=True)
            os.makedirs(os.path.join(directory, "execution_reports"), exist_ok=True)
            os.makedirs(os.path.join(directory, "monitoring"), exist_ok=True)


# Global configuration instance
pipeline_config = PipelineConfig()
