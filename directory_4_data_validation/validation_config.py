"""
Configuration for data validation pipeline.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class ValidationConfig:
    """Configuration for data validation settings."""
    
    # Base directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_LAKE_DIR: str = os.path.join(BASE_DIR, "directory_3_raw_data_storage", "data_lake")
    VALIDATION_REPORTS_DIR: str = os.path.join(BASE_DIR, "validation_reports")
    
    # Validation rules for different datasets
    DATASET_RULES: Dict[str, Dict] = None
    
    # Quality thresholds
    COMPLETENESS_THRESHOLD: float = 0.95  # 95% non-null values
    UNIQUENESS_THRESHOLD: float = 0.90    # 90% unique values for ID columns
    CONSISTENCY_THRESHOLD: float = 0.95   # 95% consistent data types
    
    # Anomaly detection settings
    OUTLIER_METHODS: List[str] = None
    STATISTICAL_TESTS: List[str] = None
    
    def __post_init__(self):
        """Initialize default validation rules."""
        if self.DATASET_RULES is None:
            self.DATASET_RULES = {
                "telco_customer_churn": {
                    "required_columns": [
                        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
                        "tenure", "PhoneService", "MultipleLines", "InternetService",
                        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
                        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
                    ],
                    "categorical_columns": [
                        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                        "PaperlessBilling", "PaymentMethod", "Churn"
                    ],
                    "numerical_columns": ["tenure", "MonthlyCharges", "TotalCharges"],
                    "id_columns": ["customerID"],
                    "target_column": "Churn",
                    "data_types": {
                        "customerID": "object",
                        "gender": "object",
                        "SeniorCitizen": "int64",
                        "tenure": "int64",
                        "MonthlyCharges": "float64",
                        "TotalCharges": "object",  # Often has spaces, needs cleaning
                        "Churn": "object"
                    },
                    "value_ranges": {
                        "tenure": {"min": 0, "max": 100},
                        "MonthlyCharges": {"min": 0, "max": 200},
                        "SeniorCitizen": {"min": 0, "max": 1}
                    },
                    "categorical_values": {
                        "gender": ["Male", "Female"],
                        "Churn": ["Yes", "No"],
                        "Partner": ["Yes", "No"],
                        "Dependents": ["Yes", "No"]
                    }
                },
                "adult_census_income": {
                    "required_columns": [
                        "age", "workclass", "fnlwgt", "education", "education-num",
                        "marital-status", "occupation", "relationship", "race", "sex",
                        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
                    ],
                    "categorical_columns": [
                        "workclass", "education", "marital-status", "occupation",
                        "relationship", "race", "sex", "native-country", "income"
                    ],
                    "numerical_columns": [
                        "age", "fnlwgt", "education-num", "capital-gain",
                        "capital-loss", "hours-per-week"
                    ],
                    "target_column": "income",
                    "data_types": {
                        "age": "int64",
                        "fnlwgt": "int64",
                        "education-num": "int64",
                        "capital-gain": "int64",
                        "capital-loss": "int64",
                        "hours-per-week": "int64",
                        "workclass": "object",
                        "education": "object",
                        "income": "object"
                    },
                    "value_ranges": {
                        "age": {"min": 17, "max": 90},
                        "education-num": {"min": 1, "max": 16},
                        "hours-per-week": {"min": 1, "max": 99}
                    },
                    "categorical_values": {
                        "sex": ["Male", "Female"],
                        "income": ["<=50K", ">50K"]
                    }
                }
            }
        
        if self.OUTLIER_METHODS is None:
            self.OUTLIER_METHODS = ["iqr", "z_score", "isolation_forest"]
        
        if self.STATISTICAL_TESTS is None:
            self.STATISTICAL_TESTS = ["normality", "correlation", "chi_square"]
        
        # Create directories
        os.makedirs(self.VALIDATION_REPORTS_DIR, exist_ok=True)


# Global configuration instance
validation_config = ValidationConfig()
