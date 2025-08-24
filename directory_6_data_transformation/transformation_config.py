"""
Configuration for data transformation and feature engineering.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class TransformationConfig:
    """Configuration for data transformation settings."""
    
    # Base directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DATA_DIR: str = os.path.join(BASE_DIR, "processed_data")
    TRANSFORMED_DATA_DIR: str = os.path.join(BASE_DIR, "transformed_data")
    FEATURE_DEFINITIONS_DIR: str = os.path.join(BASE_DIR, "feature_definitions")
    
    # Feature engineering configurations
    FEATURE_ENGINEERING_RULES: Dict[str, Dict] = None
    
    # Aggregation rules
    AGGREGATION_RULES: Dict[str, Dict] = None
    
    # Derived feature rules
    DERIVED_FEATURES: Dict[str, Dict] = None
    
    # Transformation output formats
    OUTPUT_FORMATS: List[str] = None
    
    def __post_init__(self):
        """Initialize default transformation configurations."""
        if self.FEATURE_ENGINEERING_RULES is None:
            self.FEATURE_ENGINEERING_RULES = {
                "telco_customer_churn": {
                    "target_column": "Churn",
                    "customer_id_column": "customerID",
                    "numerical_features": [
                        "tenure", "MonthlyCharges", "TotalCharges"
                    ],
                    "categorical_features": [
                        "gender", "SeniorCitizen", "Partner", "Dependents",
                        "PhoneService", "InternetService", "Contract", "PaymentMethod"
                    ],
                    "derived_features": {
                        "tenure_groups": {
                            "type": "binning",
                            "source_column": "tenure",
                            "bins": [0, 12, 24, 48, 72],
                            "labels": ["New", "1-2Years", "2-4Years", "4-6Years"]
                        },
                        "charges_per_tenure": {
                            "type": "ratio",
                            "numerator": "TotalCharges",
                            "denominator": "tenure",
                            "handle_zero": "replace_with_monthly"
                        },
                        "charges_ratio": {
                            "type": "ratio",
                            "numerator": "TotalCharges",
                            "denominator": "MonthlyCharges"
                        },
                        "senior_partner": {
                            "type": "interaction",
                            "columns": ["SeniorCitizen", "Partner"],
                            "operation": "multiply"
                        },
                        "high_charges_flag": {
                            "type": "threshold",
                            "source_column": "MonthlyCharges",
                            "threshold": 70,
                            "operation": "greater_than"
                        },
                        "long_tenure_flag": {
                            "type": "threshold",
                            "source_column": "tenure",
                            "threshold": 24,
                            "operation": "greater_than"
                        }
                    }
                },
                "adult_census_income": {
                    "target_column": "income",
                    "numerical_features": [
                        "age", "education-num", "capital-gain", "capital-loss", "hours-per-week"
                    ],
                    "categorical_features": [
                        "workclass", "education", "marital-status", "occupation",
                        "relationship", "race", "sex", "native-country"
                    ],
                    "derived_features": {
                        "age_groups": {
                            "type": "binning",
                            "source_column": "age",
                            "bins": [17, 25, 35, 50, 65, 90],
                            "labels": ["Young", "Adult", "MiddleAge", "Senior", "Elder"]
                        },
                        "capital_net": {
                            "type": "arithmetic",
                            "operation": "subtract",
                            "columns": ["capital-gain", "capital-loss"]
                        },
                        "hours_groups": {
                            "type": "binning",
                            "source_column": "hours-per-week",
                            "bins": [0, 20, 40, 60, 99],
                            "labels": ["PartTime", "FullTime", "Overtime", "Excessive"]
                        },
                        "education_experience": {
                            "type": "interaction",
                            "columns": ["education-num", "age"],
                            "operation": "multiply"
                        },
                        "high_earner_potential": {
                            "type": "composite",
                            "conditions": [
                                {"column": "education-num", "operator": ">=", "value": 13},
                                {"column": "hours-per-week", "operator": ">=", "value": 40},
                                {"column": "age", "operator": ">=", "value": 25}
                            ],
                            "logic": "all"
                        }
                    }
                }
            }
        
        if self.AGGREGATION_RULES is None:
            self.AGGREGATION_RULES = {
                "numerical_aggregations": ["mean", "std", "min", "max", "median"],
                "categorical_aggregations": ["mode", "nunique", "count"],
                "time_windows": ["daily", "weekly", "monthly"],
                "groupby_columns": {
                    "telco_customer_churn": ["Contract", "InternetService"],
                    "adult_census_income": ["workclass", "education", "occupation"]
                }
            }
        
        if self.DERIVED_FEATURES is None:
            self.DERIVED_FEATURES = {
                "polynomial_features": {
                    "enabled": True,
                    "degree": 2,
                    "max_features": 10  # Limit to prevent explosion
                },
                "interaction_features": {
                    "enabled": True,
                    "max_interactions": 5
                },
                "statistical_features": {
                    "enabled": True,
                    "features": ["z_score", "percentile_rank", "rolling_mean"]
                }
            }
        
        if self.OUTPUT_FORMATS is None:
            self.OUTPUT_FORMATS = ["csv", "parquet", "json"]
        
        # Create directories
        for directory in [self.TRANSFORMED_DATA_DIR, self.FEATURE_DEFINITIONS_DIR]:
            os.makedirs(directory, exist_ok=True)
            os.makedirs(os.path.join(directory, "features"), exist_ok=True)
            os.makedirs(os.path.join(directory, "metadata"), exist_ok=True)


# Global configuration instance
transformation_config = TransformationConfig()
