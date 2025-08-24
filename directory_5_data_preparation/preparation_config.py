"""
Configuration for data preparation pipeline.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class PreparationConfig:
    """Configuration for data preparation settings."""
    
    # Base directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_LAKE_DIR: str = os.path.join(BASE_DIR, "data_lake")
    PROCESSED_DATA_DIR: str = os.path.join(BASE_DIR, "processed_data")
    EDA_REPORTS_DIR: str = os.path.join(BASE_DIR, "eda_reports")
    
    # Dataset-specific cleaning rules
    CLEANING_RULES: Dict[str, Dict] = None
    
    # Missing value handling strategies
    MISSING_VALUE_STRATEGIES: Dict[str, str] = None
    
    # Encoding strategies
    ENCODING_STRATEGIES: Dict[str, str] = None
    
    # EDA configuration
    EDA_CONFIG: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.CLEANING_RULES is None:
            self.CLEANING_RULES = {
                "telco_customer_churn": {
                    "columns_to_clean": {
                        "TotalCharges": {
                            "type": "numeric_conversion",
                            "errors": "coerce",
                            "replace_with_zero": True
                        },
                        "customerID": {
                            "type": "string_strip",
                            "remove_duplicates": True
                        }
                    },
                    "categorical_mappings": {
                        "SeniorCitizen": {0: "No", 1: "Yes"},
                        "Churn": {"Yes": 1, "No": 0},
                        "Partner": {"Yes": 1, "No": 0},
                        "Dependents": {"Yes": 1, "No": 0},
                        "PhoneService": {"Yes": 1, "No": 0},
                        "PaperlessBilling": {"Yes": 1, "No": 0}
                    },
                    "columns_to_drop": [],
                    "outlier_treatment": {
                        "tenure": {"method": "cap", "lower": 0, "upper": 72},
                        "MonthlyCharges": {"method": "cap", "lower": 0, "upper": 200},
                        "TotalCharges": {"method": "cap", "lower": 0, "upper": 10000}
                    }
                },
                "adult_census_income": {
                    "columns_to_clean": {
                        "workclass": {
                            "type": "replace_values",
                            "mappings": {" ?": "Unknown"}
                        },
                        "occupation": {
                            "type": "replace_values", 
                            "mappings": {" ?": "Unknown"}
                        },
                        "native-country": {
                            "type": "replace_values",
                            "mappings": {" ?": "Unknown"}
                        }
                    },
                    "categorical_mappings": {
                        "income": {" <=50K": 0, " >50K": 1, "<=50K": 0, ">50K": 1},
                        "sex": {" Male": 1, " Female": 0, "Male": 1, "Female": 0}
                    },
                    "columns_to_drop": ["fnlwgt"],  # Less relevant for analysis
                    "outlier_treatment": {
                        "age": {"method": "cap", "lower": 17, "upper": 90},
                        "hours-per-week": {"method": "cap", "lower": 1, "upper": 80},
                        "capital-gain": {"method": "cap", "lower": 0, "upper": 99999},
                        "capital-loss": {"method": "cap", "lower": 0, "upper": 4356}
                    }
                }
            }
        
        if self.MISSING_VALUE_STRATEGIES is None:
            self.MISSING_VALUE_STRATEGIES = {
                "numerical": "median",  # median, mean, mode, forward_fill, backward_fill
                "categorical": "mode",  # mode, forward_fill, backward_fill, constant
                "datetime": "forward_fill",
                "boolean": "mode"
            }
        
        if self.ENCODING_STRATEGIES is None:
            self.ENCODING_STRATEGIES = {
                "high_cardinality": "target_encoding",  # For >10 unique values
                "low_cardinality": "one_hot",          # For <=10 unique values
                "ordinal": "label_encoding",           # For ordinal data
                "binary": "binary_encoding"            # For binary categorical
            }
        
        if self.EDA_CONFIG is None:
            self.EDA_CONFIG = {
                "generate_plots": True,
                "plot_types": [
                    "distribution", "correlation", "boxplot", 
                    "scatter", "bar", "heatmap"
                ],
                "figure_size": (12, 8),
                "dpi": 300,
                "color_palette": "viridis",
                "statistical_tests": ["normality", "correlation"],
                "outlier_detection": True,
                "feature_importance": True
            }
        
        # Create directories
        for directory in [self.PROCESSED_DATA_DIR, self.EDA_REPORTS_DIR]:
            os.makedirs(directory, exist_ok=True)
            os.makedirs(os.path.join(directory, "plots"), exist_ok=True)


# Global configuration instance
preparation_config = PreparationConfig()
