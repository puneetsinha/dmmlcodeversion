"""
Configuration for model building pipeline.
Students: 2024ab05134, 2024aa05664

This config file defines all the models we want to train and their hyperparameters.
We chose these specific algorithms based on what we learned in class about
different ML approaches:

- Logistic Regression: Simple linear baseline
- Random Forest: Good interpretability and handles mixed data types well
- XGBoost: Usually performs very well on tabular data
- Gradient Boosting: Another ensemble method for comparison

The hyperparameter grids were chosen based on common ranges we found in
literature and some initial experimentation. Grid search is expensive but
gives us confidence in finding good parameters.

We also configured MLflow for experiment tracking which has been super helpful
for comparing different runs and keeping track of our experiments.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class ModelConfig:
    """Configuration for model building settings."""
    
    # Base directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FEATURE_STORE_DIR: str = os.path.join(BASE_DIR, "feature_store")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    MODEL_REPORTS_DIR: str = os.path.join(BASE_DIR, "model_reports")
    
    # Model configurations
    MODELS_TO_TRAIN: Dict[str, Dict] = None
    
    # Training configurations
    TRAINING_CONFIG: Dict[str, Any] = None
    
    # Evaluation metrics
    EVALUATION_METRICS: List[str] = None
    
    # Cross-validation settings
    CV_CONFIG: Dict[str, Any] = None
    
    # MLflow settings
    MLFLOW_CONFIG: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.MODELS_TO_TRAIN is None:
            self.MODELS_TO_TRAIN = {
                "logistic_regression": {
                    "type": "classification",
                    "library": "sklearn",
                    "class": "LogisticRegression",
                    "hyperparameters": {
                        "C": [0.1, 1.0, 10.0],
                        "penalty": ["l1", "l2"],
                        "solver": ["liblinear", "saga"],
                        "max_iter": [1000]
                    },
                    "feature_selection": True,
                    "scaling": True
                },
                "random_forest": {
                    "type": "classification", 
                    "library": "sklearn",
                    "class": "RandomForestClassifier",
                    "hyperparameters": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [10, 20, None],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "bootstrap": [True]
                    },
                    "feature_selection": False,
                    "scaling": False
                },
                "xgboost": {
                    "type": "classification",
                    "library": "xgboost", 
                    "class": "XGBClassifier",
                    "hyperparameters": {
                        "n_estimators": [100, 200],
                        "max_depth": [6, 8, 10],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "subsample": [0.8, 1.0],
                        "colsample_bytree": [0.8, 1.0]
                    },
                    "feature_selection": True,
                    "scaling": False
                },
                "gradient_boosting": {
                    "type": "classification",
                    "library": "sklearn",
                    "class": "GradientBoostingClassifier", 
                    "hyperparameters": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "max_depth": [3, 5, 7],
                        "subsample": [0.8, 1.0]
                    },
                    "feature_selection": True,
                    "scaling": False
                }
            }
        
        if self.TRAINING_CONFIG is None:
            self.TRAINING_CONFIG = {
                "test_size": 0.2,
                "random_state": 42,
                "stratify": True,
                "feature_selection_k": 30,
                "hyperparameter_search": "grid_search",  # grid_search, random_search
                "cv_folds": 5,
                "scoring": "f1",
                "n_jobs": -1,
                "early_stopping": True
            }
        
        if self.EVALUATION_METRICS is None:
            self.EVALUATION_METRICS = [
                "accuracy", "precision", "recall", "f1_score", 
                "roc_auc", "precision_recall_auc", "log_loss"
            ]
        
        if self.CV_CONFIG is None:
            self.CV_CONFIG = {
                "cv_strategy": "stratified_kfold",
                "n_splits": 5,
                "shuffle": True,
                "random_state": 42
            }
        
        if self.MLFLOW_CONFIG is None:
            self.MLFLOW_CONFIG = {
                "enabled": True,
                "experiment_name": "churn_prediction",
                "tracking_uri": os.path.join(self.BASE_DIR, "mlruns"),
                "artifact_location": os.path.join(self.MODELS_DIR, "mlflow_artifacts"),
                "log_models": True,
                "log_artifacts": True
            }
        
        # Create directories
        for directory in [self.MODELS_DIR, self.MODEL_REPORTS_DIR]:
            os.makedirs(directory, exist_ok=True)
            os.makedirs(os.path.join(directory, "trained_models"), exist_ok=True)
            os.makedirs(os.path.join(directory, "evaluation_reports"), exist_ok=True)
            os.makedirs(os.path.join(directory, "plots"), exist_ok=True)


# Global configuration instance
model_config = ModelConfig()
