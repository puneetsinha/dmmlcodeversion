"""
Model Trainer with comprehensive ML capabilities.
Students: 2024ab05134, 2024aa05664

This module handles training, evaluation, and comparison of multiple ML models.
We chose 4 different algorithms to compare their performance on our churn
prediction task:

1. Logistic Regression - simple baseline model
2. Random Forest - good for interpretability 
3. XGBoost - usually performs well on tabular data
4. Gradient Boosting - another tree-based approach

The biggest learning here was understanding hyperparameter tuning. Grid search
takes a long time but really improves model performance! We also integrated
MLflow for experiment tracking which was really cool to learn.

Note: Had to handle some data type issues with categorical variables - spent
quite a bit of time debuging that!
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, log_loss, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)

# Model Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

# MLflow
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from model_config import model_config


class ModelTrainer:
    """Comprehensive model training and evaluation system."""
    
    def __init__(self):
        """Initialize the model trainer."""
        self.config = model_config
        self.models = {}
        self.trained_models = {}
        self.evaluation_results = {}
        self.feature_importance = {}
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE and self.config.MLFLOW_CONFIG["enabled"]:
            self._initialize_mlflow()
    
    def _initialize_mlflow(self):
        """Initialize MLflow experiment tracking."""
        try:
            mlflow.set_tracking_uri(self.config.MLFLOW_CONFIG["tracking_uri"])
            
            # Set or create experiment
            experiment_name = self.config.MLFLOW_CONFIG["experiment_name"]
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
                print(f" MLflow experiment set: {experiment_name}")
            except Exception as e:
                print(f" MLflow experiment setup warning: {e}")
                
        except Exception as e:
            print(f" MLflow initialization failed: {e}")
    
    def load_features_from_store(self, dataset_name: str) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Load features from the feature store.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (features DataFrame, target column name)
        """
        print(f" Loading features for: {dataset_name}")
        
        # Try to load from feature store views
        view_file = os.path.join(
            self.config.FEATURE_STORE_DIR, 
            "registry", 
            "views", 
            f"{dataset_name}_all_features.parquet"
        )
        
        if os.path.exists(view_file):
            df = pd.read_parquet(view_file)
            
            # Determine target column
            target_col = None
            if dataset_name == "telco_customer_churn":
                target_col = "Churn"
            elif dataset_name == "adult_census_income":
                target_col = "income"
            
            # Ensure target column exists
            if target_col and target_col in df.columns:
                print(f" Loaded {len(df):,} samples with {len(df.columns)} features")
                print(f" Target column: {target_col}")
                return df, target_col
            else:
                print(f" Target column '{target_col}' not found in features")
                return df, None
        
        # Fallback: load from transformed data
        transformed_file = os.path.join(
            self.config.BASE_DIR,
            "transformed_data",
            f"{dataset_name}_transformed.parquet"
        )
        
        if os.path.exists(transformed_file):
            df = pd.read_parquet(transformed_file)
            
            # Determine target column
            target_col = None
            for col in df.columns:
                if col.lower() in ['churn', 'income', 'target', 'label']:
                    target_col = col
                    break
            
            print(f" Loaded from transformed data: {len(df):,} samples with {len(df.columns)} features")
            if target_col:
                print(f" Target column: {target_col}")
            
            return df, target_col
        
        raise FileNotFoundError(f"No feature data found for {dataset_name}")
    
    def prepare_training_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare data for training including feature selection and scaling.
        
        This function does all the data prep work before we can train models.
        We learned that proper data preparation is crucial for good model performance.
        
        Args:
            df: Complete DataFrame with all our features
            target_col: The column we're trying to predict (churn/income)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) ready for training
        """
        print(f"\n Preparing Training Data")
        print("=" * 40)
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables in features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        print(f" Found {len(categorical_cols)} categorical columns to encode: {list(categorical_cols)}")
        
        for col in categorical_cols:
            # Convert categorical to string first, then fill NaN values
            X[col] = X[col].astype(str)
            X[col] = X[col].replace('nan', 'missing').fillna('missing')
            
            if X[col].nunique() <= 10:  # Low cardinality
                # One-hot encode
                n_categories = X[col].nunique()
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                print(f"    One-hot encoded: {col} ({n_categories} categories)")
            else:
                # Label encode high cardinality
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                print(f"    Label encoded: {col} ({X[col].nunique()} categories)")
        
        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"    Converting remaining object column: {col}")
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        # Handle target variable encoding
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            print(f" Target encoded: {le_target.classes_}")
        
        # Split data
        stratify = y if self.config.TRAINING_CONFIG["stratify"] else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TRAINING_CONFIG["test_size"],
            random_state=self.config.TRAINING_CONFIG["random_state"],
            stratify=stratify
        )
        
        print(f" Data split:")
        print(f"   Training: {len(X_train):,} samples")
        print(f"   Testing: {len(X_test):,} samples")
        print(f"   Features: {len(X_train.columns)}")
        
        # Feature selection if specified
        if self.config.TRAINING_CONFIG.get("feature_selection_k"):
            k = min(self.config.TRAINING_CONFIG["feature_selection_k"], len(X_train.columns))
            
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Get selected feature names
            selected_features = X_train.columns[selector.get_support()]
            X_train = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
            X_test = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
            
            print(f" Feature selection: {len(X_train.columns)} features selected")
        
        return X_train, X_test, y_train, y_test
    
    def create_model_instance(self, model_name: str, model_config: Dict):
        """Create a model instance based on configuration."""
        library = model_config["library"]
        model_class = model_config["class"]
        
        if library == "sklearn":
            if model_class == "LogisticRegression":
                return LogisticRegression(random_state=42)
            elif model_class == "RandomForestClassifier":
                return RandomForestClassifier(random_state=42)
            elif model_class == "GradientBoostingClassifier":
                return GradientBoostingClassifier(random_state=42)
        
        elif library == "xgboost":
            if model_class == "XGBClassifier":
                return xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        raise ValueError(f"Unknown model configuration: {library}.{model_class}")
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                   y_train: pd.Series, y_test: pd.Series) -> Dict:
        """
        Train a single model with hyperparameter optimization.
        
        Args:
            model_name: Name of the model to train
            X_train, X_test, y_train, y_test: Training and testing data
            
        Returns:
            Training results dictionary
        """
        print(f"\n Training Model: {model_name}")
        print("-" * 40)
        
        if model_name not in self.config.MODELS_TO_TRAIN:
            raise ValueError(f"Model {model_name} not configured")
        
        model_config = self.config.MODELS_TO_TRAIN[model_name]
        
        # Create base model
        base_model = self.create_model_instance(model_name, model_config)
        
        # Apply scaling if required
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        scaler = None
        
        if model_config.get("scaling", False):
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            print(" Feature scaling applied")
        
        # Hyperparameter optimization
        start_time = datetime.now()
        
        if self.config.TRAINING_CONFIG["hyperparameter_search"] == "grid_search":
            cv = StratifiedKFold(
                n_splits=self.config.TRAINING_CONFIG["cv_folds"],
                shuffle=True,
                random_state=42
            )
            
            grid_search = GridSearchCV(
                base_model,
                model_config["hyperparameters"],
                cv=cv,
                scoring=self.config.TRAINING_CONFIG["scoring"],
                n_jobs=self.config.TRAINING_CONFIG["n_jobs"],
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            
            print(f" Best parameters: {grid_search.best_params_}")
            print(f" Best CV score: {grid_search.best_score_:.4f}")
        
        else:
            # Simple training without hyperparameter search
            best_model = base_model
            best_model.fit(X_train_scaled, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"⏱ Training time: {training_time:.2f} seconds")
        
        # Make predictions
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Store results
        training_results = {
            "model_name": model_name,
            "model": best_model,
            "scaler": scaler,
            "hyperparameters": best_model.get_params(),
            "metrics": metrics,
            "training_time": training_time,
            "feature_names": list(X_train.columns),
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
            training_results["feature_importance"] = feature_importance
            print(f" Feature importance calculated")
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE and self.config.MLFLOW_CONFIG["enabled"]:
            self._log_to_mlflow(model_name, training_results, X_train.columns)
        
        self.trained_models[model_name] = training_results
        
        print(f" {model_name} training completed")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1 Score: {metrics['f1_score']:.4f}")
        print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
        
        return training_results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='binary'),
            "recall": recall_score(y_true, y_pred, average='binary'),
            "f1_score": f1_score(y_true, y_pred, average='binary')
        }
        
        # Add probability-based metrics if available
        if y_pred_proba is not None:
            metrics.update({
                "roc_auc": roc_auc_score(y_true, y_pred_proba),
                "precision_recall_auc": average_precision_score(y_true, y_pred_proba),
                "log_loss": log_loss(y_true, y_pred_proba)
            })
        else:
            # Use decision function for SVM-like models
            metrics.update({
                "roc_auc": 0.0,
                "precision_recall_auc": 0.0,
                "log_loss": 0.0
            })
        
        return metrics
    
    def _log_to_mlflow(self, model_name: str, results: Dict, feature_names: pd.Index):
        """Log training results to MLflow."""
        try:
            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_params(results["hyperparameters"])
                
                # Log metrics
                for metric_name, metric_value in results["metrics"].items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log additional info
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("training_time", results["training_time"])
                mlflow.log_param("num_features", len(feature_names))
                
                # Log model
                if self.config.MLFLOW_CONFIG["log_models"]:
                    if "xgb" in model_name.lower():
                        mlflow.xgboost.log_model(results["model"], "model")
                    else:
                        mlflow.sklearn.log_model(results["model"], "model")
                
                print(f" Results logged to MLflow")
                
        except Exception as e:
            print(f" MLflow logging failed: {e}")
    
    def save_model(self, model_name: str, results: Dict) -> str:
        """Save trained model to disk."""
        
        model_dir = os.path.join(self.config.MODELS_DIR, "trained_models", model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_file = os.path.join(model_dir, "model.pkl")
        joblib.dump(results["model"], model_file)
        
        # Save scaler if exists
        if results["scaler"]:
            scaler_file = os.path.join(model_dir, "scaler.pkl")
            joblib.dump(results["scaler"], scaler_file)
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "hyperparameters": results["hyperparameters"],
            "metrics": results["metrics"],
            "feature_names": results["feature_names"],
            "training_time": results["training_time"],
            "timestamp": results["timestamp"],
            "files": {
                "model": model_file,
                "scaler": scaler_file if results["scaler"] else None
            }
        }
        
        metadata_file = os.path.join(model_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f" Model saved: {model_dir}")
        return model_dir
    
    def train_all_models(self, dataset_name: str) -> Dict:
        """
        Train all configured models for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Complete training results
        """
        print(f"\n Training All Models for: {dataset_name}")
        print("=" * 60)
        
        # Load data
        df, target_col = self.load_features_from_store(dataset_name)
        
        if target_col is None:
            raise ValueError(f"No target column found for {dataset_name}")
        
        # Prepare training data
        X_train, X_test, y_train, y_test = self.prepare_training_data(df, target_col)
        
        # Train each model
        all_results = {}
        training_summary = {
            "dataset_name": dataset_name,
            "target_column": target_col,
            "data_shape": df.shape,
            "training_split": {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": len(X_train.columns)
            },
            "models_trained": [],
            "best_model": None,
            "training_timestamp": datetime.now().isoformat()
        }
        
        for model_name in self.config.MODELS_TO_TRAIN.keys():
            try:
                results = self.train_model(model_name, X_train, X_test, y_train, y_test)
                all_results[model_name] = results
                
                # Save model
                model_dir = self.save_model(model_name, results)
                results["saved_location"] = model_dir
                
                training_summary["models_trained"].append(model_name)
                
            except Exception as e:
                print(f" Training failed for {model_name}: {str(e)}")
                continue
        
        # Determine best model
        if all_results:
            best_model_name = max(
                all_results.keys(),
                key=lambda x: all_results[x]["metrics"]["f1_score"]
            )
            training_summary["best_model"] = {
                "name": best_model_name,
                "f1_score": all_results[best_model_name]["metrics"]["f1_score"],
                "roc_auc": all_results[best_model_name]["metrics"]["roc_auc"]
            }
            
            print(f"\n Best Model: {best_model_name}")
            print(f"   F1 Score: {all_results[best_model_name]['metrics']['f1_score']:.4f}")
            print(f"   ROC AUC: {all_results[best_model_name]['metrics']['roc_auc']:.4f}")
        
        # Store results
        complete_results = {
            "training_summary": training_summary,
            "model_results": all_results
        }
        
        return complete_results


if __name__ == "__main__":
    trainer = ModelTrainer()
    print("Model Trainer initialized successfully")
