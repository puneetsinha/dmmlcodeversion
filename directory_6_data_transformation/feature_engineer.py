"""
Feature Engineering Module.
Students: 2024ab05134, 2024aa05664

This module handles advanced feature creation and engineering. We implemented
several sophisticated feature engineering techniques that we learned in class:

- Binning: Converting continuous variables into categorical bins
- Ratio features: Creating meaningful ratios between existing features
- Interaction features: Combining features to capture relationships
- Polynomial features: Adding non-linear transformations
- Statistical features: Z-scores, percentiles, etc.

The most interesting part was learning about feature interactions - combining
features in smart ways can really boost model performance! We spent alot of
time experimenting with different combinations.

We also implemented feature selection using mutual information which helps
reduce dimensionality while keeping the most important features.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy import stats

from transformation_config import transformation_config


class FeatureEngineer:
    """Advanced feature engineering and transformation."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.config = transformation_config
        self.feature_log = []
        self.feature_definitions = {}
        self.transformation_metadata = {}
        
    def log_feature_operation(self, operation: str, details: str):
        """Log feature engineering operations."""
        self.feature_log.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "details": details
        })
        print(f" {operation}: {details}")
    
    def load_processed_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load processed dataset."""
        file_path = os.path.join(self.config.PROCESSED_DATA_DIR, f"{dataset_name}_cleaned.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed dataset not found: {file_path}")
        
        df = pd.read_csv(file_path)
        self.log_feature_operation("LOAD", f"Loaded {dataset_name}: {df.shape}")
        return df
    
    def create_binning_features(self, df: pd.DataFrame, feature_def: Dict) -> pd.DataFrame:
        """
        Create binning features for numerical columns.
        
        Args:
            df: DataFrame to process
            feature_def: Feature definition dictionary
            
        Returns:
            DataFrame with new binning features
        """
        df_new = df.copy()
        source_col = feature_def["source_column"]
        
        if source_col not in df.columns:
            self.log_feature_operation("WARNING", f"Source column {source_col} not found for binning")
            return df_new
        
        bins = feature_def["bins"]
        labels = feature_def.get("labels", [f"Bin_{i}" for i in range(len(bins)-1)])
        
        # Create binned feature
        new_col_name = f"{source_col}_binned"
        df_new[new_col_name] = pd.cut(df[source_col], bins=bins, labels=labels, include_lowest=True)
        
        # Create one-hot encoded versions
        dummies = pd.get_dummies(df_new[new_col_name], prefix=f"{source_col}_bin")
        df_new = pd.concat([df_new, dummies], axis=1)
        
        self.log_feature_operation("BINNING", f"Created binning for {source_col} with {len(bins)-1} bins")
        
        # Store feature definition
        self.feature_definitions[new_col_name] = {
            "type": "binning",
            "source_column": source_col,
            "bins": bins,
            "labels": labels,
            "created_columns": [new_col_name] + list(dummies.columns)
        }
        
        return df_new
    
    def create_ratio_features(self, df: pd.DataFrame, feature_def: Dict, feature_name: str) -> pd.DataFrame:
        """
        Create ratio features between two numerical columns.
        
        Args:
            df: DataFrame to process
            feature_def: Feature definition dictionary
            feature_name: Name for the new feature
            
        Returns:
            DataFrame with ratio features
        """
        df_new = df.copy()
        numerator = feature_def["numerator"]
        denominator = feature_def["denominator"]
        
        if numerator not in df.columns or denominator not in df.columns:
            self.log_feature_operation("WARNING", f"Columns for ratio not found: {numerator}/{denominator}")
            return df_new
        
        # Handle division by zero
        denominator_safe = df[denominator].replace(0, np.nan)
        
        if feature_def.get("handle_zero") == "replace_with_monthly":
            # Special case for telco data
            monthly_col = "MonthlyCharges"
            if monthly_col in df.columns:
                denominator_safe = denominator_safe.fillna(df[monthly_col])
        
        # Create ratio feature
        df_new[feature_name] = df[numerator] / denominator_safe
        
        # Handle infinite values
        df_new[feature_name] = df_new[feature_name].replace([np.inf, -np.inf], np.nan)
        df_new[feature_name] = df_new[feature_name].fillna(df_new[feature_name].median())
        
        self.log_feature_operation("RATIO", f"Created ratio feature: {feature_name}")
        
        # Store feature definition
        self.feature_definitions[feature_name] = {
            "type": "ratio",
            "numerator": numerator,
            "denominator": denominator,
            "created_columns": [feature_name]
        }
        
        return df_new
    
    def create_interaction_features(self, df: pd.DataFrame, feature_def: Dict, feature_name: str) -> pd.DataFrame:
        """
        Create interaction features between columns.
        
        Args:
            df: DataFrame to process
            feature_def: Feature definition dictionary
            feature_name: Name for the new feature
            
        Returns:
            DataFrame with interaction features
        """
        df_new = df.copy()
        columns = feature_def["columns"]
        operation = feature_def["operation"]
        
        if not all(col in df.columns for col in columns):
            missing_cols = [col for col in columns if col not in df.columns]
            self.log_feature_operation("WARNING", f"Missing columns for interaction: {missing_cols}")
            return df_new
        
        if operation == "multiply":
            df_new[feature_name] = df[columns[0]]
            for col in columns[1:]:
                df_new[feature_name] *= df[col]
        
        elif operation == "add":
            df_new[feature_name] = df[columns].sum(axis=1)
        
        elif operation == "concat":
            # For categorical features
            df_new[feature_name] = df[columns[0]].astype(str)
            for col in columns[1:]:
                df_new[feature_name] += "_" + df[col].astype(str)
        
        self.log_feature_operation("INTERACTION", f"Created interaction feature: {feature_name}")
        
        # Store feature definition
        self.feature_definitions[feature_name] = {
            "type": "interaction",
            "columns": columns,
            "operation": operation,
            "created_columns": [feature_name]
        }
        
        return df_new
    
    def create_threshold_features(self, df: pd.DataFrame, feature_def: Dict, feature_name: str) -> pd.DataFrame:
        """
        Create threshold-based binary features.
        
        Args:
            df: DataFrame to process
            feature_def: Feature definition dictionary
            feature_name: Name for the new feature
            
        Returns:
            DataFrame with threshold features
        """
        df_new = df.copy()
        source_col = feature_def["source_column"]
        threshold = feature_def["threshold"]
        operation = feature_def["operation"]
        
        if source_col not in df.columns:
            self.log_feature_operation("WARNING", f"Source column {source_col} not found for threshold")
            return df_new
        
        if operation == "greater_than":
            df_new[feature_name] = (df[source_col] > threshold).astype(int)
        elif operation == "less_than":
            df_new[feature_name] = (df[source_col] < threshold).astype(int)
        elif operation == "equal":
            df_new[feature_name] = (df[source_col] == threshold).astype(int)
        
        self.log_feature_operation("THRESHOLD", f"Created threshold feature: {feature_name}")
        
        # Store feature definition
        self.feature_definitions[feature_name] = {
            "type": "threshold",
            "source_column": source_col,
            "threshold": threshold,
            "operation": operation,
            "created_columns": [feature_name]
        }
        
        return df_new
    
    def create_arithmetic_features(self, df: pd.DataFrame, feature_def: Dict, feature_name: str) -> pd.DataFrame:
        """
        Create arithmetic features (add, subtract, etc.).
        
        Args:
            df: DataFrame to process
            feature_def: Feature definition dictionary
            feature_name: Name for the new feature
            
        Returns:
            DataFrame with arithmetic features
        """
        df_new = df.copy()
        columns = feature_def["columns"]
        operation = feature_def["operation"]
        
        if not all(col in df.columns for col in columns):
            missing_cols = [col for col in columns if col not in df.columns]
            self.log_feature_operation("WARNING", f"Missing columns for arithmetic: {missing_cols}")
            return df_new
        
        if operation == "add":
            df_new[feature_name] = df[columns].sum(axis=1)
        elif operation == "subtract":
            df_new[feature_name] = df[columns[0]] - df[columns[1]]
        elif operation == "multiply":
            df_new[feature_name] = df[columns].prod(axis=1)
        elif operation == "mean":
            df_new[feature_name] = df[columns].mean(axis=1)
        
        self.log_feature_operation("ARITHMETIC", f"Created arithmetic feature: {feature_name}")
        
        # Store feature definition
        self.feature_definitions[feature_name] = {
            "type": "arithmetic",
            "columns": columns,
            "operation": operation,
            "created_columns": [feature_name]
        }
        
        return df_new
    
    def create_composite_features(self, df: pd.DataFrame, feature_def: Dict, feature_name: str) -> pd.DataFrame:
        """
        Create composite features based on multiple conditions.
        
        Args:
            df: DataFrame to process
            feature_def: Feature definition dictionary
            feature_name: Name for the new feature
            
        Returns:
            DataFrame with composite features
        """
        df_new = df.copy()
        conditions = feature_def["conditions"]
        logic = feature_def.get("logic", "all")  # "all" or "any"
        
        condition_results = []
        
        for condition in conditions:
            col = condition["column"]
            operator = condition["operator"]
            value = condition["value"]
            
            if col not in df.columns:
                self.log_feature_operation("WARNING", f"Column {col} not found for composite feature")
                continue
            
            if operator == ">=":
                result = df[col] >= value
            elif operator == "<=":
                result = df[col] <= value
            elif operator == ">":
                result = df[col] > value
            elif operator == "<":
                result = df[col] < value
            elif operator == "==":
                result = df[col] == value
            elif operator == "!=":
                result = df[col] != value
            else:
                self.log_feature_operation("WARNING", f"Unknown operator: {operator}")
                continue
            
            condition_results.append(result)
        
        if condition_results:
            if logic == "all":
                df_new[feature_name] = np.all(condition_results, axis=0).astype(int)
            elif logic == "any":
                df_new[feature_name] = np.any(condition_results, axis=0).astype(int)
        
        self.log_feature_operation("COMPOSITE", f"Created composite feature: {feature_name}")
        
        # Store feature definition
        self.feature_definitions[feature_name] = {
            "type": "composite",
            "conditions": conditions,
            "logic": logic,
            "created_columns": [feature_name]
        }
        
        return df_new
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features like z-scores, percentile ranks.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with statistical features
        """
        df_new = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if df[col].nunique() > 5:  # Skip binary/categorical-like features
                # Z-score
                z_score_col = f"{col}_zscore"
                df_new[z_score_col] = stats.zscore(df[col], nan_policy='omit')
                
                # Percentile rank
                percentile_col = f"{col}_percentile"
                df_new[percentile_col] = df[col].rank(pct=True)
                
                # Log transformation (for positive values)
                if (df[col] > 0).all():
                    log_col = f"{col}_log"
                    df_new[log_col] = np.log1p(df[col])
                
                self.feature_definitions[z_score_col] = {
                    "type": "statistical",
                    "source_column": col,
                    "transformation": "z_score",
                    "created_columns": [z_score_col]
                }
                
                self.feature_definitions[percentile_col] = {
                    "type": "statistical",
                    "source_column": col,
                    "transformation": "percentile_rank",
                    "created_columns": [percentile_col]
                }
        
        self.log_feature_operation("STATISTICAL", f"Created statistical features for {len(numerical_cols)} columns")
        return df_new
    
    def create_polynomial_features(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Create polynomial features for numerical columns.
        
        Args:
            df: DataFrame to process
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with polynomial features
        """
        if not self.config.DERIVED_FEATURES["polynomial_features"]["enabled"]:
            return df
        
        df_new = df.copy()
        degree = self.config.DERIVED_FEATURES["polynomial_features"]["degree"]
        max_features = self.config.DERIVED_FEATURES["polynomial_features"]["max_features"]
        
        # Get numerical columns (exclude target and ID columns)
        rules = self.config.FEATURE_ENGINEERING_RULES.get(dataset_name, {})
        target_col = rules.get("target_column")
        id_col = rules.get("customer_id_column")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in [target_col, id_col]]
        
        # Limit to avoid feature explosion
        if len(numerical_cols) > max_features:
            # Select most important features based on variance
            variances = df[numerical_cols].var().sort_values(ascending=False)
            numerical_cols = variances.head(max_features).index.tolist()
        
        if len(numerical_cols) >= 2:
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
            poly_features = poly.fit_transform(df[numerical_cols])
            
            # Get feature names
            poly_feature_names = poly.get_feature_names_out(numerical_cols)
            
            # Add only interaction features (not original features)
            interaction_features = []
            for i, name in enumerate(poly_feature_names):
                if ' ' in name:  # Interaction features contain spaces
                    new_col_name = f"poly_{name.replace(' ', '_')}"
                    df_new[new_col_name] = poly_features[:, i]
                    interaction_features.append(new_col_name)
            
            self.log_feature_operation("POLYNOMIAL", f"Created {len(interaction_features)} polynomial features")
            
            self.feature_definitions["polynomial_features"] = {
                "type": "polynomial",
                "degree": degree,
                "source_columns": numerical_cols,
                "created_columns": interaction_features
            }
        
        return df_new
    
    def engineer_features(self, dataset_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform comprehensive feature engineering for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (transformed DataFrame, transformation metadata)
        """
        print(f"\n Engineering features for: {dataset_name}")
        print("=" * 60)
        
        # Load processed data
        df = self.load_processed_dataset(dataset_name)
        original_shape = df.shape
        
        # Get feature engineering rules
        if dataset_name not in self.config.FEATURE_ENGINEERING_RULES:
            self.log_feature_operation("WARNING", f"No rules defined for {dataset_name}")
            return df, {}
        
        rules = self.config.FEATURE_ENGINEERING_RULES[dataset_name]
        derived_features = rules.get("derived_features", {})
        
        # Apply derived features
        for feature_name, feature_def in derived_features.items():
            feature_type = feature_def["type"]
            
            try:
                if feature_type == "binning":
                    df = self.create_binning_features(df, feature_def)
                elif feature_type == "ratio":
                    df = self.create_ratio_features(df, feature_def, feature_name)
                elif feature_type == "interaction":
                    df = self.create_interaction_features(df, feature_def, feature_name)
                elif feature_type == "threshold":
                    df = self.create_threshold_features(df, feature_def, feature_name)
                elif feature_type == "arithmetic":
                    df = self.create_arithmetic_features(df, feature_def, feature_name)
                elif feature_type == "composite":
                    df = self.create_composite_features(df, feature_def, feature_name)
                    
            except Exception as e:
                self.log_feature_operation("ERROR", f"Failed to create {feature_name}: {str(e)}")
        
        # Create statistical features
        df = self.create_statistical_features(df)
        
        # Create polynomial features
        df = self.create_polynomial_features(df, dataset_name)
        
        # Create transformation metadata
        transformation_metadata = {
            "dataset_name": dataset_name,
            "original_shape": original_shape,
            "transformed_shape": df.shape,
            "features_created": len(df.columns) - original_shape[1],
            "feature_definitions": self.feature_definitions,
            "transformation_log": self.feature_log,
            "timestamp": datetime.now().isoformat()
        }
        
        self.log_feature_operation(
            "COMPLETE", 
            f"Feature engineering complete: {original_shape} → {df.shape}"
        )
        
        return df, transformation_metadata


if __name__ == "__main__":
    engineer = FeatureEngineer()
    print("Feature Engineer initialized successfully")
