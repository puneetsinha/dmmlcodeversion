"""
Data Cleaning and Preprocessing Module.
Students: 2024ab05134, 2024aa05664

This module handles all our data cleaning tasks including missing values, 
outliers, data type conversions, and categorical encoding.

We learned that data cleaning is probably the most important step in any ML 
pipeline - garbage in, garbage out! Our aproach focuses on preserving as much 
data as possible while ensuring quality.

Key cleaning strategies implemented:
- Missing value imputation using median/mode
- Outlier detection with IQR method  
- Smart categorical encoding based on cardinality
- Comprehensive logging of all transformations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy import stats

from preparation_config import preparation_config


class DataCleaner:
    """Comprehensive data cleaning and preprocessing."""
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.config = preparation_config
        self.cleaning_log = []
        self.transformation_summary = {}
        
    def log_operation(self, operation: str, details: str):
        """Log cleaning operations."""
        self.cleaning_log.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "details": details
        })
        print(f" {operation}: {details}")
    
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset from file."""
        try:
            if dataset_path.endswith('.parquet'):
                df = pd.read_parquet(dataset_path)
            else:
                df = pd.read_csv(dataset_path)
            
            self.log_operation("LOAD", f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
            return df.copy()
            
        except Exception as e:
            self.log_operation("ERROR", f"Failed to load dataset: {str(e)}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Handle missing values based on column types and strategies.
        
        Args:
            df: DataFrame to process
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with missing values handled
        """
        df_cleaned = df.copy()
        
        # Get initial missing value counts
        initial_missing = df.isnull().sum().sum()
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            
            if missing_count > 0:
                column_type = self._get_column_type(df[column])
                strategy = self.config.MISSING_VALUE_STRATEGIES.get(column_type, "mode")
                
                if strategy == "median" and pd.api.types.is_numeric_dtype(df[column]):
                    fill_value = df[column].median()
                    df_cleaned[column].fillna(fill_value, inplace=True)
                    
                elif strategy == "mean" and pd.api.types.is_numeric_dtype(df[column]):
                    fill_value = df[column].mean()
                    df_cleaned[column].fillna(fill_value, inplace=True)
                    
                elif strategy == "mode":
                    fill_value = df[column].mode().iloc[0] if not df[column].mode().empty else "Unknown"
                    df_cleaned[column].fillna(fill_value, inplace=True)
                    
                elif strategy == "forward_fill":
                    df_cleaned[column].fillna(method='ffill', inplace=True)
                    
                elif strategy == "backward_fill":
                    df_cleaned[column].fillna(method='bfill', inplace=True)
                
                self.log_operation(
                    "MISSING_VALUES", 
                    f"Column '{column}': {missing_count} values filled using {strategy}"
                )
        
        final_missing = df_cleaned.isnull().sum().sum()
        self.log_operation(
            "MISSING_VALUES_SUMMARY", 
            f"Missing values reduced from {initial_missing} to {final_missing}"
        )
        
        return df_cleaned
    
    def _get_column_type(self, series: pd.Series) -> str:
        """Determine the type of a column."""
        if pd.api.types.is_numeric_dtype(series):
            return "numerical"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        elif series.dtype == 'bool':
            return "boolean"
        else:
            return "categorical"
    
    def apply_cleaning_rules(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Apply dataset-specific cleaning rules.
        
        Args:
            df: DataFrame to clean
            dataset_name: Name of the dataset
            
        Returns:
            Cleaned DataFrame
        """
        if dataset_name not in self.config.CLEANING_RULES:
            self.log_operation("CLEANING", f"No specific rules for {dataset_name}")
            return df
        
        df_cleaned = df.copy()
        rules = self.config.CLEANING_RULES[dataset_name]
        
        # Apply column-specific cleaning
        for column, rule in rules.get("columns_to_clean", {}).items():
            if column in df_cleaned.columns:
                if rule["type"] == "numeric_conversion":
                    # Convert to numeric, handling errors
                    df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors=rule.get("errors", "coerce"))
                    
                    if rule.get("replace_with_zero", False):
                        df_cleaned[column].fillna(0, inplace=True)
                    
                    self.log_operation("CONVERSION", f"Column '{column}' converted to numeric")
                    
                elif rule["type"] == "string_strip":
                    df_cleaned[column] = df_cleaned[column].astype(str).str.strip()
                    self.log_operation("CLEANING", f"Column '{column}' strings stripped")
                    
                elif rule["type"] == "replace_values":
                    for old_val, new_val in rule["mappings"].items():
                        df_cleaned[column] = df_cleaned[column].replace(old_val, new_val)
                    self.log_operation("REPLACEMENT", f"Column '{column}' values replaced")
        
        # Drop specified columns
        columns_to_drop = rules.get("columns_to_drop", [])
        if columns_to_drop:
            existing_cols = [col for col in columns_to_drop if col in df_cleaned.columns]
            df_cleaned.drop(columns=existing_cols, inplace=True)
            self.log_operation("DROP_COLUMNS", f"Dropped columns: {existing_cols}")
        
        return df_cleaned
    
    def handle_outliers(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Handle outliers based on specified treatment methods.
        
        Args:
            df: DataFrame to process
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with outliers handled
        """
        if dataset_name not in self.config.CLEANING_RULES:
            return df
        
        df_cleaned = df.copy()
        outlier_rules = self.config.CLEANING_RULES[dataset_name].get("outlier_treatment", {})
        
        for column, rule in outlier_rules.items():
            if column in df_cleaned.columns:
                method = rule["method"]
                
                if method == "cap":
                    # Cap outliers at specified bounds
                    lower_bound = rule.get("lower")
                    upper_bound = rule.get("upper")
                    
                    original_outliers = 0
                    if lower_bound is not None:
                        outliers_low = (df_cleaned[column] < lower_bound).sum()
                        df_cleaned.loc[df_cleaned[column] < lower_bound, column] = lower_bound
                        original_outliers += outliers_low
                    
                    if upper_bound is not None:
                        outliers_high = (df_cleaned[column] > upper_bound).sum()
                        df_cleaned.loc[df_cleaned[column] > upper_bound, column] = upper_bound
                        original_outliers += outliers_high
                    
                    if original_outliers > 0:
                        self.log_operation(
                            "OUTLIERS", 
                            f"Column '{column}': {original_outliers} outliers capped"
                        )
                
                elif method == "remove":
                    # Remove rows with outliers
                    Q1 = df_cleaned[column].quantile(0.25)
                    Q3 = df_cleaned[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    initial_rows = len(df_cleaned)
                    df_cleaned = df_cleaned[
                        (df_cleaned[column] >= lower_bound) & 
                        (df_cleaned[column] <= upper_bound)
                    ]
                    
                    removed_rows = initial_rows - len(df_cleaned)
                    if removed_rows > 0:
                        self.log_operation(
                            "OUTLIERS", 
                            f"Column '{column}': {removed_rows} rows with outliers removed"
                        )
        
        return df_cleaned
    
    def encode_categorical_variables(self, df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode categorical variables based on their characteristics.
        
        Args:
            df: DataFrame to process
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (encoded DataFrame, encoding mappings)
        """
        df_encoded = df.copy()
        encoding_info = {}
        
        # Apply predefined categorical mappings
        if dataset_name in self.config.CLEANING_RULES:
            mappings = self.config.CLEANING_RULES[dataset_name].get("categorical_mappings", {})
            
            for column, mapping in mappings.items():
                if column in df_encoded.columns:
                    df_encoded[column] = df_encoded[column].map(mapping).fillna(df_encoded[column])
                    encoding_info[column] = {"type": "manual_mapping", "mapping": mapping}
                    self.log_operation("ENCODING", f"Column '{column}' manually mapped")
        
        # Auto-encode remaining categorical columns
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            unique_count = df_encoded[column].nunique()
            
            if unique_count <= 2:
                # Binary encoding
                le = LabelEncoder()
                df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
                encoding_info[column] = {
                    "type": "label_encoding",
                    "classes": le.classes_.tolist()
                }
                self.log_operation("ENCODING", f"Column '{column}' label encoded (binary)")
                
            elif unique_count <= 10:
                # One-hot encoding for low cardinality
                prefix = f"{column}_"
                dummies = pd.get_dummies(df_encoded[column], prefix=prefix, drop_first=True)
                df_encoded = pd.concat([df_encoded.drop(column, axis=1), dummies], axis=1)
                encoding_info[column] = {
                    "type": "one_hot",
                    "columns": dummies.columns.tolist()
                }
                self.log_operation("ENCODING", f"Column '{column}' one-hot encoded")
                
            else:
                # Label encoding for high cardinality
                le = LabelEncoder()
                df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
                encoding_info[column] = {
                    "type": "label_encoding",
                    "classes": le.classes_.tolist()
                }
                self.log_operation("ENCODING", f"Column '{column}' label encoded (high cardinality)")
        
        return df_encoded, encoding_info
    
    def standardize_numerical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Standardize numerical features.
        
        Args:
            df: DataFrame to process
            
        Returns:
            Tuple of (standardized DataFrame, scaler information)
        """
        df_standardized = df.copy()
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        scaler_info = {}
        
        if len(numerical_columns) > 0:
            scaler = StandardScaler()
            df_standardized[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            
            scaler_info = {
                "columns": numerical_columns.tolist(),
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist()
            }
            
            self.log_operation(
                "STANDARDIZATION", 
                f"Standardized {len(numerical_columns)} numerical columns"
            )
        
        return df_standardized, scaler_info
    
    def clean_dataset(self, dataset_path: str, dataset_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform complete data cleaning pipeline.
        
        Args:
            dataset_path: Path to dataset
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning summary)
        """
        print(f"\n Cleaning dataset: {dataset_name}")
        print("=" * 50)
        
        # Load dataset
        df = self.load_dataset(dataset_path)
        original_shape = df.shape
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df, dataset_name)
        
        # Step 2: Apply cleaning rules
        df = self.apply_cleaning_rules(df, dataset_name)
        
        # Step 3: Handle outliers
        df = self.handle_outliers(df, dataset_name)
        
        # Step 4: Encode categorical variables
        df, encoding_info = self.encode_categorical_variables(df, dataset_name)
        
        # Step 5: Standardize numerical features
        df_standardized, scaler_info = self.standardize_numerical_features(df)
        
        # Create cleaning summary
        cleaning_summary = {
            "dataset_name": dataset_name,
            "original_shape": original_shape,
            "final_shape": df.shape,
            "standardized_shape": df_standardized.shape,
            "cleaning_operations": self.cleaning_log,
            "encoding_info": encoding_info,
            "scaler_info": scaler_info,
            "timestamp": datetime.now().isoformat()
        }
        
        self.log_operation(
            "CLEANING_COMPLETE", 
            f"Shape changed from {original_shape} to {df.shape}"
        )
        
        return df, df_standardized, cleaning_summary


if __name__ == "__main__":
    cleaner = DataCleaner()
    print("Data Cleaner initialized successfully")
