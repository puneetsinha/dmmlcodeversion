"""
Comprehensive Data Validation Module.
Students: 2024ab05134, 2024aa05664

This module performs data quality checks, anomaly detection, and generates 
validation reports. We learned that data validation is absolutely critical - 
you can't build good models on bad data!

Our validation framework includes:
- Missing value analysis and completeness scoring
- Duplicate detection and removal strategies
- Data type consistency validation
- Statistical outlier detection using IQR and Z-score methods
- Schema validation against expected formats

The most challenging part was developing a comprehensive scoring system that
weighs different quality issues appropriately. We spent considerable time
researching industry standards for data quality assessment.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Statistical tests
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

from validation_config import validation_config


class DataValidator:
    """Comprehensive data validation and quality assessment."""
    
    def __init__(self):
        """Initialize the data validator."""
        self.config = validation_config
        self.validation_results = {}
        self.summary_stats = {}
        
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load dataset from parquet file.
        
        Args:
            dataset_path: Path to parquet file
            
        Returns:
            Loaded DataFrame
        """
        try:
            if dataset_path.endswith('.parquet'):
                df = pd.read_parquet(dataset_path)
            else:
                df = pd.read_csv(dataset_path)
            
            print(f" Loaded dataset: {os.path.basename(dataset_path)} ({len(df):,} rows, {len(df.columns)} columns)")
            return df
            
        except Exception as e:
            print(f" Failed to load dataset {dataset_path}: {str(e)}")
            raise
    
    def validate_schema(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Validate dataset schema against expected structure.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset
            
        Returns:
            Schema validation results
        """
        if dataset_name not in self.config.DATASET_RULES:
            return {"status": "SKIPPED", "reason": "No validation rules defined"}
        
        rules = self.config.DATASET_RULES[dataset_name]
        results = {
            "status": "PASS",
            "issues": [],
            "missing_columns": [],
            "extra_columns": [],
            "data_type_issues": []
        }
        
        # Check required columns
        required_cols = set(rules.get("required_columns", []))
        actual_cols = set(df.columns)
        
        missing_cols = required_cols - actual_cols
        extra_cols = actual_cols - required_cols
        
        if missing_cols:
            results["missing_columns"] = list(missing_cols)
            results["issues"].append(f"Missing columns: {missing_cols}")
            results["status"] = "FAIL"
        
        if extra_cols:
            results["extra_columns"] = list(extra_cols)
            results["issues"].append(f"Extra columns: {extra_cols}")
        
        # Check data types
        expected_types = rules.get("data_types", {})
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type and not self._is_compatible_type(actual_type, expected_type):
                    results["data_type_issues"].append({
                        "column": col,
                        "expected": expected_type,
                        "actual": actual_type
                    })
        
        if results["data_type_issues"]:
            results["status"] = "FAIL"
            results["issues"].append("Data type mismatches found")
        
        return results
    
    def _is_compatible_type(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible."""
        compatible_types = {
            "object": ["object", "string"],
            "int64": ["int64", "int32", "int8", "int16"],
            "float64": ["float64", "float32"],
        }
        
        return actual in compatible_types.get(expected, [expected])
    
    def check_completeness(self, df: pd.DataFrame) -> Dict:
        """
        Check data completeness (missing values).
        
        Args:
            df: DataFrame to check
            
        Returns:
            Completeness analysis results
        """
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness_ratio = 1 - (missing_cells / total_cells)
        
        column_completeness = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            completeness = 1 - (missing_count / len(df))
            column_completeness[col] = {
                "missing_count": int(missing_count),
                "completeness": round(completeness, 4),
                "status": "PASS" if completeness >= self.config.COMPLETENESS_THRESHOLD else "FAIL"
            }
        
        results = {
            "overall_completeness": round(completeness_ratio, 4),
            "total_missing_cells": int(missing_cells),
            "status": "PASS" if completeness_ratio >= self.config.COMPLETENESS_THRESHOLD else "FAIL",
            "column_completeness": column_completeness
        }
        
        return results
    
    def check_uniqueness(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Check data uniqueness for ID columns.
        
        Args:
            df: DataFrame to check
            dataset_name: Name of the dataset
            
        Returns:
            Uniqueness analysis results
        """
        results = {"column_uniqueness": {}}
        
        if dataset_name in self.config.DATASET_RULES:
            id_columns = self.config.DATASET_RULES[dataset_name].get("id_columns", [])
            
            for col in id_columns:
                if col in df.columns:
                    unique_count = df[col].nunique()
                    total_count = len(df[col].dropna())
                    uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
                    
                    results["column_uniqueness"][col] = {
                        "unique_count": int(unique_count),
                        "total_count": int(total_count),
                        "uniqueness_ratio": round(uniqueness_ratio, 4),
                        "status": "PASS" if uniqueness_ratio >= self.config.UNIQUENESS_THRESHOLD else "FAIL"
                    }
        
        return results
    
    def check_data_types_consistency(self, df: pd.DataFrame) -> Dict:
        """
        Check consistency of data types within columns.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Data type consistency results
        """
        results = {"column_consistency": {}}
        
        for col in df.columns:
            # Check for mixed types in object columns
            if df[col].dtype == 'object':
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    # Try to identify type inconsistencies
                    numeric_count = 0
                    string_count = 0
                    
                    for value in non_null_values.sample(min(1000, len(non_null_values))):
                        try:
                            float(value)
                            numeric_count += 1
                        except (ValueError, TypeError):
                            string_count += 1
                    
                    total_checked = numeric_count + string_count
                    consistency_ratio = max(numeric_count, string_count) / total_checked if total_checked > 0 else 1
                    
                    results["column_consistency"][col] = {
                        "numeric_count": numeric_count,
                        "string_count": string_count,
                        "consistency_ratio": round(consistency_ratio, 4),
                        "status": "PASS" if consistency_ratio >= self.config.CONSISTENCY_THRESHOLD else "FAIL"
                    }
        
        return results
    
    def check_value_ranges(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Check if numerical values are within expected ranges.
        
        Args:
            df: DataFrame to check
            dataset_name: Name of the dataset
            
        Returns:
            Value range validation results
        """
        results = {"range_violations": {}}
        
        if dataset_name in self.config.DATASET_RULES:
            value_ranges = self.config.DATASET_RULES[dataset_name].get("value_ranges", {})
            
            for col, range_def in value_ranges.items():
                if col in df.columns:
                    min_val = range_def.get("min")
                    max_val = range_def.get("max")
                    
                    # Convert to numeric, handling errors
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    
                    violations = []
                    if min_val is not None:
                        below_min = (numeric_col < min_val).sum()
                        if below_min > 0:
                            violations.append(f"{below_min} values below minimum {min_val}")
                    
                    if max_val is not None:
                        above_max = (numeric_col > max_val).sum()
                        if above_max > 0:
                            violations.append(f"{above_max} values above maximum {max_val}")
                    
                    results["range_violations"][col] = {
                        "violations": violations,
                        "status": "PASS" if not violations else "FAIL"
                    }
        
        return results
    
    def check_categorical_values(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Check if categorical values are within expected sets.
        
        Args:
            df: DataFrame to check
            dataset_name: Name of the dataset
            
        Returns:
            Categorical value validation results
        """
        results = {"categorical_violations": {}}
        
        if dataset_name in self.config.DATASET_RULES:
            categorical_values = self.config.DATASET_RULES[dataset_name].get("categorical_values", {})
            
            for col, expected_values in categorical_values.items():
                if col in df.columns:
                    actual_values = set(df[col].dropna().unique())
                    expected_set = set(expected_values)
                    
                    unexpected_values = actual_values - expected_set
                    missing_values = expected_set - actual_values
                    
                    results["categorical_violations"][col] = {
                        "unexpected_values": list(unexpected_values),
                        "missing_expected_values": list(missing_values),
                        "status": "PASS" if not unexpected_values else "FAIL"
                    }
        
        return results
    
    def detect_duplicates(self, df: pd.DataFrame) -> Dict:
        """
        Detect duplicate records.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Duplicate detection results
        """
        # Full row duplicates
        full_duplicates = df.duplicated().sum()
        
        # Partial duplicates (excluding potential ID columns)
        non_id_columns = [col for col in df.columns if not col.lower().endswith('id')]
        partial_duplicates = df[non_id_columns].duplicated().sum() if non_id_columns else 0
        
        results = {
            "full_duplicates": int(full_duplicates),
            "partial_duplicates": int(partial_duplicates),
            "duplicate_ratio": round(full_duplicates / len(df), 4),
            "status": "PASS" if full_duplicates < len(df) * 0.01 else "WARN"  # Warn if >1% duplicates
        }
        
        return results
    
    def detect_outliers(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Detect outliers using multiple methods.
        
        Args:
            df: DataFrame to check
            dataset_name: Name of the dataset
            
        Returns:
            Outlier detection results
        """
        results = {"outlier_analysis": {}}
        
        if dataset_name in self.config.DATASET_RULES:
            numerical_columns = self.config.DATASET_RULES[dataset_name].get("numerical_columns", [])
            
            for col in numerical_columns:
                if col in df.columns:
                    # Convert to numeric, handling errors
                    numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                    
                    if len(numeric_data) > 0:
                        outlier_info = {}
                        
                        # IQR method
                        Q1 = numeric_data.quantile(0.25)
                        Q3 = numeric_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        iqr_outliers = ((numeric_data < lower_bound) | (numeric_data > upper_bound)).sum()
                        outlier_info["iqr_outliers"] = int(iqr_outliers)
                        
                        # Z-score method
                        z_scores = np.abs(stats.zscore(numeric_data))
                        z_outliers = (z_scores > 3).sum()
                        outlier_info["z_score_outliers"] = int(z_outliers)
                        
                        # Statistical summary
                        outlier_info["statistics"] = {
                            "mean": round(numeric_data.mean(), 2),
                            "std": round(numeric_data.std(), 2),
                            "min": round(numeric_data.min(), 2),
                            "max": round(numeric_data.max(), 2),
                            "q1": round(Q1, 2),
                            "q3": round(Q3, 2)
                        }
                        
                        results["outlier_analysis"][col] = outlier_info
        
        return results
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive summary statistics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Summary statistics
        """
        stats = {
            "basic_info": {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024*1024), 2)
            },
            "column_info": {}
        }
        
        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique())
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    "mean": round(df[col].mean(), 2) if not df[col].isnull().all() else None,
                    "std": round(df[col].std(), 2) if not df[col].isnull().all() else None,
                    "min": round(df[col].min(), 2) if not df[col].isnull().all() else None,
                    "max": round(df[col].max(), 2) if not df[col].isnull().all() else None
                })
            else:
                # For categorical/object columns
                value_counts = df[col].value_counts().head(5)
                col_info["top_values"] = value_counts.to_dict()
            
            stats["column_info"][col] = col_info
        
        return stats
    
    def validate_dataset(self, dataset_path: str, dataset_name: str) -> Dict:
        """
        Perform comprehensive validation on a dataset.
        
        Args:
            dataset_path: Path to dataset file
            dataset_name: Name of the dataset
            
        Returns:
            Complete validation results
        """
        print(f"\n Validating dataset: {dataset_name}")
        print("=" * 50)
        
        # Load dataset
        df = self.load_dataset(dataset_path)
        
        # Perform all validation checks
        validation_results = {
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "validation_timestamp": datetime.now().isoformat(),
            "schema_validation": self.validate_schema(df, dataset_name),
            "completeness_check": self.check_completeness(df),
            "uniqueness_check": self.check_uniqueness(df, dataset_name),
            "consistency_check": self.check_data_types_consistency(df),
            "range_validation": self.check_value_ranges(df, dataset_name),
            "categorical_validation": self.check_categorical_values(df, dataset_name),
            "duplicate_detection": self.detect_duplicates(df),
            "outlier_detection": self.detect_outliers(df, dataset_name),
            "summary_statistics": self.generate_summary_statistics(df)
        }
        
        # Calculate overall quality score
        validation_results["quality_score"] = self._calculate_quality_score(validation_results)
        
        return validation_results
    
    def _calculate_quality_score(self, results: Dict) -> Dict:
        """Calculate overall data quality score."""
        scores = []
        weights = []
        
        # Schema validation (weight: 0.2)
        if results["schema_validation"]["status"] == "PASS":
            scores.append(1.0)
        else:
            scores.append(0.5)
        weights.append(0.2)
        
        # Completeness (weight: 0.3)
        completeness = results["completeness_check"]["overall_completeness"]
        scores.append(completeness)
        weights.append(0.3)
        
        # Consistency (weight: 0.2)
        consistency_checks = results["consistency_check"]["column_consistency"]
        if consistency_checks:
            avg_consistency = np.mean([c["consistency_ratio"] for c in consistency_checks.values()])
            scores.append(avg_consistency)
        else:
            scores.append(1.0)
        weights.append(0.2)
        
        # Duplicates (weight: 0.1)
        duplicate_ratio = results["duplicate_detection"]["duplicate_ratio"]
        duplicate_score = max(0, 1 - duplicate_ratio * 10)  # Penalize duplicates heavily
        scores.append(duplicate_score)
        weights.append(0.1)
        
        # Range/categorical validation (weight: 0.2)
        range_issues = len([v for v in results["range_validation"]["range_violations"].values() if v["status"] == "FAIL"])
        categorical_issues = len([v for v in results["categorical_validation"]["categorical_violations"].values() if v["status"] == "FAIL"])
        total_range_cat_checks = len(results["range_validation"]["range_violations"]) + len(results["categorical_validation"]["categorical_violations"])
        
        if total_range_cat_checks > 0:
            validation_score = 1 - ((range_issues + categorical_issues) / total_range_cat_checks)
        else:
            validation_score = 1.0
        scores.append(validation_score)
        weights.append(0.2)
        
        # Calculate weighted average
        overall_score = np.average(scores, weights=weights)
        
        return {
            "overall_score": round(overall_score, 3),
            "grade": self._get_quality_grade(overall_score),
            "component_scores": {
                "schema": round(scores[0], 3),
                "completeness": round(scores[1], 3),
                "consistency": round(scores[2], 3),
                "duplicates": round(scores[3], 3),
                "validation": round(scores[4], 3)
            }
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade."""
        if score >= 0.95:
            return "A+"
        elif score >= 0.90:
            return "A"
        elif score >= 0.85:
            return "B+"
        elif score >= 0.80:
            return "B"
        elif score >= 0.75:
            return "C+"
        elif score >= 0.70:
            return "C"
        else:
            return "D"


if __name__ == "__main__":
    validator = DataValidator()
    
    # Test validation
    data_lake_path = validation_config.DATA_LAKE_DIR
    print(f"Data lake path: {data_lake_path}")
    
    if os.path.exists(data_lake_path):
        print("Found data lake directory")
    else:
        print("Data lake directory not found")
