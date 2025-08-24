"""
Exploratory Data Analysis (EDA) Module.
Generates comprehensive visualizations and statistical analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings('ignore')

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from preparation_config import preparation_config


class EDAAnalyzer:
    """Comprehensive Exploratory Data Analysis."""
    
    def __init__(self):
        """Initialize EDA analyzer."""
        self.config = preparation_config
        self.eda_results = {}
        
        # Set plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        
        try:
            sns.set_palette(self.config.EDA_CONFIG["color_palette"])
        except:
            sns.set_palette("viridis")
        
    def generate_basic_statistics(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Generate basic statistical summary.
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with basic statistics
        """
        stats_summary = {
            "dataset_info": {
                "name": dataset_name,
                "shape": df.shape,
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024*1024), 2),
                "timestamp": datetime.now().isoformat()
            },
            "column_statistics": {},
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum())
        }
        
        # Column-wise statistics
        for column in df.columns:
            col_stats = {
                "dtype": str(df[column].dtype),
                "non_null_count": int(df[column].count()),
                "unique_count": int(df[column].nunique()),
                "null_percentage": round(df[column].isnull().mean() * 100, 2)
            }
            
            if pd.api.types.is_numeric_dtype(df[column]) and df[column].dtype != 'bool':
                try:
                    col_stats.update({
                        "mean": round(float(df[column].mean()), 3),
                        "median": round(float(df[column].median()), 3),
                        "std": round(float(df[column].std()), 3),
                        "min": round(float(df[column].min()), 3),
                        "max": round(float(df[column].max()), 3),
                        "q25": round(float(df[column].quantile(0.25)), 3),
                        "q75": round(float(df[column].quantile(0.75)), 3),
                        "skewness": round(float(df[column].skew()), 3),
                        "kurtosis": round(float(df[column].kurtosis()), 3)
                    })
                except (TypeError, ValueError):
                    # Handle edge cases where statistical calculations fail
                    col_stats["statistics_error"] = "Could not calculate numerical statistics"
            else:
                # Categorical statistics
                value_counts = df[column].value_counts().head(10)
                col_stats.update({
                    "mode": df[column].mode().iloc[0] if not df[column].mode().empty else None,
                    "top_values": value_counts.to_dict()
                })
            
            stats_summary["column_statistics"][column] = col_stats
        
        return stats_summary
    
    def create_distribution_plots(self, df: pd.DataFrame, dataset_name: str, save_path: str):
        """Create distribution plots for numerical and categorical columns."""
        # Get numerical columns excluding boolean
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if df[col].dtype != 'bool']
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Numerical distributions
        if len(numerical_cols) > 0:
            n_cols = min(3, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numerical_cols):
                if i < len(axes):
                    df[col].hist(bins=30, ax=axes[i], alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numerical_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{dataset_name}_numerical_distributions.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Categorical distributions (top categories only)
        if len(categorical_cols) > 0:
            n_cols = min(2, len(categorical_cols))
            n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(categorical_cols):
                if i < len(axes):
                    # Show top 10 categories only
                    top_categories = df[col].value_counts().head(10)
                    top_categories.plot(kind='bar', ax=axes[i], alpha=0.8)
                    axes[i].set_title(f'Top Categories in {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Count')
                    axes[i].tick_params(axis='x', rotation=45)
            
            # Hide empty subplots
            for i in range(len(categorical_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{dataset_name}_categorical_distributions.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_correlation_heatmap(self, df: pd.DataFrame, dataset_name: str, save_path: str):
        """Create correlation heatmap for numerical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if df[col].dtype != 'bool']
        
        if len(numerical_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = df[numerical_cols].corr()
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
            
            plt.title(f'Correlation Heatmap - {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{dataset_name}_correlation_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_box_plots(self, df: pd.DataFrame, dataset_name: str, save_path: str):
        """Create box plots to identify outliers."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if df[col].dtype != 'bool']
        
        if len(numerical_cols) > 0:
            n_cols = min(3, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numerical_cols):
                if i < len(axes):
                    df.boxplot(column=col, ax=axes[i])
                    axes[i].set_title(f'Box Plot of {col}')
                    axes[i].set_ylabel(col)
            
            # Hide empty subplots
            for i in range(len(numerical_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{dataset_name}_box_plots.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_scatter_plots(self, df: pd.DataFrame, dataset_name: str, save_path: str, target_col: str = None):
        """Create scatter plots for feature relationships."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if df[col].dtype != 'bool']
        
        if len(numerical_cols) >= 2:
            # Select most interesting pairs based on correlation
            corr_matrix = df[numerical_cols].corr().abs()
            
            # Find pairs with high correlation (excluding diagonal)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.3:  # threshold for interesting correlation
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            # Limit to top 6 pairs
            pairs_to_plot = high_corr_pairs[:6] if high_corr_pairs else [(numerical_cols[0], numerical_cols[1])]
            
            n_cols = min(3, len(pairs_to_plot))
            n_rows = (len(pairs_to_plot) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, (col1, col2) in enumerate(pairs_to_plot):
                if i < len(axes):
                    axes[i].scatter(df[col1], df[col2], alpha=0.6)
                    axes[i].set_xlabel(col1)
                    axes[i].set_ylabel(col2)
                    axes[i].set_title(f'{col1} vs {col2}')
                    
                    # Add correlation coefficient
                    corr_coef = df[col1].corr(df[col2])
                    axes[i].text(0.05, 0.95, f'r={corr_coef:.3f}', 
                                transform=axes[i].transAxes, fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Hide empty subplots
            for i in range(len(pairs_to_plot), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{dataset_name}_scatter_plots.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def perform_statistical_tests(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Perform statistical tests on the data."""
        test_results = {}
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if df[col].dtype != 'bool']
        
        # Normality tests
        normality_results = {}
        for col in numerical_cols:
            if df[col].nunique() > 8:  # Skip binary/categorical-like variables
                statistic, p_value = stats.shapiro(df[col].sample(min(5000, len(df))))
                normality_results[col] = {
                    "shapiro_statistic": round(statistic, 4),
                    "p_value": round(p_value, 4),
                    "is_normal": p_value > 0.05
                }
        
        test_results["normality_tests"] = normality_results
        
        # Correlation significance tests
        correlation_tests = {}
        if len(numerical_cols) >= 2:
            for i, col1 in enumerate(numerical_cols):
                for col2 in numerical_cols[i+1:]:
                    corr_coef, p_value = stats.pearsonr(df[col1], df[col2])
                    correlation_tests[f"{col1}_vs_{col2}"] = {
                        "correlation": round(corr_coef, 4),
                        "p_value": round(p_value, 4),
                        "is_significant": p_value < 0.05
                    }
        
        test_results["correlation_tests"] = correlation_tests
        
        return test_results
    
    def calculate_feature_importance(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Calculate feature importance using multiple methods."""
        if target_col not in df.columns:
            return {}
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle categorical variables by label encoding
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=['object']).columns:
            X_encoded[col] = pd.factorize(X_encoded[col])[0]
        
        importance_results = {}
        
        try:
            # Random Forest Feature Importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_encoded, y)
            
            rf_importance = dict(zip(feature_cols, rf.feature_importances_))
            importance_results["random_forest"] = {
                k: round(v, 4) for k, v in 
                sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)
            }
        except Exception as e:
            importance_results["random_forest"] = {"error": str(e)}
        
        try:
            # Mutual Information
            mi_scores = mutual_info_classif(X_encoded, y, random_state=42)
            mi_importance = dict(zip(feature_cols, mi_scores))
            importance_results["mutual_information"] = {
                k: round(v, 4) for k, v in 
                sorted(mi_importance.items(), key=lambda x: x[1], reverse=True)
            }
        except Exception as e:
            importance_results["mutual_information"] = {"error": str(e)}
        
        return importance_results
    
    def create_target_analysis(self, df: pd.DataFrame, dataset_name: str, target_col: str, save_path: str):
        """Create target variable analysis plots."""
        if target_col not in df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Target distribution
        df[target_col].value_counts().plot(kind='bar', ax=axes[0,0], alpha=0.8)
        axes[0,0].set_title(f'Distribution of {target_col}')
        axes[0,0].set_ylabel('Count')
        
        # Target vs numerical features (if any)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != target_col and df[col].dtype != 'bool']
        
        if len(numerical_cols) > 0:
            # Select first numerical column for demonstration
            first_num_col = numerical_cols[0]
            for target_val in df[target_col].unique():
                subset = df[df[target_col] == target_val][first_num_col]
                axes[0,1].hist(subset, alpha=0.6, label=f'{target_col}={target_val}', bins=20)
            
            axes[0,1].set_title(f'{first_num_col} by {target_col}')
            axes[0,1].set_xlabel(first_num_col)
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].legend()
        
        # Target correlation with numerical features
        if len(numerical_cols) > 0:
            correlations = []
            for col in numerical_cols[:10]:  # Top 10 numerical features
                corr = df[col].corr(df[target_col] if pd.api.types.is_numeric_dtype(df[target_col]) else pd.factorize(df[target_col])[0])
                correlations.append((col, abs(corr)))
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            cols, corr_vals = zip(*correlations[:8])  # Top 8
            axes[1,0].barh(cols, corr_vals, alpha=0.8)
            axes[1,0].set_title(f'Feature Correlation with {target_col}')
            axes[1,0].set_xlabel('Absolute Correlation')
        
        # Target class balance
        target_counts = df[target_col].value_counts()
        axes[1,1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title(f'{target_col} Class Balance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{dataset_name}_target_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def perform_comprehensive_eda(self, df: pd.DataFrame, dataset_name: str, target_col: str = None) -> Dict:
        """
        Perform comprehensive EDA analysis.
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset
            target_col: Target column for supervised analysis
            
        Returns:
            Comprehensive EDA results
        """
        print(f"\n Performing EDA for: {dataset_name}")
        print("=" * 50)
        
        # Create plots directory
        plots_dir = os.path.join(self.config.EDA_REPORTS_DIR, "plots", dataset_name)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate basic statistics
        basic_stats = self.generate_basic_statistics(df, dataset_name)
        print(f" Basic statistics generated")
        
        # Create visualizations
        if self.config.EDA_CONFIG["generate_plots"]:
            self.create_distribution_plots(df, dataset_name, plots_dir)
            print(f" Distribution plots created")
            
            self.create_correlation_heatmap(df, dataset_name, plots_dir)
            print(f" Correlation heatmap created")
            
            self.create_box_plots(df, dataset_name, plots_dir)
            print(f" Box plots created")
            
            self.create_scatter_plots(df, dataset_name, plots_dir, target_col)
            print(f" Scatter plots created")
            
            if target_col:
                self.create_target_analysis(df, dataset_name, target_col, plots_dir)
                print(f" Target analysis created")
        
        # Perform statistical tests
        statistical_tests = self.perform_statistical_tests(df, dataset_name)
        print(f" Statistical tests completed")
        
        # Calculate feature importance (if target provided)
        feature_importance = {}
        if target_col:
            feature_importance = self.calculate_feature_importance(df, target_col)
            print(f" Feature importance calculated")
        
        # Compile EDA results
        eda_results = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "basic_statistics": basic_stats,
            "statistical_tests": statistical_tests,
            "feature_importance": feature_importance,
            "plots_directory": plots_dir,
            "recommendations": self._generate_recommendations(df, basic_stats, statistical_tests)
        }
        
        return eda_results
    
    def _generate_recommendations(self, df: pd.DataFrame, basic_stats: Dict, tests: Dict) -> List[str]:
        """Generate data quality and analysis recommendations."""
        recommendations = []
        
        # Check for high missing values
        for col, stats in basic_stats["column_statistics"].items():
            if stats["null_percentage"] > 20:
                recommendations.append(f"High missing values in '{col}' ({stats['null_percentage']:.1f}%) - consider imputation or removal")
        
        # Check for highly skewed data
        for col, stats in basic_stats["column_statistics"].items():
            if "skewness" in stats and abs(stats["skewness"]) > 2:
                recommendations.append(f"High skewness in '{col}' (skew={stats['skewness']:.2f}) - consider transformation")
        
        # Check for low cardinality in categorical columns
        for col, stats in basic_stats["column_statistics"].items():
            if stats["dtype"] == "object" and stats["unique_count"] == 1:
                recommendations.append(f"Column '{col}' has only one unique value - consider removal")
        
        # Check correlation tests
        high_corr_pairs = []
        for test_name, result in tests.get("correlation_tests", {}).items():
            if abs(result["correlation"]) > 0.8 and result["is_significant"]:
                high_corr_pairs.append(test_name)
        
        if high_corr_pairs:
            recommendations.append(f"High correlation detected between features - consider feature selection")
        
        if not recommendations:
            recommendations.append("Data quality looks good - no major issues detected")
        
        return recommendations


if __name__ == "__main__":
    analyzer = EDAAnalyzer()
    print("EDA Analyzer initialized successfully")
