"""
Data Transformation Orchestrator.
Coordinates feature engineering, transformation, and storage.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineer import FeatureEngineer
from transformation_config import transformation_config


class TransformationOrchestrator:
    """Orchestrate data transformation and feature engineering pipeline."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.config = transformation_config
        self.engineer = FeatureEngineer()
        self.transformation_results = {}
        
    def find_processed_datasets(self) -> List[str]:
        """Find processed datasets ready for transformation."""
        datasets = []
        
        if not os.path.exists(self.config.PROCESSED_DATA_DIR):
            print(f" Processed data directory not found: {self.config.PROCESSED_DATA_DIR}")
            return datasets
        
        # Look for cleaned datasets
        for file in os.listdir(self.config.PROCESSED_DATA_DIR):
            if file.endswith('_cleaned.csv'):
                dataset_name = file.replace('_cleaned.csv', '')
                datasets.append(dataset_name)
        
        return datasets
    
    def perform_feature_selection(self, df: pd.DataFrame, target_col: str, k: int = 20) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform feature selection to keep most relevant features.
        
        Args:
            df: DataFrame with all features
            target_col: Target column name
            k: Number of top features to select
            
        Returns:
            Tuple of (selected DataFrame, selection metadata)
        """
        if target_col not in df.columns:
            return df, {"error": f"Target column {target_col} not found"}
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle categorical variables by label encoding
        X_encoded = X.copy()
        categorical_mappings = {}
        
        for col in X_encoded.select_dtypes(include=['object']).columns:
            X_encoded[col] = pd.factorize(X_encoded[col])[0]
            categorical_mappings[col] = "label_encoded"
        
        # Fill any remaining NaN values
        X_encoded = X_encoded.fillna(X_encoded.median(numeric_only=True))
        
        try:
            # Use mutual information for feature selection
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            
            # Limit k to available features
            k = min(k, len(feature_cols))
            
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_selected = selector.fit_transform(X_encoded, y)
            
            # Get selected feature names
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            
            # Create selected DataFrame
            df_selected = df[selected_features + [target_col]].copy()
            
            # Get feature scores
            feature_scores = dict(zip(feature_cols, selector.scores_))
            selected_scores = {feat: feature_scores[feat] for feat in selected_features}
            
            selection_metadata = {
                "method": "mutual_info_classif",
                "total_features": len(feature_cols),
                "selected_features": len(selected_features),
                "selected_feature_names": selected_features,
                "feature_scores": {k: round(v, 4) for k, v in selected_scores.items()},
                "categorical_mappings": categorical_mappings
            }
            
            print(f" Feature selection: {len(feature_cols)} → {len(selected_features)} features")
            
            return df_selected, selection_metadata
            
        except Exception as e:
            print(f" Feature selection failed: {str(e)}")
            return df, {"error": str(e)}
    
    def save_transformed_data(self, df: pd.DataFrame, dataset_name: str, metadata: Dict):
        """Save transformed data in multiple formats."""
        
        # Save as CSV
        csv_file = os.path.join(self.config.TRANSFORMED_DATA_DIR, f"{dataset_name}_transformed.csv")
        df.to_csv(csv_file, index=False)
        
        # Save as Parquet
        parquet_file = os.path.join(self.config.TRANSFORMED_DATA_DIR, f"{dataset_name}_transformed.parquet")
        df.to_parquet(parquet_file, index=False)
        
        # Save metadata
        metadata_file = os.path.join(
            self.config.FEATURE_DEFINITIONS_DIR, 
            "metadata", 
            f"{dataset_name}_transformation_metadata.json"
        )
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save feature definitions
        features_file = os.path.join(
            self.config.FEATURE_DEFINITIONS_DIR,
            "features",
            f"{dataset_name}_features.json"
        )
        with open(features_file, 'w') as f:
            json.dump(metadata.get("feature_definitions", {}), f, indent=2, default=str)
        
        print(f" Saved transformed data:")
        print(f"    CSV: {csv_file}")
        print(f"    Parquet: {parquet_file}")
        print(f"    Metadata: {metadata_file}")
        print(f"     Features: {features_file}")
    
    def generate_transformation_summary(self, dataset_name: str, metadata: Dict) -> Dict:
        """Generate comprehensive transformation summary."""
        
        original_shape = metadata.get("original_shape", (0, 0))
        transformed_shape = metadata.get("transformed_shape", (0, 0))
        
        summary = {
            "dataset_name": dataset_name,
            "transformation_timestamp": datetime.now().isoformat(),
            "shape_transformation": {
                "original": original_shape,
                "transformed": transformed_shape,
                "features_added": transformed_shape[1] - original_shape[1],
                "growth_ratio": round(transformed_shape[1] / original_shape[1], 2) if original_shape[1] > 0 else 0
            },
            "feature_engineering": {
                "total_operations": len(metadata.get("transformation_log", [])),
                "feature_types_created": list(set([
                    defn.get("type", "unknown") 
                    for defn in metadata.get("feature_definitions", {}).values()
                ])),
                "features_by_type": {}
            },
            "data_quality": {
                "missing_values": "calculated_after_transformation",
                "data_types": "preserved_and_enhanced"
            }
        }
        
        # Count features by type
        for defn in metadata.get("feature_definitions", {}).values():
            feat_type = defn.get("type", "unknown")
            if feat_type in summary["feature_engineering"]["features_by_type"]:
                summary["feature_engineering"]["features_by_type"][feat_type] += 1
            else:
                summary["feature_engineering"]["features_by_type"][feat_type] = 1
        
        return summary
    
    def transform_dataset(self, dataset_name: str) -> Dict:
        """
        Transform a single dataset with comprehensive feature engineering.
        
        Args:
            dataset_name: Name of the dataset to transform
            
        Returns:
            Transformation results
        """
        print(f"\n{'='*70}")
        print(f" TRANSFORMING DATASET: {dataset_name}")
        print(f"{'='*70}")
        
        try:
            # Step 1: Feature Engineering
            df_transformed, engineering_metadata = self.engineer.engineer_features(dataset_name)
            
            # Step 2: Feature Selection (optional)
            rules = self.config.FEATURE_ENGINEERING_RULES.get(dataset_name, {})
            target_col = rules.get("target_column")
            
            if target_col and target_col in df_transformed.columns:
                df_selected, selection_metadata = self.perform_feature_selection(
                    df_transformed, target_col, k=30
                )
                df_final = df_selected
                engineering_metadata["feature_selection"] = selection_metadata
            else:
                df_final = df_transformed
                print(f" Skipping feature selection: target column '{target_col}' not found")
            
            # Step 3: Generate summary
            transformation_summary = self.generate_transformation_summary(dataset_name, engineering_metadata)
            
            # Combine all metadata
            complete_metadata = {
                **engineering_metadata,
                "transformation_summary": transformation_summary
            }
            
            # Step 4: Save transformed data
            self.save_transformed_data(df_final, dataset_name, complete_metadata)
            
            # Step 5: Generate final summary
            result_summary = {
                "dataset_name": dataset_name,
                "status": "SUCCESS",
                "original_shape": engineering_metadata.get("original_shape"),
                "final_shape": df_final.shape,
                "features_engineered": engineering_metadata.get("features_created", 0),
                "transformation_summary": transformation_summary,
                "output_files": {
                    "csv": os.path.join(self.config.TRANSFORMED_DATA_DIR, f"{dataset_name}_transformed.csv"),
                    "parquet": os.path.join(self.config.TRANSFORMED_DATA_DIR, f"{dataset_name}_transformed.parquet"),
                    "metadata": os.path.join(self.config.FEATURE_DEFINITIONS_DIR, "metadata", f"{dataset_name}_transformation_metadata.json"),
                    "features": os.path.join(self.config.FEATURE_DEFINITIONS_DIR, "features", f"{dataset_name}_features.json")
                }
            }
            
            print(f" Dataset transformation completed: {dataset_name}")
            return result_summary
            
        except Exception as e:
            error_summary = {
                "dataset_name": dataset_name,
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            print(f" Dataset transformation failed: {dataset_name} - {str(e)}")
            return error_summary
    
    def run_complete_transformation(self) -> Dict:
        """
        Run complete transformation pipeline for all datasets.
        
        Returns:
            Complete transformation results
        """
        print(" Starting Complete Data Transformation Pipeline")
        print("=" * 80)
        
        # Find datasets
        datasets = self.find_processed_datasets()
        
        if not datasets:
            print(" No processed datasets found for transformation")
            return {}
        
        print(f" Found {len(datasets)} datasets for transformation:")
        for dataset in datasets:
            print(f"   {dataset}")
        
        # Transform each dataset
        all_results = []
        successful_transformations = 0
        
        for dataset_name in datasets:
            result = self.transform_dataset(dataset_name)
            all_results.append(result)
            
            if result.get("status") == "SUCCESS":
                successful_transformations += 1
        
        # Generate overall summary
        overall_summary = {
            "transformation_pipeline_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_datasets": len(datasets),
                "successful_transformations": successful_transformations,
                "failed_transformations": len(datasets) - successful_transformations,
                "success_rate": round(successful_transformations / len(datasets) * 100, 1) if datasets else 0
            },
            "dataset_results": all_results,
            "output_directories": {
                "transformed_data": self.config.TRANSFORMED_DATA_DIR,
                "feature_definitions": self.config.FEATURE_DEFINITIONS_DIR
            },
            "recommendations": self._generate_transformation_recommendations(all_results)
        }
        
        # Save overall summary
        summary_file = os.path.join(
            self.config.TRANSFORMED_DATA_DIR,
            f"transformation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(summary_file, 'w') as f:
            json.dump(overall_summary, f, indent=2, default=str)
        
        # Print final summary
        self._print_final_summary(overall_summary)
        
        print(f"\n Complete summary saved: {summary_file}")
        print(f" Data transformation pipeline completed!")
        
        return overall_summary
    
    def _generate_transformation_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate recommendations based on transformation results."""
        recommendations = []
        
        successful_results = [r for r in results if r.get("status") == "SUCCESS"]
        
        if not successful_results:
            recommendations.append("No successful transformations - review error logs and fix issues")
            return recommendations
        
        # Analyze feature creation patterns
        total_features_created = sum(r.get("features_engineered", 0) for r in successful_results)
        avg_features_per_dataset = total_features_created / len(successful_results)
        
        if avg_features_per_dataset > 50:
            recommendations.append("High number of features created - consider feature selection for model performance")
        elif avg_features_per_dataset < 10:
            recommendations.append("Moderate feature engineering - consider additional domain-specific features")
        
        # Check for feature selection
        has_feature_selection = any(
            "feature_selection" in r.get("transformation_summary", {}) 
            for r in successful_results
        )
        
        if has_feature_selection:
            recommendations.append("Feature selection applied - review selected features for model training")
        else:
            recommendations.append("Consider feature selection to improve model performance and training speed")
        
        # General recommendations
        recommendations.extend([
            "Transformed datasets ready for feature store integration",
            "Consider A/B testing different feature combinations for optimal model performance",
            "Monitor feature importance in production models to validate engineering decisions"
        ])
        
        return recommendations
    
    def _print_final_summary(self, summary: Dict):
        """Print final transformation summary."""
        print(f"\n{'='*80}")
        print(" DATA TRANSFORMATION SUMMARY")
        print("=" * 80)
        
        pipeline_summary = summary["transformation_pipeline_summary"]
        print(f"Total Datasets: {pipeline_summary['total_datasets']}")
        print(f"Successful: {pipeline_summary['successful_transformations']}")
        print(f"Failed: {pipeline_summary['failed_transformations']}")
        print(f"Success Rate: {pipeline_summary['success_rate']}%")
        
        # Dataset details
        print(f"\n TRANSFORMATION DETAILS:")
        for result in summary["dataset_results"]:
            if result.get("status") == "SUCCESS":
                name = result["dataset_name"]
                original = result["original_shape"]
                final = result["final_shape"]
                features_added = result["features_engineered"]
                
                print(f"   {name}:")
                print(f"     Shape: {original} → {final}")
                print(f"     Features Added: {features_added}")
        
        print(f"\n RECOMMENDATIONS:")
        for i, rec in enumerate(summary["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n OUTPUT DIRECTORIES:")
        for key, path in summary["output_directories"].items():
            print(f"   {key.replace('_', ' ').title()}: {path}")


def main():
    """Main function to run data transformation."""
    orchestrator = TransformationOrchestrator()
    results = orchestrator.run_complete_transformation()
    return results


if __name__ == "__main__":
    main()
