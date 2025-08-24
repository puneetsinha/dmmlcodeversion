"""
Feature Store Orchestrator.
Manages the complete feature store pipeline including registration, serving, and management.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_store import FeatureStore
from feature_store_config import feature_store_config


class FeatureStoreOrchestrator:
    """Orchestrate feature store operations and management."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.config = feature_store_config
        self.feature_store = FeatureStore()
        self.orchestration_results = {}
        
    def find_transformed_datasets(self) -> List[Dict[str, str]]:
        """Find transformed datasets ready for feature store registration."""
        datasets = []
        
        if not os.path.exists(self.config.TRANSFORMED_DATA_DIR):
            print(f" Transformed data directory not found: {self.config.TRANSFORMED_DATA_DIR}")
            return datasets
        
        # Look for transformed datasets
        for file in os.listdir(self.config.TRANSFORMED_DATA_DIR):
            if file.endswith('_transformed.parquet'):
                dataset_name = file.replace('_transformed.parquet', '')
                file_path = os.path.join(self.config.TRANSFORMED_DATA_DIR, file)
                
                datasets.append({
                    'name': dataset_name,
                    'file_path': file_path,
                    'size_mb': round(os.path.getsize(file_path) / (1024*1024), 2)
                })
        
        return datasets
    
    def register_dataset_features(self, dataset_info: Dict) -> Dict:
        """
        Register all features from a transformed dataset.
        
        Args:
            dataset_info: Dataset information dictionary
            
        Returns:
            Registration results
        """
        dataset_name = dataset_info['name']
        file_path = dataset_info['file_path']
        
        print(f"\n{'='*70}")
        print(f" REGISTERING FEATURES: {dataset_name}")
        print(f"{'='*70}")
        
        try:
            # Load transformed dataset
            df = pd.read_parquet(file_path)
            print(f" Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
            
            # Register feature group
            group_id = self.feature_store.register_feature_group(dataset_name, df)
            
            # Get registration statistics
            feature_metadata = self.feature_store.get_feature_metadata(feature_group=group_id)
            
            registration_results = {
                "dataset_name": dataset_name,
                "group_id": group_id,
                "status": "SUCCESS",
                "features_registered": len(feature_metadata),
                "data_shape": df.shape,
                "registration_timestamp": datetime.now().isoformat(),
                "feature_summary": self._summarize_features(feature_metadata)
            }
            
            print(f" Successfully registered {len(feature_metadata)} features")
            return registration_results
            
        except Exception as e:
            error_results = {
                "dataset_name": dataset_name,
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            print(f" Feature registration failed: {str(e)}")
            return error_results
    
    def _summarize_features(self, feature_metadata: List[Dict]) -> Dict:
        """Summarize feature metadata for reporting."""
        summary = {
            "total_features": len(feature_metadata),
            "by_data_type": {},
            "by_tags": {},
            "data_sources": set()
        }
        
        for feature in feature_metadata:
            # Count by data type
            data_type = feature.get("data_type", "unknown")
            summary["by_data_type"][data_type] = summary["by_data_type"].get(data_type, 0) + 1
            
            # Count by tags
            tags = json.loads(feature.get("tags", "[]"))
            for tag in tags:
                summary["by_tags"][tag] = summary["by_tags"].get(tag, 0) + 1
            
            # Collect data sources
            source = feature.get("source_dataset")
            if source:
                summary["data_sources"].add(source)
        
        summary["data_sources"] = list(summary["data_sources"])
        return summary
    
    def create_standard_feature_views(self) -> List[Dict]:
        """Create standard feature views for common ML tasks."""
        
        print(f"\n Creating Standard Feature Views")
        print("=" * 50)
        
        created_views = []
        
        # Get all available features
        all_features = self.feature_store.get_feature_metadata()
        
        if not all_features:
            print(" No features available for view creation")
            return created_views
        
        # Group features by dataset
        features_by_dataset = {}
        for feature in all_features:
            dataset = feature["source_dataset"]
            if dataset not in features_by_dataset:
                features_by_dataset[dataset] = []
            features_by_dataset[dataset].append(feature["feature_name"])
        
        # Create views for each dataset
        for dataset_name, feature_names in features_by_dataset.items():
            
            # 1. All Features View
            try:
                view_metadata = self.feature_store.create_feature_view(
                    view_name=f"{dataset_name}_all_features",
                    feature_list=feature_names,
                    dataset_name=dataset_name
                )
                created_views.append(view_metadata)
                print(f" Created view: {dataset_name}_all_features")
                
            except Exception as e:
                print(f" Failed to create all_features view for {dataset_name}: {str(e)}")
            
            # 2. Numerical Features Only
            try:
                numerical_features = []
                for feature in all_features:
                    if (feature["source_dataset"] == dataset_name and 
                        feature["data_type"] in ["numerical"]):
                        numerical_features.append(feature["feature_name"])
                
                if numerical_features:
                    view_metadata = self.feature_store.create_feature_view(
                        view_name=f"{dataset_name}_numerical_features",
                        feature_list=numerical_features,
                        dataset_name=dataset_name
                    )
                    created_views.append(view_metadata)
                    print(f" Created view: {dataset_name}_numerical_features ({len(numerical_features)} features)")
                
            except Exception as e:
                print(f" Failed to create numerical_features view for {dataset_name}: {str(e)}")
            
            # 3. Derived Features Only
            try:
                derived_features = []
                for feature in all_features:
                    if feature["source_dataset"] == dataset_name:
                        tags = json.loads(feature.get("tags", "[]"))
                        if any(tag in tags for tag in ["derived_ratio", "polynomial", "statistical", "binned"]):
                            derived_features.append(feature["feature_name"])
                
                if derived_features:
                    view_metadata = self.feature_store.create_feature_view(
                        view_name=f"{dataset_name}_derived_features",
                        feature_list=derived_features,
                        dataset_name=dataset_name
                    )
                    created_views.append(view_metadata)
                    print(f" Created view: {dataset_name}_derived_features ({len(derived_features)} features)")
                
            except Exception as e:
                print(f" Failed to create derived_features view for {dataset_name}: {str(e)}")
        
        return created_views
    
    def perform_feature_quality_analysis(self) -> Dict:
        """Perform quality analysis on stored features."""
        
        print(f"\n Performing Feature Quality Analysis")
        print("=" * 50)
        
        all_features = self.feature_store.get_feature_metadata()
        
        if not all_features:
            print(" No features available for quality analysis")
            return {}
        
        quality_report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_features_analyzed": len(all_features),
            "quality_summary": {
                "high_quality": 0,
                "medium_quality": 0,
                "low_quality": 0
            },
            "issues_detected": [],
            "recommendations": []
        }
        
        for feature in all_features:
            try:
                # Get feature data for analysis
                feature_data = self.feature_store.get_feature_data(feature["feature_id"])
                
                if feature_data is None:
                    continue
                
                # Calculate quality metrics
                completeness = 1 - (feature_data.isnull().sum() / len(feature_data))
                uniqueness = feature_data.nunique() / len(feature_data) if len(feature_data) > 0 else 0
                
                # Classify quality
                if completeness >= 0.95 and uniqueness >= 0.01:
                    quality_report["quality_summary"]["high_quality"] += 1
                elif completeness >= 0.80:
                    quality_report["quality_summary"]["medium_quality"] += 1
                else:
                    quality_report["quality_summary"]["low_quality"] += 1
                    quality_report["issues_detected"].append({
                        "feature_name": feature["feature_name"],
                        "issue": "Low completeness",
                        "completeness": round(completeness, 3)
                    })
                
                # Check for potential issues
                if uniqueness < 0.001:
                    quality_report["issues_detected"].append({
                        "feature_name": feature["feature_name"],
                        "issue": "Very low uniqueness",
                        "uniqueness": round(uniqueness, 3)
                    })
                
            except Exception as e:
                quality_report["issues_detected"].append({
                    "feature_name": feature["feature_name"],
                    "issue": f"Analysis failed: {str(e)}"
                })
        
        # Generate recommendations
        high_quality_ratio = quality_report["quality_summary"]["high_quality"] / len(all_features)
        
        if high_quality_ratio >= 0.8:
            quality_report["recommendations"].append("Feature quality is excellent - ready for production use")
        elif high_quality_ratio >= 0.6:
            quality_report["recommendations"].append("Feature quality is good - consider addressing medium/low quality features")
        else:
            quality_report["recommendations"].append("Feature quality needs improvement - review data processing pipeline")
        
        if quality_report["issues_detected"]:
            quality_report["recommendations"].append("Address detected quality issues before model training")
        
        # Save quality report
        quality_report_path = os.path.join(
            self.config.FEATURE_STORE_DIR,
            "feature_quality_report.json"
        )
        with open(quality_report_path, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        print(f" Quality Analysis Results:")
        print(f"    High Quality: {quality_report['quality_summary']['high_quality']}")
        print(f"    Medium Quality: {quality_report['quality_summary']['medium_quality']}")
        print(f"    Low Quality: {quality_report['quality_summary']['low_quality']}")
        print(f"    Issues Detected: {len(quality_report['issues_detected'])}")
        
        return quality_report
    
    def run_complete_feature_store_pipeline(self) -> Dict:
        """
        Run complete feature store pipeline.
        
        Returns:
            Complete pipeline results
        """
        print(" Starting Complete Feature Store Pipeline")
        print("=" * 80)
        
        # Find transformed datasets
        datasets = self.find_transformed_datasets()
        
        if not datasets:
            print(" No transformed datasets found for feature store registration")
            return {}
        
        print(f" Found {len(datasets)} datasets for feature store registration:")
        for dataset in datasets:
            print(f"   {dataset['name']} ({dataset['size_mb']} MB)")
        
        # Register features from each dataset
        registration_results = []
        successful_registrations = 0
        
        for dataset_info in datasets:
            result = self.register_dataset_features(dataset_info)
            registration_results.append(result)
            
            if result.get("status") == "SUCCESS":
                successful_registrations += 1
        
        # Create standard feature views
        print(f"\n{'='*80}")
        created_views = self.create_standard_feature_views()
        
        # Generate feature catalog
        print(f"\n{'='*80}")
        feature_catalog = self.feature_store.generate_feature_catalog()
        
        # Perform quality analysis
        print(f"\n{'='*80}")
        quality_report = self.perform_feature_quality_analysis()
        
        # Generate overall summary
        overall_summary = {
            "pipeline_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_datasets": len(datasets),
                "successful_registrations": successful_registrations,
                "failed_registrations": len(datasets) - successful_registrations,
                "success_rate": round(successful_registrations / len(datasets) * 100, 1) if datasets else 0,
                "total_features_registered": sum(r.get("features_registered", 0) for r in registration_results if r.get("status") == "SUCCESS"),
                "feature_views_created": len(created_views)
            },
            "registration_results": registration_results,
            "feature_views": created_views,
            "feature_catalog": feature_catalog,
            "quality_report": quality_report,
            "recommendations": self._generate_pipeline_recommendations(registration_results, quality_report)
        }
        
        # Save overall summary
        summary_file = os.path.join(
            self.config.FEATURE_STORE_DIR,
            f"feature_store_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(summary_file, 'w') as f:
            json.dump(overall_summary, f, indent=2, default=str)
        
        # Print final summary
        self._print_final_summary(overall_summary)
        
        print(f"\n Complete summary saved: {summary_file}")
        print(f" Feature store pipeline completed!")
        
        # Close feature store connection
        self.feature_store.close()
        
        return overall_summary
    
    def _generate_pipeline_recommendations(self, registration_results: List[Dict], quality_report: Dict) -> List[str]:
        """Generate recommendations based on pipeline results."""
        recommendations = []
        
        successful_results = [r for r in registration_results if r.get("status") == "SUCCESS"]
        
        if not successful_results:
            recommendations.append("No successful feature registrations - review error logs and fix issues")
            return recommendations
        
        # Feature count analysis
        total_features = sum(r.get("features_registered", 0) for r in successful_results)
        avg_features_per_dataset = total_features / len(successful_results)
        
        if total_features > 200:
            recommendations.append("Large number of features registered - consider feature selection for optimal model performance")
        elif total_features < 50:
            recommendations.append("Moderate feature count - consider additional feature engineering if model performance is suboptimal")
        
        # Quality analysis
        high_quality_ratio = quality_report["quality_summary"]["high_quality"] / quality_report["total_features_analyzed"] if quality_report["total_features_analyzed"] > 0 else 0
        
        if high_quality_ratio >= 0.8:
            recommendations.append("Excellent feature quality - features ready for production ML models")
        elif high_quality_ratio >= 0.6:
            recommendations.append("Good feature quality - review and improve medium/low quality features")
        else:
            recommendations.append("Feature quality needs attention - review data preprocessing and feature engineering steps")
        
        # General recommendations
        recommendations.extend([
            "Feature store successfully established - ready for ML model training",
            "Monitor feature usage and performance in production models",
            "Consider implementing automated feature quality monitoring",
            "Establish feature governance and documentation practices"
        ])
        
        return recommendations
    
    def _print_final_summary(self, summary: Dict):
        """Print final feature store summary."""
        print(f"\n{'='*80}")
        print(" FEATURE STORE PIPELINE SUMMARY")
        print("=" * 80)
        
        pipeline_summary = summary["pipeline_summary"]
        print(f"Total Datasets: {pipeline_summary['total_datasets']}")
        print(f"Successful Registrations: {pipeline_summary['successful_registrations']}")
        print(f"Failed Registrations: {pipeline_summary['failed_registrations']}")
        print(f"Success Rate: {pipeline_summary['success_rate']}%")
        print(f"Total Features Registered: {pipeline_summary['total_features_registered']}")
        print(f"Feature Views Created: {pipeline_summary['feature_views_created']}")
        
        # Quality summary
        quality = summary["quality_report"]["quality_summary"]
        print(f"\n FEATURE QUALITY SUMMARY:")
        print(f"    High Quality: {quality['high_quality']}")
        print(f"    Medium Quality: {quality['medium_quality']}")
        print(f"    Low Quality: {quality['low_quality']}")
        
        print(f"\n RECOMMENDATIONS:")
        for i, rec in enumerate(summary["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n FEATURE STORE LOCATION:")
        print(f"   Database: {self.config.DATABASE_CONFIG['database_path']}")
        print(f"   Metadata: {self.config.FEATURE_METADATA_DIR}")
        print(f"   Registry: {self.config.FEATURE_REGISTRY_DIR}")


def main():
    """Main function to run feature store pipeline."""
    orchestrator = FeatureStoreOrchestrator()
    results = orchestrator.run_complete_feature_store_pipeline()
    return results


if __name__ == "__main__":
    main()
