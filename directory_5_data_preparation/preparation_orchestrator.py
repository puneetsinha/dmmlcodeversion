"""
Data Preparation Orchestrator - Main script for data cleaning and EDA.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_cleaner import DataCleaner
from eda_analyzer import EDAAnalyzer
from preparation_config import preparation_config


class DataPreparationOrchestrator:
    """Orchestrate data cleaning and EDA pipeline."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.config = preparation_config
        self.cleaner = DataCleaner()
        self.eda_analyzer = EDAAnalyzer()
        self.preparation_results = {}
        
    def find_datasets(self) -> List[Dict[str, str]]:
        """Find datasets in the data lake."""
        datasets = []
        
        if not os.path.exists(self.config.DATA_LAKE_DIR):
            print(f" Data lake directory not found: {self.config.DATA_LAKE_DIR}")
            return datasets
        
        # Walk through data lake structure
        for root, dirs, files in os.walk(self.config.DATA_LAKE_DIR):
            for file in files:
                if file.endswith('.parquet'):
                    file_path = os.path.join(root, file)
                    
                    # Extract dataset info
                    if 'telco_customer_churn' in file:
                        dataset_name = 'telco_customer_churn'
                        target_col = 'Churn'
                    elif 'adult_census_income' in file:
                        dataset_name = 'adult_census_income'
                        target_col = 'income'
                    else:
                        dataset_name = file.replace('.parquet', '').split('_')[0]
                        target_col = None
                    
                    datasets.append({
                        'name': dataset_name,
                        'path': file_path,
                        'target_column': target_col,
                        'size_mb': round(os.path.getsize(file_path) / (1024*1024), 2)
                    })
        
        return datasets
    
    def prepare_dataset(self, dataset_info: Dict) -> Dict:
        """
        Prepare a single dataset (clean + EDA).
        
        Args:
            dataset_info: Dataset information dictionary
            
        Returns:
            Preparation results
        """
        dataset_name = dataset_info['name']
        dataset_path = dataset_info['path']
        target_col = dataset_info['target_column']
        
        print(f"\n{'='*60}")
        print(f" PREPARING DATASET: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Clean the dataset
            df_original, df_standardized, cleaning_summary = self.cleaner.clean_dataset(
                dataset_path, dataset_name
            )
            
            # Step 2: Perform EDA on cleaned data
            eda_results = self.eda_analyzer.perform_comprehensive_eda(
                df_original, dataset_name, target_col
            )
            
            # Step 3: Save cleaned datasets
            self._save_cleaned_datasets(df_original, df_standardized, dataset_name)
            
            # Step 4: Generate preparation summary
            preparation_summary = {
                "dataset_name": dataset_name,
                "original_path": dataset_path,
                "target_column": target_col,
                "preparation_timestamp": datetime.now().isoformat(),
                "cleaning_summary": cleaning_summary,
                "eda_results": eda_results,
                "output_files": {
                    "cleaned_data": os.path.join(self.config.PROCESSED_DATA_DIR, f"{dataset_name}_cleaned.csv"),
                    "standardized_data": os.path.join(self.config.PROCESSED_DATA_DIR, f"{dataset_name}_standardized.csv"),
                    "cleaning_log": os.path.join(self.config.PROCESSED_DATA_DIR, f"{dataset_name}_cleaning_log.json"),
                    "eda_report": os.path.join(self.config.EDA_REPORTS_DIR, f"{dataset_name}_eda_report.json")
                }
            }
            
            # Save individual reports
            self._save_preparation_reports(preparation_summary)
            
            print(f" Dataset preparation completed: {dataset_name}")
            return preparation_summary
            
        except Exception as e:
            error_summary = {
                "dataset_name": dataset_name,
                "error": str(e),
                "status": "FAILED",
                "timestamp": datetime.now().isoformat()
            }
            print(f" Dataset preparation failed: {dataset_name} - {str(e)}")
            return error_summary
    
    def _save_cleaned_datasets(self, df_original: pd.DataFrame, df_standardized: pd.DataFrame, dataset_name: str):
        """Save cleaned datasets to processed data directory."""
        # Save original cleaned version
        cleaned_file = os.path.join(self.config.PROCESSED_DATA_DIR, f"{dataset_name}_cleaned.csv")
        df_original.to_csv(cleaned_file, index=False)
        
        # Save standardized version
        standardized_file = os.path.join(self.config.PROCESSED_DATA_DIR, f"{dataset_name}_standardized.csv")
        df_standardized.to_csv(standardized_file, index=False)
        
        print(f" Saved cleaned datasets:")
        print(f"    Cleaned: {cleaned_file}")
        print(f"    Standardized: {standardized_file}")
    
    def _save_preparation_reports(self, preparation_summary: Dict):
        """Save preparation reports and logs."""
        dataset_name = preparation_summary["dataset_name"]
        
        # Save cleaning log
        cleaning_log_file = os.path.join(self.config.PROCESSED_DATA_DIR, f"{dataset_name}_cleaning_log.json")
        with open(cleaning_log_file, 'w') as f:
            json.dump(preparation_summary["cleaning_summary"], f, indent=2, default=str)
        
        # Save EDA report
        eda_report_file = os.path.join(self.config.EDA_REPORTS_DIR, f"{dataset_name}_eda_report.json")
        with open(eda_report_file, 'w') as f:
            json.dump(preparation_summary["eda_results"], f, indent=2, default=str)
        
        print(f" Saved preparation reports:")
        print(f"    Cleaning log: {cleaning_log_file}")
        print(f"    EDA report: {eda_report_file}")
    
    def generate_preparation_summary(self, all_results: List[Dict]) -> Dict:
        """Generate overall preparation summary."""
        successful_preparations = [r for r in all_results if "error" not in r]
        failed_preparations = [r for r in all_results if "error" in r]
        
        summary = {
            "preparation_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_datasets": len(all_results),
                "successful_preparations": len(successful_preparations),
                "failed_preparations": len(failed_preparations),
                "success_rate": round(len(successful_preparations) / len(all_results) * 100, 1) if all_results else 0
            },
            "dataset_results": all_results,
            "recommendations": self._generate_overall_recommendations(successful_preparations)
        }
        
        return summary
    
    def _generate_overall_recommendations(self, successful_results: List[Dict]) -> List[str]:
        """Generate overall recommendations based on all datasets."""
        recommendations = []
        
        if not successful_results:
            recommendations.append("No successful preparations to analyze")
            return recommendations
        
        # Analyze cleaning patterns
        total_cleaning_ops = 0
        common_issues = {}
        
        for result in successful_results:
            cleaning_ops = result["cleaning_summary"]["cleaning_operations"]
            total_cleaning_ops += len(cleaning_ops)
            
            for op in cleaning_ops:
                op_type = op["operation"]
                if op_type in common_issues:
                    common_issues[op_type] += 1
                else:
                    common_issues[op_type] = 1
        
        # Generate recommendations based on common issues
        if common_issues.get("MISSING_VALUES", 0) > 0:
            recommendations.append("Missing values detected across datasets - ensure robust imputation strategies")
        
        if common_issues.get("OUTLIERS", 0) > 0:
            recommendations.append("Outliers found in multiple datasets - consider domain-specific handling")
        
        if common_issues.get("ENCODING", 0) > 0:
            recommendations.append("Categorical encoding applied - verify encoding strategies for model compatibility")
        
        # EDA-based recommendations
        feature_importance_available = any(
            result["eda_results"]["feature_importance"] for result in successful_results
        )
        
        if feature_importance_available:
            recommendations.append("Feature importance calculated - use for feature selection in modeling")
        
        recommendations.append("Data preparation completed successfully - datasets ready for transformation")
        
        return recommendations
    
    def run_complete_preparation(self) -> Dict:
        """
        Run complete data preparation pipeline.
        
        Returns:
            Complete preparation results
        """
        print(" Starting Complete Data Preparation Pipeline")
        print("=" * 70)
        
        # Find datasets
        datasets = self.find_datasets()
        
        if not datasets:
            print(" No datasets found for preparation")
            return {}
        
        print(f" Found {len(datasets)} datasets for preparation:")
        for dataset in datasets:
            print(f"   {dataset['name']} ({dataset['size_mb']} MB) - Target: {dataset['target_column']}")
        
        # Prepare each dataset
        all_results = []
        
        for dataset_info in datasets:
            result = self.prepare_dataset(dataset_info)
            all_results.append(result)
        
        # Generate overall summary
        overall_summary = self.generate_preparation_summary(all_results)
        
        # Save overall summary
        summary_file = os.path.join(
            self.config.PROCESSED_DATA_DIR,
            f"preparation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(summary_file, 'w') as f:
            json.dump(overall_summary, f, indent=2, default=str)
        
        # Print final summary
        self._print_final_summary(overall_summary)
        
        print(f"\n Complete summary saved: {summary_file}")
        print(f" Data preparation pipeline completed!")
        
        return overall_summary
    
    def _print_final_summary(self, summary: Dict):
        """Print final preparation summary."""
        print(f"\n{'='*70}")
        print(" DATA PREPARATION SUMMARY")
        print("=" * 70)
        
        prep_summary = summary["preparation_summary"]
        print(f"Total Datasets: {prep_summary['total_datasets']}")
        print(f"Successful: {prep_summary['successful_preparations']}")
        print(f"Failed: {prep_summary['failed_preparations']}")
        print(f"Success Rate: {prep_summary['success_rate']}%")
        
        print(f"\n RECOMMENDATIONS:")
        for i, rec in enumerate(summary["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n OUTPUT DIRECTORIES:")
        print(f"   Processed Data: {self.config.PROCESSED_DATA_DIR}")
        print(f"   EDA Reports: {self.config.EDA_REPORTS_DIR}")


def main():
    """Main function to run data preparation."""
    orchestrator = DataPreparationOrchestrator()
    results = orchestrator.run_complete_preparation()
    return results


if __name__ == "__main__":
    main()
