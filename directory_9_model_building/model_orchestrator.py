"""
Model Building Orchestrator.
Coordinates the complete model training pipeline for all datasets.
"""

import os
import json
import sys
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_trainer import ModelTrainer
from model_config import model_config


class ModelOrchestrator:
    """Orchestrate complete model building pipeline."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.config = model_config
        self.trainer = ModelTrainer()
        self.orchestration_results = {}
        
    def find_available_datasets(self) -> List[str]:
        """Find datasets available for model training."""
        datasets = []
        
        # Check feature store for available datasets
        feature_store_views = os.path.join(self.config.FEATURE_STORE_DIR, "registry", "views")
        
        if os.path.exists(feature_store_views):
            for file in os.listdir(feature_store_views):
                if file.endswith('_all_features.parquet'):
                    dataset_name = file.replace('_all_features.parquet', '')
                    datasets.append(dataset_name)
        
        return datasets
    
    def create_model_comparison_report(self, all_training_results: Dict) -> Dict:
        """Create comprehensive model comparison report."""
        
        print("\n Creating Model Comparison Report")
        print("=" * 50)
        
        comparison_data = []
        
        for dataset_name, dataset_results in all_training_results.items():
            model_results = dataset_results.get("model_results", {})
            
            for model_name, model_data in model_results.items():
                metrics = model_data.get("metrics", {})
                
                comparison_data.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "accuracy": metrics.get("accuracy", 0),
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0), 
                    "f1_score": metrics.get("f1_score", 0),
                    "roc_auc": metrics.get("roc_auc", 0),
                    "training_time": model_data.get("training_time", 0)
                })
        
        if not comparison_data:
            return {"error": "No model results available for comparison"}
        
        # Create comparison report
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "total_models": len(comparison_data),
            "datasets_processed": len(all_training_results),
            "model_comparison": comparison_data,
            "best_models": {},
            "performance_summary": {},
            "recommendations": []
        }
        
        # Find best models per dataset
        for dataset_name in all_training_results.keys():
            dataset_models = [row for row in comparison_data if row["dataset"] == dataset_name]
            if dataset_models:
                best_model = max(dataset_models, key=lambda x: x["f1_score"])
                report["best_models"][dataset_name] = {
                    "model_name": best_model["model"],
                    "f1_score": best_model["f1_score"],
                    "roc_auc": best_model["roc_auc"],
                    "accuracy": best_model["accuracy"]
                }
        
        # Calculate performance summary
        if comparison_data:
            avg_metrics = {}
            for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
                values = [row[metric] for row in comparison_data if row[metric] > 0]
                if values:
                    avg_metrics[f"avg_{metric}"] = round(sum(values) / len(values), 4)
                    avg_metrics[f"max_{metric}"] = round(max(values), 4)
                    avg_metrics[f"min_{metric}"] = round(min(values), 4)
            
            report["performance_summary"] = avg_metrics
        
        # Generate recommendations
        report["recommendations"] = self._generate_model_recommendations(comparison_data, report["best_models"])
        
        return report
    
    def _generate_model_recommendations(self, comparison_data: List[Dict], best_models: Dict) -> List[str]:
        """Generate model recommendations based on results."""
        recommendations = []
        
        if not comparison_data:
            return ["No model results available for analysis"]
        
        # Analyze overall performance
        avg_f1 = sum(row["f1_score"] for row in comparison_data) / len(comparison_data)
        
        if avg_f1 > 0.8:
            recommendations.append("Excellent model performance across datasets - ready for production deployment")
        elif avg_f1 > 0.6:
            recommendations.append("Good model performance - consider feature engineering improvements for better results")
        else:
            recommendations.append("Model performance needs improvement - review data quality and feature engineering")
        
        # Model-specific recommendations
        model_performance = {}
        for row in comparison_data:
            model = row["model"]
            if model not in model_performance:
                model_performance[model] = []
            model_performance[model].append(row["f1_score"])
        
        # Find consistently best performing models
        avg_performance = {model: sum(scores)/len(scores) for model, scores in model_performance.items()}
        best_overall_model = max(avg_performance.keys(), key=lambda x: avg_performance[x])
        
        recommendations.append(f"'{best_overall_model}' shows best overall performance across datasets")
        
        # Dataset-specific recommendations
        for dataset, best_info in best_models.items():
            if best_info["f1_score"] > 0.8:
                recommendations.append(f"{dataset}: '{best_info['model_name']}' ready for production (F1: {best_info['f1_score']:.3f})")
            else:
                recommendations.append(f"{dataset}: Consider ensemble methods or additional feature engineering")
        
        return recommendations
    
    def create_model_visualizations(self, all_training_results: Dict):
        """Create comprehensive model performance visualizations."""
        
        print("\n Creating Model Performance Visualizations")
        print("=" * 50)
        
        # Prepare data for visualization
        plot_data = []
        
        for dataset_name, dataset_results in all_training_results.items():
            model_results = dataset_results.get("model_results", {})
            
            for model_name, model_data in model_results.items():
                metrics = model_data.get("metrics", {})
                plot_data.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Accuracy": metrics.get("accuracy", 0),
                    "Precision": metrics.get("precision", 0),
                    "Recall": metrics.get("recall", 0),
                    "F1 Score": metrics.get("f1_score", 0),
                    "ROC AUC": metrics.get("roc_auc", 0),
                    "Training Time": model_data.get("training_time", 0)
                })
        
        if not plot_data:
            print(" No data available for visualization")
            return
        
        # Create plots directory
        plots_dir = os.path.join(self.config.MODEL_REPORTS_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("viridis")
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ["Accuracy", "F1 Score", "ROC AUC", "Training Time"]
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i//2, i%2]
            
            # Create data for plotting
            datasets = list(set(row["Dataset"] for row in plot_data))
            models = list(set(row["Model"] for row in plot_data))
            
            # Create matrix for heatmap
            matrix_data = []
            for dataset in datasets:
                row_data = []
                for model in models:
                    value = next((row[metric] for row in plot_data 
                                if row["Dataset"] == dataset and row["Model"] == model), 0)
                    row_data.append(value)
                matrix_data.append(row_data)
            
            # Create heatmap
            sns.heatmap(matrix_data, 
                       xticklabels=models, 
                       yticklabels=datasets,
                       annot=True, 
                       fmt='.3f' if metric != "Training Time" else '.1f',
                       cmap='viridis',
                       ax=ax)
            ax.set_title(f'{metric} by Model and Dataset')
            ax.set_xlabel('Model')
            ax.set_ylabel('Dataset')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'model_performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model Ranking Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Calculate average F1 score for each model
        model_avg_f1 = {}
        for model in set(row["Model"] for row in plot_data):
            f1_scores = [row["F1 Score"] for row in plot_data if row["Model"] == model]
            model_avg_f1[model] = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        
        # Sort models by average F1 score
        sorted_models = sorted(model_avg_f1.items(), key=lambda x: x[1], reverse=True)
        
        models = [item[0] for item in sorted_models]
        f1_scores = [item[1] for item in sorted_models]
        
        bars = ax.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
        ax.set_title('Model Ranking by Average F1 Score', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Average F1 Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'model_ranking.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Visualizations saved to: {plots_dir}")
        print(f"    Performance comparison: model_performance_comparison.png")
        print(f"    Model ranking: model_ranking.png")
    
    def run_complete_model_building(self) -> Dict:
        """
        Run complete model building pipeline for all datasets.
        
        Returns:
            Complete model building results
        """
        print(" Starting Complete Model Building Pipeline")
        print("=" * 80)
        
        # Find available datasets
        datasets = self.find_available_datasets()
        
        if not datasets:
            print(" No datasets found for model training")
            return {}
        
        print(f" Found {len(datasets)} datasets for model training:")
        for dataset in datasets:
            print(f"   {dataset}")
        
        # Train models for each dataset
        all_training_results = {}
        successful_trainings = 0
        
        for dataset_name in datasets:
            try:
                print(f"\n{'='*80}")
                print(f" TRAINING MODELS FOR: {dataset_name}")
                print(f"{'='*80}")
                
                training_results = self.trainer.train_all_models(dataset_name)
                all_training_results[dataset_name] = training_results
                successful_trainings += 1
                
                print(f" Model training completed for {dataset_name}")
                
            except Exception as e:
                print(f" Model training failed for {dataset_name}: {str(e)}")
                continue
        
        # Create model comparison report
        print(f"\n{'='*80}")
        comparison_report = self.create_model_comparison_report(all_training_results)
        
        # Create visualizations
        self.create_model_visualizations(all_training_results)
        
        # Generate final summary
        final_summary = {
            "model_building_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_datasets": len(datasets),
                "successful_trainings": successful_trainings,
                "failed_trainings": len(datasets) - successful_trainings,
                "success_rate": round(successful_trainings / len(datasets) * 100, 1) if datasets else 0,
                "total_models_trained": sum(
                    len(result.get("model_results", {})) 
                    for result in all_training_results.values()
                )
            },
            "training_results": all_training_results,
            "comparison_report": comparison_report,
            "recommendations": self._generate_pipeline_recommendations(all_training_results, comparison_report)
        }
        
        # Save complete summary
        summary_file = os.path.join(
            self.config.MODEL_REPORTS_DIR,
            f"model_building_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(summary_file, 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)
        
        # Print final summary
        self._print_final_summary(final_summary)
        
        print(f"\n Complete summary saved: {summary_file}")
        print(f" Model building pipeline completed!")
        
        return final_summary
    
    def _generate_pipeline_recommendations(self, training_results: Dict, comparison_report: Dict) -> List[str]:
        """Generate recommendations for the complete pipeline."""
        recommendations = []
        
        if not training_results:
            recommendations.append("No successful model training - review data preparation and feature engineering")
            return recommendations
        
        # Overall success assessment
        total_models = sum(len(result.get("model_results", {})) for result in training_results.values())
        
        if total_models >= 8:  # 4 models × 2 datasets
            recommendations.append("Excellent! All models trained successfully across datasets")
        elif total_models >= 4:
            recommendations.append("Good progress - most models trained successfully")
        else:
            recommendations.append("Limited model training success - review system configuration")
        
        # Performance-based recommendations
        if comparison_report.get("performance_summary"):
            avg_f1 = comparison_report["performance_summary"].get("avg_f1_score", 0)
            max_f1 = comparison_report["performance_summary"].get("max_f1_score", 0)
            
            if max_f1 > 0.85:
                recommendations.append("Outstanding model performance achieved - ready for production deployment")
            elif max_f1 > 0.7:
                recommendations.append("Good model performance - consider hyperparameter optimization for improvement")
            else:
                recommendations.append("Model performance below expectations - review feature engineering and data quality")
        
        # Next steps
        recommendations.extend([
            "Models ready for pipeline orchestration and automation",
            "Set up model monitoring and retraining pipelines",
            "Implement A/B testing framework for model comparison",
            "Configure model serving infrastructure for production deployment"
        ])
        
        return recommendations
    
    def _print_final_summary(self, summary: Dict):
        """Print final model building summary."""
        print(f"\n{'='*80}")
        print(" MODEL BUILDING PIPELINE SUMMARY")
        print("=" * 80)
        
        building_summary = summary["model_building_summary"]
        print(f"Total Datasets: {building_summary['total_datasets']}")
        print(f"Successful Trainings: {building_summary['successful_trainings']}")
        print(f"Failed Trainings: {building_summary['failed_trainings']}")
        print(f"Success Rate: {building_summary['success_rate']}%")
        print(f"Total Models Trained: {building_summary['total_models_trained']}")
        
        # Show best models
        comparison_report = summary["comparison_report"]
        if comparison_report.get("best_models"):
            print(f"\n BEST MODELS BY DATASET:")
            for dataset, best_info in comparison_report["best_models"].items():
                print(f"   {dataset}:")
                print(f"      Model: {best_info['model_name']}")
                print(f"      F1 Score: {best_info['f1_score']:.4f}")
                print(f"      ROC AUC: {best_info['roc_auc']:.4f}")
        
        # Performance summary
        if comparison_report.get("performance_summary"):
            perf = comparison_report["performance_summary"]
            print(f"\n PERFORMANCE SUMMARY:")
            print(f"   Average F1 Score: {perf.get('avg_f1_score', 0):.4f}")
            print(f"   Maximum F1 Score: {perf.get('max_f1_score', 0):.4f}")
            print(f"   Average ROC AUC: {perf.get('avg_roc_auc', 0):.4f}")
        
        print(f"\n RECOMMENDATIONS:")
        for i, rec in enumerate(summary["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n OUTPUT LOCATIONS:")
        print(f"   Trained Models: {self.config.MODELS_DIR}/trained_models/")
        print(f"   Reports: {self.config.MODEL_REPORTS_DIR}/")
        print(f"   Visualizations: {self.config.MODEL_REPORTS_DIR}/plots/")


def main():
    """Main function to run model building pipeline."""
    orchestrator = ModelOrchestrator()
    results = orchestrator.run_complete_model_building()
    return results


if __name__ == "__main__":
    main()
