# Complete End-to-End ML Pipeline for Customer Churn Prediction

**Student IDs: 2024ab05134, 2024aa05664**  
**Course: Data Management for Machine Learning**  
**Assignment: End-to-End ML Pipeline Implementation**  
**Submission Date: August 2025**

## Project Overview

This project implements a comprehensive, production-ready machine learning pipeline for predicting customer churn. Through this assignment, we have applied theoretical concepts from our coursework to build a real-world ML system that demonstrates advanced data management, feature engineering, model training, and automated orchestration capabilities.

### Personal Learning Journey

This project has been an incredible learning experiance that challenged us to think beyond just training models. We learned to consider the entire data science lifecycle including data ingestion, quality validation, feature engineering, model comparison, and production deployment.

**Key Challenges We Overcame:**
- Handling API authentication issues with Kaggle (solved with fallback URLs)
- Managing categorical data types across different pipeline steps
- Implementing robust error handling and retry logic
- Balancing feature engineering complexity with model interpretability
- Designing a scalable pipeline architecture that could handle different datasets

### Key Achievements

- **Complete ML Pipeline**: 10-step end-to-end automation
- **Multi-Source Data**: Kaggle + Hugging Face integration
- **204 Engineered Features**: Advanced feature engineering across 2 datasets
- **8 Trained Models**: 4 algorithms × 2 datasets with hyperparameter optimization
- **Production Architecture**: Feature store, versioning, orchestration
- **Grade A Data Quality**: 95% overall quality score with comprehensive validation

## Pipeline Architecture

```
┌─────────────────┬─────────────────┬─────────────────────────┐
│   Data Sources  │  Data Processing │    ML & Deployment      │
├─────────────────┼─────────────────┼─────────────────────────┤
│ Kaggle API      │ Data Cleaning   │ Model Training           │
│ Hugging Face    │ Validation      │ Feature Store            │
│ REST APIs       │ Feature Eng.    │ Version Control          │
│                 │ EDA Analysis    │ Orchestration            │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## Project Structure

```
DMML/
├── documentation/               # Problem formulation & docs
├── directory_2_data_ingestion/  # Multi-source data ingestion
├── directory_3_raw_data_storage/ # Partitioned data lake
├── directory_4_data_validation/ # Quality checks & validation
├── directory_5_data_preparation/# Data cleaning & EDA
├── directory_6_data_transformation/ # Feature engineering
├── directory_7_feature_store/   # Centralized feature management
├── directory_8_data_versioning/ # DVC version control
├── directory_9_model_building/  # ML model training
├── directory_10_pipeline_orchestration/ # Automated execution
├── data_lake/                   # Organized data storage
├── models/                      # Trained model artifacts
├── eda_reports/                 # EDA visualizations
└── logs/                        # Comprehensive logging
```

## Quick Start

### Prerequisites
```bash
pip install pandas scikit-learn xgboost mlflow dvc plotly seaborn kaggle datasets
```

### Run Complete Pipeline
```bash
# Execute end-to-end pipeline
cd directory_10_pipeline_orchestration
python pipeline_orchestrator.py
```

### Run Individual Steps
```bash
# Data ingestion
cd directory_2_data_ingestion && python main_ingestion.py

# Model training  
cd directory_9_model_building && python model_orchestrator.py

# Feature engineering
cd directory_6_data_transformation && python transformation_orchestrator.py
```

## Results & Performance

### Model Performance
| Dataset | Best Model | F1 Score | ROC AUC | Accuracy |
|---------|------------|----------|---------|----------|
| **Telco Churn** | Gradient Boosting | 0.530 | 0.828 | 78.7% |
| **Adult Census** | XGBoost | 0.693 | 0.916 | 86.4% |

### Pipeline Metrics
- **Data Sources**: 2 (Kaggle Telco + Hugging Face Census)
- **Total Records**: ~40K across datasets
- **Features Engineered**: 204 (33→104 + 32→100)
- **Models Trained**: 8 (4 algorithms × 2 datasets)
- **Data Quality Score**: Grade A (95%)
- **Pipeline Success Rate**: 100% (individual steps)

### Technical Highlights
- **Advanced Feature Engineering**: Polynomial, statistical, interaction features
- **Comprehensive Validation**: Missing values, outliers, schema validation
- **MLflow Integration**: Experiment tracking and model versioning
- **Automated Orchestration**: Dependency management, retry logic, monitoring

## System Components

### 1. Data Ingestion (`directory_2_data_ingestion`)
- **Sources**: Kaggle API, Hugging Face datasets, REST APIs
- **Features**: Fallback URLs, authentication handling, file validation
- **Output**: Raw CSV files with metadata

### 2. Data Lake Storage (`directory_3_raw_data_storage`) 
- **Structure**: Partitioned by source/type/timestamp
- **Format**: Parquet for efficient storage
- **Catalog**: JSON metadata with statistics

### 3. Data Validation (`directory_4_data_validation`)
- **Checks**: Missing values, duplicates, schema adherence, ranges
- **Reports**: Excel, JSON, and text summaries
- **Quality Score**: Comprehensive scoring system

### 4. Data Preparation (`directory_5_data_preparation`)
- **Cleaning**: Missing value imputation, outlier handling
- **EDA**: Statistical analysis and visualizations
- **Output**: Clean datasets ready for transformation

### 5. Feature Engineering (`directory_6_data_transformation`)
- **Advanced Features**: Binning, ratios, interactions, polynomial
- **Statistical**: Z-scores, percentile ranks, rolling statistics
- **Selection**: Mutual information-based feature selection

### 6. Feature Store (`directory_7_feature_store`)
- **SQLite Backend**: Centralized feature metadata
- **Views**: Organized feature collections for ML
- **Lineage**: Complete feature transformation tracking

### 7. Version Control (`directory_8_data_versioning`)
- **DVC Integration**: Data version control with Git
- **Metadata**: Complete version history and lineage
- **Documentation**: Comprehensive versioning guides

### 8. Model Building (`directory_9_model_building`)
- **Algorithms**: Logistic Regression, Random Forest, XGBoost, Gradient Boosting
- **Optimization**: Grid search hyperparameter tuning
- **Evaluation**: Comprehensive metrics and visualizations
- **MLflow**: Experiment tracking and model registry

### 9. Pipeline Orchestration (`directory_10_pipeline_orchestration`)
- **Dependency Management**: Automatic execution ordering
- **Error Handling**: Retry logic with exponential backoff
- **Monitoring**: Memory usage, execution time tracking
- **Reporting**: Detailed JSON reports with recommendations

## Business Value

### Customer Churn Prevention
- **Predictive Models**: Identify at-risk customers with 82% ROC AUC
- **Feature Insights**: 204 engineered features for comprehensive analysis
- **Actionable Intelligence**: Feature importance and statistical analysis

### Operational Excellence  
- **Automated Pipeline**: End-to-end execution with minimal manual intervention
- **Quality Assurance**: Comprehensive validation and monitoring
- **Scalability**: Modular architecture supporting growth
- **Reproducibility**: Complete version control and metadata tracking

### Data Governance
- **Centralized Features**: Feature store with lineage tracking
- **Quality Monitoring**: Automated data quality assessment
- **Documentation**: Comprehensive documentation and reporting
- **Compliance**: Audit trails and version history

## Technical Implementation

### Core Technologies
- **Python**: Primary development language
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn/XGBoost**: Machine learning algorithms
- **MLflow**: Experiment tracking and model management
- **DVC**: Data version control
- **SQLite**: Feature store backend
- **Plotly/Seaborn**: Data visualization

### Architecture Patterns
- **Modular Design**: Separate components with clear interfaces
- **Configuration-Driven**: Extensive configuration management
- **Error Handling**: Comprehensive logging and error recovery
- **Monitoring**: Performance and resource tracking
- **Documentation**: Automated documentation generation

## Usage Examples

### Running Specific Pipeline Steps
```python
# Feature engineering
from directory_6_data_transformation import TransformationOrchestrator
orchestrator = TransformationOrchestrator()
results = orchestrator.run_complete_transformation()

# Model training
from directory_9_model_building import ModelOrchestrator  
model_orch = ModelOrchestrator()
model_results = model_orch.run_complete_model_building()
```

### Accessing Feature Store
```python
from directory_7_feature_store import FeatureStore
store = FeatureStore()

# Get feature metadata
features = store.get_feature_metadata()

# Create feature view
view = store.create_feature_view(
    "churn_features", 
    ["tenure", "monthly_charges", "contract_type"]
)
```

### Pipeline Orchestration
```python
from directory_10_pipeline_orchestration import PipelineOrchestrator
orchestrator = PipelineOrchestrator()

# Run complete pipeline
results = orchestrator.execute_pipeline()

# Run specific steps
model_results = orchestrator.execute_single_step("model_building")
```

## Monitoring & Observability

### Pipeline Metrics
- **Execution Time**: Step-by-step timing analysis
- **Memory Usage**: Resource consumption tracking
- **Success Rates**: Pipeline reliability metrics
- **Data Quality**: Continuous quality monitoring

### Model Performance
- **Accuracy Metrics**: Precision, recall, F1-score, ROC AUC
- **Feature Importance**: Top contributing features
- **Cross-Validation**: Robust performance estimation
- **MLflow Tracking**: Complete experiment history

## Future Enhancements

### Immediate Opportunities
- **Real-time Streaming**: Apache Kafka integration
- **Cloud Deployment**: AWS/GCP infrastructure
- **A/B Testing**: Model comparison framework
- **API Services**: RESTful model serving

### Advanced Features
- **AutoML**: Automated hyperparameter optimization
- **Deep Learning**: Neural network integration
- **Ensemble Methods**: Advanced model combination
- **Drift Detection**: Data and model drift monitoring

## Contributing

### Development Setup
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python -m pytest tests/`
4. Execute pipeline: `python directory_10_pipeline_orchestration/pipeline_orchestrator.py`

### Code Standards
- **PEP 8**: Python style guidelines
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Docstrings for all modules
- **Testing**: Unit tests for critical components

## License & Acknowledgments

### Data Sources
- **Kaggle**: Telco Customer Churn Dataset
- **Hugging Face**: Adult Census Income Dataset

### Technologies
- Built with Python, Scikit-learn, XGBoost, MLflow, DVC
- Visualization with Plotly, Seaborn, Matplotlib
- Infrastructure with SQLite, Pandas, NumPy

---

## Project Success Summary

This project successfully demonstrates a **complete, production-ready ML pipeline** with:

- **10 Pipeline Steps** - Fully automated end-to-end execution  
- **204 Features** - Advanced feature engineering across datasets  
- **8 Models** - Multiple algorithms with optimization  
- **Grade A Quality** - 95% data quality score  
- **Enterprise Architecture** - Feature store, versioning, orchestration  

**Ready for production deployment and scaling!**
