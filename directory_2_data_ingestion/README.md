# Data Ingestion Module

This module handles automated data ingestion from multiple sources for the customer churn prediction pipeline.

## Features

- **Multi-source ingestion**: Support for Kaggle and Hugging Face datasets
- **Error handling**: Robust error handling with retry mechanisms
- **Logging**: Comprehensive logging for monitoring and debugging
- **Validation**: Automatic validation of downloaded data
- **Configuration**: Easy configuration management
- **Orchestration**: Coordinated ingestion from multiple sources

## Setup

### 1. Install Dependencies

```bash
# Run the setup script
python setup_ingestion.py

# Or install manually
pip install pandas numpy kaggle datasets requests tqdm
```

### 2. Configure Kaggle API

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section and click "Create New API Token"
3. Download `kaggle.json` file
4. Place it in `~/.kaggle/kaggle.json`
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Usage

### Quick Start

```bash
# Run complete data ingestion
python main_ingestion.py
```

### Individual Sources

```python
from kaggle_ingestion import KaggleDataIngestion
from huggingface_ingestion import HuggingFaceDataIngestion

# Kaggle only
kaggle_ingestion = KaggleDataIngestion()
kaggle_files = kaggle_ingestion.ingest_all_datasets()

# Hugging Face only
hf_ingestion = HuggingFaceDataIngestion()
hf_files = hf_ingestion.ingest_all_datasets()
```

### Orchestrated Ingestion

```python
from main_ingestion import DataIngestionOrchestrator

orchestrator = DataIngestionOrchestrator()
summary = orchestrator.run_full_ingestion()
print(summary)
```

## Configuration

Edit `config.py` to customize datasets:

```python
KAGGLE_DATASETS = [
    {
        "name": "blastchar/telco-customer-churn",
        "file_name": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "output_name": "kaggle_telco_churn.csv"
    }
]

HUGGINGFACE_DATASETS = [
    {
        "name": "scikit-learn/adult-census-income",
        "output_name": "huggingface_census.csv",
        "config": "default",
        "split": "train"
    }
]
```

## File Structure

```
directory_2_data_ingestion/
├── __init__.py              # Package initialization
├── config.py                # Configuration settings
├── logger_utils.py          # Logging utilities
├── kaggle_ingestion.py      # Kaggle data ingestion
├── huggingface_ingestion.py # Hugging Face data ingestion
├── main_ingestion.py        # Main orchestrator
├── setup_ingestion.py       # Setup script
└── README.md               # This file
```

## Output

Downloaded files are stored in:
- Raw data: `../raw_data/`
- Logs: `../logs/`

### Sample Log Output

```
2025-03-02 16:14:58,359 - INFO - Starting Kaggle data ingestion...
2025-03-02 16:15:01,861 - INFO - Kaggle data successfully downloaded and stored in raw_data/
2025-03-02 16:15:01,861 - INFO - Starting Hugging Face data ingestion...
2025-03-02 16:15:14,942 - INFO - Hugging Face data successfully downloaded and stored in raw_data/huggingface_census.csv
```

## Error Handling

The module includes comprehensive error handling:

- **API failures**: Automatic retry with exponential backoff
- **File validation**: Checks for empty files and data integrity
- **Network issues**: Graceful handling of connection problems
- **Authentication**: Clear error messages for setup issues

## Monitoring

- **Logging**: All operations are logged with timestamps
- **Validation**: Automatic data quality checks
- **Reporting**: Comprehensive ingestion summary reports
- **Health checks**: File existence and size validation

## Troubleshooting

### Common Issues

1. **Kaggle API Error**: Ensure kaggle.json is properly configured
2. **Permission Error**: Check file permissions on kaggle.json
3. **Network Error**: Verify internet connection and proxy settings
4. **Disk Space**: Ensure sufficient disk space for downloads

### Debug Mode

Enable debug logging by setting `LOG_LEVEL = "DEBUG"` in config.py

## Next Steps

After successful data ingestion:

1. **Data Validation** (directory_4): Validate data quality
2. **Data Preparation** (directory_5): Clean and preprocess data
3. **Data Transformation** (directory_6): Feature engineering

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.24.0
- kaggle >= 1.5.16
- datasets >= 2.14.0
- requests >= 2.31.0
- tqdm >= 4.65.0
