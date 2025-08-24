# Data Transformation Component

**Students: 2024ab05134, 2024aa05664**

## Overview

Data transformation is the process of converting data from its original format into a new structure or format that is more appropriate for analysis, reporting, or machine learning.

## Implementation

Our data transformation pipeline performs comprehensive feature engineering and transformations across multiple datasets.

## Common Data Transformation Tasks Implemented

| Task | Description | Example |
|------|-------------|---------|
| **Normalization** | Scaling numeric data to a standard range | Scale income to 0-1 range |
| **Standardization** | Shifting and scaling data to have mean=0, std dev=1 | Standardize test scores |
| **Encoding** | Converting categorical data into numerical form | "Yes"/"No" → 1/0 |
| **Aggregation** | Summarizing data | Total sales per month |
| **Pivoting/Unpivoting** | Restructuring data tables | Rows to columns and vice versa |
| **Filtering** | Removing irrelevant data | Only keep rows where status = "active" |
| **Date-Time Conversion** | Changing date formats or extracting components | "2025-08-18" → year: 2025 |

## Results

- **Adult Census Dataset**: 33 → 104 features (71 new features)
- **Telco Churn Dataset**: 32 → 100 features (68 new features)
- **Total Features Engineered**: 139 across both datasets
- **Success Rate**: 100%

## Usage

```bash
cd directory_6_data_transformation
python transformation_orchestrator.py
```

## Output Files

- Transformed CSV and Parquet files
- Feature metadata and definitions
- Comprehensive transformation report

See `DATA_TRANSFORMATION_REPORT.txt` for detailed implementation and results.
