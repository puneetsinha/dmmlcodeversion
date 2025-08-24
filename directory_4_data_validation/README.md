# Data Validation Component

**Students: 2024ab05134, 2024aa05664**

## Overview

Data validation is the process of ensuring that data is accurate, clean, and useful before it is processed or stored. It checks that the data entered into a system meets certain rules or constraints, which are defined based on the type of data and the business logic involved.

## Features

This validation script performs the following checks:

- **Missing Data Detection**: Check for missing or inconsistent data
- **Data Type Validation**: Validate data types, formats, and ranges
- **Duplicate Identification**: Identify duplicates or anomalies
- **Negative Value Detection**: Check for negative values in numeric columns

## Usage

Run the validation script:

```bash
cd directory_4_data_validation
python data_validation.py
```

## Output Files

The script generates two output files in the `validation_reports/` directory:

1. **data_quality_report.csv** - Excel-style CSV file with validation results
2. **validation_summary.txt** - Text summary of issues and resolutions

## CSV Report Format

The CSV report contains the following columns:

| Column | Description |
|--------|-------------|
| missing_values | Count of missing values per column |
| duplicate | Count of duplicate values |
| data_types | Data type of each column (object, int64, float64) |
| negative_values | Count of negative values (for numeric columns) |

## Requirements

- pandas
- numpy

## Implementation

Uses pandas for automated validation checks as specified in the assignment requirements.
