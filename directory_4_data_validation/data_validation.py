"""
Data Validation Script
Students: 2024ab05134, 2024aa05664

Simple automated validation script using pandas for:
- Check for missing or inconsistent data
- Validate data types, formats, and ranges  
- Identify duplicates or anomalies
- Generate data quality report in CSV format
"""

import pandas as pd
import os
from pathlib import Path

def main():
    """Main function to run data validation."""
    print("Starting Data Validation...")
    print("=" * 40)
    
    # Import and run the validation
    from simple_data_validation import generate_validation_report
    generate_validation_report()
    
    print("\nValidation completed!")
    print("Reports generated:")
    print("- data_quality_report.csv")
    print("- validation_summary.txt")

if __name__ == "__main__":
    main()
