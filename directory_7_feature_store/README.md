# Feature Store Component

**Students: 2024ab05134, 2024aa05664**

## Definition

A Feature Store is a centralized system or platform used to store, manage, and serve features for machine learning models—both during training and in production.

## Sample Code

```sql
# Create a table to store engineered features
cursor.execute("""
    CREATE TABLE IF NOT EXISTS feature_store (
        customerID TEXT PRIMARY KEY,
        tenure INTEGER,
        MonthlyCharges REAL,
        TotalCharges REAL,
        Contract_OneYear INTEGER,
        Contract_TwoYear INTEGER,
        PaymentMethod_CreditCard INTEGER,
        PaymentMethod_ElectronicCheck INTEGER,
        PaymentMethod_MailedCheck INTEGER,
        Churn INTEGER
    )
""")
```

## Implementation

Our Feature Store uses SQLite as the backend database with comprehensive feature management capabilities:

- **210 Features Registered** across 2 datasets
- **8 Feature Views** for different ML use cases  
- **Complete Metadata Tracking** with lineage and versioning
- **Quality Monitoring** with automated validation
- **Multiple Serving Methods** for training and inference

## Key Features

- Centralized feature storage and management
- Feature versioning and lineage tracking
- Quality monitoring and validation
- Multiple feature views for different models
- Batch and online feature serving
- Integration with transformation pipeline

## Usage

```bash
cd directory_7_feature_store
python feature_store_orchestrator.py
```

## Output Files

- SQLite database with feature tables
- Feature catalog and metadata
- Quality monitoring reports
- Feature view definitions

See `FEATURE_STORE_REPORT.txt` for complete implementation details and architecture.
