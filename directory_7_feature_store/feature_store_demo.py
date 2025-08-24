"""
Feature Store Demo Script
Students: 2024ab05134, 2024aa05664

This script demonstrates the Feature Store implementation with sample code
similar to the assignment example.
"""

import sqlite3
import pandas as pd
import os

def create_feature_store_demo():
    """
    Create a demo feature store following the assignment example.
    """
    print("Feature Store Demo")
    print("==================")
    print("A Feature Store is a centralized system or platform used to store, manage, and serve")
    print("features for machine learning models—both during training and in production.")
    print()
    
    # Create database connection
    demo_db_path = "feature_store_demo.db"
    connection = sqlite3.connect(demo_db_path)
    cursor = connection.cursor()
    
    print("Sample Code:")
    print("# Create a table to store engineered features")
    
    # Create the feature store table as shown in the assignment
    create_table_sql = """
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
    """
    
    print('cursor.execute("""')
    print(create_table_sql)
    print('""")')
    
    # Execute the table creation
    cursor.execute(create_table_sql)
    
    # Insert some sample data
    sample_data = [
        ('7590-VHVEG', 1, 29.85, 29.85, 0, 0, 1, 0, 0, 0),
        ('5575-GNVDE', 34, 56.95, 1889.50, 1, 0, 0, 0, 1, 0),
        ('3668-QPYBK', 2, 53.85, 108.15, 0, 0, 0, 0, 1, 1),
        ('7795-CFOCW', 45, 42.30, 1840.75, 1, 0, 0, 0, 0, 0),
        ('9237-HQITU', 2, 70.70, 151.65, 0, 0, 1, 0, 0, 1)
    ]
    
    insert_sql = """
    INSERT OR REPLACE INTO feature_store 
    (customerID, tenure, MonthlyCharges, TotalCharges, Contract_OneYear, 
     Contract_TwoYear, PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, 
     PaymentMethod_MailedCheck, Churn)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    cursor.executemany(insert_sql, sample_data)
    connection.commit()
    
    print(f"\nTable created successfully!")
    print(f"Sample data inserted: {len(sample_data)} records")
    
    # Query the data to verify
    cursor.execute("SELECT * FROM feature_store LIMIT 5")
    results = cursor.fetchall()
    
    print("\nSample Feature Store Data:")
    print("=" * 80)
    columns = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", 
               "Contract_OneYear", "Contract_TwoYear", "PaymentMethod_CreditCard",
               "PaymentMethod_ElectronicCheck", "PaymentMethod_MailedCheck", "Churn"]
    
    # Print header
    print(f"{'customerID':<12} {'tenure':<6} {'MonthlyCharges':<13} {'TotalCharges':<12} {'Churn':<5}")
    print("-" * 80)
    
    # Print data
    for row in results:
        print(f"{row[0]:<12} {row[1]:<6} {row[2]:<13.2f} {row[3]:<12.2f} {row[9]:<5}")
    
    # Show table schema
    cursor.execute("PRAGMA table_info(feature_store)")
    schema = cursor.fetchall()
    
    print(f"\nTable Schema:")
    print("=" * 50)
    for col in schema:
        print(f"{col[1]:<25} {col[2]:<10}")
    
    # Close connection
    connection.close()
    
    print(f"\nFeature Store demo completed!")
    print(f"Database file: {demo_db_path}")
    
    return demo_db_path

def show_production_implementation():
    """
    Show our actual production feature store implementation.
    """
    print("\nOur Production Feature Store Implementation:")
    print("=" * 50)
    
    # Check if our main feature store exists
    main_db_path = "../feature_store/feature_store.db"
    if os.path.exists(main_db_path):
        connection = sqlite3.connect(main_db_path)
        cursor = connection.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"Production Feature Store Tables ({len(tables)} total):")
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"- {table_name}: {count:,} records")
        
        # Show sample from telco features
        if any('telco' in t[0] for t in tables):
            cursor.execute("SELECT * FROM telco_customer_churn_features LIMIT 3")
            telco_sample = cursor.fetchall()
            
            print(f"\nSample from Telco Customer Churn Features:")
            print("customerID     tenure  MonthlyCharges  TotalCharges  Churn")
            print("-" * 55)
            for row in telco_sample:
                print(f"{row[0]:<12} {row[4]:<7} {row[22]:<13.2f} {row[23]:<12.2f} {row[24]:<5}")
        
        connection.close()
    else:
        print("Production feature store not found. Run feature_store_orchestrator.py first.")

if __name__ == "__main__":
    # Run the demo
    demo_db = create_feature_store_demo()
    
    # Show production implementation
    show_production_implementation()
    
    print(f"\nDemo complete! Check {demo_db} for the sample feature store.")
