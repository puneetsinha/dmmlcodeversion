"""
Data Lake Organizer for partitioned storage.
Organizes raw data into a structured data lake with partitioning.
"""

import os
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json

from storage_config import storage_config


class DataLakeOrganizer:
    """Organize raw data into partitioned data lake structure."""
    
    def __init__(self):
        """Initialize the data lake organizer."""
        self.config = storage_config
        self.storage_manifest = {
            "created": datetime.now().isoformat(),
            "partitions": [],
            "total_files": 0,
            "total_size_bytes": 0
        }
    
    def create_partition_path(
        self,
        source: str,
        data_type: str,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Create partition path based on source, type, and timestamp.
        
        Args:
            source: Data source (e.g., 'kaggle', 'huggingface')
            data_type: Type of data (e.g., 'customer_churn', 'demographics')
            timestamp: Timestamp for partitioning (defaults to now)
            
        Returns:
            Partition path string
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        partition_path = os.path.join(
            self.config.PARTITIONED_STORAGE,
            f"source={source}",
            f"data_type={data_type}",
            f"year={timestamp.year}",
            f"month={timestamp.month:02d}",
            f"day={timestamp.day:02d}"
        )
        
        return partition_path
    
    def store_file(
        self,
        source_file: str,
        source: str,
        data_type: str,
        original_name: str,
        convert_to_parquet: bool = True
    ) -> Dict:
        """
        Store file in partitioned data lake.
        
        Args:
            source_file: Path to source file
            source: Data source identifier
            data_type: Data type identifier
            original_name: Original dataset name
            convert_to_parquet: Whether to convert to Parquet format
            
        Returns:
            Dictionary with storage information
        """
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Source file not found: {source_file}")
        
        # Create partition path
        partition_path = self.create_partition_path(source, data_type)
        os.makedirs(partition_path, exist_ok=True)
        
        # Read and process data
        df = pd.read_csv(source_file)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if convert_to_parquet:
            filename = f"{original_name}_{timestamp}.parquet"
            output_file = os.path.join(partition_path, filename)
            df.to_parquet(output_file, compression=self.config.COMPRESSION, index=False)
        else:
            filename = f"{original_name}_{timestamp}.csv"
            output_file = os.path.join(partition_path, filename)
            df.to_csv(output_file, index=False)
        
        # Calculate file size
        file_size = os.path.getsize(output_file)
        
        # Create metadata
        metadata = {
            "source_file": source_file,
            "stored_file": output_file,
            "partition_path": partition_path,
            "source": source,
            "data_type": data_type,
            "original_name": original_name,
            "filename": filename,
            "format": "parquet" if convert_to_parquet else "csv",
            "rows": len(df),
            "columns": len(df.columns),
            "size_bytes": file_size,
            "created": datetime.now().isoformat(),
            "column_names": list(df.columns)
        }
        
        # Save metadata file
        metadata_file = os.path.join(partition_path, f"{original_name}_{timestamp}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update manifest
        self.storage_manifest["partitions"].append(metadata)
        self.storage_manifest["total_files"] += 1
        self.storage_manifest["total_size_bytes"] += file_size
        
        print(f" Stored: {filename} ({file_size:,} bytes, {len(df):,} rows)")
        return metadata
    
    def organize_raw_data(self) -> List[Dict]:
        """
        Organize all raw data files into partitioned storage.
        
        Returns:
            List of storage metadata for each file
        """
        stored_files = []
        
        print("  Organizing raw data into partitioned data lake...")
        print("=" * 60)
        
        # Process each file in raw data directory
        raw_data_files = os.listdir(self.config.RAW_DATA_DIR)
        
        for filename in raw_data_files:
            if not filename.endswith('.csv'):
                continue
                
            source_file = os.path.join(self.config.RAW_DATA_DIR, filename)
            
            # Get source mapping
            if filename in self.config.SOURCE_MAPPINGS:
                mapping = self.config.SOURCE_MAPPINGS[filename]
                
                try:
                    metadata = self.store_file(
                        source_file=source_file,
                        source=mapping["source"],
                        data_type=mapping["data_type"],
                        original_name=mapping["original_name"],
                        convert_to_parquet=True
                    )
                    stored_files.append(metadata)
                    
                except Exception as e:
                    print(f" Failed to store {filename}: {str(e)}")
                    continue
            else:
                print(f"  No mapping found for {filename}, skipping...")
        
        return stored_files
    
    def create_data_catalog(self) -> str:
        """
        Create a data catalog with partition information.
        
        Returns:
            Path to catalog file
        """
        catalog = {
            "data_lake_info": {
                "created": self.storage_manifest["created"],
                "base_path": self.config.PARTITIONED_STORAGE,
                "partition_scheme": "source/data_type/year/month/day",
                "storage_format": self.config.STORAGE_FORMAT,
                "compression": self.config.COMPRESSION
            },
            "summary": {
                "total_partitions": len(self.storage_manifest["partitions"]),
                "total_files": self.storage_manifest["total_files"],
                "total_size_mb": round(self.storage_manifest["total_size_bytes"] / (1024*1024), 2)
            },
            "partitions": self.storage_manifest["partitions"]
        }
        
        catalog_file = os.path.join(self.config.PARTITIONED_STORAGE, "data_catalog.json")
        with open(catalog_file, 'w') as f:
            json.dump(catalog, f, indent=2)
        
        print(f" Data catalog created: {catalog_file}")
        return catalog_file
    
    def generate_storage_report(self) -> str:
        """
        Generate a human-readable storage report.
        
        Returns:
            Storage report as string
        """
        report = []
        report.append(" DATA LAKE STORAGE REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Base Path: {self.config.PARTITIONED_STORAGE}")
        report.append("")
        
        report.append(" SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Partitions: {len(self.storage_manifest['partitions'])}")
        report.append(f"Total Files: {self.storage_manifest['total_files']}")
        total_mb = self.storage_manifest['total_size_bytes'] / (1024*1024)
        report.append(f"Total Size: {total_mb:.2f} MB")
        report.append("")
        
        report.append("  PARTITION DETAILS")
        report.append("-" * 30)
        
        for partition in self.storage_manifest["partitions"]:
            report.append(f"Source: {partition['source']}")
            report.append(f"  Type: {partition['data_type']}")
            report.append(f"  File: {partition['filename']}")
            report.append(f"  Path: {partition['partition_path']}")
            report.append(f"  Rows: {partition['rows']:,}")
            report.append(f"  Size: {partition['size_bytes']:,} bytes")
            report.append("")
        
        report_text = "\n".join(report)
        
        # Save report to file
        report_file = os.path.join(self.config.PARTITIONED_STORAGE, "storage_report.txt")
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def list_partitions(self) -> List[str]:
        """
        List all partitions in the data lake.
        
        Returns:
            List of partition paths
        """
        partitions = []
        
        if not os.path.exists(self.config.PARTITIONED_STORAGE):
            return partitions
        
        for root, dirs, files in os.walk(self.config.PARTITIONED_STORAGE):
            if files and any(f.endswith(('.parquet', '.csv')) for f in files):
                relative_path = os.path.relpath(root, self.config.PARTITIONED_STORAGE)
                partitions.append(relative_path)
        
        return partitions


def main():
    """Main function to organize raw data."""
    organizer = DataLakeOrganizer()
    
    print(" Starting Data Lake Organization")
    print("=" * 50)
    
    # Organize raw data
    stored_files = organizer.organize_raw_data()
    
    if stored_files:
        # Create catalog
        catalog_file = organizer.create_data_catalog()
        
        # Generate report
        report = organizer.generate_storage_report()
        print("\n" + report)
        
        # List partitions
        partitions = organizer.list_partitions()
        print("\n  Created Partitions:")
        for partition in partitions:
            print(f"   {partition}")
        
        print(f"\n Data lake organization completed!")
        print(f" Catalog: {catalog_file}")
        
    else:
        print(" No files were organized. Check raw data directory.")


if __name__ == "__main__":
    main()
