"""
Feature Store Implementation.
Students: 2024ab05134, 2024aa05664

This module implements a centralized feature store using SQLite as the backend.
We learned about feature stores in class and decided to implement our own
simplified version to understand how they work.

Key features we implemented:
- Feature registration with comprehensive metadata
- Feature versioning for reproducibility  
- Feature lineage tracking
- Quality monitoring and validation
- Feature views for different use cases

The concept of feature stores is relatively new in the industry but becoming
very important for ML ops. Building our own helped us understand why large
companies like Uber and Netflix use them for managing thousands of features.

We used SQLite for simplicity but in production you'd probably use something
like PostgreSQL or a specialized feature store like Feast.
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

from feature_store_config import feature_store_config


class FeatureStore:
    """Centralized feature store with metadata management."""
    
    def __init__(self):
        """Initialize the feature store."""
        self.config = feature_store_config
        self.db_path = self.config.DATABASE_CONFIG["database_path"]
        self.connection = None
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize the feature store database."""
        try:
            self.connection = sqlite3.connect(
                self.db_path, 
                timeout=self.config.DATABASE_CONFIG["timeout"]
            )
            self.connection.row_factory = sqlite3.Row
            
            # Create core tables
            self._create_core_tables()
            print(f" Feature store database initialized: {self.db_path}")
            
        except Exception as e:
            print(f" Failed to initialize feature store database: {str(e)}")
            raise
    
    def _create_core_tables(self):
        """Create core feature store tables."""
        cursor = self.connection.cursor()
        
        # Feature registry table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_registry (
                feature_id TEXT PRIMARY KEY,
                feature_name TEXT NOT NULL,
                feature_group TEXT NOT NULL,
                data_type TEXT NOT NULL,
                description TEXT,
                source_dataset TEXT,
                creation_timestamp TEXT NOT NULL,
                created_by TEXT DEFAULT 'system',
                tags TEXT,  -- JSON array
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Feature versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_versions (
                version_id TEXT PRIMARY KEY,
                feature_id TEXT NOT NULL,
                version_number TEXT NOT NULL,
                data_location TEXT NOT NULL,
                schema_definition TEXT,  -- JSON
                statistics TEXT,  -- JSON
                quality_metrics TEXT,  -- JSON
                created_timestamp TEXT NOT NULL,
                is_current_version BOOLEAN DEFAULT 0,
                FOREIGN KEY (feature_id) REFERENCES feature_registry (feature_id)
            )
        """)
        
        # Feature lineage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_lineage (
                lineage_id TEXT PRIMARY KEY,
                feature_id TEXT NOT NULL,
                source_features TEXT,  -- JSON array
                transformation_logic TEXT,
                dependencies TEXT,  -- JSON array
                created_timestamp TEXT NOT NULL,
                FOREIGN KEY (feature_id) REFERENCES feature_registry (feature_id)
            )
        """)
        
        # Feature serving table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_serving (
                serving_id TEXT PRIMARY KEY,
                feature_id TEXT NOT NULL,
                serving_type TEXT NOT NULL,
                endpoint_config TEXT,  -- JSON
                performance_metrics TEXT,  -- JSON
                last_served_timestamp TEXT,
                is_enabled BOOLEAN DEFAULT 1,
                FOREIGN KEY (feature_id) REFERENCES feature_registry (feature_id)
            )
        """)
        
        # Feature quality monitoring table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_quality (
                quality_id TEXT PRIMARY KEY,
                feature_id TEXT NOT NULL,
                check_timestamp TEXT NOT NULL,
                quality_score REAL,
                quality_issues TEXT,  -- JSON
                data_drift_score REAL,
                statistical_summary TEXT,  -- JSON
                FOREIGN KEY (feature_id) REFERENCES feature_registry (feature_id)
            )
        """)
        
        self.connection.commit()
        print(" Core feature store tables created")
    
    def register_feature_group(self, dataset_name: str, feature_data: pd.DataFrame) -> str:
        """
        Register a complete feature group from transformed dataset.
        
        Args:
            dataset_name: Name of the source dataset
            feature_data: DataFrame containing features
            
        Returns:
            Feature group ID
        """
        print(f"\n Registering feature group: {dataset_name}")
        print("=" * 50)
        
        group_id = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        features_registered = 0
        
        cursor = self.connection.cursor()
        
        for column in feature_data.columns:
            try:
                # Generate feature ID
                feature_id = self._generate_feature_id(dataset_name, column)
                
                # Analyze feature characteristics
                feature_info = self._analyze_feature(feature_data[column], column)
                
                # Register feature
                cursor.execute("""
                    INSERT OR REPLACE INTO feature_registry (
                        feature_id, feature_name, feature_group, data_type, 
                        description, source_dataset, creation_timestamp, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feature_id,
                    column,
                    group_id,
                    feature_info["data_type"],
                    feature_info["description"],
                    dataset_name,
                    datetime.now().isoformat(),
                    json.dumps(feature_info["tags"])
                ))
                
                # Store feature version
                self._store_feature_version(feature_id, feature_data[column], feature_info)
                
                features_registered += 1
                
            except Exception as e:
                print(f" Failed to register feature {column}: {str(e)}")
                continue
        
        self.connection.commit()
        print(f" Registered {features_registered} features in group {group_id}")
        
        return group_id
    
    def _generate_feature_id(self, dataset_name: str, feature_name: str) -> str:
        """Generate unique feature ID."""
        id_string = f"{dataset_name}_{feature_name}_{datetime.now().isoformat()}"
        return hashlib.md5(id_string.encode()).hexdigest()[:16]
    
    def _analyze_feature(self, feature_series: pd.Series, feature_name: str) -> Dict:
        """Analyze feature characteristics and generate metadata."""
        
        # Basic statistics
        stats = {
            "count": int(feature_series.count()),
            "missing_count": int(feature_series.isnull().sum()),
            "unique_count": int(feature_series.nunique()),
            "data_type": str(feature_series.dtype)
        }
        
        # Determine feature type and characteristics
        if pd.api.types.is_numeric_dtype(feature_series) and feature_series.dtype != 'bool':
            feature_type = "numerical"
            stats.update({
                "mean": float(feature_series.mean()) if not feature_series.isnull().all() else None,
                "std": float(feature_series.std()) if not feature_series.isnull().all() else None,
                "min": float(feature_series.min()) if not feature_series.isnull().all() else None,
                "max": float(feature_series.max()) if not feature_series.isnull().all() else None,
                "median": float(feature_series.median()) if not feature_series.isnull().all() else None
            })
            
        elif feature_series.dtype == 'bool' or feature_series.nunique() <= 2:
            feature_type = "binary"
            value_counts = feature_series.value_counts()
            stats["value_distribution"] = value_counts.to_dict()
            
        else:
            feature_type = "categorical"
            value_counts = feature_series.value_counts().head(10)
            stats["top_values"] = value_counts.to_dict()
            stats["cardinality"] = int(feature_series.nunique())
        
        # Generate tags based on feature characteristics
        tags = [feature_type]
        
        # Add domain-specific tags
        if "churn" in feature_name.lower():
            tags.append("target")
        elif "id" in feature_name.lower():
            tags.append("identifier")
        elif any(word in feature_name.lower() for word in ["charge", "price", "cost", "income"]):
            tags.append("monetary")
        elif any(word in feature_name.lower() for word in ["age", "tenure", "time"]):
            tags.append("temporal")
        elif "ratio" in feature_name.lower() or "per" in feature_name.lower():
            tags.append("derived_ratio")
        elif "flag" in feature_name.lower() or "is_" in feature_name.lower():
            tags.append("flag")
        elif "bin" in feature_name.lower() or "group" in feature_name.lower():
            tags.append("binned")
        elif "poly_" in feature_name.lower():
            tags.append("polynomial")
        elif "zscore" in feature_name.lower() or "percentile" in feature_name.lower():
            tags.append("statistical")
        
        # Quality assessment
        completeness = 1 - (stats["missing_count"] / stats["count"]) if stats["count"] > 0 else 0
        if completeness < 0.9:
            tags.append("incomplete")
        elif completeness == 1.0:
            tags.append("complete")
        
        return {
            "data_type": feature_type,
            "description": f"{feature_type.title()} feature from {feature_name}",
            "statistics": stats,
            "tags": tags,
            "quality_score": completeness
        }
    
    def _store_feature_version(self, feature_id: str, feature_data: pd.Series, feature_info: Dict):
        """Store feature version with data and metadata."""
        
        # Generate version
        version_number = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version_id = f"{feature_id}_{version_number}"
        
        # Store feature data as parquet
        data_location = os.path.join(
            self.config.FEATURE_REGISTRY_DIR, 
            "tables",
            f"{feature_id}_{version_number}.parquet"
        )
        
        # Convert series to DataFrame for storage
        feature_df = feature_data.to_frame()
        feature_df.to_parquet(data_location, index=True)
        
        # Store schema definition
        schema_definition = {
            "column_name": feature_data.name,
            "data_type": str(feature_data.dtype),
            "nullable": bool(feature_data.isnull().any()),
            "unique_values": int(feature_data.nunique())
        }
        
        # Store version metadata
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO feature_versions (
                version_id, feature_id, version_number, data_location,
                schema_definition, statistics, quality_metrics, 
                created_timestamp, is_current_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            version_id,
            feature_id,
            version_number,
            data_location,
            json.dumps(schema_definition),
            json.dumps(feature_info["statistics"]),
            json.dumps({"quality_score": feature_info["quality_score"]}),
            datetime.now().isoformat(),
            1  # Mark as current version
        ))
    
    def get_feature_metadata(self, feature_name: str = None, feature_group: str = None) -> List[Dict]:
        """
        Retrieve feature metadata.
        
        Args:
            feature_name: Specific feature name (optional)
            feature_group: Feature group to filter by (optional)
            
        Returns:
            List of feature metadata dictionaries
        """
        cursor = self.connection.cursor()
        
        query = "SELECT * FROM feature_registry WHERE is_active = 1"
        params = []
        
        if feature_name:
            query += " AND feature_name = ?"
            params.append(feature_name)
        
        if feature_group:
            query += " AND feature_group = ?"
            params.append(feature_group)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        return [dict(row) for row in results]
    
    def get_feature_data(self, feature_id: str, version: str = "latest") -> Optional[pd.Series]:
        """
        Retrieve feature data by ID and version.
        
        Args:
            feature_id: Feature identifier
            version: Version to retrieve ("latest" or specific version)
            
        Returns:
            Feature data as pandas Series
        """
        cursor = self.connection.cursor()
        
        if version == "latest":
            cursor.execute("""
                SELECT data_location FROM feature_versions 
                WHERE feature_id = ? AND is_current_version = 1
            """, (feature_id,))
        else:
            cursor.execute("""
                SELECT data_location FROM feature_versions 
                WHERE feature_id = ? AND version_number = ?
            """, (feature_id, version))
        
        result = cursor.fetchone()
        
        if result:
            data_location = result["data_location"]
            if os.path.exists(data_location):
                df = pd.read_parquet(data_location)
                return df.iloc[:, 0]  # Return first (and only) column as Series
        
        return None
    
    def create_feature_view(self, view_name: str, feature_list: List[str], 
                           dataset_name: str = None) -> Dict:
        """
        Create a feature view combining multiple features.
        
        Args:
            view_name: Name for the feature view
            feature_list: List of feature names to include
            dataset_name: Source dataset name for filtering
            
        Returns:
            Feature view metadata
        """
        print(f"\n Creating feature view: {view_name}")
        
        # Get metadata for requested features
        all_features = self.get_feature_metadata()
        
        if dataset_name:
            all_features = [f for f in all_features if f["source_dataset"] == dataset_name]
        
        # Filter to requested features
        selected_features = []
        feature_data_dict = {}
        
        for feature in all_features:
            if feature["feature_name"] in feature_list:
                selected_features.append(feature)
                
                # Get feature data
                feature_data = self.get_feature_data(feature["feature_id"])
                if feature_data is not None:
                    feature_data_dict[feature["feature_name"]] = feature_data
        
        if not feature_data_dict:
            print(f" No data found for requested features")
            return {}
        
        # Combine features into DataFrame
        feature_view_df = pd.DataFrame(feature_data_dict)
        
        # Save feature view
        view_location = os.path.join(
            self.config.FEATURE_REGISTRY_DIR,
            "views",
            f"{view_name}.parquet"
        )
        feature_view_df.to_parquet(view_location, index=True)
        
        # Create view metadata
        view_metadata = {
            "view_name": view_name,
            "features_included": feature_list,
            "features_found": list(feature_data_dict.keys()),
            "total_features": len(feature_data_dict),
            "total_records": len(feature_view_df),
            "data_location": view_location,
            "created_timestamp": datetime.now().isoformat(),
            "feature_metadata": selected_features
        }
        
        # Save view metadata
        metadata_location = os.path.join(
            self.config.FEATURE_METADATA_DIR,
            f"{view_name}_metadata.json"
        )
        with open(metadata_location, 'w') as f:
            json.dump(view_metadata, f, indent=2, default=str)
        
        print(f" Feature view created: {len(feature_data_dict)} features, {len(feature_view_df)} records")
        print(f" View location: {view_location}")
        
        return view_metadata
    
    def generate_feature_catalog(self) -> Dict:
        """Generate comprehensive feature catalog."""
        
        print("\n Generating Feature Catalog")
        print("=" * 40)
        
        cursor = self.connection.cursor()
        
        # Get all features with their metadata
        cursor.execute("""
            SELECT fr.*, fv.version_number, fv.statistics, fv.quality_metrics
            FROM feature_registry fr
            LEFT JOIN feature_versions fv ON fr.feature_id = fv.feature_id 
            WHERE fr.is_active = 1 AND fv.is_current_version = 1
        """)
        
        features = cursor.fetchall()
        
        # Organize by feature groups
        catalog = {
            "generated_timestamp": datetime.now().isoformat(),
            "total_features": len(features),
            "feature_groups": {},
            "feature_types": {},
            "data_sources": {},
            "feature_details": []
        }
        
        for feature in features:
            feature_dict = dict(feature)
            
            # Parse JSON fields
            if feature_dict["tags"]:
                feature_dict["tags"] = json.loads(feature_dict["tags"])
            if feature_dict["statistics"]:
                feature_dict["statistics"] = json.loads(feature_dict["statistics"])
            if feature_dict["quality_metrics"]:
                feature_dict["quality_metrics"] = json.loads(feature_dict["quality_metrics"])
            
            catalog["feature_details"].append(feature_dict)
            
            # Group by feature group
            group = feature_dict["feature_group"]
            if group not in catalog["feature_groups"]:
                catalog["feature_groups"][group] = 0
            catalog["feature_groups"][group] += 1
            
            # Group by data type
            data_type = feature_dict["data_type"]
            if data_type not in catalog["feature_types"]:
                catalog["feature_types"][data_type] = 0
            catalog["feature_types"][data_type] += 1
            
            # Group by data source
            source = feature_dict["source_dataset"]
            if source not in catalog["data_sources"]:
                catalog["data_sources"][source] = 0
            catalog["data_sources"][source] += 1
        
        # Save catalog
        catalog_location = os.path.join(
            self.config.FEATURE_STORE_DIR,
            "feature_catalog.json"
        )
        with open(catalog_location, 'w') as f:
            json.dump(catalog, f, indent=2, default=str)
        
        print(f" Feature catalog generated:")
        print(f"    Total features: {catalog['total_features']}")
        print(f"    Feature groups: {len(catalog['feature_groups'])}")
        print(f"    Data types: {list(catalog['feature_types'].keys())}")
        print(f"    Catalog saved: {catalog_location}")
        
        return catalog
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            print(" Feature store connection closed")


if __name__ == "__main__":
    store = FeatureStore()
    print("Feature Store initialized successfully")
