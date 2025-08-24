"""
Kaggle data ingestion module.
"""

import os
import shutil
import zipfile
from typing import Dict, List, Optional
import pandas as pd
import requests

from config import config
from logger_utils import get_ingestion_logger

# Don't import Kaggle API at module level to avoid authentication issues
KAGGLE_AVAILABLE = True


class KaggleDataIngestion:
    """Handle data ingestion from Kaggle datasets."""
    
    def __init__(self):
        """Initialize Kaggle API client."""
        self.logger = get_ingestion_logger()
        self.api = None
        self._setup_kaggle_api()
    
    def _setup_kaggle_api(self):
        """Set up Kaggle API authentication."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            self.api = KaggleApi()
            self.api.authenticate()
            self.logger.info("Kaggle API authentication successful")
        except ImportError:
            self.logger.warning("Kaggle API not available, will use fallback URLs")
            self.api = None
        except Exception as e:
            self.logger.warning(f"Kaggle API authentication failed: {str(e)}")
            self.logger.info("Will attempt to use fallback URLs for data download")
            self.api = None
    
    def download_from_url(
        self,
        url: str,
        output_name: str
    ) -> str:
        """
        Download dataset from URL fallback.
        
        Args:
            url: Direct URL to the dataset
            output_name: Output file name
            
        Returns:
            Path to downloaded file
        """
        try:
            self.logger.info(f"Downloading dataset from URL: {url}")
            
            # Download file
            response = requests.get(url)
            response.raise_for_status()
            
            # Save to file
            output_file = os.path.join(config.RAW_DATA_DIR, output_name)
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            # Validate downloaded file
            self._validate_downloaded_file(output_file)
            
            self.logger.info(f"Successfully downloaded dataset from URL to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to download from URL {url}: {str(e)}")
            raise

    def download_dataset(
        self,
        dataset_name: str,
        file_name: Optional[str] = None,
        output_name: Optional[str] = None,
        fallback_url: Optional[str] = None
    ) -> str:
        """
        Download dataset from Kaggle or fallback URL.
        
        Args:
            dataset_name: Kaggle dataset identifier (e.g., 'owner/dataset-name')
            file_name: Specific file to extract (optional)
            output_name: Output file name (optional)
            fallback_url: Fallback URL if Kaggle API is not available
            
        Returns:
            Path to downloaded file
        """
        # Check if we should use fallback URL
        if self.api is None and fallback_url:
            self.logger.info(f"Using fallback URL for dataset: {dataset_name}")
            if not output_name:
                output_name = f"{dataset_name.replace('/', '_')}.csv"
            return self.download_from_url(fallback_url, output_name)
        
        if self.api is None:
            raise Exception("Kaggle API not available and no fallback URL provided")
        
        try:
            self.logger.info(f"Starting download of Kaggle dataset: {dataset_name}")
            
            # Create temporary download directory
            temp_dir = os.path.join(config.RAW_DATA_DIR, "temp_kaggle")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Download dataset
            self.api.dataset_download_files(
                dataset_name,
                path=temp_dir,
                unzip=True
            )
            
            # Handle file extraction and naming
            if file_name:
                source_file = os.path.join(temp_dir, file_name)
                if not os.path.exists(source_file):
                    raise FileNotFoundError(f"File {file_name} not found in dataset")
            else:
                # Find the first CSV file
                csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
                if not csv_files:
                    raise FileNotFoundError("No CSV files found in dataset")
                file_name = csv_files[0]
                source_file = os.path.join(temp_dir, file_name)
            
            # Determine output file name
            if output_name:
                output_file = os.path.join(config.RAW_DATA_DIR, output_name)
            else:
                output_file = os.path.join(config.RAW_DATA_DIR, file_name)
            
            # Move file to final location
            shutil.copy2(source_file, output_file)
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
            # Validate downloaded file
            self._validate_downloaded_file(output_file)
            
            self.logger.info(f"Successfully downloaded Kaggle dataset to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to download Kaggle dataset {dataset_name}: {str(e)}")
            # Clean up on failure
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise
    
    def _validate_downloaded_file(self, file_path: str):
        """Validate the downloaded file."""
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError("Downloaded file is empty")
            
            # Try to read the file to ensure it's valid
            df = pd.read_csv(file_path, nrows=5)
            self.logger.info(f"File validation successful: {file_size} bytes, {len(df.columns)} columns")
            
        except Exception as e:
            self.logger.error(f"File validation failed: {str(e)}")
            raise
    
    def ingest_all_datasets(self) -> List[str]:
        """
        Ingest all configured Kaggle datasets.
        
        Returns:
            List of paths to downloaded files
        """
        downloaded_files = []
        
        for dataset_config in config.KAGGLE_DATASETS:
            try:
                file_path = self.download_dataset(
                    dataset_name=dataset_config["name"],
                    file_name=dataset_config.get("file_name"),
                    output_name=dataset_config.get("output_name"),
                    fallback_url=dataset_config.get("fallback_url")
                )
                downloaded_files.append(file_path)
                
            except Exception as e:
                self.logger.error(f"Failed to ingest dataset {dataset_config['name']}: {str(e)}")
                continue
        
        self.logger.info(f"Kaggle ingestion completed. Downloaded {len(downloaded_files)} files")
        return downloaded_files


if __name__ == "__main__":
    # Test the Kaggle ingestion
    ingestion = KaggleDataIngestion()
    files = ingestion.ingest_all_datasets()
    print(f"Downloaded files: {files}")
