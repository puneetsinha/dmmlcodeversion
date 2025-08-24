"""
Hugging Face data ingestion module.
"""

import os
from typing import Dict, List, Optional
import pandas as pd
from datasets import load_dataset

from config import config
from logger_utils import get_ingestion_logger


class HuggingFaceDataIngestion:
    """Handle data ingestion from Hugging Face datasets."""
    
    def __init__(self):
        """Initialize Hugging Face data ingestion."""
        self.logger = get_ingestion_logger()
    
    def download_dataset(
        self,
        dataset_name: str,
        output_name: str,
        config_name: str = "default",
        split: str = "train",
        max_rows: Optional[int] = None
    ) -> str:
        """
        Download dataset from Hugging Face.
        
        Args:
            dataset_name: Hugging Face dataset identifier
            output_name: Output file name
            config_name: Dataset configuration name
            split: Dataset split to download
            max_rows: Maximum number of rows to download
            
        Returns:
            Path to downloaded file
        """
        try:
            self.logger.info(f"Starting download of Hugging Face dataset: {dataset_name}")
            
            # Load dataset from Hugging Face
            dataset = load_dataset(
                dataset_name,
                name=config_name if config_name != "default" else None,
                split=split
            )
            
            # Convert to pandas DataFrame
            df = dataset.to_pandas()
            
            # Limit rows if specified
            if max_rows and len(df) > max_rows:
                df = df.head(max_rows)
                self.logger.info(f"Limited dataset to {max_rows} rows")
            
            # Save to CSV
            output_file = os.path.join(config.RAW_DATA_DIR, output_name)
            df.to_csv(output_file, index=False)
            
            # Validate downloaded file
            self._validate_downloaded_file(output_file, df)
            
            self.logger.info(f"Successfully downloaded Hugging Face dataset to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to download Hugging Face dataset {dataset_name}: {str(e)}")
            raise
    
    def _validate_downloaded_file(self, file_path: str, df: pd.DataFrame):
        """Validate the downloaded file."""
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError("Downloaded file is empty")
            
            # Validate DataFrame
            if df.empty:
                raise ValueError("Downloaded dataset is empty")
            
            self.logger.info(
                f"File validation successful: {file_size} bytes, "
                f"{len(df)} rows, {len(df.columns)} columns"
            )
            
        except Exception as e:
            self.logger.error(f"File validation failed: {str(e)}")
            raise
    
    def ingest_all_datasets(self) -> List[str]:
        """
        Ingest all configured Hugging Face datasets.
        
        Returns:
            List of paths to downloaded files
        """
        downloaded_files = []
        
        for dataset_config in config.HUGGINGFACE_DATASETS:
            try:
                file_path = self.download_dataset(
                    dataset_name=dataset_config["name"],
                    output_name=dataset_config["output_name"],
                    config_name=dataset_config.get("config", "default"),
                    split=dataset_config.get("split", "train")
                )
                downloaded_files.append(file_path)
                
            except Exception as e:
                self.logger.error(f"Failed to ingest dataset {dataset_config['name']}: {str(e)}")
                continue
        
        self.logger.info(f"Hugging Face ingestion completed. Downloaded {len(downloaded_files)} files")
        return downloaded_files


if __name__ == "__main__":
    # Test the Hugging Face ingestion
    ingestion = HuggingFaceDataIngestion()
    files = ingestion.ingest_all_datasets()
    print(f"Downloaded files: {files}")
