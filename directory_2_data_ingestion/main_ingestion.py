"""
Main data ingestion orchestrator.
Students: 2024ab05134, 2024aa05664

This script coordinates our data downloading from multiple sources with proper 
error handling and logging. We learned that having robust error handling is 
crucial when working with external APIs that might fail.

The main challenge we faced was handling the kaggle API authentication on 
university computers - that's why we implemented fallback URL downloading 
as a backup strategy. Pretty proud of this solution actually!
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from logger_utils import get_ingestion_logger
from kaggle_ingestion import KaggleDataIngestion
from huggingface_ingestion import HuggingFaceDataIngestion


class DataIngestionOrchestrator:
    """Orchestrate data ingestion from multiple sources."""
    
    def __init__(self):
        """Initialize the data ingestion orchestrator."""
        self.logger = get_ingestion_logger()
        self.start_time = None
        self.ingestion_summary = {
            "start_time": None,
            "end_time": None,
            "duration": None,
            "kaggle_files": [],
            "huggingface_files": [],
            "total_files": 0,
            "errors": []
        }
    
    def run_kaggle_ingestion(self) -> List[str]:
        """
        Run Kaggle data ingestion.
        
        Returns:
            List of downloaded file paths
        """
        self.logger.info("Starting Kaggle data ingestion...")
        
        try:
            kaggle_ingestion = KaggleDataIngestion()
            files = kaggle_ingestion.ingest_all_datasets()
            
            self.ingestion_summary["kaggle_files"] = files
            self.logger.info(f"Kaggle ingestion completed successfully. Files: {files}")
            return files
            
        except Exception as e:
            error_msg = f"Kaggle ingestion failed: {str(e)}"
            self.logger.error(error_msg)
            self.ingestion_summary["errors"].append(error_msg)
            return []
    
    def run_huggingface_ingestion(self) -> List[str]:
        """
        Run Hugging Face data ingestion.
        
        Returns:
            List of downloaded file paths
        """
        self.logger.info("Starting Hugging Face data ingestion...")
        
        try:
            hf_ingestion = HuggingFaceDataIngestion()
            files = hf_ingestion.ingest_all_datasets()
            
            self.ingestion_summary["huggingface_files"] = files
            self.logger.info(f"Hugging Face ingestion completed successfully. Files: {files}")
            return files
            
        except Exception as e:
            error_msg = f"Hugging Face ingestion failed: {str(e)}"
            self.logger.error(error_msg)
            self.ingestion_summary["errors"].append(error_msg)
            return []
    
    def generate_ingestion_report(self) -> Dict:
        """
        Generate comprehensive ingestion report.
        
        Returns:
            Dictionary containing ingestion summary
        """
        # Calculate duration
        if self.start_time:
            duration = time.time() - self.start_time
            self.ingestion_summary["duration"] = f"{duration:.2f} seconds"
        
        # Count total files
        total_files = len(self.ingestion_summary["kaggle_files"]) + \
                     len(self.ingestion_summary["huggingface_files"])
        self.ingestion_summary["total_files"] = total_files
        
        # Log summary
        self.logger.info("=" * 50)
        self.logger.info("DATA INGESTION SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Start Time: {self.ingestion_summary['start_time']}")
        self.logger.info(f"End Time: {self.ingestion_summary['end_time']}")
        self.logger.info(f"Duration: {self.ingestion_summary['duration']}")
        self.logger.info(f"Total Files Downloaded: {total_files}")
        self.logger.info(f"Kaggle Files: {len(self.ingestion_summary['kaggle_files'])}")
        self.logger.info(f"Hugging Face Files: {len(self.ingestion_summary['huggingface_files'])}")
        
        if self.ingestion_summary["errors"]:
            self.logger.warning(f"Errors Encountered: {len(self.ingestion_summary['errors'])}")
            for error in self.ingestion_summary["errors"]:
                self.logger.warning(f"  - {error}")
        else:
            self.logger.info("No errors encountered during ingestion")
        
        self.logger.info("=" * 50)
        
        return self.ingestion_summary
    
    def run_full_ingestion(self) -> Dict:
        """
        Run complete data ingestion pipeline.
        
        Returns:
            Ingestion summary dictionary
        """
        self.start_time = time.time()
        self.ingestion_summary["start_time"] = datetime.now().isoformat()
        
        self.logger.info("Starting complete data ingestion pipeline...")
        
        # Run Kaggle ingestion
        kaggle_files = self.run_kaggle_ingestion()
        
        # Small delay between ingestions
        time.sleep(1)
        
        # Run Hugging Face ingestion
        hf_files = self.run_huggingface_ingestion()
        
        # Finalize summary
        self.ingestion_summary["end_time"] = datetime.now().isoformat()
        
        # Generate and return report
        return self.generate_ingestion_report()
    
    def verify_raw_data(self) -> bool:
        """
        Verify that raw data files exist and are accessible.
        
        Returns:
            True if all files are accessible, False otherwise
        """
        self.logger.info("Verifying raw data files...")
        
        all_files = (self.ingestion_summary["kaggle_files"] + 
                    self.ingestion_summary["huggingface_files"])
        
        if not all_files:
            self.logger.warning("No files to verify")
            return False
        
        for file_path in all_files:
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                self.logger.error(f"File is empty: {file_path}")
                return False
            
            self.logger.info(f" {file_path} ({file_size} bytes)")
        
        self.logger.info("All raw data files verified successfully")
        return True


def main():
    """
    Main function to run data ingestion.
    
    This function runs our entire data ingestion process. We start with Kaggle
    data first because it's usually more reliable, then move to HuggingFace.
    If anything fails, we have proper logging to help us debug the issues.
    
    We spent a lot of time getting the error handling right - learned that
    external APIs can be unreliable sometimes!
    """
    try:
        # Initialize orchestrator
        orchestrator = DataIngestionOrchestrator()
        
        # Run full ingestion
        summary = orchestrator.run_full_ingestion()
        
        # Verify downloaded data
        if orchestrator.verify_raw_data():
            print(" Data ingestion completed successfully!")
        else:
            print("  Data ingestion completed with issues. Check logs for details.")
        
        return summary
        
    except Exception as e:
        print(f" Data ingestion failed: {str(e)}")
        raise


if __name__ == "__main__":
    summary = main()
