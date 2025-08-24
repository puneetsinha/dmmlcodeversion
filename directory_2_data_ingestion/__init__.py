"""
Data Ingestion Module for ML Pipeline.

This module provides functionality to ingest data from multiple sources
including Kaggle and Hugging Face datasets.
"""

from .main_ingestion import DataIngestionOrchestrator
from .kaggle_ingestion import KaggleDataIngestion
from .huggingface_ingestion import HuggingFaceDataIngestion
from .config import config

__all__ = [
    'DataIngestionOrchestrator',
    'KaggleDataIngestion', 
    'HuggingFaceDataIngestion',
    'config'
]
