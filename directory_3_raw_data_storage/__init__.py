"""
Raw Data Storage Module.

This module handles the organization of raw data into a partitioned data lake structure
following best practices for data management and storage.
"""

from .data_lake_organizer import DataLakeOrganizer
from .storage_config import storage_config

__all__ = ['DataLakeOrganizer', 'storage_config']
