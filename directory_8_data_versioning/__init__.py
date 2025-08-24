"""
Data Versioning Module.

This module provides comprehensive data versioning capabilities using DVC and Git:
- Data version control with DVC
- Git integration for code and metadata tracking
- Version metadata and lineage tracking
- Automated versioning workflows
- Documentation and reporting
"""

from .data_version_manager import DataVersionManager
from .versioning_orchestrator import VersioningOrchestrator
from .versioning_config import versioning_config

__all__ = ['DataVersionManager', 'VersioningOrchestrator', 'versioning_config']
