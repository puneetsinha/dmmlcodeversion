"""
Data Version Manager for Git-based versioning.
Students: 2024ab05134, 2024aa05664

This module provides simple Git-based data versioning functionality
to track and upload data files to Git repository.
"""

import os
import json
import subprocess
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import sys
from pathlib import Path

class DataVersionManager:
    """Simple Git-based data version manager."""
    
    def __init__(self, project_path="/Users/puneetsinha/DMML"):
        """Initialize the data version manager."""
        self.project_path = Path(project_path)
        self.git_dir = self.project_path / ".git"
        self.data_dirs = [
            "raw_data",
            "transformed_data", 
            "directory_3_raw_data_storage/data_lake",
            "feature_store",
            "models/trained_models"
        ]
        
    def check_git_setup(self) -> bool:
        """Check if Git is properly set up."""
        try:
            os.chdir(self.project_path)
            result = subprocess.run(['git', 'status'], 
                                  capture_output=True, text=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(" Git repository not initialized or Git not installed")
            return False
    
    def get_data_files(self) -> List[str]:
        """Get list of all data files to version."""
        data_files = []
        
        for data_dir in self.data_dirs:
            dir_path = self.project_path / data_dir
            if dir_path.exists():
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(self.project_path)
                        data_files.append(str(rel_path))
        
        return data_files
    
    def create_data_version(self, version_name: str, description: str = "") -> bool:
        """
        Create a new data version and upload to Git.
        
        Args:
            version_name: Name for the version (e.g., 'v1.0-raw-data')
            description: Description of changes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"\nCreating Data Version: {version_name}")
            print("=" * 60)
            
            os.chdir(self.project_path)
            
            # Get list of data files
            data_files = self.get_data_files()
            print(f"Found {len(data_files)} data files to version")
            
            # Add data files to Git
            print("Adding data files to Git...")
            for data_dir in self.data_dirs:
                if (self.project_path / data_dir).exists():
                    subprocess.run(['git', 'add', data_dir], check=False)
            
            # Also add other important files
            important_files = [
                'requirements.txt',
                'README.md',
                'validation_reports/',
                'model_reports/',
                'pipeline_reports/',
                'data_versions/'
            ]
            
            for file_pattern in important_files:
                if (self.project_path / file_pattern).exists():
                    subprocess.run(['git', 'add', file_pattern], check=False)
            
            # Create version metadata
            metadata = self._create_version_metadata(version_name, description, data_files)
            
            # Save metadata
            metadata_dir = self.project_path / "data_versions"
            metadata_dir.mkdir(exist_ok=True)
            
            metadata_file = metadata_dir / f"{version_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Add metadata to Git
            subprocess.run(['git', 'add', 'data_versions/'], check=True)
            
            # Commit changes
            commit_message = f"Data version {version_name}: {description}" if description else f"Data version {version_name}"
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            
            # Create Git tag
            tag_message = f"Data version {version_name} - {description}" if description else f"Data version {version_name}"
            subprocess.run(['git', 'tag', '-a', version_name, '-m', tag_message], check=True)
            
            print(f"Data version created: {version_name}")
            print(f"Commit message: {commit_message}")
            print(f"Git tag: {version_name}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to create data version: {e}")
            return False
        except Exception as e:
            print(f"Error creating data version: {str(e)}")
            return False
    
    def _create_version_metadata(self, version_name: str, description: str, data_files: List[str]) -> Dict:
        """Create metadata for the version."""
        
        # Calculate statistics
        total_size = 0
        file_types = {}
        
        for file_path_str in data_files:
            file_path = self.project_path / file_path_str
            if file_path.exists():
                try:
                    size = file_path.stat().st_size
                    total_size += size
                    
                    ext = file_path.suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
                except:
                    continue
        
        # Get Git commit hash
        try:
            git_hash = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                    capture_output=True, text=True, check=True).stdout.strip()
        except:
            git_hash = "unknown"
        
        metadata = {
            "version_name": version_name,
            "description": description,
            "created_timestamp": datetime.now().isoformat(),
            "git_commit_hash": git_hash,
            "statistics": {
                "total_files": len(data_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": file_types
            },
            "data_directories": {
                data_dir: {
                    "exists": (self.project_path / data_dir).exists(),
                    "file_count": len(list((self.project_path / data_dir).rglob('*'))) if (self.project_path / data_dir).exists() else 0
                }
                for data_dir in self.data_dirs
            },
            "file_list": data_files[:50],  # First 50 files
            "environment": {
                "python_version": sys.version,
                "working_directory": str(self.project_path),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        return metadata
    
    def upload_to_github(self, include_tags: bool = True) -> bool:
        """
        Upload data versions to GitHub repository.
        
        Args:
            include_tags: Whether to push tags (versions)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print("\nUploading to GitHub...")
            print("=" * 40)
            
            os.chdir(self.project_path)
            
            # Push main branch
            print("Pushing main branch...")
            result = subprocess.run(['git', 'push', 'origin', 'main'], 
                                  capture_output=True, text=True, check=True)
            print("Main branch pushed successfully")
            
            if include_tags:
                # Push all tags
                print("Pushing version tags...")
                result = subprocess.run(['git', 'push', 'origin', '--tags'], 
                                      capture_output=True, text=True, check=True)
                print("All version tags pushed successfully")
            
            print("\nUpload to GitHub completed!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to upload to GitHub: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr.decode()}")
            print("\nNote: Make sure you have:")
            print("1. Valid GitHub authentication (token)")
            print("2. Correct remote repository URL")
            print("3. Push permissions to the repository")
            return False
        except Exception as e:
            print(f"Upload error: {str(e)}")
            return False
    
    def list_versions(self) -> List[Dict]:
        """List all data versions."""
        try:
            os.chdir(self.project_path)
            result = subprocess.run(['git', 'tag', '-l'], 
                                  capture_output=True, text=True, check=True)
            tags = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            versions = []
            for tag in tags:
                # Get tag details
                tag_info = subprocess.run(['git', 'show', tag, '--format=%cd|%s|%H', '--date=iso', '-s'], 
                                        capture_output=True, text=True)
                if tag_info.stdout:
                    parts = tag_info.stdout.split('\n')[0].split('|')
                    if len(parts) >= 3:
                        versions.append({
                            "version": tag,
                            "date": parts[0],
                            "description": parts[1],
                            "commit_hash": parts[2][:8]
                        })
            
            return versions
        except:
            return []
    
    def show_version_status(self):
        """Show current versioning status."""
        print("\nDATA VERSIONING STATUS")
        print("=" * 50)
        
        # Check Git status
        if not self.check_git_setup():
            print("Git repository not properly set up")
            return
        
        # Get data files
        data_files = self.get_data_files()
        print(f"Data files found: {len(data_files)}")
        
        # Calculate total size
        total_size = 0
        for file_path_str in data_files:
            file_path = self.project_path / file_path_str
            if file_path.exists():
                try:
                    total_size += file_path.stat().st_size
                except:
                    continue
        
        print(f"Total data size: {total_size / (1024*1024):.2f} MB")
        
        # List versions
        versions = self.list_versions()
        print(f"Total versions: {len(versions)}")
        
        if versions:
            print("\nRecent versions:")
            for version in versions[-3:]:  # Last 3 versions
                print(f"   {version['version']}: {version['description']} ({version['date'][:10]})")
        
        # Check for uncommitted changes
        try:
            os.chdir(self.project_path)
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                print("Uncommitted changes detected")
                print("   Consider creating a new version!")
            else:
                print("Working directory clean")
        except:
            pass
    
    def create_version_summary_report(self) -> str:
        """Create a comprehensive version summary report."""
        print("\nGenerating Version Summary Report...")
        
        versions = self.list_versions()
        data_files = self.get_data_files()
        
        # Calculate statistics
        total_size = 0
        file_types = {}
        
        for file_path_str in data_files:
            file_path = self.project_path / file_path_str
            if file_path.exists():
                try:
                    size = file_path.stat().st_size
                    total_size += size
                    
                    ext = file_path.suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
                except:
                    continue
        
        # Create summary
        summary = {
            "project": "DMML - End-to-End ML Pipeline",
            "students": ["2024ab05134", "2024aa05664"],
            "versioning_system": "Git-based Data Versioning",
            "report_generated": datetime.now().isoformat(),
            "summary": {
                "total_versions": len(versions),
                "total_data_files": len(data_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types_distribution": file_types
            },
            "data_directories": {
                data_dir: {
                    "exists": (self.project_path / data_dir).exists(),
                    "file_count": len(list((self.project_path / data_dir).rglob('*'))) if (self.project_path / data_dir).exists() else 0
                }
                for data_dir in self.data_dirs
            },
            "version_history": versions,
            "latest_version": versions[-1] if versions else None
        }
        
        # Save summary
        summary_dir = self.project_path / "data_versions"
        summary_dir.mkdir(exist_ok=True)
        
        summary_file = summary_dir / f"versioning_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Version summary saved: {summary_file}")
        return str(summary_file)


def demo_data_versioning():
    """Demonstrate data versioning functionality."""
    print("DATA VERSIONING DEMO")
    print("=" * 60)
    print("Students: 2024ab05134, 2024aa05664")
    print()
    
    # Initialize manager
    manager = DataVersionManager()
    
    # Show current status
    manager.show_version_status()
    
    # Create a new version
    print("\nCreating new data version...")
    success = manager.create_data_version(
        version_name="v1.0-complete-pipeline",
        description="Complete ML pipeline with all datasets and models"
    )
    
    if success:
        # Generate summary report
        manager.create_version_summary_report()
        
        # Ask about GitHub upload
        print("\nUpload to GitHub?")
        choice = input("Upload this version to GitHub? (y/n): ").strip().lower()
        if choice == 'y':
            manager.upload_to_github()
    
    print("\nData versioning demo completed!")


if __name__ == "__main__":
    demo_data_versioning()