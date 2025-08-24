"""
Data Version Manager with DVC integration.
Handles data versioning, tracking, and reproducibility.
"""

import os
import json
import subprocess
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import sys

from versioning_config import versioning_config


class DataVersionManager:
    """Manage data versions using DVC and Git."""
    
    def __init__(self):
        """Initialize the data version manager."""
        self.config = versioning_config
        self.base_dir = self.config.BASE_DIR
        self.is_initialized = False
        self.current_version = None
        
    def initialize_versioning(self) -> bool:
        """
        Initialize DVC and Git repositories for version control.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            print(" Initializing Data Versioning System")
            print("=" * 50)
            
            # Change to base directory
            os.chdir(self.base_dir)
            
            # Initialize Git repository if not exists
            if not os.path.exists(".git"):
                print(" Initializing Git repository...")
                subprocess.run(["git", "init"], check=True, capture_output=True)
                
                # Configure Git user
                subprocess.run([
                    "git", "config", "user.name", 
                    self.config.GIT_CONFIG["user_name"]
                ], check=True, capture_output=True)
                
                subprocess.run([
                    "git", "config", "user.email", 
                    self.config.GIT_CONFIG["user_email"]
                ], check=True, capture_output=True)
                
                print(" Git repository initialized")
            else:
                print(" Git repository already exists")
            
            # Initialize DVC if not exists
            if not os.path.exists(".dvc"):
                print(" Initializing DVC...")
                subprocess.run(["dvc", "init"], check=True, capture_output=True)
                print(" DVC initialized")
            else:
                print(" DVC already initialized")
            
            # Configure DVC remote storage
            self._configure_dvc_remote()
            
            # Create initial .gitignore
            self._create_gitignore()
            
            # Create version metadata structure
            self._initialize_version_metadata()
            
            self.is_initialized = True
            print(" Data versioning system initialized successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f" Command failed: {e.cmd}")
            print(f"   Error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            return False
        except Exception as e:
            print(f" Initialization failed: {str(e)}")
            return False
    
    def _configure_dvc_remote(self):
        """Configure DVC remote storage."""
        try:
            remote_config = self.config.DVC_CONFIG["remote_storage"]
            
            # Add remote if not exists
            result = subprocess.run([
                "dvc", "remote", "list"
            ], capture_output=True, text=True)
            
            if remote_config["name"] not in result.stdout:
                subprocess.run([
                    "dvc", "remote", "add", 
                    remote_config["name"], 
                    remote_config["path"]
                ], check=True, capture_output=True)
                
                # Set as default remote
                subprocess.run([
                    "dvc", "remote", "default", 
                    remote_config["name"]
                ], check=True, capture_output=True)
                
                print(f" DVC remote '{remote_config['name']}' configured")
            else:
                print(f" DVC remote '{remote_config['name']}' already exists")
                
        except subprocess.CalledProcessError as e:
            print(f" Failed to configure DVC remote: {e}")
    
    def _create_gitignore(self):
        """Create comprehensive .gitignore file."""
        gitignore_content = """
# DVC
/dvc.lock

# Data directories (tracked by DVC)
/raw_data/
/processed_data/
/transformed_data/
/models/

# Feature store database
/feature_store/*.db

# Logs
/logs/
*.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
*.bak

# Model artifacts
*.pkl
*.joblib
*.h5
*.onnx

# Large data files
*.csv
*.parquet
*.feather
*.json
*.xlsx
*.xls

# Plots and reports
/plots/
/reports/
        """.strip()
        
        gitignore_path = os.path.join(self.base_dir, ".gitignore")
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        
        print(" .gitignore created")
    
    def _initialize_version_metadata(self):
        """Initialize version metadata structure."""
        metadata_dir = os.path.join(self.base_dir, "version_metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Create initial metadata file
        initial_metadata = {
            "versioning_initialized": datetime.now().isoformat(),
            "versions": {},
            "current_version": None,
            "data_lineage": {},
            "version_history": []
        }
        
        metadata_file = os.path.join(metadata_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(initial_metadata, f, indent=2)
        
        # Create initial changelog
        changelog_content = """# Data Version Changelog

All notable changes to datasets and models will be documented in this file.

## [Unreleased]

### Added
- Initial data versioning system setup
- DVC configuration for data tracking
- Git repository for code and metadata versioning

"""
        
        changelog_file = os.path.join(metadata_dir, "CHANGELOG.md")
        with open(changelog_file, 'w') as f:
            f.write(changelog_content)
        
        print(" Version metadata structure created")
    
    def create_data_version(self, description: str, tag: str = "data") -> str:
        """
        Create a new data version.
        
        Args:
            description: Description of the version
            tag: Version tag (data, model, experiment, feature)
            
        Returns:
            Version identifier
        """
        if not self.is_initialized:
            print(" Versioning system not initialized. Run initialize_versioning() first.")
            return None
        
        try:
            print(f"\n Creating Data Version: {description}")
            print("=" * 60)
            
            # Generate version identifier
            now = datetime.now()
            version_id = self.config.VERSION_METADATA["version_format"].format(
                year=now.year,
                month=now.month,
                day=now.day,
                hour=now.hour,
                minute=now.minute
            )
            
            # Track data directories with DVC
            dvc_files = []
            for data_dir in self.config.DATA_DIRECTORIES:
                data_path = os.path.join(self.base_dir, data_dir)
                if os.path.exists(data_path):
                    try:
                        print(f" Tracking {data_dir} with DVC...")
                        
                        # Add directory to DVC tracking
                        result = subprocess.run([
                            "dvc", "add", data_dir
                        ], check=True, capture_output=True, text=True)
                        
                        dvc_file = f"{data_dir}.dvc"
                        if os.path.exists(dvc_file):
                            dvc_files.append(dvc_file)
                            print(f" {data_dir} tracked with DVC")
                        
                    except subprocess.CalledProcessError as e:
                        print(f" Failed to track {data_dir}: {e}")
                        continue
            
            # Create version metadata
            version_metadata = self._create_version_metadata(version_id, description, tag, dvc_files)
            
            # Save version metadata
            self._save_version_metadata(version_id, version_metadata)
            
            # Commit to Git
            self._commit_version(version_id, description, dvc_files)
            
            # Create Git tag
            self._create_git_tag(version_id, description)
            
            # Update current version
            self.current_version = version_id
            
            print(f" Data version created: {version_id}")
            return version_id
            
        except Exception as e:
            print(f" Failed to create data version: {str(e)}")
            return None
    
    def _create_version_metadata(self, version_id: str, description: str, tag: str, dvc_files: List[str]) -> Dict:
        """Create comprehensive version metadata."""
        
        # Calculate data statistics
        data_stats = {}
        for data_dir in self.config.DATA_DIRECTORIES:
            data_path = os.path.join(self.base_dir, data_dir)
            if os.path.exists(data_path):
                stats = self._calculate_directory_stats(data_path)
                data_stats[data_dir] = stats
        
        # Get Git commit hash
        try:
            git_hash = subprocess.run([
                "git", "rev-parse", "HEAD"
            ], capture_output=True, text=True, check=True).stdout.strip()
        except:
            git_hash = "initial"
        
        metadata = {
            "version_id": version_id,
            "description": description,
            "tag": tag,
            "created_timestamp": datetime.now().isoformat(),
            "git_commit": git_hash,
            "dvc_files": dvc_files,
            "data_statistics": data_stats,
            "directory_structure": self._get_directory_structure(),
            "dependencies": self._get_dependencies_info(),
            "environment": {
                "python_version": sys.version,
                "working_directory": os.getcwd(),
                "user": os.getenv("USER", "unknown")
            }
        }
        
        return metadata
    
    def _calculate_directory_stats(self, directory: str) -> Dict:
        """Calculate statistics for a directory."""
        stats = {
            "total_files": 0,
            "total_size_bytes": 0,
            "file_types": {},
            "largest_file": None,
            "last_modified": None
        }
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        stats["total_files"] += 1
                        
                        file_size = os.path.getsize(file_path)
                        stats["total_size_bytes"] += file_size
                        
                        # Track file types
                        ext = os.path.splitext(file)[1].lower()
                        stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
                        
                        # Track largest file
                        if stats["largest_file"] is None or file_size > stats["largest_file"]["size"]:
                            stats["largest_file"] = {
                                "name": file,
                                "size": file_size,
                                "path": os.path.relpath(file_path, self.base_dir)
                            }
                        
                        # Track last modified
                        mtime = os.path.getmtime(file_path)
                        if stats["last_modified"] is None or mtime > stats["last_modified"]:
                            stats["last_modified"] = mtime
            
            # Convert last_modified to readable format
            if stats["last_modified"]:
                stats["last_modified"] = datetime.fromtimestamp(stats["last_modified"]).isoformat()
            
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    def _get_directory_structure(self) -> Dict:
        """Get directory structure snapshot."""
        structure = {}
        
        for data_dir in self.config.DATA_DIRECTORIES:
            data_path = os.path.join(self.base_dir, data_dir)
            if os.path.exists(data_path):
                structure[data_dir] = {
                    "exists": True,
                    "subdirectories": [],
                    "file_count": 0
                }
                
                try:
                    for item in os.listdir(data_path):
                        item_path = os.path.join(data_path, item)
                        if os.path.isdir(item_path):
                            structure[data_dir]["subdirectories"].append(item)
                        else:
                            structure[data_dir]["file_count"] += 1
                except:
                    structure[data_dir]["error"] = "Cannot read directory"
            else:
                structure[data_dir] = {"exists": False}
        
        return structure
    
    def _get_dependencies_info(self) -> Dict:
        """Get information about dependencies and environment."""
        dependencies = {}
        
        # Check for requirements.txt
        req_file = os.path.join(self.base_dir, "requirements.txt")
        if os.path.exists(req_file):
            with open(req_file, 'r') as f:
                dependencies["requirements"] = f.read().strip().split('\n')
        
        # Get DVC version
        try:
            dvc_version = subprocess.run([
                "dvc", "version"
            ], capture_output=True, text=True).stdout.strip()
            dependencies["dvc_version"] = dvc_version
        except:
            dependencies["dvc_version"] = "unknown"
        
        return dependencies
    
    def _save_version_metadata(self, version_id: str, metadata: Dict):
        """Save version metadata to file."""
        metadata_dir = os.path.join(self.base_dir, "version_metadata")
        
        # Save individual version metadata
        version_file = os.path.join(metadata_dir, f"{version_id}.json")
        with open(version_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Update main metadata file
        main_metadata_file = os.path.join(metadata_dir, "metadata.json")
        if os.path.exists(main_metadata_file):
            with open(main_metadata_file, 'r') as f:
                main_metadata = json.load(f)
        else:
            main_metadata = {"versions": {}, "version_history": []}
        
        main_metadata["versions"][version_id] = {
            "description": metadata["description"],
            "tag": metadata["tag"],
            "created_timestamp": metadata["created_timestamp"],
            "metadata_file": version_file
        }
        
        main_metadata["current_version"] = version_id
        main_metadata["version_history"].append(version_id)
        
        with open(main_metadata_file, 'w') as f:
            json.dump(main_metadata, f, indent=2, default=str)
        
        print(f" Version metadata saved: {version_file}")
    
    def _commit_version(self, version_id: str, description: str, dvc_files: List[str]):
        """Commit version to Git."""
        try:
            # Add DVC files and metadata to Git
            files_to_add = dvc_files + [
                "version_metadata/",
                ".gitignore"
            ]
            
            for file_pattern in files_to_add:
                subprocess.run([
                    "git", "add", file_pattern
                ], check=True, capture_output=True)
            
            # Commit
            commit_message = self.config.GIT_CONFIG["commit_message_template"].format(
                version=version_id,
                description=description
            )
            
            subprocess.run([
                "git", "commit", "-m", commit_message
            ], check=True, capture_output=True)
            
            print(f" Changes committed to Git: {version_id}")
            
        except subprocess.CalledProcessError as e:
            print(f" Git commit failed: {e}")
    
    def _create_git_tag(self, version_id: str, description: str):
        """Create Git tag for version."""
        try:
            subprocess.run([
                "git", "tag", "-a", version_id, "-m", f"Data version: {description}"
            ], check=True, capture_output=True)
            
            print(f" Git tag created: {version_id}")
            
        except subprocess.CalledProcessError as e:
            print(f" Git tag creation failed: {e}")
    
    def list_versions(self) -> List[Dict]:
        """
        List all data versions.
        
        Returns:
            List of version information
        """
        metadata_file = os.path.join(self.base_dir, "version_metadata", "metadata.json")
        
        if not os.path.exists(metadata_file):
            return []
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            versions = []
            for version_id, version_info in metadata.get("versions", {}).items():
                versions.append({
                    "version_id": version_id,
                    "description": version_info["description"],
                    "tag": version_info["tag"],
                    "created_timestamp": version_info["created_timestamp"]
                })
            
            # Sort by creation time (newest first)
            versions.sort(key=lambda x: x["created_timestamp"], reverse=True)
            return versions
            
        except Exception as e:
            print(f" Failed to list versions: {str(e)}")
            return []
    
    def checkout_version(self, version_id: str) -> bool:
        """
        Checkout a specific data version.
        
        Args:
            version_id: Version to checkout
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f" Checking out version: {version_id}")
            
            # Checkout Git tag
            subprocess.run([
                "git", "checkout", version_id
            ], check=True, capture_output=True)
            
            # Checkout DVC data
            subprocess.run([
                "dvc", "checkout"
            ], check=True, capture_output=True)
            
            print(f" Successfully checked out version: {version_id}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f" Checkout failed: {e}")
            return False
    
    def generate_version_report(self) -> Dict:
        """Generate comprehensive version report."""
        
        print("\n Generating Version Report")
        print("=" * 40)
        
        versions = self.list_versions()
        
        if not versions:
            return {"error": "No versions found"}
        
        # Calculate statistics
        total_versions = len(versions)
        tags_count = {}
        
        for version in versions:
            tag = version.get("tag", "unknown")
            tags_count[tag] = tags_count.get(tag, 0) + 1
        
        # Get current status
        current_status = self._get_current_status()
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "versioning_summary": {
                "total_versions": total_versions,
                "tags_distribution": tags_count,
                "latest_version": versions[0] if versions else None,
                "current_status": current_status
            },
            "version_history": versions,
            "recommendations": self._generate_versioning_recommendations(versions, current_status)
        }
        
        # Save report
        report_file = os.path.join(
            self.base_dir, 
            "version_metadata", 
            f"version_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f" Version report saved: {report_file}")
        return report
    
    def _get_current_status(self) -> Dict:
        """Get current repository status."""
        status = {}
        
        try:
            # Git status
            git_status = subprocess.run([
                "git", "status", "--porcelain"
            ], capture_output=True, text=True).stdout.strip()
            
            status["git_clean"] = len(git_status) == 0
            status["git_changes"] = git_status.split('\n') if git_status else []
            
            # DVC status
            dvc_status = subprocess.run([
                "dvc", "status"
            ], capture_output=True, text=True).stdout.strip()
            
            status["dvc_clean"] = "Data and pipelines are up to date" in dvc_status
            
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    def _generate_versioning_recommendations(self, versions: List[Dict], current_status: Dict) -> List[str]:
        """Generate recommendations based on versioning analysis."""
        recommendations = []
        
        if not versions:
            recommendations.append("Create your first data version to start tracking changes")
            return recommendations
        
        # Check version frequency
        if len(versions) == 1:
            recommendations.append("Consider creating regular data versions as your pipeline evolves")
        elif len(versions) > 10:
            recommendations.append("Good versioning practice! Consider cleaning up old versions periodically")
        
        # Check for uncommitted changes
        if not current_status.get("git_clean", True):
            recommendations.append("You have uncommitted changes - consider creating a new version")
        
        if not current_status.get("dvc_clean", True):
            recommendations.append("DVC data is out of sync - run 'dvc checkout' or create new version")
        
        # Check tag distribution
        latest_version = versions[0] if versions else None
        if latest_version and latest_version.get("tag") == "experiment":
            recommendations.append("Latest version is experimental - consider creating a stable data version")
        
        # General recommendations
        recommendations.extend([
            "Maintain regular versioning schedule aligned with model releases",
            "Use descriptive version descriptions for better tracking",
            "Consider setting up automated versioning in CI/CD pipeline"
        ])
        
        return recommendations


if __name__ == "__main__":
    manager = DataVersionManager()
    print("Data Version Manager initialized successfully")
