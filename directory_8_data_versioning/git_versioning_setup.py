"""
Git-Based Data Versioning Setup
Students: 2024ab05134, 2024aa05664

This script sets up Git-based data versioning for the ML pipeline project.
It helps maintain data versions using Git and push them to GitHub.
"""

import os
import subprocess
import json
import shutil
from datetime import datetime
from pathlib import Path

class GitDataVersioning:
    """Git-based data versioning system."""
    
    def __init__(self, project_path="/Users/puneetsinha/DMML"):
        """Initialize Git versioning system."""
        self.project_path = Path(project_path)
        self.git_dir = self.project_path / ".git"
        
    def check_git_installation(self):
        """Check if Git is properly installed."""
        try:
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Git is installed: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Git is not installed or not accessible")
            print("Please install Xcode command line tools:")
            print("   xcode-select --install")
            return False
    
    def initialize_repository(self):
        """Initialize Git repository if not already done."""
        if self.git_dir.exists():
            print("✅ Git repository already initialized")
            return True
        
        try:
            os.chdir(self.project_path)
            subprocess.run(['git', 'init'], check=True)
            print("✅ Git repository initialized")
            
            # Configure Git user (if not already configured)
            self.configure_git_user()
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to initialize Git repository: {e}")
            return False
    
    def configure_git_user(self):
        """Configure Git user information."""
        try:
            # Check if user is already configured
            result = subprocess.run(['git', 'config', 'user.name'], 
                                  capture_output=True, text=True)
            if not result.stdout.strip():
                # Configure default user info for students
                subprocess.run(['git', 'config', 'user.name', 'Students 2024ab05134 2024aa05664'], check=True)
                subprocess.run(['git', 'config', 'user.email', 'students@university.edu'], check=True)
                print("✅ Git user configured")
            else:
                print(f"✅ Git user already configured: {result.stdout.strip()}")
        except subprocess.CalledProcessError:
            print("⚠️  Could not configure Git user - please configure manually")
    
    def create_gitignore(self):
        """Create .gitignore file for ML project."""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
*.tmp
*.temp

# Large data files (comment out if you want to version them)
# *.csv
# *.parquet
# *.xlsx
# *.db

# MLflow
mlruns/

# Model files (large)
# *.pkl
# *.joblib
"""
        
        gitignore_path = self.project_path / ".gitignore"
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print("✅ .gitignore file created")
    
    def create_data_version_tag(self, version_name, description=""):
        """Create a Git tag for data version."""
        try:
            os.chdir(self.project_path)
            
            # Add all files
            subprocess.run(['git', 'add', '.'], check=True)
            
            # Commit changes
            commit_message = f"Data version {version_name}: {description}"
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            
            # Create tag
            tag_message = f"Data version {version_name} - {description}"
            subprocess.run(['git', 'tag', '-a', version_name, '-m', tag_message], check=True)
            
            print(f"✅ Created data version tag: {version_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create version tag: {e}")
            return False
    
    def list_data_versions(self):
        """List all data version tags."""
        try:
            os.chdir(self.project_path)
            result = subprocess.run(['git', 'tag', '-l'], 
                                  capture_output=True, text=True, check=True)
            tags = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            print("📋 Data Versions:")
            if tags:
                for tag in tags:
                    # Get tag info
                    tag_info = subprocess.run(['git', 'show', tag, '--format=%cd %s', '--date=short'], 
                                            capture_output=True, text=True)
                    print(f"   {tag}: {tag_info.stdout.split('\n')[0] if tag_info.stdout else 'No info'}")
            else:
                print("   No versions found")
        except subprocess.CalledProcessError:
            print("❌ Could not list versions")
    
    def setup_github_remote(self, github_repo_url):
        """Setup GitHub remote repository."""
        try:
            os.chdir(self.project_path)
            
            # Add remote origin
            subprocess.run(['git', 'remote', 'add', 'origin', github_repo_url], check=True)
            print(f"✅ GitHub remote added: {github_repo_url}")
            return True
        except subprocess.CalledProcessError:
            # Remote might already exist
            try:
                subprocess.run(['git', 'remote', 'set-url', 'origin', github_repo_url], check=True)
                print(f"✅ GitHub remote updated: {github_repo_url}")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to setup GitHub remote: {e}")
                return False
    
    def push_to_github(self, include_tags=True):
        """Push repository and tags to GitHub."""
        try:
            os.chdir(self.project_path)
            
            # Push main branch
            subprocess.run(['git', 'push', '-u', 'origin', 'main'], check=True)
            print("✅ Pushed main branch to GitHub")
            
            if include_tags:
                # Push all tags
                subprocess.run(['git', 'push', 'origin', '--tags'], check=True)
                print("✅ Pushed all version tags to GitHub")
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to push to GitHub: {e}")
            print("Note: Make sure you have GitHub authentication set up")
            return False
    
    def create_version_summary(self):
        """Create a summary of all data versions."""
        try:
            os.chdir(self.project_path)
            result = subprocess.run(['git', 'tag', '-l'], 
                                  capture_output=True, text=True, check=True)
            tags = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            summary = {
                "project": "DMML - End-to-End ML Pipeline",
                "students": ["2024ab05134", "2024aa05664"],
                "versioning_system": "Git-based",
                "total_versions": len(tags),
                "last_updated": datetime.now().isoformat(),
                "versions": []
            }
            
            for tag in tags:
                # Get tag details
                tag_info = subprocess.run(['git', 'show', tag, '--format=%cd|%s|%H', '--date=iso'], 
                                        capture_output=True, text=True)
                if tag_info.stdout:
                    parts = tag_info.stdout.split('\n')[0].split('|')
                    if len(parts) >= 3:
                        summary["versions"].append({
                            "version": tag,
                            "date": parts[0],
                            "description": parts[1],
                            "commit_hash": parts[2][:8]
                        })
            
            # Save summary
            summary_path = self.project_path / "data_versions" / "git_version_summary.json"
            summary_path.parent.mkdir(exist_ok=True)
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"✅ Version summary created: {summary_path}")
            return summary
        except Exception as e:
            print(f"❌ Failed to create version summary: {e}")
            return None

def setup_git_versioning_guide():
    """Print setup guide for Git-based data versioning."""
    print("=" * 70)
    print("GIT-BASED DATA VERSIONING SETUP GUIDE")
    print("Students: 2024ab05134, 2024aa05664")
    print("=" * 70)
    print()
    
    print("STEP 1: Install Prerequisites")
    print("-" * 30)
    print("1. Install Xcode command line tools:")
    print("   xcode-select --install")
    print("2. Verify Git installation:")
    print("   git --version")
    print()
    
    print("STEP 2: GitHub Repository Setup")
    print("-" * 30)
    print("1. Create a new repository on GitHub:")
    print("   - Go to https://github.com")
    print("   - Click 'New repository'")
    print("   - Name: 'DMML-ML-Pipeline' or similar")
    print("   - Make it Public or Private")
    print("   - Don't initialize with README (we have our own)")
    print("2. Copy the repository URL (e.g., https://github.com/username/DMML-ML-Pipeline.git)")
    print()
    
    print("STEP 3: Run Setup Script")
    print("-" * 30)
    print("1. Run the automated setup:")
    print("   python directory_8_data_versioning/git_versioning_setup.py")
    print("2. Follow the prompts to configure GitHub remote")
    print()
    
    print("STEP 4: Create Initial Data Version")
    print("-" * 30)
    print("1. Create first version tag:")
    print("   git add .")
    print("   git commit -m 'Initial data version - raw datasets'")
    print("   git tag -a v1.0-raw-data -m 'Version 1.0: Raw datasets from Kaggle and HuggingFace'")
    print()
    
    print("STEP 5: Version Your Data Pipeline Stages")
    print("-" * 30)
    print("Create versions for each pipeline stage:")
    print("   v1.0-raw-data       - Original datasets")
    print("   v1.1-validated      - After data validation")
    print("   v1.2-cleaned        - After data preparation")
    print("   v1.3-transformed    - After feature engineering")
    print("   v1.4-feature-store  - After feature store setup")
    print("   v1.5-production     - Final production-ready version")
    print()
    
    print("STEP 6: Push to GitHub")
    print("-" * 30)
    print("1. Push repository:")
    print("   git push -u origin main")
    print("2. Push all version tags:")
    print("   git push origin --tags")
    print()
    
    print("BENEFITS OF GIT-BASED VERSIONING:")
    print("-" * 40)
    print("✅ Complete project history in one place")
    print("✅ Easy branching for experimental features")
    print("✅ Collaborative development with team members")
    print("✅ Integration with GitHub's collaboration tools")
    print("✅ Automatic backup and distributed storage")
    print("✅ Easy rollback to any previous version")
    print("✅ Tagging system for meaningful version names")
    print()

def main():
    """Main function to set up Git-based data versioning."""
    setup_git_versioning_guide()
    
    # Initialize versioning system
    git_versioning = GitDataVersioning()
    
    print("AUTOMATED SETUP")
    print("=" * 70)
    
    # Check Git installation
    if not git_versioning.check_git_installation():
        print("\n⚠️  Please install Git first, then run this script again.")
        return
    
    # Initialize repository
    if git_versioning.initialize_repository():
        git_versioning.create_gitignore()
    
    # Get GitHub repository URL from user
    print("\nGITHUB SETUP")
    print("-" * 20)
    github_url = input("Enter your GitHub repository URL (or press Enter to skip): ").strip()
    
    if github_url:
        if git_versioning.setup_github_remote(github_url):
            print("\n✅ GitHub remote configured successfully!")
            print("You can now push your versions to GitHub using:")
            print("   git push -u origin main")
            print("   git push origin --tags")
    
    # Create initial version
    print("\nCREATE INITIAL VERSION")
    print("-" * 30)
    create_initial = input("Create initial data version? (y/n): ").strip().lower()
    
    if create_initial == 'y':
        if git_versioning.create_data_version_tag("v1.0-initial", "Initial ML pipeline with all components"):
            print("✅ Initial version created!")
    
    # List current versions
    git_versioning.list_data_versions()
    
    # Create version summary
    git_versioning.create_version_summary()
    
    print("\n🎉 Git-based data versioning setup complete!")
    print("You can now version your data using Git tags and push to GitHub.")

if __name__ == "__main__":
    main()
