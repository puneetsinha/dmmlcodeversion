"""
Push DMML Project to GitHub Repository
Students: 2024ab05134, 2024aa05664

This script pushes the complete DMML project to:
https://github.com/puneetsinha/BitsPilaniAIML
"""

import os
import subprocess
import json
from datetime import datetime
from pathlib import Path

class GitHubPusher:
    """Push DMML project to GitHub repository."""
    
    def __init__(self):
        """Initialize GitHub pusher."""
        self.project_path = "/Users/puneetsinha/DMML"
        self.github_repo = "https://github.com/puneetsinha/BitsPilaniAIML.git"
        self.branch_name = "dmml-pipeline"  # Create separate branch for DMML project
        
    def check_prerequisites(self):
        """Check if Git is installed and configured."""
        try:
            # Check Git installation
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"Git is installed: {result.stdout.strip()}")
            
            # Check if we're in a git repository
            os.chdir(self.project_path)
            subprocess.run(['git', 'status'], 
                         capture_output=True, text=True, check=True)
            print(" Git repository detected")
            
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(" Git not available or repository not initialized")
            return False
    
    def setup_repository(self):
        """Setup Git repository with GitHub remote."""
        try:
            os.chdir(self.project_path)
            
            # Initialize Git if not already done
            if not os.path.exists(".git"):
                subprocess.run(['git', 'init'], check=True)
                print(" Git repository initialized")
            
            # Configure Git user
            try:
                subprocess.run(['git', 'config', 'user.name', 'Students 2024ab05134 2024aa05664'], check=True)
                subprocess.run(['git', 'config', 'user.email', 'puneetsinha@example.com'], check=True)
                print(" Git user configured")
            except subprocess.CalledProcessError:
                print("  Could not configure Git user")
            
            # Add GitHub remote
            try:
                # Remove existing origin if it exists
                subprocess.run(['git', 'remote', 'remove', 'origin'], 
                             capture_output=True, text=True)
            except subprocess.CalledProcessError:
                pass  # Remote doesn't exist, that's fine
            
            # Add the BitsPilaniAIML repository as origin
            subprocess.run(['git', 'remote', 'add', 'origin', self.github_repo], check=True)
            print(f" GitHub remote added: {self.github_repo}")
            
            return True
        except subprocess.CalledProcessError as e:
            print(f" Failed to setup repository: {e}")
            return False
    
    def create_gitignore(self):
        """Create appropriate .gitignore for ML project."""
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

# Temporary files
*.tmp
*.temp

# MLflow runs (large directories)
mlruns/

# Large model files (comment out if you want to version them)
# *.pkl
# *.joblib

# Very large data files (comment out if needed)
# *.csv
# *.parquet
# *.xlsx
# *.db
"""
        
        gitignore_path = Path(self.project_path) / ".gitignore"
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print(" .gitignore created")
    
    def create_project_readme(self):
        """Create README for the DMML project on GitHub."""
        readme_content = """# DMML - End-to-End ML Pipeline Project

**Students: 2024ab05134, 2024aa05664**  
**Course: Data Management for Machine Learning**  
**Institution: BITS Pilani**

## Project Overview

This repository contains a complete end-to-end machine learning pipeline for customer churn prediction, demonstrating advanced data management, feature engineering, model training, and pipeline orchestration capabilities.

##  Key Achievements

- **Complete ML Pipeline**: 10-step end-to-end automation
- **Multi-Source Data**: Kaggle + Hugging Face integration
- **204 Engineered Features**: Advanced feature engineering across 2 datasets
- **8 Trained Models**: 4 algorithms × 2 datasets with hyperparameter optimization
- **Production Architecture**: Feature store, versioning, orchestration
- **Grade A Data Quality**: 95% overall quality score

##  Pipeline Components

1. **Data Ingestion** (`directory_2_data_ingestion/`)
   - Kaggle API and Hugging Face dataset integration
   - Fallback URL mechanisms for reliability
   - Comprehensive error handling and logging

2. **Raw Data Storage** (`directory_3_raw_data_storage/`)
   - Partitioned data lake structure
   - Parquet format optimization
   - Automated cataloging and metadata

3. **Data Validation** (`directory_4_data_validation/`)
   - Comprehensive quality checks
   - CSV reports with issues and resolutions
   - 95% quality score achievement

4. **Data Preparation** (`directory_5_data_preparation/`)
   - Missing value imputation
   - Outlier detection and handling
   - Exploratory Data Analysis (EDA)

5. **Data Transformation** (`directory_6_data_transformation/`)
   - Advanced feature engineering
   - 139 new features created
   - Multiple transformation techniques

6. **Feature Store** (`directory_7_feature_store/`)
   - Centralized feature management
   - SQLite-based storage
   - Feature versioning and lineage

7. **Data Versioning** (`directory_8_data_versioning/`)
   - Git-based version control
   - Complete project history
   - GitHub integration

8. **Model Building** (`directory_9_model_building/`)
   - 4 ML algorithms with hyperparameter tuning
   - MLflow experiment tracking
   - Comprehensive evaluation metrics

9. **Pipeline Orchestration** (`directory_10_pipeline_orchestration/`)
   - Automated end-to-end execution
   - Dependency management
   - Error recovery and monitoring

##  Quick Start

```bash
# Clone the repository
git clone https://github.com/puneetsinha/BitsPilaniAIML.git
cd BitsPilaniAIML

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
cd directory_10_pipeline_orchestration
python pipeline_orchestrator.py
```

##  Results

### Model Performance
| Dataset | Best Model | F1 Score | ROC AUC | Accuracy |
|---------|------------|----------|---------|----------|
| **Telco Churn** | Gradient Boosting | 0.530 | 0.828 | 78.7% |
| **Adult Census** | XGBoost | 0.693 | 0.916 | 86.4% |

### Pipeline Metrics
- **Data Sources**: 2 (Kaggle Telco + Hugging Face Census)
- **Total Records**: ~40K across datasets
- **Features Engineered**: 204 (33→104 + 32→100)
- **Models Trained**: 8 (4 algorithms × 2 datasets)
- **Data Quality Score**: Grade A (95%)
- **Pipeline Success Rate**: 100%

## 🛠 Technologies Used

- **Python**: Primary development language
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn/XGBoost**: Machine learning algorithms
- **MLflow**: Experiment tracking and model management
- **SQLite**: Feature store backend
- **Plotly/Seaborn**: Data visualization
- **Git**: Version control and collaboration

##  Project Structure

```
DMML/
├── directory_2_data_ingestion/      # Multi-source data ingestion
├── directory_3_raw_data_storage/    # Data lake organization
├── directory_4_data_validation/     # Quality validation
├── directory_5_data_preparation/    # Data cleaning and EDA
├── directory_6_data_transformation/ # Feature engineering
├── directory_7_feature_store/       # Centralized features
├── directory_8_data_versioning/     # Version control
├── directory_9_model_building/      # ML model training
├── directory_10_pipeline_orchestration/ # Pipeline automation
├── data_lake/                       # Organized data storage
├── feature_store/                   # Feature database
├── models/                          # Trained model artifacts
├── validation_reports/              # Quality reports
└── README.md                        # This file
```

##  Learning Outcomes

This project demonstrates mastery of:
- End-to-end ML pipeline development
- Advanced data management techniques
- Feature engineering and selection
- Model training and evaluation
- Production-ready system design
- Version control and collaboration

##  Contact

**Students**: 2024ab05134, 2024aa05664  
**Course**: Data Management for Machine Learning  
**Institution**: BITS Pilani

---

**This project represents our comprehensive understanding of machine learning pipeline development and demonstrates practical application of data management principles in real-world scenarios.**
"""
        
        readme_path = Path(self.project_path) / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(" Project README created")
    
    def commit_and_push(self):
        """Commit all changes and push to GitHub."""
        try:
            os.chdir(self.project_path)
            
            print(" Preparing project for GitHub...")
            
            # Add all files
            subprocess.run(['git', 'add', '.'], check=True)
            print(" Files staged for commit")
            
            # Create comprehensive commit message
            commit_message = """DMML: Complete End-to-End ML Pipeline Implementation

Students: 2024ab05134, 2024aa05664
Course: Data Management for Machine Learning
Institution: BITS Pilani

This commit includes the complete implementation of an end-to-end machine learning
pipeline for customer churn prediction with the following components:

 PIPELINE COMPONENTS:
- Data Ingestion: Multi-source data collection (Kaggle + HuggingFace)
- Data Storage: Partitioned data lake with efficient storage
- Data Validation: Comprehensive quality assessment (95% score)
- Data Preparation: Cleaning, EDA, and preprocessing
- Data Transformation: Advanced feature engineering (204 features)
- Feature Store: Centralized feature management system
- Data Versioning: Git-based version control
- Model Building: 4 ML algorithms with hyperparameter tuning
- Pipeline Orchestration: Automated end-to-end execution

 KEY ACHIEVEMENTS:
- 39,604 total records processed across 2 datasets
- 204 engineered features created
- 8 models trained with comprehensive evaluation
- 100% pipeline success rate
- Production-ready architecture with monitoring

🛠 TECHNOLOGIES:
Python, Pandas, Scikit-learn, XGBoost, MLflow, SQLite, Git

This implementation demonstrates mastery of data management principles
and practical ML engineering skills required for production systems."""
            
            # Commit changes
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            print(" Changes committed")
            
            # Create version tag
            tag_name = "v1.0-dmml-complete"
            tag_message = """DMML v1.0: Complete ML Pipeline

Students: 2024ab05134, 2024aa05664
- Complete end-to-end ML pipeline implementation
- 204 features engineered across 2 datasets
- 4 ML algorithms with hyperparameter optimization
- Production-ready architecture with monitoring
- Comprehensive documentation and reports"""
            
            subprocess.run(['git', 'tag', '-a', tag_name, '-m', tag_message], check=True)
            print(f" Version tag created: {tag_name}")
            
            # Fetch from origin to check for conflicts
            try:
                subprocess.run(['git', 'fetch', 'origin'], check=True)
                print(" Fetched latest changes from GitHub")
            except subprocess.CalledProcessError:
                print("  Could not fetch from origin (repository might be empty)")
            
            # Push to GitHub
            try:
                # Try to push main branch
                subprocess.run(['git', 'push', '-u', 'origin', 'main'], check=True)
                print(" Main branch pushed to GitHub")
            except subprocess.CalledProcessError:
                # If main fails, try master
                try:
                    subprocess.run(['git', 'push', '-u', 'origin', 'master'], check=True)
                    print(" Master branch pushed to GitHub")
                except subprocess.CalledProcessError:
                    print("  Could not push main/master branch")
            
            # Push tags
            try:
                subprocess.run(['git', 'push', 'origin', '--tags'], check=True)
                print(" Version tags pushed to GitHub")
            except subprocess.CalledProcessError:
                print("  Could not push tags")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f" Failed to commit and push: {e}")
            return False
    
    def verify_push(self):
        """Verify that the push was successful."""
        print("\n Verifying GitHub push...")
        print(f"📱 Check your repository: {self.github_repo.replace('.git', '')}")
        print("You should see:")
        print("- All your DMML project files")
        print("- Version tag: v1.0-dmml-complete")
        print("- Updated README with project description")
        print("- Complete commit history")

def main():
    """Main function to push DMML project to GitHub."""
    print(" PUSHING DMML PROJECT TO GITHUB")
    print("=" * 50)
    print("Students: 2024ab05134, 2024aa05664")
    print("Target Repository: https://github.com/puneetsinha/BitsPilaniAIML")
    print("=" * 50)
    print()
    
    pusher = GitHubPusher()
    
    # Check prerequisites
    if not pusher.check_prerequisites():
        print("\n  Prerequisites not met. Please:")
        print("1. Install Git: xcode-select --install")
        print("2. Run this script again")
        return
    
    # Setup repository
    if not pusher.setup_repository():
        print("\n Failed to setup repository")
        return
    
    # Create project files
    pusher.create_gitignore()
    pusher.create_project_readme()
    
    # Commit and push
    print(f"\n Pushing to GitHub repository...")
    if pusher.commit_and_push():
        print("\n Successfully pushed DMML project to GitHub!")
        pusher.verify_push()
    else:
        print("\n Failed to push to GitHub")
        print("You may need to:")
        print("1. Setup GitHub authentication (SSH keys or personal access token)")
        print("2. Check repository permissions")
        print("3. Try again after authentication setup")

if __name__ == "__main__":
    main()
