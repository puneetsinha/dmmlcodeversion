# Git-Based Data Versioning Guide

**Students: 2024ab05134, 2024aa05664**

## Overview

This guide shows how to use Git for data versioning in your ML pipeline project and push versions to GitHub for backup and collaboration.

## Prerequisites Setup

### 1. Install Git (Required)

First, install Xcode command line tools which includes Git:

```bash
xcode-select --install
```

After installation, verify Git is working:

```bash
git --version
```

### 2. Create GitHub Repository

1. Go to [GitHub.com](https://github.com)
2. Click "New repository"
3. Repository name: `DMML-ML-Pipeline`
4. Description: `End-to-End ML Pipeline for Customer Churn Prediction`
5. Choose Public or Private
6. **Don't** initialize with README (we have our own)
7. Click "Create repository"
8. Copy the repository URL (e.g., `https://github.com/yourusername/DMML-ML-Pipeline.git`)

## Setup Git Versioning

### 1. Initialize Git Repository

```bash
cd /Users/puneetsinha/DMML

# Initialize Git repository
git init

# Configure user information
git config user.name "Students 2024ab05134 2024aa05664"
git config user.email "students@university.edu"
```

### 2. Create .gitignore File

Create a `.gitignore` file to exclude unnecessary files:

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
.Python
build/
dist/
*.egg-info/

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.vscode/
.DS_Store

# Logs
*.log

# Virtual Environment
venv/
env/

# Comment out if you want to version large data files
# *.csv
# *.parquet
# *.xlsx
# *.db
EOF
```

### 3. Add GitHub Remote

```bash
# Add your GitHub repository as remote origin
git remote add origin https://github.com/yourusername/DMML-ML-Pipeline.git

# Verify remote is added
git remote -v
```

## Data Versioning Strategy

### Version Naming Convention

Use semantic versioning with descriptive tags:

- `v1.0-raw-data` - Original datasets from sources
- `v1.1-validated` - After data validation
- `v1.2-cleaned` - After data preparation/cleaning
- `v1.3-transformed` - After feature engineering
- `v1.4-feature-store` - After feature store implementation
- `v1.5-models` - After model training
- `v1.6-production` - Final production-ready version

### Create Data Versions

#### Version 1.0: Raw Data
```bash
# Add all files
git add .

# Commit with descriptive message
git commit -m "v1.0-raw-data: Initial datasets from Kaggle and HuggingFace

- Telco Customer Churn dataset (7,043 records)
- Adult Census Income dataset (32,561 records)
- Raw CSV files from data ingestion pipeline"

# Create annotated tag
git tag -a v1.0-raw-data -m "Version 1.0: Raw datasets from Kaggle and HuggingFace"
```

#### Version 1.1: Validated Data
```bash
git add .
git commit -m "v1.1-validated: Data validation and quality reports

- Data quality reports generated
- 95% overall quality score achieved
- Schema validation completed
- Missing values and duplicates analyzed"

git tag -a v1.1-validated -m "Version 1.1: Data validation and quality assessment"
```

#### Version 1.2: Cleaned Data
```bash
git add .
git commit -m "v1.2-cleaned: Data preparation and cleaning

- Missing values imputed
- Outliers handled
- Data types standardized
- EDA reports generated"

git tag -a v1.2-cleaned -m "Version 1.2: Data preparation and cleaning"
```

#### Version 1.3: Transformed Data
```bash
git add .
git commit -m "v1.3-transformed: Feature engineering completed

- 139 new features engineered across 2 datasets
- Normalization and encoding applied
- Advanced feature transformations implemented
- Feature metadata documented"

git tag -a v1.3-transformed -m "Version 1.3: Feature engineering and transformation"
```

#### Version 1.4: Feature Store
```bash
git add .
git commit -m "v1.4-feature-store: Feature store implementation

- 210 features registered in centralized store
- Feature metadata and lineage tracking
- Quality monitoring implemented
- Multiple feature views created"

git tag -a v1.4-feature-store -m "Version 1.4: Feature store implementation"
```

#### Version 1.5: Models
```bash
git add .
git commit -m "v1.5-models: Model training and evaluation

- 4 algorithms trained (Logistic Regression, Random Forest, XGBoost, Gradient Boosting)
- Hyperparameter tuning completed
- Model evaluation and comparison
- MLflow experiment tracking"

git tag -a v1.5-models -m "Version 1.5: Model training and evaluation"
```

#### Version 1.6: Production
```bash
git add .
git commit -m "v1.6-production: Production-ready ML pipeline

- Complete end-to-end pipeline orchestration
- All components integrated and tested
- Comprehensive documentation
- Assignment submission ready"

git tag -a v1.6-production -m "Version 1.6: Production-ready ML pipeline"
```

## Push to GitHub

### Push Repository and Tags

```bash
# Push main branch to GitHub
git push -u origin main

# Push all version tags
git push origin --tags
```

### Verify on GitHub

1. Go to your GitHub repository
2. Check the "Tags" section to see all versions
3. Browse different versions by clicking on tags
4. Each tag shows the state of the project at that version

## Useful Git Commands for Data Versioning

### List All Versions
```bash
git tag -l
```

### View Version Details
```bash
git show v1.3-transformed
```

### Switch to Specific Version
```bash
git checkout v1.2-cleaned
```

### Return to Latest Version
```bash
git checkout main
```

### Compare Versions
```bash
git diff v1.0-raw-data v1.3-transformed
```

### View Version History
```bash
git log --oneline --graph --decorate --all
```

## Automation Script

Run the automated setup script:

```bash
cd /Users/puneetsinha/DMML
python directory_8_data_versioning/git_versioning_setup.py
```

This script will:
- Check Git installation
- Initialize repository
- Create .gitignore
- Setup GitHub remote
- Create initial version
- Generate version summary

## Benefits of Git-Based Versioning

✅ **Complete History**: Full project evolution in one repository  
✅ **Branching**: Easy experimental feature development  
✅ **Collaboration**: Team members can work together  
✅ **Backup**: Automatic distributed backup on GitHub  
✅ **Rollback**: Easy return to any previous version  
✅ **Integration**: Works with GitHub's collaboration tools  
✅ **Tagging**: Meaningful version names and descriptions  

## Best Practices

1. **Commit Often**: Make small, logical commits
2. **Descriptive Messages**: Use clear commit messages
3. **Tag Major Milestones**: Create tags for important versions
4. **Document Changes**: Include what changed in each version
5. **Test Before Committing**: Ensure code works before committing
6. **Consistent Naming**: Use consistent tag naming convention

## Example Workflow

```bash
# 1. Make changes to your pipeline
# ... edit files ...

# 2. Check what changed
git status
git diff

# 3. Add changes
git add .

# 4. Commit with descriptive message
git commit -m "Improve feature engineering: add interaction features"

# 5. Create version tag if this is a milestone
git tag -a v1.3.1-enhanced -m "Enhanced feature engineering with interactions"

# 6. Push to GitHub
git push origin main
git push origin --tags
```

## Troubleshooting

### Git Not Found
- Install Xcode command line tools: `xcode-select --install`

### Permission Denied (GitHub)
- Set up SSH keys or use personal access token
- See GitHub documentation for authentication setup

### Large Files
- Use Git LFS for large data files
- Or exclude them in .gitignore and document separately

## Summary

This Git-based versioning system provides a robust way to:
- Track all changes to your ML pipeline
- Maintain multiple versions of your data and code
- Collaborate with team members
- Backup your work to GitHub
- Document your project evolution

Your ML pipeline now has professional-grade version control! 🚀
