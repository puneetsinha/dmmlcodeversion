# Data Versioning Component

**Students: 2024ab05134, 2024aa05664**

## Overview

This component implements Git-based data versioning for our ML pipeline project. Instead of using DVC, we leverage Git's native capabilities to maintain data versions and push them to GitHub for backup and collaboration.

## Why Git-Based Versioning?

✅ **Integrated Workflow**: Code and data versions in one repository  
✅ **GitHub Integration**: Leverage GitHub's collaboration features  
✅ **Familiar Tools**: Uses standard Git commands  
✅ **Free Hosting**: GitHub provides free repository hosting  
✅ **Team Collaboration**: Easy sharing and collaboration  
✅ **Branching Support**: Experimental data versions  

## Quick Start

### 1. Install Prerequisites

```bash
# Install Xcode command line tools (includes Git)
xcode-select --install

# Verify installation
git --version
```

### 2. Run Automated Setup

```bash
# Option A: Python script (interactive)
cd /Users/puneetsinha/DMML
python directory_8_data_versioning/git_versioning_setup.py

# Option B: Shell script (streamlined)
./directory_8_data_versioning/manual_git_setup.sh
```

### 3. Manual Setup (if needed)

```bash
cd /Users/puneetsinha/DMML

# Initialize Git repository
git init

# Configure user
git config user.name "Students 2024ab05134 2024aa05664"
git config user.email "students@university.edu"

# Add GitHub remote (replace with your URL)
git remote add origin https://github.com/yourusername/DMML-ML-Pipeline.git

# Create initial version
git add .
git commit -m "v1.0-initial: Complete ML Pipeline"
git tag -a v1.0-initial -m "Initial ML pipeline implementation"

# Push to GitHub
git push -u origin main
git push origin --tags
```

## Versioning Strategy

### Version Naming Convention

We use semantic versioning with descriptive tags:

| Version | Description | Contents |
|---------|-------------|----------|
| `v1.0-raw-data` | Original datasets | Raw CSV files from ingestion |
| `v1.1-validated` | After validation | Quality reports and validation |
| `v1.2-cleaned` | After preparation | Cleaned and preprocessed data |
| `v1.3-transformed` | After feature engineering | 139 engineered features |
| `v1.4-feature-store` | Feature store ready | 210 features in centralized store |
| `v1.5-models` | Models trained | 4 algorithms with evaluation |
| `v1.6-production` | Production ready | Complete pipeline |

### Creating New Versions

```bash
# Make your changes
# ... edit files ...

# Stage and commit changes
git add .
git commit -m "Descriptive commit message"

# Create version tag
git tag -a v1.7-enhanced -m "Enhanced feature engineering"

# Push to GitHub
git push origin main
git push origin --tags
```

## Repository Structure on GitHub

```
DMML-ML-Pipeline/
├── directory_2_data_ingestion/     # Data ingestion scripts
├── directory_3_raw_data_storage/   # Data lake organization
├── directory_4_data_validation/    # Quality validation
├── directory_5_data_preparation/   # Data cleaning and EDA
├── directory_6_data_transformation/ # Feature engineering
├── directory_7_feature_store/      # Centralized features
├── directory_8_data_versioning/    # This component
├── directory_9_model_building/     # ML model training
├── directory_10_pipeline_orchestration/ # Pipeline automation
├── raw_data/                       # Original datasets
├── processed_data/                 # Cleaned data
├── transformed_data/               # Engineered features
├── feature_store/                  # Feature database
├── models/                         # Trained models
├── validation_reports/             # Quality reports
├── README.md                       # Project documentation
└── requirements.txt                # Dependencies
```

## Key Commands

### View Versions
```bash
# List all versions
git tag -l

# View version details
git show v1.3-transformed

# See version history
git log --oneline --graph --decorate
```

### Switch Versions
```bash
# Go to specific version
git checkout v1.2-cleaned

# Return to latest
git checkout main
```

### Compare Versions
```bash
# See differences between versions
git diff v1.0-raw-data v1.3-transformed

# See files changed
git diff --name-only v1.0-raw-data v1.3-transformed
```

### Collaborate
```bash
# Get latest changes from team
git pull origin main

# Share your changes
git push origin main
git push origin --tags
```

## GitHub Features

### Release Management
- Create releases from tags
- Attach binary files to releases
- Release notes and changelogs

### Collaboration
- Pull requests for code review
- Issues for tracking problems
- Project boards for task management
- Wiki for documentation

### Backup and Security
- Automatic backup in the cloud
- Access control and permissions
- Branch protection rules
- Security scanning

## Benefits for ML Projects

### 1. **Complete Lineage**
Track how data and models evolved through each pipeline stage.

### 2. **Experiment Management**
Use branches to test different feature engineering approaches.

### 3. **Reproducibility**
Anyone can checkout a specific version and reproduce results.

### 4. **Collaboration**
Team members can work on different components simultaneously.

### 5. **Documentation**
Commit messages and tags provide context for each change.

### 6. **Integration**
Works seamlessly with CI/CD pipelines and deployment tools.

## Files in This Component

| File | Purpose |
|------|---------|
| `git_versioning_setup.py` | Automated setup script |
| `manual_git_setup.sh` | Shell script for quick setup |
| `GIT_VERSIONING_GUIDE.md` | Comprehensive setup guide |
| `README.md` | This documentation |
| `data_version_manager.py` | Legacy DVC implementation |
| `versioning_config.py` | Configuration settings |

## Best Practices

### 1. **Commit Frequently**
Make small, logical commits with clear messages.

### 2. **Tag Milestones**
Create tags for important pipeline stages.

### 3. **Document Changes**
Use descriptive commit messages and tag annotations.

### 4. **Test Before Committing**
Ensure your pipeline works before creating versions.

### 5. **Consistent Naming**
Follow the version naming convention.

### 6. **Backup Regularly**
Push to GitHub frequently to avoid data loss.

## Troubleshooting

### Git Not Installed
```bash
# Install Xcode command line tools
xcode-select --install
```

### Large Files
```bash
# Use Git LFS for files > 100MB
git lfs track "*.csv"
git lfs track "*.parquet"
```

### Authentication Issues
- Set up SSH keys or personal access tokens
- See GitHub's authentication documentation

### Permission Denied
```bash
# Check remote URL
git remote -v

# Update remote URL if needed
git remote set-url origin https://github.com/username/repo.git
```

## Future Enhancements

1. **Git LFS Integration**: Handle large data files efficiently
2. **Automated Tagging**: Script to auto-create tags after pipeline runs
3. **Branch Strategies**: Use feature branches for experiments
4. **CI/CD Integration**: Automated testing on version changes
5. **Metadata Tracking**: Enhanced metadata in commit messages

## Summary

This Git-based versioning system provides:
- ✅ Complete project history
- ✅ Easy collaboration with GitHub
- ✅ Professional version control
- ✅ Integrated documentation
- ✅ Reliable backup system

Your ML pipeline now has enterprise-grade version control! 🚀
