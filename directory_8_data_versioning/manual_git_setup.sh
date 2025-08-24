#!/bin/bash

# Manual Git Setup Script for Data Versioning
# Students: 2024ab05134, 2024aa05664

echo "======================================"
echo "GIT-BASED DATA VERSIONING SETUP"
echo "Students: 2024ab05134, 2024aa05664"
echo "======================================"
echo

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed!"
    echo "Please install Xcode command line tools first:"
    echo "   xcode-select --install"
    echo
    echo "After installation, run this script again."
    exit 1
fi

echo "✅ Git is installed: $(git --version)"
echo

# Navigate to project directory
cd /Users/puneetsinha/DMML

# Initialize Git repository if not already done
if [ ! -d ".git" ]; then
    echo "🔧 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Configure Git user
echo "🔧 Configuring Git user..."
git config user.name "Students 2024ab05134 2024aa05664"
git config user.email "students@university.edu"
echo "✅ Git user configured"

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "🔧 Creating .gitignore file..."
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

# MLflow
mlruns/
EOF
    echo "✅ .gitignore file created"
else
    echo "✅ .gitignore file already exists"
fi

# Prompt for GitHub repository URL
echo
echo "🌐 GitHub Repository Setup"
echo "-------------------------"
echo "Please enter your GitHub repository URL:"
echo "Example: https://github.com/yourusername/DMML-ML-Pipeline.git"
read -p "GitHub URL (or press Enter to skip): " github_url

if [ ! -z "$github_url" ]; then
    echo "🔧 Adding GitHub remote..."
    
    # Remove existing origin if it exists
    git remote remove origin 2>/dev/null || true
    
    # Add new origin
    git remote add origin "$github_url"
    echo "✅ GitHub remote added: $github_url"
else
    echo "⏭️  Skipping GitHub remote setup"
fi

# Create initial commit and version
echo
echo "📦 Creating Initial Data Version"
echo "--------------------------------"
read -p "Create initial version? (y/n): " create_version

if [ "$create_version" = "y" ] || [ "$create_version" = "Y" ]; then
    echo "🔧 Adding all files to Git..."
    git add .
    
    echo "🔧 Creating initial commit..."
    git commit -m "v1.0-initial: Complete ML Pipeline Implementation

Students: 2024ab05134, 2024aa05664

This initial version includes:
- Complete end-to-end ML pipeline
- Data ingestion from Kaggle and HuggingFace
- Data validation and quality assessment
- Data preparation and feature engineering
- Feature store implementation
- Model training and evaluation
- Pipeline orchestration
- Comprehensive documentation

Pipeline Components:
- 2 datasets processed (39,604 total records)
- 139 features engineered
- 4 ML algorithms trained
- 100% pipeline success rate
- Production-ready implementation"
    
    echo "🔧 Creating version tag..."
    git tag -a v1.0-initial -m "Version 1.0: Complete ML Pipeline Implementation

Students: 2024ab05134, 2024aa05664
- End-to-end ML pipeline for customer churn prediction
- Data ingestion, validation, preparation, and transformation
- Feature engineering and centralized feature store
- Model training with 4 algorithms
- Pipeline orchestration and monitoring
- Comprehensive documentation and reports"
    
    echo "✅ Initial version created: v1.0-initial"
    
    # List current tags
    echo
    echo "📋 Current versions:"
    git tag -l
fi

# Push to GitHub if remote is configured
if git remote get-url origin &> /dev/null; then
    echo
    echo "🚀 GitHub Push Options"
    echo "---------------------"
    read -p "Push to GitHub now? (y/n): " push_now
    
    if [ "$push_now" = "y" ] || [ "$push_now" = "Y" ]; then
        echo "🔧 Pushing to GitHub..."
        
        # Push main branch
        if git push -u origin main 2>/dev/null; then
            echo "✅ Main branch pushed to GitHub"
        else
            echo "⚠️  Could not push main branch. You may need to setup authentication."
        fi
        
        # Push tags
        if git push origin --tags 2>/dev/null; then
            echo "✅ Version tags pushed to GitHub"
        else
            echo "⚠️  Could not push tags. You may need to setup authentication."
        fi
    else
        echo "⏭️  You can push later using:"
        echo "   git push -u origin main"
        echo "   git push origin --tags"
    fi
fi

echo
echo "🎉 Git-based data versioning setup complete!"
echo
echo "📖 Next Steps:"
echo "1. Check your GitHub repository to see the uploaded project"
echo "2. Use 'git tag' to create new versions as you make changes"
echo "3. Use 'git push origin --tags' to sync versions to GitHub"
echo
echo "📋 Useful Commands:"
echo "- List versions: git tag -l"
echo "- Create version: git tag -a v1.1-updated -m 'Description'"
echo "- Push to GitHub: git push origin main && git push origin --tags"
echo "- View changes: git log --oneline"
echo
echo "✅ Your ML pipeline is now version controlled with Git!"
