"""
Setup script for data ingestion module.
Handles installation of dependencies and initial configuration.
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def install_dependencies():
    """Install required Python packages."""
    print("Installing required packages...")
    
    packages = [
        "pandas>=1.5.0",
        "numpy>=1.24.0", 
        "kaggle>=1.5.16",
        "datasets>=2.14.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f" Installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f" Failed to install {package}: {e}")
            return False
    
    return True


def setup_kaggle_credentials():
    """Guide user through Kaggle API setup."""
    print("\n" + "="*50)
    print("KAGGLE API SETUP")
    print("="*50)
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        print(" Kaggle credentials already configured!")
        return True
    
    print("Kaggle credentials not found. Please follow these steps:")
    print("\n1. Go to https://www.kaggle.com/account")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Download the kaggle.json file")
    print("5. Place it in:", kaggle_dir)
    
    # Create .kaggle directory
    kaggle_dir.mkdir(exist_ok=True)
    
    # Prompt user to place the file
    input("\nPress Enter after you've placed kaggle.json in the .kaggle directory...")
    
    if kaggle_json.exists():
        # Set proper permissions
        os.chmod(kaggle_json, 0o600)
        print(" Kaggle credentials configured successfully!")
        return True
    else:
        print(" kaggle.json not found. Please try again.")
        return False


def verify_setup():
    """Verify that the setup is working correctly."""
    print("\n" + "="*50)
    print("VERIFYING SETUP")
    print("="*50)
    
    try:
        # Test imports
        import pandas as pd
        import kaggle
        import datasets
        print(" All required packages imported successfully")
        
        # Test Kaggle API
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(" Kaggle API authentication successful")
        
        # Test Hugging Face datasets
        from datasets import load_dataset
        print(" Hugging Face datasets library ready")
        
        return True
        
    except Exception as e:
        print(f" Setup verification failed: {e}")
        return False


def main():
    """Main setup function."""
    print(" Setting up Data Ingestion Module")
    print("="*50)
    
    # Install dependencies
    if not install_dependencies():
        print(" Failed to install dependencies")
        return False
    
    # Setup Kaggle credentials
    if not setup_kaggle_credentials():
        print(" Failed to setup Kaggle credentials")
        return False
    
    # Verify setup
    if not verify_setup():
        print(" Setup verification failed")
        return False
    
    print("\n Data ingestion module setup completed successfully!")
    print("\nYou can now run: python main_ingestion.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
