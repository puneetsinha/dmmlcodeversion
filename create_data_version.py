#!/usr/bin/env python3
"""
Simple Data Versioning Script
Students: 2024ab05134, 2024aa05664

This script provides a simple interface to create and upload data versions to Git.
Just run this script whenever you want to version your data changes!
"""

import sys
import os
from datetime import datetime

# Add the data versioning directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'directory_8_data_versioning'))

from data_version_manager import DataVersionManager

def main():
    """Main function to create a data version."""
    print("DATA VERSIONING TOOL")
    print("=" * 50)
    print("Students: 2024ab05134, 2024aa05664")
    print()
    
    # Initialize manager
    manager = DataVersionManager()
    
    # Show current status
    print("Current Status:")
    manager.show_version_status()
    
    print("\n" + "="*50)
    print("CREATE NEW DATA VERSION")
    print("="*50)
    
    # Get version name from user
    print("\nEnter version details:")
    version_name = input("Version name (e.g., v1.1-feature-engineering): ").strip()
    
    if not version_name:
        # Auto-generate version name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        version_name = f"v1.0-auto-{timestamp}"
        print(f"Auto-generated version name: {version_name}")
    
    description = input("Description (optional): ").strip()
    if not description:
        description = "Automated data version"
    
    print(f"\nCreating version: {version_name}")
    print(f"Description: {description}")
    
    # Create the version
    success = manager.create_data_version(version_name, description)
    
    if success:
        print("\nData version created successfully!")
        
        # Generate summary report
        report_file = manager.create_version_summary_report()
        
        # Ask about GitHub upload
        print("\nUPLOAD TO GITHUB")
        print("="*30)
        upload_choice = input("Upload this version to GitHub? (y/n): ").strip().lower()
        
        if upload_choice in ['y', 'yes']:
            print("\nUploading to GitHub...")
            upload_success = manager.upload_to_github()
            
            if upload_success:
                print("\nSUCCESS! Your data version is now live on GitHub!")
                print(f"Check your repository: https://github.com/puneetsinha/dmmlcodeversion")
                print(f"Version tag: {version_name}")
            else:
                print("\nUpload failed. Version created locally but not pushed to GitHub.")
                print("You can manually push later using:")
                print("   git push origin main")
                print("   git push origin --tags")
        else:
            print("\nVersion created locally. To upload later, run:")
            print("   git push origin main")
            print("   git push origin --tags")
    
    else:
        print("\nFailed to create data version. Please check the errors above.")
    
    print("\n" + "="*50)
    print("QUICK COMMANDS FOR FUTURE USE:")
    print("="*50)
    print("1. Create version: python create_data_version.py")
    print("2. Check status:  python -c \"from directory_8_data_versioning.data_version_manager import DataVersionManager; DataVersionManager().show_version_status()\"")
    print("3. List versions: git tag -l")
    print("4. Push to GitHub: git push origin --tags")

if __name__ == "__main__":
    main()
