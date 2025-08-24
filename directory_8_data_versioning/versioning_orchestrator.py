"""
Data Versioning Orchestrator.
Manages the complete data versioning pipeline with DVC and Git.
"""

import os
import json
import sys
from datetime import datetime
from typing import Dict, List

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_version_manager import DataVersionManager
from versioning_config import versioning_config


class VersioningOrchestrator:
    """Orchestrate data versioning pipeline."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.config = versioning_config
        self.version_manager = DataVersionManager()
        self.orchestration_results = {}
        
    def setup_data_versioning(self) -> Dict:
        """
        Set up complete data versioning system.
        
        Returns:
            Setup results
        """
        print(" Setting Up Data Versioning System")
        print("=" * 60)
        
        setup_results = {
            "setup_timestamp": datetime.now().isoformat(),
            "steps_completed": [],
            "status": "SUCCESS",
            "errors": []
        }
        
        try:
            # Step 1: Initialize versioning system
            print("\n Step 1: Initialize DVC and Git")
            init_success = self.version_manager.initialize_versioning()
            
            if init_success:
                setup_results["steps_completed"].append("versioning_initialization")
                print(" Versioning system initialized")
            else:
                setup_results["errors"].append("Failed to initialize versioning system")
                setup_results["status"] = "PARTIAL_FAILURE"
            
            # Step 2: Create initial data version
            print("\n Step 2: Create Initial Data Version")
            initial_version = self.version_manager.create_data_version(
                description="Initial data version - Complete ML pipeline setup",
                tag="data"
            )
            
            if initial_version:
                setup_results["steps_completed"].append("initial_version_creation")
                setup_results["initial_version"] = initial_version
                print(f" Initial version created: {initial_version}")
            else:
                setup_results["errors"].append("Failed to create initial version")
                setup_results["status"] = "PARTIAL_FAILURE"
            
            # Step 3: Create feature store version
            print("\n Step 3: Create Feature Store Version")
            feature_version = self.version_manager.create_data_version(
                description="Feature store with 204 engineered features",
                tag="feature"
            )
            
            if feature_version:
                setup_results["steps_completed"].append("feature_version_creation")
                setup_results["feature_version"] = feature_version
                print(f" Feature store version created: {feature_version}")
            else:
                setup_results["errors"].append("Failed to create feature store version")
                setup_results["status"] = "PARTIAL_FAILURE"
            
            # Step 4: Generate version report
            print("\n Step 4: Generate Version Report")
            version_report = self.version_manager.generate_version_report()
            
            if version_report and "error" not in version_report:
                setup_results["steps_completed"].append("version_report_generation")
                setup_results["version_report"] = version_report
                print(" Version report generated")
            else:
                setup_results["errors"].append("Failed to generate version report")
                setup_results["status"] = "PARTIAL_FAILURE"
            
            # Step 5: Create documentation
            print("\n Step 5: Create Versioning Documentation")
            docs_created = self._create_versioning_documentation(setup_results)
            
            if docs_created:
                setup_results["steps_completed"].append("documentation_creation")
                print(" Versioning documentation created")
            else:
                setup_results["errors"].append("Failed to create documentation")
                setup_results["status"] = "PARTIAL_FAILURE"
            
        except Exception as e:
            setup_results["status"] = "FAILURE"
            setup_results["errors"].append(f"Unexpected error: {str(e)}")
            print(f" Setup failed: {str(e)}")
        
        return setup_results
    
    def _create_versioning_documentation(self, setup_results: Dict) -> bool:
        """Create comprehensive versioning documentation."""
        try:
            docs_dir = os.path.join(self.config.BASE_DIR, "version_metadata")
            
            # Create versioning guide
            guide_content = f"""# Data Versioning Guide

## Overview

This project uses DVC (Data Version Control) and Git for comprehensive data and code versioning.

## Setup Summary

- **Setup Date**: {setup_results['setup_timestamp']}
- **Status**: {setup_results['status']}
- **Steps Completed**: {len(setup_results['steps_completed'])}

## Versioning Strategy

### Version Tags
- `data`: Raw and processed data versions
- `feature`: Feature store and engineered features
- `model`: Trained model versions
- `experiment`: Experimental data and models

### Directory Structure
```
.
 .dvc/                 # DVC configuration and cache
 .git/                 # Git repository
 version_metadata/     # Version metadata and reports
 raw_data/            # Raw data (DVC tracked)
 processed_data/      # Cleaned data (DVC tracked)
 transformed_data/    # Feature engineered data (DVC tracked)
 feature_store/       # Feature store (DVC tracked)
 models/              # Trained models (DVC tracked)
```

## Common Commands

### Create New Data Version
```bash
# Through Python API
python -c "
from directory_8_data_versioning.data_version_manager import DataVersionManager
manager = DataVersionManager()
manager.create_data_version('Description of changes', 'data')
"
```

### List All Versions
```bash
# Through Python API
python -c "
from directory_8_data_versioning.data_version_manager import DataVersionManager
manager = DataVersionManager()
versions = manager.list_versions()
for v in versions:
    print(f\\"{{v['version_id']}}: {{v['description']}}\\")
"
```

### Checkout Specific Version
```bash
# Through Python API
python -c "
from directory_8_data_versioning.data_version_manager import DataVersionManager
manager = DataVersionManager()
manager.checkout_version('v2025.8.23.2030')  # Replace with actual version
"
```

### Manual DVC Commands
```bash
# Add new data to tracking
dvc add data_directory/

# Commit changes
git add data_directory.dvc .gitignore
git commit -m "Update data"

# Push data to remote storage
dvc push

# Pull latest data
dvc pull

# Check status
dvc status
git status
```

## Best Practices

1. **Regular Versioning**: Create versions at key pipeline milestones
2. **Descriptive Messages**: Use clear, descriptive version descriptions
3. **Tag Organization**: Use appropriate tags for different types of changes
4. **Documentation**: Update this guide when making significant changes
5. **Remote Storage**: Ensure DVC remote storage is properly configured

## Troubleshooting

### Common Issues

1. **DVC Authentication**: Ensure proper permissions for remote storage
2. **Large Files**: Use DVC for files >100MB, Git for smaller files
3. **Merge Conflicts**: Use `dvc checkout` after resolving Git conflicts
4. **Storage Space**: Monitor DVC cache size and clean old versions periodically

### Recovery Commands
```bash
# Reset to last committed state
dvc checkout
git checkout .

# Force update from remote
dvc pull --force

# Clean DVC cache
dvc cache dir --unset
dvc cache dir /path/to/new/cache
```

## Version History

Generated automatically. See version_metadata/metadata.json for complete history.

---
*This documentation is automatically generated and updated with each version.*
"""
            
            guide_file = os.path.join(docs_dir, "VERSIONING_GUIDE.md")
            with open(guide_file, 'w') as f:
                f.write(guide_content)
            
            # Create quick reference
            reference_content = """# Quick Reference - Data Versioning

## Current Setup Status
"""
            
            if setup_results.get("initial_version"):
                reference_content += f"- Initial Version: {setup_results['initial_version']}\n"
            
            if setup_results.get("feature_version"):
                reference_content += f"- Feature Store Version: {setup_results['feature_version']}\n"
            
            reference_content += f"""
## Data Directories Tracked by DVC
"""
            
            for data_dir in self.config.DATA_DIRECTORIES:
                data_path = os.path.join(self.config.BASE_DIR, data_dir)
                if os.path.exists(data_path):
                    reference_content += f"-  {data_dir}/\n"
                else:
                    reference_content += f"-  {data_dir}/ (not found)\n"
            
            reference_content += f"""
## Quick Commands

```bash
# Create version
python directory_8_data_versioning/versioning_orchestrator.py

# Check status
dvc status
git status

# View versions
git tag -l

# Sync data
dvc pull
dvc push
```

## Support

For issues with data versioning, check:
1. DVC documentation: https://dvc.org/doc
2. Git documentation: https://git-scm.com/doc
3. Version metadata in version_metadata/ directory
"""
            
            reference_file = os.path.join(docs_dir, "QUICK_REFERENCE.md")
            with open(reference_file, 'w') as f:
                f.write(reference_content)
            
            print(f" Documentation created:")
            print(f"    Guide: {guide_file}")
            print(f"    Reference: {reference_file}")
            
            return True
            
        except Exception as e:
            print(f" Documentation creation failed: {str(e)}")
            return False
    
    def run_complete_versioning_setup(self) -> Dict:
        """
        Run complete data versioning setup pipeline.
        
        Returns:
            Complete setup results
        """
        print(" Starting Complete Data Versioning Setup")
        print("=" * 80)
        
        # Run setup
        setup_results = self.setup_data_versioning()
        
        # Generate final summary
        final_summary = {
            "versioning_pipeline_summary": {
                "timestamp": datetime.now().isoformat(),
                "setup_status": setup_results["status"],
                "steps_completed": len(setup_results["steps_completed"]),
                "total_steps": 5,
                "success_rate": round(len(setup_results["steps_completed"]) / 5 * 100, 1),
                "versions_created": 0
            },
            "setup_details": setup_results,
            "recommendations": self._generate_setup_recommendations(setup_results),
            "next_steps": self._generate_next_steps(setup_results)
        }
        
        # Count versions created
        versions_created = 0
        if setup_results.get("initial_version"):
            versions_created += 1
        if setup_results.get("feature_version"):
            versions_created += 1
        
        final_summary["versioning_pipeline_summary"]["versions_created"] = versions_created
        
        # Save final summary
        summary_file = os.path.join(
            self.config.VERSIONING_DIR,
            f"versioning_setup_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(summary_file, 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)
        
        # Print final summary
        self._print_final_summary(final_summary)
        
        print(f"\n Complete summary saved: {summary_file}")
        print(f" Data versioning setup completed!")
        
        return final_summary
    
    def _generate_setup_recommendations(self, setup_results: Dict) -> List[str]:
        """Generate recommendations based on setup results."""
        recommendations = []
        
        if setup_results["status"] == "SUCCESS":
            recommendations.extend([
                "Excellent! Data versioning system is fully operational",
                "Create regular data versions as your pipeline evolves",
                "Use appropriate tags (data/feature/model/experiment) for better organization"
            ])
        elif setup_results["status"] == "PARTIAL_FAILURE":
            recommendations.extend([
                "Some setup steps failed - review error messages and retry",
                "Ensure Git and DVC are properly installed and configured",
                "Check file permissions and disk space availability"
            ])
        else:
            recommendations.extend([
                "Setup failed - check error messages and system requirements",
                "Verify Git and DVC installation",
                "Ensure proper write permissions in project directory"
            ])
        
        # Specific recommendations based on completed steps
        completed_steps = setup_results.get("steps_completed", [])
        
        if "versioning_initialization" in completed_steps:
            recommendations.append("DVC and Git initialized - ready for data tracking")
        
        if "initial_version_creation" in completed_steps:
            recommendations.append("Initial version created - establishes baseline for future changes")
        
        if "feature_version_creation" in completed_steps:
            recommendations.append("Feature store versioned - enables reproducible feature engineering")
        
        return recommendations
    
    def _generate_next_steps(self, setup_results: Dict) -> List[str]:
        """Generate next steps based on setup results."""
        next_steps = []
        
        if setup_results["status"] in ["SUCCESS", "PARTIAL_FAILURE"]:
            next_steps.extend([
                "Proceed to model building using versioned features",
                "Set up automated versioning in CI/CD pipeline",
                "Configure DVC remote storage for team collaboration",
                "Establish versioning policies and schedules"
            ])
        
        if setup_results.get("errors"):
            next_steps.extend([
                "Address any setup errors before proceeding",
                "Verify system requirements and permissions",
                "Consult DVC and Git documentation for troubleshooting"
            ])
        
        next_steps.extend([
            "Review versioning documentation in version_metadata/",
            "Test version checkout and data recovery procedures",
            "Train team members on versioning workflows"
        ])
        
        return next_steps
    
    def _print_final_summary(self, summary: Dict):
        """Print final versioning setup summary."""
        print(f"\n{'='*80}")
        print(" DATA VERSIONING SETUP SUMMARY")
        print("=" * 80)
        
        pipeline_summary = summary["versioning_pipeline_summary"]
        print(f"Setup Status: {pipeline_summary['setup_status']}")
        print(f"Steps Completed: {pipeline_summary['steps_completed']}/{pipeline_summary['total_steps']}")
        print(f"Success Rate: {pipeline_summary['success_rate']}%")
        print(f"Versions Created: {pipeline_summary['versions_created']}")
        
        # Show errors if any
        setup_details = summary["setup_details"]
        if setup_details.get("errors"):
            print(f"\n ERRORS ENCOUNTERED:")
            for error in setup_details["errors"]:
                print(f"  • {error}")
        
        print(f"\n RECOMMENDATIONS:")
        for i, rec in enumerate(summary["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n NEXT STEPS:")
        for i, step in enumerate(summary["next_steps"], 1):
            print(f"  {i}. {step}")
        
        print(f"\n VERSIONING RESOURCES:")
        print(f"   Metadata: {self.config.BASE_DIR}/version_metadata/")
        print(f"   Documentation: {self.config.BASE_DIR}/version_metadata/VERSIONING_GUIDE.md")
        print(f"   DVC Config: {self.config.BASE_DIR}/.dvc/")
        print(f"   Git Repository: {self.config.BASE_DIR}/.git/")


def main():
    """Main function to run data versioning setup."""
    orchestrator = VersioningOrchestrator()
    results = orchestrator.run_complete_versioning_setup()
    return results


if __name__ == "__main__":
    main()
