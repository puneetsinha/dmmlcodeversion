"""
Complete ML Pipeline Orchestrator.
Students: 2024ab05134, 2024aa05664

This is the main orchestrator that coordinates our entire ML pipeline from
start to finish. This was definitely the most challenging part of the project!

We implemented several advanced concepts here:
- Dependency management: Making sure steps run in the right order
- Error handling: Retry logic with exponential backoff
- Resource monitoring: Tracking memory usage and execution time
- Comprehensive logging: Detailed logs for debugging

The hardest part was figuring out how to handle failures gracefully. We learned
that in production systems, things can and will go wrong, so having robust
error handling is essential.

We're particularly proud of the automatic dependency resolution - it figures
out the correct order to run pipeline steps based on their dependancies.
"""

import os
import sys
import json
import time
import importlib
import traceback
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import threading
import psutil

# Add base directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_config import pipeline_config


class PipelineOrchestrator:
    """Orchestrate complete ML pipeline execution."""
    
    def __init__(self):
        """Initialize the pipeline orchestrator."""
        self.config = pipeline_config
        self.execution_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.pipeline_start_time = None
        self.pipeline_end_time = None
        self.step_results = {}
        self.execution_metrics = {}
        
        # Setup logging
        self._setup_logging()
        
        # Initialize monitoring
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def _setup_logging(self):
        """Setup comprehensive logging for pipeline execution."""
        log_file = os.path.join(
            self.config.PIPELINE_LOGS_DIR,
            f"{self.execution_id}.log"
        )
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(f"PipelineOrchestrator_{self.execution_id}")
        self.logger.setLevel(getattr(logging, self.config.EXECUTION_CONFIG["log_level"]))
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Pipeline orchestrator initialized: {self.execution_id}")
    
    def _validate_dependencies(self) -> bool:
        """
        Validate that pipeline step dependencies are correctly configured.
        
        This function checks that all the dependencies we specified actually exist.
        We learned the hard way that typos in dependency names can cause the
        whole pipeline to fail in confusing ways!
        """
        step_ids = {step["step_id"] for step in self.config.PIPELINE_STEPS}
        
        for step in self.config.PIPELINE_STEPS:
            for dep in step["dependencies"]:
                if dep not in step_ids:
                    self.logger.error(f"Invalid dependency '{dep}' for step '{step['step_id']}'")
                    return False
        
        self.logger.info("Pipeline dependencies validated successfully")
        return True
    
    def _get_execution_order(self) -> List[str]:
        """Determine the correct execution order based on dependencies."""
        steps = {step["step_id"]: step for step in self.config.PIPELINE_STEPS}
        executed = set()
        execution_order = []
        
        def can_execute(step_id: str) -> bool:
            step = steps[step_id]
            return all(dep in executed for dep in step["dependencies"])
        
        while len(executed) < len(steps):
            available_steps = [
                step_id for step_id in steps 
                if step_id not in executed and can_execute(step_id) and steps[step_id]["enabled"]
            ]
            
            if not available_steps:
                # Check for circular dependencies or missing dependencies
                remaining = set(steps.keys()) - executed
                self.logger.error(f"Cannot resolve dependencies for remaining steps: {remaining}")
                break
            
            # Add available steps to execution order
            for step_id in available_steps:
                execution_order.append(step_id)
                executed.add(step_id)
        
        self.logger.info(f"Execution order determined: {execution_order}")
        return execution_order
    
    def _import_and_execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Import module and execute a pipeline step."""
        step_id = step["step_id"]
        module_path = step["module_path"]
        function_name = step["function"]
        
        step_start_time = time.time()
        step_result = {
            "step_id": step_id,
            "name": step["name"],
            "status": "FAILED",
            "start_time": datetime.now().isoformat(),
            "error": None,
            "execution_time": 0,
            "memory_usage_mb": 0,
            "retry_count": 0
        }
        
        try:
            self.logger.info(f"Importing module: {module_path}")
            
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the function
            if hasattr(module, function_name):
                target_function = getattr(module, function_name)
            else:
                raise AttributeError(f"Function '{function_name}' not found in module '{module_path}'")
            
            # Execute the function
            self.logger.info(f"Executing step: {step_id}")
            
            # Capture memory before execution
            memory_before = self.process.memory_info().rss / 1024 / 1024
            
            # Execute with timeout (simulated - actual timeout would need more complex implementation)
            result = target_function()
            
            # Capture memory after execution
            memory_after = self.process.memory_info().rss / 1024 / 1024
            
            step_result.update({
                "status": "SUCCESS",
                "result": result if isinstance(result, (dict, list, str, int, float)) else "Execution completed",
                "execution_time": time.time() - step_start_time,
                "memory_usage_mb": memory_after - memory_before,
                "end_time": datetime.now().isoformat()
            })
            
            self.logger.info(f"Step completed successfully: {step_id}")
            
        except Exception as e:
            error_msg = f"Step failed: {step_id} - {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            step_result.update({
                "status": "FAILED",
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "execution_time": time.time() - step_start_time,
                "end_time": datetime.now().isoformat()
            })
        
        return step_result
    
    def _execute_step_with_retry(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a step with retry logic."""
        step_id = step["step_id"]
        max_retries = step.get("retry_count", 1)
        
        for attempt in range(max_retries + 1):
            self.logger.info(f"Executing step '{step_id}' (attempt {attempt + 1}/{max_retries + 1})")
            
            result = self._import_and_execute_step(step)
            result["retry_count"] = attempt
            
            if result["status"] == "SUCCESS":
                return result
            
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.info(f"Step failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                self.logger.error(f"Step '{step_id}' failed after {max_retries + 1} attempts")
        
        return result
    
    def _save_step_result(self, step_result: Dict[str, Any]):
        """Save individual step result to file."""
        step_id = step_result["step_id"]
        step_log_file = os.path.join(
            self.config.PIPELINE_LOGS_DIR,
            "step_logs",
            f"{self.execution_id}_{step_id}.json"
        )
        
        with open(step_log_file, 'w') as f:
            json.dump(step_result, f, indent=2, default=str)
        
        self.logger.info(f"Step result saved: {step_log_file}")
    
    def _should_continue_after_failure(self, step: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Determine if pipeline should continue after a step failure."""
        if result["status"] == "SUCCESS":
            return True
        
        # Always stop on critical step failure
        if step.get("critical", True):
            self.logger.error(f"Critical step failed: {step['step_id']}")
            return False
        
        # Check configuration
        if self.config.EXECUTION_CONFIG["continue_on_non_critical_failure"]:
            self.logger.warning(f"Non-critical step failed, continuing: {step['step_id']}")
            return True
        
        return False
    
    def execute_pipeline(self, steps_to_run: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute the complete pipeline or specified steps.
        
        Args:
            steps_to_run: Optional list of step IDs to run. If None, runs all enabled steps.
            
        Returns:
            Complete execution results
        """
        self.pipeline_start_time = datetime.now()
        
        print("Starting Complete ML Pipeline Execution")
        print("=" * 80)
        print(f"Execution ID: {self.execution_id}")
        print(f"Start Time: {self.pipeline_start_time.isoformat()}")
        
        self.logger.info(f"Pipeline execution started: {self.execution_id}")
        
        # Validate dependencies
        if not self._validate_dependencies():
            return {"status": "FAILED", "error": "Dependency validation failed"}
        
        # Get execution order
        execution_order = self._get_execution_order()
        
        # Filter steps if specified
        if steps_to_run:
            execution_order = [step_id for step_id in execution_order if step_id in steps_to_run]
            self.logger.info(f"Filtered execution order: {execution_order}")
        
        # Get step configurations
        steps_by_id = {step["step_id"]: step for step in self.config.PIPELINE_STEPS}
        
        # Execute steps
        successful_steps = 0
        failed_steps = 0
        
        for step_id in execution_order:
            step = steps_by_id[step_id]
            
            print(f"\n{'='*80}")
            print(f" EXECUTING STEP: {step['name']} ({step_id})")
            print(f"{'='*80}")
            print(f" Description: {step['description']}")
            
            # Execute step
            step_result = self._execute_step_with_retry(step)
            
            # Save result
            self.step_results[step_id] = step_result
            self._save_step_result(step_result)
            
            # Update counters
            if step_result["status"] == "SUCCESS":
                successful_steps += 1
                print(f" Step completed successfully: {step['name']}")
            else:
                failed_steps += 1
                print(f" Step failed: {step['name']}")
            
            # Check if should continue
            if not self._should_continue_after_failure(step, step_result):
                self.logger.error("Pipeline execution stopped due to critical step failure")
                break
        
        self.pipeline_end_time = datetime.now()
        
        # Generate final results
        final_results = self._generate_final_results(successful_steps, failed_steps)
        
        # Save complete results
        self._save_pipeline_results(final_results)
        
        # Print final summary
        self._print_pipeline_summary(final_results)
        
        return final_results
    
    def _generate_final_results(self, successful_steps: int, failed_steps: int) -> Dict[str, Any]:
        """Generate comprehensive final results."""
        total_execution_time = (self.pipeline_end_time - self.pipeline_start_time).total_seconds()
        current_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Calculate overall status
        overall_status = "SUCCESS"
        if failed_steps > 0:
            critical_failures = sum(
                1 for result in self.step_results.values()
                if result["status"] == "FAILED" and 
                any(step["step_id"] == result["step_id"] and step.get("critical", True) 
                    for step in self.config.PIPELINE_STEPS)
            )
            overall_status = "CRITICAL_FAILURE" if critical_failures > 0 else "PARTIAL_SUCCESS"
        
        results = {
            "execution_id": self.execution_id,
            "overall_status": overall_status,
            "start_time": self.pipeline_start_time.isoformat(),
            "end_time": self.pipeline_end_time.isoformat(),
            "total_execution_time_seconds": total_execution_time,
            "total_execution_time_formatted": str(timedelta(seconds=int(total_execution_time))),
            "pipeline_metrics": {
                "total_steps": len(self.step_results),
                "successful_steps": successful_steps,
                "failed_steps": failed_steps,
                "success_rate": round(successful_steps / len(self.step_results) * 100, 1) if self.step_results else 0,
                "total_memory_usage_mb": current_memory - self.initial_memory,
                "peak_memory_mb": current_memory
            },
            "step_results": self.step_results,
            "recommendations": self._generate_recommendations(),
            "next_steps": self._generate_next_steps()
        }
        
        return results
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on execution results."""
        recommendations = []
        
        if not self.step_results:
            return ["No steps executed"]
        
        success_rate = sum(1 for r in self.step_results.values() if r["status"] == "SUCCESS") / len(self.step_results)
        
        if success_rate == 1.0:
            recommendations.extend([
                " Perfect execution! All pipeline steps completed successfully",
                "Pipeline is ready for production deployment",
                "Consider setting up automated scheduling for regular execution"
            ])
        elif success_rate >= 0.8:
            recommendations.extend([
                "Good execution with most steps successful",
                "Review and fix failed non-critical steps for optimal performance"
            ])
        else:
            recommendations.extend([
                "Multiple step failures detected - review configuration and dependencies",
                "Check logs for specific error messages and resolution steps"
            ])
        
        # Performance recommendations
        total_time = sum(r.get("execution_time", 0) for r in self.step_results.values())
        if total_time > 7200:  # 2 hours
            recommendations.append("Consider optimizing long-running steps or implementing parallel execution")
        
        # Memory recommendations
        total_memory = sum(r.get("memory_usage_mb", 0) for r in self.step_results.values())
        if total_memory > 1000:  # 1GB
            recommendations.append("High memory usage detected - consider data chunking or streaming approaches")
        
        return recommendations
    
    def _generate_next_steps(self) -> List[str]:
        """Generate next steps based on execution results."""
        next_steps = []
        
        failed_steps = [r for r in self.step_results.values() if r["status"] == "FAILED"]
        
        if failed_steps:
            next_steps.extend([
                "Review failed step logs and error messages",
                "Fix configuration issues and retry failed steps",
                "Update pipeline dependencies if needed"
            ])
        else:
            next_steps.extend([
                "Set up model monitoring and alerting",
                "Configure automated retraining schedule",
                "Deploy models to production environment",
                "Implement model A/B testing framework"
            ])
        
        next_steps.extend([
            "Review pipeline performance metrics and optimize bottlenecks",
            "Set up automated pipeline scheduling",
            "Implement additional data quality checks"
        ])
        
        return next_steps
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save complete pipeline results."""
        results_file = os.path.join(
            self.config.PIPELINE_REPORTS_DIR,
            "execution_reports",
            f"{self.execution_id}_complete_results.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Complete pipeline results saved: {results_file}")
    
    def _print_pipeline_summary(self, results: Dict[str, Any]):
        """Print comprehensive pipeline execution summary."""
        print(f"\n{'='*80}")
        print(" PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        
        metrics = results["pipeline_metrics"]
        print(f" Execution ID: {results['execution_id']}")
        print(f"⏱  Total Time: {results['total_execution_time_formatted']}")
        print(f" Overall Status: {results['overall_status']}")
        print(f" Success Rate: {metrics['success_rate']}%")
        print(f" Steps: {metrics['successful_steps']}/{metrics['total_steps']} successful")
        print(f" Memory Usage: {metrics['total_memory_usage_mb']:.1f} MB")
        
        # Step-by-step summary
        print(f"\n STEP EXECUTION SUMMARY:")
        for step_id, result in results["step_results"].items():
            status_emoji = "" if result["status"] == "SUCCESS" else ""
            execution_time = result.get("execution_time", 0)
            print(f"   {status_emoji} {result['name']}: {result['status']} ({execution_time:.1f}s)")
            if result["status"] == "FAILED" and result.get("retry_count", 0) > 0:
                print(f"       Retries: {result['retry_count']}")
        
        # Recommendations
        print(f"\n RECOMMENDATIONS:")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"   {i}. {rec}")
        
        # Next steps
        print(f"\n NEXT STEPS:")
        for i, step in enumerate(results["next_steps"], 1):
            print(f"   {i}. {step}")
        
        print(f"\n OUTPUTS AVAILABLE:")
        print(f"    Pipeline Logs: {self.config.PIPELINE_LOGS_DIR}")
        print(f"    Execution Reports: {self.config.PIPELINE_REPORTS_DIR}")
        print(f"    Trained Models: {os.path.join(self.config.BASE_DIR, 'models')}")
        print(f"    Feature Store: {os.path.join(self.config.BASE_DIR, 'feature_store')}")
    
    def execute_single_step(self, step_id: str) -> Dict[str, Any]:
        """Execute a single pipeline step."""
        return self.execute_pipeline(steps_to_run=[step_id])
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics."""
        return {
            "execution_id": self.execution_id,
            "is_running": self.pipeline_start_time is not None and self.pipeline_end_time is None,
            "start_time": self.pipeline_start_time.isoformat() if self.pipeline_start_time else None,
            "steps_completed": len([r for r in self.step_results.values() if r["status"] in ["SUCCESS", "FAILED"]]),
            "total_steps": len(self.config.PIPELINE_STEPS),
            "current_memory_mb": self.process.memory_info().rss / 1024 / 1024
        }


def main():
    """Main function to run complete pipeline."""
    orchestrator = PipelineOrchestrator()
    results = orchestrator.execute_pipeline()
    return results


if __name__ == "__main__":
    main()
