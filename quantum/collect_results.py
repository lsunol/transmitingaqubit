"""
Collect Results Script - Second entry point for collecting results from queued quantum jobs.

This script checks the status of pending jobs from real IBM quantum computers
and collects results when jobs are completed. It processes the CSV file to find
pending jobs, checks their status, and updates the results when available.

Usage:
    python collect_results.py [options]
"""

import argparse
import os
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers.exceptions import JobError

from state import create_state_from_args
from povm import create_povm
from csv_manager import ExperimentCSVManager
from results_processor import QuantumResultsProcessor


def parse_arguments():
    """Parse command line arguments for result collection."""
    parser = argparse.ArgumentParser(
        description='Collect results from queued quantum experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--output-dir', type=str, default='quantum/output',
                       help='Directory containing the master CSV file')
    
    return parser.parse_args()


def get_job_counts(job):
    """Extract counts from a completed job."""
    result = job.result()
    counts = result[0].data.c.get_counts()
    return counts


def get_job_results(job):
    """Extract experimental results from a completed job."""
    bit_array = next(iter(job.result()[0].data.values()))
    return bit_array.get_bitstrings()


def load_pending_jobs(master_csv_path):
    """
    Load pending jobs (jobs without results) from the master CSV.
    
    Args:
        master_csv_path (str): Path to the master CSV file
        
    Returns:
        list: List of dictionaries representing pending jobs
    """
    # This function is now handled by ExperimentCSVManager
    # Keep for backward compatibility but delegate to CSV manager
    output_dir = os.path.dirname(master_csv_path)
    csv_manager = ExperimentCSVManager(output_dir)
    return csv_manager.load_pending_jobs()


def check_job_status(job_id, service):
    """
    Check the status of a specific job.
    
    Args:
        job_id (str): The job ID to check
        service (QiskitRuntimeService): The runtime service instance
        
    Returns:
        tuple: (job_object, status_string) or (None, error_message)
    """
    try:
        job = service.job(job_id)
        status = job.status()
        return job, str(status)
    except JobError as e:
        return None, f"Job error: {e}"
    except Exception as e:
        return None, f"Unknown error: {e}"


def process_completed_job(job, job_row):
    """
    Process a completed job and extract results.
    
    This function is deprecated - use QuantumResultsProcessor instead.
    """
    print("Warning: process_completed_job is deprecated. Use QuantumResultsProcessor instead.")
    return None


def main():
    """Main function for collecting results from queued jobs."""
    args = parse_arguments()
    
    # Ensure output directory exists
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return
    
    # Initialize CSV manager and results processor
    csv_manager = ExperimentCSVManager(output_dir)
    results_processor = QuantumResultsProcessor(csv_manager)
    
    # Load pending jobs
    pending_jobs = csv_manager.load_pending_jobs()
    
    if not pending_jobs:
        print("No pending jobs found.")
        return
    
    print(f"Found {len(pending_jobs)} pending job(s)")
    
    # Initialize IBM Quantum service
    try:
        service = QiskitRuntimeService()
        print("Connected to IBM Quantum services")
    except Exception as e:
        print(f"Failed to connect to IBM Quantum services: {e}")
        print("Make sure you have valid IBM Quantum credentials configured.")
        return
    
    # Check status of each pending job
    completed_jobs_count = 0
    
    for job_row in pending_jobs:
        job_id = job_row['job_id']
        
        print(f"\nChecking job {job_id}...")
        
        job, status = check_job_status(job_id, service)
        
        if job is None:
            print(f"Job {job_id}: {status}")
            continue
        
        print(f"Job {job_id}: {status}")
        
        if status == "DONE":
            # Job is completed, process results using centralized processor
            success = results_processor.process_job_results(job, job_row)
            if success:
                completed_jobs_count += 1
                print(f"  ✓ Results collected for job {job_id}")
            else:
                print(f"  ✗ Failed to process results for job {job_id}")
        
        elif status in ["ERROR", "JobStatus.CANCELLED"]:
            print(f"  ⚠ Job {job_id} failed with status: {status}")
            # Optionally, you could mark these jobs as failed in the CSV
        
        else:
            # Job is still queued or running
            print(f"  ⏳ Job {job_id} is still {status}")
    
    # Summary
    total_pending = len(pending_jobs)
    collected = completed_jobs_count
    still_pending = total_pending - collected
    
    if collected > 0:
        print(f"\n✓ Successfully collected results from {collected} job(s)")
    else:
        print("\nNo new results to collect at this time.")
    
    print(f"\nSummary:")
    print(f"  Total pending jobs checked: {total_pending}")
    print(f"  Results collected: {collected}")
    print(f"  Still pending: {still_pending}")


if __name__ == "__main__":
    main()
