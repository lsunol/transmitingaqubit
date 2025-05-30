"""
Job Status Checker - Utility script to check the status of quantum jobs.

This script provides a quick way to check the status of jobs without 
attempting to collect results.

Usage:
    python check_status.py [options]
"""

import argparse
import csv
import os
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers.exceptions import JobError


def parse_arguments():
    """Parse command line arguments for status checking."""
    parser = argparse.ArgumentParser(
        description='Check status of quantum jobs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--output-dir', type=str, default='quantum/output',
                       help='Directory containing the master CSV file')
    parser.add_argument('--job-id', type=str,
                       help='Check status of specific job ID')
    parser.add_argument('--pending-only', action='store_true',
                       help='Show only pending jobs')
    
    return parser.parse_args()


def load_jobs_from_csv(master_csv_path, pending_only=False):
    """
    Load jobs from the master CSV.
    
    Args:
        master_csv_path (str): Path to the master CSV file
        pending_only (bool): If True, return only jobs without results
        
    Returns:
        list: List of job dictionaries
    """
    if not os.path.exists(master_csv_path):
        print(f"Master CSV file not found: {master_csv_path}")
        return []
    
    jobs = []
    with open(master_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if pending_only:
                # Only include jobs without results
                if not row['counts'] and not row['experimental_results']:
                    jobs.append(row)
            else:
                jobs.append(row)
    
    return jobs


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


def format_job_info(job_row, status=None):
    """Format job information for display."""
    job_id = job_row['job_id']
    backend = job_row.get('backend_name', job_row['backend_type'])
    state = job_row['state']
    povm = job_row['povm']
    shots = job_row['total_shots']
    datetime_str = job_row['datetime']
    
    has_results = bool(job_row['counts'] and job_row['experimental_results'])
    
    info = f"Job ID: {job_id}\n"
    info += f"  Date: {datetime_str}\n"
    info += f"  Backend: {backend}\n"
    info += f"  State: {state}, POVM: {povm}, Shots: {shots}\n"
    info += f"  Has Results: {'Yes' if has_results else 'No'}\n"
    if status:
        info += f"  Status: {status}\n"
    
    return info


def main():
    """Main function for checking job status."""
    args = parse_arguments()
    
    # Ensure output directory exists
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return
    
    # Master CSV file path
    master_csv_path = os.path.join(output_dir, "experiments_master.csv")
    
    # Initialize IBM Quantum service
    try:
        service = QiskitRuntimeService()
        print("Connected to IBM Quantum services\n")
    except Exception as e:
        print(f"Failed to connect to IBM Quantum services: {e}")
        print("Note: Status checking requires IBM Quantum credentials for remote jobs.\n")
        service = None
    
    if args.job_id:
        # Check specific job ID
        if service:
            job, status = check_job_status(args.job_id, service)
            if job:
                print(f"Job {args.job_id}: {status}")
            else:
                print(f"Job {args.job_id}: {status}")
        else:
            print(f"Cannot check status of job {args.job_id} without IBM Quantum connection")
        return
    
    # Load jobs from CSV
    jobs = load_jobs_from_csv(master_csv_path, pending_only=args.pending_only)
    
    if not jobs:
        if args.pending_only:
            print("No pending jobs found.")
        else:
            print("No jobs found in master CSV.")
        return
    
    print(f"Found {len(jobs)} job(s):")
    print("=" * 50)
    
    for i, job_row in enumerate(jobs, 1):
        job_id = job_row['job_id']
        
        # Check status if we have service connection and job doesn't have results
        status = None
        if service and not (job_row['counts'] and job_row['experimental_results']):
            _, status = check_job_status(job_id, service)
        
        print(f"\n{i}. {format_job_info(job_row, status)}")
    
    # Summary
    if not args.pending_only and service:
        total_jobs = len(jobs)
        pending_jobs = [j for j in jobs if not (j['counts'] and j['experimental_results'])]
        completed_jobs = total_jobs - len(pending_jobs)
        
        print("\n" + "=" * 50)
        print(f"Summary:")
        print(f"  Total jobs: {total_jobs}")
        print(f"  Completed: {completed_jobs}")
        print(f"  Pending: {len(pending_jobs)}")


if __name__ == "__main__":
    main()
