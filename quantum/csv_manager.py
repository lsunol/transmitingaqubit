"""
CSV Manager - Centralized handling of experiment CSV files.

This module provides a unified interface for reading from and writing to
the master experiments CSV file, eliminating code duplication between
main.py and collect_results.py.
"""

import csv
import os
from datetime import datetime
from typing import List, Dict, Optional


class ExperimentCSVManager:
    """Manages reading and writing of experiment data to CSV files."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the CSV manager.
        
        Args:
            output_dir (str): Directory where CSV files are stored
        """
        self.output_dir = output_dir
        self.master_csv_path = os.path.join(output_dir, "experiments_master.csv")
        self.fieldnames = [
            'datetime', 'job_id', 'experiment_folder', 'backend_type', 'backend_name', 
            'noise_model', 'state', 'povm', 'povm_labels', 'total_shots', 'counts', 
            'experimental_results', 'kl_divergence_shots', 'kl_divergence_value'
        ]
    
    def ensure_master_csv_exists(self):
        """Create the master CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.master_csv_path):
            print(f"Creating master CSV file: {self.master_csv_path}")
            with open(self.master_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def append_experiment_data(self, experiment_data: Dict[str, str]):
        """
        Append a single experiment record to the master CSV.
        
        Args:
            experiment_data (dict): Dictionary containing experiment data
        """
        self.ensure_master_csv_exists()
        
        with open(self.master_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(experiment_data)
        
        print(f"Experiment data appended to master CSV: {self.master_csv_path}")
    
    def load_all_experiments(self) -> List[Dict[str, str]]:
        """
        Load all experiment records from the master CSV.
        
        Returns:
            list: List of dictionaries representing all experiments
        """
        if not os.path.exists(self.master_csv_path):
            print(f"Master CSV file not found: {self.master_csv_path}")
            return []
        
        experiments = []
        with open(self.master_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            experiments = list(reader)
        
        return experiments
    
    def load_pending_jobs(self) -> List[Dict[str, str]]:
        """
        Load pending jobs (jobs without results) from the master CSV.
        
        Returns:
            list: List of dictionaries representing pending jobs
        """
        all_experiments = self.load_all_experiments()
        
        pending_jobs = []
        for row in all_experiments:
            # A job is pending if it has no counts or experimental_results
            if not row['counts'] and not row['experimental_results']:
                pending_jobs.append(row)
        
        return pending_jobs
    
    def update_experiments_with_results(self, updated_jobs: List[Dict[str, str]]):
        """
        Update the master CSV file with collected results.
        
        Args:
            updated_jobs (list): List of updated job dictionaries
        """
        if not updated_jobs:
            print("No jobs to update.")
            return
        
        # Read all existing data
        all_rows = self.load_all_experiments()
        
        if not all_rows:
            print("No existing data found in master CSV.")
            return
        
        # Create a mapping of job_id to updated data
        updates_map = {job['job_id']: job for job in updated_jobs}
        
        # Update rows with new data
        for row in all_rows:
            if row['job_id'] in updates_map:
                row.update(updates_map[row['job_id']])
        
        # Write updated CSV
        with open(self.master_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"Updated {len(updated_jobs)} job(s) in master CSV: {self.master_csv_path}")
    
    def create_experiment_record(self, job_id: str, experiment_folder: str, 
                                backend_type: str, backend_name: str, noise_model: str,
                                state: str, povm: str, total_shots: int,
                                counts: str = "", experimental_results: str = "",
                                povm_labels: str = "", kl_divergence_shots: str = "",
                                kl_divergence_value: str = "") -> Dict[str, str]:
        """
        Create a standardized experiment record dictionary.
        
        Args:
            job_id (str): The quantum job ID
            experiment_folder (str): Name of the experiment folder
            backend_type (str): Type of backend (real/simulator)
            backend_name (str): Name of the backend
            noise_model (str): Noise model used
            state (str): Quantum state
            povm (str): POVM type
            total_shots (int): Total number of shots
            counts (str): Job counts (optional)
            experimental_results (str): Experimental results (optional)
            povm_labels (str): POVM labels (optional)
            kl_divergence_shots (str): KL divergence shots data (optional)
            kl_divergence_value (str): KL divergence values (optional)
            
        Returns:
            dict: Standardized experiment record
        """
        return {
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'job_id': job_id,
            'experiment_folder': experiment_folder,
            'backend_type': backend_type,
            'backend_name': backend_name,
            'noise_model': noise_model,
            'state': state,
            'povm': povm,
            'povm_labels': povm_labels,
            'total_shots': str(total_shots),
            'counts': counts,
            'experimental_results': experimental_results,
            'kl_divergence_shots': kl_divergence_shots,
            'kl_divergence_value': kl_divergence_value
        }
