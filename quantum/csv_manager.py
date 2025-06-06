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
    
    def _get_existing_fieldnames(self) -> List[str]:
        """Get the fieldnames from existing CSV file, or return empty list if file doesn't exist."""
        if not os.path.exists(self.master_csv_path):
            return []
        
        try:
            with open(self.master_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                return list(reader.fieldnames) if reader.fieldnames else []
        except Exception:
            return []
    
    def _get_combined_fieldnames(self, new_data: Dict[str, str]) -> List[str]:
        """Combine existing fieldnames with new data keys, preserving order."""
        existing_fields = self._get_existing_fieldnames()
        new_fields = list(new_data.keys())
        
        # Combine existing and new fields, preserving order and avoiding duplicates
        combined_fields = existing_fields.copy()
        for field in new_fields:
            if field not in combined_fields:
                combined_fields.append(field)
                
        return combined_fields
    

    def ensure_master_csv_exists(self):
        """Create the master CSV file if it doesn't exist."""
        # The empty file will be created when needed with appropriate headers
        if not os.path.exists(self.master_csv_path):
            print(f"CSV file doesn't exist yet. It will be created when data is appended.")
    
    def _rewrite_csv_with_new_fields(self, fieldnames: List[str], new_data: Dict[str, str]):
        """
        Rewrite the CSV with new fieldnames and append the new data.
        
        Args:
            fieldnames: List of field names for the CSV header
            new_data: New record to append
        """
        existing_data = []
        
        # If file exists, read existing data first
        if os.path.exists(self.master_csv_path):
            existing_data = self.load_all_experiments()
        
        # Write the CSV with new headers and all data
        with open(self.master_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write existing data
            for row in existing_data:
                writer.writerow(row)
                
            # Write new data
            writer.writerow(new_data)
    
    def append_experiment_data(self, experiment_data: Dict[str, str]):
        """
        Append a single experiment record to the master CSV.
        
        Args:
            experiment_data (dict): Dictionary containing experiment data
        """
        # Get the combined fieldnames (existing + new fields)
        fieldnames = self._get_combined_fieldnames(experiment_data)
        
        # Check if we need to rewrite the file (if new fields were added)
        existing_fieldnames = self._get_existing_fieldnames()
        file_exists = os.path.exists(self.master_csv_path)
        
        if not file_exists or set(fieldnames) != set(existing_fieldnames):
            # File doesn't exist or we have new fields - need to rewrite
            self._rewrite_csv_with_new_fields(fieldnames, experiment_data)
        else:
            # File exists and no new fields - simple append
            with open(self.master_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
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
            if ((not 'counts' in row or not row['counts']) and 
                (not 'experimental_results_file' in row or not row['experimental_results_file'])):
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
        
        # --- Use existing CSV header order as base, append new fields at the end in order of first updated job ---
        existing_field_order = self._get_existing_fieldnames()
        # Find new fields in updated_jobs[0] that are not in existing_field_order
        new_fields = [f for f in updated_jobs[0].keys() if f not in existing_field_order]
        final_fieldnames = existing_field_order + new_fields
        
        # Write updated CSV
        with open(self.master_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=final_fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"Updated {len(updated_jobs)} job(s) in master CSV: {self.master_csv_path}")

    def create_experiment_record(self, **kwargs) -> Dict[str, str]:
        """
        Create a standardized experiment record dictionary.
        
        Args:
            **kwargs: Any key-value pairs to include in the experiment record
            
        Returns:
            dict: Standardized experiment record with default timestamp
        """
        # Add timestamp if not provided
        record = {'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        # Add all provided key-value pairs
        record.update(kwargs)
        
        # Convert any non-string values to strings for CSV compatibility
        for key, value in record.items():
            if value is not None and not isinstance(value, str):
                record[key] = str(value)
                
        return record
    
    def add_experiment_to_master_csv(self, job_data: Dict[str, str]):
        """
        Add a new experiment entry to the master CSV from job data.
        
        Args:
            job_data (dict): Dictionary containing job data
        """
        # Create experiment record
        experiment_record = {}
        for key, value in job_data.items():
            experiment_record[key] = value
        # Add to CSV
        self.append_experiment_data(experiment_record)

    @classmethod
    def setup_experiment(cls, output_dir: str, job_data: Dict[str, str]) -> tuple['ExperimentCSVManager', str]:
        """
        Factory method to create experiment subfolder, record experiment in master CSV, and return both manager and folder path.
        
        Args:
            output_dir (str): Base output directory (e.g., 'output/')
            job_data (dict): Dictionary containing all experiment/job metadata
        
        Returns:
            tuple: (csv_manager_instance, experiment_folder_path)
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create a unique experiment folder (use job_id if available, else timestamp)
        job_id = job_data.get('job_id')
        if job_id:
            folder_name = f"experiment_{job_id}"
        else:
            folder_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_folder = os.path.join(output_dir, folder_name)
        os.makedirs(experiment_folder, exist_ok=True)

        # Add experiment_folder to job_data (overwrite if already present)
        job_data['experiment_folder'] = folder_name

        # Create CSV manager and append experiment data
        manager = cls(output_dir)
        manager.append_experiment_data(job_data)

        print(f"Experiment folder created: {experiment_folder}")
        return manager, experiment_folder
