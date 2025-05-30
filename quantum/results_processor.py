"""
Results Processor - Centralized handling of quantum experiment results.

This module provides unified processing of quantum job results, including
KL divergence calculation and CSV updates, eliminating code duplication
between main.py and collect_results.py.
"""

from typing import Dict, List, Any
from state import create_state_from_args
from povm import create_povm
from csv_manager import ExperimentCSVManager


class QuantumResultsProcessor:
    """Handles processing of quantum experiment results."""
    
    def __init__(self, csv_manager: ExperimentCSVManager):
        """
        Initialize the results processor.
        
        Args:
            csv_manager (ExperimentCSVManager): CSV manager instance
        """
        self.csv_manager = csv_manager
    
    def process_job_results(self, job, job_data: Dict[str, str]) -> bool:
        """
        Process completed job results including KL divergence calculation and CSV updates.
        
        Args:
            job: The completed quantum job object
            job_data (dict): Job data from CSV containing state, povm, etc.
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            job_id = job.job_id()
            print(f"Processing completed job {job_id}...")
            
            # Extract job results
            counts = self._get_job_counts(job)
            experimental_results = self._get_job_results(job)
            
            print(f"  Counts: {counts}")
            print(f"  Results length: {len(experimental_results)}")
            
            # Recreate state and POVM for KL divergence calculation
            state = create_state_from_args(job_data['state'], None)
            povm = create_povm(job_data['povm'])
            
            # Calculate KL divergence
            kl_analysis = povm.calculate_kl_divergence(experimental_results, state)
            print(f"  KL Divergence calculated: {len(kl_analysis)} data points")
            
            # Get POVM labels
            povm_labels = str(list(povm.get_outcome_label_map().values()))
            
            # Create updated job data
            updated_job_data = self._create_updated_job_data(
                job_data, counts, experimental_results, povm_labels, kl_analysis
            )
            
            # Update CSV with results
            self.csv_manager.update_experiments_with_results([updated_job_data])
            
            print(f"  ✓ Results processed and saved for job {job_id}")
            return True
            
        except Exception as e:
            print(f"Error processing job {job.job_id()}: {e}")
            return False
    
    def _get_job_counts(self, job) -> Dict[str, int]:
        """Extract counts from a completed job."""
        result = job.result()
        counts = result[0].data.c.get_counts()
        return counts
    
    def _get_job_results(self, job) -> List:
        """Extract experimental results from a completed job."""
        bit_array = next(iter(job.result()[0].data.values()))
        return bit_array.get_bitstrings()
    
    def _create_updated_job_data(self, original_job_data: Dict[str, str], 
                                counts: Dict[str, int], experimental_results: List,
                                povm_labels: str, kl_analysis: List[Dict]) -> Dict[str, str]:
        """
        Create updated job data with results.
        
        Args:
            original_job_data (dict): Original job data from CSV
            counts (dict): Job measurement counts
            experimental_results (list): Experimental results
            povm_labels (str): POVM labels
            kl_analysis (list): KL divergence analysis data
            
        Returns:
            dict: Updated job data with results
        """
        updated_data = original_job_data.copy()
        updated_data['counts'] = str(counts)
        updated_data['experimental_results'] = str(experimental_results)
        updated_data['povm_labels'] = povm_labels
        
        # Format KL analysis
        if kl_analysis and len(kl_analysis) > 0:
            shots_list = []
            values_list = []
            for entry in kl_analysis:
                shots = entry['shots']
                kl_value = entry['kl_divergence']
                # Convert np.float64 to Python float if needed
                if hasattr(kl_value, 'item'):
                    kl_value = kl_value.item()
                shots_list.append(str(shots))
                values_list.append(str(kl_value))
            updated_data['kl_divergence_shots'] = ';'.join(shots_list)
            updated_data['kl_divergence_value'] = ';'.join(values_list)
        else:
            updated_data['kl_divergence_shots'] = ''
            updated_data['kl_divergence_value'] = ''
        
        return updated_data
