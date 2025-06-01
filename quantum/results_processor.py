"""
Results Processor - Centralized handling of quantum experiment results.

This module provides unified processing of quantum job results, including
KL divergence calculation and CSV updates, eliminating code duplication
between main.py and collect_results.py.
"""

from typing import Dict, List, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
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
    
    def plot_kl_divergence_analysis(self, kl_analysis: List[Dict], job_id: str, output_dir: str) -> str:
        """
        Plot KL divergence analysis and save as PNG.
        
        Args:
            kl_analysis (list): List of dicts with 'shots' and 'kl_divergence' keys
            job_id (str): Job ID for the plot title
            output_dir (str): Output directory to save the plot
            
        Returns:
            str: Path to saved plot file
        """
        if not kl_analysis or len(kl_analysis) == 0:
            print(f"  Warning: No KL analysis data to plot for job {job_id}")
            return ""
        
        # Extract shots and KL divergence values
        shots = np.array([entry['shots'] for entry in kl_analysis])
        kl_values = np.array([entry['kl_divergence'] for entry in kl_analysis])
        # Create the plot
        plt.ioff()  # Turn off interactive mode
        fig = plt.figure(figsize=(12, 8))
        
        # Main plot
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(shots, kl_values, alpha=0.7, s=30, c=kl_values, cmap='viridis', edgecolors='none')
        plt.plot(shots, kl_values, 'b-', alpha=0.5, linewidth=1)
        plt.xlabel('Number of Shots', fontweight='bold')
        plt.ylabel('KL Divergence Value', fontweight='bold')
        plt.title(f'KL Divergence vs Number of Shots', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='KL Divergence Value')
        
        # Log scale plot
        plt.subplot(2, 2, 2)
        plt.scatter(shots, kl_values, alpha=0.7, s=20, color='green')
        plt.xlabel('Number of Shots', fontweight='bold')
        plt.ylabel('KL Divergence (log scale)', fontweight='bold')
        plt.title('KL Divergence (Log Scale)', fontweight='bold')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Histogram of KL divergence values
        plt.subplot(2, 2, 3)
        n_bins = min(30, len(kl_values) // 5) if len(kl_values) > 10 else 10
        plt.hist(kl_values, bins=n_bins, color='orange', alpha=0.7, edgecolor='black', linewidth=0.5)
        plt.xlabel('KL Divergence Value', fontweight='bold')
        plt.ylabel('Frequency', fontweight='bold')
        plt.title('Distribution of KL Divergence Values', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Convergence trend (moving average if enough data points)
        plt.subplot(2, 2, 4)
        plt.plot(shots, kl_values, 'b-', alpha=0.6, linewidth=1, label='KL Divergence')
        
        # Add moving average if we have enough points
        if len(kl_values) > 20:
            window_size = min(50, len(kl_values) // 10)
            if window_size > 1:
                moving_avg = np.convolve(kl_values, np.ones(window_size)/window_size, mode='valid')
                avg_shots = shots[window_size-1:]
                plt.plot(avg_shots, moving_avg, 'red', linewidth=2, alpha=0.8, 
                        label=f'Moving Average (window={window_size})')
        
        plt.xlabel('Number of Shots', fontweight='bold')
        plt.ylabel('KL Divergence', fontweight='bold')
        plt.title('Convergence Trend', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Overall figure title
        plt.suptitle(f'KL Divergence Analysis - Job {job_id}', fontsize=14, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "kl_divergence_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # Close to free memory
        
        print(f"  ✓ KL divergence plot saved: {output_path}")
        return output_path
    
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
            
            # Plot KL divergence analysis
            output_dir = f"quantum/output/experiment_{job_id}"
            self.plot_kl_divergence_analysis(kl_analysis, job_id, output_dir)
            
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
