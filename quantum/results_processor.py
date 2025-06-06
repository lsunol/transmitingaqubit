"""
Results Processor - Centralized handling of quantum experiment results.

This module provides unified processing of quantum job results, including
KL divergence calculation and CSV updates, eliminating code duplication
between main.py and collect_results.py.
"""

from typing import Dict, List, Any
from qiskit.visualization import plot_histogram
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
from state import create_state_from_args
from povm import create_povm
from csv_manager import ExperimentCSVManager
# Use a predefined colormap from matplotlib for visual distinction
import matplotlib.cm as cm


class QuantumResultsProcessor:
    """Handles processing of quantum experiment results."""
    
    def __init__(self, csv_manager: ExperimentCSVManager):
        """
        Initialize the results processor.
        
        Args:
            csv_manager (ExperimentCSVManager): CSV manager instance
        """
        self.csv_manager = csv_manager
    
    def plot_kl_divergence_analysis(self, kl_analysis: List[Dict], job_id: str, output_dir: str, povm=None, state=None) -> str:
        """
        Plot KL divergence analysis and save as PNG. Also saves each subplot as a separate PNG file.
        """
        if not kl_analysis or len(kl_analysis) == 0:
            print(f"  Warning: No KL analysis data to plot for job {job_id}")
            return ""
        
        shots = np.array([entry['shots'] for entry in kl_analysis])
        kl_values = np.array([entry['kl_divergence'] for entry in kl_analysis])
        plt.ioff()

        # --- Helper functions for each plot ---
        def plot_vs_shots(ax):
            scatter = ax.scatter(shots, kl_values, alpha=0.7, s=30, c=kl_values, cmap='viridis', edgecolors='none')
            ax.plot(shots, kl_values, 'b-', alpha=0.5, linewidth=1)
            ax.set_xlabel('Number of Shots', fontweight='bold')
            ax.set_ylabel('KL Divergence Value', fontweight='bold')
            ax.set_title('KL Divergence vs Number of Shots', fontweight='bold')
            ax.grid(True, alpha=0.3)
            return scatter

        def plot_log_scale(ax):
            ax.scatter(shots, kl_values, alpha=0.7, s=20, color='green')
            ax.set_xlabel('Number of Shots', fontweight='bold')
            ax.set_ylabel('KL Divergence (log scale)', fontweight='bold')
            ax.set_title('KL Divergence (Log Scale)', fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        def plot_hist(ax):
            n_bins = min(30, len(kl_values) // 5) if len(kl_values) > 10 else 10
            ax.hist(kl_values, bins=n_bins, color='orange', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('KL Divergence Value', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title('Distribution of KL Divergence Values', fontweight='bold')
            ax.grid(True, alpha=0.3)

        def plot_convergence(ax):
            ax.plot(shots, kl_values, 'b-', alpha=0.6, linewidth=1, label='KL Divergence')
            if len(kl_values) > 20:
                window_size = min(50, len(kl_values) // 10)
                if window_size > 1:
                    moving_avg = np.convolve(kl_values, np.ones(window_size)/window_size, mode='valid')
                    avg_shots = shots[window_size-1:]
                    ax.plot(avg_shots, moving_avg, 'red', linewidth=2, alpha=0.8, 
                            label=f'Moving Average (window={window_size})')
            ax.set_xlabel('Number of Shots', fontweight='bold')
            ax.set_ylabel('KL Divergence', fontweight='bold')
            ax.set_title('Convergence Trend', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # --- Combined 4-subplot figure ---
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        scatter = plot_vs_shots(axs[0, 0])
        fig.colorbar(scatter, ax=axs[0, 0], label='KL Divergence Value')
        plot_log_scale(axs[0, 1])
        plot_hist(axs[1, 0])
        plot_convergence(axs[1, 1])
        povm_label = povm.get_label() if povm is not None and hasattr(povm, 'get_label') else ''
        state_label = str(state) if state is not None else ''
        fig.suptitle(f'KL divergence analysis - {povm_label} - State {state_label}', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "kl_divergence_analysis.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ KL divergence plot saved: {output_path}")

        # --- Individual plots using the same helpers ---
        # 1. KL Divergence vs Number of Shots
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        scatter1 = plot_vs_shots(ax1)
        fig1.colorbar(scatter1, ax=ax1, label='KL Divergence Value')
        fig1.tight_layout()
        file1 = os.path.join(output_dir, "kl_divergence_analysis_vs_shots.png")
        fig1.savefig(file1, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig1)

        # 2. KL Divergence (Log Scale)
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        plot_log_scale(ax2)
        fig2.tight_layout()
        file2 = os.path.join(output_dir, "kl_divergence_analysis_log_scale.png")
        fig2.savefig(file2, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig2)

        # 3. Distribution of KL Divergence Values (Histogram)
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        plot_hist(ax3)
        fig3.tight_layout()
        file3 = os.path.join(output_dir, "kl_divergence_analysis_histogram.png")
        fig3.savefig(file3, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig3)

        # 4. Convergence Trend
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        plot_convergence(ax4)
        fig4.tight_layout()
        file4 = os.path.join(output_dir, "kl_divergence_analysis_convergence.png")
        fig4.savefig(file4, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig4)

        print(f"  ✓ Individual KL divergence plots saved: {file1}, {file2}, {file3}, {file4}")
        return output_path

    def plot_povm_outcome_distribution_histogram(self, counts: dict, output_dir: str, povm, state) -> str:
        """
        Plot POVM outcome distribution using Qiskit's plot_histogram, with bitstring and label.
        
        Args:
            counts (dict): Dictionary of outcome bitstrings to counts
            job_id (str): (unused, kept for compatibility)
            output_dir (str): Output directory to save the plot
            povm: POVM object with get_outcome_label_map() and get_label()
            state: State object for labeling
            
        Returns:
            str: Path to saved plot file
        """
        label_map = povm.get_outcome_label_map()
        new_counts = {}
        for bitstring, count in counts.items():
            label = label_map.get(bitstring, bitstring)
            new_key = f"{bitstring} ({label})"
            new_counts[new_key] = count
        counts = new_counts
        state_label = str(state)
        povm_label = povm.get_label()
        label = f'{povm_label} Outcome Distribution \nfor state: {state_label}'
        fig = plot_histogram(counts, title=label, bar_labels=True)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "povm_outcome_experimental_distribution_histogram.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ POVM outcome distribution plot saved: {output_path}")
        return output_path

    def plot_povm_distribution_pie(self, distribution: dict, job_id: str, output_dir: str, filename_suffix: str, povm=None, state=None) -> str:
        """
        Plot POVM outcome distribution as a pie chart, with bitstring and label.
        Args:
            distribution (dict): Dictionary of outcome bitstrings to probabilities or counts
            job_id (str): Job ID for the plot title
            output_dir (str): Output directory to save the plot
            filename_suffix (str): Suffix for the output filename (e.g., 'experimental', 'theoretical')
            povm: POVM object with get_outcome_label_map() (optional, for labeling)
            state: State object, used to extract label or angles for the title (must not be None)
        Returns:
            str: Path to saved plot file
        """
        # If povm is provided, relabel the keys
        if povm is not None:
            label_map = povm.get_outcome_label_map()  # Should map bitstrings to labels
            new_distribution = {}
            for bitstring, value in distribution.items():
                label = label_map.get(bitstring, bitstring)
                new_key = f"{bitstring} ({label})"
                new_distribution[new_key] = value
            distribution = new_distribution
        
        # Get consistent color mapping for POVM outcomes
        color_map = self._get_consistent_colors(distribution.keys())
        
        # Sort keys for consistency
        labels = sorted(list(distribution.keys()))
        values = [distribution[label] for label in labels]
        colors = [color_map[label] for label in labels]
        
        plt.ioff()
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Compose state and POVM label for title
        state_label = str(state) if state else "Unknown"
        povm_label = povm.get_label() if hasattr(povm, 'get_label') else "POVM"
        
        # Create the pie chart with fixed colors and direct labels
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return f'{pct:.1f}%\n({val})'
            return my_autopct

        ax.pie(
            values,
            labels=labels,
            colors=colors,
            startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1},
            autopct=make_autopct(values),
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        ax.set_title(f'{povm_label} Outcome Distribution ({filename_suffix.capitalize()})\nfor state: {state_label}', fontweight='bold')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"povm_outcome_{filename_suffix}_distribution_pie.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ POVM outcome {filename_suffix} pie chart saved: {output_path}")
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
        import json
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
            
            # Set output directory for plots and data files
            output_dir = f"quantum/output/experiment_{job_id}"
            os.makedirs(output_dir, exist_ok=True)

            # --- Save experiment results as JSON ---
            experiment_results_path = os.path.join(output_dir, "experiment_results.json")
            with open(experiment_results_path, 'w', encoding='utf-8') as f:
                json.dump(experimental_results, f, ensure_ascii=False, indent=2)
            print(f"  ✓ Experimental results saved: {experiment_results_path}")

            # --- Save KL divergence values as JSON ---
            kl_divergence_json = [
                {"shots": int(entry["shots"]), "kl_divergence_value": float(entry["kl_divergence"])}
                for entry in kl_analysis
            ]
            kl_divergence_path = os.path.join(output_dir, "kl_divergence_values.json")
            with open(kl_divergence_path, 'w', encoding='utf-8') as f:
                json.dump(kl_divergence_json, f, ensure_ascii=False, indent=2)
            print(f"  ✓ KL divergence values saved: {kl_divergence_path}")

            # Plot KL divergence analysis
            self.plot_kl_divergence_analysis(kl_analysis, job_id, output_dir, povm=povm, state=state)
            # Plot POVM outcome distribution using Qiskit's histogram, with labels
            self.plot_povm_outcome_distribution_histogram(counts, output_dir, povm, state)
            
            # Prepare experimental and theoretical distributions with the same format for consistent coloring
            experimental_distribution = counts
            theoretical_distribution = povm.get_theoretical_distribution(state)
            
            # Plot POVM outcome distribution as pie chart, with labels (experimental)
            self.plot_povm_distribution_pie(experimental_distribution, job_id, output_dir, "experiment", povm=povm, state=state)
            
            # Plot POVM theoretical distribution as pie chart, with labels (using same color mapping)
            self.plot_povm_distribution_pie(theoretical_distribution, job_id, output_dir, "expected", povm=povm, state=state)
            
            # Plot POVM outcome distribution
            outcome_map = povm_labels = str(list(povm.get_outcome_label_map().values()))
            
            # --- Reconstruct original state for SICPOVM and save to CSV ---
            reconstructed_state_theta = None
            reconstructed_state_phi = None
            reconstructed_state_image_file = None
            if hasattr(povm, 'reconstruct_original_state'):
                try:
                    reconstructed_state, image_path = povm.reconstruct_original_state(counts, output_dir)
                    reconstructed_state_theta, reconstructed_state_phi = reconstructed_state.get_bloch_angles()
                    # Save relative path for CSV
                    reconstructed_state_image_file = os.path.relpath(image_path, start=os.path.dirname(self.csv_manager.master_csv_path))
                    print(f"  ✓ Reconstructed state: theta={reconstructed_state_theta}, phi={reconstructed_state_phi}")
                    print(f"  ✓ Reconstructed state image saved: {image_path}")
                    # Diagnostic: compare with expected state
                    if hasattr(state, 'get_bloch_angles'):
                        expected_theta, expected_phi = state.get_bloch_angles()
                        print(f"  [Diagnostic] Expected state: theta={expected_theta}, phi={expected_phi}")
                        d_theta = abs(reconstructed_state_theta - expected_theta)
                        d_phi = abs((reconstructed_state_phi - expected_phi + 360) % 360)
                        print(f"  [Diagnostic] Difference: Δtheta={d_theta}, Δphi={d_phi}")
                        # Check for antipodal (opposite) direction
                        if abs(d_theta - 180) < 1e-2:
                            print("  [Diagnostic] Reconstructed state is antipodal (opposite direction) to expected state.")
                except Exception as e:
                    print(f"  ✗ State reconstruction failed: {e}")
            
            # Create updated job data (store only file paths for large data)
            updated_job_data = self._create_updated_job_data(
                job_data, counts, experiment_results_path, povm_labels, kl_divergence_path
            )
            
            # Add reconstructed state info to CSV
            if reconstructed_state_theta is not None and reconstructed_state_phi is not None:
                updated_job_data['reconstructed_state_theta'] = reconstructed_state_theta
                updated_job_data['reconstructed_state_phi'] = reconstructed_state_phi
            if reconstructed_state_image_file is not None:
                updated_job_data['reconstructed_state_image_file'] = reconstructed_state_image_file
            
            # Update CSV with results
            self.csv_manager.update_experiments_with_results([updated_job_data])
            
            print(f"  ✓ Results processed and saved for job {job_id}")
            return True
            
        except Exception as e:
            print(f"Error processing job {job.job_id()}: {e}")
            return False
    
    def _get_consistent_colors(self, labels):
        """
        Generate a consistent color mapping for POVM outcome labels.
        
        Args:
            labels: List or set of outcome labels
            
        Returns:
            dict: Mapping from label to color
        """
        # Sort labels to ensure consistent color assignment across plots
        sorted_labels = sorted(list(labels))
        
        # Choose a colormap that works well for pie charts
        colormap = cm.get_cmap('tab20')  # tab20 provides distinct colors for up to 20 categories
        
        # If we have more than 20 outcomes, we'll use hsv colormap which can generate more colors
        if len(sorted_labels) > 20:
            colormap = cm.get_cmap('hsv')
        
        # Generate colors
        colors = {}
        for i, label in enumerate(sorted_labels):
            # Normalize to [0, 1] range for colormap
            color_index = i / max(1, len(sorted_labels) - 1)
            colors[label] = colormap(color_index)
            
        return colors
            
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
                                counts: Dict[str, int], experiment_results_path: str,
                                povm_labels: str, kl_divergence_path: str) -> Dict[str, str]:
        """
        Create updated job data with results, storing only file paths for large data.
        
        Args:
            original_job_data (dict): Original job data from CSV
            counts (dict): Job measurement counts
            experiment_results_path (str): Path to experiment_results.json
            povm_labels (str): POVM labels
            kl_divergence_path (str): Path to kl_divergence_values.json
        
        Returns:
            dict: Updated job data with results (file references)
        """
        updated_data = original_job_data.copy()
        updated_data['counts'] = str(counts)
        # Store only the relative path to the experiment results JSON
        updated_data['experimental_results_file'] = os.path.relpath(experiment_results_path, start=os.path.dirname(self.csv_manager.master_csv_path))
        updated_data['povm_labels'] = povm_labels
        # Store only the relative path to the KL divergence JSON
        updated_data['kl_divergence_file'] = os.path.relpath(kl_divergence_path, start=os.path.dirname(self.csv_manager.master_csv_path))
        # Remove old large fields if present
        updated_data.pop('experimental_results', None)
        updated_data.pop('kl_divergence_value', None)
        updated_data.pop('kl_divergence_shots', None)
        return updated_data
