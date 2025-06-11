import argparse
import sys
import numpy as np
import csv
import os
from datetime import datetime
        
from qiskit import transpile        
from state import create_state_from_args
from backend import create_backend
from povm import create_povm
from csv_manager import ExperimentCSVManager
from results_processor import QuantumResultsProcessor
import matplotlib.pyplot as plt

def parse_arguments():
    """Parse command line arguments with no fallback to interactive mode."""
    parser = argparse.ArgumentParser(
        description='Prepare and measure quantum experiments runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # QPU or simulator selection
    qpu_group = parser.add_argument_group('QPU or Simulator Options')
    qpu_group.add_argument('--backend-type', type=str, choices=['qpu', 'fake_backend', 'aer_simulator'], required=True, help='QPU backend name or "simulator"')
    qpu_group.add_argument('--backend-name', type=str, required=False, default="least_busy",
                           choices=['least_busy', 'ibm_aachen', 'ibm_kingston', 'ibm_marrakesh', 'ibm_fez', 'ibm_torino', 'ibm_sherbrooke', 'ibm_quebec', 'ibm_brisbane', 'ibm_kawasaki', 'ibm_rensselaer', 'ibm_brussels', 'ibm_strasbourg'], 
                           help='Specific backend name (required if --backend_type=qpu or --noise-model=real)')
    qpu_group.add_argument('--noise-model', type=str, choices=['zero_noise', 'real', 'custom'], default='real', required=False, help='Noise model to use with aer_simulator (default: real)')
    
    # Custom noise parameters (only relevant if --noise-model=custom)
    noise_group = parser.add_argument_group('Custom Noise Parameters')
    noise_group.add_argument('--prob-meas0-prep1', type=float, default=0.01, help='Probability of measuring 0 when the true state is 1 (readout error)')
    noise_group.add_argument('--prob-meas1-prep0', type=float, default=0.01, help='Probability of measuring 1 when the true state is 0 (readout error)')
    noise_group.add_argument('--error-prob-1qubit-gate', type=float, default=0.001, help='Depolarizing error probability for 1-qubit gates')
    noise_group.add_argument('--error-prob-2qubit-gate', type=float, default=0.01, help='Depolarizing error probability for 2-qubit gates')
    
    # State preparation
    state_group = parser.add_argument_group('Initial qubit preparation options')
    state_choices = ['zero', 'one', 'plus', 'minus', 'i', 'minus-i', 'custom']
    state_group.add_argument('--state', type=str, choices=state_choices, required=True, help='Quantum state to prepare')
    state_group.add_argument('--custom-state', type=str, help='Custom state as "real,imag real,imag" (required if --state=custom)')
    
    # POVM selection
    povm_group = parser.add_argument_group('Measurement Options')
    povm_choices = ['bb84', 'sic', 'mub', 'trine', 'pvm', 'random']
    povm_group.add_argument('--povm', type=str, choices=povm_choices, required=True, help='POVM type for measurement')
    povm_group.add_argument('--shots', type=int, default=1024, help='Number of shots for the experiment')
    
    args = parser.parse_args()
    
    # Additional validation
    if args.noise_model == 'real' and not args.backend_name:
        parser.error("--backend-name is required when --noise-model=real")
    
    if args.state == 'custom' and not args.custom_state:
        parser.error("--custom-state is required when --state=custom")
    
    return args

def get_job_counts(job):
    """Extract counts from a completed job."""
    result = job.result()
    counts = result[0].data.c.get_counts()
    return counts

def get_job_results(job):
    """Extract experimental results from a completed job."""
    bit_array = next(iter(job.result()[0].data.values()))
    return bit_array.get_bitstrings()

def check_job_status_and_wait(job, backend, timeout_seconds=30):
    """
    Check job status and optionally wait for completion if results are immediate.
    
    Args:
        job: The job object
        backend: The backend object that submitted the job
        timeout_seconds (int): Maximum time to wait for immediate result jobs
        
    Returns:
        tuple: (is_completed, status_message)
    """
    try:
        status = job.status()
        
        if backend.provides_immediate_results():
            # For backends with immediate results, we expect immediate completion or very fast execution
            if str(status) == "JobStatus.DONE":
                return True, "Completed"
            else:
                return False, f"Immediate result job status: {status}"
        else:
            # For delayed result backends (real QPU), just check status without waiting
            if str(status) == "JobStatus.DONE":
                return True, "Completed"
            elif str(status) in ["JobStatus.QUEUED", "JobStatus.RUNNING"]:
                return False, f"Job queued/running: {status}"
            elif str(status) in ["JobStatus.ERROR", "JobStatus.CANCELLED"]:
                return False, f"Job failed: {status}"
            else:
                return False, f"Unknown status: {status}"
                
    except Exception as e:
        return False, f"Error checking status: {e}"

def generate_quantum_circuit_images(original_circuit, transpiled_circuit, experiment_folder):
    """
    Generate and save visualizations of both original and transpiled quantum circuits.
    
    Args:
        original_circuit: The original quantum circuit
        transpiled_circuit: The transpiled version of the circuit
        experiment_folder (str): Path to the folder where images should be saved
    """
    
    # Generate and save original circuit image
    try:
        original_fig = original_circuit.draw(output="mpl")
        original_path = os.path.join(experiment_folder, "quantum_circuit.png")
        original_fig.tight_layout()
        original_fig.savefig(original_path, dpi=300, bbox_inches='tight')
        plt.close(original_fig)
        print(f"Original circuit image saved to: {original_path}")
    except Exception as e:
        print(f"Failed to save original circuit image: {e}")
    
    # Generate and save transpiled circuit image
    # try:
    #     transpiled_fig = transpiled_circuit.draw(output="mpl")
    #     transpiled_path = os.path.join(experiment_folder, "quantum_circuit_transpiled.png")
    #     transpiled_fig.tight_layout()
    #     transpiled_fig.savefig(transpiled_path, dpi=300, bbox_inches='tight')
    #     plt.close(transpiled_fig)
    #     print(f"Transpiled circuit image saved to: {transpiled_path}")
    # except Exception as e:
    #     print(f"Failed to save transpiled circuit image: {e}")

def main():

    print("Quantum Experiment Runner")

    args = parse_arguments()
    
    # Create the quantum state
    state = create_state_from_args(args.state, args.custom_state)

    # Handle POVM selection
    povm = create_povm(args.povm)    
    
    # Handle backend creation
    backend = create_backend(args.backend_type, args)

    # Create the quantum circuit
    qc, prep_qubits = povm.create_circuit()

    state.prepare(qc, prep_qubits)
    print(f"Prepared state: {state.label}")

    # Add POVM measurement
    povm.prepare_measurement(qc)
    print(f"POVM measurement added: {povm.label}")    
    print(f"Circuit for {povm.label}:")
    print(qc)
    
    transpiled_circuit = transpile(qc, backend=backend.get_backend())

    job = backend.get_sampler().run([transpiled_circuit], shots=args.shots)
    print(f"Job ID: {job.job_id()}")
    
    # Step 1: Add experiment to master CSV and create subfolder
    # Always include custom noise parameters for reproducibility
    backend_obj = backend.get_backend()
    backend_name = getattr(backend_obj, 'name', str(backend_obj)) if backend_obj is not None else str(backend)
    initial_job_data = {
        'job_id': job.job_id(),
        'backend_type': args.backend_type,
        'backend_name': backend_name,
        'noise_model': getattr(args, 'noise_model', None),
        'shots': args.shots,
        'state': args.state,
        'custom_state': args.custom_state if args.state == 'custom' else None,
        'povm': args.povm,
        'povm_labels': str(list(povm.get_outcome_label_map().values())),
        **backend.get_mean_noise_errors(args)    
    }
    # Setup experiment: create folder and CSV manager in one call
    csv_manager, experiment_folder = ExperimentCSVManager.setup_experiment(
        output_dir="quantum/output", job_data=initial_job_data)

    generate_quantum_circuit_images(qc, transpiled_circuit, experiment_folder)

    state.generate_image(experiment_folder, filename_prefix="original")
    povm.generate_image(experiment_folder)
    
    # Initialize results processor
    results_processor = QuantumResultsProcessor(csv_manager)
    
    # Check if this backend provides immediate results or delayed results
    job_completed, status_message = check_job_status_and_wait(job, backend)
    
    print(f"Job status: {status_message}")
    
    if job_completed:
        # Process results immediately using centralized processor
        print("Processing results immediately...")
        
        # Pass the full initial_job_data to preserve all parameters
        success = results_processor.process_job_results(job, initial_job_data)
        
        if success:
            print(f"Experiment completed. Results saved to master CSV.")
            print(f"Experiment files can be stored in: {experiment_folder}")
        else:
            print("Failed to process experiment results.")
        
    else:
        # Job is queued or running (delayed results backend case)
        print("Job has been submitted to quantum hardware.")
        print(f"Job ID: {job.job_id()} - Status: {status_message}")
        print(f"Initial job data saved to master CSV.")
        print(f"Experiment folder created: {experiment_folder}")
        print("\nTo collect results when the job completes, run:")
        print("python collect_results.py")
        print("\nOr check job status periodically with:")
        print("python collect_results.py")

if __name__ == "__main__":
    main()