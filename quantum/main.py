import argparse
import sys
import numpy as np
import csv
import os
from datetime import datetime
        
from qiskit import QuantumCircuit, transpile        
from state import create_state_from_args
from backend import create_backend
from povm import create_povm
from csv_manager import ExperimentCSVManager
from results_processor import QuantumResultsProcessor

def parse_arguments():
    """Parse command line arguments with no fallback to interactive mode."""
    parser = argparse.ArgumentParser(
        description='Prepare and measure quantum experiments runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # QPU or simulator selection
    qpu_group = parser.add_argument_group('QPU or Simulator Options')
    qpu_group.add_argument('--backend-type', type=str, choices=['qpu', 'fake_backend', 'aer_simulator'], required=True, help='QPU backend name or "simulator"')
    qpu_group.add_argument('--backend-name', type=str, required=False, 
                           choices=['least_busy', 'aachen', 'kingston', 'marrakesh', 'fez', 'torino', 'sherbrooke', 'quebec', 'brisbane', 'kawasaki', 'rensselaer', 'brussels', 'strasbourg'], 
                           help='Specific backend name (required if --backend_type=qpu or --noise-model=real)')
    qpu_group.add_argument('--noise-model', type=str, choices=['zero_noise', 'real', 'custom'], default='real', required=False, help='Noise model to use with aer_simulator (default: real)')
    
    # Custom noise parameters (only relevant if --noise-model=custom)
    noise_group = parser.add_argument_group('Custom Noise Parameters')
    noise_group.add_argument('--readout-error', type=float, default=0.01, help='Readout error probability')
    noise_group.add_argument('--gate-error', type=float, default=0.001, help='Gate error probability')
    noise_group.add_argument('--thermal-relaxation', type=float, default=0.01, help='Thermal relaxation probability')
    
    # State preparation
    state_group = parser.add_argument_group('Initial qubit preparation options')
    state_choices = ['0', '1', '+', '-', 'i', '-i', 'custom']
    state_group.add_argument('--state', type=str, choices=state_choices, required=True, help='Quantum state to prepare')
    state_group.add_argument('--custom-state', type=str, help='Custom state as "real,imag real,imag" (required if --state=custom)')
    
    # POVM selection
    povm_group = parser.add_argument_group('Measurement Options')
    povm_choices = ['bb84', 'sic', 'mub', 'random']
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

def create_experiment_subfolder(job_id, output_dir="quantum/output"):
    """
    Create a subfolder for the current experiment to store related files.
    
    Args:
        job_id (str): The job ID for this experiment
        output_dir (str): Base output directory
    
    Returns:
        str: Path to the created experiment subfolder
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create experiment-specific subfolder
    experiment_folder = os.path.join(output_dir, f"experiment_{job_id}")
    os.makedirs(experiment_folder, exist_ok=True)
    
    print(f"Created experiment subfolder: {experiment_folder}")
    return experiment_folder

def add_experiment_to_master_csv(job_data, output_dir="quantum/output"):
    """
    Add initial experiment data to the master CSV file.
    
    Args:
        job_data (dict): Dictionary containing initial job information
        output_dir (str): Directory containing the master CSV file
    
    Returns:
        tuple: (csv_manager, experiment_folder_path)
    """
    # Initialize CSV manager
    csv_manager = ExperimentCSVManager(output_dir)
    
    # Create experiment subfolder
    experiment_folder = create_experiment_subfolder(job_data['job_id'], output_dir)
    
    # Create experiment record
    experiment_record = csv_manager.create_experiment_record(
        job_id=job_data['job_id'],
        experiment_folder=f"experiment_{job_data['job_id']}",
        backend_type=job_data['backend_type'],
        backend_name=job_data.get('backend_name', ''),
        noise_model=job_data.get('noise_model', ''),
        state=job_data['state'],
        povm=job_data['povm'],
        total_shots=job_data['shots'],
        povm_labels=job_data.get('povm_labels', '')
    )
    
    # Add to CSV
    csv_manager.append_experiment_data(experiment_record)
    
    return csv_manager, experiment_folder

def main():

    print("Quantum Experiment Runner")

    args = parse_arguments()
    
    # Create the quantum state
    state = create_state_from_args(args.state, args.custom_state)

    # Handle POVM selection
    povm = create_povm(args.povm)    
    
    # Handle backend creation
    backend_kwargs = {}
    if hasattr(args, 'backend_name') and args.backend_name:
        backend_kwargs['backend_name'] = args.backend_name
    backend = create_backend(args.backend_type, **backend_kwargs)

    # Create the quantum circuit
    qc = povm.create_circuit()

    state.prepare(qc)

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
    initial_job_data = {
        'job_id': job.job_id(),
        'backend_type': args.backend_type,
        'backend_name': backend.get_backend().name,
        'noise_model': getattr(args, 'noise_model', None),
        'state': args.state,
        'povm': args.povm,
        'povm_labels': str(list(povm.get_outcome_label_map().values())),
        'shots': args.shots
    }
    csv_manager, experiment_folder = add_experiment_to_master_csv(initial_job_data)

    state.generate_image(experiment_folder)
    
    # Initialize results processor
    results_processor = QuantumResultsProcessor(csv_manager)
    
    # Check if this backend provides immediate results or delayed results
    job_completed, status_message = check_job_status_and_wait(job, backend)
    
    print(f"Job status: {status_message}")
    
    if job_completed:
        # Process results immediately using centralized processor
        print("Processing results immediately...")
        
        # Create job data for processor (similar to CSV row format)
        job_data_for_processor = {
            'job_id': job.job_id(),
            'state': args.state,
            'povm': args.povm
        }
        
        success = results_processor.process_job_results(job, job_data_for_processor)
        
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