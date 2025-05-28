import argparse
import sys
import numpy as np
        
from qiskit import QuantumCircuit, transpile        
from state import create_state
from backend import create_backend
from povm import create_povm

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
                           choices=['aachen', 'kingston', 'marrakesh', 'fez', 'torino', 'sherbrooke', 'quebec', 'brisbane', 'kawasaki', 'rensselaer', 'brussels', 'strasbourg'], 
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

def validate_custom_state(custom_state_str):
    """Validate and parse a custom state string."""
    try:
        parts = custom_state_str.split()
        if len(parts) != 2:
            raise ValueError("Custom state must have exactly two parts (alpha and beta)")
        
        alpha_parts = parts[0].split(',')
        beta_parts = parts[1].split(',')
        
        if len(alpha_parts) != 2 or len(beta_parts) != 2:
            raise ValueError("Each amplitude must have real and imaginary parts separated by comma")
        
        alpha = complex(float(alpha_parts[0]), float(alpha_parts[1]))
        beta = complex(float(beta_parts[0]), float(beta_parts[1]))
        
        # Normalize
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        if norm == 0:
            raise ValueError("State vector cannot be zero")
        
        alpha /= norm
        beta /= norm
        
        return f"{alpha.real},{alpha.imag},{beta.real},{beta.imag}"
    except Exception as e:
        raise ValueError(f"Invalid custom state format: {e}")

# Placeholder for future implementation
def run_experiment(params):
    """Run the quantum experiment with the specified parameters.
    
    This is a placeholder that will be implemented with quantum computing libraries.
    """
    print("\nThis is a placeholder for the experiment implementation.")
    print("The experiment would run with these parameters:")
    for key, value in params.items():
        print(f"- {key}: {value}")
    
    # Return mock results
    return {"results": "Experiment implementation pending"}

def get_job_counts(job):
    result = job.result()
    counts = result[0].data.c.get_counts()
    return counts

def get_job_results(job):
    bit_array = next(iter(job.result()[0].data.values()))
    return bit_array.get_bitstrings()

def main():
    args = parse_arguments()
    
    # Create the quantum state

    # Handle state preparation
    if args.state == 'custom':
        print(f"Not yet implemented: Custom state preparation with parameters: {args.custom_state}")
        exit(1)

    state = create_state(args.state)

    # Handle POVM selection
    povm = create_povm(args.povm) 

    # Handle backend creation
    backend = create_backend(args.backend_type)

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

    counts = get_job_counts(job)
    print(f"Counts: {counts}")
    experimental_results = get_job_results(job)

    kl_analysis = povm.calculate_kl_divergence(experimental_results, state)
    print(f"KL Divergence: {kl_analysis}")

    # Run the experiment
    # results = run_experiment(experiment_params)
    

if __name__ == "__main__":
    main()
