# Quantum Experiments Module

This module implements quantum state preparation and measurement experiments using different POVMs (Positive Operator-Valued Measures) on various quantum backends, including real IBM quantum computers and simulators.

## Overview

The quantum experiments system has two main entry points to handle the difference between simulator execution (immediate results) and real quantum hardware execution (queued jobs):

1. **`main.py`** - Job submission and execution
2. **`collect_results.py`** - Result collection from completed jobs
3. **`check_status.py`** - Status checking utility

## Architecture

### Two-Phase Execution Model

#### Phase 1: Job Submission (`main.py`)
- Creates quantum circuits with state preparation and POVM measurements
- Submits jobs to quantum backends (simulators or real hardware)
- For **simulators**: Completes full experiment with immediate results
- For **real QPU**: Queues job and saves initial metadata to CSV

#### Phase 2: Result Collection (`collect_results.py`)
- Checks status of pending jobs from real quantum hardware
- Collects results when jobs complete
- Processes experimental data and calculates KL divergence
- Updates master CSV with complete results

### Files and Components

#### Core Modules
- `main.py` - Main experiment runner and job submission
- `collect_results.py` - Result collection for queued jobs
- `check_status.py` - Job status checking utility
- `state.py` - Quantum state preparation classes
- `povm.py` - POVM measurement implementations
- `backend.py` - Quantum backend abstractions

#### Support Files
- `requirements.txt` - Python dependencies
- `output/` - Directory for experiment results and CSV files

## Usage

### Running Experiments

#### Basic Simulator Experiment
```bash
# Run experiment on AER simulator with immediate results
python main.py --backend-type aer_simulator --state "+" --povm bb84 --shots 1024
```

#### Real Quantum Hardware Experiment
```bash
# Submit job to real IBM quantum computer (requires authentication)
python main.py --backend-type qpu --backend-name aachen --state "0" --povm sic --shots 1024
```

#### Fake Backend (Realistic Simulation)
```bash
# Run on fake backend with noise model
python main.py --backend-type fake_backend --state "i" --povm mub --shots 512
```

### Collecting Results

After submitting jobs to real quantum hardware, use the collection script:

```bash
# Check and collect results from all pending jobs
python collect_results.py

# Verbose output for detailed status information
python collect_results.py --verbose

# Check specific output directory
python collect_results.py --output-dir custom/output/path
```

### Status Checking

Check job status without attempting result collection:

```bash
# Show all jobs
python check_status.py

# Show only pending jobs
python check_status.py --pending-only

# Check specific job
python check_status.py --job-id your-job-id-here
```

## Command Line Arguments

### Main Experiment Runner (`main.py`)

#### Backend Options
- `--backend-type` - Required. Choose: `qpu`, `fake_backend`, `aer_simulator`
- `--backend-name` - QPU name (required for qpu/real noise): `aachen`, `kingston`, `marrakesh`, etc.
- `--noise-model` - Noise model: `zero_noise`, `real`, `custom`

#### State Preparation
- `--state` - Required. Quantum state: `0`, `1`, `+`, `-`, `i`, `-i`, `custom`
- `--custom-state` - Custom state parameters (if `--state=custom`)

#### Measurement Options
- `--povm` - Required. POVM type: `bb84`, `sic`, `mub`, `random`
- `--shots` - Number of measurements (default: 1024)

#### Noise Parameters (for custom noise)
- `--readout-error` - Readout error probability
- `--gate-error` - Gate error probability
- `--thermal-relaxation` - Thermal relaxation probability

### Result Collection (`collect_results.py`)

- `--output-dir` - Output directory path (default: `quantum/output`)
- `--check-all` - Check all pending jobs, not just recent ones
- `--verbose` - Enable detailed output

### Status Checker (`check_status.py`)

- `--output-dir` - Output directory path (default: `quantum/output`)
- `--job-id` - Check specific job ID
- `--pending-only` - Show only jobs without results

## Examples

### Complete Workflow for Real Quantum Hardware

1. **Submit experiment to quantum computer:**
```bash
python main.py --backend-type qpu --backend-name aachen --state "+" --povm sic --shots 1024
```

2. **Check job status:**
```bash
python check_status.py --pending-only
```

3. **Collect results when ready:**
```bash
python collect_results.py --verbose
```

### Multiple Simulator Experiments
```bash
# Test different states with BB84 POVM
python main.py --backend-type aer_simulator --state "0" --povm bb84 --shots 1024
python main.py --backend-type aer_simulator --state "+" --povm bb84 --shots 1024
python main.py --backend-type aer_simulator --state "i" --povm bb84 --shots 1024

# Test different POVMs with |+⟩ state
python main.py --backend-type aer_simulator --state "+" --povm bb84 --shots 1024
python main.py --backend-type aer_simulator --state "+" --povm sic --shots 1024
python main.py --backend-type aer_simulator --state "+" --povm mub --shots 1024
```

## Output Files

### Master CSV (`output/experiments_master.csv`)
Contains all experiment metadata and results:
- Job information (ID, timestamp, backend details)
- Experiment parameters (state, POVM, shots)
- Results (counts, experimental data, KL divergence)

### Experiment Folders (`output/experiment_<job_id>/`)
Individual folders for each experiment to store additional files and analysis.

## Supported Quantum States

- **`0`** - |0⟩ state
- **`1`** - |1⟩ state  
- **`+`** - |+⟩ = (|0⟩ + |1⟩)/√2
- **`-`** - |-⟩ = (|0⟩ - |1⟩)/√2
- **`i`** - |i⟩ = (|0⟩ + i|1⟩)/√2
- **`-i`** - |-i⟩ = (|0⟩ - i|1⟩)/√2
- **`custom`** - Custom state (requires `--custom-state` parameter)

## Supported POVMs

- **`bb84`** - BB84 protocol measurements (Z or X basis)
- **`sic`** - Symmetric Informationally Complete POVM
- **`mub`** - Mutually Unbiased Bases measurements
- **`random`** - Random POVM selection

## Supported Backends

- **`aer_simulator`** - Qiskit Aer local simulator
- **`fake_backend`** - Fake backend with realistic noise models
- **`qpu`** - Real IBM quantum processors

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- qiskit
- qiskit_ibm_runtime
- qiskit_ibm_provider
- qiskit-aer
- numpy
- matplotlib

## IBM Quantum Setup

To use real quantum hardware (`--backend-type qpu`), you need:

1. IBM Quantum account and API token
2. Configure Qiskit with your credentials:
```bash
qiskit-ibm-provider accounts list  # Check existing accounts
qiskit-ibm-provider accounts set-token <your-token>  # Set token if needed
```

## Analysis and KL Divergence

The system automatically calculates Kullback-Leibler divergence between experimental and theoretical distributions for different shot counts, providing insight into:
- Convergence of experimental data to theoretical predictions
- Statistical quality of measurements
- Effect of finite sampling on POVM performance