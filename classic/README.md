# Classic Simulation: Prepare-and-Measure Protocol

This folder contains the classical implementation of the prepare-and-measure (PM) protocol. The simulation is designed to emulate the generation and transmission of classical and quantum states using Python.

## Files

- `README.md`: This file, which provides an overview and instructions for using the classical simulation.
- `pm_simulation.py`: Main script to simulate the classical PM protocol.
- `state.py`: Utility module containing functions for generating random states.
- `requirements.txt`: File containing the dependencies required to run the simulation.

## Features

1. **Random State Generation**:
   - The `state.py` module includes a function `generate_random_state(dim)` that generates normalized random quantum states for a given dimension.
   - This can be used to simulate the preparation of quantum states in a classical context.

2. **Classical Protocol Simulation**:
   - The `pm_simulation.py` script simulates the transmission and measurement of states using classical techniques.
   - This includes:
     - State preparation.
     - Transmission through a classical channel.
     - Measurement and recovery of classical information.

## Prerequisites

- Python 3.8 or newer.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lsunol/transmitingaqubit.git

2. Navigate to the `classic` folder:
   ```bash
   cd transmitingaqubit/classic

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

1. Run the simulation
   ```bash
   python pm_simulation.py
