# **Quantum Bit Transmission Cost Simulation**

This repository aims to explore and test the classical cost of transmitting a qubit, following protocols such as **Prepare-and-Measure (PM)** and **quantum teleportation**. The goal is to implement both **classical** and **quantum** simulations to evaluate and compare the resources required for quantum information transmission.

## **Repository Structure**

The project is organized as follows:
```
/
├── classic/
│   ├── README.md  # Description and instructions for the classical simulation
│   └── pm_simulation.py  # Classical prepare-and-measure simulation
├── quantum/
│   ├── README.md  # Description and instructions for the quantum simulation
│   └── pm_quantum.qasm  # Qiskit/OpenQASM implementation of PM protocol
└── README.md  # Project overview and setup instructions (this file)
```

## **Project Objectives**
1. **Understand the Prepare-and-Measure (PM) Protocol**:
   - Study how a qubit's state is "prepared" by Alice and "measured" by Bob.
   - Simulate the protocol classically to determine the bit cost of reproducing qubit behavior.

2. **Simulate Quantum Teleportation**:
   - Implement quantum teleportation to test how quantum information can be reconstructed.
   - Compare resource requirements (qubits, classical bits) between classical and quantum approaches.

3. **Compare Classical and Quantum Results**:
   - Evaluate the number of classical bits needed to simulate qubit-based correlations.
   - Identify trade-offs and practical limitations in both paradigms.

## **Setup Instructions**

### **Prerequisites**
- Python 3.8+ (for classical simulations).
- Libraries: `numpy`, `scipy`, `matplotlib` (for data analysis).
- **Qiskit** (for quantum circuit simulations).

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/username/quantum-bit-transmission-cost-simulation.git
2. Navigate to the project folder:
   ```bash
   cd quantum-bit-transmission-cost-simulation
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Contents of the Repository

### Classic Folder
Contains Python scripts for classical simulations:
* *pm_simulation.py*: Implements the PM protocol using classical random variables.

### Quantum Folder
Contains Qiskit scripts for quantum simulations:
* *pm_quantum.qasm*: Implements the PM protocol using quantum circuits.
