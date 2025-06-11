from __future__ import annotations
"""
POVM module for quantum measurements.
This module contains classes to implement different POVMs using quantum circuits.
"""

from abc import ABC, abstractmethod
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate
import qiskit.quantum_info as qi
import math
from scipy.linalg import qr
import qutip as qt
import os
import state as state_module
from state import State


class POVM(ABC):
    """
    Abstract base class for quantum POVMs.
    
    This class defines the interface for all POVMs implementations.
    Each POVM must be able to create the initial circuit with appropriate number of qubits
    and apply measurement operations to a circuit where the state has been prepared.
    """
    
    def __init__(self, label=None):
        """Initialize POVM with an optional label."""
        self.label = label if label else "Unnamed POVM"
        
    @abstractmethod
    def create_circuit(self):
        """
        Create an empty quantum circuit with the appropriate number of qubits and classical bits.
        
        Each POVM implementation will create a circuit with exactly the required number of qubits 
        and classical bits for its specific measurement type. The circuit will have qubit 0 
        available for state preparation.
        
        Returns:
            QuantumCircuit: Empty circuit with correct number of qubits and classical bits.
        """
        pass
    
    @abstractmethod
    def prepare_measurement(self, qc):
        """
        Apply the POVM measurement operations to a quantum circuit.
        
        This method assumes that the state has already been prepared on qubit 0.
        
        Args:
            qc (QuantumCircuit): The quantum circuit with the prepared state.
        
        Returns:
            tuple: (QuantumCircuit, list) - The circuit with measurement operations added
                                          and a list of outcome labels.
        """
        pass
    
    @abstractmethod
    def get_operators(self):
        """
        Return the theoretical POVM operators.
        
        Returns:
            list: List of numpy arrays representing the POVM operators.
        """
        pass

    @abstractmethod
    def get_outcome_label_map(self):
        """
        Return a dictionary mapping measurement outcomes (bitstrings) to operator labels.
        Returns:
            dict: Mapping from outcome bitstrings (e.g., '00', '1') to operator labels (e.g., 'SIC0', '+', etc.)
        """        

    def get_theoretical_distribution(self, input_state: "State"):
        """
        Calculate the theoretical probability distribution for this POVM on the given input state using the Born rule.
        Args:
            input_state (State): An instance of State (from state.py) with get_density_matrix().
        Returns:
            dict: Dictionary mapping outcome bitstrings to their theoretical probabilities.
        Raises:
            TypeError: If input_state is not an instance of State.
        """
        operators = self.get_operators()
        outcome_map = self.get_outcome_label_map()
        if outcome_map is None:
            raise ValueError("Outcome label map not implemented for this POVM")
        outcome_keys = list(outcome_map.keys())

        # Always use get_density_matrix from State
        rho = input_state.get_density_matrix()

        # Calculate Born rule probabilities
        probs = {}
        for i, M in enumerate(operators):
            key = outcome_keys[i] if i < len(outcome_keys) else str(i)
            prob = np.real(np.trace(M @ rho))
            probs[key] = float(prob)

        # Normalize to ensure sum to 1 (for numerical stability)
        total = sum(probs.values())
        if total > 0:
            for k in probs:
                probs[k] /= total
        return probs

    def calculate_kl_divergence(self, experimental_results, input_state):
        """
        Calculate KL divergence as the number of shots increases, starting from 10 shots.
        
        Args:
            experimental_results (list): List of measurement results as strings, e.g., ['00', '10', '00', '11', '01', '10', ...]
            input_state: A State object with get_bloch_angles method.
        
        Returns:
            list: List of objects with the structure:
                  [
                      {'shots': 10, 'kl_divergence': kl_div_10},
                      {'shots': 11, 'kl_divergence': kl_div_11},
                      {'shots': 12, 'kl_divergence': kl_div_12},
                      ...
                  ]
        """
        if len(experimental_results) < 10:
            raise ValueError("Need at least 10 experimental results to start KL divergence calculation")
        
        # Get theoretical distribution for the input state
        theoretical_probs = self.get_theoretical_distribution(input_state)
        
        results = []
        
        # Iterate from 10 shots to the total length of experimental_results
        for n_shots in range(10, len(experimental_results) + 1):
            # Get subset of results up to n_shots
            current_results = experimental_results[:n_shots]
            
            # Count occurrences of each outcome
            experimental_counts = {}
            for outcome in current_results:
                experimental_counts[outcome] = experimental_counts.get(outcome, 0) + 1
            
            # Convert counts to probabilities
            experimental_probs = {}
            for outcome, count in experimental_counts.items():
                experimental_probs[outcome] = count / n_shots
            
            # Ensure all possible outcomes are represented (with 0 probability if not observed)
            outcome_map = self.get_outcome_label_map()
            if outcome_map is None:
                raise ValueError("Outcome label map not implemented for this POVM")
            all_outcomes = set(outcome_map.keys())
            
            for outcome in all_outcomes:
                if outcome not in experimental_probs:
                    experimental_probs[outcome] = 0.0
            
            # Calculate KL divergence using the provided method
            try:
                kl_div = self.kl_divergence(experimental_probs, theoretical_probs)
                results.append({
                    'shots': n_shots,
                    'kl_divergence': kl_div
                })
            except ValueError as e:
                # If there's an issue with the distributions, skip this point
                print(f"Warning: Skipping n_shots={n_shots} due to error: {e}")
                continue
        
        return results

    def kl_divergence(self, P, Q, epsilon=1e-12, tolerance=1e-6):
        """
        Calculates the KL divergence between two distributions P and Q.
        Both must be arrays or dicts representing probability distributions that sum to 1.

        Args:
            P: First probability distribution (array or dict)
            Q: Second probability distribution (array or dict)
            epsilon: Small value to avoid log(0)
            tolerance: Tolerance for sum-to-one validation

        Returns:
            KL divergence value

        Raises:
            ValueError: If P or Q don't sum to 1 within tolerance
        """
        # Handle dictionary inputs by converting to arrays
        if isinstance(P, dict) and isinstance(Q, dict):
            # Ensure both dictionaries have the same keys
            all_keys = set(P.keys()) | set(Q.keys())
            P_array = np.array([P.get(k, 0) for k in all_keys], dtype=np.float64)
            Q_array = np.array([Q.get(k, 0) for k in all_keys], dtype=np.float64)
        else:
            P_array = np.array(P, dtype=np.float64)
            Q_array = np.array(Q, dtype=np.float64)

        # Validate that P and Q are probability distributions (sum to 1)
        p_sum = np.sum(P_array)
        q_sum = np.sum(Q_array)

        if abs(p_sum - 1.0) > tolerance:
            raise ValueError(f"P does not sum to 1. Sum: {p_sum}")
        if abs(q_sum - 1.0) > tolerance:
            raise ValueError(f"Q does not sum to 1. Sum: {q_sum}")

        # Add epsilon to avoid issues with log(0)
        P_array = np.clip(P_array, epsilon, 1)
        Q_array = np.clip(Q_array, epsilon, 1)

        # Normalize to ensure sum is exactly 1.0
        P_array /= np.sum(P_array)
        Q_array /= np.sum(Q_array)

        return np.sum(P_array * np.log(P_array / Q_array))

    def generate_image(self, output_dir):
        """
        Generate and save a PNG image of the POVM operators on a Bloch sphere using QuTiP.
        
        Args:
            output_dir (str): Directory path where the PNG will be saved
        
        Returns:
            str: Full path to the saved PNG file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the full file path with fixed filename
        output_path = os.path.join(output_dir, "povm.png")
        
        # Create Bloch sphere and add the POVM operators
        bloch = qt.Bloch()
        operators = self.get_operators()
        
        # Check if operators are available
        if operators is not None:
            # Get operator labels from the outcome label map
            outcome_map = self.get_outcome_label_map()
            if outcome_map is not None:
                operator_labels = list(outcome_map.values())
            else:
                # Fallback to generic labels if no outcome map
                operator_labels = [f"M{i}" for i in range(len(operators))]
            
            # Calculate Bloch vectors for each operator
            bloch_vectors = []
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
            
            for i, operator in enumerate(operators):
                # Calculate Bloch vector components for qubit operators
                if operator.shape == (2, 2):  # Only for qubit operators
                    # Pauli matrices as numpy arrays
                    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
                    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
                    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
                    
                    # Calculate Bloch vector components using trace
                    x = np.trace(operator @ sigma_x).real
                    y = np.trace(operator @ sigma_y).real  
                    z = np.trace(operator @ sigma_z).real
                    
                    bloch_vectors.append([x, y, z])
            
            # Add vectors to Bloch sphere with different colors
            if bloch_vectors:
                for i, vec in enumerate(bloch_vectors):
                    color = colors[i % len(colors)]
                    bloch.add_vectors([vec], [color])
                    
                    # Add annotations at the end of each vector
                    if i < len(operator_labels):
                        label = operator_labels[i]
                        bloch.add_annotation(vec, label, color=color, fontsize=12, ha='center', va='center')
          # Save as PNG
        bloch.save(output_path)
        return output_path

    def get_label(self):
        """Return the label of the POVM."""
        return self.label

    def postprocess_results(self, experimental_results, counts=None):
        """
        Optional post-processing for experimental results and counts.
        By default, returns the input unchanged.
        Args:
            experimental_results (list or dict): Raw results from experiment (bitstrings or counts)
            counts (dict, optional): Raw counts from experiment
        Returns:
            tuple: (processed_results, processed_counts)
        """
        return experimental_results, counts

    def reconstruct_original_state(self, experimental_distribution, output_dir):
        """
        Generic state reconstruction from experimental distribution using this POVM (qubit case).
        Uses linear inversion: p_i = Tr(M_i rho), with rho expanded in the Pauli basis.
        Returns a State.Custom object and saves a Bloch image.
        Args:
            experimental_distribution (dict): Bitstring->count or probability (will be normalized)
            output_dir (str): Directory to save the image
        Returns:
            State.Custom: The reconstructed state
            str: Path to the saved image
        """
        import warnings
        operators = self.get_operators()
        outcome_map = self.get_outcome_label_map()
        if operators is None or outcome_map is None:
            raise NotImplementedError("POVM must implement get_operators and get_outcome_label_map for reconstruction.")
        if not hasattr(operators, '__iter__') or isinstance(operators, (str, bytes)):
            raise TypeError("get_operators() must return an iterable of operator matrices.")
        operators = list(operators)
        # Normalize experimental distribution
        keys = list(outcome_map.keys())
        total = sum(experimental_distribution.get(k, 0) for k in keys)
        if total == 0:
            raise ValueError("Experimental distribution is empty or all zero.")
        p = np.array([experimental_distribution.get(k, 0) / total for k in keys], dtype=float)
        # Pauli basis: I, X, Y, Z
        paulis = [np.eye(2, dtype=complex),
                  np.array([[0, 1], [1, 0]], dtype=complex),
                  np.array([[0, -1j], [1j, 0]], dtype=complex),
                  np.array([[1, 0], [0, -1]], dtype=complex)]
        # Each operator: Tr(M_i * P_j)
        A = np.zeros((len(operators), 4), dtype=float)
        for i, M in enumerate(operators):
            for j, P in enumerate(paulis):
                A[i, j] = np.real(np.trace(M @ P))
        # Solve A x = p for x = [a0, a1, a2, a3] (a0: I, a1: X, a2: Y, a3: Z)
        # a0 should be 1 for physical states (Tr(rho) = 1)
        # Use least squares in case of over/under-determined system
        x, residuals, rank, s = np.linalg.lstsq(A, p, rcond=None)
        # Warn if the POVM is not informationally complete
        if rank < 4:
            warnings.warn("POVM is not informationally complete; reconstructed state may not be unique.")
        # Build density matrix
        rho = 0.5 * (x[0] * paulis[0] + x[1] * paulis[1] + x[2] * paulis[2] + x[3] * paulis[3])
        # Project to physical state if needed (Hermitian, positive semidefinite, trace 1)
        # Force Hermiticity
        rho = 0.5 * (rho + rho.conj().T)
        # Eigen-decompose and zero out negative eigenvalues
        eigvals, eigvecs = np.linalg.eigh(rho)
        eigvals = np.clip(eigvals, 0, None)
        rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        rho /= np.trace(rho)
        # Extract Bloch vector
        x_bloch = np.real(np.trace(rho @ paulis[1]))
        y_bloch = np.real(np.trace(rho @ paulis[2]))
        z_bloch = np.real(np.trace(rho @ paulis[3]))
        r = np.sqrt(x_bloch**2 + y_bloch**2 + z_bloch**2)
        if r < 1e-10:
            theta = 0.0
            phi_angle = 0.0
        else:
            theta = np.arccos(np.clip(z_bloch / r, -1, 1))
            phi_angle = np.arctan2(y_bloch, x_bloch)
        reconstructed_state = state_module.Custom(theta, phi_angle, label="|ψ_reconstructed⟩")
        output_path = reconstructed_state.generate_image(output_dir, filename_prefix="reconstructed")
        return reconstructed_state, output_path
    

class BB84POVM(POVM):
    """
    BB84 POVM implementation.
    
    This POVM measures in either the computational (Z) basis or the diagonal (X) basis.
    """
    
    def __init__(self, basis='random'):
        """
        Initialize a BB84 POVM.
        
        Args:
            basis (str): Basis choice ('z', 'x', or 'random'). Default is 'random'.
        """
        super().__init__(label="BB84 POVM")
        self.basis_options = ['z', 'x']
        if basis == 'random':
            self.basis = np.random.choice(self.basis_options)
        elif basis in self.basis_options:
            self.basis = basis
        else:
            raise ValueError(f"Basis must be one of {self.basis_options} or 'random'")
    
    def create_circuit(self):
        """Create a quantum circuit for BB84 POVM measurement. Returns (qc, prep_qubits)."""
        qc = QuantumCircuit(1, 1)
        prep_qubits = [0]
        return qc, prep_qubits
    
    def prepare_measurement(self, qc):
        """Apply BB84 POVM measurement to the circuit."""
        if self.basis == 'x':
            # Measure in X basis by applying H before measurement
            qc.h(0)
        
        # Add measurement
        qc.measure(0, 0)
        
        # Return the circuit and outcome labels
        outcome_labels = ['0', '1']
        return qc, outcome_labels
    
    def get_operators(self):
        """Return the theoretical BB84 POVM operators."""
        # Define the POVM operators based on the basis
        if self.basis == 'z':
            # Computational basis
            M0 = np.array([[1, 0], [0, 0]])
            M1 = np.array([[0, 0], [0, 1]])
        else:  # X basis
            # Diagonal basis
            M_plus = np.array([[0.5, 0.5], [0.5, 0.5]])
            M_minus = np.array([[0.5, -0.5], [-0.5, 0.5]])
            M0 = M_plus
            M1 = M_minus
        
        return [M0, M1]

    def get_outcome_label_map(self):
        # BB84: outcome '0' or '1' maps to '0' or '1' (Z basis) or '+'/'-' (X basis)
        if self.basis == 'z':
            return {'0': '0', '1': '1'}
        else:
            return {'0': '+', '1': '-'}


class SICPOVM(POVM):
    """
    Symmetric Informationally Complete (SIC) POVM implementation.
    
    This POVM consists of 4 operators corresponding to the vertices 
    of a tetrahedron in the Bloch sphere.
    """
    
    def __init__(self):
        super().__init__(label="SIC POVM")
        # Define the SIC-POVM state vectors (as column vectors)
        self.psi_list = [
            np.array([[1], [0]]),
            (1/np.sqrt(3)) * np.array([[1], [np.sqrt(2)]]),
            (1/np.sqrt(3)) * np.array([[1], [np.sqrt(2) * np.exp(2j * np.pi / 3)]]),
            (1/np.sqrt(3)) * np.array([[1], [np.sqrt(2) * np.exp(4j * np.pi / 3)]])
        ]
        # For compatibility, keep self.vectors for other usages
        self.vectors = self.psi_list

    def create_circuit(self):
        """Create a quantum circuit for SIC POVM measurement. Returns (qc, prep_qubits)."""
        qc = QuantumCircuit(3, 2)
        prep_qubits = [0]
        return qc, prep_qubits
    
    def prepare_measurement(self, qc):
        """Apply SIC POVM measurement to the circuit."""
        # SIC POVM implementation using ancilla qubits
        # The idea is to map the 4 SIC states to computational basis states
        # of a 2-qubit system (which gives us 4 possible outcomes)
        # Apply a unitary that approximates mapping the SIC POVM to computational basis
        # This is an approximate implementation that provides good discrimination
        
        sqrt_half = 1 / np.sqrt(2)
        K_list = []
        for phi in self.vectors:
            K = sqrt_half * phi @ phi.T.conj()
            K_list.append(K)

        U = np.zeros((8, 8), dtype=complex)
        U[0:2, 0:2] = K_list[0]  # '00' → B0
        U[2:4, 0:2] = K_list[1]  # '01' → B1
        U[4:6, 0:2] = K_list[2]  # '10' → B2
        U[6:8, 0:2] = K_list[3]  # '11' → B3

        # Perform QR decomposition and ensure we get the right types
        qr_result = qr(U)
        Q = qr_result[0]  # Extract Q matrix explicitly
        U_unitary = np.array(Q, dtype=complex)  # Ensure it's a numpy array with complex dtype

        U_gate = UnitaryGate(U_unitary, label='U-SIC')
        qc.append(U_gate, [0, 1, 2])
        
        # Measure the ancilla qubits to get the SIC POVM outcomes
        qc.measure(1, 0)
        qc.measure(2, 1)
        
        # Define outcome labels for the 4 possible measurement results
        outcome_labels = ['SIC0', 'SIC1', 'SIC2', 'SIC3']
        
        return qc, outcome_labels
    
    def get_operators(self):
        """Return the theoretical SIC POVM operators."""
        operators = []
        
        # Factor for normalization (1/4 for 2D SIC POVMs)
        factor = 1/4
        
        for vec in self.psi_list:
            # Convert state vector to projector: |ψ⟩⟨ψ|
            projector = vec @ vec.conj().T
            
            # Scale by factor for POVM
            operator = factor * projector
            operators.append(operator)
        
        return operators

    def get_outcome_label_map(self):
        # SIC: outcome '00', '01', '10', '11' map to SIC0, SIC1, SIC2, SIC3
        return {'00': 'SIC0', '01': 'SIC1', '10': 'SIC2', '11': 'SIC3'}    
    
    def reconstruct_original_state(self, experimental_distribution, output_dir):
        """
        Reconstruct the original quantum state from the experimental SIC-POVM distribution.
        Returns a State.Custom and saves a Bloch image with the given prefix.
        Minimal, direct SIC-POVM reconstruction: ρ = sum_i (3p_i - 0.5) |ψ_i⟩⟨ψ_i|.
        Extract Bloch vector directly from ρ, map to State.Custom.
        """
        ordered_keys = ['00', '01', '10', '11']
        total = sum(experimental_distribution.get(k, 0) for k in ordered_keys)
        if total == 0:
            raise ValueError("Experimental distribution is empty or all zero.")
        p = [experimental_distribution.get(k, 0) / total for k in ordered_keys]
        # Minimal SIC-POVM state reconstruction
        rho = np.zeros((2, 2), dtype=complex)
        for p_i, phi in zip(p, self.psi_list):
            ket = phi
            rho += (3 * p_i - 0.5) * (ket @ ket.conj().T)
        # Extract Bloch vector: x = Tr(rho σ_x), y = Tr(rho σ_y), z = Tr(rho σ_z)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        x = np.real(np.trace(rho @ sigma_x))
        y = np.real(np.trace(rho @ sigma_y))
        z = np.real(np.trace(rho @ sigma_z))
        # Convert to Bloch angles
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1e-10:
            theta = 0.0
            phi_angle = 0.0
        else:
            theta = np.arccos(np.clip(z / r, -1, 1))
            phi_angle = np.arctan2(y, x)
        # State.Custom expects angles in radians
        reconstructed_state = state_module.Custom(theta, phi_angle, label="|ψ_reconstructed⟩")
        output_path = reconstructed_state.generate_image(output_dir, filename_prefix="reconstructed")
        return reconstructed_state, output_path
    

class MUBPOVM(POVM):
    """
    Mutually Unbiased Bases (MUB) POVM implementation.
    
    This POVM measures in all three mutually unbiased bases
    for a qubit: Z, X, and Y, using three qubits in parallel.
    """
    
    def __init__(self):
        """
        Initialize a MUB POVM (parallel measurement in all three bases).
        """
        super().__init__(label="MUB POVM (parallel)")

    def create_circuit(self):
        """Create a quantum circuit for MUB POVM measurement (3 qubits, 3 bits)."""
        qc = QuantumCircuit(3, 3)
        prep_qubits = [0, 1, 2]
        return qc, prep_qubits

    def prepare_measurement(self, qc):
        """Apply MUB POVM measurement to the circuit (all three bases in parallel)."""
        # Qubit 0: Z basis (no rotation)
        # Qubit 1: X basis (H)
        qc.h(1)
        # Qubit 2: Y basis (S† then H)
        qc.sdg(2)
        qc.h(2)
        # Measure all three qubits into their respective classical bits
        qc.measure(0, 0)  # Z
        qc.measure(1, 1)  # X
        qc.measure(2, 2)  # Y
        # Outcome labels for each basis
        outcome_labels = [
            ['0', '1'],    # Z
            ['+', '-'],    # X
            ['+i', '-i']   # Y
        ]
        return qc, outcome_labels

    def get_operators(self):
        """Return the theoretical MUB POVM operators as a flat list: [M0_z, M1_z, M0_x, M1_x, M0_y, M1_y]."""
        z_ops = [
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 1]])
        ]
        x_ops = [
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.array([[0.5, -0.5], [-0.5, 0.5]])
        ]
        y_ops = [
            np.array([[0.5, -0.5j], [0.5j, 0.5]]),
            np.array([[0.5, 0.5j], [-0.5j, 0.5]])
        ]
        return z_ops + x_ops + y_ops

    def get_outcome_label_map(self):
        # Flat mapping for all 6 outcomes, order matches get_operators
        return {
            '0z': '0', '1z': '1',
            '0x': '+', '1x': '-',
            '0y': '+i', '1y': '-i'
        }

    def postprocess_results(self, experimental_results, counts=None):
        """
        For MUBPOVM: Convert list or dict of 3-bit results (e.g., '010') into flat POVM outcome list and counts.
        Args:
            experimental_results (list or dict): List of bitstrings or dict of bitstring->count
            counts (dict, optional): Raw counts from experiment
        Returns:
            tuple: (results, counts) where results is a flat list like ['0z', '1z', '0x', ...]
        """
        # Always regenerate both results (list) and counts (dict)

        bitstring_list = list(experimental_results)
        # Aggregate counts for each basis outcome and build flat results list
        results = []
        processed_counts = {'0z': 0, '1z': 0, '0x': 0, '1x': 0, '0y': 0, '1y': 0}
        for bitstr in bitstring_list:
            y, x, z = bitstr
            for label, bit in zip(['z', 'x', 'y'], [z, x, y]):
                povm_outcome = f'{bit}{label}'
                results.append(povm_outcome)
                processed_counts[povm_outcome] += 1
        return results, processed_counts

    def get_label(self):
        return self.label
    

class TRINEPOVM(POVM):
    """
    TRINE POVM implementation for a qubit.
    The TRINE POVM consists of 3 operators corresponding to three equally spaced states in the equatorial plane of the Bloch sphere (at 0°, 120°, 240°).
    """
    def __init__(self):
        super().__init__(label="TRINE POVM")
        # Trine state vectors (in the X-Y plane)
        self.trine_angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        self.psi_list = [
            np.array([[1], [np.exp(1j * angle)]]) / np.sqrt(2)
            for angle in self.trine_angles
        ]
        self.vectors = self.psi_list

    def create_circuit(self):
        """Create a quantum circuit for TRINE POVM measurement. Returns (qc, prep_qubits)."""
        qc = QuantumCircuit(2, 2)
        prep_qubits = [0]
        return qc, prep_qubits

    def prepare_measurement(self, qc):
        """Apply TRINE POVM measurement to the circuit using an explicit Naimark dilation unitary."""
        # Explicit Naimark dilation for TRINE POVM
        # See e.g. https://arxiv.org/abs/quant-ph/0407010
        # The three trine states are mapped to |00>, |01>, |10>
        import numpy as np
        from qiskit.circuit.library import UnitaryGate
        from scipy.linalg import qr

        # Define the three trine states as columns
        omega = np.exp(2j * np.pi / 3)
        psi0 = np.array([1, 1]) / np.sqrt(2)
        psi1 = np.array([1, omega]) / np.sqrt(2)
        psi2 = np.array([1, omega.conjugate()]) / np.sqrt(2)
        # Build the 4x3 isometry V
        V = np.zeros((4, 3), dtype=complex)
        V[0, 0] = psi0[0]  # |00> <- |0>
        V[1, 0] = psi0[1]  # |01> <- |1>
        V[0, 1] = psi1[0]
        V[1, 1] = psi1[1]
        V[0, 2] = psi2[0]
        V[1, 2] = psi2[1]
        # Complete to a 4x4 unitary
        # Add a fourth orthonormal column to make it square
        extra_col = np.zeros((4, 1), dtype=complex)
        extra_col[3, 0] = 1.0
        full_mat = np.hstack([V, extra_col])
        Q = qr(full_mat)[0]  # Only take the Q matrix
        U = np.array(Q, dtype=complex)  # Ensure it's a numpy array
        # Apply as a 2-qubit unitary
        U_gate = UnitaryGate(U, label='U-TRINE-EXPL')
        qc.append(U_gate, [0, 1])
        # Measure both qubits
        qc.measure(0, 0)
        qc.measure(1, 1)
        # Use outcome labels from get_outcome_label_map
        outcome_labels = list(self.get_outcome_label_map().values())
        return qc, outcome_labels

    def get_operators(self):
        """Return the theoretical TRINE POVM operators."""
        # Each operator: (2/3) * |psi_i><psi_i|
        factor = 2/3
        operators = [factor * (psi @ psi.conj().T) for psi in self.psi_list]
        return operators

    def get_outcome_label_map(self):
        # Only map the three valid outcomes to TRINE labels (no 'null' outcome)
        return {'00': 'TRINE0', '01': 'TRINE1', '10': 'TRINE2'}


class PVMPOVM(POVM):
    """
    Projective measurement (PVM) in the computational basis (|0⟩, |1⟩).
    """
    def __init__(self):
        super().__init__(label="PVM POVM")
        # Computational basis vectors
        self.psi_list = [
            np.array([[1], [0]]),  # |0⟩
            np.array([[0], [1]])   # |1⟩
        ]
        self.vectors = self.psi_list

    def create_circuit(self):
        """Create a quantum circuit for PVM measurement (computational basis). Returns (qc, prep_qubits)."""
        qc = QuantumCircuit(1, 1)
        prep_qubits = [0]
        return qc, prep_qubits

    def prepare_measurement(self, qc):
        """Apply computational basis measurement to the circuit."""
        qc.measure(0, 0)
        outcome_labels = ['0', '1']
        return qc, outcome_labels

    def get_operators(self):
        """Return the theoretical PVM operators (projectors onto |0⟩ and |1⟩)."""
        M0 = np.array([[1, 0], [0, 0]])
        M1 = np.array([[0, 0], [0, 1]])
        return [M0, M1]

    def get_outcome_label_map(self):
        return {'0': '0', '1': '1'}


def create_povm(povm_type='bb84', **kwargs):
    """
    Factory method to create different types of POVMs.
    
    Args:
        povm_type (str): Type of POVM to create. Options are:
                         - "bb84": BB84 POVM (Z and X bases)
                         - "sic": SIC POVM (tetrahedral measurement)
                         - "mub": MUB POVM (Z, X, and Y bases)
        **kwargs: Additional arguments passed to the specific POVM constructor.
                  For BB84 and MUB POVMs: 'basis' can be specified.
    
    Returns:
        POVM: An instance of the requested POVM type.
        
    Raises:
        ValueError: If an invalid POVM type is specified.
    """
    povm_type = povm_type.lower()
    
    if povm_type == "bb84":
        return BB84POVM(**kwargs)
    elif povm_type == "sic":
        return SICPOVM(**kwargs)
    elif povm_type == "mub":
        return MUBPOVM(**kwargs)
    elif povm_type == "trine":
        return TRINEPOVM(**kwargs)
    elif povm_type == "pvm":
        return PVMPOVM(**kwargs)
    else:
        valid_types = ["bb84", "sic", "mub", "trine", "pvm"]
        raise ValueError(f"Invalid POVM type '{povm_type}'. Must be one of {valid_types}")
