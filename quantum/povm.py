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

    @abstractmethod
    def get_theoretical_distribution(self, input_state):
        """
        Calculate the theoretical probability distribution for measuring this POVM on the given input state.
        
        Args:
            input_state: A State object with get_bloch_angles method.
          Returns:
            dict: Dictionary mapping outcome bitstrings to their theoretical probabilities.
                 Keys are the same as in get_outcome_label_map().
        """
        pass

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
        """Create a quantum circuit for BB84 POVM measurement."""
        # BB84 POVM requires 1 qubit and 1 classical bit
        return QuantumCircuit(1, 1)
    
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

    def get_theoretical_distribution(self, input_state):
        """
        Calculate the theoretical probability distribution for BB84 POVM on the given input state.
        
        Args:
            input_state (numpy.ndarray): The input quantum state as a density matrix or state vector.
        
        Returns:
            dict: Dictionary mapping outcome bitstrings to their theoretical probabilities.
        """
        # TODO: Implement theoretical distribution calculation for BB84 POVM
        # This should calculate Born rule probabilities: p_i = Tr(M_i * rho)
        # where M_i are the POVM operators and rho is the input state density matrix
        raise NotImplementedError("Theoretical distribution calculation not yet implemented for BB84 POVM")


class SICPOVM(POVM):
    """
    Symmetric Informationally Complete (SIC) POVM implementation.
    
    This POVM consists of 4 operators corresponding to the vertices 
    of a tetrahedron in the Bloch sphere.
    """
    
    def __init__(self):
        """Initialize a SIC POVM."""
        super().__init__(label="SIC POVM")
        
        # Define the Bloch vectors for the tetrahedron vertices
        self.vectors = [
            np.array([[1], [0]]),                                                       # North pole
            (1/np.sqrt(3)) * np.array([[1], [np.sqrt(2)]]),                             # First vertex
            (1/np.sqrt(3)) * np.array([[1], [np.sqrt(2) * np.exp(2j * np.pi / 3)]]),    # Second vertex
            (1/np.sqrt(3)) * np.array([[1], [np.sqrt(2) * np.exp(4j * np.pi / 3)]])     # Third vertex
        ]

    def create_circuit(self):
        """Create a quantum circuit for SIC POVM measurement."""
        # SIC POVM requires 3 qubits (1 main + 2 ancillas) and 2 classical bits
        return QuantumCircuit(3, 2)
    
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
        
        for vec in self.vectors:
            # Convert state vector to projector: |ψ⟩⟨ψ|
            projector = vec @ vec.conj().T
            
            # Scale by factor for POVM
            operator = factor * projector
            operators.append(operator)
        
        return operators

    def get_outcome_label_map(self):
        # SIC: outcome '00', '01', '10', '11' map to SIC0, SIC1, SIC2, SIC3
        return {'00': 'SIC0', '01': 'SIC1', '10': 'SIC2', '11': 'SIC3'}    
    
    def get_theoretical_distribution(self, input_state):
        """
        Calculate the theoretical probability distribution for SIC POVM on the given input state.
        
        Args:
            input_state: A State object with get_bloch_angles method.
        
        Returns:
            dict: Dictionary mapping outcome bitstrings to their theoretical probabilities.
        """
        # Get Bloch angles from State object
        theta_deg, phi_deg = input_state.get_bloch_angles()

        phi = np.deg2rad(phi_deg)
        theta = np.deg2rad(theta_deg)

        # Quantum state |ψ⟩ in spherical coordinates
        ket = np.array([
            [np.cos(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2)]
        ])

        # SIC-POVM states (matching the ones defined in __init__)
        psi_0 = np.array([[1], [0]])
        psi_1 = (1/np.sqrt(3)) * np.array([[1], [np.sqrt(2)]])
        psi_2 = (1/np.sqrt(3)) * np.array([[1], [np.sqrt(2) * np.exp(2j * np.pi / 3)]])
        psi_3 = (1/np.sqrt(3)) * np.array([[1], [np.sqrt(2) * np.exp(4j * np.pi / 3)]])

        psi_list = [psi_0, psi_1, psi_2, psi_3]

        # Calculate probabilities
        probs = {}
        for i, psi_i in enumerate(psi_list):
            overlap = np.vdot(psi_i, ket)  # ⟨ψ_i | ψ⟩
            prob = (1/4) * np.abs(overlap)**2
            probs[f'{i:02b}'] = float(prob.real)  # keys '00', '01', '10', '11'

        total = sum(probs.values())
        
        # Normalize probabilities
        for k in probs:
            probs[k] /= total

        return probs
    

class MUBPOVM(POVM):
    """
    Mutually Unbiased Bases (MUB) POVM implementation.
    
    This POVM measures in one of the three mutually unbiased bases
    for a qubit: Z, X, or Y.
    """
    
    def __init__(self, basis='random'):
        """
        Initialize a MUB POVM.
        
        Args:
            basis (str): Basis choice ('z', 'x', 'y', or 'random'). Default is 'random'.
        """
        super().__init__(label="MUB POVM")
        self.basis_options = ['z', 'x', 'y']
        if basis == 'random':
            self.basis = np.random.choice(self.basis_options)
        elif basis in self.basis_options:
            self.basis = basis
        else:
            raise ValueError(f"Basis must be one of {self.basis_options} or 'random'")
    
    def create_circuit(self):
        """Create a quantum circuit for MUB POVM measurement."""
        # MUB POVM requires 1 qubit and 1 classical bit
        return QuantumCircuit(1, 1)

    def prepare_measurement(self, qc):
        """Apply MUB POVM measurement to the circuit."""
        if self.basis == 'x':
            # Measure in X basis by applying H before measurement
            qc.h(0)
        elif self.basis == 'y':
            # Measure in Y basis by applying S† and H before measurement
            qc.sdg(0)
            qc.h(0)
        
        # Add measurement
        qc.measure(0, 0)
        
        # Return the circuit and outcome labels
        if self.basis == 'z':
            outcome_labels = ['0', '1']
        elif self.basis == 'x':
            outcome_labels = ['+', '-']
        else:  # Y basis
            outcome_labels = ['+i', '-i']
            
        return qc, outcome_labels
    
    def get_operators(self):
        """Return the theoretical MUB POVM operators."""
        # Define the POVM operators based on the basis
        if self.basis == 'z':
            # Computational basis
            M0 = np.array([[1, 0], [0, 0]])
            M1 = np.array([[0, 0], [0, 1]])
            return [M0, M1]
        elif self.basis == 'x':
            # Diagonal basis
            M_plus = np.array([[0.5, 0.5], [0.5, 0.5]])
            M_minus = np.array([[0.5, -0.5], [-0.5, 0.5]])
            return [M_plus, M_minus]
        else:  # Y basis
            # Circular basis
            M_plus_i = np.array([[0.5, -0.5j], [0.5j, 0.5]])
            M_minus_i = np.array([[0.5, 0.5j], [-0.5j, 0.5]])
            return [M_plus_i, M_minus_i]

    def get_outcome_label_map(self):
        # MUB: outcome '0'/'1' for Z, '+'/'-' for X, '+i'/'-i' for Y
        if self.basis == 'z':
            return {'0': '0', '1': '1'}
        elif self.basis == 'x':
            return {'0': '+', '1': '-'}
        else:
            return {'0': '+i', '1': '-i'}

    def get_theoretical_distribution(self, input_state):
        """
        Calculate the theoretical probability distribution for MUB POVM on the given input state.
        
        Args:
            input_state (numpy.ndarray): The input quantum state as a density matrix or state vector.
        
        Returns:
            dict: Dictionary mapping outcome bitstrings to their theoretical probabilities.
        """
        # TODO: Implement theoretical distribution calculation for MUB POVM
        # This should calculate Born rule probabilities: p_i = Tr(M_i * rho)
        # where M_i are the POVM operators and rho is the input state density matrix
        raise NotImplementedError("Theoretical distribution calculation not yet implemented for MUB POVM")


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
    else:
        valid_types = ["bb84", "sic", "mub"]
        raise ValueError(f"Invalid POVM type '{povm_type}'. Must be one of {valid_types}")
