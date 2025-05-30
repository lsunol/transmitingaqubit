"""
State module for quantum state preparations.
This module contains classes to represent and prepare various quantum states.
"""

from abc import ABC, abstractmethod
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import math


class State(ABC):
    """
    Abstract base class for quantum states.
    
    This class defines the interface for all quantum states.
    Each state must be able to prepare itself on a given quantum circuit.
    """
    
    def __init__(self, label=None):
        """Initialize state with an optional label."""
        self.label = label
        
    @abstractmethod
    def prepare(self, qc, qubit=0):
        """
        Prepare the state on the given quantum circuit.
        
        Args:
            qc (QuantumCircuit): The quantum circuit to prepare the state on.
            qubit (int, optional): The qubit index to prepare the state on. Defaults to 0.
        
        Returns:
            QuantumCircuit: The modified quantum circuit.
        """
        pass
    
    @abstractmethod
    def get_statevector(self):
        """
        Return the theoretical statevector for this state.
        
        Returns:
            numpy.ndarray: The statevector representation of this state.
        """
        pass
    
    @abstractmethod
    def get_bloch_angles(self):
        """
        Return the Bloch sphere angles (theta, phi) for this state in degrees.
        
        Returns:
            tuple: (theta_deg, phi_deg) where:
                   - theta_deg: polar angle in degrees [0, 180]
                   - phi_deg: azimuthal angle in degrees [0, 360)
        """
        pass
    
    def __str__(self):
        """String representation of the state."""
        return self.label if self.label else "Unnamed state"


class Zero(State):
    """State |0⟩."""
    
    def __init__(self):
        super().__init__(label="|0⟩")
    
    def prepare(self, qc, qubit=0):
        # |0⟩ is the default state, so no operations needed
        return qc
    
    def get_statevector(self):
        return np.array([1, 0], dtype=complex)
    
    def get_bloch_angles(self):
        # |0⟩ state: theta=0 (north pole), phi=0
        return (0.0, 0.0)

class One(State):
    """State |1⟩."""
    
    def __init__(self):
        super().__init__(label="|1⟩")
    
    def prepare(self, qc, qubit=0):
        qc.x(qubit)
        return qc
    
    def get_statevector(self):
        return np.array([0, 1], dtype=complex)
    
    def get_bloch_angles(self):
        # |1⟩ state: theta=180 (south pole), phi=0
        return (180.0, 0.0)

class Plus(State):
    """State |+⟩ = (|0⟩ + |1⟩)/√2."""
    
    def __init__(self):
        super().__init__(label="|+⟩")
    
    def prepare(self, qc, qubit=0):
        qc.h(qubit)
        return qc
    
    def get_statevector(self):
        return np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    
    def get_bloch_angles(self):
        # |+⟩ state: theta=90, phi=0 (positive X direction)
        return (90.0, 0.0)

class Minus(State):
    """State |-⟩ = (|0⟩ - |1⟩)/√2."""
    
    def __init__(self):
        super().__init__(label="|-⟩")
    
    def prepare(self, qc, qubit=0):
        qc.x(qubit)
        qc.h(qubit)
        return qc
    
    def get_statevector(self):
        return np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)
    
    def get_bloch_angles(self):
        # |-⟩ state: theta=90, phi=180 (negative X direction)
        return (90.0, 180.0)

class PlusI(State):
    """State |i⟩ = (|0⟩ + i|1⟩)/√2."""
    
    def __init__(self):
        super().__init__(label="|i⟩")
    
    def prepare(self, qc, qubit=0):
        qc.h(qubit)
        qc.s(qubit)  # S gate applies a π/2 phase to |1⟩ component
        return qc
    
    def get_statevector(self):
        return np.array([1/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex)
    
    def get_bloch_angles(self):
        # |i⟩ state: theta=90, phi=90 (positive Y direction)
        return (90.0, 90.0)

class MinusI(State):
    """State |-i⟩ = (|0⟩ - i|1⟩)/√2."""
    
    def __init__(self):
        super().__init__(label="|-i⟩")
    
    def prepare(self, qc, qubit=0):
        qc.h(qubit)
        qc.sdg(qubit)  # S† gate applies a -π/2 phase to |1⟩ component
        return qc
    
    def get_statevector(self):
        return np.array([1/np.sqrt(2), -1j/np.sqrt(2)], dtype=complex)
    
    def get_bloch_angles(self):
        # |-i⟩ state: theta=90, phi=270 (negative Y direction)
        return (90.0, 270.0)

class Custom(State):
    """
    Custom state defined by spherical coordinates on the Bloch sphere.
    |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    
    where:
    - θ (theta): polar angle [0, π]
    - φ (phi): azimuthal angle [0, 2π)
    """
    
    def __init__(self, theta, phi, label=None):
        """
        Initialize a custom state with Bloch sphere coordinates.
        
        Args:
            theta (float): The polar angle in radians [0, π].
            phi (float): The azimuthal angle in radians [0, 2π).
            label (str, optional): Label for the state.
        """
        if label is None:
            label = f"|ψ(θ={theta:.2f}, φ={phi:.2f})⟩"
        super().__init__(label=label)
        self.theta = theta
        self.phi = phi
    
    def prepare(self, qc, qubit=0):
        # U3 gate applies the most general single-qubit unitary
        # U3(theta, phi, lambda) where lambda=0 for our case
        qc.u(self.theta, self.phi, 0, qubit)
        return qc
    
    def get_statevector(self):
        # Calculate state vector based on theta and phi
        return np.array([
            np.cos(self.theta/2),
            np.exp(1j*self.phi) * np.sin(self.theta/2)
        ], dtype=complex)
    
    def get_bloch_angles(self):
        # Convert radians to degrees
        theta_deg = np.rad2deg(self.theta)
        phi_deg = np.rad2deg(self.phi)
        return (theta_deg, phi_deg)

class CustomStatevector(State):
    """
    Custom state defined directly by its statevector.
    |ψ⟩ = α|0⟩ + β|1⟩
    
    where:
    - α, β: complex amplitudes such that |α|² + |β|² = 1
    """
    
    def __init__(self, alpha, beta, label=None):
        """
        Initialize a custom state with statevector components.
        
        Args:
            alpha (complex): The amplitude of the |0⟩ component.
            beta (complex): The amplitude of the |1⟩ component.
            label (str, optional): Label for the state.
        """
        if label is None:
            label = f"|ψ({alpha:.3f},{beta:.3f})⟩"
        super().__init__(label=label)
        
        # Normalize the state vector
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        self.alpha = alpha / norm
        self.beta = beta / norm
        
        # Convert to Bloch sphere parameters for preparation
        self.theta = 2 * np.arccos(np.clip(abs(self.alpha), 0, 1))
        
        # Handle special cases to avoid division by zero
        if abs(self.beta) < 1e-10:
            self.phi = 0
        else:
            self.phi = np.angle(self.beta / abs(self.beta))
    
    def prepare(self, qc, qubit=0):
        # U3 gate applies the most general single-qubit unitary
        qc.u(self.theta, self.phi, 0, qubit)
        return qc
    
    def get_statevector(self):
        return np.array([self.alpha, self.beta], dtype=complex)
    
    def get_bloch_angles(self):
        # Convert radians to degrees (using the theta and phi calculated in __init__)
        theta_deg = np.rad2deg(self.theta)
        phi_deg = np.rad2deg(self.phi)
        return (theta_deg, phi_deg)


def create_state(state_name, custom_params=None):
    """
    Factory function to create a state based on a string identifier.
    
    Args:
        state_name (str): Name of the state ('0', '1', '+', '-', 'i', '-i', or 'custom')
        custom_params (dict or str, optional): Parameters for custom states
            For 'custom' with angles: {'theta': float, 'phi': float}
            For 'custom' with statevector: {'alpha': complex, 'beta': complex}
            For 'custom' with string: "real,imag real,imag" format
            
    Returns:
        State: An instance of a State subclass
    
    Raises:
        ValueError: If state_name is invalid or if custom_params are missing
    """
    if state_name == '0':
        return Zero()
    elif state_name == '1':
        return One()
    elif state_name == '+':
        return Plus()
    elif state_name == '-':
        return Minus()
    elif state_name == 'i':
        return PlusI()
    elif state_name == '-i':
        return MinusI()
    elif state_name == 'custom':
        if custom_params is None:
            raise ValueError("Custom state requires parameters")
        
        # Handle string input from command line
        if isinstance(custom_params, str):
            custom_params = validate_custom_state(custom_params)
        
        # Three ways to create custom states
        if 'theta' in custom_params and 'phi' in custom_params:
            return Custom(custom_params['theta'], custom_params['phi'])
        elif 'alpha' in custom_params and 'beta' in custom_params:
            return CustomStatevector(custom_params['alpha'], custom_params['beta'])
        else:
            raise ValueError("Invalid custom state parameters")
    else:
        raise ValueError(f"Unknown state: {state_name}")

def validate_custom_state(custom_state_str):
    """
    Validate and parse a custom state string from command line arguments.
    
    Args:
        custom_state_str (str): String in format "real,imag real,imag" 
                               representing alpha and beta components
    
    Returns:
        dict: Dictionary with 'alpha' and 'beta' complex values ready for CustomStatevector
        
    Raises:
        ValueError: If the string format is invalid or state vector is zero
        
    Example:
        >>> validate_custom_state("0.707,0 0.707,0")
        {'alpha': (0.707+0j), 'beta': (0.707+0j)}
    """
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
        
        # Check if state vector is zero
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        if norm == 0:
            raise ValueError("State vector cannot be zero")
        
        # Return the complex amplitudes (normalization will be done by CustomStatevector)
        return {'alpha': alpha, 'beta': beta}
        
    except Exception as e:
        raise ValueError(f"Invalid custom state format: {e}")

def create_state_from_args(state_name, custom_state_str=None):
    """
    Create a state from command line arguments with built-in validation.
    
    This function handles the logic for custom state creation including validation,
    making the main function cleaner by encapsulating all state creation logic here.
    
    Args:
        state_name (str): Name of the state ('0', '1', '+', '-', 'i', '-i', or 'custom')
        custom_state_str (str, optional): Custom state string in "real,imag real,imag" format
                                        (required only when state_name='custom')
    
    Returns:
        State: An instance of a State subclass
    
    Raises:
        ValueError: If state_name is invalid, custom_state_str is missing for custom states,
                   or if custom state format is invalid
    """
    if state_name == 'custom':
        if custom_state_str is None:
            raise ValueError("Custom state string is required when state='custom'")
        return create_state(state_name, custom_state_str)
    else:
        return create_state(state_name)