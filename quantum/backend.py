"""
Backend module for quantum computation backends.
This module contains classes to represent and configure different quantum backends.
"""

from abc import ABC, abstractmethod
from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeManilaV2        
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
import numpy as np


class Backend(ABC):
    """
    Abstract base class for quantum backends.
    
    This class defines the interface for all quantum backends.
    Each backend must be able to provide a sampler for quantum circuit execution.
    """
    
    def __init__(self, label=None):
        """Initialize backend with an optional label."""
        self.label = label if label else "Unnamed Backend"
        self._backend = None
        self._sampler = None
        
    def get_backend(self):
        """
        Return the underlying quantum backend.
        
        Returns:
            Backend: The quantum backend instance.
        """
        return self._backend
    
    def get_sampler(self):
        """
        Return a sampler configured for this backend.
        For AerSimulatorBackend, return None to signal direct backend.run() usage.
        Returns:
            Sampler or None: The sampler instance configured for this backend, or None for AerSimulatorBackend.
        """
        return self._sampler
    
    @abstractmethod
    def provides_immediate_results(self):
        """
        Check if this backend provides immediate results or queued/delayed results.
        
        Returns:
            bool: True if results are available immediately after job submission,
                  False if jobs are queued and results must be collected later.
        """
        pass
    
    def __str__(self):
        """String representation of the backend."""
        return self.label

    def get_mean_noise_errors(self, args=None):
        """
        Returns the mean values for:
        1. Probability of measuring 0 when preparing 1 (readout error)
        2. Probability of measuring 1 when preparing 0 (readout error)
        3. Error probability of 1-qubit gates
        4. Error probability of 2-qubit gates
        Returns:
            dict: {
                'prob_meas0_prep1': float,
                'prob_meas1_prep0': float,
                'error_prob_1qubit_gate': float,
                'error_prob_2qubit_gate': float
            }
        """
        # Getting backend properties
        if self._backend is None:
            raise RuntimeError("Backend is not initialized.")
        # If properties are not available, check for AerSimulator custom/zero noise
        if not hasattr(self._backend, 'properties') or self._backend.properties is None or self._backend.properties() is None:
            if args is not None and hasattr(args, 'backend_type') and args.backend_type == 'aer_simulator':
                if hasattr(args, 'noise_model') and args.noise_model == 'custom':
                    # Use custom noise parameters from args
                    return {
                        'prob_meas0_prep1': getattr(args, 'prob_meas0_prep1', 0.0),
                        'prob_meas1_prep0': getattr(args, 'prob_meas1_prep0', 0.0),
                        'error_prob_1qubit_gate': getattr(args, 'error_prob_1qubit_gate', 0.0),
                        'error_prob_2qubit_gate': getattr(args, 'error_prob_2qubit_gate', 0.0)
                    }
                else:
                    # Zero noise: return all zeros
                    return {
                        'prob_meas0_prep1': 0.0,
                        'prob_meas1_prep0': 0.0,
                        'error_prob_1qubit_gate': 0.0,
                        'error_prob_2qubit_gate': 0.0
                    }
            # If not AerSimulator or not custom/zero, fallback to zeros
            return {
                'prob_meas0_prep1': 0.0,
                'prob_meas1_prep0': 0.0,
                'error_prob_1qubit_gate': 0.0,
                'error_prob_2qubit_gate': 0.0
            }
        properties = self._backend.properties()
        config = self._backend.configuration() if hasattr(self._backend, 'configuration') else None
        if config is None:
            raise RuntimeError("Backend does not provide configuration().")
        n_qubits = config.n_qubits

        # Readout errors (symmetric, so just use one value per qubit)
        readout_errors = []
        for q in range(n_qubits):
            ro_err = properties.readout_error(q)
            readout_errors.append(ro_err)

        mean_readout_error = np.mean(readout_errors) if readout_errors else 0.0

        # Gate errors
        basis_gates = config.basis_gates
        oneq_gates = [g for g in ['x', 'sx', 'rz'] if g in basis_gates]
        twoq_gates = [g for g in ['cx', 'ecr', 'cz'] if g in basis_gates]
        oneq_gate_errors = []
        twoq_gate_errors = []
        # Defensive: properties.gates may not exist or may not be iterable
        gates = getattr(properties, 'gates', None)
        if gates is None or not hasattr(gates, '__iter__'):
            raise RuntimeError("Backend properties do not provide an iterable 'gates' attribute.")
        for gate in gates:
            name = getattr(gate, 'gate', None)
            qubits = getattr(gate, 'qubits', None)
            
            if name in oneq_gates and qubits and len(qubits) == 1:
                err = [item['value'] for item in gate.to_dict()['parameters'] if item['name'] == 'gate_error'][0]
                oneq_gate_errors.append(err)
            elif name in twoq_gates and qubits and len(qubits) == 2:
                err = [item['value'] for item in gate.to_dict()['parameters'] if item['name'] == 'gate_error'][0]
                twoq_gate_errors.append(err)

        mean_error_1q = np.mean(oneq_gate_errors) if oneq_gate_errors else 0.0
        mean_error_2q = np.mean(twoq_gate_errors) if twoq_gate_errors else 0.0

        return {
            'prob_meas0_prep1': mean_readout_error,
            'prob_meas1_prep0': mean_readout_error,
            'error_prob_1qubit_gate': mean_error_1q,
            'error_prob_2qubit_gate': mean_error_2q
        }


class QPUBackend(Backend):
    """
    QPU Backend implementation.
    
    This backend connects to real quantum hardware through IBM Quantum services.
    """
    
    def __init__(self, arguments):
        """
        Initialize a QPU backend.
        
        Args:
            backend_name (str): Name of the specific backend to use, or "least_busy" 
                               to automatically select the least busy available QPU.
                               Default is "least_busy".
        """
        self.backend_name = arguments.backend_name
        if self.backend_name == "least_busy":
            super().__init__(label="QPU Backend (Least Busy)")
        else:
            super().__init__(label=f"QPU Backend ({self.backend_name})")
        self._initialize_backend()
        
    def _initialize_backend(self):
        """Initialize the QPU backend and sampler."""
        # Define the service
        service = QiskitRuntimeService()
        
        # Get the backend based on the backend_name parameter
        if self.backend_name == "least_busy":
            # Get the least busy real quantum processor
            self._backend = service.least_busy(operational=True, simulator=False)
            print(f"Least busy backend selected: {self._backend}")
        else:
            # Get the specific backend by name
            try:
                self._backend = service.backend(self.backend_name)
                print(f"Specific backend selected: {self._backend}")
            except Exception as e:
                raise RuntimeError(f"Failed to get backend '{self.backend_name}': {e}")
        
        # Define the sampler
        self._sampler = Sampler(mode=self._backend)
    
    def provides_immediate_results(self):
        """QPU backends provide delayed results (jobs are queued)."""
        return False


class FakeBackend(Backend):
    """
    Fake Backend implementation.
    
    This backend uses a fake backend that simulates real quantum hardware
    with noise models and realistic device characteristics.
    """
    
    def __init__(self, seed_simulator=42):
        """
        Initialize a fake backend.
        
        Args:
            seed_simulator (int): Seed for the simulator to get reproducible results.
        """
        super().__init__(label="Fake Backend (FakeManilaV2)")
        self.seed_simulator = seed_simulator
        self._initialize_backend()
        
    def _initialize_backend(self):
        """Initialize the fake backend and sampler."""
        # Run the sampler job locally using FakeManilaV2
        self._backend = FakeManilaV2()
        
        # You can use a fixed seed to get fixed results.
        options = {"simulator": {"seed_simulator": self.seed_simulator}}
        
        # Define Sampler
        self._sampler = Sampler(mode=self._backend, options=options)
    
    def provides_immediate_results(self):
        """Fake backends provide immediate results (local simulation)."""
        return True


class AerSimulatorBackend(Backend):
    """
    Aer Simulator Backend implementation.
    
    This backend uses the Qiskit Aer simulator for fast, noiseless or noisy simulation.
    """
    
    def __init__(self, arguments):
        super().__init__(label="Aer Simulator Backend")
        self.noise_model = arguments.noise_model
        self.backend_name = arguments.backend_name
        self._backend = None
        self._sampler = None
        self._initialize_backend(arguments)
        
    def _initialize_backend(self, arguments):
        if self.noise_model == "zero_noise":
            self._backend = AerSimulator()
        elif self.noise_model == "real":
            service = QiskitRuntimeService()
            if self.backend_name == "least_busy":
                real_backend = service.least_busy(operational=True, simulator=False)
            else:
                real_backend = service.backend(self.backend_name)
            self._backend = AerSimulator.from_backend(real_backend)
        elif self.noise_model == "custom":
            noise_model = self.create_noise_model(arguments)
            self._backend = AerSimulator(noise_model=noise_model)

        self._sampler = Sampler(mode=self._backend)

    def create_noise_model(self, arguments):
        """
        Create a custom Qiskit Aer noise model based on user-specified parameters.
        Args:
            readout_error (float): Probability of readout error (applied to all qubits).
            gate_error (float): Probability of depolarizing error for 1q/2q gates.
            thermal_relaxation (float): Placeholder for future thermal relaxation modeling.
        Returns:
            NoiseModel: Configured Qiskit Aer noise model.
        """

        noise_model = NoiseModel()

        # Readout error: symmetric for both 0->1 and 1->0
        p0given1 = arguments.prob_meas0_prep1
        p1given0 = arguments.prob_meas1_prep0
        readout_err = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
        noise_model.add_all_qubit_readout_error(readout_err)

        # Gate errors
        prob_1q = arguments.error_prob_1qubit_gate  # 1-qubit gate depolarizing error
        prob_2q = arguments.error_prob_2qubit_gate  # 2-qubit gate depolarizing error
        error_1q = depolarizing_error(prob_1q, 1)
        error_2q = depolarizing_error(prob_2q, 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

        return noise_model

    def get_sampler(self):
        """
        For AerSimulatorBackend, always return None to signal direct backend.run() usage.
        """
        return self._sampler

    def provides_immediate_results(self):
        return True

def create_backend(backend_type="aer_simulator", arguments=None):
    """
    Factory method to create different types of quantum backends.
    
    Args:
        backend_type (str): Type of backend to create. Options are:
                           - "qpu": Real quantum hardware backend
                           - "fake_backend": Fake backend with noise model
                           - "aer_simulator": Aer simulator backend
        **kwargs: Additional arguments passed to the specific backend constructor.
                 For "qpu" backend_type:
                 - backend_name (str): Specific backend name or "least_busy" (default: "least_busy")
                 For "fake_backend" backend_type:
                 - seed_simulator (int): Seed for reproducible results (default: 42)
    
    Returns:
        Backend: An instance of the requested backend type.
        
    Raises:
        ValueError: If an invalid backend type is specified.
    """
    backend_type = backend_type.lower()
    
    if backend_type == "qpu":
        return QPUBackend(arguments)
    elif backend_type == "fake_backend":
        # Pass a default seed if arguments is None or not an int
        seed = getattr(arguments, 'seed_simulator', 42) if arguments is not None else 42
        return FakeBackend(seed)
    elif backend_type == "aer_simulator":
        return AerSimulatorBackend(arguments)
    else:
        valid_types = ["qpu", "fake_backend", "aer_simulator"]
        raise ValueError(f"Invalid backend type '{backend_type}'. Must be one of {valid_types}")
