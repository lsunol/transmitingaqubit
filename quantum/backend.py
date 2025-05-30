"""
Backend module for quantum computation backends.
This module contains classes to represent and configure different quantum backends.
"""

from abc import ABC, abstractmethod
from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeManilaV2        
from qiskit_aer import AerSimulator


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
        
        Returns:
            Sampler: The sampler instance configured for this backend.
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


class QPUBackend(Backend):
    """
    QPU Backend implementation.
    
    This backend connects to real quantum hardware through IBM Quantum services.
    """
    
    def __init__(self, backend_name="least_busy"):
        """
        Initialize a QPU backend.
        
        Args:
            backend_name (str): Name of the specific backend to use, or "least_busy" 
                               to automatically select the least busy available QPU.
                               Default is "least_busy".
        """
        self.backend_name = backend_name
        if backend_name == "least_busy":
            super().__init__(label="QPU Backend (Least Busy)")
        else:
            super().__init__(label=f"QPU Backend ({backend_name})")
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
    
    This backend uses the Qiskit Aer simulator for fast, noiseless simulation.
    """
    
    def __init__(self):
        """Initialize an Aer simulator backend."""
        super().__init__(label="Aer Simulator Backend")
        self._initialize_backend()
        
    def _initialize_backend(self):
        """Initialize the Aer simulator backend."""
        self._backend = AerSimulator()
        self._sampler = Sampler(mode=self._backend)
    
        # Note: For AerSimulator, we typically don't use the runtime Sampler
        # but rather the backend directly with execute() or the legacy sampler
        # self._sampler = None  # Will be handled differently for Aer

    def provides_immediate_results(self):
        """Aer simulator backends provide immediate results (local simulation)."""
        return True


def create_backend(backend_type="aer_simulator", **kwargs):
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
        return QPUBackend(**kwargs)
    elif backend_type == "fake_backend":
        return FakeBackend(**kwargs)
    elif backend_type == "aer_simulator":
        return AerSimulatorBackend(**kwargs)
    else:
        valid_types = ["qpu", "fake_backend", "aer_simulator"]
        raise ValueError(f"Invalid backend type '{backend_type}'. Must be one of {valid_types}")
