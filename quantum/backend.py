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
        
    @abstractmethod
    def get_backend(self):
        """
        Return the underlying quantum backend.
        
        Returns:
            Backend: The quantum backend instance.
        """
        pass
    
    @abstractmethod
    def get_sampler(self):
        """
        Return a sampler configured for this backend.
        
        Returns:
            Sampler: The sampler instance configured for this backend.
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
    
    def __init__(self):
        """Initialize a QPU backend."""
        super().__init__(label="QPU Backend")
        self._initialize_backend()
        
    def _initialize_backend(self):
        """Initialize the QPU backend and sampler."""
        # Define the service
        service = QiskitRuntimeService()
        
        # Get a backend (least busy real quantum processor)
        self._backend = service.least_busy(operational=True, simulator=False)
        print(f"Backend selected: {self._backend}")
        
        # Define the sampler
        self._sampler = Sampler(mode=self._backend)
    
    def get_backend(self):
        """Return the QPU backend."""
        return self._backend
    
    def get_sampler(self):
        """Return the QPU sampler."""
        return self._sampler


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
    
    def get_backend(self):
        """Return the fake backend."""
        return self._backend
    
    def get_sampler(self):
        """Return the fake backend sampler."""
        return self._sampler


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

    def get_backend(self):
        """Return the Aer simulator backend."""
        return self._backend
    
    def get_sampler(self):
        """
        Return the Aer simulator sampler.
        
        Note: AerSimulator typically doesn't use the runtime Sampler.
        This method returns None to indicate that direct backend usage is preferred.
        """
        return self._sampler


def create_backend(backend_type="aer_simulator", **kwargs):
    """
    Factory method to create different types of quantum backends.
    
    Args:
        backend_type (str): Type of backend to create. Options are:
                           - "qpu": Real quantum hardware backend
                           - "fake_backend": Fake backend with noise model
                           - "aer_simulator": Aer simulator backend
        **kwargs: Additional arguments passed to the specific backend constructor.
    
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


# Simple example of usage
def main():
    """Run a simple demonstration of backend implementations."""
    
    print("Creating different quantum backends...\n")
    
    # Test Aer Simulator Backend
    print("1. Aer Simulator Backend:")
    try:
        backend = create_backend("aer_simulator")
        print(f"   Created: {backend}")
        print(f"   Backend type: {type(backend.get_backend()).__name__}")
        print(f"   Sampler: {backend.get_sampler()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    
    # Test Fake Backend
    print("2. Fake Backend:")
    try:
        backend = create_backend("fake_backend", seed_simulator=123)
        print(f"   Created: {backend}")
        print(f"   Backend type: {type(backend.get_backend()).__name__}")
        print(f"   Has sampler: {backend.get_sampler() is not None}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    
    # Test QPU Backend (this might fail if not properly authenticated)
    print("3. QPU Backend:")
    try:
        backend = create_backend("qpu")
        print(f"   Created: {backend}")
        print(f"   Backend type: {type(backend.get_backend()).__name__}")
        print(f"   Has sampler: {backend.get_sampler() is not None}")
    except Exception as e:
        print(f"   Error (expected if not authenticated): {e}")
    
    print()
    
    # Test invalid backend type
    print("4. Invalid Backend Type:")
    try:
        backend = create_backend("invalid_type")
    except ValueError as e:
        print(f"   Correctly caught error: {e}")


if __name__ == "__main__":
    main()