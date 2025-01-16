import numpy as np

def generate_random_state(num_qubits: int) -> np.ndarray:
    """
    Generate a random pure quantum state for a given number of qubits.
    
    Parameters:
    - num_qubits (int): Number of qubits (e.g., 1 for single qubit, 2 for two qubits, etc.)
    
    Returns:
    - np.ndarray: Normalized random state vector of shape (2^num_qubits,).
    """
    dim = 2 ** num_qubits  # Calculate dimension based on number of qubits

    # Generate a random complex vector
    real_part = np.random.randn(dim)  # Real part
    imag_part = np.random.randn(dim)  # Imaginary part
    random_vector = real_part + 1j * imag_part

    # Normalize the vector to ensure it's a valid quantum state
    norm = np.linalg.norm(random_vector)  # Calculate the norm
    normalized_state = random_vector / norm  # Normalize the vector

    return normalized_state

# Example: Generate a random state for a single qubit
single_qubit_random_state = generate_random_state(1)
print("Random state:", single_qubit_random_state)
print("Norm of the state:", np.linalg.norm(single_qubit_random_state))  # Should be 1

# Example: Generate a random state for two qubits
two_qubits_random_state = generate_random_state(1)
print("Random state:", two_qubits_random_state)
print("Norm of the state:", np.linalg.norm(two_qubits_random_state))  # Should be 1