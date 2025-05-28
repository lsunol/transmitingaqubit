import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import qutip as qt
import scipy.linalg as la

class POVM:
    """Base class to represent a POVM"""
    def __init__(self, operators: List[np.ndarray], labels: Optional[List[str]] = None):
        self.operators = operators
        self.dimension = operators[0].shape[0]
        self.num_outcomes = len(operators)
        self.labels = labels if labels else [str(i) for i in range(self.num_outcomes)]
        
        if not self.validate():
            raise ValueError("The given operators do not form a valid POVM")
    
    def print(self):
        """Print all operators in a row"""
        lines = ["", ""]
        for op, label in zip(self.operators, self.labels):
            str_mat = []
            for row in op:
                row_str = []
                for x in row:
                    real = x.real
                    imag = x.imag
                    
                    # Format real part
                    if abs(real) < 1e-10:
                        real_str = "0".rjust(6)
                    else:
                        real_str = f"{real:.3f}".rstrip('0').rstrip('.')
                        real_str = real_str.rjust(6)
                    
                    # Handle complex part if significant
                    if abs(imag) > 1e-10:
                        imag_str = f"{abs(imag):.3f}".rstrip('0').rstrip('.')
                        sign = ' + ' if imag >= 0 else ' - '
                        row_str.append(f"{real_str}{sign}{imag_str}j")
                    else:
                        row_str.append(real_str)
                str_mat.append(row_str)
            # Add to lines with separator, using str.join() to avoid quotes
            lines[0] += f"{label}: [{', '.join(str_mat[0])}]  "
            lines[1] += f"   [{', '.join(str_mat[1])}] "
        
        
        print(lines[0])
        print(lines[1])

    def validate(self) -> bool:
        """Evaluates a POVM set being valid or not"""
        validator = POVMValidator()
        return validator.validate_povm(self.operators)
    
    def get_operator(self, label: str) -> np.ndarray:
        """Gets a specific operator by its label"""
        idx = self.labels.index(label)
        return self.operators[idx]
    
    def __str__(self) -> str:
        return f"POVM with {self.num_outcomes} outcomes in dimension {self.dimension}"

    def plot_povm(self, fig=None):
        """
        Visualize the POVM elements on the Bloch sphere using QuTiP.
        
        Args:
            fig: Optional existing Bloch sphere figure to plot on
            
        Returns:
            The QuTiP Bloch sphere figure with the POVM vectors
        """
        
        # Create a new Bloch sphere if one wasn't provided
        if fig is None:
            fig = qt.Bloch()
        
        # Only works for 2-dimensional POVMs (qubits)
        if self.dimension != 2:
            raise ValueError("POVM visualization is only implemented for 2-dimensional systems")
        
        # Colors for different POVM elements
        colors = ['r', 'g', 'b', 'y', 'm', 'c']
        
        for i, operator in enumerate(self.operators):
            # Convert operator to vector on Bloch sphere
            # For a 2D POVM element E, we can decompose it as:
            # E = a*I + b*X + c*Y + d*Z where X,Y,Z are Pauli matrices
            
            # Calculate the Bloch vector components using Pauli decomposition
            coeff = np.trace(operator) / 2  # a - coefficient of Identity
            
            # Pauli matrices
            X = np.array([[0, 1], [1, 0]])
            Y = np.array([[0, -1j], [1j, 0]])
            Z = np.array([[1, 0], [0, -1]])
            
            # Calculate components (scaled by weight)
            x = np.real(np.trace(operator @ X)) / 2
            y = np.real(np.trace(operator @ Y)) / 2
            z = np.real(np.trace(operator @ Z)) / 2
            
            # Add vector to Bloch sphere with appropriate color and label
            color = colors[i % len(colors)]
            fig.add_vectors([x, y, z])
            fig.vector_color = [color] * len(fig.vectors)
            
            # Add annotation with weight (trace)
            weight = np.trace(operator).real
            label = f"{self.labels[i]} ({weight:.2f})"
            fig.annotate.append([x, y, z, label])
        
        return fig

class POVMValidator:
    """Class to validate the mathematical properties of a POVM"""
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
    
    def validate_povm(self, operators: List[np.ndarray]) -> bool:
        """Validates all required properties of a POVM"""
        return (self._validate_hermiticity(operators) and
                self._validate_positivity(operators) and
                self._validate_completeness(operators))
    
    def _validate_hermiticity(self, operators: List[np.ndarray]) -> bool:
        """Verifies that all operators are hermitian"""
        return all(np.allclose(operator, operator.conj().T, atol=self.tolerance) 
                  for operator in operators)
    
    def _validate_positivity(self, operators: List[np.ndarray]) -> bool:
        """Verifies that all operators are positive semidefinite"""
        return all(np.all(np.real(la.eigvals(operator)) >= -self.tolerance) 
                  for operator in operators)
    
    def _validate_completeness(self, operators: List[np.ndarray]) -> bool:
        """Verifies that the operators sum to identity"""
        dim = operators[0].shape[0]
        sum_ops = sum(operators)
        return np.allclose(sum_ops, np.eye(dim), atol=self.tolerance)

class POVMGenerator(ABC):
    """Base abstract class for POVM generators"""
    @abstractmethod
    def generate(self) -> POVM:
        pass

class RandomPOVMGenerator(POVMGenerator):
    """Generates random POVMs using Naimark's dilation method"""
    def __init__(self, dimension: int = 2, num_outcomes: int = 4):
        self.dimension = dimension
        self.num_outcomes = num_outcomes
    
    def generate(self) -> POVM:
        # Generate a random unitary matrix of appropriate dimension
        extended_dim = max(self.dimension, self.num_outcomes)
        U = self._random_unitary(extended_dim)
        
        # Projectors in extended dimension
        projectors = [np.zeros((extended_dim, extended_dim)) for _ in range(self.num_outcomes)]
        for i in range(self.num_outcomes):
            projectors[i][i,i] = 1
        
        # Generate POVM operators
        operators = []
        for proj in projectors:
            # Apply the unitary transformation
            transformed = U @ proj @ U.conj().T
            # Take the relevant submatrix
            operator = transformed[:self.dimension, :self.dimension]
            operators.append(operator)
        
        return POVM(operators)
    
    def _random_unitary(self, dim: int) -> np.ndarray:
        """Generate a random unitary matrix using QR decomposition"""
        X = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        Q, R = np.linalg.qr(X)
        D = np.diag(np.diag(R) / np.abs(np.diag(R)))
        return Q @ D

class BB84POVMGenerator(POVMGenerator):
    """Generates the BB84 POVM"""
    def generate(self) -> POVM:
        # Computational basis
        comp_0 = np.array([[1, 0], [0, 0]])
        comp_1 = np.array([[0, 0], [0, 1]])
        
        # Diagonal basis
        diag_plus = np.array([[0.5, 0.5], [0.5, 0.5]])
        diag_minus = np.array([[0.5, -0.5], [-0.5, 0.5]])
        
        operators = [1/2 * comp_0, 1/2 * comp_1, 1/2 * diag_plus, 1/2 * diag_minus]
        labels = ['0', '1', '+', '-']
        
        return POVM(operators, labels)

class TrinePOVMGenerator(POVMGenerator):
    """Generates the Trine POVM"""
    def generate(self) -> POVM:
        # Constants for the Trine POVM
        factor = 2/3
        operators = []
        
        # Generate the three symmetric operators in the XY plane
        for k in range(3):
            angle = 2 * np.pi * k / 3
            vector = np.array([np.cos(angle), np.sin(angle)])
            operator = factor * np.outer(vector, vector.conj())
            operators.append(operator)
        
        labels = ['0', '1', '2']
        return POVM(operators, labels)

# Generate the POVM with B₀ = (4/3)|0⟩⟨0| and B₁ = I - B₀
class AsymmetricPOVMGenerator(POVMGenerator):
    """Generates the POVM with B₀ = (4/3)|0⟩⟨0| and B₁ = I - B₀"""
    def generate(self) -> POVM:
        # B₀ = (3/4)|0⟩⟨0|
        B0 = np.array([[3/4, 0], [0, 0]])
        
        # B₁ = I - B₀ = (1/4)|0⟩⟨0| + |1⟩⟨1|
        B1 = np.eye(2) - B0
        
        operators = [B0, B1]
        labels = ['B0', 'B1']
        
        return POVM(operators, labels)

class SICPOVMGenerator(POVMGenerator):
    """Generates a Symmetric Informationally Complete (SIC) POVM for qubits"""
    def generate(self) -> POVM:
        return self._generate_qubit_sic_povm()
    
    def _generate_qubit_sic_povm(self) -> POVM:
        """
        Generates a SIC POVM for a qubit system.
        
        For a qubit, the SIC POVM consists of 4 operators corresponding to 
        the vertices of a tetrahedron in the Bloch sphere.
        """
        # Factor for normalization (1/dimension for SIC POVMs)
        factor = 1/2
        
        # The 4 vertices of a tetrahedron in the Bloch sphere
        vectors = [
            np.array([0, 0, 1]),                     # North pole
            np.array([2*np.sqrt(2)/3, 0, -1/3]),     # First vertex
            np.array([-np.sqrt(2)/3, np.sqrt(6)/3, -1/3]),  # Second vertex
            np.array([-np.sqrt(2)/3, -np.sqrt(6)/3, -1/3])  # Third vertex
        ]
        
        operators = []
        for vec in vectors:
            # Convert Bloch vector to density matrix: (I + r·σ)/2
            x, y, z = vec
            
            # Pauli matrices
            sigma_x = np.array([[0, 1], [1, 0]])
            sigma_y = np.array([[0, -1j], [1j, 0]])
            sigma_z = np.array([[1, 0], [0, -1]])
            
            # Calculate the projector
            projector = np.eye(2) + x * sigma_x + y * sigma_y + z * sigma_z
            projector = projector / 2
            
            # Scale by factor for POVM (1/4 for 2D SIC POVM)
            operator = factor * projector
            operators.append(operator)
        
        labels = ['SIC0', 'SIC1', 'SIC2', 'SIC3']
        return POVM(operators, labels)

def main():

    # Generate BB84 POVM
    bb84_gen = BB84POVMGenerator()
    bb84_povm = bb84_gen.generate()
    print("\nBB84 POVM:")
    bb84_povm.print()
    
    # Generate Trine POVM
    trine_gen = TrinePOVMGenerator()
    trine_povm = trine_gen.generate()
    print("\nTrine POVM:")
    trine_povm.print()

    # Generate random 2 dimensions and 4 outcomes POVM
    random_gen = RandomPOVMGenerator(dimension=2, num_outcomes=4)
    random_povm = random_gen.generate()
    print("Random 2 dimensions and 4 outcomes POVM:")
    random_povm.print()
    
    # Generate random 2 dimensions and 3 outcomes POVM
    random_gen = RandomPOVMGenerator(dimension=2, num_outcomes=3)
    random_povm = random_gen.generate()
    print("Random 2 dimensions and 3 outcomes POVM:")
    random_povm.print()
    
    # Generate the asymmetric POVM
    print("\nAsymmetric POVM:")
    asymmetric_gen = AsymmetricPOVMGenerator()
    asymmetric_povm = asymmetric_gen.generate()
    asymmetric_povm.print()
    
    # Generate the SIC POVM
    print("\nSIC POVM:")
    sic_gen = SICPOVMGenerator()
    sic_povm = sic_gen.generate()
    sic_povm.print()
    
    # Visualize the SIC POVM on Bloch sphere (optional)
    # Uncomment to visualize:
    # bloch = sic_povm.plot_povm()
    # bloch.show()

if __name__ == "__main__":
    main()