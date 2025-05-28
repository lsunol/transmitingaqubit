import unittest
import numpy as np
import io
import sys

from classic.povm import (
    POVM, POVMValidator, POVMGenerator, 
    RandomPOVMGenerator, BB84POVMGenerator, TrinePOVMGenerator
)

class TestPOVMValidator(unittest.TestCase):
    def setUp(self):
        self.validator = POVMValidator()
        
    def test_validate_hermiticity(self):
        # Valid hermitian operators
        hermitian_ops = [
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 1]])
        ]
        self.assertTrue(self.validator._validate_hermiticity(hermitian_ops))
        
        # Non-hermitian operator
        non_hermitian_ops = [
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 1j], [0, 1]])
        ]
        self.assertFalse(self.validator._validate_hermiticity(non_hermitian_ops))
    
    def test_validate_positivity(self):
        # Positive semidefinite operators
        positive_ops = [
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 1]])
        ]
        self.assertTrue(self.validator._validate_positivity(positive_ops))
        
        # Non-positive operator
        non_positive_ops = [
            np.array([[1, 0], [0, 0]]),
            np.array([[-0.1, 0], [0, 1.1]])
        ]
        self.assertFalse(self.validator._validate_positivity(non_positive_ops))
    
    def test_validate_completeness(self):
        # Complete set of operators
        complete_ops = [
            np.array([[0.5, 0], [0, 0]]),
            np.array([[0, 0], [0, 0.5]]),
            np.array([[0.5, 0], [0, 0.5]])
        ]
        self.assertTrue(self.validator._validate_completeness(complete_ops))
        
        # Incomplete set
        incomplete_ops = [
            np.array([[0.4, 0], [0, 0]]),
            np.array([[0, 0], [0, 0.4]]),
            np.array([[0.5, 0], [0, 0.5]])
        ]
        self.assertFalse(self.validator._validate_completeness(incomplete_ops))
    
    def test_validate_povm(self):
        # Valid POVM
        valid_povm = [
            np.array([[0.5, 0], [0, 0]]),
            np.array([[0, 0], [0, 0.5]]),
            np.array([[0.5, 0], [0, 0.5]])
        ]
        self.assertTrue(self.validator.validate_povm(valid_povm))
        
        # Invalid POVM (non-hermitian)
        invalid_povm = [
            np.array([[0.5, 1j], [0, 0]]),
            np.array([[0, 0], [0, 0.5]]),
            np.array([[0.5, 0], [0, 0.5]])
        ]
        self.assertFalse(self.validator.validate_povm(invalid_povm))


class TestPOVM(unittest.TestCase):
    def test_init_valid(self):
        # Testing initialization with valid POVM operators
        operators = [
            np.array([[0.5, 0], [0, 0]]),
            np.array([[0, 0], [0, 0.5]]),
            np.array([[0.5, 0], [0, 0.5]])
        ]
        povm = POVM(operators)
        self.assertEqual(povm.dimension, 2)
        self.assertEqual(povm.num_outcomes, 3)
        self.assertEqual(povm.labels, ['0', '1', '2'])

    def test_init_with_labels(self):
        operators = [
            np.array([[0.5, 0], [0, 0]]),
            np.array([[0, 0], [0, 0.5]]),
            np.array([[0.5, 0], [0, 0.5]])
        ]
        labels = ['a', 'b', 'c']
        povm = POVM(operators, labels)
        self.assertEqual(povm.labels, labels)

    def test_init_invalid(self):
        # Testing initialization with invalid POVM operators
        invalid_operators = [
            np.array([[1.1, 0], [0, 0]]),
            np.array([[0, 0], [0, 0.5]]),
            np.array([[0.5, 0], [0, 0.5]])
        ]
        with self.assertRaises(ValueError):
            POVM(invalid_operators)

    def test_get_operator(self):
        operators = [
            np.array([[0.5, 0], [0, 0]]),
            np.array([[0, 0], [0, 0.5]]),
            np.array([[0.5, 0], [0, 0.5]])
        ]
        labels = ['a', 'b', 'c']
        povm = POVM(operators, labels)
        
        # Get operator by label
        np.testing.assert_array_equal(povm.get_operator('a'), operators[0])
        np.testing.assert_array_equal(povm.get_operator('c'), operators[2])
        
        # Test non-existent label
        with self.assertRaises(ValueError):
            povm.get_operator('d')
    
    def test_print(self):
        operators = [
            np.array([[0.5, 0], [0, 0]]),
            np.array([[0, 0], [0, 0.5]])
        ]
        labels = ['a', 'b']
        povm = POVM(operators, labels)
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        povm.print()
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn("a: [   0.5,      0]", output)
        self.assertIn("b: [     0,      0]", output)

    def test_str(self):
        operators = [
            np.array([[0.5, 0], [0, 0]]),
            np.array([[0, 0], [0, 0.5]]),
        ]
        povm = POVM(operators)
        self.assertEqual(str(povm), "POVM with 2 outcomes in dimension 2")


class TestPOVMGenerators(unittest.TestCase):
    def test_bb84_generator(self):
        generator = BB84POVMGenerator()
        povm = generator.generate()
        
        self.assertEqual(povm.dimension, 2)
        self.assertEqual(povm.num_outcomes, 4)
        self.assertEqual(povm.labels, ['0', '1', '+', '-'])
        
        # Check specific properties of BB84 POVM
        np.testing.assert_array_almost_equal(
            povm.get_operator('0'), 
            np.array([[0.5, 0], [0, 0]])
        )
        np.testing.assert_array_almost_equal(
            povm.get_operator('1'), 
            np.array([[0, 0], [0, 0.5]])
        )

    def test_trine_generator(self):
        generator = TrinePOVMGenerator()
        povm = generator.generate()
        
        self.assertEqual(povm.dimension, 2)
        self.assertEqual(povm.num_outcomes, 3)
        self.assertEqual(povm.labels, ['0', '1', '2'])
        
        # Verify operators form a valid POVM
        validator = POVMValidator()
        self.assertTrue(validator.validate_povm(povm.operators))
        
        # Specific test for trine geometry (120 degree separation)
        angles = []
        for i in range(3):
            operator = povm.operators[i]
            # Extract angle from the operator (assuming 2D)
            # For a pure state |ψ⟩⟨ψ| where |ψ⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
            # The first element should be cos²(θ)
            cos_squared = operator[0, 0].real / (2/3)
            theta = np.arccos(np.sqrt(cos_squared))
            angles.append(theta)
        
        # Check angles are approximately 120 degrees apart
        angle_diffs = [
            np.abs((angles[1] - angles[0]) % (2*np.pi/3)),
            np.abs((angles[2] - angles[1]) % (2*np.pi/3)),
            np.abs((angles[0] - angles[2]) % (2*np.pi/3))
        ]
        for diff in angle_diffs:
            self.assertAlmostEqual(diff, 0, places=10)

    def test_random_generator(self):
        # Test with different dimensions and outcomes
        generator1 = RandomPOVMGenerator(dimension=2, num_outcomes=4)
        povm1 = generator1.generate()
        self.assertEqual(povm1.dimension, 2)
        self.assertEqual(povm1.num_outcomes, 4)
        
        generator2 = RandomPOVMGenerator(dimension=3, num_outcomes=5)
        povm2 = generator2.generate()
        self.assertEqual(povm2.dimension, 3)
        self.assertEqual(povm2.num_outcomes, 5)
        
        # Verify they form valid POVMs
        validator = POVMValidator()
        self.assertTrue(validator.validate_povm(povm1.operators))
        self.assertTrue(validator.validate_povm(povm2.operators))
        
        # Test random unitary is actually unitary
        dim = 4
        generator = RandomPOVMGenerator()
        U = generator._random_unitary(dim)
        # Check U†U = I
        product = U.conj().T @ U
        np.testing.assert_array_almost_equal(product, np.eye(dim))


if __name__ == '__main__':
    unittest.main()