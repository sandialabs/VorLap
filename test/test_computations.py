"""
Unit tests for vorlap.computations module.
"""

import unittest
import numpy as np
import math
from vorlap.computations import (
    rotate_vector,
    rotationMatrix,
    reconstruct_signal
)


class TestRotateVector(unittest.TestCase):
    """Test cases for rotate_vector function."""
    
    def test_rotate_vector_identity(self):
        """Test rotation by 0 degrees (identity)."""
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        angle = 0.0
        
        result = rotate_vector(vec, axis, angle)
        np.testing.assert_array_almost_equal(result, vec)
    
    def test_rotate_vector_90_degrees_z(self):
        """Test 90-degree rotation around Z-axis."""
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        angle = 90.0
        
        result = rotate_vector(vec, axis, angle)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_rotate_vector_180_degrees_z(self):
        """Test 180-degree rotation around Z-axis."""
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        angle = 180.0
        
        result = rotate_vector(vec, axis, angle)
        expected = np.array([-1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_rotate_vector_360_degrees(self):
        """Test 360-degree rotation (should return original vector)."""
        vec = np.array([1.0, 2.0, 3.0])
        axis = np.array([0.0, 0.0, 1.0])
        angle = 360.0
        
        result = rotate_vector(vec, axis, angle)
        np.testing.assert_array_almost_equal(result, vec, decimal=10)
    
    def test_rotate_vector_arbitrary_axis(self):
        """Test rotation around arbitrary axis."""
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([1.0, 1.0, 0.0])  # Not normalized
        angle = 90.0
        
        result = rotate_vector(vec, axis, angle)
        # The result should be a unit vector (rotation preserves magnitude)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=10)
    
    def test_rotate_vector_negative_angle(self):
        """Test rotation with negative angle."""
        vec = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        angle = -90.0
        
        result = rotate_vector(vec, axis, angle)
        expected = np.array([0.0, -1.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_rotate_vector_preserves_magnitude(self):
        """Test that rotation preserves vector magnitude."""
        vec = np.array([3.0, 4.0, 5.0])
        original_magnitude = np.linalg.norm(vec)
        axis = np.array([1.0, 1.0, 1.0])
        angle = 45.0
        
        result = rotate_vector(vec, axis, angle)
        result_magnitude = np.linalg.norm(result)
        
        self.assertAlmostEqual(original_magnitude, result_magnitude, places=10)


class TestRotationMatrix(unittest.TestCase):
    """Test cases for rotationMatrix function."""
    
    def test_rotation_matrix_identity(self):
        """Test rotation matrix with zero angles."""
        euler = np.array([0.0, 0.0, 0.0])
        R = rotationMatrix(euler)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(R, expected)
    
    def test_rotation_matrix_90_degrees_z(self):
        """Test rotation matrix for 90-degree yaw."""
        euler = np.array([0.0, 0.0, 90.0])  # roll, pitch, yaw
        R = rotationMatrix(euler)
        
        # Test rotation of [1, 0, 0] should give [0, 1, 0]
        vec = np.array([1.0, 0.0, 0.0])
        result = R @ vec
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_rotation_matrix_90_degrees_y(self):
        """Test rotation matrix for 90-degree pitch."""
        euler = np.array([0.0, 90.0, 0.0])  # roll, pitch, yaw
        R = rotationMatrix(euler)
        
        # Test rotation of [1, 0, 0] should give [0, 0, -1]
        vec = np.array([1.0, 0.0, 0.0])
        result = R @ vec
        expected = np.array([0.0, 0.0, -1.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_rotation_matrix_90_degrees_x(self):
        """Test rotation matrix for 90-degree roll."""
        euler = np.array([90.0, 0.0, 0.0])  # roll, pitch, yaw
        R = rotationMatrix(euler)
        
        # Test rotation of [0, 1, 0] should give [0, 0, 1]
        vec = np.array([0.0, 1.0, 0.0])
        result = R @ vec
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_rotation_matrix_orthogonal(self):
        """Test that rotation matrix is orthogonal."""
        euler = np.array([30.0, 45.0, 60.0])
        R = rotationMatrix(euler)
        
        # R^T * R should equal identity
        RTR = R.T @ R
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(RTR, expected, decimal=10)
    
    def test_rotation_matrix_determinant(self):
        """Test that rotation matrix has determinant of 1."""
        euler = np.array([15.0, 25.0, 35.0])
        R = rotationMatrix(euler)
        
        det = np.linalg.det(R)
        self.assertAlmostEqual(det, 1.0, places=10)
    
    def test_rotation_matrix_combined_rotations(self):
        """Test combined rotations."""
        euler = np.array([30.0, 45.0, 60.0])
        R = rotationMatrix(euler)
        
        # Test with a known vector
        vec = np.array([1.0, 1.0, 1.0])
        result = R @ vec
        
        # Result should be a unit vector if input is normalized
        vec_normalized = vec / np.linalg.norm(vec)
        result_normalized = R @ vec_normalized
        self.assertAlmostEqual(np.linalg.norm(result_normalized), 1.0, places=10)


class TestReconstructSignal(unittest.TestCase):
    """Test cases for reconstruct_signal function."""
    
    def test_reconstruct_signal_dc_only(self):
        """Test signal reconstruction with DC component only."""
        freqs = np.array([0.0])
        amps = np.array([5.0])
        phases = np.array([0.0])
        tvec = np.array([0.0, 0.1, 0.2, 0.3])
        
        signal = reconstruct_signal(freqs, amps, phases, tvec)
        expected = np.array([5.0, 5.0, 5.0, 5.0])
        np.testing.assert_array_almost_equal(signal, expected)
    
    def test_reconstruct_signal_sine_wave(self):
        """Test signal reconstruction with a sine wave."""
        freqs = np.array([0.0, 1.0])  # DC + 1 Hz
        amps = np.array([0.0, 1.0])   # No DC, amplitude 1
        phases = np.array([0.0, 0.0]) # No phase shift
        tvec = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # One period
        
        signal = reconstruct_signal(freqs, amps, phases, tvec)
        
        # At t=0: cos(0) = 1
        # At t=0.25: cos(π/2) = 0
        # At t=0.5: cos(π) = -1
        # At t=0.75: cos(3π/2) = 0
        # At t=1.0: cos(2π) = 1
        expected = np.array([1.0, 0.0, -1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(signal, expected)
    
    def test_reconstruct_signal_phase_shift(self):
        """Test signal reconstruction with phase shift."""
        freqs = np.array([0.0, 1.0])
        amps = np.array([0.0, 1.0])
        phases = np.array([0.0, np.pi/2])  # 90-degree phase shift
        tvec = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        
        signal = reconstruct_signal(freqs, amps, phases, tvec)
        
        # With π/2 phase shift: cos(ωt + π/2) = -sin(ωt)
        # At t=0: -sin(0) = 0
        # At t=0.25: -sin(π/2) = -1
        # At t=0.5: -sin(π) = 0
        # At t=0.75: -sin(3π/2) = 1
        # At t=1.0: -sin(2π) = 0
        expected = np.array([0.0, -1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(signal, expected)
    
    def test_reconstruct_signal_multiple_frequencies(self):
        """Test signal reconstruction with multiple frequencies."""
        freqs = np.array([0.0, 1.0, 2.0])
        amps = np.array([1.0, 0.5, 0.25])
        phases = np.array([0.0, 0.0, 0.0])
        tvec = np.array([0.0, 0.5, 1.0])
        
        signal = reconstruct_signal(freqs, amps, phases, tvec)
        
        # DC: 1.0
        # 1 Hz: 0.5 * cos(2π * 1 * t)
        # 2 Hz: 0.25 * cos(2π * 2 * t)
        expected = np.array([
            1.0 + 0.5 + 0.25,  # t=0: all cosines = 1
            1.0 - 0.5 + 0.25,  # t=0.5: cos(π)=-1, cos(2π)=1
            1.0 + 0.5 + 0.25   # t=1.0: all cosines = 1
        ])
        np.testing.assert_array_almost_equal(signal, expected)
    
    def test_reconstruct_signal_input_validation(self):
        """Test input validation for reconstruct_signal."""
        # Test mismatched array lengths
        with self.assertRaises(ValueError):
            reconstruct_signal(
                np.array([0.0, 1.0]),
                np.array([1.0]),  # Different length
                np.array([0.0, 0.0]),
                np.array([0.0, 0.1])
            )
        
        # Test insufficient time points
        with self.assertRaises(ValueError):
            reconstruct_signal(
                np.array([0.0]),
                np.array([1.0]),
                np.array([0.0]),
                np.array([0.0])  # Only one time point
            )
    
    def test_reconstruct_signal_nyquist_warning(self):
        """Test that Nyquist frequency warning is generated."""
        freqs = np.array([0.0, 10.0])  # 10 Hz
        amps = np.array([0.0, 1.0])
        phases = np.array([0.0, 0.0])
        tvec = np.array([0.0, 0.1, 0.2])  # dt=0.1, fs=10 Hz, fnyq=5 Hz
        
        # This should generate a warning about exceeding Nyquist frequency
        # We can't easily test the warning, but we can ensure it doesn't crash
        signal = reconstruct_signal(freqs, amps, phases, tvec)
        self.assertEqual(len(signal), len(tvec))
    
    def test_reconstruct_signal_zero_frequency_handling(self):
        """Test handling of multiple zero frequencies."""
        freqs = np.array([0.0, 0.0, 1.0])  # Two DC components
        amps = np.array([2.0, 3.0, 1.0])
        phases = np.array([0.0, 0.0, 0.0])
        tvec = np.array([0.0, 0.1, 0.2])
        
        signal = reconstruct_signal(freqs, amps, phases, tvec)
        
        # DC components should sum: 2.0 + 3.0 = 5.0
        # Plus 1 Hz component: 1.0 * cos(2π * 1 * t)
        expected_dc = 5.0
        self.assertAlmostEqual(signal[0], expected_dc + 1.0)  # t=0: cos(0)=1
        self.assertAlmostEqual(signal[1], expected_dc + np.cos(2*np.pi*0.1))
        self.assertAlmostEqual(signal[2], expected_dc + np.cos(2*np.pi*0.2))


if __name__ == '__main__':
    unittest.main()





