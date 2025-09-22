"""
Unit tests for vorlap.interpolation module.
"""

import unittest
import numpy as np
from vorlap.interpolation import (
    interpolate_fft_spectrum,
    interpolate_fft_spectrum_batch,
    interpolate_fft_spectrum_optimized,
    resample_airfoil
)
from vorlap.structs import AirfoilFFT


class TestInterpolateFFTSpectrum(unittest.TestCase):
    """Test cases for interpolate_fft_spectrum function."""
    
    def setUp(self):
        """Set up test AirfoilFFT data."""
        self.Re = np.array([1000, 5000, 10000])
        self.AOA = np.array([0, 5, 10, 15])
        self.n_freq = 3
        
        # Create test FFT data with known patterns
        shape = (len(self.Re), len(self.AOA), self.n_freq)
        
        # Create simple linear patterns for testing
        self.CL_ST = np.zeros(shape)
        self.CL_Amp = np.zeros(shape)
        self.CL_Pha = np.zeros(shape)
        
        # Fill with simple patterns
        for i, re in enumerate(self.Re):
            for j, aoa in enumerate(self.AOA):
                for k in range(self.n_freq):
                    self.CL_ST[i, j, k] = re * 0.001 + aoa * 0.01 + k * 0.1
                    self.CL_Amp[i, j, k] = re * 0.0001 + aoa * 0.001 + k * 0.01
                    self.CL_Pha[i, j, k] = re * 0.00001 + aoa * 0.0001 + k * 0.001
        
        # Copy patterns to other fields
        self.CD_ST = self.CL_ST * 0.5
        self.CD_Amp = self.CL_Amp * 0.5
        self.CD_Pha = self.CL_Pha * 0.5
        self.CF_ST = self.CL_ST * 0.8
        self.CF_Amp = self.CL_Amp * 0.8
        self.CF_Pha = self.CL_Pha * 0.8
        self.CM_ST = self.CL_ST * 0.3
        self.CM_Amp = self.CL_Amp * 0.3
        self.CM_Pha = self.CL_Pha * 0.3
        
        self.afft = AirfoilFFT(
            name="test_airfoil",
            Re=self.Re,
            AOA=self.AOA,
            Thickness=0.12,
            CL_ST=self.CL_ST,
            CD_ST=self.CD_ST,
            CM_ST=self.CM_ST,
            CF_ST=self.CF_ST,
            CL_Amp=self.CL_Amp,
            CD_Amp=self.CD_Amp,
            CM_Amp=self.CM_Amp,
            CF_Amp=self.CF_Amp,
            CL_Pha=self.CL_Pha,
            CD_Pha=self.CD_Pha,
            CM_Pha=self.CM_Pha,
            CF_Pha=self.CF_Pha
        )
    
    def test_interpolate_fft_spectrum_cl(self):
        """Test interpolation for CL field."""
        Re_val = 5000.0
        AOA_val = 5.0
        
        ST_out, amp_out, phase_out = interpolate_fft_spectrum(
            self.afft, Re_val, AOA_val, 'CL', n_freq_depth=2
        )
        
        self.assertEqual(len(ST_out), 2)
        self.assertEqual(len(amp_out), 2)
        self.assertEqual(len(phase_out), 2)
        
        # Check that values are reasonable (not NaN or infinite)
        self.assertTrue(np.all(np.isfinite(ST_out)))
        self.assertTrue(np.all(np.isfinite(amp_out)))
        self.assertTrue(np.all(np.isfinite(phase_out)))
    
    def test_interpolate_fft_spectrum_cd(self):
        """Test interpolation for CD field."""
        Re_val = 7500.0
        AOA_val = 7.5
        
        ST_out, amp_out, phase_out = interpolate_fft_spectrum(
            self.afft, Re_val, AOA_val, 'CD'
        )
        
        self.assertEqual(len(ST_out), self.n_freq)
        self.assertEqual(len(amp_out), self.n_freq)
        self.assertEqual(len(phase_out), self.n_freq)
        
        # Check that values are reasonable
        self.assertTrue(np.all(np.isfinite(ST_out)))
        self.assertTrue(np.all(np.isfinite(amp_out)))
        self.assertTrue(np.all(np.isfinite(phase_out)))
    
    def test_interpolate_fft_spectrum_cf(self):
        """Test interpolation for CF field."""
        Re_val = 3000.0
        AOA_val = 12.0
        
        ST_out, amp_out, phase_out = interpolate_fft_spectrum(
            self.afft, Re_val, AOA_val, 'CF', n_freq_depth=1
        )
        
        self.assertEqual(len(ST_out), 1)
        self.assertEqual(len(amp_out), 1)
        self.assertEqual(len(phase_out), 1)
        
        # Check that values are reasonable
        self.assertTrue(np.all(np.isfinite(ST_out)))
        self.assertTrue(np.all(np.isfinite(amp_out)))
        self.assertTrue(np.all(np.isfinite(phase_out)))
    
    def test_interpolate_fft_spectrum_cm(self):
        """Test interpolation for CM field."""
        Re_val = 8000.0
        AOA_val = 3.0
        
        ST_out, amp_out, phase_out = interpolate_fft_spectrum(
            self.afft, Re_val, AOA_val, 'CM'
        )
        
        self.assertEqual(len(ST_out), self.n_freq)
        self.assertEqual(len(amp_out), self.n_freq)
        self.assertEqual(len(phase_out), self.n_freq)
        
        # Check that values are reasonable
        self.assertTrue(np.all(np.isfinite(ST_out)))
        self.assertTrue(np.all(np.isfinite(amp_out)))
        self.assertTrue(np.all(np.isfinite(phase_out)))
    
    def test_interpolate_fft_spectrum_invalid_field(self):
        """Test interpolation with invalid field name."""
        with self.assertRaises(ValueError):
            interpolate_fft_spectrum(self.afft, 5000.0, 5.0, 'INVALID')
    
    def test_interpolate_fft_spectrum_extrapolation(self):
        """Test interpolation with values outside the grid (extrapolation)."""
        # Test with Re and AOA values outside the grid
        Re_val = 15000.0  # Outside grid
        AOA_val = 20.0    # Outside grid
        
        ST_out, amp_out, phase_out = interpolate_fft_spectrum(
            self.afft, Re_val, AOA_val, 'CL'
        )
        
        # Should not crash and should return finite values
        self.assertTrue(np.all(np.isfinite(ST_out)))
        self.assertTrue(np.all(np.isfinite(amp_out)))
        self.assertTrue(np.all(np.isfinite(phase_out)))


class TestInterpolateFFTSpectrumBatch(unittest.TestCase):
    """Test cases for interpolate_fft_spectrum_batch function."""
    
    def setUp(self):
        """Set up test AirfoilFFT data."""
        self.Re = np.array([1000, 5000, 10000])
        self.AOA = np.array([0, 5, 10, 15])
        self.n_freq = 3
        
        # Create simple test data
        shape = (len(self.Re), len(self.AOA), self.n_freq)
        self.CL_ST = np.random.rand(*shape)
        self.CL_Amp = np.random.rand(*shape)
        self.CL_Pha = np.random.rand(*shape) * 2 * np.pi
        self.CD_ST = np.random.rand(*shape)
        self.CD_Amp = np.random.rand(*shape)
        self.CD_Pha = np.random.rand(*shape) * 2 * np.pi
        self.CF_ST = np.random.rand(*shape)
        self.CF_Amp = np.random.rand(*shape)
        self.CF_Pha = np.random.rand(*shape) * 2 * np.pi
        
        self.afft = AirfoilFFT(
            name="test_airfoil",
            Re=self.Re,
            AOA=self.AOA,
            Thickness=0.12,
            CL_ST=self.CL_ST,
            CD_ST=self.CD_ST,
            CM_ST=self.CL_ST * 0.5,
            CF_ST=self.CF_ST,
            CL_Amp=self.CL_Amp,
            CD_Amp=self.CD_Amp,
            CM_Amp=self.CL_Amp * 0.5,
            CF_Amp=self.CF_Amp,
            CL_Pha=self.CL_Pha,
            CD_Pha=self.CD_Pha,
            CM_Pha=self.CL_Pha * 0.5,
            CF_Pha=self.CF_Pha
        )
    
    def test_interpolate_fft_spectrum_batch_cl(self):
        """Test batch interpolation for CL field."""
        Re_vals = np.array([2000, 6000, 8000])
        AOA_vals = np.array([2, 7, 12])
        
        ST_out, amp_out, phase_out = interpolate_fft_spectrum_batch(
            self.afft, Re_vals, AOA_vals, 'CL', n_freq_depth=2
        )
        
        self.assertEqual(ST_out.shape, (3, 2))  # 3 points, 2 frequencies
        self.assertEqual(amp_out.shape, (3, 2))
        self.assertEqual(phase_out.shape, (3, 2))
        
        # Check that values are reasonable
        self.assertTrue(np.all(np.isfinite(ST_out)))
        self.assertTrue(np.all(np.isfinite(amp_out)))
        self.assertTrue(np.all(np.isfinite(phase_out)))
    
    def test_interpolate_fft_spectrum_batch_cd(self):
        """Test batch interpolation for CD field."""
        Re_vals = np.array([3000, 7000])
        AOA_vals = np.array([4, 8])
        
        ST_out, amp_out, phase_out = interpolate_fft_spectrum_batch(
            self.afft, Re_vals, AOA_vals, 'CD'
        )
        
        self.assertEqual(ST_out.shape, (2, self.n_freq))
        self.assertEqual(amp_out.shape, (2, self.n_freq))
        self.assertEqual(phase_out.shape, (2, self.n_freq))
        
        # Check that values are reasonable
        self.assertTrue(np.all(np.isfinite(ST_out)))
        self.assertTrue(np.all(np.isfinite(amp_out)))
        self.assertTrue(np.all(np.isfinite(phase_out)))
    
    def test_interpolate_fft_spectrum_batch_invalid_field(self):
        """Test batch interpolation with invalid field name."""
        Re_vals = np.array([5000])
        AOA_vals = np.array([5])
        
        with self.assertRaises(ValueError):
            interpolate_fft_spectrum_batch(self.afft, Re_vals, AOA_vals, 'INVALID')


class TestInterpolateFFTSpectrumOptimized(unittest.TestCase):
    """Test cases for interpolate_fft_spectrum_optimized function."""
    
    def setUp(self):
        """Set up test AirfoilFFT data."""
        self.Re = np.array([1000, 5000, 10000])
        self.AOA = np.array([0, 5, 10, 15])
        self.n_freq = 3
        
        # Create simple test data
        shape = (len(self.Re), len(self.AOA), self.n_freq)
        self.CL_ST = np.random.rand(*shape)
        self.CL_Amp = np.random.rand(*shape)
        self.CL_Pha = np.random.rand(*shape) * 2 * np.pi
        self.CD_ST = np.random.rand(*shape)
        self.CD_Amp = np.random.rand(*shape)
        self.CD_Pha = np.random.rand(*shape) * 2 * np.pi
        self.CF_ST = np.random.rand(*shape)
        self.CF_Amp = np.random.rand(*shape)
        self.CF_Pha = np.random.rand(*shape) * 2 * np.pi
        
        self.afft = AirfoilFFT(
            name="test_airfoil",
            Re=self.Re,
            AOA=self.AOA,
            Thickness=0.12,
            CL_ST=self.CL_ST,
            CD_ST=self.CD_ST,
            CM_ST=self.CL_ST * 0.5,
            CF_ST=self.CF_ST,
            CL_Amp=self.CL_Amp,
            CD_Amp=self.CD_Amp,
            CM_Amp=self.CL_Amp * 0.5,
            CF_Amp=self.CF_Amp,
            CL_Pha=self.CL_Pha,
            CD_Pha=self.CD_Pha,
            CM_Pha=self.CL_Pha * 0.5,
            CF_Pha=self.CF_Pha
        )
    
    def test_interpolate_fft_spectrum_optimized_single_field(self):
        """Test optimized interpolation for single field."""
        Re_val = 5000.0
        AOA_val = 5.0
        
        results = interpolate_fft_spectrum_optimized(
            self.afft, Re_val, AOA_val, ['CL'], n_freq_depth=2
        )
        
        self.assertIn('CL', results)
        ST_out, amp_out, phase_out = results['CL']
        
        self.assertEqual(len(ST_out), 2)
        self.assertEqual(len(amp_out), 2)
        self.assertEqual(len(phase_out), 2)
        
        # Check that values are reasonable
        self.assertTrue(np.all(np.isfinite(ST_out)))
        self.assertTrue(np.all(np.isfinite(amp_out)))
        self.assertTrue(np.all(np.isfinite(phase_out)))
    
    def test_interpolate_fft_spectrum_optimized_multiple_fields(self):
        """Test optimized interpolation for multiple fields."""
        Re_val = 7500.0
        AOA_val = 7.5
        
        results = interpolate_fft_spectrum_optimized(
            self.afft, Re_val, AOA_val, ['CL', 'CD', 'CF'], n_freq_depth=2
        )
        
        self.assertIn('CL', results)
        self.assertIn('CD', results)
        self.assertIn('CF', results)
        
        for field in ['CL', 'CD', 'CF']:
            ST_out, amp_out, phase_out = results[field]
            self.assertEqual(len(ST_out), 2)
            self.assertEqual(len(amp_out), 2)
            self.assertEqual(len(phase_out), 2)
            
            # Check that values are reasonable
            self.assertTrue(np.all(np.isfinite(ST_out)))
            self.assertTrue(np.all(np.isfinite(amp_out)))
            self.assertTrue(np.all(np.isfinite(phase_out)))
    
    def test_interpolate_fft_spectrum_optimized_invalid_field(self):
        """Test optimized interpolation with invalid field name."""
        with self.assertRaises(ValueError):
            interpolate_fft_spectrum_optimized(
                self.afft, 5000.0, 5.0, ['INVALID']
            )


class TestResampleAirfoil(unittest.TestCase):
    """Test cases for resample_airfoil function."""
    
    def setUp(self):
        """Set up test airfoil data."""
        # Create a simple airfoil shape
        self.airfoil = np.array([
            [1.0, 0.0],    # Trailing edge
            [0.5, 0.1],    # Upper surface
            [0.0, 0.0],    # Leading edge
            [0.5, -0.1],   # Lower surface
            [1.0, 0.0]     # Trailing edge (duplicate)
        ])
    
    def test_resample_airfoil_default_points(self):
        """Test airfoil resampling with default number of points."""
        resampled = resample_airfoil(self.airfoil)
        
        self.assertEqual(len(resampled), 200)
        self.assertEqual(resampled.shape[1], 2)  # x, y coordinates
        
        # Check that x-coordinates are in [0, 1] range
        self.assertGreaterEqual(np.min(resampled[:, 0]), 0.0)
        self.assertLessEqual(np.max(resampled[:, 0]), 1.0)
    
    def test_resample_airfoil_custom_points(self):
        """Test airfoil resampling with custom number of points."""
        npoints = 50
        resampled = resample_airfoil(self.airfoil, npoints=npoints)
        
        self.assertEqual(len(resampled), npoints)
        self.assertEqual(resampled.shape[1], 2)
        
        # Check that x-coordinates are in [0, 1] range
        self.assertGreaterEqual(np.min(resampled[:, 0]), 0.0)
        self.assertLessEqual(np.max(resampled[:, 0]), 1.0)
    
    def test_resample_airfoil_symmetric(self):
        """Test that resampled airfoil maintains reasonable shape."""
        resampled = resample_airfoil(self.airfoil, npoints=100)
        
        # Find leading edge (minimum x)
        le_idx = np.argmin(resampled[:, 0])
        
        # Check that leading edge is at x=0
        self.assertAlmostEqual(resampled[le_idx, 0], 0.0, places=2)
        
        # Check that trailing edges are at x=1
        te_indices = np.where(np.abs(resampled[:, 0] - 1.0) < 0.01)[0]
        self.assertGreater(len(te_indices), 0)
    
    def test_resample_airfoil_single_point(self):
        """Test resampling with very few points."""
        resampled = resample_airfoil(self.airfoil, npoints=4)
        
        self.assertEqual(len(resampled), 4)
        self.assertEqual(resampled.shape[1], 2)
        
        # Should still have reasonable x-range
        self.assertGreaterEqual(np.min(resampled[:, 0]), 0.0)
        self.assertLessEqual(np.max(resampled[:, 0]), 1.0)
    
    def test_resample_airfoil_irregular_input(self):
        """Test resampling with irregular input airfoil."""
        # Create irregular airfoil with non-monotonic x-coordinates
        irregular_airfoil = np.array([
            [1.0, 0.0],
            [0.8, 0.05],
            [0.3, 0.1],
            [0.0, 0.0],
            [0.2, -0.08],
            [0.7, -0.05],
            [1.0, 0.0]
        ])
        
        resampled = resample_airfoil(irregular_airfoil, npoints=50)
        
        self.assertEqual(len(resampled), 50)
        self.assertEqual(resampled.shape[1], 2)
        
        # Should still produce valid output
        self.assertTrue(np.all(np.isfinite(resampled)))


if __name__ == '__main__':
    unittest.main()





