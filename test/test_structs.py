"""
Unit tests for vorlap.structs module.
"""

import unittest
import numpy as np
import os
import tempfile
import h5py
from vorlap.structs import AirfoilFFT, Component, VIV_Params


class TestAirfoilFFT(unittest.TestCase):
    """Test cases for AirfoilFFT class."""
    
    def setUp(self):
        """Set up test data."""
        self.name = "test_airfoil"
        self.Re = np.array([1000, 5000, 10000])
        self.AOA = np.array([0, 5, 10, 15])
        self.thickness = 0.12
        self.n_freq = 5
        
        # Create test FFT data
        shape = (len(self.Re), len(self.AOA), self.n_freq)
        self.CL_ST = np.random.rand(*shape)
        self.CD_ST = np.random.rand(*shape)
        self.CM_ST = np.random.rand(*shape)
        self.CF_ST = np.random.rand(*shape)
        self.CL_Amp = np.random.rand(*shape)
        self.CD_Amp = np.random.rand(*shape)
        self.CM_Amp = np.random.rand(*shape)
        self.CF_Amp = np.random.rand(*shape)
        self.CL_Pha = np.random.rand(*shape) * 2 * np.pi
        self.CD_Pha = np.random.rand(*shape) * 2 * np.pi
        self.CM_Pha = np.random.rand(*shape) * 2 * np.pi
        self.CF_Pha = np.random.rand(*shape) * 2 * np.pi
    
    def test_airfoil_fft_initialization(self):
        """Test AirfoilFFT initialization."""
        afft = AirfoilFFT(
            name=self.name,
            Re=self.Re,
            AOA=self.AOA,
            Thickness=self.thickness,
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
        
        self.assertEqual(afft.name, self.name)
        np.testing.assert_array_equal(afft.Re, self.Re)
        np.testing.assert_array_equal(afft.AOA, self.AOA)
        self.assertEqual(afft.Thickness, self.thickness)
        np.testing.assert_array_equal(afft.CL_ST, self.CL_ST)
        self.assertFalse(afft._interpolators_cached)
    
    def test_cache_interpolators(self):
        """Test interpolator caching functionality."""
        afft = AirfoilFFT(
            name=self.name,
            Re=self.Re,
            AOA=self.AOA,
            Thickness=self.thickness,
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
        
        # Cache interpolators
        afft._cache_interpolators()
        
        self.assertTrue(afft._interpolators_cached)
        self.assertEqual(len(afft._cl_st_interps), self.n_freq)
        self.assertEqual(len(afft._cl_amp_interps), self.n_freq)
        self.assertEqual(len(afft._cl_pha_interps), self.n_freq)
        self.assertEqual(len(afft._cd_st_interps), self.n_freq)
        self.assertEqual(len(afft._cd_amp_interps), self.n_freq)
        self.assertEqual(len(afft._cd_pha_interps), self.n_freq)
        self.assertEqual(len(afft._cf_st_interps), self.n_freq)
        self.assertEqual(len(afft._cf_amp_interps), self.n_freq)
        self.assertEqual(len(afft._cf_pha_interps), self.n_freq)
    
    def test_cache_interpolators_idempotent(self):
        """Test that caching interpolators multiple times doesn't cause issues."""
        afft = AirfoilFFT(
            name=self.name,
            Re=self.Re,
            AOA=self.AOA,
            Thickness=self.thickness,
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
        
        # Cache multiple times
        afft._cache_interpolators()
        afft._cache_interpolators()
        afft._cache_interpolators()
        
        self.assertTrue(afft._interpolators_cached)


class TestComponent(unittest.TestCase):
    """Test cases for Component class."""
    
    def setUp(self):
        """Set up test data."""
        self.id = "test_blade"
        self.translation = np.array([1.0, 2.0, 3.0])
        self.rotation = np.array([10.0, 20.0, 30.0])
        self.pitch = np.array([5.0])
        self.shape_xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        self.shape_xyz_global = np.zeros_like(self.shape_xyz)
        self.chord = np.array([1.0, 1.0, 1.0])
        self.twist = np.array([0.0, 5.0, 10.0])
        self.thickness = np.array([0.12, 0.12, 0.12])
        self.offset = np.array([0.0, 0.0, 0.0])
        self.airfoil_ids = ["default", "default", "default"]
        self.chord_vector = np.zeros((3, 3))
        self.normal_vector = np.zeros((3, 3))
    
    def test_component_initialization(self):
        """Test Component initialization."""
        comp = Component(
            id=self.id,
            translation=self.translation,
            rotation=self.rotation,
            pitch=self.pitch,
            shape_xyz=self.shape_xyz,
            shape_xyz_global=self.shape_xyz_global,
            chord=self.chord,
            twist=self.twist,
            thickness=self.thickness,
            offset=self.offset,
            airfoil_ids=self.airfoil_ids,
            chord_vector=self.chord_vector,
            normal_vector=self.normal_vector
        )
        
        self.assertEqual(comp.id, self.id)
        np.testing.assert_array_equal(comp.translation, self.translation)
        np.testing.assert_array_equal(comp.rotation, self.rotation)
        np.testing.assert_array_equal(comp.pitch, self.pitch)
        np.testing.assert_array_equal(comp.shape_xyz, self.shape_xyz)
        np.testing.assert_array_equal(comp.chord, self.chord)
        np.testing.assert_array_equal(comp.twist, self.twist)
        np.testing.assert_array_equal(comp.thickness, self.thickness)
        np.testing.assert_array_equal(comp.offset, self.offset)
        self.assertEqual(comp.airfoil_ids, self.airfoil_ids)


class TestVIVParams(unittest.TestCase):
    """Test cases for VIV_Params class."""
    
    def test_viv_params_default_initialization(self):
        """Test VIV_Params with default values."""
        params = VIV_Params()
        
        self.assertEqual(params.fluid_density, 1.225)
        self.assertEqual(params.fluid_dynamicviscosity, 1.81e-5)
        np.testing.assert_array_equal(params.rotation_axis, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_array_equal(params.rotation_axis_offset, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(params.inflow_vec, np.array([1.0, 0.0, 0.0]))
        self.assertEqual(params.n_harmonic, 5)
        self.assertEqual(params.amplitude_coeff_cutoff, 0.01)
        self.assertEqual(params.n_freq_depth, 3)
        self.assertEqual(params.output_azimuth_vinf, (5.0, 2.0))
    
    def test_viv_params_custom_initialization(self):
        """Test VIV_Params with custom values."""
        custom_azimuths = np.arange(0, 180, 10)
        custom_speeds = np.arange(5, 25, 2)
        custom_time = np.arange(0, 5, 0.1)
        
        params = VIV_Params(
            fluid_density=1.0,
            fluid_dynamicviscosity=2.0e-5,
            rotation_axis=np.array([1.0, 0.0, 0.0]),
            rotation_axis_offset=np.array([0.0, 0.0, 1.0]),
            inflow_vec=np.array([0.0, 1.0, 0.0]),
            azimuths=custom_azimuths,
            inflow_speeds=custom_speeds,
            output_time=custom_time,
            n_harmonic=3,
            amplitude_coeff_cutoff=0.05,
            n_freq_depth=5,
            output_azimuth_vinf=(15.0, 10.0)
        )
        
        self.assertEqual(params.fluid_density, 1.0)
        self.assertEqual(params.fluid_dynamicviscosity, 2.0e-5)
        np.testing.assert_array_equal(params.rotation_axis, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(params.rotation_axis_offset, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_array_equal(params.inflow_vec, np.array([0.0, 1.0, 0.0]))
        np.testing.assert_array_equal(params.azimuths, custom_azimuths)
        np.testing.assert_array_equal(params.inflow_speeds, custom_speeds)
        np.testing.assert_array_equal(params.output_time, custom_time)
        self.assertEqual(params.n_harmonic, 3)
        self.assertEqual(params.amplitude_coeff_cutoff, 0.05)
        self.assertEqual(params.n_freq_depth, 5)
        self.assertEqual(params.output_azimuth_vinf, (15.0, 10.0))
    
    def test_viv_params_plot_cycle_default(self):
        """Test default plot cycle colors."""
        params = VIV_Params()
        expected_colors = ["#348ABD", "#A60628", "#009E73", "#7A68A6", "#D55E00", "#CC79A7"]
        self.assertEqual(params.plot_cycle, expected_colors)
    
    def test_viv_params_custom_plot_cycle(self):
        """Test custom plot cycle colors."""
        custom_colors = ["#FF0000", "#00FF00", "#0000FF"]
        params = VIV_Params(plot_cycle=custom_colors)
        self.assertEqual(params.plot_cycle, custom_colors)


if __name__ == '__main__':
    unittest.main()





