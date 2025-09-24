"""
Unit tests for vorlap.fileio module.
"""

import unittest
import numpy as np
import os
import tempfile
import h5py
import pandas as pd
from vorlap.fileio import (
    load_components_from_csv,
    load_airfoil_fft,
    load_airfoil_coords,
    write_force_time_series
)
from vorlap.structs import AirfoilFFT, Component


class TestLoadComponentsFromCSV(unittest.TestCase):
    """Test cases for load_components_from_csv function."""
    
    def setUp(self):
        """Set up test data and temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV data
        self.csv_content = """id,translation_x,translation_y,translation_z,rotation_x,rotation_y,rotation_z,pitch
blade1,1.0,2.0,3.0,10.0,20.0,30.0,5.0
x,y,z,chord,twist,thickness,offset,airfoil_id
0.0,0.0,0.0,1.0,0.0,0.12,0.0,default
1.0,0.0,0.0,1.0,5.0,0.12,0.0,default
2.0,0.0,0.0,1.0,10.0,0.12,0.0,default"""
        
        self.csv_file = os.path.join(self.temp_dir, "blade1.csv")
        with open(self.csv_file, 'w') as f:
            f.write(self.csv_content)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_components_from_csv(self):
        """Test loading components from CSV file."""
        components = load_components_from_csv(self.temp_dir)
        
        self.assertEqual(len(components), 1)
        comp = components[0]
        
        self.assertEqual(comp.id, "blade1")
        np.testing.assert_array_equal(comp.translation, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(comp.rotation, np.array([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(comp.pitch, np.array([5.0]))
        
        # Check shape data
        expected_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        np.testing.assert_array_equal(comp.shape_xyz, expected_xyz)
        np.testing.assert_array_equal(comp.chord, np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(comp.twist, np.array([0.0, 5.0, 10.0]))
        np.testing.assert_array_equal(comp.thickness, np.array([0.12, 0.12, 0.12]))
        np.testing.assert_array_equal(comp.offset, np.array([0.0, 0.0, 0.0]))
        self.assertEqual(comp.airfoil_ids, ["default", "default", "default"])
    
    def test_load_components_from_csv_no_airfoil_id(self):
        """Test loading components when airfoil_id column is missing."""
        csv_content_no_airfoil = """id,translation_x,translation_y,translation_z,rotation_x,rotation_y,rotation_z,pitch
blade1,1.0,2.0,3.0,10.0,20.0,30.0,5.0
x,y,z,chord,twist,thickness,offset
0.0,0.0,0.0,1.0,0.0,0.12,0.0
1.0,0.0,0.0,1.0,5.0,0.12,0.0"""
        
        csv_file = os.path.join(self.temp_dir, "blade2.csv")
        with open(csv_file, 'w') as f:
            f.write(csv_content_no_airfoil)
        
        components = load_components_from_csv(self.temp_dir)
        
        self.assertEqual(len(components), 2)  # blade1 + blade2
        comp2 = components[1]
        self.assertEqual(comp2.airfoil_ids, ["default", "default"])


class TestLoadAirfoilFFT(unittest.TestCase):
    """Test cases for load_airfoil_fft function."""
    
    def setUp(self):
        """Set up test HDF5 file."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        self.temp_file.close()
        
        # Create test HDF5 data
        with h5py.File(self.temp_file.name, 'w') as h5:
            h5['Airfoilname'] = b'NACA0012'
            h5['Re'] = np.array([1000, 5000, 10000])
            h5['Thickness'] = np.array([0.12])
            h5['AOA'] = np.array([0, 5, 10, 15])
            
            # Create 3D arrays with shape [Re, AOA, freq]
            shape = (3, 4, 5)
            h5['CL_ST'] = np.random.rand(*shape)
            h5['CD_ST'] = np.random.rand(*shape)
            h5['CM_ST'] = np.random.rand(*shape)
            h5['CF_ST'] = np.random.rand(*shape)
            h5['CL_Amp'] = np.random.rand(*shape)
            h5['CD_Amp'] = np.random.rand(*shape)
            h5['CM_Amp'] = np.random.rand(*shape)
            h5['CF_Amp'] = np.random.rand(*shape)
            h5['CL_Pha'] = np.random.rand(*shape) * 2 * np.pi
            h5['CD_Pha'] = np.random.rand(*shape) * 2 * np.pi
            h5['CM_Pha'] = np.random.rand(*shape) * 2 * np.pi
            h5['CF_Pha'] = np.random.rand(*shape) * 2 * np.pi
    
    def tearDown(self):
        """Clean up temporary file."""
        os.unlink(self.temp_file.name)
    
    def test_load_airfoil_fft(self):
        """Test loading airfoil FFT data from HDF5 file."""
        afft = load_airfoil_fft(self.temp_file.name)
        
        self.assertEqual(afft.name, "NACA0012")
        np.testing.assert_array_equal(afft.Re, np.array([1000, 5000, 10000]))
        np.testing.assert_array_equal(afft.AOA, np.array([0, 5, 10, 15]))
        self.assertEqual(afft.Thickness, 0.12)
        
        # Check array shapes
        expected_shape = (3, 4, 5)
        self.assertEqual(afft.CL_ST.shape, expected_shape)
        self.assertEqual(afft.CD_ST.shape, expected_shape)
        self.assertEqual(afft.CM_ST.shape, expected_shape)
        self.assertEqual(afft.CF_ST.shape, expected_shape)
        self.assertEqual(afft.CL_Amp.shape, expected_shape)
        self.assertEqual(afft.CD_Amp.shape, expected_shape)
        self.assertEqual(afft.CM_Amp.shape, expected_shape)
        self.assertEqual(afft.CF_Amp.shape, expected_shape)
        self.assertEqual(afft.CL_Pha.shape, expected_shape)
        self.assertEqual(afft.CD_Pha.shape, expected_shape)
        self.assertEqual(afft.CM_Pha.shape, expected_shape)
        self.assertEqual(afft.CF_Pha.shape, expected_shape)


class TestLoadAirfoilCoords(unittest.TestCase):
    """Test cases for load_airfoil_coords function."""
    
    def setUp(self):
        """Set up test airfoil file."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.temp_file.close()
        
        # Create test airfoil coordinates
        self.airfoil_data = np.array([
            [0.0, 0.0],
            [0.5, 0.1],
            [1.0, 0.0]
        ])
        
        np.savetxt(self.temp_file.name, self.airfoil_data, delimiter=',')
    
    def tearDown(self):
        """Clean up temporary file."""
        os.unlink(self.temp_file.name)
    
    def test_load_airfoil_coords_from_file(self):
        """Test loading airfoil coordinates from file."""
        coords = load_airfoil_coords(self.temp_file.name)
        
        # Check that coordinates are normalized
        self.assertAlmostEqual(np.min(coords[:, 0]), 0.0, places=10)
        self.assertAlmostEqual(np.max(coords[:, 0]), 1.0, places=10)
        self.assertEqual(len(coords), 3)
    
    def test_load_airfoil_coords_nonexistent_file(self):
        """Test loading airfoil coordinates when file doesn't exist."""
        coords = load_airfoil_coords("nonexistent_file.csv")
        
        # Should return default Clark Y airfoil
        self.assertGreater(len(coords), 100)  # Default airfoil has many points
        self.assertAlmostEqual(np.min(coords[:, 0]), 0.0, places=10)
        self.assertAlmostEqual(np.max(coords[:, 0]), 1.0, places=10)


class TestWriteForceTimeSeries(unittest.TestCase):
    """Test cases for write_force_time_series function."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.temp_file.close()
        
        self.output_time = np.array([0.0, 0.1, 0.2])
        self.global_force_vector_nodes = np.random.rand(3, 3, 2)  # 3 time points, 3 components, 2 nodes
    
    def tearDown(self):
        """Clean up temporary file."""
        os.unlink(self.temp_file.name)
    
    def test_write_force_time_series(self):
        """Test writing force time series to CSV."""
        write_force_time_series(self.temp_file.name, self.output_time, self.global_force_vector_nodes)
        
        # Read back the data
        data = np.loadtxt(self.temp_file.name, delimiter=',', skiprows=1)
        
        # Check dimensions
        self.assertEqual(data.shape[0], 3)  # 3 time points
        self.assertEqual(data.shape[1], 7)  # time + 2 nodes * 3 components
        
        # Check time column
        np.testing.assert_array_almost_equal(data[:, 0], self.output_time)
        
        # Check force data
        for t in range(3):
            for n in range(2):
                expected_forces = self.global_force_vector_nodes[t, :, n]
                actual_forces = data[t, 1 + n*3:4 + n*3]
                np.testing.assert_array_almost_equal(actual_forces, expected_forces)
    
    def test_write_force_time_series_header(self):
        """Test that CSV header is written correctly."""
        write_force_time_series(self.temp_file.name, self.output_time, self.global_force_vector_nodes)
        
        with open(self.temp_file.name, 'r') as f:
            header = f.readline().strip()
        
        expected_header = "time, node1x, node1y, node1z, node2x, node2y, node2z"
        self.assertEqual(header, expected_header)


if __name__ == '__main__':
    unittest.main()





