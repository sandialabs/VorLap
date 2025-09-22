"""
Unit tests for vorlap.graphics module.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from vorlap.graphics import calc_structure_vectors_andplot
from vorlap.structs import Component, VIV_Params


class TestCalcStructureVectorsAndPlot(unittest.TestCase):
    """Test cases for calc_structure_vectors_andplot function."""
    
    def setUp(self):
        """Set up test data."""
        # Create test component
        self.component = Component(
            id="test_blade",
            translation=np.array([0.0, 0.0, 0.0]),
            rotation=np.array([0.0, 0.0, 0.0]),
            pitch=np.array([0.0]),
            shape_xyz=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            shape_xyz_global=np.zeros((3, 3)),
            chord=np.array([1.0, 1.0, 1.0]),
            twist=np.array([0.0, 5.0, 10.0]),
            thickness=np.array([0.12, 0.12, 0.12]),
            offset=np.array([0.0, 0.0, 0.0]),
            airfoil_ids=["default", "default", "default"],
            chord_vector=np.zeros((3, 3)),
            normal_vector=np.zeros((3, 3))
        )
        
        # Create test VIV_Params
        self.viv_params = VIV_Params(
            rotation_axis=np.array([0.0, 0.0, 1.0]),
            rotation_axis_offset=np.array([0.0, 0.0, 0.0]),
            inflow_vec=np.array([1.0, 0.0, 0.0]),
            airfoil_folder=tempfile.mkdtemp()
        )
        
        # Create a dummy airfoil file
        self.airfoil_file = os.path.join(self.viv_params.airfoil_folder, "default.csv")
        airfoil_data = np.array([
            [1.0, 0.0],
            [0.5, 0.1],
            [0.0, 0.0],
            [0.5, -0.1],
            [1.0, 0.0]
        ])
        np.savetxt(self.airfoil_file, airfoil_data, delimiter=',')
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.viv_params.airfoil_folder)
    
    @patch('vorlap.graphics.go.Figure')
    @patch('vorlap.fileio.load_airfoil_coords')
    def test_calc_structure_vectors_andplot_basic(self, mock_load_airfoil, mock_figure):
        """Test basic functionality of calc_structure_vectors_andplot."""
        # Mock the airfoil loading
        mock_load_airfoil.return_value = np.array([
            [1.0, 0.0],
            [0.5, 0.1],
            [0.0, 0.0],
            [0.5, -0.1],
            [1.0, 0.0]
        ])
        
        # Mock the figure
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Call the function
        result = calc_structure_vectors_andplot(
            [self.component], 
            self.viv_params, 
            show_plot=False, 
            return_fig=False
        )
        
        # Check that figure was created and methods were called
        mock_figure.assert_called_once()
        mock_fig.add_trace.assert_called()
        mock_fig.update_layout.assert_called_once()
        
        # Should return None when return_fig=False
        self.assertIsNone(result)
    
    @patch('vorlap.graphics.go.Figure')
    @patch('vorlap.fileio.load_airfoil_coords')
    def test_calc_structure_vectors_andplot_return_fig(self, mock_load_airfoil, mock_figure):
        """Test calc_structure_vectors_andplot with return_fig=True."""
        # Mock the airfoil loading
        mock_load_airfoil.return_value = np.array([
            [1.0, 0.0],
            [0.5, 0.1],
            [0.0, 0.0],
            [0.5, -0.1],
            [1.0, 0.0]
        ])
        
        # Mock the figure
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Call the function with return_fig=True
        result = calc_structure_vectors_andplot(
            [self.component], 
            self.viv_params, 
            show_plot=False, 
            return_fig=True
        )
        
        # Should return the figure
        self.assertEqual(result, mock_fig)
    
    @patch('vorlap.graphics.go.Figure')
    @patch('vorlap.fileio.load_airfoil_coords')
    def test_calc_structure_vectors_andplot_multiple_components(self, mock_load_airfoil, mock_figure):
        """Test calc_structure_vectors_andplot with multiple components."""
        # Create second component
        component2 = Component(
            id="test_blade2",
            translation=np.array([0.0, 0.0, 1.0]),
            rotation=np.array([0.0, 0.0, 0.0]),
            pitch=np.array([0.0]),
            shape_xyz=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            shape_xyz_global=np.zeros((2, 3)),
            chord=np.array([1.0, 1.0]),
            twist=np.array([0.0, 5.0]),
            thickness=np.array([0.12, 0.12]),
            offset=np.array([0.0, 0.0]),
            airfoil_ids=["default", "default"],
            chord_vector=np.zeros((2, 3)),
            normal_vector=np.zeros((2, 3))
        )
        
        # Mock the airfoil loading
        mock_load_airfoil.return_value = np.array([
            [1.0, 0.0],
            [0.5, 0.1],
            [0.0, 0.0],
            [0.5, -0.1],
            [1.0, 0.0]
        ])
        
        # Mock the figure
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Call the function with multiple components
        result = calc_structure_vectors_andplot(
            [self.component, component2], 
            self.viv_params, 
            show_plot=False, 
            return_fig=False
        )
        
        # Check that figure was created and methods were called
        mock_figure.assert_called_once()
        mock_fig.add_trace.assert_called()
        mock_fig.update_layout.assert_called_once()
    
    @patch('vorlap.graphics.go.Figure')
    @patch('vorlap.fileio.load_airfoil_coords')
    def test_calc_structure_vectors_andplot_updates_vectors(self, mock_load_airfoil, mock_figure):
        """Test that calc_structure_vectors_andplot updates component vectors."""
        # Mock the airfoil loading
        mock_load_airfoil.return_value = np.array([
            [1.0, 0.0],
            [0.5, 0.1],
            [0.0, 0.0],
            [0.5, -0.1],
            [1.0, 0.0]
        ])
        
        # Mock the figure
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Store original vectors
        original_chord_vector = self.component.chord_vector.copy()
        original_normal_vector = self.component.normal_vector.copy()
        original_shape_xyz_global = self.component.shape_xyz_global.copy()
        
        # Call the function
        calc_structure_vectors_andplot(
            [self.component], 
            self.viv_params, 
            show_plot=False, 
            return_fig=False
        )
        
        # Check that vectors were updated (should not be all zeros anymore)
        self.assertFalse(np.array_equal(self.component.chord_vector, original_chord_vector))
        self.assertFalse(np.array_equal(self.component.normal_vector, original_normal_vector))
        self.assertFalse(np.array_equal(self.component.shape_xyz_global, original_shape_xyz_global))
    
    @patch('vorlap.graphics.go.Figure')
    @patch('vorlap.fileio.load_airfoil_coords')
    def test_calc_structure_vectors_andplot_rotation_axis(self, mock_load_airfoil, mock_figure):
        """Test that rotation axis is properly displayed."""
        # Mock the airfoil loading
        mock_load_airfoil.return_value = np.array([
            [1.0, 0.0],
            [0.5, 0.1],
            [0.0, 0.0],
            [0.5, -0.1],
            [1.0, 0.0]
        ])
        
        # Mock the figure
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Call the function
        calc_structure_vectors_andplot(
            [self.component], 
            self.viv_params, 
            show_plot=False, 
            return_fig=False
        )
        
        # Check that add_trace was called for rotation axis
        add_trace_calls = mock_fig.add_trace.call_args_list
        
        # Should have at least one call for rotation axis
        rotation_axis_called = any(
            'Rotation Axis' in str(call) for call in add_trace_calls
        )
        self.assertTrue(rotation_axis_called)
    
    @patch('vorlap.graphics.go.Figure')
    @patch('vorlap.fileio.load_airfoil_coords')
    def test_calc_structure_vectors_andplot_inflow_vector(self, mock_load_airfoil, mock_figure):
        """Test that inflow vector is properly displayed."""
        # Mock the airfoil loading
        mock_load_airfoil.return_value = np.array([
            [1.0, 0.0],
            [0.5, 0.1],
            [0.0, 0.0],
            [0.5, -0.1],
            [1.0, 0.0]
        ])
        
        # Mock the figure
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Call the function
        calc_structure_vectors_andplot(
            [self.component], 
            self.viv_params, 
            show_plot=False, 
            return_fig=False
        )
        
        # Check that add_trace was called for inflow vector
        add_trace_calls = mock_fig.add_trace.call_args_list
        
        # Should have at least one call for inflow vector
        inflow_called = any(
            'Inflow' in str(call) for call in add_trace_calls
        )
        self.assertTrue(inflow_called)
    
    @patch('vorlap.graphics.go.Figure')
    @patch('vorlap.fileio.load_airfoil_coords')
    def test_calc_structure_vectors_andplot_layout_update(self, mock_load_airfoil, mock_figure):
        """Test that layout is properly updated."""
        # Mock the airfoil loading
        mock_load_airfoil.return_value = np.array([
            [1.0, 0.0],
            [0.5, 0.1],
            [0.0, 0.0],
            [0.5, -0.1],
            [1.0, 0.0]
        ])
        
        # Mock the figure
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Call the function
        calc_structure_vectors_andplot(
            [self.component], 
            self.viv_params, 
            show_plot=False, 
            return_fig=False
        )
        
        # Check that update_layout was called
        mock_fig.update_layout.assert_called_once()
        
        # Check that the layout call includes scene configuration
        layout_call = mock_fig.update_layout.call_args
        self.assertIn('scene', layout_call[1])


if __name__ == '__main__':
    unittest.main()
