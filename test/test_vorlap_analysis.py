import os
import sys
import unittest
import numpy as np
import matplotlib.pyplot as plt
import time
import glob

import vorlap.graphics

# Add the parent directory to the path so we can import vorlap
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import vorlap


class TestVorlapOptimized(unittest.TestCase):
    """
    Unit test class for testing the optimized VorLap computation functionality.
    
    This test case verifies that the optimized compute_thrust_torque_spectrum function
    produces results that match the unit test reference data within acceptable tolerances.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with common parameters and data loading."""
        cls.path = os.path.dirname(os.path.abspath(__file__))
        
        # === Top-Level Inputs ===
        cls.viv_params = vorlap.VIV_Params(
            fluid_density=1.225,
            fluid_dynamicviscosity=1.81e-5,
            rotation_axis=np.array([0.0, 0.0, 1.0]),
            rotation_axis_offset=np.array([0.0, 0.0, 0.0]),
            inflow_vec=np.array([1.0, 0.0, 0.0]),
            azimuths=np.arange(0, 360, 10),
            inflow_speeds=np.arange(2.0, 50.0, 4.0),
            n_harmonic=2,
            output_time=np.arange(0.0, 0.011, 0.001),
            output_azimuth_vinf=(10.0, 6.0),
            amplitude_coeff_cutoff=0.002,
            n_freq_depth=10,
            airfoil_folder=f"{cls.path}/../data/airfoils/"
        )
        
        cls.nodal_force_time_file = os.path.join("forces_output_optimized.csv")
        
        # Load natural frequencies
        cls.natfreqs = np.loadtxt(f"{cls.path}/../data/natural_frequencies.csv", delimiter=',')
        
        # Load components
        cls.components = vorlap.load_components_from_csv(f"{cls.path}/../data/components/componentsHVAWT/")
        
        # Load airfoil data
        cls.affts = {}
        airfoil_folder = cls.viv_params.airfoil_folder
        
        for file in glob.glob(os.path.join(airfoil_folder, "*.h5")):
            afft = vorlap.load_airfoil_fft(file)
            cls.affts[afft.name] = afft
        
        # Set default airfoil if not present
        if "default" not in cls.affts:
            cls.affts["default"] = next(iter(cls.affts.values()))
    
    def test_optimized_compute_thrust_torque_spectrum(self):
        """Test the optimized compute_thrust_torque_spectrum function."""
        print("Running OPTIMIZED compute_thrust_torque_spectrum...")
        vorlap.graphics.calc_structure_vectors_andplot(self.components, self.viv_params)
        
        start_time = time.time()
        
        result = vorlap.compute_thrust_torque_spectrum_optimized(
            self.components, self.affts, self.viv_params, self.natfreqs
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"OPTIMIZED compute_thrust_torque_spectrum execution time: {execution_time:.4f} seconds")
        
        # Unpack results
        percdiff_matrix, percdiff_info, total_global_force_vector, total_global_moment_vector, global_force_vector_nodes = result
        
        # Verify result shapes and types
        self.assertIsInstance(percdiff_matrix, np.ndarray)
        self.assertIsInstance(total_global_force_vector, np.ndarray)
        self.assertIsInstance(total_global_moment_vector, np.ndarray)
        self.assertIsInstance(global_force_vector_nodes, np.ndarray)
        
        # Check dimensions
        expected_inflow_len = len(self.viv_params.inflow_speeds)
        expected_azimuth_len = len(self.viv_params.azimuths)
        
        self.assertEqual(total_global_force_vector.shape[:2], (expected_inflow_len, expected_azimuth_len))
        self.assertEqual(total_global_moment_vector.shape[:2], (expected_inflow_len, expected_azimuth_len))
        self.assertEqual(total_global_force_vector.shape[2], 3)  # 3D force vector
        self.assertEqual(total_global_moment_vector.shape[2], 3)  # 3D moment vector
        
        # Store results for comparison test
        self.percdiff_matrix = percdiff_matrix
        self.total_global_force_vector = total_global_force_vector
        self.total_global_moment_vector = total_global_moment_vector
        self.global_force_vector_nodes = global_force_vector_nodes
        
        # Write force time series
        vorlap.write_force_time_series(
            self.nodal_force_time_file, 
            self.viv_params.output_time, 
            self.global_force_vector_nodes
        )
        
        # Verify file was created and has content
        self.assertTrue(os.path.exists(self.nodal_force_time_file))
        
        # Load and verify file format
        data = np.loadtxt(self.nodal_force_time_file, delimiter=',', skiprows=1)
        self.assertGreater(data.shape[0], 0)
        self.assertGreater(data.shape[1], 0)
        
        print("\nComparing optimized results with unit test...")
        
        # Load computed results and reference data
        new_file = np.loadtxt(self.nodal_force_time_file, delimiter=',', skiprows=1)
        unit_file = np.loadtxt(f"{self.path}/forces_output_unit.csv", delimiter=',', skiprows=1)
        
        # Verify shapes match
        self.assertEqual(new_file.shape, unit_file.shape, 
                        f"Shape mismatch: computed {new_file.shape} vs reference {unit_file.shape}")
        
        # Compare values element by element
        for irow in range(new_file.shape[0]):
            for icol in range(new_file.shape[1]):
                # Use adaptive tolerance based on magnitude
                tolerance = max(1e-2, abs(unit_file[irow, icol]) * 1e-2)
                
                with self.subTest(row=irow, col=icol):
                    self.assertTrue(
                        np.isclose(new_file[irow, icol], unit_file[irow, icol], atol=tolerance),
                        f"Mismatch at [{irow}, {icol}]: {new_file[irow, icol]} vs {unit_file[irow, icol]}"
                    )
        
        figs_dir = os.path.join(self.path, "pyfigs_optimized")
        os.makedirs(figs_dir, exist_ok=True)
        
        try:
            # Test worst percent difference plot
            plt.figure(figsize=(10, 8))
            plt.imshow(self.percdiff_matrix, 
                      extent=[self.viv_params.azimuths[0], self.viv_params.azimuths[-1], 
                             self.viv_params.inflow_speeds[0], self.viv_params.inflow_speeds[-1]],
                      aspect='auto', origin='lower', cmap='viridis_r', vmin=0, vmax=50)
            plt.colorbar(label='Freq % Diff')
            plt.xlabel('Azimuth (deg)')
            plt.ylabel('Inflow (m/s)')
            plt.title('Worst Percent Difference (Optimized)')
            plt.savefig(f"{figs_dir}/worst_percent_diff_optimized.pdf", bbox_inches='tight')
            plt.close()
            
            # Test force plots (Fx, Fy, Fz)
            force_labels = ['Fx', 'Fy', 'Fz']
            for i, label in enumerate(force_labels):
                plt.figure(figsize=(10, 8))
                plt.imshow(self.total_global_force_vector[:, :, i], 
                          extent=[self.viv_params.azimuths[0], self.viv_params.azimuths[-1], 
                                 self.viv_params.inflow_speeds[0], self.viv_params.inflow_speeds[-1]],
                          aspect='auto', origin='lower', cmap='viridis_r')
                plt.colorbar(label='Force (N)')
                plt.xlabel('Azimuth (deg)')
                plt.ylabel('Inflow (m/s)')
                plt.title(f'{label} (Optimized)')
                plt.savefig(f"{figs_dir}/{label}_optimized.pdf", bbox_inches='tight')
                plt.close()
            
            # Test moment plots (Mx, My, Mz)
            moment_labels = ['Mx', 'My', 'Mz']
            for i, label in enumerate(moment_labels):
                plt.figure(figsize=(10, 8))
                plt.imshow(self.total_global_moment_vector[:, :, i], 
                          extent=[self.viv_params.azimuths[0], self.viv_params.azimuths[-1], 
                                 self.viv_params.inflow_speeds[0], self.viv_params.inflow_speeds[-1]],
                          aspect='auto', origin='lower', cmap='viridis_r')
                plt.colorbar(label='Moment (N-m)')
                plt.xlabel('Azimuth (deg)')
                plt.ylabel('Inflow (m/s)')
                plt.title(f'{label} (Optimized)')
                plt.savefig(f"{figs_dir}/{label}_optimized.pdf", bbox_inches='tight')
                plt.close()
            
            # Verify that plot files were created
            expected_plots = [
                'worst_percent_diff_optimized.pdf',
                'Fx_optimized.pdf', 'Fy_optimized.pdf', 'Fz_optimized.pdf',
                'Mx_optimized.pdf', 'My_optimized.pdf', 'Mz_optimized.pdf'
            ]
            
            for plot_file in expected_plots:
                plot_path = os.path.join(figs_dir, plot_file)
                self.assertTrue(os.path.exists(plot_path), f"Plot file {plot_file} was not created")
                
        except Exception as e:
            self.fail(f"Plot generation failed: {e}")


if __name__ == '__main__':
    # Configure test runner for verbose output
    unittest.main(verbosity=2)
