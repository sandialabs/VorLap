#!/usr/bin/env python3
"""
Simulation Setup Tab for the VorLap GUI.

This tab handles simulation configuration, parameters, and running analysis.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import numpy as np
import time
import threading

# Add the vorlap package to the path
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import vorlap

from ..widgets import PathEntry, EditableTreeview


class SimulationSetupTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._build()

    def _build(self):
        # Top: Sim name/path + full sim save path + Run & Save
        row = 0
        ttk.Label(self, text="Sim Name/Path").grid(row=row, column=0, sticky="w")
        ttk.Label(self, text="Full Simulation Save Path").grid(row=row, column=2, sticky="w")
        self.sim_save = PathEntry(self, kind="dir", title="Choose simulation save directory")
        self.sim_save.grid(row=row, column=3, sticky="ew", padx=(6, 0))
        
        # Mock mode checkbox
        self.use_mock = tk.BooleanVar(value=False)
        self.mock_check = ttk.Checkbutton(self, text="Use Mock Data (Fast)", 
                                         variable=self.use_mock,
                                         style='Accent.TCheckbutton')
        self.mock_check.grid(row=row, column=4, padx=6, sticky="w")
        
        self.run_btn = ttk.Button(self, text="Run & Save", command=self.run_and_save)
        self.run_btn.grid(row=row, column=5, padx=6)
        row += 1

        # Add a separator
        sep = ttk.Separator(self, orient='horizontal')
        sep.grid(row=row, column=0, columnspan=6, sticky="ew", pady=(10, 0))
        row += 1
        
        # Parked Modal Frequencies (simplified for single-row CSV)
        lfreq = ttk.LabelFrame(self, text="Parked Modal Frequencies")
        lfreq.grid(row=row, column=0, columnspan=6, sticky="nsew", pady=(10, 6))
        
        ttk.Label(lfreq, text="File Path").grid(row=0, column=0, sticky="w")
        self.freq_path = PathEntry(lfreq, kind="file", title="Select frequency CSV", must_exist=True)
        self.freq_path.grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(lfreq, text="Import", command=self.import_freq).grid(row=1, column=1, padx=6)
        ttk.Label(lfreq, text="Frequencies [Hz]").grid(row=2, column=0, sticky="w", pady=(6, 0))
        
        # Horizontal scrollable frame for frequencies
        freq_frame = ttk.Frame(lfreq)
        freq_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        self.freq_canvas = tk.Canvas(freq_frame, height=30)
        freq_scrollbar = ttk.Scrollbar(freq_frame, orient="horizontal", command=self.freq_canvas.xview)
        self.freq_canvas.configure(xscrollcommand=freq_scrollbar.set)
        
        self.freq_canvas.grid(row=0, column=0, sticky="ew")
        freq_scrollbar.grid(row=1, column=0, sticky="ew")
        freq_frame.columnconfigure(0, weight=1)
        
        # Frame inside canvas to hold frequency labels
        self.freq_inner_frame = ttk.Frame(self.freq_canvas)
        self.freq_canvas.create_window((0, 0), window=self.freq_inner_frame, anchor="nw")
        
        lfreq.columnconfigure(0, weight=1)
        row += 1

        # Simulation Parameters
        lpars = ttk.LabelFrame(self, text="Simulation Parameters")
        lpars.grid(row=row, column=0, columnspan=6, sticky="ew", pady=(10, 6))
        ttk.Label(lpars, text="File Path").grid(row=0, column=0, sticky="w")
        self.param_path = PathEntry(lpars, kind="file", title="Select parameters CSV", must_exist=True)
        self.param_path.grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(lpars, text="Import", command=self.import_params).grid(row=1, column=1, padx=6)
        ttk.Label(lpars, text="Simulation Parameters List").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.param_table = EditableTreeview(lpars, columns=["Description", "Parameter", "Value"], non_editable_columns=["Description"])
        self.param_table.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        lpars.columnconfigure(0, weight=1)
        lpars.rowconfigure(3, weight=0)  # Fixed weight for parameter table (non-collapsible)
        row += 1

        # Components section
        lcomp = ttk.LabelFrame(self, text="Component Geometric Definition")
        lcomp.grid(row=row, column=0, columnspan=6, sticky="ew", pady=(10, 6))
        ttk.Label(lcomp, text="Components Directory").grid(row=0, column=0, sticky="w")
        self.components_path = PathEntry(lcomp, kind="dir", title="Select components directory", must_exist=True)
        self.components_path.grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(lcomp, text="Import", command=self.load_components).grid(row=1, column=1, padx=6)
        ttk.Label(lcomp, text="Component Geometry List").grid(row=2, column=0, sticky="w", pady=(6, 0))
        
        # Component geometry table
        cols = [
            "Component ID", "Translation X", "Translation Y", "Translation Z", 
            "Rotation X", "Rotation Y", "Rotation Z", "Pitch", 
            "Segments", "Avg Chord", "Avg Twist", "Avg Thickness"
        ]
        self.geom_table = EditableTreeview(lcomp, columns=cols, height=8)
        self.geom_table.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        lcomp.columnconfigure(0, weight=1)
        lcomp.rowconfigure(3, weight=0)  # Fixed weight for geometry table (non-collapsible)
        
        # Set default components path
        default_path = os.path.join(vorlap.repo_dir, "data", "components", "componentsHVAWT")
        self.components_path.set(default_path)

        
        # Set default save path (same directory as the script)
        default_save_path = vorlap.repo_dir
        self.sim_save.set(default_save_path)
        
        # Default frequency file path
        default_freq_path = os.path.join(vorlap.repo_dir, "data", "natural_frequencies.csv")
        self.freq_path.set(default_freq_path)

        # Initialize with default parameters
        self._populate_default_params()

        # Make grid flexible
        for c in range(6):
            self.columnconfigure(c, weight=(1 if c in (1, 3) else 0))
        # Set parameter and component sections to have fixed weight (non-collapsible)
        self.rowconfigure(2, weight=0, minsize=200)  # Frequency section with minimum size
        self.rowconfigure(3, weight=0, minsize=200)  # Parameter section with minimum size  
        self.rowconfigure(4, weight=0, minsize=250)  # Component section with minimum size

    def _get_param_descriptions(self):
        """Get parameter descriptions mapping."""
        return {
            "fluid_density": "Fluid density (kg/m³)",
            "fluid_dynamicviscosity": "Fluid dynamic viscosity (Pa·s)",
            "rotation_axis_x": "X component of rotation axis vector",
            "rotation_axis_y": "Y component of rotation axis vector",
            "rotation_axis_z": "Z component of rotation axis vector",
            "rotation_axis_offset_x": "X offset of rotation axis (m)",
            "rotation_axis_offset_y": "Y offset of rotation axis (m)",
            "rotation_axis_offset_z": "Z offset of rotation axis (m)",
            "inflow_vec_x": "X component of inflow direction vector",
            "inflow_vec_y": "Y component of inflow direction vector",
            "inflow_vec_z": "Z component of inflow direction vector",
            "azimuth_start": "Starting azimuth angle (degrees)",
            "azimuth_end": "Ending azimuth angle (degrees)",
            "azimuth_step": "Step size for azimuth angle (degrees)",
            "inflow_speed_start": "Starting inflow speed (m/s)",
            "inflow_speed_end": "Ending inflow speed (m/s)",
            "inflow_speed_step": "Step size for inflow speed (m/s)",
            "n_harmonic": "Number of harmonics to consider",
            "output_time_start": "Output time series start (s)",
            "output_time_end": "Output time series end (s)",
            "output_time_step": "Output time series step size (s)",
            "output_azimuth": "Output azimuth angle (degrees)",
            "output_vinf": "Output inflow velocity (m/s)",
            "amplitude_coeff_cutoff": "Amplitude coefficient cutoff threshold",
            "n_freq_depth": "Number of frequency depth levels",
        }

    def _populate_default_params(self):
        """Populate the parameters table with default VIV_Params values."""
        descriptions = self._get_param_descriptions()
        default_params = [
            ("fluid_density", "1.225"),
            ("fluid_dynamicviscosity", "1.81e-5"),
            ("rotation_axis_x", "0.0"),
            ("rotation_axis_y", "0.0"),
            ("rotation_axis_z", "1.0"),
            ("rotation_axis_offset_x", "0.0"),
            ("rotation_axis_offset_y", "0.0"),
            ("rotation_axis_offset_z", "0.0"),
            ("inflow_vec_x", "1.0"),
            ("inflow_vec_y", "0.0"),
            ("inflow_vec_z", "0.0"),
            ("azimuth_start", "0"),
            ("azimuth_end", "255"),
            ("azimuth_step", "150"),
            # ("azimuth_step", "5"),
            ("inflow_speed_start", "2.0"),
            ("inflow_speed_end", "50.0"),
            ("inflow_speed_step", "24.0"),
            # ("inflow_speed_step", "0.5"),
            ("n_harmonic", "2"),
            ("output_time_start", "0.0"),
            ("output_time_end", "0.01"),
            ("output_time_step", "0.001"),
            ("output_azimuth", "5.0"),
            ("output_vinf", "2.0"),
            ("amplitude_coeff_cutoff", "0.2"),
            ("n_freq_depth", "10"),
        ]
        
        for param, value in default_params:
            description = descriptions.get(param, "")
            self.param_table.append_row([description, param, value])

    def get_viv_params(self):
        """Create VIV_Params object from the current parameter table."""
        params = {}
        for row in self.param_table.get_all():
            if len(row) >= 3:
                key, value = row[1], row[2]  # Parameter is at index 1, Value is at index 2
                if str(value).strip():  # Only process non-empty values
                    try:
                        # Try to convert to float first, then int if it's a whole number
                        float_val = float(value)
                        if float_val == int(float_val):
                            params[key] = int(float_val)
                        else:
                            params[key] = float_val
                    except ValueError:
                        params[key] = value  # Keep as string if conversion fails

        # Create arrays from start/end/step parameters
        azimuths = np.arange(
            params.get("azimuth_start", 0),
            params.get("azimuth_end", 255) + params.get("azimuth_step", 5),
            params.get("azimuth_step", 5)
        )
        
        inflow_speeds = np.arange(
            params.get("inflow_speed_start", 2.0),
            params.get("inflow_speed_end", 50.0) + params.get("inflow_speed_step", 0.5),
            params.get("inflow_speed_step", 0.5)
        )
        
        output_time = np.arange(
            params.get("output_time_start", 0.0),
            params.get("output_time_end", 0.01) + params.get("output_time_step", 0.001),
            params.get("output_time_step", 0.001)
        )

        # Set airfoil folder to default location
        airfoil_folder = os.path.join(vorlap.repo_dir, "data", "airfoils")

        return vorlap.VIV_Params(
            fluid_density=params.get("fluid_density", 1.225),
            fluid_dynamicviscosity=params.get("fluid_dynamicviscosity", 1.81e-5),
            rotation_axis=np.array([
                params.get("rotation_axis_x", 0.0),
                params.get("rotation_axis_y", 0.0),
                params.get("rotation_axis_z", 1.0)
            ]),
            rotation_axis_offset=np.array([
                params.get("rotation_axis_offset_x", 0.0),
                params.get("rotation_axis_offset_y", 0.0),
                params.get("rotation_axis_offset_z", 0.0)
            ]),
            inflow_vec=np.array([
                params.get("inflow_vec_x", 1.0),
                params.get("inflow_vec_y", 0.0),
                params.get("inflow_vec_z", 0.0)
            ]),
            azimuths=azimuths,
            inflow_speeds=inflow_speeds,
            n_harmonic=params.get("n_harmonic", 2),
            output_time=output_time,
            output_azimuth_vinf=(
                params.get("output_azimuth", 5.0),
                params.get("output_vinf", 2.0)
            ),
            amplitude_coeff_cutoff=params.get("amplitude_coeff_cutoff", 0.2),
            n_freq_depth=params.get("n_freq_depth", 10),
            airfoil_folder=airfoil_folder
        )

    # ---- handlers ----
    def run_and_save(self):
        """Run the complete VorLap analysis."""
        def run_analysis():
            try:
                self.app.log("Starting VorLap analysis...\n")
                self.app.log(f"  Save directory: {self.sim_save.get()}\n")
                
                # Get parameters
                viv_params = self.get_viv_params()
                
                # Get natural frequencies
                natfreqs = self.get_natural_frequencies()
                if natfreqs is None:
                    self.app.log("Error: No natural frequencies loaded\n")
                    return
                
                # Get components
                components = self.app.components
                if not components:
                    self.app.log("Error: No components loaded\n")
                    return
                
                # Load airfoils
                affts = self.app.load_airfoils(viv_params.airfoil_folder)
                if not affts:
                    self.app.log("Error: No airfoil data loaded\n")
                    return
                
                # assemble each component into a full structure, and we need the rotation axis
                # plot the full structure surface with a generic airfoil shape
                vorlap.graphics.calc_structure_vectors_andplot(components, viv_params)
            
                # Run computation (real or mock)
                if self.use_mock.get():
                    self.app.log("Running MOCK thrust/torque spectrum computation (fast)...\n")
                    start_time = time.time()
                    
                    percdiff_matrix, percdiff_info, total_global_force_vector, total_global_moment_vector, global_force_vector_nodes = vorlap.mock_compute_thrust_torque_spectrum(
                        components, affts, viv_params, natfreqs
                    )
                    
                    end_time = time.time()
                    execution_time = end_time - start_time
                    self.app.log(f"Mock computation completed in {execution_time:.4f} seconds\n")
                else:
                    self.app.log("Running thrust/torque spectrum computation...\n")
                    start_time = time.time()
                    
                    percdiff_matrix, percdiff_info, total_global_force_vector, total_global_moment_vector, global_force_vector_nodes = vorlap.compute_thrust_torque_spectrum(
                        components, affts, viv_params, natfreqs
                    )
                    
                    end_time = time.time()
                    execution_time = end_time - start_time
                    self.app.log(f"Computation completed in {execution_time:.4f} seconds\n")
                
                # Store results in app
                self.app.analysis_results = {
                    'percdiff_matrix': percdiff_matrix,
                    'percdiff_info': percdiff_info,
                    'total_global_force_vector': total_global_force_vector,
                    'total_global_moment_vector': total_global_moment_vector,
                    'global_force_vector_nodes': global_force_vector_nodes,
                    'viv_params': viv_params
                }
                
                # Save force time series if save path is provided
                if self.sim_save.get():
                    from pathlib import Path
                    save_dir = Path(self.sim_save.get())
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    force_file = save_dir / "forces_output.csv"
                    vorlap.write_force_time_series(str(force_file), viv_params.output_time, global_force_vector_nodes)
                    self.app.log(f"Force time series saved to: {force_file}\n")
                
                # Update plots
                self.app.tab_plots.update_plots()
                self.app.log("Analysis completed successfully!\n\n")
                
            except Exception as e:
                self.app.log(f"Error during analysis: {str(e)}\n")
                messagebox.showerror("Analysis Error", str(e))

        # Run analysis in separate thread to avoid GUI freezing
        threading.Thread(target=run_analysis, daemon=True).start()

    def import_freq(self):
        path = self.freq_path.get()
        if not path:
            messagebox.showwarning("No file", "Choose a CSV file.")
            return
        try:
            # Load CSV and get the first row (assuming single-row file)
            import csv
            with open(path, newline="") as f:
                reader = csv.reader(f)
                row = next(reader, [])
                if not row:
                    messagebox.showwarning("Empty file", "CSV file is empty.")
                    return
                
                # Clear previous frequencies
                for widget in self.freq_inner_frame.winfo_children():
                    widget.destroy()
                
                # Create entry widgets for each frequency value
                for i, val in enumerate(row):
                    try:
                        freq_val = float(val.strip())
                        entry = ttk.Entry(self.freq_inner_frame, width=8, justify="center")
                        entry.insert(0, f"{freq_val:.2f}")
                        entry.grid(row=0, column=i, padx=2)
                        
                        # Add validation to ensure only numeric values
                        def validate_freq(val, entry_widget=entry):
                            try:
                                if val == "":
                                    return True
                                float(val)
                                return True
                            except ValueError:
                                return False
                        
                        # Bind validation
                        entry.configure(validate="key", validatecommand=(entry.register(validate_freq), "%P"))
                    except ValueError:
                        # Skip non-numeric values
                        continue
                
                # Update canvas scroll region
                self.freq_inner_frame.update_idletasks()
                self.freq_canvas.configure(scrollregion=self.freq_canvas.bbox("all"))
                
                self.app.log(f"Loaded {len([w for w in self.freq_inner_frame.winfo_children()])} frequencies: {path}\n")
                
        except Exception as e:
            messagebox.showerror("Import failed", str(e))

    def import_params(self):
        path = self.param_path.get()
        if not path:
            messagebox.showwarning("No file", "Choose a CSV file.")
            return
        try:
            self.param_table.clear()
            descriptions = self._get_param_descriptions()
            import csv
            with open(path, newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:  # Skip rows with insufficient data
                        continue
                    
                    param_name = str(row[0]).strip()
                    param_value = row[1]
                    
                    # Convert value to float if possible
                    try:
                        param_value = float(str(param_value).strip())
                    except ValueError:
                        param_value = str(param_value)  # Keep as string if conversion fails
                    
                    # Always use predefined description, ignore any description in CSV
                    description = descriptions.get(param_name, "")
                    
                    self.param_table.append_row([description, param_name, param_value])
            
            self.app.log(f"Loaded parameters: {path}\n")
        except Exception as e:
            messagebox.showerror("Import failed", str(e))

    def load_components(self):
        """Load components from the selected directory."""
        components_dir = self.components_path.get()
        if not components_dir:
            messagebox.showwarning("No directory", "Choose a components directory.")
            return
        
        try:
            components = vorlap.load_components_from_csv(components_dir)
            self.app.components = components
            self.app.log(f"Loaded {len(components)} components from: {components_dir}\n")
            
            # Update geometry table with component-level information
            if components:
                self.geom_table.clear()
                for comp in components:
                    # Calculate average values for segments
                    avg_chord = np.mean(comp.chord) if len(comp.chord) > 0 else 0
                    avg_twist = np.mean(comp.twist) if len(comp.twist) > 0 else 0
                    avg_thickness = np.mean(comp.thickness) if len(comp.thickness) > 0 else 0
                    
                    # Get pitch value (usually a single value per component)
                    pitch_val = comp.pitch[0] if len(comp.pitch) > 0 else 0
                    
                    # Number of segments
                    num_segments = len(comp.shape_xyz)
                    
                    row = [
                        str(comp.id),
                        f"{comp.translation[0]:.3f}",
                        f"{comp.translation[1]:.3f}",
                        f"{comp.translation[2]:.3f}",
                        f"{comp.rotation[0]:.2f}",
                        f"{comp.rotation[1]:.2f}",
                        f"{comp.rotation[2]:.2f}",
                        f"{pitch_val:.2f}",
                        str(num_segments),
                        f"{avg_chord:.3f}",
                        f"{avg_twist:.2f}",
                        f"{avg_thickness:.3f}"
                    ]
                    self.geom_table.append_row(row)
                    
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            self.app.log(f"Error loading components: {str(e)}\n")

    def load_geom(self):
        path = filedialog.askopenfilename(title="Select geometry CSV")
        if not path:
            return
        try:
            self.geom_table.load_csv(path)
            self.app.log(f"Loaded component geometry: {path}\n")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def save_geom(self):
        path = self.geom_path.get() or filedialog.asksaveasfilename(title="Save component geometry CSV", defaultextension=".csv")
        if not path:
            return
        try:
            self.geom_table.save_csv(path)
            self.app.log(f"Saved component geometry: {path}\n")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def get_natural_frequencies(self):
        """Get natural frequencies from the frequency entry widgets."""
        frequencies = []
        for widget in self.freq_inner_frame.winfo_children():
            try:
                # Get frequency value from entry widget
                freq_val = float(widget.get())
                frequencies.append(freq_val)
            except (ValueError, AttributeError):
                continue
        return np.array(frequencies) if frequencies else None 