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
        self.sim_name = ttk.Entry(self)
        self.sim_name.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
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
        
        # Set default frequency path
        # default_path = "../../../componentsHVAWT/freqs.csv"
        # self.freq_path.set(default_path)
        
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

        # Simulation Parameters (moved below frequencies)
        lpars = ttk.LabelFrame(self, text="Simulation Parameters")
        lpars.grid(row=row, column=0, columnspan=6, sticky="nsew", pady=(10, 6))
        ttk.Label(lpars, text="File Path").grid(row=0, column=0, sticky="w")
        self.param_path = PathEntry(lpars, kind="file", title="Select parameters CSV", must_exist=True)
        self.param_path.grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(lpars, text="Import", command=self.import_params).grid(row=1, column=1, padx=6)
        ttk.Label(lpars, text="Simulation Parameters List").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.param_table = EditableTreeview(lpars, columns=["Parameter", "Value"])
        self.param_table.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(2, 0))
        lpars.columnconfigure(0, weight=1)
        lpars.rowconfigure(3, weight=1)
        row += 1

        # Components section
        lcomp = ttk.LabelFrame(self, text="Component Geometric Definition")
        lcomp.grid(row=row, column=0, columnspan=6, sticky="nsew", pady=(10, 6))
        
        ttk.Label(lcomp, text="Components Directory").grid(row=0, column=0, sticky="w")
        self.components_path = PathEntry(lcomp, kind="dir", title="Select components directory", must_exist=True)
        self.components_path.grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(lcomp, text="Import", command=self.load_components).grid(row=1, column=1, padx=6)
        
        # Component geometry table
        cols = [
            "Name","Dx","Dy","Dz","Rx","Ry","Rz","Pitch",
            "X","Y","Z","Chord","twist","thk%","offset","AirfoilPath"
        ]
        self.geom_table = EditableTreeview(lcomp, columns=cols, height=8)
        self.geom_table.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=10, pady=(6, 10))
        
        # Bottom controls
        # bottom_comp = ttk.Frame(lcomp)
        # bottom_comp.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 6))
        # self.geom_path = PathEntry(bottom_comp, kind="savefile", title="Save component geometry as CSV")
        # self.geom_path.grid(row=0, column=0, sticky="ew")
        # ttk.Button(bottom_comp, text="Load CSV", command=self.load_geom).grid(row=0, column=1, padx=4)
        # ttk.Button(bottom_comp, text="Save CSV", command=self.save_geom).grid(row=0, column=2)
        
        lcomp.columnconfigure(0, weight=1)
        
        # Set default components path
        default_path = os.path.join(vorlap.repo_dir, "componentsHVAWT")
        self.components_path.set(default_path)

        # Set default simulation name
        self.sim_name.insert(0, "vorlap_simulation")
        
        # Set default save path (same directory as the script)
        default_save_path = vorlap.repo_dir
        self.sim_save.set(default_save_path)
        
        # Default frequency file path
        default_freq_path = os.path.join(vorlap.repo_dir, "natural_frequencies.csv")
        self.freq_path.set(default_freq_path)

        # Initialize with default parameters
        self._populate_default_params()

        # Make grid flexible
        for c in range(6):
            self.columnconfigure(c, weight=(1 if c in (1, 3) else 0))
        self.rowconfigure(row, weight=1)

    def _populate_default_params(self):
        """Populate the parameters table with default VIV_Params values."""
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
            ("azimuth_step", "5"),
            ("inflow_speed_start", "2.0"),
            ("inflow_speed_end", "50.0"),
            ("inflow_speed_step", "0.5"),
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
            self.param_table.append_row([param, value])

    def get_viv_params(self):
        """Create VIV_Params object from the current parameter table."""
        params = {}
        for row in self.param_table.get_all():
            if len(row) >= 2:
                key, value = row[0], row[1]
                if value.strip():  # Only process non-empty values
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
        airfoil_folder = os.path.join(vorlap.repo_dir, "airfoils")

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
                self.app.log(f"  Simulation name: {self.sim_name.get()}\n")
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
                
                # Create labels for each frequency value
                for i, val in enumerate(row):
                    try:
                        freq_val = float(val.strip())
                        label = ttk.Label(self.freq_inner_frame, text=f"{freq_val:.2f} Hz", 
                                        relief="solid", borderwidth=1, padding=(8, 4))
                        label.grid(row=0, column=i, padx=2)
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
            self.param_table.load_csv(path)
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
            
            # Update geometry table with first component as example
            if components:
                self.geom_table.clear()
                comp = components[0]
                for i in range(len(comp.shape_xyz)):
                    # Safely get values with defaults
                    pitch_val = comp.pitch[0] if len(comp.pitch) > 0 else 0  # pitch is usually a single value
                    chord_val = comp.chord[i] if i < len(comp.chord) else 0
                    twist_val = comp.twist[i] if i < len(comp.twist) else 0
                    thickness_val = comp.thickness[i] if i < len(comp.thickness) else 0.18
                    offset_val = comp.offset[i] if i < len(comp.offset) else 0
                    airfoil_id = comp.airfoil_ids[i] if i < len(comp.airfoil_ids) else "default"
                    
                    row = [
                        f"Node_{i}",
                        str(comp.shape_xyz[i, 0]), str(comp.shape_xyz[i, 1]), str(comp.shape_xyz[i, 2]),
                        "0", "0", "0", str(pitch_val),
                        str(comp.shape_xyz[i, 0]), str(comp.shape_xyz[i, 1]), str(comp.shape_xyz[i, 2]),
                        str(chord_val),
                        str(twist_val),
                        str(thickness_val),
                        str(offset_val), 
                        airfoil_id
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
        """Get natural frequencies from the frequency display."""
        frequencies = []
        for widget in self.freq_inner_frame.winfo_children():
            try:
                # Extract frequency value from label text (e.g., "3.00 Hz" -> 3.00)
                text = widget.cget("text")
                freq_val = float(text.replace(" Hz", ""))
                frequencies.append(freq_val)
            except (ValueError, AttributeError):
                continue
        return frequencies if frequencies else None 