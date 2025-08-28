#!/usr/bin/env python3
"""
VORtex overLAP Tool – Tkinter GUI with tabs, editable tables, and integrated plotting.

Organization
------------
Notebook tabs:
1) Simulation Setup
2) Components
3) Plots & Outputs
4) Analysis

This GUI integrates with the VorLap backend to provide a complete wind turbine analysis interface.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import csv
import numpy as np
import os
import sys
import glob
import time
import threading

# Add the vorlap package to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import vorlap

# --- Optional plotting support ---
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False


# ---------- Small UI helpers ----------
class PathEntry(ttk.Frame):
    """Entry + Browse button (file or directory)."""
    def __init__(self, master, kind="file", title="Select...", must_exist=False, **kwargs):
        super().__init__(master, **kwargs)
        self.kind = kind          # "file" | "dir" | "savefile"
        self.title = title
        self.must_exist = must_exist
        self.var = tk.StringVar()
        self.entry = ttk.Entry(self, textvariable=self.var)
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.btn = ttk.Button(self, text="Browse", command=self.browse)
        self.btn.grid(row=0, column=1)
        self.columnconfigure(0, weight=1)

    def browse(self):
        if self.kind == "file":
            path = filedialog.askopenfilename(title=self.title)
        elif self.kind == "savefile":
            path = filedialog.asksaveasfilename(title=self.title)
        else:
            path = filedialog.askdirectory(title=self.title)
        if path:
            if self.must_exist and not Path(path).exists():
                messagebox.showerror("Path not found", f"{path}\n\ndoes not exist.")
                return
            self.var.set(path)

    def get(self) -> str:
        return self.var.get()

    def set(self, value: str):
        self.var.set(value or "")


class ScrollText(ttk.Frame):
    """A Text widget with a vertical scrollbar."""
    def __init__(self, master, height=10, **kwargs):
        super().__init__(master, **kwargs)
        self.text = tk.Text(self, wrap="word", height=height,
                           font=('Segoe UI', 10),
                           bg='#ffffff',
                           fg='#2d3748',
                           selectbackground='#4299e1',
                           selectforeground='#ffffff',
                           insertbackground='#2d3748',
                           borderwidth=1,
                           relief='solid',
                           padx=8,
                           pady=6)
        sb = ttk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def write(self, s: str):
        self.text.insert("end", s)
        self.text.see("end")

    def clear(self):
        self.text.delete("1.0", "end")


class EditableTreeview(ttk.Frame):
    """
    Spreadsheet-like table with CSV load/save and inline cell editing (double-click).
    """
    def __init__(self, master, columns, show_headings=True, height=8, **kwargs):
        super().__init__(master, **kwargs)
        self.columns = columns
        self.tree = ttk.Treeview(self, columns=columns, show=("headings" if show_headings else ""))
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")
        vsb = ttk.Scrollbar(self, command=self.tree.yview)
        hsb = ttk.Scrollbar(self, command=self.tree.xview, orient="horizontal")
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._editor = None
        self.tree.bind("<Double-1>", self._begin_edit)

    # ---- data helpers ----
    def clear(self):
        for i in self.tree.get_children():
            self.tree.delete(i)

    def append_row(self, values):
        # pad/truncate to number of columns
        vals = list(values) + [""] * (len(self.columns) - len(values))
        vals = vals[:len(self.columns)]
        self.tree.insert("", "end", values=vals)

    def get_all(self):
        return [self.tree.item(i, "values") for i in self.tree.get_children()]

    # ---- CSV I/O ----
    def load_csv(self, path):
        self.clear()
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                self.append_row(row)

    def save_csv(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in self.get_all():
                writer.writerow(row)

    # ---- inline editing ----
    def _begin_edit(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)
        if not row_id or not col_id:
            return
        col = int(col_id.replace("#", "")) - 1
        bbox = self.tree.bbox(row_id, col_id)
        if not bbox:
            return
        x, y, w, h = bbox
        value = self.tree.set(row_id, self.columns[col])

        self._editor = tk.Entry(self.tree,
                               font=('Segoe UI', 10),
                               bg='#ffffff',
                               fg='#2d3748',
                               selectbackground='#4299e1',
                               selectforeground='#ffffff',
                               insertbackground='#2d3748',
                               borderwidth=1,
                               relief='solid')
        self._editor.insert(0, value)
        self._editor.select_range(0, "end")
        self._editor.focus()
        self._editor.place(x=x, y=y, width=w, height=h)

        def _finish(e=None):
            new_val = self._editor.get()
            self.tree.set(row_id, self.columns[col], new_val)
            self._editor.destroy()
            self._editor = None

        self._editor.bind("<Return>", _finish)
        self._editor.bind("<Escape>", lambda e: (self._editor.destroy(), setattr(self, "_editor", None)))
        self._editor.bind("<FocusOut>", _finish)


# ---------- Tabs ----------
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
        
        # Two labeled frames side-by-side: Parked Modal Frequencies & Simulation Parameters
        # -- Parked Modal Frequencies
        lfreq = ttk.LabelFrame(self, text="Parked Modal Frequencies")
        lfreq.grid(row=row, column=0, columnspan=2, sticky="nsew", padx=(0, 6), pady=(10, 6))
        ttk.Label(lfreq, text="File Path").grid(row=0, column=0, sticky="w")
        self.freq_path = PathEntry(lfreq, kind="file", title="Select frequency CSV", must_exist=True)
        self.freq_path.grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(lfreq, text="Import", command=self.import_freq).grid(row=1, column=1, padx=6)
        ttk.Label(lfreq, text="Freq [Hz] List").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.freq_table = EditableTreeview(lfreq, columns=["Freq [Hz]"])
        self.freq_table.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(2, 0))
        lfreq.columnconfigure(0, weight=1)
        lfreq.rowconfigure(3, weight=1)

        # -- Simulation Parameters
        lpars = ttk.LabelFrame(self, text="Simulation Parameters")
        lpars.grid(row=row, column=2, columnspan=3, sticky="nsew", pady=(10, 6))
        ttk.Label(lpars, text="File Path").grid(row=0, column=0, sticky="w")
        self.param_path = PathEntry(lpars, kind="file", title="Select parameters CSV", must_exist=True)
        self.param_path.grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(lpars, text="Import", command=self.import_params).grid(row=1, column=1, padx=6)
        ttk.Label(lpars, text="Simulation Parameters List").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.param_table = EditableTreeview(lpars, columns=["Parameter", "Value"])
        self.param_table.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(2, 0))
        lpars.columnconfigure(0, weight=1)
        lpars.rowconfigure(3, weight=1)

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
        airfoil_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airfoils")

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
                natfreqs = self.app.get_natural_frequencies()
                if natfreqs is None:
                    self.app.log("Error: No natural frequencies loaded\n")
                    return
                
                # Get components
                components = self.app.get_components()
                if not components:
                    self.app.log("Error: No components loaded\n")
                    return
                
                # Load airfoils
                affts = self.app.load_airfoils(viv_params.airfoil_folder)
                if not affts:
                    self.app.log("Error: No airfoil data loaded\n")
                    return
                
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
            self.freq_table.load_csv(path)
            self.app.log(f"Loaded frequencies: {path}\n")
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


class ComponentsTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._build()

    def _build(self):
        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        ttk.Label(top, text="Component").grid(row=0, column=0, sticky="w")
        self.comp_spin = ttk.Spinbox(top, from_=1, to=999, width=8)
        self.comp_spin.set(1)
        self.comp_spin.grid(row=0, column=1, padx=6)
        
        # Components directory selection
        ttk.Label(top, text="Components Directory").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.components_path = PathEntry(top, kind="dir", title="Select components directory")
        self.components_path.grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Button(top, text="Load Components", command=self.load_components).grid(row=0, column=4, padx=6)

        cols = [
            "Name","Dx","Dy","Dz","Rx","Ry","Rz","Pitch",
            "X","Y","Z","Chord","twist","thk%","offset","AirfoilPath"
        ]
        lf = ttk.LabelFrame(self, text="Component Geometric Definition")
        lf.grid(row=1, column=0, sticky="nsew", padx=10, pady=(6, 10))
        self.geom_table = EditableTreeview(lf, columns=cols)
        self.geom_table.grid(row=0, column=0, columnspan=3, sticky="nsew")
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)

        self.geom_path = PathEntry(lf, kind="savefile", title="Save component geometry as CSV")
        self.geom_path.grid(row=1, column=0, sticky="ew", pady=6)
        ttk.Button(lf, text="Load CSV", command=self.load_geom).grid(row=1, column=1, padx=4)
        ttk.Button(lf, text="Save CSV", command=self.save_geom).grid(row=1, column=2)

        # Set default components path
        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "componentsHVAWT")
        self.components_path.set(default_path)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

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


class PlotsOutputsTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._build()

    def _build(self):
        # Top row: Save plots path + Save button
        top_frame = ttk.Frame(self)
        top_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        ttk.Label(top_frame, text="Save Plots Path").grid(row=0, column=0, sticky="w")
        self.plot_save_path = PathEntry(top_frame, kind="dir", title="Choose directory for plots")
        self.plot_save_path.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(top_frame, text="Save Plot", command=self.save_plot).grid(row=0, column=2, padx=6)
        top_frame.columnconfigure(1, weight=1)

        # Center-left: Plot area
        plot_frame = ttk.LabelFrame(self, text="Analysis Results")
        plot_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=10, pady=6)
        
        if MATPLOTLIB_OK:
            self.fig = Figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title("VorLap Analysis Results")
            self.ax.text(0.5, 0.5, "Run analysis to see results", ha='center', va='center', transform=self.ax.transAxes)
            
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            self.canvas = None
            self.fig = None
            ttk.Label(plot_frame, text="Matplotlib not available.\n(Plot placeholder)").pack(expand=True, pady=30)

        # Plot type selection
        plot_controls = ttk.Frame(plot_frame)
        plot_controls.pack(fill="x", padx=5, pady=5)
        ttk.Label(plot_controls, text="Plot Type:").pack(side="left")
        
        self.plot_type = tk.StringVar(value="percdiff")
        plot_types = [
            ("Frequency Overlap", "percdiff"),
            ("Force X", "fx"),
            ("Force Y", "fy"), 
            ("Force Z", "fz"),
            ("Moment X", "mx"),
            ("Moment Y", "my"),
            ("Moment Z", "mz")
        ]
        
        for text, value in plot_types:
            ttk.Radiobutton(plot_controls, text=text, variable=self.plot_type, 
                          value=value, command=self.update_plots).pack(side="left", padx=5)

        # Bottom: Sample controls + export path + Export button
        bottom = ttk.LabelFrame(self, text="Sampling & Text Output")
        bottom.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=10, pady=(0, 6))
        ttk.Label(bottom, text="Sample X").grid(row=0, column=0, sticky="e")
        self.sample_x = ttk.Entry(bottom, width=10)
        self.sample_x.grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(bottom, text="Sample Y").grid(row=0, column=2, sticky="e")
        self.sample_y = ttk.Entry(bottom, width=10)
        self.sample_y.grid(row=0, column=3, padx=4, pady=4)
        ttk.Label(bottom, text="Sampled Signal Export Path").grid(row=0, column=4, sticky="e")
        self.sample_export = PathEntry(bottom, kind="savefile", title="Save sampled signal as CSV")
        self.sample_export.grid(row=0, column=5, sticky="ew", padx=6)
        ttk.Button(bottom, text="Export", command=self.export_sample).grid(row=0, column=6, padx=6)

        # Two text areas: Sampled Detailed Text Output + Console Output
        self.sample_output = ScrollText(bottom, height=8)
        self.sample_output.grid(row=1, column=0, columnspan=7, sticky="nsew", pady=(6, 0))

        self.console = ScrollText(self, height=10)
        self.console.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=10, pady=(0, 10))

        # weights
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=3)
        self.rowconfigure(2, weight=2)
        self.rowconfigure(3, weight=2)
        bottom.columnconfigure(5, weight=1)
        bottom.rowconfigure(1, weight=1)

    def update_plots(self):
        """Update the plot based on current analysis results and selected plot type."""
        if not hasattr(self.app, 'analysis_results') or not self.app.analysis_results:
            return
            
        if not self.fig:
            return
            
        results = self.app.analysis_results
        plot_type = self.plot_type.get()
        viv_params = results['viv_params']
        
        self.ax.clear()
        
        extent = [viv_params.azimuths[0], viv_params.azimuths[-1], 
                 viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]]
        
        if plot_type == "percdiff":
            data = results['percdiff_matrix']
            im = self.ax.imshow(data, extent=extent, aspect='auto', origin='lower', 
                               cmap='viridis_r', vmin=0, vmax=50)
            self.ax.set_title('Worst Percent Difference')
            label = 'Freq % Diff'
        elif plot_type.startswith("f"):
            force_data = results['total_global_force_vector']
            idx = {'fx': 0, 'fy': 1, 'fz': 2}[plot_type]
            data = force_data[:, :, idx]
            im = self.ax.imshow(data, extent=extent, aspect='auto', origin='lower', cmap='viridis_r')
            self.ax.set_title(f'Force {plot_type.upper()}')
            label = 'Force (N)'
        elif plot_type.startswith("m"):
            moment_data = results['total_global_moment_vector']
            idx = {'mx': 0, 'my': 1, 'mz': 2}[plot_type]
            data = moment_data[:, :, idx]
            im = self.ax.imshow(data, extent=extent, aspect='auto', origin='lower', cmap='viridis_r')
            self.ax.set_title(f'Moment {plot_type.upper()}')
            label = 'Moment (N-m)'
        
        self.ax.set_xlabel('Azimuth (deg)')
        self.ax.set_ylabel('Inflow (m/s)')
        
        # Add colorbar
        if hasattr(self, '_colorbar'):
            self._colorbar.remove()
        self._colorbar = self.fig.colorbar(im, ax=self.ax, label=label)
        
        self.canvas.draw()

    # ---- handlers ----
    def save_plot(self):
        if not self.fig:
            messagebox.showinfo("Plot", "No Matplotlib figure available.")
            return
        dir_ = self.plot_save_path.get() or filedialog.askdirectory(title="Choose plot save directory")
        if not dir_:
            return
        
        plot_type = self.plot_type.get()
        path = Path(dir_) / f"{plot_type}_plot.png"
        self.fig.savefig(path, dpi=150, bbox_inches='tight')
        self.app.log(f"Plot saved to: {path}\n")

    def export_sample(self):
        path = self.sample_export.get() or filedialog.asksaveasfilename(title="Save sampled signal CSV", defaultextension=".csv")
        if not path:
            return
        # TODO: replace with real sampled data
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            x = self.sample_x.get() or "0.0"
            y = self.sample_y.get() or "0.0"
            writer.writerow([x, y])
        self.sample_output.write(f"Exported sample (x={x}, y={y}) → {path}\n")
        self.app.log(f"Sample exported → {path}\n")

    def log(self, s: str):
        self.console.write(s)


class AnalysisTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._build()

    def _build(self):
        lf = ttk.LabelFrame(self, text="Analysis Tools")
        lf.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        btns = ttk.Frame(lf)
        btns.grid(row=0, column=0, sticky="w", padx=10, pady=10)
        ttk.Label(btns, text="Visualization Type:", font=('Segoe UI', 10, 'bold')).pack(side="left", padx=(0, 10))
        analysis_types = [
            ("Geometry", self.show_geometry),
            ("Thrust", self.show_thrust),
            ("Torque", self.show_torque),
            ("Frequency Overlap", self.show_frequency_overlap)
        ]
        
        for label, command in analysis_types:
            ttk.Button(btns, text=label, command=command).pack(side="left", padx=4)

        # Add separator
        sep = ttk.Separator(lf, orient='horizontal')
        sep.grid(row=1, column=0, sticky="ew", padx=10, pady=(5, 10))

        selector = ttk.Frame(lf)
        selector.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 10))
        ttk.Label(selector, text="Analysis Mode:", font=('Segoe UI', 10, 'bold')).pack(side="left", padx=(0, 10))
        for n in (1, 2, 3):
            ttk.Button(selector, text=f"Mode {n}", width=8, command=lambda N=n: self.select_mode(N)).pack(side="left", padx=2)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def show_geometry(self):
        """Show structure geometry visualization."""
        if not hasattr(self.app, 'components') or not self.app.components:
            messagebox.showwarning("No Data", "Load components first.")
            return
            
        try:
            viv_params = self.app.tab_setup.get_viv_params()
            vorlap.graphics.calc_structure_vectors_andplot(self.app.components, viv_params)
            self.app.log("[Analysis] Geometry visualization displayed\n")
        except Exception as e:
            self.app.log(f"[Analysis] Geometry error: {str(e)}\n")

    def show_thrust(self):
        """Show thrust analysis."""
        if hasattr(self.app, 'analysis_results') and self.app.analysis_results:
            self.app.tab_plots.plot_type.set("fx")
            self.app.tab_plots.update_plots()
            self.app.log("[Analysis] Thrust visualization updated\n")
        else:
            messagebox.showwarning("No Data", "Run analysis first.")

    def show_torque(self):
        """Show torque analysis."""
        if hasattr(self.app, 'analysis_results') and self.app.analysis_results:
            self.app.tab_plots.plot_type.set("mz")
            self.app.tab_plots.update_plots()
            self.app.log("[Analysis] Torque visualization updated\n")
        else:
            messagebox.showwarning("No Data", "Run analysis first.")

    def show_frequency_overlap(self):
        """Show frequency overlap analysis."""
        if hasattr(self.app, 'analysis_results') and self.app.analysis_results:
            self.app.tab_plots.plot_type.set("percdiff")
            self.app.tab_plots.update_plots()
            self.app.log("[Analysis] Frequency overlap visualization updated\n")
        else:
            messagebox.showwarning("No Data", "Run analysis first.")

    def select_mode(self, n):
        self.app.log(f"[Analysis] Mode set to {n}\n")


# ---------- Main app ----------
class VorLapApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VORtex overLAP Tool")
        self.geometry("1300x950")
        self.minsize(1100, 750)
        
        # Set window icon if available
        try:
            # You can add an icon file here if available
            # self.iconbitmap("icon.ico")  # Windows
            # self.iconphoto(False, tk.PhotoImage(file="icon.png"))  # Linux/Mac
            pass
        except:
            pass

        # Apply modern theme and styling
        self._setup_theme_and_styling()

        # Initialize data containers
        self.components = []
        self.natural_frequencies = None
        self.analysis_results = None

        # Create notebook with padding
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        self.tab_setup = SimulationSetupTab(nb, self)
        self.tab_components = ComponentsTab(nb, self)
        self.tab_plots = PlotsOutputsTab(nb, self)
        self.tab_analysis = AnalysisTab(nb, self)

        nb.add(self.tab_setup, text="Simulation Setup")
        nb.add(self.tab_components, text="Components")
        nb.add(self.tab_plots, text="Plots & Outputs")
        nb.add(self.tab_analysis, text="Analysis")

        # status bar with improved styling
        status_frame = ttk.Frame(self, style='StatusFrame.TFrame')
        status_frame.pack(fill="x", side="bottom", padx=8, pady=(0, 8))
        
        self.status = ttk.Label(status_frame, text="Ready", anchor="w", style='Status.TLabel')
        self.status.pack(fill="x", padx=8, pady=4)

    def _setup_theme_and_styling(self):
        """Set up modern theme and improved styling."""
        style = ttk.Style(self)
        
        # Try to use the best available theme
        available_themes = style.theme_names()
        preferred_themes = ["arc", "equilux", "adapta", "clam", "alt", "default"]
        
        selected_theme = "default"
        for theme in preferred_themes:
            if theme in available_themes:
                selected_theme = theme
                break
        
        style.theme_use(selected_theme)
        
        # Configure colors for a modern look
        colors = {
            'bg': '#f0f0f0',           # Light gray background
            'fg': '#2d3748',           # Dark gray text
            'select_bg': '#4299e1',    # Blue selection
            'select_fg': '#ffffff',    # White selected text
            'frame_bg': '#ffffff',     # White frame background
            'button_bg': '#e2e8f0',    # Light button background
            'button_active': '#cbd5e0', # Button active state
            'entry_bg': '#ffffff',     # White entry background
            'tree_bg': '#ffffff',      # White treeview background
            'tree_select': '#e2e8f0'   # Light gray tree selection
        }
        
        # Configure styles with larger fonts
        base_font = ('Segoe UI', 10)
        large_font = ('Segoe UI', 12)
        heading_font = ('Segoe UI', 11, 'bold')
        
        # Configure root window
        self.configure(bg=colors['bg'])
        
        # Configure notebook (tabs)
        style.configure('TNotebook', 
                       background=colors['bg'],
                       borderwidth=0)
        style.configure('TNotebook.Tab',
                       padding=[20, 8],
                       font=heading_font,
                       background=colors['button_bg'],
                       foreground=colors['fg'])
        style.map('TNotebook.Tab',
                 background=[('selected', colors['frame_bg']),
                           ('active', colors['button_active'])])
        
        # Configure frames
        style.configure('TFrame',
                       background=colors['bg'])
        
        # Configure labels with larger font
        style.configure('TLabel',
                       font=base_font,
                       background=colors['bg'],
                       foreground=colors['fg'])
        
        # Configure LabelFrame with larger font
        style.configure('TLabelframe',
                       font=heading_font,
                       background=colors['bg'],
                       foreground=colors['fg'],
                       borderwidth=1,
                       relief='solid')
        style.configure('TLabelframe.Label',
                       font=heading_font,
                       background=colors['bg'],
                       foreground=colors['fg'])
        
        # Configure buttons with improved styling
        style.configure('TButton',
                       font=base_font,
                       padding=[12, 6],
                       background=colors['button_bg'],
                       foreground=colors['fg'],
                       borderwidth=1,
                       relief='solid')
        style.map('TButton',
                 background=[('active', colors['button_active']),
                           ('pressed', colors['select_bg'])],
                 foreground=[('pressed', colors['select_fg'])])
        
        # Configure entries with larger font
        style.configure('TEntry',
                       font=base_font,
                       fieldbackground=colors['entry_bg'],
                       foreground=colors['fg'],
                       borderwidth=1,
                       relief='solid',
                       insertwidth=2)
        style.map('TEntry',
                 focuscolor=[('!focus', 'none')])
        
        # Configure spinbox
        style.configure('TSpinbox',
                       font=base_font,
                       fieldbackground=colors['entry_bg'],
                       foreground=colors['fg'],
                       borderwidth=1,
                       relief='solid')
        
        # Configure treeview with larger font
        style.configure('Treeview',
                       font=base_font,
                       background=colors['tree_bg'],
                       foreground=colors['fg'],
                       fieldbackground=colors['tree_bg'],
                       borderwidth=1,
                       relief='solid')
        style.configure('Treeview.Heading',
                       font=heading_font,
                       background=colors['button_bg'],
                       foreground=colors['fg'],
                       borderwidth=1,
                       relief='solid')
        style.map('Treeview',
                 background=[('selected', colors['tree_select'])],
                 foreground=[('selected', colors['fg'])])
        
        # Configure scrollbars
        style.configure('TScrollbar',
                       background=colors['button_bg'],
                       troughcolor=colors['bg'],
                       borderwidth=1,
                       relief='solid')
        
        # Configure radiobuttons with larger font
        style.configure('TRadiobutton',
                       font=base_font,
                       background=colors['bg'],
                       foreground=colors['fg'],
                       focuscolor='none')
        
        # Configure checkbuttons with larger font
        style.configure('TCheckbutton',
                       font=base_font,
                       background=colors['bg'],
                       foreground=colors['fg'],
                       focuscolor='none')
        style.configure('Accent.TCheckbutton',
                       font=base_font,
                       background=colors['bg'],
                       foreground=colors['select_bg'],
                       focuscolor='none')
        
        # Configure progressbar (if used)
        style.configure('TProgressbar',
                       background=colors['select_bg'],
                       troughcolor=colors['button_bg'],
                       borderwidth=0)
        
        # Configure separators
        style.configure('TSeparator',
                       background=colors['button_bg'])
        
        # Configure status bar
        style.configure('StatusFrame.TFrame',
                       background=colors['frame_bg'],
                       borderwidth=1,
                       relief='solid')
        style.configure('Status.TLabel',
                       font=base_font,
                       background=colors['frame_bg'],
                       foreground=colors['fg'])

    def log(self, s: str):
        """Log message to console and status bar."""
        try:
            self.tab_plots.log(s)
        except Exception:
            print(s, end="")
        self.status.config(text=s.strip().splitlines()[-1] if s.strip() else "Ready")

    def get_natural_frequencies(self):
        """Get natural frequencies from the setup tab."""
        freq_data = self.tab_setup.freq_table.get_all()
        if not freq_data:
            # Try to load from default file
            default_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "natural_frequencies.csv")
            if os.path.exists(default_file):
                try:
                    self.natural_frequencies = np.loadtxt(default_file, delimiter=',')
                    self.log(f"Loaded default natural frequencies from {default_file}\n")
                    return self.natural_frequencies
                except Exception as e:
                    self.log(f"Error loading default frequencies: {str(e)}\n")
            return None
            
        try:
            frequencies = []
            for row in freq_data:
                if len(row) > 0 and row[0].strip():
                    frequencies.append(float(row[0]))
            self.natural_frequencies = np.array(frequencies)
            return self.natural_frequencies
        except Exception as e:
            self.log(f"Error parsing frequencies: {str(e)}\n")
            return None

    def get_components(self):
        """Get loaded components."""
        return self.components

    def load_airfoils(self, airfoil_folder):
        """Load airfoil FFT data from the specified folder."""
        try:
            affts = {}
            for file in glob.glob(os.path.join(airfoil_folder, "*.h5")):
                afft = vorlap.load_airfoil_fft(file)
                affts[afft.name] = afft
                
            # Ensure default airfoil exists
            if "default" not in affts and affts:
                affts["default"] = next(iter(affts.values()))
                
            self.log(f"Loaded {len(affts)} airfoil files from {airfoil_folder}\n")
            return affts
        except Exception as e:
            self.log(f"Error loading airfoils: {str(e)}\n")
            return {}


if __name__ == "__main__":
    app = VorLapApp()
    app.mainloop() 