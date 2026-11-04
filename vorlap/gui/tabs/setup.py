#!/usr/bin/env python3
"""
Simulation Setup Tab for the VorLap GUI.

This tab handles simulation configuration, parameters, and running analysis.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import numpy as np
import threading
import copy
from collections import Counter
from queue import Empty, Queue

import vorlap

from ..widgets import PathEntry, EditableTreeview, ScrollableFrame
from .qblade_bridge import default_qblade_sim_path


def _format_parameter_value(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.12g}"


def _sweep_step(values, default_step):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size >= 2:
        return float(arr[1] - arr[0])
    return float(default_step)


def _sweep_start(values, default_start):
    arr = np.asarray(values, dtype=float).reshape(-1)
    return float(arr[0]) if arr.size else float(default_start)


def _sweep_end(values, default_start, default_step):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size:
        return float(arr[-1] + _sweep_step(arr, default_step))
    return float(default_start + default_step)


GUI_PARAMETER_SCHEMA = (
    ("fluid_density", "Fluid density (kg/m³)"),
    ("fluid_dynamicviscosity", "Fluid dynamic viscosity (Pa·s)"),
    ("rotation_axis_x", "X component of rotation axis vector"),
    ("rotation_axis_y", "Y component of rotation axis vector"),
    ("rotation_axis_z", "Z component of rotation axis vector"),
    ("rotation_axis_offset_x", "X offset of rotation axis (m)"),
    ("rotation_axis_offset_y", "Y offset of rotation axis (m)"),
    ("rotation_axis_offset_z", "Z offset of rotation axis (m)"),
    ("inflow_vec_x", "X component of inflow direction vector"),
    ("inflow_vec_y", "Y component of inflow direction vector"),
    ("inflow_vec_z", "Z component of inflow direction vector"),
    ("azimuth_start", "Starting azimuth angle (degrees)"),
    ("azimuth_end", "Ending azimuth angle (degrees)"),
    ("azimuth_step", "Step size for azimuth angle (degrees)"),
    ("inflow_speed_start", "Starting inflow speed (m/s)"),
    ("inflow_speed_end", "Ending inflow speed (m/s)"),
    ("inflow_speed_step", "Step size for inflow speed (m/s)"),
    ("n_harmonic", "Number of harmonics to consider"),
    ("output_time_start", "Output time series start (s)"),
    ("output_time_end", "Output time series end (s)"),
    ("output_time_step", "Output time series step size (s)"),
    ("output_azimuth", "Output azimuth angle (degrees)"),
    ("output_vinf", "Output inflow velocity (m/s)"),
    ("amplitude_coeff_cutoff", "Amplitude coefficient cutoff threshold"),
    ("n_freq_depth", "Number of frequency depth levels"),
)


class SimulationSetupTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._worker_active = False
        self._analysis_poll_id = None
        self.sim_save_var = tk.StringVar()
        self.qblade_loading_path_var = tk.StringVar()
        self.include_qblade_tower = tk.BooleanVar(value=True)
        self.scrollable = ScrollableFrame(self)
        self.scrollable.pack(fill="both", expand=True)
        self.content = self.scrollable.content
        self._build()

    def _build(self):
        container = self.content
        row = 0
        
        # Parked Modal Frequencies (simplified for single-row CSV)
        lfreq = ttk.LabelFrame(container, text="Parked Modal Frequencies")
        lfreq.grid(row=row, column=0, columnspan=6, sticky="nsew", pady=(10, 6))
        
        ttk.Label(lfreq, text="File Path").grid(row=0, column=0, sticky="w")
        self.freq_path = PathEntry(
            lfreq,
            kind="file",
            title="Select frequency CSV",
            must_exist=True,
            on_select=lambda _path: self.import_freq(),
        )
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
        lpars = ttk.LabelFrame(container, text="Simulation Parameters")
        lpars.grid(row=row, column=0, columnspan=6, sticky="ew", pady=(10, 6))
        ttk.Label(lpars, text="File Path").grid(row=0, column=0, sticky="w")
        self.param_path = PathEntry(
            lpars,
            kind="file",
            title="Select parameters CSV",
            must_exist=True,
            on_select=lambda _path: self.import_params(),
        )
        self.param_path.grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(lpars, text="Import", command=self.import_params).grid(row=1, column=1, padx=6)
        ttk.Label(lpars, text="Simulation Parameters List").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.param_table = EditableTreeview(lpars, columns=["Description", "Parameter", "Value"], non_editable_columns=["Description"])
        self.param_table.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        lpars.columnconfigure(0, weight=1)
        lpars.rowconfigure(3, weight=0)  # Fixed weight for parameter table (non-collapsible)
        row += 1

        # Geometry source toggle + side-by-side inputs
        lgeom = ttk.LabelFrame(container, text="Geometry Input Source")
        lgeom.grid(row=row, column=0, columnspan=6, sticky="ew", pady=(10, 6))
        self.use_qblade_geometry = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            lgeom,
            text="QBlade Geometry",
            variable=self.use_qblade_geometry,
            command=self._on_qblade_toggle,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        # QBlade simulation input (left)
        self.qblade_frame = ttk.LabelFrame(lgeom, text="QBlade Simulation Input")
        self.qblade_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        ttk.Label(self.qblade_frame, text="QBlade Simulation (.sim):").grid(row=0, column=0, sticky="w")
        self.qblade_sim_path = PathEntry(
            self.qblade_frame,
            kind="file",
            title="Select QBlade simulation (.sim)",
            must_exist=True,
            on_select=lambda _path: self.load_qblade_inputs(),
            button_text="Load",
        )
        self.qblade_sim_path.var.trace_add("write", lambda *_args: self._publish_qblade_sim_path())
        self.qblade_sim_path.grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(self.qblade_frame, text="Save VorLap CSVs", command=self.save_qblade_components).grid(row=1, column=1, padx=6)
        ttk.Checkbutton(
            self.qblade_frame,
            text="Include Tower",
            variable=self.include_qblade_tower,
        ).grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.qblade_frame.columnconfigure(0, weight=1)

        # Components section (right)
        self.components_frame = ttk.LabelFrame(lgeom, text="Component Geometric Definition")
        lcomp = self.components_frame
        lcomp.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        ttk.Label(lcomp, text="Components Directory").grid(row=0, column=0, sticky="w")
        self.components_path = PathEntry(
            lcomp,
            kind="dir",
            title="Select components directory",
            must_exist=True,
            on_select=lambda _path: self.load_components(),
        )
        self.components_path.grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(lcomp, text="Import", command=self.load_components).grid(row=1, column=1, padx=6)
        ttk.Label(lcomp, text="Component Geometry List").grid(row=2, column=0, sticky="w", pady=(6, 0))
        
        # Component geometry table (read-only)
        cols = [
            "Component ID", "Translation X", "Translation Y", "Translation Z", 
            "Rotation X", "Rotation Y", "Rotation Z", "Pitch", 
            "Segments", "Avg Chord", "Avg Twist", "Avg Thickness", "Common Airfoil"
        ]
        self.geom_table = EditableTreeview(lcomp, columns=cols, height=8, non_editable_columns=cols)
        self.geom_table.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        
        # Add note about CSV editing
        note_frame = ttk.Frame(lcomp)
        note_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        note_label = ttk.Label(note_frame, 
                              text="Note: Component geometry must be edited directly in the CSV files due to software limitations. " +
                                   "Use the 'Import' or 'Load' button to refresh the table after making changes.",
                              font=('Segoe UI', 9),
                              foreground='#666666',
                              wraplength=800)
        note_label.pack(anchor="w")
        
        lcomp.columnconfigure(0, weight=1)
        lcomp.rowconfigure(3, weight=0)  # Fixed weight for geometry table (non-collapsible)
        lgeom.columnconfigure(0, weight=1)
        lgeom.columnconfigure(1, weight=1)
        row += 1

        # Time-varying inflow file (required for force time-history output)
        linflow = ttk.LabelFrame(container, text="Time-Varying Inflow Profile")
        linflow.grid(row=row, column=0, columnspan=6, sticky="ew", pady=(10, 6))
        ttk.Label(linflow, text="Profile CSV (required): time, inflow_speed, inflow_direction_deg").grid(
            row=0, column=0, sticky="w"
        )
        self.inflow_profile_path = PathEntry(
            linflow,
            kind="file",
            title="Select inflow profile CSV",
            must_exist=True,
        )
        self.inflow_profile_path.grid(row=1, column=0, sticky="ew", pady=2)
        linflow.columnconfigure(0, weight=1)
        row += 1
        
        # Set default components path
        default_path = os.path.join(vorlap.repo_dir, "data", "components", "componentsHVAWT")
        self.components_path.set(default_path)

        
        # Set default save path (same directory as the script)
        default_save_path = vorlap.repo_dir
        self.sim_save_var.set(default_save_path)
        
        # Default frequency file path
        default_freq_path = os.path.join(vorlap.repo_dir, "data", "natural_frequencies.csv")
        self.freq_path.set(default_freq_path)

        # Default inflow profile path
        default_inflow_profile_path = os.path.join(vorlap.repo_dir, "data", "inflow_profile.csv")
        self.inflow_profile_path.set(default_inflow_profile_path)

        # Both tabs resolve the same packaged example. Installed builds without
        # QBlade assets start with the packaged component CSV workflow.
        default_sim = default_qblade_sim_path()
        if default_sim is not None:
            self.qblade_sim_path.set(str(default_sim))
        else:
            self.use_qblade_geometry.set(False)
            if os.path.isdir(default_path):
                self.load_components()

        # Initialize with default parameters
        self._populate_default_params()
        self.import_freq(show_dialog_on_error=False)
        if not [w for w in self.freq_inner_frame.winfo_children() if isinstance(w, ttk.Entry)]:
            self._populate_frequency_entries([0.07])
            self.app.log("Loaded fallback default frequency: 0.07 Hz\n")
        self._apply_geometry_source_state()

        # Enforce requested visual order:
        # 1) Geometry Input Source
        # 2) Time-Varying Inflow Profile
        # 3) Simulation Parameters
        # 4) Parked Modal Frequencies
        lgeom.grid_configure(row=0)
        linflow.grid_configure(row=1)
        lpars.grid_configure(row=2)
        lfreq.grid_configure(row=3)

        # Make grid flexible
        for c in range(6):
            container.columnconfigure(c, weight=(1 if c in (1, 3) else 0))
        # Set section heights.
        container.rowconfigure(0, weight=0, minsize=260)  # Geometry source section
        container.rowconfigure(1, weight=0, minsize=90)   # Inflow profile section
        container.rowconfigure(2, weight=0, minsize=200)  # Parameter section
        container.rowconfigure(3, weight=0, minsize=200)  # Frequency section

    def on_tab_selected(self):
        try:
            self.update_idletasks()
            self.scrollable.refresh()
        except Exception:
            pass

    def _get_param_descriptions(self):
        """Get parameter descriptions mapping."""
        return dict(GUI_PARAMETER_SCHEMA)

    @staticmethod
    def _default_viv_params():
        return vorlap.VIV_Params(
            azimuths=np.arange(0.0, 360.0, 10.0, dtype=float),
            inflow_speeds=np.arange(2.0, 50.0, 4.0, dtype=float),
            output_time=np.arange(0.0, 0.011, 0.001, dtype=float),
            output_azimuth_vinf=(10.0, 6.0),
            n_harmonic=2,
            amplitude_coeff_cutoff=0.002,
            n_freq_depth=10,
        )

    @classmethod
    def _default_param_values(cls, viv_params=None):
        viv = cls._default_viv_params() if viv_params is None else viv_params
        return {
            "fluid_density": viv.fluid_density,
            "fluid_dynamicviscosity": viv.fluid_dynamicviscosity,
            "rotation_axis_x": viv.rotation_axis[0],
            "rotation_axis_y": viv.rotation_axis[1],
            "rotation_axis_z": viv.rotation_axis[2],
            "rotation_axis_offset_x": viv.rotation_axis_offset[0],
            "rotation_axis_offset_y": viv.rotation_axis_offset[1],
            "rotation_axis_offset_z": viv.rotation_axis_offset[2],
            "inflow_vec_x": viv.inflow_vec[0],
            "inflow_vec_y": viv.inflow_vec[1],
            "inflow_vec_z": viv.inflow_vec[2],
            "azimuth_start": _sweep_start(viv.azimuths, 0.0),
            "azimuth_end": _sweep_end(viv.azimuths, 0.0, 10.0),
            "azimuth_step": _sweep_step(viv.azimuths, 10.0),
            "inflow_speed_start": _sweep_start(viv.inflow_speeds, 2.0),
            "inflow_speed_end": _sweep_end(viv.inflow_speeds, 2.0, 4.0),
            "inflow_speed_step": _sweep_step(viv.inflow_speeds, 4.0),
            "n_harmonic": viv.n_harmonic,
            "output_time_start": _sweep_start(viv.output_time, 0.0),
            "output_time_end": _sweep_end(viv.output_time, 0.0, 0.001),
            "output_time_step": _sweep_step(viv.output_time, 0.001),
            "output_azimuth": viv.output_azimuth_vinf[0],
            "output_vinf": viv.output_azimuth_vinf[1],
            "amplitude_coeff_cutoff": viv.amplitude_coeff_cutoff,
            "n_freq_depth": viv.n_freq_depth,
        }

    def _populate_default_params(self):
        """Populate the parameters table with default VIV_Params values."""
        descriptions = self._get_param_descriptions()
        default_params = self._default_param_values()
        
        for param, _description in GUI_PARAMETER_SCHEMA:
            value = default_params[param]
            description = descriptions.get(param, "")
            self.param_table.append_row([description, param, _format_parameter_value(value)])

    def _set_widget_state_recursive(self, widget, enabled):
        for child in widget.winfo_children():
            self._set_widget_state_recursive(child, enabled)
        try:
            if hasattr(widget, "state"):
                if enabled:
                    widget.state(["!disabled"])
                else:
                    widget.state(["disabled"])
            else:
                widget.configure(state=("normal" if enabled else "disabled"))
        except Exception:
            pass

    def _apply_geometry_source_state(self):
        use_qblade = bool(self.use_qblade_geometry.get())
        self._set_widget_state_recursive(self.qblade_frame, True)
        self._set_widget_state_recursive(self.components_frame, not use_qblade)

    def _on_qblade_toggle(self):
        self._apply_geometry_source_state()

    def _publish_qblade_sim_path(self):
        state = getattr(self.app, "state", None)
        if hasattr(state, "set_qblade_sim_path"):
            state.set_qblade_sim_path(self.qblade_sim_path.get())

    @staticmethod
    def _validated_sweep_range(
        params,
        *,
        label,
        start_key,
        end_key,
        step_key,
        default_start,
        default_end,
        default_step,
    ):
        start = float(params.get(start_key, default_start))
        end = float(params.get(end_key, default_end))
        step = float(params.get(step_key, default_step))
        values = np.array([start, end, step], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} start, end, and step must be finite values.")
        if step <= 0.0:
            raise ValueError(f"{label} step must be positive.")
        sweep = np.arange(start, end, step)
        if sweep.size == 0:
            raise ValueError(
                f"{label} sweep is empty; set the end value greater than the start value."
            )
        return sweep

    def get_viv_params(self):
        """Create VIV_Params object from the current parameter table."""
        defaults = self._default_param_values()
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

        azimuths = self._validated_sweep_range(
            params,
            label="Azimuth",
            start_key="azimuth_start",
            end_key="azimuth_end",
            step_key="azimuth_step",
            default_start=defaults["azimuth_start"],
            default_end=defaults["azimuth_end"],
            default_step=defaults["azimuth_step"],
        )

        inflow_speeds = self._validated_sweep_range(
            params,
            label="Inflow speed",
            start_key="inflow_speed_start",
            end_key="inflow_speed_end",
            step_key="inflow_speed_step",
            default_start=defaults["inflow_speed_start"],
            default_end=defaults["inflow_speed_end"],
            default_step=defaults["inflow_speed_step"],
        )

        output_time = self._validated_sweep_range(
            params,
            label="Output time",
            start_key="output_time_start",
            end_key="output_time_end",
            step_key="output_time_step",
            default_start=defaults["output_time_start"],
            default_end=defaults["output_time_end"],
            default_step=defaults["output_time_step"],
        )

        # Set airfoil folder to default location
        airfoil_folder = os.path.join(vorlap.repo_dir, "data", "airfoils")

        return vorlap.VIV_Params(
            fluid_density=params.get("fluid_density", defaults["fluid_density"]),
            fluid_dynamicviscosity=params.get("fluid_dynamicviscosity", defaults["fluid_dynamicviscosity"]),
            rotation_axis=np.array([
                params.get("rotation_axis_x", defaults["rotation_axis_x"]),
                params.get("rotation_axis_y", defaults["rotation_axis_y"]),
                params.get("rotation_axis_z", defaults["rotation_axis_z"]),
            ]),
            rotation_axis_offset=np.array([
                params.get("rotation_axis_offset_x", defaults["rotation_axis_offset_x"]),
                params.get("rotation_axis_offset_y", defaults["rotation_axis_offset_y"]),
                params.get("rotation_axis_offset_z", defaults["rotation_axis_offset_z"]),
            ]),
            inflow_vec=np.array([
                params.get("inflow_vec_x", defaults["inflow_vec_x"]),
                params.get("inflow_vec_y", defaults["inflow_vec_y"]),
                params.get("inflow_vec_z", defaults["inflow_vec_z"]),
            ]),
            azimuths=azimuths,
            inflow_speeds=inflow_speeds,
            n_harmonic=params.get("n_harmonic", defaults["n_harmonic"]),
            output_time=output_time,
            output_azimuth_vinf=(
                params.get("output_azimuth", defaults["output_azimuth"]),
                params.get("output_vinf", defaults["output_vinf"]),
            ),
            amplitude_coeff_cutoff=params.get("amplitude_coeff_cutoff", defaults["amplitude_coeff_cutoff"]),
            n_freq_depth=params.get("n_freq_depth", defaults["n_freq_depth"]),
            airfoil_folder=airfoil_folder
        )

    # ---- handlers ----
    def run_and_save(self):
        """Snapshot inputs on the Tk thread, then run the numerical work once."""
        if self._worker_active:
            self.app.log("An offline analysis is already running.\n")
            return
        try:
            natfreqs = self.get_natural_frequencies()
            if natfreqs is None:
                raise ValueError("No natural frequencies loaded.")
            config = vorlap.VorLapOfflineAnalysisConfig(
                viv_params=self.get_viv_params(),
                natural_frequencies=natfreqs.copy(),
                inflow_profile_path=self.inflow_profile_path.get().strip(),
                # Frame preparation mutates components; keep displayed geometry
                # and later user edits independent of the running calculation.
                components=tuple(copy.deepcopy(self.app.components)),
                use_qblade_geometry=bool(self.use_qblade_geometry.get()),
                qblade_sim_path=self.qblade_sim_path.get().strip(),
                include_qblade_tower=bool(self.include_qblade_tower.get()),
                save_dir=self.sim_save_var.get().strip(),
                qblade_loading_path=self.qblade_loading_path_var.get().strip(),
            )
        except Exception as exc:
            self.app.log(f"Invalid analysis inputs: {exc}\n")
            messagebox.showerror("Analysis Error", str(exc))
            return

        self._analysis_events = Queue()
        self._worker_active = True
        self.app.analysis_results = None
        self.app.tab_plots.run_button.configure(state="disabled")

        def run_analysis():
            try:
                analysis_result = vorlap.run_offline_analysis(
                    config,
                    progress_callback=lambda message: self._analysis_events.put(("log", message)),
                )
                self._analysis_events.put(("result", analysis_result))
            except Exception as exc:
                self._analysis_events.put(("error", exc))

        self._worker = threading.Thread(target=run_analysis, daemon=True)
        self._worker.start()
        self._analysis_poll_id = self.after(30, self._poll_analysis_events)

    def _poll_analysis_events(self):
        """Apply worker messages only on the Tk thread, including state listeners."""
        self._analysis_poll_id = None
        try:
            while True:
                kind, value = self._analysis_events.get_nowait()
                if kind == "log":
                    self.app.log(value)
                    continue
                self._worker_active = False
                self.app.tab_plots.run_button.configure(state="normal")
                if kind == "error":
                    self.app.log(f"Error during analysis: {value}\n")
                    messagebox.showerror("Analysis Error", str(value))
                else:
                    self.app.components = list(value.components)
                    self.app.analysis_results = value
                    self._populate_geometry_table(self.app.components)
                    self.app.tab_plots.update_plots()
                    self.app.log("Analysis completed successfully!\n\n")
                return
        except Empty:
            if self._worker_active:
                self._analysis_poll_id = self.after(30, self._poll_analysis_events)

    def destroy(self):
        if self._analysis_poll_id is not None:
            self.after_cancel(self._analysis_poll_id)
            self._analysis_poll_id = None
        super().destroy()

    def _populate_frequency_entries(self, frequencies):
        for widget in self.freq_inner_frame.winfo_children():
            widget.destroy()

        for i, freq_val in enumerate(frequencies):
            entry = ttk.Entry(self.freq_inner_frame, width=8, justify="center")
            entry.insert(0, _format_parameter_value(freq_val))
            entry.grid(row=0, column=i, padx=2)

            def validate_freq(val):
                try:
                    if val == "":
                        return True
                    float(val)
                    return True
                except ValueError:
                    return False

            entry.configure(validate="key", validatecommand=(entry.register(validate_freq), "%P"))

        self.freq_inner_frame.update_idletasks()
        self.freq_canvas.configure(scrollregion=self.freq_canvas.bbox("all"))

    def import_freq(self, show_dialog_on_error=True):
        path = self.freq_path.get()
        if not path:
            if show_dialog_on_error:
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
                
                freq_values = []
                for val in row:
                    try:
                        freq_values.append(float(val.strip()))
                    except ValueError:
                        # Skip non-numeric values
                        continue
                self._populate_frequency_entries(freq_values)
                
                count = len([w for w in self.freq_inner_frame.winfo_children() if isinstance(w, ttk.Entry)])
                self.app.log(f"Loaded {count} frequencies: {path}\n")
                
        except Exception as e:
            if show_dialog_on_error:
                messagebox.showerror("Import failed", str(e))
            else:
                self.app.log(f"Frequency default load failed: {str(e)}\n")

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

    def _populate_geometry_table(self, components):
        """Populate the component geometry table from loaded components."""
        self.geom_table.clear()
        for comp in components:
            avg_chord = np.mean(comp.chord) if len(comp.chord) > 0 else 0
            avg_twist = np.mean(comp.twist) if len(comp.twist) > 0 else 0
            avg_thickness = np.mean(comp.thickness) if len(comp.thickness) > 0 else 0
            pitch_val = comp.pitch[0] if len(comp.pitch) > 0 else 0
            num_segments = len(comp.shape_xyz)
            most_common_airfoil = ""
            if len(comp.airfoil_ids) > 0:
                most_common_airfoil = Counter(comp.airfoil_ids).most_common(1)[0][0]

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
                f"{avg_thickness:.3f}",
                str(most_common_airfoil),
            ]
            self.geom_table.append_row(row)

    def _queue_geometry_plot_refresh(self):
        """Pre-populate the geometry plot after loading geometry inputs."""
        if not self.app.components:
            return

        def _refresh():
            try:
                tab_plots = getattr(self.app, "tab_plots", None)
                if tab_plots is None:
                    return
                tab_plots.plot_type.set("geometry")
                tab_plots._update_geometry_controls()
                tab_plots.update_plots(show_errors=False)
            except Exception as exc:
                self.app.log(f"Geometry plot pre-population failed: {exc}\n")

        self.app.after_idle(_refresh)

    def load_qblade_inputs(self):
        """Load VorLap-equivalent components from a QBlade .sim file."""
        qblade_sim = self.qblade_sim_path.get().strip()
        if not qblade_sim:
            messagebox.showwarning("No file", "Choose a QBlade .sim file.")
            return

        try:
            components, qblade_viv_params, qblade_node_ids = vorlap.convert_qblade_to_vorlap_inputs(
                qblade_sim,
                include_tower=bool(self.include_qblade_tower.get()),
            )
            self.app.components = components
            self._populate_geometry_table(components)
            self._queue_geometry_plot_refresh()
            self.app.log(
                f"Imported QBlade model: {qblade_sim}\n"
                f"  Components: {len(components)}\n"
                f"  QBlade load targets: {len(qblade_node_ids)}\n"
                f"  Include tower: {bool(self.include_qblade_tower.get())}\n"
                f"  Fluid density: {qblade_viv_params.fluid_density:.6g}\n"
                f"  Dynamic viscosity: {qblade_viv_params.fluid_dynamicviscosity:.6g}\n"
            )

            if not self.qblade_loading_path_var.get().strip():
                default_loading = os.path.join(
                    self.sim_save_var.get() or os.path.dirname(qblade_sim),
                    "qblade_external_loading.txt",
                )
                self.qblade_loading_path_var.set(default_loading)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            self.app.log(f"Error loading QBlade model: {str(e)}\n")

    def save_qblade_components(self):
        """Save the currently loaded components to VorLap component CSV files."""
        if not self.app.components:
            qblade_sim = self.qblade_sim_path.get().strip()
            if qblade_sim:
                self.load_qblade_inputs()

        if not self.app.components:
            messagebox.showwarning("No geometry", "Load a QBlade model first.")
            return

        out_dir = filedialog.askdirectory(title="Select output directory for VorLap components")
        if not out_dir:
            return

        try:
            written_files = vorlap.write_components_to_csv(out_dir, self.app.components)
            self.app.log(f"Saved {len(written_files)} VorLap component files to: {out_dir}\n")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            self.app.log(f"Error saving components: {str(e)}\n")

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
                self._populate_geometry_table(components)
                self._queue_geometry_plot_refresh()
                    
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            self.app.log(f"Error loading components: {str(e)}\n")

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
