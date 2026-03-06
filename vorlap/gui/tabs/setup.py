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
import warnings
from collections import Counter

import vorlap

from ..widgets import PathEntry, EditableTreeview, ScrollableFrame


class SimulationSetupTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
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

        # Optional default QBlade test path when present
        default_qblade_sim_path = os.path.join(
            os.path.dirname(vorlap.repo_dir),
            "examples",
            "QBladeExample.sim",
        )
        if os.path.isfile(default_qblade_sim_path):
            self.qblade_sim_path.set(default_qblade_sim_path)

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
            ("azimuth_end", "360"),
            ("azimuth_step", "10"),
            ("inflow_speed_start", "2.0"),
            ("inflow_speed_end", "50.0"),
            ("inflow_speed_step", "4.0"),
            ("n_harmonic", "2"),
            ("output_time_start", "0.0"),
            ("output_time_end", "0.011"),
            ("output_time_step", "0.001"),
            ("output_azimuth", "10.0"),
            ("output_vinf", "6.0"),
            ("amplitude_coeff_cutoff", "0.002"),
            ("n_freq_depth", "10"),
        ]
        
        for param, value in default_params:
            description = descriptions.get(param, "")
            self.param_table.append_row([description, param, value])

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
            params.get("azimuth_end", 255),
            params.get("azimuth_step", 5)
        )
        
        inflow_speeds = np.arange(
            params.get("inflow_speed_start", 2.0),
            params.get("inflow_speed_end", 50.0),
            params.get("inflow_speed_step", 0.5)
        )
        
        output_time = np.arange(
            params.get("output_time_start", 0.0),
            params.get("output_time_end", 0.01),
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
        def ui_log(message: str):
            self.app.after(0, self.app.log, message)

        def ui_error(title: str, message: str):
            self.app.after(0, lambda: messagebox.showerror(title, message))

        def ui_populate_geometry_table(components):
            self.app.after(0, lambda: self._populate_geometry_table(components))

        def ui_update_plots():
            self.app.after(0, self.app.tab_plots.update_plots)

        def run_analysis():
            try:
                ui_log("Starting VorLap analysis...\n")
                ui_log(f"  Save directory: {self.sim_save_var.get()}\n")
                
                # Get parameters
                viv_params = self.get_viv_params()
                qblade_node_ids = None

                qblade_sim_path = self.qblade_sim_path.get().strip()
                if self.use_qblade_geometry.get():
                    if not qblade_sim_path:
                        raise ValueError(
                            "QBlade Geometry is enabled but no QBlade simulation file is set. "
                            "Choose a `.sim` file or uncheck QBlade Geometry."
                        )
                    ui_log(f"Converting QBlade model from: {qblade_sim_path}\n")
                    components_from_qblade, qblade_viv_params, qblade_node_ids = vorlap.convert_qblade_to_vorlap_inputs(
                        qblade_sim_path,
                        include_tower=bool(self.include_qblade_tower.get()),
                    )
                    self.app.components = components_from_qblade
                    # Keep user sweep settings, but align key fluid/flow frame fields with QBlade.
                    viv_params.fluid_density = qblade_viv_params.fluid_density
                    viv_params.fluid_dynamicviscosity = qblade_viv_params.fluid_dynamicviscosity
                    viv_params.rotation_axis = qblade_viv_params.rotation_axis
                    viv_params.rotation_axis_offset = qblade_viv_params.rotation_axis_offset
                    viv_params.inflow_vec = qblade_viv_params.inflow_vec
                    ui_populate_geometry_table(self.app.components)
                    ui_log(
                        f"Imported {len(self.app.components)} components and {len(qblade_node_ids)} QBlade load targets\n"
                    )
                else:
                    qblade_sim_path = ""

                inflow_profile_path = self.inflow_profile_path.get().strip()
                if not inflow_profile_path:
                    raise ValueError(
                        "Inflow profile CSV is required for force output. "
                        "Provide a file with: time, inflow_speed, inflow_direction_deg."
                    )
                inflow_profile = vorlap.load_inflow_time_series(inflow_profile_path)
                ui_log(
                    f"Loaded inflow profile with {inflow_profile.time.size} samples: {inflow_profile_path}\n"
                )
                
                # Get natural frequencies
                natfreqs = self.get_natural_frequencies()
                if natfreqs is None:
                    ui_log("Error: No natural frequencies loaded\n")
                    return
                
                # Get components
                components = self.app.components
                if not components:
                    ui_log("Error: No components loaded\n")
                    return
                
                # Load airfoils
                affts = self.app.load_airfoils(viv_params.airfoil_folder)
                if not affts:
                    ui_log("Error: No airfoil data loaded\n")
                    return
                
                # assemble each component into a full structure, and we need the rotation axis
                # plot the full structure surface with a generic airfoil shape
                vorlap.graphics.calc_structure_vectors_andplot(
                    components,
                    viv_params,
                    show_plot=False,
                    return_fig=False,
                )
            
                # Run computation
                ui_log("Running thrust/torque spectrum computation...\n")
                start_time = time.time()
                with warnings.catch_warnings(record=True) as captured_warnings:
                    warnings.simplefilter("always")

                    # percdiff_matrix, percdiff_info, total_global_force_vector, total_global_moment_vector, global_force_vector_nodes = vorlap.compute_thrust_torque_spectrum(
                    #     components, affts, viv_params, natfreqs
                    # )
                    percdiff_matrix, percdiff_info, total_global_force_vector, total_global_moment_vector, global_force_vector_nodes = vorlap.compute_thrust_torque_spectrum_optimized(
                        components, affts, viv_params, natfreqs
                    ) 

                    ui_log("Running time-varying force history reconstruction...\n")
                    (
                        inflow_time,
                        tv_total_global_force_vector,
                        tv_total_global_moment_vector,
                        tv_global_force_vector_nodes,
                    ) = vorlap.compute_time_varying_force_history_optimized(
                        components=components,
                        affts=affts,
                        viv_params=viv_params,
                        inflow_profile=inflow_profile,
                        azimuth_deg=viv_params.output_azimuth_vinf[0],
                        smoothing_cycles=0.25,
                    )

                seen_warning_messages = set()
                for warning_record in captured_warnings:
                    warning_message = str(warning_record.message).strip()
                    if not warning_message or warning_message in seen_warning_messages:
                        continue
                    seen_warning_messages.add(warning_message)
                    ui_log(f"Warning: {warning_message}\n")
                end_time = time.time()
                execution_time = end_time - start_time
                ui_log(f"Computation completed in {execution_time:.4f} seconds\n")
                
                # Store results in app
                self.app.analysis_results = {
                    'percdiff_matrix': percdiff_matrix,
                    'percdiff_info': percdiff_info,
                    'total_global_force_vector': total_global_force_vector,
                    'total_global_moment_vector': total_global_moment_vector,
                    'global_force_vector_nodes_selected_case': global_force_vector_nodes,
                    'global_force_vector_nodes': tv_global_force_vector_nodes,
                    'inflow_time': inflow_time,
                    'time_varying_total_global_force_vector': tv_total_global_force_vector,
                    'time_varying_total_global_moment_vector': tv_total_global_moment_vector,
                    'qblade_node_ids': qblade_node_ids,
                    'qblade_sim_path': qblade_sim_path,
                    'viv_params': viv_params,
                }
                
                # Save force time series if save path is provided
                if self.sim_save_var.get():
                    from pathlib import Path
                    save_dir = Path(self.sim_save_var.get())
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    force_file = save_dir / "forces_output.csv"
                    vorlap.write_force_time_series(
                        str(force_file),
                        inflow_time,
                        tv_global_force_vector_nodes,
                    )
                    ui_log(f"Force time series saved to: {force_file}\n")

                    if qblade_node_ids:
                        qblade_loading_target = self.qblade_loading_path_var.get().strip()
                        if qblade_loading_target:
                            qblade_loading_file = Path(qblade_loading_target)
                            if not qblade_loading_file.is_absolute():
                                qblade_loading_file = save_dir / qblade_loading_file
                        else:
                            qblade_loading_file = save_dir / "qblade_external_loading.txt"
                        qblade_loading_file.parent.mkdir(parents=True, exist_ok=True)
                        vorlap.write_qblade_loading_file(
                            str(qblade_loading_file),
                            inflow_time,
                            tv_global_force_vector_nodes,
                            qblade_node_ids,
                            local=False,
                        )
                        ui_log(f"QBlade loading file saved to: {qblade_loading_file}\n")
                
                # Update plots
                ui_update_plots()
                ui_log("Analysis completed successfully!\n\n")
                
            except Exception as e:
                ui_log(f"Error during analysis: {str(e)}\n")
                ui_error("Analysis Error", str(e))

        # Run analysis in separate thread to avoid GUI freezing
        threading.Thread(target=run_analysis, daemon=True).start()

    def _populate_frequency_entries(self, frequencies):
        for widget in self.freq_inner_frame.winfo_children():
            widget.destroy()

        for i, freq_val in enumerate(frequencies):
            entry = ttk.Entry(self.freq_inner_frame, width=8, justify="center")
            entry.insert(0, f"{float(freq_val):.2f}")
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
