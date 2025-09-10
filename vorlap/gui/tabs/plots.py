#!/usr/bin/env python3
"""
Plots & Outputs Tab for the VorLap GUI.

This tab handles visualization, plotting, and data export functionality.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import csv

# Add the vorlap package to the path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import vorlap

from ..widgets import PathEntry

# --- Optional plotting support ---
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False


class PlotsOutputsTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._build()

    def _build(self):
        # Center-left: Plot area
        self.plot_frame = ttk.LabelFrame(self, text="Analysis Results")
        self.plot_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=10, pady=6)
        

        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("VorLap Analysis Results")
        self.ax.text(0.5, 0.5, "Run analysis to see results", ha='center', va='center', transform=self.ax.transAxes)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        
        # Configure plot frame grid weights
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(1, weight=1)
        
        # Initialize colorbar attribute
        self._colorbar = None

        # Plot type selection
        plot_controls = ttk.Frame(self.plot_frame)
        plot_controls.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Label(plot_controls, text="Plot Type:").grid(row=0, column=0, sticky="w")
        
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
        
        for i, (text, value) in enumerate(plot_types):
            ttk.Radiobutton(plot_controls, text=text, variable=self.plot_type, 
                          value=value, command=self.update_plots).grid(row=0, column=i+1, padx=5)
        
        # Add save button
        ttk.Button(plot_controls, text="Save Plot", command=self.save_plot).grid(row=0, column=len(plot_types)+1, padx=5)

        # Bottom: Sample controls + export path + Export button
        # weights
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)


    def update_plots(self):
        """Update the plot based on current analysis results and selected plot type."""
        if not hasattr(self.app, 'analysis_results') or not self.app.analysis_results:
            return
            
        results = self.app.analysis_results
        plot_type = self.plot_type.get()
        viv_params = results['viv_params']
        
        # Create a completely fresh figure and canvas
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        
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
        
        # Add colorbar to the fresh figure
        self._colorbar = self.fig.colorbar(im, ax=self.ax, label=label)
        
        # Destroy the old canvas and create a new one with the fresh figure
        old_canvas = self.canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        
        # Replace the old canvas widget in the grid
        old_canvas.get_tk_widget().destroy()
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

    # ---- handlers ----
    def save_plot(self):
        if not self.fig:
            messagebox.showinfo("Plot", "No Matplotlib figure available.")
            return
        
        # Get simulation save path from setup tab
        sim_save_path = self.app.tab_setup.sim_save.get()
        if not sim_save_path:
            messagebox.showwarning("No Save Path", "Please set a simulation save path in the Setup tab first.")
            return
        
        # Create plots subfolder in simulation directory
        plots_dir = Path(sim_save_path) / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plot_type = self.plot_type.get()
        path = plots_dir / f"{plot_type}_plot.png"
        self.fig.savefig(path, dpi=150, bbox_inches='tight')
        self.app.log(f"Plot saved to: {path}\n")


