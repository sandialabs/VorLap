#!/usr/bin/env python3
"""
Analysis Tab for the VorLap GUI.

This tab provides analysis tools and visualization options.
"""

import tkinter as tk
from tkinter import ttk, messagebox

# Add the vorlap package to the path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import vorlap


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