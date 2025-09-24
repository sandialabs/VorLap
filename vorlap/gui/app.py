#!/usr/bin/env python3
"""
Main VorLap GUI Application.

This module contains the main application class that orchestrates all GUI components.
"""

import tkinter as tk
from tkinter import ttk
import os
import sys
import glob
import numpy as np

# Add the vorlap package to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import vorlap

from vorlap.gui.tabs import SimulationSetupTab, PlotsOutputsTab #, AnalysisTab, GeometryTab
from vorlap.gui.styles import setup_theme_and_styling
from vorlap.gui.widgets import ScrollText


class VorLapApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VORtex overLAP Tool")
        self.geometry("1300x1200")
        self.minsize(1300, 1200)

        # Apply modern theme and styling
        setup_theme_and_styling(self)

        # Initialize data containers
        self.components = []
        self.natural_frequencies = None
        self.analysis_results = None

        # Use grid for proper resizing behavior
        self.rowconfigure(0, weight=1, minsize=100)   # Notebook expands with minimum size
        self.rowconfigure(1, weight=0)   # Console has fixed size
        self.columnconfigure(0, weight=1)

        # Create notebook with padding
        nb = ttk.Notebook(self)
        nb.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))

        self.tab_setup = SimulationSetupTab(nb, self)
        # self.tab_geometry = GeometryTab(nb, self)
        self.tab_plots = PlotsOutputsTab(nb, self)
        # self.tab_analysis = AnalysisTab(nb, self)

        nb.add(self.tab_setup, text="Simulation Setup")
        # nb.add(self.tab_geometry, text="Geometry")
        nb.add(self.tab_plots, text="Plots & Outputs")
        # nb.add(self.tab_analysis, text="Analysis")

        # Persistent console (non-collapsible)
        console_frame = ttk.LabelFrame(self, text="Console Output")
        console_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        console_frame.configure(height=200)  # Set fixed height
        console_frame.pack_propagate(False)  # Prevent shrinking
        
        self.console = ScrollText(console_frame, height=8)
        self.console.pack(fill="both", expand=True, padx=8, pady=8)

    def log(self, s: str):
        """Log message to console and status bar."""
        try:
            self.console.write(s)
        except Exception:
            print(s, end="")
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
