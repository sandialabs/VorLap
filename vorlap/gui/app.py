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

from vorlap.gui.tabs import SimulationSetupTab, GeometryTab, PlotsOutputsTab, AnalysisTab
from vorlap.gui.styles import setup_theme_and_styling


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
        setup_theme_and_styling(self)

        # Initialize data containers
        self.components = []
        self.natural_frequencies = None
        self.analysis_results = None

        # Create notebook with padding
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        self.tab_setup = SimulationSetupTab(nb, self)
        self.tab_geometry = GeometryTab(nb, self)
        self.tab_plots = PlotsOutputsTab(nb, self)
        self.tab_analysis = AnalysisTab(nb, self)

        nb.add(self.tab_setup, text="Simulation Setup")
        nb.add(self.tab_geometry, text="Geometry")
        nb.add(self.tab_plots, text="Plots & Outputs")
        nb.add(self.tab_analysis, text="Analysis")

        # status bar with improved styling
        status_frame = ttk.Frame(self, style='StatusFrame.TFrame')
        status_frame.pack(fill="x", side="bottom", padx=8, pady=(0, 8))
        
        self.status = ttk.Label(status_frame, text="Ready", anchor="w", style='Status.TLabel')
        self.status.pack(fill="x", padx=8, pady=4)

    def log(self, s: str):
        """Log message to console and status bar."""
        try:
            self.tab_plots.log(s)
        except Exception:
            print(s, end="")
        self.status.config(text=s.strip().splitlines()[-1] if s.strip() else "Ready")

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