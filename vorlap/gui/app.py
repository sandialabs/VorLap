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

from .tabs import SimulationSetupTab, ComponentsTab, PlotsOutputsTab, AnalysisTab
from .styles import setup_theme_and_styling


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
            default_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "natural_frequencies.csv")
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