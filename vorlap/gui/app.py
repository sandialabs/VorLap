#!/usr/bin/env python3
"""
Main VorLap GUI Application.

This module contains the main application class that orchestrates all GUI components.
"""

import tkinter as tk
from tkinter import ttk
import glob
import os

import vorlap

from vorlap.gui.tabs import SimulationSetupTab, PlotsOutputsTab #, AnalysisTab, GeometryTab
from vorlap.gui.styles import setup_theme_and_styling
from vorlap.gui.widgets import ScrollText


class VorLapApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VORtex overLAP Tool")
        self.geometry("1300x980")
        self.minsize(1100, 780)

        # Default to light theme.
        self.light_mode = tk.BooleanVar(value=True)
        setup_theme_and_styling(self, mode="light")

        # Initialize data containers
        self.components = []
        self.natural_frequencies = None
        self.analysis_results = None

        # Use grid for proper resizing behavior.
        self.rowconfigure(0, weight=0)   # Top bar
        self.rowconfigure(1, weight=1)   # Notebook
        self.rowconfigure(2, weight=0)   # Console
        self.columnconfigure(0, weight=1)

        # Top bar with theme toggle.
        top_bar = ttk.Frame(self)
        top_bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        top_bar.columnconfigure(0, weight=1)
        ttk.Label(top_bar, text="Theme").grid(row=0, column=1, padx=(0, 6))
        ttk.Checkbutton(
            top_bar,
            text="Light",
            variable=self.light_mode,
            command=self.toggle_theme,
        ).grid(row=0, column=2, sticky="e")

        # Create notebook with padding.
        self.nb = ttk.Notebook(self)
        self.nb.grid(row=1, column=0, sticky="nsew", padx=8, pady=(8, 0))

        self.tab_setup = SimulationSetupTab(self.nb, self)
        # self.tab_geometry = GeometryTab(nb, self)
        self.tab_plots = PlotsOutputsTab(self.nb, self)
        # self.tab_analysis = AnalysisTab(nb, self)

        self.nb.add(self.tab_setup, text="Simulation Setup")
        # nb.add(self.tab_geometry, text="Geometry")
        self.nb.add(self.tab_plots, text="Plots & Outputs")
        # nb.add(self.tab_analysis, text="Analysis")
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Persistent console (non-collapsible)
        console_frame = ttk.LabelFrame(self, text="Console Output")
        console_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        console_frame.configure(height=200)  # Set fixed height
        console_frame.pack_propagate(False)  # Prevent shrinking
        
        self.console = ScrollText(console_frame, height=8)
        self.console.pack(fill="both", expand=True, padx=8, pady=8)

    def toggle_theme(self):
        mode = "light" if self.light_mode.get() else "dark"
        setup_theme_and_styling(self, mode=mode)

    def _on_tab_changed(self, event):
        try:
            tab_widget = event.widget.nametowidget(event.widget.select())
            def _refresh():
                try:
                    if hasattr(tab_widget, "on_tab_selected"):
                        tab_widget.on_tab_selected()
                    if hasattr(tab_widget, "scrollable"):
                        tab_widget.scrollable.refresh()
                    tab_widget.update_idletasks()
                    self.update_idletasks()
                except Exception:
                    pass
            self.after_idle(_refresh)
            self.after(30, _refresh)
        except Exception:
            pass

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
