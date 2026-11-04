#!/usr/bin/env python3
"""
Main VorLap GUI Application.

This module contains the main application class that orchestrates all GUI components.
"""

import tkinter as tk
from tkinter import ttk
import logging

import vorlap

from vorlap.gui.tabs import SimulationSetupTab, PlotsOutputsTab, QBladeBridgeTab
from vorlap.gui.state import VorLapGuiState
from vorlap.gui.styles import setup_theme_and_styling
from vorlap.gui.widgets import ScrollText
from vorlap.qblade_bridge_service import QBladeBridgeService


LOGGER = logging.getLogger(__name__)

NOTEBOOK_TAB_LABELS = ("Simulation Setup", "Plots & Outputs", "QBlade-Bridge")


class VorLapApp(tk.Tk):
    def __init__(self, qblade_bridge_service=None):
        super().__init__()
        self.title("VORtex overLAP Tool")
        self.geometry("1300x980")
        self.minsize(1100, 780)

        # Default to light theme.
        self.light_mode = tk.BooleanVar(value=True)
        setup_theme_and_styling(self, mode="light")

        # Shared GUI state; legacy app.* properties delegate here.
        self.state = VorLapGuiState()
        self.qblade_bridge_service = qblade_bridge_service or QBladeBridgeService()

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
        self.tab_plots = PlotsOutputsTab(self.nb, self)
        self.tab_qblade_bridge = QBladeBridgeTab(self.nb, self, service=self.qblade_bridge_service)

        self.nb.add(self.tab_setup, text=NOTEBOOK_TAB_LABELS[0])
        self.nb.add(self.tab_plots, text=NOTEBOOK_TAB_LABELS[1])
        self.nb.add(self.tab_qblade_bridge, text=NOTEBOOK_TAB_LABELS[2])
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Persistent console (non-collapsible)
        console_frame = ttk.LabelFrame(self, text="Console Output")
        console_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        console_frame.configure(height=200)  # Set fixed height
        console_frame.pack_propagate(False)  # Prevent shrinking
        
        self.console = ScrollText(console_frame, height=8)
        self.console.pack(fill="both", expand=True, padx=8, pady=8)

    @property
    def components(self):
        return self.state.components

    @components.setter
    def components(self, value):
        self.state.set_components(value)

    @property
    def natural_frequencies(self):
        return self.state.natural_frequencies

    @natural_frequencies.setter
    def natural_frequencies(self, value):
        self.state.set_natural_frequencies(value)

    @property
    def analysis_results(self):
        return self.state.analysis_results

    @analysis_results.setter
    def analysis_results(self, value):
        self.state.set_analysis_results(value)

    @property
    def qblade_sim_path(self):
        return self.state.qblade_sim_path

    @qblade_sim_path.setter
    def qblade_sim_path(self, value):
        self.state.set_qblade_sim_path(value)

    def toggle_theme(self):
        mode = "light" if self.light_mode.get() else "dark"
        setup_theme_and_styling(self, mode=mode)

    def _on_tab_changed(self, event):
        try:
            selected_tab = event.widget.select()
            tab_widget = event.widget.nametowidget(selected_tab)
        except tk.TclError:
            LOGGER.debug("Could not resolve selected notebook tab for refresh.", exc_info=True)
            return

        def _refresh():
            try:
                if hasattr(tab_widget, "on_tab_selected"):
                    tab_widget.on_tab_selected()
                if hasattr(tab_widget, "scrollable"):
                    tab_widget.scrollable.refresh()
                tab_widget.update_idletasks()
                self.update_idletasks()
            except Exception:
                LOGGER.debug("Could not refresh selected notebook tab.", exc_info=True)

        self.after_idle(_refresh)
        self.after(30, _refresh)

    def log(self, s: str):
        """Log message to console and status bar."""
        try:
            self.console.write(s)
        except Exception:
            print(s, end="")
    def load_airfoils(self, airfoil_folder):
        """Load airfoil FFT data from the specified folder."""
        try:
            affts = vorlap.load_airfoils_from_folder(airfoil_folder)
            self.log(f"Loaded {len(affts)} airfoil files from {airfoil_folder}\n")
            return affts
        except Exception as e:
            self.log(f"Error loading airfoils: {str(e)}\n")
            return {} 
