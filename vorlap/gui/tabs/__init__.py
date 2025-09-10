#!/usr/bin/env python3
"""
Tab modules for the VorLap GUI.

This module provides access to all the tab classes.
"""

from .setup import SimulationSetupTab
from .geometry import GeometryTab
from .plots import PlotsOutputsTab
from .analysis import AnalysisTab

__all__ = [
    'SimulationSetupTab',
    'GeometryTab',
    'PlotsOutputsTab',
    'AnalysisTab'
] 