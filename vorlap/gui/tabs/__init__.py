#!/usr/bin/env python3
"""
Tab modules for the VorLap GUI.

This module provides access to all the tab classes.
"""

from .setup import SimulationSetupTab
from .plots import PlotsOutputsTab
from .qblade_bridge import QBladeBridgeTab

__all__ = [
    'SimulationSetupTab',
    'PlotsOutputsTab',
    'QBladeBridgeTab',
]
