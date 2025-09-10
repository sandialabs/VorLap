#!/usr/bin/env python3
"""
VORtex overLAP Tool – Tkinter GUI with tabs, editable tables, and integrated plotting.

This module provides the main entry point for the VorLap GUI application.
"""

from .app import VorLapApp

def main():
    """Main entry point for the VorLap GUI application."""
    app = VorLapApp()
    app.mainloop()

if __name__ == "__main__":
    main() 