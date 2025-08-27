#!/usr/bin/env python3
"""
Simple launcher script for the VorLap GUI.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from vorlap_gui import main
    main()
except ImportError as e:
    print(f"Error importing vorlap_gui: {e}")
    print("Make sure all dependencies are installed:")
    print("- numpy")
    print("- matplotlib")
    print("- tkinter (usually comes with Python)")
    print("- scipy")
    print("- h5py")
except Exception as e:
    print(f"Error running GUI: {e}")
    import traceback
    traceback.print_exc() 