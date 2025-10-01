#!/usr/bin/env python3
"""
Build executable for VorLap using PyInstaller.
This script creates a standalone executable that can be distributed without Python installation.

NOTE: This is currently in-development and may not work for all platforms.
Due to extra requirements to make executables for MacOS, MacOS is not supported yet.

MacOS Support Requirements:
1. Separate builds for Intel (x86_64) and Apple Silicon (arm64)
2. Use --onedir mode instead of --onefile
3. Code signing with Apple Developer certificates (for distribution)
4. Notarization for Gatekeeper compatibility
"""

import os
import sys
import platform
import shutil
import subprocess
from pathlib import Path

def get_platform_info():
    """Get platform-specific information for building."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        executable_name = "VorLap.exe"
        icon_file = None  # Add .ico file path if available
    else:  # Linux and other Unix-like systems
        executable_name = "VorLap"
        icon_file = None  # Add .png file path if available
    
    return {
        "system": system,
        "machine": machine,
        "executable_name": executable_name,
        "icon_file": icon_file
    }


def get_hidden_imports():
    """Get list of hidden imports needed for PyInstaller."""
    hidden_imports = [
        'numpy',
        'scipy',
        'pandas', 
        'h5py',
        'plotly',
        'matplotlib',
        'tkinter',
        'vorlap',
        'vorlap.computations',
        'vorlap.fileio',
        'vorlap.graphics',
        'vorlap.interpolation',
        'vorlap.structs',
        'vorlap.gui',
        'vorlap.gui.app',
        'vorlap.gui.widgets',
        'vorlap.gui.styles'
    ]
    
    # Add platform-specific imports
    system = platform.system().lower()
    if system == "windows":
        hidden_imports.extend(['win32api', 'win32gui', 'win32con'])
    
    return hidden_imports

def get_data_files():
    """Get list of data files to include in the executable."""
    data_files = []
    
    # Include data directory if it exists
    if os.path.exists("data"):
        data_files.append(("data", "data"))
    
    # Include any configuration files
    config_files = ["default_parameters.csv"]
    for config_file in config_files:
        if os.path.exists(config_file):
            data_files.append((config_file, "."))
    
    return data_files

def build_executable():
    """Build the VorLap executable using PyInstaller."""
    print("Building VorLap executable...")
    print(f"Platform: {platform.system()} {platform.machine()}")
    
    # Get platform info
    platform_info = get_platform_info()
    
    # Set entry point
    entry_point = "vorlap/gui/app.py"
    
    # Get hidden imports
    hidden_imports = get_hidden_imports()
    
    # Get data files
    data_files = get_data_files()
    
    # Build PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",  # Create a single executable file
        "--windowed",  # Don't show console window (for GUI apps)
        f"--name={platform_info['executable_name'].replace('.exe', '')}",  # Remove .exe, PyInstaller adds it
        "--clean",  # Clean PyInstaller cache
        "--noconfirm",  # Overwrite output directory without confirmation
    ]
    
    # Add Linux-specific options to improve compatibility
    if platform_info["system"] == "linux":
        cmd.extend([
            "--strip",  # Strip debug symbols to reduce size
            "--noupx"   # Disable UPX compression for better compatibility
        ])
    
    # Add hidden imports
    for import_name in hidden_imports:
        cmd.extend(["--hidden-import", import_name])
    
    # Add data files
    for src, dst in data_files:
        cmd.extend(["--add-data", f"{src}{os.pathsep}{dst}"])
    
    # Add icon if available
    if platform_info["icon_file"] and os.path.exists(platform_info["icon_file"]):
        cmd.extend(["--icon", platform_info["icon_file"]])
    
    # Add entry point
    cmd.append(entry_point)
    
    print(f"PyInstaller command: {' '.join(cmd)}")
    
    # Run PyInstaller
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("PyInstaller completed successfully!")
        print(result.stdout)
        
        # Verify executable was created
        dist_dir = Path("dist")
        executable_path = dist_dir / platform_info["executable_name"]
        
        if executable_path.exists():
            file_size = executable_path.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"Executable created: {executable_path}")
            print(f"File size: {file_size:.2f} MB")
            
            # Make executable on Unix-like systems
            if platform_info["system"] != "windows":
                os.chmod(executable_path, 0o755)
                print("Made executable file executable")
            
            return True
        else:
            print(f"Error: Executable not found at {executable_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"PyInstaller failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    finally:
        # Clean up temporary launcher if created
        if entry_point == "temp_launcher.py" and os.path.exists("temp_launcher.py"):
            os.remove("temp_launcher.py")
            print("Cleaned up temporary launcher")


def main():
    """Main function to build VorLap executable."""
    print("VorLap Executable Builder")
    print("=" * 40)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}")
    
    # Clean previous builds
    if os.path.exists("dist"):
        print("Cleaning previous build...")
        shutil.rmtree("dist")
    
    if os.path.exists("build"):
        print("Cleaning build cache...")
        shutil.rmtree("build")
    
    # Build executable
    success = build_executable()
    
    if success:
        print("\n" + "=" * 40)
        print("Build completed successfully!")
        print("Executable is available in the 'dist' directory")
        
        # List contents of dist directory
        if os.path.exists("dist"):
            print("\nDist directory contents:")
            for item in os.listdir("dist"):
                item_path = os.path.join("dist", item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path) / (1024 * 1024)
                    print(f"  {item} ({size:.2f} MB)")
                else:
                    print(f"  {item}/ (directory)")
    else:
        print("\n" + "=" * 40)
        print("Build failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
