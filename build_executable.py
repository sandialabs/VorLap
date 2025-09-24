#!/usr/bin/env python3
"""
Build executable for VorLap using PyInstaller.
This script creates a standalone executable that can be distributed without Python installation.
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
    elif system == "darwin":  # macOS
        executable_name = "VorLap"
        icon_file = None  # Add .icns file path if available
    else:  # Linux and other Unix-like systems
        executable_name = "VorLap"
        icon_file = None  # Add .png file path if available
    
    return {
        "system": system,
        "machine": machine,
        "executable_name": executable_name,
        "icon_file": icon_file
    }

def find_entry_point():
    """Find the main entry point for the VorLap application."""
    # Check for GUI entry points
    possible_entry_points = [
        "scripts/launch_gui.py",
        "scripts/launch_gui_standalone.py", 
        "vorlap/gui/app.py",
        "kirklocal/gui.py"
    ]
    
    for entry_point in possible_entry_points:
        if os.path.exists(entry_point):
            print(f"Found entry point: {entry_point}")
            return entry_point
    
    # If no GUI found, create a simple launcher
    launcher_content = '''#!/usr/bin/env python3
"""
VorLap GUI Launcher
Simple launcher for VorLap GUI application
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Try to import and launch GUI
    import vorlap.gui.app as gui_app
    if hasattr(gui_app, 'main'):
        gui_app.main()
    else:
        print("GUI main function not found")
except ImportError:
    try:
        # Fallback to basic CLI interface
        import vorlap
        print(f"VorLap {vorlap.__version__} - Command Line Interface")
        print("GUI not available. Please use VorLap as a Python library:")
        print("  import vorlap")
        print("  # See documentation for usage examples")
    except ImportError as e:
        print(f"Error importing VorLap: {e}")
        sys.exit(1)

if __name__ == "__main__":
    pass
'''
    
    # Create temporary launcher
    launcher_path = "temp_launcher.py"
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    print(f"Created temporary launcher: {launcher_path}")
    return launcher_path

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
    elif system == "darwin":
        hidden_imports.extend(['Foundation', 'AppKit'])
    
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
    
    # Find entry point
    entry_point = find_entry_point()
    if not entry_point:
        print("Error: No suitable entry point found!")
        return False
    
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
    
    # Add hidden imports
    for import_name in hidden_imports:
        cmd.extend(["--hidden-import", import_name])
    
    # Add data files
    for src, dst in data_files:
        cmd.extend(["--add-data", f"{src}{os.pathsep}{dst}"])
    
    # Add icon if available
    if platform_info["icon_file"] and os.path.exists(platform_info["icon_file"]):
        cmd.extend(["--icon", platform_info["icon_file"]])
    
    # Add platform-specific options
    if platform_info["system"] == "darwin":
        # macOS specific options
        cmd.extend([
            "--osx-bundle-identifier", "gov.sandia.vorlap",
            "--target-arch", "universal2"  # Build universal binary
        ])
    
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

def verify_dependencies():
    """Verify that all required dependencies are available."""
    print("Verifying dependencies...")
    
    required_packages = ['pyinstaller', 'vorlap']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    print("All dependencies available!")
    return True

def main():
    """Main function to build VorLap executable."""
    print("VorLap Executable Builder")
    print("=" * 40)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}")
    
    # Verify dependencies
    if not verify_dependencies():
        sys.exit(1)
    
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
