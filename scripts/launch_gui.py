#!/usr/bin/env python3
"""
VORtex overLAP Tool – Refactored Tkinter GUI Entry Point.

This is the main entry point for the refactored VorLap GUI application.
"""

import sys
import subprocess


def _check_tkinter() -> None:
    try:
        import tkinter  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name in {"tkinter", "_tkinter"}:
            print(
                "VorLap GUI requires a Python build with Tkinter support.\n"
                "Current interpreter: "
                f"{sys.executable}\n\n"
                "Quick fix on macOS:\n"
                "1) Use a Tk-capable interpreter (Conda env directly or python.org Python)\n"
                "2) Reinstall VorLap dependencies\n\n"
                "Conda example:\n"
                "  conda create -n vorlap-gui python=3.11 -y\n"
                "  conda activate vorlap-gui\n"
                "  python -m pip install -e \".[dev,docs,gui]\"\n"
                "  python scripts/launch_gui.py\n\n"
                "python.org example:\n"
                "  rm -rf .venv\n"
                "  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv\n"
                "  source .venv/bin/activate\n"
                "  python -m pip install -e \".[dev,docs,gui]\"\n"
                "  python scripts/launch_gui.py\n"
            )
            raise SystemExit(1) from exc
        raise


_check_tkinter()


def _check_tk_window() -> None:
    # In frozen builds (e.g., PyInstaller), sys.executable points to the app
    # binary, not a Python interpreter. Spawning a subprocess with
    # `sys.executable -c ...` can recurse/hang; skip this preflight there.
    if getattr(sys, "frozen", False):
        print("VorLap GUI: frozen build detected; skipping Tk subprocess preflight check.")
        return

    # Run Tk window creation in a child process so hard aborts become
    # actionable launch-time diagnostics instead of opaque crashes.
    code = (
        "import tkinter as tk;"
        "root=tk.Tk();"
        "root.withdraw();"
        "root.update_idletasks();"
        "root.destroy()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    details = "\n".join(line for line in [stdout, stderr] if line)
    if details:
        detail_lines = details.splitlines()
        max_lines = 16
        if len(detail_lines) > max_lines:
            details = "\n".join(detail_lines[:max_lines]) + "\n... (truncated)"
        details = f"\n\nSubprocess output:\n{details}"
    print(
        "VorLap GUI could import tkinter but failed creating a Tk window.\n"
        "This usually indicates an interpreter/Tk runtime mismatch.\n"
        f"Current interpreter: {sys.executable}\n\n"
        "Recommended fix:\n"
        "1) Exit Conda base if active: `conda deactivate` until `(base)` is gone\n"
        "2) Use either a dedicated Conda environment or a python.org framework Python\n"
        "3) Reinstall dependencies and relaunch\n\n"
        "Conda example:\n"
        "  conda create -n vorlap-gui python=3.11 -y\n"
        "  conda activate vorlap-gui\n"
        "  python -m pip install -e \".[dev,docs,gui]\"\n"
        "  python scripts/launch_gui.py\n\n"
        "python.org example:\n"
        "  rm -rf .venv\n"
        "  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  python -m pip install -e \".[dev,docs,gui]\"\n"
        "  python scripts/launch_gui.py"
        f"{details}"
    )
    raise SystemExit(1)


_check_tk_window()

from vorlap.gui import main


if __name__ == "__main__":
    main()
