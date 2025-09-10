#!/usr/bin/env python3
"""
Geometry Tab for the VorLap GUI.

This tab displays the structure geometry visualization.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import tempfile

# Add the vorlap package to the path
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import vorlap

from ..widgets import ScrollText


class GeometryTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.current_fig = None
        self._build()

    def _build(self):
        # Top controls
        top_frame = ttk.Frame(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        ttk.Button(top_frame, text="Show Geometry", command=self.show_geometry).pack(side="left", padx=5)
        ttk.Button(top_frame, text="Clear Display", command=self.clear_display).pack(side="left", padx=5)
        
        # Geometry display area
        self.geom_frame = ttk.LabelFrame(self, text="Structure Geometry")
        self.geom_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        
        # Placeholder text when no geometry is shown
        self.placeholder = ttk.Label(self.geom_frame, text="Click 'Show Geometry' to display the structure visualization", 
                                   font=('Segoe UI', 12), foreground='gray')
        self.placeholder.pack(expand=True, pady=50)
        
        # Console output
        self.console = ScrollText(self, height=8)
        self.console.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        
        # Configure grid weights
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

    def show_geometry(self):
        """Show structure geometry visualization."""
        if not hasattr(self.app, 'components') or not self.app.components:
            messagebox.showwarning("No Data", "Load components first.")
            return
            
        try:
            viv_params = self.app.tab_setup.get_viv_params()
            
            # Generate the plotly figure without displaying it
            fig = vorlap.graphics.calc_structure_vectors_andplot(
                self.app.components, 
                viv_params, 
                show_plot=False, 
                return_fig=True
            )
            
            if fig is not None:
                # Hide the placeholder
                self.placeholder.pack_forget()
                
                # Clear any existing plotly display
                if hasattr(self, 'plotly_html'):
                    self.plotly_html.destroy()
                
                try:
                    # Use webview for reliable Plotly rendering
                    import webview
                    
                    # Convert plotly figure to HTML
                    html_content = fig.to_html(include_plotlyjs='cdn')
                    
                    # Create a temporary HTML file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                        f.write(html_content)
                        html_file = f.name
                    
                    # Create webview window
                    webview.create_window('Structure Geometry', html_file, 
                                       width=800, height=600, 
                                       resizable=True, 
                                       text_select=True)
                    webview.start()
                    
                    # Clean up temporary file
                    try:
                        os.unlink(html_file)
                    except:
                        pass
                    
                    self.current_fig = fig
                    self.app.log("[Geometry] Structure visualization displayed in webview\n")
                    self.console.write("Geometry visualization displayed successfully\n")
                    self.console.write("Interactive 3D plot opened in webview window\n")
                    
                except ImportError:
                    # Fallback: show in browser if webview not available
                    self.console.write("webview not available, opening in browser instead\n")
                    fig.show()
                    self.app.log("[Geometry] Structure visualization displayed in browser\n")
                    self.console.write("Geometry visualization displayed in browser\n")
                
            else:
                messagebox.showwarning("No Figure", "No geometry figure was generated.")
                
        except Exception as e:
            error_msg = f"Error displaying geometry: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.app.log(f"[Geometry] {error_msg}\n")
            self.console.write(f"Error: {error_msg}\n")

    def clear_display(self):
        """Clear the geometry display."""
        # Hide the plotly HTML widget and show placeholder
        if hasattr(self, 'plotly_html'):
            self.plotly_html.destroy()
            delattr(self, 'plotly_html')
        
        self.placeholder.pack(expand=True, pady=50)
        self.current_fig = None
        
        self.console.write("Display cleared\n")
        self.app.log("[Geometry] Display cleared\n")

    def log(self, s: str):
        """Log message to console."""
        self.console.write(s) 