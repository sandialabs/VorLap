#!/usr/bin/env python3
"""
VorLap GUI - A tkinter frontend for the VorLap wind turbine analysis package.

This GUI provides a user-friendly interface for:
- Editing VIV_Params configuration
- Running VorLap analysis
- Viewing results with embedded plots
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
import glob
import time

import vorlap.graphics
try:
    import plotly.graph_objects as go
    from plotly.offline import plot
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Structure visualization will be disabled.")

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Some image features may be disabled.")

# Add the vorlap package to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import vorlap


class VorLapGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VorLap - Wind Turbine Analysis")
        self.root.geometry("1200x800")
        
        # Initialize default parameters
        self.viv_params = vorlap.VIV_Params()
        self.components = None
        self.affts = {}
        self.natfreqs = None
        self.results = None
        
        # Setup GUI
        self.setup_gui()
        self.populate_parameter_table()
        
    def setup_gui(self):
        """Setup the main GUI layout."""
        # Create main frames
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Parameters tab
        self.params_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.params_frame, text="Parameters")
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        
        # Structure visualization tab
        self.structure_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.structure_frame, text="Structure")
        
        self.setup_parameters_tab()
        self.setup_results_tab()
        self.setup_structure_tab()
        
    def setup_parameters_tab(self):
        """Setup the parameters configuration tab."""
        # File selection frame
        file_frame = ttk.LabelFrame(self.params_frame, text="File Selection", padding="5")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Components folder
        ttk.Label(file_frame, text="Components Folder:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.components_var = tk.StringVar(value=os.path.join(os.path.dirname(__file__), "componentsHVAWT"))
        ttk.Entry(file_frame, textvariable=self.components_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_components_folder).grid(row=0, column=2, padx=5)
        
        # Airfoil folder
        ttk.Label(file_frame, text="Airfoil Folder:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.airfoil_var = tk.StringVar(value=os.path.join(os.path.dirname(__file__), "airfoils"))
        ttk.Entry(file_frame, textvariable=self.airfoil_var, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_airfoil_folder).grid(row=1, column=2, padx=5)
        
        # Natural frequencies file
        ttk.Label(file_frame, text="Natural Frequencies:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.natfreq_var = tk.StringVar(value=os.path.join(os.path.dirname(__file__), "natural_frequencies.csv"))
        ttk.Entry(file_frame, textvariable=self.natfreq_var, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_natfreq_file).grid(row=2, column=2, padx=5)
        
        # Parameters table frame
        params_table_frame = ttk.LabelFrame(self.params_frame, text="VIV Parameters", padding="5")
        params_table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for parameters
        columns = ("Parameter", "Value", "Description")
        self.params_tree = ttk.Treeview(params_table_frame, columns=columns, show="headings", height=15)
        
        # Setup column headings
        self.params_tree.heading("Parameter", text="Parameter")
        self.params_tree.heading("Value", text="Value")
        self.params_tree.heading("Description", text="Description")
        
        # Setup column widths
        self.params_tree.column("Parameter", width=200)
        self.params_tree.column("Value", width=150)
        self.params_tree.column("Description", width=300)
        
        # Scrollbars for the treeview
        v_scrollbar = ttk.Scrollbar(params_table_frame, orient=tk.VERTICAL, command=self.params_tree.yview)
        h_scrollbar = ttk.Scrollbar(params_table_frame, orient=tk.HORIZONTAL, command=self.params_tree.xview)
        self.params_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.params_tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        params_table_frame.grid_columnconfigure(0, weight=1)
        params_table_frame.grid_rowconfigure(0, weight=1)
        
        # Bind double-click for editing
        self.params_tree.bind("<Double-1>", self.edit_parameter)
        
        # Control buttons frame
        control_frame = ttk.Frame(self.params_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Load Default Parameters", command=self.load_default_params).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Run Analysis", command=self.run_analysis).pack(side=tk.RIGHT, padx=5)
        
    def setup_results_tab(self):
        """Setup the results visualization tab."""
        # Create notebook for different plots
        self.plots_notebook = ttk.Notebook(self.results_frame)
        self.plots_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Status frame
        status_frame = ttk.Frame(self.results_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready - Load parameters and run analysis")
        self.status_label.pack(side=tk.LEFT)
        
        self.progress_bar = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=10)
    
    def setup_structure_tab(self):
        """Setup the structure visualization tab."""
        # Control buttons
        control_frame = ttk.Frame(self.structure_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Generate Structure Plot", command=self.generate_structure_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Structure Plot", command=self.save_structure_plot).pack(side=tk.LEFT, padx=5)
        
        # Structure display frame
        self.structure_display_frame = ttk.Frame(self.structure_frame)
        self.structure_display_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Initially show a message
        self.structure_message_label = ttk.Label(
            self.structure_display_frame, 
            text="Load components and generate structure plot to view 3D visualization",
            font=("Arial", 12)
        )
        self.structure_message_label.pack(expand=True)
        
        # Store structure plot figure
        self.structure_fig = None
        
    def populate_parameter_table(self):
        """Populate the parameter table with current VIV_Params values."""
        # Clear existing items
        for item in self.params_tree.get_children():
            self.params_tree.delete(item)
        
        # Parameter definitions with descriptions
        param_info = [
            ("fluid_density", self.viv_params.fluid_density, "Air density [kg/m³]"),
            ("fluid_dynamicviscosity", self.viv_params.fluid_dynamicviscosity, "Dynamic viscosity [Pa·s]"),
            ("rotation_axis", str(self.viv_params.rotation_axis.tolist()), "Rotation axis vector [x,y,z]"),
            ("rotation_axis_offset", str(self.viv_params.rotation_axis_offset.tolist()), "Rotation axis origin [x,y,z]"),
            ("inflow_vec", str(self.viv_params.inflow_vec.tolist()), "Inflow direction vector [x,y,z]"),
            ("azimuths", f"{self.viv_params.azimuths[0]:.0f}:{len(self.viv_params.azimuths)}:{self.viv_params.azimuths[-1]:.0f}", "Azimuth angles [deg] (start:count:end)"),
            ("inflow_speeds", f"{self.viv_params.inflow_speeds[0]:.1f}:{len(self.viv_params.inflow_speeds)}:{self.viv_params.inflow_speeds[-1]:.1f}", "Inflow speeds [m/s] (start:count:end)"),
            ("output_time", f"{self.viv_params.output_time[0]:.3f}:{len(self.viv_params.output_time)}:{self.viv_params.output_time[-1]:.3f}", "Output time [s] (start:count:end)"),
            ("n_harmonic", self.viv_params.n_harmonic, "Number of harmonics"),
            ("amplitude_coeff_cutoff", self.viv_params.amplitude_coeff_cutoff, "Amplitude coefficient cutoff"),
            ("n_freq_depth", self.viv_params.n_freq_depth, "Frequency depth"),
            ("output_azimuth_vinf", str(self.viv_params.output_azimuth_vinf), "Output azimuth/vinf limits (az, vinf)"),
        ]
        
        for param_name, value, description in param_info:
            self.params_tree.insert("", tk.END, values=(param_name, value, description))
    
    def edit_parameter(self, event):
        """Handle parameter editing when double-clicking on a parameter."""
        item = self.params_tree.selection()[0]
        param_name = self.params_tree.item(item, "values")[0]
        current_value = self.params_tree.item(item, "values")[1]
        
        # Create edit dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit {param_name}")
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text=f"Edit {param_name}:").pack(pady=10)
        
        entry_var = tk.StringVar(value=str(current_value))
        entry = ttk.Entry(dialog, textvariable=entry_var, width=50)
        entry.pack(pady=5)
        
        def save_parameter():
            try:
                new_value = entry_var.get()
                self.update_parameter(param_name, new_value)
                self.params_tree.item(item, values=(param_name, new_value, self.params_tree.item(item, "values")[2]))
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Invalid value: {str(e)}")
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Save", command=save_parameter).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        entry.focus()
        entry.select_range(0, tk.END)
    
    def update_parameter(self, param_name, value_str):
        """Update a parameter in the VIV_Params object."""
        if param_name == "fluid_density":
            self.viv_params.fluid_density = float(value_str)
        elif param_name == "fluid_dynamicviscosity":
            self.viv_params.fluid_dynamicviscosity = float(value_str)
        elif param_name == "rotation_axis":
            self.viv_params.rotation_axis = np.array(eval(value_str))
        elif param_name == "rotation_axis_offset":
            self.viv_params.rotation_axis_offset = np.array(eval(value_str))
        elif param_name == "inflow_vec":
            self.viv_params.inflow_vec = np.array(eval(value_str))
        elif param_name == "azimuths":
            # Parse format "start:count:end"
            if ":" in value_str:
                parts = value_str.split(":")
                start, count, end = float(parts[0]), int(parts[1]), float(parts[2])
                self.viv_params.azimuths = np.linspace(start, end, count)
            else:
                self.viv_params.azimuths = np.array(eval(value_str))
        elif param_name == "inflow_speeds":
            # Parse format "start:count:end"
            if ":" in value_str:
                parts = value_str.split(":")
                start, count, end = float(parts[0]), int(parts[1]), float(parts[2])
                self.viv_params.inflow_speeds = np.linspace(start, end, count)
            else:
                self.viv_params.inflow_speeds = np.array(eval(value_str))
        elif param_name == "output_time":
            # Parse format "start:count:end"
            if ":" in value_str:
                parts = value_str.split(":")
                start, count, end = float(parts[0]), int(parts[1]), float(parts[2])
                self.viv_params.output_time = np.linspace(start, end, count)
            else:
                self.viv_params.output_time = np.array(eval(value_str))
        elif param_name == "n_harmonic":
            self.viv_params.n_harmonic = int(value_str)
        elif param_name == "amplitude_coeff_cutoff":
            self.viv_params.amplitude_coeff_cutoff = float(value_str)
        elif param_name == "n_freq_depth":
            self.viv_params.n_freq_depth = int(value_str)
        elif param_name == "output_azimuth_vinf":
            self.viv_params.output_azimuth_vinf = eval(value_str)
    
    def browse_components_folder(self):
        """Browse for components folder."""
        folder = filedialog.askdirectory(title="Select Components Folder")
        if folder:
            self.components_var.set(folder)
    
    def browse_airfoil_folder(self):
        """Browse for airfoil folder."""
        folder = filedialog.askdirectory(title="Select Airfoil Folder")
        if folder:
            self.airfoil_var.set(folder)
            self.viv_params.airfoil_folder = folder
    
    def browse_natfreq_file(self):
        """Browse for natural frequencies file."""
        file = filedialog.askopenfilename(
            title="Select Natural Frequencies File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file:
            self.natfreq_var.set(file)
    
    def load_default_params(self):
        """Load default parameters into the table."""
        self.viv_params = vorlap.VIV_Params(
            fluid_density=1.225,
            fluid_dynamicviscosity=1.81e-5,
            rotation_axis=np.array([0.0, 0.0, 1.0]),
            rotation_axis_offset=np.array([0.0, 0.0, 0.0]),
            inflow_vec=np.array([1.0, 0.0, 0.0]),
            azimuths=np.arange(0, 260, 5),
            inflow_speeds=np.arange(2.0, 50.5, 0.5),
            n_harmonic=2,
            output_time=np.arange(0.0, 0.011, 0.001),
            output_azimuth_vinf=(5.0, 2.0),
            amplitude_coeff_cutoff=0.2,
            n_freq_depth=10,
            airfoil_folder=self.airfoil_var.get()
        )
        self.populate_parameter_table()
    
    def run_analysis(self):
        """Run the VorLap analysis with current parameters."""
        try:
            # Update status
            self.status_label.config(text="Loading components and airfoils...")
            self.progress_bar.start()
            self.root.update()
            
            # Load components
            if not os.path.exists(self.components_var.get()):
                messagebox.showerror("Error", f"Components folder not found: {self.components_var.get()}")
                return
            
            self.components = vorlap.load_components_from_csv(self.components_var.get())
            
            # Load airfoils
            self.affts = {}
            airfoil_folder = self.airfoil_var.get()
            if os.path.exists(airfoil_folder):
                for file in glob.glob(os.path.join(airfoil_folder, "*.h5")):
                    afft = vorlap.load_airfoil_fft(file)
                    self.affts[afft.name] = afft
            
            if "default" not in self.affts and self.affts:
                self.affts["default"] = next(iter(self.affts.values()))
            
            # Load natural frequencies
            if os.path.exists(self.natfreq_var.get()):
                self.natfreqs = np.loadtxt(self.natfreq_var.get(), delimiter=',')
            else:
                messagebox.showerror("Error", f"Natural frequencies file not found: {self.natfreq_var.get()}")
                return
            
            # Update airfoil folder in params
            self.viv_params.airfoil_folder = airfoil_folder
            
            # Generate structure visualization
            self.status_label.config(text="Generating structure visualization...")
            self.root.update()
            
            if PLOTLY_AVAILABLE:
                try:
                    self.structure_fig = vorlap.graphics.calc_structure_vectors_andplot(self.components, self.viv_params)
                except Exception as e:
                    print(f"Warning: Could not generate structure plot: {e}")
            
            # Run analysis
            self.status_label.config(text="Computing thrust and torque spectrum...")
            self.root.update()
            
            start_time = time.time()
            percdiff_matrix, percdiff_info, total_global_force_vector, total_global_moment_vector, global_force_vector_nodes = vorlap.compute_thrust_torque_spectrum(
                self.components, self.affts, self.viv_params, self.natfreqs
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Store results
            self.results = {
                'percdiff_matrix': percdiff_matrix,
                'percdiff_info': percdiff_info,
                'total_global_force_vector': total_global_force_vector,
                'total_global_moment_vector': total_global_moment_vector,
                'global_force_vector_nodes': global_force_vector_nodes,
                'execution_time': execution_time
            }
            
            # Create visualizations
            self.create_visualizations()
            
            # Display structure plot if available
            if self.structure_fig is not None:
                self.display_structure_plot()
            
            self.progress_bar.stop()
            self.status_label.config(text=f"Analysis completed in {execution_time:.2f} seconds")
            
            # Switch to results tab
            self.notebook.select(self.results_frame)
            
        except Exception as e:
            self.progress_bar.stop()
            self.status_label.config(text="Error occurred during analysis")
            messagebox.showerror("Analysis Error", f"An error occurred: {str(e)}")
    
    def create_visualizations(self):
        """Create and display the analysis results."""
        if not self.results:
            return
        
        # Clear existing plot tabs
        for tab in self.plots_notebook.tabs():
            self.plots_notebook.forget(tab)
        
        # Create plots
        self.create_percent_diff_plot()
        self.create_force_plots()
        self.create_moment_plots()
    
    def create_percent_diff_plot(self):
        """Create the percent difference plot."""
        frame = ttk.Frame(self.plots_notebook)
        self.plots_notebook.add(frame, text="Percent Difference")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(self.results['percdiff_matrix'], 
                      extent=[self.viv_params.azimuths[0], self.viv_params.azimuths[-1], 
                             self.viv_params.inflow_speeds[0], self.viv_params.inflow_speeds[-1]],
                      aspect='auto', origin='lower', cmap='viridis_r', vmin=0, vmax=50)
        
        cbar = plt.colorbar(im, ax=ax, label='Freq % Diff')
        ax.set_xlabel('Azimuth (deg)')
        ax.set_ylabel('Inflow (m/s)')
        ax.set_title('Worst Percent Difference')
        
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_force_plots(self):
        """Create force component plots."""
        force_labels = ['Fx', 'Fy', 'Fz']
        
        for i, label in enumerate(force_labels):
            frame = ttk.Frame(self.plots_notebook)
            self.plots_notebook.add(frame, text=label)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            im = ax.imshow(self.results['total_global_force_vector'][:, :, i], 
                          extent=[self.viv_params.azimuths[0], self.viv_params.azimuths[-1], 
                                 self.viv_params.inflow_speeds[0], self.viv_params.inflow_speeds[-1]],
                          aspect='auto', origin='lower', cmap='viridis_r')
            
            cbar = plt.colorbar(im, ax=ax, label='Force (N)')
            ax.set_xlabel('Azimuth (deg)')
            ax.set_ylabel('Inflow (m/s)')
            ax.set_title(label)
            
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_moment_plots(self):
        """Create moment component plots."""
        moment_labels = ['Mx', 'My', 'Mz']
        
        for i, label in enumerate(moment_labels):
            frame = ttk.Frame(self.plots_notebook)
            self.plots_notebook.add(frame, text=label)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            im = ax.imshow(self.results['total_global_moment_vector'][:, :, i], 
                          extent=[self.viv_params.azimuths[0], self.viv_params.azimuths[-1], 
                                 self.viv_params.inflow_speeds[0], self.viv_params.inflow_speeds[-1]],
                          aspect='auto', origin='lower', cmap='viridis_r')
            
            cbar = plt.colorbar(im, ax=ax, label='Moment (N-m)')
            ax.set_xlabel('Azimuth (deg)')
            ax.set_ylabel('Inflow (m/s)')
            ax.set_title(label)
            
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def generate_structure_plot(self):
        """Generate and display the structure plot."""
        try:
            if not PLOTLY_AVAILABLE:
                messagebox.showerror("Error", "Plotly is not available. Please install plotly to use structure visualization.")
                return
            
            # Load components if not already loaded
            if self.components is None:
                if not os.path.exists(self.components_var.get()):
                    messagebox.showerror("Error", f"Components folder not found: {self.components_var.get()}")
                    return
                self.components = vorlap.load_components_from_csv(self.components_var.get())
            
            # Update airfoil folder in params
            self.viv_params.airfoil_folder = self.airfoil_var.get()
            
            # Generate the structure plot using vorlap function
            self.structure_fig = vorlap.graphics.calc_structure_vectors_andplot(self.components, self.viv_params)
            
            # Convert plotly figure to static image and display in tkinter
            self.display_structure_plot()
            
        except Exception as e:
            messagebox.showerror("Structure Plot Error", f"An error occurred: {str(e)}")
    
    def display_structure_plot(self):
        """Display the structure plot in the tkinter interface."""
        if self.structure_fig is None:
            return
        
        try:
            # Clear the current display
            for widget in self.structure_display_frame.winfo_children():
                widget.destroy()
            
            # Method 1: Export as HTML and display instructions
            # This is the most reliable method since it preserves interactivity
            html_file = os.path.join(os.path.dirname(__file__), "structure_plot.html")
            self.structure_fig.write_html(html_file)
            
            # Create a frame with instructions and image preview
            info_frame = ttk.Frame(self.structure_display_frame)
            info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Instructions
            instructions = f"""3D Structure Visualization Generated!

The interactive 3D plot has been saved as HTML: {html_file}

You can:
1. Open the HTML file in your web browser for full interactivity
2. View a static preview below (limited functionality)
3. Save the plot in various formats using the buttons above

The 3D plot shows:
- Component airfoil shapes and orientations
- Rotation axis (black line)
- Inflow direction (blue line)
- Chord lines and normal vectors for each component"""
            
            ttk.Label(info_frame, text=instructions, justify=tk.LEFT, font=("Arial", 10)).pack(pady=10)
            
            # Add button to open HTML in browser
            button_frame = ttk.Frame(info_frame)
            button_frame.pack(pady=10)
            
            def open_in_browser():
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(html_file)}")
            
            ttk.Button(button_frame, text="Open Interactive Plot in Browser", command=open_in_browser).pack()
            
            # Try to create a static image preview if PIL is available
            if PIL_AVAILABLE:
                try:
                    # Convert to static image
                    img_bytes = pio.to_image(self.structure_fig, format="png", width=600, height=400)
                    
                    # Load with PIL
                    from io import BytesIO
                    img = Image.open(BytesIO(img_bytes))
                    
                    # Convert to tkinter PhotoImage
                    photo = ImageTk.PhotoImage(img)
                    
                    # Display image
                    img_label = ttk.Label(info_frame, image=photo)
                    img_label.image = photo  # Keep a reference
                    img_label.pack(pady=10)
                    
                except Exception as e:
                    ttk.Label(info_frame, text=f"Could not generate image preview: {str(e)}", foreground="orange").pack()
            else:
                ttk.Label(info_frame, text="Install Pillow (PIL) for static image preview", foreground="gray").pack()
                
        except Exception as e:
            messagebox.showerror("Display Error", f"Error displaying structure plot: {str(e)}")
    
    def save_structure_plot(self):
        """Save the structure plot in various formats."""
        if self.structure_fig is None:
            messagebox.showwarning("Warning", "No structure plot to save. Generate a plot first.")
            return
        
        # Ask user for save location and format
        file_types = [
            ("HTML", "*.html"),
            ("PNG", "*.png"),
            ("SVG", "*.svg"),
            ("PDF", "*.pdf")
        ]
        
        file_path = filedialog.asksaveasfilename(
            title="Save Structure Plot",
            filetypes=file_types,
            defaultextension=".html"
        )
        
        if file_path:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".html":
                    self.structure_fig.write_html(file_path)
                elif ext == ".png":
                    pio.write_image(self.structure_fig, file_path, format="png")
                elif ext == ".svg":
                    pio.write_image(self.structure_fig, file_path, format="svg")
                elif ext == ".pdf":
                    pio.write_image(self.structure_fig, file_path, format="pdf")
                else:
                    messagebox.showerror("Error", f"Unsupported file format: {ext}")
                    return
                
                messagebox.showinfo("Success", f"Structure plot saved to: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Save Error", f"Error saving plot: {str(e)}")


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = VorLapGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main() 