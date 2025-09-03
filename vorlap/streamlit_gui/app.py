#!/usr/bin/env python3
"""
Streamlit GUI for VorLap - Main Application.

This provides a web-based interface with the same functionality as the Tkinter GUI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import glob
import time
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add the vorlap package to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import vorlap

# Page configuration
st.set_page_config(
    page_title="VorLap - VORtex overLAP Tool",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">VORtex overLAP Tool</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Simulation Setup", "Components", "Plots & Outputs", "Analysis"]
    )
    
    # Initialize session state
    if 'components' not in st.session_state:
        st.session_state.components = []
    if 'natural_frequencies' not in st.session_state:
        st.session_state.natural_frequencies = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'components_dir' not in st.session_state:
        st.session_state.components_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "componentsHVAWT")
    if 'components_relative_path' not in st.session_state:
        st.session_state.components_relative_path = "vorlap/componentsHVAWT"
    if 'custom_airfoil_folder' not in st.session_state:
        st.session_state.custom_airfoil_folder = None
    if 'freq_file_path' not in st.session_state:
        st.session_state.freq_file_path = None
    if 'param_file_path' not in st.session_state:
        st.session_state.param_file_path = None
    if 'sim_name' not in st.session_state:
        st.session_state.sim_name = ""
    if 'sim_save_path' not in st.session_state:
        st.session_state.sim_save_path = ""
    if 'use_mock' not in st.session_state:
        st.session_state.use_mock = False
    if 'freq_input_text' not in st.session_state:
        st.session_state.freq_input_text = ""
    if 'plot_save_path' not in st.session_state:
        st.session_state.plot_save_path = ""
    if 'sample_x' not in st.session_state:
        st.session_state.sample_x = "0.0"
    if 'sample_y' not in st.session_state:
        st.session_state.sample_y = "0.0"
    if 'sample_export' not in st.session_state:
        st.session_state.sample_export = ""
    
    # Page routing
    if page == "Simulation Setup":
        simulation_setup_page()
    elif page == "Components":
        components_page()
    elif page == "Plots & Outputs":
        plots_outputs_page()
    elif page == "Analysis":
        analysis_page()

def simulation_setup_page():
    """Simulation Setup page."""
    st.markdown('<h2 class="section-header">Simulation Setup</h2>', unsafe_allow_html=True)
    
    # Top row: Sim name and save path
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        sim_name = st.text_input("Simulation Name/Path", 
                                value=st.session_state.sim_name,
                                placeholder="Enter simulation name",
                                key="sim_name_input")
        if sim_name != st.session_state.sim_name:
            st.session_state.sim_name = sim_name
    
    with col2:
        sim_save = st.text_input("Full Simulation Save Path", 
                                value=st.session_state.sim_save_path,
                                placeholder="Enter save directory path",
                                key="sim_save_input")
        if sim_save != st.session_state.sim_save_path:
            st.session_state.sim_save_path = sim_save
    
    with col3:
        use_mock = st.checkbox("Use Mock Data (Fast)", 
                              value=st.session_state.use_mock,
                              key="use_mock_checkbox")
        if use_mock != st.session_state.use_mock:
            st.session_state.use_mock = use_mock
    
    # Run button
    if st.button("Run & Save", type="primary", use_container_width=True):
        if not sim_name:
            st.error("Please enter a simulation name")
            return
        
        run_analysis(sim_name, sim_save, use_mock)
    
    st.markdown("---")
    
    # Two columns for frequencies and parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3>Parked Modal Frequencies</h3>', unsafe_allow_html=True)
        
        # File upload for frequencies
        freq_file = st.file_uploader("Upload Frequency CSV", type=['csv'], key="freq_upload")
        
        if freq_file is not None:
            try:
                # Read the file content as text first
                content = freq_file.read().decode('utf-8').strip()
                
                # Parse comma-separated values
                if content:
                    # Split by comma and convert to float, handling potential whitespace
                    freq_values = [float(f.strip()) for f in content.split(',') if f.strip()]
                    
                    if freq_values:
                        st.session_state.natural_frequencies = np.array(freq_values)
                        st.session_state.freq_input_text = '\n'.join(map(str, freq_values))
                        st.success(f"Loaded {len(freq_values)} frequencies from CSV")
                    else:
                        st.error("No valid frequency values found in the CSV")
                else:
                    st.error("CSV file is empty")
                    
            except ValueError as e:
                st.error(f"Error parsing frequency values: {str(e)}")
                st.info("Expected format: comma-separated numbers (e.g., 3.00,5.00,23.0,152.0)")
            except Exception as e:
                st.error(f"Error loading frequencies: {str(e)}")
                st.info("Expected format: comma-separated numbers (e.g., 3.00,5.00,23.0,152.0)")
        
        # Manual frequency input
        st.markdown("**Or enter frequencies manually:**")
        
        freq_input = st.text_area("Frequencies (Hz) - one per line", 
                                 value=st.session_state.get('freq_input_text', ''),
                                 height=100,
                                 key="freq_input_textarea",
                                 help="Enter frequencies, one per line, or comma-separated on one line")
        
        if freq_input:
            st.session_state.freq_input_text = freq_input
            try:
                # Handle both line-by-line and comma-separated input
                if ',' in freq_input:
                    # Comma-separated input
                    freqs = [float(f.strip()) for f in freq_input.split(',') if f.strip()]
                else:
                    # Line-by-line input
                    freqs = [float(f.strip()) for f in freq_input.split('\n') if f.strip()]
                
                if freqs:
                    st.session_state.natural_frequencies = np.array(freqs)
                    st.success(f"Set {len(freqs)} frequencies from manual input")
                else:
                    st.error("No valid frequency values entered")
            except ValueError:
                st.error("Please enter valid numbers, one per line or comma-separated")
        
        # Load default frequencies button
        default_freq_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "natural_frequencies.csv")
        if os.path.exists(default_freq_file):
            if st.button("Load Default Frequencies", key="load_default_freqs"):
                try:
                    with open(default_freq_file, 'r') as f:
                        content = f.read().strip()
                    
                    if content:
                        freq_values = [float(f.strip()) for f in content.split(',') if f.strip()]
                        if freq_values:
                            st.session_state.natural_frequencies = np.array(freq_values)
                            st.session_state.freq_input_text = '\n'.join(map(str, freq_values))
                            st.success(f"Loaded {len(freq_values)} default frequencies")
                        else:
                            st.error("Default frequency file contains no valid values")
                    else:
                        st.error("Default frequency file is empty")
                except Exception as e:
                    st.error(f"Error loading default frequencies: {str(e)}")
        else:
            st.info(f"Default frequency file not found: {default_freq_file}")
        
        # Single display of current frequencies (only shown if frequencies exist)
        if st.session_state.natural_frequencies is not None:
            st.markdown("---")
            st.markdown("**Current Frequencies:**")
            current_freqs = st.session_state.natural_frequencies
            st.write(f"{', '.join([f'{f:.2f}' for f in current_freqs])} Hz")
            st.info(f"Total: {len(current_freqs)} frequencies loaded")
    
    with col2:
        st.markdown('<h3>Simulation Parameters</h3>', unsafe_allow_html=True)
        
        # File upload for parameters
        param_file = st.file_uploader("Upload Parameters CSV", type=['csv'], key="param_upload")
        
        if param_file is not None:
            try:
                param_df = pd.read_csv(param_file)
                st.session_state.parameters = param_df
                st.success(f"Loaded {len(param_df)} parameters")
                st.dataframe(param_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading parameters: {str(e)}")
        
        # Default parameters
        if 'parameters' not in st.session_state:
            st.session_state.parameters = get_default_parameters()
        
        # Editable parameters table
        st.markdown("**Edit Parameters:**")
        edited_params = st.data_editor(
            st.session_state.parameters,
            use_container_width=True,
            num_rows="dynamic"
        )
        
        if edited_params is not None:
            st.session_state.parameters = edited_params
        
        # Airfoil folder configuration
        st.markdown('<h3>Airfoil Configuration</h3>', unsafe_allow_html=True)
        
        # Show current airfoil folder path
        current_airfoil_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "airfoils")
        st.markdown(f"**Current airfoil folder:** `{current_airfoil_folder}`")
        
        # Check if airfoil folder exists and show contents
        if os.path.exists(current_airfoil_folder):
            try:
                airfoil_files = [f for f in os.listdir(current_airfoil_folder) if f.endswith('.h5')]
                if airfoil_files:
                    st.success(f"Found {len(airfoil_files)} airfoil files")
                    st.markdown("**Available airfoil files:**")
                    for file in airfoil_files[:10]:  # Show first 10 files
                        st.text(f"  • {file}")
                    if len(airfoil_files) > 10:
                        st.text(f"  ... and {len(airfoil_files) - 10} more")
                else:
                    st.warning("No .h5 files found in airfoil folder")
                    # Show all files for debugging
                    all_files = os.listdir(current_airfoil_folder)
                    if all_files:
                        st.markdown("**All files in airfoil folder:**")
                        for file in all_files[:10]:
                            st.text(f"  • {file}")
                        if len(all_files) > 10:
                            st.text(f"  ... and {len(all_files) - 10} more")
            except Exception as e:
                st.error(f"Error reading airfoil folder: {str(e)}")
        else:
            st.error(f"Airfoil folder does not exist: {current_airfoil_folder}")
            st.markdown("**Tip:** Create the airfoil folder and add .h5 files, or update the path in the code.")
        
        # Custom airfoil folder path
        custom_airfoil_path = st.text_input("Custom airfoil folder path (optional):", 
                                          placeholder="Enter custom path to airfoil folder")
        
        if custom_airfoil_path and os.path.exists(custom_airfoil_path):
            st.session_state.custom_airfoil_folder = custom_airfoil_path
            st.success(f"Custom airfoil folder set to: {custom_airfoil_path}")
        elif custom_airfoil_path:
            st.warning(f"Path does not exist: {custom_airfoil_path}")
        
        # Components section
        st.markdown('<h3>Components Configuration</h3>', unsafe_allow_html=True)
        
        # Component selection and loading
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            comp_num = st.number_input("Component", min_value=1, max_value=999, value=1, step=1)
        
        with col2:
            # Folder selection with multiple options
            st.markdown("**Components Directory:**")
            
            # Option 1: Manual path input
            st.markdown("**Current working directory:**")
            cwd = os.getcwd()
            st.code(cwd, language="bash")
            
            manual_path = st.text_input("Enter path manually:", 
                                       value=st.session_state.get('components_dir', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "componentsHVAWT")),
                                       placeholder="Enter full path to components directory",
                                       key="components_manual_path")
            
            # Relative path option
            relative_path = st.text_input("Or enter relative path from current directory:", 
                                        value=st.session_state.get('components_relative_path', "vorlap/componentsHVAWT"),
                                        placeholder="e.g., vorlap/componentsHVAWT",
                                        key="components_relative_path")
            
            if relative_path:
                full_relative_path = os.path.abspath(relative_path)
                if os.path.exists(full_relative_path):
                    st.session_state.components_dir = full_relative_path
                    st.success(f"Relative path resolved to: {full_relative_path}")
                else:
                    st.warning(f"Relative path does not exist: {full_relative_path}")
            
            # Option 2: Quick folder selection
            st.markdown("**Or choose from common locations:**")
            common_folders = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "componentsHVAWT"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "componentsHAWT"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "componentsVAWT"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "components")
            ]
            
            folder_names = ["HVAWT Components", "HAWT Components", "VAWT Components", "Generic Components"]
            selected_folder_idx = st.selectbox("Quick select:", range(len(common_folders)), 
                                             format_func=lambda x: folder_names[x],
                                             key="components_quick_select")
            
            if selected_folder_idx is not None:
                selected_path = common_folders[selected_folder_idx]
                if os.path.exists(selected_path):
                    st.session_state.components_dir = selected_path
                    st.success(f"Selected: {folder_names[selected_folder_idx]} ({selected_path})")
                else:
                    st.warning(f"Path does not exist: {selected_path}")
            
            # Use the selected folder path
            components_dir = st.session_state.get('components_dir', manual_path)
            
            # Display current selected folder
            if components_dir:
                st.markdown(f"**Current folder:** `{components_dir}`")
                if os.path.exists(components_dir):
                    st.success("Folder exists and is accessible")
                    # Show folder contents
                    try:
                        files = os.listdir(components_dir)
                        csv_files = [f for f in files if f.endswith('.csv')]
                        if csv_files:
                            st.markdown(f"**Found {len(csv_files)} CSV files:**")
                            for file in csv_files[:5]:  # Show first 5 files
                                st.text(f"  • {file}")
                            if len(csv_files) > 5:
                                st.text(f"  ... and {len(csv_files) - 5} more")
                        else:
                            st.warning("No CSV files found in this directory")
                    except PermissionError:
                        st.error("Permission denied accessing this folder")
                    except Exception as e:
                        st.warning(f"Could not read folder contents: {str(e)}")
                else:
                    st.error("Folder does not exist")
        
        with col3:
            if st.button("Load Components", key="load_components_btn", use_container_width=True):
                if components_dir and os.path.exists(components_dir):
                    load_components(components_dir)
                else:
                    st.error("Please select a valid components directory first")
        
        # Show loaded components info
        if st.session_state.components:
            st.markdown('<h4>Loaded Components</h4>', unsafe_allow_html=True)
            st.success(f"Loaded {len(st.session_state.components)} components")
            
            # Show component details
            for i, comp in enumerate(st.session_state.components):
                with st.expander(f"Component {i+1} Details"):
                    st.write(f"**Shape XYZ points:** {len(comp.shape_xyz)}")
                    st.write(f"**Chord values:** {len(comp.chord)}")
                    st.write(f"**Twist values:** {len(comp.twist)}")
                    st.write(f"**Thickness values:** {len(comp.thickness)}")
                    st.write(f"**Airfoil IDs:** {len(comp.airfoil_ids)}")
        else:
            st.info("No components loaded. Use the Load Components button to load component data.")

def components_page():
    """Components page - Simplified view with geometry table."""
    st.markdown('<h2 class="section-header">Components</h2>', unsafe_allow_html=True)
    
    # Redirect message
    st.info("Tip: Component configuration has been moved to the Simulation Setup tab for easier workflow.")
    st.markdown("Use the Simulation Setup tab to load and configure components, then come back here to view the geometry table.")
    
    # Component geometry table
    if st.session_state.components:
        st.markdown('<h3>Component Geometric Definition</h3>', unsafe_allow_html=True)
        
        # Create geometry dataframe
        geom_data = []
        comp = st.session_state.components[0]  # Use first component
        
        for i in range(len(comp.shape_xyz)):
            pitch_val = comp.pitch[0] if len(comp.pitch) > 0 else 0
            chord_val = comp.chord[i] if i < len(comp.chord) else 0
            twist_val = comp.twist[i] if i < len(comp.twist) else 0
            thickness_val = comp.thickness[i] if i < len(comp.thickness) else 0.18
            offset_val = comp.offset[i] if i < len(comp.offset) else 0
            airfoil_id = comp.airfoil_ids[i] if i < len(comp.airfoil_ids) else "default"
            
            geom_data.append({
                "Name": f"Node_{i}",
                "Dx": comp.shape_xyz[i, 0],
                "Dy": comp.shape_xyz[i, 1],
                "Dz": comp.shape_xyz[i, 2],
                "Rx": 0,
                "Ry": 0,
                "Rz": 0,
                "Pitch": pitch_val,
                "X": comp.shape_xyz[i, 0],
                "Y": comp.shape_xyz[i, 1],
                "Z": comp.shape_xyz[i, 2],
                "Chord": chord_val,
                "twist": twist_val,
                "thk%": thickness_val,
                "offset": offset_val,
                "AirfoilPath": airfoil_id
            })
        
        geom_df = pd.DataFrame(geom_data)
        st.dataframe(geom_df, use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Geometry CSV"):
                save_geometry_csv(geom_df)
        
        with col2:
            if st.button("📤 Load Geometry CSV"):
                uploaded_file = st.file_uploader("Upload geometry CSV", type=['csv'])
                if uploaded_file is not None:
                    try:
                        loaded_df = pd.read_csv(uploaded_file)
                        st.success("Geometry loaded successfully")
                        st.dataframe(loaded_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading geometry: {str(e)}")
    else:
        st.info("No components loaded. Use the Simulation Setup tab to load components first.")

def plots_outputs_page():
    """Plots & Outputs page."""
    st.markdown('<h2 class="section-header">Plots & Outputs</h2>', unsafe_allow_html=True)
    
    # Top controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        plot_save_path = st.text_input("Save Plots Path", 
                                      value=st.session_state.get('plot_save_path', ''),
                                      placeholder="Enter directory for plots",
                                      key="plot_save_path_input")
        if plot_save_path != st.session_state.get('plot_save_path', ''):
            st.session_state.plot_save_path = plot_save_path
    
    with col2:
        if st.button("💾 Save Plot", use_container_width=True):
            if plot_save_path:
                st.success(f"Plot would be saved to: {plot_save_path}")
            else:
                st.warning("Please enter a save path")
    
    # Plot area
    if st.session_state.analysis_results:
        st.markdown('<h3>Analysis Results</h3>', unsafe_allow_html=True)
        
        # Plot type selection
        plot_type = st.selectbox(
            "Plot Type:",
            ["Frequency Overlap", "Force X", "Force Y", "Force Z", "Moment X", "Moment Y", "Moment Z"],
            format_func=lambda x: x
        )
        
        # Create plot
        fig = create_plot(plot_type, st.session_state.analysis_results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Save plot button
        if st.button("Save Current Plot"):
            if plot_save_path:
                save_plot(fig, plot_type, plot_save_path)
            else:
                st.warning("Please enter a save path")
    else:
        st.info("Run analysis first to see results")
    
    # Sampling controls
    st.markdown('<h3>Sampling & Text Output</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        sample_x = st.text_input("Sample X", 
                                value=st.session_state.get('sample_x', "0.0"),
                                key="sample_x_input")
        if sample_x != st.session_state.get('sample_x', "0.0"):
            st.session_state.sample_x = sample_x
    
    with col2:
        sample_y = st.text_input("Sample Y", 
                                value=st.session_state.get('sample_y', "0.0"),
                                key="sample_y_input")
        if sample_y != st.session_state.get('sample_y', "0.0"):
            st.session_state.sample_y = sample_y
    
    with col3:
        sample_export = st.text_input("Sampled Signal Export Path", 
                                    value=st.session_state.get('sample_export', ''),
                                    placeholder="Save sampled signal as CSV",
                                    key="sample_export_input")
        if sample_export != st.session_state.get('sample_export', ''):
            st.session_state.sample_export = sample_export
    
    if st.button("📤 Export Sample"):
        if sample_export:
            export_sample(sample_x, sample_y, sample_export)
        else:
            st.warning("Please enter an export path")
    
    # Console output
    st.markdown('<h3>Console Output</h3>', unsafe_allow_html=True)
    if 'console_log' in st.session_state:
        st.text_area("Log Output", value=st.session_state.console_log, height=200, disabled=True)
    else:
        st.text_area("Log Output", value="No output yet", height=200, disabled=True)

def analysis_page():
    """Analysis page."""
    st.markdown('<h2 class="section-header">Analysis Tools</h2>', unsafe_allow_html=True)
    
    # Visualization buttons
    st.markdown('<h3>Visualization Type</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Geometry"):
            show_geometry()
    
    with col2:
        if st.button("Thrust"):
            show_thrust()
    
    with col3:
        if st.button("Torque"):
            show_torque()
    
    with col4:
        if st.button("Frequency Overlap"):
            show_frequency_overlap()
    
    st.markdown("---")
    
    # Analysis mode selection
    st.markdown('<h3>Analysis Mode</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Mode 1"):
            select_mode(1)
    
    with col2:
        if st.button("Mode 2"):
            select_mode(2)
    
    with col3:
        if st.button("Mode 3"):
            select_mode(3)

# Helper functions
def get_default_parameters():
    """Get default VIV_Params values."""
    return pd.DataFrame([
        ("fluid_density", "1.225"),
        ("fluid_dynamicviscosity", "1.81e-5"),
        ("rotation_axis_x", "0.0"),
        ("rotation_axis_y", "0.0"),
        ("rotation_axis_z", "1.0"),
        ("rotation_axis_offset_x", "0.0"),
        ("rotation_axis_offset_y", "0.0"),
        ("rotation_axis_offset_z", "0.0"),
        ("inflow_vec_x", "1.0"),
        ("inflow_vec_y", "0.0"),
        ("inflow_vec_z", "0.0"),
        ("azimuth_start", "0"),
        ("azimuth_end", "255"),
        ("azimuth_step", "5"),
        ("inflow_speed_start", "2.0"),
        ("inflow_speed_end", "50.0"),
        ("inflow_speed_step", "0.5"),
        ("n_harmonic", "2"),
        ("output_time_start", "0.0"),
        ("output_time_end", "0.01"),
        ("output_time_step", "0.001"),
        ("output_azimuth", "5.0"),
        ("output_vinf", "2.0"),
        ("amplitude_coeff_cutoff", "0.2"),
        ("n_freq_depth", "10"),
    ], columns=["Parameter", "Value"])

def load_components(components_dir):
    """Load components from the specified directory."""
    try:
        components = vorlap.load_components_from_csv(components_dir)
        st.session_state.components = components
        st.success(f"Loaded {len(components)} components from: {components_dir}")
        log_message(f"Loaded {len(components)} components from: {components_dir}")
    except Exception as e:
        st.error(f"Error loading components: {str(e)}")
        log_message(f"Error loading components: {str(e)}")

def run_analysis(sim_name, sim_save, use_mock):
    """Run the complete VorLap analysis."""
    try:
        log_message("Starting VorLap analysis...")
        log_message(f"  Simulation name: {sim_name}")
        log_message(f"  Save directory: {sim_save}")
        
        # Get parameters
        viv_params = get_viv_params()
        
        # Get natural frequencies
        natfreqs = st.session_state.natural_frequencies
        if natfreqs is None:
            st.error("No natural frequencies loaded")
            log_message("Error: No natural frequencies loaded")
            return
        
        # Get components
        components = st.session_state.components
        if not components:
            st.error("No components loaded")
            log_message("Error: No components loaded")
            return
        
        # Load airfoils
        if 'custom_airfoil_folder' in st.session_state and st.session_state.custom_airfoil_folder:
            airfoil_folder = st.session_state.custom_airfoil_folder
            log_message(f"Using custom airfoil folder: {airfoil_folder}")
        else:
            airfoil_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "airfoils")
            log_message(f"Using default airfoil folder: {airfoil_folder}")
        
        affts = load_airfoils(airfoil_folder)
        if not affts:
            st.error("No airfoil data loaded")
            log_message("Error: No airfoil data loaded")
            return
        
        # Run computation
        if use_mock:
            log_message("Running MOCK thrust/torque spectrum computation (fast)...")
            start_time = time.time()
            
            percdiff_matrix, percdiff_info, total_global_force_vector, total_global_moment_vector, global_force_vector_nodes = vorlap.mock_compute_thrust_torque_spectrum(
                components, affts, viv_params, natfreqs
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            log_message(f"Mock computation completed in {execution_time:.4f} seconds")
        else:
            log_message("Running thrust/torque spectrum computation...")
            start_time = time.time()
            
            percdiff_matrix, percdiff_info, total_global_force_vector, total_global_moment_vector, global_force_vector_nodes = vorlap.compute_thrust_torque_spectrum(
                components, affts, viv_params, natfreqs
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            log_message(f"Computation completed in {execution_time:.4f} seconds")
        
        # Store results
        st.session_state.analysis_results = {
            'percdiff_matrix': percdiff_matrix,
            'percdiff_info': percdiff_info,
            'total_global_force_vector': total_global_force_vector,
            'total_global_moment_vector': total_global_moment_vector,
            'global_force_vector_nodes': global_force_vector_nodes,
            'viv_params': viv_params
        }
        
        # Save force time series if save path is provided
        if sim_save:
            save_dir = Path(sim_save)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            force_file = save_dir / "forces_output.csv"
            vorlap.write_force_time_series(str(force_file), viv_params.output_time, global_force_vector_nodes)
            log_message(f"Force time series saved to: {force_file}")
        
        st.success("Analysis completed successfully!")
        log_message("Analysis completed successfully!")
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        log_message(f"Error during analysis: {str(e)}")

def get_viv_params():
    """Create VIV_Params object from the current parameter table."""
    params = {}
    for _, row in st.session_state.parameters.iterrows():
        key, value = row["Parameter"], row["Value"]
        if value.strip():
            try:
                float_val = float(value)
                if float_val == int(float_val):
                    params[key] = int(float_val)
                else:
                    params[key] = float_val
            except ValueError:
                params[key] = value
    
    # Create arrays from start/end/step parameters
    azimuths = np.arange(
        params.get("azimuth_start", 0),
        params.get("azimuth_end", 255) + params.get("azimuth_step", 5),
        params.get("azimuth_step", 5)
    )
    
    inflow_speeds = np.arange(
        params.get("inflow_speed_start", 2.0),
        params.get("inflow_speed_end", 50.0) + params.get("inflow_speed_step", 0.5),
        params.get("inflow_speed_step", 0.5)
    )
    
    output_time = np.arange(
        params.get("output_time_start", 0.0),
        params.get("output_time_end", 0.01) + params.get("output_time_step", 0.001),
        params.get("output_time_step", 0.001)
    )
    
    # Set airfoil folder
    if 'custom_airfoil_folder' in st.session_state and st.session_state.custom_airfoil_folder:
        airfoil_folder = st.session_state.custom_airfoil_folder
    else:
        airfoil_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "airfoils")
    
    return vorlap.VIV_Params(
        fluid_density=params.get("fluid_density", 1.225),
        fluid_dynamicviscosity=params.get("fluid_dynamicviscosity", 1.81e-5),
        rotation_axis=np.array([
            params.get("rotation_axis_x", 0.0),
            params.get("rotation_axis_y", 0.0),
            params.get("rotation_axis_z", 1.0)
        ]),
        rotation_axis_offset=np.array([
            params.get("rotation_axis_offset_x", 0.0),
            params.get("rotation_axis_offset_y", 0.0),
            params.get("rotation_axis_offset_z", 0.0)
        ]),
        inflow_vec=np.array([
            params.get("inflow_vec_x", 1.0),
            params.get("inflow_vec_y", 0.0),
            params.get("inflow_vec_z", 0.0)
        ]),
        azimuths=azimuths,
        inflow_speeds=inflow_speeds,
        n_harmonic=params.get("n_harmonic", 2),
        output_time=output_time,
        output_azimuth_vinf=(
            params.get("output_azimuth", 5.0),
            params.get("output_vinf", 2.0)
        ),
        amplitude_coeff_cutoff=params.get("amplitude_coeff_cutoff", 0.2),
        n_freq_depth=params.get("n_freq_depth", 10),
        airfoil_folder=airfoil_folder
    )

def load_airfoils(airfoil_folder):
    """Load airfoil FFT data from the specified folder."""
    try:
        log_message(f"Attempting to load airfoils from: {airfoil_folder}")
        
        # Check if folder exists
        if not os.path.exists(airfoil_folder):
            log_message(f"Error: Airfoil folder does not exist: {airfoil_folder}")
            return {}
        
        # Check if folder is accessible
        if not os.access(airfoil_folder, os.R_OK):
            log_message(f"Error: No read access to airfoil folder: {airfoil_folder}")
            return {}
        
        # List all files in the folder
        all_files = os.listdir(airfoil_folder)
        log_message(f"Found {len(all_files)} files in airfoil folder")
        
        # Look for .h5 files
        h5_files = [f for f in all_files if f.endswith('.h5')]
        log_message(f"Found {len(h5_files)} .h5 files: {h5_files}")
        
        if not h5_files:
            log_message(f"Warning: No .h5 files found in {airfoil_folder}")
            # Try to find other airfoil-related files
            other_files = [f for f in all_files if 'airfoil' in f.lower() or 'fft' in f.lower()]
            if other_files:
                log_message(f"Found other airfoil-related files: {other_files}")
        
        affts = {}
        for file in h5_files:
            try:
                file_path = os.path.join(airfoil_folder, file)
                log_message(f"Loading airfoil file: {file}")
                afft = vorlap.load_airfoil_fft(file_path)
                affts[afft.name] = afft
                log_message(f"Successfully loaded: {afft.name}")
            except Exception as file_error:
                log_message(f"Error loading file {file}: {str(file_error)}")
        
        # Ensure default airfoil exists
        if "default" not in affts and affts:
            affts["default"] = next(iter(affts.values()))
            log_message(f"Set default airfoil to: {affts['default'].name}")
        
        log_message(f"Successfully loaded {len(affts)} airfoil files from {airfoil_folder}")
        return affts
        
    except Exception as e:
        log_message(f"Error loading airfoils: {str(e)}")
        return {}

def create_plot(plot_type, results):
    """Create a plot based on the selected type and results."""
    viv_params = results['viv_params']
    extent = [viv_params.azimuths[0], viv_params.azimuths[-1], 
             viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]]
    
    if plot_type == "Frequency Overlap":
        data = results['percdiff_matrix']
        title = 'Worst Percent Difference'
        label = 'Freq % Diff'
        colorscale = 'viridis_r'
        zmin, zmax = 0, 50
    elif plot_type.startswith("Force"):
        force_data = results['total_global_force_vector']
        idx = {'Force X': 0, 'Force Y': 1, 'Force Z': 2}[plot_type]
        data = force_data[:, :, idx]
        title = f'{plot_type}'
        label = 'Force (N)'
        colorscale = 'viridis_r'
        zmin, zmax = None, None
    elif plot_type.startswith("Moment"):
        moment_data = results['total_global_moment_vector']
        idx = {'Moment X': 0, 'Moment Y': 1, 'Moment Z': 2}[plot_type]
        data = moment_data[:, :, idx]
        title = f'{plot_type}'
        label = 'Moment (N-m)'
        colorscale = 'viridis_r'
        zmin, zmax = None, None
    else:
        return go.Figure()
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=viv_params.azimuths,
        y=viv_params.inflow_speeds,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title=label)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Azimuth (deg)',
        yaxis_title='Inflow (m/s)',
        width=800,
        height=600
    )
    
    return fig

def save_geometry_csv(df):
    """Save geometry data to CSV."""
    try:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Geometry CSV",
            data=csv,
            file_name="component_geometry.csv",
            mime="text/csv"
        )
        st.success("Geometry CSV ready for download")
    except Exception as e:
        st.error(f"Error saving geometry: {str(e)}")

def save_plot(fig, plot_type, save_path):
    """Save plot to file."""
    try:
        # Convert plot type to filename-safe string
        safe_name = plot_type.lower().replace(" ", "_")
        filename = f"{safe_name}_plot.html"
        filepath = os.path.join(save_path, filename)
        
        fig.write_html(filepath)
        st.success(f"Plot saved to: {filepath}")
    except Exception as e:
        st.error(f"Error saving plot: {str(e)}")

def export_sample(x, y, export_path):
    """Export sample data to CSV."""
    try:
        df = pd.DataFrame({"x": [float(x)], "y": [float(y)]})
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="sample_data.csv",
            mime="text/csv"
        )
        st.success(f"Sample exported (x={x}, y={y})")
    except Exception as e:
        st.error(f"Error exporting sample: {str(e)}")

def show_geometry():
    """Show structure geometry visualization."""
    if not st.session_state.components:
        st.warning("Load components first.")
        return
    
    try:
        viv_params = get_viv_params()
        vorlap.graphics.calc_structure_vectors_andplot(st.session_state.components, viv_params)
        log_message("[Analysis] Geometry visualization displayed")
        st.success("Geometry visualization displayed")
    except Exception as e:
        log_message(f"[Analysis] Geometry error: {str(e)}")
        st.error(f"Geometry error: {str(e)}")

def show_thrust():
    """Show thrust analysis."""
    if st.session_state.analysis_results:
        st.session_state.current_plot = "Force X"
        st.success("Thrust visualization updated")
        log_message("[Analysis] Thrust visualization updated")
    else:
        st.warning("Run analysis first.")

def show_torque():
    """Show torque analysis."""
    if st.session_state.analysis_results:
        st.session_state.current_plot = "Moment Z"
        st.success("Torque visualization updated")
        log_message("[Analysis] Torque visualization updated")
    else:
        st.warning("Run analysis first.")

def show_frequency_overlap():
    """Show frequency overlap analysis."""
    if st.session_state.analysis_results:
        st.session_state.current_plot = "Frequency Overlap"
        st.success("Frequency overlap visualization updated")
        log_message("[Analysis] Frequency overlap visualization updated")
    else:
        st.warning("Run analysis first.")

def select_mode(n):
    """Select analysis mode."""
    log_message(f"[Analysis] Mode set to {n}")
    st.success(f"Mode set to {n}")

def log_message(message):
    """Add message to console log."""
    if 'console_log' not in st.session_state:
        st.session_state.console_log = ""
    
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.console_log += f"[{timestamp}] {message}\n"

if __name__ == "__main__":
    main() 