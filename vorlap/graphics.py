from vorlap.interpolation import resample_airfoil
from vorlap.structs import Component, VIV_Params
from vorlap.computations import rotationMatrix


import numpy as np
import plotly.graph_objects as go


import math
from typing import List


def calc_structure_vectors_andplot(components: List[Component], viv_params: VIV_Params, show_plot: bool = True, return_fig: bool = False):
    """
    Calculate structure vectors and create a 3D visualization plot.

    Args:
        components: List of structural components.
        viv_params: Configuration parameters.
        show_plot: Whether to display the plot (default: True).
        return_fig: Whether to return the figure object (default: False).

    Returns:
        Plotly figure object if return_fig=True, otherwise None.
    """
    from .fileio import load_airfoil_coords

    # Create a new 3D figure
    fig = go.Figure()

    # Draw rotation axis
    axis_len = max([np.max(comp.shape_xyz) for comp in components]) * 1.2
    origin = viv_params.rotation_axis_offset
    arrow = viv_params.rotation_axis * axis_len + origin

    fig.add_trace(go.Scatter3d(
        x=[origin[0], arrow[0]],
        y=[origin[1], arrow[1]],
        z=[origin[2], arrow[2]],
        mode='lines',
        line=dict(color='black', width=2),
        name='Rotation Axis'
    ))

    # Draw inflow vector
    inflow_origin = np.array([-axis_len/1.5, 0.0, axis_len/2])
    inflow_arrow = viv_params.inflow_vec * axis_len * 0.5 + inflow_origin

    fig.add_trace(go.Scatter3d(
        x=[inflow_origin[0], inflow_arrow[0]],
        y=[inflow_origin[1], inflow_arrow[1]],
        z=[inflow_origin[2], inflow_arrow[2]],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Inflow'
    ))

    absmin = min(min(inflow_origin), min(inflow_arrow))
    absmin = min(absmin, min(min(origin), min(arrow)))
    absmax = max(max(inflow_origin), max(inflow_arrow))
    absmax = max(absmax, max(max(origin), max(arrow)))

    for cidx, comp in enumerate(components):
        color = viv_params.plot_cycle[cidx % len(viv_params.plot_cycle)]
        N_Airfoils = comp.shape_xyz.shape[0]
        N_Af_coords = 200
        af_coords_local = np.zeros((N_Airfoils, N_Af_coords, 3))
        chordline_local = np.zeros((N_Airfoils, 2, 3))  # 2 start and stop, 3 xyz of each
        normalline_local = np.zeros((N_Airfoils, 2, 3))  # 2 start and stop, 3 xyz of each
        pitch = comp.pitch

        for ipt in range(comp.shape_xyz.shape[0]):
            pt = comp.shape_xyz[ipt, :]
            chord = comp.chord[ipt]
            twist = comp.twist[ipt] + pitch[0]
            thickness = comp.thickness[ipt]
            offset = comp.offset[ipt]

            # Each component's direction is defined by the comp.rotation rx, ry, rz angles, where 0,0,0 is pointing straight update
            # Let's create a local point cloud of xyz airfoil points, based on the shape input, then rotate that into position
            airfoil2d = load_airfoil_coords(f"{viv_params.airfoil_folder}{comp.airfoil_ids[ipt]}.csv")
            xy_scaled = resample_airfoil(airfoil2d, npoints=N_Af_coords)
            xy_scaled[:, 0] = xy_scaled[:, 0] * chord - chord * offset
            xy_scaled[:, 1] = xy_scaled[:, 1] * thickness * chord

            R_twist = np.array([
                [math.cos(math.radians(twist)), -math.sin(math.radians(twist))],
                [math.sin(math.radians(twist)), math.cos(math.radians(twist))]
            ])

            xy_scaled_twisted = (R_twist @ xy_scaled.T).T
            xy_scaled_twisted_translated = np.column_stack([
                xy_scaled_twisted[:, 0] + pt[0],
                xy_scaled_twisted[:, 1] + pt[1],
            ])

            af_coords_local[ipt, :, :] = np.column_stack([
                xy_scaled_twisted_translated[:, 0],
                xy_scaled_twisted_translated[:, 1],
                np.zeros(xy_scaled_twisted_translated.shape[0]) + pt[2]
            ])

            chordline_scaled_twisted = (R_twist @ np.array([[0, 0], [2*chord, 0]]).T).T
            chordline_scaled_twisted_translated = np.column_stack([
                chordline_scaled_twisted[:, 0] + pt[0],
                chordline_scaled_twisted[:, 1] + pt[1],
            ])

            chordline_local[ipt, :, :] = np.column_stack([
                chordline_scaled_twisted_translated[:, 0],
                chordline_scaled_twisted_translated[:, 1],
                np.zeros(chordline_scaled_twisted_translated.shape[0]) + pt[2]
            ])

            normalline_scaled_twisted = (R_twist @ np.array([[0, 0], [0, 2*chord]]).T).T

            # Calculate the local skew/sweep angle
            if ipt == 0:
                d_xyz = comp.shape_xyz[ipt+1, :] - comp.shape_xyz[ipt, :]
            elif ipt == comp.shape_xyz.shape[0] - 1:
                d_xyz = comp.shape_xyz[ipt, :] - comp.shape_xyz[ipt-1, :]
            else:
                d_xyz1 = comp.shape_xyz[ipt+1, :] - comp.shape_xyz[ipt, :]
                d_xyz2 = comp.shape_xyz[ipt, :] - comp.shape_xyz[ipt-1, :]
                d_xyz = (d_xyz1 + d_xyz2) / 2

            skew = math.atan2(d_xyz[2], d_xyz[1])
            R_skew = rotationMatrix(np.array([math.degrees(skew) - 90, 0.0, 0.0]))

            normalline_scaled_twisted3D = np.column_stack([
                normalline_scaled_twisted[:, 0],
                normalline_scaled_twisted[:, 1],
                np.zeros(normalline_scaled_twisted.shape[0])
            ])

            normalline_scaled_twisted_skewed = (R_skew @ normalline_scaled_twisted3D.T).T
            normalline_scaled_twisted_skewed_translated = np.column_stack([
                normalline_scaled_twisted_skewed[:, 0] + pt[0],
                normalline_scaled_twisted_skewed[:, 1] + pt[1],
                normalline_scaled_twisted_skewed[:, 2] + pt[2]
            ])

            normalline_local[ipt, :, :] = normalline_scaled_twisted_skewed_translated

        # Now that the local point cloud is generated, let's rotate and move it into position
        # Use Fortran-style (column-major) order
        af_cloud_local = af_coords_local.reshape(-1, af_coords_local.shape[2], order='F')
        chordline_cloud_local = chordline_local.reshape(-1, chordline_local.shape[2], order='F')
        normalline_cloud_local = normalline_local.reshape(-1, normalline_local.shape[2], order='F')

        euler = comp.rotation
        R_global = rotationMatrix(euler)

        af_coords_global = (R_global @ af_cloud_local.T).T
        af_coords_global[:, 0] += comp.translation[0]
        af_coords_global[:, 1] += comp.translation[1]
        af_coords_global[:, 2] += comp.translation[2]

        chordline_global = (R_global @ chordline_cloud_local.T).T
        chordline_global[:, 0] += comp.translation[0]
        chordline_global[:, 1] += comp.translation[1]
        chordline_global[:, 2] += comp.translation[2]

        normalline_global = (R_global @ normalline_cloud_local.T).T
        normalline_global[:, 0] += comp.translation[0]
        normalline_global[:, 1] += comp.translation[1]
        normalline_global[:, 2] += comp.translation[2]

        # Add airfoil surface to the plot
        fig.add_trace(go.Scatter3d(
            x=af_coords_global[:, 0],
            y=af_coords_global[:, 1],
            z=af_coords_global[:, 2],
            mode='lines',
            line=dict(color=color),
            name=f'Component {comp.id}'
        ))

        # Add chord and normal lines
        halfIdx = int(chordline_global.shape[0] / 2)
        for idx in range(halfIdx):
            # Update component vectors
            comp.chord_vector[idx, :] = chordline_global[halfIdx + idx, :] - chordline_global[idx, :]
            comp.normal_vector[idx, :] = normalline_global[halfIdx + idx, :] - normalline_global[idx, :]
            comp.shape_xyz_global[idx, :] = chordline_global[idx, :]

            # Add chord line
            fig.add_trace(go.Scatter3d(
                x=[chordline_global[idx, 0], chordline_global[halfIdx + idx, 0]],
                y=[chordline_global[idx, 1], chordline_global[halfIdx + idx, 1]],
                z=[chordline_global[idx, 2], chordline_global[halfIdx + idx, 2]],
                mode='lines',
                line=dict(color=color),
                showlegend=False
            ))

            # Add normal line
            fig.add_trace(go.Scatter3d(
                x=[normalline_global[idx, 0], normalline_global[halfIdx + idx, 0]],
                y=[normalline_global[idx, 1], normalline_global[halfIdx + idx, 1]],
                z=[normalline_global[idx, 2], normalline_global[halfIdx + idx, 2]],
                mode='lines',
                line=dict(color=color, dash='dash'),
                showlegend=False
            ))

    # Set layout with equal axis scaling
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(range=[absmin, absmax]),
            yaxis=dict(range=[absmin, absmax]),
            zaxis=dict(range=[absmin, absmax])
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=-2., z=1.5)))

    # Display the figure if requested
    if show_plot:
        try:
            fig.show(renderer="browser")
        except Exception:
            # Fallback for headless environments (like CI/CD)
            try:
                fig.show(renderer="png")
            except Exception:
                # If no renderers work, just skip showing
                pass

    # Return the figure if requested
    if return_fig:
        return fig
    else:
        return None