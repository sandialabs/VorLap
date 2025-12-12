from vorlap.interpolation import resample_airfoil
from vorlap.structs import Component, VIV_Params
from vorlap.computations import rotationMatrix


import numpy as np
import plotly.graph_objects as go


import math
from typing import List


def calc_structure_vectors_andplot(components: List[Component], viv_params: VIV_Params, show_plot: bool = True, return_fig: bool = False, components_black: bool = False):
    """
    Calculates structure vectors and creates a plot.

    Args:
        components: List of structural components.
        viv_params: Configuration parameters.
        show_plot: Whether to display the plot (default: True).
        return_fig: Whether to return the figure object (default: False).

    Args:
        components_black: When True, draw component geometry in black instead of the default blue (default: False).

    Returns:
        fig: Plotly figure object if return_fig=True, otherwise None.
    """
    from .fileio import load_airfoil_coords

    # Create a new 3D figure
    fig = go.Figure()
    all_points = []

    # Draw rotation axis
    axis_len = max([(np.max(comp.shape_xyz) + np.max(comp.translation)) for comp in components])
    origin = viv_params.rotation_axis_offset
    arrow = viv_params.rotation_axis * axis_len + origin
    axis_vec = arrow - origin
    all_points.extend([origin, arrow])

    fig.add_trace(go.Scatter3d(
        x=[origin[0], arrow[0]],
        y=[origin[1], arrow[1]],
        z=[origin[2], arrow[2]],
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='Rotation Axis'
    ))
    fig.add_trace(go.Cone(
        x=[arrow[0]],
        y=[arrow[1]],
        z=[arrow[2]],
        u=[axis_vec[0]],
        v=[axis_vec[1]],
        w=[axis_vec[2]],
        anchor='tip',
        sizemode='absolute',
        sizeref=np.linalg.norm(axis_vec) * 0.01,
        showscale=False,
        colorscale=[[0, 'black'], [1, 'black']]
    ))

    # Draw inflow vector
    inflow_origin = np.array([-axis_len/1.5 * 0.75, 0.0, axis_len/2 * 1.25])
    inflow_arrow = viv_params.inflow_vec * axis_len * 0.25 + inflow_origin
    inflow_vec = inflow_arrow - inflow_origin
    all_points.extend([inflow_origin, inflow_arrow])

    fig.add_trace(go.Scatter3d(
        x=[inflow_origin[0], inflow_arrow[0]],
        y=[inflow_origin[1], inflow_arrow[1]],
        z=[inflow_origin[2], inflow_arrow[2]],
        mode='lines',
        line=dict(color='black', width=2),
        name='Inflow'
    ))
    fig.add_trace(go.Cone(
        x=[inflow_arrow[0]],
        y=[inflow_arrow[1]],
        z=[inflow_arrow[2]],
        u=[inflow_vec[0]],
        v=[inflow_vec[1]],
        w=[inflow_vec[2]],
        anchor='tip',
        sizemode='absolute',
        sizeref=np.linalg.norm(inflow_vec) * 0.07,
        showscale=False,
        colorscale=[[0, 'black'], [1, 'black']]
    ))

    for cidx, comp in enumerate(components):
        component_color = 'black' if components_black else 'rgba(0,0,0,0.15)'
        tangential_color = '#348ABD'
        normal_color = '#A60628'
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
        # Use Fortran-style (column-major) order to match Julia's reshape behavior
        af_cloud_local = af_coords_local.reshape(-1, af_coords_local.shape[2], order='F')
        chordline_cloud_local = chordline_local.reshape(-1, chordline_local.shape[2], order='F')
        normalline_cloud_local = normalline_local.reshape(-1, normalline_local.shape[2], order='F')

        euler = comp.rotation
        R_global = rotationMatrix(euler)

        af_coords_global = (R_global @ af_cloud_local.T).T
        af_coords_global[:, 0] += comp.translation[0]
        af_coords_global[:, 1] += comp.translation[1]
        af_coords_global[:, 2] += comp.translation[2]
        all_points.append(af_coords_global)

        chordline_global = (R_global @ chordline_cloud_local.T).T
        chordline_global[:, 0] += comp.translation[0]
        chordline_global[:, 1] += comp.translation[1]
        chordline_global[:, 2] += comp.translation[2]
        all_points.append(chordline_global)

        normalline_global = (R_global @ normalline_cloud_local.T).T
        normalline_global[:, 0] += comp.translation[0]
        normalline_global[:, 1] += comp.translation[1]
        normalline_global[:, 2] += comp.translation[2]
        all_points.append(normalline_global)

        # Add airfoil surface to the plot
        fig.add_trace(go.Scatter3d(
            x=af_coords_global[:, 0],
            y=af_coords_global[:, 1],
            z=af_coords_global[:, 2],
            mode='lines',
            line=dict(color=component_color),
            opacity=0.15,
            name=f'Component {comp.id}'
        ))

        # Add chord and normal lines
        halfIdx = int(chordline_global.shape[0] / 2)
        for idx in range(halfIdx):
            # Update component vectors
            comp.chord_vector[idx, :] = chordline_global[halfIdx + idx, :] - chordline_global[idx, :]
            comp.normal_vector[idx, :] = normalline_global[halfIdx + idx, :] - normalline_global[idx, :]
            comp.shape_xyz_global[idx, :] = chordline_global[idx, :]

            chord_start = chordline_global[idx, :]
            chord_end = chordline_global[halfIdx + idx, :]
            chord_vec = chord_end - chord_start
            chord_end_extended = chord_start + chord_vec * 1.3

            normal_start = normalline_global[idx, :]
            normal_end = normalline_global[halfIdx + idx, :]
            normal_vec = normal_end - normal_start
            normal_end_extended = normal_start + normal_vec * 1.3

            # Add chord line
            fig.add_trace(go.Scatter3d(
                x=[chord_start[0], chord_end_extended[0]],
                y=[chord_start[1], chord_end_extended[1]],
                z=[chord_start[2], chord_end_extended[2]],
                mode='lines',
                line=dict(color=tangential_color, width=6, dash='dash'),
                showlegend=False
            ))
            fig.add_trace(go.Cone(
                x=[chord_end_extended[0]],
                y=[chord_end_extended[1]],
                z=[chord_end_extended[2]],
                u=[chord_vec[0]],
                v=[chord_vec[1]],
                w=[chord_vec[2]],
                anchor='tip',
                sizemode='absolute',
                sizeref=np.linalg.norm(chord_vec) * (0.025*12),
                showscale=False,
                colorscale=[[0, tangential_color], [1, tangential_color]]
            ))

            # Add normal line
            fig.add_trace(go.Scatter3d(
                x=[normal_start[0], normal_end_extended[0]],
                y=[normal_start[1], normal_end_extended[1]],
                z=[normal_start[2], normal_end_extended[2]],
                mode='lines',
                line=dict(color=normal_color, width=4),
                showlegend=False
            ))
            fig.add_trace(go.Cone(
                x=[normal_end_extended[0]],
                y=[normal_end_extended[1]],
                z=[normal_end_extended[2]],
                u=[normal_vec[0]],
                v=[normal_vec[1]],
                w=[normal_vec[2]],
                anchor='tip',
                sizemode='absolute',
                sizeref=np.linalg.norm(normal_vec) * (0.025*12),
                showscale=False,
                colorscale=[[0, normal_color], [1, normal_color]]
            ))

    # Set layout with equal axis scaling
    if all_points:
        stacked_pts = np.vstack([np.atleast_2d(p) for p in all_points])
        mins = stacked_pts.min(axis=0)
        maxs = stacked_pts.max(axis=0)
        spans = maxs - mins
        max_span = np.max(spans)
        center = (mins + maxs) / 2
        padding = max_span * 0.05
        half_spanx = max_span / 2 + padding
        half_spany = max_span / 2 + padding
        half_spanz = max_span / 2 + padding
        axis_ranges = [
            (center[0] - half_spanx, center[0] + half_spanx),
            (center[1] - half_spany, center[1] + half_spany),
            (center[2] - half_spanz, center[2] + half_spanz),
        ]
    else:
        axis_ranges = [(-1, 1), (-1, 1), (-1, 1)]

    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(range=list(axis_ranges[0])),
            yaxis=dict(range=list(axis_ranges[1])),
            zaxis=dict(range=list(axis_ranges[2]))
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    # Remove background and grid
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )


    fig.update_layout(scene_camera=dict(eye=dict(x=1.1/2, y=-2.0/2, z=1.45/2)))

    # Save as transparent image (requires kaleido)
    save_path = "structure_plot_transparent.png"  # or .pdf, .svg
    fig.write_image(save_path, scale=4, width=1600, height=1200)
    print(f"Saved transparent 3D plot to {save_path}")


    # Display the figure if requested
    if show_plot:
        fig.show(renderer="browser")

    # Return the figure if requested
    if return_fig:
        return fig
    else:
        return None
