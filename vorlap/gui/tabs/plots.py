#!/usr/bin/env python3
"""
Plots & Outputs Tab for the VorLap GUI.

This tab handles visualization, plotting, and data export functionality.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib.colors as mcolors

import vorlap

from ..widgets import ScrollableFrame, PathEntry

# --- Optional plotting support ---
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False


class PlotsOutputsTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.scrollable = ScrollableFrame(self)
        self.scrollable.pack(fill="both", expand=True)
        self.content = self.scrollable.content
        self._colorbar = None
        self.toolbar = None
        self._last_plot_save_path = ""
        self._build()

    def _build(self):
        container = self.content

        top_actions = ttk.LabelFrame(container, text="Run and Outputs")
        top_actions.grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 2))
        top_actions.columnconfigure(1, weight=1)
        top_actions.columnconfigure(3, weight=1)

        ttk.Button(top_actions, text="Run and Plot", command=self.run_and_plot).grid(
            row=0, column=0, sticky="w", padx=(6, 8), pady=(6, 4)
        )
        ttk.Label(top_actions, text="Run analysis then switch plot types below.").grid(
            row=0, column=1, columnspan=3, sticky="w", pady=(6, 4)
        )

        ttk.Label(top_actions, text="Output Directory").grid(row=1, column=0, sticky="w", padx=(6, 4))
        self.output_dir_entry = PathEntry(
            top_actions,
            kind="dir",
            title="Choose output directory",
            must_exist=True,
            textvariable=self.app.tab_setup.sim_save_var,
        )
        self.output_dir_entry.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(0, 4))

        ttk.Label(top_actions, text="QBlade Loading Output (optional)").grid(row=1, column=2, sticky="w", padx=(0, 4))
        self.qblade_loading_entry = PathEntry(
            top_actions,
            kind="savefile",
            title="Choose QBlade loading output file",
            must_exist=False,
            textvariable=self.app.tab_setup.qblade_loading_path_var,
        )
        self.qblade_loading_entry.grid(row=1, column=3, sticky="ew", padx=(0, 6), pady=(0, 4))

        self.plot_frame = ttk.LabelFrame(container, text="Analysis Results")
        self.plot_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=6)
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(1, weight=1)

        controls = ttk.Frame(self.plot_frame)
        controls.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ttk.Label(controls, text="Plot Type:").grid(row=0, column=0, sticky="w")

        self.plot_type = tk.StringVar(value="geometry")
        plot_types = [
            ("Geometry", "geometry"),
            ("Frequency Overlap", "percdiff"),
            ("Force X", "fx"),
            ("Force Y", "fy"),
            ("Force Z", "fz"),
            ("Moment X", "mx"),
            ("Moment Y", "my"),
            ("Moment Z", "mz"),
        ]
        for i, (text, value) in enumerate(plot_types):
            ttk.Radiobutton(
                controls,
                text=text,
                variable=self.plot_type,
                value=value,
                command=self._on_plot_type_changed,
            ).grid(row=0, column=i + 1, padx=4)

        self.fancy_geometry_btn = ttk.Button(
            controls,
            text="Fancy Geometry",
            command=self.open_fancy_geometry,
        )
        self.fancy_geometry_btn.grid(row=1, column=1, pady=(4, 0), sticky="w")
        ttk.Button(controls, text="Save Plot", command=self.save_plot).grid(row=0, column=len(plot_types) + 1, padx=5)

        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("VorLap Analysis Results")
        self.ax.text(0.5, 0.5, "Run analysis to see results", ha="center", va="center", transform=self.ax.transAxes)
        self._apply_axis_theme(self.ax)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        self.toolbar_frame = ttk.Frame(self.plot_frame)
        self.toolbar_frame.grid(row=2, column=0, sticky="ew")
        self._refresh_toolbar()

        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        self.winfo_toplevel().bind("<<VorLapThemeChanged>>", self._on_theme_changed, add="+")
        self._update_geometry_controls()

    def _on_plot_type_changed(self):
        self._update_geometry_controls()
        self.update_plots()

    def _update_geometry_controls(self):
        if self.plot_type.get() == "geometry":
            self.fancy_geometry_btn.state(["!disabled"])
        else:
            self.fancy_geometry_btn.state(["disabled"])

    def _on_theme_changed(self, _event=None):
        if self.canvas:
            self.update_plots(show_errors=False)

    def _theme_colors(self):
        return getattr(
            self.app,
            "_vorlap_theme_colors",
            {
                "bg": "#1e1e1e",
                "fg": "#d4d4d4",
                "panel_bg": "#2d2d2d",
                "text_bg": "#1f1f1f",
            },
        )

    def _apply_axis_theme(self, axis, is_3d=False):
        colors = self._theme_colors()
        self.fig.patch.set_facecolor(colors["bg"])
        axis.set_facecolor(colors["text_bg"])
        axis.tick_params(colors=colors["fg"])
        axis.xaxis.label.set_color(colors["fg"])
        axis.yaxis.label.set_color(colors["fg"])
        axis.title.set_color(colors["fg"])

        if is_3d and hasattr(axis, "zaxis"):
            axis.zaxis.label.set_color(colors["fg"])
            try:
                axis.xaxis.pane.set_facecolor((0.10, 0.13, 0.20, 0.35))
                axis.yaxis.pane.set_facecolor((0.10, 0.13, 0.20, 0.35))
                axis.zaxis.pane.set_facecolor((0.10, 0.13, 0.20, 0.35))
            except Exception:
                pass
            try:
                axis.zaxis.set_tick_params(colors=colors["fg"])
            except Exception:
                pass

    def _refresh_toolbar(self):
        if not MATPLOTLIB_OK:
            return
        for child in self.toolbar_frame.winfo_children():
            child.destroy()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

    def run_and_plot(self):
        self.app.nb.select(self)
        self.plot_type.set("percdiff")
        self._update_geometry_controls()
        self.app.tab_setup.run_and_save()

    def open_fancy_geometry(self):
        if not self.app.components:
            messagebox.showwarning("No Data", "Load geometry and run analysis first.")
            return
        try:
            results = self.app.analysis_results or {}
            viv_params = results.get("viv_params", self.app.tab_setup.get_viv_params())
            fig = vorlap.graphics.calc_structure_vectors_andplot(
                self.app.components,
                viv_params,
                show_plot=False,
                return_fig=True,
            )
            if fig is not None:
                fig.show(renderer="browser")
        except Exception as exc:
            self.app.log(f"Fancy geometry failed: {exc}\n")
            messagebox.showerror("Fancy Geometry Error", str(exc))

    def on_tab_selected(self):
        try:
            self.update_idletasks()
            self.scrollable.refresh()
            if self.canvas:
                self.canvas.draw_idle()
        except Exception:
            pass

    def _create_norm(self, data):
        data = np.asarray(data, dtype=float)
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        if np.isclose(vmin, vmax):
            eps = max(abs(vmin) * 0.01, 1e-6)
            return mcolors.Normalize(vmin=vmin - eps, vmax=vmax + eps)
        if vmin < 0.0 < vmax:
            return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    def _set_axes_equal_3d(self, axis, points):
        pts = np.vstack(points)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2.0
        span = np.max(maxs - mins)
        if span <= 0:
            span = 1.0
        half = span * 0.55
        axis.set_xlim(center[0] - half, center[0] + half)
        axis.set_ylim(center[1] - half, center[1] + half)
        axis.set_zlim(center[2] - half, center[2] + half)

    def _plot_geometry(self):
        if not self.app.components:
            raise ValueError("No components loaded. Load geometry and run analysis first.")

        results = self.app.analysis_results or {}
        viv_params = results.get("viv_params", self.app.tab_setup.get_viv_params())

        # Populate shape_xyz_global/chord_vector/normal_vector consistently.
        vorlap.graphics.calc_structure_vectors_andplot(
            self.app.components,
            viv_params,
            show_plot=False,
            return_fig=False,
        )

        points = []
        for comp in self.app.components:
            comp_pts = np.asarray(comp.shape_xyz_global, dtype=float)
            if comp_pts.ndim != 2 or comp_pts.shape[1] != 3 or comp_pts.shape[0] == 0:
                continue
            points.append(comp_pts)
            self.ax.plot(comp_pts[:, 0], comp_pts[:, 1], comp_pts[:, 2], color="#60a5fa", linewidth=2.2, alpha=0.95)

            nseg = min(len(comp_pts), len(comp.chord_vector), len(comp.normal_vector))
            step = max(1, nseg // 18)
            for i in range(0, nseg, step):
                p = comp_pts[i]
                chord = np.asarray(comp.chord_vector[i], dtype=float)
                normal = np.asarray(comp.normal_vector[i], dtype=float)
                chord_norm = np.linalg.norm(chord)
                normal_norm = np.linalg.norm(normal)
                if chord_norm > 1e-10:
                    self.ax.quiver(
                        p[0], p[1], p[2],
                        chord[0], chord[1], chord[2],
                        length=1.0,
                        normalize=True,
                        color="#22d3ee",
                        linewidth=0.7,
                        arrow_length_ratio=0.2,
                    )
                if normal_norm > 1e-10:
                    self.ax.quiver(
                        p[0], p[1], p[2],
                        normal[0], normal[1], normal[2],
                        length=1.0,
                        normalize=True,
                        color="#f97316",
                        linewidth=0.7,
                        arrow_length_ratio=0.2,
                    )

        if not points:
            raise ValueError("Component geometry could not be rendered.")

        stacked = np.vstack(points)
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        span = np.max(maxs - mins)
        if span <= 0:
            span = 1.0

        axis_dir = np.asarray(viv_params.rotation_axis, dtype=float)
        axis_norm = np.linalg.norm(axis_dir)
        if axis_norm > 1e-10:
            axis_dir = axis_dir / axis_norm
            origin = np.asarray(viv_params.rotation_axis_offset, dtype=float)
            tip = origin + axis_dir * (0.75 * span)
            self.ax.plot(
                [origin[0], tip[0]],
                [origin[1], tip[1]],
                [origin[2], tip[2]],
                color="#e2e8f0",
                linestyle="--",
                linewidth=1.4,
            )

        inflow_vec = np.asarray(viv_params.inflow_vec, dtype=float)
        inflow_norm = np.linalg.norm(inflow_vec)
        if inflow_norm > 1e-10:
            inflow_vec = inflow_vec / inflow_norm
            center = stacked.mean(axis=0)
            start = center - inflow_vec * (0.35 * span)
            self.ax.quiver(
                start[0], start[1], start[2],
                inflow_vec[0], inflow_vec[1], inflow_vec[2],
                length=0.45 * span,
                normalize=True,
                color="#facc15",
                linewidth=1.2,
                arrow_length_ratio=0.2,
            )

        self._set_axes_equal_3d(self.ax, points)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Structure Geometry (drag to rotate)")
        self._apply_axis_theme(self.ax, is_3d=True)

    def _rebuild_canvas(self):
        old_canvas = self.canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        old_canvas.get_tk_widget().destroy()
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self._refresh_toolbar()

    def update_plots(self, show_errors=True):
        """Update the plot based on current analysis results and selected plot type."""
        if not MATPLOTLIB_OK:
            return

        plot_type = self.plot_type.get()
        results = self.app.analysis_results or {}
        if plot_type != "geometry" and not results:
            return

        try:
            self.fig = Figure(figsize=(10, 8))

            if plot_type == "geometry":
                self.ax = self.fig.add_subplot(111, projection="3d")
                self._plot_geometry()
                self._colorbar = None
            else:
                self.ax = self.fig.add_subplot(111)
                viv_params = results["viv_params"]
                extent = [
                    viv_params.azimuths[0],
                    viv_params.azimuths[-1],
                    viv_params.inflow_speeds[0],
                    viv_params.inflow_speeds[-1],
                ]

                if plot_type == "percdiff":
                    data = results["percdiff_matrix"]
                    im = self.ax.imshow(
                        data,
                        extent=extent,
                        aspect="auto",
                        origin="lower",
                        cmap="viridis_r",
                        vmin=0,
                        vmax=50,
                    )
                    self.ax.set_title("Worst Percent Difference")
                    label = "Freq % Diff"
                elif plot_type.startswith("f"):
                    force_data = results["total_global_force_vector"]
                    idx = {"fx": 0, "fy": 1, "fz": 2}[plot_type]
                    data = force_data[:, :, idx]
                    im = self.ax.imshow(
                        data,
                        extent=extent,
                        aspect="auto",
                        origin="lower",
                        cmap="coolwarm",
                        norm=self._create_norm(data),
                    )
                    self.ax.set_title(f"Force {plot_type.upper()}")
                    label = "Force (N)"
                elif plot_type.startswith("m"):
                    moment_data = results["total_global_moment_vector"]
                    idx = {"mx": 0, "my": 1, "mz": 2}[plot_type]
                    data = moment_data[:, :, idx]
                    im = self.ax.imshow(
                        data,
                        extent=extent,
                        aspect="auto",
                        origin="lower",
                        cmap="coolwarm",
                        norm=self._create_norm(data),
                    )
                    self.ax.set_title(f"Moment {plot_type.upper()}")
                    label = "Moment (N-m)"
                else:
                    raise ValueError(f"Unsupported plot type: {plot_type}")

                self.ax.set_xlabel("Azimuth (deg)")
                self.ax.set_ylabel("Inflow (m/s)")
                self._apply_axis_theme(self.ax)
                self._colorbar = self.fig.colorbar(im, ax=self.ax, label=label)
                if self._colorbar is not None:
                    self._colorbar.ax.yaxis.label.set_color(self._theme_colors()["fg"])
                    self._colorbar.ax.tick_params(colors=self._theme_colors()["fg"])
                    try:
                        self._colorbar.outline.set_edgecolor(self._theme_colors()["fg"])
                    except Exception:
                        pass

            self._rebuild_canvas()
        except Exception as exc:
            self.app.log(f"Plot update failed ({plot_type}): {exc}\n")
            if show_errors:
                messagebox.showerror("Plot Error", str(exc))

    # ---- handlers ----
    def save_plot(self):
        if not self.fig:
            messagebox.showinfo("Plot", "No Matplotlib figure available.")
            return

        suggested = self._last_plot_save_path.strip() or f"{self.plot_type.get()}_plot.png"
        path = filedialog.asksaveasfilename(
            title="Save plot image",
            defaultextension=".png",
            initialfile=suggested,
            filetypes=[("PNG Image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All Files", "*.*")],
        )
        if not path:
            return
        self._last_plot_save_path = path

        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        self.app.log(f"Plot saved to: {path}\n")
