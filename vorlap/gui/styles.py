#!/usr/bin/env python3
"""
Theme and styling configuration for the VorLap GUI.
"""

from tkinter import ttk


THEME_COLORS = {
    "dark": {
        "bg": "#1e1e1e",
        "fg": "#d4d4d4",
        "muted_fg": "#a0a0a0",
        "select_bg": "#264f78",
        "select_fg": "#ffffff",
        "frame_bg": "#252526",
        "panel_bg": "#2d2d2d",
        "button_bg": "#3c3c3c",
        "button_active": "#4e4e4e",
        "entry_bg": "#1f1f1f",
        "text_bg": "#1f1f1f",
        "tree_bg": "#252526",
        "tree_select": "#264f78",
        "border": "#3c3c3c",
    },
    "light": {
        "bg": "#f0f4f8",
        "fg": "#1f2937",
        "muted_fg": "#4b5563",
        "select_bg": "#2563eb",
        "select_fg": "#ffffff",
        "frame_bg": "#ffffff",
        "panel_bg": "#ffffff",
        "button_bg": "#e2e8f0",
        "button_active": "#cbd5e1",
        "entry_bg": "#ffffff",
        "text_bg": "#ffffff",
        "tree_bg": "#ffffff",
        "tree_select": "#dbeafe",
        "border": "#94a3b8",
    },
}


def setup_theme_and_styling(root, mode="dark"):
    """Apply all ttk styling for the selected theme mode."""
    mode = mode if mode in THEME_COLORS else "dark"
    colors = THEME_COLORS[mode]
    style = ttk.Style(root)

    available_themes = style.theme_names()
    preferred_themes = ["clam", "alt", "default"]
    selected_theme = "default"
    for theme in preferred_themes:
        if theme in available_themes:
            selected_theme = theme
            break
    style.theme_use(selected_theme)

    base_font = ("Segoe UI", 10)
    heading_font = ("Segoe UI", 11, "bold")
    tab_font = ("Segoe UI", 11, "bold")

    root.configure(bg=colors["bg"])
    root._vorlap_theme_mode = mode
    root._vorlap_theme_colors = colors

    style.configure("TFrame", background=colors["bg"])
    style.configure("TLabel", font=base_font, background=colors["bg"], foreground=colors["fg"])

    style.configure(
        "TLabelframe",
        font=heading_font,
        background=colors["bg"],
        foreground=colors["fg"],
        borderwidth=1,
        relief="solid",
    )
    style.configure(
        "TLabelframe.Label",
        font=heading_font,
        background=colors["bg"],
        foreground=colors["fg"],
    )

    style.configure(
        "TButton",
        font=base_font,
        padding=[10, 6],
        background=colors["button_bg"],
        foreground=colors["fg"],
        borderwidth=1,
        relief="solid",
    )
    style.map(
        "TButton",
        background=[("active", colors["button_active"]), ("pressed", colors["select_bg"])],
        foreground=[("pressed", colors["select_fg"])],
    )

    style.configure(
        "TEntry",
        font=base_font,
        fieldbackground=colors["entry_bg"],
        foreground=colors["fg"],
        insertcolor=colors["fg"],
        borderwidth=1,
        relief="solid",
    )

    style.configure(
        "TNotebook",
        background=colors["bg"],
        borderwidth=0,
    )
    style.configure(
        "TNotebook.Tab",
        font=tab_font,
        padding=[16, 7],
        background=colors["button_bg"],
        foreground=colors["fg"],
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", colors["panel_bg"]), ("active", colors["button_active"])],
        foreground=[("selected", colors["fg"])],
        padding=[("selected", [24, 11]), ("!selected", [16, 7])],
    )

    style.configure(
        "Treeview",
        font=base_font,
        background=colors["tree_bg"],
        foreground=colors["fg"],
        fieldbackground=colors["tree_bg"],
        borderwidth=1,
        relief="solid",
    )
    style.configure(
        "Treeview.Heading",
        font=heading_font,
        background=colors["button_bg"],
        foreground=colors["fg"],
        borderwidth=1,
        relief="solid",
    )
    style.map(
        "Treeview",
        background=[("selected", colors["tree_select"])],
        foreground=[("selected", colors["fg"] if mode == "light" else "#0f172a")],
    )

    style.configure(
        "TScrollbar",
        background=colors["button_bg"],
        troughcolor=colors["bg"],
        borderwidth=1,
        relief="solid",
    )

    style.configure("TRadiobutton", font=base_font, background=colors["bg"], foreground=colors["fg"])
    style.configure("TCheckbutton", font=base_font, background=colors["bg"], foreground=colors["fg"])

    style.configure("StatusFrame.TFrame", background=colors["frame_bg"], borderwidth=1, relief="solid")
    style.configure("Status.TLabel", font=base_font, background=colors["frame_bg"], foreground=colors["fg"])

    root.event_generate("<<VorLapThemeChanged>>", when="tail")
