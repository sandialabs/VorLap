#!/usr/bin/env python3
"""
Theme and styling configuration for the VorLap GUI.

This module contains all the styling configuration to maintain a consistent look.
"""

from tkinter import ttk

def setup_theme_and_styling(root):
    """Set up modern theme and improved styling."""
    style = ttk.Style(root)
    
    # Try to use the best available theme
    available_themes = style.theme_names()
    preferred_themes = ["arc", "equilux", "adapta", "clam", "alt", "default"]
    
    selected_theme = "default"
    for theme in preferred_themes:
        if theme in available_themes:
            selected_theme = theme
            break
    
    style.theme_use(selected_theme)
    
    # Configure colors for a modern look
    colors = {
        'bg': '#f0f0f0',           # Light gray background
        'fg': '#2d3748',           # Dark gray text
        'select_bg': '#4299e1',    # Blue selection
        'select_fg': '#ffffff',    # White selected text
        'frame_bg': '#ffffff',     # White frame background
        'button_bg': '#e2e8f0',    # Light button background
        'button_active': '#cbd5e0', # Button active state
        'entry_bg': '#ffffff',     # White entry background
        'tree_bg': '#ffffff',      # White treeview background
        'tree_select': '#e2e8f0'   # Light gray tree selection
    }
    
    # Configure styles with larger fonts
    base_font = ('Segoe UI', 10)
    large_font = ('Segoe UI', 12)
    heading_font = ('Segoe UI', 11, 'bold')
    
    # Configure root window
    root.configure(bg=colors['bg'])
    
    # Configure notebook (tabs)
    style.configure('TNotebook', 
                   background=colors['bg'],
                   borderwidth=0)
    style.configure('TNotebook.Tab',
                   padding=[20, 8],
                   font=heading_font,
                   background=colors['button_bg'],
                   foreground=colors['fg'])
    style.map('TNotebook.Tab',
             background=[('selected', colors['frame_bg']),
                       ('active', colors['button_active'])])
    
    # Configure frames
    style.configure('TFrame',
                   background=colors['bg'])
    
    # Configure labels with larger font
    style.configure('TLabel',
                   font=base_font,
                   background=colors['bg'],
                   foreground=colors['fg'])
    
    # Configure LabelFrame with larger font
    style.configure('TLabelframe',
                   font=heading_font,
                   background=colors['bg'],
                   foreground=colors['fg'],
                   borderwidth=1,
                   relief='solid')
    style.configure('TLabelframe.Label',
                   font=heading_font,
                   background=colors['bg'],
                   foreground=colors['fg'])
    
    # Configure buttons with improved styling
    style.configure('TButton',
                   font=base_font,
                   padding=[12, 6],
                   background=colors['button_bg'],
                   foreground=colors['fg'],
                   borderwidth=1,
                   relief='solid')
    style.map('TButton',
             background=[('active', colors['button_active']),
                       ('pressed', colors['select_bg'])],
             foreground=[('pressed', colors['select_fg'])])
    
    # Configure entries with larger font
    style.configure('TEntry',
                   font=base_font,
                   fieldbackground=colors['entry_bg'],
                   foreground=colors['fg'],
                   borderwidth=1,
                   relief='solid',
                   insertwidth=2)
    style.map('TEntry',
             focuscolor=[('!focus', 'none')])
    
    # Configure spinbox
    style.configure('TSpinbox',
                   font=base_font,
                   fieldbackground=colors['entry_bg'],
                   foreground=colors['fg'],
                   borderwidth=1,
                   relief='solid')
    
    # Configure treeview with larger font
    style.configure('Treeview',
                   font=base_font,
                   background=colors['tree_bg'],
                   foreground=colors['fg'],
                   fieldbackground=colors['tree_bg'],
                   borderwidth=1,
                   relief='solid')
    style.configure('Treeview.Heading',
                   font=heading_font,
                   background=colors['button_bg'],
                   foreground=colors['fg'],
                   borderwidth=1,
                   relief='solid')
    style.map('Treeview',
             background=[('selected', colors['tree_select'])],
             foreground=[('selected', colors['fg'])])
    
    # Configure scrollbars
    style.configure('TScrollbar',
                   background=colors['button_bg'],
                   troughcolor=colors['bg'],
                   borderwidth=1,
                   relief='solid')
    
    # Configure radiobuttons with larger font
    style.configure('TRadiobutton',
                   font=base_font,
                   background=colors['bg'],
                   foreground=colors['fg'],
                   focuscolor='none')
    
    # Configure checkbuttons with larger font
    style.configure('TCheckbutton',
                   font=base_font,
                   background=colors['bg'],
                   foreground=colors['fg'],
                   focuscolor='none')
    style.configure('Accent.TCheckbutton',
                   font=base_font,
                   background=colors['bg'],
                   foreground=colors['select_bg'],
                   focuscolor='none')
    
    # Configure progressbar (if used)
    style.configure('TProgressbar',
                   background=colors['select_bg'],
                   troughcolor=colors['button_bg'],
                   borderwidth=0)
    
    # Configure separators
    style.configure('TSeparator',
                   background=colors['button_bg'])
    
    # Configure status bar
    style.configure('StatusFrame.TFrame',
                   background=colors['frame_bg'],
                   borderwidth=1,
                   relief='solid')
    style.configure('Status.TLabel',
                   font=base_font,
                   background=colors['frame_bg'],
                   foreground=colors['fg']) 