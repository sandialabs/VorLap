#!/usr/bin/env python3
"""
Reusable UI widgets for the VorLap GUI.

This module contains custom widgets that are used across multiple tabs.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import csv


class PathEntry(ttk.Frame):
    """Entry + Browse button (file or directory)."""
    def __init__(self, master, kind="file", title="Select...", must_exist=False, **kwargs):
        super().__init__(master, **kwargs)
        self.kind = kind          # "file" | "dir" | "savefile"
        self.title = title
        self.must_exist = must_exist
        self.var = tk.StringVar()
        self.entry = ttk.Entry(self, textvariable=self.var)
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.btn = ttk.Button(self, text="Browse", command=self.browse)
        self.btn.grid(row=0, column=1)
        self.columnconfigure(0, weight=1)

    def browse(self):
        if self.kind == "file":
            path = filedialog.askopenfilename(title=self.title)
        elif self.kind == "savefile":
            path = filedialog.asksaveasfilename(title=self.title)
        else:
            path = filedialog.askdirectory(title=self.title)
        if path:
            if self.must_exist and not Path(path).exists():
                messagebox.showerror("Path not found", f"{path}\n\ndoes not exist.")
                return
            self.var.set(path)

    def get(self) -> str:
        return self.var.get()

    def set(self, value: str):
        self.var.set(value or "")


class ScrollText(ttk.Frame):
    """A Text widget with a vertical scrollbar."""
    def __init__(self, master, height=10, **kwargs):
        super().__init__(master, **kwargs)
        self.text = tk.Text(self, wrap="word", height=height,
                           font=('Segoe UI', 10),
                           bg='#ffffff',
                           fg='#2d3748',
                           selectbackground='#4299e1',
                           selectforeground='#ffffff',
                           insertbackground='#2d3748',
                           borderwidth=1,
                           relief='solid',
                           padx=8,
                           pady=6)
        sb = ttk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def write(self, s: str):
        self.text.insert("end", s)
        self.text.see("end")

    def clear(self):
        self.text.delete("1.0", "end")


class EditableTreeview(ttk.Frame):
    """
    Spreadsheet-like table with CSV load/save and inline cell editing (double-click).
    """
    def __init__(self, master, columns, show_headings=True, height=8, non_editable_columns=None, **kwargs):
        super().__init__(master, **kwargs)
        self.columns = columns
        self.non_editable_columns = non_editable_columns or []
        self.tree = ttk.Treeview(self, columns=columns, show=("headings" if show_headings else ""))
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")
        vsb = ttk.Scrollbar(self, command=self.tree.yview)
        hsb = ttk.Scrollbar(self, command=self.tree.xview, orient="horizontal")
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._editor = None
        self.tree.bind("<Double-1>", self._begin_edit)

    # ---- data helpers ----
    def clear(self):
        for i in self.tree.get_children():
            self.tree.delete(i)

    def append_row(self, values):
        # pad/truncate to number of columns
        vals = list(values) + [""] * (len(self.columns) - len(values))
        vals = vals[:len(self.columns)]
        self.tree.insert("", "end", values=vals)

    def get_all(self):
        return [self.tree.item(i, "values") for i in self.tree.get_children()]

    # ---- CSV I/O ----
    def load_csv(self, path):
        self.clear()
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                # Convert string values to floats if possible
                float_row = []
                for val in row:
                    try:
                        float_row.append(float(val.strip()))
                    except ValueError:
                        float_row.append(val)  # Keep original value if not a float
                self.append_row(float_row)

    def save_csv(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in self.get_all():
                writer.writerow(row)

    # ---- inline editing ----
    def _begin_edit(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)
        if not row_id or not col_id:
            return
        col = int(col_id.replace("#", "")) - 1
        
        # Check if this column is non-editable
        if self.columns[col] in self.non_editable_columns:
            return
        bbox = self.tree.bbox(row_id, col_id)
        if not bbox:
            return
        x, y, w, h = bbox
        value = self.tree.set(row_id, self.columns[col])

        self._editor = tk.Entry(self.tree,
                               font=('Segoe UI', 10),
                               bg='#ffffff',
                               fg='#2d3748',
                               selectbackground='#4299e1',
                               selectforeground='#ffffff',
                               insertbackground='#2d3748',
                               borderwidth=1,
                               relief='solid')
        self._editor.insert(0, value)
        self._editor.select_range(0, "end")
        self._editor.focus()
        self._editor.place(x=x, y=y, width=w, height=h)

        def _finish(e=None):
            new_val = self._editor.get()
            self.tree.set(row_id, self.columns[col], new_val)
            self._editor.destroy()
            self._editor = None

        self._editor.bind("<Return>", _finish)
        self._editor.bind("<Escape>", lambda e: (self._editor.destroy(), setattr(self, "_editor", None)))
        self._editor.bind("<FocusOut>", _finish) 