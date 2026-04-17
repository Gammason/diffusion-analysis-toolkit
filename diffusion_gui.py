
"""
Diffusion Analysis Toolkit

Author: Elijah

Description:
GUI-based tool for calculating diffusion coefficients and diffusion lengths
from temperature-dependent data using Arrhenius models. Supports multiple
mechanisms (lattice, grain boundary, surface, effective) and interactive plotting.
"""
# =========================================================


import os
import re
import glob
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

R_GAS = 8.314462618


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def build_marker_list():
    return ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', 'H', 'd', 'p']


def sanitize_positive(y):
    y = np.asarray(y, dtype=float)
    y[~np.isfinite(y)] = np.nan
    return y


def extract_em_label(file_path):
    name = os.path.splitext(os.path.basename(file_path))[0]
    m = re.search(r'(EM\d+)', name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else name


class DiffusionOnlyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diffusion from Temperature CSVs")
        self.root.geometry("1860x1040")

        self.data_folder = ""
        self.data_files = []
        self.calc_data = {}

        self.default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.markers = build_marker_list()

        self.current_series_info = []
        self.current_xlabel = ""
        self.current_ylabel = ""
        self.current_title = ""
        self.probe_artists = []

        self.show_grid_var = tk.BooleanVar(value=False)
        self.use_log_var = tk.BooleanVar(value=True)

        self.build_ui()
        self.build_figure()

    def build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        left_container = ttk.Frame(main)
        left_container.pack(side="left", fill="y", padx=(0, 10))

        canvas = tk.Canvas(left_container, width=650, highlightthickness=0)
        vscroll = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        left = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)

        left.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        # 1. files
        lf = ttk.LabelFrame(left, text="1. Temperature CSV files", padding=10)
        lf.pack(fill="x", pady=(0, 10))
        ttk.Button(lf, text="Load folder with *_with_temperature.csv", command=self.load_folder).pack(fill="x", pady=3)
        self.folder_var = tk.StringVar(value="No folder loaded")
        ttk.Label(lf, textvariable=self.folder_var, wraplength=600).pack(fill="x", pady=3)

        ttk.Label(lf, text="Select files to compare").pack(anchor="w", pady=(8, 3))
        list_frame = ttk.Frame(lf)
        list_frame.pack(fill="both", expand=True)
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, height=10, exportselection=False)
        self.file_listbox.pack(side="left", fill="both", expand=True)
        file_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_listbox.yview)
        file_scroll.pack(side="right", fill="y")
        self.file_listbox.config(yscrollcommand=file_scroll.set)
        ttk.Button(lf, text="Select all", command=lambda: self.file_listbox.select_set(0, "end")).pack(fill="x", pady=(6, 2))
        ttk.Button(lf, text="Clear selection", command=lambda: self.file_listbox.selection_clear(0, "end")).pack(fill="x", pady=2)

        # 2. columns
        cf = ttk.LabelFrame(left, text="2. Column settings", padding=10)
        cf.pack(fill="x", pady=(0, 10))
        ttk.Label(cf, text="Temperature column").pack(anchor="w")
        self.temp_col_var = tk.StringVar(value="Tpulse")
        self.temp_col_combo = ttk.Combobox(cf, textvariable=self.temp_col_var, state="readonly")
        self.temp_col_combo.pack(fill="x", pady=2)
        ttk.Label(cf, text="Pulse amplitude column").pack(anchor="w", pady=(8, 0))
        self.amp_col_var = tk.StringVar(value="Pulse Amplitude")
        self.amp_col_combo = ttk.Combobox(cf, textvariable=self.amp_col_var, state="readonly")
        self.amp_col_combo.pack(fill="x", pady=2)
        ttk.Label(cf, text="Amplitude unit in file").pack(anchor="w", pady=(8, 0))
        self.amp_unit_var = tk.StringVar(value="A")
        ttk.Combobox(cf, textvariable=self.amp_unit_var, state="readonly", values=["A", "mA"]).pack(fill="x", pady=2)

        # 3. plotting/general
        gf = ttk.LabelFrame(left, text="3. Plot and calculation settings", padding=10)
        gf.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(gf); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Pulse time (s)").pack(side="left")
        self.pulse_time_var = tk.StringVar(value="1.0")
        ttk.Entry(row, textvariable=self.pulse_time_var, width=16).pack(side="right")
        row = ttk.Frame(gf); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Temperature floor (K)").pack(side="left")
        self.temp_floor_var = tk.StringVar(value="300")
        ttk.Entry(row, textvariable=self.temp_floor_var, width=16).pack(side="right")
        self.use_floor_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(gf, text="Clip temperatures below floor", variable=self.use_floor_var).pack(anchor="w", pady=2)

        ttk.Label(gf, text="X axis").pack(anchor="w", pady=(8, 0))
        self.x_mode_var = tk.StringVar(value="Temperature (K)")
        ttk.Combobox(
            gf, textvariable=self.x_mode_var, state="readonly",
            values=["Temperature (K)", "1000 / T (K^-1)", "Pulse amplitude (mA)"]
        ).pack(fill="x", pady=2)

        ttk.Label(gf, text="Secondary top X axis").pack(anchor="w", pady=(8, 0))
        self.top_x_mode_var = tk.StringVar(value="None")
        ttk.Combobox(
            gf, textvariable=self.top_x_mode_var, state="readonly",
            values=["None", "Temperature (K)", "1000 / T (K^-1)", "Pulse amplitude (mA)"]
        ).pack(fill="x", pady=2)

        ttk.Label(gf, text="Plot value type").pack(anchor="w", pady=(8, 0))
        self.plot_kind_var = tk.StringVar(value="Diffusion coefficient D")
        ttk.Combobox(
            gf, textvariable=self.plot_kind_var, state="readonly",
            values=["Diffusion coefficient D", "Diffusion length L"]
        ).pack(fill="x", pady=2)

        self.use_log_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(gf, text="Use log Y axis when possible", variable=self.use_log_var).pack(anchor="w", pady=2)

        # 4. channel selection
        chf = ttk.LabelFrame(left, text="4. Mechanisms to plot", padding=10)
        chf.pack(fill="x", pady=(0, 10))
        ttk.Label(chf, text="Use Ctrl-click to choose any combination").pack(anchor="w", pady=(0, 8))

        cols = ttk.Frame(chf)
        cols.pack(fill="x", expand=True)

        cr_frame = ttk.Frame(cols)
        cr_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))
        ttk.Label(cr_frame, text="Cr in Ni").pack(anchor="w")
        self.crni_mech_listbox = tk.Listbox(cr_frame, selectmode=tk.MULTIPLE, height=5, exportselection=False)
        for item in ["lattice", "grain boundary", "surface", "effective"]:
            self.crni_mech_listbox.insert("end", item)
        self.crni_mech_listbox.pack(fill="both", expand=True)
        self.crni_mech_listbox.select_set(3)

        nicr_frame = ttk.Frame(cols)
        nicr_frame.pack(side="left", fill="both", expand=True)
        ttk.Label(nicr_frame, text="Ni in Cr").pack(anchor="w")
        self.nicr_mech_listbox = tk.Listbox(nicr_frame, selectmode=tk.MULTIPLE, height=5, exportselection=False)
        for item in ["lattice", "grain boundary", "surface", "effective"]:
            self.nicr_mech_listbox.insert("end", item)
        self.nicr_mech_listbox.pack(fill="both", expand=True)

        btns = ttk.Frame(chf)
        btns.pack(fill="x", pady=(8, 0))
        ttk.Button(btns, text="Select all Cr in Ni", command=lambda: self.crni_mech_listbox.select_set(0, "end")).pack(side="left", padx=(0, 4))
        ttk.Button(btns, text="Clear Cr in Ni", command=lambda: self.crni_mech_listbox.selection_clear(0, "end")).pack(side="left", padx=(0, 10))
        ttk.Button(btns, text="Select all Ni in Cr", command=lambda: self.nicr_mech_listbox.select_set(0, "end")).pack(side="left", padx=(0, 4))
        ttk.Button(btns, text="Clear Ni in Cr", command=lambda: self.nicr_mech_listbox.selection_clear(0, "end")).pack(side="left")

        # 5. geometry
        geom = ttk.LabelFrame(left, text="5. Geometry and weighting", padding=10)
        geom.pack(fill="x", pady=(0, 10))
        self.ni_thickness_var = tk.StringVar(value="25")
        self.cr_thickness_var = tk.StringVar(value="5")
        self.grain_size_var = tk.StringVar(value="25")
        self.gb_width_var = tk.StringVar(value="0.5")
        self.surface_depth_var = tk.StringVar(value="0.5")
        self.tortuosity_var = tk.StringVar(value="1.0")

        for label, var in [
            ("Ni thickness (nm)", self.ni_thickness_var),
            ("Cr thickness (nm)", self.cr_thickness_var),
            ("Mean grain size (nm)", self.grain_size_var),
            ("GB width δ (nm)", self.gb_width_var),
            ("Surface depth (nm)", self.surface_depth_var),
            ("Tortuosity factor", self.tortuosity_var),
        ]:
            row = ttk.Frame(geom)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label).pack(side="left")
            ttk.Entry(row, textvariable=var, width=16).pack(side="right")

        # 6. parameters
        pf = ttk.LabelFrame(left, text="6. Arrhenius parameters", padding=10)
        pf.pack(fill="x", pady=(0, 10))
        self.param_vars = {}
        defaults = {
            "crni_lat_D0": "3.0e-6", "crni_lat_Q": "171000",
            "crni_gb_D0": "8.4e-3", "crni_gb_Q": "201000",
            "crni_surf_D0": "1.0e-3", "crni_surf_Q": "165000",
            "nicr_lat_D0": "1.8e-5", "nicr_lat_Q": "407000",
            "nicr_gb_D0": "7.4e-6", "nicr_gb_Q": "133888",
            "nicr_surf_D0": "1.0e-3", "nicr_surf_Q": "180000",
        }
        groups = [("Cr in Ni", "crni"), ("Ni in Cr", "nicr")]
        chans = [("lattice", "lat"), ("grain boundary", "gb"), ("surface", "surf")]
        for gname, gkey in groups:
            ttk.Label(pf, text=gname).pack(anchor="w", pady=(4, 2))
            for cname, ckey in chans:
                ttk.Label(pf, text=f"  {cname}").pack(anchor="w")
                row = ttk.Frame(pf); row.pack(fill="x", pady=1)
                ttk.Label(row, text="D0 (m^2/s)").pack(side="left")
                d0_var = tk.StringVar(value=defaults[f"{gkey}_{ckey}_D0"])
                ttk.Entry(row, textvariable=d0_var, width=16).pack(side="right")
                self.param_vars[f"{gkey}_{ckey}_D0"] = d0_var
                row = ttk.Frame(pf); row.pack(fill="x", pady=1)
                ttk.Label(row, text="Q (J/mol)").pack(side="left")
                q_var = tk.StringVar(value=defaults[f"{gkey}_{ckey}_Q"])
                ttk.Entry(row, textvariable=q_var, width=16).pack(side="right")
                self.param_vars[f"{gkey}_{ckey}_Q"] = q_var
            ttk.Separator(pf, orient="horizontal").pack(fill="x", pady=5)

        # 7. style
        sf = ttk.LabelFrame(left, text="7. Style and legend", padding=10)
        sf.pack(fill="x", pady=(0, 10))
        ttk.Label(sf, text="Legend labels").pack(anchor="w")
        self.legend_mode_var = tk.StringVar(value="EM labels only")
        ttk.Combobox(sf, textvariable=self.legend_mode_var, state="readonly",
                     values=["EM labels only", "Full filename"]).pack(fill="x", pady=2)

        ttk.Label(sf, text="Legend location").pack(anchor="w", pady=(8, 0))
        self.legend_loc_var = tk.StringVar(value="best")
        ttk.Combobox(
            sf, textvariable=self.legend_loc_var, state="readonly",
            values=["best", "upper right", "upper left", "lower left", "lower right", "right",
                    "center left", "center right", "lower center", "upper center", "center"]
        ).pack(fill="x", pady=2)

        ttk.Label(sf, text="Font family").pack(anchor="w", pady=(8, 0))
        self.font_family_var = tk.StringVar(value="DejaVu Sans")
        ttk.Combobox(
            sf, textvariable=self.font_family_var, state="readonly",
            values=["Arial", "Times New Roman", "Cambria", "Calibri", "DejaVu Sans",
                    "Liberation Sans", "serif", "sans-serif", "monospace"]
        ).pack(fill="x", pady=2)

        for label, default, attr in [
            ("Base font size", "11", "base_font_var"),
            ("Legend font size", "10", "legend_font_var"),
            ("Title font size", "14", "title_font_var"),
            ("Axis label font size", "12", "axis_font_var"),
            ("Tick label font size", "10", "tick_font_var"),
        ]:
            row = ttk.Frame(sf); row.pack(fill="x", pady=2)
            ttk.Label(row, text=label).pack(side="left")
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Entry(row, textvariable=var, width=16).pack(side="right")

        ttk.Checkbutton(sf, text="Show grid", variable=self.show_grid_var).pack(anchor="w", pady=2)

        # 8. multi-point
        mp = ttk.LabelFrame(left, text="8. Multi-point analysis", padding=10)
        mp.pack(fill="x", pady=(0, 10))
        ttk.Label(mp, text="Probe values (comma-separated)").pack(anchor="w")
        self.probe_values_var = tk.StringVar(value="")
        ttk.Entry(mp, textvariable=self.probe_values_var).pack(fill="x", pady=2)

        ttk.Label(mp, text="Probe based on").pack(anchor="w", pady=(8, 0))
        self.probe_basis_var = tk.StringVar(value="Pulse amplitude (mA)")
        ttk.Combobox(
            mp, textvariable=self.probe_basis_var, state="readonly",
            values=["Primary X axis", "Pulse amplitude (mA)", "Temperature (K)", "1000 / T (K^-1)"]
        ).pack(fill="x", pady=2)

        ttk.Label(mp, text="Annotation position").pack(anchor="w", pady=(8, 0))
        self.annotation_position_var = tk.StringVar(value="Top right")
        ttk.Combobox(
            mp, textvariable=self.annotation_position_var, state="readonly",
            values=["Top left", "Top right", "Bottom left", "Bottom right", "Custom"]
        ).pack(fill="x", pady=2)

        row = ttk.Frame(mp); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Custom X (0-1)").pack(side="left")
        self.annotation_x_var = tk.StringVar(value="0.68")
        ttk.Entry(row, textvariable=self.annotation_x_var, width=12).pack(side="right")

        row = ttk.Frame(mp); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Custom Y (0-1)").pack(side="left")
        self.annotation_y_var = tk.StringVar(value="0.95")
        ttk.Entry(row, textvariable=self.annotation_y_var, width=12).pack(side="right")

        self.show_probe_vlines_var = tk.BooleanVar(value=True)
        self.show_probe_hlines_var = tk.BooleanVar(value=True)
        self.show_probe_box_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mp, text="Show vertical lines", variable=self.show_probe_vlines_var).pack(anchor="w", pady=2)
        ttk.Checkbutton(mp, text="Show horizontal lines", variable=self.show_probe_hlines_var).pack(anchor="w", pady=2)
        ttk.Checkbutton(mp, text="Show value box", variable=self.show_probe_box_var).pack(anchor="w", pady=2)

        self.show_onset_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mp, text="Extract onset from diffusion criterion", variable=self.show_onset_var).pack(anchor="w", pady=(8, 2))
        ttk.Label(mp, text="Onset criterion length (nm)").pack(anchor="w", pady=(6, 0))
        self.onset_length_nm_var = tk.StringVar(value="5")
        ttk.Combobox(
            mp, textvariable=self.onset_length_nm_var, state="readonly",
            values=["1", "5", "20", "25", "30"]
        ).pack(fill="x", pady=2)

        # 9. actions
        af = ttk.LabelFrame(left, text="9. Actions", padding=10)
        af.pack(fill="x", pady=(0, 10))
        ttk.Button(af, text="Run calculations", command=self.run_calculations).pack(fill="x", pady=3)
        ttk.Button(af, text="Plot selected files", command=self.plot_selected).pack(fill="x", pady=3)
        ttk.Button(af, text="Re-analyze probe points on current plot", command=lambda: (self._apply_multi_point_analysis(self.ax), self.canvas.draw())).pack(fill="x", pady=3)
        ttk.Button(af, text="Open pop-up plot", command=self.open_popup_plot).pack(fill="x", pady=3)
        ttk.Button(af, text="Save combined diffusion CSV", command=self.save_combined_csv).pack(fill="x", pady=3)
        ttk.Button(af, text="Save current figure", command=self.save_figure).pack(fill="x", pady=3)

        # 10. output
        of = ttk.LabelFrame(left, text="10. Output", padding=10)
        of.pack(fill="both", expand=True)
        self.output_text = tk.Text(of, width=68, height=16, wrap="word")
        self.output_text.pack(fill="both", expand=True)

        plot_frame = ttk.LabelFrame(right, text="Plot", padding=5)
        plot_frame.pack(fill="both", expand=True)
        self.plot_frame = plot_frame

    def build_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(11, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()
        self.ax.set_title("Load a folder to begin")
        self.ax.grid(False)
        self.fig.tight_layout()
        self.canvas.draw()

    def log(self, text):
        self.output_text.insert("end", text + "\n")
        self.output_text.see("end")

    def clear_log(self):
        self.output_text.delete("1.0", "end")

    def load_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing *_with_temperature.csv files")
        if not folder:
            return

        files = sorted(glob.glob(os.path.join(folder, "*_with_temperature.csv")))
        if not files:
            files = sorted(glob.glob(os.path.join(folder, "EM*.csv")))
        if not files:
            messagebox.showwarning("Warning", "No suitable CSV files found.")
            return

        self.data_folder = folder
        self.data_files = files
        self.calc_data = {}
        self.folder_var.set(folder)
        self.file_listbox.delete(0, "end")
        for f in files:
            self.file_listbox.insert("end", os.path.basename(f))
        self.file_listbox.select_set(0, "end")

        try:
            df0 = pd.read_csv(files[0])
            df0.columns = [str(c).strip() for c in df0.columns]
            numeric_cols = [c for c in df0.columns if safe_numeric(df0[c]).notna().sum() > 0]
            self.temp_col_combo["values"] = numeric_cols
            self.amp_col_combo["values"] = numeric_cols
            if "Tpulse" in numeric_cols:
                self.temp_col_var.set("Tpulse")
            elif numeric_cols:
                self.temp_col_var.set(numeric_cols[0])

            for cand in ["Pulse Amplitude", "Pulse Amplitude (A)", "Current", "Current (A)", "Ipulse", "I"]:
                if cand in numeric_cols:
                    self.amp_col_var.set(cand)
                    break
            else:
                if len(numeric_cols) > 1:
                    self.amp_col_var.set(numeric_cols[1])
        except Exception:
            pass

        self.clear_log()
        self.log(f"Loaded folder: {folder}")
        self.log(f"Found {len(files)} file(s).")

    def get_selected_files(self):
        return [self.data_files[i] for i in self.file_listbox.curselection()]

    def _selected_mechanisms(self):
        mapping = {
            "lattice": ("lat", "lattice"),
            "grain boundary": ("gb", "grain boundary"),
            "surface": ("surf", "surface"),
            "effective": ("eff", "effective"),
        }

        selected = []
        for idx in self.crni_mech_listbox.curselection():
            key = self.crni_mech_listbox.get(idx)
            suffix, label = mapping[key]
            selected.append(("crni", label, suffix))
        for idx in self.nicr_mech_listbox.curselection():
            key = self.nicr_mech_listbox.get(idx)
            suffix, label = mapping[key]
            selected.append(("nicr", label, suffix))
        return selected

    @staticmethod
    def arrhenius_D(T, D0, Q):
        T = np.asarray(T, dtype=float)
        return D0 * np.exp(-Q / (R_GAS * T))

    @staticmethod
    def diffusion_length(D, pulse_time, tortuosity=1.0):
        return np.sqrt(2.0 * np.asarray(D, dtype=float) * pulse_time) / max(tortuosity, 1e-12)

    def _style_plot(self, ax):
        family = self.font_family_var.get()
        base = float(self.base_font_var.get())
        title = float(self.title_font_var.get())
        axis = float(self.axis_font_var.get())
        tick = float(self.tick_font_var.get())

        plt.rcParams.update({"font.family": family, "font.size": base})
        ax.title.set_fontfamily(family)
        ax.title.set_fontsize(title)
        ax.xaxis.label.set_fontfamily(family)
        ax.yaxis.label.set_fontfamily(family)
        ax.xaxis.label.set_fontsize(axis)
        ax.yaxis.label.set_fontsize(axis)
        ax.tick_params(axis="both", labelsize=tick)
        ax.grid(False)
        if self.show_grid_var.get():
            ax.grid(True, alpha=0.25)

    def _legend_label(self, file_path):
        if self.legend_mode_var.get() == "EM labels only":
            return extract_em_label(file_path)
        return os.path.basename(file_path)

    def _mechanism_style(self, mechanism_name):
        styles = {
            "lattice": {"color": "black", "linestyle": "-", "marker": "s"},
            "grain boundary": {"color": "tab:red", "linestyle": "--", "marker": "o"},
            "surface": {"color": "tab:blue", "linestyle": ":", "marker": "^"},
            "effective": {"color": "tab:green", "linestyle": "-.", "marker": "D"},
        }
        return styles.get(mechanism_name, {"color": "tab:purple", "linestyle": "-", "marker": "o"})

    def _interp_unique(self, x_from, y_to, x_new):
        x_from = np.asarray(x_from, dtype=float)
        y_to = np.asarray(y_to, dtype=float)
        x_new = np.asarray(x_new, dtype=float)

        mask = np.isfinite(x_from) & np.isfinite(y_to)
        if mask.sum() < 2:
            return np.full_like(x_new, np.nan, dtype=float)

        xs = x_from[mask]
        ys = y_to[mask]
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        uniq_x, uniq_idx = np.unique(xs, return_index=True)
        uniq_y = ys[uniq_idx]

        if len(uniq_x) < 2:
            return np.full_like(x_new, np.nan, dtype=float)

        clipped = np.clip(x_new, uniq_x.min(), uniq_x.max())
        return np.interp(clipped, uniq_x, uniq_y)

    def _axis_array_for_mode(self, df, mode):
        if mode == "Temperature (K)":
            return df["T_used_K"].to_numpy(dtype=float), "Temperature (K)"
        elif mode == "1000 / T (K^-1)":
            return df["InvT_1000_per_K"].to_numpy(dtype=float), r"1000 / T (K$^{-1}$)"
        elif mode == "Pulse amplitude (mA)":
            return df["PulseAmplitude_mA"].to_numpy(dtype=float), "Pulse amplitude (mA)"
        else:
            return None, None

    def _add_top_x_axis(self, ax, df, primary_mode):
        top_mode = self.top_x_mode_var.get()
        if top_mode == "None" or top_mode == primary_mode:
            return None

        x_primary, _ = self._axis_array_for_mode(df, primary_mode)
        x_top, top_label = self._axis_array_for_mode(df, top_mode)
        if x_primary is None or x_top is None:
            return None

        top_ax = ax.twiny()
        top_ax.set_xlim(ax.get_xlim())
        family = self.font_family_var.get()
        axis_fs = float(self.axis_font_var.get())
        tick_fs = float(self.tick_font_var.get())

        ticks = ax.get_xticks()
        mapped = self._interp_unique(x_primary, x_top, ticks)
        labels = []
        for val in mapped:
            if not np.isfinite(val):
                labels.append("")
            elif top_mode == "Pulse amplitude (mA)":
                labels.append(f"{val:.1f}")
            elif top_mode == "Temperature (K)":
                labels.append(f"{val:.1f}")
            else:
                labels.append(f"{val:.3f}")

        top_ax.set_xticks(ticks)
        top_ax.set_xticklabels(labels)
        top_ax.set_xlabel(top_label, fontsize=axis_fs, fontfamily=family)
        top_ax.tick_params(axis="x", labelsize=tick_fs)
        for lbl in top_ax.get_xticklabels():
            lbl.set_fontfamily(family)
        return top_ax

    def _compute_all_for_df(self, df):
        pulse_time = float(self.pulse_time_var.get())
        floor = float(self.temp_floor_var.get())
        ni_thick = float(self.ni_thickness_var.get())
        cr_thick = float(self.cr_thickness_var.get())
        grain = float(self.grain_size_var.get())
        gb_width = float(self.gb_width_var.get())
        surface_depth = float(self.surface_depth_var.get())
        tortuosity = float(self.tortuosity_var.get())

        T = df["T_used_K"].to_numpy(dtype=float)
        params = {key: float(var.get()) for key, var in self.param_vars.items()}

        f_gb = gb_width / grain
        f_surf_ni = min(1.0, 2.0 * surface_depth / ni_thick)
        f_surf_cr = min(1.0, 2.0 * surface_depth / cr_thick)

        df["D_crni_lat"] = self.arrhenius_D(T, params["crni_lat_D0"], params["crni_lat_Q"])
        df["D_crni_gb"] = self.arrhenius_D(T, params["crni_gb_D0"], params["crni_gb_Q"])
        df["D_crni_surf"] = self.arrhenius_D(T, params["crni_surf_D0"], params["crni_surf_Q"])
        df["D_crni_eff"] = df["D_crni_lat"] + f_gb * df["D_crni_gb"] + f_surf_ni * df["D_crni_surf"]

        df["D_nicr_lat"] = self.arrhenius_D(T, params["nicr_lat_D0"], params["nicr_lat_Q"])
        df["D_nicr_gb"] = self.arrhenius_D(T, params["nicr_gb_D0"], params["nicr_gb_Q"])
        df["D_nicr_surf"] = self.arrhenius_D(T, params["nicr_surf_D0"], params["nicr_surf_Q"])
        df["D_nicr_eff"] = df["D_nicr_lat"] + f_gb * df["D_nicr_gb"] + f_surf_cr * df["D_nicr_surf"]

        for key in ["crni_lat", "crni_gb", "crni_surf", "crni_eff", "nicr_lat", "nicr_gb", "nicr_surf", "nicr_eff"]:
            df[f"L_{key}_nm"] = self.diffusion_length(df[f"D_{key}"].to_numpy(), pulse_time, tortuosity=tortuosity) * 1e9

        df["f_gb"] = f_gb
        df["f_surf_ni"] = f_surf_ni
        df["f_surf_cr"] = f_surf_cr
        df["TempFloor_K"] = floor
        return df

    def run_calculations(self):
        if not self.data_files:
            messagebox.showwarning("Warning", "Load a folder first.")
            return

        selected = self.get_selected_files()
        if not selected:
            messagebox.showwarning("Warning", "Select at least one file.")
            return

        temp_col = self.temp_col_var.get()
        amp_col = self.amp_col_var.get()
        self.calc_data = {}
        self.clear_log()
        self.log("Running diffusion calculations...")

        for file_path in selected:
            try:
                df = pd.read_csv(file_path)
                df.columns = [str(c).strip() for c in df.columns]

                if temp_col not in df.columns or amp_col not in df.columns:
                    raise ValueError("Selected columns not found in file.")

                df[temp_col] = safe_numeric(df[temp_col])
                df[amp_col] = safe_numeric(df[amp_col])
                df = df.dropna(subset=[temp_col, amp_col]).copy()
                if len(df) == 0:
                    raise ValueError("No valid numeric rows.")

                if "Pulse Number" in df.columns:
                    pnum = safe_numeric(df["Pulse Number"])
                    if pnum.notna().sum() == 0:
                        pnum = pd.Series(np.arange(1, len(df) + 1), index=df.index)
                else:
                    pnum = pd.Series(np.arange(1, len(df) + 1), index=df.index)

                df["Pulse_Number"] = pnum

                if self.amp_unit_var.get() == "A":
                    df["PulseAmplitude_mA"] = df[amp_col] * 1e3
                else:
                    df["PulseAmplitude_mA"] = df[amp_col]

                T = df[temp_col].copy()
                if self.use_floor_var.get():
                    T = T.clip(lower=float(self.temp_floor_var.get()))
                df["T_used_K"] = T
                df["InvT_1000_per_K"] = 1000.0 / df["T_used_K"]

                df = self._compute_all_for_df(df)
                self.calc_data[file_path] = df
                self.log(f"OK: {os.path.basename(file_path)} , {len(df)} point(s)")
            except Exception as e:
                self.log(f"ERROR: {os.path.basename(file_path)} , {e}")

        self.log(f"Processed {len(self.calc_data)} file(s).")
        if self.calc_data:
            self.plot_selected()

    def _plot_log_safely(self, ax, x, y, **kwargs):
        x = np.asarray(x, dtype=float)
        y = sanitize_positive(y)
        mask = np.isfinite(x) & np.isfinite(y)
        if getattr(self, 'use_log_var', tk.BooleanVar(value=True)).get():
            mask &= (y > 0)
        x2 = x[mask]
        y2 = y[mask]
        if len(x2) == 0:
            return None
        line, = ax.plot(x2, y2, **kwargs)
        return (line, x2, y2)

    def _apply_log_y_format(self, ax, positive_all):
        if (not getattr(self, 'use_log_var', tk.BooleanVar(value=True)).get()) or len(positive_all) == 0:
            sf = mticker.ScalarFormatter(useMathText=True)
            sf.set_scientific(True)
            sf.set_powerlimits((-2, 2))
            ax.yaxis.set_major_formatter(sf)
            return

        positive = np.asarray([v for v in positive_all if np.isfinite(v) and v > 0], dtype=float)
        if positive.size == 0:
            return

        ax.set_yscale("log")
        ymin = max(float(positive.min()), 1e-300)
        ymax = float(positive.max())
        if ymax > ymin:
            ax.set_ylim(ymin * 0.3, ymax * 2.0)

        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    def _parse_probe_values(self):
        raw = self.probe_values_var.get().strip()
        if not raw:
            return []
        vals = []
        for chunk in raw.replace(";", ",").split(","):
            s = chunk.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except Exception:
                pass
        return vals

    def _clear_probe_artists(self):
        for art in getattr(self, "probe_artists", []):
            try:
                art.remove()
            except Exception:
                pass
        self.probe_artists = []

    def _get_annotation_axes_coords(self):
        pos = self.annotation_position_var.get()
        presets = {
            "Top left": (0.02, 0.98),
            "Top right": (0.98, 0.98),
            "Bottom left": (0.02, 0.02),
            "Bottom right": (0.98, 0.02),
        }
        if pos in presets:
            return presets[pos]
        try:
            x = float(self.annotation_x_var.get())
            y = float(self.annotation_y_var.get())
        except Exception:
            x, y = 0.68, 0.95
        return x, y

    def _format_exact_value(self, value):
        if not np.isfinite(value):
            return "nan"
        if value == 0:
            return "0"
        return f"{value:.3e}"

    def _series_value_at_probe(self, series_info, probe_value, basis_mode):
        basis_mode_effective = self.x_mode_var.get() if basis_mode == "Primary X axis" else basis_mode
        df = series_info["df"]
        basis_arr, _ = self._axis_array_for_mode(df, basis_mode_effective)
        primary_arr, _ = self._axis_array_for_mode(df, self.x_mode_var.get())
        y_arr = np.asarray(series_info["y_raw"], dtype=float)

        if basis_arr is None or primary_arr is None:
            return None

        mask = np.isfinite(basis_arr) & np.isfinite(primary_arr) & np.isfinite(y_arr)
        if getattr(self, 'use_log_var', tk.BooleanVar(value=True)).get():
            mask &= (y_arr > 0)
        if mask.sum() == 0:
            return None

        b = basis_arr[mask]
        xp = primary_arr[mask]
        y = y_arr[mask]
        idx = int(np.nanargmin(np.abs(b - probe_value)))
        return {
            "x_primary": float(xp[idx]),
            "basis_value": float(b[idx]),
            "y_value": float(y[idx]),
            "label": series_info["label"],
            "color": series_info["color"],
        }

    def _apply_multi_point_analysis(self, ax):
        self._clear_probe_artists()
        probe_values = self._parse_probe_values()
        if not probe_values or not self.current_series_info:
            return

        basis_mode = self.probe_basis_var.get()
        ann_x, ann_y = self._get_annotation_axes_coords()
        ha = "left" if ann_x < 0.5 else "right"
        va = "top" if ann_y > 0.5 else "bottom"

        blocks = []
        for probe in probe_values:
            rows = []
            for s in self.current_series_info:
                res = self._series_value_at_probe(s, probe, basis_mode)
                if res is None:
                    continue
                rows.append(res)

                if self.show_probe_hlines_var.get():
                    h = ax.axhline(res["y_value"], color=res["color"], linestyle=":", linewidth=1.0, alpha=0.7)
                    self.probe_artists.append(h)

            if rows:
                x_center = rows[0]["x_primary"]
                if self.show_probe_vlines_var.get():
                    v = ax.axvline(x_center, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
                    self.probe_artists.append(v)

                block = [f"{basis_mode}: {self._format_exact_value(probe)}"]
                for r in rows:
                    block.append(
                        f"{r['label']}: x={self._format_exact_value(r['x_primary'])}, y={self._format_exact_value(r['y_value'])}"
                    )
                blocks.append("\n".join(block))

        if blocks and self.show_probe_box_var.get():
            text = "\n\n".join(blocks)
            t = ax.text(
                ann_x, ann_y, text, transform=ax.transAxes,
                ha=ha, va=va,
                fontsize=float(self.legend_font_var.get()),
                fontfamily=self.font_family_var.get(),
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9)
            )
            self.probe_artists.append(t)

    def _extract_onset_points(self, df, selected_channels):
        if not self.show_onset_var.get():
            return []

        try:
            Lc_nm = float(self.onset_length_nm_var.get())
            pulse_time = float(self.pulse_time_var.get())
        except Exception:
            Lc_nm = 5.0
            pulse_time = 1.0

        onset_points = []
        plot_length = self.plot_kind_var.get().startswith("Diffusion length")
        if plot_length:
            threshold = Lc_nm
        else:
            threshold = ((Lc_nm * 1e-9) ** 2) / (2.0 * pulse_time)

        for system_key, mech_label, suffix in selected_channels:
            if plot_length:
                col = f"L_{system_key}_{suffix}_nm"
            else:
                col = f"D_{system_key}_{suffix}"
            if col not in df.columns:
                continue
            arr = df[col].to_numpy(dtype=float)
            idxs = np.where(np.isfinite(arr) & (arr >= threshold))[0]
            if len(idxs) > 0:
                onset_points.append((int(idxs[0]), system_key, mech_label, threshold))

        return onset_points

    def _draw_onset_guides(self, ax, df, selected_channels):
        onset_points = self._extract_onset_points(df, selected_channels)
        lines = []
        for idx, system_key, mech_label, thr in onset_points:
            style = self._mechanism_style(mech_label)
            mode = self.x_mode_var.get()
            if mode == "Temperature (K)":
                xv = float(df["T_used_K"].iloc[idx])
            elif mode == "1000 / T (K^-1)":
                xv = float(df["InvT_1000_per_K"].iloc[idx])
            else:
                xv = float(df["PulseAmplitude_mA"].iloc[idx])

            ax.axvline(xv, color=style["color"], linestyle=":", linewidth=1.0, alpha=0.85)
            ax.axhline(thr, color=style["color"], linestyle=":", linewidth=1.0, alpha=0.85)

            temp = float(df["T_used_K"].iloc[idx])
            amp = float(df["PulseAmplitude_mA"].iloc[idx])
            system_name = "Cr in Ni" if system_key == "crni" else "Ni in Cr"
            lines.append(f"{system_name}, {mech_label} onset: T={temp:.1f} K, I={amp:.2f} mA, criterion={self._format_exact_value(thr)}")

        return lines

    def _make_title(self, selected_channels):
        systems = []
        if any(ch[0] == "crni" for ch in selected_channels):
            systems.append("Cr in Ni")
        if any(ch[0] == "nicr" for ch in selected_channels):
            systems.append("Ni in Cr")
        kind = "diffusion coefficients" if self.plot_kind_var.get().startswith("Diffusion coefficient") else "diffusion lengths"
        return f"{' + '.join(systems)} , selected {kind}"

    def _y_label(self):
        if self.plot_kind_var.get().startswith("Diffusion coefficient"):
            return r"D (m$^2$/s)"
        return "Diffusion length (nm)"

    def _build_col_name(self, system_key, suffix):
        if self.plot_kind_var.get().startswith("Diffusion coefficient"):
            return f"D_{system_key}_{suffix}"
        return f"L_{system_key}_{suffix}_nm"

    def plot_selected(self):
        if not self.calc_data:
            messagebox.showwarning("Warning", "Run calculations first.")
            return

        selected_files = [f for f in self.get_selected_files() if f in self.calc_data]
        if not selected_files:
            messagebox.showwarning("Warning", "No processed selected files available.")
            return

        selected_channels = self._selected_mechanisms()
        if not selected_channels:
            messagebox.showwarning("Warning", "Select at least one mechanism in Cr in Ni or Ni in Cr.")
            return

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        ax = self.ax

        positive_all = []
        self.current_series_info = []

        family = self.font_family_var.get()
        title_fs = float(self.title_font_var.get())
        axis_fs = float(self.axis_font_var.get())
        legend_fs = float(self.legend_font_var.get())

        onset_lines = []

        for i, file_path in enumerate(selected_files):
            df = self.calc_data[file_path]
            x_mode = self.x_mode_var.get()
            x, xlabel = self._axis_array_for_mode(df, x_mode)
            idx = np.argsort(x)
            x = x[idx]
            base_label = self._legend_label(file_path)

            for system_key, mech_label, suffix in selected_channels:
                col = self._build_col_name(system_key, suffix)
                if col not in df.columns:
                    continue

                y = df[col].to_numpy(dtype=float)[idx]
                system_name = "Cr in Ni" if system_key == "crni" else "Ni in Cr"
                label = f"{base_label}, {system_name}, {mech_label}"
                style = self._mechanism_style(mech_label)
                marker = self.markers[i % len(self.markers)]

                out = self._plot_log_safely(
                    ax, x, y,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=marker,
                    linewidth=2.0,
                    markersize=4.8,
                    label=label
                )
                if out is not None:
                    positive_all.extend(out[2].tolist())
                    self.current_series_info.append({
                        "df": df,
                        "label": label,
                        "x_plot": out[1],
                        "y_plot": out[2],
                        "y_raw": df[col].to_numpy(dtype=float),
                        "color": style["color"],
                    })

            if file_path == selected_files[0]:
                onset_lines = self._draw_onset_guides(ax, df, selected_channels)

        self._apply_log_y_format(ax, positive_all)

        ylabel = self._y_label()
        title = self._make_title(selected_channels)

        ax.set_xlabel(xlabel, fontsize=axis_fs, fontfamily=family)
        self._add_top_x_axis(ax, self.calc_data[selected_files[0]], self.x_mode_var.get())
        ax.set_ylabel(ylabel, fontsize=axis_fs, fontfamily=family)
        ax.set_title(title, fontsize=title_fs, fontfamily=family)
        self._style_plot(ax)

        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        new_h, new_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                new_h.append(h)
                new_l.append(l)
                seen.add(l)
        ax.legend(new_h, new_l, loc=self.legend_loc_var.get(), fontsize=legend_fs)

        self.current_xlabel = xlabel
        self.current_ylabel = ylabel
        self.current_title = title
        self._apply_multi_point_analysis(ax)

        self.fig.tight_layout()
        self.canvas.draw()

        if onset_lines:
            self.log("")
            self.log("Diffusion onset extraction")
            for line in onset_lines:
                self.log(line)

    def open_popup_plot(self):
        if not self.calc_data:
            messagebox.showwarning("Warning", "Run calculations first.")
            return

        selected_files = [f for f in self.get_selected_files() if f in self.calc_data]
        if not selected_files:
            messagebox.showwarning("Warning", "No processed selected files available.")
            return

        selected_channels = self._selected_mechanisms()
        if not selected_channels:
            messagebox.showwarning("Warning", "Select at least one mechanism in Cr in Ni or Ni in Cr.")
            return

        fig, ax = plt.subplots(figsize=(10, 7))
        positive_all = []

        for i, file_path in enumerate(selected_files):
            df = self.calc_data[file_path]
            x, xlabel = self._axis_array_for_mode(df, self.x_mode_var.get())
            idx = np.argsort(x)
            x = x[idx]
            base_label = self._legend_label(file_path)

            for system_key, mech_label, suffix in selected_channels:
                col = self._build_col_name(system_key, suffix)
                if col not in df.columns:
                    continue

                y = df[col].to_numpy(dtype=float)[idx]
                style = self._mechanism_style(mech_label)
                marker = self.markers[i % len(self.markers)]
                system_name = "Cr in Ni" if system_key == "crni" else "Ni in Cr"

                out = self._plot_log_safely(
                    ax, x, y,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=marker,
                    linewidth=2.0,
                    markersize=4.8,
                    label=f"{base_label}, {system_name}, {mech_label}"
                )
                if out is not None:
                    positive_all.extend(out[2].tolist())

        self._apply_log_y_format(ax, positive_all)
        ax.set_xlabel(xlabel)
        self._add_top_x_axis(ax, self.calc_data[selected_files[0]], self.x_mode_var.get())
        ax.set_ylabel(self._y_label())
        ax.set_title(self._make_title(selected_channels))
        ax.grid(False)
        if self.show_grid_var.get():
            ax.grid(True, alpha=0.25)

        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        new_h, new_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                new_h.append(h)
                new_l.append(l)
                seen.add(l)
        ax.legend(new_h, new_l, loc=self.legend_loc_var.get(), fontsize=float(self.legend_font_var.get()))
        fig.tight_layout()
        plt.show()

    def save_combined_csv(self):
        if not self.calc_data:
            messagebox.showwarning("Warning", "Run calculations first.")
            return

        selected = [f for f in self.get_selected_files() if f in self.calc_data]
        if not selected:
            messagebox.showwarning("Warning", "No processed selected files available.")
            return

        frames = []
        for f in selected:
            df = self.calc_data[f].copy()
            df.insert(0, "SourceFile", os.path.basename(f))
            frames.append(df)

        out_df = pd.concat(frames, ignore_index=True)
        file_path = filedialog.asksaveasfilename(
            title="Save combined diffusion CSV",
            defaultextension=".csv",
            initialfile="combined_diffusion_channels.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return

        out_df.to_csv(file_path, index=False)
        self.log(f"Saved combined CSV: {file_path}")

    def save_figure(self):
        file_path = filedialog.asksaveasfilename(
            title="Save current figure",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF file", "*.pdf"), ("SVG file", "*.svg"), ("All files", "*.*")]
        )
        if not file_path:
            return

        self.fig.savefig(file_path, dpi=300, bbox_inches="tight")
        self.log(f"Figure saved: {file_path}")


def main():
    root = tk.Tk()
    app = DiffusionOnlyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
