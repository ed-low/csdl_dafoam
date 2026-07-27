"""
Interactive viewer for ROM-study results (airfoil_rom_study.py).

Reads the long-form per-sample CSV (default: rom_study_samples_new.csv) where each row is one
test point of one option-combination, and lets you explore error metrics interactively without
editing code. Assign config columns to plot roles (x-axis / group / subplot-rows / subplot-cols)
via dropdowns, multi-select which metrics to show, pin filters, and see each metric's individual
samples plus mean +/- std and min-max range.

This is the interactive successor to the static plot_rom_comparison.py: the same per-x-position
visual vocabulary (sample jitter + mean+/-std errorbar + median + min-max vline, grouped/shaded by
the group column), but driven by live Tkinter widgets instead of module-level ROLES/METRICS/FILTERS.

The original script's multi-file "figure" role (one PNG per value) has no live-window analog; use
the matplotlib toolbar's Save button to export the current view, and the Browse button to load a
different CSV.

Usage:
    python rom_study_viewer.py [path/to/rom_study_samples_new.csv]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import ttk, filedialog


# ============================================================================
# region Schema detection
# ============================================================================
SAMPLE_COL = "point"          # per-combo sample index (0 = reference point)
ID_COLS    = {"combo_id"}     # ids: neither metric nor user-facing role
# Columns that index the multi-IC dimension: they vary within a combo (so the numeric heuristic
# would call them metrics) but are really categorical roles/filters.
FORCE_CONFIG = {"ic_index", "ic_source_point"}
NONE       = "(none)"
METRIC     = "metric"         # virtual dimension placed on a subplot axis
# Sentinels the study writes when a knob doesn't apply to a combo (e.g. distance_metric for the
# unweighted case). Rows carrying these are kept through a filter on that column so they remain
# available for comparison; see the filter loop in _redraw_impl.
NA_SENTINELS = {"n/a", "nan", "none"}

# Default metric selection if present.
DEFAULT_METRICS = ["drag_rel", "lift_rel"]
# Preferred first-launch role columns if they vary.
DEFAULT_X     = "min_modes"
DEFAULT_GROUP = "snapshot_weighting"


def detect_schema(df: pd.DataFrame):
    """Split columns into config (categorical/swept), metric (numeric per-sample), and ignored.

    A column is a *metric* if it is numeric and not a config/id/sample column. Config columns are
    everything else (object dtype, or low-cardinality numerics that index the sweep). Raw outputs
    (rom_*/fom_*) are kept as metrics so they can be plotted too.
    """
    cols = list(df.columns)
    config_cols, metric_cols = [], []
    for c in cols:
        if c in ID_COLS or c == SAMPLE_COL:
            continue
        if c in FORCE_CONFIG:
            config_cols.append(c)
        elif df[c].dtype == object:
            config_cols.append(c)
        else:
            # numeric: swept knobs (few distinct values, small ints/floats) vs. metrics.
            # Heuristic: treat as config if it looks like a swept parameter, else metric.
            metric_cols.append(c)
    # Recover swept numeric knobs (min_modes, fd_step, target_variance, T_residual_scale): they
    # repeat identically across the 11 points of every combo, whereas metrics vary per point.
    n_points = df[SAMPLE_COL].nunique() if SAMPLE_COL in df else 1
    swept = []
    for c in list(metric_cols):
        # constant within each combo -> swept parameter, not a per-sample metric
        if "combo_id" in df.columns:
            per_combo_unique = df.groupby("combo_id")[c].nunique().max()
            if per_combo_unique == 1:
                swept.append(c)
    for c in swept:
        metric_cols.remove(c)
        config_cols.append(c)
    return config_cols, metric_cols


# ============================================================================
# region Plot helpers (ported from plot_rom_comparison.py)
# ============================================================================
PALETTE = ["#3B82F6", "#F97316", "#10B981", "#A855F7",
           "#EF4444", "#EAB308", "#06B6D4", "#EC4899"]
GROUP_COLORS = {"split": "#3B82F6", "monolithic": "#F97316"}
MEAN_MARKER, MEDIAN_MARKER = "D", "o"

LABELS = {
    "corrective_only":          "Corrective Only",
    "geometric_and_corrective": "Geometric & Corrective",
    "in_basis":                 "phi in Basis",
    "interpolated":             "Interpolated IC",
    "reference":                "Reference IC",
    "split":                    "Split",
    "monolithic":               "Monolithic",
}


def pretty(val):
    return LABELS.get(val, str(val).replace("_", " ").title())


def compute_stats(series):
    return dict(
        mean=series.mean(), median=series.median(), std=series.std(),
        lo=series.min(), hi=series.max(), vals=series.values,
    )


# ============================================================================
# region Rendering
# ============================================================================
def render(fig, df, *, x_col, group_col, row_dim, col_dim, metrics, opts):
    """Clear `fig` and draw the subplot grid for the current selection.

    row_dim / col_dim are either a config column name, the literal METRIC, or None.
    `metrics` is the ordered list of selected metric columns.
    """
    fig.clear()
    rng = np.random.default_rng(42)
    # Metrics may contain exact zeros (e.g. err at the reference point); on a log axis matplotlib
    # warns when autoscaling those, but it harmlessly ignores them. Keep the console clean.
    warnings.filterwarnings("ignore", message="Attempt to set non-positive ylim")

    def axis_values(dim):
        if dim is None:
            return [None]
        if dim == METRIC:
            return list(metrics)
        return sorted(df[dim].dropna().unique())

    def select(sub, dim, val):
        if dim is None or dim == METRIC:
            return sub
        return sub[sub[dim] == val]

    def metric_for(row_val, col_val):
        if row_dim == METRIC:
            return row_val
        if col_dim == METRIC:
            return col_val
        return metrics[0]   # single-metric case

    # group values + colour map
    if group_col:
        gv = list(df[group_col].dropna().unique())
        group_values = [g for g in GROUP_COLORS if g in gv] + [g for g in gv if g not in GROUP_COLORS]
    else:
        group_values = [None]
    color_of = {g: GROUP_COLORS.get(g, PALETTE[i % len(PALETTE)]) for i, g in enumerate(group_values)}

    def x_order(sub):
        order = []
        for g in group_values:
            gsub = sub if g is None else sub[sub[group_col] == g]
            for xv in sorted(gsub[x_col].dropna().unique()):
                order.append((g, xv))
        return order

    # shared y-limits across the figure
    finite = df[metrics].replace([np.inf, -np.inf], np.nan)
    y_min = np.nanmin(finite.values) if finite.size else 0.0
    y_max = np.nanmax(finite.values) if finite.size else 1.0
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        y_min, y_max = (y_min or 1e-12), (y_max or 1.0)
    log_y = opts["log_y"] and y_min > 0
    band  = opts["band"]   # "Std" | "Min-Max only" | "None"

    row_vals = axis_values(row_dim)
    col_vals = axis_values(col_dim)
    nrows, ncols = len(row_vals), len(col_vals)
    axes = fig.subplots(nrows, ncols, sharey=True, squeeze=False)

    for r, row_val in enumerate(row_vals):
        for c, col_val in enumerate(col_vals):
            ax = axes[r][c]
            sub = select(select(df, row_dim, row_val), col_dim, col_val)
            metric_col = metric_for(row_val, col_val)
            order = x_order(sub)
            if not order or metric_col not in sub:
                ax.set_visible(False)
                continue

            tick_labels, group_at_x = [], []
            for xi, (g, xv) in enumerate(order):
                mask = (sub[x_col] == xv)
                if g is not None:
                    mask &= (sub[group_col] == g)
                data = sub.loc[mask, metric_col].replace([np.inf, -np.inf], np.nan).dropna()
                tick_labels.append(f"{xv}")
                group_at_x.append(g)
                if data.empty:
                    continue
                s = compute_stats(data)
                color = color_of[g]

                if band in ("Std", "Min-Max only"):
                    ax.vlines(xi, s["lo"], s["hi"], color=color, linewidth=2, alpha=0.35, zorder=1)
                if band == "Std":
                    lo_err = max(s["mean"] - s["std"], s["lo"])
                    hi_err = min(s["mean"] + s["std"], s["hi"])
                    ax.errorbar(xi, s["mean"],
                                yerr=[[s["mean"] - lo_err], [hi_err - s["mean"]]],
                                fmt=MEAN_MARKER, color=color, markersize=9,
                                markeredgecolor="white", markeredgewidth=1.2,
                                capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)
                else:
                    ax.plot(xi, s["mean"], marker=MEAN_MARKER, color=color, markersize=9,
                            markeredgecolor="white", markeredgewidth=1.2, zorder=4)
                ax.plot(xi, s["median"], marker=MEDIAN_MARKER, color=color, markersize=7,
                        markeredgecolor="white", markeredgewidth=1.0, zorder=5, alpha=0.85)
                jitter = rng.uniform(-0.13, 0.13, size=len(s["vals"]))
                ax.scatter(xi + jitter, s["vals"], color=color, alpha=0.35, s=28, zorder=2)

            # group separators / shading / labels
            if group_col:
                for xi in range(1, len(order)):
                    if group_at_x[xi] != group_at_x[xi - 1]:
                        ax.axvline(xi - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
                for g in group_values:
                    grp = [xi for xi, gg in enumerate(group_at_x) if gg == g]
                    if grp:
                        ax.axvspan(grp[0] - 0.45, grp[-1] + 0.45, color=color_of[g], alpha=0.06, zorder=0)

            if log_y:
                ax.set_yscale("log")
                ax.set_ylim(y_min * 0.8, y_max * 1.25)
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels(tick_labels, fontsize=9.5)
            ax.set_xlim(-0.6, len(order) - 0.4)
            ax.set_xlabel(pretty(x_col))
            ax.grid(axis="y", which="both", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.3)
            ax.tick_params(axis="x", length=0)

            if r == 0 and col_dim is not None:
                title = pretty(col_val) if col_dim != METRIC else col_val
                ax.set_title(title, fontsize=11, pad=8)
            if c == 0:
                ylab = (row_val if row_dim == METRIC else pretty(row_val)) if row_dim else metrics[0]
                ax.set_ylabel(ylab, fontsize=10)

            if group_col:
                for g in group_values:
                    grp = [xi for xi, gg in enumerate(group_at_x) if gg == g]
                    if not grp:
                        continue
                    ax.annotate(pretty(g), xy=(np.mean(grp), 1), xycoords=("data", "axes fraction"),
                                xytext=(0, -5), textcoords="offset points", ha="center", va="top",
                                fontsize=9.5, color=color_of[g], fontweight="bold", annotation_clip=False)

    legend_elements = [
        Line2D([0], [0], marker=MEAN_MARKER, color="gray", linestyle="None", markersize=9,
               label="Mean +/- Std" if band == "Std" else "Mean"),
        Line2D([0], [0], marker=MEDIAN_MARKER, color="gray", linestyle="None", markersize=7,
               label="Median", alpha=0.85),
        Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=5, alpha=0.45,
               label="Individual samples"),
    ]
    if band in ("Std", "Min-Max only"):
        legend_elements.append(Line2D([0], [0], color="gray", linewidth=2, alpha=0.4, label="Min - Max range"))
    if group_col:
        for g in group_values:
            legend_elements.append(mpatches.Patch(color=color_of[g], label=pretty(g)))
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements),
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))


def render_message(fig, msg):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, color="#b91c1c", wrap=True)
    ax.axis("off")


# ============================================================================
# region GUI
# ============================================================================
class ViewerApp:
    def __init__(self, root, csv_path):
        self.root = root
        self.root.title("ROM Study Viewer")
        self.df = None
        self.config_cols = []
        self.metric_cols = []
        self._redraw_job = None

        # ---- layout: left controls | right plot ----
        self.controls = ttk.Frame(root, padding=8)
        self.controls.pack(side=tk.LEFT, fill=tk.Y)
        plot_frame = ttk.Frame(root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(10, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame)

        # ---- CSV path row ----
        path_row = ttk.Frame(self.controls)
        path_row.pack(fill=tk.X, pady=(0, 8))
        self.path_var = tk.StringVar(value=str(csv_path))
        ttk.Entry(path_row, textvariable=self.path_var, width=28).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_row, text="Browse", width=7, command=self.browse).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(self.controls, text="Reload CSV", command=self.load_csv).pack(fill=tk.X, pady=(0, 8))

        # ---- role comboboxes ----
        role_box = ttk.LabelFrame(self.controls, text="Roles", padding=6)
        role_box.pack(fill=tk.X, pady=4)
        self.role_vars = {}
        for role in ("X-axis", "Group", "Subplot rows", "Subplot cols"):
            row = ttk.Frame(role_box)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=role, width=12).pack(side=tk.LEFT)
            var = tk.StringVar()
            cb = ttk.Combobox(row, textvariable=var, state="readonly", width=18)
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            cb.bind("<<ComboboxSelected>>", lambda e: self.on_role_change())
            self.role_vars[role] = (var, cb)

        # ---- metrics multi-select ----
        m_box = ttk.LabelFrame(self.controls, text="Metrics (Ctrl/Shift-click)", padding=6)
        m_box.pack(fill=tk.X, pady=4)
        self.metric_list = tk.Listbox(m_box, selectmode=tk.EXTENDED, height=8, exportselection=False)
        self.metric_list.pack(fill=tk.X)
        self.metric_list.bind("<<ListboxSelect>>", lambda e: self.schedule_redraw())

        # ---- options ----
        o_box = ttk.LabelFrame(self.controls, text="Options", padding=6)
        o_box.pack(fill=tk.X, pady=4)
        self.drop_ref = tk.BooleanVar(value=True)
        self.log_y = tk.BooleanVar(value=True)
        self.keep_na = tk.BooleanVar(value=True)
        ttk.Checkbutton(o_box, text="Drop reference point (point=0)", variable=self.drop_ref,
                        command=self.schedule_redraw).pack(anchor=tk.W)
        ttk.Checkbutton(o_box, text="Log y-axis", variable=self.log_y,
                        command=self.schedule_redraw).pack(anchor=tk.W)
        ttk.Checkbutton(o_box, text="Keep N/A rows through filters", variable=self.keep_na,
                        command=self.schedule_redraw).pack(anchor=tk.W)
        band_row = ttk.Frame(o_box)
        band_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(band_row, text="Error band").pack(side=tk.LEFT)
        self.band_var = tk.StringVar(value="Std")
        ttk.Combobox(band_row, textvariable=self.band_var, state="readonly", width=14,
                     values=["Std", "Min-Max only", "None"]).pack(side=tk.LEFT, padx=4)
        self.band_var.trace_add("write", lambda *a: self.schedule_redraw())

        # ---- filters (rebuilt dynamically) ----
        self.filter_box = ttk.LabelFrame(self.controls, text="Filters", padding=6)
        self.filter_box.pack(fill=tk.X, pady=4)
        self.filter_vars = {}

        self.load_csv()

    # -- data loading --
    def browse(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if path:
            self.path_var.set(path)
            self.load_csv()

    def load_csv(self):
        path = self.path_var.get()
        try:
            self.df = pd.read_csv(path)
        except Exception as e:
            render_message(self.fig, f"Could not load CSV:\n{path}\n\n{e}")
            self.canvas.draw()
            return
        self.config_cols, self.metric_cols = detect_schema(self.df)
        self.populate_controls()
        self.schedule_redraw()

    def populate_controls(self):
        varying = [c for c in self.config_cols if self.df[c].nunique() > 1]
        # role options
        x_opts   = varying
        oth_opts = [NONE] + varying
        rc_opts  = [NONE, METRIC] + varying
        for role, opts in (("X-axis", x_opts), ("Group", oth_opts),
                           ("Subplot rows", rc_opts), ("Subplot cols", rc_opts)):
            var, cb = self.role_vars[role]
            cb["values"] = opts
        # defaults
        self.role_vars["X-axis"][0].set(DEFAULT_X if DEFAULT_X in varying else (varying[0] if varying else ""))
        self.role_vars["Group"][0].set(DEFAULT_GROUP if DEFAULT_GROUP in varying else NONE)
        self.role_vars["Subplot rows"][0].set(METRIC)
        self.role_vars["Subplot cols"][0].set(NONE)

        # metrics
        self.metric_list.delete(0, tk.END)
        for m in self.metric_cols:
            self.metric_list.insert(tk.END, m)
        for i, m in enumerate(self.metric_cols):
            if m in DEFAULT_METRICS:
                self.metric_list.selection_set(i)
        if not self.metric_list.curselection() and self.metric_cols:
            self.metric_list.selection_set(0)

        self.rebuild_filters()

    def selected_roles(self):
        x = self.role_vars["X-axis"][0].get()
        g = self.role_vars["Group"][0].get()
        r = self.role_vars["Subplot rows"][0].get()
        c = self.role_vars["Subplot cols"][0].get()
        g = None if g in (NONE, "") else g
        r = None if r in (NONE, "") else r
        c = None if c in (NONE, "") else c
        return x, g, r, c

    def assigned_config_cols(self):
        x, g, r, c = self.selected_roles()
        return {v for v in (x, g, r, c) if v and v != METRIC}

    def rebuild_filters(self):
        for child in self.filter_box.winfo_children():
            child.destroy()
        self.filter_vars = {}
        assigned = self.assigned_config_cols()
        for col in self.config_cols:
            if col in assigned:
                continue
            vals = sorted(self.df[col].dropna().unique())
            if len(vals) <= 1:
                continue
            row = ttk.Frame(self.filter_box)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=col, width=16).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(vals[0]))
            ttk.Combobox(row, textvariable=var, state="readonly", width=14,
                         values=["All"] + [str(v) for v in vals]).pack(side=tk.LEFT, fill=tk.X, expand=True)
            var.trace_add("write", lambda *a: self.schedule_redraw())
            self.filter_vars[col] = var

    def on_role_change(self):
        self.rebuild_filters()
        self.schedule_redraw()

    # -- redraw (debounced) --
    def schedule_redraw(self):
        if self._redraw_job is not None:
            self.root.after_cancel(self._redraw_job)
        self._redraw_job = self.root.after(120, self.redraw)

    def redraw(self):
        self._redraw_job = None
        if self.df is None:
            return
        try:
            self._redraw_impl()
        except Exception as e:
            import traceback
            render_message(self.fig, f"Render error:\n{e}\n\n{traceback.format_exc()}")
        self.canvas.draw()

    def _redraw_impl(self):
        df = self.df
        if self.drop_ref.get() and SAMPLE_COL in df:
            df = df[df[SAMPLE_COL] != 0]

        # apply filters (string-compare against unique values). When keep_na is on, a filter never
        # discards rows whose value is a "not applicable" sentinel (n/a / NaN): those knobs simply
        # don't apply to that row (e.g. distance_metric is "n/a" for the unweighted case), so they
        # should stay visible for comparison against the filtered value.
        keep_na = self.keep_na.get()
        for col, var in self.filter_vars.items():
            val = var.get()
            if val == "All" or col not in df:
                continue
            match = df[col].astype(str) == val
            if keep_na:
                match |= df[col].isna() | df[col].astype(str).str.lower().isin(NA_SENTINELS)
            df = df[match]

        x, g, r, c = self.selected_roles()
        metrics = [self.metric_cols[i] for i in self.metric_list.curselection()]

        if not x:
            return render_message(self.fig, "Select a column for the X-axis.")
        if not metrics:
            return render_message(self.fig, "Select at least one metric.")
        if len(metrics) > 1 and METRIC not in (r, c):
            return render_message(self.fig,
                                  "More than one metric selected.\nAssign 'metric' to Subplot rows or cols\n"
                                  "(or select a single metric).")
        if r == METRIC and c == METRIC:
            return render_message(self.fig, "'metric' can only be on one subplot axis.")
        if df.empty:
            return render_message(self.fig, "No data for the current filter selection.")

        opts = {"log_y": self.log_y.get(), "band": self.band_var.get()}
        render(self.fig, df, x_col=x, group_col=g, row_dim=r, col_dim=c, metrics=metrics, opts=opts)


def default_csv():
    cwd = Path.cwd() / "rom_study_samples_new.csv"
    if cwd.is_file():
        return cwd
    return Path("rom_study_samples_new.csv")


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_csv()
    root = tk.Tk()
    root.geometry("1300x800")
    ViewerApp(root, csv_path)
    root.mainloop()


if __name__ == "__main__":
    main()
