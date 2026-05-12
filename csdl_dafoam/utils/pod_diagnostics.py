"""
pod_diagnostics.py
==================
POD basis quality diagnostics for steady CFD snapshot data.
Supports comparison across multiple datasets loaded from HDF5 files.

Usage
-----
    python pod_diagnostics.py --files /path/to/dataset1 /path/to/dataset2 \
                              --labels "Case A" "Case B"             \
                              --energy-threshold 0.999               \
                              --max-rank 50                          \
                              --loo                                   \
                              --outdir ./pod_diagnostics_output

HDF5 structure
--------------
Edit the `load_pod_data()` function to match your file layout.
Placeholders are clearly marked with  # <-- EDIT.
The function must return a dict with keys:
    "samples"  : np.ndarray, shape (n_dof, n_snaps)   – raw snapshot matrix
    "sigma"      : np.ndarray, shape (n_modes,)          – singular values
    "modes"      : np.ndarray, shape (n_dof, n_modes)    – left singular vectors (POD modes)
    "label"      : str                                    – human-readable dataset name
"""

import argparse
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress

matplotlib.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 130,
    }
)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  HDF5 LOADER  ←  edit this function to match your file layout
# ──────────────────────────────────────────────────────────────────────────────

def load_pod_data(h5_path: str, label: str) -> dict:
    """
    Load snapshot matrix, singular values, and POD modes from an HDF5 file.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    label : str
        Human-readable name for this dataset (used in plot legends/titles).

    Returns
    -------
    dict with keys: "samples", "sigma", "modes", "label"
    """
    # directory = Path(directory)

    # # ── locate the HDF5 file ─────────────────────────────────────────────────
    # # OPTION A – single known filename:
    # h5_path = directory / "pod_data.h5"          # <-- EDIT filename

    # OPTION B – first *.h5 / *.hdf5 found in the directory:
    # candidates = list(directory.glob("*.h5")) + list(directory.glob("*.hdf5"))
    # if not candidates:
    #     raise FileNotFoundError(f"No HDF5 file found in {directory}")
    # h5_path = candidates[0]

    with h5py.File(h5_path, "r") as f:

        # ── samples ────────────────────────────────────────────────────────
        # shape expected: (n_dof, n_snaps)
        print("    Loading and scaling samples...")
        samples = np.concatenate([f["samples"]["states"][state_key][:] / f["pod"]["scaling"][state_key][:] for state_key in f["samples"]["states"].keys()], axis=0)

        # ── singular values ──────────────────────────────────────────────────
        # shape expected: (n_modes,), sorted descending
        print("    Loading singular values...")
        sigma = f["pod"]["singular_values"][:]

        # ── POD modes (left singular vectors) ────────────────────────────────
        # shape expected: (n_dof, n_modes)
        print("    Loading modes...")
        modes = np.concatenate([state_mode for state_mode in f["pod"]["modes"].values()], axis=0)

        print("    Loading weights...")
        weights = np.concatenate([state_weights for state_weights in f["pod"]["weights"].values()], axis=0)

    print("    Centering samples...")
    samples = samples - samples[:, 0][:, None]
    samples = samples[:, 1:]

    # Basic sanity checks
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.sort(sigma)[::-1]   # ensure descending order

    print("    Done!")
    return {
        "samples":    np.asarray(samples, dtype=float),
        "sigma":     sigma,
        "modes":     np.asarray(modes, dtype=float),
        "label":     label,
        "weights":   weights,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DIAGNOSTIC FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def diag_energy(sigma: np.ndarray, threshold: float = 0.999) -> dict:
    """Cumulative energy and truncation rank at a given threshold."""
    energy  = sigma ** 2
    cum_pct = np.cumsum(energy) / np.sum(energy)
    rank    = int(np.searchsorted(cum_pct, threshold)) + 1
    return {
        "cum_pct":   cum_pct,
        "rank":      rank,
        "threshold": threshold,
    }


def diag_decay_fit(sigma: np.ndarray) -> dict:
    """
    Fit singular value spectrum to algebraic (log-log) and exponential
    (log-linear) models and return slopes, R², and fit residuals.
    """
    i    = np.arange(1, len(sigma) + 1, dtype=float)
    logs = np.log(sigma + 1e-300)

    # Algebraic fit  σ_i ~ C · i^{-α}
    slope_alg, intercept_alg, r_alg, *_ = linregress(np.log(i), logs)
    fit_alg  = np.exp(intercept_alg) * i ** slope_alg

    # Exponential fit  σ_i ~ C · exp(-β·i)
    slope_exp, intercept_exp, r_exp, *_ = linregress(i, logs)
    fit_exp  = np.exp(intercept_exp + slope_exp * i)

    return {
        "alpha":       -slope_alg,
        "r2_alg":      r_alg ** 2,
        "fit_alg":     fit_alg,
        "beta":        -slope_exp,
        "r2_exp":      r_exp ** 2,
        "fit_exp":     fit_exp,
    }


def diag_rank_measures(sigma: np.ndarray) -> dict:
    """Stable rank and entropy-based effective rank."""
    s2          = sigma ** 2
    stable_rank = np.sum(s2) / s2[0]
    p           = s2 / np.sum(s2)
    eff_rank    = np.exp(-np.sum(p * np.log(p + 1e-300)))
    return {
        "stable_rank": stable_rank,
        "eff_rank":    eff_rank,
    }


def _weighted_projection_error(q, Ur, w=None):
    """Project q onto span(Ur) and return error in the appropriate norm."""
    if w is None:
        coeffs = Ur.T @ q
        proj   = Ur @ coeffs
        return np.linalg.norm(q - proj) / np.linalg.norm(q)
    else:
        coeffs = Ur.T @ (w * q)          # W-inner product
        proj   = Ur @ coeffs
        resid  = q - proj
        norm_q = np.sqrt(q @ (w * q))
        norm_r = np.sqrt(resid @ (w * resid))
        return norm_r / norm_q


def diag_projection_error(sigma, modes=None, samples=None, w=None):
    """
    If modes/samples/w are supplied, compute true projection errors.
    Otherwise fall back to tail-energy proxy (always valid).
    """
    total  = np.sqrt(np.sum(sigma ** 2))
    tail_errors = np.array(
        [np.sqrt(np.sum(sigma[r:] ** 2)) / total for r in range(1, len(sigma) + 1)]
    )

    if modes is None or samples is None:
        return {"errors": tail_errors, "source": "tail energy (no W applied)"}

    proj_errors = np.zeros(modes.shape[1])
    for r in range(1, modes.shape[1] + 1):
        Ur   = modes[:, :r]
        errs = [_weighted_projection_error(samples[:, k], Ur, w)
                for k in range(samples.shape[1])]
        proj_errors[r - 1] = np.mean(errs)

    return {"errors": proj_errors, "source": "explicit W-projection"}


def diag_spectral_gap(sigma: np.ndarray) -> dict:
    """
    Ratio σ_i / σ_{i+1}.  Large local maxima indicate natural truncation points.
    """
    ratios    = sigma[:-1] / (sigma[1:] + 1e-300)
    gap_index = int(np.argmax(ratios)) + 1   # 1-indexed
    gap_value = ratios[gap_index - 1]
    return {
        "ratios":    ratios,
        "gap_index": gap_index,
        "gap_value": gap_value,
    }


def diag_loo(snapshots: np.ndarray, max_rank: int,
             weights: np.ndarray = None, verbose: bool = True) -> dict:
    """
    Leave-one-out cross-validation of POD projection error.

    Parameters
    ----------
    snapshots : (n_dof, n_snaps)
    max_rank  : int
    weights   : (n_dof,) diagonal of the weight matrix W (e.g. cell volumes).
                Pass None for standard Euclidean POD.
    """
    from numpy.linalg import norm

    n_snaps  = snapshots.shape[1]
    max_rank = min(max_rank, n_snaps - 1)
    loo_err  = np.zeros((n_snaps, max_rank))

    # Pre-compute once outside the loop
    if weights is not None:
        w_sqrt     = np.sqrt(weights)          # W^{1/2},  shape (n_dof,)
        w_sqrt_inv = 1.0 / w_sqrt             # W^{-1/2}, shape (n_dof,)

    for k in range(n_snaps):
        if verbose and k % max(1, n_snaps // 10) == 0:
            print(f"  LOO: snapshot {k+1}/{n_snaps}")

        mask = np.ones(n_snaps, dtype=bool)
        mask[k] = False
        S_k = snapshots[:, mask]              # (n_dof, n_snaps-1)

        # ── CHANGE 1: weighted SVD ──────────────────────────────────────────
        # Standard POD:  SVD(S_k)            → U_k are W=I orthonormal
        # Weighted POD:  SVD(W^{1/2} S_k)   → transform back to get
        #                W-orthonormal modes  (U_kᵀ W U_k = I)
        if weights is None:
            U_k, _, _ = np.linalg.svd(S_k, full_matrices=False)
        else:
            S_k_w     = w_sqrt[:, None] * S_k          # W^{1/2} S_k
            U_k_w, _, _ = np.linalg.svd(S_k_w, full_matrices=False)
            U_k       = w_sqrt_inv[:, None] * U_k_w    # W^{-1/2} Ũ_k

        # ── CHANGE 2: W-projection and W-norm error ─────────────────────────
        snap_k = snapshots[:, k]

        if weights is None:
            norm_k = norm(snap_k)
        else:
            norm_k = np.sqrt(snap_k @ (weights * snap_k))   # ||q||_W

        if norm_k == 0:
            continue

        for r in range(1, max_rank + 1):
            Ur    = U_k[:, :r]

            if weights is None:
                coeffs = Ur.T @ snap_k                       # Euclidean
            else:
                coeffs = Ur.T @ (weights * snap_k)          # W-inner product

            proj  = Ur @ coeffs
            resid = snap_k - proj

            if weights is None:
                err = norm(resid) / norm_k
            else:
                err = np.sqrt(resid @ (weights * resid)) / norm_k   # ||r||_W

            loo_err[k, r - 1] = err

    return {
        "mean":     loo_err.mean(axis=0),
        "std":      loo_err.std(axis=0),
        "max_rank": max_rank,
        "weighted": weights is not None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3.  PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

COLORS  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _ax_label(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)


def plot_singular_values(datasets: list, outdir: Path):
    """Panel: raw spectrum + algebraic/exponential fits."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_lin, ax_log = axes

    for idx, ds in enumerate(datasets):
        s   = ds["sigma"]
        i   = np.arange(1, len(s) + 1)
        fit = ds["_decay"]
        c   = COLORS[idx % len(COLORS)]
        lbl = ds["label"]

        ax_lin.plot(i, s, color=c, label=lbl, linewidth=1.5)
        ax_log.semilogy(i, s, color=c, label=lbl, linewidth=1.5)

        # overlay fits on log plot
        ax_log.semilogy(i, fit["fit_alg"], color=c, linestyle="--",
                        alpha=0.6, label=f"{lbl} alg. fit (α={fit['alpha']:.2f}, R²={fit['r2_alg']:.3f})")
        ax_log.semilogy(i, fit["fit_exp"], color=c, linestyle=":",
                        alpha=0.6, label=f"{lbl} exp. fit (β={fit['beta']:.3f}, R²={fit['r2_exp']:.3f})")

    _ax_label(ax_lin, "Mode index i", "σ_i", "Singular value spectrum (linear)")
    _ax_label(ax_log, "Mode index i", "σ_i  (log)", "Singular value spectrum + decay fits")

    fig.tight_layout()
    path = outdir / "01_singular_values.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_energy(datasets: list, outdir: Path):
    """Cumulative energy fraction and truncation rank markers."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, ds in enumerate(datasets):
        e   = ds["_energy"]
        i   = np.arange(1, len(e["cum_pct"]) + 1)
        c   = COLORS[idx % len(COLORS)]
        lbl = ds["label"]

        ax.plot(i, e["cum_pct"], color=c, linewidth=1.5, label=lbl)
        ax.axvline(e["rank"], color=c, linestyle="--", alpha=0.7,
                   label=f"{lbl} r*={e['rank']} ({e['threshold']*100:.1f}%)")

    ax.axhline(datasets[0]["_energy"]["threshold"], color="k",
               linestyle=":", linewidth=1, label=f"Threshold {datasets[0]['_energy']['threshold']}")
    ax.set_ylim(0, 1.02)
    _ax_label(ax, "Number of modes r", "Cumulative energy fraction", "POD energy content")
    fig.tight_layout()
    path = outdir / "02_energy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_projection_error(datasets: list, outdir: Path):
    """Relative tail projection error vs rank (semi-log)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, ds in enumerate(datasets):
        err = ds["_proj_err"]["errors"]
        i   = np.arange(1, len(err) + 1)
        c   = COLORS[idx % len(COLORS)]
        ax.semilogy(i, err, color=c, linewidth=1.5, label=ds["label"])
        # mark energy-based rank
        r_star = ds["_energy"]["rank"]
        ax.axvline(r_star, color=c, linestyle="--", alpha=0.6)
        ax.scatter([r_star], [err[r_star - 1]], color=c, zorder=5,
                   marker=MARKERS[idx % len(MARKERS)],
                   label=f"{ds['label']} r*={r_star} → err={err[r_star-1]:.2e}")

    _ax_label(ax, "Number of modes r", "Relative projection error (L2)",
              "POD projection error vs. rank")
    fig.tight_layout()
    path = outdir / "03_projection_error.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_spectral_gap(datasets: list, outdir: Path):
    """Singular value ratios σ_i / σ_{i+1}."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, ds in enumerate(datasets):
        g   = ds["_gap"]
        i   = np.arange(1, len(g["ratios"]) + 1)
        c   = COLORS[idx % len(COLORS)]
        ax.plot(i, g["ratios"], color=c, linewidth=1.5, label=ds["label"])
        ax.scatter([g["gap_index"]], [g["gap_value"]], color=c, zorder=5,
                   marker=MARKERS[idx % len(MARKERS)], s=60,
                   label=f"{ds['label']} max gap @ i={g['gap_index']} ({g['gap_value']:.2f})")

    _ax_label(ax, "Mode index i", "σ_i / σ_{i+1}", "Spectral gap (singular value ratios)")
    fig.tight_layout()
    path = outdir / "04_spectral_gap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_loo(datasets: list, outdir: Path):
    """LOO cross-validation mean ± 1σ error vs rank."""
    fig, ax = plt.subplots(figsize=(8, 5))
    has_loo = False

    for idx, ds in enumerate(datasets):
        if "_loo" not in ds:
            continue
        has_loo = True
        loo = ds["_loo"]
        i   = np.arange(1, loo["max_rank"] + 1)
        c   = COLORS[idx % len(COLORS)]
        ax.semilogy(i, loo["mean"], color=c, linewidth=1.5, label=ds["label"])
        ax.fill_between(i,
                        np.maximum(loo["mean"] - loo["std"], 1e-16),
                        loo["mean"] + loo["std"],
                        color=c, alpha=0.15)
        # energy-based truncation marker
        r_star = ds["_energy"]["rank"]
        if r_star <= loo["max_rank"]:
            ax.axvline(r_star, color=c, linestyle="--", alpha=0.6,
                       label=f"{ds['label']} energy r*={r_star}")

    if not has_loo:
        plt.close(fig)
        return

    _ax_label(ax, "Number of modes r", "Mean LOO relative error (L2)",
              "Leave-one-out cross-validation (mean ± 1σ)")
    fig.tight_layout()
    path = outdir / "05_loo_cv.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_rank_summary(datasets: list, outdir: Path):
    """
    Bar chart comparing truncation rank, stable rank, and effective rank
    across datasets.
    """
    labels     = [ds["label"] for ds in datasets]
    r_energy   = [ds["_energy"]["rank"]             for ds in datasets]
    r_stable   = [ds["_ranks"]["stable_rank"]        for ds in datasets]
    r_eff      = [ds["_ranks"]["eff_rank"]           for ds in datasets]
    r_gap      = [ds["_gap"]["gap_index"]            for ds in datasets]

    x     = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(7, 2 * len(datasets) + 4), 5))
    ax.bar(x - 1.5 * width, r_energy, width, label="Energy-threshold rank r*", color="steelblue")
    ax.bar(x - 0.5 * width, r_stable, width, label="Stable rank",              color="darkorange")
    ax.bar(x + 0.5 * width, r_eff,    width, label="Effective rank",            color="forestgreen")
    ax.bar(x + 1.5 * width, r_gap,    width, label="Spectral-gap rank",         color="firebrick")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rank")
    ax.set_title("Rank summary across datasets")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = outdir / "06_rank_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_decay_summary(datasets: list, outdir: Path):
    """
    Tabular text figure summarising decay-fit statistics for all datasets.
    """
    col_headers = ["Dataset", "α (alg.)", "R² alg.", "β (exp.)", "R² exp.",
                   "Stable rank", "Eff. rank", "Gap index"]
    rows = []
    for ds in datasets:
        d = ds["_decay"]
        r = ds["_ranks"]
        g = ds["_gap"]
        rows.append([
            ds["label"],
            f"{d['alpha']:.3f}",
            f"{d['r2_alg']:.4f}",
            f"{d['beta']:.4f}",
            f"{d['r2_exp']:.4f}",
            f"{r['stable_rank']:.1f}",
            f"{r['eff_rank']:.1f}",
            f"{g['gap_index']}",
        ])

    n_cols = len(col_headers)
    n_rows = len(rows)
    fig_h  = 0.5 * (n_rows + 2) + 1.0
    fig, ax = plt.subplots(figsize=(14, max(2.5, fig_h)))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)

    # header styling
    for j in range(n_cols):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # alternate row shading
    for i in range(1, n_rows + 1):
        fc = "#ecf0f1" if i % 2 == 0 else "white"
        for j in range(n_cols):
            tbl[i, j].set_facecolor(fc)

    ax.set_title("POD spectrum diagnostic summary", fontsize=13, pad=12)
    fig.tight_layout()
    path = outdir / "07_summary_table.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  REPORT  (plain-text console summary)
# ──────────────────────────────────────────────────────────────────────────────

def print_report(datasets: list):
    sep = "=" * 72
    print(f"\n{sep}")
    print("  POD BASIS DIAGNOSTIC REPORT")
    print(sep)

    for ds in datasets:
        e = ds["_energy"]
        d = ds["_decay"]
        r = ds["_ranks"]
        g = ds["_gap"]
        sigma = ds["sigma"]

        print(f"\n  Dataset : {ds['label']}")
        print(f"  samples : {ds['samples'].shape[1]}   DOF : {ds['samples'].shape[0]}")
        print(f"  Modes available : {len(sigma)}")
        print()
        print(f"  ── Energy ─────────────────────────────────────────────")
        print(f"     Threshold     : {e['threshold']*100:.1f}%")
        print(f"     Truncation r* : {e['rank']}")
        print(f"     Energy @ r*   : {e['cum_pct'][e['rank']-1]*100:.4f}%")
        print()
        print(f"  ── Decay fit ──────────────────────────────────────────")
        print(f"     Algebraic  : α = {d['alpha']:.4f}   R² = {d['r2_alg']:.5f}")
        print(f"     Exponential: β = {d['beta']:.5f}  R² = {d['r2_exp']:.5f}")
        decay_verdict = "exponential" if d["r2_exp"] > d["r2_alg"] else "algebraic"
        print(f"     Better fit : {decay_verdict}  {'✓ smooth manifold' if decay_verdict=='exponential' else '⚠ possible slow convergence'}")
        print()
        print(f"  ── Rank measures ──────────────────────────────────────")
        print(f"     Stable rank   : {r['stable_rank']:.2f}")
        print(f"     Effective rank: {r['eff_rank']:.2f}")
        ratio = e["rank"] / r["stable_rank"]
        flag  = "✓" if ratio < 0.5 else "⚠  r* close to stable rank"
        print(f"     r* / stable   : {ratio:.3f}  {flag}")
        print()
        print(f"  ── Spectral gap ────────────────────────────────────────")
        print(f"     Largest gap after mode : {g['gap_index']}")
        print(f"     Gap value (σ_i/σ_i+1)  : {g['gap_value']:.4f}")

        if "_loo" in ds:
            loo = ds["_loo"]
            r_star = e["rank"]
            if r_star <= loo["max_rank"]:
                loo_at_r = loo["mean"][r_star - 1]
                print()
                print(f"  ── LOO cross-validation ───────────────────────────────")
                print(f"     Mean LOO error @ r* : {loo_at_r:.4e}")
                loo_min_idx = np.argmin(loo["mean"])
                print(f"     LOO minimum @ rank  : {loo_min_idx+1}  (err={loo['mean'][loo_min_idx]:.4e})")

        print()

    print(sep)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="POD basis quality diagnostics (supports multi-dataset comparison).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--files", nargs="+", required=True,
        help="Directories containing HDF5 pod data files.",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Human-readable labels for each dataset (must match --files count).",
    )
    parser.add_argument(
        "--energy-threshold", type=float, default=0.999,
        help="Cumulative energy fraction for truncation rank selection.",
    )
    parser.add_argument(
        "--max-rank", type=int, default=None,
        help="Maximum rank plotted / used in LOO. Defaults to all available modes.",
    )
    parser.add_argument(
        "--loo", action="store_true",
        help="Run leave-one-out cross-validation (can be slow for large datasets).",
    )
    parser.add_argument(
        "--outdir", type=str, default="pod_diagnostics_output",
        help="Directory for output figures and summary.",
    )
    return parser.parse_args()


def main():
    args  = parse_args()
    n     = len(args.files)
    labels = args.labels if args.labels else [f"Dataset {i+1}" for i in range(n)]

    if len(labels) != n:
        raise ValueError("--labels count must match --files count.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    datasets = []
    for d, lbl in zip(args.files, labels):
        print(f"Loading: {d}  ({lbl})")
        ds = load_pod_data(d, lbl)
        datasets.append(ds)

    # ── Run diagnostics ───────────────────────────────────────────────────────
    for ds in datasets:
        sigma    = ds["sigma"]
        max_rank = args.max_rank if args.max_rank else len(sigma)
        max_rank = min(max_rank, len(sigma))

        ds["sigma"]     = sigma[:max_rank]                         # trim to max_rank
        print("    Diag energy...")
        ds["_energy"]   = diag_energy(ds["sigma"], args.energy_threshold)
        print("    Diag decay fit...")
        ds["_decay"]    = diag_decay_fit(ds["sigma"])
        print("    Diag rank measures...")
        ds["_ranks"]    = diag_rank_measures(ds["sigma"])
        print("    Diag projection error...")
        ds["_proj_err"] = diag_projection_error(ds["sigma"], ds.get("modes", None), ds.get("samples", None), ds.get("weights", None))
        print("    Diag spectral gap...")
        ds["_gap"]      = diag_spectral_gap(ds["sigma"])

        if args.loo:
            print(f"Running LOO for: {ds['label']}")
            loo_max = min(max_rank, ds["samples"].shape[1] - 1)
            ds["_loo"] = diag_loo(ds["samples"], loo_max, weights=ds.get("weights", None), verbose=True)

    # ── Console report ────────────────────────────────────────────────────────
    print_report(datasets)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("Generating figures ...")
    plot_singular_values(datasets, outdir)
    plot_energy(datasets, outdir)
    plot_projection_error(datasets, outdir)
    plot_spectral_gap(datasets, outdir)
    if args.loo:
        plot_loo(datasets, outdir)
    plot_rank_summary(datasets, outdir)
    plot_decay_summary(datasets, outdir)

    print(f"\nAll outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()