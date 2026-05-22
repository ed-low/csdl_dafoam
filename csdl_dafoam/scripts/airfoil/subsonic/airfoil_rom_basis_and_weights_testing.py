# ===============================
# region PACKAGES
# ===============================
import numpy as np
import os
from pathlib import Path

from mpi4py import MPI

import csdl_alpha as csdl
import lsdo_geo

from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)

from csdl_dafoam.core.csdl_idwarp import DAFoamMeshWarper
from csdl_dafoam.core.csdl_dafoam import instantiateDAFoam, DAFoamFunctions, DAFoamSolver, compute_dafoam_input_variables
from csdl_dafoam.utils.training_interface import TrainingDataInterface
import csdl_dafoam.utils.standard_atmosphere_model as sam
from csdl_dafoam.utils.runscript_helper_functions import *
from csdl_dafoam.utils.interpolation import RBFInterpolator
from csdl_dafoam.utils.custom_explicit_reduced_svd import customExplicitReducedSVD
from csdl_dafoam.core.rom.csdl_rom import CSDLROMWrapper
from csdl_dafoam.core.rom.rom_models import DAFoamLSPGModel
from csdl_dafoam.core.rom.rom_solver import BroydenNewtonSolver

import matplotlib.pyplot as plt
import faulthandler
faulthandler.enable()
os.environ["PETSC_OPTIONS"] = "-malloc_debug"

print_runscript_info()


# ===============================
# region USER INPUT
# ===============================
problem_name = 'training_data'

geometry_directory        = os.path.join(os.getcwd(), 'airfoil_geometry/')
stp_file_name             = 'airfoil_transonic_unitspan_2.stp'
geometry_pickle_file_name = 'airfoil_stored_refit.pickle'

comm           = MPI.COMM_WORLD
TIMING_ENABLED = True

# Set RUN_ROM=False to only compare POD quality (no DAFoam solves at test points).
# Set RUN_ROM=True to also evaluate each scaling variant's ROM against the FOM.
RUN_ROM = True

dafoam_directory    = os.path.join(os.getcwd(), f'results/{problem_name}/')
dafoamPrintInterval = 100

U0       = 100.08596673909351
p0       = 101325
T0       = 288.150
nuTilda0 = 0.0000181206
A0       = 0.1
rho0     = p0 / T0 / 287

da_options = {
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleCFoam",
    "primalMinResTol": 1.0e-8,
    "primalVarBounds": {"pMin": 5000, "rhoMin": 0.05},
    "primalBC": {
        "U0":       {"variable": "U",        "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0":       {"variable": "p",        "patches": ["inout"], "value": [p0]},
        "T0":       {"variable": "T",        "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda",  "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "function": {
        "drag": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "parallelToFlow",
            "patchVelocityInputName": "patch_velocity",
            "scale": 1.0,
        },
        "lift": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "normalToFlow",
            "patchVelocityInputName": "patch_velocity",
            "scale": 1.0,
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm", "useNonZeroInitGuess": False},
    "transonicPCOption": 1,
    "normalizeStates": {
        "U":       U0,
        "p":       p0,
        "T":       T0,
        "nuTilda": nuTilda0 * 10.0,
        "phi":     1.0,
    },
    "inputInfo": {
        "aero_vol_coords": {
            "type": "volCoord",
            "components": ["solver", "function"],
        },
        "patch_velocity": {
            "type": "patchVelocity",
            "patches": ["inout"],
            "flowAxis": "x",
            "normalAxis": "z",
            "components": ["solver", "function"],
        },
        "pressure": {
            "type": "patchVar",
            "varName": "p",
            "varType": "scalar",
            "patches": ["inout"],
            "components": ["solver", "function"],
        },
        "temperature": {
            "type": "patchVar",
            "varName": "T",
            "varType": "scalar",
            "patches": ["inout"],
            "components": ["solver", "function"],
        },
    },
    "writeAdjointFields": False,
    "debug": False,
    "printDAOptions": True,
    "printInterval": dafoamPrintInterval
}

mesh_options = {
    "gridFile": dafoam_directory,
    "fileType": "OpenFOAM",
    "symmetryPlanes": [],
}

dataset_keyword  = "training_set_with_perturbations_300" #'training_set_with_grad'#
storage_location = Path(dafoam_directory)

n_retained_modes = 20
num_test_samples = 5   # LHS test points (reference point always prepended)


# ===============================
# region SETUP
# ===============================
rank      = comm.Get_rank()
comm_size = comm.Get_size()
rank_str  = f"{rank:0{len(str(comm_size-1))}d}"

dafoam_instance = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)
if RUN_ROM:
    dafoam_instance_rom = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)

h5_path = Path(storage_location) / dataset_keyword / "point_0.h5"

data_generator = TrainingDataInterface(
    dafoam_instance=dafoam_instance,
    storage_location=storage_location,
    dataset_keyword=dataset_keyword,
    h5_file_base_name="point"
)
state_info = data_generator.state_info

# Load data once to read reference state values used to build the scaling dicts below.
_data_ref    = data_generator.load_h5(h5_path, only_distributed_data=False)
_ref_states  = _data_ref["samples"]["reference_states"]
rho_ref      = _ref_states["p"][0] / _ref_states["T"][0] / 287.
U_ref        = _ref_states["U"][0]
n_snapshots  = _data_ref["samples"]["converged"].size

if rank == 0:
    print(f"\n  Training snapshots (incl. reference): {n_snapshots}")
    print(f"  n_retained_modes: {n_retained_modes}")

del _data_ref   # free memory; each variant reloads from disk


# ===============================
# region SCALING VARIANTS
# ===============================
# Three variants covering the history of scaling changes in this codebase.
#
#   v0_pre_ad77952  Before commit ad77952 (Mar 25).  All variables use their freestream
#                   reference value; phi has no scaling (=1).  This was the original approach.
#
#   v1_ad77952      Introduced nuTilda over-scaling (1000×) and a physical scalar for phi
#                   (rho_ref * U_ref).  Pressure still used full static pressure p0.
#
#   v2_current      Current code.  Pressure uses dynamic pressure (0.5*rho*U²), nuTilda
#                   uses 100× over-scaling, phi uses per-face scaling with per-snapshot
#                   face-area normalization to remove mesh-deformation-driven variance.
#
# phi_per_snap_correction: whether snapshots_rbf needs per-snapshot face-area correction
# to match the space in which the POD was computed.  Only True for v2_current because
# _build_pod_inputs applies per-snapshot normalization only when scaling="reference".

SCALING_VARIANTS = {
    "v0_pre_ad77952": {
        "scaling": {
            "U":       float(_ref_states["U"][0]),
            "p":       float(_ref_states["p"][0]),
            "T":       float(_ref_states["T"][0]),
            "nuTilda": float(_ref_states["nuTilda"][0]),
            "phi":     1.0,
        },
        "phi_per_snap_correction": False,
    },
    "v1_ad77952": {
        "scaling": {
            "U":       float(_ref_states["U"][0]),
            "p":       float(_ref_states["p"][0]),
            "T":       float(_ref_states["T"][0]),
            "nuTilda": 1000.0 * float(_ref_states["nuTilda"][0]),
            "phi":     float(rho_ref * U_ref),
        },
        "phi_per_snap_correction": False,
    },
    "v2_current": {
        "scaling": "reference",
        "phi_per_snap_correction": True,
    },
}
VARIANT_NAMES = list(SCALING_VARIANTS.keys())


# ===============================
# region POD COMPUTATION (in memory, no disk writes)
# ===============================
pod_results = {}  # variant_name -> dict of POD outputs

for variant_name, variant_cfg in SCALING_VARIANTS.items():
    if rank == 0:
        print(f"\n{'='*60}\n  Computing POD: {variant_name}\n{'='*60}")

    local_modes, singular_values, reference_state_dict, weights_dict, scaling_values_dict = \
        data_generator._compute_pod_modes(
            h5filepath=h5_path,
            inner_product="reference",
            centering="reference",
            scaling=variant_cfg["scaling"],
            write_h5=False,
            write_modes_using_write_adjoint_fields=False,
        )

    pod_results[variant_name] = {
        "modes":           local_modes,          # dict: state_var -> (n_local_var, n_modes)
        "singular_values": singular_values,      # (n_modes,)
        "reference_state": reference_state_dict, # dict: state_var -> (n_local_var,)
        "weights":         weights_dict,         # dict: state_var -> (n_local_var,)
        "scaling":         scaling_values_dict,  # dict: state_var -> scalar or (n_local_var,)
        "phi_per_snap":    variant_cfg["phi_per_snap_correction"],
    }


# ===============================
# region POD QUALITY COMPARISON
# ===============================
# Project all non-reference training snapshots onto each basis and measure
# reconstruction error.  No FOM solves required.
#
# For phi with v2_current, the POD was computed from per-snapshot face-area-normalised
# phi.  Projecting raw phi through (phi - phi_ref) / scaling_values["phi"] would give
# the wrong inner product; instead we apply the same per-snapshot correction used
# during POD construction so that the projection is self-consistent.

def _print_pod_quality_summary(proj_results, weighted_proj_results, VARIANT_NAMES,
                                rank_checkpoints, rc_idx, n_retained_modes, n_rbf_snaps,
                                state_vars):
    """Print the full POD quality block (unweighted + Euclidean-weighted)."""
    print(f"\n{'='*60}")
    print("  POD Quality: In-Sample Snapshot Reconstruction Error")
    print(f"{'='*60}")

    print(f"\n  Singular value decay (s_i / s_0) — first {min(10, n_retained_modes)} modes:")
    print(f"  {'Mode':>5}", end="")
    for vn in VARIANT_NAMES:
        print(f"  {vn:>20}", end="")
    print()
    for i in range(min(10, min(pr["s_vals"].size for pr in proj_results.values()))):
        print(f"  {i:>5}", end="")
        for vn in VARIANT_NAMES:
            sv = proj_results[vn]["s_vals"]
            print(f"  {sv[i]/sv[0]:>20.6e}", end="")
        print()

    print(f"\n  Cumulative energy at r={n_retained_modes} modes:")
    for vn in VARIANT_NAMES:
        ce    = proj_results[vn]["cum_energy"]
        idx_r = min(n_retained_modes - 1, ce.size - 1)
        print(f"  {vn:<22}: {ce[idx_r]:.6f}")

    print(f"\n  Per-variable cumulative energy fraction (r={n_retained_modes} modes, unweighted):")
    header = f"  {'Variable':<12}"
    for vn in VARIANT_NAMES:
        header += f"  {vn:>22}"
    print(header)
    for sv in state_vars:
        row = f"  {sv:<12}"
        for vn in VARIANT_NAMES:
            row += f"  {proj_results[vn]['per_var_cum_frac'][sv][rc_idx]:>22.4f}"
        print(row)

    print(f"\n  Snapshot reconstruction error, unweighted (mean rel. L2, {n_rbf_snaps} snaps):")
    header = f"  {'Variable':<12}"
    for vn in VARIANT_NAMES:
        header += f"  {vn:>22}"
    print(header)
    for sv in state_vars:
        row = f"  {sv:<12}"
        for vn in VARIANT_NAMES:
            row += f"  {np.mean(proj_results[vn]['per_var_err'][sv]):>22.4e}"
        print(row)

    print(f"\n{'='*60}")
    print("  Euclidean Weighted POD Quality (centroid-RBF weighting)")
    print(f"{'='*60}")

    print(f"\n  Weighted singular value decay (σ_D_i / σ_D_0) — first {min(10, n_retained_modes)} modes:")
    print(f"  {'Mode':>5}", end="")
    for vn in VARIANT_NAMES:
        print(f"  {vn:>20}", end="")
    print()
    for i in range(min(10, min(wpr["sig_D"].size for wpr in weighted_proj_results.values()))):
        print(f"  {i:>5}", end="")
        for vn in VARIANT_NAMES:
            sd = weighted_proj_results[vn]["sig_D"]
            print(f"  {sd[i]/sd[0]:>20.6e}", end="")
        print()

    print(f"\n  Per-variable cumulative energy fraction WITH Euclidean weighting (r={n_retained_modes} modes):")
    header = f"  {'Variable':<12}"
    for vn in VARIANT_NAMES:
        header += f"  {vn:>22}"
    print(header)
    for sv in state_vars:
        row = f"  {sv:<12}"
        for vn in VARIANT_NAMES:
            row += f"  {weighted_proj_results[vn]['per_var_cum_frac'][sv][rc_idx]:>22.4f}"
        print(row)

    print(f"\n  Snapshot reconstruction error WITH Euclidean weighting (mean rel. L2):")
    header = f"  {'Variable':<12}"
    for vn in VARIANT_NAMES:
        header += f"  {vn:>22}"
    print(header)
    for sv in state_vars:
        row = f"  {sv:<12}"
        for vn in VARIANT_NAMES:
            row += f"  {np.mean(weighted_proj_results[vn]['per_var_err'][sv]):>22.4e}"
        print(row)


def _print_rom_summary(diag_dict, ROM_KEYS, state_vars):
    """Print the ROM error summary table."""
    print(f"\n{'='*60}")
    print("  ROM Error Summary (mean over test points)")
    print(f"{'='*60}")
    header = f"  {'ROM variant':<36} {'drag_rel_err':>14} {'lift_rel_err':>14}"
    for sv in state_vars:
        header += f"  {'err_norm_' + sv:>16}"
        header += f"  {'err_norm_w_' + sv:>16}"
    print(header)
    for rk in ROM_KEYS:
        row = (f"  {rk:<36}"
               f" {np.mean(diag_dict['drag_rel_err'][rk]):>14.4e}"
               f" {np.mean(diag_dict['lift_rel_err'][rk]):>14.4e}")
        for sv in state_vars:
            row += f"  {np.mean(diag_dict['err_norm'][rk][sv]):>16.4e}"
            row += f"  {np.mean(diag_dict['err_norm_w'][rk][sv]):>16.4e}"
        print(row)


def assemble_dafoam_vec(per_var_dict, state_info):
    """Pack a per-variable dict into DAFoam's cell-interleaved state vector."""
    n_local = dafoam_instance.getNLocalAdjointStates()
    out     = np.zeros(n_local)
    for sv, info in state_info.items():
        s = np.asarray(per_var_dict[sv])
        out[info["indices"]] = s
    return out


def assemble_dafoam_modes(per_var_modes, state_info, n_modes):
    n_local = dafoam_instance.getNLocalAdjointStates()
    out     = np.zeros((n_local, n_modes))
    for sv, info in state_info.items():
        out[info["indices"], :] = per_var_modes[sv][:, :n_modes]
    return out


if rank == 0:
    print(f"\n{'='*60}")
    print("  POD Quality: In-Sample Snapshot Reconstruction Error")
    print(f"{'='*60}")

# Load the raw snapshot data once for the projection error computation.
data_for_proj = data_generator.load_h5(h5_path, only_distributed_data=False)

# Snapshots excluding the reference (col 0): shape (n_var_dofs, n_rbf_snaps)
n_rbf_snaps     = n_snapshots - 1
fa_ref_local    = np.abs(data_for_proj["samples"]["mesh"]["face_areas"][:, 0])
fa_snaps_local  = np.abs(data_for_proj["samples"]["mesh"]["face_areas"][:, 1:n_rbf_snaps + 1])

# Load parameter coords and compute Euclidean RBF snapshot weights at the centroid.
# Uses the same positive-non-reproducing Gaussian RBF as the ROM's RBFInterpolator.
_dp = data_generator.load_h5(h5_path, only_distributed_data=False)
snapshot_configs_euc = np.concatenate([
    _dp["parameters"]["secondary_variables"]["normalized_percent_camber_change_dof"],
    _dp["parameters"]["secondary_variables"]["percent_change_in_thickness_dof"]
], axis=1)[1:, :]   # (n_rbf_snaps, n_dv); skip reference point
del _dp

_diffs_nn = snapshot_configs_euc[:, None, :] - snapshot_configs_euc[None, :, :]
_d_nn     = np.linalg.norm(_diffs_nn, axis=2)
np.fill_diagonal(_d_nn, np.inf)
_eps_euc  = 1.0 / np.mean(np.min(_d_nn, axis=1))
_centroid = np.mean(snapshot_configs_euc, axis=0)
_r2_cen   = np.sum((snapshot_configs_euc - _centroid) ** 2, axis=1)
w_euc_pod = np.exp(-_eps_euc**2 * _r2_cen)
w_euc_pod /= w_euc_pod.sum()

rank_checkpoints = [rc for rc in [1, 5, 10, 20, 30, 50, 70, 100] if rc <= n_rbf_snaps]

proj_results = {}   # variant_name -> dict of error arrays

for variant_name, pod_data in pod_results.items():
    n_modes  = pod_data["singular_values"].size
    n_use    = min(n_retained_modes, n_modes)
    s_vals   = pod_data["singular_values"]

    pod_modes_arr  = assemble_dafoam_modes(pod_data["modes"], state_info, n_modes)
    ref_state_arr  = assemble_dafoam_vec(pod_data["reference_state"], state_info)
    weights_arr    = assemble_dafoam_vec(pod_data["weights"],         state_info)

    # Build scaling array (scalar or per-DOF depending on variant)
    n_local      = dafoam_instance.getNLocalAdjointStates()
    scaling_arr  = np.zeros(n_local)
    for sv, info in state_info.items():
        s = np.asarray(pod_data["scaling"][sv])
        scaling_arr[info["indices"]] = s if s.ndim == 0 else s

    # Physical snapshots in DAFoam ordering, with phi correction if needed
    snaps_phys = np.zeros((n_local, n_rbf_snaps))
    for sv, info in state_info.items():
        idx      = info["indices"]
        raw_snap = data_for_proj["samples"]["states"][sv][:, 1:n_rbf_snaps + 1]

        if sv == "phi" and pod_data["phi_per_snap"]:
            # Re-express phi as if at reference face areas so that
            # (snaps_phys - ref) / scaling_arr matches the POD scaled space.
            phi_ref_col = data_for_proj["samples"]["states"][sv][:, 0:1]
            fa_j        = np.where(fa_snaps_local < 1e-300, fa_ref_local[:, None], fa_snaps_local)
            corrected   = phi_ref_col + (raw_snap - phi_ref_col) * (fa_ref_local[:, None] / fa_j)
            snaps_phys[idx, :] = corrected
        else:
            snaps_phys[idx, :] = raw_snap

    # Project: alpha = Phi^T (W/S) (snaps - ref),  recon = ref + S Phi alpha
    ws_over_s = (weights_arr / np.where(np.abs(scaling_arr) < 1e-300, 1.0, scaling_arr))
    alpha     = comm.allreduce(
        pod_modes_arr.T @ (ws_over_s[:, None] * (snaps_phys - ref_state_arr[:, None])),
        op=MPI.SUM
    )  # (n_modes, n_rbf_snaps)

    # Singular value decay
    energy      = s_vals ** 2
    cum_energy  = np.cumsum(energy) / np.sum(energy)

    # Per-variable cumulative energy fraction vs. truncation rank
    per_var_cum_frac = {}
    for sv, info in state_info.items():
        idx        = info["indices"]
        phys_diff  = snaps_phys[idx, :] - ref_state_arr[idx, None]
        total_sq   = comm.allreduce(np.sum(phys_diff ** 2), op=MPI.SUM)
        fracs = []
        for rc in rank_checkpoints:
            s_col    = scaling_arr[idx, None] if scaling_arr[idx].ndim == 1 else scaling_arr[idx]
            recon_r  = s_col * (pod_modes_arr[idx, :rc] @ alpha[:rc, :])
            err_sq_r = comm.allreduce(np.sum((phys_diff - recon_r) ** 2), op=MPI.SUM)
            fracs.append(1.0 - err_sq_r / max(total_sq, 1e-300))
        per_var_cum_frac[sv] = fracs

    # Full-basis per-variable relative reconstruction error per snapshot
    recon_phys = scaling_arr[:, None] * (pod_modes_arr @ alpha)
    per_var_err = {}
    for sv, info in state_info.items():
        idx       = info["indices"]
        diff      = snaps_phys[idx, :] - ref_state_arr[idx, None]
        err_sq    = comm.allreduce(np.sum((diff - recon_phys[idx, :]) ** 2, axis=0), op=MPI.SUM)
        norm_sq   = comm.allreduce(np.sum(diff ** 2,                         axis=0), op=MPI.SUM)
        per_var_err[sv] = np.sqrt(err_sq / np.maximum(norm_sq, 1e-30))

    proj_results[variant_name] = {
        "s_vals":            s_vals,
        "cum_energy":        cum_energy,
        "per_var_cum_frac":  per_var_cum_frac,
        "per_var_err":       per_var_err,
        "alpha":             alpha,
        "pod_modes_arr":     pod_modes_arr,
        "ref_state_arr":     ref_state_arr,
        "weights_arr":       weights_arr,
        "scaling_arr":       scaling_arr,
        "s_vals_use":        s_vals[:n_use],
        "snaps_phys":        snaps_phys,
    }

del data_for_proj


# ===============================
# region EUCLIDEAN WEIGHTED POD QUALITY
# ===============================
# For each scaling variant apply Euclidean-RBF snapshot weighting via
# D = diag(σ) VT diag(√w) redecomposition, then measure reconstruction quality
# at truncated ranks (same rank_checkpoints as the unweighted comparison).
#
# alpha = Φ^T (W/S)(snaps - ref) is already stored in proj_results.
# VT    = Σ^{-1} alpha  =>  D = diag(σ) VT diag(√w).
# The weighted modes are  Φ_w = Φ @ U_D  (W-orthonormal, same as ROM uses online).

weighted_proj_results = {}

for variant_name in VARIANT_NAMES:
    pr     = proj_results[variant_name]
    s_vals = pr["s_vals"]
    n_use        = min(n_retained_modes, s_vals.size)
    n_modes_full = s_vals.size

    alpha_all       = pr["alpha"]                 # (n_modes_full, n_rbf_snaps) — full basis
    pod_modes_local = pr["pod_modes_arr"]         # (n_local, n_modes_full)
    scaling_arr     = pr["scaling_arr"]
    snaps_phys      = pr["snaps_phys"]
    ref_arr         = pr["ref_state_arr"]

    # Build D over the full mode set so that the weighted reordering draws from
    # all available modes, then truncate to n_use after the SVD.
    # Uses D = diag(σ) VT diag(w) to match the ROM's D_svd formulation exactly.
    VT_full      = (1.0 / s_vals[:, None]) * alpha_all            # (n_modes_full, n_rbf_snaps)
    D            = s_vals[:, None] * VT_full * w_euc_pod           # (n_modes_full, n_rbf_snaps)
    U_D_full, sig_D_full, _ = np.linalg.svd(D, full_matrices=False)
    U_D   = U_D_full[:, :n_use]                                    # (n_modes_full, n_use)
    sig_D = sig_D_full[:n_use]                                     # (n_use,)

    # Weighted coordinates in truncated weighted basis
    alpha_w  = U_D.T @ alpha_all    # (n_use, n_rbf_snaps) — coords in weighted basis

    # Per-variable cumulative energy fraction at each rank checkpoint
    per_var_cum_frac_w = {}
    for sv, info in state_info.items():
        idx       = info["indices"]
        s_sv      = scaling_arr[idx]                        # (n_sv_dofs,)
        Phi_w_sv  = pod_modes_local[idx, :] @ U_D          # (n_sv_dofs, n_use)
        phys_diff = snaps_phys[idx, :] - ref_arr[idx, None]
        total_sq  = comm.allreduce(np.sum(phys_diff ** 2), op=MPI.SUM)
        fracs_w = []
        for rc in rank_checkpoints:
            rc_c     = min(rc, n_use)
            recon_r  = s_sv[:, None] * (Phi_w_sv[:, :rc_c] @ alpha_w[:rc_c, :])
            err_sq_r = comm.allreduce(np.sum((phys_diff - recon_r) ** 2), op=MPI.SUM)
            fracs_w.append(1.0 - err_sq_r / max(total_sq, 1e-300))
        per_var_cum_frac_w[sv] = fracs_w

    # Full-rank (n_use modes) per-variable reconstruction error
    recon_phys_w = scaling_arr[:, None] * ((pod_modes_local @ U_D) @ alpha_w)
    per_var_err_w = {}
    for sv, info in state_info.items():
        idx    = info["indices"]
        diff   = snaps_phys[idx, :] - ref_arr[idx, None]
        err_sq = comm.allreduce(np.sum((diff - recon_phys_w[idx, :]) ** 2, axis=0), op=MPI.SUM)
        nsq    = comm.allreduce(np.sum(diff ** 2,                           axis=0), op=MPI.SUM)
        per_var_err_w[sv] = np.sqrt(err_sq / np.maximum(nsq, 1e-30))

    weighted_proj_results[variant_name] = {
        "sig_D":             sig_D,
        "U_D":               U_D,
        "per_var_cum_frac":  per_var_cum_frac_w,
        "per_var_err":       per_var_err_w,
        "alpha_w":           alpha_w,
    }


if rank == 0:
    state_vars = list(state_info.keys())
    rc_idx     = max(i for i, rc in enumerate(rank_checkpoints) if rc <= min(n_retained_modes, rank_checkpoints[-1]))
    _print_pod_quality_summary(
        proj_results, weighted_proj_results, VARIANT_NAMES,
        rank_checkpoints, rc_idx, n_retained_modes, n_rbf_snaps, state_vars
    )


# ===============================
# region CSDL RECORDER
# (only needed if RUN_ROM — skip geometry/param setup otherwise)
# ===============================
if not RUN_ROM:
    if rank == 0:
        print("\nRUN_ROM=False — skipping ROM evaluation.")
    exit(0)


x_surf_dafoam_initial_mpi = dafoam_instance.getSurfaceCoordinates()
x_vol_dafoam_initial_mpi  = dafoam_instance.xv0

(x_surf_dafoam_initial,
 x_surf_dafoam_initial_size,
 x_surf_dafoam_initial_indices) = gather_array_to_rank0(x_surf_dafoam_initial_mpi, comm)

if rank == 0:
    x_surf_hash = hash_array_tol(x_surf_dafoam_initial, tol=1e-8)
else:
    x_surf_hash = None
x_surf_hash = comm.bcast(x_surf_hash, root=0)

geometry_pickle_file_path         = Path(geometry_directory) / geometry_pickle_file_name
stp_file_path                     = Path(geometry_directory) / stp_file_name
surface_mesh_projection_file_path = Path(dafoam_directory) / f'projected_surface_mesh_{x_surf_hash}.pickle'

recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()

geometry = lsdo_geo.import_geometry(stp_file_path, parallelize=False)

if surface_mesh_projection_file_path.is_file():
    if rank == 0:
        print('Found surface mesh projection pickle!')
    projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)
else:
    if rank == 0:
        print(f'No projected surface mesh file found at {surface_mesh_projection_file_path}')
        try:
            with Timer('projecting on surface mesh'):
                projected_surf_mesh_dafoam = geometry.project(
                    x_surf_dafoam_initial,
                    grid_search_density_parameter=1,
                    projection_tolerance=1e-10,
                    grid_search_density_cutoff=50,
                    force_reprojection=False,
                    plot=False
                )
            print('Writing surface mesh projection pickle...')
            write_simple_pickle(projected_surf_mesh_dafoam, surface_mesh_projection_file_path)
            print('Done!')
        except Exception as e:
            import traceback
            print(f"[Rank 0 ERROR] Projection/pickle step failed:\n{traceback.format_exc()}", flush=True)
            comm.Abort(1)

    comm.Barrier()
    if rank != 0:
        projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)

print(f'Rank {rank_str} done reading projected surface mesh!')
comm.Barrier()

num_ffd_coefficients_chordwise = 5
num_ffd_sections               = 2
ffd_block = construct_ffd_block_around_entities(
    entities=geometry,
    num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2),
    degree=(3, 1, 1)
)

percent_change_in_thickness_dof      = csdl.Variable(shape=(num_ffd_coefficients_chordwise - 2,), value=np.array([0, 0, 0]))
normalized_percent_camber_change_dof = csdl.Variable(shape=(num_ffd_coefficients_chordwise - 2,), value=np.array([0, 0, 0]))

percent_change_in_thickness      = csdl.Variable(shape=(num_ffd_coefficients_chordwise, num_ffd_sections), value=0.)
normalized_percent_camber_change = csdl.Variable(shape=(num_ffd_coefficients_chordwise, num_ffd_sections), value=0.)

ffd_sectional_parameterization = VolumeSectionalParameterization(
    name="ffd_sectional_parameterization",
    parameterized_points=ffd_block.coefficients,
    principal_parametric_dimension=1,
)

sectional_parameters = VolumeSectionalParameterizationInputs()
ffd_coefficients     = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

original_block_thickness    = ffd_block.coefficients.value[0, 0, 1, 2] - ffd_block.coefficients.value[0, 0, 0, 2]
percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[1:-1, 0], percent_change_in_thickness_dof)
percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[1:-1, 1], percent_change_in_thickness_dof)
delta_block_thickness       = (percent_change_in_thickness / 100) * original_block_thickness
ffd_coefficients = ffd_coefficients.set(csdl.slice[:, :, 1, 2], ffd_coefficients[:, :, 1, 2] + delta_block_thickness / 2)
ffd_coefficients = ffd_coefficients.set(csdl.slice[:, :, 0, 2], ffd_coefficients[:, :, 0, 2] - delta_block_thickness / 2)

block_length                     = ffd_block.coefficients.value[1, 0, 0, 0] - ffd_block.coefficients.value[0, 0, 0, 0]
normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1, 0], normalized_percent_camber_change_dof)
normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1, 1], normalized_percent_camber_change_dof)
camber_change                    = (normalized_percent_camber_change / 100) * block_length
ffd_coefficients = ffd_coefficients.set(
    csdl.slice[:, :, :, 2],
    ffd_coefficients[:, :, :, 2] + csdl.expand(camber_change, (num_ffd_coefficients_chordwise, num_ffd_sections, 2), 'ij->ijk')
)

geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
geometry.set_coefficients(geometry_coefficients)

with Timer('evaluating geometry component', rank, TIMING_ENABLED):
    x_surf_dafoam_full = geometry.evaluate(projected_surf_mesh_dafoam, plot=False)

data_params = data_generator.load_h5(h5_path, group_to_read="parameters", only_distributed_data=False)

flight_conditions_group                     = csdl.VariableGroup()
flight_conditions_group.airspeed_m_s        = csdl.Variable(
    value=data_params["non_sampled_variables"]["airspeed_m_s"], name="airspeed_m_s")
flight_conditions_group.angle_of_attack_deg = csdl.Variable(
    value=data_params["primary_variables"]["angle_of_attack_deg"], name="angle_of_attack_deg"
)
flight_conditions_group.altitude_m          = csdl.Variable(
    value=data_params["non_sampled_variables"]["altitude (m)"], name="altitude (m)"
)
ambient_conditions_group = sam.compute_ambient_conditions_group(flight_conditions_group.altitude_m)

i0, i1 = x_surf_dafoam_initial_indices[rank]

with csdl.experimental.mpi.enter_mpi_region(rank, comm) as mpi_region:

    x_surf_dafoam = x_surf_dafoam_full[i0:i1, :].flatten()

    idwarp_model = DAFoamMeshWarper(dafoam_instance)
    x_vol_dafoam = idwarp_model.evaluate(x_surf_dafoam)

    flight_conditions_group.angle_of_attack_deg = mpi_region.split_custom(
        flight_conditions_group.angle_of_attack_deg, split_func=lambda x: x
    )

    dafoam_input_variables_group = compute_dafoam_input_variables(
        dafoam_instance, ambient_conditions_group, flight_conditions_group, x_vol_dafoam
    )

    residual_scaling                              = np.ones_like(dafoam_instance.getStateWeights())
    residual_scaling[state_info["T"]["indices"]] *= 1005

    # Reuse parameter configs already loaded for the quality section
    snapshot_configs_rbf = snapshot_configs_euc   # (n_rbf_snaps, n_dv)
    n_dv = snapshot_configs_rbf.shape[1]

    current_config = csdl.concatenate(
        (normalized_percent_camber_change_dof, percent_change_in_thickness_dof), axis=0
    )

    # ===============================
    # region ROM LOOP
    # ===============================
    # Two ROMs per scaling variant: unweighted (first n_retained POD modes) and
    # Euclidean-RBF-weighted (D-redecomposition with online query-point weights).
    # VT and rbf_w/UD are computed once per scaling variant and shared.

    ROM_KEYS    = [f"{vn}_{wt}" for vn in VARIANT_NAMES for wt in ("unweighted", "weighted")]
    rom_outputs = {}

    for variant_name, pod_data in pod_results.items():
        if rank == 0:
            print(f"\n  Building ROMs for scaling variant: {variant_name}")

        pr           = proj_results[variant_name]
        pod_modes_np = pr["pod_modes_arr"]    # (n_local, n_modes_full)
        s_vals_full  = pr["s_vals"]           # (n_modes_full,)
        ref_np       = pr["ref_state_arr"]    # (n_local,)
        scaling_np   = pr["scaling_arr"]      # (n_local,)

        # VT = Σ^{-1} Φ^T (W/S)(snaps - ref): reuse alpha from quality section.
        # Use the full mode set so the weighted reordering draws from all modes
        # before truncating to n_retained_modes after the SVD.
        VT = (1.0 / s_vals_full[:, None]) * pr["alpha"]

        # Euclidean RBF weights (CSDL, query-point dependent)
        rbf_w_pre = RBFInterpolator(current_config, snapshot_configs_rbf, positive_non_reproducing_weights=True).weights()
        rbf_w     = rbf_w_pre / csdl.sum(rbf_w_pre)

        # Weighted SVD: D = diag(s) * VT * diag(w_rbf)
        D_svd = csdl.einsum(
            s_vals_full,
            csdl.einsum(VT, csdl.sqrt(rbf_w ** 2), action='ij,j->ij'),
            action='i,ij->ij'
        )
        UD, _, _ = customExplicitReducedSVD().evaluate(A=D_svd)

        for wt_type, pod_modes_for_rom in [
            ("unweighted", pod_modes_np[:, :n_retained_modes]),
            ("weighted",   pod_modes_np @ UD[:, :n_retained_modes]),
        ]:
            rom_key = f"{variant_name}_{wt_type}"
            if rank == 0:
                print(f"    → {rom_key}")

            rom_model = DAFoamLSPGModel(
                dafoam_input_variables_group=dafoam_input_variables_group,
                pod_modes=pod_modes_for_rom,
                reference_fom_state=ref_np,
                scaling=scaling_np,
                weights=1.0 / residual_scaling ** 2,
                dafoam_instance=dafoam_instance_rom,
                normalize_residuals=False,
                fd_step=1e-6,
                solution_prefix=rom_key,
                disable_presolve_diagnostics=False
            )
            rom_wrapper = CSDLROMWrapper(
                model=rom_model,
                solver=BroydenNewtonSolver(options={"tol_rel": 1e-9, "tol_step_abs": 1e-13}),
                start_with_zero_state=True
            )
            rom_states  = rom_wrapper.evaluate()
            state_est   = ref_np + scaling_np * (pod_modes_for_rom @ rom_states)

            fn_model    = DAFoamFunctions(dafoam_instance_rom, disable_jacvec_normalization=True)
            fn_outputs  = fn_model.evaluate(state_est, dafoam_input_variables_group)

            rom_outputs[rom_key] = {
                "state":     state_est,
                "functions": fn_outputs,
            }

            mpi_region.set_as_global_output(state_est)
            for out_name in dafoam_instance.getOption("function").keys():
                mpi_region.set_as_global_output(getattr(fn_outputs, out_name))

    # FOM solver
    dafoam_solver        = DAFoamSolver(dafoam_instance, write_residual_fields=True)
    dafoam_solver_states = dafoam_solver.evaluate(dafoam_input_variables_group)
    dafoam_fn_model      = DAFoamFunctions(dafoam_instance)
    dafoam_fn_outputs    = dafoam_fn_model.evaluate(dafoam_solver_states, dafoam_input_variables_group)

    mpi_region.set_as_global_output(dafoam_solver_states)
    for out_name in dafoam_instance.getOption("function").keys():
        mpi_region.set_as_global_output(getattr(dafoam_fn_outputs, out_name))

recorder.stop()


# ===============================
# region SIMULATION
# ===============================
sim = csdl.experimental.PySimulator(recorder)

snapshot_vars_and_limits = {
    percent_change_in_thickness_dof: {
        'range': [-10, 10],
        'ref_value': 0,
    },
    normalized_percent_camber_change_dof: {
        'range': [-10, 10],
        'ref_value': 0,
    }
}

data_generator._generate_lhs_samples(snapshot_vars_and_limits, num_samples=num_test_samples, random_state=42)
num_samples_with_ref = num_test_samples + 1

diag_dict = {
    "err_norm":     {rk: {sv: np.zeros(num_samples_with_ref) for sv in state_info} for rk in ROM_KEYS},
    "err_norm_w":   {rk: {sv: np.zeros(num_samples_with_ref) for sv in state_info} for rk in ROM_KEYS},
    "drag_rel_err": {rk: np.zeros(num_samples_with_ref) for rk in ROM_KEYS},
    "lift_rel_err": {rk: np.zeros(num_samples_with_ref) for rk in ROM_KEYS},
}

for i in range(num_samples_with_ref):
    for var, info in snapshot_vars_and_limits.items():
        sim[var] = info["samples"][i]
    sim.run()
    if rank == 0:
            print(f"================== {variant_name} WEIGHTS ==========================")
            print(rbf_w.value if is_csdl(rbf_w) else rbf_w)

    fom_state = dafoam_solver_states.value
    fom_drag  = dafoam_fn_outputs.drag.value
    fom_lift  = dafoam_fn_outputs.lift.value

    for rk in ROM_KEYS:
        rom_state = rom_outputs[rk]["state"].value
        rom_drag  = rom_outputs[rk]["functions"].drag.value
        rom_lift  = rom_outputs[rk]["functions"].lift.value

        ###
        base_key  = rk.replace("_unweighted", "")
        base_key  = base_key.replace("_weighted", "")
        weights   = pod_results[base_key]["weights"]
        ###
        

        for sv, info in state_info.items():
            idx      = info["indices"]
            diff     = np.abs(fom_state[idx] - rom_state[idx])
            err_norm = (
                np.sqrt(comm.allreduce(np.sum(diff ** 2),           op=MPI.SUM)) /
                np.sqrt(comm.allreduce(np.sum(fom_state[idx] ** 2), op=MPI.SUM))
            )
            err_norm_w = (
                np.sqrt(comm.allreduce(np.sum(weights[sv] * diff ** 2),           op=MPI.SUM)) /
                np.sqrt(comm.allreduce(np.sum(weights[sv] * fom_state[idx] ** 2), op=MPI.SUM))
            )
            diag_dict["err_norm"][rk][sv][i] = err_norm
            diag_dict["err_norm_w"][rk][sv][i] = err_norm_w

        diag_dict["drag_rel_err"][rk][i] = abs(fom_drag - rom_drag) / (abs(fom_drag) + 1e-300)
        diag_dict["lift_rel_err"][rk][i] = abs(fom_lift - rom_lift) / (abs(fom_lift) + 1e-300)


# ===============================
# region REPORTING
# ===============================


if rank == 0:
    state_vars = list(state_info.keys())
    rc_idx     = max(i for i, rc in enumerate(rank_checkpoints) if rc <= min(n_retained_modes, rank_checkpoints[-1]))

    _print_rom_summary(diag_dict, ROM_KEYS, state_vars)

    # Style maps: same color per scaling variant; hatch distinguishes weighting.
    _palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    VARIANT_COLORS = {vn: _palette[i] for i, vn in enumerate(VARIANT_NAMES)}
    WT_HATCH = {"unweighted": "", "weighted": "///"}

    def _rom_bar_style(rk):
        """Return (color, hatch) for a ROM key like 'v0_pre_ad77952_weighted'."""
        wt = "weighted" if rk.endswith("_weighted") else "unweighted"
        vn = rk[: len(rk) - len("_" + wt)]
        return VARIANT_COLORS[vn], WT_HATCH[wt]

    def _grouped_bar(ax, data_by_key, keys, ylabel, title):
        """Draw a grouped bar chart — one group per test point, one bar per ROM key."""
        n_pts  = len(next(iter(data_by_key.values())))
        n_keys = len(keys)
        width  = 0.8 / n_keys
        xs     = np.arange(n_pts)
        for i, rk in enumerate(keys):
            c, hatch = _rom_bar_style(rk)
            offset   = (i - n_keys / 2 + 0.5) * width
            ax.bar(xs + offset, data_by_key[rk], width=width,
                   color=c, hatch=hatch, edgecolor='black', linewidth=0.5, label=rk)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(j) for j in xs])
        ax.set_xlabel("Test point")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=6)

    # Per-state error norm plots
    n_states = len(state_vars)
    fig, axes = plt.subplots(1, n_states, figsize=(max(6, 2 * num_samples_with_ref) * n_states, 4), sharey=False)
    if n_states == 1:
        axes = [axes]
    for ax, sv in zip(axes, state_vars):
        _grouped_bar(ax, {rk: diag_dict["err_norm"][rk][sv] for rk in ROM_KEYS}, ROM_KEYS,
                     ylabel=r"$\|w_{ROM} - w_{FOM}\| / \|w_{FOM}\|$", title=sv)
    plt.tight_layout()
    plt.savefig("scaling_comparison_err_norm.png", dpi=150)

    # Drag / lift relative error
    fig2, (ax_drag, ax_lift) = plt.subplots(1, 2, figsize=(max(12, 4 * num_samples_with_ref), 4))
    _grouped_bar(ax_drag, diag_dict["drag_rel_err"], ROM_KEYS,
                 ylabel="Relative error", title="Drag relative error")
    _grouped_bar(ax_lift, diag_dict["lift_rel_err"], ROM_KEYS,
                 ylabel="Relative error", title="Lift relative error")
    plt.tight_layout()
    plt.savefig("scaling_comparison_functions.png", dpi=150)

    # Singular value decay comparison (unweighted and D-weighted)
    # Both subplots use the same color per variant for easy cross-panel comparison.
    fig3, (ax_sv, ax_svd) = plt.subplots(1, 2, figsize=(14, 4))
    for vn in VARIANT_NAMES:
        c = VARIANT_COLORS[vn]
        sv_arr = proj_results[vn]["s_vals"]
        ax_sv.semilogy(sv_arr / sv_arr[0], color=c, linestyle='-', marker='o', label=vn)
        sd_arr = weighted_proj_results[vn]["sig_D"]
        ax_svd.semilogy(sd_arr / sd_arr[0], color=c, linestyle='--', marker='^', label=vn)
    ax_sv.set_xlabel("Mode index")
    ax_sv.set_ylabel(r"$\sigma_i / \sigma_0$")
    ax_sv.set_title("Unweighted singular value decay")
    ax_sv.legend(fontsize=8)
    ax_svd.set_xlabel("Mode index")
    ax_svd.set_ylabel(r"$\sigma_{D,i} / \sigma_{D,0}$")
    ax_svd.set_title("Euclidean-weighted D singular value decay")
    ax_svd.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("scaling_comparison_singular_values.png", dpi=150)

    plt.show()

    # ===============================
    # region FINAL SUMMARY
    # ===============================
    print("\n")
    print("=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    _print_pod_quality_summary(
        proj_results, weighted_proj_results, VARIANT_NAMES,
        rank_checkpoints, rc_idx, n_retained_modes, n_rbf_snaps, state_vars
    )
    _print_rom_summary(diag_dict, ROM_KEYS, state_vars)
