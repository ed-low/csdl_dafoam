# ===============================
# region PACKAGES
# ===============================
import numpy as np
import sys
import os
import time
import pickle
from pathlib import Path

# MPI
from mpi4py import MPI

# CSDL packages
import csdl_alpha as csdl
import lsdo_geo

# LSDO_geo specific
from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)

# Optimization
from modopt import CSDLAlphaProblem
from modopt import PySLSQP, OpenSQP, InteriorPoint

# IDWarp and DAFoam
from csdl_dafoam.core.csdl_idwarp import DAFoamMeshWarper
from csdl_dafoam.core.csdl_dafoam import instantiateDAFoam, DAFoamFunctions, DAFoamSolver, compute_dafoam_input_variables
from csdl_dafoam.utils.training_interface import TrainingDataInterface
import csdl_dafoam.utils.standard_atmosphere_model as sam
from csdl_dafoam.utils.runscript_helper_functions import *

#---- DEBUGGING TOOLS ----
import faulthandler
faulthandler.enable()
os.environ["PETSC_OPTIONS"] = "-malloc_debug"
#-------------------------

# Write this runscript to file before anything
print_runscript_info()

# ===============================
# region USER INPUT
# ===============================
# Keyword for optimization name (optimization results folder will be saved with this name in dafoam directory)
problem_name              = 'training_data' #'rom_with_interpolation_comparison'

# Geometry
geometry_directory        =  os.path.join(os.getcwd(), 'airfoil_geometry/')
stp_file_name             = 'airfoil_transonic_unitspan_2.stp'
geometry_pickle_file_name = 'airfoil_stored_refit.pickle'

# MPI and timing
comm           = MPI.COMM_WORLD
TIMING_ENABLED = True  # True if we want timing printed for the CSDL operations

# DAFoam
dafoam_directory = os.path.join(os.getcwd(), f'results/{problem_name}/')
dafoamPrintInterval = 100

# Initial/reference values for DAFoam (best to use base conditions)
U0        = 100.08596673909351         # used for normalizing CD and CL
p0        = 101325
T0        = 288.150
nuTilda0  = 0.0000181206
aoa0      = 1.416e-1
A0        = 0.1           #
rho0      = p0 / T0 / 287 # used for normalizing CD and CL

# Input parameters for DAFoam
da_options = {
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleCFoam",
    "primalMinResTol": 1.0e-8,
    "primalVarBounds": {"pMin": 5000, "rhoMin": 0.05},
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "function": {
        "drag": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "parallelToFlow",
            "patchVelocityInputName": "patch_velocity",
            "scale": 1.0, #1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
        "lift": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "normalToFlow",
            "patchVelocityInputName": "patch_velocity",
            "scale": 1.0, #1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm", "useNonZeroInitGuess": False},
    # transonic preconditioner to speed up the adjoint convergence
    "transonicPCOption": 1,
    "normalizeStates": {
        "U": U0,
        "p": p0,
        "T": T0,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
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

# region Mesh options
mesh_options = {
    "gridFile": dafoam_directory,
    "fileType": "OpenFOAM",
    "symmetryPlanes": [],
}


# ===============================
# region Training data options
# ===============================
# Storage options
dataset_keyword       = 'training_set_with_grad' #'training_set1'
storage_location      = Path(dafoam_directory)


# ===============================
# region SETUP
# ===============================
# MPI information
rank      = comm.Get_rank()
comm_size = comm.Get_size()
rank_str  = f"{rank:0{len(str(comm_size-1))}d}" # string with zero-padded rank index (for prints)

print(f"COMM SIZE = {comm_size}")


# region DAFoam instance
dafoam_instance           = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)

# POD Data import
data_generator = TrainingDataInterface(dafoam_instance=dafoam_instance, 
                                        storage_location=storage_location, 
                                        dataset_keyword=dataset_keyword,
                                        h5_file_base_name="point")

# Manually obtaining the file for now
data        = data_generator.load_h5(Path(storage_location)/dataset_keyword/"point_0.h5", only_distributed_data=False)
state_info  = data_generator.state_info
n_snapshots = data["samples"]["converged"].size


def read_snapshots_into_state_format(states_dset, state_info, n_columns):
    n_local_states = dafoam_instance.getNLocalAdjointStates()
    snapshot_data  = np.zeros((n_local_states, n_columns)) if n_columns > 1 else np.zeros((n_local_states, ))
    for state_var, info in state_info.items():
        idx = info["indices"]
        if n_columns > 1:
            snapshot_data[idx, :] = states_dset[state_var]
        else:
            snapshot_data[idx] = states_dset[state_var]

    return snapshot_data


base_data = read_snapshots_into_state_format(data["samples"]["states"], state_info, n_snapshots)
weights   = read_snapshots_into_state_format(data["pod"]["weights"], state_info, 1)

# Build a flat list of FD Jacobian columns: one (n_y, n_snapshots) array per DV DOF.
# Skipping angle_of_attack_deg (primary variable, not a shape DV).
eps = 1e-6
J_list = []  # shape per entry: (n_local_states, n_snapshots)

for dv_key, dv_group in data["perturbations"].items():
    if dv_key in ("_attrs", "angle_of_attack_deg"):
        continue
    for dof_key, dof_group in dv_group.items():
        if dof_key == "_attrs":
            continue
        perturbed = read_snapshots_into_state_format(dof_group["states"], state_info, n_snapshots)
        J_list.append(1.0 / eps * (perturbed - base_data))  # (n_y, n_snapshots)

n_dv = len(J_list)  # expect 6 (3 camber + 3 thickness DOFs)

# ---- FD signal-level diagnostic ----
# delta_y = eps * J, so ||delta_y||_2 = eps * ||J_col||_2.
# If this is comparable to the solver tolerance * sqrt(n_dof), the FD is dominated by noise.
# Also check Jacobian column correlation: uniform cross-correlation suggests all DVs
# produce the same state response (could be physics OR noise aliasing).
delta_sq_local = np.zeros((n_dv, n_snapshots))  # local partial ||delta_y||^2
for j, J_dv in enumerate(J_list):
    delta_sq_local[j] = np.sum((eps * J_dv)**2, axis=0)  # (n_snapshots,)
delta_sq = comm.allreduce(delta_sq_local, op=MPI.SUM)
delta_norms = np.sqrt(delta_sq)  # (n_dv, n_snapshots): ||y_pert - y_base||_2 per DV/sample

# Cross-correlation of Jacobian columns (averaged over samples): reveals if all DVs
# produce the same state-space direction regardless of which DV is perturbed.
JtJ_local = np.zeros((n_dv, n_dv))
for i in range(n_snapshots):
    cols = np.stack([J_list[j][:, i] for j in range(n_dv)], axis=1)  # (n_local, n_dv)
    col_norms = np.linalg.norm(cols, axis=0, keepdims=True) + 1e-300
    cols_unit = cols / col_norms
    JtJ_local += cols_unit.T @ cols_unit
JtJ = comm.allreduce(JtJ_local, op=MPI.SUM) / n_snapshots  # mean cosine similarity matrix

n_adjoint_states_total = comm.allreduce(dafoam_instance.getNLocalAdjointStates(), op=MPI.SUM)
if rank == 0:
    print("=== FD Signal-Level Diagnostic ===")
    print(f"  eps used in script: {eps:.2e}")
    print(f"  ||y_pert - y_base||_2  (signal) per DV — should be >> solver_tol * sqrt(n_dof):")
    print(f"  n_dof = {n_adjoint_states_total}")
    for j in range(n_dv):
        lo, med, hi = np.percentile(delta_norms[j], [10, 50, 90])
        print(f"    DV {j}: p10={lo:.2e}  median={med:.2e}  p90={hi:.2e}")
    print(f"\n  Mean cosine similarity between Jacobian columns (should be << 1 if DVs differ):")
    print(np.array2string(JtJ, precision=3, suppress_small=True))
    print(f"  Off-diagonal mean: {(JtJ.sum() - np.trace(JtJ)) / (n_dv*(n_dv-1)):.3f}")
    print()

# M = (1/n_snapshots) * sum_i  J_i^T  diag(w_vol)  J_i,  shape (n_dv, n_dv)
# M_unweighted uses W=I (equal DOF weight), which upweights BL cells due to fine mesh density
M          = np.zeros((n_dv, n_dv))
M_unweighted = np.zeros((n_dv, n_dv))
n_local = dafoam_instance.getNLocalAdjointStates()
J_loc = np.zeros((n_local, n_dv))

for i in range(n_snapshots):
    for j, J_dv in enumerate(J_list):
        J_loc[:, j] = J_dv[:, i]
    M            += J_loc.T @ (weights[:, None] * J_loc)
    M_unweighted += J_loc.T @ J_loc

M            = comm.allreduce(M,            op=MPI.SUM) / n_snapshots
M_unweighted = comm.allreduce(M_unweighted, op=MPI.SUM) / n_snapshots

# ---- Objective gradient metric (optional — only built if gradients are in the h5 file) ----
# M_grad = (1/n) sum_i  grad_f_i  grad_f_i^T  in R^(n_dv x n_dv)
has_gradients = "gradients" in data["samples"] and "objective_0" in data["samples"].get("gradients", {})
if rank == 0 and has_gradients:
    grad_cols = []
    for var_name, val in data["samples"]["gradients"]["objective_0"].items():
        if var_name not in ("_attrs", "angle_of_attack_deg"):
            grad_cols.append(np.asarray(val))   # (n_snapshots,) per DV
    G = np.stack(grad_cols, axis=0)             # (n_dv, n_snapshots)
    M_grad = (G @ G.T) / n_snapshots            # (n_dv, n_dv)
else:
    if rank == 0 and not has_gradients:
        print("Note: no gradients in h5 file — skipping gradient-based metric.")
    M_grad = None
M_grad = comm.bcast(M_grad, root=0)


# ===============================
# region REDUCED PULLBACK METRIC
# ===============================
# Compute M_red = (1/n) sum_i (Phi^T W J_i)^T (Phi^T W J_i)  in R^(n_dv x n_dv)
# where Phi is the W-orthonormal POD basis computed from the base snapshots.
#
# Uses the economy SVD via the Gram matrix (n_snapshots x n_snapshots), which is
# much cheaper than the full SVD when n_y >> n_snapshots.
#
# No new data is needed: Phi^T W J_k = (sqrt(W)*Phi)^T (sqrt(W)*J_k), and
# sqrt(W)*J_k is already available from J_list.

n_pod_modes = 20  # number of POD modes to retain; tune this

sqrt_W_local  = np.sqrt(weights)                          # (n_local,)
Y_w_local     = sqrt_W_local[:, None] * base_data        # sqrt(W)*Y, (n_local, n_snapshots)

# Gram matrix of sqrt(W)*Y — symmetric, cheap to allreduce at (n_snapshots, n_snapshots)
gram_local = Y_w_local.T @ Y_w_local
gram       = comm.allreduce(gram_local, op=MPI.SUM)

eigvals_g, V_g = np.linalg.eigh(gram)
order          = np.argsort(eigvals_g)[::-1]             # descending
eigvals_g      = np.maximum(eigvals_g[order], 0.0)
V_g            = V_g[:, order]

# Cumulative energy captured by the retained modes (informational)
energy_frac = np.cumsum(eigvals_g[:n_pod_modes]) / eigvals_g.sum()

# W-orthonormal modes stored as Phi_w = sqrt(W)*Phi, shape (n_local, n_pod_modes)
# Phi_w satisfies Phi_w^T Phi_w (allreduced) = I  (proven by construction)
sigma_r   = np.sqrt(eigvals_g[:n_pod_modes])
Phi_w_loc = Y_w_local @ V_g[:, :n_pod_modes] / sigma_r[None, :]  # (n_local, n_pod_modes)

# Build M_red: for each sample i, reduced Jacobian = Phi_w^T (sqrt(W)*J_full_i)
M_red     = np.zeros((n_dv, n_dv))
J_full_i  = np.zeros((n_local, n_dv))

for i in range(n_snapshots):
    for j, J_dv in enumerate(J_list):
        J_full_i[:, j] = J_dv[:, i]
    WJ_local   = sqrt_W_local[:, None] * J_full_i        # (n_local, n_dv)
    J_red_loc  = Phi_w_loc.T @ WJ_local                  # (n_pod_modes, n_dv) — local part
    J_red_i    = comm.allreduce(J_red_loc, op=MPI.SUM)   # (n_pod_modes, n_dv) — global
    M_red     += J_red_i.T @ J_red_i                     # (n_dv, n_dv)

M_red = M_red / n_snapshots

if rank == 0:
    print(f"=== Reduced pullback metric ({n_pod_modes} POD modes) ===")
    print(f"  POD energy captured: {energy_frac[-1]:.4f}")
    eigvals_red = np.linalg.eigvalsh(M_red)[::-1]
    print(f"  M_red eigenvalues: {np.array2string(eigvals_red, precision=3, suppress_small=True)}")
    print(f"  Condition number:  {eigvals_red[0] / max(eigvals_red[-1], 1e-300):.2e}")
    dom_red = np.linalg.eigh(M_red)[1][:, -1]
    print(f"  Dominant eigenvector: {np.array2string(dom_red, precision=3)}")


def pullback_distance(xi, xj, M):
    d = (xi - xj)
    return np.sqrt(d @ M @ d)


X_temp = []
for key, val in data["parameters"]["secondary_variables"].items():
    if key != "_attrs":
        X_temp.append(val.T)

X   = np.concatenate(X_temp, axis=0)


eucl_dists = []
pull_dists = []
for i in range(n_snapshots):
    eucl_dists.append(np.sqrt(np.sum((X[:,i] - X[:,i-1]) ** 2)))
    pull_dists.append(pullback_distance(X[:, i-1], X[:, i], M))

eucl_dists = np.array(eucl_dists)
pull_dists = np.array(pull_dists)


sorted_inds = np.argsort(pull_dists)



# ===============================
# region METRIC EVALUATION
# ===============================
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from matplotlib import pyplot as plt

# Normalize all metrics so trace = n_dv (same scale as identity).
def normalize_metric(M_in, n):
    return M_in * (n / np.trace(M_in))

alpha = 0.1   # regularization strength

M_norm     = normalize_metric(M,            n_dv)
M_red_norm = normalize_metric(M_red,        n_dv)
M_unw_norm = normalize_metric(M_unweighted, n_dv)

M_reg      = M_norm     + alpha * np.eye(n_dv)
M_red_reg  = M_red_norm + alpha * np.eye(n_dv)
M_unw_reg  = M_unw_norm + alpha * np.eye(n_dv)

if M_grad is not None:
    M_grad_norm = normalize_metric(M_grad, n_dv)
    M_grad_reg  = M_grad_norm + alpha * np.eye(n_dv)

def make_pullback_dist_matrix(M_metric, X):
    eigvals, eigvecs = np.linalg.eigh(M_metric)
    eigvals = np.maximum(eigvals, 0)
    Z = X.T @ (eigvecs * np.sqrt(eigvals))
    return cdist(Z, Z)

# --- Pairwise design-space distance matrices ---
D_eucl         = cdist(X.T, X.T)
D_pull         = make_pullback_dist_matrix(M_norm,     X)
D_pull_reg     = make_pullback_dist_matrix(M_reg,      X)
D_pull_red     = make_pullback_dist_matrix(M_red_norm, X)
D_pull_red_reg = make_pullback_dist_matrix(M_red_reg,  X)
D_pull_unw     = make_pullback_dist_matrix(M_unw_norm, X)
D_pull_unw_reg = make_pullback_dist_matrix(M_unw_reg,  X)

eigvals_M, eigvecs_M = np.linalg.eigh(M_norm)

# --- Pairwise state-space distance matrix via distributed Gram matrix ---
# ||y_i - y_j||_W^2 = G[i,i] + G[j,j] - 2*G[i,j],  G[i,j] = y_i^T diag(w) y_j
WY      = weights[:, None] * base_data           # (n_local, n_snapshots)
G_local = base_data.T @ WY                       # (n_snapshots, n_snapshots)
G       = comm.allreduce(G_local, op=MPI.SUM)
diag_G  = np.diag(G)
D_state = np.sqrt(np.maximum(diag_G[:, None] + diag_G[None, :] - 2 * G, 0))


# --- LOO reconstruction error (all ranks; one allreduce per test point) ---
def loo_knn_error(D_design, Y_local, W_local, comm, k=5):
    """Mean W-norm reconstruction error: predict each state from its k-NN average."""
    total = 0.0
    for i in range(D_design.shape[0]):
        row       = D_design[i].copy(); row[i] = np.inf
        nn_idx    = np.argsort(row)[:k]
        y_pred    = np.mean(Y_local[:, nn_idx], axis=1)
        diff      = Y_local[:, i] - y_pred
        sq_local  = np.sum(W_local * diff**2)
        total    += np.sqrt(comm.allreduce(sq_local, op=MPI.SUM))
    return total / D_design.shape[0]


def loo_kernel_error(D_design, Y_local, W_local, comm, sigma=None):
    """Mean W-norm reconstruction error: predict each state with Gaussian kernel weights."""
    if sigma is None:
        sigma = np.median(D_design[D_design > 0])
    total = 0.0
    for i in range(D_design.shape[0]):
        row      = D_design[i].copy(); row[i] = np.inf
        kw       = np.exp(-row**2 / (2 * sigma**2)); kw /= kw.sum()
        y_pred   = Y_local @ kw
        diff     = Y_local[:, i] - y_pred
        sq_local = np.sum(W_local * diff**2)
        total   += np.sqrt(comm.allreduce(sq_local, op=MPI.SUM))
    return total / D_design.shape[0]


metrics = {
    "Euclidean":        D_eucl,
    "Pullback":         D_pull,
    "Pullback+reg":     D_pull_reg,
    "Pullback-red":     D_pull_red,
    "Pullback-red+reg": D_pull_red_reg,
    "Pullback-unw":     D_pull_unw,
    "Pullback-unw+reg": D_pull_unw_reg,
}
if M_grad is not None:
    metrics["Pullback-grad"]     = make_pullback_dist_matrix(M_grad_norm, X)
    metrics["Pullback-grad+reg"] = make_pullback_dist_matrix(M_grad_reg,  X)
err_knn  = {k: loo_knn_error   (D, base_data, weights, comm) for k, D in metrics.items()}
err_kern = {k: loo_kernel_error(D, base_data, weights, comm) for k, D in metrics.items()}


if rank == 0:
    # --- 1. Eigenspectrum of M ---
    raw_eigvals  = np.linalg.eigvalsh(M)[::-1]
    reg_eigvals  = np.linalg.eigvalsh(M_reg)[::-1]
    print("=== Pullback Metric M ===")
    print(f"  Eigenvalues (descending): {np.array2string(raw_eigvals, precision=3, suppress_small=True)}")
    print(f"  Condition number:         {raw_eigvals[0] / max(raw_eigvals[-1], 1e-300):.2e}")
    print(f"  Trace:                    {np.trace(M):.4e}")

    # dominant eigenvector: which DV drives the state the most?
    dom_vec = eigvecs_M[:, -1]   # eigh returns ascending order; last = largest
    print(f"\n  Dominant eigenvector (DV loadings): {np.array2string(dom_vec, precision=3)}")
    print(f"  (index of max loading: DV {np.argmax(np.abs(dom_vec))})")

    # DV ranges: uniform dominant eigenvector could be a scale artifact
    print("\n  DV ranges (check for scale mismatch):")
    for k in range(n_dv):
        lo, hi = X[k].min(), X[k].max()
        print(f"    DV {k}: [{lo:.4g}, {hi:.4g}]  (span {hi-lo:.4g})")

    # Eigenspectrum comparison across all metrics
    metric_M_dict = {
        "Pullback (W=vol)": M_norm,
        "Pullback (W=I)":   M_unw_norm,
        "Pullback-reduced": M_red_norm,
    }
    if M_grad is not None:
        metric_M_dict["Pullback-grad"] = M_grad_norm
    print(f"\n=== Metric eigenspectra (normalized, alpha={alpha}) ===")
    for name, Mm in metric_M_dict.items():
        ev = np.linalg.eigvalsh(Mm)[::-1]
        cond = ev[0] / max(ev[-1], 1e-300)
        dom = np.linalg.eigh(Mm)[1][:, -1]
        print(f"  {name}:")
        print(f"    eigvals: {np.array2string(ev, precision=3, suppress_small=True)}")
        print(f"    cond: {cond:.2e}   dom vec: {np.array2string(dom, precision=3)}")

    # --- 2a. Global Spearman rank correlation ---
    upper = np.triu_indices(n_snapshots, k=1)
    print("\n=== Global Spearman ρ(d_design, d_state)  [higher is better] ===")
    for name, D in metrics.items():
        rho, _ = spearmanr(D[upper], D_state[upper])
        print(f"  {name:<25}: {rho:.4f}")

    # --- 2b. Per-point Spearman ---
    def per_point_spearman(D_design, D_state_mat):
        n = D_design.shape[0]
        rhos = []
        for i in range(n):
            mask = np.ones(n, bool); mask[i] = False
            r, _ = spearmanr(D_design[i, mask], D_state_mat[i, mask])
            if not np.isnan(r):
                rhos.append(r)
        return np.mean(rhos), np.std(rhos)

    print("\n=== Per-point Spearman ρ (mean ± std)  [higher is better] ===")
    for name, D in metrics.items():
        mu, sd = per_point_spearman(D, D_state)
        print(f"  {name:<25}: {mu:.4f} ± {sd:.4f}")

    # --- 3. LOO reconstruction errors ---
    print("\n=== LOO k-NN reconstruction error (k=5)  [lower is better] ===")
    for name in metrics:
        print(f"  {name:<25}: {err_knn[name]:.4e}")

    print("\n=== LOO Gaussian-kernel reconstruction error (sigma = median)  [lower is better] ===")
    for name in metrics:
        print(f"  {name:<25}: {err_kern[name]:.4e}")

    # --- Plots ---
    D_state_norm        = D_state / np.max(D_state)
    D_eucl_norm         = D_eucl  / np.max(D_eucl)
    D_pull_norm         = D_pull / np.max(D_pull)
    D_pull_reg_norm     = D_pull_reg  / np.max(D_pull_reg)
    D_pull_red_norm     = D_pull_red / np.max(D_pull_red)
    D_pull_red_reg_norm = D_pull_red_reg / np.max(D_pull_red_reg)
    D_pull_unw_norm     = D_pull_unw / np.max(D_pull_unw)
    D_pull_unw_reg_norm = D_pull_unw_reg / np.max(D_pull_unw_reg)

    rel_eucl              = (D_eucl_norm         - D_state_norm) / D_state_norm
    rel_pull              = (D_pull_norm         - D_state_norm) / D_state_norm
    rel_pull_reg          = (D_pull_reg_norm     - D_state_norm) / D_state_norm
    rel_pull_red          = (D_pull_red_norm     - D_state_norm) / D_state_norm
    rel_pull_red_reg      = (D_pull_red_reg_norm - D_state_norm) / D_state_norm
    rel_pull_unw_norm     = (D_pull_unw          - D_state_norm) / D_state_norm
    rel_pull_unw_reg_norm = (D_pull_unw_reg      - D_state_norm) / D_state_norm

    import numpy.ma as ma

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    axes[0].semilogy(range(1, n_dv + 1), raw_eigvals,  'o-', label='raw')
    axes[0].semilogy(range(1, n_dv + 1), reg_eigvals,  's--', label=f'reg α={alpha}')
    axes[0].set_xlabel("Index"); axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("M eigenspectrum"); axes[0].legend(); axes[0].grid(True)

    # masks (True = hidden)
    upper_mask = np.tril(np.ones_like(D_state_norm, dtype=bool), k=-1)  # hides lower
    lower_mask = np.triu(np.ones_like(D_state_norm, dtype=bool), k=0)   # hides upper

    def dual_pcolor(ax, magnitude, rel_diff, title, alpha_label=None, clims=None):
        if clims is None:
            clims = [-1, 1]

        # Colormap for relative error
        cmap_err = plt.cm.RdBu_r.copy()
        cmap_err.set_bad(color='green')   # NaN / masked values shown in green

        # Mask triangle regions
        mag_masked = ma.masked_array(magnitude, mask=upper_mask)

        # Mask lower triangle + invalid values
        rel_mask = lower_mask | ~np.isfinite(rel_diff)
        rel_masked = ma.masked_array(rel_diff, mask=rel_mask)

        im_mag = ax.pcolor(
            mag_masked,
            cmap='viridis',
            vmin=0,
            vmax=1
        )

        im_err = ax.pcolor(
            rel_masked,
            cmap=cmap_err,
            vmin=clims[0],
            vmax=clims[1]
        )

        plt.colorbar(im_mag, ax=ax, label='Distance metric (normalized)')
        plt.colorbar(im_err, ax=ax, label='Relative diff')

        ax.set_title(title)

    
    rel_list = [rel_eucl, rel_pull, rel_pull_reg]

    max_val = max(
        np.nanmax(np.abs(rel)[np.isfinite(rel)])
        for rel in rel_list
    )

    color_lims = [-1, 1]
    dual_pcolor(axes[1], D_state_norm, np.zeros_like(D_state_norm), "State distance (upper) / self (lower)", clims=color_lims)
    dual_pcolor(axes[2], D_eucl_norm, rel_eucl,     "Upper: Rel. diff | Lower: Eucl dist.", clims=color_lims)
    dual_pcolor(axes[3], D_pull_norm, rel_pull,     "Upper: Rel. diff | State pullback dist.", clims=color_lims)
    dual_pcolor(axes[4], D_pull_red_norm, rel_pull_red,     "Upper: Rel. diff | Reduced state pullback dist.", clims=color_lims)
    # dual_pcolor(axes[4], D_pull_reg_norm, rel_pull_reg, f"Upper: Pullback reg (α={alpha}) rel. diff", clims=color_lims)

    # upper = np.triu(D_state_norm, k=0)
    # lower = np.tril(D_state_norm,  k=-1)
    # D_state_D_state = upper + lower
    # im1 = axes[1].pcolor(D_state_D_state,    cmap='viridis'); plt.colorbar(im1, ax=axes[1])
    # axes[1].set_title("State distance  ||y_i - y_j||_W")
    # im1.set_clim(0, 1)

    # upper = np.triu(D_state_norm, k=0)
    # lower = np.tril((D_eucl_norm - D_state_norm) / D_state_norm,  k=-1)
    # D_state_D_eucl = upper + lower
    # im2 = axes[2].pcolor(D_state_D_eucl,     cmap='viridis'); plt.colorbar(im2, ax=axes[2])
    # axes[2].set_title("Euclidean design distance")
    # im2.set_clim(0, 1)

    # upper = np.triu(D_state_norm, k=0)
    # lower = np.tril((D_pull_norm - D_state_norm) / D_state_norm,  k=-1)
    # D_state_D_pull = upper + lower
    # im3 = axes[3].pcolor(D_state_D_pull,     cmap='viridis'); plt.colorbar(im3, ax=axes[3])
    # axes[3].set_title("Pullback design distance")
    # im3.set_clim(0, 1)

    # upper = np.triu(D_state_norm, k=0)
    # lower = np.tril((D_pull_reg_norm - D_state_norm) / D_state_norm,  k=-1)
    # D_state_D_pull_reg = upper + lower
    # im4 = axes[4].pcolor(D_state_D_pull_reg, cmap='viridis'); plt.colorbar(im4, ax=axes[4])
    # axes[4].set_title(f"Pullback (reg, α={alpha})")
    # im4.set_clim(0, 1)

    plt.tight_layout()
    plt.savefig("pullback_metric_eval.png", dpi=150)
    plt.show()








