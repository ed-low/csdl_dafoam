# ===============================
# region PACKAGES
# ===============================
import numpy as np
import os
import pickle
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
from csdl_dafoam.core.rom.rom_solver import BroydenNewtonSolver, NewtonSolver

from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

import faulthandler
faulthandler.enable()
os.environ["PETSC_OPTIONS"] = "-malloc_debug"


# ===============================
# region USER INPUT
# ===============================
problem_name              = 'training_data'

geometry_directory        = os.path.join(os.getcwd(), 'airfoil_geometry/')
stp_file_name             = 'airfoil_transonic_unitspan_2.stp'
geometry_pickle_file_name = 'airfoil_stored_refit.pickle'

comm           = MPI.COMM_WORLD
TIMING_ENABLED = True

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

dataset_keyword  = "training_set_with_perturbations_300" #'training_set_with_grad' #
storage_location = Path(dafoam_directory)

# ROM / metric optionss
n_retained_modes   = 20
alpha_reg          = 0.1    # regularization strength added to normalized pullback metrics
COMPUTE_PROJ_ERROR = True  # set True to also record projection errors per metric
num_samples        = 8      # LHS test points (reference point always prepended)

# Distance metrics used for RBF snapshot weighting.
# Each metric defines a different coordinate transform L such that
# d_metric(xi, xj) = ||L^T (xi - xj)||_2.
METRIC_NAMES = ["unweighted", "euclidean", "pullback", "pullback_red"] #"pullback_reg", "pullback_red", "pullback_red_reg"]


# ===============================
# region SETUP
# ===============================
rank      = comm.Get_rank()
comm_size = comm.Get_size()
rank_str  = f"{rank:0{len(str(comm_size-1))}d}"

dafoam_instance     = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)
dafoam_instance_rom = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)

if rank == 0:
    print_runscript_info()

# All metric variants share one ROM instance — CSDL inline evaluation is sequential,
# so only one ROM solve is active at a time.  The solution_prefix differentiates
# each variant's OpenFOAM output directories.
dafoam_instances_rom = {name: dafoam_instance_rom for name in METRIC_NAMES}

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


h5_path = Path(storage_location) / dataset_keyword / "point_0.h5"

# import h5py
# with h5py.File(h5_path, "r+") as f:
#     grp = f["perturbations"]
#     grp.move("normalized_camber_dof", "normalized_percent_camber_change_dof")
#     grp.move("normalized_thickness_dof", "percent_change_in_thickness_dof")
#     print("Renamed datasets:", list(grp.keys()))

data_generator = TrainingDataInterface(
    dafoam_instance=dafoam_instance,
    storage_location=storage_location,
    dataset_keyword=dataset_keyword,
    h5_file_base_name="point"
)
data        = data_generator.load_h5(Path(storage_location) / dataset_keyword / "point_0.h5", only_distributed_data=False)
state_info  = data_generator.state_info
n_snapshots = data["samples"]["converged"].size
if rank == 0:
    n_pod_modes = data["pod"]["singular_values"].shape[0]
    print(f"\n  Training snapshots (incl. reference): {n_snapshots}")
    print(f"  POD modes available:                  {n_pod_modes}")
    print(f"  Effective rank of snapshot matrix:    {n_snapshots - 1}  (reference excluded)")
    print(f"  n_retained_modes:                     {n_retained_modes}")


# ===============================
# region PULLBACK METRIC COMPUTATION
# ===============================
def read_snapshots_into_state_format(states_dset, state_info, n_columns):
    n_local = dafoam_instance.getNLocalAdjointStates()
    out = np.zeros((n_local, n_columns)) if n_columns > 1 else np.zeros((n_local,))
    for state_var, info in state_info.items():
        idx = info["indices"]
        if n_columns > 1:
            out[idx, :] = states_dset[state_var]
        else:
            out[idx] = states_dset[state_var]
    return out


base_data = read_snapshots_into_state_format(data["samples"]["states"], state_info, n_snapshots)
weights   = read_snapshots_into_state_format(data["pod"]["weights"], state_info, 1)

# FD Jacobians from perturbation data (one (n_local, n_snapshots) array per DV DOF)
eps    = 1e-6
J_list = []
for dv_key, dv_group in data["perturbations"].items():
    if dv_key in ("_attrs", "angle_of_attack_deg"):
        continue
    for dof_key, dof_group in dv_group.items():
        if dof_key == "_attrs":
            continue
        perturbed = read_snapshots_into_state_format(dof_group["states"], state_info, n_snapshots)
        J_list.append((1.0 / eps) * (perturbed - base_data))

n_dv = len(J_list)

# FD signal-level diagnostic
delta_sq_local = np.zeros((n_dv, n_snapshots))
for j, J_dv in enumerate(J_list):
    delta_sq_local[j] = np.sum((eps * J_dv) ** 2, axis=0)
delta_sq    = comm.allreduce(delta_sq_local, op=MPI.SUM)
delta_norms = np.sqrt(delta_sq)

JtJ_local = np.zeros((n_dv, n_dv))
for i in range(n_snapshots):
    cols      = np.stack([J_list[j][:, i] for j in range(n_dv)], axis=1)
    col_norms = np.linalg.norm(cols, axis=0, keepdims=True) + 1e-300
    JtJ_local += (cols / col_norms).T @ (cols / col_norms)
JtJ = comm.allreduce(JtJ_local, op=MPI.SUM) / n_snapshots

n_adj_total = comm.allreduce(dafoam_instance.getNLocalAdjointStates(), op=MPI.SUM)
if rank == 0:
    print("=== FD Signal-Level Diagnostic ===")
    print(f"  eps: {eps:.2e},  n_dof: {n_adj_total}")
    for j in range(n_dv):
        lo, med, hi = np.percentile(delta_norms[j], [10, 50, 90])
        print(f"    DV {j}: p10={lo:.2e}  median={med:.2e}  p90={hi:.2e}")
    print(f"  Off-diagonal cosine mean: {(JtJ.sum() - np.trace(JtJ)) / (n_dv * (n_dv - 1)):.3f}")

# Full pullback metric M = (1/n) sum_i J_i^T diag(w) J_i
n_local_states = dafoam_instance.getNLocalAdjointStates()
M            = np.zeros((n_dv, n_dv))
M_unweighted = np.zeros((n_dv, n_dv))
J_loc        = np.zeros((n_local_states, n_dv))

for i in range(n_snapshots):
    for j, J_dv in enumerate(J_list):
        J_loc[:, j] = J_dv[:, i]
    M            += J_loc.T @ (weights[:, None] * J_loc)
    M_unweighted += J_loc.T @ J_loc

M            = comm.allreduce(M,            op=MPI.SUM) / n_snapshots
M_unweighted = comm.allreduce(M_unweighted, op=MPI.SUM) / n_snapshots

# Reduced pullback metric via POD Gram matrix (cheap: n_snapshots x n_snapshots)
sqrt_W    = np.sqrt(weights)
Y_w       = sqrt_W[:, None] * base_data
gram_loc  = Y_w.T @ Y_w
gram      = comm.allreduce(gram_loc, op=MPI.SUM)

eigvals_g, V_g = np.linalg.eigh(gram)
order          = np.argsort(eigvals_g)[::-1]
eigvals_g      = np.maximum(eigvals_g[order], 0.0)
V_g            = V_g[:, order]

sigma_r   = np.sqrt(eigvals_g[:n_retained_modes])
Phi_w_loc = Y_w @ V_g[:, :n_retained_modes] / sigma_r[None, :]   # (n_local, n_retained_modes), W-orthonormal

M_red    = np.zeros((n_dv, n_dv))
J_full_i = np.zeros((n_local_states, n_dv))
for i in range(n_snapshots):
    for j, J_dv in enumerate(J_list):
        J_full_i[:, j] = J_dv[:, i]
    WJ_local  = sqrt_W[:, None] * J_full_i
    J_red_loc = Phi_w_loc.T @ WJ_local
    J_red_i   = comm.allreduce(J_red_loc, op=MPI.SUM)
    M_red    += J_red_i.T @ J_red_i
M_red /= n_snapshots

# Gradient-based metric (optional, only if gradients stored in h5)
has_gradients = "gradients" in data["samples"] and "-L/D" in data["samples"].get("gradients", {})
if rank == 0 and has_gradients:
    grad_cols = [
        np.asarray(val)
        for var_name, val in data["samples"]["gradients"]["objective_0"].items()
        if var_name not in ("_attrs", "angle_of_attack_deg")
    ]
    G_mat  = np.stack(grad_cols, axis=0)   # (n_dv, n_snapshots)
    M_grad = (G_mat @ G_mat.T) / n_snapshots
else:
    if rank == 0 and not has_gradients:
        print("Note: no gradients in h5 — skipping gradient-based metric.")
    M_grad = None
M_grad = comm.bcast(M_grad, root=0)

# Normalize (trace = n_dv) and regularize
def normalize_metric(M_in, n):
    return M_in * (n / np.trace(M_in))

M_norm     = normalize_metric(M,     n_dv)
M_red_norm = normalize_metric(M_red, n_dv)
M_reg      = M_norm     + alpha_reg * np.eye(n_dv)
M_red_reg  = M_red_norm + alpha_reg * np.eye(n_dv)

# Coordinate transform L such that d_metric(xi, xj) = ||L^T (xi - xj)||_2
# For metric M = V diag(lambda) V^T, L = V diag(sqrt(lambda)).
def metric_to_L(M_mat):
    eigvals, eigvecs = np.linalg.eigh(M_mat)
    return eigvecs * np.sqrt(np.maximum(eigvals, 0.0))   # (n_dv, n_dv)

metric_L = {
    "unweighted":      None,
    "euclidean":       np.eye(n_dv),
    "pullback":        metric_to_L(M_norm),
    "pullback_reg":    metric_to_L(M_reg),
    "pullback_red":    metric_to_L(M_red_norm),
    "pullback_red_reg": metric_to_L(M_red_reg),
}

if rank == 0:
    print(f"\n=== Metric eigenspectra (normalized, alpha_reg={alpha_reg}) ===")
    named_metrics = {
        "Pullback (W=vol)": M_norm,
        "Pullback (W=I)":   normalize_metric(M_unweighted, n_dv),
        "Pullback-reduced": M_red_norm,
    }
    if M_grad is not None:
        named_metrics["Pullback-grad"] = normalize_metric(M_grad, n_dv)
    for name, Mm in named_metrics.items():
        ev  = np.linalg.eigvalsh(Mm)[::-1]
        dom = np.linalg.eigh(Mm)[1][:, -1]
        print(f"  {name}:  cond={ev[0]/max(ev[-1], 1e-300):.2e}  dom={np.array2string(dom, precision=3)}")


# ===============================
# region METRIC EVALUATION (no ROM)
# ===============================
# Parameter matrix: shape (n_snapshots, n_dv)
snapshot_configs_full = np.concatenate([
    data["parameters"]["secondary_variables"]["normalized_percent_camber_change_dof"],
    data["parameters"]["secondary_variables"]["percent_change_in_thickness_dof"]
], axis=1)   # (n_snapshots, n_dv)


def make_dist_matrix(L_mat, X_ns_nv):
    """Pairwise distances under the metric defined by L: d(xi,xj) = ||L^T(xi-xj)||."""
    Z = X_ns_nv @ L_mat   # (n_snapshots, n_dv)
    return cdist(Z, Z)


D_metrics = {name: make_dist_matrix(metric_L[name], snapshot_configs_full) if name != "unweighted" else None for name in METRIC_NAMES}

# State-space pairwise distances via W-weighted Gram matrix (distributed)
WY      = weights[:, None] * base_data
G_local = base_data.T @ WY
G_state = comm.allreduce(G_local, op=MPI.SUM)
diag_G  = np.diag(G_state)
D_state = np.sqrt(np.maximum(diag_G[:, None] + diag_G[None, :] - 2 * G_state, 0.0))


def loo_knn_error(D_design, Y_local, W_local, comm, k=5):
    """Mean W-norm LOO error: predict each state from its k-NN average."""
    total = 0.0
    for i in range(D_design.shape[0]):
        row    = D_design[i].copy(); row[i] = np.inf
        nn_idx = np.argsort(row)[:k]
        y_pred = np.mean(Y_local[:, nn_idx], axis=1)
        diff   = Y_local[:, i] - y_pred
        total += np.sqrt(comm.allreduce(np.sum(W_local * diff ** 2), op=MPI.SUM))
    return total / D_design.shape[0]


def loo_kernel_error(D_design, Y_local, W_local, comm, sigma=None):
    """Mean W-norm LOO error: predict each state with Gaussian kernel weights."""
    if sigma is None:
        sigma = np.median(D_design[D_design > 0])
    total = 0.0
    for i in range(D_design.shape[0]):
        row    = D_design[i].copy(); row[i] = np.inf
        kw     = np.exp(-row ** 2 / (2 * sigma ** 2)); kw /= kw.sum()
        y_pred = Y_local @ kw
        diff   = Y_local[:, i] - y_pred
        total += np.sqrt(comm.allreduce(np.sum(W_local * diff ** 2), op=MPI.SUM))
    return total / D_design.shape[0]


def per_point_spearman(D_design, D_state_mat):
    n    = D_design.shape[0]
    rhos = []
    for i in range(n):
        mask = np.ones(n, bool); mask[i] = False
        r, _ = spearmanr(D_design[i, mask], D_state_mat[i, mask])
        if not np.isnan(r):
            rhos.append(r)
    return np.mean(rhos), np.std(rhos)


err_knn         = {name: loo_knn_error   (D_metrics[name], base_data, weights, comm) if name != "unweighted" else 0 for name in METRIC_NAMES}
err_kern        = {name: loo_kernel_error(D_metrics[name], base_data, weights, comm) if name != "unweighted" else 0 for name in METRIC_NAMES}
upper_tri       = np.triu_indices(n_snapshots, k=1)
spearman_global = {name: spearmanr(D_metrics[name][upper_tri], D_state[upper_tri])[0] if name != "unweighted" else 0 for name in METRIC_NAMES}
spearman_pp     = {name: per_point_spearman(D_metrics[name], D_state) if name != "unweighted" else (0, 0) for name in METRIC_NAMES}

if rank == 0:
    print("\n=== Distance Metric Evaluation (no ROM) ===")
    print(f"  {'Metric':<22} {'LOO-kNN':>12} {'LOO-kernel':>12} {'Spearman-global':>16} {'Spearman-pp':>14}")
    for name in METRIC_NAMES:
        mu, sd = spearman_pp[name]
        print(f"  {name:<22} {err_knn[name]:>12.4e} {err_kern[name]:>12.4e} "
              f"{spearman_global[name]:>16.4f} {mu:>10.4f}±{sd:.4f}")


# ===============================
# region CSDL RECORDER
# ===============================
recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()

geometry = lsdo_geo.import_geometry(stp_file_path, parallelize=False)

# Surface mesh projection (cached to pickle)
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

# Parameterization
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

# Thickness
original_block_thickness    = ffd_block.coefficients.value[0, 0, 1, 2] - ffd_block.coefficients.value[0, 0, 0, 2]
percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[1:-1, 0], percent_change_in_thickness_dof)
percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[1:-1, 1], percent_change_in_thickness_dof)
delta_block_thickness       = (percent_change_in_thickness / 100) * original_block_thickness
ffd_coefficients = ffd_coefficients.set(csdl.slice[:, :, 1, 2], ffd_coefficients[:, :, 1, 2] + delta_block_thickness / 2)
ffd_coefficients = ffd_coefficients.set(csdl.slice[:, :, 0, 2], ffd_coefficients[:, :, 0, 2] - delta_block_thickness / 2)

# Camber
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

# Flight conditions
flight_conditions_group                     = csdl.VariableGroup()
flight_conditions_group.airspeed_m_s        = csdl.Variable(
    value=data["parameters"]["non_sampled_variables"]["airspeed_m_s"], name="airspeed_m_s")
flight_conditions_group.angle_of_attack_deg = csdl.Variable(
    value=data["parameters"]["primary_variables"]["angle_of_attack_deg"], name="angle_of_attack_deg"
)
flight_conditions_group.altitude_m          = csdl.Variable(
    value=data["parameters"]["non_sampled_variables"]["altitude (m)"], name="altitude (m)"
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

    # POD data assembled into DAFoam's cell-interleaved state-vector ordering
    s_vals          = data["pod"]["singular_values"]
    n_local         = dafoam_instance.getNLocalAdjointStates()
    pod_modes       = np.zeros((n_local, s_vals.shape[0]))
    scaling         = np.zeros(n_local)
    reference_state = np.zeros(n_local)
    snapshots_rbf   = np.zeros((n_local, s_vals.shape[0]))

    for state_var, info in state_info.items():
        idx = info["indices"]
        pod_modes[idx, :]    = data["pod"]["modes"][state_var]
        scaling[idx]         = data["pod"]["scaling"][state_var]
        weights[idx]         = data["pod"]["weights"][state_var]
        reference_state[idx] = data["pod"]["reference_state"][state_var]

        if state_var == "phi":
            # POD modes were computed from phi normalized by per-snapshot face areas
            # (rho*U*A_face_j per snapshot j), not the reference face areas stored in
            # scaling[phi_idx] = rho*U*A_face_ref.  Pre-correct the snapshots so that
            # (snapshots_rbf - ref) / scaling gives the same quantity as in the POD SVD.
            phi_raw  = data["samples"]["states"][state_var][:, 1:]      # (n_phi, n_rbf_snaps)
            n_rbf    = phi_raw.shape[1]
            fa_ref   = np.abs(data["samples"]["mesh"]["face_areas"][:, 0:1])       # (n_phi, 1)
            fa_snap  = np.abs(data["samples"]["mesh"]["face_areas"][:, 1:n_rbf+1]) # (n_phi, n_rbf_snaps)
            fa_snap  = np.where(fa_snap < 1e-300, fa_ref, fa_snap)
            phi_ref  = reference_state[idx, None]
            snapshots_rbf[idx, :] = phi_ref + (phi_raw - phi_ref) * (fa_ref / fa_snap)
        else:
            snapshots_rbf[idx, :] = data["samples"]["states"][state_var][:, 1:]
        

    # --- POD Basis Diagnostics (pre-weighting) ---
    if rank == 0:
        print("\n" + "-"*60)
        print("  POD Basis Diagnostics (unweighted)")
        print("-"*60)

        # Singular value decay and cumulative energy
        energy     = s_vals**2
        cum_energy = np.cumsum(energy) / np.sum(energy)
        n_modes    = len(s_vals)
        print(f"\n  {'Mode':>5}  {'s_i/s_0':>10}  {'cum_energy':>12}")
        for i in range(n_modes):
            print(f"  {i:>5}  {s_vals[i]/s_vals[0]:>10.4e}  {cum_energy[i]:>12.6f}")

    # Basis orthogonality checks
    # Modes live in scaled space; W = diag(weights) is the POD inner product
    PhiTPhi  = comm.allreduce(pod_modes.T @ pod_modes,                     op=MPI.SUM)
    PhiTWPhi = comm.allreduce(pod_modes.T @ (weights[:, None] * pod_modes), op=MPI.SUM)
    orth_err   = np.linalg.norm(PhiTPhi  - np.eye(PhiTPhi.shape[0]),  'fro')
    orth_W_err = np.linalg.norm(PhiTWPhi - np.eye(PhiTWPhi.shape[0]), 'fro')
    if rank == 0:
        print(f"\n  Basis ortho ‖ΦᵀΦ - I‖_F:   {orth_err:.4e}  (unweighted, expected large)")
        print(f"  Basis ortho ‖ΦᵀWΦ - I‖_F:   {orth_W_err:.4e}  (W = diag(weights), expected ~0)")

    # Per-variable contribution to ‖ΦᵀWΦ - I‖_F to locate the source of any discrepancy
    if rank == 0:
        print(f"\n  Per-variable ‖ΦᵀWΦ - I‖_F contribution:")
        print(f"  {'Variable':>10}  {'orth_W_err':>12}  {'weight_min':>12}  {'weight_max':>12}")
    for state_var, info in state_info.items():
        idx      = info["indices"]
        Phi_v    = pod_modes[idx, :]
        w_v      = weights[idx]
        PhiTWPhi_v = comm.allreduce(Phi_v.T @ (w_v[:, None] * Phi_v), op=MPI.SUM)
        err_v      = np.linalg.norm(PhiTWPhi_v - np.eye(PhiTWPhi_v.shape[0]), 'fro')
        w_min      = comm.allreduce(w_v.min(), op=MPI.MIN)
        w_max      = comm.allreduce(w_v.max(), op=MPI.MAX)
        if rank == 0:
            print(f"  {state_var:>10}  {err_v:>12.4e}  {w_min:>12.4e}  {w_max:>12.4e}")

    # Per-variable snapshot reconstruction error.
    # alpha = Phi^T W z must be computed from the FULL state (all variables), not per-variable.
    # Using only one variable's rows gives a partial projection that misses cross-variable modal
    # contributions, leading to artificially large errors even when the basis is correct.
    n_snaps = snapshots_rbf.shape[1]
    alpha_global = comm.allreduce(
        pod_modes.T @ ((weights / scaling)[:, None] * (snapshots_rbf - reference_state[:, None])),
        op=MPI.SUM
    )                                                          # (n_modes, n_snaps)

    # Per-variable cumulative energy fraction vs. truncation rank.
    # Shows how quickly each variable's snapshot content is captured as modes are added.
    rank_checkpoints = [rc for rc in [1, 5, 10, 20, 30, 50, 70, 100] if rc <= n_snaps]
    per_var_cum_frac = {}
    for state_var, info in state_info.items():
        idx        = info["indices"]
        phys_diffs = snapshots_rbf[idx, :] - reference_state[idx, None]
        total_sq   = comm.allreduce(np.sum(phys_diffs ** 2), op=MPI.SUM)
        fracs = []
        for rc in rank_checkpoints:
            recon_r  = scaling[idx, None] * (pod_modes[idx, :rc] @ alpha_global[:rc, :])
            err_sq_r = comm.allreduce(np.sum((phys_diffs - recon_r) ** 2), op=MPI.SUM)
            fracs.append(1.0 - err_sq_r / max(total_sq, 1e-300))
        per_var_cum_frac[state_var] = fracs

    if rank == 0:
        print(f"\n  Per-variable cumulative energy fraction vs. truncation rank:")
        print(f"  (fraction of variable's snapshot energy captured by first r modes)")
        header_ce = f"  {'Variable':>10}"
        for rc in rank_checkpoints:
            header_ce += f"  {'r='+str(rc):>8}"
        print(header_ce)
        for state_var, fracs in per_var_cum_frac.items():
            row_ce = f"  {state_var:>10}"
            for f in fracs:
                row_ce += f"  {f:>8.4f}"
            print(row_ce)

    if rank == 0:
        print(f"\n  Snapshot reconstruction error (mean/max relative L2 per variable):")
        print(f"  {'Variable':>10}  {'mean_rel_err':>14}  {'max_rel_err':>13}")
    for state_var, info in state_info.items():
        idx        = info["indices"]
        phys_diffs = snapshots_rbf[idx, :] - reference_state[idx, None]
        recon_phys = scaling[idx, None] * (pod_modes[idx, :] @ alpha_global)
        err_sq     = comm.allreduce(np.sum((phys_diffs - recon_phys)**2, axis=0), op=MPI.SUM)
        norm_sq    = comm.allreduce(np.sum(phys_diffs**2,                axis=0), op=MPI.SUM)
        rel_err    = np.sqrt(err_sq / np.maximum(norm_sq, 1e-30))
        if rank == 0:
            print(f"  {state_var:>10}  {rel_err.mean():>14.4e}  {rel_err.max():>13.4e}")
    if rank == 0:
        print("-"*60 + "\n")
    # --- End POD Basis Diagnostics ---

    # Temperature residual rescaling (improves LSPG conditioning)
    residual_scaling                              = np.ones_like(dafoam_instance.getStateWeights())
    residual_scaling[state_info["T"]["indices"]] *= 1005

    # Snapshot data for weighted POD.
    # Index 0 is the reference condition (already captured in reference_state), so skip it.
    # snapshots_rbf        = np.concatenate(
    #     [data["samples"]["states"][sv] for sv in state_info.keys()], axis=0
    # )[:, 1:]                                                  # (n_local, n_snapshots-1)
    snapshot_configs_rbf = np.concatenate([
        data["parameters"]["secondary_variables"]["normalized_percent_camber_change_dof"],
        data["parameters"]["secondary_variables"]["percent_change_in_thickness_dof"]
    ], axis=1)[1:, :]                                         # (n_snapshots-1, n_dv)

    # VT = Sigma^{-1} U^T diag(w/sigma) (Y - y_ref): metric-independent, computed once
    VT = (1.0 / s_vals[:, None]) * comm.allreduce(
        pod_modes.T @ ((weights / scaling)[:, None] * (snapshots_rbf - reference_state[:, None])),
        op=MPI.SUM
    )

    current_config = csdl.concatenate(
        (normalized_percent_camber_change_dof, percent_change_in_thickness_dof), axis=0
    )

    # Build one weighted-POD ROM per metric
    rom_outputs  = {}
    rbf_w_csdl   = {}   # keyed by metric_name; populated inside the loop for diagnostics

    for metric_name in METRIC_NAMES:
        L = metric_L[metric_name]   # (n_dv, n_dv) numpy transform

        if L is not None:
            # Transform parameter coordinates into the metric's space.
            # Euclidean: L = I, no transform needed.
            if np.allclose(L, np.eye(n_dv)):
                z_current   = current_config
                Z_snapshots = snapshot_configs_rbf
            else:
                L_T_const   = csdl.Variable(value=L.T, name=f"L_T_{metric_name}")
                z_current   = csdl.einsum(L_T_const, current_config, action='ij,j->i')
                Z_snapshots = snapshot_configs_rbf @ L
            
            rbf_w_pre = RBFInterpolator(z_current, Z_snapshots, positive_non_reproducing_weights=True).weights()
            rbf_w     = rbf_w_pre / csdl.sum(rbf_w_pre)
            rbf_w_csdl[metric_name] = rbf_w

            # Weighted SVD: D = diag(sigma) * VT * diag(sqrt(w_rbf))
            D_svd = csdl.einsum(
                s_vals,
                csdl.einsum(VT, csdl.sqrt(rbf_w ** 2), action='ij,j->ij'),
                action='i,ij->ij'
            )
            UD, _, _ = customExplicitReducedSVD().evaluate(A=D_svd)
            pod_modes_metric = pod_modes @ UD[:, :n_retained_modes]   # (n_local, n_retained_modes) CSDL
        else:
            pod_modes_metric = pod_modes[:, :n_retained_modes]

        rom_model = DAFoamLSPGModel(
            dafoam_input_variables_group=dafoam_input_variables_group,
            pod_modes=pod_modes_metric,
            reference_fom_state=reference_state,
            scaling=scaling,
            weights=1 / residual_scaling ** 2,
            dafoam_instance=dafoam_instances_rom[metric_name],
            normalize_residuals=False,
            fd_step=1e-6,
            solution_prefix=metric_name,  
            disable_presolve_diagnostics=False          
        )
        rom_wrapper     = CSDLROMWrapper(
            model=rom_model,
            solver=BroydenNewtonSolver(options={"tol_rel": 1e-9, "tol_step_abs": 1e-13}),
            start_with_zero_state=True
        )
        rom_states      = rom_wrapper.evaluate()
        state_est       = reference_state + scaling * (pod_modes_metric @ rom_states)

        fn_model        = DAFoamFunctions(dafoam_instances_rom[metric_name], disable_jacvec_normalization=True)
        fn_outputs      = fn_model.evaluate(state_est, dafoam_input_variables_group)

        rom_outputs[metric_name] = {
            "state":     state_est,
            "functions": fn_outputs,
            "pod_modes": pod_modes_metric,
        }

        mpi_region.set_as_global_output(state_est)
        for out_name in dafoam_instance.getOption("function").keys():
            mpi_region.set_as_global_output(getattr(fn_outputs, out_name))

    # FOM solver
    dafoam_solver        = DAFoamSolver(dafoam_instance, write_residual_fields=True)
    dafoam_solver_states = dafoam_solver.evaluate(dafoam_input_variables_group)
    dafoam_fn_model      = DAFoamFunctions(dafoam_instance)
    dafoam_fn_outputs    = dafoam_fn_model.evaluate(dafoam_solver_states, dafoam_input_variables_group)

    # Projection errors (optional)
    if COMPUTE_PROJ_ERROR:
        x_scaled = (1.0 / scaling) * (dafoam_solver_states - reference_state)
        for metric_name in METRIC_NAMES:
            Phi          = rom_outputs[metric_name]["pod_modes"]
            PhiT         = Phi.T() if is_csdl(Phi) else Phi.T
            proj_coeff   = csdl.experimental.mpi.mpi_sum(PhiT @ (weights * x_scaled), comm=comm)
            proj_err_vec = x_scaled - Phi @ proj_coeff
            rom_outputs[metric_name]["proj_error"] = proj_err_vec
            mpi_region.set_as_global_output(proj_err_vec)

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

data_generator._generate_lhs_samples(snapshot_vars_and_limits, num_samples=num_samples, random_state=42)
num_samples_with_ref = num_samples + 1  # prepend reference point (all zeros)

diag_dict = {
    "err_norm":     {m: {sv: np.zeros(num_samples_with_ref) for sv in state_info.keys()} for m in METRIC_NAMES},
    "drag_rel_err": {m: np.zeros(num_samples_with_ref) for m in METRIC_NAMES},
    "lift_rel_err": {m: np.zeros(num_samples_with_ref) for m in METRIC_NAMES},
}
if COMPUTE_PROJ_ERROR:
    diag_dict["proj_err"]         = {m: np.zeros(num_samples_with_ref) for m in METRIC_NAMES}
    diag_dict["proj_err_rel"]     = {m: np.zeros(num_samples_with_ref) for m in METRIC_NAMES}
    diag_dict["proj_err_rel_var"] = {m: {sv: np.zeros(num_samples_with_ref) for sv in state_info.keys()} for m in METRIC_NAMES}

for i in range(num_samples_with_ref):
    for var, info in snapshot_vars_and_limits.items():
        sim[var] = info["samples"][i]
    sim.run()

    fom_state = dafoam_solver_states.value
    fom_drag  = dafoam_fn_outputs.drag.value
    fom_lift  = dafoam_fn_outputs.lift.value

    if rank == 0 and i == 1:   # first non-reference test point
        print(f"\n=== RBF weight diagnostics (test point {i}) ===")
        for mname, rw in rbf_w_csdl.items():
            w = np.sort(rw.value)[::-1]
            eff_n = 1.0 / (w**2).sum()  # effective number of snapshots
            print(f"  {mname:<18} top-5: {np.array2string(w[:5], precision=3)}  eff_n={eff_n:.1f}")

    for metric_name in METRIC_NAMES:
        rom_state = rom_outputs[metric_name]["state"].value
        rom_drag  = rom_outputs[metric_name]["functions"].drag.value
        rom_lift  = rom_outputs[metric_name]["functions"].lift.value

        for state_var, info in state_info.items():
            idx      = info["indices"]
            fom_var  = fom_state[idx]
            rom_var  = rom_state[idx]
            diff     = np.abs(fom_var - rom_var)
            err_norm = (
                np.sqrt(comm.allreduce(np.sum(diff ** 2), op=MPI.SUM)) /
                np.sqrt(comm.allreduce(np.sum(fom_var ** 2), op=MPI.SUM))
            )
            diag_dict["err_norm"][metric_name][state_var][i] = err_norm

        diag_dict["drag_rel_err"][metric_name][i] = abs(fom_drag - rom_drag) / (abs(fom_drag) + 1e-300)
        diag_dict["lift_rel_err"][metric_name][i] = abs(fom_lift - rom_lift) / (abs(fom_lift) + 1e-300)

        if COMPUTE_PROJ_ERROR:
            x_sc         = (1.0 / scaling) * (fom_state - reference_state)
            sol_norm     = np.sqrt(comm.allreduce(x_sc @ (weights * x_sc), op=MPI.SUM))
            proj_ev      = rom_outputs[metric_name]["proj_error"].value
            proj_err_val = np.sqrt(comm.allreduce(proj_ev @ (weights * proj_ev), op=MPI.SUM))
            diag_dict["proj_err"][metric_name][i]     = proj_err_val
            diag_dict["proj_err_rel"][metric_name][i] = proj_err_val / (sol_norm + 1e-300)
            for state_var, sv_info in state_info.items():
                idx          = sv_info["indices"]
                num          = np.sqrt(comm.allreduce(proj_ev[idx] @ (weights[idx] * proj_ev[idx]), op=MPI.SUM))
                den          = np.sqrt(comm.allreduce(x_sc[idx]    @ (weights[idx] * x_sc[idx]),    op=MPI.SUM))
                diag_dict["proj_err_rel_var"][metric_name][state_var][i] = num / (den + 1e-300)


# ===============================
# region REPORTING
# ===============================
if rank == 0:
    state_vars = list(state_info.keys())

    print("\n=== ROM Error Summary (mean over test points) ===")
    header = f"  {'Metric':<22} {'drag_rel_err':>14} {'lift_rel_err':>14}"
    for sv in state_vars:
        header += f"  {'err_norm_' + sv:>18}"
    print(header)
    for metric_name in METRIC_NAMES:
        row = (f"  {metric_name:<22}"
               f" {np.mean(diag_dict['drag_rel_err'][metric_name]):>14.4e}"
               f" {np.mean(diag_dict['lift_rel_err'][metric_name]):>14.4e}")
        for sv in state_vars:
            row += f"  {np.mean(diag_dict['err_norm'][metric_name][sv]):>18.4e}"
        print(row)

    if COMPUTE_PROJ_ERROR:
        print("\n=== Projection Error Summary (mean over test points) ===")
        print(f"  {'Metric':<22} {'proj_err_abs':>14} {'proj_err_rel':>14}")
        for metric_name in METRIC_NAMES:
            print(f"  {metric_name:<22}"
                  f" {np.mean(diag_dict['proj_err'][metric_name]):>14.4e}"
                  f" {np.mean(diag_dict['proj_err_rel'][metric_name]):>14.4e}")

        print("\n=== Per-Variable Projection Error (relative, mean over test points, unweighted basis) ===")
        header_pv = f"  {'Variable':<12}"
        for metric_name in METRIC_NAMES:
            header_pv += f"  {metric_name:>18}"
        print(header_pv)
        for state_var in state_info.keys():
            row_pv = f"  {state_var:<12}"
            for metric_name in METRIC_NAMES:
                row_pv += f"  {np.mean(diag_dict['proj_err_rel_var'][metric_name][state_var]):>18.4e}"
            print(row_pv)

    # Per-state error norm plots
    n_states = len(state_vars)
    fig, axes = plt.subplots(1, n_states, figsize=(4 * n_states, 4), sharey=False)
    if n_states == 1:
        axes = [axes]
    for ax, sv in zip(axes, state_vars):
        for metric_name in METRIC_NAMES:
            ax.plot(diag_dict["err_norm"][metric_name][sv], marker='o', label=metric_name)
        ax.set_title(sv)
        ax.set_xlabel("Test point")
        ax.set_ylabel(r"$\|w_{ROM} - w_{FOM}\| / \|w_{FOM}\|$")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("rom_metric_comparison_err_norm.png", dpi=150)

    # Drag / lift relative error plots
    fig2, (ax_drag, ax_lift) = plt.subplots(1, 2, figsize=(10, 4))
    for metric_name in METRIC_NAMES:
        ax_drag.plot(diag_dict["drag_rel_err"][metric_name], marker='o', label=metric_name)
        ax_lift.plot(diag_dict["lift_rel_err"][metric_name], marker='o', label=metric_name)
    ax_drag.set_title("Drag relative error")
    ax_drag.set_xlabel("Test point")
    ax_drag.set_ylim(bottom=0)
    ax_drag.legend(fontsize=7)
    ax_lift.set_title("Lift relative error")
    ax_lift.set_xlabel("Test point")
    ax_lift.set_ylim(bottom=0)
    ax_lift.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("rom_metric_comparison_functions.png", dpi=150)

    plt.show()
