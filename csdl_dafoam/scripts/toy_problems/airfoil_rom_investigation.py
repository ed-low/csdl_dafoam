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

J = {}
eps = 1e-6

n_dofs = 0

for dv_key, dv_group in data["perturbations"].items():
    if dv_key != "_attrs" and dv_key != "angle_of_attack_deg":
        for dof_key, dof_group in dv_group.items():
            if dof_key != "_attrs":
                J[dv_key] = {dof_key : 1 / eps * (read_snapshots_into_state_format(dof_group["states"], state_info, n_snapshots) - base_data)}
                n_dofs += 1

M = np.zeros((n_dofs, n_dofs))

for i in range(n_snapshots):
    J_loc = np.zeros((dafoam_instance.getNLocalAdjointStates(), n_dofs))
    for dv_key, dv_val in J.items():
        for j, (dof_key, dof_val) in enumerate(dv_val.items()):
            J_loc[:, j] = J[dv_key][dof_key][:, i]

    M += J_loc.T @ (weights[:, None] * J_loc)

M = comm.allreduce(M, op=MPI.SUM) / n_dofs

def pullback_distance(xi, xj, M):
    d = (xi - xj)       # (r,)
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


from matplotlib import pyplot as plt

if rank == 0:
    ###
    from scipy.spatial.distance import cdist

    # --- Euclidean distance matrix ---
    eucl_dist_matrix = cdist(X.T, X.T)  # (n_snapshots, n_snapshots)

    # --- Pullback distance matrix ---
    # d^T M d = ||L d||^2, where M = L L^T (via eigendecomposition for robustness)
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0)          # clip any negative numerics from pinv
    L = eigvecs * np.sqrt(eigvals)            # (r, r)
    X_transformed = X.T @ L                    # (n_snapshots, r)
    pullback_dist_matrix = cdist(X_transformed, X_transformed)  # (n_snapshots, n_snapshots)


    plt.pcolor(eucl_dist_matrix)
    plt.colorbar()
    plt.figure()
    plt.pcolor(pullback_dist_matrix)
    plt.colorbar()
    plt.show()













# grad_f = []
# for var_name, val in data["samples"]["gradients"]["objective_0"].items():
#     if var_name != "angle_of_attack_deg":
#         grad_f.append(val)

# G       = np.concatenate(grad_f, axis=0)
# # grad_f_mean  = np.mean(grad_f, axis=1)
# # grad_f_fluct = grad_f - grad_f_mean[:, None]

# u, s, vt = np.linalg.svd(G, full_matrices=False)

# # print(u)
# # print(s)
# # print(vt)

# eigenvalues = s**2 / G.shape[1]
# eigenvectors = u



# import matplotlib.pyplot as plt

# # plt.semilogy(range(1, len(eigenvalues)+1), eigenvalues, 'o-')
# # plt.xlabel('Index')
# # plt.ylabel('Eigenvalue')
# # # plt.axvline(x=range(G.shape[1]), linestyle='--', label='truncation')
# # plt.show()

# # cumulative_energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
# # r = np.searchsorted(cumulative_energy, 0.999) + 1

# # print(r)


# n_modes     = 20
# state_info  = data_generator.state_info
# n_local_states = dafoam_instance.getNLocalAdjointStates()

# i0 = 0
# Y   = np.zeros((n_local_states, n_snapshots))
# X_r = u[:, :3]
# W   = np.zeros((n_local_states,))

# X_temp = []
# for key, val in data["parameters"]["secondary_variables"].items():
#     if key != "_attrs":
#         X_temp.append(val.T)

# X   = np.concatenate(X_temp, axis=0)

# for state_var, info in state_info.items():
#     idx       = info["indices"] 
#     Y[idx, :] = data["samples"]["states"][state_var]
#     W[idx]    = data["pod"]["weights"][state_var]

# # Step 1: offsets
# dX = X - X[:, i0:i0+1]          # (n_x, n_s)
# dY = Y - Y[:, i0:i0+1]          # (n_y, n_s)

# # drop the reference column (zero offset)
# mask     = np.ones(X.shape[1], bool)
# mask[i0] = False
# dX, dY   = dX[:, mask], dY[:, mask]

# # Step 2: project design offsets to reduced space
# dXr = X_r.T @ dX                # (r, n_s)

# dists_from_x0 = np.linalg.norm(dX, axis=0)
# local_idx = np.argsort(dists_from_x0)[:20]
# dXr_local = dXr[:, local_idx]
# dY_local = dY[:, local_idx]

# # Step 3: snapshot kernel (already have this from POD)
# K = dY_local.T @ (W[:, None] * dY_local)    # (n_s, n_s)

# # Step 4: form the metric in reduced design space
# A = dXr_local @ dXr_local.T                 # (r, r)
# A_inv = np.linalg.pinv(A)       # use pinv for robustness
# M = A_inv @ dXr_local @ K @ dXr_local.T @ A_inv   # (r, r)

# M_norm = M * (M.shape[0] / np.trace(M))
# M = M_norm

# # Step 5: pairwise distances
# def pullback_distance(xi, xj, X_r, M):
#     d = X_r.T @ (xi - xj)       # (r,)
#     return np.sqrt(d @ M @ d)



# # ###
# # from scipy.spatial.distance import cdist

# # # Reduced coordinates for all points: shape (n_snapshots, r)
# # Z = (X_r.T @ X).T

# # # --- Euclidean distance matrix ---
# # eucl_dist_matrix = cdist(X.T, X.T)  # (n_snapshots, n_snapshots)

# # # --- Pullback distance matrix ---
# # # d^T M d = ||L d||^2, where M = L L^T (via eigendecomposition for robustness)
# # eigvals, eigvecs = np.linalg.eigh(M)
# # eigvals = np.maximum(eigvals, 0)          # clip any negative numerics from pinv
# # L = eigvecs * np.sqrt(eigvals)            # (r, r)
# # Z_transformed = Z @ L                    # (n_snapshots, r)
# # pullback_dist_matrix = cdist(Z_transformed, Z_transformed)  # (n_snapshots, n_snapshots)


# # plt.pcolor(eucl_dist_matrix)
# # plt.colorbar()
# # plt.figure()
# # plt.pcolor(pullback_dist_matrix)
# # plt.colorbar()
# # plt.show()


Y = base_data
W = weights

###
from scipy.stats import spearmanr
import numpy as np

n = X.shape[1]
d_E, d_P, d_Y = [], [], []

for i in range(n):
    for j in range(i+1, n):
        dx = X[:, i] - X[:, j]
        dy = Y[:, i] - Y[:, j]

        d_E.append(np.sqrt(dx @ dx))
        d_P.append(np.sqrt(dx @ M @ dx))  # pullback in full space
        d_Y.append(np.sqrt(dy @ (W * dy)))               # W is diagonal, shape (n_y,)

d_E, d_P, d_Y = np.array(d_E), np.array(d_P), np.array(d_Y)

rho_E, _ = spearmanr(d_E, d_Y)
rho_P, _ = spearmanr(d_P, d_Y)
print(f"Euclidean vs state: rho = {rho_E:.3f}")
print(f"Pullback  vs state: rho = {rho_P:.3f}")



###
from sklearn.metrics import pairwise_distances

def nn_state_similarity(dist_matrix, Y, W, k=5):
    n = dist_matrix.shape[0]
    errors = []
    for i in range(n):
        row = dist_matrix[i].copy()
        row[i] = np.inf
        nn_idx = np.argsort(row)[:k]
        # mean state distance from i to its k neighbours
        diffs = Y[:, nn_idx] - Y[:, i:i+1]
        errors.append(np.mean(np.sqrt(np.sum(W[:, None] * diffs**2, axis=0))))
    return np.mean(errors)

# Build full distance matrices
DE = pairwise_distances(X.T, metric='euclidean')

DP = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        dx = X[:, i] - X[:, j]
        DP[i, j] = DP[j, i] = np.sqrt(dx @ M @ dx)

print(f"Euclidean NN state similarity: {nn_state_similarity(DE, Y, W):.4f}")
print(f"Pullback  NN state similarity: {nn_state_similarity(DP, Y, W):.4f}")



###

def loo_error(dist_matrix, Y, W, k=5, sigma=None):
    n = dist_matrix.shape[0]
    if sigma is None:
        sigma = np.median(dist_matrix[dist_matrix > 0])
    total_err = 0
    for i in range(n):
        row = dist_matrix[i].copy()
        row[i] = np.inf
        nn_idx = np.argsort(row)[:k]
        weights = np.exp(-row[nn_idx]**2 / sigma**2)
        weights /= weights.sum()
        y_pred = Y[:, nn_idx] @ weights          # weighted average
        diff = y_pred - Y[:, i]
        total_err += np.sqrt(diff @ (W * diff))
    return total_err / n

print(f"Euclidean LOO error: {loo_error(DE, Y, W):.4f}")
print(f"Pullback  LOO error: {loo_error(DP, Y, W):.4f}")




# ####
# eigvals = np.linalg.eigvalsh(M)
# print("M eigenvalues:", eigvals[::-1])
# print("Condition number:", eigvals.max() / max(eigvals.min(), 1e-12))
# print("Trace of M:", np.trace(M))
# print("Euclidean scale (trace of I_r):", M.shape[0])

# Jr = dY @  np.linalg.pinv(dXr @ dXr.T)  # (n_y, r)
# Y_pred = Jr @ dXr                                 # (n_y, n_s)
# residuals = dY - Y_pred
# rel_error = np.linalg.norm(residuals) / np.linalg.norm(dY)
# print(f"Linear model relative error: {rel_error:.3f}")

# A = dXr @ dXr.T
# print("Condition number of A:", np.linalg.cond(A))



DR = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        d = X[:, i] - X[:, j]
        DR[i, j] = DR[j, i] = np.sqrt(d @ d)

rho_R, _ = spearmanr(DR[np.triu_indices(n,1)], d_Y)
print(f"Reduced Euclidean vs state: ρ = {rho_R:.3f}")
print(f"Reduced Euclidean NN similarity: {nn_state_similarity(DR, Y, W):.4f}")
print(f"Reduced Euclidean LOO error: {loo_error(DR, Y, W):.4f}")