"""
ROM study harness — config-driven, Cartesian-product sweeps over option combinations.

This is the cleaned-up successor to airfoil_rom_separate_basis.py. It:
  * loads POD bases through the unified loader (csdl_dafoam.utils.pod_basis), so the
    monolithic and separate cases share one code path (block-diagonal assembly);
  * is driven by a single STUDY dict where any option may be a list — the harness runs
    the Cartesian product of all list-valued options;
  * reports a compact per-combo summary (drag/lift rel error, converged flag, per-variable
    state L2 error) collected into one printed table + CSV.

The graph is rebuilt per combo because CSDL ties Variables to a recorder. POD bases are
cached across combos by (labels, target_variance, min_modes) so disk reads are not repeated.

NOTE: mirrors the proven graph construction of airfoil_rom_separate_basis.py; expect a light
first-run shakedown in your environment.
"""
# ===============================
# region PACKAGES
# ===============================
import itertools
from pathlib import Path

import numpy as np
from mpi4py import MPI

import csdl_alpha as csdl
import lsdo_geo
from lsdo_geo.core.parameterization.free_form_deformation_functions import (
    construct_ffd_block_around_entities,
)
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs,
)

from csdl_dafoam.core.csdl_idwarp import DAFoamMeshWarper
from csdl_dafoam.core.csdl_dafoam import (
    instantiateDAFoam, DAFoamFunctions, DAFoamSolver, compute_dafoam_input_variables,
)
import csdl_dafoam.utils.standard_atmosphere_model as sam
from csdl_dafoam.utils.runscript_helper_functions import *
from csdl_dafoam.utils.training_interface import TrainingDataInterface

from csdl_dafoam.core.rom.csdl_rom import CSDLROMWrapper
from csdl_dafoam.core.rom.rom_models import (
    DAFoamLSPGModel, DAFoamGalerkinModel, DAFoamLSPGQRModel,
)
from csdl_dafoam.core.rom.rom_solver import BroydenNewtonSolver, NewtonSolver
from csdl_dafoam.utils.pod_basis import load_pod_basis, project_onto_basis
from csdl_dafoam.utils.interpolation import (
    RBFInterpolator, CubicPolynomialInterpolator, IDWInterpolator,
)
from csdl_dafoam.utils.custom_explicit_reduced_svd import customExplicitReducedSVD


# ===============================
# region STUDY CONFIG
# ===============================
# Any value may be a list -> the harness runs the Cartesian product of all list-valued
# options. Scalars are fixed. `basis` is special: it is a list of *basis specs*, each spec
# itself a list of POD labels to combine (one label -> monolithic; many -> block-diagonal).
STUDY = {
    "basis":              "monolithic", #[["monolithic"], ["p_U_T", "nuTilda", "phi"]],  # specs to sweep
    "target_variance":    0.999,
    "min_modes":          [10, 20, 40, 80, 160],
    "rom_model_type":     "lspg",     # "lspg" | "lspg_qr" | "galerkin_analytical"
    "inner_product_type": "geometric_and_corrective",  # "corrective_only" | "geometric_and_corrective"
    "snapshot_weighting": [False, "rbf"],                   # bool | "rbf" | "cubic" | "idw"
    "distance_metric":    ["euclidean", "pullback", "pullback_reg"],  # only used when snapshot_weighting truthy
    "initial_state":      "interpolated",  # "reference" | "interpolated" | int N (try N ICs/point)
    "ic_pool_source":     ["training", "test"],  # only used when initial_state is int; "training" | "test"
    "fd_step":            1e-6,
    "T_residual_scale":   1005.0,                  # divides T residual (cp artifact)
}

NUM_TEST_SAMPLES = 10
IC_SAMPLE_SEED   = 7          # seed for drawing the multi-IC pool (int initial_state)
METRIC_RED_MODES = 50         # modes for the reduced pullback metric's Gram basis
METRIC_ALPHA_REG = 0.1        # regularization strength for the *_reg distance metrics
METRIC_FD_EPS    = 1e-6       # FD step for pullback Jacobians (cancels under trace-normalization)
RESULTS_KEYWORD  = "rom_study_results_euclidean_vs_pullback_apply_scaling_false"


# ===============================
# region FIXED INPUT (paths, DAFoam, flight conditions)
# ===============================
problem_name              = "training_data"
geometry_directory        = Path.cwd() / "airfoil_geometry"
stp_file_name             = "airfoil_transonic_unitspan_2.stp"
geometry_pickle_file_name = "airfoil_stored_refit.pickle"
dafoam_directory          = Path.cwd() / "results" / f"{problem_name}"
dataset_keyword           = "training_set_with_perturbations_300"
h5_path                   = dafoam_directory / dataset_keyword / "point_0.h5"

comm           = MPI.COMM_WORLD
rank           = comm.Get_rank()
comm_size      = comm.Get_size()
rank_str       = f"{rank:0{len(str(comm_size-1))}d}"
TIMING_ENABLED = True

U0, p0, T0, nuTilda0 = 100.0, 101325.0, 300.0, 4.5e-5

da_options = {
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleCFoam",
    "primalMinResTol": 1.0e-8,
    "primalVarBounds": {"pMin": 5000, "rhoMin": 0.05},
    "primalBC": {
        "U0":       {"variable": "U",       "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0":       {"variable": "p",       "patches": ["inout"], "value": [p0]},
        "T0":       {"variable": "T",       "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "function": {
        "drag": {"type": "force", "source": "patchToFace", "patches": ["wing"],
                 "directionMode": "parallelToFlow", "patchVelocityInputName": "patch_velocity", "scale": 1.0},
        "lift": {"type": "force", "source": "patchToFace", "patches": ["wing"],
                 "directionMode": "normalToFlow", "patchVelocityInputName": "patch_velocity", "scale": 1.0},
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm", "useNonZeroInitGuess": False},
    "transonicPCOption": 1,
    "normalizeStates": {"U": U0, "p": p0, "T": T0, "nuTilda": nuTilda0 * 10.0, "phi": 1.0},
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
        "patch_velocity": {"type": "patchVelocity", "patches": ["inout"], "flowAxis": "x",
                            "normalAxis": "z", "components": ["solver", "function"]},
        "pressure":    {"type": "patchVar", "varName": "p", "varType": "scalar", "patches": ["inout"], "components": ["solver", "function"]},
        "temperature": {"type": "patchVar", "varName": "T", "varType": "scalar", "patches": ["inout"], "components": ["solver", "function"]},
    },
}
mesh_options = {"gridFile": str(dafoam_directory), "fileType": "OpenFOAM", "symmetryPlanes": []}

# Reconstruction conventions the ROM assumes — verified against each basis's provenance.
EXPECTED_PROVENANCE = {"centering": "reference", "scaling": "reference", "inner_product": "reference"}


# ===============================
# region STUDY EXPANSION
# ===============================
def expand_study(study: dict) -> list[dict]:
    """Cartesian product over list-valued options. Scalars are wrapped to a single choice.

    Two options are irrelevant outside their enabling mode and are canonicalized to "n/a" (with
    duplicate combos dropped) so they don't multiply the sweep: `ic_pool_source` only matters
    when `initial_state` is an int (the multi-IC mode), and `distance_metric` only matters when
    `snapshot_weighting` is truthy."""
    keys    = list(study.keys())
    choices = [v if isinstance(v, list) else [v] for v in study.values()]
    combos  = [dict(zip(keys, combo)) for combo in itertools.product(*choices)]

    normalized, seen = [], set()
    for combo in combos:
        if not isinstance(combo["initial_state"], int):
            combo = {**combo, "ic_pool_source": "n/a"}
        if not combo["snapshot_weighting"]:
            combo = {**combo, "distance_metric": "n/a"}
        sig = tuple(sorted((k, str(v)) for k, v in combo.items()))
        if sig not in seen:
            seen.add(sig)
            normalized.append(combo)
    return normalized


# ===============================
# region ONE-TIME SETUP (DAFoam, geometry projection, data interface)
# ===============================
dafoam_instance     = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)
dafoam_instance_rom = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)

x_surf_dafoam_initial_mpi = dafoam_instance.getSurfaceCoordinates()
(x_surf_dafoam_initial, _, x_surf_dafoam_initial_indices) = gather_array_to_rank0(x_surf_dafoam_initial_mpi, comm)

x_surf_hash = hash_array_tol(x_surf_dafoam_initial, tol=1e-8) if rank == 0 else None
x_surf_hash = comm.bcast(x_surf_hash, root=0)

stp_file_path                     = Path(geometry_directory) / stp_file_name
surface_mesh_projection_file_path = Path(dafoam_directory) / f"projected_surface_mesh_{x_surf_hash}.pickle"

data_generator = TrainingDataInterface(
    dafoam_instance   = dafoam_instance,
    storage_location  = dafoam_directory,
    dataset_keyword   = dataset_keyword,
    h5_file_base_name = "point",
)
state_info   = data_generator.state_info
n_local_dofs = dafoam_instance.getNLocalAdjointStates()

# Shared projected surface mesh (cached pickle) — load once for all combos.
if surface_mesh_projection_file_path.is_file():
    projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)
else:
    raise FileNotFoundError(
        f"Projected surface mesh not found at {surface_mesh_projection_file_path}. "
        f"Run airfoil_rom_separate_basis.py once to generate it, or add the projection step here."
    )
comm.Barrier()

# Flight conditions (geometry DOFs are created per-combo inside the recorder).
_params = data_generator.load_h5(h5_path, group_to_read="parameters")
flight_conditions_group_template = {
    "airspeed_m_s":        _params["non_sampled_variables"]["airspeed_m_s"],
    "angle_of_attack_deg": _params["primary_variables"]["angle_of_attack_deg"],
    "altitude_m":          _params["non_sampled_variables"]["altitude (m)"],
}

# Test points — LHS over the geometric DOFs (same sampler/seed as training). Generated once
# (identical across combos). _generate_lhs_samples keys on objects with .value/.name, so a
# lightweight shim avoids needing a live CSDL recorder here.
DOF_NAMES = ["percent_change_in_thickness_dof", "normalized_percent_camber_change_dof"]

class _DummyVar:
    def __init__(self, name, shape):
        self.name  = name
        self.value = np.zeros(shape)

def _make_test_samples():
    specs = {_DummyVar(name, (3,)): {"range": [-10, 10], "ref_value": 0} for name in DOF_NAMES}
    data_generator._generate_lhs_samples(specs, NUM_TEST_SAMPLES, random_state=42)
    return {dv.name: spec["samples"] for dv, spec in specs.items()}  # {name: (n_pts, 3)} incl. ref at idx 0

TEST_SAMPLES = _make_test_samples()


# region snapshot ingredients (loaded lazily; needed for weighting / interpolated IC)
_snapshots = {"matrix": None, "configs": None}
def get_snapshot_ingredients():
    """Shared snapshot matrix (DAFoam order, ref column dropped) + parameter configs."""
    if _snapshots["matrix"] is None:
        samples = data_generator.load_h5(h5_path, group_to_read="samples")
        params  = data_generator.load_h5(h5_path, group_to_read="parameters")
        n_snap  = samples["converged"].size
        mat     = np.zeros((n_local_dofs, n_snap - 1))
        for var, info in state_info.items():
            mat[info["indices"], :] = samples["states"][var][:, 1:]
        cfg = np.concatenate([
            params["secondary_variables"]["normalized_percent_camber_change_dof"][1:, :],
            params["secondary_variables"]["percent_change_in_thickness_dof"][1:, :],
        ], axis=1)
        _snapshots["matrix"], _snapshots["configs"] = mat, cfg
    return _snapshots["matrix"], _snapshots["configs"]


# ===============================
# region BASIS CACHE
# ===============================
_basis_cache = {}
def get_basis(combo):
    key = (tuple(combo["basis"]), combo["target_variance"],
           combo["min_modes"] if not isinstance(combo["min_modes"], dict) else tuple(sorted(combo["min_modes"].items())))
    if key not in _basis_cache:
        basis = load_pod_basis(
            data_generator, h5_path,
            labels          = combo["basis"],
            target_variance = combo["target_variance"],
            min_modes       = combo["min_modes"],
        )
        basis.check_provenance(EXPECTED_PROVENANCE, print_fn=(print if rank == 0 else (lambda *a, **k: None)))
        _basis_cache[key] = basis
    return _basis_cache[key]


# region make_interpolator
def make_interpolator(method, query, sample_points):
    # apply_scaling=False is REQUIRED: apply_metric_transform already normalizes the coords per
    # axis, and a SECOND per-axis min-max here (the BaseInterpolatorClass default) would cancel the
    # metric transform -- it divides each column by its own range, discarding the pullback
    # eigenvalue magnitudes, so pullback would collapse onto pullback_reg and euclidean. See
    # apply_metric_transform.
    if method in (True, "rbf"):
        return RBFInterpolator(query, sample_points, positive_non_reproducing_weights=True, apply_scaling=False)
    if method == "cubic":
        return CubicPolynomialInterpolator(query, sample_points, 0.8, apply_scaling=False)
    if method == "idw":
        return IDWInterpolator(query, sample_points, exponent=4, apply_scaling=False)
    raise ValueError(f"Unknown snapshot_weighting method: {method}")


# region distance-metric transforms
_metric_cache = {}
def get_metric_transforms(basis):
    """Coordinate transforms L (one per distance metric) for snapshot-weighting interpolation.

    Each metric defines d_metric(xi, xj) = ||L^T (xi - xj)||, so weighting the RBF by a metric is
    just running it on the transformed coordinates z = L^T x, Z = X @ L. The pullback metrics
    measure parameter distance by state-space sensitivity: M = (1/n) sum_i J_i^T diag(w) J_i,
    with J_i the FD state Jacobian at snapshot i (from the h5 `perturbations` group). Ported from
    airfoil_rom_weighted_pod_test.py.

    Truncation-independent (the reduced metric builds its own Gram basis) and the FD step cancels
    under trace-normalization, so this is computed once and cached by basis-label set."""
    key = tuple(basis.labels)
    if key in _metric_cache:
        return _metric_cache[key]

    weights = basis.weights                                   # (n_local,) POD inner-product W

    # FD Jacobians: skip angle_of_attack_deg; DOF order (camber dof_0..2, thickness dof_0..2)
    # matches the [camber, thickness] config column order used everywhere else.
    samples = data_generator.load_h5(h5_path, group_to_read="samples")
    pert    = data_generator.load_h5(h5_path, group_to_read="perturbations")
    n_snap  = samples["converged"].size

    def _assemble(states_dset):
        out = np.zeros((n_local_dofs, n_snap))
        for var, info in state_info.items():
            out[info["indices"], :] = states_dset[var]
        return out

    base_data = _assemble(samples["states"])
    J_list    = []
    for dv_key in sorted(k for k in pert if k not in ("_attrs", "angle_of_attack_deg")):
        for dof_key in sorted(k for k in pert[dv_key] if k != "_attrs"):
            perturbed = _assemble(pert[dv_key][dof_key]["states"])
            J_list.append((perturbed - base_data) / METRIC_FD_EPS)
    n_dv = len(J_list)

    # Full pullback metric M = (1/n) sum_i J_i^T diag(w) J_i
    M       = np.zeros((n_dv, n_dv))
    J_loc_i = np.zeros((n_local_dofs, n_dv))
    for i in range(n_snap):
        for j, J_dv in enumerate(J_list):
            J_loc_i[:, j] = J_dv[:, i]
        M += J_loc_i.T @ (weights[:, None] * J_loc_i)
    M = comm.allreduce(M, op=MPI.SUM) / n_snap

    # Reduced pullback via the W-weighted Gram matrix (cheap n_snap x n_snap eig)
    sqrt_W    = np.sqrt(weights)
    Y_w       = sqrt_W[:, None] * base_data
    gram      = comm.allreduce(Y_w.T @ Y_w, op=MPI.SUM)
    eigvals_g, V_g = np.linalg.eigh(gram)
    order     = np.argsort(eigvals_g)[::-1]
    eigvals_g = np.maximum(eigvals_g[order], 0.0)
    V_g       = V_g[:, order]
    n_red     = min(METRIC_RED_MODES, n_snap)
    sigma_r   = np.sqrt(eigvals_g[:n_red]) + 1e-300
    Phi_w     = Y_w @ V_g[:, :n_red] / sigma_r[None, :]       # (n_local, n_red), W-orthonormal
    M_red     = np.zeros((n_dv, n_dv))
    for i in range(n_snap):
        for j, J_dv in enumerate(J_list):
            J_loc_i[:, j] = J_dv[:, i]
        J_red = comm.allreduce(Phi_w.T @ (sqrt_W[:, None] * J_loc_i), op=MPI.SUM)
        M_red += J_red.T @ J_red
    M_red /= n_snap

    def normalize_metric(M_in):
        return M_in * (n_dv / np.trace(M_in))

    def metric_to_L(M_mat):
        eigvals, eigvecs = np.linalg.eigh(M_mat)
        return eigvecs * np.sqrt(np.maximum(eigvals, 0.0))

    M_norm     = normalize_metric(M)
    M_red_norm = normalize_metric(M_red)

    # Anisotropy diagnostic: the pullback can only differ from euclidean to the extent M_norm is
    # anisotropic. ratio ~ 1 -> nearly isotropic, pullback == euclidean by construction (small
    # effect expected); ratio >> 1 but error still flat -> bottleneck is basis/residual-landscape,
    # not neighbor selection. See [[weighted-pod-investigation]].
    if rank == 0:
        ev = np.sort(np.linalg.eigvalsh(M_norm))[::-1]
        print(f"[pullback] M_norm eigenvalues: {np.array2string(ev, precision=3)}")
        print(f"[pullback] anisotropy (max/min): {ev[0] / (ev[-1] + 1e-300):.2f}")
    transforms = {
        "euclidean":        np.eye(n_dv),
        "pullback":         metric_to_L(M_norm),
        "pullback_red":     metric_to_L(M_red_norm),
        "pullback_reg":     metric_to_L(M_norm     + METRIC_ALPHA_REG * np.eye(n_dv)),
        "pullback_red_reg": metric_to_L(M_red_norm + METRIC_ALPHA_REG * np.eye(n_dv)),
        "n/a":              None,   # snapshot_weighting off -> euclidean IC coords
    }
    _metric_cache[key] = transforms
    return transforms


# region metric_L
def metric_L(basis, name):
    """Transform L for a distance metric; None (identity) for euclidean / n-a, which lets
    euclidean-only studies skip the pullback (perturbation-data) computation entirely."""
    if name in ("euclidean", "n/a"):
        return None
    return get_metric_transforms(basis)[name]


# region apply_metric_transform
def apply_metric_transform(L, current_cfg, snap_cfg):
    """Normalize parameter coords per axis to [0,1] (using snapshot bounds), then apply the metric
    transform z = L^T x_hat, Z = X_hat @ L. Returning PRE-scaled coords is what lets the interpolator
    run with apply_scaling=False (see make_interpolator): the metric's anisotropy then survives
    instead of being cancelled by a second per-axis min-max. Doing the normalization here (rather
    than leaning on the interpolator's internal scaling) keeps euclidean bit-identical to the old
    behavior -- for L=I the returned coords equal the interpolator's old min-max output -- while
    letting pullback / pullback_reg actually differ.

    current_cfg is CSDL (n_dv,); snap_cfg is numpy (n_snap, n_dv). L is (n_dv,n_dv) or None."""
    lo    = snap_cfg.min(axis=0)
    rng   = snap_cfg.max(axis=0) - lo
    rng   = np.where(rng == 0.0, 1.0, rng)                     # guard axes with no variation
    snap_hat = (snap_cfg - lo) / rng
    cur_hat  = (current_cfg - csdl.Variable(value=lo)) / csdl.Variable(value=rng)
    if L is None or np.allclose(L, np.eye(L.shape[0])):
        return cur_hat, snap_hat                              # euclidean: normalized, no rotation
    L_T = csdl.Variable(value=L.T)
    return csdl.einsum(L_T, cur_hat, action="ij,j->i"), snap_hat @ L


# region build_ic_pool
def build_ic_pool(combo, pod_modes, basis, fom_data):
    """Draw N candidate initial conditions for the int (multi-IC) initial_state mode and project
    them onto the basis to ROM coordinates.

    N = combo["initial_state"]; the source batch is combo["ic_pool_source"] ("training" ->
    training snapshot columns; "test" -> the FOM-solved test points, excluding the reference
    point at index 0). The same N source indices are drawn once (fixed seed) and reused for
    every test point. Projection mirrors project_onto_basis but against the live pod_modes value
    (numpy for unweighted, .value for the weighted CSDL basis) so the IC matches the basis the
    ROM actually reconstructs from.

    Returns (ic_rom_pool, source_ids) where ic_rom_pool has shape (n_modes, N) and source_ids
    are the original column/point indices (for "test": indices into fom_data)."""
    N       = combo["initial_state"]
    modes_np = pod_modes.value if isinstance(pod_modes, csdl.Variable) else pod_modes

    if combo["ic_pool_source"] == "training":
        snap_mat, _ = get_snapshot_ingredients()          # (n_local, n_snap)
        candidate_states = snap_mat
        candidate_ids    = np.arange(snap_mat.shape[1])
    elif combo["ic_pool_source"] == "test":
        candidate_ids    = np.arange(1, len(fom_data))     # drop reference test point (idx 0)
        candidate_states = np.stack([fom_data[i]["states"] for i in candidate_ids], axis=1)
    else:
        raise ValueError(f"Unknown ic_pool_source: {combo['ic_pool_source']}")

    n_avail = candidate_states.shape[1]
    if N > n_avail:
        raise ValueError(f"Requested {N} ICs but only {n_avail} '{combo['ic_pool_source']}' states available")
    rng     = np.random.default_rng(IC_SAMPLE_SEED)
    sel     = rng.choice(n_avail, size=N, replace=False)
    source_ids = candidate_ids[sel]

    S       = candidate_states[:, sel]                     # (n_local, N)
    scaled  = basis.weights[:, None] * (S - basis.reference_fom_state[:, None]) / basis.scaling[:, None]
    ic_rom_pool = comm.allreduce(modes_np.T @ scaled, op=MPI.SUM)   # (n_modes, N)
    return ic_rom_pool, source_ids


# ===============================
# region GEOMETRY (shared by FOM and ROM builds)
# ===============================
def build_geometry():
    """Build the FFD-parameterized surface + flight conditions inside the *active* recorder.

    Returns the handles both the FOM pass and the per-combo ROM build need. Must be called
    after recorder.start(); the created Variables are tied to that recorder.
    """
    geometry = lsdo_geo.import_geometry(stp_file_path, parallelize=False)
    num_ffd_chordwise, num_ffd_sections = 5, 2
    ffd_block = construct_ffd_block_around_entities(
        entities=geometry, num_coefficients=(num_ffd_chordwise, num_ffd_sections, 2), degree=(3, 1, 1),
    )
    thickness_dof = csdl.Variable(shape=(num_ffd_chordwise - 2,), value=np.zeros(num_ffd_chordwise - 2), name="percent_change_in_thickness_dof")
    camber_dof    = csdl.Variable(shape=(num_ffd_chordwise - 2,), value=np.zeros(num_ffd_chordwise - 2), name="normalized_percent_camber_change_dof")
    pct_thickness = csdl.Variable(shape=(num_ffd_chordwise, num_ffd_sections), value=0.)
    pct_camber    = csdl.Variable(shape=(num_ffd_chordwise, num_ffd_sections), value=0.)

    ffd_param = VolumeSectionalParameterization(
        name="ffd_sectional_parameterization", parameterized_points=ffd_block.coefficients,
        principal_parametric_dimension=1,
    )
    ffd_coeffs = ffd_param.evaluate(VolumeSectionalParameterizationInputs(), plot=False)

    orig_thickness  = ffd_block.coefficients.value[0, 0, 1, 2] - ffd_block.coefficients.value[0, 0, 0, 2]
    pct_thickness   = pct_thickness.set(csdl.slice[1:-1, 0], thickness_dof).set(csdl.slice[1:-1, 1], thickness_dof)
    d_thick         = (pct_thickness / 100) * orig_thickness
    ffd_coeffs      = ffd_coeffs.set(csdl.slice[:, :, 1, 2], ffd_coeffs[:, :, 1, 2] + d_thick / 2)
    ffd_coeffs      = ffd_coeffs.set(csdl.slice[:, :, 0, 2], ffd_coeffs[:, :, 0, 2] - d_thick / 2)

    block_len  = ffd_block.coefficients.value[1, 0, 0, 0] - ffd_block.coefficients.value[0, 0, 0, 0]
    pct_camber = pct_camber.set(csdl.slice[1:-1, 0], camber_dof).set(csdl.slice[1:-1, 1], camber_dof)
    d_camber   = (pct_camber / 100) * block_len
    ffd_coeffs = ffd_coeffs.set(csdl.slice[:, :, :, 2],
                                ffd_coeffs[:, :, :, 2] + csdl.expand(d_camber, (num_ffd_chordwise, num_ffd_sections, 2), "ij->ijk"))

    geometry.set_coefficients(ffd_block.evaluate_ffd(coefficients=ffd_coeffs, plot=False))
    x_surf_full = geometry.evaluate(projected_surf_mesh_dafoam, plot=False)
    i0, i1      = x_surf_dafoam_initial_indices[rank]

    fcg = csdl.VariableGroup()
    fcg.airspeed_m_s        = csdl.Variable(value=flight_conditions_group_template["airspeed_m_s"],         name="airspeed_m_s")
    fcg.angle_of_attack_deg = csdl.Variable(value=flight_conditions_group_template["angle_of_attack_deg"],  name="angle_of_attack_deg")
    fcg.altitude_m          = csdl.Variable(value=flight_conditions_group_template["altitude_m"],           name="altitude (m)")
    ambient = sam.compute_ambient_conditions_group(fcg.altitude_m)

    return dict(thickness_dof=thickness_dof, camber_dof=camber_dof,
                x_surf_full=x_surf_full, i0=i0, i1=i1, fcg=fcg, ambient=ambient)


# ===============================
# region FOM (solved once for all combos)
# ===============================
def run_fom():
    """Solve the FOM once per test point. The study only uses the FOM for comparison
    (initial ROM states are reference/interpolated, never FOM-derived), so it is independent
    of the combo and need not be re-solved for every combination.

    Returns a list (indexed by test point) of {"drag", "lift", "states"} where "states" is the
    local-rank slice of the converged full DAFoam state vector.
    """
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    geo = build_geometry()

    with csdl.experimental.mpi.enter_mpi_region(rank, comm) as mpi_region:
        x_surf = geo["x_surf_full"][geo["i0"]:geo["i1"], :].flatten()
        x_vol  = DAFoamMeshWarper(dafoam_instance).evaluate(x_surf)
        geo["fcg"].angle_of_attack_deg = mpi_region.split_custom(geo["fcg"].angle_of_attack_deg, split_func=lambda x: x)
        dafoam_inputs = compute_dafoam_input_variables(dafoam_instance, geo["ambient"], geo["fcg"], x_vol)

        dafoam_solver        = DAFoamSolver(dafoam_instance)
        dafoam_solver_states = dafoam_solver.evaluate(dafoam_inputs)
        dafoam_fn            = DAFoamFunctions(dafoam_instance, disable_jacvec_normalization=True)
        dafoam_fn_outputs    = dafoam_fn.evaluate(dafoam_solver_states, dafoam_inputs)

        for out_name in dafoam_instance.getOption("function").keys():
            mpi_region.set_as_global_output(getattr(dafoam_fn_outputs, out_name))
        mpi_region.set_as_global_output(dafoam_solver_states)

    recorder.stop()

    sim = csdl.experimental.PySimulator(recorder)
    dof_vars = {"percent_change_in_thickness_dof": geo["thickness_dof"],
                "normalized_percent_camber_change_dof": geo["camber_dof"]}
    n_pts = TEST_SAMPLES[DOF_NAMES[0]].shape[0]

    fom_data = []
    for i in range(n_pts):
        for name in DOF_NAMES:
            sim[dof_vars[name]] = TEST_SAMPLES[name][i]
        sim.run()
        fom_data.append({
            "drag":   dafoam_fn_outputs.drag.value[0],
            "lift":   dafoam_fn_outputs.lift.value[0],
            "states": dafoam_solver_states.value.copy(),
        })
        if rank == 0:
            print(f"  [FOM] test point {i+1}/{n_pts}: drag={fom_data[-1]['drag']:.6e} lift={fom_data[-1]['lift']:.6e}")
    return fom_data


# ===============================
# region PER-COMBO BUILD + RUN
# ===============================
def run_combo(combo, fom_data):
    basis = get_basis(combo)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geo           = build_geometry()
    thickness_dof = geo["thickness_dof"]
    camber_dof    = geo["camber_dof"]

    # --- residual scaling (T cp artifact) ---
    residual_scaling = np.ones_like(dafoam_instance.getStateWeights())
    residual_scaling[state_info["T"]["indices"]] *= combo["T_residual_scale"]

    # --- POD modes (optionally snapshot-weighted) ---
    # The distance metric transforms the interpolation coordinates (z = L^T x, Z = X @ L) so the
    # RBF measures parameter distance in the chosen metric's space.
    snap_w = None
    if combo["snapshot_weighting"]:
        snap_mat, snap_cfg = get_snapshot_ingredients()
        current_cfg   = csdl.concatenate((camber_dof, thickness_dof), axis=0)
        L             = metric_L(basis, combo["distance_metric"])
        z_cur, Z_snap = apply_metric_transform(L, current_cfg, snap_cfg)
        snap_w        = make_interpolator(combo["snapshot_weighting"], z_cur, Z_snap).weights()
        pod_modes     = _weighted_block_diag(basis, basis.pod_modes, snap_w, snap_mat)
    else:
        pod_modes = basis.pod_modes

    with csdl.experimental.mpi.enter_mpi_region(rank, comm) as mpi_region:
        x_surf = geo["x_surf_full"][geo["i0"]:geo["i1"], :].flatten()
        x_vol  = DAFoamMeshWarper(dafoam_instance).evaluate(x_surf)
        geo["fcg"].angle_of_attack_deg = mpi_region.split_custom(geo["fcg"].angle_of_attack_deg, split_func=lambda x: x)
        dafoam_inputs = compute_dafoam_input_variables(dafoam_instance, geo["ambient"], geo["fcg"], x_vol)

        # residual inner-product weights (state metric / residual scaling)
        if combo["inner_product_type"] == "corrective_only":
            rom_weights = 1.0 / residual_scaling
        else:
            rom_weights = basis.weights / residual_scaling
        if combo["rom_model_type"] != "galerkin_analytical":
            rom_weights = rom_weights / residual_scaling

        model_kwargs = dict(
            dafoam_input_variables_group = dafoam_inputs,
            pod_modes                    = pod_modes,
            reference_fom_state          = basis.reference_fom_state,
            scaling                      = basis.scaling,
            weights                      = rom_weights,
            dafoam_instance              = dafoam_instance_rom,
            fd_step                      = combo["fd_step"],
            disable_presolve_diagnostics = True,
        )
        # Match the proven predecessor's tolerances: a loose tol_rel stops the (worse-conditioned)
        # weighted-POD solve short of its best-in-basis state, which exaggerates "weighted worse".
        solver_options = {"tol_rel": 1e-12, "tol_step_abs": 1e-13}
        if combo["rom_model_type"] == "lspg_qr":
            rom_model  = DAFoamLSPGQRModel(**model_kwargs)
            rom_solver = NewtonSolver(options=solver_options)
        elif combo["rom_model_type"] == "galerkin_analytical":
            rom_model  = DAFoamGalerkinModel(jac_mode="analytical", **model_kwargs)
            rom_solver = NewtonSolver(options=solver_options)
        else:
            rom_model  = DAFoamLSPGModel(**model_kwargs)
            rom_solver = BroydenNewtonSolver(options=solver_options)

        # --- initial ROM state ---
        # ic_rom_pool / source_ids are only populated for the int (multi-IC) mode; they are
        # threaded to _run_test_sweep so it can set a fresh IC per (point, IC) run.
        ic_rom_pool = source_ids = None

        # Interpolated IC: project Sum_i w_i * snapshot_i onto the basis (no FOM needed).
        # Lands in the good basin of the nonlinear LSPG residual (validated this session).
        # Use the same interpolation method AND distance metric as the basis weighting (rbf /
        # euclidean when unweighted), so IC and weighting stay consistent within a combo.
        if combo["initial_state"] == "interpolated":
            snap_mat, snap_cfg = get_snapshot_ingredients()
            current_cfg   = csdl.concatenate((camber_dof, thickness_dof), axis=0)
            ic_method     = combo["snapshot_weighting"] if combo["snapshot_weighting"] else "rbf"
            L             = metric_L(basis, combo["distance_metric"])
            z_cur, Z_snap = apply_metric_transform(L, current_cfg, snap_cfg)
            w_interp      = basis.reference_fom_state + (snap_mat - basis.reference_fom_state[:, None]) @ \
                            make_interpolator(ic_method, z_cur, Z_snap).weights()
            Phi_csdl    = pod_modes if isinstance(pod_modes, csdl.Variable) else csdl.Variable(value=pod_modes)
            init_state  = csdl.experimental.mpi.mpi_allreduce(
                Phi_csdl.T() @ (basis.weights * (w_interp - basis.reference_fom_state) / basis.scaling), comm=comm)
        elif combo["initial_state"] == "reference":
            init_state = np.zeros((pod_modes.shape[1], ))#None
        elif isinstance(combo["initial_state"], int):
            # Multi-IC: the sweep mutates rom_wrapper.constant_intitial_rom_state per candidate IC
            # (re-read on every solve; execute_inline re-runs the op unconditionally). The IC is a
            # numpy array, NOT a Variable: an initial guess has no effect on the converged output,
            # so a Variable IC gets pruned out of the root graph and can't be set via the simulator.
            ic_rom_pool, source_ids = build_ic_pool(combo, pod_modes, basis, fom_data)
            init_state = np.zeros((pod_modes.shape[1],))   # placeholder; overwritten per-IC in sweep
        else:
            raise TypeError("Unknown IC option: please use 'interpolated', 'reference', or an int")

        rom_wrapper = CSDLROMWrapper(model=rom_model, solver=rom_solver, constant_initial_rom_state=init_state)
        rom_states  = rom_wrapper.evaluate()

        # Reconstruct from the LIVE pod_modes (a Variable for the weighted basis, which depends on
        # the geometry DOFs). rom_model.pod_modes is a numpy copy frozen at build time (DOFs=0), so
        # using it would reconstruct off-reference test points with the wrong weighted basis.
        state_est      = basis.reference_fom_state + basis.scaling * (pod_modes @ rom_states)
        rom_fn         = DAFoamFunctions(dafoam_instance_rom, disable_jacvec_normalization=True)
        rom_fn_outputs = rom_fn.evaluate(state_est, dafoam_inputs)

        for out_name in dafoam_instance.getOption("function").keys():
            mpi_region.set_as_global_output(getattr(rom_fn_outputs, out_name))

    recorder.stop()

    # --- run the test sweep ---
    sim = csdl.experimental.PySimulator(recorder)
    dof_vars = {"percent_change_in_thickness_dof": thickness_dof,
                "normalized_percent_camber_change_dof": camber_dof}
    rows = _run_test_sweep(sim, dof_vars, rom_fn_outputs, rom_model, rom_wrapper, basis, pod_modes,
                           snap_w, fom_data, combo, ic_rom_pool, source_ids)
    return rows


# region _weighted_block_diag
def _weighted_block_diag(basis, pod_modes_np, snapshot_weights, snapshot_matrix):
    """Snapshot-weight each block on its FULL basis, then truncate to n_b columns as the last
    step, placing the result into a full (n_local, n_modes_total) CSDL block-diagonal matrix.

    Weighting re-SVDs the snapshot ensemble (D = Sigma VT diag(omega)) and re-orthonormalizes,
    which re-orders mode importance. Truncating *after* this rotation (not before) ensures the
    weighting sees the full subspace; see _select_n_modes / load_pod_basis for where n_b comes from.
    """
    # |omega| (matches the proven predecessor); guards against negative cubic/idw weights.
    abs_weights = csdl.sqrt(snapshot_weights ** 2)
    pod_modes = csdl.Variable(value=np.zeros_like(pod_modes_np))
    col0 = 0
    for label in basis.labels:
        n_b      = basis.n_modes[label]                          # truncated count (output width)
        Phi_full = basis.full_modes[label]                       # (n_local, n_full)
        sv_full  = basis.singular_values[label][:Phi_full.shape[1]]
        scaled   = (basis.weights / basis.scaling)[:, None] * (snapshot_matrix - basis.reference_fom_state[:, None])
        VT_full  = (1.0 / sv_full[:, None]) * comm.allreduce(Phi_full.T @ scaled, op=MPI.SUM)
        D_full   = csdl.einsum(sv_full, csdl.einsum(VT_full, abs_weights, action="ij,j->ij"), action="i,ij->ij")
        UD_full, _, _ = customExplicitReducedSVD().evaluate(A=D_full)   # (n_full, min(n_full, n_snap))
        rotated  = Phi_full @ UD_full                            # (n_local, min(n_full, n_snap))
        pod_modes = pod_modes.set(csdl.slice[:, col0:col0 + n_b], rotated[:, :n_b])  # TRUNCATE LAST
        col0 += n_b
    return pod_modes


# region _projection_error
def _projection_error(modes, basis, w_fom):
    """Relative W-projection error of a FOM state onto `modes` (truncated), in SCALED space —
    matches the predecessor's `_w_proj_rel_err` so values are directly comparable:
        s_tilde = (w - ref)/s,  c = Phi^T (W s_tilde),  rec = Phi c
        err     = ||s_tilde - rec||_W / ||s_tilde||_W
    Isolates basis quality from the ROM solve. Mirrors project_onto_basis (pod_basis.py)."""
    modes   = np.asarray(modes)
    s_tilde = (w_fom - basis.reference_fom_state) / basis.scaling
    c       = comm.allreduce(modes.T @ (basis.weights * s_tilde), op=MPI.SUM)
    rec     = modes @ c
    num     = comm.allreduce(np.sum(basis.weights * (s_tilde - rec) ** 2), op=MPI.SUM)
    den     = comm.allreduce(np.sum(basis.weights * s_tilde ** 2),         op=MPI.SUM)
    return np.sqrt(num / (den + 1e-300))


# region _residual_landscape
def _residual_landscape(rom_model, basis, modes_w, w_rom, w_fom):
    """The decisive weighted-POD diagnostic (mirrors the predecessor): compare the weighted
    residual norm ||W^0.5 r|| the solver minimizes at three states —
        proj : best-in-basis reconstruction of the TRUE state (lower bound the ROM can reach),
        rom  : the ROM's converged solution,
        fom  : the TRUE FOM state (should be ~0 on this geometry).
    rn_proj >> rn_rom  -> residual-landscape problem (the weighted residual's subspace minimum is
                          far from the accurate state); rn_proj ~ rn_rom but states differ ->
                          insensitive directions; basis is fine. Restores OF state on exit."""
    m_res = rom_model.weights
    def _rn(w):
        r = rom_model._eval_fom_residual(fom_state=w)
        return np.sqrt(comm.allreduce(np.sum(m_res * r * r), op=MPI.SUM))
    s_tilde = (w_fom - basis.reference_fom_state) / basis.scaling
    q_proj  = comm.allreduce(modes_w.T @ (basis.weights * s_tilde), op=MPI.SUM)
    w_proj  = basis.reference_fom_state + basis.scaling * (modes_w @ q_proj)
    rn_proj = _rn(w_proj)
    rn_rom  = _rn(w_rom)
    rn_fom  = _rn(w_fom)
    rom_model._set_fom_states(w_rom)   # restore (the evals above perturbed the OF state)
    return rn_proj, rn_rom, rn_fom


# region _ic_list_for_point
def _ic_list_for_point(i, combo, ic_rom_pool, source_ids):
    """Per-test-point list of (ic_index, ic_source_point, ic_state) to run.

    Non-int IC modes -> a single entry (ic_index=-1, no explicit state, uses the graph's baked
    IC). Int mode -> one entry per pooled IC; for ic_pool_source=="test" the IC whose source
    point is the current test point i is skipped (don't start a point from its own FOM state)."""
    if not isinstance(combo["initial_state"], int):
        return [(-1, -1, None)]
    entries = []
    for j, sid in enumerate(source_ids):
        if combo["ic_pool_source"] == "test" and int(sid) == i:
            continue
        entries.append((j, int(sid), ic_rom_pool[:, j]))
    return entries


# region _run_test_sweep
def _run_test_sweep(sim, dof_vars, rom_fn_outputs, rom_model, rom_wrapper, basis, pod_modes,
                    snap_w, fom_data, combo, ic_rom_pool, source_ids):
    n_pts = TEST_SAMPLES[DOF_NAMES[0]].shape[0]   # NUM_TEST_SAMPLES + 1 (ref at index 0)
    n_snap = get_snapshot_ingredients()[0].shape[1]

    rows = []
    for i in range(n_pts):
        for name in DOF_NAMES:
            sim[dof_vars[name]] = TEST_SAMPLES[name][i]

        for ic_index, ic_source_point, ic_state in _ic_list_for_point(i, combo, ic_rom_pool, source_ids):
            if ic_state is not None:
                # Re-read on every solve; not a graph input, so mutate the wrapper attribute directly.
                rom_wrapper.constant_intitial_rom_state = np.asarray(ic_state)
            sim.run()

            result   = rom_wrapper._cached_result   # refreshed on every sim.run()
            conv     = int(bool(result.converged)) if result is not None else 0
            n_iter   = result.iterations if result is not None else np.nan

            rom_drag = rom_fn_outputs.drag.value[0]; fom_drag = fom_data[i]["drag"]
            rom_lift = rom_fn_outputs.lift.value[0]; fom_lift = fom_data[i]["lift"]
            drag_rel = abs(rom_drag - fom_drag) / (abs(fom_drag) + 1e-300)
            lift_rel = abs(rom_lift - fom_lift) / (abs(fom_lift) + 1e-300)

            w_rom = dafoam_instance_rom.getStates()
            w_fom = fom_data[i]["states"]
            per_var = {}
            for var, vinfo in state_info.items():
                idx  = vinfo["indices"]
                err  = np.sqrt(comm.allreduce(np.sum((w_rom[idx] - w_fom[idx]) ** 2), op=MPI.SUM))
                nrm  = np.sqrt(comm.allreduce(np.sum(w_fom[idx] ** 2), op=MPI.SUM))
                per_var[var] = err / (nrm + 1e-300)

            # Projection diagnostics: unweighted basis vs the (possibly weighted) basis sent to the ROM.
            proj_err_u   = _projection_error(basis.pod_modes, basis, w_fom)
            modes_w      = pod_modes.value if isinstance(pod_modes, csdl.Variable) else pod_modes
            proj_err_w   = _projection_error(modes_w, basis, w_fom)

            # Residual-landscape: is "weighted worse" a basis problem or a ROM-solve problem?
            rn_proj, rn_rom, rn_fom = _residual_landscape(rom_model, basis, modes_w, w_rom, w_fom)

            snap_weights = snap_w.value if isinstance(snap_w, csdl.Variable) else np.full(n_snap, 1.0 / n_snap)
            w_arr        = np.asarray(snap_weights).ravel()
            n_eff        = (w_arr.sum() ** 2) / (np.sum(w_arr ** 2) + 1e-300)  # weight participation ratio

            if rank == 0:
                ic_tag = "" if ic_index < 0 else f" ic={ic_index}(src {ic_source_point})"
                print(f"  [pt {i}{ic_tag}] conv={conv} iters={n_iter} "
                      f"proj_err u={proj_err_u:.3e} w={proj_err_w:.3e} | "
                      f"||W^0.5 r|| proj={rn_proj:.3e} rom={rn_rom:.3e} fom={rn_fom:.3e} | "
                      f"n_eff={n_eff:.1f}/{w_arr.size}")

            rows.append({"point": i, "ic_index": ic_index, "ic_source_point": ic_source_point,
                         "converged": conv, "iterations": n_iter,
                         "rom_drag": rom_drag, "fom_drag": fom_drag, "drag_rel": drag_rel,
                         "rom_lift": rom_lift, "fom_lift": fom_lift, "lift_rel": lift_rel,
                         "state_err_max": max(per_var.values()),
                         "proj_err_w": proj_err_w, "proj_err_u": proj_err_u,
                         "resnorm_proj": rn_proj, "resnorm_rom": rn_rom, "resnorm_fom": rn_fom,
                         "weight_n_eff": n_eff,
                         "snap_weights": w_arr,
                         **{f"err_{k}": v for k, v in per_var.items()}})
    return rows


# region helpers (summary / formatting)
def _basis_label(basis_spec):
    """Join a basis spec (list of labels) for display; tolerate a bare string."""
    if isinstance(basis_spec, (list, tuple)):
        return "+".join(basis_spec)
    return str(basis_spec)


def summarize(combo_id, combo, rows):
    """One compact summary row per combo: combo_id + every STUDY variable + aggregated metrics
    (means/maxes over non-reference test points)."""
    pts = [r for r in rows if r["point"] != 0]  # drop reference point
    def agg(key, fn): return fn([r[key] for r in pts]) if pts else float("nan")
    out = {"combo_id": combo_id}
    for k, v in combo.items():
        out[k] = _basis_label(v) if k == "basis" else str(v)
    out.update({
        "drag_rel_mean":  agg("drag_rel", np.mean),
        "drag_rel_max":   agg("drag_rel", np.max),
        "lift_rel_mean":  agg("lift_rel", np.mean),
        "lift_rel_max":   agg("lift_rel", np.max),
        "state_err_max":  agg("state_err_max", np.max),
        "proj_err_w_max": agg("proj_err_w", np.max),
        "proj_err_u_max": agg("proj_err_u", np.max),
    })
    return out


# ===============================
# region MAIN
# ===============================
if __name__ == "__main__":
    import pandas as pd
    RESULTS_CSV      = f"{RESULTS_KEYWORD}.csvs"
    SAMPLES_CSV      = f"{RESULTS_KEYWORD}_samples.csv"
    SNAP_WEIGHTS_CSV = f"{RESULTS_KEYWORD}_weights.csv"

    combos = expand_study(STUDY)
    if rank == 0:
        print(f"\n=== ROM study: {len(combos)} combination(s) ===")
        print("\n--- FOM (solved once for all combos) ---")
    fom_data = run_fom()

    summary_rows = []
    sample_rows  = []
    snap_rows    = []
    for c_idx, combo in enumerate(combos):
        if rank == 0:
            print(f"\n--- combo {c_idx+1}/{len(combos)}: "
                  f"basis={combo['basis']} model={combo['rom_model_type']} "
                  f"init={combo['initial_state']} weighting={combo['snapshot_weighting']} ---")
        rows = run_combo(combo, fom_data)
        summary_rows.append(summarize(c_idx, combo, rows))

        # Per-sample long-format records (one row per test point) + snapshot-weight records.
        combo_cols = {k: (_basis_label(v) if k == "basis" else str(v)) for k, v in combo.items()}
        for r in rows:
            sample_rows.append({
                "combo_id": c_idx, **combo_cols, "point": r["point"],
                "ic_index": r["ic_index"], "ic_source_point": r["ic_source_point"],
                "converged": r["converged"], "iterations": r["iterations"],
                "rom_drag": r["rom_drag"], "fom_drag": r["fom_drag"], "drag_rel": r["drag_rel"],
                "rom_lift": r["rom_lift"], "fom_lift": r["fom_lift"], "lift_rel": r["lift_rel"],
                "proj_err_w": r["proj_err_w"], "proj_err_u": r["proj_err_u"],
                "resnorm_proj": r["resnorm_proj"], "resnorm_rom": r["resnorm_rom"],
                "resnorm_fom": r["resnorm_fom"], "weight_n_eff": r["weight_n_eff"],
                **{k: v for k, v in r.items() if k.startswith("err_")},
            })
            for j, w in enumerate(r["snap_weights"]):
                snap_rows.append({"combo_id": c_idx, "point": r["point"], "ic_index": r["ic_index"],
                                  "snapshot_index": j, "weight": float(w)})

    if rank == 0:
        summary_df = pd.DataFrame(summary_rows)
        with pd.option_context("display.max_columns", None, "display.width", 200,
                               "display.float_format", lambda x: f"{x:.4e}"):
            print("\n=== SUMMARY ===")
            print(summary_df.to_string(index=False))

        summary_df.to_csv(RESULTS_CSV, index=False)
        pd.DataFrame(sample_rows).to_csv(SAMPLES_CSV, index=False)
        pd.DataFrame(snap_rows).to_csv(SNAP_WEIGHTS_CSV, index=False)
        print(f"\nWrote {RESULTS_CSV}, {SAMPLES_CSV}, {SNAP_WEIGHTS_CSV}")
