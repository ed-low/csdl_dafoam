# ===============================
# region PACKAGES
# ===============================
import numpy as np
import os
from pathlib import Path

# MPI
from mpi4py import MPI

# CSDL packages
import csdl_alpha as csdl
import lsdo_geo

# LSDO_geo specific
from lsdo_geo.core.parameterization.free_form_deformation_functions import (
    construct_ffd_block_around_entities
)
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)

# IDWarp and DAFoam
from csdl_dafoam.core.csdl_idwarp import DAFoamMeshWarper
from csdl_dafoam.core.csdl_dafoam import instantiateDAFoam, DAFoamFunctions, DAFoamSolver, compute_dafoam_input_variables
import csdl_dafoam.utils.standard_atmosphere_model as sam
from csdl_dafoam.utils.runscript_helper_functions import *
from csdl_dafoam.utils.training_interface import TrainingDataInterface

# ROM
from csdl_dafoam.core.rom.csdl_rom import CSDLROMWrapper
from csdl_dafoam.core.rom.rom_models import DAFoamLSPGModel, DAFoamPhiComputingLSPGModel, DAFoamGalerkinModel, DAFoamLSPGQRModel
from csdl_dafoam.core.rom.rom_solver import BroydenNewtonSolver, NewtonSolver

#---- DEBUGGING TOOLS ----
import faulthandler
faulthandler.enable()
os.environ["PETSC_OPTIONS"] = "-malloc_debug"
#-------------------------


# ===============================
# region USER INPUT
# ===============================
problem_name              = 'training_data'

# Geometry
geometry_directory        = Path.cwd() / 'airfoil_geometry'
stp_file_name             = 'airfoil_transonic_unitspan_2.stp'
geometry_pickle_file_name = 'airfoil_stored_refit.pickle'

# MPI and timing
comm           = MPI.COMM_WORLD
TIMING_ENABLED = True

# DAFoam
dafoam_directory = Path.cwd() / 'results' / f'{problem_name}'

# Initial/reference values for DAFoam
U0       = 100.0
p0       = 101325.0
T0       = 300.0
nuTilda0 = 4.5e-5
A0       = 0.1
rho0     = p0 / T0 / 287

# Input parameters for DAFoam
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
    }
}

mesh_options = {
    "gridFile": str(dafoam_directory),
    "fileType": "OpenFOAM",
    "symmetryPlanes": [],
}

# ===============================
# region Training / ROM options
# ===============================
dataset_keyword  = 'training_set_with_perturbations_300'
storage_location = dafoam_directory
h5_path          = storage_location / dataset_keyword / "point_0.h5"

# ROM mode count per variable.
# int  → same cap for every variable
# dict → per-variable caps, e.g. {"U": 20, "p": 10, "T": 5, "nuTilda": 5, "phi": 5}
# None → use all available modes
N_MODES_PER_VAR = 10

# Test sweep — LHS over geometric DOFs (thickness + camber)
num_test_samples = 10   # LHS test points; reference point (all zeros) is always prepended


# ===============================
# region SETUP
# ===============================
rank      = comm.Get_rank()
comm_size = comm.Get_size()
rank_str  = f"{rank:0{len(str(comm_size-1))}d}"

dafoam_instance     = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)
dafoam_instance_rom = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)

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


# ===============================
# region Training interface and split POD load
# ===============================
data_generator = TrainingDataInterface(
    dafoam_instance  = dafoam_instance,
    storage_location = storage_location,
    dataset_keyword  = dataset_keyword,
    h5_file_base_name= "point",
)


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

# FFD parameterization
num_ffd_coefficients_chordwise = 5
num_ffd_sections               = 2
ffd_block = construct_ffd_block_around_entities(
    entities=geometry,
    num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2),
    degree=(3, 1, 1),
)

percent_change_in_thickness_dof      = csdl.Variable(shape=(num_ffd_coefficients_chordwise-2,), value=np.array([0, 0, 0]), name="percent_change_in_thickness_dof")
normalized_percent_camber_change_dof = csdl.Variable(shape=(num_ffd_coefficients_chordwise-2,), value=np.array([0, 0, 0]), name="normalized_percent_camber_change_dof")

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

i0, i1 = x_surf_dafoam_initial_indices[rank]


data = data_generator.load_h5(h5file_path=h5_path)
state_info = data_generator.state_info


# Flight conditions
flight_conditions_group                     = csdl.VariableGroup()
flight_conditions_group.airspeed_m_s        = csdl.Variable(value=data["parameters"]["non_sampled_variables"]["airspeed_m_s"],    name="airspeed_m_s")
flight_conditions_group.angle_of_attack_deg = csdl.Variable(value=data["parameters"]["primary_variables"]["angle_of_attack_deg"], name="angle_of_attack_deg")
flight_conditions_group.altitude_m          = csdl.Variable(value=data["parameters"]["non_sampled_variables"]["altitude (m)"],      name="altitude (m)")

ambient_conditions_group = sam.compute_ambient_conditions_group(flight_conditions_group.altitude_m)

# Options
pod_basis_mode       = "all"             # "all" | "separate"
phi_mode             = "in_basis"        # "in_basis" | "computed"
inner_product_type   = "geometric_and_corrective" # "corrective_only" | "geometric_and_corrective"
snapshot_weight_type = None              # None | "euclidean"
target_variance      = 0.999
min_modes            = 30
weight_method        = "rbf"
rom_model_type       = "lspg"         # "lspg" | "lspg_qr" | "galerkin_analytical" [lspg_qr/galerkin_analytical require NewtonSolver (set below)]
fd_step              = 1e-6

# Separate mode key list
key_list        = ["p_U_T", "nuTilda", "phi"]

key_colors      = {"p_U_T"  :"tab:blue", 
                   "nuTilda":"tab:orange", 
                   "phi"    :"tab:green"}

if pod_basis_mode == "separate":
    # ===============================
    # region POD load (per-variable bases)
    # ===============================

    def load_modes(mode_key):
        key = f"pod_{mode_key}"
        return np.concatenate([data[key]["modes"][var] for var in state_info.keys() if var in data[key]["_attrs"]["var_group"]])

    def load_ref(mode_key):
        key = f"pod_{mode_key}"
        return {var : data[key]["reference_state"][var] for var in state_info.keys() if var in data[key]["_attrs"]["var_group"]}

    def load_scaling(mode_key):
        key = f"pod_{mode_key}"
        return {var : data[key]["scaling"][var] for var in state_info.keys() if var in data[key]["_attrs"]["var_group"]}

    def load_weights(mode_key):
        key = f"pod_{mode_key}"
        return {var: data[key]["weights"][var] for var in state_info.keys() if var in data[key]["_attrs"]["var_group"]}

    pod_mode_set    = {key: load_modes(key)   for key in key_list}
    reference_set   = {key: load_ref(key)     for key in key_list}
    scaling_set     = {key: load_scaling(key) for key in key_list}
    weights_set     = {key: load_weights(key) for key in key_list}

    # # U_p_T_nuTilda_modes = load_modes("p_U_T_nuTilda")
    # U_p_T_modes     = load_modes("p_U_T")
    # nuTilda_modes   = load_modes("nuTilda")
    # phi_modes       = load_modes("phi")

    # # U_p_T_nuTilda_ref = load_ref("p_U_T_nuTilda")
    # U_p_T_ref       = load_ref("p_U_T")
    # nuTilda_ref     = load_ref("nuTilda")
    # phi_ref         = load_ref("phi")
    # ref_dict        = U_p_T_ref | nuTilda_ref | phi_ref

    # # U_p_T_nuTilda_scaling   = load_scaling("p_U_T_nuTilda")
    # U_p_T_scaling   = load_scaling("p_U_T")
    # nuTilda_scaling = load_scaling("nuTilda")
    # phi_scaling     = load_scaling("phi")
    # scaling_dict    = U_p_T_scaling | nuTilda_scaling | phi_scaling

    # # U_p_T_nuTilda_weights   = load_weights("p_U_T_nuTilda")
    # U_p_T_weights   = load_weights("p_U_T")
    # nuTilda_weights = load_weights("nuTilda")
    # phi_weights     = load_weights("phi")
    # weights_dict    = U_p_T_weights | nuTilda_weights | phi_weights


    import matplotlib.pyplot as plt

    singular_val_set        = {key: data[f"pod_{key}"]["singular_values"] for key in key_list}
    cumulative_energy_set   = {key: np.cumsum(singular_val_set[key] ** 2)   / np.sum(singular_val_set[key] ** 2) for key in key_list}
    n_mode_set              = {key: np.argmax(cumulative_energy_set[key] >= target_variance) for key in key_list}

    if rank == 0:
        for key in key_list:
            plt.plot(cumulative_energy_set[key],    label=key,      c=key_colors[key])
            plt.axvline(n_mode_set[key],            linestyle="--", c=key_colors[key])
        plt.legend()
        plt.show()

    quiet_barrier(comm)

    n_local_dofs = dafoam_instance.getNLocalAdjointStates()
    n_modes      = sum(n_mode_set.values())
    n_dofs_set   = {pod_mode_set[key].shape[0] for key in key_list}

    if rank == 0:
        if phi_mode == "in_basis":
            print(f"n_modes_total   : {n_modes}")
            for key in key_list:
                print(f"n_modes_{key}  : {n_mode_set[key]}")

    weights             = np.zeros(n_local_dofs)
    scaling             = np.zeros(n_local_dofs)
    reference_fom_state = np.zeros(n_local_dofs)

    for key in key_list:
        for state_var, info in state_info.items():
            if state_var in data[f"pod_{key}"]["_attrs"]["var_group"]:
                idx = info["indices"]
                scaling[idx]             = scaling_set[key][state_var]
                weights[idx]             = weights_set[key][state_var]
                reference_fom_state[idx] = reference_set[key][state_var]

    # Temperature residual rescaling (improves LSPG conditioning)
    residual_scaling                              = np.ones_like(dafoam_instance.getStateWeights())
    residual_scaling[state_info["T"]["indices"]] *= 1005

elif pod_basis_mode == "all":
    # POD data assembled into DAFoam's cell-interleaved state-vector ordering
    singular_val_set = {"monolithic" : data["pod"]["monolithic"]["singular_values"]}
    n_local_dofs     = dafoam_instance.getNLocalAdjointStates()
    n_snapshots      = singular_val_set["monolithic"].shape[0]
    pod_mode_set     = {"monolithic" : np.zeros((n_local_dofs, n_snapshots))}
    scaling          = np.zeros(n_local_dofs)
    reference_fom_state = np.zeros(n_local_dofs)
    weights         = np.zeros(n_local_dofs)

    for state_var, info in state_info.items():
        idx = info["indices"]
        pod_mode_set["monolithic"][idx, :] = data["pod"]["monolithic"]["modes"][state_var]
        scaling[idx]                       = data["pod"]["monolithic"]["scaling"][state_var]
        weights[idx]                       = data["pod"]["monolithic"]["weights"][state_var]
        reference_fom_state[idx]           = data["pod"]["monolithic"]["reference_state"][state_var]

    # Temperature residual rescaling (improves LSPG conditioning)
    residual_scaling                              = np.ones_like(dafoam_instance.getStateWeights())
    residual_scaling[state_info["T"]["indices"]] *= 1005

    cumulative_energy_set = {"monolithic" : np.cumsum(singular_val_set["monolithic"] ** 2) / np.sum(singular_val_set["monolithic"] ** 2)}
    n_mode_set            = {"monolithic": np.argmax(cumulative_energy_set["monolithic"] >= target_variance)}
    n_modes               = np.max([n_mode_set["monolithic"], min_modes])


# Snapshot weighting
n_snapshots = data["samples"]["converged"].size

def load_snapshots(mode_key):
    key = f"pod_{mode_key}"
    return np.concatenate([data["samples"]["states"][var] for var in state_info.keys() if var in data[key]["_attrs"]["var_group"]])

if pod_basis_mode == "separate":
    snapshots_set = {key : load_snapshots(key) for key in key_list}
else:
    snapshot_matrix = np.zeros((n_local_dofs, n_snapshots - 1))
    for state_var, info in state_info.items():
        snapshot_matrix[info["indices"], :] = data["samples"]["states"][state_var][:, 1:]
    snapshots_set = {"monolithic": snapshot_matrix}

if snapshot_weight_type is not None:
    from csdl_dafoam.utils.interpolation import CubicPolynomialInterpolator, RBFInterpolator, IDWInterpolator
    from csdl_dafoam.utils.custom_explicit_reduced_svd import customExplicitReducedSVD

    # Parameter matrix: shape (n_snapshots, n_dv)
    snapshot_configs_full = np.concatenate([
        data["parameters"]["secondary_variables"]["normalized_percent_camber_change_dof"][1:, :],
        data["parameters"]["secondary_variables"]["percent_change_in_thickness_dof"][1:, :]
    ], axis=1)   # (n_snapshots, n_dv)

    current_config = csdl.concatenate(
        (normalized_percent_camber_change_dof, percent_change_in_thickness_dof), axis=0
    )

    if   weight_method == "rbf":
        snapshot_weights = RBFInterpolator(current_config, snapshot_configs_full, positive_non_reproducing_weights=True).weights()
    elif weight_method == "cubic":
        snapshot_weights = CubicPolynomialInterpolator(current_config, snapshot_configs_full, 0.8).weights()
    elif weight_method == "idw":
        snapshot_weights = IDWInterpolator(current_config, snapshot_configs_full, exponent=4).weights()

    VT_set = {}
    if pod_basis_mode == "separate":
        for key, pod_modes_key in pod_mode_set.items():
            weights_key   = np.concatenate([weights_set[key][var]   for var in state_info.keys() if var in data[f"pod_{key}"]["_attrs"]["var_group"]])
            scaling_key   = np.concatenate([scaling_set[key][var] * np.ones_like(reference_set[key][var])   for var in state_info.keys() if var in data[f"pod_{key}"]["_attrs"]["var_group"]])
            reference_key = np.concatenate([reference_set[key][var] for var in state_info.keys() if var in data[f"pod_{key}"]["_attrs"]["var_group"]])
            VT_set[key]   = (1.0 / singular_val_set[key][:, None]) * comm.allreduce(
                pod_modes_key.T @ ((weights_key / scaling_key)[:, None] * (snapshots_set[key] - reference_key[:, None])),
                op=MPI.SUM
            )
    
    else:
        VT_set["monolithic"] = (1.0 / singular_val_set["monolithic"][:, None]) * comm.allreduce(
            pod_mode_set["monolithic"].T @ ((weights / scaling)[:, None] * (snapshots_set["monolithic"] - reference_fom_state[:, None])),
            op=MPI.SUM
        )

    # Stash the UNWEIGHTED (pre-rotation) bases so we can compare projection
    # quality against the rotated bases later (per test config, in the sweep).
    pod_mode_set_unweighted = {k: v.copy() for k, v in pod_mode_set.items()}

    # ---- Setup-time invariant checks (query-independent) ----
    # These validate that the VT reconstruction is consistent with the stored
    # POD (no snapshot ordering / scaling / reference mismatch). If either of
    # these fails, the weighted-POD rotation is operating on a corrupted VT and
    # WILL degrade the basis no matter how good the weights are.
    run_weighting_diagnostics = True
    if run_weighting_diagnostics and rank == 0 and "monolithic" in VT_set:
        VT_np    = np.asarray(VT_set["monolithic"])          # (n_modes_total, n_snap)
        sigma_np = np.asarray(singular_val_set["monolithic"])# (n_modes_total,)

        # (1) Right singular vectors should have orthonormal rows: VT @ VT.T = I.
        #     Large deviation => reconstruction VT = S^-1 Phi^T W (S-ref)/s is wrong.
        gram = VT_np @ VT_np.T
        off  = gram - np.eye(gram.shape[0])
        print("\n[weighted-POD diag] VT orthonormality:")
        print(f"    ||VT VT^T - I||_F        = {np.linalg.norm(off):.3e}  (≈0 if reconstruction exact)")
        print(f"    max |off-diagonal|        = {np.max(np.abs(off - np.diag(np.diag(off)))):.3e}")

        # (2) Uniform weights (omega=1) MUST recover the original POD: U_D = ±I,
        #     S_D = sigma. Any deviation here is a pure pipeline bug.
        D1            = sigma_np[:, None] * VT_np            # = Sigma VT diag(1)
        U1, S1, _     = np.linalg.svd(D1, full_matrices=False)
        U1_abs_off    = np.abs(U1) - np.eye(*U1.shape) if U1.shape[0] == U1.shape[1] else U1
        print("[weighted-POD diag] omega=1 recovery (should reproduce standard POD):")
        print(f"    ||S_D(omega=1) - sigma||  = {np.linalg.norm(S1 - sigma_np):.3e}")
        print(f"    max ||U_D| - I|           = {np.max(np.abs(U1_abs_off)):.3e}  (≈0 => rotation is identity for uniform weights)\n")

    D_svd_set = {}
    UD_set    = {}
    weighted_pod_mode_set = {}

    for key, VT in VT_set.items():
        D_svd_set[key]= csdl.einsum(
            singular_val_set[key],
            csdl.einsum(VT, csdl.sqrt(snapshot_weights ** 2), action='ij,j->ij'),
            action='i,ij->ij'
        )
        UD_set[key], _, _ = customExplicitReducedSVD().evaluate(A=D_svd_set[key])
        pod_mode_set[key] = pod_mode_set[key] @ UD_set[key]#[:, :n_mode_set[key]]   # (n_local, n_retained_modes) CSDL
else:
    snapshot_weights = csdl.Variable(value=np.ones(n_snapshots - 1, )) / (n_snapshots - 1)

# Construct basis
if pod_basis_mode == "all":
    pod_modes = pod_mode_set["monolithic"][:, :n_modes]
    if not is_csdl(pod_modes):
        pod_modes = csdl.Variable(value=pod_modes)

else:
    pod_modes = np.zeros((n_local_dofs, n_modes))
    start_dof_idx  = 0
    end_dof_idx    = 0
    start_mode_idx = 0
    end_mode_idx   = 0
    
    for key in key_list:
        n_modes_key  = n_mode_set[key]
        end_dof_idx  = start_dof_idx  + pod_mode_set[key].shape[0]
        end_mode_idx = start_mode_idx + n_modes_key 
        pod_modes[start_dof_idx:end_dof_idx, start_mode_idx:end_mode_idx] = pod_mode_set[key][:, :n_modes_key]
        
        start_dof_idx  = end_dof_idx
        start_mode_idx = end_mode_idx
    

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

    # --- FOM (for comparison) ---
    dafoam_solver        = DAFoamSolver(dafoam_instance)
    dafoam_solver_states = dafoam_solver.evaluate(dafoam_input_variables_group)

    dafoam_fn_model   = DAFoamFunctions(dafoam_instance, disable_jacvec_normalization=True)
    dafoam_fn_outputs = dafoam_fn_model.evaluate(dafoam_solver_states, dafoam_input_variables_group)

    # Interpolated initial state from training data (weights already sum to 1)
    w_interp        = reference_fom_state + (snapshot_matrix - reference_fom_state[:, None]) @ snapshot_weights
    rom_interp_state  = csdl.experimental.mpi.mpi_allreduce(
        pod_modes.T() @ (weights * (w_interp - reference_fom_state) / scaling), comm=comm)
    
    # rom_pred_state = csdl.experimental.mpi.mpi_allreduce(pod_modes.T() @ (weights * (dafoam_solver_states - reference_fom_state) / scaling), comm=comm)

    # --- ROM model selection ---
    # rom_weights = 1 / residual_scaling ** 2 if inner_product_type == "corrective_only" else weights / residual_scaling ** 2
    rom_weights = 1 / residual_scaling
    if inner_product_type == "geometric_and_corrective":
        rom_weights = weights * rom_weights
    if rom_model_type != "galerkin_analytical":
        rom_weights = rom_weights / residual_scaling

    if phi_mode == "computed":
        rom_model = DAFoamPhiComputingLSPGModel(
            dafoam_input_variables_group = dafoam_input_variables_group,
            pod_modes                    = pod_modes,
            reference_fom_state          = reference_fom_state,
            scaling                      = scaling,
            weights                      = rom_weights,
            dafoam_instance              = dafoam_instance_rom,
            fd_step                      = fd_step,
            disable_presolve_diagnostics = False,
        )
    elif rom_model_type == "lspg_qr":
        # FD-LSPG but the Gauss-Newton step is solved via TSQR of M^(1/2) Psi
        # instead of the normal equations -> avoids squaring cond(Psi). Forward only.
        rom_model = DAFoamLSPGQRModel(
            dafoam_input_variables_group = dafoam_input_variables_group,
            pod_modes                    = pod_modes,
            reference_fom_state          = reference_fom_state,
            scaling                      = scaling,
            weights                      = rom_weights,
            dafoam_instance              = dafoam_instance_rom,
            fd_step                      = fd_step,
            disable_presolve_diagnostics = False,
        )
    elif rom_model_type == "galerkin_analytical":
        # Exact reduced Jacobian via reverse-mode AD (no FD, no normal-equations squaring).
        rom_model = DAFoamGalerkinModel(
            dafoam_input_variables_group = dafoam_input_variables_group,
            pod_modes                    = pod_modes,
            reference_fom_state          = reference_fom_state,
            scaling                      = scaling,
            weights                      = rom_weights,
            dafoam_instance              = dafoam_instance_rom,
            fd_step                      = fd_step,
            disable_presolve_diagnostics = False,
            jac_mode                     = "analytical",
        )
    else:
        rom_model = DAFoamLSPGModel(
            dafoam_input_variables_group = dafoam_input_variables_group,
            pod_modes                    = pod_modes,
            reference_fom_state          = reference_fom_state,
            scaling                      = scaling,
            weights                      = rom_weights,
            dafoam_instance              = dafoam_instance_rom,
            fd_step                      = fd_step,
            disable_presolve_diagnostics = False,
        )

    # lspg_qr / galerkin_analytical recompute the basis each iteration, so use Newton
    # (Broyden's secant updates are inconsistent with the per-iteration change of basis).
    solver_options = {"tol_rel": 1e-12, "tol_step_abs": 1e-13}
    if rom_model_type in ("lspg_qr", "galerkin_analytical"):
        rom_solver = NewtonSolver(options=solver_options)
    else:
        rom_solver = BroydenNewtonSolver(options=solver_options)

    rom_wrapper = CSDLROMWrapper(
        model                      = rom_model,
        solver                     = rom_solver,
        constant_initial_rom_state = rom_interp_state,#np.zeros((pod_modes.shape[1], ))
    )
    rom_states = rom_wrapper.evaluate()

    # CSDL expression for the state: linear reconstruction via the block-diagonal Phi.
    state_est = reference_fom_state + scaling * (pod_modes @ rom_states)

    rom_fn_model   = DAFoamFunctions(dafoam_instance_rom, disable_jacvec_normalization=True)
    rom_fn_outputs = rom_fn_model.evaluate(state_est, dafoam_input_variables_group)

    for out_name in dafoam_instance.getOption("function").keys():
        mpi_region.set_as_global_output(getattr(rom_fn_outputs,    out_name))
        mpi_region.set_as_global_output(getattr(dafoam_fn_outputs, out_name))

    mpi_region.set_as_global_output(state_est)
    mpi_region.set_as_global_output(dafoam_solver_states)

# Design variables for geometric test sweep
percent_change_in_thickness_dof.set_as_design_variable(lower=-10, upper=10, scaler=1./10)
normalized_percent_camber_change_dof.set_as_design_variable(lower=-10, upper=10, scaler=1./10)

objective_fun = -rom_fn_outputs.lift / rom_fn_outputs.drag
objective_fun.set_as_objective()
objective_fun.name = "-L/D_ROM_per_var"

recorder.stop()


# ===============================
# region SIMULATION AND TEST
# ===============================
sim        = csdl.experimental.PySimulator(recorder)
state_info = data_generator.state_info

snapshot_vars_and_limits = {
    percent_change_in_thickness_dof: {
        'range':     [-10, 10],
        'ref_value': 0,
    },
    normalized_percent_camber_change_dof: {
        'range':     [-10, 10],
        'ref_value': 0,
    },
}

data_generator._generate_lhs_samples(snapshot_vars_and_limits, num_samples=num_test_samples, random_state=42)
num_samples_with_ref = num_test_samples + 1

rom_drag = []
fom_drag = []
rom_lift = []
fom_lift = []
drag_rel = []
lift_rel = []
drag_recon_w = []
drag_recon_u = []
lift_recon_w = []
lift_recon_u = []
drag_rel_recon_w = []
drag_rel_recon_u = []
lift_rel_recon_w = []
lift_rel_recon_u = []
snap_weights = []
test_samples = []
projcurve_w  = []   # per-query weighted-basis   projection rel-err vs n (numpy, len k)
projcurve_u  = []   # per-query unweighted-basis projection rel-err vs n (numpy, len k)
proj_w       = []   # per-query weighted-basis   projection rel-err at fixed n_modes
proj_u       = []   # per-query unweighted-basis projection rel-err at fixed n_modes
rom_state_e  = []   # per-query ACTUAL ROM state rel-err (W, scaled) — what LSPG achieves
proj_snorm2  = []   # per-query ||s_tilde||_W^2 — used to drop the ~zero reference point

for i in range(num_samples_with_ref):
    for var, info in snapshot_vars_and_limits.items():
        sim[var] = info["samples"][i]
    test_samples.append(np.concatenate([var["samples"][i] for var in snapshot_vars_and_limits.values()]))
    sim.run()

    snap_weights.append(snapshot_weights.value)
    rom_drag.append(rom_fn_outputs.drag.value[0])
    fom_drag.append(dafoam_fn_outputs.drag.value[0])
    rom_lift.append(rom_fn_outputs.lift.value[0])
    fom_lift.append(dafoam_fn_outputs.lift.value[0])

    drag_rel.append(abs(rom_drag[i] - fom_drag[i]) / (abs(fom_drag[i]) + 1e-300))
    lift_rel.append(abs(rom_lift[i] - fom_lift[i]) / (abs(fom_lift[i]) + 1e-300))
        

    # Per-variable state error
    w_rom = dafoam_instance_rom.getStates()
    w_fom = dafoam_solver_states.value

    # ---- Weighted-POD basis quality diagnostic (gold standard) ----
    # Project the TRUE FOM state at this (interpolated) query config onto the
    # weighted vs. unweighted truncated bases, in the W (cell-volume) inner
    # product. If the weighted basis has a LARGER projection error, the rotation
    # has moved the retained subspace away from what the query actually needs —
    # the direct mechanism for "weighted POD performs worse".
    if pod_basis_mode == "all":
        def _w_proj_rel_err(B, s_tilde, W):
            # B: (n_local, n) W-orthonormal columns; s_tilde, W: (n_local,)
            c   = comm.allreduce(B.T @ (W * s_tilde), op=MPI.SUM)   # W-orthogonal projection coeffs
            rec = B @ c

            w_rec = rec * scaling + reference_fom_state
            dafoam_instance.setStates(w_rec)

            lift_recon = dafoam_instance.solver.calcFunction("lift")
            drag_recon = dafoam_instance.solver.calcFunction("drag")


            num = comm.allreduce(np.sum(W * (s_tilde - rec) ** 2), op=MPI.SUM)
            den = comm.allreduce(np.sum(W * s_tilde ** 2),         op=MPI.SUM)
            return np.sqrt(num / (den + 1e-300)), lift_recon, drag_recon

        s_tilde = (w_fom - reference_fom_state) / scaling          # scaled FOM state at query
        B_w     = np.asarray(pod_modes.value)                      # truncated basis
        err_w, lift_r_w, drag_r_w = _w_proj_rel_err(B_w, s_tilde, weights)
        proj_w.append(err_w)
        lift_recon_w.append(lift_r_w)
        drag_recon_w.append(drag_r_w)
        lift_rel_recon_w.append(abs(lift_r_w - fom_lift[i]) / abs(fom_lift[i] + 1e-300))
        drag_rel_recon_w.append(abs(drag_r_w - fom_drag[i]) / abs(fom_drag[i] + 1e-300))

        if snapshot_weight_type is not None:
            B_u = pod_mode_set_unweighted["monolithic"][:, :n_modes]
            err_u, lift_r_u, drag_r_u = _w_proj_rel_err(B_u, s_tilde, weights)
            proj_u.append(err_u)
            lift_recon_u.append(lift_r_u)
            drag_recon_u.append(drag_r_u)
            lift_rel_recon_u.append(abs(lift_r_u - fom_lift[i]) / abs(fom_lift[i] + 1e-300))
            drag_rel_recon_u.append(abs(drag_r_u - fom_drag[i]) / abs(fom_drag[i] + 1e-300))

        # ACTUAL ROM state error (scaled, W) — comparable to the projection errors.
        # If this is >> proj_w, the LSPG solve is failing to reach the best-in-basis
        # solution (conditioning / residual minimum). If it's ~= proj_w but forces
        # are still off, the basis IS the limit and forces are just sensitive.
        s_rom_t = (w_rom - reference_fom_state) / scaling
        sn2     = comm.allreduce(np.sum(weights * s_tilde ** 2), op=MPI.SUM)
        rnum    = comm.allreduce(np.sum(weights * (s_rom_t - s_tilde) ** 2), op=MPI.SUM)
        rom_state_e.append(np.sqrt(rnum / (sn2 + 1e-300)))
        proj_snorm2.append(sn2)

        # ---- Residual-landscape check (the decisive one) ----
        # 3 solvers (incl. exact-AD Galerkin) agreeing on a state ~27% off, with
        # proj<1%, means the residual's minimizer in the trial subspace is NOT the
        # accurate state. Compare the residual the solver minimizes at:
        #   (a) w_proj : projected TRUE state (best-in-basis reconstruction)
        #   (b) w_rom  : the ROM's converged solution
        #   (c) w_fom  : the TRUE FOM state (should be ~0 on this geometry)
        # Reads:
        #   rn_fom NOT ~0      -> ROM residual operator inconsistent with the FOM
        #                         (bounded vars / frozen turbulence / wall funcs).
        #   rn_proj >> rn_rom  -> residual landscape problem: the weighted residual
        #                         norm's subspace-minimum is far from the accurate
        #                         state (inner-product / weighting / hyper-reduction).
        #   rn_proj ~ rn_rom small but states differ -> insensitive directions.
        m_res = rom_model.weights
        def _res_vec(w):
            return rom_model._eval_fom_residual(fom_state=w)
        def _res_norm_of(r):
            return np.sqrt(comm.allreduce(np.sum(m_res * r * r), op=MPI.SUM))
        q_proj  = comm.allreduce(B_w.T @ (weights * s_tilde), op=MPI.SUM)  # cell-vol W-projection
        w_proj  = reference_fom_state + scaling * (B_w @ q_proj)
        r_proj  = _res_vec(w_proj)
        rn_proj = _res_norm_of(r_proj)
        rn_rom  = _res_norm_of(_res_vec(w_rom))
        rn_fom  = _res_norm_of(_res_vec(w_fom))
        rom_model._set_fom_states(w_rom)   # restore (the evals above perturbed the state)
        if rank == 0:
            print(f"  [res landscape] ||W^0.5 r||  proj(best-in-basis)={rn_proj:.4e}  "
                  f"rom_solution={rn_rom:.4e}  true_fom={rn_fom:.4e}")

        # Per-variable breakdown of the best-in-basis residual: which physics does the
        # energy-truncated subspace fail to represent in RESIDUAL terms? The dominant
        # variable is the lever (rescale its residual, add modes for it, or freeze it).
        # (allreduce on every rank — collective — then print on rank 0 only.)
        res_by_var = {
            var: np.sqrt(comm.allreduce(np.sum(m_res[info["indices"]] * r_proj[info["indices"]] ** 2), op=MPI.SUM))
            for var, info in state_info.items()
        }
        if rank == 0:
            print("  [res by var @ best-in-basis] "
                  + "  ".join(f"{var}={val:.3e}" for var, val in res_by_var.items()))

        if snapshot_weight_type is not None:
            w_np    = np.asarray(snapshot_weights.value)
            n_eff   = (w_np.sum() ** 2) / (np.sum(w_np ** 2) + 1e-300) # participation ratio
            # Singular spectrum of the reweighted ensemble D = Sigma VT diag(omega):
            D_np    = (np.asarray(singular_val_set["monolithic"])[:, None]
                       * np.asarray(VT_set["monolithic"])) * w_np[None, :]
            S_D     = np.linalg.svd(D_np, compute_uv=False)
            cum_D   = np.cumsum(S_D ** 2) / (np.sum(S_D ** 2) + 1e-300)
            n_99    = int(np.argmax(cum_D >= target_variance)) + 1     # modes to hit target on WEIGHTED energy

            if rank == 0:
                flag = "  <-- weighted WORSE" if err_w > err_u else ""
                print(f"  [basis proj] FOM-state W-projection rel err: "
                      f"unweighted={err_u:.4e}  weighted={err_w:.4e}{flag}")
                print(f"  [basis proj] weight n_eff={n_eff:.1f} of {w_np.size} snaps | "
                      f"keeping n_modes={n_modes}, but weighted-energy needs {n_99} for "
                      f"{target_variance:.3f} (S_D tail beyond ~{n_eff:.0f} is noise)")
        else:
            if rank == 0:
                print(f"  [basis proj] FOM-state W-projection rel err: {err_w:.4e}")

        # ---- Projection rel-err vs n, for choosing a SINGLE fixed n ----
        # For a W-orthonormal basis the captured energy with the first n columns
        # is the cumulative sum of squared projection coefficients, so the whole
        # error-vs-n curve costs one coeff vector per basis (no per-n re-solve).
        Phi_u_full = pod_mode_set["monolithic"].value if is_csdl(pod_mode_set["monolithic"]) else pod_mode_set["monolithic"]                         # (n_local, k) all stored modes
        snorm2     = comm.allreduce(np.sum(weights * s_tilde ** 2), op=MPI.SUM) + 1e-300
        c_u        = comm.allreduce(Phi_u_full.T @ (weights * s_tilde), op=MPI.SUM)   # (k,)
        projcurve_u.append(np.sqrt(np.maximum(1.0 - np.cumsum(c_u ** 2) / snorm2, 0.0)))

        if snapshot_weight_type is not None:
            U_D_np, _, _ = np.linalg.svd(
                (np.asarray(singular_val_set["monolithic"])[:, None]
                 * np.asarray(VT_set["monolithic"])) * w_np[None, :],
                full_matrices=False
            )
            B_w_full = Phi_u_full @ U_D_np                              # (n_local, k_w) full rotated basis
            c_w      = comm.allreduce(B_w_full.T @ (weights * s_tilde), op=MPI.SUM)   # (k_w,)
            projcurve_w.append(np.sqrt(np.maximum(1.0 - np.cumsum(c_w ** 2) / snorm2, 0.0)))

    if rank == 0:
        print(f"  Per-variable relative L2 error (sample {i}):")
    for var, info in state_info.items():
        idx  = info["indices"]
        err  = np.sqrt(comm.allreduce(np.sum((w_rom[idx] - w_fom[idx])**2), op=MPI.SUM))
        norm = np.sqrt(comm.allreduce(np.sum(w_fom[idx]**2),                op=MPI.SUM))
        if rank == 0:
            print(f"    {var:>10}: {err / (norm + 1e-300):.4e}")

if rank == 0 and projcurve_u:
    # Worst-case (over all query configs) projection rel-err as a function of the
    # number of retained modes n. Pick the single fixed n where the worst-case
    # curve crosses your tolerance — that n is safe for every query.
    import numpy as np
    kmin = min(c.size for c in projcurve_u)
    # Drop near-reference queries (s_tilde ~ 0 => 0/0 noise that swamps the max).
    sn2  = np.asarray(proj_snorm2)
    keep = sn2 > 1e-6 * sn2.max()
    if not keep.all():
        print(f"[single-n] dropping {int((~keep).sum())} near-reference query(ies) "
              f"from worst-case (s_tilde≈0).")
    U_stack = np.vstack([c[:kmin] for c, k in zip(projcurve_u, keep) if k])
    U_wc    = np.max(U_stack, axis=0)
    n_axis  = np.arange(1, kmin + 1)

    if projcurve_w:
        kmin_w  = min(c.size for c in projcurve_w)
        kmin    = min(kmin, kmin_w)
        W_stack = np.vstack([c[:kmin] for c, k in zip(projcurve_w, keep) if k])
        W_wc    = np.max(W_stack, axis=0)
        U_wc    = U_wc[:kmin]
        n_axis  = n_axis[:kmin]
        print("\n[single-n] worst-case-over-queries projection rel-err vs n:")
        print(f"{'n':>5}  {'unweighted':>12}  {'weighted':>12}")
        for n in [5, 10, 15, 20, 30, 40, 50, 75, 100]:
            if n <= kmin:
                print(f"{n:>5}  {U_wc[n-1]:>12.4e}  {W_wc[n-1]:>12.4e}")
        for tol in (1e-2, 5e-3, 1e-3):
            nw = int(n_axis[W_wc <= tol][0]) if np.any(W_wc <= tol) else None
            nu = int(n_axis[U_wc <= tol][0]) if np.any(U_wc <= tol) else None
            print(f"[single-n] smallest n for worst-case rel-err <= {tol:.0e}:  "
                  f"weighted={nw}   unweighted={nu}")
    else:
        print("\n[single-n] worst-case-over-queries projection rel-err vs n:")
        print(f"{'n':>5}  {'proj_err':>12}")
        for n in [5, 10, 15, 20, 30, 40, 50, 75, 100]:
            if n <= kmin:
                print(f"{n:>5}  {U_wc[n-1]:>12.4e}")
        for tol in (1e-2, 5e-3, 1e-3):
            nu = int(n_axis[U_wc <= tol][0]) if np.any(U_wc <= tol) else None
            print(f"[single-n] smallest n for worst-case rel-err <= {tol:.0e}:  {nu}")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.semilogy(n_axis, U_wc, label='unweighted (worst case)')
    if projcurve_w:
        plt.semilogy(n_axis, W_wc, label='weighted (worst case)')
    plt.xlabel('n retained modes'); plt.ylabel('worst-case proj rel-err'); plt.legend()
    plt.title('Choosing a single fixed n'); plt.savefig('single_n_choice.png', dpi=200)

if rank == 0:
    print(f"\n{'Pt':>4}  {'ROM drag':>12}  {'FOM drag':>12}  {'drag_rel_err':>14}  "
          f"{'ROM lift':>12}  {'FOM lift':>12}  {'lift_rel_err':>14}")
    for i in range(num_samples_with_ref):
        print(f"{i:>4d}  {rom_drag[i]:>12.4e}  {fom_drag[i]:>12.4e}  {drag_rel[i]:>14.4e}  "
                f"{rom_lift[i]:>12.4e}  {fom_lift[i]:>12.4e}  {lift_rel[i]:>14.4e}")

# ---- Basis quality vs ROM quality (LSPG-vs-basis split) ----
# proj_w    = best-possible (orthogonal-projection) rel-err of the FOM state in the
#             (possibly weighted-rotated) basis at the fixed n_modes.
# proj_u    = same for the unweighted basis (only available when snapshot_weight_type set).
# drag_rel/lift_rel = actual ROM output error.
# If the weighted basis projects BETTER (proj_w < proj_u) yet the ROM is WORSE,
# the loss is in the LSPG solve (conditioning / residual minimum), not the basis.
if rank == 0 and proj_w:
    # proj_w   = best-possible (orthogonal projection) state rel-err in the basis
    # rom_st   = ACTUAL state rel-err the LSPG ROM achieves
    # gap      = rom_st / proj_w : how far LSPG is from best-in-basis (>~3 => LSPG suboptimal)
    have_u = bool(proj_u)
    header = (f"\n{'Pt':>4}  {'proj_w':>11}  {'rom_st':>11}  {'gap':>7}  "
              f"{'drag_rel':>11}  {'drag_rel_proj_w':>16}  "
              f"{'lift_rel':>11}  {'lift_rel_proj_w':>16}")
    if have_u:
        header += f"  {'proj_u':>11}  {'drag_rel_proj_u':>16}  {'lift_rel_proj_u':>16}"
    header += f"  {'verdict':>26}"
    print(header)
    for i in range(len(proj_w)):
        if proj_snorm2[i] <= 1e-6 * max(proj_snorm2):
            continue                                              # skip reference point (s_tilde≈0)
        rom_err = max(drag_rel[i], lift_rel[i])
        gap     = rom_state_e[i] / (proj_w[i] + 1e-300)          # LSPG suboptimality factor
        verdict = ""
        if rom_err > 1e-2:
            if gap > 3.0:
                verdict = "LSPG suboptimal (state)"
            else:
                verdict = "basis-limited / forces"      # state ~ best-in-basis; forces sensitive
        row = (f"{i:>4d}  {proj_w[i]:>11.4e}  {rom_state_e[i]:>11.4e}  "
               f"{gap:>7.1f}  {drag_rel[i]:>11.4e}  {drag_rel_recon_w[i]:>16.4e}  "
               f"{lift_rel[i]:>11.4e}  {lift_rel_recon_w[i]:>16.4e}")
        if have_u:
            row += (f"  {proj_u[i]:>11.4e}  {drag_rel_recon_u[i]:>16.4e}  "
                    f"{lift_rel_recon_u[i]:>16.4e}")
        row += f"  {verdict:>26}"
        print(row)

# ---- Collect weights over all test configs ----
# Run this in your test loop and accumulate:
# all_weights      : list of n_test numpy arrays, each (n_train,)
# all_test_configs : list of n_test numpy arrays, each (n_dv,)
#
# e.g. inside the test loop:
#   all_weights.append(snapshot_weights.value)
#   all_test_configs.append(current_config.value)
if rank == 0:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from sklearn.decomposition import PCA

    param_names = ['camber_1', 'camber_2', 'camber_3',
                'thick_1',  'thick_2',  'thick_3']

    all_points = np.vstack([snapshot_configs_full] + [c[None, :] for c in test_samples])
    pca        = PCA(n_components=2)
    pca.fit(all_points)
    train_2d   = pca.transform(snapshot_configs_full)                        # (n_train, 2)
    test_2d    = pca.transform(np.array(test_samples))                   # (n_test,  2)

    n_test = len(snap_weights)

    # ── Figure 1: PCA overview (one subplot per test point) ─────────────────
    fig, axes = plt.subplots(2, 6, figsize=(20, 8), constrained_layout=True)
    axes = axes.flatten()

    for i in range(n_test):
        ax  = axes[i]
        w   = np.array(snap_weights[i])
        sc  = ax.scatter(train_2d[:, 0], train_2d[:, 1],
                        c=w, cmap='viridis', s=25,
                        norm=plt.Normalize(vmin=0, vmax=w.max()))
        ax.scatter(*test_2d[i], c='red', s=200, marker='*', zorder=5)
        plt.colorbar(sc, ax=ax, shrink=0.8)
        nnz = int(np.sum(w > 1e-6 * w.max()))
        ax.set_title(f'Test {i}  max={w.max():.3f}  nnz={nnz}', fontsize=9)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=7)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=7)

    fig.suptitle('Snapshot weights — PCA projection of 6D param space\n(red ★ = test point)')
    plt.savefig('weights_pca.png', dpi=300)
    plt.show()

    # ── Figure 2: Sorted weight bar + cumulative mass per test point ─────────
    fig, axes = plt.subplots(2, 6, figsize=(20, 6), constrained_layout=True)
    axes = axes.flatten()

    for i in range(n_test):
        ax  = axes[i]
        w   = np.sort(np.array(snap_weights[i]))[::-1]
        cum = np.cumsum(w)
        ax.bar(range(len(w)), w, width=1.0, color='steelblue', alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(cum, color='orange', linewidth=1.5)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel('Cumul. mass', fontsize=7, color='orange')
        ax.set_title(f'Test {i}', fontsize=9)
        ax.set_xlabel('Snapshot rank', fontsize=7)
        ax.set_ylabel('Weight', fontsize=7)
        # Mark 50% and 90% mass thresholds
        for thresh, ls in [(0.5, '--'), (0.9, ':')]:
            idx = np.searchsorted(cum, thresh)
            ax2.axhline(thresh, color='orange', linestyle=ls, linewidth=0.8)
            ax.axvline(idx, color='gray', linestyle=ls, linewidth=0.8, label=f'{thresh*100:.0f}%→{idx}snaps')
        ax.legend(fontsize=6)

    fig.suptitle('Sorted snapshot weights (orange = cumulative mass)')
    plt.savefig('weights_sorted.png', dpi=300)
    plt.show()

    # ── Figure 3: Pairplot for ONE test point (pick the worst-performing) ────
    for focus_idx in range(len(snap_weights)):
        # focus_idx  = 0   # <── change to whichever test is performing worst
        w          = np.array(snap_weights[focus_idx])
        query      = np.array(test_samples[focus_idx])
        predicted  = w @ snapshot_configs_full          # weighted centroid of training configs
        n_params   = len(param_names)
        norm       = plt.Normalize(vmin=0, vmax=w.max())

        fig, axes  = plt.subplots(n_params, n_params, figsize=(14, 14), squeeze=False)
        fig.suptitle(f'Pairplot — test config {focus_idx}  '
                     f'(★=query, ◆=predicted, err={np.linalg.norm(predicted - query):.3f})', fontsize=12)

        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i][j]
                ax.tick_params(labelsize=5)
                if i == j:
                    ax.hist(snapshot_configs_full[:, i], bins=15, color='gray', alpha=0.6)
                    ax.axvline(query[i],     color='red',    linewidth=2, label='query')
                    ax.axvline(predicted[i], color='orange', linewidth=2, linestyle='--', label='predicted')
                else:
                    sc = ax.scatter(snapshot_configs_full[:, j], snapshot_configs_full[:, i],
                                    c=w, cmap='viridis', norm=norm, s=10, alpha=0.85)
                    ax.scatter(query[j],     query[i],     c='red',    s=15, marker='*', zorder=5, label='query')
                    ax.scatter(predicted[j], predicted[i], c='orange', s=15, marker='D', zorder=5, label='predicted')
                if j == 0:
                    ax.set_ylabel(param_names[i], fontsize=7)
                if i == n_params - 1:
                    ax.set_xlabel(param_names[j], fontsize=7)

        axes[0][1].legend(fontsize=6, loc='upper right')  # legend on one off-diagonal panel

        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cbar_ax, label='Weight')
        plt.savefig(f'weights_pairplot_test{focus_idx}.png', dpi=300, bbox_inches='tight')
        # plt.show()
