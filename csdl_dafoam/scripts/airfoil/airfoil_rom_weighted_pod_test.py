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

# Hashing (for file name generation)
import hashlib

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
problem_name              = 'rom_with_interpolation_comparison'

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
U0        = 206.53653128321116         # used for normalizing CD and CL
p0        = 19509.303373738785
T0        = 216.65227163736915
nuTilda0  = 4.5e-5
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
dataset_keyword       = 'training_data4'
storage_location      = Path(dafoam_directory)


# ===============================
# region SETUP
# ===============================
# MPI information
rank      = comm.Get_rank()
comm_size = comm.Get_size()
rank_str  = f"{rank:0{len(str(comm_size-1))}d}" # string with zero-padded rank index (for prints)


# region DAFoam instance
dafoam_instance              = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)
dafoam_instance_rom          = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)
dafoam_instance_rom_weighted = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)
x_surf_dafoam_initial_mpi    = dafoam_instance.getSurfaceCoordinates()
x_vol_dafoam_initial_mpi     = dafoam_instance.xv0

local_n_surf  = x_surf_dafoam_initial_mpi.shape[0]
local_n_vol   = x_vol_dafoam_initial_mpi.shape[0]

# Gathering surface mesh to rank 0 (need to do this to avoid 'no-element' ranks in the projection
# and geometry evaluation functions)
(x_surf_dafoam_initial, 
x_surf_dafoam_initial_size,
x_surf_dafoam_initial_indices) = gather_array_to_rank0(x_surf_dafoam_initial_mpi, comm)

# Get hash for surface mesh projection file read/write (broadcast to other ranks)
if rank == 0:
    x_surf_hash = hash_array_tol(x_surf_dafoam_initial, tol=1e-8)
else:
    x_surf_hash = None

x_surf_hash = comm.bcast(x_surf_hash, root=0)

# region File paths
geometry_pickle_file_path         = Path(geometry_directory)/geometry_pickle_file_name
stp_file_path                     = Path(geometry_directory)/stp_file_name
surface_mesh_projection_file_path = Path(dafoam_directory)/f'projected_surface_mesh_{x_surf_hash}.pickle'

# POD Data import
data_generator = TrainingDataInterface(dafoam_instance=dafoam_instance, 
                                        storage_location=storage_location, 
                                        dataset_keyword=dataset_keyword,
                                        h5_file_base_name="point")

# Manually obtaining the file for now
data = data_generator.load_h5(Path(storage_location)/dataset_keyword/"point_0.h5", only_distributed_data=False)


# ===============================
# region CSDL RECORDER
# ===============================
# recorder 
recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()


geometry = lsdo_geo.import_geometry(stp_file_path,
                                    parallelize=False)


# region Surface mesh projection
# Now do we do the same check for the surface mesh projection
if surface_mesh_projection_file_path.is_file():
    if rank == 0:
        print('Found surface mesh projection pickle!')
    projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)

else:
    if rank == 0:
        print(f'No projected surface mesh file found at {surface_mesh_projection_file_path}')
        try:
            # ORIGINAL CODE
            # with Timer('projecting on surface mesh'):
            #     projected_surf_mesh_dafoam = geometry.project(
            #         x_surf_dafoam_initial, 
            #         grid_search_density_parameter = 1,      # 1     (ORIGINAL)
            #         projection_tolerance          = 1e-4,   #1.e-3m (ORIGINAL)
            #         grid_search_density_cutoff    = 50,     # 20    (ORIGINAL) 50
            #         force_reprojection            = False,
            #         plot                          = False    # UCSD_LAB
            #     )

            # Debugging/timing
            import cProfile
            import pstats
            with cProfile.Profile() as pr:
                projected_surf_mesh_dafoam = geometry.project(
                    x_surf_dafoam_initial, 
                    grid_search_density_parameter = 1,      # 1     (ORIGINAL)
                    projection_tolerance          = 1e-10,   #1.e-3m (ORIGINAL)
                    grid_search_density_cutoff    = 50,     # 20    (ORIGINAL) 50
                    force_reprojection            = False,
                    plot                          = False    # UCSD_LAB
                )
            # Summarize top time-consuming functions
            stats = pstats.Stats(pr)
            stats.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats(30)

            print('Writing surface mesh projection pickle...')
            write_simple_pickle(projected_surf_mesh_dafoam, surface_mesh_projection_file_path)
            print('Done!')

        # Added this exception because I was getting an ungraceful MPI termination
        except Exception as e:
            import traceback
            print(f"[Rank 0 ERROR] Projection/pickle step failed:\n{traceback.format_exc()}", flush=True)
            comm.Abort(1) # Abort MPI processes instead of letting them hang

    comm.Barrier()
    if rank != 0:
        projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)

print(f'Rank {rank_str} done reading projected surface mesh!')
comm.Barrier()

# -------------------------------------------------------------------------------------------
# COPY PASTED GEOMETRY STUFF HERE:
# region Create Parameterization Objects
num_ffd_coefficients_chordwise = 5
num_ffd_sections               = 2  # Symmetry boundaries (left, right)
ffd_block = construct_ffd_block_around_entities(entities=geometry, 
                                                num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2), degree=(3,1,1))

# region CSDL Variable declaration
percent_change_in_thickness          = csdl.Variable(shape=(num_ffd_coefficients_chordwise, num_ffd_sections), value=0.) # (5,2)
percent_change_in_thickness_dof      = csdl.Variable(shape=(num_ffd_coefficients_chordwise-2,), value=np.array([0,0,0])) 
normalized_percent_camber_change     = csdl.Variable(shape=(num_ffd_coefficients_chordwise, num_ffd_sections),  value=0.)
normalized_percent_camber_change_dof = csdl.Variable(shape=(num_ffd_coefficients_chordwise-2,), value=np.array([0,0,0]))

# ffd_block.plot()
ffd_sectional_parameterization = VolumeSectionalParameterization(
    name="ffd_sectional_parameterization",
    parameterized_points=ffd_block.coefficients,    # ffd_block.coefficients.shape = (5, 2, 2, 3)
    principal_parametric_dimension=1,
)


# region Evaluate Inner Parameterization Map To Define Forward Model For Parameterization Solver
sectional_parameters = VolumeSectionalParameterizationInputs()
ffd_coefficients     = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)


# Apply shape variables (NEW) : (1) THICKNESS
original_block_thickness    = ffd_block.coefficients.value[0, 0, 1, 2] - ffd_block.coefficients.value[0, 0, 0, 2] # normal-thickness  
percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[1:-1,0], percent_change_in_thickness_dof)
percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[1:-1,1], percent_change_in_thickness_dof)
delta_block_thickness       = (percent_change_in_thickness / 100) * original_block_thickness
thickness_upper_translation = 1/2 * delta_block_thickness
thickness_lower_translation = -thickness_upper_translation

ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,1,2], ffd_coefficients[:,:,1,2] + thickness_upper_translation)
ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,0,2], ffd_coefficients[:,:,0,2] + thickness_lower_translation)


# Parameterize camber change as normalized by the original block (kind of like chord) length (NEW) : (2) CAMBER
normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1, 0], normalized_percent_camber_change_dof)
normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1, 1], normalized_percent_camber_change_dof)

block_length     = ffd_block.coefficients.value[1, 0, 0, 0] - ffd_block.coefficients.value[0, 0, 0, 0]
camber_change    = (normalized_percent_camber_change/100)*block_length
ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,:,2], ffd_coefficients[:,:,:,2] + csdl.expand(camber_change, (num_ffd_coefficients_chordwise, num_ffd_sections, 2), 'ij->ijk'))

geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
geometry.set_coefficients(geometry_coefficients) 
# -------------------------------------------------------------------------------------------

with Timer(f'evaluating geometry component', rank, TIMING_ENABLED):
    x_surf_dafoam_full = geometry.evaluate(projected_surf_mesh_dafoam, plot=False)

# region Surface mesh distribution
i0, i1          = x_surf_dafoam_initial_indices[rank]

# Flight condition variables
flight_conditions_group                 = csdl.VariableGroup()
flight_conditions_group.mach_number     = csdl.Variable(value=0.7, name="mach_number")
flight_conditions_group.angle_of_attack_deg = csdl.Variable(value=data["parameters"]["primary_variables"]["angle_of_attack_deg"], name="angle_of_attack_deg")
flight_conditions_group.altitude_m      = csdl.Variable(value=data["parameters"]["primary_variables"]["altitude_m"], name="altitude (m)")

# Atmospheric condition variables
ambient_conditions_group = sam.compute_ambient_conditions_group(flight_conditions_group.altitude_m)

with csdl.experimental.mpi.enter_mpi_region(rank, comm) as mpi_region:
    
    x_surf_dafoam   = x_surf_dafoam_full[i0:i1, :]
    x_surf_dafoam   = x_surf_dafoam.flatten()

    # region IDWarp and DAFoam
    idwarp_model    = DAFoamMeshWarper(dafoam_instance)
    x_vol_dafoam    = idwarp_model.evaluate(x_surf_dafoam)

    # Need to split up angle-of-attack (and any other CSDL variables which DAFoam takes the derivative with respect to)
    flight_conditions_group.angle_of_attack_deg = mpi_region.split_custom(flight_conditions_group.angle_of_attack_deg, split_func = lambda x:x)
    
    # DAFoam input variable generation
    # Generate our DAFoam CSDL input variable group 
    # (this will add airspeed_m_s to the flight conditions group if not already present)
    dafoam_input_variables_group = compute_dafoam_input_variables(dafoam_instance, 
                                                                ambient_conditions_group, 
                                                                flight_conditions_group,
                                                                x_vol_dafoam)
    
    # Assemble POD modes and relevant vectors:
    # Assemble POD modes and relevant vectors in DAFoam's state-vector ordering.
    # POD data is stored per-variable in the H5 file; info["indices"] gives each
    # variable's positions in DAFoam's local state vector (cell-interleaved for
    # adjStateOrdering="cell").  Scattering into those positions ensures that
    # setStates(), getResiduals(), and the ROM diagnostics all see consistently
    # ordered arrays.
    n_modes     = 20
    state_info  = data_generator.state_info
    n_local_states = dafoam_instance.getNLocalAdjointStates()
    s_vals          = data["pod"]["singular_values"]
    pod_modes       = np.zeros((n_local_states, s_vals.shape[0]))
    scaling         = np.zeros(n_local_states)
    weights         = np.zeros(n_local_states)
    reference_state = np.zeros(n_local_states)
    for state_var, info in state_info.items():
        idx = info["indices"]
        pod_modes[idx, :]    = data["pod"]["modes"][state_var]
        scaling[idx]         = data["pod"]["scaling"][state_var]
        weights[idx]         = data["pod"]["weights"][state_var]
        reference_state[idx] = data["pod"]["reference_state"][state_var]
    residual_scaling = np.ones_like(dafoam_instance.getStateWeights())
    residual_scaling[state_info["T"]["indices"]] *= 1005


    ### DIAGNOSTICS
    cumulative_ratio = np.cumsum(s_vals ** 2) / np.sum(s_vals ** 2)
    # print(cumulative_ratio)

    cutoff_index = np.where(cumulative_ratio >= 0.999)
    cutoff_index = cutoff_index[0][0]
    # print(np.where(np.cumsum(s_vals) / np.sum(s_vals) >= 0.999))
    print(cutoff_index, cumulative_ratio[cutoff_index])

    # Stable rank (soft rank — robust to noise)
    stable_rank = (np.sum(s_vals**2) / s_vals[0]**2)

    # Effective rank (entropy-based)
    p = s_vals**2 / np.sum(s_vals**2)       # normalized energy fractions
    eff_rank = np.exp(-np.sum(p * np.log(p + 1e-300)))

    print(f"Stable rank:    {stable_rank:.2f}")
    print(f"Effective rank: {eff_rank:.2f}")

    errors = []
    for r in range(1, len(s_vals) + 1):
        # energy not captured
        err = np.sqrt(np.sum(s_vals[r:]**2)) / np.sqrt(np.sum(s_vals**2))
        errors.append(err)

    import matplotlib.pyplot as plt
    if rank  == 0:
        plt.figure()
        plt.semilogy(range(1, len(s_vals)+1), errors)
        plt.xlabel("Number of POD modes r")
        plt.ylabel("Relative projection error (L2)")
        plt.title("POD projection error vs. rank")
        plt.grid(True)

    # snapshots   = np.array(np.concatenate([data["samples"]["states"][state_var] for state_var in state_info.keys()], axis=0))
    # scld_cntrd  = 1 / scaling[:, None] * (snapshots - reference_state[:, None])
    # phi_tmp     = pod_modes[:,:10]

    # coeffs      = comm.allreduce(phi_tmp.T @ (weights[:, None] * scld_cntrd), op=MPI.SUM)
    
    # import numpy as np
    # import pandas as pd
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # import matplotlib.pyplot as plt

    # camber_samples = data["parameters"]["secondary_variables"]["%_camber_change"]
    # thick_samples  = data["parameters"]["secondary_variables"]["%_thickness_change"]
    # params = np.concatenate([camber_samples, thick_samples], axis=1)

    # # if rank == 0:
    # # --- Build DataFrame ---
    # n_modes = coeffs.shape[0]
    # df = pd.DataFrame(
    #     coeffs.T,
    #     columns=[f"a{i+1}" for i in range(n_modes)]
    # )

    # # choose ONE parameter dimension to color by
    # # for i in range(params.shape[1]):
    # df["param"] = params[:, rank]

    # # --- Pairplot ---
    # sns.pairplot(
    #     df,
    #     vars=[f"a{i+1}" for i in range(n_modes)],
    #     hue="param",
    #     palette="viridis",
    #     corner=True,        # avoids redundant upper triangle
    #     plot_kws={"s": 25, "alpha": 0.8}
    # )

    # plt.suptitle(f"POD Coefficient Pairplot Colored by Parameter (index {rank})", y=1.02)
    # plt.show()

    # quiet_barrier(comm)
    ###


    ### TESTING REGION
    from csdl_dafoam.utils.interpolation import RBFInterpolator, IDWInterpolator
    from csdl_dafoam.utils.custom_explicit_reduced_svd import customExplicitReducedSVD
    import matplotlib.pyplot as plt

    s_vals          = data["pod"]["singular_values"]#[:n_modes]
    snapshots       = np.zeros((n_local_states, s_vals.shape[0]))
    for state_var, info in state_info.items():
        snapshots[info["indices"], :] = data["samples"]["states"][state_var][:, 1:]
    snapshot_configs = np.array(np.concatenate([data["parameters"]["secondary_variables"]["%_camber_change"], data["parameters"]["secondary_variables"]["%_thickness_change"]], axis=1))[1:, :]
    current_config  = csdl.concatenate((normalized_percent_camber_change_dof, percent_change_in_thickness_dof), axis=0)

    

    snapshot_weights_pre = RBFInterpolator(current_config, snapshot_configs, positive_non_reproducing_weights=True).weights() # IDWInterpolator(current_config, snapshot_configs, exponent=2).weights()
    snapshot_weights     = snapshot_weights_pre / csdl.sum(snapshot_weights_pre) # Enforce unity, if it happens to not be
    
    VT = 1 / s_vals[:, None] * comm.allreduce((pod_modes.T @ ((weights / scaling)[:, None] * (snapshots - reference_state[:, None]))), op=MPI.SUM)

    D = csdl.einsum(s_vals, csdl.einsum(VT, csdl.sqrt(snapshot_weights ** 2), action='ij,j->ij'), action='i,ij->ij')

    UD, SD, VTD = customExplicitReducedSVD().evaluate(A=D)

    pod_modes_weighted = pod_modes @ UD[:, :n_modes]

    num_samples=5

    snapshot_vars_and_limits = {
        percent_change_in_thickness_dof: {
            'name': '%_thickness_change',
            'range': [-5, 5],
            'ref_value': 0, 
        },
        normalized_percent_camber_change_dof: {
            'name': '%_camber_change',
            'range': [-5, 5],
            'ref_value': 0, 
        }
    }

    data_generator._generate_lhs_samples(snapshot_vars_and_limits, num_samples=num_samples, random_state=42)
    print(snapshot_vars_and_limits) if rank == 0 else None
    diag_dict = {"max_diff":{"weighted"     : {state:np.zeros(num_samples) for state in state_info.keys()}, "unweighted":{state:np.zeros(num_samples) for state in state_info.keys()}}, 
                "max_err":{"weighted"      : {state:np.zeros(num_samples) for state in state_info.keys()}, "unweighted":{state:np.zeros(num_samples) for state in state_info.keys()}}, 
                "err_norm":{"weighted"     : {state:np.zeros(num_samples) for state in state_info.keys()}, "unweighted":{state:np.zeros(num_samples) for state in state_info.keys()}}, 
                "proj_err":{"weighted"     : np.zeros(num_samples),         "unweighted":np.zeros(num_samples)}, 
                "proj_err_rel":{"weighted" : np.zeros(num_samples),         "unweighted":np.zeros(num_samples)}}

    if rank == 0:
        # plt.plot(np.cumsum(s_vals ** 2)/np.sum(s_vals ** 2), label="original", linestyle='--')
        # plt.plot(np.cumsum(SD.value ** 2)/np.sum(SD.value ** 2), label="weighted", linestyle='-.')
        # plt.title("Cumulative Energy Fraction")
        # plt.ylabel(r"$\sum_0^r{\sigma_i^2} / \sum{\sigma_i^2}$")
        # plt.xlabel("Number of modes, r")
        # plt.legend()

        # plt.figure()
        # plt.plot(s_vals, label="original", linestyle='--')
        # plt.plot(SD.value, label="weighted", linestyle='-.')
        # plt.title("Singular values")
        # plt.ylabel(r"$\sigma$")
        # plt.xlabel("Mode number")
        # plt.legend()

        # plt.show()


        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.pyplot as plt

        # --- Build the samples DataFrame ---
        camber_samples = data["parameters"]["secondary_variables"]["%_camber_change"]
        thick_samples  = data["parameters"]["secondary_variables"]["%_thickness_change"]
        params = np.concatenate([camber_samples, thick_samples], axis=1)

        if params.shape[0] == 6:
            params = params.T

        columns = ["camber_1", "camber_2", "camber_3",
                "thickness_1", "thickness_2", "thickness_3"]

        df = pd.DataFrame(params, columns=columns)
        df["type"] = "Sample"  # hue label for samples

        # --- Build the target row ---
        target_param = [info["samples"] for var, info in snapshot_vars_and_limits.items()]

        target_param_flat = np.concatenate(target_param, axis=1)  # shape (6,) after flattening

        target_df = pd.DataFrame(target_param_flat, columns=columns)
        target_df["type"] = "Target"

        # --- Combine into one DataFrame ---
        combined_df = pd.concat([df, target_df], ignore_index=True)

        # --- Pair plot with hue ---
        sns.set(style="ticks")
        palette = {"Sample": "steelblue", "Target": "crimson"}

        g = sns.pairplot(
            combined_df,
            hue="type",
            corner=True,
            palette=palette,
            diag_kind="kde",
            plot_kws={"alpha": 0.6},
            hue_order=["Sample", "Target"],  # samples drawn first, target on top
        )

        for i, row_col in enumerate(columns):
            for j, col_col in enumerate(columns):
                if i != j:
                    ax = g.axes[i, j]
                    if ax is None:  # upper triangle is None when corner=True
                        continue
                    for idx, row in target_df.iterrows():
                        ax.annotate(
                            str(idx),
                            xy=(row[col_col], row[row_col]),
                            xytext=(4, 4),           # pixel offset from the point
                            textcoords="offset points",
                            fontsize=7,
                            color=palette["Target"],
                        )

        # # --- Make the target point larger and more visible on scatter axes ---
        # # The single target row will render tiny by default, so re-draw it as a star
        # for i, row_col in enumerate(columns):
        #     for j, col_col in enumerate(columns):
        #         if i != j:  # off-diagonal only
        #             ax = g.axes[i, j]
        #             ax.scatter(
        #                 target_df[col_col],
        #                 target_df[row_col],
        #                 color=palette["Target"],
        #                 marker="*",
        #                 s=250,          # star size
        #                 zorder=5,       # draw on top
        #                 label="_nolegend_",
        #             )

        plt.suptitle("Parameter Pair Plot", y=1.02)
        # plt.show()

    quiet_barrier(comm)

    if rank == 0:
        plt.figure()
        plt.plot(np.cumsum(s_vals ** 2)/np.sum(s_vals ** 2), label="original", linestyle='--')
        plt.plot(np.cumsum(SD.value ** 2)/np.sum(SD.value ** 2), label="weighted", linestyle='-.')
        plt.title("Cumulative Energy Fraction")
        plt.ylabel(r"$\sum_0^r{\sigma_i^2} / \sum{\sigma_i^2}$")
        plt.xlabel("Number of modes, r")
        plt.legend()

        plt.figure()
        plt.plot(s_vals, label="original", linestyle='--')
        plt.plot(SD.value, label="weighted", linestyle='-.')
        plt.title("Singular values")
        plt.ylabel(r"$\sigma$")
        plt.xlabel("Mode number")
        plt.legend()

        plt.show()

    quiet_barrier(comm)
    ###

    pod_modes = pod_modes[:, :n_modes]

    from csdl_dafoam.core.rom.csdl_rom import CSDLROMWrapper
    from csdl_dafoam.core.rom.rom_models import DAFoamLSPGModel, DAFoamGalerkinModel
    from csdl_dafoam.core.rom.rom_solver import NewtonSolver, BroydenNewtonSolver
    
    # WEIGHTED ROM
    # dafoam_rom_model_weighted       = DAFoamGalerkinModel(dafoam_input_variables_group=dafoam_input_variables_group,
    #                                                   pod_modes=pod_modes_weighted,
    #                                                   reference_fom_state=reference_state,
    #                                                   scaling=scaling,
    #                                                   weights=1/residual_scaling,
    #                                                   dafoam_instance=dafoam_instance_rom_weighted,
    #                                                   normalize_residuals=False,
    #                                                   fd_step=1e-6,
    #                                                   solution_prefix="weighted")
    dafoam_rom_model_weighted       = DAFoamLSPGModel(dafoam_input_variables_group=dafoam_input_variables_group,
                                                      pod_modes=pod_modes_weighted,
                                                      reference_fom_state=reference_state,
                                                      scaling=scaling,
                                                      weights=1 / residual_scaling ** 2,
                                                      dafoam_instance=dafoam_instance_rom_weighted,
                                                      normalize_residuals=False,
                                                      fd_step=1e-6,
                                                      solution_prefix="weighted")
    
    dafoam_rom_weighted             = CSDLROMWrapper(model=dafoam_rom_model_weighted, 
                                                     solver=BroydenNewtonSolver(options={"tol_rel":1e-9, "tol_step_abs":1e-13}),
                                                     start_with_zero_state=True)   
    dafoam_rom_states_weighted      = dafoam_rom_weighted.evaluate()

    # Reconstruct state
    dafoam_state_estimate_weighted  = reference_state + scaling * (pod_modes_weighted @ dafoam_rom_states_weighted)

    # Functions
    dafoam_functions_rom_weighted   = DAFoamFunctions(dafoam_instance_rom_weighted, 
                                                      disable_jacvec_normalization=True)
    dafoam_function_rom_outputs_weighted = dafoam_functions_rom_weighted.evaluate(dafoam_state_estimate_weighted, 
                                                                                  dafoam_input_variables_group)
    

    # UNWEIGHTED ROM
    # dafoam_rom_model                = DAFoamGalerkinModel(dafoam_input_variables_group=dafoam_input_variables_group,
    #                                                   pod_modes=pod_modes_weighted,
    #                                                   reference_fom_state=reference_state,
    #                                                   scaling=scaling,
    #                                                   weights=1/residual_scaling,
    #                                                   dafoam_instance=dafoam_instance_rom_weighted,
    #                                                   normalize_residuals=False,
    #                                                   fd_step=1e-6,
    #                                                   solution_prefix="weighted")
    dafoam_rom_model                = DAFoamLSPGModel(dafoam_input_variables_group=dafoam_input_variables_group,
                                                      pod_modes=pod_modes,
                                                      reference_fom_state=reference_state,
                                                      scaling=scaling,
                                                      weights=1 / residual_scaling ** 2,
                                                      dafoam_instance=dafoam_instance_rom,
                                                      normalize_residuals=False,
                                                      fd_step=1e-6,
                                                      solution_prefix="unweighted")
    dafoam_rom                      = CSDLROMWrapper(model=dafoam_rom_model, 
                                                     solver=BroydenNewtonSolver(options={"tol_rel":1e-9, "tol_step_abs":1e-13}),
                                                     start_with_zero_state=True)   
    dafoam_rom_states               = dafoam_rom.evaluate()

    # Reconstruct state
    dafoam_state_estimate           = reference_state + scaling * (pod_modes @ dafoam_rom_states)

    # Functions
    dafoam_functions_rom            = DAFoamFunctions(dafoam_instance_rom, 
                                                      disable_jacvec_normalization=True)
    dafoam_function_rom_outputs     = dafoam_functions_rom.evaluate(dafoam_state_estimate, 
                                                                    dafoam_input_variables_group) 
    

    # DAFoamSolver Implicit component setup and evaluation
    dafoam_solver           = DAFoamSolver(dafoam_instance, write_residual_fields=True)
    dafoam_solver_states    = dafoam_solver.evaluate(dafoam_input_variables_group)

    # DAFoamFunctions Explicit component setup and evaluation
    dafoam_functions = DAFoamFunctions(dafoam_instance)
    dafoam_function_outputs = dafoam_functions.evaluate(dafoam_solver_states, 
                                                        dafoam_input_variables_group)
    
       
    
    x = 1 / scaling * (dafoam_solver_states - reference_state)

    error   = x - pod_modes          @ csdl.experimental.mpi.mpi_sum(pod_modes.T @ (weights * x), comm=comm)
    error_w = x - pod_modes_weighted @ csdl.experimental.mpi.mpi_sum(pod_modes_weighted.T() @ (weights * x), comm=comm)

    mpi_region.set_as_global_output(error)
    mpi_region.set_as_global_output(error_w)
    
    outputDict = dafoam_instance.getOption("function")
    for outputName in outputDict.keys():
        mpi_region.set_as_global_output(getattr(dafoam_function_rom_outputs_weighted, outputName))
        mpi_region.set_as_global_output(getattr(dafoam_function_rom_outputs, outputName))
        mpi_region.set_as_global_output(getattr(dafoam_function_outputs, outputName))
    # mpi_region.set_as_global_output(dafoam_function_outputs.drag)


# region Optimization problem selection
# optimization_case options
# 1: Maximize CL/CD wrt angle-of-attack
# 2: Minimize CD wrt angle-of-attack, wing shape (thickness/camber ffd), constrained by CL=0.5
# 3: Maximize CL/CD wrt angle-of-attack, wing shape (thickness/camber ffd)
# 4: Minimize D wrt angle-of-attack (test case)
# 5: Maximize CL/CD wrt wing shape (thickness/camber ffd)
optimization_case = 5


if optimization_case == 1:
    # Declaring and naming some variables
    lift = dafoam_function_outputs.lift
    drag = dafoam_function_outputs.drag

    # Design variables
    flight_conditions_group.angle_of_attack_deg.set_as_design_variable(lower=0, upper=10, scaler=1./10)

    # Objectives
    objective_fun = -lift/drag
    objective_fun.set_as_objective()


elif optimization_case == 2:
    # Declaring and naming some variables
    dynamic_pressure = 0.5*ambient_conditions_group.rho_kg_m3*flight_conditions_group.airspeed_m_s*flight_conditions_group.airspeed_m_s
    lift = dafoam_function_outputs.lift
    drag = dafoam_function_outputs.drag
    CL   = lift/(dynamic_pressure*A0)
    CD   = drag/(dynamic_pressure*A0)

    # Design variables
    flight_conditions_group.angle_of_attack_deg.set_as_design_variable(lower=0, upper=10, scaler=1./10)
    percent_change_in_thickness_dof.set_as_design_variable(lower=-100, upper=100, scaler=1./100)
    normalized_percent_camber_change_dof.set_as_design_variable(lower=-50, upper=50, scaler=1./50)

    # Constraints
    CL.set_as_constraint(equals=0.5)

    # Objective
    CD.set_as_objective()


elif optimization_case == 3:
    lift = dafoam_function_outputs.lift
    drag = dafoam_function_outputs.drag

    # Design variables
    flight_conditions_group.angle_of_attack_deg.set_as_design_variable(lower=0, upper=10, scaler=1./10)
    percent_change_in_thickness_dof.set_as_design_variable(lower=-10, upper=10, scaler=1./10)
    normalized_percent_camber_change_dof.set_as_design_variable(lower=-10, upper=10, scaler=1./10)

    # Objectives
    objective_fun = -lift/drag
    objective_fun.set_as_objective()


elif optimization_case == 4:
    # Declaring and naming some variables
    drag = dafoam_function_outputs.drag

    # Design variables
    flight_conditions_group.angle_of_attack_deg.set_as_design_variable(lower=0, upper=10, scaler=1./10)

    # Objectives
    objective_fun = drag
    objective_fun.set_as_objective()


elif optimization_case == 5:
    lift = dafoam_function_outputs.lift
    drag = dafoam_function_outputs.drag
    lift_rom = dafoam_function_rom_outputs.lift
    drag_rom = dafoam_function_rom_outputs.drag
    lift_rom_weighted = dafoam_function_rom_outputs_weighted.lift
    drag_rom_weighted = dafoam_function_rom_outputs_weighted.drag

    # Design variables
    percent_change_in_thickness_dof.set_as_design_variable(lower=-5, upper=5, scaler=1./5) #(lower=-10, upper=10, scaler=1./10)
    normalized_percent_camber_change_dof.set_as_design_variable(lower=-5, upper=5, scaler=1./5) #(lower=-10, upper=10, scaler=1./10)

    # Objectives
    objective_fun = - lift / drag - 0 * lift_rom / drag_rom - 0 * lift_rom_weighted / drag_rom_weighted
    objective_fun.set_as_objective()


else:
    print('Not a valid case number')


recorder.stop()



# ===============================
# region SIM
# ===============================
sim = csdl.experimental.PySimulator(recorder)


data_generator._generate_lhs_samples(snapshot_vars_and_limits, num_samples=num_samples, random_state=42)

num_samples_with_ref = num_samples + 1

diag_dict = {"max_diff":{"weighted"     : {state:np.zeros(num_samples_with_ref) for state in state_info.keys()}, "unweighted":{state:np.zeros(num_samples_with_ref) for state in state_info.keys()}}, 
             "max_err":{"weighted"      : {state:np.zeros(num_samples_with_ref) for state in state_info.keys()}, "unweighted":{state:np.zeros(num_samples_with_ref) for state in state_info.keys()}}, 
             "err_norm":{"weighted"     : {state:np.zeros(num_samples_with_ref) for state in state_info.keys()}, "unweighted":{state:np.zeros(num_samples_with_ref) for state in state_info.keys()}}, 
             "proj_err":{"weighted"     : np.zeros(num_samples_with_ref),         "unweighted":np.zeros(num_samples_with_ref)}, 
             "proj_err_rel":{"weighted" : np.zeros(num_samples_with_ref),         "unweighted":np.zeros(num_samples_with_ref)}}

from csdl_dafoam.core.rom.csdl_grassmann import Grassmann

manifold_local = Grassmann(m=comm.allreduce(dafoam_instance.getNLocalAdjointStates(), op=MPI.SUM), k=n_modes, inner_product_weights=weights, comm=comm)

for i in range(num_samples_with_ref):
    for var, info in snapshot_vars_and_limits.items():
        sim[var] = info["samples"][i]
    sim.run()
    print(f"Rank {rank} : Weights {snapshot_weights.value}")
    print(f"Rank {rank} : Weights_sum {np.sum(snapshot_weights.value)}")
    print(f"Rank {rank} : Orth_err {np.linalg.norm(comm.allreduce(pod_modes_weighted.value.T @ (weights[:, None] * pod_modes_weighted.value), op=MPI.SUM) - np.eye(n_modes), ord='fro')}")
    print(f"Rank {rank} : basis angles {(manifold_local.subspace_angles(pod_modes, pod_modes_weighted.value)) * 180 / np.pi}")

    fom_state = dafoam_solver_states.value
    rom_state = dafoam_state_estimate.value
    rom_w_state = dafoam_state_estimate_weighted.value

    for state_var, info in state_info.items():
        state_idx = info["indices"]

        fom_var_state   = fom_state[state_idx]
        rom_var_state   = rom_state[state_idx]
        rom_w_var_state = rom_w_state[state_idx]
        
        # Field diffs/errs
        diff     = np.abs(fom_var_state - rom_var_state)
        err      = diff / np.abs(fom_var_state)

        max_diff = comm.allreduce(max(diff), op=MPI.MAX)
        min_diff = comm.allreduce(min(diff), op=MPI.MIN)
        max_err  = comm.allreduce(max(err), op=MPI.MAX)
        min_err  = comm.allreduce(min(err), op=MPI.MIN)
        err_norm = np.sqrt(comm.allreduce(np.sum(diff ** 2), op=MPI.SUM)) / np.sqrt(comm.allreduce(np.sum(fom_var_state ** 2), op=MPI.SUM))

        diff_w     = np.abs(fom_var_state - rom_w_var_state)
        err_w      = diff_w / np.abs(fom_var_state)

        max_diff_w = comm.allreduce(max(diff_w), op=MPI.MAX)
        min_diff_w = comm.allreduce(min(diff_w), op=MPI.MIN)
        max_err_w  = comm.allreduce(max(err_w), op=MPI.MAX)
        min_err_w  = comm.allreduce(min(err_w), op=MPI.MIN)
        err_norm_w = np.sqrt(comm.allreduce(np.sum(diff_w ** 2), op=MPI.SUM)) / np.sqrt(comm.allreduce(np.sum(fom_var_state ** 2), op=MPI.SUM))

        diag_dict["max_diff"]["weighted"][state_var][i]    = max_diff_w
        diag_dict["max_err"]["weighted"][state_var][i]     = max_err_w
        diag_dict["err_norm"]["weighted"][state_var][i]    = err_norm_w

        diag_dict["max_diff"]["unweighted"][state_var][i]  = max_diff
        diag_dict["max_err"]["unweighted"][state_var][i]   = max_err
        diag_dict["err_norm"]["unweighted"][state_var][i]  = err_norm


    # Projection errors
    sol_norm     = np.sqrt(comm.allreduce(x.value.T @ (weights * x.value), op=MPI.SUM))

    proj_err     = np.sqrt(comm.allreduce(error.value.T @ (weights * error.value), op=MPI.SUM))
    proj_err_rel = proj_err / sol_norm

    proj_err_w     = np.sqrt(comm.allreduce(error_w.value.T @ (weights * error_w.value), op=MPI.SUM))
    proj_err_rel_w = proj_err_w / sol_norm

    diag_dict["proj_err"]["unweighted"][i]      = proj_err
    diag_dict["proj_err_rel"]["unweighted"][i]  = proj_err_rel

    diag_dict["proj_err"]["weighted"][i]        = proj_err_w
    diag_dict["proj_err_rel"]["weighted"][i]    = proj_err_rel_w

if rank == 0:   
    print(diag_dict)

    plt.figure()
    # for i, state_var in enumerate(state_info.keys()):
    #     plt.subplot(2, 5, i + 1)
    #     plt.plot(diag_dict["max_diff"]["weighted"][state_var], label='weighted', linestyle="--")
    #     plt.plot(diag_dict["max_diff"]["unweighted"][state_var], label='unweighted', linestyle="-.")
    #     plt.ylim(bottom=0)
    #     plt.title(state_var)
    #     if i == 0:
    #         plt.ylabel(r"Max difference, $|q_{FOM} - q_{ROM}|$")

    # for i, state_var in enumerate(state_info.keys()):
    #     plt.subplot(3, 5, i + 6)
    #     plt.plot(diag_dict["max_err"]["weighted"][state_var], label='weighted', linestyle="--")
    #     plt.plot(diag_dict["max_err"]["unweighted"][state_var], label='unweighted', linestyle="-.")
    #     plt.ylim(bottom=0)

    for i, state_var in enumerate(state_info.keys()):
        plt.subplot(1, 5, i + 1)
        plt.plot(diag_dict["err_norm"]["weighted"][state_var], label='weighted', linestyle="--")
        plt.plot(diag_dict["err_norm"]["unweighted"][state_var], label='unweighted', linestyle="-.")
        plt.ylim(bottom=0)
        if i == 0:
            plt.ylabel(r"Error norm, $\frac{||w_{ROM} - w_{FOM}||}{||w_{FOM}||}$")
        plt.xlabel("Snapshot")
        plt.title(state_var)
        plt.legend()
        

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(diag_dict["proj_err"]["weighted"], label='weighted', linestyle="--")
    plt.plot(diag_dict["proj_err"]["unweighted"], label='unweighted', linestyle="-.")
    plt.ylim(bottom=0)

    plt.subplot(1, 2, 2)
    plt.plot(diag_dict["proj_err_rel"]["weighted"], label='weighted', linestyle="--")
    plt.plot(diag_dict["proj_err_rel"]["unweighted"], label='unweighted', linestyle="-.")
    plt.ylim(bottom=0)

    plt.show()
    plt.figure()


        




# # Only allow visualization and modopt output files on the root rank
# visualize_on_this_rank           = True  if rank == 0 and not is_headless() else False
# turn_off_outputs_on_nonroot_rank = False if rank == 0 else True
# recording_on_root_rank           = True  if rank == 0 else False
# rank_outputs                     = ['x'] if rank == 0 else []

# # Optimization solver setup and run
# prob                = CSDLAlphaProblem(problem_name=f'{problem_name}', simulator=sim)

# # Print the FOM projection error norm after each model evaluation
# _orig_run_model = prob.check_if_warm_and_run_model
# def _run_model_with_proj_error(dvs, *args, **kwargs):
#     result = _orig_run_model(dvs, *args, **kwargs)
    
#     numerator = np.sqrt(comm.allreduce(error.value.T @ (weights * error.value), op=MPI.SUM))
#     denominator = np.sqrt(comm.allreduce(x.value.T @ (weights * x.value), op=MPI.SUM))
#     if rank == 0:
#         print(f'[proj_error] L2 norm: {numerator:.6e}', flush=True)
#         print(f'[proj_error] relative norm: {numerator / denominator:.6e}', flush=True)
#     return result

# prob.check_if_warm_and_run_model = _run_model_with_proj_error

# optimizer_choice    = 3 # Set to 1 for PySLSQP, 2 for OpenSQP, or 3 for InteriorPoint

# if optimizer_choice == 1:
#     # PySLSQP optimizer setup
#     solver_options = {'maxiter': 20,
#                     'iprint': 2,
#                     'readable_outputs': rank_outputs,
#                     'recording': recording_on_root_rank,
#                     'turn_off_outputs': turn_off_outputs_on_nonroot_rank}
#     optimizer   = PySLSQP(prob, solver_options=solver_options)
#     optimizer.solve()
#     optimizer.print_results()

# elif optimizer_choice == 2:
#     # OpenSQP optimizer setup
#     open_sqp_options = {'maxiter': 100,
#                         'readable_outputs': rank_outputs,
#                         'recording': recording_on_root_rank,
#                         'ls_max_step': 1.,
#                         'turn_off_outputs': turn_off_outputs_on_nonroot_rank,}
#     optimizer = OpenSQP(prob, **open_sqp_options)
#     optimizer.solve()
#     optimizer.print_results()

# elif optimizer_choice == 3:
#     # InteriorPoint optimizer setup
#     interior_point_options = {'maxiter': 100,
#                             'readable_outputs': rank_outputs,
#                             'recording': recording_on_root_rank,
#                             'ls_max_step': 1.,
#                             'turn_off_outputs': turn_off_outputs_on_nonroot_rank}
#     optimizer   = InteriorPoint(prob, **interior_point_options)
#     optimizer.solve()
#     optimizer.print_results()
    
# else:
#     print(f'Check optimizer choice. {optimizer_choice} is not an option.')




# # Used this to test the component
# from csdl_dafoam.utils.csdl_test_functions import test_jacvec_product, test_idempotence, test_inverse_jacobian
# import matplotlib.pyplot as plt
# np.random.seed(0)

# test_component = dafoam_rom

# inputs  = {k: vv.value for k, vv in test_component.input_dict.items()}
# v       = {k: np.random.rand(*vv.value.shape)*vv.value for k, vv in test_component.output_dict.items()}
# w       = {k: np.random.rand(*vv.value.shape)*vv.value for k, vv in test_component.input_dict.items()}

# for key in w.keys():
#     if key != "aero_vol_coords":
#         w[key] = 0 * w[key]

# print(f'Inputs: {inputs}')
# print(f'v: {v}')
# print(f'w: {w}')

# eps_test_vals = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
# err = np.zeros_like(eps_test_vals)
# i = 0
# for eps in eps_test_vals:
#     lhs, rhs, err[i] = test_jacvec_product(test_component, inputs, v, w, eps=eps)
#     i += 1 

# plt.rcParams['text.usetex'] = True
# plt.figure()
# # ax = plt.subplot(2, 1, 1)

# plt.loglog(eps_test_vals, err)
# plt.title(r'jacvec_product vs FD ($w^T J^T v = v^T J w$)')
# plt.xlabel(r'Stepsize, $\epsilon$')
# plt.ylabel(r'Error, $\frac{lhs - rhs}{rhs}$')
# plt.grid(visible=True)
# plt.show(block=False)

# if rank == 0:
#     input('Press ENTER to continue...')
# else:
#     quiet_barrier(comm)
