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
import lsdo_function_spaces as lfs
import lsdo_geo

# IDWarp and DAFoam
from csdl_dafoam.core.csdl_idwarp import DAFoamMeshWarper
from csdl_dafoam.core.csdl_dafoam import instantiateDAFoam, DAFoamFunctions, DAFoamSolver, compute_dafoam_input_variables
import csdl_dafoam.utils.standard_atmosphere_model as sam
from csdl_dafoam.utils.runscript_helper_functions import *

from csdl_dafoam.utils.training_interface import TrainingDataInterface


# BWB specific
from bwb_helper_functions import setup_geometry, read_geometry_pickle, write_geometry_pickle

# Plotting
from vedo import Points, Arrows, show

#---- DEBUGGING TOOLS ----
import faulthandler
faulthandler.enable()
os.environ["PETSC_OPTIONS"] = "-malloc_debug"
#-------------------------



# ===============================
# region USER INPUT
# ===============================
# Keyword for optimization name (optimization results folder will be saved with this name)
problem_name              = 'bwb_test'

# Geometry
geometry_directory        =  os.path.join(os.getcwd(), 'bwb_geometry/')
stp_file_name             = 'bwbv2_no_wingtip_coarse_refined_flat.stp'
geometry_pickle_file_name = 'bwb_stored_refit.pickle'

# Mesh
average_normals_at_edges  = False # if true, this will average the normals of the shared point between two surfaces (might be useful for some cases)

# MPI and timing
comm           = MPI.COMM_WORLD
timing_enabled = True  # True if we want timing printed for the CSDL operations

# Plotting
show_plots        = False
interactive_plots = False


# DAFoam
dafoam_directory    = os.path.join(os.getcwd(), 'openfoam_669k_bwb_symmetry/')
dafoamPrintInterval = 100 

# Initial/reference values for DAFoam (best to use base conditions)
# These correspond to M=0.75 @ 30k feet
U0        = 227.3805         # used for normalizing CD and CL
p0        = 30089.6
T0        = 228.714
nuTilda0  = 4.5e-5
CL_target = 0.5
aoa0      = 0
A0        = 518           # Projected area of entire BWB. Used for normalizing CD and CL
rho0      = p0 / T0 / 287 # used for normalizing CD and CL

# wall_list = ["wall"]
wall_list = ['wall_body_lower',
             'wall_body_upper', 
             'wall_wing_lower', 
             'wall_transition_lower', 
             'wall_wing_upper', 
             'wall_transition_upper', 
             'wall_wing_cap']  

# region Dafoam options
da_options = {
    "designSurfaces": wall_list,
    "solverName": "DARhoSimpleCFoam",
    "primalMinResTol": 1.0e-8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "primalVarBounds": {"pMin": 5000, "rhoMin": 0.05},
    "function": {
        "drag": {
            "type": "force",
            "source": "patchToFace",
            "patches": wall_list,
            "directionMode": "parallelToFlow",
            "patchVelocityInputName": "patch_velocity",
            "scale": 1.0, #1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
        "lift": {
            "type": "force",
            "source": "patchToFace",
            "patches": wall_list,
            "directionMode": "normalToFlow",
            "patchVelocityInputName": "patch_velocity",
            "scale": 1.0, #1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {"gmresRelTol": 1.0e-4, "pcFillLevel": 1, "jacMatReOrdering": "natural"},
    # transonic preconditioner to speed up the ff convergence
    "transonicPCOption": 2,
    "adjPCLag": 5,
    # "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm", "useNonZeroInitGuess": False},
    # # transonic preconditioner to speed up the adjoint convergence
    # "transonicPCOption": 1,
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


# region Training options
# Storage options
dataset_keyword       = 'bwb_training_testing'
storage_location      = Path(dafoam_directory)


# ===============================
# region SETUP
# ===============================
# MPI information
rank      = comm.Get_rank()
comm_size = comm.Get_size()
rank_str  = f"{rank:0{len(str(comm_size-1))}d}" # string with zero-padded rank index (for prints)


# region DAFoam instance
dafoam_instance               = instantiateDAFoam(da_options, comm, str(dafoam_directory), mesh_options)
dafoam_instance.printInterval = dafoamPrintInterval
x_surf_dafoam_initial_local   = dafoam_instance.getSurfaceCoordinates()
x_vol_dafoam_initial_local    = dafoam_instance.xv0

local_n_surf  = x_surf_dafoam_initial_local.shape[0]
local_n_vol   = x_vol_dafoam_initial_local.shape[0]

# Gathering surface mesh to rank 0 (need to do this to avoid 'no-element' ranks in the projection
# and geometry evaluation functions)
(x_surf_dafoam_initial, 
x_surf_dafoam_initial_size,
x_surf_dafoam_initial_indices) = gather_array_to_rank0(x_surf_dafoam_initial_local, comm)

# Get hash for surface mesh projection file read/write (broadcast to other ranks)
if rank == 0:
    x_surf_hash = hash_array_tol(x_surf_dafoam_initial)
else:
    x_surf_hash = None

x_surf_hash = comm.bcast(x_surf_hash, root=0)

# region File paths
geometry_pickle_file_path         = Path(geometry_directory)/geometry_pickle_file_name
stp_file_path                     = Path(geometry_directory)/stp_file_name
surface_mesh_projection_file_path = Path(dafoam_directory)/f'projected_surface_mesh_{x_surf_hash}.pickle'


# ########################################
# # Use this to visualize data 
# data_generator = TrainingDataInterface(dafoam_instance=dafoam_instance, 
#                                             storage_location=storage_location, 
#                                             dataset_keyword=dataset_keyword,
#                                             h5_file_base_name="point")

# data = data_generator.read_h5_file(Path(storage_location)/dataset_keyword/"point_0.h5", visualize_data=True)
# ########################################


# ===============================
# region CSDL RECORDER
# ===============================
# recorder 
recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()

# This section checks to see if a geometry object has already been generated.
#   If so, read the pickle file
#   If not
#       If mpi rank is 0
#           Compute geometry and write pickle file
#       If mpi rank is not 0
#           Wait for rank 1 to finish
#           Read file   

#region Geometry setup I
if geometry_pickle_file_path.is_file():
    with Timer(f'reading geometry', rank, timing_enabled):
        geometry = read_geometry_pickle(geometry_pickle_file_path)
        
else:
    if rank == 0:
        print('No geometry pickle file found.')
        with Timer('importing geometry', rank, timing_enabled):
            geometry = lsdo_geo.import_geometry(stp_file_path,
                                                parallelize=False)

        # These are hardcoded indices?
        oml_indices                 = [key for key in geometry.functions.keys()]
        wing_c_indices              = [0,1,8,9]
        wing_r_transition_indices   = [2,3]
        wing_r_indices              = [4,5,6,7]
        wing_l_transition_indices   = [10,11]
        wing_l_indices              = [12,13,14,15] 

        with Timer('declaring geometry components', rank, timing_enabled):
            left_wing_transition    = geometry.declare_component(wing_l_transition_indices)
            left_wing               = geometry.declare_component(wing_l_indices)
            right_wing_transition   = geometry.declare_component(wing_r_transition_indices)
            right_wing              = geometry.declare_component(wing_r_indices)
            center_wing             = geometry.declare_component(wing_c_indices)
            oml = geometry.declare_component(oml_indices)

        wing_parameterization   = 15
        num_v                   = left_wing.functions[wing_l_indices[0]].coefficients.shape[1]
        
        with Timer('BSplineSpace', rank, timing_enabled):
            wing_refit_bspline      = lfs.BSplineSpace(num_parametric_dimensions=2, degree=1, coefficients_shape=(wing_parameterization, num_v))

        with Timer('left wing refit', rank, timing_enabled):
            left_wing_function_set  = left_wing.refit(wing_refit_bspline, grid_resolution=(100,1000))

        with Timer('right wing refit', rank, timing_enabled):
            right_wing_function_set = right_wing.refit(wing_refit_bspline, grid_resolution=(100,1000))

        with Timer('allocating left wing functions', rank, timing_enabled):
            for i, function in left_wing_function_set.functions.items():
                geometry.functions[i]   = function
                left_wing.functions[i]  = function

        with Timer('allocating right wing functions', rank, timing_enabled):
            for i, function in right_wing_function_set.functions.items():
                geometry.functions[i]   = function
                right_wing.functions[i] = function

        with Timer('pickling geometry', rank, timing_enabled):
            write_geometry_pickle(geometry, geometry_pickle_file_path)
    
    # Wait for root rank to finish writing
    quiet_barrier(comm)
    if rank != 0:
        with Timer(f'reading geometry', rank, timing_enabled):
            geometry = read_geometry_pickle(geometry_pickle_file_path)   


# region Surface normal computation
points  = x_surf_dafoam_initial
normals_local, face_normals_local, face_centers_local = compute_vertex_normals(dafoam_instance, outward_ref=None)

normals      = gather_array_to_rank0(-normals_local, comm)[0]
face_normals = gather_array_to_rank0(-face_normals_local, comm)[0]
face_centers = gather_array_to_rank0(face_centers_local, comm)[0]

# Edge normal handling
if rank == 0:
    if average_normals_at_edges:
        normals = average_normals_at_duplicate_points(x_surf_dafoam_initial, normals)

# Plot the initial points and normals over the geometry for reference
if rank == 0 and show_plots and not is_headless():
    geo_plot  = geometry.plot(show=False)
    scatter   = Points(points, r=2, c='green')
    arrows    = Arrows(points, points + 0.2*normals, c='red', s=0.5)

    # Plot duplicate points
    uniq, idx, counts  = np.unique(x_surf_dafoam_initial, axis=0, return_index=True, return_counts=True)
    duplicate_points   = uniq[counts > 1]
    scatter_duplicates = Points(duplicate_points, r=2, c='yellow')

    scatter_face   = Points(face_centers, r=2, c='blue')
    arrows_face    = Arrows(face_centers, face_centers + 0.2*face_normals, c='orange', s=0.5)
    show(geo_plot, scatter, arrows, scatter_duplicates, scatter_face, arrows_face, axes=1, interactive=interactive_plots)

quiet_barrier(comm)


# region Surface mesh projection
# Now do we do the same check for the surface mesh projection
if surface_mesh_projection_file_path.is_file():
    if rank == 0:
        print('Found surface mesh projection pickle!')
    projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)

else:
    if rank == 0:
        print('No projected surface mesh file found.')
        try:
            # # ORIGINAL CODE
            with Timer('projecting on surface mesh', rank, timing_enabled):

                projected_surf_mesh_dafoam = geometry.project(
                    x_surf_dafoam_initial, 
                    grid_search_density_parameter = 1,      # 1 
                    projection_tolerance          = 1e-3,   # 1.e-3m 
                    grid_search_density_cutoff    = 12,    # 150
                    force_reprojection            = False,
                    plot                          = show_plots and not is_headless(),
                    interactive                   = interactive_plots,                     
                    direction                     = normals,
                    num_workers                   = comm_size
                )

            print('Writing surface mesh projection pickle...')
            write_simple_pickle(projected_surf_mesh_dafoam, surface_mesh_projection_file_path)
            print('Done!')

        # Added this exception because I was getting an ungraceful MPI termination
        except Exception as e:
            import traceback
            print(f"[Rank 0 ERROR] Projection/pickle step failed:\n{traceback.format_exc()}", flush=True)
            comm.Abort(1) # Abort MPI processes instead of letting them hang

    quiet_barrier(comm)

    if rank != 0:
        projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)

print(f'Rank {rank_str} done reading projected surface mesh!')
quiet_barrier(comm)


# region Design variables
# ============================ Design variables ===========================
root_twist  = csdl.Variable(shape=(1,), value=np.array([0]))
tip_twist   = csdl.Variable(shape=(1,), value=np.array([0.]))
mid_twist   = csdl.Variable(shape=(2,), value=np.array([0., 0.]))
wing_twists = csdl.concatenate((root_twist, mid_twist, tip_twist))
wing_twists.flatten()

percent_change_in_thickness_dof_wing        = csdl.Variable(shape=(8,4), value=0.)
percent_change_in_thickness_dof_body        = csdl.Variable(shape=(8,4), value=0.)
percent_change_in_thickness_dof             = csdl.concatenate(
                                                (percent_change_in_thickness_dof_wing,
                                                 percent_change_in_thickness_dof_body), axis=1)

normalized_percent_camber_change_dof_wing   = csdl.Variable(shape=(6,4), value=0.)
normalized_percent_camber_change_dof_body   = csdl.Variable(shape=(6,4), value=0.)
normalized_percent_camber_change_dof        = csdl.concatenate(
                                                (normalized_percent_camber_change_dof_wing,
                                                 normalized_percent_camber_change_dof_body), axis=1)

centerbody_chord_stretches                  = csdl.Variable(shape=(3,), value=0.)
wing_chord_stretching_b_spline_coefficients = csdl.Variable(shape=(2,), value=np.array([0., 0.]))
centerbody_span                             = csdl.Variable(value=10.)
transition_span                             = csdl.Variable(value=4.891)
wing_span                                   = csdl.Variable(value=25.852 - 9.891)
wing_sweep_translation                      = csdl.Variable(value=0.)
centerbody_dihedral_translations            = csdl.Variable(shape=(3,), value=np.array([0., 0., 0.]))
wing_dihedral_translation_b_spline_coefficients = csdl.Variable(shape=(2,), value=np.array([0., 0.]))
centerbody_twists                           = csdl.Variable(shape=(3,), value=np.array([0., 0., 0.]))
# wing_twists                                 = csdl.Variable(shape=(4,), value=np.array([0., 0., 0., 0.]))
# percent_change_in_thickness_dof             = csdl.Variable(shape=(8,8), value=0.)
# normalized_percent_camber_change_dof        = csdl.Variable(shape=(6,8), value=0.)


geometry_values_dict = {
    'centerbody_chord_stretches': centerbody_chord_stretches,
    'wing_chord_stretching_b_spline_coefficients': wing_chord_stretching_b_spline_coefficients,
    'centerbody_span': centerbody_span,
    'transition_span': transition_span,
    'wing_span': wing_span,
    'wing_sweep_translation': wing_sweep_translation,
    'centerbody_dihedral_translations': centerbody_dihedral_translations,
    'wing_dihedral_translation_b_spline_coefficients': wing_dihedral_translation_b_spline_coefficients,
    'centerbody_twists': centerbody_twists,
    'wing_twists': wing_twists,
    'percent_change_in_thickness_dof': percent_change_in_thickness_dof,
    'normalized_percent_camber_change_dof': normalized_percent_camber_change_dof,
}


# region Geometry setup II
with Timer(f'setting up geometry', rank, timing_enabled):
    # Had to "serialize" this because I was getting race conditions in cache I/O
    for r in range(comm_size):
        quiet_barrier(comm)
        if rank == r:
            geometry = setup_geometry(geometry, geometry_values_dict)
        quiet_barrier(comm)

with Timer(f'evaluating geometry component', rank, timing_enabled):
    x_surf_dafoam_full = geometry.evaluate(projected_surf_mesh_dafoam, plot=False)

# region Surface mesh distribution
i0, i1          = x_surf_dafoam_initial_indices[rank]

# Flight condition variables
flight_conditions_group                     = csdl.VariableGroup()
flight_conditions_group.mach_number         = csdl.Variable(value=0.75, name="mach_number")
flight_conditions_group.angle_of_attack_deg = csdl.Variable(value=aoa0, name="angle_of_attack")
flight_conditions_group.altitude_m          = csdl.Variable(value=9144., name="altitude (m)")
flight_conditions_group.airspeed_m_s        = csdl.Variable(value=U0, name="airspeed (m/s)")

# Atmospheric condition variables
ambient_conditions_group = sam.compute_ambient_conditions_group(flight_conditions_group.altitude_m)

# reynolds_number    = ambient_conditions_group.rho_kg_m3*flight_conditions_group.airspeed_m_s*10/ambient_conditions_group.mu_kg_m_s
# if rank == 0:
#     print(f'Reynolds number: {reynolds_number.value}')
#     print(f'Density (kg/m^3): {ambient_conditions_group.rho_kg_m3.value}')
#     print(f'Speed (m/s): {flight_conditions_group.airspeed_m_s.value}')
#     print(f'Dynamic viscosity (kg/m/s): {ambient_conditions_group.mu_kg_m_s.value}')
#     input('Press ENTER to continue...')    
# else:
#     None

# comm.Barrier()

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

    # DAFoamSolver Implicit component setup and evaluation
    dafoam_solver           = DAFoamSolver(dafoam_instance)
    dafoam_solver_states    = dafoam_solver.evaluate(dafoam_input_variables_group)

    # DAFoamFunctions Explicit component setup and evaluation
    dafoam_functions = DAFoamFunctions(dafoam_instance)
    dafoam_function_outputs = dafoam_functions.evaluate(dafoam_solver_states, 
                                                        dafoam_input_variables_group)

    mpi_region.set_as_global_output(dafoam_function_outputs.lift)
    mpi_region.set_as_global_output(dafoam_function_outputs.drag)

recorder.stop()



# ===============================
# region SIM SETUP
# ===============================
sim = csdl.experimental.PySimulator(recorder)



# ===============================
# region TRAINING
# ===============================



# Sampling options
# grassmann_variables indicates the variables which correspond to points on the Grassmann manifold
# snapshot_variables indicates the variables which correspond to "snapshots" or realizations
num_grassmann_samples     = 2
num_snapshot_samples      = 20
random_state_seed         = 0

# Specify variables and their limits for sampling
# Expect the following structure:
# grassmann_vars_and_limits = {
#   csdl_variable_1: {
#       'name': name_string,
#       'range': [min_val, max_val],
#   }
#   csdl_variable_2: {...}
#
#}
# snapshot_vars_and_limits = {
#   csdl_variable_1: {
#       'name': name_string,
#       'range': [min_val, max_val],
#       'ref_val': reference_value,
#   }
#   csdl_variable_2: {...}
#
#}
# Make sure that the csdl_variables are the actual csdl_variables
# The name is for labeling during the file save
# The range is the limits for sampling
# The ref_value is the reference value for the particular Grassmann manifold point.

grassmann_vars_and_limits = {
    # flight_conditions_group.mach_number: {
    #     'name': 'mach_number',   
    #     'range': [0.65, 0.75],
    # }, 
    flight_conditions_group.angle_of_attack_deg: {
        'name': 'angle_of_attack_deg',     
        'range': [0., 10],
    },
    # flight_conditions_group.altitude_m: {
    #     'name': 'altitude_m', 
    #     'range': [7000., 13000],
    #     }
}

snapshot_vars_and_limits = {
    # percent_change_in_thickness_dof_wing: {
    #     'name': '%_thickness_change_wing',
    #     'range': [-10, 10],
    #     'ref_value': 0, 
    # },
    normalized_percent_camber_change_dof_wing: {
        'name': '%_camber_change_wing',
        'range': [-10, 10],
        'ref_value': 0, 
    },
    root_twist: {
        'name': 'root_twist',
        'range': [-10*np.pi/180, 10*np.pi/180],
        'ref_value': 0, 
    },
    tip_twist: {
        'name': 'tip_twist',
        'range': [-10*np.pi/180, 10*np.pi/180],
        'ref_value': 0, 
    },
    mid_twist: {
        'name': 'mid_twist',
        'range': [-10*np.pi/180, 10*np.pi/180],
        'ref_value': 0, 
    },
}

data_generator = TrainingDataInterface(dafoam_instance=dafoam_instance, 
                                            storage_location=storage_location, 
                                            dataset_keyword=dataset_keyword,
                                            primary_variables=grassmann_vars_and_limits, 
                                            secondary_variables=snapshot_vars_and_limits, 
                                            csdl_simulator=sim,
                                            num_primary_samples=num_grassmann_samples,
                                            num_secondary_samples=num_snapshot_samples,
                                            random_state_seed=random_state_seed,
                                            h5_file_base_name="point")

data_generator.sample_variables()
data_generator.run_sweep()
