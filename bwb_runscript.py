# ===============================
# region PACKAGES
# ===============================
import numpy as np
import sys
import os
import pickle
from pathlib import Path

# MPI
from mpi4py import MPI

# CSDL packages
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import lsdo_geo

# Optimization
from modopt import CSDLAlphaProblem
from modopt import PySLSQP, OpenSQP

# IDWarp and DAFoam
from csdl_idwarp import DAFoamMeshWarper
from csdl_dafoam import instantiateDAFoam, DAFoamFunctions, DAFoamSolver, compute_dafoam_input_variables
import standard_atmosphere_model as sam

# BWB specific
from bwb_helper_functions import setup_geometry, read_geometry_pickle, write_geometry_pickle, gather_array_to_rank0, read_simple_pickle, write_simple_pickle

from helper_functions import Timer, hash_array_tol, quiet_barrier, compute_vertex_normals, average_normals_at_duplicate_points

# Plotting
from vedo import Points, Arrows, Mesh, show
import matplotlib.pyplot as plt
from check_headless import is_headless

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
average_normals_at_edges  = True # if true, this will average the normals of the shared point between two surfaces (might be useful for some cases)

# MPI and timing
comm           = MPI.COMM_WORLD
timing_enabled = True  # True if we want timing printed for the CSDL operations

# DAFoam
dafoam_directory    = os.path.join(os.getcwd(), 'openfoam_739k_bwb_symmetry/')
dafoamPrintInterval = 1 # This doesn't actually seem to affect anything...

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

# wall_list = ['wall_wing_cap',
#             'wall_wing',
#             'wall_transition',
#             'wall_body']
# #
#    'wall_wing_cap_edge',
#    'wall_trailing_surf_wing',
#    'wall_trailing_surf_transition',
#    'wall_trailing_surf_body', 
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
    }
}

# region Mesh options
mesh_options = {
    "gridFile": dafoam_directory,
    "fileType": "OpenFOAM",
    "symmetryPlanes": [],
}



# ===============================
# region SETUP
# ===============================
# MPI information
rank      = comm.Get_rank()
comm_size = comm.Get_size()
rank_str  = f"{rank:0{len(str(comm_size-1))}d}" # string with zero-padded rank index (for prints)


# region DAFoam instance
dafoam_instance               = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)
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


# if rank == 0:
#     with Timer('exporting geometry stp', rank, timing_enabled):
#             export_geo2step('refit_geometry.stp', geometry)


points  = x_surf_dafoam_initial
normals_local, face_normals_local, face_centers_local = compute_vertex_normals(dafoam_instance, outward_ref=None)

# print(f'Rank {rank} normals: {normals_local}')
# print(f'Rank {rank} face_normals: {face_normals_local}')
# print(f'Rank {rank} face_centers: {face_centers_local}')

normals      = gather_array_to_rank0(-normals_local, comm)[0]
face_normals = gather_array_to_rank0(-face_normals_local, comm)[0]
face_centers = gather_array_to_rank0(face_centers_local, comm)[0]


if rank == 0 and not is_headless():
    if average_normals_at_edges:
        normals = average_normals_at_duplicate_points(x_surf_dafoam_initial, normals)

    # surface   = Mesh('/media/edward/DATA/Edward/AFRL_project/csdl_project/folder_geo/geometry/bwbv2_no_wingtip_coarse_refined_flat2.stl')
    # distances = Points(points).distance_to(surface,invert=True, signed=True)
    geo_plot  = geometry.plot(show=False)
    scatter   = Points(points, r=2, c='green')
    arrows    = Arrows(points, points + 0.2*normals, c='red', s=0.5)

    # Plot duplicate points
    uniq, idx, counts  = np.unique(x_surf_dafoam_initial, axis=0, return_index=True, return_counts=True)
    duplicate_points   = uniq[counts > 1]
    scatter_duplicates = Points(duplicate_points, r=2, c='yellow')



    scatter_face   = Points(face_centers, r=2, c='blue')
    arrows_face    = Arrows(face_centers, face_centers + 0.2*face_normals, c='orange', s=0.5)
    show(geo_plot, scatter, arrows, scatter_duplicates, scatter_face, arrows_face, axes=1)
    # # Create figure
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot points
    # ax.scatter(points[:,0], points[:,1], points[:,2], color='red', s=5, alpha=0.6, label='Points')

    # # Plot normals
    # scale = 0.1  # adjust depending on mesh size
    # ax.quiver(points[:,0], points[:,1], points[:,2],
    #         normals[:,0], normals[:,1], normals[:,2],
    #         length=scale, color='blue', normalize=True, label='Normals')

    # # Set equal aspect ratio
    # all_pts = points
    # x_limits = (all_pts[:,0].min(), all_pts[:,0].max())
    # y_limits = (all_pts[:,1].min(), all_pts[:,1].max())
    # z_limits = (all_pts[:,2].min(), all_pts[:,2].max())
    # max_range = max(x_limits[1]-x_limits[0], y_limits[1]-y_limits[0], z_limits[1]-z_limits[0]) / 2.0
    # mid_x = np.mean(x_limits)
    # mid_y = np.mean(y_limits)
    # mid_z = np.mean(z_limits)
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Surface Points and Normals')
    # plt.show()

quiet_barrier(comm)
#=============================


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
                # n_surf = x_surf_dafoam_initial.shape[0]
                # projected_surf_mesh_dafoam = []
                # for i in range(n_surf):
                #     print(f'{i}/{n_surf}: p {x_surf_dafoam_initial[i, :]}, n {normals[i, :]}')
                #     projected_i = geometry.project(
                #         x_surf_dafoam_initial[i, :], 
                #         grid_search_density_parameter = 1 ,      # 1 
                #         projection_tolerance          = 1e-3,   # 1.e-3m 
                #         grid_search_density_cutoff    = 100,    # 20
                #         force_reprojection            = False,
                #         plot                          = False,
                #         direction                     = normals[i, :]
                #     )
                #     print(f'projected {projected_i}')
                #     projected_surf_mesh_dafoam.extend(projected_i)

                problem_point_index = np.argmin(np.linalg.norm(x_surf_dafoam_initial - [23.68170823699935, 9.363806102035095, 0.9857894201215491], axis=1))
                # x_surf_dafoam_initial[problem_point_index] -= [1e-6, 0, 0]

                projected_surf_mesh_dafoam = geometry.project(
                    x_surf_dafoam_initial[problem_point_index], 
                    grid_search_density_parameter = 1,      # 1 
                    projection_tolerance          = 1e-3,   # 1.e-3m 
                    grid_search_density_cutoff    = 100,    # 20
                    force_reprojection            = False,
                    plot                          = True,
                    direction                     = normals[problem_point_index],
                    num_workers                   = 48
                )

                projected_surf_mesh_dafoam = geometry.project(
                    x_surf_dafoam_initial, 
                    grid_search_density_parameter = 1,      # 1 
                    projection_tolerance          = 1e-3,   # 1.e-3m 
                    grid_search_density_cutoff    = 150,    # 20
                    force_reprojection            = False,
                    plot                          = True,
                    direction                     = normals,
                    num_workers                   = 48
                )

            # # Debugging/timing
            # import cProfile
            # import pstats
            # with cProfile.Profile() as pr:
            #     projected_surf_mesh_dafoam = geometry.project(
            #         x_surf_dafoam_initial, 
            #         grid_search_density_parameter = 1,      # 1    
            #         projection_tolerance          = 1e-2,   # 1.e-3m
            #         grid_search_density_cutoff    = 1,     # 20
            #         force_reprojection            = False,
            #         plot                          = True,
            #         # direction                     = [0, 0, 1],
            #         num_workers                   = 4
            #     )
            # # Summarize top time-consuming functions
            # stats = pstats.Stats(pr)
            # stats.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats(30)

            # print(projected_surf_mesh_dafoam)
            # input('Press ENTER to continue')

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


# region Optimization problem selection
# optimization_case options
# 1: Maximize CL/CD wrt angle-of-attack
# 2: Minimize CD wrt angle-of-attack, root/tip twist, constrained by CL=0.5
# 3: Minimize CD wrt angle-of-attack, wing shape (thickness/camber ffd), constrained by CL=0.5
# 4: Minimize CD wrt angle-of-attack, wing shape (thickness/camber ffd) and wing twists, constrained by CL=0.5
# 5: Maximize CL/CD wrt angle-of-attack and wing shape
# 6: Maximize CL/CD wrt angle-of-attack, wing shape (thickness/camber ffd) and wing twists
# 7: Maximize CL/CD wrt angle-of-attack, wing shape (camber ffd) and wing twists
optimization_case = 7


if optimization_case == 1:
    # Declaring and naming some variables
    lift = dafoam_function_outputs.lift
    drag = dafoam_function_outputs.drag

    # Design variables
    flight_conditions_group.angle_of_attack_deg.set_as_design_variable(lower=0, upper=10)

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
    root_twist.name = 'root_twist'
    tip_twist.name  = 'tip_twist'
    twist_lim_deg   = 5
    twist_lim_rad   = twist_lim_deg*np.pi/180

    # Design variables
    flight_conditions_group.angle_of_attack_deg.set_as_design_variable(lower=0, upper=10)
    root_twist.set_as_design_variable(lower=-twist_lim_rad, upper=twist_lim_rad, scaler=1/twist_lim_rad)
    tip_twist.set_as_design_variable(lower=-twist_lim_rad, upper=twist_lim_rad, scaler=1/twist_lim_rad)

    # Constraints
    CL.set_as_constraint(equals=0.5)

    # Objective
    CD.set_as_objective()


elif optimization_case == 3:
    # Declaring and naming some variables
    dynamic_pressure = 0.5*ambient_conditions_group.rho_kg_m3*flight_conditions_group.airspeed_m_s*flight_conditions_group.airspeed_m_s
    lift = dafoam_function_outputs.lift
    drag = dafoam_function_outputs.drag
    CL   = lift/(dynamic_pressure*A0)
    CD   = drag/(dynamic_pressure*A0)

    # Design variables
    flight_conditions_group.angle_of_attack_deg.set_as_design_variable(lower=0., upper=10., scaler=1./12)
    percent_change_in_thickness_dof_wing.set_as_design_variable(lower=-10, upper=30., adder=10., scaler=1./40.)
    normalized_percent_camber_change_dof_wing.set_as_design_variable(lower=-20., upper=20., scaler=1./20.)

    # Constraints
    CL.set_as_constraint(equals=0.5)

    # Objective
    CD.set_as_objective()


elif optimization_case == 4:
    # Declaring and naming some variables
    dynamic_pressure = 0.5*ambient_conditions_group.rho_kg_m3*flight_conditions_group.airspeed_m_s*flight_conditions_group.airspeed_m_s
    lift = dafoam_function_outputs.lift
    drag = dafoam_function_outputs.drag
    CL   = lift/(dynamic_pressure*A0)
    CD   = drag/(dynamic_pressure*A0)
    twist_lim_deg   = 5
    twist_lim_rad   = twist_lim_deg*np.pi/180

    # Design variables
    flight_conditions_group.angle_of_attack_deg.set_as_design_variable(lower=-2., upper=10., adder=2., scaler=1./12.)
    percent_change_in_thickness_dof_wing.set_as_design_variable(lower=-10, upper=30., adder=10., scaler=1./40.)
    normalized_percent_camber_change_dof_wing.set_as_design_variable(lower=-20., upper=20., scaler=1./20.)
    wing_twists.set_as_design_variable(lower=-twist_lim_rad, upper=twist_lim_rad, scaler=1/twist_lim_rad)

    # Constraints
    CL.set_as_constraint(equals=0.5)

    # Objective
    CD.set_as_objective()


elif optimization_case == 5:
    lift = dafoam_function_outputs.lift
    drag = dafoam_function_outputs.drag

    # Design variables
    flight_conditions_group.angle_of_attack_deg.set_as_design_variable(lower=0, upper=10, scaler=1./12.)
    percent_change_in_thickness_dof_wing.set_as_design_variable(lower=-10, upper=30., adder=10., scaler=1./40.)
    normalized_percent_camber_change_dof_wing.set_as_design_variable(lower=-20., upper=20., scaler=1./20.)

    # Objectives
    objective_fun = -lift/drag
    objective_fun.set_as_objective()


elif optimization_case == 6:
    # Declaring and naming some variables
    lift = dafoam_function_outputs.lift
    drag = dafoam_function_outputs.drag
    twist_lim_deg   = 5
    twist_lim_rad   = twist_lim_deg*np.pi/180

    # Design variables
    flight_conditions_group.angle_of_attack_deg.set_as_design_variable(lower=-2., upper=10., adder=2., scaler=1./12.)
    percent_change_in_thickness_dof_wing.set_as_design_variable(lower=-10, upper=30., adder=10., scaler=1./40.)
    normalized_percent_camber_change_dof_wing.set_as_design_variable(lower=-20., upper=20., scaler=1./20.)
    wing_twists.set_as_design_variable(lower=-twist_lim_rad, upper=twist_lim_rad, scaler=1/twist_lim_rad)

    # Objective
    objective_fun = -lift/drag
    objective_fun.set_as_objective()


elif optimization_case == 7:
    # Declaring and naming some variables
    lift = dafoam_function_outputs.lift
    drag = dafoam_function_outputs.drag
    twist_lim_deg   = 10
    twist_lim_rad   = twist_lim_deg*np.pi/180

    # Design variables
    flight_conditions_group.angle_of_attack_deg.set_as_design_variable(lower=-2., upper=10., adder=2., scaler=1./12.)
    normalized_percent_camber_change_dof_wing.set_as_design_variable(lower=-30., upper=30., scaler=1./30.)
    wing_twists.set_as_design_variable(lower=-twist_lim_rad, upper=twist_lim_rad, scaler=1/twist_lim_rad)

    # Objective
    objective_fun = -lift/drag
    objective_fun.set_as_objective()

else:
    print('Not a valid case number')


recorder.stop()



# ===============================
# region SIM
# ===============================
sim = csdl.experimental.PySimulator(recorder)

# Can set design variables here and run sim to test
# sim[root_twist]  = 3*3.14159/180
# sim.run()

# Uncomment to run and check derivatives via finite difference
# sim.check_totals()
# derivs = sim.compute_totals([CD],[root_twist, tip_twist, flight_conditions_group.angle_of_attack])

# Only allow visualization on the root rank
if rank == 0 and not is_headless():
    visualize_on_this_rank = True
else:
    visualize_on_this_rank = False

# Optimization solver setup and run
prob        = CSDLAlphaProblem(problem_name=f'{problem_name}_rank{rank_str}', simulator=sim)

# # # PySLSQP optimizer setup
# # solver_options = {'maxiter': 20,
# #                   'iprint': 2,
# #                   'visualize': visualize_on_this_rank,
# #                   'summary_filename': f'rank{rank_str}_slsqp_summary.out',
# #                   'save_figname':     f'rank{rank_str}_slsqp_plot.pdf',
# #                   'save_filename':    f'rank{rank_str}_slsqp_recorder.hdf5'}
# # optimizer   = PySLSQP(prob, solver_options=solver_options)
# # optimizer.solve()
# # optimizer.print_results()


# OpenSQP optimizer setup
open_sqp_options = {'maxiter': 40,
                    'readable_outputs': ['x'],
                    'recording': True,
                    'ls_max_step': 1.0}
optimizer   = OpenSQP(prob, **open_sqp_options)
optimizer.solve()
optimizer.print_results()





# # # Extra items to use, if necessary
# # optimizer.check_first_derivatives(prob.x0)
