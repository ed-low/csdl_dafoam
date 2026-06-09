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
from csdl_dafoam.core.csdl_dafoam import instantiateDAFoam, DAFoamFunctions, DAFoamSolver, compute_dafoam_input_variables, DAFoamForces
import csdl_dafoam.utils.standard_atmosphere_model as sam
from csdl_dafoam.utils.runscript_helper_functions import *
from csdl_dafoam.scripts.blended_wing_body.bwb_helper_functions import *

# # DAFoamROM stuff
# from csdl_dafoam.utils.training_interface import TrainingDataInterface
# from csdl_dafoam.core.rom.csdl_rom import CSDLROMWrapper
# from csdl_dafoam.core.rom.rom_models import DAFoamLSPGModel, DAFoamGalerkinModel
# from csdl_dafoam.core.rom.rom_solver import NewtonSolver, BroydenNewtonSolver

# Plotting
from vedo import Points, Arrows, show



# Importing all packages 
import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs

from modopt import CSDLAlphaProblem
from modopt import PySLSQP, OpenSQP, InteriorPoint

import lsdo_geo
import sys
import os

import aframe as af
from aeroelastic_coupling_utils import NodalMap

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from csdl_dafoam.utils.theairforce.geometry_functions import setup_geometry, compute_volume, project_centerbody_volume_points
from csdl_dafoam.utils.theairforce.geometry_functions import project_transition_volume_points
import time
import pickle
from datetime import datetime

from csdl_dafoam.utils.theairforce.additional_solvers import compute_static_margin, estimate_CDw, estimate_Cf, estimate_fuel_volume
from csdl_dafoam.utils.theairforce.additional_solvers import estimate_fuel_burn_w_reserve, atmos_model, takeoff, landing


#---- DEBUGGING TOOLS ----
import faulthandler
faulthandler.enable()
os.environ["PETSC_OPTIONS"] = "-malloc_debug"
#-------------------------


# Write this runscript to file before anything (will initialize MPI comm here)
comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    print_runscript_info()




# ===============================
# region USER INPUT
# ===============================
# Keyword for optimization name (optimization results folder will be saved with this name)
problem_name              = '191k_test_run'#'89k_test_run'#'669k_test_run'#

# Geometry
geometry_directory        =  os.path.join(os.getcwd(), 'bwb_geometry/')
stp_file_name             = 'bwbv2_no_wingtip_coarse_refined_flat.stp'
geometry_pickle_file_name = 'bwb_stored_refit.pickle'

# Mesh
average_normals_at_edges  = False # if true, this will average the normals of the shared point between two surfaces (might be useful for some cases)

# Timing
timing_enabled = True  # True if we want timing printed for the CSDL operations

# Plotting
show_plots        = False
interactive_plots = False


# DAFoam
dafoam_directory    = os.path.join(os.getcwd(), f'results/{problem_name}')
dafoamPrintInterval = 100 

# Initial/reference values for DAFoam (best to use base conditions)
# These correspond to M=0.6 @ 30k feet
U0        = 181.9044         # used for normalizing CD and CL
p0        = 30089.6
T0        = 228.714
nuTilda0  = 4.5e-5
CL_target = 0.5
aoa0      = 0
A0        = 518           # Projected area of entire BWB. Used for normalizing CD and CL
rho0      = p0 / T0 / 287 # used for normalizing CD and CL

wall_list = ["wall"]
# wall_list = ['wall_body_lower',
#              'wall_body_upper', 
#              'wall_wing_lower', 
#              'wall_transition_lower', 
#              'wall_wing_upper', 
#              'wall_transition_upper', 
#              'wall_wing_cap']  

# region Dafoam options
da_options = {
    "designSurfaces": wall_list,
    "solverName": "DARhoSimpleCFoam",
    "primalMinResTol": 4.0e-7,
    "primalMinResTolDiff":1.0e0,
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
        "moment_y": {
            "type": "moment",
            "source": "patchToFace",
            "patches": wall_list,
            "axis": [0.0, 1.0, 0.0],
            "center": [0.25, 0.0, 0.05], # TODO: NEED TO UPDATE THIS?
            "scale": 1.0 #/ (0.5 * UmagIn * UmagIn * ARef * LRef),
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
    "outputInfo": {
        "f_aero": {
            "type": "forceCouplingOutput",
            "patches": wall_list,
            "components": ["forceCoupling"],
            "pRef": p0,
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
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
}


# region Training options
# Storage options
dataset_keyword       = 'training_data_test'
storage_location      = Path(dafoam_directory)


# region Leftover options
lfs.num_workers=1
save_meshes = True
shutdown_inline = True

# ===============================
# END USER INPUT
# ===============================

# ###
# import signal, traceback, sys

# def dump_trace(sig, frame):
#     print(f"[Rank {comm.Get_rank()}] STACK TRACE:", flush=True)
#     traceback.print_stack(frame)
#     sys.stdout.flush()

# signal.signal(signal.SIGUSR1, dump_trace)
# ###



# ===============================
# region SETUP
# ===============================
# MPI information
rank      = comm.Get_rank()
comm_size = comm.Get_size()
rank_str  = f"{rank:0{len(str(comm_size-1))}d}" # string with zero-padded rank index (for prints)

# region DAFoam instance
dafoam_instance               = instantiateDAFoam(da_options, comm, str(dafoam_directory), mesh_options)
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




recorder = csdl.Recorder(inline=False, debug=True)
recorder.start()



# region ============================ geometry import and setup ============================
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
# endregion

# region ============================ aero mesh setup ============================


if surface_mesh_projection_file_path.is_file():
    if rank == 0:
        print('Found surface mesh projection pickle!')
    projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)

else:
    if rank == 0:
        print('No projected surface mesh file found.')

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






# region ============================ structures mesh setup ============================
num_nonwing_nodes = 4
num_wing_nodes = 7

# LE and TE points
LE_nonwing_pts = np.array([
    [0., 0., 0.],
    [3.899, 2.5, 0.],
    [9.813, 5.0, 0.],
    [15.541, 7.934, 0.659],
])

TE_nonwing_pts = np.array([
    [30., 0., 0.],
    [30., 2.5, 0.],
    [30., 5.0, 0.],
    [24.88, 7.934, 0.691],
])

wing_root_LE_TE = [
    np.array([17.815, 9.891, 1.040]), # LE
    np.array([23.815, 9.891, 1.040]), # TE
]

wing_tip_LE_TE = [
    np.array([29.019, 25.852, 2.123]), # LE
    np.array([32.016, 25.852, 2.254]), # TE
]

LE_wing_pts = np.linspace(wing_root_LE_TE[0], wing_tip_LE_TE[0], num_wing_nodes)
TE_wing_pts = np.linspace(wing_root_LE_TE[1], wing_tip_LE_TE[1], num_wing_nodes)

LE_points = np.concatenate((LE_nonwing_pts, LE_wing_pts)) # we don't want the center line
TE_points = np.concatenate((TE_nonwing_pts, TE_wing_pts)) # we don't want the center line

LE_points_projected = geometry.project(LE_points, plot=False)
TE_points_projected = geometry.project(TE_points, plot=False)
                                                                               
# airfoil top and bottom points
top_nonwing_pts = np.array([
    [8.973, 0, 2.251],
    [11.706, 2.5, 1.958],
    [15.851, 5., 1.514],
    [18.352, 7.934, 1.415],
])

bottom_nonwing_pts = np.array([
    [8.973, 0, -2.251],
    [11.706, 2.5, -1.958],
    [15.851, 5., -1.514],
    [18.398, 7.934, 0.078],
])

wing_root_top_bottom = [
    np.array([20.001, 9.891, 1.537]), # top
    np.array([20.040, 9.891, 0.699]), # bottom
]

tip_top_bottom = [
    np.array([30.102, 25.852, 2.384]), # top
    np.array([30.137, 25.852, 2.026]), # bottom
]

top_wing_pts = np.linspace(wing_root_top_bottom[0], tip_top_bottom[0], num_wing_nodes)
bottom_wing_pts = np.linspace(wing_root_top_bottom[1], tip_top_bottom[1], num_wing_nodes)

top_pts = np.concatenate((top_nonwing_pts, top_wing_pts))[1:,:] # we don't want the center line
bot_pts = np.concatenate((bottom_nonwing_pts, bottom_wing_pts))[1:,:] # we don't want the center line

top_pts_projected = geometry.project(top_pts, plot=False)
bot_pts_projected = geometry.project(bot_pts, plot=False)
# endregion

# region ============================ mass properties setup and point projections ============================
lbf_to_N = 4.44822
# mass properties setup
empty_weight_lbf = 126636 #lbf
empty_weight_N = empty_weight_lbf*lbf_to_N

# a guess on initial wing mass (acts as a deficit )
wing_mass_0 = 5000 # kg
wing_weight_0 = wing_mass_0*9.81

empty_weight_N_nowing = empty_weight_N - wing_weight_0

# mass point projections
engine_sec_LE_0 = np.array([7.347, -4., 0.])
engine_sec_TE_0 = np.array([30., -4., 0.])
engine_loc_0 = np.array([24.702, -4., 0.824]) # 80% of center span, around 80% of chord
ecf_0 = (engine_loc_0[0]-engine_sec_LE_0[0])/(engine_sec_TE_0[0]-engine_sec_LE_0[0]) # engine chord fraction
# ecf_0 here is around 76.6% of the chord (just a starting point)
# ecf_0 = 0.8 # engine chord fraction


engine_loc_parametric = geometry.project(engine_loc_0, plot=False)
engine_sec_LE_parametric = geometry.project(engine_sec_LE_0, plot=False)
engine_sec_TE_parametric = geometry.project(engine_sec_TE_0, plot=False)


transition_volume_LE_points = np.array([
    [9.813, 5.0, 0.],
    [12.311, 6.223, 0.158],
    [14.707, 7.445, 0.510],
    [16.604, 8.668, 0.853],
    [17.815, 9.891, 1.040],
])

transition_volume_TE_points = np.array([
    [30., 5.0, 0.],
    [28.627, 6.223, 0.167],
    [25.813, 7.445, 0.545],
    [23.963, 8.668, 0.870],
    [23.815, 9.891, 1.040],
])

transition_volume_LE_points_proj = geometry.project(transition_volume_LE_points)
transition_volume_TE_points_proj = geometry.project(transition_volume_TE_points)

transition_volume_projection_points = project_transition_volume_points(
    geometry, 
    transition_volume_LE_points, 
    transition_volume_TE_points
)

# endregion

# region ============================ mission parameters ===========================

# mission flags
problem_type = 'planform'
do_planform = False
do_stability = False
do_runway = False
do_climb = False
do_OEI = False

if problem_type == 'planform':
    do_planform = True
elif problem_type == 'stability':
    do_planform = True
    do_stability = True
    do_runway = True
    stability_level = 'relaxed'
elif problem_type == 'full':
    do_planform = True
    do_stability = True
    do_runway = True
    do_climb = True
    do_OEI = True
    stability_level = 'relaxed'

'''
NOTE: The problem types that have been tested so far 
are baseline and planform. We are still working on stability,
so it may not work out of the box.
'''
nmi_to_m = 1852.

# 9 missions ====
payload_weight_lbf = np.array([[160000]*3+[100000]*3+[50000]*3]).flatten() # lbf
cruise_range_nmi = np.array([2500, 2000, 1500]*3).flatten() # nmi
payload_weighting = np.array([0.25, 0.5, 0.25])
range_weighting = np.array([0.25, 0.5, 0.25])
nominal_ind = 4 # NOMINAL MISSION INDEX --> medium payload, medium range

payload_weight_N = payload_weight_lbf*lbf_to_N
cruise_range_m = cruise_range_nmi*nmi_to_m
weighting_grid = np.einsum('i,j->ij', payload_weighting, range_weighting)
weighting_array = weighting_grid.flatten()
print(f"WEIGHTING ARRAY SHAPE: {weighting_array.shape}")

# number of missions
num_cruise = len(weighting_array)
num_sizing = 2
num_nodes = num_cruise+num_sizing

num_stab = 1
num_nodes += num_stab
num_climb = 3
num_OEI = 2
if problem_type == 'full':
    num_nodes += (num_stab+num_climb+num_OEI)

dalpha_stab = 0.1 # for stability analysis


# cruise mission parameters
cruise_mach = 0.7
cruise_h = 30000 # altitude in feet
cruise_h_km = cruise_h*0.3048/1000

# structural sizing mission parameters (SEA LEVEL)
SS_mach = 0.7
V_inf_SS = 340.3*SS_mach
rho_SS = 1.225


'''
NOTE:
- assume cruise uses 75% of fuel fraction (to be conservative)
- original fuel weight: 44885.25 lbf (according to Nick)
'''

# endregion

# region ============================ geometry parameterization ===========================

wing_sweep = csdl.Variable(value=np.array([35.])) # DV
wing_dihedral = csdl.Variable(value=np.array([10.])) 

center_half_span = csdl.Variable(value=np.array([5.])) # DV
transition_half_span = csdl.Variable(value=np.array([4.891])) # DV
wing_half_span = csdl.Variable(value=np.array([25.852 - 9.891])) # DV

num_centerbody_ffd_sections = 3
num_wing_ffd_sections = 4
num_ffd_sections = 8 # two above + the one in transition
num_ffd_coeff_chordwise = 8

centerbody_twist_dist = csdl.Variable(value=np.zeros(num_centerbody_ffd_sections-1))# DV (not the centerline)
wing_twist_dist = csdl.Variable(value=np.linspace(0,-2.5,num_wing_ffd_sections)) # DV

init_centerbody_chord_dist = np.array([30., 30-3.899, 30-9.813])
centerbody_chord_dist = csdl.Variable(value=init_centerbody_chord_dist)
init_wing_chord_dist = np.array([6, 5.165, 3.832,3])
wing_chord_dist = csdl.Variable(value=init_wing_chord_dist)

thickness_percent_change = csdl.Variable(value=np.zeros((num_ffd_coeff_chordwise, num_ffd_sections))) # DV
camber_percent_change = csdl.Variable(value=np.zeros((num_ffd_coeff_chordwise-2, num_ffd_sections))) # DV, not applied to first and last coeff

wing_sectional_width = np.array([
    14.33-9.891,
    21.413-14.33,
    25.852-21.413
]) * wing_half_span/(25.852-9.891)

# ============ propagating distribution DVs across the entire span ============
sectional_span_dist = csdl.Variable(value=np.zeros(3))
sectional_span_dist = sectional_span_dist.set(csdl.slice[0], value=center_half_span)
sectional_span_dist = sectional_span_dist.set(csdl.slice[1], value=transition_half_span)
sectional_span_dist = sectional_span_dist.set(csdl.slice[2], value=wing_half_span)

ffd_sectional_chord_dist = csdl.Variable(value=np.zeros((num_centerbody_ffd_sections+num_wing_ffd_sections)))
ffd_sectional_chord_dist = ffd_sectional_chord_dist.set(csdl.slice[:num_centerbody_ffd_sections], centerbody_chord_dist)
ffd_sectional_chord_dist = ffd_sectional_chord_dist.set(csdl.slice[num_centerbody_ffd_sections:], wing_chord_dist)

ffd_sectional_avg_chord_dist = (ffd_sectional_chord_dist[:-1] + ffd_sectional_chord_dist[1:])/2.

ffd_sectional_span_dist = csdl.Variable(value=np.zeros((num_centerbody_ffd_sections+num_wing_ffd_sections-1)))
ffd_sectional_span_dist = ffd_sectional_span_dist.set(
    csdl.slice[:num_centerbody_ffd_sections-1],
    center_half_span/(num_centerbody_ffd_sections-1) 
)
ffd_sectional_span_dist = ffd_sectional_span_dist.set(
    csdl.slice[num_centerbody_ffd_sections-1],
    transition_half_span 
)
ffd_sectional_span_dist = ffd_sectional_span_dist.set(
    csdl.slice[num_centerbody_ffd_sections:],
    wing_sectional_width
)
planform_area = 2*csdl.sum(ffd_sectional_avg_chord_dist*ffd_sectional_span_dist)
planform_area.add_name('planform_area')
planform_area.save()

wingspan = (center_half_span+transition_half_span+wing_half_span)*2
AR = wingspan**2/planform_area
AR.add_name('AR')
AR.save()
# MAC = wingspan/AR

# MAC = 2/S*integral(c(y)^2)dy from 0 to b/2
MAC_integral_term = ffd_sectional_span_dist * \
    (ffd_sectional_chord_dist[:-1]**2 + ffd_sectional_chord_dist[1:]**2)/2
MAC = csdl.sum(MAC_integral_term)*2/planform_area
MAC.add_name('MAC')
MAC.save()

# storing in a dictionary
geometry_values_dict = {
    'sweep': wing_sweep,
    'dihedral': wing_dihedral,
    'sectional span': sectional_span_dist,
    'centerbody chord': centerbody_chord_dist,
    'wing chord': wing_chord_dist,
    'centerbody twist': centerbody_twist_dist,
    'wing twist': wing_twist_dist,
    'thickness change': thickness_percent_change,
    'camber change': camber_percent_change,
    'elevator rotation': 0., # dummy var
}

centerbody_volume_LE_points = np.array([
    [0., 0., 0.],
    [1.433, 1.25, 0.],
    [3.899, 2.5, 0.],
    [6.748, 3.75, 0.],
    [9.813, 5., 0.],
])
centerbody_volume_TE_points = np.array([
    [30., 0., 0.],
    [30., 1.25, 0.],
    [30., 2.5, 0.],
    [30., 3.75, 0.],
    [30., 5., 0.],
])

volume_projection_points = project_centerbody_volume_points(
    geometry, 
    centerbody_volume_LE_points, 
    centerbody_volume_TE_points
)

with Timer(f'setting up geometry', rank, timing_enabled):
    # Had to "serialize" this because I was getting race conditions in cache I/O
    for r in range(comm_size):
        quiet_barrier(comm)
        if rank == r:
            geometry, ffd_block = setup_geometry(geometry, geometry_values_dict, make_video=False)
        quiet_barrier(comm)

centerbody_volume = 2*compute_volume(geometry, volume_projection_points)
centerbody_volume_0 = centerbody_volume.value
print('initial centerbody volume (m^3): ', centerbody_volume_0)

geometry_coefficients = geometry.stack_coefficients()
geometry_coefficients.add_name('geometry_coefficients')
geometry_coefficients.save()

# endregion

recorder.inline = not shutdown_inline

# region ============================ aero solver setup ============================
# Flight condition variables
flight_conditions_group                     = csdl.VariableGroup()
flight_conditions_group.mach_number         = csdl.Variable(value=cruise_mach,      name="mach_number")
flight_conditions_group.angle_of_attack_deg = csdl.Variable(value=aoa0,             name="angle_of_attack")
flight_conditions_group.altitude_m          = csdl.Variable(value=cruise_h_km/1000, name="altitude (m)")

# Atmospheric condition variables
ambient_conditions_group = sam.compute_ambient_conditions_group(flight_conditions_group.altitude_m)

flight_conditions_group.airspeed_m_s  = flight_conditions_group.mach_number * ambient_conditions_group.a_m_s

with Timer(f'evaluating geometry component', rank, timing_enabled):
    x_surf_dafoam_full = geometry.evaluate(projected_surf_mesh_dafoam, plot=False)

# region Surface mesh distribution
i0, i1          = x_surf_dafoam_initial_indices[rank]


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

    # DAFoamForces explicit component
    dafoam_force_comp       = DAFoamForces(dafoam_instance=dafoam_instance)
    dafoam_forces           = dafoam_force_comp.evaluate(dafoam_solver_states, dafoam_input_variables_group)
    dafoam_forces_array     = csdl.reshape(dafoam_forces, (-1, 3))
    dafoam_forces_full      = csdl.experimental.mpi.index_gatherer(i0, i1, comm)(dafoam_forces_array)

    # DAFoamFunctions Explicit component setup and evaluation
    dafoam_functions = DAFoamFunctions(dafoam_instance)
    dafoam_function_outputs = dafoam_functions.evaluate(dafoam_solver_states, 
                                                        dafoam_input_variables_group)
    
    comm.Barrier()

    mpi_region.set_as_global_output(dafoam_function_outputs.lift)
    mpi_region.set_as_global_output(dafoam_function_outputs.drag)
    mpi_region.set_as_global_output(dafoam_function_outputs.moment_y)
    mpi_region.set_as_global_output(dafoam_forces_full)


# Perform reflection of surface points and forces for structural solver
y_reflect_mat = np.eye(3)
y_reflect_mat[1, 1] = -1
x_surf_dafoam_full_reflected  = x_surf_dafoam_full
x_surf_dafoam_full_reflected  = x_surf_dafoam_full_reflected @ y_reflect_mat
x_surf_dafoam_full_all_points = csdl.concatenate([x_surf_dafoam_full, x_surf_dafoam_full_reflected], axis=0)

# TODO: NEED ALLGATHER EQUIVALENT HERE (THESE ARE ALL DEFINED ONLY ON PARTITIONS)
dafoam_forces_full_reflected  = dafoam_forces_full @ y_reflect_mat
dafoam_forces_full_all_points = csdl.concatenate([dafoam_forces_full, dafoam_forces_full_reflected], axis=0)

# endregion

# region ============================ structures solver setup ============================
# structural mesh elements and nodes
num_half_nodes = num_nonwing_nodes + num_wing_nodes
num_half_el = num_half_nodes-1
num_beam_nodes = num_half_nodes*2 - 1
num_beam_el = num_beam_nodes-1

half_beam_mesh = (geometry.evaluate(LE_points_projected) + (geometry.evaluate(TE_points_projected)))/2.
width_fraction = 0.45
height_fraction = 0.65
projected_height = height_fraction*csdl.norm(
    geometry.evaluate(top_pts_projected) - geometry.evaluate(bot_pts_projected),
    axes=(1,)
)
projected_width = width_fraction*csdl.norm(
    geometry.evaluate(LE_points_projected)[1:,:] - geometry.evaluate(TE_points_projected)[1:,:],
    axes=(1,)
)

# beam mesh
beam_mesh = csdl.Variable(value=np.zeros((num_beam_nodes,3)))
beam_mesh = beam_mesh.set(csdl.slice[num_half_nodes-1:,:], value=half_beam_mesh)
beam_mesh = beam_mesh.set(csdl.slice[:num_half_nodes-1,0::2], value=half_beam_mesh[1:,0::2][::-1,:])
beam_mesh = beam_mesh.set(csdl.slice[:num_half_nodes-1,1], value=-half_beam_mesh[1:,1][::-1])

# beam thicknesses
half_ttop = np.ones((num_half_el,)) * 0.001
half_ttop = csdl.Variable(value=half_ttop)

ttop = csdl.Variable(value=np.zeros(num_beam_el)) # we assume symmetry on the top and bottom
ttop = ttop.set(csdl.slice[:num_half_el], value=half_ttop[::-1])
ttop = ttop.set(csdl.slice[num_half_el:], value=half_ttop)

half_tweb = np.ones((num_half_el,)) * 0.001
half_tweb = csdl.Variable(value=half_tweb)

tweb = csdl.Variable(value=np.zeros(num_beam_el))
tweb = tweb.set(csdl.slice[:num_half_el], value=half_tweb[::-1])
tweb = tweb.set(csdl.slice[num_half_el:], value=half_tweb)

# beam height
half_height = np.zeros((num_half_el,))
half_height = csdl.Variable(value=half_height)

height = csdl.Variable(value=np.zeros(num_beam_el))
height = height.set(csdl.slice[:num_half_el], value=projected_height[::-1])
height = height.set(csdl.slice[num_half_el:], value=projected_height)

height.add_name('beam_height')
height.save()

# beam width
half_width = np.zeros((num_half_el,))
half_width = csdl.Variable(value=half_width)

width = csdl.Variable(value=np.zeros(num_beam_el))
width = width.set(csdl.slice[:num_half_el], value=projected_width[::-1])
width = width.set(csdl.slice[num_half_el:], value=projected_width)

width.add_name('beam_width')
width.save()

# beam cross section
beam_CS = af.CSBox(ttop=ttop, tbot=ttop, tweb=tweb, height=height, width=width)

# beam material
aluminum = {"E":69E9, "G":26E9, "density":2700}#af.Material(name='aluminum', E=69E9, G=26E9, density=2700)

# beam OBJECT
BWB_beam = af.Beam(
    name='BWB_beam',
    mesh=beam_mesh,
    **aluminum,
    #material=aluminum,
    cs=beam_CS
)
BWB_beam_mass = BWB_beam.mass # property of beam
BWB_beam_mass.add_name('wing_mass')
BWB_beam_mass.save()
wing_weight_N = BWB_beam_mass*9.81

cg_beam = BWB_beam.cg
cg_beam.add_name('wing_cg')
cg_beam.save()

# force map from aero to beam
mapper = NodalMap(normalization_eps=1.e-6)
force_map = mapper.evaluate(x_surf_dafoam_full_all_points, beam_mesh.reshape((-1,3))) # TODO: DAFoam surface indices (Reflected)
# endregion

# region ============================ mass properties ============================
# payload cg (same as centerbody centroid)
LE_centerline = geometry.evaluate(LE_points_projected[0])
TE_centerline = geometry.evaluate(TE_points_projected[0])
payload_cg = (LE_centerline+TE_centerline)/2

payload_cg.add_name('payload_cg')
payload_cg.save()


# engine cg

engine_weight_lbs = 8760 # lbs
engine_weight_N = engine_weight_lbs*lbf_to_N
engine_diam = 2.67 # meters
engine_length = 4.24 # meters

engine_cg_evaluated = geometry.evaluate(engine_loc_parametric)
engine_sec_LE = geometry.evaluate(engine_sec_LE_parametric)
engine_sec_TE = geometry.evaluate(engine_sec_TE_parametric)

ecf = csdl.Variable(value=np.array([ecf_0])) # engine position chord fraction DV
engine_x_pos = engine_sec_TE[0]*ecf + engine_sec_LE[0]*(1-ecf)

engine_cg = csdl.Variable(value=engine_loc_0)
engine_cg = engine_cg.set(csdl.slice[0], engine_x_pos)
engine_cg = engine_cg.set(csdl.slice[1:], engine_cg_evaluated[1:])

engine_cg.add_name('engine_cg_neg')
engine_cg.save()

engine_cg_mirror = csdl.Variable(value=engine_loc_0)
engine_cg_mirror = engine_cg_mirror.set(csdl.slice[0], engine_x_pos)
engine_cg_mirror = engine_cg_mirror.set(csdl.slice[1], -engine_cg_evaluated[1])
engine_cg_mirror = engine_cg_mirror.set(csdl.slice[2], engine_cg_evaluated[2])

engine_cg_mirror.add_name('engine_cg_pos')
engine_cg_mirror.save()

# fuel cg (same as transition centroid)
transition_volume = 2*compute_volume(geometry, transition_volume_projection_points)
transition_volume.add_name('transition_volume')
transition_volume.save()

transition_LE = geometry.evaluate(transition_volume_LE_points_proj)
transition_TE = geometry.evaluate(transition_volume_TE_points_proj)

transition_avg = (transition_LE + transition_TE)/2
transition_cg = csdl.average(transition_avg, axes=(0,))

transition_cg.add_name('fuel_cg_pos')
transition_cg.save()

transition_cg_mirror = csdl.Variable(value=np.zeros(transition_cg.shape))
transition_cg_mirror = transition_cg_mirror.set(csdl.slice[:], transition_cg)
transition_cg_mirror = transition_cg_mirror.set(csdl.slice[1], -transition_cg[1])

transition_cg_mirror.add_name('fuel_cg_neg')
transition_cg_mirror.save()

fuel_cg = (transition_cg + transition_cg_mirror)/2. # dumb to do it this way but it's exact

# endregion


# region ============================ mission analysis ============================

'''
Computations for conditions:
- cruise conditions need drag approximations to measure L/D
- stability one only needs static margin
- structural conditions need none of these
'''

L = dafoam_function_outputs.lift
L.add_name('L')
L.save()
D = dafoam_function_outputs.drag
D.add_name('D')
D.save()
M = dafoam_function_outputs.moment_y
M.add_name('M_y')
M.save()

# computing coefficients.
dynamic_pressure = 0.5 * ambient_conditions_group.rho_kg_m3 * flight_conditions_group.airspeed_m_s ** 2
CL = L/(dynamic_pressure * planform_area)
CL.add_name('CL')
CL.save()
CD = D/(dynamic_pressure * planform_area)
CD.add_name('CD')
CD.save()
CM = M/(dynamic_pressure * planform_area * MAC)
CM.add_name('CMy')
CM.save()

# skin friction drag coefficient + strip theory
sec_chord_st = ffd_sectional_avg_chord_dist
sec_plan_area_st = ffd_sectional_span_dist*sec_chord_st
exp_shape = (num_cruise, sec_chord_st.shape[0])
sec_chord_st_nn = csdl.expand(sec_chord_st, exp_shape, 'i->ai')
sec_plan_area_st_nn = csdl.expand(sec_plan_area_st, exp_shape, 'i->ai')

plan_2_wetted_area_conv = 2.1
wetted_area = sec_plan_area_st_nn*plan_2_wetted_area_conv # use 2x the sectional planform area here since we say flat plate
FF = 1.5 # add a form factor equation here

# fuel burn + takeoff weight calculations
L_D = L/D
L_D.add_name('L_D')
L_D.save()
TSFC = 0.355 # lb/lbf/hr
# W2 = empty_weight_N_nowing+wing_weight_N+payload_weight_N
# Wf, TOGW = estimate_fuel_burn(cruise_range_m, TSFC, V_cruise_array, L_D, W2)
W_bar = empty_weight_N_nowing+wing_weight_N+payload_weight_N
W_bar.add_name('W_no_fuel')
W_bar.save()
Wf, TOGW, W2 = estimate_fuel_burn_w_reserve(cruise_range_m, TSFC, flight_conditions_group.airspeed_m_s, L_D, W_bar)
Wf.add_name('Wf')
Wf.save()
TOGW.add_name('TOGW')
TOGW.save()
W2.add_name('W2')
W2.save()

fuel_volume = estimate_fuel_volume(Wf)
fuel_volume.add_name('fuel_volume')
fuel_volume.save()
max_fuel_volume = csdl.maximum(fuel_volume, rho=1000)
max_fuel_volume.add_name('max_fuel_volume')
max_fuel_volume.save()

weighted_fuel_burn_array = Wf * weighting_array
weighted_fuel_burn = csdl.sum(weighted_fuel_burn_array)

# structural sizing missions
# TODO: Only need 1 structural sizing condition (keep the first one)
beam_forces = csdl.Variable(value=np.zeros((2,num_beam_nodes,3)))
beam_forces = beam_forces.set(csdl.slice[0,:], force_map.T() @ dafoam_forces_full_all_points)

beam_loads = csdl.Variable(value=np.zeros((2,num_beam_nodes, 6)))
beam_loads = beam_loads.set(csdl.slice[:,:,:3], beam_forces)
BWB_beam.fix(node=num_half_nodes-1)

# S1
safety_factor = 1.5
BWB_beam.add_load(beam_loads[0,:])
frame_S1 = af.Frame(beams=[BWB_beam])
frame_S1.solve()
stress_dict_S1 = frame_S1.compute_stress()
beam_stress_S1 = stress_dict_S1[BWB_beam.name]
beam_stress_S1_MPa = beam_stress_S1/1.e6
beam_max_stress_S1_MPa = csdl.maximum(beam_stress_S1_MPa, rho=1e5) # NOTE: TUNE (seems to work)
beam_max_stress_S1 = beam_max_stress_S1_MPa*1.e6
beam_max_stress_S1_SF = beam_max_stress_S1*safety_factor

# S2
BWB_beam.add_load(beam_loads[1,:])
frame_S2 = af.Frame(beams=[BWB_beam])
frame_S2.solve()
stress_dict_S2 = frame_S2.compute_stress()
beam_stress_S2 = stress_dict_S2[BWB_beam.name]
beam_stress_S2_MPa = beam_stress_S2/1.e6
beam_max_stress_S2_MPa = csdl.maximum(beam_stress_S2_MPa, rho=1e5) # NOTE: TUNE (seems to work)
beam_max_stress_S2 = beam_max_stress_S2_MPa*1.e6
beam_max_stress_S2_SF = beam_max_stress_S2*safety_factor

# cg computation (accounting for fuel weight)
TOGW_nn = csdl.Variable(shape=(num_nodes,), value=0.)
TOGW_nn = TOGW_nn.set(csdl.slice[:num_cruise], value=TOGW)
TOGW_nn = TOGW_nn.set(csdl.slice[num_cruise:num_cruise+num_sizing], value=TOGW[0])
if do_stability:
    TOGW_nn = TOGW_nn.set(csdl.slice[stab_ind], value=TOGW[nominal_ind])

Wf_nn = csdl.Variable(shape=(num_nodes,), value=0.)
Wf_nn = Wf_nn.set(csdl.slice[:num_cruise], value=Wf)
Wf_nn = Wf_nn.set(csdl.slice[num_cruise:num_cruise+num_sizing], value=Wf[0])
if do_stability:
    Wf_nn = Wf_nn.set(csdl.slice[stab_ind], value=Wf[nominal_ind])

payload_weight_nn = csdl.Variable(shape=(num_nodes,), value=0.)
payload_weight_nn = payload_weight_nn.set(csdl.slice[:num_cruise], value=payload_weight_N)
payload_weight_nn = payload_weight_nn.set(csdl.slice[num_cruise:num_cruise+num_sizing], value=payload_weight_N[0])
if do_stability:
    payload_weight_nn = payload_weight_nn.set(csdl.slice[stab_ind], value=payload_weight_N[nominal_ind])

add_weight = TOGW_nn - wing_weight_N - 2*engine_weight_N - Wf_nn - payload_weight_nn # computing remaining weight at CG

target_shape = (num_nodes, 3)

cg_W_prod_wing = csdl.expand(wing_weight_N*cg_beam, target_shape, 'i->ai')
cg_W_prod_engine = csdl.expand(
    engine_weight_N*(engine_cg+engine_cg_mirror),
    target_shape,
    'i->ai'
)
cg_W_prod_fuel = csdl.expand(Wf_nn, target_shape, 'i->ia')/2. * csdl.expand(
    transition_cg+transition_cg_mirror,
    target_shape,
    'i->ai'
)
cg_W_prod_payload = csdl.expand(payload_weight_nn, target_shape, 'i->ia') * csdl.expand(
    payload_cg,
    target_shape,
    'i->ai'
)
cg_W_prod = cg_W_prod_wing+cg_W_prod_engine+cg_W_prod_fuel+cg_W_prod_payload

# cg_W_prod = wing_weight_N*cg_beam + engine_weight_N*(engine_cg+engine_cg_mirror) + \
#             Wf/2*(transition_cg+transition_cg_mirror) + payload_weight_N*payload_cg

TOGW_exp = csdl.expand(TOGW_nn, target_shape, 'i->ia')
add_weight_exp = csdl.expand(add_weight, target_shape, 'i->ia')

cg = (cg_W_prod) / (TOGW_exp-add_weight_exp) # add_weight assumed to be at CG
cg.add_name('cg')
cg.save()
cg_x = cg[:,0]

# force trim:
cruise_trim = L-TOGW
print(f"Rank {rank}: num_cruise {num_cruise}")
print(f"Rank {rank}: num_sizing {num_sizing}")
print(f"Rank {rank}: L.shape {L.shape}")
L_SS = L#L[num_cruise:num_cruise+num_sizing]
load_factors = csdl.Variable(value=2.5)#np.array([2.5, -1.])) # TODO: Keep the 2.5
SS_trim = L_SS - load_factors*TOGW[0]

# moment trim
aero_ref_pt = 0.
# CM_cg_cruise = CM[nominal_ind] + CL[nominal_ind]*(aero_ref_pt-cg_beam[0])/MAC
# CM_cg = CM + CL*(aero_ref_pt-cg_x)/MAC # WRONG
CM_cg = CM + CL*(cg_x-aero_ref_pt)/MAC
CM_cg.add_name('CM_cg')
CM_cg.save()


CM_cg_cruise_nominal = CM_cg[nominal_ind]

# takeoff and landing (constrain both to be less than 10000 feet)
max_thrust_lbf = 48000*2 # lb
max_thrust_N = max_thrust_lbf*lbf_to_N
max_weight = W2[0] # using largest weight for takeoff and landing
takeoff_length = takeoff(max_weight, max_thrust_N, planform_area) # m
landing_length = landing(max_weight, planform_area) # m

takeoff_length_ft = takeoff_length/.3048
landing_length_ft = landing_length/.3048

# static margin
if do_stability: # NOTE: UPDATE FOR NOMINAL INDEX
    alpha_list = [pitch_array[nominal_ind], pitch_array[stab_ind]] # pitch array is different from pitch dv
    CL_list = [CL[nominal_ind], CL[stab_ind]]
    CM_list = [CM_cg[nominal_ind], CM_cg[stab_ind]] # y-component taken above

    static_margin = compute_static_margin(alpha_list, CL_list, CM_list)

    neutral_point = static_margin*MAC + cg[nominal_ind,0] # neutral point based on nominal condition
    neutral_point.add_name('neutral_point')
    neutral_point.save()
    SM_missions = (neutral_point-cg[:,0])/MAC
    SM_missions.add_name('static_margin_missions')
    SM_missions.save()


# endregion

# region ============================ DVs ============================
flight_conditions_group.angle_of_attack_deg.set_as_design_variable(lower=-5, upper=10)
flight_conditions_group.angle_of_attack_deg.add_name('pitch')

half_ttop.set_as_design_variable(lower=0.001, upper=0.1, scaler=100)
half_ttop.add_name('half_ttop')
half_tweb.set_as_design_variable(lower=0.001, upper=0.1, scaler=100)
half_tweb.add_name('half_tweb')

centerbody_twist_dist.set_as_design_variable(lower=-5, upper=5)
centerbody_twist_dist.add_name('centerbody_twist')
wing_twist_dist.set_as_design_variable(lower=-15, upper=5)
wing_twist_dist.add_name('wing_twist')

if do_planform:
    center_half_span.set_as_design_variable(lower=3, upper=8) # orig: 5
    center_half_span.add_name('center_half_span')
    transition_half_span.set_as_design_variable(lower=3, upper=8) # orig: 5
    transition_half_span.add_name('transition_half_span')
    wing_half_span.set_as_design_variable(lower=8, upper=22)# orig: 16
    wing_half_span.add_name('wing_half_span')

    centerbody_chord_dist.set_as_design_variable(
        lower=0.7*init_centerbody_chord_dist,
        upper=1.3*init_centerbody_chord_dist,
    )
    centerbody_chord_dist.add_name('centerbody_chord')
    wing_chord_dist.set_as_design_variable(
        lower=0.7*init_wing_chord_dist,
        upper=1.3*init_wing_chord_dist,
    )
    wing_chord_dist.add_name('wing_chord')

    wing_sweep.set_as_design_variable(lower=15, upper=50)
    wing_sweep.add_name('wing_sweep')

if do_stability:
    thickness_percent_change.set_as_design_variable(lower=-10, upper=15)
    thickness_percent_change.add_name('thickness_percent_change')

    camber_percent_change.set_as_design_variable(lower=-20, upper=10)
    camber_percent_change.add_name('camber_percent_change')

    ecf.set_as_design_variable(lower=0.6, upper=0.85)
    ecf.add_name('engine_chord_fraction')
# endregion

# region ============================ constraints ============================

# ==== trim constraints ====
# cruise trim
cruise_trim.set_as_constraint(equals=0., scaler=1.e-6)
cruise_trim.add_name('cruise_trim')

if problem_type != 'baseline':
    CM_cg_cruise_nominal.set_as_constraint(equals=0.)
    CM_cg_cruise_nominal.add_name('cruise_cg_CM_nominal_trim_constraint')

# structural sizing trim
# use MTOW of heaviest payload and largest range
SS_trim.set_as_constraint(equals=0., scaler=1.e-6)
SS_trim.add_name('structural_sizing_trim')

# mass stress constraints
max_allowable_stress = 324e6 # 
beam_max_stress_S1_SF.set_as_constraint(upper=max_allowable_stress, scaler=1.e-9)
beam_max_stress_S1_SF.add_name('S1_max_stress')
beam_max_stress_S2_SF.set_as_constraint(upper=max_allowable_stress, scaler=1.e-9)
beam_max_stress_S2_SF.add_name('S2_max_stress')

if problem_type != 'baseline':
    # # ==== volume constraints ====
    C_17_cargo_volume = 558.36 # m^3
    centerbody_volume.set_as_constraint(lower=C_17_cargo_volume, scaler=1e-2)
    centerbody_volume.add_name('centerbody_volume_constraint')
    fuel_volume_constraint = transition_volume - max_fuel_volume
    fuel_volume_constraint.set_as_constraint(lower=0., scaler=1.e-1)
    fuel_volume_constraint.add_name('fuel_volume_constraint')

    # # ==== takeoff and landing length constraint ====
    takeoff_length_ft.set_as_constraint(upper=10000, scaler=1.e-4)
    takeoff_length_ft.add_name('takeoff_length_constraint')
    landing_length_ft.set_as_constraint(upper=10000, scaler=1.e-4)
    landing_length_ft.add_name('landing_length_constraint')

    # ==== engine collision constraint ====
    # this enforces that the engines are separated by at least 1 diameter
    engine_y_pos = engine_cg_mirror[1]
    engine_gap_scaler = 1.
    engine_min_gap = (engine_y_pos - engine_diam*engine_gap_scaler)/engine_diam # normalized by actual engine diameter
    engine_min_gap.set_as_constraint(lower=0)
    engine_min_gap.add_name('engine_spacing_constraint')

if problem_type == 'stability':
    # ==== static margin constraint ====
    if stability_level == 'relaxed':
        static_margin_bound = 0.03
    elif stability_level == 'tight':
        static_margin_bound = 0.10
    static_margin.set_as_constraint(lower=static_margin_bound, scaler=1e0)
    static_margin.add_name('static_margin_constraint')

# endregion

# ============================ objective ============================
weighted_fuel_burn.set_as_objective(scaler=1e-5)
weighted_fuel_burn.add_name('weighted_fuel_burn_objective')

csdl.save_optimization_variables()


recorder.stop()


if False:
    print('starting compile of sample forward run')
    start_time = time.time()
    sim.run()
    end_time = time.time()
    print(f'compile time: {end_time-start_time} seconds')
    
    exit()

print(f'============ problem type: {problem_type} ============')
print(f'============ number of cruise missions: {num_cruise} ============')

print('setting up Problem')

now = datetime.now()
prob_start_date = now.date()
prob_start_time = now.time()
print('================')
print(f'prob start date: {prob_start_date}')
print(f'prob start time: {prob_start_time}')
print('================')

sim = csdl.experimental.PySimulator(recorder)

# Only allow visualization and modopt output files on the root rank
visualize_on_this_rank           = True  if rank == 0 and not is_headless() else False
turn_off_outputs_on_nonroot_rank = False if rank == 0 else True
recording_on_root_rank           = True  if rank == 0 else False
rank_outputs                     = ['x'] if rank == 0 else []

# Optimization solver setup and run
prob                = CSDLAlphaProblem(problem_name=f'{problem_name}', simulator=sim)
optimizer_choice    = 2 # Set to 1 for PySLSQP, 2 for OpenSQP, or 3 for InteriorPoint

if optimizer_choice == 1:
    # PySLSQP optimizer setup
    solver_options = {'maxiter': 20,
                    'iprint': 2,
                    'readable_outputs': rank_outputs,
                    'recording': recording_on_root_rank,
                    'turn_off_outputs': turn_off_outputs_on_nonroot_rank}
    optimizer   = PySLSQP(prob, solver_options=solver_options)
    optimizer.solve()
    optimizer.print_results()

elif optimizer_choice == 2:
    # OpenSQP optimizer setup
    open_sqp_options = {'maxiter': 80,
                        'readable_outputs': rank_outputs,
                        'recording': recording_on_root_rank,
                        'ls_max_step': 1.,
                        'turn_off_outputs': turn_off_outputs_on_nonroot_rank,
                        }
                        # 'hot_start_from': '/media/edward/DATA/Edward/AFRL_project/csdl_dafoam_workspace/blended_wing_body_case/results/case5_opensqp/case5_opensqp_outputs/2026-02-05_07.59.06.838711/record.hdf5',
                        # 'hot_start_rtol': 1e-4}
    optimizer = OpenSQP(prob, **open_sqp_options)
    optimizer.solve()
    optimizer.print_results()

elif optimizer_choice == 3:
    # InteriorPoint optimizer setup
    interior_point_options = {'maxiter': 40,
                            'readable_outputs': rank_outputs,
                            'recording': recording_on_root_rank,
                            'ls_max_step': 1.,
                            'turn_off_outputs': turn_off_outputs_on_nonroot_rank}
    optimizer   = InteriorPoint(prob, **interior_point_options)
    optimizer.solve()
    optimizer.print_results()
    
else:
    print(f'Check optimizer choice. {optimizer_choice} is not an option.')
