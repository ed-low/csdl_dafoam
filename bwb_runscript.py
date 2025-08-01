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
import lsdo_function_spaces as lfs
import lsdo_geo

# Optimization
from modopt import CSDLAlphaProblem
from modopt import PySLSQP

# IDWarp and DAFoam
from csdl_idwarp import DAFoamMeshWarper
from csdl_dafoam import instantiateDAFoam, DAFoamFunctions, DAFoamSolver, compute_dafoam_input_variables
import standard_atmosphere_model as sam

# BWB specific
from bwb_helper_functions import setup_geometry, read_geometry_pickle, write_geometry_pickle, gather_array_to_rank0, read_simple_pickle, write_simple_pickle

# Plotting
from vedo import Points, show
import matplotlib.pyplot as plt

# Hashing (for file name generation)
import hashlib

#---- DEBUGGING TOOLS ----
import faulthandler
faulthandler.enable()
os.environ["PETSC_OPTIONS"] = "-malloc_debug"
#-------------------------



# ===============================
# region USER INPUT
# ===============================
# Geometry
geometry_directory        =  os.path.join(os.getcwd(), 'bwb_geometry/')
stp_file_name             = 'bwbv2_no_wingtip_coarse_refined_flat.stp'
geometry_pickle_file_name = 'bwb_stored_refit.pickle'

# MPI and timing
comm           = MPI.COMM_WORLD
TIMING_ENABLED = True  # True if we want timing printed for the CSDL operations

# DAFoam
dafoam_directory = os.path.join(os.getcwd(), 'openfoam_175k_bwb/')

# Initial/reference values for DAFoam (best to use base conditions)
U0        = 238.0         # used for normalizing CD and CL
p0        = 101325.0
T0        = 300.0
nuTilda0  = 4.5e-5
CL_target = 0.5
aoa0      = 0
A0        = 518           # Projected area of entire BWB. Used for normalizing CD and CL
rho0      = p0 / T0 / 287 # used for normalizing CD and CL

# region Dafoam options
da_options = {
    "designSurfaces": ["wall"],
    "solverName": "DARhoSimpleCFoam",
    "primalMinResTol": 1.0e-8,
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
            "patches": ["wall"],
            "directionMode": "parallelToFlow",
            "patchVelocityInputName": "patch_velocity",
            "scale": 1.0, #1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
        "lift": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wall"],
            "directionMode": "normalToFlow",
            "patchVelocityInputName": "patch_velocity",
            "scale": 1.0, #1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {"gmresRelTol": 1.0e-4, "pcFillLevel": 1, "jacMatReOrdering": "natural"},
    # transonic preconditioner to speed up the adjoint convergence
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
# region HELPER FUNCTIONS
# ===============================
# TIMER
from contextlib import contextmanager
# Use this to print the timings for certain lines
timings = {}  # Optional: for logging total times

@contextmanager
def Timer(name):
    if TIMING_ENABLED:
        print(f'Rank {rank}: {name}...', flush=True)
        start = time.time()
        yield
        elapsed = time.time() - start
        print(f'Rank {rank}: {name} elapsed time: {elapsed:.3f} s')
        timings[name] = elapsed
    else:
        yield


# HASHER (for generating filenames)
import hashlib
def hash_array_tol(arr: np.ndarray, tol: float = 1e-8, length: int = 16) -> str:
    """
    Generate a tolerance-aware short hash of a NumPy array.

    Parameters:
        arr (np.ndarray): Input array to hash.
        tol (float): Tolerance for rounding (default: 1e-8).
        length (int): Number of hex characters to return from the hash (default: 16).

    Returns:
        str: A truncated SHA-256 hash of the rounded array.
    """
    # Round the array to the given tolerance
    rounded = np.round(arr / tol) * tol
    # Hash the byte representation of the rounded array
    byte_repr = rounded.astype(np.float64).tobytes()
    full_hash = hashlib.sha256(byte_repr).hexdigest()
    return full_hash[:length]



# ===============================
# region SETUP
# ===============================
# MPI information
rank                        = comm.Get_rank()
comm_size                   = comm.Get_size()


# region DAFoam instance
dafoam_instance             = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)
x_surf_dafoam_initial_mpi   = dafoam_instance.getSurfaceCoordinates()
x_vol_dafoam_initial_mpi    = dafoam_instance.xv0

local_n_surf  = x_surf_dafoam_initial_mpi.shape[0]
local_n_vol   = x_surf_dafoam_initial_mpi.shape[0]

# Gathering surface mesh to rank 0 (need to do this to avoid 'no-element' ranks in the projection
# and geometry evaluation functions)
(x_surf_dafoam_initial, 
x_surf_dafoam_initial_size,
x_surf_dafoam_initial_indices) = gather_array_to_rank0(x_surf_dafoam_initial_mpi, comm)

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
    with Timer(f'reading geometry'):
        geometry = read_geometry_pickle(geometry_pickle_file_path)
        
else:
    if rank == 0:
        print('No geometry pickle file found.')
        with Timer('importing geometry'):
            geometry = lsdo_geo.import_geometry(stp_file_path,
                                                parallelize=False)

        # These are hardcoded indices?
        oml_indices                 = [key for key in geometry.functions.keys()]
        wing_c_indices              = [0,1,8,9]
        wing_r_transition_indices   = [2,3]
        wing_r_indices              = [4,5,6,7]
        wing_l_transition_indices   = [10,11]
        wing_l_indices              = [12,13,14,15] 

        with Timer('declaring geometry components'):
            left_wing_transition    = geometry.declare_component(wing_l_transition_indices)
            left_wing               = geometry.declare_component(wing_l_indices)
            right_wing_transition   = geometry.declare_component(wing_r_transition_indices)
            right_wing              = geometry.declare_component(wing_r_indices)
            center_wing             = geometry.declare_component(wing_c_indices)
            oml = geometry.declare_component(oml_indices)

        wing_parameterization   = 15
        num_v                   = left_wing.functions[wing_l_indices[0]].coefficients.shape[1]
        
        with Timer('BSplineSpace'):
            wing_refit_bspline      = lfs.BSplineSpace(num_parametric_dimensions=2, degree=1, coefficients_shape=(wing_parameterization, num_v))

        with Timer('left wing refit'):
            left_wing_function_set  = left_wing.refit(wing_refit_bspline, grid_resolution=(100,1000))

        with Timer('right wing refit'):
            right_wing_function_set = right_wing.refit(wing_refit_bspline, grid_resolution=(100,1000))

        with Timer('allocating left wing functions'):
            for i, function in left_wing_function_set.functions.items():
                geometry.functions[i]   = function
                left_wing.functions[i]  = function

        with Timer('allocating right wing functions'):
            for i, function in right_wing_function_set.functions.items():
                geometry.functions[i]   = function
                right_wing.functions[i] = function

        with Timer('pickling geometry'):
            write_geometry_pickle(geometry, geometry_pickle_file_path)
    
    # Wait for root rank to finish writing
    comm.Barrier()
    if rank != 0:
        with Timer(f'reading geometry'):
            geometry = read_geometry_pickle(geometry_pickle_file_path)   


# region Surface mesh projection
# Now do we do the same check for the surface mesh projection
if surface_mesh_projection_file_path.is_file():
    if rank == 0:
        print('Found surface mesh projection pickle!')
    projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)

else:
    if rank == 0:
        print('No projected surface mesh file found.')
        with Timer('projecting on surface mesh'):
            projected_surf_mesh_dafoam = geometry.project(
                x_surf_dafoam_initial, 
                grid_search_density_parameter = 1,      # 1     (ORIGINAL)
                projection_tolerance          = 1e-1, #1.e-3,  # 1.e-3 (ORIGINAL)
                grid_search_density_cutoff    = 10,     # 20    (ORIGINAL) 50
                force_reprojection            = False,
                plot                          = False    # UCSD_LAB
            )

        print('Writing surface mesh projection pickle...')
        write_simple_pickle(projected_surf_mesh_dafoam, surface_mesh_projection_file_path)
        print('Done!')

    comm.Barrier()
    if rank != 0:
        projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)

print(f'Rank {rank} done reading projected surface mesh!')
comm.Barrier()


# region Design variables
# ============================ Design variables ===========================
root_twist  = csdl.Variable(shape=(1,), value=np.array([10*np.pi/180.]))
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
with Timer(f'setting up geometry'):
    # Had to "serialize" this because I was getting race conditions in cache I/O
    for r in range(comm_size):
        comm.Barrier()
        if rank == r:
            geometry = setup_geometry(geometry, geometry_values_dict)
        comm.Barrier()

with Timer(f'evaluating geometry component'):
    x_surf_dafoam_full = geometry.evaluate(projected_surf_mesh_dafoam, plot=False)

# region Surface mesh distribution
i0, i1          = x_surf_dafoam_initial_indices[rank]
x_surf_dafoam   = x_surf_dafoam_full[i0:i1, :]
x_surf_dafoam   = x_surf_dafoam.flatten()

# region IDWarp and DAFoam
idwarp_model    = DAFoamMeshWarper(dafoam_instance)
x_vol_dafoam    = idwarp_model.evaluate(x_surf_dafoam)

# Flight condition variables
flight_conditions_group                 = csdl.VariableGroup()
flight_conditions_group.mach_number     = csdl.Variable(value=0.6, name="mach_number")
flight_conditions_group.angle_of_attack = csdl.Variable(value=aoa0, name="angle_of_attack")
flight_conditions_group.altitude_m      = csdl.Variable(value=0., name="altitude (m)")

# Atmospheric condition variables
ambient_conditions_group = sam.compute_ambient_conditions_group(flight_conditions_group.altitude_m)

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


# region Optimization problem selection
# optimization_case options
# 1: Maximize CL/CD wrt angle-of-attack
# 2: Minimize CD wrt angle-of-attack, root/tip twist, constrained by CL=0.5
# 3: Minimize CD wrt wing shape (thickness/camber ffd), constrained by CL=0.5
# 4: Minimize CD wrt wing shape (thickness/camber ffd) and wing twists, constrained by CL=0.5
optimization_case = 3


if optimization_case == 1:
    # Declaring and naming some variables
    lift = dafoam_function_outputs.lift
    drag = dafoam_function_outputs.drag

    # Design variables
    flight_conditions_group.angle_of_attack.set_as_design_variable(lower=0, upper=10)

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
    tip_twist.name = 'tip_twist'

    # Design variables
    flight_conditions_group.angle_of_attack.set_as_design_variable(lower=0, upper=10)
    root_twist.set_as_design_variable(lower=-0.087266, upper=0.087266, scaler=180/np.pi)
    tip_twist.set_as_design_variable(lower=-0.087266, upper=0.087266, scaler=180/np.pi)

    # Constraints
    CL.set_as_constraint(lower=0.5, upper=0.5)

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
    flight_conditions_group.angle_of_attack.set_as_design_variable(lower=-2., upper=10., adder=2., scaler=1./10.)
    percent_change_in_thickness_dof_wing.set_as_design_variable(lower=-10, upper=30., adder=10., scaler=1./40.)
    normalized_percent_camber_change_dof_wing.set_as_design_variable(lower=-20., upper=20., scaler=1./20.)

    # Constraints
    CL.set_as_constraint(lower=0.5, upper=0.5)

    # Objective
    CD.set_as_objective()


elif optimization_case == 4:
    # Declaring and naming some variables
    dynamic_pressure = 0.5*ambient_conditions_group.rho_kg_m3*flight_conditions_group.airspeed_m_s*flight_conditions_group.airspeed_m_s
    lift = dafoam_function_outputs.lift
    drag = dafoam_function_outputs.drag
    CL   = lift/(dynamic_pressure*A0)
    CD   = drag/(dynamic_pressure*A0)

    # Design variables
    flight_conditions_group.angle_of_attack.set_as_design_variable(lower=-2., upper=10., adder=2., scaler=1./10.)
    percent_change_in_thickness_dof_wing.set_as_design_variable(lower=-10, upper=30., adder=10., scaler=1./40.)
    normalized_percent_camber_change_dof_wing.set_as_design_variable(lower=-20., upper=20., scaler=1./20.)
    wing_twists.set_as_design_variable(lower=-10*np.pi/180, upper=10*np.pi/180, scaler=18/np.pi)

    # Constraints
    CL.set_as_constraint(lower=0.5, upper=0.5)

    # Objective
    CD.set_as_objective()



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
if rank == 0:
    visualize_on_this_rank = True
else:
    visualize_on_this_rank = False

# Optimization solver setup and run
solver_options = {'maxiter': 20,
                  'iprint': 2,
                  'visualize': visualize_on_this_rank,
                  'summary_filename': f'rank{rank}_slsqp_summary.out',
                  'save_figname':     f'rank{rank}_slsqp_plot.pdf',
                  'save_filename':    f'rank{rank}_slsqp_recorder.hdf5'}

prob        = CSDLAlphaProblem(problem_name='BWB', simulator=sim)
optimizer   = PySLSQP(prob, solver_options=solver_options)

# optimizer.check_first_derivatives(prob.x0)

optimizer.solve()
optimizer.print_results()