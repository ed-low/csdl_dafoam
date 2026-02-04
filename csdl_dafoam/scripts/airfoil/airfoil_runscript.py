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

# LSDO_geo specific
from lsdo_geo.core.parameterization.free_form_deformation_functions import (
    construct_tight_fit_ffd_block,construct_ffd_block_around_entities
)
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables

# Optimization
from modopt import CSDLAlphaProblem
from modopt import PySLSQP, OpenSQP

# IDWarp and DAFoam
from csdl_dafoam.core.csdl_idwarp import DAFoamMeshWarper
from csdl_dafoam.core.csdl_dafoam import instantiateDAFoam, DAFoamFunctions, DAFoamSolver, compute_dafoam_input_variables
import csdl_dafoam.utils.standard_atmosphere_model as sam
from csdl_dafoam.utils.runscript_helper_functions import *

# Plotting
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
# Keyword for optimization name (optimization results folder will be saved with this name)
problem_name              = 'airfoil_test'

# Geometry
geometry_directory        =  os.path.join(os.getcwd(), 'airfoil_geometry/')
stp_file_name             = 'airfoil_transonic_unitspan_2.stp'
geometry_pickle_file_name = 'airfoil_stored_refit.pickle'

# MPI and timing
comm           = MPI.COMM_WORLD
TIMING_ENABLED = True  # True if we want timing printed for the CSDL operations

# DAFoam
dafoam_directory = os.path.join(os.getcwd(), f'results/{problem_name}/')

# Initial/reference values for DAFoam (best to use base conditions)
U0        = 238.0         # used for normalizing CD and CL
p0        = 101325.0
T0        = 300.0
nuTilda0  = 4.5e-5
aoa0      = 0
A0        = 0.1           #
rho0      = p0 / T0 / 287 # used for normalizing CD and CL

# Input parameters for DAFoam
da_options = {
    "designSurfaces": ["wing"],
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
rank      = comm.Get_rank()
comm_size = comm.Get_size()
rank_str  = f"{rank:0{len(str(comm_size-1))}d}" # string with zero-padded rank index (for prints)


# region DAFoam instance
dafoam_instance             = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)
x_surf_dafoam_initial   = dafoam_instance.getSurfaceCoordinates()


# region File paths
geometry_pickle_file_path         = Path(geometry_directory)/geometry_pickle_file_name
stp_file_path                     = Path(geometry_directory)/stp_file_name



# ===============================
# region CSDL RECORDER
# ===============================
# recorder 
recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()


geometry = lsdo_geo.import_geometry(stp_file_path,
                                                parallelize=False)


projected_surf_mesh_dafoam = geometry.project(
    x_surf_dafoam_initial, 
    grid_search_density_parameter = 1,      
    projection_tolerance          = 1.e-3,  
    grid_search_density_cutoff    = 50,     
    force_reprojection            = False,
    plot                          = False  
)

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

x_surf_dafoam = geometry.evaluate(projected_surf_mesh_dafoam, plot=False)
x_surf_dafoam   = x_surf_dafoam.flatten()

# region IDWarp and DAFoam
idwarp_model    = DAFoamMeshWarper(dafoam_instance)
x_vol_dafoam    = idwarp_model.evaluate(x_surf_dafoam)

# Flight condition variables
flight_conditions_group                 = csdl.VariableGroup()
flight_conditions_group.mach_number     = csdl.Variable(value=0.7, name="mach_number")
flight_conditions_group.angle_of_attack_deg = csdl.Variable(value=aoa0, name="angle_of_attack_deg")
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
dafoam_solver           = DAFoamSolver(dafoam_instance, always_use_same_ic=True)
dafoam_solver_states    = dafoam_solver.evaluate(dafoam_input_variables_group)

# DAFoamFunctions Explicit component setup and evaluation
dafoam_functions        = DAFoamFunctions(dafoam_instance, disable_jacvec_normalization=True)
dafoam_function_outputs = dafoam_functions.evaluate(dafoam_solver_states, 
                                                    dafoam_input_variables_group)


# region Optimization problem selection
# optimization_case options
# 1: Maximize CL/CD wrt angle-of-attack
# 2: Minimize CD wrt angle-of-attack, root/tip twist, constrained by CL=0.5
# 3: Minimize CD wrt angle-of-attack, wing shape (thickness/camber ffd), constrained by CL=0.5
# 4: Minimize CD wrt angle-of-attack, wing shape (thickness/camber ffd) and wing twists, constrained by CL=0.5
# 5: Maximize CL/CD wrt angle-of-attack and wing shape
optimization_case = 1


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
    percent_change_in_thickness_dof.set_as_design_variable(lower=-100, upper=100, scaler=1./100)
    normalized_percent_camber_change_dof.set_as_design_variable(lower=-50, upper=50, scaler=1./50)

    # Objectives
    objective_fun = -lift/drag
    objective_fun.set_as_objective()


else:
    print('Not a valid case number')


recorder.stop()



# ===============================
# region SIM
# ===============================
sim = csdl.experimental.PySimulator(recorder)

from csdl_dafoam.utils.csdl_test_functions import test_jacvec_product, test_idempotence, test_inverse_jacobian
np.random.seed(0)

test_component = dafoam_solver

inputs  = {k: vv.value for k, vv in test_component.input_dict.items()}
v       = {k: np.random.rand(*vv.value.shape)*vv.value for k, vv in test_component.output_dict.items()}
w       = {k: np.random.rand(*vv.value.shape)*vv.value for k, vv in test_component.input_dict.items()}

print(f'Inputs: {inputs}')
print(f'v: {v}')
print(f'w: {w}')

# test_idempotence(test_component, inputs)
# input('Press ENTER to continue...')

# test_inverse_jacobian(test_component, inputs, {name: 2*vv for name, vv in v.items()}, eps=1e-4)


eps_test_vals = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
err = np.zeros_like(eps_test_vals)
i = 0
for eps in eps_test_vals:
    lhs, rhs, err[i] = test_jacvec_product(test_component, inputs, v, w, eps=eps)
    i += 1 

plt.rcParams['text.usetex'] = True
plt.figure()
# ax = plt.subplot(2, 1, 1)

plt.loglog(eps_test_vals, err)
plt.title(r'jacvec_product vs FD ($w^T J^T v = v^T J w$)')
plt.xlabel(r'Stepsize, $\epsilon$')
plt.ylabel(r'Error, $\frac{lhs - rhs}{rhs}$')
plt.grid(visible=True)
plt.show(block=False)

input('Press ENTER to continue...')

# import shutil
# err = np.zeros_like(eps_test_vals)
# i = 0
# for eps in eps_test_vals:
#     lhs, rhs, err[i] = test_inverse_jacobian(test_component, inputs, v, eps=eps)
#     # shutil.rmtree(dafoam_directory + "0.0001")
#     i += 1

# ax = plt.subplot(2, 1, 2)
# plt.loglog(eps_test_vals, err)
# plt.title(r'inverse_jacobian vs FD ($v^T v = (J^{-T} v)^T (J v)$)')
# plt.xlabel(r'Stepsize, $\epsilon$')
# plt.ylabel(r'Error, $\frac{lhs - rhs}{rhs}$')
# plt.grid(visible=True)
# plt.show()

# input('Press ENTER to continue...')

# Can set design variables here and run sim to test
# sim[root_twist]  = 3*3.14159/180
# sim.run()

# Uncomment to run and check derivatives via finite difference
# sim.check_totals()
# derivs = sim.compute_totals([CD],[root_twist, tip_twist, flight_conditions_group.angle_of_attack_deg])

# Only allow visualization on the root rank
if rank == 0 and not is_headless():
    visualize_on_this_rank = True
else:
    visualize_on_this_rank = False

# Optimization solver setup and run
prob        = CSDLAlphaProblem(problem_name=f'{problem_name}_rank{rank_str}', simulator=sim)

# # PySLSQP optimizer setup
# solver_options = {'maxiter': 20,
#                   'iprint': 2,
#                   'visualize': visualize_on_this_rank,
#                   'summary_filename': f'rank{rank_str}_slsqp_summary.out',
#                   'save_figname':     f'rank{rank_str}_slsqp_plot.pdf',
#                   'save_filename':    f'rank{rank_str}_slsqp_recorder.hdf5'}
# optimizer   = PySLSQP(prob, solver_options=solver_options)
# optimizer.solve()
# optimizer.print_results()


# # OpenSQP optimizer setup
# open_sqp_options = {'maxiter': 20,
#                     'readable_outputs': ['x']}
# optimizer   = OpenSQP(prob, **open_sqp_options)
# optimizer.solve()





# # Extra items to use, if necessary
# optimizer.check_first_derivatives(prob.x0)
