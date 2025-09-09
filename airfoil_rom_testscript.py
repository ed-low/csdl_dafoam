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
from csdl_idwarp import DAFoamMeshWarper
from csdl_dafoam import instantiateDAFoam, DAFoamFunctions, DAFoamSolver, DAFoamROM, compute_dafoam_input_variables
import standard_atmosphere_model as sam
from pyofm import PYOFM

from rom_training_helper_functions import *

# Plotting
from vedo import Points, show
import matplotlib.pyplot as plt
from check_headless import is_headless

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
geometry_directory        =  Path.cwd()/'airfoil_geometry/'
stp_file_name             = 'airfoil_transonic_unitspan_2.stp'
geometry_pickle_file_name = 'airfoil_stored_refit.pickle'

# MPI and timing
comm           = MPI.COMM_WORLD
TIMING_ENABLED = True  # True if we want timing printed for the CSDL operations

# region DAFoam options
dafoam_directory = Path.cwd()/'openfoam_airfoil/'

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
    "gridFile": str(dafoam_directory),
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
rank_str      = f"{rank:0{len(str(comm_size-1))}d}" # string with zero-padded rank index (for prints)
print(comm_size)

# region File paths
geometry_pickle_file_path   = Path(geometry_directory)/geometry_pickle_file_name
stp_file_path               = Path(geometry_directory)/stp_file_name

# region DAFoam instance
dafoam_instance             = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)
x_surf_dafoam_initial       = dafoam_instance.getSurfaceCoordinates()

# Load ROM training data
state_store_path            = dafoam_directory/'airfoil_training/airfoil_training_point0.h5'
local_data, global_metadata = read_snapshots(state_store_path, dafoam_instance=dafoam_instance)


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


# Generate PYOFM object, and read reference mesh cell volumes
current_dir = os.getcwd()
os.chdir(dafoam_directory)
ofm = PYOFM(comm=dafoam_instance.comm)
cell_volumes = np.zeros((dafoam_instance.solver.getNLocalCells(), ))
ofm.readField('V', 'volScalarField', '0', cell_volumes)
os.chdir(current_dir)


# Flight condition variables
flight_conditions_group                     = csdl.VariableGroup()
flight_conditions_group.mach_number         = csdl.Variable(value=global_metadata['grassmann_configuration'][0], name="mach_number") #7.38907838e-01, name="mach_number")
flight_conditions_group.angle_of_attack_deg = csdl.Variable(value=global_metadata['grassmann_configuration'][1], name="angle_of_attack_deg") #1.01091987e-01, name="angle_of_attack_deg")
flight_conditions_group.altitude_m          = csdl.Variable(value=global_metadata['grassmann_configuration'][2], name="altitude (m)") #9.49785954e+03, name="altitude (m)")

# Atmospheric condition variables
ambient_conditions_group = sam.compute_ambient_conditions_group(flight_conditions_group.altitude_m)

# DAFoam input variable generation
# Generate our DAFoam CSDL input variable group 
# (this will add airspeed_m_s to the flight conditions group if not already present)
dafoam_input_variables_group = compute_dafoam_input_variables(dafoam_instance, 
                                                              ambient_conditions_group, 
                                                              flight_conditions_group,
                                                              x_vol_dafoam)



# region ROM options
# Training dataset location
# state_store_dir = Path.cwd()/f'2gs_100ss_state_store_comm_size{comm_size}_rank{rank_str}.npy' # (old version)

# Reduced dimension 
# (set to -1 for automatic selection, where target_energy will be used to determine cutoff)
num_modes     = -1
target_energy = 0.999  # e.g. use 0.999 for 99.9%

# Centering mode
# Options:
# 0: no centering
# 1: mean centered
# 2: center by reference
centering_mode = 1

# Scaling factors for numerical stability
# Options:
# 0: no scaling
# 1: scale by normalization factors in da_options["normalizeStates"]
# 2: scale by statistics (standard deviation of data)
scaling_mode = 2

# Inner product weighting
# Options:
# None: no weights
# "cell": weigh by cell volumes and face surface areas
weight_mode  = 'cell'
ignore_phi   = True

# Load states 
# training_data  = np.load(state_store_dir) # (old version)

# Select subsection (if necessary). We'll call our states, y
#y_training = training_data[:, :50] # (old version)
y_training = local_data['snapshots_local']



# TODO: MPI business. Would need to gather everything to rank zero, compute POD, and then disperse.
#       For now, we'll assume that this is serial and that everything is here.


# region ROM Mode computation
# Scaling mode assignment
if scaling_mode == 0:   # No scaling
    scaling_factors = np.ones((dafoam_instance.getNLocalAdjointStates(), ))

elif scaling_mode == 1: # Scale by normalization factors set in da_options["normalizeStates"]
    scaling_factors = dafoam_instance.getStateScalingFactors()

elif scaling_mode == 2: # Scale by statistics (standard deviation of data)
    scaling_factors = np.std(y_training, axis=1)

    # To avoid division by zero/small value
    eps = 1e-12
    scaling_factors[scaling_factors < eps] = 1 

else:
    raise NotImplementedError(f'scaling mode option "{scaling_mode}" not a valid choice yet.')

y_training = y_training/scaling_factors[:, None]


# Centering
if centering_mode == 0: # No centering
    y_reference = np.zeros_like(y_training[:, 0])

elif centering_mode == 1: # Mean centering
    y_reference = y_training.mean(axis=1)

elif centering_mode == 2: # Center by reference
    # DON'T FORGET TO SCALE
    y_reference = local_data['reference_snapshot_local']/scaling_factors
else:
    raise NotImplementedError(f'centering mode option "{centering_mode}" not a valid choice yet.')


# Weighting mode assignment
if weight_mode is None:
    weights = np.ones_like(y_reference)

elif weight_mode == 'cell':
    # Vector weights
    weights_v = np.empty(3*cell_volumes.size)
    weights_v[0::3] = cell_volumes
    weights_v[1::3] = cell_volumes
    weights_v[2::3] = cell_volumes
    # Bandaid solution to get face areas for now
    face_areas = dafoam_instance.getStateScalingFactors()[6*dafoam_instance.solver.getNLocalCells():]
    weights    = np.concatenate((weights_v, np.tile(cell_volumes, 3), face_areas), axis=None)


# Scale and zero-mean our data
z = np.sqrt(weights)[:, None]*(y_training - y_reference[:, None])


# Compute our svd and determine number of modes needed (if user specified), and compute POD modes
u, s, vh  = np.linalg.svd(z, full_matrices=False)


cumulative_energy = np.cumsum(s**2)
total_energy      = cumulative_energy[-1]
energy_fraction   = cumulative_energy / total_energy

if num_modes == -1:
    num_modes = np.searchsorted(energy_fraction, target_energy) + 1
    print(f"{num_modes} modes required to capture {target_energy*100}% total energy")
else:
    print(f"{num_modes} captures {energy_fraction[num_modes - 1]*100}% of the total energy")

pod_modes          = u[:, :num_modes]/np.sqrt(weights)[:, None]


# This was just a test to see the FOM residuals
# dafoam_instance()
# n_cells             = dafoam_instance.solver.getNLocalCells()
# plt.figure(0)
# for i in range(1,7): plt.axvline(n_cells*i, alpha=0.5, linestyle='--', color='gray')
# plt.plot(dafoam_instance.getResiduals())
# plt.show()

# ----- Plotting section -----
show_training_plots = False
n_cells             = dafoam_instance.solver.getNLocalCells()

if show_training_plots:
    plt.figure(1)
    for i in range(1,7): plt.axvline(n_cells*i, alpha=0.5, linestyle='--', color='gray')
    plt.plot(scaling_factors[:, None]*y_training)
    plt.plot(scaling_factors*y_reference, linestyle=':', label='reference state')
    plt.legend()
    plt.title('Raw training data')

    plt.figure(2)
    for i in range(1,7): plt.axvline(n_cells*i, alpha=0.5, linestyle='--', color='gray')
    plt.plot(y_training)
    plt.plot(y_reference, linestyle=':', label='reference state')
    plt.legend()
    plt.title('Scaled training data')

    plt.figure(3)
    for i in range(1,7): plt.axvline(n_cells*i, alpha=0.5, linestyle='--', color='gray')
    plt.plot(z)
    plt.title('Scaled, centered training data')

    plt.figure(4)
    plt.plot(energy_fraction)
    plt.axvline(num_modes, label=f'Target ({target_energy*100}%)', color='gray', alpha=0.5, linestyle='--')
    plt.title('POD mode variance capture')
    plt.xlabel('Number of modes')
    plt.ylabel('Cumulative variance')
    plt.legend()
    plt.show()

# dafoam_solver           = DAFoamSolver(dafoam_instance)
# dafoam_solver_states    = dafoam_solver.evaluate(dafoam_input_variables_group)

# DAFoamSolver Implicit component setup and evaluation
dafoam_rom           = DAFoamROM(dafoam_instance, 
                                 pod_modes=pod_modes, 
                                 tolerance=1e-8, 
                                 fom_ref_state=scaling_factors*y_reference, 
                                 max_iters=50, 
                                 scaling_factors=scaling_factors, 
                                 weights=weights, 
                                 update_jac_frequency=5,
                                 num_initial_jac_updates=1)

dafoam_rom_states    = dafoam_rom.evaluate(dafoam_input_variables_group)

dafoam_fom_states    =  scaling_factors*y_reference + scaling_factors*csdl.matvec(pod_modes, dafoam_rom_states)



# # Comparing FD to JVP J rom
# import matplotlib as mpl

# J_rom_jvp = dafoam_rom._compute_reduced_jacobian(pod_modes, dafoam_fom_states.value, dafoam_rom_states.value, weights, scaling_factors)
# J_rom_fd  = dafoam_rom._compute_reduced_jacobian(pod_modes, dafoam_fom_states.value, dafoam_rom_states.value, weights, scaling_factors, mode='fd')

# fig, axs = plt.subplots(2, 2)

# pclr = axs[0, 0].pcolor(J_rom_jvp)
# axs[0, 0].set_title('J_rom JVP')
# plt.colorbar(pclr, ax=axs[0, 0])

# pclr = axs[0, 1].pcolor(J_rom_fd)
# axs[0, 1].set_title('J_rom FD')
# plt.colorbar(pclr, ax=axs[0, 1])

# pclr = axs[1, 0].pcolor(J_rom_fd - J_rom_jvp)
# axs[1, 0].set_title('J_rom diff (FD - JVP)')
# plt.colorbar(pclr, ax=axs[1, 0])

# pclr = axs[1, 1].pcolor((J_rom_fd - J_rom_jvp)/J_rom_jvp, cmap=mpl.colormaps['seismic'], vmin=-0.1, vmax=0.1)
# axs[1, 1].set_title('J_rom rel diff (FD - JVP)/JVP')
# plt.colorbar(pclr, ax=axs[1, 1])

# input('Press ENTER to continue...')


# # Compare FOM and ROM initial solution
# plt.figure()
# dafoam_state_diff    = (dafoam_solver_states - dafoam_fom_states)
# plt.scatter(np.array(range(dafoam_instance.getNLocalAdjointStates())), dafoam_state_diff.value, label='FOM-ROM')
# plt.scatter(np.array(range(dafoam_instance.getNLocalAdjointStates())), dafoam_fom_states.value, label='ROM predicted')
# plt.scatter(np.array(range(dafoam_instance.getNLocalAdjointStates())), dafoam_solver_states.value, label='FOM')
# # plt.ylim((-1,1))
# plt.legend()
# plt.show()
# input('Press ENTER to continue')


# DAFoamFunctions Explicit component setup and evaluation
dafoam_functions        = DAFoamFunctions(dafoam_instance, disable_jacvec_normalization=True)
dafoam_function_outputs = dafoam_functions.evaluate(dafoam_fom_states, 
                                                    dafoam_input_variables_group)


# region Optimization problem selection
# optimization_case options
# 1: Maximize CL/CD wrt angle-of-attack
# 2: Minimize CD wrt angle-of-attack, wing shape (thickness/camber ffd), constrained by CL=0.5
# 3: Minimize CL/CD wrt wing shape (thickness/camber ffd)
optimization_case = 3


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
    percent_change_in_thickness_dof.set_as_design_variable(lower=-8, upper=8, scaler=1./10)
    normalized_percent_camber_change_dof.set_as_design_variable(lower=-8, upper=8, scaler=1./10)

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

# Can set design variables here and run sim to test
# sim[root_twist]  = 3*3.14159/180
# sim.run()
# dafoam_instance.solver.writeAdjointFields('sol', 11001, np.ascontiguousarray(dafoam_fom_states.value))

# dafoam_instance()

# dafoam_instance.solver.writeAdjointFields('sol', 11002, np.ascontiguousarray(dafoam_instance.getStates()))


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


# OpenSQP optimizer setup
open_sqp_options = {'maxiter': 20,
                    'readable_outputs': ['x']}
optimizer   = OpenSQP(prob, **open_sqp_options)
optimizer.solve()























# # indices1 = np.concatenate((1*np.ones(n_cells, ), 2*np.ones(n_cells, ), 3*np.ones(n_cells, ), 4*np.ones(n_cells, ), 5*np.ones(n_cells, ), 6*np.ones(n_cells, ), 7*np.ones_like(face_areas)))
# # dafoam_instance.solver.writeAdjointFields('indices', 8888, np.ascontiguousarray(indices1))





# plt.figure()
# n_cells          = dafoam_instance.solver.getNLocalCells()
# state_var_partition_lines = n_cells*np.array(range(1, 7))
# for i in range(6):
#     plt.axvline(x=state_var_partition_lines[i], color='gray', linestyle='--', alpha=0.5)
# plt.plot(pod_modes[:, 5:None:-1], linestyle='--')
# plt.show()


# # Some plotting
# n_cells          = dafoam_instance.solver.getNLocalCells()
# state_var_partition_lines = n_cells*np.array(range(1, 7))

# plt.figure(1) #Plotting raw stored data
# for i in range(6):
#     plt.axvline(x=state_var_partition_lines[i], color='gray', linestyle='--', alpha=0.5)
# plt.plot(state_store)
# plt.title('Raw training data [u,v,w,p,T,nuTilda,phi]')
# plt.xlabel('Index')
# plt.ylabel('Value')

# plt.figure(2) #Plotting different reference vectors
# for i in range(6):
#     plt.axvline(x=state_var_partition_lines[i], color='gray', linestyle='--', alpha=0.5)
# plt.plot(scale_factors,     label='Scale factors')
# plt.plot(state_store_mean,  label='Data mean')
# plt.plot(stat_scaling,      label='Stat scaling')
# plt.title('Reference and normalization vectors')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.legend()
# plt.show(block=False)

# plt.figure(3) #Plotting data scaled by scaling factor
# for i in range(6):
#     plt.axvline(x=state_var_partition_lines[i], color='gray', linestyle='--', alpha=0.5)
# plt.plot((state_store - state_store_mean[:, np.newaxis])/scale_factors[:, np.newaxis])
# plt.title('Mean centered training data scaled by scaling factor')
# plt.xlabel('Index')
# plt.ylabel('Value')

# # plt.figure(4) #Plotting data scaled by abs(mean) + 1
# # for i in range(6):
# #     plt.axvline(x=state_var_partition_lines[i], color='gray', linestyle='--', alpha=0.5)
# # plt.plot((state_store - state_store_mean[:, np.newaxis])/(np.abs(state_store_mean[:, np.newaxis]) + 1))
# # plt.title('Mean centered training data scaled by abs(mean) + 1')
# # plt.xlabel('Index')
# # plt.ylabel('Value')
# # plt.show()

# plt.figure(5) #Plotting data scaled by stat_scaling
# for i in range(6):
#     plt.axvline(x=state_var_partition_lines[i], color='gray', linestyle='--', alpha=0.5)
# plt.plot((state_store - state_store_mean[:, np.newaxis])/stat_scaling[:, np.newaxis])
# plt.title('Mean centered training data scaled by standard deviation')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.show()

# z = (state_store - state_store_mean[:, None])/scale_factors[:, None]

# # Compute our modes
# u, s, vh  = np.linalg.svd(z, full_matrices=False)
# pod_modes = u[:, :num_modes]

# print(f'phi.T@phi: {pod_modes.T@pod_modes}')


# plt.figure(6) #Plotting pod modes
# for i in range(6):
#     plt.axvline(x=state_var_partition_lines[i], color='gray', linestyle='--', alpha=0.5)
# plt.plot(pod_modes)
# plt.show()


# # Initialize pyOFM
# current_dir = os.getcwd()
# os.chdir(dafoam_directory)
# for i in range(num_modes):
#     ofm.writeField(f'U_mode_{i}',       'volVectorField', np.ascontiguousarray(pod_modes[0*n_cells:3*n_cells, i]))
#     ofm.writeField(f'p_mode_{i}',       'volScalarField', np.ascontiguousarray(pod_modes[3*n_cells:4*n_cells, i]))
#     ofm.writeField(f'T_mode_{i}',       'volScalarField', np.ascontiguousarray(pod_modes[4*n_cells:5*n_cells, i]))
#     ofm.writeField(f'nuTilda_mode_{i}', 'volScalarField', np.ascontiguousarray(pod_modes[5*n_cells:6*n_cells, i]))
#     os.rename(f'./0/U_mode_{i}.gz',       f'./10000/U_mode{i}.gz')
#     os.rename(f'./0/p_mode_{i}.gz',       f'./10000/p_mode{i}.gz')
#     os.rename(f'./0/T_mode_{i}.gz',       f'./10000/T_mode{i}.gz')
#     os.rename(f'./0/nuTilda_mode_{i}.gz', f'./10000/nuTilda_mode{i}.gz')
# os.chdir(current_dir)




# svals = s  # rename as appropriate
# cum_energy = np.cumsum(svals**2)
# total = cum_energy[-1]
# energy_fraction = cum_energy / total
# for k in [5,10,20,50,100]:
#     if k <= len(svals):
#         print(f"m={k:3d}  energy captured = {energy_fraction[k-1]:.6f}")
# # minimal m for target energy
# target = 0.999  # e.g. 99.9%
# m_req = np.searchsorted(energy_fraction, target) + 1
# print("m required for", target, "energy =", m_req)








# # DAFoamSolver Implicit component setup and evaluation
# dafoam_rom           = DAFoamROM(dafoam_instance, phi=pod_modes, fom_states_ref=state_store_mean, max_iters=1000, scaling_factors=scale_factors)
# dafoam_rom_states    = dafoam_rom.evaluate(dafoam_input_variables_group)

# print(f'dafoam_rom_states: {dafoam_rom_states.value}')

# dafoam_fom_states    = csdl.matvec(pod_modes, dafoam_rom_states)

# # DAFoamFunctions Explicit component setup and evaluation
# dafoam_functions = DAFoamFunctions(dafoam_instance)
# dafoam_function_outputs = dafoam_functions.evaluate(dafoam_fom_states, 
#                                                     dafoam_input_variables_group)



# recorder.stop()














# # Write modes to file for visualization
# current_dir = os.getcwd()
# os.chdir(dafoam_directory)
# base_index = 5000
# n_cells = dafoam_instance.solver.getNLocalCells()
# for i in range(num_modes):
#     try:
#         os.mkdir(f'./{base_index + i}')
#     except:
#         None
#     ofm.writeField(f'U_mode',       'volVectorField', np.ascontiguousarray(pod_modes[0*n_cells:3*n_cells, i]))
#     ofm.writeField(f'p_mode',       'volScalarField', np.ascontiguousarray(pod_modes[3*n_cells:4*n_cells, i]))
#     ofm.writeField(f'T_mode',       'volScalarField', np.ascontiguousarray(pod_modes[4*n_cells:5*n_cells, i]))
#     ofm.writeField(f'nuTilda_mode', 'volScalarField', np.ascontiguousarray(pod_modes[5*n_cells:6*n_cells, i]))
#     os.rename(f'./0/U_mode.gz',       f'./{base_index + i}/U_mode.gz')
#     os.rename(f'./0/p_mode.gz',       f'./{base_index + i}/p_mode.gz')
#     os.rename(f'./0/T_mode.gz',       f'./{base_index + i}/T_mode.gz')
#     os.rename(f'./0/nuTilda_mode.gz', f'./{base_index + i}/nuTilda_mode.gz')
# os.chdir(current_dir)

# os.chdir(dafoam_directory)
# base_index = 6000
# n_cells = dafoam_instance.solver.getNLocalCells()
# for i in range(num_modes):
#     try:
#         os.mkdir(f'./{base_index + i}')
#     except:
#         None
#     ofm.writeField(f'U_temp',       'volVectorField', np.ascontiguousarray(y_training[0*n_cells:3*n_cells, i]))
#     ofm.writeField(f'p_temp',       'volScalarField', np.ascontiguousarray(y_training[3*n_cells:4*n_cells, i]))
#     ofm.writeField(f'T_temp',       'volScalarField', np.ascontiguousarray(y_training[4*n_cells:5*n_cells, i]))
#     ofm.writeField(f'nuTilda_temp', 'volScalarField', np.ascontiguousarray(y_training[5*n_cells:6*n_cells, i]))
#     os.rename(f'./0/U_temp.gz',       f'./{base_index + i}/U_temp.gz')
#     os.rename(f'./0/p_temp.gz',       f'./{base_index + i}/p_temp.gz')
#     os.rename(f'./0/T_temp.gz',       f'./{base_index + i}/T_temp.gz')
#     os.rename(f'./0/nuTilda_temp.gz', f'./{base_index + i}/nuTilda_temp.gz')
# os.chdir(current_dir)

# base_index = 7000
# for i in range(num_modes):
#     dafoam_instance.solver.writeAdjointFields('mode', base_index + i, np.ascontiguousarray(pod_modes[:, i]))

# os.chdir(dafoam_directory)
# base_index = 8000
# n_cells = dafoam_instance.solver.getNLocalCells()
# for i in range(num_modes):
#     try:
#         os.mkdir(f'./{base_index + i}')
#     except:
#         None
#     ofm.writeField(f'z_U',       'volVectorField', np.ascontiguousarray(z[0*n_cells:3*n_cells, i]))
#     ofm.writeField(f'z_p',       'volScalarField', np.ascontiguousarray(z[3*n_cells:4*n_cells, i]))
#     ofm.writeField(f'z_T',       'volScalarField', np.ascontiguousarray(z[4*n_cells:5*n_cells, i]))
#     ofm.writeField(f'z_nuTilda', 'volScalarField', np.ascontiguousarray(z[5*n_cells:6*n_cells, i]))
#     os.rename(f'./0/z_U.gz',       f'./{base_index + i}/z_U.gz')
#     os.rename(f'./0/z_p.gz',       f'./{base_index + i}/z_p.gz')
#     os.rename(f'./0/z_T.gz',       f'./{base_index + i}/z_T.gz')
#     os.rename(f'./0/z_nuTilda.gz', f'./{base_index + i}/z_nuTilda.gz')
# os.chdir(current_dir)

# os.chdir(dafoam_directory)
# base_index = 9000
# n_cells = dafoam_instance.solver.getNLocalCells()
# for i in range(1):
#     try:
#         os.mkdir(f'./{base_index + i}')
#     except:
#         None
#     ofm.writeField(f'U_weights',         'volVectorField', np.ascontiguousarray(weights[0*n_cells:3*n_cells]))
#     ofm.writeField(f'p_weights',         'volScalarField', np.ascontiguousarray(weights[3*n_cells:4*n_cells]))
#     ofm.writeField(f'T_weights',         'volScalarField', np.ascontiguousarray(weights[4*n_cells:5*n_cells]))
#     ofm.writeField(f'nuTilda_weights',   'volScalarField', np.ascontiguousarray(weights[5*n_cells:6*n_cells]))
#     os.rename(f'./0/U_weights.gz',       f'./{base_index + i}/U_weights.gz')
#     os.rename(f'./0/p_weights.gz',       f'./{base_index + i}/p_weights.gz')
#     os.rename(f'./0/T_weights.gz',       f'./{base_index + i}/T_weights.gz')
#     os.rename(f'./0/nuTilda_weights.gz',       f'./{base_index + i}/nuTilda_weights.gz')

#     ofm.writeField(f'U_sqrt_weights',         'volVectorField', np.ascontiguousarray(np.sqrt(weights[0*n_cells:3*n_cells])))
#     ofm.writeField(f'p_sqrt_weights',         'volScalarField', np.ascontiguousarray(np.sqrt(weights[3*n_cells:4*n_cells])))
#     ofm.writeField(f'T_sqrt_weights',         'volScalarField', np.ascontiguousarray(np.sqrt(weights[4*n_cells:5*n_cells])))
#     ofm.writeField(f'nuTilda_sqrt_weights',   'volScalarField', np.ascontiguousarray(np.sqrt(weights[5*n_cells:6*n_cells])))
#     os.rename(f'./0/U_sqrt_weights.gz',       f'./{base_index + i}/U_sqrt_weights.gz')
#     os.rename(f'./0/p_sqrt_weights.gz',       f'./{base_index + i}/p_sqrt_weights.gz')
#     os.rename(f'./0/T_sqrt_weights.gz',       f'./{base_index + i}/T_sqrt_weights.gz')
#     os.rename(f'./0/nuTilda_sqrt_weights.gz',       f'./{base_index + i}/nuTilda_sqrt_weights.gz')

#     ofm.writeField(f'U_scale',         'volVectorField', np.ascontiguousarray(scaling_factors[0*n_cells:3*n_cells]))
#     ofm.writeField(f'p_scale',         'volScalarField', np.ascontiguousarray(scaling_factors[3*n_cells:4*n_cells]))
#     ofm.writeField(f'T_scale',         'volScalarField', np.ascontiguousarray(scaling_factors[4*n_cells:5*n_cells]))
#     ofm.writeField(f'nuTilda_scale',   'volScalarField', np.ascontiguousarray(scaling_factors[5*n_cells:6*n_cells]))
#     os.rename(f'./0/U_scale.gz',       f'./{base_index + i}/U_scale.gz')
#     os.rename(f'./0/p_scale.gz',       f'./{base_index + i}/p_scale.gz')
#     os.rename(f'./0/T_scale.gz',       f'./{base_index + i}/T_scale.gz')
#     os.rename(f'./0/nuTilda_scale.gz',       f'./{base_index + i}/nuTilda_scale.gz')

#     ofm.writeField(f'U_ref',         'volVectorField', np.ascontiguousarray(y_reference[0*n_cells:3*n_cells]))
#     ofm.writeField(f'p_ref',         'volScalarField', np.ascontiguousarray(y_reference[3*n_cells:4*n_cells]))
#     ofm.writeField(f'T_ref',         'volScalarField', np.ascontiguousarray(y_reference[4*n_cells:5*n_cells]))
#     ofm.writeField(f'nuTilda_ref',   'volScalarField', np.ascontiguousarray(y_reference[5*n_cells:6*n_cells]))
#     os.rename(f'./0/U_ref.gz',       f'./{base_index + i}/U_ref.gz')
#     os.rename(f'./0/p_ref.gz',       f'./{base_index + i}/p_ref.gz')
#     os.rename(f'./0/T_ref.gz',       f'./{base_index + i}/T_ref.gz')
#     os.rename(f'./0/nuTilda_ref.gz',       f'./{base_index + i}/nuTilda_ref.gz')
# os.chdir(current_dir)

# os.chdir(dafoam_directory)
# base_index = 10000
# n_cells = dafoam_instance.solver.getNLocalCells()
# for i in range(num_modes):
#     try:
#         os.mkdir(f'./{base_index + i}')
#     except:
#         None
#     ofm.writeField(f'U_centered',       'volVectorField', np.ascontiguousarray(y_training[0*n_cells:3*n_cells, i] - y_reference[0*n_cells:3*n_cells]))
#     ofm.writeField(f'p_centered',       'volScalarField', np.ascontiguousarray(y_training[3*n_cells:4*n_cells, i] - y_reference[3*n_cells:4*n_cells]))
#     ofm.writeField(f'T_centered',       'volScalarField', np.ascontiguousarray(y_training[4*n_cells:5*n_cells, i] - y_reference[4*n_cells:5*n_cells]))
#     ofm.writeField(f'nuTilda_centered', 'volScalarField', np.ascontiguousarray(y_training[5*n_cells:6*n_cells, i] - y_reference[5*n_cells:6*n_cells]))
#     os.rename(f'./0/U_centered.gz',       f'./{base_index + i}/U_centered.gz')
#     os.rename(f'./0/p_centered.gz',       f'./{base_index + i}/p_centered.gz')
#     os.rename(f'./0/T_centered.gz',       f'./{base_index + i}/T_centered.gz')
#     os.rename(f'./0/nuTilda_centered.gz', f'./{base_index + i}/nuTilda_centered.gz')
# os.chdir(current_dir)

# base_index = 11000
# for i in range(num_modes):
#     dafoam_instance.solver.writeAdjointFields('sol', base_index + i, np.ascontiguousarray(local_data["snapshots_local"][:, i]))
