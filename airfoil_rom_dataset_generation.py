# ===============================
# region PACKAGES
# ===============================
import numpy as np
import sys
import os
import time
import pickle
from pathlib import Path
import shutil

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
from csdl_dafoam import instantiateDAFoam, DAFoamFunctions, DAFoamSolver, compute_dafoam_input_variables
import standard_atmosphere_model as sam

# BWB specific
from bwb_helper_functions import setup_geometry, read_geometry_pickle, write_geometry_pickle, gather_array_to_rank0, read_simple_pickle, write_simple_pickle

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
geometry_directory        =  Path.cwd()/'airfoil_geometry'
stp_file_name             = 'airfoil_transonic_unitspan_2.stp'
geometry_pickle_file_name = 'airfoil_stored_refit.pickle'

# MPI and timing
comm           = MPI.COMM_WORLD
TIMING_ENABLED = True  # True if we want timing printed for the CSDL operations

# DAFoam
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
    "primalVarBounds": {
        "pMin":   1000,
        "rhoMin": 0.05,
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
rank_str  = f"{rank:0{len(str(comm_size-1))}d}" # string with zero-padded rank index (for prints)


print(str(dafoam_directory))
print(dafoam_directory)

# region DAFoam instance
dafoam_instance         = instantiateDAFoam(da_options, comm, str(dafoam_directory), mesh_options)
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
    grid_search_density_cutoff    = 10,     
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
flight_conditions_group                     = csdl.VariableGroup()
flight_conditions_group.mach_number         = csdl.Variable(value=0.7, name="mach_number")
flight_conditions_group.angle_of_attack_deg = csdl.Variable(value=aoa0, name="angle_of_attack_deg")
flight_conditions_group.altitude_m          = csdl.Variable(value=0., name="altitude (m)")

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


recorder.stop()



# ===============================
# region SIM SETUP
# ===============================
sim = csdl.experimental.PySimulator(recorder)



# ===============================
# region TRAINING
# ===============================
from smt.sampling_methods import LHS
from rom_training_helper_functions import *


# region Options and user setup
# Storage options
dataset_keyword       = 'airfoil_training'
storage_location      = dafoam_directory

# Sampling options
# grassmann_variables indicates the variables which correspond to points on the Grassmann manifold
# snapshot_variables indicates the variables which correspond to "snapshots" or realizations
num_grassmann_samples     = 2
num_snapshot_samples      = 100
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
    flight_conditions_group.mach_number: {
        'name': 'mach_number',   
        'range': [0.65, 0.75],
    }, 
    flight_conditions_group.angle_of_attack_deg: {
        'name': 'angle_of_attack_deg',     
        'range': [0., 10],
    },
    flight_conditions_group.altitude_m: {
        'name': 'altitude_m', 
        'range': [7000., 13000],
        }
}

snapshot_vars_and_limits = {
    percent_change_in_thickness_dof: {
        'name': '%_thickness_change',
        'range': [-10, 10],
        'ref_value': 0, 
    },
    normalized_percent_camber_change_dof: {
        'name': '%_camber_change',
        'range': [-10, 10],
        'ref_value': 0, 
    }
}



# region Initialization
# Generate PETSc vector for state storage (will use this for writing to file)
petsc_states           = dafoam_instance.array2Vec(dafoam_instance.getStates())

# Make directory
os.makedirs(storage_location/dataset_keyword, exist_ok = True)


# Build limits for sampling
xlimits_grassmann, labels_grassmann, slicer_grassmann, shapes_grassmann = build_xlimits(grassmann_vars_and_limits)
xlimits_snapshot,  labels_snapshot,  slicer_snapshot,  shapes_snapshot  = build_xlimits(snapshot_vars_and_limits)

# Create LHS samplers and sample
lhs_grassmann         = LHS(xlimits=xlimits_grassmann, criterion='m', random_state=random_state_seed)
grassmann_raw_samples = lhs_grassmann(num_grassmann_samples)
grassmann_samples     = reshape_samples(grassmann_raw_samples, slicer_grassmann, shapes_grassmann)

lhs_snapshot          = LHS(xlimits=xlimits_snapshot,  criterion='m', random_state=random_state_seed)
snapshot_raw_samples  = lhs_snapshot(num_snapshot_samples)
snapshot_samples      = reshape_samples(snapshot_raw_samples, slicer_snapshot, shapes_snapshot)


# Print to console
print_sample_table(grassmann_vars_and_limits, grassmann_raw_samples)
print_sample_table(snapshot_vars_and_limits, snapshot_raw_samples)


# region Begin sampling
# Loop through each Grassmann point
for grassmann_index in range(len(grassmann_samples)):
    # TODO: (re)initialize solver with better initial condition for new grassmann point

    # Dataset file name (.h5 file)
    file_name = storage_location/dataset_keyword/f'{dataset_keyword}_point{grassmann_index}.h5'
    
    # Make a folder for the raw OpenFOAM save
    raw_directory = storage_location/dataset_keyword/f'{dataset_keyword}_point{grassmann_index}_raw'
    os.makedirs(raw_directory, exist_ok = True)

    # Copy the constant folder to the raw directory
    current_directory = Path.cwd()
    os.chdir(dafoam_directory)
    shutil.copytree('./constant', raw_directory/'constant')
    os.chdir(current_directory)

    # Update all of the grassmann parameters for current point
    for key in grassmann_samples[grassmann_index].keys():
            sim[key] = grassmann_samples[grassmann_index][key]

    # Loop through each snapshot configuration
    for snapshot_index in range(len(snapshot_samples)):
        print('\n\n\n\n')
        print('=============================================')
        print(f'Grassmann point {grassmann_index+1}/{num_grassmann_samples}, snapshot {snapshot_index+1}/{num_snapshot_samples}')
        print('=============================================\n')
        sim.run()
        
        # Update all of the snapshot parameters for current configuration
        for key in snapshot_samples[snapshot_index].keys():
            sim[key] = snapshot_samples[snapshot_index][key]

        # Update PETSc vector to most recent solution
        dafoam_instance.arrayVal2Vec(dafoam_instance.getStates(), petsc_states)

        write_snapshot(
            file_name,
            petsc_states,
            snapshot_index=snapshot_index,
            snapshot_configurations=snapshot_raw_samples,
            grassmann_configuration=grassmann_raw_samples[grassmann_index],
            snapshot_parameter_labels=None,
            grassmann_parameter_labels=None,
            converged=dafoam_solver.last_time_converged,
            reference_snapshot=False,
            comm=dafoam_instance.comm
        )

        # Move OpenFOAM solution to solution directory
        current_directory = Path.cwd()
        os.chdir(dafoam_directory)
        dafoam_instance.renameSolution(9998)
        shutil.move('./0.9998', 
                  raw_directory/f'{snapshot_index:04}')
        os.chdir(current_directory)

    print('\n\n\n\n')
    print('=============================================')
    print(f'Grassmann point {grassmann_index+1}/{num_grassmann_samples}, reference snapshot')
    print('=============================================\n')

    # Compute the reference state for this point on the manifold
    for key in snapshot_vars_and_limits.keys():
        sim[key] = snapshot_vars_and_limits[key]['ref_value']*np.ones(key.shape)
    
    sim.run()

    # Update PETSc vector to most recent solution
    dafoam_instance.arrayVal2Vec(dafoam_instance.getStates(), petsc_states)

    write_snapshot(
        file_name,
        petsc_states,
        snapshot_index=snapshot_index,
        snapshot_configurations=snapshot_raw_samples,
        grassmann_configuration=grassmann_raw_samples[grassmann_index],
        snapshot_parameter_labels=None,
        grassmann_parameter_labels=None,
        converged=dafoam_solver.last_time_converged,
        reference_snapshot=True,
        comm=dafoam_instance.comm
    )

    # Move OpenFOAM solution to solution directory
    current_directory = Path.cwd()
    os.chdir(dafoam_directory)
    dafoam_instance.renameSolution(9998)
    shutil.move('./0.9998', 
                raw_directory/f'snapshot_ref')
    os.chdir(current_directory)









# # Original case
# from smt.sampling_methods import LHS
# from rom_training_helper_functions import *

# sim = csdl.experimental.PySimulator(recorder)

# dataset_keyword       = 'airfoil'
# state_store_file_name = f'2gs_100ss_state_store_comm_size{comm_size}_rank{rank_str}.npy'
# random_state_seed     = 0


# # Generate PETSc vector for state storage (will use this for writing to file)
# petc_states           = dafoam_instance.array2Vec(dafoam_instance.getStates())

# if not Path(state_store_file_name).is_file():

#     # Names of variables for the parametric POD grouping
#     # grassmann_variables indicates the variables which correspond to points on the Grassmann manifold
#     # shapshot_variables indicates the variables which correspond to "snapshots" or realizations
#     num_grassmann_samples     = 2
#     num_snapshot_samples      = 100

#     grassmann_vars_and_limits = {
#         flight_conditions_group.mach_number: {
#             'name': 'mach_number',   
#             'range': [0.65, 0.75],
#         }, 
#         flight_conditions_group.angle_of_attack_deg: {
#             'name': 'angle_of_attack_deg',     
#             'range': [0., 10],
#         },
#         flight_conditions_group.altitude_m: {
#             'name': 'altitude_m', 
#             'range': [7000., 13000],
#             }
#     }

#     snapshot_vars_and_limits = {
#         percent_change_in_thickness_dof: {
#             'name': 'percent_change_in_thickness_dof',
#             'range': [-10, 10],
#             'ref_value': 0, 
#         },
#         normalized_percent_camber_change_dof: {
#             'name': 'normalized_percent_camber_change_dof',
#             'range': [-10, 10],
#             'ref_value': 0, 
#         }
#     }

#     print(f'percent_change_in_thickness_dof.value.shape: {percent_change_in_thickness_dof.value.shape}')
#     print(f'normalized_percent_camber_change_dof.value.shape: {normalized_percent_camber_change_dof.value.shape}')

#     # Build limits for sampling
#     xlimits_grassmann, labels_grassmann, slicer_grassmann, shapes_grassmann = build_xlimits(grassmann_vars_and_limits)
#     xlimits_snapshot,  labels_snapshot,  slicer_snapshot,  shapes_snapshot  = build_xlimits(snapshot_vars_and_limits)


#     # Create LHS samplers and sample
#     lhs_grassmann = LHS(xlimits=xlimits_grassmann, criterion='m', random_state=random_state_seed)
#     lhs_snapshot  = LHS(xlimits=xlimits_snapshot,  criterion='m', random_state=random_state_seed)

#     grassmann_raw_samples = lhs_grassmann(num_grassmann_samples)
#     snapshot_raw_samples  = lhs_snapshot(num_snapshot_samples)

#     print(grassmann_raw_samples)
#     print(snapshot_raw_samples)

#     grassmann_samples = reshape_samples(grassmann_raw_samples, slicer_grassmann, shapes_grassmann)
#     snapshot_samples  = reshape_samples(snapshot_raw_samples, slicer_snapshot, shapes_snapshot)

#     current_directory = Path.cwd()

#     state_store = np.zeros((dafoam_solver.num_local_state_elements, num_grassmann_samples*num_snapshot_samples))
#     converged   = np.zeros((num_grassmann_samples, num_snapshot_samples))

#     for grassmann_index in range(len(grassmann_samples)):

#         for key in grassmann_samples[grassmann_index].keys():
#                 sim[key] = grassmann_samples[grassmann_index][key]

#         for snapshot_index in range(len(snapshot_samples)):
            
#             for key in snapshot_samples[snapshot_index].keys():
#                 sim[key] = snapshot_samples[snapshot_index][key]

#             sim.run()

#             print(sim[ambient_conditions_group.rho_kg_m3])

#             # Write mesh to disk
#             os.chdir(dafoam_directory)
#             dafoam_instance.renameSolution((grassmann_index + 1)*100. + snapshot_index + 1)
#             os.chdir(current_directory)

#             state_store[:, snapshot_index + grassmann_index*num_snapshot_samples] = dafoam_instance.getStates()
#             if dafoam_solver.last_time_converged:
#                 converged[grassmann_index, snapshot_index] = 1

#     print(converged)
#     np.save(current_directory/state_store_file_name, state_store)

# else:
#     current_directory = Path.cwd()
#     state_store = np.load(current_directory/state_store_file_name)
#     print(state_store)
#     plt.plot(state_store[:, range(50)])
#     plt.show()



