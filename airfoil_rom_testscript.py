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
geometry_directory        =  os.path.join(os.getcwd(), 'airfoil_geometry/')
stp_file_name             = 'airfoil_transonic_unitspan_2.stp'
geometry_pickle_file_name = 'airfoil_stored_refit.pickle'

# MPI and timing
comm           = MPI.COMM_WORLD
TIMING_ENABLED = True  # True if we want timing printed for the CSDL operations

# DAFoam
dafoam_directory = os.path.join(os.getcwd(), 'openfoam_airfoil/')

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




# ===============================
# region CSDL RECORDER
# ===============================
# recorder 
recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()

# Create a volume coordinate variable
initial_mesh_coords = dafoam_instance.xv0
x_vol_dafoam        = csdl.Variable(value=initial_mesh_coords.ravel(order="C"), name='aero_vol_coords')

n_global_states     = dafoam_instance.getStates().shape[0]

# Compute dummy PHI for now
r           = 10     # Reduced dimension
phi_full, _ = np.linalg.qr(np.random.rand(n_global_states, r)) # Create a random orthonormal basis


fom_states_ref  = np.zeros_like(dafoam_instance.getStates())


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

# # DAFoamROM Implicit component setup and evaluation
# from csdl_dafoam import DAFoamROM
# dafoam_rom           = DAFoamROM(dafoam_instance, phi=phi_full, fom_states_ref=np.zeros_like(dafoam_instance.getStates()))
# dafoam_rom_states    = dafoam_rom.evaluate(dafoam_input_variables_group)

# DAFoamSolver Implicit component setup and evaluation
dafoam_rom           = DAFoamROM(dafoam_instance, phi=phi_full, fom_states_ref=fom_states_ref)
dafoam_rom_states    = dafoam_rom.evaluate(dafoam_input_variables_group)

print(f'dafoam_rom_states: {dafoam_rom_states.value}')

dafoam_fom_states    = csdl.matvec(phi_full, dafoam_rom_states)

# DAFoamFunctions Explicit component setup and evaluation
dafoam_functions = DAFoamFunctions(dafoam_instance)
dafoam_function_outputs = dafoam_functions.evaluate(dafoam_fom_states, 
                                                    dafoam_input_variables_group)



recorder.stop()