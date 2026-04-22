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
from csdl_dafoam.core.rom.csdl_rom import CSDLROMWrapper
from csdl_dafoam.core.rom.rom_models import DAFoamLSPGModel, DAFoamGalerkinModel
from csdl_dafoam.core.rom.rom_solver import NewtonSolver
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
problem_name              = 'rom_with_interpolation'

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
h5_file_base_name     = "point"


# ===============================
# region SETUP
# ===============================
# MPI information
rank      = comm.Get_rank()
comm_size = comm.Get_size()
rank_str  = f"{rank:0{len(str(comm_size-1))}d}" # string with zero-padded rank index (for prints)


# region DAFoam instance
dafoam_instance             = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)
x_surf_dafoam_initial_mpi   = dafoam_instance.getSurfaceCoordinates()
x_vol_dafoam_initial_mpi    = dafoam_instance.xv0

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
            projected_surf_mesh_dafoam = geometry.project(
                x_surf_dafoam_initial, 
                grid_search_density_parameter = 1,      # 1     (ORIGINAL)
                projection_tolerance          = 1e-10,   #1.e-3m (ORIGINAL)
                grid_search_density_cutoff    = 50,     # 20    (ORIGINAL) 50
                force_reprojection            = False,
                plot                          = False    # UCSD_LAB
            )

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
percent_change_in_thickness_dof      = csdl.Variable(shape=(num_ffd_coefficients_chordwise-2,), value=5*np.array([0, 0, 0]), name="normalized_thickness_dof") 
normalized_percent_camber_change     = csdl.Variable(shape=(num_ffd_coefficients_chordwise, num_ffd_sections),  value=0.)
normalized_percent_camber_change_dof = csdl.Variable(shape=(num_ffd_coefficients_chordwise-2,), value=5*np.array([0, 0, 0]), name="normalized_camber_dof")

# ffd_block.plot()
ffd_sectional_parameterization = VolumeSectionalParameterization(
    name="ffd_sectional_parameterization",
    parameterized_points=ffd_block.coefficients, #ffd_block.coefficients.shape = (5, 2, 2, 3)
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
flight_conditions_group.angle_of_attack_deg = csdl.Variable(value=3.60663577576718, name="angle_of_attack_deg")
flight_conditions_group.altitude_m      = csdl.Variable(value=12922.086908545323, name="altitude (m)")

# Atmospheric condition variables
ambient_conditions_group = sam.compute_ambient_conditions_group(flight_conditions_group.altitude_m)


#===============================================================================================================================================
# IN CONSTRUCTION
#===============================================================================================================================================
# region MODE IMPORT AND SETUP
import glob

# Setup interface
data_interface = TrainingDataInterface(dafoam_instance=dafoam_instance, 
                                        storage_location=storage_location, 
                                        dataset_keyword=dataset_keyword,
                                        h5_file_base_name=h5_file_base_name)
state_info = data_interface.state_info

# Find files
files = glob.glob(str(Path(storage_location)/dataset_keyword/f"{h5_file_base_name}*.h5"))

# Primary variable names (these should be the names found in the files)
primary_var_names = ["angle_of_attack_deg", "altitude_m"]

# Setup the value array corresponding to the primary variables
# NOTE: We are assuming that the primary variables are scalars for now.
primary_var_array = np.zeros((len(files), len(primary_var_names)))

# Initialize our data arrays/lists
pod_mode_store        = []
scaling               = None
weights               = None
reference_state_store = []

# modes to retain
variance = 0.9999

num_modes = 0
# Load our parameters and POD modes
for i, file in enumerate(files):
    print(f"Reading file : {file}") if rank == 0 else None
    parameter_data = data_interface.load_h5(file, "parameters")
    for j, primary_var in enumerate(primary_var_names):
        primary_var_array[i, j] = parameter_data["primary_variables"][primary_var].item()

    pod_data = data_interface.load_h5(file, "pod")
    this_mode_set        = np.array(np.concatenate([pod_data["modes"][state_name]            for state_name in state_info.keys()], axis=0))
    this_scaling         = np.array(np.concatenate([pod_data["scaling"][state_var] * np.ones((np.size(info["indices"]), )) for state_var, info in state_info.items()], axis=0))
    this_weights         = np.array(np.concatenate([pod_data["weights"][state_var]           for state_var in state_info.keys()], axis=0))
    this_reference_state = np.array(np.concatenate([pod_data["reference_state"][state_var]   for state_var in state_info.keys()], axis=0))
    this_singular_vals   = pod_data["singular_values"]

    def compute_k(S, energy_tol=0.9999):
        energy = S**2
        cumulative = np.cumsum(energy)
        total = cumulative[-1]
        ratio = cumulative / total
        k = np.searchsorted(ratio, energy_tol) + 1
        return k
    
    this_num_modes = compute_k(this_singular_vals, variance)
    num_modes = this_num_modes if this_num_modes > num_modes else num_modes

    pod_mode_store.append(this_mode_set)
    reference_state_store.append(this_reference_state)

    if scaling is None:
        scaling = this_scaling
    elif not np.array_equal(scaling, this_scaling):
        raise ValueError("Scaling values are not consistent among files. Might need to recompute POD?")
    
    if weights is None:
        weights = this_weights
    elif not np.array_equal(weights, this_weights):
        raise ValueError("Weights values are not consistent among files. Might need to recompute POD?")

for i, pod_modes in enumerate(pod_mode_store):
    pod_mode_store[i] = pod_modes[:, :num_modes]
    
residual_scaling = np.ones_like(dafoam_instance.getStateWeights())
residual_scaling[state_info["T"]["indices"]] *= 1005


from csdl_dafoam.core.rom.csdl_grassmann import Grassmann
# Compute our tangent vectors
m = comm.allreduce(dafoam_instance.getNLocalAdjointStates(), op=MPI.SUM)
manifold_local = Grassmann(m, num_modes, comm=comm, inner_product_weights=weights)

pod_modes_mean = manifold_local.karcher_mean(pod_mode_store[0], pod_mode_store)

log_maps = []
for pod_modes_i in pod_mode_store:
    log_map_i = manifold_local.log(pod_modes_mean, pod_modes_i)
    log_maps.append(log_map_i)

# Setup interpolation
from csdl_dafoam.utils.interpolation import RBFInterpolator

# Setup interpolation
query_point    = csdl.concatenate((flight_conditions_group.angle_of_attack_deg, flight_conditions_group.altitude_m))
interp_weights = RBFInterpolator(query_point=query_point, sample_points=primary_var_array).weights()

from csdl_dafoam.utils.runscript_helper_functions import global_local_op
log_map_interp         = global_local_op(interp_weights, np.array(log_maps), lambda x,y: csdl.einsum(x, y, action="i,ijk->jk"), comm=comm)
reference_state_interp = global_local_op(interp_weights, np.array(reference_state_store), lambda x,y: csdl.einsum(x, y, action="i,ij->j"), comm=comm)
pod_modes_interp       = manifold_local.exp(pod_modes_mean, log_map_interp)


#===============================================================================================================================================



with csdl.experimental.mpi.enter_mpi_region(rank, comm) as mpi_region:

    pod_modes_interp = mpi_region.split_custom(pod_modes_interp, split_func=lambda x:x)
    reference_state_interp = mpi_region.split_custom(reference_state_interp, split_func=lambda x:x)
    
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

    dafoam_rom_model = DAFoamLSPGModel(dafoam_input_variables_group=dafoam_input_variables_group,
                                 pod_modes=pod_modes_interp,
                                 reference_fom_state=reference_state_interp,
                                 scaling=scaling,
                                 weights=1 / residual_scaling ** 2,
                                 dafoam_instance=dafoam_instance,
                                 normalize_residuals=False,
                                 fd_step=1e-6)

    
    # dafoam_rom_model = DAFoamGalerkinModel(dafoam_input_variables_group=dafoam_input_variables_group,
    #                              pod_modes=pod_modes,
    #                              reference_fom_state=reference_state,
    #                              scaling=scaling,
    #                              weights=weights / residual_scaling,
    #                              dafoam_instance=dafoam_instance,
    #                              normalize_residuals=False,
    #                              fd_step=1e-6,
    #                              jac_mode="fd")

    dafoam_rom = CSDLROMWrapper(model=dafoam_rom_model, solver=NewtonSolver(options={"tol_rel":1e-9, "tol_step_abs":1e-13}))   
    dafoam_rom_states = dafoam_rom.evaluate()

    # Reconstruct state
    dafoam_state_estimate = reference_state_interp + scaling * (pod_modes_interp @ dafoam_rom_states)

    # DAFoamFunctions Explicit component setup and evaluation
    dafoam_functions = DAFoamFunctions(dafoam_instance, disable_jacvec_normalization=True)
    dafoam_function_outputs = dafoam_functions.evaluate(dafoam_state_estimate, 
                                                        dafoam_input_variables_group)

    outputDict = dafoam_instance.getOption("function")
    for outputName in outputDict.keys():
        mpi_region.set_as_global_output(getattr(dafoam_function_outputs, outputName))
    # mpi_region.set_as_global_output(dafoam_function_outputs.drag)


# region Optimization problem selection
# optimization_case options
# 1: Maximize CL/CD wrt angle-of-attack
# 2: Minimize CD wrt angle-of-attack, wing shape (thickness/camber ffd), constrained by CL=0.5
# 3: Maximize CL/CD wrt angle-of-attack, wing shape (thickness/camber ffd)
# 4: Minimize D wrt angle-of-attack (test case)
# 5: Maximize CL/CD wrt wing shape (thickness/camber ffd)
optimization_case = 1


if optimization_case == 1:
    # Declaring and naming some variables
    lift = dafoam_function_outputs.lift
    drag = dafoam_function_outputs.drag

    # Design variables
    flight_conditions_group.angle_of_attack_deg.set_as_design_variable(lower=0, upper=5, scaler=1./5)

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

    # Design variables
    percent_change_in_thickness_dof.set_as_design_variable(lower=-5, upper=5, scaler=1./5) #(lower=-10, upper=10, scaler=1./10)
    normalized_percent_camber_change_dof.set_as_design_variable(lower=-5, upper=5, scaler=1./5) #(lower=-10, upper=10, scaler=1./10)

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

# Quick write of the variable names to file
# write_dv_names(f"{problem_name}_outputs/design_variable_map.txt", sim)



# ===============================
# region OPTIMIZER
# ===============================
# Only allow visualization and modopt output files on the root rank
visualize_on_this_rank           = True  if rank == 0 and not is_headless() else False
turn_off_outputs_on_nonroot_rank = False if rank == 0 else True
recording_on_root_rank           = True  if rank == 0 else False
rank_outputs                     = ['x'] if rank == 0 else []

# Optimization solver setup and run
prob                = CSDLAlphaProblem(problem_name=f'{problem_name}', simulator=sim)

optimizer_choice    = 3 # Set to 1 for PySLSQP, 2 for OpenSQP, or 3 for InteriorPoint

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
    open_sqp_options = {'maxiter': 100,
                        'readable_outputs': rank_outputs,
                        'recording': recording_on_root_rank,
                        'ls_max_step': 1.,
                        'turn_off_outputs': turn_off_outputs_on_nonroot_rank,}
    optimizer = OpenSQP(prob, **open_sqp_options)
    optimizer.solve()
    optimizer.print_results()

elif optimizer_choice == 3:
    # InteriorPoint optimizer setup
    interior_point_options = {'maxiter': 100,
                            'readable_outputs': rank_outputs,
                            'recording': recording_on_root_rank,
                            'ls_max_step': 1.,
                            'turn_off_outputs': turn_off_outputs_on_nonroot_rank}
    optimizer   = InteriorPoint(prob, **interior_point_options)
    optimizer.solve()
    optimizer.print_results()
    
else:
    print(f'Check optimizer choice. {optimizer_choice} is not an option.')


# # ===============================
# # region COMPONENT TESTS
# # ===============================
# from csdl_dafoam.utils.csdl_test_functions import CustomComponentChecks
# import matplotlib.pyplot as plt

# component_testing = CustomComponentChecks(dafoam_rom, comm=comm)
# component_testing.run_inverse_jacobian_fd_sweep(eps_test_values=10. ** np.array(range(-2, -10, -1)))
# component_testing.run_jacvec_fd_sweep(eps_test_values=10. ** np.array(range(-10, -2)))
