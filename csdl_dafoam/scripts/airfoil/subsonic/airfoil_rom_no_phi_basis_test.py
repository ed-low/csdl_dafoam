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
import lsdo_geo

# LSDO_geo specific
from lsdo_geo.core.parameterization.free_form_deformation_functions import (
    construct_ffd_block_around_entities
)
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)

# IDWarp and DAFoam
from csdl_dafoam.core.csdl_idwarp import DAFoamMeshWarper
from csdl_dafoam.core.csdl_dafoam import instantiateDAFoam, DAFoamFunctions, DAFoamSolver, compute_dafoam_input_variables
import csdl_dafoam.utils.standard_atmosphere_model as sam
from csdl_dafoam.utils.runscript_helper_functions import *
from csdl_dafoam.utils.training_interface import TrainingDataInterface

# ROM
from csdl_dafoam.core.rom.csdl_rom import CSDLROMWrapper
from csdl_dafoam.core.rom.rom_models import PhiComputingLSPGModel
from csdl_dafoam.core.rom.rom_solver import BroydenNewtonSolver

#---- DEBUGGING TOOLS ----
import faulthandler
faulthandler.enable()
os.environ["PETSC_OPTIONS"] = "-malloc_debug"
#-------------------------


# ===============================
# region USER INPUT
# ===============================
problem_name              = 'training_data'

# Geometry
geometry_directory        = Path.cwd() / 'airfoil_geometry'
stp_file_name             = 'airfoil_transonic_unitspan_2.stp'
geometry_pickle_file_name = 'airfoil_stored_refit.pickle'

# MPI and timing
comm           = MPI.COMM_WORLD
TIMING_ENABLED = True

# DAFoam
dafoam_directory = Path.cwd() / 'results' / f'{problem_name}'

# Initial/reference values for DAFoam
U0       = 100.0
p0       = 101325.0
T0       = 300.0
nuTilda0 = 4.5e-5
A0       = 0.1
rho0     = p0 / T0 / 287

# Input parameters for DAFoam
da_options = {
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleCFoam",
    "primalMinResTol": 1.0e-8,
    "primalVarBounds": {"pMin": 5000, "rhoMin": 0.05},
    "primalBC": {
        "U0":       {"variable": "U",       "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0":       {"variable": "p",       "patches": ["inout"], "value": [p0]},
        "T0":       {"variable": "T",       "patches": ["inout"], "value": [T0]},
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
            "scale": 1.0,
        },
        "lift": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "normalToFlow",
            "patchVelocityInputName": "patch_velocity",
            "scale": 1.0,
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm", "useNonZeroInitGuess": False},
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

mesh_options = {
    "gridFile": str(dafoam_directory),
    "fileType": "OpenFOAM",
    "symmetryPlanes": [],
}

# ===============================
# region Training / ROM options
# ===============================
dataset_keyword  = 'training_set_with_perturbations_300'
storage_location = dafoam_directory
h5_path          = storage_location / dataset_keyword / "point_0.h5"

# ROM mode count (None = use all available in h5)
N_MODES = 30

# Test sweep
aoa_test_points = [0.0, 2.0, 4.0, 6.0]   # degrees


# ===============================
# region SETUP
# ===============================
rank      = comm.Get_rank()
comm_size = comm.Get_size()
rank_str  = f"{rank:0{len(str(comm_size-1))}d}"

dafoam_instance     = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)
dafoam_instance_rom = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)

x_surf_dafoam_initial_mpi = dafoam_instance.getSurfaceCoordinates()
x_vol_dafoam_initial_mpi  = dafoam_instance.xv0

(x_surf_dafoam_initial,
 x_surf_dafoam_initial_size,
 x_surf_dafoam_initial_indices) = gather_array_to_rank0(x_surf_dafoam_initial_mpi, comm)

if rank == 0:
    x_surf_hash = hash_array_tol(x_surf_dafoam_initial, tol=1e-8)
else:
    x_surf_hash = None
x_surf_hash = comm.bcast(x_surf_hash, root=0)

geometry_pickle_file_path         = Path(geometry_directory) / geometry_pickle_file_name
stp_file_path                     = Path(geometry_directory) / stp_file_name
surface_mesh_projection_file_path = Path(dafoam_directory) / f'projected_surface_mesh_{x_surf_hash}.pickle'


# ===============================
# region Training interface and split POD load
# ===============================
data_generator = TrainingDataInterface(
    dafoam_instance  = dafoam_instance,
    storage_location = storage_location,
    dataset_keyword  = dataset_keyword,
    h5_file_base_name= "point",
)


# ===============================
# region CSDL RECORDER
# ===============================
recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()

geometry = lsdo_geo.import_geometry(stp_file_path, parallelize=False)

# Surface mesh projection (cached to pickle)
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
                grid_search_density_parameter=1,
                projection_tolerance=1e-10,
                grid_search_density_cutoff=50,
                force_reprojection=False,
                plot=False
            )
            print('Writing surface mesh projection pickle...')
            write_simple_pickle(projected_surf_mesh_dafoam, surface_mesh_projection_file_path)
            print('Done!')
        except Exception as e:
            import traceback
            print(f"[Rank 0 ERROR] Projection/pickle step failed:\n{traceback.format_exc()}", flush=True)
            comm.Abort(1)

    comm.Barrier()
    if rank != 0:
        projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)

print(f'Rank {rank_str} done reading projected surface mesh!')
comm.Barrier()

# FFD parameterization
num_ffd_coefficients_chordwise = 5
num_ffd_sections               = 2
ffd_block = construct_ffd_block_around_entities(
    entities=geometry,
    num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2),
    degree=(3, 1, 1),
)

percent_change_in_thickness_dof      = csdl.Variable(shape=(num_ffd_coefficients_chordwise-2,), value=np.array([0, 0, 0]), name="percent_change_in_thickness_dof")
normalized_percent_camber_change_dof = csdl.Variable(shape=(num_ffd_coefficients_chordwise-2,), value=np.array([0, 0, 0]), name="normalized_percent_camber_change_dof")

percent_change_in_thickness      = csdl.Variable(shape=(num_ffd_coefficients_chordwise, num_ffd_sections), value=0.)
normalized_percent_camber_change = csdl.Variable(shape=(num_ffd_coefficients_chordwise, num_ffd_sections), value=0.)

ffd_sectional_parameterization = VolumeSectionalParameterization(
    name="ffd_sectional_parameterization",
    parameterized_points=ffd_block.coefficients,
    principal_parametric_dimension=1,
)

sectional_parameters = VolumeSectionalParameterizationInputs()
ffd_coefficients     = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

# Thickness
original_block_thickness    = ffd_block.coefficients.value[0, 0, 1, 2] - ffd_block.coefficients.value[0, 0, 0, 2]
percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[1:-1, 0], percent_change_in_thickness_dof)
percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[1:-1, 1], percent_change_in_thickness_dof)
delta_block_thickness       = (percent_change_in_thickness / 100) * original_block_thickness
ffd_coefficients = ffd_coefficients.set(csdl.slice[:, :, 1, 2], ffd_coefficients[:, :, 1, 2] + delta_block_thickness / 2)
ffd_coefficients = ffd_coefficients.set(csdl.slice[:, :, 0, 2], ffd_coefficients[:, :, 0, 2] - delta_block_thickness / 2)

# Camber
block_length                     = ffd_block.coefficients.value[1, 0, 0, 0] - ffd_block.coefficients.value[0, 0, 0, 0]
normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1, 0], normalized_percent_camber_change_dof)
normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1, 1], normalized_percent_camber_change_dof)
camber_change                    = (normalized_percent_camber_change / 100) * block_length
ffd_coefficients = ffd_coefficients.set(
    csdl.slice[:, :, :, 2],
    ffd_coefficients[:, :, :, 2] + csdl.expand(camber_change, (num_ffd_coefficients_chordwise, num_ffd_sections, 2), 'ij->ijk')
)

geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
geometry.set_coefficients(geometry_coefficients)

with Timer('evaluating geometry component', rank, TIMING_ENABLED):
    x_surf_dafoam_full = geometry.evaluate(projected_surf_mesh_dafoam, plot=False)

i0, i1 = x_surf_dafoam_initial_indices[rank]

# Flight conditions
flight_conditions_group                     = csdl.VariableGroup()
flight_conditions_group.airspeed_m_s        = csdl.Variable(value=U0, name="airspeed_m_s")
flight_conditions_group.angle_of_attack_deg = csdl.Variable(value=0,  name="angle_of_attack_deg")
flight_conditions_group.altitude_m          = csdl.Variable(value=0., name="altitude (m)")

ambient_conditions_group = sam.compute_ambient_conditions_group(flight_conditions_group.altitude_m)


# ===============================
# region POD load (no-phi basis)
# ===============================
pod_modes, reference_fom_state, weights, scaling = \
    data_generator.load_pod_modes(h5_path, n_modes=N_MODES)

if rank == 0:
    print(f"\n  Loaded POD (no phi):  modes {pod_modes.shape}")
    

with csdl.experimental.mpi.enter_mpi_region(rank, comm) as mpi_region:

    x_surf_dafoam = x_surf_dafoam_full[i0:i1, :].flatten()

    idwarp_model = DAFoamMeshWarper(dafoam_instance)
    x_vol_dafoam = idwarp_model.evaluate(x_surf_dafoam)

    flight_conditions_group.angle_of_attack_deg = mpi_region.split_custom(
        flight_conditions_group.angle_of_attack_deg, split_func=lambda x: x
    )

    dafoam_input_variables_group = compute_dafoam_input_variables(
        dafoam_instance, ambient_conditions_group, flight_conditions_group, x_vol_dafoam
    )

    # --- PhiComputingLSPGModel ROM ---
    rom_model = PhiComputingLSPGModel(
        dafoam_input_variables_group = dafoam_input_variables_group,
        pod_modes                    = pod_modes,
        reference_fom_state          = reference_fom_state,
        scaling                      = scaling,
        weights                      = weights,
        dafoam_instance              = dafoam_instance_rom,
        fd_step                      = 1e-6,
        disable_presolve_diagnostics = False,
    )
    rom_wrapper = CSDLROMWrapper(
        model                = rom_model,
        solver               = BroydenNewtonSolver(options={"tol_rel": 1e-9, "tol_step_abs": 1e-13}),
        start_with_zero_state= True,
    )
    rom_states = rom_wrapper.evaluate()

    # CSDL expression for the state: linear reconstruction used only for the computation graph.
    # phi DOFs will be overwritten at solve time via PhiComputingLSPGModel._reconstruct_fom_state.
    state_est = rom_model.reference_fom_state + rom_model.scaling * (rom_model.pod_modes @ rom_states)

    rom_fn_model   = DAFoamFunctions(dafoam_instance_rom, disable_jacvec_normalization=True)
    rom_fn_outputs = rom_fn_model.evaluate(state_est, dafoam_input_variables_group)

    # --- FOM (for comparison) ---
    dafoam_solver        = DAFoamSolver(dafoam_instance)
    dafoam_solver_states = dafoam_solver.evaluate(dafoam_input_variables_group)

    dafoam_fn_model   = DAFoamFunctions(dafoam_instance, disable_jacvec_normalization=True)
    dafoam_fn_outputs = dafoam_fn_model.evaluate(dafoam_solver_states, dafoam_input_variables_group)

    for out_name in dafoam_instance.getOption("function").keys():
        mpi_region.set_as_global_output(getattr(rom_fn_outputs,    out_name))
        mpi_region.set_as_global_output(getattr(dafoam_fn_outputs, out_name))

    mpi_region.set_as_global_output(state_est)
    mpi_region.set_as_global_output(dafoam_solver_states)

# Design variable for test sweep
flight_conditions_group.angle_of_attack_deg.set_as_design_variable(lower=0, upper=10, scaler=1./10)

objective_fun = -rom_fn_outputs.lift / rom_fn_outputs.drag
objective_fun.set_as_objective()
objective_fun.name = "-L/D_ROM"

recorder.stop()


# ===============================
# region SIMULATION AND TEST
# ===============================
sim        = csdl.experimental.PySimulator(recorder)
state_info = data_generator.state_info

if rank == 0:
    print(f"\n{'AoA':>6}  {'ROM drag':>12}  {'FOM drag':>12}  {'drag_rel_err':>14}  "
          f"{'ROM lift':>12}  {'FOM lift':>12}  {'lift_rel_err':>14}")

for aoa in aoa_test_points:
    sim[flight_conditions_group.angle_of_attack_deg] = aoa
    sim.run()

    rom_drag = rom_fn_outputs.drag.value
    fom_drag = dafoam_fn_outputs.drag.value
    rom_lift = rom_fn_outputs.lift.value
    fom_lift = dafoam_fn_outputs.lift.value

    if rank == 0:
        drag_rel = abs(rom_drag - fom_drag) / (abs(fom_drag) + 1e-300)
        lift_rel = abs(rom_lift - fom_lift) / (abs(fom_lift) + 1e-300)
        print(f"{aoa:>6.1f}  {rom_drag:>12.4e}  {fom_drag:>12.4e}  {drag_rel:>14.4e}  "
              f"{rom_lift:>12.4e}  {fom_lift:>12.4e}  {lift_rel:>14.4e}")

    # Per-variable state error using the correctly-reconstructed ROM state from DAFoam
    w_rom = dafoam_instance_rom.getStates()
    w_fom = dafoam_solver_states.value

    if rank == 0:
        print(f"  Per-variable relative L2 error (AoA={aoa}):")
    for var, info in state_info.items():
        idx  = info["indices"]
        err  = np.sqrt(comm.allreduce(np.sum((w_rom[idx] - w_fom[idx])**2), op=MPI.SUM))
        norm = np.sqrt(comm.allreduce(np.sum(w_fom[idx]**2),                op=MPI.SUM))
        if rank == 0:
            print(f"    {var:>10}: {err / (norm + 1e-300):.4e}")
