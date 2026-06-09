from __future__ import annotations
import numpy as np
import shutil
import gzip
import h5py
import os
from typing import Dict, Any, List
from smt.sampling_methods import LHS
from pathlib import Path
from mpi4py import MPI
from vedo import Arrows, Points, Plotter, Text2D
from csdl_dafoam.utils.runscript_helper_functions import quiet_barrier
from csdl_dafoam.utils.decompositions import method_of_snapshots_distributed
from csdl_dafoam.core.rom.rom_models import NormalizationConfig
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from warnings import warn
from urllib.parse import quote, unquote

from typing import TYPE_CHECKING, Dict, List, Tuple, Any
if TYPE_CHECKING:
    from dafoam import PYDAFOAM
    from csdl_alpha import Variable, experimental


# region TRAININGDATAINTERFACE
class TrainingDataInterface():  
    def __init__(self,
                 dafoam_instance:PYDAFOAM,
                 storage_location:str,
                 dataset_keyword:str,
                 primary_variables:Dict[Variable, Dict[str, Any]]=None,
                 secondary_variables:Dict[Variable, Dict[str, Any]]=None,
                 non_sampled_variables:List[Variable]=None,
                 csdl_simulator:experimental.PySimulator=None,
                 reference_patch:str=None,
                 num_primary_samples:int=2,
                 num_secondary_samples:int=20,
                 random_state_seed:float=0,
                 store_residuals:bool=False,
                 h5_file_base_name:str="point",
                 gather_raw_files:bool=True,
                 parallel_write:bool=True,
                 parallel_read:bool=True,
                 ):
        
        # TODO: See if there is DAFoam API to get whether a variable is volVectorStates, volScalarStates, modelStates, or surfaceScalarStates.
        # We'll just define here for the rhoSimpleCFoam case for now. Can add to here if necessary for other models
        self.solver_variable_storage_type = {"centroid_coordinates":    "volVectorStates", # These first 3 entries are for cell/face information, but they match
                                             "cell_volumes":            "volScalarStates", # the given state ordering
                                             "face_areas":              "surfaceScalarStates",
                                             "U":                       "volVectorStates",
                                             "p":                       "volScalarStates",
                                             "T":                       "volScalarStates",
                                             "nuTilda":                 "modelStates",
                                             "phi":                     "surfaceScalarStates"}
                
        self.dafoam_instance            = dafoam_instance
        self.csdl_simulator             = csdl_simulator
        self.reference_patch            = reference_patch
        self.primary_variables          = primary_variables
        self.secondary_variables        = secondary_variables
        self.non_sampled_variables      = non_sampled_variables
        self.storage_location           = Path(storage_location)
        self.dataset_keyword            = dataset_keyword
        self.num_primary_samples        = num_primary_samples
        self.num_secondary_samples      = num_secondary_samples
        self.store_residuals            = store_residuals
        self.random_state_seed          = random_state_seed
        self.h5_file_base_name          = h5_file_base_name
        self.gather_raw_files           = gather_raw_files
        self.parallel_write             = parallel_write
        self.parallel_read              = parallel_read

        # MPI values for easier access
        self.comm                       = dafoam_instance.comm
        self.rank                       = dafoam_instance.comm.rank
        self.comm_size                  = self.comm.Get_size()

        # Get sizes
        self.num_state_elements         = dafoam_instance.getNLocalAdjointStates()
        self.num_cells                  = dafoam_instance.solver.getNLocalCells()
        self.num_faces                  = dafoam_instance.solver.getNLocalFaces()
        self.num_primary_variables      = None if primary_variables is None else len(primary_variables)
        self.num_secondary_variables    = None if secondary_variables is None else len(secondary_variables)

        self._setup_indices_state_info_and_global_counts()

        self.ran_sampling               = False

        self._validate_variable_names()

        # Create directory
        self.print0('Creating storage directory...')
        if self.rank == 0:
            os.makedirs(storage_location/dataset_keyword, exist_ok = True)

        # Get objective variables if there are any
        self.objectives = list(csdl_simulator.recorder.objectives.keys()) if csdl_simulator is not None else None


    # region sample_variables
    def sample_variables(self, print_sampled_values:bool=True, random_state_seed:float=None):
        if random_state_seed is None:
            random_state_seed = self.random_state_seed

        primary_has_ref     = self._generate_lhs_samples(self.primary_variables,  
                                                         self.num_primary_samples,   
                                                         random_state_seed)
        
        secondary_has_ref   = self._generate_lhs_samples(self.secondary_variables, 
                                                         self.num_secondary_samples, 
                                                         random_state_seed)

        self.primary_has_ref    = primary_has_ref
        self.secondary_has_ref  = secondary_has_ref
        self.ran_sampling       = True

        # TODO: Print out results to console and/or to file


    # region write_sampling_results
    def write_sampling_results(self):
        self.print0("Not yet implemented")


    # region run_sweep
    def run_sweep(self, compute_pod:bool=True, compute_objective_grad:bool=False, separate_pod_file:bool=False, pod_options:Dict=None,
                  compute_perturbations:bool=False, perturbation_epsilon:float|Dict[Variable, float]=1e-4,
                  perturb_primary:bool=True, perturb_secondary:bool=True,
                  compute_split_pod:bool=False, split_pod_options:Dict=None):
        sim                         = self.csdl_simulator
        rank                        = self.rank
        comm_size                   = self.comm_size
        dafoam_instance             = self.dafoam_instance
        primary_variables           = self.primary_variables
        secondary_variables         = self.secondary_variables
        dafoam_directory            = Path(dafoam_instance.run_directory)

        # Default POD options
        default_pod_options = {"inner_product":"reference",
                               "centering":"reference",
                               "scaling":"reference",
                               "new_h5_file":separate_pod_file}

        # Assign default pod options to user supplied if not present
        pod_options = {} if pod_options is None else pod_options
        for key in default_pod_options.keys():
            if key not in pod_options.keys():
                pod_options[key] = default_pod_options[key]

        if sim is None:
            self.print0('Must supply the associated CSDL simulator to the TrainingInterface before calling run_sweep.')
            return

        # Throw error if LHS hasn't been run yet
        if not self.ran_sampling:
            self.print0("Must run sample_variables before run_sweep.")
            return

        # Throw error if we don't have a reference patch
        if self.reference_patch is None:
            self.print0("Please supply reference_patch to the TrainingDataInterface before running a sweep.")
            return

        # Adjust sample size to accomodate refrence values if present
        adj_num_primary_samples   = self.num_primary_samples   + self.primary_has_ref
        adj_num_secondary_samples = self.num_secondary_samples + self.secondary_has_ref

        # Build the ordered list of (var, info) pairs whose DoFs will be perturbed
        perturbation_dvs = []
        if compute_perturbations:
            if perturb_primary and primary_variables:
                perturbation_dvs.extend(primary_variables.items())
            if perturb_secondary and secondary_variables:
                perturbation_dvs.extend(secondary_variables.items())

        # Check for an interrupted sweep to resume from
        resume_primary_idx, base_secondary_start, skip_h5_init, pert_secondary_start = \
            self._check_for_interrupted_sweep(adj_num_secondary_samples, compute_perturbations)

        # Actual loop start for the resumed primary (may be lower than base_secondary_start
        # when perturbations are behind and need to catch up)
        if compute_perturbations and pert_secondary_start is not None:
            actual_resume_secondary_start = min(base_secondary_start, pert_secondary_start)
        else:
            actual_resume_secondary_start = base_secondary_start

        for primary_idx in range(adj_num_primary_samples):
            h5file_path = self.storage_location/self.dataset_keyword/f'{self.h5_file_base_name}_{primary_idx}.h5'

            # Skip primary indices already fully written before the resume point
            if resume_primary_idx is not None and primary_idx < resume_primary_idx:
                self.print0(f'Skipping primary index {primary_idx} (already complete).')
                continue

            # is_first_resumed: True only for the very first primary index we actually process
            is_first_resumed      = (resume_primary_idx is not None and primary_idx == resume_primary_idx)
            secondary_start       = actual_resume_secondary_start if is_first_resumed else 0
            # base_catchup_end: secondary indices below this were already base-written; re-run
            # the solve to restore DAFoam state but skip write/raw-file/grad operations.
            base_catchup_end      = base_secondary_start if is_first_resumed else 0
            pert_start_this_primary = (pert_secondary_start if pert_secondary_start is not None else 0) \
                                      if (is_first_resumed and compute_perturbations) else 0

            # skip_h5_init is True only when resuming mid-file (file already exists on disk).
            # When the previous primary completed fully and we're starting a fresh file for the
            # next primary, skip_h5_init is False and the file must be initialised normally.
            if not (is_first_resumed and skip_h5_init):
                self.initialize_h5_file(h5file_path, primary_idx, compute_objective_grad=compute_objective_grad,
                                        perturbation_dvs=perturbation_dvs if compute_perturbations else None, perturbation_epsilon=perturbation_epsilon)

            if self.rank == 0 and self.gather_raw_files:
                if is_first_resumed and skip_h5_init:
                    raw_directory = self._find_existing_raw_directory(primary_idx)
                    print(f'Resuming into existing raw data folder: {raw_directory}')
                else:
                    # Make a folder for the raw OpenFOAM save
                    raw_directory = self.storage_location/self.dataset_keyword/f'{self.h5_file_base_name}_{primary_idx}_raw'
                    print(f'Setting up raw data folder.')

                    for i in range(10): # max rename attempts
                        candidate = raw_directory if i == 0 else raw_directory.with_name(f"{raw_directory.name}_{i}")

                        try:
                            os.makedirs(candidate)
                            raw_directory = candidate
                            print(f'Set up raw data directory: {raw_directory}')
                            break
                        except FileExistsError:
                            print('Raw file directory seems to exist. Trying again with incremented name...')
                            continue

                    else:
                        raise RuntimeError("Could not create a unique directory name.")

                    # Copy the constant folder to the raw directory
                    current_directory = Path.cwd()
                    os.chdir(dafoam_directory)
                    shutil.copytree('./constant', raw_directory/'constant')
                    os.chdir(current_directory)

            # Update all of the primary parameters for current point
            for var, info in primary_variables.items():
                sim[var] = info["samples"][primary_idx]

            # Before the first solve of a resumed primary point, clean up any stale
            # OpenFOAM time directories left by the interrupted run.  This prevents:
            #   (a) a leftover 0.9998/ from causing renameSolution to move the wrong
            #       solution into the raw directory (yielding duplicate snapshots), and
            #   (b) partial time directories from a mid-solve interrupt being picked up
            #       as the initial condition for the next solve.
            if is_first_resumed and skip_h5_init:
                self._cleanup_openfoam_for_resume(dafoam_directory)

            for secondary_idx in range(secondary_start, adj_num_secondary_samples):
                if rank == 0:
                    print('\n\n\n\n')
                    print('=============================================')
                    print(f'Primary sample {primary_idx+1}/{adj_num_primary_samples}, secondary sample {secondary_idx+1}/{adj_num_secondary_samples}')
                    if is_first_resumed and secondary_idx == secondary_start and secondary_start > 0:
                        print(f'(Resuming from secondary index {secondary_start})')
                    print('=============================================\n')

                # Update all of the primary parameters for current point and sample
                for var, info in secondary_variables.items():
                    sim[var] = info["samples"][secondary_idx]

                # Primal solve
                sim.run()

                # Do a check to see if the first primal solve of the primary point failed
                # If so, retry by running with a new initial condition taken from
                # the freestream value (reference patch).
                if secondary_idx == 0 and dafoam_instance.primalFail and self.reference_patch is not None:

                    self.print0("Initial solution failed to converge for this primary sample configuration. Trying again with new initial condition...")

                    # Get the values from the reference patch
                    state_reference_values = dafoam_instance.getPatchStateAverages(self.reference_patch, returnVector=True)
                    state_reference_values.pop("phi")

                    # Set these values as initial condition
                    dafoam_instance.setOption("primalInitCondition", state_reference_values)
                    dafoam_instance.updateDAOption()
                    dafoam_instance.setPrimalInitialConditions()

                    # Set flag as not fail so that DAFoamSolver won't reassign last successful solution as IC
                    dafoam_instance.primalFail = 0

                    # Try running again
                    sim.run()

                # is_catchup: base sample was already written in a prior (interrupted) run.
                # We still re-run the solve to restore the correct DAFoam state (needed for
                # perturbations), but we skip write/raw-file/grad operations to avoid
                # corrupting already-stored data.
                is_catchup = (secondary_idx < base_catchup_end)

                if not is_catchup:
                    self.write_sample(h5file_path, secondary_idx)

                # Move OpenFOAM solution to solution directory (skip during catch-up)
                if not is_catchup:
                    if self.gather_raw_files:
                        current_directory = Path.cwd()
                        os.chdir(dafoam_directory)
                        dafoam_instance.renameSolution(9998)

                        quiet_barrier(self.comm)
                        if rank == 0:
                            if comm_size > 1:
                                for i in range(comm_size):
                                    shutil.move(dafoam_directory/f'processor{i}'/'0.9998/',
                                            raw_directory/f'processor{i}'/f'{secondary_idx:04}')

                                    # Copy constant folder to directory (only need to do this once)
                                    if secondary_idx == 0:
                                        shutil.copytree(dafoam_directory/f'processor{i}'/'constant',
                                        raw_directory/f'processor{i}'/'constant')
                            else:
                                shutil.move(dafoam_directory/'0.9998',
                                        raw_directory/f'{secondary_idx:04}')
                        quiet_barrier(self.comm)
                        os.chdir(current_directory)

                    else:
                        current_directory = Path.cwd()
                        os.chdir(dafoam_directory)
                        dafoam_instance.renameSolution(primary_idx * (self.num_secondary_samples + 1) + secondary_idx + 1)
                        os.chdir(current_directory)

                if not is_catchup and compute_objective_grad:
                    # Do an objective check and issue warning only on the first iteration
                    if primary_idx == 0 and secondary_idx == 0:
                        if not self.objectives:
                            warn(f"Rank {rank}: Objectives don't seem to be specified. Skipping gradient computation.")
                            compute_objective_grad = False

                    gradients = sim.compute_totals(self.objectives, list(self.secondary_variables.keys()) + list(self.primary_variables.keys()))

                    with h5py.File(h5file_path, "a", driver="mpio", comm=self.comm) as f:
                        for objective_idx, objective_var in enumerate(self.objectives):
                            objective_name = quote(objective_var.name , safe="")if objective_var.name is not None else f"objective_{objective_idx}"
                            for primary_var in self.primary_variables:
                                dset = f["samples"]["gradients"][objective_name][primary_var.name]
                                if rank  == 0:
                                    dset[:, secondary_idx] = gradients[objective_var, primary_var].flatten()

                            for secondary_var in self.secondary_variables:
                                dset = f["samples"]["gradients"][objective_name][secondary_var.name]
                                if rank == 0:
                                    dset[:, secondary_idx] = gradients[objective_var, secondary_var].flatten()

                if compute_perturbations and secondary_idx >= pert_start_this_primary:
                    self._run_perturbations(h5file_path, secondary_idx, perturbation_dvs, perturbation_epsilon, sim)

            if compute_pod:
                self._compute_pod_modes(h5filepath=h5file_path, **pod_options)

            if compute_split_pod:
                opts = split_pod_options or {}
                if "norm_config" not in opts:
                    self.print0("WARNING: compute_split_pod=True but 'norm_config' not in split_pod_options. Skipping split POD.")
                else:
                    self._compute_split_pod_modes(h5filepath=h5file_path, **opts)

    
    # region _check_for_interrupted_sweep
    def _check_for_interrupted_sweep(self, adj_num_secondary_samples:int, compute_perturbations:bool=False):
        """
        Scan the storage directory for the largest-indexed h5 data file and determine
        where the sweep should resume.  Cases handled:

          - Incomplete base samples: resume mid-secondary loop; h5 file and raw directory
            already exist, so their initialisation must be skipped.

          - Base complete, perturbations incomplete (only when compute_perturbations=True):
            stay on the same primary, skip h5 init, resume perturbations from the first
            unfinished secondary index.

          - Both complete (or no perturbations requested): advance to the next primary.

        Rank 0 performs all file I/O; the result is broadcast to every rank.

        Returns
        -------
        (resume_primary_idx, base_secondary_start, skip_h5_init, pert_secondary_start)
            resume_primary_idx      - first primary index that still needs work, or None.
            base_secondary_start    - first secondary index needing a base solve (0 when
                                      starting a fresh primary file).
            skip_h5_init            - True when the h5 file already exists on disk.
            pert_secondary_start    - first secondary index needing perturbations, or None
                                      when compute_perturbations=False.
        """
        self.print0("Checking for interrupted sweep...")

        result = (None, 0, False, None)

        if self.rank == 0:
            dataset_dir = self.storage_location / self.dataset_keyword

            # Collect files whose stem ends with a bare integer (exclude _modes, etc.)
            data_files = []
            for f in dataset_dir.glob(f'{self.h5_file_base_name}_*.h5'):
                suffix = f.stem[len(self.h5_file_base_name) + 1:]
                if suffix.isdigit():
                    data_files.append((int(suffix), f))

            if data_files:
                data_files.sort(key=lambda x: x[0])
                largest_primary_idx, largest_file = data_files[-1]

                try:
                    with h5py.File(largest_file, "r") as f:
                        sample_group   = f["samples"]
                        param_group    = f["parameters"]
                        sec_var_group  = param_group["secondary_variables"]
                        prim_var_group = param_group["primary_variables"]

                        last_written    = int(sample_group.attrs["last_written_sample_index"])
                        total_snapshots = len(sample_group["converged"])
                        file_complete   = last_written >= total_snapshots - 1

                    # Validate secondary samples (same check regardless of which case)
                    secondary_match = True
                    with h5py.File(largest_file, "r") as f:
                        sec_var_group = f["parameters"]["secondary_variables"]
                        for var, info in self.secondary_variables.items():
                            var_name = var.name
                            if var_name not in sec_var_group:
                                print(f"Resume check: secondary variable '{var_name}' not found in {largest_file.name}. Cannot resume.")
                                secondary_match = False
                                break
                            if not np.allclose(sec_var_group[var_name][()], info["samples"]):
                                print(f"Resume check: secondary variable '{var_name}' samples do not match. Cannot resume.")
                                secondary_match = False
                                break

                    if secondary_match:
                        with h5py.File(largest_file, "r") as f:
                            prim_var_group = f["parameters"]["primary_variables"]
                            primary_match = True
                            for var, info in self.primary_variables.items():
                                var_name = var.name
                                if var_name not in prim_var_group:
                                    print(f"Resume check: primary variable '{var_name}' not found in {largest_file.name}. Cannot resume.")
                                    primary_match = False
                                    break
                                if not np.allclose(prim_var_group[var_name][()], info["samples"][largest_primary_idx]):
                                    print(f"Resume check: primary variable '{var_name}' at index {largest_primary_idx} does not match. Cannot resume.")
                                    primary_match = False
                                    break

                        if primary_match:
                            if not file_complete:
                                # Case: incomplete base samples — resume mid-secondary loop
                                base_secondary_start = last_written + 1
                                print(f"\nFound incomplete sweep file: {largest_file.name}")
                                print(f"  Last written secondary index: {last_written} / {total_snapshots - 1}")
                                if compute_perturbations:
                                    pert_index = self._read_last_perturbation_index(largest_file)
                                    pert_secondary_start = pert_index + 1
                                    actual_start = min(base_secondary_start, pert_secondary_start)
                                    print(f"  Last written perturbation index: {pert_index} / {total_snapshots - 1}")
                                    print(f"  Resuming from primary index {largest_primary_idx}, "
                                          f"secondary index {actual_start} "
                                          f"(base: {base_secondary_start}, pert: {pert_secondary_start}).\n")
                                    result = (largest_primary_idx, base_secondary_start, True, pert_secondary_start)
                                else:
                                    print(f"  Resuming from primary index {largest_primary_idx}, secondary index {base_secondary_start}.\n")
                                    result = (largest_primary_idx, base_secondary_start, True, None)
                            else:
                                if compute_perturbations:
                                    pert_index = self._read_last_perturbation_index(largest_file)
                                    if pert_index < total_snapshots - 1:
                                        # Base complete, perturbations still in progress
                                        pert_secondary_start = pert_index + 1
                                        print(f"\nAll base samples in {largest_file.name} are complete.")
                                        print(f"  Last written perturbation index: {pert_index} / {total_snapshots - 1}")
                                        print(f"  Resuming perturbations from secondary index {pert_secondary_start}.\n")
                                        result = (largest_primary_idx, total_snapshots, True, pert_secondary_start)
                                    else:
                                        # Both base and perturbations complete — advance to next primary
                                        next_primary_idx = largest_primary_idx + 1
                                        print(f"\nAll base samples and perturbations in {largest_file.name} are complete.")
                                        print(f"  Resuming at the start of primary index {next_primary_idx}.\n")
                                        result = (next_primary_idx, 0, False, 0)
                                else:
                                    # Case: file complete, no perturbations — create next primary fresh
                                    next_primary_idx = largest_primary_idx + 1
                                    print(f"\nAll secondary samples in {largest_file.name} are complete.")
                                    print(f"  Resuming at the start of primary index {next_primary_idx}.\n")
                                    result = (next_primary_idx, 0, False, None)

                except Exception as e:
                    print(f"Warning: could not read existing h5 file for resume check: {e}")

        result = self.comm.bcast(result, root=0)
        return result


    # region _find_existing_raw_directory
    def _find_existing_raw_directory(self, primary_idx:int) -> Path:
        """Return the path of an existing raw directory for the given primary index."""
        base = self.storage_location/self.dataset_keyword/f'{self.h5_file_base_name}_{primary_idx}_raw'
        if base.exists():
            return base
        for i in range(1, 10):
            candidate = base.with_name(f"{base.name}_{i}")
            if candidate.exists():
                return candidate
        return base  # Fallback: return the canonical name even if missing


    # region _cleanup_openfoam_for_resume
    def _cleanup_openfoam_for_resume(self, dafoam_directory:Path):
        """
        Remove stale OpenFOAM time directories left by an interrupted run.

        Two failure modes are addressed:
          1. A leftover 0.9998/ directory (rename completed but move did not).
             If this is not removed, the next renameSolution call may fail or
             silently leave the old directory in place, causing the wrong solution
             to be moved into the raw directory — producing duplicate snapshots.
          2. Partial time directories from a mid-solve interrupt (e.g. 0.0001/).
             If these persist, DAFoam may pick up latestTime as the IC rather
             than the intended initial condition, contaminating the resumed solve.

        Only rank 0 performs filesystem operations; a barrier ensures all ranks
        wait until cleanup is complete before the first sim.run() is called.
        """
        if self.rank == 0:
            dafoam_directory = Path(dafoam_directory)

            def is_numeric_time_dir(p):
                try:
                    t = float(p.name)
                    return t > 0  # keep the 0/ IC directory
                except ValueError:
                    return False

            if self.comm_size > 1:
                dirs_to_check = [dafoam_directory / f'processor{i}' for i in range(self.comm_size)]
            else:
                dirs_to_check = [dafoam_directory]

            for proc_dir in dirs_to_check:
                if not proc_dir.exists():
                    continue
                for child in proc_dir.iterdir():
                    if child.is_dir() and is_numeric_time_dir(child):
                        print(f'  Resume cleanup: removing stale time directory {child}')
                        shutil.rmtree(child)

        quiet_barrier(self.comm)


    # region initialize_h5_file
    def initialize_h5_file(self, h5filepath:Path|str, primary_idx:int, compute_objective_grad:bool=False,
                           perturbation_dvs:list=None, perturbation_epsilon:float|Dict[Variable, float]=None):
        comm = self.comm

        # Global sizes across ranks
        num_cells_global                    = self.num_cells_global
        num_faces_no_proc_boundaries_global = self.num_faces_no_proc_boundaries_global

        # Adjust sample size to accomodate refrence values if present
        adj_num_primary_samples   = self.num_primary_samples   + self.primary_has_ref
        adj_num_secondary_samples = self.num_secondary_samples + self.secondary_has_ref

        self.print0(f'Setting up {h5filepath} for writing...', end=" ")

        with h5py.File(h5filepath, "a", driver="mpio", comm=comm) as f:
            parameter_group = f.create_group("parameters")
            sample_group    = f.create_group("samples")
            state_group     = sample_group.create_group("states")
            ref_group       = sample_group.create_group("reference_states")
            mesh_group      = sample_group.create_group("mesh")             

            if self.store_residuals:
                res_group   = sample_group.create_group("residuals")

            for state_name, info in self.state_info.items():
                state_type          = info["type"]
                
                # Determine size based on what kind of quantity
                if      state_type == "volScalarStates" or state_type == "modelStates": size = self.num_cells_global
                elif    state_type == "volVectorStates":                                size = 3 * self.num_cells_global
                elif    state_type == "surfaceScalarStates":                            size = self.num_faces_no_proc_boundaries_global
                else: 
                    raise NotImplementedError(f"Unknown state type, {state_type}. Might need to be added to solver_variable_storage_type?")
                
                state_group.create_dataset(state_name,                      (size, adj_num_secondary_samples),      dtype="f8")
                state_group[state_name].attrs.create("addressing_type", state_type)

                if self.store_residuals:
                    res_group.create_dataset(state_name,                      (size, adj_num_secondary_samples),      dtype="f8")
                    res_group[state_name].attrs.create("addressing_type", state_type)

                if self.reference_patch is not None:
                    ref_group.create_dataset(f"{state_name}",           (adj_num_secondary_samples,),           dtype="f8")

            if compute_objective_grad:
                gradient_group  = sample_group.create_group("gradients")
                for objective_idx, objective in enumerate(self.objectives):
                    objective_name = quote(objective.name, safe="") if objective.name is not None else f"objective_{objective_idx}"
                    if objective.name is None:
                        objective_name = f"objective_{objective_idx}"
                        warn(f"Rank {self.rank}: Found objective without name ({objective}). Saving as {objective_name}.")
                    objective_group = gradient_group.create_group(objective_name) # Write the safe version of the name
                    
                    for primary_var in self.primary_variables:
                        objective_group.create_dataset(primary_var.name, (np.prod(primary_var.shape), adj_num_secondary_samples), dtype="f8")

                    for secondary_var in self.secondary_variables:
                        objective_group.create_dataset(secondary_var.name, (np.prod(secondary_var.shape), adj_num_secondary_samples), dtype="f8")

            mesh_group.create_dataset("centroid_coordinates",     (3 * num_cells_global, adj_num_secondary_samples),                dtype="f8")
            mesh_group.create_dataset("cell_volumes",             (num_cells_global, adj_num_secondary_samples),                    dtype="f8")
            mesh_group.create_dataset("face_areas",               (num_faces_no_proc_boundaries_global, adj_num_secondary_samples), dtype="f8") 
            
            # The following two inidces datasets are useful for debugging
            mesh_group.create_dataset("cell_indices",             (num_cells_global, ),                                             dtype="i8")
            mesh_group.create_dataset("face_indices",             (num_faces_no_proc_boundaries_global, ),                          dtype="i8")
            self._write_field_data_to_dataset(mesh_group["cell_indices"],  self.cell_global_indices, "volScalarStates")
            self._write_field_data_to_dataset(mesh_group["face_indices"], self.face_global_indices, "surfaceScalarStates")

            mesh_group["centroid_coordinates"].attrs.create("addressing_type",  "volVectorStates")
            mesh_group["cell_volumes"].attrs.create("addressing_type",          "volScalarStates")
            mesh_group["face_areas"].attrs.create("addressing_type",            "surfaceScalarStates")
            mesh_group["cell_indices"].attrs.create("addressing_type",          "volScalarStates")
            mesh_group["face_indices"].attrs.create("addressing_type",          "surfaceScalarStates")

            sample_group.attrs.create("last_written_sample_index",          data=-1,                                dtype="i8")
            sample_group.attrs.create("generated_on_n_processors",          data=self.comm_size,                    dtype="i8")
            sample_group.attrs.create("num_cells",                          data=num_cells_global,                  dtype="i8")     

            sample_group.create_dataset("converged",                        (adj_num_secondary_samples, ),          dtype="bool")  

            parameter_group.attrs.create("num_primary_samples",             data=adj_num_primary_samples,           dtype="i8")
            parameter_group.attrs.create("sample_number",                   data=primary_idx,                       dtype="i8")
            parameter_group.attrs.create("num_secondary_samples",           data=adj_num_secondary_samples,         dtype="i8")
            parameter_group.attrs.create("random_state_seed",               data=self.random_state_seed,            dtype="f8")

            primary_var_group = parameter_group.create_group("primary_variables")
            for var, info in self.primary_variables.items():
                primary_var_group.create_dataset(var.name,                  data=info["samples"][primary_idx],      dtype="f8")

            secondary_var_group = parameter_group.create_group("secondary_variables")
            for var, info in self.secondary_variables.items():
                secondary_var_group.create_dataset(var.name,                data=info["samples"],                   dtype="f8")
            secondary_var_group.attrs.create("first_sample_is_reference",   data=self.secondary_has_ref,            dtype="bool")

            if self.non_sampled_variables is not None:
                non_sampled_var_group = parameter_group.create_group("non_sampled_variables")
                for var in self.non_sampled_variables:
                    non_sampled_var_group.create_dataset(var.name,          data=var.value,                         dtype="f8")

            if perturbation_dvs:
                self._initialize_perturbation_group(f, perturbation_dvs, adj_num_secondary_samples, perturbation_epsilon=perturbation_epsilon)

        self.print0('All set!')

    
    # region write_sample
    def write_sample(self, h5filepath:Path, sample_idx:int):
        self.print0('Adding sample...')
        dafoam_instance = self.dafoam_instance
        states          = dafoam_instance.getStates()
        cell_coords     = dafoam_instance.getCellCentroids()
        state_weights   = np.abs(dafoam_instance.getStateWeights())
        state_reference_values = (dafoam_instance.getPatchStateAverages(self.reference_patch)
                                  if self.reference_patch is not None else None)
        if self.store_residuals:
            residuals = dafoam_instance.getResiduals()

        if self.parallel_write:
            self._write_sample_parallel(h5filepath, sample_idx, states, cell_coords,
                                        state_weights, state_reference_values,
                                        residuals if self.store_residuals else None)
        else:
            self._write_sample_root(h5filepath, sample_idx, states, cell_coords,
                                    state_weights, state_reference_values,
                                    residuals if self.store_residuals else None)


    # region _write_sample_parallel
    def _write_sample_parallel(self, h5filepath:Path, sample_idx:int, states:np.ndarray, cell_coords:np.ndarray,
                                state_weights:np.ndarray, state_reference_values:np.ndarray, residuals:np.ndarray):
        with h5py.File(h5filepath, "a", driver="mpio", comm=self.comm) as f:
            sample_group = f["samples"]
            state_group  = sample_group["states"]
            ref_group    = sample_group["reference_states"]
            mesh_group   = sample_group["mesh"]
            if self.store_residuals:
                res_group = sample_group["residuals"]

            added_cell_volumes = False
            added_face_areas   = False

            for state_name, info in self.state_info.items():
                dset       = state_group[state_name]
                indices    = info['indices']
                state_type = info['type']

                if self.store_residuals:
                    self._write_field_data_to_dataset(res_group[state_name], residuals[indices], state_type, sample_idx)

                if self.reference_patch is not None:
                    ref_group[state_name][sample_idx] = state_reference_values[state_name]

                self._write_field_data_to_dataset(dset, states[indices], state_type, sample_idx)

                if state_type == "volScalarStates" and not added_cell_volumes:
                    self._write_field_data_to_dataset(mesh_group["cell_volumes"], state_weights[indices], state_type, sample_idx)
                    added_cell_volumes = True

                if state_type == "surfaceScalarStates" and not added_face_areas:
                    self._write_field_data_to_dataset(mesh_group["face_areas"], state_weights[indices], state_type, sample_idx)
                    added_face_areas = True

            self._write_field_data_to_dataset(mesh_group["centroid_coordinates"], cell_coords, "volVectorStates", sample_idx)

            sample_group["converged"][sample_idx]           = not self.dafoam_instance.primalFail
            sample_group.attrs["last_written_sample_index"] = sample_idx


    # region _write_sample_root
    def _write_sample_root(self, h5filepath:Path, sample_idx:int, states:np.ndarray, cell_coords:np.ndarray,
                           state_weights:np.ndarray, state_reference_values:np.ndarray, residuals:np.ndarray):
        # Gather all distributed field data to rank 0, then rank 0 writes the file
        # serially with contiguous slice writes — avoids MPI-IO point-selection overhead.
        gathered = self._gather_field_data(states, cell_coords, state_weights,
                                           residuals=residuals)

        if self.rank == 0:
            with h5py.File(h5filepath, "a") as f:
                sample_group = f["samples"]
                state_group  = sample_group["states"]
                ref_group    = sample_group["reference_states"]
                mesh_group   = sample_group["mesh"]
                if self.store_residuals:
                    res_group = sample_group["residuals"]

                for state_name in self.state_info:
                    state_group[state_name][:, sample_idx] = gathered["states"][state_name]
                    if self.store_residuals:
                        res_group[state_name][:, sample_idx] = gathered["residuals"][state_name]
                    if self.reference_patch is not None:
                        ref_group[state_name][sample_idx] = state_reference_values[state_name]

                mesh_group["cell_volumes"][:, sample_idx]         = gathered["cell_volumes"]
                mesh_group["face_areas"][:, sample_idx]           = gathered["face_areas"]
                mesh_group["centroid_coordinates"][:, sample_idx] = gathered["centroid_coordinates"]

                sample_group["converged"][sample_idx]           = not self.dafoam_instance.primalFail
                sample_group.attrs["last_written_sample_index"] = sample_idx

        self.comm.Barrier()


    # region _initialize_perturbation_group
    def _initialize_perturbation_group(self, f:h5py.File, perturbation_dvs:list, adj_num_secondary_samples:int, perturbation_epsilon:float|Dict[Variable, float]):
        """Pre-allocate perturbations/<dv_name>/dof_<i>/states/ datasets inside an open h5 file."""
        pert_group = f.create_group("perturbations")
        pert_group.attrs.create("last_written_perturbation_index", data=-1, dtype="i8")

        for var, _ in perturbation_dvs:
            dv_group = pert_group.create_group(var.name)
            dv_group.attrs.create("stepsize", data=perturbation_epsilon[var] if isinstance(perturbation_epsilon, dict) else perturbation_epsilon, dtype="f8")
            shape    = var.value.shape
            num_dofs = int(np.prod(shape)) if shape else 1

            for dof_idx in range(num_dofs):
                dof_group    = dv_group.create_group(f"dof_{dof_idx}")
                states_group = dof_group.create_group("states")

                for state_name, info in self.state_info.items():
                    state_type = info["type"]
                    if   state_type in ("volScalarStates", "modelStates"): size = self.num_cells_global
                    elif state_type == "volVectorStates":                  size = 3 * self.num_cells_global
                    elif state_type == "surfaceScalarStates":              size = self.num_faces_no_proc_boundaries_global
                    else:
                        raise NotImplementedError(f"Unknown state type: {state_type}")

                    dset = states_group.create_dataset(state_name, (size, adj_num_secondary_samples), dtype="f8")
                    dset.attrs.create("addressing_type", state_type)

                dof_group.create_dataset("converged", (adj_num_secondary_samples,), dtype="bool")


    # region _run_perturbations
    def _run_perturbations(self, h5filepath:Path, secondary_idx:int, perturbation_dvs:list,
                           perturbation_epsilon:float|Dict[Variable, float], sim):
        """For each DV DoF, apply a perturbation, run a solve, write the result, then restore."""
        for var, _ in perturbation_dvs:
            base_val = np.asarray(sim[var]).copy()
            shape    = base_val.shape
            flat     = base_val.flatten()
            num_dofs = flat.size if flat.size > 0 else 1
            eps      = perturbation_epsilon if isinstance(perturbation_epsilon, (int, float)) else \
                       perturbation_epsilon.get(var, 1e-4)

            for dof_idx in range(num_dofs):
                self.print0(f'  Perturbation: {var.name} dof_{dof_idx} (+{eps})')
                perturbed          = flat.copy()
                perturbed[dof_idx] += eps
                sim[var]           = perturbed.reshape(shape) if shape else float(perturbed[0])

                sim.run()
                self.write_perturbation_sample(h5filepath, secondary_idx, var.name, dof_idx)

                sim[var] = base_val  # restore base value

        self._update_perturbation_index(h5filepath, secondary_idx)


    # region write_perturbation_sample
    def write_perturbation_sample(self, h5filepath:Path, secondary_idx:int, dv_name:str, dof_idx:int):
        states    = self.dafoam_instance.getStates()
        converged = not self.dafoam_instance.primalFail

        if self.parallel_write:
            self._write_perturbation_parallel(h5filepath, secondary_idx, dv_name, dof_idx, states, converged)
        else:
            self._write_perturbation_root(h5filepath, secondary_idx, dv_name, dof_idx, states, converged)


    # region _write_perturbation_parallel
    def _write_perturbation_parallel(self, h5filepath:Path, secondary_idx:int, dv_name:str, dof_idx:int,
                                     states:np.ndarray, converged:bool):
        with h5py.File(h5filepath, "a", driver="mpio", comm=self.comm) as f:
            dof_group    = f["perturbations"][dv_name][f"dof_{dof_idx}"]
            states_group = dof_group["states"]

            for state_name, info in self.state_info.items():
                self._write_field_data_to_dataset(states_group[state_name], states[info['indices']], info['type'], secondary_idx)

            dof_group["converged"][secondary_idx] = converged


    # region _write_perturbation_root
    def _write_perturbation_root(self, h5filepath:Path, secondary_idx:int, dv_name:str, dof_idx:int,
                                 states:np.ndarray, converged:bool):
        # Gather states to rank 0 (pass dummy arrays for cell_coords/state_weights — not needed here)
        dummy_coords   = np.zeros(3 * self.num_cells, dtype=np.float64)
        dummy_weights  = np.zeros(self.num_state_elements, dtype=np.float64)
        gathered = self._gather_field_data(states, dummy_coords, dummy_weights)

        if self.rank == 0:
            with h5py.File(h5filepath, "a") as f:
                dof_group    = f["perturbations"][dv_name][f"dof_{dof_idx}"]
                states_group = dof_group["states"]

                for state_name in self.state_info:
                    states_group[state_name][:, secondary_idx] = gathered["states"][state_name]

                dof_group["converged"][secondary_idx] = converged

        self.comm.Barrier()


    # region _update_perturbation_index
    def _update_perturbation_index(self, h5filepath:Path, secondary_idx:int):
        """Atomically record that all perturbation DoFs for secondary_idx have been written."""
        if self.parallel_write:
            with h5py.File(h5filepath, "a", driver="mpio", comm=self.comm) as f:
                f["perturbations"].attrs["last_written_perturbation_index"] = secondary_idx
        else:
            if self.rank == 0:
                with h5py.File(h5filepath, "a") as f:
                    f["perturbations"].attrs["last_written_perturbation_index"] = secondary_idx
            self.comm.Barrier()


    # region _read_last_perturbation_index
    def _read_last_perturbation_index(self, h5filepath:Path) -> int:
        """Rank-0 helper: return last_written_perturbation_index from h5, or -1 if absent."""
        try:
            with h5py.File(h5filepath, "r") as f:
                if "perturbations" in f:
                    return int(f["perturbations"].attrs.get("last_written_perturbation_index", -1))
        except Exception:
            pass
        return -1


    # region _gather_field_data
    def _gather_field_data(self, states:np.ndarray, cell_coords:np.ndarray, state_weights:np.ndarray, residuals:np.ndarray|None=None):
        """
        Gather all distributed field arrays to rank 0 and assemble the global arrays.
        Returns a dict of global arrays on rank 0; on other ranks returns None values.
        """
        cell_global_indices             = self.cell_global_indices
        cell_vector_global_indices      = self.cell_vector_global_indices
        face_proc_boundary_mask         = self.face_proc_boundary_mask
        face_masked_sorted_global_indices = self.face_masked_sorted_global_indices
        face_masked_sorting_indices     = self.face_masked_sorting_indices

        def gather_and_assemble(local_data, global_indices, global_size):
            all_data    = self.comm.gather(local_data,    root=0)
            all_indices = self.comm.gather(global_indices, root=0)
            if self.rank == 0:
                out = np.empty(global_size, dtype=np.float64)
                for data_i, idx_i in zip(all_data, all_indices):
                    out[idx_i] = data_i
                return out
            return None

        gathered = {"states": {}, "residuals": {}}

        added_cell_volumes = False
        added_face_areas   = False

        for state_name, info in self.state_info.items():
            indices    = info['indices']
            state_type = info['type']

            if state_type in ("volScalarStates", "modelStates"):
                global_size = self.num_cells_global
                g_indices   = cell_global_indices
            elif state_type == "volVectorStates":
                global_size = 3 * self.num_cells_global
                g_indices   = cell_vector_global_indices
            elif state_type == "surfaceScalarStates":
                global_size = self.num_faces_no_proc_boundaries_global
                local_reordered = states[indices][face_proc_boundary_mask][face_masked_sorting_indices]
                g_indices       = face_masked_sorted_global_indices
                gathered["states"][state_name] = gather_and_assemble(local_reordered, g_indices, global_size)
                if not added_face_areas:
                    local_wa = state_weights[indices][face_proc_boundary_mask][face_masked_sorting_indices]
                    gathered["face_areas"] = gather_and_assemble(local_wa, g_indices, global_size)
                    added_face_areas = True
                if residuals is not None:
                    local_res = residuals[indices][face_proc_boundary_mask][face_masked_sorting_indices]
                    gathered["residuals"][state_name] = gather_and_assemble(local_res, g_indices, global_size)
                continue

            gathered["states"][state_name] = gather_and_assemble(states[indices], g_indices, global_size)

            if state_type == "volScalarStates" and not added_cell_volumes:
                gathered["cell_volumes"] = gather_and_assemble(state_weights[indices], g_indices, global_size)
                added_cell_volumes = True

            if residuals is not None:
                gathered["residuals"][state_name] = gather_and_assemble(residuals[indices], g_indices, global_size)

        # Cell centroid coordinates (volVectorStates)
        gathered["centroid_coordinates"] = gather_and_assemble(
            cell_coords, cell_vector_global_indices, 3 * self.num_cells_global
        )

        return gathered

    
    # region load_h5
    def load_h5(self, h5file_path, group_to_read:str=None, only_distributed_data:bool=False)->Dict:
        
        # Recursive function to walk through dataset (will handle the distributed datatypes)
        def recurse(h5obj):
            result = {}

            # Store group attributes if present
            if len(h5obj.attrs) > 0:
                result["_attrs"] = dict(h5obj.attrs)

            for key, item in h5obj.items():
                # Recurse into groups
                if isinstance(item, h5py.Group):
                    subgroup = recurse(item)
                    if subgroup:
                        result[key] = subgroup

                # Handle datasets
                elif isinstance(item, h5py.Dataset):
                    is_distributed = True if "addressing_type" in item.attrs else False

                    if only_distributed_data and not is_distributed:
                        continue

                    if is_distributed:
                        addressing_type         = item.attrs["addressing_type"]
                        apply_sign_convention   = item.attrs.get("apply_sign_convention", True)
                        result[key] = self._read_field_data_from_dataset(item, addressing_type, apply_sign_convention=apply_sign_convention)

                    else:
                        result[key] = item[()]

            return result
        
        with h5py.File(h5file_path, "r") as f:
            if group_to_read is None:
                return recurse(f)
            else:
                return recurse(f[group_to_read])


    #region _setup_indices_and_state_maps_and_names
    def _setup_indices_state_info_and_global_counts(self):
        self.print0('Setting up state map and processor addressing...')

        state_names, state_map     = self.dafoam_instance.getStateVariableMap(includeComponentSuffix=False)

        face_global_indices                    = np.array(self._read_proc_addressing("face"))
        cell_global_indices                    = np.array(self._read_proc_addressing("cell"))
        cell_vector_global_indices             = np.array((cell_global_indices[:, None]) * 3 + np.arange(3)).ravel()
        
        state_info = {}
        for name in state_names:
            state_index = state_names.index(name)
            state_type  = self.solver_variable_storage_type[name]
            indices     = np.where(np.array(state_map) == state_index)[0]
            state_info[name] = {'indices':  indices,
                                'type':     state_type}

        self.face_global_indices            = face_global_indices
        self.cell_global_indices            = cell_global_indices
        self.cell_vector_global_indices     = cell_vector_global_indices
        self.state_info                     = state_info
        self.state_map                      = state_map

        # Handling processor boundary faces, as well as non-increasing indices (this would be an issue for hdf5)
        self.face_proc_boundary_mask             = face_global_indices >= 0
        self.face_masked_global_indices          = face_global_indices[self.face_proc_boundary_mask] - 1
        self.face_masked_sorting_indices         = np.argsort(self.face_masked_global_indices)
        self.face_masked_sorted_global_indices   = self.face_masked_global_indices[self.face_masked_sorting_indices]
        self.num_faces_no_proc_boundaries        = sum(self.face_proc_boundary_mask)

        # Check whether each rank's cell/face global indices are contiguous ranges.
        # If so, HDF5 slice writes can be used instead of fancy (point-selection) indexing,
        # which is significantly faster in both serial and parallel HDF5.
        cell_diffs = np.diff(cell_global_indices)
        self.cells_are_contiguous = bool(np.all(cell_diffs == 1))
        self.cell_slice_start     = int(cell_global_indices[0]) if self.cells_are_contiguous else None

        face_sorted = self.face_masked_sorted_global_indices
        face_diffs  = np.diff(face_sorted) if len(face_sorted) > 1 else np.array([1])
        self.faces_are_contiguous = bool(np.all(face_diffs == 1))
        self.face_slice_start     = int(face_sorted[0]) if self.faces_are_contiguous else None

        # cell_vector indices are always contiguous when cell indices are (stride-1 by construction)
        self.cell_vector_slice_start = 3 * self.cell_slice_start if self.cells_are_contiguous else None

        if self.cells_are_contiguous:
            self.print0("  Cell indices are contiguous — using slice writes for cell data.")
        else:
            self.print0("  Cell indices are non-contiguous — using fancy-index writes for cell data.")
        if self.faces_are_contiguous:
            self.print0("  Face indices are contiguous — using slice writes for face data.")
        else:
            self.print0("  Face indices are non-contiguous — using fancy-index writes for face data.")

        # Global sizes across ranks
        self.num_cells_global                    = self.comm.allreduce(self.num_cells,                      op=MPI.SUM)
        self.num_faces_global                    = self.comm.allreduce(self.num_faces,                      op=MPI.SUM)
        self.num_faces_no_proc_boundaries_global = self.comm.allreduce(self.num_faces_no_proc_boundaries,   op=MPI.SUM)
        self.num_state_elements_global           = self.comm.allreduce(self.num_state_elements,             op=MPI.SUM)

        # Gather each rank's global index arrays to rank 0.  Used by _read_field_data_from_dataset
        # when parallel_read=False to read contiguously then scatter each rank's portion.
        # These are small (O(local_cells) ints) so the one-time gather cost is negligible.
        positive_face_global_indices_zero_indexed = np.abs(face_global_indices) - 1
        face_sorted_read_indices                  = positive_face_global_indices_zero_indexed[
                                                        np.argsort(positive_face_global_indices_zero_indexed)]
        self.all_cell_global_indices         = self.comm.gather(cell_global_indices,         root=0)
        self.all_cell_vector_global_indices  = self.comm.gather(cell_vector_global_indices,  root=0)
        self.all_face_sorted_read_indices    = self.comm.gather(face_sorted_read_indices,    root=0)

        self.print0('All set!')


    # region _validate_variable_names
    def _validate_variable_names(self):
        """Check that every variable dict key has a non-None .name attribute."""
        sampled = {"primary_variables": self.primary_variables,
                   "secondary_variables": self.secondary_variables}
        for label, var_dict in sampled.items():
            if var_dict is None:
                continue
            for var in var_dict:
                if var.name is None:
                    raise ValueError(
                        f"Variable in '{label}' has no name. "
                        f"Assign one when creating the variable, e.g. "
                        f"csdl.Variable(name='my_var', ...)."
                    )

        if self.non_sampled_variables is not None:
            for var in self.non_sampled_variables:
                if var.name is None:
                    raise ValueError(
                        f"Variable in 'non_sampled_variables' has no name. "
                        f"Assign one when creating the variable, e.g. "
                        f"csdl.Variable(name='my_var', ...)."
                    )


    # region _generate_lhs_samples
    def _generate_lhs_samples(self,
                              var_limits: Dict[Any, Dict[str, Any]], 
                              num_samples: int, 
                              random_state: int = 0
                            ) -> List[Dict[Any, np.ndarray]]:
        """
        Generate Latin Hypercube Samples for variables with arbitrary shapes.
        
        Each element of multi-dimensional variables is sampled independently using
        the parent variable's range.
        
        Args:
            var_limits: Dictionary mapping CSDL variables to their specifications (example below). A 'samples' entry will be appended to a variable's
                        sub-dictionary (along with range) which contains a (num_samples, variable_shape) array of samples
                    csdl_var: {
                        'range': [min, max],   # Sampling range
                        'ref_value': float,    # (Optional) Reference value
                    }
            num_samples: Number of LHS samples to generate
            random_state: Random seed for reproducibility and consistency among ranks
        
        Returns:
            has_ref: Boolean indicating whether all of the variables had a reference value

        """
        # Build flat sampling space
        xlimits       = []
        var_metadata  = []  # Store (var, name, shape, start_idx, end_idx)
        current_idx   = 0
        has_ref       = True

        # Check if all entries contain "ref_value"
        for var_spec in var_limits.values():
            if "ref_value" not in var_spec:
                self.print0("None/not all variables in variable limits have 'ref_value' key. Assuming no reference.")
                has_ref = False
        
        for var, spec in var_limits.items():
            var_range   = spec['range']

            if len(var_range) != 2:
                raise ValueError(f"{var.name}: range must be [min, max], got {var_range}")
            
            shape        = var.value.shape
            num_elements = int(np.prod(shape)) if shape else 1
            
            # Add one xlimit row per element
            xlimits.extend([var_range] * num_elements)
            
            # Store metadata for reconstruction
            var_metadata.append({
                'var': var,
                'shape': shape,
                'start_idx': current_idx,
                'end_idx': current_idx + num_elements,
                'ref_value': spec.get('ref_value', None)
            })
            
            current_idx += num_elements
        
        # Generate samples
        xlimits     = np.array(xlimits)
        sampler     = LHS(xlimits=xlimits, criterion='m', random_state=random_state)
        raw_samples = sampler(num_samples)  # Shape: (num_samples, total_elements)
        
        for meta in var_metadata:
            samples_unfolded    = raw_samples[:, meta['start_idx']:meta['end_idx']]
            folded_shape        = (num_samples,) + meta["shape"] if meta["shape"] else (num_samples,)
            samples_reshaped    = samples_unfolded.reshape(folded_shape)

            if has_ref:
                if meta['shape']:
                    ref_sample  = np.full((1,) + meta["shape"], meta["ref_value"])
                else:
                    ref_sample  = np.array(meta["ref_value"])

                samples = np.concatenate([ref_sample, samples_reshaped], axis=0)
            
            else:
                samples = samples_reshaped

            var                         = meta["var"]
            var_limits[var]["samples"]  = samples

        return has_ref

    
    # region print0
    def print0(self, statement, **kwargs):
        if self.rank == 0:
            print(statement, **kwargs)


    # region _read_proc_addressing
    def _read_proc_addressing(self, key:str="cell") -> np.ndarray:
        key = key.lower()
        if key not in ["boundary", "cell", "face", "point"]:
            raise ValueError( f'{key} does not have an associated ProcAddressing. Please specify "boundary", "cell", "face", or "point".')

        run_directory = Path(self.dafoam_instance.run_directory)
        if self.comm_size > 1:
            filename      = run_directory/f'processor{self.rank}'/'constant'/'polyMesh'/f'{key}ProcAddressing.gz'
        
        # Serial case: just return the range of number of faces or points
        else:
            if key == "cell":
                data = range(0, self.dafoam_instance.solver.getNLocalCells())
            elif key == "face":
                data = range(1, self.dafoam_instance.solver.getNLocalFaces() + 1)
            else:
                raise NotImplementedError("boundary and point proc_addressing haven't been implemented for serial case yet.")
            return data

        openf = gzip.open if filename.suffix == ".gz" else open

        with openf(filename, "rt") as f:
            # Skip header until we hit the size
            for line in f:
                line = line.strip()
                if line.isdigit():
                    n = int(line)
                    break
            else:
                raise RuntimeError("No labelList size found")

            # Expect opening parenthesis
            if f.readline().strip() != "(":
                raise RuntimeError("Malformed labelList: missing '('")

            data = np.fromiter(
                        (int(f.readline()) for _ in range(n)),
                        dtype=np.int64,
                        count=n
                    )
        return data
    

    # region _compute_pod_modes
    def _compute_pod_modes(self, h5filepath:Path, inner_product:str|None=None, centering:str|None='mean', scaling:str|None="reference",
                           exclude_vars:list[str]|None=None,
                           write_h5:bool=True, new_h5_file:bool=True, overwrite_datasets:bool=False, new_file_suffix:str="modes",
                           write_modes_using_write_adjoint_fields:bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        # Accepts either an h5 file path or a pre-loaded dict with keys "data" and "metadata"
        # (internal shortcut used by _leave_one_out_test; see * at end of file for expected structure)
        if isinstance(h5filepath, dict):
            data_dict   = h5filepath["data"]
            metadata    = h5filepath["metadata"]
        else:
            data_dict       = self.load_h5(h5file_path=h5filepath, group_to_read="samples", only_distributed_data=False)
            metadata        = self.load_h5(h5file_path=h5filepath, group_to_read="parameters", only_distributed_data=False)

        reference_state, weights, scaling_values = self._build_pod_inputs(data_dict, metadata, centering, inner_product, scaling)

        _exclude  = set(exclude_vars or [])
        active_vars = [v for v in self.state_info if v not in _exclude]

        # Only need state data and number of samples for POD computation
        data_array    = np.concatenate([data_dict["states"][state_var] for state_var in active_vars], axis=0)
        weights_array = np.concatenate([weights[state_var] for state_var in active_vars], axis=0)

        # Actual POD computation
        modes_array, singular_values = method_of_snapshots_distributed(matrix_local=data_array,
                                                                       comm=self.comm, method="tsqr",
                                                                       weights_local=weights_array,
                                                                       orthogonality_check=True)

        # Extract per-variable modes using variable-order offsets in data_array, not info["indices"].
        # info["indices"] index into DAFoam's state vector, which with adjStateOrdering="cell" is
        # cell-interleaved and does not match data_array's variable-order layout.
        offset = 0
        data_array_slices = {}
        for state_var in active_vars:
            n_rows = data_dict["states"][state_var].shape[0]
            data_array_slices[state_var] = slice(offset, offset + n_rows)
            offset += n_rows

        local_modes = {state_name: modes_array[data_array_slices[state_name], :] for state_name in active_vars}

        if write_h5:
            # Change file path name if new file requested
            outfilepath = Path(h5filepath)
            if new_h5_file:
                outfilepath = outfilepath.with_name(outfilepath.stem + f'_{new_file_suffix}' + outfilepath.suffix)
            
            with h5py.File(outfilepath, "a", driver="mpio", comm=self.comm) as f:
                pod_group       = f.require_group("pod")
                mode_group      = pod_group.require_group("modes")
                reference_group = pod_group.require_group("reference_state")
                if weights is not None:
                    weights_group   = pod_group.require_group("weights")
                scaling_group   = pod_group.require_group("scaling")

                for state_var in active_vars:
                    info        = self.state_info[state_var]
                    state_type  = info["type"]
                    num_modes   = local_modes[state_var].shape[1]
                    if      state_type == "volScalarStates" or state_type == "modelStates": num_rows = self.num_cells_global
                    elif    state_type == "volVectorStates":                                num_rows = 3 * self.num_cells_global
                    elif    state_type == "surfaceScalarStates":                            num_rows = self.num_faces_no_proc_boundaries_global

                    if overwrite_datasets and state_var in mode_group:
                        del mode_group[state_var]
                    mode_group.require_dataset(state_var,        (num_rows, num_modes),              dtype="f8")
                    self._write_field_data_to_dataset(mode_group[state_var], local_modes[state_var], state_type)
                    mode_group[state_var].attrs.create("addressing_type", state_type)
                    if state_type == "surfaceScalarStates":
                        mode_group[state_var].attrs.create("apply_sign_convention", True)

                    if overwrite_datasets and state_var in reference_group:
                        del reference_group[state_var]
                    reference_group.require_dataset(state_var,   (num_rows, ),                       dtype="f8")
                    self._write_field_data_to_dataset(reference_group[state_var], reference_state[state_var], state_type)
                    reference_group[state_var].attrs.create("addressing_type", state_type)

                    if weights is not None:
                        if overwrite_datasets and state_var in weights_group:
                            del weights_group[state_var]
                        weights_group.require_dataset(state_var, (num_rows, ),                       dtype="f8")
                        self._write_field_data_to_dataset(weights_group[state_var], weights[state_var], state_type)
                        weights_group[state_var].attrs.create("addressing_type", state_type)
                        if state_type == "surfaceScalarStates":
                            weights_group[state_var].attrs.create("apply_sign_convention", False) # Flag to tell that we want magnitudes when loading face data (no negatives for processor boundaries)
                    
                    if scaling is not None:
                        if overwrite_datasets and state_var in scaling_group:
                            del scaling_group[state_var]
                        s = np.asarray(scaling_values[state_var])
                        if s.ndim == 1 and s.size > 1:
                            # Per-DOF scaling (e.g. per-face for phi): store as distributed field
                            dset = scaling_group.require_dataset(state_var, (num_rows,), dtype="f8")
                            self._write_field_data_to_dataset(dset, scaling_values[state_var], state_type)
                            dset.attrs.create("addressing_type", state_type)
                            dset.attrs.create("apply_sign_convention", False)
                        else:
                            dset = scaling_group.require_dataset(state_var, shape=(1,), dtype="f8")
                            dset[...] = float(s)
                
                if overwrite_datasets and 'singular_values' in pod_group:
                    del pod_group['singular_values']
                dset = pod_group.require_dataset('singular_values',     shape=singular_values.shape,               dtype="f8")
                dset[...] = singular_values

        if write_modes_using_write_adjoint_fields:

            for i in range(singular_values.size):

                leading_integer         = 2
                solution_write_number   = leading_integer + (i + 1) / 10000

                # Write the mode; excluded vars contribute zeros so the state vector is complete
                mode_pieces = [
                    local_modes[sv][:, i] if sv in local_modes
                    else np.zeros(data_dict["states"][sv].shape[0])
                    for sv in self.state_info
                ]
                self.dafoam_instance.solver.writeAdjointFields("pod_mode_",
                                                               solution_write_number,
                                                               np.concatenate(mode_pieces, axis=0),
                                                               True)

                # Write the mesh
                mesh = np.zeros_like(self.dafoam_instance.xv.flatten())
                self.dafoam_instance.solver.getOFMeshPoints(mesh)
                self.dafoam_instance.solver.writeMeshPoints(mesh, solution_write_number)

        return local_modes, singular_values, reference_state, weights, scaling_values


    # region _compute_split_pod_modes
    def _compute_split_pod_modes(self,
                                  h5filepath: Path,
                                  norm_config: NormalizationConfig,
                                  flow_var_names: List[str] | None = None,
                                  turb_var_name: str = "nuTilda",
                                  centering: str | None = "reference",
                                  write_h5: bool = True,
                                  new_h5_file: bool = True,
                                  overwrite_datasets: bool = False,
                                  new_file_suffix: str = "split_modes",
                                  write_modes_using_write_adjoint_fields: bool = True,
                                  ) -> Tuple[dict, dict, np.ndarray, np.ndarray, dict]:
        """Compute two separate POD bases: one for flow (p, U, T) and one for turbulence (nuTilda).

        Flow basis uses Chu energy norm weights × cell volumes and linear scaling via norm_config.
        Turbulence basis operates in log space (log(1 + nu/nu_ref)) with cell-volume weights.
        phi is excluded from both bases.

        Returns (flow_modes_dict, turb_modes_dict, flow_sv, turb_sv, reference_state_dict).
        """
        if isinstance(h5filepath, dict):
            data_dict = h5filepath["data"]
            metadata  = h5filepath["metadata"]
        else:
            data_dict = self.load_h5(h5file_path=h5filepath, group_to_read="samples", only_distributed_data=False)
            metadata  = self.load_h5(h5file_path=h5filepath, group_to_read="parameters", only_distributed_data=False)

        # --- Default variable groupings ---
        if flow_var_names is None:
            flow_var_names = [v for v in self.state_info if v not in (turb_var_name, "phi")]

        cell_volumes    = data_dict["mesh"]["cell_volumes"][:, 0]    # (n_local_cells,)
        reference_state = {}

        # --- Centering (reference only for split bases) ---
        if centering == "reference":
            if not metadata["secondary_variables"]["_attrs"]["first_sample_is_reference"]:
                self.print0("WARNING: Using first sample as reference state, but dataset may not have been sampled this way.")
            for var in list(self.state_info.keys()):
                if var in data_dict["states"]:
                    reference_state[var]            = data_dict["states"][var][:, 0].copy()
                    data_dict["states"][var]        = data_dict["states"][var][:, 1:]
        else:
            raise NotImplementedError("_compute_split_pod_modes only supports centering='reference'.")

        # ----------------------------------------------------------------
        # FLOW BASIS
        # ----------------------------------------------------------------
        _chu_T_weight = 1.0 / ((norm_config.gamma - 1.0) * norm_config.gamma * norm_config.M_ref**2)

        def _flow_weight_for_var(var: str) -> np.ndarray:
            info = self.state_info[var]
            if info["type"] == "volVectorStates":
                return np.repeat(cell_volumes, 3) * 1.0
            elif var == "p":
                return cell_volumes * norm_config.M_ref**2
            elif var == "T":
                return cell_volumes * _chu_T_weight
            else:
                return cell_volumes * 1.0

        def _flow_scale_for_var(var: str) -> float:
            if var == "p":
                return norm_config.rho_ref * norm_config.U_ref**2
            elif var == "U":
                return norm_config.U_ref
            elif var == "T":
                return norm_config.T_ref
            else:
                return 1.0

        flow_snapshot_arrays = []
        flow_weight_arrays   = []
        flow_data_slices     = {}
        offset = 0
        for var in flow_var_names:
            snap = data_dict["states"][var].copy()
            ref  = reference_state[var]
            s    = _flow_scale_for_var(var)
            snap = (snap - ref[:, None]) / s
            w    = _flow_weight_for_var(var)

            flow_snapshot_arrays.append(snap)
            flow_weight_arrays.append(w)
            n_rows = snap.shape[0]
            flow_data_slices[var] = slice(offset, offset + n_rows)
            offset += n_rows

        flow_matrix  = np.concatenate(flow_snapshot_arrays, axis=0)
        flow_weights = np.concatenate(flow_weight_arrays,   axis=0)

        self.print0("\n  === Flow basis snapshot energy (after centering + Chu scaling) ===")
        flow_total = 0.0
        for var in flow_var_names:
            sl   = flow_data_slices[var]
            data = flow_matrix[sl, :]
            w    = flow_weights[sl]
            e    = self.comm.allreduce(float(np.sum(w[:, None] * data**2)), op=MPI.SUM)
            mx   = self.comm.allreduce(float(np.max(np.abs(data))) if data.size > 0 else 0.0, op=MPI.MAX)
            flow_total += e
            self.print0(f"  {var:>10}: M-weighted energy = {e:.4e}  max|data| = {mx:.4e}")
        self.print0(f"  {'TOTAL':>10}: M-weighted energy = {flow_total:.4e}\n")

        flow_modes_array, flow_sv = method_of_snapshots_distributed(
            matrix_local=flow_matrix, comm=self.comm, method="tsqr",
            weights_local=flow_weights, orthogonality_check=True)

        flow_modes = {var: flow_modes_array[flow_data_slices[var], :] for var in flow_var_names}

        # ----------------------------------------------------------------
        # TURBULENCE BASIS (log space)
        # ----------------------------------------------------------------
        nu_snaps = data_dict["states"][turb_var_name].copy()   # physical space (columns 1..n, reference stripped but not subtracted)
        nu_ref   = reference_state[turb_var_name]              # physical reference snapshot (column 0)
        log_ref  = norm_config.apply_log_scaling_sa(nu_ref)    # (n_local_cells,)

        # Transform each snapshot to log space then centre by log_ref
        nu_log_all     = norm_config.apply_log_scaling_sa(nu_snaps)   # log(1 + nu_i / nu_ref_config) per snapshot
        nu_log_centred = nu_log_all - log_ref[:, None]                 # subtract log of reference snapshot

        turb_weights = cell_volumes.copy()   # unit Chu weight for nuTilda

        self.print0("  === Turb basis snapshot energy (log space, after centering) ===")
        e_nu  = self.comm.allreduce(float(np.sum(turb_weights[:, None] * nu_log_centred**2)), op=MPI.SUM)
        mx_nu = self.comm.allreduce(float(np.max(np.abs(nu_log_centred))) if nu_log_centred.size > 0 else 0.0, op=MPI.MAX)
        self.print0(f"  {turb_var_name:>10}: M-weighted energy = {e_nu:.4e}  max|data| = {mx_nu:.4e}\n")

        turb_modes_array, turb_sv = method_of_snapshots_distributed(
            matrix_local=nu_log_centred, comm=self.comm, method="tsqr",
            weights_local=turb_weights, orthogonality_check=True)

        turb_modes = {turb_var_name: turb_modes_array}

        # ----------------------------------------------------------------
        # Write h5
        # ----------------------------------------------------------------
        if write_h5:
            outfilepath = Path(h5filepath) if not isinstance(h5filepath, dict) else Path(h5filepath.get("_path", "split_modes.h5"))
            if new_h5_file:
                outfilepath = outfilepath.with_name(outfilepath.stem + f'_{new_file_suffix}' + outfilepath.suffix)

            with h5py.File(outfilepath, "a", driver="mpio", comm=self.comm) as f:
                split_group = f.require_group("pod_split")

                # --- norm_config ---
                nc_group = split_group.require_group("norm_config")
                for field in ("p_ref", "rho_ref", "U_ref", "T_ref", "nu_ref", "M_ref", "cv", "gamma"):
                    val = getattr(norm_config, field)
                    if field not in nc_group:
                        nc_group.create_dataset(field, data=float(val))

                # --- flow group ---
                flow_group = split_group.require_group("flow")
                flow_mode_group = flow_group.require_group("modes")
                flow_ref_group  = flow_group.require_group("reference_state")
                flow_wt_group   = flow_group.require_group("weights")

                for var in flow_var_names:
                    info       = self.state_info[var]
                    state_type = info["type"]
                    n_modes    = flow_modes[var].shape[1]
                    if   state_type in ("volScalarStates", "modelStates"): num_rows = self.num_cells_global
                    elif state_type == "volVectorStates":                  num_rows = 3 * self.num_cells_global
                    elif state_type == "surfaceScalarStates":              num_rows = self.num_faces_no_proc_boundaries_global

                    if overwrite_datasets and var in flow_mode_group: del flow_mode_group[var]
                    flow_mode_group.require_dataset(var, (num_rows, n_modes), dtype="f8")
                    self._write_field_data_to_dataset(flow_mode_group[var], flow_modes[var], state_type)
                    flow_mode_group[var].attrs["addressing_type"] = state_type

                    if overwrite_datasets and var in flow_ref_group: del flow_ref_group[var]
                    flow_ref_group.require_dataset(var, (num_rows,), dtype="f8")
                    self._write_field_data_to_dataset(flow_ref_group[var], reference_state[var], state_type)
                    flow_ref_group[var].attrs["addressing_type"] = state_type

                    w_arr = _flow_weight_for_var(var)
                    if overwrite_datasets and var in flow_wt_group: del flow_wt_group[var]
                    flow_wt_group.require_dataset(var, (num_rows,), dtype="f8")
                    self._write_field_data_to_dataset(flow_wt_group[var], w_arr, state_type)
                    flow_wt_group[var].attrs["addressing_type"] = state_type

                if overwrite_datasets and "singular_values" in flow_group: del flow_group["singular_values"]
                dset = flow_group.require_dataset("singular_values", shape=flow_sv.shape, dtype="f8")
                dset[...] = flow_sv

                # --- turb group ---
                turb_group    = split_group.require_group("turb")
                turb_mode_grp = turb_group.require_group("modes")
                turb_ref_grp  = turb_group.require_group("reference_state")
                turb_log_grp  = turb_group.require_group("log_reference")
                turb_wt_grp   = turb_group.require_group("weights")

                nu_info      = self.state_info[turb_var_name]
                nu_type      = nu_info["type"]
                nu_num_rows  = self.num_cells_global   # nuTilda is modelStates (scalar per cell)
                n_turb_modes = turb_modes_array.shape[1]

                if overwrite_datasets and turb_var_name in turb_mode_grp: del turb_mode_grp[turb_var_name]
                turb_mode_grp.require_dataset(turb_var_name, (nu_num_rows, n_turb_modes), dtype="f8")
                self._write_field_data_to_dataset(turb_mode_grp[turb_var_name], turb_modes_array, nu_type)
                turb_mode_grp[turb_var_name].attrs["addressing_type"] = nu_type

                if overwrite_datasets and turb_var_name in turb_ref_grp: del turb_ref_grp[turb_var_name]
                turb_ref_grp.require_dataset(turb_var_name, (nu_num_rows,), dtype="f8")
                self._write_field_data_to_dataset(turb_ref_grp[turb_var_name], nu_ref, nu_type)
                turb_ref_grp[turb_var_name].attrs["addressing_type"] = nu_type

                if overwrite_datasets and turb_var_name in turb_log_grp: del turb_log_grp[turb_var_name]
                turb_log_grp.require_dataset(turb_var_name, (nu_num_rows,), dtype="f8")
                self._write_field_data_to_dataset(turb_log_grp[turb_var_name], log_ref, nu_type)
                turb_log_grp[turb_var_name].attrs["addressing_type"] = nu_type

                if overwrite_datasets and turb_var_name in turb_wt_grp: del turb_wt_grp[turb_var_name]
                turb_wt_grp.require_dataset(turb_var_name, (nu_num_rows,), dtype="f8")
                self._write_field_data_to_dataset(turb_wt_grp[turb_var_name], turb_weights, nu_type)
                turb_wt_grp[turb_var_name].attrs["addressing_type"] = nu_type

                if overwrite_datasets and "singular_values" in turb_group: del turb_group["singular_values"]
                dset = turb_group.require_dataset("singular_values", shape=turb_sv.shape, dtype="f8")
                dset[...] = turb_sv

        # ----------------------------------------------------------------
        # Write OpenFOAM adjoint fields for visualisation
        # ----------------------------------------------------------------
        if write_modes_using_write_adjoint_fields:
            for i in range(flow_sv.size):
                leading_integer       = 2
                solution_write_number = leading_integer + (i + 1) / 10000
                full_mode             = np.zeros(self.num_state_elements)
                for var in flow_var_names:
                    full_mode[self.state_info[var]["indices"]] = flow_modes[var][:, i]
                self.dafoam_instance.solver.writeAdjointFields(
                    "split_flow_mode_", solution_write_number, full_mode, True)
                mesh = np.zeros_like(self.dafoam_instance.xv.flatten())
                self.dafoam_instance.solver.getOFMeshPoints(mesh)
                self.dafoam_instance.solver.writeMeshPoints(mesh, solution_write_number)

            for i in range(turb_sv.size):
                leading_integer       = 3
                solution_write_number = leading_integer + (i + 1) / 10000
                full_mode             = np.zeros(self.num_state_elements)
                full_mode[self.state_info[turb_var_name]["indices"]] = turb_modes_array[:, i]
                self.dafoam_instance.solver.writeAdjointFields(
                    "split_turb_mode_", solution_write_number, full_mode, True)
                mesh = np.zeros_like(self.dafoam_instance.xv.flatten())
                self.dafoam_instance.solver.getOFMeshPoints(mesh)
                self.dafoam_instance.solver.writeMeshPoints(mesh, solution_write_number)

        return flow_modes, turb_modes, flow_sv, turb_sv, reference_state


    # region load_split_pod_modes
    def load_split_pod_modes(self,
                              h5filepath: Path,
                              n_modes_flow: int | None = None,
                              n_modes_turb: int | None = None,
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load split POD bases from h5 and assemble distributed arrays for SplitBasisLSPGModel.

        Parameters
        ----------
        h5filepath      : path to the h5 file containing a 'pod_split' group.
        n_modes_flow    : number of flow modes to retain (None = all).
        n_modes_turb    : number of turb modes to retain (None = all).

        Returns
        -------
        pod_modes_flow      : (n_flow_dofs_local, n_modes_flow)
        pod_modes_turb      : (n_turb_dofs_local, n_modes_turb)
        reference_fom_state : (n_local_states,) full DAFoam state vector, distributed
        cell_volumes        : (n_local_cells,) mesh cell volumes, distributed
        """
        h5filepath = Path(h5filepath)

        # Also need sample data for phi reference and cell volumes
        # Derive the sample h5 path: strip the "_split_modes" suffix added by _compute_split_pod_modes
        stem = h5filepath.stem
        sample_h5 = h5filepath
        for suffix in ("_split_modes",):
            if stem.endswith(suffix):
                sample_h5 = h5filepath.with_name(stem[: -len(suffix)] + h5filepath.suffix)
                break

        sample_data = self.load_h5(h5file_path=sample_h5, group_to_read="samples", only_distributed_data=False)
        cell_volumes = sample_data["mesh"]["cell_volumes"][:, 0]

        with h5py.File(h5filepath, "r", driver="mpio", comm=self.comm) as f:
            split_group = f["pod_split"]

            # --- Flow modes ---
            flow_mode_grp = split_group["flow"]["modes"]
            flow_ref_grp  = split_group["flow"]["reference_state"]

            flow_modes_list = []
            for var in self.state_info:
                if var in flow_mode_grp:
                    dset       = flow_mode_grp[var]
                    state_type = dset.attrs["addressing_type"]
                    modes_full = self._read_field_data_from_dataset(dset, state_type)  # (n_local_dofs, n_modes)
                    if n_modes_flow is not None:
                        modes_full = modes_full[:, :n_modes_flow]
                    flow_modes_list.append(modes_full)

            pod_modes_flow = np.concatenate(flow_modes_list, axis=0)  # (n_flow_dofs_local, n_modes_flow)

            # --- Turb modes ---
            turb_mode_grp  = split_group["turb"]["modes"]
            turb_var_names = list(turb_mode_grp.keys())
            assert len(turb_var_names) == 1, "Expected exactly one turbulence variable in pod_split/turb/modes"
            turb_var_name  = turb_var_names[0]

            dset_nu        = turb_mode_grp[turb_var_name]
            nu_type        = dset_nu.attrs["addressing_type"]
            pod_modes_turb = self._read_field_data_from_dataset(dset_nu, nu_type)
            if n_modes_turb is not None:
                pod_modes_turb = pod_modes_turb[:, :n_modes_turb]

            # --- Reference FOM state (full state vector including phi) ---
            n_local_states      = self.num_state_elements
            reference_fom_state = np.zeros(n_local_states)

            # Flow variables
            for var in self.state_info:
                if var in flow_ref_grp:
                    dset       = flow_ref_grp[var]
                    state_type = dset.attrs["addressing_type"]
                    ref_vals   = self._read_field_data_from_dataset(dset, state_type)
                    reference_fom_state[self.state_info[var]["indices"]] = ref_vals

            # Turbulence variable (physical space reference)
            dset_nu_ref  = split_group["turb"]["reference_state"][turb_var_name]
            nu_ref_state = self._read_field_data_from_dataset(dset_nu_ref, dset_nu_ref.attrs["addressing_type"])
            reference_fom_state[self.state_info[turb_var_name]["indices"]] = nu_ref_state

            # phi: use first snapshot from sample file
            if "phi" in self.state_info and "phi" in sample_data["states"]:
                reference_fom_state[self.state_info["phi"]["indices"]] = sample_data["states"]["phi"][:, 0]

        return pod_modes_flow, pod_modes_turb, reference_fom_state, cell_volumes


    # region load_pod_modes
    def load_pod_modes(self,
                       h5filepath: Path,
                       n_modes: int | None = None,
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load a standard (non-split) POD basis from h5 for PhiComputingLSPGModel.

        Reads the ``pod/`` group written by ``_compute_pod_modes`` and assembles
        distributed arrays in DAFoam's cell-interleaved state-vector ordering.
        Variables that were excluded from the basis (e.g. phi) receive zero mode
        rows; phi reference values are taken from the first snapshot.

        Parameters
        ----------
        h5filepath : path to the h5 file containing a ``pod`` group.
        n_modes    : number of modes to retain (None = all available).

        Returns
        -------
        pod_modes           : (n_local_states, n_modes)   cell-interleaved; excluded-var rows = 0
        reference_fom_state : (n_local_states,)           cell-interleaved; phi from first snapshot
        weights             : (n_local_states,)           cell-interleaved; excluded-var entries = 1
        scaling             : (n_local_states,)           cell-interleaved; excluded-var entries = 1
        """
        h5filepath = Path(h5filepath)

        sample_data    = self.load_h5(h5file_path=h5filepath, group_to_read="samples", only_distributed_data=False)
        n_local_states = self.num_state_elements

        pod_modes           = None   # allocated once n_modes is known
        reference_fom_state = np.zeros(n_local_states)
        weights             = np.ones(n_local_states)
        scaling             = np.ones(n_local_states)

        with h5py.File(h5filepath, "r", driver="mpio", comm=self.comm) as f:
            pod_group  = f["pod"]
            mode_grp   = pod_group["modes"]
            ref_grp    = pod_group["reference_state"]
            has_weights = "weights" in pod_group
            has_scaling = "scaling" in pod_group

            # Determine n_modes from the first variable present in the mode group
            first_var     = next(v for v in self.state_info if v in mode_grp)
            n_modes_avail = mode_grp[first_var].shape[1]
            if n_modes is None:
                n_modes = n_modes_avail
            else:
                n_modes = min(n_modes, n_modes_avail)

            pod_modes = np.zeros((n_local_states, n_modes))

            for var, info in self.state_info.items():
                idx        = info["indices"]
                state_type = info["type"]

                if var in mode_grp:
                    dset       = mode_grp[var]
                    modes_local = self._read_field_data_from_dataset(dset, dset.attrs["addressing_type"])
                    pod_modes[idx, :] = modes_local[:, :n_modes]

                if var in ref_grp:
                    dset    = ref_grp[var]
                    ref_vals = self._read_field_data_from_dataset(dset, dset.attrs["addressing_type"])
                    reference_fom_state[idx] = ref_vals
                elif var == "phi" and "phi" in sample_data.get("states", {}):
                    reference_fom_state[idx] = sample_data["states"]["phi"][:, 0]

                if has_weights and var in pod_group["weights"]:
                    dset_w = pod_group["weights"][var]
                    w_vals = self._read_field_data_from_dataset(dset_w, dset_w.attrs["addressing_type"])
                    weights[idx] = w_vals

                if has_scaling and var in pod_group["scaling"]:
                    dset_s = pod_group["scaling"][var]
                    if "addressing_type" in dset_s.attrs:
                        s_vals  = self._read_field_data_from_dataset(dset_s, dset_s.attrs["addressing_type"])
                        scaling[idx] = s_vals
                    else:
                        scaling[idx] = float(dset_s[0])

        return pod_modes, reference_fom_state, weights, scaling


    # region _compute_per_variable_pod_modes
    def _compute_per_variable_pod_modes(self, h5filepath: Path, inner_product: str | None = None,
                                         centering: str | None = 'mean', scaling: str | None = "reference",
                                         write_h5: bool = True, new_h5_file: bool = True,
                                         overwrite_datasets: bool = False, new_file_suffix: str = "modes_per_var",
                                         write_modes_using_write_adjoint_fields: bool = True):
        """Compute a separate POD basis for each state variable.

        Calls _build_pod_inputs once for shared centering/weighting/scaling, then runs
        an independent SVD on each variable's snapshot matrix.  Results are stored under
        ``pod_per_var/{var}/`` subgroups in the h5 file.

        Returns
        -------
        dict mapping var_name → (local_modes, singular_values, reference_state, weights, scaling_vals)
        """
        if isinstance(h5filepath, dict):
            data_dict = h5filepath["data"]
            metadata  = h5filepath["metadata"]
        else:
            data_dict = self.load_h5(h5file_path=h5filepath, group_to_read="samples", only_distributed_data=False)
            metadata  = self.load_h5(h5file_path=h5filepath, group_to_read="parameters", only_distributed_data=False)

        reference_state, weights, scaling_values = self._build_pod_inputs(data_dict, metadata, centering, inner_product, scaling)

        results = {}
        for state_var, info in self.state_info.items():
            state_type    = info["type"]
            data_array    = data_dict["states"][state_var]          # (n_local_dofs_var, n_snapshots)
            weights_array = weights[state_var] if weights is not None else None

            modes_array, singular_values = method_of_snapshots_distributed(
                matrix_local   = data_array,
                comm           = self.comm,
                method         = "tsqr",
                weights_local  = weights_array,
                orthogonality_check = True,
            )
            results[state_var] = (modes_array, singular_values, reference_state[state_var],
                                  weights[state_var] if weights is not None else None,
                                  scaling_values[state_var] if scaling_values is not None else None)

        if write_h5:
            outfilepath = Path(h5filepath)
            if new_h5_file:
                outfilepath = outfilepath.with_name(outfilepath.stem + f'_{new_file_suffix}' + outfilepath.suffix)

            with h5py.File(outfilepath, "a", driver="mpio", comm=self.comm) as f:
                root_group = f.require_group("pod_per_var")

                for state_var, info in self.state_info.items():
                    state_type    = info["type"]
                    local_modes, singular_values, ref_state_var, weights_var, scaling_var = results[state_var]
                    num_modes = local_modes.shape[1]

                    if   state_type in ("volScalarStates", "modelStates"): num_rows = self.num_cells_global
                    elif state_type == "volVectorStates":                   num_rows = 3 * self.num_cells_global
                    elif state_type == "surfaceScalarStates":               num_rows = self.num_faces_no_proc_boundaries_global

                    var_group       = root_group.require_group(state_var)
                    mode_group      = var_group.require_group("modes")
                    reference_group = var_group.require_group("reference_state")
                    weights_group   = var_group.require_group("weights")
                    scaling_group   = var_group.require_group("scaling")

                    # Modes
                    if overwrite_datasets and state_var in mode_group:
                        del mode_group[state_var]
                    mode_group.require_dataset(state_var, (num_rows, num_modes), dtype="f8")
                    self._write_field_data_to_dataset(mode_group[state_var], local_modes, state_type)
                    mode_group[state_var].attrs.create("addressing_type", state_type)
                    if state_type == "surfaceScalarStates":
                        mode_group[state_var].attrs.create("apply_sign_convention", True)

                    # Reference state
                    if overwrite_datasets and state_var in reference_group:
                        del reference_group[state_var]
                    reference_group.require_dataset(state_var, (num_rows,), dtype="f8")
                    self._write_field_data_to_dataset(reference_group[state_var], ref_state_var, state_type)
                    reference_group[state_var].attrs.create("addressing_type", state_type)

                    # Weights
                    if weights_var is not None:
                        if overwrite_datasets and state_var in weights_group:
                            del weights_group[state_var]
                        weights_group.require_dataset(state_var, (num_rows,), dtype="f8")
                        self._write_field_data_to_dataset(weights_group[state_var], weights_var, state_type)
                        weights_group[state_var].attrs.create("addressing_type", state_type)
                        if state_type == "surfaceScalarStates":
                            weights_group[state_var].attrs.create("apply_sign_convention", False)

                    # Scaling
                    if scaling_var is not None:
                        if overwrite_datasets and state_var in scaling_group:
                            del scaling_group[state_var]
                        s = np.asarray(scaling_var)
                        if s.ndim == 1 and s.size > 1:
                            dset = scaling_group.require_dataset(state_var, (num_rows,), dtype="f8")
                            self._write_field_data_to_dataset(dset, scaling_var, state_type)
                            dset.attrs.create("addressing_type", state_type)
                            dset.attrs.create("apply_sign_convention", False)
                        else:
                            dset = scaling_group.require_dataset(state_var, shape=(1,), dtype="f8")
                            dset[...] = float(s)

                    # Singular values
                    sv_key = "singular_values"
                    if overwrite_datasets and sv_key in var_group:
                        del var_group[sv_key]
                    dset = var_group.require_dataset(sv_key, shape=singular_values.shape, dtype="f8")
                    dset[...] = singular_values

        if write_modes_using_write_adjoint_fields:
            # Write the first mode of each variable as a full adjoint field for visualization
            for state_var, (local_modes, singular_values, *_) in results.items():
                for i in range(singular_values.size):
                    leading_integer       = 3
                    solution_write_number = leading_integer + (i + 1) / 10000
                    mode_pieces = [
                        results[sv][0][:, i] if sv == state_var
                        else np.zeros(data_dict["states"][sv].shape[0])
                        for sv in self.state_info
                    ]
                    self.dafoam_instance.solver.writeAdjointFields(
                        f"pod_mode_{state_var}_", solution_write_number,
                        np.concatenate(mode_pieces, axis=0), True)
                    mesh = np.zeros_like(self.dafoam_instance.xv.flatten())
                    self.dafoam_instance.solver.getOFMeshPoints(mesh)
                    self.dafoam_instance.solver.writeMeshPoints(mesh, solution_write_number)

        return results


    # region load_per_variable_pod_modes
    def load_per_variable_pod_modes(self,
                                     h5filepath: Path,
                                     n_modes_per_var: dict | int | None = None,
                                     ) -> tuple:
        """Load per-variable POD bases from h5 for PerVariableLSPGModel.

        Parameters
        ----------
        h5filepath      : path to h5 file containing a ``pod_per_var`` group.
        n_modes_per_var : int  → same cap for every variable
                          dict → {var_name: n_modes} per-variable caps
                          None → use all available modes

        Returns
        -------
        pod_modes_per_var   : dict[str, np.ndarray] — {var: (n_local_dofs_var, n_modes)}
                              in variable-order DOF layout (NOT cell-interleaved)
        reference_fom_state : (n_local_states,) cell-interleaved
        weights             : (n_local_states,) cell-interleaved
        scaling             : (n_local_states,) cell-interleaved
        """
        h5filepath     = Path(h5filepath)
        n_local_states = self.num_state_elements

        # Normalise n_modes_per_var to a dict
        if n_modes_per_var is None or isinstance(n_modes_per_var, int):
            _n = n_modes_per_var
            n_modes_per_var = {v: _n for v in self.state_info}

        reference_fom_state = np.zeros(n_local_states)
        weights             = np.ones(n_local_states)
        scaling             = np.ones(n_local_states)
        pod_modes_per_var_out = {}

        with h5py.File(h5filepath, "r", driver="mpio", comm=self.comm) as f:
            root_group = f["pod_per_var"]

            for var, info in self.state_info.items():
                idx        = info["indices"]
                state_type = info["type"]
                var_group  = root_group[var]

                # Modes (variable-order layout, NOT re-indexed to cell-interleaved)
                dset         = var_group["modes"][var]
                modes_local  = self._read_field_data_from_dataset(dset, dset.attrs["addressing_type"])
                n_cap        = n_modes_per_var.get(var)
                if n_cap is not None:
                    modes_local = modes_local[:, :n_cap]
                pod_modes_per_var_out[var] = modes_local

                # Reference state → cell-interleaved
                dset_ref = var_group["reference_state"][var]
                ref_vals = self._read_field_data_from_dataset(dset_ref, dset_ref.attrs["addressing_type"])
                reference_fom_state[idx] = ref_vals

                # Weights → cell-interleaved
                if "weights" in var_group and var in var_group["weights"]:
                    dset_w = var_group["weights"][var]
                    w_vals = self._read_field_data_from_dataset(dset_w, dset_w.attrs["addressing_type"])
                    weights[idx] = w_vals

                # Scaling → cell-interleaved
                if "scaling" in var_group and var in var_group["scaling"]:
                    dset_s = var_group["scaling"][var]
                    if "addressing_type" in dset_s.attrs:
                        s_vals  = self._read_field_data_from_dataset(dset_s, dset_s.attrs["addressing_type"])
                        scaling[idx] = s_vals
                    else:
                        scaling[idx] = float(dset_s[0])

        return pod_modes_per_var_out, reference_fom_state, weights, scaling


    # region _build_pod_inputs
    def _build_pod_inputs(self, data_dict:Dict, metadata:Dict, centering:str|None, 
                          inner_product:str|None, scaling:str|None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply centering, weighting, and scaling to data_dict["states"] in place.

        centering:     None | 'mean' | 'reference' | dict of arrays
        inner_product: None | 'reference' | dict of arrays
        scaling:       None | 'reference' | dict of scalars

        Returns reference_state, weights, scaling_values dicts (keyed by state name).
        weights is None when inner_product is None.
        """
        reference_state = {}
        weights         = None if inner_product is None else {}
        scaling_values  = {}
        _phi_rho_ref    = None  # saved for per-snapshot phi normalization
        _phi_U_ref      = None

        for state_var, info in self.state_info.items():
            state_data = data_dict["states"][state_var]
            state_type = info["type"]

            # --- Centering ---
            # Options: None (zero), 'mean', 'reference' (first snapshot), or a dict of arrays
            if centering is None:
                reference_state[state_var] = np.zeros_like(state_data[:, 0])
            elif isinstance(centering, str):
                if centering == 'mean':
                    reference_state[state_var] = np.mean(state_data, axis=1)
                elif centering == 'reference':
                    if not metadata["secondary_variables"]["_attrs"]["first_sample_is_reference"]:
                        self.print0("WARNING: Using first sample as reference state, even though it seems the dataset was not sampled this way.")
                    reference_state[state_var]      = state_data[:, 0]
                    data_dict["states"][state_var]  = state_data[:, 1:]
                else:
                    raise TypeError(f'"{centering}" is not a valid centering method.')
            elif isinstance(centering, dict):
                reference_state[state_var] = centering[state_var]
            else:
                raise TypeError(f'"{centering}" is not a valid centering method.')

            # --- Weighting (inner product) ---
            # Options: None (skip), 'reference' (cell volumes / face areas of first sample), or a dict
            if inner_product is not None:
                if isinstance(inner_product, str):
                    if inner_product == "reference":
                        if state_type == "volVectorStates":
                            weights[state_var] = np.repeat(data_dict["mesh"]['cell_volumes'][:, 0], repeats=3, axis=0)
                        elif state_type == 'volScalarStates' or state_type == "modelStates":
                            weights[state_var] = data_dict["mesh"]['cell_volumes'][:, 0]
                        elif state_type == "surfaceScalarStates":
                            # Face-area weights have units m², cell-volume weights have units m³.
                            # Without correction the face-area integral dominates because the totals
                            # are numerically incomparable (sum A_f >> sum V_c for large 3-D meshes).
                            # Multiplying by L_char = V_total / A_total converts to effective volumes
                            # so that phi's weighted norm is O(V_total * d²), matching cell variables.
                            face_areas   = np.abs(data_dict["mesh"]['face_areas'][:, 0])
                            cell_volumes = data_dict["mesh"]['cell_volumes'][:, 0]
                            V_total = self.comm.allreduce(np.sum(cell_volumes), op=MPI.SUM)
                            A_total = self.comm.allreduce(np.sum(face_areas),   op=MPI.SUM)
                            L_char  = V_total / A_total
                            weights[state_var] = face_areas * L_char
                        else:
                            raise TypeError(f"State type {state_type} not recognized.")
                    else:
                        raise NotImplementedError(f"Inner product weight '{inner_product}' has not been implemented.")
                elif isinstance(inner_product, dict):
                    weights[state_var] = inner_product[state_var]
                else:
                    assert np.asarray(inner_product).shape == np.asarray(reference_state[state_var]).shape, \
                        "Supplied inner_product weight vector shape is incompatible."

            # --- Scaling ---
            # Options: None (ones), 'reference' (freestream patch averages), or a dict of scalars
            # nuTilda is over-scaled by 100x to reduce its contribution to POD mode energy.
            # phi uses per-face reference values so that large-area faces don't dominate the inner
            # product via the A_face^3 effect (phi_f ~ rho*U*A_f, so a scalar rho*U normalization
            # leaves phi_normalized ~ A_f, and the face-area weighted norm picks up A_f^3).
            if scaling is None:
                scaling_values[state_var] = np.ones_like(reference_state[state_var])
            elif scaling == 'reference':
                if "reference_states" not in data_dict:
                    raise TypeError('Reference values not found in dataset during POD compute setup.')
                reference_states = data_dict["reference_states"]
                if state_var not in reference_states:
                    raise TypeError(f'Reference value not found for {state_var} in dataset during POD compute setup.')
                if state_var == "p":
                    # Scale by dynamic pressure q_inf = 0.5*rho*U^2 rather than absolute pressure p0.
                    # p0 (~1e5 Pa) >> q_inf (~few kPa), so scaling by p0 makes pressure fluctuations
                    # O(1e-3) in the scaled space, effectively invisible to the SVD and yielding near-
                    # zero pressure representation in the POD basis — causing large lift/drag errors.
                    rho_ref = reference_states["p"][0] / reference_states["T"][0] / 287.
                    U_ref   = reference_states["U"][0]
                    scaling_values[state_var] = 0.5 * rho_ref * U_ref**2
                elif state_var == "nuTilda":
                    scaling_values[state_var] = 100 * reference_states[state_var][0]
                elif state_var == "phi":
                    # scaling_values["phi"] = rho_ref * U_ref * A_f_ref (per face, using reference
                    # snapshot face areas).  This is what the ROM uses for state reconstruction.
                    #
                    # The SNAPSHOT DATA is normalised below by rho_ref * U_ref * A_f_j (per-snapshot
                    # face area) rather than A_f_ref.  This removes the mesh-deformation-driven
                    # component of phi from the POD variance: phi = rho * U_n * A_f, so changes in
                    # A_f across geometrically-varied snapshots would otherwise dominate the POD even
                    # when the flow barely changes.  After per-snapshot normalisation,
                    # phi_norm ≈ U_n_j / U_ref, which is bounded in [-1, 1] for subsonic flow and
                    # comparable in magnitude to the normalised cell variables.
                    face_areas_ref = np.abs(data_dict["mesh"]['face_areas'][:, 0])
                    rho_ref = reference_states["p"][0] / reference_states["T"][0] / 287.
                    U_ref   = reference_states["U"][0]
                    phi_face_scale = rho_ref * U_ref * face_areas_ref
                    phi_face_scale = np.where(phi_face_scale < 1e-300, 1.0, phi_face_scale)
                    scaling_values[state_var] = phi_face_scale
                    _phi_rho_ref = rho_ref  # saved for per-snapshot normalization in the apply step
                    _phi_U_ref   = U_ref
                else:
                    scaling_values[state_var] = reference_states[state_var][0]
            elif isinstance(scaling, dict):
                scaling_values[state_var] = scaling[state_var]
            else:
                raise TypeError("Not a valid scaling method. Please supply None, 'reference', or a dict.")

            if state_var == "phi" and _phi_rho_ref is not None:
                # Per-snapshot face-area normalization: divide snapshot j's phi by A_f_j (not A_f_ref).
                # face_areas columns follow the same snapshot ordering as state data; after
                # 'reference' centering removes column 0, the remaining k columns correspond to
                # face_areas[:, 1:k+1].  For 'mean'/None centering all n columns are kept.
                n_cols  = data_dict["states"]["phi"].shape[1]
                offset  = 1 if (isinstance(centering, str) and centering == "reference") else 0
                fa_snap = np.abs(data_dict["mesh"]["face_areas"][:, offset:offset + n_cols])
                phi_scale_per_snap = _phi_rho_ref * _phi_U_ref * fa_snap
                phi_scale_per_snap = np.where(phi_scale_per_snap < 1e-300, 1.0, phi_scale_per_snap)
                data_dict["states"]["phi"] = (
                    data_dict["states"]["phi"] - reference_state["phi"][:, None]
                ) / phi_scale_per_snap
            else:
                s = np.asarray(scaling_values[state_var])
                s_col = s[:, None] if s.ndim == 1 and s.size > 1 else s
                data_dict["states"][state_var] = (data_dict["states"][state_var] - reference_state[state_var][:, None]) / s_col

        # Diagnostic: M-weighted snapshot energy per variable (printed once during POD setup)
        self.print0("\n  === Snapshot energy diagnostic (after centering + scaling) ===")
        total_energy = 0.0
        energies = {}
        for state_var in self.state_info:
            data = data_dict["states"][state_var]
            w    = weights[state_var] if weights is not None else np.ones(data.shape[0])
            local_e  = float(np.sum(w[:, None] * data**2))
            local_mx = float(np.max(np.abs(data))) if data.size > 0 else 0.0
            global_e  = self.comm.allreduce(local_e,  op=MPI.SUM)
            global_mx = self.comm.allreduce(local_mx, op=MPI.MAX)
            energies[state_var] = global_e
            total_energy += global_e
            n_active = self.comm.allreduce(float(np.sum(w > 0)), op=MPI.SUM)
            self.print0(f"  {state_var:>10}: M-weighted energy = {global_e:.4e}  max|data| = {global_mx:.4e}  active DOFs = {n_active:.0f}")
        self.print0(f"  {'TOTAL':>10}: M-weighted energy = {total_energy:.4e}")
        self.print0(f"  {'':>10}  Fraction per variable:")
        for state_var, e in energies.items():
            self.print0(f"  {state_var:>10}: {e/max(total_energy, 1e-300):.4f}")
        self.print0("")

        return reference_state, weights, scaling_values


    # region _leave_one_out_test
    def _leave_one_out_test(self, h5filepath:Path, num_modes:bool=None, pod_options:Dict={}):
        data_dict       = self.load_h5(h5file_path=h5filepath, group_to_read="samples", only_distributed_data=False)
        metadata        = self.load_h5(h5file_path=h5filepath, group_to_read="parameters", only_distributed_data=False)

        first_sample_is_ref = metadata["secondary_variables"]["_attrs"]["first_sample_is_reference"]
        num_samples         = metadata["_attrs"]["num_secondary_samples"]


        # We'll copy the states, just so we don't have to worry about accidentally deleting
        # TODO: improve memory management by reducing the amount of copies floating around
        data_dict["original_states"] = data_dict["states"].copy()

        supply_dict     = {"data": data_dict, "metadata": metadata}
        
        out_state       = {} # This will be our left out state
        remaining_state = {}
        e               = np.zeros((num_samples - first_sample_is_ref,))

        # Main loop for leave one out. Will loop through all snapshots and perform POD without the current snapshot and compute projection error
        for read_index in range(first_sample_is_ref, num_samples):
            
            for state_var, info in self.state_info.items():
                state_data                      = data_dict["original_states"][state_var].copy()
                out_state[state_var]            = state_data[:, read_index]
                remaining_state[state_var]      = np.delete(state_data, read_index, axis=1)
                supply_dict["data"]["states"]   = remaining_state

            local_modes, _, reference_state, weights, scaling_values = self._compute_pod_modes(supply_dict, **pod_options, write_h5=False)

            '''
            For reference:
            Let's say CFD snapshots are stored in U. Before computing the POD we scale the state:
            X = S^{-1} * (U - u_ref)
            where S is a diagonal scaling matrix. POD is performed on X using a weighted inner product defined by diagonal matrix M.

            The POD modes satisfy:
            Phi^T * M * Phi = I

            Reconstruction of a physical state is
            u ~ u_ref + S * Phi * a

            Define the scaled state
            x = S^{-1} * (u - u_ref)

            Projection onto the POD space is done in the scaled coordinates:
            a      = Phi^T * M * x
            x_hat  = Phi * a

            Transforming back to the physical state gives
            u_hat = u_ref + S * x^hat
                  = u_ref + S * Phi * Phi^T * M * S^{-1} * (u - u_ref)

            Projection error is measured in the same weighted norm used for POD:
            ||x||_M = sqrt( x^T * M * x )

            Relative projection error for snapshot i:
            e_i = ||x_i - x_hat_i||_M / ||x_i||_M

            where
            x_i     = S^{-1} * (u_i - u_ref)
            x_hat_i = Phi * Phi^T * M * x_i
            '''
            # We need to adjust the index here for the case where the first data index was the reference
            i = read_index - first_sample_is_ref

            # vectorize everything:
            Phi     = np.concatenate([local_modes[var]      for var in self.state_info.keys()], axis=0)
            u_i     = np.concatenate([out_state[var]        for var in self.state_info.keys()], axis=0)
            u_ref   = np.concatenate([reference_state[var]  for var in self.state_info.keys()], axis=0)   
            m       = np.concatenate([weights[var]          for var in self.state_info.keys()], axis=0)
            s       = np.concatenate([scaling_values[var] * np.ones_like(out_state[var]) for var in self.state_info.keys()], axis=0)  

            # Retain given number of modes
            num_modes = Phi.shape[1] if num_modes is None else num_modes
            Phi       = Phi[:, 0:num_modes]

            # Local projection and global reduction
            x_i         = 1 / s * (u_i - u_ref)
            local_a     = Phi.T @ (m * x_i)
            global_a    = self.comm.allreduce(local_a, op=MPI.SUM)

            # Reconstruct local state and difference
            x_hat       = Phi @ global_a
            x_diff      = x_hat - x_i
            
            # Compute norms of scaled data under weighted norm (used for POD)
            norm_x_diff = np.sqrt(self.comm.allreduce(np.sum((x_diff * m * x_diff)), op=MPI.SUM))
            norm_x_i    = np.sqrt(self.comm.allreduce(np.sum((x_i * m * x_i)),       op=MPI.SUM))

            # Relative error
            e_i         = norm_x_diff / norm_x_i
            e[i]        = e_i

            self.print0(f"Projection error for leave-one-out of sample {i}: {e_i:.4e}")

        # Print out configurations with largest error
        error_sort_indices = np.argsort(e)

        self.print0(f"Configurations with largest error:")
        for i in error_sort_indices[-5:]:
            self.print0(i)
            for param, configs in metadata["secondary_variables"].items():
                if param == "_attrs":
                    continue
                self.print0(f"     {param}: {configs[i, :]}")
            self.print0(f"     Projection error: {e[i]}")

        if self.rank == 0:
            plt.figure()
            plt.semilogy(e[error_sort_indices])
            plt.xlabel("Sample")
            plt.ylabel("Error (||x_i - x_hat_i||_M / ||x_i||_M)")
            
            all_configs = np.concatenate(
                            [configs[first_sample_is_ref:, :] for param, configs in metadata["secondary_variables"].items() if param != "_attrs"],
                            axis=1
                        )
            
            lower_bounds = np.min(all_configs, axis=0)
            upper_bounds = np.max(all_configs, axis=0)
            all_configs_norm = (all_configs - lower_bounds) / (upper_bounds - lower_bounds)

            
            norm = mpl.colors.Normalize(vmin=e.min(), vmax=e.max())
            cmap = plt.cm.Reds

            plt.figure()
            print(f"all_configs_norm.shape = {all_configs_norm.shape}")
            print(f"e.shape = {e.shape}")
            for i in error_sort_indices:
                plt.plot(all_configs_norm[i, :], color=cmap(norm(e[i])), alpha=0.7)

            
            mu          = all_configs.mean(axis=0)
            Sigma_inv   = np.linalg.inv(np.cov(all_configs, rowvar=False))
            d           = np.array([distance.mahalanobis(x, mu, Sigma_inv) for x in all_configs])

            plt.figure()
            plt.scatter(d, e)
            plt.xlabel(f"Mahalanobis distance")
            plt.ylabel(f"Projection error")

            k = 5  # number of neighbors
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(all_configs)
            distances, indices = nbrs.kneighbors(all_configs)
            # skip the first column (distance to itself = 0)
            d_knn = distances[:, 1:].mean(axis=1)
            plt.figure()
            plt.scatter(d_knn, e)
            plt.xlabel(f"K-nearest neighbor distance (k = {k})")
            plt.ylabel(f"Projection error")

            plt.figure()
            plt.scatter(d, d_knn, c=e)
            plt.xlabel(f"K-nearest neighbor distance (k = {k})")
            plt.ylabel(f"Mahalanobis distance")
            plt.show()
            
        quiet_barrier(self.comm)


    # region _write_field_data_to_dataset
    def _write_field_data_to_dataset(self, dset:h5py.Dataset, data:np.ndarray, field_type:str, column_idx:int=None):

        cell_global_indices                     = self.cell_global_indices
        cell_vector_global_indices              = self.cell_vector_global_indices
        face_masked_sorted_global_indices       = self.face_masked_sorted_global_indices
        face_masked_sorting_indices             = self.face_masked_sorting_indices
        face_proc_boundary_mask                 = self.face_proc_boundary_mask

        with dset.collective:
            if field_type == "volScalarStates" or field_type == "modelStates":
                if self.cells_are_contiguous:
                    s = self.cell_slice_start
                    sl = slice(s, s + len(data))
                    if column_idx is None:  dset[sl]           = data
                    else:                   dset[sl, column_idx] = data
                else:
                    if column_idx is None:  dset[cell_global_indices]           = data
                    else:                   dset[cell_global_indices, column_idx] = data

            elif field_type == "volVectorStates":
                if self.cells_are_contiguous:
                    s = self.cell_vector_slice_start
                    sl = slice(s, s + len(data))
                    if column_idx is None:  dset[sl]           = data
                    else:                   dset[sl, column_idx] = data
                else:
                    if column_idx is None:  dset[cell_vector_global_indices]           = data
                    else:                   dset[cell_vector_global_indices, column_idx] = data

            elif field_type == "surfaceScalarStates":
                reordered = data[face_proc_boundary_mask][face_masked_sorting_indices]
                if self.faces_are_contiguous:
                    s = self.face_slice_start
                    sl = slice(s, s + len(reordered))
                    if column_idx is None:  dset[sl]           = reordered
                    else:                   dset[sl, column_idx] = reordered
                else:
                    if column_idx is None:  dset[face_masked_sorted_global_indices]             = reordered
                    else:                   dset[face_masked_sorted_global_indices, column_idx] = reordered

            else:
                raise NotImplementedError(f"Unknown state type, {field_type}. Might need to be added to solver_variable_storage_type?")
        

    # region _read_field_data_from_dataset
    def _read_field_data_from_dataset(self, dset:h5py.Dataset, field_type:str, column_idx:int=None, 
                                      apply_sign_convention:bool=True) -> np.ndarray:
        # apply_sign_convention = true will negate the stored data for the oppositely oriented faces
        # (This matches DAFoam face area convention for processor boundary faces - set to false for magnitudes instead)

        if not self.parallel_read:
            return self._read_field_data_from_dataset_root(dset, field_type, column_idx, apply_sign_convention)

        cell_global_indices                     = self.cell_global_indices
        cell_vector_global_indices              = self.cell_vector_global_indices
        face_global_indices                     = self.face_global_indices

        # Have to recover the proper mapping for the faces
        negative_indices                            = self.face_global_indices < 0
        positive_face_global_indices_zero_indexed   = (np.abs(face_global_indices) - 1)
        negative_mask                               = np.ones_like(face_global_indices)
        negative_mask[negative_indices]             = -1 if apply_sign_convention else 1

        positive_face_zero_indexed_ordered_indices = np.argsort(positive_face_global_indices_zero_indexed)

        if field_type == "volScalarStates" or field_type == "modelStates":
            data = dset[cell_global_indices]

        elif field_type == "volVectorStates":
            data = dset[cell_vector_global_indices]

        elif field_type == "surfaceScalarStates":
            if column_idx is None:
                if dset.ndim > 1:
                    num_columns     = dset.shape[1]
                    local_data      = np.zeros((self.num_faces, num_columns))
                else:
                    local_data      = np.zeros((self.num_faces, ))
            else:
                local_data     = np.zeros((self.num_faces,))
                dset           = dset[:, column_idx]

            if dset.ndim > 1:
                dset_sorted                                             = dset[positive_face_global_indices_zero_indexed[positive_face_zero_indexed_ordered_indices], :]
            else:
                dset_sorted                                             = dset[positive_face_global_indices_zero_indexed[positive_face_zero_indexed_ordered_indices]]

            negative_mask_sorted                                        = negative_mask[positive_face_zero_indexed_ordered_indices] # Recall: Negative mask won't do anything if apply_sign_convention=False

            if dset.ndim > 1:
                local_data_sorted                                           = negative_mask_sorted[:, None] * dset_sorted # Recall: Negative mask won't do anything if apply_sign_convention=False
                local_data[positive_face_zero_indexed_ordered_indices, :]   = local_data_sorted
            else:
                local_data_sorted                                           = negative_mask_sorted * dset_sorted # Recall: Negative mask won't do anything if apply_sign_convention=False
                local_data[positive_face_zero_indexed_ordered_indices]      = local_data_sorted

            data                                                        = local_data

        else:
            raise NotImplementedError(f"Unknown state type, {field_type}. Might need to be added to solver_variable_storage_type?")

        return data


    # region _read_field_data_from_dataset_root
    def _read_field_data_from_dataset_root(self, dset:h5py.Dataset, field_type:str, column_idx:int=None, 
                                           apply_sign_convention:bool=True) -> np.ndarray:
        """
        Root-read variant: rank 0 reads the full dataset contiguously, then each rank
        receives its local portion via MPI scatter.  Avoids HDF5 point-selection overhead
        on non-contiguous global index patterns.
        """
        face_global_indices  = self.face_global_indices
        negative_indices     = face_global_indices < 0
        negative_mask        = np.ones_like(face_global_indices, dtype=np.float64)
        negative_mask[negative_indices] = -1.0 if apply_sign_convention else 1.0
        positive_face_zero_indexed_ordered_indices = np.argsort(np.abs(face_global_indices) - 1)

        if field_type in ("volScalarStates", "modelStates"):
            if self.rank == 0:
                full = dset[:] if column_idx is None else dset[:, column_idx]
                pieces = [full[idx] for idx in self.all_cell_global_indices]
            else:
                pieces = None
            data = self._scatter_pieces(pieces)

        elif field_type == "volVectorStates":
            if self.rank == 0:
                full = dset[:] if column_idx is None else dset[:, column_idx]
                pieces = [full[idx] for idx in self.all_cell_vector_global_indices]
            else:
                pieces = None
            data = self._scatter_pieces(pieces)

        elif field_type == "surfaceScalarStates":
            if self.rank == 0:
                full = dset[:] if column_idx is None else dset[:, column_idx]
                pieces = [full[idx] for idx in self.all_face_sorted_read_indices]
            else:
                pieces = None
            dset_sorted = self._scatter_pieces(pieces)

            negative_mask_sorted = negative_mask[positive_face_zero_indexed_ordered_indices]
            if dset_sorted.ndim > 1:
                local_data = np.zeros((self.num_faces, dset_sorted.shape[1]))
                local_data[positive_face_zero_indexed_ordered_indices, :] = negative_mask_sorted[:, None] * dset_sorted
            else:
                local_data = np.zeros(self.num_faces)
                local_data[positive_face_zero_indexed_ordered_indices] = negative_mask_sorted * dset_sorted
            data = local_data

        else:
            raise NotImplementedError(f"Unknown state type, {field_type}. Might need to be added to solver_variable_storage_type?")

        return data


    # region _scatter_pieces
    def _scatter_pieces(self, pieces:List[np.ndarray]) -> np.ndarray:
        """
        Scatter a list of numpy arrays from rank 0 to all ranks using individual
        send/recv pairs rather than comm.scatter.  comm.scatter (pickle-based) packs
        all pieces into a single buffer before sending, which overflows a 32-bit int
        when the total data size exceeds ~2 GB.  Individual sends avoid that limit
        because each message is only one rank's slice.
        """
        if self.rank == 0:
            for i in range(1, self.comm_size):
                self.comm.send(pieces[i], dest=i, tag=i)
            return pieces[0]
        else:
            return self.comm.recv(source=0, tag=self.rank)
    

    # region _visualize_imported_data
    def _visualize_imported_data(self, data, input_coordinates, center_colormap=False):
        current     = {"var": "p", "snap":0, "arrows":None}
        color_map   = "coolwarm" if center_colormap else "viridis" 
        num_samples = None

        if input_coordinates.ndim > 1:
            grid_varies = True
        else:
            input_coordinates = input_coordinates.reshape(-1, 1)
            grid_varies = False

        gathered_centroid_coordinates   = self.comm.gather(input_coordinates, root=0)
        gathered_vars = {}
        for data_var, file_data in data.items():
            if self.solver_variable_storage_type[data_var] != "surfaceScalarStates":
                gathered_vars[data_var] = self.comm.gather(file_data, root=0)
                num_samples = file_data.shape[1] if num_samples is None else num_samples

        if self.rank == 0:

            # Variable slider (discrete)
            var_names = list(gathered_vars.keys())

            plt = Plotter()
            var_label = Text2D(f"Variable: {var_names[0]}", pos="top-center")
            plt += var_label

            gathered_vars_array = {}
            pts_array = np.vstack(gathered_centroid_coordinates)
            for gathered_var, gathered_var_data in gathered_vars.items():
                gathered_vars_array[gathered_var] = np.vstack(gathered_var_data)

            pts = Points(np.reshape(pts_array[:, current["snap"]], (-1, 3)))
            pts.pointdata[current["var"]] = gathered_vars_array[current["var"]][:, current["snap"]]
            pts.cmap(color_map, current["var"]).add_scalarbar()
            
            def update_plot():
                nonlocal plt
                nonlocal color_map
                var     = current["var"]
                snap    = current["snap"]

                if grid_varies:
                    coordinates  = np.reshape(pts_array[:, snap], (-1, 3))
                else:
                    coordinates  = np.reshape(pts_array, (-1, 3))
                pts.vertices = coordinates

                pts.pointdata.clear()

                if "arrows" in current:
                    plt.remove(current["arrows"])

                if self.solver_variable_storage_type[var] == "volVectorStates":
                    vec_data = np.reshape(gathered_vars_array[var][:, snap], (-1, 3))
                    vec_mag  = np.linalg.norm(vec_data, axis=1)

                    pts.pointdata[var] = vec_data
                    pts.pointdata[f'{var}mag'] = vec_mag

                    arrow_scale = 0.1/np.max(vec_mag)

                    current["arrows"] = Arrows(coordinates, coordinates + arrow_scale * vec_data, c=color_map, thickness=0.1)
                    plt += current["arrows"]

                    n_pts_per_arrow = current["arrows"].dataset.GetNumberOfPoints() // len(vec_mag)
                    repeated_mag = np.repeat(vec_mag, n_pts_per_arrow)

                    if center_colormap:
                        m = np.max(np.abs(vec_mag))
                        if m == 0:
                            m = 1e-12
                        current["arrows"].cmap(color_map, repeated_mag, vmin=-m, vmax=m)
                        current["arrows"].mapper.SetScalarRange(-m, m)
                    else:
                        current["arrows"].cmap(color_map, repeated_mag)
                        current["arrows"].mapper.SetScalarRange(vec_mag.min(), vec_mag.max())

                    current["arrows"].mapper.lookup_table.SetRange(current["arrows"].mapper.scalar_range)
                    current["arrows"].mapper.lookup_table.Build()

                    plt.remove(current["arrows"].scalarbar)
                    current["arrows"].add_scalarbar()
                    plt += current["arrows"]
                    plt += current["arrows"].scalarbar

                    plt.remove(pts.scalarbar)


                else:
                    scalar_field = gathered_vars_array[var][:, snap]
                    pts.pointdata[var] = scalar_field


                    if center_colormap:
                        m = np.max(np.abs(scalar_field))
                        if m == 0:
                            m = 1e-12
                        pts.cmap(color_map, var, vmin=-m, vmax=m)
                    else:
                        pts.cmap(color_map, var)

                    pts.mapper.lookup_table.SetRange(pts.mapper.scalar_range)
                    pts.mapper.lookup_table.Build()

                    # Remove old scalarbar and add a fresh one
                    plt.remove(pts.scalarbar)
                    pts.add_scalarbar()
                    plt += pts.scalarbar
                    
                    # Remove the arrows scalarbar if it exists
                    if current["arrows"] is not None:
                        plt.remove(current["arrows"].scalarbar)
                                        

            # Snapshot slider
            def snap_slider(widget, event):
                new_snap = int(widget.value)
                if current["snap"] != new_snap:
                    current["snap"] = new_snap
                    update_plot()

            plt.add_slider(
                snap_slider,
                xmin=0,
                xmax=num_samples - 1,
                value=0,
                title="Snapshot",
                pos=[(0.1, 0.05), (0.9, 0.05)])

            def var_slider(widget, event):
                idx = int(round(widget.value))
                new_var = var_names[idx]
                if current["var"] != new_var:
                    current["var"] = new_var
                    update_plot()
                    var_label.text(f"Variable: {current['var']}")

            plt.add_slider(
                var_slider,
                xmin=0,
                xmax=len(var_names) - 1,
                value=0,
                title="Variable",
                pos=[(0.05, 0.1), (0.05, 0.9)])
            
            plt.show(pts)

        quiet_barrier(self.comm)


# * Expected structure for the dict form of h5filepath in _compute_pod_modes:
#
# h5filepath = {
#     "data": {
#         "states": {
#             "U": float_array,
#             "p": float_array,
#             ...
#         },
#         "mesh": {
#             "cell_volumes": float_array,
#             "face_areas":   float_array,
#         },
#         "reference_states": {
#             "U": float_array,
#             "p": float_array,
#             ...
#         },
#     },
#     "metadata": {
#         "secondary_variables": {
#             "_attrs": {"first_sample_is_reference": bool}
#         },
#         "_attrs": {"num_secondary_samples": int},
#     },
# }
