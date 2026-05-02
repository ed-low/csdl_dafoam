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
from csdl_dafoam.utils.standard_atmosphere_model import compute_ambient_conditions_group
from csdl_dafoam.utils.decompositions import method_of_snapshots_distributed
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors


# region TRAININGDATAINTERFACE
class TrainingDataInterface():  
    def __init__(self,
                 dafoam_instance,
                 storage_location,
                 dataset_keyword,
                 primary_variables=None,
                 secondary_variables=None,
                 non_sampled_variables=None,
                 csdl_simulator=None,
                 reference_patch=None,
                 num_primary_samples=2,
                 num_secondary_samples=20,
                 random_state_seed=0,
                 store_residuals=False,
                 h5_file_base_name="point",
                 gather_raw_files=True,
                 parallel_write=True,
                 parallel_read=True,
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

        # Create directory
        self.print0('Creating storage directory...')
        if self.rank == 0:
            os.makedirs(storage_location/dataset_keyword, exist_ok = True)


    # region sample_variables
    def sample_variables(self, print_sampled_values=True, random_state_seed=None):
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
    def run_sweep(self, compute_pod=True, separate_pod_file=False, pod_options=None):
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

        # Check for an interrupted sweep to resume from
        resume_primary_idx, resume_secondary_start, skip_h5_init = self._check_for_interrupted_sweep(adj_num_secondary_samples)

        for primary_idx in range(adj_num_primary_samples):
            h5file_path = self.storage_location/self.dataset_keyword/f'{self.h5_file_base_name}_{primary_idx}.h5'

            # Skip primary indices already fully written before the resume point
            if resume_primary_idx is not None and primary_idx < resume_primary_idx:
                self.print0(f'Skipping primary index {primary_idx} (already complete).')
                continue

            # is_first_resumed: True only for the very first primary index we actually process
            is_first_resumed = (resume_primary_idx is not None and primary_idx == resume_primary_idx)
            secondary_start  = resume_secondary_start if is_first_resumed else 0

            # skip_h5_init is True only when resuming mid-file (file already exists on disk).
            # When the previous primary completed fully and we're starting a fresh file for the
            # next primary, skip_h5_init is False and the file must be initialised normally.
            if not (is_first_resumed and skip_h5_init):
                self.initialize_h5_file(h5file_path, primary_idx)

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

                # We'll do a check to see if the first primal solve of the primary point failed
                # If so, we'll retry by running with a new initial condition taken from
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

                # TODO: Add a derivative computation here

                self.write_sample(h5file_path, secondary_idx)

                # Move OpenFOAM solution to solution directory
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

            if compute_pod:
                self._compute_pod_modes(h5filepath=h5file_path, **pod_options)

    
    # region _check_for_interrupted_sweep
    def _check_for_interrupted_sweep(self, adj_num_secondary_samples):
        """
        Scan the storage directory for the largest-indexed h5 data file and determine
        where the sweep should resume.  Two cases are handled:

          - Incomplete file: the largest file exists but has not had all secondary
            samples written.  Resume mid-file; the h5 file and raw directory already
            exist, so their initialisation must be skipped.

          - Complete file, next primary missing: the largest file is fully written but
            the next primary index has no h5 file.  Resume at the start of that next
            primary index; the h5 file and raw directory must be created fresh.

        Rank 0 performs all file I/O; the result is broadcast to every rank.

        Returns
        -------
        (resume_primary_idx, resume_secondary_start, skip_h5_init)
            resume_primary_idx      - first primary index that still needs work, or None
                                      if no existing files were found / files don't match.
            resume_secondary_start  - first secondary index to run for that primary (0 when
                                      starting a brand-new primary file).
            skip_h5_init            - True only when resuming mid-file (the h5 file already
                                      exists and must not be re-initialised).
        """
        self.print0("Checking for interrupted sweep...")

        result = (None, 0, False)

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
                            var_name = info["name"]
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
                                var_name = info["name"]
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
                                # Case: incomplete file — resume mid-secondary loop, skip h5 init
                                resume_secondary_start = last_written + 1
                                print(f"\nFound incomplete sweep file: {largest_file.name}")
                                print(f"  Last written secondary index: {last_written} / {total_snapshots - 1}")
                                print(f"  Resuming from primary index {largest_primary_idx}, secondary index {resume_secondary_start}.\n")
                                result = (largest_primary_idx, resume_secondary_start, True)
                            else:
                                # Case: file complete but next primary not started — create it fresh
                                next_primary_idx = largest_primary_idx + 1
                                print(f"\nAll secondary samples in {largest_file.name} are complete.")
                                print(f"  Resuming at the start of primary index {next_primary_idx}.\n")
                                result = (next_primary_idx, 0, False)

                except Exception as e:
                    print(f"Warning: could not read existing h5 file for resume check: {e}")

        result = self.comm.bcast(result, root=0)
        return result


    # region _find_existing_raw_directory
    def _find_existing_raw_directory(self, primary_idx):
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
    def _cleanup_openfoam_for_resume(self, dafoam_directory):
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
    def initialize_h5_file(self, h5filepath, primary_idx):
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

            mesh_group.create_dataset("centroid_coordinates",     (3 * num_cells_global, adj_num_secondary_samples),                dtype="f8")
            mesh_group.create_dataset("cell_volumes",             (num_cells_global, adj_num_secondary_samples),                    dtype="f8")
            mesh_group.create_dataset("face_areas",               (num_faces_no_proc_boundaries_global, adj_num_secondary_samples), dtype="f8") 
            # The following two datasets are for debugging
            mesh_group.create_dataset("cell_indices",             (num_cells_global, ),                                             dtype="i8")
            mesh_group.create_dataset("face_indices",             (num_faces_no_proc_boundaries_global, ),                          dtype="i8")

            self._write_field_data_to_dataset(mesh_group["cell_indices"],  self.cell_global_indices, "volScalarStates")
            self._write_field_data_to_dataset(mesh_group["face_indices"], self.face_global_indices, "surfaceScalarStates")

            mesh_group["centroid_coordinates"].attrs.create("addressing_type",  "volVectorStates")
            mesh_group["cell_volumes"].attrs.create("addressing_type",          "volScalarStates")
            mesh_group["face_areas"].attrs.create("addressing_type",            "surfaceScalarStates")

            sample_group.attrs.create("last_written_sample_index",          data=-1,                                dtype="i8")
            sample_group.attrs.create("generated_on_n_processors",          data=self.comm_size,                    dtype="i8")
            sample_group.attrs.create("num_cells",                          data=num_cells_global,                  dtype="i8")     

            sample_group.create_dataset("converged",                        (adj_num_secondary_samples, ),          dtype="bool")  

            parameter_group.attrs.create("num_primary_samples",             data=adj_num_primary_samples,           dtype="i8")
            parameter_group.attrs.create("sample_number",                   data=primary_idx,                       dtype="i8")
            parameter_group.attrs.create("num_secondary_samples",           data=adj_num_secondary_samples,         dtype="i8")
            parameter_group.attrs.create("random_state_seed",               data=self.random_state_seed,            dtype="f8")

            primary_var_group = parameter_group.create_group("primary_variables")
            for info in self.primary_variables.values():
                primary_var_group.create_dataset(info["name"],              data=info["samples"][primary_idx],      dtype="f8")

            secondary_var_group = parameter_group.create_group("secondary_variables")
            for info in self.secondary_variables.values():
                secondary_var_group.create_dataset(info["name"],            data=info["samples"],                   dtype="f8")
            secondary_var_group.attrs.create("first_sample_is_reference",   data=self.secondary_has_ref,            dtype="bool")
            
            if self.non_sampled_variables is not None:
                non_sampled_var_group = parameter_group.create_group("non_sampled_variables")
                for var, info in self.non_sampled_variables.items():
                    non_sampled_var_group.create_dataset(info["name"],      data=var.value,                         dtype="f8")

        self.print0('All set!')

    
    # region write_sample
    def write_sample(self, h5filepath, sample_idx):
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
    def _write_sample_parallel(self, h5filepath, sample_idx, states, cell_coords,
                                state_weights, state_reference_values, residuals):
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
    def _write_sample_root(self, h5filepath, sample_idx, states, cell_coords,
                           state_weights, state_reference_values, residuals):
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


    # region _gather_field_data
    def _gather_field_data(self, states, cell_coords, state_weights, residuals=None):
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
    def load_h5(self, h5file_path, group_to_read=None, only_distributed_data=False):
        
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
        self.print0('Setting up state map and processor addressing...', end=" ")

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
                        sub-dictionary (along with name, range) which contains a (num_samples, variable_shape) array of samples
                    csdl_var: {
                        'name': str,           # Variable name for labeling
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
            name        = spec['name']
            var_range   = spec['range']
            
            if len(var_range) != 2:
                raise ValueError(f"{name}: range must be [min, max], got {var_range}")
            
            shape        = var.value.shape
            num_elements = int(np.prod(shape)) if shape else 1
            
            # Add one xlimit row per element
            xlimits.extend([var_range] * num_elements)
            
            # Store metadata for reconstruction
            var_metadata.append({
                'var': var,
                'name': name,
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
    def _read_proc_addressing(self, key="cell"):
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
    def _compute_pod_modes(self, h5filepath, inner_product=None, centering='mean', scaling="reference", write_h5=True, new_h5_file=True, overwrite_datasets=False, new_file_suffix="modes", write_modes_using_write_adjoint_fields=True):
        
        # Accepts either an h5 file path or a pre-loaded dict with keys "data" and "metadata"
        # (internal shortcut used by _leave_one_out_test; see * at end of file for expected structure)
        if isinstance(h5filepath, dict):
            data_dict   = h5filepath["data"]
            metadata    = h5filepath["metadata"]
        else:
            data_dict       = self.load_h5(h5file_path=h5filepath, group_to_read="samples", only_distributed_data=False)
            metadata        = self.load_h5(h5file_path=h5filepath, group_to_read="parameters", only_distributed_data=False)

        reference_state, weights, scaling_values = self._build_pod_inputs(data_dict, metadata, centering, inner_product, scaling)

        # Only need state data and number of samples for POD computation
        data_array    = np.concatenate([data_dict["states"][state_var] for state_var in self.state_info.keys()], axis=0)
        weights_array = np.concatenate([weights[state_var] for state_var in self.state_info.keys()], axis=0)

        # Actual POD computation
        modes_array, singular_values = method_of_snapshots_distributed(matrix_local=data_array,
                                                                       comm=self.comm, method="tsqr",
                                                                       weights_local=weights_array)
        
        local_modes = {state_name:modes_array[info["indices"], :] for state_name, info in self.state_info.items()}

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

                for state_var, info in self.state_info.items():
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
                        dset = scaling_group.require_dataset(state_var, shape=(1,), dtype="f8")
                        dset[...] = scaling_values[state_var]
                
                if overwrite_datasets and 'singular_values' in pod_group:
                    del pod_group['singular_values']
                dset = pod_group.require_dataset('singular_values',     shape=singular_values.shape,               dtype="f8")
                dset[...] = singular_values

        if write_modes_using_write_adjoint_fields:

            for i in range(singular_values.size):

                leading_integer         = 2
                solution_write_number   = leading_integer + (i + 1) / 10000

                # Write the mode
                self.dafoam_instance.solver.writeAdjointFields("pod_mode_", 
                                                               solution_write_number, 
                                                               np.concatenate([local_modes[state_var][:, i] for state_var in self.state_info.keys()], axis=0), 
                                                               True)

                # Write the mesh
                mesh = np.zeros_like(self.dafoam_instance.xv.flatten())
                self.dafoam_instance.solver.getOFMeshPoints(mesh)
                self.dafoam_instance.solver.writeMeshPoints(mesh, solution_write_number)

        return local_modes, reference_state, weights, scaling_values


    # region _build_pod_inputs
    def _build_pod_inputs(self, data_dict, metadata, centering, inner_product, scaling):
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
                            weights[state_var] = np.abs(data_dict["mesh"]['face_areas'][:, 0])
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
            # nuTilda is over-scaled by 1000x to reduce its contribution to POD mode energy.
            # phi uses a derived velocity-pressure scale rather than its own patch average.
            if scaling is None:
                scaling_values[state_var] = np.ones_like(reference_state[state_var])
            elif scaling == 'reference':
                if "reference_states" not in data_dict:
                    raise TypeError('Reference values not found in dataset during POD compute setup.')
                reference_states = data_dict["reference_states"]
                if state_var not in reference_states:
                    raise TypeError(f'Reference value not found for {state_var} in dataset during POD compute setup.')
                if state_var == "nuTilda":
                    scaling_values[state_var] = 1000 * reference_states[state_var][0]
                elif state_var == "phi":
                    scaling_values[state_var] = (reference_states["p"][0] / reference_states["T"][0] / 287.
                                                 * reference_states["U"][0])
                else:
                    scaling_values[state_var] = reference_states[state_var][0]
            elif isinstance(scaling, dict):
                scaling_values[state_var] = scaling[state_var]
            else:
                raise TypeError("Not a valid scaling method. Please supply None, 'reference', or a dict.")

            data_dict["states"][state_var] = (1 / scaling_values[state_var]
                                              * (data_dict["states"][state_var] - reference_state[state_var][:, None]))

        return reference_state, weights, scaling_values


    # region _leave_one_out_test
    def _leave_one_out_test(self, h5filepath, num_modes=None, pod_options={}):
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

            local_modes, reference_state, weights, scaling_values = self._compute_pod_modes(supply_dict, **pod_options, write_h5=False)

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
    def _write_field_data_to_dataset(self, dset, data, field_type, column_idx=None):

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
    def _read_field_data_from_dataset(self, dset, field_type, column_idx=None, apply_sign_convention=True):
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
    def _read_field_data_from_dataset_root(self, dset, field_type, column_idx=None, apply_sign_convention=True):
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
    def _scatter_pieces(self, pieces):
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
