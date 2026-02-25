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
                 gather_raw_files=True
                 ):
        
        # TODO: See if there is DAFoam API to get whether a variable is volVectorStates, volScalarStates, modelStates, or surfaceScalarStates.
        # We'll just define here for the rhoSimpleFoam case for now. Can add to here if necessary for other models
        self.solver_variable_storage_type = {"centroid_coordinates":    "volVectorStates", # We'll say that the cell coordinates match the volume vector state form
                                             "cell_volumes":            "volScalarStates",
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
                               "new_file":separate_pod_file}
        
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

        for primary_idx in range(adj_num_primary_samples):
            h5file_path = self.storage_location/self.dataset_keyword/f'{self.h5_file_base_name}_{primary_idx}.h5'
            
            self.initialize_h5_file(h5file_path, primary_idx)

            if self.rank == 0 and self.gather_raw_files:
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

            for secondary_idx in range(adj_num_secondary_samples):
                if rank == 0:
                    print('\n\n\n\n')
                    print('=============================================')
                    print(f'Primary sample {primary_idx+1}/{adj_num_primary_samples}, secondary sample {secondary_idx+1}/{adj_num_secondary_samples}')
                    print('=============================================\n')
                
                # Update all of the primary parameters for current point and sample
                for var, info in secondary_variables.items():
                    sim[var] = info["samples"][secondary_idx]

                # Primal solve
                sim.run()

                self.write_sample(h5file_path, secondary_idx)

                # Move OpenFOAM solution to solution directory
                if self.gather_raw_files:
                    current_directory = Path.cwd()
                    os.chdir(dafoam_directory)
                    dafoam_instance.renameSolution(9998)
                    
                    if rank == 0:
                        if comm_size > 1:
                            for i in range(comm_size):
                                shutil.move(dafoam_directory/f'processor{i}'/'0.9998/', 
                                        raw_directory/f'processor{i}'/f'{secondary_idx:04}')
                            
                                # Copy constant folder to directory (only need to do this once)
                                if primary_idx == 0 and secondary_idx == 0:
                                    shutil.copytree(dafoam_directory/f'processor{i}'/'constant', 
                                    raw_directory/f'processor{i}'/'constant')
                        else:
                            shutil.move(dafoam_directory/'0.9998', 
                                    raw_directory/f'{secondary_idx:04}')
                    
                    os.chdir(current_directory)

                else:
                    current_directory = Path.cwd()
                    os.chdir(dafoam_directory)
                    dafoam_instance.renameSolution(primary_idx * (self.num_secondary_samples + 1) + secondary_idx + 1)
                    os.chdir(current_directory)

            if compute_pod:
                self._compute_pod_modes(h5filepath=h5file_path, **pod_options)

    
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
        states                                  = dafoam_instance.getStates()
        cell_coords                             = dafoam_instance.getCellCentroids()
        state_weights                           = dafoam_instance.getStateWeights()
        state_reference_values                  = dafoam_instance.getPatchStateAverages(self.reference_patch) if self.reference_patch is not None else None
        if self.store_residuals:
            residuals                           = dafoam_instance.getResiduals()

        with h5py.File(h5filepath, "a", driver="mpio", comm=self.comm) as f:
            sample_group    = f["samples"]
            state_group     = sample_group["states"]
            ref_group       = sample_group["reference_states"]
            mesh_group      = sample_group["mesh"]

            if self.store_residuals:
                res_group   = sample_group["residuals"]

            added_cell_volumes = False
            added_face_areas   = False

            # Loop through states to update sample data
            for state_name, info in self.state_info.items():
                dset        = state_group[state_name]
                indices     = info['indices']
                state_type  = info['type']

                if self.store_residuals:
                    rset    = res_group[state_name]
                    self._write_field_data_to_dataset(rset, residuals[indices], state_type, sample_idx)

                if self.reference_patch is not None:
                    ref_group[state_name][sample_idx] = state_reference_values[state_name]

                self._write_field_data_to_dataset(dset, states[indices], state_type, sample_idx)

                # Update the cell volumes and face areas only for first time of respective addressing_type
                if state_type == "volScalarStates" and not added_cell_volumes:
                    self._write_field_data_to_dataset(mesh_group["cell_volumes"], state_weights[indices], state_type, sample_idx)
                    added_cell_volumes = True
                    
                if state_type == "surfaceScalarStates" and not added_face_areas:
                    self._write_field_data_to_dataset(mesh_group["face_areas"], state_weights[indices], state_type, sample_idx)
                    added_face_areas = True

            self._write_field_data_to_dataset(mesh_group["centroid_coordinates"], cell_coords, "volVectorStates", sample_idx)
            
            # Update scalar values
            sample_group["converged"][sample_idx]             = not self.dafoam_instance.primalFail
            sample_group.attrs["last_written_sample_index"]   = sample_idx

    
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
                        addressing_type = item.attrs["addressing_type"]
                        result[key] = self._read_field_data_from_dataset(item, addressing_type)

                    else:
                        result[key] = item[()]

                    # # Add attributes if present
                    # if len(item.attrs) > 0:
                    #     result[key] = {
                    #         "data": data,
                    #         "_attrs": dict(item.attrs),
                    #     }
                    # else:
                    #     result[key] = data

            return result
        
        with h5py.File(h5file_path, "r") as f:
            if group_to_read is None:
                return recurse(f)
            else:
                return recurse(f[group_to_read])


    #region _setup_indices_and_state_maps_and_names
    def _setup_indices_state_info_and_global_counts(self):
        self.print0('Setting up state map and processor addresing...', end=" ")

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

        # Global sizes across ranks
        self.num_cells_global                    = self.comm.allreduce(self.num_cells,                      op=MPI.SUM)
        self.num_faces_global                    = self.comm.allreduce(self.num_faces,                      op=MPI.SUM)
        self.num_faces_no_proc_boundaries_global = self.comm.allreduce(self.num_faces_no_proc_boundaries,   op=MPI.SUM)
        self.num_state_elements_global           = self.comm.allreduce(self.num_state_elements,             op=MPI.SUM)

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
        for dict in var_limits.values():
            if "ref_value" not in dict:
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


    # region _compute_pod_modes
    def _compute_pod_modes(self, h5filepath, inner_product=None, centering='mean', scaling="reference", new_file=True, write_modes_using_write_adjoint_fields=False):

        # data, metadata  = self.read_h5_file(h5filepath=h5filepath, return_metadata=True, visualize_data=False)
        data_dict       = self.load_h5(h5file_path=h5filepath, group_to_read="samples", only_distributed_data=False)
        metadata        = self.load_h5(h5file_path=h5filepath, group_to_read="parameters", only_distributed_data=False)

        reference_state     = {}
        weights             = {}
        scaling_values      = {}

        for state_var, info in self.state_info.items():
            state_data = data_dict["states"][state_var]
            state_type = info["type"]

            # Centering methods (centering)
            # 1. (None) No centering - will keep states as they are
            # 2. ('mean') Mean centered. Will take the mean of the snapshots and subtract that from each snapshot
            # 3. ('reference') Will use the reference sample in the file if it exists
            # 4. (dict) User supplied reference
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

            elif isinstance(centering, dict):
                # TODO: Add check to see if keys and dimensions match up
                reference_state[state_var] = centering[state_var]

            else:
                raise TypeError(f'"{centering}" is not a valid centering method. ')
        
            # Weighting methods (inner_product)
            # 1. (None) No weighting applied. Will use ones
            # 2. ("reference") Generate the weights from the cell volumes and face areas of the first sample
            # 4. (dict) User supplied weights
            if inner_product is None:
                weights = None
        
            elif isinstance(inner_product, str):        
                if inner_product == "reference":
                    # We only consider the weights for the reference state (first sample) for now
                    if state_type == "volVectorStates":
                        weights[state_var] = np.repeat(data_dict["mesh"]['cell_volumes'][:, 0], repeats=3, axis=0)
                    elif state_type == 'volScalarStates' or state_type == "modelStates":
                        weights[state_var] = data_dict["mesh"]['cell_volumes'][:, 0]
                    elif state_type == "surfaceScalarStates":
                        weights[state_var] = data_dict["mesh"]['face_areas'][:, 0]
                    else:
                        raise TypeError(f"State type of {state_type} not recognized. May need to be implemented?")
                        
                else:
                    raise NotImplementedError(f"Inner product weight of type {inner_product} has not yet been implemented")
            
            elif isinstance(inner_product, dict):
                # TODO: Add check to see if keys and dimensions match up
                weights[state_var] = inner_product[state_var]
                
            else:
                assert np.asarray(inner_product).shape == np.asarray(reference_state[state_var]).shape, "Supplied inner_product weight vector seems to be incompatible?"
            
            # Scaling methods (scaling)
            # 1. (None) No scaling applied to the data. Terms like pressure will most likely dominate
            # 2. ('reference') Will use the reference value found in the file (if it exists)
            # 3. (dict) Will use the supplied scaling value. Must be a dictionary whose keys are state names (as seen by OpenFOAM - 'p', 'T', etc), and entries are scalar values
            if scaling is None:
                scaling_values[state_var] = np.ones_like(centering[state_var])

            elif scaling == 'reference':
                if "reference_states" in data_dict.keys():
                    reference_states = data_dict["reference_states"]
                    if f'{state_var}' in reference_states:
                        if state_var != "phi":
                            scaling_values[state_var] = reference_states[state_var][0]
                        else:
                            scaling_values[state_var] = 1 #data["face_areas"][:, 0] # Use face areas for phi weighting
                    else:
                        raise TypeError(f'Reference value not found for {state_var} in dataset during POD compute setup.')
                else:
                    raise TypeError(f'Reference values not found in dataset during POD compute setup.')
                    
            elif isinstance(scaling, dict):
                # TODO: Add check to see if keys and dimensions match up  
                scaling_values[state_var] = scaling[state_var]
                    
            else:
                raise TypeError("Not a valid scaling method. Please supply None, 'reference', or a dict with the proper entries.")
            
            # Rescaling data for POD computation
            # if state_var == "phi":
            #     data[state_var] = 1/scaling_values[state_var][:, None] * (data[state_var] - reference_state[state_var][:, None])
            # else:
            data_dict["states"][state_var] = 1/scaling_values[state_var] * (data_dict["states"][state_var] - reference_state[state_var][:, None])
        
        # Only need state data and number of samples for POD computation
        data                = data_dict["states"]

        # Number of sample correction
        num_samples         = metadata["_attrs"]["num_secondary_samples"]
        data["num_samples"] = num_samples - 1 if centering == "reference" else num_samples

        # Actual POD computation
        local_modes, singular_values = self._method_of_snapshots(data, weights)

        # Change file path name if new file requested
        outfilepath = Path(h5filepath)
        if new_file:
            outfilepath = outfilepath.with_name(outfilepath.stem + '_modes' + outfilepath.suffix)

        with h5py.File(outfilepath, "a", driver="mpio", comm=self.comm) as f:
            pod_group       = f.create_group("pod")
            mode_group      = pod_group.create_group("modes")
            reference_group = pod_group.create_group("reference_state")
            if weights is not None:
                weights_group   = pod_group.create_group("weights")
            scaling_group   = pod_group.create_group("scaling")

            for state_var, info in self.state_info.items():
                state_type  = info["type"]
                num_modes   = local_modes[state_var].shape[1]
                if      state_type == "volScalarStates" or state_type == "modelStates": num_rows = self.num_cells_global
                elif    state_type == "volVectorStates":                                num_rows = 3 * self.num_cells_global
                elif    state_type == "surfaceScalarStates":                            num_rows = self.num_faces_no_proc_boundaries_global

                mode_group.create_dataset(state_var,        (num_rows, num_modes),              dtype="f8")
                self._write_field_data_to_dataset(mode_group[state_var], local_modes[state_var], state_type)
                mode_group[state_var].attrs.create("addressing_type", state_type)

                reference_group.create_dataset(state_var,   (num_rows, ),                       dtype="f8")
                self._write_field_data_to_dataset(reference_group[state_var], reference_state[state_var], state_type)
                reference_group[state_var].attrs.create("addressing_type", state_type)

                if weights is not None:
                    weights_group.create_dataset(state_var, (num_rows, ),                       dtype="f8")
                    self._write_field_data_to_dataset(weights_group[state_var], weights[state_var], state_type)
                    weights_group[state_var].attrs.create("addressing_type", state_type)
                
                if scaling is not None:
                    scaling_group.create_dataset(state_var,     data=scaling_values[state_var], dtype="f8")

            pod_group.create_dataset('singular_values',     data=singular_values,               dtype="f8")

        if write_modes_using_write_adjoint_fields:
            for i in range(singular_values.size):
                state_vec = np.concatenate([local_modes[state_var][:, i] for state_var in self.state_info.keys()])
                self.dafoam_instance.solver.writeAdjointFields("pod_mode", i+1, state_vec)
    

    # region _method_of_snapshots
    def _method_of_snapshots(self, local_data, local_weights):
        comm = self.comm
        rank = self.rank

        # Method of snapshots (referenced https://willcox-research-group.github.io/rom-operator-inference-Python3/_modules/opinf/basis/_pod.html#method_of_snapshots)
        min_thresh      = 1e-15
        if isinstance(local_data, dict):
            n_snapshots     = local_data["num_samples"]
            local_gramian   = np.zeros((n_snapshots, n_snapshots))
            for state_var in self.state_info.keys():
                if local_weights is None:
                    local_gramian += local_data[state_var].T @ local_data[state_var]
                else:   
                    local_gramian += local_data[state_var].T @ (local_weights[state_var][:, None] * local_data[state_var])   
        else:
            n_snapshots     = local_data.shape[1]
            if local_weights is None:
                local_gramian   = local_data.T @ local_data
            else:
                local_gramian   = local_data.T @ (local_weights[:, None] * local_data)

        total_gramian   = np.zeros_like(local_gramian) if rank == 0 else None
        comm.Reduce(local_gramian, total_gramian, op=MPI.SUM, root=0)

        if rank == 0:

            # DEBUG
            print(f"Total Gramian diagonal: {np.diag(total_gramian)}")
            print(f"Total Gramian sum: {np.sum(total_gramian)}")
            ########

            eigvals, eigvecs = np.linalg.eigh(total_gramian)

            # Re-order (largest to smallest).
            eigvals     = eigvals[::-1]
            eigvecs     = eigvecs[:, ::-1]

            # By definition the Gramian is symmetric positive semi-definite.
            # If any eigenvalues are smaller than zero, they are only measuring
            # numerical error and can be truncated.
            positives   = eigvals > max(min_thresh, abs(np.min(eigvals)))
            eigvecs     = eigvecs[:, positives]
            eigvals     = eigvals[positives]
            s_vals      = np.sqrt(eigvals) # * n_global_states)

        quiet_barrier(comm)

        # Broadcast eigenvalues/vectors and singular values
        if rank == 0:
            n_retained_modes = eigvals.size
        else:
            n_retained_modes = None
        n_retained_modes = comm.bcast(n_retained_modes, root=0)

        # ALL ranks need to allocate buffers, including rank 0!
        eigvals_bcast = np.empty((n_retained_modes, ),                dtype=np.float64)
        eigvecs_bcast = np.empty((n_snapshots, n_retained_modes),     dtype=np.float64)
        s_vals_bcast  = np.empty((n_retained_modes, ),                dtype=np.float64)

        # Copy from rank 0's computed values into broadcast buffer
        if rank == 0:
            eigvals_bcast[:] = eigvals
            eigvecs_bcast[:] = eigvecs
            s_vals_bcast[:]  = s_vals

        comm.Bcast([eigvals_bcast, MPI.DOUBLE], root=0)
        comm.Bcast([eigvecs_bcast, MPI.DOUBLE], root=0)
        comm.Bcast([s_vals_bcast,  MPI.DOUBLE], root=0)

        # Use the broadcast versions
        eigvals = eigvals_bcast
        eigvecs = eigvecs_bcast
        s_vals = s_vals_bcast

        # DEBUG
        print(f"Rank {self.rank}: First eigenvector sum = {np.sum(eigvecs[:, 0])}")
        print(f"Rank {self.rank}: First eigenvalue = {eigvals[0]}")
        ###

        # Rescale and square root eigenvalues to get singular values.
        if isinstance(local_data, dict):
            local_modes = {}
            for state_var in self.state_info.keys():
                local_modes[state_var] = local_data[state_var] @ (eigvecs / s_vals)
        else:
            local_modes = local_data @ (eigvecs / s_vals)

        return local_modes, s_vals


    # region _write_field_data_to_dataset
    def _write_field_data_to_dataset(self, dset, data, field_type, column_idx=None):

        cell_global_indices                     = self.cell_global_indices
        cell_vector_global_indices              = self.cell_vector_global_indices
        face_masked_sorted_global_indices       = self.face_masked_sorted_global_indices
        face_masked_sorting_indices             = self.face_masked_sorting_indices
        face_proc_boundary_mask                 = self.face_proc_boundary_mask
        
        if field_type == "volScalarStates" or field_type == "modelStates":
            if column_idx is None:
                dset[cell_global_indices]                           = data
            else:
                dset[cell_global_indices, column_idx]               = data

        elif field_type == "volVectorStates":
            if column_idx is None:
                dset[cell_vector_global_indices]                    = data
            else:
                dset[cell_vector_global_indices, column_idx]        = data

        elif field_type == "surfaceScalarStates":
            if column_idx is None:
                dset[face_masked_sorted_global_indices]             = data[face_proc_boundary_mask][face_masked_sorting_indices]
            else:
                dset[face_masked_sorted_global_indices, column_idx] = data[face_proc_boundary_mask][face_masked_sorting_indices]

        else:
            raise NotImplementedError(f"Unknown state type, {field_type}. Might need to be added to solver_variable_storage_type?")
        

    # region _read_field_data_from_dataset
    def _read_field_data_from_dataset(self, dset, field_type, column_idx=None):

        cell_global_indices                     = self.cell_global_indices
        cell_vector_global_indices              = self.cell_vector_global_indices
        face_global_indices                     = self.face_global_indices

        # Have to recover the proper mapping for the faces
        negative_indices                            = self.face_global_indices < 0
        positive_face_global_indices_zero_indexed   = (np.abs(face_global_indices) - 1)
        negative_mask                               = np.ones_like(face_global_indices)
        negative_mask[negative_indices]             = -1

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
            
            if dset.ndim > 1:
                dset_sorted                                             = dset[positive_face_global_indices_zero_indexed[positive_face_zero_indexed_ordered_indices], :]
            else:
                dset_sorted                                             = dset[positive_face_global_indices_zero_indexed[positive_face_zero_indexed_ordered_indices]]
            negative_mask_sorted                                        = negative_mask[positive_face_zero_indexed_ordered_indices]

            if dset.ndim > 1:
                local_data_sorted                                           = negative_mask_sorted[:, None] * dset_sorted 
                local_data[positive_face_zero_indexed_ordered_indices, :]   = local_data_sorted
            else:
                local_data_sorted                                           = negative_mask_sorted * dset_sorted 
                local_data[positive_face_zero_indexed_ordered_indices]      = local_data_sorted

            data                                                        = local_data

        else:
            raise NotImplementedError(f"Unknown state type, {field_type}. Might need to be added to solver_variable_storage_type?")

        return data