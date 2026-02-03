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


# region TRAININGDATAINTERFACE
class TrainingDataInterface():  
    def __init__(self, 
                 dafoam_instance,  
                 storage_location,
                 dataset_keyword,
                 primary_variables=None, 
                 secondary_variables=None,
                 csdl_simulator=None,
                 num_primary_samples=2,
                 num_secondary_samples=20,
                 random_state_seed=0,
                 h5_file_base_name="point"
                 ):
        
        # TODO: See if there is DAFoam API to get whether a variable is volVectorStates, volScalarStates, modelStates, or surfaceScalarStates.
        # We'll just define here for the rhoSimpleFoam case for now. Can add to here if necessary for other models
        self.solver_variable_storage_type = {"centroid_coordinates":    "volVectorStates", # We'll say that the cell coordinates match the volume vector state form
                                             "U":                       "volVectorStates",
                                             "p":                       "volScalarStates",
                                             "T":                       "volScalarStates",
                                             "nuTilda":                 "modelStates",
                                             "phi":                     "surfaceScalarStates"}
                
        self.dafoam_instance            = dafoam_instance
        self.csdl_simulator             = csdl_simulator
        self.primary_variables          = primary_variables
        self.secondary_variables        = secondary_variables
        self.storage_location           = Path(storage_location)
        self.dataset_keyword            = dataset_keyword
        self.num_primary_samples        = num_primary_samples
        self.num_secondary_samples      = num_secondary_samples
        self.random_state_seed          = random_state_seed
        self.h5_file_base_name          = h5_file_base_name

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
    def run_sweep(self):
        sim                         = self.csdl_simulator
        rank                        = self.rank
        comm_size                   = self.comm_size
        dafoam_instance             = self.dafoam_instance
        primary_variables           = self.primary_variables
        secondary_variables         = self.secondary_variables
        dafoam_directory            = Path(dafoam_instance.run_directory)

        if sim is None:
            self.print0('Must supply the associated CSDL simulator to the TrainingInterface before calling run_sweep.')
            return

        # Throw error if LHS hasn't been run yet
        if not self.ran_sampling:
            self.print0("Must run sample_variables before run_sweep.")
            return

        # Adjust sample size to accomodate refrence values if present
        adj_num_primary_samples   = self.num_primary_samples   + self.primary_has_ref
        adj_num_secondary_samples = self.num_secondary_samples + self.secondary_has_ref

        for primary_idx in range(adj_num_primary_samples):
            h5file_path = self.storage_location/self.dataset_keyword/f'{self.h5_file_base_name}_{primary_idx}.h5'
            
            self.initialize_h5_file(h5file_path, primary_idx)

            print(f'Rank {rank} done initializing!')

            if self.rank == 0:
                # Make a folder for the raw OpenFOAM save
                raw_directory = self.storage_location/self.dataset_keyword/f'{self.h5_file_base_name}_{primary_idx}_raw'
                os.makedirs(raw_directory, exist_ok = True)

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

    
    # region initialize_h5_file
    def initialize_h5_file(self, h5filepath, primary_idx):
        comm = self.comm

        # Global sizes across ranks
        num_cells_global = self.num_cells_global

        # Adjust sample size to accomodate refrence values if present
        adj_num_secondary_samples = self.num_secondary_samples + self.secondary_has_ref

        self.print0(f'Setting up {h5filepath} for writing...')

        with h5py.File(h5filepath, "a", driver="mpio", comm=comm) as f:
            data_group = f.create_group("sample_data")
            for state_name, info in self.state_info.items():
                state_type = info["type"]
                
                # Determine size based on what kind of quantity
                if      state_type == "volScalarStates" or state_type == "modelStates": size = self.num_cells_global
                elif    state_type == "volVectorStates":                                size = 3 * self.num_cells_global
                elif    state_type == "surfaceScalarStates":                            size = self.num_faces_no_proc_boundaries_global
                else: 
                    raise NotImplementedError(f"Unknown state type, {state_type}. Might need to be added to solver_variable_storage_type?")
                
                data_group.create_dataset(state_name,                       (size, adj_num_secondary_samples),                  dtype="f8")

            data_group.create_dataset("centroid_coordinates",           (3 * num_cells_global,   adj_num_secondary_samples),    dtype="f8")
            
            metadata_group = f.create_group("metadata")
            metadata_group.create_dataset("last_written_sample_index",  (),                                 dtype="i8")
            metadata_group.create_dataset("first_sample_is_reference",  data=self.secondary_has_ref,        dtype="bool")
            metadata_group.create_dataset("generated_on_n_processors",  data=self.comm_size,                dtype="i8")
            metadata_group.create_dataset("num_cells",                  data=num_cells_global,              dtype="i8")
            metadata_group.create_dataset("converged",                  (adj_num_secondary_samples,),       dtype="bool")
            metadata_group.create_dataset("random_state_seed",          data=self.random_state_seed,        dtype="f8")
            metadata_group.create_dataset("num_secondary_samples",      data=adj_num_secondary_samples,     dtype="i8")

            primary_sample_group = f.create_group("primary_sample")
            for info in self.primary_variables.values():
                primary_sample_group.create_dataset(info["name"],       data=info["samples"][primary_idx], dtype="f8")

            secondary_sample_group = f.create_group("secondary_samples")
            for info in self.secondary_variables.values():
                secondary_sample_group.create_dataset(info["name"],     data=info["samples"],              dtype="f8")

    
    # region write_sample
    def write_sample(self, h5filepath, sample_idx):
        self.print0('Adding sample')
        states                                  = self.dafoam_instance.getStates()
        cell_coords                             = self.dafoam_instance.getCellCentroids()
        cell_global_indices                     = self.cell_global_indices
        cell_vector_global_indices              = self.cell_vector_global_indices
        face_masked_sorted_global_indices       = self.face_masked_sorted_global_indices
        face_masked_sorting_indices             = self.face_masked_sorting_indices
        face_proc_boundary_mask                 = self.face_proc_boundary_mask

        with h5py.File(h5filepath, "a", driver="mpio", comm=self.comm) as f:
            data_group      = f["sample_data"]
            metadata_group  = f["metadata"]

            # Loop through states to update sample data
            for state_name, info in self.state_info.items():
                dset        = data_group[state_name]
                indices     = info['indices']
                state_type  = info['type']

                if state_type == "volScalarStates" or state_type == "modelStates":
                    dset[cell_global_indices, sample_idx]                   = states[indices]
                elif state_type == "volVectorStates":
                    dset[cell_vector_global_indices, sample_idx]            = states[indices]
                elif state_type == "surfaceScalarStates":
                    dset[face_masked_sorted_global_indices, sample_idx]     = states[indices[face_proc_boundary_mask][face_masked_sorting_indices]]
                else:
                    raise NotImplementedError(f"Unknown state type, {state_type}. Might need to be added to solver_variable_storage_type?")
                    
            # Update centroid coordinate data
            data_group["centroid_coordinates"][cell_vector_global_indices, sample_idx] = cell_coords
            
            # Update scalar values
            if self.rank == 0:
                metadata_group["converged"][sample_idx]         = not self.dafoam_instance.primalFail
                metadata_group["last_written_sample_index"][()] = sample_idx

            quiet_barrier(self.comm)

    
    # region read_h5_file
    def read_h5_file(self, h5filepath, return_metadata=False, visualize_data=False):
        cell_global_indices         = self.cell_global_indices
        cell_vector_global_indices  = self.cell_vector_global_indices
        face_global_indices         = self.face_global_indices

        # Have to recover the proper mapping for the faces
        negative_indices = face_global_indices < 0
        positive_face_global_indices_zero_indexed = (np.abs(face_global_indices) - 1)
        negative_mask                   = np.ones_like(face_global_indices)
        negative_mask[negative_indices] = -1

        positive_face_zero_indexed_ordered_indices = np.argsort(positive_face_global_indices_zero_indexed)

        data = {}
        meta = {}

        with h5py.File(h5filepath, "r", driver="mpio", comm=self.comm) as f:
            data_grp    = f["sample_data"]
            meta_grp    = f["metadata"]

            num_samples = int(meta_grp["num_secondary_samples"][()])

            for data_var, file_data in data_grp.items():            
                state_type = self.solver_variable_storage_type[data_var]
                
                if state_type == "volScalarStates" or state_type == "modelStates":
                    data[data_var] = file_data[cell_global_indices]

                elif state_type == "volVectorStates":
                    data[data_var] = file_data[cell_vector_global_indices]

                elif state_type == "surfaceScalarStates":
                    local_data     = np.zeros((self.num_faces_global, num_samples))
                    local_data[positive_face_zero_indexed_ordered_indices, :] = negative_mask[positive_face_zero_indexed_ordered_indices, None] * file_data[positive_face_global_indices_zero_indexed[positive_face_zero_indexed_ordered_indices], :]
                    data[data_var] = local_data

                else:
                    raise NotImplementedError(f"Unknown state type, {state_type}. Might need to be added to solver_variable_storage_type?")
                
            if return_metadata:
                for meta_var, file_meta in meta_grp.items():
                    meta[meta_var] = file_meta
        
        if visualize_data:
            self._visualize_imported_data(data, num_samples)
            
        if return_metadata:
            return data, meta
        else:
            return data


    #region _setup_indices_and_state_maps_and_names
    def _setup_indices_state_info_and_global_counts(self):
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

        # Hadnling processor boundary faces, as well as non-increasing indices (this would be an issue for hdf5)
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
    def print0(self, statement):
        if self.rank == 0:
            print(statement)


    # region _read_proc_addressing
    def _read_proc_addressing(self, key="cell"):
        key = key.lower()
        if key not in ["boundary", "cell", "face", "point"]:
            raise ValueError( f'{key} does not have an associated ProcAddressing. Please specify "boundary", "cell", "face", or "point".')

        run_directory = Path(self.dafoam_instance.run_directory)
        filename      = run_directory/f'processor{self.rank}'/'constant'/'polyMesh'/f'{key}ProcAddressing.gz'

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
    def _visualize_imported_data(self, data, num_samples):
        current     = {"var": "p", "snap":0, "arrows":None}
        color_map   = "viridis"

        gathered_centroid_coordinates   = self.comm.gather(data["centroid_coordinates"], root=0)
        gathered_vars = {}
        for data_var, file_data in data.items():
            if data_var != "centroid_coordinates" and self.solver_variable_storage_type[data_var] != "surfaceScalarStates":
                gathered_vars[data_var] = self.comm.gather(file_data, root=0)

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

                coordinates  = np.reshape(pts_array[:, snap], (-1, 3))
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
                else:
                    pts.pointdata[var] = gathered_vars_array[var][:, snap]

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