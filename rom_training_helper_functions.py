import numpy as np
from typing import Dict, Tuple, Union, Iterable
from mpi4py import MPI
import h5py
import numpy as np
from petsc4py import PETSc



def write_snapshot(
    filename,
    states: PETSc.Vec,
    snapshot_index: int,
    snapshot_configurations=None,
    grassmann_configuration=None,
    snapshot_parameter_labels=None,
    grassmann_parameter_labels=None,
    vertex_coordinates:PETSc.Vec=None,
    centroid_coordinates:PETSc.Vec=None,
    converged=True,
    reference_snapshot=False,
    comm=MPI.COMM_WORLD,
):
    """
    Write one snapshot to an HDF5 file in parallel, along with metadata.

    Parameters
    ----------
    filename : str
        Name of the HDF5 file.
    states : PETSc.Vec
        Distributed PETSc vector (local partition of the snapshot).
    snapshot_index : int
        Index of the snapshot (column in the snapshots matrix).
    snapshot_configurations : np.ndarray, optional
        Shape (n_snapshots, n_params). Written once (by rank 0).
    grassmann_configuration : np.ndarray, optional
        Shape (n_params,). Written once (by rank 0).
    snapshot_parameter_labels : list of str, optional
        Names of parameters used in snapshots. Written once (by rank 0).
    grassmann_parameter_labels : list of str, optional
        Names of parameters used in Grassmann configuration. Written once (by rank 0).
    converged : bool
        Whether the snapshot is considered converged.
    reference_snapshot: bool
        Whether to save the passed snapshot separately in the reference snapshot location
    comm : MPI.Comm
        MPI communicator.
    """
    rank = comm.Get_rank()

    # PETSc vector distribution info
    START_state, END_state = states.getOwnershipRange()
    local_states           = states.getArray()
    GLOBAL_SIZE_states     = states.getSize()

    START_vert, END_vert = vertex_coordinates.getOwnershipRange()
    local_vert_coords    = vertex_coordinates.getArray()
    GLOBAL_SIZE_vert     = vertex_coordinates.getSize()

    START_centroid, END_centroid = centroid_coordinates.getOwnershipRange()
    local_centroid_coords        = centroid_coordinates.getArray()
    GLOBAL_SIZE_centroid         = centroid_coordinates.getSize()



    # Open file in parallel mode
    with h5py.File(filename, "a", driver="mpio", comm=comm) as f:
        # --- Ensure main datasets exist ---
        if not reference_snapshot and "snapshots" not in f:
            if rank == 0 and snapshot_configurations is None:
                raise ValueError("snapshot_configurations required on first write")
            
            n_snapshots = snapshot_configurations.shape[0] if snapshot_configurations is not None else snapshot_index + 1
            
            f.create_dataset("snapshots",             (GLOBAL_SIZE_states,   n_snapshots), dtype="f8")
            f.create_dataset("vertex_coordinates",    (GLOBAL_SIZE_vert,     n_snapshots), dtype="f8")
            f.create_dataset("centroid_coordinates",  (GLOBAL_SIZE_centroid, n_snapshots), dtype="f8")
            f.create_dataset("converged",             (n_snapshots,), dtype="bool")
            f.create_dataset("last_written_snapshot", (), dtype="i8")
           
            if rank == 0:
                f["last_written_snapshot"][()] = -1
            comm.Barrier()

        # --- Write snapshot ---
        if reference_snapshot:
            # Create reference snapshot dataset if it does not exist
            if "reference_snapshot" not in f:
                f.create_dataset("reference_snapshot", (GLOBAL_SIZE_states,), dtype="f8")
            # Write the reference snapshot
            f["reference_snapshot"][START_state:END_state] = local_states

            if "reference_vertex_coordinates" not in f:
                f.create_dataset("reference_vertex_coordinates", (GLOBAL_SIZE_vert,), dtype="f8")
            # Write the reference coordinates
            f["reference_vertex_coordinates"][START_vert:END_vert] = local_vert_coords

            if "reference_centroid_coordinates" not in f:
                f.create_dataset("reference_centroid_coordinates", (GLOBAL_SIZE_centroid,), dtype="f8")
            # Write the reference coordinates
            f["reference_centroid_coordinates"][START_centroid:END_centroid] = local_centroid_coords

            if "reference_converged" not in f:
                f.create_dataset("reference_converged", data=converged, dtype="bool")
        else:
            # Normal snapshot written to main dataset
            dset = f["snapshots"]
            dset[START_state:END_state, snapshot_index] = local_states

            if local_vert_coords is not None:
                dset_vert_coords = f["vertex_coordinates"]
                dset_vert_coords[START_vert:END_vert, snapshot_index] = local_vert_coords

            if local_centroid_coords is not None:
                dset_centroid_coords = f["centroid_coordinates"]
                dset_centroid_coords[START_centroid:END_centroid, snapshot_index] = local_centroid_coords

            # Metadata updates
            if rank == 0:
                f["converged"][snapshot_index] = converged
                f["last_written_snapshot"][()] = snapshot_index

        # --- Write other metadata ---
        if snapshot_configurations is not None and "snapshot_configurations" not in f:
            f.create_dataset("snapshot_configurations", data=snapshot_configurations)
        if grassmann_configuration is not None and "grassmann_configuration" not in f:
            f.create_dataset("grassmann_configuration", data=grassmann_configuration)

        if snapshot_parameter_labels is not None and "snapshot_parameter_labels" not in f:
            max_string_len = max(len(s) for s in snapshot_parameter_labels)
            dt = f"S{max_string_len}"
            snapshot_labels_array = np.array(snapshot_parameter_labels, dtype=dt)
            f.create_dataset("snapshot_parameter_labels", data=np.array(snapshot_labels_array, dtype=dt))

        if grassmann_parameter_labels is not None and "grassmann_parameter_labels" not in f:
            max_string_len = max(len(s) for s in grassmann_parameter_labels)
            dt = f"S{max_string_len}"
            grassmann_labels_array = np.array(grassmann_parameter_labels, dtype=dt)
            f.create_dataset("grassmann_parameter_labels", data=np.array(grassmann_labels_array, dtype=dt))



def read_snapshots(
    filename,
    dafoam_instance,
):
    """
    Read snapshots and metadata from an HDF5 file in parallel.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    dafoam_instance : object
        DAFoam instance (must expose a PETSc Vec with correct partitioning).
    comm : MPI.Comm
        MPI communicator.

    Returns
    -------
    local_data : dict
        Dictionary containing local slices of distributed data:
            - "snapshots_local" : np.ndarray, shape (local_size, n_snapshots)
            - "reference_snapshot_local" : np.ndarray, shape (local_size,) or None
    global_metadata : dict
        Dictionary containing metadata (identical on all ranks):
            - "converged" : np.ndarray
            - "last_written_snapshot" : int
            - "snapshot_configurations" : np.ndarray or None
            - "grassmann_configuration" : np.ndarray or None
            - "snapshot_parameter_labels" : list[str] or None
            - "grassmann_parameter_labels" : list[str] or None
    """
    comm = dafoam_instance.comm

    # Use  PETSc Vecs to determine partitioning (consistent with DAFoam)
    petsc_states = dafoam_instance.array2Vec(dafoam_instance.getStates())  # or however you obtain PETSc Vec
    START_state, END_state   = petsc_states.getOwnershipRange()            # global indices for this rank

    petsc_vert_coords = dafoam_instance.array2Vec(dafoam_instance.xv.ravel())  
    START_vert, END_vert   = petsc_vert_coords.getOwnershipRange()

    petsc_centroid_coords = dafoam_instance.array2Vec(dafoam_instance.getCellCentroids()) 
    START_centroid, END_centroid   = petsc_centroid_coords.getOwnershipRange()

    local_data = {}
    global_metadata = {}

    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:
        # --- Parallel read: snapshots ---
        if "snapshots" in f:
            dset        = f["snapshots"]
            local_data["snapshots"] = dset[START_state:END_state, :]

        # --- Parallel read: reference snapshot ---
        if "reference_snapshot" in f:
            ref = f["reference_snapshot"]
            local_data["reference_snapshot"] = ref[START_state:END_state]
        else:
            local_data["reference_snapshot"] = None

        # --- Parallel read: vertex coordinates ---
        if "vertex_coordinates" in f:
            ref = f["vertex_coordinates"]
            local_data["vertex_coordinates"] = ref[START_vert:END_vert]
        else:
            local_data["vertex_coordinates"] = None

        # --- Parallel read: volume coordinates (old vertex coordinates name)---
        if "volume_coordinates" in f:
            ref = f["volume_coordinates"]
            local_data["volume_coordinates"] = ref[START_vert:END_vert]
        else:
            local_data["volume_coordinates"] = None

        # --- Parallel read: centroid coordinates ---
        if "centroid_coordinates" in f:
            ref = f["centroid_coordinates"]
            local_data["centroid_coordinates"] = ref[START_centroid:END_centroid]
        else:
            local_data["centroid_coordinates"] = None

        # # --- metadata ---
        if "converged" in f:
            global_metadata["converged"] = f["converged"][:]
        if "reference_converged" in f:
            global_metadata["reference_converged"] = f["reference_converged"]
        if "last_written_snapshot" in f:
            global_metadata["last_written_snapshot"] = f["last_written_snapshot"][()]
        if "snapshot_configurations" in f:
            global_metadata["snapshot_configurations"] = f["snapshot_configurations"][:]
        else:
            global_metadata["snapshot_configurations"] = None
        if "grassmann_configuration" in f:
            global_metadata["grassmann_configuration"] = f["grassmann_configuration"][:]
        else:
            global_metadata["grassmann_configuration"] = None
        if "snapshot_parameter_labels" in f:
            global_metadata["snapshot_parameter_labels"] = f["snapshot_parameter_labels"][:].astype(str).tolist()
        else:
            global_metadata["snapshot_parameter_labels"] = None
        if "grassmann_parameter_labels" in f:
            global_metadata["grassmann_parameter_labels"] = f["grassmann_parameter_labels"][:].astype(str).tolist()
        else:
            global_metadata["grassmann_parameter_labels"] = None

    return local_data, global_metadata










def _infer_shape_from_key(key) -> Tuple[int, ...]:
    """
    Infers shape from a CSDL variable-like object (with `.value.shape`)
    or treats as scalar if no shape attribute is found.
    """
    if not isinstance(key, str): 
        if hasattr(key, "value") and hasattr(key.value, "shape"):
            return tuple(key.value.shape)
    else:
        TypeError('Please use the actual CSDL variable instead of a string in your {variable: limits} dictionary')

def build_xlimits(var_limits: Dict[Union[str, object], Iterable[float]]):
    """
    Build xlimits array for SMT sampling, inferring shapes automatically from
    CSDL variable instances (keys with `.value.shape`) or treating as scalars otherwise.
    """
    rows = []
    labels = []
    slicer = {}
    shapes = {}
    cursor = 0

    for var_key, lim in var_limits.items():
        var_name      = lim['name']
        sample_limits = list(lim['range'])

        if len(sample_limits) != 2:
            raise ValueError(f"{var_key}: limits must be [low, high], got {sample_limits}")

        shp = _infer_shape_from_key(var_key)
        shapes[var_key] = shp

        if shp == () or shp == 1:  # scalar
            rows.append(sample_limits)
            labels.append(var_name)
            slicer[var_key] = slice(cursor, cursor + 1)
            cursor += 1
        else:
            count = int(np.prod(shp))
            start = cursor
            for flat_idx in range(count):
                idx = np.unravel_index(flat_idx, shp)
                rows.append(sample_limits)
                labels.append(f'{var_name}_{"_".join(map(str, idx))}')
                cursor += 1
            slicer[var_key] = slice(start, start + count)

    xlimits = np.array(rows, dtype=float)
    return xlimits, labels, slicer, shapes

def reshape_samples(
    X: np.ndarray,
    slicer: Dict[Union[str, object], slice],
    shapes: Dict[Union[str, object], Tuple[int, ...]],
):
    """
    Convert SMT samples back into dict of variable-shaped arrays.
    Keys will match the original var_keys (string or object).
    """
    N, D = X.shape
    out = []
    for k in range(N):
        sample_k = {}
        for key, sl in slicer.items():
            shp = shapes[key]
            block = X[k, sl]
            if shp == ():
                sample_k[key] = float(block[0])
            else:
                sample_k[key] = block.reshape(shp)  # preserve original shape
        out.append(sample_k)
    return out



import pandas as pd
def print_sample_table(range_table, samples):
    """
    Print a nicely formatted table of sampled variables.

    Parameters
    ----------
    range_table : dict
        Keys are CSDL variables (with .shape attribute)
        Each value is a dict with at least a 'name' key.
    samples : array-like of shape (n_samples, n_total_columns)
        Each row is one sampled configuration. Columns are ordered as variables in
        the dictionary, flattening multi-dimensional variables.
    """
    col_labels  = []
    start_col   = 0

    

    # Generate column labels for each variable
    for csdl_var, info in range_table.items():
        name = info['name']
        shape = csdl_var.shape
        n_cols = int(np.prod(shape))

        if n_cols == 1:
            labels = [name]
        else:
            labels = [f'{name}_{"_".join(map(str, np.unravel_index(i, shape)))}' for i in range(np.prod(shape))]

        col_labels.extend(labels)
        start_col += n_cols

    # Convert samples to DataFrame
    df = pd.DataFrame(np.array(samples), columns=col_labels)

    # Add sample index as the first column
    df.insert(0, "Sample", range(len(df)))

    print(df.to_string(index=False))