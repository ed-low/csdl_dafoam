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
    start, end = states.getOwnershipRange()
    local_states = states.getArray()
    global_size = states.getSize()

    # Open file in parallel mode
    with h5py.File(filename, "a", driver="mpio", comm=comm) as f:
        # --- Ensure main datasets exist ---
        if not reference_snapshot and "snapshots" not in f:
            if rank == 0 and snapshot_configurations is None:
                raise ValueError("snapshot_configurations required on first write")
            n_snapshots = snapshot_configurations.shape[0] if snapshot_configurations is not None else snapshot_index + 1
            if rank == 0:
                f.create_dataset("snapshots", (global_size, n_snapshots), dtype="f8")
                f.create_dataset("converged", (n_snapshots,), dtype="bool")
                f.create_dataset("last_written_snapshot", (), dtype="i8")
                f["last_written_snapshot"][()] = -1
            comm.Barrier()

        # --- Write snapshot ---
        if reference_snapshot:
            # Create reference snapshot dataset if it does not exist
            if "reference_snapshot" not in f:
                f.create_dataset("reference_snapshot", (global_size,), dtype="f8")
            # Write the reference snapshot
            f["reference_snapshot"][start:end] = local_states

            if "reference_converged" not in f:
                f.create_dataset("reference_converged", data=converged, dtype="bool")
        else:
            # Normal snapshot written to main dataset
            dset = f["snapshots"]
            dset[start:end, snapshot_index] = local_states

            # Metadata updates
            if rank == 0:
                f["converged"][snapshot_index] = converged
                f["last_written_snapshot"][()] = snapshot_index

        # --- Write other metadata only once ---
        if rank == 0:
            if snapshot_configurations is not None and "snapshot_configurations" not in f:
                f.create_dataset("snapshot_configurations", data=snapshot_configurations)
            if grassmann_configuration is not None and "grassmann_configuration" not in f:
                f.create_dataset("grassmann_configuration", data=grassmann_configuration)

            dt = h5py.string_dtype(encoding="utf-8")
            if snapshot_parameter_labels is not None and "snapshot_parameter_labels" not in f:
                f.create_dataset("snapshot_parameter_labels", data=np.array(snapshot_parameter_labels, dtype=dt))
            if grassmann_parameter_labels is not None and "grassmann_parameter_labels" not in f:
                f.create_dataset("grassmann_parameter_labels", data=np.array(grassmann_parameter_labels, dtype=dt))



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
    rank = comm.Get_rank()

    # Use a PETSc Vec to determine partitioning (consistent with DAFoam)
    petsc_states = dafoam_instance.array2Vec(dafoam_instance.getStates())  # or however you obtain PETSc Vec
    start, end   = petsc_states.getOwnershipRange()                              # global indices for this rank

    local_data = {}
    global_metadata = {}

    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:
        # --- Parallel read: snapshots ---
        if "snapshots" in f:
            dset        = f["snapshots"]
            n_snapshots = dset.shape[1]
            local_data["snapshots_local"] = dset[start:end, :]

        # --- Parallel read: reference snapshot ---
        if "reference_snapshot" in f:
            ref = f["reference_snapshot"]
            local_data["reference_snapshot_local"] = ref[start:end]
        else:
            local_data["reference_snapshot_local"] = None

        # --- Serial read: metadata (rank 0 only) ---
        if rank == 0:
            if "converged" in f:
                global_metadata["converged"] = f["converged"][:]
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
        else:
            global_metadata = None

        # Broadcast metadata so all ranks have it
        global_metadata = comm.bcast(global_metadata, root=0)

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
        lim = list(lim['range'])
        if len(lim) != 2:
            raise ValueError(f"{var_key}: limits must be [low, high], got {lim}")

        shp = _infer_shape_from_key(var_key)
        shapes[var_key] = shp

        if shp == ():  # scalar
            rows.append(lim)
            labels.append((var_key, ()))
            slicer[var_key] = slice(cursor, cursor + 1)
            cursor += 1
        else:
            count = int(np.prod(shp))
            start = cursor
            for flat_idx in range(count):
                idx = np.unravel_index(flat_idx, shp)
                rows.append(lim)
                labels.append((var_key, idx))
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




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

def get_cell_centroids(dafoam_instance):
    """
    Approximate cell centroids by averaging the coordinates of all points
    belonging to a cell. Works for visualization.
    """
    # Mesh points (vertices)
    points = dafoam_instance.xv  # shape (nPoints, 3)

    # Face connectivity
    faces = dafoam_instance.faces

    # Owner/neighbour arrays
    owners = dafoam_instance.owners
    neighbours = dafoam_instance.neighbours

    # Map: cell -> set of point indices
    cell_points = {}
    for face_id, pt_indices in enumerate(faces):
        owner = owners[face_id]
        cell_points.setdefault(owner, set()).update(pt_indices)

        if face_id < len(neighbours):  # internal face
            neighbour = neighbours[face_id]
            cell_points.setdefault(neighbour, set()).update(pt_indices)

    # Compute approximate centroids
    centroids = np.zeros((len(cell_points), 3))
    for cell_id, pts in cell_points.items():
        coords = points[list(pts)]
        centroids[cell_id] = coords.mean(axis=0)

    return centroids


def plot_mesh_field(centroids, field, cmap="viridis"):
    """
    Plot a scalar/vector field on mesh cell centroids.

    Parameters
    ----------
    centroids : (nCells, 3) ndarray
        Cell centroid coordinates
    field : (nCells,) or (nCells,3) ndarray
        Scalar field (color by value) or vector field (color by magnitude)
    cmap : str
        Matplotlib colormap
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # If vector field → compute magnitude
    if field.ndim == 2 and field.shape[1] == 3:
        field_vals = np.linalg.norm(field, axis=1)
    else:
        field_vals = field

    sc = ax.scatter(
        centroids[:, 0], centroids[:, 1], centroids[:, 2],
        c=field_vals, cmap=cmap, s=8, alpha=0.8
    )

    fig.colorbar(sc, ax=ax, label="Field value")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.show()


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
    col_labels = []
    start_col = 0

    # Generate column labels for each variable
    for csdl_var, info in range_table.items():
        name = info['name']
        shape = csdl_var.shape
        n_cols = int(np.prod(shape))

        if n_cols == 1:
            labels = [name]
        else:
            labels = [f"{name}_{i}" for i in range(n_cols)]

        col_labels.extend(labels)
        start_col += n_cols

    # Convert samples to DataFrame
    df = pd.DataFrame(np.array(samples), columns=col_labels)

    # Add sample index as the first column
    df.insert(0, "Sample", range(len(df)))

    print(df.to_string(index=False))