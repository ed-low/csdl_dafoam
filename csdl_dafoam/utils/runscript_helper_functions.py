import numpy as np
import time
import hashlib
import os
import sys
import contextlib
from contextlib import contextmanager
import pickle
from mpi4py import MPI



# region read_simple_pickle
def read_simple_pickle(file_path):
    with open(file_path, 'rb') as handle:
        contents = pickle.load(handle)

    return contents



# region write_simple_pickle
def write_simple_pickle(var_to_write, file_path):
    with open(file_path, 'wb+') as handle:
        var_to_write_copy = var_to_write.copy()
        pickle.dump(var_to_write_copy, handle, protocol=pickle.HIGHEST_PROTOCOL)



# region gather_array_to_rank0
def gather_array_to_rank0(x_local: np.ndarray, comm: MPI.Comm = MPI.COMM_WORLD):
    """
    Gather local arrays to rank 0.
    
    Returns on rank 0:
        - x_full: (N_total, dim) array
        - sizes: list of local row counts from each rank
        - index_ranges: list of (start, stop) tuples for each rank's slice
    
    On other ranks:
        - x_full is None
        - sizes and index_ranges are available for consistency
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_npts = np.array([x_local.shape[0]], dtype=np.int32)
    sizes = np.zeros(size, dtype=np.int32)
    comm.Allgather([local_npts, MPI.INT], [sizes, MPI.INT])

    if x_local.ndim > 1:
        dim = x_local.shape[1]
        counts = sizes * dim
    else:
        dim = -1
        counts = sizes
    displacements = np.insert(np.cumsum(counts), 0, 0)[:-1]

    # Compute start/stop indices for slicing back full array
    starts = np.insert(np.cumsum(sizes), 0, 0)[:-1]
    stops = starts + sizes
    index_ranges = list(zip(starts, stops))

    sendbuf = x_local.flatten()
    recvbuf = None
    if rank == 0:
        total_count = np.sum(counts)
        recvbuf = np.empty(total_count, dtype=np.float64)

    comm.Gatherv(sendbuf, (recvbuf, counts, displacements, MPI.DOUBLE), root=0)

    if rank == 0:
        if dim == -1:
            x_full = recvbuf.reshape((-1))
        else:
            x_full = recvbuf.reshape((-1, dim))
        return x_full, sizes, index_ranges
    else:
        return None, sizes, index_ranges



#region is_headless
def is_headless():
    # 1. Common Unix/Linux: no X11 display variable
    if sys.platform != "win32":
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return True

    # 2. Try to create a simple GUI context (Tkinter) to verify display availability.
    try:
        # suppress stderr from Tk if it complains
        with contextlib.redirect_stderr(open(os.devnull, "w")):
            import tkinter
            root = tkinter.Tk()
            root.withdraw()
            root.update_idletasks()
            root.destroy()
    except Exception:
        return True  # failed to create even a minimal window -> likely headless

    # 3. Check common headless backends (e.g., matplotlib using Agg implies no interactive display)
    try:
        import matplotlib
        backend = matplotlib.get_backend().lower()
        if "agg" in backend and not ("tk" in backend or "qt" in backend or "wx" in backend):
            # Agg can be explicitly set even in non-headless cases, so this is a soft signal.
            pass  # don't conclusively declare headless based solely on this
    except ImportError:
        pass  # matplotlib not present, ignore

    return False  # if we reached here, a display seems available



# region Timer
# Use this to print the timings for certain lines
timings = {}  # Optional: for logging total times

@contextmanager
def Timer(name, rank, timing_enabled):
    if timing_enabled:
        print(f'Rank {rank}: {name}...', flush=True)
        start = time.time()
        yield
        elapsed = time.time() - start
        print(f'Rank {rank}: {name} elapsed time: {elapsed:.3f} s')
        timings[name] = elapsed
    else:
        yield



# region hash_array_tol
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
    scaled = np.rint(arr / tol).astype(np.int64)
    scaled = np.ascontiguousarray(scaled)

    hasher = hashlib.sha256()
    hasher.update(np.array(arr.shape, dtype=np.int64).tobytes())
    hasher.update(scaled.tobytes())

    return hasher.hexdigest()[:length]



# region quiet_barrier
# (replacement for comm.Barrier() so that cpu doesn't idle at 100%)
def quiet_barrier(comm, interval=0.01):
    req = comm.Ibarrier()
    while not req.Test():
        time.sleep(interval)



# region compute_vertex_normals
def compute_vertex_normals(dafoam_instance, outward_ref=None):
    """
    Compute per-vertex normals from mesh connectivity.

    Parameters
    ----------
    points : (N_nodes, 3) array
        Coordinates of the mesh nodes.
    conn : list or 1D array
        Flattened connectivity list of indices for all faces.
    faceSizes : list or 1D array
        Number of nodes in each face (length = number of faces).
    outward_ref : array-like, optional
        Reference point outside the mesh to ensure outward-pointing normals.
        If None, normals are not flipped.

    Returns
    -------
    vertex_normals : (N_nodes, 3) array
        Normal vector at each vertex (unit vectors).
    """
    rank = dafoam_instance.comm.rank

    points          = np.array(dafoam_instance.getSurfaceCoordinates())
    conn, faceSizes = dafoam_instance.getSurfaceConnectivity()
    conn            = np.array(conn)
    faceSizes       = np.array(faceSizes)

    N_nodes         = points.shape[0]
    vertex_normals  = np.zeros((N_nodes, 3), dtype=float)
    vertex_counts   = np.zeros(N_nodes, dtype=float)

    face_start = 0
    face_normals = np.zeros((faceSizes.shape[0], 3))
    face_centers = np.zeros((faceSizes.shape[0], 3))

    # 1. Compute face normals
    for faceI in range(faceSizes.shape[0]):
        nPts         = faceSizes[faceI]
        face_indices = conn[face_start: face_start + nPts]
        face_pts     = points[face_indices, :]
        n = np.zeros(3)
        for i in range(nPts):
            p0 = face_pts[i, :]
            p1 = face_pts[(i + 1) % nPts, :]  # wrap around
            n[0] += (p0[1] - p1[1]) * (p0[2] + p1[2])
            n[1] += (p0[2] - p1[2]) * (p0[0] + p1[0])
            n[2] += (p0[0] - p1[0]) * (p0[1] + p1[1])

        # Normalize normal
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-14:
            print(f'Rank {rank}, face {i}')
            n[:] = 0.0  # handle degenerate faces
        else:
            n /= norm_n

        # 2. Accumulate to vertices
        for idx in face_indices:
            vertex_normals[idx] += n
            vertex_counts[idx]  += 1

        face_normals[faceI, :] = n
        face_centers[faceI, :] = np.mean(face_pts, axis=0)
        # print(f'Rank {rank}, face {faceI}, normal {n}, center {np.mean(face_pts, axis=0)}')
        face_start += nPts

    # face_normals = np.array(face_normals)
    # face_centers = np.array(face_centers)

    # 3. Normalize vertex normals
    # vertex_normals /= vertex_counts[:, None] + 1e-12
    vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, None] + 1e-12

    # 4. Optional: flip normals to point outward
    if outward_ref is not None:
        outward_ref = np.array(outward_ref)
        vec_to_ref  = outward_ref[None, :] - points
        dot         = np.einsum('ij,ij->i', vertex_normals, vec_to_ref)
        vertex_normals[dot < 0] *= -1

    return vertex_normals, face_normals, face_centers



# region average_normals_at_duplicate_points
def average_normals_at_duplicate_points(points, normals):
    """
    points:  (N, 3)
    normals: (N, 3)
    returns: normals_new (N, 3) where duplicate points share averaged normals
    """
    # Identify unique points and how points map to them
    uniq, inverse, counts = np.unique(points, axis=0,
                                      return_inverse=True,
                                      return_counts=True)

    normals_out = normals.copy()

    # For each unique point that appears more than once
    for u_idx in np.where(counts > 1)[0]:
        # indices of all duplicates in the original array
        dup_indices = np.where(inverse == u_idx)[0]

        # average their normals (optionally renormalize depending on use case)
        avg_n = normals[dup_indices].mean(axis=0)

        # assign back to all duplicates and renormalize
        normals_out[dup_indices] = avg_n/np.linalg.norm(avg_n)

    return normals_out



# region print_runscript_info
def print_runscript_info():
    """
    Print the command line invocation and script content to console.
    Useful for documenting simulation runs in output logs.
    """
    # Print command line invocation
    print("="*80)
    print("COMMAND LINE:")
    print(" ".join(sys.argv))
    print("="*80)
    
    # Print script content
    script_path = sys.argv[0]
    try:
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        print("BEGIN RUNSCRIPT CONTENT:")
        print("="*80)
        print(script_content)
        print("END RUNSCRIPT CONTENT:")
        print("="*80)
        print()  # Extra newline before actual output starts
    except FileNotFoundError:
        print(f"Warning: Could not read script file at {script_path}")
        print("="*80)
        print()