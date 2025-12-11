import numpy as np
import time
import hashlib


# TIMER
from contextlib import contextmanager
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


# QUIET_BARRIER (replacement for comm.Barrier() so that cpu doesn't idle at 100%)
def quiet_barrier(comm, interval=0.01):
    req = comm.Ibarrier()
    while not req.Test():
        time.sleep(interval)



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