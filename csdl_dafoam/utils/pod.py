import numpy as np
from mpi4py import MPI

# region _method_of_snapshots
def _method_of_snapshots_distributed(local_data, local_weights, comm=MPI.Comm, run_orthogonality_check=True):
    rank = comm.Get_rank()

    # Method of snapshots (referenced https://willcox-research-group.github.io/rom-operator-inference-Python3/_modules/opinf/basis/_pod.html#method_of_snapshots)
    min_thresh      = 1e-15
    n_snapshots     = local_data.shape[1]
    if local_weights is None:
        local_gramian   = local_data.T @ local_data
    else:
        local_gramian   = local_data.T @ (local_weights[:, None] * local_data)

    total_gramian   = np.zeros_like(local_gramian) if rank == 0 else None
    comm.Reduce(local_gramian, total_gramian, op=MPI.SUM, root=0)

    if rank == 0:
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

    # Rescale and square root eigenvalues to get singular values.
    local_modes = local_data @ (eigvecs / s_vals)

    if run_orthogonality_check:
        # Orthogonality check: should be I by construction
        # Uses the same weights as the Gramian, so any violation is numerical, not a mismatch
        if local_weights is None:
            PhiTMPhi_local = local_modes.T @ local_modes
        else:
            PhiTMPhi_local = local_modes.T @ (local_weights[:, None] * local_modes)

        PhiTMPhi = np.zeros_like(PhiTMPhi_local)
        comm.Allreduce(PhiTMPhi_local, PhiTMPhi, op=MPI.SUM)

        orth_err = np.linalg.norm(PhiTMPhi - np.eye(PhiTMPhi.shape[0]), ord='fro')
        tol      = 1e-10 * PhiTMPhi.shape[0]  # scale tolerance with number of modes

        if rank == 0:
            if orth_err > 1e-6:
                print(f"  POD orthogonality check (internal, same M as Gramian): WARNING ({orth_err:.2e})")
                print(f"  --> This is a numerical issue in the POD computation itself, not an M mismatch.")
            else:
                print(f"  POD orthogonality check (internal, same M as Gramian): PASSED ({orth_err:.2e})")

    return local_modes, s_vals


