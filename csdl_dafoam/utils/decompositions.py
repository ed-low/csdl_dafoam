import numpy as np
from mpi4py import MPI
from typing import Literal


# These are universal constants for the reorthogonalization check
# in the gram methods. Might be worth making these options for
# each method?
FLOAT64_DIGITS           = 16
ORTHOGONALITY_TOL_DIGITS = 10  # i.e. we want orth error < 1e-10


# region method_of_snapshots_distributed
def method_of_snapshots_distributed(matrix_local:np.ndarray, 
                                    comm:MPI.Comm, 
                                    method:Literal["tsqr", "gram"]="tsqr", 
                                    weights_local:np.ndarray=None, 
                                    n_retain:int|None=None,
                                    orthogonality_check:bool=False,
                                    disable_reorthogonalize:bool=False):

    # Method check
    valid_methods = {"tsqr", "gram"}
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}.")
   
    if method.lower() == "tsqr":
        local_modes, s_vals = method_of_snapshots_tsqr(matrix_local=matrix_local, 
                                        comm=comm, 
                                        weights_local=weights_local,
                                        n_retain=n_retain,
                                        orthogonality_check=orthogonality_check,
                                        disable_reorthogonalize=disable_reorthogonalize)
    
    if method.lower() == "gram":
        local_modes, s_vals = method_of_snapshots_gram(matrix_local=matrix_local, 
                                        comm=comm, 
                                        weights_local=weights_local,
                                        n_retain=n_retain,
                                        orthogonality_check=orthogonality_check,
                                        disable_reorthogonalize=disable_reorthogonalize)
        
    return local_modes, s_vals


# region svd_distributed
def svd_distributed(matrix_local:np.ndarray, 
                    comm:MPI.Comm, 
                    method:Literal["tsqr", "gram"]="tsqr", 
                    method_opts:dict|None=None):
    
    # Method check
    valid_methods = {"tsqr", "gram"}
    if method not in valid_methods:
     raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}.")
    
    method_opts = {} if method_opts is None else method_opts
    
    if method.lower() == "tsqr":
        U_local, s, VT = svd_tsqr(matrix_local=matrix_local, comm=comm)
    
    if method.lower() == "gram":
        disable_reorthogonalize = method_opts.get("disable_reorthogonality", False)
        U_local, s, VT = svd_gram(matrix_local=matrix_local, 
                                  comm=comm,
                                  disable_reorthogonalize=disable_reorthogonalize)
        
    return U_local, s, VT



# region method_of_snapshots_gram
def method_of_snapshots_gram(matrix_local:np.ndarray, comm:MPI.Comm, weights_local:np.ndarray=None, 
                             n_retain:int|None=None, orthogonality_check:bool=False, disable_reorthogonalize:bool=False):
    A_local = matrix_local
    m_local = weights_local

    rank = comm.Get_rank()

    # Method of snapshots (referenced https://willcox-research-group.github.io/rom-operator-inference-Python3/_modules/opinf/basis/_pod.html#method_of_snapshots)
    min_thresh      = 1e-15
    n_snapshots     = A_local.shape[1]
    
    if m_local is None:
        local_gramian   = A_local.T @ A_local
    else:
        local_gramian   = A_local.T @ (m_local[:, None] * A_local)

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
        positives   = eigvals > max(min_thresh, abs(np.min(eigvals))) # This was the old method

        # # Only use negative eigenvalue magnitude as a guard against numerical noise
        # noise_floor = max(min_thresh, eigvals[0] * 1e-12)  # ~4 digits of margin before float64 limit
        # positives = eigvals > noise_floor
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

    # Reconstruct modes
    local_modes = A_local @ (eigvecs / s_vals)

    # Will run reorthogonalization if necessary:
    condition_number = s_vals[0] / s_vals[-1]
    digits_lost      = np.log10(condition_number)
    reorth_needed    = digits_lost > (FLOAT64_DIGITS - ORTHOGONALITY_TOL_DIGITS)

    # We'll retain only the number requested (would make reorthogonalization cheaper)
    n_retain    = local_modes.shape[1] if n_retain is None or n_retain <= 0 else n_retain
    if n_retain > local_modes.shape[1]:
        print("WARNING: Cannot deliver requested number of modes.") if rank == 0 else None
    local_modes = local_modes[:, :n_retain]
    s_vals      = s_vals[:n_retain]

    if reorth_needed and not disable_reorthogonalize:
        if rank == 0:
            print(f"Reorthogonalization triggered: condition number = {condition_number:.2e} ({digits_lost:.1f} digits lost)")
        orthogonalize_distributed(matrix_local=local_modes, comm=comm, weights_local=weights_local)
    
    if orthogonality_check:
        orthogonality_check_distributed(matrix_local=local_modes, comm=comm, weights_local=m_local)

    return local_modes, s_vals


# region method_of_snapshots_tsqr
def method_of_snapshots_tsqr(matrix_local:np.ndarray, comm:MPI.Comm, weights_local:np.ndarray=None, 
                             n_retain:int|None=None, orthogonality_check:bool=False, disable_reorthogonalize:bool=False):
    A_local = matrix_local
    m_local = weights_local

    rank = comm.Get_rank()

    # Bake weights into A to form Ã
    if m_local is not None:
        A_weighted_local = np.sqrt(m_local)[:, None] * A_local
    else:
        A_weighted_local = A_local

    # Distributed TSQR on Ã: condition number of R = condition number of Ã
    # (NOT squared, unlike forming AᵀWA directly)
    _, R = tsqr(A_weighted_local, comm)

    # SVD of small R (n×n): cheap, replicated on all ranks
    # This is the key: SVD of R avoids squaring the condition number
    _, s_vals, VT = np.linalg.svd(R, full_matrices=False)
    V = VT.T  # (n x n)

    # Recover distributed modes in ORIGINAL (unweighted) space
    # Weights were baked into Ã, so A_local here is the original unweighted data
    local_modes = A_local @ (V / s_vals)

    # Will run reorthogonalization if necessary:
    condition_number = s_vals[0] / s_vals[-1]
    digits_lost      = np.log10(condition_number)
    reorth_needed    = digits_lost > (FLOAT64_DIGITS - ORTHOGONALITY_TOL_DIGITS)

    # We'll retain only the number requested (would make reorthogonalization cheaper)
    n_retain    = local_modes.shape[1] if n_retain is None or n_retain <= 0 else n_retain
    if n_retain > local_modes.shape[1]:
        print("WARNING: Cannot deliver requested number of modes.") if rank == 0 else None
    local_modes = local_modes[:, :n_retain]
    s_vals      = s_vals[:n_retain]

    if reorth_needed and not disable_reorthogonalize:
        if rank == 0:
            print(f"Reorthogonalization triggered: condition number = {condition_number:.2e} ({digits_lost:.1f} digits lost)")
        orthogonalize_distributed(matrix_local=local_modes, comm=comm, weights_local=weights_local)
    
    if orthogonality_check:
        orthogonality_check_distributed(matrix_local=local_modes, comm=comm, weights_local=m_local)

    return local_modes, s_vals


# region svd_gram
def svd_gram(matrix_local:np.ndarray, comm:MPI.Comm, disable_reorthogonalize:bool=False): 
    A_local = matrix_local

    if np.iscomplexobj(A_local): 
        print('WARING: Complex array found. customExplicitReducedSVD currently only defines derivatives for real valued arrays.') 
    
    ATA = comm.allreduce(A_local.T @ A_local, op=MPI.SUM)
    eigvals, eigvecs = np.linalg.eigh(ATA)

    # eigh returns ascending order, reverse to descending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # Truncate numerically zero eigenvalues
    min_thresh  = 1e-15
    positives   = eigvals >= max(min_thresh, abs(np.min(eigvals))) # This was the old method
    eigvals, eigvecs = eigvals[positives], eigvecs[:, positives]
    s_vals      = np.sqrt(eigvals)
    VT          = eigvecs.T                          
    U_local     = A_local @ (eigvecs / s_vals)

    # Will run reorthogonalization if necessary:
    condition_number = eigvals[0] / eigvals[-1]
    digits_lost      = np.log10(condition_number)
    reorth_needed    = digits_lost > (FLOAT64_DIGITS - ORTHOGONALITY_TOL_DIGITS)
    
    if reorth_needed and not disable_reorthogonalize:
        if rank == 0:
            print(f"Reorthogonalization triggered: condition number = {condition_number:.2e} ({digits_lost:.1f} digits lost)")
        orthogonalize_distributed(matrix_local=U_local, comm=comm)

    return U_local, s_vals, VT


# region svd_tsqr
def svd_tsqr(matrix_local:np.ndarray, comm:MPI.Comm):
    A_local = matrix_local

    # TSQR: get distributed Q and replicated R
    Q_local, R = tsqr(A_local, comm)

    diag = np.sign(np.diag(R))
    diag[diag == 0] = 1.0
    D = np.diag(diag)

    Q_local = Q_local @ D
    R = D @ R

    # SVD of small R (n x n, replicated on all ranks)
    # This is exact — no approximation
    U_R, s_vals, VT = np.linalg.svd(R, full_matrices=False)

    idx = np.argmax(np.abs(U_R), axis=0)
    signs = np.sign(U_R[idx, np.arange(U_R.shape[1])])
    signs[signs == 0] = 1.0

    U_R = U_R * signs
    VT = VT * signs[:, None]
    
    # Recover distributed U
    U_local = Q_local @ U_R  # (local_rows x n_retained)

    return U_local, s_vals, VT


# region tsqr
def tsqr(matrix_local:np.ndarray, comm:MPI.Comm):
    A_local     = matrix_local
    rank        = comm.Get_rank()
    comm_size   = comm.Get_size()
    k           = A_local.shape[1]

    assert A_local.shape[0] >= A_local.shape[1], (
        f"TSQR requires local_rows >= n_cols on every rank, "
        f"got {A_local.shape[0]} rows and {k} cols on rank {rank}. "
        f"Either reduce n_cols, increase rows per rank, or use fewer MPI ranks."
    )

    # Local QR
    Q_local, R = np.linalg.qr(A_local, mode='reduced')

    # Forward pass
    Q_tree       = []
    step         = 1
    my_exit_step = None
    while step < comm_size:
        if rank % (2 * step) == 0:
            partner = rank + step
            
            if partner < comm_size:
                R_partner = np.empty_like(R)
                comm.Recv(R_partner, source=partner, tag=step)
                Q_r, R = np.linalg.qr(np.vstack([R, R_partner]), mode='reduced')
                Q_tree.append((Q_r, step))
            
            else:
                Q_tree.append((None, step))
        
        elif rank % (2 * step) == step:
            comm.Send(R, dest=rank - step, tag=step)
            my_exit_step = step
            break
        step *= 2

    R = comm.bcast(R, root=0)
    comm.Barrier()

    # Backward pass
    BACK_TAG_OFFSET = 100 # Offset is to avoid MPI collisions
    Q_top = np.eye(k)

    # Receive from parent FIRST, before processing our own subtree
    if my_exit_step is not None:
        Q_top = np.empty((k, k))
        comm.Recv(Q_top, source=rank - my_exit_step, tag=my_exit_step + BACK_TAG_OFFSET)

    # Now propagate downward through our own Q_tree using the correct Q_top
    for Q_r, step in reversed(Q_tree):
        if Q_r is not None:
            partner = rank + step
            
            if partner < comm_size:
                Q_both    = Q_r @ Q_top
                Q_self    = Q_both[:k, :]
                Q_partner = Q_both[k:, :]
                comm.Send(Q_partner.copy(), dest=partner, tag=step + BACK_TAG_OFFSET)
                Q_top = Q_self

    Q_local = Q_local @ Q_top
    return Q_local, R


# region orthogonalize_distributed
def orthogonalize_distributed(matrix_local:np.ndarray, comm:MPI.Comm, weights_local:np.ndarray=None):

    A_local = matrix_local
    m_local = weights_local

    # Reorthogonalization via distributed Gram-Schmidt
    n_modes = A_local.shape[1]

    for i in range(n_modes):
        col = A_local[:, i]
        
        # Normalize against all previously orthogonalized columns
        for j in range(i):
            prev = A_local[:, j]
            
            # Distributed dot product
            if m_local is None:
                local_dot = np.dot(prev, col)
            else:
                local_dot = np.dot(prev, m_local * col)
            
            if comm is not None:
                global_dot = comm.allreduce(local_dot, op=MPI.SUM)
                col -= global_dot * prev
            else:
                col -= local_dot * prev
        
        # Normalize col
        if m_local is None:
            local_norm_sq = np.dot(col, col)
        else:
            local_norm_sq = np.dot(col, m_local * col)
        
        if comm is not None:
            global_norm = np.sqrt(comm.allreduce(local_norm_sq, op=MPI.SUM))
        else:
            global_norm = np.sqrt(local_norm_sq)
        
        A_local[:, i] = col / global_norm


# region orthogonality_check_distributed
def orthogonality_check_distributed(matrix_local:np.ndarray, comm:MPI.Comm, weights_local:np.ndarray=None):
    A_local = matrix_local
    m_local = weights_local

    # Orthogonality check: should be I by construction
    if m_local is None:
        ATMA_local = A_local.T @ A_local
    else:
        ATMA_local = A_local.T @ (m_local[:, None] * A_local)

    ATMA = np.zeros_like(ATMA_local)
    comm.Allreduce(ATMA_local, ATMA, op=MPI.SUM)

    orth_err = np.linalg.norm(ATMA - np.eye(ATMA.shape[0]), ord='fro')
    tol      = 1e-10 * ATMA.shape[0]

    if rank == 0:
        if orth_err > tol:
            print(f"Orthogonality check: WARNING ({orth_err:.2e} > {tol})")
        else:
            print(f"Orthogonality check: PASSED ({orth_err:.2e} < {tol})")

    return orth_err


# region _make_test_matrix
def _make_test_matrix(m, n, rank, decay='linear'):
    # Set seed for consistency among ranks
    np.random.seed(seed=42)
    U, _ = np.linalg.qr(np.random.randn(m, rank))
    V, _ = np.linalg.qr(np.random.randn(n, rank))

    if decay == 'linear':
        S = np.linspace(10, 1, rank)
    elif decay == 'exp':
        S = np.exp(-np.arange(rank))
    elif decay == 'flat':
        S = np.ones(rank)

    A = U @ (S[:, None] * V.T)
    return A, S


# region main
if __name__ == "__main__":

    import time
    import pandas as pd

    # Set up communicator
    comm      = MPI.COMM_WORLD
    rank      = comm.Get_rank()
    comm_size = comm.Get_size()

    # Generate a tall skinny array
    print("Generating array...") if rank == 0 else None
    m = 3000
    n = 30
    A, _ = _make_test_matrix(m, n, rank=n, decay="exp")

    # We'll stripe the array (our MPI partitioning)
    print("Paritioning array...") if rank == 0 else None
    A_local = A[rank::comm_size, :]

    # Dict containing the names of our methods (keep them short)
    methods = ["MoS (Gram)", "MoS (TSQR)", "SVD (Numpy)", "SVD (Gram)", "SVD (TSQR)"]
    metrics = ["Time", "Orth (LSV)", "Orth (RSV)", "Recon. max diff", "Rank"]
    results = {method: {metric: None for metric in metrics} for method in methods}

    # Method of snapshots (Gram)
    print("Computing method of snapshots (Gram)...") if rank == 0 else None
    comm.Barrier(); t0 = time.time()
    phi, s = method_of_snapshots_distributed(matrix_local=A_local, 
                                                           weights_local=None, 
                                                           method="gram",
                                                           comm=comm)
    comm.Barrier(); t1  = time.time()
    results["MoS (Gram)"]["Time"]            = comm.allreduce(t1 - t0, op=MPI.SUM) / comm_size
    results["MoS (Gram)"]["Orth (LSV)"]      = orthogonality_check_distributed(matrix_local=phi, comm=comm, weights_local=None)
    results["MoS (Gram)"]["Orth (RSV)"]      = "---"
    results["MoS (Gram)"]["Recon. max diff"] = "---"
    results["MoS (Gram)"]["Rank"]            = phi.shape[1]

    # Method of snapshots (TSQR)
    print("Computing method of snapshots (TSQR)...") if rank == 0 else None
    comm.Barrier(); t0 = time.time()
    phi, s = method_of_snapshots_distributed(matrix_local=A_local, 
                                                           weights_local=None, 
                                                           method="tsqr",
                                                           comm=comm)
    comm.Barrier(); t1  = time.time()
    results["MoS (TSQR)"]["Time"]            = comm.allreduce(t1 - t0, op=MPI.SUM) / comm_size
    results["MoS (TSQR)"]["Orth (LSV)"]      = orthogonality_check_distributed(matrix_local=phi, comm=comm, weights_local=None)
    results["MoS (TSQR)"]["Orth (RSV)"]      = "---"
    results["MoS (TSQR)"]["Recon. max diff"] = "---"
    results["MoS (TSQR)"]["Rank"]            = phi.shape[1]

    # SVD (Numpy, as baseline)
    print("Computing SVD (Numpy)...") if rank == 0 else None
    comm.Barrier(); t0 = time.time()
    U, s, VT = np.linalg.svd(A, full_matrices=False) if rank == 0 else (None, None, None)
    U  = comm.bcast(U,  root=0)
    s  = comm.bcast(s,  root=0)
    VT = comm.bcast(VT, root=0)
    comm.Barrier(); t1  = time.time()
    local_diff = A - U @ (s[:, None] * VT)
    results["SVD (Numpy)"]["Time"]            = t1 - t0
    results["SVD (Numpy)"]["Orth (LSV)"]      = np.linalg.norm(U.T @ U - np.eye(U.shape[1]), ord='fro')     if rank == 0 else None
    results["SVD (Numpy)"]["Orth (RSV)"]      = np.linalg.norm(VT @ VT.T - np.eye(VT.shape[0]), ord='fro')  if rank == 0 else None
    results["SVD (Numpy)"]["Recon. max diff"] = np.max(np.abs(local_diff))
    results["SVD (Numpy)"]["Rank"]            = U.shape[1]

    # SVD (Gram)
    print("Computing SVD (Gram)...") if rank == 0 else None
    comm.Barrier(); t0 = time.time()
    U, s, VT = svd_distributed(matrix_local=A_local, comm=comm, method="gram")
    comm.Barrier(); t1  = time.time()
    local_diff = A_local - U @ (s[:, None] * VT)
    results["SVD (Gram)"]["Time"]            = t1 - t0
    results["SVD (Gram)"]["Orth (LSV)"]      = orthogonality_check_distributed(matrix_local=U, comm=comm, weights_local=None)
    results["SVD (Gram)"]["Orth (RSV)"]      = np.linalg.norm(VT @ VT.T - np.eye(VT.shape[0]), ord='fro') if rank == 0 else None
    results["SVD (Gram)"]["Recon. max diff"] = comm.allreduce(np.max(np.abs(local_diff)), op=MPI.MAX)
    results["SVD (Gram)"]["Rank"]            = U.shape[1]

    # SVD (TSQR)
    print("Computing SVD (TSQR)...") if rank == 0 else None
    comm.Barrier(); t0 = time.time()
    U, s, VT = svd_distributed(matrix_local=A_local, comm=comm, method="tsqr")
    comm.Barrier(); t1  = time.time()
    local_diff = A_local - U @ (s[:, None] * VT)
    results["SVD (TSQR)"]["Time"]            = t1 - t0
    results["SVD (TSQR)"]["Orth (LSV)"]      = orthogonality_check_distributed(matrix_local=U, comm=comm, weights_local=None)
    results["SVD (TSQR)"]["Orth (RSV)"]      = np.linalg.norm(VT @ VT.T - np.eye(VT.shape[0]), ord='fro') if rank == 0 else None
    results["SVD (TSQR)"]["Recon. max diff"] = comm.allreduce(np.max(np.abs(local_diff)), op=MPI.MAX)
    results["SVD (TSQR)"]["Rank"]            = U.shape[1]

    # Print the output
    if rank == 0:
        df = pd.DataFrame.from_dict(results, orient="index")
        pd.set_option("display.float_format", "{:.3e}".format)
        print(df.T)