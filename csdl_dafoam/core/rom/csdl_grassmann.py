import numpy as np
import csdl_alpha as csdl
from csdl_dafoam.utils.custom_explicit_reduced_svd import customExplicitReducedSVD, customExplicitReducedSVDDistributed
from csdl_dafoam.utils.decompositions import svd_distributed
from csdl_dafoam.utils.runscript_helper_functions import is_csdl
from mpi4py import MPI
from typing import List


# region GRASSMANN
class Grassmann:
    def __init__(self, m:int, k:int, comm:MPI.Comm|None=None, inner_product_weights:np.ndarray|None=None):
        """
        Represent the Grassmann manifold Gr(n, k):
        - n: ambient dimension
        - k: subspace dimension
        - inner_product_weights: weights under which the subspaces are orthonormal
        - comm: MPI communicator for row-distributed basis cases
        """
        self.m       = m
        self.k       = k
        self.comm    = comm
        self.weights = inner_product_weights

        # Handy quantities
        if self.weights is None:
            self.sqrt_weights = None
            self.inverse_sqrt_weights = None
        else:
            self.sqrt_weights = np.sqrt(self.weights)
            self.inverse_sqrt_weights = 1.0 / self.sqrt_weights


    # region exp
    def exp(self, Y0:csdl.Variable|np.ndarray, Ydot:csdl.Variable|np.ndarray):
        """Exponential map at point Y0 with tangent vector Ydot."""
        
        if len(Y0.shape) > 2 or len(Ydot.shape) > 2:
            raise NotImplementedError("Batch mode not implemented yet for Grassmann exp map.")
        
        # Using tilde to represent weighted values (we'll weight and then unweight)
        Ydot_tilde      = self._apply_sqrt_weights(Ydot)
        U_tilde, S, VT  = self._svd(Ydot_tilde)
        U               = self._apply_inverse_sqrt_weights(U_tilde)
        V               = self._T(VT)
        return Y0 @ self._A_times_diagB(V, self._cos(S)) + self._A_times_diagB(U, self._sin(S))


    # region log
    def log(self, Y0:csdl.Variable|np.ndarray, Y1:csdl.Variable|np.ndarray):
        """Logarithm map: tangent vector at Y0 pointing to Y1."""

        if len(Y0.shape) > 2 or len(Y1.shape) > 2:
            raise NotImplementedError("Batch mode not implemented yet for Grassmann log map.")

        P, _, RT = self._svd(self._inner_product(Y1, Y0), is_global=True)
        Y_star   = Y1 @ (P @ RT) 
        L        = Y_star - Y0 @ self._inner_product(Y0, Y_star)


        # Using tilde to represent weighted values (we'll weight and then unweight)
        L_tilde  = self._apply_sqrt_weights(L)
        Q_tilde, E, VT = self._svd(L_tilde)
        Q        = self._apply_inverse_sqrt_weights(Q_tilde)
        theta    = self._arcsin(E)
        return Q @ self._diagA_times_B(theta, VT)
    

    # region subspace_angles
    def subspace_angles(self, Y0:csdl.Variable|np.ndarray, Y1:csdl.Variable|np.ndarray):
        """Subspace angles: compute the angles between two subspaces given their respective bases."""
        G = self._inner_product(Y0, Y1)
        _, sigma, _ = self._svd(G, is_global=True, clip_range=[-1., 1.])
        angles = self._arccos(sigma)
        return angles
    
    
    # region geodesic
    def geodesic(self, Y0:csdl.Variable|np.ndarray, Y1:csdl.Variable|np.ndarray):
        angles = self.subspace_angles(Y0, Y1)
        return self._norm(angles, is_global=True)

    

   # region karcher_mean
    def karcher_mean(
        self,
        Y0: np.ndarray,
        Y_list: List[np.ndarray],
        max_iters: int = 50,
        tol: float = 1e-10,
        step_size: float = 1.0,
        return_history: bool = False,
        allow_csdl_vars:bool = False
    ):
        """
        Compute the Karcher mean of a list of Grassmann points.

        Parameters
        ----------
        Y0 : array_like
            Initial guess for the mean.
        Y_list : list of array_like
            List of orthonormal bases on the Grassmann manifold.
        max_iters : int
            Maximum number of fixed-point iterations.
        tol : float
            Convergence tolerance on the norm of the average tangent update.
        step_size : float
            Damping factor for the update. Usually 1.0 is fine.
        return_history : bool
            If True, return (mean, update_norm_history).

        Returns
        -------
        Y_mean : array_like
            Estimated Karcher mean.
        history : list, optional
            Norms of the average tangent updates at each iteration.
        """
        if len(Y_list) == 0:
            raise ValueError("Y_list must contain at least one basis.")
        
        if (is_csdl(Y0) or any(is_csdl(Yi) for Yi in Y_list)) and not allow_csdl_vars:
            raise ValueError("Karcher Mean is not optimized to handle CSDL Variable inputs. Pass True to allow_csdl_vars to override (this will be expensive)")

        Y = Y0
        history = []

        print("COMPUTING KARCHER MEAN") if self.comm.Get_rank() == 0 else None
        print(f"{'Iteration':^20s} {'Tangent norm':^20s}") if self.comm.Get_rank() == 0 else None
        print(f"{0:^20d} {'-':^20s}") if self.comm.Get_rank() == 0 else None
        for k in range(max_iters):
            # Average tangent vector at current iterate
            tangent_sum = None
            for Yi in Y_list:
                Yi_log = self.log(Y, Yi)
                tangent_sum = Yi_log if tangent_sum is None else tangent_sum + Yi_log

            tangent_mean = tangent_sum / len(Y_list)

            # Record update size
            update_norm = self._norm(self._apply_sqrt_weights(tangent_mean.value if is_csdl(tangent_mean) else tangent_mean))

            history.append(update_norm)

            print(f"{k + 1:^20d} {update_norm:^20.5e}") if self.comm.Get_rank() == 0 else None

            # Update on the manifold
            Y_new = self.exp(Y, step_size * tangent_mean)

            # Early stopping only when the norm is a real scalar
            if update_norm < tol:
                Y = Y_new
                break

            Y = Y_new

        return (Y, history) if return_history else Y
        

    # The below functions abstract the CSDL/Numpy variables and the MPI handling


    # region _svd
    def _svd(self, matrix:csdl.Variable|np.ndarray, is_global:bool=False, clip_range:List[float]|None=None):
        comm = self.comm
        if is_csdl(matrix):
            if comm is None or is_global:
                U, S, VT = customExplicitReducedSVD(clip_singular_vals=clip_range).evaluate(matrix)
            else:
                U, S, VT = customExplicitReducedSVDDistributed(comm=comm).evaluate(matrix)
        else:
            if comm is None or is_global:
                U, S, VT = np.linalg.svd(matrix, full_matrices=False)
                if clip_range is not None:
                    S = np.clip(S, clip_range[0], clip_range[1])
            else:
                U, S, VT = svd_distributed(matrix, comm=comm)
        return U, S, VT
    

    # region _inner_product
    def _inner_product(self, A:csdl.Variable|np.ndarray, B:csdl.Variable|np.ndarray):
        comm = self.comm
        m    = self.weights

        # Check if we have csdl variables
        csdl_vars = is_csdl(A) or is_csdl(B)

        prod = self._T(A) @ B if self.weights is None else self._T(A) @ self._diagA_times_B(m, B)

        # Perform the reduction for the distributed case (assumed when the communicator is passed)
        if comm is None:
            out_value = prod
        else:
            out_value = csdl.experimental.mpi.mpi_allreduce(prod, comm=comm) if csdl_vars else comm.allreduce(prod, op=MPI.SUM)
        return out_value
        

    # region _diagA_times_B
    def _diagA_times_B(self, A:csdl.Variable|np.ndarray, B:csdl.Variable|np.ndarray):
        if is_csdl(A) or is_csdl(B):
            diagA_B = csdl.einsum(A, B, action='i,ij->ij')
        else:
            diagA_B = A[:, None] * B
        return diagA_B
    

    # region _A_times_diagB
    def _A_times_diagB(self, A:csdl.Variable|np.ndarray, B:csdl.Variable|np.ndarray):
        if is_csdl(A) or is_csdl(B):
            A_diagB = csdl.einsum(A, B, action='ij,j->ij')
        else:
            A_diagB = A * B[None, :]
        return A_diagB
    

    # region _apply_sqrt_weights
    def _apply_sqrt_weights(self, X:csdl.Variable|np.ndarray):
        if self.weights is None:
            return X
        else:
            return self._diagA_times_B(self.sqrt_weights, X)


    # region _apply_inverse_sqrt_weights
    def _apply_inverse_sqrt_weights(self, X:csdl.Variable|np.ndarray):
        if self.weights is None:
            return X
        else:
            return self._diagA_times_B(self.inverse_sqrt_weights, X)
    

    # region _sin
    def _sin(self, value:csdl.Variable|np.ndarray):
        return csdl.sin(value) if is_csdl(value) else np.sin(value)
    

    # region _cos
    def _cos(self, value:csdl.Variable|np.ndarray):
        return csdl.cos(value) if is_csdl(value) else np.cos(value)
    

    # region _arcsin
    def _arcsin(self, value:csdl.Variable|np.ndarray):
        return csdl.arcsin(value) if is_csdl(value) else np.arcsin(value)
    

    # region _arccos
    def _arccos(self, value:csdl.Variable|np.ndarray):
        return csdl.arccos(value) if is_csdl(value) else np.arccos(value)
    

    # region _T
    def _T(self, matrix:csdl.Variable|np.ndarray):
        return matrix.T() if is_csdl(matrix) else matrix.T
    

    # region _sqrt
    def _sqrt(self, x:csdl.Variable|np.ndarray):
        return csdl.sqrt(x) if is_csdl(x) else np.sqrt(x)


    # region _arctan2
    def _arctan2(self, y:csdl.Variable|np.ndarray, x:csdl.Variable|np.ndarray):
        # csdl does not have arctan2... Use csdl.arctan with care!
        # arctan(y/x) is fine for principal angles since x=sigma >= 0 
        if is_csdl(y) or is_csdl(x):
            return csdl.arctan(y / (x + 1e-30))  # x>=0 always, tiny guard only for grad
        else:
            return np.arctan2(y, x)
        
    
    # region _norm
    def _norm(self, x:csdl.Variable|np.ndarray, is_global:bool=False):
        comm = self.comm
        if is_global:
            return csdl.norm(x) if is_csdl(x) else np.linalg.norm(x)
        else:
            if is_csdl(x):
                if comm is None:
                    return csdl.norm(x)
                else:
                    return csdl.sqrt(csdl.experimental.mpi.mpi_sum(csdl.sum(x ** 2), self.comm))
            else:
                if comm is None:
                    return np.linalg.norm(x)
                else:
                    return np.sqrt(comm.allreduce(np.sum(x ** 2), op=MPI.SUM))



def pod_modes(U, w=None, r=None):
    # U: (n_space, n_snap)
    # w: (n_space,) quadrature weights (optional)
    # r: number of modes to retain (optional)

    if w is not None:
        W_sqrt = np.sqrt(w)[:, None]
        U_w = W_sqrt * U
    else:
        U_w = U

    C = U_w.T @ U_w                      # snapshot correlation matrix
    eigvals, V = np.linalg.eigh(C)       # symmetric eigendecomposition
    idx = np.argsort(eigvals)[::-1]      # sort descending
    eigvals, V = eigvals[idx], V[:, idx]

    if r is not None:
        eigvals, V = eigvals[:r], V[:, :r]

    S = np.sqrt(eigvals)
    Phi = U @ (V / S)                    # POD modes

    return Phi, S, V

# region MAIN
if __name__ == "__main__":

    import time
    import pandas as pd
    from csdl_dafoam.utils.runscript_helper_functions import global_local_op
    # np.random.seed(0) # Set seed for rank consistency
    
    # Set up communicator
    comm      = MPI.COMM_WORLD
    rank      = comm.Get_rank()
    comm_size = comm.Get_size()

    # mxn matrix retaining k modes
    m = 1000
    n = 100
    k = 20
    
    p = 20 # Number of bases to make

    # Partitioning selection
    distribution = "block" #strided

    A           = [None] * p
    U_np        = [None] * p
    U_local_np  = [None] * p
    U           = [None] * p
    U_local     = [None] * p


    # Generate bases
    print(f"Generating arrays...") if rank == 0 else None
    if rank == 0:
        A = [np.random.randn(m, n) for _ in range(p)]
        weights = np.random.random(m)
    else:
        A = [None] * p
        weights = None

    A       = comm.bcast(A, root=0)
    weights = comm.bcast(weights, root=0)
    
    for i in range(p):
        if rank == 0:
            U_temp, _, _ = pod_modes(A[i], weights, k)
        else:
            U_temp = None
        U_temp = comm.bcast(U_temp, root=0)
        U_np[i] = U_temp
    
        # Block partitioning
        if distribution == "block":
            rows_per_rank = m // comm_size
            remainder = m % comm_size
            start = rank * rows_per_rank + min(rank, remainder)
            end   = start + rows_per_rank + (1 if rank < remainder else 0)
            U_local_np[i] = U_np[i][start:end, :]
            weights_local = weights[start:end]

        elif distribution == "strided":
            U_local_np[i] = U_np[i][rank::comm_size, :]
            weights_local = weights[rank::comm_size]

    # CSDL Setup
    recorder = csdl.Recorder(inline=True, debug=True)
    recorder.start()

    alpha    = csdl.Variable(value=1)
    for i in range(p):
        U[i]       = alpha * U_np[i]
        U_local[i] = global_local_op(alpha, U_local_np[i], lambda x,y:x*y, comm=comm)

    manifold_local  = Grassmann(m, k, comm=comm, inner_product_weights=weights_local)
    
    Udot_local = manifold_local.log(U_local[0], U_local[1])
    U05_local  = manifold_local.exp(U_local[0], 0.5 * Udot_local)

    print(f"Computing mean...") if rank == 0 else None
    U_mean_local, history = manifold_local.karcher_mean(U_local_np[0], U_local_np, return_history=True, max_iters=50)

    for i in range(p):
        d1 = manifold_local.subspace_angles(U_mean_local, U_local_np[i])
        d2 = manifold_local.subspace_angles(U_local_np[0], U_local_np[i])
        if rank == 0:
            print(i, np.max(d1) * 180 / np.pi, np.max(d2) * 180 / np.pi)

    sim = csdl.experimental.PySimulator(recorder=recorder)





    # # Generate bases
    # np.random.seed(0) # Set seed for rank consistency
    # print(f"Generating arrays...") if rank == 0 else None
    # A0 = np.random.random((m, n))
    # A1 = np.random.random((m, n))

    # print(f"Forming bases...") if rank == 0 else None
    # U0_np, _ = np.linalg.qr(A0)
    # U1_np, _ = np.linalg.qr(A1)

    # # Numpy variables
    # U0_np    = U0_np[:, :k]
    # U1_np    = U1_np[:, :k]

    # # Partitioning selection
    # distribution = "block" #strided

	# # # Block partitioning
    # if distribution == "block":
    #     rows_per_rank = m // comm_size
    #     remainder = m % comm_size
    #     start = rank * rows_per_rank + min(rank, remainder)
    #     end   = start + rows_per_rank + (1 if rank < remainder else 0)
    #     U0_local_np = U0_np[start:end, :]
    #     U1_local_np = U1_np[start:end, :]

    # elif distribution == "strided":
    #     U0_local_np = U0_np[rank::comm_size, :]
    #     U1_local_np = U1_np[rank::comm_size, :]

    # # CSDL Setup
    # recorder = csdl.Recorder(inline=True, debug=True)
    # recorder.start()

    # # CSDL variables
    # alpha    = csdl.Variable(value=1)
    # U0       = alpha * U0_np
    # U1       = alpha * U1_np
    # U0_local = global_local_op(alpha, U0_local_np, lambda x,y:x*y, comm=comm)
    # U1_local = global_local_op(alpha, U1_local_np, lambda x,y:x*y, comm=comm)
    
    # manifold_local  = Grassmann(m, k, comm=comm)

    # gamma_local = manifold_local.log(U0_local, U1_local)
    # U05_local   = manifold_local.exp(U0_local, 0.5 * gamma_local)
    # angles      = manifold_local.subspace_angles(U0_local, U05_local)
    # obj_local   = csdl.norm(angles) / comm_size
    # obj_global  = csdl.experimental.mpi.mpi_sum(obj_local, comm)
    
    # dv  = alpha
    # obj = obj_global

    # recorder.stop()
    # sim = csdl.experimental.PySimulator(recorder=recorder)

    # analytical_grad  = sim.compute_totals(obj, dv)[obj, dv]

    # print(f"Rank {rank} obj         : {obj.value}")
    # print(f"Rank {rank} Analytical  : {analytical_grad}")



