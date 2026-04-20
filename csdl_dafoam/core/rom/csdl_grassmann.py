import numpy as np
import csdl_alpha as csdl
from csdl_dafoam.utils.custom_explicit_reduced_svd import customExplicitReducedSVD, customExplicitReducedSVDDistributed
from csdl_dafoam.utils.decompositions import svd_distributed
from mpi4py import MPI
from typing import List


# region GRASSMANN
class Grassmann:
    def __init__(self, m:int, k:int, comm:MPI.Comm|None=None, inner_product_weights:np.ndarray|None=None):
        """
        Represent the Grassmann manifold Gr(n, k):
        - n: ambient dimension
        - k: subspace dimension
        """
        self.m       = m
        self.k       = k
        self.comm    = comm
        self.rank    = comm.Get_rank()
        self.weights = inner_product_weights


    # region exp
    def exp(self, Y0:csdl.Variable|np.ndarray, Ydot:csdl.Variable|np.ndarray):
        """Exponential map at point Y0 with tangent vector Ydot."""
        
        if len(Y0.shape) > 2 or len(Ydot.shape) > 2:
            raise NotImplementedError("Batch mode not implemented yet for Grassmann exp map.")
        U, S, VT  = self._svd(Ydot)
        V         = self._T(VT)
        return Y0 @ self._A_times_diagB(V, self._cos(S)) + self._A_times_diagB(U, self._sin(S))


    # region log
    def log(self, Y0:csdl.Variable|np.ndarray, Y1:csdl.Variable|np.ndarray):
        """Logarithm map: tangent vector at Y0 pointing to Y1."""

        if len(Y0.shape) > 2 or len(Y1.shape) > 2:
            raise NotImplementedError("Batch mode not implemented yet for Grassmann log map.")
        P, _, RT = self._svd(self._inner_product(Y1, Y0), is_global=True)
        Y_star   = Y1 @ (P @ RT) #self.local_global_op(Y1, (P @ RT), lambda x,y:x@y)
        L        = Y_star - Y0 @ self._inner_product(Y0, Y_star) #self.local_global_op(Y0, self._inner_product(Y0, Y_star), lambda x,y:x@y)
        Q, E, VT = self._svd(L)
        theta    = self._arcsin(E)
        return Q @ self._diagA_times_B(theta, VT)
    

    # region subspace_angles
    def subspace_angles(self, Y0, Y1):
        G = self._inner_product(Y0, Y1)
        _, sigma, _ = self._svd(G, is_global=True, clip_range=[-1., 1.])
        angles = self._arccos(sigma)
        return angles
    

    # The below functions abstract the CSDL/Numpy variables and the MPI handling


    # region _svd
    def _svd(self, matrix:csdl.Variable|np.ndarray, is_global:bool=False, clip_range:List[float]|None=None):
        comm = self.comm
        if _is_csdl(matrix):
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
        csdl_vars = _is_csdl(A) or _is_csdl(B)

        prod = self._T(A) @ B if self.weights is None else self._T(A) @ self._diagA_times_B(m, B)

        # Perform the reduction for the distributed case (assumed when the communicator is passed)
        if comm is None:
            out_value = prod
        else:
            out_value = csdl.experimental.mpi.mpi_allreduce(prod, comm=comm) if csdl_vars else comm.allreduce(prod, op=MPI.SUM)
        return out_value
        

    # region _diagA_times_B
    def _diagA_times_B(self, A:csdl.Variable|np.ndarray, B:csdl.Variable|np.ndarray):
        if _is_csdl(A) or _is_csdl(B):
            diagA_B = csdl.einsum(A, B, action='i,ij->ij')
        else:
            diagA_B = A[:, None] * B
        return diagA_B
    

    # region _A_times_diagB
    def _A_times_diagB(self, A:csdl.Variable|np.ndarray, B:csdl.Variable|np.ndarray):
        if _is_csdl(A) or _is_csdl(B):
            A_diagB = csdl.einsum(A, B, action='ij,j->ij')
        else:
            A_diagB = A * B[None, :]
        return A_diagB
    

    # region _sin
    def _sin(self, value:csdl.Variable|np.ndarray):
        return csdl.sin(value) if _is_csdl(value) else np.sin(value)
    

    # region _cos
    def _cos(self, value:csdl.Variable|np.ndarray):
        return csdl.cos(value) if _is_csdl(value) else np.cos(value)
    

    # region _arcsin
    def _arcsin(self, value:csdl.Variable|np.ndarray):
        return csdl.arcsin(value) if _is_csdl(value) else np.arcsin(value)
    

    # region _arccos
    def _arccos(self, value:csdl.Variable|np.ndarray):
        return csdl.arccos(value) if _is_csdl(value) else np.arccos(value)
    

    # region _T
    def _T(self, matrix:csdl.Variable|np.ndarray):
        return matrix.T() if _is_csdl(matrix) else matrix.T
    

    # region _sqrt
    def _sqrt(self, x:csdl.Variable|np.ndarray):
        return csdl.sqrt(x) if _is_csdl(x) else np.sqrt(x)


    # region _arctan2
    def _arctan2(self, y:csdl.Variable|np.ndarray, x:csdl.Variable|np.ndarray):
        # csdl does not have arctan2... Use csdl.arctan with care!
        # arctan(y/x) is fine for principal angles since x=sigma >= 0 
        if _is_csdl(y) or _is_csdl(x):
            return csdl.arctan(y / (x + 1e-30))  # x>=0 always, tiny guard only for grad
        else:
            return np.arctan2(y, x)


# region _is_csdl
def _is_csdl(var:csdl.Variable|np.ndarray):
    return isinstance(var, csdl.Variable)


# region global_local_op
def global_local_op(global_var, local_var, op, comm):
    rank = comm.Get_rank()
    with csdl.experimental.mpi.enter_mpi_region(rank, comm) as region:
        if isinstance(local_var, csdl.Variable):
            local_var_split = region.split_custom(local_var, lambda x:x)
        else:
            local_var_split = region.split_constant(local_var)
        out_split        = op(global_var, local_var_split)
        out = region.merge_custom(out_split, lambda x:x)
        return out


# region local_global_op
def local_global_op(local_var, global_var, op, comm):
    rank = comm.Get_rank()
    with csdl.experimental.mpi.enter_mpi_region(rank, comm) as region:
        if isinstance(local_var, csdl.Variable):
            local_var_split = region.split_custom(local_var, lambda x:x)
        else:
            local_var_split = region.split_constant(local_var)
        out_split        = op(local_var_split, global_var)
        out = region.merge_custom(out_split, lambda x:x)
        return out

    

# region MAIN
if __name__ == "__main__":

    import time
    import pandas as pd

    # Set up communicator
    comm      = MPI.COMM_WORLD
    rank      = comm.Get_rank()
    comm_size = comm.Get_size()

    # mxn matrix retaining k modes
    m = 9
    n = 6
    k = 3

    # Generate bases
    np.random.seed(0) # Set seed for rank consistency
    print(f"Generating arrays...") if rank == 0 else None
    A0 = np.random.random((m, n))
    A1 = np.random.random((m, n))

    print(f"Forming bases...") if rank == 0 else None
    U0_np, _ = np.linalg.qr(A0)
    U1_np, _ = np.linalg.qr(A1)

    # Numpy variables
    U0_np    = U0_np[:, :k]
    U1_np    = U1_np[:, :k]

    # Partitioning selection
    distribution = "block" #strided

	# # Block partitioning
    if distribution == "block":
        rows_per_rank = m // comm_size
        remainder = m % comm_size
        start = rank * rows_per_rank + min(rank, remainder)
        end   = start + rows_per_rank + (1 if rank < remainder else 0)
        U0_local_np = U0_np[start:end, :]
        U1_local_np = U1_np[start:end, :]

    elif distribution == "strided":
        U0_local_np = U0_np[rank::comm_size, :]
        U1_local_np = U1_np[rank::comm_size, :]

    # CSDL Setup
    recorder = csdl.Recorder(inline=True, debug=True)
    recorder.start()

    # CSDL variables
    alpha    = csdl.Variable(value=1)
    U0       = alpha * U0_np
    U1       = alpha * U1_np
    U0_local = global_local_op(alpha, U0_local_np, lambda x,y:x*y, comm=comm)
    U1_local = global_local_op(alpha, U1_local_np, lambda x,y:x*y, comm=comm)
    
    manifold_local  = Grassmann(m, k, comm=comm)

    gamma_local = manifold_local.log(U0_local, U1_local)
    U05_local   = manifold_local.exp(U0_local, 0.5 * gamma_local)
    angles      = manifold_local.subspace_angles(U0_local, U05_local)
    obj_local   = csdl.norm(angles) / comm_size
    obj_global  = csdl.experimental.mpi.mpi_sum(obj_local, comm)
    
    dv  = alpha
    obj = obj_global

    recorder.stop()
    sim = csdl.experimental.PySimulator(recorder=recorder)

    analytical_grad  = sim.compute_totals(obj, dv)[obj, dv]

    print(f"Rank {rank} obj         : {obj.value}")
    print(f"Rank {rank} Analytical  : {analytical_grad}")



