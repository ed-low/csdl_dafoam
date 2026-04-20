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

    # Block partitioning
    rows_per_rank = m // comm_size
    remainder = m % comm_size
    start = rank * rows_per_rank + min(rank, remainder)
    end   = start + rows_per_rank + (1 if rank < remainder else 0)
    U0_local_np = U0_np[start:end, :]
    U1_local_np = U1_np[start:end, :]

    # # Strided partitioning
    # U0_local_np = U0_np[rank::comm_size, :]
    # U1_local_np = U1_np[rank::comm_size, :]

    # CSDL Setup
    recorder = csdl.Recorder(inline=True, debug=True)
    recorder.start()

    # CSDL variables
    alpha    = csdl.Variable(value=1)
    U0       = alpha * U0_np
    U1       = alpha * U1_np
    U0_local = global_local_op(alpha, U0_local_np, lambda x,y:x*y, comm=comm) #csdl.Variable(value=U0_local_np)
    U1_local = global_local_op(alpha, U1_local_np, lambda x,y:x*y, comm=comm) #csdl.Variable(value=U1_local_np) 

    manifold_local  = Grassmann(m, k, comm=comm)

    gamma_local = manifold_local.log(U0_local, U1_local)
    U05_local   = manifold_local.exp(U0_local, 0.5 * gamma_local)
    angles      = manifold_local.subspace_angles(U0_local, U05_local)
    obj_local   = csdl.norm(angles) / comm_size
    obj_global  = csdl.experimental.mpi.mpi_sum(obj_local, comm)
    
    # obj_local   = csdl.experimental.mpi.mpi_sum(csdl.sum(U05_local), comm=comm)
    #     mpi_region.set_as_global_output(obj_local)
    
    # obj_local = csdl.experimental.mpi.mpi_sum(csdl.sum(gamma_local), comm=comm)
    dv  = alpha
    obj = obj_global

    recorder.stop()
    sim = csdl.experimental.PySimulator(recorder=recorder)

    analytical_grad  = sim.compute_totals(obj, dv)[obj, dv]
    # finite_diff_grad = sim.compute_totals(obj, dv, use_finite_difference=True)[obj, dv]


    print(f"Rank {rank} obj         : {obj.value}")
    print(f"Rank {rank} Analytical  : {analytical_grad}")
    # print(f"Rank {rank} Finite Diff : {finite_diff_grad}")






















##################################
####### ORIGINAL CHECK ###########
##################################


    # # Make a timer for the funciton calls
    # def timed_call(func, *args, **kwargs):
    #     start = time.perf_counter()
    #     result = func(*args, **kwargs)
    #     end = time.perf_counter()
    #     dt  = comm.allreduce(end - start, op=MPI.MAX) #op=MPI.SUM) / comm_size
    #     return result, dt
    
    # # We'll put all of the repeated operations in a single function
    # def log_and_exp(manifold:Grassmann, Y0:csdl.Variable|np.ndarray, Y1:csdl.Variable|np.ndarray):
    #     Ydot,       time_log = timed_call(manifold.log, Y0=Y0, Y1=Y1)
    #     Ydot_eval = 0.5 * Ydot
    #     Y1_mapped,  time_exp = timed_call(manifold.exp, Y0=Y0, Ydot=Ydot_eval)
    #     return {"tangent":Ydot, "predicted":Y1_mapped, "log_time":time_log, "exp_time":time_exp}

    # # mxn matrix retaining k modes
    # m = 10
    # n = 6
    # k = 3

    # # Generate bases
    # np.random.seed(0) # Set seed for rank consistency
    # print(f"Generating arrays...") if rank == 0 else None
    # A0 = np.random.random((m, n))
    # A1 = np.random.random((m, n))

    # print(f"Forming bases...") if rank == 0 else None
    # U0, _ = np.linalg.qr(A0)
    # U1, _ = np.linalg.qr(A1)

    # # Numpy variables
    # U0       = U0[:, :k]
    # U1       = U1[:, :k]

    # # Block partitioning
    # rows_per_rank = m // comm_size
    # remainder = m % comm_size
    # start = rank * rows_per_rank + min(rank, remainder)
    # end   = start + rows_per_rank + (1 if rank < remainder else 0)
    # U0_local = U0[start:end, :]
    # U1_local = U1[start:end, :]

    # # # Strided partitioning
    # # U0_local = U0[rank::comm_size, :]
    # # U1_local = U1[rank::comm_size, :]

    # # CSDL Setup
    # recorder = csdl.Recorder(inline=True, debug=True)
    # recorder.start()

    # # CSDL variables
    # U0_csdl       = csdl.Variable(value=U0)
    # U1_csdl       = csdl.Variable(value=U1)
    # U0_local_csdl = csdl.Variable(value=U0_local)
    # U1_local_csdl = csdl.Variable(value=U1_local)

    # # Prepping the manifolds
    # manifold_serial       = Grassmann(m, k)
    # manifold_distributed  = Grassmann(m, k, comm=comm)

    # # CASES
    # # 1) Y0, Y1: NUMPY,  NUMPY
    # # 2) Y0, Y1: CSDL,   NUMPY
    # # 3) Y0, Y1: NUMPY,  CSDL
    # # 4) Y0, Y1: CSDL,   CSDL

    # # Diagnostic value setup
    # data = {f"case{i}": {"serial": {}, "distributed": {}} for i in range(1, 5)}

    # data["case1"]["serial"]      = {"manifold":manifold_serial,      "Y0":U0,            "Y1":U1}
    # data["case2"]["serial"]      = {"manifold":manifold_serial,      "Y0":U0_csdl,       "Y1":U1}
    # data["case3"]["serial"]      = {"manifold":manifold_serial,      "Y0":U0,            "Y1":U1_csdl}
    # data["case4"]["serial"]      = {"manifold":manifold_serial,      "Y0":U0_csdl,       "Y1":U1_csdl}

    # data["case1"]["distributed"] = {"manifold":manifold_distributed, "Y0":U0_local,      "Y1":U1_local}
    # data["case2"]["distributed"] = {"manifold":manifold_distributed, "Y0":U0_local_csdl, "Y1":U1_local}
    # data["case3"]["distributed"] = {"manifold":manifold_distributed, "Y0":U0_local,      "Y1":U1_local_csdl}
    # data["case4"]["distributed"] = {"manifold":manifold_distributed, "Y0":U0_local_csdl, "Y1":U1_local_csdl}
    
    # with csdl.experimental.mpi.enter_mpi_region(rank, comm) as mpi_region:

    #     # Loop through cases
    #     for case, type_dict in data.items():
    #         for type, value_dict in type_dict.items():
    #             print(f"Running {case}, {type}...") if rank == 0 else None
    #             manifold = value_dict.pop("manifold")
    #             Y0       = value_dict.pop("Y0")
    #             Y1       = value_dict.pop("Y1")
                
    #             out_values = log_and_exp(manifold=manifold, Y0=Y0, Y1=Y1)

    #             Y1_pred = out_values["predicted"]
                
    #             comm_or_none = comm if type == "distributed" else None

    #             init_angles = manifold.subspace_angles(Y0=Y0, Y1=Y1)
    #             pred_angles = manifold.subspace_angles(Y0=Y1, Y1=Y1_pred)

    #             init_dist   = csdl.norm(init_angles) if _is_csdl(init_angles) else np.linalg.norm(init_angles)
    #             pred_dist   = csdl.norm(pred_angles) if _is_csdl(pred_angles) else np.linalg.norm(pred_angles)
            
    #             value_dict.update({"exp_time":out_values["exp_time"],
    #                             "log_time":out_values["log_time"],
    #                             "Y0Y1_angle": init_dist.value[0] if _is_csdl(init_dist) else init_dist,
    #                             "Y1Y1_angle": pred_dist.value[0] if _is_csdl(pred_dist) else pred_dist})
                
    #             obj = pred_dist
                    
    #     mpi_region.set_as_global_output(obj)
            
    # # Print the output
    # if rank == 0:
    #     for key, case_dict in data.items():
    #         df = pd.DataFrame.from_dict(case_dict, orient="index")
    #         pd.set_option("display.float_format", "{:.3e}".format)
    #         print(key)
    #         print("-------")
    #         print(df)
    #         print("")

    # U0_local_csdl.set_as_design_variable()
    # obj.set_as_objective()

    # recorder.stop()

    # sim = csdl.experimental.PySimulator(recorder=recorder)

    # # Manual derivative check
    # analytical_grad  = sim.compute_totals(obj, U0_local_csdl)[obj, U0_local_csdl]
    # finite_diff_grad = sim.compute_totals(obj, U0_local_csdl, use_finite_difference=True, )[obj, U0_local_csdl]

    # print(f"Rank {rank} Analytical  : {analytical_grad}")
    # print(f"Rank {rank} Finite Diff : {finite_diff_grad}")

    # # # Check derivatives
    # # import modopt as mo

    # # sim         = csdl.experimental.PySimulator(recorder=recorder)
    # # prob        = mo.CSDLAlphaProblem(problem_name="test", simulator=sim)
    # # optimizer   = mo.SLSQP(problem=prob, solver_options={'ftol':1e-6, 'maxiter':20})
    # # optimizer.check_first_derivatives(step=1e-6)






##################################
####### OLD CODE SNIPPETS ########
##################################

    # # region subspace_angles
    # def subspace_angles(self, Y0:csdl.Variable|np.ndarray, Y1:csdl.Variable|np.ndarray):
    #     G = self._inner_product(Y0, Y1)
    #     U, sigma, VT = self._svd(G, is_global=True)

    #     # Residual matrix directly — its singular values ARE sin(theta)
    #     # R = Y1 - Y0 @ U @ VT  (shape: n_dof x n_modes, distributed)
    #     Q = U @ VT         # n_modes x n_modes, global
    #     R = Y1 - Y0 @ Q             # n_dof x n_modes, distributed

    #     # All ranks should agree on Q
    #     Q_rank0 = comm.bcast(Q.value if isinstance(Q, csdl.Variable) else Q, root=0)
    #     assert np.allclose(Q.value if isinstance(Q, csdl.Variable) else Q, Q_rank0), f"Q mismatch on rank {comm.rank}"

    #     # SVD of R directly (not R^T R) — singular values in [0,1], no sqrt needed
    #     _, sin_sigma, _ = self._svd(R, is_global=False)  # distributed SVD of R

    #     # arctan2: both inputs are clean, no clipping, no sqrt of near-zero
    #     angles = self._arctan2(sin_sigma, sigma)
    #     return angles