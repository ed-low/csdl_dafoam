# BaseModel packages
from abc import ABC, abstractmethod

# DAFoamROMModel packages
from csdl_dafoam.core.csdl_dafoam import has_global_nan_or_inf
import numpy as np
from mpi4py import MPI
from dafoam import PYDAFOAM # For typing
from csdl_alpha import Variable, VariableGroup

# DAFoamLSPGModel
from contextlib import contextmanager, nullcontext



# region BASEMODEL
class BaseModel(ABC):  
    def __init__(self):
        pass
    
    # region evaluate_input_output
    @abstractmethod
    def evaluate_input_output(self):
        # FOR CSDL INTERFACE
        # Return input dict will be {name:variable}, output will be {name:str, shape:tuple}
        raise NotImplementedError()

    # region update_from_input_vals
    @abstractmethod
    def update_from_input_vals(self, input_vals):
        # FOR CSDL INTERFACE
        # Take the input_val dictionaries from the CSDL variable and update any of the values held here
        # or using any FOM API/interfacing
        raise NotImplementedError()

    # region input_jacvec_transpose
    @abstractmethod
    def input_jacvec_transpose(self, rom_state, vec, input_vals, mode):
        # FOR CSDL INTERFACE
        # Compute dR/dx^T@vec for x in input_vals (input values will be a dict with {name: np.ndarray})
        raise NotImplementedError()

    # region evaluate_residuals
    @abstractmethod
    def evaluate_residuals(self, rom_state):
        # FOR SOLVER AND CSDL INTERFACE
        raise NotImplementedError()
    
    # region write_solution
    @abstractmethod
    def write_solution(self, rom_state):
        # FOR CSDL INTERFACE
        raise NotImplementedError()

    # region compute_reduced_jacobian
    def compute_reduced_jacobian(self, rom_state):
        # FOR SOLVER
        # Optional reduced Jacobian evaluation.
        # Solver may want/need this or revert to FD or other strategy, depending on solver.
        return None
    
    # region freeze_jacobian
    def freeze_jacobian(self):
        # FOR SOLVER
        # Optional Context manager. Override in subclasses that have a freezable Jacobian.
        return nullcontext()
    
    # region print_fn
    @abstractmethod
    def print_fn(self, msg: str, **kwargs):
        # FOR SOLVER AND CSDL INTERFACE
        # Custom print function, if necessary (mainly for MPI business)
        print(msg, **kwargs)





# region DAFOAMPROJECTIONROMMODEL
class DAFoamProjectionROMModel(BaseModel):  
    def __init__(self,
                 dafoam_input_variables_group:VariableGroup,
                 pod_modes:np.ndarray,
                 reference_fom_state:np.ndarray,
                 scaling:np.ndarray,
                 weights:np.ndarray,
                 dafoam_instance:PYDAFOAM,
                 normalize_residuals:bool=True,
                 fd_step:float=1e-6,
                 jac_fd_central:bool=True,
    ):
        self.model_name          = "DAFoamProjectionModel"

        self.dafoam_input_variables_group = dafoam_input_variables_group
        self.pod_modes           = pod_modes            # Note we'll keep only the array representation
        self.reference_fom_state = reference_fom_state  # for these two variables in evaluate_input_output
        self.scaling             = scaling 
        self.weights             = weights
        self.dafoam_instance     = dafoam_instance
        self.normalize_residuals = normalize_residuals
        self.fd_step             = fd_step
        self.jac_fd_central      = jac_fd_central
        self.n_local_states      = dafoam_instance.getNLocalAdjointStates()

        # To be setup later
        self.n_modes             = None

        # Values for writing to disk
        self.solution_iter       = 0

        # convenient MPI parameters
        self.comm       = dafoam_instance.comm
        self.rank       = dafoam_instance.comm.rank
        self.comm_size  = dafoam_instance.comm.Get_size()


    # region evaluate_input_output
    def evaluate_input_output(self):
        # Return input dict will be {name:variable}, output will be {name:str, shape:tuple}
        # Initialize return dict for the input variables
        input_dict = {}

        # DAFoam solver inputs (this will get the CSDL variable from the group and assign it
        # to the dict. Names will be according to those found in the da_options)
        dafoam_input_dict = self.dafoam_instance.getOption("inputInfo")
        for name, info in dafoam_input_dict.items():
            if "solver" in info["components"]:
                input_dict[name] = getattr(self.dafoam_input_variables_group, name)

        if isinstance(self.pod_modes, Variable):
            input_dict["pod_modes"] = self.pod_modes
            self.pod_modes          = self.pod_modes.value.copy() # We'll keep only the array representation

        if isinstance(self.reference_fom_state, Variable):
            input_dict["reference_fom_state"] = self.reference_fom_state
            self.reference_fom_state          = self.reference_fom_state.value.copy()

        self.csdl_input_names = list(input_dict.keys())

        self.n_modes = self.pod_modes.shape[1]

        output_info = {"name":"dafoam_rom_states", "shape":(self.n_modes,)}

        return input_dict, output_info


    # region update_from_input_vals
    def update_from_input_vals(self, input_vals):
        self.dafoam_instance.set_solver_input(input_vals)

        if "pod_modes" in input_vals:
            self.pod_modes = input_vals["pod_modes"]

        if "reference_fom_state" in input_vals:
            self.reference_fom_state = input_vals["reference_fom_state"]


    # region input_jacvec_transpose
    def input_jacvec_transpose(self, rom_state, vec, input_vals, mode):
        # Will return this
        input_sensitivities = {}

        q = rom_state
        w = self._reconstruct_fom_state(rom_state=q)
        
        if not has_global_nan_or_inf(w, self.comm):
            self.dafoam_instance.setStates(w)
        else:
            self.print_fn("DAFoamROMModel: Detected NaN(s) in input_vals. Skipping DAFoam setStates")

        # Can't do forward mode
        if mode == 'fwd':
            raise NotImplementedError('forward mode has not been implemented for DAFoamROMModel')

        elif mode == 'rev':
            m   = self.weights
            Psi = self._get_test_basis(fom_state=w)
            Phi = self.pod_modes
            s   = self.scaling

            # Shared seed: M @ Psi @ lam (fom size, distributed)
            seed        = np.ascontiguousarray(m * (Psi @ vec))

            if not self.normalize_residuals:
                seed_norm   = seed.copy() * self.dafoam_instance.getStateWeights()
            else:
                seed_norm   = seed

            # FOM inputs: mesh, flow DVs
            input_dict = self.dafoam_instance.getOption("inputInfo")

            for input_name in list(input_vals.keys()):
                if input_name not in input_dict:
                    continue # skip reconstruction inputs, handled below

                input_type = input_dict[input_name]["type"]
                jac_input  = input_vals[input_name].copy()
                product    = np.zeros_like(jac_input)

                self.dafoam_instance.solverAD.calcJacTVecProduct(
                    input_name,
                    input_type,
                    jac_input,
                    "aero_residuals",
                    "residual",
                    seed_norm,
                    product
                )

                input_sensitivities[input_name] = product

            # Reconstruction inputs (these are handled if the POD reconstruction inputs are actually CSDL variables)
            needs_reconstruction_sens = any(k in self.csdl_input_names for k in ["reference_fom_state","pod_modes"])

            if needs_reconstruction_sens:
                # This vector is shared among all of the sensitivites. Compute once here
                # J^T M Psi lam
                v_shared = self._jacT_vec_product(fom_state=w, vec=seed)

                # Compute contribution from the reference state variable
                # (∂r_rom/∂w_ref)^T lam = (Psi^T M ∂r/∂w ∂w/∂w_ref)^T lam = (Psi^T M J ∂w/∂w_ref)^T lam
                # ∂w/∂w_ref = I
                # (∂r_rom/∂w_ref)^T lam = (Psi^T M J)^T lam = J^T M Psi lam = v_shared
                if "reference_fom_state" in self.csdl_input_names:
                    input_sensitivities["reference_fom_state"] = v_shared

                # Compute contribution from the scaling variable
                # (∂r_rom/∂s)^T lam = (Psi^T M ∂r/∂w ∂w/s)^T lam = (Psi^T M J ∂w/s)^T lam
                # ∂w/s = Phi q
                # (∂r_rom/∂s)^T lam = (Psi^T M J Phi q)^T lam = (Phi q) J^T M Psi lam = (Phi q) v_shared [Phi q is a vector, so transpose is dropped]
                if "scaling" in self.csdl_input_names:
                    Phi_q = Phi @ q # (n_local,)
                    input_sensitivities["scaling"] = Phi_q * v_shared

                # Compute contribution from the modes
                if "pod_modes" in self.csdl_input_names:
                    # We have two paths: one from the projection, and one from the reconstruction
                    # (∂r_rom/∂Phi)^T lam = [∂/∂Phi(Psi^T M R)] ^ T lam
                    # Where Psi and R have Phi dependence
                    # Being loose with notation here (since we have arrays)
                    # ∂/∂Phi(Psi^T M R) = ∂/∂Phi(Psi^T) M R + Psi^T M ∂/∂Phi(R)
                    #                    |----projection---| |-reconstruction--|

                    # Path 1: reconstruction (this is shared by both Galerkin and LSPG)
                    # [Psi^T M ∂/∂Phi(R)]^T lam
                    # dw = S dPhi q
                    input_sensitivities["pod_modes"] = s[:, None] * np.outer(v_shared, q)

                    # Path 2: projection - Phi appears in Psi^T M R
                    r = self._eval_fom_residual(fom_state=w)

                    # This will be dependent on the type of projection we use (Galerkin, LSPG)
                    # Will need to implement this _projection_phi_term for each model
                    input_sensitivities["pod_modes"] += self._projection_phi_term(w, r, vec)
    
        return input_sensitivities
    

    # region _projection_phi_term
    def _projection_phi_term(self, fom_state, fom_residual, vec):
        raise NotImplementedError
    

    # region evaluate_residuals
    def evaluate_residuals(self, rom_state):
        q = rom_state
        w = self._reconstruct_fom_state(q)
        r = self._eval_fom_residual(w)
        return self._project_and_reduce(distributed_val=r, fom_state=w)
    

    # region write_solution
    def write_solution(self, rom_state):
        q = rom_state
        w = self._reconstruct_fom_state(rom_state=q)
        r = self._eval_fom_residual(fom_state=w)

        # Write state to file 
        # Need to convert to the writing format. Will match DAFoam style, except use a different leading value
        # Eg, DAFoam writes to 0.0001, 0.0002, etc
        # Change the leading integer to 1 to write to 1.0001, 1.0002, etc
        leading_integer         = 1
        solution_write_number   = leading_integer + (self.solution_iter + 1) / 10000
        self.print_fn(f"Writing solution to {solution_write_number}.")

        # Write the state and residuals
        self.dafoam_instance.solver.writeAdjointFields("",     solution_write_number, w, True)
        self.dafoam_instance.solver.writeAdjointFields("res_", solution_write_number, r, True)

        # Write the mesh
        mesh = np.zeros_like(self.dafoam_instance.xv.flatten())
        self.dafoam_instance.solver.getOFMeshPoints(mesh)
        self.dafoam_instance.solver.writeMeshPoints(mesh, solution_write_number)
        self.solution_iter += 1
    

    # region compute_reduced_jacobian
    def compute_reduced_jacobian(self, rom_state):
        raise NotImplementedError("This needs to be implemented for the particular projection model.")


    # region _get_test_basis
    def _get_test_basis(self, fom_state=None):
        raise NotImplementedError("This needs to be implemented for the particular projection model.")


    # region _reconstruct_fom_state
    def _reconstruct_fom_state(self, rom_state):
        q     = rom_state
        Phi   = self.pod_modes
        s     = self.scaling
        w_ref = self.reference_fom_state
        w     = w_ref + s * (Phi @ q)
        return w
    

    # region _eval_fom_residual
    def _eval_fom_residual(self, fom_state):
        # This assumes we have already set the inputs! Should generally be the case
        w = fom_state
        dafoam_instance = self.dafoam_instance
        dafoam_instance.setStates(w)
        residuals = dafoam_instance.getResiduals()
        return residuals if self.normalize_residuals else residuals * dafoam_instance.getStateWeights()
        

    # region _project_and_reduce
    def _project_and_reduce(self, distributed_val, fom_state=None):
        comm = self.comm
        m    = self.weights
        Psi  = self._get_test_basis(fom_state)

        # Consider if the distributed value is a vector
        if distributed_val.ndim == 1 or (distributed_val.ndim == 2 and distributed_val.shape[1] == 1):
            distributed_val = np.reshape(distributed_val, (-1,))
            v_local = Psi.T @ (m * distributed_val)
            v       = np.zeros_like(v_local)
            comm.Allreduce(v_local, v, op=MPI.SUM)

        # Check if distributed value is a matrix
        elif distributed_val.ndim == 2:
            v_local = Psi.T @ (m[:, None] *  distributed_val)
            v       = np.zeros_like(v_local)
            comm.Allreduce(v_local, v, op=MPI.SUM)
        
        return v
    

    # region _jacT_vec_product
    def _jacT_vec_product(self, fom_state, vec):
        dafoam_instance = self.dafoam_instance
        v = vec
        w = fom_state

        dafoam_instance.setStates(w)

        seed    = np.ascontiguousarray(v.copy())
        product = np.zeros_like(seed)

        if not self.normalize_residuals:
            seed *= dafoam_instance.getStateWeights()
        
        dafoam_instance.solverAD.calcJacTVecProduct(
            'dafoam_solver_states',
            "stateVar",
            w,
            'aero_residuals',
            "residual",
            seed,
            product,
            )
        
        return product / dafoam_instance.getStateScalingFactors()
    

    # region _jac_mat_product
    def _jac_mat_product(self, fom_state, matrix, step=1e-6):
        w = fom_state
        M = matrix
        n = self.n_local_states

        # Check dimensions
        n_rows = M.shape[0]
        n_cols = M.shape[1]
        assert n == n_rows, f"Matrices must have compatible sizes! Jacobian has dimension ({n}, {n}), while supplied matrix has dimensions {M.shape}"

        r0 = self._eval_fom_residual(fom_state=w)

        JM = np.zeros_like(M)

        for i in range(n_cols):
            v        = M[:, i]
            JM[:, i] = self._jac_vec_product(fom_state=w, direction=v, fom_residual=r0, step=step, reset_state=False)

        self.dafoam_instance.setStates(w)  # reset OF state to w (due to perturbation step in _jac_vec_product)

        return JM


    # region _jac_vec_product
    def _jac_vec_product(self, fom_state, direction, fom_residual=None, step=1e-6, reset_state=True):
        w = fom_state
        v = direction

        # Scale h relative to the direction magnitude to avoid truncation/cancellation
        v_norm_local  = np.dot(v, v)
        v_norm_global = np.zeros(1)
        self.comm.Allreduce(v_norm_local, v_norm_global, op=MPI.SUM)
        v_norm = np.sqrt(v_norm_global[0])

        w_norm = np.sqrt(self.comm.allreduce(np.dot(w, w), op=MPI.SUM))
        h = step * (1.0 + w_norm) / v_norm if v_norm > 0 else step

        if self.jac_fd_central:
            # Central difference: O(h^2) accuracy, 2 residual evals, fom_residual unused
            r_fwd = self._eval_fom_residual(fom_state=w + h * v)
            r_bwd = self._eval_fom_residual(fom_state=w - h * v)
            result = (r_fwd - r_bwd) / (2 * h)
        else:
            # Forward difference: O(h) accuracy, 1 residual eval (+ r0 if not cached)
            r0     = self._eval_fom_residual(fom_state=w) if fom_residual is None else fom_residual
            r_fwd  = self._eval_fom_residual(fom_state=w + h * v)
            result = (r_fwd - r0) / h

        if reset_state:
            self.dafoam_instance.setStates(w)

        return result
        

    # region print_fn
    # Custom print function, if necessary (mainly for MPI business)
    def print_fn(self, msg: str, **kwargs):
        if self.rank == 0:
            print(msg, **kwargs)





# region DAFOAMGALERKINMODEL
class DAFoamGalerkinModel(DAFoamProjectionROMModel):
    def __init__(self, 
                 dafoam_input_variables_group:VariableGroup,
                 pod_modes:np.ndarray,
                 reference_fom_state:np.ndarray,
                 scaling:np.ndarray,
                 weights:np.ndarray,
                 dafoam_instance:PYDAFOAM,
                 normalize_residuals:bool=True,
                 fd_step:float=1e-6,
                 jac_fd_central:bool=True,
                 jac_mode:str="fd"
    ):
        super().__init__(
            dafoam_input_variables_group,
            pod_modes,
            reference_fom_state,
            scaling,
            weights,
            dafoam_instance,
            normalize_residuals,
            fd_step,
            jac_fd_central
        )
        
        self.jac_mode = jac_mode
        if jac_mode.lower() not in ["fd", "analytical"]:
            self.print_fn(f"WARNING: jac_mode {jac_mode} not recognized. Defaulting to finite difference (jac_mode='fd')")
            self.jac_mode = "fd"
    
    # region compute_reduced_jacobian
    def compute_reduced_jacobian(self, rom_state):
        q   = rom_state
        w   = self._reconstruct_fom_state(rom_state=q)
        Phi = self.pod_modes
        m   = self.weights
        s   = self.scaling

        if self.jac_mode == "fd":
            J_rom = self._project_and_reduce(self._jac_mat_product(fom_state=w, matrix=s[:, None] * Phi, step=self.fd_step))

        elif self.jac_mode == "analytical":
            JT_SMT_Phi  = self._jacT_mat_product(fom_state=w, matrix=(s * m)[:, None] * Phi)
            J_rom_local = JT_SMT_Phi.T @ (s[:, None] * Phi)

            J_rom = np.zeros_like(J_rom_local)
            self.comm.Allreduce(J_rom_local, J_rom, op=MPI.SUM)

        return J_rom


    # region _get_test_basis
    def _get_test_basis(self, fom_state=None):
        return self.scaling[:, None] * self.pod_modes


    # region _projection_phi_term
    def _projection_phi_term(self, fom_state, fom_residual, vec):
        # In this case, Psi = Phi
        # (See notes in DAFoamProjectionROMModel)
        s = self.scaling
        m = self.weights
        r = fom_residual
        return np.outer(s * m * r, vec)
    

    # region _jacT_mat_product
    def _jacT_mat_product(self, fom_state, matrix):
        M = matrix
        w = fom_state
        JT_M = np.zeros_like(M)

        for i in range(M.shape[1]):
            v           = M[:, i]
            JT_M[:, i]  = self._jacT_vec_product(fom_state=w, vec=v)

        return JT_M





# region DAFOAMLSPGMODEL
class DAFoamLSPGModel(DAFoamProjectionROMModel):
    def __init__(self,
                 dafoam_input_variables_group:VariableGroup,
                 pod_modes:np.ndarray,
                 reference_fom_state:np.ndarray,
                 scaling:np.ndarray,
                 weights:np.ndarray,
                 dafoam_instance:PYDAFOAM,
                 normalize_residuals:bool=True,
                 fd_step:float=1e-6,
                 jac_fd_central:bool=True,
    ):
        super().__init__(
            dafoam_input_variables_group,
            pod_modes,
            reference_fom_state,
            scaling,
            weights,
            dafoam_instance,
            normalize_residuals,
            fd_step,
            jac_fd_central
        )
        self._test_basis        = None
        self._freeze_test_basis = False


    # region compute_reduced_jacobian
    def compute_reduced_jacobian(self, rom_state):
        q     = rom_state
        w     = self._reconstruct_fom_state(rom_state=q)
        Psi   = self._get_test_basis(fom_state=w)
        J_rom = self._project_and_reduce(distributed_val=Psi)
        return J_rom
    

    # region freeze_jacobian
    @contextmanager
    def freeze_jacobian(self):
        self._freeze_test_basis = True
        try:
            yield
        finally:
            self._freeze_test_basis = False

    
    # region _get_test_basis
    def _get_test_basis(self, fom_state=None):
        Phi = self.pod_modes
        s   = self.scaling

        # If we're given a FOM state, we'll go ahead and recompute the basis
        if not self._freeze_test_basis and fom_state is not None:
            test_basis = self._jac_mat_product(fom_state=fom_state, matrix=s[:, None] * Phi, step=self.fd_step)
            self._test_basis = test_basis
            return test_basis

        elif self._test_basis is None:
            raise ValueError("Test basis not initialized. Provide 'fom_state' to compute it.")
        
        return self._test_basis
        

    # region _projection_phi_term
    def _projection_phi_term(self, fom_state, fom_residual, vec):
        # In this case, Psi = J S Phi
        # (See notes in DAFoamProjectionROMModel)
        s = self.scaling
        m = self.weights
        r = fom_residual
        JT_r = self._jacT_vec_product(fom_state=fom_state, vec=m * r)
        return s[:, None] * np.outer(JT_r, vec)


            
            



import numpy as np
from mpi4py import MPI
from csdl_dafoam.core.rom.rom_models import BaseModel


class SyntheticROMModel(BaseModel):

    def __init__(self, N=40, r=6, use_analytic_jacobian=True):
        super().__init__()

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size

        self.N = N
        self.r = r
        self.use_analytic_jacobian = use_analytic_jacobian

        # MPI Partitioning
        # # simple contiguous split
        # n_per_rank = N // self.size
        # start = self.rank * n_per_rank
        # end = start + n_per_rank if self.rank != self.size-1 else N
        # self.local_idx = np.arange(start, end)

        # Strided partition
        self.local_idx = np.arange(self.rank, self.N, self.size)

        # trial basis
        np.random.seed(0)
        V           = np.random.randn(N, r)
        self.V, _   = np.linalg.qr(V)
        self.V_local   = self.V[self.local_idx, :]

        # true FOM state
        self.w_true         = np.linspace(1.0, 2.0, N)
        self.w_true_local   = self.w_true[self.local_idx]

        # exact ROM solution
        self.q_true = self.V.T @ self.w_true


    # ------------------------------
    # CSDL placeholders
    # ------------------------------

    def evaluate_input_output(self):
        return {}, {}

    def update_from_input_vals(self, input_vals):
        pass

    def input_jacvec_transpose(self, rom_state, vec, input_vals, mode):
        raise NotImplementedError


    # ------------------------------
    # residual evaluation
    # ------------------------------

    def evaluate_residuals(self, rom_state):

        q = rom_state

        # reconstruct FOM state
        w = self.reconstruct_fom_state(q)
        f = self.compute_fom_residual(w)

        # ROM projection
        R_local = self.V_local.T @ f

        # Sum contributions across ranks
        R = self.comm.allreduce(R_local, op=MPI.SUM)

        return R


    # ------------------------------
    # analytic reduced Jacobian
    # ------------------------------

    def compute_reduced_jacobian(self, rom_state):

        if not self.use_analytic_jacobian:
            return None

        q = rom_state
        w = self.reconstruct_fom_state(q)

        J_fom_local = 2 * w

        J_local = self.V_local.T @ (np.diag(J_fom_local) @ self.V_local)

        # Reduce J across ranks
        J = np.zeros_like(J_local)
        self.comm.Allreduce(J_local, J, op=MPI.SUM)

        return J
    
    
    def reconstruct_fom_state(self, rom_state):
        return self.V_local @ rom_state
    
    def compute_fom_residual(self, fom_state):
        return fom_state**2 - self.w_true_local**2


    # ------------------------------
    # printing
    # ------------------------------

    def print_fn(self, msg: str, **kwargs):
        if self.rank == 0:
            print(msg, **kwargs)



import numpy as np
from mpi4py import MPI
from abc import ABC

class Burgers1DFOM(BaseModel):
    def __init__(self, N=50, nu=0.01, u0=1.0, u1=2.0, use_analytic_jacobian=True):
        super().__init__()

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size

        self.N = N
        self.nu = nu
        self.u0 = u0
        self.u1 = u1
        self.use_analytic_jacobian = use_analytic_jacobian

        # Grid
        self.x = np.linspace(0, 1, N)
        self.dx = self.x[1] - self.x[0]

        # MPI partitioning (strided)
        self.local_idx = np.arange(self.rank, N, self.size)
        
        # Initial guess
        self.u_init = np.linspace(u0, u1, N)[self.local_idx]

    # --------------------------
    # Evaluate residuals
    # --------------------------
    def evaluate_residuals(self, rom_state=None):
        # For FOM, rom_state is just the local state
        u = rom_state if rom_state is not None else self.u_init.copy()

        # Extend to full vector for boundary conditions
        u_full = np.zeros(self.N)
        # gather u_local to u_full
        all_u   = np.concatenate(self.comm.allgather(u))
        all_idx = np.concatenate(self.comm.allgather(self.local_idx))
        u_full[all_idx] = all_u

        R = np.zeros_like(u)
        idx = self.local_idx

        for i_local, i in enumerate(idx):
            if i == 0:
                R[i_local] = u_full[i] - self.u0  # left BC
            elif i == self.N-1:
                R[i_local] = u_full[i] - self.u1  # right BC
            else:
                # central differences for d^2u/dx^2
                du_dx = (u_full[i+1] - u_full[i-1]) / (2*self.dx)
                d2u_dx2 = (u_full[i+1] - 2*u_full[i] + u_full[i-1]) / (self.dx**2)
                R[i_local] = u_full[i]*du_dx - self.nu*d2u_dx2

        return R

    # --------------------------
    # Analytical Jacobian
    # --------------------------
    def compute_reduced_jacobian(self, rom_state=None):
        if not self.use_analytic_jacobian:
            return None

        u = rom_state if rom_state is not None else self.u_init.copy()
        N_local = len(u)
        J_local = np.zeros((N_local, N_local))

        # Extend to full vector for boundary conditions
        u_full = np.zeros(self.N)
        # gather u_local to u_full
        all_u   = np.concatenate(self.comm.allgather(u))
        all_idx = np.concatenate(self.comm.allgather(self.local_idx))
        u_full[all_idx] = all_u

        for i_local, i in enumerate(self.local_idx):
            if i == 0 or i == self.N-1:
                J_local[i_local, i_local] = 1.0
            else:
                # derivatives for interior points
                idx_m = i-1
                idx_c = i
                idx_p = i+1

                # check if these indices are local
                local_map = {idx: j for j, idx in enumerate(self.local_idx)}
                
                # central diff terms
                val_c = u_full[idx_c]*0.0  # placeholder
                val_m = -self.nu / (self.dx**2) - u_full[idx_c]/(2*self.dx)
                val_c = (u_full[idx_c+1] - u_full[idx_c-1])/(2*self.dx) + 2*self.nu/(self.dx**2)
                val_p = -self.nu/(self.dx**2) + u_full[idx_c]/(2*self.dx)

                for idx_, val in zip([idx_m, idx_c, idx_p], [val_m, val_c, val_p]):
                    if idx_ in local_map:
                        J_local[i_local, local_map[idx_]] = val

        return J_local

    # --------------------------
    # CSDL placeholders
    # --------------------------
    def evaluate_input_output(self):
        return {}, {}

    def update_from_input_vals(self, input_vals):
        pass

    def input_jacvec_transpose(self, rom_state, vec, input_vals, mode):
        raise NotImplementedError

    # --------------------------
    # Printing
    # --------------------------
    def print_fn(self, msg, **kwargs):
        if self.rank == 0:
            print(msg, **kwargs)