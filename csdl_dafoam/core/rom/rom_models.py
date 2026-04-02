# BaseModel packages
from abc import ABC, abstractmethod

# DAFoamROMModel packages
from csdl_dafoam.core.csdl_dafoam import has_global_nan_or_inf
import numpy as np
from mpi4py import MPI
from dafoam import PYDAFOAM # For typing
import csdl_alpha as csdl   # For typing and checking

# region BASEMODEL
class BaseModel(ABC):  
    def __init__(self):
        pass
    
    # region evaluate_input_output
    @abstractmethod
    def evaluate_input_output(self):
        # Return input dict will be {name:variable}, output will be {name:str, shape:tuple}
        raise NotImplementedError()

    # region update_from_input_vals
    @abstractmethod
    def update_from_input_vals(self, input_vals):
        # Take the input_val dictionaries from the CSDL variable and update any of the values held here
        # or using any FOM API/interfacing
        raise NotImplementedError()

    # region input_jacvec_transpose
    @abstractmethod
    def input_jacvec_transpose(self, rom_state, vec, input_vals, mode):
        # Compute dR/dx^T@vec for x in input_vals (input values will be a dict with {name: np.ndarray})
        raise NotImplementedError()

    # region evaluate_residuals
    @abstractmethod
    def evaluate_residuals(self, rom_state):
        raise NotImplementedError()

    # region compute_reduced_jacobian
    def compute_reduced_jacobian(self, rom_state):
        # Optional reduced Jacobian evaluation.
        # Solver may want/need this or revert to FD or other strategy, depending on solver.
        return None
    
    # region print_fn
    @abstractmethod
    def print_fn(self, msg: str, **kwargs):
    # Custom print function, if necessary (mainly for MPI business)
        print(msg, **kwargs)





# region DAFOAMPROJECTIONROMMODEL
class DAFoamProjectionROMModel(BaseModel):  
    def __init__(self,
                 dafoam_input_variables_group:csdl.VariableGroup,
                 pod_modes:np.ndarray,
                 reference_fom_state:np.ndarray,
                 scaling:np.ndarray,
                 weights:np.ndarray,
                 dafoam_instance:PYDAFOAM,
                 fd_step:float=1e-6,
                 jac_fd_central:bool=True,
     ):
        self.dafoam_input_variables_group = dafoam_input_variables_group
        self.pod_modes           = pod_modes            # Note we'll keep only the array representation
        self.reference_fom_state = reference_fom_state  # for these two variables in evaluate_input_output
        self.scaling             = scaling 
        self.weights             = weights
        self.dafoam_instance     = dafoam_instance
        self.fd_step             = fd_step
        self.jac_fd_central      = jac_fd_central
        self.n_local_states      = dafoam_instance.getNLocalAdjointStates()

        # To be setup later
        self.n_modes             = None

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

        if isinstance(self.pod_modes, csdl.Variable):
            input_dict["pod_modes"] = self.pod_modes
            self.pod_modes          = self.pod_modes.value.copy() # We'll keep only the array representation

        if isinstance(self.reference_fom_state, csdl.Variable):
            input_dict["reference_fom_state"] = self.reference_fom_state
            self.reference_fom_state          = self.reference_fom_state.value.copy()

        self.csdl_input_names = list(input_dict.keys())

        self.n_modes = self.pod_modes.shape[1]

        output_info = {"dafoam_rom_states": (self.n_modes,)}

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
        
        if not has_global_nan_or_inf(w):
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
            seed_norm   = seed.copy()

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
        w     = w_ref + s[:, None] * (Phi @ q)
        return w
    

    # region _eval_fom_residual
    def _eval_fom_residual(self, fom_state):
        # This assumes we have already set the inputs! Should generally be the case
        w = fom_state
        dafoam_instance = self.dafoam_instance
        dafoam_instance.setStates(w)
        residuals = dafoam_instance.getResiduals()
        return residuals
        

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

        seed    = np.ascontiguousarray(v.copy())
        product = np.zeros_like(seed)
        
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
                 dafoam_input_variables_group:csdl.VariableGroup,
                 pod_modes:np.ndarray,
                 reference_fom_state:np.ndarray,
                 scaling:np.ndarray,
                 weights:np.ndarray,
                 dafoam_instance:PYDAFOAM,
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
        return self.pod_modes


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
                 dafoam_input_variables_group:csdl.VariableGroup,
                 pod_modes:np.ndarray,
                 reference_fom_state:np.ndarray,
                 scaling:np.ndarray,
                 weights:np.ndarray,
                 dafoam_instance:PYDAFOAM,
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
            fd_step,
            jac_fd_central
        )
        self._test_basis = None


    # region compute_reduced_jacobian
    def compute_reduced_jacobian(self, rom_state):
        q     = rom_state
        w     = self._reconstruct_fom_state(rom_state=q)
        Psi   = self._get_test_basis(fom_state=w)
        J_rom = self._project_and_reduce(distributed_val=Psi)
        return J_rom

    
    # region _get_test_basis
    def _get_test_basis(self, fom_state=None):
        Phi = self.pod_modes
        s   = self.scaling

        # If we're given a FOM state, we'll go ahead and recompute the basis
        if fom_state is not None or self._test_basis is None:
            if self._test_basis is None:
                self.print_fn("Test basis not initialized. Computing...")

            test_basis = self._jac_mat_product(fom_state=fom_state, matrix=s[:, None] * Phi, step=self.fd_step)
            self._test_basis = test_basis
            return test_basis

        # If we're not given a FOM state, we'll return the stored basis
        elif fom_state is None:
            return self._test_basis
        
        else:
            raise ValueError("Test basis has not been initialized. You must provide 'fom_state' to compute it.")
        

    # region _projection_phi_term
    def _projection_phi_term(self, fom_state, fom_residual, vec):
        # In this case, Psi = J S Phi
        # (See notes in DAFoamProjectionROMModel)
        s = self.scaling
        m = self.weights
        r = fom_residual
        JT_r = self._jacT_vec_product(fom_state=fom_state, vec=m * r)
        return s[:, None] * np.outer(JT_r, vec)


            
            





    