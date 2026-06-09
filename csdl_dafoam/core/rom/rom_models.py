# BaseModel packages
from abc import ABC, abstractmethod
from dataclasses import dataclass

# DAFoamROMModel packages
from csdl_dafoam.core.csdl_dafoam import has_global_nan_or_inf
import numpy as np
from mpi4py import MPI
from dafoam import PYDAFOAM # For typing
from csdl_alpha import Variable, VariableGroup

# DAFoamLSPGModel
from contextlib import contextmanager, nullcontext



# region NORMALIZATIONCONFIG
@dataclass
class NormalizationConfig:
    """Fixed global reference values for non-dimensionalisation of compressible flow ROMs.

    Usage (no DAFoam dependency — safe to use in offline basis scripts):
        cfg = NormalizationConfig(p_ref=..., rho_ref=..., U_ref=..., T_ref=...,
                                  nu_ref=..., M_ref=..., cv=...)
        s = cfg.make_scaling_vector(state_indices)
        w = cfg.make_chu_weight_vector(state_indices)
    """
    p_ref:   float          # reference pressure [Pa]
    rho_ref: float          # reference density [kg/m^3]
    U_ref:   float          # reference velocity magnitude [m/s]
    T_ref:   float          # reference temperature [K]
    nu_ref:  float          # reference kinematic viscosity [m^2/s]
    M_ref:   float          # reference Mach number (for Chu energy weights)
    cv:      float          # specific heat at constant volume [J/(kg K)]
    gamma:   float = 1.4    # ratio of specific heats

    # ------------------------------------------------------------------
    # Scaling vector
    # ------------------------------------------------------------------
    def make_scaling_vector(self, state_indices: dict) -> np.ndarray:
        """Per-DOF linear scaling denominators s (length n_local_states).

        Variable conventions (DAFoam getStateVariableMap with includeComponentSuffix=False):
            p       -> rho_ref * U_ref^2   (dynamic pressure scale)
            U       -> U_ref               (all velocity components grouped as "U")
            T       -> T_ref
            nuTilda -> 1.0                 (log scaling is nonlinear; handled at reconstruction)
            phi     -> 1.0                 (face flux; not in ROM basis)
            other   -> 1.0  (with warning)
        """
        n_total = sum(mask.sum() for mask in state_indices.values())
        s = np.ones(n_total)

        _var_scale = {
            "p":       self.rho_ref * self.U_ref**2,
            "U":       self.U_ref,
            "T":       self.T_ref,
            "nuTilda": 1.0,
            "phi":     1.0,
        }

        for var, mask in state_indices.items():
            if var in _var_scale:
                s[mask] = _var_scale[var]
            else:
                print(f"NormalizationConfig.make_scaling_vector: unknown variable '{var}', defaulting to 1.0")

        return s

    # ------------------------------------------------------------------
    # Chu energy weight vector
    # ------------------------------------------------------------------
    def make_chu_weight_vector(self, state_indices: dict) -> np.ndarray:
        """Per-DOF Chu energy norm weights on *scaled* variables (length n_local_states).

        Cell volumes V_i are applied externally (multiply result by expand_cell_volumes output).

        Weights on scaled variables:
            p*      -> M_ref^2
            U*      -> 1.0   (each velocity component)
            T*      -> 1.0 / ((gamma - 1) * gamma * M_ref^2)
            nuTilda*-> 1.0   (turbulence, unit weight; cell volume applied externally)
            phi     -> 1.0   (not in basis; weight inconsequential)
            other   -> 1.0  (with warning)
        """
        n_total = sum(mask.sum() for mask in state_indices.values())
        w = np.ones(n_total)

        T_weight = 1.0 / ((self.gamma - 1.0) * self.gamma * self.M_ref**2)
        _var_weight = {
            "p":       self.M_ref**2,
            "U":       1.0,
            "T":       T_weight,
            "nuTilda": 1.0,
            "phi":     1.0,
        }

        for var, mask in state_indices.items():
            if var in _var_weight:
                w[mask] = _var_weight[var]
            else:
                print(f"NormalizationConfig.make_chu_weight_vector: unknown variable '{var}', defaulting to 1.0")

        return w

    # ------------------------------------------------------------------
    # Spalart–Allmaras log scaling (applied to nuTilda DOFs)
    # ------------------------------------------------------------------
    def apply_log_scaling_sa(self, nu_tilda: np.ndarray) -> np.ndarray:
        """Map nuTilda -> log(1 + nuTilda / nu_ref) elementwise (numerically stable)."""
        return np.log1p(nu_tilda / self.nu_ref)

    def invert_log_scaling_sa(self, nu_tilda_star: np.ndarray) -> np.ndarray:
        """Inverse: nu_ref * (exp(nu_tilda_star) - 1) elementwise (numerically stable)."""
        return self.nu_ref * np.expm1(nu_tilda_star)


# region EXPAND_CELL_VOLUMES
def expand_cell_volumes(cell_volumes: np.ndarray, state_indices: dict) -> np.ndarray:
    """Expand a per-cell volume array to a per-DOF array.

    For each variable block identified by state_indices, repeats V_i for every
    DOF belonging to cell i.

    NOTE: assumes cell-major (interleaved) DOF ordering for vector fields, i.e.
    [cell_0=(comp0,comp1,comp2), cell_1=(comp0,comp1,comp2), ...],
    consistent with OpenFOAM/DAFoam and with how _compute_split_pod_modes weights snapshots.

    Parameters
    ----------
    cell_volumes : (n_local_cells,) array of mesh cell volumes (distributed).
    state_indices : dict mapping variable names to boolean DOF index arrays,
                    as returned by getStateVariableMap(includeComponentSuffix=False).

    Returns
    -------
    (n_local_dofs,) array of per-DOF volumes.
    """
    n_local_cells = len(cell_volumes)
    n_total_dofs  = sum(mask.sum() for mask in state_indices.values())
    result        = np.ones(n_total_dofs)  # default 1.0 for non-cell variables (e.g. phi)

    for var, mask in state_indices.items():
        n_var_dofs = int(mask.sum())
        if n_var_dofs % n_local_cells != 0:
            # Face variable (e.g. phi): DOF count not divisible by n_cells — leave at 1.0.
            # phi rows are zero in the combined basis so this weight has no effect.
            continue
        n_components = n_var_dofs // n_local_cells
        # DAFoam/OpenFOAM stores vector fields in cell-major (interleaved) order:
        # [Ux0,Uy0,Uz0, Ux1,Uy1,Uz1, ...] so each cell's volume repeats n_components times.
        # np.repeat is consistent with how _compute_split_pod_modes weights vector snapshots.
        result[mask] = np.repeat(cell_volumes, n_components)

    return result


# region BASEMODEL
class BaseModel(ABC):  
    def __init__(self, disable_presolve_diagnostics:bool=False):
        self.disable_presolve_diagnostics = disable_presolve_diagnostics
    
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

    # region pre_solve_diagnostics
    def pre_solve_diagnostics(self, initial_rom_state: np.ndarray) -> None:
        # FOR SOLVER: called once before the Newton loop begins (after initial residual/basis setup).
        # Print any pre-solve diagnostics here (basis quality, scaling, etc.). No return value.
        if self.disable_presolve_diagnostics:
            return None

    # region iter_diagnostic_headers
    def iter_diagnostic_headers(self) -> list:
        # FOR SOLVER: return list of (header_str, column_width) pairs for extra Newton iteration columns.
        # Called once to build the table header row; order must match iter_diagnostics().
        return []

    # region iter_diagnostics
    def iter_diagnostics(self, rom_state: np.ndarray, jacobian: np.ndarray = None) -> list:
        # FOR SOLVER: return list of float values for extra Newton iteration columns.
        # Order must match iter_diagnostic_headers(). Return [] if no extra columns.
        return []

    # region post_solve_diagnostics
    def post_solve_diagnostics(self, result) -> None:
        # FOR SOLVER: called once after the Newton loop completes (converged or maxiter).
        # result is a SolverResult. Print any post-solve diagnostics here. No return value.
        pass

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
                 pod_modes:Variable|np.ndarray,
                 reference_fom_state:Variable|np.ndarray,
                 scaling:np.ndarray,
                 weights:np.ndarray,
                 dafoam_instance:PYDAFOAM,
                 normalize_residuals:bool=True,
                 fd_step:float=1e-6,
                 jac_fd_central:bool=True,
                 solution_leading_integer:int=1,
                 solution_prefix:str|None=None,
                 disable_presolve_diagnostics:bool=False
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
        self.solution_leading_integer = solution_leading_integer
        self.solution_prefix     = solution_prefix
        self.disable_presolve_diagnostics = disable_presolve_diagnostics

        # To be setup later
        self.n_modes             = None

        # Values for writing to disk
        self.solution_iter       = 0

        # convenient MPI parameters
        self.comm       = dafoam_instance.comm
        self.rank       = dafoam_instance.comm.rank
        self.comm_size  = dafoam_instance.comm.Get_size()

        # State variable index map — used by diagnostics for per-variable breakdowns
        names, indices       = dafoam_instance.getStateVariableMap(includeComponentSuffix=False)
        self.state_indices   = {name: indices == names.index(name) for name in names}


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

        vec = self.comm.allreduce(vec, op=MPI.SUM)
        # print(f"Rank {self.comm.rank}: vec norm before={np.linalg.norm(vec):.6f}, after={np.linalg.norm(vec_full):.6f}")

        q = rom_state
        w = self._reconstruct_fom_state(rom_state=q)
        
        if not has_global_nan_or_inf(w, self.comm):
            self._set_fom_states(w)
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

                # print(f"[Rank {self.rank}] input={input_name}, type={input_type}, product={product}")

                input_sensitivities[input_name] = product

            # Reconstruction inputs (these are handled if the POD reconstruction inputs are actually CSDL variables)
            needs_reconstruction_sens = any(k in self.csdl_input_names for k in ["reference_fom_state","pod_modes"])

            if needs_reconstruction_sens:
                # This vector is shared among all of the sensitivites. Compute once here
                # J^T M Psi lam
                v_shared = self._jacT_vec_product(fom_state=w, vec=seed)

                v_shared_gnorm = np.sqrt(self.comm.allreduce(np.dot(v_shared, v_shared), op=MPI.SUM))

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

                    pod_sens_gnorm = np.sqrt(self.comm.allreduce(
                                np.sum(input_sensitivities["pod_modes"]**2), op=MPI.SUM))

                    # Path 2: projection - Phi appears in Psi^T M R
                    r = self._eval_fom_residual(fom_state=w)

                    # This will be dependent on the type of projection we use (Galerkin, LSPG)
                    # Will need to implement this _projection_phi_term for each model
                    input_sensitivities["pod_modes"] += self._projection_phi_term(w, r, vec)

                    if self.rank == 0:
                        print(f"[Diag] v_shared global norm: {v_shared_gnorm:.8e}")
                        print(f"[Diag] pod_modes sens global norm: {pod_sens_gnorm:.8e}")
    
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
        leading_integer         = self.solution_leading_integer
        solution_write_number   = leading_integer + (self.solution_iter + 1) / 10000
        solution_prefix         = "" if self.solution_prefix is None else self.solution_prefix
        self.print_fn(f"Writing solution to {solution_write_number}.")

        # Write the state and residuals
        self.dafoam_instance.solver.writeAdjointFields(f"{solution_prefix}_",     solution_write_number, w, True)
        self.dafoam_instance.solver.writeAdjointFields(f"{solution_prefix}_res_", solution_write_number, r, True)

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
        self._set_fom_states(w)
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

        self._set_fom_states(w)

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

        self._set_fom_states(w)  # reset OF state to w (due to perturbation step in _jac_vec_product)

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
            self._set_fom_states(w)

        return result
        

    # region _set_fom_states
    def _set_fom_states(self, w: np.ndarray) -> None:
        """Set DAFoam states. Subclasses may override to add post-set processing."""
        self.dafoam_instance.setStates(w)

    # region print_fn
    # Custom print function, if necessary (mainly for MPI business)
    def print_fn(self, msg: str, **kwargs):
        if self.rank == 0:
            print(msg, **kwargs)


    # -----------------------------------------------------------------------
    # region DIAGNOSTICS
    # -----------------------------------------------------------------------

    # region pre_solve_diagnostics
    def pre_solve_diagnostics(self, initial_rom_state: np.ndarray) -> None:
        if self.disable_presolve_diagnostics:
            return None

        W   = 72
        sep = "-" * W
        self.print_fn(f"\n{sep}")
        self.print_fn(f"  DAFoam ROM — Pre-Solve Diagnostics")
        self.print_fn(sep)

        # --- FOM residual at reference state (q=0) ---
        w_ref     = self.reference_fom_state
        r_ref     = self._eval_fom_residual(fom_state=w_ref)
        r_ref_sq  = self.comm.allreduce(np.dot(r_ref, r_ref), op=MPI.SUM)
        r_ref_norm = np.sqrt(r_ref_sq)
        self.print_fn(f"  {'‖r_fom(w_ref)‖ (q=0 initial residual)':<40} {r_ref_norm:.4e}")
        self.print_fn(f"  {'Per-variable ‖r_fom(w_ref)‖':<40}")
        for var, idx in self.state_indices.items():
            rv    = r_ref[idx]
            n_v   = np.sqrt(self.comm.allreduce(np.dot(rv, rv), op=MPI.SUM))
            self.print_fn(f"    {var:>10}: {n_v:.4e}")
        self.print_fn("")

        # --- Basis orthogonality: Phi^T M Phi = I ---
        Phi = self.pod_modes
        m   = self.weights
        n   = Phi.shape[1]

        G_local     = Phi.T @ (m[:, None] * Phi)
        G           = self.comm.allreduce(G_local, op=MPI.SUM)
        ortho_error = np.linalg.norm(G - np.eye(n), "fro")
        ortho_ok    = ortho_error < 1e-10

        self.print_fn(
            f"  {'Basis ortho ‖Φᵀ M Φ - I‖_F':<36} "
            + (f"PASS ({ortho_error:.2e})" if ortho_ok else f"WARN ({ortho_error:.2e})")
        )

        # --- Scaling / weight summary per state variable ---
        s = self.scaling
        self.print_fn(f"\n  {'Variable':>10}  {'S min':>12}  {'S max':>12}  {'M min':>12}  {'M max':>12}")
        self.print_fn(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
        for var, idx in self.state_indices.items():
            s_v = s[idx]
            m_v = m[idx]
            self.print_fn(
                f"  {var:>10}"
                f"  {self.comm.allreduce(s_v.min(), op=MPI.MIN):>12.4e}"
                f"  {self.comm.allreduce(s_v.max(), op=MPI.MAX):>12.4e}"
                f"  {self.comm.allreduce(m_v.min(), op=MPI.MIN):>12.4e}"
                f"  {self.comm.allreduce(m_v.max(), op=MPI.MAX):>12.4e}"
            )

        # --- Mode energy breakdown per state variable ---
        self.print_fn(f"\n  Mode M-weighted energy fraction per variable")
        var_names   = list(self.state_indices.keys())
        header      = f"  {'Mode':>6}" + "".join(f"  {v:>10}" for v in var_names) + f"  {'‖φ‖²_M':>10}"
        self.print_fn(header)
        self.print_fn("  " + "-" * (len(header) - 2))
        for k in range(n):
            phi_k = Phi[:, k]
            total = self.comm.allreduce(np.sum(m * phi_k**2), op=MPI.SUM)
            fracs = []
            for var in var_names:
                idx   = self.state_indices[var]
                v_tot = self.comm.allreduce(np.sum(m[idx] * phi_k[idx]**2), op=MPI.SUM)
                fracs.append(v_tot / max(total, 1e-300))
            row = f"  {k:>6d}" + "".join(f"  {f:>10.4f}" for f in fracs) + f"  {total:>10.4f}"
            self.print_fn(row)

        self.print_fn(sep)


    # region iter_diagnostic_headers
    def iter_diagnostic_headers(self) -> list:
        return [("cond(J_rom)", 12)]


    # region iter_diagnostics
    def iter_diagnostics(self, rom_state: np.ndarray, jacobian: np.ndarray = None) -> list:
        cond = np.linalg.cond(jacobian) if jacobian is not None else float("nan")
        return [cond]


    # region post_solve_diagnostics
    def post_solve_diagnostics(self, result) -> None:
        W   = 72
        sep = "-" * W
        self.print_fn(f"\n{sep}")
        self.print_fn(f"  DAFoam ROM — Post-Solve Diagnostics")
        self.print_fn(sep)

        # --- Summary ---
        r_rom_norm     = np.linalg.norm(result.rom_residual)
        r_rom_norm_ref = r_rom_norm  # fallback — we don't cache r_norm_ref on the model

        self.print_fn(f"  {'Converged':<32} {result.converged}")
        self.print_fn(f"  {'Reason':<32} {result.reason}")
        self.print_fn(f"  {'Iterations':<32} {result.iterations}")
        self.print_fn(f"  {'‖r_rom‖':<32} {r_rom_norm:.6e}")
        if result.rom_jacobian is not None:
            self.print_fn(f"  {'cond(J_rom)':<32} {np.linalg.cond(result.rom_jacobian):.4e}")

        # --- Per-variable FOM residual breakdown ---
        q     = result.rom_state
        w     = self._reconstruct_fom_state(rom_state=q)
        r_fom = self._eval_fom_residual(fom_state=w)

        r_fom_sq_global = self.comm.allreduce(np.dot(r_fom, r_fom), op=MPI.SUM)
        r_fom_norm      = np.sqrt(r_fom_sq_global)

        # Also evaluate FOM residual at reference (q=0) for ratio context
        w_ref       = self.reference_fom_state
        r_ref       = self._eval_fom_residual(fom_state=w_ref)
        r_ref_sq    = self.comm.allreduce(np.dot(r_ref, r_ref), op=MPI.SUM)
        r_ref_norm  = np.sqrt(r_ref_sq)

        self.print_fn(f"\n  FOM Residual Breakdown at Converged ROM State")
        self.print_fn(f"  {'Variable':<16} {'‖r_ref‖':>14}  {'‖r_fom‖':>14}  {'ratio':>10}")
        self.print_fn(f"  {'-'*16} {'-'*14}  {'-'*14}  {'-'*10}")
        for var, idx in self.state_indices.items():
            r_v     = r_fom[idx]
            rr_v    = r_ref[idx]
            n_v     = np.sqrt(self.comm.allreduce(np.dot(r_v, r_v), op=MPI.SUM))
            nr_v    = np.sqrt(self.comm.allreduce(np.dot(rr_v, rr_v), op=MPI.SUM))
            ratio   = n_v / max(nr_v, 1e-300)
            flag    = "  **" if ratio > 1.0 else ""
            self.print_fn(f"  {var:<16} {nr_v:>14.6e}  {n_v:>14.6e}  {ratio:>10.4e}{flag}")
        self.print_fn(
            f"  {'TOTAL':<16} {r_ref_norm:>14.6e}  {r_fom_norm:>14.6e}"
            f"  {r_fom_norm / max(r_ref_norm, 1e-300):>10.4e}"
        )
        self.print_fn(sep)



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
                 jac_mode:str="fd",
                 solution_leading_integer:int=1,
                 solution_prefix:str|None=None,
                 disable_presolve_diagnostics:bool=False
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
            jac_fd_central,
            solution_leading_integer,
            solution_prefix,
            disable_presolve_diagnostics
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
                 solution_leading_integer:int=1,
                 solution_prefix:str|None=None,
                 disable_presolve_diagnostics:bool=False
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
            jac_fd_central,
            solution_leading_integer,
            solution_prefix,
            disable_presolve_diagnostics
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


# region PHICOMPUTINGLSPGMODEL
class PhiComputingLSPGModel(DAFoamLSPGModel):
    """LSPG ROM that computes face-flux phi from the reconstructed velocity field.

    Identical to DAFoamLSPGModel except that after the standard linear reconstruction
    (w_ref + s * Phi @ q), phi DOFs are overwritten with values derived from the
    reconstructed velocity via setPhiFromU / computePhiFromU.  The POD basis is a
    single combined basis spanning all state variables (including phi columns, which
    are simply ignored at reconstruction time).

    Use this as a stepping-stone between the plain LSPG and the full SplitBasisLSPGModel
    to isolate the effect of phi computation alone.
    """

    def __init__(self,
                 dafoam_input_variables_group,
                 pod_modes: np.ndarray,
                 reference_fom_state: np.ndarray,
                 scaling: np.ndarray,
                 weights: np.ndarray,
                 dafoam_instance,
                 phi_var_name: str = "phi",
                 **kwargs):
        super().__init__(
            dafoam_input_variables_group,
            pod_modes,
            reference_fom_state,
            scaling,
            weights,
            dafoam_instance,
            **kwargs,
        )

        n_local_states = dafoam_instance.getNLocalAdjointStates()
        names, indices = dafoam_instance.getStateVariableMap(includeComponentSuffix=False)
        state_indices  = {name: indices == names.index(name) for name in names}
        self._phi_mask = state_indices.get(phi_var_name, np.zeros(n_local_states, dtype=bool))

    # region _reconstruct_fom_state
    def _reconstruct_fom_state(self, rom_state: np.ndarray) -> np.ndarray:
        w = super()._reconstruct_fom_state(rom_state)
        if self._phi_mask.any():
            w[self._phi_mask] = self._compute_phi_from_states(w)
        return w

    # region _compute_phi_from_states
    def _compute_phi_from_states(self, w: np.ndarray) -> np.ndarray:
        """Return face-flux phi consistent with the velocity in w."""
        self.dafoam_instance.setStates(w)
        self.dafoam_instance.setPhiFromU()
        return self.dafoam_instance.computePhiFromU()


# region PERVARIABLELSPGMODEL
class PerVariableLSPGModel(DAFoamLSPGModel):
    """LSPG ROM with a separate POD basis for every state variable.

    The combined trial basis Phi is block-diagonal: each variable occupies its own
    column block, sized by the number of modes chosen for that variable.  Phi is
    included as one of those blocks (linear reconstruction, no velocity-based phi
    computation).  All state variables are reconstructed linearly.

    Parameters
    ----------
    dafoam_input_variables_group : VariableGroup
    pod_modes_per_var : dict[str, np.ndarray]
        {var_name: (n_local_dofs_var, n_modes_var)} in variable-order DOF layout
        as returned by TrainingDataInterface.load_per_variable_pod_modes.
    reference_fom_state : (n_local_states,) full state vector, cell-interleaved
    scaling  : (n_local_states,) per-DOF scaling, cell-interleaved
    weights  : (n_local_states,) per-DOF inner-product weights, cell-interleaved
    dafoam_instance : PYDAFOAM solver
    **kwargs : forwarded to DAFoamLSPGModel (normalize_residuals, fd_step, etc.)
    """

    def __init__(self,
                 dafoam_input_variables_group,
                 pod_modes_per_var: dict,
                 reference_fom_state: np.ndarray,
                 scaling: np.ndarray,
                 weights: np.ndarray,
                 dafoam_instance,
                 **kwargs):

        n_local_states = dafoam_instance.getNLocalAdjointStates()
        names, indices = dafoam_instance.getStateVariableMap(includeComponentSuffix=False)
        state_indices  = {name: indices == names.index(name) for name in names}

        n_modes_total = sum(m.shape[1] for m in pod_modes_per_var.values())
        Phi           = np.zeros((n_local_states, n_modes_total))

        col_offset = 0
        var_mode_slices = {}
        for var in names:
            modes       = pod_modes_per_var[var]
            n_modes_var = modes.shape[1]
            mask        = state_indices[var]
            Phi[mask, col_offset:col_offset + n_modes_var] = modes
            var_mode_slices[var] = slice(col_offset, col_offset + n_modes_var)
            col_offset += n_modes_var

        super().__init__(
            dafoam_input_variables_group,
            Phi,
            reference_fom_state,
            scaling,
            weights,
            dafoam_instance,
            **kwargs,
        )

        self._var_mode_slices = var_mode_slices


# region SPLITBASISLSPGMODEL
class SplitBasisLSPGModel(DAFoamLSPGModel):
    """LSPG ROM with separate POD bases for flow ([p, U, T]) and turbulence (nuTilda).

    The face-flux variable phi is excluded from both bases. After reconstructing the
    flow and turbulence states, phi must be computed separately by overriding
    _compute_phi_from_states().

    Turbulence is handled in log space: nuTilda is reconstructed as
        w_nu = nu_ref * (exp(log_ref + Phi_nu @ q_nu) - 1)
    where log_ref = log(1 + w_ref_nu / nu_ref).

    Inner product weights follow the fixed-reference Chu energy norm on scaled variables,
    multiplied by cell volumes.

    Parameters
    ----------
    dafoam_input_variables_group : VariableGroup
    pod_modes_flow  : (n_flow_dofs_local, n_modes_flow)  — basis for p, U, T
    pod_modes_turb  : (n_turb_dofs_local, n_modes_turb)  — basis for nuTilda
    reference_fom_state : (n_local_states,) full FOM state vector (distributed)
    norm_config     : NormalizationConfig instance
    cell_volumes    : (n_local_cells,) mesh cell volumes (distributed)
    dafoam_instance : PYDAFOAM solver
    turb_var_name   : variable name for turbulence in getStateVariableMap (default "nuTilda")
    **kwargs        : forwarded to DAFoamLSPGModel (normalize_residuals, fd_step, etc.)
    """

    def __init__(self,
                 dafoam_input_variables_group,
                 pod_modes_flow:  np.ndarray,
                 pod_modes_turb:  np.ndarray,
                 reference_fom_state: np.ndarray,
                 norm_config: NormalizationConfig,
                 cell_volumes: np.ndarray,
                 dafoam_instance,
                 turb_var_name: str = "nuTilda",
                 **kwargs):

        n_local_states = dafoam_instance.getNLocalAdjointStates()
        n_modes_flow   = pod_modes_flow.shape[1]
        n_modes_turb   = pod_modes_turb.shape[1]

        # --- DOF masks ---------------------------------------------------
        names, indices = dafoam_instance.getStateVariableMap(includeComponentSuffix=False)
        state_indices  = {name: indices == names.index(name) for name in names}

        phi_var_name = "phi"
        turb_mask    = state_indices[turb_var_name]
        phi_mask     = state_indices.get(phi_var_name, np.zeros(n_local_states, dtype=bool))
        flow_mask    = np.zeros(n_local_states, dtype=bool)
        for var, mask in state_indices.items():
            if var not in (turb_var_name, phi_var_name):
                flow_mask |= mask

        # --- Block-diagonal combined basis Phi ---------------------------
        # phi rows remain zero — phi is reconstructed separately via _compute_phi_from_states
        n_modes_total = n_modes_flow + n_modes_turb
        Phi           = np.zeros((n_local_states, n_modes_total))
        Phi[flow_mask, :n_modes_flow] = pod_modes_flow
        Phi[turb_mask, n_modes_flow:] = pod_modes_turb

        # --- Scaling and weight vectors ----------------------------------
        scaling          = norm_config.make_scaling_vector(state_indices)
        chu_weights      = norm_config.make_chu_weight_vector(state_indices)
        cell_vol_per_dof = expand_cell_volumes(cell_volumes, state_indices)
        weights          = cell_vol_per_dof * chu_weights

        # --- Initialise parent -------------------------------------------
        super().__init__(
            dafoam_input_variables_group,
            Phi,
            reference_fom_state,
            scaling,
            weights,
            dafoam_instance,
            **kwargs,
        )

        # --- Private attributes ------------------------------------------
        self._flow_mask      = flow_mask
        self._turb_mask      = turb_mask
        self._phi_mask       = phi_mask
        self._n_modes_flow   = n_modes_flow
        self._n_modes_turb   = n_modes_turb
        self._pod_modes_flow = pod_modes_flow
        self._pod_modes_turb = pod_modes_turb
        self._norm_config    = norm_config
        # Precompute log-space reference for turbulence reconstruction.
        # NOTE: if reference_fom_state is a csdl Variable, its .value will be extracted
        # by the parent's evaluate_input_output(); update _log_ref_nu there if needed.
        self._log_ref_nu     = norm_config.apply_log_scaling_sa(reference_fom_state[turb_mask])

        # Store split state_indices for diagnostics
        self._split_state_indices = state_indices

    # region _reconstruct_fom_state
    def _reconstruct_fom_state(self, rom_state: np.ndarray) -> np.ndarray:
        # 1. Linear reconstruction for all DOFs via parent formula: w_ref + s * Phi @ q.
        #    phi rows are zero in Phi, so phi DOFs initialise to w_ref[phi_mask].
        w = super()._reconstruct_fom_state(rom_state)

        # 2. Override turbulence DOFs with log-space reconstruction.
        q_nu = rom_state[self._n_modes_flow:]
        w[self._turb_mask] = self._norm_config.invert_log_scaling_sa(
            self._log_ref_nu + self._pod_modes_turb @ q_nu
        )

        # 3. Compute face-flux phi from the reconstructed flow state and overwrite.
        if self._phi_mask.any():
            w[self._phi_mask] = self._compute_phi_from_states(w)

        return w

    # region _compute_phi_from_states
    def _compute_phi_from_states(self, w: np.ndarray) -> np.ndarray:
        """Return face-flux phi consistent with the velocity in w.

        setPhiFromU() is called here (once, during reconstruction) but NOT in
        _set_fom_states, because w already carries the correct phi after reconstruction.
        Calling setPhiFromU() again during residual evaluation would overwrite phi
        with a potentially inconsistent value (e.g. volume flux U·A instead of mass
        flux ρU·A for a compressible solver), corrupting the energy residual.
        """
        self.dafoam_instance.setStates(w)
        self.dafoam_instance.setPhiFromU()
        return self.dafoam_instance.computePhiFromU()

    # region pre_solve_diagnostics
    def pre_solve_diagnostics(self, initial_rom_state: np.ndarray) -> None:
        # Parent prints combined-basis diagnostics (ortho check + variable summary + energy table)
        super().pre_solve_diagnostics(initial_rom_state)

        if self.disable_presolve_diagnostics:
            return

        W   = 72
        sep = "-" * W
        self.print_fn(f"\n{sep}")
        self.print_fn(f"  SplitBasisLSPGModel — Per-Basis Diagnostics")
        self.print_fn(sep)

        m = self.weights

        # --- Flow basis orthogonality: Phi_f^T W_f Phi_f = I ---
        Phi_f = self._pod_modes_flow
        m_f   = m[self._flow_mask]
        nf    = Phi_f.shape[1]
        Gf_local    = Phi_f.T @ (m_f[:, None] * Phi_f)
        Gf          = self.comm.allreduce(Gf_local, op=MPI.SUM)
        err_f       = np.linalg.norm(Gf - np.eye(nf), "fro")
        ok_f        = err_f < 1e-10
        self.print_fn(
            f"  {'Flow basis ortho ‖Φf^T Wf Φf - I‖_F':<40} "
            + (f"PASS ({err_f:.2e})" if ok_f else f"WARN ({err_f:.2e})")
        )

        # --- Turb basis orthogonality: Phi_nu^T W_nu Phi_nu = I ---
        Phi_nu = self._pod_modes_turb
        m_nu   = m[self._turb_mask]
        nnu    = Phi_nu.shape[1]
        Gnu_local   = Phi_nu.T @ (m_nu[:, None] * Phi_nu)
        Gnu         = self.comm.allreduce(Gnu_local, op=MPI.SUM)
        err_nu      = np.linalg.norm(Gnu - np.eye(nnu), "fro")
        ok_nu       = err_nu < 1e-10
        self.print_fn(
            f"  {'Turb  basis ortho ‖Φν^T Wν Φν - I‖_F':<40} "
            + (f"PASS ({err_nu:.2e})" if ok_nu else f"WARN ({err_nu:.2e})")
        )

        # --- Flow mode energy fractions per sub-variable ---
        flow_var_names = [v for v in self._split_state_indices
                          if v not in ("nuTilda", "phi")]
        # Map from full-state indices to flow-DOF local indices
        flow_idx_map   = {v: self._split_state_indices[v][self._flow_mask]
                          for v in flow_var_names}

        self.print_fn(f"\n  Flow basis — mode M-weighted energy fraction per variable")
        header = f"  {'Mode':>6}" + "".join(f"  {v:>10}" for v in flow_var_names) + f"  {'‖φ‖²_M':>10}"
        self.print_fn(header)
        self.print_fn("  " + "-" * (len(header) - 2))
        for k in range(nf):
            phi_k = Phi_f[:, k]
            total = self.comm.allreduce(np.sum(m_f * phi_k**2), op=MPI.SUM)
            fracs = []
            for v in flow_var_names:
                idx_v  = flow_idx_map[v]
                v_tot  = self.comm.allreduce(np.sum(m_f[idx_v] * phi_k[idx_v]**2), op=MPI.SUM)
                fracs.append(v_tot / max(total, 1e-300))
            row = f"  {k:>6d}" + "".join(f"  {f:>10.4f}" for f in fracs) + f"  {total:>10.4f}"
            self.print_fn(row)

        # --- Turb mode energies ---
        self.print_fn(f"\n  Turb  basis — mode M-weighted energy")
        self.print_fn(f"  {'Mode':>6}  {'‖φ‖²_M':>10}")
        self.print_fn(f"  {'------':>6}  {'----------':>10}")
        for k in range(nnu):
            phi_k = Phi_nu[:, k]
            total = self.comm.allreduce(np.sum(m_nu * phi_k**2), op=MPI.SUM)
            self.print_fn(f"  {k:>6d}  {total:>10.4f}")

        self.print_fn(sep)

    # region from_norm_config
    @classmethod
    def from_norm_config(cls,
                         dafoam_input_variables_group,
                         pod_modes_flow:  np.ndarray,
                         pod_modes_turb:  np.ndarray,
                         reference_fom_state: np.ndarray,
                         norm_config: NormalizationConfig,
                         cell_volumes: np.ndarray,
                         dafoam_instance,
                         **kwargs) -> "SplitBasisLSPGModel":
        """Convenience constructor — delegates directly to __init__.

        Provided as a named entry point for clarity in offline basis-construction scripts
        that import NormalizationConfig without a live DAFoam instance for scaling setup.
        """
        return cls(
            dafoam_input_variables_group,
            pod_modes_flow,
            pod_modes_turb,
            reference_fom_state,
            norm_config,
            cell_volumes,
            dafoam_instance,
            **kwargs,
        )