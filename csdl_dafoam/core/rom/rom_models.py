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