from abc import ABC, abstractmethod
import numpy as np
from csdl_dafoam.core.rom.rom_models import BaseModel





from dataclasses import dataclass
# Creating an "immutable" dataclass to contain the
# output of the solver. This makes sure that the solution
# is referenced at a particular state.
# region SOLVERRESULT
@dataclass(frozen=True)
class SolverResult():     
    rom_state:          np.ndarray
    rom_residual:       np.ndarray
    converged:          bool
    reason:             list[str]
    rom_jacobian:       np.ndarray=None
    fom_state:          np.ndarray=None
    iterations:         int=None
    history:            list=None




# region BASESOLVER
class BaseSolver(ABC):
    def __init__(self, model:BaseModel=None, options:dict={}):
        self.model = model
        self.opts  = options
        pass
    
    @abstractmethod
    def solve(self, initial_state):
        pass

    @abstractmethod
    def adjoint_solve(self, result, rhs, mode):
        # Will need to solve J^T x = rhs for x
        # result variable should contain the latest J (if a direct solve is feasible)
        pass





# region NEWTONSOLVER
class NewtonSolver(BaseSolver):
    # region solve
    def solve(self, initial_state:np.ndarray, datum_state:np.ndarray=None):
        # Get solver options
        opts    = self.opts
        alpha   = opts.get("ls_alpha0", 1.0)
        maxiter = opts.get("maxiter", 30)
        tol_abs = opts.get("tol_abs", 1e-12)
        tol_rel = opts.get("tol_rel", 1e-8)
        tol_step_rel = opts.get("tol_step_rel", 1e-10)
        tol_step_abs = opts.get("tol_step_abs", 1e-14)

        # Reference value for residual
        q_ref       = datum_state if datum_state is not None else np.zeros_like(initial_state)
        r_ref       = self.model.evaluate_residuals(rom_state=q_ref)
        r_norm_ref  = np.linalg.norm(r_ref)

        # Initial state
        q       = initial_state.copy()
        dq_prev = np.full_like(q, np.inf)
        r       = self.model.evaluate_residuals(rom_state=q)
        r_norm  = np.linalg.norm(r)
        J       = None

        # Extra flags and return info initialization
        history     = []
        converged   = False
        reason      = []
        ls_success  = True
        result      = None

        self._print_header(residual_norm=r_norm, reference_residual_norm=r_norm_ref, initial_state=q, print_fn=self.model.print_fn)

        if r_norm == 0.0:
            return SolverResult(rom_state=q, 
                                rom_residual=r, 
                                converged=True, 
                                reason=["initial residual zero"], 
                                rom_jacobian=J,
                                iterations=0,
                                history=history)
                

        # Main Newton loop
        for k in range(maxiter):
            
            # Convergence check
            rel_res  = r_norm / r_norm_ref
            step_abs = np.linalg.norm(dq_prev)
            step_rel = step_abs / max(np.linalg.norm(q), 1e-14)

            if r_norm < tol_abs and ls_success:
                converged = True
                reason.append(f"absolute residual tolerance {r_norm} < {tol_abs}")
            
            if rel_res < tol_rel and ls_success:
                converged = True
                reason.append(f"relative residual tolerance {rel_res} < {tol_rel}")
            
            if step_rel < tol_step_rel and ls_success:
                converged = True
                reason.append(f"relative steps size tolerance {step_rel} < {tol_step_rel}")

            if step_abs < tol_step_abs and ls_success:
                converged = True
                reason.append(f"absolute steps size tolerance {step_abs} < {tol_step_abs}")

            if converged:
                J = self.model.compute_reduced_jacobian(rom_state=q)
                if J is None:
                    J = self._compute_fd_jacobian(state=q, residual=r)
                # Bring residual up to date (this is for cases where the residual might depend on the Jacobian - e.g., LSPG)
                with self.model.freeze_jacobian():
                    r = self.model.evaluate_residuals(rom_state=q)
                    
                result =  SolverResult(rom_state=q, 
                                    rom_residual=r, 
                                    converged=True, 
                                    reason=reason, 
                                    rom_jacobian=J,
                                    iterations=k,
                                    history=history)

                break
            
            # Jacobian computation with FD fallback if Model doesn't support direct Jacobian return
            J = self.model.compute_reduced_jacobian(rom_state=q)  
            if J is None:
                J = self._compute_fd_jacobian(state=q, residual=r)

            # Bring residual up to date (this is for cases where the residual might depend on the Jacobian - e.g., LSPG)
            with self.model.freeze_jacobian():
                r = self.model.evaluate_residuals(rom_state=q)

            # Step and linesearch
            try:
                dq = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dq, *_ = np.linalg.lstsq(J, -r, rcond=None)


            # Conduct the line search with the frozen Jacobian (again, mainly for LSPG)
            with self.model.freeze_jacobian():
                alpha, q_trial, r_trial, ls_success = self._line_search(q, dq, r_norm)

            q = q_trial
            r = r_trial
            dq_prev = alpha * dq
            r_norm  = np.linalg.norm(r)

            self._print_iter(iter=k+1, residual_norm=r_norm, reference_residual_norm=r_norm_ref, alpha=alpha, state=q, state_step=dq_prev, print_fn=self.model.print_fn)

            history.append(
                {
                    "residual_norm": np.linalg.norm(r),
                    "alpha": alpha,
                    "line_search_success": ls_success
               }
            )
        

        if result is None:
            reason = ["maxiter"]
            result = SolverResult(rom_state=q, 
                                rom_residual=r, 
                                converged=False, 
                                reason=reason, 
                                rom_jacobian=J,
                                iterations=maxiter,
                                history=history)
            
        self._print_footer(result=result, print_fn=self.model.print_fn)

        return result
    

    # region adjoint_solve
    def adjoint_solve(self, result:SolverResult, rhs:np.ndarray, mode:str):
        J = result.rom_jacobian
        # If we already have the reduced Jacobian in the result, then we can just
        # solve the linear system
        if J is not None:
            return np.linalg.solve(J.T, rhs)
        
        # Otherwise, we'll have to get the model compute for us
        else:
            raise NotImplementedError


    # region _line_search
    def _line_search(self, state:np.ndarray, step:np.ndarray, residual_norm:float):
        q       = state
        dq      = step
        r_norm0 = residual_norm
        opts = self.opts
        alpha = opts.get("ls_alpha0", 1.0)
        rho   = opts.get("ls_rho", 0.5)
        c1    = opts.get("ls_c1", 1e-4)
        max_ls = opts.get("ls_maxiter", 10)

        for _ in range(max_ls):
            q_trial = q + alpha * dq
            r_trial = self.model.evaluate_residuals(rom_state=q_trial)
            r_trial_norm = np.linalg.norm(r_trial)

            if r_trial_norm <= (1.0 - c1 * alpha) * r_norm0:
                return alpha, q_trial, r_trial, True
            
            alpha *= rho

        return alpha, q_trial, r_trial, False
    

    # region _compute_fd_jacobian
    def _compute_fd_jacobian(self, state:np.ndarray, residual:np.ndarray=None):
        step    = self.opts.get("jac_fd_step",    1e-6)
        central = self.opts.get("jac_fd_central", True)
        q  = state
        r0 = residual
        
        r0 = self.model.evaluate_residuals(state) if r0 is None else r0
        m = len(r0)
        n = len(q)
        J = np.zeros((m, n))

        for j in range(n):
            h = step * max(1, np.abs(q[j])) # Scale the stepsize with q when q is large
            dq = np.zeros_like(q)
            dq[j] = h

            if central:
                rp = self.model.evaluate_residuals(q + dq)
                rm = self.model.evaluate_residuals(q - dq)
                J[:, j] = (rp - rm) / (2.0 * h)
            else:
                rp = self.model.evaluate_residuals(q + dq)
                J[:, j] = (rp - r0) / h

        return J


    # region print_header
    def _print_header(self, residual_norm, reference_residual_norm, initial_state, print_fn=None):
        print_fn = print() if print_fn is None else print_fn
        separator_width = 60
        separator_str   = "_" * separator_width
        print_fn(separator_str)
        print_fn("NEWTON SOLVER")

        # Column headers
        print_fn(
            f"  {'Iter':>4}  {'‖r_rom‖':>16}  {'‖r_rom‖/‖r_ref‖':>16}"
            f"  {'alpha':>16}  {'‖dq‖':>16}  {'‖q‖':>16}"
           
        )
        print_fn(
            f"  {'-' *4}  {'-'*16}  {'-'*16}"
            f"  {'-'*16}  {'-'*16}  {'-'*16}"
        )
        print_fn(
            f"  {0:>4}  {(residual_norm):>16.6e}  {residual_norm / reference_residual_norm:>16.6e}"
            f"  {'-':>16}  {'-':>16}  {np.linalg.norm(initial_state):>16.6e}"
        )

    
    # region print_iter
    def _print_iter(self, iter, residual_norm, reference_residual_norm, alpha, state, state_step, print_fn=None):
        print_fn = print() if print_fn is None else print_fn

        print_fn(
            f"  {iter:>4}  {(residual_norm):>16.6e}  {residual_norm / reference_residual_norm:>16.6e}"
            f"  {alpha:>16.6e}  {np.linalg.norm(state_step):>16.6e}  {np.linalg.norm(state):>16.6e}"
        )

    
    # region print_footer
    def _print_footer(self, result:SolverResult, print_fn=None):
        print_fn = print() if print_fn is None else print_fn
        separator_width = 60
        separator_str   = "_" * separator_width
        
        print_fn(f"Converged: {result.converged}")
        print_fn(f"Reason:    {result.reason}")
        
        print_fn(separator_str)

        
            
            
            
            

            
            
