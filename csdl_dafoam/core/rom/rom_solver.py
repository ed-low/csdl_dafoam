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
    def __init__(self, model:BaseModel=None, options:dict=None):
        self.model = model
        self.opts  = options
        pass
    
    @abstractmethod
    def solve(self, initial_state):
        pass

    @abstractmethod
    def adjoint_solve(self, result, rhs, mode):
        pass





# region NEWTONSOLVER
class NewtonSolver(BaseSolver):
    def __init__(self, model:BaseModel, options:dict):
        self.model = model
        self.opts  = options


    # region solve
    def solve(self, initial_state:np.ndarray):
        # Get solver options
        opts    = self.opts
        alpha   = opts.get("ls_alpha0", 1.0)
        maxiter = opts.get("maxiter", 30)
        tol_abs = opts.get("tol_abs", 1e-12)
        tol_rel = opts.get("tol_rel", 1e-8)
        tol_step_rel = opts.get("tol_step_rel", 1e-10)

        # Initial value setup
        q0      = initial_state
        q       = initial_state.copy()
        dq_prev = np.zeros_like(q)
        r       = self.model.evaluate_residuals(rom_state=q)
        r_norm0 = np.linalg.norm(r)
        J       = None

        # Extra flags and return info initialization
        history     = []
        converged   = False
        reason      = []
        ls_success  = True

        if r_norm0 == 0.0:
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
            r_norm   = np.linalg.norm(r)
            rel_res  = r_norm / r_norm0
            step_rel = np.linalg.norm(dq_prev) * alpha / max(np.linalg.norm(q), 1e-14)

            if r_norm < tol_abs and ls_success:
                converged = True
                reason.append(f"absolute residual tolerance {r_norm} < {tol_abs}")
            
            if rel_res < tol_rel and ls_success:
                converged = True
                reason.append(f"relative residual tolerance {rel_res} < {tol_rel}")
            
            if step_rel < tol_step_rel and ls_success:
                converged = True
                reason.append(f"relative steps size tolerance {step_rel} < {tol_step_rel}")

            if converged:
                return SolverResult(rom_state=q, 
                                    rom_residual=r, 
                                    converged=True, 
                                    reason=reason, 
                                    rom_jacobian=J,
                                    iterations=k,
                                    history=history)
            
            # Jacobian computation with FD fallback if Model doesn't support direct Jacobian return
            J = self.model.compute_reduced_jacobian(rom_state=q)  
            if J is None:
                J = self._compute_fd_jacobian(state=q, residual=r)

            # Step and linesearch
            try:
                dq = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dq, *_ = np.linalg.lstsq(J, -r, rcond=None)

            alpha, q_trial, r_trial, ls_success = self._line_search(q, dq, r_norm)

            q = q_trial
            r = r_trial
            dq_prev = alpha * dq

            history.append(
                {
                    "residual_norm": np.linalg.norm(r),
                    "alpha": alpha,
                    "line_search_success": ls_success
               }
            )

        return SolverResult(rom_state=q, 
                            rom_residual=r, 
                            converged=False, 
                            reason=["maxiter"], 
                            rom_jacobian=J,
                            iterations=maxiter,
                            history=history)
    


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
    def _line_search(self, state:np.ndarray, step:float, residual_norm:float):
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

            
            
            
            

            
            
