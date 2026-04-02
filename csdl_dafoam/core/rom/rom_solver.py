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
    reason:             str
    rom_jacobian:       np.ndarray=None
    fom_state:          np.ndarray=None
    iterations:         int=None
    history:            list=None




# region BASESOLVER
class BaseSolver(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def solve(self, initial_state):
        pass

    @abstractmethod
    def adjoint_solve(self, result, rhs, mode):
        pass




class NewtonSolver(BaseSolver):
    def __init__(self, model:BaseModel, options:dict):
        self.model = model
        self.opts  = options

    def solver(self, initial_state):
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

        # Extra return info initialization
        history     = []
        converged   = False
        reason      = []

        if r_norm0 == 0.0:
            return q, {"converged": True, "reason": "initial residual zero"}   ################################CHANGE THIS TO SOLVERRESULT

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
            
            # Jacobian computation with FD fallback if Model doesn't support it
            J = self.model.compute_reduced_jacobian(rom_state=q)  
            if J is None:
                J = self._compute_fd_jacobian(rom_state=q, rom_residual=r)

            # Step and linesearch
            try:
                dq = np.linalg.solve(J, -r)
            except:
                dq, *_ = np.linalg.lstsq(J, -r, rcond=None)

            
            
            
            

            
            
