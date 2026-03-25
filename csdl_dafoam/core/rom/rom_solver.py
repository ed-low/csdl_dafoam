import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple


# ============================================================
# region BASE CLASS
# ============================================================
class ROMSolverBase(ABC):
    """
    Abstract base class for ROM solvers.

    Subclasses implement `solve`, which finds the reduced state q such that
    the ROM residual r_rom(q) = 0.

    The solver operates entirely in reduced space (ROM state q is a dense
    vector of size n_modes). All FOM-level operations (residual evaluation,
    Jacobian-vector products) are accessed through callables passed to solve,
    keeping this class free of DAFoam or MPI dependencies.

    Parameters
    ----------
    print_fn : callable, optional
        Function used for printing solver progress. Defaults to Python's
        built-in print. Useful for suppressing output on non-root MPI ranks.
    """

    def __init__(self, print_fn: Callable = print):
        self.print_fn = print_fn

    @abstractmethod
    def solve(
        self,
        residual_fn:  Callable[[np.ndarray], np.ndarray],
        jacobian_fn:  Callable[[np.ndarray], np.ndarray],
        q0:           np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        Solve the ROM system r_rom(q) = 0.

        Parameters
        ----------
        residual_fn : callable
            Maps ROM state q (n_modes,) -> ROM residual r_rom (n_modes,).
        jacobian_fn : callable
            Maps ROM state q (n_modes,) -> ROM Jacobian J_rom (n_modes, n_modes).
        q0 : np.ndarray, shape (n_modes,)
            Initial guess for the ROM state.

        Returns
        -------
        q : np.ndarray, shape (n_modes,)
            Converged (or best available) ROM state.
        converged : bool
            Whether the solver converged within tolerance.
        """
        pass


# ============================================================
# region NEWTON SOLVER
# ============================================================
class NewtonSolver(ROMSolverBase):
    """
    Newton solver with Armijo backtracking line search for ROM systems.

    Solves r_rom(q) = 0 using Newton's method:

        J_rom(q_k) dq_k = -r_rom(q_k)
        q_{k+1} = q_k + alpha_k * dq_k

    where alpha_k is determined by an Armijo backtracking line search.

    Parameters
    ----------
    max_iter : int
        Maximum number of Newton iterations.
    tol_rel : float
        Relative residual tolerance (convergence if ||r||/||r0|| < tol_rel).
    tol_abs : float
        Absolute residual tolerance (convergence if ||r|| < tol_abs).
    ls_alpha0 : float
        Initial line search step length.
    ls_rho : float
        Backtracking factor (step is multiplied by this each backtrack).
    ls_c1 : float
        Armijo sufficient decrease constant.
    ls_max_iter : int
        Maximum number of line search iterations.
    print_fn : callable, optional
        Function used for printing solver progress.

    Examples
    --------
    Solving a simple quadratic ROM system for testing:

        def residual_fn(q):
            return q**2 - 4.0        # root at q = [2., 2., ...]

        def jacobian_fn(q):
            return np.diag(2.0 * q)  # diagonal Jacobian

        solver = NewtonSolver(tol_rel=1e-10)
        q, converged = solver.solve(residual_fn, jacobian_fn, q0=np.ones(2))
        # q ≈ [2., 2.], converged = True
    """

    DEFAULT_OPTIONS = {
        "max_iter":   50,
        "tol_rel":    1e-6,
        "tol_abs":    1e-10,
        "ls_alpha0":  1.0,
        "ls_rho":     0.5,
        "ls_c1":      1e-4,
        "ls_max_iter": 10,
    }

    def __init__(self, print_fn: Callable = print, **kwargs):
        super().__init__(print_fn=print_fn)
        self.opts = {**self.DEFAULT_OPTIONS, **kwargs}

    def solve(
        self,
        residual_fn:  Callable[[np.ndarray], np.ndarray],
        jacobian_fn:  Callable[[np.ndarray], np.ndarray],
        q0:           np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        Run the Newton solve. See ROMSolverBase.solve for full signature.
        """
        opts      = self.opts
        q         = q0.copy()
        converged = False

        r        = residual_fn(q)
        r_norm0  = np.linalg.norm(r)
        r_norm   = r_norm0

        self._print_header()
        self._print_iter(0, r_norm, 1.0, "-")

        for k in range(opts["max_iter"]):

            # --- convergence check ---
            if r_norm < opts["tol_abs"]:
                self.print_fn(f"Newton converged (absolute tolerance) at iteration {k}.")
                converged = True
                break
            if r_norm0 > 0 and r_norm / r_norm0 < opts["tol_rel"]:
                self.print_fn(f"Newton converged (relative tolerance) at iteration {k}.")
                converged = True
                break

            # --- compute Newton step ---
            J   = jacobian_fn(q)
            dq  = self._solve_linear(J, -r)

            # --- Armijo backtracking line search ---
            alpha, q, r, r_norm, ls_ok = self._line_search(
                q, dq, r_norm, residual_fn, opts
            )

            self._print_iter(k + 1, r_norm, r_norm / r_norm0 if r_norm0 > 0 else 0., alpha)

            if not ls_ok:
                self.print_fn("Warning: line search failed. Taking minimum step.")

        else:
            self.print_fn(
                f"Warning: Newton solver reached max iterations ({opts['max_iter']}) "
                "without converging."
            )

        return q, converged

    # ----------------------------------------------------------
    # region helpers
    # ----------------------------------------------------------
    @staticmethod
    def _solve_linear(J: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Solve J dq = rhs, falling back to least-squares if J is singular."""
        try:
            return np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            solution, _, _, _ = np.linalg.lstsq(J, rhs, rcond=None)
            return solution

    @staticmethod
    def _line_search(
        q:           np.ndarray,
        dq:          np.ndarray,
        r_norm:      float,
        residual_fn: Callable,
        opts:        dict,
    ) -> Tuple[float, np.ndarray, np.ndarray, float, bool]:
        """
        Armijo backtracking line search.

        Returns
        -------
        alpha : float
            Accepted step length.
        q_new : np.ndarray
            Updated state.
        r_new : np.ndarray
            Residual at updated state.
        r_norm_new : float
            Norm of residual at updated state.
        success : bool
            Whether sufficient decrease was achieved.
        """
        alpha   = opts["ls_alpha0"]
        c1      = opts["ls_c1"]
        rho     = opts["ls_rho"]
        success = False

        for _ in range(opts["ls_max_iter"]):
            q_trial      = q + alpha * dq
            r_trial      = residual_fn(q_trial)
            r_norm_trial = np.linalg.norm(r_trial)

            if r_norm_trial <= (1.0 - c1 * alpha) * r_norm:
                success = True
                return alpha, q_trial, r_trial, r_norm_trial, success

            alpha *= rho

        # Accept minimum step even without sufficient decrease
        return alpha, q_trial, r_trial, r_norm_trial, success

    def _print_header(self):
        self.print_fn(
            f"{'Iter':>5} {'||r_rom||':>14} {'||r||/||r0||':>14} {'alpha':>8}"
        )

    def _print_iter(self, k, r_norm, r_rel, alpha):
        alpha_str = f"{alpha:>8.2e}" if isinstance(alpha, float) else f"{alpha:>8}"
        self.print_fn(f"{k:>5} {r_norm:>14.6e} {r_rel:>14.6e} {alpha_str}")


# ============================================================
# region FIXED POINT SOLVER
# ============================================================
class FixedPointSolver(ROMSolverBase):
    """
    Fixed-point (Richardson) iteration for ROM systems.

    Iterates:
        q_{k+1} = q_k - omega * r_rom(q_k)

    This is cheaper per iteration than Newton (no Jacobian required) but
    converges more slowly and only works well when the ROM residual is
    well-conditioned. Primarily useful as a cheap alternative for Galerkin
    ROMs or as a sanity-check solver.

    Note: `jacobian_fn` is accepted in the signature for interface
    compatibility but is not used.

    Parameters
    ----------
    omega : float
        Relaxation factor. Smaller values improve stability at the cost
        of slower convergence.
    max_iter : int
        Maximum number of iterations.
    tol_rel : float
        Relative residual tolerance.
    tol_abs : float
        Absolute residual tolerance.
    print_fn : callable, optional
        Function used for printing solver progress.
    """

    DEFAULT_OPTIONS = {
        "omega":    1.0,
        "max_iter": 200,
        "tol_rel":  1e-6,
        "tol_abs":  1e-10,
    }

    def __init__(self, print_fn: Callable = print, **kwargs):
        super().__init__(print_fn=print_fn)
        self.opts = {**self.DEFAULT_OPTIONS, **kwargs}

    def solve(
        self,
        residual_fn:  Callable[[np.ndarray], np.ndarray],
        jacobian_fn:  Optional[Callable[[np.ndarray], np.ndarray]],
        q0:           np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        Run the fixed-point iteration. See ROMSolverBase.solve for full signature.
        """
        opts      = self.opts
        q         = q0.copy()
        converged = False

        r       = residual_fn(q)
        r_norm0 = np.linalg.norm(r)
        r_norm  = r_norm0

        self.print_fn(f"{'Iter':>5} {'||r_rom||':>14} {'||r||/||r0||':>14}")
        self.print_fn(f"{0:>5} {r_norm0:>14.6e} {1.0:>14.6e}")

        for k in range(opts["max_iter"]):
            if r_norm < opts["tol_abs"]:
                self.print_fn(f"Fixed-point converged (absolute tolerance) at iteration {k}.")
                converged = True
                break
            if r_norm0 > 0 and r_norm / r_norm0 < opts["tol_rel"]:
                self.print_fn(f"Fixed-point converged (relative tolerance) at iteration {k}.")
                converged = True
                break

            q      = q - opts["omega"] * r
            r      = residual_fn(q)
            r_norm = np.linalg.norm(r)

            self.print_fn(
                f"{k+1:>5} {r_norm:>14.6e} "
                f"{r_norm / r_norm0 if r_norm0 > 0 else 0.:>14.6e}"
            )

        else:
            self.print_fn(
                f"Warning: fixed-point solver reached max iterations ({opts['max_iter']}) "
                "without converging."
            )

        return q, converged
