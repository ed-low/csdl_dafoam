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
    def solve(self, initial_state)->SolverResult:
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

        # Initial state — evaluated first so the model (e.g. LSPG) can set up its test basis
        q       = initial_state.copy()
        dq_prev = np.full_like(q, np.inf)
        r       = self.model.evaluate_residuals(rom_state=q)
        r_norm  = np.linalg.norm(r)
        J       = None

        # Reference residual — frozen so LSPG reuses the test basis set above
        q_ref = datum_state if datum_state is not None else np.zeros_like(initial_state)
        with self.model.freeze_jacobian():
            r_ref      = self.model.evaluate_residuals(rom_state=q_ref)
        r_norm_ref = np.linalg.norm(r_ref)

        # Extra flags and return info initialization
        history    = []
        converged  = False
        reason     = []
        ls_success = False
        result     = None

        # Query extra iteration column specs once — used in header and every iter print
        extra_headers = self.model.iter_diagnostic_headers()

        # Pre-solve diagnostics (basis quality, scaling, etc.)
        self.model.pre_solve_diagnostics(initial_rom_state=q)

        self._print_header(
            residual_norm=r_norm,
            reference_residual_norm=r_norm_ref,
            initial_state=q,
            extra_headers=extra_headers,
            print_fn=self.model.print_fn,
        )

        if r_norm == 0.0:
            result = SolverResult(rom_state=q,
                                  rom_residual=r,
                                  converged=True,
                                  reason=["initial residual zero"],
                                  rom_jacobian=J,
                                  iterations=0,
                                  history=history)
            self._print_footer(result=result, print_fn=self.model.print_fn)
            self.model.post_solve_diagnostics(result)
            return result


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
                reason.append(f"relative step size tolerance {step_rel} < {tol_step_rel}")

            if step_abs < tol_step_abs and ls_success:
                converged = True
                reason.append(f"absolute step size tolerance {step_abs} < {tol_step_abs}")

            if converged:
                J = self.model.compute_reduced_jacobian(rom_state=q)
                if J is None:
                    J = self._compute_fd_jacobian(state=q, residual=r)
                # Bring residual up to date (residual depends on Jacobian for LSPG)
                with self.model.freeze_jacobian():
                    r = self.model.evaluate_residuals(rom_state=q)

                result = SolverResult(rom_state=q,
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

            # Bring residual up to date (residual depends on Jacobian for LSPG)
            with self.model.freeze_jacobian():
                r = self.model.evaluate_residuals(rom_state=q)
            r_norm = np.linalg.norm(r)

            # Linear solve
            try:
                dq = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dq, *_ = np.linalg.lstsq(J, -r, rcond=None)

            # Line search with the frozen Jacobian (mainly for LSPG)
            with self.model.freeze_jacobian():
                alpha, q_trial, r_trial, ls_success = self._line_search(q, dq, r_norm)

            q       = q_trial
            r       = r_trial
            dq_prev = alpha * dq
            r_norm  = np.linalg.norm(r)

            extra_vals = self.model.iter_diagnostics(rom_state=q, jacobian=J)
            self._print_iter(
                iter=k+1,
                residual_norm=r_norm,
                reference_residual_norm=r_norm_ref,
                alpha=alpha,
                state=q,
                state_step=dq_prev,
                ls_success=ls_success,
                extra_headers=extra_headers,
                extra_vals=extra_vals,
                print_fn=self.model.print_fn,
            )

            history.append({
                "residual_norm":        r_norm,
                "alpha":                alpha,
                "line_search_success":  ls_success,
            })


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
        self.model.post_solve_diagnostics(result)

        return result


    # region adjoint_solve
    def adjoint_solve(self, result:SolverResult, rhs:np.ndarray, mode:str):
        J = result.rom_jacobian
        if J is not None:
            return np.linalg.solve(J.T, rhs)
        else:
            raise NotImplementedError


    # region _line_search
    def _line_search(self, state:np.ndarray, step:np.ndarray, residual_norm:float):
        q       = state
        dq      = step
        r_norm0 = residual_norm
        opts    = self.opts
        alpha   = opts.get("ls_alpha0", 1.0)
        rho     = opts.get("ls_rho", 0.5)
        c1      = opts.get("ls_c1", 1e-4)
        max_ls  = opts.get("ls_maxiter", 10)

        for _ in range(max_ls):
            q_trial      = q + alpha * dq
            r_trial      = self.model.evaluate_residuals(rom_state=q_trial)
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
            h  = step * max(1, np.abs(q[j]))
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


    # region _format_extra_cols
    @staticmethod
    def _format_extra_cols(values:list, headers:list) -> list[str]:
        strs = []
        for v, (_, w) in zip(values, headers):
            try:
                f = float(v)
                strs.append(f"{f:>{w}.4e}" if not np.isnan(f) else f"{'nan':>{w}}")
            except (TypeError, ValueError):
                strs.append(f"{str(v):>{w}}")
        return strs


    # region _print_header
    def _print_header(self, residual_norm, reference_residual_norm, initial_state,
                      extra_headers=None, print_fn=None, solver_name="NEWTON SOLVER"):
        extra_headers = extra_headers or []
        print_fn      = print if print_fn is None else print_fn

        extra_h   = "".join(f"  {h:>{w}}"  for h, w in extra_headers)
        extra_sep = "".join(f"  {'-'*w}"   for _, w in extra_headers)
        extra_0   = "".join(f"  {'-':>{w}}" for _, w in extra_headers)

        sep_width = 98 + sum(w + 2 for _, w in extra_headers)
        print_fn("_" * sep_width)
        print_fn(solver_name)
        print_fn(
            f"  {'Iter':>4}  {'‖r_rom‖':>16}  {'‖r_rom‖/‖r_ref‖':>16}"
            f"  {'alpha':>16}  {'‖dq‖':>16}  {'‖q‖':>16}" + extra_h
        )
        print_fn(
            f"  {'-'*4}  {'-'*16}  {'-'*16}"
            f"  {'-'*16}  {'-'*16}  {'-'*16}" + extra_sep
        )
        print_fn(
            f"  {0:>4}  {residual_norm:>16.6e}  {residual_norm / reference_residual_norm:>16.6e}"
            f"  {'-':>16}  {'-':>16}  {np.linalg.norm(initial_state):>16.6e}" + extra_0
        )


    # region _print_iter
    def _print_iter(self, iter, residual_norm, reference_residual_norm, alpha, state, state_step,
                    ls_success=True, extra_headers=None, extra_vals=None, print_fn=None):
        extra_headers = extra_headers or []
        extra_vals    = extra_vals    or []
        print_fn      = print if print_fn is None else print_fn

        extra_str = "".join(
            f"  {s}" for s in self._format_extra_cols(extra_vals, extra_headers)
        )
        ls_flag = "" if ls_success else "  [LS fail]"

        print_fn(
            f"  {iter:>4}  {residual_norm:>16.6e}  {residual_norm / reference_residual_norm:>16.6e}"
            f"  {alpha:>16.6e}  {np.linalg.norm(state_step):>16.6e}  {np.linalg.norm(state):>16.6e}"
            + extra_str + ls_flag
        )


    # region _print_footer
    def _print_footer(self, result:SolverResult, print_fn=None):
        print_fn = print if print_fn is None else print_fn

        print_fn(f"Converged: {result.converged}")
        print_fn(f"Reason:    {result.reason}")
        print_fn("_" * 60)




# region BROYDENNEWTON
class BroydenNewtonSolver(NewtonSolver):
    """
    Quasi-Newton solver using Good Broyden (rank-1) Jacobian updates.

    A full Jacobian is computed once at q_0 via model.compute_reduced_jacobian().
    Each accepted Newton step then produces a free rank-1 update using only the
    (Δq, Δr) pair already available from the line search — zero extra FOM calls
    between refreshes.

    At convergence, an accurate Jacobian is optionally recomputed so the returned
    SolverResult carries a reliable J for the adjoint solve.

    Options (all NewtonSolver options apply, plus):
        jac_refresh_interval (int, default 0):
            Recompute the full Jacobian every this many accepted steps.
            0 = never refresh after the initial computation (pure Broyden).
            1 = refresh every step (equivalent to NewtonSolver).
        refresh_jac_at_convergence (bool, default True):
            Recompute an accurate Jacobian at the converged state before returning.
            Disable only if the adjoint will not use result.rom_jacobian.
    """

    # region solve
    def solve(self, initial_state: np.ndarray, datum_state: np.ndarray = None):
        opts = self.opts
        maxiter          = opts.get("maxiter", 30)
        tol_abs          = opts.get("tol_abs", 1e-12)
        tol_rel          = opts.get("tol_rel", 1e-8)
        tol_step_rel     = opts.get("tol_step_rel", 1e-10)
        tol_step_abs     = opts.get("tol_step_abs", 1e-14)
        refresh_interval = opts.get("jac_refresh_interval", 0)
        refresh_at_conv  = opts.get("refresh_jac_at_convergence", True)

        q       = initial_state.copy()
        dq_prev = np.full_like(q, np.inf)
        r       = self.model.evaluate_residuals(rom_state=q)

        # Reference residual — frozen so LSPG reuses the test basis set above
        q_ref = datum_state if datum_state is not None else np.zeros_like(initial_state)
        with self.model.freeze_jacobian():
            r_ref      = self.model.evaluate_residuals(rom_state=q_ref)
        r_norm_ref = np.linalg.norm(r_ref)

        # Full Jacobian once at q_0; all subsequent iterations use Broyden updates
        J = self.model.compute_reduced_jacobian(rom_state=q)
        if J is None:
            J = self._compute_fd_jacobian(state=q, residual=r)
        # Bring residual up to date under the Jacobian just computed (LSPG: test basis now at q_0)
        with self.model.freeze_jacobian():
            r = self.model.evaluate_residuals(rom_state=q)
        r_norm = np.linalg.norm(r)

        history         = []
        converged       = False
        reason          = []
        ls_success      = True
        result          = None
        n_since_refresh = 0

        extra_headers = self.model.iter_diagnostic_headers() + [("J type", 8)]

        self.model.pre_solve_diagnostics(initial_rom_state=q)
        self._print_header(
            residual_norm=r_norm,
            reference_residual_norm=r_norm_ref,
            initial_state=q,
            extra_headers=extra_headers,
            print_fn=self.model.print_fn,
            solver_name="BROYDEN-NEWTON SOLVER",
        )

        if r_norm == 0.0:
            result = SolverResult(rom_state=q, rom_residual=r, converged=True,
                                  reason=["initial residual zero"], rom_jacobian=J,
                                  iterations=0, history=history)
            self._print_footer(result=result, print_fn=self.model.print_fn)
            self.model.post_solve_diagnostics(result)
            return result

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
                reason.append(f"relative step size tolerance {step_rel} < {tol_step_rel}")

            if step_abs < tol_step_abs and ls_success:
                converged = True
                reason.append(f"absolute step size tolerance {step_abs} < {tol_step_abs}")

            if converged:
                if refresh_at_conv:
                    J = self.model.compute_reduced_jacobian(rom_state=q)
                    if J is None:
                        J = self._compute_fd_jacobian(state=q, residual=r)
                    with self.model.freeze_jacobian():
                        r = self.model.evaluate_residuals(rom_state=q)

                result = SolverResult(rom_state=q, rom_residual=r, converged=True,
                                      reason=reason, rom_jacobian=J,
                                      iterations=k, history=history)
                break

            # Periodic full Jacobian refresh (skipped at k==0: J was just computed above)
            do_refresh = k > 0 and refresh_interval > 0 and n_since_refresh >= refresh_interval
            if do_refresh:
                J = self.model.compute_reduced_jacobian(rom_state=q)
                if J is None:
                    J = self._compute_fd_jacobian(state=q, residual=r)
                with self.model.freeze_jacobian():
                    r = self.model.evaluate_residuals(rom_state=q)
                r_norm          = np.linalg.norm(r)
                n_since_refresh = 0
                jac_type        = "FD"
            else:
                jac_type = "FD" if k == 0 else "Broyden"

            # Linear solve
            try:
                dq = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dq, *_ = np.linalg.lstsq(J, -r, rcond=None)

            # Line search with frozen Jacobian (prevents LSPG test basis recomputation)
            with self.model.freeze_jacobian():
                alpha, q_trial, r_trial, ls_success = self._line_search(q, dq, r_norm)

            # Good Broyden rank-1 update: J ← J + (Δr − J Δq) Δq^T / ‖Δq‖²
            # Uses only quantities already computed — zero extra FOM calls
            dq_acc     = alpha * dq
            dr         = r_trial - r
            dq_norm_sq = dq_acc @ dq_acc
            if dq_norm_sq > 1e-28:
                J = J + np.outer(dr - J @ dq_acc, dq_acc) / dq_norm_sq

            q       = q_trial
            r       = r_trial
            dq_prev = dq_acc
            r_norm  = np.linalg.norm(r)
            n_since_refresh += 1

            extra_vals = self.model.iter_diagnostics(rom_state=q, jacobian=J) + [jac_type]
            self._print_iter(
                iter=k + 1,
                residual_norm=r_norm,
                reference_residual_norm=r_norm_ref,
                alpha=alpha,
                state=q,
                state_step=dq_prev,
                ls_success=ls_success,
                extra_headers=extra_headers,
                extra_vals=extra_vals,
                print_fn=self.model.print_fn,
            )

            history.append({
                "residual_norm":       r_norm,
                "alpha":               alpha,
                "line_search_success": ls_success,
                "jac_type":            jac_type,
            })

        if result is None:
            reason = ["maxiter"]
            result = SolverResult(rom_state=q, rom_residual=r, converged=False,
                                  reason=reason, rom_jacobian=J,
                                  iterations=maxiter, history=history)

        self._print_footer(result=result, print_fn=self.model.print_fn)
        self.model.post_solve_diagnostics(result)
        return result
