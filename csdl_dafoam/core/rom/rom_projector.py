import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional
from mpi4py import MPI


# ============================================================
# region BASE CLASS
# ============================================================
class ROMProjectorBase(ABC):
    """
    Abstract base class for ROM projectors.

    A projector defines how the FOM residual r(w) is projected into the
    reduced space to form the ROM residual r_rom(q), and how the ROM
    Jacobian J_rom is assembled. The choice of test basis Psi distinguishes
    Galerkin from Petrov-Galerkin methods.

    All methods that operate on FOM-sized distributed vectors accept and
    return local arrays (one per MPI rank). Reduction across ranks is
    handled internally using the provided communicator.

    Parameters
    ----------
    pod_modes : np.ndarray, shape (n_local, n_modes)
        Local portion of the POD trial basis Phi.
    reference_state : np.ndarray, shape (n_local,)
        Local portion of the FOM reference state w_ref.
    scaling : np.ndarray, shape (n_local,)
        Local portion of the diagonal scaling matrix S.
    weights : np.ndarray, shape (n_local,)
        Local portion of the diagonal inner product weight matrix M.
    comm : MPI.Comm
        MPI communicator.
    """

    def __init__(
        self,
        pod_modes:       np.ndarray,
        reference_state: np.ndarray,
        scaling:         np.ndarray,
        weights:         np.ndarray,
        comm:            MPI.Comm,
    ):
        self.Phi     = pod_modes
        self.w_ref   = reference_state
        self.s       = scaling
        self.m       = weights
        self.comm    = comm
        self.n_modes = pod_modes.shape[1]

    # ----------------------------------------------------------
    # region shared public methods
    # ----------------------------------------------------------
    def reconstruct_fom_state(self, q: np.ndarray) -> np.ndarray:
        """
        Reconstruct the FOM state from a ROM state.

        w = w_ref + s * (Phi @ q)

        Parameters
        ----------
        q : np.ndarray, shape (n_modes,)
            ROM state vector.

        Returns
        -------
        w : np.ndarray, shape (n_local,)
            Local portion of the reconstructed FOM state.
        """
        return self.w_ref + self.s * (self.Phi @ q)

    def eval_rom_residual(
        self,
        q:           np.ndarray,
        residual_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Evaluate the ROM residual at a given ROM state q.

        r_rom = Psi^T M r(w(q))

        Parameters
        ----------
        q : np.ndarray, shape (n_modes,)
            ROM state vector.
        residual_fn : callable
            Maps local FOM state w (n_local,) -> local FOM residual r (n_local,).

        Returns
        -------
        r_rom : np.ndarray, shape (n_modes,)
            ROM residual vector (same on all ranks after reduction).
        """
        w   = self.reconstruct_fom_state(q)
        r   = residual_fn(w)
        Psi = self.get_test_basis(fom_state=w, residual_fn=residual_fn)
        return self._reduce(Psi, r)

    def project_fom_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Project a distributed FOM vector into the reduced space using the
        current test basis.

        result = Psi^T M v

        Parameters
        ----------
        v : np.ndarray, shape (n_local,) or (n_local, k)
            Local portion of the FOM vector (or matrix) to project.

        Returns
        -------
        result : np.ndarray, shape (n_modes,) or (n_modes, k)
            Projected vector/matrix (same on all ranks after reduction).
        """
        Psi = self._cached_test_basis
        if Psi is None:
            raise RuntimeError(
                "Test basis has not been computed yet. "
                "Call get_test_basis or update_test_basis first."
            )
        return self._reduce(Psi, v)

    # ----------------------------------------------------------
    # region abstract methods
    # ----------------------------------------------------------
    @abstractmethod
    def get_test_basis(
        self,
        fom_state:   np.ndarray,
        residual_fn: Optional[Callable] = None,
        jacvec_fn:   Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Return the local test basis Psi evaluated at the given FOM state.

        For Galerkin: Psi = Phi (independent of FOM state).
        For LSPG:     Psi = J S Phi (depends on FOM state via Jacobian).

        Parameters
        ----------
        fom_state : np.ndarray, shape (n_local,)
            Current local FOM state (needed for LSPG).
        residual_fn : callable, optional
            FOM residual function (needed by LSPG for FD Jacobian).
        jacvec_fn : callable, optional
            Analytical J^T v function (used by LSPG if available).

        Returns
        -------
        Psi : np.ndarray, shape (n_local, n_modes)
            Local test basis.
        """
        pass

    @abstractmethod
    def compute_rom_jacobian(
        self,
        fom_state:   np.ndarray,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        jacvec_fn:   Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Assemble the ROM Jacobian J_rom = Psi^T M J S Phi.

        Parameters
        ----------
        fom_state : np.ndarray, shape (n_local,)
            Current local FOM state.
        residual_fn : callable
            FOM residual function (used for FD Jacobian columns).
        jacvec_fn : callable, optional
            Analytical J^T v function (used if available, avoids FD).

        Returns
        -------
        J_rom : np.ndarray, shape (n_modes, n_modes)
            Dense ROM Jacobian (same on all ranks after reduction).
        """
        pass

    # ----------------------------------------------------------
    # region private helpers
    # ----------------------------------------------------------
    _cached_test_basis = None  # set by subclasses after computing Psi

    def _reduce(self, Psi: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute Psi^T (m * v) and reduce across MPI ranks.

        Handles both vector (n_local,) and matrix (n_local, k) inputs.
        """
        m = self.m
        if v.ndim == 1:
            local  = Psi.T @ (m * v)
            result = np.zeros_like(local)
        else:
            local  = Psi.T @ (m[:, None] * v)
            result = np.zeros_like(local)

        self.comm.Allreduce(local, result, op=MPI.SUM)
        return result

    def _jac_vec_fd(
        self,
        fom_state:   np.ndarray,
        direction:   np.ndarray,
        residual_fn: Callable,
        r0:          Optional[np.ndarray] = None,
        step:        float = 1e-6,
    ) -> np.ndarray:
        """
        Finite-difference approximation of J @ direction.

        Uses a step size scaled by the direction norm to balance truncation
        and cancellation errors.
        """
        v = direction

        # Scale h relative to the direction magnitude
        v_norm_local  = np.dot(v, v)
        v_norm_global = np.zeros(1)
        self.comm.Allreduce(v_norm_local, v_norm_global, op=MPI.SUM)
        v_norm = np.sqrt(v_norm_global[0])
        h      = step * v_norm if v_norm > 0 else step

        r0     = residual_fn(fom_state) if r0 is None else r0
        r_pert = residual_fn(fom_state + h * v)
        return (r_pert - r0) / h

    def _jac_mat_fd(
        self,
        fom_state:   np.ndarray,
        matrix:      np.ndarray,
        residual_fn: Callable,
        step:        float = 1e-6,
    ) -> np.ndarray:
        """
        Finite-difference approximation of J @ matrix, column by column.

        Parameters
        ----------
        matrix : np.ndarray, shape (n_local, n_cols)
            Matrix whose columns are used as FD directions.

        Returns
        -------
        JM : np.ndarray, shape (n_local, n_cols)
            Approximate J @ matrix.
        """
        r0  = residual_fn(fom_state)
        JM  = np.zeros_like(matrix)
        for i in range(matrix.shape[1]):
            JM[:, i] = self._jac_vec_fd(fom_state, matrix[:, i], residual_fn, r0=r0, step=step)
        return JM


# ============================================================
# region GALERKIN PROJECTOR
# ============================================================
class GalerkinProjector(ROMProjectorBase):
    """
    Galerkin projector: test basis equals trial basis (Psi = Phi).

    The ROM residual and Jacobian are:

        r_rom = Phi^T M r(w)
        J_rom = Phi^T M J S Phi

    This is the simplest and cheapest projector. The test basis is constant
    (independent of the FOM state), so no Jacobian evaluation is needed to
    form Psi, only to assemble J_rom.

    Parameters
    ----------
    jac_mode : str
        How J_rom columns are computed. Options:
        - 'fd'         : finite differences (default, works for any FOM)
        - 'analytical' : uses jacvec_fn if provided (avoids FD)
    fd_step : float
        Step size for finite-difference Jacobian approximation.
    """

    def __init__(self, jac_mode: str = "fd", fd_step: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        if jac_mode not in ("fd", "analytical"):
            raise ValueError(f"jac_mode must be 'fd' or 'analytical', got '{jac_mode}'")
        self.jac_mode = jac_mode
        self.fd_step  = fd_step
        self._cached_test_basis = self.Phi  # constant for Galerkin

    def get_test_basis(self, fom_state=None, residual_fn=None, jacvec_fn=None):
        """For Galerkin, Psi = Phi regardless of FOM state."""
        return self.Phi

    def compute_rom_jacobian(
        self,
        fom_state:   np.ndarray,
        residual_fn: Callable,
        jacvec_fn:   Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Assemble J_rom = Phi^T M J S Phi.

        With jac_mode='fd', each column of S Phi is used as an FD direction.
        With jac_mode='analytical', jacvec_fn(v) = J^T v is used instead,
        computing (Phi^T M J S Phi)^T = (S Phi)^T J^T M Phi column by column.
        """
        Phi = self.Phi
        s   = self.s
        m   = self.m

        if self.jac_mode == "fd":
            S_Phi  = s[:, None] * Phi                           # (n_local, n_modes)
            JS_Phi = self._jac_mat_fd(fom_state, S_Phi, residual_fn, step=self.fd_step)
            return self._reduce(Phi, JS_Phi)                    # Phi^T M (J S Phi)

        elif self.jac_mode == "analytical":
            if jacvec_fn is None:
                raise ValueError("jacvec_fn must be provided when jac_mode='analytical'.")
            # Compute J^T (M Phi) column by column, then transpose
            m_Phi     = m[:, None] * Phi                        # (n_local, n_modes)
            JT_m_Phi  = np.zeros_like(m_Phi)
            for i in range(self.n_modes):
                JT_m_Phi[:, i] = jacvec_fn(m_Phi[:, i])

            # J_rom = (J^T M Phi)^T (S Phi), reduced across ranks
            local  = JT_m_Phi.T @ (s[:, None] * Phi)
            J_rom  = np.zeros_like(local)
            self.comm.Allreduce(local, J_rom, op=MPI.SUM)
            return J_rom


# ============================================================
# region LSPG PROJECTOR
# ============================================================
class LSPGProjector(ROMProjectorBase):
    """
    Least-Squares Petrov-Galerkin (LSPG) projector.

    The test basis is Psi = J S Phi (the Jacobian applied to the scaled
    trial basis), giving the ROM system:

        r_rom = (J S Phi)^T M r(w)
        J_rom = (J S Phi)^T M (J S Phi)

    J_rom is always symmetric positive semi-definite for LSPG, which makes
    the reduced linear system better conditioned than Galerkin in general.

    The test basis must be recomputed whenever the FOM state changes, making
    each Newton iteration more expensive than Galerkin (requires n_modes + 1
    FOM residual evaluations per iteration to update Psi via FD).

    Parameters
    ----------
    fd_step : float
        Step size for finite-difference Jacobian columns (used to build Psi).
    update_psi_every : int
        Recompute Psi every this many Newton iterations. Setting > 1 reduces
        cost at the expense of using a slightly outdated test basis.
    """

    def __init__(self, fd_step: float = 1e-6, update_psi_every: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.fd_step          = fd_step
        self.update_psi_every = update_psi_every
        self._cached_test_basis = None
        self._psi_iter          = 0   # tracks when Psi was last updated

    def get_test_basis(
        self,
        fom_state:   np.ndarray,
        residual_fn: Optional[Callable] = None,
        jacvec_fn:   Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Return Psi = J S Phi, recomputing if update_psi_every iterations
        have passed since the last update.

        Either residual_fn (for FD) or jacvec_fn (for analytical J v)
        must be provided.
        """
        should_update = (
            self._cached_test_basis is None
            or self._psi_iter % self.update_psi_every == 0
        )

        if should_update:
            self._cached_test_basis = self._compute_psi(fom_state, residual_fn, jacvec_fn)

        self._psi_iter += 1
        return self._cached_test_basis

    def compute_rom_jacobian(
        self,
        fom_state:   np.ndarray,
        residual_fn: Callable,
        jacvec_fn:   Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Assemble J_rom = Psi^T M Psi (symmetric, positive semi-definite).

        Psi must already be computed via get_test_basis before calling this.
        """
        Psi = self._cached_test_basis
        if Psi is None:
            Psi = self.get_test_basis(fom_state, residual_fn, jacvec_fn)

        local = Psi.T @ (self.m[:, None] * Psi)
        J_rom = np.zeros_like(local)
        self.comm.Allreduce(local, J_rom, op=MPI.SUM)
        return J_rom

    # ----------------------------------------------------------
    # region private helpers
    # ----------------------------------------------------------
    def _compute_psi(
        self,
        fom_state:   np.ndarray,
        residual_fn: Optional[Callable],
        jacvec_fn:   Optional[Callable],
    ) -> np.ndarray:
        """Compute Psi = J S Phi."""
        S_Phi = self.s[:, None] * self.Phi  # (n_local, n_modes)

        if jacvec_fn is not None:
            # Analytical: apply J to each column of S Phi
            Psi = np.zeros_like(S_Phi)
            for i in range(self.n_modes):
                Psi[:, i] = jacvec_fn(S_Phi[:, i])
        elif residual_fn is not None:
            # FD: J S Phi column by column
            Psi = self._jac_mat_fd(fom_state, S_Phi, residual_fn, step=self.fd_step)
        else:
            raise ValueError(
                "Either residual_fn (for FD Jacobian) or jacvec_fn "
                "(for analytical Jacobian) must be provided to build the LSPG test basis."
            )

        return Psi
