import numpy as np
import csdl_alpha as csdl
from mpi4py import MPI

from csdl_dafoam.core.csdl_dafoam import has_global_nan_or_inf
from csdl_dafoam.core.rom.rom_solver import ROMSolverBase, NewtonSolver
from csdl_dafoam.core.rom.rom_projector import ROMProjectorBase, GalerkinProjector, LSPGProjector


"""
DAFoam ROM implemented as a CSDL implicit component.

Notation used throughout:
    w       FOM state vector          (distributed, n_local)
    q       ROM state vector          (dense, n_modes)
    Phi     POD trial basis           (distributed, n_local x n_modes)
    w_ref   FOM reference state       (distributed, n_local)
    s       Scaling diagonal          (distributed, n_local)
    m       Inner product weights     (distributed, n_local)
    r       FOM residual              (distributed, n_local)
    r_rom   ROM residual              (dense, n_modes)
    J_rom   ROM Jacobian              (dense, n_modes x n_modes)
    Psi     Test basis                (distributed, n_local x n_modes)
              Galerkin: Psi = Phi
              LSPG:     Psi = J S Phi

The ROM seeks q such that:
    Psi^T M r(w_ref + S Phi q) = 0

Solved iteratively (Newton by default) in solve_residual_equations.
Derivatives are propagated through apply_inverse_jacobian and
compute_jacvec_product, keeping the CSDL adjoint chain intact.
"""


# ============================================================
# region DAFOAMROM
# ============================================================
class DAFoamROM(csdl.experimental.CustomImplicitOperation):
    """
    DAFoam Galerkin / LSPG ROM as a CSDL implicit component.

    This class is a thin wrapper. All ROM math lives in ROMProjectorBase
    (rom_projector.py) and ROMSolverBase (rom_solver.py). This class is
    responsible only for:
        - declaring CSDL inputs/outputs (evaluate)
        - driving the ROM solve (solve_residual_equations)
        - applying the inverse Jacobian for the adjoint (apply_inverse_jacobian)
        - computing parameter sensitivities (compute_jacvec_product)

    Parameters
    ----------
    dafoam_instance : PYDAFOAM
        Initialized DAFoam instance.
    pod_modes : np.ndarray, shape (n_local, n_modes)
        Local portion of POD trial basis. Can also be passed as a CSDL
        variable in evaluate() if it varies during optimization.
    reference_state : np.ndarray, shape (n_local,)
        Local FOM reference state w_ref.
    scaling : np.ndarray, shape (n_local,)
        Diagonal scaling vector s.
    weights : np.ndarray, shape (n_local,)
        Inner product weight vector m (e.g. cell volumes).
    rom_type : str
        'galerkin' or 'lspg'.
    solver : ROMSolverBase, optional
        ROM solver instance. Defaults to NewtonSolver with default options.
        Pass a custom solver to change convergence tolerances or algorithm.
    projector_kwargs : dict, optional
        Extra keyword arguments forwarded to the projector constructor
        (e.g. jac_mode, fd_step, update_psi_every).

    Examples
    --------
    Basic Galerkin ROM with default Newton solver::

        rom = DAFoamROM(
            dafoam_instance  = dafoam_instance,
            pod_modes        = phi_local,
            reference_state  = w_ref_local,
            scaling          = scaling_local,
            weights          = weights_local,
            rom_type         = 'galerkin',
        )
        q = rom.evaluate(dafoam_input_variables_group)

    LSPG ROM with a custom Newton solver::

        solver = NewtonSolver(max_iter=100, tol_rel=1e-8, print_fn=print0)
        rom = DAFoamROM(..., rom_type='lspg', solver=solver,
                        projector_kwargs={'update_psi_every': 2})
    """

    def __init__(
        self,
        dafoam_instance,
        pod_modes:        np.ndarray       = None,
        reference_state:  np.ndarray       = None,
        scaling:          np.ndarray       = None,
        weights:          np.ndarray       = None,
        rom_type:         str              = "galerkin",
        solver:           ROMSolverBase    = None,
        projector_kwargs: dict             = None,
    ):
        super().__init__()

        self.dafoam_instance = dafoam_instance
        self.comm            = dafoam_instance.comm
        self.rank            = self.comm.rank

        # Store constant arrays (may be overridden by CSDL variables in evaluate)
        self._pod_modes_const       = pod_modes
        self._reference_state_const = reference_state
        self._scaling_const         = scaling
        self._weights_const         = weights

        # Track which of the above are CSDL inputs (set in evaluate)
        self._pod_modes_is_input       = False
        self._reference_state_is_input = False
        self._scaling_is_input         = False

        # ROM configuration
        if rom_type.lower() not in ("galerkin", "lspg"):
            raise ValueError(f"rom_type must be 'galerkin' or 'lspg', got '{rom_type}'")
        self.rom_type         = rom_type.lower()
        self.projector_kwargs = projector_kwargs or {}
        self.solver           = solver or NewtonSolver(print_fn=self.print0)

        # Set during solve, used in derivative methods
        self._projector:   ROMProjectorBase = None
        self._cached_q:    np.ndarray       = None
        self._cached_w:    np.ndarray       = None
        self._cached_J_rom: np.ndarray      = None

        self._n_local_states = dafoam_instance.getNLocalAdjointStates()
        self._n_modes        = None  # set in evaluate


    # ============================================================
    # region evaluate
    # ============================================================
    def evaluate(
        self,
        dafoam_input_variables_group: csdl.VariableGroup,
        pod_modes:       csdl.Variable = None,
        reference_state: csdl.Variable = None,
        scaling:         csdl.Variable = None,
    ) -> csdl.Variable:
        """
        Declare CSDL inputs/outputs and return the ROM state variable.

        DAFoam solver inputs (mesh coords, patch velocity, etc.) are read
        from dafoam_input_variables_group via the inputInfo DAOption.

        pod_modes, reference_state, and scaling are optional CSDL variables.
        If not provided here, the constants passed to __init__ are used.
        Pass them as CSDL variables only if you need gradients w.r.t. them.

        Returns
        -------
        dafoam_rom_states : csdl.Variable, shape (n_modes,)
            The converged ROM state q.
        """
        # DAFoam solver inputs
        input_dict = self.dafoam_instance.getOption("inputInfo")
        for name, info in input_dict.items():
            if "solver" in info["components"]:
                self.declare_input(name, getattr(dafoam_input_variables_group, name))

        # Optional CSDL variables for ROM quantities
        self._n_modes = self._declare_optional_input(pod_modes,       "pod_modes",
                                                      self._pod_modes_const,
                                                      "_pod_modes_is_input",
                                                      return_n_modes=True)
        self._declare_optional_input(reference_state, "reference_state",
                                     self._reference_state_const,
                                     "_reference_state_is_input")
        self._declare_optional_input(scaling,         "scaling",
                                     self._scaling_const,
                                     "_scaling_is_input")

        dafoam_rom_states = self.create_output("dafoam_rom_states", (self._n_modes,))
        return dafoam_rom_states


    # ============================================================
    # region solve_residual_equations
    # ============================================================
    def solve_residual_equations(self, input_vals, output_vals):
        """
        Drive the ROM Newton solve to find q such that r_rom(q) = 0.

        Steps:
        1. Update DAFoam with current input values (mesh, BCs, etc.)
        2. Build the projector from current ROM arrays
        3. Call solver.solve with residual_fn and jacobian_fn closures
        4. Cache converged q, w, J_rom for use in derivative methods
        """
        dafoam_instance = self.dafoam_instance

        # --- update DAFoam inputs ---
        dafoam_instance.set_solver_input(input_vals)

        # --- assemble current ROM arrays ---
        arrays = self._get_rom_arrays(input_vals)

        # --- build projector ---
        self._projector = self._build_projector(arrays)

        # --- define residual and Jacobian closures ---
        def residual_fn(q):
            return self._projector.eval_rom_residual(q, self._eval_fom_residual)

        def jacobian_fn(q):
            w = self._projector.reconstruct_fom_state(q)
            self._projector.get_test_basis(
                fom_state   = w,
                residual_fn = self._eval_fom_residual,
                jacvec_fn   = self._jacvec_fn,
            )
            return self._projector.compute_rom_jacobian(
                fom_state   = w,
                residual_fn = self._eval_fom_residual,
                jacvec_fn   = self._jacvec_fn,
            )

        # --- solve ---
        q0 = np.zeros(self._n_modes) if self._cached_q is None else self._cached_q.copy()
        q, converged = self.solver.solve(residual_fn, jacobian_fn, q0)

        if not converged:
            self.print0("ROM solve did not converge. Writing NaNs to output.")
            output_vals["dafoam_rom_states"] = np.full(self._n_modes, np.nan)
        else:
            output_vals["dafoam_rom_states"] = q

        # Cache for derivative methods
        self._cached_q    = q
        self._cached_w    = self._projector.reconstruct_fom_state(q)
        self._cached_J_rom = jacobian_fn(q)


    # ============================================================
    # region apply_inverse_jacobian
    # ============================================================
    def apply_inverse_jacobian(self, input_vals, output_vals, d_outputs, d_residuals, mode):
        """
        Apply (J_rom)^{-T} to d_outputs["dafoam_rom_states"].

        This is the adjoint linear solve in the reduced space. Because J_rom
        is dense and small (n_modes x n_modes), a direct solve is used.
        """
        if mode == "fwd":
            raise NotImplementedError("Forward mode not implemented for DAFoamROM.")

        if self._cached_J_rom is None:
            raise RuntimeError(
                "Cached ROM Jacobian is None. solve_residual_equations must "
                "run successfully before apply_inverse_jacobian."
            )

        v = d_outputs["dafoam_rom_states"]
        try:
            d_residuals["dafoam_rom_states"] += np.linalg.solve(self._cached_J_rom.T, v)
        except np.linalg.LinAlgError:
            self.print0("Warning: ROM Jacobian is singular. Using least-squares solve.")
            sol, _, _, _ = np.linalg.lstsq(self._cached_J_rom.T, v, rcond=None)
            d_residuals["dafoam_rom_states"] += sol


    # ============================================================
    # region compute_jacvec_product
    # ============================================================
    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, d_residuals, mode):
        """
        Compute (dr_rom/d_inputs)^T @ lam where lam = d_residuals["dafoam_rom_states"].

        The ROM residual r_rom = Psi^T M r(w) depends on the FOM inputs
        through two paths:
            1. Reconstruction: w = w_ref + S Phi q depends on w_ref, S, Phi
            2. FOM residual:   r(w, inputs) depends directly on FOM inputs

        Sensitivity to FOM inputs (mesh, BCs) uses DAFoam's calcJacTVecProduct.
        Sensitivity to ROM arrays (reference_state, scaling, pod_modes) is
        computed analytically where possible.
        """
        if mode == "fwd":
            raise NotImplementedError("Forward mode not implemented for DAFoamROM.")

        if "dafoam_rom_states" not in d_residuals:
            return

        lam = d_residuals["dafoam_rom_states"]
        w   = self._cached_w
        q   = self._cached_q
        Psi = self._projector._cached_test_basis
        Phi = self._projector.Phi
        m   = self._projector.m
        s   = self._projector.s

        # Set DAFoam state to cached converged state
        if not has_global_nan_or_inf(w, self.comm):
            self.dafoam_instance.setStates(w)
        else:
            self.print0("compute_jacvec_product: NaN in cached state, skipping setStates.")
            return

        # Shared seed: M Psi lam  (FOM-sized distributed vector)
        m_Psi_lam = np.ascontiguousarray(m * (Psi @ lam))

        # --- sensitivity to DAFoam inputs (mesh coords, BCs, etc.) ---
        input_dict = self.dafoam_instance.getOption("inputInfo")
        for name in list(input_vals.keys()):
            if name not in input_dict or name not in d_inputs:
                continue
            input_type = input_dict[name]["type"]
            jac_input  = input_vals[name].copy()
            product    = np.zeros_like(jac_input)
            self.dafoam_instance.solverAD.calcJacTVecProduct(
                name, input_type, jac_input,
                "aero_residuals", "residual",
                m_Psi_lam, product,
            )
            d_inputs[name] += product

        # --- sensitivity to ROM arrays (if declared as CSDL inputs) ---
        needs_rom_sens = any(
            k in d_inputs for k in ("reference_state", "scaling", "pod_modes")
        )
        if not needs_rom_sens:
            return

        # J^T (M Psi lam) — shared by reference_state and scaling sensitivities
        JT_seed = self._jacvec_fn(m_Psi_lam)

        if "reference_state" in d_inputs:
            # d(r_rom)/d(w_ref) = Psi^T M J,  sensitivity = J^T M Psi lam
            d_inputs["reference_state"] += JT_seed

        if "scaling" in d_inputs:
            # d(w)/d(s) = diag(Phi q),  chain rule gives (Phi q) * J^T M Psi lam
            d_inputs["scaling"] += (Phi @ q) * JT_seed

        if "pod_modes" in d_inputs:
            r = self._eval_fom_residual(w)

            # Path 1 (reconstruction): d(w)/d(Phi) = s q^T
            d_inputs["pod_modes"] += s[:, None] * np.outer(JT_seed, q)

            # Path 2 (projection): d(Psi^T M r)/d(Phi), depends on rom_type
            if self.rom_type == "galerkin":
                # Psi = Phi, so d(Phi^T M r)/d(Phi) = m * r  (outer product)
                d_inputs["pod_modes"] += np.outer(m * r, lam)
            elif self.rom_type == "lspg":
                # Psi = J S Phi, sensitivity requires J^T applied to (m * r)
                JT_m_r = self._jacvec_fn(m * r)
                d_inputs["pod_modes"] += s[:, None] * np.outer(JT_m_r, lam)


    # ============================================================
    # region private helpers
    # ============================================================
    def _eval_fom_residual(self, fom_state: np.ndarray) -> np.ndarray:
        """Evaluate the FOM residual r(w). Assumes DAFoam inputs are already set."""
        self.dafoam_instance.setStates(fom_state)
        return self.dafoam_instance.getResiduals()

    def _jacvec_fn(self, vec: np.ndarray) -> np.ndarray:
        """Compute J^T @ vec using DAFoam's adjoint product."""
        seed    = np.ascontiguousarray(vec)
        product = np.zeros_like(seed)
        self.dafoam_instance.solverAD.calcJacTVecProduct(
            "dafoam_solver_states", "stateVar",
            self.dafoam_instance.getStates(),
            "aero_residuals", "residual",
            seed, product,
        )
        return product

    def _get_rom_arrays(self, input_vals: dict) -> dict:
        """Resolve ROM arrays from either CSDL inputs or stored constants."""
        return {
            "pod_modes":       input_vals["pod_modes"]       if self._pod_modes_is_input       else self._pod_modes_const,
            "reference_state": input_vals["reference_state"] if self._reference_state_is_input else self._reference_state_const,
            "scaling":         input_vals["scaling"]         if self._scaling_is_input         else self._scaling_const,
            "weights":         self._weights_const,  # weights never a CSDL input for now
        }

    def _build_projector(self, arrays: dict) -> ROMProjectorBase:
        """Construct the appropriate projector from current ROM arrays."""
        common = dict(
            pod_modes       = arrays["pod_modes"],
            reference_state = arrays["reference_state"],
            scaling         = arrays["scaling"],
            weights         = arrays["weights"],
            comm            = self.comm,
        )
        if self.rom_type == "galerkin":
            return GalerkinProjector(**common, **self.projector_kwargs)
        elif self.rom_type == "lspg":
            return LSPGProjector(**common, **self.projector_kwargs)

    def _declare_optional_input(
        self,
        variable,
        name:            str,
        constant:        np.ndarray,
        is_input_attr:   str,
        return_n_modes:  bool = False,
    ):
        """
        Declare a CSDL input if variable is provided, otherwise use constant.

        Sets the corresponding _*_is_input flag on self.
        If return_n_modes=True, returns the number of modes inferred from
        the pod_modes shape.
        """
        if variable is not None:
            if constant is not None:
                self.print0(
                    f"Warning: '{name}' passed to evaluate() overrides the "
                    "constant value passed to __init__."
                )
            self.declare_input(name, variable)
            setattr(self, is_input_attr, True)
            if return_n_modes:
                return variable.value.shape[1]
        else:
            if constant is None:
                raise ValueError(
                    f"'{name}' must be provided either as a constant in __init__ "
                    "or as a CSDL variable in evaluate()."
                )
            setattr(self, is_input_attr, False)
            if return_n_modes:
                return constant.shape[1]

    def print0(self, msg: str, **kwargs):
        """Print only on MPI rank 0."""
        if self.rank == 0:
            print(msg, **kwargs)
