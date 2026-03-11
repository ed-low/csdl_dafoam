import numpy as np
import csdl_alpha as csdl
from csdl_dafoam.core.csdl_dafoam import has_global_nan_or_inf
from mpi4py import MPI


'''
This is the DAFoam (Least Squares Petrov) Galerkin POD ROM implemented as a CSDL implicit component.
Within this code, we'll use the following shorthand for representing certain quantities:

    FOM state ............. w       (vector, distributed)
    ROM state ............. q       (vector)
    POD modes ............. Phi     (matrix, distributed)
    FOM reference state ... w_ref   (vector, distributed)
    Scaling ............... s       (vector, distributed)
    Residual scaling ...... s_r     (vector, distributed)
    Inner product weights . m       (vector, distributed)
    FOM residual .......... r       (vector, distribuited)
    ROM residual .......... r_rom   (vector)
    FOM Jacobian .......... J       (matrix, not actually stored)
    ROM Jacobian .......... J_rom   (matrix)

where "distributed" indicates that this quantity is partitioned among ranks and not just copied.

The full order model (FOM) can be written as:
    r(w) = 0

We've constructed our POD such that:
    w = w_ref + S @ Phi @ q
where s forms the diagonal of S (our scaling vector/matrix), and Phi is our basis.

In our reduced order model (ROM), we seek to solve
    Psi^T @ M @ r(w) = 0
where m forms the diagonal of M (our inner product weight vector/matrix), and Psi is our test basis

For a standard Galerkin ROM, we choose Psi to be Phi, our POD modes.
For a LSPG ROM, we choose Psi to be J @ Phi

To solve our reduced system, we use Newton's method:
    J_rom_k @ dq_k = - r_rom_k
where J_rom_k = Psi^T @ M @ J_k @ S @ Phi, and k is our current iteration.

The update follows q_{k+1} = q_k + alpha_k * dq_k, where alpha_k is a step length (constant or from line search)

In summary:
    (Newton)
    J_rom_k dq_k = -r_rom_k   
    dq_{k+1} = q_k + alpha_k * dq_k
    
    (Expansion)
    w = w_ref + S @ Phi @ q

    (ROM quantities)
    r_rom = Psi^T @ M @ r
    J_rom = Psi^T @ M @ J @ S @ Phi
    Psi = { S @ Phi     if Galerkin
          { J @ S @ Phi if Petrov-Galerkin
    
'''

# dafault newton_options = {
#                         'maxiter':          50,       # max Newton iterations
#                         'tol_rel':          1e-6,     # relative residual tolerance
#                         'tol_abs':          1e-10,    # absolute residual tolerance
#                         'tol_step_rel':     1e-8,     # Newton will consider converged if the stepsize vs state magnitudes reach this tolerance
#                         'ls_alpha0':        1.0,      # initial line search step
#                         'ls_rho':           0.5,      # backtrack factor
#                         'ls_c1':            1e-4,     # Armijo sufficient decrease constant
#                         'ls_maxiter':       10,       # max line search iterations
#                         'ls_freeze_basis':  True,     # Whether or not to freeze the trial basis for the line search (disabling will increase cost)
#                         'jac_fd_step':      1e-6,     # FD step for Jacobian
#                         'update_test_basis_every': 1, # how often to recompute Psi (every n iters)
#                         'min_newton_steps:  1,        # force the newton solver to take this many steps, even if initially converged
#                         'verbose': {
#                                     'level':          1,       # 0=silent, 1=standard, 2=diagnostic, 3=expensive
#                                     'progress':       True,    # per-iteration table (Level 1)
#                                     'header':         True,    # solve header (Level 1)
#                                     'footer':         True,    # solve footer (Level 1)
#                                     'numerics':       False,   # cond, SVD, FD Jacobian check (Level 2)
#                                     'residuals':      False,   # per-variable breakdown (Level 2)
#                                     'basis_quality':  False,   # projection capture, q ratio (Level 2)
#                                     'basis_expensive':False,   # mode contributions (Level 3)
#                                     }
#                         }



# region DAFOAMROM
class DAFoamROM(csdl.experimental.CustomImplicitOperation):
    def __init__(
        self, 
        dafoam_instance,
        pod_modes=None,          # constant modes (numpy, distributed)
        reference_state=None,    # distributed numpy
        scaling=None,            # distributed numpy (diagonal of S)
        residual_scaling=None,   # distributed numpy (diagonal of S_r)
        weights=None,            # distributed numpy (diagonal of M)
        rom_type='lspg',         # 'galerkin' or 'lspg'
        jac_mode='fd',           # 'analytical' (galerkin only) or 'fd'
        param_sens_mode='galerkin_adjoint', # or 'fd'
        newton_options=None,     # maxiter, tol, line search params, fd eps
        exclude_from_projection=None, # list of strings indicating variables to ignore during ROM computation
        use_normalized_residuals=True,
        apply_temperature_residual_fix=True,
        temperature_residual_cp_val=1005.,
        write_residuals_with_solutions=False,
    ):
        super().__init__()
        self.dafoam_instance = dafoam_instance

        # These will either be the user supplied constants, or we'll update them
        # As the optimization progresses (that way all functions can reference them)
        self.pod_modes          = pod_modes
        self.reference_state    = reference_state
        self.scaling            = scaling
        self.residual_scaling   = residual_scaling
        self.weights            = weights
        self.test_basis         = None

        self.rom_type           = self._check_if_in_list(rom_type,          ["galerkin", "lspg"])
        self.jac_mode           = self._check_if_in_list(jac_mode,          ["fd", "analytical"])
        self.param_sens_mode    = self._check_if_in_list(param_sens_mode,   ["galerkin_adjoint", "fd"])

        self.use_normalized_residuals   = use_normalized_residuals
        
        # Merge user newton options, then resolve verbose once
        self.newton_options             = self._init_newton_options(newton_options)

        self.exclude_from_projection    = exclude_from_projection

        # These are used in the evaluate section, but initialized here
        self.pod_modes_is_csdl_var       = None
        self.reference_state_is_csdl_var = None
        self.scaling_is_csdl_var         = None
        self.checked_basis_orthogonality = None

        # Temperature residual fixes
        self.apply_temperature_residual_fix = apply_temperature_residual_fix
        self.temperature_residual_cp_val    = temperature_residual_cp_val

        # File I/O
        self.write_residuals_with_solutions = write_residuals_with_solutions

        # DAFoam
        self.n_local_states = dafoam_instance.getNLocalAdjointStates()
        
        # setup indexing for local state vector in case we want to obtain a specific variable
        names, indices      = dafoam_instance.getStateVariableMap(includeComponentSuffix=False)
        self.state_indices  = {name: indices == names.index(name) for name in names}

        # MPI
        self.comm           = dafoam_instance.comm
        self.rank           = self.comm.rank
        self.comm_size      = self.comm.size

        # Cached values (set during solve, used in derivative methods)
        self._cached_q      = None # converged reduced state
        self._cached_J_r    = None # ROM Jacobian at convergence
        self._cached_w      = None # reconstructed FOM state at convergence
        self._m_eff         = None # Effective weights: m/s_r (Galerkin) m/(s_r ** 2) (LSPG)
        self.solution_iter  = 0

     
    # region evaluate
    def evaluate(self, dafoam_input_variables_group:csdl.VariableGroup, 
                 pod_modes:csdl.Variable=None, 
                 scaling:csdl.Variable=None, 
                 reference_state:csdl.Variable=None):
        
        # Read daOptions to set proper inputs
        inputDict = self.dafoam_instance.getOption("inputInfo")
        for inputName in inputDict.keys():
            if "solver" in inputDict[inputName]["components"]:
                self.declare_input(inputName, getattr(dafoam_input_variables_group, inputName))

        num_modes = self._set_as_constant_or_input(pod_modes, "pod_modes")
        self._set_as_constant_or_input(scaling, "scaling")
        self._set_as_constant_or_input(reference_state, "reference_state")

        # Set outputs
        dafoam_rom_states = self.create_output('dafoam_rom_states', (num_modes,))
        self.basis_size   = num_modes

        return dafoam_rom_states


    # region solve_residual_equations
    def solve_residual_equations(self, input_vals, output_vals):
        dafoam_instance         = self.dafoam_instance
        
        # Initialize/update quantities
        self.pod_modes          = input_vals["pod_modes"]       if self.pod_modes_is_csdl_var           else self.pod_modes
        self.scaling            = input_vals["scaling"]         if self.scaling_is_csdl_var             else self.scaling
        self.reference_state    = input_vals["reference_state"] if self.reference_state_is_csdl_var     else self.reference_state
        self.weights            = np.ones_like(self.reference_state) if self.weights is None            else self.weights
        self.residual_scaling   = np.ones_like(self.reference_state) if self.residual_scaling is None   else self.residual_scaling

        if self.rom_type == 'galerkin':
            self._m_eff = self.weights / self.residual_scaling           # M R^{-1}
        elif self.rom_type == 'lspg':
            self._m_eff = self.weights / (self.residual_scaling ** 2)    # R^{-1} M R^{-1}

        # Check orthogonaliy (only once if constant basis)
        if self.pod_modes_is_csdl_var or self.checked_basis_orthogonality:
            self._check_basis_orthogonality()
            self.checked_basis_orthogonality = True

        # Zero out excluded variables in m_eff (e.g. T for incompressible SA)
        if self.exclude_from_projection:
            var_names, var_idx = self.dafoam_instance.getStateVariableMap(includeComponentSuffix=False)
            for exclude_var in self.exclude_from_projection:
                if exclude_var not in var_names:
                    self.print0(f"Warning: '{exclude_var}' not found in state variable map. Skipping.")
                    continue
                local_mask              = var_idx == var_names.index(exclude_var)
                self._m_eff[local_mask] = 0.0

        # Make sure solver is updated with the most recent input values
        dafoam_instance.set_solver_input(input_vals)

        # Initial guess: warm start from previous solution if available, else zero
        # TODO: See if there is a best warm start option
        q0 = self._cached_q.copy() if self._cached_q is not None else np.zeros(self.basis_size)

        q, reason = self._rom_newton_solve(initial_rom_state=q0)

        if reason > 0:
            output_vals["dafoam_rom_states"] = q

        elif reason < 0:
            self.print0("Newton solver failed. Setting DAFoam ROM states to NaNs.")
            output_vals["dafoam_rom_states"] = np.nan * np.ones_like(q)

        else:
            output_vals["dafoam_rom_states"] = q

    # region apply_inverse_jacobian
    def apply_inverse_jacobian(self, input_vals, output_vals, d_outputs, d_residuals, mode):
        # Can't do forward mode
        if mode == 'fwd':
            raise NotImplementedError('forward mode has not been implemented for DAFoamROM')
        
        elif mode == 'rev':
            # Raise error in case cached Jacobian was not computed.
            # TODO: Maybe make this just compute the Jacobian if that is the case?
            if self._cached_J_r is None:
                raise RuntimeError("Cached ROM Jacobian is None. ensure solve_residual_equations ran successfully before calling apply_inverse_jacobian.")
            
            J_romT = self._cached_J_r.T
            v      = d_outputs["dafoam_rom_states"]

            # TODO: Should I have this return NaNs instead of attempting a least-squares solve?
            # TODO: Also, check if accumulation is necessary - that is, = vs +=. Accumulating for now 
            try:
                d_residuals["dafoam_rom_states"] += np.linalg.solve(J_romT, v)
            except np.linalg.LinAlgError:
                self.print0("Warning: ROM Jacobian is singular. Attempting least-squares solve.")
                solution, _, _, _ = np.linalg.lstsq(J_romT, v, rcond=None)
                d_residuals["dafoam_rom_states"] += solution

        else:
            raise ValueError(f'"{mode}" not recognized. Only support "fwd" and "rev" modes')
        
        # Write state to file
        leading_integer = 1
        self.dafoam_instance.solver.writeAdjointFields("", leading_integer + (self.solution_iter + 1) / 10000, self._cached_w, True)
        if self.write_residuals_with_solutions:
            current_residual = self._eval_fom_residual(self._cached_w)
            self.dafoam_instance.solver.writeAdjointFields("res_", leading_integer + (self.solution_iter + 1) / 10000, current_residual, True)
        mesh = np.zeros_like(self.dafoam_instance.xv.flatten())
        self.dafoam_instance.solver.getOFMeshPoints(mesh)
        self.dafoam_instance.solver.writeMeshPoints(mesh, leading_integer + (self.solution_iter + 1) / 10000)
        self.solution_iter += 1


    # region compute_jacvec_product
    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, d_residuals, mode):
        dafoam_instance = self.dafoam_instance

        # We'll use cached values from the previous solve
        # NOTE: Would it be best to use output_vals to get ROM state?
        q = self._cached_q
        w = self._cached_w

        if not has_global_nan_or_inf(w, self.comm):
            self.dafoam_instance.setStates(w)
        else:
            self.print0('DAFoamROM.compute_jacvec_product: Detected NaN(s) in input_vals. Skipping DAFoam setStates')
       
        # Can't do forward mode
        if mode == 'fwd':
            raise NotImplementedError('forward mode has not been implemented for DAFoamROM')

        elif mode == 'rev':
            if "dafoam_rom_states" not in d_residuals:
                self.print0('DAFoamROM.compute_jacvec_product: no "dafoam_rom_states" detected in d_residuals. Continuing without assignment.')
                return

            lam = d_residuals["dafoam_rom_states"]

            m   = self._m_eff
            Psi = self.test_basis
            Phi = self.pod_modes
            s   = self.scaling

            # Shared seed: M @ Psi @ lam (fom size, distributed)
            seed        = np.ascontiguousarray(m * (Psi @ lam))
            seed_norm   = seed.copy()
            
            # Prescale for temperature residual fix
            if self.apply_temperature_residual_fix:
                seed_norm[self.state_indices["T"]] /= self.temperature_residual_cp_val

            # TODO: Assuming that the dafoam_instance always has normalizeResiduals for all states
            # Maybe add some logic to make this more robust for the case that this is not true?
            if not self.use_normalized_residuals:
                res_scale_factors = dafoam_instance.getStateWeights()
                res_scale_factors[self.state_indices["T"]] /= 1005.
                seed_norm *= res_scale_factors

            # FOM inputs: mesh, flow DVs
            input_dict = dafoam_instance.getOption("inputInfo")

            for input_name in list(input_vals.keys()):
                if input_name not in input_dict:
                    continue # skip reconstruction inputs, handled below
                if input_name not in d_inputs:
                    continue # skip if not needed

                input_type = input_dict[input_name]["type"]
                jac_input  = input_vals[input_name].copy()
                product    = np.zeros_like(jac_input)

                dafoam_instance.solverAD.calcJacTVecProduct(
                    input_name,
                    input_type,
                    jac_input,
                    "aero_residuals",
                    "residual",
                    seed_norm,
                    product
                )

                d_inputs[input_name] += product

            # Reconstruction inputs (these are handled if the POD reconstruction inputs are actually CSDL variables)
            needs_reconstruction_sens = any(k in d_inputs for k in ["reference_state", "scaling", "pod_modes"])

            if needs_reconstruction_sens:
                # This vector is shared among all of the sensitivites. Compute once here
                # J^T M Psi lam
                v_shared = self._jacT_vec_product(fom_state=w, vec=seed)

                # Compute contribution from the reference state variable
                # (∂r_rom/∂w_ref)^T lam = (Psi^T M ∂r/∂w ∂w/∂w_ref)^T lam = (Psi^T M J ∂w/∂w_ref)^T lam
                # ∂w/∂w_ref = I
                # (∂r_rom/∂w_ref)^T lam = (Psi^T M J)^T lam = J^T M Psi lam = v_shared
                if "reference_state" in d_inputs:
                    d_inputs["reference_state"] += v_shared

                # Compute contribution from the scaling variable
                # (∂r_rom/∂s)^T lam = (Psi^T M ∂r/∂w ∂w/s)^T lam = (Psi^T M J ∂w/s)^T lam
                # ∂w/s = Phi q
                # (∂r_rom/∂s)^T lam = (Psi^T M J Phi q)^T lam = (Phi q) J^T M Psi lam = (Phi q) v_shared [Phi q is a vector, so transpose is dropped]
                if "scaling" in d_inputs:
                    Phi_q = Phi @ q # (n_local,)
                    d_inputs["scaling"] += Phi_q * v_shared

                # Compute contribution from the modes
                if "pod_modes" in d_inputs:
                    # We have two paths: one from the projection, and one from the reconstruction
                    # (∂r_rom/∂Phi)^T lam = [∂/∂Phi(Psi^T M R)] ^ T lam
                    # Where Psi and R have Phi dependence
                    # Being loose with notation here (since we have arrays)
                    # ∂/∂Phi(Psi^T M R) = ∂/∂Phi(Psi^T) M R + Psi^T M ∂/∂Phi(R)
                    #                    |----projection---| |-reconstruction--|

                    # Path 1: reconstruction (this is shared by both Galerkin and LSPG)
                    # [Psi^T M ∂/∂Phi(R)]^T lam
                    # dw = S dPhi q
                    d_inputs["pod_modes"] += s[:, None] * np.outer(v_shared, q)

                    # Path 2: projection - Phi appears in Psi^T M R
                    r = self._eval_fom_residual(fom_state=w)

                    if self.rom_type == "galerkin":
                        # In this case, Psi = Phi
                        d_inputs["pod_modes"] += np.outer(s * m * r, lam)

                    elif self.rom_type == "lspg":
                        # In this case, Psi = J S Phi
                        JT_r = self._jacT_vec_product(fom_state=w, vec=m * r)
                        d_inputs["pod_modes"] += s[:, None] * np.outer(JT_r, lam)
    
        else:
            raise ValueError(f'"{mode}" not recognized. Only support "fwd" and "rev" modes')
        
    
    # region evaluate_residuals
    def evaluate_residuals(self, input_vals, output_vals, residual_vals):
        self.dafoam_instance.set_solver_input(input_vals)
        q = output_vals["dafoam_rom_states"]
        residual_vals["dafoam_rom_states"] = self._eval_rom_residual(rom_state=q)
        
    
    # region _rom_newton_solve
    def _rom_newton_solve(self, initial_rom_state):
        opts            = self.newton_options
        verb            = opts["verbose"]

        # Initialize some values
        q               = initial_rom_state.copy()
        dq              = np.inf
        alpha           = opts["ls_alpha0"]
        J_rom           = None
        reason          = 0 # Will return 1 for successful solution, 2 for success with failed line search, -2 for max iterations
        ls_success      = True  # assume success until line search runs

        # Setup
        jac_fd_step = opts["jac_fd_step"]
        w           = self._reconstruct_fom_state(q)
        self._update_test_basis(fom_state=w, step=jac_fd_step)
        r_rom       = self._eval_rom_residual(q)
        r_rom_norm  = np.linalg.norm(r_rom)
        
        # Reference residual: evaluated at q=0, used as fixed normalizer
        q_ref          = np.zeros(self.basis_size)
        w_ref_state    = self._reconstruct_fom_state(q_ref)
        r_rom_ref      = self._eval_rom_residual(q_ref)
        r_rom_norm_ref = np.linalg.norm(r_rom_ref)
        r_fom_norm_ref = np.linalg.norm(self._eval_fom_residual(w_ref_state))

        if verb['header']:
            self._print_solve_header(q, w_ref_state, r_rom_norm_ref, r_fom_norm_ref, opts)    

        # Main loop
        for k in range(opts["maxiter"]):

            # Check convergence
            residual_converged = r_rom_norm < opts["tol_abs"] or r_rom_norm / r_rom_norm_ref < opts["tol_rel"]
            step_converged     = np.linalg.norm(dq) * alpha / max(np.linalg.norm(q), 1e-14) < opts.get("tol_step_rel", 1e-8)

            if residual_converged or (step_converged and ls_success):
                if k >= opts["min_newton_steps"]:
                    if verb['progress']:
                        if residual_converged:
                            if r_rom_norm < opts["tol_abs"]:
                                self.print0(f"Newton converged (absolute tolerance)...")
                            elif r_rom_norm / r_rom_norm_ref < opts["tol_rel"]:
                                self.print0(f"Newton converged (relative tolerance)...")
                        if step_converged:
                            self.print0(f"Netwon converged (step tolerance {np.linalg.norm(dq) * alpha / max(np.linalg.norm(q), 1e-14)} < {opts.get('tol_step_rel', 1e-8)})...")
                    
                    # Update the reduced jacobian to latest state
                    J_rom = self._compute_rom_jacobian(fom_state=w, step=jac_fd_step)
                        
                    reason = 1 if ls_success else 2
                    break
                

            # Compute ROM Jacobian (only rebuild on successful steps)
            w = self._reconstruct_fom_state(q)
            if ls_success:  # ls_success from previous iteration
                if k % opts["update_test_basis_every"] == 0:
                    self._update_test_basis(fom_state=w, step=jac_fd_step)
                J_rom = self._compute_rom_jacobian(fom_state=w, step=jac_fd_step)

            # Linear solve
            try:
                dq = np.linalg.solve(J_rom, -r_rom)
            except np.linalg.LinAlgError:
                self.print0("Warning: ROM Jacobian is singular. Attempting least-squares solve.")
                dq, _, _, _ = np.linalg.lstsq(J_rom, -r_rom, rcond=None)

            # Line search
            alpha, q_trial, r_rom_trial, ls_success = self._line_search(q, dq, r_rom_norm, opts)

            # Accept step/minimum step
            q           = q_trial
            r_rom       = r_rom_trial
            r_rom_norm  = np.linalg.norm(r_rom)

            if verb['progress']:
                self._print_iteration(k+1, r_rom_norm, r_rom_norm_ref, alpha, dq, q, J_rom, verb, ls_success)

        else:
            self.print0(f"Warning: Newton solver reached max iterations ({opts['maxiter']}) without converging.")
            reason = -2

        # Cache and return
        self._cached_q   = q
        self._cached_w   = self._reconstruct_fom_state(q)
        self._cached_J_r = J_rom

        # if verb["footer"]:
        #     self._print_solve_footer(q, r_rom_norm, r_rom_norm_ref, reason, verb)

        self._print_diagnostics(q, dq, r_rom, r_rom_norm, r_rom_norm_ref, reason, opts)

        return q, reason
    

    # region _line_search
    def _line_search(self, q, dq, r_rom_norm, opts):
        # Armijo backtracking line search
        alpha       = opts["ls_alpha0"]
        ls_success  = False

        for ls_iter in range(opts["ls_maxiter"]):
            q_trial             = q + alpha * dq
            if not opts["ls_freeze_basis"]:
                w_trial             = self._reconstruct_fom_state(q_trial)
                self._update_test_basis(fom_state=w_trial, step=opts["jac_fd_step"])
            r_rom_trial         = self._eval_rom_residual(q_trial)
            r_rom_norm_trial    = np.linalg.norm(r_rom_trial)

            # Armijo condition
            if r_rom_norm_trial <= (1.0 - opts['ls_c1'] * alpha) * r_rom_norm:
                ls_success  = True
                break

            alpha *= opts['ls_rho']

        return alpha, q_trial, r_rom_trial, ls_success
        

    # region _reconstruct_fom_state
    def _reconstruct_fom_state(self, rom_state):
        q       = rom_state
        s       = self.scaling
        Phi     = self.pod_modes
        w_ref   = self.reference_state

        return w_ref + s * (Phi @ q)
    

    # region _eval_fom_residual
    def _eval_fom_residual(self, fom_state):
        # This assumes we have already set the inputs! Should generally be the case
        w = fom_state
        dafoam_instance = self.dafoam_instance
        
        w_old = dafoam_instance.getStates()
        dafoam_instance.setStates(w)
        residuals = dafoam_instance.getResiduals()
        dafoam_instance.setStates(w_old)

        if self.apply_temperature_residual_fix:
            residuals[self.state_indices["T"]] /= self.temperature_residual_cp_val

        # TODO: Assuming that the dafoam_instance always has normalizeResiduals for all states
        # Maybe add some logic to make this more robust for the case that this is not true?
        if not self.use_normalized_residuals:
            res_scale_factors = dafoam_instance.getStateWeights()
            # res_scale_factors[self.state_indices["T"]] /= 1005.
            residuals         = residuals * res_scale_factors

        return residuals


    # region _project_and_reduce
    def _project_and_reduce(self, distributed_val):
        comm = self.comm
        m    = self._m_eff
        Psi  = self.test_basis

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
        # TODO: Maybe make the 
        v_norm_local  = np.dot(v, v)
        v_norm_global = np.zeros(1)
        self.comm.Allreduce(v_norm_local, v_norm_global, op=MPI.SUM)
        v_norm = np.sqrt(v_norm_global[0])

        h = step * v_norm if v_norm > 0 else step

        # h = step
    
        # Use directional derivative fd approximation
        r0      = self._eval_fom_residual(fom_state=w) if fom_residual is None else fom_residual
        r_pert  = self._eval_fom_residual(fom_state=w + h * v)

        # Return the dafoam states back to original state
        # We'd set this to false in the case of using _jac_mad_product
        if reset_state:
            self.dafoam_instance.setStates(w)

        return (r_pert - r0) / h

    
    # region _jacT_vec_product
    def _jacT_vec_product(self, fom_state, vec):
        dafoam_instance = self.dafoam_instance
        v = vec
        w = fom_state

        seed    = np.ascontiguousarray(v)
        product = np.zeros_like(seed)
        
        # Prescale (for temperature residual fix)
        if self.apply_temperature_residual_fix:
            product[self.state_indices["T"]] /= self.temperature_residual_cp_val

        dafoam_instance.solverAD.calcJacTVecProduct(
            'dafoam_solver_states',
            "stateVar",
            w,
            'aero_residuals',
            "residual",
            seed,
            product,
            )
        
        # TODO: Assuming that the dafoam_instance always has normalizeResiduals for all states
        # Maybe add some logic to make this more robust for the case that this is not true?
        if not self.use_normalized_residuals:
            res_scale_factors = dafoam_instance.getStateWeights()
            product *= res_scale_factors
        
        return product / dafoam_instance.getStateScalingFactors()
    

    # region _jacT_mat_product
    def _jacT_mat_product(self, fom_state, matrix):
        M = matrix
        w = fom_state

        JT_M = np.zeros_like(M)

        for i in range(M.shape[1]):
            v           = M[:, i]
            JT_M[:, i]  = self._jacT_vec_product(fom_state=w, vec=v)

        return JT_M


    # region _update_test_basis
    def _update_test_basis(self, fom_state, step=1e-6):
        Phi = self.pod_modes
        s   = self.scaling

        if self.rom_type == "galerkin":
            self.test_basis = s[:, None] * Phi

        elif self.rom_type == 'lspg':
            w   = fom_state
            self.test_basis = self._jac_mat_product(fom_state=w, matrix=s[:, None] * Phi, step=step)
   

    # region _eval_rom_residual
    def _eval_rom_residual(self, rom_state):
        assert self.test_basis is not None, \
            "test_basis is None. Call _update_test_basis before _eval_rom_residual"
        q = rom_state
        w = self._reconstruct_fom_state(q)
        r = self._eval_fom_residual(w)

        return self._project_and_reduce(distributed_val=r)
        

    # region _compute_rom_jacobian_fd
    def _compute_rom_jacobian(self, fom_state, step=1e-6):
        Psi = self.test_basis
        Phi = self.pod_modes
        m   = self._m_eff
        s   = self.scaling
        w   = fom_state

        if self.jac_mode == "fd":
            if self.rom_type == "lspg":
                J_rom = self._project_and_reduce(Psi)

            elif self.rom_type == "galerkin":
                # _project_and_reduce already handles the reduction
                J_rom = self._project_and_reduce(self._jac_mat_product(fom_state=w, matrix=s[:, None] * Phi, step=step))

        elif self.jac_mode == "analytical":
            if self.rom_type == "lspg":
                raise NotImplemented("Analytical Jacobian products are not supported for Least Squares Petrov-Galerkin.")
            
            JT_SMT_Phi  = self._jacT_mat_product(fom_state=w, matrix=(s * m)[:, None] * Phi)
            J_rom_local = JT_SMT_Phi.T @ (s[:, None] * Phi)

            J_rom = np.zeros_like(J_rom_local)
            self.comm.Allreduce(J_rom_local, J_rom, op=MPI.SUM)

        return J_rom


    
        

    # region _set_as_constant_or_input
    # Function with logic for determining if we have constant or variable inputs
    def _set_as_constant_or_input(self, variable, name):
        # Set as an input variable if not already set to a constant
        if variable is not None:
            if getattr(self, name) is not None:
                self.print0(f'{name} passed to DAFoamROM evalute method. Neglecting constant/global value passed in initialization.')
            self.declare_input(name, variable)
            if name == "pod_modes":
                num_modes = variable.value.shape[1]
            setattr(self, f"{name}_is_csdl_var", True)
        else:
            if getattr(self, name) is None:
                raise TypeError(f'{name} not assigned to ROM. Please pass {name} during initialization or evaluation.')
            else:
                if name == "pod_modes":
                    num_modes = getattr(self, name).shape[1]
                setattr(self, f"{name}_is_csdl_var", False)
        
        if name == "pod_modes":
            return num_modes
        

    # region _check_if_in_list_and_return_if_so
    def _check_if_in_list(self, name, list):
        if name.lower() in list:
            return name.lower()
        else:
            raise ValueError(f"{name} is not a valid option. Please choose from {list}")


    # region _init_newton_options
    def _init_newton_options(self, user_options):
        DEFAULT_NEWTON_OPTIONS = {
            'maxiter':                  50,
            'tol_rel':                  1e-6,
            'tol_abs':                  1e-10,
            'tol_step_rel':             1e-8,
            'ls_alpha0':                1.0,
            'ls_rho':                   0.5,
            'ls_c1':                    1e-4,
            'ls_maxiter':               10,
            'ls_freeze_basis':          True,
            'jac_fd_step':              1e-6,
            'update_test_basis_every':  1,
            'verbose':                  1,
            'min_newton_steps':         1,
        }
        opts            = {**DEFAULT_NEWTON_OPTIONS, **(user_options or {})}

        # Resolve the verbosity, if user supplied an integer
        if isinstance(opts["verbose"], int):
            level = opts["verbose"]
            opts["verbose"] = {
                                'level':           level,
                                'progress':        level >= 1,
                                'header':          level >= 1,
                                'footer':          level >= 1,
                                'numerics':        level >= 2,
                                'residuals':       level >= 2,
                                'basis_quality':   level >= 2,
                                'basis_expensive': level >= 3,
                            }
        return opts


    # region print0
    def print0(self, statement, **kwargs):
        if self.rank == 0:
            print(statement, **kwargs)


    #region _print_solve_header
    def _print_solve_header(self, q0, w_ref, r_rom_norm_ref, r_fom_norm_ref, opts):
        verb = opts["verbose"]
        W    = 68

        self.print0(f"\n{'-'*W}")
        self.print0(
            f"  DAFoamROM Newton Solve  "
            f"[{self.rom_type.upper()} | jac: {self.jac_mode} | "
            f"n_modes: {self.basis_size}]"
        )
        self.print0(f"{'-'*W}")
        self.print0(f"  {'tol_rel / tol_abs':<28} "
                    f"{opts['tol_rel']:.1e} / {opts['tol_abs']:.1e}")
        self.print0(f"  {'fd step':<28} {opts['jac_fd_step']:.2e}")
        self.print0(f"  {'‖r_fom‖ (at q=0)':<28} {r_fom_norm_ref:.6e}")
        self.print0(f"  {'‖r_rom‖ (at q=0, normalizer)':<28} {r_rom_norm_ref:.6e}")
        self.print0(f"{'-'*W}")

        # Column headers
        self.print0(
            f"  {'Iter':>4}  {'‖r_rom‖':>12}  {'‖r_rom‖/‖r_ref‖':>16}  "
            f"{'alpha':>8}  {'‖dq‖':>10}  {'‖q‖':>10}"
            + (f"  {'cond(J)':>10}" if verb.get('numerics') else "")
        )
        self.print0(
            f"  {'-'*4}  {'-'*12}  {'-'*16}  "
            f"{'-'*8}  {'-'*10}  {'-'*10}"
            + (f"  {'-'*10}" if verb.get('numerics') else "")
        )
        self.print0(
            f"  {0:>4}  {np.linalg.norm(r_rom_norm_ref):>12.6e}  {1.0:>16.6e}  "
            f"{'-':>8}  {'-':>10}  {np.linalg.norm(q0):>10.4e}"
            + (f"  {'-':>10}" if verb.get('numerics') else "")
        )

    
    # region _print_iteration
    def _print_iteration(self, k, r_rom_norm, r_rom_norm_ref, alpha, dq, q, J_rom, verb, ls_success):
        ls_flag  = "" if ls_success else " <- Line search failed!"
        cond_str = ""

        if verb.get('numerics') and J_rom is not None:
            cond_str = f"  {np.linalg.cond(J_rom):>10.3e}"

        self.print0(
            f"  {k:>4}  {r_rom_norm:>12.6e}  "
            f"{r_rom_norm/max(r_rom_norm_ref,1e-14):>16.6e}  "
            f"{alpha:>8.2e}  {np.linalg.norm(dq):>10.4e}  "
            f"{np.linalg.norm(q):>10.4e}"
            + cond_str
            + ls_flag
        )


    # region _check_basis_orthogonality
    def _check_basis_orthogonality(self):
        """
        Always called at solver initialization, regardless of verbosity.
        Checks Phi^T M Phi = I under the POD inner product weight.
        Prints one line: PASS or WARNING.
        """
        Phi = self.pod_modes
        m   = self.weights        # M inner product weights (not M_eff)
        n   = self.basis_size

        # Gram matrix: G_ij = Phi_i^T M Phi_j
        G_local = Phi.T @ (m[:, None] * Phi)
        G       = np.zeros_like(G_local)
        self.comm.Allreduce(G_local, G, op=MPI.SUM)

        ortho_error = np.linalg.norm(G - np.eye(n), 'fro')
        tol         = 1e-10

        if ortho_error < tol:
            self.print0(
                f"  Basis orthogonality check (‖Phi^T M Phi - I‖_F):  "
                f"PASS ({ortho_error:.2e})"
            )
        else:
            self.print0(
                f"  Basis orthogonality check (‖Phi^T M Phi - I‖_F):  "
                f"WARNING ({ortho_error:.2e}) -- downstream math may be affected"
            )


    # region _print_diagnostics
    def _print_diagnostics(self, q, dq, r_rom, r_rom_norm, r_rom_norm_ref, reason, opts):
        """
        Orchestrator for all diagnostic output. Calls each section
        conditionally based on verbosity level.

        Verbosity structure:
            level >= 1: footer   (exit reason, r_rom ratio, q norm, SA flag)
            level >= 2: residuals, numerics, basis_quality
            level >= 3: basis_expensive (mode contributions)
        """
        verb = opts["verbose"]

        if verb.get("footer"):
            self._print_solve_footer(q, r_rom_norm, r_rom_norm_ref, reason, opts)

        if verb.get("residuals"):
            self._print_residuals(opts)

        if verb.get("numerics"):
            self._print_numerics(q, dq, r_rom, r_rom_norm, opts)

        if verb.get("basis_quality"):
            self._print_basis_quality(opts)

        if verb.get("basis_expensive"):
            self._print_mode_contributions(q, opts)


    # region _print_solve_footer
    def _print_solve_footer(self, q, r_rom_norm, r_rom_norm_ref, reason, opts):
        """
        Level 1 footer. Always cheap — no FOM residual evaluations.
        Uses self._cached_w for SA check.
        """
        W          = 68
        reason_str = {
            1:  "converged",
            2:  "converged with failed line search",
            -2: "max iterations",
            0:  "unknown"
        }.get(reason, "unknown")

        w_final     = self._cached_w
        sa_negative = (
            bool(self.comm.allreduce(
                int((w_final[self.state_indices["nuTilda"]] < 0).any()), op=MPI.MAX
            ))
            if "nuTilda" in self.state_indices
            else False
        )

        self.print0(f"\n{'-'*W}")
        self.print0(f"  {'Exit reason':<32} {reason_str}")
        self.print0(f"  {'‖r_rom‖/‖r_rom_ref‖':<32} {r_rom_norm/max(r_rom_norm_ref, 1e-14):.6e}")
        self.print0(f"  {'‖q‖':<32} {np.linalg.norm(q):.6e}")
        self.print0(f"  {'SA var negative?':<32} {'**YES**' if sa_negative else 'no'}")
        self.print0(f"{'-'*W}")


    # region _print_residuals
    def _print_residuals(self, opts):
        """
        Level 2 residuals. Evaluates FOM residual at both w_final and w_ref.
        Prints global norms, ratio, and per-variable breakdown.
        Flags variables where ROM solution is worse than reference (**).
        """
        W = 68

        w_final = self._cached_w

        # Evaluate FOM residuals
        r_fom_final = self._eval_fom_residual(w_final)
        r_fom_ref   = self._eval_fom_residual(self.reference_state)

        # Global norms
        r_fom_final_norm = np.sqrt(self.comm.allreduce(np.dot(r_fom_final, r_fom_final), op=MPI.SUM))
        r_fom_ref_norm   = np.sqrt(self.comm.allreduce(np.dot(r_fom_ref,   r_fom_ref),   op=MPI.SUM))
        ratio_global     = r_fom_final_norm / max(r_fom_ref_norm, 1e-14)

        # Warn if reference residual is near zero
        if r_fom_ref_norm < 1e-14:
            self.print0(
                f"  WARNING: ‖r_fom_ref‖ near zero -- "
                f"design point may match reference condition"
            )

        self.print0(f"\n{'-'*W}")
        self.print0(f"  FOM Residuals")
        self.print0(f"{'-'*W}")

        # Per-variable table
        self.print0(f"\n  {'Variable':<16} {'‖r_ref‖':>14}  {'‖r_final‖':>14}  {'ratio':>10}")
        self.print0(f"  {'-'*16} {'-'*14}  {'-'*14}  {'-'*10}")

        for var_name, indices in self.state_indices.items():
            r_ref_var   = r_fom_ref[indices]
            r_final_var = r_fom_final[indices]

            r_ref_norm   = np.sqrt(self.comm.allreduce(np.dot(r_ref_var,   r_ref_var),   op=MPI.SUM))
            r_final_norm = np.sqrt(self.comm.allreduce(np.dot(r_final_var, r_final_var), op=MPI.SUM))
            ratio        = r_final_norm / max(r_ref_norm, 1e-14)
            flag         = "  **" if ratio > 1.0 else ""

            self.print0(
                f"  {var_name:<16} {r_ref_norm:>14.6e}  {r_final_norm:>14.6e}  {ratio:>10.4e}{flag}"
            )

        self.print0(
            f"  {'TOTAL':<16} {r_fom_ref_norm:>14.6e}  "
            f"{r_fom_final_norm:>14.6e}  {ratio_global:>10.4e}"
        )
        self.print0(f"{'-'*W}")


    # region _print_numerics
    def _print_numerics(self, q, dq, r_rom, r_rom_norm, opts):
        """
        Level 2 numerics. SVD summary of final Jacobian + FD accuracy check.
        FD accuracy poor => warning only, not a convergence criterion.
        """
        W     = 68
        J_rom = self._cached_J_r

        if J_rom is None:
            self.print0(f"  WARNING: No cached Jacobian available for numerics summary")
            return

        # SVD
        svd_vals  = np.linalg.svd(J_rom, compute_uv=False)
        sigma_max = svd_vals[0]
        sigma_min = svd_vals[-1]
        cond      = sigma_max / max(sigma_min, 1e-14)
        eff_rank  = int(np.sum(svd_vals > sigma_max * 1e-10))

        self.print0(f"\n{'-'*W}")
        self.print0(f"  Jacobian Summary (final iteration)")
        self.print0(f"{'-'*W}")
        self.print0(f"  {'cond(J_rom)':<36} {cond:.4e}")
        self.print0(f"  {'sigma_max / sigma_min':<36} {sigma_max:.4e} / {sigma_min:.4e}")
        self.print0(f"  {'effective rank':<36} {eff_rank} / {len(svd_vals)}")
        self.print0(f"  {'‖J_rom‖ (Frobenius)':<36} {np.linalg.norm(J_rom, 'fro'):.4e}")

        # Always print 5 smallest singular values - most diagnostic for ill-conditioning
        n_small = min(5, len(svd_vals))
        self.print0(f"  Smallest {n_small} singular values:")
        for i, v in enumerate(svd_vals[-n_small:][::-1]):
            self.print0(f"    s_{len(svd_vals)-1-i:02d} = {v:.5e}")

        # Adjoint check
        self.print0(f"\n  Quick Consistency Check (should be small values)")
        self.print0(f"  {'-'*40}")
        self.print0(f"  {'||J dq + r||':<36} {np.linalg.norm(J_rom.dot(dq) + r_rom):.4e}")

        u = np.random.randn(J_rom.shape[1])
        v = np.random.randn(J_rom.shape[0])
        lhs = v @ (J_rom.dot(u))
        rhs = u @ (J_rom.T.dot(v))
        self.print0(f"  {'||lhs - rhs|| / ||rhs||':<36} {abs(lhs - rhs)/abs(rhs)}")
        self.print0(f"  {'Where':<36}")
        self.print0(f"  {'    lhs = v^T J u':<36} {lhs}")
        self.print0(f"  {'    rhs = u^T J^T v':<36} {rhs}")

        # FD accuracy check
        self.print0(f"\n  FD Accuracy Check")
        self.print0(f"  {'-'*40}")

        rng      = np.random.default_rng(seed=42)
        dq_test  = rng.standard_normal(len(q))
        dq_test /= max(np.linalg.norm(dq_test), 1e-14)
        eps      = opts.get('jac_fd_step', 1e-6) * max(r_rom_norm, 1.0)

        dr_pred   = J_rom @ (eps * dq_test)
        r_pert    = self._eval_rom_residual(q + eps * dq_test)
        dr_actual = r_pert - r_rom

        abs_err = np.linalg.norm(dr_pred - dr_actual)
        rel_err = abs_err / max(np.linalg.norm(dr_actual), 1e-14)

        self.print0(f"  {'‖J dq‖ (predicted)':<36} {np.linalg.norm(dr_pred):.4e}")
        self.print0(f"  {'‖Δr_rom‖ (actual)':<36} {np.linalg.norm(dr_actual):.4e}")
        self.print0(f"  {'absolute error':<36} {abs_err:.4e}")
        self.print0(f"  {'relative error':<36} {rel_err:.4e}")

        if rel_err < 1e-2:
            self.print0(f"  {'Jacobian quality':<36} good")
        elif rel_err < 1e-1:
            self.print0(f"  {'Jacobian quality':<36} acceptable")
        else:
            self.print0(f"  {'Jacobian quality':<36} poor  ** consider reducing jac_fd_step **")

        self.print0(f"{'-'*W}")


    # region _print_basis_quality
    def _print_basis_quality(self, opts):
        """
        Level 2 basis quality.
        - cond(G): condition of Gram matrix Psi^T M_eff Psi
        - Residual capture: fraction of r_fom lying in span(Psi) under M_eff norm
        - Per-variable reconstruction: actual (using converged q) vs best-fit
            (independent q_v per variable), plus gap between them
        Note: projection uses full Gram solve, not assuming Psi is orthonormal.
        """
        W       = 68
        Psi     = self.test_basis
        m       = self._m_eff
        Phi     = self.pod_modes
        S       = self.scaling
        w_final = self._cached_w
        q       = self._cached_q

        # --- Gram matrix G = Psi^T M_eff Psi ---
        G_local = Psi.T @ (m[:, None] * Psi)
        G       = np.zeros_like(G_local)
        self.comm.Allreduce(G_local, G, op=MPI.SUM)

        cond_G = np.linalg.cond(G)

        # --- Residual capture ---
        r_fom  = self._eval_fom_residual(w_final)
        rhs    = self._project_and_reduce(r_fom)    # Psi^T M_eff r_fom
        c      = np.linalg.solve(G, rhs)            # (Psi^T M_eff Psi)^{-1} Psi^T M_eff r_fom
        r_proj = Psi @ c                            # projected residual

        r_orth = r_fom - r_proj

        r_fom_norm_sq  = self.comm.allreduce(np.dot(r_fom,  m * r_fom),  op=MPI.SUM)
        r_orth_norm_sq = self.comm.allreduce(np.dot(r_orth, m * r_orth), op=MPI.SUM)

        r_fom_M_norm  = np.sqrt(max(r_fom_norm_sq,  0.0))
        r_orth_M_norm = np.sqrt(max(r_orth_norm_sq, 0.0))
        capture       = 1.0 - r_orth_M_norm / max(r_fom_M_norm, 1e-14)

        self.print0(f"\n{'-'*W}")
        self.print0(f"  Basis Quality")
        self.print0(f"{'-'*W}")
        self.print0(f"  {'cond(G)  (Psi^T M_eff Psi)':<36} {cond_G:.4e}")
        self.print0(f"  {'residual capture':<36} {capture*100:.3f}%")

        # --- Per-variable reconstruction table ---
        self.print0(f"\n  {'Variable':<16} {'e_actual':>12}  {'e_bestfit':>12}  {'gap':>12}")
        self.print0(f"  {'-'*16} {'-'*12}  {'-'*12}  {'-'*12}")

        w_delta = w_final - self.reference_state
        SPhi    = S[:, None] * Phi      # S Phi, full field

        for var_name, indices in self.state_indices.items():
            w_v    = w_delta[indices]
            SPhi_v = SPhi[indices, :]   # S Phi restricted to variable v

            # Global norm of w_v
            w_v_norm_sq = self.comm.allreduce(np.dot(w_v, w_v), op=MPI.SUM)
            w_v_norm    = np.sqrt(max(w_v_norm_sq, 0.0))

            # --- Actual reconstruction error (using converged q) ---
            w_v_recon_actual = SPhi_v @ q
            err_actual_sq    = self.comm.allreduce(
                np.dot(w_v - w_v_recon_actual, w_v - w_v_recon_actual), op=MPI.SUM
            )
            e_actual = np.sqrt(max(err_actual_sq, 0.0)) / max(w_v_norm, 1e-14)

            # --- Best-fit reconstruction error (independent q_v per variable) ---
            G_v_local   = SPhi_v.T @ SPhi_v
            G_v         = np.zeros_like(G_v_local)
            self.comm.Allreduce(G_v_local, G_v, op=MPI.SUM)

            rhs_v_local = SPhi_v.T @ w_v
            rhs_v       = np.zeros_like(rhs_v_local)
            self.comm.Allreduce(rhs_v_local, rhs_v, op=MPI.SUM)

            q_v          = np.linalg.solve(G_v, rhs_v)
            w_v_recon_bf = SPhi_v @ q_v
            err_bf_sq    = self.comm.allreduce(
                np.dot(w_v - w_v_recon_bf, w_v - w_v_recon_bf), op=MPI.SUM
            )
            e_bestfit = np.sqrt(max(err_bf_sq, 0.0)) / max(w_v_norm, 1e-14)

            gap = e_actual - e_bestfit

            self.print0(
                f"  {var_name:<16} {e_actual:>12.4e}  {e_bestfit:>12.4e}  {gap:>12.4e}"
            )

        self.print0(f"{'-'*W}")


    # region _print_mode_contributions
    def _print_mode_contributions(self, q, opts):
        """
        Level 3 mode contributions.
        Decoupled residual projection coefficients: c = G^{-1} Psi^T M_eff r_fom
        Printed alongside converged q_i and ratio |c_i|/|q_i|.
        Computes r_fom fresh (independent of level 2).
        """
        W       = 68
        Psi     = self.test_basis
        m       = self._m_eff
        w_final = self._cached_w

        # Evaluate FOM residual fresh
        r_fom = self._eval_fom_residual(w_final)

        # Gram matrix
        G_local = Psi.T @ (m[:, None] * Psi)
        G       = np.zeros_like(G_local)
        self.comm.Allreduce(G_local, G, op=MPI.SUM)

        # Decoupled mode contributions: c = G^{-1} Psi^T M_eff r_fom
        rhs = self._project_and_reduce(r_fom)   # Psi^T M_eff r_fom
        c   = np.linalg.solve(G, rhs)           # decoupled coefficients

        self.print0(f"\n{'-'*W}")
        self.print0(f"  Mode Contributions")
        self.print0(f"{'-'*W}")
        self.print0(
            f"  {'Mode':>6}  {'|c_i| (decoupled)':>18}  {'q_i':>12}  {'|c_i|/|q_i|':>14}"
        )
        self.print0(
            f"  {'-'*6}  {'-'*18}  {'-'*12}  {'-'*14}"
        )

        for i in range(self.basis_size):
            abs_ci    = abs(c[i])
            qi        = q[i]
            abs_qi    = abs(qi)
            ratio_str = f"{abs_ci/abs_qi:>14.4e}" if abs_qi > 1e-14 else f"{'--':>14}"

            self.print0(
                f"  {i+1:>6}  {abs_ci:>18.4e}  {qi:>12.4e}  {ratio_str}"
            )

        self.print0(f"{'-'*W}")