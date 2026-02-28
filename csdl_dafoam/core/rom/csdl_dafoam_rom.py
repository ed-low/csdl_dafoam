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
#                         'ls_alpha0':        1.0,      # initial line search step
#                         'ls_rho':           0.5,      # backtrack factor
#                         'ls_c1':            1e-4,     # Armijo sufficient decrease constant
#                         'ls_maxiter':       10,       # max line search iterations
#                         'jac_fd_step':      1e-6,     # FD step for Jacobian
#                         'update_test_basis_every': 1, # how often to recompute Psi (every n iters)
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
        
        # Merge user newton options, then resolve verbose once
        self.newton_options             = self._init_newton_options(newton_options)

        self.exclude_from_projection    = exclude_from_projection

        # These are used in the evaluate section, but initialized here
        self.pod_modes_is_csdl_var       = None
        self.reference_state_is_csdl_var = None
        self.scaling_is_csdl_var         = None

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
        s_r                     = np.ones_like(self.reference_state) if self.residual_scaling is None   else self.residual_scaling
        if self.rom_type == 'galerkin':
            self._m_eff = self.weights / s_r           # M R^{-1}
        elif self.rom_type == 'lspg':
            self._m_eff = self.weights / (s_r ** 2)    # R^{-1} M R^{-1}

        # Zero out excluded variables in m_eff (e.g. T for incompressible SA)
        if self.exclude_from_projection:
            var_names, var_idx = self.dafoam_instance.getStateVariableMap(includeComponentSuffix=False)
            for exclude_var in self.exclude_from_projection:
                if exclude_var not in var_names:
                    self.print0(f"Warning: '{exclude_var}' not found in state variable map. Skipping.")
                    continue
                local_mask           = var_idx == var_names.index(exclude_var)
                self._m_eff[local_mask] = 0.0

        # Make sure solver is updated with the most recent input values
        dafoam_instance.set_solver_input(input_vals)

        # Initial guess: warm start from previous solution if available, else zero
        # TODO: See if there is a best warm start option
        q0 = self._cached_q.copy() if self._cached_q is not None else np.zeros(self.basis_size)

        q, reason = self._rom_newtwon_solve(initial_rom_state=q0)

        if reason == 1:
            output_vals["dafoam_rom_states"] = q

        elif reason == -1 or reason == -2:
            self.print0("Newton solver failed. Setting DAFoam ROM states to NaNs.")
            output_vals["dafoam_rom_states"] = np.nan * np.ones_like(q)


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
        self.dafoam_instance.solver.writeAdjointFields("solution", 1 + (self.solution_iter + 1) / 10000, self._cached_w, True)
        mesh = np.zeros_like(self.dafoam_instance.xv.flatten())
        self.dafoam_instance.solver.getOFMeshPoints(mesh)
        self.dafoam_instance.solver.writeMeshPoints(mesh, 1 + (self.solution_iter + 1) / 10000)
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
            seed = np.ascontiguousarray(m * (Psi @ lam))

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
                    seed,
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
        
    
    # region _rom_newtwon_solve
    def _rom_newtwon_solve(self, initial_rom_state):
        opts            = self.newton_options
        verb            = opts["verbose"]

        # Initialize some values
        q               = initial_rom_state.copy()
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
            if r_rom_norm < opts["tol_abs"] or r_rom_norm / r_rom_norm_ref < opts["tol_rel"]:
                if verb['progress']:
                    self.print0(f"Newton converged (absolute tolerance) at iteration {k}")
                
                if verb['progress']:
                    self.print0(f"Newton converged (relative tolerance) at iteration {k}")
                
                # Compute the reduced jacobian if we converged before running anything
                if k == 0:
                    J_rom = self._compute_rom_jacobian(fom_state=w, step=jac_fd_step)
                    
                reason = 1 if ls_success else 2
                break

            # Compute ROM Jacobian
            w = self._reconstruct_fom_state(q)
            if k % opts["update_test_basis_every"] == 0:
                self._update_test_basis(fom_state=w, step=jac_fd_step)
            J_rom = self._compute_rom_jacobian(fom_state=w, step=jac_fd_step)

            # Linear solve
            # We'll do this on all ranks, redundantly for now (consider solve on root->broadcast, parallelize)
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

        if verb["footer"]:
            self._print_solve_footer(q, r_rom_norm, r_rom_norm_ref, reason, verb)

        # Cache and return
        self._cached_q   = q
        self._cached_w   = self._reconstruct_fom_state(q)
        self._cached_J_r = J_rom

        return q, reason
    

    # region _line_search
    def _line_search(self, q, dq, r_rom_norm, opts):
        # Armijo backtracking line search
        alpha       = opts["ls_alpha0"]
        ls_success  = False

        for ls_iter in range(opts["ls_maxiter"]):
            q_trial             = q + alpha * dq
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
        dafoam_instance.setStates(w)
        return dafoam_instance.getResiduals()


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
            JM[:, i] = self._jac_vec_product(fom_state=w, direction=v, fom_residual=r0, step=step)

        return JM


    # region _jac_vec_product
    def _jac_vec_product(self, fom_state, direction, fom_residual=None, step=1e-6):
        w = fom_state
        v = direction

        # Scale h relative to the direction magnitude to avoid truncation/cancellation
        v_norm_local  = np.dot(v, v)
        v_norm_global = np.zeros(1)
        self.comm.Allreduce(v_norm_local, v_norm_global, op=MPI.SUM)
        v_norm = np.sqrt(v_norm_global[0])

        h = step * v_norm if v_norm > 0 else step
    
        # Use directional derivative fd approximation
        r0      = self._eval_fom_residual(fom_state=w) if fom_residual is None else fom_residual
        r_pert  = self._eval_fom_residual(fom_state=w + h * v)

        return (r_pert - r0) / h

    
    # region _jacT_vec_product
    def _jacT_vec_product(self, fom_state, vec):
        dafoam_instance = self.dafoam_instance
        v = vec
        w = fom_state

        seed    = np.ascontiguousarray(v)
        product = np.zeros_like(seed)
        dafoam_instance.solverAD.calcJacTVecProduct(
            'dafoam_solver_states',
            "stateVar",
            w,
            'aero_residuals',
            "residual",
            seed,
            product,
            )
        
        return product
    

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
            'ls_alpha0':                1.0,
            'ls_rho':                   0.5,
            'ls_c1':                    1e-4,
            'ls_maxiter':               10,
            'jac_fd_step':              1e-6,
            'update_test_basis_every':  1,
            'verbose':                  1,
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
            f"  {0:>4}  {np.linalg.norm(q0):>12.6e}  {1.0:>16.6e}  "
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


    # region _print_solve_footer
    def _print_solve_footer(self, q, r_rom_norm, r_rom_norm_ref, reason, verb):
        W          = 68 # Divider width
        reason_str = {
            1:  "converged",
            2: "converged with failed line search",
            -1: "max iterations",
            0:  "unknown"
        }.get(reason, "unknown")

        # Level 1: always printed in footer
        w_final     = self._reconstruct_fom_state(q)
        sa_negative = (
                        bool(self.comm.allreduce(int((w_final[self.state_indices["nuTilda"]] < 0).any()), op=MPI.MAX))
                        if "nuTilda" in self.state_indices
                        else False
                    )

        self.print0(f"{'-'*W}")
        self.print0(f"  {'Exit reason':<32} {reason_str}")
        self.print0(f"  {'‖r_rom‖ (final)':<32} {r_rom_norm:.6e}")
        self.print0(f"  {'‖r_rom‖/‖r_ref‖ (final)':<32} "
                    f"{r_rom_norm/max(r_rom_norm_ref,1e-14):.6e}")
        self.print0(f"  {'‖q‖ (final)':<32} {np.linalg.norm(q):.6e}")
        self.print0(f"  {'SA var negative?':<32} {'**YES**' if sa_negative else 'no'}")

        # Level 2: residuals category
        if verb.get('residuals'):
            r_fom_final      = self._eval_fom_residual(w_final)
            r_fom_norm_final = np.sqrt(self.comm.allreduce(np.dot(r_fom_final, r_fom_final), op=MPI.SUM))
            self.print0(f"  {'-'*66}")
            self.print0(f"  {'‖r_fom‖ (final)':<32} {r_fom_norm_final:.6e}")
            self._print_per_variable_residuals(r_fom_final)

        # Level 2: basis quality category
        if verb.get('basis_quality'):
            self._print_basis_quality(q, w_final, r_fom=r_fom_final if verb.get('residuals') else None)

        if verb.get('numerics') and self._cached_J_r is not None:
            r_rom      = self._eval_rom_residual(q)
            self._print_jacobian_summary(
                J_rom      = self._cached_J_r,
                q          = q,
                r_rom      = r_rom,
                r_rom_norm = np.linalg.norm(r_rom),
                opts       = self.newton_options,
            )

        # Level 3: mode contributions
        if verb.get('basis_expensive'):
            w_final = w_final if w_final is not None else self._reconstruct_fom_state(q)
            self._print_mode_contributions(q, w_final)
            self._print_orthogonality_check()

        self.print0(f"{'-'*W}\n")


    # region _print_per_variable_residuals
    def _print_per_variable_residuals(self, r_fom):
        # Level 2 output - residuals category
        r_norm_total  = np.sqrt(self.comm.allreduce(np.dot(r_fom, r_fom), op=MPI.SUM))

        self.print0(f"  {'Variable':<16} {'‖r_var‖':>14}  {'fraction':>10}")
        self.print0(f"  {'-'*16} {'-'*14}  {'-'*10}")

        for var_name, indices in self.state_indices.items():
            r_var       = r_fom[indices]
            r_var_norm  = np.sqrt(self.comm.allreduce(np.dot(r_var, r_var), op=MPI.SUM))
            fraction    = r_var_norm / max(r_norm_total, 1e-14)
            self.print0(f"  {var_name:<16} {r_var_norm:>14.6e}  {fraction:>10.4f}")

    
    # region _print_basis_quality
    def _print_basis_quality(self, q, w_final, r_fom=None):
        # Level 2 output - basis quality category
        r_fom      = r_fom if r_fom is not None else self._eval_fom_residual(w_final)
        r_fom_norm = np.sqrt(self.comm.allreduce(np.dot(r_fom, r_fom), op=MPI.SUM))
        r_proj     = self._project_and_reduce(r_fom)
        r_proj_norm = np.linalg.norm(r_proj)
        capture    = r_proj_norm / max(r_fom_norm, 1e-14)

        self.print0(f"  {'-'*66}")
        self.print0(f"  {'‖r_fom‖ captured by Psi':<32} {capture*100:.1f}%")
        self.print0(f"  {'‖q‖ (final)':<32} {np.linalg.norm(q):.6e}")

        if hasattr(self, 'q_max_training') and self.q_max_training is not None:
            ratio = np.linalg.norm(q) / max(self.q_max_training, 1e-14)
            self.print0(f"  {'‖q‖/‖q‖_train_max':<32} {ratio:.4f}")


    # region _print_jacobian_summary
    def _print_jacobian_summary(self, J_rom, q=None, r_rom=None, r_rom_norm=None, opts=None):
        # Level 2 output - numerics category.
        W = 56
        self.print0(f"  {'-'*W}")
        self.print0(f"  Jacobian Summary")
        self.print0(f"  {'-'*W}")

        # --- SVD-based quantities (always computed here) ---
        svd_vals = np.linalg.svd(J_rom, compute_uv=False)
        sigma_max = svd_vals[0]
        sigma_min = svd_vals[-1]
        cond      = sigma_max / max(sigma_min, 1e-14)
        eff_rank  = np.sum(svd_vals > sigma_max * 1e-10)

        self.print0(f"  {'cond(J_rom)':<36} {cond:.4e}")
        self.print0(f"  {'sigma_max / sigma_min':<36} {sigma_max:.4e} / {sigma_min:.4e}")
        self.print0(f"  {'effective rank':<36} {eff_rank} / {len(svd_vals)}")
        self.print0(f"  {'‖J_rom‖ (Frobenius)':<36} {np.linalg.norm(J_rom, 'fro'):.4e}")

        # Estimate ‖J_rom^{-1}‖ via smallest singular value
        J_inv_norm_est = 1.0 / max(sigma_min, 1e-14)
        self.print0(f"  {'‖J_rom^(-1)‖ (est. via sigma_min)':<36} {J_inv_norm_est:.4e}")

        # Print all singular values if basis is small enough to be readable
        if len(svd_vals) <= 30:
            self.print0(f"  Singular values:")
            for i, v in enumerate(svd_vals):
                self.print0(f"  s{i:02} = {v:.5e}")

        # --- FD Jacobian accuracy check ---
        # Perturb q in a random direction, compare predicted vs actual r_rom change
        # Cost: one _eval_rom_residual call
        if all(x is not None for x in [q, r_rom, r_rom_norm, opts]):
            self.print0(f"  {'-'*W}")
            self.print0(f"  Jacobian FD accuracy check:")

            rng      = np.random.default_rng(seed=42)    # fixed seed for reproducibility
            dq_test  = rng.standard_normal(len(q))
            dq_test /= max(np.linalg.norm(dq_test), 1e-14)
            eps      = opts.get('jac_fd_step', 1e-6) * max(r_rom_norm, 1.0)

            # Predicted change via J_rom
            dr_pred  = J_rom @ (eps * dq_test)

            # Actual change via residual evaluation
            r_pert   = self._eval_rom_residual(q + eps * dq_test)
            dr_actual = r_pert - r_rom

            pred_norm   = np.linalg.norm(dr_pred)
            actual_norm = np.linalg.norm(dr_actual)
            abs_err     = np.linalg.norm(dr_pred - dr_actual)
            rel_err     = abs_err / max(actual_norm, 1e-14)

            self.print0(f"  {'‖J dq‖ (predicted)':<36} {pred_norm:.4e}")
            self.print0(f"  {'‖Δr_rom‖ (actual)':<36} {actual_norm:.4e}")
            self.print0(f"  {'absolute error':<36} {abs_err:.4e}")
            self.print0(f"  {'relative error':<36} {rel_err:.4e}")

            if rel_err < 1e-2:
                self.print0(f"  {'Jacobian quality':<36} good")
            elif rel_err < 1e-1:
                self.print0(f"  {'Jacobian quality':<36} acceptable")
            else:
                self.print0(f"  {'Jacobian quality':<36} poor ⚠  (consider reducing jac_fd_step)")
        

    # region _print_mode_contributions
    def _print_mode_contributions(self, q, w_final):
        #Level 3 output - which modes contribute to the residual projection
        r_fom  = self._eval_fom_residual(w_final)
        m      = self._m_eff
        s      = self.scaling
        Psi    = self.test_basis

        self.print0(f"  {'-'*56}")
        self.print0(f"  Mode contributions to r_rom:")
        self.print0(f"  {'Mode':>6}  {'|<Psi_i, r>|':>14}  {'q_i':>12}  {'active':>8}")
        self.print0(f"  {'-'*6}  {'-'*14}  {'-'*12}  {'-'*8}")

        contribs = []
        for i in range(self.basis_size):
            psi_i        = Psi[:, i]                          # i-th test basis vector (S Phi_i)
            proj_local   = np.dot(psi_i * m, r_fom)
            proj_global  = np.zeros(1)
            self.comm.Allreduce(proj_local, proj_global, op=MPI.SUM)
            contribs.append(abs(proj_global[0]))

        total = max(sum(contribs), 1e-14)
        for i, contrib in enumerate(contribs):
            active = "yes" if abs(q[i]) > 1e-14 else "-"
            self.print0(
                f"  {i+1:>6}  {contrib:>14.4e}  {q[i]:>12.4e}  {active:>8}"
            )
        self.print0(f"  {'-'*56}")

    
    # region _print_orthogonality_check
    def _print_orthogonality_check(self):
        # Level 3 - verify Phi^T M Phi = I under current inner product
        Phi = self.pod_modes
        s   = self.scaling
        m   = self.weights
        n   = self.basis_size

        # Build Gram matrix: G_ij = (S Phi_i)^T M (S Phi_j) = local sum, then Allreduce
        G_local = Phi.T @ (m[:, None] * Phi)  # (n, n)
        G       = np.zeros_like(G_local)
        self.comm.Allreduce(G_local, G, op=MPI.SUM)

        residual     = G - np.eye(n)
        ortho_error  = np.linalg.norm(residual, 'fro')
        diag_error   = np.max(np.abs(np.diag(G) - 1.0))
        offdiag_error = np.max(np.abs(residual - np.diag(np.diag(residual))))

        self.print0(f"  {'-'*56}")
        self.print0(f"  M-orthogonality check  (‖Phi^T M Phi - I‖_F):")
        self.print0(f"  {'‖G - I‖_F':<32} {ortho_error:.4e}")
        self.print0(f"  {'max diag error':<32} {diag_error:.4e}")
        self.print0(f"  {'max off-diag error':<32} {offdiag_error:.4e}")
        self.print0(f"  {'basis orthonormal?':<32} {'yes' if ortho_error < 1e-10 else '**NO**'}")