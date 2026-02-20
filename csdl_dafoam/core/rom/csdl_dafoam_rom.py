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
    Psi = { Phi if Galerkin
          { J @ S @ Phi if Petrov-Galerkin
    
'''



# region DAFOAMROM
class DAFoamROM(csdl.experimental.CustomImplicitOperation):
    def __init__(
        self, 
        dafoam_instance,
        pod_modes=None,          # constant modes (numpy, distributed)
        reference_state=None,    # distributed numpy
        scaling=None,            # distributed numpy (diagonal of S)
        weights=None,            # distributed numpy (diagonal of M)
        rom_type='lspg',         # 'galerkin' or 'lspg'
        jac_mode='fd',           # 'analytical' (galerkin only) or 'fd'
        param_sens_mode='galerkin_adjoint', # or 'fd'
        newton_options=None,     # maxiter, tol, line search params, fd eps
    ):
        super().__init__()
        self.dafoam_instance = dafoam_instance

        # These will either be the user supplied constants, or we'll update them
        # As the optimization progresses (that way all functions can reference them)
        self.pod_modes          = pod_modes
        self.reference_state    = reference_state
        self.scaling            = scaling
        self.weights            = weights
        self.test_basis         = None

        self.rom_type           = self._check_if_in_list(rom_type,          ["galerkin", "lspg"])
        self.jac_mode           = self._check_if_in_list(jac_mode,          ["fd", "analytical"])
        self.param_sens_mode    = self._check_if_in_list(param_sens_mode,   ["galerkin_adjoint", "fd"])
        self.newton_options     = newton_options

        # These are used in the evaluate section, but initialized here
        self.pod_modes_is_csdl_var       = None
        self.reference_state_is_csdl_var = None
        self.scaling_is_csdl_var         = None

        # DAFoam
        self.n_local_states = dafoam_instance.getNLocalAdjointStates()

        # MPI
        self.comm           = dafoam_instance.comm
        self.rank           = self.comm.rank
        self.comm_size      = self.comm.size

        # Cached values (set during solve, used in derivative methods)
        self._cached_q      = None # converged reduced state
        self._cached_J_r    = None # ROM Jacobian at convergence
        self._cached_w      = None # reconstructed FOM state at convergence

     
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

        return dafoam_rom_states


    # region solve_residual_equations
    def solve_residual_equations(self, input_vals, output_vals):
        dafoam_instance         = self.dafoam_instance
        
        # Initialize/update quantities
        self.pod_modes          = input_vals["pod_modes"]       if self.pod_modes_is_csdl_var       else self.pod_modes
        self.scaling            = input_vals["scaling"]         if self.scaling_is_csdl_var         else self.scaling
        self.reference_state    = input_vals["reference_state"] if self.reference_state_is_csdl_var else self.reference_state
        self.weights            = np.ones_like(self.reference_state) if self.weights is None        else self.weights


    # region apply_inverse_jacobian
    def apply_inverse_jacobian(self, input_vals, output_vals, d_outputs, d_residuals, mode):
        # Can't do forward mode
        if mode == 'fwd':
            raise NotImplementedError('forward mode has not been implemented for DAFoamROM')
        
        elif mode == 'rev':
            raise NotImplementedError('reverse mode has not been implemented for DAFoamROM')

        else:
            raise ValueError(f'"{mode}" not recognized. Only support "fwd" and "rev" modes')


    # region compute_jacvec_product
    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, d_residuals, mode):
       
        # Can't do forward mode
        if mode == 'fwd':
            raise NotImplementedError('forward mode has not been implemented for DAFoamROM')

        elif mode == 'rev':
            raise NotImplementedError('reverse mode has not been implemented for DAFoamROM')
        
        else:
            raise ValueError(f'"{mode}" not recognized. Only support "fwd" and "rev" modes')
        

    # region _reconstruct_fom_state_reconstruct_fom_state
    def _reconstruct_fom_state(self, rom_state):
        q       = rom_state
        s       = self.scaling
        Phi     = self.pod_modes
        w_ref   = self.reference_state

        return w_ref + (s[:, None] * Phi) @ q
    

    # region _eval_fom_residual
    def _eval_fom_residual(self, fom_state):
        # This assumes we have already set the inputs! Should generally be the case
        w = fom_state
        dafoam_instance = self.dafoam_instance
        dafoam_instance.setStates(w)
        return dafoam_instance.getResiduals()


    # region _project_to_reduced
    def _project(self, distributed_val):
        comm = self.comm
        m    = self.weights
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
        h = step

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
        if self.rom_type == "galerkin":
            self.test_basis = self.pod_modes

        elif self.rom_type == 'lspg':
            Phi = self.pod_modes
            s   = self.scaling
            w   = fom_state
            self.test_basis = self._jac_mat_product(fom_state=w, matrix=s[:, None] * Phi, step=step)
   

    # region _eval_rom_residual
    def _eval_rom_residual(self, rom_state):
        q = rom_state
        w = self._reconstruct_fom_state(q)
        r = self._eval_fom_residual(w)

        return self._project(distributed_vec=r)
        

    # region _eval_rom_jacobian_fd
    def _compute_rom_jacobian(self, fom_state):
        Psi = self.test_basis
        Phi = self.pod_modes
        m   = self.weights
        s   = self.scaling
        w   = fom_state

        if self.jac_mode == "fd":
            if self.rom_type == "lspg":
                J_rom_local = Psi.T @ (m[:, None] * Psi)

                J_rom = np.zeros_like(J_rom_local)
                self.comm.Allreduce(J_rom_local, J_rom, op=MPI.SUM)

            elif self.rom_type == "galerkin":
                # _project already handles the reduction
                J_rom = self._project(self._jac_mat_product(fom_state=w, matrix=s[:, None] * Phi))

        elif self.jac_mode == "analytical":
            if self.rom_type == "lspg":
                raise NotImplemented("Analytical Jacobian products are not supported for Least Squares Petrov-Galerkin.")
            
            JT_MT_Phi = self._jacT_mat_product(fom_state=w, matrix=m[:, None] * Phi)
            J_rom_local     = JT_MT_Phi.T @ (s[:, None] * Phi)

            J_rom = np.zeros_like(J_rom_local)
            self.comm.Allreduce(J_rom_local, J_rom, op=MPI.SUM)

        return J_rom


    # region _rom_newtwon_solve
    def _rom_newtwon_solve(self, q0, input_vales, modes):
        raise NotImplementedError("to be implemented")
    

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
                TypeError(f'{name} not assigned to ROM. Please pass {name} during initialization or evaluation.')
            else:
                if name == "pod_modes":
                    num_modes = variable.shape[1]
                setattr(self, f"{name}_is_csdl_var", False)
        
        if name == "pod_modes":
            return num_modes
        

    # region _check_if_in_list_and_return_if_so
    def _check_if_in_list(self, name, list):
        if name.lower() in list:
            return name.lower()
        else:
            raise ValueError(f"{name} is not a valid option. Please choose from {list}")


    # region print0
    def print0(self, statement, **kwargs):
        if self.rank == 0:
            print(statement, **kwargs)








# # region DAFOAMROM
# class DAFoamROM(csdl.experimental.CustomImplicitOperation):
#     def __init__(self, 
#                  dafoam_instance,
#                  pod_modes:np.array         =None,
#                  fom_ref_state:np.array     =None, 
#                  tolerance                  =1e-6, 
#                  max_iters:int              =100, 
#                  update_jac_frequency:int   =10,
#                  num_initial_jac_updates:int=1, 
#                  weights:np.array           =None,
#                  scaling_factors:np.array   =None):

#         super().__init__()
#         self.dafoam_instance        = dafoam_instance
#         self.tolerance              = tolerance
#         self.max_iters              = max_iters
#         self.solution_counter       = 1
#         self.update_jac_frequency   = update_jac_frequency if update_jac_frequency > 0 else np.nan
#         self.num_initial_jac_updates= num_initial_jac_updates
#         self.reduced_jacobian       = None
#         self.n_local_state_elements = dafoam_instance.getNLocalAdjointStates()
#         self.n_local_cells          = dafoam_instance.solver.getNLocalCells()

#         # Do some checks here to assign default values if none passed. Alert user if none are passed.
#         if weights is None:
#             print('No weights passed to DAFoamROM. Assuming no weighting used in POD mode computation...')
#             self.weights = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
#         else:
#             self.weights = weights

#         if scaling_factors is None:
#             print('No weights passed to DAFoamROM. Assuming no scaling factor used in POD mode computation...')
#             self.scaling_factors = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
#         else:
#             self.scaling_factors = scaling_factors

#         if fom_ref_state is None:
#             print('No FOM reference state passed to DAFoamROM. Assuming no mean/reference subtraction used in POD mode computation...')
#             self.fom_ref_state = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
#         else:
#             self.fom_ref_state = fom_ref_state

#         if pod_modes is None:
#             print('No POD modes passed to DAFoamROM. Expecting POD modes as CSDL variable for the evaluate method...')
#         else:
#             print('POD modes passed to DAFoamROM. Assuming constant/global POD modes unless modes are passed to the evaluate method...')
        
#         self.pod_modes = pod_modes

        
#     # region evaluate
#     def evaluate(self, dafoam_input_variables_group:csdl.VariableGroup, pod_modes:csdl.Variable=None, design_variable_configuration:csdl.Variable=None):
        
#         # Read daOptions to set proper inputs
#         inputDict = self.dafoam_instance.getOption("inputInfo")
#         for inputName in inputDict.keys():
#             if "solver" in inputDict[inputName]["components"]:
#                 self.declare_input(inputName, getattr(dafoam_input_variables_group, inputName))

#         # Set our basis as an input variable if not already set to a constant
#         if pod_modes is not None:
#             if self.pod_modes is None:
#                 print('POD modes passed to DAFoamROM evalute method. Neglecting constant/global modes passed in initialization.')
#             self.declare_input('pod_modes', pod_modes)
#             num_modes = pod_modes.value.shape[1]
#             self.pod_modes_is_csdl_var = True
#         else:
#             if self.pod_modes is None:
#                 TypeError('No POD modes assigned to ROM. Please pass a constant POD mode set during initialization, or a POD mode set in the evaluate section.')
#             else:
#                 num_modes = self.pod_modes.shape[1]
#                 self.pod_modes_is_csdl_var = False
        
#         # Set outputs
#         dafoam_rom_states = self.create_output('dafoam_rom_states', (num_modes,))

#         return dafoam_rom_states


#     # region solve_residual_equations
#     def solve_residual_equations(self, input_vals, output_vals):
#         dafoam_instance       = self.dafoam_instance

#         # Solver parameters/options (not all have been exposed as options yet. Tweak here if necessary.)
#         max_iters                       = self.max_iters
#         update_jac_frequency            = self.update_jac_frequency
#         num_initial_jac_updates         = self.num_initial_jac_updates
#         tolerance                       = self.tolerance
#         line_search_minimum_reduction   = 1e-4  # Minimum search distance for line search (Where full step = 1)
#         line_search_shrink_factor       = 0.5   # The factor by which the line search reduces its search
#         trigger_jac_recompute_threshold = 3     # Number of sequential line search failures necessary to trigger jacobian recompute
       
#         # Check if we're working with the pod modes as a constant or as a variable
#         pod_modes_is_csdl_var = self.pod_modes_is_csdl_var

#         # Some important quanitites
#         y_fom_ref  = self.fom_ref_state
#         W          = self.weights
#         D          = self.scaling_factors
#         phi        = input_vals['pod_modes'] if pod_modes_is_csdl_var else self.pod_modes
#         num_modes  = phi.shape[1]
        
#         # MPI stuff
#         # TODO: will have to make this all MPI compatible eventually
#         rank = dafoam_instance.rank

#         # Make sure solver is updated with the most recent input values
#         dafoam_instance.set_solver_input(input_vals)

#         # We also need to just calculate the residual for the AD mode to initialize vars like URes
#         # We do not print the residual for AD, though
#         # NOTE: This was taken from DAFoamSolver.solve_residual_equation
#         dafoam_instance.solverAD.calcPrimalResidualStatistics("calc")
        
#         # Initialize our ROM/FOM states and ROM/FOM residuals
#         y_rom = np.zeros((num_modes, ))
#         y_fom, r_fom, r_rom = self._rom_states_to_res_and_fom(phi, y_rom, W, D)
#         r_rom_norm = np.linalg.norm(r_rom)

#         # Some plotting for diagnostics
#         #----------------------
#         import matplotlib.pyplot as plt
#         # Plotting coefficients
#         # Initialize figure
#         plt.ion()  # Interactive mode on
#         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,6))

#         # --- subplot 1: ROM state progression ---
#         lines = [ax1.plot([], [], label=f'element {i}')[0] for i in range(num_modes)]

#         # Store data
#         x_data = []                     # iteration index
#         y_data = [[] for _ in range(num_modes) ]  # data for each element

#         # --- subplot 2: FOM update ---
#         fom_state_line,    = ax2.plot(np.zeros_like(y_fom_ref), 'r-')

#         # --- subplot 3: residual update ---
#         fom_residual_line, = ax3.plot(np.zeros_like(y_fom_ref), 'b')

#         x_data.append(0)
#         for i, line in enumerate(lines):
#             y_data[i].append(y_rom[i])
#             line.set_data(x_data, y_data[i])
#             plt.draw()
#             plt.pause(0.1)
#         #-------------------------

#         # Initializing loop counters and flags
#         trigger_jac_recompute_counter   = 0
#         trigger_jac_recompute           = False
#         last_accepted                   = True
#         converged                       = False

#         # Main Newton iteration loop
#         for iter in range(max_iters):
#             print('\n')
#             print('-------------------------------------')
#             print(f'ROM Newton iteration {iter}')
#             print('-------------------------------------')

#             # Save old ROM values (for Broyden update)
#             y_rom_old = y_rom
#             r_rom_old = r_rom

#             # (Re)compute Jacobian on first iteration(s) or if triggered (maybe by update frequency)
#             if iter%update_jac_frequency == 0 or iter < num_initial_jac_updates or trigger_jac_recompute:
#                 print('Computing reduced Jacobian...')
#                 J_rom = self._compute_reduced_jacobian(phi, y_fom, y_rom, W, D, mode='fd')
#                 self.reduced_jacobian = J_rom
#                 trigger_jac_recompute = False

#             # Compute step (robust linear solve with lstsq fallback) [This was a GPT suggestion - not sure if the execpt will ever trigger]
#             print('Updating ROM and FOM states...')
#             try:
#                 delta_y_rom = np.linalg.solve(J_rom, -r_rom)
#             except np.linalg.LinAlgError:
#                 # fallback to least-squares if matrix singular or near-singular
#                 delta_y_rom = np.linalg.lstsq(J_rom, -r_rom, rcond=1e-12)[0]
#                 delta_y_rom = delta_y_rom.reshape(-1,)
#                 print('\tWarning: reduced Jacobian singular; used least-squares fallback.')

#             # Initialize line search params and begin search
#             accepted = False
#             alpha    = 1
#             while alpha > line_search_minimum_reduction:
#                 print(f'Line search: alpha = {alpha}')
#                 y_rom_test = y_rom + alpha*delta_y_rom
#                 y_fom_test, r_fom_test, r_rom_test = self._rom_states_to_res_and_fom(phi, y_rom_test, W, D)
#                 r_rom_test_norm = np.linalg.norm(r_rom_test)

#                 if r_rom_test_norm < r_rom_norm:
#                     y_rom      = y_rom_test
#                     y_fom      = y_fom_test
#                     r_rom      = r_rom_test
#                     r_fom      = r_fom_test
#                     r_rom_norm = r_rom_test_norm
#                     accepted   = True
#                     trigger_jac_recompute_counter = 0
#                     break

#                 alpha *= line_search_shrink_factor

#             if not accepted:
#                 print('Line search failed! Taking small step...')
#                 y_rom = y_rom + line_search_minimum_reduction*delta_y_rom
#                 y_fom, r_fom, r_rom = self._rom_states_to_res_and_fom(phi, y_rom, W, D)
#                 r_rom_norm = np.linalg.norm(r_rom)

#                 # Check if the line search failed last time
#                 if not last_accepted:
#                     trigger_jac_recompute_counter += 1
                    
#                     # Trigger recompute after threshold number of sequential failures (have to subtract 1 because counter starts at 0)
#                     if trigger_jac_recompute_counter >= trigger_jac_recompute_threshold - 1:
#                         trigger_jac_recompute = True

#             last_accepted = accepted

#             # Save the initial value of the reduced residual
#             if iter == 0:
#                 r_rom_norm0 = r_rom_norm.copy()

#             # Print out the relevant residual statistics
#             print('Residual summary:')
#             print('\t{:<30} : {:.4E}'.format('FOM residual norm', np.linalg.norm(r_fom)))
#             print('\t{:<30} : {:.4E}'.format('ROM residual norm', r_rom_norm))
#             print('\t{:<30} : {:.4E}'.format('Start/current ROM norm ratio', r_rom_norm/r_rom_norm0))

#             # Exit condition
#             if r_rom_norm < tolerance:
#                 converged = True
#                 break

#             # Broyden update
#             self._broyden_update(J_rom, y_rom_old, y_rom, r_rom_old, r_rom)

#             # Upate plots
#             x_data.append(iter+1)
#             for i, line in enumerate(lines):
#                 y_data[i].append(y_rom[i])
#                 line.set_data(x_data, y_data[i])
#             ax1.relim()
#             ax1.autoscale_view()

#             fom_state_line.set_ydata(y_fom)
#             ax2.relim()
#             ax2.autoscale_view()

#             fom_residual_line.set_ydata(r_fom)
#             ax3.relim()
#             ax3.autoscale_view()
#             plt.draw()
#             plt.pause(0.1)
        
#         # Return NaN array as output if failed (for the OpenSQP optimizer), otherwise return rom states
#         if converged:
#             print('\n\n Specified tolerance reached!')
#             # Update reduced jacobain to most recent state
#             print('Updating Jacobian to most recent state value...')
#             J_rom = self._compute_reduced_jacobian(phi, y_fom, y_rom, W, D)
#             self.reduced_jacobian = J_rom

#             output_vals['dafoam_rom_states'] = y_rom

#         else:
#             print('Warning: ROM iteration limit reached!')
#             output_vals['dafoam_rom_states'] = np.full((num_modes, ), np.nan)


#     # region apply_inverse_jacobian
#     def apply_inverse_jacobian(self, input_vals, output_vals, d_outputs, d_residuals, mode):
#         # Can't do forward mode
#         if mode == 'fwd':
#             raise NotImplementedError('forward mode has not been implemented for DAFoamROM')
        
#         elif mode == 'rev':
#             J_romT = self.reduced_jacobian.T
#             v      = d_outputs['dafoam_rom_states']
#             d_residuals['dafoam_rom_states'] = np.linalg.solve(J_romT, v)

#         else:
#             raise ValueError(f'"{mode}" not recognized. Only support "fwd" and "rev" modes')

#         # Components for writing solution to file
#         dafoam_instance = self.dafoam_instance
#         D               = self.scaling_factors
#         W               = self.weights
#         y_fom_ref       = self.fom_ref_state
#         phi             = input_vals['pod_modes'] if self.pod_modes_is_csdl_var else self.pod_modes
#         y_rom           = output_vals['dafoam_rom_states']
#         y_fom           = y_fom_ref + D*(phi@y_rom)
#         n_points        = dafoam_instance.solver.getNLocalPoints()

#         # Initialize mesh array and write to file
#         write_number    = round(1e-4*self.solution_counter, 4)
#         points0         = np.zeros(n_points*3)
#         dafoam_instance.solver.getOFMeshPoints(points0)
#         dafoam_instance.solver.writeMeshPoints(points0, write_number)

#         # Write primitives
#         dafoam_instance.solver.writeAdjointFields('sol', write_number, y_fom)
#         self.solution_counter += 1


#     # region compute_jacvec_product
#     def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, d_residuals, mode):
#         dafoam_instance = self.dafoam_instance
#         comm            = dafoam_instance.comm
#         rank            = dafoam_instance.rank

#         D         = self.scaling_factors
#         W         = self.weights
#         phi       = self.pod_modes
#         y_fom_ref = self.fom_ref_state

#         dafoam_scaling_factors = dafoam_instance.getStateScalingFactors()

#         # assign the states in outputs to the OpenFOAM flow fields
#         # NOTE: this is not quite necessary because setStates have been called before in the solve_nonlinear
#         # here we call it just be on the safe side
#         # Check if states contain any NaN values (NaNs would be passed from optimizer)
#         y_rom = output_vals['dafoam_rom_states']
#         y_fom = y_fom_ref + D*(phi@y_rom)

#         # Update solver states only if no NaNs exist
#         if not has_global_nan_or_inf(y_fom, comm):
#             dafoam_instance.setStates(y_fom)
#         else:
#             if rank == 0:
#                 print('DAFoamSolver.compute_jacvec_product: Detected NaN(s) in input_vals. Skipping DAFoam setStates')

#         # Can't do forward mode
#         if mode == 'fwd':
#             raise NotImplementedError('forward mode has not been implemented for DAFoamROM')

#         elif mode == 'rev':
#             if 'dafoam_rom_states' in d_residuals:
#                 # get the reverse mode AD seed from d_residuals and expand to FOM size
#                 seed_r = d_residuals['dafoam_rom_states']
#                 seed   = W*(phi@seed_r)
    
#                 # loop over all inputs keys and compute the matrix-vector products accordingly
#                 input_dict = dafoam_instance.getOption("inputInfo")
#                 for input_name in list(input_vals.keys()):
#                     # (Taking care of the DAFoam inputs here)
#                     if input_name in input_dict:
#                         input_type = input_dict[input_name]["type"]
#                         jac_input  = input_vals[input_name].copy()
#                         product    = np.zeros_like(jac_input)
#                         dafoam_instance.solverAD.calcJacTVecProduct(
#                             input_name,
#                             input_type,
#                             jac_input,
#                             "aero_residuals",
#                             "residual",
#                             seed,
#                             product,
#                         )
#                         d_inputs[input_name] += product
        
#         else:
#             raise ValueError(f'"{mode}" not recognized. Only support "fwd" and "rev" modes')


#     # region _compute_reduced_jacobian
#     def _compute_reduced_jacobian(self, phi, y_fom, y_rom, W, D, mode='fom_jTvp', fd_eps=1e-6):
#         dafoam_instance = self.dafoam_instance
#         num_modes       = phi.shape[1]

#         if mode == 'fom_jTvp': # Computing using DAFoamSolver's calcJacTVecProduct column by column and projecting
#             J_fomT_W_phi           = np.zeros_like(phi)
#             W_phi                  = W[:, None]*phi
#             dafoam_scaling_factors = dafoam_instance.getStateScalingFactors()

#             # Update solver for desired states
#             dafoam_instance.setStates(y_fom)
            
#             # Loop over columns of phi
#             for i in range(num_modes):
#                 seed    = np.ascontiguousarray(W_phi[:, i])
#                 product = np.zeros_like(seed)
#                 dafoam_instance.solverAD.calcJacTVecProduct(
#                     'dafoam_solver_states',
#                     "stateVar",
#                     y_fom,
#                     'aero_residuals',
#                     "residual",
#                     seed,
#                     product,
#                 )

#                 J_fomT_W_phi[:, i] = product/dafoam_scaling_factors
                
#             J_romT = phi.T@(D[:, None]*J_fomT_W_phi)
#             J_rom  = J_romT.T

#             return J_rom

#         elif mode == 'fd': # Finite difference method in reduced space
#             J_rom  = np.zeros((num_modes, num_modes))
            
#             # Get reduced residual
#             r_fom, r_rom0 = self._compute_residuals(phi, y_fom, W, D)

#             # Will choose this value if the stepsize gets too small
#             min_stepsize_magnitude = 1e-8

#             for j in range(num_modes):         
#                 # Perturb the reduced coordinates by fd_eps (choose lower bound if too small)
#                 delta_j    = max(np.abs(fd_eps*y_rom[j]), min_stepsize_magnitude)
#                 y_fom_pert = y_fom + D*phi[:, j]*delta_j

#                 # Get perturbed reduced residuals
#                 r_rom_pert = self._compute_residuals(phi, y_fom_pert, W, D)[1]

#                 # Add to respective reduced jacobian column
#                 J_rom[:, j] = (r_rom_pert - r_rom0)/delta_j

#             # Reset states in DAFoam
#             dafoam_instance.setStates(y_fom) 

#             return J_rom

#         else:
#             raise ValueError(f"Unknown J_rom computation mode '{mode}'. Choose from 'fom_jTvp' or 'fd'")
    

#     # region _compute_residuals
#     # This is just a helper function for convenience
#     def _compute_residuals(self, phi, y_fom, W, D):
#         dafoam_instance = self.dafoam_instance

#         dafoam_instance.setStates(y_fom)
#         r_fom = dafoam_instance.getResiduals()
#         r_rom = phi.T@(W*r_fom)

#         return r_fom, r_rom

    
#     # region _rom_states_to_res_and_fom
#     def _rom_states_to_res_and_fom(self, phi, y_rom, W, D):
#         dafoam_instance = self.dafoam_instance
#         y_fom_ref       = self.fom_ref_state

#         y_fom = y_fom_ref + D*(phi@y_rom)
#         r_fom, r_rom = self._compute_residuals(phi, y_fom, W, D)

#         return y_fom, r_fom, r_rom

    
#     # region _broyden_update
#     def _broyden_update(self, J, x_old, x_new, f_old, f_new):
#         dx = x_new - x_old
#         df = f_new - f_old
#         J += np.outer((df - np.dot(J, dx)), dx)/np.dot(dx, dx)





# # region DAFOAMROM2
# class DAFoamROM2(csdl.experimental.CustomImplicitOperation):
#     def __init__(self, 
#                  dafoam_instance,
#                  pod_modes:np.array         =None,
#                  fom_ref_state:np.array     =None, 
#                  tolerance                  =1e-6, 
#                  max_iters:int              =100, 
#                  update_jac_frequency:int   =10,
#                  num_initial_jac_updates:int=1, 
#                  weights:np.array           =None,
#                  scaling_factors:np.array   =None):

#         super().__init__()
#         self.dafoam_instance        = dafoam_instance
#         self.tolerance              = tolerance
#         self.max_iters              = max_iters
#         self.solution_counter       = 1
#         self.update_jac_frequency   = update_jac_frequency if update_jac_frequency > 0 else np.nan
#         self.num_initial_jac_updates= num_initial_jac_updates
#         self.reduced_jacobian       = None
#         self.n_local_state_elements = dafoam_instance.getNLocalAdjointStates()
#         self.n_local_cells          = dafoam_instance.solver.getNLocalCells()

#         # Do some checks here to assign default values if none passed. Alert user if none are passed.
#         if weights is None:
#             print('No weights passed to DAFoamROM. Assuming no weighting used in POD mode computation...')
#             self.weights = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
#         else:
#             self.weights = weights

#         if scaling_factors is None:
#             print('No weights passed to DAFoamROM. Assuming no scaling factor used in POD mode computation...')
#             self.scaling_factors = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
#         else:
#             self.scaling_factors = scaling_factors

#         if fom_ref_state is None:
#             print('No FOM reference state passed to DAFoamROM. Assuming no mean/reference subtraction used in POD mode computation...')
#             self.fom_ref_state = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
#         else:
#             self.fom_ref_state = fom_ref_state

#         if pod_modes is None:
#             print('No POD modes passed to DAFoamROM. Expecting POD modes as CSDL variable for the evaluate method...')
#         else:
#             print('POD modes passed to DAFoamROM. Assuming constant/global POD modes unless modes are passed to the evaluate method...')
        
#         self.pod_modes = pod_modes

        
#     # region evaluate
#     def evaluate(self, dafoam_input_variables_group:csdl.VariableGroup, pod_modes:csdl.Variable=None, design_variable_configuration:csdl.Variable=None):
        
#         # Read daOptions to set proper inputs
#         inputDict = self.dafoam_instance.getOption("inputInfo")
#         for inputName in inputDict.keys():
#             if "solver" in inputDict[inputName]["components"]:
#                 self.declare_input(inputName, getattr(dafoam_input_variables_group, inputName))

#         # Set our basis as an input variable if not already set to a constant
#         if pod_modes is not None:
#             if self.pod_modes is None:
#                 print('POD modes passed to DAFoamROM evalute method. Neglecting constant/global modes passed in initialization.')
#             self.declare_input('pod_modes', pod_modes)
#             num_modes = pod_modes.value.shape[1]
#             self.pod_modes_is_csdl_var = True
#         else:
#             if self.pod_modes is None:
#                 TypeError('No POD modes assigned to ROM. Please pass a constant POD mode set during initialization, or a POD mode set in the evaluate section.')
#             else:
#                 num_modes = self.pod_modes.shape[1]
#                 self.pod_modes_is_csdl_var = False
        
#         # Set outputs
#         dafoam_rom_states = self.create_output('dafoam_rom_states', (num_modes,))

#         return dafoam_rom_states


#     # region solve_residual_equations
#     def solve_residual_equations(self, input_vals, output_vals):
#         from scipy.optimize import root, least_squares

#         dafoam_instance       = self.dafoam_instance

#         # Solver parameters/options (not all have been exposed as options yet. Tweak here if necessary.)
#         max_iters                       = self.max_iters
#         update_jac_frequency            = self.update_jac_frequency
#         num_initial_jac_updates         = self.num_initial_jac_updates
#         tolerance                       = self.tolerance
#         line_search_minimum_reduction   = 1e-4  # Minimum search distance for line search (Where full step = 1)
#         line_search_shrink_factor       = 0.5   # The factor by which the line search reduces its search
#         trigger_jac_recompute_threshold = 3     # Number of sequential line search failures necessary to trigger jacobian recompute
       
#         # Check if we're working with the pod modes as a constant or as a variable
#         pod_modes_is_csdl_var = self.pod_modes_is_csdl_var

#         # Some important quanitites
#         y_fom_ref  = self.fom_ref_state
#         W          = self.weights
#         D          = self.scaling_factors
#         phi        = input_vals['pod_modes'] if pod_modes_is_csdl_var else self.pod_modes
#         num_modes  = phi.shape[1]
        
#         # MPI stuff
#         # TODO: will have to make this all MPI compatible eventually
#         rank = dafoam_instance.rank

#         # Make sure solver is updated with the most recent input values
#         dafoam_instance.set_solver_input(input_vals)

#         # We also need to just calculate the residual for the AD mode to initialize vars like URes
#         # We do not print the residual for AD, though
#         # NOTE: This was taken from DAFoamSolver.solve_residual_equation
#         dafoam_instance.solverAD.calcPrimalResidualStatistics("calc")
        
#         # Initialize our ROM/FOM states and ROM/FOM residuals
#         y_rom = np.zeros((num_modes, ))
#         y_fom, r_fom, r_rom = self._rom_states_to_res_and_fom(phi, y_rom, W, D)
#         r_rom_norm = np.linalg.norm(r_rom)

#         # Initializing loop counters and flags
#         trigger_jac_recompute_counter   = 0
#         trigger_jac_recompute           = False
#         last_accepted                   = True
#         converged                       = False

#         root_or_ls = 'root'
#         if root_or_ls == 'root':
#             def _reduced_residual(y_rom_fun):
#                 y_fom, r_fom, r_rom_fun = self._rom_states_to_res_and_fom(phi, y_rom_fun, W, D)

#                 J_rom = self._compute_reduced_jacobian(phi, y_fom, y_rom_fun, W, D, mode='fd', fd_eps=1e-6)
#                 return r_rom_fun, J_rom

#             opts = {"disp": True,
#                     "maxiter": 100,
#                     "ftol": 1e-8,
#                     }
#             result = root(_reduced_residual, y_rom, method='lm', options=opts, jac=True)

#         else:
#             def _reduced_residual(y_rom_fun):
#                 y_fom, r_fom, r_rom_fun = self._rom_states_to_res_and_fom(phi, y_rom_fun, W, D)
#                 return r_rom_fun

#             def _reduced_jacobian(y_rom_fun):
#                 J_rom = self._compute_reduced_jacobian(phi, y_fom, y_rom_fun, W, D, mode='fd', fd_eps=1e-6)
#                 return J_rom

#             result = least_squares(_reduced_residual, y_rom, method='lm', verbose = 1)#, jac=_reduced_jacobian, verbose=1)

#         print(result)
#         y_rom     = result.x
#         converged = result.success

#         # Return NaN array as output if failed (for the OpenSQP optimizer), otherwise return rom states
#         if converged:
#             print('\n\n Specified tolerance reached!')
#             # Update reduced jacobain to most recent state
#             print('Updating Jacobian to most recent state value...')
#             J_rom = self._compute_reduced_jacobian(phi, y_fom, y_rom, W, D, mode='fd')
#             self.reduced_jacobian = J_rom

#             output_vals['dafoam_rom_states'] = y_rom

#         else:
#             print('Warning: ROM iteration limit reached!')
#             output_vals['dafoam_rom_states'] = np.full((num_modes, ), np.nan)


#     # region apply_inverse_jacobian
#     def apply_inverse_jacobian(self, input_vals, output_vals, d_outputs, d_residuals, mode):
#         # Can't do forward mode
#         if mode == 'fwd':
#             raise NotImplementedError('forward mode has not been implemented for DAFoamROM')
        
#         elif mode == 'rev':
#             J_romT = self.reduced_jacobian.T
#             v      = d_outputs['dafoam_rom_states']
#             d_residuals['dafoam_rom_states'] = np.linalg.solve(J_romT, v)

#         else:
#             raise ValueError(f'"{mode}" not recognized. Only support "fwd" and "rev" modes')

#         # Components for writing solution to file
#         dafoam_instance = self.dafoam_instance
#         D               = self.scaling_factors
#         W               = self.weights
#         y_fom_ref       = self.fom_ref_state
#         phi             = input_vals['pod_modes'] if self.pod_modes_is_csdl_var else self.pod_modes
#         y_rom           = output_vals['dafoam_rom_states']
#         y_fom           = y_fom_ref + D*(phi@y_rom)
#         n_points        = dafoam_instance.solver.getNLocalPoints()

#         # Initialize mesh array and write to file
#         write_number    = round(1e-4*self.solution_counter, 4)
#         points0         = np.zeros(n_points*3)
#         dafoam_instance.solver.getOFMeshPoints(points0)
#         dafoam_instance.solver.writeMeshPoints(points0, write_number)

#         # Write primitives
#         dafoam_instance.solver.writeAdjointFields('sol', write_number, y_fom)
#         self.solution_counter += 1


#     # region compute_jacvec_product
#     def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, d_residuals, mode):
#         dafoam_instance = self.dafoam_instance
#         comm            = dafoam_instance.comm
#         rank            = dafoam_instance.rank

#         D         = self.scaling_factors
#         W         = self.weights
#         phi       = self.pod_modes
#         y_fom_ref = self.fom_ref_state

#         dafoam_scaling_factors = dafoam_instance.getStateScalingFactors()

#         # assign the states in outputs to the OpenFOAM flow fields
#         # NOTE: this is not quite necessary because setStates have been called before in the solve_nonlinear
#         # here we call it just be on the safe side
#         # Check if states contain any NaN values (NaNs would be passed from optimizer)
#         y_rom = output_vals['dafoam_rom_states']
#         y_fom = y_fom_ref + D*(phi@y_rom)

#         # Update solver states only if no NaNs exist
#         if not has_global_nan_or_inf(y_fom, comm):
#             dafoam_instance.setStates(y_fom)
#         else:
#             if rank == 0:
#                 print('DAFoamSolver.compute_jacvec_product: Detected NaN(s) in input_vals. Skipping DAFoam setStates')

#         # Can't do forward mode
#         if mode == 'fwd':
#             raise NotImplementedError('forward mode has not been implemented for DAFoamROM')

#         elif mode == 'rev':
#             if 'dafoam_rom_states' in d_residuals:
#                 # get the reverse mode AD seed from d_residuals and expand to FOM size
#                 seed_r = d_residuals['dafoam_rom_states']
#                 seed   = W*(phi@seed_r)
    
#                 # loop over all inputs keys and compute the matrix-vector products accordingly
#                 input_dict = dafoam_instance.getOption("inputInfo")
#                 for input_name in list(input_vals.keys()):
#                     # (Taking care of the DAFoam inputs here)
#                     if input_name in input_dict:
#                         input_type = input_dict[input_name]["type"]
#                         jac_input  = input_vals[input_name].copy()
#                         product    = np.zeros_like(jac_input)
#                         dafoam_instance.solverAD.calcJacTVecProduct(
#                             input_name,
#                             input_type,
#                             jac_input,
#                             "aero_residuals",
#                             "residual",
#                             seed,
#                             product,
#                         )
#                         d_inputs[input_name] += product
#                         print(f"product shape: {product.shape}")
        
#         else:
#             raise ValueError(f'"{mode}" not recognized. Only support "fwd" and "rev" modes')


    
#     # region _compute_residuals
#     # This is just a helper function for convenience
#     def _compute_residuals(self, phi, y_fom, W, D):
#         dafoam_instance = self.dafoam_instance

#         dafoam_instance.setStates(y_fom)
#         r_fom = dafoam_instance.getResiduals()
#         r_rom = phi.T@(W*r_fom)

#         return r_fom, r_rom

    
#     # region _rom_states_to_res_and_fom
#     def _rom_states_to_res_and_fom(self, phi, y_rom, W, D):
#         dafoam_instance = self.dafoam_instance
#         y_fom_ref       = self.fom_ref_state

#         y_fom = y_fom_ref + D*(phi@y_rom)
#         r_fom, r_rom = self._compute_residuals(phi, y_fom, W, D)

#         return y_fom, r_fom, r_rom


#     # region _compute_reduced_jacobian
#     def _compute_reduced_jacobian(self, phi, y_fom, y_rom, W, D, mode='fom_jTvp', fd_eps=1e-6):
#         dafoam_instance = self.dafoam_instance
#         num_modes       = phi.shape[1]

#         if mode == 'fom_jTvp': # Computing using DAFoamSolver's calcJacTVecProduct column by column and projecting
#             J_romT_W_phi           = np.zeros_like(phi)
#             W_phi                  = W[:, None]*phi
#             dafoam_scaling_factors = dafoam_instance.getStateScalingFactors()

#             # Update solver for desired states
#             dafoam_instance.setStates(y_fom)
            
#             # Loop over columns of phi
#             for i in range(num_modes):
#                 seed    = np.ascontiguousarray(W_phi[:, i])
#                 product = np.zeros_like(seed)
#                 dafoam_instance.solverAD.calcJacTVecProduct(
#                     'dafoam_solver_states',
#                     "stateVar",
#                     y_fom,
#                     'aero_residuals',
#                     "residual",
#                     seed,
#                     product,
#                 )

#                 J_romT_W_phi[:, i] = product/dafoam_scaling_factors
                
#             J_romT = phi.T@(D[:, None]*J_romT_W_phi)
#             J_rom   = J_romT.T

#             return J_rom

#         elif mode == 'fd': # Finite difference method in reduced space
#             J_rom  = np.zeros((num_modes, num_modes))
            
#             # Get reduced residual
#             r_fom, r_rom0 = self._compute_residuals(phi, y_fom, W, D)

#             # Will choose this value if the stepsize gets too small
#             min_stepsize_magnitude = 1e-8

#             for j in range(num_modes):         
#                 # Perturb the reduced coordinates by fd_eps (choose lower bound if too small)
#                 delta_j    = max(np.abs(fd_eps*y_rom[j]), min_stepsize_magnitude)
#                 y_fom_pert = y_fom + D*phi[:, j]*delta_j

#                 # Get perturbed reduced residuals
#                 r_rom_pert = self._compute_residuals(phi, y_fom_pert, W, D)[1]

#                 # Add to respective reduced jacobian column
#                 J_rom[:, j] = (r_rom_pert - r_rom0)/delta_j

#             # Reset states in DAFoam
#             dafoam_instance.setStates(y_fom) 

#             return J_rom
        




# import matplotlib.pyplot as plt
# # region DAFOAMROM3
# class DAFoamROM3(csdl.experimental.CustomImplicitOperation):
#     def __init__(self, 
#                  dafoam_instance,
#                  pod_modes:np.array         =None,
#                  fom_ref_state:np.array     =None, 
#                  tolerance                  =1e-6, 
#                  max_iters:int              =100, 
#                  update_jac_frequency:int   =10,
#                  num_initial_jac_updates:int=1, 
#                  weights:np.array           =None,
#                  scaling_factors:np.array   =None,
#                  rom_type:str               ='galerkin',
#                  root_method:str            ='lm'):

#         super().__init__()
#         self.dafoam_instance        = dafoam_instance
#         self.tolerance              = tolerance
#         self.max_iters              = max_iters
#         self.solution_counter       = 1
#         self.update_jac_frequency   = update_jac_frequency if update_jac_frequency > 0 else np.nan
#         self.num_initial_jac_updates= num_initial_jac_updates
#         self.reduced_jacobian       = None
#         self.n_local_state_elements = dafoam_instance.getNLocalAdjointStates()
#         self.n_local_cells          = dafoam_instance.solver.getNLocalCells()

#         # Do some checks here to assign default values if none passed. Alert user if none are passed.
#         if weights is None:
#             print('No weights passed to DAFoamROM. Assuming no weighting used in POD mode computation...')
#             self.weights = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
#         else:
#             self.weights = weights

#         if scaling_factors is None:
#             print('No weights passed to DAFoamROM. Assuming no scaling factor used in POD mode computation...')
#             self.scaling_factors = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
#         else:
#             self.scaling_factors = scaling_factors

#         if fom_ref_state is None:
#             print('No FOM reference state passed to DAFoamROM. Assuming no mean/reference subtraction used in POD mode computation...')
#             self.fom_ref_state = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
#         else:
#             self.fom_ref_state = fom_ref_state
        
#         self.fom_ref_residual  = self._compute_fom_residuals(fom_ref_state)

#         if pod_modes is None:
#             print('No POD modes passed to DAFoamROM. Expecting POD modes as CSDL variable for the evaluate method...')
#         else:
#             print('POD modes passed to DAFoamROM. Assuming constant/global POD modes unless modes are passed to the evaluate method...')
        
#         self.pod_modes   = pod_modes

#         self.rom_type    = rom_type
#         self.root_method = root_method
#         self.counter     = 0

        
#     # region evaluate
#     def evaluate(self, dafoam_input_variables_group:csdl.VariableGroup, pod_modes:csdl.Variable=None, design_variable_configuration:csdl.Variable=None):
        
#         # Read daOptions to set proper inputs
#         inputDict = self.dafoam_instance.getOption("inputInfo")
#         for inputName in inputDict.keys():
#             if "solver" in inputDict[inputName]["components"]:
#                 self.declare_input(inputName, getattr(dafoam_input_variables_group, inputName))

#         # Set our basis as an input variable if not already set to a constant
#         if pod_modes is not None:
#             if self.pod_modes is None:
#                 print('POD modes passed to DAFoamROM evalute method. Neglecting constant/global modes passed in initialization.')
#             self.declare_input('pod_modes', pod_modes)
#             num_modes = pod_modes.value.shape[1]
#             self.pod_modes_is_csdl_var = True
#         else:
#             if self.pod_modes is None:
#                 TypeError('No POD modes assigned to ROM. Please pass a constant POD mode set during initialization, or a POD mode set in the evaluate section.')
#             else: # Constant POD mode case
#                 num_modes = self.pod_modes.shape[1]
#                 self.pod_modes_is_csdl_var = False
                
#                 # Check to make sure the POD modes are orthogonal under given weights
#                 print('Checking weighted inner product of supplied constant POD modes')
#                 weighted_inner_product = self.pod_modes.T @ (self.weights[:, None] * self.pod_modes)
#                 I                      = np.eye(weighted_inner_product.shape[0])
#                 fro_error              = np.linalg.norm(weighted_inner_product - I, ord='fro')
#                 max_entry_error        = np.max(np.abs(weighted_inner_product - I))

#                 print(f'Frobenius error: {fro_error}')
#                 print(f'Max entry error: {max_entry_error}')

#                 if fro_error > 1e-6:
#                     print(f'WARNING: high error detected (fro_error, {fro_error} > 1e-6)')         
        
#         # Set outputs
#         dafoam_rom_states = self.create_output('dafoam_rom_states', (num_modes,))

#         return dafoam_rom_states


#     # region solve_residual_equations
#     def solve_residual_equations(self, input_vals, output_vals):
#         from scipy.optimize import root, least_squares

#         dafoam_instance = self.dafoam_instance

#         # Solver parameters/options (not all have been exposed as options yet. Tweak here if necessary.)
#         root_method     = self.root_method
       
#         # Check if we're working with the pod modes as a constant or as a variable
#         pod_modes_is_csdl_var = self.pod_modes_is_csdl_var

#         # Some important quanitites
#         y_fom_ref  = self.fom_ref_state
#         W          = self.weights
#         D          = self.scaling_factors
#         phi        = input_vals['pod_modes'] if pod_modes_is_csdl_var else self.pod_modes
#         num_modes  = phi.shape[1]
        
#         # MPI stuff
#         # TODO: will have to make this all MPI compatible eventually
#         rank = dafoam_instance.rank

#         # Make sure solver is updated with the most recent input values
#         dafoam_instance.set_solver_input(input_vals)

#         # We also need to just calculate the residual for the AD mode to initialize vars like URes
#         # We do not print the residual for AD, though
#         # NOTE: This was taken from DAFoamSolver.solve_residual_equation
#         dafoam_instance.solverAD.calcPrimalResidualStatistics("calc")
        
#         # Initialize our ROM/FOM states and ROM/FOM residuals
#         y_rom = np.zeros((num_modes, ))
#         self.r_rom_old = y_rom.copy()

#         result = root(lambda x: self._compute_rom_residual_and_jacobian(x, phi, W, D), y_rom, method=root_method, jac=True)

#         print(result)
#         y_rom     = result.x
#         converged = result.success

#         self.counter = -1

#         # Return NaN array as output if failed (for the OpenSQP optimizer), otherwise return rom states
#         if converged:
#             print('\n\n Specified tolerance reached!')
#             # Update reduced jacobain to most recent state
#             print('Updating Jacobian to most recent state value...')
#             J_rom = self._compute_rom_residual_and_jacobian(y_rom, phi, W, D)[1]
#             self.reduced_jacobian = J_rom

#             output_vals['dafoam_rom_states'] = y_rom

#         else:
#             print('Warning: ROM iteration limit reached!')
#             output_vals['dafoam_rom_states'] = np.full((num_modes, ), np.nan)


#     # region _estimate_jac_mat_product
#     def _estimate_jac_mat_product(self, M, y_fom):
        
#         JM              = np.zeros_like(M)
#         r_fom_current   = self._compute_fom_residuals(y_fom)

#         eps_machine = np.finfo(float).eps
#         min_eps         = 1e-8
        
#         for i_column in range(M.shape[1]):
#             eps             = max(np.sqrt(eps_machine)*(1 + np.linalg.norm(M[:, i_column])), min_eps)
#             r_fom_perturb   = self._compute_fom_residuals(y_fom + eps * M[:, i_column])

#             JM[:, i_column] = (r_fom_perturb - r_fom_current)/eps
        
#         return JM


#     # region _compute_fom_residuals
#     def _compute_fom_residuals(self, y_fom):
#         dafoam_instance = self.dafoam_instance
        
#         # Save current state so we can revert to it later
#         hold_states = dafoam_instance.getStates()

#         # Set states and get residuals
#         dafoam_instance.setStates(y_fom)
#         r_fom = dafoam_instance.getResiduals()

#         # Revert to original states
#         dafoam_instance.setStates(hold_states)

#         return r_fom
    

#     # region _compute_rom_residual_and_jacobian
#     def _compute_rom_residual_and_jacobian(self, y_rom, phi, W, D, compute_jacobian=True):
#         dafoam_instance = self.dafoam_instance
#         rom_type  = self.rom_type
#         y_fom_ref = self.fom_ref_state

#         y_fom     = y_fom_ref + D * (phi @ y_rom)
#         r_fom     = self._compute_fom_residuals(y_fom)
#         r_fom_ref = self.fom_ref_residual

#         if rom_type.lower() == 'galerkin':
#             psi   = phi

#         elif rom_type.lower() == 'petrov-galerkin':
#             psi   = self._estimate_jac_mat_product(phi, y_fom)
        
#         r_rom = psi.T @ (W * r_fom)

#         width = 10
        
#         print(f"Step {self.counter}")
#         for i, vec in enumerate([y_rom, r_rom], start=1):
#             row = " ".join(f"{x:{width}.3e}" for x in vec)
#             print(f"{row}")

#         print(f'r_(k+1) - r_k/r_k: {(np.linalg.norm(r_rom) - np.linalg.norm(self.r_rom_old))/np.linalg.norm(self.r_rom_old)}')

        
#         self.r_rom_old = r_rom
#         self.counter += 1

#         if compute_jacobian:
#             J_rom = psi.T @ (W[:, None] * self._estimate_jac_mat_product(D[:, None] * phi, y_fom))
#             print(f'JTr/r: {np.linalg.norm(J_rom.T@r_rom)/np.linalg.norm(r_rom)}')
#             return r_rom, J_rom
#         else:
#             return r_rom










# import matplotlib.pyplot as plt
# # region DAFOAMROM4
# class DAFoamROM4(csdl.experimental.CustomImplicitOperation):
#     def __init__(self, 
#                  dafoam_instance,
#                  pod_modes:np.array         =None,
#                  fom_ref_state:np.array     =None, 
#                  tolerance                  =1e-6, 
#                  max_iters:int              =100, 
#                  update_jac_frequency:int   =10,
#                  num_initial_jac_updates:int=1, 
#                  weights:np.array           =None,
#                  rom_type:str               ='petrov-galerkin',
#                  root_method:str            ='lm',
#                  scaling_factors:np.array   =None):

#         super().__init__()
#         self.dafoam_instance        = dafoam_instance
#         self.tolerance              = tolerance
#         self.max_iters              = max_iters
#         self.solution_counter       = 1
#         self.update_jac_frequency   = update_jac_frequency if update_jac_frequency > 0 else np.nan
#         self.num_initial_jac_updates= num_initial_jac_updates
#         self.reduced_jacobian       = None
#         self.n_local_state_elements = dafoam_instance.getNLocalAdjointStates()
#         self.n_local_cells          = dafoam_instance.solver.getNLocalCells()

#         # Do some checks here to assign default values if none passed. Alert user if none are passed.
#         if weights is None:
#             print('No weights passed to DAFoamROM. Assuming no weighting used in POD mode computation...')
#             self.weights = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
#         else:
#             self.weights = weights

#         if scaling_factors is None:
#             print('No weights passed to DAFoamROM. Assuming no scaling factor used in POD mode computation...')
#             self.scaling_factors = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
#         else:
#             self.scaling_factors = scaling_factors

#         if fom_ref_state is None:
#             print('No FOM reference state passed to DAFoamROM. Assuming no mean/reference subtraction used in POD mode computation...')
#             self.fom_ref_state = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
#         else:
#             self.fom_ref_state = fom_ref_state
        
#         self.fom_ref_residual  = self._compute_fom_residuals(fom_ref_state)

#         if pod_modes is None:
#             print('No POD modes passed to DAFoamROM. Expecting POD modes as CSDL variable for the evaluate method...')
#         else:
#             print('POD modes passed to DAFoamROM. Assuming constant/global POD modes unless modes are passed to the evaluate method...')
        
#         self.pod_modes   = pod_modes

#         self.rom_type    = rom_type
#         self.root_method = root_method
#         self.counter     = 0

        
#     # region evaluate
#     def evaluate(self, dafoam_input_variables_group:csdl.VariableGroup, pod_modes:csdl.Variable=None, design_variable_configuration:csdl.Variable=None):
        
#         # Read daOptions to set proper inputs
#         inputDict = self.dafoam_instance.getOption("inputInfo")
#         for inputName in inputDict.keys():
#             if "solver" in inputDict[inputName]["components"]:
#                 self.declare_input(inputName, getattr(dafoam_input_variables_group, inputName))

#         # Set our basis as an input variable if not already set to a constant
#         if pod_modes is not None:
#             if self.pod_modes is None:
#                 print('POD modes passed to DAFoamROM evalute method. Neglecting constant/global modes passed in initialization.')
#             self.declare_input('pod_modes', pod_modes)
#             num_modes = pod_modes.value.shape[1]
#             self.pod_modes_is_csdl_var = True
#         else:
#             if self.pod_modes is None:
#                 TypeError('No POD modes assigned to ROM. Please pass a constant POD mode set during initialization, or a POD mode set in the evaluate section.')
#             else: # Constant POD mode case
#                 num_modes = self.pod_modes.shape[1]
#                 self.pod_modes_is_csdl_var = False
                
#                 # Check to make sure the POD modes are orthogonal under given weights
#                 print('Checking weighted inner product of supplied constant POD modes')
#                 weighted_inner_product = self.pod_modes.T @ (self.weights[:, None] * self.pod_modes)
#                 I                      = np.eye(weighted_inner_product.shape[0])
#                 fro_error              = np.linalg.norm(weighted_inner_product - I, ord='fro')
#                 max_entry_error        = np.max(np.abs(weighted_inner_product - I))

#                 print(f'Frobenius error: {fro_error}')
#                 print(f'Max entry error: {max_entry_error}')

#                 if fro_error > 1e-6:
#                     print(f'WARNING: high error detected (fro_error, {fro_error} > 1e-6)')         
        
#         # Set outputs
#         dafoam_rom_states = self.create_output('dafoam_rom_states', (num_modes,))

#         return dafoam_rom_states


#     # region solve_residual_equations
#     def solve_residual_equations(self, input_vals, output_vals):
#         from scipy.optimize import root, least_squares

#         dafoam_instance = self.dafoam_instance

#         # Solver parameters/options (not all have been exposed as options yet. Tweak here if necessary.)
#         root_method     = self.root_method
       
#         # Check if we're working with the pod modes as a constant or as a variable
#         pod_modes_is_csdl_var = self.pod_modes_is_csdl_var

#         # Some important quanitites
#         y_fom_ref  = self.fom_ref_state
#         W          = self.weights
#         phi        = input_vals['pod_modes'] if pod_modes_is_csdl_var else self.pod_modes
#         num_modes  = phi.shape[1]
        
#         # MPI stuff
#         # TODO: will have to make this all MPI compatible eventually
#         rank = dafoam_instance.rank

#         # Make sure solver is updated with the most recent input values
#         dafoam_instance.set_solver_input(input_vals)

#         # We also need to just calculate the residual for the AD mode to initialize vars like URes
#         # We do not print the residual for AD, though
#         # NOTE: This was taken from DAFoamSolver.solve_residual_equation
#         dafoam_instance.solverAD.calcPrimalResidualStatistics("calc")
        
#         # Initialize our ROM/FOM states and ROM/FOM residuals
#         y_rom = np.zeros((num_modes, ))
#         self.r_rom_old = y_rom.copy()
#         # phi_eff = D[:, None] * phi
#         # phi_hat = phi_eff.copy()
#         # for j in range(num_modes):
#         #     print(np.dot(phi_eff[:,j], W * phi_eff[:,j]))
#         #     s_j     = np.sqrt(np.dot(phi_eff[:,j], W * phi_eff[:,j]))
#         #     phi_hat[:, j] /= s_j
#         # print(phi_hat.shape)
#         # print(phi_hat.T @ (W[:, None] * phi_hat))
#         result  = root(lambda x: self._compute_rom_residual_and_jacobian(x, phi, W), y_rom, method=root_method, jac=True)

#         print(result)
#         y_rom     = result.x
#         converged = result.success

#         self.counter = -1

#         # Return NaN array as output if failed (for the OpenSQP optimizer), otherwise return rom states
#         if converged:
#             print('\n\n Specified tolerance reached!')
#             # Update reduced jacobain to most recent state
#             print('Updating Jacobian to most recent state value...')
#             J_rom = self._compute_rom_residual_and_jacobian(y_rom, phi, W)[1]
#             self.reduced_jacobian = J_rom

#             output_vals['dafoam_rom_states'] = y_rom

#         else:
#             print('Warning: ROM iteration limit reached!')
#             output_vals['dafoam_rom_states'] = np.full((num_modes, ), np.nan)


#     # region _estimate_jac_mat_product
#     def _estimate_jac_mat_product(self, M, y_fom):
        
#         JM              = np.zeros_like(M)
#         r_fom_current   = self._compute_fom_residuals(y_fom)

#         eps_machine = np.finfo(float).eps
#         min_eps         = 1e-8
        
#         for i_column in range(M.shape[1]):
#             eps             = max(np.sqrt(eps_machine)*(1 + np.linalg.norm(M[:, i_column])), min_eps)
#             r_fom_perturb   = self._compute_fom_residuals(y_fom + eps * M[:, i_column])

#             JM[:, i_column] = (r_fom_perturb - r_fom_current)/eps
        
#         return JM


#     # region _compute_fom_residuals
#     def _compute_fom_residuals(self, y_fom):
#         dafoam_instance = self.dafoam_instance
        
#         # Save current state so we can revert to it later
#         hold_states = dafoam_instance.getStates()

#         # Set states and get residuals
#         dafoam_instance.setStates(y_fom)
#         r_fom = dafoam_instance.getResiduals()

#         # Revert to original states
#         dafoam_instance.setStates(hold_states)

#         return r_fom
    

#     # region _compute_rom_residual_and_jacobian
#     def _compute_rom_residual_and_jacobian(self, y_rom, phi, W, compute_jacobian=True):
#         dafoam_instance = self.dafoam_instance
#         rom_type  = self.rom_type
#         y_fom_ref = self.fom_ref_state

#         y_fom     = y_fom_ref + self.scaling_factors * (phi @ y_rom)
#         r_fom     = self._compute_fom_residuals(y_fom)
#         r_fom_ref = self.fom_ref_residual

#         if rom_type.lower() == 'galerkin':
#             psi   = phi

#         elif rom_type.lower() == 'petrov-galerkin':
#             psi   = self._estimate_jac_mat_product(phi, y_fom)
        
#         r_rom = psi.T @ (W * r_fom)

#         width = 10
        
#         print(f"Step {self.counter}")
#         for i, vec in enumerate([y_rom, r_rom], start=1):
#             row = " ".join(f"{x:{width}.3e}" for x in vec)
#             print(f"{row}")

#         print(f'r_(k+1) - r_k/r_k: {(np.linalg.norm(r_rom) - np.linalg.norm(self.r_rom_old))/np.linalg.norm(self.r_rom_old)}')

        
#         self.r_rom_old = r_rom
#         self.counter += 1

#         if compute_jacobian:
#             J_rom = psi.T @ (W[:, None] * psi)
#             print(f'JTr/r: {np.linalg.norm(J_rom.T@r_rom)/np.linalg.norm(r_rom)}')
#             return r_rom, J_rom
#         else:
#             return r_rom
        





# import matplotlib.pyplot as plt
# # region DAFOAMROM4MPI
# class DAFoamROM4MPI(csdl.experimental.CustomImplicitOperation):
#     def __init__(self, 
#                  dafoam_instance,
#                  pod_modes:np.array         =None,
#                  fom_ref_state:np.array     =None, 
#                  tolerance                  =1e-6, 
#                  max_iters:int              =100, 
#                  update_jac_frequency:int   =10,
#                  num_initial_jac_updates:int=1, 
#                  weights:np.array           =None,
#                  rom_type:str               ='petrov-galerkin',
#                  root_method:str            ='lm',
#                  scaling_factors:np.array   =None):

#         super().__init__()
#         self.dafoam_instance        = dafoam_instance
#         self.tolerance              = tolerance
#         self.max_iters              = max_iters
#         self.solution_counter       = 1
#         self.update_jac_frequency   = update_jac_frequency if update_jac_frequency > 0 else np.nan
#         self.num_initial_jac_updates= num_initial_jac_updates
#         self.reduced_jacobian       = None
#         self.n_local_state_elements = dafoam_instance.getNLocalAdjointStates()
#         self.n_local_cells          = dafoam_instance.solver.getNLocalCells()

#         # Do some checks here to assign default values if none passed. Alert user if none are passed.
#         if weights is None:
#             print('No weights passed to DAFoamROM. Assuming no weighting used in POD mode computation...')
#             self.weights = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
#         else:
#             self.weights = weights

#         if scaling_factors is None:
#             print('No weights passed to DAFoamROM. Assuming no scaling factor used in POD mode computation...')
#             self.scaling_factors = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
#         else:
#             self.scaling_factors = scaling_factors

#         if fom_ref_state is None:
#             print('No FOM reference state passed to DAFoamROM. Assuming no mean/reference subtraction used in POD mode computation...')
#             self.fom_ref_state = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
#         else:
#             self.fom_ref_state = fom_ref_state
        
#         self.fom_ref_residual  = self._compute_fom_residuals(fom_ref_state)

#         if pod_modes is None:
#             print('No POD modes passed to DAFoamROM. Expecting POD modes as CSDL variable for the evaluate method...')
#         else:
#             print('POD modes passed to DAFoamROM. Assuming constant/global POD modes unless modes are passed to the evaluate method...')
        
#         self.pod_modes   = pod_modes

#         self.rom_type    = rom_type
#         self.root_method = root_method
#         self.counter     = 0

        
#     # region evaluate
#     def evaluate(self, dafoam_input_variables_group:csdl.VariableGroup, pod_modes:csdl.Variable=None, design_variable_configuration:csdl.Variable=None):
        
#         # Read daOptions to set proper inputs
#         inputDict = self.dafoam_instance.getOption("inputInfo")
#         for inputName in inputDict.keys():
#             if "solver" in inputDict[inputName]["components"]:
#                 self.declare_input(inputName, getattr(dafoam_input_variables_group, inputName))

#         # Set our basis as an input variable if not already set to a constant
#         if pod_modes is not None:
#             if self.pod_modes is None:
#                 print('POD modes passed to DAFoamROM evalute method. Neglecting constant/global modes passed in initialization.')
#             self.declare_input('pod_modes', pod_modes)
#             num_modes = pod_modes.value.shape[1]
#             self.pod_modes_is_csdl_var = True
#         else:
#             if self.pod_modes is None:
#                 TypeError('No POD modes assigned to ROM. Please pass a constant POD mode set during initialization, or a POD mode set in the evaluate section.')
#             else: # Constant POD mode case
#                 num_modes = self.pod_modes.shape[1]
#                 self.pod_modes_is_csdl_var = False
                
#                 # Check to make sure the POD modes are orthogonal under given weights
#                 print('Checking weighted inner product of supplied constant POD modes')
#                 weighted_inner_product = self.pod_modes.T @ (self.weights[:, None] * self.pod_modes)
#                 I                      = np.eye(weighted_inner_product.shape[0])
#                 fro_error              = np.linalg.norm(weighted_inner_product - I, ord='fro')
#                 max_entry_error        = np.max(np.abs(weighted_inner_product - I))

#                 print(f'Frobenius error: {fro_error}')
#                 print(f'Max entry error: {max_entry_error}')

#                 if fro_error > 1e-6:
#                     print(f'WARNING: high error detected (fro_error, {fro_error} > 1e-6)')         
        
#         # Set outputs
#         dafoam_rom_states = self.create_output('dafoam_rom_states', (num_modes,))

#         return dafoam_rom_states


#     # region solve_residual_equations
#     def solve_residual_equations(self, input_vals, output_vals):
#         from scipy.optimize import root, least_squares

#         dafoam_instance = self.dafoam_instance

#         # Solver parameters/options (not all have been exposed as options yet. Tweak here if necessary.)
#         root_method     = self.root_method
       
#         # Check if we're working with the pod modes as a constant or as a variable
#         pod_modes_is_csdl_var = self.pod_modes_is_csdl_var

#         # Some important quanitites
#         y_fom_ref  = self.fom_ref_state
#         W          = self.weights
#         phi        = input_vals['pod_modes'] if pod_modes_is_csdl_var else self.pod_modes
#         num_modes  = phi.shape[1]
        
#         # MPI stuff
#         # TODO: will have to make this all MPI compatible eventually
#         rank = dafoam_instance.rank

#         # Make sure solver is updated with the most recent input values
#         dafoam_instance.set_solver_input(input_vals)

#         # We also need to just calculate the residual for the AD mode to initialize vars like URes
#         # We do not print the residual for AD, though
#         # NOTE: This was taken from DAFoamSolver.solve_residual_equation
#         dafoam_instance.solverAD.calcPrimalResidualStatistics("calc")
        
#         # Initialize our ROM/FOM states and ROM/FOM residuals
#         y_rom = np.zeros((num_modes, ))
#         self.r_rom_old = y_rom.copy()
#         # phi_eff = D[:, None] * phi
#         # phi_hat = phi_eff.copy()
#         # for j in range(num_modes):
#         #     print(np.dot(phi_eff[:,j], W * phi_eff[:,j]))
#         #     s_j     = np.sqrt(np.dot(phi_eff[:,j], W * phi_eff[:,j]))
#         #     phi_hat[:, j] /= s_j
#         # print(phi_hat.shape)
#         # print(phi_hat.T @ (W[:, None] * phi_hat))
#         result  = root(lambda x: self._compute_rom_residual_and_jacobian(x, phi, W), y_rom, method=root_method, jac=True)

#         print(result)
#         y_rom     = result.x
#         converged = result.success

#         self.counter = -1

#         # Return NaN array as output if failed (for the OpenSQP optimizer), otherwise return rom states
#         if converged:
#             print('\n\n Specified tolerance reached!')
#             # Update reduced jacobain to most recent state
#             print('Updating Jacobian to most recent state value...')
#             J_rom = self._compute_rom_residual_and_jacobian(y_rom, phi, W)[1]
#             self.reduced_jacobian = J_rom

#             output_vals['dafoam_rom_states'] = y_rom

#         else:
#             print('Warning: ROM iteration limit reached!')
#             output_vals['dafoam_rom_states'] = np.full((num_modes, ), np.nan)


#     # region _estimate_jac_mat_product
#     def _estimate_jac_mat_product(self, M, y_fom):
        
#         JM              = np.zeros_like(M)
#         r_fom_current   = self._compute_fom_residuals(y_fom)

#         eps_machine = np.finfo(float).eps
#         min_eps         = 1e-8
        
#         for i_column in range(M.shape[1]):
#             eps             = max(np.sqrt(eps_machine)*(1 + np.linalg.norm(M[:, i_column])), min_eps)
#             r_fom_perturb   = self._compute_fom_residuals(y_fom + eps * M[:, i_column])

#             JM[:, i_column] = (r_fom_perturb - r_fom_current)/eps
        
#         return JM


#     # region _compute_fom_residuals
#     def _compute_fom_residuals(self, y_fom):
#         dafoam_instance = self.dafoam_instance
        
#         # Save current state so we can revert to it later
#         hold_states = dafoam_instance.getStates()

#         # Set states and get residuals
#         dafoam_instance.setStates(y_fom)
#         r_fom = dafoam_instance.getResiduals()

#         # Revert to original states
#         dafoam_instance.setStates(hold_states)

#         return r_fom
    

#     # region _compute_rom_residual_and_jacobian
#     def _compute_rom_residual_and_jacobian(self, y_rom, phi, W, compute_jacobian=True):
#         dafoam_instance = self.dafoam_instance
#         rom_type  = self.rom_type
#         y_fom_ref = self.fom_ref_state

#         y_fom     = y_fom_ref + self.scaling_factors * (phi @ y_rom)
#         r_fom     = self._compute_fom_residuals(y_fom)
#         r_fom_ref = self.fom_ref_residual

#         if rom_type.lower() == 'galerkin':
#             psi   = phi

#         elif rom_type.lower() == 'petrov-galerkin':
#             psi   = self._estimate_jac_mat_product(phi, y_fom)
        
#         r_rom = psi.T @ (W * r_fom)

#         width = 10
        
#         print(f"Step {self.counter}")
#         for i, vec in enumerate([y_rom, r_rom], start=1):
#             row = " ".join(f"{x:{width}.3e}" for x in vec)
#             print(f"{row}")

#         print(f'r_(k+1) - r_k/r_k: {(np.linalg.norm(r_rom) - np.linalg.norm(self.r_rom_old))/np.linalg.norm(self.r_rom_old)}')

        
#         self.r_rom_old = r_rom
#         self.counter += 1

#         if compute_jacobian:
#             J_rom = psi.T @ (W[:, None] * psi)
#             print(f'JTr/r: {np.linalg.norm(J_rom.T@r_rom)/np.linalg.norm(r_rom)}')
#             return r_rom, J_rom
#         else:
#             return r_rom