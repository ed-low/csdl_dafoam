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