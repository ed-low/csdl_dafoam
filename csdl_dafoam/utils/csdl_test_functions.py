import numpy as np
import matplotlib.pyplot as plt
import pprint
from csdl_dafoam.utils.runscript_helper_functions import quiet_barrier

# MPI
from mpi4py import MPI


# region CUSTOMCOMPONENTCHECKS
class CustomComponentChecks():
    def __init__(self,
                 component, 
                 fd_step=1e-6, 
                 mode="rev", 
                 comm=None, 
                 random_seed=None,
                 random_scalar=1):
        
        self.component  = component
        self.fd_step    = fd_step
        self.mode       = mode
        self.comm       = comm
        self.random_seed    = random_seed
        self.random_scalar  = random_scalar

        # Setup some comm stuff, if available
        if comm is not None:
            self.rank       = comm.Get_rank()
            self.comm_size  = comm.Get_size()
        else:
            self.rank       = 0
            self.comm_size  = 1
        
        # Check to see if we have an explicit component or an implicit component
        self.component_info = self._determine_component_info(component)


    # region run_jacvec_fd_sweep
    def run_jacvec_fd_sweep(self, input_vals=None, v=None, w=None, eps_test_values=[1e-3, 1e-5, 1e-7, 1e-9], mode=None, random_seed=None, random_scalar=None, plot_on_all_ranks=False):
        
        # Get our user or default values
        component     = self.component
        potential_distributed_output = self.component_info["output_distributed"]
        potential_distributed_input  = self.component_info["input_distributed"]

        random_scalar = self.random_scalar if self.random_scalar is not None and random_scalar is None else 1
        random_seed   = self.random_seed if self.random_seed is not None and random_seed is None else None
        mode          = self.mode if self.mode is not None and mode is None else 'rev'

        output_vals_base    = {k: vv.value for k, vv in component.output_dict.items()} 
        input_vals_base     = {k: vv.value for k, vv in component.input_dict.items()} if input_vals is None else input_vals

        v = v if v is not None else self._generate_random_like(output_vals_base, potential_distributed_output, random_scalar=random_scalar, random_seed=random_seed) 
        w = w if w is not None else self._generate_random_like(input_vals_base,  potential_distributed_input,  random_scalar=random_scalar, random_seed=random_seed)
        
        err = np.zeros_like(eps_test_values)
        lhs = np.zeros_like(eps_test_values)
        rhs = np.zeros_like(eps_test_values)
        i = 0
        for eps in eps_test_values:
            lhs[i], rhs[i], err[i] = self.check_jacvec_product(eps=eps, v=v, w=w, input_vals=input_vals, mode=mode, random_seed=random_seed)
            i += 1 

        self.print0("\n------------------------------------")
        self.print0("JACVEC_PRODUCT SWEEP SUMMARY")
        self.print0("------------------------------------")
        if self.rank == 0:
            print("Component info:")
            pprint.pprint(self.component_info)

        # We'll print out the summaries per rank (for easy rank comparison)
        for result_idx in range(err.size):
            self.print0(f"\nFD Step: {eps_test_values[result_idx]:<15.5E}")
            self.print0(f"   {'Rank':^8} {'LHS':^15} {'RHS':^15} {'ERR':^15}")
            lhs_all = self.comm.gather(lhs[result_idx], root=0)
            rhs_all = self.comm.gather(rhs[result_idx], root=0)
            err_all = self.comm.gather(err[result_idx], root=0)

            for r in range(self.comm_size):
                if self.rank == 0:
                    print(f"   {r:^8} {lhs_all[r]:>15.5E} {rhs_all[r]:>15.5E} {err_all[r]:>15.5E}")
        
        self.print0("------------------------------------")
                
        plot_on_this_rank = self.rank == 0 or plot_on_all_ranks

        if plot_on_this_rank:
            plt.rcParams['text.usetex'] = True
            plt.figure()

            plt.loglog(eps_test_values, err, marker='o')
            plt.title(r'jacvec_product vs FD ($w^T J^T v = v^T J w$)')
            plt.xlabel(r'Stepsize, $\epsilon$')
            plt.ylabel(r'Error, $\frac{lhs - rhs}{rhs}$')
            plt.grid(visible=True)
            plt.show()

        if self.comm is not None:
            quiet_barrier(self.comm)


    # region run_inverse_jacobian_fd_sweep
    def run_inverse_jacobian_fd_sweep(self, input_vals=None, v=None, eps_test_values=[1e-3, 1e-5, 1e-7, 1e-9], mode=None, random_seed=None, random_scalar=None, plot_on_all_ranks=False):
        
        # Get our user or default values
        component     = self.component
        potential_distributed_output = self.component_info["output_distributed"]

        random_scalar = self.random_scalar if self.random_scalar is not None and random_scalar is None else 1
        random_seed   = self.random_seed if self.random_seed is not None and random_seed is None else None
        mode          = self.mode if self.mode is not None and mode is None else 'rev'

        output_vals_base    = {k: vv.value for k, vv in component.output_dict.items()} 

        v = v if v is not None else self._generate_random_like(output_vals_base, potential_distributed_output, random_scalar=random_scalar, random_seed=random_seed) 
        
        err = np.zeros_like(eps_test_values)
        lhs = np.zeros_like(eps_test_values)
        rhs = np.zeros_like(eps_test_values)
        i = 0
        for eps in eps_test_values:
            lhs[i], rhs[i], err[i] = self.check_inverse_jacobian(eps=eps, v=v, input_vals=input_vals, mode=mode, random_seed=random_seed, random_scalar=random_scalar)
            i += 1 

        self.print0("\n------------------------------------")
        self.print0("INVERSE JACOBIAN SWEEP SUMMARY")
        self.print0("------------------------------------")
        if self.rank == 0:
            print("Component info:")
            pprint.pprint(self.component_info)

        # We'll print out the summaries per rank (for easy rank comparison)
        for result_idx in range(err.size):
            self.print0(f"\nFD Step: {eps_test_values[result_idx]:<15.5E}")
            self.print0(f"   {'Rank':^8} {'LHS':^15} {'RHS':^15} {'ERR':^15}")
            lhs_all = self.comm.gather(lhs[result_idx], root=0)
            rhs_all = self.comm.gather(rhs[result_idx], root=0)
            err_all = self.comm.gather(err[result_idx], root=0)

            for r in range(self.comm_size):
                if self.rank == 0:
                    print(f"   {r:^8} {lhs_all[r]:>15.5E} {rhs_all[r]:>15.5E} {err_all[r]:>15.5E}")
        
        self.print0("------------------------------------")
                
        plot_on_this_rank = self.rank == 0 or plot_on_all_ranks

        if plot_on_this_rank:
            plt.rcParams['text.usetex'] = True
            plt.figure()

            plt.loglog(eps_test_values, err, marker='o')
            plt.title(r'apply_inverse_jacobian vs FD ($v^T v = (J^{-T} v)^T (J v)$)')
            plt.xlabel(r'Stepsize, $\epsilon$')
            plt.ylabel(r'Error, $\frac{lhs - rhs}{rhs}$')
            plt.grid(visible=True)
            plt.show()

        if self.comm is not None:
            quiet_barrier(self.comm)


    # region check_jacvec_product
    def check_jacvec_product(self, v=None, w=None, input_vals=None, eps=1e-6, mode='rev', random_seed=None, random_scalar=None):
        """
        Generic test for compute_jacvec_product in a CSDL component. Tests via finite differenece
        w^T J^T v = v^T J w
        where J = dR/dx for implicit component, dF/dx for explicit component

        Parameters
        ----------
        component : CSDL component (explicit or implicit)
            For explicit: Must have compute and compute_jacvec_product.
            For implicit: Must have solve_residual_equations, evaluate_residuals, and compute_jacvec_product.
        input_vals : dict
            Dictionary of input arrays (keys must match component input names).
        v : dict
            Direction vector for outputs.
        w : dict
            Direction vector for inputs (same structure as input_vals).
        eps : float
            Step size for finite differences.

        Returns
        -------
        float
            Left hand side value.
        float
            Right hand side value.
        float
            Relative error between the two scalar evaluations.
        """

        component = self.component
        comm      = self.comm
        potential_distributed_input  = self.component_info["input_distributed"]
        potential_distributed_output = self.component_info["output_distributed"]
        random_scalar = self.random_scalar if random_scalar is None and self.random_scalar is not None else 1
    
        # --- Get inputs and compute residuals/outputs at base state ---
        input_vals  = input_vals if input_vals is not None else {k: vv.value for k, vv in component.input_dict.items()}
        
        if hasattr(component, 'solve_residual_equations'):
            residual_vals  = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
            output_vals    = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
            component.solve_residual_equations(input_vals, output_vals)
            component.evaluate_residuals(input_vals, output_vals, residual_vals)
            component_type = 'implicit'
        
        elif hasattr(component, 'compute'):
            output_vals    = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
            component.compute(input_vals, output_vals)
            component_type = 'explicit'

        else:
            TypeError('the supplied component doesn''t seem to be a CSDL Implicit or Explicit component')

        # --- Generate/assign variables ---
        v = v if v is not None else self._generate_random_like(output_vals, potential_distributed_output, random_scalar=random_scalar, random_seed=random_seed) 
        w = w if w is not None else self._generate_random_like(input_vals,  potential_distributed_input,  random_scalar=random_scalar, random_seed=random_seed) 
        
        if mode == 'fwd':
            raise NotImplementedError('forward mode has not been implemented for test_jacvec_product')

        if mode == 'rev':
            # --- Compute J w (via finite diff) ---
            # Perturb input in the w-direction
            perturbed_input_vals = {k: np.copy(vv) for k, vv in input_vals.items()}
            for in_name in input_vals.keys():
                perturbed_input_vals[in_name] += eps*w[in_name]

            # Compute perturbed outputs/residuals
            if component_type == 'explicit':
                perturbed_output_vals = {k: np.zeros_like(vv) for k, vv in output_vals.items()}
                component.compute(perturbed_input_vals, perturbed_output_vals)
                J_w = {name: (perturbed_output_vals[name] - output_vals[name]) / eps for name in output_vals.keys()}

            else:
                perturbed_residual_vals = {k: np.zeros_like(vv) for k, vv in residual_vals.items()}
                component.evaluate_residuals(perturbed_input_vals, output_vals, perturbed_residual_vals)
                J_w = {name: (perturbed_residual_vals[name] - residual_vals[name]) / eps for name in residual_vals.keys()}

            # Assemble RHS
            rhs = 0
            for name in v:
                name_dot = np.vdot(v[name], J_w[name])
                if comm is not None:
                    if potential_distributed_output[name]:
                        name_dot = comm.allreduce(name_dot, op=MPI.SUM)
                rhs += name_dot

            # --- Compute J^T v (adjoint mode) ---
            # Initialize our product to zeros (this is our J^T v)
            JT_v    = {name: np.zeros_like(val) for name, val in input_vals.items()}

            if component_type == 'explicit':   
                component.compute_jacvec_product(input_vals, output_vals, JT_v, v, 'rev')

            else:
                d_outputs   = {name: np.zeros_like(val) for name, val in residual_vals.items()}
                component.compute_jacvec_product(input_vals, output_vals, JT_v, d_outputs, v, 'rev')

            # For global inputs, check if the component already allreduced internally
            # by testing whether all ranks hold the same value.
            # If consistent -> already complete, skip allreduce
            # If inconsistent -> partial local contributions, allreduce needed
            if comm is not None:
                for name in JT_v:
                    if not potential_distributed_input[name]:
                        if not self._arrays_replicated_fast(JT_v[name]):
                            JT_v[name] = comm.allreduce(JT_v[name], op=MPI.SUM)
                        else:
                            self.print0(f'Component appears to allreduce internally for {name} skipping J^T*v allreduce. Consider uncommenting /= comm_size if this is not the case.')
                            #JT_v[name] /= self.comm_size


            # Assemble LHS
            lhs = 0
            for name in w:
                name_dot = np.vdot(w[name], JT_v[name])
                if comm is not None:
                    if potential_distributed_input[name]:
                        name_dot = comm.allreduce(name_dot, op=MPI.SUM)
                lhs += name_dot

        else:
            raise ValueError(f'"{mode}" not recognized. Only support "fwd" and "rev" modes') 

        # --- Compare ---
        err = np.abs((lhs - rhs) / (rhs))

        self.print0(f"LHS (w^T J^T v): {lhs}")
        self.print0(f"RHS (v^T J w): {rhs}")
        self.print0(f"Relative error: {err:.3e}")

        return lhs, rhs, err


    # region check_inverse_jacobian
    def check_inverse_jacobian(self, input_vals=None, v=None, eps=1e-6, mode='rev', random_seed=None, random_scalar=None):
        """
        Generic test for apply_inverse_jacobian in a CSDL implicit component. Tests via finite difference
        v^T v = (J^-T v)^T (J v)
        where J = dR/dy
        J^-T v is computed via apply_inverse_jacobian and (J v) is computed via finite difference

        Parameters
        ----------
        component : CSDL component (explicit or implicit)
            For explicit: Must have compute and compute_jacvec_product.
            For implicit: Must have solve_residual_equations, evaluate_residuals, and compute_jacvec_product.
        input_vals : dict
            Dictionary of input arrays around which we'll center the evaluation (keys must match component input names).
        v : dict
            Direction vector for residuals.
        eps : float
            Step size for finite differences.

        Returns
        -------
        float
            Left hand side value.
        float
            Right hand side value.
        float
            Relative error between the two scalar evaluations.
        """
        comm      = self.comm
        component = self.component
        potential_distributed_output = self.component_info["output_distributed"]
        potential_distributed_input  = self.component_info["input_distributed"]

        # --- Get inputs and compute residuals/outputs at base state ---
        input_vals      = input_vals if input_vals is not None else {k: vv.value for k, vv in component.input_dict.items()}
        residual_vals   = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
        output_vals     = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
        component.solve_residual_equations(input_vals, output_vals)
        component.evaluate_residuals(input_vals, output_vals, residual_vals)
        
        # --- Generate/assign variables ---
        v = v if v is not None else self._generate_random_like(value_dict=output_vals, distributed_dict=potential_distributed_output, random_seed=random_seed, random_scalar=random_scalar)

        if mode == 'fwd':
            raise NotImplementedError('forward mode has not been implemented for test_jacvec_product')

        if mode == 'rev':
            # --- Compute J v (via finite diff) ---
            # Perturb input in the v-direction
            perturbed_output_vals = {k: np.copy(vv) for k, vv in output_vals.items()}
            for key in output_vals:
                perturbed_output_vals[key] += eps*v[key]

            # Compute perturbed residuals
            perturbed_residual_vals = {k: np.zeros_like(vv) for k, vv in residual_vals.items()}
            component.evaluate_residuals(input_vals, perturbed_output_vals, perturbed_residual_vals)
            J_v = {name: (perturbed_residual_vals[name] - residual_vals[name]) / eps for name in residual_vals.keys()}

            # --- Compute J^-T v (adjoint mode) ---
            # Initialize our product to zeros (this is our J^-T v)
            JinvT_v    = {name: np.zeros_like(val) for name, val in residual_vals.items()}
            component.evaluate_residuals(input_vals, output_vals, residual_vals) ### DEBUG Adding this so that the states are reverted back to the original output    
            component.apply_inverse_jacobian(input_vals, output_vals, v, JinvT_v, 'rev')
            
            if comm is not None:
                for name in JinvT_v:
                    if not potential_distributed_output[name]:
                        if not self._arrays_replicated_fast(JinvT_v[name]):
                            JinvT_v[name] = comm.allreduce(JinvT_v[name], op=MPI.SUM)
                        else:
                            self.print0(f'Component appears to allreduce internally for {name} skipping J^-T*v allreduce. Consider uncommenting /= comm_size if this is not the case.')
                            #JinvT_v[name] /= self.comm_size

            # Assemble LHS
            lhs = 0
            for name, val in v.items():
                name_dot = np.vdot(val, val)
                if comm is not None:
                    if potential_distributed_output[name]:
                        name_dot = comm.allreduce(name_dot, op=MPI.SUM)
                lhs += name_dot

            # Assemble RHS
            rhs = 0
            for name in J_v.keys():
                name_dot = np.vdot(JinvT_v[name], J_v[name])
                if comm is not None:
                    if potential_distributed_output[name]:
                        name_dot = comm.allreduce(name_dot, op=MPI.SUM)
                rhs += name_dot

        # --- Compare ---
        err = np.abs(lhs - rhs) / (rhs)

        self.print0(f"LHS (v^T v): {lhs}")
        self.print0(f"RHS ((J^-T v)^T (J v)): {rhs}")
        self.print0(f"Relative error: {err:.3e}")

        return lhs, rhs, err


    # region _arrays_replicated_fast
    def _arrays_replicated_fast(self, x):
        comm = self.comm
        local_min = np.min(x)
        local_max = np.max(x)
        local_sum = np.sum(x)

        min_equal = comm.allreduce(local_min, MPI.MIN) == \
                    comm.allreduce(local_min, MPI.MAX)

        max_equal = comm.allreduce(local_max, MPI.MIN) == \
                    comm.allreduce(local_max, MPI.MAX)

        sum_equal = comm.allreduce(local_sum, MPI.MIN) == \
                    comm.allreduce(local_sum, MPI.MAX)

        return min_equal and max_equal and sum_equal


    # region _check_if_distributed
    def _check_if_distributed(self, x):
        comm = self.comm
        # We'll first compare the size of the input among ranks (cheaper than content comparison) ----
        local_shape = x.shape
        all_shapes  = comm.allgather(local_shape)
        if len(set(all_shapes)) > 1:
            return True

        # If same size, compare contents
        if not self._arrays_replicated_fast(x):
            return True

        return False


    # region _generate_random_like
    def _generate_random_like(self, value_dict, distributed_dict=None, random_seed=None, random_scalar=1):
        comm            = self.comm
        random_scalar   = 1 if random_scalar is None else random_scalar
        out_dict = {}

        if random_seed is not None:
            np.random.seed(random_seed)

        for name, vv in value_dict.items():
            shape = vv.shape

            if comm is None or distributed_dict is None:
                rand_array = np.random.rand(*shape)

            else: 
                # If the variable is of a distributed type then we'll let each rank do the sample
                # (For a non-comm case, this will be the same as well)
                if distributed_dict[name]:
                    rand_array = np.random.rand(*shape)

                # Otherwise, we'll have to copy it
                else:
                    if comm.Get_rank() == 0:
                        rand_array = np.random.rand(*shape)
                    else:
                        rand_array = np.empty(shape, dtype=vv.dtype)

                    comm.Bcast(rand_array, root=0)

            out_dict[name] = random_scalar * rand_array * vv
        
        return out_dict


    # region _determine_component_info
    def _determine_component_info(self, component):
        comm            = self.comm
        component_info  = {}
        
        # We'll first define some local functions here (mainly for repeated operations)
        def has_method_check(local_component, method_name):
            check_result = hasattr(local_component, method_name)
            self.print0(f"      Has {f'{method_name}?':<25} {check_result}")
            return check_result
        
        def check_variable_distribution(local_component, var_type):
            self.print0(f"   {var_type.title()}s")
            potential_distributed_dict = {}
            for name, vv in getattr(local_component, f"{var_type}_dict").items():
                val = vv.value
                if self._check_if_distributed(val):
                    potential_distributed_dict[name] = True
                    if comm.Get_rank() == 0:
                        self.print0(f"      {name:<25} distributed")
                else:
                    self.print0(f"      {name:<25} global")
                    potential_distributed_dict[name] = False
            
            return potential_distributed_dict

        self.print0("------------------------------------")
        self.print0("COMPONENT CHECK")
        self.print0("------------------------------------")

        # --- Check the component methods ---
        self.print0('-Methods check')
        self.print0("   Implicit component methods:")
        has_solve_residual_equations    = has_method_check(component, "solve_residual_equations")
        has_apply_inverse_jacobian      = has_method_check(component, "apply_inverse_jacobian")
        has_evaluate_residuals          = has_method_check(component, "evaluate_residuals")

        self.print0("   Explicit component methods:")
        has_compute                     = has_method_check(component, "compute")

        self.print0("   Shared methods:")
        has_compute_jacvec_product      = has_method_check(component, "compute_jacvec_product")

        appears_implicit     = has_apply_inverse_jacobian or has_solve_residual_equations or has_evaluate_residuals
        appears_explicit    = has_compute

        if appears_explicit:
            if appears_implicit:
                component_info["type"] = "implicit"
                self.print0("\n**Methods signature best matches IMPLICIT COMPONENT type.\n")
            else:
                component_info["type"] = "explicit"
                self.print0("\n**Methods signature best matches EXPLICIT COMPONENT type.\n")
        else:
            raise TypeError("This doesn't appear to be an implicit or explicit component")

        
        component_info["has_solve_residual_equations"]  = has_solve_residual_equations
        component_info["has_apply_inverse_jacobian"]    = has_apply_inverse_jacobian
        component_info["has_evaluate_residuals"]        = has_evaluate_residuals
        component_info["has_compute"]                   = has_compute
        component_info["has_compute_jacvec_product"]    = has_compute_jacvec_product

        # --- If we have a communicator, we can check for distributed variables ---
        if comm is not None:
            self.print0("-Variable distribution check")
        component_info["input_distributed"]  = check_variable_distribution(component, "input") if comm is not None else None
        component_info["output_distributed"] = check_variable_distribution(component, "output") if comm is not None else None

        self.print0("------------------------------------")

        return component_info


    # region print0
    def print0(self, statement, **kwargs):
        if self.rank == 0:
            print(statement, **kwargs)





# # region TEST_JACVEC_PRODUCT
# def test_jacvec_product(component, v=None, w=None, input_vals=None, eps=1e-6, mode='rev', comm=None, random_seed=None):
#     """
#     Generic test for compute_jacvec_product in a CSDL component. Tests via finite differenece
#     w^T J^T v = v^T J w
#     where J = dR/dx for implicit component, dF/dx for explicit component

#     Parameters
#     ----------
#     component : CSDL component (explicit or implicit)
#         For explicit: Must have compute and compute_jacvec_product.
#         For implicit: Must have solve_residual_equations, evaluate_residuals, and compute_jacvec_product.
#     input_vals : dict
#         Dictionary of input arrays (keys must match component input names).
#     v : dict
#         Direction vector for outputs.
#     w : dict
#         Direction vector for inputs (same structure as input_vals).
#     eps : float
#         Step size for finite differences.

#     Returns
#     -------
#     float
#         Left hand side value.
#     float
#         Right hand side value.
#     float
#         Relative error between the two scalar evaluations.
#     """   
  
#     # --- Get inputs and compute residuals/outputs at base state ---
#     input_vals  = input_vals if input_vals is not None else {k: vv.value for k, vv in component.input_dict.items()}
    
#     if hasattr(component, 'solve_residual_equations'):
#         residual_vals  = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
#         output_vals    = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
#         component.solve_residual_equations(input_vals, output_vals)
#         component.evaluate_residuals(input_vals, output_vals, residual_vals)
#         component_type = 'implicit'
    
#     elif hasattr(component, 'compute'):
#         output_vals    = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
#         component.compute(input_vals, output_vals)
#         component_type = 'explicit'

#     else:
#         TypeError('the supplied component doesn''t seem to be a CSDL Implicit or Explicit component')

#     # --- Check inputs and outputs for potential distributed variables ---
#     potential_distributed_input  = None
#     potential_distributed_output = None
#     if comm is not None:
#         potential_distributed_input = {}
#         for name, val in input_vals.items():
#             if check_if_distributed(val, comm):
#                 potential_distributed_input[name] = True
#                 if comm.Get_rank() == 0:
#                     print(f"{name} appears to be a distributed input. Assuming this is the case...")
#             else:
#                 potential_distributed_input[name] = False

#         potential_distributed_output = {}
#         for name, val in output_vals.items():
#             if check_if_distributed(val, comm):
#                 potential_distributed_output[name] = True
#                 if comm.Get_rank() == 0:
#                     print(f"{name} appears to be a distributed output. Assuming this is the case...")
#             else:
#                 potential_distributed_output[name] = False

#     # --- Generate/assign variables ---
#     v = v if v is not None else generate_random_like(output_vals, potential_distributed_output, scalar=1, comm=comm, random_seed=random_seed) #{k: np.random.rand(*vv.value.shape)*vv.value for k, vv in component.output_dict.items()}
#     w = w if w is not None else generate_random_like(input_vals,  potential_distributed_input,  scalar=1, comm=comm, random_seed=random_seed) #{k: np.random.rand(*vv.value.shape)*vv.value for k, vv in component.input_dict.items()}
    
#     if mode == 'fwd':
#         raise NotImplementedError('forward mode has not been implemented for test_jacvec_product')

#     if mode == 'rev':
#         # --- Compute J w (via finite diff) ---
#         # Perturb input in the w-direction
#         perturbed_input_vals = {k: np.copy(vv) for k, vv in input_vals.items()}
#         for in_name in input_vals.keys():
#             perturbed_input_vals[in_name] += eps*w[in_name]

#         # Compute perturbed outputs/residuals
#         if component_type == 'explicit':
#             perturbed_output_vals = {k: np.zeros_like(vv) for k, vv in output_vals.items()}
#             component.compute(perturbed_input_vals, perturbed_output_vals)
#             J_w = {name: (perturbed_output_vals[name] - output_vals[name]) / eps for name in output_vals.keys()}

#         else:
#             perturbed_residual_vals = {k: np.zeros_like(vv) for k, vv in residual_vals.items()}
#             component.evaluate_residuals(perturbed_input_vals, output_vals, perturbed_residual_vals)
#             J_w = {name: (perturbed_residual_vals[name] - residual_vals[name]) / eps for name in residual_vals.keys()}

#         # Assemble RHS
#         # New
#         rhs = 0
#         for name in v:
#             name_dot = np.vdot(v[name], J_w[name])
#             if comm is not None:
#                 if potential_distributed_output[name]:
#                     name_dot = comm.allreduce(name_dot, op=MPI.SUM)
#             rhs += name_dot
#         # Old
#         # rhs = sum(np.vdot(v[name], J_w[name]) for name in v)

#         # --- Compute J^T v (adjoint mode) ---
#         # Initialize our product to zeros (this is our J^T v)
#         JT_v    = {name: np.zeros_like(val) for name, val in input_vals.items()}

#         if component_type == 'explicit':   
#             component.compute_jacvec_product(input_vals, output_vals, JT_v, v, 'rev')

#         else:
#             d_outputs   = {name: np.zeros_like(val) for name, val in residual_vals.items()}
#             component.compute_jacvec_product(input_vals, output_vals, JT_v, d_outputs, v, 'rev')

#         # Assemble LHS
#         # New
#         lhs = 0
#         for name in w:
#             name_dot = np.vdot(w[name], JT_v[name])
#             if comm is not None:
#                 if potential_distributed_input[name]:
#                     name_dot = comm.allreduce(name_dot, op=MPI.SUM)
#             lhs += name_dot
#         # Old
#         # lhs = sum(np.vdot(w[name], JT_v[name]) for name in input_vals)

#     else:
#         raise ValueError(f'"{mode}" not recognized. Only support "fwd" and "rev" modes') 

#     # --- Compare ---
#     err = np.abs((lhs - rhs) / (rhs))

#     print(f"LHS (w^T J^T v): {lhs}")
#     print(f"RHS (v^T J w): {rhs}")
#     print(f"Relative error: {err:.3e}")

#     return lhs, rhs, err



# # region TEST_INVERSE_JACIBIAN
# def test_inverse_jacobian(component, input_vals, v, eps=1e-6, mode='rev'):
#     """
#     Generic test for apply_inverse_jacobian in a CSDL implicit component. Tests via finite difference
#     v^T v = (J^-T v)^T (J v)
#     where J = dR/dy
#     J^-T v is computed via apply_inverse_jacobian and (J v) is computed via finite difference

#     Parameters
#     ----------
#     component : CSDL component (explicit or implicit)
#         For explicit: Must have compute and compute_jacvec_product.
#         For implicit: Must have solve_residual_equations, evaluate_residuals, and compute_jacvec_product.
#     input_vals : dict
#         Dictionary of input arrays around which we'll center the evaluation (keys must match component input names).
#     v : dict
#         Direction vector for residuals.
#     eps : float
#         Step size for finite differences.

#     Returns
#     -------
#     float
#         Left hand side value.
#     float
#         Right hand side value.
#     float
#         Relative error between the two scalar evaluations.
#     """

#     # --- Compute residuals/outputs at base state --
#     residual_vals  = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
#     output_vals    = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
#     component.solve_residual_equations(input_vals, output_vals)
#     component.evaluate_residuals(input_vals, output_vals, residual_vals)

#     if mode == 'fwd':
#         raise NotImplementedError('forward mode has not been implemented for test_jacvec_product')

#     if mode == 'rev':
#         # --- Compute J v (via finite diff) ---
#         # Perturb input in the v-direction
#         perturbed_output_vals = {k: np.copy(vv) for k, vv in output_vals.items()}
#         for key in output_vals:
#             perturbed_output_vals[key] += eps*v[key]

#         # Compute perturbed residuals
#         perturbed_residual_vals = {k: np.zeros_like(vv) for k, vv in residual_vals.items()}
#         component.evaluate_residuals(input_vals, perturbed_output_vals, perturbed_residual_vals)
#         J_v = {name: (perturbed_residual_vals[name] - residual_vals[name]) / eps for name in residual_vals.keys()}

#         # --- Compute J^-T v (adjoint mode) ---
#         # Initialize our product to zeros (this is our J^-T v)
#         JinvT_v    = {name: np.zeros_like(val) for name, val in residual_vals.items()}
        
#         component.apply_inverse_jacobian(input_vals, output_vals, v, JinvT_v, 'rev')

#         vols = None #component.dafoam_instance.getStateWeights()
#         def inner(a, b):
#             if vols is None:
#                 return np.vdot(a, b)
#             else:
#                 return np.vdot(vols * a, b)

#         lhs = sum(inner(v[name], v[name]) for name in v)
#         rhs = sum(inner(JinvT_v[name], J_v[name]) for name in J_v)

#         # # --- Compute LHS and RHS ---
#         # lhs = sum(np.vdot(v[name], v[name]) for name in v)
#         # rhs = sum(np.vdot(JinvT_v[name], J_v[name]) for name in JinvT_v)

#     # --- Compare ---
#     err = np.abs(lhs - rhs) / (rhs)

#     print(f"LHS (v^T v): {lhs}")
#     print(f"RHS ((J^-T v)^T (J v)): {rhs}")
#     print(f"Relative error: {err:.3e}")

#     return lhs, rhs, err



# # region TEST_IDEMPOTENCE
# def test_idempotence(component, input_vals, tolerance=None, show_plots=False):
#     """
#     Generic test for idempotence a CSDL component. That is, will a repeated call of the
#     compute, or solve_residual_equations and evaluate residuals yield the same outputs and/or residuals?

#     Parameters
#     ----------
#     component : CSDL component (explicit or implicit)
#         Must implement compute and compute_jacvec_product.
#     input_vals : dict
#         Dictionary of input arrays (keys must match component input names).
#     tolerance : float
#         A maximum tolerance which serves as the threshold for determining if two solutions are similar.
#         If the maximum difference between two solutions is below this value, then we say component is idempotent.
#         Will choose a factor or eps for the component's datatype if no tolerance specified.

#     Returns
#     -------
#     bool
#         Whether or not component is idempotent
#     """

#     # Initialize outputs
#     output_vals_1    = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
#     output_vals_2    = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
    
#     # --- Compute outputs sequentially --
#     if hasattr(component, 'solve_residual_equations'):
#         print("Testing solve_residual_equations idempotence...")
#         component.solve_residual_equations(input_vals, output_vals_1)
#         component.solve_residual_equations(input_vals, output_vals_2)
#         component_type = 'implicit'
#         component_representation = "R(x, y) = 0"

#     elif hasattr(component, 'compute'):
#         print("Testing compute idempotence...")
#         component.compute(input_vals, output_vals_1)
#         component.compute(input_vals, output_vals_2)
#         component_type = 'explicit'
#         component_representation = "F(x) = y"

#     else:
#         TypeError('the supplied component doesn''t seem to be a CSDL Implicit or Explicit component')

#     # --- Compute residuals sequentially --
#     if component_type == 'implicit' and hasattr(component, 'evaluate_residuals'):
#         print("Testing evaluate_residuals idempotence...")
#         residual_vals_1 = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
#         residual_vals_2 = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}

#         component.evaluate_residuals(input_vals, output_vals_2, residual_vals_1)
#         component.evaluate_residuals(input_vals, output_vals_2, residual_vals_2)

#         can_evaluate_residuals = True

#     else:
#         can_evaluate_residuals = False
#         print('No evalaute_residual method found for this implicit component. Skipping residual idempotence test.')

#     # Automatically get tolerance based on dtype if none provided
#     if tolerance is None:
#         # Get epsilon for arrays (This assumes all outputs share same dtype)
#         key1 = list(output_vals_1.keys())[0]
#         val  = output_vals_1[key1]

#         # Ensure the value is treated as a NumPy array/scalar
#         val_arr = np.asarray(val)

#         if np.issubdtype(val_arr.dtype, np.floating):
#             eps = np.finfo(val_arr.dtype).eps
#         else:
#             print("Epsilon is not defined for non-floating types. Using eps = 1e-9")
#             eps = 1e-10

#         # We'll make the tolerance a few orders of magnitude greater than epsilon
#         tolerance = 1000*eps

#     # Initial summary
#     is_idempotent        = True
#     is_idempotent_global = True
#     print('----------------------------------------')
#     print('IDEMPOTENCE TEST SUMMARY')
#     print('')
#     print(f'Component type: {component_type}, {component_representation}')
#     print(f"Inputs: {', '.join(input_vals.keys())}")
#     print(f"Outputs: {', '.join(output_vals_1.keys())}")
    
#     # Compute error metrics and print
#     for key in output_vals_1:
#         # Output error metrics
#         y_diff_norm            = np.linalg.norm(output_vals_2[key] - output_vals_1[key])
#         y_diff_norm_normalized = np.linalg.norm(output_vals_2[key] - output_vals_1[key])/np.linalg.norm(output_vals_1[key])
#         max_y_diff             = np.max(np.abs(output_vals_2[key] - output_vals_1[key]))
#         if max_y_diff > tolerance:
#             is_idempotent        = False
#             is_idempotent_global = False

#         if is_idempotent:
#             tag = "Satisfies tolerance"
#         else:
#             tag = "***Larger than tolerance!***"

#         print(f'Output idempotence for {key}:')
#         print('\t{:<30} : {:.4E}'.format('norm(y2 - y1)', y_diff_norm))
#         print('\t{:<30} : {:.4E}'.format('norm(y2 - y1)/norm(y1)', y_diff_norm_normalized))
#         print('\t{:<30} : {:.4E} {}'.format('max(|y2 - y1|)', max_y_diff, tag))

#         is_idempotent = True

#         if can_evaluate_residuals:
#             # Residual error metrics
#             r_diff_norm            = np.linalg.norm(residual_vals_2[key] - residual_vals_1[key])
#             r_diff_norm_normalized = np.linalg.norm(residual_vals_2[key] - residual_vals_1[key])/np.linalg.norm(residual_vals_1[key])
#             max_r_diff             = np.max(np.abs(residual_vals_2[key] - residual_vals_1[key]))
#             if max_r_diff > tolerance:
#                 is_idempotent        = False
#                 is_idempotent_global = False

#             if is_idempotent:
#                 tag = "Satisfies tolerance"
#             else:
#                 tag = "***Larger than tolerance!***"

#             print(f'Residual idempotence for {key}:')
#             print('\t{:<30} : {:.4E}'.format('norm(r2 - r1)', np.linalg.norm(residual_vals_2[key] - residual_vals_1[key])))
#             print('\t{:<30} : {:.4E}'.format('norm(r2 - r1)/norm(r1)', np.linalg.norm(residual_vals_2[key] - residual_vals_1[key])/np.linalg.norm(residual_vals_1[key])))
#             print('\t{:<30} : {:.4E} {}'.format('max(|r2 - r1|)', max_r_diff, tag))

#             is_idempotent = True

#     if is_idempotent_global:
#         print(f"For given tolerance, {tolerance}, this component *appears* to be idempotent.")
#     else:
#         print(f"For given tolerance, {tolerance}, this component does NOT *appear* to be idempotent!")

#     print('----------------------------------------')

#     # Plots
#     if show_plots:
#         for key in output_vals_1.keys():
#             plt.figure()
#             plt.title(f'Output idempotence test, {key}')
#             plt.scatter(output_vals_1[key] - output_vals_2[key], label="Outputs1 - Outputs2")
#             plt.legend()
#             plt.show(block=False)

#         if component_type == 'implicit':
#             for key in residual_vals_1.keys():
#                 plt.figure()
#                 plt.scatter(f'Residual idempotence test, key: {key}')
#                 plt.plot(residual_vals_1[key] - residual_vals_2[key], label="Residuals1 - Residuals2")
#                 plt.legend()
#                 plt.show(block=False)

#     return is_idempotent_global



# def arrays_replicated_fast(x, comm):
#     local_min = np.min(x)
#     local_max = np.max(x)
#     local_sum = np.sum(x)

#     min_equal = comm.allreduce(local_min, MPI.MIN) == \
#                 comm.allreduce(local_min, MPI.MAX)

#     max_equal = comm.allreduce(local_max, MPI.MIN) == \
#                 comm.allreduce(local_max, MPI.MAX)

#     sum_equal = comm.allreduce(local_sum, MPI.MIN) == \
#                 comm.allreduce(local_sum, MPI.MAX)

#     return min_equal and max_equal and sum_equal


# def check_if_distributed(x, comm):
#     # We'll first compare the size of the input among ranks (cheaper than content comparison) ----
#     local_shape = x.shape
#     all_shapes  = comm.allgather(local_shape)
#     if len(set(all_shapes)) > 1:
#         return True

#     # If same size, compare contents
#     if not arrays_replicated_fast(x, comm):
#         return True

#     return False

# def generate_random_like(value_dict, distributed_dict=None, scalar=1, comm=None, random_seed=None):
#     out_dict = {}

#     if random_seed is not None:
#         np.random.seed(random_seed)

#     for name, vv in value_dict.items():
#         shape = vv.shape

#         if comm is None or distributed_dict is None:
#             rand_array = np.random.rand(*shape)

#         else: 
#             # If the variable is of a distributed type then we'll let each rank do the sample
#             # (For a non-comm case, this will be the same as well)
#             if distributed_dict[name]:
#                 rand_array = np.random.rand(*shape)

#             # Otherwise, we'll have to copy it
#             else:
#                 if comm.Get_rank() == 0:
#                     rand_array = np.random.rand(*shape)
#                 else:
#                     rand_array = np.empty(shape, dtype=vv.dtype)

#                 comm.Bcast(rand_array, root=0)

#         out_dict[name] = scalar * rand_array * vv
    
#     return out_dict


        
        

