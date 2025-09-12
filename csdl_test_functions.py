import numpy as np
import matplotlib.pyplot as plt

# region TEST_JACVEC_PRODUCT
def test_jacvec_product(component, input_vals, v, w, eps=1e-6):
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

    # --- Compute residuals/outputs at base state --
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
    
    # --- Compute J w (forward mode, via finite diff) ---
    #  perturb input in the w-direction
    perturbed_input_vals = {k: np.copy(vv) for k, vv in input_vals.items()}
    for in_name in input_vals:
        perturbed_input_vals[in_name] += eps*w[in_name]

    if component_type == 'explicit':
        perturbed_output_vals = {k: np.zeros_like(vv) for k, vv in output_vals.items()}
        component.compute(perturbed_input_vals, perturbed_output_vals)
        J_w = {name: (perturbed_output_vals[name] - output_vals[name]) / eps for name in output_vals.keys()}

    else:
        perturbed_residual_vals = {k: np.zeros_like(vv) for k, vv in residual_vals.items()}
        component.evaluate_residuals(perturbed_input_vals, output_vals, perturbed_residual_vals)
        J_w = {name: (perturbed_residual_vals[name] - residual_vals[name]) / eps for name in residual_vals.keys()}

    rhs = sum(np.vdot(v[name], J_w[name]) for name in v)

    # --- Compute J^T v (adjoint mode) ---
    # Initialize our product to zeros (this is our J^T v)
    JT_v    = {name: np.zeros_like(val) for name, val in input_vals.items()}

    if component_type == 'explicit':   
        component.compute_jacvec_product(input_vals, output_vals, JT_v, v, 'rev')

    else:
        d_outputs   = {name: np.zeros_like(val) for name, val in residual_vals.items()}
        component.compute_jacvec_product(input_vals, output_vals, JT_v, d_outputs, v, 'rev')

    lhs = sum(np.vdot(w[name], JT_v[name]) for name in input_vals)

    # --- Compare ---
    err = np.abs(lhs - rhs) / (rhs)

    print(f"LHS (w^T J^T v): {lhs}")
    print(f"RHS (v^T J w): {rhs}")
    print(f"Relative error: {err:.3e}")

    return lhs, rhs, err



# region TEST_IDEMPOTENCE
def test_idempotence(component, input_vals, tolerance=None, show_plots=False):
    """
    Generic test for idempotence a CSDL component. That is, will a repeated call of the
    compute, or solve_residual_equations and evaluate residuals yield the same outputs and/or residuals?

    Parameters
    ----------
    component : CSDL component (explicit or implicit)
        Must implement compute and compute_jacvec_product.
    input_vals : dict
        Dictionary of input arrays (keys must match component input names).
    tolerance : float
        A maximum tolerance which serves as the threshold for determining if two solutions are similar.
        If the maximum difference between two solutions is below this value, then we say component is idempotent.
        Will choose a factor or eps for the component's datatype if no tolerance specified.

    Returns
    -------
    bool
        Whether or not component is idempotent
    """

    # Initialize outputs
    output_vals_1    = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
    output_vals_2    = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
    
    # --- Compute outputs sequentially --
    if hasattr(component, 'solve_residual_equations'):
        print("Testing solve_residual_equations idempotence...")
        component.solve_residual_equations(input_vals, output_vals_1)
        component.solve_residual_equations(input_vals, output_vals_2)
        component_type = 'implicit'
        component_representation = "R(x, y) = 0"

    elif hasattr(component, 'compute'):
        print("Testing compute idempotence...")
        component.compute(input_vals, output_vals_1)
        component.compute(input_vals, output_vals_2)
        component_type = 'explicit'
        component_representation = "F(x) = y"

    else:
        TypeError('the supplied component doesn''t seem to be a CSDL Implicit or Explicit component')

    # --- Compute residuals sequentially --
    if component_type == 'implicit' and hasattr(component, 'evaluate_residuals'):
        print("Testing evaluate_residuals idempotence...")
        residual_vals_1 = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}
        residual_vals_2 = {k: np.zeros_like(vv) for k, vv in component.output_dict.items()}

        component.evaluate_residuals(input_vals, output_vals_2, residual_vals_1)
        component.evaluate_residuals(input_vals, output_vals_2, residual_vals_2)

        can_evaluate_residuals = True

    else:
        can_evaluate_residuals = False
        print('No evalaute_residual method found for this implicit component. Skipping residual idempotence test.')

    # Automatically get tolerance based on dtype if none provided
    if tolerance is None:
        # Get epsilon for arrays (This assumes all outputs share same dtype)
        key1 = list(output_vals_1.keys())[0]
        val  = output_vals_1[key1]

        # Ensure the value is treated as a NumPy array/scalar
        val_arr = np.asarray(val)

        if np.issubdtype(val_arr.dtype, np.floating):
            eps = np.finfo(val_arr.dtype).eps
        else:
            print("Epsilon is not defined for non-floating types. Using eps = 1e-9")
            eps = 1e-10

        # We'll make the tolerance a few orders of magnitude greater than epsilon
        tolerance = 1000*eps

    # Initial summary
    is_idempotent        = True
    is_idempotent_global = True
    print('----------------------------------------')
    print('IDEMPOTENCE TEST SUMMARY')
    print('')
    print(f'Component type: {component_type}, {component_representation}')
    print(f"Inputs: {', '.join(input_vals.keys())}")
    print(f"Outputs: {', '.join(output_vals_1.keys())}")
    
    # Compute error metrics and print
    for key in output_vals_1:
        # Output error metrics
        y_diff_norm            = np.linalg.norm(output_vals_2[key] - output_vals_1[key])
        y_diff_norm_normalized = np.linalg.norm(output_vals_2[key] - output_vals_1[key])/np.linalg.norm(output_vals_1[key])
        max_y_diff             = np.max(np.abs(output_vals_2[key] - output_vals_1[key]))
        if max_y_diff > tolerance:
            is_idempotent        = False
            is_idempotent_global = False

        if is_idempotent:
            tag = "Satisfies tolerance"
        else:
            tag = "***Larger than tolerance!***"

        print(f'Output idempotence for {key}:')
        print('\t{:<30} : {:.4E}'.format('norm(y2 - y1)', y_diff_norm))
        print('\t{:<30} : {:.4E}'.format('norm(y2 - y1)/norm(y1)', y_diff_norm_normalized))
        print('\t{:<30} : {:.4E} {}'.format('max(|y2 - y1|)', max_y_diff, tag))

        is_idempotent = True

        if can_evaluate_residuals:
            # Residual error metrics
            r_diff_norm            = np.linalg.norm(residual_vals_2[key] - residual_vals_1[key])
            r_diff_norm_normalized = np.linalg.norm(residual_vals_2[key] - residual_vals_1[key])/np.linalg.norm(residual_vals_1[key])
            max_r_diff             = np.max(np.abs(residual_vals_2[key] - residual_vals_1[key]))
            if max_r_diff > tolerance:
                is_idempotent        = False
                is_idempotent_global = False

            if is_idempotent:
                tag = "Satisfies tolerance"
            else:
                tag = "***Larger than tolerance!***"

            print(f'Residual idempotence for {key}:')
            print('\t{:<30} : {:.4E}'.format('norm(r2 - r1)', np.linalg.norm(residual_vals_2[key] - residual_vals_1[key])))
            print('\t{:<30} : {:.4E}'.format('norm(r2 - r1)/norm(r1)', np.linalg.norm(residual_vals_2[key] - residual_vals_1[key])/np.linalg.norm(residual_vals_1[key])))
            print('\t{:<30} : {:.4E} {}'.format('max(|r2 - r1|)', max_r_diff, tag))

            is_idempotent = True

    if is_idempotent_global:
        print(f"For given tolerance, {tolerance}, this component *appears* to be idempotent.")
    else:
        print(f"For given tolerance, {tolerance}, this component does NOT *appear* to be idempotent!")

    print('----------------------------------------')

    # Plots
    if show_plots:
        for key in output_vals_1.keys():
            plt.figure()
            plt.title(f'Output idempotence test, {key}')
            plt.scatter(output_vals_1[key] - output_vals_2[key], label="Outputs1 - Outputs2")
            plt.legend()
            plt.show(block=False)

        if component_type == 'implicit':
            for key in residual_vals_1.keys():
                plt.figure()
                plt.scatter(f'Residual idempotence test, key: {key}')
                plt.plot(residual_vals_1[key] - residual_vals_2[key], label="Residuals1 - Residuals2")
                plt.legend()
                plt.show(block=False)

    return is_idempotent_global
    
    