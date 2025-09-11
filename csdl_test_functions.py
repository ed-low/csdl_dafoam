import numpy as np

def test_jacvec_product(component, input_vals, v, w, eps=1e-6, atol=1e-6, rtol=1e-6):
    """
    Generic test for compute_jacvec_product in a CSDL component. Tests via finite differenece
    w^T J^T v = v^T J w
    where J = dR/dx for implicit component, dF/dx for explicit component

    Parameters
    ----------
    component : CSDL component (explicit or implicit)
        Must implement compute and compute_jacvec_product.
    input_vals : dict
        Dictionary of input arrays (keys must match component input names).
    v : dict
        Direction vector for outputs.
    w : dict
        Direction vector for inputs (same structure as input_vals).
    eps : float
        Step size for finite differences.
    atol, rtol : float
        Absolute and relative tolerances for comparison.

    Returns
    -------
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

    rhs = 0.0
    if component_type == 'explicit':
        perturbed_output_vals = {k: np.zeros_like(vv) for k, vv in output_vals.items()}
        component.compute(perturbed_input_vals, perturbed_output_vals)
        J_w = {name: (perturbed_output_vals[name] - output_vals[name]) / eps for name in output_vals.keys()}

    else:
        perturbed_residual_vals = {k: np.zeros_like(vv) for k, vv in residual_vals.items()}
        perturbed_output_vals   = {k: np.zeros_like(vv) for k, vv in output_vals.items()}
        component.solve_residual_equations(perturbed_input_vals, perturbed_output_vals)
        component.evaluate_residuals(perturbed_input_vals, perturbed_output_vals, perturbed_residual_vals)
        J_w = {name: (perturbed_residual_vals[name] - residual_vals[name]) / eps for name in residual_vals.keys()}

        import matplotlib.pyplot as plt
        for k, vv in perturbed_residual_vals.items():
            plt.figure()
            plt.plot(perturbed_residual_vals[k], label=f'perturbed {k} residual')
            plt.plot(residual_vals[k], label=f'{k} residual')
            plt.legend()
            plt.show()

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
    err = (lhs - rhs) / (rhs)

    print(f"LHS (w^T J^T v): {lhs}")
    print(f"RHS (v^T J w): {rhs}")
    print(f"Relative error: {err:.3e}")

    return err