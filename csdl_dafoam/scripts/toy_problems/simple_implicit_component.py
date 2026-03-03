import csdl_alpha as csdl
import numpy as np

class DistributedSqrtSolve(csdl.experimental.CustomImplicitOperation):
    def __init__(self, comm):
        """
        Distributed implicit sqrt solve with an additional global output.

        Residuals:
            R_y(y; x, a) = y^2 - a*x = 0    → y = sqrt(a*x)   [distributed]
            R_p(p; a)    = p - a^2   = 0    → p = a^2          [global scalar]

        Jacobian (block diagonal):
            dR_y/dy = diag(2y)   → J^{-T} v_y = v_y / (2y)
            dR_p/dp = 1          → J^{-T} v_p = v_p
        """
        super().__init__()
        self.comm = comm
        
    def evaluate(self, x, a):
        self.declare_input('x', x)  # distributed
        self.declare_input('a', a)  # global scalar

        y = self.create_output('y', x.shape)    # distributed
        p = self.create_output('p', a.shape)    # global scalar

        output = csdl.VariableGroup()
        output.y = y
        output.p = p
        return output

    def solve_residual_equations(self, input_vals, output_vals):
        a = input_vals['a']
        output_vals['y'] = np.sqrt(a * input_vals['x'])
        output_vals['p'] = a**2

    def evaluate_residuals(self, input_vals, output_vals, residual_vals):
        a = input_vals['a']
        residual_vals['y'] = output_vals['y']**2 - a * input_vals['x']
        residual_vals['p'] = output_vals['p'] - a**2

    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_residuals, d_outputs, mode):
        if mode == 'rev':
            y = output_vals['y']
            a = input_vals['a']
            x = input_vals['x']

            # dR_y/dy = diag(2y)  — distributed, fine as-is
            d_residuals['y'] += 2.0 * y * d_outputs['y']
            # dR_y/dx = -a*I  — distributed, fine as-is
            d_inputs['x']    += -a * d_outputs['y']
            # dR_y/da = -x  — distributed input dotted with distributed output
            #                  each rank accumulates local contribution, test allreduces after
            d_inputs['a']    += np.vdot(-x, d_outputs['y'])

            # dR_p/dp = 1  — global, fine as-is
            d_residuals['p'] += d_outputs['p']
            # dR_p/da = -2a  — global→global: ONLY accumulate on rank 0
            #                   otherwise every rank adds the same value and allreduce overcounts
            if self.comm is None or self.comm.Get_rank() == 0:
                d_inputs['a'] += np.vdot(-2.0 * a, d_outputs['p'])

    def apply_inverse_jacobian(self, input_vals, output_vals, d_outputs, d_residuals, mode):
        if mode == 'rev':
            # distributed block: J^{-T} v_y = v_y / (2y)
            d_residuals['y'] = d_outputs['y'] / (2.0 * output_vals['y'])
            # global block: J^{-T} v_p = v_p / 1
            d_residuals['p'] = d_outputs['p']