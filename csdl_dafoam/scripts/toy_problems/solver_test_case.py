from csdl_dafoam.core.rom.rom_models import SyntheticROMModel, Burgers1DFOM
from csdl_dafoam.core.rom.rom_solver import NewtonSolver
import numpy as np


# # Sythetic ROM test case
# basis_size = 40
# model = SyntheticROMModel(N=40, r=basis_size, use_analytic_jacobian=True)
# q0 = np.ones(basis_size)
# q0 = np.array([-0.62107826,  1.20090678,  0.7566991 ,  2.71026963,  1.76450142,
#        -0.35667145, -0.7154965 , -0.97830997, -0.20256139, -1.30892025,
#         2.7849818 , -2.40077271, -0.10158651, -0.13930641,  0.4007413 ,
#        -2.10015558, -0.21550262,  2.12448416, -1.16679522, -1.26670513,
#        -2.72151169, -0.91121849,  1.92804491, -0.07109326,  2.00455857,
#        -0.06004273, -0.56654414, -0.62219732,  1.25555467, -0.1108758 ])



# Burgers Equation test case
model = Burgers1DFOM(N=50, nu=0.01, u0=1.0, u1=2.0, use_analytic_jacobian=True)
q0 = model.u_init


solver = NewtonSolver(
    model,
    options={
        "maxiter": 30,
        "tol_abs": 1e-12
    }
)
result = solver.solve(q0)

model.print_fn(result)

# model.print_fn(f"True solution: {repr(model.q_true)}")
# model.print_fn(f"ROM Error:     {np.linalg.norm(result.rom_state - model.q_true) / np.linalg.norm(model.q_true)}")
# print(f"FOM Error ({model.rank}):  {np.linalg.norm(model.reconstruct_fom_state(result.rom_state) - model.w_true_local) / np.linalg.norm(model.w_true_local)}")
# print(f"{model.rank} FOM solution:  {model.reconstruct_fom_state(result.rom_state)}")
# print(f"{model.rank} True FOM solution:  {model.w_true_local}")
