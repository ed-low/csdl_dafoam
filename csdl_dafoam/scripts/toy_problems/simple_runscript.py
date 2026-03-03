import csdl_alpha as csdl
from simple_implicit_component import DistributedSqrtSolve
from mpi4py import MPI
from csdl_dafoam.utils.csdl_test_functions import CustomComponentChecks
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

recorder = csdl.Recorder(inline=True)
recorder.start()

x = csdl.Variable(value=np.random.rand(3,), name='x')   # distributed
a = csdl.Variable(value=np.array([2.0]), name='a')       # global scalar

sqrt_component = DistributedSqrtSolve(comm=comm)
outputs = sqrt_component.evaluate(x, a)

y = outputs.y
p = outputs.p

# rank 0 sanity checks (a=2, x random in [0,1]):
# y ≈ sqrt(2*x),  p = 4.0
if rank == 0:
    print(f"p = {p.value}")  # should be 4.0

recorder.stop()

component_testing = CustomComponentChecks(sqrt_component, comm=comm)
component_testing.run_inverse_jacobian_fd_sweep(
    eps_test_values=10. ** np.array(range(-10, -0)),
    random_scalar=1
)

component_testing.run_jacvec_fd_sweep(
    eps_test_values=10. ** np.array(range(-10, -0)),
    random_scalar=1)