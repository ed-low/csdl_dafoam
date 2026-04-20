import csdl_alpha as csdl
from csdl_dafoam.utils.csdl_test_functions import CustomComponentChecks
from mpi4py import MPI
import numpy as np


# MPI Setup
comm 	  = MPI.COMM_WORLD
rank 	  = comm.Get_rank()
comm_size = comm.Get_size()

# Matrix setup
# Generate a tall skinny array
m 	 = 11
n 	 = 3
np.random.seed(0)
A_np = np.random.random((m, n))
y_np = np.random.random(m)
w_np = np.random.random(m)
x_np = np.random.random(n)

# Partitioning selection
distribution = "block" #strided

# # Block partitioning
if distribution == "block":
    rows_per_rank = m // comm_size
    remainder = m % comm_size
    start = rank * rows_per_rank + min(rank, remainder)
    end   = start + rows_per_rank + (1 if rank < remainder else 0)
    A_local_np = A_np[start:end, :]
    y_local_np = y_np[start:end]
    w_local_np = w_np[start:end]

elif distribution == "strided":
    A_local_np = A_np[rank::comm_size, :]
    y_local_np = y_np[rank::comm_size]
    w_local_np = w_np[rank::comm_size]

# Get local sizes
m_local    = y_local_np.size

# CSDL Setup
recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()

# Weights for one of our loss tests
alpha   = csdl.Variable(value=1.)


# # BASE CASE
A 	  	  = alpha * A_np #csdl.Variable(value=A_np)
y 		  = csdl.Variable(value=y_np)
w 		  = csdl.Variable(value=w_np)
x 	 	  = csdl.Variable(value=x_np)
obj_base = csdl.sum(y.reshape((m,1)) * (A @ x.reshape((n,1))))

# DISTRIBUTED CASE
def mpi_sum(x):
    return csdl.experimental.mpi.mpi_sum(x, comm=comm)

def mpi_allreduce(x):
    return csdl.experimental.mpi.mpi_allreduce(x, comm=comm)


def global_scalar_times_distributed_array(scalar, array):
    with csdl.experimental.mpi.enter_mpi_region(rank, comm) as region:
        array_local = region.split_custom(array, lambda v:v)
        b_local     = scalar * array_local
        b_out       = region.merge_custom(b_local, merge_func=lambda v:v)
        return b_out
    

with csdl.experimental.mpi.enter_mpi_region(rank, comm, name='dist_obj') as region:
    # y_local is a local constant — create it inside the region
    y_local = region.split_constant(y_local_np)

    # alpha is external → auto-marked 'global' → cotangent will be allreduced
    # A_local_np is just a numpy constant, embedded per-rank
    A_local = alpha * A_local_np

    # x is external → also auto-marked 'global' → cotangent allreduced too
    local_obj = csdl.sum(y_local.reshape((m_local, 1)) * (A_local @ x.reshape((n, 1))))

    # merge_custom registers obj_dist as the global output
    obj_dist = region.merge_custom(
        local_obj,
        merge_func=lambda v: csdl.experimental.mpi.mpi_sum(v, comm=comm)
    )

    print(region.mpi_region_graph.inputs)

# A_local  = alpha * A_local_np
# y_local  = csdl.Variable(value=y_local_np)
# obj_dist = mpi_sum(csdl.sum(y_local.reshape((m_local,1)) * (A_local @ x.reshape((n,1)))))

recorder.stop()

sim = csdl.experimental.PySimulator(recorder=recorder)

grad_base = sim.compute_totals(obj_base, alpha)[obj_base, alpha]
grad_dist = sim.compute_totals(obj_dist, alpha)[obj_dist, alpha]

print(f"Rank {rank} grad_base: {grad_base}")
print(f"Rank {rank} grad_dist: {grad_dist}")
