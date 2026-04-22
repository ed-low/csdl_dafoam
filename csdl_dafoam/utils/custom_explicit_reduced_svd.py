import numpy as np
import csdl_alpha as csdl
from typing import List, Literal


#  compute_jacvec_product
#  ---------- analytic adjoint (real case, distinct s) ----------
# Reference: https://github.com/pytorch/pytorch/blob/7a8152530d490b30a56bb090e9a67397d20e16b1/torch/csrc/autograd/FunctionsManual.cpp#L3228
# The real case (A \in R)
# See e.g. https://j-towns.github.io/papers/svd-derivative.pdf
#
# Denote by skew(X) = X - X^T, and by A o B the coordinatewise product.
# Let:
# M = [(skew(U^T U_bar) / E)S + S(skew(V^T V_bar) / E) + I o gS ]
# E_{jk} = S_k^2 - S_j^2 if j != k and 1 otherwise
#
# Then
# if m == n
#   A_bar = U M V^T
#
# elif m > n
#   A_bar = (U M + (I_m - UU^T)U_bar S^{-1}) V^T
#
# elif m < n
#   A_bar = U (M V^T + S^{-1} (V_bar)^T (I_n - VV^T))



# region CUSTOMEXPLICITREDUCEDSVD
class customExplicitReducedSVD(csdl.CustomExplicitOperation):
	def __init__(self, clip_singular_vals:List[float]|None=None):
		super().__init__()
		self.clip_singular_values = clip_singular_vals

    # region evaluate
	def evaluate(self, A:csdl.Variable):
		self.declare_input("A", A)
		
		dimensions = A.shape
		
		if len(dimensions) > 2:
			leading_dims = dimensions[:-2]
		else:
			leading_dims = None
		
		m = dimensions[-2]
		n = dimensions[-1]
		
		k = min(m, n)

		if leading_dims is not None:
			U  = self.create_output("U",  (*leading_dims, m, k))
			S  = self.create_output("S",  (*leading_dims, k))
			VT = self.create_output("VT", (*leading_dims, k, n))
		else:
			U  = self.create_output("U",  (m, k))
			S  = self.create_output("S",  (k,))
			VT = self.create_output("VT", (k, n))

		return U, S, VT

    # region compute
	def compute(self, input_vals, output_vals):
        
		A = input_vals["A"]

		if np.iscomplexobj(A):
			print('WARING: Complex array found. customExplicitReducedSVD currently only defines derivatives for real valued arrays.')
		
		U, S, VT = np.linalg.svd(A, full_matrices=False)
		
		# TODO: Implement some checks for S. See if we have any zero values or small differences between values
		# This is implemented purely for the subspace angle computation, where numerical noise causes the singular values
		# to be larger than 1
		if self.clip_singular_values is not None:
			S = np.clip(S, self.clip_singular_values[0], self.clip_singular_values[1])
		
		output_vals["U"]  = U
		output_vals["S"]  = S
		output_vals["VT"] = VT


    # region compute_jacvec_product
	def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, mode):
		# Extract input and output vals
		A  = input_vals["A"]
		U  = output_vals["U"]
		S  = output_vals["S"]
		VT = output_vals["VT"]

		UT = U.swapaxes(-2, -1)
		V  = VT.swapaxes(-2, -1)

		# Get our dimensions
		m = U.shape[-2]
		n = V.shape[-2]
		k = S.shape[-1]	

		if mode == 'fwd':
			raise NotImplementedError(
				'forward mode has not been implemented for customExplicitReducedSVD'
			)

		elif mode == 'rev':
			# Some easy cases
			if d_outputs["U"] is None and d_outputs["VT"] is None:
				# Trivial case where no gradients are specified
				if d_outputs["S"] is None:
					d_inputs["A"] += np.zeros_like(A)
				
				# Just only singular value gradient
				else:
					S_bar 	  = d_outputs["S"]	
					d_inputs["A"] += U @ (S_bar[..., :, None] * VT) if m >= n else (U * S_bar[..., None, :]) @ VT

				return					
		
			# Extract output cotangents (default to zeros if not provided)
			U_bar  = d_outputs["U"]  if d_outputs["U"]  is not None else np.zeros_like(U)
			S_bar  = d_outputs["S"]  if d_outputs["S"]  is not None else np.zeros_like(S)
			VT_bar = d_outputs["VT"] if d_outputs["VT"] is not None else np.zeros_like(VT)

			V_bar  = VT_bar.swapaxes(-2, -1)

			# Premultiply and skew some arrays
			UTU_bar = UT @ U_bar
			VTV_bar = VT @ V_bar

			skew_UTU_bar = UTU_bar - UTU_bar.swapaxes(-2, -1)
			skew_VTV_bar = VTV_bar - VTV_bar.swapaxes(-2, -1)

			# Construct our E array
			tol = 1e-12
			S2 = S**2
			E = S2[..., None, :] - S2[..., :, None]
			E[..., np.eye(k, dtype=bool)] = 1.0
			E = np.where(np.abs(E) > tol, E, np.inf)
			invS = np.where(S > tol, 1.0 / S, 0.0)

			# Build our M array
			numerator = skew_UTU_bar*S[..., None, :] + S[..., :, None]*skew_VTV_bar
			M = numerator / E
			I = np.eye(S.shape[-1], dtype=S.dtype)
			M += S_bar[..., :, None] * I

			# Square case
			if m == n:
				A_bar            = U @ M @ VT

			# Rectangular correction cases
			elif m > n:
				U_barSinv        = U_bar * invS[..., None, :]
				U_barSinv_proj   = U_barSinv - U @ (UT @ U_barSinv)
				A_bar            = U @ M + U_barSinv_proj
				A_bar            = A_bar @ VT
			
			elif m < n:
				Sinv_V_barT      = (V_bar * invS[..., None, :]).swapaxes(-2, -1)
				Sinv_V_barT_proj = Sinv_V_barT - (Sinv_V_barT @ V) @ VT
				A_bar            = M @ VT + Sinv_V_barT_proj
				A_bar 			 = U @ A_bar

			# Accumulate into input adjoints
			d_inputs["A"] += A_bar

		else:
			raise ValueError(f'"{mode}" not recognized. Only support "fwd" and "rev" modes')
		



from mpi4py import MPI
from csdl_dafoam.utils.decompositions import svd_distributed


# region CUSTOMEXPLICITREDUCEDSVDDISTRIBUTED
class customExplicitReducedSVDDistributed(csdl.CustomExplicitOperation):
	def __init__(self, comm:MPI.Comm, method:Literal["tsqr", "gram"]="tsqr"):
		super().__init__()
		self.comm 	= comm
		self.method = method

    # region evaluate
	def evaluate(self, A_rows_local:csdl.Variable):
		self.declare_input("A_local", A_rows_local)
		
		dimensions = A_rows_local.shape
		
		if len(dimensions) > 2:
			leading_dims = dimensions[:-2]
		else:
			leading_dims = None
		
		m = dimensions[-2]
		n = dimensions[-1]
		
		k = min(m, n)

		if leading_dims is not None:
			U_local  = self.create_output("U_local", (*leading_dims, m, k))
			S  		 = self.create_output("S",  	 (*leading_dims, k))
			VT 		 = self.create_output("VT", 	 (*leading_dims, k, n))
		else:
			U_local  = self.create_output("U_local", (m, k))
			S  		 = self.create_output("S",  	 (k,))
			VT 		 = self.create_output("VT", 	 (k, n))

		return U_local, S, VT
	

	# region compute
	def compute(self, input_vals, output_vals):
		comm 	= self.comm
		A_local = input_vals["A_local"]

		if np.iscomplexobj(A_local):
			print('WARING: Complex array found. customExplicitReducedSVD currently only defines derivatives for real valued arrays.')

		U_local, S, VT = svd_distributed(matrix_local=A_local, method=self.method, comm=comm)

		# TODO: Implement some checks for S. See if we have any zero values or small differences between valuess
		
		output_vals["U_local"]  = U_local
		output_vals["S"]  = S
		output_vals["VT"] = VT


    # region compute_jacvec_product
	def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, mode):
		comm = self.comm
		rank = comm.Get_rank()

		# Extract input and output vals
		A_local  = input_vals["A_local"]
		U_local  = output_vals["U_local"]
		S  		 = output_vals["S"]
		VT 		 = output_vals["VT"]

		U_localT = U_local.swapaxes(-2, -1)
		V  	     = VT.swapaxes(-2, -1)

		# Get our dimensions
		m_local = U_local.shape[-2]
		m = comm.allreduce(m_local, op=MPI.SUM)
		n = V.shape[-2]
		k = S.shape[-1]		

		if mode == 'fwd':
			raise NotImplementedError(
				'forward mode has not been implemented for customExplicitReducedSVD'
			)

		elif mode == 'rev':
			# Some easy cases
			if d_outputs["U_local"] is None and d_outputs["VT"] is None:
				# Trivial case where no gradients are specified
				if d_outputs["S"] is None:
					d_inputs["A_local"] += np.zeros_like(A_local)
				
				# Just only singular value gradient
				else:
					S_bar 	  		     = d_outputs["S"]	
					d_inputs["A_local"] += U_local @ (S_bar[..., :, None] * VT) if m >= n else (U_local * S_bar[..., None, :]) @ VT

				return					
		
			# Extract output cotangents (default to zeros if not provided)
			U_bar_local  = d_outputs["U_local"]  if d_outputs["U_local"]  is not None else np.zeros_like(U_local)
			# S_bar  		 = d_outputs["S"]  		 if d_outputs["S"]  	  is not None else np.zeros_like(S)
			# VT_bar 		 = d_outputs["VT"] 		 if d_outputs["VT"] 	  is not None else np.zeros_like(VT)
			# V_bar  		 = VT_bar.swapaxes(-2, -1)	

			# Have to gather all of the upstream gradient contributions for our "global variables" (S and VT)
			VT_bar_local = d_outputs["VT"].copy() if d_outputs["VT"] is not None else np.zeros_like(VT)
			VT_bar = np.zeros_like(VT_bar_local)
			comm.Allreduce(VT_bar_local, VT_bar, op=MPI.SUM)
			V_bar = VT_bar.swapaxes(-2, -1)

			S_bar_local = d_outputs["S"].copy() if d_outputs["S"] is not None else np.zeros_like(S)
			S_bar = np.zeros_like(S_bar_local)
			comm.Allreduce(S_bar_local, S_bar, op=MPI.SUM)

			# Premultiply and skew some arrays	
			local_UTU_bar = U_localT @ U_bar_local          # shape (..., k, k)
			UTU_bar = np.zeros_like(local_UTU_bar)
			comm.Allreduce(local_UTU_bar, UTU_bar, op=MPI.SUM)
			
			VTV_bar = VT @ V_bar

			skew_UTU_bar = UTU_bar - UTU_bar.swapaxes(-2, -1)
			skew_VTV_bar = VTV_bar - VTV_bar.swapaxes(-2, -1)

			# Construct our E array
			tol  = 1e-12
			S2   = S**2
			E    = S2[..., None, :] - S2[..., :, None]
			# E = S2[..., :, None] - S2[..., None, :]
			E[..., np.eye(k, dtype=bool)] = 1.0
			E 	 = np.where(np.abs(E) > tol, E, np.inf)
			invS = np.where(S > tol, 1.0 / S, 0.0)

			# Build our M array
			numerator = skew_UTU_bar * S[..., None, :] + S[..., :, None] * skew_VTV_bar
			M  = numerator / E
			I  = np.eye(k, dtype=S.dtype)
			M += S_bar[..., :, None] * I

			# Square case
			if m == n:
				A_bar_local            = U_local @ M @ VT

			# Rectangular correction cases
			elif m > n:
				U_bar_localSinv   	 = U_bar_local * invS[..., None, :]
				U_barSinv_proj_local = U_bar_localSinv - U_local @ (UTU_bar * invS[..., None, :])
				A_bar_local       	 = U_local @ M + U_barSinv_proj_local
				A_bar_local       	 = A_bar_local @ VT
			
			elif m < n:
				Sinv_V_barT      = (V_bar * invS[..., None, :]).swapaxes(-2, -1)
				Sinv_V_barT_proj = Sinv_V_barT - (Sinv_V_barT @ V) @ VT
				A_bar_local      = M @ VT + Sinv_V_barT_proj
				A_bar_local 	 = U_local @ A_bar_local

			# Accumulate into input adjoints
			d_inputs["A_local"] += A_bar_local

		else:
			raise ValueError(f'"{mode}" not recognized. Only support "fwd" and "rev" modes')



		

# region _make_test_matrix
def _make_test_matrix(m, n, mat_rank, decay='linear', seed=42):
	# Set seed for consistency among ranks
	np.random.seed(seed=seed)
	U, _ = np.linalg.qr(np.random.randn(m, mat_rank))
	V, _ = np.linalg.qr(np.random.randn(n, mat_rank))

	if decay == 'linear':
		S = np.linspace(10, 1, mat_rank)
	elif decay == 'exp':
		S = 10 ** (-np.arange(0, 6, 6. / mat_rank))
	elif decay == 'flat':
		S = np.ones(mat_rank)

	A = U @ np.diag(S) @ V.T

	return A, S

# region main
if __name__ == "__main__":

	import csdl_alpha as csdl
	from csdl_dafoam.utils.csdl_test_functions import CustomComponentChecks
	from mpi4py import MPI
	

	# MPI Setup
	comm 	  = MPI.COMM_WORLD
	rank 	  = comm.Get_rank()
	comm_size = comm.Get_size()

	# Matrix setup
	# Generate a tall skinny array
	m 	 = 11
	n 	 = 3
	A_np, _ = _make_test_matrix(m, n, mat_rank=n, decay="linear")
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
	weights = csdl.Variable(value=np.random.rand(n))
	alpha   = csdl.Variable(value=1.)

	loss_dict = {name:{loss_name:None for loss_name in ["loss_1", "loss_2", "loss_3", "loss_4", "loss_5"]} for name in ["base", "dist"]}
	grad_norm = {name:{loss_name:None for loss_name in ["loss_1", "loss_2", "loss_3", "loss_4", "loss_5"]} for name in ["base", "dist"]}

	# # BASE CASE
	A 	  	  = alpha * A_np #csdl.Variable(value=A_np)
	y 		  = csdl.Variable(value=y_np)
	w 		  = csdl.Variable(value=w_np)
	x 	 	  = csdl.Variable(value=x_np)
	customSVD = customExplicitReducedSVD()
	U, S, VT  = customSVD.evaluate(A=A)
	
	# loss functions
	loss_dict["base"]["loss_1"] = csdl.sum(S)
	loss_dict["base"]["loss_2"] = csdl.sum(S * weights)
	loss_dict["base"]["loss_3"] = csdl.sum(w.reshape((m,1)) * (U @ (U.T() @ y.reshape((m,1)))))
	loss_dict["base"]["loss_4"] = csdl.sum(y.reshape((m,1)) * (U @ (csdl.einsum(S, VT, action='i,ij->ij')) @ x.reshape((n,1))))
	loss_dict["base"]["loss_5"] = csdl.sum(y.reshape((m,1)) * (A @ x.reshape((n,1))))
 
	# The distributed cases
	def mpi_sum(x):
		return csdl.experimental.mpi.mpi_sum(x, comm=comm)
	
	def mpi_allreduce(x):
		return csdl.experimental.mpi.mpi_allreduce(x, comm=comm)
	
	def global_local_op(global_var, local_var, op):
		with csdl.experimental.mpi.enter_mpi_region(rank, comm) as region:
			if isinstance(local_var, csdl.Variable):
				local_var_split = region.split_custom(local_var, lambda x:x)
			else:
				local_var_split = region.split_constant(local_var)
			out_split        = op(global_var, local_var_split)
			out = region.merge_custom(out_split, lambda x:x)
			return out
	
	A_local = global_local_op(alpha, A_local_np, lambda x, y: x * y)
	y_local = csdl.Variable(value=y_local_np)
	w_local = csdl.Variable(value=w_local_np)

	# FULLY DISTRIBUTED
	customSVDDist 		 		= customExplicitReducedSVDDistributed(comm=comm)
	U_dist, S_dist, VT_dist 	= customSVDDist.evaluate(A_rows_local=A_local)
	loss_dict["dist"]["loss_1"] = csdl.sum(S_dist) / comm_size
	loss_dict["dist"]["loss_2"] = csdl.sum(S_dist * weights) / comm_size
	loss_dict["dist"]["loss_3"] = (csdl.sum(w_local.reshape((m_local,1)) * (U_dist @ mpi_allreduce(U_dist.T() @ y_local.reshape((m_local,1))))))
	loss_dict["dist"]["loss_4"] = (csdl.sum(y_local.reshape((m_local,1)) * (U_dist @ (csdl.einsum(S_dist, VT_dist, action='i,ij->ij')) @ x.reshape((n,1)))))
	loss_dict["dist"]["loss_5"] = (csdl.sum(y_local.reshape((m_local,1)) * (A_local @ x.reshape((n,1)))))
	
	recorder.stop()
	sim = csdl.experimental.PySimulator(recorder=recorder)

	# Manual derivative check
	for svd_type, losses in loss_dict.items():
		for loss_case, loss_var in losses.items():
			# wrt = A if svd_type == "base" else A_local
			wrt = alpha
			analytical_grad       		   = sim.compute_totals(loss_var, wrt)[loss_var, wrt]
			grad_norm[svd_type][loss_case] = analytical_grad[0][0]

	import pandas as pd

	
	df = pd.DataFrame.from_dict(grad_norm, orient="index")
	pd.set_option("display.float_format", "{:.8f}".format)
	print(f"------------ \n Rank {rank} \n {df}")