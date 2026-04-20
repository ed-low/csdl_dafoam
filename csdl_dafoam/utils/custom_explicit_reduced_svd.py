import numpy as np
import csdl_alpha as csdl
from typing import List


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

# region CUSTOMEXPLICITREDUCEDSVDROOTDISTRIBUTED
class customExplicitReducedSVDRootDistributed(csdl.CustomExplicitOperation):
	def __init__(self, comm:MPI.Comm,):
		super().__init__()
		self.comm = comm

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
		rank    = comm.Get_rank()
		A_local = input_vals["A_local"]

		if np.iscomplexobj(A_local):
			print('WARING: Complex array found. customExplicitReducedSVD currently only defines derivatives for real valued arrays.')

		A_global, counts_rows = self.gather_rows(A_local)
		
		if rank == 0:
			U, S, VT = np.linalg.svd(A_global, full_matrices=False)
		else:
			U  = None
			S  = None
			VT = None
		
		# Broadcast S and VT
		S  = self.broadcast_array(S)
		VT = self.broadcast_array(VT)

		U_local = self.scatter_rows(U, counts_rows=counts_rows)

		output_vals["U_local"]  = U_local
		output_vals["S"]  = S
		output_vals["VT"] = VT


	# region compute_jacvec_product
	def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, mode):
		# Extract input and output vals
		A_local  = input_vals["A_local"]
		U_local  = output_vals["U_local"]
		S  = output_vals["S"]
		VT = output_vals["VT"]

		U, U_counts_rows = self.gather_rows(U_local)
		
		# Extract output cotangents (default to zeros if not provided)
		U_bar_local  = d_outputs["U_local"]  if d_outputs["U_local"]  is not None else np.zeros_like(U)
		U_bar, _  = self.gather_rows(U_bar_local)

		# Have to gather all of the upstream gradient contributions for our "global variables" (S and VT)
		VT_bar_local = d_outputs["VT"].copy() if d_outputs["VT"] is not None else np.zeros_like(VT)
		VT_bar = np.zeros_like(VT_bar_local)
		comm.Allreduce(VT_bar_local, VT_bar, op=MPI.SUM)
		V_bar = VT_bar.swapaxes(-2, -1)

		S_bar_local = d_outputs["S"].copy() if d_outputs["S"] is not None else np.zeros_like(S)
		S_bar = np.zeros_like(S_bar_local)
		comm.Allreduce(S_bar_local, S_bar, op=MPI.SUM)
		
		if self.comm.Get_rank() == 0:
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
				skip_full = False
				# Some easy cases
				if d_outputs["U_local"] is None and d_outputs["VT"] is None:
					# Trivial case where no gradients are specified
					if d_outputs["S"] is None:
						A_bar = np.zeros_like(A_local)
						skip_full = True
					
					# Just only singular value gradient
					else:
						S_bar 	  = d_outputs["S"]	
						A_bar = U @ (S_bar[..., :, None] * VT) if m >= n else (U * S_bar[..., None, :]) @ VT
						skip_full = True

				if not skip_full:
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

			else:
				raise ValueError(f'"{mode}" not recognized. Only support "fwd" and "rev" modes')
		
		else:
			A_bar = None
			
		# Accumulate into input adjoints
		A_bar_local = self.scatter_rows(A_bar, U_counts_rows)
		d_inputs["A_local"] += A_bar_local
		

	# region gather_rows
	def gather_rows(self, A_local, root=0):
		comm = self.comm
		rank = comm.Get_rank()

		A_local = np.ascontiguousarray(A_local)
		m_local, n = A_local.shape

		counts_rows = comm.allgather(m_local)

		# Convert to element counts (as Python ints)
		counts = [int(c * n) for c in counts_rows]
		displs = [int(sum(counts[:i])) for i in range(len(counts))]

		if rank == root:
			m_global = sum(counts_rows)
			A_global = np.empty((m_global, n), dtype=A_local.dtype)
		else:
			A_global = None

		comm.Gatherv(
			sendbuf=A_local.ravel(),
			recvbuf=(A_global.ravel(), (counts, displs)) if rank == root else None,
			root=root
		)

		return A_global, counts_rows
	

	# region scatter_rows
	def scatter_rows(self, A_global, counts_rows, root=0):
		"""
		Scatter a row-partitioned matrix from root to all ranks.

		Parameters
		----------
		A_global : (m_global, n) ndarray on root, None elsewhere
		counts_rows : list of row counts per rank
		comm : MPI communicator

		Returns
		-------
		A_local : (m_local, n) ndarray on each rank
		"""
		comm = self.comm
		rank = comm.Get_rank()

		# Broadcast n (number of columns)
		if rank == root:
			n = A_global.shape[1]
		else:
			n = None
		n = comm.bcast(n, root=root)

		m_local = counts_rows[rank]

		A_local = np.empty((m_local, n), dtype=A_global.dtype if rank == root else float)

		counts = np.array(counts_rows) * n
		displs = np.cumsum([0] + list(counts[:-1]))

		comm.Scatterv(
			sendbuf=(A_global.ravel(), (counts, displs)) if rank == root else None,
			recvbuf=A_local.ravel(),
			root=root
		)

		return A_local
	

	# region broadcast_array
	def broadcast_array(self, arr, root=0):
		"""
		Broadcast a NumPy array of arbitrary shape.
		"""
		comm = self.comm
		rank = comm.Get_rank()

		if rank == root:
			shape = arr.shape
			dtype = arr.dtype
		else:
			shape = None
			dtype = None

		shape = comm.bcast(shape, root=root)
		dtype = comm.bcast(dtype, root=root)

		if rank != root:
			arr = np.empty(shape, dtype=dtype)

		comm.Bcast(arr, root=root)

		return arr



# region CUSTOMEXPLICITREDUCEDSVDDISTRIBUTED
class customExplicitReducedSVDDistributed(csdl.CustomExplicitOperation):
	def __init__(self, comm:MPI.Comm,):
		super().__init__()
		self.comm = comm

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

		U_local, S, VT = svd_distributed(matrix_local=A_local, method="tsqr", comm=comm)

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

			### DEBUG
			# print(f"[Rank {rank}] A_bar_local norm: {np.linalg.norm(A_bar_local)}", flush=True)
			###
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

	# Block partitioning
	rows_per_rank = m // comm_size
	remainder = m % comm_size
	start = rank * rows_per_rank + min(rank, remainder)
	end   = start + rows_per_rank + (1 if rank < remainder else 0)
	A_local_np = A_np[start:end, :]
	y_local_np = y_np[start:end]
	w_local_np = y_np[start:end]

	# # We'll stripe the array (our MPI partitioning
	# A_local_np = A_np[rank::comm_size, :]
	# y_local_np = y_np[rank::comm_size]
	# w_local_np = w_np[rank::comm_size]

	# Get local sizes
	m_local    = y_local_np.size

	# CSDL Setup
	recorder = csdl.Recorder(inline=True, debug=True)
	recorder.start()

	# Weights for one of our loss tests
	weights = csdl.Variable(value=np.random.rand(n))
	alpha   = csdl.Variable(value=1.)

	loss_dict = {name:{loss_name:None for loss_name in ["loss_1", "loss_2", "loss_3", "loss_4"]} for name in ["base", "root", "dist"]}
	grad_norm = {name:{loss_name:None for loss_name in ["loss_1", "loss_2", "loss_3", "loss_4"]} for name in ["base", "root", "dist"]}

	# # BASE CASE
	A 	  	  = alpha * A_np #csdl.Variable(value=A_np)
	y 		  = csdl.Variable(value=y_np)
	w 		  = csdl.Variable(value=w_np)
	x 	 	  = csdl.Variable(value=x_np)
	customSVD = customExplicitReducedSVD()
	U, S, VT  = customSVD.evaluate(A=A)
	
	# loss functions
	UTy = U.T() @ y.reshape((m,1))
	loss_dict["base"]["loss_1"] = csdl.sum(S)
	loss_dict["base"]["loss_2"] = csdl.sum(S * weights)
	loss_dict["base"]["loss_3"] = csdl.sum(w.reshape((m,1)) * (U @ (U.T() @ y.reshape((m,1)))))
	loss_dict["base"]["loss_4"] = csdl.sum(y.reshape((m,1)) * (U @ (csdl.einsum(S, VT, action='i,ij->ij')) @ x.reshape((n,1))))
 
	# The distributed cases
	def mpi_sum(x):
		return csdl.experimental.mpi.mpi_sum(x, comm=comm)
	
	def mpi_allreduce(x):
		return csdl.experimental.mpi.mpi_allreduce(x, comm=comm)
	
	A_local = alpha * A_local_np	 #csdl.Variable(value=A_local_np)
	y_local = csdl.Variable(value=y_local_np)
	w_local = csdl.Variable(value=w_local_np)

	# ROOT THEN SCATTER
	customSVDRootDist 			= customExplicitReducedSVDRootDistributed(comm=comm)
	U_root, S_root, VT_root 	= customSVDRootDist.evaluate(A_rows_local=A_local)
	loss_dict["root"]["loss_1"] = csdl.sum(S_root)
	loss_dict["root"]["loss_2"] = csdl.sum(S_root * weights)
	loss_dict["root"]["loss_3"] = (csdl.sum(w_local.reshape((m_local,1)) * (U_root @ mpi_allreduce(U_root.T() @ y_local.reshape((m_local,1))))))
	loss_dict["root"]["loss_4"] = mpi_sum(csdl.sum(y_local.reshape((m_local,1)) * (U_root @ (csdl.einsum(S_root, VT_root, action='i,ij->ij')) @ x.reshape((n,1)))))

	# FULLY DISTRIBUTED
	customSVDDist 		 		= customExplicitReducedSVDDistributed(comm=comm)
	U_dist, S_dist, VT_dist 	= customSVDDist.evaluate(A_rows_local=A_local)
	loss_dict["dist"]["loss_1"] = csdl.sum(S_dist)
	loss_dict["dist"]["loss_2"] = csdl.sum(S_dist * weights)
	loss_dict["dist"]["loss_3"] = mpi_sum(csdl.sum(w_local.reshape((m_local,1)) * (U_dist @ mpi_allreduce(U_dist.T() @ y_local.reshape((m_local,1))))))
	loss_dict["dist"]["loss_4"] = mpi_sum(csdl.sum(y_local.reshape((m_local,1)) * (U_dist @ (csdl.einsum(S_dist, VT_dist, action='i,ij->ij')) @ x.reshape((n,1)))))

	recorder.stop()
	sim = csdl.experimental.PySimulator(recorder=recorder)

	# Manual derivative check
	for svd_type, losses in loss_dict.items():
		for loss_case, loss_var in losses.items():
			# wrt = A if svd_type == "base" else A_local
			wrt = alpha
			analytical_grad       		   = sim.compute_totals(loss_var, wrt)[loss_var, wrt]
			
			if svd_type == "base":
				grad_norm[svd_type][loss_case] = np.linalg.norm(analytical_grad)
			else:
				grad_norm[svd_type][loss_case] = np.sqrt(comm.allreduce(np.sum(analytical_grad** 2), op=MPI.SUM))

	import pandas as pd

	if rank  == 0:
		df = pd.DataFrame.from_dict(grad_norm, orient="index")
		pd.set_option("display.float_format", "{:.8f}".format)
		print(df)

	# customSVDDist = customExplicitReducedSVDDistributed(comm=comm)
	# U_d_csdl, S_d_csdl, VT_d_csdl = customSVDDist.evaluate(A_rows_local=A_local_csdl)

	# with csdl.experimental.mpi.enter_mpi_region(rank=rank, comm=comm) as mpi_region:
	# 	mpi_region.split_custom(A_local_csdl, split_func=lambda x:x)

	# 	customSVDDist = customExplicitReducedSVDDistributed(comm=comm)
	# 	U_d_csdl, S_d_csdl, VT_d_csdl = customSVDDist.evaluate(A_rows_local=A_local_csdl)

	# 	mpi_region.set_as_global_output(U_d_csdl)
	# 	mpi_region.set_as_global_output(S_d_csdl)
	# 	mpi_region.set_as_global_output(VT_d_csdl)

	# A_reconstructed = U_d_csdl @ csdl.einsum(S_d_csdl, VT_d_csdl, action='i,ij->ij')
 
	# obj_serial = csdl.sum(U)
	# obj_dist   = csdl.sum(U_d_csdl @ U_d_csdl.T())

	# obj = obj_dist
	# dv  = A_local_csdl

	# dv.set_as_design_variable()
	# obj.set_as_objective()

	# recorder.stop()
	
	# sim = csdl.experimental.PySimulator(recorder=recorder)

	# Manual derivative check
	# analytical_grad  = sim.compute_totals(obj, dv)[obj, dv]
	# finite_diff_grad = sim.compute_totals(obj, dv, use_finite_difference=True)[obj, dv]
	# relative_diff    = (analytical_grad - finite_diff_grad) / analytical_grad

	# print(f"Rank {rank} Analytical  : {analytical_grad}")
	# print(f"Rank {rank} Finite Diff : {finite_diff_grad}")


	# # print(relative_diff)
	# print(f"Rank {rank} ||relative_diff||      : {np.linalg.norm(relative_diff)}")
	# print(f"Rank {rank} ||computed_grad||      : {np.linalg.norm(analytical_grad)}")
	# print(f"Rank {rank} ||fd_grad||            : {np.linalg.norm(finite_diff_grad)}")
	# print(f"Rank {rank} ||cg||-||fdg||/||fdg|| : {(np.linalg.norm(analytical_grad) - np.linalg.norm(finite_diff_grad)) / np.linalg.norm(analytical_grad)}")

    # # Check derivatives
	# import modopt as mo

	
	# prob        = mo.CSDLAlphaProblem(problem_name="test", simulator=sim)
	# optimizer   = mo.SLSQP(problem=prob, solver_options={'ftol':1e-6, 'maxiter':20})
	# optimizer.check_first_derivatives(step=1e-6)

	# component_check = CustomComponentChecks(component=customSVDSerial, random_seed=0, fd_step=1e-6, comm=comm)
	# component_check.run_jacvec_fd_sweep(eps_test_values=10. ** np.array(range(-2, -12, -1)))

	# component_check = CustomComponentChecks(component=customSVDDist, random_seed=0, fd_step=1e-6, comm=comm)
	# component_check.run_jacvec_fd_sweep(eps_test_values=10. ** np.array(range(-2, -12, -1)))










# with csdl.experimental.mpi.enter_mpi_region(rank, comm) as mpi_region:
		# 
		# A_local_csdl = mpi_region.split_custom(A_local_csdl, split_func=lambda x:x)
		# U_d_csdl     = mpi_region.split_custom(U_d_csdl, split_func=lambda x:x)

		
		# A_reconstructed = U_d_csdl @ csdl.einsum(S_d_csdl, VT_d_csdl, action='i,ij->ij')

		# loss0d = csdl.sum(S_d_csdl)
		# loss1d = csdl.sum(U_d_csdl)
		# loss2d = csdl.sum(VT_d_csdl)
		# loss3d = csdl.sum(U_d_csdl) + csdl.sum(VT_d_csdl)
		# loss4d = csdl.sum(U_d_csdl.T() @ U_d_csdl)
		# loss5d = csdl.experimental.mpi.mpi_sum(csdl.sum((A_reconstructed - A_local_csdl)**2), comm=comm)
	
		# mpi_region.set_as_global_output(loss0d)
		# mpi_region.set_as_global_output(loss1d)
		# mpi_region.set_as_global_output(loss2d)
		# mpi_region.set_as_global_output(loss3d)
		# mpi_region.set_as_global_output(loss4d)
		# mpi_region.set_as_global_output(loss5d)
		# mpi_region.set_as_global_output(U_d_csdl)
		# mpi_region.set_as_global_output(S_d_csdl)
		# mpi_region.set_as_global_output(VT_d_csdl)






	# from csdl_dafoam.utils.training_interface import TrainingDataInterface
	# from csdl_dafoam.core.csdl_dafoam import instantiateDAFoam
	# import os
	# from pathlib import Path

	# # DAFoam
	# problem_name        = 'rom_test2'
	# dafoam_directory    = "/media/edward/DATA/Edward/AFRL_project/csdl_dafoam_workspace/airfoil_case/results/rom_test2"
	# dafoamPrintInterval = 100
	# dataset_keyword 	= 'training_data2'
	# storage_location    = Path(dafoam_directory)

	# # Initial/reference values for DAFoam (best to use base conditions)
	# U0        = 206.53653128321116         # used for normalizing CD and CL
	# p0        = 19509.303373738785
	# T0        = 216.65227163736915
	# nuTilda0  = 4.5e-5
	# aoa0      = 1.416e-1
	# A0        = 0.1           #
	# rho0      = p0 / T0 / 287 # used for normalizing CD and CL

	# # Input parameters for DAFoam
	# da_options = {
	# 	"designSurfaces": ["wing"],
	# 	"solverName": "DARhoSimpleCFoam",
	# 	"primalMinResTol": 1.0e-8,
	# 	"primalVarBounds": {"pMin": 5000, "rhoMin": 0.05},
	# 	"primalBC": {
	# 		"U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
	# 		"p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
	# 		"T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
	# 		"nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
	# 		"useWallFunction": True,
	# 	},
	# 	"function": {
	# 		"drag": {
	# 			"type": "force",
	# 			"source": "patchToFace",
	# 			"patches": ["wing"],
	# 			"directionMode": "parallelToFlow",
	# 			"patchVelocityInputName": "patch_velocity",
	# 			"scale": 1.0, #1.0 / (0.5 * U0 * U0 * A0 * rho0),
	# 		},
	# 		"lift": {
	# 			"type": "force",
	# 			"source": "patchToFace",
	# 			"patches": ["wing"],
	# 			"directionMode": "normalToFlow",
	# 			"patchVelocityInputName": "patch_velocity",
	# 			"scale": 1.0, #1.0 / (0.5 * U0 * U0 * A0 * rho0),
	# 		},
	# 	},
	# 	"adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm", "useNonZeroInitGuess": False},
	# 	# transonic preconditioner to speed up the adjoint convergence
	# 	"transonicPCOption": 1,
	# 	"normalizeStates": {
	# 		"U": U0,
	# 		"p": p0,
	# 		"T": T0,
	# 		"nuTilda": nuTilda0 * 10.0,
	# 		"phi": 1.0,
	# 	},
	# 	"inputInfo": {
	# 		"aero_vol_coords": {
	# 			"type": "volCoord", 
	# 			"components": ["solver", "function"],
	# 		},
	# 		"patch_velocity": {
	# 			"type": "patchVelocity",
	# 			"patches": ["inout"],
	# 			"flowAxis": "x",
	# 			"normalAxis": "z",
	# 			"components": ["solver", "function"],
	# 		},
	# 		"pressure": {
	# 			"type": "patchVar",
	# 			"varName": "p",
	# 			"varType": "scalar",
	# 			"patches": ["inout"],
	# 			"components": ["solver", "function"],
	# 		},
	# 		"temperature": {
	# 			"type": "patchVar",
	# 			"varName": "T",
	# 			"varType": "scalar",
	# 			"patches": ["inout"],
	# 			"components": ["solver", "function"],
	# 		},
	# 	},
	# 	"writeAdjointFields": False,
	# 	"debug": False,
	# 	"printDAOptions": True,
	# 	"printInterval": dafoamPrintInterval
	# }

	# # region Mesh options
	# mesh_options = {
	# 	"gridFile": dafoam_directory,
	# 	"fileType": "OpenFOAM",
	# 	"symmetryPlanes": [],
	# }

	# dafoam_instance = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)
	# data_generator  = TrainingDataInterface(dafoam_instance=dafoam_instance, 
    #                                     storage_location=storage_location, 
    #                                     dataset_keyword=dataset_keyword,
    #                                     h5_file_base_name="point")

	# data = data_generator.load_h5(Path(storage_location)/dataset_keyword/"point_0.h5", only_distributed_data=False)

	# state_info    = data_generator.state_info # Get our state variable names
	# A_local       = np.array(np.concatenate([data["pod"]["modes"][state_var] for state_var in state_info.keys()], axis=0))[:, 0:n]
