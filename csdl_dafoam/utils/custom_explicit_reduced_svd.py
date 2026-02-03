import numpy as np
import csdl_alpha as csdl

class customExplicitReducedSVD(csdl.CustomExplicitOperation):
	def __init__(self):
		super().__init__()

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

		if np.iscomplex(A.any):
			print('WARING: Complex array found. customExplicitReducedSVD currently only defines derivatives for real valued arrays.')
		
		U, S, VT = np.linalg.svd(A, full_matrices=False)
		
		# TODO: Implement some checks for S. See if we have any zero values or small differences between values
		
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

		if mode == 'fwd':
			raise NotImplementedError(
				'forward mode has not been implemented for customExplicitReducedSVD'
			)

		elif mode == 'rev':
			# ---------- analytic adjoint (real case, distinct s) ----------
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


			# Some easy cases
			if d_outputs["U"] is None and d_outputs["VT"] is None:
				# Trivial case where no gradients are specified
				if d_outputs["S"] is None:
					d_inputs["A"] += np.zeros_like(A)
				
				# Just only singular value gradient
				else:
					S_bar 	  = d_outputs["S"]	
					d_inputs["A"] += U @ (S_bar[..., :, None] * VT) if m >= n else (U * S[..., None, :]) @ VT

				return					
		
			# Extract output cotangents (default to zeros if not provided)
			U_bar  = d_outputs["U"]  if d_outputs["U"]  is not None else np.zeros_like(U)
			S_bar  = d_outputs["S"]  if d_outputs["S"]  is not None else np.zeros_like(S)
			VT_bar = d_outputs["VT"] if d_outputs["VT"] is not None else np.zeros_like(VT)

			V_bar  = VT_bar.swapaxes(-2, -1)

			# Get our dimensions
			m = U.shape[-2]
			n = V.shape[-2]
			k = U.shape[-1]			

			# Premultiply and skew some arrays
			UTU_bar = UT @ U_bar
			VTV_bar = VT @ V_bar

			skew_UTU_bar = UTU_bar - UTU_bar.swapaxes(-2, -1)
			skew_VTV_bar = VTV_bar - VTV_bar.swapaxes(-2, -1)

			# Construct our E array
			S2 = S**2
			E = S2[..., None, :] - S2[..., :, None]
			E[..., np.eye(k, dtype=bool)] = 1.0

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
				U_barSinv        = U_bar / S[..., None, :]
				U_barSinv_proj   = U_barSinv - U @ (UT @ U_barSinv)
				A_bar            = U @ M + U_barSinv_proj
				A_bar            = A_bar @ VT
			
			elif m < n:
				Sinv_V_barT      = (V_bar / S[..., None, :]).swapaxes(-2, -1)
				Sinv_V_barT_proj = Sinv_V_barT - (Sinv_V_barT @ V) @ VT
				A_bar            = M @ VT + Sinv_V_barT_proj
				A_bar 			 = U @ A_bar

			# Accumulate into input adjoints
			if "A" in d_inputs:
				d_inputs["A"] += A_bar
			else:
				d_inputs["A"] =  A_bar

		else:
			raise ValueError(f'"{mode}" not recognized. Only support "fwd" and "rev" modes')

			
			

