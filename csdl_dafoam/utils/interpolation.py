
import numpy as np
from abc import ABC, abstractmethod
from typing import Union
from csdl_alpha import CustomExplicitOperation, Variable
import csdl_alpha as csdl


# region BASEINTERPOLATOR
class BaseInterpolatorClass(ABC):
    def __init__(self, 
                 query_point:Variable, 
                 sample_points:Union[np.ndarray, Variable],
                 apply_scaling:bool=True
        ):
        # Check if we have  CSDL variable for our samples (should be a rare occurance?)
        self.samples_are_variable = isinstance(sample_points, Variable)

        # Make sure that our variables are oriented correctly
        _query_point, _sample_points = self._check_dimensions(query_point, sample_points)
        
        self.apply_scaling = apply_scaling

        if apply_scaling:
            self._query_point, self._sample_points = self._get_scaled(_query_point, _sample_points)
        else:
            self._query_point   = _query_point
            self._sample_points = _sample_points
        

    # region weights
    @abstractmethod
    def weights(self):
        return None
    
    # region _check_dimensions
    def _check_dimensions(self, query_point:Variable, sample_points:Union[Variable, np.ndarray]):
        x   = query_point
        x_i = sample_points

        # Check sample points
        if isinstance(x_i, Variable):
            # Ensure 2D (N,d)
            if len(x_i.shape) == 1:
                x_i = csdl.reshape(x_i, (1, x_i.shape[0]))
            elif len(x_i.shape) == 2:
                pass
            else:
                raise ValueError(f"Sample points must be 1D or 2D, got shape {x_i.shape}")
            
        else: # NumPy Array
            x_i = np.atleast_2d(x_i)
            x_i = x_i

        N, d = x_i.shape

        # Check query point
        if isinstance(x, Variable):
            if len(x.shape) == 2 and x.shape[0] == 1:
                x = csdl.reshape(x, (x.shape[1], ))
            elif len(x.shape) == 1:
                pass
            else:
                raise ValueError(f"query_point must be 1D or shape (1,d), got {x.shape}")
            
        else:
            x = np.atleast_1d(x)
            if x.ndim == 2 and x.shape[0] == 1:
                x = x.ravel()
            elif x.ndim == 1:
                pass
            else:
                raise ValueError(f"query_point must be 1D or shape (1,d), got {x.shape}")
            
        # Dimension consistency check
        if x.shape[0] != d:
            raise ValueError(f"Query point dimension {x.shape[0]} "
                f"does not match sample points dimension {d}")
        
        return x, x_i
    

    # region _get_scaled
    def _get_scaled(self, _query_point, _sample_points):
        x   = _query_point
        x_i = _sample_points

        if self.samples_are_variable:
            lower_bounds = csdl.minimum(x_i, axis=0)
            upper_bounds = csdl.maximum(x_i, axis=0)
            sample_range = upper_bounds - lower_bounds
        else:
            lower_bounds = np.min(x_i, axis=0)
            upper_bounds = np.max(x_i, axis=0)
            sample_range = upper_bounds - lower_bounds
        
        return_query_point   = (x   - lower_bounds) / sample_range
        return_sample_points = (x_i - lower_bounds) / sample_range

        return return_query_point, return_sample_points



# region IDWINTERPOLATOR
class IDWInterpolator(BaseInterpolatorClass):
    def __init__(self, 
                 query_point:Variable, 
                 sample_points:Union[np.ndarray, Variable], 
                 exponent:float=2,
                 apply_scaling:bool=True
    ):
        super().__init__(query_point=query_point,
                       sample_points=sample_points,
                       apply_scaling=apply_scaling)
        
        self.exponent = exponent


    # region weights
    def weights(self):
        x   = self._query_point
        x_i = self._sample_points
        p   = self.exponent

        diffs    = csdl.expand(x, x_i.shape, 'j->ij') - x_i
        d        = csdl.norm(diffs, axes=(1,))
        a        = 1 / d ** p

        return a / csdl.sum(a)
    


# region RBFINTERPOLATOR
class RBFInterpolator(BaseInterpolatorClass):
    def __init__(self, 
                 query_point:Variable, 
                 sample_points:Union[np.ndarray, Variable],
                 kernel:str='gaussian', 
                 kernel_parameter:float|None=None,
                 apply_scaling:bool=True,
                 positive_non_reproducing_weights:bool=False
    ):
        super().__init__(query_point=query_point,
                       sample_points=sample_points,
                       apply_scaling=apply_scaling)
        
        self.kernel                           = kernel
        self.positive_non_reproducing_weights = positive_non_reproducing_weights
        self.kernel_parameter = kernel_parameter if kernel_parameter is not None else self._estimate_parameter()

        # We'll need the RBF interpolation matrix
        self.interpolation_matrix = self._setup_interpolation_matrix()
    

    # region weights
    def weights(self):
        x   = self._query_point
        x_i = self._sample_points
        A   = self.interpolation_matrix

        diffs    = csdl.expand(x, x_i.shape, 'j->ij') - x_i
        r2       = csdl.sum(diffs * diffs, axes=(1, ))
        b        = self._phi_from_r2(r2)

        if self.positive_non_reproducing_weights:
            return b / csdl.sum(b)
        else:
            return csdl.solve_linear(A, b)


    # region _phi
    def _phi_from_r2(self, r2):
        eps     = self.kernel_parameter
        kernel  = self.kernel
        is_csdl = isinstance(r2, Variable)

        if kernel.lower()=="gaussian":
            return csdl.exp(-eps * eps * r2) if is_csdl else np.exp(-eps * eps * r2)

        elif kernel.lower()=="inverse_quadratic":
            return 1 / (1 + (eps * eps * r2))
            
        elif kernel.lower()=="inverse_multiquadratic":
            return 1 / csdl.sqrt(1 + (eps * eps * r2)) if is_csdl else 1 / np.sqrt(1 + (eps * eps * r2))
            
        else:
            raise NotImplementedError(f"RBF kernel, {kernel}, not yet implemented.")


    # region _setup_interpolation_matrix
    def _setup_interpolation_matrix(self):
        x_i = self._sample_points
        phi_from_r2 = self._phi_from_r2

        # Compute assuming CSDL variable
        if self.samples_are_variable:
            raise NotImplementedError("Haven't implemented variable sample points")
        
        elif isinstance(x_i, np.ndarray):
            diff = x_i[:, None, :] - x_i[None, :, :]
            r2   = np.sum(diff * diff, axis=2)
            A    = phi_from_r2(r2)

        return A
    

    # region _estimate_parameter
    def _estimate_parameter(self):
        kernel = self.kernel.lower()

        if isinstance(self._sample_points, Variable):
            raise NotImplementedError("Haven't implemented the auto-computation of kernel parameter for variable sample points.")

        x_i = self._sample_points

        # For normalized (positive non-reproducing) weights, locality matters more than
        # interpolation matrix conditioning, so tune ε to the nearest-neighbor distance
        # so the kernel decays to ~1/e at that spacing rather than being nearly flat.
        if self.positive_non_reproducing_weights:
            diff = x_i[:, None, :] - x_i[None, :, :]
            d    = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            d_nn = np.mean(np.min(d, axis=1))  # mean nearest-neighbor distance

            return 1.0 / d_nn

        c      = 3 # a constant for the scaled values
        N, dim = x_i.shape

        # If the data is scaled between 0 and 1, these values should be decent?
        if self.apply_scaling:
            if kernel == "gaussian":
                return N ** (2 / dim) / c ** 3
            elif kernel == "inverse_quadratic":
                return N ** (1 / dim) / c
            elif kernel == "inverse_multiquadratic":
                return N ** (1 / dim) / c

        # If we still have unscaled data, then we can try using some values based off
        # the mean distance (these would normalize the r values in the basis functions)
        else:
            diff = x_i[:, None, :] - x_i[None, :, :]
            d    = np.linalg.norm(diff, axis=2)
            d_mn = 1 / x_i.shape[0] * (np.sum(d) - np.sum(np.diag(d)))

            if kernel == "gaussian":
                return 1 / d_mn ** 2
            elif kernel == "inverse_quadratic" or kernel == "inverse_multiquadratic":
                return 1 / d_mn



# region _CUBICPOLYNOMIALWEIGHTOP
class _CubicPolynomialWeightOp(csdl.CustomExplicitOperation):
    def __init__(self, sample_points: np.ndarray, cutoff_value: float = 0.8):
        super().__init__()
        self.x_i = sample_points  # (N, d)
        self.c   = cutoff_value

    def evaluate(self, query_point: Variable):
        N = self.x_i.shape[0]
        self.declare_input("x", query_point)
        weights = self.create_output("weights", (N,))
        return weights

    def compute(self, input_vals, output_vals):
        mu_hat = input_vals["x"]  # (d,)
        x_i    = self.x_i         # (N, d)
        c      = self.c
        N      = x_i.shape[0]

        diff  = mu_hat[None, :] - x_i  # (N, d)
        delta = np.linalg.norm(diff, axis=1)  # (N,)

        # Coincident-point edge case
        if np.any(delta == 0.0):
            w = np.zeros(N)
            w[np.argmin(delta)] = 1.0
            output_vals["weights"] = w
            return

        delta_min = np.min(delta)
        delta_max = np.max(delta)
        delta_cut = c * delta_min + (1.0 - c) * delta_max

        # Degenerate: all equidistant or cutoff collapses
        if delta_cut <= delta_min:
            output_vals["weights"] = np.ones(N) / N
            return

        t     = (delta - delta_min) / (delta_cut - delta_min)
        mask  = t < 1.0
        w_raw = np.where(mask, (1.0 + 2.0 * t) * (1.0 - t) ** 2, 0.0)

        S = np.sum(w_raw)
        if S == 0.0:
            output_vals["weights"] = np.ones(N) / N
            return

        output_vals["weights"] = w_raw / S

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        mu_hat = input_vals["x"]  # (d,)
        x_i    = self.x_i         # (N, d)
        c      = self.c
        N, d   = x_i.shape

        diff  = mu_hat[None, :] - x_i  # (N, d)
        delta = np.linalg.norm(diff, axis=1)  # (N,)

        if np.any(delta == 0.0):
            derivatives["weights", "x"] = np.zeros((N, d))
            return

        j_min     = np.argmin(delta)
        j_max     = np.argmax(delta)
        delta_min = delta[j_min]
        delta_max = delta[j_max]
        delta_cut = c * delta_min + (1.0 - c) * delta_max
        span      = delta_cut - delta_min

        if span <= 0.0:
            derivatives["weights", "x"] = np.zeros((N, d))
            return

        t     = (delta - delta_min) / span
        mask  = t < 1.0
        w_raw = np.where(mask, (1.0 + 2.0 * t) * (1.0 - t) ** 2, 0.0)
        S     = np.sum(w_raw)

        if S == 0.0:
            derivatives["weights", "x"] = np.zeros((N, d))
            return

        # Unit vectors ∂δⱼ/∂μ̂ = (μ̂ − xⱼ) / δⱼ,  shape (N, d)
        e     = diff / delta[:, None]
        e_min = e[j_min]  # (d,)
        e_max = e[j_max]  # (d,)

        # ∂span/∂μ̂ = (1−c)(e_max − e_min),  shape (d,)
        d_span = (1.0 - c) * (e_max - e_min)

        # Full dt_j/dμ̂ accounting for δ_min and span varying with μ̂:
        # dt_j/dμ̂ = (e_j − e_min)/span − t_j · d_span/span,  shape (N, d)
        dt_d_muhat = (e - e_min[None, :]) / span - t[:, None] * d_span[None, :] / span

        # dw_raw_j/dt_j = −6t(1−t),  shape (N,)
        dw_dt = np.where(mask, -6.0 * t * (1.0 - t), 0.0)

        # dw_raw_j/dμ̂ = dw_dt_j · dt_j/dμ̂,  shape (N, d)
        dw_d_muhat = dw_dt[:, None] * dt_d_muhat

        # dS/dμ̂ = Σᵢ dw_raw_i/dμ̂,  shape (d,)
        dS_d_muhat = np.sum(dw_d_muhat, axis=0)

        # J[j,:] = (S · dw_j/dμ̂ − w_raw_j · dS/dμ̂) / S²
        J = (S * dw_d_muhat - w_raw[:, None] * dS_d_muhat[None, :]) / S ** 2

        derivatives["weights", "x"] = J



#region CUBICPOLYNOMIALINTERPOLATOR
class CubicPolynomialInterpolator(BaseInterpolatorClass):
    def __init__(self,
                query_point:Variable,
                sample_points:Union[np.ndarray, Variable],
                cutoff_value:float=0.8,
                apply_scaling:bool=True
        ):
        super().__init__(query_point=query_point,
                    sample_points=sample_points,
                    apply_scaling=apply_scaling)

        self.cutoff_value = cutoff_value

    # region weights
    def weights(self):
        if self.samples_are_variable:
            raise NotImplementedError("CubicPolynomialInterpolator requires numpy sample_points")
        op = _CubicPolynomialWeightOp(self._sample_points, self.cutoff_value)
        return op.evaluate(self._query_point)



# region INVERSEDISTANCEWEIGHTINGCOMPONENT
class InverseDistanceWeightingComponent(CustomExplicitOperation):
    def __init__(self, data:np.ndarray, exponent:float=2):
        super().__init__()
        self.x_set = data
        self.p     = exponent


    # region evaluate
    def evaluate(self, query_point:Variable):
        self.declare_input('x_target', query_point)

        weights = self.create_output('weights', shape=(self.x_set.shape[0], ))

        self.declare_derivative_parameters("weights", "x_target")

        return weights
    

    # region compute
    def compute(self, input_vals, output_vals):
        x       = input_vals["x_target"]
        x_set   = self.x_set

        d = np.sqrt(np.sum((x_set - x) ** 2, axis=1))
        a = 1 / d ** self.p
        S = np.sum(a)

        output_vals["weights"] = a / S


    # region compute_derivatives
    def compute_derivatives(self, inputs, outputs, derivatives):
        x     = inputs["x_target"]
        x_set = self.x_set
        p = self.p
        eps = 1e-12

        diff = x - x_set                      # (N_pts, N_dim)
        d = np.sqrt(np.sum(diff**2, axis=1))  # (N_pts,)
        d_safe = np.maximum(d, eps)

        a = d_safe ** (-p)
        S = np.sum(a)

        # da_i/dx
        da_dx = -p * d_safe[:,None]**(-p-2) * diff   # (N_pts, N_dim)

        # dS/dx
        dS_dx = np.sum(da_dx, axis=0)                # (N_dim,)

        # Jacobian
        der = (da_dx * S - a[:,None] * dS_dx[None,:]) / S**2

        derivatives["weights", "x_target"] = der




# region main
if __name__ == "__main__":

    import csdl_alpha as csdl
    from smt.sampling_methods import LHS

    variable_dimension = 2
    num_samples        = 50
    exponent           = 4
    factor             = 1 #10000
    margin             = 0.2
    cutoff_value       = 0.8


    x_i_min  = np.array([0, 7000])
    x_i_max  = np.array([5, 13000])
    x_i_span = x_i_max - x_i_min

    # Generate samples
    xlimits     = np.array([x_i_min, x_i_max]).T
    sampler     = LHS(xlimits=xlimits, criterion='m', seed=0)
    x_i         = sampler(num_samples)  # Shape: (num_samples, total_elements)

    # Generate value
    x_value  = x_i_min + x_i_span * (margin + (1 - 2 * margin) * np.random.random(size=(variable_dimension, )))

    recorder = csdl.Recorder(inline=True, debug=True)
    recorder.start()

    x   = csdl.Variable(value=x_value)

    idw_weights         = IDWInterpolator(query_point=x, sample_points=x_i, exponent=exponent).weights()
    idw_weights_unscld  = IDWInterpolator(query_point=x, sample_points=x_i, exponent=exponent, apply_scaling=False).weights()

    rbf_weights         = RBFInterpolator(query_point=x, sample_points=x_i, kernel='gaussian', kernel_parameter=None).weights()
    rbf_weights_unscld  = RBFInterpolator(query_point=x, sample_points=x_i, kernel='gaussian', kernel_parameter=None, apply_scaling=False).weights()

    cbc_weights         = CubicPolynomialInterpolator(query_point=x, sample_points=x_i, cutoff_value=cutoff_value).weights()
    cbc_weights_unscld  = CubicPolynomialInterpolator(query_point=x, sample_points=x_i, cutoff_value=cutoff_value, apply_scaling=False).weights()

    weightFun      = InverseDistanceWeightingComponent(data=x_i, exponent=exponent)
    idw_ce_weights = weightFun.evaluate(x)

    idw_weights_expanded        = csdl.expand(idw_weights,          x_i.shape, 'i->ij')
    idw_weights_unscld_expanded = csdl.expand(idw_weights_unscld,   x_i.shape, 'i->ij')
    rbf_weights_expanded        = csdl.expand(rbf_weights,          x_i.shape, 'i->ij')
    rbf_weights_unscld_expanded = csdl.expand(rbf_weights_unscld,   x_i.shape, 'i->ij')
    cbc_weights_expanded        = csdl.expand(cbc_weights,       x_i.shape, 'i->ij')
    cbc_weights_unscld_expanded = csdl.expand(cbc_weights_unscld,   x_i.shape, 'i->ij')
    idw_ce_weights_expanded     = csdl.expand(idw_ce_weights,       x_i.shape, 'i->ij')
    
    

    x_idw        = csdl.sum(idw_weights_expanded * x_i,         axes=(0,))
    x_idw_unscld = csdl.sum(idw_weights_unscld_expanded * x_i,  axes=(0,))
    x_rbf        = csdl.sum(rbf_weights_expanded * x_i,         axes=(0,))
    x_rbf_unscld = csdl.sum(rbf_weights_unscld_expanded * x_i,  axes=(0,))
    x_cbc        = csdl.sum(cbc_weights_expanded * x_i,         axes=(0,))
    x_cbc_unscld = csdl.sum(cbc_weights_unscld_expanded * x_i,  axes=(0,))
    x_idw_ce     = csdl.sum(idw_ce_weights_expanded * x_i,      axes=(0,))

    
    obj = csdl.norm(x_cbc - x)

    x.set_as_design_variable(lower=np.zeros(variable_dimension,), upper=np.ones(variable_dimension,))
    obj.set_as_objective()

    recorder.stop()

    import modopt as mo

    sim         = csdl.experimental.PySimulator(recorder=recorder)
    prob        = mo.CSDLAlphaProblem(problem_name="test", simulator=sim)
    optimizer   = mo.SLSQP(problem=prob, solver_options={'ftol':1e-6, 'maxiter':20})
    optimizer.check_first_derivatives(step=1e-6)
    
    import matplotlib.pyplot as plt

    print(f"Sum of IDW weights (Scaled): {np.sum(idw_weights.value)}")
    print(f"Sum of IDW weights         : {np.sum(idw_weights_unscld.value)}")
    print(f"Sum of RBF weights (Scaled): {np.sum(rbf_weights.value)}")
    print(f"Sum of RBF weights         : {np.sum(rbf_weights_unscld.value)}")
    print(f"Sum of CBC weights (Scaled): {np.sum(cbc_weights.value)}")
    print(f"Sum of CBC weights         : {np.sum(cbc_weights_unscld.value)}")
    print(f"Sum of IDW weights         : {np.sum(idw_ce_weights.value)}")

    print(f"")
    
    
    plt.figure()
    plt.scatter(x_i[:, 0],              x_i[:, 1],              marker='o', label="Training")
    plt.scatter(x.value[0],             x.value[1],             marker='o', label="Target")
    plt.scatter(x_idw.value[0],         x_idw.value[1],         marker='+', label="IDW (scaled)")
    plt.scatter(x_idw_unscld.value[0],  x_idw_unscld.value[1],  marker='+', label="IDW")
    plt.scatter(x_rbf.value[0],         x_rbf.value[1],         marker='^', label="RBF (scaled)")
    plt.scatter(x_rbf_unscld.value[0],  x_rbf_unscld.value[1],  marker='v', label="RBF")
    plt.scatter(x_cbc.value[0],         x_cbc.value[1],         marker='^', label="CBC (scaled)")
    plt.scatter(x_cbc_unscld.value[0],  x_cbc_unscld.value[1],  marker='v', label="CBC")
    plt.scatter(x_idw_ce.value[0],      x_idw_ce.value[1],      marker='x', label="IDW (CustomExplicit)")
    plt.legend()
    for i, txt in enumerate(rbf_weights.value):
        plt.annotate(f"{txt:.3f}", (x_i[i, 0], x_i[i, 1]))


    plt.figure()
    plt.plot(idw_weights.value,         label="IDW (scaled)")
    plt.plot(idw_weights_unscld.value,  label="IDW")
    plt.plot(rbf_weights.value,         label="RBF (scaled)")
    plt.plot(rbf_weights_unscld.value,  label="RBF")
    plt.plot(cbc_weights.value,         label="CBC (scaled)")
    plt.plot(cbc_weights_unscld.value,  label="CBC")
    plt.plot(idw_ce_weights.value,      label="IDW (CustomExplicit)")
    plt.legend()
    plt.show()
