import numpy as np
import csdl_alpha as csdl
import os
from csdl_dafoam.utils.custom_explicit_reduced_svd import customExplicitReducedSVD
from mpi4py import MPI


# region GRASSMANN
class Grassmann:
    def __init__(self, n: int, k: int):
        """
        Represent the Grassmann manifold Gr(n, k):
        - n: ambient dimension
        - k: subspace dimension
        """
        self.n = n
        self.k = k

    # region exp
    def exp(self, Y0, Ydot):
        """Exponential map at point Y0 with tangent vector Ydot."""

        if len(Y0.shape) > 2 or len(Ydot.shape) > 2:
            raise NotImplementedError("Batch mode not implemented yet for Grassmann exp map.")

        else:
            csdl_svd = customExplicitReducedSVD()
            U, S, VT  = csdl_svd.evaluate(Ydot)
            return U @ (csdl.einsum(VT.T(), csdl.cos(S), action='ij,j->ij')) + csdl.einsum(U, csdl.sin(S), action='ij,j->ij')


    # region log
    def log(self, Y0, Y1):
        """Logarithm map: tangent vector at Y0 pointing to Y1."""

        if len(Y0.shape) > 2 or len(Y1.shape) > 2:
            raise NotImplementedError("Batch mode not implemented yet for Grassmann log map.")

        else:
            csdl_svd1 = customExplicitReducedSVD()
            csdl_svd2 = customExplicitReducedSVD()
            I = np.eye(Y0.shape[0])

            P, S, RT = csdl_svd1.evaluate(Y1.T() @ Y0)
            Y_star   = Y1 @ (P @ RT)
            L        = (I - Y0 @ Y0.T()) @ Y_star
            Q, E, VT = csdl_svd2.evaluate(L)
            theta    = csdl.arcsin(E)
            return Q @ csdl.einsum(theta, VT, action='i,ij->ij')


    # region distance
    def distance(self, Y0, Y1):
        """Geodesic distance between two points on the manifold."""
        pass


    # region geodesic
    def geodesic(self, Y0, Ydot, t):
        """Point along geodesic starting at Y0 in direction Ydot."""
        pass
    

    # region project_tangent
    def project_tangent(self, Y, Z):
        """Project matrix Z onto tangent space at Y."""
        pass


# region GRASSMANNINTERPOLATOR
class GrassmannInterpolator:
    def __init__(self, manifold: Grassmann, parameters:list[np.ndarray]=[], points:list[np.ndarray]=[], normalize_parameters:bool=True):
        self.manifold   = manifold
        self.parameters = parameters
        self.points     = points
        self.normalize_parameters = normalize_parameters

    def add_point(self, mu, Y):
        """Add new sample point (parameter mu, basis Y)."""
        self.parameters.append(mu)
        self.points.append(Y)

    def interpolate(self, mu_new):
        """
        Interpolate subspace at new parameter.
        Typically: choose reference Y_ref,
        log map all others to tangent space,
        interpolate tangent vectors,
        map back with exp.
        """

        manifold = self.manifold

        # Find closest point as reference (in parameter space)
        distances = [np.linalg.norm(mu_new - mu) for mu in self.parameters]
        sorted_indices = np.argsort(distances)
        idx_ref   = sorted_indices[0]
        Y_ref     = self.points[idx_ref]

        # Log map all points to tangent space at Y_ref
        tangent_vectors = []
        for i, Y in enumerate(self.points):
            if i == idx_ref:
                tangent_vectors.append(np.zeros_like(Y_ref))
            else:
                Ydot = manifold.log(Y_ref, Y)
                tangent_vectors.append(Ydot)
        
        # Interpolate tangent vectors

        # Exp map back to manifold

        
        pass



def weighted_interp(values:np.ndarray, desired_value:np.ndarray, normalize:bool=True, method:str='idw', 
                    basis_function:str='gaussian', distance_exponent:float=2.0, gauss_constant:float=np.inf,
                    multi_constant:float=np.inf, thin_constant:float=np.inf, element_weights:np.ndarray=None, semivariogram:str='linear'):
    
    """ Interpolation function for scalar or vector values based on various methods.

    Parameters
    ----------
    values : np.ndarray
        Array of known values to interpolate from. Shape (num_samples, value_dim).
    desired_value : np.ndarray
        The point at which to interpolate. Shape (value_dim,).
    normalize : bool, optional
        If True, min-max normalize each column using values' min and max, by default True.
    method : str, optional
        Interpolation method: 'idw', 'rbf', 'kriging', by default 'idw'.
    basis_function : str, optional
        Basis function for RBF: 'gaussian', 'multiquadric', 'thin_plate', by default 'gaussian'.
    distance_exponent : float, optional
        Exponent for distance in IDW, by default 2.0. (1/x^p)
    guass_constant, multi_constant, thin_constant : float, optional
        Tuning constants for RBF basis functions, by default np.inf.
    element_weights : np.ndarray, optional
        Weights for each sample point, by default None.
    semivariogram : str, optional
        Semivariogram model for Kriging: 'linear', 'spherical', 'exponential', by default 'linear'.

    Returns
    -------
    np.ndarray
        Interpolation weights for each row of values. Shape (num_samples,).
    np.ndarray
        Euclidian distances (with element_weights applied) from desired_value to each row in values. Shape (num_samples,).
    """

    # Make sure arrays are numpy ndarrays and have correct shapes
    values        = np.asarray(values, dtype=float)
    desired_value = np.asarray(desired_value, dtype=float).flatten()

    if values.ndim != 2:
        raise ValueError("values must be a 2D array with shape (num_samples, value_dim).")
    m, n = values.shape

    # Dimension checks
    if n != desired_value.size:
        if m == desired_value.size:
            warnings.warn("desired_value has incompatible shape; attempting to reshape.")
            values = values.T
            m, n = values.shape
        else:
            raise ValueError("Incompatible shapes between values and desired_value.")
    
    # Normalize values and desired_value if specified
    if element_weights is None:
        element_weights = np.ones(n)
    else:
        element_weights = np.asarray(element_weights, dtype=float).flatten()
        if element_weights.size != n:
            raise ValueError("element_weights must have shape (value_dim,).")

    s = element_weights.sum()

    if s == 0:
        raise ValueError("Sum of element_weights cannot be zero.")
    if not np.isclose(s, 1.0):
        element_weights = element_weights / s

    # Normalize columns if requested (min-max per column)
    if normalize:
        min_values = np.min(values, axis=0)
        max_values = np.max(values, axis=0)
        denom      = max_values - min_values
        denom[denom == 0] = 1.0  # Prevent division by zero
        values = (values - min_values) / denom
        desired_value = (desired_value - min_values) / denom

    # Compute weighted Euclidean distances
    diffs     = (values - desired_value) * element_weights
    distances = np.sqrt(np.sum(diffs **2, axis=1))

    # Default parameters
    method_ls = method.lower()
    p         = distance_exponent
    basis     = basis_function.lower()

    # Sigma logic: min(gauss_constant, min(distances))
    dist_min = distances.min()
    dist_max = distances.max()
    sigma    = min(gauss_constant, dist_min)
    if sigma == 0:
        sigma = 0.1 * (dist_max - dist_min)

    b = min(multi_constant, np.mean([distances.mean(), dist_max]))
    c = min(thin_constant, dist_min)

    # Case for exact match
    zero_idx = np.where(distances == 0)[0]
    if zero_idx.size > 0:
        weights = np.zeros(m)
        weights[zero_idx[0]] = 1.0
        return weights, distances

    # Choose weighting / basis
    if method_lc == "idw":
        def phi(x): return 1.0 / np.power(x, p)
    
    elif method_lc == "rbf":
        if basis == "gaussian":
            def phi(x): return np.exp(- (x ** 2) / (2 * sigma ** 2))
        elif basis == "multiquadric":
            def phi(x): return np.sqrt(x ** 2 + b ** 2)
        elif basis == "thin_plate":
            def phi(x): return (x ** 2) * np.log(x / c)
        else:
            raise ValueError(f"Unknown basis_function '{basis_function}' for RBF.")
    
    elif method_lc == "kriging":

        var_vec = 1 / element_weights

    
    
