import numpy as np
from typing import Dict, Tuple, Union, Iterable

def _infer_shape_from_key(key) -> Tuple[int, ...]:
    """
    Infers shape from a CSDL variable-like object (with `.value.shape`)
    or treats as scalar if no shape attribute is found.
    """
    if not isinstance(key, str): 
        if hasattr(key, "value") and hasattr(key.value, "shape"):
            return tuple(key.value.shape)
    else:
        TypeError('Please use the actual CSDL variable instead of a string in your {variable: limits} dictionary')

def build_xlimits(var_limits: Dict[Union[str, object], Iterable[float]]):
    """
    Build xlimits array for SMT sampling, inferring shapes automatically from
    CSDL variable instances (keys with `.value.shape`) or treating as scalars otherwise.
    """
    rows = []
    labels = []
    slicer = {}
    shapes = {}
    cursor = 0

    for var_key, lim in var_limits.items():
        lim = list(lim)
        if len(lim) != 2:
            raise ValueError(f"{var_key}: limits must be [low, high], got {lim}")

        shp = _infer_shape_from_key(var_key)
        shapes[var_key] = shp

        if shp == ():  # scalar
            rows.append(lim)
            labels.append((var_key, ()))
            slicer[var_key] = slice(cursor, cursor + 1)
            cursor += 1
        else:
            count = int(np.prod(shp))
            start = cursor
            for flat_idx in range(count):
                idx = np.unravel_index(flat_idx, shp)
                rows.append(lim)
                labels.append((var_key, idx))
                cursor += 1
            slicer[var_key] = slice(start, start + count)

    xlimits = np.array(rows, dtype=float)
    return xlimits, labels, slicer, shapes

def reshape_samples(
    X: np.ndarray,
    slicer: Dict[Union[str, object], slice],
    shapes: Dict[Union[str, object], Tuple[int, ...]],
):
    """
    Convert SMT samples back into dict of variable-shaped arrays.
    Keys will match the original var_keys (string or object).
    """
    N, D = X.shape
    out = []
    for k in range(N):
        sample_k = {}
        for key, sl in slicer.items():
            shp = shapes[key]
            block = X[k, sl]
            if shp == ():
                sample_k[key] = float(block[0])
            else:
                sample_k[key] = block.reshape(shp)  # preserve original shape
        out.append(sample_k)
    return out
