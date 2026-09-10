"""Search-time normalization, including correction for float16 storage rounding."""
import numpy as np


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return normalized float32 rows without modifying the input/index on disk."""
    result = np.array(matrix, dtype=np.float32, copy=True)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError('Expected a finite two-dimensional embedding matrix')
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('Cannot normalize a zero embedding')
    result /= norms
    return result
