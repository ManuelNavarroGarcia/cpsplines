import numpy as np
from scipy.linalg import ldl


def cholesky_semidef(A: np.ndarray) -> np.ndarray:

    """Computes a Cholesky decomposition of a positive semidefinite symmetric
    matrix. This is done by applying a LDL decomposition implementation in
    scipy.linalg, which admits any symmetric matrix independent of his
    signature. The lower triangular matrix L is obtained multiplying the L
    matrix in the LDL decomposition by the square root of the diagonal elements
    in D, which are all non-negative due to the positive semidefinitess of the
    input matrix.

    Parameters
    ----------
    A : np.ndarray
        A positive semidefinite symmetric matrix.

    Returns
    -------
    np.ndarray
        The lower triangular matrix of a Cholesky decomposition of the input
        matrix.
    """

    # Compute the LDL decomposition of the matrix
    LDL_decomp = ldl(A, lower=False)
    # Compute the square root of the diagonal of D. In many cases, these values
    # may be negative due to numerical errors, and they are clipped to zero
    sqrt_decomp = np.sqrt(np.clip(np.diag(LDL_decomp[1]), a_min=0, a_max=1e16))
    # Get L in the Cholesky decomposition as sqrt(D) @ L
    return sqrt_decomp * LDL_decomp[0]
