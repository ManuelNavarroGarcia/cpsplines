import numpy as np


def box_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:

    """
    Given a m x n matrix A and a m x p matrix B, computes the face-splitting
    product or box product using that
                A box B = (A kron 1_n^T) circ (1_n^T kron B),
    where 1_n is a vector of ones with length n.

    Parameters
    ----------
    A : np.ndarray with shape (m, n)
        The m x n matrix A.
    B : np.ndarray with shape (m, p)
        The m x p matrix B.

    Returns
    -------
    np.ndarray
        The resulting m x np array.

    Raises
    ------
    ValueError
        If any input array is not a matrix.
    """

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Only two-dimensional arrays are allowed.")
    return np.multiply(np.repeat(A, B.shape[1], axis=1), np.tile(B, A.shape[1]))
