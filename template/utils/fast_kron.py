import numpy as np
from scipy.linalg import block_diag
import tensorly as tl
from typing import Iterable


def matrix_by_transpose(A: np.ndarray):
    return A.T @ A


def kronecker_matrix_by_identity(A: np.ndarray, n: int) -> np.ndarray:

    """Given a p x q matrix A and the identity matrix I_n, computes the
    Kronecker product A kron I_n.

    Parameters
    ----------
    A : np.ndarray
        The p x q matrix A.
    n : int
        The order of the identity matrix I_n.

    Returns
    -------
    np.ndarray
        The Kronecker product np.kron(A, np.eye(n)), whose dimension is np x nq.
    """

    p, q = A.shape
    # The output is initialized as a 4th order array
    kron_prod = np.zeros((p, n, q, n))
    # Each of the values of A is repeated n times
    m = np.arange(n)
    # Index into the second and fourth axes and selecting all elements along
    # the rest to assign values from A. The values are broadcasted.
    kron_prod[:, m, :, m] = A
    return kron_prod.reshape(p * n, q * n)


def fast_kronecker_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:

    """Given a m x n matrix A and a p x q matrix B, computes the Kronecker
    product np.kron(A, B) using the broadcasting operation.

    Parameters
    ----------
    A : np.ndarray
        The m x n matrix A.
    B : np.ndarray
        The p x q matrix B.

    Returns
    -------
    np.ndarray
        The Kronecker product np.kron(A, B), whose dimension is mp x nq.
    """

    m, n = A.shape
    p, q = B.shape
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(m * p, n * q)


def kron_tens_prod(mat_list: Iterable[np.ndarray], T: np.ndarray) -> np.ndarray:

    """Given a n_1 x n_2 x ... x n_N multidimensional array T, computes
    efficiently  the product T x_N A_N x_{N-1} ... x_1 A_1, where A_i are
    the elements of the list mat_list and x_i represent the i-mode product.

    Parameters
    ----------
    mat_list : Iterable[np.ndarray]
        A list containing matrices with dimensions m_i x n_i
    T : np.ndarray
        The n_1 x n_2 x ... x n_N multidimensional array.

    Returns
    -------
    np.ndarray
        The resulting m_1 x m_2 x ... x m_N multidimensional array.
    """

    # Initialize computing the multidimensional array shape
    dims = list(T.shape)
    for i, A in enumerate(mat_list):
        # Substitute the new dimension of the i-mode.
        dims[i] = A.shape[0]
        # The process consists of unfolding the multidimensional array
        # along the i-mode, then perform the product by the matrix of
        # the list and finally reshaping the matrix with the new shape
        T = tl.fold(A @ tl.unfold(T, i), i, dims)
    return T


def penalization_term(mat_list: Iterable[np.ndarray]) -> np.ndarray:

    """Given the penalty matrices, computes the penalization term
    of the objective function.

    Parameters
    ----------
    mat_list : Iterable[np.ndarray]
        A list containing matrices with order n_i.

    Returns
    -------
    np.ndarray
        The penalization term of the objective function.
    """

    # Compute the order of the penalty matrices
    shapes = [mat.shape[0] for mat in mat_list]
    # Initialize the output as a zero matrix
    output = []
    for i in range(len(mat_list)):
        # Compute the shapes of the identity matrices located before
        # and after the penalty matrix for each summand
        left_identity_shape = int(np.prod([sh for sh in shapes[:i]]))
        right_identity_shape = int(np.prod([sh for sh in shapes[i + 1 :]]))
        # Compute the product np.kron(D, np.eye(right_identity_shape))
        right_kron_prod = kronecker_matrix_by_identity(
            mat_list[i], right_identity_shape
        )
        # Compute the product np.kron(np.eye(left_identity_shape), right_kron_prod)
        output.append(block_diag(*([right_kron_prod] * left_identity_shape)))
    return output
