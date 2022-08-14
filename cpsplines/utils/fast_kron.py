from typing import Iterable, List

import numpy as np
import tensorly as tl
from cpsplines.utils.box_product import box_product
from scipy.linalg import block_diag


def matrix_by_transpose(A: np.ndarray) -> np.ndarray:

    """
    Compute the product of a matrix by its transpose.

    Parameters
    ----------
    A : np.ndarray
        The input matrix

    Returns
    -------
    np.ndarray
        The product A.T @ A
    """

    return A.T @ A


def kronecker_matrix_by_identity(A: np.ndarray, n: int) -> np.ndarray:

    """
    Given a p x q matrix A and the identity matrix I_n, computes the
    Kronecker product np.kron(A, I_n). To compare performances with NumPy
    method, check https://stackoverflow.com/a/44461842/4983192

    Parameters
    ----------
    A : np.ndarray with shape (p, q)
        The input matrix to multiply.
    n : int
        The order of the identity matrix I_n.

    Returns
    -------
    np.ndarray with shape (np, nq)
        The Kronecker product np.kron(A, np.eye(n)).

    Raises
    ------
    ValueError
        If input array is not a matrix.
    """

    if A.ndim != 2:
        raise ValueError("Only two-dimensional arrays are allowed.")
    p, q = A.shape
    # The output is initialized as a 4th order array
    kron_prod = np.zeros((p, n, q, n))
    # Each of the values of A is repeated n times
    m = np.arange(n)
    # Index into the second and fourth axes and selecting all elements along
    # the rest to assign values from A. The values are broadcasted.
    kron_prod[:, m, :, m] = A
    # Finally reshape back to 2D
    return kron_prod.reshape(p * n, q * n)


def fast_kronecker_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:

    """
    Given a m x n matrix A and a p x q matrix B, computes the Kronecker
    product np.kron(A, B) using the broadcasting operation. To compare
    performances with NumPy method, check
    https://stackoverflow.com/a/56067827/4983192

    Parameters
    ----------
    A : np.ndarray with shape (m, n)
        The m x n matrix A.
    B : np.ndarray with shape (p, q)
        The p x q matrix B.

    Returns
    -------
    np.ndarray with shape (mp, nq)
        The Kronecker product np.kron(A, B).

    Raises
    ------
    ValueError
        If any input array is not a matrix.
    """

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Only two-dimensional arrays are allowed.")
    m, n = A.shape
    p, q = B.shape
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(m * p, n * q)


def matrix_by_tensor_product(
    matrices: Iterable[np.ndarray], T: np.ndarray
) -> np.ndarray:

    """
    Given a n_1 x n_2 x ... x n_N multidimensional array T, computes efficiently
    the product T x_N A_N x_{N-1} ... x_1 A_1, where A_i are the elements of
    `matrices` and x_i represent the i-mode product.

    Parameters
    ----------
    matrices : Iterable[np.ndarray]
        An iterable containing matrices with dimensions m_i x n_i
    T : np.ndarray
        The n_1 x n_2 x ... x n_N multidimensional array.

    Returns
    -------
    np.ndarray
        The resulting m_1 x m_2 x ... x m_N multidimensional array.

    Raises
    ------
    ValueError
        If any input array is not a matrix.
    """
    if any(x.ndim != 2 for x in matrices):
        raise ValueError("Only two-dimensional arrays are allowed.")

    # Initialize computing the multidimensional array shape
    dims = list(T.shape)
    for i, A in enumerate(matrices):
        # Substitute the new dimension of the i-mode.
        dims[i] = A.shape[0]
        # The process consists of unfolding the multidimensional array
        # along the i-mode, then perform the product by the matrix of
        # the list and finally reshaping the matrix with the new shape
        T = tl.fold(A @ tl.unfold(T, i), i, dims)
    return T


def penalization_term(matrices: Iterable[np.ndarray]) -> List[np.ndarray]:

    """
    Given the penalty matrices defined over every axis, computes the
    penalization term of the objective function.

    Parameters
    ----------
    matrices : Iterable[np.ndarray]
        A list containing matrices with order n_i.

    Returns
    -------
    np.ndarray
        The penalization term items of the objective function, whose orders are
        the product over all the n_i.

    Raises
    ------
    ValueError
        If any input array is not a matrix.
    """

    if any(x.ndim != 2 for x in matrices):
        raise ValueError("Only two-dimensional arrays are allowed.")

    # Compute the order of the penalty matrices
    shapes = [P.shape[1] for P in matrices]
    # Initialize the output as a zero matrix
    output = []
    for i, P in enumerate(matrices):
        # Compute the shapes of the identity matrices located before
        # and after the penalty matrix for each summand
        left_id_shape = int(np.prod([s for s in shapes[:i]]))
        right_id_shape = int(np.prod([s for s in shapes[i + 1 :]]))
        # Compute the product np.kron(D, np.eye(right_identity_shape))
        right_kron_prod = kronecker_matrix_by_identity(P, right_id_shape)
        # Compute the product np.kron(np.eye(left_identity_shape), right_kron_prod)
        output.append(block_diag(*([right_kron_prod] * left_id_shape)))
    return output


def weighted_double_kronecker(
    matrices: Iterable[np.ndarray], W: np.ndarray
) -> np.ndarray:
    """Computes np.kron(A_1, ..., A_N).T @ np.diag(W) @ np.kron(A_1, ..., A_N)
    efficiently, where A_i are matrices with dimensions m_i x n_i and W is a
    multidimensional array with shape n_1 x n_2 x ... x n_N. To do so, the
    product W x_N (A_N box A_N) x_{N-1} ... x_1 (A_1 box A_1) is carried out,
    where "box" denotes the box product, and the output, with new dimensions
    m_1^2 x ... x m_N^2, is reshaped into a m_1 x m_1 x ... x m_N x m_N
    multidimensional array. Then, the axes are permuted using the permutation
    (0, 2, 4, ..., 1, 3, 5, ...) and finally the output is reshaped into an
    square matrix with order m_1 · m_2 · ... · m_N. The algorithm for the
    two-dimensional case (two matrices involved) can be found in Eilers, Currie
    and Durban, (2006).

    Parameters
    ----------
    matrices : Iterable[np.ndarray]
        An iterable containing matrices with dimensions m_i x n_i.
    W : np.ndarray
        The n_1 x n_2 x ... x n_N multidimensional array.

    References
    ----------
    - Eilers, P. H., Currie, I. D., & Durbán, M. (2006). Fast and compact
      smoothing on large multidimensional grids. Computational Statistics & Data
      Analysis, 50(1), 61-76.

    Returns
    -------
    np.ndarray
        The resulting m_1 · m_2 · ... · m_N ordered matrix.
    """
    # Get the dimensions m_1, m_2, ..., m_N
    dim = [d.shape[1] for d in matrices]
    # Compute the box products for every matrices
    box_prod = [box_product(M, M).T for M in matrices]
    # Compute the matrix by tensor product and reshape into m_1 x m_1 x ... x
    # m_N x m_N
    out = matrix_by_tensor_product(box_prod, W).reshape(np.repeat(dim, 2))
    # np.transpose is used here to permute. To get (0, 2, 4, ..., 1, 3, 5, ...),
    # we use np.arange, reshape the values in a matrix with two columns and then
    # flatten the array
    return np.transpose(
        out, np.arange(2 * len(matrices)).reshape(len(matrices), 2).flatten("F")
    ).reshape((np.prod(dim), np.prod(dim)))
