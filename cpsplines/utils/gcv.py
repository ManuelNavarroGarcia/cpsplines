from functools import reduce
from typing import Iterable, Tuple, Union

import numpy as np
from cpsplines.utils.fast_kron import (
    fast_kronecker_product,
    matrix_by_tensor_product,
    penalization_term,
)


def gcv_mat(
    B_mul: Iterable[np.ndarray], D_mul: Iterable[np.ndarray]
) -> Tuple[np.ndarray]:

    """
    Generate the matrices used to compute the hat matrix on the Generalized
    Cross Validation.

    Parameters
    ----------
    B_mul : Iterable[np.ndarray]
        An iterable containing the products B_i.T @ B_i, where B_i is the design
        matrix of the B-spline basis along the i-th axis.
    D_mul : Iterable[np.ndarray]
        An iterable containing the penalty matrix P_i, where P_i is the penalty
        matrix along the i-th axis (without multiplying the smoogthing
        parameters).

    Returns
    -------
    Tuple[np.ndarray]
        The matrices used in the GCV computation. The first is the B-spline
        bases related matrix and the rest are the penalty matrices.
    """
    # The order must be reversed since vec(Y) = Bvec(Theta), where the
    # vectorization is performed stacking the columns, so B is the Kronecker
    # product of B_N kron ··· kron B_1
    bases_term = reduce(fast_kronecker_product, B_mul[::-1])
    penalty_term = penalization_term(matrices=D_mul[::-1])
    return tuple([bases_term] + penalty_term)


def quadratic_term(
    sp: Iterable[Union[int, float]], Q_matrices: Iterable[np.ndarray]
) -> np.ndarray:

    """
    Given a smoothing parameter vector and the set of matrices used in the
    computation of the hat matrix, computes the matrix to be inverted in this
    computation.

    Parameters
    ----------
    sp : Iterable[Union[int, float]]
        The smoothing paramater vector.
    Q_matrices : Iterable[np.ndarray]
        The array of matrices of the matrices to be inverted. The first is
        related to the B-spline bases and the rest to the penalty terms along
        each dimension.

    Returns
    -------
    np.ndarray
        The matrix to be inverted on the computation of the Generalized Cross
        Validation.

    Raises
    ------
    ValueError
        If matrices dimensions are different.
    """

    if len(set([A.shape for A in Q_matrices])) > 1:
        raise ValueError("Matrices dimensions must agree.")
    penalty_term = np.zeros(Q_matrices[0].shape)
    reversed_sp = sp[::-1]
    for i, s in enumerate(reversed_sp):
        penalty_term += np.multiply(s, Q_matrices[i + 1])
    return Q_matrices[0] + penalty_term


def explicit_y_hat(
    Q: np.ndarray, B_weighted: Iterable[np.ndarray], y: np.ndarray
) -> np.ndarray:

    """
    Computes the fitted values for the response variable using the explicit
    formula on the unconstrained framework.

    Parameters
    ----------
    Q : np.ndarray
        The resulting matrix from `quadratic_term`.
    B_weighted : Iterable[np.ndarray]
        The weighted design matrices from B-spline basis.
    y : np.ndarray
        The response variable sample.

    Returns
    -------
    np.ndarray
        The fitted values for the response variable.
    """
    y_contribution = matrix_by_tensor_product([B.T for B in B_weighted], y).flatten("F")
    theta = np.reshape(
        np.linalg.solve(Q, y_contribution),
        tuple([mat.shape[1] for mat in B_weighted]),
        order="F",
    )
    return matrix_by_tensor_product([mat for mat in B_weighted], theta)


def GCV(
    sp: Iterable[Union[int, float]],
    B_weighted: Iterable[np.ndarray],
    Q_matrices: Iterable[np.ndarray],
    y: np.ndarray,
) -> float:

    """Computes the Generalized Cross Validation (Golub et al., 1979).

    Parameters
    ----------
    sp : Iterable[Union[int, float]]
        The smoothing paramater vector.
    B_weighted : Iterable[np.ndarray]
        The weighted design matrices from B-spline basis.
    Q_matrices : Iterable[np.ndarray]
        The array of matrices of the matrices to be inverted. The first is
        related to the B-spline bases and the rest to the penalty terms along
        each dimension.
    y : np.ndarray
        The response variable sample.

    References
    ----------
    - Golub, G. H., Heath, M., & Wahba, G. (1979). Generalized cross-validation
      as a method for choosing a good ridge parameter. Technometrics, 21(2),
      215-223.

    Returns
    -------
    float
        The GCV value.
    """
    Q = quadratic_term(sp=sp, Q_matrices=Q_matrices)
    # The fitted y
    y_hat = explicit_y_hat(Q=Q, B_weighted=B_weighted, y=y)
    # Return the GCV value, which is RSS * n / (n - tr(H))**2, where RSS is the
    # residual sum of squares, n is the product of the dimensions of y and H is
    # the hat matrix of the unconstrained problem
    return (np.linalg.norm((y - y_hat)) ** 2 * np.prod(y.shape)) / (
        np.prod(y.shape) - np.trace(np.linalg.solve(Q, Q_matrices[0]))
    ) ** 2
