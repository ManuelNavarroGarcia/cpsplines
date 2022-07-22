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
    sp: Iterable[Union[int, float]],
    obj_matrices: Dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
    family: statsmodels.genmod.families.family,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Computes the quadratic terms involved in the objective function, i.e., the
    bases related term and the penalty term.

    Parameters
    ----------
    sp : Iterable[Union[int, float]]
        The smoothing paramater vector.
    obj_matrices : Dict[str, Union[np.ndarray, Iterable[np.ndarray]]]
        A dictionary containing the necessary arrays (the basis matrices, the
        penalty matrices and the response variable sample) used to compute the
        quadratic terms in the objective function.
    family : statsmodels.genmod.families.family, optional
        The specific exponential family distribution where the response variable
        belongs to, by default "gaussian".

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the bases related term and the penalty term (in this
        order).
    """

    mu = family.starting_mu(obj_matrices["y"])
    W = family.weights(mu)
    bases_term = weighted_double_kronecker(matrices=obj_matrices["B_w"], W=W)
    penalty_list = penalization_term(matrices=obj_matrices["D_mul"])
    penalty_term = np.add.reduce([np.multiply(s, P) for P, s in zip(penalty_list, sp)])
    return (bases_term, penalty_term)


def GCV(
    sp: Iterable[Union[int, float]],
    B_weighted: Iterable[np.ndarray],
    Q_matrices: Iterable[np.ndarray],
    y: np.ndarray,
    family: str = "gaussian",
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
    y_hat = explicit_y_hat(Q=Q, B_weighted=B_weighted, y=y, family=family)
    # Return the GCV value, which is RSS * n / (n - tr(H))**2, where RSS is the
    # residual sum of squares, n is the product of the dimensions of y and H is
    # the hat matrix of the unconstrained problem
    return (np.linalg.norm((y - y_hat)) ** 2 * np.prod(y.shape)) / (
        np.prod(y.shape) - np.trace(np.linalg.solve(Q, Q_matrices[0]))
    ) ** 2
