import numpy as np
from functools import reduce
from typing import Iterable, Tuple, Union

from template.utils.fast_kron import (
    kron_tens_prod,
    fast_kronecker_product,
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


def quadratic_matrix(
    sp: Iterable[Union[int, float]], qua_term: Iterable[np.ndarray]
) -> np.ndarray:
    penalty_term = np.zeros(qua_term[0].shape)
    reversed_sp = sp[::-1]
    for i, s in enumerate(reversed_sp):
        penalty_term += np.multiply(s, qua_term[i + 1])
    return qua_term[0] + penalty_term


def explicit_y_hat(Q: np.ndarray, B_weighted: Iterable[np.ndarray], y: np.ndarray):
    y_contribution = kron_tens_prod([B.T for B in B_weighted], y).flatten("F")
    theta = np.reshape(
        np.linalg.solve(Q, y_contribution),
        tuple([mat.shape[1] for mat in B_weighted]),
        order="F",
    )
    return kron_tens_prod([mat for mat in B_weighted], theta)


def GCV(
    sp: Iterable[Union[int, float]],
    B_weighted: Iterable[np.ndarray],
    qua_term: Iterable[np.ndarray],
    y: np.ndarray,
) -> float:
    Q = quadratic_matrix(sp=sp, qua_term=qua_term)
    y_hat = explicit_y_hat(Q=Q, B_weighted=B_weighted, y=y)
    return (np.linalg.norm((y - y_hat)) ** 2 * np.prod(y.shape)) / (
        np.prod(y.shape) - np.trace(np.linalg.solve(Q, qua_term[0]))
    ) ** 2
