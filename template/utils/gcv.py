import numpy as np
from functools import reduce

from template.utils.fast_kron import (
    kron_tens_prod,
    fast_kronecker_product,
    penalization_term,
)


def gcv_mat(B_mul, D_mul) -> tuple:
    qua_term = reduce(fast_kronecker_product, B_mul[::-1])
    pen = penalization_term(mat_list=D_mul[::-1])
    return tuple([qua_term] + pen)


def quadratic_matrix(sp_list, qua_term):
    complete_pen_term = np.zeros(qua_term[0].shape)
    for i in range(len(sp_list)):
        complete_pen_term += np.multiply(sp_list[::-1][i], qua_term[i + 1])
    return qua_term[0] + complete_pen_term


def explicit_y_hat(Q, B_pred, y_sam):
    y_sam_prod = kron_tens_prod([mat.T for mat in B_pred], y_sam).flatten("F")
    theta = np.reshape(
        np.linalg.solve(Q, y_sam_prod),
        tuple([mat.shape[1] for mat in B_pred]),
        order="F",
    )
    return kron_tens_prod([mat for mat in B_pred], theta)


def GCV(
    sp_list: list,
    B_pred: list,
    qua_term: list,
    y_sam: np.ndarray,
) -> float:
    Q = quadratic_matrix(sp_list=sp_list, qua_term=qua_term)
    y_hat = explicit_y_hat(Q=Q, B_pred=B_pred, y_sam=y_sam)
    len_prod = np.prod(y_sam.shape)
    return (np.linalg.norm((y_sam - y_hat)) ** 2 * len_prod) / (
        len_prod - np.trace(np.linalg.solve(Q, qua_term[0]))
    ) ** 2
