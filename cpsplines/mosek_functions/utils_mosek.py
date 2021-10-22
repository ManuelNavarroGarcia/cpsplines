from functools import reduce
from typing import Iterable

import mosek.fusion
import numpy as np
from cpsplines.utils.fast_kron import fast_kronecker_product


def matrix_by_tensor_product_mosek(
    matrices: Iterable[np.ndarray], mosek_var: mosek.fusion.Variable
) -> mosek.fusion.Expr:

    """
    Performs the product of a multidimensional MOSEK decision variable with
    shape k_1 x ··· x k_N by a set of matrices A_i with shapes s_i x k_i along
    each mode. The decision variable is converted to a matrix along the 0-th
    mode, denoted by T_0, and the product A_0 @ T_0 (A_1 kron ··· kron A_N).T is
    computed. Finally, the resulting multidimensional array is reshaped so the
    final shape is (s_0, s_1, ..., s_N).

    Parameters
    ----------
    matrices : Iterable[np.ndarray]
        The matrices A_i with shapes s_i x k_i to be multiplied by the
        multidimensional decision variable.
    mosek_var : mosek.fusion.Variable
        The k_1 x ··· x k_N MOSEK decision variable.

    Returns
    -------
    mosek.fusion.Expr with shape (s_0, s_1, ..., s_N)
        The output of the product.

    Raises
    ------
    ValueError
        If any matrix in `matrices` is one-dimensional.

    """

    if any(x.ndim != 2 for x in matrices):
        raise ValueError("Matrices must be two-dimensional. No vectors allowed.")

    init_shape = mosek_var.getShape()
    # The decision variable is unfolding along the 0-th mode and the product
    # A_0 @ T_0 is performed
    out = mosek.fusion.Expr.mul(
        matrices[0], mosek_var.reshape([init_shape[0], np.prod(init_shape[1:])])
    )
    # If there is more than one input matrix, the previous out is multiplied by
    # (A_1 kron ··· kron A_N).T, where kron denotes the Kronecker product
    if matrices[1:]:
        right_mul = reduce(fast_kronecker_product, matrices[1:]).T
        out = mosek.fusion.Expr.mul(out, right_mul)
    # The output is folded along 0-th mode
    final_shape = [mat.shape[0] for mat in matrices]
    return mosek.fusion.Expr.reshape(out, final_shape)
