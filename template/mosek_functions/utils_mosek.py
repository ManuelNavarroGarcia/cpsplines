import mosek.fusion
import numpy as np
from functools import reduce
from typing import Iterable

from template.utils.fast_kron import fast_kronecker_product


def kron_tens_prod_mosek(
    matrices: Iterable[np.ndarray], mosek_var: mosek.fusion.Variable
) -> mosek.fusion.Expr:

    """
    Performs the product of a multidimensional MOSEK decision variable with
    shape k_1 x ··· x k_N by a set of matrices A_i with shapes s_i x k_i along
    each mode. The decision variable is converted to a matrix along the 0-th
    mode, denoted by T_0, and the product A_0 @ T_0 (A_1 kron ··· kron A_N).T is
    computed. Finally, the resulting multidimensional array is reshaped so the
    final shape is (s_0, s_1, ..., s_N).

    Returns
    -------
    mosek.fusion.Expr with shape (s_0, s_1, ..., s_N)
        The output of the product.
    """

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
