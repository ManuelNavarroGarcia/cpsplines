import mosek.fusion
import numpy as np
from functools import reduce

from template.utils.fast_kron import fast_kronecker_product


def kron_tens_prod_mosek(mat_list, var):
    init_shape = var.getShape()
    out = mosek.fusion.Expr.mul(
        mat_list[0], var.reshape([init_shape[0], np.prod(init_shape[1:])])
    )
    if mat_list[1:]:
        right_mul = reduce(fast_kronecker_product, mat_list[1:]).T
        out = mosek.fusion.Expr.mul(out, right_mul)
    final_shape = [mat.shape[0] for mat in mat_list]
    return mosek.fusion.Expr.reshape(out, final_shape)
