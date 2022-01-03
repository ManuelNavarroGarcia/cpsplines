import itertools
from typing import Dict, Iterable, List, Tuple, Union

import mosek.fusion
import numpy as np
from cpsplines.mosek_functions.utils_mosek import matrix_by_tensor_product_mosek
from cpsplines.psplines.bspline_basis import BsplineBasis
from scipy.sparse import diags


class PDFConstraints:
    def __init__(
        self,
        bspline: Iterable[BsplineBasis],
    ):
        self.bspline = bspline

    def integrate_to_one(
        self,
        var_dict: Dict[str, mosek.fusion.LinearVariable],
        model: mosek.fusion.Model,
    ):
        coef_list = []
        for bsp in self.bspline:
            vander = np.vander(bsp.knots, N=bsp.deg + 2, increasing=True)[
                bsp.deg : -bsp.deg, 1:
            ]
            coef = 1 / np.linspace(1, bsp.deg + 1, bsp.deg + 1)
            diff_mat = np.einsum("ij,j->ij", np.diff(vander, axis=0), coef)
            banded = diags(
                np.dot(diff_mat[0], bsp.get_matrices_S()[0]),
                range(bsp.deg + 1),
                shape=(bsp.n_int, bsp.n_int + bsp.deg),
            ).toarray()
            coef_list.append(banded)

        sum_coef = mosek.fusion.Expr.sum(
            matrix_by_tensor_product_mosek(
                matrices=coef_list, mosek_var=var_dict["theta"]
            )
        )
        cons = model.constraint(sum_coef, mosek.fusion.Domain.equalsTo(1.0))
        return cons
