import mosek.fusion
import numpy as np
from typing import Dict, Iterable, Tuple, Union

from template.psplines.bspline import Bspline
from template.mosek_functions.utils_mosek import kron_tens_prod_mosek


class PointConstraints:
    def __init__(
        self,
        pts: Iterable[np.ndarray],
        value: Iterable[Union[int, float]],
        derivative: Iterable[int],
        bsp_list: Iterable[Bspline],
        tolerance: Union[int, float],
    ):
        self.pts = pts
        self.value = value
        self.derivative = derivative
        self.bsp_list = bsp_list
        self.tolerance = tolerance

    def point_cons(
        self,
        var_dict: Dict[str, mosek.fusion.LinearVariable],
        model: mosek.fusion.Model,
    ) -> Tuple[mosek.fusion.LinearConstraint]:
        bsp_pt = {}
        for i, bsp in enumerate(self.bsp_list):
            bsp_pt[i] = bsp.bspline_basis.derivative(nu=self.derivative[i])(self.pts[i])

        list_cons = []
        for i, v in enumerate(self.value):
            mat_list = []
            for j in range(len(self.bsp_list)):
                mat_list.append(np.expand_dims(bsp_pt[j][i, :], axis=1).T)
            coef = kron_tens_prod_mosek(mat_list=mat_list, var=var_dict["theta"])
            list_cons.append(
                model.constraint(
                    coef,
                    mosek.fusion.Domain.greaterThan(v - self.tolerance),
                )
            )
            list_cons.append(
                model.constraint(
                    coef,
                    mosek.fusion.Domain.lessThan(v + self.tolerance),
                )
            )
        return tuple(list_cons)
