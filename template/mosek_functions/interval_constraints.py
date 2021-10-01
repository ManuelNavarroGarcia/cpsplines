import itertools
import mosek.fusion
import numpy as np
from scipy.special import comb, factorial
from typing import Dict, List, Iterable, Tuple, Union

from template.psplines.bspline_basis import BsplineBasis
from template.mosek_functions.utils_mosek import kron_tens_prod_mosek


class IntConstraints:
    def __init__(
        self,
        bsp_list: Iterable[BsplineBasis],
        var_name: int,
        derivative: int,
        constraints: Dict[str, Union[int, float]],
    ):
        self.bsp_list = bsp_list
        self.var_name = var_name
        self.derivative = derivative
        self.constraints = constraints
        self.deg_w = self.bsp_list[self.var_name].deg - self.derivative
        self.matricesW = self._get_matricesW()
        self.zeroH = self._get_zero_H()
        self.nonzeroH = self._get_non_zero_H()

    def _get_matricesW(self) -> List[np.ndarray]:
        diff_coef = np.array(
            [
                factorial(self.derivative)
                * comb(self.derivative + i - 1, self.derivative)
                for i in range(1, self.deg_w + 2)
            ],
        )
        W = []
        bsp = self.bsp_list[self.var_name]
        for i in range(bsp.matrixB.shape[1] - bsp.deg):
            W_i = np.zeros(shape=(self.deg_w + 1, self.deg_w + 1))
            for q in range(self.deg_w + 1):
                for m in range(q + 1):
                    for r in range(m, self.deg_w + 1 + m - q):
                        W_i[q, r] += (
                            comb(r, m)
                            * comb(self.deg_w - r, q - m)
                            * (bsp.knots[bsp.deg : -bsp.deg][i]) ** (r - m)
                            * (bsp.knots[bsp.deg : -bsp.deg][i + 1]) ** (m)
                        )
            W.append(diff_coef * W_i)
        return W

    def _get_zero_H(self) -> List[mosek.fusion.Matrix]:
        rows, _ = np.indices((self.deg_w + 1, self.deg_w + 1))
        k_diag = np.linspace(self.deg_w - 1, 1 - self.deg_w, self.deg_w)
        H = []
        for i in range(self.deg_w):
            h_row = np.diag(rows, k=int(k_diag[i])).tolist()
            h_col = h_row[::-1]
            H.append(
                mosek.fusion.Matrix.sparse(
                    self.deg_w + 1, self.deg_w + 1, h_row, h_col, [1] * len(h_row)
                )
            )
        return H

    def _get_non_zero_H(self) -> List[mosek.fusion.Matrix]:
        rows, _ = np.indices((self.deg_w + 1, self.deg_w + 1))
        k_diag = np.linspace(self.deg_w, -self.deg_w, self.deg_w + 1)
        H = []
        for i in range(self.deg_w + 1):
            h_row = np.diag(rows, k=int(k_diag[i])).tolist()
            h_col = h_row[::-1]
            H.append(
                mosek.fusion.Matrix.sparse(
                    self.deg_w + 1, self.deg_w + 1, h_row, h_col, [1] * len(h_row)
                )
            )
        return H

    def _create_PSD_var(self, model: mosek.fusion.Model) -> mosek.fusion.PSDVariable:
        shapes_B = [bsp.matrixB.shape[1] - bsp.deg for bsp in self.bsp_list]
        return model.variable(
            mosek.fusion.Domain.inPSDCone(
                self.deg_w + 1, len(self.constraints.keys()) * np.prod(shapes_B)
            )
        )

    def interval_cons(
        self,
        var_dict: Dict[str, mosek.fusion.LinearVariable],
        model: mosek.fusion.Model,
        S_dict: Dict[int, Iterable[np.ndarray]],
    ) -> Tuple[mosek.fusion.LinearConstraint]:
        X = self._create_PSD_var(model=model)
        ind_term = self.matricesW[0][:, 0]
        for j, bsp in enumerate(self.bsp_list):
            if self.var_name == j:
                S_dict[self.var_name] = [
                    self.matricesW[i] @ np.delete(s, range(self.derivative), axis=0)
                    for i, s in enumerate(S_dict[self.var_name])
                ]
            else:
                value_at_knots = np.expand_dims(
                    bsp.matrixB[
                        bsp.int_back, bsp.int_back : bsp.int_back + bsp.deg + 1
                    ],
                    axis=0,
                )
                S_dict[j] = [value_at_knots for _ in range(len(S_dict[j]))]
        list_cons = []
        if len(S_dict.keys()) == 1:
            num_inter = 1
        else:
            num_inter = np.prod(
                np.array(
                    [
                        bsp.matrixB.shape[1] - bsp.deg
                        for i, bsp in enumerate(self.bsp_list)
                        if i != self.var_name
                    ]
                )
            )
        num_cons = len(self.constraints.keys())
        for w in range(
            self.bsp_list[self.var_name].matrixB.shape[1]
            - self.bsp_list[self.var_name].deg
        ):
            a = [v for i, v in enumerate(S_dict.values()) if i != self.var_name]
            a_idx = [
                range(len(v))
                for i, v in enumerate(S_dict.values())
                if i != self.var_name
            ]
            a.insert(self.var_name, [S_dict[self.var_name][w]])
            a_idx.insert(self.var_name, range(w, w + 1))
            iter_a = list(itertools.product(*a))
            iter_idx = list(itertools.product(*a_idx))
            for j, (id, mat) in enumerate(zip(iter_idx, iter_a)):
                last_id = [id[i] + bsp.deg + 1 for i, bsp in enumerate(self.bsp_list)]
                coef_theta = var_dict["theta"].slice(
                    np.array(id, dtype=np.int32), np.array(last_id, dtype=np.int32)
                )
                poly_coef = mosek.fusion.Expr.flatten(
                    kron_tens_prod_mosek(mat_list=mat, var=coef_theta)
                )
                for k, key in enumerate(self.constraints.keys()):
                    actual_index = k + num_cons * (j + w * num_inter)
                    slice_X = X.slice(
                        [actual_index, 0, 0],
                        [
                            actual_index + 1,
                            self.deg_w + 1,
                            self.deg_w + 1,
                        ],
                    ).reshape([self.deg_w + 1, self.deg_w + 1])
                    for i in range(self.deg_w):
                        # Creates the homogeneous equations
                        list_cons.append(
                            model.constraint(
                                mosek.fusion.Expr.dot(self.zeroH[i], slice_X),
                                mosek.fusion.Domain.equalsTo(0.0),
                            )
                        )
                    if key == "+":
                        for i in range(self.deg_w + 1):
                            # Creates the nonhomogeneous equations
                            list_cons.append(
                                model.constraint(
                                    mosek.fusion.Expr.sub(
                                        poly_coef.slice(i, i + 1),
                                        mosek.fusion.Expr.dot(
                                            self.nonzeroH[i], slice_X
                                        ),
                                    ),
                                    mosek.fusion.Domain.equalsTo(
                                        ind_term[i] * self.constraints[key]
                                    ),
                                )
                            )
                    elif key == "-":
                        for i in range(self.deg_w + 1):
                            # Creates the nonhomogeneous equations
                            list_cons.append(
                                model.constraint(
                                    mosek.fusion.Expr.add(
                                        poly_coef.slice(i, i + 1),
                                        mosek.fusion.Expr.dot(
                                            self.nonzeroH[i], slice_X
                                        ),
                                    ),
                                    mosek.fusion.Domain.equalsTo(
                                        ind_term[i] * self.constraints[key]
                                    ),
                                )
                            )
                    else:
                        raise TypeError(
                            "Only interval constraints related to the sign are allowed"
                        )
        return tuple(list_cons)
