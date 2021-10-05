import itertools
import mosek.fusion
import numpy as np
from scipy.special import comb, factorial
from typing import Dict, List, Iterable, Tuple, Union

from template.psplines.bspline_basis import BsplineBasis
from template.mosek_functions.utils_mosek import kron_tens_prod_mosek


class IntConstraints:

    """
    Define the set of constraints that ensures the non-negativity (or
    non-positivity) for the derivative order `derivative`along the respective
    axis of the variable `var_name`. For the unidimensional case, the
    requirements are completely fulfilled for every point in the range where the
    B-spline basis is defined. For the multivariate case, the constraints are
    imposed at the curves that pass through the inner knots of the B-spline
    basis. For all cases, the constraints are imposed following the Proposition
    1 in Bertsimas and Popescu (2002).

    Parameters
    ----------
    bspline : Iterable[BsplineBasis]
        An iterable containing the B-spline bases objects used to approximate
        the function to estimate.
    var_name : int
        The name of the variable along the constraints are imposed.
    derivative : int
        The derivative order of the function that needs to be constrained.
    constraints : Dict[str, Union[int, float]]
        The constraints that are imposed. The keys of the dictionary are either
        "+" (the function (or some derivative) needs to be above some threshold)
        or "-" (the function (or some derivative) needs to be below some
        threshold), and the values are the thresholds.

    Attributes
    ----------
    deg_w : int
        The shape of the matrices W.
    matricesW : List[np.ndarray] of shape (deg_w, deg_w)
        The matrices W used to impose the constraints by virtue of Proposition 1
        in Bertsimas and Popescu (2002).
    matricesH : mosek.fusion.Matrix
        Auxiliary matrices to extract the correct coefficients from the
        semidefinite matrix variable in the left-hand side from the equations of
        Proposition 1 in Bertsimas and Popescu (2002).

    References
    ----------
    - Bertsimas, D., & Popescu, I. (2002). On the relation between option
      and stock prices: a convex optimization approach. Operations Research,
      50(2), 358-374.
    """

    def __init__(
        self,
        bspline: Iterable[BsplineBasis],
        var_name: int,
        derivative: int,
        constraints: Dict[str, Union[int, float]],
    ):
        self.bspline = bspline
        self.var_name = var_name
        self.derivative = derivative
        self.constraints = constraints

    def _get_matrices_W(self) -> List[np.ndarray]:

        """
        Generates matrices W containing the weights that have to be multiplied
        by the coefficients of the polynomials from the B-spline basis to employ
        Proposition 1 in Bertsimas and Popescu (2002). For each interval of the
        B-spline basis, one matrix W is defined.

        Returns
        -------
        List[np.ndarray]
            The list of matrices W.
        """

        # Select the B-spline basis along the corresponding axis
        bsp = self.bspline[self.var_name]
        # Define deg_w, which coincides with the order of the matrices W
        self.deg_w = bsp.deg - self.derivative
        # Deriving a polynomial gives rise to a coefficient vector whose length
        # reduces by one the degree of the polynomial (only non-zero
        # coefficients are considered). Recursively, a polynomial with degree d
        # once derivated n times will leave a d-n vector. This vector is given
        # by the n column and the first (n-d) rows of the symmetric Pascal
        # matrix of order d.
        pascal_coef = np.array(
            [
                factorial(self.derivative) * comb(self.derivative + i, self.derivative)
                for i in range(self.deg_w + 1)
            ],
            dtype=np.int32,
        )
        W = []
        # Generate the matrices W, containing the weights from the Proposition 1
        # of Bertsimas and Popescu (2002)
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
            # We multiply this vector coefficient by each matrix W instead of
            # multiplying it by matrices S, which contain the non-zero
            # coefficients of the polynomials, since at the end of the day we
            # will take the product WS on the axis where constraints are
            # enforced
            W.append(pascal_coef * W_i)
        return W

    def _get_matrices_H(self) -> List[List[mosek.fusion.Matrix]]:

        """
        Generates matrices H used to extract the right coefficients from the
        semidefinite matrices variables to fulfill the equations of Proposition
        1 in Bertsimas and Popescu (2002). This is done by using the Frobenius
        inner product and, as it is stated in the proposition, matrices have
        order `deg_w`+ 1. For each interval of the B-spline basis, one matrix H
        is defined.

        Returns
        -------
        List[List[mosek.fusion.Matrix]]
            A list containing two list of matrices: the first one corresponds to
            the matrices used on the homogeneous equations, while the second is
            used on the non-homogeneous equations. For the last, the first
            matrix corresponds to the independent term equation, the second to
            the linear term equation, and so on.
        """
        diag_zero = np.linspace(
            1 - self.deg_w, self.deg_w - 1, self.deg_w, dtype=np.int8
        )
        diag_nonzero = np.linspace(
            -self.deg_w, self.deg_w, self.deg_w + 1, dtype=np.int8
        )
        H = []
        for diag in [diag_zero, diag_nonzero]:
            H_by_diag = []
            for k in diag:
                H_by_diag.append(
                    mosek.fusion.Matrix.sparse(
                        np.rot90(np.eye(self.deg_w + 1, k=k, dtype=np.int32))
                    )
                )
            H.append(H_by_diag)
        return H

    def _create_PSD_var(self, model: mosek.fusion.Model) -> mosek.fusion.PSDVariable:
        shapes_B = [bsp.matrixB.shape[1] - bsp.deg for bsp in self.bspline]
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
        self.matricesW = self._get_matrices_W()
        self.matricesH = self._get_matrices_H()

        X = self._create_PSD_var(model=model)
        ind_term = self.matricesW[0][:, 0]
        for j, bsp in enumerate(self.bspline):
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
                        for i, bsp in enumerate(self.bspline)
                        if i != self.var_name
                    ]
                )
            )
        num_cons = len(self.constraints.keys())
        for w in range(
            self.bspline[self.var_name].matrixB.shape[1]
            - self.bspline[self.var_name].deg
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
                last_id = [id[i] + bsp.deg + 1 for i, bsp in enumerate(self.bspline)]
                coef_theta = var_dict["theta"].slice(
                    np.array(id, dtype=np.int32), np.array(last_id, dtype=np.int32)
                )
                poly_coef = mosek.fusion.Expr.flatten(
                    kron_tens_prod_mosek(matrices=mat, mosek_var=coef_theta)
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
                                mosek.fusion.Expr.dot(self.matricesH[0][i], slice_X),
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
                                            self.matricesH[1][i], slice_X
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
                                            self.matricesH[1][i], slice_X
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
