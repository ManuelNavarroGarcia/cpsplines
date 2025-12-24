import itertools
from collections.abc import Iterable
from functools import reduce

import mosek.fusion
import numpy as np
from scipy.special import comb, factorial

from src.cpsplines.psplines.bspline_basis import BsplineBasis


class IntConstraints:
    """
    Define the set of constraints that ensures the non-negativity (or
    non-positivity) for the derivative order `derivative` along the respective
    axis of the variable `var_name`. For the unidimensional case, the
    requirements are completely fulfilled for every point in the range where the
    B-spline basis is defined. For the multivariate case, the constraints are
    imposed at the curves that pass through the inner knots of the B-spline
    basis. For all cases, the constraints are imposed following the Proposition
    1 in Bertsimas and Popescu (2002).

    Parameters
    ----------
    bspline : Dict[str, BsplineBasis]
        A dictionary containing the B-spline bases objects used to approximate
        the function to estimate as values, and as key the feature names.
    var_name : str
        The name of the variable along the constraints are imposed.
    derivative : int
        The derivative order of the function that needs to be constrained. Must
        be a non-negative integer.
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
        Auxiliary matrices to extract the correct coefficients from the positive
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
        bspline: dict[str, BsplineBasis],
        var_name: str,
        derivative: int,
        constraints: dict[str, int | float],
    ):
        self.bspline = bspline
        self.var_name = var_name
        self.derivative = derivative
        self.constraints = constraints

    def _get_matrices_W(self) -> list[np.ndarray]:
        """
        Generates matrices W containing the weights that have to be multiplied
        by the coefficients of the polynomials from the B-spline basis to employ
        Proposition 1 in Bertsimas and Popescu (2002). For each interval of the
        B-spline basis, one matrix W is defined.

        Returns
        -------
        List[np.ndarray]
            The list of matrices W.

        Raises
        ------
        ValueError
            If `derivative` is not a non-negative integer.
        ValueError
            If `deg_w` is not a non-negative integer.
        """

        if self.derivative < 0:
            raise ValueError("The derivative order must be a non-negative integer.")

        # Select the B-spline basis along the corresponding axis
        bsp = self.bspline[self.var_name]
        # Define deg_w, which coincides with the order of the matrices W
        if bsp.deg - self.derivative < 0:
            raise ValueError("The derivative order must be lower than the B-spline basis degree.")
        self.deg_w = bsp.deg - self.derivative
        # Deriving a polynomial gives rise to a coefficient vector whose length
        # reduces by one the degree of the polynomial (only non-zero
        # coefficients are considered). Recursively, a polynomial with degree d
        # once derivated n times will leave a d-n vector. This vector is given
        # by the n column and the first (n-d) rows of the symmetric Pascal
        # matrix of order d.
        pascal_coef = np.array(
            [factorial(self.derivative) * comb(self.derivative + i, self.derivative) for i in range(self.deg_w + 1)],
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

    def _get_matrices_H(self) -> list[mosek.fusion.SparseMatrix]:
        """
        Generates matrices H used to extract the right coefficients from the
        positive semidefinite matrix variables to fulfill the equations of
        Proposition 1 in Bertsimas and Popescu (2002). This is done by using the
        Frobenius inner product and, as it is stated in the proposition,
        matrices have order `deg_w`+ 1. For each interval of the B-spline basis,
        one matrix H is defined.

        Returns
        -------
        List[mosek.fusion.SparseMatrix]
            A list containing 2 * `deg_w` + 1 mosek.fusion.SparseMatrix: the
            first `deg_w` corresponds to the matrices used on the homogeneous
            equations, while the rest is used on the non-homogeneous equations.
            For this last set, the first matrix corresponds to the independent
            term equation, the second to the linear term equation, and so on.
        """

        # The elements from X in the homogeneous equations are located on its
        # even antidiagonals, while the ones from the non-homogeneous equations
        # are situated on the odd antidiagonals
        diag_zero = np.linspace(1 - self.deg_w, self.deg_w - 1, self.deg_w, dtype=np.int8)
        diag_nonzero = np.linspace(-self.deg_w, self.deg_w, self.deg_w + 1, dtype=np.int8)
        H = []
        for diag in [diag_zero, diag_nonzero]:
            for k in diag:
                # Create an identity matrix along the corresponding diagonal and
                # the rotate it 90 degrees to get the antidiagonal. Then convert
                # it to a MOSEK sparse matrix
                H.append(mosek.fusion.Matrix.sparse(np.rot90(np.eye(self.deg_w + 1, k=k, dtype=np.int32))))
        return H

    def _create_PSD_var(self, model: mosek.fusion.Model) -> mosek.fusion.PSDVariable:
        """
        Create the positive semidefinite matrix variables used to formulate the
        constraints. They have shape (`deg_w` + 1, `deg_w` + 1) and one by
        interval where constraints are applied is needed.

        Parameters
        ----------
        model : mosek.fusion.Model
            The MOSEK model of the problem.

        Returns
        -------
        mosek.fusion.PSDVariable
            A three-dimensional variable with shape (`deg_w` + 1, `deg_w` + 1,
            n * m), where n is the number of interval where the constraint need
            to be imposed, m is the number of sign constraints related to this
            derivative order and variable, and each of the matrices (`deg_w` +
            1, `deg_w` + 1) are required to be positive semidefinite.
        """

        # The number of intervals where the constraint need to be enforced can
        # be computed as the product of interior intervals
        n = [bsp.matrixB.shape[1] - bsp.deg for bsp in self.bspline.values()]
        return model.variable(mosek.fusion.Domain.inPSDCone(self.deg_w + 1, len(self.constraints.keys()) * np.prod(n)))

    def interval_cons(
        self,
        var_dict: dict[str, mosek.fusion.LinearVariable],
        model: mosek.fusion.Model,
        matrices_S: dict[int, Iterable[np.ndarray]],
    ) -> tuple[mosek.fusion.LinearConstraint]:
        """
        Defines the non-negative related constraints over a finite interval. For
        each interval and each sign constraint, 2 * `deg_w` - 1 equations are
        defined.

        Parameters
        ----------
        var_dict :  Dict[str, mosek.fusion.LinearVariable]
            The dictionary that contains the decision variables used to define
            the objective function of the problem.
        model : mosek.fusion.Model
            The MOSEK model of the problem.
        matrices_S : Dict[int, Iterable[np.ndarray]]
            Contains as keys the indices of the variable, and as values the
            matrices S.

        Returns
        -------
        Tuple[mosek.fusion.LinearConstraint]
            The set of constraints along the variable `var_name` and derivative
            order `derivative`.

        Raises
        ------
        ValueError
            If the constraints are not either "+" or "-".
        """

        if len(set(self.constraints.keys()) - set(["+", "-"])) > 0:
            raise ValueError("The constraint sign must be either `+` or `-`.")

        # Generate the matrices H and W, and the decision variables X
        self.matricesW = self._get_matrices_W()
        self.matricesH = self._get_matrices_H()
        var_dict |= {"X": self._create_PSD_var(model=model)}
        # Extract the weights related to the independent term, which is used
        # when some derivative of the curve is constrained to be greater or
        # lower than a certain threshold. `deg_w` are included since the first
        # `deg_w` of the proposition are homogenenous
        ind_term = np.concatenate((np.zeros(self.deg_w), self.matricesW[0][:, 0]))
        # For every axis, get the contribution to the estimated function at the
        # knots
        for name, bsp in self.bspline.items():
            # The contribution along the direction where the constraints are
            # imposed is W (that already include the vector coefficient arised
            # when differentiate the polynomial) times S once the first
            # `derivative` rows are deleted
            if self.var_name == name:
                matrices_S[self.var_name] = [
                    np.r_[
                        np.zeros((self.derivative, s.shape[1])),
                        w @ np.delete(s, range(self.derivative), axis=0),
                    ]
                    for w, s in zip(self.matricesW, matrices_S[self.var_name])
                ]
            # Since the knot sequence is evenly spaced, the value of the
            # B-splines is periodic an it is always the same, so we pick up the
            # value of all the B-splines at the first knot
            else:
                value_at_knots = np.vander(bsp.knots[bsp.deg : -bsp.deg], N=bsp.deg + 1, increasing=True)
                matrices_S[name] = [value_at_knots[i, :] @ s for i, s in enumerate(matrices_S[name])]
        # For every interval on the `var_name` axis, count how many interval
        # constraints of the same sign and fixed derivative need to be
        # considered
        num_by_interval = (
            np.prod(
                np.array(
                    [bsp.matrixB.shape[1] - bsp.deg for name, bsp in self.bspline.items() if name != self.var_name]
                )
            )
            if len(matrices_S.keys()) > 1
            else 1
        )
        # For every interval on the `var_name` axis, count how many interval
        # constraints types by sign there are
        num_by_sign = len(self.constraints.keys())
        # The list of interval constraints
        list_cons = []
        # Retain the left-hand side terms of the equations (<H, X>)
        trace_list = []
        # Retain the right-hand side terms of the equations (S @ theta)
        coef_list = []
        # Retain the independent coefficients from the right-hand side terms of
        # the equations
        ind_term_list = []
        # Loop over every interval on the `var_name` axis
        for w in range(self.bspline[self.var_name].matrixB.shape[1] - self.bspline[self.var_name].deg):
            # Create a list containing the lists with the contribution to the
            # estimated function at the knots
            a = [s for name, s in matrices_S.items() if name != self.var_name]
            # Create a list containing ranges of the same length as previous list
            a_idx = [range(len(s)) for name, s in matrices_S.items() if name != self.var_name]
            # Insert at the position `var_name` the correct value/index of S
            # along the `var_name` direction
            a.insert(list(self.bspline).index(self.var_name), [matrices_S[self.var_name][w]])
            a_idx.insert(list(self.bspline).index(self.var_name), range(w, w + 1))
            # Generate all the combinations possible from previous lists
            iter_a = list(itertools.product(*a))
            iter_idx = list(itertools.product(*a_idx))
            # Loop over the size of all the combinations (same length as
            # positive semidefinite variables) and their corresponding values of
            # the contribution
            for j, (id, mat) in enumerate(zip(iter_idx, iter_a)):
                last_id = [id[i] + bsp.deg + 1 for i, bsp in enumerate(self.bspline.values())]
                # Slice the multidimensional coefficient variable on the
                # interval on the values the corresponding B-spline is non-zero
                coef_theta = var_dict["theta"].slice(np.array(id, dtype=np.int32), np.array(last_id, dtype=np.int32))
                # Multiply the sliced variable on each face by the correct
                # contribution
                poly_coef = mosek.fusion.Expr.mul(reduce(np.kron, mat), mosek.fusion.Expr.flatten(coef_theta)).slice(
                    self.derivative, self.bspline[self.var_name].deg + 1
                )
                # Loop over the different sign constraints
                for k, key in enumerate(self.constraints.keys()):
                    sign_cons = 1 if key == "+" else -1
                    # Get current index
                    actual_index = k + num_by_sign * (j + w * num_by_interval)
                    # Get current positive semidefinite variable
                    slice_X = (
                        var_dict["X"]
                        .slice(
                            [actual_index, 0, 0],
                            [
                                actual_index + 1,
                                self.deg_w + 1,
                                self.deg_w + 1,
                            ],
                        )
                        .reshape([self.deg_w + 1, self.deg_w + 1])
                    )
                    # Append the corresponding terms for this interval
                    trace_list.append([mosek.fusion.Expr.dot(H, slice_X) for H in self.matricesH])
                    coef_list.append(
                        mosek.fusion.Expr.vstack(
                            mosek.fusion.Expr.constTerm(self.deg_w, 0),
                            mosek.fusion.Expr.mul(
                                sign_cons,
                                poly_coef,
                            ),
                        )
                    )
                    ind_term_list.append(sign_cons * ind_term * self.constraints[key])

        list_cons.append(
            model.constraint(
                mosek.fusion.Expr.sub(
                    mosek.fusion.Expr.vstack(coef_list),
                    mosek.fusion.Expr.vstack(list(itertools.chain.from_iterable(trace_list))),
                ),
                mosek.fusion.Domain.equalsTo(np.concatenate(ind_term_list)),
            )
        )
        return tuple(list_cons)
