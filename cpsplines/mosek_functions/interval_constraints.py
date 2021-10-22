import itertools
from typing import Dict, Iterable, List, Tuple, Union

import mosek.fusion
import numpy as np
from cpsplines.mosek_functions.utils_mosek import matrix_by_tensor_product_mosek
from cpsplines.psplines.bspline_basis import BsplineBasis
from scipy.special import comb, factorial


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
        The name of the variable along the constraints are imposed. Must be a
        non-negative integer.
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

        Raises
        ------
        ValueError
            If `var_name` is not a non-negative integer.
        ValueError
            If `derivative` is not a non-negative integer.
        ValueError
            If `deg_w` is not a non-negative integer.
        """

        if self.var_name < 0:
            raise ValueError("The variable name must be a non-negative integer.")
        if self.derivative < 0:
            raise ValueError("The derivative order must be a non-negative integer.")

        # Select the B-spline basis along the corresponding axis
        bsp = self.bspline[self.var_name]
        # Define deg_w, which coincides with the order of the matrices W
        if bsp.deg - self.derivative < 0:
            raise ValueError(
                "The derivative order must be lower than the B-spline basis degree."
            )
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
        positive semidefinite matrix variables to fulfill the equations of
        Proposition 1 in Bertsimas and Popescu (2002). This is done by using the
        Frobenius inner product and, as it is stated in the proposition,
        matrices have order `deg_w`+ 1. For each interval of the B-spline basis,
        one matrix H is defined.

        Returns
        -------
        List[List[mosek.fusion.Matrix]]
            A list containing two list of matrices: the first one corresponds to
            the matrices used on the homogeneous equations, while the second is
            used on the non-homogeneous equations. For the last, the first
            matrix corresponds to the independent term equation, the second to
            the linear term equation, and so on.
        """

        # The elements from X in the homogeneous equations are located on its
        # even antidiagonals, while the ones from the non-homogeneous equations
        # are situated on the odd antidiagonals
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
                # Create an identity matrix along the corresponding diagonal and
                # the rotate it 90 degrees to get the antidiagonal. Then convert
                # it to a MOSEK sparse matrix
                H_by_diag.append(
                    mosek.fusion.Matrix.sparse(
                        np.rot90(np.eye(self.deg_w + 1, k=k, dtype=np.int32))
                    )
                )
            H.append(H_by_diag)
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
        n = [bsp.matrixB.shape[1] - bsp.deg for bsp in self.bspline]
        return model.variable(
            mosek.fusion.Domain.inPSDCone(
                self.deg_w + 1, len(self.constraints.keys()) * np.prod(n)
            )
        )

    def interval_cons(
        self,
        var_dict: Dict[str, mosek.fusion.LinearVariable],
        model: mosek.fusion.Model,
        matrices_S: Dict[int, Iterable[np.ndarray]],
    ) -> Tuple[mosek.fusion.LinearConstraint]:

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
        # lower than a certain threshold
        ind_term = self.matricesW[0][:, 0]
        # For every axis, get the contribution to the estimated function at the
        # knots
        for j, bsp in enumerate(self.bspline):
            # The contribution along the direction where the constraints are
            # imposed is W (that already include the vector coefficient arised
            # when differentiate the polynomial) times S once the first
            # `derivative` rows are deleted
            if self.var_name == j:
                matrices_S[self.var_name] = [
                    self.matricesW[i] @ np.delete(s, range(self.derivative), axis=0)
                    for i, s in enumerate(matrices_S[self.var_name])
                ]
            # Since the knot sequence is evenly spaced, the value of the
            # B-splines is periodic an it is always the same, so we pick up the
            # value of all the B-splines at the first knot
            else:
                value_at_knots = np.expand_dims(
                    bsp.matrixB[
                        bsp.int_back, bsp.int_back : bsp.int_back + bsp.deg + 1
                    ],
                    axis=0,
                )
                matrices_S[j] = [value_at_knots for _ in range(len(matrices_S[j]))]
        # For every interval on the `var_name` axis, count how many interval
        # constraints of the same sign and fixed derivative need to be
        # considered
        num_by_interval = (
            np.prod(
                np.array(
                    [
                        bsp.matrixB.shape[1] - bsp.deg
                        for i, bsp in enumerate(self.bspline)
                        if i != self.var_name
                    ]
                )
            )
            if len(matrices_S.keys()) > 1
            else 1
        )
        # For every interval on the `var_name` axis, count how many interval
        # constraints types by sign there are
        num_by_sign = len(self.constraints.keys())
        list_cons = []
        # Loop over every interval on the `var_name` axis
        for w in range(
            self.bspline[self.var_name].matrixB.shape[1]
            - self.bspline[self.var_name].deg
        ):
            # Create a list containing the lists with the contribution to the
            # estimated function at the knots
            a = [v for i, v in matrices_S.items() if i != self.var_name]
            # Create a list containing ranges of the same length as previous list
            a_idx = [range(len(v)) for i, v in matrices_S.items() if i != self.var_name]
            # Insert at the position `var_name` the correct value/index of S
            # along the `var_name` direction
            a.insert(self.var_name, [matrices_S[self.var_name][w]])
            a_idx.insert(self.var_name, range(w, w + 1))
            # Generate all the combinations possible from previous lists
            iter_a = list(itertools.product(*a))
            iter_idx = list(itertools.product(*a_idx))
            # Loop over the size of all the combinations (same length as
            # positive semidefinite variables) and their corresponding values of
            # the contribution
            for j, (id, mat) in enumerate(zip(iter_idx, iter_a)):
                last_id = [id[i] + bsp.deg + 1 for i, bsp in enumerate(self.bspline)]
                # Slice the multidimensional coefficient variable on the
                # interval on the values the corresponding B-spline is non-zero
                coef_theta = var_dict["theta"].slice(
                    np.array(id, dtype=np.int32), np.array(last_id, dtype=np.int32)
                )
                # Multiply the sliced variable on each face by the correct
                # contribution
                poly_coef = mosek.fusion.Expr.flatten(
                    matrix_by_tensor_product_mosek(matrices=mat, mosek_var=coef_theta)
                )
                # Loop over the different sign constraints
                for k, key in enumerate(self.constraints.keys()):
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
                    # Creates the homogeneous equations
                    for i in range(self.deg_w):
                        list_cons.append(
                            model.constraint(
                                mosek.fusion.Expr.dot(self.matricesH[0][i], slice_X),
                                mosek.fusion.Domain.equalsTo(0.0),
                            )
                        )
                    # Creates the non-homogeneous equations
                    for i in range(self.deg_w + 1):
                        if key == "+":
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
                        else:
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
        return tuple(list_cons)
