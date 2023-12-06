import math
from typing import Dict, Union

import numpy as np
from scipy.interpolate import BSpline


class BsplineBasis:

    """
    Generate a univariate uniform B-spline basis from a regressor vector. The
    knot sequence is defined using `n_int + 2 * deg + 1` evenly spaced knots so
    that the (`deg` + 1)-th knot coincides with the minimum value of `xsample` and
    the (`n_int` + `deg` + 1) matches the maximum value of `xsample`. The B-spline
    basis may be extended outwards the range defined by `xsample` adding extra
    knots (preserving the same step length) to the knot sequence as it is done
    in Currie I., Durban M. and Eilers P. (2004).

    Parameters
    ----------
    xsample : np.ndarray
        The regressor vector. Must be a unidimensional array.
    deg : int, optional
        The polynomial degree of the B-spline basis. Must be a non-negative
        integer. By default, 3.
    n_int : int, optional
        The number of equal intervals which [min(`xsample`), max(`xsample`)] is
        split. Must be greater or equal than 2. By default, 40.
    prediction : Dict[str, Union[int, float]], optional
        A dictionary containing the most extreme values that the extended
        basis needs to capture. The keys are 'backwards' and 'forward',
        depending on the direction which the basis must be extended, and the
        values are the points on the basis axis. If 'backwards' ('forward') keys
        is not empty, the value must be at the left (right) of the minimum
        (maximum) value of 'xsample'. By default, {}.

    Attributes
    ----------
    int_back : int
        The number of extra knots used to extend the B-spline basis to the left.
    int_forw : int
        The number of extra knots used to extend the B-spline basis to the right.
    knots : np.ndarray of shape (`n_int` + 2 * `deg` + 1, )
        The knot sequence of the B-spline basis.
    bspline_basis : scipy.interpolate.Bspline
        The `n_int` + `int_back` + `int_forw` + `deg` elements of the B-spline basis.
    matrixB : np.ndarray of shape (`int_back` + len(`xsample`) + `int_forw`,
        `n_int` + `deg`)
        The design matrix of the B-spline basis. The ij-th element of the matrix
        contains the evaluation of the j-th B-spline from the basis at the i-th
        element of the vector `xsample`.

    References
    ----------
    - Currie, I. D., Durban, M., & Eilers, P. H. (2004). Smoothing and
      forecasting mortality rates. Statistical modelling, 4(4), 279-298.
    - De Boor, C., & De Boor, C. (1978). A practical guide to splines (Vol. 27).
      New York: Springer-Verlag.
    """

    def __init__(
        self,
        xsample: np.ndarray,
        deg: int = 3,
        n_int: int = 40,
        prediction: Dict[str, Union[int, float]] = {},
    ):
        self.xsample = xsample
        self.deg = deg
        self.n_int = n_int
        self.prediction = prediction

    def get_matrix_B(self) -> None:
        """
        Defines the design matrix of the B-spline basis, consisting on the
        evaluation of the B-spline basis polynomials at `xsample`. This is
        achieved determining how many extra knots are needed to extend the basis
        on the desired range, defining the knot sequence of the basis, which is
        constructed using `construct_fast` from scipy.interpolate.Bspline, and
        finally evaluating this basis at `xsample`. Hence, the shape of the
        matrix is  (`int_back` + len(`xsample`) + `int_forw`, `n_int` + `deg`)

        Raises
        ------
        ValueError
            If `deg` is a negative integer.
        ValueError
            If `xsample` is not a unidimensional array.
        ValueError
            If `n_int` is less than 2.
        ValueError
            If any key in `prediction` is different from 'forward' or
            'backwards'.
        ValueError
            If the bound given in 'backwards' is inside the convex hull of
            `xsample`.
        ValueError
            If the bound given in 'forward' is inside the convex hull of
            `xsample`.
        """
        min_x = self.xsample.min()
        max_x = self.xsample.max()
        if self.deg < 0:
            raise ValueError("The degree of the B-spline basis must be at least 0.")
        if self.xsample.ndim != 1:
            raise ValueError("Regressor vector must be one-dimensional.")
        if self.n_int < 2:
            raise ValueError(
                "The fitting regions must be split in at least 2 intervals."
            )
        if len(set(self.prediction) - set(["backwards", "forward"])) > 0:
            raise ValueError(
                "Prediction only admits as keys `forward` and `backwards`."
            )

        if "backwards" in self.prediction:
            if self.prediction["backwards"] >= min_x:
                raise ValueError(
                    (
                        "Backwards prediction limit must stand on the "
                        "left-hand side of the regressor vector."
                    )
                )
        if "forward" in self.prediction:
            if self.prediction["forward"] <= max_x:
                raise ValueError(
                    (
                        "Forward prediction limit must stand on the "
                        "right-hand side of the regressor vector."
                    )
                )

        # Compute the distance between adjacent knots
        step_length = (max_x - min_x) / self.n_int

        # Determine how many knots at the left (right) of min(`xsample`)
        # (max(`xsample`)) are needed to extend the basis backwards (forward).
        # The step length between these new knots must be the same
        self.int_back = (
            math.ceil((min_x - self.prediction["backwards"]) / step_length)
            if "backwards" in self.prediction
            else 0
        )

        self.int_forw = (
            math.ceil((self.prediction["forward"] - max_x) / step_length)
            if "forward" in self.prediction
            else 0
        )
        # Construct the knot sequence of the B-spline basis, consisting on
        # `n_int` + 2 * `deg` + 1 equally spaced knots
        knots = np.linspace(
            min_x - (self.int_back + self.deg) * step_length,
            max_x + (self.int_forw + self.deg) * step_length,
            self.n_int + self.int_back + self.int_forw + 2 * self.deg + 1,
        )
        # To avoid floating point error, force that the (`deg` + 1)-th knot
        # coincides with the minimum value of `xsample` and the
        # (`n_int` + `deg` + 1) matches the maximum value of `xsample`
        knots[self.int_back + self.deg] = min_x
        knots[-(self.int_forw + self.deg + 1)] = max_x
        self.knots = knots

        # Construct the B-spline basis, consisting on
        # (`n_int` + `int_back` + `int_forw` + `deg`) elements
        self.bspline_basis = BSpline.construct_fast(
            t=self.knots,
            c=np.eye(
                self.n_int + self.deg + self.int_forw + self.int_back, dtype=float
            ),
            k=self.deg,
        )
        # Return the design matrix of the B-spline basis
        x_eval = np.concatenate(
            [
                self.knots[self.deg : self.deg + self.int_back],
                self.xsample,
                self.knots[self.int_back + self.n_int + self.deg + 1 : -self.deg],
            ]
        )
        self.matrixB = self.bspline_basis(x=x_eval)
        return None

    def get_matrices_S(self):
        """
        Generate a list of matrices (one by interval) containing the polynomial
        coefficients from the non-zero B-spline basis elements . For the j-th
        matrix, the first row contains the independent coefficients from the
        (j-`deg`)-th to the j-th B-spline basis elements, the second row
        contains the linear coefficients from the polynomials, and so on.

        These matrices are required to enforce non-negativity constraints using
        Proposition 1 in Bertsimas and Popescu (2002).

        References
        ----------
        - Bertsimas, D., & Popescu, I. (2002). On the relation between option
          and stock prices: a convex optimization approach. Operations Research,
          50(2), 358-374.

        """

        # Since knots are evenly spaced, the value of the B-spline basis
        # elements is periodic with period the step length of the knot sequence.
        # Hence, we compute the value of the non-zero basis elements at only one
        # interval (take advantage of the periodicity) at `deg` + 1 points.
        # These points should be located at the same interval, and we choose
        #          linspace(knots[`deg`], knots[`deg` + 1], `deg` + 1),
        # since this is the first interval with (`deg` + 1) non-zero elements.
        C = np.zeros(shape=(self.deg + 1, self.deg + 1))
        for i in range(self.deg + 1):
            C[:, i] = BSpline.basis_element(t=self.knots[i : self.deg + 2 + i])(
                x=np.linspace(
                    self.knots[self.deg], self.knots[self.deg + 1], self.deg + 1
                )
            )
        S = []
        # The matrices S_k that we are looking for satisfy that S_k @ T_k = C,
        # where T_k has as columns the array (1, x, x**2, ... x**`deg`)
        # evaluated at the points
        #     linspace(knots[k + `deg`], knots[k + `deg` + 1], `deg` + 1)
        for k in range(self.n_int + self.int_back + self.int_forw):
            T_k = np.vander(
                np.linspace(
                    self.knots[k + self.deg], self.knots[k + self.deg + 1], self.deg + 1
                ),
                increasing=True,
            )
            S_k = np.linalg.solve(T_k, C)
            S.append(S_k)
        self.matrices_S = S
        return None
