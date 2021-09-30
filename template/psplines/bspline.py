import numpy as np
import math
from scipy.interpolate import BSpline
from scipy.linalg import block_diag


class Bspline:

    """
    A class used to define the polynomials of the B-splines basis and the matrices related to them
    employed in the optimization problem.

    Attributes
    ----------
    deg: int
         The degree of the B-splines basis.
    xsample: list
         The training sample of the covariate.
    n_int: int
         The number of subintervals in which [x_min, x_max] is split.
    prediction: dict, optional (default is {})
         A dictionary that contains the most extreme prediction values on the covariate axis.
         The accepted keys are "forward" and "backwards" (one or both can be empty).
    step_length: float
         The length of the subintervals.
    int_forw: int
         The number of extra subintervals added to perform forward prediction.
    int_back: int
         The number of extra subintervals added to perform backwards prediction.
    knots: list
         The knots of the B-splines basis (including the knots of the prediction regions).
    """

    def __init__(
        self,
        deg: int,
        xsample: np.ndarray,
        n_int: int,
        prediction: dict = {},
    ):
        self.deg = deg
        self.xsample = np.sort(xsample)
        self.n_int = n_int
        self.prediction = prediction
        self.step_length = self.get_step_length()
        self.int_forw = self.get_int_forw()
        self.int_back = self.get_int_back()
        self.knots = self.get_knots()
        self.back_knots = self.get_back_knots()
        self.forw_knots = self.get_forw_knots()
        self.inner_knots = self.get_inner_knots()
        self.bspline_basis = self.get_bspline_basis()
        self.matrixB = self.B_eval(
            pts=np.concatenate([self.back_knots, self.xsample, self.forw_knots]),
            derivative=0,
        )

    def get_step_length(self) -> float:

        """
        Computes the length of the subintervals.

        Returns
        -------
        (float) The length of the subintervals.
        """

        return (self.xsample[-1] - self.xsample[0]) / self.n_int

    def get_int_forw(self) -> int:

        """
        Computes the number of extra knots needed for forward prediction.

        Returns
        -------
        (int) If 'forward' is not a key of prediction dict, then it returns 0. Otherwise,
        it computes the number of subintervals needed to reach the desired prediction value.
        """

        if "forward" in self.prediction:
            int_forw = math.ceil(
                (self.prediction["forward"] - self.xsample[-1]) / self.step_length
            )
        else:
            int_forw = 0
        return int_forw

    def get_int_back(self) -> int:

        """
        Computes the number of extra knots needed for backwards prediction.

        Returns
        -------
        (int) If 'backwards' is not a key of prediction dict, then it returns 0. Otherwise,
        it computes the number of subintervals needed to reach the desired prediction value.
        """

        if "backwards" in self.prediction:
            int_back = math.ceil(
                (self.xsample[0] - self.prediction["backwards"]) / self.step_length
            )
        else:
            int_back = 0
        return int_back

    def get_knots(self) -> np.ndarray:

        """
        Computes the position of the equally spaced knots of the B-splines basis.

        Returns
        -------
        (list) A list with the position of the knots, including the ones of the prediction
        region (if needed).
        """
        knots = np.linspace(
            self.xsample[0] - (self.int_back + self.deg) * self.step_length,
            self.xsample[-1] + (self.int_forw + self.deg) * self.step_length,
            self.n_int + self.int_back + self.int_forw + 2 * self.deg + 1,
        )
        knots[self.int_back + self.deg] = self.xsample[0]
        knots[-(self.int_forw + self.deg + 1)] = self.xsample[-1]
        return knots

    def get_back_knots(self) -> np.ndarray:
        if self.int_back > 0:
            back_knots = self.knots[self.deg : self.deg + self.int_back]
        else:
            back_knots = np.array([])
        return back_knots

    def get_forw_knots(self) -> np.ndarray:
        if self.int_forw > 0:
            forw_knots = self.knots[
                self.deg
                + self.n_int
                + self.int_back
                + 1 : self.deg
                + self.n_int
                + self.int_back
                + self.int_forw
                + 1
            ]
        else:
            forw_knots = np.array([])
        return forw_knots

    def get_inner_knots(self) -> np.ndarray:
        return self.knots[
            self.deg : self.deg + self.n_int + self.int_back + self.int_forw + 1
        ]

    def get_bspline_basis(self):
        return BSpline.construct_fast(
            t=self.knots,
            c=np.eye(
                self.n_int + self.deg + self.int_forw + self.int_back, dtype=float
            ),
            k=self.deg,
        )

    def B_eval(self, pts: np.ndarray, derivative: int) -> np.ndarray:
        return self.bspline_basis.derivative(nu=derivative)(np.sort(pts))

    def matricesS(self) -> list[np.ndarray]:

        """
        Computes the matrices S, which contains the monomial coefficients of all the
        coefficients in the B-splines basis for each of the subinterval.

        Returns
        -------
        (list) Each element of the list is a np.ndarray matrix. The first row correspond to the
        independent coefficient, the second row to the linear coefficients and so on.
        The first column corresponds to the monomials of the 1st B-spline of the basis,
        the second column to the 2nd B-spline and so on.
        """

        pt_mat = np.zeros(shape=(self.deg + 1, self.deg + 1))
        for i in range(self.deg + 1):
            pt_mat[i, :] = BSpline.basis_element(t=self.knots[i : self.deg + 2 + i])(
                x=np.linspace(
                    self.knots[self.deg], self.knots[self.deg + 1], self.deg + 1
                )
            )
        S = []
        for k in range(self.n_int + self.int_back + self.int_forw):
            vander = np.vander(
                np.linspace(
                    self.knots[k + self.deg], self.knots[k + self.deg + 1], self.deg + 1
                ),
                increasing=True,
            )
            S_k = np.linalg.solve(vander, pt_mat.T)
            S.append(S_k)
        return S

    def matrixV(self) -> np.ndarray:

        """
        Computes the matrix V, which assign a weight 1 to the points on the training sample
        and weight 0 to the points on the prediction region.

        Returns
        -------
        (np.ndarray) The matrix V
        """

        return block_diag(
            np.zeros(shape=(self.int_back, self.int_back)),
            np.eye(len(self.xsample)),
            np.zeros(shape=(self.int_forw, self.int_forw)),
        )

    def B_expanded(self, x_back_pts: list, x_forw_pts: list) -> np.ndarray:

        """
        Build the expanded B matrix using points in the prediction regions.

        Inputs
        ------
        x_back_pts: list

        x_forw_pts: list

        Returns
        -------
        (np.ndarray) The expanded B matrix
        """
        B_expanded = np.zeros(
            shape=(
                len(self.xsample) + len(x_back_pts) + len(x_forw_pts),
                self.n_int + self.deg + self.int_back + self.int_forw,
            )
        )
        # Remove the rows related with the extra knots used in the prediction
        B_expanded[
            len(x_back_pts) : len(self.xsample) + len(x_back_pts), :
        ] = self.matrixB[self.int_back : len(self.xsample) + self.int_back, :]

        # Create the rows of matrix B corresponding to backwards prediction
        # points and concatenate
        if len(x_back_pts) > 0:
            B_expanded[: len(x_back_pts), :] = self.B_eval(pts=x_back_pts, derivative=0)

        # Create the rows of matrix B corresponding to forward prediction
        # points and concatenate
        if len(x_forw_pts) > 0:
            B_expanded[len(self.xsample) + len(x_back_pts) :, :] = self.B_eval(
                pts=x_forw_pts, derivative=0
            )
        return B_expanded
