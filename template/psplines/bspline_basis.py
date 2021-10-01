import numpy as np
import math
from scipy.interpolate import BSpline
from typing import Dict, List, Union


class BsplineBasis:
    def __init__(
        self,
        deg: int,
        xsample: np.ndarray,
        n_int: int,
        prediction: Dict[str, Union[int, float]] = {},
    ):
        self.deg = deg
        self.xsample = np.sort(xsample)
        self.n_int = n_int
        self.prediction = prediction
        self.bspline_basis = self._construct_bspline_basis()
        self.matrixB = self.get_matrix_B()

    def _construct_bspline_basis(self) -> BSpline:

        if self.deg < 1:
            raise ValueError("The degree of the B-spline basis must be at least 1.")
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
            if self.prediction["backwards"] >= self.xsample[0]:
                raise ValueError(
                    "Backwards prediction limit must stand on the left-hand side of the regressor vector."
                )
        if "forward" in self.prediction:
            if self.prediction["forward"] <= self.xsample[-1]:
                raise ValueError(
                    "Forward prediction limit must stand on the right-hand side of the regressor vector."
                )

        step_length = (self.xsample[-1] - self.xsample[0]) / self.n_int

        self.int_back = (
            math.ceil((self.xsample[0] - self.prediction["backwards"]) / step_length)
            if "backwards" in self.prediction
            else 0
        )

        self.int_forw = (
            math.ceil((self.prediction["forward"] - self.xsample[-1]) / step_length)
            if "forward" in self.prediction
            else 0
        )

        knots = np.linspace(
            self.xsample[0] - (self.int_back + self.deg) * step_length,
            self.xsample[-1] + (self.int_forw + self.deg) * step_length,
            self.n_int + self.int_back + self.int_forw + 2 * self.deg + 1,
        )
        knots[self.int_back + self.deg] = self.xsample[0]
        knots[-(self.int_forw + self.deg + 1)] = self.xsample[-1]

        self.knots = knots

        return BSpline.construct_fast(
            t=self.knots,
            c=np.eye(
                self.n_int + self.deg + self.int_forw + self.int_back, dtype=float
            ),
            k=self.deg,
        )

    def get_matrix_B(self) -> np.ndarray:
        return self.bspline_basis.derivative(nu=0)(
            np.concatenate(
                [
                    self.knots[self.deg : self.deg + self.int_back],
                    self.xsample,
                    self.knots[self.int_back + self.n_int + self.deg + 1 : -self.deg],
                ]
            )
        )

    def get_matrices_S(self) -> List[np.ndarray]:

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
