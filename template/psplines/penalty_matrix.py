import numpy as np

from template.psplines.bspline_basis import BsplineBasis


class PenaltyMatrix:
    def __init__(self, bspline: BsplineBasis):
        self.bspline = bspline

    def get_diff_matrix(self, ord_d: int) -> np.ndarray:
        if self.bspline.deg <= ord_d:
            raise ValueError(
                "The penalty order must be less than the B-spline basis degree."
            )
        dim = (
            self.bspline.n_int
            + self.bspline.int_forw
            + self.bspline.int_back
            + self.bspline.deg
            + ord_d
        )
        return np.diff(np.eye(dim), n=ord_d)[ord_d:-ord_d, :].astype(int)
