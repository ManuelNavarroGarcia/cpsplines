import numpy as np

from template.psplines.bspline import Bspline


class PenaltyMatrix:
    def __init__(self, bspline: Bspline):
        self.bspline = bspline

    def get_diff_matrix(self, ord_d: int) -> np.ndarray:
        dim = (
            self.bspline.n_int
            + self.bspline.int_forw
            + self.bspline.int_back
            + self.bspline.deg
            + ord_d
        )
        return np.diff(np.eye(dim), n=ord_d)[ord_d:-ord_d, :].astype(int)
