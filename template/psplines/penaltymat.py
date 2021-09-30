import numpy as np


class Penaltymat:

    """
    A class used to define the penalty matrix P, which penalizes the difference between
    adjacent coefficients. It is defined using the difference operator.

    Attributes
    ----------
    ord_d : int
         the order of the penalty.
    bspline : bs.Bspline
         The B-splines basis.
    """

    def __init__(self, ord_d: int, bspline):
        self._ord_d = ord_d
        self._bspline = bspline

    @property
    def ord_d(self):
        return self._ord_d

    @property
    def bspline(self):
        return self._bspline

    @ord_d.setter
    def ord_d(self, ord_d: int):
        self._ord_d = ord_d

    @bspline.setter
    def bspline(self, bspline):
        self._bspline = bspline

    def matrix_D(self) -> np.ndarray:

        """It computes the difference matrix D

        Returns
        -------
        (np.ndarray) The D matrix with shape (n_int + int_forw + int_back - ord_d + deg,
        n_int + int_forw + int_back + deg)
        """
        dim = (
            self.bspline.n_int
            + self.bspline.int_forw
            + self.bspline.int_back
            + self.bspline.deg
            + self.ord_d
        )
        return np.diff(np.eye(dim), n=self.ord_d)[self.ord_d : -self.ord_d, :].astype(
            int
        )
