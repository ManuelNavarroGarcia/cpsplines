import numpy as np
from cpsplines.psplines.bspline_basis import BsplineBasis


class PenaltyMatrix:

    """
    Constructs the penalty matrix of the P-splines (Eilers, P. H., Marx, B. D.
    (1996)) given a B-spline basis. Standard P-splines use a penalty that is
    based on repeated differences (`variation`='diff').

    Parameters
    ----------
    bspline : BsplineBasis
        The B-spline basis object.
    variation : str, default = 'diff'
        The penalty variation. A summary of different penalties can be found in
        Eilers, P. H., Marx, B. D., & Durbán, M. (2015).

    References
    ----------
    - Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with B-splines and
      penalties. Statistical science, 11(2), 89-121.
    - Eilers, P. H., Marx, B. D., & Durbán, M. (2015). Twenty years of
      P-splines. SORT: statistics and operations research transactions, 39(2),
      149-186.
    """

    def __init__(self, bspline: BsplineBasis, variation: str = "diff"):
        self.bspline = bspline
        self.variation = variation

    def _get_diff_matrix(self, ord_d: int = 2) -> np.ndarray:

        """
        Generate the penalty matrix based on finite differences of the
        coefficients of adjacent B-splines

        Parameters
        ----------
        ord_d : int
            The order of the penalty. Must be a non-negative integer. By
            default, 2.

        Returns
        -------
        np.ndarray of shape (`bspline.n_int` + `bspline.int_forw` +
        `bspline.int_back` + `bspline.deg`, `bspline.n_int` + `bspline.int_forw`
        + `bspline.int_back` + `bspline.deg`)
            The penalty matrix of 'diff' variation

        Raises
        ------
        ValueError
            If penalty order is less than B-spline basis degree.
        ValueError
            If `ord_d` is not a non-negative integer.
        """

        if self.bspline.deg <= ord_d:
            raise ValueError(
                "The penalty order must be less than the B-spline basis degree."
            )
        if ord_d < 0:
            raise ValueError("The penalty order must be a non-negative integer.")

        dim = (
            self.bspline.n_int
            + self.bspline.int_forw
            + self.bspline.int_back
            + self.bspline.deg
            + ord_d
        )
        # Generate an identity matrix of order `bspline.n_int` +
        # `bspline.int_forw` +  `bspline.int_back` + `bspline.deg` and generate
        # its difference matrix. Then, remove the first and last `ord_d` rows
        D = np.diff(np.eye(dim, dtype=np.int32), n=ord_d)[ord_d:-ord_d, :]
        return D.T @ D

    def get_penalty_matrix(self, **kwargs) -> np.ndarray:

        """Generates the penalty matrix associated with the B-spline basis
        `bspline` and the variation `variation`. If the chosen variation needs
        some parameters to be defined, they must be included on the kwargs.

        Returns
        -------
        np.ndarray of shape (`bspline.n_int` + `bspline.int_forw` +
        `bspline.int_back` + `bspline.deg`, `bspline.n_int` + `bspline.int_forw`
        + `bspline.int_back` + `bspline.deg`)
            The penalty matrix.

        Raises
        ------
        ValueError
            Variation type must be on ("diff", )
        """

        if self.variation not in ("diff",):
            raise ValueError("Penalty matrix type not valid.")
        if self.variation == "diff":
            return self._get_diff_matrix(**kwargs)
