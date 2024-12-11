from typing import Iterable

from cpsplines.psplines.bspline_basis import BsplineBasis


def get_idx_fitting_region(bspline_bases: Iterable[BsplineBasis]) -> tuple[slice]:
    """
    Get the fitting region indices on the expanded sample regressor vector
    (containing the extra knots from the extended B-spline basis and the sample
    regressor vector).

    Parameters
    ----------
    bspline_bases : Iterable[BsplineBasis]
        The B-spline bases.

    Returns
    -------
    Tuple[slice]
        Tuple of slices ranging from the first to the last index in the fitting
        region, one by axis.
    """

    return tuple(
        slice(bsp.int_back, bsp.int_back + len(bsp.x), None) for bsp in bspline_bases
    )
