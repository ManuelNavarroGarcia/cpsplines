from typing import Iterable, List, Tuple

import numpy as np
from cpsplines.psplines.bspline_basis import BsplineBasis


def get_idx_fitting_region(bspline_bases: Iterable[BsplineBasis]) -> Tuple[slice]:

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
        slice(bsp.int_back, bsp.int_back + len(bsp.xsample), None)
        for bsp in bspline_bases
    )


def get_weighted_B(bspline_bases: Iterable[BsplineBasis]) -> List[np.ndarray]:

    """
    Get the weighted design matrices from the B-spline bases, which assing a
    zero weight to the rows of the values to be predicted and one weight to the
    rows on the fitting region.

    Parameters
    ----------
    bspline_bases : Iterable[BsplineBasis]
        The B-spline bases.

    Returns
    -------
    List[np.ndarray]
        The weighted matrices B.
    """

    mat = []
    idx_fit = get_idx_fitting_region(bspline_bases=bspline_bases)
    for i, bsp in enumerate(bspline_bases):
        B_weighted = np.zeros((bsp.matrixB.shape))
        B_weighted[idx_fit[i], :] = bsp.matrixB[idx_fit[i], :]
        mat.append(B_weighted)
    return mat
