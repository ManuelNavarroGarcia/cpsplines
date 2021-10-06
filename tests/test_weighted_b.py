import numpy as np
import pytest

from template.psplines.bspline_basis import BsplineBasis
from template.utils.fast_forecast_mat import get_weighted_B, get_idx_fitting_region

BV1 = (1 / 50) * np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 25, 25, 0, 0, 0, 0, 0],
        [0, 4, 37, 9, 0, 0, 0, 0],
        [0, 0, 16, 33, 1, 0, 0, 0],
        [0, 0, 1, 33, 16, 0, 0, 0],
        [0, 0, 0, 9, 37, 4, 0, 0],
        [0, 0, 0, 0, 25, 25, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
BV2 = (1 / 162) * np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 27, 108, 27, 0, 0, 0, 0],
        [0, 0, 1, 60, 93, 8, 0, 0, 0],
        [0, 0, 0, 8, 93, 60, 1, 0, 0],
        [0, 0, 0, 0, 27, 108, 27, 0, 0],
        [0, 0, 0, 0, 1, 60, 93, 8, 0],
        [0, 0, 0, 0, 0, 8, 93, 60, 1],
        [0, 0, 0, 0, 0, 0, 27, 108, 27],
    ]
)

BV3 = (1 / 98) * np.array(
    [
        [49, 49, 0, 0, 0, 0, 0, 0, 0],
        [4, 69, 25, 0, 0, 0, 0, 0, 0],
        [0, 16, 73, 9, 0, 0, 0, 0, 0],
        [0, 0, 36, 61, 1, 0, 0, 0, 0],
        [0, 0, 1, 61, 36, 0, 0, 0, 0],
        [0, 0, 0, 9, 73, 16, 0, 0, 0],
        [0, 0, 0, 0, 25, 69, 4, 0, 0],
        [0, 0, 0, 0, 0, 49, 49, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


@pytest.mark.parametrize(
    "x_sam, deg, n_int, prediction, VB",
    [
        (
            [np.linspace(0, 5, 6)],
            [2],
            [3],
            [{"backwards": -1, "forward": 8}],
            [BV1],
        ),
        (
            [np.linspace(0, 6, 7), np.linspace(-2.5, 3.7, 8)],
            [3, 2],
            [4, 5],
            [{"backwards": -2.5}, {"forward": 5}],
            [BV2, BV3],
        ),
    ],
)
def test_weighted_B(x_sam, deg, n_int, prediction, VB):
    bsp_out = []
    for x, d, n, pred in zip(x_sam, deg, n_int, prediction):
        bsp = BsplineBasis(deg=d, xsample=x, n_int=n, prediction=pred)
        bsp.get_matrix_B()
        bsp_out.append(bsp)

    out = get_weighted_B(bspline_bases=bsp_out)

    for mat_out, mat_in in zip(out, VB):
        np.testing.assert_allclose(mat_in, mat_out)


@pytest.mark.parametrize(
    "x_sam, deg, n_int, prediction, slice",
    [
        (
            [np.linspace(0, 5, 6)],
            [2],
            [3],
            [{"backwards": -1, "forward": 8}],
            (slice(1, 7, None),),
        ),
        (
            [np.linspace(0, 6, 7), np.linspace(-2.5, 3.7, 8)],
            [3, 2],
            [4, 5],
            [{"backwards": -2.5}, {"forward": 5}],
            (slice(2, 9, None), slice(0, 8, None)),
        ),
        (
            [np.linspace(0, 8, 71), np.linspace(-2, 4, 83), np.linspace(10, 11, 10)],
            [5, 4, 6],
            [7, 8, 3],
            [{"forward": 8.5}, {}, {"backwards": 9, "forward": 12.34}],
            (slice(0, 71, None), slice(0, 83, None), slice(3, 13, None)),
        ),
    ],
)
def test_get_idx_fit(x_sam, deg, n_int, prediction, slice):
    bsp_out = []
    for x, d, n, pred in zip(x_sam, deg, n_int, prediction):
        bsp = BsplineBasis(deg=d, xsample=x, n_int=n, prediction=pred)
        bsp.get_matrix_B()
        bsp_out.append(bsp)

    out = get_idx_fitting_region(bspline_bases=bsp_out)

    for slice_out, slice_in in zip(out, slice):
        assert slice_in == slice_out
