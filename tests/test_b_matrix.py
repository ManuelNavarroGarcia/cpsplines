import numpy as np
import pytest
from cpsplines.psplines.bspline_basis import BsplineBasis
from cpsplines.utils.weighted_b import get_idx_fitting_region, get_weighted_B

B1 = (1 / 8) * np.array(
    [
        [4, 4, 0, 0, 0, 0, 0],
        [1, 6, 1, 0, 0, 0, 0],
        [0, 4, 4, 0, 0, 0, 0],
        [0, 1, 6, 1, 0, 0, 0],
        [0, 0, 4, 4, 0, 0, 0],
        [0, 0, 1, 6, 1, 0, 0],
        [0, 0, 0, 4, 4, 0, 0],
        [0, 0, 0, 1, 6, 1, 0],
        [0, 0, 0, 0, 4, 4, 0],
        [0, 0, 0, 0, 1, 6, 1],
        [0, 0, 0, 0, 0, 4, 4],
    ]
)

B2 = (1 / 750) * np.array(
    [
        [125, 500, 125, 0, 0, 0, 0, 0, 0],
        [8, 311, 404, 27, 0, 0, 0, 0, 0],
        [0, 64, 473, 212, 1, 0, 0, 0, 0],
        [0, 1, 212, 473, 64, 0, 0, 0, 0],
        [0, 0, 27, 404, 311, 8, 0, 0, 0],
        [0, 0, 0, 125, 500, 125, 0, 0, 0],
        [0, 0, 0, 8, 311, 404, 27, 0, 0],
        [0, 0, 0, 0, 64, 473, 212, 1, 0],
        [0, 0, 0, 0, 1, 212, 473, 64, 0],
        [0, 0, 0, 0, 0, 27, 404, 311, 8],
        [0, 0, 0, 0, 0, 0, 125, 500, 125],
    ]
)

B3 = (1 / 240000) * np.array(
    [
        [10000, 110000, 110000, 10000, 0, 0, 0, 0, 0, 0, 0],
        [81, 28156, 137846, 71516, 2401, 0, 0, 0, 0, 0, 0],
        [0, 1296, 59056, 142256, 37136, 256, 0, 0, 0, 0, 0],
        [0, 0, 6561, 97516, 121286, 14636, 1, 0, 0, 0, 0],
        [0, 0, 16, 20656, 130736, 84496, 4096, 0, 0, 0, 0],
        [0, 0, 0, 625, 47500, 143750, 47500, 625, 0, 0, 0],
        [0, 0, 0, 0, 4096, 84496, 130736, 20656, 16, 0, 0],
        [0, 0, 0, 0, 1, 14636, 121286, 97516, 6561, 0, 0],
        [0, 0, 0, 0, 0, 256, 37136, 142256, 59056, 1296, 0],
        [0, 0, 0, 0, 0, 0, 2401, 71516, 137846, 28156, 81],
        [0, 0, 0, 0, 0, 0, 0, 10000, 110000, 110000, 10000],
    ]
)

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

# Test if the matrix B is computed properly using different parameters. Since at
# the end we are interested on the weighted matrices, the output is the list of
# weighted matrices from `get_weighted_B`.


@pytest.mark.parametrize(
    "x_sam, deg, n_int, prediction, B",
    [
        ([np.linspace(0, 10, 11)], [2], [5], [{}], [B1]),
        ([np.linspace(0, 10, 11)], [3], [6], [{}], [B2]),
        ([np.linspace(0, 10, 11)], [4], [7], [{}], [B3]),
        ([np.linspace(0, 5, 6)], [2], [3], [{"backwards": -1, "forward": 8}], [BV1]),
        (
            [np.linspace(0, 6, 7), np.linspace(-2.5, 3.7, 8)],
            [3, 2],
            [4, 5],
            [{"backwards": -2.5}, {"forward": 5}],
            [BV2, BV3],
        ),
    ],
)
def test_B_matrix(x_sam, deg, n_int, prediction, B):
    bspline = []
    for x, d, n, pred in zip(x_sam, deg, n_int, prediction):
        bsp = BsplineBasis(deg=d, xsample=x, n_int=n, prediction=pred)
        bsp.get_matrix_B()
        bspline.append(bsp)

    B_out = get_weighted_B(bspline_bases=bspline)

    for P, Q in zip(B_out, B):
        np.testing.assert_allclose(P, Q)


# Test correct ranges of the fitting region given a regressor sample
@pytest.mark.parametrize(
    "x_sam, deg, n_int, prediction, x_range",
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
def test_get_idx_fit(x_sam, deg, n_int, prediction, x_range):
    bspline = []
    for x, d, n, pred in zip(x_sam, deg, n_int, prediction):
        bsp = BsplineBasis(deg=d, xsample=x, n_int=n, prediction=pred)
        bsp.get_matrix_B()
        bspline.append(bsp)

    range_out = get_idx_fitting_region(bspline_bases=bspline)

    for slice_out, slice_in in zip(range_out, x_range):
        assert slice_in == slice_out
