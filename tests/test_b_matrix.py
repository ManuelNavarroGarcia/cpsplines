import numpy as np
import pytest

from template.psplines.bspline_basis import BsplineBasis

B2 = (1 / 8) * np.array(
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

B3 = (1 / 750) * np.array(
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

B4 = (1 / 240000) * np.array(
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


@pytest.mark.parametrize(
    "x_sam, deg, n_int, B",
    [
        (np.linspace(0, 10, 11), 2, 5, B2),
        (np.linspace(0, 10, 11), 3, 6, B3),
        (np.linspace(0, 10, 11), 4, 7, B4),
    ],
)
def test_B_matrix(x_sam, deg, n_int, B):

    bsp = BsplineBasis(deg=deg, xsample=x_sam, n_int=n_int)
    bsp.get_matrix_B()
    np.testing.assert_allclose(bsp.matrixB, B)
