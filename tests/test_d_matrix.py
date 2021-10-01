import numpy as np
import pytest

from template.psplines.bspline import Bspline
from template.psplines.penalty_matrix import PenaltyMatrix


D1 = np.array(
    [
        [1, -2, 1, 0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0, 0, 0, 0],
        [0, 0, 1, -2, 1, 0, 0, 0],
        [0, 0, 0, 1, -2, 1, 0, 0],
        [0, 0, 0, 0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0, 1, -2, 1],
    ]
)

D2 = np.array(
    [
        [1, -4, 6, -4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, -4, 6, -4, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -4, 6, -4, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, -4, 6, -4, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -4, 6, -4, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -4, 6, -4, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -4, 6, -4, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, -4, 6, -4, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -4, 6, -4, 1],
    ]
)

D3 = np.array(
    [
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
    ]
)


@pytest.mark.parametrize(
    "x_sam, deg, n_int, prediction, ord_d, D",
    [
        (np.linspace(0, 10, 11), 3, 5, {}, 2, D1),
        (np.linspace(-2, 12, 17), 6, 7, {}, 4, D2),
        (np.linspace(0, 10, 11), 4, 5, {"backwards": -2.5, "forward": 11.9}, 1, D3),
    ],
)
def test_D_matrix(x_sam, deg, n_int, prediction, ord_d, D):

    bsp = Bspline(deg=deg, xsample=x_sam, n_int=n_int, prediction=prediction)
    D_out = PenaltyMatrix(bspline=bsp).get_diff_matrix(ord_d=ord_d)
    np.testing.assert_allclose(D_out, D)
