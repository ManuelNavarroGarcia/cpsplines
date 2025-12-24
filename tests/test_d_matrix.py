import numpy as np
import pytest

from src.cpsplines.psplines.bspline_basis import BsplineBasis
from src.cpsplines.psplines.penalty_matrix import PenaltyMatrix

P1 = np.array(
    [
        [1, -2, 1, 0, 0, 0, 0, 0],
        [-2, 5, -4, 1, 0, 0, 0, 0],
        [1, -4, 6, -4, 1, 0, 0, 0],
        [0, 1, -4, 6, -4, 1, 0, 0],
        [0, 0, 1, -4, 6, -4, 1, 0],
        [0, 0, 0, 1, -4, 6, -4, 1],
        [0, 0, 0, 0, 1, -4, 5, -2],
        [0, 0, 0, 0, 0, 1, -2, 1],
    ]
)

P2 = np.array(
    [
        [1, -4, 6, -4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-4, 17, -28, 22, -8, 1, 0, 0, 0, 0, 0, 0, 0],
        [6, -28, 53, -52, 28, -8, 1, 0, 0, 0, 0, 0, 0],
        [-4, 22, -52, 69, -56, 28, -8, 1, 0, 0, 0, 0, 0],
        [1, -8, 28, -56, 70, -56, 28, -8, 1, 0, 0, 0, 0],
        [0, 1, -8, 28, -56, 70, -56, 28, -8, 1, 0, 0, 0],
        [0, 0, 1, -8, 28, -56, 70, -56, 28, -8, 1, 0, 0],
        [0, 0, 0, 1, -8, 28, -56, 70, -56, 28, -8, 1, 0],
        [0, 0, 0, 0, 1, -8, 28, -56, 70, -56, 28, -8, 1],
        [0, 0, 0, 0, 0, 1, -8, 28, -56, 69, -52, 22, -4],
        [0, 0, 0, 0, 0, 0, 1, -8, 28, -52, 53, -28, 6],
        [0, 0, 0, 0, 0, 0, 0, 1, -8, 22, -28, 17, -4],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -4, 6, -4, 1],
    ]
)

P3 = np.array(
    [
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1],
    ]
)


# Test the computation of the penalty matrices, for a variety of parameters on
# the B-spline basis construction and the arguments of the penalty (such as the
# penalty order for discrete difference matrices)
@pytest.mark.parametrize(
    "x, deg, k, prediction, ord_d, P",
    [
        (np.linspace(0, 10, 11), 3, 5, {}, 2, P1),
        (np.linspace(-2, 12, 17), 6, 7, {}, 4, P2),
        (np.linspace(0, 10, 11), 4, 5, {"backwards": -2.5, "forward": 11.9}, 1, P3),
    ],
)
def test_D_matrix(x, deg, k, prediction, ord_d, P):
    bsp = BsplineBasis(deg=deg, x=x, k=k, prediction=prediction)
    P_out = PenaltyMatrix(bspline=bsp).get_penalty_matrix(**{"ord_d": ord_d})
    np.testing.assert_allclose(P_out, P)
