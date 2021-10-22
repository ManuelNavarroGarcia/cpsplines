import numpy as np
import pytest
from cpsplines.psplines.bspline_basis import BsplineBasis
from cpsplines.psplines.penalty_matrix import PenaltyMatrix

D1 = np.array(
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

D2 = np.array(
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

D3 = np.array(
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
    "x_sam, deg, n_int, prediction, ord_d, D",
    [
        (np.linspace(0, 10, 11), 3, 5, {}, 2, D1),
        (np.linspace(-2, 12, 17), 6, 7, {}, 4, D2),
        (np.linspace(0, 10, 11), 4, 5, {"backwards": -2.5, "forward": 11.9}, 1, D3),
    ],
)
def test_D_matrix(x_sam, deg, n_int, prediction, ord_d, D):

    bsp = BsplineBasis(deg=deg, xsample=x_sam, n_int=n_int, prediction=prediction)
    bsp.get_matrix_B()
    D_out = PenaltyMatrix(bspline=bsp).get_penalty_matrix(**{"ord_d": ord_d})
    np.testing.assert_allclose(D_out, D)
