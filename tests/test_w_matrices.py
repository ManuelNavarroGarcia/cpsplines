import numpy as np
import pytest
from cpsplines.mosek_functions.interval_constraints import IntConstraints
from cpsplines.psplines.bspline_basis import BsplineBasis

W1 = [
    np.array(
        [
            [1, 0, 0, 0],
            [3, 5 / 2, 0, 0],
            [3, 5, 25 / 4, 0],
            [1, 5 / 2, 25 / 4, 125 / 8],
        ]
    ),
    np.array(
        [
            [1, 5 / 2, 25 / 4, 125 / 8],
            [3, 10, 125 / 4, 375 / 4],
            [3, 25 / 2, 50, 375 / 2],
            [1, 5, 25, 125],
        ]
    ),
    np.array(
        [
            [1, 5, 25, 125],
            [3, 35 / 2, 100, 1125 / 2],
            [3, 20, 131.25, 3375 / 4],
            [1, 15 / 2, 225 / 4, 3375 / 8],
        ]
    ),
    np.array(
        [
            [1, 15 / 2, 225 / 4, 3375 / 8],
            [3, 25, 825 / 4, 3375 / 2],
            [3, 55 / 2, 250, 2250],
            [1, 10, 100, 1000],
        ]
    ),
]

W2 = [
    np.array([[1, -14 / 3], [1, 0]]),
    np.array([[1, 0], [1, 14 / 3]]),
    np.array([[1, 14 / 3], [1, 28 / 3]]),
    np.array([[1, 28 / 3], [1, 14]]),
    np.array([[1, 14], [1, 56 / 3]]),
    np.array([[1, 56 / 3], [1, 70 / 3]]),
]

W3 = [
    np.array([[2, -12, 48], [4, -18, 48], [2, -6, 12]]),
    np.array([[2, -6, 12], [4, -6, 0], [2, 0, 0]]),
    np.array([[2, 0, 0], [4, 6, 0], [2, 6, 12]]),
    np.array([[2, 6, 12], [4, 18, 48], [2, 12, 48]]),
    np.array([[2, 12, 48], [4, 30, 144], [2, 18, 108]]),
]

# Test the creation of the weigth matrices W in the Proposition 1 of Bertsimas
# and Popescu (2002)
@pytest.mark.parametrize(
    "x_sam, deg, n_int, prediction, deriv, W",
    [
        (np.linspace(0, 10, 11), 3, 4, {}, 0, W1),
        (np.linspace(0, 7, 8), 2, 3, {"backwards": -1.5, "forward": 10}, 1, W2),
        (np.linspace(-2, 3, 18), 4, 5, {}, 2, W3),
    ],
)
def test_W_matrices(x_sam, deg, n_int, prediction, deriv, W):
    bsp = BsplineBasis(deg=deg, xsample=x_sam, n_int=n_int, prediction=prediction)
    bsp.get_matrix_B()
    W_out = IntConstraints(
        bspline=[bsp], var_name=0, derivative=deriv, constraints={}
    )._get_matrices_W()
    for mat, mat_out in zip(W, W_out):
        np.testing.assert_allclose(mat, mat_out, atol=1e-12)
