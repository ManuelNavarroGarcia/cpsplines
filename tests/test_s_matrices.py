import numpy as np
import pytest
from cpsplines.psplines.bspline_basis import BsplineBasis

S1 = [
    np.array(
        [
            [1 / 6, 2 / 3, 1 / 6, 0],
            [-1 / 5, 0, 1 / 5, 0],
            [2 / 25, -4 / 25, 2 / 25, 0],
            [-4 / 375, 4 / 125, -4 / 125, 4 / 375],
        ]
    ),
    np.array(
        [
            [4 / 3, -5 / 6, 2 / 3, -1 / 6],
            [-4 / 5, 7 / 5, -4 / 5, 1 / 5],
            [4 / 25, -2 / 5, 8 / 25, -2 / 25],
            [-4 / 375, 4 / 125, -4 / 125, 4 / 375],
        ]
    ),
    np.array(
        [
            [9 / 2, -22 / 3, 31 / 6, -4 / 3],
            [-9 / 5, 4, -3, 4 / 5],
            [6 / 25, -16 / 25, 14 / 25, -4 / 25],
            [-4 / 375, 4 / 125, -4 / 125, 4 / 375],
        ]
    ),
    np.array(
        [
            [32 / 3, -131 / 6, 50 / 3, -9 / 2],
            [-16 / 5, 39 / 5, -32 / 5, 9 / 5],
            [8 / 25, -22 / 25, 4 / 5, -6 / 25],
            [-4 / 375, 4 / 125, -4 / 125, 4 / 375],
        ]
    ),
]

S2 = [
    np.array(
        [
            [0, 1 / 2, 1 / 2],
            [0, -3 / 7, 3 / 7],
            [9 / 98, -9 / 49, 9 / 98],
        ]
    ),
    np.array(
        [
            [1 / 2, 1 / 2, 0],
            [-3 / 7, 3 / 7, 0.0],
            [9 / 98, -9 / 49, 9 / 98],
        ]
    ),
    np.array(
        [
            [2, -3 / 2, 1 / 2],
            [-6 / 7, 9 / 7, -3 / 7],
            [9 / 98, -9 / 49, 9 / 98],
        ]
    ),
    np.array(
        [
            [9 / 2, -11 / 2, 2],
            [-9 / 7, 15 / 7, -6 / 7],
            [9 / 98, -9 / 49, 9 / 98],
        ]
    ),
    np.array(
        [
            [8, -23 / 2, 9 / 2],
            [-12 / 7, 3, -9 / 7],
            [9 / 98, -9 / 49, 9 / 98],
        ]
    ),
    np.array(
        [
            [12.5, -19.5, 8.0],
            [-15 / 7, 27 / 7, -12 / 7],
            [9 / 98, -9 / 49, 9 / 98],
        ]
    ),
]

# Test the matrices S list for different parameters on the B-spline basis and
# with and without prediction


@pytest.mark.parametrize(
    "x_sam, deg, n_int, prediction, S",
    [
        (np.linspace(0, 10, 11), 3, 4, {}, S1),
        (np.linspace(0, 7, 8), 2, 3, {"backwards": -1.5, "forward": 10}, S2),
    ],
)
def test_S_matrices(x_sam, deg, n_int, prediction, S):
    bsp = BsplineBasis(deg=deg, xsample=x_sam, n_int=n_int, prediction=prediction)
    bsp.get_matrix_B()
    S_out = bsp.get_matrices_S()
    for mat, mat_out in zip(S, S_out):
        np.testing.assert_allclose(mat, mat_out, atol=1e-12)
