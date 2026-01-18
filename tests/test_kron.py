from functools import reduce

import numpy as np
import pytest
from scipy.linalg import block_diag

from src.cpsplines.utils.fast_kron import (
    kronecker_matrix_by_identity,
    weighted_double_kronecker,
)
from src.cpsplines.utils.timer import timer


# Take the Kronecker product of a random matrix and an identity matrix
# with a fast method, the fast Kronecker product and the NumPy method
@pytest.mark.parametrize(
    "A, n",
    [
        (np.random.rand(13, 17), 19),
        (np.random.rand(53, 59), 62),
    ],
)
def test_mat_kron_identity(A, n):
    with timer(tag="Using fast kronecker product by identity"):
        out = kronecker_matrix_by_identity(A, n)

    with timer(tag="Using np.kron"):
        exp_out = np.kron(A, np.eye(n))

    np.testing.assert_allclose(exp_out, out)


# Take the Kronecker product of an identity matrix and a random matrix
# with a fast method, the fast Kronecker product and the NumPy method
@pytest.mark.parametrize(
    "n, A",
    [
        (19, np.random.rand(13, 17)),
        (62, np.random.rand(53, 59)),
    ],
)
def test_identity_kron_mat(n, A):
    with timer(tag="Using fast kronecker product identity times matrix"):
        out = block_diag(*[A] * n)

    with timer(tag="Using np.kron"):
        exp_out = np.kron(np.eye(n), A)

    np.testing.assert_allclose(exp_out, out)


# Take the Kronecker product of a list of random matrices with different shapes
# with a fast method and the NumPy method
@pytest.mark.parametrize(
    "dim_mat, dim_W",
    [
        (((11, 3), (13, 5), (17, 7)), (11, 13, 17)),
    ],
)
def test_weighted_double_kronecker(dim_mat, dim_W):
    matrices = [np.random.rand(*d) for d in dim_mat]
    W = np.random.rand(*dim_W)

    with timer(tag="Using np.kron"):
        exp_out = (
            reduce(np.kron, matrices).T
            @ np.diag(W.flatten())
            @ reduce(np.kron, matrices)
        )

    with timer(tag="Using reshaping and permuting"):
        out = weighted_double_kronecker(
            matrices=matrices, W=W, data_arrangement="gridded"
        )

    np.testing.assert_allclose(exp_out, out)
