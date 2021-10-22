from functools import reduce

import numpy as np
import pytest
from cpsplines.utils.fast_kron import (
    fast_kronecker_product,
    kronecker_matrix_by_identity,
)
from cpsplines.utils.timer import timer
from scipy.linalg import block_diag


# Take the Kronecker product of a list of random matrices with different shapes
# with a fast method and the NumPy method
@pytest.mark.parametrize(
    "dim",
    [
        ((4, 7), (5, 6), (8, 3), (1, 9), (10, 2)),
    ],
)
def test_fast_kronecker(dim):
    matrices = [np.random.rand(*d) for d in dim]

    with timer(tag="Using np.kron"):
        exp_out = reduce(np.kron, matrices)

    with timer(tag="Using broadcasting"):
        out = reduce(fast_kronecker_product, matrices)

    np.testing.assert_allclose(exp_out, out)


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

    with timer(tag="Using broadcasting"):
        exp_out2 = fast_kronecker_product(A=A, B=np.eye(n))

    np.testing.assert_allclose(exp_out, out)
    np.testing.assert_allclose(exp_out2, out)


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

    with timer(tag="Using broadcasting"):
        exp_out2 = fast_kronecker_product(A=np.eye(n), B=A)

    np.testing.assert_allclose(exp_out, out)
    np.testing.assert_allclose(exp_out2, out)
