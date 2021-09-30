from functools import reduce
import numpy as np
import pytest
from scipy.linalg import block_diag

from template.utils.fast_kron import (
    fast_kronecker_product,
    kronecker_matrix_by_identity,
)
from template.utils.timer import timer


@pytest.mark.parametrize(
    "dim_list",
    [
        ([[4, 7], [5, 6], [8, 3], [1, 9], [10, 2]]),
    ],
)
def test_fast_kronecker(dim_list):
    random_mat = []
    for dim in dim_list:
        random_mat.append(np.random.rand(*dim))

    with timer():
        print("Using np.kron")
        exp_out = reduce(np.kron, random_mat)

    with timer():
        print("Using broadcasting")
        out = reduce(fast_kronecker_product, random_mat)

    np.testing.assert_allclose(exp_out, out)


@pytest.mark.parametrize(
    "A, n",
    [
        (np.random.rand(13, 17), 19),
        (np.random.rand(53, 59), 62),
    ],
)
def test_mat_kron_identity(A, n):

    with timer():
        print("Using fast kronecker product by identity")
        out = kronecker_matrix_by_identity(A, n)

    with timer():
        print("Using np.kron")
        exp_out = np.kron(A, np.eye(n))

    with timer():
        print("Using broadcasting")
        exp_out2 = fast_kronecker_product(A=A, B=np.eye(n))

    np.testing.assert_allclose(exp_out, out)
    np.testing.assert_allclose(exp_out2, out)


@pytest.mark.parametrize(
    "n, A",
    [
        (19, np.random.rand(13, 17)),
        (62, np.random.rand(53, 59)),
    ],
)
def test_identity_kron_mat(n, A):

    with timer():
        print("Using fast kronecker product identity times matrix")
        out = block_diag(*[A] * n)

    with timer():
        print("Using np.kron")
        exp_out = np.kron(np.eye(n), A)

    with timer():
        print("Using broadcasting")
        exp_out2 = fast_kronecker_product(A=np.eye(n), B=A)

    np.testing.assert_allclose(exp_out, out)
    np.testing.assert_allclose(exp_out2, out)
