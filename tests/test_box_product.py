from functools import reduce

import numpy as np
import pytest
from cpsplines.utils.box_product import box_product
from cpsplines.utils.timer import timer


def box_product_kron(A, B):
    return np.multiply(
        np.kron(A, np.ones(B.shape[1]).T), np.kron(np.ones(A.shape[1]).T, B)
    )


@pytest.mark.parametrize(
    "dim",
    [((2017, 23), (2017, 17)), ((997, 19), (997, 29), (997, 31))],
)
def test_box_product(dim):
    matrices = [np.random.rand(*d) for d in dim]

    with timer(tag="Using definition involving Kronecker products"):
        exp_out = reduce(box_product_kron, matrices)

    with timer(tag="Using np.tile and np.repeat"):
        out = reduce(box_product, matrices)

    np.testing.assert_allclose(exp_out, out)
