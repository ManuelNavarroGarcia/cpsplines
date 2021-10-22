from functools import reduce

import numpy as np
import pytest
from cpsplines.utils.fast_kron import matrix_by_tensor_product
from cpsplines.utils.timer import timer


# Test the efficiency of folding and unfolding matrices compared to brute force
# approximation
@pytest.mark.parametrize(
    "common_dim, different_dim",
    [
        ([200], [23]),
        ([150, 175], [43, 67]),
        ([75, 53, 30], [7, 8, 9]),
    ],
)
def test_matrices_by_tensor(common_dim, different_dim):
    random_tensor = np.random.rand(*common_dim)
    random_mat = [np.random.rand(c, d) for c, d in zip(common_dim, different_dim)]

    with timer(tag="Using np.kron"):
        exp_out = (random_tensor.flatten() @ reduce(np.kron, random_mat)).reshape(
            *different_dim
        )

    with timer(tag="Using folding and unfolding"):
        out = matrix_by_tensor_product(
            matrices=[A.T for A in random_mat], T=random_tensor
        )

    np.testing.assert_allclose(exp_out, out)
