import numpy as np
import pytest
import tensorly as tl

from template.utils.fast_kron import kron_tens_prod
from template.utils.timer import timer


def tensor_prod_brute_force(mat_list, tensor):
    if len(mat_list) == 1:
        return mat_list[0] @ tensor
    elif len(mat_list) == 2:
        return mat_list[0] @ tensor @ mat_list[1].T
    elif len(mat_list) == 3:
        dim = tuple([mat.shape[0] for mat in mat_list])
        return tl.fold(
            mat_list[0] @ tl.unfold(tensor, 0) @ np.kron(mat_list[1], mat_list[2]).T,
            0,
            dim,
        )
    else:
        print("This test method is not implemented for dimensions greater than 3.")
        return None


@pytest.mark.parametrize(
    "common_dim, different_dim",
    [
        ([200], [23]),
        ([200, 300], [43, 67]),
        ([75, 150, 225], [20, 30, 40]),
    ],
)
def test_tensor_product(common_dim, different_dim):
    random_tensor = np.random.rand(*common_dim)
    random_mat = []
    for c, d in zip(common_dim, different_dim):
        random_mat.append(np.random.rand(d, c))

    with timer():
        print("Using np.kron")
        exp_out = tensor_prod_brute_force(mat_list=random_mat, tensor=random_tensor)

    with timer():
        print("Using folding and unfolding")
        out = kron_tens_prod(mat_list=random_mat, T=random_tensor)

    np.testing.assert_allclose(exp_out, out)
