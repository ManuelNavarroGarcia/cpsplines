import numpy as np
import pytest

from template.utils.fast_kron import penalization_term
from template.utils.timer import timer


def penalization_brute_force(matrices, sp):
    dim_list = [mat.shape[0] for mat in matrices]
    if len(dim_list) == 1:
        return np.multiply(sp[0], matrices[0])
    elif len(dim_list) == 2:
        return np.kron(np.multiply(sp[0], matrices[0]), np.eye(dim_list[1])) + np.kron(
            np.eye(dim_list[0]), np.multiply(sp[1], matrices[1])
        )
    elif len(dim_list) == 3:
        return (
            np.kron(
                np.multiply(sp[0], matrices[0]),
                np.eye(dim_list[1] * dim_list[2]),
            )
            + np.kron(
                np.eye(dim_list[0]),
                np.kron(np.multiply(sp[1], matrices[1]), np.eye(dim_list[2])),
            )
            + np.kron(
                np.eye(dim_list[0] * dim_list[1]), np.multiply(sp[2], matrices[2])
            )
        )
    else:
        print("This test method is not implemented for dimensions greater than 3.")
        return None


@pytest.mark.parametrize(
    "dim_list, sp_list",
    [
        ([41], [17]),
        ([37, 5], [19, 29]),
        ([23, 11, 17], [31, 67, 93]),
    ],
)
def test_penalty_matrix(dim_list, sp_list):
    matrices = []
    for dim in dim_list:
        matrices.append(np.random.rand(dim, dim))

    with timer():
        print("Using brute force computation")
        penalty_out = penalization_brute_force(matrices=matrices, sp=sp_list)

    with timer():
        print("Using fast kronecker products")
        penalty = np.add.reduce(
            [
                np.multiply(sp, mat)
                for sp, mat in zip(sp_list, penalization_term(matrices=matrices))
            ]
        )
    np.testing.assert_allclose(penalty_out, penalty)
