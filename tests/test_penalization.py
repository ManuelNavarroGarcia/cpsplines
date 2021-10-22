from typing import Iterable, Union

import numpy as np
import pytest
from cpsplines.utils.fast_kron import penalization_term
from cpsplines.utils.timer import timer


def penalty_brute_force(
    D_mul: Iterable[np.ndarray], sp: Iterable[Union[int, float]]
) -> np.ndarray:

    """
    Computes the penalization term using the exact definition including all
    the matrices involved. It only supports three or less dimensions.

    Parameters
    ----------
    D_mul : Iterable[np.ndarray]
        The univariate penalty matrices. Must have length less or equal than 3.
    sp : Iterable[Union[int, float]]
        The smoothing parameter vector. Must have length less or equal than 3.

    Returns
    -------
    np.ndarray
        The penalization term.

    Raises
    ------
    ValueError
        If the number of input smoothing paramters and penalty matrices differ.
    ValueError
        If the number of input penalty matrices is greater than 3.
    """

    if len(sp) != len(D_mul):
        raise ValueError(
            "Number of smoothing parameters must be equal to number of D elements."
        )
    if len(D_mul) > 3:
        raise ValueError("Not implemented for more than three dimensions.")

    dim = [P.shape[0] for P in D_mul]
    if len(dim) == 1:
        P = np.multiply(sp[0], D_mul[0])
    elif len(dim) == 2:
        P = np.kron(np.multiply(sp[0], D_mul[0]), np.eye(dim[1])) + np.kron(
            np.eye(dim[0]), np.multiply(sp[1], D_mul[1])
        )
    else:
        P = (
            np.kron(
                np.multiply(sp[0], D_mul[0]),
                np.eye(dim[1] * dim[2]),
            )
            + np.kron(
                np.eye(dim[0]),
                np.kron(np.multiply(sp[1], D_mul[1]), np.eye(dim[2])),
            )
            + np.kron(np.eye(dim[0] * dim[1]), np.multiply(sp[2], D_mul[2]))
        )
    return P


# Test the computation of the penalization term in the objective function for a
# variety of number of dimensions
@pytest.mark.parametrize(
    "dim, sp",
    [
        ([41], [17]),
        ([37, 5], [19, 29]),
        ([23, 11, 17], [31, 67, 93]),
    ],
)
def test_penalty_matrix(dim, sp):
    matrices = [np.random.rand(d, d) for d in dim]

    with timer(tag="Using brute force computation"):
        penalty_out = penalty_brute_force(D_mul=matrices, sp=sp)

    with timer(tag="Using fast kronecker products"):
        penalty = np.add.reduce(
            [
                np.multiply(s, A)
                for s, A in zip(sp, penalization_term(matrices=matrices))
            ]
        )
    np.testing.assert_allclose(penalty_out, penalty)
