from collections.abc import Iterable

import numpy as np
from statsmodels.genmod.families.family import Family

from src.cpsplines.utils.fast_kron import penalization_term, weighted_double_kronecker
from src.cpsplines.utils.irls import fit_irls


def GCV(
    sp: Iterable[int | float],
    obj_matrices: dict[str, np.ndarray | Iterable[np.ndarray]],
    family: Family,
    data_arrangement: str,
) -> float:
    """
    Computes the Generalized Cross Validation (Golub et al., 1979).

    Parameters
    ----------
    sp : Iterable[Union[int, float]]
        The smoothing paramater vector.
    obj_matrices : Dict[str, Union[np.ndarray, Iterable[np.ndarray]]]
        A dictionary containing the necessary arrays (the basis matrices, the
        penalty matrices and the response variable sample) used to compute the
        quadratic terms in the objective function.
    family : Family
        The specific exponential family distribution where the response variable
        belongs to.
    data_arrangement : str
        The way the data is arranged.

    References
    ----------
    - Golub, G. H., Heath, M., & Wahba, G. (1979). Generalized cross-validation
      as a method for choosing a good ridge parameter. Technometrics, 21(2),
      215-223.

    Returns
    -------
    float
        The GCV value.
    """

    penalty_list = penalization_term(matrices=obj_matrices["D_mul"])
    penalty_term = np.add.reduce([np.multiply(s, P) for P, s in zip(penalty_list, sp)])

    y_hat = fit_irls(
        obj_matrices=obj_matrices,
        family=family,
        penalty_term=penalty_term,
        data_arrangement=data_arrangement,
    )
    bases_term = weighted_double_kronecker(
        matrices=obj_matrices["B"],
        W=family.weights(family.starting_mu(y_hat)),
        data_arrangement=data_arrangement,
    )

    n = np.prod(obj_matrices["y"].shape)
    # Return the GCV value, which is n * Dev / (n - tr(H))**2, where Dev is the
    # deviance, `n` is the product of the dimensions of `y` and `H` is the hat matrix of
    # the unconstrained problem
    return (n * family.deviance(obj_matrices["y"], y_hat)) / np.square(
        n - np.trace(np.linalg.solve(bases_term + penalty_term, bases_term))
    )
