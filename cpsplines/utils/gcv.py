from typing import Dict, Iterable, Tuple, Union

import numpy as np
import statsmodels.genmod.families.family

from cpsplines.utils.fast_kron import penalization_term, weighted_double_kronecker
from cpsplines.utils.irls import fit_irls


def quadratic_term(
    sp: Iterable[Union[int, float]],
    obj_matrices: Dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
    family: statsmodels.genmod.families.family,
    data_arrangement: str,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Computes the quadratic terms involved in the objective function, i.e., the
    bases related term and the penalty term.

    Parameters
    ----------
    sp : Iterable[Union[int, float]]
        The smoothing paramater vector.
    obj_matrices : Dict[str, Union[np.ndarray, Iterable[np.ndarray]]]
        A dictionary containing the necessary arrays (the basis matrices, the
        penalty matrices and the response variable sample) used to compute the
        quadratic terms in the objective function.
    family : statsmodels.genmod.families.family, optional
        The specific exponential family distribution where the response variable
        belongs to, by default "gaussian".
    data_arrangement : str
            The way the data is arranged.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the bases related term and the penalty term (in this
        order).

    Raises
    ------
    ValueError
        If `data_arrangement` is not "gridded" or "scattered".
    """

    if data_arrangement not in ("gridded", "scattered"):
        raise ValueError(f"Invalid `data_arrangement`: {data_arrangement}.")

    mu = family.starting_mu(obj_matrices["y"])
    W = family.weights(mu)
    bases_term = weighted_double_kronecker(
        matrices=obj_matrices["B"],
        W=W if data_arrangement == "gridded" else np.diag(W),
    )
    penalty_list = penalization_term(matrices=obj_matrices["D_mul"])
    penalty_term = np.add.reduce([np.multiply(s, P) for P, s in zip(penalty_list, sp)])
    return (bases_term, penalty_term)


def GCV(
    sp: Iterable[Union[int, float]],
    obj_matrices: Dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
    family: statsmodels.genmod.families.family,
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
    family : statsmodels.genmod.families.family
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

    bases_term, penalty_term = quadratic_term(
        sp=sp,
        obj_matrices=obj_matrices,
        family=family,
        data_arrangement=data_arrangement,
    )
    y_hat = fit_irls(
        obj_matrices=obj_matrices,
        family=family,
        penalty_term=penalty_term,
        data_arrangement=data_arrangement,
    )
    # Return the GCV value, which is n * RSS / (n - tr(H))**2, where RSS is the
    # residual sum of squares, n is the product of the dimensions of y and H is
    # the hat matrix of the unconstrained problem
    return (
        np.prod(obj_matrices["y"].shape)
        * np.square(np.linalg.norm((obj_matrices["y"] - y_hat)))
    ) / np.square(
        np.prod(obj_matrices["y"].shape)
        - np.trace(np.linalg.solve(bases_term + penalty_term, bases_term))
    )
