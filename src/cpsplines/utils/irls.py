from collections.abc import Iterable
from functools import reduce

import numpy as np
from statsmodels.genmod.families.family import Family

from src.cpsplines.utils.box_product import box_product
from src.cpsplines.utils.fast_kron import (
    matrix_by_tensor_product,
    weighted_double_kronecker,
)


def fit_irls(
    obj_matrices: dict[str, np.ndarray | Iterable[np.ndarray]],
    penalty_term: np.ndarray,
    family: Family,
    data_arrangement: str,
    threshold: int | float = 1e-8,
    maxiter: int = 100,
    verbose: bool = False,
) -> np.ndarray:
    """Given the basis and the penalty matrices of the model, provides the
    fitting of a Generalized Additive Model (GAM) through Iteratively
    Re-weighted Least Squares estimation (IRLS) (Nelder and Wedderburn, 1972).

    Parameters
    ----------
    obj_matrices : Dict[str, Union[np.ndarray, Iterable[np.ndarray]]]
        A dictionary containing the necessary arrays (the basis matrices and
        the response variable sample) used in the algorithm.
    penalty_term : np.ndarray
        The penalty term of the model.
    family : Family
        The specific exponential family distribution where the response variable
        belongs to.
    data_arrangement : str
        The way the data is arranged.
    threshold : Union[int, float], optional
        An optional quantity to use as the convergence criterion for the change
        in L2-norm of the fitted coefficients, by default 1e-8.
    maxiter : int, optional
        Maximum number of iterations carried out by the IRLS algorithm, by
        default 100.
    verbose : bool, optional
        Print information about the process, by default False.

    References
    ----------
    - Nelder, J. and Wedderburn, R. (1972). Generalized linear models. Journal
      of the Royal Statistical Society, Series A, 135, 370-385.

    Returns
    -------
    np.ndarray
        The fitted values for the response variable.

    Raises
    ------
    ValueError
        If `data_arrangement` is not "gridded" or "scattered".
    """

    if data_arrangement not in ("gridded", "scattered"):
        raise ValueError(f"Invalid `data_arrangement`: {data_arrangement}.")
    # Obtain an initial value of the fitting coefficients
    theta_old = np.zeros(tuple([mat.shape[1] for mat in obj_matrices["B"]]))
    # Use this initial value to estimate initial values for `mu` (mean of the
    # exponential family) and `eta` (transformed mu through the link function)
    mu = family.starting_mu(obj_matrices["y"])
    eta = family.predict(mu)

    for iter in range(maxiter):
        # Get the weights and the modified dependent variable
        W = family.weights(mu)
        Z = eta + family.link.deriv(mu) * (obj_matrices["y"] - mu)

        # With this modified dependent variable, update the coefficients
        bases_term = weighted_double_kronecker(matrices=obj_matrices["B"], W=W, data_arrangement=data_arrangement)

        T = np.multiply(W, Z)
        if data_arrangement == "gridded":
            F = matrix_by_tensor_product([B.T for B in obj_matrices["B"]], T).flatten()
        else:
            F = np.dot(reduce(box_product, obj_matrices["B"]).T, T.flatten())
        theta = np.reshape(
            np.linalg.solve(
                bases_term + penalty_term,
                F,
            ),
            tuple([mat.shape[1] for mat in obj_matrices["B"]]),
        )

        # Update `eta` and `mu`
        if data_arrangement == "gridded":
            eta = matrix_by_tensor_product([mat for mat in obj_matrices["B"]], theta)
        else:
            eta = np.dot(reduce(box_product, obj_matrices["B"]), theta.flatten())
        mu = family.fitted(np.clip(eta, a_min=-500, a_max=500))
        # Check convergence
        if np.linalg.norm(theta - theta_old) < threshold:
            if verbose:
                print(f"Algorithm has converged after {iter} iterations.")
            break
        # Update the initial value of the coefficients
        theta_old = theta.copy()
    return mu
