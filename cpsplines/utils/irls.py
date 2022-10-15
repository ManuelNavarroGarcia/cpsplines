from typing import Dict, Iterable, Union

import numpy as np
import statsmodels.genmod.families.family
from cpsplines.utils.fast_kron import (
    matrix_by_tensor_product,
    weighted_double_kronecker,
)


def fit_irls(
    obj_matrices: Dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
    penalty_term: np.ndarray,
    family: statsmodels.genmod.families.family,
    data_arrangement: str,
    threshold: Union[int, float] = 1e-8,
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
    family : statsmodels.genmod.families.family
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
    """

    # Obtain an initial value of the fitting coefficients
    theta_old = np.zeros(tuple([mat.shape[1] for mat in obj_matrices["B_w"]]))
    # Use this initial value to estimate initial values for `mu` (mean of the
    # exponential family) and `eta` (transformed mu through the link function)
    mu = family.starting_mu(obj_matrices["y"])
    eta = family.predict(mu)

    for iter in range(maxiter):
        # Get the weights and the modified dependent variable
        W = family.weights(mu)
        Z = eta + family.link.deriv(mu) * (obj_matrices["y"] - mu)

        # With this modified dependent variable, update the coefficients
        bases_term = weighted_double_kronecker(
            matrices=obj_matrices["B_w"],
            W=W if data_arrangement == "gridded" else np.diag(W),
        )

        T = np.multiply(W, Z)
        theta = np.reshape(
            np.linalg.solve(
                bases_term + penalty_term,
                matrix_by_tensor_product(
                    [B.T for B in obj_matrices["B_w"]],
                    T if data_arrangement == "gridded" else np.diag(T),
                ).flatten(),
            ),
            tuple([mat.shape[1] for mat in obj_matrices["B_w"]]),
        )

        # Update `eta` and `mu`
        if data_arrangement == "gridded":
            eta = matrix_by_tensor_product(
                [mat for mat in obj_matrices["B_w"]],
                theta,
            )
        else:
            eta = np.dot(reduce(box_product, obj_matrices["B_w"]), theta.flatten())
        mu = family.fitted(eta)
        # Check convergence
        if np.linalg.norm(theta - theta_old) < threshold:
            if verbose:
                print(f"Algorithm has converged after {iter} iterations.")
            break
        # Update the initial value of the coefficients
        theta_old = theta.copy()
    return mu
