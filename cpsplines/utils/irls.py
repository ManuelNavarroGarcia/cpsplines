from typing import Union

import numpy as np
import statsmodels.regression._tools as reg_tools
from statsmodels.genmod.families.family import Gaussian, Poisson


def fit_irls(
    X: np.ndarray,
    y: np.ndarray,
    threshold: Union[int, float] = 1e-8,
    maxiter: int = 100,
    family: str = "gaussian",
    verbose: bool = False,
) -> np.ndarray:

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "The design matrix `X` and the response vector `y` have different number of rows."
        )

    if family == "gaussian":
        family = Gaussian()
    elif family == "poisson":
        family = Poisson()
    else:
        raise ValueError(f"Family {family} is not implemented.")

    beta_old = np.zeros(X.shape[1])
    mu = family.starting_mu(y)
    lin_pred = family.predict(mu)

    for iter in range(maxiter):
        weights = family.weights(mu)
        z = lin_pred + family.link.deriv(mu) * (y - mu)

        weighted_ls = reg_tools._MinimalWLS(
            z, X, weights, check_endog=True, check_weights=True
        )
        beta = weighted_ls.fit(method="lstsq")["params"]
        lin_pred = np.dot(X, beta)
        mu = family.fitted(lin_pred)
        if np.linalg.norm(beta - beta_old) < threshold:
            if verbose:
                print(f"Algorithm has converged after {iter} iterations.")
            break
        beta_old = beta.copy()
    return beta
