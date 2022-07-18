from typing import Union

import numpy as np


def get_utils_irls(y, eta, family):
    if family == "gaussian":
        g = eta
        gprime = eta
        variance = 1
    elif family == "poisson":
        aux = np.exp(eta)
        g = aux
        gprime = aux
        variance = aux
    elif family == "binomial":
        aux = np.exp(eta)
        g = aux / (1 + aux)
        gprime = g / (1 + aux)
        variance = gprime
    else:
        raise ValueError("Family {family} is not implemented.")

    return {"z": eta + (y - g) / gprime, "W": np.square(gprime) / variance}


def irls(
    X: np.ndarray,
    y: np.ndarray,
    threshold: Union[int, float] = 1e-8,
    max_iter: int = 100,
    family: str = "gaussian",
    verbose: bool = False,
) -> np.ndarray:

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "The design matrix `X` and the response vector `y` have different number of rows."
        )

    beta = np.zeros((X.shape[1],), dtype=int)
    for iter in range(max_iter):
        beta_old = beta.copy()
        eta = np.dot(X, beta)
        utils_irls = get_utils_irls(eta=eta, y=y, family=family)
        beta = np.linalg.solve(
            X.T @ utils_irls["W"] @ X, X.T @ utils_irls["W"] @ utils_irls["Z"]
        )
        if np.linalg.norm(beta - beta_old) < threshold:
            break
    if verbose:
        if max_iter > iter:
            print(f"Algorithm has converged after iteration {iter}.")
        else:
            print("Algorithm has not converged.")

    return beta
