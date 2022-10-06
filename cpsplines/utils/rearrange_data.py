import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def grid_to_scatter(
    x: List[np.ndarray],
    y: np.ndarray,
    x_cols: Optional[List[str]] = None,
    y_col: str = "y",
) -> pd.DataFrame:

    """
    Given a list of arrays containing the value of the covariates and a grid
    of target values with these coordinates, generates a DataFrame with the same
    data but in scatter format.

    Parameters
    ----------
    x : List[np.ndarray]
        The values of the covariates.
    y : np.ndarray
        The values of the target variable.
    x_cols : Optional[List[str]], optional
        Name of the covariate columns. If None, the names "x0", "x1", ... are
        assigned. By default None.
    ycol : str, optional
        Name of the target column, by default "y".

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the data in scatter format.

    Raises
    ------
    ValueError
        If the number of data arrays and the number of their column names
        inputted differ.
    """

    if x_cols is None:
        logging.info("`x_cols` was not inputted. Assigning their default name...")
        x_cols = [f"x{i}" for i, _ in enumerate(x)]
    else:
        if len({len(i) for i in [x, x_cols]}) != 1:
            raise ValueError("The lengths of `x`, `x_cols` must agree.")

    return pd.DataFrame(
        np.c_[
            np.stack(np.meshgrid(*x, indexing="ij"), -1).reshape(-1, len(x)),
            y.flatten(),
        ],
        columns=x_cols + [y_col],
    )


