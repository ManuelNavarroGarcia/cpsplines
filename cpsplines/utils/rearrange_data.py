import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class RearrangingError(Exception):
    pass


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

    return (
        pd.DataFrame(
            np.c_[
                np.stack(np.meshgrid(*x, indexing="ij"), -1).reshape(-1, len(x)),
                y.flatten(),
            ],
            columns=x_cols + [y_col],
        )
        .dropna()
        .reset_index(drop=True)
    )


def scatter_to_grid(
    data: pd.DataFrame, y_col: str
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Given a DataFrame containing the value of the covariates and the target
    variable, generates a list of arrays containing the value of the covariates
    and a grid of target values with these coordinates.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y_col : str
        Column name of the target variable.

    Returns
    -------
    Tuple[List[np.ndarray], np.ndarray]
        A tuple containing the coordinates as the first element and the array
        of response values a the second element.

    Raises
    ------
    ValueError
        If multiple responses are found for the same coordinates.

    References
    ----------
    - https://stackoverflow.com/a/35049899
    """

    # Get regressor columns
    x_cols = data.columns.drop(y_col).tolist()
    if len(x_cols) > 1:
        # The indexes are now the coordinates, and the values the response sample
        df = data.groupby(x_cols).mean()
        if len(data) != len(df):
            raise RearrangingError(
                "Multiple responses for the same coordinates. Data cannot be rearranged into a grid."
            )
        # Create an empty array of NaN of the right dimensions
        arr = np.full(tuple(map(len, df.index.levels)), np.nan)
        # Fill it using Numpy's advanced indexing
        arr[tuple(df.index.codes)] = df.values.flat
        # Get the first unique values for each axis
        x = [np.unique(row) for row in np.array(df.index.tolist()).T]
        y = arr.copy()
    else:
        x = [data[x_cols].values.flatten()]
        y = data[y_col].values
    return x, y
