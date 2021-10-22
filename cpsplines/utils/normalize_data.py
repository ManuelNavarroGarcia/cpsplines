import numpy as np
import pandas as pd
from typing import Union


def normalize_data(x: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:

    """
    Normalize an array between 0 and 1.

    Parameters
    ----------
    x : Union[np.ndarray, pd.Series]
        The data to be normalized.

    Returns
    -------
    Union[np.ndarray, pd.Series]
        The data normalized.
    """

    return (x - np.min(x)) / np.ptp(x)
