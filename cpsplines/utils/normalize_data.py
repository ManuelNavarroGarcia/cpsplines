from typing import Iterable, Union

import numpy as np
import pandas as pd


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


class DataNormalizer:
    def __init__(self, feature_range: Iterable[Union[int, float]] = (0, 1)):
        self.feature_range = feature_range

    def fit(self, y: np.ndarray):
        if self.feature_range[0] >= self.feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum."
            )
        if len(self.feature_range) != 2:
            raise ValueError("Range must consist of two elements.")

        self.y_min = np.min(y)
        self.y_range = np.ptp(y)
        return self

    def transform(self, y: np.ndarray):
        return (
            self.feature_range[0]
            + (y - self.y_min)
            * (self.feature_range[1] - self.feature_range[0])
            / self.y_range
        )

    def inverse_transform(self, y: np.ndarray):
        return self.y_min + self.y_range * (y - self.feature_range[0]) / (
            self.feature_range[1] - self.feature_range[0]
        )
