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

    """
    Transform a numeric array by scaling it to a given range. To scale the array
    between an arbitrary set of values [a, b], the transformation we use is

    x' = a + (x-min(x)) * (b - a) / (max(x) - min(x))

    To return to the original scale, the inverse transformation is given by

    x = min(x) + (x' - a) * (max(x) - min(x)) / (b - a)

    Parameters
    ----------
    feature_range : Iterable[Union[int, float]], optional
        The desired range of transformed data. By default, [0,1].

    Attributes
    ----------
    y_min : Union[int, float]
        The minimum value of the numeric array.
    y_range : Union[int, float]
        The range of the numeric array, i.e., the difference between the maximum
        and the minimum value.

    """

    def __init__(self, feature_range: Iterable[Union[int, float]] = (0, 1)):
        self.feature_range = feature_range

    def fit(self, y: np.ndarray) -> object:

        """
        Compute the minimum and the range of the numeric array which is to be
        scaled.

        Parameters
        ----------
        y : np.ndarray
            The numeric array.

        Returns
        -------
        self : object
            Fitted scaler.

        Raises
        ------
        ValueError
            If the first value of the input range is greater or equal to the
            second value.
        ValueError
            If the input range does not contain two values.
        """

        if self.feature_range[0] >= self.feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum."
            )
        if len(self.feature_range) != 2:
            raise ValueError("Range must consist of two elements.")

        self.y_min = np.min(y)
        self.y_range = np.ptp(y)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:

        """
        Transform the numeric array according to the desired range.

        Parameters
        ----------
        y : np.ndarray
            The numeric array.

        Returns
        -------
        np.ndarray
            The scaled numeric array.
        """

        return (
            self.feature_range[0]
            + (y - self.y_min)
            * (self.feature_range[1] - self.feature_range[0])
            / self.y_range
        )

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:

        """
        Transform the scaled numeric array into the original scale.

        Parameters
        ----------
        y : np.ndarray
            The scaled numeric array.

        Returns
        -------
        np.ndarray
            The numeric array.
        """
        return self.y_min + self.y_range * (y - self.feature_range[0]) / (
            self.feature_range[1] - self.feature_range[0]
        )
