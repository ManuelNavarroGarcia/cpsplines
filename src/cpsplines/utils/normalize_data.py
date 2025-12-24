from typing import Iterable

import numpy as np


class DataNormalizer:
    """
    Transform a numeric array by scaling it to a given range. To scale the array
    between an arbitrary set of values [a, b], the transformation used is

    x' = a + (x-min(x)) * (b - a) / (max(x) - min(x))

    To return to the original scale, the inverse transformation is given by

    x = min(x) + (x' - a) * (max(x) - min(x)) / (b - a)

    Parameters
    ----------
    feature_range : Iterable[Union[int, float]], optional
        The desired range of transformed data. By default, (0, 1).

    Attributes
    ----------
    y_min : Union[int, float]
        The minimum value of the numeric array.
    y_range : Union[int, float]
        The range of the numeric array, i.e., the difference between the maximum
        and the minimum value.

    """

    def __init__(self, feature_range: Iterable[int | float] = (0, 1)):
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

    def transform(
        self, y: int | float | np.ndarray, derivative: bool = False
    ) -> np.ndarray:
        """
        Transform the numeric array according to the desired range.

        Parameters
        ----------
        y : Union[int, float, np.ndarray]
            The numeric array.
        derivative : bool
            If True, the input array to be transformed is a derivative with
            respect the array used to fit, by default False.

        Returns
        -------
        np.ndarray
            The scaled numeric array.
        """

        if derivative:
            out = y * (self.feature_range[1] - self.feature_range[0]) / self.y_range
        else:
            out = (
                self.feature_range[0]
                + (y - self.y_min)
                * (self.feature_range[1] - self.feature_range[0])
                / self.y_range
            )
        return out

    def inverse_transform(
        self, y: int | float | np.ndarray, derivative: bool = False
    ) -> np.ndarray:
        """
        Transform the scaled numeric array into the original scale.

        Parameters
        ----------
        y : Union[int, float, np.ndarray]
            The scaled numeric array.
        derivative : bool
            If True, the input array to be transformed is a derivative with
            respect the array used to fit, by default False.

        Returns
        -------
        np.ndarray
            The numeric array.
        """

        if derivative:
            out = self.y_range * y / (self.feature_range[1] - self.feature_range[0])
        else:
            out = self.y_min + self.y_range * (y - self.feature_range[0]) / (
                self.feature_range[1] - self.feature_range[0]
            )
        return out
