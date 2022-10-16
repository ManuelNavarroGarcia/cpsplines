from typing import Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cpsplines.fittings.grid_cpsplines import GridCPsplines
from cpsplines.graphics.plot_utils import granulate_prediction_range


class CurvesDisplay:
    def __init__(
        self, X: Union[pd.Series, pd.DataFrame], y_true: pd.Series, y_pred: pd.Series
    ):
        self.X = X
        self.y_true = y_true
        self.y_pred = y_pred

    def plot(
        self,
        ax: Optional[plt.axes] = None,
        **kwargs,
    ):

        if ax is None:
            _, ax = plt.subplots()

        _ = ax.plot(self.X, self.y_pred, **kwargs)

        self.ax_ = ax
        self.figure_ = ax.figure

        return self

    @classmethod
    def from_estimator(
        cls,
        estimator: GridCPsplines,
        X: Union[pd.Series, pd.DataFrame],
        y: pd.Series,
        knot_positions: bool = False,
        constant_constraints: bool = False,
        prediction_step: Iterable[Union[int, float]] = (0.5, 0.5),
        ax: Optional[plt.axes] = None,
        col_pt: Optional[Iterable[str]] = None,
        alpha: Union[int, float] = 0.25,
        figsize: Tuple[Union[int, float]] = (15, 10),
        **kwargs,
    ):
        bsp = estimator.bspline_bases[0]

        x_left, x_right = granulate_prediction_range(
            bspline_bases=[bsp], prediction_step=[prediction_step]
        )
        y = pd.Series(
            np.concatenate(
                [
                    [np.nan] * len(x_left[0]),
                    y.values[np.argsort(X.values)],
                    [np.nan] * len(x_right[0]),
                ]
            )
        )
        X = pd.Series(np.concatenate([x_left[0], np.sort(X.values), x_right[0]]))
        y_pred = estimator.predict(X.sort_values())

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        _ = ax.figure.set_size_inches(*figsize)

        if knot_positions:
            for knot in bsp.knots[bsp.deg : -bsp.deg]:
                _ = ax.axvline(knot, color="grey", alpha=0.25)

        # If it is required, threshold of the zero-order derivative constraints
        if constant_constraints:
            if estimator.int_constraints:
                if 0 in estimator.int_constraints[0].keys():
                    for value in estimator.int_constraints[0][0].values():
                        _ = ax.axhline(
                            value,
                            color="red",
                            linewidth=1.0,
                            linestyle="--",
                        )

        if bsp.int_back > 0:
            _ = ax.axvline(bsp.xsample.min(), linewidth=1.0, linestyle="--", **kwargs)
        if bsp.int_forw > 0:
            _ = ax.axvline(bsp.xsample.max(), linewidth=1.0, linestyle="--", **kwargs)

        _ = ax.scatter(x=X, y=y, c=col_pt, alpha=alpha)

        viz = CurvesDisplay(X, y, y_pred)

        return viz.plot(ax=ax, **kwargs)
