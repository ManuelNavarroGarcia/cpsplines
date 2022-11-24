from typing import Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cpsplines.fittings.grid_cpsplines import CPsplines
from cpsplines.graphics.plot_utils import granulate_prediction_range


class CurvesDisplay:

    """Fitted curve visualization.

    Parameters
    ----------
    X : Union[pd.Series, pd.DataFrame]
        The abscissa of the points used in the fitting procedure.
    y_true : pd.Series
        The ordinate of the points used in the fitting procedure
    y_pred : np.ndarray
        The predicted values at `X`.

    Attributes
    ----------
    ax_ : matplotlib Axes
        Axes with fitted curve.
    figure_ : matplotlib Figure
        Figure containing the curve.
    """

    def __init__(
        self, X: Union[pd.Series, pd.DataFrame], y_true: pd.Series, y_pred: np.ndarray
    ):
        self.X = X
        self.y_true = y_true
        self.y_pred = y_pred

    def plot(
        self,
        ax: Optional[plt.axes] = None,
        **kwargs,
    ):
        """Plot visualization. Extra keyword arguments will be passed to
        matplotlib's `plot`.

        Parameters
        ----------
        ax : Optional[plt.axes], optional
           Axes object to plot on. If `None`, a new figure and axes is created.
           By default, None.

        Returns
        -------
        display : :class:`~cpsplines.graphics.CurvesDisplay`
            Object that stores computed values.
        """

        if ax is None:
            _, ax = plt.subplots()

        _ = ax.plot(self.X, self.y_pred, **kwargs)

        self.ax_ = ax
        self.figure_ = ax.figure

        return self

    @classmethod
    def from_estimator(
        cls,
        estimator: CPsplines,
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
        """Create a curve fitting display from an estimator.

        Parameters
        ----------
        estimator : CPsplines
            A fitted `CPsplines` object.
        X : Union[pd.Series, pd.DataFrame]
            The abscissa of the points used in the fitting procedure.
        y : pd.Series
            The ordinate of the points used in the fitting procedure, which
            are to be plotted as solid dots.
        knot_positions : bool, optional
           If True, the positions where the inner knots are located are marked
           as grey vertical lines. By default, False.
        constant_constraints : bool, optional
            If True, horizontal lines at the threshold of the zero-order
            derivative constraints are plotted with red dashed lines. By
            default, False.
        prediction_step : Iterable[Union[int, float]], optional
            The step used to produce equidistant extra points at the prediction
            regions so the graph of the curves seems smoother (if this is not
            applied, the only points at the prediction points are the knots).
            The first element of the iterable is the step on the backwards
            prediction and the second element corresponds to the step on the
            forward prediction. By default, (0.5, 0.5).
        ax : Optional[plt.axes], optional
           Axes object to plot on. If `None`, a new figure and axes is created.
           By default, None.
        col_pt : Optional[Iterable[str]], optional
            The colour used to plot the points. If None, it is default color of
            matplotlib for scatter plots. By default, None.
        alpha : Union[int, float], optional
            The transparency level of the points, by default 0.25.
        figsize : Tuple[Union[int, float]], optional
            The size of the figure, by default (15, 10).

        Returns
        -------
        display : :class:`~cpsplines.graphics.CurvesDisplay`
            Object that stores computed values.
        """
        bsp = estimator.bspline_bases[0]

        # Generate extra points at the prediction regions with the
        # `prediction_step` parameter so the curve is plotted smoother
        x_left, x_right = granulate_prediction_range(
            bspline_bases=[bsp], prediction_step=[prediction_step]
        )
        # Append NaNs at the prediction regions
        y = pd.Series(
            np.concatenate(
                [
                    [np.nan] * len(x_left[0]),
                    y.values[np.argsort(X.values)],
                    [np.nan] * len(x_right[0]),
                ]
            )
        )
        # Append extra points at the prediction regions
        X = pd.Series(np.concatenate([x_left[0], np.sort(X.values), x_right[0]]))
        # Generate the predictions
        y_pred = estimator.predict(X.sort_values())

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        _ = ax.figure.set_size_inches(*figsize)

        # If it is required, plot the position of the knots
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

        # If the prediction region is not empty, plot vertical dashed lines at
        # the extremes of the fitting region
        if bsp.int_back > 0:
            _ = ax.axvline(bsp.xsample.min(), linewidth=1.0, linestyle="--", **kwargs)
        if bsp.int_forw > 0:
            _ = ax.axvline(bsp.xsample.max(), linewidth=1.0, linestyle="--", **kwargs)

        _ = ax.scatter(x=X, y=y, c=col_pt, alpha=alpha)

        viz = CurvesDisplay(X, y, y_pred)

        return viz.plot(ax=ax, **kwargs)
