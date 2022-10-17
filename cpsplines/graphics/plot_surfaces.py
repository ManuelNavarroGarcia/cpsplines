from typing import Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cpsplines.fittings.grid_cpsplines import GridCPsplines
from cpsplines.graphics.plot_utils import granulate_prediction_range
from cpsplines.utils.rearrange_data import scatter_to_grid


class SurfacesDisplay:
    def __init__(self, X: Union[pd.Series, pd.DataFrame], y_pred: pd.Series):
        self.X = X
        self.y_pred = y_pred

    def plot(
        self,
        ax: Optional[plt.axes] = None,
        ax_contour: Optional[plt.axes] = None,
        contour_plot: bool = True,
        **kwargs,
    ):

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        if contour_plot:
            if ax_contour is None:
                _, ax_contour = plt.subplots(figsize=ax.figure.get_size_inches())

        data = pd.concat((self.X, pd.Series(self.y_pred)), axis=1)
        x, y = scatter_to_grid(data=data, y_col=data.columns[-1])
        x0, x1 = np.meshgrid(x[0], x[1])
        surf = ax.plot_surface(x0, x1, y.T, **kwargs)
        _ = ax.figure.colorbar(surf, ax=ax)

        if contour_plot:
            _ = ax_contour.contourf(x0, x1, y.T, 100, **kwargs)
            _ = ax_contour.figure.colorbar(surf, ax=ax_contour)
            # Establish the limits on the two axis (otherwise some extra knots
            # in the prediction regions may be further apart from extreme points
            # of the extended regressor samples)
            _ = ax_contour.set_xlim(self.X.iloc[:, 0].min(), self.X.iloc[:, 0].max())
            _ = ax_contour.set_ylim(self.X.iloc[:, 1].min(), self.X.iloc[:, 1].max())

        self.ax_ = ax
        self.figure_ = ax.figure

        if contour_plot:
            self.ax_contour_ = ax_contour
            self.figure_contour_ = ax_contour.figure

        return self

    @classmethod
    def from_estimator(
        cls,
        estimator: GridCPsplines,
        ax: Optional[plt.axes] = None,
        ax_contour: Optional[plt.axes] = None,
        contour_plot: bool = True,
        knot_positions: bool = False,
        zlim: Optional[Tuple[Union[int, float]]] = None,
        orientation: Optional[Tuple[Union[int, float]]] = None,
        figsize: Tuple[Union[int, float]] = (15, 10),
        **kwargs,
    ):
        bsp1 = estimator.bspline_bases[0]
        bsp2 = estimator.bspline_bases[1]

        X = pd.DataFrame(
            {
                "x0": np.repeat(bsp1.xsample, len(bsp2.xsample)),
                "x1": np.tile(bsp2.xsample, len(bsp1.xsample)),
            }
        )

        y_pred = estimator.predict(X)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        _ = fig.set_size_inches(*figsize)

        if contour_plot:
            if ax_contour is None:
                _, ax_contour = plt.subplots(figsize=figsize)
            _ = ax_contour.figure.set_size_inches(*figsize)

        # Provide the required orientation
        if orientation is not None:
            if len(orientation) != 2:
                raise ValueError("Only two angle coordinates may be passed.")
            _ = ax.view_init(*orientation)
        # Restrict to the required bounds the z-axis.
        if zlim is not None:
            if len(orientation) != 2:
                raise ValueError("Only two z-axis limits may be passed.")
            _ = ax.set_zlim3d(zlim)
            kwargs |= {"vmin": zlim[0], "vmax": zlim[1]}

        if contour_plot:
            if knot_positions:
                for knot in bsp1.knots[bsp1.deg : -bsp1.deg]:
                    _ = ax_contour.plot(
                        [knot] * len(X),
                        X.iloc[:, 1],
                        color="red",
                        linestyle="--",
                        alpha=0.3,
                    )
                for knot in bsp2.knots[bsp2.deg : -bsp2.deg]:
                    _ = ax_contour.plot(
                        X.iloc[:, 0],
                        [knot] * len(X.iloc[:, 0]),
                        color="red",
                        linestyle="--",
                        alpha=0.3,
                    )
        viz = SurfacesDisplay(X, y_pred)

        return viz.plot(
            ax=ax, ax_contour=ax_contour, contour_plot=contour_plot, **kwargs
        )
