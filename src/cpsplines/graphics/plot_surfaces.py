import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cpsplines.fittings.fit_cpsplines import CPsplines


class SurfacesDisplay:
    """Fitted surface visualization.

    Parameters
    ----------
    X : Union[pd.Series, pd.DataFrame]
        The abscissa of the points used in the fitting procedure.
    y_pred : np.ndarray
        The predicted values at `X`.

    Attributes
    ----------
    ax_ : matplotlib Axes
        Axes with fitted surface.
    figure_ : matplotlib Figure
        Figure containing the surface.
    ax_contour_ : matplotlib Axes
        Axes with the contour plot of the fitted surface.
    figure_contour_ : matplotlib Figure
        Figure containing the contour plot of the fitted surface.
    """

    def __init__(self, X: pd.Series | pd.DataFrame, y_pred: pd.Series):
        self.X = X
        self.y_pred = y_pred

    def plot(
        self,
        contour_plot: bool = True,
        ax: plt.Axes | None = None,
        ax_contour: plt.Axes | None = None,
        **kwargs,
    ):
        """Plot visualization. Extra keyword arguments will be passed to
        matplotlib's `plot`.

        Parameters
        ----------
        contour_plot : bool, optional
            If True, the contour plot of the surface is plotted with the same
            colormap as the surface plot. By default, True.
        ax : Optional[plt.axes], optional
           Axes object to plot on. If `None`, a new figure and axes is created.
           By default, None.
        ax_contour : Optional[plt.axes], optional
           Axes object to plot the contour on. If `None`, a new figure and axes
           is created. By default, None.

        Returns
        -------
        display : :class:`~src.cpsplines.graphics.SurfacesDisplay`
            Object that stores computed values.
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        if contour_plot:
            if ax_contour is None:
                _, ax_contour = plt.subplots(figsize=ax.figure.get_size_inches())

        data = pd.concat((self.X, pd.Series(self.y_pred)), axis=1)
        # Plot the surface and include the colorbar
        surf = ax.plot_trisurf(*data.to_numpy().T, **kwargs)
        _ = ax.figure.colorbar(surf, ax=ax)

        # If required, plot the contour plot of the surface
        if contour_plot:
            _ = ax_contour.tricontourf(*data.to_numpy().T, 100, **kwargs)
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
        estimator: CPsplines,
        contour_plot: bool = True,
        ax: plt.Axes | None = None,
        ax_contour: plt.Axes | None = None,
        knot_positions: bool = False,
        density: int = 5,
        zlim: tuple[int | float] | None = None,
        orientation: tuple[int | float] | None = None,
        figsize: tuple[int | float] = (15, 10),
        **kwargs,
    ):
        """Create a surface fitting display from an estimator.

        Parameters
        ----------
        estimator : CPsplines
            A fitted `CPsplines` object.
        contour_plot : bool, optional
            If True, the contour plot of the surface is plotted with the same
            colormap as the surface plot. By default, True.
        ax : Optional[plt.axes], optional
           Axes object to plot on. If `None`, a new figure and axes is created.
           By default, None.
        ax_contour : Optional[plt.axes], optional
           Axes object to plot the contour on. If `None`, a new figure and axes
           is created. By default, None.
        knot_positions : bool, optional
           If True, the positions where the inner knots are located are marked
           as grey vertical lines. By default, False.
        density : int, optional
            Number of points in which the interval between adjacent knots along
            each dimension is splitted.
        zlim : Optional[Tuple[Union[int, float]]], optional
            An iterable with two elements used to restrict the range on the
            z-axis. First element is the lower bound and second element
            corresponds to upper bound. By default, None.
        orientation : Optional[Tuple[Union[int, float]]], optional
            Set the elevation angle in the z plane and azimuth angle in the xy
            plane of the axes. If None, the default value is (30,-60).
        figsize : Tuple[Union[int, float]], optional
            The size of the figure, by default (15, 10).

        Returns
        -------
        display : :class:`~src.cpsplines.graphics.SurfacesDisplay`
            Object that stores computed values.

        Raises
        ------
        ValueError
            If wrong values are passed to `orientation`.
        ValueError
            If wrong values are passed to `zlim`.
        """
        bsp1 = estimator.bspline_bases[0]
        bsp2 = estimator.bspline_bases[1]

        ext_1 = np.linspace(
            bsp1.knots[bsp1.deg],
            bsp1.knots[-bsp1.deg - 1],
            len(bsp1.knots[bsp1.deg : -bsp1.deg - 1]) * density + 1,
        )

        ext_2 = np.linspace(
            bsp2.knots[bsp2.deg],
            bsp2.knots[-bsp2.deg - 1],
            len(bsp2.knots[bsp2.deg : -bsp2.deg - 1]) * density + 1,
        )
        X = pd.DataFrame(
            {
                "x0": np.repeat(ext_1, len(ext_2)),
                "x1": np.tile(ext_2, len(ext_1)),
            }
        )
        X = X[estimator.data_hull.find_simplex(X) >= 0].reset_index(drop=True)

        # Generate the predictions
        y_pred = estimator.predict(X)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        _ = ax.figure.set_size_inches(*figsize)

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
            # If it is required, plot the position of the knots
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

        return viz.plot(contour_plot=contour_plot, ax=ax, ax_contour=ax_contour, **kwargs)
