import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
from typing import Iterable, Optional, Tuple, Union

from template.smoothings.one_smoothing import OneSmoothing
from template.graphics.plot_utils import thin_prediction_range


def plot_curves(
    fittings: Iterable[OneSmoothing],
    col_curve: Optional[Iterable[str]] = None,
    x: Optional[Iterable[np.ndarray]] = None,
    y: Optional[Iterable[np.ndarray]] = None,
    col_pt: Optional[Iterable[str]] = None,
    alpha: Union[int, float] = 0.25,
    prediction_step: Iterable[Union[int, float]] = (0.5, 0.5),
    knot_positions: bool = False,
    constant_constraints: bool = False,
    figsize: Tuple[Union[int, float]] = (15, 10),
) -> Tuple[matplotlib.figure.Figure, plt.axes]:

    fig, ax = plt.subplots(figsize=figsize)

    if col_curve is None:
        col_curve = [
            "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
            for _ in range(len(fittings))
        ]

    if (x is not None) and (y is not None):
        if col_pt is None:
            col_pt = [
                "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
                for _ in range(len(x))
            ]
        for i, (pt_x, pt_y) in enumerate(zip(x, y)):
            _ = ax.scatter(
                x=pt_x,
                y=pt_y,
                c=col_pt[i],
                alpha=alpha,
            )

    for i, curve in enumerate(fittings):
        bsp = curve.bspline_bases[0]
        x_left, x_right = thin_prediction_range(
            bspline_bases=[bsp], prediction_step=[prediction_step]
        )
        y_end = None if bsp.int_forw == 0 else -bsp.int_forw
        _ = ax.plot(
            np.concatenate([x_left[0], bsp.xsample, x_right[0]]),
            np.concatenate(
                [
                    curve.predict([x_left[0]]),
                    curve.y_fitted[bsp.int_back : y_end],
                    curve.predict([x_right[0]]),
                ]
            ),
            col_curve[i],
            linewidth=2.0,
            label=str(i),
        )

        if bsp.int_back > 0:
            _ = ax.axvline(
                bsp.xsample[0], color=col_curve[i], linewidth=1.0, linestyle="--"
            )
        if bsp.int_forw > 0:
            _ = ax.axvline(
                bsp.xsample[-1], color=col_curve[i], linewidth=1.0, linestyle="--"
            )
        # If it is required, plot the position of the knots
        if knot_positions:
            for i in range(len(bsp.knots[bsp.deg : -bsp.deg])):
                _ = ax.axvline(
                    bsp.knots[bsp.deg : -bsp.deg][i], color="grey", alpha=0.25
                )
        if constant_constraints:
            if curve.int_constraints:
                if 0 in curve.int_constraints[0].keys():
                    for value in curve.int_constraints[0][0].values():
                        _ = ax.axhline(
                            value,
                            color="red",
                            linewidth=1.0,
                            linestyle="--",
                        )
    return fig, ax


def plot_surfaces(
    fittings: Iterable[OneSmoothing],
    col_surface: Optional[Iterable[str]] = None,
    contour_plot: bool = True,
    prediction_step: Iterable[Iterable[Union[int, float]]] = ((0.5, 0.5), (0.5, 0.5)),
    figsize: Tuple[Union[int, float]] = (12, 6),
    knot_positions: bool = False,
    zlim: Optional[Tuple[Union[int, float]]] = None,
    orientation: Optional[Tuple[Union[int, float]]] = None,
) -> Tuple[Tuple[matplotlib.figure.Figure, plt.axes]]:

    # Create the figure and provide the required orientation
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if orientation is not None:
        _ = ax.view_init(*orientation)
    if zlim is not None:
        _ = ax.set_zlim3d(zlim)
        kwargs_surf = {"vmin": zlim[0], "vmax": zlim[1]}
    else:
        kwargs_surf = {}
    if col_surface is None:
        col_surface = random.sample(plt.colormaps(), len(fittings))

    fig_ax = [(fig, ax)]

    for i, surface in enumerate(fittings):
        x_left, x_right = thin_prediction_range(
            bspline_bases=surface.bspline_bases, prediction_step=prediction_step
        )
        ext_x0 = np.concatenate(
            [x_left[0], surface.bspline_bases[0].xsample, x_right[0]]
        )
        ext_x1 = np.concatenate(
            [x_left[1], surface.bspline_bases[1].xsample, x_right[1]]
        )
        y_end_0 = (
            None
            if surface.bspline_bases[0].int_forw == 0
            else -surface.bspline_bases[0].int_forw
        )
        y_end_1 = (
            None
            if surface.bspline_bases[1].int_forw == 0
            else -surface.bspline_bases[1].int_forw
        )

        ext_y = surface.y_fitted[
            surface.bspline_bases[0].int_back : y_end_0,
            surface.bspline_bases[1].int_back : y_end_1,
        ]
        if x_left[1].size > 0:
            pred_left = surface.predict(x=[surface.bspline_bases[0].xsample, x_left[1]])
            ext_y = np.concatenate([pred_left, ext_y], axis=1)
        if x_right[1].size > 0:
            pred_right = surface.predict(
                x=[surface.bspline_bases[0].xsample, x_right[1]]
            )
            ext_y = np.concatenate([ext_y, pred_right], axis=1)
        if x_left[0].size > 0:
            pred_up = surface.predict(x=[x_left[0], ext_x1])
            ext_y = np.concatenate([pred_up, ext_y], axis=0)
        if x_right[0].size > 0:
            pred_down = surface.predict(x=[x_right[0], ext_x1])
            ext_y = np.concatenate([ext_y, pred_down], axis=0)
        Z, X = np.meshgrid(ext_x0, ext_x1)
        surf = ax.plot_surface(
            Z, X, ext_y.T, cmap=col_surface[i], rstride=2, cstride=2, **kwargs_surf
        )
        _ = fig.colorbar(surf, ax=ax)

        if contour_plot:
            fig_contour, ax_contour = plt.subplots(figsize=figsize)
            _ = ax_contour.contourf(
                Z, X, ext_y.T, 100, cmap=col_surface[i], **kwargs_surf
            )
            _ = fig_contour.colorbar(surf, ax=ax_contour)
            fig_ax.append((fig_contour, ax_contour))
        if knot_positions:
            for knot in surface.bspline_bases[0].knots[
                surface.bspline_bases[0].deg : -surface.bspline_bases[0].deg
            ]:
                _ = ax_contour.plot(
                    [knot] * len(surface.bspline_bases[1].xsample),
                    surface.bspline_bases[1].xsample,
                    color="red",
                    linestyle="--",
                    alpha=0.3,
                )
            for knot in surface.bspline_bases[1].knots[
                surface.bspline_bases[1].deg : -surface.bspline_bases[1].deg
            ]:
                _ = ax_contour.plot(
                    surface.bspline_bases[0].xsample,
                    [knot] * len(surface.bspline_bases[0].xsample),
                    color="red",
                    linestyle="--",
                    alpha=0.3,
                )
    return tuple(fig_ax)
