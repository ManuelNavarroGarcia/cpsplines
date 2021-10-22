from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
from cpsplines.psplines.bspline_basis import BsplineBasis


def granulate_prediction_range(
    bspline_bases: Iterable[BsplineBasis],
    prediction_step: Optional[Iterable[Iterable[Union[int, float]]]] = None,
) -> Tuple[Dict[int, np.ndarray]]:

    """
    Generate an array of equidistant points with small difference between each
    other at every prediction region.

    Parameters
    ----------
    bspline_bases : Iterable[BsplineBasis]
        The B-spline bases objects.
    prediction_step : Optional[Iterable[Iterable[Union[int, float]]]], optional
        The step length used to generate the granular regressor vector. The
        first term of each tuple corresponds to backwards region while the last
        term corresponds to forecast prediction. If None, a default step of 0.5
        on every direction is generated. By defualt, None.

    Returns
    -------
    Tuple[Dict[int, np.ndarray]]
        A tuple containing two dictionaries, one for backwards regions and one
        for forecast regions. Both dictionaries have as keys the ordinal names
        of the variables and as values the granular regressor vector on this
        region and for this variable.

    Raises
    ------
    ValueError
        If the number of B-spline bases and the prediction step passed have not
        the same shape.
    """

    if prediction_step is None:
        prediction_step = [(0.5, 0.5) for _ in range(len(bspline_bases))]
    if len({len(i) for i in [bspline_bases, prediction_step]}) != 1:
        raise ValueError(
            "The lengths of `bspline_bases` and `prediction_step` must agree."
        )

    x_left_pred = {}
    x_right_pred = {}
    for i, bsp in enumerate(bspline_bases):
        if bsp.int_back > 0:
            # This way, the minimum value of xsample is not picked up in the
            # np.arange of the backwards region
            x_left_pred[i] = np.arange(
                bsp.prediction["backwards"],
                bsp.xsample[0],
                prediction_step[i][0],
            )
        else:
            x_left_pred[i] = np.array([])

        if bsp.int_forw > 0:
            # This way, the maximum value of xsample is not picked up in the
            # np.arange of the forecast region
            x_right_pred[i] = np.arange(
                bsp.prediction["forward"],
                bsp.xsample[-1],
                -prediction_step[i][1],
            )[::-1]
        else:
            x_right_pred[i] = np.array([])
    return x_left_pred, x_right_pred
