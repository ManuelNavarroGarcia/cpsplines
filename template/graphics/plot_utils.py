import numpy as np
from typing import Iterable, Optional, Union

from template.psplines.bspline_basis import BsplineBasis


def thin_prediction_range(
    bspline_bases: Iterable[BsplineBasis],
    prediction_step: Optional[Iterable[Iterable[Union[int, float]]]] = None,
):
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
            x_left_pred[i] = np.arange(
                bsp.prediction["backwards"],
                bsp.xsample[0],
                prediction_step[i][0],
            )
        else:
            x_left_pred[i] = np.array([])

        if bsp.int_forw > 0:
            x_right_pred[i] = np.arange(
                bsp.prediction["forward"],
                bsp.xsample[-1],
                -prediction_step[i][1],
            )[::-1]
        else:
            x_right_pred[i] = np.array([])
    return x_left_pred, x_right_pred
