import numpy as np
import pandas as pd
import pytest
from cpsplines.utils.rearrange_data import grid_to_scatter, scatter_to_grid

data = pd.DataFrame(
    {
        "x": [1, 1, 1, 1, 2, 2, 2, 2, 2],
        "y": [1, 1, 2, 2, 1, 1, 1, 2, 2],
        "z": [1, 2, 1, 3, 1, 2, 3, 1, 2],
        "value": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
)
x_cols = ["x", "y", "z"]
y_col = "value"
coordinates = [np.unique(x) for x in data[x_cols].values.T]
arr = np.array(
    [[[1.0, 2.0, np.nan], [3.0, np.nan, 4.0]], [[5.0, 6.0, 7.0], [8.0, 9.0, np.nan]]]
)


@pytest.mark.parametrize(
    "data, y_col, out_x, out_y",
    [
        (data, "value", coordinates, arr),
    ],
)
def test_scatter_to_grid(data, y_col, out_x, out_y):
    x, y = scatter_to_grid(data=data, y_col=y_col)
    for array, expected in zip(x + [y], out_x + [out_y]):
        np.testing.assert_allclose(array, expected)


@pytest.mark.parametrize(
    "x, y, x_cols, y_col, out_data",
    [
        (coordinates, arr, x_cols, y_col, data),
    ],
)
def test_grid_to_scatter(x, y, x_cols, y_col, out_data):
    data = grid_to_scatter(x=x, y=arr, x_cols=x_cols, y_col=y_col)
    pd.testing.assert_frame_equal(data, out_data, check_dtype=False)
