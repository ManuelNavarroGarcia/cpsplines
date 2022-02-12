import numpy as np
import pytest
from cpsplines.utils.normalize_data import DataNormalizer


@pytest.mark.parametrize(
    "dim, derivative",
    [
        ((7,), True),
        ((7,), False),
        ((19, 23), True),
        ((19, 23), False),
        ((33, 29, 13), True),
        ((33, 29, 13), False),
    ],
)
def test_data_normalizer(dim, derivative):
    y = np.random.random(dim)
    scaler = DataNormalizer()
    _ = scaler.fit(y=y)
    y_scaled = scaler.transform(y=y, derivative=derivative)
    y_new = scaler.inverse_transform(y=y_scaled, derivative=derivative)
    np.testing.assert_allclose(y, y_new)
