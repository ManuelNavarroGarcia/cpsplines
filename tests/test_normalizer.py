import numpy as np
import pytest
from cpsplines.utils.normalize_data import DataNormalizer


@pytest.mark.parametrize(
    "dim",
    [
        (7,),
        (19, 23),
        (33, 29, 13),
    ],
)
def test_data_normalizer(dim):
    y = np.random.random(dim)
    scaler = DataNormalizer()
    _ = scaler.fit(y=y)
    y_scaled = scaler.transform(y=y)
    y_new = scaler.inverse_transform(y=y_scaled)
    np.testing.assert_allclose(y, y_new)
