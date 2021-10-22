from typing import Iterable, Union

import numpy as np
import pytest
from cpsplines.psplines.bspline_basis import BsplineBasis
from cpsplines.psplines.penalty_matrix import PenaltyMatrix
from cpsplines.utils.fast_kron import matrix_by_transpose
from cpsplines.utils.gcv import GCV, gcv_mat


def gcv_brute_force(
    B_mul: Iterable[np.ndarray],
    D_mul: Iterable[np.ndarray],
    sp: Iterable[Union[int, float]],
    y: np.ndarray,
    y_fit: np.ndarray,
) -> float:
    """Computes the Generalized Cross Validation using the exact definition
    including all the matrices involved. It only supports three or less
    dimensions.

    Parameters
    ----------
    B_mul : Iterable[np.ndarray]
        The product of the design matrices from the B-splines bases. Must have
        length less or equal than 3.
    D_mul : Iterable[np.ndarray]
        The univariate penalty matrices. Must have length less or equal than 3.
    sp : Iterable[Union[int, float]]
        The smoothing parameter vector.
    y : np.ndarray
        The response variable sample
    y_fit : np.ndarray
        The fitted response variable array.

    Returns
    -------
    float
        The Generalized Cross Validation value.

    Raises
    ------
    ValueError
        If the shapes of response variable arrays differ.
    ValueError
        If the number of input B and D matrices differ.
    ValueError
        If the number of input B matrices is greater than 3.
    """

    if y.shape != y_fit.shape:
        raise ValueError("The shape of the response variable array must agree.")
    if len(B_mul) != len(D_mul):
        raise ValueError("Number of B elements must be equal to number of D elements.")
    if len(B_mul) > 3:
        raise ValueError("Not implemented for more than three dimensions.")
    # Compute the hat matrix depending on the number of dimensions
    if len(B_mul) == 1:
        H = np.linalg.solve(B_mul[0] + np.multiply(sp[0], D_mul[0]), B_mul[0])
    elif len(B_mul) == 2:
        Q = (
            np.kron(B_mul[1], B_mul[0])
            + np.kron(np.eye(D_mul[1].shape[1]), np.multiply(sp[0], D_mul[0]))
            + np.kron(np.multiply(sp[1], D_mul[1]), np.eye(D_mul[0].shape[1]))
        )
        H = np.linalg.solve(Q, np.kron(B_mul[1], B_mul[0]))
    else:
        Q = (
            np.kron(B_mul[2], np.kron(B_mul[1], B_mul[0]))
            + np.kron(
                np.eye(D_mul[2].shape[1] * D_mul[1].shape[1]),
                np.multiply(sp[0], D_mul[0]),
            )
            + np.kron(
                np.kron(np.eye(D_mul[2].shape[1]), np.multiply(sp[1], D_mul[1])),
                np.eye(D_mul[0].shape[1]),
            )
            + np.kron(
                np.multiply(sp[2], D_mul[2]),
                np.eye(D_mul[1].shape[1] * D_mul[0].shape[1]),
            )
        )
        H = np.linalg.solve(Q, np.kron(B_mul[2], np.kron(B_mul[1], B_mul[0])))
    # Get the Generalized Cross Validation value
    return (np.linalg.norm((y - y_fit)) ** 2 * np.prod(y.shape)) / (
        np.prod(y.shape) - np.trace(H)
    ) ** 2


y_1 = np.array(
    [
        1.04412275,
        0.77592998,
        0.55209411,
        -0.33422621,
        -0.79805601,
        -0.84175189,
        -0.89994023,
        -0.36818066,
        0.32777732,
        0.77603,
        0.88072354,
    ]
)

y_fit_1 = np.array(
    [
        1.14934175,
        0.71452073,
        0.23994151,
        -0.25476975,
        -0.64462143,
        -0.80211703,
        -0.69510624,
        -0.34214611,
        0.13072284,
        0.59543651,
        1.02331994,
    ]
)

y_2 = np.array(
    [
        [0.44122749, -0.33087015, 2.43077119, -0.25209213, 0.10960984],
        [1.58248112, 0.0907676, -0.59163666, -0.81239677, -0.32986996],
        [-1.19276461, -0.20487651, -0.35882895, 0.6034716, -1.66478853],
        [-0.70017904, 0.15139101, 1.85733101, -0.51117956, 0.64484751],
        [-0.98060789, -0.85685315, -0.87187918, -0.42250793, 0.99643983],
        [0.71242127, 1.05914424, -0.36331088, -0.99671116, -0.10593044],
        [0.79305332, -0.63157163, -0.00619491, -0.10106761, -0.05230815],
    ]
)

y_fit_2 = np.array(
    [
        [0.39583918, 0.37761835, 0.347656, 0.29440816, 0.25379401],
        [0.10072316, 0.0758906, 0.03645742, -0.01922271, -0.06118704],
        [-0.14077813, -0.12723966, -0.12434936, -0.15179711, -0.18186141],
        [-0.24670913, -0.20462625, -0.16470918, -0.15275488, -0.14929084],
        [-0.18820456, -0.1676606, -0.1495958, -0.12901441, -0.09894657],
        [-0.04233988, -0.0633497, -0.09345534, -0.09881817, -0.07713563],
        [0.07827233, 0.04195354, 0.00104969, -0.0186971, -0.01738777],
    ]
)

y_3 = np.array(
    [
        [
            [0.11030687, -0.08271754, 0.6076928, -0.06302303, 0.02740246],
            [0.39562028, -0.2273081, -0.14790916, 0.04690081, -0.08246749],
            [-0.29819115, -0.05121913, -0.08970724, 0.1508679, -0.41619713],
            [-0.17504476, 0.28784775, 0.46433275, -0.37779489, 0.16121188],
            [-0.24515197, -0.21421329, -0.2179698, -0.10562698, 0.24910996],
            [0.17810532, 0.01478606, -0.09082772, 0.00082221, -0.02648261],
        ],
        [
            [0.19826333, -0.15789291, -0.00154873, -0.0252669, -0.01307704],
            [0.06230441, -0.62308349, -0.61734437, -0.69421741, 0.39038307],
            [-0.07646326, -0.53505979, -0.56260071, -0.32676732, 0.0674031],
            [0.32299085, 0.70046268, 0.71139535, 0.33154287, -0.02515359],
            [0.3533495, 0.72781204, 0.62336323, 0.5001072, -0.14437831],
            [0.28805119, -0.026791, 0.56502669, 0.16415487, 0.03120171],
        ],
        [
            [-0.10892598, 0.24304483, -0.06017779, -0.20603086, 0.14203318],
            [0.00318958, 0.29726518, -0.01839833, -0.71492199, 0.1973416],
            [-0.46943522, 0.38468904, 0.45534118, -0.10675785, -0.29117548],
            [-0.34926851, 0.21816366, -0.05052954, -0.14958998, -0.06085493],
            [0.52212867, 0.08672983, 0.18643174, 0.1942269, 0.25460528],
            [0.26533786, -0.17761661, -0.05379695, -0.19019008, -0.17779081],
        ],
    ]
)

y_fit_3 = np.array(
    [
        [
            [0.14620947, -0.02079486, 0.03249839, -0.01959649, 0.04563657],
            [-0.00385404, -0.06855719, -0.01572303, -0.08444499, -0.01953818],
            [-0.07897774, -0.03803142, 0.00907818, -0.08265218, -0.03439175],
            [-0.0779496, 0.02579857, 0.06662659, -0.03843807, -0.01485294],
            [-0.00316984, 0.06794833, 0.09807687, 0.01076603, 0.01284133],
            [0.14396795, 0.05743982, 0.06449239, 0.03444781, 0.02533011],
        ],
        [
            [0.0767984, -0.08772433, -0.14766937, -0.18173234, 0.06940114],
            [-0.04312678, -0.08850095, -0.1452439, -0.22113327, 0.01157201],
            [-0.05037519, 0.023461, -0.00802457, -0.13514036, -0.00366228],
            [0.02866757, 0.17066411, 0.17644919, 0.01137911, 0.0087248],
            [0.15145333, 0.25266291, 0.28827765, 0.12868107, 0.02306828],
            [0.28503526, 0.20551734, 0.24240499, 0.14925631, 0.01693122],
        ],
        [
            [0.04397969, 0.063017, -0.08388446, -0.22594632, 0.06931447],
            [-0.08028535, 0.09021596, -0.03810249, -0.25304778, 0.00197333],
            [-0.09032671, 0.14538972, 0.06174555, -0.17604932, -0.02575864],
            [-0.00734497, 0.19282519, 0.15912736, -0.05390058, -0.0260306],
            [0.12939478, 0.1886465, 0.18850045, 0.03988198, -0.02174704],
            [0.28249543, 0.10646542, 0.10742523, 0.05319826, -0.03304391],
        ],
    ]
)


@pytest.mark.parametrize(
    "deg, regr_sample, n_int, prediction, ord_d, sp, y, y_fit",
    [
        (
            [3],
            [np.linspace(0, 2 * np.pi, 11)],
            [5],
            [{}],
            [2],
            [0.123],
            y_1,
            y_fit_1,
        ),
        (
            [3, 2],
            [np.linspace(0, 3 * np.pi, 7), np.linspace(0, 2 * np.pi, 5)],
            [5, 4],
            [{}, {}],
            [2, 1],
            [0.456, 7.89],
            y_2,
            y_fit_2,
        ),
        (
            [2, 4, 3],
            [
                np.linspace(0, 3 * np.pi, 3),
                np.linspace(0, 2 * np.pi, 6),
                np.linspace(0, 1 * np.pi, 5),
            ],
            [2, 5, 4],
            [{}, {}, {}],
            [1, 3, 2],
            [0.12, 1.345, 0.011],
            y_3,
            y_fit_3,
        ),
    ],
)
def test_gcv(deg, regr_sample, n_int, prediction, ord_d, sp, y, y_fit):
    bspline = [
        BsplineBasis(deg=d, xsample=xsam, n_int=n, prediction=pred)
        for d, xsam, n, pred in zip(deg, regr_sample, n_int, prediction)
    ]
    B = []
    for bsp in bspline:
        bsp.get_matrix_B()
        B.append(bsp.matrixB)
    D_mul = [
        PenaltyMatrix(bspline=bsp).get_penalty_matrix(**{"ord_d": o})
        for bsp, o in zip(bspline, ord_d)
    ]
    B_mul = list(map(matrix_by_transpose, B))
    Q_matrices = gcv_mat(B_mul=B_mul, D_mul=D_mul)
    gcv_out = GCV(sp=sp, B_weighted=B, Q_matrices=Q_matrices, y=y)
    gcv_brute = gcv_brute_force(B_mul=B_mul, D_mul=D_mul, sp=sp, y=y, y_fit=y_fit)
    np.testing.assert_allclose(gcv_out, gcv_brute)
