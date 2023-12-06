import numpy as np
import pandas as pd
import pytest
from statsmodels.genmod.families.family import Binomial, Gaussian, Poisson

from cpsplines.psplines.bspline_basis import BsplineBasis
from cpsplines.psplines.penalty_matrix import PenaltyMatrix
from cpsplines.utils.gcv import GCV

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

ethanol = pd.read_csv("./data/ethanol.csv")
faithful = pd.read_csv("./data/faithful.csv")
kyphosis = pd.read_csv("./data/kyphosis.csv")

out1 = 0.06655360566286558
out2 = 0.8886437800778839
out3 = 0.11298981533051085
out4 = 0.8648914644625973
out5 = 0.47471342844068937
out6 = 0.9991894426009834


@pytest.mark.parametrize(
    "deg, ord_d, n_int, sp, family, data_arrangement, x, y, gcv",
    [
        (
            [3],
            [2],
            [5],
            [0.123],
            Gaussian(),
            "gridded",
            [np.linspace(0, 2 * np.pi, 11)],
            y_1,
            out1,
        ),
        (
            [3, 2],
            [2, 1],
            [5, 4],
            [0.456, 7.89],
            Gaussian(),
            "gridded",
            [np.linspace(0, 3 * np.pi, 7), np.linspace(0, 2 * np.pi, 5)],
            y_2,
            out2,
        ),
        (
            [2, 4, 3],
            [1, 3, 2],
            [2, 5, 4],
            [0.12, 1.345, 0.011],
            Gaussian(),
            "gridded",
            [
                np.linspace(0, 3 * np.pi, 3),
                np.linspace(0, 2 * np.pi, 6),
                np.linspace(0, 1 * np.pi, 5),
            ],
            y_3,
            out3,
        ),
        (
            [3, 3],
            [2, 1],
            [10, 8],
            [1.23, 3.45],
            Gaussian(),
            "scattered",
            [ethanol["C"].values, ethanol["E"].values],
            ethanol["NOx"].values,
            out4,
        ),
        (
            [3],
            [2],
            [5],
            [0.123],
            Poisson(),
            "gridded",
            [faithful["eruptions"].values],
            faithful["waiting"].values,
            out5,
        ),
        (
            [3],
            [2],
            [5],
            [0.123],
            Binomial(),
            "gridded",
            [kyphosis["Age"].values],
            pd.get_dummies(kyphosis["Kyphosis"])["Present"].astype(int).values,
            out6,
        ),
    ],
)
def test_gcv(deg, ord_d, n_int, sp, family, data_arrangement, x, y, gcv):
    bspline = [
        BsplineBasis(deg=d, xsample=xsam, n_int=n) for d, xsam, n in zip(deg, x, n_int)
    ]
    B = []
    for bsp in bspline:
        bsp.get_matrix_B()
        B.append(bsp.matrixB)
    D_mul = [
        PenaltyMatrix(bspline=bsp).get_penalty_matrix(**{"ord_d": o})
        for bsp, o in zip(bspline, ord_d)
    ]
    obj_matrices = {"B": B, "D_mul": D_mul, "y": y}

    gcv_out = GCV(
        sp=sp,
        obj_matrices=obj_matrices,
        family=family,
        data_arrangement=data_arrangement,
    )
    np.testing.assert_allclose(gcv_out, gcv)
