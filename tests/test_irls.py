import numpy as np
import pytest
from cpsplines.psplines.bspline_basis import BsplineBasis
from cpsplines.psplines.penalty_matrix import PenaltyMatrix
from cpsplines.utils.fast_kron import penalization_term
from cpsplines.utils.irls import fit_irls
from statsmodels.genmod.families.family import Gaussian, Poisson

# Test IRLS algorithm for multidimensional data. The results coincides with the
# ones from R package JOPS, version 0.1.15. The code used to fit the models with
# this R packages are located above the fitted values to be checked

x_1 = np.array(
    [
        3.6,
        1.8,
        3.333,
        2.283,
        4.533,
        2.883,
        4.7,
        3.6,
        1.95,
        4.35,
        1.833,
        3.917,
        4.2,
        1.75,
        4.7,
    ]
)

y_1 = np.array([79, 54, 74, 62, 85, 55, 88, 85, 51, 85, 54, 84, 78, 47, 83])

# library("JOPS")
# x = faithful$eruptions[1:15]
# y = faithful$waiting[1:15]
# fit <- psNormal(
#   x,
#   y,
#   xl = min(x),
#   xr = max(x),
#   nseg = 5,
#   bdeg = 3,
#   pord = 2,
#   lambda = 0.135,
# )
# print(fit$muhat)

y_fit_1 = np.array(
    [
        78.50366225,
        51.48825473,
        73.43345923,
        56.64612567,
        84.8472375,
        63.92317022,
        85.27375252,
        78.50366225,
        53.16202261,
        84.32297898,
        51.86376328,
        82.08723744,
        83.76026389,
        50.9106569,
        85.27375252,
    ]
)

# library("JOPS")
# x = faithful$eruptions[1:15]
# y = faithful$waiting[1:15]
# fit <- psPoisson(
#   x,
#   y,
#   xl = min(x),
#   xr = max(x),
#   nseg = 5,
#   bdeg = 3,
#   pord = 2,
#   lambda = 0.135,
# )
# print(fit$muhat)

y_fit_2 = np.array(
    [
        82.35511065,
        50.96275795,
        72.55098169,
        60.05593726,
        84.5474684,
        56.67552777,
        85.75987629,
        82.35511065,
        54.72773342,
        82.6016659,
        51.81016557,
        82.65526924,
        81.50269236,
        49.67982656,
        85.75987629,
    ]
)

x_2 = np.array(
    [1.934375, 2.303125, 2.671875, 3.040625, 3.409375, 3.778125, 4.146875, 4.515625]
)
x_3 = np.array([49.5625, 54.6875, 59.8125, 64.9375, 70.0625, 75.1875, 80.3125, 85.4375])

y_2 = np.array(
    [
        [2, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 2],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 4],
    ]
)

# library("JOPS")
# x = faithful$eruptions[1:15]
# y = faithful$waiting[1:15]
# h = hist2d(x, y, c(8, 8))

# fit <- ps2DGLM(
#   Data = cbind(rep(h$xgrid, 8), as.vector((matrix(rep(h$ygrid, 8), nrow=8, byrow=TRUE))), as.vector(h$H)),
#   Pars = rbind(c(min(h$xgrid), max(h$xgrid), 5, 4, 0.135, 3),
#                c(min(h$ygrid), max(h$ygrid), 4, 3, 12.87, 2)),
#   family = "gaussian")
# matrix(fit$mu, nrow=8)

y_fit_3 = np.array(
    [
        [
            1.33509196,
            1.07438801,
            0.81731511,
            0.56650344,
            0.32408945,
            0.0899548,
            -0.13825074,
            -0.36317142,
        ],
        [
            0.5552544,
            0.43543849,
            0.31831329,
            0.20568227,
            0.0993843,
            -0.00054931,
            -0.09592998,
            -0.18884011,
        ],
        [
            0.15487935,
            0.12313367,
            0.09204177,
            0.06247127,
            0.03586915,
            0.01260177,
            -0.00813326,
            -0.02732664,
        ],
        [
            0.00994953,
            0.02654332,
            0.04278757,
            0.0595545,
            0.07837383,
            0.09975293,
            0.12309561,
            0.14775666,
        ],
        [
            -0.03779263,
            0.00793275,
            0.05496879,
            0.10526317,
            0.16102885,
            0.2227058,
            0.2888825,
            0.35772875,
        ],
        [
            -0.10949641,
            -0.03055095,
            0.05281384,
            0.14427619,
            0.24750474,
            0.36335799,
            0.48966878,
            0.62238417,
        ],
        [
            -0.2643397,
            -0.12407884,
            0.02250183,
            0.18062482,
            0.35574362,
            0.54987336,
            0.76110955,
            0.98358915,
        ],
        [
            -0.52697836,
            -0.27449211,
            -0.01557849,
            0.25505428,
            0.54308294,
            0.85100078,
            1.17752944,
            1.51665641,
        ],
    ]
)

# library("JOPS")
# x = faithful$eruptions[1:15]
# y = faithful$waiting[1:15]
# h = hist2d(x, y, c(8, 8))

# fit <- ps2DGLM(
#   Data = cbind(rep(h$xgrid, 8), as.vector((matrix(rep(h$ygrid, 8), nrow=8, byrow=TRUE))), as.vector(h$H)),
#   Pars = rbind(c(min(h$xgrid), max(h$xgrid), 5, 4, 0.135, 3),
#                c(min(h$ygrid), max(h$ygrid), 4, 3, 12.87, 2)),
#   family = "poisson")
# matrix(fit$mu, nrow=8)

y_fit_4 = np.array(
    [
        [
            2.3900848,
            0.78805355,
            0.259058,
            0.08483452,
            0.02770283,
            0.00903048,
            0.00294102,
            0.00095741,
        ],
        [
            0.68291173,
            0.3434418,
            0.17223872,
            0.08604518,
            0.04285978,
            0.02130952,
            0.01058553,
            0.0052567,
        ],
        [
            0.20809607,
            0.15531862,
            0.11565778,
            0.0858586,
            0.06359116,
            0.04702959,
            0.03475292,
            0.02567369,
        ],
        [
            0.05866588,
            0.06437937,
            0.07050156,
            0.07701841,
            0.08398189,
            0.09143379,
            0.09939307,
            0.10793443,
        ],
        [
            0.01252717,
            0.02039918,
            0.03316905,
            0.05384517,
            0.08727898,
            0.14120798,
            0.2278558,
            0.36684819,
        ],
        [
            0.00169431,
            0.00420129,
            0.01041168,
            0.02578375,
            0.06379034,
            0.15755834,
            0.38811954,
            0.95376223,
        ],
        [
            0.00012973,
            0.00051098,
            0.00201236,
            0.00792351,
            0.03118942,
            0.12270036,
            0.48221702,
            1.89362129,
        ],
        [
            0.00000533,
            0.00003537,
            0.00023483,
            0.00155967,
            0.01036475,
            0.06892781,
            0.45872057,
            3.05479559,
        ],
    ]
)


@pytest.mark.parametrize(
    "deg, ord_d, n_int, sp, family, x, y, y_fit",
    [
        (
            [3],
            [2],
            [5],
            [0.135],
            Gaussian(),
            [x_1],
            y_1,
            y_fit_1,
        ),
        (
            [3],
            [2],
            [5],
            [0.135],
            Poisson(),
            [x_1],
            y_1,
            y_fit_2,
        ),
        (
            [4, 3],
            [3, 2],
            [5, 4],
            [0.135, 12.87],
            Gaussian(),
            [x_2, x_3],
            y_2,
            y_fit_3,
        ),
        (
            [4, 3],
            [3, 2],
            [5, 4],
            [0.135, 12.87],
            Poisson(),
            [x_2, x_3],
            y_2,
            y_fit_4,
        ),
    ],
)
def test_gcv(deg, ord_d, n_int, sp, family, x, y, y_fit):
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

    penalty_list = penalization_term(matrices=D_mul)
    penalty_term = np.add.reduce([np.multiply(s, P) for P, s in zip(penalty_list, sp)])

    obj_matrices = {"B_w": B, "y": y}

    out = fit_irls(obj_matrices=obj_matrices, penalty_term=penalty_term, family=family)
    np.testing.assert_allclose(out, y_fit, atol=1e-6)
