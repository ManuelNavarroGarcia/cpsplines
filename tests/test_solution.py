import itertools

import numpy as np
import pytest
from cpsplines.fittings.grid_cpsplines import GridCPsplines
from scipy.stats import multivariate_normal, norm

# np.cos(x) (unconstrained)
# Using grid search
# No out-of-range prediction
sol1 = np.array(
    [
        1.13211815,
        1.03923459,
        0.87985899,
        0.44198769,
        -0.15101383,
        -0.69063377,
        -1.0120022,
        -1.0120022,
        -0.69063377,
        -0.15101383,
        0.44198769,
        0.87985899,
        1.03923459,
        1.13211815,
    ]
)

# np.cos(x) (non-negativity)
# Using grid search
# No out-of-range prediction
sol2 = np.array(
    [
        1.09034519,
        1.03523205,
        0.91087855,
        0.33618185,
        -0.0050857,
        0.00169804,
        -0.00007382,
        -0.00007385,
        0.00169818,
        -0.00508605,
        0.33618199,
        0.91087852,
        1.03523205,
        1.09034524,
    ]
)

# np.exp(4 - x / 25) + 4 * np.cos(x / 8) (unconstrained)
# Using grid search
# No out-of-range prediction
sol3 = np.array(
    [
        157.72305396,
        99.04296211,
        55.61628242,
        27.45043306,
        14.22655173,
        11.38107956,
        8.594066,
        2.37726194,
        -0.03712625,
        2.73446081,
        2.48116084,
        -1.50266035,
        -1.25842823,
        2.52896426,
        1.7301561,
        -2.8234932,
        -3.28028957,
        3.59042672,
        17.78015563,
        39.27004736,
    ]
)

# np.exp(4 - x / 25) + 4 * np.cos(x / 8) (10 <= y <= 40)
# Using grid search
# No out-of-range prediction
sol4 = np.array(
    [
        29.56171239,
        42.49430705,
        42.91977505,
        31.40441799,
        16.36601712,
        9.90236658,
        10.01162103,
        9.99838448,
        10.00053177,
        9.9998132,
        10.00010617,
        9.99995007,
        10.00001995,
        9.9999921,
        10.00000559,
        9.9999981,
        9.99999981,
        10.00000434,
        9.99997961,
        10.00046096,
    ]
)

# 2 * (2 * x - 1) ** 3 (unconstrained)
# Using optimizer
# No out-of-range prediction
sol5 = np.array(
    [
        -2.5951675,
        -1.40413346,
        -0.64394994,
        -0.22001628,
        -0.03598511,
        0.00399533,
        -0.00399533,
        0.03598511,
        0.22001628,
        0.64394994,
        1.40413346,
        2.5951675,
    ]
)

# 2 * (2 * x - 1) ** 3 (non-decreasing and concave)
# Using optimizer
# No out-of-range prediction
sol6 = np.array(
    [
        -2.60600802,
        -1.39765479,
        -0.64450191,
        -0.43731106,
        -0.23012034,
        -0.02292962,
        0.18426112,
        0.39145186,
        0.59864262,
        0.8058334,
        1.01302418,
        1.22021493,
    ]
)

# np.sin(3 * pi * x) * np.sin(2 * pi * y) (unconstrained)
# Using grid search
# No out-of-range prediction
sol7 = np.array(
    [
        [-1.026387, 0.1079524, 0.36420651, -0.36420651, -0.1079524, 1.026387],
        [-0.59581144, 0.55225976, 0.68119664, -0.68119664, -0.55225976, 0.59581144],
        [-0.18216655, 0.64291108, 0.64651309, -0.64651309, -0.64291108, 0.18216655],
        [0.05908496, -0.4352502, -0.52001599, 0.52001599, 0.4352502, -0.05908496],
        [0.05908496, -0.4352502, -0.52001599, 0.52001599, 0.4352502, -0.05908496],
        [-0.18216655, 0.64291108, 0.64651309, -0.64651309, -0.64291108, 0.18216655],
        [-0.59581144, 0.55225976, 0.68119664, -0.68119664, -0.55225976, 0.59581144],
        [-1.026387, 0.1079524, 0.36420651, -0.36420651, -0.1079524, 1.026387],
    ]
)

# np.sin(3 * pi * x) * np.sin(2 * pi * y) (non-negativity)
# Using grid search
# No out-of-range prediction
sol8 = np.array(
    [
        [-0.80985508, 0.31668373, -0.03557304, -0.03012587, 0.1575913, -0.13556319],
        [-0.60077917, 0.64374306, 0.32618134, -0.04510788, 0.14143147, -0.12715063],
        [-0.37227211, 0.6935879, 0.41873278, -0.14854258, 0.02396532, -0.1031168],
        [-0.13824966, 0.12405808, -0.17147697, 0.29493385, 0.42637766, -0.12405261],
        [-0.13796792, 0.12420349, -0.17358708, 0.29943516, 0.42120213, -0.10072458],
        [-0.37020297, 0.69188251, 0.42227456, -0.14375134, 0.02598244, -0.08707477],
        [-0.59664828, 0.6396059, 0.33519406, -0.08044187, 0.17006577, -0.24617402],
        [-0.80385412, 0.31034413, -0.02467559, -0.14880051, 0.18024642, -0.42036441],
    ]
)

# np.sin(3 * pi * x) * np.sin(2 * pi * y) (unconstrained)
# Using optimizer
# No out-of-range prediction
sol9 = np.array(
    [
        [0.48733637, -0.48700016, -0.49660183, 0.49660183, 0.48700016, -0.48733637],
        [-0.51039247, 0.49298224, 0.49012297, -0.49012297, -0.49298224, 0.51039247],
        [-1.14278011, 1.0832033, 1.06131396, -1.06131396, -1.0832033, 1.14278011],
        [-0.81254844, 0.77377079, 0.76089859, -0.76089859, -0.77377079, 0.81254844],
        [0.17915835, -0.17083564, -0.16816641, 0.16816641, 0.17083564, -0.17915835],
        [1.02647689, -0.97568549, -0.95807095, 0.95807095, 0.97568549, -1.02647689],
        [1.02647689, -0.97568549, -0.95807095, 0.95807095, 0.97568549, -1.02647689],
        [0.17915835, -0.17083564, -0.16816641, 0.16816641, 0.17083564, -0.17915835],
        [-0.81254844, 0.77377079, 0.76089859, -0.76089859, -0.77377079, 0.81254844],
        [-1.14278011, 1.0832033, 1.06131396, -1.06131396, -1.0832033, 1.14278011],
        [-0.51039247, 0.49298224, 0.49012297, -0.49012297, -0.49298224, 0.51039247],
        [0.48733637, -0.48700016, -0.49660183, 0.49660183, 0.48700016, -0.48733637],
    ]
)

# np.sin(3 * pi * x) * np.sin(2 * pi * y) (non-negativity and non-decreasing
# along y-direction)
# Using optimizer
# No out-of-range prediction
sol10 = np.array(
    [
        [-0.02166063, -0.01945414, -0.02166059, 0.01751593, 0.01700371, 0.01551912],
        [0.03167037, 0.02946398, 0.03167043, -0.00750608, -0.00699385, -0.0055092],
        [-0.01066619, -0.00845979, -0.01066625, 0.02851026, 0.02799803, 0.02651337],
        [0.02808215, 0.02587576, 0.02808221, -0.0110943, -0.01058207, -0.00909742],
        [-0.00110323, 0.00110322, -0.00110323, 0.03807328, 0.03756105, 0.03607641],
        [0.00110322, -0.00110323, 0.00110322, 0.62336888, 0.62388114, 0.62536579],
        [-0.00110323, 0.00110322, -0.00110324, 0.63059674, 0.63008449, 0.62859984],
        [0.00110322, -0.00110323, 0.00110323, 0.02709003, 0.02760231, 0.02908696],
        [0.01850419, 0.0207107, 0.01850424, -0.00748256, -0.00799484, -0.00947948],
        [-0.00628688, -0.00849338, -0.00628693, 0.01969987, 0.02021214, 0.02169679],
        [0.02416809, 0.0263746, 0.02416814, -0.00181866, -0.00233094, -0.00381558],
        [-0.38163306, 0.3310909, 0.18195818, -0.1318668, 0.13794232, -0.4430381],
    ]
)


# np.sin(3 * pi * x) * np.sin(2 * pi * y) ** np.sin(pi * z) (unconstrained)
# Using grid search
# No out-of-range prediction
sol11 = np.array(
    [
        [
            [0.38925069, 0.47707755, 0.55354554, 0.47707755, 0.38925069],
            [-0.14464081, -0.29610914, -0.50742697, -0.29610914, -0.14464081],
            [-0.64522275, -1.32407838, -2.15137408, -1.32407838, -0.64522275],
            [-0.84158214, -1.95384513, -3.28062299, -1.95384513, -0.84158214],
            [-0.39181305, -0.95061127, -1.62597974, -0.95061127, -0.39181305],
            [0.39181305, 0.95061127, 1.62597974, 0.95061127, 0.39181305],
            [0.84158214, 1.95384513, 3.28062299, 1.95384513, 0.84158214],
            [0.64522275, 1.32407838, 2.15137408, 1.32407838, 0.64522275],
            [0.14464081, 0.29610914, 0.50742697, 0.29610914, 0.14464081],
            [-0.38925069, -0.47707755, -0.55354554, -0.47707755, -0.38925069],
        ],
        [
            [0.03270436, -0.20351954, -0.39630309, -0.20351954, 0.03270436],
            [0.04188003, -0.07267593, -0.34508471, -0.07267593, 0.04188003],
            [-0.11023917, 0.34836578, 0.38298832, 0.34836578, -0.11023917],
            [-0.37037796, 0.42633328, 0.56576006, 0.42633328, -0.37037796],
            [-0.20156171, 0.1948345, 0.20331437, 0.1948345, -0.20156171],
            [0.20156171, -0.1948345, -0.20331437, -0.1948345, 0.20156171],
            [0.37037796, -0.42633328, -0.56576006, -0.42633328, 0.37037796],
            [0.11023917, -0.34836578, -0.38298832, -0.34836578, 0.11023917],
            [-0.04188003, 0.07267593, 0.34508471, 0.07267593, -0.04188003],
            [-0.03270436, 0.20351954, 0.39630309, 0.20351954, -0.03270436],
        ],
        [
            [-0.09503939, -0.29296608, -0.47444872, -0.29296608, -0.09503939],
            [-0.0998388, -0.2955443, -0.53919599, -0.2955443, -0.0998388],
            [-0.25695137, 0.32301617, 0.89369978, 0.32301617, -0.25695137],
            [-0.42449481, 0.92987358, 2.37820252, 0.92987358, -0.42449481],
            [-0.22247976, 0.45153893, 1.17848755, 0.45153893, -0.22247976],
            [0.22247976, -0.45153893, -1.17848755, -0.45153893, 0.22247976],
            [0.42449481, -0.92987358, -2.37820252, -0.92987358, 0.42449481],
            [0.25695137, -0.32301617, -0.89369978, -0.32301617, 0.25695137],
            [0.0998388, 0.2955443, 0.53919599, 0.2955443, 0.0998388],
            [0.09503939, 0.29296608, 0.47444872, 0.29296608, 0.09503939],
        ],
        [
            [-0.03215197, 0.25120674, 0.51134329, 0.25120674, -0.03215197],
            [0.1502849, 0.45958753, 0.90612018, 0.45958753, 0.1502849],
            [0.44191707, -0.57386978, -1.37150169, -0.57386978, 0.44191707],
            [0.74784696, -1.46536376, -3.5909169, -1.46536376, 0.74784696],
            [0.39782222, -0.6939776, -1.73860332, -0.6939776, 0.39782222],
            [-0.39782222, 0.6939776, 1.73860332, 0.6939776, -0.39782222],
            [-0.74784696, 1.46536376, 3.5909169, 1.46536376, -0.74784696],
            [-0.44191707, 0.57386978, 1.37150169, 0.57386978, -0.44191707],
            [-0.1502849, -0.45958753, -0.90612018, -0.45958753, -0.1502849],
            [0.03215197, -0.25120674, -0.51134329, -0.25120674, 0.03215197],
        ],
        [
            [-0.09503939, -0.29296608, -0.47444872, -0.29296608, -0.09503939],
            [-0.0998388, -0.2955443, -0.53919599, -0.2955443, -0.0998388],
            [-0.25695137, 0.32301617, 0.89369978, 0.32301617, -0.25695137],
            [-0.42449481, 0.92987358, 2.37820252, 0.92987358, -0.42449481],
            [-0.22247976, 0.45153893, 1.17848755, 0.45153893, -0.22247976],
            [0.22247976, -0.45153893, -1.17848755, -0.45153893, 0.22247976],
            [0.42449481, -0.92987358, -2.37820252, -0.92987358, 0.42449481],
            [0.25695137, -0.32301617, -0.89369978, -0.32301617, 0.25695137],
            [0.0998388, 0.2955443, 0.53919599, 0.2955443, 0.0998388],
            [0.09503939, 0.29296608, 0.47444872, 0.29296608, 0.09503939],
        ],
        [
            [0.03270436, -0.20351954, -0.39630309, -0.20351954, 0.03270436],
            [0.04188003, -0.07267593, -0.34508471, -0.07267593, 0.04188003],
            [-0.11023917, 0.34836578, 0.38298832, 0.34836578, -0.11023917],
            [-0.37037796, 0.42633328, 0.56576006, 0.42633328, -0.37037796],
            [-0.20156171, 0.1948345, 0.20331437, 0.1948345, -0.20156171],
            [0.20156171, -0.1948345, -0.20331437, -0.1948345, 0.20156171],
            [0.37037796, -0.42633328, -0.56576006, -0.42633328, 0.37037796],
            [0.11023917, -0.34836578, -0.38298832, -0.34836578, 0.11023917],
            [-0.04188003, 0.07267593, 0.34508471, 0.07267593, -0.04188003],
            [-0.03270436, 0.20351954, 0.39630309, 0.20351954, -0.03270436],
        ],
        [
            [0.38925069, 0.47707755, 0.55354554, 0.47707755, 0.38925069],
            [-0.14464081, -0.29610914, -0.50742697, -0.29610914, -0.14464081],
            [-0.64522275, -1.32407838, -2.15137408, -1.32407838, -0.64522275],
            [-0.84158214, -1.95384513, -3.28062299, -1.95384513, -0.84158214],
            [-0.39181305, -0.95061127, -1.62597974, -0.95061127, -0.39181305],
            [0.39181305, 0.95061127, 1.62597974, 0.95061127, 0.39181305],
            [0.84158214, 1.95384513, 3.28062299, 1.95384513, 0.84158214],
            [0.64522275, 1.32407838, 2.15137408, 1.32407838, 0.64522275],
            [0.14464081, 0.29610914, 0.50742697, 0.29610914, 0.14464081],
            [-0.38925069, -0.47707755, -0.55354554, -0.47707755, -0.38925069],
        ],
    ]
)

# np.sin(3 * pi * x) * np.sin(2 * pi * y) ** np.sin(pi * z) (non-negativity)
# Using grid search
# No out-of-range prediction
sol12 = np.array(
    [
        [
            [0.82456643, 1.19603669, 1.46902689, 1.18705258, 0.8011448],
            [0.02293622, -0.22325963, -0.5328345, -0.28410583, -0.10300906],
            [-0.50312036, -1.99300355, -3.50832897, -2.0862241, -0.66779239],
            [-0.7470301, -2.99493343, -5.17172379, -3.06530592, -0.83990058],
            [-0.31975682, -1.54378178, -2.87406937, -1.60830226, -0.31427196],
            [0.06906908, -0.12635022, -0.66721734, -0.08263473, 0.56122301],
            [0.08119788, -0.39555244, -1.21340365, -0.27582609, 0.89007258],
            [0.06913923, -0.19604079, -0.68124987, -0.09370757, 0.73698996],
            [0.29854099, 0.41647502, 0.43062276, 0.44094787, 0.56897793],
            [0.63505528, 0.804317, 0.94023579, 0.78773379, 0.60047011],
        ],
        [
            [0.18849409, 0.209783, 0.22042101, 0.23093084, 0.21617103],
            [0.00789017, 0.06030113, 0.17174233, 0.08796634, 0.01261895],
            [0.20229326, 0.46191311, 0.7728985, 0.46755417, 0.2037238],
            [0.0553124, 0.86164744, 1.78873245, 0.90793021, 0.16878089],
            [0.13321483, 0.36879382, 0.67977809, 0.36765565, 0.06237822],
            [-0.05277337, 0.07043619, 0.2599249, 0.0678618, -0.10065405],
            [-0.01733423, 0.12517577, 0.36623295, 0.079853, -0.28958709],
            [-0.05873073, 0.09280603, 0.26471171, 0.07869331, -0.16094769],
            [-0.05938954, -0.14979599, -0.21572505, -0.18208646, -0.09440694],
            [0.22280451, 0.28881371, 0.36243439, 0.29763353, 0.1971701],
        ],
        [
            [0.08516399, -0.06364211, -0.21230057, -0.06347808, 0.0854789],
            [0.06034927, -0.13088378, -0.34686893, -0.11972279, 0.0869071],
            [-0.1511267, -0.01282503, 0.12849543, -0.01633754, -0.21039188],
            [-0.37211988, 0.4503684, 1.29085619, 0.43158222, -0.50591939],
            [-0.08366277, -0.06379212, -0.07002147, -0.05329951, -0.07500775],
            [0.08196036, -0.09032714, -0.3018602, -0.09187974, 0.07800625],
            [0.01187754, -0.13567771, -0.31886415, -0.10126585, -0.00704303],
            [0.07560367, -0.06072391, -0.31722931, -0.05051884, 0.06098197],
            [-0.01619113, 0.02941192, 0.12215387, 0.03668516, -0.18320566],
            [0.09799334, 0.22780104, 0.37662732, 0.21500266, -0.04577748],
        ],
        [
            [0.09085642, 0.02744574, -0.01417808, 0.00079426, -0.01389667],
            [0.05801948, 0.1072217, 0.22204566, 0.09655839, -0.0628738],
            [0.08124847, -0.00408649, -0.08720299, 0.00044836, 0.11317741],
            [0.17648025, -0.2152616, -0.62370241, -0.20309712, 0.21081191],
            [0.05362174, 0.01925073, 0.00220213, 0.02416041, 0.04981966],
            [-0.06382576, 0.13089653, 0.33605053, 0.1095422, -0.08951485],
            [-0.47790382, 0.90522612, 2.22441887, 0.85555391, -0.14744808],
            [-0.24556347, 0.2168269, 0.5287319, 0.19392493, -0.10996303],
            [0.00528083, -0.05285207, -0.14185623, -0.05881784, -0.2060113],
            [0.11019291, 0.03473527, -0.02788398, -0.01529883, -0.10952584],
        ],
        [
            [0.11428395, -0.04971131, -0.1885393, -0.04868754, 0.1207648],
            [0.07754979, -0.13845397, -0.3426365, -0.12785028, 0.08701981],
            [-0.15986929, -0.01223954, 0.12650102, -0.01753819, -0.18148252],
            [-0.37224486, 0.458445, 1.28855964, 0.41844946, -0.41136795],
            [-0.07045125, -0.08462573, -0.08666099, -0.09781625, -0.04681517],
            [0.0896434, -0.09117247, -0.2657989, -0.09025481, 0.14282445],
            [-0.00157151, -0.12580758, -0.34636307, -0.12429952, 0.10297684],
            [0.09165392, -0.08601266, -0.29542704, -0.0641858, 0.09968695],
            [-0.04065783, 0.06483085, 0.19994931, 0.02838897, -0.09713394],
            [0.09535473, 0.26153094, 0.41094228, 0.24455896, 0.047886],
        ],
        [
            [0.2162169, 0.23679379, 0.26877498, 0.29292859, 0.40335355],
            [-0.01369639, 0.05436341, 0.18824606, 0.11678417, 0.10017041],
            [0.17093687, 0.46181774, 0.77732181, 0.5257314, -0.09867071],
            [0.0849722, 0.87301915, 1.73371582, 1.02202624, -0.18067714],
            [0.0439983, 0.39690197, 0.76151485, 0.49214454, -0.0279041],
            [-0.00244589, 0.08762801, 0.11201294, 0.09857797, -0.37249741],
            [-0.15031171, 0.18307363, 0.40823938, 0.17253294, -0.7496304],
            [0.00221264, 0.13532824, 0.18189469, 0.11339816, -0.32220926],
            [-0.04801114, -0.2780198, -0.46637989, -0.17672334, 0.06853776],
            [0.0933802, 0.1514442, 0.22381453, 0.25421477, 0.30020631],
        ],
        [
            [0.82230521, 1.21199829, 1.53291051, 1.31012417, 1.03095485],
            [-0.02386809, -0.25622263, -0.53531305, -0.40834727, -0.3054755],
            [-0.54242705, -2.01521128, -3.53357636, -2.41303804, -1.39208648],
            [-0.73002502, -2.95131242, -5.16984798, -3.39863785, -1.74960181],
            [-0.32711541, -1.52217734, -2.83267052, -1.87315017, -1.13009262],
            [0.13404603, -0.23115494, -0.73499817, -0.30573519, -0.37492675],
            [0.12520012, -0.38023803, -0.97832448, -0.30467578, -0.09233001],
            [0.08366657, -0.29663138, -0.80789861, -0.30303737, 0.05838411],
            [0.08041618, -0.04873173, -0.20574371, 0.02369934, 0.23400918],
            [0.15596433, 0.22293771, 0.31625026, 0.3685182, 0.41868983],
        ],
    ]
)

# np.sin(x) (unconstrained)
# Using optimizer
# Forward and backwards prediction
sol13 = np.array(
    [
        -0.74726464,
        -0.74726464,
        -0.74726464,
        -0.74726464,
        -0.74726464,
        -0.74726464,
        -0.74726464,
        -0.39705951,
        -0.00013062,
        0.39779593,
        0.73490884,
        0.96026329,
        1.03934091,
        0.96026329,
        0.73490884,
        0.39779593,
        -0.00013062,
        -0.39705951,
        -0.74726464,
        -0.74726464,
        -0.74726464,
        -0.74726464,
        -0.74726464,
        -0.74726464,
    ]
)

# np.sin(x) (non-negativity)
# Using optimizer
# Forward prediction
sol14 = np.array(
    [
        -0.26889314,
        -0.43197358,
        0.00893099,
        0.39312619,
        0.73871564,
        0.95606554,
        1.04491353,
        0.9518388,
        0.74945849,
        0.36693268,
        0.09276984,
        -0.90773984,
        8.72846841,
        7.12014439,
        6.82050448,
        6.72589677,
    ]
)

# np.arctan(x)*np.arctan(y) (Non-decreasing along y-direction)
# Using grid search
# Forward prediction along x-direction and backwards prediction along
# y-direction
sol15 = np.array(
    [
        [-0.01500464, -0.01189536, -0.01122042, -0.02328383, -0.02845942, 0.01222135],
        [-0.01811855, -0.01257072, 0.01559344, 0.04464826, 0.06245488, 0.09902459],
        [-0.02678618, -0.02500935, 0.04483974, 0.14713605, 0.2289398, 0.26637],
        [-0.03723566, -0.0320246, 0.06878376, 0.22052246, 0.34214508, 0.40664596],
        [-0.02510833, -0.03031934, 0.07789991, 0.28900298, 0.45881136, 0.51424841],
        [0.02001968, 0.03299117, 0.12664449, 0.3087991, 0.45968607, 0.50011128],
        [0.05217587, 0.08433176, 0.16782851, 0.29250935, 0.40090043, 0.45050585],
    ]
)

# 2 * (2 * x - 1) ** 3 (unconstrained)
# Using optimizer
# No out-of-range prediction
# Enforce the second derivative value at x = 0.8 is 700 with tolerance 1e-8
sol16 = np.array(
    [
        -3.09653031,
        -1.99192066,
        -0.90492261,
        -0.45651211,
        0.02461047,
        -0.21738733,
        0.40066724,
        -0.73585269,
        1.38526053,
        -1.37360499,
        2.86752948,
        -0.75904986,
        13.43686682,
    ]
)

# np.sin(3 * pi * x) * np.sin(2 * pi * y)
# Using grid search
# No out-of-range prediction
# Enforce the value at (x,y) = (4,3) is 4 with tolerance 1e-8
sol17 = np.array(
    [
        [-2.38182916, 2.39970157, -2.49938286, -0.96619358, 1.04954661, 0.19003811],
        [0.01612231, 0.87760275, -0.46720752, -1.24180009, -0.26714339, 0.68551821],
        [2.32620017, -1.08846068, 2.09609414, -0.58610299, -1.30374277, 1.16618394],
        [3.59396809, -3.46041227, 6.5086408, 3.78973664, -0.89258962, 1.40486524],
        [3.05269953, -2.53999397, 2.11784542, 1.20669247, -0.386364, 1.15311934],
        [1.11294575, 0.20577495, 0.22849025, -1.17002123, -0.6987175, 0.64769801],
        [-1.30450399, 1.12302325, 0.25770622, -0.62351554, -0.32455208, 0.22031567],
        [-3.7141894, 1.48500847, 0.19189211, 0.34426358, 0.29490921, -0.17344455],
    ]
)

# Gaussian pdf with mean 0 and standard deviation 2 (probability density function)
# Using optimizer
# No prediction
sol18 = np.array(
    [
        -0.24948697,
        0.07550771,
        -0.05254383,
        0.14799594,
        0.14799594,
        -0.05254383,
        0.07550771,
        -0.24948697,
    ]
)

# Multivariate gaussian pdf with mean (0,0) and covariate matrix [[2, 0.5],
# [0.5, 1]] (probability density function)
# Using grid search
# No prediction
sol19 = np.array(
    [
        [-0.01427045, 0.02801064, -0.01554955, 0.00433423, 0.00859512, 0.0079357],
        [0.00116722, -0.00702815, 0.00701625, -0.00139193, -0.00194026, -0.00335714],
        [0.03499568, -0.01297275, 0.05606468, 0.0042945, -0.00098802, 0.00567942],
        [0.06971778, -0.02908149, 0.10883901, 0.06729358, -0.01955591, 0.05103627],
        [0.05087725, -0.01948732, 0.06739461, 0.10901138, -0.02908846, 0.06958817],
        [0.0058145, -0.00121249, 0.00422258, 0.05575263, -0.01301563, 0.03582697],
        [-0.00170639, -0.00146472, -0.00137067, 0.00789573, -0.00704689, -0.00108668],
        [0.00489731, 0.00383655, 0.00647637, -0.01385008, 0.00546359, -0.02966302],
    ]
)


@pytest.mark.parametrize(
    "deg, ord_d, n_int, x, y, x_range, sp_method, sp_args, int_constraints, pt_constraints, pdf_constraint, sol",
    [
        (
            (3,),
            (2,),
            (11,),
            (np.linspace(0, 2 * np.pi, 101),),
            np.cos(np.linspace(0, 2 * np.pi, 101)),
            None,
            "grid_search",
            {"grid": ((0.1,),), "verbose": False, "parallel": False},
            {},
            {},
            False,
            sol1,
        ),
        (
            (3,),
            (2,),
            (11,),
            (np.linspace(0, 2 * np.pi, 101),),
            np.cos(np.linspace(0, 2 * np.pi, 101)),
            None,
            "grid_search",
            {"grid": ((0.1,),), "verbose": False, "parallel": False},
            {0: {0: {"+": 0.0}}},
            {},
            False,
            sol2,
        ),
        (
            (5,),
            (3,),
            (15,),
            (np.linspace(0, 200, 201),),
            np.exp(4 - np.linspace(0, 200, 201) / 25)
            + 4 * np.cos(np.linspace(0, 200, 201) / 8),
            None,
            "grid_search",
            {"grid": ((0.73,),), "verbose": False, "parallel": False},
            {},
            {},
            False,
            sol3,
        ),
        (
            (5,),
            (3,),
            (15,),
            (np.linspace(0, 200, 201),),
            np.exp(4 - np.linspace(0, 200, 201) / 25)
            + 4 * np.cos(np.linspace(0, 200, 201) / 8),
            None,
            "grid_search",
            {"grid": ((0.73,),), "verbose": False, "parallel": False},
            {0: {0: {"+": 10.0, "-": 40.0}}},
            {},
            False,
            sol4,
        ),
        (
            (2,),
            (1,),
            (10,),
            (np.linspace(0, 1, 50),),
            2 * (2 * np.linspace(0, 1, 50) - 1) ** 3,
            None,
            "optimizer",
            {
                "verbose": False,
                "x0": np.ones(1),
                "method": "SLSQP",
                "options": {"ftol": 1e-12, "maxiter": 100},
                "bounds": ((1e-10, 1e16),),
            },
            {},
            {},
            False,
            sol5,
        ),
        (
            (2,),
            (1,),
            (10,),
            (np.linspace(0, 1, 50),),
            2 * (2 * np.linspace(0, 1, 50) - 1) ** 3,
            None,
            "optimizer",
            {
                "verbose": False,
                "x0": np.ones(1),
                "method": "SLSQP",
                "options": {"ftol": 1e-12, "maxiter": 100},
                "bounds": ((1e-10, 1e16),),
            },
            {0: {1: {"+": 0.0}, 2: {"-": 0.0}}},
            {},
            False,
            sol6,
        ),
        (
            (3, 2),
            (2, 1),
            (5, 4),
            (np.linspace(0, 1, 30), np.linspace(0, 1, 20)),
            np.outer(
                np.sin(3 * np.pi * np.linspace(0, 1, 30)),
                np.sin(2 * np.pi * np.linspace(0, 1, 20)),
            ),
            None,
            "grid_search",
            {
                "grid": ((0.89, 5.96), (3.45, 0.012)),
                "verbose": False,
                "parallel": False,
            },
            {},
            {},
            False,
            sol7,
        ),
        (
            (3, 2),
            (2, 1),
            (5, 4),
            (np.linspace(0, 1, 30), np.linspace(0, 1, 20)),
            np.outer(
                np.sin(3 * np.pi * np.linspace(0, 1, 30)),
                np.sin(2 * np.pi * np.linspace(0, 1, 20)),
            ),
            None,
            "grid_search",
            {
                "grid": ((0.89, 5.96), (3.45, 0.012)),
                "verbose": False,
                "parallel": False,
            },
            {0: {0: {"+": 0}}, 1: {0: {"+": 0}}},
            {},
            False,
            sol8,
        ),
        (
            (2, 2),
            (1, 1),
            (10, 4),
            (np.linspace(0, 1, 30), np.linspace(0, 1, 20)),
            np.outer(
                np.sin(3 * np.pi * np.linspace(0, 1, 30)),
                np.sin(2 * np.pi * np.linspace(0, 1, 20)),
            ),
            None,
            "optimizer",
            {
                "verbose": False,
                "x0": np.ones(2),
                "method": "SLSQP",
                "options": {"ftol": 1e-12, "maxiter": 100},
                "bounds": ((1e-10, 1e16), (1e-10, 1e16)),
            },
            {},
            {},
            False,
            sol9,
        ),
        (
            (2, 2),
            (1, 1),
            (10, 4),
            (np.linspace(0, 1, 30), np.linspace(0, 1, 20)),
            np.outer(
                np.sin(3 * np.pi * np.linspace(0, 1, 30)),
                np.sin(2 * np.pi * np.linspace(0, 1, 20)),
            ),
            None,
            "optimizer",
            {
                "verbose": False,
                "x0": np.ones(2),
                "method": "SLSQP",
                "options": {"ftol": 1e-12, "maxiter": 100},
                "bounds": ((1e-10, 1e16), (1e-10, 1e16)),
            },
            {0: {0: {"+": 0}}, 1: {0: {"+": 0}, 1: {"+": 0}}},
            {},
            False,
            sol10,
        ),
        (
            (3, 4, 2),
            (2, 2, 1),
            (4, 6, 3),
            (np.linspace(0, 1, 30), np.linspace(0, 1, 40), np.linspace(0, 1, 50)),
            np.einsum(
                "i,j,k->ijk",
                np.sin(3 * np.pi * np.linspace(0, 1, 30)),
                np.sin(2 * np.pi * np.linspace(0, 1, 40)),
                np.sin(np.pi * np.linspace(0, 1, 50)),
            ),
            None,
            "grid_search",
            {"verbose": False, "parallel": False, "grid": ((0.1,), (0.2,), (0.3,))},
            {},
            {},
            False,
            sol11,
        ),
        (
            (3, 4, 2),
            (2, 2, 1),
            (4, 6, 3),
            (np.linspace(0, 1, 30), np.linspace(0, 1, 40), np.linspace(0, 1, 50)),
            np.einsum(
                "i,j,k->ijk",
                np.sin(3 * np.pi * np.linspace(0, 1, 30)),
                np.sin(2 * np.pi * np.linspace(0, 1, 40)),
                np.sin(np.pi * np.linspace(0, 1, 50)),
            ),
            None,
            "grid_search",
            {"verbose": False, "parallel": False, "grid": ((0.1,), (0.2,), (0.3,))},
            {0: {0: {"+": 0}}, 1: {0: {"+": 0}}, 2: {0: {"+": 0}}},
            {},
            False,
            sol12,
        ),
        (
            (5,),
            (1,),
            (8,),
            (np.linspace(0, np.pi, 51),),
            np.sin(np.linspace(0, np.pi, 51)),
            {0: (-2, 5)},
            "optimizer",
            {
                "verbose": False,
                "x0": np.ones(1),
                "method": "L-BFGS-B",
                "options": {"ftol": 1e-12, "maxiter": 100},
                "bounds": ((1e-10, 1e16),),
            },
            {},
            {},
            False,
            sol13,
        ),
        (
            (5,),
            (1,),
            (8,),
            (np.linspace(0, np.pi, 51),),
            np.sin(np.linspace(0, np.pi, 51)),
            {0: (4,)},
            "optimizer",
            {
                "verbose": False,
                "x0": np.ones(1),
                "method": "L-BFGS-B",
                "options": {"ftol": 1e-12, "maxiter": 100},
                "bounds": ((1e-10, 1e16),),
            },
            {0: {0: {"+": 0}}},
            {},
            False,
            sol14,
        ),
        (
            (2, 2),
            (1, 1),
            (4, 3),
            (np.linspace(0, np.pi / 3, 51), np.linspace(0, np.pi / 4, 41)),
            np.outer(
                np.arctan(np.linspace(0, np.pi / 3, 51)),
                np.arctan(np.linspace(0, np.pi / 4, 41)),
            ),
            {0: (1.1,), 1: (-0.1,)},
            "grid_search",
            {"verbose": False, "parallel": False, "grid": ((2,), (2,))},
            {1: {1: {"+": 0}}},
            {},
            False,
            sol15,
        ),
        (
            (3,),
            (2,),
            (10,),
            (np.linspace(0, 1, 50),),
            2 * (2 * np.linspace(0, 1, 50) - 1) ** 3,
            None,
            "optimizer",
            {
                "verbose": False,
                "x0": np.ones(1),
                "method": "SLSQP",
                "options": {"ftol": 1e-12, "maxiter": 100},
                "bounds": ((1e-10, 1e16),),
            },
            {},
            {(2,): ((np.array([0.8]),), np.array([700]), 1e-8)},
            False,
            sol16,
        ),
        (
            (3, 2),
            (2, 1),
            (5, 4),
            (np.linspace(0, 3 * np.pi, 30), np.linspace(0, 2 * np.pi, 20)),
            np.outer(
                np.sin(np.linspace(0, 3 * np.pi, 30)),
                np.sin(np.linspace(0, 2 * np.pi, 20)),
            ),
            None,
            "grid_search",
            {
                "grid": ((0.89, 5.96), (3.45, 0.012)),
                "verbose": False,
                "parallel": False,
            },
            {},
            {(0, 0): ((np.array([4]), np.array([3])), np.array([4]), 1e-8)},
            False,
            sol17,
        ),
        (
            (3,),
            (2,),
            (11,),
            (np.linspace(0, 2 * np.pi, 101)[::-1],),  # Unordered data in 1-D
            np.cos(np.linspace(0, 2 * np.pi, 101))[::-1],
            None,
            "grid_search",
            {"grid": ((0.1,),), "verbose": False, "parallel": False},
            {},
            {},
            False,
            sol1,
        ),
        (
            (3, 2),
            (2, 1),
            (5, 4),
            (
                np.linspace(0, 1, 30)[::-1],
                np.linspace(0, 1, 20)[::-1],
            ),  # Unordered data in 2-D
            np.outer(
                np.sin(3 * np.pi * np.linspace(0, 1, 30)[::-1]),
                np.sin(2 * np.pi * np.linspace(0, 1, 20)[::-1]),
            ),
            None,
            "grid_search",
            {
                "grid": ((0.89, 5.96), (3.45, 0.012)),
                "verbose": False,
                "parallel": False,
            },
            {},
            {},
            False,
            sol7,
        ),
        (
            (3,),
            (2,),
            (5,),
            (np.linspace(-10, 10, 51),),
            norm.pdf(np.linspace(-10, 10, 51), 0, 2),
            None,
            "optimizer",
            {
                "verbose": False,
                "x0": np.ones(1),
                "method": "SLSQP",
                "options": {"ftol": 1e-12, "maxiter": 100},
                "bounds": ((1e-10, 1e16),),
            },
            {},  # Do not include non-negative constraint explicitly
            {},
            True,
            sol18,
        ),
        (
            (3, 2),
            (2, 1),
            (5, 4),
            (np.linspace(-3, 3, 50), np.linspace(-4, 4, 60)),
            multivariate_normal.pdf(
                x=list(
                    itertools.product(np.linspace(-3, 3, 50), np.linspace(-4, 4, 60))
                ),
                mean=[0, 0],
                cov=[[2, 0.5], [0.5, 1]],
            ).reshape((len(np.linspace(-3, 3, 50)), len(np.linspace(-4, 4, 60)))),
            None,
            "grid_search",
            {
                "grid": ((0.1,), (0.01,)),
                "verbose": False,
                "parallel": False,
            },
            {0: {0: {"+": 0}}, 1: {0: {"+": 0}}},
            {},
            True,
            sol19,
        ),
    ],
)


# Test the decision variable with the expansion coefficients of the B-spline
# basis
def test_sol(
    deg,
    ord_d,
    n_int,
    x,
    y,
    x_range,
    sp_method,
    sp_args,
    int_constraints,
    pt_constraints,
    pdf_constraint,
    sol,
):
    out = GridCPsplines(
        deg=deg,
        ord_d=ord_d,
        n_int=n_int,
        sp_method=sp_method,
        sp_args=sp_args,
        x_range=x_range,
        int_constraints=int_constraints,
        pt_constraints=pt_constraints,
        pdf_constraint=pdf_constraint,
    )
    out.fit(x=x, y=y)
    np.testing.assert_allclose(out.sol, sol, rtol=0.2, atol=1e-2)
